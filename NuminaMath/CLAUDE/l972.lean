import Mathlib

namespace NUMINAMATH_CALUDE_simplify_fraction_expression_l972_97200

theorem simplify_fraction_expression (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m / n - n / m) / (1 / m - 1 / n) = -m - n := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_expression_l972_97200


namespace NUMINAMATH_CALUDE_min_value_fraction_min_value_achievable_l972_97204

theorem min_value_fraction (x : ℝ) (h : x > 0) :
  (x^2 + x + 3) / (x + 1) ≥ 2 * Real.sqrt 3 - 1 :=
by sorry

theorem min_value_achievable :
  ∃ x > 0, (x^2 + x + 3) / (x + 1) = 2 * Real.sqrt 3 - 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_min_value_achievable_l972_97204


namespace NUMINAMATH_CALUDE_fraction_comparison_and_differences_l972_97282

theorem fraction_comparison_and_differences :
  (1 / 3 : ℚ) < (1 / 2 : ℚ) ∧ (1 / 2 : ℚ) < (3 / 5 : ℚ) ∧
  (1 / 2 : ℚ) - (1 / 3 : ℚ) = (1 / 6 : ℚ) ∧
  (3 / 5 : ℚ) - (1 / 2 : ℚ) = (1 / 10 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_and_differences_l972_97282


namespace NUMINAMATH_CALUDE_hyperbola_sum_l972_97239

theorem hyperbola_sum (h k a b c : ℝ) : 
  (h, k) = (3, -1) →
  (3 + Real.sqrt 53, -1) = (h + c, k) →
  (7, -1) = (h + a, k) →
  b^2 = c^2 - a^2 →
  h + k + a + b = 6 + Real.sqrt 37 := by sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l972_97239


namespace NUMINAMATH_CALUDE_power_sum_equality_l972_97258

theorem power_sum_equality : -2^2005 + (-2)^2006 + 2^2007 - 2^2008 = 2^2005 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l972_97258


namespace NUMINAMATH_CALUDE_fifteen_people_on_boats_l972_97283

/-- Given a lake with boats and people, calculate the total number of people on the boats. -/
def total_people_on_boats (num_boats : ℕ) (people_per_boat : ℕ) : ℕ :=
  num_boats * people_per_boat

/-- Theorem: In a lake with 5 boats and 3 people per boat, there are 15 people on boats in total. -/
theorem fifteen_people_on_boats :
  total_people_on_boats 5 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_people_on_boats_l972_97283


namespace NUMINAMATH_CALUDE_pet_store_birds_l972_97217

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 6

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 6

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds : total_birds = 48 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_birds_l972_97217


namespace NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l972_97251

theorem min_value_fraction (x : ℝ) (h : x > 6) : x^2 / (x - 6) ≥ 18 := by
  sorry

theorem equality_condition (x : ℝ) (h : x > 6) : x^2 / (x - 6) = 18 ↔ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l972_97251


namespace NUMINAMATH_CALUDE_equation_one_solutions_l972_97261

theorem equation_one_solutions : 
  ∀ x : ℝ, (x - 3)^2 = 4 ↔ x = 5 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l972_97261


namespace NUMINAMATH_CALUDE_triangle_properties_l972_97243

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The vectors (cos A, cos B) and (a, 2c - b) are parallel -/
def vectors_parallel (t : Triangle) : Prop :=
  (2 * t.c - t.b) * Real.cos t.A = t.a * Real.cos t.B

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) (h : vectors_parallel t) :
  t.A = π / 3 ∧ (t.a = 4 → ∃ (max_area : ℝ), max_area = 4 * Real.sqrt 3 ∧
    ∀ (area : ℝ), area = 1 / 2 * t.b * t.c * Real.sin t.A → area ≤ max_area) :=
sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l972_97243


namespace NUMINAMATH_CALUDE_parabola_properties_l972_97272

-- Define the parabola
def parabola (x : ℝ) : ℝ := (x - 3)^2 + 5

-- Theorem stating the properties of the parabola
theorem parabola_properties :
  (∀ x : ℝ, parabola x ≥ 5) ∧ 
  (∀ x : ℝ, parabola (3 + x) = parabola (3 - x)) ∧
  (parabola 3 = 5) := by
  sorry


end NUMINAMATH_CALUDE_parabola_properties_l972_97272


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l972_97260

noncomputable def P : (ℝ → ℝ → ℝ) → Prop :=
  λ p => ∀ x y : ℝ, p (x^2) (y^2) = p ((x + y)^2 / 2) ((x - y)^2 / 2)

theorem polynomial_functional_equation :
  ∀ p : ℝ → ℝ → ℝ, P p ↔ ∃ q : ℝ → ℝ → ℝ, ∀ x y : ℝ, p x y = q (x + y) (x * y * (x - y)^2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l972_97260


namespace NUMINAMATH_CALUDE_angle_b_measure_l972_97257

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  A : Real
  B : Real
  C : Real
  sum_angles : A + B + C = 180
  isosceles : B = C
  angle_relation : C = 3 * A

/-- Theorem: In the given isosceles triangle, angle B measures 540/7 degrees -/
theorem angle_b_measure (t : IsoscelesTriangle) : t.B = 540 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_measure_l972_97257


namespace NUMINAMATH_CALUDE_tan_150_and_pythagorean_identity_l972_97229

theorem tan_150_and_pythagorean_identity :
  (Real.tan (150 * π / 180) = -Real.sqrt 3 / 3) ∧
  (Real.sin (150 * π / 180))^2 + (Real.cos (150 * π / 180))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_and_pythagorean_identity_l972_97229


namespace NUMINAMATH_CALUDE_evaluate_expression_l972_97269

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l972_97269


namespace NUMINAMATH_CALUDE_passing_grade_fraction_l972_97278

theorem passing_grade_fraction (students_A students_B students_C students_D students_F : ℚ) :
  students_A = 1/4 →
  students_B = 1/2 →
  students_C = 1/8 →
  students_D = 1/12 →
  students_F = 1/24 →
  students_A + students_B + students_C = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_passing_grade_fraction_l972_97278


namespace NUMINAMATH_CALUDE_rotated_line_x_intercept_l972_97228

/-- The x-coordinate of the x-intercept of a rotated line -/
theorem rotated_line_x_intercept 
  (m : Real → Real → Prop) -- Original line
  (θ : Real) -- Rotation angle
  (p : Real × Real) -- Point of rotation
  (n : Real → Real → Prop) -- Rotated line
  (h1 : ∀ x y, m x y ↔ 4 * x - 3 * y + 20 = 0) -- Equation of line m
  (h2 : θ = π / 3) -- 60° in radians
  (h3 : p = (10, 10)) -- Point of rotation
  (h4 : ∀ x y, n x y ↔ 
    y - p.2 = ((24 + 25 * Real.sqrt 3) / (-39)) * (x - p.1)) -- Equation of line n
  (C : Real) -- Constant C
  (h5 : C = 10 - (240 + 250 * Real.sqrt 3) / (-39)) -- Definition of C
  : ∃ x_intercept : Real, 
    x_intercept = -39 * C / (24 + 25 * Real.sqrt 3) ∧ 
    n x_intercept 0 := by sorry

end NUMINAMATH_CALUDE_rotated_line_x_intercept_l972_97228


namespace NUMINAMATH_CALUDE_train_speed_l972_97218

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 160) (h2 : time = 4) :
  length / time = 40 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l972_97218


namespace NUMINAMATH_CALUDE_contractor_problem_l972_97230

/-- Proves that the original number of men employed is 12 --/
theorem contractor_problem (initial_days : ℕ) (absent_men : ℕ) (actual_days : ℕ) 
  (h1 : initial_days = 5)
  (h2 : absent_men = 8)
  (h3 : actual_days = 15) :
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * actual_days ∧ 
    original_men = 12 := by
  sorry

end NUMINAMATH_CALUDE_contractor_problem_l972_97230


namespace NUMINAMATH_CALUDE_carls_dad_contribution_l972_97232

def weekly_savings : ℕ := 25
def weeks_saved : ℕ := 6
def coat_cost : ℕ := 170
def bill_fraction : ℚ := 1/3

theorem carls_dad_contribution :
  let total_savings := weekly_savings * weeks_saved
  let remaining_savings := total_savings - (bill_fraction * total_savings).floor
  coat_cost - remaining_savings = 70 := by
  sorry

end NUMINAMATH_CALUDE_carls_dad_contribution_l972_97232


namespace NUMINAMATH_CALUDE_line_symmetry_l972_97292

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2 * x - y + 3 = 0
def line2 (x y : ℝ) : Prop := x - y + 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (l1 l2 l3 : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), 
    l1 x1 y1 → l2 x2 y2 → 
    ∃ (xm ym : ℝ), l3 xm ym ∧ 
      xm = (x1 + x2) / 2 ∧ 
      ym = (y1 + y2) / 2

-- Theorem statement
theorem line_symmetry : symmetric_wrt line1 line3 line2 := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l972_97292


namespace NUMINAMATH_CALUDE_sqrt_five_minus_one_gt_one_l972_97252

theorem sqrt_five_minus_one_gt_one : Real.sqrt 5 - 1 > 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_minus_one_gt_one_l972_97252


namespace NUMINAMATH_CALUDE_factor_cubic_expression_l972_97250

theorem factor_cubic_expression (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) 
  = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_factor_cubic_expression_l972_97250


namespace NUMINAMATH_CALUDE_shoe_selection_theorem_l972_97215

/-- The number of ways to select 4 shoes from 10 pairs such that 2 form a pair and 2 do not -/
def shoeSelectionWays (totalPairs : Nat) : Nat :=
  if totalPairs = 10 then
    Nat.choose totalPairs 1 * Nat.choose (totalPairs - 1) 2 * 4
  else
    0

theorem shoe_selection_theorem :
  shoeSelectionWays 10 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_shoe_selection_theorem_l972_97215


namespace NUMINAMATH_CALUDE_f_properties_l972_97227

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 * x + 1) + 2 * a * x - 4 * a * Real.exp x + 4

theorem f_properties (a : ℝ) (h : a > 0) :
  (∃ x, f 1 x ≤ f 1 0) ∧
  ((0 < a ∧ a < 1 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ∧
   (a = 1 → ∃! x, f a x = 0) ∧
   (a > 1 → ∀ x, f a x ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l972_97227


namespace NUMINAMATH_CALUDE_circle_has_most_symmetry_lines_l972_97275

/-- Represents the number of lines of symmetry for a geometrical figure. -/
inductive SymmetryCount
  | Finite (n : ℕ)
  | Infinite

/-- Represents the geometrical figures mentioned in the problem. -/
inductive GeometricalFigure
  | Circle
  | Semicircle
  | EquilateralTriangle
  | RegularPentagon
  | Ellipse

/-- Returns the number of lines of symmetry for a given geometrical figure. -/
def symmetryLines (figure : GeometricalFigure) : SymmetryCount :=
  match figure with
  | GeometricalFigure.Circle => SymmetryCount.Infinite
  | GeometricalFigure.Semicircle => SymmetryCount.Finite 1
  | GeometricalFigure.EquilateralTriangle => SymmetryCount.Finite 3
  | GeometricalFigure.RegularPentagon => SymmetryCount.Finite 5
  | GeometricalFigure.Ellipse => SymmetryCount.Finite 2

/-- Compares two SymmetryCount values. -/
def symmetryCountLe (a b : SymmetryCount) : Prop :=
  match a, b with
  | SymmetryCount.Finite n, SymmetryCount.Finite m => n ≤ m
  | _, SymmetryCount.Infinite => True
  | SymmetryCount.Infinite, SymmetryCount.Finite _ => False

/-- States that the circle has the greatest number of lines of symmetry among the given figures. -/
theorem circle_has_most_symmetry_lines :
    ∀ (figure : GeometricalFigure),
      symmetryCountLe (symmetryLines figure) (symmetryLines GeometricalFigure.Circle) :=
  sorry

end NUMINAMATH_CALUDE_circle_has_most_symmetry_lines_l972_97275


namespace NUMINAMATH_CALUDE_find_q_l972_97266

theorem find_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : 
  q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_find_q_l972_97266


namespace NUMINAMATH_CALUDE_park_cycling_time_l972_97216

/-- Proves that for a rectangular park with given specifications, 
    a cyclist completes one round in 8 minutes -/
theorem park_cycling_time (length width : ℝ) (area perimeter : ℝ) (speed : ℝ) :
  width = 4 * length →
  area = length * width →
  area = 102400 →
  perimeter = 2 * (length + width) →
  speed = 12 * 1000 / 60 →
  (perimeter / speed) = 8 :=
by sorry

end NUMINAMATH_CALUDE_park_cycling_time_l972_97216


namespace NUMINAMATH_CALUDE_tenth_configuration_stones_l972_97209

/-- The number of stones in the n-th configuration of Anya's pentagon pattern -/
def stones (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- We define n = 0 as having no stones for completeness
  | 1 => 1
  | n + 1 => stones n + 3 * (n + 1) - 2

/-- The theorem stating that the 10th configuration has 145 stones -/
theorem tenth_configuration_stones :
  stones 10 = 145 := by
  sorry

/-- Helper lemma to show the first four configurations match the given values -/
lemma first_four_configurations :
  stones 1 = 1 ∧ stones 2 = 5 ∧ stones 3 = 12 ∧ stones 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_tenth_configuration_stones_l972_97209


namespace NUMINAMATH_CALUDE_parallelepiped_diagonal_l972_97296

/-- The diagonal of a rectangular parallelepiped given its face diagonals -/
theorem parallelepiped_diagonal (m n p : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) :
  ∃ (d : ℝ), d > 0 ∧ d^2 = (m^2 + n^2 + p^2) / 2 := by
  sorry

#check parallelepiped_diagonal

end NUMINAMATH_CALUDE_parallelepiped_diagonal_l972_97296


namespace NUMINAMATH_CALUDE_smallest_m_equals_n_l972_97219

theorem smallest_m_equals_n (n : ℕ) (hn : n > 1) :
  ∃ (m : ℕ),
    (∀ (a b : ℕ) (ha : a ∈ Finset.range (2 * n)) (hb : b ∈ Finset.range (2 * n)) (hab : a ≠ b),
      ∃ (x y : ℕ) (hxy : x + y > 0),
        (2 * n ∣ a * x + b * y) ∧ (x + y ≤ m)) ∧
    (∀ (k : ℕ),
      (∀ (a b : ℕ) (ha : a ∈ Finset.range (2 * n)) (hb : b ∈ Finset.range (2 * n)) (hab : a ≠ b),
        ∃ (x y : ℕ) (hxy : x + y > 0),
          (2 * n ∣ a * x + b * y) ∧ (x + y ≤ k)) →
      k ≥ m) ∧
    m = n :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_equals_n_l972_97219


namespace NUMINAMATH_CALUDE_fraction_unchanged_l972_97262

theorem fraction_unchanged (x y : ℝ) : 
  (2 * x) / (3 * x - 2 * y) = (4 * x) / (6 * x - 4 * y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l972_97262


namespace NUMINAMATH_CALUDE_martin_position_l972_97277

/-- Represents a queue with the given properties -/
structure Queue where
  total : ℕ
  martin_pos : ℕ
  friend1_pos : ℕ
  friend2_pos : ℕ
  total_multiple_of_3 : total % 3 = 0
  martin_centered : martin_pos - 1 = total - martin_pos
  friend1_behind : friend1_pos > martin_pos
  friend2_behind : friend2_pos > martin_pos
  friend1_is_19th : friend1_pos = 19
  friend2_is_28th : friend2_pos = 28

/-- The theorem stating Martin's position in the queue -/
theorem martin_position (q : Queue) : q.martin_pos = 17 := by
  sorry

end NUMINAMATH_CALUDE_martin_position_l972_97277


namespace NUMINAMATH_CALUDE_jucas_marbles_l972_97210

theorem jucas_marbles :
  ∃! B : ℕ, 0 < B ∧ B < 800 ∧
  B % 3 = 2 ∧
  B % 4 = 3 ∧
  B % 5 = 4 ∧
  B % 7 = 6 ∧
  B % 20 = 19 ∧
  B = 419 :=
by sorry

end NUMINAMATH_CALUDE_jucas_marbles_l972_97210


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l972_97297

theorem expression_equals_negative_one
  (a b y : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ a)
  (hy1 : y ≠ a)
  (hy2 : y ≠ -a) :
  (((a + b) / (a + y) + y / (a - y)) /
   ((y + b) / (a + y) - a / (a - y)) = -1) ↔
  (y = a - b) :=
sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l972_97297


namespace NUMINAMATH_CALUDE_percentage_problem_l972_97242

theorem percentage_problem (x : ℝ) : 
  (16 / 100) * ((40 / 100) * x) = 6 → x = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l972_97242


namespace NUMINAMATH_CALUDE_five_long_sides_l972_97214

/-- A convex hexagon with specific properties -/
structure ConvexHexagon where
  -- The two distinct side lengths
  short_side : ℝ
  long_side : ℝ
  -- The number of sides with each length
  num_short_sides : ℕ
  num_long_sides : ℕ
  -- Properties
  is_convex : Bool
  distinct_lengths : short_side ≠ long_side
  total_sides : num_short_sides + num_long_sides = 6
  perimeter : num_short_sides * short_side + num_long_sides * long_side = 40
  short_side_length : short_side = 4
  long_side_length : long_side = 7

/-- Theorem: In a convex hexagon with the given properties, there are exactly 5 sides measuring 7 units -/
theorem five_long_sides (h : ConvexHexagon) : h.num_long_sides = 5 :=
  sorry

end NUMINAMATH_CALUDE_five_long_sides_l972_97214


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l972_97236

def vector_a (m : ℝ) : Fin 2 → ℝ := ![3, -2*m]
def vector_b (m : ℝ) : Fin 2 → ℝ := ![8, 3*m]

def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

theorem perpendicular_vectors_m_value :
  ∀ m : ℝ, dot_product (vector_a m) (vector_b m) = 0 → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l972_97236


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l972_97244

/-- Calculates the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth) + 2 * (width * depth)

/-- Theorem stating the total wet surface area of a specific cistern -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 8 6 1.85 = 99.8 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l972_97244


namespace NUMINAMATH_CALUDE_difference_between_point_eight_and_half_l972_97284

theorem difference_between_point_eight_and_half : 0.8 - (1/2 : ℚ) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_difference_between_point_eight_and_half_l972_97284


namespace NUMINAMATH_CALUDE_rain_difference_l972_97233

/-- The amount of rain Greg experienced while camping, in millimeters. -/
def camping_rain : List ℝ := [3, 6, 5]

/-- The amount of rain at Greg's house during the same week, in millimeters. -/
def house_rain : ℝ := 26

/-- The difference in rainfall between Greg's house and his camping trip. -/
theorem rain_difference : house_rain - (camping_rain.sum) = 12 := by
  sorry

end NUMINAMATH_CALUDE_rain_difference_l972_97233


namespace NUMINAMATH_CALUDE_gcd_power_difference_l972_97224

theorem gcd_power_difference (a b n : ℕ) (ha : a ≥ 2) (hb : b ≥ 2) (hn : n ≥ 2) (hgcd : Nat.gcd a b = 1) :
  Nat.gcd ((a^n - b^n) / (a - b)) (a - b) = Nat.gcd (a - b) n := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_difference_l972_97224


namespace NUMINAMATH_CALUDE_percentage_problem_l972_97286

theorem percentage_problem (x : ℝ) (h : 25 = 0.4 * x) : x = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l972_97286


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l972_97201

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We don't need to define all properties of an isosceles triangle,
  -- just the ones relevant to our problem
  vertex_angle : ℝ
  base_angle : ℝ
  is_valid : vertex_angle + 2 * base_angle = 180

-- Define our theorem
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.vertex_angle = 70 ∨ triangle.base_angle = 70) : 
  triangle.vertex_angle = 40 ∨ triangle.vertex_angle = 70 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l972_97201


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l972_97256

def grade10_students : ℕ := 300
def grade11_students : ℕ := 200
def grade12_students : ℕ := 400
def total_selected : ℕ := 18

def total_students : ℕ := grade10_students + grade11_students + grade12_students

def stratified_sample (grade_students : ℕ) : ℕ :=
  (total_selected * grade_students) / total_students

theorem stratified_sampling_result :
  (stratified_sample grade10_students,
   stratified_sample grade11_students,
   stratified_sample grade12_students) = (6, 4, 8) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l972_97256


namespace NUMINAMATH_CALUDE_circle_line_intersection_sum_l972_97220

/-- Given a circle with radius 4 centered at the origin and a line y = 4 - (2 - √3)x
    intersecting the circle at points A and B, the sum of the length of segment AB
    and the length of the shorter arc AB is 4√(2 - √3) + (2π/3) -/
theorem circle_line_intersection_sum (A B : ℝ × ℝ) : 
  let r : ℝ := 4
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | y = 4 - (2 - Real.sqrt 3) * x}
  A ∈ circle ∧ A ∈ line ∧ B ∈ circle ∧ B ∈ line ∧ A ≠ B →
  let segment_length := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let angle := Real.arccos ((2 * r^2 - segment_length^2) / (2 * r^2))
  let arc_length := angle * r
  segment_length + min arc_length (2 * π * r - arc_length) = 4 * Real.sqrt (2 - Real.sqrt 3) + (2 * π / 3) := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_sum_l972_97220


namespace NUMINAMATH_CALUDE_amy_game_score_l972_97246

theorem amy_game_score (points_per_treasure : ℕ) (treasures_level1 : ℕ) (treasures_level2 : ℕ)
  (h1 : points_per_treasure = 4)
  (h2 : treasures_level1 = 6)
  (h3 : treasures_level2 = 2) :
  points_per_treasure * treasures_level1 + points_per_treasure * treasures_level2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_amy_game_score_l972_97246


namespace NUMINAMATH_CALUDE_fraction_value_when_y_is_three_l972_97288

theorem fraction_value_when_y_is_three :
  let y : ℝ := 3
  (y^3 + y) / (y^2 - y) = 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_when_y_is_three_l972_97288


namespace NUMINAMATH_CALUDE_complex_root_modulus_sqrt5_l972_97247

theorem complex_root_modulus_sqrt5 (k : ℝ) :
  (∃ (z : ℂ), z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  k = -1 ∨ k = 3 := by
sorry

end NUMINAMATH_CALUDE_complex_root_modulus_sqrt5_l972_97247


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l972_97291

/-- Given a square carpet with the described shaded squares, prove the total shaded area is 45 square feet. -/
theorem shaded_area_calculation (S T : ℝ) : 
  S > 0 → T > 0 → (12 : ℝ) / S = 4 → S / T = 2 → 
  S^2 + 16 * T^2 = 45 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l972_97291


namespace NUMINAMATH_CALUDE_equation_solutions_l972_97276

theorem equation_solutions :
  (∀ x, x * (x - 5) = 3 * x - 15 ↔ x = 5 ∨ x = 3) ∧
  (∀ y, 2 * y^2 - 9 * y + 5 = 0 ↔ y = (9 + Real.sqrt 41) / 4 ∨ y = (9 - Real.sqrt 41) / 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l972_97276


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l972_97206

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 2 + a 3 + a 4 + a 5 + a 6 = 90) →
  (a 1 + a 7 = 36) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l972_97206


namespace NUMINAMATH_CALUDE_perpendicular_planes_parallel_l972_97267

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_parallel
  (α β : Plane) (m : Line)
  (h_diff_planes : α ≠ β)
  (h_perp_α : perpendicular m α)
  (h_perp_β : perpendicular m β) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_parallel_l972_97267


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l972_97271

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes 
    such that each box contains at least one ball -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 540 ways to distribute 6 distinguishable balls
    into 3 distinguishable boxes such that each box contains at least one ball -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 540 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l972_97271


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l972_97268

theorem difference_of_squares_special_case : (827 : ℤ) * 827 - 826 * 828 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l972_97268


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_8a_l972_97280

theorem factorization_a_squared_minus_8a (a : ℝ) : a^2 - 8*a = a*(a - 8) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_8a_l972_97280


namespace NUMINAMATH_CALUDE_x_minus_y_negative_l972_97299

theorem x_minus_y_negative (x y : ℝ) (h : x < y) : x - y < 0 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_negative_l972_97299


namespace NUMINAMATH_CALUDE_function_domain_iff_m_range_l972_97259

/-- The function f(x) = lg(x^2 - 2mx + m + 2) has domain R if and only if m ∈ (-1, 2) -/
theorem function_domain_iff_m_range (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2 - 2*m*x + m + 2)) ↔ m > -1 ∧ m < 2 := by
sorry


end NUMINAMATH_CALUDE_function_domain_iff_m_range_l972_97259


namespace NUMINAMATH_CALUDE_greatest_common_length_l972_97208

theorem greatest_common_length (a b c d : ℕ) 
  (ha : a = 72) (hb : b = 48) (hc : c = 120) (hd : d = 96) : 
  Nat.gcd a (Nat.gcd b (Nat.gcd c d)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_length_l972_97208


namespace NUMINAMATH_CALUDE_family_reunion_handshakes_count_l972_97295

/-- Represents the number of handshakes at a family reunion --/
def family_reunion_handshakes : ℕ :=
  let twin_sets := 7
  let triplet_sets := 4
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let twin_handshakes := twins * (twins - 2)
  let triplet_handshakes := triplets * (triplets - 3)
  let cross_handshakes := twins * (triplets / 3) + triplets * (twins / 4)
  (twin_handshakes + triplet_handshakes + cross_handshakes) / 2

/-- Theorem stating that the number of handshakes at the family reunion is 184 --/
theorem family_reunion_handshakes_count : family_reunion_handshakes = 184 := by
  sorry

end NUMINAMATH_CALUDE_family_reunion_handshakes_count_l972_97295


namespace NUMINAMATH_CALUDE_inequality_proof_l972_97221

theorem inequality_proof (a b c d : ℝ) (h : a > b ∧ b > c ∧ c > d) :
  c < (c * d - a * b) / (c - a + d - b) ∧ (c * d - a * b) / (c - a + d - b) < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l972_97221


namespace NUMINAMATH_CALUDE_inverse_expression_equals_one_thirteenth_l972_97273

theorem inverse_expression_equals_one_thirteenth :
  (3 - 5 * (3 - 4)⁻¹ * 2)⁻¹ = (1 : ℚ) / 13 := by
  sorry

end NUMINAMATH_CALUDE_inverse_expression_equals_one_thirteenth_l972_97273


namespace NUMINAMATH_CALUDE_theta_range_l972_97245

theorem theta_range (θ : Real) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) →
  π / 12 < θ ∧ θ < 5 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_theta_range_l972_97245


namespace NUMINAMATH_CALUDE_binomial_18_choose_6_l972_97281

theorem binomial_18_choose_6 : Nat.choose 18 6 = 7280 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_choose_6_l972_97281


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l972_97254

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) :
  ∃ d : ℚ, d = 3/2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l972_97254


namespace NUMINAMATH_CALUDE_interval_sum_theorem_l972_97222

open Real

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

/-- The function g as defined in the problem -/
noncomputable def g (x : ℝ) : ℝ := (floor x : ℝ) * (2013^(frac x) - 2)

/-- The theorem statement -/
theorem interval_sum_theorem :
  ∃ (S : Set ℝ), S = {x : ℝ | 1 ≤ x ∧ x < 2013 ∧ g x ≤ 0} ∧
  (∫ x in S, 1) = 2012 * (log 2 / log 2013) := by sorry

end NUMINAMATH_CALUDE_interval_sum_theorem_l972_97222


namespace NUMINAMATH_CALUDE_exist_cubes_sum_100_power_100_l972_97223

theorem exist_cubes_sum_100_power_100 : ∃ (a b c d : ℕ+), (a.val ^ 3 + b.val ^ 3 + c.val ^ 3 + d.val ^ 3 : ℕ) = 100 ^ 100 := by
  sorry

end NUMINAMATH_CALUDE_exist_cubes_sum_100_power_100_l972_97223


namespace NUMINAMATH_CALUDE_faster_train_speed_l972_97235

theorem faster_train_speed
  (train_length : ℝ)
  (speed_difference : ℝ)
  (passing_time : ℝ)
  (h1 : train_length = 37.5)
  (h2 : speed_difference = 36)
  (h3 : passing_time = 27)
  : ∃ (faster_speed : ℝ),
    faster_speed = 46 ∧
    (faster_speed - speed_difference) * 1000 / 3600 * passing_time = 2 * train_length :=
by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l972_97235


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_polynomial_l972_97293

theorem sum_of_roots_cubic_polynomial : 
  let p (x : ℝ) := 3 * x^3 - 9 * x^2 - 72 * x - 18
  ∃ (r s t : ℝ), p r = 0 ∧ p s = 0 ∧ p t = 0 ∧ r + s + t = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_polynomial_l972_97293


namespace NUMINAMATH_CALUDE_sum_after_2023_operations_l972_97207

def starting_sequence : List Int := [7, 3, 5]

def operation (seq : List Int) : List Int :=
  seq ++ (List.zip seq (List.tail seq)).map (fun (a, b) => a - b)

def sum_after_n_operations (n : Nat) : Int :=
  n * 2 + (starting_sequence.sum)

theorem sum_after_2023_operations :
  sum_after_n_operations 2023 = 4061 := by sorry

end NUMINAMATH_CALUDE_sum_after_2023_operations_l972_97207


namespace NUMINAMATH_CALUDE_extreme_point_and_extrema_l972_97298

/-- The function f(x) = ax³ - 3x² -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem extreme_point_and_extrema 
  (a : ℝ) 
  (h1 : f_derivative a 2 = 0) :
  a = 1 ∧ 
  (∀ x ∈ Set.Icc (-1 : ℝ) 5, f 1 x ≥ -4) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 5, f 1 x ≤ 50) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 5, f 1 x = -4) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 5, f 1 x = 50) := by
  sorry

end NUMINAMATH_CALUDE_extreme_point_and_extrema_l972_97298


namespace NUMINAMATH_CALUDE_willsons_work_hours_l972_97248

theorem willsons_work_hours : 
  let monday : ℚ := 3/4
  let tuesday : ℚ := 1/2
  let wednesday : ℚ := 2/3
  let thursday : ℚ := 5/6
  let friday : ℚ := 75/60
  monday + tuesday + wednesday + thursday + friday = 4 := by
  sorry

end NUMINAMATH_CALUDE_willsons_work_hours_l972_97248


namespace NUMINAMATH_CALUDE_katie_juice_problem_l972_97289

theorem katie_juice_problem (initial_juice : ℚ) (given_away : ℚ) (remaining : ℚ) :
  initial_juice = 5 →
  given_away = 18 / 7 →
  remaining = initial_juice - given_away →
  remaining = 17 / 7 := by
sorry

end NUMINAMATH_CALUDE_katie_juice_problem_l972_97289


namespace NUMINAMATH_CALUDE_half_sum_abs_diff_squares_l972_97231

theorem half_sum_abs_diff_squares : 
  (1/2 : ℝ) * (|20^2 - 15^2| + |15^2 - 20^2|) = 175 := by
  sorry

end NUMINAMATH_CALUDE_half_sum_abs_diff_squares_l972_97231


namespace NUMINAMATH_CALUDE_perpendicular_line_through_vertex_l972_97237

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 - 4*x + 4

/-- The given line equation -/
def given_line (x y : ℝ) : Prop := x/4 + y/3 = 1

/-- The perpendicular line equation to be proved -/
def perp_line (x y : ℝ) : Prop := y = (4/3)*x - 8/3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 0)

theorem perpendicular_line_through_vertex :
  ∃ (m b : ℝ), 
    (∀ x y, perp_line x y ↔ y = m*x + b) ∧ 
    perp_line vertex.1 vertex.2 ∧
    (∀ x₁ y₁ x₂ y₂, given_line x₁ y₁ → given_line x₂ y₂ → x₁ ≠ x₂ → 
      (y₂ - y₁)/(x₂ - x₁) * m = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_vertex_l972_97237


namespace NUMINAMATH_CALUDE_simons_score_l972_97234

theorem simons_score (n : ℕ) (avg_before avg_after simons_score : ℚ) : 
  n = 21 →
  avg_before = 86 →
  avg_after = 88 →
  n * avg_after = (n - 1) * avg_before + simons_score →
  simons_score = 128 :=
by sorry

end NUMINAMATH_CALUDE_simons_score_l972_97234


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l972_97212

theorem cube_sum_reciprocal (r : ℝ) (h : (r + 1/r)^2 = 3) : r^3 + 1/r^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l972_97212


namespace NUMINAMATH_CALUDE_dan_has_five_marbles_l972_97285

/-- The number of blue marbles Dan has -/
def dans_marbles : ℕ := sorry

/-- The number of blue marbles Mary has -/
def marys_marbles : ℕ := 10

/-- Mary has 2 times more blue marbles than Dan -/
axiom mary_double_dan : marys_marbles = 2 * dans_marbles

theorem dan_has_five_marbles : dans_marbles = 5 := by
  sorry

end NUMINAMATH_CALUDE_dan_has_five_marbles_l972_97285


namespace NUMINAMATH_CALUDE_square_root_of_16_l972_97241

theorem square_root_of_16 : {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_16_l972_97241


namespace NUMINAMATH_CALUDE_natalia_cycling_distance_l972_97264

/-- The total distance ridden by Natalia over four days --/
def total_distance (monday tuesday wednesday thursday : ℕ) : ℕ :=
  monday + tuesday + wednesday + thursday

/-- The problem statement --/
theorem natalia_cycling_distance :
  ∀ (monday tuesday wednesday thursday : ℕ),
    monday = 40 →
    tuesday = 50 →
    wednesday = tuesday / 2 →
    thursday = monday + wednesday →
    total_distance monday tuesday wednesday thursday = 180 := by
  sorry

end NUMINAMATH_CALUDE_natalia_cycling_distance_l972_97264


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_l972_97240

/-- Given 54 pencils and 9 more pencils than pens, prove that the ratio of pens to pencils is 5:6 -/
theorem pen_pencil_ratio : 
  ∀ (num_pens num_pencils : ℕ), 
  num_pencils = 54 → 
  num_pencils = num_pens + 9 → 
  (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_l972_97240


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l972_97274

/-- Two points are symmetric about the origin if their coordinates have opposite signs -/
def symmetric_about_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- If points M(3,a-2) and N(b,a) are symmetric about the origin, then a + b = -2 -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_about_origin 3 (a - 2) b a → a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l972_97274


namespace NUMINAMATH_CALUDE_perfect_square_base8_l972_97213

/-- Represents a number in base 8 of the form ab3c -/
structure Base8Number where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_valid : b < 8
  c_valid : c < 8

/-- Converts a Base8Number to its decimal representation -/
def toDecimal (n : Base8Number) : Nat :=
  512 * n.a + 64 * n.b + 24 + n.c

theorem perfect_square_base8 (n : Base8Number) :
  (∃ m : Nat, toDecimal n = m * m) → n.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_base8_l972_97213


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l972_97205

theorem largest_triangle_perimeter :
  ∀ x : ℕ,
  (7 + 9 > x) ∧ (7 + x > 9) ∧ (9 + x > 7) →
  (∀ y : ℕ, (7 + 9 > y) ∧ (7 + y > 9) ∧ (9 + y > 7) → x ≥ y) →
  7 + 9 + x = 31 :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l972_97205


namespace NUMINAMATH_CALUDE_tan_150_degrees_l972_97202

theorem tan_150_degrees : Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l972_97202


namespace NUMINAMATH_CALUDE_age_difference_l972_97225

theorem age_difference (A B : ℕ) : B = 35 → A + 10 = 2 * (B - 10) → A - B = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l972_97225


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l972_97203

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The first three terms of the arithmetic progression -/
def first_three_terms (x : ℝ) : ℕ → ℝ
| 0 => x - 2
| 1 => x + 2
| 2 => 3*x + 4
| _ => 0  -- This is just a placeholder for other terms

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, is_arithmetic_progression (first_three_terms x) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l972_97203


namespace NUMINAMATH_CALUDE_initial_apps_correct_l972_97270

/-- The initial number of apps Dave had on his phone -/
def initial_apps : ℕ := 10

/-- The number of apps Dave added -/
def added_apps : ℕ := 11

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 17

/-- The number of apps Dave had left after adding and deleting -/
def remaining_apps : ℕ := 4

/-- Theorem stating that the initial number of apps is correct -/
theorem initial_apps_correct : 
  initial_apps + added_apps - deleted_apps = remaining_apps := by
  sorry

#check initial_apps_correct

end NUMINAMATH_CALUDE_initial_apps_correct_l972_97270


namespace NUMINAMATH_CALUDE_sum_difference_zero_l972_97294

theorem sum_difference_zero (x y z : ℝ) 
  (h : (2*x^2 + 8*x + 11)*(y^2 - 10*y + 29)*(3*z^2 - 18*z + 32) ≤ 60) : 
  x + y - z = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_zero_l972_97294


namespace NUMINAMATH_CALUDE_negation_of_exists_cube_positive_l972_97265

theorem negation_of_exists_cube_positive :
  (¬ ∃ x : ℝ, x^3 > 0) ↔ (∀ x : ℝ, x^3 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_exists_cube_positive_l972_97265


namespace NUMINAMATH_CALUDE_circle_line_intersection_l972_97249

theorem circle_line_intersection (a b : ℝ) : 
  (a^2 + b^2 > 1) →
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y + 2 = 0) →
  (a^2 + b^2 > 1) ∧
  ¬(∀ (a b : ℝ), a^2 + b^2 > 1 → ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*x + b*y + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l972_97249


namespace NUMINAMATH_CALUDE_nested_average_equals_seven_sixths_l972_97287

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of three numbers
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

-- The main theorem
theorem nested_average_equals_seven_sixths :
  avg3 (avg3 2 1 0) (avg2 1 2) 1 = 7/6 := by sorry

end NUMINAMATH_CALUDE_nested_average_equals_seven_sixths_l972_97287


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l972_97263

theorem average_of_remaining_numbers
  (total : ℝ) (group1 : ℝ) (group2 : ℝ) (group3 : ℝ)
  (h1 : total = 6 * 3.95)
  (h2 : group1 = 2 * 4.4)
  (h3 : group2 = 2 * 3.85)
  (h4 : group3 = total - (group1 + group2)) :
  group3 / 2 = 3.6 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l972_97263


namespace NUMINAMATH_CALUDE_tangent_segments_area_l972_97238

/-- The area of the region formed by all line segments of length 2 units 
    tangent to a circle with radius 3 units is equal to 4π. -/
theorem tangent_segments_area (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 2) : 
  let outer_radius := Real.sqrt (r^2 + l^2)
  π * outer_radius^2 - π * r^2 = 4 * π := by
sorry

end NUMINAMATH_CALUDE_tangent_segments_area_l972_97238


namespace NUMINAMATH_CALUDE_money_distribution_l972_97226

theorem money_distribution (x : ℚ) : 
  x > 0 →
  let moe_initial := 6 * x
  let loki_initial := 5 * x
  let nick_initial := 4 * x
  let ott_received := 3 * x
  let total_money := moe_initial + loki_initial + nick_initial
  ott_received / total_money = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l972_97226


namespace NUMINAMATH_CALUDE_base5_calculation_l972_97290

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Theorem: 231₅ × 24₅ - 12₅ = 12132₅ in base 5 --/
theorem base5_calculation : 
  base10ToBase5 (base5ToBase10 231 * base5ToBase10 24 - base5ToBase10 12) = 12132 := by sorry

end NUMINAMATH_CALUDE_base5_calculation_l972_97290


namespace NUMINAMATH_CALUDE_triangle_height_l972_97211

theorem triangle_height (A B C : Real) (a b c : Real) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  1 + 2 * Real.cos (B + C) = 0 →
  ∃ h : Real, h = b * Real.sin C ∧ h = (Real.sqrt 3 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_l972_97211


namespace NUMINAMATH_CALUDE_inequality_equivalence_l972_97253

theorem inequality_equivalence (x : ℝ) : -1/2 * x + 3 < 0 ↔ x > 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l972_97253


namespace NUMINAMATH_CALUDE_cistern_leak_empty_time_l972_97255

/-- Given a cistern with normal fill time and leak-affected fill time, 
    calculate the time it takes for the leak to empty the full cistern. -/
theorem cistern_leak_empty_time 
  (normal_fill_time : ℝ) 
  (leak_fill_time : ℝ) 
  (h1 : normal_fill_time = 12) 
  (h2 : leak_fill_time = normal_fill_time + 2) : 
  (1 / ((1 / normal_fill_time) - (1 / leak_fill_time))) = 84 := by
  sorry

end NUMINAMATH_CALUDE_cistern_leak_empty_time_l972_97255


namespace NUMINAMATH_CALUDE_min_side_arithmetic_angles_l972_97279

/-- Given a triangle ABC where the internal angles form an arithmetic sequence and the area is 2√3,
    the minimum value of side AB is 2√2. -/
theorem min_side_arithmetic_angles (A B C : ℝ) (a b c : ℝ) :
  -- Angles form an arithmetic sequence
  2 * C = A + B →
  -- Sum of angles in a triangle is π
  A + B + C = π →
  -- Area of the triangle is 2√3
  (1 / 2) * a * b * Real.sin C = 2 * Real.sqrt 3 →
  -- AB is the side opposite to angle C
  c = (a^2 + b^2 - 2*a*b*(Real.cos C))^(1/2) →
  -- Minimum value of AB (c) is 2√2
  c ≥ 2 * Real.sqrt 2 ∧ ∃ (a' b' : ℝ), c = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_side_arithmetic_angles_l972_97279
