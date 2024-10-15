import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l269_26908

-- Define the region D
def D : Set (ℝ × ℝ) := {(x, y) | (x - 1)^2 + (y - 2)^2 ≤ 4}

-- Define proposition p
def p : Prop := ∀ (x y : ℝ), (x, y) ∈ D → 2*x + y ≤ 8

-- Define proposition q
def q : Prop := ∃ (x y : ℝ), (x, y) ∈ D ∧ 2*x + y ≤ -1

-- Theorem to prove
theorem problem_solution : (¬p ∨ q) ∧ (¬p ∧ ¬q) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l269_26908


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l269_26939

theorem quadratic_inequality_solution_set : 
  {x : ℝ | x^2 - 50*x + 500 ≤ 9} = {x : ℝ | 13.42 ≤ x ∧ x ≤ 36.58} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l269_26939


namespace NUMINAMATH_CALUDE_olympiad_colors_l269_26996

-- Define the colors
inductive Color
  | Red
  | Yellow
  | Blue

-- Define a person's outfit
structure Outfit :=
  (dress : Color)
  (notebook : Color)

-- Define the problem statement
theorem olympiad_colors :
  ∃ (sveta tanya ira : Outfit),
    -- All dress colors are different
    sveta.dress ≠ tanya.dress ∧ sveta.dress ≠ ira.dress ∧ tanya.dress ≠ ira.dress ∧
    -- All notebook colors are different
    sveta.notebook ≠ tanya.notebook ∧ sveta.notebook ≠ ira.notebook ∧ tanya.notebook ≠ ira.notebook ∧
    -- Only Sveta's dress and notebook colors match
    (sveta.dress = sveta.notebook) ∧
    (tanya.dress ≠ tanya.notebook) ∧
    (ira.dress ≠ ira.notebook) ∧
    -- Tanya's dress and notebook are not red
    (tanya.dress ≠ Color.Red) ∧ (tanya.notebook ≠ Color.Red) ∧
    -- Ira has a yellow notebook
    (ira.notebook = Color.Yellow) ∧
    -- The solution
    sveta = Outfit.mk Color.Red Color.Red ∧
    ira = Outfit.mk Color.Blue Color.Yellow ∧
    tanya = Outfit.mk Color.Yellow Color.Blue :=
by
  sorry

end NUMINAMATH_CALUDE_olympiad_colors_l269_26996


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l269_26937

theorem yellow_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 85 →
  blue = 3 * red →
  red = 14 →
  total = red + blue + yellow →
  yellow = 29 := by
sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l269_26937


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l269_26915

theorem solution_set_of_equation (x : ℝ) : 
  (16 * Real.sin (π * x) * Real.cos (π * x) = 16 * x + 1 / x) ↔ (x = 1/4 ∨ x = -1/4) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l269_26915


namespace NUMINAMATH_CALUDE_function_range_exclusion_l269_26993

theorem function_range_exclusion (a : ℕ) : 
  (a > 3 → ∃ x : ℝ, -4 ≤ (8*x - 20) / (a - x^2) ∧ (8*x - 20) / (a - x^2) ≤ -1) ∧ 
  (∀ x : ℝ, (8*x - 20) / (3 - x^2) < -4 ∨ (8*x - 20) / (3 - x^2) > -1) :=
sorry

end NUMINAMATH_CALUDE_function_range_exclusion_l269_26993


namespace NUMINAMATH_CALUDE_diana_statues_l269_26941

/-- Given the amount of paint available and the amount required per statue, 
    calculate the number of statues that can be painted. -/
def statues_paintable (paint_available : ℚ) (paint_per_statue : ℚ) : ℚ :=
  paint_available / paint_per_statue

/-- Theorem: Diana can paint 2 statues with the remaining paint. -/
theorem diana_statues : 
  let paint_available : ℚ := 1/2
  let paint_per_statue : ℚ := 1/4
  statues_paintable paint_available paint_per_statue = 2 := by
  sorry

end NUMINAMATH_CALUDE_diana_statues_l269_26941


namespace NUMINAMATH_CALUDE_midpoint_lines_perpendicular_l269_26922

/-- A circle in which a quadrilateral is inscribed -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point on a circle -/
structure CirclePoint (c : Circle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral (c : Circle) where
  A : CirclePoint c
  B : CirclePoint c
  C : CirclePoint c
  D : CirclePoint c

/-- Midpoint of an arc on a circle -/
def arcMidpoint (c : Circle) (p1 p2 : CirclePoint c) : CirclePoint c :=
  sorry

/-- Perpendicularity of two lines defined by four points -/
def arePerpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  sorry

/-- Theorem statement -/
theorem midpoint_lines_perpendicular (c : Circle) (quad : InscribedQuadrilateral c) :
  let M := arcMidpoint c quad.A quad.B
  let N := arcMidpoint c quad.B quad.C
  let P := arcMidpoint c quad.C quad.D
  let Q := arcMidpoint c quad.D quad.A
  arePerpendicular M.point P.point N.point Q.point :=
by sorry

end NUMINAMATH_CALUDE_midpoint_lines_perpendicular_l269_26922


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l269_26940

def cost_price : ℝ := 1500
def selling_price : ℝ := 1335

theorem loss_percentage_calculation :
  (cost_price - selling_price) / cost_price * 100 = 11 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l269_26940


namespace NUMINAMATH_CALUDE_point_P_properties_l269_26921

def P (a : ℝ) : ℝ × ℝ := (-3*a - 4, 2 + a)

def Q : ℝ × ℝ := (5, 8)

theorem point_P_properties :
  (∀ a : ℝ, P a = (2, 0) → (P a).2 = 0) ∧
  (∀ a : ℝ, (P a).1 = Q.1 → P a = (5, -1)) := by
  sorry

end NUMINAMATH_CALUDE_point_P_properties_l269_26921


namespace NUMINAMATH_CALUDE_projection_matrix_values_l269_26975

/-- A projection matrix P satisfies P² = P -/
def IsProjectionMatrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific matrix form given in the problem -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 15/34],
    ![c, 25/34]]

theorem projection_matrix_values :
  ∀ a c : ℚ, IsProjectionMatrix (P a c) ↔ a = 9/34 ∧ c = 15/34 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l269_26975


namespace NUMINAMATH_CALUDE_calculation_proof_l269_26928

theorem calculation_proof : 
  100 - (25/8) / ((25/12) - (5/8)) * ((8/5) + (8/3)) = 636/7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l269_26928


namespace NUMINAMATH_CALUDE_probability_not_black_l269_26936

theorem probability_not_black (white black red : ℕ) (h1 : white = 7) (h2 : black = 6) (h3 : red = 4) :
  (white + red : ℚ) / (white + black + red) = 11 / 17 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_black_l269_26936


namespace NUMINAMATH_CALUDE_max_k_for_circle_intersection_l269_26933

/-- The maximum value of k for which a circle with radius 1 centered on the line y = kx - 2
    has a common point with the circle x^2 + y^2 - 8x + 15 = 0 is 4/3 -/
theorem max_k_for_circle_intersection :
  let C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 8*p.1 + 15 = 0}
  let line (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k*p.1 - 2}
  let unit_circle_on_line (k : ℝ) : Set (Set (ℝ × ℝ)) :=
    {S | ∃ c ∈ line k, S = {p | (p.1 - c.1)^2 + (p.2 - c.2)^2 = 1}}
  ∀ k > 4/3, ∀ S ∈ unit_circle_on_line k, S ∩ C = ∅ ∧
  ∃ S ∈ unit_circle_on_line (4/3), S ∩ C ≠ ∅ :=
by sorry

end NUMINAMATH_CALUDE_max_k_for_circle_intersection_l269_26933


namespace NUMINAMATH_CALUDE_chord_length_perpendicular_chord_m_l269_26923

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equations
def line1_equation (x y : ℝ) : Prop :=
  x + y - 1 = 0

def line2_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Part 1: Chord length
theorem chord_length : 
  ∀ x y : ℝ, circle_equation x y 1 ∧ line1_equation x y → 
  ∃ chord_length : ℝ, chord_length = 2 * Real.sqrt 2 :=
sorry

-- Part 2: Value of m
theorem perpendicular_chord_m :
  ∃ m : ℝ, ∀ x1 y1 x2 y2 : ℝ,
    circle_equation x1 y1 m ∧ circle_equation x2 y2 m ∧
    line2_equation x1 y1 ∧ line2_equation x2 y2 ∧
    x1 * x2 + y1 * y2 = 0 →
    m = 8/5 :=
sorry

end NUMINAMATH_CALUDE_chord_length_perpendicular_chord_m_l269_26923


namespace NUMINAMATH_CALUDE_triangle_integer_points_l269_26931

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Checks if a point has integer coordinates -/
def hasIntegerCoordinates (p : Point) : Prop :=
  ∃ (ix iy : ℤ), p.x = ↑ix ∧ p.y = ↑iy

/-- Checks if a point is inside or on the boundary of the triangle formed by three points -/
def isInsideOrOnBoundary (p A B C : Point) : Prop :=
  sorry -- Definition of this predicate

/-- Counts the number of points with integer coordinates inside or on the boundary of the triangle -/
def countIntegerPoints (A B C : Point) : ℕ :=
  sorry -- Definition of this function

/-- The main theorem -/
theorem triangle_integer_points (a : ℝ) :
  a > 0 →
  let A : Point := ⟨2 + a, 0⟩
  let B : Point := ⟨2 - a, 0⟩
  let C : Point := ⟨2, 1⟩
  (countIntegerPoints A B C = 4) ↔ (1 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_integer_points_l269_26931


namespace NUMINAMATH_CALUDE_intersection_point_sum_l269_26983

/-- Two lines in a plane -/
structure TwoLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ

/-- Points P, Q, and T for the given lines -/
structure LinePoints (l : TwoLines) where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  T : ℝ × ℝ
  h_P : l.line1 P.1 = P.2 ∧ P.2 = 0
  h_Q : l.line1 Q.1 = Q.2 ∧ Q.1 = 0
  h_T : l.line1 T.1 = T.2 ∧ l.line2 T.1 = T.2

/-- The theorem statement -/
theorem intersection_point_sum (l : TwoLines) (pts : LinePoints l) 
  (h_line1 : ∀ x, l.line1 x = -2/3 * x + 8)
  (h_line2 : ∀ x, l.line2 x = 3/2 * x - 9)
  (h_area : (pts.P.1 * pts.Q.2) / 2 = 2 * ((pts.P.1 - pts.T.1) * pts.T.2) / 2) :
  pts.T.1 + pts.T.2 = 138/13 := by sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l269_26983


namespace NUMINAMATH_CALUDE_f_properties_l269_26916

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * (x - 1)^2 - 2*x + 3 + Real.log x

theorem f_properties (m : ℝ) (h : m ≥ 1) :
  (∃ a b, a > 0 ∧ b > 0 ∧ a < b ∧ ∀ x ∈ Set.Icc a b, (deriv (f m)) x ≤ 0) ∧
  (∃! m, ∀ x, x > 0 → (f m x = -x + 2 → x = 1)) ∧
  (∀ x, x > 0 → (f 1 x = -x + 2 → x = 1)) := by sorry

end NUMINAMATH_CALUDE_f_properties_l269_26916


namespace NUMINAMATH_CALUDE_total_area_is_135_l269_26959

/-- Represents the geometry of villages, roads, fields, and forest --/
structure VillageGeometry where
  /-- Side length of the square field --/
  r : ℝ
  /-- Side length of the rectangular field along the road --/
  p : ℝ
  /-- Side length of the rectangular forest along the road --/
  q : ℝ

/-- The total area of the forest and fields is 135 sq km --/
theorem total_area_is_135 (g : VillageGeometry) : 
  g.r^2 + 4 * g.p^2 + 12 * g.q = 135 :=
by
  sorry

/-- The forest area is 45 sq km more than the sum of field areas --/
axiom forest_area_relation (g : VillageGeometry) : 
  12 * g.q = g.r^2 + 4 * g.p^2 + 45

/-- The side of the rectangular field perpendicular to the road is 4 times longer --/
axiom rectangular_field_proportion (g : VillageGeometry) : 
  4 * g.p = g.q

/-- The side of the rectangular forest perpendicular to the road is 12 km --/
axiom forest_width (g : VillageGeometry) : g.q = 12

end NUMINAMATH_CALUDE_total_area_is_135_l269_26959


namespace NUMINAMATH_CALUDE_triangular_pyramid_distance_sum_l269_26932

/-- A triangular pyramid with volume V, face areas (S₁, S₂, S₃, S₄), and distances (H₁, H₂, H₃, H₄) from any internal point Q to each face. -/
structure TriangularPyramid where
  V : ℝ
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  S₄ : ℝ
  H₁ : ℝ
  H₂ : ℝ
  H₃ : ℝ
  H₄ : ℝ
  K : ℝ
  volume_positive : V > 0
  areas_positive : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0
  distances_positive : H₁ > 0 ∧ H₂ > 0 ∧ H₃ > 0 ∧ H₄ > 0
  K_positive : K > 0
  area_ratios : S₁ = K ∧ S₂ = 2*K ∧ S₃ = 3*K ∧ S₄ = 4*K

/-- The theorem stating the relationship between distances, volume, and K for a triangular pyramid. -/
theorem triangular_pyramid_distance_sum (p : TriangularPyramid) :
  p.H₁ + 2*p.H₂ + 3*p.H₃ + 4*p.H₄ = 3*p.V/p.K :=
by sorry

end NUMINAMATH_CALUDE_triangular_pyramid_distance_sum_l269_26932


namespace NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_odd_unique_digits_l269_26951

/-- A function that checks if a natural number has all odd and unique digits -/
def hasOddUniqueDigits (n : ℕ) : Prop := sorry

/-- A function that returns the remainder when n is divided by m -/
def remainder (n m : ℕ) : ℕ := sorry

/-- The theorem statement -/
theorem greatest_multiple_of_nine_with_odd_unique_digits :
  ∃ M : ℕ,
    M % 9 = 0 ∧
    hasOddUniqueDigits M ∧
    (∀ N : ℕ, N % 9 = 0 → hasOddUniqueDigits N → N ≤ M) ∧
    remainder M 1000 = 531 := by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_nine_with_odd_unique_digits_l269_26951


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l269_26991

def total_balls : ℕ := 15
def white_balls : ℕ := 7
def black_balls : ℕ := 8
def drawn_balls : ℕ := 3

theorem probability_three_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l269_26991


namespace NUMINAMATH_CALUDE_sqrt_four_squared_l269_26977

theorem sqrt_four_squared : Real.sqrt (4^2) = 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_l269_26977


namespace NUMINAMATH_CALUDE_x_value_l269_26924

theorem x_value : ∃ x : ℚ, (10 * x = x + 20) ∧ (x = 20 / 9) := by sorry

end NUMINAMATH_CALUDE_x_value_l269_26924


namespace NUMINAMATH_CALUDE_discount_savings_l269_26938

theorem discount_savings (original_price discounted_price : ℝ) 
  (h1 : discounted_price = original_price * 0.8)
  (h2 : discounted_price = 48)
  (h3 : original_price > 0) :
  (original_price - discounted_price) / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_discount_savings_l269_26938


namespace NUMINAMATH_CALUDE_six_coin_flip_probability_six_coin_flip_probability_is_one_thirtysecond_l269_26986

theorem six_coin_flip_probability : ℝ :=
  let n : ℕ := 6  -- number of coins
  let p : ℝ := 1 / 2  -- probability of heads for a fair coin
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := 2  -- all heads or all tails
  favorable_outcomes / total_outcomes

theorem six_coin_flip_probability_is_one_thirtysecond : 
  six_coin_flip_probability = 1 / 32 := by
  sorry

end NUMINAMATH_CALUDE_six_coin_flip_probability_six_coin_flip_probability_is_one_thirtysecond_l269_26986


namespace NUMINAMATH_CALUDE_expression_factorization_l269_26943

theorem expression_factorization (y : ℝ) :
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 9 * y^4 + 9) = 3 * (4 * y^6 + 15 * y^4 - 6) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l269_26943


namespace NUMINAMATH_CALUDE_physics_score_l269_26926

/-- Represents the scores in physics, chemistry, and mathematics --/
structure Scores where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ

/-- The average score of all three subjects is 65 --/
def average_all (s : Scores) : Prop :=
  (s.physics + s.chemistry + s.mathematics) / 3 = 65

/-- The average score of physics and mathematics is 90 --/
def average_physics_math (s : Scores) : Prop :=
  (s.physics + s.mathematics) / 2 = 90

/-- The average score of physics and chemistry is 70 --/
def average_physics_chem (s : Scores) : Prop :=
  (s.physics + s.chemistry) / 2 = 70

/-- Given the conditions, prove that the score in physics is 125 --/
theorem physics_score (s : Scores) 
  (h1 : average_all s) 
  (h2 : average_physics_math s) 
  (h3 : average_physics_chem s) : 
  s.physics = 125 := by
  sorry

end NUMINAMATH_CALUDE_physics_score_l269_26926


namespace NUMINAMATH_CALUDE_ellipse_k_value_l269_26963

-- Define the ellipse equation
def ellipse_equation (k : ℝ) (x y : ℝ) : Prop :=
  4 * x^2 + k * y^2 = 4

-- Define the focus point
def focus : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem ellipse_k_value :
  ∃ (k : ℝ), 
    (∀ (x y : ℝ), ellipse_equation k x y → 
      ∃ (c : ℝ), c^2 = (4/k) - 1 ∧ c = 1) →
    k = 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l269_26963


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l269_26930

theorem sqrt_abs_sum_zero_implies_power (x y : ℝ) :
  Real.sqrt (2 * x + 8) + |y - 3| = 0 → (x + y)^2021 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l269_26930


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l269_26935

theorem min_value_sum_of_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) : 
  2/a + 4/b + 6/c + 16/d + 20/e + 30/f ≥ 2053.78 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l269_26935


namespace NUMINAMATH_CALUDE_music_school_population_l269_26968

/-- Given a music school with boys, girls, and teachers, prove that the total number of people is 9b/7, where b is the number of boys. -/
theorem music_school_population (b g t : ℚ) : 
  b = 4 * g ∧ g = 7 * t → b + g + t = 9 * b / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_music_school_population_l269_26968


namespace NUMINAMATH_CALUDE_no_obtuse_angles_l269_26974

-- Define an isosceles triangle with two 70-degree angles
structure IsoscelesTriangle70 where
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  is_isosceles : angle_a = angle_b
  angles_70 : angle_a = 70 ∧ angle_b = 70
  sum_180 : angle_a + angle_b + angle_c = 180

-- Define what an obtuse angle is
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem no_obtuse_angles (t : IsoscelesTriangle70) :
  ¬ (is_obtuse t.angle_a ∨ is_obtuse t.angle_b ∨ is_obtuse t.angle_c) :=
by sorry

end NUMINAMATH_CALUDE_no_obtuse_angles_l269_26974


namespace NUMINAMATH_CALUDE_inscribed_squares_max_distance_l269_26994

def inner_square_perimeter : ℝ := 20
def outer_square_perimeter : ℝ := 28

theorem inscribed_squares_max_distance :
  let inner_side := inner_square_perimeter / 4
  let outer_side := outer_square_perimeter / 4
  ∃ (x y : ℝ),
    x + y = outer_side ∧
    x^2 + y^2 = inner_side^2 ∧
    Real.sqrt (x^2 + (x + y)^2) = Real.sqrt 65 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_max_distance_l269_26994


namespace NUMINAMATH_CALUDE_cylinder_volume_equals_54_sqrt3_over_sqrt_pi_l269_26913

/-- Given a cube with side length 3 and a cylinder with the same surface area as the cube,
    where the cylinder's height equals its diameter, prove that the volume of the cylinder
    is 54 * sqrt(3) / sqrt(π). -/
theorem cylinder_volume_equals_54_sqrt3_over_sqrt_pi
  (cube_side : ℝ)
  (cylinder_radius : ℝ)
  (cylinder_height : ℝ)
  (h1 : cube_side = 3)
  (h2 : 6 * cube_side^2 = 2 * π * cylinder_radius^2 + 2 * π * cylinder_radius * cylinder_height)
  (h3 : cylinder_height = 2 * cylinder_radius) :
  π * cylinder_radius^2 * cylinder_height = 54 * Real.sqrt 3 / Real.sqrt π :=
by sorry


end NUMINAMATH_CALUDE_cylinder_volume_equals_54_sqrt3_over_sqrt_pi_l269_26913


namespace NUMINAMATH_CALUDE_y0_minus_one_is_perfect_square_l269_26976

theorem y0_minus_one_is_perfect_square 
  (x y : ℕ → ℕ) 
  (h : ∀ n, (x n : ℝ) + Real.sqrt 2 * (y n) = Real.sqrt 2 * (3 + 2 * Real.sqrt 2) ^ (2 ^ n)) : 
  ∃ k : ℕ, y 0 - 1 = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_y0_minus_one_is_perfect_square_l269_26976


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l269_26965

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x = 0 → x^2 - 2*x = 0) ∧ ¬(x^2 - 2*x = 0 → x = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l269_26965


namespace NUMINAMATH_CALUDE_simplify_square_roots_l269_26900

theorem simplify_square_roots : 
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l269_26900


namespace NUMINAMATH_CALUDE_equation_has_one_solution_l269_26929

theorem equation_has_one_solution :
  ∃! x : ℝ, x - 8 / (x - 2) = 4 - 8 / (x - 2) ∧ x ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_one_solution_l269_26929


namespace NUMINAMATH_CALUDE_hawks_score_l269_26966

/-- Calculates the total score for a team given the number of touchdowns and points per touchdown -/
def totalScore (touchdowns : ℕ) (pointsPerTouchdown : ℕ) : ℕ :=
  touchdowns * pointsPerTouchdown

/-- Theorem: If a team scores 3 touchdowns, and each touchdown is worth 7 points, then the team's total score is 21 points -/
theorem hawks_score :
  totalScore 3 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l269_26966


namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_seven_l269_26956

theorem unique_three_digit_divisible_by_seven :
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    n % 10 = 5 ∧          -- units digit is 5
    n / 100 = 6 ∧         -- hundreds digit is 6
    n % 7 = 0             -- divisible by 7
  := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_seven_l269_26956


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l269_26961

def f (x : ℝ) : ℝ := 8*x^5 - 10*x^4 + 3*x^3 + 5*x^2 - 7*x - 35

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := by sorry

theorem polynomial_remainder (x : ℝ) :
  ∃ q : ℝ → ℝ, f x = (x - 5) * q x + 19180 := by
  have h := remainder_theorem f 5
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l269_26961


namespace NUMINAMATH_CALUDE_sqrt_of_repeating_ones_100_l269_26948

theorem sqrt_of_repeating_ones_100 :
  let x := (10^100 - 1) / (9 * 10^100)
  0.10049987498 < Real.sqrt x ∧ Real.sqrt x < 0.10049987499 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_repeating_ones_100_l269_26948


namespace NUMINAMATH_CALUDE_functional_equation_properties_l269_26995

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem functional_equation_properties (f : ℝ → ℝ) (h : FunctionalEquation f) :
  (f 0 = 0) ∧ (f 1 = 0) ∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_properties_l269_26995


namespace NUMINAMATH_CALUDE_taxi_journey_theorem_l269_26971

def itinerary : List Int := [-15, 4, -5, 10, -12, 5, 8, -7]
def gasoline_consumption : Rat := 10 / 100  -- 10 liters per 100 km
def gasoline_price : Rat := 8  -- 8 yuan per liter

def total_distance (route : List Int) : Int :=
  route.map (Int.natAbs) |>.sum

theorem taxi_journey_theorem :
  let distance := total_distance itinerary
  let cost := (distance : Rat) * gasoline_consumption * gasoline_price
  distance = 66 ∧ cost = 52.8 := by sorry

end NUMINAMATH_CALUDE_taxi_journey_theorem_l269_26971


namespace NUMINAMATH_CALUDE_soap_survey_ratio_l269_26985

/-- Represents the survey results of household soap usage -/
structure SoapSurvey where
  total : ℕ
  neither : ℕ
  onlyE : ℕ
  both : ℕ
  onlyB : ℕ

/-- The ratio of households using only brand B to those using both brands -/
def brandBRatio (survey : SoapSurvey) : ℚ :=
  survey.onlyB / survey.both

/-- The survey satisfies the given conditions -/
def validSurvey (survey : SoapSurvey) : Prop :=
  survey.total = 200 ∧
  survey.neither = 80 ∧
  survey.onlyE = 60 ∧
  survey.both = 40 ∧
  survey.total = survey.neither + survey.onlyE + survey.onlyB + survey.both

theorem soap_survey_ratio (survey : SoapSurvey) (h : validSurvey survey) :
  brandBRatio survey = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_soap_survey_ratio_l269_26985


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l269_26927

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_standard_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (1^2 + 2^2 = (2 * a^2 - 1)) →
  (b / a = 2) →
  (∃ (x y : ℝ), x^2 - y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l269_26927


namespace NUMINAMATH_CALUDE_max_sum_squares_triangle_sides_l269_26997

theorem max_sum_squares_triangle_sides (a : ℝ) (α : ℝ) 
  (h_a_pos : a > 0) (h_α_acute : 0 < α ∧ α < π / 2) :
  ∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ 
    b^2 + c^2 = a^2 / (1 - Real.cos α) ∧
    ∀ (b' c' : ℝ), b' > 0 → c' > 0 → 
      b'^2 + a^2 = c'^2 + 2 * a * b' * Real.cos α →
      b'^2 + c'^2 ≤ a^2 / (1 - Real.cos α) := by
sorry


end NUMINAMATH_CALUDE_max_sum_squares_triangle_sides_l269_26997


namespace NUMINAMATH_CALUDE_prime_triplet_equation_l269_26902

theorem prime_triplet_equation : 
  ∀ p q r : ℕ, 
    Prime p → Prime q → Prime r → 
    p^q + q^p = r → 
    ((p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17)) := by
  sorry

end NUMINAMATH_CALUDE_prime_triplet_equation_l269_26902


namespace NUMINAMATH_CALUDE_final_expression_l269_26903

theorem final_expression (y : ℝ) : 3 * (1/2 * (12*y + 3)) = 18*y + 4.5 := by
  sorry

end NUMINAMATH_CALUDE_final_expression_l269_26903


namespace NUMINAMATH_CALUDE_min_x_plus_y_min_value_is_9_4_min_achieved_l269_26907

-- Define the optimization problem
theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 / y + 1 / x = 4) :
  ∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 / y' + 1 / x' = 4 → x + y ≤ x' + y' :=
by sorry

-- State the minimum value
theorem min_value_is_9_4 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 / y + 1 / x = 4) :
  x + y ≥ 9 / 4 :=
by sorry

-- Prove the minimum is achieved
theorem min_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 / y + 1 / x = 4 ∧ x + y < 9 / 4 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_x_plus_y_min_value_is_9_4_min_achieved_l269_26907


namespace NUMINAMATH_CALUDE_tower_remainder_l269_26953

/-- Represents the number of towers that can be built with cubes of sizes 1 to n -/
def T : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 6
| 3 => 18
| n+4 => 4 * T (n+3)

/-- The main theorem stating the result for 9 cubes -/
theorem tower_remainder : T 9 % 1000 = 296 := by
  sorry


end NUMINAMATH_CALUDE_tower_remainder_l269_26953


namespace NUMINAMATH_CALUDE_tangent_circle_condition_l269_26914

/-- The line 2x + y - 2 = 0 is tangent to the circle (x - 1)^2 + (y - a)^2 = 1 -/
def is_tangent (a : ℝ) : Prop :=
  ∃ (x y : ℝ), (2 * x + y - 2 = 0) ∧ ((x - 1)^2 + (y - a)^2 = 1)

/-- If the line is tangent to the circle, then a = ± √5 -/
theorem tangent_circle_condition (a : ℝ) :
  is_tangent a → (a = Real.sqrt 5 ∨ a = -Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_tangent_circle_condition_l269_26914


namespace NUMINAMATH_CALUDE_special_polynomial_form_l269_26978

/-- A polynomial in two variables satisfying specific conditions -/
structure SpecialPolynomial where
  P : ℝ → ℝ → ℝ
  n : ℕ+
  homogeneous : ∀ (t x y : ℝ), P (t * x) (t * y) = t ^ n.val * P x y
  cyclic_sum : ∀ (x y z : ℝ), P (y + z) x + P (z + x) y + P (x + y) z = 0
  normalization : P 1 0 = 1

/-- The theorem stating the form of the special polynomial -/
theorem special_polynomial_form (sp : SpecialPolynomial) :
  ∀ (x y : ℝ), sp.P x y = (x + y) ^ sp.n.val * (x - 2 * y) := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_form_l269_26978


namespace NUMINAMATH_CALUDE_investment_amount_is_14400_l269_26912

/-- Represents the investment scenario --/
structure Investment where
  face_value : ℕ
  premium_percentage : ℕ
  dividend_percentage : ℕ
  total_dividend : ℕ

/-- Calculates the amount invested given the investment parameters --/
def amount_invested (i : Investment) : ℕ :=
  let share_price := i.face_value + i.face_value * i.premium_percentage / 100
  let dividend_per_share := i.face_value * i.dividend_percentage / 100
  let num_shares := i.total_dividend / dividend_per_share
  num_shares * share_price

/-- Theorem stating that the amount invested is 14400 given the specific conditions --/
theorem investment_amount_is_14400 :
  ∀ i : Investment,
    i.face_value = 100 ∧
    i.premium_percentage = 20 ∧
    i.dividend_percentage = 5 ∧
    i.total_dividend = 600 →
    amount_invested i = 14400 :=
by
  sorry

end NUMINAMATH_CALUDE_investment_amount_is_14400_l269_26912


namespace NUMINAMATH_CALUDE_marble_fraction_l269_26911

theorem marble_fraction (x : ℚ) : 
  let initial_blue := (2 : ℚ) / 3 * x
  let initial_red := x - initial_blue
  let new_red := 2 * initial_red
  let new_blue := (3 : ℚ) / 2 * initial_blue
  let total_new := new_red + new_blue
  new_red / total_new = (2 : ℚ) / 5 := by sorry

end NUMINAMATH_CALUDE_marble_fraction_l269_26911


namespace NUMINAMATH_CALUDE_no_special_pentagon_l269_26949

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a pentagon as a set of 5 points
def Pentagon : Type := { p : Finset Point3D // p.card = 5 }

-- Define a function to check if three points are colinear
def areColinear (p q r : Point3D) : Prop := sorry

-- Define a function to check if a point is in the interior of a triangle
def isInteriorPoint (p : Point3D) (t1 t2 t3 : Point3D) : Prop := sorry

-- Define a function to check if a line segment intersects a plane at an interior point of a triangle
def intersectsTriangleInterior (p1 p2 t1 t2 t3 : Point3D) : Prop := sorry

-- Main theorem
theorem no_special_pentagon : 
  ¬ ∃ (pent : Pentagon), 
    ∀ (v1 v2 v3 v4 v5 : Point3D),
      v1 ∈ pent.val → v2 ∈ pent.val → v3 ∈ pent.val → v4 ∈ pent.val → v5 ∈ pent.val →
      v1 ≠ v2 → v1 ≠ v3 → v1 ≠ v4 → v1 ≠ v5 → v2 ≠ v3 → v2 ≠ v4 → v2 ≠ v5 → v3 ≠ v4 → v3 ≠ v5 → v4 ≠ v5 →
      (intersectsTriangleInterior v1 v3 v2 v4 v5 ∧
       intersectsTriangleInterior v1 v4 v2 v3 v5 ∧
       intersectsTriangleInterior v2 v4 v1 v3 v5 ∧
       intersectsTriangleInterior v2 v5 v1 v3 v4 ∧
       intersectsTriangleInterior v3 v5 v1 v2 v4) :=
by sorry


end NUMINAMATH_CALUDE_no_special_pentagon_l269_26949


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l269_26947

-- Problem 1
theorem problem_1 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 := by sorry

-- Problem 2
theorem problem_2 (p q : ℝ) : (-p*q)^3 = -p^3 * q^3 := by sorry

-- Problem 3
theorem problem_3 (a : ℝ) : a^3 * a^4 * a + (a^2)^4 - (-2*a^4)^2 = -2 * a^8 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l269_26947


namespace NUMINAMATH_CALUDE_inverse_f_at_407_l269_26958

noncomputable def f (x : ℝ) : ℝ := 5 * x^4 + 2

theorem inverse_f_at_407 : Function.invFun f 407 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_407_l269_26958


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_6_range_of_m_for_f_geq_m_squared_minus_3m_l269_26901

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |2*x - 4|

-- Theorem for the solution set of f(x) < 6
theorem solution_set_f_less_than_6 :
  {x : ℝ | f x < 6} = {x : ℝ | 0 < x ∧ x < 8/3} := by sorry

-- Theorem for the range of m
theorem range_of_m_for_f_geq_m_squared_minus_3m :
  {m : ℝ | ∀ x, f x ≥ m^2 - 3*m} = {m : ℝ | -1 ≤ m ∧ m ≤ 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_6_range_of_m_for_f_geq_m_squared_minus_3m_l269_26901


namespace NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l269_26919

/-- The ratio of the area of a square inscribed in a quarter-circle to the area of a square inscribed in a full circle, both with radius r, is 1/4. -/
theorem inscribed_squares_area_ratio (r : ℝ) (hr : r > 0) :
  let s1 := r / Real.sqrt 2
  let s2 := r * Real.sqrt 2
  (s1 ^ 2) / (s2 ^ 2) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l269_26919


namespace NUMINAMATH_CALUDE_employee_reduction_percentage_l269_26970

def original_employees : ℝ := 227
def reduced_employees : ℝ := 195

theorem employee_reduction_percentage : 
  let difference := original_employees - reduced_employees
  let percentage := (difference / original_employees) * 100
  abs (percentage - 14.1) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_employee_reduction_percentage_l269_26970


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l269_26967

theorem lcm_gcd_product (a b : ℕ) (h1 : a = 15) (h2 : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 135 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l269_26967


namespace NUMINAMATH_CALUDE_special_triangle_is_equilateral_l269_26989

/-- A triangle with sides in geometric progression and angles in arithmetic progression -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  q : ℝ
  -- Angles of the triangle
  α : ℝ
  δ : ℝ
  -- Side lengths form a geometric progression
  side_gp : q > 0
  -- Angles form an arithmetic progression
  angle_ap : True
  -- Sum of angles is 180 degrees
  angle_sum : α - δ + α + (α + δ) = 180

/-- The theorem stating that a SpecialTriangle must be equilateral -/
theorem special_triangle_is_equilateral (t : SpecialTriangle) : t.q = 1 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_is_equilateral_l269_26989


namespace NUMINAMATH_CALUDE_container_production_l269_26988

/-- Container production problem -/
theorem container_production
  (december : ℕ)
  (h_dec_nov : december = (110 * november) / 100)
  (h_nov_oct : november = (105 * october) / 100)
  (h_oct_sep : october = (120 * september) / 100)
  (h_december : december = 11088) :
  november = 10080 ∧ october = 9600 ∧ september = 8000 := by
  sorry

end NUMINAMATH_CALUDE_container_production_l269_26988


namespace NUMINAMATH_CALUDE_negative_square_of_two_l269_26950

theorem negative_square_of_two : -2^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_of_two_l269_26950


namespace NUMINAMATH_CALUDE_correct_multiplier_problem_solution_l269_26973

theorem correct_multiplier (number_to_multiply : ℕ) (mistaken_multiplier : ℕ) (difference : ℕ) : ℕ :=
  let correct_multiplier := (mistaken_multiplier * number_to_multiply + difference) / number_to_multiply
  correct_multiplier

theorem problem_solution :
  correct_multiplier 135 34 1215 = 43 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplier_problem_solution_l269_26973


namespace NUMINAMATH_CALUDE_probability_two_green_marbles_l269_26984

def red_marbles : ℕ := 3
def green_marbles : ℕ := 4
def white_marbles : ℕ := 13

def total_marbles : ℕ := red_marbles + green_marbles + white_marbles

theorem probability_two_green_marbles :
  (green_marbles / total_marbles) * ((green_marbles - 1) / (total_marbles - 1)) = 3 / 95 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_green_marbles_l269_26984


namespace NUMINAMATH_CALUDE_correct_selling_price_l269_26910

/-- The markup percentage applied to the cost price -/
def markup : ℚ := 25 / 100

/-- The cost price of the computer table in rupees -/
def cost_price : ℕ := 6672

/-- The selling price of the computer table in rupees -/
def selling_price : ℕ := 8340

/-- Theorem stating that the selling price is correct given the cost price and markup -/
theorem correct_selling_price : 
  (cost_price : ℚ) * (1 + markup) = selling_price := by sorry

end NUMINAMATH_CALUDE_correct_selling_price_l269_26910


namespace NUMINAMATH_CALUDE_paper_used_l269_26990

theorem paper_used (initial : ℕ) (remaining : ℕ) (used : ℕ) 
  (h1 : initial = 900) 
  (h2 : remaining = 744) 
  (h3 : used = initial - remaining) : used = 156 := by
  sorry

end NUMINAMATH_CALUDE_paper_used_l269_26990


namespace NUMINAMATH_CALUDE_evelyn_bottle_caps_l269_26955

/-- The number of bottle caps Evelyn ends with after losing some -/
def bottle_caps_remaining (initial : Float) (lost : Float) : Float :=
  initial - lost

/-- Theorem: If Evelyn starts with 63.0 bottle caps and loses 18.0, she ends with 45.0 -/
theorem evelyn_bottle_caps : bottle_caps_remaining 63.0 18.0 = 45.0 := by
  sorry

end NUMINAMATH_CALUDE_evelyn_bottle_caps_l269_26955


namespace NUMINAMATH_CALUDE_top_price_calculation_l269_26981

def shorts_price : ℝ := 7
def shoes_price : ℝ := 10
def hats_price : ℝ := 6
def socks_price : ℝ := 2

def shorts_quantity : ℕ := 5
def shoes_quantity : ℕ := 2
def hats_quantity : ℕ := 3
def socks_quantity : ℕ := 6
def tops_quantity : ℕ := 4

def total_spent : ℝ := 102

theorem top_price_calculation :
  let other_items_cost := shorts_price * shorts_quantity + shoes_price * shoes_quantity +
                          hats_price * hats_quantity + socks_price * socks_quantity
  let tops_total_cost := total_spent - other_items_cost
  tops_total_cost / tops_quantity = 4.25 := by sorry

end NUMINAMATH_CALUDE_top_price_calculation_l269_26981


namespace NUMINAMATH_CALUDE_garden_area_increase_l269_26982

/-- Given a rectangular garden with length 60 feet and width 20 feet,
    prove that changing it to a square garden with the same perimeter
    increases the area by 400 square feet. -/
theorem garden_area_increase :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter := 2 * (rect_length + rect_width)
  let square_side := rect_perimeter / 4
  let rect_area := rect_length * rect_width
  let square_area := square_side * square_side
  square_area - rect_area = 400 := by sorry

end NUMINAMATH_CALUDE_garden_area_increase_l269_26982


namespace NUMINAMATH_CALUDE_elephants_after_three_years_is_zero_l269_26999

/-- Represents the different types of animals in the zoo -/
inductive Animal
| Giraffe
| Penguin
| Elephant
| Lion
| Bear

/-- Represents the state of the zoo -/
structure ZooState where
  animalCount : Animal → ℕ
  budget : ℕ

/-- The cost of each animal type -/
def animalCost : Animal → ℕ
| Animal.Giraffe => 1000
| Animal.Penguin => 500
| Animal.Elephant => 1200
| Animal.Lion => 1100
| Animal.Bear => 1300

/-- The initial state of the zoo -/
def initialState : ZooState :=
  { animalCount := λ a => match a with
      | Animal.Giraffe => 5
      | Animal.Penguin => 10
      | Animal.Elephant => 0
      | Animal.Lion => 5
      | Animal.Bear => 0
    budget := 10000 }

/-- The maximum capacity of the zoo -/
def maxCapacity : ℕ := 300

/-- Theorem stating that the number of elephants after three years is zero -/
theorem elephants_after_three_years_is_zero :
  (initialState.animalCount Animal.Elephant) = 0 → 
  ∀ (finalState : ZooState),
    (finalState.animalCount Animal.Elephant) = 0 := by
  sorry

#check elephants_after_three_years_is_zero

end NUMINAMATH_CALUDE_elephants_after_three_years_is_zero_l269_26999


namespace NUMINAMATH_CALUDE_fraction_comparison_and_absolute_value_inequality_l269_26962

theorem fraction_comparison_and_absolute_value_inequality :
  (-3 : ℚ) / 7 < (-2 : ℚ) / 5 ∧
  ∃ (a b : ℚ), |a + b| ≠ |a| + |b| :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_and_absolute_value_inequality_l269_26962


namespace NUMINAMATH_CALUDE_wedding_chairs_l269_26992

theorem wedding_chairs (rows : ℕ) (chairs_per_row : ℕ) (extra_chairs : ℕ) : 
  rows = 7 → chairs_per_row = 12 → extra_chairs = 11 → 
  rows * chairs_per_row + extra_chairs = 95 := by
sorry

end NUMINAMATH_CALUDE_wedding_chairs_l269_26992


namespace NUMINAMATH_CALUDE_function_characterization_l269_26954

def f (x a b : ℝ) : ℝ := (x + a) * (b * x + 2 * a)

theorem function_characterization (a b : ℝ) :
  (∀ x, f x a b = f (-x) a b) →  -- f is even
  (∀ y, y ∈ Set.Iic 4 → ∃ x, f x a b = y) →  -- range is (-∞, 4]
  (∀ y, ∃ x, f x a b = y → y ≤ 4) →  -- range is (-∞, 4]
  (∀ x, f x a b = -2 * x^2 + 4) :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l269_26954


namespace NUMINAMATH_CALUDE_chef_potato_problem_chef_leftover_potatoes_l269_26987

/-- A chef's potato problem -/
theorem chef_potato_problem (total_potatoes : ℕ) 
  (fries_needed : ℕ) (fries_per_potato : ℕ) 
  (cubes_needed : ℕ) (cubes_per_potato : ℕ) : ℕ :=
  let potatoes_for_fries := (fries_needed + fries_per_potato - 1) / fries_per_potato
  let potatoes_for_cubes := (cubes_needed + cubes_per_potato - 1) / cubes_per_potato
  let potatoes_used := potatoes_for_fries + potatoes_for_cubes
  total_potatoes - potatoes_used

/-- The chef will have 17 potatoes leftover -/
theorem chef_leftover_potatoes : 
  chef_potato_problem 30 200 25 50 10 = 17 := by
  sorry

end NUMINAMATH_CALUDE_chef_potato_problem_chef_leftover_potatoes_l269_26987


namespace NUMINAMATH_CALUDE_divisibility_condition_l269_26944

theorem divisibility_condition (n : ℤ) : 
  (n + 2) ∣ (n^2 + 3) ↔ n ∈ ({-9, -3, -1, 5} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l269_26944


namespace NUMINAMATH_CALUDE_boys_in_class_l269_26998

theorem boys_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h1 : total = 56) (h2 : ratio_girls = 4) (h3 : ratio_boys = 3) : 
  (total * ratio_boys) / (ratio_girls + ratio_boys) = 24 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l269_26998


namespace NUMINAMATH_CALUDE_lawn_width_is_60_l269_26972

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  roadWidth : ℝ
  costPerSqm : ℝ
  totalCost : ℝ

/-- Calculates the total area of the roads -/
def roadArea (l : LawnWithRoads) : ℝ :=
  l.roadWidth * l.length + l.roadWidth * l.width - l.roadWidth * l.roadWidth

/-- Theorem: Given the specifications, the width of the lawn is 60 meters -/
theorem lawn_width_is_60 (l : LawnWithRoads) 
    (h1 : l.length = 90)
    (h2 : l.roadWidth = 10)
    (h3 : l.costPerSqm = 3)
    (h4 : l.totalCost = 4200)
    (h5 : l.totalCost = l.costPerSqm * roadArea l) : 
  l.width = 60 := by
  sorry

end NUMINAMATH_CALUDE_lawn_width_is_60_l269_26972


namespace NUMINAMATH_CALUDE_john_needs_two_planks_l269_26904

/-- The number of planks needed for a house wall --/
def planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) : ℕ :=
  total_nails / nails_per_plank

/-- Theorem: John needs 2 planks for the house wall --/
theorem john_needs_two_planks :
  planks_needed 4 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_two_planks_l269_26904


namespace NUMINAMATH_CALUDE_tshirt_pricing_l269_26952

def first_batch_cost : ℝ := 4000
def second_batch_cost : ℝ := 8800
def cost_difference : ℝ := 4
def discounted_quantity : ℕ := 40
def discount_rate : ℝ := 0.3
def min_profit_margin : ℝ := 0.8

def cost_price_first_batch : ℝ := 40
def cost_price_second_batch : ℝ := 44
def min_retail_price : ℝ := 80

theorem tshirt_pricing :
  let first_quantity := first_batch_cost / cost_price_first_batch
  let second_quantity := second_batch_cost / cost_price_second_batch
  let total_quantity := first_quantity + second_quantity
  (2 * first_quantity = second_quantity) ∧
  (cost_price_second_batch = cost_price_first_batch + cost_difference) ∧
  (min_retail_price * (total_quantity - discounted_quantity) +
   min_retail_price * (1 - discount_rate) * discounted_quantity ≥
   (first_batch_cost + second_batch_cost) * (1 + min_profit_margin)) :=
by sorry

end NUMINAMATH_CALUDE_tshirt_pricing_l269_26952


namespace NUMINAMATH_CALUDE_chess_tournament_games_l269_26957

/-- The number of unique games in a chess tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group with 7 players, where each player plays every other player once,
    the total number of games played is 21. -/
theorem chess_tournament_games :
  num_games 7 = 21 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l269_26957


namespace NUMINAMATH_CALUDE_some_trinks_not_zorbs_l269_26980

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for Zorb, Glarb, and Trink
variable (Zorb Glarb Trink : U → Prop)

-- Hypothesis I: All Zorbs are not Glarbs
variable (h1 : ∀ x, Zorb x → ¬Glarb x)

-- Hypothesis II: Some Glarbs are Trinks
variable (h2 : ∃ x, Glarb x ∧ Trink x)

-- Theorem: Some Trinks are not Zorbs
theorem some_trinks_not_zorbs :
  ∃ x, Trink x ∧ ¬Zorb x :=
sorry

end NUMINAMATH_CALUDE_some_trinks_not_zorbs_l269_26980


namespace NUMINAMATH_CALUDE_exam_score_problem_l269_26942

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 140 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 40 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l269_26942


namespace NUMINAMATH_CALUDE_intersection_M_N_l269_26909

def M : Set ℤ := {m : ℤ | -3 < m ∧ m < 2}
def N : Set ℤ := {n : ℤ | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l269_26909


namespace NUMINAMATH_CALUDE_pie_crust_flour_usage_l269_26934

theorem pie_crust_flour_usage 
  (original_crusts : ℕ) 
  (original_flour_per_crust : ℚ) 
  (new_crusts : ℕ) :
  original_crusts = 30 →
  original_flour_per_crust = 1/6 →
  new_crusts = 25 →
  (original_crusts * original_flour_per_crust) / new_crusts = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_pie_crust_flour_usage_l269_26934


namespace NUMINAMATH_CALUDE_ln_inequality_implies_inequality_l269_26918

theorem ln_inequality_implies_inequality (a b : ℝ) : 
  Real.log a > Real.log b → a > b := by sorry

end NUMINAMATH_CALUDE_ln_inequality_implies_inequality_l269_26918


namespace NUMINAMATH_CALUDE_straight_angle_average_l269_26906

theorem straight_angle_average (p q r s t : ℝ) : 
  p + q + r + s + t = 180 → (p + q + r + s + t) / 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_straight_angle_average_l269_26906


namespace NUMINAMATH_CALUDE_password_has_14_characters_l269_26917

/- Define the components of the password -/
def lowercase_length : ℕ := 8
def uppercase_number_length : ℕ := lowercase_length / 2
def symbol_count : ℕ := 2

/- Define the total password length -/
def password_length : ℕ := lowercase_length + uppercase_number_length + symbol_count

/- Theorem statement -/
theorem password_has_14_characters : password_length = 14 := by
  sorry

end NUMINAMATH_CALUDE_password_has_14_characters_l269_26917


namespace NUMINAMATH_CALUDE_fiftyFourthCardIsSpadeTwo_l269_26946

/-- Represents a playing card suit -/
inductive Suit
| Spades
| Hearts

/-- Represents a playing card value -/
inductive Value
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents a playing card -/
structure Card where
  suit : Suit
  value : Value

/-- The sequence of cards in order -/
def cardSequence : List Card := sorry

/-- The length of one complete cycle in the sequence -/
def cycleLength : Nat := 26

/-- Function to get the nth card in the sequence -/
def getNthCard (n : Nat) : Card := sorry

theorem fiftyFourthCardIsSpadeTwo : 
  getNthCard 54 = Card.mk Suit.Spades Value.Two := by sorry

end NUMINAMATH_CALUDE_fiftyFourthCardIsSpadeTwo_l269_26946


namespace NUMINAMATH_CALUDE_rotation_sum_110_l269_26960

/-- A structure representing a triangle in a 2D coordinate plane -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The rotation parameters -/
structure RotationParams where
  n : ℝ
  u : ℝ
  v : ℝ

/-- Predicate to check if a rotation transforms one triangle to another -/
def rotates (t1 t2 : Triangle) (r : RotationParams) : Prop :=
  sorry  -- Definition of rotation transformation

theorem rotation_sum_110 (DEF D'E'F' : Triangle) (r : RotationParams) :
  DEF.D = (0, 0) →
  DEF.E = (0, 10) →
  DEF.F = (20, 0) →
  D'E'F'.D = (30, 20) →
  D'E'F'.E = (40, 20) →
  D'E'F'.F = (30, 6) →
  0 < r.n →
  r.n < 180 →
  rotates DEF D'E'F' r →
  r.n + r.u + r.v = 110 := by
  sorry

#check rotation_sum_110

end NUMINAMATH_CALUDE_rotation_sum_110_l269_26960


namespace NUMINAMATH_CALUDE_equation_system_equivalent_quadratic_l269_26969

theorem equation_system_equivalent_quadratic (x y : ℝ) :
  (3 * x^2 + 4 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 4 = 0) →
  4 * y^2 + 29 * y + 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_system_equivalent_quadratic_l269_26969


namespace NUMINAMATH_CALUDE_sculpture_cost_in_inr_l269_26964

/-- Exchange rate between US dollars and Namibian dollars -/
def usd_to_nad : ℝ := 10

/-- Exchange rate between US dollars and Chinese yuan -/
def usd_to_cny : ℝ := 7

/-- Exchange rate between Chinese yuan and Indian Rupees -/
def cny_to_inr : ℝ := 10

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℝ := 200

/-- Theorem stating the cost of the sculpture in Indian Rupees -/
theorem sculpture_cost_in_inr :
  (sculpture_cost_nad / usd_to_nad) * usd_to_cny * cny_to_inr = 1400 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_inr_l269_26964


namespace NUMINAMATH_CALUDE_generate_numbers_l269_26925

/-- A type representing arithmetic expressions using five 3's -/
inductive Expr
  | three : Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr
  | pow : Expr → Expr → Expr

/-- Evaluate an expression to a rational number -/
def eval : Expr → ℚ
  | Expr.three => 3
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2
  | Expr.pow e1 e2 => (eval e1) ^ (eval e2).num

/-- Count the number of 3's used in an expression -/
def count_threes : Expr → Nat
  | Expr.three => 1
  | Expr.add e1 e2 => count_threes e1 + count_threes e2
  | Expr.sub e1 e2 => count_threes e1 + count_threes e2
  | Expr.mul e1 e2 => count_threes e1 + count_threes e2
  | Expr.div e1 e2 => count_threes e1 + count_threes e2
  | Expr.pow e1 e2 => count_threes e1 + count_threes e2

/-- The main theorem stating that all integers from 1 to 39 can be generated -/
theorem generate_numbers :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 39 →
  ∃ e : Expr, count_threes e = 5 ∧ eval e = n := by sorry

end NUMINAMATH_CALUDE_generate_numbers_l269_26925


namespace NUMINAMATH_CALUDE_horseback_riding_distance_l269_26905

/-- Calculates the total distance traveled during a 3-day horseback riding trip -/
theorem horseback_riding_distance : 
  let day1_speed : ℝ := 5
  let day1_time : ℝ := 7
  let day2_speed1 : ℝ := 6
  let day2_time1 : ℝ := 6
  let day2_speed2 : ℝ := day2_speed1 / 2
  let day2_time2 : ℝ := 3
  let day3_speed : ℝ := 7
  let day3_time : ℝ := 5
  let total_distance : ℝ := 
    day1_speed * day1_time + 
    day2_speed1 * day2_time1 + day2_speed2 * day2_time2 + 
    day3_speed * day3_time
  total_distance = 115 := by
  sorry


end NUMINAMATH_CALUDE_horseback_riding_distance_l269_26905


namespace NUMINAMATH_CALUDE_plane_intersection_properties_l269_26920

-- Define the planes
variable (α β γ : Set (ℝ × ℝ × ℝ))

-- Define the perpendicularity and intersection relations
def perpendicular (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def intersects (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def intersects_not_perpendicularly (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define what it means for a line to be in a plane
def line_in_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define parallel and perpendicular for lines and planes
def line_parallel_to_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def line_perpendicular_to_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- State the theorem
theorem plane_intersection_properties 
  (h1 : perpendicular β γ)
  (h2 : intersects_not_perpendicularly α γ) :
  (∃ (a : Set (ℝ × ℝ × ℝ)), line_in_plane a α ∧ line_parallel_to_plane a γ) ∧
  (∃ (c : Set (ℝ × ℝ × ℝ)), line_in_plane c γ ∧ line_perpendicular_to_plane c β) := by
  sorry

end NUMINAMATH_CALUDE_plane_intersection_properties_l269_26920


namespace NUMINAMATH_CALUDE_hamburger_count_l269_26945

/-- The total number of hamburgers made for lunch -/
def total_hamburgers (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the total number of hamburgers is the sum of initial and additional -/
theorem hamburger_count (initial : ℕ) (additional : ℕ) :
  total_hamburgers initial additional = initial + additional :=
by sorry

end NUMINAMATH_CALUDE_hamburger_count_l269_26945


namespace NUMINAMATH_CALUDE_items_sold_l269_26979

/-- Given the following conditions:
  1. A grocery store ordered 4458 items to restock.
  2. They have 575 items in the storeroom.
  3. They have 3,472 items left in the whole store.
  Prove that the number of items sold that day is 1561. -/
theorem items_sold (restocked : ℕ) (in_storeroom : ℕ) (left_in_store : ℕ) 
  (h1 : restocked = 4458)
  (h2 : in_storeroom = 575)
  (h3 : left_in_store = 3472) :
  restocked + in_storeroom - left_in_store = 1561 :=
by sorry

end NUMINAMATH_CALUDE_items_sold_l269_26979
