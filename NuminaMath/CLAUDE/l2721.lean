import Mathlib

namespace NUMINAMATH_CALUDE_friend_score_l2721_272107

theorem friend_score (edward_score : ℕ) (total_score : ℕ) (friend_score : ℕ) : 
  edward_score = 7 → 
  total_score = 13 → 
  total_score = edward_score + friend_score →
  friend_score = 6 := by
sorry

end NUMINAMATH_CALUDE_friend_score_l2721_272107


namespace NUMINAMATH_CALUDE_geometric_solid_height_l2721_272174

-- Define the geometric solid
structure GeometricSolid where
  radius1 : ℝ
  radius2 : ℝ
  water_height1 : ℝ
  water_height2 : ℝ

-- Define the theorem
theorem geometric_solid_height (s : GeometricSolid) 
  (h1 : s.radius1 = 1)
  (h2 : s.radius2 = 3)
  (h3 : s.water_height1 = 20)
  (h4 : s.water_height2 = 28) :
  ∃ (total_height : ℝ), total_height = 29 := by
  sorry

end NUMINAMATH_CALUDE_geometric_solid_height_l2721_272174


namespace NUMINAMATH_CALUDE_factorization_identity_l2721_272184

theorem factorization_identity (a b : ℝ) : a^2 + a*b = a*(a + b) := by sorry

end NUMINAMATH_CALUDE_factorization_identity_l2721_272184


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l2721_272178

theorem sufficient_condition_for_inequality (a : ℝ) (h : a > 4) :
  ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l2721_272178


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_left_l2721_272194

/-- Given a series with a total number of books and a number of books read,
    calculate the number of books left to read. -/
def booksLeftToRead (totalBooks readBooks : ℕ) : ℕ :=
  totalBooks - readBooks

/-- Theorem: In a series with 19 books, if 4 books have been read,
    then the number of books left to read is 15. -/
theorem crazy_silly_school_books_left :
  booksLeftToRead 19 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_left_l2721_272194


namespace NUMINAMATH_CALUDE_xiao_ming_run_distance_l2721_272163

/-- The distance between two adjacent trees in meters -/
def tree_spacing : ℕ := 6

/-- The number of the last tree Xiao Ming runs to -/
def last_tree : ℕ := 200

/-- The total distance Xiao Ming runs in meters -/
def total_distance : ℕ := (last_tree - 1) * tree_spacing

theorem xiao_ming_run_distance :
  total_distance = 1194 :=
sorry

end NUMINAMATH_CALUDE_xiao_ming_run_distance_l2721_272163


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l2721_272129

/-- Represents the ages of John and Mary -/
structure Ages where
  john : ℕ
  mary : ℕ

/-- The conditions from the problem -/
def age_conditions (a : Ages) : Prop :=
  (a.john - 3 = 2 * (a.mary - 3)) ∧ 
  (a.john - 7 = 3 * (a.mary - 7))

/-- The future condition we're looking for -/
def future_ratio (a : Ages) (years : ℕ) : Prop :=
  3 * (a.mary + years) = 2 * (a.john + years)

/-- The main theorem -/
theorem age_ratio_theorem (a : Ages) :
  age_conditions a → ∃ y : ℕ, y = 5 ∧ future_ratio a y := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_theorem_l2721_272129


namespace NUMINAMATH_CALUDE_pentagon_count_l2721_272106

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct points on the circle -/
def num_points : ℕ := 15

/-- The number of vertices in a pentagon -/
def pentagon_vertices : ℕ := 5

/-- Theorem: The number of different convex pentagons that can be formed
    by selecting 5 points from 15 distinct points on the circumference of a circle
    is equal to 3003 -/
theorem pentagon_count :
  binomial num_points pentagon_vertices = 3003 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_count_l2721_272106


namespace NUMINAMATH_CALUDE_slope_range_for_line_l2721_272168

/-- Given a line passing through (1, 1) with y-intercept in (0, 2), its slope is in (-1, 1) -/
theorem slope_range_for_line (l : Set (ℝ × ℝ)) (y_intercept : ℝ) (k : ℝ) : 
  (∀ x y, (x, y) ∈ l ↔ y = k * x + (1 - k)) →  -- Line equation
  (1, 1) ∈ l →  -- Line passes through (1, 1)
  0 < y_intercept ∧ y_intercept < 2 →  -- y-intercept in (0, 2)
  y_intercept = 1 - k →  -- y-intercept calculation
  -1 < k ∧ k < 1 :=  -- Slope is in (-1, 1)
by sorry

end NUMINAMATH_CALUDE_slope_range_for_line_l2721_272168


namespace NUMINAMATH_CALUDE_det_transformation_l2721_272111

/-- Given a 2x2 matrix with determinant -3, prove that a specific transformation of this matrix also has determinant -3 -/
theorem det_transformation (x y z w : ℝ) 
  (h : Matrix.det !![x, y; z, w] = -3) :
  Matrix.det !![x + 2*z, y + 2*w; z, w] = -3 := by
sorry

end NUMINAMATH_CALUDE_det_transformation_l2721_272111


namespace NUMINAMATH_CALUDE_boxes_given_away_l2721_272177

def total_cupcakes : ℕ := 53
def cupcakes_left_at_home : ℕ := 2
def cupcakes_per_box : ℕ := 3

theorem boxes_given_away : 
  (total_cupcakes - cupcakes_left_at_home) / cupcakes_per_box = 17 := by
  sorry

end NUMINAMATH_CALUDE_boxes_given_away_l2721_272177


namespace NUMINAMATH_CALUDE_largest_integer_under_sqrt_constraint_l2721_272143

theorem largest_integer_under_sqrt_constraint : 
  ∀ x : ℤ, (Real.sqrt (x^2 : ℝ) < 15) → x ≤ 14 ∧ ∃ y : ℤ, y > x ∧ ¬(Real.sqrt (y^2 : ℝ) < 15) :=
sorry

end NUMINAMATH_CALUDE_largest_integer_under_sqrt_constraint_l2721_272143


namespace NUMINAMATH_CALUDE_uv_length_in_triangle_l2721_272188

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)
  (xy_length : Real)
  (xz_length : Real)
  (yz_length : Real)

-- Define the angle bisector points S and T
structure AngleBisectorPoints :=
  (S : ℝ × ℝ)
  (T : ℝ × ℝ)

-- Define the perpendicular feet U and V
structure PerpendicularFeet :=
  (U : ℝ × ℝ)
  (V : ℝ × ℝ)

-- Define the theorem
theorem uv_length_in_triangle (t : Triangle) (ab : AngleBisectorPoints) (pf : PerpendicularFeet) :
  t.xy_length = 140 ∧ t.xz_length = 130 ∧ t.yz_length = 150 →
  -- S is on the angle bisector of angle X and YZ
  -- T is on the angle bisector of angle Y and XZ
  -- U is the foot of the perpendicular from Z to YT
  -- V is the foot of the perpendicular from Z to XS
  Real.sqrt ((pf.U.1 - pf.V.1)^2 + (pf.U.2 - pf.V.2)^2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_uv_length_in_triangle_l2721_272188


namespace NUMINAMATH_CALUDE_melissa_bought_four_packs_l2721_272140

/-- The number of packs of tennis balls Melissa bought -/
def num_packs : ℕ := sorry

/-- The total cost of all packs in dollars -/
def total_cost : ℕ := 24

/-- The number of balls in each pack -/
def balls_per_pack : ℕ := 3

/-- The cost of each ball in dollars -/
def cost_per_ball : ℕ := 2

/-- Theorem stating that Melissa bought 4 packs of tennis balls -/
theorem melissa_bought_four_packs : num_packs = 4 := by sorry

end NUMINAMATH_CALUDE_melissa_bought_four_packs_l2721_272140


namespace NUMINAMATH_CALUDE_even_quadratic_sum_l2721_272193

/-- A function f is even on an interval if f(-x) = f(x) for all x in the interval -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x, x ∈ Set.Icc a b → f (-x) = f x

/-- The main theorem -/
theorem even_quadratic_sum (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  let interval := Set.Icc (-2 * a - 5) 1
  IsEvenOn f (-2 * a - 5) 1 → a + 2 * b = -2 := by
sorry

end NUMINAMATH_CALUDE_even_quadratic_sum_l2721_272193


namespace NUMINAMATH_CALUDE_remainder_987543_div_12_l2721_272161

theorem remainder_987543_div_12 : 987543 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_987543_div_12_l2721_272161


namespace NUMINAMATH_CALUDE_solve_for_b_l2721_272115

theorem solve_for_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * b) : b = 49 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l2721_272115


namespace NUMINAMATH_CALUDE_camel_cost_l2721_272189

/-- Proves that the cost of one camel is 5600 given the specified conditions --/
theorem camel_cost (camel horse ox elephant : ℕ → ℚ) 
  (h1 : 10 * camel 1 = 24 * horse 1)
  (h2 : 16 * horse 1 = 4 * ox 1)
  (h3 : 6 * ox 1 = 4 * elephant 1)
  (h4 : 10 * elephant 1 = 140000) : 
  camel 1 = 5600 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l2721_272189


namespace NUMINAMATH_CALUDE_rectangle_longest_side_l2721_272170

/-- Given a rectangle with perimeter 240 feet and area equal to 8 times its perimeter,
    the length of its longest side is 96 feet. -/
theorem rectangle_longest_side (l w : ℝ) : 
  l > 0 → w > 0 →
  2 * l + 2 * w = 240 →
  l * w = 8 * (2 * l + 2 * w) →
  max l w = 96 := by
sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l2721_272170


namespace NUMINAMATH_CALUDE_max_value_trig_function_l2721_272147

theorem max_value_trig_function :
  (∀ x : ℝ, 3 * Real.sin x - 3 * Real.sqrt 3 * Real.cos x ≤ 6) ∧
  (∃ x : ℝ, 3 * Real.sin x - 3 * Real.sqrt 3 * Real.cos x = 6) := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_function_l2721_272147


namespace NUMINAMATH_CALUDE_least_possible_third_side_length_l2721_272155

theorem least_possible_third_side_length (a b c : ℝ) : 
  a = 8 → b = 15 → c > 0 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c ≥ Real.sqrt 161 :=
by sorry

end NUMINAMATH_CALUDE_least_possible_third_side_length_l2721_272155


namespace NUMINAMATH_CALUDE_proportion_problem_l2721_272128

theorem proportion_problem (x y : ℝ) : 
  x / 5 = 5 / 6 → x = 0.9 → y / x = 5 / 6 → y = 0.75 := by sorry

end NUMINAMATH_CALUDE_proportion_problem_l2721_272128


namespace NUMINAMATH_CALUDE_candle_placement_impossibility_l2721_272125

theorem candle_placement_impossibility (n : ℕ) (d r : ℝ) (h_n : n = 13) (h_d : d = 10) (h_r : r = 18) :
  ¬ ∃ (points : Fin n → ℝ × ℝ),
    (∀ i, (points i).1^2 + (points i).2^2 = r^2) ∧
    (∀ i j, i ≠ j → Real.sqrt ((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2 ≥ d) :=
by sorry


end NUMINAMATH_CALUDE_candle_placement_impossibility_l2721_272125


namespace NUMINAMATH_CALUDE_line_equation_l2721_272196

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Point M -/
def M : ℝ × ℝ := (1, 1)

/-- A line passing through point (1, 1) -/
def line_through_M (m : ℝ) (x y : ℝ) : Prop := x = m * (y - 1) + 1

/-- The line intersects the ellipse at two points -/
def intersects_twice (m : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧
    ellipse (m * (y₁ - 1) + 1) y₁ ∧
    ellipse (m * (y₂ - 1) + 1) y₂

/-- M is the midpoint of the line segment AB -/
def M_is_midpoint (m : ℝ) : Prop :=
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧
    ellipse (m * (y₁ - 1) + 1) y₁ ∧
    ellipse (m * (y₂ - 1) + 1) y₂ ∧
    (y₁ + y₂) / 2 = 1

/-- The main theorem -/
theorem line_equation :
  ∃ m : ℝ, ellipse M.1 M.2 ∧
    intersects_twice m ∧
    M_is_midpoint m ∧
    ∀ x y : ℝ, line_through_M m x y ↔ 3 * x + 4 * y - 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l2721_272196


namespace NUMINAMATH_CALUDE_sheridan_cats_l2721_272134

/-- The number of cats Mrs. Sheridan has after giving some away -/
def remaining_cats (initial : Float) (given_away : Float) : Float :=
  initial - given_away

/-- Theorem: Mrs. Sheridan has 3.0 cats after giving away 14.0 cats from her initial 17.0 cats -/
theorem sheridan_cats : remaining_cats 17.0 14.0 = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_l2721_272134


namespace NUMINAMATH_CALUDE_profit_reached_l2721_272175

/-- The number of pencils bought for 6 dollars -/
def pencils_bought : ℕ := 5

/-- The cost in dollars for buying pencils_bought pencils -/
def cost : ℚ := 6

/-- The number of pencils sold for 7 dollars -/
def pencils_sold : ℕ := 4

/-- The revenue in dollars for selling pencils_sold pencils -/
def revenue : ℚ := 7

/-- The target profit in dollars -/
def target_profit : ℚ := 80

/-- The minimum number of pencils that must be sold to reach the target profit -/
def min_pencils_to_sell : ℕ := 146

theorem profit_reached : 
  ∃ (n : ℕ), n ≥ min_pencils_to_sell ∧ 
  n * (revenue / pencils_sold - cost / pencils_bought) ≥ target_profit ∧
  ∀ (m : ℕ), m < min_pencils_to_sell → 
  m * (revenue / pencils_sold - cost / pencils_bought) < target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_reached_l2721_272175


namespace NUMINAMATH_CALUDE_rectangle_area_l2721_272169

theorem rectangle_area (square_side : ℝ) (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ) : 
  square_side ^ 2 = 16 →
  circle_radius = square_side →
  rectangle_length = 5 * circle_radius →
  rectangle_breadth = 11 →
  rectangle_length * rectangle_breadth = 220 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2721_272169


namespace NUMINAMATH_CALUDE_parabola_sine_no_intersection_l2721_272114

theorem parabola_sine_no_intersection :
  ∀ x : ℝ, x^2 - x + 5.35 > 2 * Real.sin x + 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_sine_no_intersection_l2721_272114


namespace NUMINAMATH_CALUDE_triangle_existence_condition_l2721_272162

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_existence_condition (x : ℕ) :
  x > 0 →
  (triangle_exists 8 11 (2 * x + 1) ↔ x ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_condition_l2721_272162


namespace NUMINAMATH_CALUDE_gathering_attendance_l2721_272192

theorem gathering_attendance (wine soda both : ℕ) 
  (h1 : wine = 26) 
  (h2 : soda = 22) 
  (h3 : both = 17) : 
  wine + soda - both = 31 := by
  sorry

end NUMINAMATH_CALUDE_gathering_attendance_l2721_272192


namespace NUMINAMATH_CALUDE_square_difference_l2721_272132

theorem square_difference (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 4) : x^2 - y^2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2721_272132


namespace NUMINAMATH_CALUDE_circle_parameters_l2721_272148

/-- Definition of the circle C -/
def circle_equation (x y a b : ℝ) : Prop :=
  x^2 + y^2 + a*x - 2*y + b = 0

/-- Definition of a point being on the circle -/
def point_on_circle (x y a b : ℝ) : Prop :=
  circle_equation x y a b

/-- Definition of the line x + y - 1 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- Definition of symmetric point with respect to the line x + y - 1 = 0 -/
def symmetric_point (x y x' y' : ℝ) : Prop :=
  x' + y' = x + y ∧ line_equation ((x + x')/2) ((y + y')/2)

/-- Main theorem -/
theorem circle_parameters :
  ∀ (a b : ℝ),
    (point_on_circle 2 1 a b) →
    (∃ (x' y' : ℝ), symmetric_point 2 1 x' y' ∧ point_on_circle x' y' a b) →
    (a = 0 ∧ b = -3) :=
by sorry

end NUMINAMATH_CALUDE_circle_parameters_l2721_272148


namespace NUMINAMATH_CALUDE_rhombus_triangle_inscribed_circle_ratio_l2721_272119

/-- Given a rhombus ABCD with acute angle α and a triangle ABC formed by two sides of the rhombus
    and its longer diagonal, this theorem states that the ratio of the radius of the circle
    inscribed in the rhombus to the radius of the circle inscribed in the triangle ABC
    is equal to 1 + cos(α/2). -/
theorem rhombus_triangle_inscribed_circle_ratio (α : Real) 
  (h1 : 0 < α ∧ α < π / 2) : 
  ∃ (r1 r2 : Real), r1 > 0 ∧ r2 > 0 ∧
    (r1 / r2 = 1 + Real.cos (α / 2)) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_triangle_inscribed_circle_ratio_l2721_272119


namespace NUMINAMATH_CALUDE_decimal_fraction_equality_l2721_272108

theorem decimal_fraction_equality (b : ℕ+) : 
  (4 * b + 19 : ℚ) / (6 * b + 11) = 19 / 25 → b = 19 := by
  sorry

end NUMINAMATH_CALUDE_decimal_fraction_equality_l2721_272108


namespace NUMINAMATH_CALUDE_value_of_expression_l2721_272121

theorem value_of_expression (x : ℝ) (h : x^2 - 2*x = 1) : 
  2023 + 6*x - 3*x^2 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2721_272121


namespace NUMINAMATH_CALUDE_repetend_of_five_elevenths_l2721_272133

/-- The decimal representation of 5/11 has a repetend of 45. -/
theorem repetend_of_five_elevenths : ∃ (a b : ℕ), 
  (5 : ℚ) / 11 = (a : ℚ) / 100 + (b : ℚ) / 99 * (1 / 100) ∧ b = 45 := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_five_elevenths_l2721_272133


namespace NUMINAMATH_CALUDE_min_value_of_squared_sum_l2721_272150

theorem min_value_of_squared_sum (a b c t : ℝ) 
  (sum_condition : a + b + c = t) 
  (squared_sum_condition : a^2 + b^2 + c^2 = 1) : 
  2 * (a^2 + b^2 + c^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_squared_sum_l2721_272150


namespace NUMINAMATH_CALUDE_project_remaining_time_l2721_272103

/-- Given the time spent on various tasks of a project, proves that the remaining time for writing the report is 9 hours. -/
theorem project_remaining_time (total_time research_time proposal_time visual_aids_time editing_time rehearsal_time : ℕ)
  (h_total : total_time = 40)
  (h_research : research_time = 12)
  (h_proposal : proposal_time = 4)
  (h_visual : visual_aids_time = 7)
  (h_editing : editing_time = 5)
  (h_rehearsal : rehearsal_time = 3) :
  total_time - (research_time + proposal_time + visual_aids_time + editing_time + rehearsal_time) = 9 := by
  sorry

end NUMINAMATH_CALUDE_project_remaining_time_l2721_272103


namespace NUMINAMATH_CALUDE_second_derivative_of_cosine_at_pi_third_l2721_272179

open Real

theorem second_derivative_of_cosine_at_pi_third :
  let f : ℝ → ℝ := fun x ↦ cos x
  (deriv (deriv f)) (π / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_second_derivative_of_cosine_at_pi_third_l2721_272179


namespace NUMINAMATH_CALUDE_solution_xy_l2721_272180

theorem solution_xy : ∃ (x y : ℝ), 
  (x + 2*y = (7 - x) + (7 - 2*y)) ∧ 
  (x - y = 2*(x - 2) - (y - 3)) ∧ 
  (x = 1) ∧ (y = 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_xy_l2721_272180


namespace NUMINAMATH_CALUDE_P_in_second_quadrant_l2721_272165

-- Define the point P
def P (x : ℝ) : ℝ × ℝ := (-2, x^2 + 1)

-- Define what it means for a point to be in the second quadrant
def is_in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem stating that P is in the second quadrant for all real x
theorem P_in_second_quadrant (x : ℝ) : is_in_second_quadrant (P x) := by
  sorry


end NUMINAMATH_CALUDE_P_in_second_quadrant_l2721_272165


namespace NUMINAMATH_CALUDE_triangle_area_is_two_l2721_272123

/-- The first line bounding the triangle -/
def line1 (x y : ℝ) : Prop := y - 2*x = 4

/-- The second line bounding the triangle -/
def line2 (x y : ℝ) : Prop := 2*y - x = 6

/-- The x-axis -/
def x_axis (y : ℝ) : Prop := y = 0

/-- A point is in the triangle if it satisfies the equations of both lines and is above or on the x-axis -/
def in_triangle (x y : ℝ) : Prop :=
  line1 x y ∧ line2 x y ∧ y ≥ 0

/-- The area of the triangle -/
noncomputable def triangle_area : ℝ := 2

theorem triangle_area_is_two :
  triangle_area = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_two_l2721_272123


namespace NUMINAMATH_CALUDE_xyz_inequality_l2721_272181

theorem xyz_inequality (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hsum : x + y + z = 1) : 
  0 ≤ x*y + y*z + x*z - 2*x*y*z ∧ x*y + y*z + x*z - 2*x*y*z ≤ 7/27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2721_272181


namespace NUMINAMATH_CALUDE_book_reading_theorem_l2721_272151

def days_to_read (n : ℕ) : ℕ := n * (n + 1) / 2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem book_reading_theorem :
  let num_books := 18
  let start_day := 0  -- 0 represents Sunday
  let total_days := days_to_read num_books
  day_of_week start_day total_days = 3  -- 3 represents Wednesday
:= by sorry

end NUMINAMATH_CALUDE_book_reading_theorem_l2721_272151


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2721_272141

/-- The perimeter of a rectangle with length 0.54 meters and width 0.08 meters shorter than the length is 2 meters. -/
theorem rectangle_perimeter : 
  let length : ℝ := 0.54
  let width_difference : ℝ := 0.08
  let width : ℝ := length - width_difference
  let perimeter : ℝ := 2 * (length + width)
  perimeter = 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2721_272141


namespace NUMINAMATH_CALUDE_b_joined_after_five_months_l2721_272139

/-- Represents the number of months after A started the business that B joined as a partner. -/
def months_before_b_joined : ℕ := 5

/-- Represents A's initial investment in rupees. -/
def a_investment : ℕ := 3500

/-- Represents B's investment in rupees. -/
def b_investment : ℕ := 9000

/-- Represents the total number of months in a year. -/
def months_in_year : ℕ := 12

/-- Theorem stating that B joined the business 5 months after A started, given the conditions. -/
theorem b_joined_after_five_months :
  let a_capital := a_investment * months_in_year
  let b_capital := b_investment * (months_in_year - months_before_b_joined)
  (a_capital : ℚ) / b_capital = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_b_joined_after_five_months_l2721_272139


namespace NUMINAMATH_CALUDE_profit_percentage_is_50_percent_l2721_272190

/-- Calculates the profit percentage given the costs and selling price -/
def profit_percentage (purchase_price repair_cost transport_cost selling_price : ℕ) : ℚ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := selling_price - total_cost
  (profit : ℚ) / (total_cost : ℚ) * 100

/-- Theorem stating that the profit percentage for the given scenario is 50% -/
theorem profit_percentage_is_50_percent :
  profit_percentage 11000 5000 1000 25500 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_50_percent_l2721_272190


namespace NUMINAMATH_CALUDE_average_score_calculation_l2721_272142

/-- Calculates the average score of all students given the proportion of male students,
    the average score of male students, and the average score of female students. -/
def average_score (male_proportion : ℝ) (male_avg : ℝ) (female_avg : ℝ) : ℝ :=
  male_proportion * male_avg + (1 - male_proportion) * female_avg

/-- Theorem stating that when 40% of students are male, with male average score 75
    and female average score 80, the overall average score is 78. -/
theorem average_score_calculation :
  average_score 0.4 75 80 = 78 := by
  sorry

end NUMINAMATH_CALUDE_average_score_calculation_l2721_272142


namespace NUMINAMATH_CALUDE_right_triangle_side_lengths_l2721_272131

theorem right_triangle_side_lengths (a : ℝ) : 
  (∃ (x y z : ℝ), x = a + 1 ∧ y = a + 2 ∧ z = a + 3 ∧ 
  x^2 + y^2 = z^2) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_side_lengths_l2721_272131


namespace NUMINAMATH_CALUDE_repair_cost_is_13000_l2721_272199

/-- Calculates the repair cost given the purchase price, selling price, and profit percentage --/
def calculate_repair_cost (purchase_price selling_price profit_percent : ℚ) : ℚ :=
  let total_cost := selling_price / (1 + profit_percent / 100)
  total_cost - purchase_price

/-- Theorem stating that the repair cost is 13000 given the problem conditions --/
theorem repair_cost_is_13000 :
  let purchase_price : ℚ := 42000
  let selling_price : ℚ := 60900
  let profit_percent : ℚ := 10.727272727272727
  calculate_repair_cost purchase_price selling_price profit_percent = 13000 := by
  sorry


end NUMINAMATH_CALUDE_repair_cost_is_13000_l2721_272199


namespace NUMINAMATH_CALUDE_prism_volume_l2721_272197

/-- A right rectangular prism with face areas 18, 32, and 48 square inches has a volume of 288 cubic inches. -/
theorem prism_volume (l w h : ℝ) 
  (area1 : l * w = 18) 
  (area2 : w * h = 32) 
  (area3 : l * h = 48) : 
  l * w * h = 288 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2721_272197


namespace NUMINAMATH_CALUDE_endpoint_sum_l2721_272122

/-- Given a line segment with one endpoint (1, -2) and midpoint (5, 4),
    the sum of coordinates of the other endpoint is 19. -/
theorem endpoint_sum (x y : ℝ) : 
  (1 + x) / 2 = 5 ∧ (-2 + y) / 2 = 4 → x + y = 19 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_l2721_272122


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2721_272113

/-- The polynomial x^4 - 6x^3 + 16x^2 - 25x + 10 -/
def P (x : ℝ) : ℝ := x^4 - 6*x^3 + 16*x^2 - 25*x + 10

/-- The divisor x^2 - 2x + k -/
def D (x k : ℝ) : ℝ := x^2 - 2*x + k

/-- The remainder x + a -/
def R (x a : ℝ) : ℝ := x + a

/-- There exist q such that P(x) = D(x, k) * q(x) + R(x, a) for all x -/
def divides (k a : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, P x = D x k * q x + R x a

theorem polynomial_division_theorem :
  ∀ k a : ℝ, divides k a ↔ k = 5 ∧ a = -5 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2721_272113


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2721_272124

theorem inequality_solution_range (a : ℝ) (h_a : a > 0) :
  (∃ x : ℝ, x ∈ Set.Icc (-1) 2 ∧ 
   Real.exp 2 * Real.log a - Real.exp 2 * x + x - Real.log a ≥ 2 * a / Real.exp x - 2) ↔
  a ∈ Set.Icc (1 / Real.exp 1) (Real.exp 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2721_272124


namespace NUMINAMATH_CALUDE_heights_sum_l2721_272183

/-- Given the heights of John, Lena, and Rebeca, prove that the sum of Lena's and Rebeca's heights is 295 cm. -/
theorem heights_sum (john lena rebeca : ℕ) 
  (h1 : john = 152)
  (h2 : john = lena + 15)
  (h3 : rebeca = john + 6) :
  lena + rebeca = 295 := by
  sorry

end NUMINAMATH_CALUDE_heights_sum_l2721_272183


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2721_272187

/-- The equation (x+4)(x+1) = m + 2x has exactly one real solution if and only if m = 7/4 -/
theorem unique_solution_condition (m : ℝ) : 
  (∃! x : ℝ, (x + 4) * (x + 1) = m + 2 * x) ↔ m = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2721_272187


namespace NUMINAMATH_CALUDE_cuboid_height_l2721_272102

/-- Proves that the height of a cuboid with given base area and volume is 7 cm -/
theorem cuboid_height (base_area volume : ℝ) (h_base : base_area = 36) (h_volume : volume = 252) :
  volume / base_area = 7 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_l2721_272102


namespace NUMINAMATH_CALUDE_lcm_gcd_275_570_l2721_272135

theorem lcm_gcd_275_570 : 
  (Nat.lcm 275 570 = 31350) ∧ (Nat.gcd 275 570 = 5) := by sorry

end NUMINAMATH_CALUDE_lcm_gcd_275_570_l2721_272135


namespace NUMINAMATH_CALUDE_jims_journey_distance_l2721_272153

/-- The total distance of Jim's journey, given the miles driven and miles left to drive -/
def total_distance (miles_driven : ℕ) (miles_left : ℕ) : ℕ :=
  miles_driven + miles_left

/-- Theorem stating that the total distance of Jim's journey is 1200 miles -/
theorem jims_journey_distance :
  total_distance 384 816 = 1200 := by sorry

end NUMINAMATH_CALUDE_jims_journey_distance_l2721_272153


namespace NUMINAMATH_CALUDE_perpendicular_unit_vectors_l2721_272130

def vector_a : ℝ × ℝ := (2, -2)

def is_unit_vector (v : ℝ × ℝ) : Prop :=
  v.1 ^ 2 + v.2 ^ 2 = 1

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_unit_vectors :
  ∀ v : ℝ × ℝ, is_unit_vector v ∧ is_perpendicular v vector_a →
    v = (Real.sqrt 2 / 2, Real.sqrt 2 / 2) ∨ v = (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_unit_vectors_l2721_272130


namespace NUMINAMATH_CALUDE_basketball_volume_after_drilling_l2721_272117

/-- The volume of a basketball after drilling holes for handles -/
theorem basketball_volume_after_drilling (d : ℝ) (r1 r2 h : ℝ) :
  d = 50 ∧ r1 = 2 ∧ r2 = 1.5 ∧ h = 10 →
  (4/3 * π * (d/2)^3) - (2 * π * r1^2 * h + 2 * π * r2^2 * h) = (62250/3) * π :=
by sorry

end NUMINAMATH_CALUDE_basketball_volume_after_drilling_l2721_272117


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2721_272145

theorem geometric_series_ratio (a r : ℝ) (hr : |r| < 1) :
  (a / (1 - r) = 16 * (a * r^2 / (1 - r))) → |r| = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2721_272145


namespace NUMINAMATH_CALUDE_prob_condition_one_before_two_l2721_272160

/-- Represents the state of ball draws as a sorted list of integers -/
def DrawState := List Nat

/-- The probability of reaching a certain draw state -/
def StateProbability := DrawState → ℚ

/-- Checks if some ball has been drawn at least three times -/
def conditionOne (state : DrawState) : Prop :=
  state.head! ≥ 3

/-- Checks if every ball has been drawn at least once -/
def conditionTwo (state : DrawState) : Prop :=
  state.length = 3 ∧ state.all (· > 0)

/-- The probability of condition one occurring before condition two -/
def probConditionOneBeforeTwo (probMap : StateProbability) : ℚ :=
  probMap [3, 0, 0] + probMap [3, 1, 0] + probMap [3, 2, 0]

theorem prob_condition_one_before_two :
  ∃ (probMap : StateProbability),
    (∀ state, conditionOne state → conditionTwo state → probMap state = 0) →
    (probMap [0, 0, 0] = 1) →
    (∀ state, probMap state ≥ 0) →
    (∀ state, probMap state ≤ 1) →
    probConditionOneBeforeTwo probMap = 13 / 27 := by
  sorry

end NUMINAMATH_CALUDE_prob_condition_one_before_two_l2721_272160


namespace NUMINAMATH_CALUDE_wills_remaining_money_l2721_272156

-- Define the given amounts
def initial_amount : ℚ := 74
def sweater_cost : ℚ := 9
def tshirt_cost : ℚ := 11
def shoes_cost : ℚ := 30
def refund_percentage : ℚ := 90 / 100

-- Define the theorem
theorem wills_remaining_money :
  let clothes_cost := sweater_cost + tshirt_cost
  let refund := shoes_cost * refund_percentage
  let remaining := initial_amount - clothes_cost - shoes_cost + refund
  remaining = 81 := by sorry

end NUMINAMATH_CALUDE_wills_remaining_money_l2721_272156


namespace NUMINAMATH_CALUDE_not_all_right_triangles_are_isosceles_l2721_272182

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  sum_angles : angleA + angleB + angleC = 180
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define an isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Define a right triangle
def IsRight (t : Triangle) : Prop :=
  t.angleA = 90 ∨ t.angleB = 90 ∨ t.angleC = 90

-- The theorem to prove
theorem not_all_right_triangles_are_isosceles :
  ¬ (∀ t : Triangle, IsRight t → IsIsosceles t) :=
sorry

end NUMINAMATH_CALUDE_not_all_right_triangles_are_isosceles_l2721_272182


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_and_constant_term_l2721_272172

theorem binomial_coefficient_sum_and_constant_term 
  (x : ℝ) (a : ℝ) (n : ℕ) :
  (1 + a)^n = 32 →
  (∃ (r : ℕ), (n.choose r) * a^r = 80 ∧ 10 - 5*r = 0) →
  a = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_and_constant_term_l2721_272172


namespace NUMINAMATH_CALUDE_coefficient_of_x_l2721_272157

theorem coefficient_of_x (x : ℝ) : 
  ∃ (a b c d e : ℝ), (1 + x) * (x - 2/x)^3 = a*x^3 + b*x^2 + (-6)*x + c + d/x + e/(x^2) :=
sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l2721_272157


namespace NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l2721_272167

-- Define the vectors
def m (a : ℝ) : ℝ × ℝ := (a, a - 4)
def n (b : ℝ) : ℝ × ℝ := (b, 1 - b)

-- Define parallelism condition
def are_parallel (a b : ℝ) : Prop :=
  a * (1 - b) = b * (a - 4)

theorem min_sum_of_parallel_vectors (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_parallel : are_parallel a b) :
  a + b ≥ 9/2 ∧ (a + b = 9/2 ↔ a = 4 ∧ b = 2) := by
  sorry


end NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l2721_272167


namespace NUMINAMATH_CALUDE_min_area_archimedean_triangle_l2721_272116

/-- Represents a parabola with equation y^2 = 2px where p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Represents a chord of a parabola -/
structure Chord (para : Parabola) where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Represents the Archimedean triangle of a parabola and chord -/
structure ArchimedeanTriangle (para : Parabola) (chord : Chord para) where
  Q : ℝ × ℝ

/-- Predicate to check if a chord passes through the focus of a parabola -/
def passes_through_focus (para : Parabola) (chord : Chord para) : Prop :=
  ∃ t : ℝ, chord.A = (para.p / 2, t) ∨ chord.B = (para.p / 2, t)

/-- Calculate the area of a triangle given its vertices -/
def triangle_area (A B Q : ℝ × ℝ) : ℝ := sorry

/-- The main theorem: The minimum area of the Archimedean triangle is p^2 -/
theorem min_area_archimedean_triangle (para : Parabola) 
  (chord : Chord para) (arch_tri : ArchimedeanTriangle para chord)
  (h_focus : passes_through_focus para chord) :
  ∃ (min_area : ℝ), 
    (∀ (other_chord : Chord para) (other_tri : ArchimedeanTriangle para other_chord),
      passes_through_focus para other_chord → 
      triangle_area arch_tri.Q chord.A chord.B ≤ triangle_area other_tri.Q other_chord.A other_chord.B) ∧
    min_area = para.p^2 := by
  sorry

end NUMINAMATH_CALUDE_min_area_archimedean_triangle_l2721_272116


namespace NUMINAMATH_CALUDE_units_digit_of_8_power_47_l2721_272136

theorem units_digit_of_8_power_47 : Nat.mod (8^47) 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_8_power_47_l2721_272136


namespace NUMINAMATH_CALUDE_cards_distribution_l2721_272176

theorem cards_distribution (total_cards : ℕ) (num_people : ℕ) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l2721_272176


namespace NUMINAMATH_CALUDE_a_divisibility_l2721_272173

/-- Sequence a_n defined recursively -/
def a (k : ℤ) : ℕ → ℤ
  | 0 => 0
  | 1 => k
  | (n + 2) => k^2 * a k (n + 1) - a k n

/-- Theorem stating that a_{n+1} * a_n + 1 divides a_{n+1}^2 + a_n^2 for all n -/
theorem a_divisibility (k : ℤ) (n : ℕ) :
  ∃ m : ℤ, (a k (n + 1))^2 + (a k n)^2 = ((a k (n + 1)) * (a k n) + 1) * m := by
  sorry

end NUMINAMATH_CALUDE_a_divisibility_l2721_272173


namespace NUMINAMATH_CALUDE_picnic_attendance_theorem_l2721_272118

/-- Represents the percentage of employees who are men -/
def male_percentage : ℝ := 0.5

/-- Represents the percentage of women who attended the picnic -/
def women_attendance_percentage : ℝ := 0.4

/-- Represents the percentage of all employees who attended the picnic -/
def total_attendance_percentage : ℝ := 0.3

/-- Represents the percentage of men who attended the picnic -/
def male_attendance_percentage : ℝ := 0.2

theorem picnic_attendance_theorem :
  male_attendance_percentage * male_percentage + 
  women_attendance_percentage * (1 - male_percentage) = 
  total_attendance_percentage := by
  sorry

#check picnic_attendance_theorem

end NUMINAMATH_CALUDE_picnic_attendance_theorem_l2721_272118


namespace NUMINAMATH_CALUDE_remainder_zero_l2721_272109

/-- A polynomial of degree 5 with real coefficients -/
structure Poly5 (D E F G H : ℝ) where
  q : ℝ → ℝ
  eq : ∀ x, q x = D * x^5 + E * x^4 + F * x^3 + G * x^2 + H * x + 2

/-- The remainder theorem for polynomials -/
axiom remainder_theorem {p : ℝ → ℝ} {a r : ℝ} :
  (∀ x, ∃ q, p x = (x - a) * q + r) ↔ p a = r

/-- Main theorem: If the remainder of q(x) divided by (x - 4) is 15,
    then the remainder of q(x) divided by (x + 4) is 0 -/
theorem remainder_zero {D E F G H : ℝ} (p : Poly5 D E F G H) :
  p.q 4 = 15 → p.q (-4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_zero_l2721_272109


namespace NUMINAMATH_CALUDE_work_completion_days_l2721_272104

/-- The number of days B can finish the work -/
def b_days : ℕ := 15

/-- The number of days B worked before leaving -/
def b_worked : ℕ := 10

/-- The number of days A needs to finish the remaining work after B left -/
def a_remaining : ℕ := 2

/-- The number of days A can finish the entire work -/
def a_days : ℕ := 6

theorem work_completion_days :
  (b_worked : ℚ) / b_days + a_remaining / a_days = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_days_l2721_272104


namespace NUMINAMATH_CALUDE_first_girl_productivity_higher_l2721_272185

/-- Represents a knitter with their working time and break time -/
structure Knitter where
  workTime : ℕ
  breakTime : ℕ

/-- Calculates the total cycle time for a knitter -/
def cycleTime (k : Knitter) : ℕ :=
  k.workTime + k.breakTime

/-- Calculates the number of complete cycles in a given time -/
def completeCycles (k : Knitter) (totalTime : ℕ) : ℕ :=
  totalTime / cycleTime k

/-- Calculates the total working time within a given time period -/
def totalWorkTime (k : Knitter) (totalTime : ℕ) : ℕ :=
  completeCycles k totalTime * k.workTime

/-- Theorem: The first girl's productivity is 5% higher than the second girl's -/
theorem first_girl_productivity_higher (girl1 girl2 : Knitter)
    (h1 : girl1.workTime = 5)
    (h2 : girl2.workTime = 7)
    (h3 : girl1.breakTime = 1)
    (h4 : girl2.breakTime = 1)
    (h5 : ∃ t : ℕ, totalWorkTime girl1 t = totalWorkTime girl2 t ∧ t > 0) :
    (21 : ℚ) / 20 = girl2.workTime / girl1.workTime := by
  sorry

end NUMINAMATH_CALUDE_first_girl_productivity_higher_l2721_272185


namespace NUMINAMATH_CALUDE_sum_of_digits_seven_power_fifteen_l2721_272137

/-- The sum of the tens digit and the ones digit of 7^15 is 7 -/
theorem sum_of_digits_seven_power_fifteen : ∃ (a b : ℕ), 
  7^15 % 100 = 10 * a + b ∧ a + b = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_seven_power_fifteen_l2721_272137


namespace NUMINAMATH_CALUDE_bca_equals_341_l2721_272159

def repeating_decimal_bc (b c : ℕ) : ℚ :=
  (10 * b + c : ℚ) / 99

def repeating_decimal_bcabc (b c a : ℕ) : ℚ :=
  (10000 * b + 1000 * c + 100 * a + 10 * b + c : ℚ) / 99999

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem bca_equals_341 (b c a : ℕ) 
  (hb : is_digit b) (hc : is_digit c) (ha : is_digit a)
  (h_eq : repeating_decimal_bc b c + repeating_decimal_bcabc b c a = 41 / 111) :
  100 * b + 10 * c + a = 341 := by
sorry

end NUMINAMATH_CALUDE_bca_equals_341_l2721_272159


namespace NUMINAMATH_CALUDE_diagonals_difference_octagon_heptagon_l2721_272158

/-- Number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

/-- A heptagon has 7 sides -/
def heptagon_sides : ℕ := 7

theorem diagonals_difference_octagon_heptagon :
  num_diagonals octagon_sides - num_diagonals heptagon_sides = 6 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_difference_octagon_heptagon_l2721_272158


namespace NUMINAMATH_CALUDE_million_millimeters_equals_one_kilometer_l2721_272112

-- Define the conversion factors
def millimeters_per_meter : ℕ := 1000
def meters_per_kilometer : ℕ := 1000

-- Define the question
def million_millimeters : ℕ := 1000000

-- Theorem to prove
theorem million_millimeters_equals_one_kilometer :
  (million_millimeters / millimeters_per_meter) / meters_per_kilometer = 1 := by
  sorry

end NUMINAMATH_CALUDE_million_millimeters_equals_one_kilometer_l2721_272112


namespace NUMINAMATH_CALUDE_bisecting_line_sum_l2721_272195

/-- A line that bisects the circumference of a circle -/
structure BisectingLine where
  a : ℝ
  b : ℝ
  bisects : ∀ (x y : ℝ), a * x + b * y + 1 = 0 → 
    (x^2 + y^2 + 2*x + 2*y - 1 = 0 → 
      ∃ (p q : ℝ), p^2 + q^2 + 2*p + 2*q - 1 = 0 ∧ 
        a * p + b * q + 1 = 0 ∧ (p, q) ≠ (x, y))

/-- Theorem: If a line ax + by + 1 = 0 bisects the circumference of 
    the circle x^2 + y^2 + 2x + 2y - 1 = 0, then a + b = 1 -/
theorem bisecting_line_sum (l : BisectingLine) : l.a + l.b = 1 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_l2721_272195


namespace NUMINAMATH_CALUDE_system_positive_solution_l2721_272171

theorem system_positive_solution (a b : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    x₁ - x₂ = a ∧ 
    x₃ - x₄ = b ∧ 
    x₁ + x₂ + x₃ + x₄ = 1 ∧ 
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) ↔ 
  abs a + abs b < 1 :=
by sorry

end NUMINAMATH_CALUDE_system_positive_solution_l2721_272171


namespace NUMINAMATH_CALUDE_molecular_weight_BaBr2_is_297_l2721_272149

/-- The molecular weight of BaBr2 in grams per mole -/
def molecular_weight_BaBr2 : ℝ := 297

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 8

/-- The total weight of the given moles of BaBr2 in grams -/
def total_weight : ℝ := 2376

/-- Theorem: The molecular weight of BaBr2 is 297 grams/mole -/
theorem molecular_weight_BaBr2_is_297 :
  molecular_weight_BaBr2 = total_weight / given_moles :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_BaBr2_is_297_l2721_272149


namespace NUMINAMATH_CALUDE_cost_price_is_95_l2721_272144

/-- Represents the cost price for one metre of cloth -/
def cost_price_per_metre (total_metres : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  total_selling_price / total_metres + loss_per_metre

/-- Theorem stating that the cost price for one metre of cloth is 95 -/
theorem cost_price_is_95 :
  cost_price_per_metre 200 18000 5 = 95 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_is_95_l2721_272144


namespace NUMINAMATH_CALUDE_renata_final_amount_l2721_272198

/-- Represents Renata's financial transactions --/
def renataTransactions : List Int :=
  [10, -4, 90, -50, -10, -5, -1, -1, 65]

/-- Calculates the final amount Renata has --/
def finalAmount (transactions : List Int) : Int :=
  transactions.sum

/-- Theorem stating that Renata ends up with $94 --/
theorem renata_final_amount :
  finalAmount renataTransactions = 94 := by
  sorry

#eval finalAmount renataTransactions

end NUMINAMATH_CALUDE_renata_final_amount_l2721_272198


namespace NUMINAMATH_CALUDE_pure_imaginary_quadratic_l2721_272152

theorem pure_imaginary_quadratic (a : ℝ) : 
  (Complex.mk (a^2 - 4*a + 3) (a - 1)).im ≠ 0 ∧ (Complex.mk (a^2 - 4*a + 3) (a - 1)).re = 0 → 
  a = 1 ∨ a = 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_quadratic_l2721_272152


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2721_272138

theorem quadratic_roots_property : ∃ (x y : ℝ), 
  (x + y = 10) ∧ 
  (|x - y| = 6) ∧ 
  (∀ z : ℝ, z^2 - 10*z + 16 = 0 ↔ (z = x ∨ z = y)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2721_272138


namespace NUMINAMATH_CALUDE_fraction_of_nuts_eaten_l2721_272120

def initial_nuts : ℕ := 30
def remaining_nuts : ℕ := 5

theorem fraction_of_nuts_eaten :
  (initial_nuts - remaining_nuts) / initial_nuts = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_nuts_eaten_l2721_272120


namespace NUMINAMATH_CALUDE_jordan_list_count_l2721_272110

def smallest_square_multiple (n : ℕ) : ℕ := 
  Nat.lcm (n^2) n

def smallest_cube_multiple (n : ℕ) : ℕ := 
  Nat.lcm (n^3) n

theorem jordan_list_count : 
  let lower_bound := smallest_square_multiple 30
  let upper_bound := smallest_cube_multiple 30
  (upper_bound - lower_bound) / 30 + 1 = 871 := by sorry

end NUMINAMATH_CALUDE_jordan_list_count_l2721_272110


namespace NUMINAMATH_CALUDE_triangle_isosceles_from_quadrilateral_property_l2721_272126

structure Triangle where
  angles : Fin 3 → ℝ
  sum_eq_pi : angles 0 + angles 1 + angles 2 = π

structure Quadrilateral where
  angles : Fin 4 → ℝ
  sum_eq_2pi : angles 0 + angles 1 + angles 2 + angles 3 = 2 * π

def has_sum_angle_property (t : Triangle) (q : Quadrilateral) : Prop :=
  ∀ (i j : Fin 3), i ≠ j → ∃ (k : Fin 4), q.angles k = t.angles i + t.angles j

def is_isosceles (t : Triangle) : Prop :=
  ∃ (i j : Fin 3), i ≠ j ∧ t.angles i = t.angles j

theorem triangle_isosceles_from_quadrilateral_property
  (t : Triangle) (q : Quadrilateral) (h : has_sum_angle_property t q) :
  is_isosceles t :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_from_quadrilateral_property_l2721_272126


namespace NUMINAMATH_CALUDE_exists_four_digit_number_divisible_by_101_when_reversed_l2721_272105

/-- Reverses a four-digit number -/
def reverse (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

/-- Checks if a number has distinct non-zero digits -/
def has_distinct_nonzero_digits (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n % 10 ≠ 0) ∧ ((n / 10) % 10 ≠ 0) ∧ ((n / 100) % 10 ≠ 0) ∧ (n / 1000 ≠ 0) ∧
  (n % 10 ≠ (n / 10) % 10) ∧ (n % 10 ≠ (n / 100) % 10) ∧ (n % 10 ≠ n / 1000) ∧
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧ ((n / 10) % 10 ≠ n / 1000) ∧
  ((n / 100) % 10 ≠ n / 1000)

theorem exists_four_digit_number_divisible_by_101_when_reversed :
  ∃ n : ℕ, has_distinct_nonzero_digits n ∧ (n + reverse n) % 101 = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_four_digit_number_divisible_by_101_when_reversed_l2721_272105


namespace NUMINAMATH_CALUDE_variance_of_white_balls_l2721_272191

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := 7

/-- The number of trials -/
def num_trials : ℕ := 7

/-- The probability of drawing a white ball -/
def p : ℚ := white_balls / total_balls

/-- X is the random variable representing the number of white balls drawn -/
def X : Type := Unit

/-- The variance of a binomial distribution -/
def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem variance_of_white_balls :
  binomial_variance num_trials p = 12 / 7 :=
sorry

end NUMINAMATH_CALUDE_variance_of_white_balls_l2721_272191


namespace NUMINAMATH_CALUDE_right_vertex_intersection_l2721_272100

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 4) = 1

-- Define the line
def line (x y a : ℝ) : Prop := y = x - a

-- State the theorem
theorem right_vertex_intersection (a : ℝ) :
  ellipse 3 0 ∧ line 3 0 a → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_right_vertex_intersection_l2721_272100


namespace NUMINAMATH_CALUDE_inequality_proof_l2721_272127

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  a + b + Real.sqrt 2 * c ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2721_272127


namespace NUMINAMATH_CALUDE_inequality_range_l2721_272101

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 ≤ 0) ↔ -2 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l2721_272101


namespace NUMINAMATH_CALUDE_card_ratio_l2721_272186

/-- Proves the ratio of cards eaten by the dog to the total cards before the incident -/
theorem card_ratio (new_cards : ℕ) (remaining_cards : ℕ) : 
  new_cards = 4 → remaining_cards = 34 → 
  (new_cards + remaining_cards - remaining_cards) / (new_cards + remaining_cards) = 2 / 19 := by
  sorry

end NUMINAMATH_CALUDE_card_ratio_l2721_272186


namespace NUMINAMATH_CALUDE_circle_area_equality_l2721_272164

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) : 
  r₁ = 17 → r₂ = 27 → r₃ = 10 * Real.sqrt 11 → 
  π * r₃^2 = π * (r₂^2 - r₁^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equality_l2721_272164


namespace NUMINAMATH_CALUDE_complex_product_example_l2721_272166

theorem complex_product_example : (1 + Complex.I) * (2 + Complex.I) = 1 + 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_example_l2721_272166


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l2721_272146

/-- The ratio of a man's age to his son's age after two years, given their current ages. -/
theorem man_son_age_ratio (son_age : ℕ) (man_age : ℕ) : 
  son_age = 22 →
  man_age = son_age + 24 →
  (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l2721_272146


namespace NUMINAMATH_CALUDE_unknown_table_has_one_leg_l2721_272154

/-- The number of legs on the table with the unknown number of legs -/
def unknown_table_legs : ℕ := sorry

/-- The total number of legs in the room -/
def total_legs : ℕ := 40

/-- The number of legs on all furniture except the unknown table -/
def known_legs : ℕ := 
  4 * 4 +  -- 4 tables with 4 legs each
  1 * 4 +  -- 1 sofa with 4 legs
  2 * 4 +  -- 2 chairs with 4 legs each
  3 * 3 +  -- 3 tables with 3 legs each
  1 * 2    -- 1 rocking chair with 2 legs

theorem unknown_table_has_one_leg : 
  unknown_table_legs = 1 :=
by
  sorry

#check unknown_table_has_one_leg

end NUMINAMATH_CALUDE_unknown_table_has_one_leg_l2721_272154
