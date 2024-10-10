import Mathlib

namespace max_true_statements_l3208_320894

theorem max_true_statements (a b : ℝ) : 
  0 < a ∧ 0 < b ∧ a < b →
  (1 / a < 1 / b) ∧ 
  (a^2 > b^2) ∧ 
  (a < b) ∧ 
  (a > 0) ∧ 
  (b > 0) := by
  sorry

end max_true_statements_l3208_320894


namespace circle_equation_k_range_l3208_320858

theorem circle_equation_k_range (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 4*x + 4*y + 10 - k = 0) → k > 2 :=
by sorry

end circle_equation_k_range_l3208_320858


namespace euler_most_prolific_l3208_320824

/-- Represents a mathematician -/
structure Mathematician where
  name : String
  country : String
  published_volumes : ℕ

/-- The Swiss Society of Natural Sciences -/
def SwissSocietyOfNaturalSciences : Set Mathematician := sorry

/-- Leonhard Euler -/
def euler : Mathematician := {
  name := "Leonhard Euler",
  country := "Switzerland",
  published_volumes := 76  -- More than 75 volumes
}

/-- Predicate for being the most prolific mathematician -/
def most_prolific (m : Mathematician) : Prop :=
  ∀ n : Mathematician, n.published_volumes ≤ m.published_volumes

theorem euler_most_prolific :
  euler ∈ SwissSocietyOfNaturalSciences →
  euler.country = "Switzerland" →
  euler.published_volumes > 75 →
  most_prolific euler :=
sorry

end euler_most_prolific_l3208_320824


namespace complex_sum_problem_l3208_320815

theorem complex_sum_problem (a b c d e f : ℝ) : 
  d = 2 →
  e = -a - 2*c →
  (a + b*Complex.I) + (c + d*Complex.I) + (e + f*Complex.I) = -7*Complex.I →
  b + f = -9 := by
sorry

end complex_sum_problem_l3208_320815


namespace jake_has_more_apples_l3208_320865

def steven_apples : ℕ := 14

theorem jake_has_more_apples (jake_apples : ℕ) (h : jake_apples > steven_apples) :
  jake_apples > steven_apples := by sorry

end jake_has_more_apples_l3208_320865


namespace fraction_equality_l3208_320802

theorem fraction_equality : (35 : ℚ) / (6 - 2/5) = 25/4 := by sorry

end fraction_equality_l3208_320802


namespace parabola_circle_fixed_points_l3208_320878

/-- Parabola C: x^2 = -4y -/
def parabola (x y : ℝ) : Prop := x^2 = -4*y

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (0, -1)

/-- Line l with non-zero slope k passing through the focus -/
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*x - 1 ∧ k ≠ 0

/-- Intersection points M and N of line l with parabola C -/
def intersection_points (k : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Points A and B where y = -1 intersects OM and ON -/
def points_AB (k : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Circle with diameter AB -/
def circle_AB (k : ℝ) (x y : ℝ) : Prop :=
  ∃ (xA yA xB yB : ℝ), (points_AB k = (xA, yA, xB, yB)) ∧
  (x - (xA + xB)/2)^2 + (y - (yA + yB)/2)^2 = ((xA - xB)^2 + (yA - yB)^2) / 4

theorem parabola_circle_fixed_points (k : ℝ) :
  (∀ x y, circle_AB k x y → (x = 0 ∧ y = 1) ∨ (x = 0 ∧ y = -3)) ∧
  (circle_AB k 0 1 ∧ circle_AB k 0 (-3)) :=
sorry

end parabola_circle_fixed_points_l3208_320878


namespace integral_tangent_sine_cosine_l3208_320867

open Real MeasureTheory

theorem integral_tangent_sine_cosine :
  ∫ x in (Set.Icc 0 (π/4)), (7 + 3 * tan x) / (sin x + 2 * cos x)^2 = 3 * log (3/2) + 1/6 := by
  sorry

end integral_tangent_sine_cosine_l3208_320867


namespace equal_sums_exist_l3208_320840

/-- Represents the direction a recruit is facing -/
inductive Direction
  | Left : Direction
  | Right : Direction
  | Around : Direction

/-- A line of recruits is represented as a list of their facing directions -/
def RecruitLine := List Direction

/-- Converts a Direction to an integer value -/
def directionToInt (d : Direction) : Int :=
  match d with
  | Direction.Left => -1
  | Direction.Right => 1
  | Direction.Around => 0

/-- Calculates the sum of directions to the left of a given index -/
def leftSum (line : RecruitLine) (index : Nat) : Int :=
  (line.take index).map directionToInt |>.sum

/-- Calculates the sum of directions to the right of a given index -/
def rightSum (line : RecruitLine) (index : Nat) : Int :=
  (line.drop (index + 1)).map directionToInt |>.sum

/-- Theorem: There always exists a position where the left sum equals the right sum -/
theorem equal_sums_exist (line : RecruitLine) :
  ∃ (index : Nat), leftSum line index = rightSum line index :=
  sorry

end equal_sums_exist_l3208_320840


namespace circle_sum_center_radius_l3208_320810

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x - 4 = -y^2 + 2*y

-- Define the center and radius of the circle
def circle_center_radius (a b r : ℝ) : Prop :=
  ∀ x y, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem circle_sum_center_radius :
  ∃ a b r, circle_center_radius a b r ∧ a + b + r = 5 + Real.sqrt 21 :=
sorry

end circle_sum_center_radius_l3208_320810


namespace max_intersection_points_circle_triangle_l3208_320876

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A triangle in a plane --/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The number of intersection points between a circle and a line segment --/
def intersectionPointsCircleLine (c : Circle) (p1 p2 : ℝ × ℝ) : ℕ := sorry

/-- The number of intersection points between a circle and a triangle --/
def intersectionPointsCircleTriangle (c : Circle) (t : Triangle) : ℕ :=
  (intersectionPointsCircleLine c (t.vertices 0) (t.vertices 1)) +
  (intersectionPointsCircleLine c (t.vertices 1) (t.vertices 2)) +
  (intersectionPointsCircleLine c (t.vertices 2) (t.vertices 0))

/-- The maximum number of intersection points between a circle and a triangle is 6 --/
theorem max_intersection_points_circle_triangle :
  ∃ (c : Circle) (t : Triangle), 
    (∀ (c' : Circle) (t' : Triangle), intersectionPointsCircleTriangle c' t' ≤ 6) ∧
    intersectionPointsCircleTriangle c t = 6 :=
  sorry

end max_intersection_points_circle_triangle_l3208_320876


namespace college_students_count_l3208_320880

theorem college_students_count :
  ∀ (total : ℕ) (enrolled_percent : ℚ) (not_enrolled : ℕ),
    enrolled_percent = 1/2 →
    not_enrolled = 440 →
    (1 - enrolled_percent) * total = not_enrolled →
    total = 880 := by
  sorry

end college_students_count_l3208_320880


namespace wheel_marking_theorem_l3208_320809

theorem wheel_marking_theorem :
  ∃ (R : ℝ), R > 0 ∧ 
    ∀ (θ : ℝ), 0 ≤ θ ∧ θ < 360 → 
      ∃ (n : ℕ), 0 ≤ n - R * θ / 360 ∧ n - R * θ / 360 < R / 360 := by
  sorry

end wheel_marking_theorem_l3208_320809


namespace total_games_five_months_l3208_320814

def games_month1 : ℕ := 32
def games_month2 : ℕ := 24
def games_month3 : ℕ := 29
def games_month4 : ℕ := 19
def games_month5 : ℕ := 34

theorem total_games_five_months :
  games_month1 + games_month2 + games_month3 + games_month4 + games_month5 = 138 := by
  sorry

end total_games_five_months_l3208_320814


namespace max_PXQ_value_l3208_320877

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ (n / 10 = n % 10)

def is_one_digit (n : ℕ) : Prop :=
  0 < n ∧ n ≤ 9

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem max_PXQ_value :
  ∀ X XX PXQ : ℕ,
  is_two_digit_with_equal_digits XX →
  is_one_digit X →
  is_three_digit PXQ →
  XX * X = PXQ →
  PXQ ≤ 396 :=
sorry

end max_PXQ_value_l3208_320877


namespace grandmothers_age_is_52_l3208_320848

/-- The age of the grandmother given the average age of the family and the ages of the children -/
def grandmothers_age (average_age : ℝ) (child1_age child2_age child3_age : ℕ) : ℝ :=
  4 * average_age - (child1_age + child2_age + child3_age)

/-- Theorem stating that the grandmother's age is 52 given the problem conditions -/
theorem grandmothers_age_is_52 :
  grandmothers_age 20 5 10 13 = 52 := by
  sorry

end grandmothers_age_is_52_l3208_320848


namespace rectangle_area_l3208_320826

theorem rectangle_area (L W : ℝ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) :
  L * W = 140 := by
  sorry

end rectangle_area_l3208_320826


namespace codes_lost_calculation_l3208_320891

/-- The number of digits in each code -/
def code_length : ℕ := 4

/-- The base of the number system (decimal) -/
def base : ℕ := 10

/-- The total number of possible codes with leading zeros -/
def total_codes : ℕ := base ^ code_length

/-- The number of possible codes without leading zeros -/
def codes_without_leading_zeros : ℕ := (base - 1) * (base ^ (code_length - 1))

/-- The number of codes lost when disallowing leading zeros -/
def codes_lost : ℕ := total_codes - codes_without_leading_zeros

theorem codes_lost_calculation :
  codes_lost = 1000 :=
sorry

end codes_lost_calculation_l3208_320891


namespace jm_length_l3208_320852

/-- Triangle DEF with medians and centroid -/
structure TriangleWithCentroid where
  -- Define the triangle
  DE : ℝ
  DF : ℝ
  EF : ℝ
  -- Define the centroid
  J : ℝ × ℝ
  -- Define M as the foot of the altitude from J to EF
  M : ℝ × ℝ

/-- The theorem stating the length of JM in the given triangle -/
theorem jm_length (t : TriangleWithCentroid) 
  (h1 : t.DE = 14) 
  (h2 : t.DF = 15) 
  (h3 : t.EF = 21) : 
  Real.sqrt ((t.J.1 - t.M.1)^2 + (t.J.2 - t.M.2)^2) = 8/3 := by
  sorry


end jm_length_l3208_320852


namespace milk_water_ratio_l3208_320888

theorem milk_water_ratio (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) (added_water : ℚ) :
  initial_volume = 45 ∧
  initial_milk_ratio = 4 ∧
  initial_water_ratio = 1 ∧
  added_water = 3 →
  let total_parts := initial_milk_ratio + initial_water_ratio
  let initial_milk_volume := (initial_milk_ratio / total_parts) * initial_volume
  let initial_water_volume := (initial_water_ratio / total_parts) * initial_volume
  let new_water_volume := initial_water_volume + added_water
  let new_milk_ratio := initial_milk_volume
  let new_water_ratio := new_water_volume
  (new_milk_ratio : ℚ) / new_water_ratio = 3 / 1 :=
by sorry

end milk_water_ratio_l3208_320888


namespace triangle_shape_l3208_320887

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = π) →
  (a^2 + b^2 + c^2 = 2 * Real.sqrt 3 * a * b * Real.sin C) →
  (a = b ∧ b = c ∧ A = B ∧ B = C ∧ C = π/3) :=
by sorry

end triangle_shape_l3208_320887


namespace orange_straws_count_l3208_320801

/-- The number of orange straws needed for each mat -/
def orange_straws : ℕ := 30

/-- The number of red straws needed for each mat -/
def red_straws : ℕ := 20

/-- The number of green straws needed for each mat -/
def green_straws : ℕ := orange_straws / 2

/-- The total number of mats -/
def total_mats : ℕ := 10

/-- The total number of straws needed for all mats -/
def total_straws : ℕ := 650

theorem orange_straws_count :
  orange_straws = 30 ∧
  red_straws = 20 ∧
  green_straws = orange_straws / 2 ∧
  total_mats * (red_straws + orange_straws + green_straws) = total_straws :=
by sorry

end orange_straws_count_l3208_320801


namespace square_area_given_equal_perimeter_triangle_l3208_320872

theorem square_area_given_equal_perimeter_triangle (s : ℝ) (a : ℝ) : 
  s > 0 → -- side length of equilateral triangle is positive
  a > 0 → -- side length of square is positive
  3 * s = 4 * a → -- equal perimeters
  s^2 * Real.sqrt 3 / 4 = 9 → -- area of equilateral triangle is 9
  a^2 = 27 * Real.sqrt 3 / 4 := by
sorry

end square_area_given_equal_perimeter_triangle_l3208_320872


namespace connected_triangles_theorem_l3208_320811

/-- A sequence of three connected right-angled triangles -/
structure TriangleSequence where
  -- First triangle
  AE : ℝ
  BE : ℝ
  -- Second triangle
  CE : ℝ
  -- Angles
  angleAEB : Real
  angleBEC : Real
  angleCED : Real

/-- The theorem statement -/
theorem connected_triangles_theorem (t : TriangleSequence) : 
  t.AE = 20 ∧ 
  t.angleAEB = 45 ∧ 
  t.angleBEC = 45 ∧ 
  t.angleCED = 45 → 
  t.CE = 10 := by
  sorry


end connected_triangles_theorem_l3208_320811


namespace x_value_proof_l3208_320807

theorem x_value_proof (x : ℝ) (h : 9 / x^3 = x / 81) : x = 3 * Real.sqrt 3 := by
  sorry

end x_value_proof_l3208_320807


namespace hyperbola_eccentricity_range_l3208_320813

/-- Given a hyperbola and an intersecting line, prove the eccentricity range -/
theorem hyperbola_eccentricity_range 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (m : ℝ) 
  (h_intersect : ∃ x y : ℝ, y = 2*x + m ∧ x^2/a^2 - y^2/b^2 = 1) :
  ∃ e : ℝ, e^2 = (a^2 + b^2) / a^2 ∧ e > Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_range_l3208_320813


namespace sum_of_integers_l3208_320864

theorem sum_of_integers (x y : ℕ+) 
  (sum_of_squares : x^2 + y^2 = 245)
  (product : x * y = 120) : 
  (x : ℝ) + y = Real.sqrt 485 := by
sorry

end sum_of_integers_l3208_320864


namespace unused_bricks_fraction_l3208_320800

def bricks_used : ℝ := 20
def bricks_remaining : ℝ := 10

theorem unused_bricks_fraction :
  bricks_remaining / (bricks_used + bricks_remaining) = 1/3 := by
  sorry

end unused_bricks_fraction_l3208_320800


namespace expression_equality_l3208_320838

theorem expression_equality : 
  Real.sqrt 16 - 4 * (Real.sqrt 2 / 2) + abs (-Real.sqrt 3 * Real.sqrt 6) + (-1)^2023 = 3 + Real.sqrt 2 := by
  sorry

end expression_equality_l3208_320838


namespace locus_and_max_area_l3208_320866

noncomputable section

-- Define the points E, F, and D
def E : ℝ × ℝ := (-2, 0)
def F : ℝ × ℝ := (2, 0)
def D : ℝ × ℝ := (0, -2)

-- Define the moving point P
def P : ℝ × ℝ → Prop
  | (x, y) => (x + 2) * x + (y - 0) * y = 0 ∧ (x - 2) * x + (y - 0) * y = 0

-- Define the point M
def M : ℝ × ℝ → Prop
  | (x, y) => ∃ (px py : ℝ), P (px, py) ∧ px = x ∧ py = 2 * y

-- Define the locus C
def C : ℝ × ℝ → Prop
  | (x, y) => M (x, y)

-- Define the line l
def l (k : ℝ) : ℝ × ℝ → Prop
  | (x, y) => y = k * x - 2

-- Define the area of quadrilateral OANB
def area_OANB (k : ℝ) : ℝ := 
  8 * Real.sqrt ((4 * k^2 - 3) / (1 + 4 * k^2)^2)

-- Theorem statement
theorem locus_and_max_area :
  (∀ x y, C (x, y) ↔ x^2 / 4 + y^2 = 1) ∧
  (∃ k₁ k₂, k₁ ≠ k₂ ∧
    area_OANB k₁ = 2 ∧
    area_OANB k₂ = 2 ∧
    (∀ k, area_OANB k ≤ 2) ∧
    l k₁ = λ (x, y) => y = Real.sqrt 7 / 2 * x - 2 ∧
    l k₂ = λ (x, y) => y = -Real.sqrt 7 / 2 * x - 2) :=
by sorry

end locus_and_max_area_l3208_320866


namespace field_trip_bus_occupancy_l3208_320869

theorem field_trip_bus_occupancy
  (num_vans : ℕ)
  (num_buses : ℕ)
  (people_per_van : ℕ)
  (total_people : ℕ)
  (h1 : num_vans = 6)
  (h2 : num_buses = 8)
  (h3 : people_per_van = 6)
  (h4 : total_people = 180)
  : (total_people - num_vans * people_per_van) / num_buses = 18 := by
  sorry

end field_trip_bus_occupancy_l3208_320869


namespace average_apples_sold_example_l3208_320828

/-- Calculates the average number of kg of apples sold per hour given the sales in two hours -/
def average_apples_sold (first_hour_sales second_hour_sales : ℕ) : ℚ :=
  (first_hour_sales + second_hour_sales : ℚ) / 2

theorem average_apples_sold_example : average_apples_sold 10 2 = 6 := by
  sorry

end average_apples_sold_example_l3208_320828


namespace scooter_price_l3208_320851

/-- Given an upfront payment of 20% of the total cost, which amounts to $240, prove that the total price of the scooter is $1200. -/
theorem scooter_price (upfront_percentage : ℝ) (upfront_amount : ℝ) (total_price : ℝ) : 
  upfront_percentage = 0.20 → 
  upfront_amount = 240 → 
  upfront_percentage * total_price = upfront_amount → 
  total_price = 1200 := by
sorry

end scooter_price_l3208_320851


namespace song_ratio_after_deletion_l3208_320818

theorem song_ratio_after_deletion (total : ℕ) (deletion_percentage : ℚ) 
  (h1 : total = 720) 
  (h2 : deletion_percentage = 1/5) : 
  (total - (deletion_percentage * total).floor) / (deletion_percentage * total).floor = 4 := by
  sorry

end song_ratio_after_deletion_l3208_320818


namespace affected_days_in_factory_l3208_320895

/-- Proves the number of affected days in a TV factory --/
theorem affected_days_in_factory (first_25_avg : ℝ) (overall_avg : ℝ) (affected_avg : ℝ)
  (h1 : first_25_avg = 60)
  (h2 : overall_avg = 58)
  (h3 : affected_avg = 48) :
  ∃ x : ℝ, x = 5 ∧ 25 * first_25_avg + x * affected_avg = (25 + x) * overall_avg :=
by sorry

end affected_days_in_factory_l3208_320895


namespace tangencyTriangleAreaTheorem_l3208_320843

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a triangle formed by the points of tangency of three circles -/
structure TangencyTriangle where
  c1 : Circle
  c2 : Circle
  c3 : Circle

/-- The area of the triangle formed by the points of tangency of three mutually externally tangent circles -/
def tangencyTriangleArea (t : TangencyTriangle) : ℝ :=
  sorry

/-- Theorem stating that the area of the triangle formed by the points of tangency
    of three mutually externally tangent circles with radii 1, 3, and 5 is 5/3 -/
theorem tangencyTriangleAreaTheorem :
  let c1 : Circle := { radius := 1 }
  let c2 : Circle := { radius := 3 }
  let c3 : Circle := { radius := 5 }
  let t : TangencyTriangle := { c1 := c1, c2 := c2, c3 := c3 }
  tangencyTriangleArea t = 5/3 := by
  sorry

end tangencyTriangleAreaTheorem_l3208_320843


namespace graveyard_bones_problem_l3208_320861

theorem graveyard_bones_problem :
  let total_skeletons : ℕ := 20
  let adult_women : ℕ := total_skeletons / 2
  let adult_men : ℕ := (total_skeletons - adult_women) / 2
  let children : ℕ := total_skeletons - adult_women - adult_men
  let total_bones : ℕ := 375
  let woman_bones : ℕ → ℕ := λ x => x
  let man_bones : ℕ → ℕ := λ x => x + 5
  let child_bones : ℕ → ℕ := λ x => x / 2

  ∃ (w : ℕ), 
    adult_women * (woman_bones w) + 
    adult_men * (man_bones w) + 
    children * (child_bones w) = total_bones ∧ 
    w = 20 :=
by
  sorry

end graveyard_bones_problem_l3208_320861


namespace inverse_difference_equals_negative_reciprocal_l3208_320834

theorem inverse_difference_equals_negative_reciprocal (a b : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hab : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ - (b / 3)⁻¹) = -(a * b)⁻¹ := by
  sorry

end inverse_difference_equals_negative_reciprocal_l3208_320834


namespace composite_divides_factorial_l3208_320808

theorem composite_divides_factorial (k n : ℕ) (P_k : ℕ) : 
  k ≥ 14 →
  P_k < k →
  (∀ p, p < k ∧ Nat.Prime p → p ≤ P_k) →
  Nat.Prime P_k →
  P_k ≥ 3 * k / 4 →
  ¬Nat.Prime n →
  n > 2 * P_k →
  n ∣ Nat.factorial (n - k) :=
by sorry

end composite_divides_factorial_l3208_320808


namespace point_transformation_l3208_320837

/-- Rotate a point (x,y) by 180° around (h,k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2*h - x, 2*k - y)

/-- Reflect a point (x,y) about the line y = -x -/
def reflectAboutNegativeX (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem point_transformation (a b : ℝ) :
  let p₁ := rotate180 a b 2 4
  let p₂ := reflectAboutNegativeX p₁.1 p₁.2
  p₂ = (-1, 4) → a - b = -9 := by
sorry

end point_transformation_l3208_320837


namespace rectangle_triangle_altitude_l3208_320889

theorem rectangle_triangle_altitude (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) :
  let rectangle_area := a * b
  let triangle_leg := 2 * b
  let triangle_hypotenuse := Real.sqrt (a^2 + triangle_leg^2)
  let triangle_area := (1/2) * a * triangle_leg
  triangle_area = rectangle_area →
  (2 * rectangle_area) / triangle_hypotenuse = (2 * a * b) / Real.sqrt (a^2 + 4 * b^2) :=
by sorry

end rectangle_triangle_altitude_l3208_320889


namespace fourth_segment_length_l3208_320846

/-- Represents an acute triangle with two altitudes dividing opposite sides -/
structure AcuteTriangleWithAltitudes where
  -- Lengths of segments created by altitudes
  segment1 : ℝ
  segment2 : ℝ
  segment3 : ℝ
  segment4 : ℝ
  -- Conditions
  acute : segment1 > 0 ∧ segment2 > 0 ∧ segment3 > 0 ∧ segment4 > 0
  segment1_eq : segment1 = 4
  segment2_eq : segment2 = 6
  segment3_eq : segment3 = 3

/-- Theorem stating that the fourth segment length is 3 -/
theorem fourth_segment_length (t : AcuteTriangleWithAltitudes) : t.segment4 = 3 := by
  sorry

end fourth_segment_length_l3208_320846


namespace sharp_value_theorem_l3208_320822

/-- Define the function # -/
def sharp (k : ℚ) (p : ℚ) : ℚ := k * p + 20

/-- Main theorem -/
theorem sharp_value_theorem :
  ∀ k : ℚ, 
  (sharp k (sharp k (sharp k 18)) = -4) → 
  k = -4/3 := by
sorry

end sharp_value_theorem_l3208_320822


namespace intersection_of_A_and_B_l3208_320886

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-2, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l3208_320886


namespace log_difference_equals_four_l3208_320875

theorem log_difference_equals_four (a : ℝ) (h : a > 0) :
  Real.log (100 * a) / Real.log 10 - Real.log (a / 100) / Real.log 10 = 4 := by
  sorry

end log_difference_equals_four_l3208_320875


namespace expression_equals_24_times_30_to_1001_l3208_320870

theorem expression_equals_24_times_30_to_1001 :
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 :=
by sorry

end expression_equals_24_times_30_to_1001_l3208_320870


namespace max_value_inequality_max_value_attained_l3208_320836

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 5 * y < 75) :
  x * y * (75 - 2 * x - 5 * y) ≤ 1562.5 := by
  sorry

theorem max_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + 5 * y < 75 ∧
  x * y * (75 - 2 * x - 5 * y) > 1562.5 - ε := by
  sorry

end max_value_inequality_max_value_attained_l3208_320836


namespace isosceles_triangle_cut_l3208_320820

-- Define the triangle PQR
structure Triangle :=
  (area : ℝ)
  (altitude : ℝ)

-- Define the line segment ST and resulting areas
structure Segment :=
  (length : ℝ)
  (trapezoid_area : ℝ)

-- Define the theorem
theorem isosceles_triangle_cut (PQR : Triangle) (ST : Segment) :
  PQR.area = 144 →
  PQR.altitude = 24 →
  ST.trapezoid_area = 108 →
  ST.length = 6 := by
  sorry

end isosceles_triangle_cut_l3208_320820


namespace alpha_value_l3208_320862

theorem alpha_value (α γ : ℂ) 
  (h1 : (α + γ).re > 0)
  (h2 : (Complex.I * (α - 3 * γ)).re > 0)
  (h3 : γ = 4 + 3 * Complex.I) :
  α = 10.5 + 0.5 * Complex.I :=
by sorry

end alpha_value_l3208_320862


namespace orange_juice_distribution_l3208_320860

theorem orange_juice_distribution (C : ℝ) (h : C > 0) : 
  let juice_volume := (2 / 3) * C
  let num_cups := 6
  let juice_per_cup := juice_volume / num_cups
  juice_per_cup / C * 100 = 100 / 9 := by sorry

end orange_juice_distribution_l3208_320860


namespace ceiling_squared_fraction_l3208_320839

theorem ceiling_squared_fraction : ⌈((-7/4 + 1/4) : ℚ)^2⌉ = 3 := by sorry

end ceiling_squared_fraction_l3208_320839


namespace units_digit_37_power_l3208_320806

/-- The units digit of 37^(5*(14^14)) is 1 -/
theorem units_digit_37_power : ∃ k : ℕ, 37^(5*(14^14)) ≡ 1 [ZMOD 10] := by
  sorry

end units_digit_37_power_l3208_320806


namespace square_field_area_l3208_320829

theorem square_field_area (side_length : ℝ) (h : side_length = 16) : 
  side_length * side_length = 256 := by
  sorry

end square_field_area_l3208_320829


namespace coffee_price_proof_l3208_320856

/-- The regular price of coffee in dollars per pound -/
def regular_price : ℝ := 40

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.6

/-- The price of a discounted quarter-pound package with a free chocolate bar -/
def discounted_quarter_pound_price : ℝ := 4

theorem coffee_price_proof :
  regular_price * (1 - discount_rate) / 4 = discounted_quarter_pound_price :=
by sorry

end coffee_price_proof_l3208_320856


namespace absent_student_grade_calculation_l3208_320884

/-- Given a class where one student was initially absent for a test, prove that the absent student's grade can be determined from the class averages before and after including their score. -/
theorem absent_student_grade_calculation (total_students : ℕ) 
  (initial_students : ℕ) (initial_average : ℚ) (final_average : ℚ) 
  (h1 : total_students = 25) 
  (h2 : initial_students = 24)
  (h3 : initial_average = 82)
  (h4 : final_average = 84) :
  (total_students : ℚ) * final_average - (initial_students : ℚ) * initial_average = 132 := by
  sorry

end absent_student_grade_calculation_l3208_320884


namespace min_cut_length_for_non_triangle_l3208_320874

/-- Given three sticks of lengths 9, 18, and 21 inches, this theorem proves that
    the minimum integral length that can be cut from each stick to prevent
    the remaining pieces from forming a triangle is 6 inches. -/
theorem min_cut_length_for_non_triangle : ∃ (x : ℕ),
  (∀ y : ℕ, y < x → (9 - y) + (18 - y) > 21 - y) ∧
  (9 - x) + (18 - x) ≤ 21 - x ∧
  x = 6 :=
sorry

end min_cut_length_for_non_triangle_l3208_320874


namespace basketball_players_l3208_320816

theorem basketball_players (cricket : ℕ) (both : ℕ) (total : ℕ)
  (h1 : cricket = 8)
  (h2 : both = 3)
  (h3 : total = 12) :
  ∃ basketball : ℕ, basketball = total - cricket + both :=
by
  sorry

end basketball_players_l3208_320816


namespace isosceles_diagonal_implies_two_equal_among_four_l3208_320803

-- Define a convex n-gon
structure ConvexNGon where
  n : ℕ
  sides : Fin n → ℝ
  is_convex : Bool
  n_gt_4 : n > 4

-- Define the isosceles triangle property
def isosceles_diagonal_property (polygon : ConvexNGon) : Prop :=
  ∀ (i j : Fin polygon.n), i ≠ j → 
    ∃ (k : Fin polygon.n), k ≠ i ∧ k ≠ j ∧ 
      (polygon.sides i = polygon.sides k ∨ polygon.sides j = polygon.sides k)

-- Define the property of having at least two equal sides among any four
def two_equal_among_four (polygon : ConvexNGon) : Prop :=
  ∀ (i j k l : Fin polygon.n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i →
    polygon.sides i = polygon.sides j ∨ polygon.sides i = polygon.sides k ∨
    polygon.sides i = polygon.sides l ∨ polygon.sides j = polygon.sides k ∨
    polygon.sides j = polygon.sides l ∨ polygon.sides k = polygon.sides l

-- The theorem to be proved
theorem isosceles_diagonal_implies_two_equal_among_four 
  (polygon : ConvexNGon) (h : isosceles_diagonal_property polygon) :
  two_equal_among_four polygon := by
  sorry

end isosceles_diagonal_implies_two_equal_among_four_l3208_320803


namespace total_gain_percentage_approx_l3208_320896

/-- Calculates the total gain percentage for three items given their purchase and sale prices -/
def total_gain_percentage (cycle_cp cycle_sp scooter_cp scooter_sp skateboard_cp skateboard_sp : ℚ) : ℚ :=
  let total_gain := (cycle_sp - cycle_cp) + (scooter_sp - scooter_cp) + (skateboard_sp - skateboard_cp)
  let total_cost := cycle_cp + scooter_cp + skateboard_cp
  (total_gain / total_cost) * 100

/-- The total gain percentage for the given items is approximately 28.18% -/
theorem total_gain_percentage_approx :
  ∃ ε > 0, abs (total_gain_percentage 900 1260 4500 5400 1200 1800 - 2818/100) < ε :=
sorry

end total_gain_percentage_approx_l3208_320896


namespace final_sum_of_numbers_l3208_320882

theorem final_sum_of_numbers (n : ℕ) (h1 : n = 2013) : 
  ∃ (a b c d : ℕ), 
    (a * b * c * d = 27) ∧ 
    (a + b + c + d ≡ (n * (n + 1) / 2) [MOD 9]) ∧
    (a + b + c + d = 30) := by
  sorry

end final_sum_of_numbers_l3208_320882


namespace increase_in_circumference_l3208_320857

/-- The increase in circumference of a circle when its diameter increases by π units -/
theorem increase_in_circumference (d : ℝ) : 
  let original_circumference := π * d
  let new_circumference := π * (d + π)
  let increase := new_circumference - original_circumference
  increase = π^2 := by sorry

end increase_in_circumference_l3208_320857


namespace female_average_score_l3208_320817

theorem female_average_score (total_average : ℝ) (male_average : ℝ) (male_count : ℕ) (female_count : ℕ) 
  (h1 : total_average = 90)
  (h2 : male_average = 84)
  (h3 : male_count = 8)
  (h4 : female_count = 24) :
  let total_count := male_count + female_count
  let total_sum := total_average * total_count
  let male_sum := male_average * male_count
  let female_sum := total_sum - male_sum
  female_sum / female_count = 92 := by sorry

end female_average_score_l3208_320817


namespace range_of_m_for_quadratic_equation_l3208_320804

theorem range_of_m_for_quadratic_equation (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ioo (-1) 0 ∧ x₂ ∈ Set.Ioi 3 ∧
    x₁^2 - 2*m*x₁ + m - 3 = 0 ∧ x₂^2 - 2*m*x₂ + m - 3 = 0) →
  m ∈ Set.Ioo (6/5) 3 :=
by sorry

end range_of_m_for_quadratic_equation_l3208_320804


namespace average_of_five_quantities_l3208_320823

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 14) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 8 := by
  sorry

end average_of_five_quantities_l3208_320823


namespace negative_fraction_comparison_l3208_320854

theorem negative_fraction_comparison : -1/3 < -1/5 := by
  sorry

end negative_fraction_comparison_l3208_320854


namespace triangle_formation_constraint_l3208_320847

/-- A line in 2D space represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines form a triangle -/
def form_triangle (l1 l2 l3 : Line) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (l1.a * x1 + l1.b * y1 = l1.c) ∧
    (l2.a * x2 + l2.b * y2 = l2.c) ∧
    (l3.a * x3 + l3.b * y3 = l3.c) ∧
    ((x1 ≠ x2) ∨ (y1 ≠ y2)) ∧
    ((x2 ≠ x3) ∨ (y2 ≠ y3)) ∧
    ((x3 ≠ x1) ∨ (y3 ≠ y1))

theorem triangle_formation_constraint (a : ℝ) :
  let l1 : Line := ⟨1, 1, 0⟩
  let l2 : Line := ⟨1, -1, 0⟩
  let l3 : Line := ⟨1, a, 3⟩
  form_triangle l1 l2 l3 → a ≠ 1 ∧ a ≠ -1 :=
by sorry

end triangle_formation_constraint_l3208_320847


namespace quadratic_negative_root_l3208_320890

theorem quadratic_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 := by
  sorry

end quadratic_negative_root_l3208_320890


namespace milly_study_time_l3208_320892

/-- Calculates the total study time for Milly given her homework durations. -/
theorem milly_study_time (math_time : ℕ) (math_time_eq : math_time = 60) :
  let geography_time := math_time / 2
  let science_time := (math_time + geography_time) / 2
  math_time + geography_time + science_time = 135 := by
  sorry

end milly_study_time_l3208_320892


namespace angle_half_in_fourth_quadrant_l3208_320835

/-- Represents the four quadrants of the coordinate plane. -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines if an angle is in a specific quadrant. -/
def in_quadrant (angle : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.first => 0 < angle ∧ angle < Real.pi / 2
  | Quadrant.second => Real.pi / 2 < angle ∧ angle < Real.pi
  | Quadrant.third => Real.pi < angle ∧ angle < 3 * Real.pi / 2
  | Quadrant.fourth => 3 * Real.pi / 2 < angle ∧ angle < 2 * Real.pi

theorem angle_half_in_fourth_quadrant (α : ℝ) 
  (h1 : in_quadrant α Quadrant.third) 
  (h2 : |Real.sin (α/2)| = -Real.sin (α/2)) : 
  in_quadrant (α/2) Quadrant.fourth :=
sorry

end angle_half_in_fourth_quadrant_l3208_320835


namespace calculation_proof_l3208_320819

theorem calculation_proof : -50 * 3 - (-2.5) / 0.1 = -125 := by
  sorry

end calculation_proof_l3208_320819


namespace system_of_equations_l3208_320830

theorem system_of_equations (x y : ℚ) 
  (eq1 : 4 * x + y = 8) 
  (eq2 : 3 * x - 4 * y = 5) : 
  7 * x - 3 * y = 247 / 19 := by
  sorry

end system_of_equations_l3208_320830


namespace functional_equation_solutions_l3208_320853

/-- A real-valued function that satisfies the given functional equation. -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2002 * x - f 0) = 2002 * x^2

/-- The theorem stating the only two functions that satisfy the functional equation. -/
theorem functional_equation_solutions :
  ∀ f : ℝ → ℝ, SatisfyingFunction f ↔ 
    (∀ x : ℝ, f x = x^2 / 2002) ∨ 
    (∀ x : ℝ, f x = x^2 / 2002 + 2 * x + 2002) :=
sorry

end functional_equation_solutions_l3208_320853


namespace linear_function_characterization_l3208_320873

/-- A function f: ℚ → ℚ satisfies the arithmetic progression property if
    f(x) + f(t) = f(y) + f(z) for all rational x < y < z < t in arithmetic progression -/
def ArithmeticProgressionProperty (f : ℚ → ℚ) : Prop :=
  ∀ (x y z t : ℚ), x < y ∧ y < z ∧ z < t ∧ (y - x = z - y) ∧ (z - y = t - z) →
    f x + f t = f y + f z

/-- The main theorem: if f satisfies the arithmetic progression property,
    then f is a linear function -/
theorem linear_function_characterization (f : ℚ → ℚ) 
  (h : ArithmeticProgressionProperty f) :
  ∃ (c b : ℚ), ∀ (q : ℚ), f q = c * q + b :=
sorry

end linear_function_characterization_l3208_320873


namespace angle_bisector_theorem_application_l3208_320859

theorem angle_bisector_theorem_application (DE DF EF D₁F D₁E XY XZ YZ X₁Z X₁Y XX₁ : ℝ) : 
  DE = 13 →
  DF = 5 →
  EF = (DE^2 - DF^2).sqrt →
  D₁F / D₁E = DF / EF →
  D₁F + D₁E = EF →
  XY = D₁E →
  XZ = D₁F →
  YZ = (XY^2 - XZ^2).sqrt →
  X₁Z / X₁Y = XZ / XY →
  X₁Z + X₁Y = YZ →
  XX₁ = XZ - X₁Z →
  XX₁ = 0 := by
sorry

#eval "QED"

end angle_bisector_theorem_application_l3208_320859


namespace sixth_root_of_12984301300421_l3208_320831

theorem sixth_root_of_12984301300421 : 
  (12984301300421 : ℝ) ^ (1/6 : ℝ) = 51 := by sorry

end sixth_root_of_12984301300421_l3208_320831


namespace mike_weekly_spending_l3208_320849

/-- Given that Mike made $14 mowing lawns and $26 weed eating, and the money would last him 8 weeks,
    prove that he spent $5 per week. -/
theorem mike_weekly_spending (lawn_money : ℕ) (weed_money : ℕ) (weeks : ℕ) 
  (h1 : lawn_money = 14)
  (h2 : weed_money = 26)
  (h3 : weeks = 8) :
  (lawn_money + weed_money) / weeks = 5 := by
  sorry

end mike_weekly_spending_l3208_320849


namespace circular_permutation_sum_l3208_320844

def CircularPermutation (xs : List ℕ) : Prop :=
  xs.length = 6 ∧ xs.toFinset = {1, 2, 3, 4, 6}

def CircularProduct (xs : List ℕ) : ℕ :=
  (List.zip xs (xs.rotate 1)).map (λ (a, b) => a * b) |>.sum

def MaxCircularProduct : ℕ := sorry

def MaxCircularProductPermutations : ℕ := sorry

theorem circular_permutation_sum :
  MaxCircularProduct + MaxCircularProductPermutations = 96 := by sorry

end circular_permutation_sum_l3208_320844


namespace samuel_coaching_fee_l3208_320863

/-- Calculate the total coaching fee for Samuel --/
theorem samuel_coaching_fee :
  let days_in_period : ℕ := 307 -- Days from Jan 1 to Nov 4 in a non-leap year
  let holidays : ℕ := 5
  let daily_fee : ℕ := 23
  let discount_period : ℕ := 30
  let discount_rate : ℚ := 1 / 10

  let coaching_days : ℕ := days_in_period - holidays
  let full_discount_periods : ℕ := coaching_days / discount_period
  let base_fee : ℕ := coaching_days * daily_fee
  let discount_per_period : ℚ := (discount_period * daily_fee : ℚ) * discount_rate
  let total_discount : ℚ := discount_per_period * full_discount_periods
  
  (base_fee : ℚ) - total_discount = 6256 := by
  sorry

end samuel_coaching_fee_l3208_320863


namespace range_of_a_l3208_320821

def A (a : ℝ) := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B := {x : ℝ | x < 0 ∨ x > 19}

theorem range_of_a (a : ℝ) : 
  (A a ⊆ (A a ∩ B)) → (a < 6 ∨ a > 9) := by
  sorry

end range_of_a_l3208_320821


namespace max_value_x_cubed_over_y_fourth_l3208_320805

theorem max_value_x_cubed_over_y_fourth (x y : ℝ) 
  (h1 : 3 ≤ x * y^2 ∧ x * y^2 ≤ 8) 
  (h2 : 4 ≤ x^2 / y ∧ x^2 / y ≤ 9) : 
  ∃ (M : ℝ), M = 27 ∧ x^3 / y^4 ≤ M ∧ ∃ (x₀ y₀ : ℝ), 
    3 ≤ x₀ * y₀^2 ∧ x₀ * y₀^2 ≤ 8 ∧ 
    4 ≤ x₀^2 / y₀ ∧ x₀^2 / y₀ ≤ 9 ∧ 
    x₀^3 / y₀^4 = M :=
by sorry

end max_value_x_cubed_over_y_fourth_l3208_320805


namespace probability_three_primes_l3208_320868

-- Define a 12-sided die
def Die := Finset (Fin 12)

-- Define the set of prime numbers on a 12-sided die
def PrimeNumbers : Finset (Fin 12) := {2, 3, 5, 7, 11}

-- Define the probability of rolling a prime number on a single die
def ProbPrime : ℚ := (PrimeNumbers.card : ℚ) / 12

-- Define the probability of not rolling a prime number on a single die
def ProbNotPrime : ℚ := 1 - ProbPrime

-- Define the number of dice
def NumDice : ℕ := 4

-- Define the number of dice that should show a prime
def NumPrimeDice : ℕ := 3

-- Theorem statement
theorem probability_three_primes :
  (NumDice.choose NumPrimeDice : ℚ) * ProbPrime ^ NumPrimeDice * ProbNotPrime ^ (NumDice - NumPrimeDice) = 875 / 5184 :=
sorry

end probability_three_primes_l3208_320868


namespace intersection_nonempty_range_union_equals_B_l3208_320841

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + 2*(m+1)*x + m^2 - 1 = 2}

-- Theorem for part (1)
theorem intersection_nonempty_range (m : ℝ) : 
  (A ∩ B m).Nonempty → m = -Real.sqrt 3 :=
sorry

-- Theorem for part (2)
theorem union_equals_B (m : ℝ) : 
  A ∪ B m = B m → m = -Real.sqrt 3 :=
sorry

end intersection_nonempty_range_union_equals_B_l3208_320841


namespace athlete_stability_l3208_320879

/-- Represents an athlete's shooting performance -/
structure Athlete where
  average_score : ℝ
  variance : ℝ
  shot_count : ℕ

/-- Defines when one athlete's performance is more stable than another's -/
def more_stable (a b : Athlete) : Prop :=
  a.variance < b.variance

theorem athlete_stability 
  (A B : Athlete)
  (h1 : A.average_score = B.average_score)
  (h2 : A.shot_count = 10)
  (h3 : B.shot_count = 10)
  (h4 : A.variance = 0.4)
  (h5 : B.variance = 2)
  : more_stable A B :=
sorry

end athlete_stability_l3208_320879


namespace imaginary_part_of_z_l3208_320850

theorem imaginary_part_of_z (z : ℂ) (h : 1 + z * Complex.I = z - 2 * Complex.I) :
  z.im = 3 / 2 := by
  sorry

end imaginary_part_of_z_l3208_320850


namespace list_size_theorem_l3208_320812

theorem list_size_theorem (L : List ℝ) (n : ℝ) : 
  L.Nodup → 
  n ∈ L → 
  n = 5 * ((L.sum - n) / (L.length - 1)) → 
  n = 0.2 * L.sum → 
  L.length = 21 :=
sorry

end list_size_theorem_l3208_320812


namespace expression_equals_one_l3208_320885

theorem expression_equals_one (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) : 
  ((((x + 2)^3 * (x^2 - 2*x + 2)^3) / (x^3 + 8)^3)^2 * 
   (((x - 2)^3 * (x^2 + 2*x + 2)^3) / (x^3 - 8)^3)^2) = 1 := by
  sorry

end expression_equals_one_l3208_320885


namespace profit_distribution_l3208_320897

/-- Profit distribution in a business partnership --/
theorem profit_distribution (a b c : ℕ) (profit_b : ℕ) : 
  a = 8000 → b = 10000 → c = 12000 → profit_b = 4000 →
  ∃ (profit_a profit_c : ℕ),
    profit_a * b = profit_b * a ∧
    profit_c * b = profit_b * c ∧
    profit_c - profit_a = 1600 :=
by sorry

end profit_distribution_l3208_320897


namespace correct_propositions_l3208_320833

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the operations and relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_not_in_plane : Line → Plane → Prop)
variable (line_parallel_line : Line → Line → Prop)

-- Axioms for the properties of these operations
axiom perpendicular_sym {α β : Plane} : perpendicular α β → perpendicular β α
axiom parallel_sym {α β : Plane} : parallel α β → parallel β α
axiom line_parallel_plane_sym {l : Line} {α : Plane} : line_parallel_plane l α → line_parallel_plane l α

-- The theorem to be proved
theorem correct_propositions 
  (m n : Line) (α β γ : Plane) : 
  (perpendicular α β ∧ line_perpendicular_plane m β ∧ line_not_in_plane m α → line_parallel_plane m α) ∧
  (parallel α β ∧ line_in_plane m α → line_parallel_plane m β) ∧
  ¬(perpendicular α β ∧ line_parallel_line n m → line_parallel_plane n α ∧ line_parallel_plane n β) ∧
  ¬(perpendicular α β ∧ perpendicular α γ → parallel β γ) :=
sorry

end correct_propositions_l3208_320833


namespace largest_m_binomial_sum_l3208_320842

theorem largest_m_binomial_sum (m : ℕ) : (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ 6 :=
sorry

end largest_m_binomial_sum_l3208_320842


namespace apple_consumption_duration_l3208_320893

theorem apple_consumption_duration (apples_per_box : ℕ) (num_boxes : ℕ) (num_people : ℕ) (apples_per_person_per_day : ℕ) :
  apples_per_box = 14 →
  num_boxes = 3 →
  num_people = 2 →
  apples_per_person_per_day = 1 →
  (apples_per_box * num_boxes) / (num_people * apples_per_person_per_day * 7) = 3 := by
  sorry

end apple_consumption_duration_l3208_320893


namespace inequality_theorem_l3208_320845

open Set

-- Define the interval (0,+∞)
def openPositiveReals : Set ℝ := {x : ℝ | x > 0}

-- Define the properties of functions f and g
def hasContinuousDerivative (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ Differentiable ℝ f

-- Define the inequality condition
def satisfiesInequality (f g : ℝ → ℝ) : Prop :=
  ∀ x ∈ openPositiveReals, f x > x * (deriv f x) - x^2 * (deriv g x)

-- Theorem statement
theorem inequality_theorem (f g : ℝ → ℝ) 
  (hf : hasContinuousDerivative f) (hg : hasContinuousDerivative g)
  (h_ineq : satisfiesInequality f g) :
  2 * g 2 + 2 * f 1 > f 2 + 2 * g 1 :=
sorry

end inequality_theorem_l3208_320845


namespace expand_expression_l3208_320871

theorem expand_expression (x : ℝ) : (17 * x + 12) * (3 * x) = 51 * x^2 + 36 * x := by
  sorry

end expand_expression_l3208_320871


namespace dima_numbers_l3208_320899

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) ∨
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 4) ∨ (a = 6 ∧ b = 3)

theorem dima_numbers (a b : ℕ) :
  (4 * a = b + (a + b) + (a * b)) ∨ (4 * b = a + (a + b) + (a * b)) ∨
  (4 * (a + b) = a + b + (a * b)) →
  is_valid_pair a b := by
  sorry

end dima_numbers_l3208_320899


namespace rectangular_prism_width_l3208_320825

/-- Given a rectangular prism with length 4 units, height 10 units, and diagonal 14 units,
    prove that its width is 4√5 units. -/
theorem rectangular_prism_width (l h d w : ℝ) : 
  l = 4 → h = 10 → d = 14 → d^2 = l^2 + w^2 + h^2 → w = 4 * Real.sqrt 5 := by
  sorry

end rectangular_prism_width_l3208_320825


namespace sqrt_equation_solution_l3208_320898

theorem sqrt_equation_solution (a : ℝ) :
  Real.sqrt 3 * (a * Real.sqrt 6) = 6 * Real.sqrt 2 → a = 2 := by
  sorry

end sqrt_equation_solution_l3208_320898


namespace prob_even_sum_half_l3208_320827

/-- Represents a die with a specified number of faces -/
structure Die where
  faces : ℕ
  face_range : faces > 0

/-- The probability of getting an even sum when rolling two dice -/
def prob_even_sum (d1 d2 : Die) : ℚ :=
  let even_outcomes := (d1.faces.div 2) * (d2.faces.div 2) + 
                       ((d1.faces + 1).div 2) * ((d2.faces + 1).div 2)
  even_outcomes / (d1.faces * d2.faces)

/-- Theorem stating that the probability of an even sum with the specified dice is 1/2 -/
theorem prob_even_sum_half :
  let d1 : Die := ⟨8, by norm_num⟩
  let d2 : Die := ⟨6, by norm_num⟩
  prob_even_sum d1 d2 = 1/2 := by
  sorry

end prob_even_sum_half_l3208_320827


namespace only_fourth_equation_has_real_roots_l3208_320883

-- Define the discriminant function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Define a function to check if a quadratic equation has real roots
def hasRealRoots (a b c : ℝ) : Prop := discriminant a b c ≥ 0

-- Theorem statement
theorem only_fourth_equation_has_real_roots :
  ¬(hasRealRoots 1 0 1) ∧
  ¬(hasRealRoots 1 1 1) ∧
  ¬(hasRealRoots 1 (-1) 1) ∧
  hasRealRoots 1 (-1) (-1) :=
sorry

end only_fourth_equation_has_real_roots_l3208_320883


namespace jake_debt_work_hours_l3208_320855

def total_hours_worked (initial_debt_A initial_debt_B initial_debt_C : ℕ)
                       (payment_A payment_B payment_C : ℕ)
                       (rate_A rate_B rate_C : ℕ) : ℕ :=
  let remaining_debt_A := initial_debt_A - payment_A
  let remaining_debt_B := initial_debt_B - payment_B
  let remaining_debt_C := initial_debt_C - payment_C
  let hours_A := remaining_debt_A / rate_A
  let hours_B := remaining_debt_B / rate_B
  let hours_C := remaining_debt_C / rate_C
  hours_A + hours_B + hours_C

theorem jake_debt_work_hours :
  total_hours_worked 150 200 250 60 80 100 15 20 25 = 18 := by
  sorry

end jake_debt_work_hours_l3208_320855


namespace correct_expression_l3208_320881

/-- A type representing mathematical expressions --/
inductive MathExpression
  | DivideABC : MathExpression
  | MixedFraction : MathExpression
  | MultiplyAB : MathExpression
  | ThreeM : MathExpression

/-- A predicate that determines if an expression is correctly written --/
def is_correctly_written (e : MathExpression) : Prop :=
  match e with
  | MathExpression.ThreeM => True
  | _ => False

/-- The set of given expressions --/
def expression_set : Set MathExpression :=
  {MathExpression.DivideABC, MathExpression.MixedFraction, 
   MathExpression.MultiplyAB, MathExpression.ThreeM}

theorem correct_expression :
  ∃ (e : MathExpression), e ∈ expression_set ∧ is_correctly_written e :=
by sorry

end correct_expression_l3208_320881


namespace inequalities_hold_l3208_320832

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^2 + b^2 ≥ 2 ∧ 1/a + 1/b ≥ 2 := by
  sorry

end inequalities_hold_l3208_320832
