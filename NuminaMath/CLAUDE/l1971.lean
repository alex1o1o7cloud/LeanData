import Mathlib

namespace NUMINAMATH_CALUDE_mean_of_car_counts_l1971_197195

theorem mean_of_car_counts : 
  let counts : List ℝ := [30, 14, 14, 21, 25]
  (counts.sum / counts.length : ℝ) = 20.8 := by
sorry

end NUMINAMATH_CALUDE_mean_of_car_counts_l1971_197195


namespace NUMINAMATH_CALUDE_triangle_side_length_l1971_197144

theorem triangle_side_length 
  (A B C : Real) 
  (a b c : Real) 
  (area : Real) :
  area = Real.sqrt 3 →
  B = 60 * π / 180 →
  a^2 + c^2 = 3 * a * c →
  b = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1971_197144


namespace NUMINAMATH_CALUDE_line_parallel_in_perp_planes_l1971_197136

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Define the relation of a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_in_perp_planes
  (α β : Plane) (l m n : Line)
  (h1 : perp_plane α β)
  (h2 : l = intersection α β)
  (h3 : in_plane n β)
  (h4 : perp_line n l)
  (h5 : perp_line_plane m α) :
  parallel m n :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_in_perp_planes_l1971_197136


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1971_197145

/-- Given a line L1 with equation x - 2y + m = 0 and a point P (-1, 3),
    this theorem states that the line L2 with equation 2x + y - 1 = 0
    passes through P and is perpendicular to L1. -/
theorem perpendicular_line_through_point (m : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ x - 2*y + m = 0
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 2*x + y - 1 = 0
  let P : ℝ × ℝ := (-1, 3)
  (L2 P.1 P.2) ∧                   -- L2 passes through P
  (∀ x1 y1 x2 y2, L1 x1 y1 → L2 x2 y2 →
    (x2 - x1) * (x1 - 2*y1) + (y2 - y1) * (-2*x1 - y1) = 0) -- L2 is perpendicular to L1
  := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1971_197145


namespace NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l1971_197143

theorem min_values_xy_and_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 9/y = 1) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → x * y ≤ a * b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → x + 2*y ≤ a + 2*b) ∧
  x * y = 36 ∧
  x + 2*y = 19 + 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_values_xy_and_x_plus_2y_l1971_197143


namespace NUMINAMATH_CALUDE_small_cube_edge_length_l1971_197186

theorem small_cube_edge_length (large_cube_edge : ℕ) (small_cube_edge : ℕ) : 
  large_cube_edge = 12 →
  (large_cube_edge / small_cube_edge) > 0 →
  6 * ((large_cube_edge / small_cube_edge - 2) ^ 2) = 12 * (large_cube_edge / small_cube_edge - 2) →
  small_cube_edge = 3 := by
sorry

end NUMINAMATH_CALUDE_small_cube_edge_length_l1971_197186


namespace NUMINAMATH_CALUDE_smallest_positive_time_for_104_degrees_l1971_197111

def temperature (t : ℝ) : ℝ := -t^2 + 16*t + 40

theorem smallest_positive_time_for_104_degrees :
  let t := 8 + 8 * Real.sqrt 2
  (∀ s, s > 0 ∧ temperature s = 104 → s ≥ t) ∧ temperature t = 104 ∧ t > 0 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_time_for_104_degrees_l1971_197111


namespace NUMINAMATH_CALUDE_basketball_score_ratio_l1971_197157

/-- Given the scoring information for two basketball teams, prove the ratio of 2-pointers scored by the opponents to Mark's team. -/
theorem basketball_score_ratio :
  let marks_two_pointers : ℕ := 25
  let marks_three_pointers : ℕ := 8
  let marks_free_throws : ℕ := 10
  let opponents_three_pointers : ℕ := marks_three_pointers / 2
  let opponents_free_throws : ℕ := marks_free_throws / 2
  let total_points : ℕ := 201
  ∃ (x : ℚ),
    (2 * marks_two_pointers + 3 * marks_three_pointers + marks_free_throws) +
    (2 * (x * marks_two_pointers) + 3 * opponents_three_pointers + opponents_free_throws) = total_points ∧
    x = 2 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_ratio_l1971_197157


namespace NUMINAMATH_CALUDE_luke_lawn_mowing_earnings_l1971_197181

theorem luke_lawn_mowing_earnings :
  ∀ (L : ℝ),
  (∃ (total_earnings : ℝ),
    total_earnings = L + 18 ∧
    total_earnings = 3 * 9) →
  L = 9 := by
sorry

end NUMINAMATH_CALUDE_luke_lawn_mowing_earnings_l1971_197181


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1971_197172

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 2 + a 8 = 12) : 
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1971_197172


namespace NUMINAMATH_CALUDE_halfway_point_between_fractions_l1971_197194

theorem halfway_point_between_fractions :
  let a := (1 : ℚ) / 9
  let b := (1 : ℚ) / 11
  let midpoint := (a + b) / 2
  midpoint = 10 / 99 := by sorry

end NUMINAMATH_CALUDE_halfway_point_between_fractions_l1971_197194


namespace NUMINAMATH_CALUDE_money_difference_l1971_197188

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- Mrs. Hilt's coin counts -/
def mrs_hilt_coins : Fin 3 → ℕ
| 0 => 2  -- pennies
| 1 => 2  -- nickels
| 2 => 2  -- dimes
| _ => 0

/-- Jacob's coin counts -/
def jacob_coins : Fin 3 → ℕ
| 0 => 4  -- pennies
| 1 => 1  -- nickel
| 2 => 1  -- dime
| _ => 0

/-- The value of a coin type in dollars -/
def coin_value : Fin 3 → ℚ
| 0 => penny_value
| 1 => nickel_value
| 2 => dime_value
| _ => 0

/-- Calculate the total value of coins -/
def total_value (coins : Fin 3 → ℕ) : ℚ :=
  (coins 0 : ℚ) * penny_value + (coins 1 : ℚ) * nickel_value + (coins 2 : ℚ) * dime_value

theorem money_difference :
  total_value mrs_hilt_coins - total_value jacob_coins = 13 / 100 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l1971_197188


namespace NUMINAMATH_CALUDE_selina_shorts_sold_l1971_197187

/-- Represents the number of pairs of shorts Selina sold -/
def shorts_sold : ℕ := sorry

/-- The price of a pair of pants in dollars -/
def pants_price : ℕ := 5

/-- The price of a pair of shorts in dollars -/
def shorts_price : ℕ := 3

/-- The price of a shirt in dollars -/
def shirt_price : ℕ := 4

/-- The number of pairs of pants Selina sold -/
def pants_sold : ℕ := 3

/-- The number of shirts Selina sold -/
def shirts_sold : ℕ := 5

/-- The price of each new shirt Selina bought -/
def new_shirt_price : ℕ := 10

/-- The number of new shirts Selina bought -/
def new_shirts_bought : ℕ := 2

/-- The amount of money Selina left the store with -/
def money_left : ℕ := 30

theorem selina_shorts_sold :
  shorts_sold = 5 ∧
  pants_sold * pants_price + shirts_sold * shirt_price + shorts_sold * shorts_price =
    money_left + new_shirts_bought * new_shirt_price :=
by sorry

end NUMINAMATH_CALUDE_selina_shorts_sold_l1971_197187


namespace NUMINAMATH_CALUDE_selectPeopleCount_l1971_197135

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to select 4 people from 4 boys and 3 girls, 
    ensuring both boys and girls are included -/
def selectPeople : ℕ :=
  choose 4 3 * choose 3 1 + 
  choose 4 2 * choose 3 2 + 
  choose 4 1 * choose 3 3

theorem selectPeopleCount : selectPeople = 34 := by sorry

end NUMINAMATH_CALUDE_selectPeopleCount_l1971_197135


namespace NUMINAMATH_CALUDE_joan_toy_cars_cost_l1971_197112

theorem joan_toy_cars_cost (total_toys cost_skateboard cost_trucks : ℚ)
  (h1 : total_toys = 25.62)
  (h2 : cost_skateboard = 4.88)
  (h3 : cost_trucks = 5.86) :
  total_toys - cost_skateboard - cost_trucks = 14.88 := by
  sorry

end NUMINAMATH_CALUDE_joan_toy_cars_cost_l1971_197112


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1971_197140

/-- Given a geometric sequence {a_n} with S_n being the sum of its first n terms,
    if a_6 = 8a_3, then S_6 / S_3 = 9 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
    (h_sum : ∀ n, S n = a 1 * (1 - (a 2 / a 1)^n) / (1 - a 2 / a 1)) 
    (h_ratio : a 6 = 8 * a 3) : 
  S 6 / S 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1971_197140


namespace NUMINAMATH_CALUDE_sum_of_series_equals_three_fourths_l1971_197134

/-- The sum of the infinite series ∑(k=1 to ∞) k/3^k is equal to 3/4 -/
theorem sum_of_series_equals_three_fourths :
  (∑' k : ℕ+, (k : ℝ) / (3 : ℝ) ^ (k : ℕ)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_three_fourths_l1971_197134


namespace NUMINAMATH_CALUDE_evaluate_expression_l1971_197190

theorem evaluate_expression (b : ℕ) (h : b = 4) :
  b^3 * b^6 * 2 = 524288 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1971_197190


namespace NUMINAMATH_CALUDE_equation_solution_l1971_197105

theorem equation_solution (x : ℝ) : 
  x ≠ (1 / 3) → x ≠ -3 → 
  ((3 * x + 2) / (3 * x^2 + 8 * x - 3) = (3 * x) / (3 * x - 1)) ↔ 
  (x = -1 + Real.sqrt 15 / 3 ∨ x = -1 - Real.sqrt 15 / 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1971_197105


namespace NUMINAMATH_CALUDE_equidistant_point_is_perpendicular_bisector_intersection_l1971_197110

-- Define a triangle in a 2D plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point in a 2D plane
def Point := ℝ × ℝ

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define a perpendicular bisector of a line segment
def perpendicularBisector (p1 p2 : Point) : Set Point := sorry

-- Define the intersection of three sets
def intersectionOfThree (s1 s2 s3 : Set Point) : Set Point := sorry

-- Theorem statement
theorem equidistant_point_is_perpendicular_bisector_intersection (t : Triangle) :
  ∃ (p : Point),
    (distance p t.A = distance p t.B ∧ distance p t.B = distance p t.C) ↔
    p ∈ intersectionOfThree
      (perpendicularBisector t.A t.B)
      (perpendicularBisector t.B t.C)
      (perpendicularBisector t.C t.A) :=
sorry

end NUMINAMATH_CALUDE_equidistant_point_is_perpendicular_bisector_intersection_l1971_197110


namespace NUMINAMATH_CALUDE_scatter_plot_for_linear_relationships_l1971_197126

-- Define the concept of a data visualization method
def DataVisualizationMethod : Type := String

-- Define scatter plot as a data visualization method
def scatter_plot : DataVisualizationMethod := "Scatter plot"

-- Define the property of showing relationships between data points
def shows_point_relationships (method : DataVisualizationMethod) : Prop := 
  method = scatter_plot

-- Define the property of being appropriate for determining linear relationships
def appropriate_for_linear_relationships (method : DataVisualizationMethod) : Prop :=
  shows_point_relationships method

-- Theorem stating that scatter plot is appropriate for determining linear relationships
theorem scatter_plot_for_linear_relationships :
  appropriate_for_linear_relationships scatter_plot :=
by
  sorry


end NUMINAMATH_CALUDE_scatter_plot_for_linear_relationships_l1971_197126


namespace NUMINAMATH_CALUDE_nuts_in_third_box_l1971_197142

theorem nuts_in_third_box (x y z : ℕ) : 
  x + 6 = y + z → y + 10 = x + z → z = 8 := by sorry

end NUMINAMATH_CALUDE_nuts_in_third_box_l1971_197142


namespace NUMINAMATH_CALUDE_investment_problem_l1971_197123

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_problem :
  let principal : ℝ := 3000
  let rate : ℝ := 0.1
  let time : ℕ := 2
  compound_interest principal rate time = 3630.0000000000005 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l1971_197123


namespace NUMINAMATH_CALUDE_no_integral_points_on_tangent_line_l1971_197184

theorem no_integral_points_on_tangent_line (k m n : ℤ) : 
  ∀ x y : ℤ, (m^3 - m) * x + (n^3 - n) * y ≠ (3*k + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integral_points_on_tangent_line_l1971_197184


namespace NUMINAMATH_CALUDE_equal_height_locus_is_circle_l1971_197124

/-- Two flagpoles in a plane -/
structure Flagpoles where
  h : ℝ  -- height of first flagpole
  k : ℝ  -- height of second flagpole
  a : ℝ  -- half the distance between flagpoles

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The locus of points from which the flagpoles appear equally tall -/
def equalHeightLocus (f : Flagpoles) : Set Point := {p : Point | ∃ (t : ℝ), 
  p.x^2 + p.y^2 = t^2 ∧ 
  (p.x + f.a)^2 + p.y^2 = (t * f.k / f.h)^2 ∧
  (p.x - f.a)^2 + p.y^2 = (t * f.k / f.h)^2}

/-- The circle with diameter AB -/
def circleAB (f : Flagpoles) : Set Point := {p : Point | 
  (p.x + f.a * (f.k - f.h) / (f.k + f.h))^2 + p.y^2 = 
  (2 * f.a * f.h * f.k / (f.k + f.h))^2}

theorem equal_height_locus_is_circle (f : Flagpoles) : 
  equalHeightLocus f = circleAB f := by sorry

end NUMINAMATH_CALUDE_equal_height_locus_is_circle_l1971_197124


namespace NUMINAMATH_CALUDE_mikes_initial_amount_solve_mikes_initial_amount_l1971_197180

/-- Proves that Mike's initial amount is $90 given the conditions of the problem -/
theorem mikes_initial_amount (carol_initial : ℕ) (carol_weekly_savings : ℕ) 
  (mike_weekly_savings : ℕ) (weeks : ℕ) (mike_initial : ℕ) : Prop :=
  carol_initial = 60 →
  carol_weekly_savings = 9 →
  mike_weekly_savings = 3 →
  weeks = 5 →
  carol_initial + carol_weekly_savings * weeks = mike_initial + mike_weekly_savings * weeks →
  mike_initial = 90

/-- The main theorem that proves Mike's initial amount -/
theorem solve_mikes_initial_amount : 
  ∃ (mike_initial : ℕ), mikes_initial_amount 60 9 3 5 mike_initial :=
by
  sorry

end NUMINAMATH_CALUDE_mikes_initial_amount_solve_mikes_initial_amount_l1971_197180


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1971_197174

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hbc : b + c ≥ a) :
  b / c + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1971_197174


namespace NUMINAMATH_CALUDE_unique_integer_l1971_197117

def is_valid_integer (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    0 < a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧
    a + b + c + d = 14 ∧
    b + c = 9 ∧
    a - d = 1 ∧
    n % 11 = 0

theorem unique_integer : ∃! n : ℕ, is_valid_integer n ∧ n = 3542 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_l1971_197117


namespace NUMINAMATH_CALUDE_expression_value_l1971_197101

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) :
  3 * x - 4 * y = 1 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1971_197101


namespace NUMINAMATH_CALUDE_corn_acres_calculation_l1971_197137

def total_land : ℝ := 1634
def beans_ratio : ℝ := 4.5
def wheat_ratio : ℝ := 2.3
def corn_ratio : ℝ := 3.8
def barley_ratio : ℝ := 3.4

theorem corn_acres_calculation :
  let total_ratio := beans_ratio + wheat_ratio + corn_ratio + barley_ratio
  let acres_per_part := total_land / total_ratio
  let corn_acres := corn_ratio * acres_per_part
  ∃ ε > 0, |corn_acres - 443.51| < ε :=
by sorry

end NUMINAMATH_CALUDE_corn_acres_calculation_l1971_197137


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l1971_197175

theorem polynomial_product_sum (a b c d e : ℝ) : 
  (∀ x : ℝ, (3 * x^3 - 5 * x^2 + 4 * x - 6) * (7 - 2 * x) = a * x^4 + b * x^3 + c * x^2 + d * x + e) →
  16 * a + 8 * b + 4 * c + 2 * d + e = 42 := by
sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l1971_197175


namespace NUMINAMATH_CALUDE_oldest_child_age_l1971_197121

theorem oldest_child_age (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) 
  (h3 : (a + b + c) / 3 = 9) (h4 : c ≥ b) (h5 : b ≥ a) : c = 13 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_l1971_197121


namespace NUMINAMATH_CALUDE_stratified_sampling_seniors_l1971_197170

theorem stratified_sampling_seniors (total_students : ℕ) (seniors : ℕ) (sample_size : ℕ) 
  (h_total : total_students = 900)
  (h_seniors : seniors = 400)
  (h_sample : sample_size = 45) :
  (seniors * sample_size) / total_students = 20 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_seniors_l1971_197170


namespace NUMINAMATH_CALUDE_go_complexity_ratio_l1971_197173

/-- The upper limit of the state space complexity of Go -/
def M : ℝ := 3^361

/-- The total number of atoms of ordinary matter in the observable universe -/
def N : ℝ := 10^80

/-- Theorem stating that M/N is approximately equal to 10^93 -/
theorem go_complexity_ratio : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |M / N - 10^93| < ε := by
  sorry

end NUMINAMATH_CALUDE_go_complexity_ratio_l1971_197173


namespace NUMINAMATH_CALUDE_base_conversion_l1971_197138

/-- Given that in base x, the decimal number 67 is written as 47, prove that x = 15 -/
theorem base_conversion (x : ℕ) (h : 4 * x + 7 = 67) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l1971_197138


namespace NUMINAMATH_CALUDE_product_abcd_l1971_197128

theorem product_abcd (a b c d : ℚ) : 
  3*a + 4*b + 6*c + 8*d = 42 →
  4*(d+c) = b →
  4*b + 2*c = a →
  c - 2 = d →
  a * b * c * d = (367/37) * (76/37) * (93/74) * (-55/74) := by
sorry

end NUMINAMATH_CALUDE_product_abcd_l1971_197128


namespace NUMINAMATH_CALUDE_intersection_locus_l1971_197153

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line2D where
  slope : ℝ
  intercept : ℝ

/-- Represents a parabola in the form y² = x -/
def parabola (p : Point2D) : Prop :=
  p.y^2 = p.x

/-- Checks if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.x = l.slope * p.y + l.intercept

/-- Checks if four points are concyclic (lie on the same circle) -/
def areConcyclic (p1 p2 p3 p4 : Point2D) : Prop :=
  ∃ (center : Point2D) (radius : ℝ),
    (center.x - p1.x)^2 + (center.y - p1.y)^2 = radius^2 ∧
    (center.x - p2.x)^2 + (center.y - p2.y)^2 = radius^2 ∧
    (center.x - p3.x)^2 + (center.y - p3.y)^2 = radius^2 ∧
    (center.x - p4.x)^2 + (center.y - p4.y)^2 = radius^2

theorem intersection_locus
  (a b : ℝ)
  (ha : 0 < a)
  (hab : a < b)
  (l m : Line2D)
  (hl : l.intercept = a)
  (hm : m.intercept = b)
  (p1 p2 p3 p4 : Point2D)
  (h_distinct : p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4)
  (h_on_parabola : parabola p1 ∧ parabola p2 ∧ parabola p3 ∧ parabola p4)
  (h_on_lines : (pointOnLine p1 l ∨ pointOnLine p1 m) ∧
                (pointOnLine p2 l ∨ pointOnLine p2 m) ∧
                (pointOnLine p3 l ∨ pointOnLine p3 m) ∧
                (pointOnLine p4 l ∨ pointOnLine p4 m))
  (h_concyclic : areConcyclic p1 p2 p3 p4)
  (P : Point2D)
  (h_intersection : pointOnLine P l ∧ pointOnLine P m) :
  P.x = (a + b) / 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_l1971_197153


namespace NUMINAMATH_CALUDE_tony_fish_count_l1971_197160

/-- The number of fish Tony has after a given number of years -/
def fish_count (initial_fish : ℕ) (years : ℕ) : ℕ :=
  initial_fish + years * (3 - 2)

/-- Theorem: Tony will have 15 fish after 10 years -/
theorem tony_fish_count : fish_count 5 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_tony_fish_count_l1971_197160


namespace NUMINAMATH_CALUDE_xiao_ming_school_time_l1971_197139

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two Time values in minutes -/
def timeDifference (t1 t2 : Time) : ℕ :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

/-- Converts minutes to Time -/
def minutesToTime (m : ℕ) : Time :=
  { hours := m / 60,
    minutes := m % 60,
    valid := by sorry }

theorem xiao_ming_school_time :
  let morning_arrival : Time := { hours := 7, minutes := 50, valid := by sorry }
  let morning_departure : Time := { hours := 11, minutes := 50, valid := by sorry }
  let afternoon_arrival : Time := { hours := 14, minutes := 10, valid := by sorry }
  let afternoon_departure : Time := { hours := 17, minutes := 0, valid := by sorry }
  let morning_time := timeDifference morning_arrival morning_departure
  let afternoon_time := timeDifference afternoon_arrival afternoon_departure
  let total_time := morning_time + afternoon_time
  minutesToTime total_time = { hours := 6, minutes := 50, valid := by sorry } :=
by sorry

end NUMINAMATH_CALUDE_xiao_ming_school_time_l1971_197139


namespace NUMINAMATH_CALUDE_x_range_l1971_197148

theorem x_range (x : ℝ) (h1 : (1 : ℝ) / x < 3) (h2 : (1 : ℝ) / x > -2) :
  x > (1 : ℝ) / 3 ∨ x < -(1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1971_197148


namespace NUMINAMATH_CALUDE_inverse_A_times_B_l1971_197146

open Matrix

theorem inverse_A_times_B :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 1]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 2, 5]
  A⁻¹ * B = !![1/2, -1/2; 2, 5] := by
sorry

end NUMINAMATH_CALUDE_inverse_A_times_B_l1971_197146


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l1971_197154

theorem quadratic_rewrite_sum (a b c : ℤ) : 
  (49 : ℤ) * x^2 + 70 * x - 121 = 0 ↔ (a * x + b)^2 = c ∧ 
  a > 0 ∧ 
  a + b + c = -134 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l1971_197154


namespace NUMINAMATH_CALUDE_eagles_per_section_l1971_197100

theorem eagles_per_section 
  (total_eagles : ℕ) 
  (total_sections : ℕ) 
  (h1 : total_eagles = 18) 
  (h2 : total_sections = 3) 
  (h3 : total_eagles % total_sections = 0) : 
  total_eagles / total_sections = 6 := by
  sorry

end NUMINAMATH_CALUDE_eagles_per_section_l1971_197100


namespace NUMINAMATH_CALUDE_debra_accusation_l1971_197119

/-- Represents the number of cookies in various states -/
structure CookieCount where
  initial : ℕ
  louSeniorEaten : ℕ
  louieJuniorTaken : ℕ
  remaining : ℕ

/-- The cookie scenario as described in the problem -/
def cookieScenario : CookieCount where
  initial := 22
  louSeniorEaten := 4
  louieJuniorTaken := 7
  remaining := 11

/-- Theorem stating the portion of cookies Debra accuses Lou Senior of eating -/
theorem debra_accusation (c : CookieCount) (h1 : c = cookieScenario) :
  c.louSeniorEaten = 4 ∧ c.initial = 22 := by sorry

end NUMINAMATH_CALUDE_debra_accusation_l1971_197119


namespace NUMINAMATH_CALUDE_max_three_digit_gp_length_l1971_197122

/-- A geometric progression of 3-digit natural numbers -/
def ThreeDigitGP (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ (q : ℚ), q > 1 ∧
  (∀ i ≤ n, 100 ≤ a i ∧ a i < 1000) ∧
  (∀ i < n, a (i + 1) = (a i : ℚ) * q)

/-- The maximum length of a 3-digit geometric progression -/
def MaxGPLength : ℕ := 6

/-- Theorem stating that 6 is the maximum length of a 3-digit geometric progression -/
theorem max_three_digit_gp_length :
  (∃ a : ℕ → ℕ, ThreeDigitGP a MaxGPLength) ∧
  (∀ n > MaxGPLength, ∀ a : ℕ → ℕ, ¬ ThreeDigitGP a n) :=
sorry

end NUMINAMATH_CALUDE_max_three_digit_gp_length_l1971_197122


namespace NUMINAMATH_CALUDE_number_of_classes_l1971_197178

theorem number_of_classes (single_sided_per_class_per_day : ℕ)
                          (double_sided_per_class_per_day : ℕ)
                          (school_days_per_week : ℕ)
                          (total_single_sided_per_week : ℕ)
                          (total_double_sided_per_week : ℕ)
                          (h1 : single_sided_per_class_per_day = 175)
                          (h2 : double_sided_per_class_per_day = 75)
                          (h3 : school_days_per_week = 5)
                          (h4 : total_single_sided_per_week = 16000)
                          (h5 : total_double_sided_per_week = 7000) :
  ⌊(total_single_sided_per_week + total_double_sided_per_week : ℚ) /
   ((single_sided_per_class_per_day + double_sided_per_class_per_day) * school_days_per_week)⌋ = 18 :=
by sorry

end NUMINAMATH_CALUDE_number_of_classes_l1971_197178


namespace NUMINAMATH_CALUDE_lattice_points_on_segment_l1971_197141

/-- The number of lattice points on a line segment --/
def latticePointCount (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem stating the number of lattice points on the given line segment --/
theorem lattice_points_on_segment : latticePointCount 5 13 35 97 = 7 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_segment_l1971_197141


namespace NUMINAMATH_CALUDE_refrigerator_theorem_l1971_197167

def refrigerator_problem (P : ℝ) : Prop :=
  let discount_rate : ℝ := 0.20
  let profit_rate : ℝ := 0.10
  let additional_costs : ℝ := 375
  let selling_price : ℝ := 18975
  let purchase_price : ℝ := P * (1 - discount_rate)
  let total_price : ℝ := purchase_price + additional_costs
  (P * (1 + profit_rate) = selling_price) → (total_price = 14175)

theorem refrigerator_theorem :
  ∃ P : ℝ, refrigerator_problem P :=
sorry

end NUMINAMATH_CALUDE_refrigerator_theorem_l1971_197167


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l1971_197196

/-- For the equation (m-1)x^2 + mx - 1 = 0 to be a quadratic equation in x,
    m must not equal 1. -/
theorem quadratic_equation_condition (m : ℝ) :
  (∀ x, (m - 1) * x^2 + m * x - 1 = 0 → (m - 1) ≠ 0) ↔ m ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l1971_197196


namespace NUMINAMATH_CALUDE_stock_recovery_l1971_197166

theorem stock_recovery (initial_price : ℝ) (initial_price_pos : initial_price > 0) : 
  let price_after_drops := initial_price * (1 - 0.1)^4
  ∃ n : ℕ, n ≥ 5 ∧ price_after_drops * (1 + 0.1)^n ≥ initial_price :=
by sorry

end NUMINAMATH_CALUDE_stock_recovery_l1971_197166


namespace NUMINAMATH_CALUDE_expression_value_l1971_197197

theorem expression_value (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1971_197197


namespace NUMINAMATH_CALUDE_three_flower_purchase_options_l1971_197115

/-- Represents a flower purchase option -/
structure FlowerPurchase where
  carnations : Nat
  lilies : Nat

/-- The cost of a single carnation in yuan -/
def carnationCost : Nat := 2

/-- The cost of a single lily in yuan -/
def lilyCost : Nat := 3

/-- The total amount Xiaoming has to spend in yuan -/
def totalSpend : Nat := 20

/-- Predicate to check if a flower purchase is valid -/
def isValidPurchase (purchase : FlowerPurchase) : Prop :=
  carnationCost * purchase.carnations + lilyCost * purchase.lilies = totalSpend

/-- The theorem stating that there are exactly 3 valid flower purchase options -/
theorem three_flower_purchase_options :
  ∃ (options : List FlowerPurchase),
    (options.length = 3) ∧
    (∀ purchase ∈ options, isValidPurchase purchase) ∧
    (∀ purchase, isValidPurchase purchase → purchase ∈ options) :=
sorry

end NUMINAMATH_CALUDE_three_flower_purchase_options_l1971_197115


namespace NUMINAMATH_CALUDE_peach_difference_l1971_197163

def red_peaches : ℕ := 19
def yellow_peaches : ℕ := 11

theorem peach_difference : red_peaches - yellow_peaches = 8 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l1971_197163


namespace NUMINAMATH_CALUDE_vector_inequality_l1971_197127

/-- Given vectors u, v, and w in ℝ², prove that w ≠ u - 3v -/
theorem vector_inequality (u v w : ℝ × ℝ) 
  (hu : u = (3, -6)) 
  (hv : v = (4, 2)) 
  (hw : w = (-12, -6)) : 
  w ≠ u - 3 • v := by sorry

end NUMINAMATH_CALUDE_vector_inequality_l1971_197127


namespace NUMINAMATH_CALUDE_max_value_of_operation_l1971_197104

theorem max_value_of_operation : ∃ (m : ℕ), 
  (∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 → 3 * (300 - n) ≤ m) ∧ 
  (∃ (n : ℕ), 10 ≤ n ∧ n ≤ 99 ∧ 3 * (300 - n) = m) ∧ 
  m = 870 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_operation_l1971_197104


namespace NUMINAMATH_CALUDE_complex_number_properties_l1971_197198

theorem complex_number_properties (z : ℂ) (h : (2 + I) * z = 1 + 3 * I) : 
  Complex.abs z = Real.sqrt 2 ∧ z^2 - 2*z + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l1971_197198


namespace NUMINAMATH_CALUDE_sqrt_27_minus_3_sqrt_one_third_l1971_197155

theorem sqrt_27_minus_3_sqrt_one_third : 
  Real.sqrt 27 - 3 * Real.sqrt (1/3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_minus_3_sqrt_one_third_l1971_197155


namespace NUMINAMATH_CALUDE_spider_web_problem_l1971_197161

theorem spider_web_problem (S : ℕ) : 
  (∃ (W D : ℕ), 
    S = W ∧              -- Number of spiders equals number of webs made by each spider
    S = D ∧              -- Number of spiders equals number of days taken
    7 * S = W * D) →     -- Relationship between 1 spider making 1 web in 7 days
  S = 7 := by
sorry

end NUMINAMATH_CALUDE_spider_web_problem_l1971_197161


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1971_197118

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    x₁^2 + p*x₁ + q = 0 ∧
    x₂^2 + p*x₂ + q = 0 ∧
    |x₁ - x₂| = 1) →
  p = Real.sqrt (4*q + 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1971_197118


namespace NUMINAMATH_CALUDE_triangle_circle_relation_l1971_197168

theorem triangle_circle_relation 
  (AO' AO₁ AB AC t s s₁ s₂ s₃ r r₁ α : ℝ) 
  (h1 : AO' * Real.sin (α/2) = r ∧ r = t/s)
  (h2 : AO₁ * Real.sin (α/2) = r₁ ∧ r₁ = t/s₁)
  (h3 : AO' * AO₁ = t^2 / (s * s₁ * Real.sin (α/2)^2))
  (h4 : Real.sin (α/2)^2 = (s₂ * s₃) / (AB * AC)) :
  AO' * AO₁ = AB * AC := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_relation_l1971_197168


namespace NUMINAMATH_CALUDE_percentage_of_160_to_50_l1971_197102

theorem percentage_of_160_to_50 : ∀ x : ℝ, (160 / 50) * 100 = x → x = 320 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_160_to_50_l1971_197102


namespace NUMINAMATH_CALUDE_cars_with_both_features_l1971_197108

/-- Represents the car lot scenario -/
structure CarLot where
  total : Nat
  with_airbag : Nat
  with_power_windows : Nat
  with_neither : Nat

/-- Theorem stating the number of cars with both air-bag and power windows -/
theorem cars_with_both_features (lot : CarLot) 
  (h1 : lot.total = 65)
  (h2 : lot.with_airbag = 45)
  (h3 : lot.with_power_windows = 30)
  (h4 : lot.with_neither = 2) :
  lot.with_airbag + lot.with_power_windows - (lot.total - lot.with_neither) = 12 := by
  sorry

#check cars_with_both_features

end NUMINAMATH_CALUDE_cars_with_both_features_l1971_197108


namespace NUMINAMATH_CALUDE_unique_solution_l1971_197169

def is_divisible (x y : ℕ) : Prop := ∃ k : ℕ, x = y * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def condition1 (a b : ℕ) : Prop := is_divisible (a^2 + 4*a + 3) b

def condition2 (a b : ℕ) : Prop := a^2 + a*b - 6*b^2 - 2*a - 16*b - 8 = 0

def condition3 (a b : ℕ) : Prop := is_divisible (a + 2*b + 1) 4

def condition4 (a b : ℕ) : Prop := is_prime (a + 6*b + 1)

def exactly_three_true (a b : ℕ) : Prop :=
  (condition1 a b ∧ condition2 a b ∧ condition3 a b ∧ ¬condition4 a b) ∨
  (condition1 a b ∧ condition2 a b ∧ ¬condition3 a b ∧ condition4 a b) ∨
  (condition1 a b ∧ ¬condition2 a b ∧ condition3 a b ∧ condition4 a b) ∨
  (¬condition1 a b ∧ condition2 a b ∧ condition3 a b ∧ condition4 a b)

theorem unique_solution :
  ∀ a b : ℕ, exactly_three_true a b ↔ (a = 6 ∧ b = 1) ∨ (a = 18 ∧ b = 7) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1971_197169


namespace NUMINAMATH_CALUDE_rectangle_area_is_30_l1971_197125

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle -/
def original : Rectangle := { length := 0, width := 0 }

/-- The rectangle with increased length -/
def increased_length : Rectangle := { length := original.length + 2, width := original.width }

/-- The rectangle with decreased width -/
def decreased_width : Rectangle := { length := original.length, width := original.width - 3 }

theorem rectangle_area_is_30 :
  increased_length.area - original.area = 10 →
  original.area - decreased_width.area = 18 →
  original.area = 30 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_is_30_l1971_197125


namespace NUMINAMATH_CALUDE_arithmetic_progression_problem_l1971_197149

theorem arithmetic_progression_problem (a d : ℚ) : 
  (3 * ((a - d) + a) = 2 * (a + d)) →
  ((a - 2)^2 = (a - d) * (a + d)) →
  ((a = 5 ∧ d = 4) ∨ (a = 5/4 ∧ d = 1)) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_problem_l1971_197149


namespace NUMINAMATH_CALUDE_pigeon_percentage_among_non_swans_l1971_197113

def bird_distribution (total : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ := 
  (0.20 * total, 0.30 * total, 0.15 * total, 0.25 * total, 0.10 * total)

theorem pigeon_percentage_among_non_swans (total : ℝ) (h : total > 0) :
  let (geese, swans, herons, ducks, pigeons) := bird_distribution total
  let non_swans := total - swans
  (pigeons / non_swans) * 100 = 14 := by
  sorry

end NUMINAMATH_CALUDE_pigeon_percentage_among_non_swans_l1971_197113


namespace NUMINAMATH_CALUDE_parallel_planes_condition_l1971_197189

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define the lines and planes
variable (m n : Line)
variable (α β γ : Plane)

-- State the theorem
theorem parallel_planes_condition 
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β ∧ α ≠ γ ∧ β ≠ γ)
  (h_parallel : parallel m n)
  (h_perp1 : perpendicular n α)
  (h_perp2 : perpendicular m β) :
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_condition_l1971_197189


namespace NUMINAMATH_CALUDE_batting_average_is_60_l1971_197185

/-- A batsman's batting statistics -/
structure BattingStats where
  total_innings : ℕ
  highest_score : ℕ
  lowest_score : ℕ
  average_excluding_extremes : ℚ

/-- The batting average for all innings -/
def batting_average (stats : BattingStats) : ℚ :=
  let total_runs := stats.average_excluding_extremes * (stats.total_innings - 2 : ℚ) + stats.highest_score + stats.lowest_score
  total_runs / stats.total_innings

/-- Theorem stating the batting average for the given conditions -/
theorem batting_average_is_60 (stats : BattingStats) 
    (h1 : stats.total_innings = 46)
    (h2 : stats.highest_score = 194)
    (h3 : stats.highest_score - stats.lowest_score = 180)
    (h4 : stats.average_excluding_extremes = 58) :
    batting_average stats = 60 := by
  sorry

end NUMINAMATH_CALUDE_batting_average_is_60_l1971_197185


namespace NUMINAMATH_CALUDE_right_triangle_condition_l1971_197129

theorem right_triangle_condition (A B C : ℝ) (h_triangle : A + B + C = Real.pi) 
  (h_condition : Real.sin A * Real.cos B = 1 - Real.cos A * Real.sin B) : C = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_condition_l1971_197129


namespace NUMINAMATH_CALUDE_part1_part2_l1971_197150

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the intersection point of l₁ and l₂
def intersection : ℝ × ℝ := (-2, 2)

-- Define the line parallel to 3x + y - 1 = 0
def parallel_line (x y : ℝ) : Prop := 3 * x + y - 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 1)

-- Part 1: Prove that if l passes through the intersection and is parallel to 3x + y - 1 = 0,
-- then its equation is 3x + y + 4 = 0
theorem part1 (l : ℝ → ℝ → Prop) :
  (∀ x y, l x y ↔ ∃ k, 3 * x + y + k = 0) →
  l (intersection.1) (intersection.2) →
  (∀ x y, l x y → parallel_line x y) →
  (∀ x y, l x y ↔ 3 * x + y + 4 = 0) :=
sorry

-- Part 2: Prove that if l passes through the intersection and the distance from A to l is 5,
-- then its equation is either x = -2 or 12x - 5y + 34 = 0
theorem part2 (l : ℝ → ℝ → Prop) :
  l (intersection.1) (intersection.2) →
  (∀ x y, l x y → (((x - point_A.1) ^ 2 + (y - point_A.2) ^ 2) : ℝ).sqrt = 5) →
  (∀ x y, l x y ↔ x = -2 ∨ 12 * x - 5 * y + 34 = 0) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l1971_197150


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1971_197151

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ n : ℕ, 2^n > 1000) ↔ (∀ n : ℕ, 2^n ≤ 1000) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1971_197151


namespace NUMINAMATH_CALUDE_fibonacci_determinant_identity_l1971_197133

def fibonacci : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def fibonacci_matrix (n : ℕ) : Matrix (Fin 2) (Fin 2) ℤ :=
  !![fibonacci (n + 1), fibonacci n; fibonacci n, fibonacci (n - 1)]

theorem fibonacci_determinant_identity (n : ℕ) :
  fibonacci (n + 1) * fibonacci (n - 1) - fibonacci n ^ 2 = (-1) ^ n :=
sorry

end NUMINAMATH_CALUDE_fibonacci_determinant_identity_l1971_197133


namespace NUMINAMATH_CALUDE_combined_tax_rate_l1971_197162

theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.30) 
  (h2 : mindy_rate = 0.20) 
  (h3 : income_ratio = 3) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.225 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l1971_197162


namespace NUMINAMATH_CALUDE_percentage_to_number_l1971_197120

theorem percentage_to_number (x : ℝ) (h : x = 209) :
  x / 100 * 100 = 209 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_number_l1971_197120


namespace NUMINAMATH_CALUDE_gas_station_sales_l1971_197158

/-- The total number of boxes sold at a gas station -/
def total_boxes (chocolate_boxes sugar_boxes gum_boxes : ℕ) : ℕ :=
  chocolate_boxes + sugar_boxes + gum_boxes

/-- Theorem: The gas station sold 9 boxes in total -/
theorem gas_station_sales : total_boxes 2 5 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gas_station_sales_l1971_197158


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1971_197183

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha1 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2016) + 2016
  f 2016 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1971_197183


namespace NUMINAMATH_CALUDE_triangle_side_a_value_l1971_197130

noncomputable def triangle_side_a (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Define the triangle ABC
  (0 < A ∧ 0 < B ∧ 0 < C) ∧
  (A + B + C = Real.pi) ∧
  -- Relate sides to angles using sine law
  (a / (Real.sin A) = b / (Real.sin B)) ∧
  (b / (Real.sin B) = c / (Real.sin C)) ∧
  -- Given conditions
  (Real.sin B = 3/5) ∧
  (b = 5) ∧
  (A = 2 * B) ∧
  -- Conclusion
  (a = 8)

theorem triangle_side_a_value :
  ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_side_a A B C a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_side_a_value_l1971_197130


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1971_197171

theorem complex_fraction_evaluation : 
  (((10/3 / 10 + 0.175 / 0.35) / (1.75 - (28/17) * (51/56))) - 
   ((11/18 - 1/15) / 1.4) / ((0.5 - 1/9) * 3)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1971_197171


namespace NUMINAMATH_CALUDE_platform_length_calculation_l1971_197191

-- Define the given parameters
def train_length : ℝ := 250
def train_speed_kmph : ℝ := 72
def train_speed_mps : ℝ := 20
def time_to_cross : ℝ := 25

-- Define the theorem
theorem platform_length_calculation :
  let total_distance := train_speed_mps * time_to_cross
  let platform_length := total_distance - train_length
  platform_length = 250 := by sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l1971_197191


namespace NUMINAMATH_CALUDE_prob_at_least_one_spade_or_ace_value_l1971_197116

/-- The number of cards in the deck -/
def deck_size : ℕ := 54

/-- The number of cards that are either spades or aces -/
def spade_or_ace_count : ℕ := 16

/-- The probability of drawing at least one spade or ace in two independent draws with replacement -/
def prob_at_least_one_spade_or_ace : ℚ :=
  1 - (1 - spade_or_ace_count / deck_size) ^ 2

theorem prob_at_least_one_spade_or_ace_value :
  prob_at_least_one_spade_or_ace = 368 / 729 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_spade_or_ace_value_l1971_197116


namespace NUMINAMATH_CALUDE_wind_pressure_theorem_l1971_197164

/-- The pressure-area-velocity relationship for wind on a sail -/
theorem wind_pressure_theorem (k : ℝ) :
  (∃ P A V : ℝ, P = k * A * V^2 ∧ P = 1.25 ∧ A = 1 ∧ V = 20) →
  (∃ P A V : ℝ, P = k * A * V^2 ∧ P = 20 ∧ A = 4 ∧ V = 40) :=
by sorry

end NUMINAMATH_CALUDE_wind_pressure_theorem_l1971_197164


namespace NUMINAMATH_CALUDE_tangent_line_parallel_to_given_line_l1971_197152

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 10

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_parallel_to_given_line :
  ∀ x₀ y₀ : ℝ,
  f x₀ = y₀ →
  f' x₀ = 4 →
  ((x₀ = 1 ∧ y₀ = -8) ∨ (x₀ = -1 ∧ y₀ = -12)) ∧
  ((y₀ = 4 * x₀ - 12) ∨ (y₀ = 4 * x₀ - 8)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_to_given_line_l1971_197152


namespace NUMINAMATH_CALUDE_janet_action_figures_l1971_197199

/-- Calculates the final number of action figures Janet has after selling, buying, and receiving a gift. -/
theorem janet_action_figures (initial : ℕ) (sold : ℕ) (bought : ℕ) (gift_multiplier : ℕ) : 
  initial = 10 → sold = 6 → bought = 4 → gift_multiplier = 2 →
  (initial - sold + bought) * (gift_multiplier + 1) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_janet_action_figures_l1971_197199


namespace NUMINAMATH_CALUDE_range_of_a_l1971_197192

/-- The range of values for a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (a^2 - 2*a - 2)^x < (a^2 - 2*a - 2)^y) ∧ 
  ¬(0 < a ∧ a < 4) →
  a ≥ 4 ∨ a < -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1971_197192


namespace NUMINAMATH_CALUDE_percentage_increase_problem_l1971_197106

theorem percentage_increase_problem (initial : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) (final : ℝ) :
  initial = 1500 →
  increase_percent = 20 →
  decrease_percent = 40 →
  final = 1080 →
  final = initial * (1 + increase_percent / 100) * (1 - decrease_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_problem_l1971_197106


namespace NUMINAMATH_CALUDE_megawheel_capacity_l1971_197114

/-- The Megawheel problem -/
theorem megawheel_capacity (total_seats : ℕ) (total_people : ℕ) (people_per_seat : ℕ) 
  (h1 : total_seats = 15)
  (h2 : total_people = 75)
  (h3 : people_per_seat * total_seats = total_people) :
  people_per_seat = 5 := by
  sorry

end NUMINAMATH_CALUDE_megawheel_capacity_l1971_197114


namespace NUMINAMATH_CALUDE_perfect_square_example_l1971_197165

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem perfect_square_example : is_perfect_square (4^10 * 5^5 * 6^10) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_example_l1971_197165


namespace NUMINAMATH_CALUDE_score_difference_l1971_197193

def score_distribution : List (ℝ × ℝ) := [
  (0.15, 60),
  (0.25, 75),
  (0.35, 85),
  (0.20, 95),
  (0.05, 110)
]

def median_score : ℝ := 85

def mean_score : ℝ := (score_distribution.map (λ (p, s) => p * s)).sum

theorem score_difference : median_score - mean_score = 3 := by sorry

end NUMINAMATH_CALUDE_score_difference_l1971_197193


namespace NUMINAMATH_CALUDE_bird_families_to_asia_l1971_197176

theorem bird_families_to_asia (total_to_africa : ℕ) (difference : ℕ) : total_to_africa = 42 → difference = 11 → total_to_africa - difference = 31 := by
  sorry

end NUMINAMATH_CALUDE_bird_families_to_asia_l1971_197176


namespace NUMINAMATH_CALUDE_square_sum_value_l1971_197177

theorem square_sum_value (x y : ℝ) (h1 : x + 3*y = 6) (h2 : x*y = -9) : x^2 + 9*y^2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l1971_197177


namespace NUMINAMATH_CALUDE_geometry_class_size_l1971_197182

theorem geometry_class_size :
  ∀ (total_students : ℕ),
  (total_students / 2 : ℚ) = (total_students : ℚ) / 2 →
  ((total_students / 2) / 5 : ℚ) = (total_students : ℚ) / 10 →
  (total_students : ℚ) / 10 = 10 →
  total_students = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_geometry_class_size_l1971_197182


namespace NUMINAMATH_CALUDE_zanders_stickers_l1971_197156

theorem zanders_stickers (S : ℚ) : 
  (1/5 : ℚ) * S + (3/10 : ℚ) * (S - (1/5 : ℚ) * S) = 44 → S = 100 := by
sorry

end NUMINAMATH_CALUDE_zanders_stickers_l1971_197156


namespace NUMINAMATH_CALUDE_selling_price_ratio_l1971_197131

theorem selling_price_ratio (C : ℝ) (S1 S2 : ℝ) 
  (h1 : S1 = C + 0.60 * C) 
  (h2 : S2 = C + 3.20 * C) : 
  S2 / S1 = 21 / 8 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l1971_197131


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1971_197109

/-- A geometric sequence with first term 3 and specific arithmetic property -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) ∧
  a 1 = 3 ∧
  ∃ d : ℝ, 2 * a 2 = 4 * a 1 + d ∧ a 3 = 2 * a 2 + d

theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 3 + a 5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1971_197109


namespace NUMINAMATH_CALUDE_fraction_simplification_l1971_197103

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hyd : y - 2/x ≠ 0) :
  (2*x - 3/y) / (3*y - 2/x) = (2*x*y - 3) / (3*x*y - 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1971_197103


namespace NUMINAMATH_CALUDE_trajectory_equation_l1971_197179

/-- 
Given a point P(x, y) in the Cartesian coordinate system,
if the product of its distances to the x-axis and y-axis equals 1,
then the equation of its trajectory is xy = ± 1.
-/
theorem trajectory_equation (x y : ℝ) : 
  (|x| * |y| = 1) → (x * y = 1 ∨ x * y = -1) := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1971_197179


namespace NUMINAMATH_CALUDE_longest_segment_through_interior_point_l1971_197159

/-- A convex polygon in 2D space -/
structure ConvexPolygon where
  -- Define the properties of a convex polygon
  -- (This is a simplified representation)
  vertices : Set (ℝ × ℝ)
  is_convex : Bool

/-- A point in 2D space -/
def Point := ℝ × ℝ

/-- A direction in 2D space -/
def Direction := ℝ × ℝ

/-- Checks if a point is inside a convex polygon -/
def is_inside (K : ConvexPolygon) (P : Point) : Prop := sorry

/-- The length of the intersection of a line with a polygon -/
def intersection_length (K : ConvexPolygon) (P : Point) (d : Direction) : ℝ := sorry

/-- The theorem statement -/
theorem longest_segment_through_interior_point 
  (K : ConvexPolygon) (P : Point) (h : is_inside K P) :
  ∃ (d : Direction), 
    ∀ (Q : Point), is_inside K Q → 
      intersection_length K P d ≥ intersection_length K Q d := by sorry

end NUMINAMATH_CALUDE_longest_segment_through_interior_point_l1971_197159


namespace NUMINAMATH_CALUDE_total_cost_is_21_93_l1971_197147

/-- The amount Alyssa paid for grapes in dollars -/
def grapes_cost : ℚ := 12.08

/-- The amount Alyssa paid for cherries in dollars -/
def cherries_cost : ℚ := 9.85

/-- The total amount Alyssa spent on fruits -/
def total_cost : ℚ := grapes_cost + cherries_cost

/-- Theorem stating that the total cost of fruits is $21.93 -/
theorem total_cost_is_21_93 : total_cost = 21.93 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_21_93_l1971_197147


namespace NUMINAMATH_CALUDE_sum_remainder_mod_16_l1971_197132

theorem sum_remainder_mod_16 : (List.sum [75, 76, 77, 78, 79, 80, 81, 82]) % 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_16_l1971_197132


namespace NUMINAMATH_CALUDE_inequality_of_squares_existence_of_positive_l1971_197107

theorem inequality_of_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

theorem existence_of_positive (x y z : ℝ) :
  let a := x^2 - 2*y + Real.pi/2
  let b := y^2 - 2*z + Real.pi/3
  let c := z^2 - 2*x + Real.pi/6
  (a > 0) ∨ (b > 0) ∨ (c > 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_squares_existence_of_positive_l1971_197107
