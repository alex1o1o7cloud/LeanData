import Mathlib

namespace NUMINAMATH_CALUDE_product_equals_square_l851_85152

theorem product_equals_square : 100 * 29.98 * 2.998 * 1000 = (2998 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l851_85152


namespace NUMINAMATH_CALUDE_green_bay_high_relay_race_length_l851_85163

/-- The length of a relay race given the number of team members and distance per member -/
def relay_race_length (team_members : ℕ) (distance_per_member : ℕ) : ℕ :=
  team_members * distance_per_member

/-- Theorem: The relay race length for 5 team members running 30 meters each is 150 meters -/
theorem green_bay_high_relay_race_length :
  relay_race_length 5 30 = 150 := by
  sorry

end NUMINAMATH_CALUDE_green_bay_high_relay_race_length_l851_85163


namespace NUMINAMATH_CALUDE_carla_counting_theorem_l851_85112

theorem carla_counting_theorem (ceiling_tiles : ℕ) (books : ℕ) 
  (h1 : ceiling_tiles = 38) 
  (h2 : books = 75) : 
  ceiling_tiles * 2 + books * 3 = 301 := by
  sorry

end NUMINAMATH_CALUDE_carla_counting_theorem_l851_85112


namespace NUMINAMATH_CALUDE_constant_odd_function_is_zero_l851_85183

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem constant_odd_function_is_zero (k : ℝ) (h : IsOdd (fun x ↦ k)) : k = 0 := by
  sorry

end NUMINAMATH_CALUDE_constant_odd_function_is_zero_l851_85183


namespace NUMINAMATH_CALUDE_pizza_theorem_l851_85126

def pizza_problem (craig_day1 craig_day2 heather_day1 heather_day2 : ℕ) : Prop :=
  craig_day1 = 40 ∧
  craig_day2 = craig_day1 + 60 ∧
  heather_day1 = 4 * craig_day1 ∧
  heather_day2 = craig_day2 - 20 ∧
  craig_day1 + craig_day2 + heather_day1 + heather_day2 = 380

theorem pizza_theorem : ∃ craig_day1 craig_day2 heather_day1 heather_day2 : ℕ,
  pizza_problem craig_day1 craig_day2 heather_day1 heather_day2 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_theorem_l851_85126


namespace NUMINAMATH_CALUDE_odd_integers_sum_169_l851_85144

/-- Sum of consecutive odd integers from 1 to n -/
def sumOddIntegers (n : ℕ) : ℕ :=
  (n * n + n) / 2

/-- The problem statement -/
theorem odd_integers_sum_169 :
  ∃ n : ℕ, n % 2 = 1 ∧ sumOddIntegers n = 169 ∧ n = 25 := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_sum_169_l851_85144


namespace NUMINAMATH_CALUDE_line_circle_relationship_l851_85116

theorem line_circle_relationship (m : ℝ) (h : m > 0) :
  let line := {(x, y) : ℝ × ℝ | Real.sqrt 2 * (x + y) + 1 + m = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = m}
  (∃ p, p ∈ line ∩ circle ∧ (∀ q ∈ line ∩ circle, q = p)) ∨
  (line ∩ circle = ∅) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_relationship_l851_85116


namespace NUMINAMATH_CALUDE_gcd_of_168_56_224_l851_85169

theorem gcd_of_168_56_224 : Nat.gcd 168 (Nat.gcd 56 224) = 56 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_168_56_224_l851_85169


namespace NUMINAMATH_CALUDE_binomial_distribution_problem_l851_85138

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

variable (ξ : BinomialDistribution)

/-- The expected value of a binomial distribution -/
def expectation (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: For a binomial distribution with E[ξ] = 300 and D[ξ] = 200, p = 1/3 -/
theorem binomial_distribution_problem (ξ : BinomialDistribution) 
  (h2 : expectation ξ = 300) (h3 : variance ξ = 200) : ξ.p = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_binomial_distribution_problem_l851_85138


namespace NUMINAMATH_CALUDE_max_volume_right_triangle_rotation_l851_85187

theorem max_volume_right_triangle_rotation (a b c : ℝ) : 
  a = 3 → b = 4 → c = 5 → a^2 + b^2 = c^2 →
  (max (1/3 * Real.pi * a^2 * b) (max (1/3 * Real.pi * b^2 * a) (1/3 * Real.pi * (2 * (1/2 * a * b) / c)^2 * c))) = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_max_volume_right_triangle_rotation_l851_85187


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l851_85171

def circle1_center : ℝ × ℝ := (-32, 42)
def circle2_center : ℝ × ℝ := (0, 0)
def circle1_radius : ℝ := 52
def circle2_radius : ℝ := 3

theorem circles_externally_tangent :
  let d := Real.sqrt ((circle1_center.1 - circle2_center.1)^2 + (circle1_center.2 - circle2_center.2)^2)
  d = circle1_radius + circle2_radius := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l851_85171


namespace NUMINAMATH_CALUDE_total_carriages_eq_460_l851_85124

/-- The number of carriages in Euston -/
def euston : ℕ := 130

/-- The number of carriages in Norfolk -/
def norfolk : ℕ := euston - 20

/-- The number of carriages in Norwich -/
def norwich : ℕ := 100

/-- The number of carriages in Flying Scotsman -/
def flying_scotsman : ℕ := norwich + 20

/-- The total number of carriages -/
def total_carriages : ℕ := euston + norfolk + norwich + flying_scotsman

theorem total_carriages_eq_460 : total_carriages = 460 := by
  sorry

end NUMINAMATH_CALUDE_total_carriages_eq_460_l851_85124


namespace NUMINAMATH_CALUDE_problem_solution_l851_85143

theorem problem_solution : 
  ∀ N : ℝ, (2 + 3 + 4) / 3 = (1990 + 1991 + 1992) / N → N = 1991 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l851_85143


namespace NUMINAMATH_CALUDE_orthocenters_collinear_l851_85180

-- Define a type for lines in a plane
def Line : Type := ℝ → ℝ → Prop

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop :=
  ∃ (t : ℝ), q.1 - p.1 = t * (r.1 - p.1) ∧ q.2 - p.2 = t * (r.2 - p.2)

-- Define a function to get the intersection point of two lines
noncomputable def intersect (l1 l2 : Line) : Point :=
  sorry

-- Define a function to get the orthocenter of a triangle
noncomputable def orthocenter (a b c : Point) : Point :=
  sorry

-- Main theorem
theorem orthocenters_collinear
  (l1 l2 l3 l4 : Line)
  (p1 p2 p3 p4 p5 p6 : Point)
  (h1 : p1 = intersect l1 l2)
  (h2 : p2 = intersect l1 l3)
  (h3 : p3 = intersect l1 l4)
  (h4 : p4 = intersect l2 l3)
  (h5 : p5 = intersect l2 l4)
  (h6 : p6 = intersect l3 l4)
  : collinear
      (orthocenter p1 p2 p4)
      (orthocenter p1 p3 p5)
      (orthocenter p2 p3 p6) :=
by
  sorry

end NUMINAMATH_CALUDE_orthocenters_collinear_l851_85180


namespace NUMINAMATH_CALUDE_jakes_weight_l851_85182

theorem jakes_weight (jake sister brother : ℝ) 
  (h1 : jake - 40 = 3 * sister)
  (h2 : jake - (sister + 10) = brother)
  (h3 : jake + sister + brother = 300) :
  jake = 155 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l851_85182


namespace NUMINAMATH_CALUDE_skateboard_bicycle_problem_l851_85129

theorem skateboard_bicycle_problem (skateboards bicycles : ℕ) : 
  (skateboards : ℚ) / bicycles = 7 / 4 →
  skateboards = bicycles + 12 →
  skateboards + bicycles = 44 := by
sorry

end NUMINAMATH_CALUDE_skateboard_bicycle_problem_l851_85129


namespace NUMINAMATH_CALUDE_remaining_work_days_for_z_l851_85155

-- Define work rates for each person
def work_rate_x : ℚ := 1 / 5
def work_rate_y : ℚ := 1 / 20
def work_rate_z : ℚ := 1 / 30

-- Define the total work as 1 (100%)
def total_work : ℚ := 1

-- Define the number of days all three work together
def days_together : ℚ := 2

-- Theorem statement
theorem remaining_work_days_for_z :
  let combined_rate := work_rate_x + work_rate_y + work_rate_z
  let work_done_together := combined_rate * days_together
  let remaining_work := total_work - work_done_together
  (remaining_work / work_rate_z : ℚ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_remaining_work_days_for_z_l851_85155


namespace NUMINAMATH_CALUDE_complex_magnitude_range_l851_85164

theorem complex_magnitude_range (θ : ℝ) :
  let z : ℂ := Complex.mk (Real.sqrt 3 * Real.sin θ) (Real.cos θ)
  Complex.abs z < Real.sqrt 2 ↔ ∃ k : ℤ, -π/4 + k*π < θ ∧ θ < π/4 + k*π :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_range_l851_85164


namespace NUMINAMATH_CALUDE_vitamin_d3_capsules_per_bottle_l851_85146

/-- Calculates the number of capsules in each bottle given the total days, 
    daily serving size, and total number of bottles. -/
def capsules_per_bottle (total_days : ℕ) (daily_serving : ℕ) (total_bottles : ℕ) : ℕ :=
  (total_days * daily_serving) / total_bottles

/-- Theorem stating that given the specific conditions, the number of capsules
    per bottle is 60. -/
theorem vitamin_d3_capsules_per_bottle :
  capsules_per_bottle 180 2 6 = 60 := by
  sorry

#eval capsules_per_bottle 180 2 6

end NUMINAMATH_CALUDE_vitamin_d3_capsules_per_bottle_l851_85146


namespace NUMINAMATH_CALUDE_escalator_speed_l851_85196

theorem escalator_speed (escalator_speed : ℝ) (escalator_length : ℝ) (time_taken : ℝ) 
  (h1 : escalator_speed = 12)
  (h2 : escalator_length = 150)
  (h3 : time_taken = 10) :
  let person_speed := (escalator_length / time_taken) - escalator_speed
  person_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_escalator_speed_l851_85196


namespace NUMINAMATH_CALUDE_quadratic_roots_l851_85128

/-- A quadratic function passing through specific points has roots -4 and 1 -/
theorem quadratic_roots (a b c : ℝ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (f (-5) = 6) ∧ (f (-4) = 0) ∧ (f (-2) = -6) ∧ (f 0 = -4) ∧ (f 2 = 6) →
  (∀ x, f x = 0 ↔ x = -4 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l851_85128


namespace NUMINAMATH_CALUDE_expression_value_l851_85158

theorem expression_value (p q r : ℝ) 
  (hp : p ≠ 2) (hq : q ≠ 5) (hr : r ≠ 7) : 
  ((p - 2) / (7 - r)) * ((q - 5) / (2 - p)) * ((r - 7) / (5 - q)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l851_85158


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l851_85110

theorem least_positive_integer_multiple_of_53 :
  ∃ (x : ℕ+), (2 * x.val)^2 + 2 * 41 * (2 * x.val) + 41^2 ≡ 0 [MOD 53] ∧
  ∀ (y : ℕ+), ((2 * y.val)^2 + 2 * 41 * (2 * y.val) + 41^2 ≡ 0 [MOD 53]) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_of_53_l851_85110


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_l851_85100

theorem wire_length_around_square_field (field_area : Real) (num_rounds : Nat) : 
  field_area = 24336 ∧ num_rounds = 13 → 
  13 * 4 * Real.sqrt field_area = 8112 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_around_square_field_l851_85100


namespace NUMINAMATH_CALUDE_circle_constant_l851_85161

theorem circle_constant (r : ℝ) (k : ℝ) (h1 : r = 36) (h2 : 2 * π * r = 72 * k) : k = π := by
  sorry

end NUMINAMATH_CALUDE_circle_constant_l851_85161


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_product_l851_85103

/-- Represents a point on a 2D grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a rectangle on a 2D grid --/
structure Rectangle where
  topLeft : Point
  topRight : Point
  bottomRight : Point
  bottomLeft : Point

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℕ :=
  (r.topRight.x - r.topLeft.x) * (r.topLeft.y - r.bottomLeft.y)

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℕ :=
  2 * ((r.topRight.x - r.topLeft.x) + (r.topLeft.y - r.bottomLeft.y))

/-- The main theorem to prove --/
theorem rectangle_area_perimeter_product :
  let r := Rectangle.mk
    (Point.mk 1 5) (Point.mk 5 5)
    (Point.mk 5 2) (Point.mk 1 2)
  area r * perimeter r = 168 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_product_l851_85103


namespace NUMINAMATH_CALUDE_susan_remaining_moves_l851_85111

/-- Represents the board game with 100 spaces -/
def BoardGame := 100

/-- Susan's movements over 7 turns -/
def susanMoves : List ℤ := [15, 2, 20, 0, 2, 0, 12]

/-- The total distance Susan has moved -/
def totalDistance : ℤ := susanMoves.sum

/-- Theorem: Susan needs to move 49 more spaces to reach the end -/
theorem susan_remaining_moves : BoardGame - totalDistance = 49 := by
  sorry

end NUMINAMATH_CALUDE_susan_remaining_moves_l851_85111


namespace NUMINAMATH_CALUDE_female_students_count_l851_85153

/-- Given a school with stratified sampling, prove the number of female students -/
theorem female_students_count (total_students sample_size : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : ∃ (girls boys : ℕ), girls + boys = sample_size ∧ boys = girls + 10) :
  (760 : ℝ) = (total_students : ℝ) * (95 : ℝ) / (sample_size : ℝ) :=
sorry

end NUMINAMATH_CALUDE_female_students_count_l851_85153


namespace NUMINAMATH_CALUDE_r_value_when_n_is_3_l851_85141

theorem r_value_when_n_is_3 (n : ℕ) (s : ℕ) (r : ℕ) 
  (h1 : s = 3^n + 2) 
  (h2 : r = 4^s - s) 
  (h3 : n = 3) : 
  r = 4^29 - 29 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_3_l851_85141


namespace NUMINAMATH_CALUDE_division_problem_l851_85109

theorem division_problem (x : ℝ) (h : 0.009 / x = 0.03) : x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l851_85109


namespace NUMINAMATH_CALUDE_boat_distance_theorem_l851_85179

/-- Proves that a boat traveling downstream in 2 hours and upstream in 3 hours,
    with a speed of 5 km/h in still water, covers a distance of 12 km. -/
theorem boat_distance_theorem (boat_speed : ℝ) (downstream_time upstream_time : ℝ) :
  boat_speed = 5 ∧ downstream_time = 2 ∧ upstream_time = 3 →
  ∃ (stream_speed : ℝ),
    (boat_speed + stream_speed) * downstream_time = (boat_speed - stream_speed) * upstream_time ∧
    (boat_speed + stream_speed) * downstream_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_theorem_l851_85179


namespace NUMINAMATH_CALUDE_y_value_l851_85189

theorem y_value (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -5) : y = 18 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l851_85189


namespace NUMINAMATH_CALUDE_max_cards_jasmine_can_buy_l851_85162

/-- The maximum number of cards Jasmine can buy given her budget and the pricing conditions --/
theorem max_cards_jasmine_can_buy :
  let initial_price : ℚ := 95 / 100  -- $0.95 per card
  let discounted_price : ℚ := 85 / 100  -- $0.85 per card
  let budget : ℚ := 9  -- $9.00 budget
  let discount_threshold : ℕ := 6  -- Discount applies after 6 cards

  ∃ (n : ℕ), 
    (n ≤ discount_threshold ∧ n * initial_price ≤ budget) ∨
    (n > discount_threshold ∧ 
     discount_threshold * initial_price + (n - discount_threshold) * discounted_price ≤ budget) ∧
    ∀ (m : ℕ), m > n → 
      (m ≤ discount_threshold → m * initial_price > budget) ∧
      (m > discount_threshold → 
       discount_threshold * initial_price + (m - discount_threshold) * discounted_price > budget) ∧
    n = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_max_cards_jasmine_can_buy_l851_85162


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l851_85197

def cement_bags : ℕ := 500
def cement_price_per_bag : ℚ := 10
def cement_discount_rate : ℚ := 5 / 100
def sand_lorries : ℕ := 20
def sand_tons_per_lorry : ℕ := 10
def sand_price_per_ton : ℚ := 40
def tax_rate_first_half : ℚ := 7 / 100
def tax_rate_second_half : ℚ := 5 / 100

def total_cost : ℚ := sorry

theorem total_cost_is_correct : 
  total_cost = 13230 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l851_85197


namespace NUMINAMATH_CALUDE_smallest_value_when_x_is_9_l851_85106

theorem smallest_value_when_x_is_9 (x : ℝ) (h : x = 9) :
  min (9/x) (min (9/(x+1)) (min (9/(x-2)) (min (9/(6-x)) ((x-2)/9)))) = 9/(x+1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_when_x_is_9_l851_85106


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l851_85193

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  x + 2*y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = x₀*y₀ ∧ x₀ + 2*y₀ = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l851_85193


namespace NUMINAMATH_CALUDE_fourth_degree_polynomial_property_l851_85176

/-- A fourth-degree polynomial with real coefficients -/
def FourthDegreePolynomial : Type := ℝ → ℝ

/-- The property that |f(-2)| = |f(0)| = |f(1)| = |f(3)| = |f(4)| = 16 -/
def HasSpecifiedValues (f : FourthDegreePolynomial) : Prop :=
  |f (-2)| = 16 ∧ |f 0| = 16 ∧ |f 1| = 16 ∧ |f 3| = 16 ∧ |f 4| = 16

/-- The main theorem -/
theorem fourth_degree_polynomial_property (f : FourthDegreePolynomial) 
  (h : HasSpecifiedValues f) : |f 5| = 208 := by
  sorry


end NUMINAMATH_CALUDE_fourth_degree_polynomial_property_l851_85176


namespace NUMINAMATH_CALUDE_rectangle_perimeter_in_square_l851_85178

/-- Given a square of side length y containing a smaller square of side length x,
    the perimeter of one of the four congruent rectangles formed in the remaining area
    is equal to 2y. -/
theorem rectangle_perimeter_in_square (y x : ℝ) (h1 : 0 < y) (h2 : 0 < x) (h3 : x < y) :
  2 * (y - x) + 2 * x = 2 * y :=
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_in_square_l851_85178


namespace NUMINAMATH_CALUDE_school_fire_problem_l851_85132

/-- Represents the initial state and changes in a school after a fire incident -/
structure SchoolState where
  initialClassCount : ℕ
  initialStudentsPerClass : ℕ
  firstUnusableClasses : ℕ
  firstAddedStudents : ℕ
  secondUnusableClasses : ℕ
  secondAddedStudents : ℕ

/-- Calculates the total number of students after the changes -/
def totalStudentsAfterChanges (s : SchoolState) : ℕ :=
  let remainingClasses := s.initialClassCount - s.firstUnusableClasses - s.secondUnusableClasses
  remainingClasses * (s.initialStudentsPerClass + s.firstAddedStudents + s.secondAddedStudents)

/-- Theorem stating that the initial number of students in the school was 900 -/
theorem school_fire_problem (s : SchoolState) 
  (h1 : s.firstUnusableClasses = 6)
  (h2 : s.firstAddedStudents = 5)
  (h3 : s.secondUnusableClasses = 10)
  (h4 : s.secondAddedStudents = 15)
  (h5 : totalStudentsAfterChanges s = s.initialClassCount * s.initialStudentsPerClass) :
  s.initialClassCount * s.initialStudentsPerClass = 900 := by
  sorry

end NUMINAMATH_CALUDE_school_fire_problem_l851_85132


namespace NUMINAMATH_CALUDE_tuesday_to_monday_work_ratio_l851_85137

theorem tuesday_to_monday_work_ratio :
  let monday : ℚ := 3/4
  let wednesday : ℚ := 2/3
  let thursday : ℚ := 5/6
  let friday : ℚ := 75/60
  let total : ℚ := 4
  let tuesday : ℚ := total - (monday + wednesday + thursday + friday)
  tuesday / monday = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_to_monday_work_ratio_l851_85137


namespace NUMINAMATH_CALUDE_range_of_a_l851_85156

def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_decreasing : IsDecreasing f)
  (h_odd : IsOdd f)
  (h_domain : ∀ x, x ∈ Set.Ioo (-1) 1 → f x ∈ Set.univ)
  (h_condition : f (1 - a) + f (1 - 2*a) < 0) :
  0 < a ∧ a < 2/3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l851_85156


namespace NUMINAMATH_CALUDE_trick_decks_cost_l851_85154

def price_per_deck (quantity : ℕ) : ℕ :=
  if quantity ≤ 3 then 8
  else if quantity ≤ 6 then 7
  else 6

def total_cost (victor_decks : ℕ) (friend_decks : ℕ) : ℕ :=
  victor_decks * price_per_deck victor_decks + friend_decks * price_per_deck friend_decks

theorem trick_decks_cost (victor_decks friend_decks : ℕ) 
  (h1 : victor_decks = 6) (h2 : friend_decks = 2) : 
  total_cost victor_decks friend_decks = 58 := by
  sorry

end NUMINAMATH_CALUDE_trick_decks_cost_l851_85154


namespace NUMINAMATH_CALUDE_rubber_band_length_l851_85185

theorem rubber_band_length (r₁ r₂ d : ℝ) (hr₁ : r₁ = 3) (hr₂ : r₂ = 9) (hd : d = 12) :
  ∃ (L : ℝ), L = 4 * Real.pi + 12 * Real.sqrt 3 ∧
  L = 2 * (r₁ * Real.arctan ((Real.sqrt (d^2 - (r₂ - r₁)^2)) / (r₂ - r₁)) +
           r₂ * Real.arctan ((Real.sqrt (d^2 - (r₂ - r₁)^2)) / (r₂ - r₁)) +
           Real.sqrt (d^2 - (r₂ - r₁)^2)) :=
by sorry

end NUMINAMATH_CALUDE_rubber_band_length_l851_85185


namespace NUMINAMATH_CALUDE_existence_condition_equiv_range_l851_85188

theorem existence_condition_equiv_range (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 3 ∧ |x₀^2 - a*x₀ + 4| ≤ 3*x₀) ↔ 
  (2 ≤ a ∧ a ≤ 7 + 1/3) := by
sorry

end NUMINAMATH_CALUDE_existence_condition_equiv_range_l851_85188


namespace NUMINAMATH_CALUDE_diamond_four_three_l851_85147

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := a^2 + a*b - b^3

-- Theorem statement
theorem diamond_four_three : diamond 4 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_diamond_four_three_l851_85147


namespace NUMINAMATH_CALUDE_students_neither_sport_l851_85167

theorem students_neither_sport (total : ℕ) (football : ℕ) (cricket : ℕ) (both : ℕ) :
  total = 470 →
  football = 325 →
  cricket = 175 →
  both = 80 →
  total - (football + cricket - both) = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_neither_sport_l851_85167


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l851_85150

theorem arithmetic_mean_difference (a b c d : ℝ) 
  (h1 : (a + d + b + d) / 2 = 80)
  (h2 : (b + d + c + d) / 2 = 180) : 
  a - c = -200 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l851_85150


namespace NUMINAMATH_CALUDE_volume_of_specific_pyramid_l851_85175

/-- Regular quadrilateral pyramid with given properties -/
structure RegularQuadPyramid where
  -- Point P is on the height VO
  p_on_height : Bool
  -- P is equidistant from base and apex
  p_midpoint : Bool
  -- Distance from P to any side face
  dist_p_to_side : ℝ
  -- Distance from P to base
  dist_p_to_base : ℝ

/-- Volume of a regular quadrilateral pyramid -/
def volume (pyramid : RegularQuadPyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific pyramid -/
theorem volume_of_specific_pyramid :
  ∀ (pyramid : RegularQuadPyramid),
    pyramid.p_on_height ∧
    pyramid.p_midpoint ∧
    pyramid.dist_p_to_side = 3 ∧
    pyramid.dist_p_to_base = 5 →
    volume pyramid = 750 :=
by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_pyramid_l851_85175


namespace NUMINAMATH_CALUDE_extraneous_roots_imply_k_equals_one_l851_85123

-- Define the equation
def equation (x k : ℝ) : Prop :=
  (x - 6) / (x - 5) = k / (5 - x)

-- Define the condition for extraneous roots
def has_extraneous_roots (k : ℝ) : Prop :=
  ∃ x, equation x k ∧ x = 5

-- Theorem statement
theorem extraneous_roots_imply_k_equals_one :
  ∀ k : ℝ, has_extraneous_roots k → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_extraneous_roots_imply_k_equals_one_l851_85123


namespace NUMINAMATH_CALUDE_kenneth_earnings_l851_85130

def earnings_problem (E : ℝ) : Prop :=
  let joystick := 0.10 * E
  let accessories := 0.15 * E
  let phone_bill := 0.05 * E
  let snacks := 0.20 * E - 25
  let utility := 0.25 * E - 15
  let remaining := 405
  E = joystick + accessories + phone_bill + snacks + utility + remaining

theorem kenneth_earnings : 
  ∃ E : ℝ, earnings_problem E ∧ E = 1460 :=
sorry

end NUMINAMATH_CALUDE_kenneth_earnings_l851_85130


namespace NUMINAMATH_CALUDE_group_distribution_methods_l851_85194

theorem group_distribution_methods (total_boys : ℕ) (total_girls : ℕ)
  (group_size : ℕ) (boys_per_group : ℕ) (girls_per_group : ℕ) :
  total_boys = 6 →
  total_girls = 4 →
  group_size = 5 →
  boys_per_group = 3 →
  girls_per_group = 2 →
  (Nat.choose total_boys boys_per_group * Nat.choose total_girls girls_per_group) / 2 = 60 :=
by sorry

end NUMINAMATH_CALUDE_group_distribution_methods_l851_85194


namespace NUMINAMATH_CALUDE_surface_is_cone_l851_85172

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The equation of the surface --/
def surfaceEquation (p : Point3D) (a b c d θ : ℝ) : Prop :=
  (p.x - a)^2 + (p.y - b)^2 + (p.z - c)^2 = (d * Real.cos θ)^2

/-- The set of points satisfying the equation --/
def surfaceSet (a b c d : ℝ) : Set Point3D :=
  {p : Point3D | ∃ θ, surfaceEquation p a b c d θ}

/-- Definition of a cone --/
def isCone (S : Set Point3D) : Prop :=
  ∃ v : Point3D, ∃ axis : Point3D → Point3D → Prop,
    ∀ p ∈ S, ∃ r θ : ℝ, r ≥ 0 ∧ 
      p = Point3D.mk (v.x + r * Real.cos θ) (v.y + r * Real.sin θ) (v.z + r)

theorem surface_is_cone (d : ℝ) (h : d > 0) :
  isCone (surfaceSet 0 0 0 d) := by
  sorry

end NUMINAMATH_CALUDE_surface_is_cone_l851_85172


namespace NUMINAMATH_CALUDE_estimate_red_balls_l851_85117

theorem estimate_red_balls (black_balls : ℕ) (total_draws : ℕ) (black_draws : ℕ) : 
  black_balls = 4 → 
  total_draws = 100 → 
  black_draws = 40 → 
  ∃ red_balls : ℕ, (black_balls : ℚ) / (black_balls + red_balls : ℚ) = 2 / 5 ∧ red_balls = 6 :=
by sorry

end NUMINAMATH_CALUDE_estimate_red_balls_l851_85117


namespace NUMINAMATH_CALUDE_student_selection_probability_l851_85148

theorem student_selection_probability (b g o : ℝ) : 
  b + g + o = 1 →  -- total probability
  b > 0 ∧ g > 0 ∧ o > 0 →  -- probabilities are positive
  b = (1/2) * o →  -- boy probability is half of other
  g = o - b →  -- girl probability is difference between other and boy
  b = 1/4 :=  -- ratio of boys to total is 1/4
by sorry

end NUMINAMATH_CALUDE_student_selection_probability_l851_85148


namespace NUMINAMATH_CALUDE_non_basketball_theater_percentage_l851_85135

/-- Represents the student body of Maple Town High School -/
structure School where
  total : ℝ
  basketball : ℝ
  theater : ℝ
  both : ℝ

/-- The conditions given in the problem -/
def school_conditions (s : School) : Prop :=
  s.basketball = 0.7 * s.total ∧
  s.theater = 0.4 * s.total ∧
  s.both = 0.2 * s.basketball ∧
  (s.basketball - s.both) = 0.6 * (s.total - s.theater)

/-- The theorem to be proved -/
theorem non_basketball_theater_percentage (s : School) 
  (h : school_conditions s) : 
  (s.theater - s.both) / (s.total - s.basketball) = 0.87 := by
  sorry

end NUMINAMATH_CALUDE_non_basketball_theater_percentage_l851_85135


namespace NUMINAMATH_CALUDE_smallest_n_for_3003_terms_l851_85118

theorem smallest_n_for_3003_terms : ∃ (N : ℕ), 
  (N = 19) ∧ 
  (∀ k < N, (Nat.choose (k + 1) 5) < 3003) ∧
  (Nat.choose (N + 1) 5 = 3003) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_3003_terms_l851_85118


namespace NUMINAMATH_CALUDE_students_opted_both_math_and_science_l851_85190

theorem students_opted_both_math_and_science 
  (total_students : ℕ) 
  (not_math : ℕ) 
  (not_science : ℕ) 
  (not_either : ℕ) 
  (h1 : total_students = 40)
  (h2 : not_math = 10)
  (h3 : not_science = 15)
  (h4 : not_either = 2) :
  total_students - (not_math + not_science - not_either) = 17 := by
  sorry

#check students_opted_both_math_and_science

end NUMINAMATH_CALUDE_students_opted_both_math_and_science_l851_85190


namespace NUMINAMATH_CALUDE_divisors_of_90_l851_85177

def n : ℕ := 90

/-- The number of positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The sum of all positive divisors of n -/
def sum_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem divisors_of_90 :
  num_divisors n = 12 ∧ sum_divisors n = 234 := by sorry

end NUMINAMATH_CALUDE_divisors_of_90_l851_85177


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l851_85160

/-- A piecewise function f defined on real numbers -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then x^2 - 2*x else a*x - 1

/-- Proposition: If f is monotonically increasing on ℝ, then 0 < a ≤ 1/2 -/
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (0 < a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l851_85160


namespace NUMINAMATH_CALUDE_john_money_left_l851_85102

/-- The amount of money John has left after shopping --/
def money_left (initial_amount : ℝ) (roast_cost : ℝ) (vegetable_cost : ℝ) (wine_cost : ℝ) (dessert_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  let total_cost := roast_cost + vegetable_cost + wine_cost + dessert_cost
  let discounted_cost := total_cost * (1 - discount_rate)
  initial_amount - discounted_cost

/-- Theorem stating that John has €56.8 left after shopping --/
theorem john_money_left :
  money_left 100 17 11 12 8 0.1 = 56.8 := by
  sorry

end NUMINAMATH_CALUDE_john_money_left_l851_85102


namespace NUMINAMATH_CALUDE_function_symmetry_l851_85108

/-- For any function f(x) = x^5 - ax^3 + bx + 2, f(x) + f(-x) = 4 for all real x -/
theorem function_symmetry (a b : ℝ) :
  let f := fun (x : ℝ) => x^5 - a*x^3 + b*x + 2
  ∀ x, f x + f (-x) = 4 := by sorry

end NUMINAMATH_CALUDE_function_symmetry_l851_85108


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l851_85195

-- Define a circle with a center and radius
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of two circles being non-intersecting
def non_intersecting (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 ≥ (c1.radius + c2.radius)^2

-- Define the property of a circle intersecting another circle
def intersects (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 ≤ (c1.radius + c2.radius)^2

-- Theorem statement
theorem circle_intersection_theorem 
  (c1 c2 c3 c4 c5 c6 : Circle) 
  (h1 : c1.radius ≥ 1) 
  (h2 : c2.radius ≥ 1) 
  (h3 : c3.radius ≥ 1) 
  (h4 : c4.radius ≥ 1) 
  (h5 : c5.radius ≥ 1) 
  (h6 : c6.radius ≥ 1) 
  (h_non_intersect : 
    non_intersecting c1 c2 ∧ non_intersecting c1 c3 ∧ non_intersecting c1 c4 ∧ 
    non_intersecting c1 c5 ∧ non_intersecting c1 c6 ∧ non_intersecting c2 c3 ∧ 
    non_intersecting c2 c4 ∧ non_intersecting c2 c5 ∧ non_intersecting c2 c6 ∧ 
    non_intersecting c3 c4 ∧ non_intersecting c3 c5 ∧ non_intersecting c3 c6 ∧ 
    non_intersecting c4 c5 ∧ non_intersecting c4 c6 ∧ non_intersecting c5 c6)
  (c : Circle)
  (h_intersect : 
    intersects c c1 ∧ intersects c c2 ∧ intersects c c3 ∧ 
    intersects c c4 ∧ intersects c c5 ∧ intersects c c6) :
  c.radius ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_circle_intersection_theorem_l851_85195


namespace NUMINAMATH_CALUDE_brownies_on_counter_l851_85157

def initial_brownies : ℕ := 24
def father_ate : ℕ := 8
def mooney_ate : ℕ := 4
def mother_added : ℕ := 24

theorem brownies_on_counter : 
  initial_brownies - father_ate - mooney_ate + mother_added = 36 := by
  sorry

end NUMINAMATH_CALUDE_brownies_on_counter_l851_85157


namespace NUMINAMATH_CALUDE_cubic_roots_from_quadratic_l851_85113

theorem cubic_roots_from_quadratic (b c : ℝ) :
  let x₁ := b + c
  let x₂ := b - c
  (∀ x, x^2 - 2*b*x + b^2 - c^2 = 0 ↔ x = x₁ ∨ x = x₂) →
  (∀ x, x^2 - 2*b*(b^2 + 3*c^2)*x + (b^2 - c^2)^3 = 0 ↔ x = x₁^3 ∨ x = x₂^3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_from_quadratic_l851_85113


namespace NUMINAMATH_CALUDE_twenty_percent_value_l851_85114

theorem twenty_percent_value (x : ℝ) (h : 1.2 * x = 1200) : 0.2 * x = 200 := by
  sorry

end NUMINAMATH_CALUDE_twenty_percent_value_l851_85114


namespace NUMINAMATH_CALUDE_deans_height_l851_85181

theorem deans_height (depth water_depth : ℝ) (h1 : water_depth = 10 * depth) (h2 : water_depth = depth + 81) : depth = 9 := by
  sorry

end NUMINAMATH_CALUDE_deans_height_l851_85181


namespace NUMINAMATH_CALUDE_prob_specific_quarter_is_one_eighth_l851_85115

/-- Represents a piece of paper with two sides, each divided into four quarters -/
structure Paper :=
  (sides : Fin 2)
  (quarters : Fin 4)

/-- The total number of distinct parts (quarters) on the paper -/
def total_parts : ℕ := 8

/-- The probability of a specific quarter being on top after random folding -/
def prob_specific_quarter_on_top : ℚ := 1 / 8

/-- Theorem stating that the probability of a specific quarter being on top is 1/8 -/
theorem prob_specific_quarter_is_one_eighth :
  prob_specific_quarter_on_top = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_specific_quarter_is_one_eighth_l851_85115


namespace NUMINAMATH_CALUDE_last_digit_of_tower_of_power_l851_85121

def tower_of_power (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | n + 1 => 2^(tower_of_power n)

theorem last_digit_of_tower_of_power :
  tower_of_power 2007 % 10 = 6 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_tower_of_power_l851_85121


namespace NUMINAMATH_CALUDE_king_middle_school_teachers_l851_85186

/-- Calculates the number of teachers at King Middle School given the specified conditions -/
theorem king_middle_school_teachers :
  let num_students : ℕ := 1500
  let classes_per_student : ℕ := 5
  let classes_per_teacher : ℕ := 5
  let students_per_class : ℕ := 25
  let total_class_instances : ℕ := num_students * classes_per_student
  let unique_classes : ℕ := total_class_instances / students_per_class
  let num_teachers : ℕ := unique_classes / classes_per_teacher
  num_teachers = 60 := by sorry

end NUMINAMATH_CALUDE_king_middle_school_teachers_l851_85186


namespace NUMINAMATH_CALUDE_solution_set_implies_a_range_l851_85122

theorem solution_set_implies_a_range (a : ℝ) : 
  (∃ P : Set ℝ, (∀ x ∈ P, (x + 1) / (x + a) < 2) ∧ 1 ∉ P) → 
  a ∈ Set.Icc (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_range_l851_85122


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l851_85191

theorem arithmetic_calculation : 12 * 11 + 7 * 8 - 5 * 6 + 10 * 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l851_85191


namespace NUMINAMATH_CALUDE_inequality_proof_l851_85142

theorem inequality_proof (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l851_85142


namespace NUMINAMATH_CALUDE_tangent_line_to_reciprocal_curve_l851_85151

theorem tangent_line_to_reciprocal_curve (a : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ -x + a = 1/x ∧ -1 = -(1/x^2)) → (a = 2 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_reciprocal_curve_l851_85151


namespace NUMINAMATH_CALUDE_field_width_l851_85173

/-- The width of a rectangular field given its area and length -/
theorem field_width (area : ℝ) (length : ℝ) (h1 : area = 143.2) (h2 : length = 4) :
  area / length = 35.8 := by
sorry

end NUMINAMATH_CALUDE_field_width_l851_85173


namespace NUMINAMATH_CALUDE_shaded_rectangle_probability_l851_85125

theorem shaded_rectangle_probability : 
  let width : ℕ := 2004
  let height : ℕ := 2
  let shaded_pos1 : ℕ := 501
  let shaded_pos2 : ℕ := 1504
  let total_rectangles : ℕ := height * (width.choose 2)
  let shaded_rectangles_per_row : ℕ := 
    shaded_pos1 * (width - shaded_pos1 + 1) + 
    (shaded_pos2 - shaded_pos1) * (width - shaded_pos2 + 1)
  let total_shaded_rectangles : ℕ := height * shaded_rectangles_per_row
  (total_rectangles - total_shaded_rectangles : ℚ) / total_rectangles = 1501 / 4008 :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_rectangle_probability_l851_85125


namespace NUMINAMATH_CALUDE_bakery_items_l851_85136

theorem bakery_items (total : ℕ) (bread_rolls : ℕ) (croissants : ℕ) (bagels : ℕ)
  (h1 : total = 90)
  (h2 : bread_rolls = 49)
  (h3 : croissants = 19)
  (h4 : total = bread_rolls + croissants + bagels) :
  bagels = 22 := by
sorry

end NUMINAMATH_CALUDE_bakery_items_l851_85136


namespace NUMINAMATH_CALUDE_water_needed_in_quarts_l851_85133

/-- Represents the ratio of water to lemon juice in the lemonade mixture -/
def water_to_lemon_ratio : ℚ := 4

/-- Represents the total number of parts in the mixture -/
def total_parts : ℚ := water_to_lemon_ratio + 1

/-- Represents the total volume of the mixture in gallons -/
def total_volume : ℚ := 1

/-- Represents the number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- Theorem stating the amount of water needed in quarts -/
theorem water_needed_in_quarts : 
  (water_to_lemon_ratio / total_parts) * total_volume * quarts_per_gallon = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_in_quarts_l851_85133


namespace NUMINAMATH_CALUDE_triangle_abc_problem_l851_85192

theorem triangle_abc_problem (a b c : ℝ) (A B C : ℝ) :
  A = π / 6 →
  (1 + Real.sqrt 3) * c = 2 * b →
  b * a * Real.cos C = 1 + Real.sqrt 3 →
  C = π / 4 ∧ a = Real.sqrt 2 ∧ b = 1 + Real.sqrt 3 ∧ c = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_problem_l851_85192


namespace NUMINAMATH_CALUDE_not_prime_sum_of_squares_l851_85174

/-- The equation has exactly two positive integer roots -/
def has_two_positive_integer_roots (a b : ℝ) : Prop :=
  ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x ≠ y ∧
    (∀ z : ℤ, z > 0 → a * z * (z^2 + a * z + 1) = b * (z^2 + b + 1) ↔ z = x ∨ z = y)

/-- Main theorem -/
theorem not_prime_sum_of_squares (a b : ℝ) :
  ab < 0 →
  has_two_positive_integer_roots a b →
  ¬ Nat.Prime (Int.natAbs (Int.floor (a^2 + b^2))) :=
sorry

end NUMINAMATH_CALUDE_not_prime_sum_of_squares_l851_85174


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l851_85145

def total_employees : ℕ := 10
def men : ℕ := 6
def women : ℕ := 4
def unavailable_men : ℕ := 1
def unavailable_women : ℕ := 1
def selection_size : ℕ := 3

def available_men : ℕ := men - unavailable_men
def available_women : ℕ := women - unavailable_women
def total_available : ℕ := available_men + available_women

theorem probability_at_least_one_woman :
  (1 - (Nat.choose available_men selection_size : ℚ) / (Nat.choose total_available selection_size : ℚ)) = 23/28 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l851_85145


namespace NUMINAMATH_CALUDE_log4_of_16_equals_2_l851_85168

-- Define the logarithm function for base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- State the theorem
theorem log4_of_16_equals_2 : log4 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log4_of_16_equals_2_l851_85168


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_13_14_l851_85199

theorem smallest_divisible_by_12_13_14 : ∃ n : ℕ, n > 0 ∧ 12 ∣ n ∧ 13 ∣ n ∧ 14 ∣ n ∧ ∀ m : ℕ, m > 0 → 12 ∣ m → 13 ∣ m → 14 ∣ m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_13_14_l851_85199


namespace NUMINAMATH_CALUDE_hat_number_sum_l851_85140

/-- A four-digit perfect square number with tens digit 0 and non-zero units digit -/
def ValidNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ k : ℕ, n = k^2 ∧ (n / 10) % 10 = 0 ∧ n % 10 ≠ 0

/-- The property that two numbers have the same units digit -/
def SameUnitsDigit (a b : ℕ) : Prop := a % 10 = b % 10

/-- The property that a number has an even units digit -/
def EvenUnitsDigit (n : ℕ) : Prop := n % 2 = 0

theorem hat_number_sum :
  ∀ a b c : ℕ,
    ValidNumber a ∧ ValidNumber b ∧ ValidNumber c ∧
    SameUnitsDigit b c ∧
    EvenUnitsDigit a ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    a + b + c = 14612 := by
  sorry

end NUMINAMATH_CALUDE_hat_number_sum_l851_85140


namespace NUMINAMATH_CALUDE_mango_count_proof_l851_85165

/-- Given a ratio of mangoes to apples and the number of apples, 
    calculate the number of mangoes -/
def calculate_mangoes (mango_ratio : ℕ) (apple_ratio : ℕ) (apple_count : ℕ) : ℕ :=
  (mango_ratio * apple_count) / apple_ratio

/-- Theorem: Given the ratio of mangoes to apples is 10:3 and there are 36 apples,
    prove that the number of mangoes is 120 -/
theorem mango_count_proof :
  calculate_mangoes 10 3 36 = 120 := by
  sorry

end NUMINAMATH_CALUDE_mango_count_proof_l851_85165


namespace NUMINAMATH_CALUDE_length_of_AB_l851_85166

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = -12*x

-- Define the line
def line (x y : ℝ) : Prop := y = x + 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ parabola_C B.1 B.2 ∧ 
  line A.1 A.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) : 
  intersection_points A B → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 30 :=
sorry

end NUMINAMATH_CALUDE_length_of_AB_l851_85166


namespace NUMINAMATH_CALUDE_midpoint_fraction_l851_85170

theorem midpoint_fraction : ∃ (n d : ℕ), d ≠ 0 ∧ (n : ℚ) / d = (3 : ℚ) / 4 / 2 + (5 : ℚ) / 6 / 2 ∧ n = 19 ∧ d = 24 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_fraction_l851_85170


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l851_85104

theorem imaginary_part_of_complex_expression : 
  Complex.im ((2 * Complex.I) / (1 - Complex.I) + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l851_85104


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l851_85159

/-- Represents the problem of calculating sales tax percentage --/
theorem sales_tax_percentage
  (total_worth : ℝ)
  (tax_rate : ℝ)
  (tax_free_cost : ℝ)
  (h1 : total_worth = 40)
  (h2 : tax_rate = 0.06)
  (h3 : tax_free_cost = 34.7) :
  (total_worth - tax_free_cost) * tax_rate / total_worth = 0.0075 := by
  sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l851_85159


namespace NUMINAMATH_CALUDE_isosceles_triangle_l851_85107

theorem isosceles_triangle (A B C : ℝ) (h1 : 0 < A ∧ A < π)
    (h2 : 0 < B ∧ B < π) (h3 : 0 < C ∧ C < π) (h4 : A + B + C = π)
    (h5 : 2 * Real.cos B * Real.sin A = Real.sin C) : A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l851_85107


namespace NUMINAMATH_CALUDE_garden_flowers_l851_85131

theorem garden_flowers (roses tulips : ℕ) (percent_not_roses : ℚ) (total daisies : ℕ) : 
  roses = 25 →
  tulips = 40 →
  percent_not_roses = 3/4 →
  total = roses + tulips + daisies →
  (total : ℚ) * (1 - percent_not_roses) = roses →
  daisies = 35 :=
by sorry

end NUMINAMATH_CALUDE_garden_flowers_l851_85131


namespace NUMINAMATH_CALUDE_equation_solution_l851_85119

theorem equation_solution : 
  ∃ x : ℚ, (x^2 + x + 1) / (x + 2) = x + 1 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l851_85119


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l851_85127

theorem other_root_of_quadratic (a : ℝ) : 
  ((-1)^2 + a*(-1) - 2 = 0) → (2^2 + a*2 - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l851_85127


namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l851_85198

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 6*a*x^2

theorem tangent_line_and_monotonicity :
  -- Part I: Tangent line equation
  (∀ x y : ℝ, y = f (-1) x → (x = 1 ∧ y = 7) →
    ∃ k m : ℝ, k = 15 ∧ m = -8 ∧ k*x + (-1)*y + m = 0) ∧
  -- Part II: Monotonicity
  (∀ a : ℝ,
    -- Case a = 0
    (a = 0 → ∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ < f a x₂) ∧
    -- Case a < 0
    (a < 0 → ∀ x₁ x₂ : ℝ,
      ((x₁ < x₂ ∧ x₂ < 4*a) ∨ (x₁ < x₂ ∧ 0 < x₁)) → f a x₁ < f a x₂) ∧
    (a < 0 → ∀ x₁ x₂ : ℝ, (4*a < x₁ ∧ x₁ < x₂ ∧ x₂ < 0) → f a x₁ > f a x₂) ∧
    -- Case a > 0
    (a > 0 → ∀ x₁ x₂ : ℝ,
      ((x₁ < x₂ ∧ x₂ < 0) ∨ (4*a < x₁ ∧ x₁ < x₂)) → f a x₁ < f a x₂) ∧
    (a > 0 → ∀ x₁ x₂ : ℝ, (0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 4*a) → f a x₁ > f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l851_85198


namespace NUMINAMATH_CALUDE_triangle_area_squared_l851_85134

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circle
def Circle := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 = 16}

-- Define the conditions
def isInscribed (t : Triangle) : Prop :=
  t.A ∈ Circle ∧ t.B ∈ Circle ∧ t.C ∈ Circle

def angleA (t : Triangle) : ℝ := sorry

def sideDifference (t : Triangle) : ℝ := sorry

def area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem triangle_area_squared (t : Triangle) 
  (h1 : isInscribed t)
  (h2 : angleA t = π / 3)  -- 60 degrees in radians
  (h3 : sideDifference t = 4)
  : (area t)^2 = 192 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_squared_l851_85134


namespace NUMINAMATH_CALUDE_bizarre_coin_expected_value_l851_85149

/-- A bizarre weighted coin with three possible outcomes -/
inductive CoinOutcome
| Heads
| Tails
| Edge

/-- The probability of each outcome for the bizarre weighted coin -/
def probability (outcome : CoinOutcome) : ℚ :=
  match outcome with
  | CoinOutcome.Heads => 1/4
  | CoinOutcome.Tails => 1/2
  | CoinOutcome.Edge => 1/4

/-- The payoff for each outcome of the bizarre weighted coin -/
def payoff (outcome : CoinOutcome) : ℤ :=
  match outcome with
  | CoinOutcome.Heads => 1
  | CoinOutcome.Tails => 3
  | CoinOutcome.Edge => -8

/-- The expected value of flipping the bizarre weighted coin -/
def expected_value : ℚ :=
  (probability CoinOutcome.Heads * payoff CoinOutcome.Heads) +
  (probability CoinOutcome.Tails * payoff CoinOutcome.Tails) +
  (probability CoinOutcome.Edge * payoff CoinOutcome.Edge)

/-- Theorem stating that the expected value of flipping the bizarre weighted coin is -1/4 -/
theorem bizarre_coin_expected_value :
  expected_value = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_bizarre_coin_expected_value_l851_85149


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l851_85184

def repeating_decimal_4 : ℚ := 4/9
def repeating_decimal_7 : ℚ := 7/9

theorem product_of_repeating_decimals :
  repeating_decimal_4 * repeating_decimal_7 = 28/81 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l851_85184


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l851_85120

def A : Set ℤ := {0, 1}
def B : Set ℤ := {0, -1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l851_85120


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l851_85105

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1)*x + 1

-- State the theorem
theorem increasing_function_a_range :
  (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) →
  a ∈ Set.Ici (-(3/2)) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l851_85105


namespace NUMINAMATH_CALUDE_a_gt_b_neither_sufficient_nor_necessary_for_a_sq_gt_b_sq_l851_85139

theorem a_gt_b_neither_sufficient_nor_necessary_for_a_sq_gt_b_sq :
  ∃ a b : ℝ, (a > b ∧ ¬(a^2 > b^2)) ∧ ∃ c d : ℝ, (c^2 > d^2 ∧ ¬(c > d)) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_neither_sufficient_nor_necessary_for_a_sq_gt_b_sq_l851_85139


namespace NUMINAMATH_CALUDE_max_angle_MPN_x_coordinate_l851_85101

/-- The x-coordinate of point P when angle MPN is maximum -/
def max_angle_x_coordinate : ℝ := 1

/-- Point M with coordinates (-1, 2) -/
def M : ℝ × ℝ := (-1, 2)

/-- Point N with coordinates (1, 4) -/
def N : ℝ × ℝ := (1, 4)

/-- Point P moves on the positive half of the x-axis -/
def P (x : ℝ) : ℝ × ℝ := (x, 0)

/-- The angle MPN as a function of the x-coordinate of P -/
noncomputable def angle_MPN (x : ℝ) : ℝ := sorry

theorem max_angle_MPN_x_coordinate :
  ∃ (x : ℝ), x > 0 ∧ 
  (∀ (y : ℝ), y > 0 → angle_MPN y ≤ angle_MPN x) ∧
  x = max_angle_x_coordinate := by sorry

end NUMINAMATH_CALUDE_max_angle_MPN_x_coordinate_l851_85101
