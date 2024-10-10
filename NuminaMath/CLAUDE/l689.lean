import Mathlib

namespace student_venue_arrangements_l689_68989

theorem student_venue_arrangements (n : Nat) (a b c : Nat) 
  (h1 : n = 6)
  (h2 : a = 3)
  (h3 : b = 1)
  (h4 : c = 2)
  (h5 : a + b + c = n) :
  Nat.choose n a * Nat.choose (n - a) b * Nat.choose (n - a - b) c = 60 :=
by sorry

end student_venue_arrangements_l689_68989


namespace triangle_n_range_l689_68926

-- Define a triangle with the given properties
structure Triangle where
  n : ℝ
  angle1 : ℝ := 180 - n
  angle2 : ℝ
  angle3 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 = 180
  angle_difference : max angle1 (max angle2 angle3) - min angle1 (min angle2 angle3) = 24

-- Theorem statement
theorem triangle_n_range (t : Triangle) : 104 ≤ t.n ∧ t.n ≤ 136 := by
  sorry

end triangle_n_range_l689_68926


namespace arithmetic_mean_three_digit_multiples_of_seven_l689_68904

theorem arithmetic_mean_three_digit_multiples_of_seven :
  let first : ℕ := 105
  let last : ℕ := 994
  let count : ℕ := 128
  let sum : ℕ := count * (first + last) / 2
  (sum : ℚ) / count = 549.5 :=
by sorry

end arithmetic_mean_three_digit_multiples_of_seven_l689_68904


namespace afternoon_sales_proof_l689_68948

/-- A salesman sells pears in the morning and afternoon. -/
structure PearSales where
  morning : ℝ
  afternoon : ℝ

/-- The total amount of pears sold in a day. -/
def total_sales (s : PearSales) : ℝ := s.morning + s.afternoon

/-- Theorem: Given a salesman who sold twice as much pears in the afternoon than in the morning,
    and sold 390 kilograms in total that day, the amount sold in the afternoon is 260 kilograms. -/
theorem afternoon_sales_proof (s : PearSales) 
    (h1 : s.afternoon = 2 * s.morning) 
    (h2 : total_sales s = 390) : 
    s.afternoon = 260 := by
  sorry

end afternoon_sales_proof_l689_68948


namespace sandys_initial_money_l689_68901

/-- Sandy's shopping problem -/
theorem sandys_initial_money 
  (shirt_cost : ℝ) 
  (jacket_cost : ℝ) 
  (pocket_money : ℝ) 
  (h1 : shirt_cost = 12.14)
  (h2 : jacket_cost = 9.28)
  (h3 : pocket_money = 7.43) :
  shirt_cost + jacket_cost + pocket_money = 28.85 := by
sorry

end sandys_initial_money_l689_68901


namespace absolute_value_inequality_l689_68917

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x| ≥ a * x) → |a| ≤ 1 := by
  sorry

end absolute_value_inequality_l689_68917


namespace larger_interior_angle_measure_l689_68950

/-- A circular arch bridge constructed with congruent isosceles trapezoids -/
structure CircularArchBridge where
  /-- The number of trapezoids in the bridge construction -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of each trapezoid in degrees -/
  larger_interior_angle : ℝ
  /-- The two end trapezoids rest horizontally on the ground -/
  end_trapezoids_horizontal : Prop

/-- Theorem stating the measure of the larger interior angle in a circular arch bridge with 12 trapezoids -/
theorem larger_interior_angle_measure (bridge : CircularArchBridge) 
  (h1 : bridge.num_trapezoids = 12)
  (h2 : bridge.end_trapezoids_horizontal) :
  bridge.larger_interior_angle = 97.5 := by
  sorry

#check larger_interior_angle_measure

end larger_interior_angle_measure_l689_68950


namespace marie_gift_boxes_l689_68905

/-- Represents the number of gift boxes Marie used to pack chocolate eggs. -/
def num_gift_boxes (total_eggs : ℕ) (egg_weight : ℕ) (remaining_weight : ℕ) : ℕ :=
  let total_weight := total_eggs * egg_weight
  let melted_weight := total_weight - remaining_weight
  let eggs_per_box := melted_weight / egg_weight
  total_eggs / eggs_per_box

/-- Proves that Marie packed the chocolate eggs in 4 gift boxes. -/
theorem marie_gift_boxes :
  num_gift_boxes 12 10 90 = 4 := by
  sorry

end marie_gift_boxes_l689_68905


namespace problem_statement_l689_68961

theorem problem_statement (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (heq1 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2012)
  (heq2 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2012) :
  (a*b)^2012 - (c*d)^2012 = -2012 := by
  sorry

end problem_statement_l689_68961


namespace factorization_equality_l689_68977

/-- Proves that the factorization of 3x(x - 5) + 4(x - 5) - 2x^2 is (x - 15)(x + 4) for all real x -/
theorem factorization_equality (x : ℝ) : 3*x*(x - 5) + 4*(x - 5) - 2*x^2 = (x - 15)*(x + 4) := by
  sorry

end factorization_equality_l689_68977


namespace intersection_tangents_perpendicular_l689_68982

/-- Two circles in the plane -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0

def circle2 (a x y : ℝ) : Prop := x^2 + y^2 + 2*(a-1)*x + 2*y + a^2 = 0

/-- The common chord of the two circles -/
def common_chord (a x y : ℝ) : Prop := 2*(a-1)*x - 2*y + a^2 = 0

/-- The condition for perpendicular tangents at intersection points -/
def perpendicular_tangents (a x y : ℝ) : Prop :=
  (y + 2) / x * (y + 1) / (x - (1 - a)) = -1

/-- The main theorem -/
theorem intersection_tangents_perpendicular (a : ℝ) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 a x y ∧ common_chord a x y ∧ perpendicular_tangents a x y) →
  a = -2 :=
sorry

end intersection_tangents_perpendicular_l689_68982


namespace quadratic_symmetric_derivative_l689_68920

-- Define a quadratic function symmetric about x = 1
def f (a b : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + b

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2 * a * (x - 1)

theorem quadratic_symmetric_derivative (a b : ℝ) :
  (f' a 0 = -2) → (f' a 2 = 2) := by
  sorry

end quadratic_symmetric_derivative_l689_68920


namespace floor_sqrt_20_squared_l689_68928

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end floor_sqrt_20_squared_l689_68928


namespace highest_score_l689_68945

theorem highest_score (total_innings : ℕ) (overall_average : ℚ) (score_difference : ℕ) (average_without_extremes : ℚ) :
  total_innings = 46 →
  overall_average = 63 →
  score_difference = 150 →
  average_without_extremes = 58 →
  ∃ (highest_score lowest_score : ℕ),
    highest_score - lowest_score = score_difference ∧
    (total_innings : ℚ) * overall_average = (total_innings - 2 : ℚ) * average_without_extremes + highest_score + lowest_score ∧
    highest_score = 248 := by
  sorry

end highest_score_l689_68945


namespace tim_balloons_l689_68995

theorem tim_balloons (dan_balloons : ℝ) (ratio : ℝ) : 
  dan_balloons = 29.0 → 
  ratio = 7.0 → 
  ⌊dan_balloons / ratio⌋ = 4 := by
sorry

end tim_balloons_l689_68995


namespace least_lcm_a_c_l689_68956

theorem least_lcm_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 12) (h2 : Nat.lcm b c = 15) :
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 20 ∧ (∀ (x y : ℕ), Nat.lcm x b = 12 → Nat.lcm b y = 15 → Nat.lcm a' c' ≤ Nat.lcm x y) :=
sorry

end least_lcm_a_c_l689_68956


namespace equation_solution_l689_68985

theorem equation_solution : ∃ x : ℝ, ((x * 0.85) / 2.5) - (8 * 2.25) = 5.5 := by
  sorry

end equation_solution_l689_68985


namespace smallest_multiple_of_6_and_15_l689_68991

theorem smallest_multiple_of_6_and_15 :
  ∃ b : ℕ+, (∀ n : ℕ+, 6 ∣ n ∧ 15 ∣ n → b ≤ n) ∧ 6 ∣ b ∧ 15 ∣ b ∧ b = 30 := by
  sorry

end smallest_multiple_of_6_and_15_l689_68991


namespace total_trees_is_fifteen_l689_68931

/-- The number of apple trees Ava planted -/
def ava_trees : ℕ := 9

/-- The difference between Ava's and Lily's trees -/
def difference : ℕ := 3

/-- The number of apple trees Lily planted -/
def lily_trees : ℕ := ava_trees - difference

/-- The total number of apple trees planted by Ava and Lily -/
def total_trees : ℕ := ava_trees + lily_trees

theorem total_trees_is_fifteen : total_trees = 15 := by
  sorry

end total_trees_is_fifteen_l689_68931


namespace intersection_of_A_and_B_l689_68966

def A : Set (ℝ × ℝ) := {p | 3 * p.1 - p.2 = 7}
def B : Set (ℝ × ℝ) := {p | 2 * p.1 + p.2 = 3}

theorem intersection_of_A_and_B : A ∩ B = {(2, -1)} := by
  sorry

end intersection_of_A_and_B_l689_68966


namespace not_cylinder_if_triangle_front_view_l689_68996

/-- A type representing geometric bodies -/
inductive GeometricBody
  | Cylinder
  | Cone
  | Tetrahedron
  | TriangularPrism

/-- A type representing possible front views -/
inductive FrontView
  | Triangle
  | Rectangle
  | Circle

/-- A function that returns the front view of a geometric body -/
def frontView (body : GeometricBody) : FrontView :=
  match body with
  | GeometricBody.Cylinder => FrontView.Rectangle
  | GeometricBody.Cone => FrontView.Triangle
  | GeometricBody.Tetrahedron => FrontView.Triangle
  | GeometricBody.TriangularPrism => FrontView.Triangle

/-- Theorem: If a geometric body has a triangle as its front view, it cannot be a cylinder -/
theorem not_cylinder_if_triangle_front_view (body : GeometricBody) :
  frontView body = FrontView.Triangle → body ≠ GeometricBody.Cylinder :=
by
  sorry

end not_cylinder_if_triangle_front_view_l689_68996


namespace range_of_a_for_intersection_equality_l689_68993

/-- The set A defined by the quadratic equation x^2 - 3x + 2 = 0 -/
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

/-- The set B defined by the quadratic equation x^2 - ax + 3a - 5 = 0, parameterized by a -/
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + 3*a - 5 = 0}

/-- The theorem stating the range of a for which A ∩ B = B -/
theorem range_of_a_for_intersection_equality :
  ∀ a : ℝ, (A ∩ B a = B a) → (a ∈ Set.Icc 2 10 ∪ {1}) :=
by sorry

end range_of_a_for_intersection_equality_l689_68993


namespace ellipse_perpendicular_point_l689_68990

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the left focus F
def left_focus : ℝ × ℝ := (-1, 0)

-- Define line l (implicitly through its properties)
def line_l (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x + 1)

-- Define points A and B
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the perpendicular point H
def point_H : ℝ × ℝ := sorry

-- State the theorem
theorem ellipse_perpendicular_point :
  ∀ (A B : ℝ × ℝ),
    ellipse A.1 A.2 →
    ellipse B.1 B.2 →
    line_l A.1 A.2 →
    line_l B.1 B.2 →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    (point_H = (-2/3, Real.sqrt 2/3) ∨ point_H = (-2/3, -Real.sqrt 2/3)) :=
sorry

end ellipse_perpendicular_point_l689_68990


namespace function_transformation_l689_68987

/-- Given a function f such that f(1/x) = x/(1-x) for all x ≠ 0 and x ≠ 1,
    prove that f(x) = 1/(x-1) for all x ≠ 0 and x ≠ 1. -/
theorem function_transformation (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f (1/x) = x / (1 - x)) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f x = 1 / (x - 1)) :=
by sorry

end function_transformation_l689_68987


namespace area_of_quadrilateral_DFEJ_l689_68970

/-- Given a right isosceles triangle ABC with side lengths AB = AC = 10 and BC = 10√2,
    and points D, E, F as midpoints of AB, BC, AC respectively,
    and J as the midpoint of DE,
    the area of quadrilateral DFEJ is 6.25. -/
theorem area_of_quadrilateral_DFEJ (A B C D E F J : ℝ × ℝ) : 
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  A = (0, 0) →
  B = (0, 10) →
  C = (10, 0) →
  d A B = 10 →
  d A C = 10 →
  d B C = 10 * Real.sqrt 2 →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  J = ((D.1 + E.1) / 2, (D.2 + E.2) / 2) →
  abs ((D.1 * F.2 + F.1 * E.2 + E.1 * J.2 + J.1 * D.2) -
       (D.2 * F.1 + F.2 * E.1 + E.2 * J.1 + J.2 * D.1)) / 2 = 6.25 :=
by sorry

end area_of_quadrilateral_DFEJ_l689_68970


namespace monkey_to_snake_ratio_l689_68958

/-- Represents the number of animals in John's zoo --/
structure ZooAnimals where
  snakes : ℕ
  monkeys : ℕ
  lions : ℕ
  pandas : ℕ
  dogs : ℕ

/-- Conditions for John's zoo --/
def zoo_conditions (z : ZooAnimals) : Prop :=
  z.snakes = 15 ∧
  z.lions = z.monkeys - 5 ∧
  z.pandas = z.lions + 8 ∧
  z.dogs * 3 = z.pandas ∧
  z.snakes + z.monkeys + z.lions + z.pandas + z.dogs = 114

/-- Theorem stating the ratio of monkeys to snakes is 2:1 --/
theorem monkey_to_snake_ratio (z : ZooAnimals) (h : zoo_conditions z) :
  z.monkeys = 2 * z.snakes := by
  sorry

end monkey_to_snake_ratio_l689_68958


namespace solve_equation_l689_68942

theorem solve_equation : ∃ x : ℝ, 7 * (x - 1) = 21 ∧ x = 4 := by
  sorry

end solve_equation_l689_68942


namespace arithmetic_mean_min_value_l689_68906

theorem arithmetic_mean_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : Real.sqrt (a * b) = 1) :
  (a + b) / 2 ≥ 1 := by
  sorry

end arithmetic_mean_min_value_l689_68906


namespace arithmetic_sequence_ninth_term_l689_68918

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence satisfying
    certain conditions, the 9th term equals 0. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_third : a 3 = 6)
  (h_sum : a 1 + a 11 = 6) :
  a 9 = 0 := by
  sorry

end arithmetic_sequence_ninth_term_l689_68918


namespace square_area_increase_l689_68972

/-- The increase in area of a square when its side length is increased by 2 -/
theorem square_area_increase (a : ℝ) : 
  (a + 2)^2 - a^2 = 4*a + 4 := by sorry

end square_area_increase_l689_68972


namespace minimize_expression_l689_68900

theorem minimize_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 50 / n ≥ 8.1667 ∧
  ((n : ℝ) / 3 + 50 / n = 8.1667 ↔ n = 12) :=
sorry

end minimize_expression_l689_68900


namespace area_bisecting_line_sum_l689_68986

/-- Triangle ABC with vertices A(0, 10), B(3, 0), C(9, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (y_intercept : ℝ)

/-- Predicate to check if a line bisects the area of a triangle through a specific vertex -/
def bisects_area (t : Triangle) (l : Line) (vertex : ℝ × ℝ) : Prop :=
  sorry

/-- The triangle ABC with given vertices -/
def triangle_ABC : Triangle :=
  { A := (0, 10),
    B := (3, 0),
    C := (9, 0) }

theorem area_bisecting_line_sum :
  ∃ l : Line, bisects_area triangle_ABC l triangle_ABC.B ∧ l.slope + l.y_intercept = -20/3 := by
  sorry

end area_bisecting_line_sum_l689_68986


namespace negative_one_in_M_l689_68938

def M : Set ℝ := {x | x^2 - 1 = 0}

theorem negative_one_in_M : (-1 : ℝ) ∈ M := by sorry

end negative_one_in_M_l689_68938


namespace min_seats_occupied_min_occupied_seats_is_fifty_l689_68946

/-- Represents the number of seats in a row -/
def total_seats : Nat := 200

/-- Represents the size of each group (one person + three empty seats) -/
def group_size : Nat := 4

/-- The minimum number of occupied seats required -/
def min_occupied_seats : Nat := total_seats / group_size

/-- Theorem stating the minimum number of occupied seats -/
theorem min_seats_occupied (n : Nat) : 
  n ≥ min_occupied_seats → 
  ∀ (new_seat : Nat), new_seat > n ∧ new_seat ≤ total_seats → 
  ∃ (occupied_seat : Nat), occupied_seat ≤ n ∧ (new_seat = occupied_seat + 1 ∨ new_seat = occupied_seat - 1) :=
sorry

/-- Theorem proving the minimum number of occupied seats is indeed 50 -/
theorem min_occupied_seats_is_fifty : min_occupied_seats = 50 :=
sorry

end min_seats_occupied_min_occupied_seats_is_fifty_l689_68946


namespace winning_candidate_percentage_l689_68925

def candidate1_votes : ℕ := 6136
def candidate2_votes : ℕ := 7636
def candidate3_votes : ℕ := 11628

def total_votes : ℕ := candidate1_votes + candidate2_votes + candidate3_votes

def winning_votes : ℕ := max candidate1_votes (max candidate2_votes candidate3_votes)

def winning_percentage : ℚ := (winning_votes : ℚ) / (total_votes : ℚ) * 100

theorem winning_candidate_percentage :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |winning_percentage - 45.78| < ε :=
sorry

end winning_candidate_percentage_l689_68925


namespace part_one_part_two_l689_68927

-- Define the sets A, B, and U
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 3}
def U : Set ℝ := {x | x ≤ 4}

-- Part 1
theorem part_one (m : ℝ) (h : m = -1) :
  (Uᶜ ∩ A)ᶜ ∪ B m = {x | x < 2 ∨ x = 4} ∧
  A ∩ (Uᶜ ∩ B m)ᶜ = {x | 2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two :
  {m : ℝ | A ∪ B m = A} = {m | -1/2 ≤ m ∧ m ≤ 1} ∪ {m | 4 ≤ m} := by sorry

end part_one_part_two_l689_68927


namespace min_route_length_5x5_city_l689_68997

/-- Represents a square grid city -/
structure SquareGridCity where
  size : Nat

/-- Represents a route in the city -/
structure CityRoute where
  length : Nat
  covers_all_streets : Bool
  returns_to_start : Bool

/-- The minimum length of a route that covers all streets and returns to the starting point -/
def min_route_length (city : SquareGridCity) : Nat :=
  sorry

theorem min_route_length_5x5_city :
  ∀ (city : SquareGridCity) (route : CityRoute),
    city.size = 5 →
    route.covers_all_streets = true →
    route.returns_to_start = true →
    route.length ≥ min_route_length city →
    min_route_length city = 68 :=
by sorry

end min_route_length_5x5_city_l689_68997


namespace triangle_satisfies_conditions_l689_68916

/-- Triangle ABC with given properties --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  euler_line : ℝ → ℝ → Prop

/-- The specific triangle we're considering --/
def our_triangle : Triangle where
  A := (-4, 0)
  B := (0, 4)
  C := (0, -2)
  euler_line := fun x y => x - y + 2 = 0

/-- Theorem stating that the given triangle satisfies the conditions --/
theorem triangle_satisfies_conditions (t : Triangle) : 
  t.A = (-4, 0) ∧ 
  t.B = (0, 4) ∧ 
  t.C = (0, -2) ∧ 
  (∀ x y, t.euler_line x y ↔ x - y + 2 = 0) →
  t = our_triangle :=
sorry

end triangle_satisfies_conditions_l689_68916


namespace problem_G2_1_l689_68912

theorem problem_G2_1 (a : ℚ) :
  137 / a = 0.1234234234235 → a = 1110 := by sorry

end problem_G2_1_l689_68912


namespace grouping_ways_correct_l689_68930

/-- The number of ways to place 4 men and 5 women into three groups. -/
def groupingWays : ℕ := 360

/-- The total number of men. -/
def numMen : ℕ := 4

/-- The total number of women. -/
def numWomen : ℕ := 5

/-- The number of groups. -/
def numGroups : ℕ := 3

/-- The size of each group. -/
def groupSize : ℕ := 3

/-- Predicate to check if a group composition is valid. -/
def validGroup (men women : ℕ) : Prop :=
  men > 0 ∧ women > 0 ∧ men + women = groupSize

/-- The theorem stating the number of ways to group people. -/
theorem grouping_ways_correct :
  ∃ (g1_men g1_women g2_men g2_women g3_men g3_women : ℕ),
    validGroup g1_men g1_women ∧
    validGroup g2_men g2_women ∧
    validGroup g3_men g3_women ∧
    g1_men + g2_men + g3_men = numMen ∧
    g1_women + g2_women + g3_women = numWomen ∧
    groupingWays = 360 :=
  sorry

end grouping_ways_correct_l689_68930


namespace arrange_digits_eq_sixteen_l689_68919

/-- The number of ways to arrange the digits of 45,550 to form a 5-digit number, where numbers cannot begin with 0 -/
def arrange_digits : ℕ :=
  let digits : Multiset ℕ := {0, 4, 5, 5, 5}
  let non_zero_positions := 4  -- Number of valid positions for 0 (2nd to 5th)
  let remaining_digits := 4    -- Number of digits to arrange after placing 0
  let repeated_digit := 3      -- Number of 5's
  non_zero_positions * (remaining_digits.factorial / repeated_digit.factorial)

theorem arrange_digits_eq_sixteen : arrange_digits = 16 := by
  sorry

end arrange_digits_eq_sixteen_l689_68919


namespace hair_cut_length_l689_68983

def initial_length : ℕ := 14
def current_length : ℕ := 1

theorem hair_cut_length : initial_length - current_length = 13 := by
  sorry

end hair_cut_length_l689_68983


namespace truncated_cube_edges_l689_68951

/-- Represents a cube with truncated corners -/
structure TruncatedCube where
  initialEdges : Nat
  vertices : Nat
  newEdgesPerVertex : Nat

/-- Calculates the total number of edges in a truncated cube -/
def totalEdges (c : TruncatedCube) : Nat :=
  c.initialEdges + c.vertices * c.newEdgesPerVertex

/-- Theorem stating that a cube with truncated corners has 36 edges -/
theorem truncated_cube_edges :
  ∀ (c : TruncatedCube),
  c.initialEdges = 12 ∧ c.vertices = 8 ∧ c.newEdgesPerVertex = 3 →
  totalEdges c = 36 := by
  sorry

end truncated_cube_edges_l689_68951


namespace part_one_part_two_part_three_l689_68937

-- Part 1
theorem part_one : 12 - (-11) - 1 = 22 := by sorry

-- Part 2
theorem part_two : -1^4 / (-3)^2 / (9/5) = -5/81 := by sorry

-- Part 3
theorem part_three : -8 * (1/2 - 3/4 + 5/8) = -3 := by sorry

end part_one_part_two_part_three_l689_68937


namespace f_positive_iff_l689_68957

/-- The function f(x) = 2x + 5 -/
def f (x : ℝ) : ℝ := 2 * x + 5

/-- Theorem: f(x) > 0 if and only if x > -5/2 -/
theorem f_positive_iff (x : ℝ) : f x > 0 ↔ x > -5/2 := by
  sorry

end f_positive_iff_l689_68957


namespace train_meeting_correct_l689_68939

/-- Represents the properties of two trains meeting between two cities -/
structure TrainMeeting where
  normal_meet_time : ℝ  -- in hours
  early_a_distance : ℝ  -- in km
  early_b_distance : ℝ  -- in km
  early_time : ℝ        -- in hours

/-- The solution to the train meeting problem -/
def train_meeting_solution (tm : TrainMeeting) : ℝ × ℝ × ℝ :=
  let distance := 660
  let speed_a := 115
  let speed_b := 85
  (distance, speed_a, speed_b)

/-- Theorem stating that the given solution satisfies the train meeting conditions -/
theorem train_meeting_correct (tm : TrainMeeting) 
  (h1 : tm.normal_meet_time = 3 + 18/60)
  (h2 : tm.early_a_distance = 14)
  (h3 : tm.early_b_distance = 9)
  (h4 : tm.early_time = 3) :
  let (distance, speed_a, speed_b) := train_meeting_solution tm
  (speed_a + speed_b) * tm.normal_meet_time = distance ∧
  speed_a * (tm.normal_meet_time + 24/60) = distance - tm.early_a_distance + speed_b * tm.early_time ∧
  speed_b * (tm.normal_meet_time + 36/60) = distance - tm.early_b_distance + speed_a * tm.early_time :=
by
  sorry

end train_meeting_correct_l689_68939


namespace floor_sqrt_20_squared_l689_68911

theorem floor_sqrt_20_squared : ⌊Real.sqrt 20⌋^2 = 16 := by
  sorry

end floor_sqrt_20_squared_l689_68911


namespace sequence_inequality_l689_68910

theorem sequence_inequality (n : ℕ) : n / (n + 2) < (n + 1) / (n + 3) := by
  sorry

end sequence_inequality_l689_68910


namespace new_car_travel_distance_l689_68941

/-- The distance traveled by the older car in miles -/
def older_car_distance : ℝ := 150

/-- The percentage increase in distance for the new car -/
def percentage_increase : ℝ := 0.30

/-- The distance traveled by the new car in miles -/
def new_car_distance : ℝ := older_car_distance * (1 + percentage_increase)

theorem new_car_travel_distance : new_car_distance = 195 := by
  sorry

end new_car_travel_distance_l689_68941


namespace cube_sum_geq_triple_sum_products_l689_68947

theorem cube_sum_geq_triple_sum_products
  (a b c : ℝ)
  (ha : a ≥ 0)
  (hb : b ≥ 0)
  (hc : c ≥ 0)
  (h_sum_squares : a^2 + b^2 + c^2 ≥ 3) :
  (a + b + c)^3 ≥ 9 * (a*b + b*c + c*a) := by
  sorry

end cube_sum_geq_triple_sum_products_l689_68947


namespace antecedent_value_l689_68976

/-- Given a ratio of 4:6 and a consequent of 30, prove the antecedent is 20 -/
theorem antecedent_value (ratio_antecedent ratio_consequent consequent : ℕ) 
  (h1 : ratio_antecedent = 4)
  (h2 : ratio_consequent = 6)
  (h3 : consequent = 30) :
  ratio_antecedent * consequent / ratio_consequent = 20 := by
  sorry

end antecedent_value_l689_68976


namespace tank_fill_time_l689_68934

/-- The time to fill a tank with a pump and a leak -/
theorem tank_fill_time (pump_rate leak_rate : ℝ) (pump_rate_pos : pump_rate > 0) 
  (leak_rate_pos : leak_rate > 0) (pump_faster : pump_rate > leak_rate) :
  let fill_time := 1 / (pump_rate - leak_rate)
  fill_time = 1 / (1 / 2 - 1 / 26) :=
by
  sorry

#eval 1 / (1 / 2 - 1 / 26)

end tank_fill_time_l689_68934


namespace parallelogram_most_analogous_to_parallelepiped_l689_68902

-- Define the characteristics of a parallelepiped
structure Parallelepiped :=
  (has_parallel_faces : Bool)

-- Define planar figures
inductive PlanarFigure
| Triangle
| Trapezoid
| Parallelogram
| Rectangle

-- Define the analogy relation
def is_analogous (p : Parallelepiped) (f : PlanarFigure) : Prop :=
  match f with
  | PlanarFigure.Parallelogram => p.has_parallel_faces
  | _ => False

-- Theorem statement
theorem parallelogram_most_analogous_to_parallelepiped :
  ∀ (p : Parallelepiped) (f : PlanarFigure),
    p.has_parallel_faces →
    is_analogous p f →
    f = PlanarFigure.Parallelogram :=
sorry

end parallelogram_most_analogous_to_parallelepiped_l689_68902


namespace union_of_sets_l689_68943

theorem union_of_sets : 
  let A : Set ℕ := {1, 3}
  let B : Set ℕ := {1, 2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by
sorry

end union_of_sets_l689_68943


namespace volumetric_contraction_of_mixed_liquids_l689_68964

/-- Proves that the volumetric contraction when mixing two liquids with given properties is 21 cm³ -/
theorem volumetric_contraction_of_mixed_liquids :
  let density1 : ℝ := 1.7
  let mass1 : ℝ := 400
  let density2 : ℝ := 1.2
  let mass2 : ℝ := 600
  let total_mass : ℝ := mass1 + mass2
  let mixed_density : ℝ := 1.4
  let volume1 : ℝ := mass1 / density1
  let volume2 : ℝ := mass2 / density2
  let total_volume : ℝ := volume1 + volume2
  let actual_volume : ℝ := total_mass / mixed_density
  let contraction : ℝ := total_volume - actual_volume
  contraction = 21 := by sorry

end volumetric_contraction_of_mixed_liquids_l689_68964


namespace wire_ratio_l689_68913

theorem wire_ratio (a b : ℝ) (h : a > 0) (k : b > 0) : 
  (a^2 / 16 = b^2 / (4 * Real.pi)) → a / b = 2 / Real.sqrt Real.pi := by
sorry

end wire_ratio_l689_68913


namespace smallest_n_for_3003_combinations_l689_68973

theorem smallest_n_for_3003_combinations : ∃ (N : ℕ), N > 0 ∧ (
  (∀ k < N, Nat.choose k 5 < 3003) ∧
  Nat.choose N 5 = 3003
) := by sorry

end smallest_n_for_3003_combinations_l689_68973


namespace quadratic_odd_coefficients_irrational_roots_l689_68923

theorem quadratic_odd_coefficients_irrational_roots (a b c : ℤ) :
  (Odd a ∧ Odd b ∧ Odd c) →
  ∀ x : ℚ, a * x^2 + b * x + c ≠ 0 := by
sorry

end quadratic_odd_coefficients_irrational_roots_l689_68923


namespace union_of_P_and_Q_l689_68967

-- Define the sets P and Q
def P : Set ℝ := {x | -2 < x ∧ x ≤ 3}
def Q : Set ℝ := {x | (1 + x) / (x - 3) ≤ 0}

-- State the theorem
theorem union_of_P_and_Q : P ∪ Q = {x : ℝ | -2 < x ∧ x ≤ 3} := by sorry

end union_of_P_and_Q_l689_68967


namespace repeating_decimal_to_fraction_l689_68907

theorem repeating_decimal_to_fraction :
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ (n / d : ℚ) = ∑' k, 6 * (1 / 10 : ℚ)^(k + 1) ∧ n / d = 2 / 3 := by
  sorry

end repeating_decimal_to_fraction_l689_68907


namespace base_8_units_digit_l689_68998

theorem base_8_units_digit : (((324 + 73) * 27) % 8 = 7) := by
  sorry

end base_8_units_digit_l689_68998


namespace max_min_f_l689_68949

def f (x y : ℝ) : ℝ := 3 * |x + y| + |4 * y + 9| + |7 * y - 3 * x - 18|

theorem max_min_f :
  ∀ x y : ℝ, x^2 + y^2 ≤ 5 →
  (∀ x' y' : ℝ, x'^2 + y'^2 ≤ 5 → f x' y' ≤ f x y) → f x y = 27 + 6 * Real.sqrt 5 ∧
  (∀ x' y' : ℝ, x'^2 + y'^2 ≤ 5 → f x y ≤ f x' y') → f x y = 27 - 3 * Real.sqrt 10 :=
by sorry

end max_min_f_l689_68949


namespace curve_equation_l689_68922

/-- Given vectors and their relationships, prove the equation of the resulting curve. -/
theorem curve_equation (x y : ℝ) : 
  let m₁ : ℝ × ℝ := (0, x)
  let n₁ : ℝ × ℝ := (1, 1)
  let m₂ : ℝ × ℝ := (x, 0)
  let n₂ : ℝ × ℝ := (y^2, 1)
  let m : ℝ × ℝ := m₁ + Real.sqrt 2 • n₂
  let n : ℝ × ℝ := m₂ - Real.sqrt 2 • n₁
  (m.1 * n.2 = m.2 * n.1) →  -- m is parallel to n
  x^2 / 2 + y^2 = 1 :=
by sorry

end curve_equation_l689_68922


namespace number_of_observations_l689_68988

/-- Given a set of observations with an initial mean, a correction to one observation,
    and a new mean, prove the number of observations. -/
theorem number_of_observations
  (initial_mean : ℝ)
  (wrong_value : ℝ)
  (correct_value : ℝ)
  (new_mean : ℝ)
  (h1 : initial_mean = 36)
  (h2 : wrong_value = 23)
  (h3 : correct_value = 45)
  (h4 : new_mean = 36.5) :
  ∃ n : ℕ, n * initial_mean + (correct_value - wrong_value) = n * new_mean ∧ n = 44 := by
  sorry

end number_of_observations_l689_68988


namespace binomial_coefficient_two_l689_68932

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l689_68932


namespace power_sum_equation_l689_68915

theorem power_sum_equation : 
  let x : ℚ := 1/2
  2^(0 : ℤ) + x^(-2 : ℤ) = 5 := by sorry

end power_sum_equation_l689_68915


namespace power_mod_eleven_l689_68960

theorem power_mod_eleven : 5^2023 % 11 = 3 := by
  sorry

end power_mod_eleven_l689_68960


namespace ludwig_earnings_proof_l689_68974

/-- Ludwig's weekly work schedule and earnings --/
def ludwig_weekly_earnings : ℕ :=
  let full_day_salary : ℕ := 10
  let full_days : ℕ := 4
  let half_days : ℕ := 3
  let full_day_earnings : ℕ := full_day_salary * full_days
  let half_day_earnings : ℕ := (full_day_salary / 2) * half_days
  full_day_earnings + half_day_earnings

/-- Theorem: Ludwig's weekly earnings are $55 --/
theorem ludwig_earnings_proof : ludwig_weekly_earnings = 55 := by
  sorry

end ludwig_earnings_proof_l689_68974


namespace s_equals_one_l689_68903

theorem s_equals_one (k R : ℝ) (h : |k + R| / |R| = 0) : |k + 2*R| / |2*k + R| = 1 := by
  sorry

end s_equals_one_l689_68903


namespace brendan_grass_cutting_l689_68978

/-- Brendan's grass cutting capacity over a week -/
theorem brendan_grass_cutting (initial_capacity : ℝ) (increase_percentage : ℝ) (days_in_week : ℕ) :
  initial_capacity = 8 →
  increase_percentage = 0.5 →
  days_in_week = 7 →
  (initial_capacity + initial_capacity * increase_percentage) * days_in_week = 84 := by
  sorry

end brendan_grass_cutting_l689_68978


namespace sum_of_all_alternating_sums_l689_68959

-- Define the set of numbers
def S : Finset ℕ := Finset.range 9

-- Define the alternating sum function
noncomputable def alternatingSum (subset : Finset ℕ) : ℤ :=
  sorry

-- Define the modified alternating sum that adds 9 again if present
noncomputable def modifiedAlternatingSum (subset : Finset ℕ) : ℤ :=
  if 9 ∈ subset then alternatingSum subset + 9 else alternatingSum subset

-- Theorem statement
theorem sum_of_all_alternating_sums : 
  (Finset.powerset S).sum modifiedAlternatingSum = 2304 :=
sorry

end sum_of_all_alternating_sums_l689_68959


namespace largest_n_binomial_equality_l689_68980

theorem largest_n_binomial_equality : 
  (∃ n : ℕ, (Nat.choose 8 3 + Nat.choose 8 4 = Nat.choose 9 n)) ∧ 
  (∀ m : ℕ, Nat.choose 8 3 + Nat.choose 8 4 = Nat.choose 9 m → m ≤ 5) := by
  sorry

end largest_n_binomial_equality_l689_68980


namespace train_distance_l689_68921

/-- The distance between two towns given train speeds and meeting time -/
theorem train_distance (faster_speed slower_speed meeting_time : ℝ) : 
  faster_speed = 48 ∧ 
  faster_speed = slower_speed + 6 ∧ 
  meeting_time = 5 →
  (faster_speed + slower_speed) * meeting_time = 450 := by
sorry

end train_distance_l689_68921


namespace investment_time_calculation_l689_68963

/-- Represents the investment scenario of two partners A and B --/
structure Investment where
  a_capital : ℝ
  a_time : ℝ
  b_capital : ℝ
  b_time : ℝ
  profit_ratio : ℝ

/-- Theorem stating the time B's investment was effective --/
theorem investment_time_calculation (i : Investment) 
  (h1 : i.a_capital = 27000)
  (h2 : i.b_capital = 36000)
  (h3 : i.a_time = 12)
  (h4 : i.profit_ratio = 2/1) :
  i.b_time = 4.5 := by
  sorry

#check investment_time_calculation

end investment_time_calculation_l689_68963


namespace intersection_of_M_and_N_l689_68955

def M : Set ℝ := {x : ℝ | -5 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end intersection_of_M_and_N_l689_68955


namespace steve_blank_questions_l689_68954

def total_questions : ℕ := 60
def word_problems : ℕ := 20
def add_sub_problems : ℕ := 25
def algebra_problems : ℕ := 10
def geometry_problems : ℕ := 5

def steve_word : ℕ := 15
def steve_add_sub : ℕ := 22
def steve_algebra : ℕ := 8
def steve_geometry : ℕ := 3

theorem steve_blank_questions :
  total_questions - (steve_word + steve_add_sub + steve_algebra + steve_geometry) = 12 :=
by sorry

end steve_blank_questions_l689_68954


namespace subtraction_from_percentage_l689_68994

theorem subtraction_from_percentage (x : ℝ) : x = 100 → (0.7 * x - 40 = 30) := by
  sorry

end subtraction_from_percentage_l689_68994


namespace total_carrots_grown_l689_68929

theorem total_carrots_grown (sandy_carrots sam_carrots : ℕ) 
  (h1 : sandy_carrots = 6) 
  (h2 : sam_carrots = 3) : 
  sandy_carrots + sam_carrots = 9 := by
  sorry

end total_carrots_grown_l689_68929


namespace base3_sum_equality_l689_68909

/-- Converts a list of digits in base 3 to its decimal representation -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 3 * acc) 0

/-- Converts a decimal number to its base 3 representation -/
def toBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 3) ((m % 3) :: acc)
    go n []

/-- The sum of the given numbers in base 3 is equal to 112010 in base 3 -/
theorem base3_sum_equality : 
  let a := [2]
  let b := [1, 1]
  let c := [2, 0, 2]
  let d := [1, 0, 0, 2]
  let e := [2, 2, 1, 1, 1]
  let sum := [0, 1, 0, 2, 1, 1]
  toBase3 (toDecimal a + toDecimal b + toDecimal c + toDecimal d + toDecimal e) = sum := by
  sorry

end base3_sum_equality_l689_68909


namespace probability_at_least_one_inferior_l689_68975

def total_pencils : ℕ := 10
def good_pencils : ℕ := 8
def inferior_pencils : ℕ := 2
def drawn_pencils : ℕ := 2

theorem probability_at_least_one_inferior :
  let total_ways := Nat.choose total_pencils drawn_pencils
  let ways_no_inferior := Nat.choose good_pencils drawn_pencils
  (total_ways - ways_no_inferior : ℚ) / total_ways = 17 / 45 := by
  sorry

end probability_at_least_one_inferior_l689_68975


namespace sum_of_special_numbers_l689_68981

/-- The set of digits to be used -/
def digits : Finset Nat := {1, 3, 4, 7, 9}

/-- A function to check if a number is a multiple of 11 -/
def is_multiple_of_11 (n : Nat) : Bool :=
  n % 11 = 0

/-- A function to generate all 5-digit numbers from the given digits -/
def generate_numbers (digits : Finset Nat) : Finset Nat :=
  sorry

/-- A function to sum all numbers in a set that are not multiples of 11 -/
def sum_non_multiples_of_11 (numbers : Finset Nat) : Nat :=
  sorry

/-- The main theorem -/
theorem sum_of_special_numbers :
  sum_non_multiples_of_11 (generate_numbers digits) = 5842368 :=
sorry

end sum_of_special_numbers_l689_68981


namespace satisfactory_grade_fraction_l689_68952

/-- Represents the grades in a science class -/
inductive Grade
  | A
  | B
  | C
  | D
  | F

/-- Returns true if the grade is satisfactory (A, B, or C) -/
def isSatisfactory (g : Grade) : Bool :=
  match g with
  | Grade.A => true
  | Grade.B => true
  | Grade.C => true
  | _ => false

/-- Represents the distribution of grades in the class -/
def gradeDistribution : List (Grade × Nat) :=
  [(Grade.A, 8), (Grade.B, 6), (Grade.C, 4), (Grade.D, 2), (Grade.F, 6)]

/-- Theorem: The fraction of satisfactory grades is 9/13 -/
theorem satisfactory_grade_fraction :
  let totalGrades := (gradeDistribution.map (·.2)).sum
  let satisfactoryGrades := (gradeDistribution.filter (isSatisfactory ·.1)).map (·.2) |>.sum
  (satisfactoryGrades : ℚ) / totalGrades = 9 / 13 := by
  sorry


end satisfactory_grade_fraction_l689_68952


namespace committee_formation_count_l689_68968

theorem committee_formation_count : Nat.choose 15 6 = 5005 := by
  sorry

end committee_formation_count_l689_68968


namespace pages_left_to_read_l689_68992

theorem pages_left_to_read 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (pages_to_skip : ℕ) 
  (h1 : total_pages = 372) 
  (h2 : pages_read = 125) 
  (h3 : pages_to_skip = 16) :
  total_pages - (pages_read + pages_to_skip) = 231 :=
by sorry

end pages_left_to_read_l689_68992


namespace P_less_than_Q_l689_68999

theorem P_less_than_Q (a : ℝ) (ha : a ≥ 0) :
  Real.sqrt (a + 3) + Real.sqrt (a + 5) < Real.sqrt (a + 1) + Real.sqrt (a + 7) := by
  sorry

end P_less_than_Q_l689_68999


namespace negative_two_squared_times_negative_two_squared_l689_68969

theorem negative_two_squared_times_negative_two_squared : -2^2 * (-2)^2 = -16 := by
  sorry

end negative_two_squared_times_negative_two_squared_l689_68969


namespace canteen_distance_l689_68953

/-- Given a right triangle with legs 450 and 600 rods, prove that a point on the hypotenuse 
    that is equidistant from both ends of the hypotenuse is 468.75 rods from each end. -/
theorem canteen_distance (a b c x : ℝ) (h1 : a = 450) (h2 : b = 600) 
  (h3 : c^2 = a^2 + b^2) (h4 : x^2 = a^2 + (b - x)^2) : x = 468.75 := by
  sorry

end canteen_distance_l689_68953


namespace triangle_altitude_l689_68940

/-- Given a triangle with area 600 square feet and base length 30 feet,
    prove that its altitude is 40 feet. -/
theorem triangle_altitude (A : ℝ) (b : ℝ) (h : ℝ) 
    (area_eq : A = 600)
    (base_eq : b = 30)
    (area_formula : A = (1/2) * b * h) : h = 40 := by
  sorry

end triangle_altitude_l689_68940


namespace g_in_M_l689_68936

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∀ x₁ x₂ : ℝ, |x₁| ≤ 1 → |x₂| ≤ 1 → |f x₁ - f x₂| ≤ 4 * |x₁ - x₂|}

-- Define the function g
def g (x : ℝ) : ℝ := x^2 + 2*x - 1

-- Theorem statement
theorem g_in_M : g ∈ M := by
  sorry

end g_in_M_l689_68936


namespace gcd_of_three_numbers_l689_68984

theorem gcd_of_three_numbers : Nat.gcd 9242 (Nat.gcd 13863 34657) = 1 := by
  sorry

end gcd_of_three_numbers_l689_68984


namespace sqrt_175_range_l689_68924

theorem sqrt_175_range : 13 < Real.sqrt 175 ∧ Real.sqrt 175 < 14 := by
  sorry

end sqrt_175_range_l689_68924


namespace inequality_proof_l689_68933

theorem inequality_proof (a b c e f : ℝ) 
  (h1 : a > b) (h2 : e > f) (h3 : c > 0) : f - a*c < e - b*c := by
  sorry

end inequality_proof_l689_68933


namespace inequality_proof_l689_68979

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^3 + b^3 + c^3 = 3) : 
  1/(a^4 + 3) + 1/(b^4 + 3) + 1/(c^4 + 3) ≥ 3/4 := by
sorry

end inequality_proof_l689_68979


namespace two_roots_theorem_l689_68914

theorem two_roots_theorem (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    (x₁ - a) * (x₁ - b) + (x₁ - a) * (x₁ - c) + (x₁ - b) * (x₁ - c) = 0 ∧
    (x₂ - a) * (x₂ - b) + (x₂ - a) * (x₂ - c) + (x₂ - b) * (x₂ - c) = 0 ∧
    a < x₁ ∧ x₁ < b ∧ b < x₂ ∧ x₂ < c :=
by sorry

end two_roots_theorem_l689_68914


namespace modified_short_bingo_first_column_possibilities_l689_68935

theorem modified_short_bingo_first_column_possibilities : 
  (Finset.univ.filter (λ x : Finset (Fin 12) => x.card = 5)).card = 95040 := by
  sorry

end modified_short_bingo_first_column_possibilities_l689_68935


namespace percentage_increase_problem_l689_68962

theorem percentage_increase_problem : 
  let initial := 100
  let after_first_increase := initial * (1 + 0.2)
  let final := after_first_increase * (1 + 0.5)
  final = 180 := by sorry

end percentage_increase_problem_l689_68962


namespace max_teams_tied_for_most_wins_l689_68971

/-- Represents a round-robin tournament --/
structure Tournament where
  n : ℕ  -- number of teams
  games : Fin n → Fin n → Bool
  -- games i j is true if team i wins against team j
  irreflexive : ∀ i, games i i = false
  asymmetric : ∀ i j, games i j = !games j i

/-- The number of wins for a team in a tournament --/
def wins (t : Tournament) (i : Fin t.n) : ℕ :=
  (Finset.univ.filter (λ j => t.games i j)).card

/-- The maximum number of wins in a tournament --/
def max_wins (t : Tournament) : ℕ :=
  Finset.univ.sup (wins t)

/-- The number of teams tied for the maximum number of wins --/
def num_teams_with_max_wins (t : Tournament) : ℕ :=
  (Finset.univ.filter (λ i => wins t i = max_wins t)).card

theorem max_teams_tied_for_most_wins :
  ∃ t : Tournament, t.n = 8 ∧ num_teams_with_max_wins t = 7 ∧
  ∀ t' : Tournament, t'.n = 8 → num_teams_with_max_wins t' ≤ 7 := by
  sorry

end max_teams_tied_for_most_wins_l689_68971


namespace inequality_selection_l689_68965

/-- Given positive real numbers a, b, c, and a function f with minimum value 4,
    prove that a + b + c = 4 and find the minimum value of (1/4)a² + (1/9)b² + c² -/
theorem inequality_selection (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : ∀ x, |x + a| + |x - b| + c ≥ 4)
  (h5 : ∃ x, |x + a| + |x - b| + c = 4) :
  (a + b + c = 4) ∧
  (∀ a' b' c', a' > 0 → b' > 0 → c' > 0 → a' + b' + c' = 4 →
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 ≥ 8/7) ∧
  (∃ a' b' c', a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 4 ∧
    (1/4) * a'^2 + (1/9) * b'^2 + c'^2 = 8/7) :=
by sorry

end inequality_selection_l689_68965


namespace hyperbola_asymptotes_l689_68908

/-- Given a hyperbola with equation x²/16 - y²/25 = 1, 
    its asymptotes have the equation y = ±(5/4)x -/
theorem hyperbola_asymptotes (x y : ℝ) :
  x^2 / 16 - y^2 / 25 = 1 →
  ∃ (k : ℝ), k = 5/4 ∧ (y = k*x ∨ y = -k*x) :=
by sorry

end hyperbola_asymptotes_l689_68908


namespace ellipse_equation_l689_68944

/-- An ellipse with one focus at (1, 0) and eccentricity √2/2 has the equation x^2/2 + y^2 = 1 -/
theorem ellipse_equation (e : ℝ × ℝ → Prop) :
  (∃ (a b c : ℝ), 
    -- One focus is at (1, 0)
    c = 1 ∧
    -- Eccentricity is √2/2
    c / a = Real.sqrt 2 / 2 ∧
    -- Standard form of ellipse equation
    b^2 = a^2 - c^2 ∧
    (∀ (x y : ℝ), e (x, y) ↔ x^2 / a^2 + y^2 / b^2 = 1)) →
  (∀ (x y : ℝ), e (x, y) ↔ x^2 / 2 + y^2 = 1) :=
by sorry

end ellipse_equation_l689_68944
