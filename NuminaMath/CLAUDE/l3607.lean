import Mathlib

namespace jaime_savings_time_l3607_360705

/-- Calculates the number of weeks needed to save a target amount given weekly savings and bi-weekly expenses -/
def weeksToSave (weeklySavings : ℚ) (biWeeklyExpense : ℚ) (targetAmount : ℚ) : ℚ :=
  let netBiWeeklySavings := 2 * weeklySavings - biWeeklyExpense
  let netWeeklySavings := netBiWeeklySavings / 2
  targetAmount / netWeeklySavings

/-- Proves that it takes 5 weeks to save $135 with $50 weekly savings and $46 bi-weekly expense -/
theorem jaime_savings_time : weeksToSave 50 46 135 = 5 := by
  sorry

end jaime_savings_time_l3607_360705


namespace seventh_root_unity_product_l3607_360781

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end seventh_root_unity_product_l3607_360781


namespace mailman_delivery_l3607_360797

theorem mailman_delivery (total_mail junk_mail : ℕ) 
  (h1 : total_mail = 11) 
  (h2 : junk_mail = 6) : 
  total_mail - junk_mail = 5 := by
  sorry

end mailman_delivery_l3607_360797


namespace garden_trees_l3607_360700

/-- The number of trees in a garden with specific planting conditions. -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  yard_length / tree_distance + 1

/-- Theorem stating that the number of trees in the garden is 26. -/
theorem garden_trees :
  number_of_trees 800 32 = 26 := by
  sorry

end garden_trees_l3607_360700


namespace engine_capacity_l3607_360762

/-- The engine capacity (in cc) for which 85 litres of diesel is required to travel 600 km -/
def C : ℝ := 595

/-- The volume of diesel (in litres) required for the reference engine -/
def V₁ : ℝ := 170

/-- The capacity (in cc) of the reference engine -/
def C₁ : ℝ := 1200

/-- The volume of diesel (in litres) required for the engine capacity C -/
def V₂ : ℝ := 85

/-- The ratio of volume to capacity is constant -/
axiom volume_capacity_ratio : V₁ / C₁ = V₂ / C

theorem engine_capacity : C = 595 := by sorry

end engine_capacity_l3607_360762


namespace sum_of_three_consecutive_odd_squares_divisible_by_two_l3607_360723

theorem sum_of_three_consecutive_odd_squares_divisible_by_two (n : ℤ) (h : Odd n) :
  ∃ k : ℤ, 3 * n^2 + 8 = 2 * k := by
  sorry

end sum_of_three_consecutive_odd_squares_divisible_by_two_l3607_360723


namespace fruit_purchase_cost_l3607_360771

/-- Given information about oranges and apples, calculates the total cost of a specific purchase. -/
theorem fruit_purchase_cost 
  (orange_bags : ℕ) (orange_weight : ℝ) (apple_bags : ℕ) (apple_weight : ℝ)
  (orange_price : ℝ) (apple_price : ℝ)
  (h_orange : orange_bags * (orange_weight / orange_bags) = 24)
  (h_apple : apple_bags * (apple_weight / apple_bags) = 30)
  (h_orange_price : orange_price = 1.5)
  (h_apple_price : apple_price = 2) :
  5 * (orange_weight / orange_bags) * orange_price + 
  4 * (apple_weight / apple_bags) * apple_price = 45 := by
  sorry


end fruit_purchase_cost_l3607_360771


namespace dog_play_area_l3607_360772

/-- The area outside of a square doghouse that a dog can reach with a leash -/
theorem dog_play_area (leash_length : ℝ) (doghouse_side : ℝ) : 
  leash_length = 4 →
  doghouse_side = 2 →
  (3 / 4 * π * leash_length^2 + 2 * (1 / 4 * π * doghouse_side^2)) = 14 * π :=
by sorry

end dog_play_area_l3607_360772


namespace competition_outcomes_l3607_360733

/-- The number of participants in the competition -/
def n : ℕ := 6

/-- The number of places to be filled (1st, 2nd, 3rd) -/
def k : ℕ := 3

/-- The number of different ways to arrange k distinct items from a set of n distinct items -/
def arrangement_count (n k : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem competition_outcomes :
  arrangement_count n k = 120 :=
sorry

end competition_outcomes_l3607_360733


namespace slower_speed_calculation_l3607_360727

theorem slower_speed_calculation (distance : ℝ) (time_saved : ℝ) (faster_speed : ℝ) :
  distance = 1200 →
  time_saved = 4 →
  faster_speed = 60 →
  ∃ slower_speed : ℝ,
    (distance / slower_speed) - (distance / faster_speed) = time_saved ∧
    slower_speed = 50 := by
  sorry

end slower_speed_calculation_l3607_360727


namespace reflection_sum_coordinates_l3607_360769

/-- Given a point C with coordinates (3, y), when reflected over the x-axis to point D,
    the sum of all coordinate values of C and D is equal to 6. -/
theorem reflection_sum_coordinates (y : ℝ) : 
  let C : ℝ × ℝ := (3, y)
  let D : ℝ × ℝ := (3, -y)
  (C.1 + C.2 + D.1 + D.2) = 6 := by sorry

end reflection_sum_coordinates_l3607_360769


namespace salary_change_l3607_360749

theorem salary_change (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * (1 + 0.15)
  let final_salary := increased_salary * (1 - 0.15)
  (final_salary - initial_salary) / initial_salary = -0.0225 := by
sorry

end salary_change_l3607_360749


namespace pigeonhole_divisibility_l3607_360768

theorem pigeonhole_divisibility (x : Fin 2020 → ℤ) :
  ∃ i j : Fin 2020, i ≠ j ∧ (x j - x i) % 2019 = 0 := by
  sorry

end pigeonhole_divisibility_l3607_360768


namespace power_of_two_greater_than_square_l3607_360776

theorem power_of_two_greater_than_square (n : ℕ) (h : n ≥ 1) :
  2^n > n^2 := by
  sorry

end power_of_two_greater_than_square_l3607_360776


namespace distinct_z_values_l3607_360738

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

def z (x : ℕ) : ℕ := Int.natAbs (x - reverse_digits x)

theorem distinct_z_values :
  ∃ (S : Finset ℕ), (∀ x, is_two_digit x → z x ∈ S) ∧ S.card = 10 :=
sorry

end distinct_z_values_l3607_360738


namespace choir_members_count_l3607_360790

theorem choir_members_count : ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 12 = 10 ∧ n % 14 = 12 := by
  sorry

end choir_members_count_l3607_360790


namespace min_area_rectangle_l3607_360778

/-- A rectangle with integer dimensions and perimeter 200 has a minimum area of 99 square units. -/
theorem min_area_rectangle (l w : ℕ) : 
  (2 * l + 2 * w = 200) →  -- perimeter condition
  (l * w ≥ 99) :=          -- minimum area
by sorry

end min_area_rectangle_l3607_360778


namespace max_gcd_consecutive_b_terms_is_one_l3607_360721

/-- The sequence b_n defined as n! + n^2 + 1 -/
def b (n : ℕ) : ℕ := n.factorial + n^2 + 1

/-- The theorem stating that the maximum GCD of consecutive terms in the sequence is 1 -/
theorem max_gcd_consecutive_b_terms_is_one :
  ∀ n : ℕ, ∃ m : ℕ, m ≥ n → (∀ k ≥ m, Nat.gcd (b k) (b (k + 1)) = 1) ∧
    (∀ i j : ℕ, i ≥ n → j = i + 1 → Nat.gcd (b i) (b j) ≤ 1) := by
  sorry

end max_gcd_consecutive_b_terms_is_one_l3607_360721


namespace pencils_per_box_l3607_360736

theorem pencils_per_box (total_pencils : ℕ) (num_boxes : ℚ) 
  (h1 : total_pencils = 2592) 
  (h2 : num_boxes = 4) : 
  (total_pencils : ℚ) / num_boxes = 648 := by
sorry

end pencils_per_box_l3607_360736


namespace binary_101101_equals_45_l3607_360794

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end binary_101101_equals_45_l3607_360794


namespace a_zero_necessary_not_sufficient_l3607_360707

-- Define a complex number
def complex (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem a_zero_necessary_not_sufficient :
  (∀ z : ℂ, is_purely_imaginary z → z.re = 0) ∧
  ¬(∀ z : ℂ, z.re = 0 → is_purely_imaginary z) :=
by sorry

end a_zero_necessary_not_sufficient_l3607_360707


namespace book_arrangement_count_l3607_360746

theorem book_arrangement_count : 
  let total_books : ℕ := 7
  let identical_math_books : ℕ := 3
  let identical_physics_books : ℕ := 2
  let distinct_books : ℕ := total_books - identical_math_books - identical_physics_books
  ↑(Nat.factorial total_books) / (↑(Nat.factorial identical_math_books) * ↑(Nat.factorial identical_physics_books)) = 420 :=
by sorry

end book_arrangement_count_l3607_360746


namespace area_of_specific_rectangle_l3607_360784

/-- Represents a rectangle with given properties -/
structure Rectangle where
  breadth : ℝ
  length : ℝ
  perimeter : ℝ
  area : ℝ

/-- Theorem: Area of a specific rectangle -/
theorem area_of_specific_rectangle :
  ∀ (rect : Rectangle),
  rect.length = 3 * rect.breadth →
  rect.perimeter = 104 →
  rect.area = rect.length * rect.breadth →
  rect.area = 507 := by
sorry

end area_of_specific_rectangle_l3607_360784


namespace parallelogram_vertex_sum_l3607_360766

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (10, 5)

-- Define the property that A and C are diagonally opposite
def diagonally_opposite (A C : ℝ × ℝ) : Prop :=
  ∃ (M : ℝ × ℝ), M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define the property of a parallelogram
def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1 = D.1 - C.1) ∧ (B.2 - A.2 = D.2 - C.2)

-- Theorem statement
theorem parallelogram_vertex_sum :
  ∀ D : ℝ × ℝ,
  is_parallelogram A B C D →
  diagonally_opposite A C →
  D.1 + D.2 = 14 :=
by sorry

end parallelogram_vertex_sum_l3607_360766


namespace circus_ticket_price_l3607_360774

theorem circus_ticket_price :
  ∀ (adult_price kid_price : ℝ),
    kid_price = (1/2) * adult_price →
    6 * kid_price + 2 * adult_price = 50 →
    kid_price = 5 :=
by
  sorry

end circus_ticket_price_l3607_360774


namespace zhang_daily_distance_l3607_360737

/-- Given a one-way distance and number of round trips, calculates the total distance driven. -/
def total_distance (one_way_distance : ℕ) (num_round_trips : ℕ) : ℕ :=
  2 * one_way_distance * num_round_trips

/-- Proves that given a one-way distance of 33 kilometers and 5 round trips per day, 
    the total distance driven is 330 kilometers. -/
theorem zhang_daily_distance : total_distance 33 5 = 330 := by
  sorry

end zhang_daily_distance_l3607_360737


namespace convex_polygon_sides_l3607_360704

theorem convex_polygon_sides (sum_except_one : ℝ) (missing_angle : ℝ) : 
  sum_except_one = 2970 ∧ missing_angle = 150 → 
  (∃ (n : ℕ), n = 20 ∧ 180 * (n - 2) = sum_except_one + missing_angle) :=
by sorry

end convex_polygon_sides_l3607_360704


namespace olympic_photo_arrangements_l3607_360713

/-- Represents the number of athletes -/
def num_athletes : ℕ := 5

/-- Represents the number of athletes that can occupy the leftmost position -/
def num_leftmost_athletes : ℕ := 2

/-- Represents whether athlete A can occupy the rightmost position -/
def a_can_be_rightmost : Bool := false

/-- The total number of different arrangement possibilities -/
def total_arrangements : ℕ := 42

/-- Theorem stating that the total number of arrangements is 42 -/
theorem olympic_photo_arrangements :
  (num_athletes = 5) →
  (num_leftmost_athletes = 2) →
  (a_can_be_rightmost = false) →
  (total_arrangements = 42) := by
  sorry

end olympic_photo_arrangements_l3607_360713


namespace hypotenuse_length_l3607_360793

/-- A rectangle with an inscribed circle -/
structure RectangleWithInscribedCircle where
  -- The length of side AB
  ab : ℝ
  -- The length of side BC
  bc : ℝ
  -- The point where the circle touches AB
  p : ℝ × ℝ
  -- The point where the circle touches BC
  q : ℝ × ℝ
  -- The point where the circle touches CD
  r : ℝ × ℝ
  -- The point where the circle touches DA
  s : ℝ × ℝ

/-- The theorem stating the length of the hypotenuse of triangle APD -/
theorem hypotenuse_length (rect : RectangleWithInscribedCircle)
  (h_ab : rect.ab = 20)
  (h_bc : rect.bc = 10) :
  Real.sqrt ((rect.ab - 2 * (rect.ab * rect.bc) / (2 * (rect.ab + rect.bc)))^2 + rect.bc^2) = 50 / 3 :=
by sorry

end hypotenuse_length_l3607_360793


namespace cubic_root_reciprocal_sum_l3607_360703

theorem cubic_root_reciprocal_sum (p q r : ℝ) : 
  p^3 - 9*p^2 + 8*p + 2 = 0 →
  q^3 - 9*q^2 + 8*q + 2 = 0 →
  r^3 - 9*r^2 + 8*r + 2 = 0 →
  p ≠ q → p ≠ r → q ≠ r →
  1/p^2 + 1/q^2 + 1/r^2 = 25 := by
sorry

end cubic_root_reciprocal_sum_l3607_360703


namespace prime_power_fraction_implies_prime_l3607_360710

theorem prime_power_fraction_implies_prime (n : ℕ) (h1 : n ≥ 2) :
  (∃ b : ℕ+, ∃ p : ℕ, ∃ k : ℕ, Prime p ∧ (b^n - 1) / (b - 1) = p^k) →
  Prime n :=
by sorry

end prime_power_fraction_implies_prime_l3607_360710


namespace factors_comparison_infinite_equal_factors_infinite_more_4k1_factors_l3607_360753

-- Define f(n) as the number of prime factors of n of the form 4k+1
def f (n : ℕ+) : ℕ := sorry

-- Define g(n) as the number of prime factors of n of the form 4k+3
def g (n : ℕ+) : ℕ := sorry

-- Statement 1
theorem factors_comparison (n : ℕ+) : f n ≥ g n := by sorry

-- Statement 2
theorem infinite_equal_factors : Set.Infinite {n : ℕ+ | f n = g n} := by sorry

-- Statement 3
theorem infinite_more_4k1_factors : Set.Infinite {n : ℕ+ | f n > g n} := by sorry

end factors_comparison_infinite_equal_factors_infinite_more_4k1_factors_l3607_360753


namespace tangent_point_at_negative_one_slope_l3607_360729

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

theorem tangent_point_at_negative_one_slope :
  ∃ (x : ℝ), f' x = -1 ∧ (x, f x) = (1, 0) := by
  sorry

end tangent_point_at_negative_one_slope_l3607_360729


namespace special_rhombus_sum_l3607_360787

/-- A rhombus with specific vertex coordinates and area -/
structure SpecialRhombus where
  a : ℤ
  b : ℤ
  a_pos : 0 < a
  b_pos : 0 < b
  a_neq_b : a ≠ b
  area_eq : 2 * (a - b)^2 = 32

/-- The sum of a and b in a SpecialRhombus is 8 -/
theorem special_rhombus_sum (r : SpecialRhombus) : r.a + r.b = 8 := by
  sorry

end special_rhombus_sum_l3607_360787


namespace bus_problem_l3607_360741

/-- Calculates the number of children who got on the bus -/
def children_got_on (initial : ℕ) (got_off : ℕ) (final : ℕ) : ℕ :=
  final - (initial - got_off)

/-- Proves that 5 children got on the bus given the initial, final, and number of children who got off -/
theorem bus_problem : children_got_on 21 10 16 = 5 := by
  sorry

end bus_problem_l3607_360741


namespace restaurant_at_park_office_l3607_360743

/-- Represents the time in minutes for various parts of Dante's journey -/
structure JourneyTimes where
  toHiddenLake : ℕ
  fromHiddenLake : ℕ
  toRestaurant : ℕ

/-- The actual journey times given in the problem -/
def actualJourney : JourneyTimes where
  toHiddenLake := 15
  fromHiddenLake := 7
  toRestaurant := 0

/-- Calculates the total time for a journey without visiting the restaurant -/
def totalTimeWithoutRestaurant (j : JourneyTimes) : ℕ :=
  j.toHiddenLake + j.fromHiddenLake

/-- Calculates the total time for a journey with visiting the restaurant -/
def totalTimeWithRestaurant (j : JourneyTimes) : ℕ :=
  j.toRestaurant + j.toRestaurant + j.toHiddenLake + j.fromHiddenLake

/-- Theorem stating that the time to the restaurant is 0 given the journey times are equal -/
theorem restaurant_at_park_office (j : JourneyTimes) 
  (h : totalTimeWithoutRestaurant j = totalTimeWithRestaurant j) : 
  j.toRestaurant = 0 := by
  sorry

#check restaurant_at_park_office

end restaurant_at_park_office_l3607_360743


namespace circle_area_ratio_l3607_360732

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) : 
  (30 / 360 : ℝ) * (2 * Real.pi * r₁) = (45 / 360 : ℝ) * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 9 / 4 := by
  sorry

end circle_area_ratio_l3607_360732


namespace simplify_expression_l3607_360773

theorem simplify_expression : 
  3 * Real.sqrt 48 - 9 * Real.sqrt (1/3) - Real.sqrt 3 * (2 - Real.sqrt 27) = 7 * Real.sqrt 3 + 9 := by
  sorry

end simplify_expression_l3607_360773


namespace jaden_initial_cars_l3607_360706

/-- The number of toy cars Jaden had initially -/
def initial_cars : ℕ := 14

/-- The number of cars Jaden bought -/
def bought_cars : ℕ := 28

/-- The number of cars Jaden received as gifts -/
def gift_cars : ℕ := 12

/-- The number of cars Jaden gave to his sister -/
def sister_cars : ℕ := 8

/-- The number of cars Jaden gave to his friend -/
def friend_cars : ℕ := 3

/-- The number of cars Jaden has left -/
def remaining_cars : ℕ := 43

theorem jaden_initial_cars :
  initial_cars + bought_cars + gift_cars - sister_cars - friend_cars = remaining_cars :=
by sorry

end jaden_initial_cars_l3607_360706


namespace min_occupied_seats_for_150_l3607_360751

/-- The minimum number of occupied seats required to ensure the next person must sit next to someone, given a total number of seats. -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  2 * (total_seats / 4)

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 74 := by
  sorry

#eval min_occupied_seats 150

end min_occupied_seats_for_150_l3607_360751


namespace simplify_sqrt_expression_l3607_360764

theorem simplify_sqrt_expression :
  Real.sqrt (37 - 20 * Real.sqrt 3) = 5 - 2 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_expression_l3607_360764


namespace water_remaining_in_cylinder_l3607_360750

/-- The volume of water remaining in a cylinder after pouring some into a cone -/
theorem water_remaining_in_cylinder (cylinder_volume cone_volume : ℝ) : 
  cylinder_volume = 18 →
  cylinder_volume = 3 * cone_volume →
  cylinder_volume - cone_volume = 12 :=
by sorry

end water_remaining_in_cylinder_l3607_360750


namespace complex_equation_result_l3607_360735

theorem complex_equation_result (x y : ℝ) (h : x * Complex.I - y = -1 + Complex.I) :
  (1 - Complex.I) * (x - y * Complex.I) = -2 * Complex.I := by
  sorry

end complex_equation_result_l3607_360735


namespace intersection_point_l3607_360747

/-- The linear function f(x) = 5x + 1 -/
def f (x : ℝ) : ℝ := 5 * x + 1

/-- The y-axis is the set of points with x-coordinate 0 -/
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}

/-- The graph of f is the set of points (x, f(x)) -/
def graph_f : Set (ℝ × ℝ) := {p | p.2 = f p.1}

theorem intersection_point : 
  (Set.inter graph_f y_axis) = {(0, 1)} := by sorry

end intersection_point_l3607_360747


namespace max_balloon_surface_area_l3607_360760

/-- The maximum surface area of a spherical balloon inscribed in a cube --/
theorem max_balloon_surface_area (a : ℝ) (h : a > 0) :
  ∃ (A : ℝ), A = 2 * Real.pi * a^2 ∧ 
  ∀ (r : ℝ), r > 0 → r ≤ a * Real.sqrt 2 / 2 → 
  4 * Real.pi * r^2 ≤ A := by
  sorry

end max_balloon_surface_area_l3607_360760


namespace simplify_expressions_l3607_360791

theorem simplify_expressions :
  (∃ x, x = Real.sqrt 20 - Real.sqrt 5 + Real.sqrt (1/5) ∧ x = (6 * Real.sqrt 5) / 5) ∧
  (∃ y, y = (Real.sqrt 12 + Real.sqrt 18) / Real.sqrt 3 - 2 * Real.sqrt (1/2) * Real.sqrt 3 ∧ y = 2) := by
  sorry

end simplify_expressions_l3607_360791


namespace cyclists_distance_l3607_360783

/-- Calculates the distance between two cyclists traveling in opposite directions -/
def distance_between_cyclists (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 + speed2) * time

theorem cyclists_distance :
  let speed1 : ℝ := 10
  let speed2 : ℝ := 25
  let time : ℝ := 1.4285714285714286
  distance_between_cyclists speed1 speed2 time = 50 := by
  sorry

end cyclists_distance_l3607_360783


namespace library_books_distribution_l3607_360767

theorem library_books_distribution (a b c d : ℕ) 
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  (a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28) := by
  sorry

end library_books_distribution_l3607_360767


namespace schools_count_proof_l3607_360759

def number_of_schools : ℕ := 24

theorem schools_count_proof :
  ∀ (total_students : ℕ) (andrew_rank : ℕ),
    total_students = 4 * number_of_schools →
    andrew_rank = (total_students + 1) / 2 →
    andrew_rank < 50 →
    andrew_rank > 48 →
    number_of_schools = 24 := by
  sorry

end schools_count_proof_l3607_360759


namespace population_change_l3607_360788

theorem population_change (k m : ℝ) : 
  let decrease_factor : ℝ := 1 - k / 100
  let increase_factor : ℝ := 1 + m / 100
  let total_factor : ℝ := decrease_factor * increase_factor
  total_factor = 1 + (m - k - k * m / 100) / 100 := by sorry

end population_change_l3607_360788


namespace bus_truck_meeting_time_l3607_360763

theorem bus_truck_meeting_time 
  (initial_distance : ℝ) 
  (truck_speed : ℝ) 
  (bus_speed : ℝ) 
  (final_distance : ℝ) 
  (h1 : initial_distance = 8)
  (h2 : truck_speed = 60)
  (h3 : bus_speed = 40)
  (h4 : final_distance = 78) :
  (final_distance - initial_distance) / (truck_speed + bus_speed) = 0.7 := by
sorry

end bus_truck_meeting_time_l3607_360763


namespace at_least_two_equal_l3607_360798

theorem at_least_two_equal (x y z : ℝ) (h : x/y + y/z + z/x = z/y + y/x + x/z) :
  (x = y) ∨ (y = z) ∨ (z = x) := by
  sorry

end at_least_two_equal_l3607_360798


namespace election_winner_percentage_l3607_360734

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 1944 →
  margin = 288 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) * 100 = 54 := by
sorry

end election_winner_percentage_l3607_360734


namespace time_difference_to_halfway_point_l3607_360715

/-- Given that Danny can reach Steve's house in 29 minutes and Steve takes twice as long to reach Danny's house,
    prove that Steve takes 14.5 minutes longer than Danny to reach the halfway point between their houses. -/
theorem time_difference_to_halfway_point (danny_time : ℝ) (steve_time : ℝ) : 
  danny_time = 29 → steve_time = 2 * danny_time → steve_time / 2 - danny_time / 2 = 14.5 := by
  sorry

end time_difference_to_halfway_point_l3607_360715


namespace polynomial_equality_l3607_360745

theorem polynomial_equality (h : ℝ → ℝ) : 
  (∀ x : ℝ, 7 * x^4 - 4 * x^3 + x + h x = 5 * x^3 - 7 * x + 6) →
  (∀ x : ℝ, h x = -7 * x^4 + 9 * x^3 - 8 * x + 6) :=
by
  sorry

end polynomial_equality_l3607_360745


namespace january_salary_l3607_360711

/-- Given the average salaries for two sets of four months and the salary for May,
    calculate the salary for January. -/
theorem january_salary
  (avg_jan_to_apr : ℝ)
  (avg_feb_to_may : ℝ)
  (may_salary : ℝ)
  (h1 : avg_jan_to_apr = 8000)
  (h2 : avg_feb_to_may = 8500)
  (h3 : may_salary = 6500) :
  ∃ (jan feb mar apr : ℝ),
    (jan + feb + mar + apr) / 4 = avg_jan_to_apr ∧
    (feb + mar + apr + may_salary) / 4 = avg_feb_to_may ∧
    jan = 4500 := by
  sorry

#check january_salary

end january_salary_l3607_360711


namespace final_apple_count_l3607_360702

def apples_on_tree (initial : ℕ) (picked : ℕ) (new_growth : ℕ) : ℕ :=
  initial - picked + new_growth

theorem final_apple_count :
  apples_on_tree 11 7 2 = 6 := by
  sorry

end final_apple_count_l3607_360702


namespace steve_initial_berries_l3607_360728

theorem steve_initial_berries (stacy_initial : ℕ) (steve_takes : ℕ) (difference : ℕ) : 
  stacy_initial = 32 →
  steve_takes = 4 →
  stacy_initial - (steve_takes + difference) = stacy_initial - 7 →
  stacy_initial - difference - steve_takes = 21 :=
by
  sorry

end steve_initial_berries_l3607_360728


namespace field_trip_attendance_calculation_l3607_360717

/-- The number of people on a field trip -/
def field_trip_attendance (num_vans : ℕ) (num_buses : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ) : ℕ :=
  num_vans * people_per_van + num_buses * people_per_bus

/-- Theorem stating the total number of people on the field trip -/
theorem field_trip_attendance_calculation :
  field_trip_attendance 6 8 6 18 = 180 := by
  sorry

end field_trip_attendance_calculation_l3607_360717


namespace quadratic_to_linear_inequality_l3607_360777

theorem quadratic_to_linear_inequality (a b : ℝ) :
  (∀ x, x^2 + a*x + b > 0 ↔ x < 3 ∨ x > 1) →
  (∀ x, a*x + b < 0 ↔ x > 3/4) :=
by sorry

end quadratic_to_linear_inequality_l3607_360777


namespace scientific_notation_4040000_l3607_360775

theorem scientific_notation_4040000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 4040000 = a * (10 : ℝ) ^ n ∧ a = 4.04 ∧ n = 6 := by
  sorry

end scientific_notation_4040000_l3607_360775


namespace F_equality_implies_a_half_l3607_360799

/-- Definition of function F -/
def F (a b c : ℝ) : ℝ := a * (b^2 + c^2) + b * c

/-- Theorem: If F(a, 3, 4) = F(a, 2, 5), then a = 1/2 -/
theorem F_equality_implies_a_half :
  ∀ a : ℝ, F a 3 4 = F a 2 5 → a = 1/2 := by
  sorry

end F_equality_implies_a_half_l3607_360799


namespace sixth_sample_number_l3607_360761

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is valid (between 000 and 799) --/
def isValidNumber (n : Nat) : Bool :=
  n ≤ 799

/-- Finds the nth valid number in a list --/
def findNthValidNumber (numbers : List Nat) (n : Nat) : Option Nat :=
  let validNumbers := numbers.filter isValidNumber
  validNumbers.get? (n - 1)

/-- The main theorem --/
theorem sixth_sample_number
  (table : RandomNumberTable)
  (startRow : Nat)
  (startCol : Nat) :
  findNthValidNumber (table.join.drop (startRow * table.head!.length + startCol)) 6 = some 245 :=
sorry

end sixth_sample_number_l3607_360761


namespace juans_number_puzzle_l3607_360779

theorem juans_number_puzzle (n : ℝ) : ((2 * (n + 2) - 2) / 2 = 7) → n = 6 := by
  sorry

end juans_number_puzzle_l3607_360779


namespace fast_food_fries_l3607_360789

theorem fast_food_fries (total : ℕ) (ratio : ℚ) (small : ℕ) : 
  total = 24 → ratio = 5 → small * (1 + ratio) = total → small = 4 := by
  sorry

end fast_food_fries_l3607_360789


namespace division_remainder_problem_l3607_360714

theorem division_remainder_problem (x y u v : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + 3 * u * y + 4 * v) % y = 5 * v % y := by
  sorry

end division_remainder_problem_l3607_360714


namespace product_simplification_l3607_360740

theorem product_simplification :
  (240 : ℚ) / 18 * (9 : ℚ) / 160 * (10 : ℚ) / 3 = 5 / 2 := by
  sorry

end product_simplification_l3607_360740


namespace paper_width_calculation_l3607_360719

theorem paper_width_calculation (w : ℝ) : 
  (2 * w * 17 = 2 * 8.5 * 11 + 100) → w = 287 / 34 := by
  sorry

end paper_width_calculation_l3607_360719


namespace game_x_vs_game_y_l3607_360757

def coin_prob_heads : ℚ := 3/4
def coin_prob_tails : ℚ := 1/4

def game_x_win_prob : ℚ :=
  4 * (coin_prob_heads^4 * coin_prob_tails + coin_prob_tails^4 * coin_prob_heads)

def game_y_win_prob : ℚ :=
  coin_prob_heads^6 + coin_prob_tails^6

theorem game_x_vs_game_y :
  game_x_win_prob - game_y_win_prob = 298/2048 := by sorry

end game_x_vs_game_y_l3607_360757


namespace danny_bottle_caps_l3607_360718

def initial_bottle_caps : ℕ := 6
def found_bottle_caps : ℕ := 22

theorem danny_bottle_caps :
  initial_bottle_caps + found_bottle_caps = 28 :=
by sorry

end danny_bottle_caps_l3607_360718


namespace valid_documents_l3607_360782

theorem valid_documents (total_papers : ℕ) (invalid_percentage : ℚ) 
  (h1 : total_papers = 400)
  (h2 : invalid_percentage = 40 / 100) :
  (total_papers : ℚ) * (1 - invalid_percentage) = 240 := by
  sorry

end valid_documents_l3607_360782


namespace inscribed_rectangle_sides_l3607_360795

/-- A right triangle with an inscribed rectangle -/
structure RightTriangleWithRectangle where
  -- The lengths of the legs of the right triangle
  ab : ℝ
  ac : ℝ
  -- The sides of the inscribed rectangle
  ad : ℝ
  am : ℝ
  -- Conditions
  ab_positive : 0 < ab
  ac_positive : 0 < ac
  ad_positive : 0 < ad
  am_positive : 0 < am
  ad_le_ab : ad ≤ ab
  am_le_ac : am ≤ ac

/-- The theorem statement -/
theorem inscribed_rectangle_sides
  (triangle : RightTriangleWithRectangle)
  (h_ab : triangle.ab = 5)
  (h_ac : triangle.ac = 12)
  (h_area : triangle.ad * triangle.am = 40 / 3)
  (h_diagonal : triangle.ad ^ 2 + triangle.am ^ 2 < 8 ^ 2) :
  triangle.ad = 4 ∧ triangle.am = 10 / 3 :=
sorry

end inscribed_rectangle_sides_l3607_360795


namespace fraction_addition_l3607_360792

theorem fraction_addition (d : ℝ) : (5 + 4 * d) / 8 + 3 = (29 + 4 * d) / 8 := by
  sorry

end fraction_addition_l3607_360792


namespace new_tax_rate_is_32_percent_l3607_360770

/-- Calculates the new tax rate given the original rate, income, and differential savings -/
def calculate_new_tax_rate (original_rate : ℚ) (income : ℚ) (differential_savings : ℚ) : ℚ :=
  (original_rate * income - differential_savings) / income

/-- Theorem stating that the new tax rate is 32% given the problem conditions -/
theorem new_tax_rate_is_32_percent :
  let original_rate : ℚ := 42 / 100
  let income : ℚ := 42400
  let differential_savings : ℚ := 4240
  calculate_new_tax_rate original_rate income differential_savings = 32 / 100 := by
  sorry

#eval calculate_new_tax_rate (42 / 100) 42400 4240

end new_tax_rate_is_32_percent_l3607_360770


namespace min_surface_area_3x3x3_minus_5_l3607_360726

/-- Represents a 3D cube composed of unit cubes -/
structure Cube3D where
  size : Nat
  total_units : Nat

/-- Represents the remaining solid after removing some unit cubes -/
structure RemainingCube where
  original : Cube3D
  removed : Nat

/-- Calculates the minimum surface area of the remaining solid -/
def min_surface_area (rc : RemainingCube) : Nat :=
  sorry

/-- Theorem stating the minimum surface area after removing 5 unit cubes from a 3x3x3 cube -/
theorem min_surface_area_3x3x3_minus_5 :
  let original_cube : Cube3D := { size := 3, total_units := 27 }
  let remaining_cube : RemainingCube := { original := original_cube, removed := 5 }
  min_surface_area remaining_cube = 50 := by
  sorry

end min_surface_area_3x3x3_minus_5_l3607_360726


namespace cube_sum_equality_l3607_360780

theorem cube_sum_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^3 + b^3 + c^3 - 3*a*b*c = 0) : a = b ∧ b = c := by
  sorry

end cube_sum_equality_l3607_360780


namespace log_equality_implies_golden_ratio_l3607_360725

theorem log_equality_implies_golden_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 8 = Real.log q / Real.log 18) ∧
  (Real.log q / Real.log 18 = Real.log (p + q) / Real.log 32) →
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end log_equality_implies_golden_ratio_l3607_360725


namespace books_ratio_proof_l3607_360712

def books_problem (initial_books : ℕ) (rebecca_books : ℕ) (remaining_books : ℕ) : Prop :=
  let mara_books := initial_books - remaining_books - rebecca_books
  mara_books / rebecca_books = 3

theorem books_ratio_proof (initial_books : ℕ) (rebecca_books : ℕ) (remaining_books : ℕ)
  (h1 : initial_books = 220)
  (h2 : rebecca_books = 40)
  (h3 : remaining_books = 60) :
  books_problem initial_books rebecca_books remaining_books :=
by
  sorry

end books_ratio_proof_l3607_360712


namespace muffin_milk_calculation_l3607_360752

/-- Given that 24 muffins require 3 liters of milk and 1 liter equals 4 cups,
    prove that 6 muffins require 3 cups of milk. -/
theorem muffin_milk_calculation (muffins_large : ℕ) (milk_liters : ℕ) (cups_per_liter : ℕ) 
  (muffins_small : ℕ) :
  muffins_large = 24 →
  milk_liters = 3 →
  cups_per_liter = 4 →
  muffins_small = 6 →
  (milk_liters * cups_per_liter * muffins_small) / muffins_large = 3 :=
by
  sorry

#check muffin_milk_calculation

end muffin_milk_calculation_l3607_360752


namespace evaluate_32_to_5_over_2_l3607_360796

theorem evaluate_32_to_5_over_2 : 32^(5/2) = 4096 * Real.sqrt 2 := by
  sorry

end evaluate_32_to_5_over_2_l3607_360796


namespace ninety_percent_of_nine_thousand_l3607_360758

theorem ninety_percent_of_nine_thousand (total_population : ℕ) (percentage : ℚ) : 
  total_population = 9000 → percentage = 90 / 100 → 
  (percentage * total_population : ℚ) = 8100 := by
  sorry

end ninety_percent_of_nine_thousand_l3607_360758


namespace min_value_sum_product_l3607_360756

theorem min_value_sum_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * ((x + y)⁻¹ + (x + z)⁻¹ + (y + z)⁻¹) ≥ (9 : ℝ) / 2 := by
  sorry

end min_value_sum_product_l3607_360756


namespace simplify_expression_1_simplify_expression_2_l3607_360722

-- Problem 1
theorem simplify_expression_1 : 
  (3 * Real.sqrt 8 - 12 * Real.sqrt (1/2) + Real.sqrt 18) * 2 * Real.sqrt 3 = 6 * Real.sqrt 6 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (hx : x > 0) : 
  (6 * Real.sqrt (x/4) - 2*x * Real.sqrt (1/x)) / (3 * Real.sqrt x) = 1/3 := by
  sorry

end simplify_expression_1_simplify_expression_2_l3607_360722


namespace divisibility_by_thirteen_l3607_360744

theorem divisibility_by_thirteen (n : ℕ) : (4 * 3^(2^n) + 3 * 4^(2^n)) % 13 = 0 ↔ n % 2 = 0 := by
  sorry

end divisibility_by_thirteen_l3607_360744


namespace vector_problem_l3607_360754

/-- Given planar vectors a and b with angle π/3 between them, |a| = 2, and |b| = 1,
    prove that a · b = 1 and |a + 2b| = 2√3 -/
theorem vector_problem (a b : ℝ × ℝ) 
    (angle : Real.cos (π / 3) = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
    (mag_a : a.1^2 + a.2^2 = 4)
    (mag_b : b.1^2 + b.2^2 = 1) :
  (a.1 * b.1 + a.2 * b.2 = 1) ∧ 
  ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2 = 12) :=
by sorry

end vector_problem_l3607_360754


namespace quadratic_solution_l3607_360720

theorem quadratic_solution (m : ℝ) : (2 : ℝ)^2 + m * 2 + 2 = 0 → m = -3 := by
  sorry

end quadratic_solution_l3607_360720


namespace paint_calculation_l3607_360731

theorem paint_calculation (initial_paint : ℚ) : 
  (initial_paint / 9 + (initial_paint - initial_paint / 9) / 5 = 104) →
  initial_paint = 360 := by
  sorry

end paint_calculation_l3607_360731


namespace hemisphere_volume_l3607_360755

/-- The volume of a hemisphere with radius 21.002817118114375 cm is 96993.17249452507 cubic centimeters. -/
theorem hemisphere_volume : 
  let r : Real := 21.002817118114375
  let V : Real := (2/3) * Real.pi * r^3
  V = 96993.17249452507 := by sorry

end hemisphere_volume_l3607_360755


namespace thursday_tuesday_difference_l3607_360716

/-- The amount of money Max's mom gave him on Tuesday -/
def tuesday_amount : ℕ := 8

/-- The amount of money Max's mom gave him on Wednesday -/
def wednesday_amount : ℕ := 5 * tuesday_amount

/-- The amount of money Max's mom gave him on Thursday -/
def thursday_amount : ℕ := wednesday_amount + 9

/-- The theorem stating the difference between Thursday's and Tuesday's amounts -/
theorem thursday_tuesday_difference : thursday_amount - tuesday_amount = 41 := by
  sorry

end thursday_tuesday_difference_l3607_360716


namespace quadratic_inequality_no_solution_l3607_360785

theorem quadratic_inequality_no_solution : 
  {x : ℝ | x^2 + 4*x + 4 < 0} = ∅ := by sorry

end quadratic_inequality_no_solution_l3607_360785


namespace walnuts_amount_l3607_360701

/-- The amount of walnuts in the trail mix -/
def walnuts : ℝ := sorry

/-- The total amount of nuts in the trail mix -/
def total_nuts : ℝ := 0.5

/-- The amount of almonds in the trail mix -/
def almonds : ℝ := 0.25

/-- Theorem stating that the amount of walnuts is 0.25 cups -/
theorem walnuts_amount : walnuts = 0.25 := by sorry

end walnuts_amount_l3607_360701


namespace class_receives_reward_l3607_360742

def standard_jumps : ℕ := 160

def performance_records : List ℤ := [16, -1, 20, -2, -5, 11, -7, 6, 9, 13]

def score (x : ℤ) : ℚ :=
  if x ≥ 0 then x
  else -0.5 * x.natAbs

def total_score (records : List ℤ) : ℚ :=
  (records.map score).sum

theorem class_receives_reward (records : List ℤ) :
  records = performance_records →
  total_score records > 65 := by
  sorry

end class_receives_reward_l3607_360742


namespace new_person_weight_l3607_360708

theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 45 →
  avg_increase = 2.5 →
  ∃ (new_weight : ℝ), new_weight = replaced_weight + (initial_count * avg_increase) :=
by
  sorry

end new_person_weight_l3607_360708


namespace calculation_result_l3607_360739

theorem calculation_result : 2009 * 20082008 - 2008 * 20092009 = 0 := by
  sorry

end calculation_result_l3607_360739


namespace trees_in_yard_l3607_360730

/-- Calculates the number of trees in a yard given the yard length and tree spacing -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  (yard_length / tree_spacing) + 1

/-- Theorem stating that the number of trees in a 375-meter yard with 15-meter spacing is 26 -/
theorem trees_in_yard : num_trees 375 15 = 26 := by
  sorry

end trees_in_yard_l3607_360730


namespace quadratic_inequality_always_true_l3607_360724

theorem quadratic_inequality_always_true (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * a * x - 4 < 0) ↔ -4 < a ∧ a ≤ 0 := by
  sorry

end quadratic_inequality_always_true_l3607_360724


namespace grover_boxes_l3607_360748

/-- Represents the number of face masks in each box -/
def masks_per_box : ℕ := 20

/-- Represents the cost of each box in dollars -/
def cost_per_box : ℚ := 15

/-- Represents the selling price of each face mask in dollars -/
def price_per_mask : ℚ := 5/4  -- $1.25

/-- Represents the total profit in dollars -/
def total_profit : ℚ := 15

/-- Calculates the revenue from selling one box of face masks -/
def revenue_per_box : ℚ := masks_per_box * price_per_mask

/-- Calculates the profit from selling one box of face masks -/
def profit_per_box : ℚ := revenue_per_box - cost_per_box

/-- Theorem: Given the conditions, Grover bought 3 boxes of face masks -/
theorem grover_boxes : 
  ∃ (n : ℕ), n * profit_per_box = total_profit ∧ n = 3 := by
  sorry

end grover_boxes_l3607_360748


namespace third_year_planting_l3607_360709

def initial_planting : ℝ := 10000
def annual_increase : ℝ := 0.2

def acres_planted (year : ℕ) : ℝ :=
  initial_planting * (1 + annual_increase) ^ (year - 1)

theorem third_year_planting :
  acres_planted 3 = 14400 := by sorry

end third_year_planting_l3607_360709


namespace smallest_valid_seating_l3607_360765

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : Nat
  seated_people : Nat

/-- Checks if a seating arrangement is valid (no isolated seats). -/
def is_valid_seating (table : CircularTable) : Prop :=
  ∀ n : Nat, n < table.total_chairs → 
    ∃ m : Nat, m < table.seated_people ∧ 
      (n = m ∨ n = (m + 1) % table.total_chairs ∨ n = (m - 1 + table.total_chairs) % table.total_chairs)

/-- The main theorem to be proved. -/
theorem smallest_valid_seating :
  ∀ table : CircularTable, 
    table.total_chairs = 60 →
    (is_valid_seating table ↔ table.seated_people ≥ 15) :=
by sorry

end smallest_valid_seating_l3607_360765


namespace movie_theater_ticket_sales_l3607_360786

/-- Represents the type of ticket --/
inductive TicketType
  | Adult
  | Child
  | SeniorOrStudent

/-- Represents the showtime --/
inductive Showtime
  | Matinee
  | Evening

/-- Returns the price of a ticket based on its type and showtime --/
def ticketPrice (t : TicketType) (s : Showtime) : ℕ :=
  match s, t with
  | Showtime.Matinee, TicketType.Adult => 5
  | Showtime.Matinee, TicketType.Child => 3
  | Showtime.Matinee, TicketType.SeniorOrStudent => 4
  | Showtime.Evening, TicketType.Adult => 9
  | Showtime.Evening, TicketType.Child => 5
  | Showtime.Evening, TicketType.SeniorOrStudent => 6

theorem movie_theater_ticket_sales
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (adult_tickets : ℕ)
  (child_tickets : ℕ)
  (senior_student_tickets : ℕ)
  (h1 : total_tickets = 1500)
  (h2 : total_revenue = 10500)
  (h3 : child_tickets = adult_tickets + 300)
  (h4 : 2 * (adult_tickets + child_tickets) = senior_student_tickets)
  (h5 : total_tickets = adult_tickets + child_tickets + senior_student_tickets) :
  adult_tickets = 100 := by
  sorry

#check movie_theater_ticket_sales

end movie_theater_ticket_sales_l3607_360786
