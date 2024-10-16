import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_one_l3289_328928

theorem negation_of_forall_positive_square_plus_one (P : Real → Prop) : 
  (¬ ∀ x > 1, x^2 + 1 ≥ 0) ↔ (∃ x > 1, x^2 + 1 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_one_l3289_328928


namespace NUMINAMATH_CALUDE_puppy_cost_l3289_328917

/-- Calculates the cost of a puppy given the total cost, food requirements, and food prices. -/
theorem puppy_cost (total_cost : ℚ) (weeks : ℕ) (daily_food : ℚ) (bag_size : ℚ) (bag_cost : ℚ) : 
  total_cost = 14 →
  weeks = 3 →
  daily_food = 1/3 →
  bag_size = 7/2 →
  bag_cost = 2 →
  total_cost - (((weeks * 7 * daily_food) / bag_size).ceil * bag_cost) = 10 := by
  sorry

end NUMINAMATH_CALUDE_puppy_cost_l3289_328917


namespace NUMINAMATH_CALUDE_movie_book_difference_l3289_328924

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 17

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 11

/-- Theorem: The difference between the number of movies and books in the 'crazy silly school' series is 6 -/
theorem movie_book_difference : num_movies - num_books = 6 := by
  sorry

end NUMINAMATH_CALUDE_movie_book_difference_l3289_328924


namespace NUMINAMATH_CALUDE_parabola_shift_equation_l3289_328904

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (k : ℝ) : Parabola :=
  { f := λ x => p.f (x + k) }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (m : ℝ) : Parabola :=
  { f := λ x => p.f x - m }

/-- The original parabola y = x^2 -/
def original_parabola : Parabola :=
  { f := λ x => x^2 }

/-- The resulting parabola after shifting -/
def shifted_parabola : Parabola :=
  shift_vertical (shift_horizontal original_parabola 3) 4

theorem parabola_shift_equation :
  ∀ x, shifted_parabola.f x = (x + 3)^2 - 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_shift_equation_l3289_328904


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3289_328988

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) : 
  c = 2 → b = 2 * a → C = π / 3 → a + b + c = 2 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3289_328988


namespace NUMINAMATH_CALUDE_roosevelt_bonus_points_l3289_328967

/-- Represents the points scored by Roosevelt High School in each game and the bonus points received --/
structure RooseveltPoints where
  first_game : ℕ
  second_game : ℕ
  third_game : ℕ
  bonus : ℕ

/-- Represents the total points scored by Greendale High School --/
def greendale_points : ℕ := 130

/-- Calculates the total points scored by Roosevelt High School before bonus --/
def roosevelt_total (p : RooseveltPoints) : ℕ :=
  p.first_game + p.second_game + p.third_game

/-- Theorem stating the bonus points received by Roosevelt High School --/
theorem roosevelt_bonus_points :
  ∀ p : RooseveltPoints,
  p.first_game = 30 →
  p.second_game = p.first_game / 2 →
  p.third_game = p.second_game * 3 →
  greendale_points = roosevelt_total p + p.bonus →
  p.bonus = 40 := by
  sorry

end NUMINAMATH_CALUDE_roosevelt_bonus_points_l3289_328967


namespace NUMINAMATH_CALUDE_jose_profit_share_l3289_328970

def calculate_share (investment : ℕ) (duration : ℕ) (total_ratio : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * duration * total_profit) / total_ratio

theorem jose_profit_share 
  (tom_investment : ℕ) (tom_duration : ℕ)
  (jose_investment : ℕ) (jose_duration : ℕ)
  (total_profit : ℕ) :
  tom_investment = 30000 →
  tom_duration = 12 →
  jose_investment = 45000 →
  jose_duration = 10 →
  total_profit = 45000 →
  calculate_share jose_investment jose_duration 
    (tom_investment * tom_duration + jose_investment * jose_duration) 
    total_profit = 25000 := by
  sorry

end NUMINAMATH_CALUDE_jose_profit_share_l3289_328970


namespace NUMINAMATH_CALUDE_no_reassignment_possible_l3289_328984

/-- Represents a classroom with rows and columns of chairs -/
structure Classroom where
  rows : Nat
  columns : Nat

/-- Represents the total number of chairs in the classroom -/
def Classroom.totalChairs (c : Classroom) : Nat :=
  c.rows * c.columns

/-- Represents the number of occupied chairs -/
def Classroom.occupiedChairs (c : Classroom) (students : Nat) : Nat :=
  students

/-- Represents whether a reassignment is possible -/
def isReassignmentPossible (c : Classroom) (students : Nat) : Prop :=
  ∃ (redChairs blackChairs : Nat),
    redChairs + blackChairs = c.totalChairs - 1 ∧
    redChairs = students ∧
    blackChairs > redChairs

theorem no_reassignment_possible (c : Classroom) (students : Nat) :
  c.rows = 5 →
  c.columns = 7 →
  students = 34 →
  ¬ isReassignmentPossible c students :=
sorry

end NUMINAMATH_CALUDE_no_reassignment_possible_l3289_328984


namespace NUMINAMATH_CALUDE_community_event_earnings_sharing_l3289_328975

theorem community_event_earnings_sharing (earnings : Fin 3 → ℕ) 
  (h1 : earnings 0 = 18)
  (h2 : earnings 1 = 24)
  (h3 : earnings 2 = 36) :
  36 - (earnings 0 + earnings 1 + earnings 2) / 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_community_event_earnings_sharing_l3289_328975


namespace NUMINAMATH_CALUDE_rigged_coin_probability_l3289_328947

theorem rigged_coin_probability (p : ℝ) (h1 : p < 1/2) 
  (h2 : 20 * p^3 * (1-p)^3 = 1/12) : p = (1 - Real.sqrt 0.86) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rigged_coin_probability_l3289_328947


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3289_328960

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - a 8 = -8 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3289_328960


namespace NUMINAMATH_CALUDE_f1_times_g0_l3289_328953

-- Define f as an odd function on ℝ
def f : ℝ → ℝ := sorry

-- Define g as an even function on ℝ
def g : ℝ → ℝ := sorry

-- Define the relationship between f and g
axiom fg_relation : ∀ x : ℝ, f x - g x = 2^x

-- Define the property of odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Define the property of even function
axiom g_even : ∀ x : ℝ, g (-x) = g x

-- Theorem to prove
theorem f1_times_g0 : f 1 * g 0 = -3/4 := by sorry

end NUMINAMATH_CALUDE_f1_times_g0_l3289_328953


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l3289_328905

theorem line_tangent_to_circle (m n : ℝ) : 
  (∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 ≥ 1) ∧
  (∃ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 ∧ (x - 1)^2 + (y - 1)^2 = 1) →
  m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l3289_328905


namespace NUMINAMATH_CALUDE_debbys_store_inventory_l3289_328951

/-- Represents a DVD rental store inventory --/
structure DVDStore where
  initial_count : ℕ
  rental_rate : ℚ
  sold_count : ℕ

/-- Calculates the remaining DVD count after sales --/
def remaining_dvds (store : DVDStore) : ℕ :=
  store.initial_count - store.sold_count

/-- Theorem stating the remaining DVD count for Debby's store --/
theorem debbys_store_inventory :
  let store : DVDStore := {
    initial_count := 150,
    rental_rate := 35 / 100,
    sold_count := 20
  }
  remaining_dvds store = 130 := by
  sorry

end NUMINAMATH_CALUDE_debbys_store_inventory_l3289_328951


namespace NUMINAMATH_CALUDE_jose_bottle_caps_l3289_328919

theorem jose_bottle_caps (initial : ℝ) (given_away : ℝ) (remaining : ℝ) : 
  initial = 7.0 → given_away = 2.0 → remaining = initial - given_away → remaining = 5.0 := by
  sorry

end NUMINAMATH_CALUDE_jose_bottle_caps_l3289_328919


namespace NUMINAMATH_CALUDE_catch_up_distance_l3289_328978

/-- Represents a car traveling between two cities -/
structure Car where
  speed : ℝ
  startTime : ℝ

/-- Represents the problem scenario -/
structure TwoCarsScenario where
  carA : Car
  carB : Car
  totalDistance : ℝ

/-- The conditions of the problem -/
def problemConditions (scenario : TwoCarsScenario) : Prop :=
  scenario.totalDistance = 300 ∧
  scenario.carA.startTime = scenario.carB.startTime + 1 ∧
  (scenario.totalDistance / scenario.carA.speed) + scenario.carA.startTime =
    (scenario.totalDistance / scenario.carB.speed) + scenario.carB.startTime - 1

/-- The point where carA catches up with carB -/
def catchUpPoint (scenario : TwoCarsScenario) : ℝ :=
  scenario.totalDistance - (scenario.carA.speed * (scenario.carB.startTime - scenario.carA.startTime))

/-- The theorem to be proved -/
theorem catch_up_distance (scenario : TwoCarsScenario) :
  problemConditions scenario → catchUpPoint scenario = 150 := by
  sorry

end NUMINAMATH_CALUDE_catch_up_distance_l3289_328978


namespace NUMINAMATH_CALUDE_triangle_inequality_l3289_328926

theorem triangle_inequality (R r p : ℝ) (hR : R > 0) (hr : r > 0) (hp : p > 0) :
  16 * R * r - 5 * r^2 ≤ p^2 ∧ p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3289_328926


namespace NUMINAMATH_CALUDE_amanda_earnings_l3289_328989

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℝ := 20

/-- Hours worked on Monday -/
def monday_hours : ℝ := 5 * 1.5

/-- Hours worked on Tuesday -/
def tuesday_hours : ℝ := 3

/-- Hours worked on Thursday -/
def thursday_hours : ℝ := 2 * 2

/-- Hours worked on Saturday -/
def saturday_hours : ℝ := 6

/-- Total hours worked in the week -/
def total_hours : ℝ := monday_hours + tuesday_hours + thursday_hours + saturday_hours

/-- Amanda's earnings for the week -/
def weekly_earnings : ℝ := total_hours * hourly_rate

theorem amanda_earnings : weekly_earnings = 410 := by
  sorry

end NUMINAMATH_CALUDE_amanda_earnings_l3289_328989


namespace NUMINAMATH_CALUDE_notebook_buyers_difference_l3289_328964

theorem notebook_buyers_difference : ∃ (price : ℕ) (eighth_buyers fifth_buyers : ℕ),
  price > 0 ∧
  price * eighth_buyers = 210 ∧
  price * fifth_buyers = 240 ∧
  fifth_buyers = 25 ∧
  fifth_buyers - eighth_buyers = 2 :=
by sorry

end NUMINAMATH_CALUDE_notebook_buyers_difference_l3289_328964


namespace NUMINAMATH_CALUDE_chocolate_squares_multiple_l3289_328937

theorem chocolate_squares_multiple (mike_squares jenny_squares : ℕ) 
  (h1 : mike_squares = 20) 
  (h2 : jenny_squares = 65) 
  (h3 : ∃ m : ℕ, jenny_squares = mike_squares * m + 5) : 
  ∃ m : ℕ, m = 3 ∧ jenny_squares = mike_squares * m + 5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_squares_multiple_l3289_328937


namespace NUMINAMATH_CALUDE_equation_with_prime_solutions_l3289_328916

theorem equation_with_prime_solutions (m : ℕ) : 
  (∃ x y : ℕ, Prime x ∧ Prime y ∧ x ≠ y ∧ x^2 - 1999*x + m = 0 ∧ y^2 - 1999*y + m = 0) → 
  m = 3994 := by
sorry

end NUMINAMATH_CALUDE_equation_with_prime_solutions_l3289_328916


namespace NUMINAMATH_CALUDE_abs_ratio_equal_sqrt_seven_thirds_l3289_328998

theorem abs_ratio_equal_sqrt_seven_thirds (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 5*a*b) :
  |((a + b) / (a - b))| = Real.sqrt (7/3) := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_equal_sqrt_seven_thirds_l3289_328998


namespace NUMINAMATH_CALUDE_max_point_of_f_l3289_328939

noncomputable def f (x : ℝ) : ℝ := x^3 - 12*x

theorem max_point_of_f :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x : ℝ), f x ≤ f m :=
sorry

end NUMINAMATH_CALUDE_max_point_of_f_l3289_328939


namespace NUMINAMATH_CALUDE_marble_jar_ratio_l3289_328980

/-- 
Given a collection of marbles distributed in three jars, where:
- The first jar contains 80 marbles
- The second jar contains twice the amount of the first jar
- The total number of marbles is 260

This theorem proves that the ratio of marbles in the third jar to the first jar is 1/4.
-/
theorem marble_jar_ratio : 
  ∀ (jar1 jar2 jar3 : ℕ),
  jar1 = 80 →
  jar2 = 2 * jar1 →
  jar1 + jar2 + jar3 = 260 →
  (jar3 : ℚ) / jar1 = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_marble_jar_ratio_l3289_328980


namespace NUMINAMATH_CALUDE_exists_multiple_with_digit_sum_l3289_328948

/-- Given a natural number, return the sum of its digits -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number that is a multiple of 2007 and whose sum of digits equals 2007 -/
theorem exists_multiple_with_digit_sum :
  ∃ n : ℕ, (∃ k : ℕ, n = k * 2007) ∧ sumOfDigits n = 2007 := by sorry

end NUMINAMATH_CALUDE_exists_multiple_with_digit_sum_l3289_328948


namespace NUMINAMATH_CALUDE_concave_quadrilateral_perimeter_theorem_l3289_328912

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle in 2D space -/
structure Rectangle where
  topLeft : Point
  bottomRight : Point

/-- Check if a point is inside a rectangle -/
def isInsideRectangle (p : Point) (r : Rectangle) : Prop :=
  r.topLeft.x ≤ p.x ∧ p.x ≤ r.bottomRight.x ∧
  r.bottomRight.y ≤ p.y ∧ p.y ≤ r.topLeft.y

/-- Check if a point is inside a triangle formed by three points -/
def isInsideTriangle (p : Point) (a b c : Point) : Prop :=
  sorry  -- Definition of point inside triangle

/-- Calculate the perimeter of a quadrilateral -/
def quadrilateralPerimeter (a b c d : Point) : ℝ :=
  sorry  -- Definition of quadrilateral perimeter

/-- Calculate the perimeter of a rectangle -/
def rectanglePerimeter (r : Rectangle) : ℝ :=
  sorry  -- Definition of rectangle perimeter

theorem concave_quadrilateral_perimeter_theorem 
  (r : Rectangle) (a x y z : Point) :
  isInsideRectangle a r ∧ 
  isInsideRectangle x r ∧ 
  isInsideRectangle y r ∧
  isInsideTriangle z a x y →
  (quadrilateralPerimeter a x y z < rectanglePerimeter r) ∨
  (quadrilateralPerimeter a x z y < rectanglePerimeter r) ∨
  (quadrilateralPerimeter a y z x < rectanglePerimeter r) :=
by sorry

end NUMINAMATH_CALUDE_concave_quadrilateral_perimeter_theorem_l3289_328912


namespace NUMINAMATH_CALUDE_jam_cost_proof_l3289_328973

/-- The cost of jam used for all sandwiches --/
def jam_cost (N B J H : ℕ+) : ℚ :=
  (N * J * 7 : ℚ) / 100

/-- The total cost of ingredients for all sandwiches --/
def total_cost (N B J H : ℕ+) : ℚ :=
  (N * (6 * B + 7 * J + 4 * H) : ℚ) / 100

theorem jam_cost_proof (N B J H : ℕ+) (h1 : N > 1) (h2 : total_cost N B J H = 462/100) :
  jam_cost N B J H = 462/100 := by
  sorry

end NUMINAMATH_CALUDE_jam_cost_proof_l3289_328973


namespace NUMINAMATH_CALUDE_f_2_equals_4_l3289_328992

def f (n : ℕ) : ℕ := 
  (List.range n).sum + n + (List.range n).sum

theorem f_2_equals_4 : f 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_f_2_equals_4_l3289_328992


namespace NUMINAMATH_CALUDE_total_distance_walked_l3289_328914

/-- The total distance walked by two girls, given one walked twice as far as the other -/
theorem total_distance_walked (nadia_distance : ℝ) (h_nadia : nadia_distance = 18) 
  (h_twice : nadia_distance = 2 * (nadia_distance / 2)) : 
  nadia_distance + (nadia_distance / 2) = 27 := by
  sorry

#check total_distance_walked

end NUMINAMATH_CALUDE_total_distance_walked_l3289_328914


namespace NUMINAMATH_CALUDE_tank_emptying_time_specific_tank_emptying_time_l3289_328957

/-- Proves that a tank with given volume and flow rates empties in the specified time -/
theorem tank_emptying_time (tank_volume_cubic_feet : ℝ) 
                            (inlet_rate : ℝ) 
                            (outlet_rate_1 outlet_rate_2 outlet_rate_3 outlet_rate_4 : ℝ) 
                            (inches_per_foot : ℝ) : ℝ :=
  let tank_volume_cubic_inches := tank_volume_cubic_feet * (inches_per_foot^3)
  let total_outflow_rate := outlet_rate_1 + outlet_rate_2 + outlet_rate_3 + outlet_rate_4
  let net_outflow_rate := total_outflow_rate - inlet_rate
  tank_volume_cubic_inches / net_outflow_rate

/-- The specific instance of the tank emptying problem -/
theorem specific_tank_emptying_time : 
  tank_emptying_time 60 3 12 6 18 9 12 = 2468.57 := by
  sorry

end NUMINAMATH_CALUDE_tank_emptying_time_specific_tank_emptying_time_l3289_328957


namespace NUMINAMATH_CALUDE_reporter_wrong_l3289_328900

/-- Represents a round-robin chess tournament --/
structure ChessTournament where
  num_players : ℕ
  wins : Fin num_players → ℕ
  draws : Fin num_players → ℕ
  losses : Fin num_players → ℕ

/-- The total number of games in a round-robin tournament --/
def total_games (t : ChessTournament) : ℕ :=
  t.num_players * (t.num_players - 1) / 2

/-- The total points scored in the tournament --/
def total_points (t : ChessTournament) : ℕ :=
  2 * total_games t

/-- Theorem stating that it's impossible for each player to have won as many games as they drew --/
theorem reporter_wrong (t : ChessTournament) (h1 : t.num_players = 20) 
    (h2 : ∀ i, t.wins i = t.draws i) : False := by
  sorry


end NUMINAMATH_CALUDE_reporter_wrong_l3289_328900


namespace NUMINAMATH_CALUDE_two_dogs_food_consumption_l3289_328941

/-- The amount of dog food eaten by two dogs per day -/
def total_dog_food (dog1_food : ℝ) (dog2_food : ℝ) : ℝ :=
  dog1_food + dog2_food

/-- Theorem stating that two dogs eating 0.125 scoops each consume 0.25 scoops in total -/
theorem two_dogs_food_consumption :
  total_dog_food 0.125 0.125 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_two_dogs_food_consumption_l3289_328941


namespace NUMINAMATH_CALUDE_right_triangle_height_l3289_328976

theorem right_triangle_height (a b c h : ℝ) : 
  a = 25 → b = 20 → c^2 = a^2 - b^2 → h * a = 2 * (1/2 * b * c) → h = 12 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_height_l3289_328976


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3289_328999

/-- The number of ways to choose k items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The problem statement -/
theorem pizza_toppings_combinations :
  choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3289_328999


namespace NUMINAMATH_CALUDE_binomial_square_expansion_l3289_328934

theorem binomial_square_expansion (x : ℝ) : (1 - x)^2 = 1 - 2*x + x^2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_expansion_l3289_328934


namespace NUMINAMATH_CALUDE_inequality_proof_l3289_328907

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x^2019 + y = 1) :
  x + y^2019 > 1 - 1/300 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3289_328907


namespace NUMINAMATH_CALUDE_race_distance_l3289_328990

theorem race_distance (d : ℝ) (vA vB vC : ℝ) 
  (h1 : d / vA = (d - 20) / vB)
  (h2 : d / vB = (d - 10) / vC)
  (h3 : d / vA = (d - 28) / vC)
  (h4 : d > 0) (h5 : vA > 0) (h6 : vB > 0) (h7 : vC > 0) : d = 100 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l3289_328990


namespace NUMINAMATH_CALUDE_unique_c_for_degree_3_l3289_328940

/-- The polynomial f(x) = 2 - 10x + 4x^2 - 5x^3 + 7x^4 -/
def f (x : ℝ) : ℝ := 2 - 10*x + 4*x^2 - 5*x^3 + 7*x^4

/-- The polynomial g(x) = 5 - 3x - 8x^3 + 11x^4 -/
def g (x : ℝ) : ℝ := 5 - 3*x - 8*x^3 + 11*x^4

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The coefficient of x^4 in h(x) -/
def coeff_x4 (c : ℝ) : ℝ := 7 + 11*c

/-- The coefficient of x^3 in h(x) -/
def coeff_x3 (c : ℝ) : ℝ := -5 - 8*c

theorem unique_c_for_degree_3 :
  ∃! c : ℝ, coeff_x4 c = 0 ∧ coeff_x3 c ≠ 0 ∧ c = -7/11 := by sorry

end NUMINAMATH_CALUDE_unique_c_for_degree_3_l3289_328940


namespace NUMINAMATH_CALUDE_tv_screen_height_l3289_328991

theorem tv_screen_height (area : ℝ) (base1 : ℝ) (base2 : ℝ) (h : area = 21 ∧ base1 = 3 ∧ base2 = 5) :
  ∃ height : ℝ, area = (1/2) * (base1 + base2) * height ∧ height = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_tv_screen_height_l3289_328991


namespace NUMINAMATH_CALUDE_prob_A_or_B_prob_A_prob_B_given_A_l3289_328906

-- Define the class composition
def total_officials : ℕ := 6
def male_officials : ℕ := 4
def female_officials : ℕ := 2
def selected_officials : ℕ := 3

-- Define the events
def event_A : Set (Fin total_officials) := sorry
def event_B : Set (Fin total_officials) := sorry

-- Define the probability measure
noncomputable def P : Set (Fin total_officials) → ℝ := sorry

-- Theorem statements
theorem prob_A_or_B : P (event_A ∪ event_B) = 4/5 := by sorry

theorem prob_A : P event_A = 1/2 := by sorry

theorem prob_B_given_A : P (event_B ∩ event_A) / P event_A = 2/5 := by sorry

end NUMINAMATH_CALUDE_prob_A_or_B_prob_A_prob_B_given_A_l3289_328906


namespace NUMINAMATH_CALUDE_highest_throw_l3289_328965

def christine_throw_1 : ℕ := 20
def janice_throw_1 : ℕ := christine_throw_1 - 4
def christine_throw_2 : ℕ := christine_throw_1 + 10
def janice_throw_2 : ℕ := janice_throw_1 * 2
def christine_throw_3 : ℕ := christine_throw_2 + 4
def janice_throw_3 : ℕ := christine_throw_1 + 17

theorem highest_throw :
  max christine_throw_1 (max christine_throw_2 (max christine_throw_3 (max janice_throw_1 (max janice_throw_2 janice_throw_3)))) = 37 := by
  sorry

end NUMINAMATH_CALUDE_highest_throw_l3289_328965


namespace NUMINAMATH_CALUDE_twenty_triangles_l3289_328961

/-- Represents a rectangle divided into smaller rectangles with diagonal and vertical lines -/
structure DividedRectangle where
  smallRectangles : Nat
  diagonalsPerSmallRectangle : Nat
  verticalLinesPerSmallRectangle : Nat

/-- Counts the total number of triangles in the divided rectangle -/
def countTriangles (r : DividedRectangle) : Nat :=
  sorry

/-- Theorem stating that the specific configuration results in 20 triangles -/
theorem twenty_triangles :
  let r : DividedRectangle := {
    smallRectangles := 4,
    diagonalsPerSmallRectangle := 1,
    verticalLinesPerSmallRectangle := 1
  }
  countTriangles r = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_triangles_l3289_328961


namespace NUMINAMATH_CALUDE_room_area_in_sq_meters_l3289_328909

/-- The conversion factor from square feet to square meters -/
def sq_ft_to_sq_m : ℝ := 0.092903

/-- The length of the room in feet -/
def room_length : ℝ := 15

/-- The width of the room in feet -/
def room_width : ℝ := 8

/-- Theorem stating that the area of the room in square meters is approximately 11.14836 -/
theorem room_area_in_sq_meters :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ 
  |room_length * room_width * sq_ft_to_sq_m - 11.14836| < ε :=
sorry

end NUMINAMATH_CALUDE_room_area_in_sq_meters_l3289_328909


namespace NUMINAMATH_CALUDE_cat_whiskers_relationship_l3289_328946

theorem cat_whiskers_relationship (princess_puff_whiskers catman_do_whiskers : ℕ) 
  (h1 : princess_puff_whiskers = 14) 
  (h2 : catman_do_whiskers = 22) : 
  (catman_do_whiskers - princess_puff_whiskers = 8) ∧ 
  (catman_do_whiskers : ℚ) / (princess_puff_whiskers : ℚ) = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cat_whiskers_relationship_l3289_328946


namespace NUMINAMATH_CALUDE_sum_of_solutions_l3289_328913

theorem sum_of_solutions (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ a b c : ℝ) 
  (eq1 : a₁ * (b₂ * c₃ - b₃ * c₂) - a₂ * (b₁ * c₃ - b₃ * c₁) + a₃ * (b₁ * c₂ - b₂ * c₁) = 9)
  (eq2 : a * (b₂ * c₃ - b₃ * c₂) - a₂ * (b * c₃ - b₃ * c) + a₃ * (b * c₂ - b₂ * c) = 17)
  (eq3 : a₁ * (b * c₃ - b₃ * c) - a * (b₁ * c₃ - b₃ * c₁) + a₃ * (b₁ * c - b * c₁) = -8)
  (eq4 : a₁ * (b₂ * c - b * c₂) - a₂ * (b₁ * c - b * c₁) + a * (b₁ * c₂ - b₂ * c₁) = 7)
  (sys1 : a₁ * x + a₂ * y + a₃ * z = a)
  (sys2 : b₁ * x + b₂ * y + b₃ * z = b)
  (sys3 : c₁ * x + c₂ * y + c₃ * z = c) :
  x + y + z = 16/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l3289_328913


namespace NUMINAMATH_CALUDE_tommy_bike_ride_l3289_328985

/-- Tommy's bike riding problem -/
theorem tommy_bike_ride (tommy_width : ℕ) (tommy_north : ℕ) (friend_area : ℕ) 
  (h1 : tommy_width = 1)
  (h2 : tommy_north = 2)
  (h3 : friend_area = 80)
  (h4 : ∃ s : ℕ, 4 * (tommy_width * (tommy_north + s)) = friend_area) :
  ∃ s : ℕ, s = 18 ∧ 4 * (tommy_width * (tommy_north + s)) = friend_area := by
sorry

end NUMINAMATH_CALUDE_tommy_bike_ride_l3289_328985


namespace NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l3289_328954

/-- Represents the ratio of boarders to day students -/
structure Ratio where
  boarders : ℕ
  day_students : ℕ

/-- Calculates the new ratio after adding new boarders -/
def new_ratio (initial : Ratio) (new_boarders : ℕ) : Ratio :=
  { boarders := initial.boarders + new_boarders,
    day_students := initial.day_students }

/-- Simplifies a ratio by dividing both numbers by their GCD -/
def simplify_ratio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.boarders r.day_students
  { boarders := r.boarders / gcd,
    day_students := r.day_students / gcd }

theorem new_ratio_is_one_to_two :
  let initial_ratio : Ratio := { boarders := 330, day_students := 792 }
  let new_boarders : ℕ := 66
  let final_ratio := simplify_ratio (new_ratio initial_ratio new_boarders)
  final_ratio.boarders = 1 ∧ final_ratio.day_students = 2 := by
  sorry


end NUMINAMATH_CALUDE_new_ratio_is_one_to_two_l3289_328954


namespace NUMINAMATH_CALUDE_basketball_cards_per_box_l3289_328972

theorem basketball_cards_per_box : 
  ∀ (basketball_cards_per_box : ℕ),
    (4 * basketball_cards_per_box + 5 * 8 = 58 + 22) → 
    basketball_cards_per_box = 10 := by
  sorry

end NUMINAMATH_CALUDE_basketball_cards_per_box_l3289_328972


namespace NUMINAMATH_CALUDE_sqrt_256_squared_plus_100_l3289_328952

theorem sqrt_256_squared_plus_100 : (Real.sqrt 256)^2 + 100 = 356 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_256_squared_plus_100_l3289_328952


namespace NUMINAMATH_CALUDE_factor_x4_minus_16_l3289_328962

theorem factor_x4_minus_16 (x : ℂ) : x^4 - 16 = (x - 2) * (x + 2) * (x - 2*I) * (x + 2*I) := by
  sorry

end NUMINAMATH_CALUDE_factor_x4_minus_16_l3289_328962


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3289_328915

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 179) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3289_328915


namespace NUMINAMATH_CALUDE_union_is_reals_intersect_complement_l3289_328956

-- Define the sets A and B
def A : Set ℝ := {x | x - 2 ≥ 0}
def B : Set ℝ := {x | x < 5}

-- Theorem for the union of A and B
theorem union_is_reals : A ∪ B = Set.univ := by sorry

-- Theorem for the intersection of complement of A and B
theorem intersect_complement : (Set.univ \ A) ∩ B = {x : ℝ | x < 2} := by sorry

end NUMINAMATH_CALUDE_union_is_reals_intersect_complement_l3289_328956


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3289_328979

theorem least_addition_for_divisibility : ∃! n : ℕ, 
  (∀ m : ℕ, m < n → ¬((1077 + m) % 23 = 0 ∧ (1077 + m) % 17 = 0)) ∧ 
  ((1077 + n) % 23 = 0 ∧ (1077 + n) % 17 = 0) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3289_328979


namespace NUMINAMATH_CALUDE_a_is_positive_l3289_328923

theorem a_is_positive (x y a : ℝ) (h1 : x < y) (h2 : a * x < a * y) : a > 0 := by
  sorry

end NUMINAMATH_CALUDE_a_is_positive_l3289_328923


namespace NUMINAMATH_CALUDE_f_range_l3289_328997

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 5

-- Define the domain
def domain : Set ℝ := { x | -3 ≤ x ∧ x ≤ 0 }

-- Define the range
def range : Set ℝ := { y | ∃ x ∈ domain, f x = y }

-- Theorem statement
theorem f_range : range = { y | -6 ≤ y ∧ y ≤ -2 } := by sorry

end NUMINAMATH_CALUDE_f_range_l3289_328997


namespace NUMINAMATH_CALUDE_point_M_coordinates_l3289_328955

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 4 * x

theorem point_M_coordinates :
  ∃ (x y : ℝ), f' x = -4 ∧ f x = y ∧ x = -1 ∧ y = 3 :=
by
  sorry

#check point_M_coordinates

end NUMINAMATH_CALUDE_point_M_coordinates_l3289_328955


namespace NUMINAMATH_CALUDE_tower_heights_count_l3289_328966

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of different tower heights achievable -/
def countTowerHeights (numBricks : ℕ) (brickDim : BrickDimensions) : ℕ :=
  sorry

/-- The main theorem stating the number of different tower heights -/
theorem tower_heights_count :
  let numBricks : ℕ := 200
  let brickDim : BrickDimensions := ⟨3, 8, 20⟩
  countTowerHeights numBricks brickDim = 680 :=
by sorry

end NUMINAMATH_CALUDE_tower_heights_count_l3289_328966


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3289_328959

/-- An arithmetic sequence with common difference d ≠ 0 where a₁, a₄, and a₁₀ form a geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℝ  -- The arithmetic sequence
  d : ℝ      -- Common difference
  hd : d ≠ 0 -- d is non-zero
  arithmetic_seq : ∀ n, a (n + 1) = a n + d  -- Arithmetic sequence property
  geometric_seq : (a 4) ^ 2 = a 1 * a 10     -- Geometric sequence property for a₁, a₄, a₁₀

/-- The ratio of the first term to the common difference is 3 -/
theorem arithmetic_geometric_ratio (seq : ArithmeticGeometricSequence) : seq.a 1 / seq.d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3289_328959


namespace NUMINAMATH_CALUDE_max_k_value_l3289_328901

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * Real.log x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := a + Real.log x + 1

-- State the theorem
theorem max_k_value (a : ℝ) :
  (f' a (Real.exp (-1)) = 1) →
  (∃ k : ℤ, ∀ x > 1, f a x - k * x + k > 0) →
  (∀ k : ℤ, k > 3 → ∃ x > 1, f a x - k * x + k ≤ 0) :=
sorry

end

end NUMINAMATH_CALUDE_max_k_value_l3289_328901


namespace NUMINAMATH_CALUDE_inequality_proof_l3289_328903

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  Real.sqrt (a * b / (c + a * b)) + Real.sqrt (b * c / (a + b * c)) + Real.sqrt (c * a / (b + c * a)) ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3289_328903


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l3289_328930

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | (x+2)*(4-x) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a+1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) : B ∪ C a = B → a ∈ Set.Icc (-2) 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l3289_328930


namespace NUMINAMATH_CALUDE_cooper_fence_bricks_l3289_328971

/-- Represents the dimensions of a wall in bricks -/
structure WallDimensions where
  length : Nat
  height : Nat
  depth : Nat

/-- Calculates the number of bricks needed for a wall -/
def bricksForWall (wall : WallDimensions) : Nat :=
  wall.length * wall.height * wall.depth

/-- The dimensions of Cooper's four walls -/
def wall1 : WallDimensions := { length := 15, height := 6, depth := 3 }
def wall2 : WallDimensions := { length := 20, height := 4, depth := 2 }
def wall3 : WallDimensions := { length := 25, height := 5, depth := 3 }
def wall4 : WallDimensions := { length := 17, height := 7, depth := 2 }

/-- Theorem: The total number of bricks needed for Cooper's fence is 1043 -/
theorem cooper_fence_bricks :
  bricksForWall wall1 + bricksForWall wall2 + bricksForWall wall3 + bricksForWall wall4 = 1043 := by
  sorry

end NUMINAMATH_CALUDE_cooper_fence_bricks_l3289_328971


namespace NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l3289_328993

theorem square_difference_formula_inapplicable (a b : ℝ) :
  ¬∃ (x y : ℝ), (a - b) * (b - a) = x^2 - y^2 :=
sorry

end NUMINAMATH_CALUDE_square_difference_formula_inapplicable_l3289_328993


namespace NUMINAMATH_CALUDE_sum_equals_16x_l3289_328929

/-- Given real numbers x, y, z, and w, where y = 2x, z = 3y, and w = z + x,
    prove that x + y + z + w = 16x -/
theorem sum_equals_16x (x y z w : ℝ) 
    (h1 : y = 2 * x) 
    (h2 : z = 3 * y) 
    (h3 : w = z + x) : 
  x + y + z + w = 16 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_16x_l3289_328929


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l3289_328996

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → A < π →
  B > 0 → B < π →
  C > 0 → C < π →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  A = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l3289_328996


namespace NUMINAMATH_CALUDE_mower_team_size_l3289_328963

/-- Represents the mowing rate of one mower per day -/
def mower_rate : ℝ := 1

/-- Represents the area of the smaller meadow -/
def small_meadow : ℝ := 2 * mower_rate

/-- Represents the area of the larger meadow -/
def large_meadow : ℝ := 2 * small_meadow

/-- Represents the number of mowers in the team -/
def team_size : ℕ := 8

theorem mower_team_size :
  (team_size : ℝ) * mower_rate / 2 + (team_size : ℝ) * mower_rate / 2 = large_meadow ∧
  (team_size : ℝ) * mower_rate / 4 + mower_rate = small_meadow :=
by sorry

#check mower_team_size

end NUMINAMATH_CALUDE_mower_team_size_l3289_328963


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3289_328935

theorem contrapositive_equivalence (f : ℝ → ℝ) (a b : ℝ) :
  (¬(f a + f b ≥ f (-a) + f (-b)) → ¬(a + b ≥ 0)) ↔
  (f a + f b < f (-a) + f (-b) → a + b < 0) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3289_328935


namespace NUMINAMATH_CALUDE_triangle_area_comparison_l3289_328911

/-- A triangle with side lengths and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

/-- Predicate to check if a triangle is acute -/
def is_acute (t : Triangle) : Prop := sorry

theorem triangle_area_comparison (t₁ t₂ : Triangle) 
  (h_acute : is_acute t₂)
  (h_a : t₁.a ≤ t₂.a)
  (h_b : t₁.b ≤ t₂.b)
  (h_c : t₁.c ≤ t₂.c) :
  t₁.area ≤ t₂.area :=
sorry

end NUMINAMATH_CALUDE_triangle_area_comparison_l3289_328911


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l3289_328981

/-- Given a positive geometric sequence {a_n} with a_2 = 2 and 2a_3 + a_4 = 16,
    prove that the general term formula is a_n = 2^(n-1) -/
theorem geometric_sequence_formula (a : ℕ → ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_a2 : a 2 = 2)
  (h_sum : 2 * a 3 + a 4 = 16) :
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l3289_328981


namespace NUMINAMATH_CALUDE_dresses_count_l3289_328902

/-- The total number of dresses for Emily, Melissa, Debora, and Sophia -/
def total_dresses (emily melissa debora sophia : ℕ) : ℕ :=
  emily + melissa + debora + sophia

/-- Theorem stating the total number of dresses given the conditions -/
theorem dresses_count :
  ∀ (emily melissa debora sophia : ℕ),
    emily = 16 →
    melissa = emily / 2 →
    debora = melissa + 12 →
    sophia = debora - 5 →
    total_dresses emily melissa debora sophia = 59 := by
  sorry

end NUMINAMATH_CALUDE_dresses_count_l3289_328902


namespace NUMINAMATH_CALUDE_house_store_transaction_l3289_328920

theorem house_store_transaction : 
  ∀ (house_cost store_cost : ℝ),
  house_cost * 0.9 = 9000 →
  store_cost * 1.3 = 13000 →
  (9000 + 13000) - (house_cost + store_cost) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_house_store_transaction_l3289_328920


namespace NUMINAMATH_CALUDE_final_score_proof_l3289_328994

def game_score (initial : ℕ) (penalty : ℕ) (additional : ℕ) : ℕ :=
  initial - penalty + additional

theorem final_score_proof :
  game_score 92 15 3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_final_score_proof_l3289_328994


namespace NUMINAMATH_CALUDE_first_car_speed_l3289_328945

/-- Proves that the speed of the first car is 50 km/h given the specified conditions. -/
theorem first_car_speed (time1 : ℝ) (speed2 distance_ratio : ℝ) 
  (h1 : time1 = 6)
  (h2 : speed2 = 100)
  (h3 : distance_ratio = 3)
  (h4 : speed2 * 1 = distance_ratio * (speed2 * 1)) :
  ∃ (speed1 : ℝ), speed1 * time1 = distance_ratio * (speed2 * 1) ∧ speed1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_first_car_speed_l3289_328945


namespace NUMINAMATH_CALUDE_exists_correct_coloring_l3289_328932

/-- Represents the color of a square on the board -/
inductive Color
| White
| Black

/-- Represents a position on the board -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the game board -/
def Board := Position → Color

/-- Checks if two positions are adjacent -/
def adjacent (p1 p2 : Position) : Bool :=
  (p1.row = p2.row ∧ (p1.col + 1 = p2.col ∨ p2.col + 1 = p1.col)) ∨
  (p1.col = p2.col ∧ (p1.row + 1 = p2.row ∨ p2.row + 1 = p1.row))

/-- Checks if a position is within the 4x8 board -/
def validPosition (p : Position) : Bool :=
  p.row < 4 ∧ p.col < 8

/-- Inverts the color -/
def invertColor (c : Color) : Color :=
  match c with
  | Color.White => Color.Black
  | Color.Black => Color.White

/-- Applies a move to the board -/
def applyMove (board : Board) (topLeft : Position) : Board :=
  λ p => if p.row ∈ [topLeft.row, topLeft.row + 1] ∧ 
            p.col ∈ [topLeft.col, topLeft.col + 1]
         then invertColor (board p)
         else board p

/-- Checks if the board is correctly colored -/
def isCorrectlyColored (board : Board) : Prop :=
  ∀ p1 p2, validPosition p1 ∧ validPosition p2 ∧ adjacent p1 p2 →
    board p1 ≠ board p2

/-- The main theorem to prove -/
theorem exists_correct_coloring :
  ∃ (finalBoard : Board),
    (∃ (moves : List Position), 
      finalBoard = (moves.foldl applyMove (λ _ => Color.White)) ∧
      isCorrectlyColored finalBoard) :=
sorry

end NUMINAMATH_CALUDE_exists_correct_coloring_l3289_328932


namespace NUMINAMATH_CALUDE_product_digits_sum_base7_l3289_328977

/-- Converts a base-7 number to base-10 --/
def toBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a number in base-7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_digits_sum_base7 :
  let x := 35
  let y := 21
  sumOfDigitsBase7 (toBase7 (toBase10 x * toBase10 y)) = 15 := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_base7_l3289_328977


namespace NUMINAMATH_CALUDE_goldbach_for_given_numbers_l3289_328983

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def goldbach_for_number (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem goldbach_for_given_numbers :
  goldbach_for_number 102 ∧
  goldbach_for_number 144 ∧
  goldbach_for_number 178 ∧
  goldbach_for_number 200 :=
sorry

end NUMINAMATH_CALUDE_goldbach_for_given_numbers_l3289_328983


namespace NUMINAMATH_CALUDE_altitude_intersection_property_l3289_328925

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is acute -/
def isAcute (t : Triangle) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Checks if a line is an altitude of a triangle -/
def isAltitude (t : Triangle) (p1 p2 : Point) : Prop := sorry

/-- Checks if two lines intersect at a point -/
def intersectAt (p1 p2 p3 p4 p5 : Point) : Prop := sorry

theorem altitude_intersection_property (ABC : Triangle) (D E H : Point) :
  isAcute ABC →
  isAltitude ABC A D →
  isAltitude ABC B E →
  intersectAt A D B E H →
  distance H D = 3 →
  distance H E = 4 →
  ∃ (BD DC AE EC : ℝ),
    BD * DC - AE * EC = 3 * distance A D - 7 := by
  sorry

end NUMINAMATH_CALUDE_altitude_intersection_property_l3289_328925


namespace NUMINAMATH_CALUDE_fermat_like_contradiction_l3289_328958

theorem fermat_like_contradiction (a b c : ℝ) (m n : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hm : 0 < m) (hn : 0 < n) (hmn : m ≠ n) :
  ¬(a^m + b^m = c^m ∧ a^n + b^n = c^n) :=
sorry

end NUMINAMATH_CALUDE_fermat_like_contradiction_l3289_328958


namespace NUMINAMATH_CALUDE_cheyenne_pots_count_l3289_328969

/-- The number of clay pots Cheyenne made -/
def total_pots : ℕ := 80

/-- The fraction of pots that cracked -/
def cracked_fraction : ℚ := 2/5

/-- The revenue from selling the remaining pots -/
def revenue : ℕ := 1920

/-- The price of each clay pot -/
def price_per_pot : ℕ := 40

/-- Theorem stating that the number of clay pots Cheyenne made is 80 -/
theorem cheyenne_pots_count :
  total_pots = 80 ∧
  cracked_fraction = 2/5 ∧
  revenue = 1920 ∧
  price_per_pot = 40 ∧
  (1 - cracked_fraction) * total_pots * price_per_pot = revenue :=
by sorry

end NUMINAMATH_CALUDE_cheyenne_pots_count_l3289_328969


namespace NUMINAMATH_CALUDE_converse_xy_zero_x_zero_is_true_l3289_328922

theorem converse_xy_zero_x_zero_is_true :
  ∀ (x y : ℝ), x = 0 → x * y = 0 :=
by sorry

end NUMINAMATH_CALUDE_converse_xy_zero_x_zero_is_true_l3289_328922


namespace NUMINAMATH_CALUDE_star_emilio_sum_difference_l3289_328943

def star_list := List.range 30

def emilio_list := star_list.map (fun n => 
  if n % 10 = 3 then n - 1
  else if n ≥ 30 then n - 10
  else n)

theorem star_emilio_sum_difference :
  (star_list.sum - emilio_list.sum) = 13 := by sorry

end NUMINAMATH_CALUDE_star_emilio_sum_difference_l3289_328943


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3289_328986

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a + 2) * x^2 + 2 * (a + 2) * x + 4 > 0) ↔ a ∈ Set.Ici (-2) ∩ Set.Iio 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3289_328986


namespace NUMINAMATH_CALUDE_triangle_structure_pieces_l3289_328944

/-- Calculates the sum of arithmetic sequence -/
def arithmeticSum (a₁ n d : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

/-- Represents the structure of the triangle -/
structure TriangleStructure where
  rows : ℕ
  firstRowRods : ℕ
  rodIncrement : ℕ

/-- Calculates the total number of pieces in the triangle structure with base -/
def totalPieces (t : TriangleStructure) : ℕ :=
  let topRods := arithmeticSum t.firstRowRods t.rows t.rodIncrement
  let topConnectors := arithmeticSum 1 (t.rows + 1) 1
  let baseRods := 2 * (t.firstRowRods + (t.rows - 1) * t.rodIncrement)
  let basePieces := 2 * baseRods
  topRods + topConnectors + basePieces

/-- The main theorem to prove -/
theorem triangle_structure_pieces :
  let t : TriangleStructure := { rows := 10, firstRowRods := 3, rodIncrement := 3 }
  totalPieces t = 351 := by
  sorry

end NUMINAMATH_CALUDE_triangle_structure_pieces_l3289_328944


namespace NUMINAMATH_CALUDE_equation_solution_l3289_328942

theorem equation_solution : 
  ∃ x : ℚ, (3 / 4 - 2 / 5 : ℚ) = 1 / x ∧ x = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3289_328942


namespace NUMINAMATH_CALUDE_shirt_price_l3289_328918

theorem shirt_price (total_cost : ℝ) (price_difference : ℝ) (shirt_price : ℝ) :
  total_cost = 80.34 →
  shirt_price = (total_cost + price_difference) / 2 - price_difference →
  price_difference = 7.43 →
  shirt_price = 36.455 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_l3289_328918


namespace NUMINAMATH_CALUDE_banana_solution_l3289_328936

/-- Represents the banana cutting scenario -/
def banana_problem (initial_bananas : ℕ) (cut_bananas : ℕ) (eaten_bananas : ℕ) : Prop :=
  initial_bananas ≥ cut_bananas ∧
  cut_bananas > eaten_bananas ∧
  cut_bananas - eaten_bananas = 2 * (initial_bananas - cut_bananas)

/-- Theorem stating the solution to the banana problem -/
theorem banana_solution :
  ∃ (cut_bananas : ℕ),
    banana_problem 310 cut_bananas 70 ∧
    310 - cut_bananas = 100 := by
  sorry

end NUMINAMATH_CALUDE_banana_solution_l3289_328936


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l3289_328931

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let A : ℝ × ℝ := (x, 8)
  let B : ℝ × ℝ := reflect_over_y_axis A
  A.1 + A.2 + B.1 + B.2 = 16 := by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l3289_328931


namespace NUMINAMATH_CALUDE_mod_twelve_equiv_nine_l3289_328949

theorem mod_twelve_equiv_nine : 
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ -2187 [ZMOD 12] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_twelve_equiv_nine_l3289_328949


namespace NUMINAMATH_CALUDE_rachel_homework_difference_l3289_328968

def rachel_homework (math_pages reading_pages biology_pages : ℕ) : Prop :=
  math_pages - reading_pages = 7

theorem rachel_homework_difference :
  rachel_homework 9 2 96 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_difference_l3289_328968


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l3289_328974

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (h1 : area = 36) (h2 : side * side = area) :
  4 * side = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l3289_328974


namespace NUMINAMATH_CALUDE_team_b_size_l3289_328938

/-- Proves that Team B has 9 people given the competition conditions -/
theorem team_b_size (team_a_avg : ℝ) (team_b_avg : ℝ) (total_avg : ℝ) (size_diff : ℕ) :
  team_a_avg = 75 →
  team_b_avg = 73 →
  total_avg = 73.5 →
  size_diff = 6 →
  ∃ (x : ℕ), x + size_diff = 9 ∧
    (team_a_avg * x + team_b_avg * (x + size_diff)) / (x + (x + size_diff)) = total_avg :=
by
  sorry

#check team_b_size

end NUMINAMATH_CALUDE_team_b_size_l3289_328938


namespace NUMINAMATH_CALUDE_shoe_repair_time_l3289_328950

theorem shoe_repair_time (heel_time shoe_count total_time : ℕ) (h1 : heel_time = 10) (h2 : shoe_count = 2) (h3 : total_time = 30) :
  (total_time - heel_time * shoe_count) / shoe_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_shoe_repair_time_l3289_328950


namespace NUMINAMATH_CALUDE_hayley_sticker_distribution_l3289_328933

def distribute_stickers (total_stickers : ℕ) (num_friends : ℕ) : ℕ :=
  total_stickers / num_friends

theorem hayley_sticker_distribution :
  let total_stickers : ℕ := 72
  let num_friends : ℕ := 9
  distribute_stickers total_stickers num_friends = 8 := by sorry

end NUMINAMATH_CALUDE_hayley_sticker_distribution_l3289_328933


namespace NUMINAMATH_CALUDE_negation_of_all_greater_than_sin_l3289_328982

theorem negation_of_all_greater_than_sin :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x₀ : ℝ, x₀ ≤ Real.sin x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_all_greater_than_sin_l3289_328982


namespace NUMINAMATH_CALUDE_total_students_in_classes_l3289_328908

theorem total_students_in_classes (class_a class_b : ℕ) : 
  (80 * class_a = 90 * (class_a - 8) + 20 * 8) →
  (70 * class_b = 85 * (class_b - 6) + 30 * 6) →
  class_a + class_b = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_students_in_classes_l3289_328908


namespace NUMINAMATH_CALUDE_cube_edge_length_proof_l3289_328910

/-- The edge length of a cube that, when fully immersed in a rectangular vessel
    with base dimensions 20 cm × 15 cm, causes a water level rise of 5.76 cm. -/
def cube_edge_length : ℝ := 12

/-- The base area of the rectangular vessel in square centimeters. -/
def vessel_base_area : ℝ := 20 * 15

/-- The rise in water level in centimeters when the cube is fully immersed. -/
def water_level_rise : ℝ := 5.76

/-- The volume of water displaced by the cube in cubic centimeters. -/
def displaced_volume : ℝ := vessel_base_area * water_level_rise

theorem cube_edge_length_proof :
  cube_edge_length ^ 3 = displaced_volume :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_length_proof_l3289_328910


namespace NUMINAMATH_CALUDE_second_chapter_page_difference_l3289_328921

/-- A book with three chapters -/
structure Book where
  chapter1_pages : ℕ
  chapter2_pages : ℕ
  chapter3_pages : ℕ

/-- The specific book described in the problem -/
def my_book : Book := {
  chapter1_pages := 35
  chapter2_pages := 18
  chapter3_pages := 3
}

/-- Theorem stating the difference in pages between the second and third chapters -/
theorem second_chapter_page_difference (b : Book := my_book) :
  b.chapter2_pages - b.chapter3_pages = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_page_difference_l3289_328921


namespace NUMINAMATH_CALUDE_parabola_equation_l3289_328927

/-- Given a parabola y² = 2mx (m ≠ 0) intersected by the line y = x - 4,
    if the length of the chord formed by this intersection is 6√2,
    then the equation of the parabola is either y² = (-4 + √34)x or y² = (-4 - √34)x. -/
theorem parabola_equation (m : ℝ) (h1 : m ≠ 0) :
  let f (x : ℝ) := 2 * m * x
  let g (x : ℝ) := x - 4
  let chord_length := (∃ x₁ x₂, x₁ ≠ x₂ ∧ f (g x₁) = (g x₁)^2 ∧ f (g x₂) = (g x₂)^2 ∧
    Real.sqrt ((x₁ - x₂)^2 + (g x₁ - g x₂)^2) = 6 * Real.sqrt 2)
  chord_length →
    (∀ x, f x = (-4 + Real.sqrt 34) * x) ∨ (∀ x, f x = (-4 - Real.sqrt 34) * x) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3289_328927


namespace NUMINAMATH_CALUDE_no_horizontal_asymptote_l3289_328995

noncomputable def f (x : ℝ) : ℝ :=
  (18 * x^5 + 12 * x^4 + 4 * x^3 + 9 * x^2 + 5 * x + 3) /
  (3 * x^4 + 2 * x^3 + 8 * x^2 + 3 * x + 1)

theorem no_horizontal_asymptote :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ N, ∀ x > N, |f x - L| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_no_horizontal_asymptote_l3289_328995


namespace NUMINAMATH_CALUDE_complex_multiplication_l3289_328987

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The property that i^2 = -1 -/
axiom i_squared : i ^ 2 = -1

/-- Theorem stating that (1+2i)(2+i) = 5i -/
theorem complex_multiplication : (1 + 2 * i) * (2 + i) = 5 * i := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3289_328987
