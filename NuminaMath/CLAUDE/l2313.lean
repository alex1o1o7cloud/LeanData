import Mathlib

namespace NUMINAMATH_CALUDE_goats_sold_proof_l2313_231392

/-- Represents the number of animals sold -/
def total_animals : ℕ := 80

/-- Represents the total reduction in legs -/
def total_leg_reduction : ℕ := 200

/-- Represents the number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- Represents the number of legs a goat has -/
def goat_legs : ℕ := 4

/-- Represents the number of goats sold -/
def goats_sold : ℕ := 20

/-- Represents the number of chickens sold -/
def chickens_sold : ℕ := total_animals - goats_sold

theorem goats_sold_proof :
  goats_sold * goat_legs + chickens_sold * chicken_legs = total_leg_reduction ∧
  goats_sold + chickens_sold = total_animals :=
by sorry

end NUMINAMATH_CALUDE_goats_sold_proof_l2313_231392


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2313_231350

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_ms : ℝ := train_speed * 1000 / 3600
  speed_ms * crossing_time

/-- Proof that a train with speed 48 km/hr crossing a pole in 9 seconds has a length of approximately 119.97 meters -/
theorem train_length_proof :
  ∃ ε > 0, |train_length 48 9 - 119.97| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2313_231350


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l2313_231345

/-- Given a shirt with an initial price and a required price increase to achieve
    a 100% markup, calculate the initial markup percentage. -/
theorem initial_markup_percentage
  (initial_price : ℝ)
  (price_increase : ℝ)
  (h1 : initial_price = 27)
  (h2 : price_increase = 3)
  (h3 : initial_price + price_increase = 2 * (initial_price - (initial_price - (initial_price / (1 + 1))))): 
  (initial_price - (initial_price / (1 + 1))) / (initial_price / (1 + 1)) * 100 = 80 :=
by sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l2313_231345


namespace NUMINAMATH_CALUDE_women_in_room_l2313_231351

theorem women_in_room (x : ℕ) (h1 : 4 * x + 2 = 14) : 2 * (5 * x - 3) = 24 := by
  sorry

end NUMINAMATH_CALUDE_women_in_room_l2313_231351


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2313_231379

/-- Given a square with side length 2 and four congruent isosceles triangles
    constructed on its sides, if the sum of the triangles' areas equals
    the square's area, then each triangle's congruent side length is √2. -/
theorem isosceles_triangle_side_length :
  let square_side : ℝ := 2
  let square_area : ℝ := square_side ^ 2
  let triangle_area : ℝ := square_area / 4
  let triangle_base : ℝ := square_side
  let triangle_height : ℝ := 2 * triangle_area / triangle_base
  let triangle_side : ℝ := Real.sqrt (triangle_height ^ 2 + (triangle_base / 2) ^ 2)
  triangle_side = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l2313_231379


namespace NUMINAMATH_CALUDE_paint_cans_theorem_l2313_231354

/-- Represents the number of rooms that can be painted with one can of paint -/
def rooms_per_can : ℚ :=
  (36 - 28) / 4

/-- The number of cans used to paint 28 rooms -/
def cans_used : ℚ := 28 / rooms_per_can

theorem paint_cans_theorem :
  cans_used = 14 := by sorry

end NUMINAMATH_CALUDE_paint_cans_theorem_l2313_231354


namespace NUMINAMATH_CALUDE_money_made_washing_cars_l2313_231365

def initial_amount : ℕ := 74
def current_amount : ℕ := 86

theorem money_made_washing_cars :
  current_amount - initial_amount = 12 :=
by sorry

end NUMINAMATH_CALUDE_money_made_washing_cars_l2313_231365


namespace NUMINAMATH_CALUDE_new_shoes_duration_proof_l2313_231339

/-- The duration of new shoes in years -/
def new_shoes_duration : ℝ := 2

/-- The cost of repairing used shoes -/
def used_shoes_repair_cost : ℝ := 10.5

/-- The duration of used shoes after repair in years -/
def used_shoes_duration : ℝ := 1

/-- The cost of new shoes -/
def new_shoes_cost : ℝ := 30

/-- The percentage increase in average cost per year of new shoes compared to repaired used shoes -/
def cost_increase_percentage : ℝ := 42.857142857142854

theorem new_shoes_duration_proof :
  new_shoes_duration = new_shoes_cost / (used_shoes_repair_cost * (1 + cost_increase_percentage / 100)) :=
by sorry

end NUMINAMATH_CALUDE_new_shoes_duration_proof_l2313_231339


namespace NUMINAMATH_CALUDE_eggs_left_over_l2313_231393

def total_eggs : ℕ := 114
def carton_size : ℕ := 15

theorem eggs_left_over : total_eggs % carton_size = 9 := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_over_l2313_231393


namespace NUMINAMATH_CALUDE_greg_extra_books_l2313_231331

theorem greg_extra_books (megan_books kelcie_books greg_books : ℕ) : 
  megan_books = 32 →
  kelcie_books = megan_books / 4 →
  greg_books > 2 * kelcie_books →
  megan_books + kelcie_books + greg_books = 65 →
  greg_books - 2 * kelcie_books = 9 := by
sorry

end NUMINAMATH_CALUDE_greg_extra_books_l2313_231331


namespace NUMINAMATH_CALUDE_stopped_clock_more_accurate_l2313_231364

/-- Represents the frequency of showing correct time for a clock --/
structure ClockAccuracy where
  correct_times_per_day : ℚ

/-- A clock that is one minute slow --/
def slow_clock : ClockAccuracy where
  correct_times_per_day := 1 / 720

/-- A stopped clock --/
def stopped_clock : ClockAccuracy where
  correct_times_per_day := 2

theorem stopped_clock_more_accurate : 
  stopped_clock.correct_times_per_day > slow_clock.correct_times_per_day := by
  sorry

#check stopped_clock_more_accurate

end NUMINAMATH_CALUDE_stopped_clock_more_accurate_l2313_231364


namespace NUMINAMATH_CALUDE_binomial_expression_is_integer_l2313_231346

theorem binomial_expression_is_integer (m n : ℕ) : 
  ∃ k : ℤ, k = (m.factorial * (2*n + 2*m).factorial) / 
              ((2*m).factorial * n.factorial * (n+m).factorial) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expression_is_integer_l2313_231346


namespace NUMINAMATH_CALUDE_nested_root_simplification_l2313_231333

theorem nested_root_simplification (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x^2 * Real.sqrt (x^3 * Real.sqrt (x^4))) = (x^9)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_nested_root_simplification_l2313_231333


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2313_231353

/-- The area of the shaded region in a square with circles at its vertices -/
theorem shaded_area_square_with_circles (square_side : ℝ) (circle_radius : ℝ) 
  (h1 : square_side = 8) (h2 : circle_radius = 3) : 
  ∃ (shaded_area : ℝ), shaded_area = square_side^2 - 12 * Real.sqrt 7 - 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2313_231353


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2313_231374

theorem solution_set_inequality (m : ℝ) (h : m < 5) :
  {x : ℝ | m * x > 6 * x + 3} = {x : ℝ | x < 3 / (m - 6)} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2313_231374


namespace NUMINAMATH_CALUDE_book_price_change_book_price_problem_l2313_231399

theorem book_price_change (initial_price : ℝ) 
  (decrease_percent : ℝ) (increase_percent : ℝ) : ℝ :=
  let price_after_decrease := initial_price * (1 - decrease_percent)
  let final_price := price_after_decrease * (1 + increase_percent)
  final_price

theorem book_price_problem : 
  book_price_change 400 0.15 0.40 = 476 := by
  sorry

end NUMINAMATH_CALUDE_book_price_change_book_price_problem_l2313_231399


namespace NUMINAMATH_CALUDE_equation_solution_l2313_231397

theorem equation_solution :
  ∃ (x : ℝ), x^2 - 4 ≠ 0 ∧ (x - 2) / (x + 2) + 4 / (x^2 - 4) = 1 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2313_231397


namespace NUMINAMATH_CALUDE_arcade_tickets_l2313_231308

theorem arcade_tickets (whack_a_mole skee_ball spent remaining : ℕ) :
  skee_ball = 25 ∧ spent = 7 ∧ remaining = 50 →
  whack_a_mole + skee_ball = remaining + spent →
  whack_a_mole = 7 := by
sorry

end NUMINAMATH_CALUDE_arcade_tickets_l2313_231308


namespace NUMINAMATH_CALUDE_farmer_initial_apples_l2313_231358

/-- The number of apples the farmer gave away -/
def apples_given_away : ℕ := 88

/-- The number of apples the farmer has left -/
def apples_left : ℕ := 39

/-- The initial number of apples the farmer had -/
def initial_apples : ℕ := apples_given_away + apples_left

theorem farmer_initial_apples : initial_apples = 127 := by
  sorry

end NUMINAMATH_CALUDE_farmer_initial_apples_l2313_231358


namespace NUMINAMATH_CALUDE_origin_and_point_opposite_sides_l2313_231343

/-- Determines if two points are on opposite sides of a line -/
def areOnOppositeSides (x1 y1 x2 y2 a b c : ℝ) : Prop :=
  (a * x1 + b * y1 + c) * (a * x2 + b * y2 + c) < 0

theorem origin_and_point_opposite_sides :
  areOnOppositeSides 0 0 2 1 (-6) 2 1 := by
  sorry

end NUMINAMATH_CALUDE_origin_and_point_opposite_sides_l2313_231343


namespace NUMINAMATH_CALUDE_eliza_height_difference_l2313_231347

/-- Given the heights of Eliza and her siblings, prove that Eliza is 2 inches shorter than the tallest sibling -/
theorem eliza_height_difference (total_height : ℕ) (sibling1_height sibling2_height sibling3_height eliza_height : ℕ) :
  total_height = 330 ∧
  sibling1_height = 66 ∧
  sibling2_height = 66 ∧
  sibling3_height = 60 ∧
  eliza_height = 68 →
  ∃ (tallest_sibling_height : ℕ),
    tallest_sibling_height + sibling1_height + sibling2_height + sibling3_height + eliza_height = total_height ∧
    tallest_sibling_height - eliza_height = 2 :=
by sorry

end NUMINAMATH_CALUDE_eliza_height_difference_l2313_231347


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2313_231396

/-- Given an arithmetic sequence {a_n} where a_2 + a_3 + a_10 + a_11 = 48, prove that a_6 + a_7 = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h2 : a 2 + a 3 + a 10 + a 11 = 48) : a 6 + a 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2313_231396


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l2313_231380

def initial_amount : ℚ := 150.75

def sandwich_fraction : ℚ := 3/10
def museum_fraction : ℚ := 1/4
def book_fraction : ℚ := 1/8
def coffee_percentage : ℚ := 2.5/100

def remaining_amount : ℚ := initial_amount - (
  initial_amount * sandwich_fraction +
  initial_amount * museum_fraction +
  initial_amount * book_fraction +
  initial_amount * coffee_percentage
)

theorem jennifer_remaining_money :
  remaining_amount = 45.225 := by sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l2313_231380


namespace NUMINAMATH_CALUDE_complete_square_formula_l2313_231361

theorem complete_square_formula (x : ℝ) : x^2 + 4*x + 4 = (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_formula_l2313_231361


namespace NUMINAMATH_CALUDE_fraction_simplification_l2313_231325

theorem fraction_simplification :
  (1 : ℝ) / (1 + Real.sqrt 3) * (1 / (1 - Real.sqrt 3)) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2313_231325


namespace NUMINAMATH_CALUDE_school_bought_fifty_marker_cartons_l2313_231313

/-- Represents the purchase of school supplies -/
structure SchoolSupplies where
  pencil_cartons : ℕ
  pencil_boxes_per_carton : ℕ
  pencil_box_cost : ℕ
  marker_carton_cost : ℕ
  total_spent : ℕ

/-- Calculates the number of marker cartons bought -/
def marker_cartons_bought (supplies : SchoolSupplies) : ℕ :=
  (supplies.total_spent - supplies.pencil_cartons * supplies.pencil_boxes_per_carton * supplies.pencil_box_cost) / supplies.marker_carton_cost

/-- Theorem stating that the school bought 50 cartons of markers -/
theorem school_bought_fifty_marker_cartons :
  let supplies : SchoolSupplies := {
    pencil_cartons := 20,
    pencil_boxes_per_carton := 10,
    pencil_box_cost := 2,
    marker_carton_cost := 4,
    total_spent := 600
  }
  marker_cartons_bought supplies = 50 := by
  sorry


end NUMINAMATH_CALUDE_school_bought_fifty_marker_cartons_l2313_231313


namespace NUMINAMATH_CALUDE_cylinder_height_ratio_l2313_231326

theorem cylinder_height_ratio (h : ℝ) (h_pos : h > 0) : 
  ∃ (H : ℝ), H = (14 / 15) * h ∧ 
  (7 / 8) * π * h = (3 / 5) * π * ((5 / 4) ^ 2) * H :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_ratio_l2313_231326


namespace NUMINAMATH_CALUDE_rolling_circle_arc_angle_range_l2313_231373

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the angle between three points -/
def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the arc angle given three points on a circle -/
def arcAngle (circle : Circle) (p1 p2 p3 : Point) : ℝ := sorry

/-- Main theorem -/
theorem rolling_circle_arc_angle_range 
  (triangle : Triangle) 
  (isRightTriangle : angle triangle.A triangle.B triangle.C = 90)
  (has30DegreeAngle : angle triangle.A triangle.C triangle.B = 30)
  (circle : Circle)
  (circleRadiusHalfBC : circle.radius = distance triangle.B triangle.C / 2)
  (T : Point → Point) -- T is a function of the circle's position
  (M : Point → Point) -- M is a function of the circle's position
  (N : Point → Point) -- N is a function of the circle's position
  (circleTangentToAB : ∀ (circlePos : Point), distance (T circlePos) circlePos = circle.radius)
  (circleIntersectsAC : ∀ (circlePos : Point), (M circlePos).x = triangle.A.x ∨ (M circlePos).y = triangle.A.y)
  (circleIntersectsBC : ∀ (circlePos : Point), distance (N circlePos) triangle.B = distance (N circlePos) triangle.C) :
  ∃ (circlePos1 circlePos2 : Point),
    arcAngle circle (M circlePos1) (T circlePos1) (N circlePos1) = 180 ∧
    arcAngle circle (M circlePos2) (T circlePos2) (N circlePos2) = 0 ∧
    ∀ (circlePos : Point),
      0 ≤ arcAngle circle (M circlePos) (T circlePos) (N circlePos) ∧
      arcAngle circle (M circlePos) (T circlePos) (N circlePos) ≤ 180 :=
sorry

end NUMINAMATH_CALUDE_rolling_circle_arc_angle_range_l2313_231373


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2313_231362

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2313_231362


namespace NUMINAMATH_CALUDE_part_one_part_two_l2313_231338

-- Define the new operation ※
def star (a b : ℝ) : ℝ := a^2 - b^2

-- Theorem for part 1
theorem part_one : star 2 (-4) = -12 := by sorry

-- Theorem for part 2
theorem part_two : ∀ x : ℝ, star (x + 5) 3 = 0 ↔ x = -8 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2313_231338


namespace NUMINAMATH_CALUDE_handshake_problem_l2313_231363

theorem handshake_problem (n : ℕ) : n * (n - 1) / 2 = 78 → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_handshake_problem_l2313_231363


namespace NUMINAMATH_CALUDE_seven_b_value_l2313_231337

theorem seven_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) : 7 * b = 168 / 11 := by
  sorry

end NUMINAMATH_CALUDE_seven_b_value_l2313_231337


namespace NUMINAMATH_CALUDE_paint_for_sun_l2313_231356

/-- The amount of paint left for the sun, given Mary's and Mike's usage --/
def paint_left_for_sun (mary_paint : ℝ) (mike_extra_paint : ℝ) (total_paint : ℝ) : ℝ :=
  total_paint - (mary_paint + (mary_paint + mike_extra_paint))

/-- Theorem stating the amount of paint left for the sun --/
theorem paint_for_sun :
  paint_left_for_sun 3 2 13 = 5 := by
  sorry

end NUMINAMATH_CALUDE_paint_for_sun_l2313_231356


namespace NUMINAMATH_CALUDE_fraction_simplification_l2313_231349

theorem fraction_simplification (x y : ℚ) (hx : x = 4/6) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 9/20 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2313_231349


namespace NUMINAMATH_CALUDE_crayons_lost_or_given_away_l2313_231360

theorem crayons_lost_or_given_away (start_crayons end_crayons : ℕ) 
  (h1 : start_crayons = 253)
  (h2 : end_crayons = 183) :
  start_crayons - end_crayons = 70 := by
  sorry

end NUMINAMATH_CALUDE_crayons_lost_or_given_away_l2313_231360


namespace NUMINAMATH_CALUDE_limit_example_l2313_231382

/-- The limit of (5x^2 - 4x - 1)/(x - 1) as x approaches 1 is 6 -/
theorem limit_example : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 1| → |x - 1| < δ → 
    |(5*x^2 - 4*x - 1)/(x - 1) - 6| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_example_l2313_231382


namespace NUMINAMATH_CALUDE_equal_representations_l2313_231394

/-- Represents the number of ways to write a positive integer as a product of powers of primes,
    where each factor is greater than or equal to the previous one. -/
def primeRepresentations (n : ℕ+) : ℕ := sorry

/-- Represents the number of ways to write a positive integer as a product of integers greater than 1,
    where each factor is divisible by all previous factors. -/
def divisibilityRepresentations (n : ℕ+) : ℕ := sorry

/-- Theorem stating that for any positive integer n, the number of prime representations
    is equal to the number of divisibility representations. -/
theorem equal_representations (n : ℕ+) : primeRepresentations n = divisibilityRepresentations n := by
  sorry

end NUMINAMATH_CALUDE_equal_representations_l2313_231394


namespace NUMINAMATH_CALUDE_conference_exchanges_l2313_231330

/-- The number of business card exchanges in a conference -/
def businessCardExchanges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a conference of 10 people, where each person exchanges 
    business cards with every other person exactly once, 
    the total number of exchanges is 45 -/
theorem conference_exchanges : businessCardExchanges 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_conference_exchanges_l2313_231330


namespace NUMINAMATH_CALUDE_lunch_break_duration_l2313_231310

-- Define the painting rates and lunch break duration
variable (p : ℝ) -- Paula's painting rate (building/hour)
variable (h : ℝ) -- Combined rate of two helpers (building/hour)
variable (L : ℝ) -- Lunch break duration (hours)

-- Define the equations based on the given conditions
def monday_equation : Prop := (9 - L) * (p + h) = 0.4
def tuesday_equation : Prop := (7 - L) * h = 0.3
def wednesday_equation : Prop := (12 - L) * p = 0.3

-- Theorem statement
theorem lunch_break_duration 
  (eq1 : monday_equation p h L)
  (eq2 : tuesday_equation h L)
  (eq3 : wednesday_equation p L) :
  L = 0.5 := by sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l2313_231310


namespace NUMINAMATH_CALUDE_inequality_proof_l2313_231390

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  (1 / (b*c + a + 1/a)) + (1 / (a*c + b + 1/b)) + (1 / (a*b + c + 1/c)) ≤ 27/31 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2313_231390


namespace NUMINAMATH_CALUDE_value_of_expression_l2313_231302

theorem value_of_expression (x : ℝ) (h : x = 3) : 5 - 2 * x^2 = -13 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2313_231302


namespace NUMINAMATH_CALUDE_three_digit_factorial_sum_l2313_231335

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_of_digit_factorials (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  factorial hundreds + factorial tens + factorial units

theorem three_digit_factorial_sum :
  ∃ (n : Nat), 100 ≤ n ∧ n < 1000 ∧ n / 100 = 2 ∧ n = sum_of_digit_factorials n :=
by
  sorry

end NUMINAMATH_CALUDE_three_digit_factorial_sum_l2313_231335


namespace NUMINAMATH_CALUDE_earnings_difference_l2313_231344

/-- Mateo's hourly rate in dollars -/
def mateo_hourly_rate : ℕ := 20

/-- Sydney's daily rate in dollars -/
def sydney_daily_rate : ℕ := 400

/-- Number of hours in a week -/
def hours_per_week : ℕ := 24 * 7

/-- Number of days in a week -/
def days_per_week : ℕ := 7

/-- Mateo's total earnings for one week in dollars -/
def mateo_earnings : ℕ := mateo_hourly_rate * hours_per_week

/-- Sydney's total earnings for one week in dollars -/
def sydney_earnings : ℕ := sydney_daily_rate * days_per_week

theorem earnings_difference : mateo_earnings - sydney_earnings = 560 := by
  sorry

end NUMINAMATH_CALUDE_earnings_difference_l2313_231344


namespace NUMINAMATH_CALUDE_largest_value_l2313_231309

theorem largest_value (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) :
  b > 1/2 ∧ b > 2*a*b ∧ b > a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_l2313_231309


namespace NUMINAMATH_CALUDE_strengthened_erdos_mordell_inequality_l2313_231324

theorem strengthened_erdos_mordell_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * area + (a - b)^2 + (b - c)^2 + (c - a)^2 := by
sorry

end NUMINAMATH_CALUDE_strengthened_erdos_mordell_inequality_l2313_231324


namespace NUMINAMATH_CALUDE_cylinder_base_area_l2313_231320

theorem cylinder_base_area (S : ℝ) (h : S > 0) :
  let cross_section_area := 4 * S
  let cross_section_is_square := true
  let base_area := π * S
  cross_section_is_square ∧ cross_section_area = 4 * S → base_area = π * S :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_base_area_l2313_231320


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l2313_231378

/-- Proves that the breadth of a rectangular plot is 14 meters, given that its length is thrice its breadth and its area is 588 square meters. -/
theorem rectangular_plot_breadth (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth → 
  area = length * breadth → 
  area = 588 → 
  breadth = 14 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l2313_231378


namespace NUMINAMATH_CALUDE_zoe_coloring_books_l2313_231389

/-- The number of pictures left to color given the initial number of pictures in two books and the number of pictures already colored. -/
def pictures_left_to_color (book1_pictures : ℕ) (book2_pictures : ℕ) (colored_pictures : ℕ) : ℕ :=
  book1_pictures + book2_pictures - colored_pictures

/-- Theorem stating that given two coloring books with 44 pictures each, and 20 pictures already colored, the number of pictures left to color is 68. -/
theorem zoe_coloring_books : pictures_left_to_color 44 44 20 = 68 := by
  sorry


end NUMINAMATH_CALUDE_zoe_coloring_books_l2313_231389


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2313_231318

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {2, 5, 8}
def B : Set Nat := {1, 3, 5, 7}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {1, 3, 7} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2313_231318


namespace NUMINAMATH_CALUDE_scientific_notation_35_million_l2313_231384

theorem scientific_notation_35_million :
  35000000 = 3.5 * (10 ^ 7) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_35_million_l2313_231384


namespace NUMINAMATH_CALUDE_janet_shampoo_duration_l2313_231328

/-- Calculates the number of days Janet's shampoo will last -/
def shampoo_duration (rose_shampoo : Rat) (jasmine_shampoo : Rat) (usage_per_day : Rat) : Nat :=
  Nat.floor ((rose_shampoo + jasmine_shampoo) / usage_per_day)

/-- Theorem: Janet's shampoo will last for 7 days -/
theorem janet_shampoo_duration :
  shampoo_duration (1/3) (1/4) (1/12) = 7 := by
  sorry

end NUMINAMATH_CALUDE_janet_shampoo_duration_l2313_231328


namespace NUMINAMATH_CALUDE_point_coordinates_product_l2313_231368

theorem point_coordinates_product (y₁ y₂ : ℝ) : 
  (((4 : ℝ) - 7)^2 + (y₁ - (-3))^2 = 13^2) →
  (((4 : ℝ) - 7)^2 + (y₂ - (-3))^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -151 := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_product_l2313_231368


namespace NUMINAMATH_CALUDE_girls_in_class_l2313_231367

theorem girls_in_class (total : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ) (h_total : total = 35) (h_ratio : ratio_girls = 3 ∧ ratio_boys = 4) :
  ∃ (girls : ℕ), girls * ratio_boys = (total - girls) * ratio_girls ∧ girls = 15 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l2313_231367


namespace NUMINAMATH_CALUDE_boys_share_l2313_231348

theorem boys_share (total_amount : ℕ) (total_children : ℕ) (num_boys : ℕ) (amount_per_girl : ℕ) 
  (h1 : total_amount = 460)
  (h2 : total_children = 41)
  (h3 : num_boys = 33)
  (h4 : amount_per_girl = 8) :
  (total_amount - (total_children - num_boys) * amount_per_girl) / num_boys = 12 := by
  sorry

end NUMINAMATH_CALUDE_boys_share_l2313_231348


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2313_231371

def P : Set ℤ := {x | |x - 1| < 2}
def Q : Set ℤ := {x | -1 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2313_231371


namespace NUMINAMATH_CALUDE_sin_15_cos_15_equals_quarter_l2313_231395

theorem sin_15_cos_15_equals_quarter :
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_equals_quarter_l2313_231395


namespace NUMINAMATH_CALUDE_inversion_preserves_angle_l2313_231342

-- Define a type for geometric objects (circles or lines)
inductive GeometricObject
  | Circle : ℝ → ℝ → ℝ → GeometricObject  -- center_x, center_y, radius
  | Line : ℝ → ℝ → ℝ → GeometricObject    -- a, b, c for ax + by + c = 0

-- Define the inversion transformation
def inversion (center : ℝ × ℝ) (k : ℝ) (obj : GeometricObject) : GeometricObject :=
  sorry

-- Define the angle between two geometric objects
def angle_between (obj1 obj2 : GeometricObject) : ℝ :=
  sorry

-- State the theorem
theorem inversion_preserves_angle (center : ℝ × ℝ) (k : ℝ) (obj1 obj2 : GeometricObject) :
  angle_between obj1 obj2 = angle_between (inversion center k obj1) (inversion center k obj2) :=
  sorry

end NUMINAMATH_CALUDE_inversion_preserves_angle_l2313_231342


namespace NUMINAMATH_CALUDE_problem_proof_l2313_231319

theorem problem_proof : (-1)^2019 + (Real.pi - 3.14)^0 - Real.sqrt 16 + 2 * Real.sin (30 * π / 180) = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2313_231319


namespace NUMINAMATH_CALUDE_sqrt_meaningful_l2313_231372

theorem sqrt_meaningful (x : ℝ) : (∃ y : ℝ, y^2 = x - 2) → x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_l2313_231372


namespace NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l2313_231383

theorem sqrt_mixed_number_simplification :
  Real.sqrt (7 + 9 / 16) = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_mixed_number_simplification_l2313_231383


namespace NUMINAMATH_CALUDE_min_value_implies_a_inequality_solution_l2313_231370

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 2|

-- Theorem for part (i)
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f x a ≥ 2) ∧ (∃ x, f x a = 2) → a = 0 ∨ a = -4 := by sorry

-- Theorem for part (ii)
theorem inequality_solution (x : ℝ) :
  f x 2 ≤ 6 ↔ x ∈ Set.Icc (-3) 3 := by sorry

-- Note: Set.Icc represents a closed interval [a, b]

end NUMINAMATH_CALUDE_min_value_implies_a_inequality_solution_l2313_231370


namespace NUMINAMATH_CALUDE_trajectory_equation_l2313_231391

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

-- Define the property of being tangent internally to C₁ and tangent to C₂
def is_tangent_to_circles (x y : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    (∀ p q : ℝ, C₁ p q → (x - p)^2 + (y - q)^2 = (r - 1)^2) ∧
    (∀ p q : ℝ, C₂ p q → (x - p)^2 + (y - q)^2 = (r + 9)^2)

-- Theorem statement
theorem trajectory_equation :
  ∀ x y : ℝ, is_tangent_to_circles x y ↔ x^2/16 + y^2/7 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l2313_231391


namespace NUMINAMATH_CALUDE_vector_AD_in_triangle_l2313_231304

-- Define the triangle ABC and point D
variable (A B C D : Euclidean_space ℝ (Fin 2))

-- Define the condition that D is on side BC
variable (h1 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = t • B + (1 - t) • C)

-- Define the condition that BC = 3 * BD
variable (h2 : C - B = 3 • (D - B))

-- State the theorem
theorem vector_AD_in_triangle :
  D - A = (2/3) • (B - A) + (1/3) • (C - A) :=
sorry

end NUMINAMATH_CALUDE_vector_AD_in_triangle_l2313_231304


namespace NUMINAMATH_CALUDE_value_of_x_l2313_231340

theorem value_of_x (x y : ℚ) (h1 : x / y = 7 / 3) (h2 : y = 21) : x = 49 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l2313_231340


namespace NUMINAMATH_CALUDE_ellipse_equation_l2313_231300

-- Define the ellipse
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  majorAxisLength : ℝ

-- Define the standard form of an ellipse equation
def StandardEllipseEquation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem ellipse_equation (e : Ellipse) : 
  e.foci = ((-2, 0), (2, 0)) ∧ e.majorAxisLength = 10 →
  ∀ x y : ℝ, StandardEllipseEquation 25 21 x y :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2313_231300


namespace NUMINAMATH_CALUDE_complement_B_union_A_equals_open_interval_l2313_231323

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 3*x < 4}

-- Define set B
def B : Set ℝ := {x : ℝ | |x| ≥ 2}

-- Theorem statement
theorem complement_B_union_A_equals_open_interval :
  (Set.compl B) ∪ A = Set.Ioo (-2 : ℝ) 4 :=
sorry

end NUMINAMATH_CALUDE_complement_B_union_A_equals_open_interval_l2313_231323


namespace NUMINAMATH_CALUDE_total_cost_with_tip_l2313_231317

def hair_cost : ℝ := 50
def nail_cost : ℝ := 30
def tip_percentage : ℝ := 0.20

theorem total_cost_with_tip : 
  (hair_cost + nail_cost) * (1 + tip_percentage) = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_with_tip_l2313_231317


namespace NUMINAMATH_CALUDE_fraction_zero_at_five_l2313_231329

theorem fraction_zero_at_five (x : ℝ) : 
  (x - 5) / (6 * x - 12) = 0 ↔ x = 5 ∧ 6 * x - 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_at_five_l2313_231329


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_negative_two_satisfies_inequality_least_integer_is_negative_two_l2313_231385

theorem least_integer_satisfying_inequality :
  ∀ y : ℤ, (2 * y^2 + 2 * |y| + 7 < 25) → y ≥ -2 :=
by
  sorry

theorem negative_two_satisfies_inequality :
  2 * (-2)^2 + 2 * |-2| + 7 < 25 :=
by
  sorry

theorem least_integer_is_negative_two :
  ∃ x : ℤ, (2 * x^2 + 2 * |x| + 7 < 25) ∧ 
    (∀ y : ℤ, (2 * y^2 + 2 * |y| + 7 < 25) → y ≥ x) ∧
    x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_negative_two_satisfies_inequality_least_integer_is_negative_two_l2313_231385


namespace NUMINAMATH_CALUDE_A_on_x_axis_A_on_y_axis_l2313_231305

-- Define point A
def A (a : ℝ) : ℝ × ℝ := (a - 3, a^2 - 4)

-- Theorem for when A lies on the x-axis
theorem A_on_x_axis :
  ∃ a : ℝ, (A a).2 = 0 → (A a = (-1, 0) ∨ A a = (-5, 0)) :=
sorry

-- Theorem for when A lies on the y-axis
theorem A_on_y_axis :
  ∃ a : ℝ, (A a).1 = 0 → A a = (0, 5) :=
sorry

end NUMINAMATH_CALUDE_A_on_x_axis_A_on_y_axis_l2313_231305


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l2313_231352

theorem polar_to_cartesian :
  let r : ℝ := 4
  let θ : ℝ := 5 * Real.pi / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -2 * Real.sqrt 3 ∧ y = 2) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l2313_231352


namespace NUMINAMATH_CALUDE_cheese_pizzas_sold_l2313_231307

/-- The number of cheese pizzas sold by a pizza store on Friday -/
def cheese_pizzas (pepperoni bacon total : ℕ) : ℕ :=
  total - (pepperoni + bacon)

/-- Theorem stating the number of cheese pizzas sold -/
theorem cheese_pizzas_sold :
  cheese_pizzas 2 6 14 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cheese_pizzas_sold_l2313_231307


namespace NUMINAMATH_CALUDE_jerrys_pool_depth_l2313_231312

/-- Calculates the depth of Jerry's pool given water usage constraints -/
theorem jerrys_pool_depth :
  ∀ (total_water drinking_cooking shower_water showers pool_length pool_width : ℕ),
  total_water = 1000 →
  drinking_cooking = 100 →
  shower_water = 20 →
  showers = 15 →
  pool_length = 10 →
  pool_width = 10 →
  (total_water - (drinking_cooking + shower_water * showers)) / (pool_length * pool_width) = 6 := by
  sorry

#check jerrys_pool_depth

end NUMINAMATH_CALUDE_jerrys_pool_depth_l2313_231312


namespace NUMINAMATH_CALUDE_square_complex_real_iff_a_or_b_zero_l2313_231388

theorem square_complex_real_iff_a_or_b_zero (a b : ℝ) :
  let z : ℂ := Complex.mk a b
  (∃ (r : ℝ), z^2 = (r : ℂ)) ↔ a = 0 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_complex_real_iff_a_or_b_zero_l2313_231388


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2313_231369

theorem inequality_solution_set (a : ℝ) (ha : a > 0) :
  {x : ℝ | x^2 - (a + 1/a + 1)*x + a + 1/a < 0} = {x : ℝ | 1 < x ∧ x < a + 1/a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2313_231369


namespace NUMINAMATH_CALUDE_range_of_x_range_of_m_l2313_231311

-- Define propositions p and q
def p (x m : ℝ) : Prop := x^2 - 3*m*x + 2*m^2 ≤ 0
def q (x : ℝ) : Prop := (x + 2)^2 < 1

-- Part 1
theorem range_of_x (x : ℝ) :
  p x (-2) ∧ q x → x ∈ Set.Ioc (-3) (-2) :=
sorry

-- Part 2
theorem range_of_m (m : ℝ) :
  m < 0 ∧ (∀ x, q x ↔ ¬p x m) →
  m ∈ Set.Iic (-3) ∪ Set.Icc (-1/2) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_m_l2313_231311


namespace NUMINAMATH_CALUDE_car_overtakes_buses_l2313_231387

/-- The time interval between bus departures in minutes -/
def bus_interval : ℕ := 3

/-- The time taken by a bus to reach the city centre in minutes -/
def bus_travel_time : ℕ := 60

/-- The time taken by the car to reach the city centre in minutes -/
def car_travel_time : ℕ := 35

/-- The number of buses overtaken by the car -/
def buses_overtaken : ℕ := (bus_travel_time - car_travel_time) / bus_interval

theorem car_overtakes_buses :
  buses_overtaken = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_overtakes_buses_l2313_231387


namespace NUMINAMATH_CALUDE_opposite_of_negative_eight_l2313_231375

theorem opposite_of_negative_eight :
  (∃ x : ℤ, -8 + x = 0) ∧ (∀ y : ℤ, -8 + y = 0 → y = 8) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_eight_l2313_231375


namespace NUMINAMATH_CALUDE_sin_cube_identity_l2313_231334

theorem sin_cube_identity (θ : Real) : 
  Real.sin θ ^ 3 = (-1/4) * Real.sin (3*θ) + (3/4) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l2313_231334


namespace NUMINAMATH_CALUDE_projection_theorem_l2313_231301

def vector_projection (u v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  let dot_product := (u 0 * v 0 + u 1 * v 1)
  let norm_squared := (v 0 * v 0 + v 1 * v 1)
  fun i => (dot_product / norm_squared) * v i

def linear_transformation (v : Fin 2 → ℚ) : Fin 2 → ℚ :=
  vector_projection v (fun i => if i = 0 then 2 else -3)

theorem projection_theorem :
  let v : Fin 2 → ℚ := fun i => if i = 0 then 3 else -1
  let result := linear_transformation v
  result 0 = 18/13 ∧ result 1 = -27/13 := by
  sorry

end NUMINAMATH_CALUDE_projection_theorem_l2313_231301


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2313_231314

def M : Set ℝ := {x : ℝ | -5 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2313_231314


namespace NUMINAMATH_CALUDE_vet_formula_portions_l2313_231327

/-- Calculates the total number of formula portions needed for puppies -/
def total_formula_portions (num_puppies : ℕ) (num_days : ℕ) (feedings_per_day : ℕ) : ℕ :=
  num_puppies * num_days * feedings_per_day

/-- Theorem: The vet gave Sandra 105 portions of formula for her puppies -/
theorem vet_formula_portions : total_formula_portions 7 5 3 = 105 := by
  sorry

end NUMINAMATH_CALUDE_vet_formula_portions_l2313_231327


namespace NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l2313_231303

/-- An arithmetic progression where the sum of the first 15 terms is three times the sum of the first 8 terms -/
def ArithmeticProgression (a d : ℝ) : Prop :=
  let S : ℕ → ℝ := λ n => n / 2 * (2 * a + (n - 1) * d)
  S 15 = 3 * S 8

theorem ratio_first_term_to_common_difference 
  {a d : ℝ} (h : ArithmeticProgression a d) : 
  a / d = 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_first_term_to_common_difference_l2313_231303


namespace NUMINAMATH_CALUDE_nth_equation_and_specific_case_l2313_231359

theorem nth_equation_and_specific_case :
  (∀ n : ℕ, n > 0 → Real.sqrt (1 - (2 * n - 1) / (n * n)) = (n - 1) / n) ∧
  Real.sqrt (1 - 199 / 10000) = 99 / 100 :=
by sorry

end NUMINAMATH_CALUDE_nth_equation_and_specific_case_l2313_231359


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l2313_231306

/-- Given a principal amount with 5% interest rate for 2 years, 
    if the compound interest is 51.25, then the simple interest is 50 -/
theorem simple_interest_calculation (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 51.25 → P * 0.05 * 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l2313_231306


namespace NUMINAMATH_CALUDE_rectangle_circles_inequality_l2313_231357

/-- Given a rectangle ABCD with sides a and b, and circles with radii r1 and r2
    as defined, prove that r1 + r2 ≥ 5/8 * (a + b). -/
theorem rectangle_circles_inequality (a b r1 r2 : ℝ) 
    (ha : a > 0) (hb : b > 0) (hr1 : r1 > 0) (hr2 : r2 > 0)
    (h_r1 : r1 = b / 2 + a^2 / (8 * b))
    (h_r2 : r2 = a / 2 + b^2 / (8 * a)) :
    r1 + r2 ≥ 5/8 * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circles_inequality_l2313_231357


namespace NUMINAMATH_CALUDE_inequality_proof_l2313_231316

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2313_231316


namespace NUMINAMATH_CALUDE_cally_white_shirts_l2313_231355

/-- Represents the number of clothes of each type for a person -/
structure ClothesCount where
  white_shirts : ℕ
  colored_shirts : ℕ
  shorts : ℕ
  pants : ℕ

/-- Calculates the total number of clothes for a person -/
def total_clothes (c : ClothesCount) : ℕ :=
  c.white_shirts + c.colored_shirts + c.shorts + c.pants

/-- Theorem: Cally washed 10 white shirts -/
theorem cally_white_shirts :
  ∀ (cally_clothes : ClothesCount),
    cally_clothes.colored_shirts = 5 →
    cally_clothes.shorts = 7 →
    cally_clothes.pants = 6 →
    ∀ (danny_clothes : ClothesCount),
      danny_clothes.white_shirts = 6 →
      danny_clothes.colored_shirts = 8 →
      danny_clothes.shorts = 10 →
      danny_clothes.pants = 6 →
      total_clothes cally_clothes + total_clothes danny_clothes = 58 →
      cally_clothes.white_shirts = 10 :=
by sorry

end NUMINAMATH_CALUDE_cally_white_shirts_l2313_231355


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l2313_231332

theorem express_y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 2 + 2^p) (hy : y = 1 + 2^(-p)) : 
  y = (x - 1) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l2313_231332


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2313_231381

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 ∧ n ≥ 100 ∧ 17 ∣ n → n ≤ 986 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2313_231381


namespace NUMINAMATH_CALUDE_cardinality_of_B_l2313_231341

def A : Finset Int := {-3, -2, -1, 1, 2, 3, 4}

def f (a : Int) : Int := Int.natAbs a

def B : Finset Int := Finset.image f A

theorem cardinality_of_B : Finset.card B = 4 := by
  sorry

end NUMINAMATH_CALUDE_cardinality_of_B_l2313_231341


namespace NUMINAMATH_CALUDE_cloth_price_calculation_l2313_231321

theorem cloth_price_calculation (quantity : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) (total_cost : ℝ) :
  quantity = 9.25 →
  discount_rate = 0.12 →
  tax_rate = 0.05 →
  total_cost = 397.75 →
  ∃ P : ℝ, (quantity * (P - discount_rate * P)) * (1 + tax_rate) = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_cloth_price_calculation_l2313_231321


namespace NUMINAMATH_CALUDE_divisibility_condition_l2313_231376

theorem divisibility_condition (n : ℕ) (h : n ≥ 2) :
  (20^n + 19^n) % (20^(n-2) + 19^(n-2)) = 0 ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2313_231376


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2313_231386

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (3, -4)
  let b : ℝ × ℝ := (-1, m)
  are_parallel a b → m = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2313_231386


namespace NUMINAMATH_CALUDE_gcd_of_four_numbers_l2313_231398

theorem gcd_of_four_numbers : Nat.gcd 546 (Nat.gcd 1288 (Nat.gcd 3042 5535)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_four_numbers_l2313_231398


namespace NUMINAMATH_CALUDE_triangle_exterior_angle_l2313_231336

theorem triangle_exterior_angle (A B C : Real) (h1 : A + B + C = 180) 
  (h2 : A = B) (h3 : A = 40 ∨ B = 40 ∨ C = 40) : 
  180 - C = 80 ∨ 180 - C = 140 := by
  sorry

end NUMINAMATH_CALUDE_triangle_exterior_angle_l2313_231336


namespace NUMINAMATH_CALUDE_correct_book_arrangements_l2313_231377

/-- The number of ways to arrange 11 books (3 Arabic, 4 German, 4 Spanish) on a shelf, keeping the Arabic books together -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let german_books : ℕ := 4
  let spanish_books : ℕ := 4
  let arabic_unit : ℕ := 1
  let total_units : ℕ := arabic_unit + german_books + spanish_books
  (Nat.factorial total_units) * (Nat.factorial arabic_books)

theorem correct_book_arrangements :
  book_arrangements = 2177280 :=
by sorry

end NUMINAMATH_CALUDE_correct_book_arrangements_l2313_231377


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2313_231366

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 6 + x) ↔ x ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2313_231366


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l2313_231322

theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 - 1 ∧ p.2^2 = 4 * p.1) → k = 0 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l2313_231322


namespace NUMINAMATH_CALUDE_sets_equality_l2313_231315

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem sets_equality : M = N := by sorry

end NUMINAMATH_CALUDE_sets_equality_l2313_231315
