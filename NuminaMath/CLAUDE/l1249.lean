import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_identity_l1249_124903

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sin θ)^4 / a + (Real.cos θ)^4 / b = 1 / (a + b) →
  (Real.sin θ)^8 / a^3 + (Real.cos θ)^8 / b^3 = 1 / (a + b)^3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1249_124903


namespace NUMINAMATH_CALUDE_red_numbers_structure_l1249_124941

-- Define the color type
inductive Color
| White
| Red

-- Define the coloring function
def coloring : ℕ → Color := sorry

-- Define properties of the coloring
axiom exists_white : ∃ n : ℕ, coloring n = Color.White
axiom exists_red : ∃ n : ℕ, coloring n = Color.Red
axiom sum_white_red_is_white :
  ∀ w r : ℕ, coloring w = Color.White → coloring r = Color.Red →
  coloring (w + r) = Color.White
axiom product_white_red_is_red :
  ∀ w r : ℕ, coloring w = Color.White → coloring r = Color.Red →
  coloring (w * r) = Color.Red

-- Define the set of red numbers
def RedNumbers : Set ℕ := {n : ℕ | coloring n = Color.Red}

-- State the theorem
theorem red_numbers_structure :
  ∃ r₀ : ℕ, r₀ > 0 ∧ r₀ ∈ RedNumbers ∧
  ∀ n : ℕ, n ∈ RedNumbers ↔ ∃ k : ℕ, n = k * r₀ :=
sorry

end NUMINAMATH_CALUDE_red_numbers_structure_l1249_124941


namespace NUMINAMATH_CALUDE_prime_squared_plus_two_prime_l1249_124989

theorem prime_squared_plus_two_prime (p : ℕ) : 
  Prime p → Prime (p^2 + 2) → p = 3 := by sorry

end NUMINAMATH_CALUDE_prime_squared_plus_two_prime_l1249_124989


namespace NUMINAMATH_CALUDE_frank_candy_bags_l1249_124940

/-- The number of bags Frank used to store his candy -/
def num_bags (total_candy : ℕ) (candy_per_bag : ℕ) : ℕ :=
  total_candy / candy_per_bag

/-- Theorem: Frank used 26 bags to store his candy -/
theorem frank_candy_bags : num_bags 858 33 = 26 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_bags_l1249_124940


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l1249_124910

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l1249_124910


namespace NUMINAMATH_CALUDE_exists_sequence_equal_one_l1249_124977

/-- Represents a mathematical operation --/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide

/-- Evaluates the result of applying operations to the given sequence of digits --/
def evaluate (digits : List Nat) (ops : List Operation) : Option Rat :=
  sorry

/-- Theorem stating that there exists a sequence of operations that results in 1 --/
theorem exists_sequence_equal_one :
  ∃ (ops : List Operation),
    evaluate [1, 2, 3, 4, 5, 6, 7, 8] ops = some 1 :=
  sorry

end NUMINAMATH_CALUDE_exists_sequence_equal_one_l1249_124977


namespace NUMINAMATH_CALUDE_square_function_property_l1249_124962

def is_multiplicative (f : ℕ → ℕ) : Prop :=
  ∀ m n, f (m * n) = f m * f n

theorem square_function_property (f : ℕ → ℕ) 
  (h_mult : is_multiplicative f)
  (h_four : f 4 = 4)
  (h_sum_squares : ∀ m n : ℕ, f (m^2 + n^2) = f (m^2) + f (n^2)) :
  ∀ m : ℕ, m > 0 → f (m^2) = m^2 :=
sorry

end NUMINAMATH_CALUDE_square_function_property_l1249_124962


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1249_124948

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {x | x^2 ≥ 2*x}

theorem intersection_of_M_and_N : M ∩ N = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1249_124948


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l1249_124901

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧ 
  (∃ x : ℝ, x > 0 ∧ ¬(x > 1)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_x_gt_zero_l1249_124901


namespace NUMINAMATH_CALUDE_smallest_covering_l1249_124957

/-- A rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.width * r.height

/-- A configuration of rectangles covering a larger rectangle -/
structure Configuration where
  covering : Rectangle
  tiles : List Rectangle

/-- The total area covered by a list of rectangles -/
def total_area (tiles : List Rectangle) : ℕ := tiles.foldl (fun acc r => acc + r.area) 0

/-- A valid configuration has no gaps or overhangs -/
def Configuration.valid (c : Configuration) : Prop :=
  c.covering.area = total_area c.tiles

/-- The smallest valid configuration for covering with 3x4 rectangles -/
def smallest_valid_configuration : Configuration :=
  { covering := { width := 6, height := 8 }
  , tiles := List.replicate 4 { width := 3, height := 4 } }

theorem smallest_covering :
  smallest_valid_configuration.valid ∧
  (∀ c : Configuration, c.valid → c.covering.area ≥ smallest_valid_configuration.covering.area) ∧
  smallest_valid_configuration.tiles.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_covering_l1249_124957


namespace NUMINAMATH_CALUDE_differential_equation_holds_l1249_124914

open Real

noncomputable def y (x : ℝ) : ℝ := 1 / sqrt (sin x + x)

theorem differential_equation_holds (x : ℝ) (h : sin x + x > 0) :
  2 * sin x * (deriv y x) + y x * cos x = (y x)^3 * (x * cos x - sin x) := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_holds_l1249_124914


namespace NUMINAMATH_CALUDE_min_max_expression_l1249_124911

theorem min_max_expression (a b c : ℝ) 
  (eq1 : a^2 + a*b + b^2 = 19)
  (eq2 : b^2 + b*c + c^2 = 19) :
  (∃ x y z : ℝ, x^2 + x*y + y^2 = 19 ∧ y^2 + y*z + z^2 = 19 ∧ z^2 + z*x + x^2 = 0) ∧
  (∀ x y z : ℝ, x^2 + x*y + y^2 = 19 → y^2 + y*z + z^2 = 19 → z^2 + z*x + x^2 ≤ 76) :=
by
  sorry

end NUMINAMATH_CALUDE_min_max_expression_l1249_124911


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l1249_124956

open Complex

theorem smallest_distance_between_complex_points (z w : ℂ) 
  (hz : Complex.abs (z + 3 + 4*I) = 2)
  (hw : Complex.abs (w - 6 - 10*I) = 4) :
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), 
      Complex.abs (z' + 3 + 4*I) = 2 → 
      Complex.abs (w' - 6 - 10*I) = 4 → 
      Complex.abs (z' - w') ≥ min_dist) ∧ 
    min_dist = Real.sqrt 277 - 6 :=
sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_points_l1249_124956


namespace NUMINAMATH_CALUDE_sum_of_squares_of_reciprocals_l1249_124990

theorem sum_of_squares_of_reciprocals (x y : ℝ) 
  (sum_eq : x + y = 12) 
  (product_eq : x * y = 32) : 
  (1 / x)^2 + (1 / y)^2 = 5 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_reciprocals_l1249_124990


namespace NUMINAMATH_CALUDE_alternating_draw_probability_l1249_124929

/-- The number of white balls in the box -/
def white_balls : ℕ := 5

/-- The number of black balls in the box -/
def black_balls : ℕ := 3

/-- The total number of balls in the box -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of ways to choose positions for black balls -/
def total_arrangements : ℕ := Nat.choose total_balls black_balls

/-- The number of successful alternating color arrangements -/
def successful_arrangements : ℕ := Nat.choose (total_balls - 2) black_balls

/-- The probability of drawing an alternating color sequence -/
def alternating_probability : ℚ := successful_arrangements / total_arrangements

theorem alternating_draw_probability :
  alternating_probability = 5 / 14 := by sorry

end NUMINAMATH_CALUDE_alternating_draw_probability_l1249_124929


namespace NUMINAMATH_CALUDE_decagon_diagonals_l1249_124997

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l1249_124997


namespace NUMINAMATH_CALUDE_mothers_house_distance_l1249_124944

/-- The distance between your house and your mother's house -/
def total_distance : ℝ := 234.0

/-- The distance you have traveled so far -/
def traveled_distance : ℝ := 156.0

/-- Theorem stating that the total distance to your mother's house is 234.0 miles -/
theorem mothers_house_distance :
  (traveled_distance = (2/3) * total_distance) →
  total_distance = 234.0 :=
by
  sorry

#eval total_distance

end NUMINAMATH_CALUDE_mothers_house_distance_l1249_124944


namespace NUMINAMATH_CALUDE_min_days_to_plant_trees_eight_is_min_days_l1249_124959

theorem min_days_to_plant_trees (n : ℕ) : n ≥ 8 ↔ 2 * (2^n - 1) ≥ 100 := by
  sorry

theorem eight_is_min_days : ∃ (n : ℕ), n = 8 ∧ 2 * (2^n - 1) ≥ 100 ∧ ∀ (m : ℕ), m < n → 2 * (2^m - 1) < 100 := by
  sorry

end NUMINAMATH_CALUDE_min_days_to_plant_trees_eight_is_min_days_l1249_124959


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1249_124947

theorem sqrt_equation_solution :
  ∀ x : ℚ, (Real.sqrt (7 * x) / Real.sqrt (4 * (x + 2)) = 3) → x = -72 / 29 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1249_124947


namespace NUMINAMATH_CALUDE_train_speed_l1249_124986

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 140) (h2 : time = 16) :
  length / time = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1249_124986


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l1249_124902

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem f_decreasing_interval :
  ∀ x y : ℝ, x < y → y ≤ 1 → f y ≤ f x := by
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l1249_124902


namespace NUMINAMATH_CALUDE_train_length_calculation_l1249_124936

/-- Conversion factor from km/hr to m/s -/
def kmhr_to_ms : ℚ := 5 / 18

/-- Calculate the length of a train given its speed in km/hr and crossing time in seconds -/
def train_length (speed : ℚ) (time : ℚ) : ℚ :=
  speed * kmhr_to_ms * time

/-- The cumulative length of two trains -/
def cumulative_length (speed1 speed2 time1 time2 : ℚ) : ℚ :=
  train_length speed1 time1 + train_length speed2 time2

theorem train_length_calculation (speed1 speed2 time1 time2 : ℚ) :
  speed1 = 27 ∧ speed2 = 45 ∧ time1 = 20 ∧ time2 = 30 →
  cumulative_length speed1 speed2 time1 time2 = 525 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1249_124936


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l1249_124919

theorem probability_at_least_one_correct (n : ℕ) (choices : ℕ) : 
  n = 6 → choices = 6 → 1 - (1 - 1 / choices) ^ n = 31031 / 46656 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l1249_124919


namespace NUMINAMATH_CALUDE_x_minus_y_equals_two_l1249_124996

theorem x_minus_y_equals_two (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (diff_squares_eq : x^2 - y^2 = 16) : 
  x - y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_two_l1249_124996


namespace NUMINAMATH_CALUDE_book_area_calculation_l1249_124943

/-- Calculates the area of a book given its length expression, width, and conversion factor. -/
theorem book_area_calculation (x : ℝ) (inch_to_cm : ℝ) : 
  x = 5 → 
  inch_to_cm = 2.54 → 
  (3 * x - 4) * ((5 / 2) * inch_to_cm) = 69.85 := by
  sorry

end NUMINAMATH_CALUDE_book_area_calculation_l1249_124943


namespace NUMINAMATH_CALUDE_submarine_hit_guaranteed_l1249_124973

/-- Represents the position of a submarine at time t -/
def submarinePosition (v : ℕ+) (t : ℕ) : ℕ := v.val * t

/-- Represents the position of a missile fired at time n -/
def missilePosition (n : ℕ) : ℕ := n ^ 2

/-- Theorem stating that there exists a firing sequence that will hit the submarine -/
theorem submarine_hit_guaranteed :
  ∀ (v : ℕ+), ∃ (t : ℕ), submarinePosition v t = missilePosition t := by
  sorry


end NUMINAMATH_CALUDE_submarine_hit_guaranteed_l1249_124973


namespace NUMINAMATH_CALUDE_bucket_filling_time_l1249_124909

theorem bucket_filling_time (total_time : ℝ) (h : total_time = 135) : 
  (2 / 3 : ℝ) * total_time = 90 := by
  sorry

end NUMINAMATH_CALUDE_bucket_filling_time_l1249_124909


namespace NUMINAMATH_CALUDE_y_quadratic_iff_m_eq_2_y_linear_iff_m_special_l1249_124928

noncomputable def y (m : ℝ) (x : ℝ) : ℝ := (m + 3) * x^(m^2 + m - 4) + (m + 2) * x + 3

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def is_linear (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

theorem y_quadratic_iff_m_eq_2 (m : ℝ) :
  is_quadratic (y m) ↔ m = 2 :=
sorry

theorem y_linear_iff_m_special (m : ℝ) :
  is_linear (y m) ↔ 
    m = -3 ∨ 
    m = (-1 + Real.sqrt 17) / 2 ∨ 
    m = (-1 - Real.sqrt 17) / 2 ∨
    m = (-1 + Real.sqrt 21) / 2 ∨
    m = (-1 - Real.sqrt 21) / 2 :=
sorry

end NUMINAMATH_CALUDE_y_quadratic_iff_m_eq_2_y_linear_iff_m_special_l1249_124928


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1249_124908

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1 = (x-1)^4 * (x+1)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1249_124908


namespace NUMINAMATH_CALUDE_solution_set_f_positive_range_of_a_l1249_124983

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for the solution set of f(x) > 0
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1/3 ∨ x > 3} := by sorry

-- Theorem for the range of a
theorem range_of_a :
  {a : ℝ | ∃ x₀ : ℝ, f x₀ + 2*a^2 < 4*a} = {a : ℝ | -1/2 < a ∧ a < 5/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_range_of_a_l1249_124983


namespace NUMINAMATH_CALUDE_roger_tray_trips_l1249_124932

/-- Calculates the number of trips needed to carry a given number of trays -/
def trips_needed (trays_per_trip : ℕ) (total_trays : ℕ) : ℕ :=
  (total_trays + trays_per_trip - 1) / trays_per_trip

/-- Proves that 3 trips are needed to carry 12 trays when 4 trays can be carried per trip -/
theorem roger_tray_trips : trips_needed 4 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_roger_tray_trips_l1249_124932


namespace NUMINAMATH_CALUDE_total_discount_calculation_l1249_124904

/-- Calculates the total discount percentage given a sale discount, coupon discount, and loyalty discount -/
theorem total_discount_calculation (original_price : ℝ) (sale_discount : ℝ) (coupon_discount : ℝ) (loyalty_discount : ℝ) :
  sale_discount = 1/3 →
  coupon_discount = 0.25 →
  loyalty_discount = 0.05 →
  let sale_price := original_price * (1 - sale_discount)
  let price_after_coupon := sale_price * (1 - coupon_discount)
  let final_price := price_after_coupon * (1 - loyalty_discount)
  (original_price - final_price) / original_price = 0.525 :=
by sorry

end NUMINAMATH_CALUDE_total_discount_calculation_l1249_124904


namespace NUMINAMATH_CALUDE_convex_quadrilateral_symmetric_division_l1249_124921

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  -- We don't need to define the specifics of the quadrilateral,
  -- just that it exists and is convex
  isConvex : Bool

/-- A polygon with an axis of symmetry -/
structure SymmetricPolygon where
  -- We don't need to define the specifics of the polygon,
  -- just that it exists and has an axis of symmetry
  hasSymmetryAxis : Bool

/-- A division of a quadrilateral into polygons -/
structure QuadrilateralDivision (q : ConvexQuadrilateral) where
  polygons : List SymmetricPolygon
  divisionValid : Bool  -- This would ensure the division is valid

/-- The main theorem -/
theorem convex_quadrilateral_symmetric_division 
  (q : ConvexQuadrilateral) : 
  ∃ (d : QuadrilateralDivision q), 
    d.polygons.length = 5 ∧ 
    d.divisionValid ∧ 
    ∀ p ∈ d.polygons, p.hasSymmetryAxis := by
  sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_symmetric_division_l1249_124921


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1249_124925

/-- A quadratic equation in terms of x is of the form ax² + bx + c = 0, where a ≠ 0 --/
def is_quadratic_in_x (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x² - x + 1 --/
def f (x : ℝ) : ℝ := x^2 - x + 1

/-- Theorem: f(x) = x² - x + 1 is a quadratic equation in terms of x --/
theorem f_is_quadratic : is_quadratic_in_x f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l1249_124925


namespace NUMINAMATH_CALUDE_proportion_inequality_l1249_124952

theorem proportion_inequality (a b c : ℝ) (h : a / b = b / c) : a^2 + c^2 ≥ 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_inequality_l1249_124952


namespace NUMINAMATH_CALUDE_rice_wheat_division_l1249_124963

/-- Calculates the approximate amount of wheat grains in a large quantity of mixed grains,
    given a sample ratio. -/
def approximate_wheat_amount (total_amount : ℕ) (sample_size : ℕ) (wheat_in_sample : ℕ) : ℕ :=
  (total_amount * wheat_in_sample) / sample_size

/-- The rice and wheat division problem from "Jiuzhang Suanshu" -/
theorem rice_wheat_division :
  let total_amount : ℕ := 1536
  let sample_size : ℕ := 224
  let wheat_in_sample : ℕ := 28
  approximate_wheat_amount total_amount sample_size wheat_in_sample = 192 := by
  sorry

#eval approximate_wheat_amount 1536 224 28

end NUMINAMATH_CALUDE_rice_wheat_division_l1249_124963


namespace NUMINAMATH_CALUDE_gala_handshakes_l1249_124960

/-- Number of married couples at the gala -/
def num_couples : ℕ := 15

/-- Total number of people at the gala -/
def total_people : ℕ := 2 * num_couples

/-- Number of handshakes between men -/
def handshakes_men : ℕ := num_couples.choose 2

/-- Number of handshakes between men and women -/
def handshakes_men_women : ℕ := num_couples * num_couples

/-- Total number of handshakes at the gala -/
def total_handshakes : ℕ := handshakes_men + handshakes_men_women

theorem gala_handshakes : total_handshakes = 330 := by
  sorry

end NUMINAMATH_CALUDE_gala_handshakes_l1249_124960


namespace NUMINAMATH_CALUDE_coefficient_c_nonzero_l1249_124965

/-- A polynomial of degree 4 with four distinct roots, one of which is 0 -/
structure QuarticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  has_four_distinct_roots : ∃ (p q r : ℝ), p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧
    ∀ x, x^4 + a*x^3 + b*x^2 + c*x + d = x*(x-p)*(x-q)*(x-r)
  zero_is_root : d = 0

theorem coefficient_c_nonzero (Q : QuarticPolynomial) : Q.c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_c_nonzero_l1249_124965


namespace NUMINAMATH_CALUDE_hotel_distance_proof_l1249_124900

/-- Calculates the remaining distance to the hotel given a series of travel segments --/
def remaining_distance_to_hotel (total_distance : ℝ) 
  (initial_speed initial_time : ℝ)
  (rain_speed rain_time : ℝ)
  (high_speed high_time : ℝ)
  (diff_car_speed diff_car_time : ℝ)
  (moto_init_speed moto_init_time : ℝ)
  (moto_malf_speed moto_malf_time : ℝ) : ℝ :=
  total_distance - (
    initial_speed * initial_time +
    rain_speed * rain_time +
    high_speed * high_time +
    diff_car_speed * diff_car_time +
    moto_init_speed * moto_init_time +
    moto_malf_speed * moto_malf_time
  )

theorem hotel_distance_proof :
  remaining_distance_to_hotel 1200 60 2 40 1 70 2.5 50 4 80 1 60 3 = 405 := by
  sorry

end NUMINAMATH_CALUDE_hotel_distance_proof_l1249_124900


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1249_124980

theorem absolute_value_equation_solution :
  let f : ℝ → ℝ := λ x => 3*x + 6
  let g : ℝ → ℝ := λ x => |(-20 + x^2)|
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 113) / 2 ∧
              x₂ = (3 - Real.sqrt 113) / 2 ∧
              (∀ x : ℝ, f x = g x ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1249_124980


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1249_124993

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x, -1 ≤ x ∧ x ≤ 2 ↔ (3 * x - 1 ≤ a ∧ 2 * x ≥ 6 - b)) →
  a + b = 13 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1249_124993


namespace NUMINAMATH_CALUDE_tan_675_degrees_l1249_124984

theorem tan_675_degrees (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (675 * π / 180) →
  n = 135 ∨ n = -45 := by
sorry

end NUMINAMATH_CALUDE_tan_675_degrees_l1249_124984


namespace NUMINAMATH_CALUDE_baseball_team_size_l1249_124985

/-- Given a baseball team with the following properties:
  * The team scored a total of 270 points in the year
  * 5 players averaged 50 points each
  * The remaining players averaged 5 points each
  Prove that the total number of players on the team is 9. -/
theorem baseball_team_size :
  ∀ (total_score : ℕ) (top_players : ℕ) (top_avg : ℕ) (rest_avg : ℕ),
  total_score = 270 →
  top_players = 5 →
  top_avg = 50 →
  rest_avg = 5 →
  ∃ (total_players : ℕ),
    total_players = top_players + (total_score - top_players * top_avg) / rest_avg ∧
    total_players = 9 :=
by sorry

end NUMINAMATH_CALUDE_baseball_team_size_l1249_124985


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l1249_124961

theorem absolute_value_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 5*x*y) :
  |((x + y) / (x - y))| = Real.sqrt (7/3) := by
sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l1249_124961


namespace NUMINAMATH_CALUDE_pens_purchased_l1249_124955

theorem pens_purchased (total_cost : ℝ) (num_pencils : ℕ) (pencil_price : ℝ) (pen_price : ℝ)
  (h1 : total_cost = 690)
  (h2 : num_pencils = 75)
  (h3 : pencil_price = 2)
  (h4 : pen_price = 18) :
  (total_cost - num_pencils * pencil_price) / pen_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_pens_purchased_l1249_124955


namespace NUMINAMATH_CALUDE_ceiling_product_equation_solution_l1249_124978

theorem ceiling_product_equation_solution :
  ∃! (x : ℝ), ⌈x⌉ * x = 225 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_solution_l1249_124978


namespace NUMINAMATH_CALUDE_brendan_grass_cutting_l1249_124988

/-- Proves that Brendan can cut 84 yards of grass in a week with his new lawnmower -/
theorem brendan_grass_cutting (initial_capacity : ℕ) (increase_percentage : ℚ) (days_in_week : ℕ) :
  initial_capacity = 8 →
  increase_percentage = 1/2 →
  days_in_week = 7 →
  (initial_capacity + initial_capacity * increase_percentage) * days_in_week = 84 :=
by sorry

end NUMINAMATH_CALUDE_brendan_grass_cutting_l1249_124988


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l1249_124964

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n = 15625) ∧
  (∀ m : ℕ, m < n → m < 10000 ∨ m > 99999 ∨ ¬∃ a : ℕ, m = a^2 ∨ ¬∃ b : ℕ, m = b^3) ∧
  (∃ x : ℕ, n = x^2) ∧
  (∃ y : ℕ, n = y^3) ∧
  (n ≥ 10000) ∧
  (n ≤ 99999) :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l1249_124964


namespace NUMINAMATH_CALUDE_largest_quotient_and_smallest_product_l1249_124972

def S : Set ℤ := {-25, -4, -1, 3, 5, 9}

theorem largest_quotient_and_smallest_product (a b : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hb_nonzero : b ≠ 0) :
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S) (hy_nonzero : y ≠ 0), (a / b : ℚ) ≤ (x / y : ℚ)) ∧
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S), a * b ≥ x * y) ∧
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S) (hy_nonzero : y ≠ 0), (x / y : ℚ) = 3) ∧
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S), x * y = -225) := by
  sorry

end NUMINAMATH_CALUDE_largest_quotient_and_smallest_product_l1249_124972


namespace NUMINAMATH_CALUDE_part1_part2_l1249_124938

-- Define the operation
def star_op (a b : ℚ) : ℚ := (a * b) / (a + b)

-- Part 1: Prove the specific calculation
theorem part1 : star_op (-3) (-1/3) = -3/10 := by sorry

-- Part 2: Prove when the operation is undefined
theorem part2 (a b : ℚ) : 
  a + b = 0 → ¬ ∃ (q : ℚ), star_op a b = q := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1249_124938


namespace NUMINAMATH_CALUDE_replaced_girl_weight_l1249_124924

theorem replaced_girl_weight
  (n : ℕ)
  (original_average : ℝ)
  (new_average : ℝ)
  (new_girl_weight : ℝ)
  (h1 : n = 25)
  (h2 : new_average = original_average + 1)
  (h3 : new_girl_weight = 80) :
  ∃ (replaced_weight : ℝ),
    replaced_weight = new_girl_weight - n * (new_average - original_average) ∧
    replaced_weight = 55 := by
  sorry

end NUMINAMATH_CALUDE_replaced_girl_weight_l1249_124924


namespace NUMINAMATH_CALUDE_max_value_of_sum_cube_roots_l1249_124920

open Real

theorem max_value_of_sum_cube_roots (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_constraint : a + b + c + d = 100) : 
  let S := (a / (b + 7)) ^ (1/3) + (b / (c + 7)) ^ (1/3) + 
           (c / (d + 7)) ^ (1/3) + (d / (a + 7)) ^ (1/3)
  S ≤ 8 / 7 ^ (1/3) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_cube_roots_l1249_124920


namespace NUMINAMATH_CALUDE_simplify_exponents_l1249_124935

theorem simplify_exponents (a b : ℝ) : (a^4 * a^3) * (b^2 * b^5) = a^7 * b^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_exponents_l1249_124935


namespace NUMINAMATH_CALUDE_ratio_evaluation_l1249_124934

theorem ratio_evaluation : (2^2005 * 3^2003) / 6^2004 = 2/3 := by sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l1249_124934


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1249_124999

theorem polynomial_factorization (x : ℝ) :
  x^6 - 3*x^4 + 3*x^2 - 1 = (x-1)^3*(x+1)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1249_124999


namespace NUMINAMATH_CALUDE_engineering_collections_l1249_124927

/-- Represents the count of each letter in "ENGINEERING" -/
structure LetterCount where
  e : Nat -- vowel
  n : Nat -- consonant
  g : Nat -- consonant
  r : Nat -- consonant
  i : Nat -- consonant

/-- Represents a collection of letters -/
structure LetterCollection where
  vowels : Nat
  consonants : Nat

/-- Checks if a letter collection is valid -/
def isValidCollection (lc : LetterCollection) : Prop :=
  lc.vowels = 3 ∧ lc.consonants = 3

/-- Counts the number of distinct letter collections -/
noncomputable def countDistinctCollections (word : LetterCount) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem engineering_collections (word : LetterCount) 
  (h1 : word.e = 5) -- number of E's
  (h2 : word.n = 2) -- number of N's
  (h3 : word.g = 3) -- number of G's
  (h4 : word.r = 1) -- number of R's
  (h5 : word.i = 1) -- number of I's
  : countDistinctCollections word = 13 := by sorry

end NUMINAMATH_CALUDE_engineering_collections_l1249_124927


namespace NUMINAMATH_CALUDE_problem_solution_l1249_124942

theorem problem_solution (x y : ℝ) (hx : x = 3) (hy : y = 4) : 
  (x^3 + 3*y^2) / 7 = 75/7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1249_124942


namespace NUMINAMATH_CALUDE_cake_division_l1249_124991

theorem cake_division (n_cakes : ℕ) (n_girls : ℕ) (share : ℚ) :
  n_cakes = 11 →
  n_girls = 6 →
  share = 1 + 1/2 + 1/4 + 1/12 →
  ∃ (division : List (List ℚ)),
    (∀ piece ∈ division.join, piece ≠ 1/6) ∧
    (division.length = n_girls) ∧
    (∀ girl_share ∈ division, girl_share.sum = share) ∧
    (division.join.sum = n_cakes) :=
by sorry

end NUMINAMATH_CALUDE_cake_division_l1249_124991


namespace NUMINAMATH_CALUDE_complex_arg_range_l1249_124950

theorem complex_arg_range (z : ℂ) (h : Complex.abs (2 * z + 1 / z) = 1) :
  ∃ k : ℤ, k ∈ ({0, 1} : Set ℤ) ∧
    k * Real.pi + Real.pi / 2 - (1 / 2) * Real.arccos (3 / 4) ≤ Complex.arg z ∧
    Complex.arg z ≤ k * Real.pi + Real.pi / 2 + (1 / 2) * Real.arccos (3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_complex_arg_range_l1249_124950


namespace NUMINAMATH_CALUDE_cylinder_dimensions_l1249_124917

/-- A cylinder whose bases' centers coincide with two opposite vertices of a unit cube,
    and whose lateral surface contains the remaining vertices of the cube -/
structure CylinderWithUnitCube where
  -- The height of the cylinder
  height : ℝ
  -- The base radius of the cylinder
  radius : ℝ
  -- The opposite vertices of the unit cube coincide with the centers of the cylinder bases
  opposite_vertices_on_bases : height = Real.sqrt 3
  -- The remaining vertices of the cube are on the lateral surface of the cylinder
  other_vertices_on_surface : radius = Real.sqrt 6 / 3

/-- The height and radius of a cylinder satisfying the given conditions -/
theorem cylinder_dimensions (c : CylinderWithUnitCube) :
  c.height = Real.sqrt 3 ∧ c.radius = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_cylinder_dimensions_l1249_124917


namespace NUMINAMATH_CALUDE_ribbon_division_l1249_124912

theorem ribbon_division (total_ribbon : ℚ) (num_boxes : ℕ) (ribbon_per_box : ℚ) :
  total_ribbon = 5 / 8 →
  num_boxes = 5 →
  ribbon_per_box = total_ribbon / num_boxes →
  ribbon_per_box = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_division_l1249_124912


namespace NUMINAMATH_CALUDE_trivia_competition_score_l1249_124982

theorem trivia_competition_score :
  ∀ (total_members absent_members points_per_member : ℕ),
    total_members = 120 →
    absent_members = 37 →
    points_per_member = 24 →
    (total_members - absent_members) * points_per_member = 1992 :=
by
  sorry

end NUMINAMATH_CALUDE_trivia_competition_score_l1249_124982


namespace NUMINAMATH_CALUDE_junior_fraction_l1249_124930

/-- Represents the number of students in each category -/
structure StudentCounts where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (s : StudentCounts) : Prop :=
  s.freshmen + s.sophomores + s.juniors + s.seniors = 120 ∧
  s.freshmen > 0 ∧ s.sophomores > 0 ∧ s.juniors > 0 ∧ s.seniors > 0 ∧
  s.freshmen = 2 * s.sophomores ∧
  s.juniors = 4 * s.seniors ∧
  (s.freshmen : ℚ) / 2 + (s.sophomores : ℚ) / 3 = (s.juniors : ℚ) * 2 / 3 - (s.seniors : ℚ) / 4

/-- The theorem to be proved -/
theorem junior_fraction (s : StudentCounts) (h : satisfiesConditions s) :
    (s.juniors : ℚ) / (s.freshmen + s.sophomores + s.juniors + s.seniors) = 32 / 167 := by
  sorry

end NUMINAMATH_CALUDE_junior_fraction_l1249_124930


namespace NUMINAMATH_CALUDE_maximum_marks_proof_l1249_124915

/-- Given a student needs 50% to pass, got 200 marks, and failed by 20 marks, prove the maximum marks are 440. -/
theorem maximum_marks_proof (passing_percentage : Real) (student_marks : Nat) (failing_margin : Nat) :
  passing_percentage = 0.5 →
  student_marks = 200 →
  failing_margin = 20 →
  ∃ (max_marks : Nat), max_marks = 440 ∧ 
    passing_percentage * max_marks = student_marks + failing_margin :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_proof_l1249_124915


namespace NUMINAMATH_CALUDE_cookie_accident_l1249_124969

/-- Problem: Cookie Baking Accident -/
theorem cookie_accident (alice_initial bob_initial alice_additional bob_additional final_edible : ℕ) :
  alice_initial = 74 →
  bob_initial = 7 →
  alice_additional = 5 →
  bob_additional = 36 →
  final_edible = 93 →
  (alice_initial + bob_initial + alice_additional + bob_additional) - final_edible = 29 :=
by sorry

end NUMINAMATH_CALUDE_cookie_accident_l1249_124969


namespace NUMINAMATH_CALUDE_pencil_distribution_l1249_124905

theorem pencil_distribution (x y : ℕ+) (h1 : 3 * x < 48) (h2 : 48 < 4 * x) 
  (h3 : 4 * y < 48) (h4 : 48 < 5 * y) : 
  (3 * x < 48 ∧ 48 < 4 * x) ∧ (4 * y < 48 ∧ 48 < 5 * y) := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1249_124905


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1249_124933

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 20

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 30

theorem bowling_ball_weight_proof :
  (9 * bowling_ball_weight = 6 * canoe_weight) ∧
  (4 * canoe_weight = 120) →
  bowling_ball_weight = 20 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1249_124933


namespace NUMINAMATH_CALUDE_inheritance_investment_rate_l1249_124946

theorem inheritance_investment_rate 
  (inheritance : ℝ) 
  (first_investment : ℝ) 
  (second_rate : ℝ) 
  (total_interest : ℝ) : 
  inheritance = 12000 →
  first_investment = 5000 →
  second_rate = 0.08 →
  total_interest = 860 →
  ∃ (r : ℝ), 
    r * first_investment + second_rate * (inheritance - first_investment) = total_interest ∧ 
    r = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_investment_rate_l1249_124946


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_eight_l1249_124951

theorem sum_of_roots_eq_eight : 
  let f : ℝ → ℝ := λ x => (x - 4)^2 - 16
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_eight_l1249_124951


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l1249_124939

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t2.a = k * t1.a ∧ t2.b = k * t1.b ∧ t2.c = k * t1.c

theorem similar_triangle_perimeter (small large : Triangle) :
  small.isIsosceles ∧
  small.a = 15 ∧ small.b = 15 ∧ small.c = 6 ∧
  small.isSimilar large ∧
  large.c = 18 →
  large.perimeter = 108 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l1249_124939


namespace NUMINAMATH_CALUDE_penalty_kicks_required_l1249_124987

theorem penalty_kicks_required (total_players : ℕ) (goalies : ℕ) (h1 : total_players = 18) (h2 : goalies = 4) : 
  (total_players - goalies) * goalies = 68 := by
  sorry

end NUMINAMATH_CALUDE_penalty_kicks_required_l1249_124987


namespace NUMINAMATH_CALUDE_complex_modulus_l1249_124970

theorem complex_modulus (z : ℂ) : z + Complex.I = (2 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1249_124970


namespace NUMINAMATH_CALUDE_multiple_in_difference_l1249_124906

theorem multiple_in_difference (n m : ℤ) (h1 : n = -7) (h2 : 3 * n = m * n - 7) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_in_difference_l1249_124906


namespace NUMINAMATH_CALUDE_prob_sum_27_l1249_124958

/-- Represents a die with 20 faces -/
structure Die :=
  (faces : Finset ℕ)
  (fair : faces.card = 20)

/-- The first die with faces 1 through 19 -/
def die1 : Die :=
  { faces := Finset.range 20 \ {20},
    fair := sorry }

/-- The second die with faces 1 through 7 and 9 through 21 -/
def die2 : Die :=
  { faces := (Finset.range 22 \ {0, 8}),
    fair := sorry }

/-- The set of all possible outcomes when rolling both dice -/
def allOutcomes : Finset (ℕ × ℕ) :=
  die1.faces.product die2.faces

/-- The set of outcomes that sum to 27 -/
def sumTo27 : Finset (ℕ × ℕ) :=
  allOutcomes.filter (fun p => p.1 + p.2 = 27)

/-- The probability of rolling a sum of 27 -/
def probSum27 : ℚ :=
  sumTo27.card / allOutcomes.card

theorem prob_sum_27 : probSum27 = 3 / 100 := by sorry

end NUMINAMATH_CALUDE_prob_sum_27_l1249_124958


namespace NUMINAMATH_CALUDE_system_unique_solution_l1249_124994

/-- The system of equations has a unique solution (1, 2) -/
theorem system_unique_solution :
  ∃! (x y : ℝ), x + 2*y = 5 ∧ 3*x - y = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_system_unique_solution_l1249_124994


namespace NUMINAMATH_CALUDE_shelbys_driving_time_l1249_124953

/-- Shelby's driving problem -/
theorem shelbys_driving_time (speed_no_rain speed_rain : ℝ) (total_time total_distance : ℝ) 
  (h1 : speed_no_rain = 40)
  (h2 : speed_rain = 25)
  (h3 : total_time = 3)
  (h4 : total_distance = 85) :
  let rain_time := (total_distance - speed_no_rain * total_time) / (speed_rain - speed_no_rain)
  rain_time * 60 = 140 := by sorry

end NUMINAMATH_CALUDE_shelbys_driving_time_l1249_124953


namespace NUMINAMATH_CALUDE_replacement_philosophy_in_lines_one_and_three_l1249_124954

/-- Represents a line of poetry -/
inductive PoeticLine
| EndlessFalling
| SpringRiver
| NewLeaves
| Waterfall

/-- Checks if a poetic line contains the philosophy of new things replacing old ones -/
def containsReplacementPhilosophy (line : PoeticLine) : Prop :=
  match line with
  | PoeticLine.EndlessFalling => True
  | PoeticLine.SpringRiver => False
  | PoeticLine.NewLeaves => True
  | PoeticLine.Waterfall => False

/-- The theorem stating that only lines ① and ③ contain the replacement philosophy -/
theorem replacement_philosophy_in_lines_one_and_three :
  (∀ line : PoeticLine, containsReplacementPhilosophy line ↔
    (line = PoeticLine.EndlessFalling ∨ line = PoeticLine.NewLeaves)) :=
by sorry

end NUMINAMATH_CALUDE_replacement_philosophy_in_lines_one_and_three_l1249_124954


namespace NUMINAMATH_CALUDE_barrel_tank_ratio_l1249_124995

theorem barrel_tank_ratio : 
  ∀ (barrel_volume tank_volume : ℝ),
  barrel_volume > 0 → tank_volume > 0 →
  (3/4 : ℝ) * barrel_volume = (5/8 : ℝ) * tank_volume →
  barrel_volume / tank_volume = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_barrel_tank_ratio_l1249_124995


namespace NUMINAMATH_CALUDE_order_of_exponents_l1249_124931

theorem order_of_exponents :
  let a : ℝ := (36 : ℝ) ^ (1/5)
  let b : ℝ := (3 : ℝ) ^ (4/3)
  let c : ℝ := (9 : ℝ) ^ (2/5)
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_order_of_exponents_l1249_124931


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l1249_124975

theorem area_between_concentric_circles (r_small : ℝ) (r_large : ℝ) : 
  r_small = 3 →
  r_large = 3 * r_small →
  π * r_large^2 - π * r_small^2 = 72 * π :=
by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l1249_124975


namespace NUMINAMATH_CALUDE_complex_power_2007_l1249_124998

theorem complex_power_2007 : (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I)) ^ 2007 = -Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_2007_l1249_124998


namespace NUMINAMATH_CALUDE_quadratic_triple_root_l1249_124922

/-- For a quadratic equation ax^2 + bx + c = 0, if one root is triple the other, 
    then 3b^2 = 16ac -/
theorem quadratic_triple_root (a b c : ℝ) (h : ∃ x y : ℝ, x ≠ 0 ∧ 
  a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) : 
  3 * b^2 = 16 * a * c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_triple_root_l1249_124922


namespace NUMINAMATH_CALUDE_problem_solution_l1249_124949

theorem problem_solution : ∃ x : ℝ, (0.2 * 30 = 0.25 * x + 2) ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1249_124949


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l1249_124992

/-- A parabola with vertex at the origin passing through (-2, 4) -/
structure Parabola where
  /-- The equation of the parabola is either x^2 = ay or y^2 = bx for some a, b ∈ ℝ -/
  equation : (∃ a : ℝ, ∀ x y : ℝ, y = a * x^2) ∨ (∃ b : ℝ, ∀ x y : ℝ, x = b * y^2)
  /-- The parabola passes through the point (-2, 4) -/
  point : (∃ a : ℝ, 4 = a * (-2)^2) ∨ (∃ b : ℝ, -2 = b * 4^2)

/-- The standard equation of the parabola is either x^2 = y or y^2 = -8x -/
theorem parabola_standard_equation (p : Parabola) :
  (∀ x y : ℝ, y = x^2) ∨ (∀ x y : ℝ, x = -8 * y^2) :=
sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l1249_124992


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_l1249_124907

/-- The number of faces on each cube -/
def faces_per_cube : ℕ := 6

/-- The number of digits (0 to 9) -/
def num_digits : ℕ := 10

/-- The length of the number we need to be able to form -/
def number_length : ℕ := 30

/-- The minimum number of each non-zero digit needed -/
def min_nonzero_digits : ℕ := number_length

/-- The minimum number of zero digits needed -/
def min_zero_digits : ℕ := number_length - 1

/-- The total minimum number of digit instances needed -/
def total_min_digits : ℕ := min_nonzero_digits * (num_digits - 1) + min_zero_digits

/-- The smallest number of cubes needed to form any 30-digit number -/
def min_cubes : ℕ := 50

theorem smallest_number_of_cubes : 
  faces_per_cube * min_cubes ≥ total_min_digits ∧ 
  ∀ n : ℕ, n < min_cubes → faces_per_cube * n < total_min_digits :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_cubes_l1249_124907


namespace NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l1249_124923

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

theorem fourth_term_of_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 0 = 12 →
  a 5 = 47 →
  a 3 = 29.5 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l1249_124923


namespace NUMINAMATH_CALUDE_one_less_than_negative_one_l1249_124971

theorem one_less_than_negative_one : -1 - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_one_less_than_negative_one_l1249_124971


namespace NUMINAMATH_CALUDE_pizza_slices_theorem_l1249_124974

/-- Represents the types of pizzas available --/
inductive PizzaType
  | Small
  | Medium
  | Large

/-- Returns the number of slices for a given pizza type --/
def slicesPerPizza (pt : PizzaType) : Nat :=
  match pt with
  | .Small => 6
  | .Medium => 8
  | .Large => 12

/-- Calculates the total number of slices for a given number of pizzas of a specific type --/
def totalSlices (pt : PizzaType) (count : Nat) : Nat :=
  (slicesPerPizza pt) * count

/-- Represents the order of pizzas --/
structure PizzaOrder where
  small : Nat
  medium : Nat
  large : Nat
  total : Nat

theorem pizza_slices_theorem (order : PizzaOrder)
  (h1 : order.small = 4)
  (h2 : order.medium = 5)
  (h3 : order.total = 15)
  (h4 : order.large = order.total - order.small - order.medium) :
  totalSlices .Small order.small +
  totalSlices .Medium order.medium +
  totalSlices .Large order.large = 136 := by
    sorry

#check pizza_slices_theorem

end NUMINAMATH_CALUDE_pizza_slices_theorem_l1249_124974


namespace NUMINAMATH_CALUDE_longest_tape_l1249_124981

theorem longest_tape (red_tape blue_tape yellow_tape : ℚ) 
  (h_red : red_tape = 11/6)
  (h_blue : blue_tape = 7/4)
  (h_yellow : yellow_tape = 13/8) :
  red_tape > blue_tape ∧ red_tape > yellow_tape := by
  sorry

end NUMINAMATH_CALUDE_longest_tape_l1249_124981


namespace NUMINAMATH_CALUDE_stuffed_animals_count_l1249_124937

/-- The number of stuffed animals McKenna has -/
def mckenna_stuffed_animals : ℕ := 34

/-- The number of stuffed animals Kenley has -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- The number of stuffed animals Tenly has -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

theorem stuffed_animals_count : total_stuffed_animals = 175 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_animals_count_l1249_124937


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1249_124926

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1249_124926


namespace NUMINAMATH_CALUDE_billy_ferris_wheel_rides_l1249_124918

/-- The number of times Billy rode the ferris wheel -/
def F : ℕ := sorry

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost of each ride in tickets -/
def ticket_cost : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := 50

theorem billy_ferris_wheel_rides : F = 7 := by
  sorry

end NUMINAMATH_CALUDE_billy_ferris_wheel_rides_l1249_124918


namespace NUMINAMATH_CALUDE_kite_area_is_102_l1249_124968

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : Int
  y : Int

/-- Represents a kite shape -/
structure Kite where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Calculate the area of a kite given its vertices -/
def kiteArea (k : Kite) : Int :=
  sorry

/-- The kite in the problem -/
def problemKite : Kite := {
  p1 := { x := 0, y := 10 }
  p2 := { x := 6, y := 14 }
  p3 := { x := 12, y := 10 }
  p4 := { x := 6, y := 0 }
}

theorem kite_area_is_102 : kiteArea problemKite = 102 := by
  sorry

end NUMINAMATH_CALUDE_kite_area_is_102_l1249_124968


namespace NUMINAMATH_CALUDE_marco_cards_l1249_124945

theorem marco_cards (C : ℕ) : 
  (C / 4 : ℚ) * (1 / 5 : ℚ) = 25 → C = 500 := by
  sorry

end NUMINAMATH_CALUDE_marco_cards_l1249_124945


namespace NUMINAMATH_CALUDE_octal_sum_equality_l1249_124916

/-- Converts a base-8 number to base-10 --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def decimal_to_octal (n : ℕ) : ℕ := sorry

/-- The sum of three octal numbers is equal to another octal number --/
theorem octal_sum_equality : 
  decimal_to_octal (octal_to_decimal 236 + octal_to_decimal 425 + octal_to_decimal 157) = 1042 := by
  sorry

end NUMINAMATH_CALUDE_octal_sum_equality_l1249_124916


namespace NUMINAMATH_CALUDE_ball_probability_l1249_124966

theorem ball_probability (m n : ℕ) : 
  (∃ (total : ℕ), total = m + 8 + n ∧ 
   (8 : ℚ) / total = (m + n : ℚ) / total) → 
  m + n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ball_probability_l1249_124966


namespace NUMINAMATH_CALUDE_function_equation_solution_l1249_124976

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a^2 + a*b + f (b^2)) = a * f b + b^2 + f (a^2)

/-- The main theorem stating that any function satisfying the equation is either the identity or negation -/
theorem function_equation_solution (f : ℝ → ℝ) (hf : SatisfiesEquation f) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l1249_124976


namespace NUMINAMATH_CALUDE_fifty_bees_honey_production_l1249_124913

/-- The amount of honey (in grams) produced by a given number of bees in 50 days -/
def honey_production (num_bees : ℕ) : ℕ :=
  num_bees * 1

theorem fifty_bees_honey_production :
  honey_production 50 = 50 := by sorry

end NUMINAMATH_CALUDE_fifty_bees_honey_production_l1249_124913


namespace NUMINAMATH_CALUDE_two_x_power_x_eq_sqrt_two_solutions_l1249_124979

theorem two_x_power_x_eq_sqrt_two_solutions (x : ℝ) :
  x > 0 ∧ 2 * (x ^ x) = Real.sqrt 2 ↔ x = 1/2 ∨ x = 1/4 :=
sorry

end NUMINAMATH_CALUDE_two_x_power_x_eq_sqrt_two_solutions_l1249_124979


namespace NUMINAMATH_CALUDE_zias_club_size_l1249_124967

/-- Represents the number of people with one coin -/
def one_coin_people : ℕ := 7

/-- Represents the angle of the smallest sector in degrees -/
def smallest_sector : ℕ := 35

/-- Represents the angle increment between sectors in degrees -/
def angle_increment : ℕ := 10

/-- Calculates the total number of sectors in the pie chart -/
def total_sectors : ℕ := 6

/-- Represents the total angle of a full circle in degrees -/
def full_circle : ℕ := 360

/-- Theorem: The number of people in Zia's club is 72 -/
theorem zias_club_size : 
  (full_circle / (smallest_sector / one_coin_people) : ℕ) = 72 := by
  sorry

end NUMINAMATH_CALUDE_zias_club_size_l1249_124967
