import Mathlib

namespace NUMINAMATH_CALUDE_sodium_atom_diameter_scientific_notation_l3793_379381

theorem sodium_atom_diameter_scientific_notation :
  let d : ℝ := 0.0000000599
  let s : ℝ := 5.99
  ∃ n : ℤ, d = s * (10 : ℝ) ^ n ∧ n = -8 :=
by sorry

end NUMINAMATH_CALUDE_sodium_atom_diameter_scientific_notation_l3793_379381


namespace NUMINAMATH_CALUDE_product_equals_20152015_l3793_379362

theorem product_equals_20152015 : 5 * 13 * 31 * 73 * 137 = 20152015 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_20152015_l3793_379362


namespace NUMINAMATH_CALUDE_equal_roots_condition_l3793_379366

/-- For a quadratic equation of the form (2kx^2 + Bx + 2) = 0 to have equal roots, B must be equal to 4√k -/
theorem equal_roots_condition (k : ℝ) (B : ℝ) :
  (∀ x : ℝ, (2 * k * x^2 + B * x + 2 = 0) → (∃! r : ℝ, 2 * k * r^2 + B * r + 2 = 0)) ↔ B = 4 * Real.sqrt k := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l3793_379366


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_seven_l3793_379344

-- Define the opposite function
def opposite (x : ℝ) : ℝ := -x

-- State the theorem
theorem opposite_of_sqrt_seven :
  opposite (Real.sqrt 7) = -(Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_seven_l3793_379344


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l3793_379368

-- Problem 1
theorem factorization_1 (a x y : ℝ) : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) := by
  sorry

-- Problem 2
theorem factorization_2 (x : ℝ) : x^3 - 8 * x^2 + 16 * x = x * (x - 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l3793_379368


namespace NUMINAMATH_CALUDE_square_root_2023_plus_2_squared_minus_4_times_plus_5_l3793_379326

theorem square_root_2023_plus_2_squared_minus_4_times_plus_5 :
  let m : ℝ := Real.sqrt 2023 + 2
  m^2 - 4*m + 5 = 2024 := by
sorry

end NUMINAMATH_CALUDE_square_root_2023_plus_2_squared_minus_4_times_plus_5_l3793_379326


namespace NUMINAMATH_CALUDE_converse_of_negative_square_positive_l3793_379320

theorem converse_of_negative_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, x^2 > 0 → x < 0) :=
sorry

end NUMINAMATH_CALUDE_converse_of_negative_square_positive_l3793_379320


namespace NUMINAMATH_CALUDE_wall_length_calculation_l3793_379388

theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 21 →
  wall_width = 28 →
  (mirror_side ^ 2) * 2 = wall_width * (31.5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_wall_length_calculation_l3793_379388


namespace NUMINAMATH_CALUDE_twenty_first_figure_squares_l3793_379384

/-- The number of squares in the nth figure of the sequence -/
def num_squares (n : ℕ) : ℕ := n^2 + (n-1)^2

/-- The theorem stating that the 21st figure has 841 squares -/
theorem twenty_first_figure_squares : num_squares 21 = 841 := by
  sorry

end NUMINAMATH_CALUDE_twenty_first_figure_squares_l3793_379384


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3793_379307

def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (x + 2, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors_x_value :
  ∃ (x : ℝ), x > 0 ∧ dot_product (vector_a x) (vector_b x) = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3793_379307


namespace NUMINAMATH_CALUDE_sum_of_first_100_terms_l3793_379345

/-- The optimal decomposition of a positive integer n is the product p × q
    where p and q are positive integers, p ≤ q, and |q - p| is minimized. -/
def OptimalDecomposition (n : ℕ+) : ℕ+ × ℕ+ :=
  sorry

/-- f(n) is defined as q - p where p × q is the optimal decomposition of n -/
def f (n : ℕ+) : ℕ :=
  let (p, q) := OptimalDecomposition n
  q - p

/-- The sequence a_n is defined as f(3^n) -/
def a (n : ℕ) : ℕ :=
  f (3^n : ℕ+)

/-- The sum of the first 100 terms of the sequence a_n -/
def S : ℕ :=
  (Finset.range 100).sum a

theorem sum_of_first_100_terms : S = 3^50 - 1 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_first_100_terms_l3793_379345


namespace NUMINAMATH_CALUDE_wire_cut_problem_l3793_379308

theorem wire_cut_problem (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) :
  total_length = 90 →
  ratio = 2 / 7 →
  shorter_length = ratio * (total_length - shorter_length) →
  shorter_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_problem_l3793_379308


namespace NUMINAMATH_CALUDE_boat_fuel_cost_is_50_l3793_379348

/-- The boat fuel cost per hour for Pat's shark hunting -/
def boat_fuel_cost_per_hour : ℚ :=
  let photo_earning : ℚ := 15
  let shark_interval : ℚ := 10 / 60  -- 10 minutes in hours
  let hunting_duration : ℚ := 5
  let expected_profit : ℚ := 200
  let total_sharks : ℚ := hunting_duration / shark_interval
  let total_earnings : ℚ := total_sharks * photo_earning
  let total_fuel_cost : ℚ := total_earnings - expected_profit
  total_fuel_cost / hunting_duration

/-- Theorem stating that the boat fuel cost per hour is $50 -/
theorem boat_fuel_cost_is_50 : boat_fuel_cost_per_hour = 50 := by
  sorry

end NUMINAMATH_CALUDE_boat_fuel_cost_is_50_l3793_379348


namespace NUMINAMATH_CALUDE_fraction_problem_l3793_379373

theorem fraction_problem (f : ℚ) : f = 1/3 → 0.75 * 264 = f * 264 + 110 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3793_379373


namespace NUMINAMATH_CALUDE_grade11_sample_count_l3793_379317

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  grade10Students : ℕ
  grade11Students : ℕ
  grade12Students : ℕ
  firstDrawn : ℕ

/-- Calculates the number of grade 11 students in the sample. -/
def grade11InSample (s : SystematicSampling) : ℕ :=
  let interval := s.totalStudents / s.sampleSize
  let grade11Start := s.grade10Students + 1
  let grade11End := grade11Start + s.grade11Students - 1
  let firstSampleInGrade11 := (((grade11Start - 1) / interval) * interval + s.firstDrawn - 1) / interval + 1
  let lastSampleInGrade11 := (((grade11End - 1) / interval) * interval + s.firstDrawn - 1) / interval + 1
  lastSampleInGrade11 - firstSampleInGrade11 + 1

/-- Theorem stating that for the given conditions, the number of grade 11 students in the sample is 17. -/
theorem grade11_sample_count (s : SystematicSampling) 
    (h1 : s.totalStudents = 1470)
    (h2 : s.sampleSize = 49)
    (h3 : s.grade10Students = 495)
    (h4 : s.grade11Students = 493)
    (h5 : s.grade12Students = 482)
    (h6 : s.firstDrawn = 23) :
    grade11InSample s = 17 := by
  sorry

end NUMINAMATH_CALUDE_grade11_sample_count_l3793_379317


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l3793_379369

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l3793_379369


namespace NUMINAMATH_CALUDE_fraction_power_equality_l3793_379329

theorem fraction_power_equality (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^y * y^x) / (y^y * x^x) = (x/y)^(y-x) := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l3793_379329


namespace NUMINAMATH_CALUDE_line_and_circle_problem_l3793_379328

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line3 (x y : ℝ) : Prop := x + y - 2 = 0

-- Define line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Theorem statement
theorem line_and_circle_problem :
  ∀ (x_int y_int : ℝ),
    (line1 x_int y_int ∧ line2 x_int y_int) →  -- Intersection point condition
    (∀ (x y : ℝ), line_l x y → (x + y - 2 ≠ 0)) →  -- Perpendicularity condition
    circle_C 1 0 →  -- Circle passes through (1,0)
    (∃ (a : ℝ), a > 0 ∧ circle_C a 0) →  -- Center on positive x-axis
    (∃ (x1 y1 x2 y2 : ℝ),
      line_l x1 y1 ∧ line_l x2 y2 ∧
      circle_C x1 y1 ∧ circle_C x2 y2 ∧
      (x2 - x1)^2 + (y2 - y1)^2 = 8) →  -- Chord length condition
    (∀ (x y : ℝ), line_l x y ↔ y = x - 1) ∧
    (∀ (x y : ℝ), circle_C x y ↔ (x - 3)^2 + y^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_line_and_circle_problem_l3793_379328


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3793_379316

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|x| - 2 = 1) ↔ (x = 3 ∨ x = -3) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3793_379316


namespace NUMINAMATH_CALUDE_inequality_solution_l3793_379306

theorem inequality_solution (x : ℝ) : 
  (12 * x^3 + 24 * x^2 - 75 * x - 3) / ((3 * x - 4) * (x + 5)) < 6 ↔ -5 < x ∧ x < 4/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3793_379306


namespace NUMINAMATH_CALUDE_edward_tickets_l3793_379321

/-- The number of tickets Edward spent at the 'dunk a clown' booth -/
def spent_tickets : ℕ := 23

/-- The cost of each ride in tickets -/
def ride_cost : ℕ := 7

/-- The number of rides Edward could have gone on with the remaining tickets -/
def possible_rides : ℕ := 8

/-- The total number of tickets Edward bought at the state fair -/
def total_tickets : ℕ := spent_tickets + ride_cost * possible_rides

theorem edward_tickets : total_tickets = 79 := by sorry

end NUMINAMATH_CALUDE_edward_tickets_l3793_379321


namespace NUMINAMATH_CALUDE_two_queens_or_one_king_probability_l3793_379367

-- Define a standard deck
def standard_deck : ℕ := 52

-- Define the number of queens in a deck
def num_queens : ℕ := 4

-- Define the number of kings in a deck
def num_kings : ℕ := 4

-- Define the number of cards drawn
def cards_drawn : ℕ := 3

-- Define the probability of the event
def event_probability : ℚ := 49 / 221

-- Theorem statement
theorem two_queens_or_one_king_probability :
  let p_two_queens := (num_queens / standard_deck) * ((num_queens - 1) / (standard_deck - 1))
  let p_no_kings := (standard_deck - num_kings) / standard_deck *
                    ((standard_deck - num_kings - 1) / (standard_deck - 1)) *
                    ((standard_deck - num_kings - 2) / (standard_deck - 2))
  let p_at_least_one_king := 1 - p_no_kings
  p_two_queens + p_at_least_one_king = event_probability :=
by sorry

end NUMINAMATH_CALUDE_two_queens_or_one_king_probability_l3793_379367


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3793_379302

theorem complex_equation_solution :
  ∀ (x y : ℝ), (1 + x * Complex.I) * (1 - 2 * Complex.I) = y → x = 2 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3793_379302


namespace NUMINAMATH_CALUDE_largest_fraction_sum_l3793_379380

theorem largest_fraction_sum : 
  let A := (3009 : ℚ) / 3008 + (3009 : ℚ) / 3010
  let B := (3011 : ℚ) / 3010 + (3011 : ℚ) / 3012
  let C := (3010 : ℚ) / 3009 + (3010 : ℚ) / 3011
  A > B ∧ A > C := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_sum_l3793_379380


namespace NUMINAMATH_CALUDE_real_solution_exists_l3793_379318

theorem real_solution_exists (x : ℝ) : ∃ y : ℝ, 9 * y^2 + 3 * x * y + x - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_real_solution_exists_l3793_379318


namespace NUMINAMATH_CALUDE_inscribed_triangle_sides_l3793_379385

/-- A triangle with an inscribed circle -/
structure InscribedTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first side of the triangle -/
  a : ℝ
  /-- The length of the second side of the triangle -/
  b : ℝ
  /-- The length of the third side of the triangle -/
  c : ℝ
  /-- The length of the first segment of side 'a' -/
  x : ℝ
  /-- The length of the second segment of side 'a' -/
  y : ℝ
  /-- The side 'a' is divided by the point of tangency -/
  side_division : a = x + y
  /-- All sides are positive -/
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  /-- All segments are positive -/
  pos_segments : 0 < x ∧ 0 < y

/-- Theorem about the sides of a triangle with an inscribed circle -/
theorem inscribed_triangle_sides (t : InscribedTriangle) 
  (h1 : t.r = 2)
  (h2 : t.x = 6)
  (h3 : t.y = 14) :
  (t.b = 7 ∧ t.c = 15) ∨ (t.b = 15 ∧ t.c = 7) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_sides_l3793_379385


namespace NUMINAMATH_CALUDE_opera_house_empty_seats_percentage_l3793_379371

theorem opera_house_empty_seats_percentage
  (total_rows : ℕ)
  (seats_per_row : ℕ)
  (ticket_price : ℕ)
  (earnings : ℕ)
  (h1 : total_rows = 150)
  (h2 : seats_per_row = 10)
  (h3 : ticket_price = 10)
  (h4 : earnings = 12000) :
  (((total_rows * seats_per_row) - (earnings / ticket_price)) * 100) / (total_rows * seats_per_row) = 20 := by
  sorry

#check opera_house_empty_seats_percentage

end NUMINAMATH_CALUDE_opera_house_empty_seats_percentage_l3793_379371


namespace NUMINAMATH_CALUDE_binary_110101_to_hex_35_l3793_379314

def binary_to_decimal (b : List Bool) : ℕ :=
  List.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0 b

def decimal_to_hexadecimal (n : ℕ) : String :=
  let rec aux (m : ℕ) (acc : String) : String :=
    if m = 0 then
      if acc.isEmpty then "0" else acc
    else
      let digit := m % 16
      let hex_digit := if digit < 10 then 
        Char.toString (Char.ofNat (digit + 48))
      else
        Char.toString (Char.ofNat (digit + 55))
      aux (m / 16) (hex_digit ++ acc)
  aux n ""

def binary_110101 : List Bool := [true, true, false, true, false, true]

theorem binary_110101_to_hex_35 :
  decimal_to_hexadecimal (binary_to_decimal binary_110101) = "35" :=
by sorry

end NUMINAMATH_CALUDE_binary_110101_to_hex_35_l3793_379314


namespace NUMINAMATH_CALUDE_time_after_2700_minutes_l3793_379341

-- Define a custom type for time
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

-- Define a function to add minutes to a given time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := (totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  { hours := newHours, minutes := newMinutes }

-- Define the starting time (6:00 a.m.)
def startTime : Time := { hours := 6, minutes := 0 }

-- Define the number of minutes to add
def minutesToAdd : Nat := 2700

-- Define the expected end time (3:00 a.m. the next day)
def expectedEndTime : Time := { hours := 3, minutes := 0 }

-- Theorem statement
theorem time_after_2700_minutes :
  addMinutes startTime minutesToAdd = expectedEndTime := by
  sorry

end NUMINAMATH_CALUDE_time_after_2700_minutes_l3793_379341


namespace NUMINAMATH_CALUDE_student_arrangements_eq_20_l3793_379378

/-- The number of ways to arrange 7 students of different heights in a row,
    with the tallest in the middle and the others decreasing in height towards both ends. -/
def student_arrangements : ℕ :=
  Nat.choose 6 3

/-- Theorem stating that the number of student arrangements is 20. -/
theorem student_arrangements_eq_20 : student_arrangements = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangements_eq_20_l3793_379378


namespace NUMINAMATH_CALUDE_sugar_solution_replacement_l3793_379337

theorem sugar_solution_replacement (original_sugar_percent : ℝ) 
                                   (second_sugar_percent : ℝ) 
                                   (final_sugar_percent : ℝ) :
  original_sugar_percent = 10 →
  second_sugar_percent = 26.000000000000007 →
  final_sugar_percent = 14 →
  ∃ (x : ℝ), x = 0.25 ∧ 
             (1 - x) * (original_sugar_percent / 100) + 
             x * (second_sugar_percent / 100) = 
             final_sugar_percent / 100 :=
by sorry

end NUMINAMATH_CALUDE_sugar_solution_replacement_l3793_379337


namespace NUMINAMATH_CALUDE_sum_formula_and_difference_l3793_379364

def f (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (3 * n - 2)

theorem sum_formula_and_difference (n k : ℕ) (h : n > 0) (h' : k > 0) : 
  f n = (2 * n - 1)^2 ∧ f (k + 1) - f k = 8 * k := by sorry

end NUMINAMATH_CALUDE_sum_formula_and_difference_l3793_379364


namespace NUMINAMATH_CALUDE_soap_brands_survey_l3793_379393

theorem soap_brands_survey (total : Nat) (neither : Nat) (only_a : Nat) (both : Nat) : 
  total = 300 →
  neither = 80 →
  only_a = 60 →
  total = neither + only_a + 3 * both + both →
  both = 40 := by
sorry

end NUMINAMATH_CALUDE_soap_brands_survey_l3793_379393


namespace NUMINAMATH_CALUDE_value_two_std_dev_below_mean_l3793_379331

-- Define the properties of the normal distribution
def mean : ℝ := 16.2
def std_dev : ℝ := 2.3

-- Define the value we're looking for
def value : ℝ := mean - 2 * std_dev

-- Theorem stating that the value is 11.6
theorem value_two_std_dev_below_mean :
  value = 11.6 := by sorry

end NUMINAMATH_CALUDE_value_two_std_dev_below_mean_l3793_379331


namespace NUMINAMATH_CALUDE_tylers_dogs_l3793_379340

/-- Proves that Tyler had 15 dogs initially, given the conditions of the problem -/
theorem tylers_dogs : ∀ (initial_dogs : ℕ), 
  (initial_dogs * 5 = 75) → initial_dogs = 15 := by
  sorry

end NUMINAMATH_CALUDE_tylers_dogs_l3793_379340


namespace NUMINAMATH_CALUDE_polynomial_sum_l3793_379372

theorem polynomial_sum (x : ℝ) (h1 : x^5 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^4 + x^3 + x^2 + x + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3793_379372


namespace NUMINAMATH_CALUDE_distance_point_to_line_l3793_379377

def vector_AB : Fin 3 → ℝ := ![1, 1, 2]
def vector_AC : Fin 3 → ℝ := ![2, 1, 1]

theorem distance_point_to_line :
  let distance := Real.sqrt (6 - (5 * Real.sqrt 6 / 6) ^ 2)
  distance = Real.sqrt 66 / 6 := by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l3793_379377


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l3793_379360

/-- A parabola in 2D space -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an intersection point between a parabola and a circle -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is a tangent point between a parabola and a circle -/
def is_tangent_point (p : Parabola) (c : Circle) (point : IntersectionPoint) : Prop :=
  sorry

/-- Theorem stating that if a circle and a parabola intersect at exactly two points,
    and one is a tangent point, then the other must also be a tangent point -/
theorem parabola_circle_tangency
  (p : Parabola) (c : Circle) 
  (i1 i2 : IntersectionPoint) 
  (h_distinct : i1 ≠ i2)
  (h_only_two : ∀ i : IntersectionPoint, i = i1 ∨ i = i2)
  (h_tangent : is_tangent_point p c i1) :
  is_tangent_point p c i2 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l3793_379360


namespace NUMINAMATH_CALUDE_derivative_at_one_l3793_379327

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l3793_379327


namespace NUMINAMATH_CALUDE_binomial_26_7_l3793_379311

theorem binomial_26_7 (h1 : Nat.choose 24 5 = 42504)
                      (h2 : Nat.choose 24 6 = 134596)
                      (h3 : Nat.choose 24 7 = 346104) :
  Nat.choose 26 7 = 657800 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_7_l3793_379311


namespace NUMINAMATH_CALUDE_inequality_proof_l3793_379361

theorem inequality_proof (r s : ℝ) (hr : 0 < r) (hs : 0 < s) (hrs : r + s = 1) :
  r^r * s^s + r^s * s^r ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3793_379361


namespace NUMINAMATH_CALUDE_max_guests_left_l3793_379347

/-- Represents a guest with their galoshes size -/
structure Guest where
  size : ℕ

/-- Represents the state of galoshes in the hallway -/
structure GaloshesState where
  sizes : Finset ℕ

/-- Defines when a guest can wear a pair of galoshes -/
def canWear (g : Guest) (s : ℕ) : Prop := g.size ≤ s

/-- Defines the initial state with 10 guests and their galoshes -/
def initialState : Finset Guest × GaloshesState :=
  sorry

/-- Simulates guests leaving and wearing galoshes -/
def guestsLeave (state : Finset Guest × GaloshesState) : Finset Guest × GaloshesState :=
  sorry

/-- Checks if any remaining guest can wear any remaining galoshes -/
def canAnyGuestLeave (state : Finset Guest × GaloshesState) : Prop :=
  sorry

theorem max_guests_left (final_state : Finset Guest × GaloshesState) :
  final_state = guestsLeave initialState →
  ¬canAnyGuestLeave final_state →
  Finset.card final_state.1 ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_guests_left_l3793_379347


namespace NUMINAMATH_CALUDE_yellow_paint_theorem_l3793_379343

/-- Represents the ratio of paints in the mixture -/
structure PaintRatio :=
  (blue : ℚ)
  (yellow : ℚ)
  (white : ℚ)

/-- Calculates the amount of yellow paint needed given the amount of white paint and the ratio -/
def yellow_paint_amount (ratio : PaintRatio) (white_amount : ℚ) : ℚ :=
  (ratio.yellow / ratio.white) * white_amount

/-- Theorem stating that given the specific ratio and white paint amount, 
    the yellow paint amount should be 9 quarts -/
theorem yellow_paint_theorem (ratio : PaintRatio) (white_amount : ℚ) :
  ratio.blue = 4 ∧ ratio.yellow = 3 ∧ ratio.white = 5 ∧ white_amount = 15 →
  yellow_paint_amount ratio white_amount = 9 := by
  sorry


end NUMINAMATH_CALUDE_yellow_paint_theorem_l3793_379343


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3793_379313

theorem smallest_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 6 = 3) ∧ 
  (x % 8 = 2) ∧ 
  (∀ y : ℕ, y > 0 ∧ y % 6 = 3 ∧ y % 8 = 2 → x ≤ y) ∧
  (x = 33) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3793_379313


namespace NUMINAMATH_CALUDE_min_value_n_over_2_plus_50_over_n_l3793_379325

theorem min_value_n_over_2_plus_50_over_n (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 2 + 50 / n ≥ 10 ∧
  ((n : ℝ) / 2 + 50 / n = 10 ↔ n = 10) := by
  sorry

end NUMINAMATH_CALUDE_min_value_n_over_2_plus_50_over_n_l3793_379325


namespace NUMINAMATH_CALUDE_classroom_children_count_l3793_379386

theorem classroom_children_count :
  ∀ (total_children : ℕ),
  (total_children : ℚ) / 3 = total_children - 30 →
  30 ≤ total_children →
  total_children = 45 := by
sorry

end NUMINAMATH_CALUDE_classroom_children_count_l3793_379386


namespace NUMINAMATH_CALUDE_percentage_difference_l3793_379390

theorem percentage_difference (x y z : ℝ) : 
  x = 1.2 * y ∧ x = 0.36 * z → y = 0.3 * z :=
by sorry

end NUMINAMATH_CALUDE_percentage_difference_l3793_379390


namespace NUMINAMATH_CALUDE_sum_of_legs_special_triangle_l3793_379323

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  a : ℕ  -- shorter leg
  b : ℕ  -- longer leg
  c : ℕ  -- hypotenuse
  right_angle : a^2 + b^2 = c^2
  consecutive_even : b = a + 2

/-- The sum of legs of a right triangle with hypotenuse 50 and consecutive even legs is 70 -/
theorem sum_of_legs_special_triangle :
  ∀ (t : RightTriangle), t.c = 50 → t.a + t.b = 70 := by
  sorry

#check sum_of_legs_special_triangle

end NUMINAMATH_CALUDE_sum_of_legs_special_triangle_l3793_379323


namespace NUMINAMATH_CALUDE_value_of_m_l3793_379356

theorem value_of_m (m : ℕ) : 
  (((1 : ℚ) ^ m) / (5 ^ m)) * (((1 : ℚ) ^ 16) / (4 ^ 16)) = 1 / (2 * (10 ^ 31)) → 
  m = 31 := by
sorry

end NUMINAMATH_CALUDE_value_of_m_l3793_379356


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3793_379309

theorem polynomial_simplification (x : ℝ) :
  (2 * x^2 + 5*x - 4) - (x^2 - 2*x + 1) + (3 * x^2 + 4*x - 7) = 4 * x^2 + 11*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3793_379309


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_ratio_l3793_379391

/-- An ellipse intersecting a line with a specific midpoint property -/
structure EllipseLineIntersection where
  m : ℝ
  n : ℝ
  -- Ellipse equation: mx^2 + ny^2 = 1
  -- Line equation: x + y - 1 = 0
  -- Intersection points exist (implicit)
  -- Line through midpoint and origin has slope √2/2 (implicit)

/-- The ratio m/n equals √2/2 for the given ellipse-line intersection -/
theorem ellipse_line_intersection_ratio (e : EllipseLineIntersection) : e.m / e.n = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_ratio_l3793_379391


namespace NUMINAMATH_CALUDE_fraction_simplification_l3793_379363

theorem fraction_simplification :
  (252 : ℚ) / 18 * 7 / 189 * 9 / 4 = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3793_379363


namespace NUMINAMATH_CALUDE_total_jars_is_24_l3793_379301

/-- Represents the number of each type of jar -/
def num_each_jar : ℕ := 8

/-- Represents the total volume of water in gallons -/
def total_water : ℕ := 14

/-- Represents the volume of water in quarts held by all quart jars -/
def quart_jars_volume : ℕ := num_each_jar

/-- Represents the volume of water in quarts held by all half-gallon jars -/
def half_gallon_jars_volume : ℕ := 2 * num_each_jar

/-- Represents the volume of water in quarts held by all one-gallon jars -/
def gallon_jars_volume : ℕ := 4 * num_each_jar

/-- Theorem stating that the total number of water-filled jars is 24 -/
theorem total_jars_is_24 : 
  quart_jars_volume + half_gallon_jars_volume + gallon_jars_volume = total_water * 4 ∧
  3 * num_each_jar = 24 := by
  sorry

#check total_jars_is_24

end NUMINAMATH_CALUDE_total_jars_is_24_l3793_379301


namespace NUMINAMATH_CALUDE_meet_five_times_l3793_379333

/-- Represents the meeting problem between Michael and the garbage truck --/
structure MeetingProblem where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (problem : MeetingProblem) : ℕ :=
  sorry

/-- The theorem stating that Michael and the truck meet exactly 5 times --/
theorem meet_five_times (problem : MeetingProblem) 
  (h1 : problem.michael_speed = 5)
  (h2 : problem.truck_speed = 10)
  (h3 : problem.pail_distance = 200)
  (h4 : problem.truck_stop_time = 30) :
  number_of_meetings problem = 5 :=
  sorry

end NUMINAMATH_CALUDE_meet_five_times_l3793_379333


namespace NUMINAMATH_CALUDE_slope_sum_constant_l3793_379365

-- Define the curve C
def C (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the line l passing through (2, 0)
def l (k x y : ℝ) : Prop := y = k * (x - 2)

-- Define point A
def A : ℝ × ℝ := (-3, 0)

-- Define the theorem
theorem slope_sum_constant 
  (k k₁ k₂ x₁ y₁ x₂ y₂ : ℝ) 
  (hM : C x₁ y₁ ∧ l k x₁ y₁) 
  (hN : C x₂ y₂ ∧ l k x₂ y₂) 
  (hk₁ : k₁ = y₁ / (x₁ + 3)) 
  (hk₂ : k₂ = y₂ / (x₂ + 3)) :
  k / k₁ + k / k₂ = -1/2 := by sorry

end NUMINAMATH_CALUDE_slope_sum_constant_l3793_379365


namespace NUMINAMATH_CALUDE_percentage_less_than_l3793_379319

theorem percentage_less_than (w x y z P : ℝ) : 
  w = x * (1 - P / 100) →
  x = y * 0.6 →
  z = y * 0.54 →
  z = w * 1.5 →
  P = 40 :=
by sorry

end NUMINAMATH_CALUDE_percentage_less_than_l3793_379319


namespace NUMINAMATH_CALUDE_tom_teaching_years_l3793_379303

theorem tom_teaching_years (total_years devin_years tom_years : ℕ) : 
  total_years = 70 →
  devin_years = tom_years / 2 - 5 →
  total_years = tom_years + devin_years →
  tom_years = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_tom_teaching_years_l3793_379303


namespace NUMINAMATH_CALUDE_floor_times_self_equals_45_l3793_379304

theorem floor_times_self_equals_45 (y : ℝ) (h1 : y > 0) (h2 : ⌊y⌋ * y = 45) : y = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_equals_45_l3793_379304


namespace NUMINAMATH_CALUDE_min_value_expression_l3793_379336

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 6*a*b + 9*b^2 + 4*c^2 ≥ 180 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ * b₀ * c₀ = 27 ∧ 
    a₀^2 + 6*a₀*b₀ + 9*b₀^2 + 4*c₀^2 = 180 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3793_379336


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3793_379379

/-- A hyperbola is represented by an equation of the form a*x² + b*y² = c, where a and b have opposite signs and c ≠ 0 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a ≠ 0
  h2 : b ≠ 0
  h3 : c ≠ 0
  h4 : a * b < 0

/-- The equation x²/(9-k) + y²/(k-4) = 1 -/
def equation (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (9 - k) + y^2 / (k - 4) = 1

/-- The condition k > 9 is sufficient but not necessary for the equation to represent a hyperbola -/
theorem sufficient_not_necessary (k : ℝ) :
  (k > 9 → ∃ h : Hyperbola, equation k = λ x y ↦ h.a * x^2 + h.b * y^2 = h.c) ∧
  ¬(∀ h : Hyperbola, equation k = λ x y ↦ h.a * x^2 + h.b * y^2 = h.c → k > 9) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3793_379379


namespace NUMINAMATH_CALUDE_parabola_properties_l3793_379355

-- Define the parabola and its coefficients
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the points that the parabola passes through
def point_A : ℝ × ℝ := (-1, 0)
def point_C : ℝ × ℝ := (0, 3)
def point_B : ℝ × ℝ := (2, -3)

-- Theorem stating the properties of the parabola
theorem parabola_properties :
  ∃ (a b c : ℝ),
    -- The parabola passes through the given points
    (parabola a b c (point_A.1) = point_A.2) ∧
    (parabola a b c (point_C.1) = point_C.2) ∧
    (parabola a b c (point_B.1) = point_B.2) ∧
    -- The parabola equation is y = -2x² + x + 3
    (a = -2 ∧ b = 1 ∧ c = 3) ∧
    -- The axis of symmetry is x = 1/4
    (- b / (2 * a) = 1 / 4) ∧
    -- The vertex coordinates are (1/4, 25/8)
    (parabola a b c (1 / 4) = 25 / 8) := by
  sorry


end NUMINAMATH_CALUDE_parabola_properties_l3793_379355


namespace NUMINAMATH_CALUDE_smallest_k_for_three_reals_l3793_379339

theorem smallest_k_for_three_reals : ∃ (k : ℝ),
  (∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
    (|x - y| ≤ k ∨ |1/x - 1/y| ≤ k) ∨
    (|y - z| ≤ k ∨ |1/y - 1/z| ≤ k) ∨
    (|x - z| ≤ k ∨ |1/x - 1/z| ≤ k)) ∧
  (∀ (k' : ℝ), k' < k →
    ∃ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
      (|x - y| > k' ∧ |1/x - 1/y| > k') ∧
      (|y - z| > k' ∧ |1/y - 1/z| > k') ∧
      (|x - z| > k' ∧ |1/x - 1/z| > k')) ∧
  k = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_three_reals_l3793_379339


namespace NUMINAMATH_CALUDE_caroline_lassis_l3793_379334

/-- The number of lassis Caroline can make from a given number of mangoes -/
def lassis_from_mangoes (mangoes : ℕ) : ℕ :=
  8 * mangoes / 3

/-- Theorem stating that Caroline can make 40 lassis from 15 mangoes -/
theorem caroline_lassis : lassis_from_mangoes 15 = 40 := by
  sorry

end NUMINAMATH_CALUDE_caroline_lassis_l3793_379334


namespace NUMINAMATH_CALUDE_concert_revenue_l3793_379342

/-- Calculate the total revenue from concert ticket sales --/
theorem concert_revenue (ticket_price : ℝ) (first_discount : ℝ) (second_discount : ℝ)
  (first_group : ℕ) (second_group : ℕ) (total_people : ℕ) :
  ticket_price = 20 →
  first_discount = 0.4 →
  second_discount = 0.15 →
  first_group = 10 →
  second_group = 20 →
  total_people = 45 →
  (first_group * ticket_price * (1 - first_discount) +
   second_group * ticket_price * (1 - second_discount) +
   (total_people - first_group - second_group) * ticket_price) = 760 := by
sorry

end NUMINAMATH_CALUDE_concert_revenue_l3793_379342


namespace NUMINAMATH_CALUDE_not_always_geometric_sequence_l3793_379315

def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ+, a (n + 1) = q * a n

theorem not_always_geometric_sequence :
  ¬ (∀ a : ℕ+ → ℝ, (∃ q : ℝ, ∀ n : ℕ+, a (n + 1) = q * a n) → is_geometric_sequence a) :=
by
  sorry

end NUMINAMATH_CALUDE_not_always_geometric_sequence_l3793_379315


namespace NUMINAMATH_CALUDE_expression_evaluation_l3793_379354

theorem expression_evaluation :
  ∀ x y : ℝ,
  (|x| = 2) →
  (y = 1) →
  (x * y < 0) →
  3 * x^2 * y - 2 * x^2 - (x * y)^2 - 3 * x^2 * y - 4 * (x * y)^2 = -18 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3793_379354


namespace NUMINAMATH_CALUDE_circle_m_range_l3793_379357

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Checks if a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 > c.radius^2

/-- The circle equation in the form x^2 + y^2 - 2x + 1 - m = 0 -/
def circleEquation (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 1 - m = 0

theorem circle_m_range :
  ∀ m : ℝ,
  (∃ c : Circle, 
    (∀ x y : ℝ, circleEquation m x y ↔ (x - c.center.x)^2 + (y - c.center.y)^2 = c.radius^2) ∧
    isOutside ⟨1, 1⟩ c) →
  0 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_circle_m_range_l3793_379357


namespace NUMINAMATH_CALUDE_max_sin_a_value_l3793_379300

theorem max_sin_a_value (a b c : Real) 
  (h1 : Real.cos a = Real.tan b)
  (h2 : Real.cos b = Real.tan c)
  (h3 : Real.cos c = Real.tan a) :
  ∃ (max_sin_a : Real), 
    (∀ a' b' c' : Real, 
      Real.cos a' = Real.tan b' → 
      Real.cos b' = Real.tan c' → 
      Real.cos c' = Real.tan a' → 
      Real.sin a' ≤ max_sin_a) ∧
    max_sin_a = Real.sqrt ((3 - Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_sin_a_value_l3793_379300


namespace NUMINAMATH_CALUDE_min_ratio_two_digit_integers_l3793_379312

theorem min_ratio_two_digit_integers (x y : ℕ) : 
  x ≥ 10 ∧ x ≤ 99 ∧ y ≥ 10 ∧ y ≤ 99 →  -- x and y are two-digit integers
  (x + y) / 2 = 65 →                   -- mean is 65
  x > 50 ∧ y > 50 →                    -- x and y are both greater than 50
  ∀ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (a + b) / 2 = 65 ∧ a > 50 ∧ b > 50 →
  (a : ℚ) / b ≥ (51 : ℚ) / 79 →
  (x : ℚ) / y ≥ (51 : ℚ) / 79 := by
sorry

end NUMINAMATH_CALUDE_min_ratio_two_digit_integers_l3793_379312


namespace NUMINAMATH_CALUDE_sophies_money_correct_l3793_379350

/-- The amount of money Sophie's aunt gave her --/
def sophies_money : ℝ := 260

/-- The cost of one shirt --/
def shirt_cost : ℝ := 18.50

/-- The number of shirts Sophie bought --/
def num_shirts : ℕ := 2

/-- The cost of the trousers --/
def trouser_cost : ℝ := 63

/-- The cost of one additional article of clothing --/
def additional_item_cost : ℝ := 40

/-- The number of additional articles of clothing Sophie plans to buy --/
def num_additional_items : ℕ := 4

/-- Theorem stating that the amount of money Sophie's aunt gave her is correct --/
theorem sophies_money_correct : 
  sophies_money = 
    shirt_cost * num_shirts + 
    trouser_cost + 
    additional_item_cost * num_additional_items := by
  sorry

end NUMINAMATH_CALUDE_sophies_money_correct_l3793_379350


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l3793_379397

/-- 
Theorem: For the quadratic equation x^2 - 6x + k = 0 to have two distinct real roots, k must be less than 9.
-/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6*x + k = 0 ∧ y^2 - 6*y + k = 0) → k < 9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l3793_379397


namespace NUMINAMATH_CALUDE_cd_cost_l3793_379358

/-- Given that two identical CDs cost $24, prove that seven CDs cost $84. -/
theorem cd_cost (cost_of_two : ℕ) (h : cost_of_two = 24) : 7 * (cost_of_two / 2) = 84 := by
  sorry

end NUMINAMATH_CALUDE_cd_cost_l3793_379358


namespace NUMINAMATH_CALUDE_tangent_line_parallel_l3793_379305

/-- Given a curve y = 3x^2 + 2x with a tangent line at (1, 5) parallel to 2ax - y - 6 = 0, 
    the value of a is 4. -/
theorem tangent_line_parallel (a : ℝ) : 
  (∃ (f : ℝ → ℝ) (line : ℝ → ℝ), 
    (∀ x, f x = 3 * x^2 + 2 * x) ∧ 
    (∀ x, line x = 2 * a * x - 6) ∧
    (∃ (tangent : ℝ → ℝ), 
      (tangent 1 = f 1) ∧
      (∀ h : ℝ, h ≠ 0 → (tangent (1 + h) - tangent 1) / h = (line (1 + h) - line 1) / h))) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_l3793_379305


namespace NUMINAMATH_CALUDE_last_three_digits_of_2_to_9000_l3793_379389

theorem last_three_digits_of_2_to_9000 (h : 2^300 ≡ 1 [ZMOD 1000]) :
  2^9000 ≡ 1 [ZMOD 1000] := by
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_2_to_9000_l3793_379389


namespace NUMINAMATH_CALUDE_cube_root_equality_l3793_379370

theorem cube_root_equality (a b : ℝ) :
  (a ^ (1/3 : ℝ) = -(b ^ (1/3 : ℝ))) → a = -b := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equality_l3793_379370


namespace NUMINAMATH_CALUDE_squirrels_and_nuts_l3793_379396

theorem squirrels_and_nuts (squirrels : ℕ) (nuts : ℕ) : 
  squirrels = 4 → squirrels - nuts = 2 → nuts = 2 := by
  sorry

end NUMINAMATH_CALUDE_squirrels_and_nuts_l3793_379396


namespace NUMINAMATH_CALUDE_volume_region_equivalence_l3793_379335

theorem volume_region_equivalence (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  |x + 2*y + z| + |x - y - z| ≤ 10 ↔ max (x + 2*y + z) (x - y - z) ≤ 5 := by
  sorry

#check volume_region_equivalence

end NUMINAMATH_CALUDE_volume_region_equivalence_l3793_379335


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3793_379353

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 8) :
  a 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3793_379353


namespace NUMINAMATH_CALUDE_sector_area_l3793_379387

/-- Given a sector with radius 8 cm and central angle 45°, its area is 8π cm². -/
theorem sector_area (r : ℝ) (θ : ℝ) : 
  r = 8 → θ = 45 * (π / 180) → (1/2) * r^2 * θ = 8 * π := by sorry

end NUMINAMATH_CALUDE_sector_area_l3793_379387


namespace NUMINAMATH_CALUDE_alcohol_dilution_l3793_379330

/-- Proves that adding 3 litres of water to a 20-litre mixture containing 20% alcohol
    results in a new mixture with 17.391304347826086% alcohol. -/
theorem alcohol_dilution (original_volume : ℝ) (original_alcohol_percentage : ℝ) 
    (added_water : ℝ) (new_alcohol_percentage : ℝ) : 
    original_volume = 20 →
    original_alcohol_percentage = 0.20 →
    added_water = 3 →
    new_alcohol_percentage = 0.17391304347826086 →
    (original_volume * original_alcohol_percentage) / (original_volume + added_water) = new_alcohol_percentage :=
by sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l3793_379330


namespace NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_prop_4_false_l3793_379398

-- Define the function f
def f (x b c : ℝ) : ℝ := x * abs x + b * x + c

-- Proposition ①
theorem prop_1 (x : ℝ) : f x 0 0 = -f (-x) 0 0 := by sorry

-- Proposition ②
theorem prop_2 : ∃! x : ℝ, f x 0 1 = 0 := by sorry

-- Proposition ③
theorem prop_3 (x b c : ℝ) : f x b c - c = -(f (-x) b c - c) := by sorry

-- Proposition ④ (false)
theorem prop_4_false : ∃ b c : ℝ, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0 := by sorry

end NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_prop_4_false_l3793_379398


namespace NUMINAMATH_CALUDE_square_plus_self_even_l3793_379322

theorem square_plus_self_even (n : ℤ) : ∃ k : ℤ, n^2 + n = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_square_plus_self_even_l3793_379322


namespace NUMINAMATH_CALUDE_discount_per_shirt_calculation_l3793_379324

theorem discount_per_shirt_calculation (num_shirts : ℕ) (total_cost discount_percentage : ℚ) 
  (h1 : num_shirts = 3)
  (h2 : total_cost = 60)
  (h3 : discount_percentage = 40/100) :
  let discount_amount := total_cost * discount_percentage
  let discounted_total := total_cost - discount_amount
  let price_per_shirt := discounted_total / num_shirts
  price_per_shirt = 12 := by
sorry

end NUMINAMATH_CALUDE_discount_per_shirt_calculation_l3793_379324


namespace NUMINAMATH_CALUDE_problem_solution_l3793_379375

theorem problem_solution (x y z w : ℕ+) 
  (h1 : x^3 = y^4)
  (h2 : z^5 = w^2)
  (h3 : z - x = 31) :
  (w : ℤ) - y = -759439 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3793_379375


namespace NUMINAMATH_CALUDE_max_m_value_min_quadratic_sum_l3793_379359

-- Part 1
theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x + m| ≥ 2*m) → m ≤ 1 :=
sorry

-- Part 2
theorem min_quadratic_sum {a b c : ℝ} 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_one : a + b + c = 1) :
  2*a^2 + 3*b^2 + 4*c^2 ≥ 12/13 ∧ 
  (2*a^2 + 3*b^2 + 4*c^2 = 12/13 ↔ a = 6/13 ∧ b = 4/13 ∧ c = 3/13) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_min_quadratic_sum_l3793_379359


namespace NUMINAMATH_CALUDE_second_part_distance_l3793_379374

/-- Given a rowing trip with three parts, prove that the distance of the second part is 15 miles. -/
theorem second_part_distance (total distance_1 distance_2 distance_3 : ℝ) 
  (h1 : distance_1 = 6)
  (h2 : distance_3 = 18)
  (h3 : total = 39)
  (h4 : total = distance_1 + distance_2 + distance_3) :
  distance_2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_part_distance_l3793_379374


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l3793_379332

theorem modulo_eleven_residue : (341 + 6 * 50 + 4 * 156 + 3 * 12^2) % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l3793_379332


namespace NUMINAMATH_CALUDE_farmer_goats_and_sheep_l3793_379346

theorem farmer_goats_and_sheep :
  ∃! (g h : ℕ), 
    28 * g + 30 * h = 1200 ∧ 
    h > g ∧ 
    g > 0 ∧ 
    h > 0 :=
by sorry

end NUMINAMATH_CALUDE_farmer_goats_and_sheep_l3793_379346


namespace NUMINAMATH_CALUDE_wine_price_increase_l3793_379394

/-- Proves that the percentage increase in wine price is 25% given the initial and future prices -/
theorem wine_price_increase (initial_price : ℝ) (future_price_increase : ℝ) : 
  initial_price = 20 →
  future_price_increase = 25 →
  (((initial_price + future_price_increase / 5) - initial_price) / initial_price) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_wine_price_increase_l3793_379394


namespace NUMINAMATH_CALUDE_angle_PSQ_measure_l3793_379392

-- Define the points
variable (K L M N P Q S : Point) (ω : Circle)

-- Define the trapezoid
def is_trapezoid (K L M N : Point) : Prop := sorry

-- Define the circle passing through L and M
def circle_through (ω : Circle) (L M : Point) : Prop := sorry

-- Define the circle intersecting KL at P and MN at Q
def circle_intersects (ω : Circle) (K L M N P Q : Point) : Prop := sorry

-- Define the circle tangent to KN at S
def circle_tangent_at (ω : Circle) (K N S : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (A B C : Point) : ℝ := sorry

-- State the theorem
theorem angle_PSQ_measure 
  (h_trapezoid : is_trapezoid K L M N)
  (h_circle_through : circle_through ω L M)
  (h_circle_intersects : circle_intersects ω K L M N P Q)
  (h_circle_tangent : circle_tangent_at ω K N S)
  (h_angle_LSM : angle_measure L S M = 50)
  (h_angle_equal : angle_measure K L S = angle_measure S N M) :
  angle_measure P S Q = 65 := by sorry

end NUMINAMATH_CALUDE_angle_PSQ_measure_l3793_379392


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l3793_379310

/-- A quadratic function with specific properties -/
def f : ℝ → ℝ := sorry

/-- The properties of the quadratic function -/
axiom f_prop1 : f (-1) = 0
axiom f_prop2 : f 4 = 0
axiom f_prop3 : f 0 = 4

/-- The decreasing interval of f -/
def decreasing_interval : Set ℝ := {x | x ≥ 3/2}

/-- Theorem stating that the given set is the decreasing interval of f -/
theorem f_decreasing_interval : 
  ∀ x ∈ decreasing_interval, ∀ y ∈ decreasing_interval, x < y → f x > f y :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l3793_379310


namespace NUMINAMATH_CALUDE_multiplicative_inverse_143_mod_391_l3793_379395

theorem multiplicative_inverse_143_mod_391 :
  ∃ a : ℕ, a < 391 ∧ (143 * a) % 391 = 1 :=
by
  use 28
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_143_mod_391_l3793_379395


namespace NUMINAMATH_CALUDE_range_of_a_l3793_379349

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + 2*x + a > 0) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3793_379349


namespace NUMINAMATH_CALUDE_carol_trivia_score_l3793_379338

/-- Represents Carol's trivia game scores -/
structure TriviaGame where
  round1 : Int
  round2 : Int
  round3 : Int

/-- Calculates the total score of a trivia game -/
def totalScore (game : TriviaGame) : Int :=
  game.round1 + game.round2 + game.round3

/-- Theorem: Carol's total score at the end of the game is 7 points -/
theorem carol_trivia_score :
  ∃ (game : TriviaGame), game.round1 = 17 ∧ game.round2 = 6 ∧ game.round3 = -16 ∧ totalScore game = 7 := by
  sorry

end NUMINAMATH_CALUDE_carol_trivia_score_l3793_379338


namespace NUMINAMATH_CALUDE_distance_from_pole_to_line_l3793_379351

/-- Given a line with polar equation ρ sin(θ + π/4) = 1, 
    the distance from the pole to this line is 1. -/
theorem distance_from_pole_to_line (ρ θ : ℝ) : 
  ρ * Real.sin (θ + π/4) = 1 → 
  (∃ d : ℝ, d = 1 ∧ d = abs (2) / Real.sqrt (2 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_distance_from_pole_to_line_l3793_379351


namespace NUMINAMATH_CALUDE_marcus_initial_cards_l3793_379399

/-- The number of baseball cards Carter gave to Marcus -/
def cards_from_carter : ℝ := 58.0

/-- The total number of baseball cards Marcus has after receiving cards from Carter -/
def total_cards : ℕ := 268

/-- The initial number of baseball cards Marcus had -/
def initial_cards : ℕ := 210

theorem marcus_initial_cards : 
  (total_cards : ℝ) - cards_from_carter = initial_cards := by sorry

end NUMINAMATH_CALUDE_marcus_initial_cards_l3793_379399


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3793_379383

theorem fraction_equivalence : ∃ n : ℤ, (2 + n) / (7 + n) = 3 / 4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3793_379383


namespace NUMINAMATH_CALUDE_problem_statement_l3793_379382

theorem problem_statement (a b : ℝ) (h : |a - 3| + (b + 2)^2 = 0) :
  (a + b)^2023 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3793_379382


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3793_379352

/-- Given a polynomial Q with Q(25) = 50 and Q(50) = 25, 
    the remainder when Q is divided by (x - 25)(x - 50) is -x + 75 -/
theorem polynomial_remainder (Q : ℝ → ℝ) (h1 : Q 25 = 50) (h2 : Q 50 = 25) :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 25) * (x - 50) * R x + (-x + 75) :=
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3793_379352


namespace NUMINAMATH_CALUDE_largest_number_in_l_pattern_l3793_379376

/-- Represents the L-shaped pattern in the number arrangement --/
structure LPattern where
  largest : ℕ
  second : ℕ
  third : ℕ

/-- The sum of numbers in the L-shaped pattern is 2015 --/
def sum_is_2015 (p : LPattern) : Prop :=
  p.largest + p.second + p.third = 2015

/-- The L-shaped pattern follows the specific arrangement described --/
def valid_arrangement (p : LPattern) : Prop :=
  (p.second = p.largest - 6 ∧ p.third = p.largest - 7) ∨
  (p.second = p.largest - 7 ∧ p.third = p.largest - 8) ∨
  (p.second = p.largest - 1 ∧ p.third = p.largest - 7) ∨
  (p.second = p.largest - 1 ∧ p.third = p.largest - 8)

theorem largest_number_in_l_pattern :
  ∀ p : LPattern, sum_is_2015 p → valid_arrangement p → p.largest = 676 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_l_pattern_l3793_379376
