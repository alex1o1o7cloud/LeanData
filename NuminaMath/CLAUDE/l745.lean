import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l745_74527

theorem geometric_sequence_problem (b : ℝ) : 
  b > 0 ∧ 
  (∃ r : ℝ, 125 * r = b ∧ b * r = 60 / 49) → 
  b = 50 * Real.sqrt 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l745_74527


namespace NUMINAMATH_CALUDE_two_segment_trip_average_speed_l745_74531

/-- Calculates the average speed of a two-segment trip -/
def average_speed (d1 d2 v1 v2 : ℚ) : ℚ :=
  (d1 + d2) / (d1 / v1 + d2 / v2)

theorem two_segment_trip_average_speed :
  let d1 : ℚ := 50
  let d2 : ℚ := 25
  let v1 : ℚ := 15
  let v2 : ℚ := 45
  average_speed d1 d2 v1 v2 = 675 / 35 := by
  sorry

end NUMINAMATH_CALUDE_two_segment_trip_average_speed_l745_74531


namespace NUMINAMATH_CALUDE_avery_build_time_l745_74560

theorem avery_build_time (tom_time : ℝ) (joint_work_time : ℝ) (tom_remaining_time : ℝ)
  (h1 : tom_time = 2.5)
  (h2 : joint_work_time = 1)
  (h3 : tom_remaining_time = 2/3) :
  ∃ avery_time : ℝ, 
    (1 / avery_time + 1 / tom_time) * joint_work_time + 
    (1 / tom_time) * tom_remaining_time = 1 ∧ 
    avery_time = 3 := by
sorry

end NUMINAMATH_CALUDE_avery_build_time_l745_74560


namespace NUMINAMATH_CALUDE_train_speed_with_stops_l745_74541

/-- Proves that a train's average speed with stoppages is half of its speed without stoppages,
    given that it stops for half of each hour. -/
theorem train_speed_with_stops (D : ℝ) (h : D > 0) :
  let speed_without_stops : ℝ := 250
  let stop_ratio : ℝ := 0.5
  let time_without_stops : ℝ := D / speed_without_stops
  let time_with_stops : ℝ := time_without_stops / (1 - stop_ratio)
  let speed_with_stops : ℝ := D / time_with_stops
  speed_with_stops = speed_without_stops * (1 - stop_ratio) := by
sorry

end NUMINAMATH_CALUDE_train_speed_with_stops_l745_74541


namespace NUMINAMATH_CALUDE_inverse_proposition_l745_74557

theorem inverse_proposition :
  (∀ a : ℝ, a > 0 → a > 1) →
  (∀ a : ℝ, a > 1 → a > 0) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l745_74557


namespace NUMINAMATH_CALUDE_seven_thousand_six_hundred_scientific_notation_l745_74590

/-- Scientific notation representation -/
structure ScientificNotation where
  a : ℝ
  n : ℤ
  h1 : 1 ≤ |a|
  h2 : |a| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem seven_thousand_six_hundred_scientific_notation :
  toScientificNotation 7600 = ScientificNotation.mk 7.6 3 sorry sorry :=
sorry

end NUMINAMATH_CALUDE_seven_thousand_six_hundred_scientific_notation_l745_74590


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l745_74517

/-- Given a line with equation y + 3 = -2(x + 5), 
    the sum of its x-intercept and y-intercept is -39/2 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 3 = -2*(x + 5)) → 
  (∃ x_int y_int : ℝ, 
    (y_int + 3 = -2*(x_int + 5)) ∧ 
    (0 + 3 = -2*(x_int + 5)) ∧ 
    (y_int + 3 = -2*(0 + 5)) ∧ 
    (x_int + y_int = -39/2)) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l745_74517


namespace NUMINAMATH_CALUDE_value_of_y_l745_74586

theorem value_of_y (x y : ℤ) (h1 : x^2 + x + 4 = y - 4) (h2 : x = -7) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l745_74586


namespace NUMINAMATH_CALUDE_set_intersection_complement_l745_74508

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}

def M : Set ℕ := {1, 4}

def N : Set ℕ := {1, 3, 5}

theorem set_intersection_complement :
  N ∩ (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_set_intersection_complement_l745_74508


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l745_74510

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ |x| + |y| ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 4 ∧ |x₀| + |y₀| = M :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l745_74510


namespace NUMINAMATH_CALUDE_constant_expression_l745_74551

theorem constant_expression (x y k : ℝ) 
  (eq1 : x + 2*y = k + 2) 
  (eq2 : 2*x - 3*y = 3*k - 1) : 
  x + 9*y = 7 := by
sorry

end NUMINAMATH_CALUDE_constant_expression_l745_74551


namespace NUMINAMATH_CALUDE_no_solution_cubic_inequality_l745_74585

theorem no_solution_cubic_inequality :
  ¬∃ x : ℝ, x ≠ 2 ∧ (x^3 - 8) / (x - 2) < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_cubic_inequality_l745_74585


namespace NUMINAMATH_CALUDE_divisibility_theorem_l745_74522

theorem divisibility_theorem (a : ℤ) : 
  (2 ∣ a^2 - a) ∧ (3 ∣ a^3 - a) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l745_74522


namespace NUMINAMATH_CALUDE_total_cars_count_l745_74512

/-- Represents the number of cars counted by each person -/
structure CarCounts where
  jared : ℕ
  ann : ℕ
  alfred : ℕ
  bella : ℕ

/-- Calculates the total number of cars counted by all people -/
def total_count (counts : CarCounts) : ℕ :=
  counts.jared + counts.ann + counts.alfred + counts.bella

/-- Theorem stating the total count of cars after Alfred's recount -/
theorem total_cars_count (counts : CarCounts) :
  counts.jared = 300 ∧
  counts.ann = counts.jared + counts.jared * 15 / 100 ∧
  counts.alfred = counts.ann - 7 + (counts.ann - 7) * 12 / 100 ∧
  counts.bella = counts.jared + counts.jared * 20 / 100 ∧
  counts.bella = counts.alfred - counts.alfred * 10 / 100 →
  total_count counts = 1365 := by
  sorry

#eval total_count { jared := 300, ann := 345, alfred := 379, bella := 341 }

end NUMINAMATH_CALUDE_total_cars_count_l745_74512


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l745_74537

/-- Given two points A and B as endpoints of a diameter of a circle,
    this theorem proves the equation of the circle. -/
theorem circle_equation_from_diameter_endpoints 
  (A B : ℝ × ℝ) 
  (h_A : A = (1, 4)) 
  (h_B : B = (3, -2)) : 
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧ 
    r^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4 ∧
    ∀ (x y : ℝ), (x - C.1)^2 + (y - C.2)^2 = r^2 ↔ (x - 2)^2 + (y - 1)^2 = 10 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l745_74537


namespace NUMINAMATH_CALUDE_library_shelves_l745_74556

theorem library_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 14240) (h2 : books_per_shelf = 8) :
  total_books / books_per_shelf = 1780 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_l745_74556


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l745_74578

/-- Proves that the number of child tickets sold is 63 given the theater conditions --/
theorem theater_ticket_sales (total_seats : ℕ) (adult_price child_price : ℕ) (total_revenue : ℕ) 
  (h1 : total_seats = 80)
  (h2 : adult_price = 12)
  (h3 : child_price = 5)
  (h4 : total_revenue = 519) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_seats ∧
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    child_tickets = 63 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l745_74578


namespace NUMINAMATH_CALUDE_expression_simplification_l745_74593

theorem expression_simplification (b : ℝ) : 
  ((3 * b + 10 - 5 * b^2) / 5) = -b^2 + (3 * b / 5) + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l745_74593


namespace NUMINAMATH_CALUDE_custom_operation_equation_l745_74582

-- Define the custom operation *
def star (a b : ℤ) : ℤ := 2 * a + b

-- State the theorem
theorem custom_operation_equation :
  ∃ x : ℤ, star 3 (star 4 x) = -1 ∧ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equation_l745_74582


namespace NUMINAMATH_CALUDE_johns_sister_age_l745_74528

/-- Given the ages of John, his dad, and his sister, prove that John's sister is 37.5 years old -/
theorem johns_sister_age :
  ∀ (john dad sister : ℝ),
  dad = john + 15 →
  john + dad = 100 →
  sister = john - 5 →
  sister = 37.5 := by
sorry

end NUMINAMATH_CALUDE_johns_sister_age_l745_74528


namespace NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l745_74546

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def coinTossProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

/-- Theorem: The probability of getting exactly 3 heads when tossing a fair coin 8 times is 7/32 -/
theorem three_heads_in_eight_tosses :
  coinTossProbability 8 3 = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l745_74546


namespace NUMINAMATH_CALUDE_xyz_value_l745_74594

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 45)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 15)
  (h3 : x + y + z = 5) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l745_74594


namespace NUMINAMATH_CALUDE_curve_is_circle_l745_74520

theorem curve_is_circle (θ : Real) (r : Real → Real) :
  (∀ θ, r θ = 1 / (1 - Real.sin θ)) →
  ∃ (x y : Real → Real), ∀ θ,
    x θ ^ 2 + (y θ - 1) ^ 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_curve_is_circle_l745_74520


namespace NUMINAMATH_CALUDE_divisibility_condition_l745_74575

theorem divisibility_condition (a b : ℤ) (ha : a ≥ 3) (hb : b ≥ 3) :
  (a * b^2 + b + 7 ∣ a^2 * b + a + b) ↔ ∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l745_74575


namespace NUMINAMATH_CALUDE_coin_distribution_l745_74542

theorem coin_distribution (a b c d e : ℚ) : 
  a + b + c + d + e = 5 →  -- total is 5 coins
  ∃ (x y : ℚ), (a = x - 2*y ∧ b = x - y ∧ c = x ∧ d = x + y ∧ e = x + 2*y) →  -- arithmetic sequence
  a + b = c + d + e →  -- sum of first two equals sum of last three
  b = 4/3 :=  -- second person receives 4/3 coins
by sorry

end NUMINAMATH_CALUDE_coin_distribution_l745_74542


namespace NUMINAMATH_CALUDE_similar_transformation_l745_74515

structure Square where
  diagonal : ℝ

structure Transformation where
  area_after : ℝ
  is_similar : Bool

def original_square : Square := { diagonal := 2 }

def transformation : Transformation := { area_after := 4, is_similar := true }

theorem similar_transformation (s : Square) (t : Transformation) :
  s.diagonal = 2 ∧ t.area_after = 4 → t.is_similar = true := by
  sorry

end NUMINAMATH_CALUDE_similar_transformation_l745_74515


namespace NUMINAMATH_CALUDE_certain_number_problem_l745_74540

theorem certain_number_problem (x : ℝ) : 
  (0.90 * x = 0.50 * 1080) → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l745_74540


namespace NUMINAMATH_CALUDE_a_over_b_equals_half_l745_74569

theorem a_over_b_equals_half (a b : ℤ) (h : a + Real.sqrt b = Real.sqrt (15 + Real.sqrt 216)) : a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_a_over_b_equals_half_l745_74569


namespace NUMINAMATH_CALUDE_trout_percentage_is_sixty_percent_l745_74595

def total_fish : ℕ := 5
def trout_price : ℕ := 5
def bluegill_price : ℕ := 4
def sunday_earnings : ℕ := 23

theorem trout_percentage_is_sixty_percent :
  ∃ (trout blue_gill : ℕ),
    trout + blue_gill = total_fish ∧
    trout * trout_price + blue_gill * bluegill_price = sunday_earnings ∧
    (trout : ℚ) / (total_fish : ℚ) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_trout_percentage_is_sixty_percent_l745_74595


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l745_74588

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 3) :
  (1 / a + 1 / b) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l745_74588


namespace NUMINAMATH_CALUDE_m_range_l745_74504

theorem m_range (p q : Prop) (m : ℝ) 
  (hp : ∀ x : ℝ, 2*x - x^2 < m)
  (hq : m^2 - 2*m - 3 ≥ 0)
  (hnp : ¬(¬p))
  (hpq : ¬(p ∧ q)) :
  1 < m ∧ m < 3 := by
sorry

end NUMINAMATH_CALUDE_m_range_l745_74504


namespace NUMINAMATH_CALUDE_max_candies_for_class_l745_74570

def max_candies (num_students : ℕ) (mean_candies : ℕ) (min_candies : ℕ) : ℕ :=
  (num_students * mean_candies) - (min_candies * (num_students - 1))

theorem max_candies_for_class (num_students : ℕ) (mean_candies : ℕ) (min_candies : ℕ) 
  (h1 : num_students = 24)
  (h2 : mean_candies = 7)
  (h3 : min_candies = 3) :
  max_candies num_students mean_candies min_candies = 99 :=
by
  sorry

#eval max_candies 24 7 3

end NUMINAMATH_CALUDE_max_candies_for_class_l745_74570


namespace NUMINAMATH_CALUDE_rectangle_configuration_exists_l745_74576

/-- Represents a rectangle with vertical and horizontal sides -/
structure Rectangle where
  x : ℝ × ℝ  -- x-coordinates of left and right sides
  y : ℝ × ℝ  -- y-coordinates of bottom and top sides

/-- Checks if two rectangles meet (have at least one point in common) -/
def rectangles_meet (r1 r2 : Rectangle) : Prop :=
  (r1.x.1 ≤ r2.x.2 ∧ r2.x.1 ≤ r1.x.2) ∧ (r1.y.1 ≤ r2.y.2 ∧ r2.y.1 ≤ r1.y.2)

/-- Checks if two rectangles follow each other based on their indices -/
def rectangles_follow (i j n : ℕ) : Prop :=
  i % n = (j + 1) % n ∨ j % n = (i + 1) % n

/-- Represents a valid configuration of n rectangles -/
def valid_configuration (n : ℕ) (rectangles : Fin n → Rectangle) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    rectangles_meet (rectangles i) (rectangles j) ↔ ¬rectangles_follow i.val j.val n

/-- The main theorem stating that a valid configuration exists if and only if n ≤ 5 -/
theorem rectangle_configuration_exists (n : ℕ) (h : n ≥ 1) :
  (∃ rectangles : Fin n → Rectangle, valid_configuration n rectangles) ↔ n ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_rectangle_configuration_exists_l745_74576


namespace NUMINAMATH_CALUDE_expression_evaluation_l745_74591

/-- Proves that the given expression evaluates to -5 when x = -2 and y = -1 -/
theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = -1) :
  2 * (x + y) * (-x - y) - (2 * x + y) * (-2 * x + y) = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l745_74591


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l745_74511

theorem reciprocal_of_negative_two_thirds :
  ((-2 : ℚ) / 3)⁻¹ = -3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l745_74511


namespace NUMINAMATH_CALUDE_f_minimum_at_cos2x_neg_half_l745_74583

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * Real.cos (2 * x) + 16) - Real.sin x ^ 2

theorem f_minimum_at_cos2x_neg_half :
  ∀ x : ℝ, f x ≥ 0 ∧ (f x = 0 ↔ Real.cos (2 * x) = -1/2) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_at_cos2x_neg_half_l745_74583


namespace NUMINAMATH_CALUDE_inequality_proof_l745_74581

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a^4 / (4*a^4 + b^4 + c^4)) + (b^4 / (a^4 + 4*b^4 + c^4)) + (c^4 / (a^4 + b^4 + 4*c^4)) ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l745_74581


namespace NUMINAMATH_CALUDE_average_problem_l745_74568

theorem average_problem (x : ℝ) : (2 + 76 + x) / 3 = 5 → x = -63 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l745_74568


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l745_74584

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ+), x^2 + 2*y^2 = 2*x^3 - x := by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l745_74584


namespace NUMINAMATH_CALUDE_johns_allowance_l745_74505

theorem johns_allowance (allowance : ℚ) : 
  (allowance * (2/5) * (2/3) = 64/100) → allowance = 24/10 := by
  sorry

end NUMINAMATH_CALUDE_johns_allowance_l745_74505


namespace NUMINAMATH_CALUDE_expression_equals_one_l745_74538

theorem expression_equals_one :
  (120^2 - 9^2) / (90^2 - 18^2) * ((90-18)*(90+18)) / ((120-9)*(120+9)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l745_74538


namespace NUMINAMATH_CALUDE_extreme_points_condition_l745_74535

/-- The function f(x) = ln x + ax^2 - 2x has two distinct extreme points
    if and only if 0 < a < 1/2, where x > 0 -/
theorem extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧
    (∀ x : ℝ, x > 0 → (((1 : ℝ) / x + 2 * a * x - 2 = 0) ↔ (x = x₁ ∨ x = x₂))))
  ↔ (0 < a ∧ a < (1 : ℝ) / 2) :=
by sorry


end NUMINAMATH_CALUDE_extreme_points_condition_l745_74535


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l745_74516

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- Converts a number to its representation in a given base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- The length of a number's representation in a given base -/
def digitCount (n : ℕ) (base : ℕ) : ℕ :=
  sorry

theorem smallest_dual_palindrome :
  ∀ m : ℕ, m < 17 →
    ¬(isPalindrome m 2 ∧ digitCount m 2 = 5 ∧
      isPalindrome m 3 ∧ digitCount m 3 = 3) →
  isPalindrome 17 2 ∧ digitCount 17 2 = 5 ∧
  isPalindrome 17 3 ∧ digitCount 17 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l745_74516


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l745_74548

theorem consecutive_integers_cube_sum (a b c d : ℕ) : 
  (a + 1 = b) ∧ (b + 1 = c) ∧ (c + 1 = d) ∧ 
  (a^2 + b^2 + c^2 + d^2 = 9340) →
  (a^3 + b^3 + c^3 + d^3 = 457064) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l745_74548


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l745_74554

theorem complex_magnitude_equality (t : ℝ) (h1 : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 2 * Real.sqrt 17 → t = Real.sqrt 59 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l745_74554


namespace NUMINAMATH_CALUDE_rectangle_width_l745_74589

theorem rectangle_width (width : ℝ) (h1 : width > 0) : 
  (2 * width) * width = 50 → width = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l745_74589


namespace NUMINAMATH_CALUDE_pebble_collection_proof_l745_74514

def initial_pebbles : ℕ := 3
def collection_days : ℕ := 15
def first_day_collection : ℕ := 2
def daily_increase : ℕ := 1

def total_pebbles : ℕ := initial_pebbles + (collection_days * (2 * first_day_collection + (collection_days - 1) * daily_increase)) / 2

theorem pebble_collection_proof :
  total_pebbles = 138 := by
  sorry

end NUMINAMATH_CALUDE_pebble_collection_proof_l745_74514


namespace NUMINAMATH_CALUDE_smallest_n_with_odd_digits_l745_74529

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d, d ∈ (97 * n).digits 10 → d % 2 = 1

theorem smallest_n_with_odd_digits :
  ∀ n : ℕ, n > 1 →
    (all_digits_odd n → n ≥ 35) ∧
    (all_digits_odd 35) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_odd_digits_l745_74529


namespace NUMINAMATH_CALUDE_bucket_weight_l745_74524

/-- Given a bucket with unknown weight and unknown full water weight,
    if the total weight is p when it's three-quarters full and q when it's one-third full,
    then the total weight when it's completely full is (1/5)(8p - 3q). -/
theorem bucket_weight (p q : ℝ) : 
  (∃ (x y : ℝ), x + 3/4 * y = p ∧ x + 1/3 * y = q) → 
  (∃ (x y : ℝ), x + 3/4 * y = p ∧ x + 1/3 * y = q ∧ x + y = 1/5 * (8*p - 3*q)) :=
by sorry

end NUMINAMATH_CALUDE_bucket_weight_l745_74524


namespace NUMINAMATH_CALUDE_right_and_obtuse_angles_in_clerts_l745_74532

-- Define the number of clerts in a full Martian circle
def martian_full_circle : ℕ := 600

-- Define Earth angles in degrees
def earth_right_angle : ℕ := 90
def earth_obtuse_angle : ℕ := 135
def earth_full_circle : ℕ := 360

-- Define the conversion function from Earth degrees to Martian clerts
def earth_to_martian (earth_angle : ℕ) : ℕ :=
  (earth_angle * martian_full_circle) / earth_full_circle

-- Theorem statement
theorem right_and_obtuse_angles_in_clerts :
  earth_to_martian earth_right_angle = 150 ∧
  earth_to_martian earth_obtuse_angle = 225 := by
  sorry


end NUMINAMATH_CALUDE_right_and_obtuse_angles_in_clerts_l745_74532


namespace NUMINAMATH_CALUDE_distinct_x_intercepts_l745_74567

/-- The number of distinct real solutions to the equation (x-5)(x^2 - x - 6) = 0 -/
def num_solutions : ℕ := 3

/-- The equation representing the x-intercepts of the graph -/
def equation (x : ℝ) : ℝ := (x - 5) * (x^2 - x - 6)

theorem distinct_x_intercepts :
  ∃ (s : Finset ℝ), (∀ x ∈ s, equation x = 0) ∧ s.card = num_solutions :=
sorry

end NUMINAMATH_CALUDE_distinct_x_intercepts_l745_74567


namespace NUMINAMATH_CALUDE_tournament_theorem_l745_74599

/-- A tournament is a complete directed graph -/
structure Tournament (n : ℕ) where
  edges : Fin n → Fin n → Bool
  complete : ∀ i j, i ≠ j → edges i j ≠ edges j i
  no_self_edges : ∀ i, edges i i = false

/-- A set of edges in a tournament -/
def EdgeSet (n : ℕ) := Fin n → Fin n → Bool

/-- Reverse the orientation of edges in the given set -/
def reverseEdges (T : Tournament n) (S : EdgeSet n) : Tournament n where
  edges i j := if S i j then ¬(T.edges i j) else T.edges i j
  complete := sorry
  no_self_edges := sorry

/-- A graph contains a cycle -/
def hasCycle (T : Tournament n) : Prop := sorry

/-- A graph is acyclic -/
def isAcyclic (T : Tournament n) : Prop := ¬(hasCycle T)

/-- The number of edges in an edge set -/
def edgeCount (S : EdgeSet n) : ℕ := sorry

theorem tournament_theorem (n : ℕ) (h : n = 8) :
  (∃ T : Tournament n, ∀ S : EdgeSet n, edgeCount S ≤ 7 → hasCycle (reverseEdges T S)) ∧
  (∀ T : Tournament n, ∃ S : EdgeSet n, edgeCount S ≤ 8 ∧ isAcyclic (reverseEdges T S)) :=
sorry

end NUMINAMATH_CALUDE_tournament_theorem_l745_74599


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l745_74502

-- Define the sets A, B, C, and D
def A : Set ℝ := {x | x^2 + 3*x - 4 ≥ 0}
def B : Set ℝ := {x | (x-2)/x ≤ 0}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < 1+a}
def D (m : ℝ) : Set ℝ := {x | x^2 - (2*m+1/2)*x + m*(m+1/2) ≤ 0}

-- Part 1
theorem range_of_a :
  ∀ a : ℝ, (C a ⊆ (A ∩ B)) ↔ a ≥ 1/2 :=
sorry

-- Part 2
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ D m → x ∈ A ∩ B) ∧
           (∃ x : ℝ, x ∈ A ∩ B ∧ x ∉ D m) ↔
  1 ≤ m ∧ m ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l745_74502


namespace NUMINAMATH_CALUDE_hyperbola_equation_l745_74545

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0
  asymptote_eq : ∀ (x y : ℝ), x = 2 * y ∨ x = -2 * y
  point_on_curve : (4 : ℝ)^2 / a^2 - 1^2 / b^2 = 1

/-- The specific equation of the hyperbola -/
def specific_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 12 - y^2 / 3 = 1

/-- Theorem stating that the specific equation holds for the given hyperbola -/
theorem hyperbola_equation (h : Hyperbola) :
  ∀ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1 ↔ specific_equation h x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l745_74545


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l745_74564

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (7 * a) % 72 = 1 ∧ (13 * b) % 72 = 1 →
  (3 * a + 9 * b) % 72 = 18 := by
sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l745_74564


namespace NUMINAMATH_CALUDE_geometric_sequence_min_a3_l745_74539

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_min_a3 (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 - a 1 = 1 →
  (∀ b : ℕ → ℝ, is_geometric_sequence b → (∀ n : ℕ, b n > 0) → b 2 - b 1 = 1 → a 3 ≤ b 3) →
  ∀ n : ℕ, a n = 2^(n - 1) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_a3_l745_74539


namespace NUMINAMATH_CALUDE_crayons_given_away_l745_74571

theorem crayons_given_away (initial_crayons remaining_crayons : ℕ) 
  (h1 : initial_crayons = 106)
  (h2 : remaining_crayons = 52) : 
  initial_crayons - remaining_crayons = 54 := by
  sorry

end NUMINAMATH_CALUDE_crayons_given_away_l745_74571


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_mutually_exclusive_but_not_opposite_l745_74500

/-- Represents the contents of a pencil case -/
structure PencilCase where
  pencils : ℕ
  pens : ℕ

/-- Represents the outcome of selecting two items -/
inductive Selection
  | TwoPencils
  | OnePencilOnePen
  | TwoPens

/-- Defines the pencil case with 2 pencils and 2 pens -/
def case : PencilCase := ⟨2, 2⟩

/-- Predicate for exactly one pen being selected -/
def exactlyOnePen (s : Selection) : Prop :=
  s = Selection.OnePencilOnePen

/-- Predicate for exactly two pencils being selected -/
def exactlyTwoPencils (s : Selection) : Prop :=
  s = Selection.TwoPencils

/-- Theorem stating that "Exactly 1 pen" and "Exactly 2 pencils" are mutually exclusive -/
theorem mutually_exclusive :
  ∀ s : Selection, ¬(exactlyOnePen s ∧ exactlyTwoPencils s) :=
sorry

/-- Theorem stating that "Exactly 1 pen" and "Exactly 2 pencils" are not opposite events -/
theorem not_opposite :
  ∃ s : Selection, ¬(exactlyOnePen s ∨ exactlyTwoPencils s) :=
sorry

/-- Main theorem combining the above results -/
theorem mutually_exclusive_but_not_opposite :
  (∀ s : Selection, ¬(exactlyOnePen s ∧ exactlyTwoPencils s)) ∧
  (∃ s : Selection, ¬(exactlyOnePen s ∨ exactlyTwoPencils s)) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_mutually_exclusive_but_not_opposite_l745_74500


namespace NUMINAMATH_CALUDE_factorize_x4_plus_81_l745_74525

theorem factorize_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6*x + 9) * (x^2 - 6*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factorize_x4_plus_81_l745_74525


namespace NUMINAMATH_CALUDE_roots_of_polynomials_l745_74523

theorem roots_of_polynomials (r : ℝ) : 
  r^2 - 2*r - 1 = 0 → r^5 - 12*r^4 - 29*r - 12 = 0 := by
  sorry

#check roots_of_polynomials

end NUMINAMATH_CALUDE_roots_of_polynomials_l745_74523


namespace NUMINAMATH_CALUDE_matrix_value_example_l745_74509

def matrix_value (p q r s : ℤ) : ℤ := p * s - q * r

theorem matrix_value_example : matrix_value 4 5 2 3 = 2 := by sorry

end NUMINAMATH_CALUDE_matrix_value_example_l745_74509


namespace NUMINAMATH_CALUDE_min_value_expression_l745_74506

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) ≥ 3 ∧
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) = 3 ↔ y = x * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l745_74506


namespace NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l745_74592

/-- A function is even if f(-x) = f(x) for all x in its domain --/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The sum of two functions --/
def FunctionSum (f g : ℝ → ℝ) : ℝ → ℝ := fun x ↦ f x + g x

theorem even_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (FunctionSum f g)) ∧
  (∃ f g : ℝ → ℝ, IsEven (FunctionSum f g) ∧ (¬IsEven f ∨ ¬IsEven g)) := by
  sorry

#check even_sum_sufficient_not_necessary

end NUMINAMATH_CALUDE_even_sum_sufficient_not_necessary_l745_74592


namespace NUMINAMATH_CALUDE_complement_A_subset_B_l745_74565

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 1}

-- Define set B
def B : Set ℝ := {y | y ≥ 0}

-- Define the complement of A
def complementA : Set ℝ := {x | x ≥ 1}

theorem complement_A_subset_B : complementA ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_complement_A_subset_B_l745_74565


namespace NUMINAMATH_CALUDE_cube_edge_length_l745_74513

theorem cube_edge_length (volume : ℝ) (edge_length : ℝ) :
  volume = 2744 ∧ volume = edge_length ^ 3 → edge_length = 14 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l745_74513


namespace NUMINAMATH_CALUDE_gcd_pow_minus_one_l745_74553

theorem gcd_pow_minus_one (a n m : ℕ) (ha : a > 0) :
  Nat.gcd (a^n - 1) (a^m - 1) = a^(Nat.gcd n m) - 1 :=
by sorry

end NUMINAMATH_CALUDE_gcd_pow_minus_one_l745_74553


namespace NUMINAMATH_CALUDE_tomato_price_per_pound_l745_74561

/-- Calculates the price per pound of tomatoes in Scott's ratatouille recipe --/
theorem tomato_price_per_pound
  (eggplant_weight : ℝ) (eggplant_price : ℝ)
  (zucchini_weight : ℝ) (zucchini_price : ℝ)
  (tomato_weight : ℝ)
  (onion_weight : ℝ) (onion_price : ℝ)
  (basil_weight : ℝ) (basil_price : ℝ)
  (yield_quarts : ℝ) (price_per_quart : ℝ)
  (h1 : eggplant_weight = 5)
  (h2 : eggplant_price = 2)
  (h3 : zucchini_weight = 4)
  (h4 : zucchini_price = 2)
  (h5 : tomato_weight = 4)
  (h6 : onion_weight = 3)
  (h7 : onion_price = 1)
  (h8 : basil_weight = 1)
  (h9 : basil_price = 2.5 * 2)  -- $2.50 per half pound, so double for 1 pound
  (h10 : yield_quarts = 4)
  (h11 : price_per_quart = 10) :
  (yield_quarts * price_per_quart - 
   (eggplant_weight * eggplant_price + 
    zucchini_weight * zucchini_price + 
    onion_weight * onion_price + 
    basil_weight * basil_price)) / tomato_weight = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_tomato_price_per_pound_l745_74561


namespace NUMINAMATH_CALUDE_circle_origin_outside_l745_74566

theorem circle_origin_outside (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - x + y + m = 0 → (x^2 + y^2 > 0)) → 
  (0 < m ∧ m < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_circle_origin_outside_l745_74566


namespace NUMINAMATH_CALUDE_min_additional_packs_needed_l745_74596

/-- The number of sticker packs in each basket -/
def packsPerBasket : ℕ := 7

/-- The current number of sticker packs Matilda has -/
def currentPacks : ℕ := 40

/-- The minimum number of additional packs needed -/
def minAdditionalPacks : ℕ := 2

/-- Theorem stating the minimum number of additional packs needed -/
theorem min_additional_packs_needed : 
  ∃ (totalPacks : ℕ), 
    totalPacks = currentPacks + minAdditionalPacks ∧ 
    totalPacks % packsPerBasket = 0 ∧
    ∀ (k : ℕ), k < minAdditionalPacks → 
      (currentPacks + k) % packsPerBasket ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_min_additional_packs_needed_l745_74596


namespace NUMINAMATH_CALUDE_circle_triangle_area_relation_l745_74518

theorem circle_triangle_area_relation :
  ∀ (A B C : ℝ),
  (15 : ℝ)^2 + 20^2 = 25^2 →  -- Right triangle condition
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Areas are positive
  C ≥ A ∧ C ≥ B →  -- C is the largest area
  A + B + (1/2 * 15 * 20) = (π * 25^2) / 8 →  -- Area relation
  A + B + 150 = C :=
by sorry

end NUMINAMATH_CALUDE_circle_triangle_area_relation_l745_74518


namespace NUMINAMATH_CALUDE_exercise_books_quantity_l745_74547

/-- Given a ratio of items and the quantity of one item, calculate the quantity of another item in the ratio. -/
def calculate_quantity (ratio_a : ℕ) (ratio_b : ℕ) (quantity_a : ℕ) : ℕ :=
  (quantity_a * ratio_b) / ratio_a

/-- Prove that given 140 pencils and a ratio of 14 : 4 : 3 for pencils : pens : exercise books, 
    the number of exercise books is 30. -/
theorem exercise_books_quantity (pencils : ℕ) (ratio_pencils ratio_pens ratio_books : ℕ) 
    (h1 : pencils = 140)
    (h2 : ratio_pencils = 14)
    (h3 : ratio_pens = 4)
    (h4 : ratio_books = 3) :
  calculate_quantity ratio_pencils ratio_books pencils = 30 := by
  sorry

#eval calculate_quantity 14 3 140

end NUMINAMATH_CALUDE_exercise_books_quantity_l745_74547


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_intersection_contains_one_integer_l745_74558

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 3 > 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 1 ≤ 0 ∧ a > 0}

-- Theorem for part I
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x | 1 < x ∧ x ≤ 1 + Real.sqrt 2} := by sorry

-- Theorem for part II
theorem intersection_contains_one_integer (a : ℝ) :
  (∃! (n : ℤ), (n : ℝ) ∈ A ∩ B a) ↔ 3/4 ≤ a ∧ a < 4/3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_intersection_contains_one_integer_l745_74558


namespace NUMINAMATH_CALUDE_sum_reciprocal_pairs_gt_one_l745_74550

theorem sum_reciprocal_pairs_gt_one (a₁ a₂ a₃ : ℝ) (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) :
  let S := a₁ + a₂ + a₃
  (a₁^2 / (a₁ - 1) > S) ∧ (a₂^2 / (a₂ - 1) > S) ∧ (a₃^2 / (a₃ - 1) > S) →
  1 / (a₁ + a₂) + 1 / (a₂ + a₃) + 1 / (a₃ + a₁) > 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_pairs_gt_one_l745_74550


namespace NUMINAMATH_CALUDE_new_person_weight_is_102_l745_74563

/-- The weight of a new person joining a group, given the initial group size,
    average weight increase, and weight of the person being replaced. -/
def new_person_weight (initial_group_size : ℕ) (avg_weight_increase : ℝ) (replaced_person_weight : ℝ) : ℝ :=
  replaced_person_weight + initial_group_size * avg_weight_increase

/-- Theorem stating that the weight of the new person is 102 kg -/
theorem new_person_weight_is_102 :
  new_person_weight 6 4.5 75 = 102 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_102_l745_74563


namespace NUMINAMATH_CALUDE_rectangle_y_value_l745_74574

/-- A rectangle with vertices at (-3, y), (5, y), (-3, -2), and (5, -2) has an area of 96 square units. -/
def rectangle_area (y : ℝ) : Prop :=
  (5 - (-3)) * (y - (-2)) = 96

/-- The theorem states that if y is negative and satisfies the rectangle_area condition, then y = -14. -/
theorem rectangle_y_value (y : ℝ) (h1 : y < 0) (h2 : rectangle_area y) : y = -14 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l745_74574


namespace NUMINAMATH_CALUDE_caterpillar_eggs_hatched_l745_74549

theorem caterpillar_eggs_hatched (initial_caterpillars : ℕ) (caterpillars_left : ℕ) (final_caterpillars : ℕ) 
  (h1 : initial_caterpillars = 14)
  (h2 : caterpillars_left = 8)
  (h3 : final_caterpillars = 10) :
  initial_caterpillars + (caterpillars_left + final_caterpillars - initial_caterpillars) - caterpillars_left = final_caterpillars :=
by sorry

end NUMINAMATH_CALUDE_caterpillar_eggs_hatched_l745_74549


namespace NUMINAMATH_CALUDE_distance_ratio_on_rough_terrain_l745_74533

theorem distance_ratio_on_rough_terrain
  (total_distance : ℝ)
  (speed_ratio : ℝ → ℝ → Prop)
  (rough_terrain_speed : ℝ → ℝ)
  (rough_terrain_length : ℝ)
  (meeting_point : ℝ)
  (h1 : speed_ratio 2 3)
  (h2 : ∀ x, rough_terrain_speed x = x / 2)
  (h3 : rough_terrain_length = 2 / 3 * total_distance)
  (h4 : meeting_point = total_distance / 2) :
  ∃ (d1 d2 : ℝ), d1 + d2 = rough_terrain_length ∧ d1 / d2 = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_distance_ratio_on_rough_terrain_l745_74533


namespace NUMINAMATH_CALUDE_alice_number_problem_l745_74572

theorem alice_number_problem (x : ℝ) : ((x + 3) * 3 - 5) / 3 = 10 → x = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_alice_number_problem_l745_74572


namespace NUMINAMATH_CALUDE_ellipse_equation_l745_74562

/-- The equation of an ellipse with specific properties -/
theorem ellipse_equation (a b c : ℝ) (h1 : a + b = 10) (h2 : 2 * c = 4 * Real.sqrt 5) 
  (h3 : a^2 = c^2 + b^2) (h4 : a > b) (h5 : b > 0) :
  ∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / a^2 + (y t)^2 / b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l745_74562


namespace NUMINAMATH_CALUDE_smallest_number_in_sequence_l745_74552

theorem smallest_number_in_sequence (x : ℝ) : 
  let second := 4 * x
  let third := 2 * x
  (x + second + third) / 3 = 77 →
  x = 33 := by sorry

end NUMINAMATH_CALUDE_smallest_number_in_sequence_l745_74552


namespace NUMINAMATH_CALUDE_smallest_subtrahend_for_multiple_of_five_l745_74501

theorem smallest_subtrahend_for_multiple_of_five :
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → m < n → ¬(∃ k : ℤ, 425 - m = 5 * k)) ∧ (∃ k : ℤ, 425 - n = 5 * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_subtrahend_for_multiple_of_five_l745_74501


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l745_74559

theorem exponential_equation_solution :
  ∃ x : ℝ, (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x * (9 : ℝ)^x = (81 : ℝ)^6 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l745_74559


namespace NUMINAMATH_CALUDE_min_distance_to_locus_l745_74543

open Complex

theorem min_distance_to_locus (z : ℂ) :
  (abs (z - 1) = abs (z + 2*I)) →
  ∃ min_val : ℝ, (min_val = (9 * Real.sqrt 5) / 10) ∧
  (∀ w : ℂ, abs (z - 1) = abs (z + 2*I) → abs (w - 1 - I) ≥ min_val) ∧
  (∃ z₀ : ℂ, abs (z₀ - 1) = abs (z₀ + 2*I) ∧ abs (z₀ - 1 - I) = min_val) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_locus_l745_74543


namespace NUMINAMATH_CALUDE_expression_evaluation_l745_74579

theorem expression_evaluation (b : ℝ) :
  let x : ℝ := b + 9
  2 * x - b + 5 = b + 23 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l745_74579


namespace NUMINAMATH_CALUDE_fib_50_div_5_l745_74503

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The 50th Fibonacci number is divisible by 5 -/
theorem fib_50_div_5 : 5 ∣ fib 50 := by sorry

end NUMINAMATH_CALUDE_fib_50_div_5_l745_74503


namespace NUMINAMATH_CALUDE_boys_in_class_l745_74530

/-- Given a class with a 4:3 ratio of girls to boys and 49 total students,
    prove that the number of boys is 21. -/
theorem boys_in_class (girls boys : ℕ) : 
  4 * boys = 3 * girls →  -- ratio of girls to boys is 4:3
  girls + boys = 49 →     -- total number of students is 49
  boys = 21 :=            -- prove that the number of boys is 21
by sorry

end NUMINAMATH_CALUDE_boys_in_class_l745_74530


namespace NUMINAMATH_CALUDE_inequality_ordering_l745_74597

theorem inequality_ordering (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_ordering_l745_74597


namespace NUMINAMATH_CALUDE_hyperbolic_matrix_det_is_one_cosh_sq_sub_sinh_sq_l745_74534

open Matrix Real

/-- The determinant of a specific 3x3 matrix involving hyperbolic functions is 1 -/
theorem hyperbolic_matrix_det_is_one (α β : ℝ) : 
  det !![cosh α * cosh β, cosh α * sinh β, -sinh α;
         -sinh β, cosh β, 0;
         sinh α * cosh β, sinh α * sinh β, cosh α] = 1 := by
  sorry

/-- The fundamental hyperbolic identity -/
theorem cosh_sq_sub_sinh_sq (x : ℝ) : cosh x * cosh x - sinh x * sinh x = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolic_matrix_det_is_one_cosh_sq_sub_sinh_sq_l745_74534


namespace NUMINAMATH_CALUDE_system_solutions_correct_l745_74507

theorem system_solutions_correct :
  -- System 1
  (∃ x y : ℝ, 3 * x + 2 * y = 10 ∧ x / 2 - (y + 1) / 3 = 1 ∧ x = 3 ∧ y = 1/2) ∧
  -- System 2
  (∃ x y : ℝ, 4 * x - 5 * y = 3 ∧ (x - 2 * y) / 0.4 = 0.6 ∧ x = 1.6 ∧ y = 0.68) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_correct_l745_74507


namespace NUMINAMATH_CALUDE_existence_of_prime_and_power_l745_74555

/-- The distance from a real number to its nearest integer -/
noncomputable def dist_to_nearest_int (x : ℝ) : ℝ :=
  |x - round x|

/-- The statement of the theorem -/
theorem existence_of_prime_and_power (a b : ℕ+) :
  ∃ (p : ℕ) (k : ℕ), Prime p ∧ p % 2 = 1 ∧
    dist_to_nearest_int (a / p^k : ℝ) +
    dist_to_nearest_int (b / p^k : ℝ) +
    dist_to_nearest_int ((a + b) / p^k : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_and_power_l745_74555


namespace NUMINAMATH_CALUDE_total_balls_count_l745_74573

/-- The number of balls owned by Jungkook -/
def jungkook_balls : ℕ := 3

/-- The number of balls owned by Yoongi -/
def yoongi_balls : ℕ := 2

/-- The total number of balls owned by Jungkook and Yoongi -/
def total_balls : ℕ := jungkook_balls + yoongi_balls

theorem total_balls_count : total_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_count_l745_74573


namespace NUMINAMATH_CALUDE_circle_radius_l745_74519

/-- Given a circle with area M cm² and circumference N cm,
    where M/N = 15 and the area is 60π cm²,
    prove that the radius of the circle is 2√15 cm. -/
theorem circle_radius (M N : ℝ) (h1 : M / N = 15) (h2 : M = 60 * Real.pi) :
  ∃ (r : ℝ), r = 2 * Real.sqrt 15 ∧ M = Real.pi * r^2 ∧ N = 2 * Real.pi * r :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l745_74519


namespace NUMINAMATH_CALUDE_coin_stack_arrangements_l745_74577

/-- The number of distinguishable arrangements of coins -/
def coin_arrangements (gold : Nat) (silver : Nat) : Nat :=
  Nat.choose (gold + silver) gold * (gold + silver + 1)

/-- Theorem stating the number of distinguishable arrangements for the given problem -/
theorem coin_stack_arrangements :
  coin_arrangements 5 3 = 504 := by
  sorry

end NUMINAMATH_CALUDE_coin_stack_arrangements_l745_74577


namespace NUMINAMATH_CALUDE_table_sum_difference_l745_74521

/-- Represents a cell in the N × N table -/
structure Cell (N : ℕ) where
  row : Fin N
  col : Fin N

/-- The rule for placing numbers in the table -/
def placeNumber (N : ℕ) (n : Fin (N^2)) : Cell N → Prop :=
  sorry

/-- The sum of numbers in a given column -/
def columnSum (N : ℕ) (col : Fin N) : ℕ :=
  sorry

/-- The sum of numbers in a given row -/
def rowSum (N : ℕ) (row : Fin N) : ℕ :=
  sorry

/-- The column containing N² -/
def lastColumn (N : ℕ) : Fin N :=
  sorry

/-- The row containing 1 -/
def firstRow (N : ℕ) : Fin N :=
  sorry

theorem table_sum_difference (N : ℕ) :
  columnSum N (lastColumn N) - rowSum N (firstRow N) = N^2 - N :=
sorry

end NUMINAMATH_CALUDE_table_sum_difference_l745_74521


namespace NUMINAMATH_CALUDE_inequality_range_l745_74526

-- Define the inequality function
def f (x a : ℝ) : Prop := x^2 + a*x > 4*x + a - 3

-- State the theorem
theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x, f x a ↔ x < -1 ∨ x > 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l745_74526


namespace NUMINAMATH_CALUDE_complex_modulus_product_l745_74580

theorem complex_modulus_product : Complex.abs ((10 - 7*I) * (9 + 11*I)) = Real.sqrt 30098 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l745_74580


namespace NUMINAMATH_CALUDE_expression_simplification_l745_74544

theorem expression_simplification (x y : ℚ) (hx : x = 1/8) (hy : y = -4) :
  ((x * y - 2) * (x * y + 2) - 2 * x^2 * y^2 + 4) / (-x * y) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l745_74544


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l745_74536

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (4, x) (-4, 4) → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l745_74536


namespace NUMINAMATH_CALUDE_rivet_distribution_l745_74587

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a point with integer coordinates -/
structure Point where
  x : ℕ
  y : ℕ

/-- Checks if a point is inside a rectangle -/
def Point.insideRectangle (p : Point) (r : Rectangle) : Prop :=
  p.x < r.width ∧ p.y < r.height

/-- Checks if a point is on the grid lines of a rectangle divided into unit squares -/
def Point.onGridLines (p : Point) : Prop :=
  p.x = 0 ∨ p.y = 0

/-- Theorem: In a 9x11 rectangle divided into unit squares, 
    with 200 points inside and not on grid lines, 
    there exists at least one unit square with 3 or more points -/
theorem rivet_distribution (points : List Point) : 
  points.length = 200 → 
  (∀ p ∈ points, p.insideRectangle ⟨9, 11⟩ ∧ ¬p.onGridLines) →
  ∃ (x y : ℕ), x < 9 ∧ y < 11 ∧ 
    (points.filter (λ p => p.x ≥ x ∧ p.x < x + 1 ∧ p.y ≥ y ∧ p.y < y + 1)).length ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_rivet_distribution_l745_74587


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l745_74598

theorem algebraic_expression_value : 
  let x : ℝ := Real.sqrt 3 + 2
  (x^2 - 4*x + 3) = 2 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l745_74598
