import Mathlib

namespace NUMINAMATH_CALUDE_triangle_problem_l3469_346936

/-- Given a triangle ABC with circumradius 1 and the relation between sides, prove the value of a and the area when b = 1. -/
theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Conditions
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / (2 * Real.sin A) = 1 ∧  -- circumradius = 1
  b = a * Real.cos C - (Real.sqrt 3 / 6) * a * c →
  -- Conclusions
  a = Real.sqrt 3 ∧
  (b = 1 → Real.sqrt 3 / 4 = 1/2 * b * c * Real.sin A) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3469_346936


namespace NUMINAMATH_CALUDE_fraction_sum_bounds_l3469_346913

theorem fraction_sum_bounds (a b c d : ℕ+) 
  (sum_num : a + c = 1000)
  (sum_denom : b + d = 1000) :
  (999 : ℚ) / 969 + 1 / 31 ≤ (a : ℚ) / b + (c : ℚ) / d ∧ 
  (a : ℚ) / b + (c : ℚ) / d ≤ 999 + 1 / 999 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_bounds_l3469_346913


namespace NUMINAMATH_CALUDE_pencil_cost_theorem_l3469_346991

/-- Calculates the average cost per pencil in cents, rounded to the nearest cent -/
def averageCostPerPencil (pencilCount : ℕ) (pencilCost : ℚ) (shippingCost : ℚ) (discount : ℚ) : ℕ :=
  let totalCost := pencilCost + shippingCost - discount
  let totalCostInCents := (totalCost * 100).floor
  ((totalCostInCents + pencilCount / 2) / pencilCount).toNat

theorem pencil_cost_theorem :
  let pencilCount : ℕ := 150
  let pencilCost : ℚ := 15.5
  let shippingCost : ℚ := 5.75
  let discount : ℚ := 1

  averageCostPerPencil pencilCount pencilCost shippingCost discount = 14 := by
    sorry

#eval averageCostPerPencil 150 15.5 5.75 1

end NUMINAMATH_CALUDE_pencil_cost_theorem_l3469_346991


namespace NUMINAMATH_CALUDE_circle_central_symmetry_l3469_346927

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- Central symmetry for a figure in 2D plane --/
def CentralSymmetry (F : Set (ℝ × ℝ)) :=
  ∃ c : ℝ × ℝ, ∀ p : ℝ × ℝ, p ∈ F → (2 * c.1 - p.1, 2 * c.2 - p.2) ∈ F

/-- The set of points in a circle --/
def CirclePoints (c : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2}

/-- Theorem: A circle has central symmetry --/
theorem circle_central_symmetry (c : Circle) : CentralSymmetry (CirclePoints c) := by
  sorry


end NUMINAMATH_CALUDE_circle_central_symmetry_l3469_346927


namespace NUMINAMATH_CALUDE_sin_squared_50_over_1_plus_sin_10_l3469_346917

theorem sin_squared_50_over_1_plus_sin_10 :
  (Real.sin (50 * π / 180))^2 / (1 + Real.sin (10 * π / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_50_over_1_plus_sin_10_l3469_346917


namespace NUMINAMATH_CALUDE_max_of_three_l3469_346990

theorem max_of_three (a b c : ℝ) :
  let x := max a b
  ∀ m : ℝ, (m = max a (max b c) ↔ (m = x ∨ (c > x ∧ m = c))) :=
by sorry

end NUMINAMATH_CALUDE_max_of_three_l3469_346990


namespace NUMINAMATH_CALUDE_lemon_square_price_is_correct_l3469_346971

/-- Represents the price of a lemon square -/
def lemon_square_price : ℝ := 2

/-- The number of brownies sold -/
def brownies_sold : ℕ := 4

/-- The price of each brownie -/
def brownie_price : ℝ := 3

/-- The number of lemon squares sold -/
def lemon_squares_sold : ℕ := 5

/-- The number of cookies to be sold -/
def cookies_to_sell : ℕ := 7

/-- The price of each cookie -/
def cookie_price : ℝ := 4

/-- The total revenue goal -/
def total_revenue_goal : ℝ := 50

theorem lemon_square_price_is_correct :
  (brownies_sold : ℝ) * brownie_price +
  (lemon_squares_sold : ℝ) * lemon_square_price +
  (cookies_to_sell : ℝ) * cookie_price =
  total_revenue_goal :=
by sorry

end NUMINAMATH_CALUDE_lemon_square_price_is_correct_l3469_346971


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l3469_346996

theorem concentric_circles_ratio (r R k : ℝ) (hr : r > 0) (hR : R > r) (hk : k > 0) :
  (π * R^2 - π * r^2) = k * (π * r^2) → R / r = Real.sqrt (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l3469_346996


namespace NUMINAMATH_CALUDE_larger_integer_value_l3469_346910

theorem larger_integer_value (a b : ℕ+) (h1 : (a : ℚ) / (b : ℚ) = 7 / 3) (h2 : (a : ℕ) * b = 189) : a = 21 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l3469_346910


namespace NUMINAMATH_CALUDE_wedding_cost_theorem_l3469_346980

/-- Calculates the total cost of a wedding given the venue cost, cost per guest, 
    John's desired number of guests, and the percentage increase desired by John's wife. -/
def wedding_cost (venue_cost : ℕ) (cost_per_guest : ℕ) (john_guests : ℕ) (wife_increase_percent : ℕ) : ℕ :=
  let total_guests := john_guests + john_guests * wife_increase_percent / 100
  venue_cost + cost_per_guest * total_guests

/-- Proves that the total cost of the wedding is $50,000 given the specified conditions. -/
theorem wedding_cost_theorem : 
  wedding_cost 10000 500 50 60 = 50000 := by
  sorry

#eval wedding_cost 10000 500 50 60

end NUMINAMATH_CALUDE_wedding_cost_theorem_l3469_346980


namespace NUMINAMATH_CALUDE_arithmetic_mistakes_calculation_difference_l3469_346901

theorem arithmetic_mistakes (x : ℤ) : 
  ((-1 - 8) * 2 - x = -24) → (x = 6) :=
by sorry

theorem calculation_difference : 
  ((-1 - 8) + 2 - 5) - ((-1 - 8) * 2 - 5) = 11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mistakes_calculation_difference_l3469_346901


namespace NUMINAMATH_CALUDE_max_value_sqrt_expression_l3469_346984

theorem max_value_sqrt_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 16) :
  Real.sqrt (x + 64) + Real.sqrt (16 - x) + 2 * Real.sqrt x ≤ 4 * Real.sqrt 5 + 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_expression_l3469_346984


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l3469_346934

def m : ℕ := 2023^2 + 2^2023

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ) : 
  m = 2023^2 + 2^2023 → (m^2 + 2^m) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_two_to_m_l3469_346934


namespace NUMINAMATH_CALUDE_factorization_of_8a_squared_minus_2_l3469_346978

theorem factorization_of_8a_squared_minus_2 (a : ℝ) : 8 * a^2 - 2 = 2 * (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_8a_squared_minus_2_l3469_346978


namespace NUMINAMATH_CALUDE_cans_display_rows_l3469_346973

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 2 * n + 1

/-- The total number of cans in a display with n rows -/
def total_cans (n : ℕ) : ℕ := n * (n + 2)

/-- The number of rows in the display -/
def num_rows : ℕ := 12

theorem cans_display_rows :
  (cans_in_row 1 = 3) ∧
  (∀ n : ℕ, n > 0 → cans_in_row (n + 1) = cans_in_row n + 2) ∧
  (total_cans num_rows = 169) ∧
  (∀ m : ℕ, m ≠ num_rows → total_cans m ≠ 169) :=
by sorry

end NUMINAMATH_CALUDE_cans_display_rows_l3469_346973


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3469_346988

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 250 →
  percentage = 75 →
  final = initial * (1 + percentage / 100) →
  final = 437.5 := by
sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3469_346988


namespace NUMINAMATH_CALUDE_friends_assignment_count_l3469_346969

theorem friends_assignment_count : 
  (∀ n : ℕ, n > 0 → n ^ 8 = (n * n ^ 7)) →
  4 ^ 8 = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friends_assignment_count_l3469_346969


namespace NUMINAMATH_CALUDE_first_year_more_rabbits_l3469_346930

def squirrels (k : ℕ) : ℕ := 2020 * 2^k - 2019

def rabbits (k : ℕ) : ℕ := (4^k + 2) / 3

def more_rabbits_than_squirrels (k : ℕ) : Prop :=
  rabbits k > squirrels k

theorem first_year_more_rabbits : 
  (∀ n < 13, ¬(more_rabbits_than_squirrels n)) ∧ 
  more_rabbits_than_squirrels 13 := by
  sorry

#check first_year_more_rabbits

end NUMINAMATH_CALUDE_first_year_more_rabbits_l3469_346930


namespace NUMINAMATH_CALUDE_factorization_equality_l3469_346995

theorem factorization_equality (x y : ℝ) : 
  (x - y)^2 - (3*x^2 - 3*x*y + y^2) = x*(y - 2*x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3469_346995


namespace NUMINAMATH_CALUDE_sin_pi_minus_alpha_l3469_346987

theorem sin_pi_minus_alpha (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.sin α) : 
  Real.sin (π - α) = 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_minus_alpha_l3469_346987


namespace NUMINAMATH_CALUDE_trig_identity_l3469_346976

theorem trig_identity : (1 / (2 * Real.sin (10 * π / 180))) - 2 * Real.sin (70 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3469_346976


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3469_346983

theorem inequality_system_solution_set (x : ℝ) :
  (x + 1 > 0 ∧ x - 3 > 0) ↔ x > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3469_346983


namespace NUMINAMATH_CALUDE_total_subjects_theorem_l3469_346912

/-- The number of subjects taken by Monica, Marius, and Millie -/
def total_subjects (monica : ℕ) (marius_extra : ℕ) (millie_extra : ℕ) : ℕ :=
  monica + (monica + marius_extra) + (monica + marius_extra + millie_extra)

/-- Theorem stating the total number of subjects taken by the three students -/
theorem total_subjects_theorem :
  total_subjects 10 4 3 = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_subjects_theorem_l3469_346912


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3469_346929

theorem contrapositive_equivalence (p q : Prop) :
  (¬p → q) → (¬q → p) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3469_346929


namespace NUMINAMATH_CALUDE_inequality_preservation_l3469_346953

theorem inequality_preservation (x y z : ℝ) (k : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x * y * z = 1) (h5 : 1/x + 1/y + 1/z ≥ x + y + z) :
  1/(x^k) + 1/(y^k) + 1/(z^k) ≥ x^k + y^k + z^k := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3469_346953


namespace NUMINAMATH_CALUDE_equal_areas_imply_all_equal_l3469_346957

-- Define a square
structure Square where
  side : ℝ
  area : ℝ
  area_eq : area = side * side

-- Define the four parts of the square
structure SquareParts where
  square : Square
  part1 : ℝ
  part2 : ℝ
  part3 : ℝ
  part4 : ℝ
  sum_eq_area : part1 + part2 + part3 + part4 = square.area

-- Define the perpendicular lines
structure PerpendicularLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ
  perpendicular : ∀ x y, line1 x * line2 y = -1

-- Theorem statement
theorem equal_areas_imply_all_equal (sq : Square) (parts : SquareParts) (lines : PerpendicularLines)
  (h1 : parts.square = sq)
  (h2 : parts.part1 = parts.part2)
  (h3 : parts.part2 = parts.part3)
  (h4 : ∃ x y, x ∈ Set.Icc 0 sq.side ∧ y ∈ Set.Icc 0 sq.side ∧ 
       lines.line1 x = lines.line2 y) :
  parts.part1 = parts.part2 ∧ parts.part2 = parts.part3 ∧ parts.part3 = parts.part4 :=
by sorry

end NUMINAMATH_CALUDE_equal_areas_imply_all_equal_l3469_346957


namespace NUMINAMATH_CALUDE_gcd_diff_is_square_l3469_346960

theorem gcd_diff_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_diff_is_square_l3469_346960


namespace NUMINAMATH_CALUDE_journey_time_proof_l3469_346948

/-- The total distance of the journey in miles -/
def total_distance : ℝ := 120

/-- The speed of the car in miles per hour -/
def car_speed : ℝ := 30

/-- The walking speed in miles per hour -/
def walking_speed : ℝ := 5

/-- The distance Tom and Harry initially travel by car -/
def initial_car_distance : ℝ := 40

/-- Theorem stating that under the given conditions, the total journey time is 52/3 hours -/
theorem journey_time_proof :
  ∃ (T d : ℝ),
    -- Tom and Harry's initial car journey
    car_speed * (4/3) = initial_car_distance ∧
    -- Harry's walk back
    walking_speed * (T - 4/3) = d ∧
    -- Dick's walk
    walking_speed * (T - 4/3) = total_distance - d ∧
    -- Tom's return journey
    car_speed * T = 2 * initial_car_distance + d ∧
    -- Total journey time
    T = 52/3 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_proof_l3469_346948


namespace NUMINAMATH_CALUDE_fruit_cost_percentage_increase_l3469_346951

theorem fruit_cost_percentage_increase (max_cost min_cost : ℝ) 
  (h_max : max_cost = 45)
  (h_min : min_cost = 30) :
  (max_cost - min_cost) / min_cost * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_fruit_cost_percentage_increase_l3469_346951


namespace NUMINAMATH_CALUDE_probability_geometry_second_draw_l3469_346977

/-- Represents the set of questions in the problem -/
structure QuestionSet where
  total : ℕ
  algebra : ℕ
  geometry : ℕ
  algebra_first_draw : Prop

/-- The probability of selecting a geometry question on the second draw,
    given an algebra question was selected on the first draw -/
def conditional_probability (qs : QuestionSet) : ℚ :=
  qs.geometry / (qs.total - 1)

/-- The main theorem to prove -/
theorem probability_geometry_second_draw 
  (qs : QuestionSet) 
  (h1 : qs.total = 5) 
  (h2 : qs.algebra = 3) 
  (h3 : qs.geometry = 2) 
  (h4 : qs.algebra_first_draw) : 
  conditional_probability qs = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_geometry_second_draw_l3469_346977


namespace NUMINAMATH_CALUDE_harkamal_grapes_purchase_l3469_346993

/-- The amount of grapes purchased by Harkamal -/
def grapes_kg : ℝ := 8

/-- The cost of grapes per kg -/
def grapes_cost_per_kg : ℝ := 70

/-- The cost of mangoes per kg -/
def mangoes_cost_per_kg : ℝ := 60

/-- The amount of mangoes purchased by Harkamal -/
def mangoes_kg : ℝ := 9

/-- The total amount paid by Harkamal -/
def total_paid : ℝ := 1100

theorem harkamal_grapes_purchase :
  grapes_kg * grapes_cost_per_kg + mangoes_kg * mangoes_cost_per_kg = total_paid :=
by sorry

end NUMINAMATH_CALUDE_harkamal_grapes_purchase_l3469_346993


namespace NUMINAMATH_CALUDE_peters_class_size_l3469_346963

theorem peters_class_size :
  ∀ (hands_without_peter : ℕ) (hands_per_student : ℕ),
    hands_without_peter = 20 →
    hands_per_student = 2 →
    hands_without_peter / hands_per_student + 1 = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_peters_class_size_l3469_346963


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l3469_346900

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistanceTraveled (initialHeight : ℝ) (reboundRatio : ℝ) (bounces : ℕ) : ℝ :=
  let descentDistances := List.range (bounces + 1) |>.map (fun i => initialHeight * reboundRatio ^ i)
  let ascentDistances := descentDistances.tail
  (descentDistances.sum + ascentDistances.sum)

/-- The theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  totalDistanceTraveled 200 (1/3) 4 = 397 + 2/9 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l3469_346900


namespace NUMINAMATH_CALUDE_range_of_negative_values_l3469_346970

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is increasing on (-∞, 0) if f(x) ≤ f(y) for all x < y < 0 -/
def IncreasingOnNegative (f : ℝ → ℝ) : Prop := ∀ x y, x < y → y < 0 → f x ≤ f y

theorem range_of_negative_values (f : ℝ → ℝ) 
  (h_even : IsEven f) 
  (h_inc_neg : IncreasingOnNegative f) 
  (h_f2 : f 2 = 0) :
  ∀ x, f x < 0 ↔ x < -2 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l3469_346970


namespace NUMINAMATH_CALUDE_energetic_cycling_hours_l3469_346964

theorem energetic_cycling_hours 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (energetic_speed : ℝ) 
  (fatigued_speed : ℝ) 
  (h1 : total_distance = 150) 
  (h2 : total_time = 12) 
  (h3 : energetic_speed = 15) 
  (h4 : fatigued_speed = 10) : 
  ∃ (energetic_hours : ℝ), 
    energetic_hours * energetic_speed + (total_time - energetic_hours) * fatigued_speed = total_distance ∧ 
    energetic_hours = 6 := by
  sorry

end NUMINAMATH_CALUDE_energetic_cycling_hours_l3469_346964


namespace NUMINAMATH_CALUDE_valid_p_values_l3469_346919

def is_valid_p (p : ℤ) : Prop :=
  ∃ (k : ℤ), k > 0 ∧ (4 * p + 20) = k * (3 * p - 6)

theorem valid_p_values :
  {p : ℤ | is_valid_p p} = {3, 4, 15, 28} :=
by sorry

end NUMINAMATH_CALUDE_valid_p_values_l3469_346919


namespace NUMINAMATH_CALUDE_operations_result_l3469_346981

-- Define operation S
def S (a b : ℤ) : ℤ := 4*a + 6*b

-- Define operation T
def T (a b : ℤ) : ℤ := 5*a + 3*b

-- Theorem to prove
theorem operations_result : (S 6 3 = 42) ∧ (T 6 3 = 39) := by
  sorry

end NUMINAMATH_CALUDE_operations_result_l3469_346981


namespace NUMINAMATH_CALUDE_max_xy_value_l3469_346905

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) : x * y ≤ 28 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l3469_346905


namespace NUMINAMATH_CALUDE_cos_double_angle_proof_l3469_346923

theorem cos_double_angle_proof (α : ℝ) (a : ℝ × ℝ) : 
  a = (Real.cos α, Real.sqrt 2 / 2) → 
  Real.sqrt ((a.1)^2 + (a.2)^2) = Real.sqrt 3 / 2 → 
  Real.cos (2 * α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_proof_l3469_346923


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3469_346918

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 6 = 38 →
  a 4 + a 5 = 55 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3469_346918


namespace NUMINAMATH_CALUDE_data_median_and_variance_l3469_346972

def data : List ℝ := [5, 9, 8, 8, 10]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_median_and_variance :
  median data = 8 ∧ variance data = 2.8 := by sorry

end NUMINAMATH_CALUDE_data_median_and_variance_l3469_346972


namespace NUMINAMATH_CALUDE_parking_cost_theorem_l3469_346992

/-- The number of hours for the initial parking cost. -/
def h : ℝ := 2

/-- The initial parking cost in dollars. -/
def initial_cost : ℝ := 10

/-- The additional cost per hour after the initial period in dollars. -/
def additional_cost_per_hour : ℝ := 1.75

/-- The total number of hours parked. -/
def total_hours : ℝ := 9

/-- The average cost per hour for the total parking time. -/
def average_cost_per_hour : ℝ := 2.4722222222222223

theorem parking_cost_theorem :
  h = 2 ∧
  initial_cost = 10 ∧
  additional_cost_per_hour = 1.75 ∧
  total_hours = 9 ∧
  average_cost_per_hour = 2.4722222222222223 →
  initial_cost + additional_cost_per_hour * (total_hours - h) = average_cost_per_hour * total_hours :=
by sorry

end NUMINAMATH_CALUDE_parking_cost_theorem_l3469_346992


namespace NUMINAMATH_CALUDE_larger_number_proof_l3469_346965

theorem larger_number_proof (x y : ℤ) : 
  y = 2 * x + 3 → 
  x + y = 27 → 
  max x y = 19 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3469_346965


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3469_346999

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.a = Real.sqrt 2)
  (h2 : t.b = Real.sqrt 3)
  (h3 : t.B = 60 * π / 180) :
  t.A = 45 * π / 180 ∧ 
  t.C = 75 * π / 180 ∧ 
  t.c = (Real.sqrt 2 + Real.sqrt 6) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l3469_346999


namespace NUMINAMATH_CALUDE_sin_plus_cos_value_l3469_346974

theorem sin_plus_cos_value (A : Real) (h : Real.sin (2 * A) = 2/3) :
  Real.sin A + Real.cos A = Real.sqrt (5/3) := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_value_l3469_346974


namespace NUMINAMATH_CALUDE_rectangle_folding_l3469_346924

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the folding properties of the rectangle -/
structure FoldedRectangle extends Rectangle where
  pointE : Point
  pointF : Point
  coincideOnDiagonal : Bool

/-- The main theorem statement -/
theorem rectangle_folding (rect : FoldedRectangle) (k m : ℕ) :
  rect.width = 2 ∧ 
  rect.height = 1 ∧
  rect.pointE.x = rect.width - rect.pointF.x ∧
  rect.coincideOnDiagonal = true ∧
  Real.sqrt k - m = rect.pointE.x
  → k + m = 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_folding_l3469_346924


namespace NUMINAMATH_CALUDE_rowing_current_rate_l3469_346922

/-- Proves that the current rate is 1.1 km/hr given the conditions of the rowing problem -/
theorem rowing_current_rate (man_speed : ℝ) (upstream_time_ratio : ℝ) :
  man_speed = 3.3 →
  upstream_time_ratio = 2 →
  ∃ (current_rate : ℝ),
    current_rate = 1.1 ∧
    (man_speed + current_rate) * upstream_time_ratio = man_speed - current_rate :=
by sorry

end NUMINAMATH_CALUDE_rowing_current_rate_l3469_346922


namespace NUMINAMATH_CALUDE_literacy_test_probabilities_l3469_346935

/-- Scientific literacy test model -/
structure LiteracyTest where
  /-- Probability of answering a question correctly -/
  p_correct : ℝ
  /-- Number of questions in the test -/
  total_questions : ℕ
  /-- Number of correct answers in a row needed for A rating -/
  a_threshold : ℕ
  /-- Number of incorrect answers in a row needed for C rating -/
  c_threshold : ℕ

/-- Probabilities of different outcomes in the literacy test -/
def test_probabilities (test : LiteracyTest) :
  (ℝ × ℝ × ℝ × ℝ) :=
  sorry

/-- The main theorem about the scientific literacy test -/
theorem literacy_test_probabilities :
  let test := LiteracyTest.mk (2/3) 5 4 3
  let (p_a, p_b, p_four, p_five) := test_probabilities test
  p_a = 64/243 ∧ p_b = 158/243 ∧ p_four = 2/9 ∧ p_five = 20/27 :=
sorry

end NUMINAMATH_CALUDE_literacy_test_probabilities_l3469_346935


namespace NUMINAMATH_CALUDE_subtracted_amount_l3469_346944

theorem subtracted_amount (chosen_number : ℕ) (subtracted_amount : ℕ) : 
  chosen_number = 208 → 
  (chosen_number / 2 : ℚ) - subtracted_amount = 4 → 
  subtracted_amount = 100 := by
sorry

end NUMINAMATH_CALUDE_subtracted_amount_l3469_346944


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l3469_346956

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l3469_346956


namespace NUMINAMATH_CALUDE_fraction_simplification_l3469_346904

theorem fraction_simplification :
  let x := 5 / (1 + (32 * (Real.cos (15 * π / 180))^4 - 10 - 8 * Real.sqrt 3)^(1/3))
  x = 1 - 4^(1/3) + 16^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3469_346904


namespace NUMINAMATH_CALUDE_sqrt_200_equals_10_l3469_346940

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_equals_10_l3469_346940


namespace NUMINAMATH_CALUDE_person_a_age_l3469_346937

/-- The ages of two people, A and B, satisfy certain conditions. -/
structure AgeProblem where
  /-- Age of Person A this year -/
  a : ℕ
  /-- Age of Person B this year -/
  b : ℕ
  /-- The sum of their ages this year is 43 -/
  sum_constraint : a + b = 43
  /-- In 4 years, A will be 3 years older than B -/
  future_constraint : a + 4 = (b + 4) + 3

/-- Given the age constraints, Person A's age this year is 23 -/
theorem person_a_age (p : AgeProblem) : p.a = 23 := by
  sorry

end NUMINAMATH_CALUDE_person_a_age_l3469_346937


namespace NUMINAMATH_CALUDE_algebraic_identities_l3469_346954

theorem algebraic_identities (x : ℝ) (h : x + 1/x = 8) :
  (x^2 + 1/x^2 = 62) ∧ (x^3 + 1/x^3 = 488) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identities_l3469_346954


namespace NUMINAMATH_CALUDE_square_of_x_minus_three_l3469_346932

theorem square_of_x_minus_three (x : ℝ) (h : x = -3) : (x - 3)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_of_x_minus_three_l3469_346932


namespace NUMINAMATH_CALUDE_product_of_1010_2_and_102_3_l3469_346997

/-- Converts a binary number represented as a list of digits to its decimal value -/
def binary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 2^i) 0

/-- Converts a ternary number represented as a list of digits to its decimal value -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

theorem product_of_1010_2_and_102_3 : 
  let binary_num := [0, 1, 0, 1]  -- 1010 in binary, least significant bit first
  let ternary_num := [2, 0, 1]    -- 102 in ternary, least significant digit first
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 110 := by
  sorry

end NUMINAMATH_CALUDE_product_of_1010_2_and_102_3_l3469_346997


namespace NUMINAMATH_CALUDE_iphone_price_decrease_l3469_346986

theorem iphone_price_decrease (initial_price : ℝ) (second_month_decrease : ℝ) (final_price : ℝ) :
  initial_price = 1000 →
  second_month_decrease = 20 →
  final_price = 720 →
  ∃ (first_month_decrease : ℝ),
    first_month_decrease = 10 ∧
    final_price = initial_price * (1 - first_month_decrease / 100) * (1 - second_month_decrease / 100) :=
by sorry


end NUMINAMATH_CALUDE_iphone_price_decrease_l3469_346986


namespace NUMINAMATH_CALUDE_eight_six_four_combinations_l3469_346908

/-- The number of unique outfit combinations given the number of shirts, ties, and belts. -/
def outfitCombinations (shirts : ℕ) (ties : ℕ) (belts : ℕ) : ℕ :=
  shirts * ties * belts

/-- Theorem stating that 8 shirts, 6 ties, and 4 belts result in 192 unique combinations. -/
theorem eight_six_four_combinations :
  outfitCombinations 8 6 4 = 192 := by
  sorry

end NUMINAMATH_CALUDE_eight_six_four_combinations_l3469_346908


namespace NUMINAMATH_CALUDE_matthew_initial_cakes_l3469_346982

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 29

/-- The number of friends Matthew gave crackers and cakes to -/
def num_friends : ℕ := 2

/-- The number of cakes each person ate -/
def cakes_eaten_per_person : ℕ := 15

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := initial_crackers + num_friends * cakes_eaten_per_person

theorem matthew_initial_cakes : initial_cakes = 59 := by sorry

end NUMINAMATH_CALUDE_matthew_initial_cakes_l3469_346982


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l3469_346914

/-- Checks if three numbers can form a triangle --/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_theorem :
  (¬ canFormTriangle 2 5 7) ∧
  (¬ canFormTriangle 9 3 5) ∧
  (canFormTriangle 4 5 6) ∧
  (¬ canFormTriangle 4 5 10) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l3469_346914


namespace NUMINAMATH_CALUDE_tetrahedron_sum_l3469_346941

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  -- We don't need to define any fields, as we're only interested in its properties

/-- The number of edges in a regular tetrahedron -/
def num_edges (t : RegularTetrahedron) : ℕ := 6

/-- The number of corners in a regular tetrahedron -/
def num_corners (t : RegularTetrahedron) : ℕ := 4

/-- The number of faces in a regular tetrahedron -/
def num_faces (t : RegularTetrahedron) : ℕ := 4

/-- Theorem: The sum of edges, corners, and faces of a regular tetrahedron is 14 -/
theorem tetrahedron_sum (t : RegularTetrahedron) : 
  num_edges t + num_corners t + num_faces t = 14 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sum_l3469_346941


namespace NUMINAMATH_CALUDE_carter_cake_difference_l3469_346947

def regular_cheesecakes : ℕ := 6
def regular_muffins : ℕ := 5
def regular_red_velvet : ℕ := 8

def regular_total : ℕ := regular_cheesecakes + regular_muffins + regular_red_velvet

def triple_total : ℕ := 3 * regular_total

theorem carter_cake_difference : triple_total - regular_total = 38 := by
  sorry

end NUMINAMATH_CALUDE_carter_cake_difference_l3469_346947


namespace NUMINAMATH_CALUDE_divide_by_recurring_decimal_l3469_346909

/-- The recurring decimal 0.363636... represented as a rational number -/
def recurring_decimal : ℚ := 4 / 11

/-- The result of dividing 12 by the recurring decimal 0.363636... -/
theorem divide_by_recurring_decimal : 12 / recurring_decimal = 33 := by
  sorry

end NUMINAMATH_CALUDE_divide_by_recurring_decimal_l3469_346909


namespace NUMINAMATH_CALUDE_factorization_equality_l3469_346902

theorem factorization_equality (x : ℝ) : 5*x*(x-2) + 9*(x-2) = (x-2)*(5*x+9) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3469_346902


namespace NUMINAMATH_CALUDE_triangle_property_l3469_346915

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles of the triangle
  (a b c : ℝ)  -- Sides of the triangle opposite to angles A, B, C respectively

-- Define the property that makes a triangle right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2

-- State the theorem
theorem triangle_property (t : Triangle) 
  (h : Real.sin t.C = Real.sin t.A * Real.cos t.B) : 
  isRightTriangle t :=
sorry

end NUMINAMATH_CALUDE_triangle_property_l3469_346915


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_powers_l3469_346931

theorem quadratic_roots_sum_powers (t q : ℝ) (a₁ a₂ : ℝ) : 
  (∀ x : ℝ, x^2 - t*x + q = 0 ↔ x = a₁ ∨ x = a₂) →
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1003 → a₁^n + a₂^n = a₁ + a₂) →
  a₁^1004 + a₂^1004 = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_powers_l3469_346931


namespace NUMINAMATH_CALUDE_correct_calculation_l3469_346906

theorem correct_calculation (a b : ℝ) : 4 * a^2 * b - 3 * b * a^2 = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3469_346906


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l3469_346926

/-- The number of ways five people can sit in a row of six chairs -/
def seating_arrangements : ℕ :=
  let total_chairs : ℕ := 6
  let total_people : ℕ := 5
  let odd_numbered_chairs : ℕ := 3  -- chairs 1, 3, and 5
  odd_numbered_chairs * (total_chairs - 1) * (total_chairs - 2) * (total_chairs - 3) * (total_chairs - 4)

/-- Theorem stating that the number of seating arrangements is 360 -/
theorem seating_arrangements_count : seating_arrangements = 360 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l3469_346926


namespace NUMINAMATH_CALUDE_books_bought_at_yard_sale_l3469_346979

-- Define the initial number of books
def initial_books : ℕ := 35

-- Define the final number of books
def final_books : ℕ := 56

-- Theorem: The number of books bought at the yard sale is 21
theorem books_bought_at_yard_sale :
  final_books - initial_books = 21 :=
by sorry

end NUMINAMATH_CALUDE_books_bought_at_yard_sale_l3469_346979


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3469_346939

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a₂ = 4 and a₆ = 6, prove that a₁₀ = 9 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_a2 : a 2 = 4) 
    (h_a6 : a 6 = 6) : 
  a 10 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3469_346939


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3469_346907

theorem fraction_equation_solution : 
  {x : ℝ | (1 / (x^2 + 17*x + 20) + 1 / (x^2 + 12*x + 20) + 1 / (x^2 - 15*x + 20) = 0) ∧ 
           x ≠ -20 ∧ x ≠ -5 ∧ x ≠ -4 ∧ x ≠ -1} = 
  {x : ℝ | x = -20 ∨ x = -5 ∨ x = -4 ∨ x = -1} := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3469_346907


namespace NUMINAMATH_CALUDE_power_product_simplification_l3469_346952

theorem power_product_simplification : (-0.25)^11 * (-4)^12 = -4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l3469_346952


namespace NUMINAMATH_CALUDE_workshop_ratio_l3469_346959

theorem workshop_ratio (total : ℕ) (novelists : ℕ) (poets : ℕ) : 
  total = 24 → novelists = 15 → poets = total - novelists → 
  ∃ (a b : ℕ), a = 3 ∧ b = 5 ∧ poets * b = novelists * a :=
sorry

end NUMINAMATH_CALUDE_workshop_ratio_l3469_346959


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l3469_346945

/-- 
Proves that the number of years until a man's age is twice his son's age is 2,
given the initial conditions of their ages.
-/
theorem mans_age_twice_sons (man_age son_age : ℕ) (h1 : man_age = son_age + 26) (h2 : son_age = 24) :
  ∃ y : ℕ, y = 2 ∧ man_age + y = 2 * (son_age + y) :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l3469_346945


namespace NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l3469_346933

theorem complex_exp_13pi_div_2 : Complex.exp (13 * π * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_13pi_div_2_l3469_346933


namespace NUMINAMATH_CALUDE_don_bottles_from_shop_C_l3469_346942

/-- The number of bottles Don buys from Shop A -/
def bottles_from_A : ℕ := 150

/-- The number of bottles Don buys from Shop B -/
def bottles_from_B : ℕ := 180

/-- The total number of bottles Don is capable of buying -/
def total_bottles : ℕ := 550

/-- The number of bottles Don buys from Shop C -/
def bottles_from_C : ℕ := total_bottles - (bottles_from_A + bottles_from_B)

theorem don_bottles_from_shop_C :
  bottles_from_C = 220 :=
by sorry

end NUMINAMATH_CALUDE_don_bottles_from_shop_C_l3469_346942


namespace NUMINAMATH_CALUDE_min_burgers_recovery_l3469_346968

/-- The minimum whole number of burgers Sarah must sell to recover her initial investment -/
def min_burgers : ℕ := 637

/-- Sarah's initial investment in dollars -/
def initial_investment : ℕ := 7000

/-- Sarah's earnings per burger in dollars -/
def earnings_per_burger : ℕ := 15

/-- Sarah's ingredient cost per burger in dollars -/
def ingredient_cost_per_burger : ℕ := 4

/-- Theorem stating that min_burgers is the minimum whole number of burgers
    Sarah must sell to recover her initial investment -/
theorem min_burgers_recovery :
  (min_burgers * (earnings_per_burger - ingredient_cost_per_burger) ≥ initial_investment) ∧
  ∀ n : ℕ, n < min_burgers → n * (earnings_per_burger - ingredient_cost_per_burger) < initial_investment :=
by sorry

end NUMINAMATH_CALUDE_min_burgers_recovery_l3469_346968


namespace NUMINAMATH_CALUDE_solve_equation_l3469_346985

theorem solve_equation : ∃ x : ℝ, 25 - (4 + 3) = 5 + x ∧ x = 13 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l3469_346985


namespace NUMINAMATH_CALUDE_new_average_production_l3469_346943

theorem new_average_production (n : ℕ) (past_avg : ℝ) (today_prod : ℝ) :
  n = 12 ∧ past_avg = 50 ∧ today_prod = 115 →
  (n * past_avg + today_prod) / (n + 1) = 55 := by
  sorry

end NUMINAMATH_CALUDE_new_average_production_l3469_346943


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l3469_346958

theorem complex_magnitude_proof (z : ℂ) (h : z = 1 + Complex.I) : 
  Complex.abs (z^2 - 2*z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l3469_346958


namespace NUMINAMATH_CALUDE_train_passing_time_l3469_346994

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 280 →
  train_speed_kmh = 72 →
  passing_time = 14 →
  passing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l3469_346994


namespace NUMINAMATH_CALUDE_cheryl_strawberries_l3469_346946

theorem cheryl_strawberries (total : ℕ) (buckets : ℕ) (left_in_each : ℕ) 
  (h1 : total = 300)
  (h2 : buckets = 5)
  (h3 : left_in_each = 40) :
  total / buckets - left_in_each = 20 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_strawberries_l3469_346946


namespace NUMINAMATH_CALUDE_cinema_ticket_cost_l3469_346950

/-- Given Samuel and Kevin's cinema outing expenses, prove their combined ticket cost --/
theorem cinema_ticket_cost (total_budget : ℕ) 
  (samuel_food_drink : ℕ) (kevin_drink : ℕ) (kevin_food : ℕ) 
  (h1 : total_budget = 20)
  (h2 : samuel_food_drink = 6)
  (h3 : kevin_drink = 2)
  (h4 : kevin_food = 4) :
  ∃ (samuel_ticket kevin_ticket : ℕ),
    samuel_ticket + kevin_ticket = total_budget - (samuel_food_drink + kevin_drink + kevin_food) :=
by sorry

end NUMINAMATH_CALUDE_cinema_ticket_cost_l3469_346950


namespace NUMINAMATH_CALUDE_room_length_perimeter_ratio_l3469_346921

/-- Given a rectangular room with length 19 feet and width 11 feet,
    prove that the ratio of its length to its perimeter is 19:60. -/
theorem room_length_perimeter_ratio :
  let length : ℕ := 19
  let width : ℕ := 11
  let perimeter : ℕ := 2 * (length + width)
  (length : ℚ) / perimeter = 19 / 60 := by sorry

end NUMINAMATH_CALUDE_room_length_perimeter_ratio_l3469_346921


namespace NUMINAMATH_CALUDE_inequality_solution_l3469_346966

theorem inequality_solution (x : ℝ) : 
  (9 * x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5)) < 2 ↔ 
  (x > -5 ∧ x < -20/3) ∨ (x > 2/3 ∧ x < 4/3) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3469_346966


namespace NUMINAMATH_CALUDE_space_station_arrangements_count_l3469_346998

/-- The number of ways to distribute n distinguishable objects into k distinguishable boxes,
    with each box containing at least min and at most max objects. -/
def distribute (n k min max : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable objects into 3 distinguishable boxes,
    with each box containing at least 1 and at most 3 objects. -/
def space_station_arrangements : ℕ := distribute 6 3 1 3

theorem space_station_arrangements_count : space_station_arrangements = 450 := by sorry

end NUMINAMATH_CALUDE_space_station_arrangements_count_l3469_346998


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_six_l3469_346962

/-- A function with the property f(1-x) = f(3+x) for all x -/
def symmetric_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 - x) = f (3 + x)

/-- The set of zeros of a function -/
def zeros (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = 0}

/-- Theorem: If f is a symmetric function with exactly three distinct zeros,
    then the sum of these zeros is 6 -/
theorem sum_of_zeros_is_six (f : ℝ → ℝ) 
    (h_sym : symmetric_function f) 
    (h_zeros : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ zeros f = {a, b, c}) :
  ∃ a b c : ℝ, zeros f = {a, b, c} ∧ a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_six_l3469_346962


namespace NUMINAMATH_CALUDE_average_of_xyz_l3469_346903

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) :
  (x + y + z) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_of_xyz_l3469_346903


namespace NUMINAMATH_CALUDE_total_carrots_grown_l3469_346961

/-- The total number of carrots grown by Joan, Jessica, and Michael is 77. -/
theorem total_carrots_grown (joan_carrots : ℕ) (jessica_carrots : ℕ) (michael_carrots : ℕ)
  (h1 : joan_carrots = 29)
  (h2 : jessica_carrots = 11)
  (h3 : michael_carrots = 37) :
  joan_carrots + jessica_carrots + michael_carrots = 77 :=
by sorry

end NUMINAMATH_CALUDE_total_carrots_grown_l3469_346961


namespace NUMINAMATH_CALUDE_initial_weight_calculation_calvins_initial_weight_l3469_346967

/-- Calculates the initial weight of a person given their weight loss rate and final weight --/
theorem initial_weight_calculation 
  (weight_loss_per_month : ℕ) 
  (months : ℕ) 
  (final_weight : ℕ) : ℕ :=
  let total_weight_loss := weight_loss_per_month * months
  final_weight + total_weight_loss

/-- Proves that given the conditions, the initial weight was 250 pounds --/
theorem calvins_initial_weight :
  let weight_loss_per_month : ℕ := 8
  let months : ℕ := 12
  let final_weight : ℕ := 154
  initial_weight_calculation weight_loss_per_month months final_weight = 250 := by
  sorry

end NUMINAMATH_CALUDE_initial_weight_calculation_calvins_initial_weight_l3469_346967


namespace NUMINAMATH_CALUDE_binomial_distribution_params_l3469_346925

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectation (bd : BinomialDistribution) : ℝ := bd.n * bd.p

/-- The variance of a binomial distribution -/
def variance (bd : BinomialDistribution) : ℝ := bd.n * bd.p * (1 - bd.p)

/-- Theorem: For a binomial distribution with expectation 8 and variance 1.6,
    the parameters are n = 10 and p = 0.8 -/
theorem binomial_distribution_params :
  ∀ (bd : BinomialDistribution),
    expectation bd = 8 →
    variance bd = 1.6 →
    bd.n = 10 ∧ bd.p = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_params_l3469_346925


namespace NUMINAMATH_CALUDE_michael_passes_donovan_in_four_laps_l3469_346955

/-- Represents the race conditions and calculates the number of laps for Michael to pass Donovan -/
def raceLaps (trackLength : ℕ) (donovanNormalTime : ℕ) (michaelNormalTime : ℕ) 
              (obstacles : ℕ) (donovanObstacleTime : ℕ) (michaelObstacleTime : ℕ) : ℕ :=
  let donovanLapTime := donovanNormalTime + obstacles * donovanObstacleTime
  let michaelLapTime := michaelNormalTime + obstacles * michaelObstacleTime
  let timeDiffPerLap := donovanLapTime - michaelLapTime
  let lapsToPass := (donovanLapTime + timeDiffPerLap - 1) / timeDiffPerLap
  lapsToPass

/-- Theorem stating that Michael needs 4 laps to pass Donovan under the given conditions -/
theorem michael_passes_donovan_in_four_laps :
  raceLaps 300 45 40 3 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_michael_passes_donovan_in_four_laps_l3469_346955


namespace NUMINAMATH_CALUDE_magnitude_of_z_l3469_346938

theorem magnitude_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l3469_346938


namespace NUMINAMATH_CALUDE_final_time_sum_is_82_l3469_346920

def initial_hour : Nat := 3
def initial_minute : Nat := 0
def initial_second : Nat := 0
def hours_elapsed : Nat := 314
def minutes_elapsed : Nat := 21
def seconds_elapsed : Nat := 56

def final_time (ih im is he me se : Nat) : Nat × Nat × Nat :=
  let total_seconds := (ih * 3600 + im * 60 + is + he * 3600 + me * 60 + se) % 86400
  let h := (total_seconds / 3600) % 12
  let m := (total_seconds % 3600) / 60
  let s := total_seconds % 60
  (h, m, s)

theorem final_time_sum_is_82 :
  let (h, m, s) := final_time initial_hour initial_minute initial_second hours_elapsed minutes_elapsed seconds_elapsed
  h + m + s = 82 := by sorry

end NUMINAMATH_CALUDE_final_time_sum_is_82_l3469_346920


namespace NUMINAMATH_CALUDE_stephanies_remaining_payment_l3469_346916

/-- Calculates the remaining amount to pay for Stephanie's bills -/
def remaining_payment (electricity_bill gas_bill water_bill internet_bill : ℚ)
  (gas_paid_fraction : ℚ) (gas_additional_payment : ℚ)
  (water_paid_fraction : ℚ) (internet_payments : ℕ) (internet_payment_amount : ℚ) : ℚ :=
  let gas_remaining := gas_bill - (gas_paid_fraction * gas_bill + gas_additional_payment)
  let water_remaining := water_bill - (water_paid_fraction * water_bill)
  let internet_remaining := internet_bill - (internet_payments : ℚ) * internet_payment_amount
  gas_remaining + water_remaining + internet_remaining

/-- Stephanie's remaining bill payment is $30 -/
theorem stephanies_remaining_payment :
  remaining_payment 60 40 40 25 (3/4) 5 (1/2) 4 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_stephanies_remaining_payment_l3469_346916


namespace NUMINAMATH_CALUDE_disprove_statement_l3469_346911

theorem disprove_statement : ∃ (a b c : ℤ), c < b ∧ b < a ∧ a * c < 0 ∧ a * b ≥ a * c := by
  sorry

end NUMINAMATH_CALUDE_disprove_statement_l3469_346911


namespace NUMINAMATH_CALUDE_sand_price_per_ton_l3469_346989

theorem sand_price_per_ton 
  (total_cost : ℕ) 
  (cement_bags : ℕ) 
  (cement_price_per_bag : ℕ) 
  (sand_lorries : ℕ) 
  (sand_tons_per_lorry : ℕ) 
  (h1 : total_cost = 13000)
  (h2 : cement_bags = 500)
  (h3 : cement_price_per_bag = 10)
  (h4 : sand_lorries = 20)
  (h5 : sand_tons_per_lorry = 10) : 
  (total_cost - cement_bags * cement_price_per_bag) / (sand_lorries * sand_tons_per_lorry) = 40 := by
sorry

end NUMINAMATH_CALUDE_sand_price_per_ton_l3469_346989


namespace NUMINAMATH_CALUDE_spinsters_cats_ratio_l3469_346975

theorem spinsters_cats_ratio : 
  ∀ (spinsters cats : ℕ),
    spinsters = 12 →
    cats = spinsters + 42 →
    (spinsters : ℚ) / cats = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_spinsters_cats_ratio_l3469_346975


namespace NUMINAMATH_CALUDE_num_bedrooms_is_three_l3469_346949

/-- The number of bedrooms in the house -/
def num_bedrooms : ℕ := 3

/-- Time to renovate one bedroom (in hours) -/
def bedroom_time : ℕ := 4

/-- Time to renovate the kitchen (in hours) -/
def kitchen_time : ℕ := 6

/-- Total renovation time (in hours) -/
def total_time : ℕ := 54

/-- Theorem: The number of bedrooms is 3 given the renovation times -/
theorem num_bedrooms_is_three :
  num_bedrooms = 3 ∧
  bedroom_time = 4 ∧
  kitchen_time = 6 ∧
  total_time = 54 ∧
  total_time = num_bedrooms * bedroom_time + kitchen_time + 2 * (num_bedrooms * bedroom_time + kitchen_time) :=
by sorry

end NUMINAMATH_CALUDE_num_bedrooms_is_three_l3469_346949


namespace NUMINAMATH_CALUDE_doctors_lawyers_ratio_l3469_346928

theorem doctors_lawyers_ratio (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (40 * (m + n) = 35 * m + 50 * n) → (m : ℚ) / n = 2 := by
  sorry

end NUMINAMATH_CALUDE_doctors_lawyers_ratio_l3469_346928
