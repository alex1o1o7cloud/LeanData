import Mathlib

namespace NUMINAMATH_CALUDE_factorization_valid_l1842_184282

theorem factorization_valid (x : ℝ) : 10 * x^2 - 5 * x = 5 * x * (2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_valid_l1842_184282


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l1842_184208

theorem line_slope_intercept_product (m b : ℝ) : 
  m = 3/4 → b = -2 → m * b < -3/2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l1842_184208


namespace NUMINAMATH_CALUDE_gildas_marbles_theorem_l1842_184223

/-- The percentage of marbles Gilda has left after giving away to her friends and family -/
def gildas_remaining_marbles : ℝ :=
  let after_pedro := 1 - 0.30
  let after_ebony := after_pedro * (1 - 0.20)
  let after_jimmy := after_ebony * (1 - 0.15)
  let after_clara := after_jimmy * (1 - 0.10)
  after_clara * 100

/-- Theorem stating that Gilda has 42.84% of her original marbles left -/
theorem gildas_marbles_theorem : 
  ∃ ε > 0, |gildas_remaining_marbles - 42.84| < ε :=
sorry

end NUMINAMATH_CALUDE_gildas_marbles_theorem_l1842_184223


namespace NUMINAMATH_CALUDE_hoopit_toes_count_l1842_184273

/-- Represents the number of toes a Hoopit has on each hand -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of hands a Hoopit has -/
def hoopit_hands : ℕ := 4

/-- Represents the number of toes a Neglart has on each hand -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands a Neglart has -/
def neglart_hands : ℕ := 5

/-- Represents the number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Represents the number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

theorem hoopit_toes_count : 
  hoopit_toes_per_hand * hoopit_hands * hoopit_students + 
  neglart_toes_per_hand * neglart_hands * neglart_students = total_toes :=
by sorry

end NUMINAMATH_CALUDE_hoopit_toes_count_l1842_184273


namespace NUMINAMATH_CALUDE_copy_pages_theorem_l1842_184230

/-- Given a cost per page in cents and a budget in dollars, 
    calculate the number of pages that can be copied. -/
def pages_copied (cost_per_page : ℕ) (budget : ℕ) : ℕ :=
  (budget * 100) / cost_per_page

/-- Theorem: With a cost of 3 cents per page and a budget of $15, 
    500 pages can be copied. -/
theorem copy_pages_theorem : pages_copied 3 15 = 500 := by
  sorry

end NUMINAMATH_CALUDE_copy_pages_theorem_l1842_184230


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1842_184217

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l1842_184217


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l1842_184205

/-- Product of digits of a two-digit number -/
def P (n : ℕ) : ℕ := (n / 10) * (n % 10)

/-- Sum of digits of a two-digit number -/
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

/-- A two-digit number is between 10 and 99, inclusive -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_two_digit_number :
  ∃! M : ℕ, is_two_digit M ∧ M = P M + S M + 1 :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l1842_184205


namespace NUMINAMATH_CALUDE_river_speed_proof_l1842_184207

-- Define the problem parameters
def distance : ℝ := 200
def timeInterval : ℝ := 4
def speedA : ℝ := 36
def speedB : ℝ := 64

-- Define the river current speed as a variable
def riverSpeed : ℝ := 14

-- Theorem statement
theorem river_speed_proof :
  -- First meeting time
  let firstMeetTime : ℝ := distance / (speedA + speedB)
  -- Total time
  let totalTime : ℝ := firstMeetTime + timeInterval
  -- Total distance covered
  let totalDistance : ℝ := 3 * distance
  -- Equation for boat A's journey
  totalDistance = (speedA + riverSpeed + speedA - riverSpeed) * totalTime →
  -- Conclusion
  riverSpeed = 14 := by
  sorry

end NUMINAMATH_CALUDE_river_speed_proof_l1842_184207


namespace NUMINAMATH_CALUDE_fraction_1800_1809_l1842_184296

/-- The number of states that joined the union during 1800-1809. -/
def states_1800_1809 : ℕ := 4

/-- The total number of states in Walter's collection. -/
def total_states : ℕ := 30

/-- The fraction of states that joined during 1800-1809 out of the first 30 states. -/
theorem fraction_1800_1809 : (states_1800_1809 : ℚ) / total_states = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_1800_1809_l1842_184296


namespace NUMINAMATH_CALUDE_polynomial_division_l1842_184274

theorem polynomial_division (a b : ℝ) (h : b ≠ 2 * a) :
  (4 * a^2 - b^2) / (b - 2 * a) = -2 * a - b := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l1842_184274


namespace NUMINAMATH_CALUDE_sphere_volume_fraction_l1842_184289

theorem sphere_volume_fraction (R : ℝ) (h : R > 0) :
  let sphereVolume := (4 / 3) * Real.pi * R^3
  let capVolume := Real.pi * R^3 * ((2 / 3) - (5 * Real.sqrt 2) / 12)
  capVolume / sphereVolume = (8 - 5 * Real.sqrt 2) / 16 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_fraction_l1842_184289


namespace NUMINAMATH_CALUDE_ferry_speed_difference_l1842_184244

/-- Proves the speed difference between two ferries given their travel conditions -/
theorem ferry_speed_difference :
  -- Ferry P's travel time
  let t_p : ℝ := 2 
  -- Ferry P's speed
  let v_p : ℝ := 8
  -- Ferry Q's route length multiplier
  let route_multiplier : ℝ := 3
  -- Additional time for Ferry Q's journey
  let additional_time : ℝ := 2

  -- Distance traveled by Ferry P
  let d_p : ℝ := t_p * v_p
  -- Distance traveled by Ferry Q
  let d_q : ℝ := route_multiplier * d_p
  -- Total time for Ferry Q's journey
  let t_q : ℝ := t_p + additional_time
  -- Speed of Ferry Q
  let v_q : ℝ := d_q / t_q

  -- The speed difference between Ferry Q and Ferry P is 4 km/hour
  v_q - v_p = 4 := by sorry

end NUMINAMATH_CALUDE_ferry_speed_difference_l1842_184244


namespace NUMINAMATH_CALUDE_supplier_A_lower_variance_l1842_184278

-- Define the purity data for Supplier A
def purity_A : List Nat := [72, 73, 74, 74, 74, 74, 74, 75, 75, 75, 76, 76, 76, 78, 79]

-- Define the purity data for Supplier B
def purity_B : List Nat := [72, 75, 72, 75, 78, 77, 73, 75, 76, 77, 71, 78, 79, 72, 75]

-- Define the statistical measures for Supplier A
def mean_A : Nat := 75
def median_A : Nat := 75
def mode_A : Nat := 74
def variance_A : Float := 3.7

-- Define the statistical measures for Supplier B
def mean_B : Nat := 75
def median_B : Nat := 75
def mode_B : Nat := 75
def variance_B : Float := 6.0

-- Theorem statement
theorem supplier_A_lower_variance :
  variance_A < variance_B ∧
  List.length purity_A = 15 ∧
  List.length purity_B = 15 ∧
  mean_A = mean_B ∧
  median_A = median_B :=
sorry

end NUMINAMATH_CALUDE_supplier_A_lower_variance_l1842_184278


namespace NUMINAMATH_CALUDE_units_digit_G_3_l1842_184228

-- Define G_n
def G (n : ℕ) : ℕ := 2^(2^(2^n)) + 1

-- Theorem statement
theorem units_digit_G_3 : G 3 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_3_l1842_184228


namespace NUMINAMATH_CALUDE_total_vehicles_calculation_l1842_184227

theorem total_vehicles_calculation (lanes : ℕ) (trucks_per_lane : ℕ) : 
  lanes = 4 →
  trucks_per_lane = 60 →
  (lanes * trucks_per_lane + lanes * (2 * lanes * trucks_per_lane)) = 2160 := by
  sorry

end NUMINAMATH_CALUDE_total_vehicles_calculation_l1842_184227


namespace NUMINAMATH_CALUDE_fifth_power_sum_equality_l1842_184231

theorem fifth_power_sum_equality : ∃! (n : ℕ), n > 0 ∧ 120^5 + 105^5 + 78^5 + 33^5 = n^5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_equality_l1842_184231


namespace NUMINAMATH_CALUDE_workers_count_l1842_184221

/-- Given a work that can be completed by some workers in 35 days,
    and adding 10 workers reduces the completion time by 10 days,
    prove that the original number of workers is 25. -/
theorem workers_count (work : ℕ) : ∃ (workers : ℕ), 
  (workers * 35 = (workers + 10) * 25) ∧ 
  workers = 25 := by
  sorry

end NUMINAMATH_CALUDE_workers_count_l1842_184221


namespace NUMINAMATH_CALUDE_sum_of_squares_l1842_184280

theorem sum_of_squares (a b : ℝ) (h1 : a - b = 10) (h2 : a * b = 25) : a^2 + b^2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1842_184280


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1842_184254

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  (a * Real.sin A + b * Real.sin B - c * Real.sin C) / (a * Real.sin B) = 2 * Real.sqrt 3 * Real.sin C →
  C = π / 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1842_184254


namespace NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l1842_184225

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ),
    n % 4 = 0 ∧
    n % 9 = 0 ∧
    n = 94444 + 90000 * k ∧
    k ≥ 0

theorem smallest_valid_number_last_four_digits :
  ∃ (n : ℕ),
    is_valid_number n ∧
    (∀ m, is_valid_number m → n ≤ m) ∧
    n % 10000 = 4444 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_last_four_digits_l1842_184225


namespace NUMINAMATH_CALUDE_M_mod_51_l1842_184232

def M : ℕ := sorry

theorem M_mod_51 : M % 51 = 34 := by sorry

end NUMINAMATH_CALUDE_M_mod_51_l1842_184232


namespace NUMINAMATH_CALUDE_sum_57_68_rounded_l1842_184269

/-- Rounds a number to the nearest ten -/
def roundToNearestTen (x : ℤ) : ℤ :=
  10 * ((x + 5) / 10)

/-- The sum of 57 and 68, when rounded to the nearest ten, equals 130 -/
theorem sum_57_68_rounded : roundToNearestTen (57 + 68) = 130 := by
  sorry

end NUMINAMATH_CALUDE_sum_57_68_rounded_l1842_184269


namespace NUMINAMATH_CALUDE_set_A_equals_explicit_set_l1842_184258

def set_A : Set (ℤ × ℤ) :=
  {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_equals_explicit_set : 
  set_A = {(-1, 0), (0, -1), (1, 0)} := by sorry

end NUMINAMATH_CALUDE_set_A_equals_explicit_set_l1842_184258


namespace NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l1842_184260

/-- A fraction a/b is a terminating decimal if b can be written as 2^m * 5^n for some non-negative integers m and n. -/
def IsTerminatingDecimal (a b : ℕ) : Prop :=
  ∃ (m n : ℕ), b = 2^m * 5^n

/-- The smallest positive integer n such that n/(n+107) is a terminating decimal is 143. -/
theorem smallest_n_for_terminating_decimal : 
  (∀ k : ℕ, 0 < k → k < 143 → ¬ IsTerminatingDecimal k (k + 107)) ∧ 
  IsTerminatingDecimal 143 (143 + 107) := by
  sorry

#check smallest_n_for_terminating_decimal

end NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l1842_184260


namespace NUMINAMATH_CALUDE_parrot_initial_phrases_l1842_184294

/-- The number of phrases a parrot initially knew, given the current number of phrases,
    the rate of learning, and the duration of ownership. -/
theorem parrot_initial_phrases (current_phrases : ℕ) (phrases_per_week : ℕ) (days_owned : ℕ) 
    (h1 : current_phrases = 17)
    (h2 : phrases_per_week = 2)
    (h3 : days_owned = 49) :
  current_phrases - (days_owned / 7 * phrases_per_week) = 3 := by
  sorry

#check parrot_initial_phrases

end NUMINAMATH_CALUDE_parrot_initial_phrases_l1842_184294


namespace NUMINAMATH_CALUDE_ferris_wheel_seats_l1842_184283

theorem ferris_wheel_seats (people_per_seat : ℕ) (total_people : ℕ) (h1 : people_per_seat = 6) (h2 : total_people = 84) :
  total_people / people_per_seat = 14 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_seats_l1842_184283


namespace NUMINAMATH_CALUDE_max_value_of_function_l1842_184218

theorem max_value_of_function (x : ℝ) : 
  1 / (x^2 + x + 1) ≤ 4/3 ∧ ∃ y : ℝ, 1 / (y^2 + y + 1) = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1842_184218


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l1842_184215

theorem rectangle_measurement_error (x : ℝ) : 
  (1 + x / 100) * (1 - 4 / 100) = 100.8 / 100 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l1842_184215


namespace NUMINAMATH_CALUDE_jennifer_sweets_sharing_l1842_184265

theorem jennifer_sweets_sharing (total_sweets : ℕ) (sweets_per_person : ℕ) (h1 : total_sweets = 1024) (h2 : sweets_per_person = 256) :
  (total_sweets / sweets_per_person) - 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jennifer_sweets_sharing_l1842_184265


namespace NUMINAMATH_CALUDE_not_perfect_square_l1842_184292

theorem not_perfect_square (a : ℤ) : a ≠ 0 → ¬∃ x : ℤ, a^2 + 4 = x^2 := by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1842_184292


namespace NUMINAMATH_CALUDE_cube_volume_from_face_diagonal_l1842_184210

theorem cube_volume_from_face_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_diagonal_l1842_184210


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1842_184206

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point (P : Point) (l1 l2 : Line) :
  P.liesOn l2 →
  l2.isParallelTo l1 →
  l1 = Line.mk 3 (-4) 6 →
  P = Point.mk 4 (-1) →
  l2 = Line.mk 3 (-4) (-16) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1842_184206


namespace NUMINAMATH_CALUDE_light_path_length_in_cube_l1842_184270

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents a light beam path in the cube -/
structure LightPath where
  start : Point3D
  reflection : Point3D
  cubeSideLength : ℝ

/-- Calculates the length of the light path -/
def lightPathLength (path : LightPath) : ℝ :=
  sorry

theorem light_path_length_in_cube (c : Cube) (path : LightPath) :
  c.sideLength = 10 ∧
  path.start = Point3D.mk 0 0 0 ∧
  path.reflection = Point3D.mk 10 3 4 ∧
  path.cubeSideLength = c.sideLength →
  lightPathLength path = 50 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_light_path_length_in_cube_l1842_184270


namespace NUMINAMATH_CALUDE_greatest_two_digit_product_12_l1842_184247

/-- A function that returns true if a number is a two-digit whole number --/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- A function that returns the product of digits of a two-digit number --/
def digitProduct (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

/-- Theorem stating that 62 is the greatest two-digit number whose digits have a product of 12 --/
theorem greatest_two_digit_product_12 :
  ∀ n : ℕ, isTwoDigit n → digitProduct n = 12 → n ≤ 62 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_product_12_l1842_184247


namespace NUMINAMATH_CALUDE_cloth_profit_theorem_l1842_184297

/-- Calculates the profit per meter of cloth given the total meters sold, 
    total selling price, and cost price per meter. -/
def profit_per_meter (meters_sold : ℕ) (total_selling_price : ℚ) (cost_price_per_meter : ℚ) : ℚ :=
  (total_selling_price - (meters_sold : ℚ) * cost_price_per_meter) / (meters_sold : ℚ)

/-- Theorem stating that given 85 meters of cloth sold for $8925 
    with a cost price of $90 per meter, the profit per meter is $15. -/
theorem cloth_profit_theorem :
  profit_per_meter 85 8925 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cloth_profit_theorem_l1842_184297


namespace NUMINAMATH_CALUDE_handshakes_and_highfives_l1842_184262

/-- The number of unique pairings in a group of n people -/
def uniquePairings (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of people at the gathering -/
def numberOfPeople : ℕ := 12

theorem handshakes_and_highfives :
  uniquePairings numberOfPeople = 66 ∧
  uniquePairings numberOfPeople = 66 := by
  sorry

#eval uniquePairings numberOfPeople

end NUMINAMATH_CALUDE_handshakes_and_highfives_l1842_184262


namespace NUMINAMATH_CALUDE_combine_like_terms_l1842_184216

theorem combine_like_terms (x y : ℝ) : 3 * x * y - 6 * x * y + (-2 * x * y) = -5 * x * y := by
  sorry

end NUMINAMATH_CALUDE_combine_like_terms_l1842_184216


namespace NUMINAMATH_CALUDE_special_sequence_representation_l1842_184251

/-- A sequence of natural numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, 0 < a n) ∧ 
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, a n < 2 * n)

/-- The main theorem statement -/
theorem special_sequence_representation (a : ℕ → ℕ) (h : SpecialSequence a) :
  ∀ m : ℕ, (∃ n, a n = m) ∨ (∃ k l, a k - a l = m) :=
sorry

end NUMINAMATH_CALUDE_special_sequence_representation_l1842_184251


namespace NUMINAMATH_CALUDE_mary_sticker_problem_l1842_184277

/-- Given the conditions about Mary's stickers, prove the total number of students in the class. -/
theorem mary_sticker_problem (total_stickers : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) 
  (stickers_per_other : ℕ) (leftover_stickers : ℕ) :
  total_stickers = 250 →
  friends = 10 →
  stickers_per_friend = 15 →
  stickers_per_other = 5 →
  leftover_stickers = 25 →
  ∃ (total_students : ℕ), total_students = 26 ∧ 
    total_stickers = friends * stickers_per_friend + 
    (total_students - friends - 1) * stickers_per_other + leftover_stickers :=
by sorry

end NUMINAMATH_CALUDE_mary_sticker_problem_l1842_184277


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1842_184241

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 3) :
  (1 / a + 1 / b) ≥ 1 + 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1842_184241


namespace NUMINAMATH_CALUDE_reading_materials_cost_l1842_184238

/-- The total cost of purchasing reading materials -/
def total_cost (a b : ℕ) : ℕ := 10 * a + 8 * b

/-- Theorem: The total cost of purchasing 'a' copies of type A reading materials
    at 10 yuan per copy and 'b' copies of type B reading materials at 8 yuan
    per copy is equal to 10a + 8b yuan. -/
theorem reading_materials_cost (a b : ℕ) :
  total_cost a b = 10 * a + 8 * b := by
  sorry

end NUMINAMATH_CALUDE_reading_materials_cost_l1842_184238


namespace NUMINAMATH_CALUDE_half_angle_quadrants_l1842_184261

theorem half_angle_quadrants (α : Real) : 
  (∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) →
  (∃ n : ℤ, 2 * n * π < α / 2 ∧ α / 2 < 2 * n * π + π / 2) ∨
  (∃ n : ℤ, (2 * n + 1) * π < α / 2 ∧ α / 2 < (2 * n + 1) * π + π / 2) := by
sorry


end NUMINAMATH_CALUDE_half_angle_quadrants_l1842_184261


namespace NUMINAMATH_CALUDE_rick_ironing_theorem_l1842_184291

/-- The number of dress shirts Rick can iron in one hour -/
def shirts_per_hour : ℕ := 4

/-- The number of dress pants Rick can iron in one hour -/
def pants_per_hour : ℕ := 3

/-- The number of hours Rick spends ironing dress shirts -/
def hours_ironing_shirts : ℕ := 3

/-- The number of hours Rick spends ironing dress pants -/
def hours_ironing_pants : ℕ := 5

/-- The total number of pieces of clothing Rick has ironed -/
def total_pieces : ℕ := shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants

theorem rick_ironing_theorem : total_pieces = 27 := by
  sorry

end NUMINAMATH_CALUDE_rick_ironing_theorem_l1842_184291


namespace NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l1842_184219

/-- An arithmetic sequence with common difference d and first term 2d -/
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 2*d + (n - 1)*d

/-- The value of k for which a_k is the geometric mean of a_1 and a_{2k+1} -/
def k_value : ℕ := 3

theorem arithmetic_sequence_geometric_mean (d : ℝ) (h : d ≠ 0) :
  let a := arithmetic_sequence d
  (a k_value)^2 = a 1 * a (2*k_value + 1) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_geometric_mean_l1842_184219


namespace NUMINAMATH_CALUDE_volume_to_surface_area_ratio_is_one_fifth_l1842_184259

/-- Represents a shape created by joining nine unit cubes -/
structure CubeShape where
  /-- The total number of unit cubes in the shape -/
  total_cubes : ℕ
  /-- The number of exposed faces of the shape -/
  exposed_faces : ℕ
  /-- Assertion that the total number of cubes is 9 -/
  cube_count : total_cubes = 9
  /-- Assertion that the number of exposed faces is 45 -/
  face_count : exposed_faces = 45

/-- Calculates the ratio of volume to surface area for the cube shape -/
def volumeToSurfaceAreaRatio (shape : CubeShape) : ℚ :=
  shape.total_cubes / shape.exposed_faces

/-- Theorem stating that the ratio of volume to surface area is 1/5 -/
theorem volume_to_surface_area_ratio_is_one_fifth (shape : CubeShape) :
  volumeToSurfaceAreaRatio shape = 1 / 5 := by
  sorry

#check volume_to_surface_area_ratio_is_one_fifth

end NUMINAMATH_CALUDE_volume_to_surface_area_ratio_is_one_fifth_l1842_184259


namespace NUMINAMATH_CALUDE_emerson_rowing_trip_l1842_184287

theorem emerson_rowing_trip (total_distance initial_distance second_part_distance : ℕ) 
  (h1 : total_distance = 39)
  (h2 : initial_distance = 6)
  (h3 : second_part_distance = 15) :
  total_distance - (initial_distance + second_part_distance) = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_emerson_rowing_trip_l1842_184287


namespace NUMINAMATH_CALUDE_article_cost_l1842_184253

/-- Proves that the cost of an article is 90, given the selling prices and gain difference --/
theorem article_cost (selling_price_1 selling_price_2 : ℕ) 
  (h1 : selling_price_1 = 340)
  (h2 : selling_price_2 = 350)
  (h3 : selling_price_2 - selling_price_1 = (4 : ℕ) * selling_price_1 / 100) :
  ∃ (cost : ℕ), cost = 90 ∧ 
    selling_price_1 = cost + (selling_price_1 - cost) ∧
    selling_price_2 = cost + (selling_price_2 - cost) :=
by
  sorry

end NUMINAMATH_CALUDE_article_cost_l1842_184253


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l1842_184200

theorem gift_wrapping_combinations : 
  let wrapping_paper := 8
  let ribbon := 5
  let gift_card := 4
  let gift_sticker := 6
  wrapping_paper * ribbon * gift_card * gift_sticker = 960 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l1842_184200


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1842_184242

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b) ∨ (a = c) ∨ (b = c)
  sumAngles : a + b + c = 180

-- Define the condition of angle ratio
def hasAngleRatio (t : IsoscelesTriangle) : Prop :=
  (t.a = 2 * t.b) ∨ (t.a = 2 * t.c) ∨ (t.b = 2 * t.c) ∨
  (2 * t.a = t.b) ∨ (2 * t.a = t.c) ∨ (2 * t.b = t.c)

-- Theorem statement
theorem isosceles_triangle_base_angle 
  (t : IsoscelesTriangle) 
  (h : hasAngleRatio t) : 
  (t.a = 45 ∧ (t.b = 45 ∨ t.c = 45)) ∨
  (t.b = 45 ∧ (t.a = 45 ∨ t.c = 45)) ∨
  (t.c = 45 ∧ (t.a = 45 ∨ t.b = 45)) ∨
  (t.a = 72 ∧ (t.b = 72 ∨ t.c = 72)) ∨
  (t.b = 72 ∧ (t.a = 72 ∨ t.c = 72)) ∨
  (t.c = 72 ∧ (t.a = 72 ∨ t.b = 72)) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l1842_184242


namespace NUMINAMATH_CALUDE_value_of_x_l1842_184264

theorem value_of_x (x y z : ℝ) 
  (h1 : x = y / 3) 
  (h2 : y = z / 6) 
  (h3 : z = 72) : 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1842_184264


namespace NUMINAMATH_CALUDE_solution_difference_l1842_184290

theorem solution_difference (p q : ℝ) : 
  p ≠ q →
  (p - 5) * (p + 3) = 24 * p - 72 →
  (q - 5) * (q + 3) = 24 * q - 72 →
  p > q →
  p - q = 20 := by
  sorry

end NUMINAMATH_CALUDE_solution_difference_l1842_184290


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1842_184237

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, foci at (-c, 0) and (c, 0),
    and an isosceles right triangle with hypotenuse connecting the foci,
    if the midpoint of the legs of this triangle lies on the hyperbola,
    then c/a = (√10 + √2)/2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∃ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ∧ 
              x = -c/2 ∧ 
              y = c/2) →
  c/a = (Real.sqrt 10 + Real.sqrt 2)/2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1842_184237


namespace NUMINAMATH_CALUDE_decimal_sum_difference_l1842_184250

theorem decimal_sum_difference : (0.5 : ℚ) - 0.03 + 0.007 = 0.477 := by sorry

end NUMINAMATH_CALUDE_decimal_sum_difference_l1842_184250


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1842_184293

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1842_184293


namespace NUMINAMATH_CALUDE_distinct_reals_integer_combination_l1842_184211

theorem distinct_reals_integer_combination (x y : ℝ) (h : x ≠ y) :
  ∃ (m n : ℤ), m * x + n * y > 0 ∧ n * x + m * y < 0 := by
  sorry

end NUMINAMATH_CALUDE_distinct_reals_integer_combination_l1842_184211


namespace NUMINAMATH_CALUDE_most_suitable_for_sample_survey_l1842_184284

/-- Represents a survey scenario -/
structure SurveyScenario where
  name : String
  quantity : Nat
  easySurvey : Bool

/-- Determines if a scenario is suitable for a sample survey -/
def suitableForSampleSurvey (scenario : SurveyScenario) : Prop :=
  scenario.quantity > 1000 ∧ ¬scenario.easySurvey

/-- The list of survey scenarios -/
def scenarios : List SurveyScenario := [
  { name := "Body temperature during H1N1", quantity := 100, easySurvey := false },
  { name := "Quality of Zongzi from Wufangzhai", quantity := 10000, easySurvey := false },
  { name := "Vision condition of classmates", quantity := 50, easySurvey := true },
  { name := "Mathematics learning in eighth grade", quantity := 200, easySurvey := true }
]

theorem most_suitable_for_sample_survey :
  ∃ (s : SurveyScenario), s ∈ scenarios ∧ 
  suitableForSampleSurvey s ∧ 
  (∀ (t : SurveyScenario), t ∈ scenarios → suitableForSampleSurvey t → s = t) :=
sorry

end NUMINAMATH_CALUDE_most_suitable_for_sample_survey_l1842_184284


namespace NUMINAMATH_CALUDE_square_equation_solution_l1842_184279

theorem square_equation_solution : ∃! (N : ℕ), N > 0 ∧ 36^2 * 60^2 = 30^2 * N^2 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1842_184279


namespace NUMINAMATH_CALUDE_seminar_attendees_l1842_184233

theorem seminar_attendees (total : ℕ) (company_a : ℕ) : 
  total = 185 →
  company_a = 30 →
  20 = total - (company_a + 2 * company_a + (company_a + 10) + (company_a + 5)) :=
by sorry

end NUMINAMATH_CALUDE_seminar_attendees_l1842_184233


namespace NUMINAMATH_CALUDE_only_fatigued_drivers_accidents_correlative_l1842_184229

/-- Represents a pair of quantities -/
inductive QuantityPair
  | StudentGradesWeight
  | TimeDisplacement
  | WaterVolumeWeight
  | FatiguedDriversAccidents

/-- Describes the relationship between two quantities -/
inductive Relationship
  | Correlative
  | Functional
  | Independent

/-- Function that determines the relationship for a given pair of quantities -/
def determineRelationship (pair : QuantityPair) : Relationship :=
  match pair with
  | QuantityPair.StudentGradesWeight => Relationship.Independent
  | QuantityPair.TimeDisplacement => Relationship.Functional
  | QuantityPair.WaterVolumeWeight => Relationship.Functional
  | QuantityPair.FatiguedDriversAccidents => Relationship.Correlative

/-- Theorem stating that only the FatiguedDriversAccidents pair has a correlative relationship -/
theorem only_fatigued_drivers_accidents_correlative :
  ∀ (pair : QuantityPair),
    determineRelationship pair = Relationship.Correlative ↔ pair = QuantityPair.FatiguedDriversAccidents :=
by
  sorry


end NUMINAMATH_CALUDE_only_fatigued_drivers_accidents_correlative_l1842_184229


namespace NUMINAMATH_CALUDE_total_attendance_l1842_184220

def wedding_reception (bride_couples groom_couples friends : ℕ) : ℕ :=
  2 * (bride_couples + groom_couples) + friends

theorem total_attendance : wedding_reception 20 20 100 = 180 := by
  sorry

end NUMINAMATH_CALUDE_total_attendance_l1842_184220


namespace NUMINAMATH_CALUDE_problem_solution_l1842_184257

theorem problem_solution (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 3) :
  (a^2 + b^2 = 22) ∧ ((a - 2) * (b + 2) = 7) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1842_184257


namespace NUMINAMATH_CALUDE_cube_inequality_l1842_184243

theorem cube_inequality (a b : ℝ) (h : a < 0 ∧ 0 < b) : a^3 < b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l1842_184243


namespace NUMINAMATH_CALUDE_rectangular_hyperbola_equation_l1842_184275

/-- A rectangular hyperbola with coordinate axes as its axes of symmetry
    passing through the point (2, √2) has the equation x² - y² = 2 -/
theorem rectangular_hyperbola_equation :
  ∀ (f : ℝ → ℝ → Prop),
    (∀ x y, f x y ↔ x^2 - y^2 = 2) →  -- Definition of the hyperbola equation
    (∀ x, f x 0 ↔ f 0 x) →            -- Symmetry about y = x
    (∀ x, f x 0 ↔ f (-x) 0) →         -- Symmetry about y-axis
    (∀ y, f 0 y ↔ f 0 (-y)) →         -- Symmetry about x-axis
    f 2 (Real.sqrt 2) →               -- Point (2, √2) lies on the hyperbola
    ∀ x y, f x y ↔ x^2 - y^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_hyperbola_equation_l1842_184275


namespace NUMINAMATH_CALUDE_emily_beads_used_l1842_184281

/-- The number of beads Emily has used so far -/
def beads_used (necklaces_made : ℕ) (beads_per_necklace : ℕ) (necklaces_given_away : ℕ) : ℕ :=
  necklaces_made * beads_per_necklace - necklaces_given_away * beads_per_necklace

/-- Theorem stating that Emily has used 92 beads so far -/
theorem emily_beads_used :
  beads_used 35 4 12 = 92 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_used_l1842_184281


namespace NUMINAMATH_CALUDE_big_eighteen_soccer_league_games_l1842_184245

/-- Calculates the number of games in a soccer league with specific rules --/
def soccer_league_games (num_divisions : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let intra_games := num_divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_games := num_divisions * teams_per_division * (num_divisions - 1) * teams_per_division * inter_division_games
  (intra_games + inter_games) / 2

/-- The Big Eighteen Soccer League schedule theorem --/
theorem big_eighteen_soccer_league_games : 
  soccer_league_games 3 6 3 2 = 351 := by
  sorry

end NUMINAMATH_CALUDE_big_eighteen_soccer_league_games_l1842_184245


namespace NUMINAMATH_CALUDE_net_gain_calculation_l1842_184268

def initial_value : ℝ := 15000
def profit_percentage : ℝ := 0.20
def loss_percentage : ℝ := 0.15
def transaction_fee : ℝ := 300

def first_sale_price : ℝ := initial_value * (1 + profit_percentage)
def second_sale_price : ℝ := first_sale_price * (1 - loss_percentage)
def total_cost : ℝ := second_sale_price + transaction_fee

theorem net_gain_calculation :
  first_sale_price - total_cost = 2400 := by sorry

end NUMINAMATH_CALUDE_net_gain_calculation_l1842_184268


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l1842_184299

/-- If the sum of two monomials 2a^5*b^(2m+4) and a^(2n-3)*b^8 is still a monomial,
    then m = 2 and n = 4 -/
theorem monomial_sum_condition (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ) (p q : ℕ), 2 * a^5 * b^(2*m+4) + a^(2*n-3) * b^8 = k * a^p * b^q) →
  m = 2 ∧ n = 4 := by
sorry


end NUMINAMATH_CALUDE_monomial_sum_condition_l1842_184299


namespace NUMINAMATH_CALUDE_derivative_of_odd_is_even_l1842_184234

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem derivative_of_odd_is_even
  (f : ℝ → ℝ) (hf : Differentiable ℝ f) (hodd : OddFunction f) :
  OddFunction f → ∀ x, deriv f (-x) = deriv f x :=
sorry

end NUMINAMATH_CALUDE_derivative_of_odd_is_even_l1842_184234


namespace NUMINAMATH_CALUDE_specific_can_stack_total_l1842_184295

/-- Represents a stack of cans forming an arithmetic sequence -/
structure CanStack where
  bottom_layer : ℕ
  difference : ℕ
  top_layer : ℕ

/-- Calculates the number of layers in the stack -/
def num_layers (stack : CanStack) : ℕ :=
  (stack.bottom_layer - stack.top_layer) / stack.difference + 1

/-- Calculates the total number of cans in the stack -/
def total_cans (stack : CanStack) : ℕ :=
  let n := num_layers stack
  (n * (stack.bottom_layer + stack.top_layer)) / 2

/-- Theorem stating that a specific can stack contains 172 cans -/
theorem specific_can_stack_total :
  let stack : CanStack := { bottom_layer := 35, difference := 4, top_layer := 1 }
  total_cans stack = 172 := by
  sorry

end NUMINAMATH_CALUDE_specific_can_stack_total_l1842_184295


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l1842_184246

/-- Proves that the initial ratio of milk to water in a mixture is 3:2, given specific conditions -/
theorem initial_milk_water_ratio
  (total_initial : ℝ)
  (water_added : ℝ)
  (milk : ℝ)
  (water : ℝ)
  (h1 : total_initial = 165)
  (h2 : water_added = 66)
  (h3 : milk + water = total_initial)
  (h4 : milk / (water + water_added) = 3 / 4)
  : milk / water = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l1842_184246


namespace NUMINAMATH_CALUDE_meeting_time_is_lcm_l1842_184214

/-- The time (in seconds) it takes for all racers to meet at the starting point for the second time -/
def meeting_time : ℕ := 3600

/-- The lap time (in seconds) for Racing Magic -/
def racing_magic_time : ℕ := 60

/-- The lap time (in seconds) for Charging Bull -/
def charging_bull_time : ℕ := 90

/-- The lap time (in seconds) for Swift Shadow -/
def swift_shadow_time : ℕ := 80

/-- The lap time (in seconds) for Speedy Storm -/
def speedy_storm_time : ℕ := 100

/-- Theorem stating that the meeting time is the least common multiple of all racers' lap times -/
theorem meeting_time_is_lcm :
  meeting_time = Nat.lcm racing_magic_time (Nat.lcm charging_bull_time (Nat.lcm swift_shadow_time speedy_storm_time)) :=
by sorry

end NUMINAMATH_CALUDE_meeting_time_is_lcm_l1842_184214


namespace NUMINAMATH_CALUDE_cow_characteristic_difference_l1842_184271

def total_cows : ℕ := 600
def male_ratio : ℕ := 5
def female_ratio : ℕ := 3
def transgender_ratio : ℕ := 2

def male_horned_percentage : ℚ := 50 / 100
def male_spotted_percentage : ℚ := 40 / 100
def male_brown_percentage : ℚ := 20 / 100

def female_spotted_percentage : ℚ := 35 / 100
def female_horned_percentage : ℚ := 25 / 100
def female_white_percentage : ℚ := 60 / 100

def transgender_unique_pattern_percentage : ℚ := 45 / 100
def transgender_spotted_horned_percentage : ℚ := 30 / 100
def transgender_black_percentage : ℚ := 50 / 100

theorem cow_characteristic_difference :
  let total_ratio := male_ratio + female_ratio + transgender_ratio
  let male_count := (male_ratio : ℚ) / total_ratio * total_cows
  let female_count := (female_ratio : ℚ) / total_ratio * total_cows
  let transgender_count := (transgender_ratio : ℚ) / total_ratio * total_cows
  let spotted_females := female_spotted_percentage * female_count
  let horned_males := male_horned_percentage * male_count
  let brown_males := male_brown_percentage * male_count
  let unique_pattern_transgender := transgender_unique_pattern_percentage * transgender_count
  let white_horned_females := female_horned_percentage * female_white_percentage * female_count
  let characteristic_sum := horned_males + brown_males + unique_pattern_transgender + white_horned_females
  spotted_females - characteristic_sum = -291 := by sorry

end NUMINAMATH_CALUDE_cow_characteristic_difference_l1842_184271


namespace NUMINAMATH_CALUDE_first_car_manufacture_year_l1842_184213

/-- Given three cars with different manufacture dates, prove that the first car was made in 1970. -/
theorem first_car_manufacture_year :
  ∀ (year1 year2 year3 : ℕ),
  year1 < year2 → year2 < year3 →
  year2 = year1 + 10 →
  year3 = year2 + 20 →
  year3 = 2000 →
  year1 = 1970 := by
sorry

end NUMINAMATH_CALUDE_first_car_manufacture_year_l1842_184213


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1842_184286

theorem absolute_value_equality (y : ℝ) : |y + 2| = |y - 3| → y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1842_184286


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1842_184266

theorem solve_linear_equation (x : ℝ) : 2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1842_184266


namespace NUMINAMATH_CALUDE_parabola_p_value_l1842_184212

/-- A parabola with focus distance 12 and y-axis distance 9 has p = 6 -/
theorem parabola_p_value (p : ℝ) (A : ℝ × ℝ) :
  p > 0 →
  A.2^2 = 2 * p * A.1 →
  dist A (p / 2, 0) = 12 →
  A.1 = 9 →
  p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_p_value_l1842_184212


namespace NUMINAMATH_CALUDE_sophies_perceived_height_l1842_184255

/-- Calculates the perceived height in centimeters when doubled in a mirror reflection. -/
def perceivedHeightCm (actualHeightInches : ℝ) (conversionRate : ℝ) : ℝ :=
  2 * actualHeightInches * conversionRate

/-- Theorem stating that Sophie's perceived height in the mirror is 250.0 cm. -/
theorem sophies_perceived_height :
  let actualHeight : ℝ := 50
  let conversionRate : ℝ := 2.50
  perceivedHeightCm actualHeight conversionRate = 250.0 := by
  sorry

end NUMINAMATH_CALUDE_sophies_perceived_height_l1842_184255


namespace NUMINAMATH_CALUDE_gcd_of_f_over_primes_ge_11_l1842_184267

-- Define the function f(p)
def f (p : ℕ) : ℕ := p^6 - 7*p^2 + 6

-- Define the set of prime numbers greater than or equal to 11
def P : Set ℕ := {p : ℕ | Nat.Prime p ∧ p ≥ 11}

-- Theorem statement
theorem gcd_of_f_over_primes_ge_11 : 
  ∃ (d : ℕ), d > 0 ∧ (∀ (p : ℕ), p ∈ P → (f p).gcd d = d) ∧ 
  (∀ (m : ℕ), (∀ (p : ℕ), p ∈ P → (f p).gcd m = m) → m ≤ d) ∧ d = 16 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_f_over_primes_ge_11_l1842_184267


namespace NUMINAMATH_CALUDE_rivet_distribution_l1842_184240

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

end NUMINAMATH_CALUDE_rivet_distribution_l1842_184240


namespace NUMINAMATH_CALUDE_solve_quadratic_l1842_184276

-- Define the universal set U
def U : Set ℕ := {2, 3, 5}

-- Define the set A
def A (b c : ℤ) : Set ℕ := {x ∈ U | x^2 + b*x + c = 0}

-- Define the complement of A with respect to U
def complement_A (b c : ℤ) : Set ℕ := U \ A b c

-- Theorem statement
theorem solve_quadratic (b c : ℤ) : complement_A b c = {2} → b = -8 ∧ c = 15 := by
  sorry


end NUMINAMATH_CALUDE_solve_quadratic_l1842_184276


namespace NUMINAMATH_CALUDE_pages_to_read_tomorrow_l1842_184256

/-- Given a book and Julie's reading progress, calculate the number of pages to read tomorrow --/
theorem pages_to_read_tomorrow (total_pages yesterday_pages : ℕ) : 
  total_pages = 120 →
  yesterday_pages = 12 →
  (total_pages - (yesterday_pages + 2 * yesterday_pages)) / 2 = 42 := by
sorry

end NUMINAMATH_CALUDE_pages_to_read_tomorrow_l1842_184256


namespace NUMINAMATH_CALUDE_caravan_feet_heads_difference_l1842_184252

/-- Represents the number of feet for each animal type and humans -/
def feet_count (animal : String) : ℕ :=
  match animal with
  | "hen" => 2
  | "goat" => 4
  | "camel" => 4
  | "human" => 2
  | _ => 0

/-- Calculates the total number of heads in the caravan -/
def total_heads : ℕ := 50 + 45 + 8 + 15

/-- Calculates the total number of feet in the caravan -/
def total_feet : ℕ := 
  50 * feet_count "hen" + 
  45 * feet_count "goat" + 
  8 * feet_count "camel" + 
  15 * feet_count "human"

/-- States that the difference between total feet and total heads in the caravan is 224 -/
theorem caravan_feet_heads_difference : total_feet - total_heads = 224 := by
  sorry

end NUMINAMATH_CALUDE_caravan_feet_heads_difference_l1842_184252


namespace NUMINAMATH_CALUDE_willie_stickers_l1842_184203

/-- Given that Willie starts with 124 stickers and gives away 43 stickers,
    prove that he ends up with 81 stickers. -/
theorem willie_stickers : ∀ (initial given_away final : ℕ),
  initial = 124 →
  given_away = 43 →
  final = initial - given_away →
  final = 81 := by
  sorry

end NUMINAMATH_CALUDE_willie_stickers_l1842_184203


namespace NUMINAMATH_CALUDE_max_value_sum_l1842_184236

theorem max_value_sum (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e)
  (h_sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 504) :
  ∃ (N a_N b_N c_N d_N e_N : ℝ),
    (∀ x y z w v : ℝ, x > 0 → y > 0 → z > 0 → w > 0 → v > 0 →
      x^2 + y^2 + z^2 + w^2 + v^2 = 504 →
      x*z + 3*y*z + 4*z*w + 8*z*v ≤ N) ∧
    a_N > 0 ∧ b_N > 0 ∧ c_N > 0 ∧ d_N > 0 ∧ e_N > 0 ∧
    a_N^2 + b_N^2 + c_N^2 + d_N^2 + e_N^2 = 504 ∧
    a_N*c_N + 3*b_N*c_N + 4*c_N*d_N + 8*c_N*e_N = N ∧
    N + a_N + b_N + c_N + d_N + e_N = 96 * Real.sqrt 2 + 6 * Real.sqrt 21 + 756 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_l1842_184236


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1842_184222

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ 1}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1842_184222


namespace NUMINAMATH_CALUDE_angle_Y_value_l1842_184201

-- Define the angles as real numbers
def A : ℝ := 50
def Z : ℝ := 50

-- Define the theorem
theorem angle_Y_value :
  ∀ B X Y : ℝ,
  A + B = 180 →
  X = Y →
  B + Z = 180 →
  B + X + Y = 180 →
  Y = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_angle_Y_value_l1842_184201


namespace NUMINAMATH_CALUDE_cylinder_section_area_l1842_184204

/-- The area of a plane section in a cylinder --/
theorem cylinder_section_area (r h : ℝ) (arc_angle : ℝ) : 
  r = 8 → h = 10 → arc_angle = 150 * π / 180 →
  ∃ (area : ℝ), area = (400/3) * π + 40 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_section_area_l1842_184204


namespace NUMINAMATH_CALUDE_bird_cage_problem_l1842_184239

theorem bird_cage_problem (total : ℕ) (remaining : ℕ) : total = 60 ∧ remaining = 8 →
  ∃ (x : ℚ), x = 2/3 ∧ 
  (total - total/3 - (total - total/3)*2/5) * (1 - x) = remaining :=
by sorry

end NUMINAMATH_CALUDE_bird_cage_problem_l1842_184239


namespace NUMINAMATH_CALUDE_mars_mission_cost_share_l1842_184249

-- Define the total cost in billions of dollars
def total_cost : ℕ := 25

-- Define the number of people sharing the cost in millions
def num_people : ℕ := 200

-- Define the conversion factor from billions to millions
def billion_to_million : ℕ := 1000

-- Theorem statement
theorem mars_mission_cost_share :
  (total_cost * billion_to_million) / num_people = 125 := by
sorry

end NUMINAMATH_CALUDE_mars_mission_cost_share_l1842_184249


namespace NUMINAMATH_CALUDE_marble_jar_theorem_l1842_184248

/-- Represents a jar of marbles with orange, purple, and yellow colors. -/
structure MarbleJar where
  orange : ℝ
  purple : ℝ
  yellow : ℝ

/-- The total number of marbles in the jar. -/
def MarbleJar.total (jar : MarbleJar) : ℝ :=
  jar.orange + jar.purple + jar.yellow

/-- A jar satisfying the given conditions. -/
def specialJar : MarbleJar :=
  { orange := 0,  -- placeholder values
    purple := 0,
    yellow := 0 }

theorem marble_jar_theorem (jar : MarbleJar) :
  jar.purple + jar.yellow = 7 →
  jar.orange + jar.yellow = 5 →
  jar.orange + jar.purple = 9 →
  jar.total = 10.5 := by
  sorry

#check marble_jar_theorem

end NUMINAMATH_CALUDE_marble_jar_theorem_l1842_184248


namespace NUMINAMATH_CALUDE_specific_courses_not_consecutive_l1842_184263

-- Define the number of courses
def n : ℕ := 6

-- Define the number of specific courses we're interested in
def k : ℕ := 3

-- Theorem statement
theorem specific_courses_not_consecutive :
  (n.factorial : ℕ) - (n - k + 1).factorial * k.factorial = 576 := by
  sorry

end NUMINAMATH_CALUDE_specific_courses_not_consecutive_l1842_184263


namespace NUMINAMATH_CALUDE_no_quadratic_factorization_l1842_184272

theorem no_quadratic_factorization :
  ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
    x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) :=
by sorry

end NUMINAMATH_CALUDE_no_quadratic_factorization_l1842_184272


namespace NUMINAMATH_CALUDE_fifth_minus_fourth_cube_volume_l1842_184226

/-- The volume of a cube with side length n -/
def cube_volume (n : ℕ) : ℕ := n ^ 3

/-- The difference in volume between two cubes in the sequence -/
def volume_difference (m n : ℕ) : ℕ := cube_volume m - cube_volume n

theorem fifth_minus_fourth_cube_volume : volume_difference 5 4 = 61 := by
  sorry

end NUMINAMATH_CALUDE_fifth_minus_fourth_cube_volume_l1842_184226


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1842_184209

theorem complex_fraction_simplification (i : ℂ) :
  i^2 = -1 →
  (2 + i) * (3 - 4*i) / (2 - i) = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1842_184209


namespace NUMINAMATH_CALUDE_binomial_coefficient_self_l1842_184235

theorem binomial_coefficient_self : (510 : ℕ).choose 510 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_self_l1842_184235


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l1842_184224

def trailing_zeros (n : ℕ) : ℕ := sorry

theorem product_trailing_zeros :
  trailing_zeros (50 * 720 * 125) = 5 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l1842_184224


namespace NUMINAMATH_CALUDE_comparison_inequalities_l1842_184298

theorem comparison_inequalities :
  (Real.sqrt 37 > 6) ∧ ((Real.sqrt 5 - 1) / 2 > 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_comparison_inequalities_l1842_184298


namespace NUMINAMATH_CALUDE_max_min_S_l1842_184285

theorem max_min_S (x y z : ℚ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (eq1 : 3 * x + 2 * y + z = 5)
  (eq2 : x + y - z = 2)
  (S : ℚ := 2 * x + y - z) :
  (∀ s : ℚ, S ≤ s → s ≤ 3) ∧ (∀ s : ℚ, 2 ≤ s → s ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_max_min_S_l1842_184285


namespace NUMINAMATH_CALUDE_hedge_cost_proof_l1842_184202

/-- The number of concrete blocks used in each section of the hedge. -/
def blocks_per_section : ℕ := 30

/-- The cost of each concrete block in dollars. -/
def cost_per_block : ℕ := 2

/-- The number of sections in the hedge. -/
def number_of_sections : ℕ := 8

/-- The total cost of concrete blocks for the hedge. -/
def total_cost : ℕ := blocks_per_section * number_of_sections * cost_per_block

theorem hedge_cost_proof : total_cost = 480 := by
  sorry

end NUMINAMATH_CALUDE_hedge_cost_proof_l1842_184202


namespace NUMINAMATH_CALUDE_world_cup_teams_l1842_184288

theorem world_cup_teams (total_gifts : ℕ) (gifts_per_team : ℕ) : 
  total_gifts = 14 → 
  gifts_per_team = 2 → 
  total_gifts / gifts_per_team = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_world_cup_teams_l1842_184288
