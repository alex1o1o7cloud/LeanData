import Mathlib

namespace NUMINAMATH_CALUDE_limit_x2y_over_x2_plus_y2_is_zero_l110_11026

open Real

/-- The limit of (x^2 * y) / (x^2 + y^2) as x and y approach 0 is 0. -/
theorem limit_x2y_over_x2_plus_y2_is_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    0 < Real.sqrt (x^2 + y^2) ∧ Real.sqrt (x^2 + y^2) < δ →
    |x^2 * y / (x^2 + y^2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_x2y_over_x2_plus_y2_is_zero_l110_11026


namespace NUMINAMATH_CALUDE_lansing_elementary_students_l110_11001

/-- The number of elementary schools in Lansing -/
def num_schools : ℕ := 25

/-- The number of students in each elementary school in Lansing -/
def students_per_school : ℕ := 247

/-- The total number of elementary students in Lansing -/
def total_students : ℕ := num_schools * students_per_school

theorem lansing_elementary_students :
  total_students = 6175 :=
by sorry

end NUMINAMATH_CALUDE_lansing_elementary_students_l110_11001


namespace NUMINAMATH_CALUDE_first_term_of_a_10_l110_11031

def first_term (n : ℕ) : ℕ :=
  1 + 2 * (List.range n).sum

theorem first_term_of_a_10 : first_term 10 = 91 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_a_10_l110_11031


namespace NUMINAMATH_CALUDE_shopkeeper_weight_problem_l110_11007

theorem shopkeeper_weight_problem (actual_weight : ℝ) (profit_percentage : ℝ) :
  actual_weight = 800 →
  profit_percentage = 25 →
  ∃ standard_weight : ℝ,
    standard_weight = 1000 ∧
    (standard_weight - actual_weight) / actual_weight * 100 = profit_percentage :=
by sorry

end NUMINAMATH_CALUDE_shopkeeper_weight_problem_l110_11007


namespace NUMINAMATH_CALUDE_books_per_shelf_l110_11052

theorem books_per_shelf 
  (total_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : total_shelves = 150) 
  (h2 : total_books = 2250) : 
  total_books / total_shelves = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l110_11052


namespace NUMINAMATH_CALUDE_system_solution_unique_l110_11010

theorem system_solution_unique : 
  ∃! (x y : ℚ), (6 * x = -9 - 3 * y) ∧ (4 * x = 5 * y - 34) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l110_11010


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l110_11025

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 1  -- diameter of smaller circle
  let d₂ : ℝ := 3  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let area_small : ℝ := π * r₁^2  -- area of smaller circle
  let area_large : ℝ := π * r₂^2  -- area of larger circle
  let area_between : ℝ := area_large - area_small  -- area between circles
  (area_between / area_small) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l110_11025


namespace NUMINAMATH_CALUDE_tomatoes_sold_on_saturday_l110_11088

theorem tomatoes_sold_on_saturday (initial_shipment : ℕ) (rotted_amount : ℕ) (final_amount : ℕ) :
  initial_shipment = 1000 →
  rotted_amount = 200 →
  final_amount = 2500 →
  ∃ (sold_amount : ℕ),
    sold_amount = 300 ∧
    final_amount = initial_shipment - sold_amount - rotted_amount + 2 * initial_shipment :=
by sorry

end NUMINAMATH_CALUDE_tomatoes_sold_on_saturday_l110_11088


namespace NUMINAMATH_CALUDE_triangle_solution_l110_11027

theorem triangle_solution (a b c : ℝ) (A B C : ℝ) : 
  a = 42 →
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  C = π - A - B →
  b = a * Real.sin B / Real.sin A →
  c = a * Real.sin C / Real.sin A →
  b = 21 * Real.sqrt 6 ∧ c = 21 * (Real.sqrt 3 + 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_solution_l110_11027


namespace NUMINAMATH_CALUDE_hot_dog_packaging_l110_11043

theorem hot_dog_packaging :
  let total_hot_dogs : ℕ := 25197625
  let package_size : ℕ := 5
  let full_sets : ℕ := 5039525
  total_hot_dogs / package_size = full_sets ∧
  total_hot_dogs % package_size = 0 := by
sorry

end NUMINAMATH_CALUDE_hot_dog_packaging_l110_11043


namespace NUMINAMATH_CALUDE_fraction_division_five_sixths_divided_by_nine_tenths_l110_11021

theorem fraction_division (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : c ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem five_sixths_divided_by_nine_tenths :
  (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_five_sixths_divided_by_nine_tenths_l110_11021


namespace NUMINAMATH_CALUDE_intersection_M_N_l110_11045

def M : Set ℝ := {x | (x - 1)^2 < 4}

def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l110_11045


namespace NUMINAMATH_CALUDE_red_apples_count_l110_11038

def basket_problem (total_apples green_apples : ℕ) : Prop :=
  total_apples = 9 ∧ green_apples = 2 → total_apples - green_apples = 7

theorem red_apples_count : basket_problem 9 2 := by
  sorry

end NUMINAMATH_CALUDE_red_apples_count_l110_11038


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l110_11006

theorem sum_of_three_numbers : 72.52 + 12.23 + 5.21 = 89.96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l110_11006


namespace NUMINAMATH_CALUDE_min_removals_for_three_by_three_l110_11070

/-- Represents a 3x3 square figure made of matches -/
structure MatchSquare where
  size : Nat
  total_matches : Nat
  matches_per_side : Nat

/-- Defines the properties of our specific 3x3 match square -/
def three_by_three_square : MatchSquare :=
  { size := 3
  , total_matches := 24
  , matches_per_side := 1 }

/-- Defines what it means for a number of removals to be valid -/
def is_valid_removal (square : MatchSquare) (removals : Nat) : Prop :=
  removals ≤ square.total_matches ∧
  ∀ (x y : Nat), x < square.size ∧ y < square.size →
    ∃ (side : Nat), side < 4 ∧ 
      (removals > (x * square.size + y) * 4 + side)

/-- The main theorem statement -/
theorem min_removals_for_three_by_three (square : MatchSquare) 
  (h1 : square = three_by_three_square) :
  ∃ (n : Nat), is_valid_removal square n ∧
    ∀ (m : Nat), m < n → ¬ is_valid_removal square m :=
  sorry

end NUMINAMATH_CALUDE_min_removals_for_three_by_three_l110_11070


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l110_11015

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h1 : ArithmeticSequence a) 
    (h2 : a 1 + 3 * a 8 + a 15 = 120) : 
    2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l110_11015


namespace NUMINAMATH_CALUDE_course_selection_count_l110_11092

def num_courses_A : ℕ := 3
def num_courses_B : ℕ := 4
def total_courses_selected : ℕ := 3

theorem course_selection_count : 
  (Nat.choose num_courses_A 2 * Nat.choose num_courses_B 1) + 
  (Nat.choose num_courses_A 1 * Nat.choose num_courses_B 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_count_l110_11092


namespace NUMINAMATH_CALUDE_vector_expression_evaluation_l110_11076

/-- Prove that the vector expression evaluates to the given result -/
theorem vector_expression_evaluation :
  (⟨3, -8⟩ : ℝ × ℝ) - 5 • (⟨2, -4⟩ : ℝ × ℝ) = (⟨-7, 12⟩ : ℝ × ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_expression_evaluation_l110_11076


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l110_11008

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_12_less_than_square : 
  (∃ n : ℕ, is_perfect_square n ∧ is_prime (n - 12)) ∧ 
  (∀ m : ℕ, is_perfect_square m ∧ is_prime (m - 12) → m - 12 ≥ 13) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l110_11008


namespace NUMINAMATH_CALUDE_prob_at_least_one_l110_11095

/-- The probability of possessing at least one of two independent events,
    given their individual probabilities -/
theorem prob_at_least_one (p_ballpoint p_ink : ℚ) 
  (h_ballpoint : p_ballpoint = 3/5)
  (h_ink : p_ink = 2/3)
  (h_independent : True) -- Assumption of independence
  : p_ballpoint + p_ink - p_ballpoint * p_ink = 13/15 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_l110_11095


namespace NUMINAMATH_CALUDE_coefficient_equals_168_l110_11042

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x^2y^2 in (1+x)^8(1+y)^4
def coefficient_x2y2 : ℕ := binomial 8 2 * binomial 4 2

-- Theorem statement
theorem coefficient_equals_168 : coefficient_x2y2 = 168 := by sorry

end NUMINAMATH_CALUDE_coefficient_equals_168_l110_11042


namespace NUMINAMATH_CALUDE_hayden_earnings_354_l110_11036

/-- Represents Hayden's work day at the limousine company -/
structure HaydenWorkDay where
  shortRideReimbursement : ℕ := 3
  longRideReimbursement : ℕ := 4
  baseHourlyWage : ℕ := 15
  shortRideBonus : ℕ := 7
  longRideBonus : ℕ := 10
  goodReviewBonus : ℕ := 20
  excellentReviewBonus : ℕ := 30
  totalRides : ℕ := 5
  longRides : ℕ := 2
  hoursWorked : ℕ := 11
  shortRideGas : ℕ := 10
  longRideGas : ℕ := 15
  tollFee : ℕ := 6
  numTolls : ℕ := 2
  goodReviews : ℕ := 2
  excellentReviews : ℕ := 1

/-- Calculates Hayden's total earnings for the day -/
def calculateEarnings (day : HaydenWorkDay) : ℕ :=
  let baseEarnings := day.baseHourlyWage * day.hoursWorked
  let shortRides := day.totalRides - day.longRides
  let rideBonuses := shortRides * day.shortRideBonus + day.longRides * day.longRideBonus
  let gasReimbursement := day.shortRideGas * day.shortRideReimbursement + day.longRideGas * day.longRideReimbursement
  let reviewBonuses := day.goodReviews * day.goodReviewBonus + day.excellentReviews * day.excellentReviewBonus
  let totalBeforeTolls := baseEarnings + rideBonuses + gasReimbursement + reviewBonuses
  totalBeforeTolls - (day.numTolls * day.tollFee)

/-- Theorem stating that Hayden's earnings for the day equal $354 -/
theorem hayden_earnings_354 (day : HaydenWorkDay) : calculateEarnings day = 354 := by
  sorry

end NUMINAMATH_CALUDE_hayden_earnings_354_l110_11036


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l110_11066

/-- A positive geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_min_value (a : ℕ → ℝ) (m n : ℕ) :
  geometric_sequence a →
  (a 6 = a 5 + 2 * a 4) →
  (Real.sqrt (a m * a n) = 4 * a 1) →
  (∃ min_value : ℝ, min_value = (3 + 2 * Real.sqrt 2) / 6 ∧
    ∀ x y : ℕ, (Real.sqrt (a x * a y) = 4 * a 1) → (1 / x + 2 / y) ≥ min_value) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l110_11066


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l110_11040

theorem ferris_wheel_capacity (num_seats : ℕ) (people_per_seat : ℕ) 
  (h1 : num_seats = 14) (h2 : people_per_seat = 6) : 
  num_seats * people_per_seat = 84 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l110_11040


namespace NUMINAMATH_CALUDE_kvass_price_after_increases_l110_11075

theorem kvass_price_after_increases (x y : ℝ) : 
  x + y = 1 →
  1.2 * (0.5 * x + y) = 1 →
  1.44 * y < 1 :=
by sorry

end NUMINAMATH_CALUDE_kvass_price_after_increases_l110_11075


namespace NUMINAMATH_CALUDE_q_sum_zero_five_l110_11069

/-- A monic polynomial of degree 5 -/
def MonicPolynomial5 (q : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, q x = x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + q 0

/-- The main theorem -/
theorem q_sum_zero_five
  (q : ℝ → ℝ)
  (monic : MonicPolynomial5 q)
  (h1 : q 1 = 24)
  (h2 : q 2 = 48)
  (h3 : q 3 = 72) :
  q 0 + q 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_q_sum_zero_five_l110_11069


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l110_11000

def is_valid_digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def is_unique_assignment (K P O C S R T : ℕ) : Prop :=
  is_valid_digit K ∧ is_valid_digit P ∧ is_valid_digit O ∧ 
  is_valid_digit C ∧ is_valid_digit S ∧ is_valid_digit R ∧
  is_valid_digit T ∧
  K ≠ P ∧ K ≠ O ∧ K ≠ C ∧ K ≠ S ∧ K ≠ R ∧ K ≠ T ∧
  P ≠ O ∧ P ≠ C ∧ P ≠ S ∧ P ≠ R ∧ P ≠ T ∧
  O ≠ C ∧ O ≠ S ∧ O ≠ R ∧ O ≠ T ∧
  C ≠ S ∧ C ≠ R ∧ C ≠ T ∧
  S ≠ R ∧ S ≠ T ∧
  R ≠ T

def satisfies_equation (K P O C S R T : ℕ) : Prop :=
  10000 * K + 1000 * P + 100 * O + 10 * C + C +
  10000 * K + 1000 * P + 100 * O + 10 * C + C =
  10000 * S + 1000 * P + 100 * O + 10 * R + T

theorem cryptarithm_solution :
  ∃! (K P O C S R T : ℕ),
    is_unique_assignment K P O C S R T ∧
    satisfies_equation K P O C S R T ∧
    K = 3 ∧ P = 5 ∧ O = 9 ∧ C = 7 ∧ S = 7 ∧ R = 5 ∧ T = 4 :=
sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l110_11000


namespace NUMINAMATH_CALUDE_miles_collection_height_l110_11047

/-- Represents the height of a book collection in inches and pages -/
structure BookCollection where
  height_inches : ℝ
  total_pages : ℝ

/-- Calculates the total pages in a book collection given the height in inches and pages per inch -/
def total_pages (height : ℝ) (pages_per_inch : ℝ) : ℝ :=
  height * pages_per_inch

theorem miles_collection_height 
  (miles_ratio : ℝ) 
  (daphne_ratio : ℝ) 
  (daphne_height : ℝ) 
  (longest_collection_pages : ℝ)
  (h1 : miles_ratio = 5)
  (h2 : daphne_ratio = 50)
  (h3 : daphne_height = 25)
  (h4 : longest_collection_pages = 1250)
  (h5 : total_pages daphne_height daphne_ratio = longest_collection_pages) :
  ∃ (miles_collection : BookCollection), 
    miles_collection.height_inches = 250 ∧ 
    miles_collection.total_pages = longest_collection_pages :=
sorry

end NUMINAMATH_CALUDE_miles_collection_height_l110_11047


namespace NUMINAMATH_CALUDE_min_binomial_ratio_five_seven_l110_11050

theorem min_binomial_ratio_five_seven (n : ℕ) : n > 0 → (
  (∃ r : ℕ, r < n ∧ (n.choose r : ℚ) / (n.choose (r + 1)) = 5 / 7) ↔ n ≥ 11
) := by sorry

end NUMINAMATH_CALUDE_min_binomial_ratio_five_seven_l110_11050


namespace NUMINAMATH_CALUDE_modulus_of_z_l110_11064

def i : ℂ := Complex.I

theorem modulus_of_z (z : ℂ) (h : z / (1 + i) = 1 - 2*i) : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l110_11064


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l110_11004

theorem average_of_remaining_numbers 
  (n : ℕ) 
  (total_avg : ℚ) 
  (subset_sum : ℚ) 
  (h1 : n = 5) 
  (h2 : total_avg = 20) 
  (h3 : subset_sum = 48) : 
  ((n : ℚ) * total_avg - subset_sum) / ((n : ℚ) - 3) = 26 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l110_11004


namespace NUMINAMATH_CALUDE_f_digit_sum_properties_l110_11090

/-- The function f(n) = 3n^2 + n + 1 -/
def f (n : ℕ+) : ℕ := 3 * n.val^2 + n.val + 1

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the smallest sum of digits and existence of 1999 sum -/
theorem f_digit_sum_properties :
  (∃ (n : ℕ+), sum_of_digits (f n) = 3) ∧ 
  (∀ (n : ℕ+), sum_of_digits (f n) ≥ 3) ∧
  (∃ (n : ℕ+), sum_of_digits (f n) = 1999) :=
sorry

end NUMINAMATH_CALUDE_f_digit_sum_properties_l110_11090


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l110_11074

theorem rectangular_field_perimeter (a b d A : ℝ) : 
  a = 2 * b →                 -- One side is twice as long as the other
  a * b = A →                 -- Area is A
  a^2 + b^2 = d^2 →           -- Pythagorean theorem for diagonal
  A = 240 →                   -- Area is 240 square meters
  d = 34 →                    -- Diagonal is 34 meters
  2 * (a + b) = 91.2 :=       -- Perimeter is 91.2 meters
by sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l110_11074


namespace NUMINAMATH_CALUDE_geometric_mean_exponent_sum_l110_11018

theorem geometric_mean_exponent_sum (a b : ℝ) : 
  a > 0 → b > 0 → (Real.sqrt 3)^2 = 3^a * 3^b → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_exponent_sum_l110_11018


namespace NUMINAMATH_CALUDE_power_mod_23_l110_11082

theorem power_mod_23 : 17^1499 % 23 = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_23_l110_11082


namespace NUMINAMATH_CALUDE_vector_dot_product_equals_three_l110_11094

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
  ab_length : (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1
  bc_length : (C.1 - B.1)^2 + (C.2 - B.2)^2 = 1

-- Define vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)
def vec_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem vector_dot_product_equals_three 
  (A B C M : ℝ × ℝ) 
  (h : Triangle A B C) 
  (hm : vec_sub B M = vec_scale 2 (vec_sub A M)) : 
  dot_product (vec_sub C M) (vec_sub C A) = 3 := by
  sorry


end NUMINAMATH_CALUDE_vector_dot_product_equals_three_l110_11094


namespace NUMINAMATH_CALUDE_tile_arrangement_count_l110_11060

def brown_tiles : ℕ := 1
def purple_tiles : ℕ := 2
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 3

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

def distinguishable_arrangements : ℕ := total_tiles.factorial / (brown_tiles.factorial * purple_tiles.factorial * green_tiles.factorial * yellow_tiles.factorial)

theorem tile_arrangement_count : distinguishable_arrangements = 5040 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangement_count_l110_11060


namespace NUMINAMATH_CALUDE_veronica_brown_balls_l110_11087

/-- Given that Veronica carried 27 yellow balls and 45% of the total balls were yellow,
    prove that she carried 33 brown balls. -/
theorem veronica_brown_balls :
  ∀ (total_balls : ℕ) (yellow_balls : ℕ) (brown_balls : ℕ),
    yellow_balls = 27 →
    (yellow_balls : ℚ) / (total_balls : ℚ) = 45 / 100 →
    total_balls = yellow_balls + brown_balls →
    brown_balls = 33 := by
  sorry

end NUMINAMATH_CALUDE_veronica_brown_balls_l110_11087


namespace NUMINAMATH_CALUDE_max_fraction_over65_l110_11072

/-- Represents the number of people in a room with age-related conditions -/
structure RoomPopulation where
  total : ℕ
  under21 : ℕ
  over65 : ℕ
  h1 : under21 = (3 * total) / 7
  h2 : 50 < total
  h3 : total < 100
  h4 : under21 = 30

/-- The maximum fraction of people over 65 in the room is 4/7 -/
theorem max_fraction_over65 (room : RoomPopulation) :
  (room.over65 : ℚ) / room.total ≤ 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_over65_l110_11072


namespace NUMINAMATH_CALUDE_average_age_combined_rooms_l110_11044

theorem average_age_combined_rooms (room_a_count room_b_count room_c_count : ℕ)
                                   (room_a_avg room_b_avg room_c_avg : ℝ)
                                   (h1 : room_a_count = 8)
                                   (h2 : room_b_count = 5)
                                   (h3 : room_c_count = 7)
                                   (h4 : room_a_avg = 35)
                                   (h5 : room_b_avg = 30)
                                   (h6 : room_c_avg = 50) :
  let total_count := room_a_count + room_b_count + room_c_count
  let total_age := room_a_count * room_a_avg + room_b_count * room_b_avg + room_c_count * room_c_avg
  total_age / total_count = 39 := by
sorry

end NUMINAMATH_CALUDE_average_age_combined_rooms_l110_11044


namespace NUMINAMATH_CALUDE_tammy_mountain_climb_l110_11024

/-- Tammy's mountain climbing problem -/
theorem tammy_mountain_climb 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_diff : ℝ) 
  (time_diff : ℝ) 
  (h_total_time : total_time = 14) 
  (h_total_distance : total_distance = 52) 
  (h_speed_diff : speed_diff = 0.5) 
  (h_time_diff : time_diff = 2) :
  ∃ (v : ℝ), 
    v > 0 ∧ 
    v + speed_diff > 0 ∧ 
    (∃ (t : ℝ), 
      t > 0 ∧ 
      t - time_diff > 0 ∧ 
      t + (t - time_diff) = total_time ∧ 
      v * t + (v + speed_diff) * (t - time_diff) = total_distance) ∧ 
    v + speed_diff = 4 := by
  sorry

end NUMINAMATH_CALUDE_tammy_mountain_climb_l110_11024


namespace NUMINAMATH_CALUDE_unique_prime_with_no_cubic_sum_l110_11020

-- Define the property for a prime p
def has_no_cubic_sum (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℤ, ∀ x y : ℤ, (x^3 + y^3) % p ≠ n % p

-- State the theorem
theorem unique_prime_with_no_cubic_sum :
  ∀ p : ℕ, has_no_cubic_sum p ↔ p = 7 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_with_no_cubic_sum_l110_11020


namespace NUMINAMATH_CALUDE_min_subset_size_l110_11039

def is_valid_subset (s : Finset ℕ) : Prop :=
  s ⊆ Finset.range 11 ∧
  ∀ n : ℕ, n ∈ Finset.range 21 →
    (n ∈ s ∨ ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a + b = n)

theorem min_subset_size :
  ∃ (s : Finset ℕ), is_valid_subset s ∧ s.card = 6 ∧
  ∀ (t : Finset ℕ), is_valid_subset t → t.card ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_subset_size_l110_11039


namespace NUMINAMATH_CALUDE_equation_solutions_l110_11041

def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -2 ∧ -x^2 = (5*x - 2)/(x - 2) - (x + 4)/(x + 2)

theorem equation_solutions :
  {x : ℝ | equation x} = {3, -1, -1 + Real.sqrt 5, -1 - Real.sqrt 5} := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l110_11041


namespace NUMINAMATH_CALUDE_smallest_single_discount_l110_11083

theorem smallest_single_discount (m : ℕ) : m = 29 ↔ 
  (∀ k : ℕ, k < m → 
    ((1 - k / 100 : ℝ) ≥ (1 - 0.20) * (1 - 0.10) ∨
     (1 - k / 100 : ℝ) ≥ (1 - 0.08)^3 ∨
     (1 - k / 100 : ℝ) ≥ (1 - 0.12)^2)) ∧
  ((1 - m / 100 : ℝ) < (1 - 0.20) * (1 - 0.10) ∧
   (1 - m / 100 : ℝ) < (1 - 0.08)^3 ∧
   (1 - m / 100 : ℝ) < (1 - 0.12)^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_single_discount_l110_11083


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l110_11085

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Three terms form a geometric sequence -/
def FormGeometricSequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  FormGeometricSequence (a 3) (a 6) (a 9) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l110_11085


namespace NUMINAMATH_CALUDE_village_population_l110_11096

theorem village_population (P : ℝ) : 
  (P * (1 - 0.1) * (1 - 0.2) = 3312) → P = 4600 := by sorry

end NUMINAMATH_CALUDE_village_population_l110_11096


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l110_11097

theorem no_solution_for_equation :
  ¬ ∃ (p q r : ℕ), 2^p + 5^q = 19^r := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l110_11097


namespace NUMINAMATH_CALUDE_first_four_super_nice_sum_l110_11068

def is_super_nice (n : ℕ) : Prop :=
  n > 1 ∧
  (∃ (divisors : Finset ℕ),
    divisors = {d : ℕ | d ∣ n ∧ d ≠ 1 ∧ d ≠ n} ∧
    n = (Finset.prod divisors id) ∧
    n = (Finset.sum divisors id))

theorem first_four_super_nice_sum :
  ∃ (a b c d : ℕ),
    a < b ∧ b < c ∧ c < d ∧
    is_super_nice a ∧
    is_super_nice b ∧
    is_super_nice c ∧
    is_super_nice d ∧
    a + b + c + d = 45 :=
  sorry

end NUMINAMATH_CALUDE_first_four_super_nice_sum_l110_11068


namespace NUMINAMATH_CALUDE_paving_stones_required_l110_11049

-- Define the dimensions of the courtyard and paving stone
def courtyard_length : ℝ := 158.5
def courtyard_width : ℝ := 35.4
def stone_length : ℝ := 3.2
def stone_width : ℝ := 2.7

-- Define the theorem
theorem paving_stones_required :
  ∃ (n : ℕ), n = 650 ∧ 
  (n : ℝ) * (stone_length * stone_width) ≥ courtyard_length * courtyard_width ∧
  ∀ (m : ℕ), (m : ℝ) * (stone_length * stone_width) ≥ courtyard_length * courtyard_width → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_paving_stones_required_l110_11049


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_theorem_l110_11051

theorem quadratic_function_inequality_theorem :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = 0 → x = -1) ∧
    (∀ x : ℝ, x ≤ a * x^2 + b * x + c) ∧
    (∀ x : ℝ, a * x^2 + b * x + c ≤ (1 + x^2) / 2) ∧
    a = 1/4 ∧ b = 1/2 ∧ c = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_theorem_l110_11051


namespace NUMINAMATH_CALUDE_no_integer_solution_l110_11084

def is_all_twos (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2

theorem no_integer_solution : ¬ ∃ (N : ℤ), is_all_twos (2008 * N.natAbs) :=
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l110_11084


namespace NUMINAMATH_CALUDE_solution_set_when_m_is_2_range_of_m_for_all_real_solution_l110_11089

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-1)*x - m

-- Part I
theorem solution_set_when_m_is_2 :
  ∀ x : ℝ, f 2 x < 0 ↔ -2 < x ∧ x < 1 := by sorry

-- Part II
theorem range_of_m_for_all_real_solution :
  ∀ m : ℝ, (∀ x : ℝ, f m x ≥ -1) ↔ -3 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_m_is_2_range_of_m_for_all_real_solution_l110_11089


namespace NUMINAMATH_CALUDE_total_pies_sold_l110_11079

/-- Represents a type of pie --/
inductive PieType
| Shepherds
| ChickenPot
| VegetablePot
| BeefPot

/-- Represents the size of a pie --/
inductive PieSize
| Small
| Large

/-- Represents the number of pieces a pie is cut into --/
def pieceCount (t : PieType) (s : PieSize) : ℕ :=
  match t, s with
  | PieType.Shepherds, PieSize.Small => 4
  | PieType.Shepherds, PieSize.Large => 8
  | PieType.ChickenPot, PieSize.Small => 5
  | PieType.ChickenPot, PieSize.Large => 10
  | PieType.VegetablePot, PieSize.Small => 6
  | PieType.VegetablePot, PieSize.Large => 12
  | PieType.BeefPot, PieSize.Small => 7
  | PieType.BeefPot, PieSize.Large => 14

/-- Represents the number of customers who ordered each type and size of pie --/
def customerCount (t : PieType) (s : PieSize) : ℕ :=
  match t, s with
  | PieType.Shepherds, PieSize.Small => 52
  | PieType.Shepherds, PieSize.Large => 76
  | PieType.ChickenPot, PieSize.Small => 80
  | PieType.ChickenPot, PieSize.Large => 130
  | PieType.VegetablePot, PieSize.Small => 42
  | PieType.VegetablePot, PieSize.Large => 96
  | PieType.BeefPot, PieSize.Small => 35
  | PieType.BeefPot, PieSize.Large => 105

/-- Calculates the number of pies sold for a given type and size --/
def piesSold (t : PieType) (s : PieSize) : ℕ :=
  (customerCount t s + pieceCount t s - 1) / pieceCount t s

/-- Theorem: The total number of pies sold is 80 --/
theorem total_pies_sold :
  (piesSold PieType.Shepherds PieSize.Small +
   piesSold PieType.Shepherds PieSize.Large +
   piesSold PieType.ChickenPot PieSize.Small +
   piesSold PieType.ChickenPot PieSize.Large +
   piesSold PieType.VegetablePot PieSize.Small +
   piesSold PieType.VegetablePot PieSize.Large +
   piesSold PieType.BeefPot PieSize.Small +
   piesSold PieType.BeefPot PieSize.Large) = 80 :=
by sorry

end NUMINAMATH_CALUDE_total_pies_sold_l110_11079


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_two_increasing_l110_11028

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_increasing_iff_first_two_increasing
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : a 1 > 0) :
  IncreasingSequence a ↔ a 1 < a 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_first_two_increasing_l110_11028


namespace NUMINAMATH_CALUDE_donut_ratio_l110_11080

/-- Given a total of 40 donuts shared among Delta, Beta, and Gamma,
    where Delta takes 8 donuts and Gamma takes 8 donuts,
    prove that the ratio of Beta's donuts to Gamma's donuts is 3:1. -/
theorem donut_ratio :
  ∀ (total delta gamma beta : ℕ),
    total = 40 →
    delta = 8 →
    gamma = 8 →
    beta = total - delta - gamma →
    beta / gamma = 3 := by
  sorry

end NUMINAMATH_CALUDE_donut_ratio_l110_11080


namespace NUMINAMATH_CALUDE_not_all_squares_congruent_l110_11086

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem not_all_squares_congruent :
  ¬ (∀ s1 s2 : Square, congruent s1 s2) :=
sorry

end NUMINAMATH_CALUDE_not_all_squares_congruent_l110_11086


namespace NUMINAMATH_CALUDE_smallest_cube_root_integer_l110_11019

theorem smallest_cube_root_integer (m n : ℕ) (s : ℝ) : 
  (0 < n) →
  (0 < s) →
  (s < 1 / 2000) →
  (m = (n + s)^3) →
  (∀ k < n, ∀ t > 0, t < 1 / 2000 → ¬ (∃ l : ℕ, l = (k + t)^3)) →
  (n = 26) := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_integer_l110_11019


namespace NUMINAMATH_CALUDE_brother_siblings_sibling_product_l110_11071

/-- Represents a family with sisters and brothers -/
structure Family where
  sisters : Nat
  brothers : Nat

/-- Theorem: In a family where one sister has 4 sisters and 6 brothers,
    her brother has 5 sisters and 6 brothers -/
theorem brother_siblings (f : Family) (h : f.sisters = 5 ∧ f.brothers = 7) :
  ∃ (s b : Nat), s = 5 ∧ b = 6 := by
  sorry

/-- Corollary: The product of the number of sisters and brothers
    that the brother has is 30 -/
theorem sibling_product (f : Family) (h : f.sisters = 5 ∧ f.brothers = 7) :
  ∃ (s b : Nat), s * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_brother_siblings_sibling_product_l110_11071


namespace NUMINAMATH_CALUDE_similar_triangles_proportion_l110_11032

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def SimilarTriangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_proportion 
  (P Q R X Y Z : ℝ × ℝ) 
  (h_similar : SimilarTriangles {P, Q, R} {X, Y, Z})
  (h_PQ : dist P Q = 8)
  (h_QR : dist Q R = 16)
  (h_ZY : dist Z Y = 32) :
  dist X Y = 16 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_proportion_l110_11032


namespace NUMINAMATH_CALUDE_sequence_general_term_l110_11013

def sequence_a : ℕ → ℤ
  | 0 => 3
  | 1 => 9
  | (n + 2) => 4 * sequence_a (n + 1) - 3 * sequence_a n - 4 * (n + 2) + 2

theorem sequence_general_term (n : ℕ) : 
  sequence_a n = 3^n + n^2 + 3*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l110_11013


namespace NUMINAMATH_CALUDE_unique_solution_l110_11046

def SatisfiesEquation (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f n + f (f n) + f (f (f n)) = 3 * n

theorem unique_solution :
  ∀ f : ℕ → ℕ, SatisfiesEquation f → (∀ n : ℕ, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l110_11046


namespace NUMINAMATH_CALUDE_candy_distribution_l110_11062

theorem candy_distribution (total_candy : Nat) (num_people : Nat) : 
  total_candy = 30 → num_people = 5 → 
  (∃ (pieces_per_person : Nat), total_candy = pieces_per_person * num_people) → 
  0 = total_candy - (total_candy / num_people) * num_people :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l110_11062


namespace NUMINAMATH_CALUDE_ellipse_ratio_l110_11053

theorem ellipse_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a / b = b / c) : a^2 / b^2 = 2 / (-1 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_ratio_l110_11053


namespace NUMINAMATH_CALUDE_sqrt_representation_condition_l110_11017

theorem sqrt_representation_condition (A B : ℚ) :
  (∃ x y : ℚ, ∀ (sign : Bool), 
    Real.sqrt (A + (-1)^(sign.toNat : ℕ) * Real.sqrt B) = 
    Real.sqrt x + (-1)^(sign.toNat : ℕ) * Real.sqrt y) 
  ↔ 
  ∃ k : ℚ, A^2 - B = k^2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_representation_condition_l110_11017


namespace NUMINAMATH_CALUDE_probability_two_acceptable_cans_l110_11078

theorem probability_two_acceptable_cans (total_cans : Nat) (acceptable_cans : Nat) 
  (h1 : total_cans = 6)
  (h2 : acceptable_cans = 4) : 
  (Nat.choose acceptable_cans 2 : ℚ) / (Nat.choose total_cans 2 : ℚ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_acceptable_cans_l110_11078


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l110_11081

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The factorial of a natural number n, denoted as n!, is the product of all positive integers less than or equal to n. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_sixth : a 6 = factorial 9)
  (h_ninth : a 9 = factorial 10) :
  a 1 = (factorial 9 : ℝ) / (10 ^ (5/3)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l110_11081


namespace NUMINAMATH_CALUDE_smallest_inverse_domain_l110_11014

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2)^2 - 5

-- State the theorem
theorem smallest_inverse_domain (c : ℝ) : 
  (∀ x ≥ c, ∀ y ≥ c, f x = f y → x = y) ∧ 
  (∀ d < c, ∃ x y, d ≤ x ∧ d ≤ y ∧ x ≠ y ∧ f x = f y) ↔ 
  c = -2 :=
sorry

end NUMINAMATH_CALUDE_smallest_inverse_domain_l110_11014


namespace NUMINAMATH_CALUDE_interval_intersection_l110_11055

theorem interval_intersection (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ 1/2 < x ∧ x < 3/5 := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l110_11055


namespace NUMINAMATH_CALUDE_participants_in_both_competitions_l110_11091

theorem participants_in_both_competitions
  (total : ℕ)
  (chinese : ℕ)
  (math : ℕ)
  (neither : ℕ)
  (h1 : total = 50)
  (h2 : chinese = 30)
  (h3 : math = 38)
  (h4 : neither = 2) :
  chinese + math - (total - neither) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_participants_in_both_competitions_l110_11091


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l110_11063

theorem pure_imaginary_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = y * Complex.I) : 
  p / q = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l110_11063


namespace NUMINAMATH_CALUDE_sum_and_product_identities_l110_11022

theorem sum_and_product_identities (a b : ℝ) 
  (sum_eq : a + b = 4) 
  (product_eq : a * b = 1) : 
  a^2 + b^2 = 14 ∧ (a - b)^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_identities_l110_11022


namespace NUMINAMATH_CALUDE_y_finishing_time_l110_11056

/-- The number of days it takes y to finish the remaining work after x has worked for 8 days -/
def days_for_y_to_finish (x_total_days y_total_days x_worked_days : ℕ) : ℕ :=
  (y_total_days * (x_total_days - x_worked_days)) / x_total_days

theorem y_finishing_time 
  (x_total_days : ℕ) 
  (y_total_days : ℕ) 
  (x_worked_days : ℕ) 
  (h1 : x_total_days = 40)
  (h2 : y_total_days = 40)
  (h3 : x_worked_days = 8) :
  days_for_y_to_finish x_total_days y_total_days x_worked_days = 32 := by
sorry

#eval days_for_y_to_finish 40 40 8

end NUMINAMATH_CALUDE_y_finishing_time_l110_11056


namespace NUMINAMATH_CALUDE_negation_of_universal_negation_of_proposition_l110_11023

theorem negation_of_universal (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) := by sorry

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 2*x - 1 > 0) ↔ (∃ x : ℝ, 2*x - 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_negation_of_proposition_l110_11023


namespace NUMINAMATH_CALUDE_triangle_inequality_l110_11034

theorem triangle_inequality (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) → 
  (¬(a + b > c ∧ b + c > a ∧ c + a > b) ↔ a + b ≤ c ∨ b + c ≤ a ∨ c + a ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l110_11034


namespace NUMINAMATH_CALUDE_direct_proportion_percentage_change_l110_11030

theorem direct_proportion_percentage_change 
  (x y : ℝ) (q : ℝ) (c : ℝ) (hx : x > 0) (hy : y > 0) (hq : q > 0) (hc : c > 0) 
  (h_prop : y = c * x) :
  let x' := x * (1 - q / 100)
  let y' := c * x'
  (y' - y) / y * 100 = q := by
sorry

end NUMINAMATH_CALUDE_direct_proportion_percentage_change_l110_11030


namespace NUMINAMATH_CALUDE_cauchy_schwarz_and_max_value_l110_11005

theorem cauchy_schwarz_and_max_value :
  (∀ a b c d : ℝ, (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2) ∧
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a + b = 1 → (Real.sqrt (3*a + 1) + Real.sqrt (3*b + 1))^2 ≤ 10) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_and_max_value_l110_11005


namespace NUMINAMATH_CALUDE_initial_bushes_count_l110_11099

/-- The number of orchid bushes to be planted today -/
def bushes_to_plant : ℕ := 4

/-- The final number of orchid bushes after planting -/
def final_bushes : ℕ := 6

/-- The initial number of orchid bushes in the park -/
def initial_bushes : ℕ := final_bushes - bushes_to_plant

theorem initial_bushes_count : initial_bushes = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_bushes_count_l110_11099


namespace NUMINAMATH_CALUDE_cory_chairs_proof_l110_11067

/-- The number of chairs Cory bought -/
def num_chairs : ℕ := 4

/-- The cost of the patio table -/
def table_cost : ℕ := 55

/-- The cost of each chair -/
def chair_cost : ℕ := 20

/-- The total cost of the table and chairs -/
def total_cost : ℕ := 135

theorem cory_chairs_proof :
  num_chairs * chair_cost + table_cost = total_cost :=
by sorry

end NUMINAMATH_CALUDE_cory_chairs_proof_l110_11067


namespace NUMINAMATH_CALUDE_red_peaches_count_l110_11093

/-- The number of red peaches in a basket with yellow, green, and red peaches. -/
def num_red_peaches (yellow green red_and_green : ℕ) : ℕ :=
  red_and_green - green

/-- Theorem stating that the number of red peaches is 6. -/
theorem red_peaches_count :
  let yellow : ℕ := 90
  let green : ℕ := 16
  let red_and_green : ℕ := 22
  num_red_peaches yellow green red_and_green = 6 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l110_11093


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l110_11002

theorem at_least_one_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b > 1) :
  a > 1 ∨ b > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l110_11002


namespace NUMINAMATH_CALUDE_find_m_l110_11054

-- Define the universal set U
def U : Set Nat := {1, 2, 3}

-- Define set A
def A (m : Nat) : Set Nat := {1, m}

-- Define the complement of A in U
def complementA : Set Nat := {2}

-- Theorem to prove
theorem find_m : ∃ m : Nat, m ∈ U ∧ A m ∪ complementA = U := by
  sorry

end NUMINAMATH_CALUDE_find_m_l110_11054


namespace NUMINAMATH_CALUDE_cube_surface_area_l110_11029

/-- The surface area of a cube with side length 20 cm is 2400 square centimeters. -/
theorem cube_surface_area : 
  let side_length : ℝ := 20
  6 * side_length ^ 2 = 2400 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l110_11029


namespace NUMINAMATH_CALUDE_lunch_packet_cost_l110_11033

/-- Represents the field trip scenario -/
structure FieldTrip where
  total_students : Nat
  lunch_buyers : Nat
  packet_cost : Nat
  apples_per_packet : Nat
  total_cost : Nat

/-- The field trip satisfies the given conditions -/
def valid_field_trip (ft : FieldTrip) : Prop :=
  ft.total_students = 50 ∧
  ft.lunch_buyers > ft.total_students / 2 ∧
  ft.apples_per_packet < ft.packet_cost ∧
  ft.lunch_buyers * ft.packet_cost = ft.total_cost ∧
  ft.total_cost = 3087

theorem lunch_packet_cost (ft : FieldTrip) :
  valid_field_trip ft → ft.packet_cost = 9 := by
  sorry

#check lunch_packet_cost

end NUMINAMATH_CALUDE_lunch_packet_cost_l110_11033


namespace NUMINAMATH_CALUDE_consecutive_product_problem_l110_11016

theorem consecutive_product_problem :
  let n : ℕ := 77
  let product := n * (n + 1) * (n + 2)
  (product ≥ 100000 ∧ product < 1000000) ∧  -- six-digit number
  (product / 10000 = 47) ∧                  -- left-hand digits are '47'
  (product % 100 = 74)                      -- right-hand digits are '74'
  :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_problem_l110_11016


namespace NUMINAMATH_CALUDE_mischievous_quadratic_min_root_product_l110_11061

/-- A quadratic polynomial with real coefficients and leading coefficient 1 -/
def QuadraticPolynomial (r s : ℝ) (x : ℝ) : ℝ := x^2 - (r + s) * x + r * s

/-- A polynomial is mischievous if p(p(x)) = 0 has exactly four real roots -/
def IsMischievous (p : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x, p (p x) = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)

/-- The theorem stating that the mischievous quadratic polynomial with minimized root product evaluates to 1 at x = 1 -/
theorem mischievous_quadratic_min_root_product (r s : ℝ) :
  IsMischievous (QuadraticPolynomial r s) →
  (∀ r' s' : ℝ, IsMischievous (QuadraticPolynomial r' s') → r * s ≤ r' * s') →
  QuadraticPolynomial r s 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_mischievous_quadratic_min_root_product_l110_11061


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l110_11048

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {1, 9}

theorem complement_of_A_in_U : 
  (U \ A) = {3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l110_11048


namespace NUMINAMATH_CALUDE_set_relations_l110_11077

def A (k : ℝ) : Set ℝ := {x | k * x^2 - 2 * x + 6 * k < 0 ∧ k ≠ 0}

theorem set_relations (k : ℝ) :
  (A k ⊆ Set.Ioo 2 3 → k ≥ 2/5) ∧
  (Set.Ioo 2 3 ⊆ A k → k ≤ 2/5) ∧
  (Set.inter (A k) (Set.Ioo 2 3) ≠ ∅ → k < Real.sqrt 6 / 6) :=
sorry

end NUMINAMATH_CALUDE_set_relations_l110_11077


namespace NUMINAMATH_CALUDE_power_of_64_l110_11058

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 := by sorry

end NUMINAMATH_CALUDE_power_of_64_l110_11058


namespace NUMINAMATH_CALUDE_task_completion_probability_l110_11073

theorem task_completion_probability (p1 p2 : ℚ) 
  (h1 : p1 = 2/3) 
  (h2 : p2 = 3/5) : 
  p1 * (1 - p2) = 4/15 := by
sorry

end NUMINAMATH_CALUDE_task_completion_probability_l110_11073


namespace NUMINAMATH_CALUDE_diplomats_not_speaking_russian_l110_11037

theorem diplomats_not_speaking_russian (total : ℕ) (french : ℕ) (both_percent : ℚ) (neither_percent : ℚ) 
  (h_total : total = 70)
  (h_french : french = 25)
  (h_both : both_percent = 1/10)
  (h_neither : neither_percent = 1/5) : 
  total - (total : ℚ) * (1 - neither_percent) + french - total * both_percent = 39 := by
  sorry

end NUMINAMATH_CALUDE_diplomats_not_speaking_russian_l110_11037


namespace NUMINAMATH_CALUDE_unique_root_of_equation_l110_11035

theorem unique_root_of_equation :
  ∃! x : ℝ, (3 : ℝ)^x + (5 : ℝ)^x + (11 : ℝ)^x = (19 : ℝ)^x * Real.sqrt (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_root_of_equation_l110_11035


namespace NUMINAMATH_CALUDE_smallest_value_for_x_between_0_and_1_l110_11098

theorem smallest_value_for_x_between_0_and_1 (x : ℝ) (h : 0 < x ∧ x < 1) :
  x^2 < x ∧ x^2 < 2*x ∧ x^2 < Real.sqrt x ∧ x^2 < 1/x :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_between_0_and_1_l110_11098


namespace NUMINAMATH_CALUDE_stella_profit_l110_11065

def dolls : ℕ := 3
def clocks : ℕ := 2
def glasses : ℕ := 5

def doll_price : ℕ := 5
def clock_price : ℕ := 15
def glass_price : ℕ := 4

def total_cost : ℕ := 40

def total_sales : ℕ := dolls * doll_price + clocks * clock_price + glasses * glass_price

def profit : ℕ := total_sales - total_cost

theorem stella_profit : profit = 25 := by
  sorry

end NUMINAMATH_CALUDE_stella_profit_l110_11065


namespace NUMINAMATH_CALUDE_silver_cost_per_ounce_l110_11059

theorem silver_cost_per_ounce
  (silver_amount : ℝ)
  (gold_amount : ℝ)
  (gold_silver_price_ratio : ℝ)
  (total_spent : ℝ)
  (h1 : silver_amount = 1.5)
  (h2 : gold_amount = 2 * silver_amount)
  (h3 : gold_silver_price_ratio = 50)
  (h4 : total_spent = 3030)
  (h5 : silver_amount * silver_cost + gold_amount * (gold_silver_price_ratio * silver_cost) = total_spent) :
  silver_cost = 20 :=
by sorry

end NUMINAMATH_CALUDE_silver_cost_per_ounce_l110_11059


namespace NUMINAMATH_CALUDE_frank_five_dollar_bills_l110_11011

def peanut_cost_per_pound : ℕ := 3
def days_in_week : ℕ := 7
def pounds_per_day : ℕ := 3
def one_dollar_bills : ℕ := 7
def ten_dollar_bills : ℕ := 2
def twenty_dollar_bills : ℕ := 1
def change_amount : ℕ := 4

def total_without_fives : ℕ := one_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills

def total_pounds_needed : ℕ := days_in_week * pounds_per_day

theorem frank_five_dollar_bills :
  ∃ (five_dollar_bills : ℕ),
    total_without_fives + 5 * five_dollar_bills - change_amount = peanut_cost_per_pound * total_pounds_needed ∧
    five_dollar_bills = 4 := by
  sorry

end NUMINAMATH_CALUDE_frank_five_dollar_bills_l110_11011


namespace NUMINAMATH_CALUDE_pipe_crate_height_difference_l110_11012

/-- The height difference between two crates of cylindrical pipes -/
theorem pipe_crate_height_difference (pipe_diameter : ℝ) (crate_a_rows : ℕ) (crate_b_rows : ℕ) :
  pipe_diameter = 20 →
  crate_a_rows = 10 →
  crate_b_rows = 9 →
  let crate_a_height := crate_a_rows * pipe_diameter
  let crate_b_height := crate_b_rows * pipe_diameter + (crate_b_rows - 1) * pipe_diameter * Real.sqrt 3
  crate_a_height - crate_b_height = 20 - 160 * Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_pipe_crate_height_difference_l110_11012


namespace NUMINAMATH_CALUDE_gold_cube_buying_price_l110_11057

/-- Proves that the buying price of gold is $60 per gram given the specified conditions -/
theorem gold_cube_buying_price (cube_side : ℝ) (gold_density : ℝ) (selling_factor : ℝ) (profit : ℝ) :
  cube_side = 6 →
  gold_density = 19 →
  selling_factor = 1.5 →
  profit = 123120 →
  let volume := cube_side ^ 3
  let mass := gold_density * volume
  let buying_price := profit / (selling_factor * mass - mass)
  buying_price = 60 := by sorry

end NUMINAMATH_CALUDE_gold_cube_buying_price_l110_11057


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l110_11009

/-- The number of ways to arrange n distinct objects into k distinct positions --/
def arrangements (n k : ℕ) : ℕ := (k.factorial) / ((k - n).factorial)

/-- The number of seating arrangements for three people in a row of eight chairs
    with an empty seat on either side of each person --/
def seatingArrangements : ℕ :=
  let totalChairs : ℕ := 8
  let peopleToSeat : ℕ := 3
  let availablePositions : ℕ := totalChairs - 2 - (peopleToSeat - 1)
  arrangements peopleToSeat availablePositions

theorem seating_arrangements_count :
  seatingArrangements = 24 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l110_11009


namespace NUMINAMATH_CALUDE_arrangements_five_not_adjacent_l110_11003

/-- The number of permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange n distinct objects in a line, 
    where two specific objects are not adjacent -/
def arrangements_not_adjacent (n : ℕ) : ℕ :=
  factorial n - 2 * factorial (n - 1)

theorem arrangements_five_not_adjacent :
  arrangements_not_adjacent 5 = 72 := by
  sorry

#eval arrangements_not_adjacent 5

end NUMINAMATH_CALUDE_arrangements_five_not_adjacent_l110_11003
