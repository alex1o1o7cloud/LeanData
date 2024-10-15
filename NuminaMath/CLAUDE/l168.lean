import Mathlib

namespace NUMINAMATH_CALUDE_books_read_per_year_l168_16897

/-- The total number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  84 * c * s

/-- Theorem: The total number of books read by the entire student body in one year
    is equal to 84 * c * s, given the conditions of the reading program -/
theorem books_read_per_year (c s : ℕ) (books_per_month : ℕ) (months_per_year : ℕ)
    (h1 : books_per_month = 7)
    (h2 : months_per_year = 12)
    (h3 : c > 0)
    (h4 : s > 0) :
    total_books_read c s = books_per_month * months_per_year * c * s :=
  sorry

end NUMINAMATH_CALUDE_books_read_per_year_l168_16897


namespace NUMINAMATH_CALUDE_minimum_correct_answers_l168_16896

theorem minimum_correct_answers (total_questions : ℕ) 
  (correct_points : ℕ) (incorrect_points : ℤ) (min_score : ℕ) :
  total_questions = 20 →
  correct_points = 10 →
  incorrect_points = -5 →
  min_score = 120 →
  (∃ x : ℕ, x * correct_points + (total_questions - x) * incorrect_points > min_score ∧
    ∀ y : ℕ, y < x → y * correct_points + (total_questions - y) * incorrect_points ≤ min_score) →
  (∃ x : ℕ, x * correct_points + (total_questions - x) * incorrect_points > min_score ∧
    ∀ y : ℕ, y < x → y * correct_points + (total_questions - y) * incorrect_points ≤ min_score) →
  x = 15 :=
by sorry

end NUMINAMATH_CALUDE_minimum_correct_answers_l168_16896


namespace NUMINAMATH_CALUDE_smarties_remainder_l168_16852

theorem smarties_remainder (n : ℕ) (h : n % 11 = 8) : (2 * n) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_smarties_remainder_l168_16852


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l168_16869

theorem complex_number_quadrant (z : ℂ) (h : z * (2 - Complex.I) = 2 + Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l168_16869


namespace NUMINAMATH_CALUDE_certain_number_problem_l168_16832

theorem certain_number_problem :
  ∃ x : ℝ, x ≥ 0 ∧ 5 * (Real.sqrt x + 3) = 19 ∧ x = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l168_16832


namespace NUMINAMATH_CALUDE_logo_shaded_area_l168_16858

/-- The shaded area of a logo design with a square containing four larger circles and one smaller circle -/
theorem logo_shaded_area (square_side : ℝ) (large_circle_radius : ℝ) (small_circle_radius : ℝ) : 
  square_side = 24 →
  large_circle_radius = 6 →
  small_circle_radius = 3 →
  (square_side ^ 2) - (4 * Real.pi * large_circle_radius ^ 2) - (Real.pi * small_circle_radius ^ 2) = 576 - 153 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_logo_shaded_area_l168_16858


namespace NUMINAMATH_CALUDE_shelby_total_stars_l168_16814

/-- The number of gold stars Shelby earned yesterday -/
def yesterday_stars : ℕ := 4

/-- The number of gold stars Shelby earned today -/
def today_stars : ℕ := 3

/-- The total number of gold stars Shelby earned -/
def total_stars : ℕ := yesterday_stars + today_stars

theorem shelby_total_stars : total_stars = 7 := by
  sorry

end NUMINAMATH_CALUDE_shelby_total_stars_l168_16814


namespace NUMINAMATH_CALUDE_sample_volume_calculation_l168_16864

theorem sample_volume_calculation (m : ℝ) 
  (h1 : m > 0)  -- Ensure m is positive
  (h2 : 8 / m + 0.15 + 0.45 = 1) : m = 20 := by
  sorry

end NUMINAMATH_CALUDE_sample_volume_calculation_l168_16864


namespace NUMINAMATH_CALUDE_calculation_proof_l168_16828

theorem calculation_proof : 
  let sin_30 : ℝ := 1/2
  let sqrt_2_gt_1 : 1 < Real.sqrt 2 := by sorry
  let power_zero (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by sorry
  2 * sin_30 - |1 - Real.sqrt 2| + (π - 2022)^0 = 3 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_calculation_proof_l168_16828


namespace NUMINAMATH_CALUDE_career_preference_graph_degrees_l168_16804

theorem career_preference_graph_degrees 
  (total_students : ℕ) 
  (male_ratio female_ratio : ℚ) 
  (male_preference female_preference : ℚ) :
  male_ratio / (male_ratio + female_ratio) = 2 / 5 →
  female_ratio / (male_ratio + female_ratio) = 3 / 5 →
  male_preference = 1 / 4 →
  female_preference = 1 / 2 →
  (male_ratio * male_preference + female_ratio * female_preference) / (male_ratio + female_ratio) * 360 = 144 := by
  sorry

#check career_preference_graph_degrees

end NUMINAMATH_CALUDE_career_preference_graph_degrees_l168_16804


namespace NUMINAMATH_CALUDE_radical_axes_property_l168_16800

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line in 2D space (ax + by + c = 0)
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the radical axis of two circles
def radical_axis (c1 c2 : Circle) : Line :=
  sorry

-- Define the property of lines being coincident
def coincident (l1 l2 l3 : Line) : Prop :=
  sorry

-- Define the property of lines being concurrent
def concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

-- Define the property of lines being parallel
def parallel (l1 l2 l3 : Line) : Prop :=
  sorry

-- Theorem statement
theorem radical_axes_property (Γ₁ Γ₂ Γ₃ : Circle) :
  let Δ₁ := radical_axis Γ₁ Γ₂
  let Δ₂ := radical_axis Γ₂ Γ₃
  let Δ₃ := radical_axis Γ₃ Γ₁
  coincident Δ₁ Δ₂ Δ₃ ∨ concurrent Δ₁ Δ₂ Δ₃ ∨ parallel Δ₁ Δ₂ Δ₃ :=
by
  sorry

end NUMINAMATH_CALUDE_radical_axes_property_l168_16800


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l168_16833

/-- The coefficients of the quadratic equation in general form -/
def quadratic_coefficients (a b c : ℝ) : ℝ × ℝ := (a, b)

/-- The original quadratic equation -/
def original_equation (x : ℝ) : Prop := 3 * x^2 + 1 = 6 * x

/-- The general form of the quadratic equation -/
def general_form (x : ℝ) : Prop := 3 * x^2 - 6 * x + 1 = 0

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), (∀ x, original_equation x ↔ general_form x) ∧
  quadratic_coefficients a b c = (3, -6) := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l168_16833


namespace NUMINAMATH_CALUDE_sin_cos_15_product_eq_neg_sqrt3_div_2_l168_16871

theorem sin_cos_15_product_eq_neg_sqrt3_div_2 :
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) *
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_product_eq_neg_sqrt3_div_2_l168_16871


namespace NUMINAMATH_CALUDE_suitcase_electronics_weight_l168_16856

/-- Given a suitcase with books, clothes, and electronics, prove the weight of electronics -/
theorem suitcase_electronics_weight 
  (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive
  (h2 : 4 * x > 7) -- Ensure we can remove 7 pounds of clothing
  (h3 : 5 * x / (4 * x - 7) = 5 / 2) -- Ratio doubles after removing 7 pounds
  : 2 * x = 7 := by
  sorry

end NUMINAMATH_CALUDE_suitcase_electronics_weight_l168_16856


namespace NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l168_16848

/-- The function g(x) = 5x - 7 -/
def g (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: g(7/5) = 0 -/
theorem g_zero_at_seven_fifths : g (7/5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_at_seven_fifths_l168_16848


namespace NUMINAMATH_CALUDE_quadratic_transformation_l168_16836

-- Define the coefficients of the quadratic equation
variable (a b c : ℝ)

-- Define the condition that ax^2 + bx + c can be expressed as 3(x - 5)^2 + 7
def quadratic_condition (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 3 * (x - 5)^2 + 7

-- Define the expanded form of 4ax^2 + 4bx + 4c
def expanded_quadratic (x : ℝ) : ℝ :=
  4 * a * x^2 + 4 * b * x + 4 * c

-- Theorem statement
theorem quadratic_transformation (h : ∀ x, quadratic_condition a b c x) :
  ∃ (n k : ℝ), ∀ x, expanded_quadratic a b c x = n * (x - 5)^2 + k :=
sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l168_16836


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_l168_16812

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 0 → m < n → n % m ≠ 0

/-- A one-digit number is a natural number less than 10. -/
def isOneDigit (n : ℕ) : Prop := n < 10

/-- A two-digit number is a natural number greater than or equal to 10 and less than 100. -/
def isTwoDigit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- The two smallest one-digit primes are 2 and 3. -/
axiom smallest_one_digit_primes : ∀ n : ℕ, isPrime n → isOneDigit n → n = 2 ∨ n = 3

/-- The smallest two-digit prime is 11. -/
axiom smallest_two_digit_prime : ∀ n : ℕ, isPrime n → isTwoDigit n → n ≥ 11

theorem product_of_smallest_primes : 
  ∃ p q r : ℕ, 
    isPrime p ∧ isOneDigit p ∧
    isPrime q ∧ isOneDigit q ∧
    isPrime r ∧ isTwoDigit r ∧
    p * q * r = 66 :=
sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_l168_16812


namespace NUMINAMATH_CALUDE_complex_coordinate_to_z_l168_16870

theorem complex_coordinate_to_z (z : ℂ) :
  (z / Complex.I).re = 3 ∧ (z / Complex.I).im = -1 → z = 1 + 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_coordinate_to_z_l168_16870


namespace NUMINAMATH_CALUDE_first_hour_rate_is_25_l168_16827

/-- Represents the rental cost structure for a power tool -/
structure RentalCost where
  firstHourRate : ℕ
  additionalHourRate : ℕ

/-- Represents the rental details for Ashwin -/
structure RentalDetails where
  totalCost : ℕ
  totalHours : ℕ

/-- Theorem stating that given the rental conditions, the first hour rate was $25 -/
theorem first_hour_rate_is_25 (rental : RentalCost) (details : RentalDetails) :
  rental.additionalHourRate = 10 ∧
  details.totalCost = 125 ∧
  details.totalHours = 11 →
  rental.firstHourRate = 25 := by
  sorry

#check first_hour_rate_is_25

end NUMINAMATH_CALUDE_first_hour_rate_is_25_l168_16827


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l168_16819

theorem sum_remainder_mod_seven (n : ℤ) : ((7 + n) + (n + 5)) % 7 = (5 + 2*n) % 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l168_16819


namespace NUMINAMATH_CALUDE_three_diamonds_balance_two_circles_l168_16834

/-- Represents the balance of symbols in the problem -/
structure Balance where
  triangle : ℕ  -- Δ
  diamond : ℕ   -- ◊
  circle : ℕ    -- •

/-- First balance equation: 4Δ + 2◊ = 12• -/
def balance_equation1 (b : Balance) : Prop :=
  4 * b.triangle + 2 * b.diamond = 12 * b.circle

/-- Second balance equation: Δ = ◊ + 2• -/
def balance_equation2 (b : Balance) : Prop :=
  b.triangle = b.diamond + 2 * b.circle

/-- Theorem stating that 3◊ balances 2• -/
theorem three_diamonds_balance_two_circles (b : Balance) 
  (h1 : balance_equation1 b) (h2 : balance_equation2 b) : 
  3 * b.diamond = 2 * b.circle :=
sorry

end NUMINAMATH_CALUDE_three_diamonds_balance_two_circles_l168_16834


namespace NUMINAMATH_CALUDE_transformation_correctness_l168_16853

theorem transformation_correctness (a b : ℝ) (h : a > b) : 1 + 2*a > 1 + 2*b := by
  sorry

end NUMINAMATH_CALUDE_transformation_correctness_l168_16853


namespace NUMINAMATH_CALUDE_array_sum_proof_l168_16805

def grid := [[1, 0, 0, 0], [0, 9, 0, 5], [0, 0, 14, 0]]
def available_numbers := [2, 3, 4, 7, 10, 11, 12, 13, 15]

theorem array_sum_proof :
  ∃ (arrangement : List (List Nat)),
    (∀ row ∈ arrangement, row.sum = 32) ∧
    (∀ col ∈ arrangement.transpose, col.sum = 32) ∧
    (arrangement.join.toFinset = (available_numbers.toFinset \ {10}) ∪ grid.join.toFinset) :=
  by sorry

end NUMINAMATH_CALUDE_array_sum_proof_l168_16805


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l168_16885

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l168_16885


namespace NUMINAMATH_CALUDE_exponential_linear_inequalities_l168_16809

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x

/-- A linear function with slope k -/
def g (k : ℝ) (x : ℝ) : ℝ := k * x + 1

theorem exponential_linear_inequalities (k : ℝ) :
  (∃ (y : ℝ), ∀ (x : ℝ), f x - (x + 1) ≥ y ∧ ∃ (x : ℝ), f x - (x + 1) = y) ∧
  (k > 1 → ∃ (x₀ : ℝ), x₀ > 0 ∧ ∀ (x : ℝ), 0 < x ∧ x < x₀ → f x < g k x) ∧
  (∃ (m : ℝ), m > 0 ∧ ∀ (x : ℝ), 0 < x ∧ x < m → |f x - g k x| > x) ↔ (k ≤ 0 ∨ k > 2) := by
  sorry

end NUMINAMATH_CALUDE_exponential_linear_inequalities_l168_16809


namespace NUMINAMATH_CALUDE_maria_number_transformation_l168_16829

theorem maria_number_transformation (x : ℚ) : 
  (2 * (x + 3) - 2) / 3 = 8 → x = 10 := by sorry

end NUMINAMATH_CALUDE_maria_number_transformation_l168_16829


namespace NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l168_16841

/-- Represents a club with members having different characteristics -/
structure Club where
  total : Nat
  leftHanded : Nat
  jazzLovers : Nat
  rightHandedNonJazz : Nat

/-- The number of left-handed jazz lovers in the club -/
def leftHandedJazzLovers (c : Club) : Nat :=
  c.leftHanded + c.jazzLovers - (c.total - c.rightHandedNonJazz)

/-- Theorem stating the number of left-handed jazz lovers in the specific club -/
theorem left_handed_jazz_lovers_count (c : Club) 
  (h1 : c.total = 30)
  (h2 : c.leftHanded = 12)
  (h3 : c.jazzLovers = 20)
  (h4 : c.rightHandedNonJazz = 3) :
  leftHandedJazzLovers c = 5 := by
  sorry

#check left_handed_jazz_lovers_count

end NUMINAMATH_CALUDE_left_handed_jazz_lovers_count_l168_16841


namespace NUMINAMATH_CALUDE_belinda_passed_twenty_percent_l168_16886

-- Define the total number of flyers
def total_flyers : ℕ := 200

-- Define the number of flyers passed out by each person
def ryan_flyers : ℕ := 42
def alyssa_flyers : ℕ := 67
def scott_flyers : ℕ := 51

-- Define Belinda's flyers as the remaining flyers
def belinda_flyers : ℕ := total_flyers - (ryan_flyers + alyssa_flyers + scott_flyers)

-- Define the percentage of flyers Belinda passed out
def belinda_percentage : ℚ := (belinda_flyers : ℚ) / (total_flyers : ℚ) * 100

-- Theorem stating that Belinda passed out 20% of the flyers
theorem belinda_passed_twenty_percent : belinda_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_belinda_passed_twenty_percent_l168_16886


namespace NUMINAMATH_CALUDE_y_plus_z_squared_positive_l168_16891

theorem y_plus_z_squared_positive 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 2 < z ∧ z < 3) : 
  0 < y + z^2 := by
  sorry

end NUMINAMATH_CALUDE_y_plus_z_squared_positive_l168_16891


namespace NUMINAMATH_CALUDE_total_team_score_l168_16802

def team_score (team_size : ℕ) (faye_score : ℕ) (other_player_score : ℕ) : ℕ :=
  faye_score + (team_size - 1) * other_player_score

theorem total_team_score :
  team_score 5 28 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_team_score_l168_16802


namespace NUMINAMATH_CALUDE_math_competition_nonparticipants_l168_16808

theorem math_competition_nonparticipants (total_students : ℕ) 
  (h1 : total_students = 39) 
  (h2 : ∃ participants : ℕ, participants = total_students / 3) : 
  ∃ nonparticipants : ℕ, nonparticipants = 26 ∧ nonparticipants = total_students - (total_students / 3) :=
by sorry

end NUMINAMATH_CALUDE_math_competition_nonparticipants_l168_16808


namespace NUMINAMATH_CALUDE_multiply_72518_by_9999_l168_16826

theorem multiply_72518_by_9999 : 72518 * 9999 = 725107482 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72518_by_9999_l168_16826


namespace NUMINAMATH_CALUDE_no_solution_exists_l168_16830

theorem no_solution_exists : ¬∃ (a b c d : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧ 
  (a/b + b/c + c/d + d/a = 6) ∧ 
  (b/a + c/b + d/c + a/d = 32) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l168_16830


namespace NUMINAMATH_CALUDE_hamburger_combinations_count_l168_16817

/-- The number of condiment choices available. -/
def num_condiments : ℕ := 9

/-- The number of options for meat patties. -/
def patty_options : ℕ := 3

/-- Calculates the number of different hamburger combinations. -/
def hamburger_combinations : ℕ := patty_options * 2^num_condiments

/-- Theorem stating that the number of different hamburger combinations is 1536. -/
theorem hamburger_combinations_count : hamburger_combinations = 1536 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_count_l168_16817


namespace NUMINAMATH_CALUDE_x_equals_one_l168_16835

theorem x_equals_one (x y : ℝ) (h1 : x + 3 * y = 10) (h2 : y = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_one_l168_16835


namespace NUMINAMATH_CALUDE_unique_function_existence_l168_16815

theorem unique_function_existence : 
  ∃! f : ℕ → ℕ, f 1 = 1 ∧ ∀ n : ℕ, f n * f (n + 2) = f (n + 1) ^ 2 + 1997 := by
  sorry

end NUMINAMATH_CALUDE_unique_function_existence_l168_16815


namespace NUMINAMATH_CALUDE_car_a_speed_car_a_speed_is_58_l168_16865

/-- Proves that Car A's speed is 58 mph given the problem conditions -/
theorem car_a_speed : ℝ → Prop :=
  fun (speed_a : ℝ) =>
    let initial_gap : ℝ := 10
    let overtake_distance : ℝ := 8
    let speed_b : ℝ := 50
    let time : ℝ := 2.25
    let distance_b : ℝ := speed_b * time
    let distance_a : ℝ := distance_b + initial_gap + overtake_distance
    speed_a = distance_a / time ∧ speed_a = 58

/-- The theorem is true -/
theorem car_a_speed_is_58 : ∃ (speed_a : ℝ), car_a_speed speed_a :=
sorry

end NUMINAMATH_CALUDE_car_a_speed_car_a_speed_is_58_l168_16865


namespace NUMINAMATH_CALUDE_vegetable_pieces_count_l168_16880

/-- Calculates the total number of vegetable pieces after cutting -/
def total_vegetable_pieces (bell_peppers onions zucchinis : ℕ) : ℕ :=
  let bell_pepper_thin := (bell_peppers / 4) * 20
  let bell_pepper_large := (bell_peppers - bell_peppers / 4) * 10
  let bell_pepper_small := (bell_pepper_large / 2) * 3
  let onion_thin := (onions / 2) * 18
  let onion_chunk := (onions - onions / 2) * 8
  let zucchini_thin := (zucchinis * 3 / 10) * 15
  let zucchini_chunk := (zucchinis - zucchinis * 3 / 10) * 8
  bell_pepper_thin + bell_pepper_large + bell_pepper_small + onion_thin + onion_chunk + zucchini_thin + zucchini_chunk

/-- Theorem stating that given the conditions, the total number of vegetable pieces is 441 -/
theorem vegetable_pieces_count : total_vegetable_pieces 10 7 15 = 441 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_pieces_count_l168_16880


namespace NUMINAMATH_CALUDE_decimal_places_of_fraction_l168_16816

theorem decimal_places_of_fraction : ∃ (n : ℕ), 
  (5^5 : ℚ) / (10^3 * 8) = n / 10 ∧ n % 10 ≠ 0 ∧ n < 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_places_of_fraction_l168_16816


namespace NUMINAMATH_CALUDE_difference_divisible_by_nine_l168_16899

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem difference_divisible_by_nine (N : ℕ) :
  ∃ k : ℤ, N - (sum_of_digits N) = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_divisible_by_nine_l168_16899


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l168_16875

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + (2*m - 3)*x + (m^2 - 3) = 0) ↔ m ≤ 7/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l168_16875


namespace NUMINAMATH_CALUDE_equation_rewrite_l168_16851

theorem equation_rewrite (x y : ℝ) : 5 * x + 3 * y = 1 ↔ y = (1 - 5 * x) / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_rewrite_l168_16851


namespace NUMINAMATH_CALUDE_solution_set_inequality_l168_16839

theorem solution_set_inequality (x : ℝ) : (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l168_16839


namespace NUMINAMATH_CALUDE_evaluate_expression_l168_16890

theorem evaluate_expression : 3^5 * 6^5 * 3^6 * 6^6 = 18^11 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l168_16890


namespace NUMINAMATH_CALUDE_line_equation_l168_16857

-- Define a line by its slope and y-intercept
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

-- Function to translate a line
def translate_line (l : Line) (dx : ℝ) (dy : ℝ) : Line :=
  { slope := l.slope,
    y_intercept := l.y_intercept - l.slope * dx + dy }

-- Theorem statement
theorem line_equation (l : Line) :
  point_on_line { x := 1, y := 1 } l ∧
  translate_line (translate_line l 2 0) 0 (-1) = l →
  l.slope = 1/2 ∧ l.y_intercept = 1/2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l168_16857


namespace NUMINAMATH_CALUDE_square_division_square_coverage_l168_16824

/-- A square is a shape with four equal sides and four right angles -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- A larger square can be divided into four equal smaller squares -/
theorem square_division (large : Square) : 
  ∃ (small : Square), 
    4 * small.side^2 = large.side^2 ∧ 
    small.side > 0 :=
sorry

/-- Four smaller squares can completely cover a larger square without gaps or overlaps -/
theorem square_coverage (large : Square) 
  (h : ∃ (small : Square), 4 * small.side^2 = large.side^2 ∧ small.side > 0) : 
  ∃ (small : Square), 
    4 * small.side^2 = large.side^2 ∧ 
    small.side > 0 ∧
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ large.side ∧ 0 ≤ y ∧ y ≤ large.side → 
      ∃ (i j : Fin 2), 
        i * small.side ≤ x ∧ x < (i + 1) * small.side ∧
        j * small.side ≤ y ∧ y < (j + 1) * small.side) :=
sorry

end NUMINAMATH_CALUDE_square_division_square_coverage_l168_16824


namespace NUMINAMATH_CALUDE_olivias_wallet_problem_l168_16843

/-- The initial amount of money in Olivia's wallet -/
def initial_money : ℕ := 100

/-- The amount of money Olivia collected from the ATM -/
def atm_money : ℕ := 148

/-- The amount of money Olivia spent at the supermarket -/
def spent_money : ℕ := 89

/-- The amount of money left after visiting the supermarket -/
def remaining_money : ℕ := 159

theorem olivias_wallet_problem :
  initial_money + atm_money = remaining_money + spent_money :=
by sorry

end NUMINAMATH_CALUDE_olivias_wallet_problem_l168_16843


namespace NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l168_16825

/-- Given a triangle ABC with centroid G, if GA^2 + GB^2 + GC^2 = 88, 
    then AB^2 + AC^2 + BC^2 = 396 -/
theorem triangle_centroid_distance_sum (A B C G : ℝ × ℝ) : 
  (G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) →
  ((G.1 - A.1)^2 + (G.2 - A.2)^2 + 
   (G.1 - B.1)^2 + (G.2 - B.2)^2 + 
   (G.1 - C.1)^2 + (G.2 - C.2)^2 = 88) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 + 
   (A.1 - C.1)^2 + (A.2 - C.2)^2 + 
   (B.1 - C.1)^2 + (B.2 - C.2)^2 = 396) := by
sorry

end NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l168_16825


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l168_16882

theorem gcd_lcm_product (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l168_16882


namespace NUMINAMATH_CALUDE_min_max_values_l168_16883

theorem min_max_values (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) 
  (h : x^2 + 4*y^2 + 4*x*y + 4*x^2*y^2 = 32) : 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a^2 + 4*b^2 + 4*a*b + 4*a^2*b^2 = 32 → x + 2*y ≤ a + 2*b) ∧ 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a^2 + 4*b^2 + 4*a*b + 4*a^2*b^2 = 32 → 
    Real.sqrt 7 * (x + 2*y) + 2*x*y ≥ Real.sqrt 7 * (a + 2*b) + 2*a*b) ∧
  x + 2*y = 4 ∧ 
  Real.sqrt 7 * (x + 2*y) + 2*x*y = 4 * Real.sqrt 7 + 4 := by
  sorry

end NUMINAMATH_CALUDE_min_max_values_l168_16883


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l168_16894

theorem polynomial_division_remainder : ∃ q r : Polynomial ℤ,
  (3 * X^4 + 14 * X^3 - 35 * X^2 - 80 * X + 56) = 
  (X^2 + 8 * X - 6) * q + r ∧ 
  r.degree < 2 ∧ 
  r = 364 * X - 322 :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l168_16894


namespace NUMINAMATH_CALUDE_triathlete_average_speed_l168_16898

/-- Calculates the average speed for a triathlete's swimming and running events,
    assuming equal distances for both activities. -/
theorem triathlete_average_speed
  (swim_speed : ℝ)
  (run_speed : ℝ)
  (h1 : swim_speed = 1)
  (h2 : run_speed = 7) :
  (2 * swim_speed * run_speed) / (swim_speed + run_speed) = 1.75 := by
  sorry

#check triathlete_average_speed

end NUMINAMATH_CALUDE_triathlete_average_speed_l168_16898


namespace NUMINAMATH_CALUDE_desmond_toy_purchase_l168_16868

/-- The number of toys Mr. Desmond bought for his elder son -/
def elder_son_toys : ℕ := 60

/-- The number of toys Mr. Desmond bought for his younger son -/
def younger_son_toys : ℕ := 3 * elder_son_toys

/-- The total number of toys Mr. Desmond bought -/
def total_toys : ℕ := elder_son_toys + younger_son_toys

theorem desmond_toy_purchase :
  total_toys = 240 := by sorry

end NUMINAMATH_CALUDE_desmond_toy_purchase_l168_16868


namespace NUMINAMATH_CALUDE_thirteen_binary_l168_16866

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Checks if a list of booleans represents a given natural number in binary -/
def is_binary_rep (n : ℕ) (bits : List Bool) : Prop :=
  to_binary n = bits

theorem thirteen_binary :
  is_binary_rep 13 [true, false, true, true] := by sorry

end NUMINAMATH_CALUDE_thirteen_binary_l168_16866


namespace NUMINAMATH_CALUDE_no_infinite_sequence_exists_l168_16884

theorem no_infinite_sequence_exists : 
  ¬ ∃ (a : ℕ → ℕ), ∀ (n : ℕ), 
    a (n + 2) = a (n + 1) + Real.sqrt (a (n + 1) + a n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_exists_l168_16884


namespace NUMINAMATH_CALUDE_hyperbola_focus_parameter_l168_16837

/-- Given a hyperbola with equation y²/m - x²/9 = 1 and a focus at (0, 5),
    prove that m = 16. -/
theorem hyperbola_focus_parameter (m : ℝ) : 
  (∀ x y : ℝ, y^2/m - x^2/9 = 1 → (x = 0 ∧ y = 5) → m = 16) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_parameter_l168_16837


namespace NUMINAMATH_CALUDE_distance_between_complex_points_l168_16845

theorem distance_between_complex_points :
  let z₁ : ℂ := 2 + 3*I
  let z₂ : ℂ := -2 + 2*I
  Complex.abs (z₁ - z₂) = Real.sqrt 17 := by
sorry

end NUMINAMATH_CALUDE_distance_between_complex_points_l168_16845


namespace NUMINAMATH_CALUDE_helen_cookies_proof_l168_16846

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 31

/-- The number of cookies Helen baked this morning -/
def cookies_this_morning : ℕ := 270

/-- The total number of cookies Helen baked till last night -/
def total_cookies : ℕ := 450

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_before_yesterday : ℕ := total_cookies - (cookies_yesterday + cookies_this_morning)

theorem helen_cookies_proof : 
  cookies_before_yesterday = 149 := by sorry

end NUMINAMATH_CALUDE_helen_cookies_proof_l168_16846


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l168_16818

/-- Represents the number of students selected from each year in a stratified sample. -/
structure StratifiedSample where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the stratified sample given the total number of students and sample size. -/
def calculate_stratified_sample (total_students : ℕ) (first_year : ℕ) (second_year : ℕ) (third_year : ℕ) (sample_size : ℕ) : StratifiedSample :=
  { first_year := (first_year * sample_size) / total_students,
    second_year := (second_year * sample_size) / total_students,
    third_year := (third_year * sample_size) / total_students }

theorem stratified_sample_theorem :
  let total_students : ℕ := 900
  let first_year : ℕ := 300
  let second_year : ℕ := 200
  let third_year : ℕ := 400
  let sample_size : ℕ := 45
  let result := calculate_stratified_sample total_students first_year second_year third_year sample_size
  result.first_year = 15 ∧ result.second_year = 10 ∧ result.third_year = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l168_16818


namespace NUMINAMATH_CALUDE_postcard_area_l168_16878

/-- Represents a rectangular postcard -/
structure Postcard where
  vertical_length : ℝ
  horizontal_length : ℝ

/-- Calculates the area of a postcard -/
def area (p : Postcard) : ℝ := p.vertical_length * p.horizontal_length

/-- Calculates the perimeter of two attached postcards -/
def attached_perimeter (p : Postcard) : ℝ := 2 * p.vertical_length + 4 * p.horizontal_length

theorem postcard_area (p : Postcard) 
  (h1 : p.vertical_length = 15)
  (h2 : attached_perimeter p = 70) : 
  area p = 150 := by
  sorry

#check postcard_area

end NUMINAMATH_CALUDE_postcard_area_l168_16878


namespace NUMINAMATH_CALUDE_press_conference_seating_l168_16860

/-- Represents the number of ways to seat players from different teams -/
def seating_arrangements (cubs : Nat) (red_sox : Nat) : Nat :=
  2 * 2 * (Nat.factorial cubs) * (Nat.factorial red_sox)

/-- Theorem stating the number of seating arrangements for the given conditions -/
theorem press_conference_seating :
  seating_arrangements 4 3 = 576 :=
by sorry

end NUMINAMATH_CALUDE_press_conference_seating_l168_16860


namespace NUMINAMATH_CALUDE_coefficient_of_x_l168_16881

theorem coefficient_of_x (some_number : ℝ) : 
  (2 * (1/2)^2 + some_number * (1/2) - 5 = 0) → some_number = 9 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_l168_16881


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l168_16855

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_unit_cubes : Nat
  painted_grid_size : Nat

/-- Calculates the number of unpainted unit cubes in the painted cube -/
def unpainted_cubes (cube : PaintedCube) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem unpainted_cubes_in_6x6x6 :
  let cube : PaintedCube := {
    size := 6,
    total_unit_cubes := 216,
    painted_grid_size := 4
  }
  unpainted_cubes cube = 176 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_6x6x6_l168_16855


namespace NUMINAMATH_CALUDE_equation_solution_l168_16850

theorem equation_solution : ∃ x : ℝ, (x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) ∧ 
                            (2 * x - 5 ≠ 0) ∧ (5 - 2 * x ≠ 0) ∧ (x = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l168_16850


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l168_16872

theorem min_value_of_sum_of_ratios (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : (a + c) * (b + d) = a * c + b * d) : 
  a / b + b / c + c / d + d / a ≥ 8 ∧ 
  ∃ (a' b' c' d' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0 ∧
    (a' + c') * (b' + d') = a' * c' + b' * d' ∧
    a' / b' + b' / c' + c' / d' + d' / a' = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l168_16872


namespace NUMINAMATH_CALUDE_simplify_fraction_l168_16838

theorem simplify_fraction (a : ℚ) (h : a = -2) : 18 * a^5 / (27 * a^3) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l168_16838


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l168_16847

/-- Represents a node in the hexagonal grid --/
structure Node :=
  (x : ℤ)
  (y : ℤ)
  (z : ℤ)
  (sum_zero : x + y + z = 0)

/-- Represents the hexagonal grid game --/
structure HexagonGame :=
  (n : ℕ)
  (current_player : ℕ)
  (token : Node)
  (visited : Set Node)

/-- Defines a valid move in the game --/
def valid_move (game : HexagonGame) (new_pos : Node) : Prop :=
  (abs (new_pos.x - game.token.x) + abs (new_pos.y - game.token.y) + abs (new_pos.z - game.token.z) = 2) ∧
  (new_pos ∉ game.visited)

/-- Defines the winning condition for the second player --/
def second_player_wins (n : ℕ) : Prop :=
  ∀ (game : HexagonGame),
    game.n = n →
    (game.current_player = 1 → ∃ (move : Node), valid_move game move) →
    (game.current_player = 2 → ∀ (move : Node), valid_move game move → 
      ∃ (counter_move : Node), valid_move (HexagonGame.mk n 1 move (game.visited.insert game.token)) counter_move)

/-- The main theorem: The second player has a winning strategy for all n --/
theorem second_player_winning_strategy :
  ∀ n : ℕ, second_player_wins n :=
sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l168_16847


namespace NUMINAMATH_CALUDE_f_properties_l168_16895

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_periodic : ∀ x : ℝ, f (x + 6) = f x + f 3

-- Theorem to prove
theorem f_properties :
  (f 3 = 0) ∧
  (f (-3) = 0) ∧
  (∀ x : ℝ, f (6 + x) = f (6 - x)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l168_16895


namespace NUMINAMATH_CALUDE_half_coverage_days_l168_16887

/-- Represents the number of days it takes for the lily pad patch to cover the entire lake -/
def full_coverage_days : ℕ := 58

/-- Represents the growth factor of the lily pad patch per day -/
def daily_growth_factor : ℕ := 2

/-- Theorem stating that the number of days to cover half the lake is one less than the number of days to cover the full lake -/
theorem half_coverage_days : 
  ∃ (half_days : ℕ), half_days = full_coverage_days - 1 ∧ 
  (daily_growth_factor ^ half_days) * 2 = daily_growth_factor ^ full_coverage_days :=
sorry

end NUMINAMATH_CALUDE_half_coverage_days_l168_16887


namespace NUMINAMATH_CALUDE_otimes_inequality_implies_a_range_l168_16813

-- Define the custom operation ⊗
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- Define the theorem
theorem otimes_inequality_implies_a_range :
  (∀ x ∈ Set.Icc 1 2, otimes (x - a) (x + a) < 2) →
  -1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_otimes_inequality_implies_a_range_l168_16813


namespace NUMINAMATH_CALUDE_abs_x_squared_lt_x_solution_set_l168_16873

theorem abs_x_squared_lt_x_solution_set :
  {x : ℝ | |x| * |x| < x} = {x : ℝ | (0 < x ∧ x < 1) ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_abs_x_squared_lt_x_solution_set_l168_16873


namespace NUMINAMATH_CALUDE_count_equal_pairs_l168_16867

/-- Definition of the sequence a_n -/
def a (n : ℕ) : ℤ := n^2 - 22*n + 10

/-- The number of pairs of distinct positive integers (m,n) satisfying a_m = a_n -/
def num_pairs : ℕ := 10

/-- Theorem stating that there are exactly 10 pairs of distinct positive integers (m,n) 
    satisfying a_m = a_n -/
theorem count_equal_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    pairs.card = num_pairs ∧ 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ 
      m ≠ n ∧ m > 0 ∧ n > 0 ∧ a m = a n) :=
sorry

end NUMINAMATH_CALUDE_count_equal_pairs_l168_16867


namespace NUMINAMATH_CALUDE_range_of_m_l168_16859

/-- Given conditions p and q, prove that m ∈ [4, +∞) -/
theorem range_of_m (x m : ℝ) : 
  (∀ x, x^2 - 3*x - 4 ≤ 0 → |x - 3| ≤ m) ∧ 
  (∃ x, |x - 3| ≤ m ∧ x^2 - 3*x - 4 > 0) →
  m ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l168_16859


namespace NUMINAMATH_CALUDE_special_square_area_l168_16823

/-- A square with one side on a line and two vertices on a parabola -/
structure SpecialSquare where
  /-- The y-coordinate of vertex C -/
  y1 : ℝ
  /-- The y-coordinate of vertex D -/
  y2 : ℝ
  /-- C and D lie on the parabola y^2 = x -/
  h1 : y1^2 = (y1 : ℝ)
  h2 : y2^2 = (y2 : ℝ)
  /-- Side AB lies on the line y = x + 4 -/
  h3 : y2^2 - y1^2 + y1 = y1^2 + y1 - y2 + 4
  /-- The slope condition -/
  h4 : y1 - y2 = y1^2 - y2^2

/-- The area of a SpecialSquare is either 18 or 50 -/
theorem special_square_area (s : SpecialSquare) : (s.y2 - s.y1)^2 = 18 ∨ (s.y2 - s.y1)^2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_special_square_area_l168_16823


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l168_16844

theorem polynomial_division_theorem (x : ℝ) :
  ∃ (q r : ℝ), 5*x^3 - 4*x^2 + 6*x - 9 = (x - 1) * (5*x^2 + x + 7) + r ∧ r = -2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l168_16844


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l168_16889

theorem mod_equivalence_unique_solution :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ -5033 [ZMOD 12] := by sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l168_16889


namespace NUMINAMATH_CALUDE_closest_point_on_line_l168_16876

/-- The point on the line y = 3x - 1 that is closest to (1,4) is (-3/5, -4/5) -/
theorem closest_point_on_line (x y : ℝ) : 
  y = 3 * x - 1 → 
  (x - (-3/5))^2 + (y - (-4/5))^2 ≤ (x - 1)^2 + (y - 4)^2 :=
by sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l168_16876


namespace NUMINAMATH_CALUDE_john_popcorn_profit_l168_16854

/-- Calculates the profit for John's popcorn business --/
theorem john_popcorn_profit :
  let regular_price : ℚ := 4
  let discount_rate : ℚ := 0.1
  let adult_price : ℚ := 8
  let child_price : ℚ := 6
  let packaging_cost : ℚ := 0.5
  let transport_fee : ℚ := 10
  let adult_bags : ℕ := 20
  let child_bags : ℕ := 10
  let total_bags : ℕ := adult_bags + child_bags
  let discounted_price : ℚ := regular_price * (1 - discount_rate)
  let total_cost : ℚ := discounted_price * total_bags + packaging_cost * total_bags + transport_fee
  let total_revenue : ℚ := adult_price * adult_bags + child_price * child_bags
  let profit : ℚ := total_revenue - total_cost
  profit = 87 := by
    sorry

end NUMINAMATH_CALUDE_john_popcorn_profit_l168_16854


namespace NUMINAMATH_CALUDE_min_value_theorem_l168_16874

theorem min_value_theorem (x : ℝ) (h : x > 2) :
  (x + 6) / Real.sqrt (x - 2) ≥ 4 * Real.sqrt 2 ∧
  ((x + 6) / Real.sqrt (x - 2) = 4 * Real.sqrt 2 ↔ x = 10) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l168_16874


namespace NUMINAMATH_CALUDE_hyperbola_center_l168_16806

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 1001 = 0

/-- The center of a hyperbola -/
def is_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ), 
    eq x y ↔ ((x - c.1)^2 / a^2) - ((y - c.2)^2 / b^2) = 1

/-- Theorem: The center of the given hyperbola is (3, 5) -/
theorem hyperbola_center : is_center (3, 5) hyperbola_equation :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l168_16806


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_on_zero_one_l168_16831

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem f_monotone_decreasing_on_zero_one :
  ∀ x ∈ Set.Ioo (0 : ℝ) 1, StrictMonoOn f (Set.Ioo (0 : ℝ) 1) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_on_zero_one_l168_16831


namespace NUMINAMATH_CALUDE_prob_jack_and_jill_selected_l168_16879

/-- The probability of Jack being selected for the interview. -/
def prob_jack : ℝ := 0.20

/-- The probability of Jill being selected for the interview. -/
def prob_jill : ℝ := 0.15

/-- The number of workers in the hospital. -/
def num_workers : ℕ := 8

/-- The number of workers to be interviewed. -/
def num_interviewed : ℕ := 2

/-- Assumption that the selection of Jack and Jill are independent events. -/
axiom selection_independent : True

theorem prob_jack_and_jill_selected : 
  prob_jack * prob_jill = 0.03 := by sorry

end NUMINAMATH_CALUDE_prob_jack_and_jill_selected_l168_16879


namespace NUMINAMATH_CALUDE_lukes_trivia_score_l168_16820

/-- Luke's trivia game score calculation -/
theorem lukes_trivia_score (rounds : ℕ) (points_per_round : ℕ) (h1 : rounds = 177) (h2 : points_per_round = 46) :
  rounds * points_per_round = 8142 := by
  sorry

end NUMINAMATH_CALUDE_lukes_trivia_score_l168_16820


namespace NUMINAMATH_CALUDE_max_value_of_expression_l168_16849

theorem max_value_of_expression (x y : ℝ) (h1 : x ≥ 1) (h2 : y ≥ 1) (h3 : x + y = 8) :
  (∀ a b : ℝ, a ≥ 1 → b ≥ 1 → a + b = 8 → 
    |Real.sqrt (x - 1/y) + Real.sqrt (y - 1/x)| ≥ |Real.sqrt (a - 1/b) + Real.sqrt (b - 1/a)|) ∧
  |Real.sqrt (x - 1/y) + Real.sqrt (y - 1/x)| ≤ Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l168_16849


namespace NUMINAMATH_CALUDE_percentage_gain_calculation_l168_16803

/-- Calculates the percentage gain when selling an article --/
theorem percentage_gain_calculation (cost_price selling_price : ℚ) : 
  cost_price = 160 → 
  selling_price = 192 → 
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_gain_calculation_l168_16803


namespace NUMINAMATH_CALUDE_ab_equals_e_cubed_l168_16888

theorem ab_equals_e_cubed (a b : ℝ) (h1 : Real.exp (2 - a) = a) (h2 : b * (Real.log b - 1) = Real.exp 3) : a * b = Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_e_cubed_l168_16888


namespace NUMINAMATH_CALUDE_polynomial_value_l168_16810

theorem polynomial_value (a b : ℝ) (h : a^2 - 2*b - 1 = 0) :
  -2*a^2 + 4*b + 2025 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l168_16810


namespace NUMINAMATH_CALUDE_geometric_sum_n1_l168_16821

theorem geometric_sum_n1 (a : ℝ) (h : a ≠ 1) :
  1 + a + a^2 = (1 - a^3) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_n1_l168_16821


namespace NUMINAMATH_CALUDE_field_length_proof_l168_16811

theorem field_length_proof (width : ℝ) (length : ℝ) (pond_side : ℝ) :
  length = 2 * width →
  pond_side = 9 →
  pond_side^2 = (1/8) * (length * width) →
  length = 36 := by
sorry

end NUMINAMATH_CALUDE_field_length_proof_l168_16811


namespace NUMINAMATH_CALUDE_fraction_inequality_l168_16822

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  c / a < c / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l168_16822


namespace NUMINAMATH_CALUDE_square_roots_problem_l168_16862

theorem square_roots_problem (m : ℝ) (n : ℝ) (h1 : n > 0) (h2 : 2*m - 1 = (n ^ (1/2 : ℝ))) (h3 : 2 - m = (n ^ (1/2 : ℝ))) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l168_16862


namespace NUMINAMATH_CALUDE_chord_length_l168_16892

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def C₃ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 14/5 = 0

-- Define the common chord of C₁ and C₂
def common_chord (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- Theorem statement
theorem chord_length :
  ∃ (a b : ℝ), 
    (∀ x y, C₁ x y ∧ C₂ x y → common_chord x y) ∧
    (∃ x₁ y₁ x₂ y₂, 
      C₃ x₁ y₁ ∧ C₃ x₂ y₂ ∧ 
      common_chord x₁ y₁ ∧ common_chord x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4^2) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l168_16892


namespace NUMINAMATH_CALUDE_radio_cost_price_l168_16801

/-- Calculates the cost price of an item given its selling price and loss percentage. -/
def cost_price (selling_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  selling_price / (1 - loss_percentage / 100)

/-- Proves that the cost price of a radio sold for 1305 with a 13% loss is 1500. -/
theorem radio_cost_price : cost_price 1305 13 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_radio_cost_price_l168_16801


namespace NUMINAMATH_CALUDE_negation_of_forall_leq_zero_l168_16877

theorem negation_of_forall_leq_zero :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_leq_zero_l168_16877


namespace NUMINAMATH_CALUDE_can_achieve_any_coloring_can_achieve_checkerboard_l168_16861

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the color of a square -/
inductive Color
  | White
  | Black

/-- Represents the state of the chessboard -/
def Board := Square → Color

/-- Represents a move that changes the color of squares in a row and column -/
structure Move where
  row : Fin 8
  col : Fin 8

/-- Applies a move to a board, changing colors in the specified row and column -/
def applyMove (b : Board) (m : Move) : Board :=
  fun s => if s.row = m.row || s.col = m.col then
             match b s with
             | Color.White => Color.Black
             | Color.Black => Color.White
           else b s

/-- The initial all-white board -/
def initialBoard : Board := fun _ => Color.White

/-- The standard checkerboard pattern -/
def checkerboardPattern : Board :=
  fun s => if (s.row.val + s.col.val) % 2 = 0 then Color.White else Color.Black

/-- Theorem stating that any desired board coloring can be achieved -/
theorem can_achieve_any_coloring :
  ∀ (targetBoard : Board), ∃ (moves : List Move),
    (moves.foldl applyMove initialBoard) = targetBoard :=
  sorry

/-- Corollary stating that the standard checkerboard pattern can be achieved -/
theorem can_achieve_checkerboard :
  ∃ (moves : List Move),
    (moves.foldl applyMove initialBoard) = checkerboardPattern :=
  sorry

end NUMINAMATH_CALUDE_can_achieve_any_coloring_can_achieve_checkerboard_l168_16861


namespace NUMINAMATH_CALUDE_last_digit_of_large_prime_l168_16893

theorem last_digit_of_large_prime (h : 859433 = 214858 * 4 + 1) :
  (2^859433 - 1) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_large_prime_l168_16893


namespace NUMINAMATH_CALUDE_friends_seinfeld_relationship_l168_16842

-- Define the variables
variable (x y z : ℚ)

-- Define the conditions
def friends_episodes : ℚ := 50
def seinfeld_episodes : ℚ := 75

-- State the theorem
theorem friends_seinfeld_relationship 
  (h1 : x * z = friends_episodes) 
  (h2 : y * z = seinfeld_episodes) :
  y = 1.5 * x := by
  sorry

end NUMINAMATH_CALUDE_friends_seinfeld_relationship_l168_16842


namespace NUMINAMATH_CALUDE_money_distribution_l168_16863

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (AC_sum : A + C = 200)
  (C_amount : C = 10) :
  B + C = 310 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l168_16863


namespace NUMINAMATH_CALUDE_largest_three_digit_base7_decimal_l168_16840

/-- The largest three-digit number in base 7 -/
def largest_base7 : ℕ := 666

/-- Conversion from base 7 to decimal -/
def base7_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- Theorem stating that the largest decimal number represented by a three-digit base-7 number is 342 -/
theorem largest_three_digit_base7_decimal :
  base7_to_decimal largest_base7 = 342 := by sorry

end NUMINAMATH_CALUDE_largest_three_digit_base7_decimal_l168_16840


namespace NUMINAMATH_CALUDE_equation_solution_l168_16807

theorem equation_solution : 
  ∃ (x : ℝ), ((3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1)) ∧ 
  (x = 6 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l168_16807
