import Mathlib

namespace NUMINAMATH_CALUDE_certain_value_proof_l4036_403658

theorem certain_value_proof (n : ℤ) (v : ℤ) : 
  (∀ m : ℤ, 101 * m^2 ≤ v → m ≤ 8) → 
  (101 * 8^2 ≤ v) →
  v = 6464 := by
sorry

end NUMINAMATH_CALUDE_certain_value_proof_l4036_403658


namespace NUMINAMATH_CALUDE_soccer_cards_l4036_403661

theorem soccer_cards (total_players : ℕ) (no_caution_players : ℕ) (yellow_to_red : ℕ) : 
  total_players = 11 →
  no_caution_players = 5 →
  yellow_to_red = 2 →
  (total_players - no_caution_players) / yellow_to_red = 3 := by
  sorry

end NUMINAMATH_CALUDE_soccer_cards_l4036_403661


namespace NUMINAMATH_CALUDE_geometric_sequence_theorem_l4036_403695

/-- A geometric sequence with given second and fifth terms -/
structure GeometricSequence where
  b₂ : ℝ
  b₅ : ℝ
  h₁ : b₂ = 24.5
  h₂ : b₅ = 196

/-- Properties of the geometric sequence -/
def GeometricSequence.properties (g : GeometricSequence) : Prop :=
  ∃ (b₁ r : ℝ),
    r > 0 ∧
    g.b₂ = b₁ * r ∧
    g.b₅ = b₁ * r^4 ∧
    let b₃ := b₁ * r^2
    let S₄ := b₁ * (r^4 - 1) / (r - 1)
    b₃ = 49 ∧ S₄ = 183.75

/-- Main theorem: The third term is 49 and the sum of first four terms is 183.75 -/
theorem geometric_sequence_theorem (g : GeometricSequence) :
  g.properties := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_theorem_l4036_403695


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4036_403629

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 6 * p - 9 = 0) → 
  (3 * q^3 - 2 * q^2 + 6 * q - 9 = 0) → 
  (3 * r^3 - 2 * r^2 + 6 * r - 9 = 0) → 
  p^2 + q^2 + r^2 = -32/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l4036_403629


namespace NUMINAMATH_CALUDE_base_8_to_10_367_l4036_403624

-- Define the base-8 number as a list of digits
def base_8_number : List Nat := [3, 6, 7]

-- Define the function to convert base-8 to base-10
def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem base_8_to_10_367 :
  base_8_to_10 base_8_number = 247 := by sorry

end NUMINAMATH_CALUDE_base_8_to_10_367_l4036_403624


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_three_l4036_403688

theorem smallest_four_digit_divisible_by_three :
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ 
           n % 3 = 0 ∧
           (∀ m : ℕ, (1000 ≤ m ∧ m < n) → m % 3 ≠ 0) ∧
           n = 1002 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_three_l4036_403688


namespace NUMINAMATH_CALUDE_non_square_difference_characterization_l4036_403650

/-- A natural number that cannot be represented as the difference of squares of any two natural numbers. -/
def NonSquareDifference (n : ℕ) : Prop :=
  ∀ x y : ℕ, n ≠ x^2 - y^2

/-- Characterization of numbers that cannot be represented as the difference of squares. -/
theorem non_square_difference_characterization (n : ℕ) :
  NonSquareDifference n ↔ n = 1 ∨ n = 4 ∨ ∃ k : ℕ, n = 4*k + 2 :=
sorry

end NUMINAMATH_CALUDE_non_square_difference_characterization_l4036_403650


namespace NUMINAMATH_CALUDE_video_game_expenditure_l4036_403668

theorem video_game_expenditure (total : ℝ) (books snacks stationery shoes : ℝ) :
  total = 50 →
  books = (1 / 4) * total →
  snacks = (1 / 5) * total →
  stationery = (1 / 10) * total →
  shoes = (3 / 10) * total →
  total - (books + snacks + stationery + shoes) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_video_game_expenditure_l4036_403668


namespace NUMINAMATH_CALUDE_log_base_32_integer_count_l4036_403647

theorem log_base_32_integer_count : 
  ∃! (n : ℕ), n > 0 ∧ (∃ (S : Finset ℕ), 
    (∀ b ∈ S, b > 0 ∧ ∃ (k : ℕ), k > 0 ∧ (↑b : ℝ) ^ k = 32) ∧
    S.card = n) :=
by sorry

end NUMINAMATH_CALUDE_log_base_32_integer_count_l4036_403647


namespace NUMINAMATH_CALUDE_exists_point_sum_distances_gt_perimeter_l4036_403600

/-- A convex n-gon in a 2D plane -/
structure ConvexNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_convex : sorry -- Axiom for convexity

/-- The perimeter of a convex n-gon -/
def perimeter (polygon : ConvexNGon n) : ℝ := sorry

/-- The sum of distances from a point to all vertices of a convex n-gon -/
def sum_distances (polygon : ConvexNGon n) (point : ℝ × ℝ) : ℝ := sorry

/-- For any convex n-gon with n ≥ 7, there exists a point inside the n-gon
    such that the sum of distances from this point to all vertices
    is greater than the perimeter of the n-gon -/
theorem exists_point_sum_distances_gt_perimeter (n : ℕ) (h : n ≥ 7) (polygon : ConvexNGon n) :
  ∃ (point : ℝ × ℝ), sum_distances polygon point > perimeter polygon := by
  sorry

end NUMINAMATH_CALUDE_exists_point_sum_distances_gt_perimeter_l4036_403600


namespace NUMINAMATH_CALUDE_rower_downstream_speed_l4036_403667

/-- Calculates the downstream speed of a rower given their upstream and still water speeds. -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that given a man's upstream speed of 25 kmph and still water speed of 30 kmph,
    his downstream speed is 35 kmph. -/
theorem rower_downstream_speed :
  downstream_speed 25 30 = 35 := by
  sorry

end NUMINAMATH_CALUDE_rower_downstream_speed_l4036_403667


namespace NUMINAMATH_CALUDE_min_river_width_for_race_l4036_403643

/-- The width of a river that can accommodate a boat race -/
def river_width (num_boats : ℕ) (boat_width : ℕ) (space_between : ℕ) : ℕ :=
  num_boats * boat_width + (num_boats - 1) * space_between + 2 * space_between

/-- Theorem stating the minimum width of the river for the given conditions -/
theorem min_river_width_for_race : river_width 8 3 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_river_width_for_race_l4036_403643


namespace NUMINAMATH_CALUDE_quadratic_even_function_sum_l4036_403623

/-- A quadratic function of the form f(x) = x^2 + (a-1)x + a + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + (a - 1) * x + a + b

/-- f is an even function -/
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem quadratic_even_function_sum (a b : ℝ) :
  is_even_function (f a b) → f a b 2 = 0 → a + b = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_even_function_sum_l4036_403623


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4036_403697

theorem expression_simplification_and_evaluation (x : ℝ) (h : x ≠ 2) :
  ((x + 2) / (x - 2) + (x - x^2) / (x^2 - 4*x + 4)) / ((x - 4) / (x - 2)) = 1 / (x - 2) ∧
  (((3 + 2) / (3 - 2) + (3 - 3^2) / (3^2 - 4*3 + 4)) / ((3 - 4) / (3 - 2)) = 1) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l4036_403697


namespace NUMINAMATH_CALUDE_divisibility_by_17_l4036_403634

theorem divisibility_by_17 (a b : ℤ) : 
  let x : ℤ := 3 * b - 5 * a
  let y : ℤ := 9 * a - 2 * b
  (17 ∣ (2 * x + 3 * y)) ∧ (17 ∣ (9 * x + 5 * y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_17_l4036_403634


namespace NUMINAMATH_CALUDE_problem_solution_l4036_403601

theorem problem_solution (a b : ℝ) (h1 : a - b = 4) (h2 : a * b = 1) :
  (2 * a + 3 * b - 2 * a * b) - (a + 4 * b + a * b) - (3 * a * b + 2 * b - 2 * a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4036_403601


namespace NUMINAMATH_CALUDE_regular_soda_bottles_l4036_403611

/-- Given a grocery store with diet and regular soda bottles, prove the number of regular soda bottles. -/
theorem regular_soda_bottles (diet_soda : ℕ) (difference : ℕ) (regular_soda : ℕ) : 
  diet_soda = 9 → difference = 58 → regular_soda = diet_soda + difference → regular_soda = 67 := by
  sorry

end NUMINAMATH_CALUDE_regular_soda_bottles_l4036_403611


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l4036_403636

theorem triangle_angle_inequality (a b c α β γ : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : α > 0 ∧ β > 0 ∧ γ > 0)
  (h3 : α + β + γ = π)
  (h4 : a + b > c ∧ b + c > a ∧ c + a > b) : 
  π / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l4036_403636


namespace NUMINAMATH_CALUDE_a_n_properties_smallest_n_perfect_square_sum_l4036_403687

/-- The largest n-digit number that is neither the sum nor the difference of two perfect squares -/
def a_n (n : ℕ) : ℕ := 10^n - 2

/-- The sum of squares of digits of a number -/
def sum_of_squares_of_digits (m : ℕ) : ℕ := sorry

/-- Theorem stating the properties of a_n -/
theorem a_n_properties :
  ∀ (n : ℕ), n > 2 →
  (∀ (x y : ℕ), a_n n ≠ x^2 + y^2 ∧ a_n n ≠ x^2 - y^2) ∧
  (∀ (m : ℕ), m < n → ∃ (x y : ℕ), 10^m - 2 = x^2 + y^2 ∨ 10^m - 2 = x^2 - y^2) :=
sorry

/-- Theorem stating the smallest n for which the sum of squares of digits of a_n is a perfect square -/
theorem smallest_n_perfect_square_sum :
  ∃ (k : ℕ), sum_of_squares_of_digits (a_n 66) = k^2 ∧
  ∀ (n : ℕ), n < 66 → ¬∃ (k : ℕ), sum_of_squares_of_digits (a_n n) = k^2 :=
sorry

end NUMINAMATH_CALUDE_a_n_properties_smallest_n_perfect_square_sum_l4036_403687


namespace NUMINAMATH_CALUDE_max_time_digit_sum_l4036_403644

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours ≤ 23
  minute_valid : minutes ≤ 59

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits in a 24-hour digital watch display is 24 -/
theorem max_time_digit_sum : 
  ∃ (t : Time24), ∀ (t' : Time24), timeDigitSum t' ≤ timeDigitSum t ∧ timeDigitSum t = 24 :=
sorry

end NUMINAMATH_CALUDE_max_time_digit_sum_l4036_403644


namespace NUMINAMATH_CALUDE_inverse_proportion_in_first_third_quadrants_l4036_403614

/-- An inverse proportion function -/
def InverseProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- A function whose graph lies in the first and third quadrants -/
def FirstThirdQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → (x > 0 → f x > 0) ∧ (x < 0 → f x < 0)

theorem inverse_proportion_in_first_third_quadrants
  (f : ℝ → ℝ) (h1 : InverseProportion f) (h2 : FirstThirdQuadrants f) :
  ∃ k : ℝ, k > 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_in_first_third_quadrants_l4036_403614


namespace NUMINAMATH_CALUDE_progression_check_l4036_403620

theorem progression_check (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 6) (hc : c = 4.5) :
  (∃ (r m : ℝ), (b / a) ^ r = (c / a) ^ m) ∧
  ¬(∃ (d : ℝ), b - a = c - b) :=
by sorry

end NUMINAMATH_CALUDE_progression_check_l4036_403620


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_perimeter_l4036_403616

/-- Given a square divided into 4 identical rectangles, each with a perimeter of 20,
    the area of the square is 1600/9. -/
theorem square_area_from_rectangle_perimeter :
  ∀ (s : ℝ), s > 0 →
  (∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ 2 * (l + w) = 20 ∧ 2 * l = s ∧ 2 * w = s) →
  s^2 = 1600 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_perimeter_l4036_403616


namespace NUMINAMATH_CALUDE_race_outcomes_count_l4036_403691

/-- The number of participants in the race -/
def n : ℕ := 6

/-- The number of places we're considering -/
def k : ℕ := 4

/-- The number of different possible outcomes for the first four places in the race -/
def race_outcomes : ℕ := n * (n - 1) * (n - 2) * (n - 3)

theorem race_outcomes_count : race_outcomes = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_count_l4036_403691


namespace NUMINAMATH_CALUDE_lcm_gcd_ratio_240_360_l4036_403608

theorem lcm_gcd_ratio_240_360 : (lcm 240 360) / (gcd 240 360) = 6 := by sorry

end NUMINAMATH_CALUDE_lcm_gcd_ratio_240_360_l4036_403608


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l4036_403639

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 9 to base 10 -/
def base9ToBase10 (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem base_conversion_subtraction :
  base8ToBase10 52103 - base9ToBase10 1452 = 20471 := by sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l4036_403639


namespace NUMINAMATH_CALUDE_a_1_value_c_is_arithmetic_l4036_403673

def sequence_a (n : ℕ) : ℝ := sorry

def sum_S (n : ℕ) : ℝ := sorry

def sequence_c (n : ℕ) : ℝ := sorry

axiom sum_relation (n : ℕ) : sum_S n / 2 = sequence_a n - 2^n

axiom a_relation (n : ℕ) : sequence_a n = 2^n * sequence_c n

theorem a_1_value : sequence_a 1 = 4 := sorry

theorem c_is_arithmetic : ∃ (d : ℝ), ∀ (n : ℕ), n > 0 → sequence_c (n + 1) - sequence_c n = d := sorry

end NUMINAMATH_CALUDE_a_1_value_c_is_arithmetic_l4036_403673


namespace NUMINAMATH_CALUDE_dogwood_trees_in_park_l4036_403618

theorem dogwood_trees_in_park (current : ℕ) (planted : ℕ) (total : ℕ) : 
  planted = 49 → total = 83 → current + planted = total → current = 34 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_in_park_l4036_403618


namespace NUMINAMATH_CALUDE_f_increasing_on_2_3_l4036_403630

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem f_increasing_on_2_3 (heven : is_even f) (hperiodic : is_periodic f 2) 
  (hdecr : is_decreasing_on f (-1) 0) : is_increasing_on f 2 3 := by sorry

end NUMINAMATH_CALUDE_f_increasing_on_2_3_l4036_403630


namespace NUMINAMATH_CALUDE_factors_of_28350_l4036_403641

/-- The number of positive factors of 28350 -/
def num_factors_28350 : ℕ := sorry

/-- 28350 is the number we are analyzing -/
def n : ℕ := 28350

theorem factors_of_28350 : num_factors_28350 = 48 := by sorry

end NUMINAMATH_CALUDE_factors_of_28350_l4036_403641


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l4036_403683

def ends_with_6 (n : ℕ) : Prop := n % 10 = 6

def move_6_to_front (n : ℕ) : ℕ :=
  let d := (Nat.log 10 n) + 1
  6 * 10^d + n / 10

theorem smallest_number_with_properties : ℕ := by
  let n := 1538466
  have h1 : ends_with_6 n := by sorry
  have h2 : move_6_to_front n = 4 * n := by sorry
  have h3 : ∀ m < n, ¬(ends_with_6 m ∧ move_6_to_front m = 4 * m) := by sorry
  exact n

end NUMINAMATH_CALUDE_smallest_number_with_properties_l4036_403683


namespace NUMINAMATH_CALUDE_cheese_division_theorem_l4036_403649

/-- Represents the weight of cheese pieces -/
structure CheesePair :=
  (larger : ℕ)
  (smaller : ℕ)

/-- Simulates taking a bite from the larger piece -/
def takeBite (pair : CheesePair) : CheesePair :=
  ⟨pair.larger - pair.smaller, pair.smaller⟩

/-- Theorem: If after three bites, the cheese pieces are equal and weigh 20 grams each,
    then the original cheese weight was 680 grams -/
theorem cheese_division_theorem (initial : CheesePair) :
  (takeBite (takeBite (takeBite initial))) = ⟨20, 20⟩ →
  initial.larger + initial.smaller = 680 :=
by
  sorry

#check cheese_division_theorem

end NUMINAMATH_CALUDE_cheese_division_theorem_l4036_403649


namespace NUMINAMATH_CALUDE_f_is_odd_and_decreasing_l4036_403652

def f (x : ℝ) : ℝ := -x^3

theorem f_is_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_f_is_odd_and_decreasing_l4036_403652


namespace NUMINAMATH_CALUDE_farm_hens_count_l4036_403603

/-- Proves that the number of hens on a farm is 67, given the conditions. -/
theorem farm_hens_count (roosters hens : ℕ) : 
  hens = 9 * roosters - 5 →
  hens + roosters = 75 →
  hens = 67 := by
  sorry

end NUMINAMATH_CALUDE_farm_hens_count_l4036_403603


namespace NUMINAMATH_CALUDE_number_problem_l4036_403633

theorem number_problem : ∃ x : ℝ, (x / 5 - 5 = 5) ∧ (x = 50) := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4036_403633


namespace NUMINAMATH_CALUDE_cricket_game_solution_l4036_403670

def cricket_game (initial_run_rate : ℝ) (required_rate : ℝ) (total_target : ℝ) : Prop :=
  ∃ (initial_overs : ℝ),
    initial_overs > 0 ∧
    initial_overs < 50 ∧
    initial_overs + 40 = 50 ∧
    initial_run_rate * initial_overs + required_rate * 40 = total_target

theorem cricket_game_solution :
  cricket_game 3.2 5.5 252 → ∃ (initial_overs : ℝ), initial_overs = 10 := by
  sorry

end NUMINAMATH_CALUDE_cricket_game_solution_l4036_403670


namespace NUMINAMATH_CALUDE_triangle_property_l4036_403680

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if acosB = (1/2)b + c, then A = 2π/3 and (b^2 + c^2 + bc) / (4R^2) = 3/4,
    where R is the radius of the circumcircle of triangle ABC -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.cos B = (1/2) * b + c →
  R > 0 →
  A = 2 * π / 3 ∧ (b^2 + c^2 + b*c) / (4 * R^2) = 3/4 := by sorry

end NUMINAMATH_CALUDE_triangle_property_l4036_403680


namespace NUMINAMATH_CALUDE_sector_central_angle_l4036_403646

theorem sector_central_angle (area : ℝ) (perimeter : ℝ) (h1 : area = 5) (h2 : perimeter = 9) :
  ∃ (r : ℝ) (l : ℝ),
    2 * r + l = perimeter ∧
    1/2 * l * r = area ∧
    (l / r = 5/2 ∨ l / r = 8/5) :=
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l4036_403646


namespace NUMINAMATH_CALUDE_problem_3_l4036_403605

theorem problem_3 (a : ℝ) : a = 1 / (Real.sqrt 5 - 2) → 2 * a^2 - 8 * a + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_3_l4036_403605


namespace NUMINAMATH_CALUDE_randys_trip_l4036_403612

theorem randys_trip (x : ℚ) 
  (h1 : x / 4 + 30 + x / 6 = x) : x = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_randys_trip_l4036_403612


namespace NUMINAMATH_CALUDE_least_possible_area_of_square_l4036_403637

/-- The least possible side length of a square when measured as 5 cm to the nearest centimeter -/
def least_side_length : ℝ := 4.5

/-- The reported measurement of the square's side length in centimeters -/
def reported_length : ℕ := 5

/-- Theorem: The least possible area of a square with sides measured as 5 cm to the nearest centimeter is 20.25 cm² -/
theorem least_possible_area_of_square (side : ℝ) 
    (h1 : side ≥ least_side_length) 
    (h2 : side < least_side_length + 1) 
    (h3 : ⌊side⌋ = reported_length ∨ ⌈side⌉ = reported_length) : 
  least_side_length ^ 2 ≤ side ^ 2 :=
sorry

end NUMINAMATH_CALUDE_least_possible_area_of_square_l4036_403637


namespace NUMINAMATH_CALUDE_largest_x_value_l4036_403678

theorem largest_x_value : 
  let f : ℝ → ℝ := λ x => 7 * (9 * x^2 + 8 * x + 12) - x * (9 * x - 45)
  ∃ (x : ℝ), f x = 0 ∧ ∀ (y : ℝ), f y = 0 → y ≤ x ∧ x = -7/6 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l4036_403678


namespace NUMINAMATH_CALUDE_circle_area_and_diameter_l4036_403684

/-- For a circle with circumference 36 cm, prove its area and diameter -/
theorem circle_area_and_diameter (C : ℝ) (h : C = 36) :
  ∃ (A d : ℝ),
    A = 324 / Real.pi ∧
    d = 36 / Real.pi ∧
    C = Real.pi * d ∧
    A = Real.pi * (d / 2)^2 := by
sorry


end NUMINAMATH_CALUDE_circle_area_and_diameter_l4036_403684


namespace NUMINAMATH_CALUDE_igloo_construction_l4036_403627

def igloo_bricks (n : ℕ) : ℕ :=
  if n ≤ 6 then
    14 + 2 * (n - 1)
  else
    24 - 3 * (n - 6)

def total_bricks : ℕ := (List.range 10).map (λ i => igloo_bricks (i + 1)) |>.sum

theorem igloo_construction :
  total_bricks = 170 := by
  sorry

end NUMINAMATH_CALUDE_igloo_construction_l4036_403627


namespace NUMINAMATH_CALUDE_prob_at_least_two_pass_written_is_0_6_expected_students_with_advantage_is_0_96_l4036_403674

-- Define the probabilities for each student passing the written test
def prob_written_A : ℝ := 0.4
def prob_written_B : ℝ := 0.8
def prob_written_C : ℝ := 0.5

-- Define the probabilities for each student passing the interview
def prob_interview_A : ℝ := 0.8
def prob_interview_B : ℝ := 0.4
def prob_interview_C : ℝ := 0.64

-- Function to calculate the probability of at least two students passing the written test
def prob_at_least_two_pass_written : ℝ :=
  prob_written_A * prob_written_B * (1 - prob_written_C) +
  prob_written_A * (1 - prob_written_B) * prob_written_C +
  (1 - prob_written_A) * prob_written_B * prob_written_C +
  prob_written_A * prob_written_B * prob_written_C

-- Function to calculate the probability of a student receiving admission advantage
def prob_admission_advantage (written_prob interview_prob : ℝ) : ℝ :=
  written_prob * interview_prob

-- Function to calculate the mathematical expectation of students receiving admission advantage
def expected_students_with_advantage : ℝ :=
  3 * (prob_admission_advantage prob_written_A prob_interview_A)

-- Theorem statements
theorem prob_at_least_two_pass_written_is_0_6 :
  prob_at_least_two_pass_written = 0.6 := by sorry

theorem expected_students_with_advantage_is_0_96 :
  expected_students_with_advantage = 0.96 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_pass_written_is_0_6_expected_students_with_advantage_is_0_96_l4036_403674


namespace NUMINAMATH_CALUDE_hexagon_triangle_count_l4036_403606

/-- Represents a regular hexagon -/
structure RegularHexagon :=
  (area : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (area : ℝ)

/-- Counts the number of equilateral triangles with a given area that can be formed from the vertices of a set of regular hexagons -/
def countEquilateralTriangles (hexagons : List RegularHexagon) (targetTriangle : EquilateralTriangle) : ℕ :=
  sorry

/-- The main theorem stating that 4 regular hexagons with area 6 can form 8 equilateral triangles with area 4 -/
theorem hexagon_triangle_count :
  let hexagons := List.replicate 4 { area := 6 : RegularHexagon }
  let targetTriangle := { area := 4 : EquilateralTriangle }
  countEquilateralTriangles hexagons targetTriangle = 8 := by sorry

end NUMINAMATH_CALUDE_hexagon_triangle_count_l4036_403606


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l4036_403665

/-- Given a hyperbola with equation x²/9 - y²/m = 1 and one focus at (-5, 0),
    prove that its asymptotes are y = ±(4/3)x -/
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ (x y : ℝ), x^2/9 - y^2/m = 1) →
  (5^2 = 9 + m) →
  (∀ (x y : ℝ), x^2/9 - y^2/m = 1 → (y = (4/3)*x ∨ y = -(4/3)*x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l4036_403665


namespace NUMINAMATH_CALUDE_same_color_probability_problem_die_l4036_403607

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)
  (golden : ℕ)
  (total : ℕ)
  (h_total : red + green + blue + golden = total)

/-- The probability of rolling the same color on two identical colored dice -/
def same_color_probability (d : ColoredDie) : ℚ :=
  (d.red^2 + d.green^2 + d.blue^2 + d.golden^2 : ℚ) / d.total^2

/-- The specific 30-sided die described in the problem -/
def problem_die : ColoredDie :=
  { red := 6
  , green := 8
  , blue := 10
  , golden := 6
  , total := 30
  , h_total := by simp }

/-- Theorem stating the probability of rolling the same color on two problem_die is 59/225 -/
theorem same_color_probability_problem_die :
  same_color_probability problem_die = 59 / 225 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_problem_die_l4036_403607


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l4036_403604

def a : ℝ × ℝ := (2, 3)
def b (t : ℝ) : ℝ × ℝ := (t, -1)

theorem perpendicular_vectors_magnitude (t : ℝ) :
  (a.1 * (b t).1 + a.2 * (b t).2 = 0) →
  Real.sqrt ((a.1 - 2 * (b t).1)^2 + (a.2 - 2 * (b t).2)^2) = Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l4036_403604


namespace NUMINAMATH_CALUDE_sticker_distribution_l4036_403613

theorem sticker_distribution (n k : ℕ) (hn : n = 10) (hk : k = 5) :
  Nat.choose (n + k - 1) (k - 1) = 1001 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l4036_403613


namespace NUMINAMATH_CALUDE_copies_equals_totient_copies_2019_l4036_403682

-- Define the pattern
def pattern (n : ℕ) : List ℕ :=
  match n with
  | 0 => []
  | 1 => [1, 1]
  | n + 1 => 
    let prev := pattern n
    List.zipWith (·+·) prev (prev.tail! ++ [0])

-- Define the property we want to prove
theorem copies_equals_totient (n : ℕ) : 
  (pattern n).count n = Nat.totient n :=
sorry

-- The specific case for 2019
theorem copies_2019 : (pattern 2019).count 2019 = 1344 :=
sorry

end NUMINAMATH_CALUDE_copies_equals_totient_copies_2019_l4036_403682


namespace NUMINAMATH_CALUDE_power_two_divisibility_l4036_403610

theorem power_two_divisibility (n k : ℕ) (a b : ℤ) : 
  2^n - 1 = a * b →
  (∃ m : ℕ, 2^k * m = 2^(n-2) + a - b ∧ ∀ l : ℕ, 2^l * m = 2^(n-2) + a - b → l ≤ k) →
  ∃ m : ℕ, k = 2 * m :=
by sorry

end NUMINAMATH_CALUDE_power_two_divisibility_l4036_403610


namespace NUMINAMATH_CALUDE_units_digit_of_N_l4036_403635

def N : ℕ := 3^1001 + 7^1002 + 13^1003

theorem units_digit_of_N (n : ℕ) (h : n = N) : n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_N_l4036_403635


namespace NUMINAMATH_CALUDE_fraction_numerator_l4036_403648

theorem fraction_numerator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : y / 20 + x = 0.35 * y) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_numerator_l4036_403648


namespace NUMINAMATH_CALUDE_magic_square_sum_l4036_403656

/-- Represents a 3x3 magic square with numbers 1, 2, 3 -/
def MagicSquare : Type := Fin 3 → Fin 3 → Fin 3

/-- Checks if a row contains 1, 2, 3 exactly once -/
def valid_row (square : MagicSquare) (row : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃! col : Fin 3, square row col = n

/-- Checks if a column contains 1, 2, 3 exactly once -/
def valid_column (square : MagicSquare) (col : Fin 3) : Prop :=
  ∀ n : Fin 3, ∃! row : Fin 3, square row col = n

/-- Checks if the main diagonal contains 1, 2, 3 exactly once -/
def valid_diagonal (square : MagicSquare) : Prop :=
  ∀ n : Fin 3, ∃! i : Fin 3, square i i = n

/-- Defines a valid magic square -/
def is_valid_square (square : MagicSquare) : Prop :=
  (∀ row : Fin 3, valid_row square row) ∧
  (∀ col : Fin 3, valid_column square col) ∧
  valid_diagonal square

theorem magic_square_sum :
  ∀ square : MagicSquare,
  is_valid_square square →
  square 0 0 = 2 →
  (square 1 0).val + (square 2 2).val + (square 1 1).val = 6 :=
by sorry

end NUMINAMATH_CALUDE_magic_square_sum_l4036_403656


namespace NUMINAMATH_CALUDE_gcd_735_1287_l4036_403664

theorem gcd_735_1287 : Nat.gcd 735 1287 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_735_1287_l4036_403664


namespace NUMINAMATH_CALUDE_max_abs_z_value_l4036_403663

theorem max_abs_z_value (a b c z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs a = 2 * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ (1 + Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_value_l4036_403663


namespace NUMINAMATH_CALUDE_oranges_per_bag_l4036_403672

theorem oranges_per_bag (total_oranges : ℕ) (num_bags : ℕ) (h1 : total_oranges = 1035) (h2 : num_bags = 45) (h3 : total_oranges % num_bags = 0) : 
  total_oranges / num_bags = 23 := by
sorry

end NUMINAMATH_CALUDE_oranges_per_bag_l4036_403672


namespace NUMINAMATH_CALUDE_min_value_of_f_l4036_403626

/-- The function f(x) = 3x^2 - 12x + 7 + 749 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 7 + 749

theorem min_value_of_f :
  ∃ (m : ℝ), m = 744 ∧ ∀ (x : ℝ), f x ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l4036_403626


namespace NUMINAMATH_CALUDE_point_P_in_second_quadrant_l4036_403602

def point_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_P_in_second_quadrant :
  point_in_second_quadrant (-1 : ℝ) (2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_point_P_in_second_quadrant_l4036_403602


namespace NUMINAMATH_CALUDE_complex_function_minimum_on_unit_circle_l4036_403622

/-- For a ∈ (0,1) and f(z) = z^2 - z + a, for any complex number z with |z| ≥ 1,
    there exists a complex number z₀ with |z₀| = 1 such that |f(z₀)| ≤ |f(z)| -/
theorem complex_function_minimum_on_unit_circle (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∀ z : ℂ, Complex.abs z ≥ 1 →
    ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧
      Complex.abs (z₀^2 - z₀ + a) ≤ Complex.abs (z^2 - z + a) :=
by sorry

end NUMINAMATH_CALUDE_complex_function_minimum_on_unit_circle_l4036_403622


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l4036_403628

theorem fractional_equation_solution :
  ∃ x : ℝ, (1 / x = 2 / (x + 1)) ∧ (x = 1) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l4036_403628


namespace NUMINAMATH_CALUDE_extra_time_at_reduced_speed_l4036_403631

theorem extra_time_at_reduced_speed 
  (usual_time : ℝ) 
  (speed_ratio : ℝ) 
  (h1 : usual_time = 72.00000000000001)
  (h2 : speed_ratio = 0.75) : 
  (usual_time / speed_ratio) - usual_time = 24 := by
sorry

end NUMINAMATH_CALUDE_extra_time_at_reduced_speed_l4036_403631


namespace NUMINAMATH_CALUDE_expression_evaluation_l4036_403609

theorem expression_evaluation :
  (4^1001 * 9^1002) / (6^1002 * 4^1000) = 3^1002 / 2^1000 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4036_403609


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l4036_403698

/-- A line parallel to y = -3x + 6 passing through (3, -2) has y-intercept 7 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b y = -3 * x + b 0) →  -- b is a linear function with slope -3
  b (-2) = 3 →                   -- b passes through (3, -2)
  b 0 = 7 :=                     -- y-intercept of b is 7
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l4036_403698


namespace NUMINAMATH_CALUDE_mayoral_race_vote_distribution_l4036_403638

theorem mayoral_race_vote_distribution (total_voters : ℝ) 
  (h1 : total_voters > 0) 
  (dem_percent : ℝ) 
  (h2 : dem_percent = 0.6) 
  (dem_vote_for_a : ℝ) 
  (h3 : dem_vote_for_a = 0.7) 
  (total_vote_for_a : ℝ) 
  (h4 : total_vote_for_a = 0.5) : 
  (total_vote_for_a * total_voters - dem_vote_for_a * dem_percent * total_voters) / 
  ((1 - dem_percent) * total_voters) = 0.2 := by
sorry

end NUMINAMATH_CALUDE_mayoral_race_vote_distribution_l4036_403638


namespace NUMINAMATH_CALUDE_right_triangle_area_l4036_403621

/-- The area of a right triangle with given side lengths -/
theorem right_triangle_area 
  (X Y Z : ℝ × ℝ) -- Points in 2D plane
  (h_right : (X.1 - Y.1) * (X.1 - Z.1) + (X.2 - Y.2) * (X.2 - Z.2) = 0) -- Right angle at X
  (h_xy : Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 15) -- XY = 15
  (h_xz : Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 10) -- XZ = 10
  (h_median : ∃ M : ℝ × ℝ, M.1 = (Y.1 + Z.1) / 2 ∧ M.2 = (Y.2 + Z.2) / 2 ∧ 
    (X.1 - M.1) * (Y.1 - Z.1) + (X.2 - M.2) * (Y.2 - Z.2) = 0) -- Median bisects angle X
  : (1 / 2 : ℝ) * 15 * 10 = 75 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l4036_403621


namespace NUMINAMATH_CALUDE_ages_sum_after_three_years_l4036_403653

/-- Given four persons a, b, c, and d with the following age relationships:
    - The sum of their present ages is S
    - a's age is twice b's age
    - c's age is half of a's age
    - d's age is the difference between a's and c's ages
    This theorem proves that the sum of their ages after 3 years is S + 12 -/
theorem ages_sum_after_three_years
  (S : ℝ) -- Sum of present ages
  (a b c d : ℝ) -- Present ages of individuals
  (h1 : a + b + c + d = S) -- Sum of present ages is S
  (h2 : a = 2 * b) -- a's age is twice b's age
  (h3 : c = a / 2) -- c's age is half of a's age
  (h4 : d = a - c) -- d's age is the difference between a's and c's ages
  : (a + 3) + (b + 3) + (c + 3) + (d + 3) = S + 12 := by
  sorry


end NUMINAMATH_CALUDE_ages_sum_after_three_years_l4036_403653


namespace NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l4036_403666

noncomputable def f (x : ℝ) := x^3 - 3*x - 1

theorem f_monotonicity_and_intersection (x : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ ∧ ((x₁ < -1 ∧ x₂ < -1) ∨ (x₁ > 1 ∧ x₂ > 1)) → f x₁ < f x₂) ∧
  (∀ x₁ x₂, -1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f x₁ ≥ f x₂) ∧
  (∀ m : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ f x₁ = m ∧ f x₂ = m ∧ f x₃ = m) ↔ -3 < m ∧ m < 1) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_intersection_l4036_403666


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l4036_403617

theorem completing_square_quadratic (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l4036_403617


namespace NUMINAMATH_CALUDE_quadratic_root_sum_squares_l4036_403696

theorem quadratic_root_sum_squares (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  (∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + b*x + c = 0) ∧
  (∃! y : ℝ, y^2 + b*y + c = 0 ∧ y^2 + c*y + a = 0) ∧
  (∃! z : ℝ, z^2 + c*z + a = 0 ∧ z^2 + a*z + b = 0) →
  a^2 + b^2 + c^2 = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_squares_l4036_403696


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l4036_403692

theorem davids_chemistry_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (biology : ℕ)
  (average : ℚ)
  (h1 : english = 96)
  (h2 : mathematics = 95)
  (h3 : physics = 82)
  (h4 : biology = 92)
  (h5 : average = 90.4)
  (h6 : (english + mathematics + physics + biology + chemistry : ℚ) / 5 = average) :
  chemistry = 87 :=
by sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l4036_403692


namespace NUMINAMATH_CALUDE_min_a_for_g_zeros_l4036_403659

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x| + |x - 1|

-- Define the function g(x) in terms of f(x) and a
def g (a : ℝ) (x : ℝ) : ℝ := f x - a

-- Theorem statement
theorem min_a_for_g_zeros :
  ∃ (a : ℝ), (∃ (x : ℝ), g a x = 0) ∧
  (∀ (b : ℝ), b < a → ¬∃ (x : ℝ), g b x = 0) ∧
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_g_zeros_l4036_403659


namespace NUMINAMATH_CALUDE_puppies_given_to_friends_l4036_403686

/-- The number of puppies Alyssa started with -/
def initial_puppies : ℕ := 12

/-- The number of puppies Alyssa has left -/
def remaining_puppies : ℕ := 5

/-- The number of puppies Alyssa gave to her friends -/
def given_puppies : ℕ := initial_puppies - remaining_puppies

theorem puppies_given_to_friends : given_puppies = 7 := by
  sorry

end NUMINAMATH_CALUDE_puppies_given_to_friends_l4036_403686


namespace NUMINAMATH_CALUDE_pete_susan_speed_ratio_l4036_403655

/-- Given the walking and cartwheel speeds of Pete, Susan, and Tracy, prove that the ratio of Pete's backward walking speed to Susan's forward walking speed is 3. -/
theorem pete_susan_speed_ratio :
  ∀ (pete_backward pete_hands tracy_cartwheel susan_forward : ℝ),
  pete_hands > 0 →
  pete_backward > 0 →
  tracy_cartwheel > 0 →
  susan_forward > 0 →
  tracy_cartwheel = 2 * susan_forward →
  pete_hands = (1 / 4) * tracy_cartwheel →
  pete_hands = 2 →
  pete_backward = 12 →
  pete_backward / susan_forward = 3 := by
sorry

end NUMINAMATH_CALUDE_pete_susan_speed_ratio_l4036_403655


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l4036_403642

theorem fraction_equation_solution : ∃ x : ℚ, (1 / 3 - 1 / 4 : ℚ) = 1 / x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l4036_403642


namespace NUMINAMATH_CALUDE_artist_painted_thirteen_pictures_l4036_403669

/-- The number of pictures painted by an artist over three months -/
def total_pictures (june july august : ℕ) : ℕ := june + july + august

/-- Theorem stating that the artist painted 13 pictures in total -/
theorem artist_painted_thirteen_pictures : 
  total_pictures 2 2 9 = 13 := by sorry

end NUMINAMATH_CALUDE_artist_painted_thirteen_pictures_l4036_403669


namespace NUMINAMATH_CALUDE_farm_problem_solution_l4036_403694

/-- Represents the farm ploughing problem -/
structure FarmProblem where
  planned_daily_area : ℕ  -- Planned area to plough per day
  actual_daily_area : ℕ   -- Actual area ploughed per day
  extra_days : ℕ          -- Extra days worked
  total_field_area : ℕ    -- Total area of the farm field

/-- Calculates the area left to plough -/
def area_left_to_plough (fp : FarmProblem) : ℕ :=
  let planned_days := fp.total_field_area / fp.planned_daily_area
  let actual_days := planned_days + fp.extra_days
  let ploughed_area := fp.actual_daily_area * actual_days
  fp.total_field_area - ploughed_area

/-- Theorem stating the correct result for the given problem -/
theorem farm_problem_solution :
  let fp : FarmProblem := {
    planned_daily_area := 340,
    actual_daily_area := 85,
    extra_days := 2,
    total_field_area := 280
  }
  area_left_to_plough fp = 25 := by
  sorry

end NUMINAMATH_CALUDE_farm_problem_solution_l4036_403694


namespace NUMINAMATH_CALUDE_expression_evaluation_l4036_403662

theorem expression_evaluation :
  3 + 3 * Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (3 - Real.sqrt 3) = 4 + 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4036_403662


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4036_403660

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 2) * a n = (a (n + 1))^2) 
  (h_a1 : a 1 = 1) 
  (h_product : a 1 * a 2 * a 3 = -8) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = -2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4036_403660


namespace NUMINAMATH_CALUDE_total_ice_cream_scoops_l4036_403681

def single_cone : ℕ := 1
def double_cone : ℕ := 3
def milkshake : ℕ := 2  -- Rounded up from 1.5
def banana_split : ℕ := 4 * single_cone
def waffle_bowl : ℕ := banana_split + 2
def ice_cream_sandwich : ℕ := waffle_bowl - 3

theorem total_ice_cream_scoops : 
  single_cone + double_cone + milkshake + banana_split + waffle_bowl + ice_cream_sandwich = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_ice_cream_scoops_l4036_403681


namespace NUMINAMATH_CALUDE_tyrones_money_value_l4036_403685

/-- Represents the total value of Tyrone's money in US dollars -/
def tyrones_money : ℚ :=
  let us_currency : ℚ :=
    4 * 1 +  -- $1 bills
    1 * 10 +  -- $10 bill
    2 * 5 +  -- $5 bills
    30 * (1/4) +  -- quarters
    5 * (1/2) +  -- half-dollar coins
    48 * (1/10) +  -- dimes
    12 * (1/20) +  -- nickels
    4 * 1 +  -- one-dollar coins
    64 * (1/100) +  -- pennies
    3 * 2 +  -- two-dollar bills
    5 * (1/2)  -- 50-cent coins

  let foreign_currency : ℚ :=
    20 * (11/10) +  -- Euro coins
    15 * (132/100) +  -- British Pound coins
    6 * (76/100)  -- Canadian Dollar coins

  us_currency + foreign_currency

/-- The theorem stating that Tyrone's money equals $98.90 -/
theorem tyrones_money_value : tyrones_money = 989/10 := by
  sorry

end NUMINAMATH_CALUDE_tyrones_money_value_l4036_403685


namespace NUMINAMATH_CALUDE_product_digits_l4036_403689

def a : ℕ := 8476235982145327
def b : ℕ := 2983674531

theorem product_digits : (String.length (toString (a * b))) = 28 := by
  sorry

end NUMINAMATH_CALUDE_product_digits_l4036_403689


namespace NUMINAMATH_CALUDE_polynomial_value_l4036_403679

/-- A polynomial of degree 5 with integer coefficients -/
def polynomial (a₁ a₂ a₃ a₄ a₅ : ℤ) (x : ℝ) : ℝ :=
  x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅

theorem polynomial_value (a₁ a₂ a₃ a₄ a₅ : ℤ) :
  let f := polynomial a₁ a₂ a₃ a₄ a₅
  (f (Real.sqrt 3 + Real.sqrt 2) = 0) →
  (f 1 + f 3 = 0) →
  (f (-1) = 24) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l4036_403679


namespace NUMINAMATH_CALUDE_triangle_sides_simplification_l4036_403645

theorem triangle_sides_simplification (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : a + c > b) 
  (h4 : a > 0) 
  (h5 : b > 0) 
  (h6 : c > 0) : 
  |c - a - b| + |c + b - a| = 2 * b := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_simplification_l4036_403645


namespace NUMINAMATH_CALUDE_circles_tangent_implies_m_equals_four_l4036_403619

-- Define the circles
def circle_C (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 5 - m}
def circle_E : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 16}

-- Define the condition for external tangency
def externally_tangent (C E : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ), p ∈ C ∧ p ∈ E ∧ 
  ∀ (q : ℝ × ℝ), q ∈ C ∩ E → q = p

-- State the theorem
theorem circles_tangent_implies_m_equals_four :
  ∀ (m : ℝ), externally_tangent (circle_C m) circle_E → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_circles_tangent_implies_m_equals_four_l4036_403619


namespace NUMINAMATH_CALUDE_bryden_received_value_l4036_403693

/-- The face value of a state quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 10

/-- The percentage of face value the collector offers -/
def collector_offer_percentage : ℕ := 1500

/-- The amount Bryden will receive for his quarters in dollars -/
def bryden_received : ℚ := (bryden_quarters : ℚ) * quarter_value * (collector_offer_percentage : ℚ) / 100

theorem bryden_received_value : bryden_received = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_bryden_received_value_l4036_403693


namespace NUMINAMATH_CALUDE_lower_variance_less_volatile_l4036_403699

/-- Represents a shooter's performance --/
structure ShooterPerformance where
  average_score : ℝ
  variance : ℝ
  num_shots : ℕ

/-- Defines volatility based on variance --/
def less_volatile (a b : ShooterPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two shooters with the same average score but different variances,
    the shooter with the lower variance has less volatile performance --/
theorem lower_variance_less_volatile (a b : ShooterPerformance) 
  (h1 : a.average_score = b.average_score)
  (h2 : a.variance ≠ b.variance)
  (h3 : a.num_shots = b.num_shots)
  : less_volatile (if a.variance < b.variance then a else b) (if a.variance > b.variance then a else b) :=
by
  sorry

end NUMINAMATH_CALUDE_lower_variance_less_volatile_l4036_403699


namespace NUMINAMATH_CALUDE_five_is_solution_l4036_403677

/-- The equation we're working with -/
def equation (x : ℝ) : Prop :=
  x^3 + 2*(x+1)^3 + 3*(x+2)^3 = 3*(x+3)^3

/-- Theorem stating that 5 is a solution to the equation -/
theorem five_is_solution : equation 5 := by
  sorry

end NUMINAMATH_CALUDE_five_is_solution_l4036_403677


namespace NUMINAMATH_CALUDE_x_squared_eq_5_is_quadratic_l4036_403632

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 5 -/
def f (x : ℝ) : ℝ := x^2 - 5

theorem x_squared_eq_5_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_x_squared_eq_5_is_quadratic_l4036_403632


namespace NUMINAMATH_CALUDE_k_equals_nine_l4036_403671

/-- Two circles centered at the origin with specific points and distances -/
structure TwoCircles where
  -- Radius of the larger circle
  R : ℝ
  -- Radius of the smaller circle
  r : ℝ
  -- Point P on the larger circle
  P : ℝ × ℝ
  -- Point S on the smaller circle
  S : ℝ × ℝ
  -- Distance QR
  QR : ℝ
  -- Conditions
  P_on_larger : P.1^2 + P.2^2 = R^2
  S_on_smaller : S.1^2 + S.2^2 = r^2
  P_coords : P = (5, 12)
  S_coords : S = (0, S.2)
  QR_value : QR = 4

/-- The theorem stating that k (the y-coordinate of S) equals 9 -/
theorem k_equals_nine (c : TwoCircles) : c.S.2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_k_equals_nine_l4036_403671


namespace NUMINAMATH_CALUDE_division_remainder_problem_l4036_403651

theorem division_remainder_problem :
  let dividend : ℕ := 171
  let divisor : ℕ := 21
  let quotient : ℕ := 8
  let remainder : ℕ := dividend - divisor * quotient
  remainder = 3 := by sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l4036_403651


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4036_403615

def A : Set ℝ := {x | 2 * x - 1 > 0}
def B : Set ℝ := {x | x * (x - 2) < 0}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 1/2 < x ∧ x < 2} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4036_403615


namespace NUMINAMATH_CALUDE_prime_sequence_ones_digit_l4036_403690

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_sequence_ones_digit (p q r s : ℕ) :
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧
  p > 10 ∧
  q = p + 10 ∧ r = q + 10 ∧ s = r + 10 →
  ones_digit p = 1 :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_ones_digit_l4036_403690


namespace NUMINAMATH_CALUDE_equation_solution_l4036_403676

theorem equation_solution : ∃ X : ℝ, 
  (0.125 * X) / ((19/24 - 21/40) * 8*(7/16)) = 
  ((1 + 28/63 - 17/21) * 0.7) / (0.675 * 2.4 - 0.02) ∧ X = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4036_403676


namespace NUMINAMATH_CALUDE_zero_area_quadrilateral_l4036_403640

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of a quadrilateral given its four vertices in 3D space -/
def quadrilateralArea (A B C D : Point3D) : ℝ :=
  sorry

/-- The main theorem stating that the area of the quadrilateral with given vertices is 0 -/
theorem zero_area_quadrilateral :
  let A : Point3D := ⟨2, 4, 6⟩
  let B : Point3D := ⟨7, 9, 11⟩
  let C : Point3D := ⟨1, 3, 5⟩
  let D : Point3D := ⟨6, 8, 10⟩
  quadrilateralArea A B C D = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_area_quadrilateral_l4036_403640


namespace NUMINAMATH_CALUDE_nonreal_roots_product_l4036_403625

theorem nonreal_roots_product (x : ℂ) : 
  (x^4 - 4*x^3 + 6*x^2 - 4*x = 2047) → 
  (∃ a b : ℂ, a ≠ b ∧ a.im ≠ 0 ∧ b.im ≠ 0 ∧ 
   (x = a ∨ x = b) ∧ (x^4 - 4*x^3 + 6*x^2 - 4*x = 2047) ∧
   (a * b = 257)) := by
sorry

end NUMINAMATH_CALUDE_nonreal_roots_product_l4036_403625


namespace NUMINAMATH_CALUDE_remaining_distance_to_hotel_l4036_403675

/-- Calculates the remaining distance to the hotel given the initial conditions of Samuel's journey --/
theorem remaining_distance_to_hotel (total_distance : ℝ) (initial_speed : ℝ) (initial_time : ℝ) (second_speed : ℝ) (second_time : ℝ) :
  total_distance = 600 ∧
  initial_speed = 50 ∧
  initial_time = 3 ∧
  second_speed = 80 ∧
  second_time = 4 →
  total_distance - (initial_speed * initial_time + second_speed * second_time) = 130 := by
  sorry

#check remaining_distance_to_hotel

end NUMINAMATH_CALUDE_remaining_distance_to_hotel_l4036_403675


namespace NUMINAMATH_CALUDE_acme_soup_words_count_l4036_403654

/-- Represents the number of times each vowel (A, E, I, O, U) appears -/
def vowel_count : ℕ := 5

/-- Represents the number of times Y appears -/
def y_count : ℕ := 3

/-- Represents the length of words to be formed -/
def word_length : ℕ := 5

/-- Represents the number of vowels (A, E, I, O, U) -/
def num_vowels : ℕ := 5

/-- Calculates the number of five-letter words that can be formed -/
def acme_soup_words : ℕ := 
  (num_vowels ^ word_length) + 
  (word_length * (num_vowels ^ (word_length - 1))) +
  (Nat.choose word_length 2 * (num_vowels ^ (word_length - 2))) +
  (Nat.choose word_length 3 * (num_vowels ^ (word_length - 3)))

theorem acme_soup_words_count : acme_soup_words = 7750 := by
  sorry

end NUMINAMATH_CALUDE_acme_soup_words_count_l4036_403654


namespace NUMINAMATH_CALUDE_probability_both_preferred_is_one_fourth_l4036_403657

/-- Represents the colors of the balls -/
inductive Color
| Red
| Yellow
| Blue
| Green
| Purple

/-- Represents a person -/
structure Person where
  name : String
  preferredColors : List Color

/-- Represents the bag of balls -/
def bag : List Color := [Color.Red, Color.Yellow, Color.Blue, Color.Green, Color.Purple]

/-- Person A's preferred colors -/
def personA : Person := { name := "A", preferredColors := [Color.Red, Color.Yellow] }

/-- Person B's preferred colors -/
def personB : Person := { name := "B", preferredColors := [Color.Yellow, Color.Green, Color.Purple] }

/-- Calculates the probability of both persons drawing their preferred colors -/
def probabilityBothPreferred (bag : List Color) (personA personB : Person) : ℚ :=
  sorry

/-- Theorem stating the probability of both persons drawing their preferred colors is 1/4 -/
theorem probability_both_preferred_is_one_fourth :
  probabilityBothPreferred bag personA personB = 1/4 :=
sorry

end NUMINAMATH_CALUDE_probability_both_preferred_is_one_fourth_l4036_403657
