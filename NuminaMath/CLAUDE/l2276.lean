import Mathlib

namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l2276_227665

/-- Given a quadratic function with vertex (4, -2) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 7. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = -2 ↔ x = 4) →  -- vertex condition
  a * 1^2 + b * 1 + c = 0 →                 -- x-intercept condition
  ∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l2276_227665


namespace NUMINAMATH_CALUDE_sum_remainder_nine_specific_sum_remainder_l2276_227658

theorem sum_remainder_nine (n : ℕ) : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) % 9 = ((n % 9 + (n + 1) % 9 + (n + 2) % 9 + (n + 3) % 9 + (n + 4) % 9) % 9) := by
  sorry

theorem specific_sum_remainder :
  (9150 + 9151 + 9152 + 9153 + 9154) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_nine_specific_sum_remainder_l2276_227658


namespace NUMINAMATH_CALUDE_prime_pairs_perfect_square_l2276_227681

theorem prime_pairs_perfect_square :
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (∃ a : ℕ, p^2 + p*q + q^2 = a^2) → 
    ((p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_perfect_square_l2276_227681


namespace NUMINAMATH_CALUDE_sum_of_rearranged_digits_l2276_227647

theorem sum_of_rearranged_digits : 1357 + 3571 + 5713 + 7135 = 17776 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rearranged_digits_l2276_227647


namespace NUMINAMATH_CALUDE_square_land_multiple_l2276_227651

theorem square_land_multiple (a p k : ℝ) : 
  a > 0 → 
  p > 0 → 
  p = 36 → 
  a = (p / 4) ^ 2 → 
  5 * a = k * p + 45 → 
  k = 10 := by
sorry

end NUMINAMATH_CALUDE_square_land_multiple_l2276_227651


namespace NUMINAMATH_CALUDE_continuous_function_composite_power_l2276_227686

theorem continuous_function_composite_power (k : ℝ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ ∀ x, f (f x) = k * x^9) → k ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_composite_power_l2276_227686


namespace NUMINAMATH_CALUDE_return_trip_duration_l2276_227639

/-- Represents the flight scenario with given conditions -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- speed of the plane in still air
  w : ℝ  -- speed of the wind
  time_against_wind : ℝ -- time flying against the wind
  time_diff_still_air : ℝ -- time difference compared to still air for return trip

/-- The possible durations for the return trip -/
def possible_return_times : Set ℝ := {60, 40}

/-- Theorem stating that the return trip duration is either 60 or 40 minutes -/
theorem return_trip_duration (scenario : FlightScenario) 
  (h1 : scenario.time_against_wind = 120)
  (h2 : scenario.time_diff_still_air = 20)
  (h3 : scenario.d > 0)
  (h4 : scenario.p > scenario.w)
  (h5 : scenario.w > 0) :
  ∃ (t : ℝ), t ∈ possible_return_times ∧ 
    scenario.d / (scenario.p + scenario.w) = t := by
  sorry


end NUMINAMATH_CALUDE_return_trip_duration_l2276_227639


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2276_227697

theorem trigonometric_identity (A B C : ℝ) : 
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 2 + 2 * Real.cos A * Real.cos B * Real.cos C := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2276_227697


namespace NUMINAMATH_CALUDE_det_matrix_eq_one_l2276_227600

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![5, 7; 2, 3]

theorem det_matrix_eq_one : Matrix.det matrix = 1 := by sorry

end NUMINAMATH_CALUDE_det_matrix_eq_one_l2276_227600


namespace NUMINAMATH_CALUDE_wedding_tables_l2276_227609

theorem wedding_tables (total_fish : ℕ) (fish_per_regular_table : ℕ) (fish_at_special_table : ℕ) :
  total_fish = 65 →
  fish_per_regular_table = 2 →
  fish_at_special_table = 3 →
  ∃ (num_tables : ℕ), num_tables * fish_per_regular_table + (fish_at_special_table - fish_per_regular_table) = total_fish ∧
                       num_tables = 32 := by
  sorry

end NUMINAMATH_CALUDE_wedding_tables_l2276_227609


namespace NUMINAMATH_CALUDE_sum_of_ages_l2276_227659

/-- Given that Rachel is 19 years old and 4 years older than Leah, 
    prove that the sum of their ages is 34. -/
theorem sum_of_ages (rachel_age : ℕ) (leah_age : ℕ) : 
  rachel_age = 19 → rachel_age = leah_age + 4 → rachel_age + leah_age = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2276_227659


namespace NUMINAMATH_CALUDE_questions_per_day_l2276_227688

/-- Given a mathematician who needs to write a certain number of questions for two projects in one week,
    this theorem proves the number of questions he should complete each day. -/
theorem questions_per_day
  (project1_questions : ℕ)
  (project2_questions : ℕ)
  (days_in_week : ℕ)
  (h1 : project1_questions = 518)
  (h2 : project2_questions = 476)
  (h3 : days_in_week = 7) :
  (project1_questions + project2_questions) / days_in_week = 142 := by
  sorry

end NUMINAMATH_CALUDE_questions_per_day_l2276_227688


namespace NUMINAMATH_CALUDE_final_book_count_l2276_227667

/-- The number of storybooks in a library after borrowing and returning books. -/
def library_books (initial : ℕ) (borrowed : ℕ) (returned : ℕ) : ℕ :=
  initial - borrowed + returned

/-- Theorem stating that given the initial conditions, the library ends up with 72 books. -/
theorem final_book_count :
  library_books 95 58 35 = 72 := by
  sorry

end NUMINAMATH_CALUDE_final_book_count_l2276_227667


namespace NUMINAMATH_CALUDE_homework_difference_is_two_l2276_227611

/-- The number of pages of reading homework Rachel had to complete -/
def reading_pages : ℕ := 2

/-- The number of pages of math homework Rachel had to complete -/
def math_pages : ℕ := 4

/-- The difference between math homework pages and reading homework pages -/
def homework_difference : ℕ := math_pages - reading_pages

theorem homework_difference_is_two : homework_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_homework_difference_is_two_l2276_227611


namespace NUMINAMATH_CALUDE_butterfly_flight_l2276_227679

theorem butterfly_flight (field_length field_width start_distance : ℝ) 
  (h1 : field_length = 20)
  (h2 : field_width = 15)
  (h3 : start_distance = 6)
  (h4 : start_distance < field_length / 2) :
  let end_distance := field_length - 2 * start_distance
  let flight_distance := Real.sqrt (field_width ^ 2 + end_distance ^ 2)
  flight_distance = 17 := by sorry

end NUMINAMATH_CALUDE_butterfly_flight_l2276_227679


namespace NUMINAMATH_CALUDE_min_value_expression_l2276_227644

theorem min_value_expression (x : ℝ) (h : x > 2) :
  (x^2 + 8) / Real.sqrt (x - 2) ≥ 22 ∧
  ∃ x₀ > 2, (x₀^2 + 8) / Real.sqrt (x₀ - 2) = 22 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2276_227644


namespace NUMINAMATH_CALUDE_roots_polynomial_equation_l2276_227676

theorem roots_polynomial_equation (p q : ℝ) (α β γ δ : ℂ) :
  (α^2 + p*α + 1 = 0) →
  (β^2 + p*β + 1 = 0) →
  (γ^2 + q*γ + 1 = 0) →
  (δ^2 + q*δ + 1 = 0) →
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
sorry

end NUMINAMATH_CALUDE_roots_polynomial_equation_l2276_227676


namespace NUMINAMATH_CALUDE_special_key_102_presses_l2276_227632

def f (x : ℚ) : ℚ := 1 / (1 - x)

def iterate_f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem special_key_102_presses :
  iterate_f 102 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_special_key_102_presses_l2276_227632


namespace NUMINAMATH_CALUDE_area_30_60_90_triangle_l2276_227625

/-- The area of a 30-60-90 triangle with hypotenuse 6 is 9√3/2 -/
theorem area_30_60_90_triangle (h : Real) (A : Real) : 
  h = 6 → -- hypotenuse is 6 units
  A = (9 * Real.sqrt 3) / 2 → -- area is 9√3/2 square units
  ∃ (s1 s2 : Real), -- there exist two sides s1 and s2 such that
    s1^2 + s2^2 = h^2 ∧ -- Pythagorean theorem
    s1 = h / 2 ∧ -- shortest side is half the hypotenuse
    s2 = s1 * Real.sqrt 3 ∧ -- longer side is √3 times the shorter side
    A = (1 / 2) * s1 * s2 -- area formula
  := by sorry

end NUMINAMATH_CALUDE_area_30_60_90_triangle_l2276_227625


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2276_227696

theorem rectangle_diagonal (perimeter : ℝ) (length_ratio width_ratio : ℕ) 
  (h_perimeter : perimeter = 72) 
  (h_ratio : length_ratio = 5 ∧ width_ratio = 4) : 
  ∃ (length width : ℝ),
    2 * (length + width) = perimeter ∧ 
    length * width_ratio = width * length_ratio ∧
    length^2 + width^2 = 656 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2276_227696


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2276_227615

def P : Set (ℝ × ℝ) := {(x, y) | x + y = 0}
def Q : Set (ℝ × ℝ) := {(x, y) | x - y = 2}

theorem intersection_of_P_and_Q : P ∩ Q = {(1, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2276_227615


namespace NUMINAMATH_CALUDE_white_balls_count_l2276_227601

theorem white_balls_count (n : ℕ) : 
  n = 27 ∧ 
  (∃ (total : ℕ), 
    total = n + 3 ∧ 
    (3 : ℚ) / total = 1 / 10) := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l2276_227601


namespace NUMINAMATH_CALUDE_second_polygon_sides_l2276_227626

theorem second_polygon_sides (p1 p2 : ℕ → ℝ) (n2 : ℕ) :
  (∀ k : ℕ, p1 k = p2 k) →  -- Same perimeter
  (p1 45 = 45 * (3 * p2 n2)) →  -- First polygon has 45 sides and 3 times the side length
  n2 * p2 n2 = p2 n2 * 135 →  -- Perimeter of second polygon
  n2 = 135 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l2276_227626


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_9_l2276_227682

/-- Represents a four-digit number in the form 5BB3 where B is a single digit -/
def fourDigitNumber (B : Nat) : Nat :=
  5000 + 100 * B + 10 * B + 3

/-- Checks if a number is divisible by 9 -/
def isDivisibleBy9 (n : Nat) : Prop :=
  n % 9 = 0

/-- B is a single digit -/
def isSingleDigit (B : Nat) : Prop :=
  B ≥ 0 ∧ B ≤ 9

theorem four_digit_divisible_by_9 :
  ∃ B : Nat, isSingleDigit B ∧ isDivisibleBy9 (fourDigitNumber B) → B = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_9_l2276_227682


namespace NUMINAMATH_CALUDE_smallest_number_l2276_227663

theorem smallest_number (a b c d : ℤ) (ha : a = 0) (hb : b = -3) (hc : c = 1) (hd : d = -1) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l2276_227663


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l2276_227627

/-- The standard equation of an ellipse with specific properties -/
theorem ellipse_standard_equation :
  ∀ (a b : ℝ),
  (a = 2 ∧ b = 1) →
  (∀ (x y : ℝ), (y^2 / 16 + x^2 / 4 = 1) ↔ 
    (y^2 / a^2 + (x - 2)^2 / b^2 = 1 ∧ 
     a > b ∧ 
     a = 2 * b)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l2276_227627


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l2276_227694

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ := sorry

/-- The extended quadrilateral formed by extending the sides of the original quadrilateral -/
noncomputable def extendedQuadrilateral (q : Quadrilateral) : Quadrilateral := sorry

/-- Theorem: The area of the extended quadrilateral is five times the area of the original quadrilateral -/
theorem extended_quadrilateral_area (q : Quadrilateral) :
  area (extendedQuadrilateral q) = 5 * area q := by sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l2276_227694


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2276_227668

/-- The perimeter of a triangle with vertices at (1, 4), (-7, 0), and (1, 0) is equal to 4√5 + 12. -/
theorem triangle_perimeter : 
  let A : ℝ × ℝ := (1, 4)
  let B : ℝ × ℝ := (-7, 0)
  let C : ℝ × ℝ := (1, 0)
  let d₁ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let d₂ := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let d₃ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  d₁ + d₂ + d₃ = 4 * Real.sqrt 5 + 12 := by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l2276_227668


namespace NUMINAMATH_CALUDE_advanced_vowel_soup_sequences_l2276_227602

/-- The number of vowels in the alphabet soup -/
def num_vowels : ℕ := 5

/-- The number of consonants in the alphabet soup -/
def num_consonants : ℕ := 2

/-- The number of times each vowel appears -/
def vowel_occurrences : ℕ := 7

/-- The number of times each consonant appears -/
def consonant_occurrences : ℕ := 3

/-- The length of each sequence -/
def sequence_length : ℕ := 7

/-- The number of valid sequences in the Advanced Vowel Soup -/
theorem advanced_vowel_soup_sequences : 
  (num_vowels + num_consonants)^sequence_length - 
  num_vowels^sequence_length - 
  num_consonants^sequence_length = 745290 := by
  sorry

end NUMINAMATH_CALUDE_advanced_vowel_soup_sequences_l2276_227602


namespace NUMINAMATH_CALUDE_rectangle_max_regions_l2276_227646

/-- The maximum number of regions a rectangle can be divided into with n line segments --/
def max_regions (n : ℕ) : ℕ :=
  if n = 0 then 1
  else max_regions (n - 1) + n

/-- Theorem: A rectangle with 5 line segments can be divided into at most 16 regions --/
theorem rectangle_max_regions :
  max_regions 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_regions_l2276_227646


namespace NUMINAMATH_CALUDE_stream_speed_l2276_227621

/-- 
Given a boat with a speed of 22 km/hr in still water that travels 108 km downstream in 4 hours,
prove that the speed of the stream is 5 km/hr.
-/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 22 →
  distance = 108 →
  time = 4 →
  boat_speed + stream_speed = distance / time →
  stream_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l2276_227621


namespace NUMINAMATH_CALUDE_travelers_getting_off_subway_l2276_227606

/-- The number of stations ahead -/
def num_stations : ℕ := 10

/-- The number of travelers -/
def num_travelers : ℕ := 3

/-- The total number of ways travelers can get off at any station -/
def total_ways : ℕ := num_stations ^ num_travelers

/-- The number of ways all travelers can get off at the same station -/
def same_station_ways : ℕ := num_stations

/-- The number of ways travelers can get off without all disembarking at the same station -/
def different_station_ways : ℕ := total_ways - same_station_ways

theorem travelers_getting_off_subway :
  different_station_ways = 990 := by sorry

end NUMINAMATH_CALUDE_travelers_getting_off_subway_l2276_227606


namespace NUMINAMATH_CALUDE_curve_property_l2276_227642

/-- Given a function f(x) = a*ln(x) + b*x + 1 with specific properties, prove a - b = 10 -/
theorem curve_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * Real.log x + b * x + 1
  (∀ x, HasDerivAt f (a / x + b) x) →
  HasDerivAt f (-2) 1 →
  HasDerivAt f 0 (2/3) →
  a - b = 10 := by
sorry

end NUMINAMATH_CALUDE_curve_property_l2276_227642


namespace NUMINAMATH_CALUDE_no_alpha_exists_for_all_x_l2276_227614

theorem no_alpha_exists_for_all_x (α : ℝ) (h : α > 0) : 
  ∃ x : ℝ, |Real.cos x| + |Real.cos (α * x)| ≤ Real.sin x + Real.sin (α * x) := by
sorry

end NUMINAMATH_CALUDE_no_alpha_exists_for_all_x_l2276_227614


namespace NUMINAMATH_CALUDE_product_equals_sum_implies_x_value_l2276_227640

theorem product_equals_sum_implies_x_value (x : ℝ) : 
  let S : Set ℝ := {3, 6, 9, x}
  (∃ (a b : ℝ), a ∈ S ∧ b ∈ S ∧ (∀ y ∈ S, a ≤ y ∧ y ≤ b) ∧ a * b = (3 + 6 + 9 + x)) →
  x = 9/4 := by
sorry

end NUMINAMATH_CALUDE_product_equals_sum_implies_x_value_l2276_227640


namespace NUMINAMATH_CALUDE_yoongi_subtraction_l2276_227680

theorem yoongi_subtraction (A B C : Nat) (h1 : A ≥ 1) (h2 : A ≤ 9) (h3 : B ≤ 9) (h4 : C ≤ 9) :
  (1000 * A + 100 * B + 10 * C + 6) - 57 = 1819 →
  (1000 * A + 100 * B + 10 * C + 9) - 57 = 1822 := by
sorry

end NUMINAMATH_CALUDE_yoongi_subtraction_l2276_227680


namespace NUMINAMATH_CALUDE_triple_sharp_100_l2276_227629

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.6 * N + 1

-- State the theorem
theorem triple_sharp_100 : sharp (sharp (sharp 100)) = 23.56 := by
  sorry

end NUMINAMATH_CALUDE_triple_sharp_100_l2276_227629


namespace NUMINAMATH_CALUDE_intersection_sum_l2276_227636

theorem intersection_sum (c d : ℝ) : 
  (2 * 4 + c = 6) →  -- First line passes through (4, 6)
  (5 * 4 + d = 6) →  -- Second line passes through (4, 6)
  c + d = -16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2276_227636


namespace NUMINAMATH_CALUDE_ternary_121_equals_16_l2276_227689

def ternary_to_decimal (d₂ d₁ d₀ : ℕ) : ℕ :=
  d₂ * 3^2 + d₁ * 3^1 + d₀ * 3^0

theorem ternary_121_equals_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ternary_121_equals_16_l2276_227689


namespace NUMINAMATH_CALUDE_strawberries_per_jar_solution_l2276_227633

/-- The number of strawberries used in one jar of jam -/
def strawberries_per_jar (betty_strawberries : ℕ) (matthew_extra : ℕ) (jar_price : ℕ) (total_revenue : ℕ) : ℕ :=
  let matthew_strawberries := betty_strawberries + matthew_extra
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let jars_sold := total_revenue / jar_price
  total_strawberries / jars_sold

theorem strawberries_per_jar_solution :
  strawberries_per_jar 16 20 4 40 = 7 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_per_jar_solution_l2276_227633


namespace NUMINAMATH_CALUDE_perpendicular_vectors_t_value_l2276_227608

def a : Fin 2 → ℝ := ![3, 1]
def b : Fin 2 → ℝ := ![1, 3]
def c (t : ℝ) : Fin 2 → ℝ := ![t, 2]

theorem perpendicular_vectors_t_value :
  ∀ t : ℝ, (∀ i : Fin 2, (a i - c t i) * b i = 0) → t = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_t_value_l2276_227608


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2276_227684

theorem quadratic_inequality (x : ℝ) : x^2 - x - 12 < 0 ↔ -3 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2276_227684


namespace NUMINAMATH_CALUDE_alden_nephews_ratio_l2276_227641

-- Define the number of nephews Alden has now
def alden_nephews_now : ℕ := 100

-- Define the number of nephews Alden had 10 years ago
def alden_nephews_10_years_ago : ℕ := 50

-- Define the number of nephews Vihaan has now
def vihaan_nephews : ℕ := alden_nephews_now + 60

-- Theorem stating the ratio of Alden's nephews 10 years ago to now
theorem alden_nephews_ratio : 
  (alden_nephews_10_years_ago : ℚ) / (alden_nephews_now : ℚ) = 1 / 2 :=
by
  -- Assume the total number of nephews is 260
  have total_nephews : alden_nephews_now + vihaan_nephews = 260 := by sorry
  
  -- Prove the ratio
  sorry


end NUMINAMATH_CALUDE_alden_nephews_ratio_l2276_227641


namespace NUMINAMATH_CALUDE_nellie_legos_proof_l2276_227643

/-- Calculates the remaining number of legos after losing some and giving some away. -/
def remaining_legos (initial : ℕ) (lost : ℕ) (given : ℕ) : ℕ :=
  initial - lost - given

/-- Proves that given 380 initial legos, after losing 57 and giving away 24, 299 legos remain. -/
theorem nellie_legos_proof :
  remaining_legos 380 57 24 = 299 := by
  sorry

end NUMINAMATH_CALUDE_nellie_legos_proof_l2276_227643


namespace NUMINAMATH_CALUDE_bills_toilet_paper_duration_l2276_227605

/-- The number of days Bill's toilet paper supply will last -/
def toilet_paper_duration (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ) 
  (total_rolls : ℕ) (squares_per_roll : ℕ) : ℕ :=
  (total_rolls * squares_per_roll) / (bathroom_visits_per_day * squares_per_visit)

/-- Theorem stating that Bill's toilet paper supply will last 20,000 days -/
theorem bills_toilet_paper_duration :
  toilet_paper_duration 3 5 1000 300 = 20000 := by
  sorry

#eval toilet_paper_duration 3 5 1000 300

end NUMINAMATH_CALUDE_bills_toilet_paper_duration_l2276_227605


namespace NUMINAMATH_CALUDE_simplify_expression_l2276_227650

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (1 / (a * b) + 1 / (a * c) + 1 / (b * c)) * (a * b / (a^2 - (b + c)^2)) = 1 / (c * (a - b - c)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2276_227650


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l2276_227620

/-- Given a rectangular metallic sheet with length 48 m, from which squares of side 
    length 6 m are cut from each corner to form an open box, prove that if the 
    volume of the resulting box is 5184 m³, then the width of the original 
    metallic sheet is 36 m. -/
theorem metallic_sheet_width (sheet_length : ℝ) (cut_square_side : ℝ) (box_volume : ℝ) 
  (sheet_width : ℝ) :
  sheet_length = 48 →
  cut_square_side = 6 →
  box_volume = 5184 →
  box_volume = (sheet_length - 2 * cut_square_side) * 
               (sheet_width - 2 * cut_square_side) * 
               cut_square_side →
  sheet_width = 36 :=
by sorry

end NUMINAMATH_CALUDE_metallic_sheet_width_l2276_227620


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l2276_227607

theorem rectangle_measurement_error (L W : ℝ) (p : ℝ) (h_positive : L > 0 ∧ W > 0) :
  let measured_area := (1.05 * L) * (W * (1 - p))
  let actual_area := L * W
  let error_percent := |measured_area - actual_area| / actual_area
  error_percent = 0.008 → p = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l2276_227607


namespace NUMINAMATH_CALUDE_existential_vs_universal_quantifier_l2276_227685

theorem existential_vs_universal_quantifier :
  ¬(∀ (x₀ : ℝ), x₀^2 > 3 ↔ ∃ (x₀ : ℝ), x₀^2 > 3) :=
by sorry

end NUMINAMATH_CALUDE_existential_vs_universal_quantifier_l2276_227685


namespace NUMINAMATH_CALUDE_basketball_spectators_l2276_227673

theorem basketball_spectators 
  (total_spectators : ℕ) 
  (men : ℕ) 
  (ratio_children : ℕ) 
  (ratio_women : ℕ) : 
  total_spectators = 25000 →
  men = 15320 →
  ratio_children = 7 →
  ratio_women = 3 →
  (total_spectators - men) * ratio_children / (ratio_children + ratio_women) = 6776 :=
by sorry

end NUMINAMATH_CALUDE_basketball_spectators_l2276_227673


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2276_227674

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ

/-- The y-intercept of a line tangent to two specific circles -/
def yIntercept (line : TangentLine) : ℝ :=
  sorry

/-- The main theorem stating the y-intercept of the tangent line -/
theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (6, 0), radius := 1 }
  ∀ (line : TangentLine),
    line.circle1 = c1 →
    line.circle2 = c2 →
    line.tangentPoint1.1 > 3 →
    line.tangentPoint1.2 > 0 →
    line.tangentPoint2.1 > 6 →
    line.tangentPoint2.2 > 0 →
    yIntercept line = 6 * Real.sqrt 2 :=
  by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2276_227674


namespace NUMINAMATH_CALUDE_complex_distance_theorem_l2276_227622

theorem complex_distance_theorem : ∃ (c : ℂ) (d : ℝ), c ≠ 0 ∧ 
  ∀ (z : ℂ), Complex.abs z = 1 → 1 + z + z^2 ≠ 0 → 
    Complex.abs (1 / (1 + z + z^2)) - Complex.abs (1 / (1 + z + z^2) - c) = d :=
by sorry

end NUMINAMATH_CALUDE_complex_distance_theorem_l2276_227622


namespace NUMINAMATH_CALUDE_apple_purchase_theorem_l2276_227661

/-- The cost of apples with a two-tier pricing system -/
def apple_cost (l q : ℚ) (x : ℚ) : ℚ :=
  if x ≤ 30 then l * x
  else l * 30 + q * (x - 30)

theorem apple_purchase_theorem (l q : ℚ) :
  (∀ x, x ≤ 30 → apple_cost l q x = l * x) ∧
  (∀ x, x > 30 → apple_cost l q x = l * 30 + q * (x - 30)) ∧
  (apple_cost l q 36 = 366) ∧
  (apple_cost l q 15 = 150) ∧
  (∃ x, apple_cost l q x = 333) →
  ∃ x, apple_cost l q x = 333 ∧ x = 33 :=
by sorry

end NUMINAMATH_CALUDE_apple_purchase_theorem_l2276_227661


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l2276_227656

theorem least_integer_absolute_value (x : ℤ) : (∀ y : ℤ, y < x → |3 * y + 10| > 25) ∧ |3 * x + 10| ≤ 25 ↔ x = -11 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l2276_227656


namespace NUMINAMATH_CALUDE_min_value_theorem_l2276_227604

theorem min_value_theorem (x : ℝ) (h : x > 0) : x + 16 / (x + 1) ≥ 7 ∧ ∃ y > 0, y + 16 / (y + 1) = 7 :=
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2276_227604


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2276_227687

/-- An arithmetic sequence with a common difference of 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y / x = z / y

/-- The main theorem -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2276_227687


namespace NUMINAMATH_CALUDE_tan_forty_five_degrees_equals_one_l2276_227630

theorem tan_forty_five_degrees_equals_one : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_forty_five_degrees_equals_one_l2276_227630


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l2276_227631

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l2276_227631


namespace NUMINAMATH_CALUDE_triangle_area_implies_p_value_l2276_227610

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    if the area of the triangle is 36, then p = 12.75 -/
theorem triangle_area_implies_p_value :
  ∀ (p : ℝ),
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  triangle_area = 36 → p = 12.75 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_implies_p_value_l2276_227610


namespace NUMINAMATH_CALUDE_eldest_boy_age_l2276_227683

theorem eldest_boy_age (boys : Fin 3 → ℕ) 
  (avg_age : (boys 0 + boys 1 + boys 2) / 3 = 15)
  (proportion : ∃ (x : ℕ), boys 0 = 3 * x ∧ boys 1 = 5 * x ∧ boys 2 = 7 * x) :
  boys 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eldest_boy_age_l2276_227683


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l2276_227613

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge_length := 1.6 * L
  let new_area := 6 * new_edge_length^2
  (new_area - original_area) / original_area * 100 = 156 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l2276_227613


namespace NUMINAMATH_CALUDE_luke_candy_purchase_l2276_227634

/-- The number of candy pieces Luke can buy given his tickets and candy cost -/
def candyPieces (whackAMoleTickets skeeBallTickets candyCost : ℕ) : ℕ :=
  (whackAMoleTickets + skeeBallTickets) / candyCost

/-- Proof that Luke can buy 5 pieces of candy -/
theorem luke_candy_purchase :
  candyPieces 2 13 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_luke_candy_purchase_l2276_227634


namespace NUMINAMATH_CALUDE_sum_of_differences_mod_1000_l2276_227616

def S : Finset ℕ := Finset.range 11

def pairDifference (i j : ℕ) : ℕ := 
  if i < j then 2^j - 2^i else 2^i - 2^j

def N : ℕ := Finset.sum (S.product S) (fun (p : ℕ × ℕ) => pairDifference p.1 p.2)

theorem sum_of_differences_mod_1000 : N % 1000 = 304 := by sorry

end NUMINAMATH_CALUDE_sum_of_differences_mod_1000_l2276_227616


namespace NUMINAMATH_CALUDE_remainder_3042_div_29_l2276_227649

theorem remainder_3042_div_29 : 3042 % 29 = 26 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3042_div_29_l2276_227649


namespace NUMINAMATH_CALUDE_fair_coin_five_tosses_l2276_227638

/-- The probability of getting heads in a single toss of a fair coin -/
def p_heads : ℚ := 1/2

/-- The number of tosses -/
def n : ℕ := 5

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def prob_exactly (k : ℕ) : ℚ :=
  ↑(n.choose k) * p_heads ^ k * (1 - p_heads) ^ (n - k)

/-- The probability of getting at least 2 heads in 5 tosses of a fair coin -/
def prob_at_least_two : ℚ :=
  1 - prob_exactly 0 - prob_exactly 1

theorem fair_coin_five_tosses :
  prob_at_least_two = 13/16 := by sorry

end NUMINAMATH_CALUDE_fair_coin_five_tosses_l2276_227638


namespace NUMINAMATH_CALUDE_total_triangles_l2276_227678

/-- Represents a triangle divided into smaller triangles -/
structure DividedTriangle where
  small_triangles : ℕ

/-- Counts the total number of triangles in a divided triangle -/
def count_triangles (t : DividedTriangle) : ℕ :=
  t.small_triangles + (t.small_triangles - 1) + 1

/-- The problem setup -/
def triangle_problem : Prop :=
  ∃ (t1 t2 : DividedTriangle),
    t1.small_triangles = 3 ∧
    t2.small_triangles = 3 ∧
    count_triangles t1 + count_triangles t2 = 13

/-- The theorem to prove -/
theorem total_triangles : triangle_problem := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_l2276_227678


namespace NUMINAMATH_CALUDE_class_grouping_l2276_227670

/-- Given a class where students can form 8 pairs when grouped in twos,
    prove that the number of groups formed when students are grouped in fours is 4. -/
theorem class_grouping (num_pairs : ℕ) (h : num_pairs = 8) :
  (2 * num_pairs) / 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_class_grouping_l2276_227670


namespace NUMINAMATH_CALUDE_sum_1_to_15_mod_11_l2276_227624

theorem sum_1_to_15_mod_11 : (List.range 15).sum % 11 = 10 := by sorry

end NUMINAMATH_CALUDE_sum_1_to_15_mod_11_l2276_227624


namespace NUMINAMATH_CALUDE_inequality_proof_l2276_227669

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b + b / c + c / a)^2 ≥ (3/2) * ((a + b) / c + (b + c) / a + (c + a) / b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2276_227669


namespace NUMINAMATH_CALUDE_gcd_5280_12155_l2276_227652

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 55 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5280_12155_l2276_227652


namespace NUMINAMATH_CALUDE_triangle_area_l2276_227662

def line1 (x : ℝ) : ℝ := x
def line2 (x : ℝ) : ℝ := -2 * x + 3

theorem triangle_area : 
  let x_intercept := (3 : ℝ) / 2
  let intersection_x := (1 : ℝ)
  let intersection_y := line1 intersection_x
  let base := x_intercept
  let height := intersection_y
  (1 / 2) * base * height = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2276_227662


namespace NUMINAMATH_CALUDE_point_N_coordinates_l2276_227637

-- Define the point M and vector a
def M : ℝ × ℝ := (5, -6)
def a : ℝ × ℝ := (1, -2)

-- Define the relation between MN and a
def MN_relation (N : ℝ × ℝ) : Prop :=
  (N.1 - M.1, N.2 - M.2) = (-3 * a.1, -3 * a.2)

-- Theorem statement
theorem point_N_coordinates :
  ∃ N : ℝ × ℝ, MN_relation N ∧ N = (2, 0) := by sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l2276_227637


namespace NUMINAMATH_CALUDE_min_value_when_k_is_one_l2276_227654

/-- The function for which we want to find the minimum value -/
def f (x k : ℝ) : ℝ := x^2 - (2*k + 3)*x + 2*k^2 - k - 3

/-- The theorem stating the minimum value of the function when k = 1 -/
theorem min_value_when_k_is_one :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x 1 ≥ f x_min 1 ∧ f x_min 1 = -33/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_when_k_is_one_l2276_227654


namespace NUMINAMATH_CALUDE_thompson_children_ages_l2276_227691

/-- Represents the ages of Miss Thompson's children -/
def ChildrenAges : Type := Fin 5 → Nat

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  t_range : tens ≥ 0 ∧ tens ≤ 9
  o_range : ones ≥ 0 ∧ ones ≤ 9
  different : hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones

theorem thompson_children_ages
  (ages : ChildrenAges)
  (number : ThreeDigitNumber)
  (h_oldest : ages 0 = 11)
  (h_middle : ages 2 = 7)
  (h_different : ∀ i j, i ≠ j → ages i ≠ ages j)
  (h_divisible_oldest : (number.hundreds * 100 + number.tens * 10 + number.ones) % 11 = 0)
  (h_divisible_middle : (number.hundreds * 100 + number.tens * 10 + number.ones) % 7 = 0)
  (h_youngest : ∃ i, ages i = number.ones)
  : ¬(∃ i, ages i = 6) :=
by sorry

end NUMINAMATH_CALUDE_thompson_children_ages_l2276_227691


namespace NUMINAMATH_CALUDE_chairs_per_trip_l2276_227699

theorem chairs_per_trip 
  (num_students : ℕ) 
  (trips_per_student : ℕ) 
  (total_chairs : ℕ) 
  (h1 : num_students = 5) 
  (h2 : trips_per_student = 10) 
  (h3 : total_chairs = 250) : 
  (total_chairs / (num_students * trips_per_student) : ℚ) = 5 := by
sorry

end NUMINAMATH_CALUDE_chairs_per_trip_l2276_227699


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2276_227619

theorem interest_rate_calculation (P t : ℝ) (diff : ℝ) : 
  P = 10000 → 
  t = 2 → 
  diff = 49 → 
  P * (1 + 7/100)^t - P - (P * 7 * t / 100) = diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2276_227619


namespace NUMINAMATH_CALUDE_vacation_cost_sharing_l2276_227672

theorem vacation_cost_sharing (john_paid mary_paid lisa_paid : ℝ) (j m : ℝ) : 
  john_paid = 150 →
  mary_paid = 90 →
  lisa_paid = 210 →
  j = 150 - john_paid →
  m = 150 - mary_paid →
  j - m = -60 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_sharing_l2276_227672


namespace NUMINAMATH_CALUDE_increase_decrease_threshold_l2276_227617

theorem increase_decrease_threshold (S x y : ℝ) 
  (hS : S > 0) (hxy : x > y) (hy : y > 0) : 
  ((S * (1 + x/100) + 15) * (1 - y/100) > S + 10) ↔ 
  (x > y + (x*y/100) + 500 - (1500*y/S)) :=
sorry

end NUMINAMATH_CALUDE_increase_decrease_threshold_l2276_227617


namespace NUMINAMATH_CALUDE_hyperbola_condition_l2276_227695

/-- Defines a hyperbola in terms of its equation -/
def is_hyperbola (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), m * x^2 + n * y^2 = 1 ∧ 
  ∀ (a b : ℝ), a^2 / (1/m) - b^2 / (1/n) = 1 ∨ a^2 / (1/n) - b^2 / (1/m) = 1

/-- Theorem stating that mn < 0 is a necessary and sufficient condition for mx^2 + ny^2 = 1 to represent a hyperbola -/
theorem hyperbola_condition (m n : ℝ) :
  is_hyperbola m n ↔ m * n < 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l2276_227695


namespace NUMINAMATH_CALUDE_problem_proof_l2276_227618

theorem problem_proof : (-24) * (1/3 - 5/6 + 3/8) = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l2276_227618


namespace NUMINAMATH_CALUDE_subset_implies_x_value_l2276_227692

theorem subset_implies_x_value (A B : Set ℝ) (x : ℝ) : 
  A = {-2, 1} → 
  B = {0, 1, x + 1} → 
  A ⊆ B → 
  x = -3 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_x_value_l2276_227692


namespace NUMINAMATH_CALUDE_cakes_sold_l2276_227648

theorem cakes_sold (initial_cakes : ℕ) (additional_cakes : ℕ) (remaining_cakes : ℕ) :
  initial_cakes = 62 →
  additional_cakes = 149 →
  remaining_cakes = 67 →
  initial_cakes + additional_cakes - remaining_cakes = 144 :=
by sorry

end NUMINAMATH_CALUDE_cakes_sold_l2276_227648


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2276_227664

theorem trigonometric_identities (α : Real) 
  (h1 : 3 * Real.pi / 4 < α) 
  (h2 : α < Real.pi) 
  (h3 : Real.tan α + 1 / Real.tan α = -10 / 3) : 
  (Real.tan α = -1 / 3) ∧ 
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -1 / 2) ∧ 
  (2 * Real.sin α ^ 2 - Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = -11 / 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2276_227664


namespace NUMINAMATH_CALUDE_sin_cos_sum_formula_l2276_227693

theorem sin_cos_sum_formula (α β : ℝ) : 
  Real.sin α * Real.sin β - Real.cos α * Real.cos β = - Real.cos (α + β) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_formula_l2276_227693


namespace NUMINAMATH_CALUDE_recipe_total_cups_l2276_227671

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of flour -/
def totalIngredients (ratio : RecipeRatio) (flourCups : ℕ) : ℕ :=
  let partSize := flourCups / ratio.flour
  (ratio.butter + ratio.flour + ratio.sugar) * partSize

/-- Theorem: Given a recipe with ratio 2:5:3 and 10 cups of flour, the total ingredients is 20 cups -/
theorem recipe_total_cups (ratio : RecipeRatio) (h1 : ratio.butter = 2) (h2 : ratio.flour = 5) (h3 : ratio.sugar = 3) :
  totalIngredients ratio 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l2276_227671


namespace NUMINAMATH_CALUDE_distance_to_walk_back_l2276_227690

/-- Represents the distance traveled by Vintik and Shpuntik -/
def TravelDistance (x y : ℝ) : Prop :=
  -- Vintik's total distance is 12 km
  2 * x + y = 12 ∧
  -- Total fuel consumption is 75 liters
  3 * x + 15 * y = 75 ∧
  -- x represents half of Vintik's forward distance
  x > 0 ∧ y > 0

/-- The theorem stating the distance to walk back home -/
theorem distance_to_walk_back (x y : ℝ) (h : TravelDistance x y) : 
  3 * x - 3 * y = 9 :=
sorry

end NUMINAMATH_CALUDE_distance_to_walk_back_l2276_227690


namespace NUMINAMATH_CALUDE_thursday_withdrawal_l2276_227603

/-- Calculates the number of books withdrawn on Thursday given the initial number of books,
    the number of books taken out on Tuesday, the number of books returned on Wednesday,
    and the final number of books in the library. -/
def books_withdrawn_thursday (initial : ℕ) (taken_tuesday : ℕ) (returned_wednesday : ℕ) (final : ℕ) : ℕ :=
  initial - taken_tuesday + returned_wednesday - final

/-- Proves that the number of books withdrawn on Thursday is 15, given the specific values
    from the problem. -/
theorem thursday_withdrawal : books_withdrawn_thursday 250 120 35 150 = 15 := by
  sorry

end NUMINAMATH_CALUDE_thursday_withdrawal_l2276_227603


namespace NUMINAMATH_CALUDE_division_problem_l2276_227653

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2276_227653


namespace NUMINAMATH_CALUDE_wire_service_reporters_l2276_227660

/-- The percentage of reporters who cover local politics in country x -/
def local_politics_percentage : ℝ := 35

/-- The percentage of reporters who cover politics but not local politics in country x -/
def non_local_politics_percentage : ℝ := 30

/-- The percentage of reporters who do not cover politics -/
def non_politics_percentage : ℝ := 50

theorem wire_service_reporters :
  local_politics_percentage = 35 →
  non_local_politics_percentage = 30 →
  non_politics_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l2276_227660


namespace NUMINAMATH_CALUDE_service_center_location_example_highway_valid_l2276_227628

/-- Represents a highway with exits and a service center -/
structure Highway where
  fourth_exit : ℝ
  ninth_exit : ℝ
  service_center : ℝ

/-- The service center is halfway between the fourth and ninth exits -/
def is_halfway (h : Highway) : Prop :=
  h.service_center = (h.fourth_exit + h.ninth_exit) / 2

/-- Theorem: Given the conditions, the service center is at milepost 90 -/
theorem service_center_location (h : Highway)
  (h_fourth : h.fourth_exit = 30)
  (h_ninth : h.ninth_exit = 150)
  (h_halfway : is_halfway h) :
  h.service_center = 90 := by
  sorry

/-- Example highway satisfying the conditions -/
def example_highway : Highway :=
  { fourth_exit := 30
  , ninth_exit := 150
  , service_center := 90 }

/-- The example highway satisfies all conditions -/
theorem example_highway_valid :
  is_halfway example_highway ∧
  example_highway.fourth_exit = 30 ∧
  example_highway.ninth_exit = 150 ∧
  example_highway.service_center = 90 := by
  sorry

end NUMINAMATH_CALUDE_service_center_location_example_highway_valid_l2276_227628


namespace NUMINAMATH_CALUDE_sara_meets_bus_probability_l2276_227612

/-- Represents the time in minutes after 3:30 pm -/
def TimeAfter330 := { t : ℝ // 0 ≤ t ∧ t ≤ 60 }

/-- The bus arrives at a random time between 3:30 pm and 4:30 pm -/
def bus_arrival : TimeAfter330 := sorry

/-- Sara arrives at a random time between 3:30 pm and 4:30 pm -/
def sara_arrival : TimeAfter330 := sorry

/-- The bus waits for 40 minutes after arrival -/
def bus_wait_time : ℝ := 40

/-- The probability that Sara arrives while the bus is still waiting -/
def probability_sara_meets_bus : ℝ := sorry

theorem sara_meets_bus_probability :
  probability_sara_meets_bus = 2/3 := by sorry

end NUMINAMATH_CALUDE_sara_meets_bus_probability_l2276_227612


namespace NUMINAMATH_CALUDE_parabola_equation_l2276_227635

/-- A parabola is defined by its directrix. -/
structure Parabola where
  directrix : ℝ

/-- The standard equation of a parabola. -/
def standard_equation (p : Parabola) : Prop :=
  ∀ x y : ℝ, y^2 = 28 * x

/-- Theorem: If the directrix of a parabola is x = -7, then its standard equation is y² = 28x. -/
theorem parabola_equation (p : Parabola) (h : p.directrix = -7) : standard_equation p := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2276_227635


namespace NUMINAMATH_CALUDE_circumscribed_radius_of_specific_trapezoid_l2276_227675

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  lateral : ℝ

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumscribedRadius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the radius of the circumscribed circle of the given isosceles trapezoid is 5√2 -/
theorem circumscribed_radius_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := { base1 := 2, base2 := 14, lateral := 10 }
  circumscribedRadius t = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_radius_of_specific_trapezoid_l2276_227675


namespace NUMINAMATH_CALUDE_min_pencils_for_ten_correct_l2276_227677

/-- Represents the number of pencils of each color in the drawer -/
structure PencilDrawer :=
  (orange : ℕ)
  (purple : ℕ)
  (grey : ℕ)
  (cyan : ℕ)
  (violet : ℕ)

/-- The minimum number of pencils to ensure at least 10 of one color -/
def minPencilsForTen (drawer : PencilDrawer) : ℕ := 43

/-- Theorem stating the minimum number of pencils needed -/
theorem min_pencils_for_ten_correct (drawer : PencilDrawer) 
  (h1 : drawer.orange = 26)
  (h2 : drawer.purple = 22)
  (h3 : drawer.grey = 18)
  (h4 : drawer.cyan = 15)
  (h5 : drawer.violet = 10) :
  minPencilsForTen drawer = 43 ∧
  ∀ n : ℕ, n < 43 → ¬(∃ color : ℕ, color ≥ 10 ∧ 
    (color ≤ drawer.orange ∨ 
     color ≤ drawer.purple ∨ 
     color ≤ drawer.grey ∨ 
     color ≤ drawer.cyan ∨ 
     color ≤ drawer.violet)) := by
  sorry

end NUMINAMATH_CALUDE_min_pencils_for_ten_correct_l2276_227677


namespace NUMINAMATH_CALUDE_smallest_matching_end_digits_correct_l2276_227623

/-- The smallest positive integer M such that M and M^2 + 1 end in the same sequence of four digits in base 10, where the first digit of the four is not zero. -/
def smallest_matching_end_digits : ℕ := 3125

/-- Predicate to check if a number ends with the same four digits as its square plus one. -/
def ends_with_same_four_digits (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≡ n^2 + 1 [ZMOD 10000]

theorem smallest_matching_end_digits_correct :
  ends_with_same_four_digits smallest_matching_end_digits ∧
  ∀ m : ℕ, m < smallest_matching_end_digits → ¬ends_with_same_four_digits m := by
  sorry

end NUMINAMATH_CALUDE_smallest_matching_end_digits_correct_l2276_227623


namespace NUMINAMATH_CALUDE_sibling_age_sum_l2276_227645

/-- Given the ages of three siblings, proves that the sum of two siblings' ages is correct. -/
theorem sibling_age_sum (juliet maggie ralph : ℕ) : 
  juliet = maggie + 3 →
  juliet + 2 = ralph →
  juliet = 10 →
  maggie + ralph = 19 := by
sorry

end NUMINAMATH_CALUDE_sibling_age_sum_l2276_227645


namespace NUMINAMATH_CALUDE_square_difference_l2276_227655

theorem square_difference : (30 : ℕ)^2 - (29 : ℕ)^2 = 59 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2276_227655


namespace NUMINAMATH_CALUDE_perfect_square_condition_solution_uniqueness_l2276_227666

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def repeated_digits (x y : ℕ) (n : ℕ) : ℕ :=
  x * 10^(2*n) + 6 * 10^n + y

theorem perfect_square_condition (x y : ℕ) : Prop :=
  x ≠ 0 ∧ ∀ n : ℕ, n ≥ 1 → is_perfect_square (repeated_digits x y n)

theorem solution_uniqueness :
  ∀ x y : ℕ, perfect_square_condition x y →
    ((x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_solution_uniqueness_l2276_227666


namespace NUMINAMATH_CALUDE_sin_is_2_type_function_x_plus_cos_is_2_type_function_l2276_227698

-- Define what it means for a function to be a t-type function
def is_t_type_function (f : ℝ → ℝ) (t : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (deriv f x₁) + (deriv f x₂) = t

-- State the theorem for sin x
theorem sin_is_2_type_function :
  is_t_type_function Real.sin 2 :=
sorry

-- State the theorem for x + cos x
theorem x_plus_cos_is_2_type_function :
  is_t_type_function (fun x => x + Real.cos x) 2 :=
sorry

end NUMINAMATH_CALUDE_sin_is_2_type_function_x_plus_cos_is_2_type_function_l2276_227698


namespace NUMINAMATH_CALUDE_shoe_store_sale_l2276_227657

theorem shoe_store_sale (sneakers sandals boots : ℕ) 
  (h1 : sneakers = 2) 
  (h2 : sandals = 4) 
  (h3 : boots = 11) : 
  sneakers + sandals + boots = 17 := by
  sorry

end NUMINAMATH_CALUDE_shoe_store_sale_l2276_227657
