import Mathlib

namespace third_year_sample_size_l3907_390797

/-- Represents the number of students to be selected in a stratified sampling -/
def sample_size : ℕ := 200

/-- Represents the total number of first-year students -/
def first_year_students : ℕ := 700

/-- Represents the total number of second-year students -/
def second_year_students : ℕ := 670

/-- Represents the total number of third-year students -/
def third_year_students : ℕ := 630

/-- Represents the total number of students in all three years -/
def total_students : ℕ := first_year_students + second_year_students + third_year_students

/-- Theorem stating that the number of third-year students to be selected in the stratified sampling is 63 -/
theorem third_year_sample_size :
  (sample_size * third_year_students) / total_students = 63 := by
  sorry

end third_year_sample_size_l3907_390797


namespace greatest_base6_digit_sum_l3907_390761

/-- Represents a base-6 digit -/
def Base6Digit := Fin 6

/-- Converts a natural number to its base-6 representation -/
def toBase6 (n : ℕ) : List Base6Digit :=
  sorry

/-- Calculates the sum of digits in a list -/
def digitSum (digits : List Base6Digit) : ℕ :=
  sorry

/-- Theorem: The greatest possible sum of digits in the base-6 representation
    of a positive integer less than 1728 is 20 -/
theorem greatest_base6_digit_sum :
  ∃ (n : ℕ), n > 0 ∧ n < 1728 ∧
  digitSum (toBase6 n) = 20 ∧
  ∀ (m : ℕ), m > 0 → m < 1728 → digitSum (toBase6 m) ≤ 20 :=
sorry

end greatest_base6_digit_sum_l3907_390761


namespace california_new_york_ratio_l3907_390732

/-- Proves that the ratio of Coronavirus cases in California to New York is 1:2 --/
theorem california_new_york_ratio : 
  ∀ (california texas : ℕ), 
  california = texas + 400 →
  2000 + california + texas = 3600 →
  california * 2 = 2000 :=
by
  sorry

end california_new_york_ratio_l3907_390732


namespace line_intercepts_sum_zero_l3907_390774

/-- Given a line l with equation 2x+(k-3)y-2k+6=0 where k ≠ 3,
    if the sum of its x-intercept and y-intercept is 0, then k = 1 -/
theorem line_intercepts_sum_zero (k : ℝ) (h1 : k ≠ 3) :
  (∃ x y : ℝ, 2*x + (k-3)*y - 2*k + 6 = 0) →
  (∃ x_int y_int : ℝ,
    (2*x_int - 2*k + 6 = 0) ∧
    ((k-3)*y_int - 2*k + 6 = 0) ∧
    (x_int + y_int = 0)) →
  k = 1 := by
sorry

end line_intercepts_sum_zero_l3907_390774


namespace least_four_digit_multiple_l3907_390749

/-- The least 4-digit number divisible by 15, 25, 40, and 75 is 1200 -/
theorem least_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  15 ∣ n ∧ 25 ∣ n ∧ 40 ∣ n ∧ 75 ∣ n ∧
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) ∧ 15 ∣ m ∧ 25 ∣ m ∧ 40 ∣ m ∧ 75 ∣ m → m ≥ n) ∧
  n = 1200 :=
by sorry

end least_four_digit_multiple_l3907_390749


namespace pedros_test_scores_l3907_390763

theorem pedros_test_scores :
  let scores : List ℕ := [92, 91, 89, 85, 78]
  let first_three : List ℕ := [92, 85, 78]
  ∀ (s : List ℕ),
    s.length = 5 →
    s.take 3 = first_three →
    s.sum / s.length = 87 →
    (∀ x ∈ s, x < 100) →
    s.Nodup →
    s = scores :=
by sorry

end pedros_test_scores_l3907_390763


namespace palindrome_with_seven_percentage_l3907_390726

-- Define a palindrome in the range 100 to 999
def IsPalindrome (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

-- Define a number containing at least one 7
def ContainsSeven (n : Nat) : Prop :=
  (n / 100 = 7) ∨ ((n / 10) % 10 = 7) ∨ (n % 10 = 7)

-- Count of palindromes with at least one 7
def PalindromeWithSeven : Nat :=
  19

-- Total count of palindromes between 100 and 999
def TotalPalindromes : Nat :=
  90

-- Theorem statement
theorem palindrome_with_seven_percentage :
  (PalindromeWithSeven : ℚ) / TotalPalindromes = 19 / 90 := by
  sorry

end palindrome_with_seven_percentage_l3907_390726


namespace smallest_n_divisibility_l3907_390712

theorem smallest_n_divisibility (x y z : ℕ+) 
  (h1 : x ∣ y^3) 
  (h2 : y ∣ z^3) 
  (h3 : z ∣ x^3) : 
  (∀ n : ℕ, n ≥ 13 → (x*y*z : ℕ) ∣ (x + y + z : ℕ)^n) ∧ 
  (∀ m : ℕ, m < 13 → ∃ a b c : ℕ+, 
    a ∣ b^3 ∧ b ∣ c^3 ∧ c ∣ a^3 ∧ ¬((a*b*c : ℕ) ∣ (a + b + c : ℕ)^m)) :=
sorry

end smallest_n_divisibility_l3907_390712


namespace sum_in_arithmetic_sequence_l3907_390759

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_in_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 + a 7 = 37 →
  a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end sum_in_arithmetic_sequence_l3907_390759


namespace sugar_price_increase_l3907_390730

theorem sugar_price_increase (initial_price : ℝ) (initial_quantity : ℝ) : 
  initial_quantity > 0 →
  initial_price > 0 →
  initial_price * initial_quantity = 5 * (0.4 * initial_quantity) →
  initial_price = 2 := by
sorry

end sugar_price_increase_l3907_390730


namespace number_of_divisors_of_90_l3907_390704

theorem number_of_divisors_of_90 : Finset.card (Nat.divisors 90) = 12 := by
  sorry

end number_of_divisors_of_90_l3907_390704


namespace sachin_age_l3907_390768

/-- Represents the ages of Sachin, Rahul, and Praveen -/
structure Ages where
  sachin : ℝ
  rahul : ℝ
  praveen : ℝ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.rahul = ages.sachin + 7 ∧
  ages.sachin / ages.rahul = 7 / 9 ∧
  ages.praveen = 2 * ages.rahul ∧
  ages.sachin / ages.rahul = 7 / 9 ∧
  ages.rahul / ages.praveen = 9 / 18

/-- Theorem stating that if the ages satisfy the conditions, then Sachin's age is 24.5 -/
theorem sachin_age (ages : Ages) : 
  satisfiesConditions ages → ages.sachin = 24.5 := by
  sorry

end sachin_age_l3907_390768


namespace female_student_count_l3907_390702

theorem female_student_count (total_students : ℕ) (selection_ways : ℕ) :
  total_students = 8 →
  selection_ways = 30 →
  (∃ (male_students : ℕ) (female_students : ℕ),
    male_students + female_students = total_students ∧
    (male_students.choose 2) * female_students = selection_ways ∧
    (female_students = 2 ∨ female_students = 3)) :=
by sorry

end female_student_count_l3907_390702


namespace cubic_root_sum_l3907_390707

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 11*p - 3 = 0 →
  q^3 - 8*q^2 + 11*q - 3 = 0 →
  r^3 - 8*r^2 + 11*r - 3 = 0 →
  p/(q*r + 1) + q/(p*r + 1) + r/(p*q + 1) = 32/15 := by
sorry

end cubic_root_sum_l3907_390707


namespace smallest_sum_of_coefficients_l3907_390757

theorem smallest_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + 2*a*x + 3*b = 0) →
  (∃ y : ℝ, y^2 + 3*b*y + 2*a = 0) →
  a + b ≥ 5 := by
sorry

end smallest_sum_of_coefficients_l3907_390757


namespace pencil_distribution_l3907_390735

/-- The number of ways to distribute n identical objects among k people,
    where each person gets at least one object. -/
def distributionWays (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 identical pencils among 3 friends,
    where each friend gets at least one pencil. -/
theorem pencil_distribution : distributionWays 6 3 = 10 := by sorry

end pencil_distribution_l3907_390735


namespace function_inequality_condition_l3907_390739

open Real

theorem function_inequality_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ →
    (a * exp x₁ / x₁ - x₁) / x₂ - (a * exp x₂ / x₂ - x₂) / x₁ < 0) ↔
  a ≥ -exp 1 :=
sorry

end function_inequality_condition_l3907_390739


namespace intersection_length_tangent_line_m_range_l3907_390721

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0

-- Define circle C
def circle_C (a x y : ℝ) : Prop := x^2 + y^2 + 2*a*x - 2*a*y + 2*a^2 - 4*a = 0

-- Theorem for part 1
theorem intersection_length :
  ∀ (x1 y1 x2 y2 : ℝ),
  circle_O x1 y1 ∧ circle_O x2 y2 ∧
  circle_C 3 x1 y1 ∧ circle_C 3 x2 y2 →
  ((x1 - x2)^2 + (y1 - y2)^2)^(1/2) = Real.sqrt 94 / 3 := by sorry

-- Theorem for part 2
theorem tangent_line_m_range :
  ∀ (a m : ℝ),
  0 < a ∧ a ≤ 4 ∧
  (∃ (x y : ℝ), line_l m x y ∧ circle_C a x y) ∧
  (∀ (x y : ℝ), line_l m x y → (x + a)^2 + (y - a)^2 ≥ 4*a) →
  -1 ≤ m ∧ m ≤ 8 - 4*Real.sqrt 2 := by sorry

end intersection_length_tangent_line_m_range_l3907_390721


namespace juliet_age_l3907_390742

theorem juliet_age (maggie ralph juliet : ℕ) 
  (h1 : juliet = maggie + 3)
  (h2 : juliet = ralph - 2)
  (h3 : maggie + ralph = 19) :
  juliet = 10 := by
  sorry

end juliet_age_l3907_390742


namespace batsman_average_after_15th_innings_l3907_390755

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored) / (b.innings + 1)

theorem batsman_average_after_15th_innings 
  (b : Batsman)
  (h1 : b.innings = 14)
  (h2 : newAverage b 85 = b.average + 3) :
  newAverage b 85 = 43 := by
  sorry

#check batsman_average_after_15th_innings

end batsman_average_after_15th_innings_l3907_390755


namespace sum_of_a_and_t_is_71_l3907_390734

/-- Given a natural number n, this function represents the equation
    √(n+1 + (n+1)/((n+1)²-1)) = (n+1)√((n+1)/((n+1)²-1)) -/
def equation_pattern (n : ℕ) : Prop :=
  Real.sqrt ((n + 1 : ℝ) + (n + 1) / ((n + 1)^2 - 1)) = (n + 1 : ℝ) * Real.sqrt ((n + 1) / ((n + 1)^2 - 1))

/-- The main theorem stating that given the pattern for n = 1 to 7,
    the sum of a and t in the equation √(8 + a/t) = 8√(a/t) is 71 -/
theorem sum_of_a_and_t_is_71 
  (h1 : equation_pattern 1)
  (h2 : equation_pattern 2)
  (h3 : equation_pattern 3)
  (h4 : equation_pattern 4)
  (h5 : equation_pattern 5)
  (h6 : equation_pattern 6)
  (h7 : equation_pattern 7)
  (a t : ℝ)
  (ha : a > 0)
  (ht : t > 0)
  (h : Real.sqrt (8 + a/t) = 8 * Real.sqrt (a/t)) :
  a + t = 71 := by
  sorry

end sum_of_a_and_t_is_71_l3907_390734


namespace complex_expression_equals_negative_48_l3907_390789

theorem complex_expression_equals_negative_48 : 
  ((-1/2 * (1/100))^5 * (2/3 * (2/100))^4 * (-3/4 * (3/100))^3 * (4/5 * (4/100))^2 * (-5/6 * (5/100))) * (10^30) = -48 := by
  sorry

end complex_expression_equals_negative_48_l3907_390789


namespace prob_defective_bulb_selection_l3907_390700

/-- Given a box of electric bulbs, this function calculates the probability of
    selecting at least one defective bulb when choosing two bulbs at random. -/
def prob_at_least_one_defective (total : ℕ) (defective : ℕ) : ℚ :=
  1 - (total - defective : ℚ) / total * ((total - defective - 1) : ℚ) / (total - 1)

/-- Theorem stating that for a box with 24 bulbs, 4 of which are defective,
    the probability of choosing at least one defective bulb when randomly
    selecting two bulbs is equal to 43/138. -/
theorem prob_defective_bulb_selection :
  prob_at_least_one_defective 24 4 = 43 / 138 := by
  sorry

end prob_defective_bulb_selection_l3907_390700


namespace two_quadrilaterals_nine_regions_l3907_390723

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral in the plane -/
structure Quadrilateral :=
  (p1 p2 p3 p4 : Point)

/-- The plane divided by quadrilaterals -/
def PlaneDivision :=
  List Quadrilateral

/-- Count the number of regions in a plane division -/
def countRegions (division : PlaneDivision) : ℕ :=
  sorry

/-- Theorem: There exists a plane division with two quadrilaterals that results in 9 regions -/
theorem two_quadrilaterals_nine_regions :
  ∃ (division : PlaneDivision),
    division.length = 2 ∧ countRegions division = 9 :=
  sorry

end two_quadrilaterals_nine_regions_l3907_390723


namespace solve_equation_l3907_390748

theorem solve_equation (t x : ℝ) (h1 : (5 + x) / (t + x) = 2 / 3) (h2 : t = 13) : x = 11 := by
  sorry

end solve_equation_l3907_390748


namespace sequence_sum_problem_l3907_390745

theorem sequence_sum_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n ≥ 2, a n + 2 * S n * S (n - 1) = 0) →
  S 5 = 1 / 11 →
  a 1 = 1 / 3 := by
sorry

end sequence_sum_problem_l3907_390745


namespace return_trip_time_l3907_390751

/-- Represents the time for a plane's journey between two cities -/
structure FlightTime where
  against_wind : ℝ  -- Time flying against the wind
  still_air : ℝ     -- Time flying in still air
  with_wind : ℝ     -- Time flying with the wind

/-- Checks if the flight times are valid according to the problem conditions -/
def is_valid_flight (ft : FlightTime) : Prop :=
  ft.against_wind = 75 ∧ ft.with_wind = ft.still_air - 10

/-- Theorem stating the possible return trip times -/
theorem return_trip_time (ft : FlightTime) :
  is_valid_flight ft → ft.with_wind = 15 ∨ ft.with_wind = 50 := by
  sorry

end return_trip_time_l3907_390751


namespace regular_polygon_sides_l3907_390769

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n > 2 → 
  interior_angle = 160 →
  (n : ℝ) * interior_angle = 180 * (n - 2) →
  n = 18 := by
sorry

end regular_polygon_sides_l3907_390769


namespace expression_simplification_l3907_390718

theorem expression_simplification (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c + 3)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (a*b + b*c + c*a + 3)⁻¹ * ((a*b)⁻¹ + (b*c)⁻¹ + (c*a)⁻¹ + 3) = (a*b*c)⁻¹ := by
  sorry

end expression_simplification_l3907_390718


namespace base_conversion_proof_l3907_390711

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

theorem base_conversion_proof :
  let base_5_101 := to_base_10 [1, 0, 1] 5
  let base_7_1234 := to_base_10 [4, 3, 2, 1] 7
  let base_9_3456 := to_base_10 [6, 5, 4, 3] 9
  2468 / base_5_101 * base_7_1234 - base_9_3456 = 41708 := by
sorry

end base_conversion_proof_l3907_390711


namespace box_2_neg2_3_l3907_390787

/-- Define the box operation for integers a, b, and c -/
def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

/-- Theorem stating that box(2, -2, 3) = 69/4 -/
theorem box_2_neg2_3 : box 2 (-2) 3 = 69/4 := by sorry

end box_2_neg2_3_l3907_390787


namespace problem_solution_l3907_390786

theorem problem_solution (a b m n k : ℝ) 
  (h1 : a + b = 0) 
  (h2 : m * n = 1) 
  (h3 : k^2 = 2) : 
  2011*a + 2012*b + m*n*a + k^2 = 2 := by
  sorry

end problem_solution_l3907_390786


namespace range_of_a_l3907_390750

theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, x > 1 → x > a) ∧ (∃ x : ℝ, x > a ∧ x ≤ 1) ↔ a < 1 := by
  sorry

end range_of_a_l3907_390750


namespace project_time_difference_l3907_390701

/-- Represents the working times of three people on a project -/
structure ProjectTime where
  t1 : ℕ  -- Time of person 1
  t2 : ℕ  -- Time of person 2
  t3 : ℕ  -- Time of person 3

/-- The proposition that the working times are in the ratio 1:2:3 -/
def ratio_correct (pt : ProjectTime) : Prop :=
  2 * pt.t1 = pt.t2 ∧ 3 * pt.t1 = pt.t3

/-- The total project time is 120 hours -/
def total_time_correct (pt : ProjectTime) : Prop :=
  pt.t1 + pt.t2 + pt.t3 = 120

/-- The main theorem stating the difference between longest and shortest working times -/
theorem project_time_difference (pt : ProjectTime) 
  (h1 : ratio_correct pt) (h2 : total_time_correct pt) : 
  pt.t3 - pt.t1 = 40 := by
  sorry


end project_time_difference_l3907_390701


namespace square_area_ratio_when_tripled_l3907_390766

theorem square_area_ratio_when_tripled (s : ℝ) (h : s > 0) :
  (s^2) / ((3*s)^2) = 1/9 := by sorry

end square_area_ratio_when_tripled_l3907_390766


namespace shoe_factory_production_l3907_390747

/-- The monthly production plan of a shoe factory. -/
def monthly_plan : ℝ := 5000

/-- The production in the first week as a fraction of the monthly plan. -/
def first_week : ℝ := 0.2

/-- The production in the second week as a fraction of the first week's production. -/
def second_week : ℝ := 1.2

/-- The production in the third week as a fraction of the first two weeks' combined production. -/
def third_week : ℝ := 0.6

/-- The production in the fourth week in pairs of shoes. -/
def fourth_week : ℝ := 1480

/-- Theorem stating that the given production schedule results in the monthly plan. -/
theorem shoe_factory_production :
  first_week * monthly_plan +
  second_week * first_week * monthly_plan +
  third_week * (first_week * monthly_plan + second_week * first_week * monthly_plan) +
  fourth_week = monthly_plan := by sorry

end shoe_factory_production_l3907_390747


namespace sum_of_series_equals_three_l3907_390728

theorem sum_of_series_equals_three : 
  ∑' k : ℕ+, (k : ℝ)^2 / 2^(k : ℝ) = 3 := by
  sorry

end sum_of_series_equals_three_l3907_390728


namespace new_ratio_second_term_l3907_390765

theorem new_ratio_second_term 
  (a b x : ℤ) 
  (h1 : a = 7)
  (h2 : b = 11)
  (h3 : x = 5)
  (h4 : a + x = 3) :
  ∃ y : ℤ, (a + x) * y = 3 * (b + x) ∧ y = 16 := by
  sorry

end new_ratio_second_term_l3907_390765


namespace prime_sequence_existence_l3907_390705

theorem prime_sequence_existence (k : ℕ) (hk : k > 1) :
  ∃ (p : ℕ) (a : ℕ → ℕ),
    Prime p ∧
    (∀ n m, n < m → a n < a m) ∧
    (∀ n, n > 1 → Prime (p + k * a n)) := by
  sorry

end prime_sequence_existence_l3907_390705


namespace choir_average_age_l3907_390724

theorem choir_average_age 
  (num_females : ℕ) (num_males : ℕ) (num_children : ℕ)
  (avg_age_females : ℚ) (avg_age_males : ℚ) (avg_age_children : ℚ)
  (h1 : num_females = 12)
  (h2 : num_males = 20)
  (h3 : num_children = 8)
  (h4 : avg_age_females = 28)
  (h5 : avg_age_males = 38)
  (h6 : avg_age_children = 10) :
  (num_females * avg_age_females + num_males * avg_age_males + num_children * avg_age_children) / 
  (num_females + num_males + num_children : ℚ) = 1176 / 40 := by
  sorry

end choir_average_age_l3907_390724


namespace ship_journey_day1_distance_l3907_390791

/-- Represents the distance traveled by a ship over three days -/
structure ShipJourney where
  day1_north : ℝ
  day2_east : ℝ
  day3_east : ℝ

/-- Calculates the total distance traveled by the ship -/
def total_distance (journey : ShipJourney) : ℝ :=
  journey.day1_north + journey.day2_east + journey.day3_east

/-- Theorem stating the distance traveled north on the first day -/
theorem ship_journey_day1_distance :
  ∀ (journey : ShipJourney),
    journey.day2_east = 3 * journey.day1_north →
    journey.day3_east = journey.day2_east + 110 →
    total_distance journey = 810 →
    journey.day1_north = 100 := by
  sorry

end ship_journey_day1_distance_l3907_390791


namespace steve_coins_value_l3907_390710

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Calculates the total value of coins given the number of nickels and dimes -/
def total_value (nickels dimes : ℕ) : ℕ :=
  nickels * nickel_value + dimes * dime_value

/-- Proves that given 2 nickels and 4 more dimes than nickels, the total value is 70 cents -/
theorem steve_coins_value : 
  ∀ (nickels : ℕ), nickels = 2 → total_value nickels (nickels + 4) = 70 := by
  sorry

end steve_coins_value_l3907_390710


namespace unique_non_expressible_l3907_390783

/-- Checks if a number can be expressed as x^2 + y^5 for some integers x and y -/
def isExpressible (n : ℤ) : Prop :=
  ∃ x y : ℤ, n = x^2 + y^5

/-- The list of numbers to check -/
def numberList : List ℤ := [59170, 59149, 59130, 59121, 59012]

/-- Theorem stating that 59121 is the only number in the list that cannot be expressed as x^2 + y^5 -/
theorem unique_non_expressible :
  ∀ n ∈ numberList, n ≠ 59121 → isExpressible n ∧ ¬isExpressible 59121 := by
  sorry

end unique_non_expressible_l3907_390783


namespace star_equality_implies_x_equals_three_l3907_390773

/-- Binary operation ⋆ on ordered pairs of integers -/
def star : (Int × Int) → (Int × Int) → (Int × Int) :=
  fun (a, b) (c, d) ↦ (a + c, b - d)

/-- Theorem stating that if (4,5) ⋆ (1,3) = (x,y) ⋆ (2,1), then x = 3 -/
theorem star_equality_implies_x_equals_three (x y : Int) :
  star (4, 5) (1, 3) = star (x, y) (2, 1) → x = 3 := by
  sorry


end star_equality_implies_x_equals_three_l3907_390773


namespace log_inequality_range_l3907_390737

open Real

theorem log_inequality_range (f : ℝ → ℝ) (t : ℝ) : 
  (∀ x > 0, f x = log x) →
  (∀ x > 0, f x + f t ≤ f (x^2 + t)) →
  0 < t ∧ t ≤ 4 :=
by sorry

end log_inequality_range_l3907_390737


namespace power_of_ten_division_l3907_390796

theorem power_of_ten_division : (10 ^ 8) / (10 * 10 ^ 5) = 100 := by
  sorry

end power_of_ten_division_l3907_390796


namespace percentage_reading_both_books_l3907_390772

theorem percentage_reading_both_books (total_students : ℕ) 
  (read_A : ℕ) (read_B : ℕ) (read_both : ℕ) :
  total_students = 600 →
  read_both = (20 * read_A) / 100 →
  read_A + read_B - read_both = total_students →
  read_A - read_both - (read_B - read_both) = 75 →
  (read_both * 100) / read_B = 25 :=
by sorry

end percentage_reading_both_books_l3907_390772


namespace gcd_228_1995_l3907_390790

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l3907_390790


namespace m_range_l3907_390706

def p (m : ℝ) : Prop := ∀ x > 0, m^2 + 2*m - 1 ≤ x + 1/x

def q (m : ℝ) : Prop := ∀ x₁ x₂, x₁ < x₂ → (5 - m^2)^x₁ < (5 - m^2)^x₂

theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) :
  (-3 ≤ m ∧ m ≤ -2) ∨ (1 < m ∧ m < 2) := by
  sorry

end m_range_l3907_390706


namespace least_possible_xy_l3907_390743

theorem least_possible_xy (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 8) :
  (x * y : ℕ) ≥ 128 ∧ ∃ (a b : ℕ+), (a : ℕ) * (b : ℕ) = 128 ∧ (1 : ℚ) / a + (1 : ℚ) / (2 * b) = (1 : ℚ) / 8 :=
sorry

end least_possible_xy_l3907_390743


namespace chenny_cups_bought_l3907_390717

def plate_cost : ℝ := 2
def spoon_cost : ℝ := 1.5
def fork_cost : ℝ := 1.25
def cup_cost : ℝ := 3
def num_plates : ℕ := 9

def total_spoons_forks_cost : ℝ := 13.5
def total_plates_cups_cost : ℝ := 25.5

theorem chenny_cups_bought :
  ∃ (num_spoons num_forks num_cups : ℕ),
    num_spoons = num_forks ∧
    num_spoons * spoon_cost + num_forks * fork_cost = total_spoons_forks_cost ∧
    num_plates * plate_cost + num_cups * cup_cost = total_plates_cups_cost ∧
    num_cups = 2 :=
by sorry

end chenny_cups_bought_l3907_390717


namespace two_digit_integer_problem_l3907_390741

theorem two_digit_integer_problem (n : ℕ) :
  n ≥ 10 ∧ n ≤ 99 →
  (60 + n) / 2 = 60 + n / 100 →
  min 60 n = 59 := by
sorry

end two_digit_integer_problem_l3907_390741


namespace even_function_property_l3907_390780

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_prop : ∀ x, f (x + 2) = x * f x) : 
  f 1 = 0 := by sorry

end even_function_property_l3907_390780


namespace prime_divisor_existence_l3907_390740

theorem prime_divisor_existence (p : Nat) (hp : p.Prime ∧ p ≥ 3) :
  ∃ N : Nat, ∀ x ≥ N, ∃ i ∈ Finset.range ((p + 3) / 2), 
    ∃ q : Nat, q.Prime ∧ q > p ∧ q ∣ (x + i + 1) :=
by sorry

end prime_divisor_existence_l3907_390740


namespace infinite_solutions_condition_l3907_390770

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 8)) ↔ b = -6 := by
  sorry

end infinite_solutions_condition_l3907_390770


namespace production_days_calculation_l3907_390758

theorem production_days_calculation (n : ℕ) : 
  (∀ (k : ℕ), k ≤ n → (60 * k : ℝ) = (60 : ℝ) * k) → 
  ((60 * n + 90 : ℝ) / (n + 1) = 62) → 
  n = 14 :=
sorry

end production_days_calculation_l3907_390758


namespace cube_root_simplification_l3907_390764

theorem cube_root_simplification :
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 112 ^ (1/3) :=
by sorry

end cube_root_simplification_l3907_390764


namespace cone_volume_lateral_area_l3907_390733

/-- The volume of a cone in terms of its lateral surface area and the distance from the center of the base to the slant height. -/
theorem cone_volume_lateral_area (S r : ℝ) (h1 : S > 0) (h2 : r > 0) : ∃ V : ℝ, V = (1/3) * S * r ∧ V > 0 := by
  sorry

end cone_volume_lateral_area_l3907_390733


namespace max_quotient_value_l3907_390753

theorem max_quotient_value (a b : ℝ) 
  (ha : 100 ≤ a ∧ a ≤ 300)
  (hb : 800 ≤ b ∧ b ≤ 1600)
  (hab : a + b ≤ 1800) :
  ∃ (a' b' : ℝ), 
    100 ≤ a' ∧ a' ≤ 300 ∧
    800 ≤ b' ∧ b' ≤ 1600 ∧
    a' + b' ≤ 1800 ∧
    b' / a' = 5 ∧
    ∀ (x y : ℝ), 
      100 ≤ x ∧ x ≤ 300 → 
      800 ≤ y ∧ y ≤ 1600 → 
      x + y ≤ 1800 → 
      y / x ≤ 5 :=
sorry

end max_quotient_value_l3907_390753


namespace rectangle_area_l3907_390779

theorem rectangle_area (breadth : ℝ) (h1 : breadth > 0) : 
  let length := 3 * breadth
  let perimeter := 2 * (length + breadth)
  perimeter = 48 → breadth * length = 108 := by
  sorry

end rectangle_area_l3907_390779


namespace gcd_288_123_l3907_390777

theorem gcd_288_123 : Nat.gcd 288 123 = 3 := by
  sorry

end gcd_288_123_l3907_390777


namespace fraction_simplification_l3907_390793

theorem fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (8 * a^4 * b^2 * c) / (4 * a^3 * b) = 2 * a * b * c := by
  sorry

end fraction_simplification_l3907_390793


namespace smallest_k_for_no_real_roots_l3907_390776

theorem smallest_k_for_no_real_roots : 
  ∃ (k : ℤ), k = 1 ∧ 
  (∀ (x : ℝ), (k + 1 : ℝ) * x^2 - (6 * k + 2 : ℝ) * x + (3 * k + 2 : ℝ) ≠ 0) ∧
  (∀ (j : ℤ), j < k → ∃ (x : ℝ), (j + 1 : ℝ) * x^2 - (6 * j + 2 : ℝ) * x + (3 * j + 2 : ℝ) = 0) :=
by sorry

#check smallest_k_for_no_real_roots

end smallest_k_for_no_real_roots_l3907_390776


namespace grasshopper_jump_distance_l3907_390788

/-- The distance jumped by the grasshopper in inches -/
def grasshopper_jump : ℕ := sorry

/-- The distance jumped by the frog in inches -/
def frog_jump : ℕ := 53

/-- The difference between the frog's jump and the grasshopper's jump in inches -/
def frog_grasshopper_diff : ℕ := 17

theorem grasshopper_jump_distance :
  grasshopper_jump = frog_jump - frog_grasshopper_diff :=
by sorry

end grasshopper_jump_distance_l3907_390788


namespace yard_length_is_700_l3907_390715

/-- The length of a yard with trees planted at equal distances -/
def yard_length (num_trees : ℕ) (tree_distance : ℕ) : ℕ :=
  (num_trees - 1) * tree_distance

/-- Proof that the yard length is 700 meters -/
theorem yard_length_is_700 :
  yard_length 26 28 = 700 := by
  sorry

end yard_length_is_700_l3907_390715


namespace arithmetic_sequence_length_l3907_390709

theorem arithmetic_sequence_length 
  (a : ℤ) (an : ℤ) (d : ℤ) (n : ℕ) 
  (h1 : a = -50) 
  (h2 : an = 74) 
  (h3 : d = 6) 
  (h4 : an = a + (n - 1) * d) : n = 22 := by
  sorry

end arithmetic_sequence_length_l3907_390709


namespace abs_neg_three_eq_three_l3907_390727

theorem abs_neg_three_eq_three : |(-3 : ℝ)| = 3 := by
  sorry

end abs_neg_three_eq_three_l3907_390727


namespace bread_slices_proof_l3907_390752

/-- The number of slices Andy ate at each time -/
def slices_eaten_per_time : ℕ := 3

/-- The number of times Andy ate slices -/
def times_andy_ate : ℕ := 2

/-- The number of slices needed to make one piece of toast bread -/
def slices_per_toast : ℕ := 2

/-- The number of pieces of toast bread made -/
def toast_pieces_made : ℕ := 10

/-- The number of slices left after making toast -/
def slices_left : ℕ := 1

/-- The total number of slices in the original loaf of bread -/
def total_slices : ℕ := 27

theorem bread_slices_proof :
  total_slices = 
    slices_eaten_per_time * times_andy_ate + 
    slices_per_toast * toast_pieces_made + 
    slices_left :=
by
  sorry

end bread_slices_proof_l3907_390752


namespace g_difference_l3907_390785

/-- A linear function with a constant difference of 4 between consecutive integers -/
def g_property (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, g (x + y) = g x + g y) ∧ 
  (∀ h : ℝ, g (h + 1) - g h = 4)

/-- The difference between g(3) and g(7) is -16 -/
theorem g_difference (g : ℝ → ℝ) (hg : g_property g) : g 3 - g 7 = -16 := by
  sorry

end g_difference_l3907_390785


namespace maria_friends_money_l3907_390799

/-- The amount of money Rene received from Maria -/
def rene_amount : ℕ := 300

/-- The amount of money Florence received from Maria -/
def florence_amount : ℕ := 3 * rene_amount

/-- The amount of money Isha received from Maria -/
def isha_amount : ℕ := florence_amount / 2

/-- The total amount of money Maria gave to her three friends -/
def total_amount : ℕ := isha_amount + florence_amount + rene_amount

/-- Theorem stating that the total amount Maria gave to her friends is $1650 -/
theorem maria_friends_money : total_amount = 1650 := by sorry

end maria_friends_money_l3907_390799


namespace max_fraction_sum_l3907_390719

theorem max_fraction_sum (x y : ℝ) :
  (Real.sqrt 3 * x - y + Real.sqrt 3 ≥ 0) →
  (Real.sqrt 3 * x + y - Real.sqrt 3 ≤ 0) →
  (y ≥ 0) →
  (∀ x' y' : ℝ, (Real.sqrt 3 * x' - y' + Real.sqrt 3 ≥ 0) →
                (Real.sqrt 3 * x' + y' - Real.sqrt 3 ≤ 0) →
                (y' ≥ 0) →
                ((y' + 1) / (x' + 3) ≤ (y + 1) / (x + 3))) →
  x + y = Real.sqrt 3 := by
sorry

end max_fraction_sum_l3907_390719


namespace p_necessary_not_sufficient_for_q_l3907_390782

/-- p is a necessary but not sufficient condition for q -/
theorem p_necessary_not_sufficient_for_q :
  (∀ x, (-1 < x ∧ x < 2) → x < 3) ∧
  ¬(∀ x, x < 3 → (-1 < x ∧ x < 2)) := by
  sorry

end p_necessary_not_sufficient_for_q_l3907_390782


namespace school_boys_count_l3907_390792

theorem school_boys_count (muslim_percent : ℝ) (hindu_percent : ℝ) (sikh_percent : ℝ) (other_count : ℕ) :
  muslim_percent = 0.44 →
  hindu_percent = 0.28 →
  sikh_percent = 0.10 →
  other_count = 72 →
  ∃ (total : ℕ), 
    (muslim_percent + hindu_percent + sikh_percent + (other_count : ℝ) / total) = 1 ∧
    total = 400 := by
  sorry

end school_boys_count_l3907_390792


namespace complex_multiplication_l3907_390775

def i : ℂ := Complex.I

theorem complex_multiplication :
  (6 - 3 * i) * (-7 + 2 * i) = -36 + 33 * i :=
by
  -- The proof goes here
  sorry

end complex_multiplication_l3907_390775


namespace complex_expression_simplification_l3907_390703

theorem complex_expression_simplification :
  (7 + 4 * Real.sqrt 3) * (2 - Real.sqrt 3)^2 + (2 + Real.sqrt 3) * (2 - Real.sqrt 3) - Real.sqrt 3 = 2 - Real.sqrt 3 :=
by sorry

end complex_expression_simplification_l3907_390703


namespace geometric_sequence_first_term_l3907_390746

/-- 
Given a geometric sequence where the fifth term is 64 and the sixth term is 128,
prove that the first term of the sequence is 4.
-/
theorem geometric_sequence_first_term (a b c d : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ 
    b = a * r ∧ 
    c = b * r ∧ 
    d = c * r ∧ 
    64 = d * r ∧ 
    128 = 64 * r) → 
  a = 4 :=
by sorry

end geometric_sequence_first_term_l3907_390746


namespace complex_multiplication_l3907_390794

theorem complex_multiplication (z : ℂ) (i : ℂ) : 
  z.re = 1 ∧ z.im = 1 ∧ i * i = -1 → z * (1 - i) = 2 := by sorry

end complex_multiplication_l3907_390794


namespace sum_of_ages_l3907_390722

theorem sum_of_ages (bella_age : ℕ) (age_difference : ℕ) : 
  bella_age = 5 → 
  age_difference = 9 → 
  bella_age + (bella_age + age_difference) = 19 := by
sorry

end sum_of_ages_l3907_390722


namespace vector_parallelism_l3907_390784

theorem vector_parallelism (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![1, x]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (2 • a - b)) → x = 1/2 := by
  sorry

end vector_parallelism_l3907_390784


namespace sqrt_64_equals_8_l3907_390738

theorem sqrt_64_equals_8 : Real.sqrt 64 = 8 := by
  sorry

end sqrt_64_equals_8_l3907_390738


namespace dans_initial_money_l3907_390744

/-- The amount of money Dan spent on the candy bar -/
def candy_bar_cost : ℝ := 1.00

/-- The amount of money Dan has left after buying the candy bar -/
def money_left : ℝ := 2.00

/-- Dan's initial amount of money -/
def initial_money : ℝ := candy_bar_cost + money_left

theorem dans_initial_money : initial_money = 3.00 := by sorry

end dans_initial_money_l3907_390744


namespace invitation_combinations_l3907_390716

theorem invitation_combinations (n m : ℕ) (h : n = 10 ∧ m = 6) : 
  (Nat.choose n m) - (Nat.choose (n - 2) (m - 2)) = 140 :=
sorry

end invitation_combinations_l3907_390716


namespace parallel_lines_c_value_l3907_390729

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 8x + 2 and y = (2c)x - 4 are parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = 8 * x + 2 ↔ y = (2 * c) * x - 4) → c = 4 :=
by sorry

end parallel_lines_c_value_l3907_390729


namespace power_sum_equals_zero_l3907_390713

theorem power_sum_equals_zero : 1^2009 + (-1)^2009 = 0 := by
  sorry

end power_sum_equals_zero_l3907_390713


namespace range_of_m_l3907_390725

/-- A decreasing function on the open interval (-2, 2) -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 < x ∧ x < y ∧ y < 2 → f x > f y

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (m - 1) > f (2 * m - 1)) :
  0 < m ∧ m < 3/2 := by
  sorry

end range_of_m_l3907_390725


namespace smallest_4digit_divisible_by_5_6_2_l3907_390756

def is_divisible (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_4digit_divisible_by_5_6_2 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 →
  (is_divisible n 5 ∧ is_divisible n 6 ∧ is_divisible n 2) →
  1020 ≤ n :=
by sorry

end smallest_4digit_divisible_by_5_6_2_l3907_390756


namespace arithmetic_calculations_l3907_390762

theorem arithmetic_calculations :
  (12 - (-18) + (-7) - 15 = 8) ∧
  (5 + 1 / 7 : ℚ) * (7 / 8 : ℚ) / (-8 / 9 : ℚ) / 3 = -27 / 16 := by
  sorry

end arithmetic_calculations_l3907_390762


namespace exists_monochromatic_triangle_l3907_390778

-- Define the vertices of the hexagon
inductive Vertex : Type
  | A | B | C | D | E | F

-- Define the colors
inductive Color : Type
  | Blue | Yellow

-- Define an edge as a pair of vertices
def Edge : Type := Vertex × Vertex

-- Function to get the color of an edge
def edge_color : Edge → Color := sorry

-- Define the hexagon
def hexagon : Set Edge := sorry

-- Theorem statement
theorem exists_monochromatic_triangle :
  ∃ (v1 v2 v3 : Vertex) (c : Color),
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    edge_color (v1, v2) = c ∧
    edge_color (v2, v3) = c ∧
    edge_color (v1, v3) = c :=
  sorry

end exists_monochromatic_triangle_l3907_390778


namespace two_variables_scatter_plot_l3907_390736

-- Define a type for statistical variables
def StatisticalVariable : Type := ℝ

-- Define a type for a dataset of two variables
def Dataset : Type := List (StatisticalVariable × StatisticalVariable)

-- Statement: Any two statistical variables can be represented with a scatter plot
theorem two_variables_scatter_plot (data : Dataset) :
  ∃ (scatter_plot : Dataset → Bool), scatter_plot data = true :=
sorry

end two_variables_scatter_plot_l3907_390736


namespace distribute_six_balls_four_boxes_l3907_390798

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes,
    with at least one ball in each box. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 2 ways to distribute 6 indistinguishable balls into 4 indistinguishable boxes,
    with at least one ball in each box. -/
theorem distribute_six_balls_four_boxes :
  distribute_balls 6 4 = 2 := by
  sorry

end distribute_six_balls_four_boxes_l3907_390798


namespace complement_A_intersect_B_l3907_390714

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3}

-- Define set A
def A : Set Nat := {0, 1}

-- Define set B
def B : Set Nat := {1, 2, 3}

-- Theorem statement
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {2, 3} := by
  sorry

-- Note: Aᶜ represents the complement of A in the universal set U

end complement_A_intersect_B_l3907_390714


namespace problem_statement_l3907_390720

theorem problem_statement : 
  let A := (16 * Real.sqrt 2) ^ (1/3 : ℝ)
  let B := Real.sqrt (9 * 9 ^ (1/3 : ℝ))
  let C := ((2 ^ (1/5 : ℝ)) ^ 2) ^ 2
  A ^ 2 + B ^ 3 + C ^ 5 = 105 := by sorry

end problem_statement_l3907_390720


namespace pizza_consumption_order_l3907_390760

structure Sibling where
  name : String
  pizza_fraction : ℚ

def pizza_problem (alex beth cyril eve dan : Sibling) : Prop :=
  alex.name = "Alex" ∧
  beth.name = "Beth" ∧
  cyril.name = "Cyril" ∧
  eve.name = "Eve" ∧
  dan.name = "Dan" ∧
  alex.pizza_fraction = 1/7 ∧
  beth.pizza_fraction = 1/5 ∧
  cyril.pizza_fraction = 1/6 ∧
  eve.pizza_fraction = 1/9 ∧
  dan.pizza_fraction = 1 - (alex.pizza_fraction + beth.pizza_fraction + cyril.pizza_fraction + eve.pizza_fraction)

theorem pizza_consumption_order (alex beth cyril eve dan : Sibling) 
  (h : pizza_problem alex beth cyril eve dan) :
  dan.pizza_fraction > beth.pizza_fraction ∧
  beth.pizza_fraction > cyril.pizza_fraction ∧
  cyril.pizza_fraction > alex.pizza_fraction ∧
  alex.pizza_fraction > eve.pizza_fraction :=
by sorry

end pizza_consumption_order_l3907_390760


namespace theater_sales_total_cost_l3907_390754

/-- Represents the theater ticket sales problem --/
structure TheaterSales where
  total_tickets : ℕ
  balcony_surplus : ℕ
  orchestra_price : ℕ
  balcony_price : ℕ

/-- Calculate the total cost of tickets sold --/
def total_cost (sales : TheaterSales) : ℕ :=
  let orchestra_tickets := (sales.total_tickets - sales.balcony_surplus) / 2
  let balcony_tickets := sales.total_tickets - orchestra_tickets
  orchestra_tickets * sales.orchestra_price + balcony_tickets * sales.balcony_price

/-- Theorem stating that the total cost for the given conditions is $3320 --/
theorem theater_sales_total_cost :
  let sales : TheaterSales := {
    total_tickets := 370,
    balcony_surplus := 190,
    orchestra_price := 12,
    balcony_price := 8
  }
  total_cost sales = 3320 := by
  sorry


end theater_sales_total_cost_l3907_390754


namespace penelope_greta_ratio_l3907_390731

/-- The amount of food animals eat per day in pounds -/
structure AnimalFood where
  penelope : ℝ
  greta : ℝ
  milton : ℝ
  elmer : ℝ

/-- The conditions given in the problem -/
def problem_conditions (food : AnimalFood) : Prop :=
  food.penelope = 20 ∧
  food.milton = food.greta / 100 ∧
  food.elmer = 4000 * food.milton ∧
  food.elmer = food.penelope + 60

/-- The theorem to be proved -/
theorem penelope_greta_ratio (food : AnimalFood) :
  problem_conditions food → food.penelope / food.greta = 10 := by
  sorry

end penelope_greta_ratio_l3907_390731


namespace chocolate_bar_count_l3907_390771

/-- The number of people sharing the box of chocolate bars -/
def num_people : ℕ := 3

/-- The number of bars two people got combined -/
def bars_two_people : ℕ := 8

/-- The total number of bars in the box -/
def total_bars : ℕ := 16

/-- Theorem stating that the total number of bars is 16 -/
theorem chocolate_bar_count :
  (num_people : ℕ) = 3 →
  (bars_two_people : ℕ) = 8 →
  (total_bars : ℕ) = 16 :=
by
  sorry

end chocolate_bar_count_l3907_390771


namespace arithmetic_sequence_ninth_term_l3907_390708

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  firstTerm : ℤ
  commonDiff : ℤ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.firstTerm + (n - 1) * seq.commonDiff

theorem arithmetic_sequence_ninth_term
  (seq : ArithmeticSequence)
  (h3 : seq.nthTerm 3 = 5)
  (h6 : seq.nthTerm 6 = 17) :
  seq.nthTerm 9 = 29 := by
  sorry

#check arithmetic_sequence_ninth_term

end arithmetic_sequence_ninth_term_l3907_390708


namespace information_spread_l3907_390767

theorem information_spread (population : ℕ) (h : population = 1000000) : 
  ∃ (n : ℕ), (2^(n+1) - 1 ≥ population) ∧ (∀ m : ℕ, m < n → 2^(m+1) - 1 < population) :=
sorry

end information_spread_l3907_390767


namespace tangent_line_intersection_l3907_390795

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m
def g (x : ℝ) : ℝ := 6 * Real.log x - 4 * x

theorem tangent_line_intersection (m : ℝ) : 
  (∃ a : ℝ, a > 0 ∧ 
    f m a = g a ∧ 
    (deriv (f m)) a = (deriv g) a) → 
  m = -5 :=
sorry

end

end tangent_line_intersection_l3907_390795


namespace complex_cube_sum_div_product_l3907_390781

theorem complex_cube_sum_div_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 8 := by
sorry

end complex_cube_sum_div_product_l3907_390781
