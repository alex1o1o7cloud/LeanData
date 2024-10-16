import Mathlib

namespace NUMINAMATH_CALUDE_fifteenth_student_age_l838_83836

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℕ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℕ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℕ) 
  (h1 : total_students = 15) 
  (h2 : avg_age_all = 15) 
  (h3 : group1_size = 5) 
  (h4 : avg_age_group1 = 14) 
  (h5 : group2_size = 9) 
  (h6 : avg_age_group2 = 16) :
  total_students * avg_age_all = 
    group1_size * avg_age_group1 + 
    group2_size * avg_age_group2 + 11 :=
by sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l838_83836


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l838_83879

-- Define the equation of an ellipse
def is_ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / a + y^2 / b = 1 → (a > 0 ∧ b > 0 ∧ a ≠ b)

-- Define the condition ab > 0
def condition (a b : ℝ) : Prop := a * b > 0

-- Theorem stating that the condition is necessary but not sufficient
theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, is_ellipse_equation a b → condition a b) ∧
  (∃ a b : ℝ, condition a b ∧ ¬is_ellipse_equation a b) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l838_83879


namespace NUMINAMATH_CALUDE_equation_solution_l838_83806

theorem equation_solution :
  ∀ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) → x = -48 / 23 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l838_83806


namespace NUMINAMATH_CALUDE_quadratic_roots_value_l838_83807

theorem quadratic_roots_value (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → 
  d = 49/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_value_l838_83807


namespace NUMINAMATH_CALUDE_smallest_y_l838_83865

theorem smallest_y (y : ℕ) 
  (h1 : y % 6 = 5) 
  (h2 : y % 7 = 6) 
  (h3 : y % 8 = 7) : 
  y ≥ 167 ∧ ∃ (z : ℕ), z % 6 = 5 ∧ z % 7 = 6 ∧ z % 8 = 7 ∧ z = 167 :=
sorry

end NUMINAMATH_CALUDE_smallest_y_l838_83865


namespace NUMINAMATH_CALUDE_right_triangle_with_specific_altitude_and_segment_difference_l838_83819

/-- Represents a right-angled triangle with an altitude to the hypotenuse -/
structure RightTriangleWithAltitude where
  /-- First leg of the triangle -/
  leg1 : ℝ
  /-- Second leg of the triangle -/
  leg2 : ℝ
  /-- Hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- Altitude drawn to the hypotenuse -/
  altitude : ℝ
  /-- First segment of the hypotenuse -/
  segment1 : ℝ
  /-- Second segment of the hypotenuse -/
  segment2 : ℝ
  /-- The triangle is right-angled -/
  right_angle : leg1^2 + leg2^2 = hypotenuse^2
  /-- The altitude divides the hypotenuse into two segments -/
  hypotenuse_segments : segment1 + segment2 = hypotenuse
  /-- The altitude creates similar triangles -/
  similar_triangles : altitude^2 = segment1 * segment2

/-- Theorem: Given a right-angled triangle with specific altitude and hypotenuse segment difference, prove its sides -/
theorem right_triangle_with_specific_altitude_and_segment_difference
  (t : RightTriangleWithAltitude)
  (h_altitude : t.altitude = 12)
  (h_segment_diff : t.segment1 - t.segment2 = 7) :
  t.leg1 = 15 ∧ t.leg2 = 20 ∧ t.hypotenuse = 25 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_specific_altitude_and_segment_difference_l838_83819


namespace NUMINAMATH_CALUDE_exam_results_l838_83842

theorem exam_results (total_students : ℕ) 
  (percent_8_or_more : ℚ) (percent_5_or_less : ℚ) :
  total_students = 40 →
  percent_8_or_more = 20 / 100 →
  percent_5_or_less = 45 / 100 →
  (1 : ℚ) - percent_8_or_more - percent_5_or_less = 35 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l838_83842


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l838_83853

-- Define the parabola P
def P (x : ℝ) : ℝ := x^2 + 5

-- Define the point Q
def Q : ℝ × ℝ := (10, 10)

-- Define the line through Q with slope m
def line_through_Q (m : ℝ) (x : ℝ) : ℝ := m * (x - Q.1) + Q.2

-- Define the condition for no intersection
def no_intersection (m : ℝ) : Prop :=
  ∀ x : ℝ, line_through_Q m x ≠ P x

-- Define r and s
noncomputable def r : ℝ := 20 - 10 * Real.sqrt 38
noncomputable def s : ℝ := 20 + 10 * Real.sqrt 38

-- Theorem statement
theorem parabola_line_intersection :
  (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) →
  r + s = 40 :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l838_83853


namespace NUMINAMATH_CALUDE_zeros_after_one_in_10000_pow_50_l838_83855

theorem zeros_after_one_in_10000_pow_50 :
  ∃ (n : ℕ), 10000^50 = 10^n ∧ n = 200 := by
  sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_10000_pow_50_l838_83855


namespace NUMINAMATH_CALUDE_find_n_l838_83891

/-- Definition of S_n -/
def S (n : ℕ) : ℚ := n / (n + 1)

/-- Theorem stating that n = 6 satisfies the given conditions -/
theorem find_n : ∃ (n : ℕ), S n * S (n + 1) = 3/4 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l838_83891


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l838_83857

theorem crazy_silly_school_series (total_books : ℕ) (read_books : ℕ) (remaining_books : ℕ) (watched_movies : ℕ) :
  total_books = 22 →
  read_books = 12 →
  remaining_books = 10 →
  watched_movies = 56 →
  ¬∃ (total_movies : ℕ), ∀ (n : ℕ), n = total_movies → n ≥ watched_movies :=
by sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l838_83857


namespace NUMINAMATH_CALUDE_initial_number_of_girls_l838_83887

theorem initial_number_of_girls (initial_boys : ℕ) (girls_joined : ℕ) (final_girls : ℕ) : 
  initial_boys = 761 → girls_joined = 682 → final_girls = 1414 → 
  final_girls - girls_joined = 732 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_girls_l838_83887


namespace NUMINAMATH_CALUDE_smallest_prime_sum_l838_83832

def digit_set : Set Nat := {1, 2, 3, 5}

def is_valid_prime (p : Nat) (used_digits : Set Nat) : Prop :=
  Nat.Prime p ∧ 
  (p % 10) ∈ digit_set ∧
  (p % 10) ∉ used_digits ∧
  (∀ d ∈ digit_set, d ≠ p % 10 → ¬ (∃ k, p / 10^k % 10 = d))

def valid_prime_triple (p q r : Nat) : Prop :=
  is_valid_prime p ∅ ∧
  is_valid_prime q {p % 10} ∧
  is_valid_prime r {p % 10, q % 10}

theorem smallest_prime_sum :
  ∀ p q r, valid_prime_triple p q r → p + q + r ≥ 71 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_sum_l838_83832


namespace NUMINAMATH_CALUDE_quadratic_inequality_l838_83830

theorem quadratic_inequality (x : ℝ) : x^2 - 8*x + 12 < 0 ↔ 2 < x ∧ x < 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l838_83830


namespace NUMINAMATH_CALUDE_half_circle_roll_distance_l838_83878

/-- The length of the path traveled by the center of a half-circle when rolled along a straight line -/
theorem half_circle_roll_distance (r : ℝ) (h : r = 3 / Real.pi) : 
  let roll_distance := r * Real.pi + r
  roll_distance = 3 + 3 / Real.pi := by sorry

end NUMINAMATH_CALUDE_half_circle_roll_distance_l838_83878


namespace NUMINAMATH_CALUDE_probability_between_lines_in_first_quadrant_l838_83846

/-- Line represented by a linear equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def Line.eval (l : Line) (x : ℝ) : ℝ := l.m * x + l.b

def is_below (p : Point) (l : Line) : Prop := p.y ≤ l.eval p.x

def is_in_first_quadrant (p : Point) : Prop := p.x ≥ 0 ∧ p.y ≥ 0

def is_between_lines (p : Point) (l1 l2 : Line) : Prop :=
  is_below p l1 ∧ ¬is_below p l2

theorem probability_between_lines_in_first_quadrant
  (l m : Line)
  (h1 : l.m = -3 ∧ l.b = 9)
  (h2 : m.m = -1 ∧ m.b = 3)
  (h3 : ∀ (p : Point), is_in_first_quadrant p → is_below p l → is_below p m) :
  (∀ (p : Point), is_in_first_quadrant p → is_below p l → is_between_lines p l m) :=
sorry

end NUMINAMATH_CALUDE_probability_between_lines_in_first_quadrant_l838_83846


namespace NUMINAMATH_CALUDE_parallel_tangents_sum_bound_l838_83840

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k + 4/k) * Real.log x + (4 - x^2) / x

theorem parallel_tangents_sum_bound (k : ℝ) (x₁ x₂ : ℝ) (h_k : k ≥ 4) 
  (h_distinct : x₁ ≠ x₂) (h_positive : x₁ > 0 ∧ x₂ > 0) 
  (h_parallel : (deriv (f k)) x₁ = (deriv (f k)) x₂) :
  x₁ + x₂ > 16/5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_sum_bound_l838_83840


namespace NUMINAMATH_CALUDE_mersenne_last_two_digits_l838_83812

/-- The exponent used in the Mersenne prime -/
def p : ℕ := 82589933

/-- The Mersenne number -/
def mersenne_number : ℕ := 2^p - 1

/-- The last two digits of a number -/
def last_two_digits (n : ℕ) : ℕ := n % 100

theorem mersenne_last_two_digits : last_two_digits mersenne_number = 91 := by
  sorry

end NUMINAMATH_CALUDE_mersenne_last_two_digits_l838_83812


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l838_83848

theorem greatest_integer_satisfying_conditions : 
  ∃ (n : ℕ), n < 150 ∧ 
  (∃ (k : ℕ), n = 11 * k - 1) ∧ 
  (∃ (l : ℕ), n = 9 * l + 2) ∧
  (∀ (m : ℕ), m < 150 → 
    (∃ (k' : ℕ), m = 11 * k' - 1) → 
    (∃ (l' : ℕ), m = 9 * l' + 2) → 
    m ≤ n) ∧
  n = 65 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l838_83848


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l838_83880

theorem arithmetic_calculations :
  (3 * 232 + 456 = 1152) ∧
  (760 * 5 - 2880 = 920) ∧
  (805 / 7 = 115) ∧
  (45 + 255 / 5 = 96) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l838_83880


namespace NUMINAMATH_CALUDE_sum_of_roots_of_equation_l838_83863

theorem sum_of_roots_of_equation (x : ℝ) : 
  (∃ a b : ℝ, (a - 7)^2 = 16 ∧ (b - 7)^2 = 16 ∧ a + b = 14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_of_equation_l838_83863


namespace NUMINAMATH_CALUDE_dandelion_puff_distribution_l838_83844

theorem dandelion_puff_distribution (total : ℕ) (given_away : ℕ) (friends : ℕ) 
  (h1 : total = 100) 
  (h2 : given_away = 42) 
  (h3 : friends = 7) :
  (total - given_away) / friends = 8 ∧ 
  (8 : ℚ) / (total - given_away) = 4 / 29 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_puff_distribution_l838_83844


namespace NUMINAMATH_CALUDE_jogs_five_miles_per_day_l838_83868

/-- Represents the number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- Represents the number of weeks -/
def num_weeks : ℕ := 3

/-- Represents the total miles run over the given weeks -/
def total_miles : ℕ := 75

/-- Calculates the number of miles jogged per day -/
def miles_per_day : ℚ :=
  total_miles / (weekdays_per_week * num_weeks)

/-- Theorem stating that the person jogs 5 miles per day -/
theorem jogs_five_miles_per_day : miles_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_jogs_five_miles_per_day_l838_83868


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l838_83885

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) ↔ x ≥ 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l838_83885


namespace NUMINAMATH_CALUDE_number_of_students_l838_83893

theorem number_of_students (initial_average : ℝ) (wrong_mark : ℝ) (correct_mark : ℝ) (correct_average : ℝ) :
  initial_average = 100 →
  wrong_mark = 90 →
  correct_mark = 10 →
  correct_average = 92 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * initial_average - (wrong_mark - correct_mark) = (n : ℝ) * correct_average ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_l838_83893


namespace NUMINAMATH_CALUDE_water_level_accurate_l838_83828

/-- Represents the water level function for a reservoir -/
def waterLevel (x : ℝ) : ℝ := 6 + 0.3 * x

/-- Theorem stating that the water level function accurately describes the reservoir's water level -/
theorem water_level_accurate (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) : 
  waterLevel x = 6 + 0.3 * x ∧ 
  waterLevel 0 = 6 ∧
  ∀ t₁ t₂, 0 ≤ t₁ ∧ t₁ < t₂ ∧ t₂ ≤ 5 → (waterLevel t₂ - waterLevel t₁) / (t₂ - t₁) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_water_level_accurate_l838_83828


namespace NUMINAMATH_CALUDE_inequality_solution_set_l838_83890

theorem inequality_solution_set (a b c : ℝ) : 
  a > 0 → 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 2 ↔ 0 ≤ a * x^2 + b * x + c ∧ a * x^2 + b * x + c ≤ 1) →
  4 * a + 5 * b + c = -1/4 ∨ 4 * a + 5 * b + c = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l838_83890


namespace NUMINAMATH_CALUDE_total_distance_traveled_l838_83802

def father_step_length : ℕ := 80
def son_step_length : ℕ := 60
def coincidences : ℕ := 601

def lcm_step_lengths : ℕ := Nat.lcm father_step_length son_step_length

theorem total_distance_traveled :
  (coincidences - 1) * lcm_step_lengths = 144000 :=
by sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l838_83802


namespace NUMINAMATH_CALUDE_real_part_of_z_l838_83870

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  (z.re : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l838_83870


namespace NUMINAMATH_CALUDE_complex_cube_eq_negative_eight_l838_83896

theorem complex_cube_eq_negative_eight :
  (1 + Complex.I * Real.sqrt 3) ^ 3 = -8 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_eq_negative_eight_l838_83896


namespace NUMINAMATH_CALUDE_centimeters_per_kilometer_l838_83867

-- Define the conversion factors
def meters_per_kilometer : ℝ := 1000
def centimeters_per_meter : ℝ := 100

-- Theorem statement
theorem centimeters_per_kilometer : 
  meters_per_kilometer * centimeters_per_meter = 100000 := by
  sorry

end NUMINAMATH_CALUDE_centimeters_per_kilometer_l838_83867


namespace NUMINAMATH_CALUDE_probability_more_than_third_correct_l838_83822

-- Define the number of questions
def n : ℕ := 12

-- Define the probability of guessing correctly
def p : ℚ := 1/2

-- Define the minimum number of correct answers needed
def k : ℕ := 5

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the probability of getting at least k correct answers
def prob_at_least_k (n k : ℕ) (p : ℚ) : ℚ := sorry

-- State the theorem
theorem probability_more_than_third_correct :
  prob_at_least_k n k p = 825/1024 := by sorry

end NUMINAMATH_CALUDE_probability_more_than_third_correct_l838_83822


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l838_83854

-- Define the lines that bound the region
def line1 (x y : ℝ) : Prop := x + y = 6
def line2 (y : ℝ) : Prop := y = 4
def line3 (x : ℝ) : Prop := x = 0
def line4 (y : ℝ) : Prop := y = 0

-- Define the vertices of the quadrilateral
def P : ℝ × ℝ := (6, 0)
def Q : ℝ × ℝ := (2, 4)
def R : ℝ × ℝ := (0, 6)
def O : ℝ × ℝ := (0, 0)

-- Define the area of the quadrilateral
def area_quadrilateral : ℝ := 18

-- Theorem statement
theorem area_of_bounded_region :
  area_quadrilateral = 18 :=
sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l838_83854


namespace NUMINAMATH_CALUDE_inequality_solution_parity_of_f_l838_83883

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a/x

theorem inequality_solution :
  (∀ x, 0 < x ∧ x < 1 ↔ f x 2 - f (x-1) 2 > 2*x - 1) :=
sorry

theorem parity_of_f :
  (∀ x ≠ 0, f (-x) 0 = f x 0) ∧
  (∀ a ≠ 0, ∃ x ≠ 0, f (-x) a ≠ f x a ∧ f (-x) a ≠ -f x a) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_parity_of_f_l838_83883


namespace NUMINAMATH_CALUDE_expected_value_is_six_point_five_l838_83889

/-- A fair 12-sided die with faces numbered from 1 to 12 -/
def twelve_sided_die : Finset ℕ := Finset.range 12

/-- The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 -/
def expected_value : ℚ :=
  (Finset.sum twelve_sided_die (λ i => i + 1)) / 12

/-- Theorem: The expected value of rolling a fair 12-sided die with faces numbered from 1 to 12 is 6.5 -/
theorem expected_value_is_six_point_five :
  expected_value = 13/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_six_point_five_l838_83889


namespace NUMINAMATH_CALUDE_problem_solution_l838_83801

theorem problem_solution (x : ℝ) : 3 * x + 15 = (1/3) * (6 * x + 45) → x - 5 = -5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l838_83801


namespace NUMINAMATH_CALUDE_opponents_total_score_l838_83803

def baseball_problem (team_scores : List ℕ) (games_lost : ℕ) : Prop :=
  let total_games := team_scores.length
  let lost_scores := team_scores.take games_lost
  let won_scores := team_scores.drop games_lost
  
  -- Conditions
  total_games = 7 ∧
  team_scores = [1, 3, 5, 6, 7, 8, 10] ∧
  games_lost = 3 ∧
  
  -- Lost games: opponent scored 2 more than the team
  (List.sum (lost_scores.map (· + 2))) +
  -- Won games: team scored 3 times opponent's score
  (List.sum (won_scores.map (· / 3))) = 24

theorem opponents_total_score :
  baseball_problem [1, 3, 5, 6, 7, 8, 10] 3 := by
  sorry

end NUMINAMATH_CALUDE_opponents_total_score_l838_83803


namespace NUMINAMATH_CALUDE_equation_solutions_l838_83808

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.sqrt (Real.sqrt x) = 15 / (8 - Real.sqrt (Real.sqrt x))

-- State the theorem
theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 625 ∨ x = 81) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l838_83808


namespace NUMINAMATH_CALUDE_complex_equation_solution_l838_83862

theorem complex_equation_solution (x₁ x₂ A : ℂ) (h_distinct : x₁ ≠ x₂)
  (h_eq1 : x₁ * (x₁ + 1) = A)
  (h_eq2 : x₂ * (x₂ + 1) = A)
  (h_eq3 : x₁^4 + 3*x₁^3 + 5*x₁ = x₂^4 + 3*x₂^3 + 5*x₂) :
  A = -7 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l838_83862


namespace NUMINAMATH_CALUDE_jerry_lawsuit_years_l838_83823

def salary_per_year : ℕ := 50000
def medical_bills : ℕ := 200000
def punitive_multiplier : ℕ := 3
def settlement_percentage : ℚ := 4/5
def total_received : ℕ := 5440000

def total_damages (Y : ℕ) : ℕ :=
  Y * salary_per_year + medical_bills + punitive_multiplier * (Y * salary_per_year + medical_bills)

theorem jerry_lawsuit_years :
  ∃ Y : ℕ, (↑total_received : ℚ) = settlement_percentage * (↑(total_damages Y) : ℚ) ∧ Y = 30 :=
by sorry

end NUMINAMATH_CALUDE_jerry_lawsuit_years_l838_83823


namespace NUMINAMATH_CALUDE_six_people_arrangement_l838_83897

def arrangement_count (n : ℕ) : ℕ := 
  (n.choose 2) * ((n-2).choose 2) * ((n-4).choose 2)

theorem six_people_arrangement : arrangement_count 6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l838_83897


namespace NUMINAMATH_CALUDE_arithmetic_progression_nth_term_l838_83811

theorem arithmetic_progression_nth_term (a d n : ℕ) (Tn : ℕ) 
  (h1 : a = 2) 
  (h2 : d = 8) 
  (h3 : Tn = 90) 
  (h4 : Tn = a + (n - 1) * d) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_nth_term_l838_83811


namespace NUMINAMATH_CALUDE_total_stones_l838_83816

/-- Represents the number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- Defines the conditions for the stone piles -/
def ValidStonePiles (p : StonePiles) : Prop :=
  p.pile5 = 6 * p.pile3 ∧
  p.pile2 = 2 * (p.pile3 + p.pile5) ∧
  p.pile1 = p.pile5 / 3 ∧
  p.pile1 = p.pile4 - 10 ∧
  p.pile4 = 2 * p.pile2

/-- The theorem to be proved -/
theorem total_stones (p : StonePiles) (h : ValidStonePiles p) : 
  p.pile1 + p.pile2 + p.pile3 + p.pile4 + p.pile5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_stones_l838_83816


namespace NUMINAMATH_CALUDE_speedster_convertibles_count_l838_83826

theorem speedster_convertibles_count (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) :
  speedsters = total / 3 →
  30 = total - speedsters →
  convertibles = (4 * speedsters) / 5 →
  convertibles = 12 := by
sorry

end NUMINAMATH_CALUDE_speedster_convertibles_count_l838_83826


namespace NUMINAMATH_CALUDE_constant_value_l838_83820

theorem constant_value (x : ℝ) (constant : ℝ) 
  (eq : 5 * x + 3 = 10 * x - constant) 
  (h : x = 5) : 
  constant = 22 := by
sorry

end NUMINAMATH_CALUDE_constant_value_l838_83820


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l838_83877

/-- Given a hyperbola with equation x^2 / (1+m) - y^2 / (3-m) = 1 and eccentricity > √2,
    prove that the range of m is (-1, 1) -/
theorem hyperbola_m_range (m : ℝ) :
  (∃ x y : ℝ, x^2 / (1 + m) - y^2 / (3 - m) = 1) ∧ 
  (2 / Real.sqrt (1 + m) > Real.sqrt 2) →
  -1 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l838_83877


namespace NUMINAMATH_CALUDE_greg_distance_when_azarah_finishes_l838_83838

/-- Represents the constant speed of a runner -/
structure Speed : Type :=
  (value : ℝ)
  (pos : value > 0)

/-- Calculates the distance traveled given speed and time -/
def distance (s : Speed) (t : ℝ) : ℝ := s.value * t

theorem greg_distance_when_azarah_finishes 
  (azarah_speed charlize_speed greg_speed : Speed)
  (h1 : distance azarah_speed 1 = 100)
  (h2 : distance charlize_speed 1 = 80)
  (h3 : distance charlize_speed (100 / charlize_speed.value) = 100)
  (h4 : distance greg_speed (100 / charlize_speed.value) = 90) :
  distance greg_speed (100 / azarah_speed.value) = 72 :=
sorry

end NUMINAMATH_CALUDE_greg_distance_when_azarah_finishes_l838_83838


namespace NUMINAMATH_CALUDE_capital_after_18_years_l838_83847

def initial_investment : ℝ := 2000
def increase_rate : ℝ := 0.5
def years_per_period : ℕ := 3
def total_years : ℕ := 18

theorem capital_after_18_years :
  let periods : ℕ := total_years / years_per_period
  let growth_factor : ℝ := 1 + increase_rate
  let final_capital : ℝ := initial_investment * growth_factor ^ periods
  final_capital = 22781.25 := by sorry

end NUMINAMATH_CALUDE_capital_after_18_years_l838_83847


namespace NUMINAMATH_CALUDE_gcd_g_x_l838_83833

def g (x : ℤ) : ℤ := (5*x+3)*(11*x+2)*(7*x+4)^2*(8*x+5)

theorem gcd_g_x (x : ℤ) (h : ∃ k : ℤ, x = 360 * k) : 
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 120 := by
sorry

end NUMINAMATH_CALUDE_gcd_g_x_l838_83833


namespace NUMINAMATH_CALUDE_negation_equivalence_l838_83858

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l838_83858


namespace NUMINAMATH_CALUDE_sin_phi_value_l838_83805

/-- Given two functions f and g, where f is shifted right by φ to obtain g,
    prove that sinφ = 24/25 -/
theorem sin_phi_value (f g : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = 3 * Real.sin x + 4 * Real.cos x) →
  (∀ x, g x = 3 * Real.sin x - 4 * Real.cos x) →
  (∀ x, g x = f (x - φ)) →
  Real.sin φ = 24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_phi_value_l838_83805


namespace NUMINAMATH_CALUDE_points_per_round_l838_83873

theorem points_per_round (total_points : ℕ) (num_rounds : ℕ) (points_per_round : ℕ) 
  (h1 : total_points = 84)
  (h2 : num_rounds = 2)
  (h3 : total_points = num_rounds * points_per_round) :
  points_per_round = 42 := by
sorry

end NUMINAMATH_CALUDE_points_per_round_l838_83873


namespace NUMINAMATH_CALUDE_inverse_343_mod_103_l838_83827

theorem inverse_343_mod_103 (h : (7⁻¹ : ZMod 103) = 44) : (343⁻¹ : ZMod 103) = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_343_mod_103_l838_83827


namespace NUMINAMATH_CALUDE_fruit_filling_probability_is_five_eighths_l838_83829

/-- The number of fruit types available -/
def num_fruits : ℕ := 5

/-- The number of meat types available -/
def num_meats : ℕ := 4

/-- The number of ingredient types required for a filling -/
def ingredients_per_filling : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of making a mooncake with fruit filling -/
def fruit_filling_probability : ℚ :=
  choose num_fruits ingredients_per_filling /
  (choose num_fruits ingredients_per_filling + choose num_meats ingredients_per_filling)

theorem fruit_filling_probability_is_five_eighths :
  fruit_filling_probability = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fruit_filling_probability_is_five_eighths_l838_83829


namespace NUMINAMATH_CALUDE_candidate_A_votes_l838_83810

def total_votes : ℕ := 560000
def invalid_percentage : ℚ := 15 / 100
def candidate_A_percentage : ℚ := 85 / 100

theorem candidate_A_votes : 
  ⌊(1 - invalid_percentage) * candidate_A_percentage * total_votes⌋ = 404600 := by
  sorry

end NUMINAMATH_CALUDE_candidate_A_votes_l838_83810


namespace NUMINAMATH_CALUDE_part_one_part_two_l838_83849

-- Define the conditions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0

def q (x : ℝ) : Prop := x^2 - 6*x + 8 < 0 ∧ x^2 - 8*x + 15 > 0

-- Part 1
theorem part_one : 
  ∀ x : ℝ, p x 1 ∧ q x → 2 < x ∧ x < 3 :=
sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, q x → (∃ a : ℝ, a > 0 ∧ p x a)) ∧
  (∃ x : ℝ, (∃ a : ℝ, a > 0 ∧ p x a) ∧ ¬q x) ↔
  (∃ a : ℝ, 1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l838_83849


namespace NUMINAMATH_CALUDE_power_equation_solution_l838_83804

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^22 ↔ n = 21 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l838_83804


namespace NUMINAMATH_CALUDE_function_properties_l838_83817

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2 - 3 / 2

theorem function_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (A B C a b c : ℝ),
    f C = 0 →
    c = 3 →
    2 * Real.sin A - Real.sin B = 0 →
    a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C = c ^ 2 →
    a = Real.sqrt 3 ∧ b = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l838_83817


namespace NUMINAMATH_CALUDE_max_value_of_expression_l838_83895

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5*x + 6*y < 90) :
  ∃ (M : ℝ), M = 900 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → 5*a + 6*b < 90 → a*b*(90 - 5*a - 6*b) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l838_83895


namespace NUMINAMATH_CALUDE_elsa_angus_token_difference_l838_83834

/-- Calculates the difference in total token value between two people -/
def tokenValueDifference (elsa_tokens : ℕ) (angus_tokens : ℕ) (token_value : ℕ) : ℕ :=
  (elsa_tokens * token_value) - (angus_tokens * token_value)

/-- Proves that the difference in token value between Elsa and Angus is $20 -/
theorem elsa_angus_token_difference :
  tokenValueDifference 60 55 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_elsa_angus_token_difference_l838_83834


namespace NUMINAMATH_CALUDE_abc_equation_l838_83860

theorem abc_equation (a b c p : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq1 : a + 2/b = p)
  (h_eq2 : b + 2/c = p)
  (h_eq3 : c + 2/a = p) :
  a * b * c + 2 * p = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_equation_l838_83860


namespace NUMINAMATH_CALUDE_mindy_emails_l838_83843

theorem mindy_emails (e m : ℕ) (h1 : e = 9 * m - 7) (h2 : e + m = 93) : e = 83 := by
  sorry

end NUMINAMATH_CALUDE_mindy_emails_l838_83843


namespace NUMINAMATH_CALUDE_electricity_consumption_for_2_75_yuan_l838_83869

-- Define the relationship between electricity consumption and charges
def electricity_charge (consumption : ℝ) : ℝ := 0.55 * consumption

-- Theorem statement
theorem electricity_consumption_for_2_75_yuan :
  ∃ (consumption : ℝ), electricity_charge consumption = 2.75 ∧ consumption = 5 :=
sorry

end NUMINAMATH_CALUDE_electricity_consumption_for_2_75_yuan_l838_83869


namespace NUMINAMATH_CALUDE_exists_polygon_with_1980_degrees_l838_83866

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

/-- 1980 degrees is a valid sum of interior angles for some polygon -/
theorem exists_polygon_with_1980_degrees :
  ∃ (n : ℕ), sum_interior_angles n = 1980 :=
sorry

end NUMINAMATH_CALUDE_exists_polygon_with_1980_degrees_l838_83866


namespace NUMINAMATH_CALUDE_carls_yard_area_l838_83839

/-- Represents a rectangular yard with fence posts. -/
structure FencedYard where
  short_posts : ℕ  -- Number of posts on the shorter side
  long_posts : ℕ   -- Number of posts on the longer side
  post_spacing : ℕ -- Distance between adjacent posts in yards

/-- Calculates the total number of fence posts. -/
def total_posts (yard : FencedYard) : ℕ :=
  2 * (yard.short_posts + yard.long_posts) - 4

/-- Calculates the area of the fenced yard in square yards. -/
def yard_area (yard : FencedYard) : ℕ :=
  (yard.short_posts - 1) * (yard.long_posts - 1) * yard.post_spacing^2

/-- Theorem stating the area of Carl's yard. -/
theorem carls_yard_area :
  ∃ (yard : FencedYard),
    yard.short_posts = 4 ∧
    yard.long_posts = 12 ∧
    yard.post_spacing = 5 ∧
    total_posts yard = 24 ∧
    yard.long_posts = 3 * yard.short_posts ∧
    yard_area yard = 825 :=
by sorry

end NUMINAMATH_CALUDE_carls_yard_area_l838_83839


namespace NUMINAMATH_CALUDE_paving_stone_length_l838_83851

/-- Given a rectangular courtyard and paving stones with specific dimensions,
    calculate the length of each paving stone. -/
theorem paving_stone_length
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (total_stones : ℕ)
  (stone_width : ℝ)
  (h1 : courtyard_length = 30)
  (h2 : courtyard_width = 16)
  (h3 : total_stones = 240)
  (h4 : stone_width = 1)
  : (courtyard_length * courtyard_width) / (total_stones * stone_width) = 2 := by
  sorry

#check paving_stone_length

end NUMINAMATH_CALUDE_paving_stone_length_l838_83851


namespace NUMINAMATH_CALUDE_john_plays_three_times_a_month_l838_83898

/-- The number of times John plays paintball in a month -/
def plays_per_month : ℕ := sorry

/-- The number of boxes John buys each time he plays -/
def boxes_per_play : ℕ := 3

/-- The cost of each box of paintballs in dollars -/
def cost_per_box : ℕ := 25

/-- The total amount John spends on paintballs per month in dollars -/
def total_spent_per_month : ℕ := 225

/-- Theorem stating that John plays paintball 3 times a month -/
theorem john_plays_three_times_a_month : 
  plays_per_month = 3 ∧ 
  plays_per_month * boxes_per_play * cost_per_box = total_spent_per_month := by
  sorry

end NUMINAMATH_CALUDE_john_plays_three_times_a_month_l838_83898


namespace NUMINAMATH_CALUDE_tree_height_after_four_months_l838_83894

/-- Calculates the height of a tree after a given number of months -/
def tree_height (initial_height : ℕ) (growth_rate : ℕ) (growth_period : ℕ) (months : ℕ) : ℕ :=
  initial_height * 100 + (months * 4 / growth_period) * growth_rate

/-- Theorem stating that a tree with given growth parameters reaches 600 cm after 4 months -/
theorem tree_height_after_four_months :
  tree_height 2 50 2 4 = 600 := by
  sorry

#eval tree_height 2 50 2 4

end NUMINAMATH_CALUDE_tree_height_after_four_months_l838_83894


namespace NUMINAMATH_CALUDE_number_of_apricot_trees_apricot_trees_count_l838_83831

/-- Proves that the number of apricot trees is 135, given the conditions stated in the problem. -/
theorem number_of_apricot_trees : ℕ → Prop :=
  fun n : ℕ =>
    (∃ peach_trees : ℕ,
      peach_trees = 300 ∧
      peach_trees = 2 * n + 30) →
    n = 135

/-- The main theorem stating that there are 135 apricot trees. -/
theorem apricot_trees_count : ∃ n : ℕ, number_of_apricot_trees n :=
  sorry

end NUMINAMATH_CALUDE_number_of_apricot_trees_apricot_trees_count_l838_83831


namespace NUMINAMATH_CALUDE_fraction_equality_l838_83871

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 5 / 8) :
  (b - a) / a = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l838_83871


namespace NUMINAMATH_CALUDE_fraction_of_25_l838_83881

theorem fraction_of_25 : ∃ x : ℚ, x * 25 = 0.9 * 40 - 16 ∧ x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_25_l838_83881


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l838_83876

theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : ∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m) 
  (h2 : 3 * a 3 ^ 2 - 25 * a 3 + 27 = 0) (h3 : 3 * a 11 ^ 2 - 25 * a 11 + 27 = 0) : a 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l838_83876


namespace NUMINAMATH_CALUDE_smallest_a_is_9_l838_83841

-- Define the arithmetic sequence
def is_arithmetic_sequence (a b c : ℕ) : Prop := b - a = c - b

-- Define the function f
def f (a b c : ℕ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem smallest_a_is_9 
  (a b c : ℕ) 
  (r s : ℝ) 
  (h_arith : is_arithmetic_sequence a b c)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a < b ∧ b < c)
  (h_f_r : f a b c r = s)
  (h_f_s : f a b c s = r)
  (h_rs : r * s = 2017)
  (h_distinct : r ≠ s) :
  ∀ a' : ℕ, (∃ b' c' : ℕ, 
    is_arithmetic_sequence a' b' c' ∧ 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    a' < b' ∧ b' < c' ∧
    (∃ r' s' : ℝ, f a' b' c' r' = s' ∧ f a' b' c' s' = r' ∧ r' * s' = 2017 ∧ r' ≠ s')) →
  a' ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_9_l838_83841


namespace NUMINAMATH_CALUDE_differential_equation_satisfied_l838_83872

theorem differential_equation_satisfied 
  (x c : ℝ) 
  (y : ℝ → ℝ)
  (h1 : ∀ x, y x = 2 + c * Real.sqrt (1 - x^2))
  (h2 : Differentiable ℝ y) :
  (1 - x^2) * (deriv y x) + x * (y x) = 2 * x :=
by sorry

end NUMINAMATH_CALUDE_differential_equation_satisfied_l838_83872


namespace NUMINAMATH_CALUDE_tan_alpha_value_l838_83837

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin α + Real.cos α = (1 - Real.sqrt 3) / 2) : 
  Real.tan α = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l838_83837


namespace NUMINAMATH_CALUDE_slower_traveler_speed_l838_83864

/-- Proves that given two people traveling in opposite directions for 1.5 hours,
    where one travels 3 miles per hour faster than the other, and they end up 19.5 miles apart,
    the slower person's speed is 5 miles per hour. -/
theorem slower_traveler_speed
  (time : ℝ)
  (distance_apart : ℝ)
  (speed_difference : ℝ)
  (h1 : time = 1.5)
  (h2 : distance_apart = 19.5)
  (h3 : speed_difference = 3)
  : ∃ (slower_speed : ℝ), slower_speed = 5 ∧
    distance_apart = time * (slower_speed + (slower_speed + speed_difference)) :=
by sorry

end NUMINAMATH_CALUDE_slower_traveler_speed_l838_83864


namespace NUMINAMATH_CALUDE_no_real_solution_l838_83809

theorem no_real_solution (a b c : ℝ) : ¬∃ (x y z : ℝ), 
  (a^2 + b^2 + c^2 + 3*(x^2 + y^2 + z^2) = 6) ∧ (a*x + b*y + c*z = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l838_83809


namespace NUMINAMATH_CALUDE_mr_orange_yield_l838_83888

/-- Calculates the expected orange yield from a triangular garden --/
def expected_orange_yield (base_paces : ℕ) (height_paces : ℕ) (feet_per_pace : ℕ) (yield_per_sqft : ℚ) : ℚ :=
  let base_feet := base_paces * feet_per_pace
  let height_feet := height_paces * feet_per_pace
  let area := (base_feet * height_feet : ℚ) / 2
  area * yield_per_sqft

/-- Theorem stating the expected orange yield for Mr. Orange's garden --/
theorem mr_orange_yield :
  expected_orange_yield 18 24 3 (3/4) = 1458 := by
  sorry

end NUMINAMATH_CALUDE_mr_orange_yield_l838_83888


namespace NUMINAMATH_CALUDE_integral_even_function_l838_83899

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem integral_even_function 
  (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_integral : ∫ x in (0:ℝ)..6, f x = 8) : 
  ∫ x in (-6:ℝ)..6, f x = 16 := by
  sorry

end NUMINAMATH_CALUDE_integral_even_function_l838_83899


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_is_nine_l838_83835

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a * b > 0) (h2 : a + 4 * b = 1) :
  ∀ x y : ℝ, x * y > 0 ∧ x + 4 * y = 1 → 1 / x + 1 / y ≥ 1 / a + 1 / b :=
by sorry

theorem min_value_is_nine (a b : ℝ) (h1 : a * b > 0) (h2 : a + 4 * b = 1) :
  1 / a + 1 / b = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_is_nine_l838_83835


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l838_83884

def f (x : ℝ) : ℝ := x^3 + x

theorem f_strictly_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l838_83884


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l838_83813

/-- An isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  perimeter_eq : base + 2 * leg = 20

/-- The lengths of the sides when each leg is twice the base -/
def legsTwiceBase (t : IsoscelesTriangle) : Prop :=
  t.leg = 2 * t.base ∧ t.base = 4 ∧ t.leg = 8

/-- The lengths of the sides when one side is 6 -/
def oneSideSix (t : IsoscelesTriangle) : Prop :=
  (t.base = 6 ∧ t.leg = 7) ∨ (t.base = 8 ∧ t.leg = 6)

theorem isosceles_triangle_sides :
  (∀ t : IsoscelesTriangle, t.leg = 2 * t.base → legsTwiceBase t) ∧
  (∀ t : IsoscelesTriangle, (t.base = 6 ∨ t.leg = 6) → oneSideSix t) := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l838_83813


namespace NUMINAMATH_CALUDE_electricity_fee_properties_l838_83818

-- Define the relationship between electricity usage and fee
def electricity_fee (x : ℝ) : ℝ := 0.55 * x

-- Theorem stating the properties of the electricity fee function
theorem electricity_fee_properties :
  -- 1. x is independent, y is dependent (implicit in the function definition)
  -- 2. For every increase of 1 in x, y increases by 0.55
  (∀ x : ℝ, electricity_fee (x + 1) = electricity_fee x + 0.55) ∧
  -- 3. When x = 8, y = 4.4
  (electricity_fee 8 = 4.4) ∧
  -- 4. When y = 3.75, x ≠ 7
  (∀ x : ℝ, electricity_fee x = 3.75 → x ≠ 7) := by
  sorry


end NUMINAMATH_CALUDE_electricity_fee_properties_l838_83818


namespace NUMINAMATH_CALUDE_least_tiles_for_room_l838_83886

theorem least_tiles_for_room (room_length room_width : ℕ) : 
  room_length = 7550 → room_width = 2085 → 
  (∃ (tile_size : ℕ), 
    tile_size > 0 ∧ 
    room_length % tile_size = 0 ∧ 
    room_width % tile_size = 0 ∧
    (∀ (larger_tile : ℕ), larger_tile > tile_size → 
      room_length % larger_tile ≠ 0 ∨ room_width % larger_tile ≠ 0) ∧
    (room_length * room_width) / (tile_size * tile_size) = 630270) :=
by sorry

end NUMINAMATH_CALUDE_least_tiles_for_room_l838_83886


namespace NUMINAMATH_CALUDE_equilateral_triangle_on_curve_l838_83815

def curve (x : ℝ) : ℝ := -2 * x^2

theorem equilateral_triangle_on_curve :
  ∃ (P Q : ℝ × ℝ),
    (P.2 = curve P.1) ∧
    (Q.2 = curve Q.1) ∧
    (P.1 = -Q.1) ∧
    (P.2 = Q.2) ∧
    (dist P (0, 0) = dist Q (0, 0)) ∧
    (dist P Q = dist P (0, 0)) ∧
    (dist P (0, 0) = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_on_curve_l838_83815


namespace NUMINAMATH_CALUDE_regions_theorem_l838_83856

/-- The number of regions formed by n lines in a plane -/
def total_regions (n : ℕ) : ℚ :=
  (n^2 + n + 2) / 2

/-- The number of bounded regions formed by n lines in a plane -/
def bounded_regions (n : ℕ) : ℚ :=
  (n^2 - 3*n + 2) / 2

/-- Theorem stating the formulas for total and bounded regions -/
theorem regions_theorem (n : ℕ) :
  (total_regions n = (n^2 + n + 2) / 2) ∧
  (bounded_regions n = (n^2 - 3*n + 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_regions_theorem_l838_83856


namespace NUMINAMATH_CALUDE_compute_b_l838_83850

-- Define the polynomial
def f (a b : ℚ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 21

-- State the theorem
theorem compute_b (a b : ℚ) :
  (f a b (3 + Real.sqrt 5) = 0) → b = -27.5 := by
  sorry

end NUMINAMATH_CALUDE_compute_b_l838_83850


namespace NUMINAMATH_CALUDE_probability_divisible_by_45_is_zero_l838_83874

def digits : List Nat := [1, 3, 3, 4, 5, 9]

def is_divisible_by_45 (n : Nat) : Prop :=
  n % 45 = 0

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length = 6 ∧ arr.toFinset = digits.toFinset

def to_number (arr : List Nat) : Nat :=
  arr.foldl (fun acc d => acc * 10 + d) 0

theorem probability_divisible_by_45_is_zero :
  ∀ arr : List Nat, is_valid_arrangement arr →
    ¬(is_divisible_by_45 (to_number arr)) :=
sorry

end NUMINAMATH_CALUDE_probability_divisible_by_45_is_zero_l838_83874


namespace NUMINAMATH_CALUDE_lunch_group_size_l838_83852

/-- The number of people having lunch, including Benny -/
def num_people : ℕ := 3

/-- The cost of one lunch special in dollars -/
def lunch_cost : ℕ := 8

/-- The total bill in dollars -/
def total_bill : ℕ := 24

theorem lunch_group_size :
  num_people * lunch_cost = total_bill :=
by sorry

end NUMINAMATH_CALUDE_lunch_group_size_l838_83852


namespace NUMINAMATH_CALUDE_train_crossing_time_l838_83825

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 400 ∧ 
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) ∧
  crossing_time = 10 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l838_83825


namespace NUMINAMATH_CALUDE_total_pictures_taken_l838_83821

def pictures_already_taken : ℕ := 28
def pictures_at_dolphin_show : ℕ := 16

theorem total_pictures_taken : 
  pictures_already_taken + pictures_at_dolphin_show = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_pictures_taken_l838_83821


namespace NUMINAMATH_CALUDE_gary_remaining_money_l838_83892

/-- The amount of money Gary has left after buying a pet snake -/
def money_left (initial_amount spent_amount : ℕ) : ℕ :=
  initial_amount - spent_amount

/-- Theorem stating that Gary has 18 dollars left after buying a pet snake -/
theorem gary_remaining_money :
  money_left 73 55 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gary_remaining_money_l838_83892


namespace NUMINAMATH_CALUDE_log_equation_solution_l838_83859

theorem log_equation_solution :
  ∀ y : ℝ, (Real.log y / Real.log 9 = Real.log 8 / Real.log 2) → y = 729 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l838_83859


namespace NUMINAMATH_CALUDE_equation_solution_l838_83861

theorem equation_solution : 
  {x : ℝ | (5 - 2*x)^(x + 1) = 1} = {-1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l838_83861


namespace NUMINAMATH_CALUDE_vegetarian_eaters_count_l838_83882

/-- Represents the eating habits in a family -/
structure FamilyDiet where
  onlyVegetarian : ℕ
  onlyNonVegetarian : ℕ
  both : ℕ

/-- Calculates the total number of people who eat vegetarian food -/
def vegetarianEaters (f : FamilyDiet) : ℕ :=
  f.onlyVegetarian + f.both

/-- Theorem: Given the family diet information, prove that the number of vegetarian eaters
    is the sum of those who eat only vegetarian and those who eat both -/
theorem vegetarian_eaters_count (f : FamilyDiet) 
    (h1 : f.onlyVegetarian = 13)
    (h2 : f.onlyNonVegetarian = 7)
    (h3 : f.both = 8) :
    vegetarianEaters f = 21 := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_count_l838_83882


namespace NUMINAMATH_CALUDE_letters_with_both_count_l838_83845

/-- Represents the number of letters in the alphabet. -/
def total_letters : ℕ := 40

/-- Represents the number of letters with a straight line but no dot. -/
def line_only : ℕ := 24

/-- Represents the number of letters with a dot but no straight line. -/
def dot_only : ℕ := 6

/-- Represents the number of letters with both a dot and a straight line. -/
def both : ℕ := total_letters - line_only - dot_only

theorem letters_with_both_count :
  both = 10 :=
sorry

end NUMINAMATH_CALUDE_letters_with_both_count_l838_83845


namespace NUMINAMATH_CALUDE_rectangle_area_comparison_l838_83814

theorem rectangle_area_comparison (a b : ℝ) (ha : a = 8) (hb : b = 15) : 
  let d := Real.sqrt (a^2 + b^2)
  let new_rectangle_area := (d + b) * (d - b)
  let square_area := (a + b)^2
  new_rectangle_area ≠ square_area := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_comparison_l838_83814


namespace NUMINAMATH_CALUDE_small_perturbation_approximation_l838_83824

/-- For small α and β, (1 + α)(1 + β) ≈ 1 + α + β -/
theorem small_perturbation_approximation (α β : ℝ) (hα : |α| < 1) (hβ : |β| < 1) :
  ∃ ε > 0, |(1 + α) * (1 + β) - (1 + α + β)| < ε := by
  sorry

end NUMINAMATH_CALUDE_small_perturbation_approximation_l838_83824


namespace NUMINAMATH_CALUDE_perfect_square_prob_l838_83800

/-- A function that represents the number of ways to roll a 10-sided die n times
    such that the product of the rolls is a perfect square -/
def b : ℕ → ℕ
  | 0 => 1
  | n + 1 => 10^n + 2 * b n

/-- The probability of rolling a 10-sided die 4 times and getting a product
    that is a perfect square -/
def prob_perfect_square : ℚ :=
  b 4 / 10^4

theorem perfect_square_prob :
  prob_perfect_square = 316 / 2500 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_prob_l838_83800


namespace NUMINAMATH_CALUDE_correct_equations_l838_83875

/-- Represents the money held by a person -/
structure Money where
  amount : ℚ
  deriving Repr

/-- The problem setup -/
def problem_setup (a b : Money) : Prop :=
  (a.amount + (1/2) * b.amount = 50) ∧
  ((2/3) * a.amount + b.amount = 50)

/-- The theorem to prove -/
theorem correct_equations (a b : Money) :
  problem_setup a b ↔
  (a.amount + (1/2) * b.amount = 50 ∧ (2/3) * a.amount + b.amount = 50) :=
by sorry

end NUMINAMATH_CALUDE_correct_equations_l838_83875
