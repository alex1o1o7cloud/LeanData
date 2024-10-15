import Mathlib

namespace NUMINAMATH_CALUDE_regular_polygons_ratio_l3760_376038

/-- The interior angle of a regular polygon with n sides -/
def interior_angle (n : ℕ) : ℚ := 180 - 360 / n

/-- The theorem statement -/
theorem regular_polygons_ratio (r k : ℕ) : 
  (r > 2 ∧ k > 2) →  -- Ensure polygons have at least 3 sides
  (interior_angle r / interior_angle k = 5 / 3) →
  (r = 2 * k) →
  (r = 8 ∧ k = 4) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygons_ratio_l3760_376038


namespace NUMINAMATH_CALUDE_four_times_angle_triangle_l3760_376067

theorem four_times_angle_triangle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  (α = 40 ∧ β = 4 * γ) ∨ (α = 40 ∧ γ = 4 * β) ∨ (β = 40 ∧ α = 4 * γ) →  -- One angle is 40° and another is 4 times the third
  ((β = 130 ∧ γ = 10) ∨ (β = 112 ∧ γ = 28)) ∨ 
  ((α = 130 ∧ γ = 10) ∨ (α = 112 ∧ γ = 28)) ∨ 
  ((α = 130 ∧ β = 10) ∨ (α = 112 ∧ β = 28)) :=
by sorry

end NUMINAMATH_CALUDE_four_times_angle_triangle_l3760_376067


namespace NUMINAMATH_CALUDE_collector_problem_l3760_376068

/-- The number of items in the collection --/
def n : ℕ := 10

/-- The probability of finding each item --/
def p : ℝ := 0.1

/-- The probability of having exactly k items missing in the second collection
    when the first collection is complete --/
def prob_missing (k : ℕ) : ℝ := sorry

theorem collector_problem :
  (prob_missing 1 = prob_missing 2) ∧
  (∀ k ∈ Finset.range 9, prob_missing (k + 2) > prob_missing (k + 3)) :=
sorry

end NUMINAMATH_CALUDE_collector_problem_l3760_376068


namespace NUMINAMATH_CALUDE_deepak_current_age_l3760_376089

/-- Represents the ages of Rahul and Deepak -/
structure Ages where
  rahul : ℕ
  deepak : ℕ

/-- The condition that the ratio of Rahul's age to Deepak's age is 4:3 -/
def ratio_condition (ages : Ages) : Prop :=
  4 * ages.deepak = 3 * ages.rahul

/-- The condition that Rahul will be 26 years old in 6 years -/
def future_condition (ages : Ages) : Prop :=
  ages.rahul + 6 = 26

/-- The theorem stating Deepak's current age given the conditions -/
theorem deepak_current_age (ages : Ages) 
  (h1 : ratio_condition ages) 
  (h2 : future_condition ages) : 
  ages.deepak = 15 := by
  sorry

end NUMINAMATH_CALUDE_deepak_current_age_l3760_376089


namespace NUMINAMATH_CALUDE_first_car_speed_l3760_376003

/-- Proves that the speed of the first car is 40 miles per hour given the conditions of the problem -/
theorem first_car_speed (black_car_speed : ℝ) (initial_distance : ℝ) (overtake_time : ℝ) : 
  black_car_speed = 50 →
  initial_distance = 10 →
  overtake_time = 1 →
  (black_car_speed * overtake_time - initial_distance) / overtake_time = 40 :=
by sorry

end NUMINAMATH_CALUDE_first_car_speed_l3760_376003


namespace NUMINAMATH_CALUDE_percentage_increase_l3760_376010

theorem percentage_increase (N : ℝ) (P : ℝ) : 
  N = 40 →
  N + (P / 100) * N - (N - (30 / 100) * N) = 22 →
  P = 25 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l3760_376010


namespace NUMINAMATH_CALUDE_symmetry_condition_l3760_376009

/-- Given a curve y = (2px + q) / (rx - 2s) where p, q, r, s are nonzero real numbers,
    if the line y = x is an axis of symmetry for this curve, then r - 2s = 0. -/
theorem symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (2*p*x + q) / (r*x - 2*s) ↔ x = (2*p*y + q) / (r*y - 2*s)) →
  r - 2*s = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_condition_l3760_376009


namespace NUMINAMATH_CALUDE_trapezoid_bisector_length_l3760_376024

/-- 
Given a trapezoid with parallel sides of length a and c,
the length of a segment parallel to these sides that bisects the trapezoid's area
is √((a² + c²) / 2).
-/
theorem trapezoid_bisector_length (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  ∃ x : ℝ, x > 0 ∧ x^2 = (a^2 + c^2) / 2 ∧
  (∀ m : ℝ, m > 0 → (a + c) * m / 2 = (x + c) * (2 * m / (c + x)) / 2 + (x + a) * (2 * m / (a + x)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_bisector_length_l3760_376024


namespace NUMINAMATH_CALUDE_inequality_proof_l3760_376001

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x*y + y*z + z*x ≤ 1) : 
  (x + 1/x) * (y + 1/y) * (z + 1/z) ≥ 8 * (x + y) * (y + z) * (z + x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3760_376001


namespace NUMINAMATH_CALUDE_three_digit_numbers_after_exclusion_l3760_376018

/-- The count of three-digit numbers (100 to 999) -/
def total_three_digit_numbers : ℕ := 900

/-- The count of numbers in the form ABA where A and B are digits and A ≠ 0 -/
def count_ABA : ℕ := 81

/-- The count of numbers in the form AAB or BAA where A and B are digits and A ≠ 0 -/
def count_AAB_BAA : ℕ := 81

/-- The total count of excluded numbers -/
def total_excluded : ℕ := count_ABA + count_AAB_BAA

theorem three_digit_numbers_after_exclusion :
  total_three_digit_numbers - total_excluded = 738 := by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_after_exclusion_l3760_376018


namespace NUMINAMATH_CALUDE_girls_distance_calculation_l3760_376002

/-- The number of laps run by boys -/
def boys_laps : ℕ := 124

/-- The additional laps run by girls compared to boys -/
def extra_girls_laps : ℕ := 48

/-- The fraction of a mile per lap -/
def mile_per_lap : ℚ := 5 / 13

/-- The distance run by girls in miles -/
def girls_distance : ℚ := (boys_laps + extra_girls_laps) * mile_per_lap

theorem girls_distance_calculation :
  girls_distance = (124 + 48) * (5 / 13) := by sorry

end NUMINAMATH_CALUDE_girls_distance_calculation_l3760_376002


namespace NUMINAMATH_CALUDE_non_congruent_triangles_count_l3760_376048

-- Define the type for 2D points
structure Point where
  x : ℝ
  y : ℝ

-- Define the set of points
def points : List Point := [
  ⟨0, 0⟩, ⟨1, 0⟩, ⟨2, 0⟩,
  ⟨0, 1⟩, ⟨1, 1⟩, ⟨2, 1⟩,
  ⟨0.5, 2⟩, ⟨1.5, 2⟩, ⟨2.5, 2⟩
]

-- Function to check if two triangles are congruent
def are_congruent (t1 t2 : List Point) : Prop := sorry

-- Function to count non-congruent triangles
def count_non_congruent_triangles (pts : List Point) : ℕ := sorry

-- Theorem stating the number of non-congruent triangles
theorem non_congruent_triangles_count :
  count_non_congruent_triangles points = 18 := by sorry

end NUMINAMATH_CALUDE_non_congruent_triangles_count_l3760_376048


namespace NUMINAMATH_CALUDE_difference_of_squares_72_48_l3760_376060

theorem difference_of_squares_72_48 : 72^2 - 48^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_72_48_l3760_376060


namespace NUMINAMATH_CALUDE_geli_workout_days_l3760_376020

/-- Calculates the total number of push-ups for a given number of days -/
def totalPushUps (initialPushUps : ℕ) (dailyIncrease : ℕ) (days : ℕ) : ℕ :=
  days * initialPushUps + (days * (days - 1) * dailyIncrease) / 2

/-- Proves that Geli works out 3 times a week -/
theorem geli_workout_days : 
  ∃ (days : ℕ), days > 0 ∧ totalPushUps 10 5 days = 45 ∧ days = 3 := by
  sorry

#eval totalPushUps 10 5 3

end NUMINAMATH_CALUDE_geli_workout_days_l3760_376020


namespace NUMINAMATH_CALUDE_average_blanket_price_l3760_376028

/-- The average price of blankets given specific purchase conditions -/
theorem average_blanket_price : 
  let blanket_group1 := (3, 100)  -- (quantity, price)
  let blanket_group2 := (5, 150)
  let blanket_group3 := (2, 275)  -- 550 / 2 = 275
  let total_blankets := blanket_group1.1 + blanket_group2.1 + blanket_group3.1
  let total_cost := blanket_group1.1 * blanket_group1.2 + 
                    blanket_group2.1 * blanket_group2.2 + 
                    blanket_group3.1 * blanket_group3.2
  (total_cost / total_blankets : ℚ) = 160 := by
  sorry

end NUMINAMATH_CALUDE_average_blanket_price_l3760_376028


namespace NUMINAMATH_CALUDE_solution_set_abs_fraction_l3760_376090

theorem solution_set_abs_fraction (x : ℝ) : 
  (|x / (x - 1)| = x / (x - 1)) ↔ (x ≤ 0 ∨ x > 1) :=
sorry

end NUMINAMATH_CALUDE_solution_set_abs_fraction_l3760_376090


namespace NUMINAMATH_CALUDE_exam_score_calculation_l3760_376087

/-- Calculate total marks in an exam with penalties for incorrect answers -/
theorem exam_score_calculation 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (marks_per_correct : ℕ) 
  (penalty_per_wrong : ℕ) :
  total_questions = 60 →
  correct_answers = 36 →
  marks_per_correct = 4 →
  penalty_per_wrong = 1 →
  (correct_answers * marks_per_correct) - 
  ((total_questions - correct_answers) * penalty_per_wrong) = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l3760_376087


namespace NUMINAMATH_CALUDE_number_equation_solution_l3760_376079

theorem number_equation_solution : 
  ∃ x : ℝ, x^2 + 95 = (x - 20)^2 ∧ x = 7.625 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3760_376079


namespace NUMINAMATH_CALUDE_power_function_increasing_iff_m_eq_two_l3760_376012

/-- A power function f(x) = (m^2 - m - 1)x^m is increasing on (0, +∞) if and only if m = 2 -/
theorem power_function_increasing_iff_m_eq_two (m : ℝ) :
  (∀ x > 0, StrictMono (fun x => (m^2 - m - 1) * x^m)) ↔ m = 2 :=
by sorry

end NUMINAMATH_CALUDE_power_function_increasing_iff_m_eq_two_l3760_376012


namespace NUMINAMATH_CALUDE_inequality_proof_l3760_376030

theorem inequality_proof (x : ℝ) : (x^2 - 16) / (x^2 + 10*x + 25) < 0 ↔ -4 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3760_376030


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3760_376026

theorem cube_root_equation_solution (y : ℝ) :
  (5 - 2 / y) ^ (1/3 : ℝ) = -3 → y = 1/16 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3760_376026


namespace NUMINAMATH_CALUDE_hyperbola_slope_theorem_l3760_376000

/-- A hyperbola passing through specific points with given asymptote slopes -/
structure Hyperbola where
  -- Points the hyperbola passes through
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ
  point4 : ℝ × ℝ
  -- Slope of one asymptote
  slope1 : ℚ
  -- Slope of the other asymptote
  slope2 : ℚ
  -- Condition that the hyperbola passes through the given points
  passes_through : point1 = (2, 5) ∧ point2 = (7, 3) ∧ point3 = (1, 1) ∧ point4 = (10, 10)
  -- Condition that slope1 is 20/17
  slope1_value : slope1 = 20/17
  -- Condition that the product of slopes is -1
  slopes_product : slope1 * slope2 = -1

theorem hyperbola_slope_theorem (h : Hyperbola) :
  h.slope2 = -17/20 ∧ (100 * 17 + 20 = 1720) := by
  sorry

#check hyperbola_slope_theorem

end NUMINAMATH_CALUDE_hyperbola_slope_theorem_l3760_376000


namespace NUMINAMATH_CALUDE_floor_neg_five_thirds_l3760_376065

theorem floor_neg_five_thirds : ⌊(-5/3 : ℚ)⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_neg_five_thirds_l3760_376065


namespace NUMINAMATH_CALUDE_product_divisible_by_seven_l3760_376088

theorem product_divisible_by_seven :
  (7 * 17 * 27 * 37 * 47 * 57 * 67) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_seven_l3760_376088


namespace NUMINAMATH_CALUDE_smaller_root_of_quadratic_l3760_376073

theorem smaller_root_of_quadratic (x : ℚ) : 
  (x - 2/3) * (x - 5/6) + (x - 2/3) * (x - 2/3) - 1 = 0 →
  x = -1/12 ∨ x = 4/3 ∧ 
  -1/12 < 4/3 :=
sorry

end NUMINAMATH_CALUDE_smaller_root_of_quadratic_l3760_376073


namespace NUMINAMATH_CALUDE_pop_survey_result_l3760_376054

/-- Given a survey of 600 people where the central angle for "Pop" is 270°
    (to the nearest whole degree), prove that 450 people chose "Pop". -/
theorem pop_survey_result (total : ℕ) (angle : ℕ) (h_total : total = 600) (h_angle : angle = 270) :
  ∃ (pop : ℕ), pop = 450 ∧ 
  (pop : ℝ) / total * 360 ≥ angle - 0.5 ∧
  (pop : ℝ) / total * 360 < angle + 0.5 :=
by sorry

end NUMINAMATH_CALUDE_pop_survey_result_l3760_376054


namespace NUMINAMATH_CALUDE_selling_price_ratio_l3760_376084

theorem selling_price_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l3760_376084


namespace NUMINAMATH_CALUDE_certain_number_problem_l3760_376034

theorem certain_number_problem (x N : ℤ) (h1 : 3 * x = (N - x) + 18) (h2 : x = 11) : N = 26 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3760_376034


namespace NUMINAMATH_CALUDE_total_cost_is_49_27_l3760_376080

/-- Represents the cost of tickets for a family outing to a theme park -/
def theme_park_tickets : ℝ → Prop :=
  λ total_cost : ℝ =>
    ∃ (regular_price : ℝ),
      -- A senior ticket (30% discount) costs $7.50
      0.7 * regular_price = 7.5 ∧
      -- Total cost calculation
      total_cost = 2 * 7.5 + -- Two senior tickets
                   2 * regular_price + -- Two regular tickets
                   2 * (0.6 * regular_price) -- Two children tickets (40% discount)

/-- The total cost for all tickets is $49.27 -/
theorem total_cost_is_49_27 : theme_park_tickets 49.27 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_49_27_l3760_376080


namespace NUMINAMATH_CALUDE_simplify_and_ratio_l3760_376044

theorem simplify_and_ratio (m : ℝ) : 
  (6*m + 12) / 6 = m + 2 ∧ (1 : ℝ) / 2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_ratio_l3760_376044


namespace NUMINAMATH_CALUDE_boys_on_trip_l3760_376076

/-- Calculates the number of boys on a family trip given the specified conditions. -/
def number_of_boys (adults : ℕ) (total_eggs : ℕ) (eggs_per_adult : ℕ) (girls : ℕ) (eggs_per_girl : ℕ) : ℕ :=
  let eggs_for_children := total_eggs - adults * eggs_per_adult
  let eggs_for_girls := girls * eggs_per_girl
  let eggs_for_boys := eggs_for_children - eggs_for_girls
  let eggs_per_boy := eggs_per_girl + 1
  eggs_for_boys / eggs_per_boy

/-- Theorem stating that the number of boys on the trip is 10 under the given conditions. -/
theorem boys_on_trip :
  number_of_boys 3 (3 * 12) 3 7 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_boys_on_trip_l3760_376076


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3760_376082

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ + m = 0 ∧ x₂^2 + 4*x₂ + m = 0) → m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3760_376082


namespace NUMINAMATH_CALUDE_student_group_size_l3760_376033

/-- The number of students in a group with overlapping class registrations --/
def num_students (history math english all_three two_classes : ℕ) : ℕ :=
  history + math + english - two_classes - 2 * all_three + all_three

theorem student_group_size :
  let history := 19
  let math := 14
  let english := 26
  let all_three := 3
  let two_classes := 7
  num_students history math english all_three two_classes = 46 := by
  sorry

end NUMINAMATH_CALUDE_student_group_size_l3760_376033


namespace NUMINAMATH_CALUDE_quadratic_coefficient_value_l3760_376046

theorem quadratic_coefficient_value (b : ℝ) (n : ℝ) : 
  b > 0 → 
  (∀ x, x^2 + b*x + 88 = (x + n)^2 + 16) → 
  b = 12 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_value_l3760_376046


namespace NUMINAMATH_CALUDE_always_greater_than_m_l3760_376071

theorem always_greater_than_m (m : ℚ) : m + 2 > m := by
  sorry

end NUMINAMATH_CALUDE_always_greater_than_m_l3760_376071


namespace NUMINAMATH_CALUDE_tank_filling_time_l3760_376016

theorem tank_filling_time (fill_rate : ℝ) (leak_rate : ℝ) (fill_time_no_leak : ℝ) (empty_time_leak : ℝ) :
  fill_rate = 1 / fill_time_no_leak →
  leak_rate = 1 / empty_time_leak →
  fill_time_no_leak = 8 →
  empty_time_leak = 72 →
  (1 : ℝ) / (fill_rate - leak_rate) = 9 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l3760_376016


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_seventeen_is_dual_palindrome_smallest_dual_palindrome_is_17_l3760_376031

/-- Checks if a number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 15 → 
  (isPalindrome n 2 ∧ isPalindrome n 4) → 
  n ≥ 17 := by sorry

theorem seventeen_is_dual_palindrome : 
  isPalindrome 17 2 ∧ isPalindrome 17 4 := by sorry

theorem smallest_dual_palindrome_is_17 : 
  ∀ n : ℕ, n > 15 → 
  (isPalindrome n 2 ∧ isPalindrome n 4) → 
  n = 17 := by sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_seventeen_is_dual_palindrome_smallest_dual_palindrome_is_17_l3760_376031


namespace NUMINAMATH_CALUDE_negative_quadratic_range_l3760_376011

theorem negative_quadratic_range (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (a > 2 ∨ a < -2) := by sorry

end NUMINAMATH_CALUDE_negative_quadratic_range_l3760_376011


namespace NUMINAMATH_CALUDE_system_solution_l3760_376099

theorem system_solution (x y : Real) : 
  (Real.sin x)^2 + (Real.cos y)^2 = y^4 ∧ 
  (Real.sin y)^2 + (Real.cos x)^2 = x^2 → 
  (x = 1 ∨ x = -1) ∧ (y = 1 ∨ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3760_376099


namespace NUMINAMATH_CALUDE_product_evaluation_l3760_376004

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3760_376004


namespace NUMINAMATH_CALUDE_triangle_properties_l3760_376085

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Area condition
  (b^2 / (3 * Real.sin B)) = (1/2) * a * c * Real.sin B →
  -- Given condition
  Real.cos A * Real.cos C = 1/6 →
  -- Prove these statements
  Real.sin A * Real.sin C = 2/3 ∧ B = π/3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3760_376085


namespace NUMINAMATH_CALUDE_problem_solution_l3760_376045

theorem problem_solution : (π - 3.14) ^ 0 + (-1/2) ^ (-1 : ℤ) + |3 - Real.sqrt 8| - 4 * Real.cos (π/4) = 2 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3760_376045


namespace NUMINAMATH_CALUDE_line_slope_point_value_l3760_376081

theorem line_slope_point_value (m : ℝ) : 
  m > 0 → 
  (((m - 5) / (2 - m)) = Real.sqrt 2) → 
  m = 2 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_point_value_l3760_376081


namespace NUMINAMATH_CALUDE_f_maximum_properties_l3760_376029

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem f_maximum_properties (x₀ : ℝ) 
  (h1 : ∀ x > 0, f x ≤ f x₀) 
  (h2 : x₀ > 0) : 
  f x₀ = x₀ ∧ f x₀ > 1/9 := by
  sorry

end NUMINAMATH_CALUDE_f_maximum_properties_l3760_376029


namespace NUMINAMATH_CALUDE_lcm_48_140_l3760_376091

theorem lcm_48_140 : Nat.lcm 48 140 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_140_l3760_376091


namespace NUMINAMATH_CALUDE_average_glasses_per_box_l3760_376092

/-- Prove that the average number of glasses per box is 15, given the following conditions:
  - There are two types of boxes: small (12 glasses) and large (16 glasses)
  - There are 16 more large boxes than small boxes
  - The total number of glasses is 480
-/
theorem average_glasses_per_box (small_box : ℕ) (large_box : ℕ) :
  small_box * 12 + large_box * 16 = 480 →
  large_box = small_box + 16 →
  (480 : ℚ) / (small_box + large_box) = 15 := by
sorry


end NUMINAMATH_CALUDE_average_glasses_per_box_l3760_376092


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_theorem_l3760_376069

/-- Represents a rectangular parallelepiped -/
structure RectParallelepiped where
  base_side : ℝ
  cos_angle : ℝ

/-- Represents a vector configuration -/
structure VectorConfig where
  a_magnitude : ℝ
  a_dot_e : ℝ

theorem rectangular_parallelepiped_theorem (rp : RectParallelepiped) (vc : VectorConfig) :
  rp.base_side = 2 * Real.sqrt 2 →
  rp.cos_angle = Real.sqrt 3 / 3 →
  vc.a_magnitude = 2 * Real.sqrt 6 →
  vc.a_dot_e = 2 * Real.sqrt 2 →
  (∃ (sphere_surface_area : ℝ), sphere_surface_area = 24 * Real.pi) ∧
  (∃ (min_value : ℝ), min_value = 2 * Real.sqrt 2) := by
  sorry

#check rectangular_parallelepiped_theorem

end NUMINAMATH_CALUDE_rectangular_parallelepiped_theorem_l3760_376069


namespace NUMINAMATH_CALUDE_solutions_eq1_solutions_eq2_l3760_376036

-- First equation
theorem solutions_eq1 (x : ℝ) : x^2 - 2*x - 3 = 0 ↔ x = 3 ∨ x = -1 := by sorry

-- Second equation
theorem solutions_eq2 (x : ℝ) : x*(x-2) + x - 2 = 0 ↔ x = -1 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_solutions_eq1_solutions_eq2_l3760_376036


namespace NUMINAMATH_CALUDE_granola_bars_eaten_by_parents_l3760_376035

theorem granola_bars_eaten_by_parents (total : ℕ) (children : ℕ) (per_child : ℕ) 
  (h1 : total = 200) 
  (h2 : children = 6) 
  (h3 : per_child = 20) : 
  total - (children * per_child) = 80 :=
by sorry

end NUMINAMATH_CALUDE_granola_bars_eaten_by_parents_l3760_376035


namespace NUMINAMATH_CALUDE_trig_identities_l3760_376075

/-- Given tan α = 2, prove two trigonometric identities -/
theorem trig_identities (α : Real) (h : Real.tan α = 2) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 6/13 ∧
  3 * Real.sin α^2 + 3 * Real.sin α * Real.cos α - 2 * Real.cos α^2 = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l3760_376075


namespace NUMINAMATH_CALUDE_town_street_lights_l3760_376059

/-- Calculates the total number of street lights in a town given the number of neighborhoods,
    roads per neighborhood, and street lights per side of each road. -/
def totalStreetLights (neighborhoods : ℕ) (roadsPerNeighborhood : ℕ) (lightsPerSide : ℕ) : ℕ :=
  neighborhoods * roadsPerNeighborhood * lightsPerSide * 2

/-- Theorem stating that the total number of street lights in the described town is 20000. -/
theorem town_street_lights :
  totalStreetLights 10 4 250 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_town_street_lights_l3760_376059


namespace NUMINAMATH_CALUDE_one_language_speakers_l3760_376049

theorem one_language_speakers (total : ℕ) (latin french spanish : ℕ) (none : ℕ) 
  (latin_french latin_spanish french_spanish : ℕ) (all_three : ℕ) 
  (h1 : total = 40)
  (h2 : latin = 20)
  (h3 : french = 22)
  (h4 : spanish = 15)
  (h5 : none = 5)
  (h6 : latin_french = 8)
  (h7 : latin_spanish = 6)
  (h8 : french_spanish = 4)
  (h9 : all_three = 3) :
  total - none - (latin_french + latin_spanish + french_spanish - 2 * all_three) - all_three = 20 := by
  sorry

#check one_language_speakers

end NUMINAMATH_CALUDE_one_language_speakers_l3760_376049


namespace NUMINAMATH_CALUDE_smallest_multiple_l3760_376023

theorem smallest_multiple (n : ℕ) : 
  (∃ k : ℕ, n = 32 * k) ∧ 
  (∃ m : ℕ, n - 6 = 97 * m) ∧
  (∀ x : ℕ, x < n → ¬((∃ k : ℕ, x = 32 * k) ∧ (∃ m : ℕ, x - 6 = 97 * m))) →
  n = 2528 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_l3760_376023


namespace NUMINAMATH_CALUDE_road_travel_cost_l3760_376064

/-- Calculates the cost of traveling two intersecting roads on a rectangular lawn. -/
theorem road_travel_cost
  (lawn_length lawn_width road_width : ℕ)
  (cost_per_sqm : ℚ)
  (h1 : lawn_length = 80)
  (h2 : lawn_width = 60)
  (h3 : road_width = 10)
  (h4 : cost_per_sqm = 4) :
  (((lawn_length * road_width + lawn_width * road_width - road_width * road_width) : ℚ) * cost_per_sqm) = 5200 := by
  sorry

end NUMINAMATH_CALUDE_road_travel_cost_l3760_376064


namespace NUMINAMATH_CALUDE_translate_line_2x_minus_1_l3760_376027

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically -/
def translate_line (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + units }

/-- The theorem stating that translating y = 2x - 1 by 2 units 
    upward results in y = 2x + 1 -/
theorem translate_line_2x_minus_1 :
  let original_line : Line := { slope := 2, intercept := -1 }
  let translated_line := translate_line original_line 2
  translated_line = { slope := 2, intercept := 1 } := by
  sorry

end NUMINAMATH_CALUDE_translate_line_2x_minus_1_l3760_376027


namespace NUMINAMATH_CALUDE_cube_root_of_a_plus_b_l3760_376061

theorem cube_root_of_a_plus_b (a b : ℝ) (ha : a > 0) 
  (h1 : (2*b - 1)^2 = a) (h2 : (b + 4)^2 = a) (h3 : (2*b - 1) + (b + 4) = 0) : 
  (a + b)^(1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_a_plus_b_l3760_376061


namespace NUMINAMATH_CALUDE_dirk_profit_l3760_376077

/-- Calculates the profit for selling amulets at a Ren Faire --/
def amulet_profit (days : ℕ) (amulets_per_day : ℕ) (sell_price : ℕ) (cost_price : ℕ) (faire_fee_percent : ℕ) : ℕ :=
  let total_amulets := days * amulets_per_day
  let revenue := total_amulets * sell_price
  let faire_fee := revenue * faire_fee_percent / 100
  let revenue_after_fee := revenue - faire_fee
  let total_cost := total_amulets * cost_price
  revenue_after_fee - total_cost

/-- Theorem stating that Dirk's profit is 300 dollars --/
theorem dirk_profit :
  amulet_profit 2 25 40 30 10 = 300 := by
  sorry

end NUMINAMATH_CALUDE_dirk_profit_l3760_376077


namespace NUMINAMATH_CALUDE_brick_height_l3760_376008

/-- The surface area of a rectangular prism given its length, width, and height. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem stating that a rectangular prism with length 10, width 4, and surface area 136 has height 2. -/
theorem brick_height : ∃ (h : ℝ), h > 0 ∧ surface_area 10 4 h = 136 → h = 2 := by
  sorry

end NUMINAMATH_CALUDE_brick_height_l3760_376008


namespace NUMINAMATH_CALUDE_odd_function_value_l3760_376017

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x - 3*x + k else -(2^(-x) - 3*(-x) + k)
  where k : ℝ := -1 -- We define k here to make the function complete

-- State the theorem
theorem odd_function_value : f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l3760_376017


namespace NUMINAMATH_CALUDE_intersection_M_N_l3760_376006

def M : Set ℝ := {x | x^2 > 1}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3760_376006


namespace NUMINAMATH_CALUDE_remainder_1234567_div_256_l3760_376056

theorem remainder_1234567_div_256 : 1234567 % 256 = 45 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567_div_256_l3760_376056


namespace NUMINAMATH_CALUDE_smallest_max_sum_l3760_376015

theorem smallest_max_sum (a b c d e f : ℕ+) 
  (sum_eq : a + b + c + d + e + f = 2512) : 
  (∃ (M : ℕ), M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧ 
   (∀ (M' : ℕ), M' = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) → M ≤ M') ∧
   M = 1005) := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l3760_376015


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3760_376095

def U : Set Nat := {1,2,3,4,5,6,7}
def A : Set Nat := {1,3,5,7}
def B : Set Nat := {1,3,5,6,7}

theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {2,4,6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3760_376095


namespace NUMINAMATH_CALUDE_same_color_probability_l3760_376096

/-- The probability of drawing two balls of the same color from a box with 2 red balls and 3 white balls,
    when drawing with replacement. -/
theorem same_color_probability (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = red_balls + white_balls)
  (h2 : red_balls = 2)
  (h3 : white_balls = 3) :
  (red_balls : ℚ) / total_balls * (red_balls : ℚ) / total_balls + 
  (white_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 13 / 25 :=
sorry

end NUMINAMATH_CALUDE_same_color_probability_l3760_376096


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_20_l3760_376053

theorem x_plus_2y_equals_20 (x y : ℝ) (hx : x = 10) (hy : y = 5) : x + 2 * y = 20 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_20_l3760_376053


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3760_376050

structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  h_p_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x
  h_focus : focus = (1, 0)

structure Line where
  m : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop
  h_eq : ∀ x y, eq x y ↔ y = m*x + b

def intersect (C : Parabola) (l : Line) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | C.eq p.1 p.2 ∧ l.eq p.1 p.2}

def perpendicular (A B O : ℝ × ℝ) : Prop :=
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) = 0

theorem parabola_intersection_theorem (C : Parabola) (l : Line) 
  (A B : ℝ × ℝ) (h_AB : A ∈ intersect C l ∧ B ∈ intersect C l) 
  (h_perp : perpendicular A B (0, 0)) :
  ∃ (T : ℝ × ℝ), 
    (∃ (k : ℝ), ∀ (X : ℝ × ℝ), X ∈ intersect C l → 
      (X.2 / (X.1 - 4) + X.2 / (X.1 - T.1) = k)) ∧
    T = (-4, 0) ∧ 
    k = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3760_376050


namespace NUMINAMATH_CALUDE_company_employees_l3760_376086

/-- 
Given a company that had 15% more employees in December than in January,
and 460 employees in December, prove that it had 400 employees in January.
-/
theorem company_employees (december_employees : ℕ) (january_employees : ℕ) : 
  december_employees = 460 ∧ 
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 400 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l3760_376086


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l3760_376094

/-- A circle is defined by the equation x^2 + y^2 + Dx + Ey + F = 0 -/
def Circle (D E F : ℝ) := fun (x y : ℝ) => x^2 + y^2 + D*x + E*y + F = 0

/-- The specific circle we're interested in -/
def SpecificCircle := Circle (-4) (-6) 0

theorem circle_passes_through_points :
  (SpecificCircle 0 0) ∧ 
  (SpecificCircle 4 0) ∧ 
  (SpecificCircle (-1) 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l3760_376094


namespace NUMINAMATH_CALUDE_dust_particles_problem_l3760_376025

theorem dust_particles_problem (initial_dust : ℕ) : 
  (initial_dust / 10 + 223 = 331) → initial_dust = 1080 := by
  sorry

end NUMINAMATH_CALUDE_dust_particles_problem_l3760_376025


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l3760_376066

/-- Given a quadratic function y = x^2 - 2x + 3, prove it can be expressed as y = (x + m)^2 + h
    where m = -1 and h = 2 -/
theorem quadratic_complete_square :
  ∃ (m h : ℝ), ∀ (x y : ℝ),
    y = x^2 - 2*x + 3 → y = (x + m)^2 + h ∧ m = -1 ∧ h = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l3760_376066


namespace NUMINAMATH_CALUDE_purple_cars_count_l3760_376093

theorem purple_cars_count (purple red green : ℕ) : 
  green = 4 * red →
  red = purple + 6 →
  purple + red + green = 312 →
  purple = 47 := by
sorry

end NUMINAMATH_CALUDE_purple_cars_count_l3760_376093


namespace NUMINAMATH_CALUDE_roots_of_cubic_equation_l3760_376013

variable (a b c d α β : ℝ)

def original_quadratic (x : ℝ) : ℝ := x^2 - (a + d)*x + (a*d - b*c)

def new_quadratic (x : ℝ) : ℝ := x^2 - (a^3 + d^3 + 3*a*b*c + 3*b*c*d)*x + (a*d - b*c)^3

theorem roots_of_cubic_equation 
  (h1 : original_quadratic α = 0)
  (h2 : original_quadratic β = 0) :
  new_quadratic (α^3) = 0 ∧ new_quadratic (β^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_cubic_equation_l3760_376013


namespace NUMINAMATH_CALUDE_largest_odd_integer_in_range_l3760_376039

theorem largest_odd_integer_in_range : 
  ∃ (x : ℤ), (x % 2 = 1) ∧ (1/4 < x/6) ∧ (x/6 < 7/9) ∧
  ∀ (y : ℤ), (y % 2 = 1) ∧ (1/4 < y/6) ∧ (y/6 < 7/9) → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_odd_integer_in_range_l3760_376039


namespace NUMINAMATH_CALUDE_expected_value_S_squared_l3760_376042

/-- ω is a primitive 2018th root of unity -/
def ω : ℂ :=
  sorry

/-- The set of complex numbers from which subsets are chosen -/
def complexSet : Finset ℂ :=
  sorry

/-- S is the sum of elements in a randomly chosen subset of complexSet -/
def S : Finset ℂ → ℂ :=
  sorry

/-- The expected value of |S|² -/
def expectedValueS : ℝ :=
  sorry

theorem expected_value_S_squared :
  expectedValueS = 1009 / 2 :=
sorry

end NUMINAMATH_CALUDE_expected_value_S_squared_l3760_376042


namespace NUMINAMATH_CALUDE_prob_even_sum_l3760_376021

/-- Probability of selecting an even number from the first wheel -/
def P_even1 : ℚ := 2/3

/-- Probability of selecting an odd number from the first wheel -/
def P_odd1 : ℚ := 1/3

/-- Probability of selecting an even number from the second wheel -/
def P_even2 : ℚ := 1/2

/-- Probability of selecting an odd number from the second wheel -/
def P_odd2 : ℚ := 1/2

/-- The probability of selecting an even sum from two wheels with the given probability distributions -/
theorem prob_even_sum : P_even1 * P_even2 + P_odd1 * P_odd2 = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_prob_even_sum_l3760_376021


namespace NUMINAMATH_CALUDE_certain_number_proof_l3760_376062

theorem certain_number_proof : ∃ x : ℝ, (7.5 * 7.5) + x + (2.5 * 2.5) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3760_376062


namespace NUMINAMATH_CALUDE_umbrella_arrangement_count_l3760_376014

/-- The number of ways to arrange n people with distinct heights in an umbrella shape -/
def umbrella_arrangements (n : ℕ) : ℕ :=
  sorry

/-- There are 7 actors with distinct heights to be arranged -/
def num_actors : ℕ := 7

theorem umbrella_arrangement_count :
  umbrella_arrangements num_actors = 20 := by sorry

end NUMINAMATH_CALUDE_umbrella_arrangement_count_l3760_376014


namespace NUMINAMATH_CALUDE_yunas_math_score_l3760_376072

theorem yunas_math_score (score1 score2 : ℝ) (h1 : (score1 + score2) / 2 = 92) 
  (h2 : ∃ (score3 : ℝ), (score1 + score2 + score3) / 3 = 94) : 
  ∃ (score3 : ℝ), score3 = 98 ∧ (score1 + score2 + score3) / 3 = 94 := by
  sorry

end NUMINAMATH_CALUDE_yunas_math_score_l3760_376072


namespace NUMINAMATH_CALUDE_triangle_movement_path_length_l3760_376055

/-- Represents the movement of a triangle inside a square -/
structure TriangleMovement where
  square_side : ℝ
  triangle_side : ℝ
  initial_rotation_radius : ℝ
  final_rotation_radius : ℝ
  initial_rotation_angle : ℝ
  final_rotation_angle : ℝ

/-- Calculates the total path traversed by vertex P -/
def total_path_length (m : TriangleMovement) : ℝ :=
  m.initial_rotation_radius * m.initial_rotation_angle +
  m.final_rotation_radius * m.final_rotation_angle

/-- The theorem to be proved -/
theorem triangle_movement_path_length :
  ∀ (m : TriangleMovement),
  m.square_side = 6 ∧
  m.triangle_side = 3 ∧
  m.initial_rotation_radius = m.triangle_side ∧
  m.final_rotation_radius = (m.square_side / 2 + m.triangle_side / 2) ∧
  m.initial_rotation_angle = Real.pi ∧
  m.final_rotation_angle = 2 * Real.pi →
  total_path_length m = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_triangle_movement_path_length_l3760_376055


namespace NUMINAMATH_CALUDE_table_chair_price_ratio_l3760_376051

/-- The price ratio of tables to chairs in a store -/
theorem table_chair_price_ratio :
  ∀ (chair_price table_price : ℝ),
  chair_price > 0 →
  table_price > 0 →
  2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price) →
  table_price = 7 * chair_price :=
by
  sorry

end NUMINAMATH_CALUDE_table_chair_price_ratio_l3760_376051


namespace NUMINAMATH_CALUDE_sequence_formula_T_formula_l3760_376057

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom S_def (n : ℕ) : n > 0 → S n = 2 * sequence_a n - 2

theorem sequence_formula (n : ℕ) (h : n > 0) : sequence_a n = 2^n := by sorry

def T (n : ℕ) : ℝ := sorry

theorem T_formula (n : ℕ) (h : n > 0) : T n = 2^(n+2) - 4 - 2*n := by sorry

end NUMINAMATH_CALUDE_sequence_formula_T_formula_l3760_376057


namespace NUMINAMATH_CALUDE_ratio_transitivity_l3760_376007

theorem ratio_transitivity (a b c : ℚ) 
  (hab : a / b = 8 / 3) 
  (hbc : b / c = 1 / 5) : 
  a / c = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_transitivity_l3760_376007


namespace NUMINAMATH_CALUDE_integral_f_equals_five_sixths_l3760_376022

-- Define the piecewise function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x ≥ 0 ∧ x ≤ 1 then x^2
  else if x > 1 ∧ x ≤ 2 then 2 - x
  else 0  -- Define a value for x outside [0,2] to make f total

-- State the theorem
theorem integral_f_equals_five_sixths :
  ∫ x in (0)..(2), f x = 5/6 := by sorry

end NUMINAMATH_CALUDE_integral_f_equals_five_sixths_l3760_376022


namespace NUMINAMATH_CALUDE_movie_date_candy_cost_l3760_376097

theorem movie_date_candy_cost
  (ticket_cost : ℝ)
  (combo_cost : ℝ)
  (total_spend : ℝ)
  (num_candy : ℕ)
  (h1 : ticket_cost = 20)
  (h2 : combo_cost = 11)
  (h3 : total_spend = 36)
  (h4 : num_candy = 2) :
  (total_spend - ticket_cost - combo_cost) / num_candy = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_movie_date_candy_cost_l3760_376097


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3760_376083

theorem min_value_of_expression (a b c d : ℝ) 
  (hb : b > 0) (hc : c > 0) (ha : a ≥ 0) (hd : d ≥ 0) 
  (h_sum : b + c ≥ a + d) : 
  (b / (c + d) + c / (a + b)) ≥ Real.sqrt 2 - 1/2 := 
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3760_376083


namespace NUMINAMATH_CALUDE_trees_chopped_second_half_proof_l3760_376019

def trees_chopped_first_half : ℕ := 200
def trees_planted_per_chopped : ℕ := 3
def total_trees_to_plant : ℕ := 1500

def trees_chopped_second_half : ℕ := 300

theorem trees_chopped_second_half_proof :
  trees_chopped_second_half = 
    (total_trees_to_plant - trees_planted_per_chopped * trees_chopped_first_half) / 
    trees_planted_per_chopped := by
  sorry

end NUMINAMATH_CALUDE_trees_chopped_second_half_proof_l3760_376019


namespace NUMINAMATH_CALUDE_kia_vehicles_count_l3760_376037

def total_vehicles : ℕ := 400

def dodge_vehicles : ℕ := total_vehicles / 2

def hyundai_vehicles : ℕ := dodge_vehicles / 2

def kia_vehicles : ℕ := total_vehicles - dodge_vehicles - hyundai_vehicles

theorem kia_vehicles_count : kia_vehicles = 100 := by
  sorry

end NUMINAMATH_CALUDE_kia_vehicles_count_l3760_376037


namespace NUMINAMATH_CALUDE_count_ordered_pairs_l3760_376063

theorem count_ordered_pairs (n : ℕ) (hn : n > 1) :
  (Finset.sum (Finset.range (n - 1)) (fun k => n - k)) = (n - 1) * n / 2 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_l3760_376063


namespace NUMINAMATH_CALUDE_midpoint_of_fractions_l3760_376052

theorem midpoint_of_fractions :
  (1 / 6 + 1 / 12) / 2 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_of_fractions_l3760_376052


namespace NUMINAMATH_CALUDE_seth_oranges_l3760_376058

theorem seth_oranges (initial_boxes : ℕ) : 
  (initial_boxes - 1) / 2 = 4 → initial_boxes = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_seth_oranges_l3760_376058


namespace NUMINAMATH_CALUDE_infinitely_many_odd_terms_l3760_376005

theorem infinitely_many_odd_terms (n : ℕ) (hn : n > 1) :
  ∀ m : ℕ, ∃ k > m, Odd (⌊(n^k : ℝ) / k⌋) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_odd_terms_l3760_376005


namespace NUMINAMATH_CALUDE_subtraction_of_large_numbers_l3760_376041

theorem subtraction_of_large_numbers :
  10000000000000 - (5555555555555 * 2) = -1111111111110 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_large_numbers_l3760_376041


namespace NUMINAMATH_CALUDE_even_product_probability_l3760_376098

def set_A_odd : ℕ := 7
def set_A_even : ℕ := 9
def set_B_odd : ℕ := 5
def set_B_even : ℕ := 4

def total_A : ℕ := set_A_odd + set_A_even
def total_B : ℕ := set_B_odd + set_B_even

def prob_even_product : ℚ := 109 / 144

theorem even_product_probability :
  (set_A_even : ℚ) / total_A * (set_B_even : ℚ) / total_B +
  (set_A_odd : ℚ) / total_A * (set_B_even : ℚ) / total_B +
  (set_A_even : ℚ) / total_A * (set_B_odd : ℚ) / total_B = prob_even_product :=
by sorry

end NUMINAMATH_CALUDE_even_product_probability_l3760_376098


namespace NUMINAMATH_CALUDE_probability_one_pair_one_triplet_proof_l3760_376047

/-- The probability of rolling six standard six-sided dice and getting exactly
    one pair, one triplet, and the remaining dice showing different values. -/
def probability_one_pair_one_triplet : ℚ := 25 / 162

/-- The number of possible outcomes when rolling six standard six-sided dice. -/
def total_outcomes : ℕ := 6^6

/-- The number of successful outcomes (one pair, one triplet, remaining different). -/
def successful_outcomes : ℕ := 7200

theorem probability_one_pair_one_triplet_proof :
  probability_one_pair_one_triplet = successful_outcomes / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_probability_one_pair_one_triplet_proof_l3760_376047


namespace NUMINAMATH_CALUDE_quiz_probabilities_l3760_376040

/-- Represents the quiz with multiple-choice and true/false questions -/
structure Quiz where
  total_questions : ℕ
  multiple_choice : ℕ
  true_false : ℕ

/-- Calculates the probability of A drawing a multiple-choice question and B drawing a true/false question -/
def prob_a_multiple_b_true_false (q : Quiz) : ℚ :=
  (q.multiple_choice * q.true_false) / (q.total_questions * (q.total_questions - 1))

/-- Calculates the probability of at least one of A or B drawing a multiple-choice question -/
def prob_at_least_one_multiple (q : Quiz) : ℚ :=
  1 - (q.true_false * (q.true_false - 1)) / (q.total_questions * (q.total_questions - 1))

theorem quiz_probabilities (q : Quiz) 
  (h1 : q.total_questions = 10)
  (h2 : q.multiple_choice = 6)
  (h3 : q.true_false = 4) :
  prob_a_multiple_b_true_false q = 4 / 15 ∧ 
  prob_at_least_one_multiple q = 13 / 15 := by
  sorry


end NUMINAMATH_CALUDE_quiz_probabilities_l3760_376040


namespace NUMINAMATH_CALUDE_pete_total_books_matt_year2_increase_l3760_376043

/-- The number of books Matt read in the first year -/
def matt_year1 : ℕ := 50

/-- The number of books Matt read in the second year -/
def matt_year2 : ℕ := 75

/-- The number of books Pete read in the first year -/
def pete_year1 : ℕ := 2 * matt_year1

/-- The number of books Pete read in the second year -/
def pete_year2 : ℕ := 2 * pete_year1

/-- Theorem stating that Pete read 300 books across both years -/
theorem pete_total_books : pete_year1 + pete_year2 = 300 := by
  sorry

/-- Verification that Matt's second year reading increased by 50% -/
theorem matt_year2_increase : matt_year2 = (3 * matt_year1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_pete_total_books_matt_year2_increase_l3760_376043


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3760_376074

theorem quadratic_equation_roots (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 9 ∧ x * y = 20) →
  m + n = 87 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3760_376074


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3760_376032

/-- An isosceles triangle with side lengths satisfying x^2 - 5x + 6 = 0 has perimeter 7 or 8 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  (a^2 - 5*a + 6 = 0) →
  (b^2 - 5*b + 6 = 0) →
  (a = b ∨ a = c ∨ b = c) →  -- isosceles condition
  (a + b > c ∧ a + c > b ∧ b + c > a) →  -- triangle inequality
  (a + b + c = 7 ∨ a + b + c = 8) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3760_376032


namespace NUMINAMATH_CALUDE_cone_volume_ratio_l3760_376078

-- Define the ratio of central angles
def angle_ratio : ℚ := 3 / 4

-- Define a function to calculate the volume ratio given the angle ratio
def volume_ratio (r : ℚ) : ℚ := r^2

-- Theorem statement
theorem cone_volume_ratio :
  volume_ratio angle_ratio = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_ratio_l3760_376078


namespace NUMINAMATH_CALUDE_circle_and_chord_theorem_l3760_376070

/-- The polar coordinate equation of a circle C that passes through the point (√2, π/4)
    and has its center at the intersection of the polar axis and the line ρ sin(θ - π/3) = -√3/2 -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- The length of the chord intercepted by the line θ = π/3 on the circle C defined by ρ = 2cos(θ) -/
def chord_length : ℝ := 1

theorem circle_and_chord_theorem :
  /- Circle C passes through (√2, π/4) -/
  (circle_equation (Real.sqrt 2) (π / 4)) ∧
  /- The center of C is at the intersection of the polar axis and ρ sin(θ - π/3) = -√3/2 -/
  (∃ ρ₀ : ℝ, ρ₀ * Real.sin (0 - π / 3) = -Real.sqrt 3 / 2) ∧
  /- The polar coordinate equation of circle C is ρ = 2cos(θ) -/
  (∀ ρ θ : ℝ, circle_equation ρ θ ↔ ρ = 2 * Real.cos θ) ∧
  /- The length of the chord intercepted by θ = π/3 on circle C is 1 -/
  chord_length = 1 := by
    sorry

end NUMINAMATH_CALUDE_circle_and_chord_theorem_l3760_376070
