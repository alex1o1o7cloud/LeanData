import Mathlib

namespace NUMINAMATH_CALUDE_balloon_difference_l1204_120485

theorem balloon_difference (allan_balloons jake_balloons : ℕ) : 
  allan_balloons = 5 →
  jake_balloons = 11 →
  jake_balloons > allan_balloons →
  jake_balloons - allan_balloons = 6 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l1204_120485


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l1204_120453

theorem last_two_digits_sum (n : ℕ) : n = 25 → (6^n + 14^n) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l1204_120453


namespace NUMINAMATH_CALUDE_speed_of_northern_cyclist_l1204_120450

/-- Theorem: Speed of northern cyclist
Given two cyclists starting from the same place in opposite directions,
with one going north at speed v kmph and the other going south at 40 kmph,
if they are 50 km apart after 0.7142857142857143 hours, then v = 30 kmph. -/
theorem speed_of_northern_cyclist (v : ℝ) : 
  v > 0 → -- Assuming positive speed
  50 = (v + 40) * 0.7142857142857143 →
  v = 30 := by
  sorry

end NUMINAMATH_CALUDE_speed_of_northern_cyclist_l1204_120450


namespace NUMINAMATH_CALUDE_tax_percentage_calculation_l1204_120497

theorem tax_percentage_calculation (original_cost total_paid : ℝ) 
  (h1 : original_cost = 200)
  (h2 : total_paid = 230) :
  (total_paid - original_cost) / original_cost * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_tax_percentage_calculation_l1204_120497


namespace NUMINAMATH_CALUDE_janet_freelance_earnings_l1204_120420

/-- Calculates the difference in monthly earnings between freelancing and current job --/
def freelance_earnings_difference (
  hours_per_week : ℕ)
  (current_wage : ℚ)
  (freelance_wage : ℚ)
  (weeks_per_month : ℕ)
  (extra_fica_per_week : ℚ)
  (healthcare_premium_per_month : ℚ) : ℚ :=
  let wage_difference := freelance_wage - current_wage
  let weekly_earnings_difference := wage_difference * hours_per_week
  let monthly_earnings_difference := weekly_earnings_difference * weeks_per_month
  let extra_monthly_expenses := extra_fica_per_week * weeks_per_month + healthcare_premium_per_month
  monthly_earnings_difference - extra_monthly_expenses

/-- Theorem stating the earnings difference for Janet's specific situation --/
theorem janet_freelance_earnings :
  freelance_earnings_difference 40 30 40 4 25 400 = 1100 := by
  sorry

end NUMINAMATH_CALUDE_janet_freelance_earnings_l1204_120420


namespace NUMINAMATH_CALUDE_june_video_hours_l1204_120452

/-- Calculates the total video hours uploaded in a month with varying upload rates -/
def total_video_hours (days : ℕ) (initial_rate : ℕ) (doubled_rate : ℕ) : ℕ :=
  let half_days := days / 2
  (half_days * initial_rate) + (half_days * doubled_rate)

/-- Proves that the total video hours uploaded in June is 450 -/
theorem june_video_hours :
  total_video_hours 30 10 20 = 450 := by
  sorry

end NUMINAMATH_CALUDE_june_video_hours_l1204_120452


namespace NUMINAMATH_CALUDE_rectangle_width_on_square_diagonal_l1204_120477

theorem rectangle_width_on_square_diagonal (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let diagonal := s * Real.sqrt 2
  let rectangle_length := diagonal
  let rectangle_width := s / Real.sqrt 2
  square_area = rectangle_length * rectangle_width :=
by sorry

end NUMINAMATH_CALUDE_rectangle_width_on_square_diagonal_l1204_120477


namespace NUMINAMATH_CALUDE_teachers_count_l1204_120428

/-- Given a school with girls, boys, and teachers, calculates the number of teachers. -/
def calculate_teachers (girls boys total : ℕ) : ℕ :=
  total - (girls + boys)

/-- Proves that there are 772 teachers in a school with 315 girls, 309 boys, and 1396 people in total. -/
theorem teachers_count : calculate_teachers 315 309 1396 = 772 := by
  sorry

end NUMINAMATH_CALUDE_teachers_count_l1204_120428


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1204_120435

-- Problem 1
theorem problem_1 : -4.7 + 0.9 = -3.8 := by sorry

-- Problem 2
theorem problem_2 : -1/2 - (-1/3) = -1/6 := by sorry

-- Problem 3
theorem problem_3 : (-1 - 1/9) * (-0.6) = 2/3 := by sorry

-- Problem 4
theorem problem_4 : 0 * (-5) = 0 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1204_120435


namespace NUMINAMATH_CALUDE_metro_ticket_sales_l1204_120436

/-- Proves that the average number of tickets sold per minute is 5,
    given the cost per ticket and total earnings over 6 minutes. -/
theorem metro_ticket_sales
  (ticket_cost : ℝ)
  (total_earnings : ℝ)
  (duration : ℕ)
  (h1 : ticket_cost = 3)
  (h2 : total_earnings = 90)
  (h3 : duration = 6) :
  total_earnings / (ticket_cost * duration) = 5 := by
  sorry

end NUMINAMATH_CALUDE_metro_ticket_sales_l1204_120436


namespace NUMINAMATH_CALUDE_c_constant_when_n_doubled_l1204_120445

/-- Given positive constants e, R, and r, and a positive variable n,
    the function C(n) remains constant when n is doubled. -/
theorem c_constant_when_n_doubled
  (e R r : ℝ) (n : ℝ) 
  (he : e > 0) (hR : R > 0) (hr : r > 0) (hn : n > 0) :
  let C : ℝ → ℝ := fun n => (e^2 * n) / (R + n * r^2)
  C n = C (2 * n) := by
  sorry

end NUMINAMATH_CALUDE_c_constant_when_n_doubled_l1204_120445


namespace NUMINAMATH_CALUDE_inverse_proportion_relationship_l1204_120468

/-- Given points A(-1, y₁), B(2, y₂), and C(3, y₃) on the graph of y = -6/x,
    prove that y₁ > y₃ > y₂ -/
theorem inverse_proportion_relationship (y₁ y₂ y₃ : ℝ) 
  (h₁ : y₁ = 6 / (-1))
  (h₂ : y₂ = -6 / 2)
  (h₃ : y₃ = -6 / 3) :
  y₁ > y₃ ∧ y₃ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_relationship_l1204_120468


namespace NUMINAMATH_CALUDE_circle_and_locus_equations_l1204_120487

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a : ℝ), (x - a)^2 + (y - (2 - a))^2 = (a - 4)^2 + (2 - a)^2 ∧
              (x - a)^2 + (y - (2 - a))^2 = (a - 2)^2 + (2 - a - 2)^2

-- Define the locus of midpoint M
def locus_M (x y : ℝ) : Prop :=
  ∃ (x₁ y₁ : ℝ), circle_C x₁ y₁ ∧ x = (x₁ + 5) / 2 ∧ y = y₁ / 2

theorem circle_and_locus_equations :
  (∀ x y, circle_C x y ↔ (x - 2)^2 + y^2 = 4) ∧
  (∀ x y, locus_M x y ↔ x^2 - 7*x + y^2 + 45/4 = 0) :=
sorry

end NUMINAMATH_CALUDE_circle_and_locus_equations_l1204_120487


namespace NUMINAMATH_CALUDE_two_a_squared_eq_three_b_cubed_l1204_120495

theorem two_a_squared_eq_three_b_cubed (a b : ℕ+) :
  2 * a ^ 2 = 3 * b ^ 3 ↔ ∃ d : ℕ+, a = 18 * d ^ 3 ∧ b = 6 * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_two_a_squared_eq_three_b_cubed_l1204_120495


namespace NUMINAMATH_CALUDE_expression_evaluation_l1204_120464

theorem expression_evaluation :
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 2
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 5 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1204_120464


namespace NUMINAMATH_CALUDE_optimal_k_value_l1204_120469

theorem optimal_k_value : ∃! k : ℝ, 
  (∀ a b c d : ℝ, a ≥ -1 ∧ b ≥ -1 ∧ c ≥ -1 ∧ d ≥ -1 → 
    a^3 + b^3 + c^3 + d^3 + 1 ≥ k * (a + b + c + d)) ∧ 
  k = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_optimal_k_value_l1204_120469


namespace NUMINAMATH_CALUDE_shifted_roots_polynomial_l1204_120458

theorem shifted_roots_polynomial (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 5*x + 7 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 + 9*x^2 + 22*x + 19 = 0 ↔ x = a - 3 ∨ x = b - 3 ∨ x = c - 3) := by
sorry

end NUMINAMATH_CALUDE_shifted_roots_polynomial_l1204_120458


namespace NUMINAMATH_CALUDE_seventh_observation_value_l1204_120489

theorem seventh_observation_value
  (n : ℕ) -- number of initial observations
  (initial_avg : ℚ) -- initial average
  (new_avg : ℚ) -- new average after adding one observation
  (h1 : n = 6) -- there are 6 initial observations
  (h2 : initial_avg = 16) -- the initial average is 16
  (h3 : new_avg = initial_avg - 1) -- the new average is decreased by 1
  : (n + 1) * new_avg - n * initial_avg = 9 := by
  sorry

end NUMINAMATH_CALUDE_seventh_observation_value_l1204_120489


namespace NUMINAMATH_CALUDE_max_consecutive_sum_of_5_to_7_l1204_120466

theorem max_consecutive_sum_of_5_to_7 :
  ∀ p : ℕ+, 
    (∃ a : ℕ+, (Finset.range p).sum (λ i => a + i) = 5^7) →
    p ≤ 125 :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_of_5_to_7_l1204_120466


namespace NUMINAMATH_CALUDE_matrix_power_result_l1204_120462

theorem matrix_power_result (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.mulVec ![4, -1] = ![12, -3]) :
  (B ^ 4).mulVec ![4, -1] = ![324, -81] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_result_l1204_120462


namespace NUMINAMATH_CALUDE_marias_school_students_l1204_120419

theorem marias_school_students (m d : ℕ) : 
  m = 4 * d → 
  m - d = 1800 → 
  m = 2400 := by
sorry

end NUMINAMATH_CALUDE_marias_school_students_l1204_120419


namespace NUMINAMATH_CALUDE_system_integer_solutions_l1204_120491

theorem system_integer_solutions (a b c d : ℤ) 
  (h : ∀ (m n : ℤ), ∃ (x y : ℤ), a * x + b * y = m ∧ c * x + d * y = n) : 
  (a * d - b * c = 1) ∨ (a * d - b * c = -1) := by sorry

end NUMINAMATH_CALUDE_system_integer_solutions_l1204_120491


namespace NUMINAMATH_CALUDE_solve_refrigerator_problem_l1204_120414

def refrigerator_problem (purchase_price : ℝ) (discount_rate : ℝ) (transport_cost : ℝ) (profit_rate : ℝ) (selling_price : ℝ) : Prop :=
  let labelled_price := purchase_price / (1 - discount_rate)
  let profit := labelled_price * profit_rate
  let calculated_selling_price := labelled_price + profit
  let extra_amount := selling_price - calculated_selling_price
  let installation_cost := extra_amount - transport_cost
  installation_cost = 310

theorem solve_refrigerator_problem :
  refrigerator_problem 12500 0.20 125 0.16 18560 := by
  sorry

end NUMINAMATH_CALUDE_solve_refrigerator_problem_l1204_120414


namespace NUMINAMATH_CALUDE_sequence_divisibility_l1204_120483

theorem sequence_divisibility (k : ℕ+) 
  (a : ℕ → ℤ)
  (h : ∀ n : ℕ, n ≥ 1 → a n = (a (n - 1) + n^(k : ℕ)) / n) :
  3 ∣ (k : ℤ) - 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l1204_120483


namespace NUMINAMATH_CALUDE_root_expression_value_l1204_120454

theorem root_expression_value (a : ℝ) : 
  2 * a^2 - 7 * a - 1 = 0 → a * (2 * a - 7) + 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_expression_value_l1204_120454


namespace NUMINAMATH_CALUDE_johnny_age_puzzle_l1204_120403

/-- Represents Johnny's age now -/
def current_age : ℕ := 8

/-- Represents the number of years into the future Johnny is referring to -/
def future_years : ℕ := 2

/-- Theorem stating that the number of years into the future Johnny was referring to is correct -/
theorem johnny_age_puzzle :
  (current_age + future_years = 2 * (current_age - 3)) ∧
  (future_years = 2) := by
  sorry

end NUMINAMATH_CALUDE_johnny_age_puzzle_l1204_120403


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l1204_120457

/-- The perpendicular bisector of a line segment from (2, 5) to (8, 11) has equation 2x - y = c. -/
theorem perpendicular_bisector_c_value :
  ∃ (c : ℝ), 
    (∀ (x y : ℝ), (2 * x - y = c) ↔ 
      (x - 5)^2 + (y - 8)^2 = (5 - 2)^2 + (8 - 5)^2 ∧ 
      (x - 5) * (8 - 2) = -(y - 8) * (11 - 5)) → 
    c = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l1204_120457


namespace NUMINAMATH_CALUDE_squirrel_acorns_l1204_120490

theorem squirrel_acorns (num_squirrels : ℕ) (total_acorns : ℕ) (acorns_needed : ℕ) 
  (h1 : num_squirrels = 20)
  (h2 : total_acorns = 4500)
  (h3 : acorns_needed = 300) :
  acorns_needed - (total_acorns / num_squirrels) = 75 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l1204_120490


namespace NUMINAMATH_CALUDE_range_of_a_l1204_120488

/-- The range of a satisfying the given conditions -/
theorem range_of_a : ∀ a : ℝ, 
  ((∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → x^2 + 2*x - 8 > 0) ∧ 
   (∃ x : ℝ, x^2 + 2*x - 8 > 0 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0)) ↔ 
  (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1204_120488


namespace NUMINAMATH_CALUDE_exists_abc_sum_product_l1204_120451

def NatPos := {n : ℕ | n > 0}

def A : Set ℕ := {x | ∃ m ∈ NatPos, x = 3 * m}
def B : Set ℕ := {x | ∃ m ∈ NatPos, x = 3 * m - 1}
def C : Set ℕ := {x | ∃ m ∈ NatPos, x = 3 * m - 2}

theorem exists_abc_sum_product (a : ℕ) (b : ℕ) (c : ℕ) 
  (ha : a ∈ A) (hb : b ∈ B) (hc : c ∈ C) :
  ∃ a b c, a ∈ A ∧ b ∈ B ∧ c ∈ C ∧ 2006 = a + b * c :=
by sorry

end NUMINAMATH_CALUDE_exists_abc_sum_product_l1204_120451


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_16_l1204_120409

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (acc : ℕ) : ℕ :=
    if m = 0 then acc
    else aux (m / 10) (acc + m % 10)
  aux n 0

-- Define the property for a number to be a four-digit number
def isFourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

-- Theorem statement
theorem largest_four_digit_sum_16 :
  ∀ n : ℕ, isFourDigitNumber n → sumOfDigits n = 16 → n ≤ 9700 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_16_l1204_120409


namespace NUMINAMATH_CALUDE_v2_equals_14_l1204_120486

/-- Qin Jiushao's algorithm for polynomial evaluation -/
def qin_jiushao (a b c d e : ℝ) (x : ℝ) : ℝ × ℝ × ℝ := 
  let v₀ := x
  let v₁ := a * x + b
  let v₂ := v₁ * x + c
  (v₀, v₁, v₂)

/-- The theorem stating that v₂ = 14 for the given function and x = 2 -/
theorem v2_equals_14 : 
  let (v₀, v₁, v₂) := qin_jiushao 2 3 0 5 (-4) 2
  v₂ = 14 := by
sorry

end NUMINAMATH_CALUDE_v2_equals_14_l1204_120486


namespace NUMINAMATH_CALUDE_sum_convergence_implies_k_value_l1204_120412

/-- Given a real number k > 1 such that the sum of (7n-3)/k^n from n=1 to infinity equals 20/3,
    prove that k = 1.9125 -/
theorem sum_convergence_implies_k_value (k : ℝ) 
  (h1 : k > 1)
  (h2 : ∑' n, (7 * n - 3) / k^n = 20/3) : 
  k = 1.9125 := by
  sorry

end NUMINAMATH_CALUDE_sum_convergence_implies_k_value_l1204_120412


namespace NUMINAMATH_CALUDE_coach_votes_l1204_120498

theorem coach_votes (num_coaches : ℕ) (num_voters : ℕ) (votes_per_voter : ℕ) 
  (h1 : num_coaches = 36)
  (h2 : num_voters = 60)
  (h3 : votes_per_voter = 3)
  (h4 : num_voters * votes_per_voter % num_coaches = 0) :
  (num_voters * votes_per_voter) / num_coaches = 5 := by
sorry

end NUMINAMATH_CALUDE_coach_votes_l1204_120498


namespace NUMINAMATH_CALUDE_garden_area_l1204_120433

theorem garden_area (width length : ℝ) (h1 : length = 3 * width + 30) 
  (h2 : 2 * (length + width) = 800) : width * length = 28443.75 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l1204_120433


namespace NUMINAMATH_CALUDE_quadratic_touch_existence_l1204_120434

theorem quadratic_touch_existence (p q : ℤ) (h : p^2 = 4*q) :
  ∃ (a b : ℤ), b = a^2 + p*a + q ∧ a^2 = 4*b :=
sorry

end NUMINAMATH_CALUDE_quadratic_touch_existence_l1204_120434


namespace NUMINAMATH_CALUDE_sqrt_225_equals_15_l1204_120439

theorem sqrt_225_equals_15 : Real.sqrt 225 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_225_equals_15_l1204_120439


namespace NUMINAMATH_CALUDE_horner_method_example_l1204_120424

def f (x : ℝ) : ℝ := 9 + 15*x - 8*x^2 - 20*x^3 + 6*x^4 + 3*x^5

theorem horner_method_example : f 4 = 3269 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_example_l1204_120424


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1204_120423

def z : ℂ := (1 + Complex.I) * (1 - 2 * Complex.I)

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1204_120423


namespace NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l1204_120479

theorem circle_radius_from_longest_chord (c : Real) (h : c > 0) : 
  ∃ (r : Real), r > 0 ∧ r = c / 2 := by sorry

end NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l1204_120479


namespace NUMINAMATH_CALUDE_spade_calculation_l1204_120422

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_calculation : spade 2 (spade 3 (spade 1 4)) = -46652 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l1204_120422


namespace NUMINAMATH_CALUDE_george_number_l1204_120455

/-- Checks if a number is skipped by a student given their position in the sequence -/
def isSkipped (num : ℕ) (studentPosition : ℕ) : Prop :=
  ∃ k : ℕ, num = 5^studentPosition * (5 * k - 1) - 1

/-- Checks if a number is the sum of squares of two consecutive integers -/
def isSumOfConsecutiveSquares (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 + (k+1)^2

theorem george_number :
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 1005 ∧
  (∀ i : ℕ, i ≥ 1 ∧ i ≤ 6 → ¬isSkipped n i) ∧
  isSumOfConsecutiveSquares n ∧
  n = 25 := by sorry

end NUMINAMATH_CALUDE_george_number_l1204_120455


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l1204_120418

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∃ (k : ℕ), (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) = 15 * k) ∧
  (∀ (d : ℕ), d > 15 → ∃ (m : ℕ), Even m ∧ m > 0 ∧
    ¬(∃ (k : ℕ), (m + 3) * (m + 5) * (m + 7) * (m + 9) * (m + 11) = d * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l1204_120418


namespace NUMINAMATH_CALUDE_conference_seating_optimization_l1204_120404

theorem conference_seating_optimization
  (initial_chairs : ℕ)
  (chairs_per_row : ℕ)
  (expected_participants : ℕ)
  (h1 : initial_chairs = 144)
  (h2 : chairs_per_row = 12)
  (h3 : expected_participants = 100)
  : ∃ (chairs_to_remove : ℕ),
    chairs_to_remove = 36 ∧
    (initial_chairs - chairs_to_remove) % chairs_per_row = 0 ∧
    initial_chairs - chairs_to_remove ≥ expected_participants ∧
    ∀ (x : ℕ), x < chairs_to_remove →
      (initial_chairs - x) % chairs_per_row ≠ 0 ∨
      initial_chairs - x > expected_participants + chairs_per_row - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_conference_seating_optimization_l1204_120404


namespace NUMINAMATH_CALUDE_red_balls_count_l1204_120475

theorem red_balls_count (total_balls : ℕ) (prob_red : ℚ) (red_balls : ℕ) : 
  total_balls = 1000 →
  prob_red = 1/5 →
  red_balls = (total_balls : ℚ) * prob_red →
  red_balls = 200 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l1204_120475


namespace NUMINAMATH_CALUDE_cookie_cost_l1204_120430

def total_spent : ℕ := 53
def candy_cost : ℕ := 14

theorem cookie_cost : total_spent - candy_cost = 39 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_l1204_120430


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_280_5_l1204_120447

/-- Triangle DEF with inscribed circle -/
structure Triangle where
  /-- Side DE of the triangle -/
  de : ℝ
  /-- Side DF of the triangle -/
  df : ℝ
  /-- Side EF of the triangle -/
  ef : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Distance from D to the tangency point P on DE -/
  dp : ℝ
  /-- Distance from E to the tangency point P on DE -/
  pe : ℝ

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.de + t.df + t.ef

/-- Theorem: The perimeter of the given triangle is 280.5 -/
theorem triangle_perimeter_is_280_5 (t : Triangle) 
  (h_r : t.r = 15)
  (h_dp : t.dp = 15)
  (h_pe : t.pe = 18) :
  perimeter t = 280.5 := by
  sorry

#eval 280.5

end NUMINAMATH_CALUDE_triangle_perimeter_is_280_5_l1204_120447


namespace NUMINAMATH_CALUDE_consecutive_shots_count_l1204_120496

/-- The number of ways to arrange 3 successful shots out of 8 attempts, 
    with exactly 2 consecutive successful shots. -/
def consecutiveShots : ℕ := 30

/-- The total number of attempts. -/
def totalAttempts : ℕ := 8

/-- The number of successful shots. -/
def successfulShots : ℕ := 3

/-- The number of consecutive successful shots required. -/
def consecutiveHits : ℕ := 2

theorem consecutive_shots_count :
  consecutiveShots = 
    (totalAttempts - successfulShots + 1).choose 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_shots_count_l1204_120496


namespace NUMINAMATH_CALUDE_percentage_relation_l1204_120440

theorem percentage_relation (j k l m x : ℝ) 
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100))
  (hj : j > 0) (hk : k > 0) (hl : l > 0) (hm : m > 0) :
  x = 500 := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l1204_120440


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l1204_120405

theorem arithmetic_sequence_count : 
  let a₁ : ℝ := 4.5
  let aₙ : ℝ := 56.5
  let d : ℝ := 4
  let n := (aₙ - a₁) / d + 1
  n = 14 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l1204_120405


namespace NUMINAMATH_CALUDE_sum_reciprocal_equality_l1204_120408

theorem sum_reciprocal_equality (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c ≠ 0) (h5 : (a + b) / (a * b) + 1 / c = 1 / (a + b + c)) :
  (∀ n : ℕ, 1 / a^n + 1 / b^n + 1 / c^n = 1 / (a^n + b^n + c^n)) := by
  sorry

#check sum_reciprocal_equality

end NUMINAMATH_CALUDE_sum_reciprocal_equality_l1204_120408


namespace NUMINAMATH_CALUDE_g_of_7_eq_92_l1204_120417

def g (n : ℕ) : ℕ := n^2 + 2*n + 29

theorem g_of_7_eq_92 : g 7 = 92 := by sorry

end NUMINAMATH_CALUDE_g_of_7_eq_92_l1204_120417


namespace NUMINAMATH_CALUDE_candy_theorem_l1204_120467

def candy_problem (corey_candies tapanga_candies total_candies : ℕ) : Prop :=
  (tapanga_candies = corey_candies + 8) ∧
  (corey_candies = 29) ∧
  (total_candies = corey_candies + tapanga_candies)

theorem candy_theorem : ∃ (corey_candies tapanga_candies total_candies : ℕ),
  candy_problem corey_candies tapanga_candies total_candies ∧ total_candies = 66 := by
  sorry

end NUMINAMATH_CALUDE_candy_theorem_l1204_120467


namespace NUMINAMATH_CALUDE_offices_assignment_equals_factorial4_l1204_120407

/-- The number of ways to assign 4 distinct offices to 4 distinct people -/
def assignOffices : ℕ := 24

/-- The factorial of 4 -/
def factorial4 : ℕ := 4 * 3 * 2 * 1

/-- Proof that the number of ways to assign 4 distinct offices to 4 distinct people
    is equal to 4 factorial -/
theorem offices_assignment_equals_factorial4 : assignOffices = factorial4 := by
  sorry

end NUMINAMATH_CALUDE_offices_assignment_equals_factorial4_l1204_120407


namespace NUMINAMATH_CALUDE_min_values_theorem_l1204_120492

theorem min_values_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a * b = a + 3 * b) :
  (∀ x y, x > 0 → y > 0 → 3 * x * y = x + 3 * y → 3 * a + b ≤ 3 * x + y) ∧
  (∀ x y, x > 0 → y > 0 → 3 * x * y = x + 3 * y → a * b ≤ x * y) ∧
  (∀ x y, x > 0 → y > 0 → 3 * x * y = x + 3 * y → a^2 + 9 * b^2 ≤ x^2 + 9 * y^2) ∧
  (3 * a + b = 16 / 3 ∨ a * b = 4 / 3 ∨ a^2 + 9 * b^2 = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_values_theorem_l1204_120492


namespace NUMINAMATH_CALUDE_pets_remaining_l1204_120444

theorem pets_remaining (initial_pets : ℕ) (lost_pets : ℕ) (death_rate : ℚ) : 
  initial_pets = 16 → 
  lost_pets = 6 → 
  death_rate = 1/5 → 
  initial_pets - lost_pets - (death_rate * (initial_pets - lost_pets : ℚ)).floor = 8 := by
  sorry

end NUMINAMATH_CALUDE_pets_remaining_l1204_120444


namespace NUMINAMATH_CALUDE_chess_club_mixed_groups_l1204_120427

/-- Represents the chess club scenario -/
structure ChessClub where
  total_children : ℕ
  num_groups : ℕ
  children_per_group : ℕ
  boy_vs_boy_games : ℕ
  girl_vs_girl_games : ℕ

/-- The number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ :=
  (club.total_children - club.boy_vs_boy_games - club.girl_vs_girl_games) / 2

/-- The theorem stating the number of mixed groups in the given scenario -/
theorem chess_club_mixed_groups :
  let club := ChessClub.mk 90 30 3 30 14
  mixed_groups club = 23 := by sorry

end NUMINAMATH_CALUDE_chess_club_mixed_groups_l1204_120427


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1204_120443

-- Define the circles
def circle1 (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 25

-- Define the theorem
theorem circle_intersection_range (a : ℝ) :
  (a ≥ 0) →
  (∃ x y : ℝ, circle1 a x y ∧ circle2 x y) →
  2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l1204_120443


namespace NUMINAMATH_CALUDE_billboard_average_is_twenty_l1204_120442

/-- Calculates the average number of billboards seen per hour given the counts for three consecutive hours. -/
def average_billboards (hour1 hour2 hour3 : ℕ) : ℚ :=
  (hour1 + hour2 + hour3 : ℚ) / 3

/-- Theorem stating that the average number of billboards seen per hour is 20 given the specific counts. -/
theorem billboard_average_is_twenty :
  average_billboards 17 20 23 = 20 := by
  sorry

end NUMINAMATH_CALUDE_billboard_average_is_twenty_l1204_120442


namespace NUMINAMATH_CALUDE_min_value_theorem_l1204_120425

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ (3 * x + 4 * y = 5 ↔ x = 1 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1204_120425


namespace NUMINAMATH_CALUDE_ellipse_second_focus_x_coordinate_l1204_120499

-- Define the ellipse properties
structure Ellipse where
  inFirstQuadrant : Bool
  tangentToXAxis : Bool
  tangentToYAxis : Bool
  focus1 : ℝ × ℝ
  tangentToY1 : Bool

-- Define the theorem
theorem ellipse_second_focus_x_coordinate
  (e : Ellipse)
  (h1 : e.inFirstQuadrant = true)
  (h2 : e.tangentToXAxis = true)
  (h3 : e.tangentToYAxis = true)
  (h4 : e.focus1 = (4, 9))
  (h5 : e.tangentToY1 = true) :
  ∃ d : ℝ, d = 16 ∧ (∃ y : ℝ, (d, y) = e.focus1 ∨ (d, 9) ≠ e.focus1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_second_focus_x_coordinate_l1204_120499


namespace NUMINAMATH_CALUDE_triangle_equal_sides_l1204_120411

/-- 
Given a triangle with one side of length 6 cm and two equal sides,
where the sum of all sides is 20 cm, prove that each of the equal sides is 7 cm long.
-/
theorem triangle_equal_sides (a b c : ℝ) : 
  a = 6 → -- One side is 6 cm
  b = c → -- Two sides are equal
  a + b + c = 20 → -- Sum of all sides is 20 cm
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_equal_sides_l1204_120411


namespace NUMINAMATH_CALUDE_no_common_root_l1204_120400

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ x : ℝ, (x^2 + b*x + c = 0) ∧ (x^2 + a*x + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_common_root_l1204_120400


namespace NUMINAMATH_CALUDE_book_cost_problem_l1204_120406

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ)
  (h_total : total_cost = 540)
  (h_loss : loss_percent = 15)
  (h_gain : gain_percent = 19)
  (h_equal_sell : (1 - loss_percent / 100) * cost_loss = (1 + gain_percent / 100) * (total_cost - cost_loss)) :
  ∃ (cost_loss : ℝ), cost_loss = 315 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l1204_120406


namespace NUMINAMATH_CALUDE_first_group_size_l1204_120431

/-- Represents the work done by a group of workers --/
def work (persons : ℕ) (days : ℕ) (hours : ℕ) : ℕ := persons * days * hours

/-- Proves that the number of persons in the first group is 45 --/
theorem first_group_size :
  ∃ (P : ℕ), work P 12 5 = work 30 15 6 ∧ P = 45 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l1204_120431


namespace NUMINAMATH_CALUDE_base_conversion_1623_to_base7_l1204_120484

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (a b c d : Nat) : Nat :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

/-- Theorem: 1623 in base 10 is equal to 4506 in base 7 --/
theorem base_conversion_1623_to_base7 : 
  1623 = base7ToBase10 4 5 0 6 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1623_to_base7_l1204_120484


namespace NUMINAMATH_CALUDE_total_animals_seen_l1204_120472

theorem total_animals_seen (initial_beavers initial_chipmunks : ℕ) : 
  initial_beavers = 35 →
  initial_chipmunks = 60 →
  (initial_beavers + initial_chipmunks) + (3 * initial_beavers + (initial_chipmunks - 15)) = 245 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_seen_l1204_120472


namespace NUMINAMATH_CALUDE_inequality_one_l1204_120482

theorem inequality_one (x y : ℝ) : (x + 1) * (x - 2*y + 1) + y^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_one_l1204_120482


namespace NUMINAMATH_CALUDE_sams_age_two_years_ago_l1204_120494

/-- Given the ages of John and Sam, prove Sam's age two years ago -/
theorem sams_age_two_years_ago (john_age sam_age : ℕ) : 
  john_age = 3 * sam_age →
  john_age + 9 = 2 * (sam_age + 9) →
  sam_age - 2 = 7 := by
  sorry

#check sams_age_two_years_ago

end NUMINAMATH_CALUDE_sams_age_two_years_ago_l1204_120494


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l1204_120465

theorem quadratic_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (2 * x₁^2 - 7 * x₁ + 1 = x₁ + 31) ∧
  (2 * x₂^2 - 7 * x₂ + 1 = x₂ + 31) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 2 * Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_difference_l1204_120465


namespace NUMINAMATH_CALUDE_percentage_difference_l1204_120413

theorem percentage_difference : (60 / 100 * 50) - (40 / 100 * 30) = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1204_120413


namespace NUMINAMATH_CALUDE_musicians_performing_l1204_120441

/-- Represents a musical group --/
inductive MusicalGroup
| Quartet
| Trio
| Duet

/-- The number of musicians in each type of group --/
def group_size (g : MusicalGroup) : ℕ :=
  match g with
  | MusicalGroup.Quartet => 4
  | MusicalGroup.Trio => 3
  | MusicalGroup.Duet => 2

/-- The original schedule of performances --/
def original_schedule : List (MusicalGroup × ℕ) :=
  [(MusicalGroup.Quartet, 4), (MusicalGroup.Duet, 5), (MusicalGroup.Trio, 6)]

/-- The changes to the schedule --/
def schedule_changes : List (MusicalGroup × ℕ) :=
  [(MusicalGroup.Quartet, 1), (MusicalGroup.Duet, 2), (MusicalGroup.Trio, 1)]

/-- Calculate the total number of musicians given a schedule --/
def total_musicians (schedule : List (MusicalGroup × ℕ)) : ℕ :=
  schedule.foldl (fun acc (g, n) => acc + n * group_size g) 0

/-- The main theorem --/
theorem musicians_performing (
  orig_schedule : List (MusicalGroup × ℕ)) 
  (changes : List (MusicalGroup × ℕ)) :
  orig_schedule = original_schedule →
  changes = schedule_changes →
  total_musicians orig_schedule - 
  (total_musicians changes + 1) = 35 := by
  sorry

end NUMINAMATH_CALUDE_musicians_performing_l1204_120441


namespace NUMINAMATH_CALUDE_f_f_one_eq_one_l1204_120481

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x - 1 else -x^2 - 2*x

theorem f_f_one_eq_one : f (f 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_f_one_eq_one_l1204_120481


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1204_120446

/-- The standard equation of a hyperbola passing through specific points and sharing asymptotes with another hyperbola -/
theorem hyperbola_equation : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ (x y : ℝ), y^2 / a - x^2 / b = 1 ↔ 
    ((x = -3 ∧ y = 2 * Real.sqrt 7) ∨ 
     (x = -6 * Real.sqrt 2 ∧ y = -7) ∨ 
     (x = 2 ∧ y = 2 * Real.sqrt 3)) ∧
    (∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), y^2 / a - x^2 / b = 1 ↔ x^2 / 4 - y^2 / 3 = k)) ∧
  a = 9 ∧ b = 12 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1204_120446


namespace NUMINAMATH_CALUDE_smallest_with_12_divisors_l1204_120474

/-- The number of positive integer divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- n has exactly 12 positive integer divisors -/
def has12Divisors (n : ℕ) : Prop :=
  numDivisors n = 12

/-- 72 is the smallest positive integer with exactly 12 positive integer divisors -/
theorem smallest_with_12_divisors :
  has12Divisors 72 ∧ ∀ m : ℕ, 0 < m → m < 72 → ¬has12Divisors m := by sorry

end NUMINAMATH_CALUDE_smallest_with_12_divisors_l1204_120474


namespace NUMINAMATH_CALUDE_age_difference_correct_l1204_120461

/-- The age difference between Emma and her sister -/
def age_difference : ℕ := 9

/-- Emma's age when her sister is 56 -/
def emma_future_age : ℕ := 47

/-- Emma's sister's age when Emma is 47 -/
def sister_future_age : ℕ := 56

/-- Proof that the age difference is correct -/
theorem age_difference_correct : 
  emma_future_age + age_difference = sister_future_age :=
by sorry

end NUMINAMATH_CALUDE_age_difference_correct_l1204_120461


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l1204_120463

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : IsEven f) : IsEven (f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l1204_120463


namespace NUMINAMATH_CALUDE_intersection_distance_from_origin_l1204_120448

/-- Given two lines l₁ and l₂ defined by their equations, prove that their intersection point P
    is always at a distance of 2 from the origin O, regardless of the value of m. -/
theorem intersection_distance_from_origin (m : ℝ) :
  let l₁ := {p : ℝ × ℝ | p.1 + m * p.2 - 2 = 0}
  let l₂ := {p : ℝ × ℝ | m * p.1 - p.2 + 2 * m = 0}
  ∀ P ∈ l₁ ∩ l₂, Real.sqrt (P.1^2 + P.2^2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_from_origin_l1204_120448


namespace NUMINAMATH_CALUDE_garden_area_l1204_120480

/-- A rectangular garden with width one-third of its length and perimeter 72 meters has an area of 243 square meters. -/
theorem garden_area (width length : ℝ) : 
  width > 0 ∧ 
  length > 0 ∧ 
  width = length / 3 ∧ 
  2 * (width + length) = 72 →
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l1204_120480


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l1204_120401

def rahul_age_after_6_years : ℕ := 26
def years_until_rahul_age : ℕ := 6
def deepak_current_age : ℕ := 8

theorem rahul_deepak_age_ratio :
  let rahul_current_age := rahul_age_after_6_years - years_until_rahul_age
  (rahul_current_age : ℚ) / deepak_current_age = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l1204_120401


namespace NUMINAMATH_CALUDE_square_corner_distance_l1204_120456

theorem square_corner_distance (small_perimeter large_area : ℝ) 
  (h_small : small_perimeter = 8)
  (h_large : large_area = 36) : ∃ (distance : ℝ), distance = Real.sqrt 32 :=
by
  sorry

end NUMINAMATH_CALUDE_square_corner_distance_l1204_120456


namespace NUMINAMATH_CALUDE_h_function_iff_increasing_or_constant_l1204_120460

/-- Definition of an "H function" -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ ≥ x₁ * f x₂ + x₂ * f x₁

/-- A function is increasing -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- A function is constant -/
def is_constant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y

theorem h_function_iff_increasing_or_constant (f : ℝ → ℝ) :
  is_h_function f ↔ is_increasing f ∨ is_constant f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_increasing_or_constant_l1204_120460


namespace NUMINAMATH_CALUDE_oliver_earnings_l1204_120449

def laundry_shop_earnings (price_per_kilo : ℕ) (day1_kilos : ℕ) : ℕ :=
  let day2_kilos := day1_kilos + 5
  let day3_kilos := 2 * day2_kilos
  price_per_kilo * (day1_kilos + day2_kilos + day3_kilos)

theorem oliver_earnings :
  laundry_shop_earnings 2 5 = 70 :=
by sorry

end NUMINAMATH_CALUDE_oliver_earnings_l1204_120449


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1204_120471

/-- If a 60° arc on circle A has the same length as a 40° arc on circle B,
    then the ratio of the area of circle A to the area of circle B is 4/9 -/
theorem circle_area_ratio (r_A r_B : ℝ) (h : r_A > 0 ∧ r_B > 0) :
  (60 / 360) * (2 * Real.pi * r_A) = (40 / 360) * (2 * Real.pi * r_B) →
  (Real.pi * r_A ^ 2) / (Real.pi * r_B ^ 2) = 4 / 9 := by
sorry


end NUMINAMATH_CALUDE_circle_area_ratio_l1204_120471


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l1204_120459

/-- A quadrilateral with vertices at (1,2), (4,5), (5,4), and (4,1) has a perimeter of 4√2 + 2√10 -/
theorem quadrilateral_perimeter : 
  let vertices : List (ℝ × ℝ) := [(1, 2), (4, 5), (5, 4), (4, 1)]
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := (List.zip vertices (vertices.rotateLeft 1)).map (fun (p, q) => distance p q) |>.sum
  perimeter = 4 * Real.sqrt 2 + 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l1204_120459


namespace NUMINAMATH_CALUDE_b_over_a_is_sqrt_2_angle_B_is_45_degrees_l1204_120438

noncomputable section

variable (A B C : ℝ)
variable (a b c : ℝ)

-- Define the triangle ABC
axiom triangle_abc : a > 0 ∧ b > 0 ∧ c > 0

-- Define the relationship between sides and angles
axiom sine_law : a / Real.sin A = b / Real.sin B

-- Given conditions
axiom condition1 : a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = Real.sqrt 2 * a
axiom condition2 : c ^ 2 = b ^ 2 + Real.sqrt 3 * a ^ 2

-- Theorems to prove
theorem b_over_a_is_sqrt_2 : b / a = Real.sqrt 2 := by sorry

theorem angle_B_is_45_degrees : B = Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_b_over_a_is_sqrt_2_angle_B_is_45_degrees_l1204_120438


namespace NUMINAMATH_CALUDE_problem_statement_l1204_120473

theorem problem_statement (a b : ℚ) 
  (h1 : 3 * a + 4 * b = 0) 
  (h2 : a = 2 * b - 3) : 
  5 * b = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1204_120473


namespace NUMINAMATH_CALUDE_dividend_calculation_l1204_120476

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 18) 
  (h2 : quotient = 9) 
  (h3 : remainder = 5) : 
  divisor * quotient + remainder = 167 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1204_120476


namespace NUMINAMATH_CALUDE_f_minimum_value_l1204_120432

noncomputable def f (x : ℝ) : ℝ := |2 * Real.sqrt x * (Real.log (2 * x) / Real.log (Real.sqrt 2))|

theorem f_minimum_value :
  (∀ x > 0, f x ≥ 0) ∧ (∃ x > 0, f x = 0) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1204_120432


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1204_120470

theorem at_least_one_not_less_than_two (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1204_120470


namespace NUMINAMATH_CALUDE_system_solution_unique_l1204_120402

theorem system_solution_unique :
  ∃! (x y : ℚ),
    1 / (2 - x + 2 * y) - 1 / (x + 2 * y - 1) = 2 ∧
    1 / (2 - x + 2 * y) - 1 / (1 - x - 2 * y) = 4 ∧
    x = 11 / 6 ∧
    y = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1204_120402


namespace NUMINAMATH_CALUDE_circle_area_increase_l1204_120437

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l1204_120437


namespace NUMINAMATH_CALUDE_ellipse_range_l1204_120429

-- Define the ellipse
def on_ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 12

-- Define the function we're interested in
def f (x y : ℝ) : ℝ := x + 2 * y

-- Theorem statement
theorem ellipse_range :
  ∀ x y : ℝ, on_ellipse x y → -Real.sqrt 22 ≤ f x y ∧ f x y ≤ Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_range_l1204_120429


namespace NUMINAMATH_CALUDE_circle_area_difference_l1204_120421

/-- The difference in area between a circle with radius 30 inches and a circle with circumference 60π inches is 0 square inches. -/
theorem circle_area_difference : 
  let r1 : ℝ := 30
  let c2 : ℝ := 60 * Real.pi
  let r2 : ℝ := c2 / (2 * Real.pi)
  let area1 : ℝ := Real.pi * r1^2
  let area2 : ℝ := Real.pi * r2^2
  area1 - area2 = 0 := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1204_120421


namespace NUMINAMATH_CALUDE_impossible_coloring_exists_l1204_120410

-- Define the grid
def Grid := ℤ × ℤ

-- Define a chessboard polygon
def ChessboardPolygon := Set Grid

-- Define a coloring of the grid
def Coloring := Grid → Bool

-- Define congruence for chessboard polygons
def Congruent (F G : ChessboardPolygon) : Prop := sorry

-- Define the number of green cells in a polygon given a coloring
def GreenCells (F : ChessboardPolygon) (c : Coloring) : ℕ := sorry

-- The main theorem
theorem impossible_coloring_exists :
  ∃ F : ChessboardPolygon,
    ∀ c : Coloring,
      (∃ G : ChessboardPolygon, Congruent F G ∧ GreenCells G c = 0) ∨
      (∃ G : ChessboardPolygon, Congruent F G ∧ GreenCells G c > 2020) := by
  sorry

end NUMINAMATH_CALUDE_impossible_coloring_exists_l1204_120410


namespace NUMINAMATH_CALUDE_absolute_difference_inequality_l1204_120426

theorem absolute_difference_inequality (x : ℝ) : 
  |x - 1| - |x - 2| > (1/2) ↔ x > (7/4) := by sorry

end NUMINAMATH_CALUDE_absolute_difference_inequality_l1204_120426


namespace NUMINAMATH_CALUDE_mileage_pay_is_104_l1204_120478

/-- Calculates the mileage pay for a delivery driver given the distances for three packages and the pay rate per mile. -/
def calculate_mileage_pay (first_package : ℝ) (second_package : ℝ) (third_package : ℝ) (pay_rate : ℝ) : ℝ :=
  (first_package + second_package + third_package) * pay_rate

/-- Theorem stating that given specific package distances and pay rate, the mileage pay is $104. -/
theorem mileage_pay_is_104 :
  let first_package : ℝ := 10
  let second_package : ℝ := 28
  let third_package : ℝ := second_package / 2
  let pay_rate : ℝ := 2
  calculate_mileage_pay first_package second_package third_package pay_rate = 104 := by
  sorry

#check mileage_pay_is_104

end NUMINAMATH_CALUDE_mileage_pay_is_104_l1204_120478


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1204_120416

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 + 4 * x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 + 8 * x

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (14 * x - y - 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1204_120416


namespace NUMINAMATH_CALUDE_correct_propositions_l1204_120493

theorem correct_propositions :
  let prop1 := (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔ (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0)
  let prop2 := ∀ p q : Prop, (p ∨ q) → (p ∧ q)
  let prop3 := ∀ p q : Prop, ¬(p ∧ q) → (¬p ∧ ¬q)
  let prop4 := (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 + x + 1 ≥ 0)
  prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4 := by sorry

end NUMINAMATH_CALUDE_correct_propositions_l1204_120493


namespace NUMINAMATH_CALUDE_unique_solution_l1204_120415

theorem unique_solution (a b c : ℝ) 
  (ha : a > 4) (hb : b > 4) (hc : c > 4)
  (heq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 48) :
  a = 13 ∧ b = 11 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1204_120415
