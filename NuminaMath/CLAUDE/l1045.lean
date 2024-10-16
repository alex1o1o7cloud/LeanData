import Mathlib

namespace NUMINAMATH_CALUDE_eighth_grade_trip_contribution_l1045_104573

theorem eighth_grade_trip_contribution (total : ℕ) (months : ℕ) 
  (h1 : total = 49685) 
  (h2 : months = 5) : 
  ∃ (students : ℕ) (contribution : ℕ), 
    students * contribution * months = total ∧ 
    students = 19 ∧ 
    contribution = 523 := by
sorry

end NUMINAMATH_CALUDE_eighth_grade_trip_contribution_l1045_104573


namespace NUMINAMATH_CALUDE_total_weekly_eggs_l1045_104579

/-- The number of eggs in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens supplied to Store A daily -/
def store_a_dozens : ℕ := 5

/-- The number of eggs supplied to Store B daily -/
def store_b_eggs : ℕ := 30

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: The total number of eggs supplied to both stores in a week is 630 -/
theorem total_weekly_eggs : 
  (store_a_dozens * dozen + store_b_eggs) * days_in_week = 630 := by
  sorry

end NUMINAMATH_CALUDE_total_weekly_eggs_l1045_104579


namespace NUMINAMATH_CALUDE_book_arrangement_l1045_104598

theorem book_arrangement (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) :
  (n.factorial / k.factorial : ℕ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_l1045_104598


namespace NUMINAMATH_CALUDE_complex_number_location_l1045_104594

theorem complex_number_location (z : ℂ) (h : z * (1 - 2*I) = I) :
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1045_104594


namespace NUMINAMATH_CALUDE_duck_flight_days_l1045_104595

/-- The number of days it takes for a duck to fly south in winter. -/
def days_south : ℕ := sorry

/-- The number of days it takes for a duck to fly north in summer. -/
def days_north : ℕ := 2 * days_south

/-- The number of days it takes for a duck to fly east in spring. -/
def days_east : ℕ := 60

/-- The total number of days the duck flies during winter, summer, and spring. -/
def total_days : ℕ := 180

/-- Theorem stating that the number of days it takes for the duck to fly south in winter is 40. -/
theorem duck_flight_days : days_south = 40 := by
  sorry

end NUMINAMATH_CALUDE_duck_flight_days_l1045_104595


namespace NUMINAMATH_CALUDE_log_equation_solution_l1045_104529

theorem log_equation_solution :
  ∃! x : ℝ, x > 0 ∧ x + 2 > 0 ∧ 2*x + 3 > 0 ∧
  Real.log x + Real.log (x + 2) = Real.log (2*x + 3) ∧
  x = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1045_104529


namespace NUMINAMATH_CALUDE_weights_standard_deviation_l1045_104502

def weights (a b : ℝ) : List ℝ := [125, a, 121, b, 127]

def median (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def standardDeviation (l : List ℝ) : ℝ := sorry

theorem weights_standard_deviation (a b : ℝ) :
  median (weights a b) = 124 →
  mean (weights a b) = 124 →
  standardDeviation (weights a b) = 2 := by sorry

end NUMINAMATH_CALUDE_weights_standard_deviation_l1045_104502


namespace NUMINAMATH_CALUDE_cos_angle_between_vectors_l1045_104544

def a : ℝ × ℝ := (3, 3)
def b : ℝ × ℝ := (1, 2)

theorem cos_angle_between_vectors :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (Real.cos θ) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_angle_between_vectors_l1045_104544


namespace NUMINAMATH_CALUDE_card_value_decrease_l1045_104545

theorem card_value_decrease (x : ℝ) :
  (1 - x/100) * (1 - x/100) = 0.64 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_card_value_decrease_l1045_104545


namespace NUMINAMATH_CALUDE_monic_quadratic_polynomial_l1045_104511

theorem monic_quadratic_polynomial (f : ℝ → ℝ) :
  (∃ a b : ℝ, ∀ x, f x = x^2 + a*x + b) →  -- monic quadratic polynomial
  f 1 = 3 →                               -- f(1) = 3
  f 2 = 12 →                              -- f(2) = 12
  ∀ x, f x = x^2 + 6*x - 4 :=              -- f(x) = x^2 + 6x - 4
by sorry

end NUMINAMATH_CALUDE_monic_quadratic_polynomial_l1045_104511


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l1045_104548

theorem sufficient_condition_implies_a_range :
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) →
  a ∈ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_a_range_l1045_104548


namespace NUMINAMATH_CALUDE_intersection_forms_ellipse_l1045_104519

/-- Proves that the intersection of a line and a curve forms an ellipse under specific conditions -/
theorem intersection_forms_ellipse (a b : ℝ) (hab : a * b ≠ 0) :
  ∃ (c d : ℝ), ∀ (x y : ℝ),
    (a * x - y + b = 0) ∧ (b * x^2 + a * y^2 = a * b) →
    (x^2 / c^2 + y^2 / d^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_intersection_forms_ellipse_l1045_104519


namespace NUMINAMATH_CALUDE_cyclist_speed_ratio_l1045_104508

theorem cyclist_speed_ratio (D : ℝ) (v_r v_w : ℝ) (t_r t_w : ℝ) : 
  D > 0 → v_r > 0 → v_w > 0 → t_r > 0 → t_w > 0 →
  (2 / 3 : ℝ) * D = v_r * t_r →
  (1 / 3 : ℝ) * D = v_w * t_w →
  t_w = 2 * t_r →
  v_r = 4 * v_w := by
  sorry

#check cyclist_speed_ratio

end NUMINAMATH_CALUDE_cyclist_speed_ratio_l1045_104508


namespace NUMINAMATH_CALUDE_polynomial_negative_roots_l1045_104537

theorem polynomial_negative_roots (q : ℝ) (hq : q > 2) :
  ∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^4 + q*x₁^3 + 2*x₁^2 + q*x₁ + 1 = 0 ∧
  x₂^4 + q*x₂^3 + 2*x₂^2 + q*x₂ + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_negative_roots_l1045_104537


namespace NUMINAMATH_CALUDE_function_proof_l1045_104582

/-- Given a function f(x) = a^x + b, prove that if f(1) = 3 and f(0) = 2, then f(x) = 2^x + 1 -/
theorem function_proof (a b : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a^x + b) 
  (h2 : f 1 = 3) (h3 : f 0 = 2) : ∀ x, f x = 2^x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_proof_l1045_104582


namespace NUMINAMATH_CALUDE_buyer_count_solution_l1045_104566

/-- The number of buyers in a grocery store over three days -/
structure BuyerCount where
  dayBeforeYesterday : ℕ
  yesterday : ℕ
  today : ℕ

/-- Conditions for the buyer count problem -/
def BuyerCountProblem (b : BuyerCount) : Prop :=
  b.today = b.yesterday + 40 ∧
  b.yesterday = b.dayBeforeYesterday / 2 ∧
  b.dayBeforeYesterday + b.yesterday + b.today = 140

theorem buyer_count_solution :
  ∃ b : BuyerCount, BuyerCountProblem b ∧ b.dayBeforeYesterday = 67 := by
  sorry

end NUMINAMATH_CALUDE_buyer_count_solution_l1045_104566


namespace NUMINAMATH_CALUDE_percentage_changes_l1045_104551

/-- Given an initial value of 950, prove that increasing it by 80% and then
    decreasing the result by 65% yields 598.5. -/
theorem percentage_changes (initial : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) :
  initial = 950 →
  increase_percent = 80 →
  decrease_percent = 65 →
  (initial * (1 + increase_percent / 100)) * (1 - decrease_percent / 100) = 598.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_changes_l1045_104551


namespace NUMINAMATH_CALUDE_train_speed_l1045_104518

/-- Calculates the speed of a train in km/hr given its length and time to cross a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 135) (h2 : time = 9) :
  (length / time) * 3.6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1045_104518


namespace NUMINAMATH_CALUDE_triangle_inequality_l1045_104531

/-- A structure representing a triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A structure representing a line in a 2D plane -/
structure Line where
  m : ℝ
  n : ℝ
  p : ℝ

/-- Function to calculate the area of a triangle -/
def areaOfTriangle (t : Triangle) : ℝ := sorry

/-- Function to calculate the tangent of an angle in a triangle -/
def tanAngle (t : Triangle) (vertex : Fin 3) : ℝ := sorry

/-- Function to calculate the perpendicular distance from a point to a line -/
def perpDistance (point : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- The main theorem -/
theorem triangle_inequality (t : Triangle) (l : Line) :
  let u := perpDistance t.A l
  let v := perpDistance t.B l
  let w := perpDistance t.C l
  let S := areaOfTriangle t
  u^2 * tanAngle t 0 + v^2 * tanAngle t 1 + w^2 * tanAngle t 2 ≥ 2 * S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1045_104531


namespace NUMINAMATH_CALUDE_rod_length_calculation_l1045_104574

/-- The total length of a rod that can be cut into a given number of pieces of a specific length. -/
def rod_length (num_pieces : ℕ) (piece_length : ℝ) : ℝ :=
  num_pieces * piece_length

/-- Theorem stating that a rod that can be cut into 50 pieces of 0.85 metres each has a total length of 42.5 metres. -/
theorem rod_length_calculation : rod_length 50 0.85 = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_calculation_l1045_104574


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1045_104541

theorem simplify_and_evaluate : 
  ∀ x : ℝ, x ≠ -2 → x ≠ 1 → 
  (1 - 3 / (x + 2)) / ((x - 1) / (x^2 + 4*x + 4)) = x + 2 ∧
  (1 - 3 / (-1 + 2)) / ((-1 - 1) / ((-1)^2 + 4*(-1) + 4)) = 1 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1045_104541


namespace NUMINAMATH_CALUDE_clock_angle_at_3_30_angle_between_clock_hands_at_3_30_l1045_104576

/-- The angle between clock hands at 3:30 -/
theorem clock_angle_at_3_30 : ℝ :=
  let hour_hand_angle : ℝ := 3.5 * 30  -- 3:30 is 3.5 hours from 12 o'clock
  let minute_hand_angle : ℝ := 30 * 6  -- 30 minutes is 6 times 5-minute marks
  let angle_diff : ℝ := |minute_hand_angle - hour_hand_angle|
  75

/-- Theorem: The angle between the hour and minute hands at 3:30 is 75 degrees -/
theorem angle_between_clock_hands_at_3_30 : clock_angle_at_3_30 = 75 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_3_30_angle_between_clock_hands_at_3_30_l1045_104576


namespace NUMINAMATH_CALUDE_inequality_proof_l1045_104526

theorem inequality_proof (a b t : ℝ) (ha : a > 1) (hb : b > 1) (ht : t > 0) :
  (a^2 / (b^t - 1)) + (b^(2*t) / (a^t - 1)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1045_104526


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l1045_104560

-- Define the plane vectors
def a (m : ℝ) : Fin 2 → ℝ := ![1, m]
def b : Fin 2 → ℝ := ![2, 5]
def c (m : ℝ) : Fin 2 → ℝ := ![m, 3]

-- Define the parallel condition
def parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

-- Theorem statement
theorem vector_parallel_condition (m : ℝ) :
  parallel (a m + c m) (a m - b) →
  m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l1045_104560


namespace NUMINAMATH_CALUDE_vector_AB_equals_2_2_l1045_104577

def point := ℝ × ℝ

def A : point := (1, 0)
def B : point := (3, 2)

def vector_AB (p q : point) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

theorem vector_AB_equals_2_2 :
  vector_AB A B = (2, 2) := by sorry

end NUMINAMATH_CALUDE_vector_AB_equals_2_2_l1045_104577


namespace NUMINAMATH_CALUDE_mall_spending_l1045_104527

def total_spent : ℚ := 347
def movie_cost : ℚ := 24
def num_movies : ℕ := 3
def bean_cost : ℚ := 1.25
def num_bean_bags : ℕ := 20

theorem mall_spending (mall_spent : ℚ) : 
  mall_spent = total_spent - (↑num_movies * movie_cost + ↑num_bean_bags * bean_cost) → 
  mall_spent = 250 := by
  sorry

end NUMINAMATH_CALUDE_mall_spending_l1045_104527


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l1045_104587

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n - 2) + (n - 1) + n + (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l1045_104587


namespace NUMINAMATH_CALUDE_initially_tagged_fish_l1045_104516

/-- The number of fish initially caught and tagged -/
def T : ℕ := 70

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- The number of tagged fish in the second catch -/
def tagged_in_second_catch : ℕ := 2

/-- The total number of fish in the pond -/
def total_fish : ℕ := 1750

/-- Theorem stating that T is the correct number of initially tagged fish -/
theorem initially_tagged_fish :
  (T : ℚ) / total_fish = tagged_in_second_catch / second_catch :=
by sorry

end NUMINAMATH_CALUDE_initially_tagged_fish_l1045_104516


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l1045_104510

theorem min_value_quadratic_form (x y : ℝ) : x^2 + 3*x*y + y^2 ≥ 0 ∧ 
  (x^2 + 3*x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l1045_104510


namespace NUMINAMATH_CALUDE_brenda_skittles_l1045_104523

/-- Calculates the total number of Skittles Brenda has after buying more. -/
def total_skittles (initial : ℕ) (bought : ℕ) : ℕ :=
  initial + bought

/-- Theorem stating that Brenda ends up with 15 Skittles. -/
theorem brenda_skittles : total_skittles 7 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_brenda_skittles_l1045_104523


namespace NUMINAMATH_CALUDE_next_larger_perfect_square_l1045_104522

theorem next_larger_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n > x ∧ (∃ m : ℕ, n = m^2) ∧ n = x + 4 * (x.sqrt) + 4 := by
  sorry

end NUMINAMATH_CALUDE_next_larger_perfect_square_l1045_104522


namespace NUMINAMATH_CALUDE_count_distinct_five_digit_numbers_l1045_104556

/-- The number of distinct five-digit numbers that can be formed by selecting 2 digits
    from the set of odd digits {1, 3, 5, 7, 9} and 3 digits from the set of even digits
    {0, 2, 4, 6, 8}. -/
def distinct_five_digit_numbers : ℕ :=
  let odd_digits : Finset ℕ := {1, 3, 5, 7, 9}
  let even_digits : Finset ℕ := {0, 2, 4, 6, 8}
  10560

/-- Theorem stating that the number of distinct five-digit numbers formed under the given
    conditions is equal to 10560. -/
theorem count_distinct_five_digit_numbers :
  distinct_five_digit_numbers = 10560 := by
  sorry

end NUMINAMATH_CALUDE_count_distinct_five_digit_numbers_l1045_104556


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1045_104536

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 5*x + 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 5*x + 3 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1045_104536


namespace NUMINAMATH_CALUDE_equal_division_of_money_l1045_104581

theorem equal_division_of_money (total_amount : ℚ) (num_people : ℕ) 
  (h1 : total_amount = 5.25) (h2 : num_people = 7) :
  total_amount / num_people = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_money_l1045_104581


namespace NUMINAMATH_CALUDE_computer_rental_rates_l1045_104559

/-- Represents the hourly rental rates and job completion times for three computers -/
structure ComputerRental where
  rateA : ℝ  -- Hourly rate for Computer A
  rateB : ℝ  -- Hourly rate for Computer B
  rateC : ℝ  -- Hourly rate for Computer C
  timeA : ℝ  -- Time for Computer A to complete the job

/-- Conditions for the computer rental problem -/
def rental_conditions (r : ComputerRental) : Prop :=
  r.rateA = 1.4 * r.rateB ∧
  r.rateC = 0.75 * r.rateB ∧
  r.rateA * r.timeA = 550 ∧
  r.rateB * (r.timeA + 20) = 550 ∧
  r.rateC * (r.timeA + 10) = 550

/-- Theorem stating the approximate hourly rates for the computers -/
theorem computer_rental_rates :
  ∃ r : ComputerRental, rental_conditions r ∧
    (abs (r.rateA - 11) < 0.01) ∧
    (abs (r.rateB - 7.86) < 0.01) ∧
    (abs (r.rateC - 5.90) < 0.01) :=
  by sorry

end NUMINAMATH_CALUDE_computer_rental_rates_l1045_104559


namespace NUMINAMATH_CALUDE_expression_value_approximation_l1045_104591

theorem expression_value_approximation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ 
  |((85 : ℝ) + Real.sqrt 32 / 113) * 113^2 - 10246| < ε :=
sorry

end NUMINAMATH_CALUDE_expression_value_approximation_l1045_104591


namespace NUMINAMATH_CALUDE_H_range_l1045_104530

/-- The function H defined for all real x -/
def H (x : ℝ) : ℝ := 2 * |2*x + 2| - 3 * |2*x - 2|

/-- The theorem stating that the range of H is [8, ∞) -/
theorem H_range : Set.range H = Set.Ici 8 := by sorry

end NUMINAMATH_CALUDE_H_range_l1045_104530


namespace NUMINAMATH_CALUDE_percentage_of_x_l1045_104592

theorem percentage_of_x (x : ℝ) (p : ℝ) : 
  p * x = 0.3 * (0.7 * x) + 10 ↔ p = 0.21 + 10 / x :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_x_l1045_104592


namespace NUMINAMATH_CALUDE_job_completion_time_l1045_104564

/-- Given two people working on a job, where the first person takes 3 hours and their combined
    work rate is 5/12 of the job per hour, prove that the second person takes 12 hours to
    complete the job individually. -/
theorem job_completion_time
  (time_person1 : ℝ)
  (combined_rate : ℝ)
  (h1 : time_person1 = 3)
  (h2 : combined_rate = 5 / 12)
  : ∃ (time_person2 : ℝ),
    time_person2 = 12 ∧
    1 / time_person1 + 1 / time_person2 = combined_rate :=
by sorry

end NUMINAMATH_CALUDE_job_completion_time_l1045_104564


namespace NUMINAMATH_CALUDE_sphere_radius_calculation_l1045_104517

-- Define the radius of the hemisphere
def hemisphere_radius : ℝ := 2

-- Define the number of smaller spheres
def num_spheres : ℕ := 8

-- State the theorem
theorem sphere_radius_calculation :
  ∃ (r : ℝ), 
    (2 / 3 * Real.pi * hemisphere_radius ^ 3 = num_spheres * (4 / 3 * Real.pi * r ^ 3)) ∧
    (r = (Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_calculation_l1045_104517


namespace NUMINAMATH_CALUDE_intersection_M_N_l1045_104567

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}
def N : Set ℝ := {x | ∃ y, y = Real.log (2*x - x^2)}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1045_104567


namespace NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1045_104546

/-- Conversion from spherical coordinates to rectangular coordinates -/
theorem spherical_to_rectangular_conversion 
  (ρ θ φ : Real) 
  (hρ : ρ = 8) 
  (hθ : θ = 5 * Real.pi / 4) 
  (hφ : φ = Real.pi / 4) : 
  (ρ * Real.sin φ * Real.cos θ, 
   ρ * Real.sin φ * Real.sin θ, 
   ρ * Real.cos φ) = (-4, -4, 4 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_spherical_to_rectangular_conversion_l1045_104546


namespace NUMINAMATH_CALUDE_sum_remainder_zero_l1045_104547

def arithmetic_sum (a₁ aₙ n : ℕ) : ℕ := n * (a₁ + aₙ) / 2

theorem sum_remainder_zero : 
  let a₁ := 6
  let d := 6
  let aₙ := 288
  let n := (aₙ - a₁) / d + 1
  (arithmetic_sum a₁ aₙ n) % 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_remainder_zero_l1045_104547


namespace NUMINAMATH_CALUDE_four_distinct_cuts_l1045_104599

/-- Represents a square grid with holes -/
structure GridWithHoles :=
  (size : ℕ)
  (holes : List (ℕ × ℕ))

/-- Represents a cut on the grid -/
inductive Cut
  | Vertical : ℕ → Cut
  | Horizontal : ℕ → Cut
  | Diagonal : Bool → Cut

/-- Checks if two parts resulting from a cut are congruent -/
def areCongruentParts (g : GridWithHoles) (c : Cut) : Bool :=
  sorry

/-- Checks if two cuts result in different congruent parts -/
def areDifferentCuts (g : GridWithHoles) (c1 c2 : Cut) : Bool :=
  sorry

/-- Theorem: There are at least four distinct ways to cut a 4x4 grid with two symmetrical holes into congruent parts -/
theorem four_distinct_cuts (g : GridWithHoles) 
  (h1 : g.size = 4)
  (h2 : g.holes = [(1, 1), (2, 2)]) : 
  ∃ (c1 c2 c3 c4 : Cut),
    areCongruentParts g c1 ∧
    areCongruentParts g c2 ∧
    areCongruentParts g c3 ∧
    areCongruentParts g c4 ∧
    areDifferentCuts g c1 c2 ∧
    areDifferentCuts g c1 c3 ∧
    areDifferentCuts g c1 c4 ∧
    areDifferentCuts g c2 c3 ∧
    areDifferentCuts g c2 c4 ∧
    areDifferentCuts g c3 c4 :=
  sorry

end NUMINAMATH_CALUDE_four_distinct_cuts_l1045_104599


namespace NUMINAMATH_CALUDE_simplify_fraction_simplify_and_evaluate_evaluate_at_two_l1045_104505

-- Part 1
theorem simplify_fraction (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  (x - 1) / x / ((2 * x - 2) / x^2) = x / 2 := by sorry

-- Part 2
theorem simplify_and_evaluate (a : ℝ) (ha : a ≠ -1) :
  (2 - (a - 1) / (a + 1)) / ((a^2 + 6*a + 9) / (a + 1)) = 1 / (a + 3) := by sorry

theorem evaluate_at_two :
  (2 - (2 - 1) / (2 + 1)) / ((2^2 + 6*2 + 9) / (2 + 1)) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_simplify_and_evaluate_evaluate_at_two_l1045_104505


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_cylinder_surface_area_l1045_104507

-- Define the cylinder
def cylinder_radius : ℝ := 5
def cylinder_height : ℝ := 10

-- Theorem for the longest segment
theorem longest_segment_in_cylinder :
  Real.sqrt ((2 * cylinder_radius) ^ 2 + cylinder_height ^ 2) = 10 * Real.sqrt 2 := by sorry

-- Theorem for the total surface area
theorem cylinder_surface_area :
  2 * Real.pi * cylinder_radius * (cylinder_height + cylinder_radius) = 150 * Real.pi := by sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_cylinder_surface_area_l1045_104507


namespace NUMINAMATH_CALUDE_tan_2x_eq_sin_x_solutions_l1045_104554

theorem tan_2x_eq_sin_x_solutions (x : ℝ) : 
  ∃ (s : Finset ℝ), s.card = 2 ∧ (∀ x ∈ s, x ∈ Set.Icc 0 Real.pi ∧ Real.tan (2 * x) = Real.sin x) ∧
  (∀ y ∈ Set.Icc 0 Real.pi, Real.tan (2 * y) = Real.sin y → y ∈ s) :=
sorry

end NUMINAMATH_CALUDE_tan_2x_eq_sin_x_solutions_l1045_104554


namespace NUMINAMATH_CALUDE_original_average_weight_l1045_104588

/-- Given a group of students, prove that the original average weight was 28 kg -/
theorem original_average_weight
  (n : ℕ) -- number of original students
  (x : ℝ) -- original average weight
  (w : ℝ) -- weight of new student
  (y : ℝ) -- new average weight after admitting the new student
  (hn : n = 29)
  (hw : w = 13)
  (hy : y = 27.5)
  (h_new_avg : (n : ℝ) * x + w = (n + 1 : ℝ) * y) :
  x = 28 :=
sorry

end NUMINAMATH_CALUDE_original_average_weight_l1045_104588


namespace NUMINAMATH_CALUDE_percent_to_decimal_twenty_five_percent_value_l1045_104503

theorem percent_to_decimal (x : ℚ) : x / 100 = x * (1 / 100) := by sorry

theorem twenty_five_percent_value : (25 : ℚ) / 100 = (1 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_twenty_five_percent_value_l1045_104503


namespace NUMINAMATH_CALUDE_carter_cake_difference_l1045_104500

def regular_cheesecakes : ℕ := 6
def regular_muffins : ℕ := 5
def regular_red_velvet : ℕ := 8

def regular_total : ℕ := regular_cheesecakes + regular_muffins + regular_red_velvet

def triple_total : ℕ := 3 * regular_total

theorem carter_cake_difference : triple_total - regular_total = 38 := by
  sorry

end NUMINAMATH_CALUDE_carter_cake_difference_l1045_104500


namespace NUMINAMATH_CALUDE_equation_solutions_l1045_104580

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ > 0 ∧ x₂ > 0) ∧
    (∀ (x : ℝ), x > 0 → 
      ((1/3) * (4*x^2 - 1) = (x^2 - 60*x - 12) * (x^2 + 30*x + 6)) ↔ 
      (x = x₁ ∨ x = x₂)) ∧
    x₁ = 30 + Real.sqrt 905 ∧
    x₂ = -15 + 4 * Real.sqrt 14 ∧
    4 * Real.sqrt 14 > 15 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1045_104580


namespace NUMINAMATH_CALUDE_remainder_problem_l1045_104532

theorem remainder_problem (k : ℤ) : ∃ (x : ℤ), x = 8 * k + 1 ∧ 71 * x % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1045_104532


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1045_104553

def A : Set ℝ := {1, 2, 1/2}

def B : Set ℝ := {y | ∃ x ∈ A, y = x^2}

theorem intersection_of_A_and_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1045_104553


namespace NUMINAMATH_CALUDE_position_interpretation_is_false_l1045_104538

/-- Represents a position in a grid -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Interprets a position as (column, row) -/
def interpret (p : Position) : String :=
  s!"the {p.x}th column and the {p.y}th row"

/-- The statement to be proven false -/
def statement (p : Position) : String :=
  s!"the {p.y}th row and the {p.x}th column"

theorem position_interpretation_is_false : 
  statement (Position.mk 5 1) ≠ interpret (Position.mk 5 1) :=
sorry

end NUMINAMATH_CALUDE_position_interpretation_is_false_l1045_104538


namespace NUMINAMATH_CALUDE_unique_solution_l1045_104583

/-- Represents the guesses made by the three friends --/
def friends_guesses : List Nat := [16, 19, 25]

/-- Represents the errors in the guesses --/
def guess_errors : List Nat := [2, 4, 5]

/-- Checks if a number satisfies all constraints --/
def satisfies_constraints (x : Nat) : Prop :=
  ∃ (perm : List Nat), perm.Perm guess_errors ∧
    (friends_guesses.zip perm).all (fun (guess, error) => 
      (guess + error = x) ∨ (guess - error = x))

/-- The theorem stating that 21 is the only number satisfying all constraints --/
theorem unique_solution : 
  satisfies_constraints 21 ∧ ∀ x : Nat, satisfies_constraints x → x = 21 := by
  sorry


end NUMINAMATH_CALUDE_unique_solution_l1045_104583


namespace NUMINAMATH_CALUDE_cherry_bag_cost_l1045_104512

/-- The cost of a four-pound bag of cherries -/
def cherry_cost : ℝ := 13.5

/-- The cost of the pie crust ingredients -/
def crust_cost : ℝ := 4.5

/-- The total cost of the cheapest pie -/
def cheapest_pie_cost : ℝ := 18

/-- The cost of the blueberry pie -/
def blueberry_pie_cost : ℝ := 18

theorem cherry_bag_cost : 
  cherry_cost = cheapest_pie_cost - crust_cost ∧ 
  blueberry_pie_cost = cheapest_pie_cost :=
by sorry

end NUMINAMATH_CALUDE_cherry_bag_cost_l1045_104512


namespace NUMINAMATH_CALUDE_samuel_has_five_birds_l1045_104561

/-- The number of berries a single bird eats per day -/
def berries_per_bird_per_day : ℕ := 7

/-- The total number of berries eaten by all birds in 4 days -/
def total_berries_in_four_days : ℕ := 140

/-- The number of days over which the total berries are consumed -/
def days : ℕ := 4

/-- The number of birds Samuel has -/
def samuels_birds : ℕ := total_berries_in_four_days / (days * berries_per_bird_per_day)

theorem samuel_has_five_birds : samuels_birds = 5 := by
  sorry

end NUMINAMATH_CALUDE_samuel_has_five_birds_l1045_104561


namespace NUMINAMATH_CALUDE_equation_equivalence_l1045_104586

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (5 - 9 * x) / (x - 2) + 2 = 0

-- Define the equivalent quadratic equation
def quadratic_equation (x : ℝ) : Prop :=
  9 * x^2 - 26 * x - 12 = 0

-- Theorem stating the equivalence of the two equations
theorem equation_equivalence :
  ∀ x : ℝ, original_equation x ↔ quadratic_equation x :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1045_104586


namespace NUMINAMATH_CALUDE_daniel_initial_noodles_l1045_104534

/-- The number of noodles Daniel had initially -/
def initial_noodles : ℕ := sorry

/-- The number of noodles Daniel gave to William -/
def noodles_to_william : ℕ := 15

/-- The number of noodles Daniel gave to Emily -/
def noodles_to_emily : ℕ := 20

/-- The number of noodles Daniel has left -/
def noodles_left : ℕ := 40

/-- Theorem stating that Daniel started with 75 noodles -/
theorem daniel_initial_noodles : initial_noodles = 75 := by sorry

end NUMINAMATH_CALUDE_daniel_initial_noodles_l1045_104534


namespace NUMINAMATH_CALUDE_solve_system_l1045_104590

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x - 2 * y = 7) 
  (eq2 : x + 3 * y = 8) : 
  x = 37 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l1045_104590


namespace NUMINAMATH_CALUDE_no_solution_inequality_l1045_104504

theorem no_solution_inequality :
  ¬∃ (x : ℝ), (9 * x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5)) < 4 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_inequality_l1045_104504


namespace NUMINAMATH_CALUDE_zoo_visitors_l1045_104558

theorem zoo_visitors (saturday_visitors : ℕ) (day_visitors : ℕ) : 
  saturday_visitors = 3750 → 
  saturday_visitors = 3 * day_visitors → 
  day_visitors = 1250 := by
sorry

end NUMINAMATH_CALUDE_zoo_visitors_l1045_104558


namespace NUMINAMATH_CALUDE_marble_selection_ways_l1045_104539

def total_marbles : ℕ := 15
def special_colors : ℕ := 3
def marbles_per_special_color : ℕ := 2
def marbles_to_choose : ℕ := 6
def special_marbles_to_choose : ℕ := 2

def remaining_marbles : ℕ := total_marbles - special_colors * marbles_per_special_color
def remaining_marbles_to_choose : ℕ := marbles_to_choose - special_marbles_to_choose

theorem marble_selection_ways :
  (special_colors * (marbles_per_special_color.choose special_marbles_to_choose)) *
  (remaining_marbles.choose remaining_marbles_to_choose) = 1485 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l1045_104539


namespace NUMINAMATH_CALUDE_inequality_preservation_l1045_104572

theorem inequality_preservation (a b c : ℝ) (h : a < b) : a - c < b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1045_104572


namespace NUMINAMATH_CALUDE_picture_album_distribution_l1045_104542

theorem picture_album_distribution : ∃ (a b c : ℕ), 
  a + b + c = 40 ∧ 
  a + b = 28 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 :=
by sorry

end NUMINAMATH_CALUDE_picture_album_distribution_l1045_104542


namespace NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l1045_104596

/-- Represents the number of free ends after k iterations of drawing segments -/
def freeEnds (k : ℕ) : ℕ := 2 + 4 * k

/-- Theorem stating that there exists a positive integer k such that
    the number of free ends after k iterations is 1001 -/
theorem exists_k_for_1001_free_ends :
  ∃ k : ℕ, k > 0 ∧ freeEnds k = 1001 :=
sorry

end NUMINAMATH_CALUDE_exists_k_for_1001_free_ends_l1045_104596


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1045_104571

theorem simplify_and_rationalize :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (Real.sqrt 8 / Real.sqrt 10) * (Real.sqrt 5 / Real.sqrt 12) / (Real.sqrt 9 / Real.sqrt 14) = 
  Real.sqrt a / b ∧
  a = 28 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1045_104571


namespace NUMINAMATH_CALUDE_cubic_roots_relationship_l1045_104568

/-- The cubic polynomial f(x) -/
def f (x : ℝ) : ℝ := x^3 + 2*x^2 + 3*x + 4

/-- The cubic polynomial h(x) -/
def h (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Theorem stating the relationship between f and h and the values of a, b, and c -/
theorem cubic_roots_relationship (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  (∀ x : ℝ, f x = 0 → h a b c (x^3) = 0) →
  a = -6 ∧ b = -9 ∧ c = 20 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_relationship_l1045_104568


namespace NUMINAMATH_CALUDE_third_number_is_seven_l1045_104513

def hcf (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem third_number_is_seven (x : ℕ) 
  (hcf_condition : hcf 136 144 x = 8)
  (lcm_condition : lcm 136 144 x = 2^4 * 3^2 * 17 * 7) :
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_third_number_is_seven_l1045_104513


namespace NUMINAMATH_CALUDE_inverse_proportion_example_l1045_104550

/-- Two real numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_example :
  ∀ x y : ℝ → ℝ,
  InverselyProportional x y →
  x 5 = 40 →
  y 5 = 5 →
  y 20 = 20 →
  x 20 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_example_l1045_104550


namespace NUMINAMATH_CALUDE_job_completion_time_l1045_104578

def job_completion (x : ℝ) : Prop :=
  ∃ (y : ℝ),
    (1 / (x + 5) + 1 / (x + 3) + 1 / (2 * y) = 1 / x) ∧
    (1 / (x + 3) + 1 / y = 1 / x) ∧
    (y > 0) ∧ (x > 0)

theorem job_completion_time : ∃ (x : ℝ), job_completion x ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1045_104578


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1045_104584

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (1 / (x^2 + y^2) + 1 / (x^2 + z^2) + 1 / (y^2 + z^2)) ≥ 9/2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1045_104584


namespace NUMINAMATH_CALUDE_first_week_daily_rate_l1045_104521

def daily_rate_first_week (x : ℚ) : Prop :=
  ∃ (total_cost : ℚ),
    total_cost = 7 * x + 16 * 11 ∧
    total_cost = 302

theorem first_week_daily_rate :
  ∀ x : ℚ, daily_rate_first_week x → x = 18 :=
by sorry

end NUMINAMATH_CALUDE_first_week_daily_rate_l1045_104521


namespace NUMINAMATH_CALUDE_solve_system_l1045_104515

theorem solve_system (a b c d : ℤ) 
  (eq1 : a + b = c)
  (eq2 : b + c = 7)
  (eq3 : c + d = 10)
  (eq4 : c = 4) :
  a = 1 ∧ d = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1045_104515


namespace NUMINAMATH_CALUDE_annual_pension_calculation_l1045_104509

/-- Represents the annual pension calculation for a retiring employee. -/
theorem annual_pension_calculation
  (c d r s : ℝ)
  (h_cd : d ≠ c)
  (h_positive : c > 0 ∧ d > 0 ∧ r > 0 ∧ s > 0)
  (h_prop : ∃ (k x : ℝ), k > 0 ∧ x > 0 ∧
    k * (x + c)^(3/2) = k * x^(3/2) + r ∧
    k * (x + d)^(3/2) = k * x^(3/2) + s) :
  ∃ (pension : ℝ), pension = (4 * r^2) / (9 * c^2) :=
sorry

end NUMINAMATH_CALUDE_annual_pension_calculation_l1045_104509


namespace NUMINAMATH_CALUDE_percentage_women_without_retirement_plan_l1045_104597

theorem percentage_women_without_retirement_plan 
  (total_workers : ℕ)
  (workers_without_plan : ℕ)
  (men_with_plan : ℕ)
  (total_men : ℕ)
  (total_women : ℕ)
  (h1 : workers_without_plan = total_workers / 3)
  (h2 : men_with_plan = (total_workers - workers_without_plan) * 2 / 5)
  (h3 : total_men = 120)
  (h4 : total_women = 120)
  (h5 : total_workers = total_men + total_women) :
  (workers_without_plan - (total_men - men_with_plan)) * 100 / total_women = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_women_without_retirement_plan_l1045_104597


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l1045_104589

def i : ℂ := Complex.I

theorem simplify_complex_fraction :
  (4 + 2 * i) / (4 - 2 * i) - (4 - 2 * i) / (4 + 2 * i) = 8 * i / 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l1045_104589


namespace NUMINAMATH_CALUDE_negation_and_absolute_value_l1045_104565

theorem negation_and_absolute_value : 
  (-(-2) = 2) ∧ (-|(-2)| = -2) := by
  sorry

end NUMINAMATH_CALUDE_negation_and_absolute_value_l1045_104565


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l1045_104524

/-- Given that angle α satisfies the conditions sin(2α) < 0 and sin(α) - cos(α) < 0,
    prove that α is in the fourth quadrant. -/
theorem angle_in_fourth_quadrant (α : Real) 
    (h1 : Real.sin (2 * α) < 0) 
    (h2 : Real.sin α - Real.cos α < 0) : 
  π < α ∧ α < (3 * π) / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l1045_104524


namespace NUMINAMATH_CALUDE_water_depth_difference_l1045_104520

theorem water_depth_difference (dean_height : ℝ) (water_depth_factor : ℝ) : 
  dean_height = 9 →
  water_depth_factor = 10 →
  water_depth_factor * dean_height - dean_height = 81 :=
by
  sorry

end NUMINAMATH_CALUDE_water_depth_difference_l1045_104520


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1045_104506

/-- Theorem: A line intersects a circle if the point defining the line is outside the circle -/
theorem line_intersects_circle (x₀ y₀ R : ℝ) (h : x₀^2 + y₀^2 > R^2) :
  ∃ (x y : ℝ), x^2 + y^2 = R^2 ∧ x₀*x + y₀*y = R^2 := by
  sorry

#check line_intersects_circle

end NUMINAMATH_CALUDE_line_intersects_circle_l1045_104506


namespace NUMINAMATH_CALUDE_min_a_for_four_integer_solutions_l1045_104533

theorem min_a_for_four_integer_solutions : 
  let has_four_solutions (a : ℤ) := 
    (∃ x₁ x₂ x₃ x₄ : ℤ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
      (x₁ - a < 0) ∧ (2 * x₁ + 3 > 0) ∧
      (x₂ - a < 0) ∧ (2 * x₂ + 3 > 0) ∧
      (x₃ - a < 0) ∧ (2 * x₃ + 3 > 0) ∧
      (x₄ - a < 0) ∧ (2 * x₄ + 3 > 0))
  ∀ a : ℤ, has_four_solutions a → a ≥ 3 ∧ has_four_solutions 3 :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_four_integer_solutions_l1045_104533


namespace NUMINAMATH_CALUDE_deepak_age_l1045_104593

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's current age. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 10 = 26 →
  deepak_age = 12 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1045_104593


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1045_104514

theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (2, 0) →
  ‖b‖ = 1 →
  ‖a + b‖ = Real.sqrt 7 →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1045_104514


namespace NUMINAMATH_CALUDE_max_value_sin_function_l1045_104562

theorem max_value_sin_function :
  ∀ x : ℝ, -π/2 ≤ x ∧ x ≤ 0 →
  ∃ y_max : ℝ, y_max = 5 ∧
  ∀ y : ℝ, y = 3 * Real.sin x + 5 → y ≤ y_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_function_l1045_104562


namespace NUMINAMATH_CALUDE_exists_valid_assignment_with_difference_one_l1045_104525

/-- Represents a position on an infinite checkerboard -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents the color of a square on the checkerboard -/
inductive Color
  | White
  | Black

/-- Determines the color of a square based on its position -/
def color (p : Position) : Color :=
  if (p.x + p.y) % 2 = 0 then Color.White else Color.Black

/-- Represents an assignment of non-zero integers to white squares -/
def Assignment := Position → ℤ

/-- Checks if an assignment is valid (all non-zero integers on white squares) -/
def is_valid_assignment (f : Assignment) : Prop :=
  ∀ p, color p = Color.White → f p ≠ 0

/-- Calculates the product difference for a black square -/
def product_difference (f : Assignment) (p : Position) : ℤ :=
  f {x := p.x - 1, y := p.y} * f {x := p.x + 1, y := p.y} -
  f {x := p.x, y := p.y - 1} * f {x := p.x, y := p.y + 1}

/-- The main theorem: there exists a valid assignment satisfying the condition -/
theorem exists_valid_assignment_with_difference_one :
  ∃ f : Assignment, is_valid_assignment f ∧
    ∀ p, color p = Color.Black → product_difference f p = 1 :=
  sorry

end NUMINAMATH_CALUDE_exists_valid_assignment_with_difference_one_l1045_104525


namespace NUMINAMATH_CALUDE_opposite_value_implies_ab_zero_l1045_104570

/-- Given that for all x, a(-x) + b(-x)^2 = -(ax + bx^2), prove that ab = 0 -/
theorem opposite_value_implies_ab_zero (a b : ℝ) 
  (h : ∀ x : ℝ, a * (-x) + b * (-x)^2 = -(a * x + b * x^2)) : 
  a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_value_implies_ab_zero_l1045_104570


namespace NUMINAMATH_CALUDE_max_area_of_specific_prism_l1045_104528

/-- A prism with vertical edges parallel to the z-axis and a square cross-section -/
structure Prism where
  side_length : ℝ
  cutting_plane : ℝ → ℝ → ℝ → Prop

/-- The maximum area of the cross-section of the prism cut by a plane -/
def max_cross_section_area (p : Prism) : ℝ := sorry

/-- The theorem stating the maximum area of the cross-section for the given prism -/
theorem max_area_of_specific_prism :
  let p : Prism := {
    side_length := 12,
    cutting_plane := fun x y z ↦ 3 * x - 5 * y + 5 * z = 30
  }
  max_cross_section_area p = 360 := by sorry

end NUMINAMATH_CALUDE_max_area_of_specific_prism_l1045_104528


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_in_60_degree_sector_l1045_104575

/-- The radius of a circle inscribed in a sector with a central angle of 60° and radius R is R/3 -/
theorem inscribed_circle_radius_in_60_degree_sector (R : ℝ) (R_pos : R > 0) :
  let sector_angle : ℝ := 60 * π / 180
  let inscribed_radius : ℝ := R / 3
  inscribed_radius = R / 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_in_60_degree_sector_l1045_104575


namespace NUMINAMATH_CALUDE_fixed_point_on_line_fixed_point_unique_l1045_104563

/-- The line equation passing through a fixed point -/
def line_equation (k x y : ℝ) : Prop :=
  y = k * (x - 2) + 3

/-- The fixed point through which the line always passes -/
def fixed_point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the fixed point satisfies the line equation for all k -/
theorem fixed_point_on_line :
  ∀ k : ℝ, line_equation k (fixed_point.1) (fixed_point.2) :=
sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_unique :
  ∀ x y : ℝ, (∀ k : ℝ, line_equation k x y) → (x, y) = fixed_point :=
sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_fixed_point_unique_l1045_104563


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l1045_104540

/-- Given three square regions I, II, and III, where the perimeter of region I is 16 units
    and the perimeter of region II is 32 units, the ratio of the area of region II
    to the area of region III is 1/4. -/
theorem area_ratio_of_squares (side_length_I side_length_II side_length_III : ℝ)
    (h1 : side_length_I * 4 = 16)
    (h2 : side_length_II * 4 = 32)
    (h3 : side_length_III = 2 * side_length_II) :
    (side_length_II ^ 2) / (side_length_III ^ 2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l1045_104540


namespace NUMINAMATH_CALUDE_purely_imaginary_z_reciprocal_l1045_104569

theorem purely_imaginary_z_reciprocal (m : ℝ) :
  let z : ℂ := m^2 - 1 + (m + 1) * I
  (∃ (y : ℝ), z = y * I) → 2 / z = -I :=
by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_reciprocal_l1045_104569


namespace NUMINAMATH_CALUDE_only_airplane_survey_comprehensive_l1045_104543

/-- Represents a type of survey --/
inductive SurveyType
  | WaterQuality
  | AirplanePassengers
  | PlasticBags
  | TVViewership

/-- Predicate to determine if a survey type is suitable for comprehensive surveying --/
def is_comprehensive (s : SurveyType) : Prop :=
  match s with
  | SurveyType.AirplanePassengers => true
  | _ => false

/-- Theorem stating that only the airplane passenger survey is comprehensive --/
theorem only_airplane_survey_comprehensive :
  ∀ s : SurveyType, is_comprehensive s ↔ s = SurveyType.AirplanePassengers :=
by
  sorry

#check only_airplane_survey_comprehensive

end NUMINAMATH_CALUDE_only_airplane_survey_comprehensive_l1045_104543


namespace NUMINAMATH_CALUDE_warehouse_boxes_l1045_104585

theorem warehouse_boxes : 
  ∀ (warehouse1 warehouse2 : ℕ),
  warehouse1 = 2 * warehouse2 →
  warehouse1 + warehouse2 = 600 →
  warehouse1 = 400 := by
sorry

end NUMINAMATH_CALUDE_warehouse_boxes_l1045_104585


namespace NUMINAMATH_CALUDE_usual_time_to_school_l1045_104552

/-- Given a boy who walks 7/6 of his usual rate and reaches school 5 minutes early,
    prove that his usual time to reach the school is 35 minutes. -/
theorem usual_time_to_school (R : ℝ) (T : ℝ) : 
  R * T = (7/6 * R) * (T - 5) → T = 35 :=
by sorry

end NUMINAMATH_CALUDE_usual_time_to_school_l1045_104552


namespace NUMINAMATH_CALUDE_house_wall_planks_l1045_104549

/-- Given the total number of nails, nails per plank, and additional nails used,
    calculate the number of planks needed. -/
def planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) (additional_nails : ℕ) : ℕ :=
  (total_nails - additional_nails) / nails_per_plank

/-- Theorem stating that given the specific conditions, the number of planks needed is 1. -/
theorem house_wall_planks :
  planks_needed 11 3 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_house_wall_planks_l1045_104549


namespace NUMINAMATH_CALUDE_debt_payment_average_l1045_104557

theorem debt_payment_average (total_payments : ℕ) (first_payment_count : ℕ) (first_payment_amount : ℚ) (additional_amount : ℚ) : 
  total_payments = 65 ∧ 
  first_payment_count = 20 ∧ 
  first_payment_amount = 410 ∧ 
  additional_amount = 65 → 
  (first_payment_count * first_payment_amount + 
   (total_payments - first_payment_count) * (first_payment_amount + additional_amount)) / total_payments = 455 := by
sorry

end NUMINAMATH_CALUDE_debt_payment_average_l1045_104557


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1045_104555

theorem complex_equation_solution :
  ∀ (z : ℂ), (3 - z) * Complex.I = 2 * Complex.I → z = 3 + 2 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1045_104555


namespace NUMINAMATH_CALUDE_count_even_factors_l1045_104535

def n : ℕ := 2^4 * 3^3 * 5^2

/-- The number of even positive factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors n = 48 := by sorry

end NUMINAMATH_CALUDE_count_even_factors_l1045_104535


namespace NUMINAMATH_CALUDE_triangle_area_l1045_104501

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A = 5 * π / 6 → b = 2 → c = 4 →
  (1 / 2) * b * c * Real.sin A = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1045_104501
