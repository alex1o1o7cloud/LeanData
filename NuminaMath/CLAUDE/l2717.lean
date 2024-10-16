import Mathlib

namespace NUMINAMATH_CALUDE_common_root_not_implies_equal_coefficients_l2717_271738

theorem common_root_not_implies_equal_coefficients
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (x : ℝ), (a * x^2 + b * x + c = 0 ∧ c * x^2 + b * x + a = 0) → ¬(a = c) :=
sorry

end NUMINAMATH_CALUDE_common_root_not_implies_equal_coefficients_l2717_271738


namespace NUMINAMATH_CALUDE_vector_angle_proof_l2717_271713

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_angle_proof (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 4) 
  (h3 : (a + b) • a = 0) : 
  angle_between_vectors a b = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_proof_l2717_271713


namespace NUMINAMATH_CALUDE_product_of_sines_equals_2_25_l2717_271726

theorem product_of_sines_equals_2_25 :
  (1 + Real.sin (π / 12)) * (1 + Real.sin (5 * π / 12)) *
  (1 + Real.sin (7 * π / 12)) * (1 + Real.sin (11 * π / 12)) = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_equals_2_25_l2717_271726


namespace NUMINAMATH_CALUDE_steps_climbed_proof_l2717_271734

/-- Calculates the total number of steps climbed given the number of steps and climbs for two ladders -/
def total_steps_climbed (full_ladder_steps : ℕ) (full_ladder_climbs : ℕ) 
                        (small_ladder_steps : ℕ) (small_ladder_climbs : ℕ) : ℕ :=
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs

/-- Proves that the total number of steps climbed is 152 given the specific ladder configurations -/
theorem steps_climbed_proof :
  total_steps_climbed 11 10 6 7 = 152 := by
  sorry

end NUMINAMATH_CALUDE_steps_climbed_proof_l2717_271734


namespace NUMINAMATH_CALUDE_tan_double_angle_l2717_271777

theorem tan_double_angle (x : Real) (h : Real.tan x = 3) : Real.tan (2 * x) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l2717_271777


namespace NUMINAMATH_CALUDE_factorization_proof_l2717_271765

theorem factorization_proof :
  (∀ x : ℝ, 4 * x^2 - 36 = 4 * (x + 3) * (x - 3)) ∧
  (∀ x y : ℝ, x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2) := by
sorry

end NUMINAMATH_CALUDE_factorization_proof_l2717_271765


namespace NUMINAMATH_CALUDE_lion_death_rate_l2717_271736

/-- Calculates the death rate of lions given initial population, birth rate, and final population after a year. -/
theorem lion_death_rate (initial_population : ℕ) (birth_rate : ℕ) (final_population : ℕ) : 
  initial_population = 100 →
  birth_rate = 5 →
  final_population = 148 →
  ∃ (death_rate : ℕ), 
    initial_population + 12 * birth_rate - 12 * death_rate = final_population ∧
    death_rate = 1 :=
by sorry

end NUMINAMATH_CALUDE_lion_death_rate_l2717_271736


namespace NUMINAMATH_CALUDE_next_perfect_square_l2717_271781

theorem next_perfect_square (original : Nat) (h : original = 1296) :
  ∃ n : Nat, n > 0 ∧ IsSquare (original + n) ∧
  ∀ m : Nat, 0 < m → m < n → ¬IsSquare (original + m) :=
by sorry

end NUMINAMATH_CALUDE_next_perfect_square_l2717_271781


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l2717_271795

theorem binomial_coefficient_equality (n : ℕ) : 
  Nat.choose 18 n = Nat.choose 18 2 → n = 2 ∨ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l2717_271795


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l2717_271763

theorem sqrt_sum_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt a + Real.sqrt (a + 5) < Real.sqrt (a + 2) + Real.sqrt (a + 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l2717_271763


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l2717_271798

def angle : ℝ := 2017

theorem point_in_third_quadrant :
  let x := Real.cos (angle * π / 180)
  let y := Real.sin (angle * π / 180)
  x < 0 ∧ y < 0 :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l2717_271798


namespace NUMINAMATH_CALUDE_imaginary_sum_equals_4_minus_4i_l2717_271767

theorem imaginary_sum_equals_4_minus_4i :
  let i : ℂ := Complex.I
  (i + 2 * i^2 + 3 * i^3 + 4 * i^4 + 5 * i^5 + 6 * i^6 + 7 * i^7 + 8 * i^8) = (4 : ℂ) - 4 * i :=
by sorry

end NUMINAMATH_CALUDE_imaginary_sum_equals_4_minus_4i_l2717_271767


namespace NUMINAMATH_CALUDE_william_journey_time_l2717_271790

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_minutes : minutes < 60

/-- Calculates the difference between two times in hours -/
def timeDifferenceInHours (t1 t2 : Time) : ℚ :=
  (t2.hours - t1.hours : ℚ) + (t2.minutes - t1.minutes : ℚ) / 60

/-- Represents a journey with stops and delays -/
structure Journey where
  departureTime : Time
  arrivalTime : Time
  timeZoneDifference : ℕ
  stops : List ℕ
  trafficDelay : ℕ

theorem william_journey_time (j : Journey) 
  (h1 : j.departureTime = ⟨7, 0, by norm_num⟩)
  (h2 : j.arrivalTime = ⟨20, 0, by norm_num⟩)
  (h3 : j.timeZoneDifference = 2)
  (h4 : j.stops = [25, 10, 25])
  (h5 : j.trafficDelay = 45) :
  timeDifferenceInHours j.departureTime ⟨18, 0, by norm_num⟩ + 
  (j.stops.sum / 60 : ℚ) + (j.trafficDelay / 60 : ℚ) = 12.75 := by
  sorry

#check william_journey_time

end NUMINAMATH_CALUDE_william_journey_time_l2717_271790


namespace NUMINAMATH_CALUDE_slower_truck_speed_calculation_l2717_271719

-- Define the length of each truck
def truck_length : ℝ := 250

-- Define the speed of the faster truck
def faster_truck_speed : ℝ := 30

-- Define the time taken for the slower truck to pass the faster one
def passing_time : ℝ := 35.997120230381576

-- Define the speed of the slower truck
def slower_truck_speed : ℝ := 20

-- Theorem statement
theorem slower_truck_speed_calculation :
  let total_length := 2 * truck_length
  let faster_speed_ms := faster_truck_speed * (1000 / 3600)
  let slower_speed_ms := slower_truck_speed * (1000 / 3600)
  let relative_speed := faster_speed_ms + slower_speed_ms
  total_length = relative_speed * passing_time :=
by sorry

end NUMINAMATH_CALUDE_slower_truck_speed_calculation_l2717_271719


namespace NUMINAMATH_CALUDE_matrix_inverse_problem_l2717_271758

def A : Matrix (Fin 2) (Fin 2) ℚ := !![5, -3; 2, 1]

theorem matrix_inverse_problem :
  (∃ (B : Matrix (Fin 2) (Fin 2) ℚ), A * B = 1 ∧ B * A = 1) →
  (A⁻¹ = !![1/11, 3/11; -2/11, 5/11]) ∨
  (¬∃ (B : Matrix (Fin 2) (Fin 2) ℚ), A * B = 1 ∧ B * A = 1) →
  (A⁻¹ = 0) :=
sorry

end NUMINAMATH_CALUDE_matrix_inverse_problem_l2717_271758


namespace NUMINAMATH_CALUDE_smallest_four_digit_sum_27_l2717_271754

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_sum_27 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 27 → 1899 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_sum_27_l2717_271754


namespace NUMINAMATH_CALUDE_smallest_number_of_cars_l2717_271773

theorem smallest_number_of_cars (N : ℕ) : 
  N > 2 ∧ 
  N % 5 = 2 ∧ 
  N % 6 = 2 ∧ 
  N % 7 = 2 → 
  (∀ m : ℕ, m > 2 ∧ m % 5 = 2 ∧ m % 6 = 2 ∧ m % 7 = 2 → m ≥ N) →
  N = 212 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_of_cars_l2717_271773


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l2717_271796

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

theorem largest_digit_divisible_by_6 : 
  ∀ N : ℕ, N ≤ 9 → 
    (is_divisible_by_6 (71820 + N) → N ≤ 6) ∧ 
    (is_divisible_by_6 (71826)) := by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l2717_271796


namespace NUMINAMATH_CALUDE_percent_of_number_l2717_271717

theorem percent_of_number (N M : ℝ) (h : M ≠ 0) : (N / M) * 100 = (100 * N) / M := by
  sorry

end NUMINAMATH_CALUDE_percent_of_number_l2717_271717


namespace NUMINAMATH_CALUDE_casey_calculation_l2717_271786

theorem casey_calculation (x : ℝ) : (x / 7) - 20 = 19 → (x * 7) + 20 = 1931 := by
  sorry

end NUMINAMATH_CALUDE_casey_calculation_l2717_271786


namespace NUMINAMATH_CALUDE_other_candidate_votes_l2717_271731

theorem other_candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ)
  (h_total : total_votes = 8500)
  (h_invalid : invalid_percent = 25 / 100)
  (h_winner : winner_percent = 60 / 100) :
  ⌊(1 - winner_percent) * ((1 - invalid_percent) * total_votes)⌋ = 2550 := by
  sorry

end NUMINAMATH_CALUDE_other_candidate_votes_l2717_271731


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l2717_271702

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (2, x) (3, 6) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l2717_271702


namespace NUMINAMATH_CALUDE_root_conditions_imply_inequalities_l2717_271716

theorem root_conditions_imply_inequalities (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b > 0) (hc : c ≠ 0)
  (h_distinct : ∃ x y : ℝ, x ≠ y ∧ 
    a * x^2 + b * x - c = 0 ∧ 
    a * y^2 + b * y - c = 0)
  (h_cubic : ∀ x : ℝ, a * x^2 + b * x - c = 0 → 
    x^3 + b * x^2 + a * x - c = 0) :
  a * b * c > 16 ∧ a * b * c ≥ 3125 / 108 := by
  sorry

end NUMINAMATH_CALUDE_root_conditions_imply_inequalities_l2717_271716


namespace NUMINAMATH_CALUDE_number_of_students_l2717_271709

theorem number_of_students (N : ℕ) : 
  (N : ℚ) * 15 = 4 * 14 + 10 * 16 + 9 → N = 15 := by
  sorry

#check number_of_students

end NUMINAMATH_CALUDE_number_of_students_l2717_271709


namespace NUMINAMATH_CALUDE_probability_one_white_one_black_l2717_271750

/-- The probability of drawing one white ball and one black ball from a box -/
theorem probability_one_white_one_black (w b : ℕ) (hw : w = 7) (hb : b = 8) :
  let total := w + b
  let favorable := w * b
  let total_combinations := (total * (total - 1)) / 2
  (favorable : ℚ) / total_combinations = 56 / 105 := by sorry

end NUMINAMATH_CALUDE_probability_one_white_one_black_l2717_271750


namespace NUMINAMATH_CALUDE_two_thousand_fourteenth_smallest_perimeter_l2717_271768

/-- A right triangle with integer side lengths forming an arithmetic sequence -/
structure ArithmeticRightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  a_lt_b : a < b
  b_lt_c : b < c
  is_arithmetic : b - a = c - b
  is_right : a^2 + b^2 = c^2

/-- The perimeter of an arithmetic right triangle -/
def perimeter (t : ArithmeticRightTriangle) : ℕ := t.a + t.b + t.c

/-- The theorem stating that the 2014th smallest perimeter of arithmetic right triangles is 24168 -/
theorem two_thousand_fourteenth_smallest_perimeter :
  (ArithmeticRightTriangle.mk 6042 8056 10070 (by sorry) (by sorry) (by sorry) (by sorry) |>
    perimeter) = 24168 := by sorry

end NUMINAMATH_CALUDE_two_thousand_fourteenth_smallest_perimeter_l2717_271768


namespace NUMINAMATH_CALUDE_political_test_analysis_l2717_271721

def class_A_scores : List ℝ := [41, 47, 43, 45, 50, 49, 48, 50, 50, 49, 48, 47, 44, 50, 43, 50, 50, 50, 49, 47]

structure FrequencyDistribution :=
  (range1 : ℕ)
  (range2 : ℕ)
  (range3 : ℕ)
  (range4 : ℕ)
  (range5 : ℕ)

def class_B_dist : FrequencyDistribution :=
  { range1 := 1
  , range2 := 1
  , range3 := 3  -- This is 'a' in the original problem
  , range4 := 6
  , range5 := 9 }

def class_B_46_to_48 : List ℝ := [47, 48, 48, 47, 48, 48]

structure ClassStats :=
  (average : ℝ)
  (median : ℝ)
  (mode : ℝ)

def class_A_stats : ClassStats :=
  { average := 47.5
  , median := 48.5
  , mode := 50 }  -- This is 'c' in the original problem

def class_B_stats : ClassStats :=
  { average := 47.5
  , median := 48  -- This is 'b' in the original problem
  , mode := 49 }

def total_students : ℕ := 800

theorem political_test_analysis :
  (class_B_dist.range3 = 3) ∧
  (class_B_stats.median = 48) ∧
  (class_A_stats.mode = 50) ∧
  (((List.filter (λ x => x ≥ 49) class_A_scores).length +
    class_B_dist.range5) / 40 * total_students = 380) := by
  sorry


end NUMINAMATH_CALUDE_political_test_analysis_l2717_271721


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2717_271739

def A : Set ℤ := {1, 3, 5}
def B : Set ℤ := {-1, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2717_271739


namespace NUMINAMATH_CALUDE_coin_flip_probability_difference_l2717_271772

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def binomial_probability (n k : ℕ) : ℚ :=
  (n.choose k) * (1 / 2) ^ n

/-- The theorem stating the difference between probabilities of 4 and 3 heads in 5 flips -/
theorem coin_flip_probability_difference :
  |binomial_probability 5 4 - binomial_probability 5 3| = 5 / 32 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_difference_l2717_271772


namespace NUMINAMATH_CALUDE_function_upper_bound_l2717_271789

theorem function_upper_bound
  (f : ℝ → ℝ)
  (h1 : ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2))
  (h2 : ∃ (M : ℝ), M > 0 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |f x| ≤ M)
  : ∀ (x : ℝ), x ≥ 0 → f x ≤ x^2 :=
by sorry

end NUMINAMATH_CALUDE_function_upper_bound_l2717_271789


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l2717_271797

theorem corrected_mean_calculation (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 36 →
  incorrect_value = 23 →
  correct_value = 48 →
  let original_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := original_sum + difference
  corrected_sum / n = 36.5 := by
sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l2717_271797


namespace NUMINAMATH_CALUDE_quadratic_solution_l2717_271764

theorem quadratic_solution (p q : ℝ) :
  let x : ℝ → ℝ := λ y => y - p / 2
  ∀ y, x y * x y + p * x y + q = 0 ↔ y * y = p * p / 4 - q :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2717_271764


namespace NUMINAMATH_CALUDE_seating_theorem_l2717_271760

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to seat athletes from three teams in a row, with teammates seated together -/
def seating_arrangements (team_a : ℕ) (team_b : ℕ) (team_c : ℕ) : ℕ :=
  factorial 3 * factorial team_a * factorial team_b * factorial team_c

theorem seating_theorem :
  seating_arrangements 4 3 3 = 5184 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l2717_271760


namespace NUMINAMATH_CALUDE_tax_reduction_theorem_l2717_271730

/-- Proves that a tax reduction of 15% results in a 6.5% revenue decrease
    when consumption increases by 10% -/
theorem tax_reduction_theorem (T C : ℝ) (X : ℝ) 
  (h_positive_T : T > 0) 
  (h_positive_C : C > 0) 
  (h_consumption_increase : 1.1 * C = C + 0.1 * C) 
  (h_revenue_decrease : (T * (1 - X / 100) * (C * 1.1)) = T * C * 0.935) :
  X = 15 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_theorem_l2717_271730


namespace NUMINAMATH_CALUDE_hotel_charge_difference_l2717_271776

theorem hotel_charge_difference (P_s R_s G_s P_d R_d G_d P_su R_su G_su : ℝ) 
  (h1 : P_s = R_s * 0.45)
  (h2 : P_s = G_s * 0.90)
  (h3 : P_d = R_d * 0.70)
  (h4 : P_d = G_d * 0.80)
  (h5 : P_su = R_su * 0.60)
  (h6 : P_su = G_su * 0.85) :
  (R_s / G_s - 1) * 100 - (R_d / G_d - 1) * 100 = 85.7143 := by
sorry

end NUMINAMATH_CALUDE_hotel_charge_difference_l2717_271776


namespace NUMINAMATH_CALUDE_matthews_friends_l2717_271707

theorem matthews_friends (initial_crackers initial_cakes cakes_per_person : ℕ) 
  (h1 : initial_crackers = 10)
  (h2 : initial_cakes = 8)
  (h3 : cakes_per_person = 2)
  (h4 : initial_cakes % cakes_per_person = 0) :
  initial_cakes / cakes_per_person = 4 := by
  sorry

end NUMINAMATH_CALUDE_matthews_friends_l2717_271707


namespace NUMINAMATH_CALUDE_quadratic_integer_values_l2717_271747

theorem quadratic_integer_values (a b c : ℝ) :
  (∀ x : ℤ, ∃ n : ℤ, a * x^2 + b * x + c = n) ↔
  (∃ m : ℤ, 2 * a = m) ∧ (∃ n : ℤ, a + b = n) ∧ (∃ p : ℤ, c = p) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_values_l2717_271747


namespace NUMINAMATH_CALUDE_water_wave_area_increase_rate_l2717_271720

/-- The rate of increase of the area of a circular water wave -/
theorem water_wave_area_increase_rate 
  (v : ℝ) -- velocity of radius expansion
  (r : ℝ) -- current radius
  (h1 : v = 50) -- given velocity
  (h2 : r = 250) -- given radius
  : (π * v * r * 2) = 25000 * π := by
  sorry

end NUMINAMATH_CALUDE_water_wave_area_increase_rate_l2717_271720


namespace NUMINAMATH_CALUDE_problem_i4_1_l2717_271770

theorem problem_i4_1 (f : ℝ → ℝ) :
  (∀ x, f x = (x^2 + x - 2)^2002 + 3) →
  f ((Real.sqrt 5 / 2) - 1/2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_i4_1_l2717_271770


namespace NUMINAMATH_CALUDE_line_inclination_l2717_271729

/-- The inclination angle of a line given by parametric equations -/
def inclination_angle (x_eq : ℝ → ℝ) (y_eq : ℝ → ℝ) : ℝ :=
  sorry

theorem line_inclination :
  let x_eq := λ t : ℝ => 3 + t * Real.sin (25 * π / 180)
  let y_eq := λ t : ℝ => -t * Real.cos (25 * π / 180)
  inclination_angle x_eq y_eq = 115 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_line_inclination_l2717_271729


namespace NUMINAMATH_CALUDE_difference_median_mode_l2717_271748

def data : List ℕ := [30, 31, 32, 33, 33, 33, 40, 41, 42, 43, 44, 45, 51, 51, 51, 52, 53, 55, 60, 61, 62, 64, 65, 67, 71, 72, 73, 74, 75, 76]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem difference_median_mode :
  |median data - (mode data : ℚ)| = 19.5 := by sorry

end NUMINAMATH_CALUDE_difference_median_mode_l2717_271748


namespace NUMINAMATH_CALUDE_number_of_workers_l2717_271749

theorem number_of_workers (total_contribution : ℕ) (extra_contribution : ℕ) (new_total : ℕ) : 
  total_contribution = 300000 →
  extra_contribution = 50 →
  new_total = 320000 →
  ∃ (workers : ℕ), 
    workers * (total_contribution / workers + extra_contribution) = new_total ∧
    workers = 400 := by
  sorry

end NUMINAMATH_CALUDE_number_of_workers_l2717_271749


namespace NUMINAMATH_CALUDE_nonlinear_system_solutions_l2717_271784

theorem nonlinear_system_solutions :
  let f₁ (x y z : ℝ) := x + 4*y + 6*z - 16
  let f₂ (x y z : ℝ) := x + 6*y + 12*z - 24
  let f₃ (x y z : ℝ) := x^2 + 4*y^2 + 36*z^2 - 76
  ∀ (x y z : ℝ),
    f₁ x y z = 0 ∧ f₂ x y z = 0 ∧ f₃ x y z = 0 ↔
    (x = 6 ∧ y = 1 ∧ z = 1) ∨ (x = -2/3 ∧ y = 13/3 ∧ z = -1/9) :=
by
  sorry

#check nonlinear_system_solutions

end NUMINAMATH_CALUDE_nonlinear_system_solutions_l2717_271784


namespace NUMINAMATH_CALUDE_percentage_difference_l2717_271708

theorem percentage_difference : (80 / 100 * 60) - (4 / 5 * 25) = 28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2717_271708


namespace NUMINAMATH_CALUDE_ratio_is_five_l2717_271700

/-- The equation holds for all real x except -3, 0, and 6 -/
def equation_holds (P Q : ℤ) : Prop :=
  ∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 6 →
    (P : ℝ) / (x + 3) + (Q : ℝ) / (x^2 - 6*x) = (x^2 - 4*x + 15) / (x^3 + x^2 - 18*x)

theorem ratio_is_five (P Q : ℤ) (h : equation_holds P Q) : (Q : ℚ) / P = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_is_five_l2717_271700


namespace NUMINAMATH_CALUDE_paint_cube_cost_l2717_271755

/-- The cost to paint a cube given paint price, coverage, and cube dimensions -/
theorem paint_cube_cost (paint_price : ℝ) (paint_coverage : ℝ) (cube_side : ℝ) : 
  paint_price = 36.5 →
  paint_coverage = 16 →
  cube_side = 8 →
  6 * cube_side^2 / paint_coverage * paint_price = 876 := by
sorry

end NUMINAMATH_CALUDE_paint_cube_cost_l2717_271755


namespace NUMINAMATH_CALUDE_max_area_of_divided_rectangle_l2717_271711

/-- Given a large rectangle divided into 8 smaller rectangles with specific perimeters,
    prove that its maximum area is 512 square centimeters. -/
theorem max_area_of_divided_rectangle :
  ∀ (pA pB pC pD pE : ℝ) (area : ℝ → ℝ),
  pA = 26 →
  pB = 28 →
  pC = 30 →
  pD = 32 →
  pE = 34 →
  (∀ x, area x ≤ 512) →
  (∃ x, area x = 512) :=
by sorry

end NUMINAMATH_CALUDE_max_area_of_divided_rectangle_l2717_271711


namespace NUMINAMATH_CALUDE_sale_result_l2717_271791

/-- Represents the total number of cases of cat food sold during a sale. -/
def total_cases_sold (first_group : Nat) (second_group : Nat) (third_group : Nat) 
  (first_group_cases : Nat) (second_group_cases : Nat) (third_group_cases : Nat) : Nat :=
  first_group * first_group_cases + second_group * second_group_cases + third_group * third_group_cases

/-- Theorem stating that the total number of cases sold is 40 given the specific customer purchase patterns. -/
theorem sale_result : 
  total_cases_sold 8 4 8 3 2 1 = 40 := by
  sorry

#check sale_result

end NUMINAMATH_CALUDE_sale_result_l2717_271791


namespace NUMINAMATH_CALUDE_ratio_equality_l2717_271775

theorem ratio_equality : ∃ x : ℝ, (12 : ℝ) / 8 = x / 240 ∧ x = 360 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2717_271775


namespace NUMINAMATH_CALUDE_star_arrangements_l2717_271725

/-- The number of points on a regular six-pointed star -/
def num_points : ℕ := 12

/-- The number of rotational symmetries of a regular six-pointed star -/
def num_rotations : ℕ := 6

/-- The number of reflectional symmetries of a regular six-pointed star -/
def num_reflections : ℕ := 2

/-- The total number of symmetries of a regular six-pointed star -/
def total_symmetries : ℕ := num_rotations * num_reflections

/-- The number of distinct arrangements of objects on a regular six-pointed star -/
def distinct_arrangements : ℕ := Nat.factorial num_points / total_symmetries

theorem star_arrangements :
  distinct_arrangements = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_star_arrangements_l2717_271725


namespace NUMINAMATH_CALUDE_mitchell_gum_packets_l2717_271780

theorem mitchell_gum_packets (pieces_per_packet : ℕ) (pieces_chewed : ℕ) (pieces_left : ℕ) : 
  pieces_per_packet = 7 →
  pieces_left = 2 →
  pieces_chewed = 54 →
  (pieces_chewed + pieces_left) / pieces_per_packet = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_mitchell_gum_packets_l2717_271780


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2717_271718

theorem min_value_quadratic (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^2 - a * x + (a + 1000) ≥ 1 + 999 / a) ∧
  (∃ x : ℝ, a * x^2 - a * x + (a + 1000) = 1 + 999 / a) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2717_271718


namespace NUMINAMATH_CALUDE_kids_staying_home_l2717_271701

theorem kids_staying_home (total_kids : ℕ) (kids_at_camp : ℕ) 
  (h1 : total_kids = 898051)
  (h2 : kids_at_camp = 629424) :
  total_kids - kids_at_camp = 268627 := by
  sorry

end NUMINAMATH_CALUDE_kids_staying_home_l2717_271701


namespace NUMINAMATH_CALUDE_equation_transformation_l2717_271782

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 5*x^2 + x + 1 = 0 ↔ x^2*(y^2 + y - 7) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l2717_271782


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2717_271759

theorem reciprocal_of_negative_2023 :
  ∃ (x : ℚ), x * (-2023) = 1 ∧ x = -1/2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2717_271759


namespace NUMINAMATH_CALUDE_identical_solutions_condition_l2717_271723

theorem identical_solutions_condition (k : ℝ) : 
  (∃! x y : ℝ, y = x^2 ∧ y = 4*x + k) ↔ k = -4 := by
sorry

end NUMINAMATH_CALUDE_identical_solutions_condition_l2717_271723


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2717_271778

theorem product_of_three_numbers (x y z m : ℚ) : 
  x + y + z = 240 ∧ 
  9 * x = m ∧ 
  y - 11 = m ∧ 
  z + 11 = m ∧ 
  x < y ∧ 
  x < z → 
  x * y * z = 7514700 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2717_271778


namespace NUMINAMATH_CALUDE_car_speed_change_l2717_271714

theorem car_speed_change (V : ℝ) (x : ℝ) (h_V : V > 0) (h_x : x > 0) : 
  V * (1 - x / 100) * (1 + 0.5 * x / 100) = V * (1 - 0.6 * x / 100) → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_change_l2717_271714


namespace NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_divided_by_11_l2717_271740

theorem units_digit_of_sum_of_powers_divided_by_11 : 
  (3^2018 + 7^2018) % 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_of_powers_divided_by_11_l2717_271740


namespace NUMINAMATH_CALUDE_pizza_pieces_l2717_271766

theorem pizza_pieces (total_pizzas : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) : 
  total_pizzas = 4 → total_cost = 80 → cost_per_piece = 4 → 
  (total_cost / total_pizzas) / cost_per_piece = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_pieces_l2717_271766


namespace NUMINAMATH_CALUDE_parallel_vectors_k_l2717_271762

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k (k : ℝ) :
  let a : ℝ × ℝ := (2*k + 2, 4)
  let b : ℝ × ℝ := (k + 1, 8)
  parallel a b → k = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_l2717_271762


namespace NUMINAMATH_CALUDE_train_speed_l2717_271745

/-- The speed of a train given its length and time to pass a stationary point -/
theorem train_speed (length time : ℝ) (h1 : length = 160) (h2 : time = 8) :
  length / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2717_271745


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2717_271743

theorem smaller_number_problem (x y : ℝ) : 
  y = 3 * x + 11 → x + y = 55 → x = 11 := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2717_271743


namespace NUMINAMATH_CALUDE_cube_edge_increase_l2717_271727

theorem cube_edge_increase (surface_area_increase : Real) 
  (h : surface_area_increase = 69.00000000000001) : 
  ∃ edge_increase : Real, 
    edge_increase = 30 ∧ 
    (1 + edge_increase / 100)^2 = 1 + surface_area_increase / 100 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_increase_l2717_271727


namespace NUMINAMATH_CALUDE_flour_sugar_difference_l2717_271742

theorem flour_sugar_difference (recipe_sugar : ℕ) (recipe_flour : ℕ) (recipe_salt : ℕ) (flour_added : ℕ) :
  recipe_sugar = 9 →
  recipe_flour = 14 →
  recipe_salt = 40 →
  flour_added = 4 →
  recipe_flour - flour_added - recipe_sugar = 1 := by
sorry

end NUMINAMATH_CALUDE_flour_sugar_difference_l2717_271742


namespace NUMINAMATH_CALUDE_min_sum_of_product_1020_l2717_271715

theorem min_sum_of_product_1020 (a b c : ℕ+) (h : a * b * c = 1020) :
  ∃ (x y z : ℕ+), x * y * z = 1020 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 33 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1020_l2717_271715


namespace NUMINAMATH_CALUDE_min_pizzas_to_break_even_l2717_271724

def car_cost : ℕ := 6500
def net_profit_per_pizza : ℕ := 7

theorem min_pizzas_to_break_even :
  ∀ n : ℕ, (n * net_profit_per_pizza ≥ car_cost) ∧ 
           (∀ m : ℕ, m < n → m * net_profit_per_pizza < car_cost) →
  n = 929 := by
  sorry

end NUMINAMATH_CALUDE_min_pizzas_to_break_even_l2717_271724


namespace NUMINAMATH_CALUDE_problem_statement_l2717_271774

theorem problem_statement (X Y Z : ℕ+) 
  (h_coprime : Nat.gcd X.val (Nat.gcd Y.val Z.val) = 1)
  (h_equation : X.val * Real.log 3 / Real.log 100 + Y.val * Real.log 4 / Real.log 100 = (Z.val : ℝ)^2) :
  X.val + Y.val + Z.val = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2717_271774


namespace NUMINAMATH_CALUDE_division_problem_l2717_271779

theorem division_problem : (107.8 : ℝ) / 11 = 9.8 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2717_271779


namespace NUMINAMATH_CALUDE_triangle_area_change_l2717_271751

theorem triangle_area_change (b h : ℝ) (h_pos : 0 < h) (b_pos : 0 < b) :
  let new_height := 0.9 * h
  let new_base := 1.2 * b
  let original_area := (b * h) / 2
  let new_area := (new_base * new_height) / 2
  new_area = 1.08 * original_area := by
sorry

end NUMINAMATH_CALUDE_triangle_area_change_l2717_271751


namespace NUMINAMATH_CALUDE_min_third_side_right_triangle_l2717_271769

theorem min_third_side_right_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  (a = 7 ∨ b = 7 ∨ c = 7) → 
  (a = 24 ∨ b = 24 ∨ c = 24) → 
  a^2 + b^2 = c^2 → 
  min a (min b c) ≥ Real.sqrt 527 :=
by sorry

end NUMINAMATH_CALUDE_min_third_side_right_triangle_l2717_271769


namespace NUMINAMATH_CALUDE_milk_production_l2717_271793

/-- Given x cows with efficiency α producing y gallons in z days,
    calculate the milk production of w cows with efficiency β in v days -/
theorem milk_production
  (x y z w v : ℝ) (α β : ℝ) (hx : x > 0) (hz : z > 0) (hα : α > 0) :
  let production := (β * y * w * v) / (α^2 * x * z)
  production = (β * y * w * v) / (α^2 * x * z) := by
  sorry

end NUMINAMATH_CALUDE_milk_production_l2717_271793


namespace NUMINAMATH_CALUDE_speed_doubling_l2717_271792

theorem speed_doubling (distance : ℝ) (original_time : ℝ) (new_time : ℝ) 
  (h1 : distance = 440)
  (h2 : original_time = 3)
  (h3 : new_time = original_time / 2)
  : (distance / new_time) = 2 * (distance / original_time) := by
  sorry

#check speed_doubling

end NUMINAMATH_CALUDE_speed_doubling_l2717_271792


namespace NUMINAMATH_CALUDE_box_height_l2717_271746

/-- Proves that a rectangular box with given dimensions has a height of 3 cm -/
theorem box_height (base_length base_width volume : ℝ) 
  (h1 : base_length = 2)
  (h2 : base_width = 5)
  (h3 : volume = 30) :
  volume / (base_length * base_width) = 3 := by
  sorry

end NUMINAMATH_CALUDE_box_height_l2717_271746


namespace NUMINAMATH_CALUDE_smallest_cube_with_four_8s_l2717_271744

/-- A function that returns the first k digits of a natural number n -/
def firstKDigits (n : ℕ) (k : ℕ) : ℕ := sorry

/-- A function that checks if the first k digits of n are all 8 -/
def startsWithK8s (n : ℕ) (k : ℕ) : Prop := 
  firstKDigits n k = (8 : ℕ) * (10^k - 1) / 9

theorem smallest_cube_with_four_8s :
  (∀ m : ℕ, m < 9615 → ¬ startsWithK8s (m^3) 4) ∧ startsWithK8s (9615^3) 4 := by sorry

end NUMINAMATH_CALUDE_smallest_cube_with_four_8s_l2717_271744


namespace NUMINAMATH_CALUDE_license_plate_combinations_l2717_271735

def consonants : ℕ := 20
def vowels : ℕ := 6
def digits : ℕ := 10

theorem license_plate_combinations : 
  consonants^2 * vowels^2 * digits = 144000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l2717_271735


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l2717_271705

/-- Represents a parabola in the form y = (x - h)² + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Shifts a point horizontally -/
def shift_horizontal (p : Point) (shift : ℝ) : Point :=
  { x := p.x + shift, y := p.y }

/-- Shifts a point vertically -/
def shift_vertical (p : Point) (shift : ℝ) : Point :=
  { x := p.x, y := p.y + shift }

/-- The vertex of a parabola -/
def vertex (p : Parabola) : Point :=
  { x := p.h, y := p.k }

theorem parabola_shift_theorem (p : Parabola) :
  p.h = 2 ∧ p.k = 3 →
  (shift_vertical (shift_horizontal (vertex p) (-3)) (-5)) = { x := -1, y := -2 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l2717_271705


namespace NUMINAMATH_CALUDE_product_of_n_values_product_of_possible_n_values_l2717_271761

-- Define the temperatures at noon
def temp_minneapolis (n : ℝ) (l : ℝ) : ℝ := l + n
def temp_stlouis (l : ℝ) : ℝ := l

-- Define the temperatures at 4:00 PM
def temp_minneapolis_4pm (n : ℝ) (l : ℝ) : ℝ := temp_minneapolis n l - 7
def temp_stlouis_4pm (l : ℝ) : ℝ := temp_stlouis l + 5

-- Define the temperature difference at 4:00 PM
def temp_diff_4pm (n : ℝ) (l : ℝ) : ℝ := |temp_minneapolis_4pm n l - temp_stlouis_4pm l|

-- Theorem statement
theorem product_of_n_values (n : ℝ) (l : ℝ) :
  (temp_diff_4pm n l = 4) → (n = 16 ∨ n = 8) ∧ (16 * 8 = 128) := by
  sorry

-- Main theorem
theorem product_of_possible_n_values : 
  ∃ (n₁ n₂ : ℝ), (n₁ ≠ n₂) ∧ (∀ l : ℝ, temp_diff_4pm n₁ l = 4 ∧ temp_diff_4pm n₂ l = 4) ∧ (n₁ * n₂ = 128) := by
  sorry

end NUMINAMATH_CALUDE_product_of_n_values_product_of_possible_n_values_l2717_271761


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2717_271704

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ k * x₁^2 - 2 * x₁ + 3 = 0 ∧ k * x₂^2 - 2 * x₂ + 3 = 0) ↔ 
  (k ≤ 1/3 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2717_271704


namespace NUMINAMATH_CALUDE_choir_average_age_l2717_271706

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 12)
  (h2 : num_males = 13)
  (h3 : avg_age_females = 32)
  (h4 : avg_age_males = 33)
  (h5 : num_females + num_males = 25) :
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  let total_members := num_females + num_males
  total_age / total_members = 32.52 := by
sorry

end NUMINAMATH_CALUDE_choir_average_age_l2717_271706


namespace NUMINAMATH_CALUDE_student_scores_l2717_271741

theorem student_scores (M P C : ℕ) : 
  M + P = 50 →
  (M + C) / 2 = 35 →
  C > P →
  C - P = 20 := by
sorry

end NUMINAMATH_CALUDE_student_scores_l2717_271741


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2717_271785

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), a.1 * b.2 = t * a.2 * b.1

/-- The given vectors -/
def a : ℝ × ℝ := (6, 2)
def b (k : ℝ) : ℝ × ℝ := (-3, k)

/-- The theorem stating that if the given vectors are parallel, then k = -1 -/
theorem parallel_vectors_k_value :
  parallel a (b k) → k = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2717_271785


namespace NUMINAMATH_CALUDE_unique_polygon_pair_l2717_271752

/-- The interior angle of a regular polygon with n sides --/
def interior_angle (n : ℕ) : ℚ :=
  180 - 360 / n

/-- The condition for the ratio of interior angles to be 5:3 --/
def angle_ratio_condition (a b : ℕ) : Prop :=
  interior_angle a / interior_angle b = 5 / 3

/-- The main theorem --/
theorem unique_polygon_pair :
  ∃! (pair : ℕ × ℕ), 
    pair.1 > 2 ∧ 
    pair.2 > 2 ∧ 
    angle_ratio_condition pair.1 pair.2 :=
sorry

end NUMINAMATH_CALUDE_unique_polygon_pair_l2717_271752


namespace NUMINAMATH_CALUDE_renata_final_balance_l2717_271788

/-- Represents Renata's financial transactions throughout the day -/
def renata_transactions : ℤ → ℤ
| 0 => 10                  -- Initial amount
| 1 => -4                  -- Charity ticket donation
| 2 => 90                  -- Charity draw winnings
| 3 => -50                 -- First slot machine loss
| 4 => -10                 -- Second slot machine loss
| 5 => -5                  -- Third slot machine loss
| 6 => -1                  -- Water bottle purchase
| 7 => -1                  -- Lottery ticket purchase
| 8 => 65                  -- Lottery winnings
| _ => 0                   -- No more transactions

/-- The final balance after all transactions -/
def final_balance : ℤ := (List.range 9).foldl (· + renata_transactions ·) 0

/-- Theorem stating that Renata's final balance is $94 -/
theorem renata_final_balance : final_balance = 94 := by
  sorry

end NUMINAMATH_CALUDE_renata_final_balance_l2717_271788


namespace NUMINAMATH_CALUDE_star_five_three_l2717_271757

def star (a b : ℤ) : ℤ := a^2 + a*b - b^2

theorem star_five_three : star 5 3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l2717_271757


namespace NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_l2717_271722

/-- Represents the number of blocks Ray walks in different segments of his route -/
structure RayWalk where
  to_park : Nat
  to_school : Nat
  to_home : Nat

/-- Represents Ray's daily dog walking routine -/
structure DailyWalk where
  route : RayWalk
  walks_per_day : Nat

/-- Calculates the total number of blocks Ray's dog walks in a day -/
def total_blocks_walked (daily : DailyWalk) : Nat :=
  (daily.route.to_park + daily.route.to_school + daily.route.to_home) * daily.walks_per_day

/-- Theorem stating that Ray's dog walks 66 blocks each day -/
theorem rays_dog_walks_66_blocks (daily : DailyWalk) 
  (h1 : daily.route.to_park = 4)
  (h2 : daily.route.to_school = 7)
  (h3 : daily.route.to_home = 11)
  (h4 : daily.walks_per_day = 3) : 
  total_blocks_walked daily = 66 := by
  sorry

end NUMINAMATH_CALUDE_rays_dog_walks_66_blocks_l2717_271722


namespace NUMINAMATH_CALUDE_travel_time_ratio_l2717_271783

/-- Proves the ratio of travel times for a given distance at different speeds -/
theorem travel_time_ratio 
  (distance : ℝ) 
  (original_time : ℝ) 
  (new_speed : ℝ) 
  (h1 : distance = 144)
  (h2 : original_time = 6)
  (h3 : new_speed = 16) : 
  (distance / new_speed) / original_time = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_ratio_l2717_271783


namespace NUMINAMATH_CALUDE_extra_page_number_l2717_271703

theorem extra_page_number (n : ℕ) (k : ℕ) : 
  n = 62 → 
  (n * (n + 1)) / 2 + k = 1986 → 
  k = 33 := by
sorry

end NUMINAMATH_CALUDE_extra_page_number_l2717_271703


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2717_271753

def quadratic_function (x k : ℝ) : ℝ := -2 * x^2 + 4 * x + k

theorem quadratic_inequality (k : ℝ) :
  let x1 : ℝ := -0.99
  let x2 : ℝ := 0.98
  let x3 : ℝ := 0.99
  let y1 : ℝ := quadratic_function x1 k
  let y2 : ℝ := quadratic_function x2 k
  let y3 : ℝ := quadratic_function x3 k
  y1 < y2 ∧ y2 < y3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2717_271753


namespace NUMINAMATH_CALUDE_initial_short_trees_count_l2717_271737

/-- The number of short trees in the park after planting -/
def final_short_trees : ℕ := 95

/-- The number of short trees planted today -/
def planted_short_trees : ℕ := 64

/-- The initial number of short trees in the park -/
def initial_short_trees : ℕ := final_short_trees - planted_short_trees

theorem initial_short_trees_count : initial_short_trees = 31 := by
  sorry

end NUMINAMATH_CALUDE_initial_short_trees_count_l2717_271737


namespace NUMINAMATH_CALUDE_smallest_of_three_consecutive_odds_l2717_271787

theorem smallest_of_three_consecutive_odds (a b c : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  b = a + 2 →  -- b is the next consecutive odd number
  c = b + 2 →  -- c is the next consecutive odd number after b
  a + b + c = 69 →  -- their sum is 69
  a = 21 :=  -- the smallest (a) is 21
by sorry

end NUMINAMATH_CALUDE_smallest_of_three_consecutive_odds_l2717_271787


namespace NUMINAMATH_CALUDE_det_equality_l2717_271799

theorem det_equality (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 7 →
  Matrix.det !![x - 2*z, y - 2*w; z, w] = 7 := by
  sorry

end NUMINAMATH_CALUDE_det_equality_l2717_271799


namespace NUMINAMATH_CALUDE_fraction_transformation_l2717_271710

theorem fraction_transformation (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) :
  (a^3 : ℚ) / (b + 3) = 2 * (a / b) → a = 2 ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l2717_271710


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2717_271794

theorem smallest_positive_integer_with_remainders : ∃ b : ℕ, 
  b > 0 ∧ 
  b % 4 = 3 ∧ 
  b % 6 = 5 ∧ 
  (∀ c : ℕ, c > 0 ∧ c % 4 = 3 ∧ c % 6 = 5 → b ≤ c) ∧
  b = 11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2717_271794


namespace NUMINAMATH_CALUDE_calculate_salary_e_l2717_271756

/-- Calculates the salary of person E given the salaries of A, B, C, D, and the average salary of all five people. -/
theorem calculate_salary_e (salary_a salary_b salary_c salary_d avg_salary : ℕ) :
  salary_a = 8000 →
  salary_b = 5000 →
  salary_c = 11000 →
  salary_d = 7000 →
  avg_salary = 8000 →
  (salary_a + salary_b + salary_c + salary_d + (avg_salary * 5 - (salary_a + salary_b + salary_c + salary_d))) / 5 = avg_salary →
  avg_salary * 5 - (salary_a + salary_b + salary_c + salary_d) = 9000 := by
sorry

end NUMINAMATH_CALUDE_calculate_salary_e_l2717_271756


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l2717_271732

theorem min_value_of_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x + 9 / y) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l2717_271732


namespace NUMINAMATH_CALUDE_inequality_solution_l2717_271728

theorem inequality_solution (x : ℝ) :
  (2 / (x^2 + 1) > 4 / x + 5 / 2) ↔ -2 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2717_271728


namespace NUMINAMATH_CALUDE_total_players_in_ground_l2717_271771

theorem total_players_in_ground (cricket_players hockey_players football_players softball_players : ℕ) 
  (h1 : cricket_players = 22)
  (h2 : hockey_players = 15)
  (h3 : football_players = 21)
  (h4 : softball_players = 19) :
  cricket_players + hockey_players + football_players + softball_players = 77 := by
  sorry

end NUMINAMATH_CALUDE_total_players_in_ground_l2717_271771


namespace NUMINAMATH_CALUDE_rayden_lily_duck_ratio_l2717_271733

/-- Proves the ratio of Rayden's ducks to Lily's ducks is 3:1 -/
theorem rayden_lily_duck_ratio :
  let lily_ducks : ℕ := 20
  let lily_geese : ℕ := 10
  let rayden_geese : ℕ := 4 * lily_geese
  let total_difference : ℕ := 70
  let rayden_total : ℕ := lily_ducks + lily_geese + total_difference
  let rayden_ducks : ℕ := rayden_total - rayden_geese
  (rayden_ducks : ℚ) / lily_ducks = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_rayden_lily_duck_ratio_l2717_271733


namespace NUMINAMATH_CALUDE_farm_tax_calculation_l2717_271712

/-- The farm tax calculation problem -/
theorem farm_tax_calculation 
  (tax_percentage : Real) 
  (total_tax_collected : Real) 
  (willam_land_percentage : Real) : 
  tax_percentage = 0.4 →
  total_tax_collected = 3840 →
  willam_land_percentage = 0.3125 →
  willam_land_percentage * (total_tax_collected / tax_percentage) = 3000 := by
  sorry

#check farm_tax_calculation

end NUMINAMATH_CALUDE_farm_tax_calculation_l2717_271712
