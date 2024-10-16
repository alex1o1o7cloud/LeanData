import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_unique_minimum_l1340_134008

theorem min_value_theorem (x : ℝ) (h : x > 0) : x^2 + 10*x + 100/x^3 ≥ 40 := by
  sorry

theorem equality_condition : ∃ x > 0, x^2 + 10*x + 100/x^3 = 40 := by
  sorry

theorem unique_minimum (x : ℝ) (h1 : x > 0) (h2 : x^2 + 10*x + 100/x^3 = 40) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_unique_minimum_l1340_134008


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l1340_134089

theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 2310 → 
  ∀ (a b c : ℕ+), a * b * c = 2310 → l + w + h ≤ a + b + c → 
  l + w + h = 42 :=
sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l1340_134089


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1340_134006

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 + 1 = (X^2 - 4*X + 7) * q + (8*X - 62) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1340_134006


namespace NUMINAMATH_CALUDE_common_point_theorem_l1340_134003

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Constructs a line based on the given conditions -/
def construct_line (a d r : ℝ) : Line :=
  { a := a
  , b := a + d
  , c := a * r + 2 * d }

theorem common_point_theorem (a d r : ℝ) :
  (construct_line a d r).contains (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_common_point_theorem_l1340_134003


namespace NUMINAMATH_CALUDE_second_boy_speed_l1340_134054

/-- Given two boys walking in the same direction for 7 hours, with the first boy
    walking at 4 kmph and ending up 10.5 km apart, prove that the speed of the
    second boy is 5.5 kmph. -/
theorem second_boy_speed (v : ℝ) 
  (h1 : (v - 4) * 7 = 10.5) : v = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_second_boy_speed_l1340_134054


namespace NUMINAMATH_CALUDE_base_5_to_octal_conversion_l1340_134024

def base_5_to_decimal (n : ℕ) : ℕ := n

def decimal_to_octal (n : ℕ) : ℕ := n

theorem base_5_to_octal_conversion :
  decimal_to_octal (base_5_to_decimal 1234) = 302 := by
  sorry

end NUMINAMATH_CALUDE_base_5_to_octal_conversion_l1340_134024


namespace NUMINAMATH_CALUDE_diagonal_passes_810_cubes_l1340_134014

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem: The number of unit cubes an internal diagonal passes through
    in a 160 × 330 × 380 rectangular solid is 810 -/
theorem diagonal_passes_810_cubes :
  cubes_passed_by_diagonal 160 330 380 = 810 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_810_cubes_l1340_134014


namespace NUMINAMATH_CALUDE_runner_ends_at_start_l1340_134039

/-- A runner on a circular track -/
structure Runner where
  start_position : ℝ  -- Position on the track (0 ≤ position < track_length)
  distance_run : ℝ    -- Total distance run
  track_length : ℝ    -- Length of the circular track

/-- Theorem: A runner who completes an integer number of laps ends at the starting position -/
theorem runner_ends_at_start (runner : Runner) (h : runner.track_length > 0) :
  runner.distance_run % runner.track_length = 0 →
  (runner.start_position + runner.distance_run) % runner.track_length = runner.start_position :=
by sorry

end NUMINAMATH_CALUDE_runner_ends_at_start_l1340_134039


namespace NUMINAMATH_CALUDE_jeff_calculation_correction_l1340_134080

theorem jeff_calculation_correction (incorrect_input : ℕ × ℕ) (incorrect_result : ℕ) 
  (h1 : incorrect_input.1 = 52) 
  (h2 : incorrect_input.2 = 735) 
  (h3 : incorrect_input.1 * incorrect_input.2 = incorrect_result) 
  (h4 : incorrect_result = 38220) : 
  (0.52 : ℝ) * 7.35 = 3.822 := by
sorry

end NUMINAMATH_CALUDE_jeff_calculation_correction_l1340_134080


namespace NUMINAMATH_CALUDE_digit_equation_solution_l1340_134045

/-- Represents a four-digit number ABBD --/
def ABBD (A B D : Nat) : Nat := A * 1000 + B * 100 + B * 10 + D

/-- Represents a four-digit number BCAC --/
def BCAC (B C A : Nat) : Nat := B * 1000 + C * 100 + A * 10 + C

/-- Represents a five-digit number DDBBD --/
def DDBBD (D B : Nat) : Nat := D * 10000 + D * 1000 + B * 100 + B * 10 + D

theorem digit_equation_solution 
  (A B C D : Nat) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10) 
  (h_equation : ABBD A B D + BCAC B C A = DDBBD D B) : 
  D = 0 := by
  sorry

end NUMINAMATH_CALUDE_digit_equation_solution_l1340_134045


namespace NUMINAMATH_CALUDE_balls_per_bag_l1340_134042

theorem balls_per_bag (total_balls : ℕ) (num_bags : ℕ) (balls_per_bag : ℕ) 
  (h1 : total_balls = 36)
  (h2 : num_bags = 9)
  (h3 : total_balls = num_bags * balls_per_bag) :
  balls_per_bag = 4 := by
sorry

end NUMINAMATH_CALUDE_balls_per_bag_l1340_134042


namespace NUMINAMATH_CALUDE_quadratic_prime_square_l1340_134050

/-- A function that represents the given quadratic expression -/
def f (n : ℕ) : ℤ := 2 * n^2 - 5 * n - 33

/-- Predicate to check if a number is prime -/
def isPrime (p : ℕ) : Prop := Nat.Prime p

/-- The main theorem stating that 6 and 14 are the only natural numbers
    for which f(n) is the square of a prime number -/
theorem quadratic_prime_square : 
  ∀ n : ℕ, (∃ p : ℕ, isPrime p ∧ f n = p^2) ↔ n = 6 ∨ n = 14 :=
sorry

end NUMINAMATH_CALUDE_quadratic_prime_square_l1340_134050


namespace NUMINAMATH_CALUDE_inverse_A_times_B_l1340_134065

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ := !![0, 1; 2, 3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![2, 0; 1, 8]

theorem inverse_A_times_B :
  A⁻¹ * B = !![-(5/2), 4; 2, 0] := by sorry

end NUMINAMATH_CALUDE_inverse_A_times_B_l1340_134065


namespace NUMINAMATH_CALUDE_inverse_tangent_sum_l1340_134099

theorem inverse_tangent_sum : Real.arctan (1/2) + Real.arctan (1/5) + Real.arctan (1/8) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_tangent_sum_l1340_134099


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l1340_134044

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l1340_134044


namespace NUMINAMATH_CALUDE_marions_score_l1340_134048

theorem marions_score (total_items : ℕ) (ellas_incorrect : ℕ) (marions_additional : ℕ) : 
  total_items = 40 →
  ellas_incorrect = 4 →
  marions_additional = 6 →
  (total_items - ellas_incorrect) / 2 + marions_additional = 24 :=
by sorry

end NUMINAMATH_CALUDE_marions_score_l1340_134048


namespace NUMINAMATH_CALUDE_tim_dan_balloon_ratio_l1340_134066

theorem tim_dan_balloon_ratio :
  let dan_balloons : ℕ := 29
  let tim_balloons : ℕ := 203
  (tim_balloons / dan_balloons : ℚ) = 7 := by sorry

end NUMINAMATH_CALUDE_tim_dan_balloon_ratio_l1340_134066


namespace NUMINAMATH_CALUDE_total_raisins_added_l1340_134034

theorem total_raisins_added (yellow_raisins : ℝ) (black_raisins : ℝ)
  (h1 : yellow_raisins = 0.3)
  (h2 : black_raisins = 0.4) :
  yellow_raisins + black_raisins = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_total_raisins_added_l1340_134034


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l1340_134093

theorem solution_implies_m_value (x m : ℝ) : 
  x = 2 → 4 * x + 2 * m - 14 = 0 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l1340_134093


namespace NUMINAMATH_CALUDE_mixture_problem_l1340_134030

/-- Represents the initial ratio of liquid A to liquid B -/
def initial_ratio : ℚ := 7 / 5

/-- Represents the amount of mixture drawn off in liters -/
def drawn_off : ℚ := 9

/-- Represents the new ratio of liquid A to liquid B after refilling -/
def new_ratio : ℚ := 7 / 9

/-- Represents the initial amount of liquid A in the can -/
def initial_amount_A : ℚ := 21

theorem mixture_problem :
  ∃ (total : ℚ),
    total > 0 ∧
    initial_amount_A / (total - initial_amount_A) = initial_ratio ∧
    (initial_amount_A - (initial_amount_A / total) * drawn_off) /
    (total - initial_amount_A - ((total - initial_amount_A) / total) * drawn_off + drawn_off) = new_ratio :=
by sorry

end NUMINAMATH_CALUDE_mixture_problem_l1340_134030


namespace NUMINAMATH_CALUDE_area_of_similar_rectangle_l1340_134084

/-- Given a rectangle R1 with one side of 3 inches and an area of 18 square inches,
    and a similar rectangle R2 with a diagonal of 18 inches,
    prove that the area of R2 is 14.4 square inches. -/
theorem area_of_similar_rectangle (r1_side : ℝ) (r1_area : ℝ) (r2_diagonal : ℝ) :
  r1_side = 3 →
  r1_area = 18 →
  r2_diagonal = 18 →
  ∃ (r2_side1 r2_side2 : ℝ),
    r2_side1 * r2_side2 = 14.4 ∧
    r2_side1^2 + r2_side2^2 = r2_diagonal^2 ∧
    r2_side2 / r2_side1 = r1_area / r1_side^2 :=
by sorry

end NUMINAMATH_CALUDE_area_of_similar_rectangle_l1340_134084


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_120_l1340_134094

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) :
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_120_l1340_134094


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1340_134096

theorem min_value_reciprocal_sum (m n : ℝ) : 
  m > 0 → n > 0 → m * 1 + n * 1 = 2 → (1 / m + 1 / n) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1340_134096


namespace NUMINAMATH_CALUDE_cubic_minus_three_divisibility_l1340_134001

theorem cubic_minus_three_divisibility (n : ℕ) (h : n > 1) :
  (n - 1) ∣ (n^3 - 3) ↔ n = 2 ∨ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_three_divisibility_l1340_134001


namespace NUMINAMATH_CALUDE_work_completion_time_l1340_134052

/-- Given that A can do a work in 9 days and A and B together can do the work in 6 days,
    prove that B can do the work alone in 18 days. -/
theorem work_completion_time (a_time b_time ab_time : ℝ) 
    (ha : a_time = 9)
    (hab : ab_time = 6)
    (h_work_rate : 1 / a_time + 1 / b_time = 1 / ab_time) : 
  b_time = 18 := by
sorry


end NUMINAMATH_CALUDE_work_completion_time_l1340_134052


namespace NUMINAMATH_CALUDE_cone_volume_increase_l1340_134049

/-- The volume of a cone increases by 612.8% when its height is increased by 120% and its radius is increased by 80% -/
theorem cone_volume_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  let v := (1/3) * Real.pi * r^2 * h
  let r_new := 1.8 * r
  let h_new := 2.2 * h
  let v_new := (1/3) * Real.pi * r_new^2 * h_new
  (v_new - v) / v * 100 = 612.8 := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_increase_l1340_134049


namespace NUMINAMATH_CALUDE_q_div_p_equals_225_l1340_134015

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The probability of drawing 5 cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The probability of drawing 4 cards of one number and 1 card of a different number -/
def q : ℚ := (distinct_numbers * (distinct_numbers - 1) * Nat.choose cards_per_number 4 * Nat.choose cards_per_number 1 : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The theorem stating that q/p = 225 -/
theorem q_div_p_equals_225 : q / p = 225 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_225_l1340_134015


namespace NUMINAMATH_CALUDE_sin_cos_sum_27_63_l1340_134029

theorem sin_cos_sum_27_63 : 
  Real.sin (27 * π / 180) * Real.cos (63 * π / 180) + 
  Real.cos (27 * π / 180) * Real.sin (63 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_27_63_l1340_134029


namespace NUMINAMATH_CALUDE_simplify_sqrt_500_l1340_134068

theorem simplify_sqrt_500 : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_500_l1340_134068


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l1340_134013

theorem factor_difference_of_squares (t : ℝ) : 4 * t^2 - 81 = (2*t - 9) * (2*t + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l1340_134013


namespace NUMINAMATH_CALUDE_parallelogram_height_l1340_134071

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area base height : ℝ) : 
  area = base * height → area = 960 → base = 60 → height = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1340_134071


namespace NUMINAMATH_CALUDE_order_mnpq_l1340_134092

theorem order_mnpq (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n :=
by sorry

end NUMINAMATH_CALUDE_order_mnpq_l1340_134092


namespace NUMINAMATH_CALUDE_shop_annual_rent_per_square_foot_l1340_134095

/-- Calculates the annual rent per square foot of a shop -/
theorem shop_annual_rent_per_square_foot
  (length : ℝ)
  (width : ℝ)
  (monthly_rent : ℝ)
  (h1 : length = 10)
  (h2 : width = 8)
  (h3 : monthly_rent = 2400) :
  (monthly_rent * 12) / (length * width) = 360 := by
  sorry

end NUMINAMATH_CALUDE_shop_annual_rent_per_square_foot_l1340_134095


namespace NUMINAMATH_CALUDE_unique_line_theorem_l1340_134019

/-- The parabola y = x^2 + 4x + 4 -/
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 4

/-- The line y = 8x - 9 -/
def line (x : ℝ) : ℝ := 8*x - 9

/-- The distance between two points on a vertical line -/
def vertical_distance (y₁ y₂ : ℝ) : ℝ := |y₁ - y₂|

theorem unique_line_theorem :
  (∃! k : ℝ, vertical_distance (parabola k) (line k) = 6) ∧
  line 2 = 7 :=
sorry

end NUMINAMATH_CALUDE_unique_line_theorem_l1340_134019


namespace NUMINAMATH_CALUDE_simplify_expression_l1340_134018

theorem simplify_expression : (8 * 10^12) / (4 * 10^4) = 200000000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1340_134018


namespace NUMINAMATH_CALUDE_fermat_like_equation_power_l1340_134060

theorem fermat_like_equation_power (x y p n k : ℕ) : 
  x^n + y^n = p^k →
  n > 1 →
  Odd n →
  Nat.Prime p →
  Odd p →
  ∃ l : ℕ, n = p^l :=
by sorry

end NUMINAMATH_CALUDE_fermat_like_equation_power_l1340_134060


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l1340_134081

-- Define the distance function
def s (t : ℝ) : ℝ := 3 * t^2 + t

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 6 * t + 1

-- Theorem statement
theorem instantaneous_velocity_at_2 : v 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l1340_134081


namespace NUMINAMATH_CALUDE_banana_distribution_l1340_134007

theorem banana_distribution (total_bananas : ℕ) : 
  (∀ (children : ℕ), 
    (children * 2 = total_bananas) →
    ((children - 160) * 4 = total_bananas)) →
  ∃ (actual_children : ℕ), actual_children = 320 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_l1340_134007


namespace NUMINAMATH_CALUDE_staircase_step_difference_l1340_134057

/-- Theorem: Difference in steps between second and third staircases --/
theorem staircase_step_difference :
  ∀ (steps1 steps2 steps3 : ℕ) (step_height : ℚ) (total_height : ℚ),
    steps1 = 20 →
    steps2 = 2 * steps1 →
    step_height = 1/2 →
    total_height = 45 →
    (steps1 + steps2 + steps3 : ℚ) * step_height = total_height →
    steps2 - steps3 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_staircase_step_difference_l1340_134057


namespace NUMINAMATH_CALUDE_lcm_24_90_l1340_134040

theorem lcm_24_90 : Nat.lcm 24 90 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_90_l1340_134040


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1340_134035

theorem algebraic_expression_value : 
  let x : ℝ := -1
  3 * x^2 + 2 * x - 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1340_134035


namespace NUMINAMATH_CALUDE_addison_raffle_tickets_l1340_134053

/-- The number of raffle tickets Addison sold on Friday -/
def friday_tickets : ℕ := 181

/-- The number of raffle tickets Addison sold on Saturday -/
def saturday_tickets : ℕ := 2 * friday_tickets

/-- The number of raffle tickets Addison sold on Sunday -/
def sunday_tickets : ℕ := 78

theorem addison_raffle_tickets :
  friday_tickets = 181 ∧
  saturday_tickets = 2 * friday_tickets ∧
  sunday_tickets = 78 ∧
  saturday_tickets = sunday_tickets + 284 :=
by sorry

end NUMINAMATH_CALUDE_addison_raffle_tickets_l1340_134053


namespace NUMINAMATH_CALUDE_correct_num_ways_to_choose_l1340_134046

/-- The number of humanities courses -/
def num_humanities : ℕ := 4

/-- The number of natural science courses -/
def num_sciences : ℕ := 3

/-- The total number of courses to be chosen -/
def courses_to_choose : ℕ := 3

/-- The number of conflicting course pairs (A₁ and B₁) -/
def num_conflicts : ℕ := 1

/-- The function that calculates the number of ways to choose courses -/
def num_ways_to_choose : ℕ := sorry

theorem correct_num_ways_to_choose :
  num_ways_to_choose = 25 := by sorry

end NUMINAMATH_CALUDE_correct_num_ways_to_choose_l1340_134046


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1340_134091

theorem unique_solution_condition (k : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 3) = k + 3 * x) ↔ k = 35 / 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1340_134091


namespace NUMINAMATH_CALUDE_final_black_fraction_is_512_729_l1340_134038

/-- Represents the fraction of black area remaining after one change -/
def remaining_black_fraction : ℚ := 8 / 9

/-- Represents the number of changes applied to the triangle -/
def num_changes : ℕ := 3

/-- Represents the fraction of the original area that remains black after the specified number of changes -/
def final_black_fraction : ℚ := remaining_black_fraction ^ num_changes

/-- Theorem stating that the final black fraction is equal to 512/729 -/
theorem final_black_fraction_is_512_729 : 
  final_black_fraction = 512 / 729 := by sorry

end NUMINAMATH_CALUDE_final_black_fraction_is_512_729_l1340_134038


namespace NUMINAMATH_CALUDE_largest_absolute_value_l1340_134000

theorem largest_absolute_value : 
  let numbers : List ℤ := [4, -5, 0, -1]
  ∀ x ∈ numbers, |x| ≤ |-5| :=
by sorry

end NUMINAMATH_CALUDE_largest_absolute_value_l1340_134000


namespace NUMINAMATH_CALUDE_cosine_sine_identity_l1340_134041

theorem cosine_sine_identity : 
  Real.cos (80 * π / 180) * Real.cos (35 * π / 180) + 
  Real.sin (80 * π / 180) * Real.cos (55 * π / 180) = 
  (1 / 2) * (Real.sin (65 * π / 180) + Real.sin (25 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_identity_l1340_134041


namespace NUMINAMATH_CALUDE_factorization_equality_l1340_134078

theorem factorization_equality (x : ℝ) : 
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 = 
  (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + 2) * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1340_134078


namespace NUMINAMATH_CALUDE_highest_power_of_three_in_N_l1340_134055

def N : ℕ := sorry

-- Define the property that N is formed by writing down two-digit integers from 19 to 92 continuously
def is_valid_N (n : ℕ) : Prop := sorry

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem highest_power_of_three_in_N :
  is_valid_N N →
  ∃ m : ℕ, (sum_of_digits N = 3^2 * m) ∧ (m % 3 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_highest_power_of_three_in_N_l1340_134055


namespace NUMINAMATH_CALUDE_unique_solution_l1340_134073

/-- The function F as defined in the problem -/
def F (t : ℝ) : ℝ := 32 * t^5 + 48 * t^3 + 17 * t - 15

/-- The system of equations -/
def system_equations (x y z : ℝ) : Prop :=
  1/x = 32/y^5 + 48/y^3 + 17/y - 15 ∧
  1/y = 32/z^5 + 48/z^3 + 17/z - 15 ∧
  1/z = 32/x^5 + 48/x^3 + 17/x - 15

/-- The theorem stating the unique solution -/
theorem unique_solution :
  ∃! (x y z : ℝ), system_equations x y z ∧ x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1340_134073


namespace NUMINAMATH_CALUDE_solution_a_is_correct_l1340_134036

/-- The amount of Solution A used in milliliters -/
def solution_a : ℝ := 100

/-- The amount of Solution B used in milliliters -/
def solution_b : ℝ := solution_a + 500

/-- The alcohol percentage in Solution A -/
def alcohol_percent_a : ℝ := 0.16

/-- The alcohol percentage in Solution B -/
def alcohol_percent_b : ℝ := 0.10

/-- The total amount of pure alcohol in the resulting mixture in milliliters -/
def total_pure_alcohol : ℝ := 76

theorem solution_a_is_correct :
  solution_a * alcohol_percent_a + solution_b * alcohol_percent_b = total_pure_alcohol :=
sorry

end NUMINAMATH_CALUDE_solution_a_is_correct_l1340_134036


namespace NUMINAMATH_CALUDE_binary_channel_properties_l1340_134088

/-- A binary channel with error probabilities α and β -/
structure BinaryChannel where
  α : ℝ
  β : ℝ
  α_pos : 0 < α
  α_lt_one : α < 1
  β_pos : 0 < β
  β_lt_one : β < 1

/-- Probability of receiving 1,0,1 when sending 1,0,1 in single transmission -/
def prob_single_101 (bc : BinaryChannel) : ℝ := (1 - bc.α) * (1 - bc.β)^2

/-- Probability of receiving 1,0,1 when sending 1 in triple transmission -/
def prob_triple_101 (bc : BinaryChannel) : ℝ := bc.β * (1 - bc.β)^2

/-- Probability of decoding 1 when sending 1 in triple transmission -/
def prob_triple_decode_1 (bc : BinaryChannel) : ℝ := 3 * bc.β * (1 - bc.β)^2 + (1 - bc.β)^3

/-- Probability of decoding 0 when sending 0 in single transmission -/
def prob_single_decode_0 (bc : BinaryChannel) : ℝ := 1 - bc.α

/-- Probability of decoding 0 when sending 0 in triple transmission -/
def prob_triple_decode_0 (bc : BinaryChannel) : ℝ := 3 * bc.α * (1 - bc.α)^2 + (1 - bc.α)^3

theorem binary_channel_properties (bc : BinaryChannel) :
  prob_single_101 bc = (1 - bc.α) * (1 - bc.β)^2 ∧
  prob_triple_101 bc = bc.β * (1 - bc.β)^2 ∧
  prob_triple_decode_1 bc = 3 * bc.β * (1 - bc.β)^2 + (1 - bc.β)^3 ∧
  (bc.α < 0.5 → prob_triple_decode_0 bc > prob_single_decode_0 bc) :=
by sorry

end NUMINAMATH_CALUDE_binary_channel_properties_l1340_134088


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l1340_134062

def f (x : ℝ) := x^3 + x

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l1340_134062


namespace NUMINAMATH_CALUDE_jeong_hyeok_is_nine_l1340_134020

/-- Jeong-hyeok's age -/
def jeong_hyeok_age : ℕ := sorry

/-- Jeong-hyeok's uncle's age -/
def uncle_age : ℕ := sorry

/-- Condition 1: Jeong-hyeok's age is 1 year less than 1/4 of his uncle's age -/
axiom condition1 : jeong_hyeok_age = uncle_age / 4 - 1

/-- Condition 2: His uncle's age is 5 years less than 5 times Jeong-hyeok's age -/
axiom condition2 : uncle_age = 5 * jeong_hyeok_age - 5

/-- Theorem: Jeong-hyeok is 9 years old -/
theorem jeong_hyeok_is_nine : jeong_hyeok_age = 9 := by sorry

end NUMINAMATH_CALUDE_jeong_hyeok_is_nine_l1340_134020


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1340_134072

theorem inequality_solution_set : 
  {x : ℝ | x^2 - 7*x + 12 < 0} = Set.Ioo 3 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1340_134072


namespace NUMINAMATH_CALUDE_circle_center_correct_l1340_134079

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (2, -3)

/-- Theorem: The center of the circle defined by circle_equation is circle_center -/
theorem circle_center_correct :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l1340_134079


namespace NUMINAMATH_CALUDE_vector_relation_l1340_134059

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem vector_relation (h : B - A = 2 • (D - C)) :
  B - D = A - C - (3/2 : ℝ) • (B - A) := by
  sorry

end NUMINAMATH_CALUDE_vector_relation_l1340_134059


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l1340_134090

theorem decimal_to_fraction_sum (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 0.3421 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 0.3421 → a ≤ c ∧ b ≤ d →
  a + b = 13421 :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l1340_134090


namespace NUMINAMATH_CALUDE_initial_apples_count_l1340_134069

/-- The number of apples in a package -/
def apples_per_package : ℕ := 11

/-- The number of apples added to the pile -/
def apples_added : ℕ := 5

/-- The final number of apples in the pile -/
def final_apples : ℕ := 13

/-- The initial number of apples in the pile -/
def initial_apples : ℕ := final_apples - apples_added

theorem initial_apples_count : initial_apples = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_apples_count_l1340_134069


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1340_134010

/-- The line (m-1)x + (2m-1)y = m-5 passes through the point (9, -4) for any real m -/
theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1340_134010


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1340_134067

/-- Given an arithmetic sequence {a_n} with first term a_1 = 1 and common difference d = 3,
    if a_n = 2005, then n = 669. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (n : ℕ) : 
  a 1 = 1 →                                 -- First term is 1
  (∀ k, a (k + 1) - a k = 3) →              -- Common difference is 3
  a n = 2005 →                              -- nth term is 2005
  n = 669 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1340_134067


namespace NUMINAMATH_CALUDE_marbles_selection_count_l1340_134070

def total_marbles : ℕ := 15
def special_marbles : ℕ := 4
def marbles_to_choose : ℕ := 5
def special_marbles_to_choose : ℕ := 2

theorem marbles_selection_count :
  (Nat.choose special_marbles special_marbles_to_choose) *
  (Nat.choose (total_marbles - special_marbles) (marbles_to_choose - special_marbles_to_choose)) =
  990 := by sorry

end NUMINAMATH_CALUDE_marbles_selection_count_l1340_134070


namespace NUMINAMATH_CALUDE_fraction_equals_five_l1340_134074

theorem fraction_equals_five (a b k : ℕ+) : 
  (a.val^2 + b.val^2) / (a.val * b.val - 1) = k.val → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_five_l1340_134074


namespace NUMINAMATH_CALUDE_certain_number_divided_by_ten_l1340_134047

theorem certain_number_divided_by_ten (x : ℝ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_divided_by_ten_l1340_134047


namespace NUMINAMATH_CALUDE_water_balloon_problem_l1340_134012

/-- The number of water balloons that popped on the ground --/
def popped_balloons (max_rate max_time zach_rate zach_time total_filled : ℕ) : ℕ :=
  max_rate * max_time + zach_rate * zach_time - total_filled

theorem water_balloon_problem :
  popped_balloons 2 30 3 40 170 = 10 := by
  sorry

end NUMINAMATH_CALUDE_water_balloon_problem_l1340_134012


namespace NUMINAMATH_CALUDE_platform_length_l1340_134087

/-- Given a train of length 900 meters that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, prove that the length of the platform is 1050 meters. -/
theorem platform_length
  (train_length : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_pole : ℝ)
  (h1 : train_length = 900)
  (h2 : time_cross_platform = 39)
  (h3 : time_cross_pole = 18) :
  let train_speed := train_length / time_cross_pole
  let platform_length := train_speed * time_cross_platform - train_length
  platform_length = 1050 := by sorry

end NUMINAMATH_CALUDE_platform_length_l1340_134087


namespace NUMINAMATH_CALUDE_sum_of_min_values_is_zero_l1340_134026

-- Define the polynomials P and Q
def P (a b x : ℝ) : ℝ := x^2 + a*x + b
def Q (c d x : ℝ) : ℝ := x^2 + c*x + d

-- Define the composition of P and Q
def PQ (a b c d x : ℝ) : ℝ := P a b (Q c d x)
def QP (a b c d x : ℝ) : ℝ := Q c d (P a b x)

-- State the theorem
theorem sum_of_min_values_is_zero 
  (a b c d : ℝ) 
  (h1 : PQ a b c d 1 = 0)
  (h2 : PQ a b c d 3 = 0)
  (h3 : PQ a b c d 5 = 0)
  (h4 : PQ a b c d 7 = 0)
  (h5 : QP a b c d 2 = 0)
  (h6 : QP a b c d 6 = 0)
  (h7 : QP a b c d 10 = 0)
  (h8 : QP a b c d 14 = 0) :
  ∃ (x y : ℝ), P a b x + Q c d y = 0 ∧ 
  (∀ z, P a b z ≥ P a b x) ∧ 
  (∀ w, Q c d w ≥ Q c d y) :=
sorry

end NUMINAMATH_CALUDE_sum_of_min_values_is_zero_l1340_134026


namespace NUMINAMATH_CALUDE_mean_of_six_numbers_with_sum_two_thirds_l1340_134058

theorem mean_of_six_numbers_with_sum_two_thirds :
  ∀ (a b c d e f : ℚ),
  a + b + c + d + e + f = 2/3 →
  (a + b + c + d + e + f) / 6 = 1/9 := by
sorry

end NUMINAMATH_CALUDE_mean_of_six_numbers_with_sum_two_thirds_l1340_134058


namespace NUMINAMATH_CALUDE_smith_initial_markers_l1340_134083

/-- The number of new boxes of markers Mr. Smith buys -/
def new_boxes : ℕ := 6

/-- The number of markers in each new box -/
def markers_per_box : ℕ := 9

/-- The total number of markers Mr. Smith has after buying new boxes -/
def total_markers : ℕ := 86

/-- The number of markers Mr. Smith had initially -/
def initial_markers : ℕ := total_markers - (new_boxes * markers_per_box)

theorem smith_initial_markers :
  initial_markers = 32 := by sorry

end NUMINAMATH_CALUDE_smith_initial_markers_l1340_134083


namespace NUMINAMATH_CALUDE_uncle_ben_eggs_l1340_134037

theorem uncle_ben_eggs (total_chickens : ℕ) (roosters : ℕ) (non_laying_hens : ℕ) (eggs_per_hen : ℕ) 
  (h1 : total_chickens = 440)
  (h2 : roosters = 39)
  (h3 : non_laying_hens = 15)
  (h4 : eggs_per_hen = 3) :
  total_chickens - roosters - non_laying_hens * eggs_per_hen = 1158 := by
  sorry

end NUMINAMATH_CALUDE_uncle_ben_eggs_l1340_134037


namespace NUMINAMATH_CALUDE_quotient_problem_l1340_134005

theorem quotient_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : 1 / (3 * a) + 1 / b = 2011)
  (h2 : 1 / a + 1 / (3 * b) = 1) :
  (a + b) / (a * b) = 1509 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l1340_134005


namespace NUMINAMATH_CALUDE_aqua_opposite_red_l1340_134021

-- Define the set of colors
inductive Color : Type
  | Red | White | Green | Brown | Aqua | Purple

-- Define a cube as a function from face positions to colors
def Cube := Fin 6 → Color

-- Define face positions
def top : Fin 6 := 0
def bottom : Fin 6 := 1
def front : Fin 6 := 2
def back : Fin 6 := 3
def right : Fin 6 := 4
def left : Fin 6 := 5

-- Define the conditions of the problem
def cube_conditions (c : Cube) : Prop :=
  (c top = Color.Brown) ∧
  (c right = Color.Green) ∧
  (c front = Color.Red ∨ c front = Color.White ∨ c front = Color.Purple) ∧
  (c back = Color.Aqua)

-- State the theorem
theorem aqua_opposite_red (c : Cube) :
  cube_conditions c → c front = Color.Red :=
by sorry

end NUMINAMATH_CALUDE_aqua_opposite_red_l1340_134021


namespace NUMINAMATH_CALUDE_ticket_sales_total_l1340_134009

/-- Calculates the total money collected from ticket sales -/
def total_money_collected (adult_price child_price : ℕ) (total_tickets children_tickets : ℕ) : ℕ :=
  let adult_tickets := total_tickets - children_tickets
  adult_price * adult_tickets + child_price * children_tickets

/-- Theorem stating that the total money collected is $104 -/
theorem ticket_sales_total : 
  total_money_collected 6 4 21 11 = 104 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_total_l1340_134009


namespace NUMINAMATH_CALUDE_polynomial_equality_l1340_134082

theorem polynomial_equality (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 9 * p^8 * q = 36 * p^7 * q^2 → p = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1340_134082


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l1340_134027

theorem complex_fraction_calculation : 
  let initial := 104 + 2 / 5
  let step1 := (initial / (3 / 8))
  let step2 := step1 / 2
  let step3 := step2 + (14 + 1 / 2)
  let step4 := step3 * (4 / 7)
  let final := step4 - (2 + 3 / 28)
  final = 86 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l1340_134027


namespace NUMINAMATH_CALUDE_wire_cutting_l1340_134017

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) :
  total_length = 49 →
  ratio = 2 / 5 →
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 14 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l1340_134017


namespace NUMINAMATH_CALUDE_region_area_correct_l1340_134023

/-- Given a circle with radius 36, two chords of length 66 intersecting at a point 12 units from
    the center at a 45° angle, this function calculates the area of one region formed by the
    intersection of the chords. -/
def calculate_region_area (radius : ℝ) (chord_length : ℝ) (intersection_distance : ℝ) 
    (intersection_angle : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the calculated area is correct for the given conditions. -/
theorem region_area_correct (radius : ℝ) (chord_length : ℝ) (intersection_distance : ℝ) 
    (intersection_angle : ℝ) :
  radius = 36 ∧ 
  chord_length = 66 ∧ 
  intersection_distance = 12 ∧ 
  intersection_angle = 45 * π / 180 →
  calculate_region_area radius chord_length intersection_distance intersection_angle > 0 :=
by sorry

end NUMINAMATH_CALUDE_region_area_correct_l1340_134023


namespace NUMINAMATH_CALUDE_james_local_taxes_l1340_134002

/-- Calculates the amount of local taxes paid in cents per hour -/
def local_taxes_cents (hourly_wage : ℚ) (tax_rate : ℚ) : ℚ :=
  hourly_wage * 100 * tax_rate

theorem james_local_taxes :
  local_taxes_cents 25 (24/1000) = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_local_taxes_l1340_134002


namespace NUMINAMATH_CALUDE_speed_conversion_l1340_134076

/-- Conversion factor from km/h to m/s -/
def kmph_to_mps : ℝ := 0.277778

/-- The given speed in km/h -/
def given_speed_kmph : ℝ := 162

/-- The speed in m/s to be proven -/
def speed_mps : ℝ := given_speed_kmph * kmph_to_mps

theorem speed_conversion :
  speed_mps = 45 := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l1340_134076


namespace NUMINAMATH_CALUDE_parallelogram_zk_product_l1340_134004

structure Parallelogram where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ

def is_valid_parallelogram (p : Parallelogram) (z k : ℝ) : Prop :=
  p.EF = 5 * z + 5 ∧
  p.FG = 4 * k^2 ∧
  p.GH = 40 ∧
  p.HE = k + 20

theorem parallelogram_zk_product (p : Parallelogram) (z k : ℝ) :
  is_valid_parallelogram p z k → z * k = (7 + 7 * Real.sqrt 321) / 8 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_zk_product_l1340_134004


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_nineteen_fourths_l1340_134063

theorem floor_plus_self_eq_nineteen_fourths :
  ∃! (x : ℚ), ⌊x⌋ + x = 19/4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_nineteen_fourths_l1340_134063


namespace NUMINAMATH_CALUDE_compare_with_one_twentieth_l1340_134051

theorem compare_with_one_twentieth : 
  (1 / 15 : ℚ) > 1 / 20 ∧ 
  (1 / 25 : ℚ) < 1 / 20 ∧ 
  (1 / 2 : ℚ) > 1 / 20 ∧ 
  (55 / 1000 : ℚ) > 1 / 20 ∧ 
  (1 / 10 : ℚ) > 1 / 20 := by
  sorry

#check compare_with_one_twentieth

end NUMINAMATH_CALUDE_compare_with_one_twentieth_l1340_134051


namespace NUMINAMATH_CALUDE_cow_sheep_value_l1340_134022

/-- The value of cows and sheep in taels of gold -/
theorem cow_sheep_value (x y : ℚ) 
  (h1 : 5 * x + 2 * y = 10) 
  (h2 : 2 * x + 5 * y = 8) : 
  x + y = 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cow_sheep_value_l1340_134022


namespace NUMINAMATH_CALUDE_M_equals_P_l1340_134064

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def P : Set ℝ := {a | ∃ x : ℝ, a = x^2 - 1}

theorem M_equals_P : M = P := by sorry

end NUMINAMATH_CALUDE_M_equals_P_l1340_134064


namespace NUMINAMATH_CALUDE_negation_equivalence_l1340_134086

theorem negation_equivalence (x y : ℝ) :
  ¬(x^2 + y^2 > 2 → |x| > 1 ∨ |y| > 1) ↔ (x^2 + y^2 ≤ 2 → |x| ≤ 1 ∧ |y| ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1340_134086


namespace NUMINAMATH_CALUDE_mason_car_nuts_l1340_134056

/-- The number of nuts in Mason's car after squirrels stockpile for a given number of days -/
def nuts_in_car (busy_squirrels : ℕ) (busy_nuts_per_day : ℕ) (sleepy_squirrels : ℕ) (sleepy_nuts_per_day : ℕ) (days : ℕ) : ℕ :=
  (busy_squirrels * busy_nuts_per_day + sleepy_squirrels * sleepy_nuts_per_day) * days

/-- Theorem stating the number of nuts in Mason's car -/
theorem mason_car_nuts :
  nuts_in_car 2 30 1 20 40 = 3200 :=
by sorry

end NUMINAMATH_CALUDE_mason_car_nuts_l1340_134056


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_one_l1340_134031

theorem arithmetic_expression_equals_one : 3 * (7 - 5) - 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_one_l1340_134031


namespace NUMINAMATH_CALUDE_inequality_solution_l1340_134077

theorem inequality_solution (x : ℝ) : 3 - 2 / (3 * x + 4) < 5 ↔ x < -4/3 ∨ x > -5/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1340_134077


namespace NUMINAMATH_CALUDE_correct_ages_l1340_134061

/-- Represents the ages of family members -/
structure FamilyAges where
  father : ℕ
  son : ℕ
  mother : ℕ

/-- Calculates the correct ages given the problem conditions -/
def calculateAges : FamilyAges :=
  let father := 44
  let son := father / 2
  let mother := son + 5
  { father := father, son := son, mother := mother }

/-- Theorem stating that the calculated ages satisfy the given conditions -/
theorem correct_ages (ages : FamilyAges := calculateAges) :
  ages.father = 44 ∧
  ages.father = ages.son + ages.son ∧
  ages.son - 5 = ages.mother - 10 ∧
  ages.father = 44 ∧
  ages.son = 22 ∧
  ages.mother = 27 :=
by sorry

end NUMINAMATH_CALUDE_correct_ages_l1340_134061


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1340_134028

/-- Given a principal amount where the simple interest for 3 years at 10% per annum is 900,
    prove that the compound interest for the same period and rate is 993. -/
theorem compound_interest_problem (P : ℝ) : 
  P * 0.10 * 3 = 900 → 
  P * (1 + 0.10)^3 - P = 993 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l1340_134028


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l1340_134033

theorem complex_modulus_sqrt_two (x y : ℝ) (h : (1 + Complex.I) * x = 1 + y * Complex.I) :
  Complex.abs (x + y * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l1340_134033


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l1340_134075

theorem sum_of_roots_zero (m n : ℝ) : 
  (∀ x, x^2 - 2*(m+n)*x + m*n = 0 ↔ x = m ∨ x = n) → m + n = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l1340_134075


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1340_134098

theorem smallest_x_absolute_value_equation :
  ∃ (x : ℝ), x = -5.5 ∧ |4*x + 7| = 15 ∧ ∀ (y : ℝ), |4*y + 7| = 15 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1340_134098


namespace NUMINAMATH_CALUDE_complex_number_equation_l1340_134011

/-- Given a complex number z = 1 + √2i, prove that z^2 - 2z = -3 -/
theorem complex_number_equation : 
  let z : ℂ := 1 + Complex.I * Real.sqrt 2
  z^2 - 2*z = -3 := by sorry

end NUMINAMATH_CALUDE_complex_number_equation_l1340_134011


namespace NUMINAMATH_CALUDE_greatest_integer_c_for_domain_all_reals_l1340_134043

theorem greatest_integer_c_for_domain_all_reals : 
  (∃ c : ℤ, (∀ x : ℝ, (x^2 + c*x + 10 ≠ 0)) ∧ 
   (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 10 = 0)) → 
  (∃ c : ℤ, c = 6 ∧ 
   (∀ x : ℝ, (x^2 + c*x + 10 ≠ 0)) ∧ 
   (∀ d : ℤ, d > c → ∃ x : ℝ, x^2 + d*x + 10 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_c_for_domain_all_reals_l1340_134043


namespace NUMINAMATH_CALUDE_regular_ngon_diagonal_difference_l1340_134016

/-- The difference between the longest and shortest diagonals of a regular n-gon equals its side if and only if n = 9 -/
theorem regular_ngon_diagonal_difference (n : ℕ) : n ≥ 3 → (
  let R := (1 : ℝ)  -- Assume unit circumradius for simplicity
  let side_length := 2 * R * Real.sin (π / n)
  let shortest_diagonal := 2 * R * Real.sin (2 * π / n)
  let longest_diagonal := if n % 2 = 0 then 2 * R else 2 * R * Real.cos (π / (2 * n))
  longest_diagonal - shortest_diagonal = side_length
) ↔ n = 9 := by sorry

end NUMINAMATH_CALUDE_regular_ngon_diagonal_difference_l1340_134016


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l1340_134025

theorem complex_imaginary_part (z : ℂ) (h : (3 + 4*I)*z = 5) : 
  z.im = -4/5 := by sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l1340_134025


namespace NUMINAMATH_CALUDE_remainder_theorem_l1340_134085

theorem remainder_theorem (P D Q R D' Q' R' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + R')
  (h3 : R < D) :
  P % (D + D') = R :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1340_134085


namespace NUMINAMATH_CALUDE_road_repair_workers_l1340_134032

/-- The number of persons in the first group -/
def first_group : ℕ := 63

/-- The number of days the first group works -/
def first_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_hours : ℕ := 5

/-- The number of days the second group works -/
def second_days : ℕ := 21

/-- The number of hours per day the second group works -/
def second_hours : ℕ := 6

/-- The total man-hours required to complete the work -/
def total_man_hours : ℕ := first_group * first_days * first_hours

/-- The number of persons in the second group -/
def second_group : ℕ := total_man_hours / (second_days * second_hours)

theorem road_repair_workers :
  second_group = 30 :=
by sorry

end NUMINAMATH_CALUDE_road_repair_workers_l1340_134032


namespace NUMINAMATH_CALUDE_production_increase_l1340_134097

/-- Calculates the number of units produced today given the previous average, 
    number of days, and new average including today's production. -/
def units_produced_today (prev_avg : ℝ) (prev_days : ℕ) (new_avg : ℝ) : ℝ :=
  (new_avg * (prev_days + 1)) - (prev_avg * prev_days)

/-- Proves that given the conditions, the number of units produced today is 90. -/
theorem production_increase (prev_avg : ℝ) (prev_days : ℕ) (new_avg : ℝ) 
  (h1 : prev_avg = 60)
  (h2 : prev_days = 5)
  (h3 : new_avg = 65) :
  units_produced_today prev_avg prev_days new_avg = 90 := by
  sorry

#eval units_produced_today 60 5 65

end NUMINAMATH_CALUDE_production_increase_l1340_134097
