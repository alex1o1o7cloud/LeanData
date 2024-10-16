import Mathlib

namespace NUMINAMATH_CALUDE_josh_ribbon_shortage_l4175_417529

/-- Calculates the shortage of ribbon for gift wrapping --/
def ribbon_shortage (total_ribbon : ℝ) (num_gifts : ℕ) 
  (wrap_per_gift : ℝ) (bow_per_gift : ℝ) (tag_per_gift : ℝ) (trim_per_gift : ℝ) : ℝ :=
  let required_ribbon := num_gifts * (wrap_per_gift + bow_per_gift + tag_per_gift + trim_per_gift)
  required_ribbon - total_ribbon

/-- Proves that Josh is short by 7.5 yards of ribbon --/
theorem josh_ribbon_shortage : 
  ribbon_shortage 18 6 2 1.5 0.25 0.5 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_josh_ribbon_shortage_l4175_417529


namespace NUMINAMATH_CALUDE_thomas_monthly_pay_l4175_417571

/-- The amount paid to a worker after one month, given their weekly rate and the number of weeks in a month -/
def monthly_pay (weekly_rate : ℕ) (weeks_per_month : ℕ) : ℕ :=
  weekly_rate * weeks_per_month

theorem thomas_monthly_pay :
  monthly_pay 4550 4 = 18200 := by
  sorry

end NUMINAMATH_CALUDE_thomas_monthly_pay_l4175_417571


namespace NUMINAMATH_CALUDE_characterize_g_l4175_417579

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the properties of g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 12 * x + 4

-- Theorem statement
theorem characterize_g :
  ∀ g : ℝ → ℝ, is_valid_g g ↔ (∀ x, g x = 3 * x - 2 ∨ g x = -3 * x + 2) :=
by sorry

end NUMINAMATH_CALUDE_characterize_g_l4175_417579


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l4175_417590

theorem smallest_integer_with_remainders : ∃! N : ℕ+, 
  (N : ℤ) % 7 = 5 ∧ 
  (N : ℤ) % 8 = 6 ∧ 
  (N : ℤ) % 9 = 7 ∧ 
  ∀ M : ℕ+, 
    ((M : ℤ) % 7 = 5 ∧ (M : ℤ) % 8 = 6 ∧ (M : ℤ) % 9 = 7) → N ≤ M :=
by
  use 502
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l4175_417590


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l4175_417503

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A line passing through a point and intersecting an ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  intersectionPoints : Fin 2 → ℝ × ℝ

/-- The problem statement -/
theorem ellipse_and_line_properties
  (E : Ellipse)
  (l : IntersectingLine E)
  (h₁ : E.a^2 - E.b^2 = 1) -- Condition for foci at (-1,0) and (1,0)
  (h₂ : (l.intersectionPoints 0).1 + (l.intersectionPoints 1).1 +
        ((l.intersectionPoints 0).1 + 1)^2 + (l.intersectionPoints 0).2^2 +
        ((l.intersectionPoints 1).1 + 1)^2 + (l.intersectionPoints 1).2^2 = 16) -- Perimeter condition
  (h₃ : (l.intersectionPoints 0).1 * (l.intersectionPoints 1).1 +
        (l.intersectionPoints 0).2 * (l.intersectionPoints 1).2 = 0) -- Perpendicularity condition
  : (E.a = Real.sqrt 3 ∧ E.b = Real.sqrt 2) ∧
    (l.k = Real.sqrt 2 ∨ l.k = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l4175_417503


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l4175_417559

/-- The function f(x) = x^3 - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_parallel_points :
  ∀ x : ℝ, (f' x = 2) ↔ (x = 1 ∨ x = -1) := by sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l4175_417559


namespace NUMINAMATH_CALUDE_count_numbers_with_2_and_3_is_52_l4175_417597

/-- A function that counts the number of three-digit numbers with at least one 2 and one 3 -/
def count_numbers_with_2_and_3 : ℕ :=
  let hundreds_not_2_or_3 := 7 * 2  -- Case 1
  let hundreds_is_2 := 10 + 9       -- Case 2
  let hundreds_is_3 := 10 + 9       -- Case 3
  hundreds_not_2_or_3 + hundreds_is_2 + hundreds_is_3

/-- Theorem stating that the count of three-digit numbers with at least one 2 and one 3 is 52 -/
theorem count_numbers_with_2_and_3_is_52 : count_numbers_with_2_and_3 = 52 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_2_and_3_is_52_l4175_417597


namespace NUMINAMATH_CALUDE_sqrt_122_between_integers_product_l4175_417572

theorem sqrt_122_between_integers_product : ∃ (n : ℕ), 
  (n : ℝ) < Real.sqrt 122 ∧ 
  Real.sqrt 122 < (n + 1 : ℝ) ∧ 
  n * (n + 1) = 132 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_122_between_integers_product_l4175_417572


namespace NUMINAMATH_CALUDE_pond_A_has_more_fish_l4175_417569

-- Define the capture-recapture estimation function
def estimateFishPopulation (totalSecondCatch : ℕ) (totalMarkedReleased : ℕ) (markedInSecondCatch : ℕ) : ℚ :=
  (totalSecondCatch * totalMarkedReleased : ℚ) / markedInSecondCatch

-- Define the parameters for each pond
def pondAMarkedFish : ℕ := 8
def pondBMarkedFish : ℕ := 16
def fishCaught : ℕ := 200
def fishMarked : ℕ := 200

-- Theorem statement
theorem pond_A_has_more_fish :
  estimateFishPopulation fishCaught fishMarked pondAMarkedFish >
  estimateFishPopulation fishCaught fishMarked pondBMarkedFish :=
by
  sorry

end NUMINAMATH_CALUDE_pond_A_has_more_fish_l4175_417569


namespace NUMINAMATH_CALUDE_g_fixed_points_l4175_417516

def g (x : ℝ) : ℝ := x^2 - 5*x

theorem g_fixed_points :
  ∀ x : ℝ, g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = -2 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_g_fixed_points_l4175_417516


namespace NUMINAMATH_CALUDE_max_value_of_product_sum_l4175_417586

theorem max_value_of_product_sum (x y z : ℝ) (h : x + 2*y + z = 7) :
  ∃ (max : ℝ), max = 7 ∧ ∀ (x' y' z' : ℝ), x' + 2*y' + z' = 7 → x'*y' + x'*z' + y'*z' ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_product_sum_l4175_417586


namespace NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l4175_417575

/-- 
Given a quadratic equation ax^2 + 3bx + c = 0 with zero discriminant,
prove that the coefficients a, b, and c form a geometric progression.
-/
theorem quadratic_zero_discriminant_geometric_progression 
  (a b c : ℝ) (h_nonzero : a ≠ 0) 
  (h_discriminant : 9 * b^2 - 4 * a * c = 0) :
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r :=
sorry

end NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l4175_417575


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l4175_417531

theorem binomial_coefficient_sum (n : ℕ) : 4^n - 2^n = 992 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l4175_417531


namespace NUMINAMATH_CALUDE_not_arithmetic_sequence_sqrt_2_3_5_l4175_417568

theorem not_arithmetic_sequence_sqrt_2_3_5 : ¬∃ (a b c : ℝ), 
  (a = Real.sqrt 2) ∧ 
  (b = Real.sqrt 3) ∧ 
  (c = Real.sqrt 5) ∧ 
  (b - a = c - b) :=
by sorry

end NUMINAMATH_CALUDE_not_arithmetic_sequence_sqrt_2_3_5_l4175_417568


namespace NUMINAMATH_CALUDE_floor_equation_solution_l4175_417578

theorem floor_equation_solution (x : ℝ) : 
  ⌊(3:ℝ) * x + 4⌋ = ⌊(5:ℝ) * x - 1⌋ ↔ 
  ((11:ℝ)/5 ≤ x ∧ x < 7/3) ∨ 
  ((12:ℝ)/5 ≤ x ∧ x < 13/5) ∨ 
  ((8:ℝ)/3 ≤ x ∧ x < 14/5) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l4175_417578


namespace NUMINAMATH_CALUDE_initial_number_proof_l4175_417548

theorem initial_number_proof : ∃ n : ℕ, n ≥ 102 ∧ (n - 5) % 97 = 0 ∧ ∀ m : ℕ, m < n → (m - 5) % 97 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l4175_417548


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4175_417546

theorem quadratic_inequality (x : ℝ) : x^2 + 9*x + 8 < 0 ↔ -8 < x ∧ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4175_417546


namespace NUMINAMATH_CALUDE_interest_difference_l4175_417513

/-- Calculate the difference between compound interest and simple interest -/
theorem interest_difference (P : ℝ) (r : ℝ) (t : ℝ) (n : ℝ) : 
  P = 6000.000000000128 →
  r = 0.05 →
  t = 2 →
  n = 1 →
  let CI := P * (1 + r/n)^(n*t) - P
  let SI := P * r * t
  abs (CI - SI - 15.0000000006914) < 1e-10 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_l4175_417513


namespace NUMINAMATH_CALUDE_find_m_l4175_417505

def U : Set ℕ := {1, 2, 3, 4}

def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5*x + m = 0}

theorem find_m : ∃ m : ℕ, (U \ A m) = {1, 4} := by sorry

end NUMINAMATH_CALUDE_find_m_l4175_417505


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l4175_417555

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 2) :
  ∃ (m : ℝ), m = -5/2 ∧ ∀ x, 8 * x^2 + 6 * x + 5 = 2 → 3 * x + 2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l4175_417555


namespace NUMINAMATH_CALUDE_elevator_problem_l4175_417582

theorem elevator_problem (x y z w v a b c n : ℕ) : 
  x = 20 ∧ 
  y = 7 ∧ 
  z = 3^2 ∧ 
  w = 5^2 ∧ 
  v = 3^2 ∧ 
  a = 3^2 - 2 ∧ 
  b = 3 ∧ 
  c = 1^3 ∧ 
  x - y + z - w + v - a + b - c = n 
  → n = 1 := by
sorry

end NUMINAMATH_CALUDE_elevator_problem_l4175_417582


namespace NUMINAMATH_CALUDE_polynomial_equation_l4175_417506

variables (x : ℝ)

def f (x : ℝ) : ℝ := x^4 - 3*x^2 - x + 5

def g (x : ℝ) : ℝ := -x^4 + 7*x^2 + x - 6

theorem polynomial_equation :
  f x + g x = 4*x^2 + x - 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_equation_l4175_417506


namespace NUMINAMATH_CALUDE_min_value_of_f_l4175_417528

def f (x : ℝ) : ℝ := x^2 - 8*x + 15

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x₀ : ℝ), f x₀ = m) ∧ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l4175_417528


namespace NUMINAMATH_CALUDE_difference_of_squares_times_three_l4175_417537

theorem difference_of_squares_times_three :
  (650^2 - 350^2) * 3 = 900000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_times_three_l4175_417537


namespace NUMINAMATH_CALUDE_average_geometric_sequence_l4175_417556

theorem average_geometric_sequence (y : ℝ) : 
  let sequence := [0, 3*y, 9*y, 27*y, 81*y]
  (sequence.sum / sequence.length : ℝ) = 24*y := by
  sorry

end NUMINAMATH_CALUDE_average_geometric_sequence_l4175_417556


namespace NUMINAMATH_CALUDE_log2_derivative_l4175_417522

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
sorry

end NUMINAMATH_CALUDE_log2_derivative_l4175_417522


namespace NUMINAMATH_CALUDE_expansion_equals_fourth_power_l4175_417576

theorem expansion_equals_fourth_power (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*(x - 1) + 1 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equals_fourth_power_l4175_417576


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l4175_417563

theorem boat_speed_ratio (v : ℝ) (c : ℝ) (d : ℝ) 
  (hv : v = 24) -- Boat speed in still water
  (hc : c = 6)  -- River current speed
  (hd : d = 3)  -- Distance traveled downstream and upstream
  : (2 * d) / ((d / (v + c)) + (d / (v - c))) / v = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l4175_417563


namespace NUMINAMATH_CALUDE_quadratic_function_sum_l4175_417538

theorem quadratic_function_sum (a b : ℝ) (h1 : a ≠ 0) : 
  (∀ x y : ℝ, y = a * x^2 + b * x - 1) → 
  (1 = a * 1^2 + b * 1 - 1) →
  a + b + 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_sum_l4175_417538


namespace NUMINAMATH_CALUDE_average_transformation_l4175_417514

theorem average_transformation (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = 8) : 
  ((a₁ + 10) + (a₂ - 10) + (a₃ + 10) + (a₄ - 10) + (a₅ + 10)) / 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_transformation_l4175_417514


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_7500_l4175_417558

theorem last_three_digits_of_7_to_7500 (h : 7^500 ≡ 1 [ZMOD 1250]) :
  7^7500 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_7500_l4175_417558


namespace NUMINAMATH_CALUDE_x_values_proof_l4175_417577

theorem x_values_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 7) (h2 : y + 1 / x = 7 / 8) :
  x = 1 ∨ x = 8 := by
sorry

end NUMINAMATH_CALUDE_x_values_proof_l4175_417577


namespace NUMINAMATH_CALUDE_loan_amount_calculation_l4175_417591

/-- Proves that given the initial amount, interest rate, and final amount, 
    the calculated loan amount is correct. -/
theorem loan_amount_calculation 
  (initial_amount : ℝ) 
  (interest_rate : ℝ) 
  (final_amount : ℝ) 
  (loan_amount : ℝ) : 
  initial_amount = 30 ∧ 
  interest_rate = 0.20 ∧ 
  final_amount = 33 ∧
  loan_amount = 2.50 → 
  initial_amount + loan_amount * (1 + interest_rate) = final_amount :=
by sorry

end NUMINAMATH_CALUDE_loan_amount_calculation_l4175_417591


namespace NUMINAMATH_CALUDE_min_bricks_needed_l4175_417570

/-- Represents the dimensions of a brick -/
structure BrickDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of the parallelepiped -/
structure ParallelepipedDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The theorem statement -/
theorem min_bricks_needed
  (brick : BrickDimensions)
  (parallelepiped : ParallelepipedDimensions)
  (h1 : brick.length = 22)
  (h2 : brick.width = 11)
  (h3 : brick.height = 6)
  (h4 : parallelepiped.length = 5 * parallelepiped.height / 4)
  (h5 : parallelepiped.width = 3 * parallelepiped.height / 2)
  (h6 : parallelepiped.length % brick.length = 0)
  (h7 : parallelepiped.width % brick.width = 0)
  (h8 : parallelepiped.height % brick.height = 0) :
  (parallelepiped.length / brick.length) *
  (parallelepiped.width / brick.width) *
  (parallelepiped.height / brick.height) = 13200 := by
  sorry

end NUMINAMATH_CALUDE_min_bricks_needed_l4175_417570


namespace NUMINAMATH_CALUDE_apples_handed_out_correct_l4175_417512

/-- Represents the cafeteria's apple distribution problem -/
def apples_handed_out (initial_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - (num_pies * apples_per_pie)

/-- Proves that the number of apples handed out is correct -/
theorem apples_handed_out_correct (initial_apples : ℕ) (num_pies : ℕ) (apples_per_pie : ℕ) :
  apples_handed_out initial_apples num_pies apples_per_pie =
  initial_apples - (num_pies * apples_per_pie) :=
by
  sorry

#eval apples_handed_out 47 5 4

end NUMINAMATH_CALUDE_apples_handed_out_correct_l4175_417512


namespace NUMINAMATH_CALUDE_four_variable_inequality_l4175_417530

theorem four_variable_inequality (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_one : a + b + c + d = 1) :
  a * b * c * d + b * c * d * a + c * d * a * b + d * a * b * c ≤ 1 / 27 + 176 / 27 * a * b * c * d := by
  sorry

end NUMINAMATH_CALUDE_four_variable_inequality_l4175_417530


namespace NUMINAMATH_CALUDE_rope_remaining_l4175_417541

theorem rope_remaining (initial_length : ℝ) (fraction_to_allan : ℝ) (fraction_to_jack : ℝ) :
  initial_length = 20 ∧ 
  fraction_to_allan = 1/4 ∧ 
  fraction_to_jack = 2/3 →
  initial_length * (1 - fraction_to_allan) * (1 - fraction_to_jack) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rope_remaining_l4175_417541


namespace NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l4175_417543

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (π / 2 - α) = Real.sqrt 2 / 3) :
  Real.cos (π - 2 * α) = -5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l4175_417543


namespace NUMINAMATH_CALUDE_roulette_wheel_probability_l4175_417560

/-- The probability of a roulette wheel landing on section F -/
def prob_F (prob_D prob_E prob_G : ℚ) : ℚ :=
  1 - (prob_D + prob_E + prob_G)

/-- Theorem: The probability of landing on section F is 1/4 -/
theorem roulette_wheel_probability :
  let prob_D : ℚ := 3/8
  let prob_E : ℚ := 1/4
  let prob_G : ℚ := 1/8
  prob_F prob_D prob_E prob_G = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_roulette_wheel_probability_l4175_417560


namespace NUMINAMATH_CALUDE_book_selection_ways_l4175_417521

def num_books : ℕ := 5
def num_students : ℕ := 2

theorem book_selection_ways :
  (num_books ^ num_students : ℕ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_ways_l4175_417521


namespace NUMINAMATH_CALUDE_consecutive_integers_sqrt_13_l4175_417501

theorem consecutive_integers_sqrt_13 (m n : ℤ) : 
  (n = m + 1) → (m < Real.sqrt 13) → (Real.sqrt 13 < n) → m * n = 12 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sqrt_13_l4175_417501


namespace NUMINAMATH_CALUDE_playground_fence_posts_l4175_417587

/-- Calculates the number of fence posts required for a rectangular playground -/
def fence_posts (width : ℕ) (length : ℕ) (post_interval : ℕ) : ℕ :=
  let long_side_posts := length / post_interval + 2
  let short_side_posts := width / post_interval + 1
  long_side_posts + 2 * short_side_posts

/-- Theorem stating the number of fence posts for a 50m by 90m playground -/
theorem playground_fence_posts :
  fence_posts 50 90 10 = 25 := by
  sorry

#eval fence_posts 50 90 10

end NUMINAMATH_CALUDE_playground_fence_posts_l4175_417587


namespace NUMINAMATH_CALUDE_pencil_distribution_l4175_417584

/-- Given an initial number of pencils, number of containers, and additional pencils,
    calculate the number of pencils that can be evenly distributed per container. -/
def evenDistribution (initialPencils : ℕ) (containers : ℕ) (additionalPencils : ℕ) : ℕ :=
  (initialPencils + additionalPencils) / containers

/-- Prove that given 150 initial pencils, 5 containers, and 30 additional pencils,
    the number of pencils that can be evenly distributed between the containers
    after receiving additional pencils is 36. -/
theorem pencil_distribution :
  evenDistribution 150 5 30 = 36 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l4175_417584


namespace NUMINAMATH_CALUDE_parabola_vertex_l4175_417566

/-- The vertex of a parabola defined by y = -(x+1)^2 is the point (-1, 0) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -(x + 1)^2 → (∃ (a : ℝ), y = a * (x + 1)^2 ∧ a = -1) → 
  (∃ (h k : ℝ), y = -(x - h)^2 + k ∧ h = -1 ∧ k = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l4175_417566


namespace NUMINAMATH_CALUDE_apple_distribution_l4175_417552

theorem apple_distribution (total_apples : ℕ) (new_people : ℕ) (apple_reduction : ℕ) 
  (h1 : total_apples = 2750)
  (h2 : new_people = 60)
  (h3 : apple_reduction = 12) :
  ∃ (original_people : ℕ),
    (total_apples / original_people : ℚ) - 
    (total_apples / (original_people + new_people) : ℚ) = apple_reduction ∧
    total_apples / original_people = 30 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l4175_417552


namespace NUMINAMATH_CALUDE_sum_multiple_special_property_l4175_417527

def is_sum_multiple (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ m = (n / 100 + (n / 10) % 10 + n % 10) ∧ n % m = 0

def digit_sum (n : ℕ) : ℕ :=
  n / 100 + (n / 10) % 10 + n % 10

def F (n : ℕ) : ℕ :=
  max (n / 100 * 10 + (n / 10) % 10) (max (n / 100 * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

def G (n : ℕ) : ℕ :=
  min (n / 100 * 10 + (n / 10) % 10) (min (n / 100 * 10 + n % 10) ((n / 10) % 10 * 10 + n % 10))

theorem sum_multiple_special_property :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧
    is_sum_multiple n ∧
    digit_sum n = 12 ∧
    n / 100 > (n / 10) % 10 ∧ (n / 10) % 10 > n % 10 ∧
    (F n + G n) % 16 = 0} =
  {732, 372, 516, 156} := by
  sorry

end NUMINAMATH_CALUDE_sum_multiple_special_property_l4175_417527


namespace NUMINAMATH_CALUDE_divisibility_problem_l4175_417589

theorem divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    ((a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l4175_417589


namespace NUMINAMATH_CALUDE_tax_difference_is_twenty_cents_l4175_417565

/-- The price of the item before tax -/
def price : ℝ := 40

/-- The first tax rate as a percentage -/
def tax_rate1 : ℝ := 7.25

/-- The second tax rate as a percentage -/
def tax_rate2 : ℝ := 6.75

/-- Theorem stating the difference between the two tax amounts -/
theorem tax_difference_is_twenty_cents :
  (price * (tax_rate1 / 100)) - (price * (tax_rate2 / 100)) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_tax_difference_is_twenty_cents_l4175_417565


namespace NUMINAMATH_CALUDE_only_square_and_pentagon_equal_diagonals_l4175_417504

/-- A polygon is a closed planar figure with straight sides. -/
structure Polygon where
  sides : ℕ
  vertices : ℕ
  diagonals : ℕ

/-- A regular polygon is a polygon with all sides and angles equal. -/
structure RegularPolygon extends Polygon

/-- Predicate to check if all diagonals of a polygon are equal. -/
def all_diagonals_equal (p : Polygon) : Prop := sorry

/-- Theorem stating that only squares and regular pentagons have all diagonals equal. -/
theorem only_square_and_pentagon_equal_diagonals :
  ∀ p : Polygon, p.sides ≥ 3 →
    (all_diagonals_equal p ↔ 
      (p.sides = 4 ∧ ∃ (sq : RegularPolygon), sq.sides = 4) ∨
      (p.sides = 5 ∧ ∃ (pent : RegularPolygon), pent.sides = 5)) :=
by sorry

end NUMINAMATH_CALUDE_only_square_and_pentagon_equal_diagonals_l4175_417504


namespace NUMINAMATH_CALUDE_rabbit_carrot_problem_l4175_417567

theorem rabbit_carrot_problem (initial_carrots : ℕ) : 
  (((initial_carrots * 3 - 30) * 3 - 30) * 3 - 30) * 3 - 30 = 0 → 
  initial_carrots = 15 := by
sorry

end NUMINAMATH_CALUDE_rabbit_carrot_problem_l4175_417567


namespace NUMINAMATH_CALUDE_rectangle_area_l4175_417553

theorem rectangle_area (y : ℝ) (h : y > 0) :
  ∃ (w l : ℝ), w > 0 ∧ l > 0 ∧ l = 3 * w ∧ y^2 = l^2 + w^2 ∧ w * l = (3 * y^2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l4175_417553


namespace NUMINAMATH_CALUDE_cube_divisibility_l4175_417525

theorem cube_divisibility (n : ℕ) (h : ∀ k : ℕ, k > 0 → k < 42 → ¬(n ∣ k^3)) : n = 74088 := by
  sorry

end NUMINAMATH_CALUDE_cube_divisibility_l4175_417525


namespace NUMINAMATH_CALUDE_rearranged_balls_theorem_l4175_417535

/-- Represents a ball with its initial and final pile sizes -/
structure Ball where
  initialPileSize : ℕ+
  finalPileSize : ℕ+

/-- The problem statement -/
theorem rearranged_balls_theorem (n k : ℕ+) (balls : Finset Ball) 
    (h_initial_piles : (balls.sum fun b => (1 : ℚ) / b.initialPileSize) = n)
    (h_final_piles : (balls.sum fun b => (1 : ℚ) / b.finalPileSize) = n + k) :
    ∃ (subset : Finset Ball), subset.card = k + 1 ∧ 
    ∀ b ∈ subset, b.initialPileSize > b.finalPileSize :=
  sorry

end NUMINAMATH_CALUDE_rearranged_balls_theorem_l4175_417535


namespace NUMINAMATH_CALUDE_tangent_circles_count_l4175_417520

/-- Represents a line in a plane -/
structure Line where
  -- Add necessary fields for a line

/-- Represents a circle in a plane -/
structure Circle where
  -- Add necessary fields for a circle

/-- Checks if a circle is tangent to a line -/
def is_tangent (c : Circle) (l : Line) : Prop :=
  sorry

/-- Counts the number of circles tangent to all three given lines -/
def count_tangent_circles (l1 l2 l3 : Line) : Nat :=
  sorry

/-- The main theorem stating the possible values for the number of tangent circles -/
theorem tangent_circles_count (l1 l2 l3 : Line) :
  l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 →
  (count_tangent_circles l1 l2 l3 = 0 ∨
   count_tangent_circles l1 l2 l3 = 2 ∨
   count_tangent_circles l1 l2 l3 = 4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l4175_417520


namespace NUMINAMATH_CALUDE_square_perimeter_from_pi_shape_l4175_417564

/-- Given a square cut into four equal rectangles that form a П shape with a perimeter of 56,
    prove that the perimeter of the original square is 32. -/
theorem square_perimeter_from_pi_shape (x : ℝ) : 
  x > 0 →                    -- Ensure positive dimensions
  28 * x = 56 →              -- Perimeter of П shape is 56
  (4 * x) * 4 = 32 :=        -- Perimeter of original square is 32
by
  sorry


end NUMINAMATH_CALUDE_square_perimeter_from_pi_shape_l4175_417564


namespace NUMINAMATH_CALUDE_congruence_solution_l4175_417551

theorem congruence_solution (x : ℤ) :
  (15 * x + 2) % 18 = 7 % 18 → x % 6 = 1 % 6 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l4175_417551


namespace NUMINAMATH_CALUDE_equation_solution_l4175_417588

theorem equation_solution : ∃! x : ℝ, (Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 4) ∧ (x = 4 * Real.sqrt 11) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4175_417588


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l4175_417519

theorem gcd_power_two_minus_one : Nat.gcd (2^1025 - 1) (2^1056 - 1) = 2^31 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l4175_417519


namespace NUMINAMATH_CALUDE_range_of_a_l4175_417502

/-- Given a ≥ -2, prove that if C ⊆ B, then a ∈ [1/2, 3] -/
theorem range_of_a (a : ℝ) (ha : a ≥ -2) :
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ a}
  let B := {y : ℝ | ∃ x ∈ A, y = 2 * x + 3}
  let C := {t : ℝ | ∃ x ∈ A, t = x^2}
  C ⊆ B → a ∈ Set.Icc (1/2 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4175_417502


namespace NUMINAMATH_CALUDE_zachary_sold_40_games_l4175_417592

/-- Represents the sale of video games by three friends -/
structure VideoGameSale where
  /-- Amount of money Zachary received -/
  zachary_amount : ℝ
  /-- Price per game Zachary sold -/
  price_per_game : ℝ
  /-- Total amount received by all three friends -/
  total_amount : ℝ

/-- Theorem stating that Zachary sold 40 games given the conditions -/
theorem zachary_sold_40_games (sale : VideoGameSale)
  (h1 : sale.price_per_game = 5)
  (h2 : sale.zachary_amount + (sale.zachary_amount * 1.3) + (sale.zachary_amount * 1.3 + 50) = sale.total_amount)
  (h3 : sale.total_amount = 770) :
  sale.zachary_amount / sale.price_per_game = 40 := by
sorry


end NUMINAMATH_CALUDE_zachary_sold_40_games_l4175_417592


namespace NUMINAMATH_CALUDE_parabola_equation_fixed_point_property_l4175_417542

-- Define the ellipse E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 / 8 = 1}

-- Define the parabola C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the right focus of the ellipse E
def right_focus_E : ℝ × ℝ := (1, 0)

-- Define the directrix of parabola C
def directrix_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Theorem for part (I)
theorem parabola_equation : 
  C = {p : ℝ × ℝ | p.2^2 = 4 * p.1} := by sorry

-- Theorem for part (II)
theorem fixed_point_property (P Q : ℝ × ℝ) 
  (hP : P ∈ C) (hQ : Q ∈ C) (hO : P ≠ (0, 0) ∧ Q ≠ (0, 0)) 
  (hPerp : (P.1 * Q.1 + P.2 * Q.2 = 0)) :
  ∃ (m n : ℝ), m * P.2 = P.1 + n ∧ m * Q.2 = Q.1 + n ∧ n = -4 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_fixed_point_property_l4175_417542


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l4175_417526

/-- The coordinates of a point P with respect to the origin are the same as its given coordinates in a Cartesian coordinate system. -/
theorem point_coordinates_wrt_origin (x y : ℝ) : 
  let P : ℝ × ℝ := (x, y)
  P = (x, y) := by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l4175_417526


namespace NUMINAMATH_CALUDE_sin_270_degrees_l4175_417594

theorem sin_270_degrees : Real.sin (270 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_270_degrees_l4175_417594


namespace NUMINAMATH_CALUDE_binomial_equation_solution_l4175_417561

theorem binomial_equation_solution :
  ∃! (A B C : ℝ), ∀ (n : ℕ), n > 0 →
    2 * n^3 + 3 * n^2 = A * (n.choose 3) + B * (n.choose 2) + C * (n.choose 1) ∧
    A = 12 ∧ B = 18 ∧ C = 5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_equation_solution_l4175_417561


namespace NUMINAMATH_CALUDE_total_distance_driven_l4175_417510

theorem total_distance_driven (renaldo_distance : ℝ) (ernesto_extra : ℝ) (marcos_percentage : ℝ) : 
  renaldo_distance = 15 →
  ernesto_extra = 7 →
  marcos_percentage = 0.2 →
  let ernesto_distance := renaldo_distance / 3 + ernesto_extra
  let marcos_distance := (renaldo_distance + ernesto_distance) * (1 + marcos_percentage)
  renaldo_distance + ernesto_distance + marcos_distance = 59.4 := by
sorry

end NUMINAMATH_CALUDE_total_distance_driven_l4175_417510


namespace NUMINAMATH_CALUDE_power_sum_difference_l4175_417562

theorem power_sum_difference (m k p q : ℕ) : 
  2^m + 2^k = p → 2^m - 2^k = q → 2^(m+k) = (p^2 - q^2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_power_sum_difference_l4175_417562


namespace NUMINAMATH_CALUDE_next_year_with_digit_sum_five_l4175_417533

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem next_year_with_digit_sum_five : 
  ∀ y : ℕ, y > 2021 ∧ y < 2030 → sum_of_digits y ≠ 5 ∧ sum_of_digits 2030 = 5 :=
by sorry

end NUMINAMATH_CALUDE_next_year_with_digit_sum_five_l4175_417533


namespace NUMINAMATH_CALUDE_number_of_students_l4175_417557

theorem number_of_students (total_books : ℕ) : 
  (∃ (x : ℕ), 3 * x + 20 = total_books ∧ 4 * x = total_books + 25) → 
  (∃ (x : ℕ), x = 45 ∧ 3 * x + 20 = total_books ∧ 4 * x = total_books + 25) :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_l4175_417557


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_l4175_417532

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -5*x :=
by sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_l4175_417532


namespace NUMINAMATH_CALUDE_mice_eaten_in_decade_l4175_417547

/-- Calculates the number of mice eaten by a snake in a decade -/
theorem mice_eaten_in_decade (weeks_per_mouse : ℕ) (years_per_decade : ℕ) (weeks_per_year : ℕ) : 
  weeks_per_mouse = 4 → years_per_decade = 10 → weeks_per_year = 52 →
  (years_per_decade * weeks_per_year) / weeks_per_mouse = 130 := by
sorry

end NUMINAMATH_CALUDE_mice_eaten_in_decade_l4175_417547


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4175_417554

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) → (a > 3 ∨ a < -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4175_417554


namespace NUMINAMATH_CALUDE_ascending_order_negative_a_l4175_417581

theorem ascending_order_negative_a (a : ℝ) (h1 : -1 < a) (h2 : a < 0) :
  1 / a < a ∧ a < a^2 ∧ a^2 < |a| := by sorry

end NUMINAMATH_CALUDE_ascending_order_negative_a_l4175_417581


namespace NUMINAMATH_CALUDE_solve_equation_l4175_417595

theorem solve_equation : ∃ x : ℝ, 0.035 * x = 42 ∧ x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4175_417595


namespace NUMINAMATH_CALUDE_parabola_line_theorem_l4175_417544

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Checks if a point lies on a given parabola -/
def isOnParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is the centroid of a triangle -/
def isCentroid (centroid : Point) (p1 p2 p3 : Point) : Prop :=
  centroid.x = (p1.x + p2.x + p3.x) / 3 ∧
  centroid.y = (p1.y + p2.y + p3.y) / 3

theorem parabola_line_theorem (parabola : Parabola) 
    (A B C F : Point) : 
    isOnParabola A parabola → 
    isOnParabola B parabola → 
    isOnParabola C parabola → 
    A.x = 1 → 
    A.y = 2 → 
    F.x = parabola.p → 
    F.y = 0 → 
    isCentroid F A B C → 
    ∃ (line : Line), 
      line.a = 2 ∧ 
      line.b = 1 ∧ 
      line.c = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_theorem_l4175_417544


namespace NUMINAMATH_CALUDE_hot_dog_truck_profit_l4175_417517

/-- Calculates the profit for a hot dog food truck over a three-day period --/
theorem hot_dog_truck_profit
  (friday_customers : ℕ)
  (friday_tip_average : ℝ)
  (saturday_customer_multiplier : ℕ)
  (saturday_tip_average : ℝ)
  (sunday_customers : ℕ)
  (sunday_tip_average : ℝ)
  (hot_dog_price : ℝ)
  (ingredient_cost : ℝ)
  (daily_maintenance : ℝ)
  (weekend_taxes : ℝ)
  (h1 : friday_customers = 28)
  (h2 : friday_tip_average = 2)
  (h3 : saturday_customer_multiplier = 3)
  (h4 : saturday_tip_average = 2.5)
  (h5 : sunday_customers = 36)
  (h6 : sunday_tip_average = 1.5)
  (h7 : hot_dog_price = 4)
  (h8 : ingredient_cost = 1.25)
  (h9 : daily_maintenance = 50)
  (h10 : weekend_taxes = 150) :
  (friday_customers * friday_tip_average + 
   friday_customers * saturday_customer_multiplier * saturday_tip_average + 
   sunday_customers * sunday_tip_average +
   (friday_customers + friday_customers * saturday_customer_multiplier + sunday_customers) * 
   (hot_dog_price - ingredient_cost) - 
   3 * daily_maintenance - weekend_taxes) = 427 := by
  sorry


end NUMINAMATH_CALUDE_hot_dog_truck_profit_l4175_417517


namespace NUMINAMATH_CALUDE_min_remaining_fruits_last_fruit_is_banana_cannot_remove_all_fruits_l4175_417596

/-- Represents the types of fruits on the magical tree -/
inductive Fruit
  | Banana
  | Orange

/-- Represents the state of the magical tree -/
structure TreeState where
  bananas : Nat
  oranges : Nat

/-- Represents the possible picking actions -/
inductive PickAction
  | PickOne (f : Fruit)
  | PickTwo (f1 f2 : Fruit)

/-- Applies a picking action to the tree state -/
def applyAction (state : TreeState) (action : PickAction) : TreeState :=
  match action with
  | PickAction.PickOne Fruit.Banana => state
  | PickAction.PickOne Fruit.Orange => state
  | PickAction.PickTwo Fruit.Banana Fruit.Banana => 
      { bananas := state.bananas - 2, oranges := state.oranges + 1 }
  | PickAction.PickTwo Fruit.Orange Fruit.Orange => 
      { bananas := state.bananas, oranges := state.oranges - 1 }
  | PickAction.PickTwo Fruit.Banana Fruit.Orange => 
      { bananas := state.bananas, oranges := state.oranges - 1 }
  | PickAction.PickTwo Fruit.Orange Fruit.Banana => 
      { bananas := state.bananas, oranges := state.oranges - 1 }

/-- Defines the initial state of the tree -/
def initialState : TreeState := { bananas := 15, oranges := 20 }

/-- Theorem: The minimum number of fruits that can remain on the tree is 1 -/
theorem min_remaining_fruits (actions : List PickAction) :
  ∃ (finalState : TreeState), 
    (List.foldl applyAction initialState actions).bananas + 
    (List.foldl applyAction initialState actions).oranges ≥ 1 :=
  sorry

/-- Theorem: The last remaining fruit is always a banana -/
theorem last_fruit_is_banana (actions : List PickAction) :
  ∃ (finalState : TreeState), 
    (List.foldl applyAction initialState actions).bananas = 1 ∧
    (List.foldl applyAction initialState actions).oranges = 0 :=
  sorry

/-- Theorem: It's impossible to remove all fruits from the tree -/
theorem cannot_remove_all_fruits (actions : List PickAction) :
  ¬(∃ (finalState : TreeState), 
    (List.foldl applyAction initialState actions).bananas = 0 ∧
    (List.foldl applyAction initialState actions).oranges = 0) :=
  sorry

end NUMINAMATH_CALUDE_min_remaining_fruits_last_fruit_is_banana_cannot_remove_all_fruits_l4175_417596


namespace NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l4175_417500

theorem range_of_a_minus_abs_b (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : -4 < b ∧ b < 2) :
  ∀ x : ℝ, x = a - |b| → -3 < x ∧ x < 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_minus_abs_b_l4175_417500


namespace NUMINAMATH_CALUDE_number_of_elements_in_set_l4175_417515

theorem number_of_elements_in_set (initial_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) (final_avg : ℚ) : 
  initial_avg = 17 →
  incorrect_num = 26 →
  correct_num = 56 →
  final_avg = 20 →
  (∃ n : ℕ, n > 0 ∧ n * final_avg = n * initial_avg + (correct_num - incorrect_num) ∧ n = 10) :=
by sorry

end NUMINAMATH_CALUDE_number_of_elements_in_set_l4175_417515


namespace NUMINAMATH_CALUDE_coin_set_existence_l4175_417599

def is_valid_coin_set (weights : List Nat) : Prop :=
  ∀ k, k ∈ weights → 
    ∃ (A B : List Nat), 
      A ∪ B = weights.erase k ∧ 
      A.sum = B.sum

theorem coin_set_existence (n : Nat) : 
  (∃ weights : List Nat, 
    weights.length = n ∧ 
    weights.Nodup ∧
    is_valid_coin_set weights) ↔ 
  (Odd n ∧ n ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_coin_set_existence_l4175_417599


namespace NUMINAMATH_CALUDE_forty_six_in_sequence_l4175_417593

def laila_sequence (n : ℕ) : ℕ :=
  4 + 7 * (n - 1)

theorem forty_six_in_sequence : ∃ n : ℕ, laila_sequence n = 46 := by
  sorry

end NUMINAMATH_CALUDE_forty_six_in_sequence_l4175_417593


namespace NUMINAMATH_CALUDE_purple_balls_count_l4175_417509

theorem purple_balls_count (total : ℕ) (white green yellow red : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  white = 20 →
  green = 30 →
  yellow = 10 →
  red = 37 →
  prob_not_red_purple = 60 / 100 →
  (white + green + yellow : ℚ) / total = prob_not_red_purple →
  total - (white + green + yellow + red) = 3 := by
  sorry

end NUMINAMATH_CALUDE_purple_balls_count_l4175_417509


namespace NUMINAMATH_CALUDE_solve_linear_equation_l4175_417523

theorem solve_linear_equation (x : ℝ) :
  2*x - 3*x + 4*x = 150 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l4175_417523


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4175_417524

theorem inequality_solution_set (x : ℝ) :
  x ≠ 0 →
  ((2 * x - 5) * (x - 3)) / x ≥ 0 ↔ (0 < x ∧ x ≤ 5/2) ∨ (x ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4175_417524


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l4175_417580

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 2 - 3) :
  (1 - 3 / (m + 3)) / (m / (m^2 + 6*m + 9)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l4175_417580


namespace NUMINAMATH_CALUDE_problem_statement_l4175_417540

theorem problem_statement (x y z : ℝ) 
  (h1 : x^2 + 1/x^2 = 7)
  (h2 : x*y = 1)
  (h3 : z^2 + 1/z^2 = 9) :
  x^4 + y^4 - z^4 = 15 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4175_417540


namespace NUMINAMATH_CALUDE_common_chord_length_l4175_417583

theorem common_chord_length (r : ℝ) (d : ℝ) (h1 : r = 12) (h2 : d = 8) :
  let chord_length := 2 * Real.sqrt (r^2 - (d/2)^2)
  chord_length = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_l4175_417583


namespace NUMINAMATH_CALUDE_equal_expressions_l4175_417511

theorem equal_expressions (x : ℝ) (h : x > 0) : 
  (∃! n : ℕ, n = (if x^x + x^x = 2*x^x then 1 else 0) + 
              (if x^x + x^x = x^(2*x) then 1 else 0) + 
              (if x^x + x^x = (2*x)^x then 1 else 0) + 
              (if x^x + x^x = (2*x)^(2*x) then 1 else 0)) ∧
  (x^x + x^x = 2*x^x) ∧
  (x^x + x^x ≠ x^(2*x)) ∧
  (x^x + x^x ≠ (2*x)^x) ∧
  (x^x + x^x ≠ (2*x)^(2*x)) :=
by sorry

end NUMINAMATH_CALUDE_equal_expressions_l4175_417511


namespace NUMINAMATH_CALUDE_money_division_l4175_417508

theorem money_division (a b c : ℚ) : 
  a = (1/2) * b ∧ b = (1/2) * c ∧ c = 232 → a + b + c = 406 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l4175_417508


namespace NUMINAMATH_CALUDE_piecewise_continuity_l4175_417574

/-- A piecewise function f defined on real numbers -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then x^2 + x + 2 else 2*x + a

/-- Theorem stating that the piecewise function f is continuous at x = 3 if and only if a = 8 -/
theorem piecewise_continuity (a : ℝ) :
  ContinuousAt (f a) 3 ↔ a = 8 := by sorry

end NUMINAMATH_CALUDE_piecewise_continuity_l4175_417574


namespace NUMINAMATH_CALUDE_diophantine_equation_only_zero_solution_l4175_417585

theorem diophantine_equation_only_zero_solution (x y u t : ℤ) 
  (h : x^2 + y^2 = 1974 * (u^2 + t^2)) : x = 0 ∧ y = 0 ∧ u = 0 ∧ t = 0 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_only_zero_solution_l4175_417585


namespace NUMINAMATH_CALUDE_lottery_probabilities_l4175_417549

/-- Represents the probability of winning a single lottery event -/
def p : ℝ := 0.05

/-- The probability of winning both lotteries -/
def win_both : ℝ := p * p

/-- The probability of winning exactly one lottery -/
def win_one : ℝ := p * (1 - p) + (1 - p) * p

/-- The probability of winning at least one lottery -/
def win_at_least_one : ℝ := win_both + win_one

theorem lottery_probabilities :
  win_both = 0.0025 ∧ win_one = 0.095 ∧ win_at_least_one = 0.0975 := by
  sorry


end NUMINAMATH_CALUDE_lottery_probabilities_l4175_417549


namespace NUMINAMATH_CALUDE_six_balls_four_boxes_l4175_417550

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem six_balls_four_boxes : distribute_balls 6 4 = 157 := by sorry

end NUMINAMATH_CALUDE_six_balls_four_boxes_l4175_417550


namespace NUMINAMATH_CALUDE_min_pieces_is_3n_plus_1_l4175_417539

/-- A rectangular sheet of paper with holes -/
structure PerforatedSheet :=
  (n : ℕ)  -- number of holes
  (noOverlap : Bool)  -- holes do not overlap
  (parallelSides : Bool)  -- holes' sides are parallel to sheet edges

/-- The minimum number of rectangular pieces a perforated sheet can be divided into -/
def minPieces (sheet : PerforatedSheet) : ℕ :=
  3 * sheet.n + 1

/-- Theorem: The minimum number of rectangular pieces is 3n + 1 -/
theorem min_pieces_is_3n_plus_1 (sheet : PerforatedSheet) 
  (h1 : sheet.noOverlap = true) 
  (h2 : sheet.parallelSides = true) : 
  minPieces sheet = 3 * sheet.n + 1 := by
  sorry

end NUMINAMATH_CALUDE_min_pieces_is_3n_plus_1_l4175_417539


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4175_417536

/-- The sum of an arithmetic sequence with first term 2, common difference 4, and 15 terms -/
def arithmetic_sum : ℕ → ℕ
  | 0 => 0
  | n + 1 => (2 + 4 * n) + arithmetic_sum n

theorem arithmetic_sequence_sum : arithmetic_sum 15 = 450 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4175_417536


namespace NUMINAMATH_CALUDE_equivalent_proposition_and_truth_l4175_417573

theorem equivalent_proposition_and_truth :
  (∀ x : ℝ, x > 1 → (x - 1) * (x + 3) > 0) ↔
  (∀ x : ℝ, (x - 1) * (x + 3) ≤ 0 → x ≤ 1) ∧
  (∀ x : ℝ, x > 1 → (x - 1) * (x + 3) > 0) ∧
  (∀ x : ℝ, (x - 1) * (x + 3) ≤ 0 → x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_proposition_and_truth_l4175_417573


namespace NUMINAMATH_CALUDE_even_number_power_of_two_l4175_417507

theorem even_number_power_of_two (A : ℕ) :
  A % 2 = 0 →
  (∀ P : ℕ, Nat.Prime P → P ∣ A → (P - 1) ∣ (A - 1)) →
  ∃ k : ℕ, A = 2^k :=
sorry

end NUMINAMATH_CALUDE_even_number_power_of_two_l4175_417507


namespace NUMINAMATH_CALUDE_remaining_math_problems_l4175_417598

theorem remaining_math_problems (total : ℕ) (completed : ℕ) (remaining : ℕ) : 
  total = 9 → completed = 5 → remaining = total - completed → remaining = 4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_math_problems_l4175_417598


namespace NUMINAMATH_CALUDE_original_prices_l4175_417545

-- Define the sale prices and discount rates
def book_sale_price : ℚ := 8
def book_discount_rate : ℚ := 1 / 8
def pen_sale_price : ℚ := 4
def pen_discount_rate : ℚ := 1 / 5

-- Theorem statement
theorem original_prices :
  (book_sale_price / book_discount_rate = 64) ∧
  (pen_sale_price / pen_discount_rate = 20) :=
by sorry

end NUMINAMATH_CALUDE_original_prices_l4175_417545


namespace NUMINAMATH_CALUDE_franks_boxes_l4175_417518

/-- The number of boxes Frank filled with toys -/
def filled_boxes : ℕ := 8

/-- The number of boxes Frank has left empty -/
def empty_boxes : ℕ := 5

/-- The total number of boxes Frank had initially -/
def total_boxes : ℕ := filled_boxes + empty_boxes

theorem franks_boxes : total_boxes = 13 := by
  sorry

end NUMINAMATH_CALUDE_franks_boxes_l4175_417518


namespace NUMINAMATH_CALUDE_local_minimum_implies_a_in_open_unit_interval_l4175_417534

/-- The function f(x) = x³ - 3ax + 1 has a local minimum in the interval (0,1) -/
def has_local_minimum (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x, x ∈ Set.Ioo 0 1 ∧ ∀ y ∈ Set.Ioo 0 1, f y ≥ f x

/-- The main theorem stating that if f(x) = x³ - 3ax + 1 has a local minimum 
    in the interval (0,1), then 0 < a < 1 -/
theorem local_minimum_implies_a_in_open_unit_interval (a : ℝ) :
  has_local_minimum (fun x => x^3 - 3*a*x + 1) a → 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_implies_a_in_open_unit_interval_l4175_417534
