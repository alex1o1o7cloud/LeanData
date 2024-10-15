import Mathlib

namespace NUMINAMATH_CALUDE_range_of_m_l2409_240935

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (9 : ℝ)^x - m*(3 : ℝ)^x + 4 ≤ 0) → m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2409_240935


namespace NUMINAMATH_CALUDE_right_triangle_sin_c_l2409_240925

theorem right_triangle_sin_c (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : B = Real.pi / 2) (h3 : Real.sin A = 3 / 5) : Real.sin C = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_c_l2409_240925


namespace NUMINAMATH_CALUDE_solution_difference_l2409_240990

theorem solution_difference (r s : ℝ) : 
  (((5 * r - 15) / (r^2 + 3*r - 18) = r + 3) ∧
   ((5 * s - 15) / (s^2 + 3*s - 18) = s + 3) ∧
   (r ≠ s) ∧ (r > s)) →
  r - s = 13 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2409_240990


namespace NUMINAMATH_CALUDE_problem_solution_l2409_240943

theorem problem_solution : ∃! n : ℕ, n > 1 ∧ Nat.Prime n ∧ Even n ∧ n ≠ 9 ∧ ¬(15 ∣ n) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2409_240943


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l2409_240922

/-- Proves that the volume of fuel A added is 82 gallons given the specified conditions -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_A : ℝ) (ethanol_B : ℝ) (total_ethanol : ℝ) 
  (h1 : tank_capacity = 208)
  (h2 : ethanol_A = 0.12)
  (h3 : ethanol_B = 0.16)
  (h4 : total_ethanol = 30) :
  ∃ (fuel_A : ℝ), fuel_A = 82 ∧ 
  ethanol_A * fuel_A + ethanol_B * (tank_capacity - fuel_A) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l2409_240922


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l2409_240960

theorem binomial_coefficient_equality (n : ℕ) : 
  Nat.choose 18 n = Nat.choose 18 2 → n = 2 ∨ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l2409_240960


namespace NUMINAMATH_CALUDE_expenditure_recording_l2409_240911

/-- Represents the sign of a financial transaction -/
inductive TransactionSign
| Positive
| Negative

/-- Represents a financial transaction -/
structure Transaction where
  amount : ℕ
  sign : TransactionSign

/-- Records a transaction with the given amount and sign -/
def recordTransaction (amount : ℕ) (sign : TransactionSign) : Transaction :=
  { amount := amount, sign := sign }

/-- The rule for recording incomes and expenditures -/
axiom opposite_signs : 
  ∀ (income expenditure : Transaction), 
    income.sign = TransactionSign.Positive → 
    expenditure.sign = TransactionSign.Negative

/-- The main theorem -/
theorem expenditure_recording 
  (income : Transaction) 
  (h_income : income = recordTransaction 500 TransactionSign.Positive) :
  ∃ (expenditure : Transaction), 
    expenditure = recordTransaction 200 TransactionSign.Negative :=
sorry

end NUMINAMATH_CALUDE_expenditure_recording_l2409_240911


namespace NUMINAMATH_CALUDE_ab_value_l2409_240956

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 107) : a * b = 405 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2409_240956


namespace NUMINAMATH_CALUDE_true_propositions_l2409_240981

-- Define the propositions p and q
def p : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2)
def q : Prop := ∀ y : ℝ, y > 0 → ∃ x : ℝ, y = 3^x

-- Define the set of derived propositions
def derived_props : Set Prop := {p ∨ q, p ∧ q, ¬p, ¬q}

-- Define the set of true propositions
def true_props : Set Prop := {p ∨ q, ¬p}

-- Theorem statement
theorem true_propositions : 
  {prop ∈ derived_props | prop} = true_props := by sorry

end NUMINAMATH_CALUDE_true_propositions_l2409_240981


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_18_24_42_l2409_240986

theorem arithmetic_mean_of_18_24_42 :
  let numbers : List ℕ := [18, 24, 42]
  (numbers.sum : ℚ) / numbers.length = 28 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_18_24_42_l2409_240986


namespace NUMINAMATH_CALUDE_hyperbola_max_eccentricity_l2409_240933

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    and a point P on the right branch of the hyperbola satisfying |PF₁| = 4|PF₂|,
    the maximum value of the eccentricity e is 5/3. -/
theorem hyperbola_max_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ e_max : ℝ, e_max = 5/3 ∧
  ∀ (x y e : ℝ),
    x^2/a^2 - y^2/b^2 = 1 →
    x ≥ a →
    ∃ (F₁ F₂ : ℝ × ℝ),
      let P := (x, y)
      let d₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
      let d₂ := Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)
      d₁ = 4 * d₂ →
      e = Real.sqrt (1 + b^2/a^2) →
      e ≤ e_max :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_max_eccentricity_l2409_240933


namespace NUMINAMATH_CALUDE_square_area_ratio_l2409_240927

/-- The ratio of the areas of two squares with side lengths 3x and 5x respectively is 9/25 -/
theorem square_area_ratio (x : ℝ) (h : x > 0) :
  (3 * x)^2 / (5 * x)^2 = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2409_240927


namespace NUMINAMATH_CALUDE_elisas_painting_l2409_240963

theorem elisas_painting (monday : ℝ) 
  (h1 : monday > 0)
  (h2 : monday + 2 * monday + monday / 2 = 105) : 
  monday = 30 := by
  sorry

end NUMINAMATH_CALUDE_elisas_painting_l2409_240963


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2409_240974

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℂ, x^2 - (2*Complex.I - 1)*x + 3*m - Complex.I = 0 ∧ x.im = 0) → m = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2409_240974


namespace NUMINAMATH_CALUDE_problem_statement_l2409_240954

theorem problem_statement (X Y Z : ℕ+) 
  (h_coprime : Nat.gcd X.val (Nat.gcd Y.val Z.val) = 1)
  (h_equation : X.val * Real.log 3 / Real.log 100 + Y.val * Real.log 4 / Real.log 100 = (Z.val : ℝ)^2) :
  X.val + Y.val + Z.val = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2409_240954


namespace NUMINAMATH_CALUDE_white_balls_count_l2409_240992

/-- Given a bag of balls with the following properties:
  * The total number of balls is 40
  * The probability of drawing a red ball is 0.15
  * The probability of drawing a black ball is 0.45
  * The remaining balls are white
  
  This theorem proves that the number of white balls in the bag is 16. -/
theorem white_balls_count (total : ℕ) (p_red p_black : ℝ) :
  total = 40 →
  p_red = 0.15 →
  p_black = 0.45 →
  (total : ℝ) * (1 - p_red - p_black) = 16 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l2409_240992


namespace NUMINAMATH_CALUDE_min_offers_for_conviction_l2409_240940

/-- The minimum number of additional offers needed to be convinced with high probability. -/
def min_additional_offers : ℕ := 58

/-- The probability threshold for conviction. -/
def conviction_threshold : ℝ := 0.99

/-- The number of models already observed. -/
def observed_models : ℕ := 12

theorem min_offers_for_conviction :
  ∀ n : ℕ, n > observed_models →
    (observed_models : ℝ) / n ^ min_additional_offers < 1 - conviction_threshold :=
by sorry

end NUMINAMATH_CALUDE_min_offers_for_conviction_l2409_240940


namespace NUMINAMATH_CALUDE_opposite_of_negative_eight_l2409_240918

theorem opposite_of_negative_eight : 
  -((-8 : ℤ)) = (8 : ℤ) := by
sorry

end NUMINAMATH_CALUDE_opposite_of_negative_eight_l2409_240918


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l2409_240957

theorem angle_sum_at_point (y : ℝ) : 
  150 + y + 2*y = 360 → y = 70 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l2409_240957


namespace NUMINAMATH_CALUDE_solve_equation_l2409_240930

theorem solve_equation : ∃ x : ℝ, (7 - x = 9.5) ∧ (x = -2.5) := by sorry

end NUMINAMATH_CALUDE_solve_equation_l2409_240930


namespace NUMINAMATH_CALUDE_glove_profit_is_810_l2409_240938

/-- Calculates the profit from selling gloves given the purchase and sales information. -/
def glove_profit (total_pairs : ℕ) (cost_per_pair : ℚ) (sold_pairs_high : ℕ) (price_high : ℚ) (price_low : ℚ) : ℚ :=
  let remaining_pairs := total_pairs - sold_pairs_high
  let total_cost := cost_per_pair * total_pairs
  let revenue_high := price_high * sold_pairs_high
  let revenue_low := price_low * remaining_pairs
  let total_revenue := revenue_high + revenue_low
  total_revenue - total_cost

/-- The profit from selling gloves under the given conditions is 810 yuan. -/
theorem glove_profit_is_810 :
  glove_profit 600 12 470 14 11 = 810 := by
  sorry

#eval glove_profit 600 12 470 14 11

end NUMINAMATH_CALUDE_glove_profit_is_810_l2409_240938


namespace NUMINAMATH_CALUDE_dog_grouping_combinations_l2409_240912

def total_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 5
def group3_size : ℕ := 3

def buster_in_group1 : Prop := True
def whiskers_in_group2 : Prop := True

def remaining_dogs : ℕ := total_dogs - 2
def remaining_group1 : ℕ := group1_size - 1
def remaining_group2 : ℕ := group2_size - 1

theorem dog_grouping_combinations :
  buster_in_group1 →
  whiskers_in_group2 →
  Nat.choose remaining_dogs remaining_group1 * Nat.choose (remaining_dogs - remaining_group1) remaining_group2 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_combinations_l2409_240912


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2409_240923

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_3 : a + b + c = 3) :
  (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ (3 / 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2409_240923


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l2409_240970

-- Equation 1
theorem equation_one_solution (x : ℝ) : 9 * x^2 = 27 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) : -2 * (x - 3)^3 + 16 = 0 ↔ x = 5 := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l2409_240970


namespace NUMINAMATH_CALUDE_henry_tournament_points_l2409_240977

/-- A structure representing a tic-tac-toe tournament result -/
structure TournamentResult where
  win_points : ℕ
  loss_points : ℕ
  draw_points : ℕ
  wins : ℕ
  losses : ℕ
  draws : ℕ

/-- Calculate the total points for a given tournament result -/
def calculate_points (result : TournamentResult) : ℕ :=
  result.win_points * result.wins +
  result.loss_points * result.losses +
  result.draw_points * result.draws

/-- Theorem: Henry's tournament result yields 44 points -/
theorem henry_tournament_points :
  let henry_result : TournamentResult := {
    win_points := 5,
    loss_points := 2,
    draw_points := 3,
    wins := 2,
    losses := 2,
    draws := 10
  }
  calculate_points henry_result = 44 := by sorry

end NUMINAMATH_CALUDE_henry_tournament_points_l2409_240977


namespace NUMINAMATH_CALUDE_workers_count_l2409_240999

/-- Given a work that can be completed by some workers in 35 days,
    and adding 10 workers reduces the completion time by 10 days,
    prove that the original number of workers is 25. -/
theorem workers_count (work : ℕ) : ∃ (workers : ℕ), 
  (workers * 35 = (workers + 10) * 25) ∧ 
  workers = 25 := by
  sorry

end NUMINAMATH_CALUDE_workers_count_l2409_240999


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l2409_240926

theorem function_satisfies_conditions (x : ℝ) :
  1 < x → x < 2 → -2 < x - 3 ∧ x - 3 < -1 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l2409_240926


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l2409_240905

/-- Given an arithmetic progression where the sum of n terms is 2n + 3n^2 for every n,
    prove that the r-th term is 6r - 1. -/
theorem arithmetic_progression_rth_term (r : ℕ) :
  let S : ℕ → ℕ := λ n => 2*n + 3*n^2
  let a : ℕ → ℤ := λ k => S k - S (k-1)
  a r = 6*r - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l2409_240905


namespace NUMINAMATH_CALUDE_leak_empty_time_correct_l2409_240962

/-- Represents a tank with a leak and an inlet pipe -/
structure Tank where
  capacity : ℝ
  inletRate : ℝ
  emptyTimeWithInlet : ℝ

/-- Calculates the time it takes for the leak alone to empty the tank -/
def leakEmptyTime (t : Tank) : ℝ :=
  -- Definition to be proved
  9

/-- Theorem stating the correct leak empty time for the given tank -/
theorem leak_empty_time_correct (t : Tank) 
  (h1 : t.capacity = 12960)
  (h2 : t.inletRate = 6 * 60)  -- 6 litres per minute converted to per hour
  (h3 : t.emptyTimeWithInlet = 12) : 
  leakEmptyTime t = 9 := by
  sorry

#check leak_empty_time_correct

end NUMINAMATH_CALUDE_leak_empty_time_correct_l2409_240962


namespace NUMINAMATH_CALUDE_average_candies_per_packet_l2409_240951

def candy_counts : List Nat := [5, 7, 9, 11, 13, 15]
def num_packets : Nat := 6

theorem average_candies_per_packet :
  (candy_counts.sum / num_packets : ℚ) = 10 := by sorry

end NUMINAMATH_CALUDE_average_candies_per_packet_l2409_240951


namespace NUMINAMATH_CALUDE_modified_binomial_coefficient_integrality_l2409_240934

theorem modified_binomial_coefficient_integrality 
  (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) : 
  ∃ m : ℤ, (n - 3 * k - 2 : ℤ) * (n.factorial) = 
    (k + 2 : ℤ) * m * (k.factorial) * ((n - k).factorial) := by
  sorry

end NUMINAMATH_CALUDE_modified_binomial_coefficient_integrality_l2409_240934


namespace NUMINAMATH_CALUDE_sumata_vacation_miles_l2409_240978

/-- The total miles driven during a vacation -/
def total_miles (days : ℝ) (miles_per_day : ℝ) : ℝ :=
  days * miles_per_day

/-- Theorem: The Sumata family drove 1250 miles during their 5.0-day vacation -/
theorem sumata_vacation_miles :
  total_miles 5.0 250 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_sumata_vacation_miles_l2409_240978


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l2409_240961

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

theorem largest_digit_divisible_by_6 : 
  ∀ N : ℕ, N ≤ 9 → 
    (is_divisible_by_6 (71820 + N) → N ≤ 6) ∧ 
    (is_divisible_by_6 (71826)) := by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l2409_240961


namespace NUMINAMATH_CALUDE_circle_radius_three_inches_l2409_240924

theorem circle_radius_three_inches (r : ℝ) (h : 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2)) : r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_three_inches_l2409_240924


namespace NUMINAMATH_CALUDE_polynomial_positive_root_l2409_240907

/-- The polynomial has at least one positive real root if and only if q ≥ 3/2 -/
theorem polynomial_positive_root (q : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x^6 + 3*q*x^4 + 3*x^4 + 3*q*x^2 + x^2 + 3*q + 1 = 0) ↔ q ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_positive_root_l2409_240907


namespace NUMINAMATH_CALUDE_special_polynomial_f_one_l2409_240902

/-- A polynomial function satisfying a specific equation -/
def SpecialPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x : ℝ, f x = a * x^2 + b * x + c) ∧
    (∀ x : ℝ, x ≠ 0 → f (x - 1) + f x + f (x + 1) = (f x)^2 / (2027 * x))

/-- The theorem stating that for a special polynomial, f(1) must equal 6081 -/
theorem special_polynomial_f_one (f : ℝ → ℝ) (hf : SpecialPolynomial f) : f 1 = 6081 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_f_one_l2409_240902


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2409_240949

theorem complex_absolute_value (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2409_240949


namespace NUMINAMATH_CALUDE_garden_fencing_cost_l2409_240966

/-- The cost of fencing a rectangular garden -/
theorem garden_fencing_cost
  (garden_width : ℝ)
  (playground_length playground_width : ℝ)
  (fencing_price : ℝ)
  (h1 : garden_width = 12)
  (h2 : playground_length = 16)
  (h3 : playground_width = 12)
  (h4 : fencing_price = 15)
  (h5 : garden_width * (playground_length * playground_width / garden_width) = playground_length * playground_width) :
  2 * (garden_width + (playground_length * playground_width / garden_width)) * fencing_price = 840 :=
by sorry

end NUMINAMATH_CALUDE_garden_fencing_cost_l2409_240966


namespace NUMINAMATH_CALUDE_salary_change_percentage_l2409_240914

theorem salary_change_percentage (x : ℝ) : 
  (1 - (x / 100)^2) = 0.91 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l2409_240914


namespace NUMINAMATH_CALUDE_candles_per_small_box_l2409_240900

theorem candles_per_small_box 
  (small_boxes_per_big_box : Nat) 
  (num_big_boxes : Nat) 
  (total_candles : Nat) :
  small_boxes_per_big_box = 4 →
  num_big_boxes = 50 →
  total_candles = 8000 →
  (total_candles / (small_boxes_per_big_box * num_big_boxes) : Nat) = 40 := by
  sorry

end NUMINAMATH_CALUDE_candles_per_small_box_l2409_240900


namespace NUMINAMATH_CALUDE_sin_2010th_derivative_l2409_240947

open Real

-- Define the recursive function for the nth derivative of sin x
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => sin
  | n + 1 => deriv (f n)

-- State the theorem
theorem sin_2010th_derivative :
  ∀ x, f 2010 x = -sin x :=
by
  sorry

end NUMINAMATH_CALUDE_sin_2010th_derivative_l2409_240947


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l2409_240920

theorem sqrt_x_minus_3_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l2409_240920


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l2409_240910

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 17

/-- The number of chocolate bars in each small box -/
def choc_per_small_box : ℕ := 26

/-- The total number of chocolate bars in the large box -/
def total_chocolate_bars : ℕ := num_small_boxes * choc_per_small_box

theorem chocolate_bar_count : total_chocolate_bars = 442 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l2409_240910


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2409_240906

theorem polynomial_multiplication (x : ℝ) :
  (3*x - 2) * (6*x^12 + 3*x^11 + 5*x^9 + x^8 + 7*x^7) =
  18*x^13 - 3*x^12 + 15*x^10 - 7*x^9 + 19*x^8 - 14*x^7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2409_240906


namespace NUMINAMATH_CALUDE_shaded_area_in_rectangle_with_circles_l2409_240915

/-- Given a rectangle containing two tangent circles, calculate the area not occupied by the circles. -/
theorem shaded_area_in_rectangle_with_circles 
  (rectangle_length : ℝ) 
  (rectangle_height : ℝ)
  (small_circle_radius : ℝ)
  (large_circle_radius : ℝ) :
  rectangle_length = 20 →
  rectangle_height = 10 →
  small_circle_radius = 3 →
  large_circle_radius = 5 →
  ∃ (shaded_area : ℝ), 
    shaded_area = rectangle_length * rectangle_height - π * (small_circle_radius^2 + large_circle_radius^2) ∧
    shaded_area = 200 - 34 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_in_rectangle_with_circles_l2409_240915


namespace NUMINAMATH_CALUDE_smallest_n_sum_all_digits_same_l2409_240953

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Checks if a number has all digits the same -/
def all_digits_same (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ n = d * 111

/-- The smallest n such that sum_first_n(n) is a three-digit number with all digits the same -/
theorem smallest_n_sum_all_digits_same :
  ∃ n : ℕ, 
    (∀ m : ℕ, m < n → ¬(all_digits_same (sum_first_n m))) ∧
    (all_digits_same (sum_first_n n)) ∧
    n = 36 := by sorry

end NUMINAMATH_CALUDE_smallest_n_sum_all_digits_same_l2409_240953


namespace NUMINAMATH_CALUDE_train_speed_l2409_240995

/-- The speed of a train given the time to pass a pole and a stationary train -/
theorem train_speed (t_pole : ℝ) (t_stationary : ℝ) (l_stationary : ℝ) :
  t_pole = 8 →
  t_stationary = 18 →
  l_stationary = 400 →
  ∃ (speed : ℝ), speed = 144 ∧ speed * 1000 / 3600 * t_pole = speed * 1000 / 3600 * t_stationary - l_stationary :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l2409_240995


namespace NUMINAMATH_CALUDE_solve_necklace_cost_l2409_240971

def necklace_cost_problem (necklace_cost book_cost total_cost spending_limit overspend : ℚ) : Prop :=
  book_cost = necklace_cost + 5 ∧
  spending_limit = 70 ∧
  overspend = 3 ∧
  total_cost = necklace_cost + book_cost ∧
  total_cost = spending_limit + overspend ∧
  necklace_cost = 34

theorem solve_necklace_cost :
  ∃ (necklace_cost book_cost total_cost spending_limit overspend : ℚ),
    necklace_cost_problem necklace_cost book_cost total_cost spending_limit overspend :=
by sorry

end NUMINAMATH_CALUDE_solve_necklace_cost_l2409_240971


namespace NUMINAMATH_CALUDE_circumference_difference_l2409_240997

/-- Given two circles A and B with areas and π as specified, 
    prove that the difference between their circumferences is 6.2 cm -/
theorem circumference_difference (π : ℝ) (area_A area_B : ℝ) :
  π = 3.1 →
  area_A = 198.4 →
  area_B = 251.1 →
  let radius_A := Real.sqrt (area_A / π)
  let radius_B := Real.sqrt (area_B / π)
  let circumference_A := 2 * π * radius_A
  let circumference_B := 2 * π * radius_B
  circumference_B - circumference_A = 6.2 := by
  sorry

end NUMINAMATH_CALUDE_circumference_difference_l2409_240997


namespace NUMINAMATH_CALUDE_problem_i4_1_l2409_240996

theorem problem_i4_1 (f : ℝ → ℝ) :
  (∀ x, f x = (x^2 + x - 2)^2002 + 3) →
  f ((Real.sqrt 5 / 2) - 1/2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_i4_1_l2409_240996


namespace NUMINAMATH_CALUDE_zacks_marbles_l2409_240980

/-- Zack's marble distribution problem -/
theorem zacks_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ) 
  (h1 : initial_marbles = 65)
  (h2 : friends = 3)
  (h3 : marbles_per_friend = 20)
  (h4 : initial_marbles % friends ≠ 0) : 
  initial_marbles - friends * marbles_per_friend = 5 :=
by sorry

end NUMINAMATH_CALUDE_zacks_marbles_l2409_240980


namespace NUMINAMATH_CALUDE_harriets_siblings_product_l2409_240993

/-- Given a family where Harry has 4 sisters and 6 brothers, and Harriet is one of Harry's sisters,
    this theorem proves that the product of the number of Harriet's sisters and brothers is 24. -/
theorem harriets_siblings_product (harry_sisters : ℕ) (harry_brothers : ℕ) 
  (harriet_sisters : ℕ) (harriet_brothers : ℕ) :
  harry_sisters = 4 →
  harry_brothers = 6 →
  harriet_sisters = harry_sisters - 1 →
  harriet_brothers = harry_brothers →
  harriet_sisters * harriet_brothers = 24 :=
by sorry

end NUMINAMATH_CALUDE_harriets_siblings_product_l2409_240993


namespace NUMINAMATH_CALUDE_alton_daily_earnings_l2409_240904

/-- Calculates daily earnings given weekly rent, weekly profit, and number of workdays --/
def daily_earnings (weekly_rent : ℚ) (weekly_profit : ℚ) (workdays : ℕ) : ℚ :=
  (weekly_rent + weekly_profit) / workdays

/-- Proves that given the specified conditions, daily earnings are $11.20 --/
theorem alton_daily_earnings :
  let weekly_rent : ℚ := 20
  let weekly_profit : ℚ := 36
  let workdays : ℕ := 5
  daily_earnings weekly_rent weekly_profit workdays = 11.2 := by
sorry

end NUMINAMATH_CALUDE_alton_daily_earnings_l2409_240904


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l2409_240964

/-- The cost per foot of building a fence around a square plot -/
theorem fence_cost_per_foot (area : ℝ) (total_cost : ℝ) : area = 289 → total_cost = 4080 → (total_cost / (4 * Real.sqrt area)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l2409_240964


namespace NUMINAMATH_CALUDE_product_of_binomials_l2409_240968

theorem product_of_binomials (a : ℝ) : (a + 2) * (2 * a - 3) = 2 * a^2 + a - 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binomials_l2409_240968


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_root_l2409_240919

/-- Given a, b, c are side lengths of a triangle, 
    prove that ∛((a²+bc)(b²+ca)(c²+ab)) > (a²+b²+c²)/2 -/
theorem triangle_inequality_cube_root (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (((a^2 + b*c) * (b^2 + c*a) * (c^2 + a*b))^(1/3) : ℝ) > (a^2 + b^2 + c^2) / 2 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_root_l2409_240919


namespace NUMINAMATH_CALUDE_toy_selling_price_l2409_240944

/-- Calculates the total selling price of toys given the number of toys sold,
    the cost price per toy, and the number of toys whose cost price equals the total gain. -/
def total_selling_price (num_toys : ℕ) (cost_price : ℕ) (gain_toys : ℕ) : ℕ :=
  num_toys * cost_price + gain_toys * cost_price

/-- Proves that the total selling price of 18 toys is 23100,
    given a cost price of 1100 per toy and a gain equal to the cost of 3 toys. -/
theorem toy_selling_price :
  total_selling_price 18 1100 3 = 23100 := by
  sorry

end NUMINAMATH_CALUDE_toy_selling_price_l2409_240944


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l2409_240967

-- Define the point
def point : ℝ × ℝ := (8, -15)

-- Theorem statement
theorem distance_from_origin_to_point :
  Real.sqrt ((point.1 - 0)^2 + (point.2 - 0)^2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l2409_240967


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l2409_240988

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- Theorem: If the ratio S_{4n}/S_n is constant for all positive n,
    then the first term of the sequence is 5/2 -/
theorem arithmetic_sequence_constant_ratio
  (h : ∃ (c : ℚ), ∀ (n : ℕ), n > 0 → S a (4*n) / S a n = c) :
  a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_ratio_l2409_240988


namespace NUMINAMATH_CALUDE_f_composition_equal_range_l2409_240946

/-- The function f(x) = x^2 + ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

/-- The theorem stating the range of a -/
theorem f_composition_equal_range (a : ℝ) :
  ({y | ∃ x, y = f a (f a x)} = {y | ∃ x, y = f a x}) →
  (a ≥ 4 ∨ a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_f_composition_equal_range_l2409_240946


namespace NUMINAMATH_CALUDE_probability_three_white_balls_l2409_240945

/-- The probability of drawing 3 white balls from a box containing 8 white balls and 7 black balls -/
theorem probability_three_white_balls (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (h1 : total_balls = white_balls + black_balls)
  (h2 : white_balls = 8)
  (h3 : black_balls = 7)
  (h4 : total_balls ≥ 3) :
  (Nat.choose white_balls 3 : ℚ) / (Nat.choose total_balls 3) = 8 / 65 := by
sorry

end NUMINAMATH_CALUDE_probability_three_white_balls_l2409_240945


namespace NUMINAMATH_CALUDE_first_load_pieces_l2409_240928

theorem first_load_pieces (total : ℕ) (equal_loads : ℕ) (pieces_per_load : ℕ)
  (h1 : total = 36)
  (h2 : equal_loads = 2)
  (h3 : pieces_per_load = 9)
  : total - (equal_loads * pieces_per_load) = 18 :=
by sorry

end NUMINAMATH_CALUDE_first_load_pieces_l2409_240928


namespace NUMINAMATH_CALUDE_mary_biking_time_l2409_240939

def total_away_time : ℕ := 570 -- 9.5 hours in minutes
def class_time : ℕ := 45
def num_classes : ℕ := 7
def lunch_time : ℕ := 40
def additional_time : ℕ := 105 -- 1 hour 45 minutes in minutes

def total_school_time : ℕ := class_time * num_classes + lunch_time + additional_time

theorem mary_biking_time :
  total_away_time - total_school_time = 110 :=
sorry

end NUMINAMATH_CALUDE_mary_biking_time_l2409_240939


namespace NUMINAMATH_CALUDE_total_attendance_l2409_240998

def wedding_reception (bride_couples groom_couples friends : ℕ) : ℕ :=
  2 * (bride_couples + groom_couples) + friends

theorem total_attendance : wedding_reception 20 20 100 = 180 := by
  sorry

end NUMINAMATH_CALUDE_total_attendance_l2409_240998


namespace NUMINAMATH_CALUDE_number_equation_l2409_240950

theorem number_equation (x : ℝ) : x - (105 / 21) = 5995 ↔ x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2409_240950


namespace NUMINAMATH_CALUDE_coordinates_of_P_wrt_x_axis_l2409_240952

/-- Given a point P in the Cartesian coordinate system, this function
    returns its coordinates with respect to the x-axis. -/
def coordinates_wrt_x_axis (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

/-- Theorem stating that the coordinates of P(-2, 3) with respect to the x-axis are (-2, -3). -/
theorem coordinates_of_P_wrt_x_axis :
  coordinates_wrt_x_axis (-2, 3) = (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_P_wrt_x_axis_l2409_240952


namespace NUMINAMATH_CALUDE_base7_246_equals_base10_132_l2409_240942

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- Proves that 246 in base 7 is equal to 132 in base 10 -/
theorem base7_246_equals_base10_132 : base7ToBase10 2 4 6 = 132 := by
  sorry

end NUMINAMATH_CALUDE_base7_246_equals_base10_132_l2409_240942


namespace NUMINAMATH_CALUDE_black_cows_exceeding_half_l2409_240991

theorem black_cows_exceeding_half (total_cows : ℕ) (non_black_cows : ℕ) : 
  total_cows = 18 → non_black_cows = 4 → 
  (total_cows - non_black_cows) - (total_cows / 2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_black_cows_exceeding_half_l2409_240991


namespace NUMINAMATH_CALUDE_max_sum_with_constraint_l2409_240908

theorem max_sum_with_constraint (a b c d e : ℕ) 
  (h : 625 * a + 250 * b + 100 * c + 40 * d + 16 * e = 15^3) :
  a + b + c + d + e ≤ 153 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_constraint_l2409_240908


namespace NUMINAMATH_CALUDE_march_text_messages_l2409_240955

/-- Represents the number of text messages sent in the nth month -/
def T (n : ℕ) : ℕ := n^3 - n^2 + n

/-- Theorem stating that the number of text messages in the 5th month (March) is 105 -/
theorem march_text_messages : T 5 = 105 := by
  sorry

end NUMINAMATH_CALUDE_march_text_messages_l2409_240955


namespace NUMINAMATH_CALUDE_fraction_simplification_l2409_240916

theorem fraction_simplification :
  1 / (1 / ((1/2)^2) + 1 / ((1/2)^3) + 1 / ((1/2)^4) + 1 / ((1/2)^5)) = 1 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2409_240916


namespace NUMINAMATH_CALUDE_min_value_theorem_l2409_240969

/-- Triangle ABC with area 2 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (area : ℝ)
  (area_eq : area = 2)

/-- Function f mapping a point to areas of subtriangles -/
def f (T : Triangle) (P : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating the minimum value of 1/x + 4/y -/
theorem min_value_theorem (T : Triangle) :
  ∀ P : ℝ × ℝ, 
  ∀ x y : ℝ,
  f T P = (1, x, y) →
  (∀ a b : ℝ, f T (a, b) = (1, x, y) → 1/x + 4/y ≥ 9) ∧ 
  (∃ a b : ℝ, f T (a, b) = (1, x, y) ∧ 1/x + 4/y = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2409_240969


namespace NUMINAMATH_CALUDE_mollys_current_age_l2409_240931

/-- Represents the ages of Sandy and Molly -/
structure Ages where
  sandy : ℕ
  molly : ℕ

/-- The ratio of Sandy's age to Molly's age is 4:3 -/
def age_ratio (ages : Ages) : Prop :=
  4 * ages.molly = 3 * ages.sandy

/-- Sandy will be 42 years old in 6 years -/
def sandy_future_age (ages : Ages) : Prop :=
  ages.sandy + 6 = 42

theorem mollys_current_age (ages : Ages) :
  age_ratio ages → sandy_future_age ages → ages.molly = 27 := by
  sorry

end NUMINAMATH_CALUDE_mollys_current_age_l2409_240931


namespace NUMINAMATH_CALUDE_student_sample_size_l2409_240982

theorem student_sample_size :
  ∀ (T : ℝ) (freshmen sophomores juniors seniors : ℝ),
    -- All students are either freshmen, sophomores, juniors, or seniors
    T = freshmen + sophomores + juniors + seniors →
    -- 27% are juniors
    juniors = 0.27 * T →
    -- 75% are not sophomores (which means 25% are sophomores)
    sophomores = 0.25 * T →
    -- There are 160 seniors
    seniors = 160 →
    -- There are 24 more freshmen than sophomores
    freshmen = sophomores + 24 →
    -- Prove that the total number of students is 800
    T = 800 := by
  sorry

end NUMINAMATH_CALUDE_student_sample_size_l2409_240982


namespace NUMINAMATH_CALUDE_sqrt_21_minus_1_bounds_l2409_240983

theorem sqrt_21_minus_1_bounds : 3 < Real.sqrt 21 - 1 ∧ Real.sqrt 21 - 1 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_21_minus_1_bounds_l2409_240983


namespace NUMINAMATH_CALUDE_meaningful_expression_l2409_240994

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 2)) ↔ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2409_240994


namespace NUMINAMATH_CALUDE_six_power_plus_one_same_digits_l2409_240917

def has_same_digits (m : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ ∀ k : ℕ, (m / 10^k) % 10 = d

theorem six_power_plus_one_same_digits :
  {n : ℕ | n > 0 ∧ has_same_digits (6^n + 1)} = {1, 5} := by sorry

end NUMINAMATH_CALUDE_six_power_plus_one_same_digits_l2409_240917


namespace NUMINAMATH_CALUDE_b_completes_in_20_days_l2409_240909

/-- The number of days it takes for worker A to complete the work alone -/
def days_a : ℝ := 15

/-- The number of days A and B work together -/
def days_together : ℝ := 7

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.18333333333333335

/-- The number of days it takes for worker B to complete the work alone -/
def days_b : ℝ := 20

/-- Theorem stating that given the conditions, B can complete the work in 20 days -/
theorem b_completes_in_20_days :
  (days_together * (1 / days_a + 1 / days_b) = 1 - work_left) →
  days_b = 20 := by
  sorry

end NUMINAMATH_CALUDE_b_completes_in_20_days_l2409_240909


namespace NUMINAMATH_CALUDE_tank_height_is_16_l2409_240958

/-- The height of a cylindrical water tank with specific conditions -/
def tank_height : ℝ := 16

/-- The base radius of the cylindrical water tank -/
def base_radius : ℝ := 3

/-- Theorem stating that the height of the tank is 16 cm under given conditions -/
theorem tank_height_is_16 :
  tank_height = 16 ∧
  base_radius = 3 ∧
  (π * base_radius^2 * (tank_height / 2) = 2 * (4/3) * π * base_radius^3) :=
by sorry

end NUMINAMATH_CALUDE_tank_height_is_16_l2409_240958


namespace NUMINAMATH_CALUDE_gcd_factorial_plus_two_l2409_240972

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_plus_two : 
  Nat.gcd (factorial 6 + 2) (factorial 8 + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_plus_two_l2409_240972


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2409_240929

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 + a 3 = 10 →
  a 4 + a 6 = 5/4 →
  ∃ (q : ℝ), ∀ n : ℕ, a n = 2^(4-n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2409_240929


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2409_240901

theorem inequality_system_solution :
  let S : Set ℤ := {x | (3 * x - 5 ≥ 2 * (x - 2)) ∧ (x / 2 ≥ x - 2)}
  S = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2409_240901


namespace NUMINAMATH_CALUDE_sandy_comic_books_l2409_240979

theorem sandy_comic_books (initial : ℕ) : 
  (initial / 2 + 6 = 13) → initial = 14 := by
  sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l2409_240979


namespace NUMINAMATH_CALUDE_f_range_l2409_240913

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + 2

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.Icc (-2 : ℝ) 2, ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x = y ∧
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ∈ Set.Icc (-2 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_f_range_l2409_240913


namespace NUMINAMATH_CALUDE_emily_egg_collection_l2409_240937

/-- The total number of eggs Emily collected -/
def total_eggs : ℕ :=
  let set_a_eggs := 200 * 36 + 250 * 24
  let set_b_eggs := 375 * 42 - 80
  let set_c_eggs := (560 / 2) * 50 + (560 / 2) * 32
  set_a_eggs + set_b_eggs + set_c_eggs

/-- Theorem stating that Emily collected 51830 eggs in total -/
theorem emily_egg_collection : total_eggs = 51830 := by
  sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l2409_240937


namespace NUMINAMATH_CALUDE_product_equals_result_l2409_240973

theorem product_equals_result : 582964 * 99999 = 58295817036 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_result_l2409_240973


namespace NUMINAMATH_CALUDE_hypotenuse_ratio_l2409_240959

/-- Represents a right-angled triangle with a 30° angle -/
structure Triangle30 where
  hypotenuse : ℝ
  shared_side : ℝ
  hypotenuse_gt_shared : hypotenuse > shared_side

/-- The three triangles in our problem -/
def three_triangles (a b c : Triangle30) : Prop :=
  a.shared_side = b.shared_side ∧ 
  b.shared_side = c.shared_side ∧ 
  a.hypotenuse ≠ b.hypotenuse ∧ 
  b.hypotenuse ≠ c.hypotenuse ∧ 
  a.hypotenuse ≠ c.hypotenuse

theorem hypotenuse_ratio (a b c : Triangle30) :
  three_triangles a b c →
  (∃ (k : ℝ), k > 0 ∧ 
    (max a.hypotenuse (max b.hypotenuse c.hypotenuse) = 2 * k) ∧
    (max (min a.hypotenuse b.hypotenuse) c.hypotenuse = 2 * k / Real.sqrt 3) ∧
    (min a.hypotenuse (min b.hypotenuse c.hypotenuse) = k)) :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_ratio_l2409_240959


namespace NUMINAMATH_CALUDE_female_students_count_l2409_240941

theorem female_students_count (total_average : ℝ) (male_count : ℕ) (male_average : ℝ) (female_average : ℝ) :
  total_average = 90 →
  male_count = 8 →
  male_average = 84 →
  female_average = 92 →
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 24 :=
by sorry

end NUMINAMATH_CALUDE_female_students_count_l2409_240941


namespace NUMINAMATH_CALUDE_coprime_27x_plus_4_and_18x_plus_3_l2409_240965

theorem coprime_27x_plus_4_and_18x_plus_3 (x : ℕ) : Nat.gcd (27 * x + 4) (18 * x + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_coprime_27x_plus_4_and_18x_plus_3_l2409_240965


namespace NUMINAMATH_CALUDE_paint_cans_theorem_l2409_240984

/-- The number of rooms that can be painted with one can of paint -/
def rooms_per_can : ℚ :=
  (40 - 32) / 4

/-- The number of cans needed to paint 32 rooms -/
def cans_for_32_rooms : ℚ :=
  32 / rooms_per_can

theorem paint_cans_theorem :
  cans_for_32_rooms = 16 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_theorem_l2409_240984


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_equals_four_l2409_240987

theorem sum_of_a_and_b_equals_four (a b : ℝ) (h : b + (a - 2) * Complex.I = 1 + Complex.I) : a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_equals_four_l2409_240987


namespace NUMINAMATH_CALUDE_recreation_spending_percentage_l2409_240975

theorem recreation_spending_percentage
  (last_week_wages : ℝ)
  (last_week_recreation_percent : ℝ)
  (wage_decrease_percent : ℝ)
  (this_week_recreation_increase : ℝ)
  (h1 : last_week_recreation_percent = 10)
  (h2 : wage_decrease_percent = 10)
  (h3 : this_week_recreation_increase = 360) :
  let this_week_wages := last_week_wages * (1 - wage_decrease_percent / 100)
  let last_week_recreation := last_week_wages * (last_week_recreation_percent / 100)
  let this_week_recreation := last_week_recreation * (this_week_recreation_increase / 100)
  this_week_recreation / this_week_wages * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_recreation_spending_percentage_l2409_240975


namespace NUMINAMATH_CALUDE_hcf_problem_l2409_240948

theorem hcf_problem (a b H : ℕ+) : 
  (Nat.gcd a b = H) →
  (Nat.lcm a b = H * 13 * 14) →
  (max a b = 322) →
  (H = 14) := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l2409_240948


namespace NUMINAMATH_CALUDE_ratio_sum_to_last_l2409_240936

theorem ratio_sum_to_last {a b c : ℝ} (h : a / c = 3 / 7 ∧ b / c = 4 / 7) :
  (a + b + c) / c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_to_last_l2409_240936


namespace NUMINAMATH_CALUDE_variance_scaling_l2409_240989

-- Define a set of data points
def DataSet : Type := List ℝ

-- Define the variance function
noncomputable def variance (data : DataSet) : ℝ := sorry

-- Define a function to multiply each data point by a scalar
def scaleData (data : DataSet) (scalar : ℝ) : DataSet :=
  data.map (· * scalar)

-- Theorem statement
theorem variance_scaling (data : DataSet) (s : ℝ) :
  variance data = s^2 → variance (scaleData data 2) = 4 * s^2 := by
  sorry

end NUMINAMATH_CALUDE_variance_scaling_l2409_240989


namespace NUMINAMATH_CALUDE_total_distance_traveled_l2409_240903

def trip_duration : ℕ := 12
def speed1 : ℕ := 70
def time1 : ℕ := 3
def speed2 : ℕ := 80
def time2 : ℕ := 4
def speed3 : ℕ := 65
def time3 : ℕ := 3
def speed4 : ℕ := 90
def time4 : ℕ := 2

theorem total_distance_traveled :
  speed1 * time1 + speed2 * time2 + speed3 * time3 + speed4 * time4 = 905 :=
by
  sorry

#check total_distance_traveled

end NUMINAMATH_CALUDE_total_distance_traveled_l2409_240903


namespace NUMINAMATH_CALUDE_equation_solution_l2409_240921

theorem equation_solution : ∃ x : ℝ, 3^(x - 1) = (1 : ℝ) / 9 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2409_240921


namespace NUMINAMATH_CALUDE_determine_fourth_player_wins_l2409_240976

/-- Represents a player in the chess tournament -/
structure Player where
  wins : Nat
  losses : Nat

/-- Represents a chess tournament -/
structure ChessTournament where
  players : Fin 4 → Player
  total_games : Nat

/-- The theorem states that given the wins and losses of three players in a four-player
    round-robin chess tournament, we can determine the number of wins for the fourth player. -/
theorem determine_fourth_player_wins (t : ChessTournament) 
  (h1 : t.players 0 = { wins := 5, losses := 3 })
  (h2 : t.players 1 = { wins := 4, losses := 4 })
  (h3 : t.players 2 = { wins := 2, losses := 6 })
  (h_total : t.total_games = 16)
  (h_balance : ∀ i, (t.players i).wins + (t.players i).losses = 8) :
  (t.players 3).wins = 5 := by
  sorry

end NUMINAMATH_CALUDE_determine_fourth_player_wins_l2409_240976


namespace NUMINAMATH_CALUDE_integral_proof_l2409_240985

open Real

theorem integral_proof (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ -2) :
  deriv (fun x => log (abs (x + 1)) - 1 / (2 * (x + 2)^2)) x =
  (x^3 + 6*x^2 + 13*x + 9) / ((x + 1) * (x + 2)^3) := by
sorry

end NUMINAMATH_CALUDE_integral_proof_l2409_240985


namespace NUMINAMATH_CALUDE_color_film_fraction_l2409_240932

/-- Given a film festival selection process, prove the fraction of color films in the selection. -/
theorem color_film_fraction (x y : ℚ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw : ℚ := 40 * x
  let total_color : ℚ := 10 * y
  let selected_bw : ℚ := (y / x) * (total_bw / 100)
  let selected_color : ℚ := total_color
  let total_selected : ℚ := selected_bw + selected_color
  (selected_color / total_selected) = 25 / 26 := by
  sorry

end NUMINAMATH_CALUDE_color_film_fraction_l2409_240932
