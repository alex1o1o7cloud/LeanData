import Mathlib

namespace NUMINAMATH_CALUDE_apples_left_after_pie_l2816_281602

def apples_left (initial : ℝ) (contribution : ℝ) (pie_requirement : ℝ) : ℝ :=
  initial + contribution - pie_requirement

theorem apples_left_after_pie : apples_left 10 5 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_after_pie_l2816_281602


namespace NUMINAMATH_CALUDE_max_roses_is_316_l2816_281627

/-- The price of an individual rose in cents -/
def individual_price : ℕ := 730

/-- The price of one dozen roses in cents -/
def dozen_price : ℕ := 3600

/-- The price of two dozen roses in cents -/
def two_dozen_price : ℕ := 5000

/-- The total budget in cents -/
def budget : ℕ := 68000

/-- The function to calculate the maximum number of roses that can be purchased -/
def max_roses : ℕ :=
  let two_dozen_sets := budget / two_dozen_price
  let remaining := budget % two_dozen_price
  let individual_roses := remaining / individual_price
  two_dozen_sets * 24 + individual_roses

/-- Theorem stating that the maximum number of roses that can be purchased is 316 -/
theorem max_roses_is_316 : max_roses = 316 := by
  sorry

end NUMINAMATH_CALUDE_max_roses_is_316_l2816_281627


namespace NUMINAMATH_CALUDE_smallest_number_l2816_281660

def base_2_to_10 (n : ℕ) : ℕ := n

def base_4_to_10 (n : ℕ) : ℕ := n

def base_8_to_10 (n : ℕ) : ℕ := n

theorem smallest_number :
  let a := base_4_to_10 321
  let b := 58
  let c := base_2_to_10 111000
  let d := base_8_to_10 73
  c < a ∧ c < b ∧ c < d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l2816_281660


namespace NUMINAMATH_CALUDE_base6_divisibility_by_11_l2816_281649

/-- Converts a base-6 number of the form 2dd5₆ to base 10 --/
def base6ToBase10 (d : Nat) : Nat :=
  2 * 6^3 + d * 6^2 + d * 6^1 + 5

/-- Checks if a number is divisible by 11 --/
def isDivisibleBy11 (n : Nat) : Prop :=
  n % 11 = 0

/-- Represents a base-6 digit --/
def isBase6Digit (d : Nat) : Prop :=
  d < 6

theorem base6_divisibility_by_11 :
  ∃ (d : Nat), isBase6Digit d ∧ isDivisibleBy11 (base6ToBase10 d) ↔ d = 4 := by
  sorry

end NUMINAMATH_CALUDE_base6_divisibility_by_11_l2816_281649


namespace NUMINAMATH_CALUDE_equation_solution_l2816_281622

theorem equation_solution :
  ∃ x : ℝ, x ≠ 0 ∧ (2 / x + 3 * ((4 / x) / (8 / x)) = 1.2) ∧ x = -20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2816_281622


namespace NUMINAMATH_CALUDE_chord_intersection_probability_l2816_281609

/-- A circle with n evenly spaced points -/
structure CirclePoints where
  n : ℕ
  h : n ≥ 4

/-- Four distinct points selected from the circle -/
structure FourPoints (c : CirclePoints) where
  A : Fin c.n
  B : Fin c.n
  C : Fin c.n
  D : Fin c.n
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

/-- The probability that chord AB intersects chord CD -/
def intersectionProbability (c : CirclePoints) : ℚ :=
  1 / 3

/-- Theorem: The probability of chord AB intersecting chord CD is 1/3 -/
theorem chord_intersection_probability (c : CirclePoints) :
  intersectionProbability c = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_chord_intersection_probability_l2816_281609


namespace NUMINAMATH_CALUDE_stream_speed_l2816_281694

/-- Given Julie's rowing distances and times, prove that the speed of the stream is 5 km/h -/
theorem stream_speed (v_j v_s : ℝ) 
  (h1 : 32 / (v_j - v_s) = 4)  -- Upstream equation
  (h2 : 72 / (v_j + v_s) = 4)  -- Downstream equation
  : v_s = 5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l2816_281694


namespace NUMINAMATH_CALUDE_angle_460_in_second_quadrant_l2816_281671

/-- An angle is in the second quadrant if it's between 90° and 180° in its standard position -/
def in_second_quadrant (angle : ℝ) : Prop :=
  let standard_angle := angle % 360
  90 < standard_angle ∧ standard_angle ≤ 180

/-- 460° is in the second quadrant -/
theorem angle_460_in_second_quadrant : in_second_quadrant 460 := by
  sorry

end NUMINAMATH_CALUDE_angle_460_in_second_quadrant_l2816_281671


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2816_281604

/-- Given two vectors a and b in ℝ³, where a = (2, -3, 1) and b = (4, -6, x),
    if a is perpendicular to b, then x = -26. -/
theorem perpendicular_vectors_x_value :
  let a : Fin 3 → ℝ := ![2, -3, 1]
  let b : Fin 3 → ℝ := ![4, -6, x]
  (∀ i : Fin 3, a i * b i = 0) → x = -26 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2816_281604


namespace NUMINAMATH_CALUDE_chairs_subset_count_l2816_281641

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- The minimum number of adjacent chairs required in a subset -/
def k : ℕ := 4

/-- The number of subsets of n chairs arranged in a circle that contain at least k adjacent chairs -/
def subsets_with_adjacent_chairs (n k : ℕ) : ℕ := sorry

theorem chairs_subset_count : subsets_with_adjacent_chairs n k = 1610 := by sorry

end NUMINAMATH_CALUDE_chairs_subset_count_l2816_281641


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2816_281617

theorem trigonometric_identity (α : Real) : 
  0 < α ∧ α < π/2 →
  Real.sin (5*π/12 + 2*α) = -3/5 →
  Real.sin (π/12 + α) * Real.sin (5*π/12 - α) = Real.sqrt 2 / 20 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2816_281617


namespace NUMINAMATH_CALUDE_market_fruit_count_l2816_281626

theorem market_fruit_count (apples oranges bananas : ℕ) 
  (h1 : apples = oranges + 27)
  (h2 : oranges = bananas + 11)
  (h3 : apples + oranges + bananas = 301) :
  apples = 122 := by
sorry

end NUMINAMATH_CALUDE_market_fruit_count_l2816_281626


namespace NUMINAMATH_CALUDE_total_hair_product_usage_l2816_281635

/-- Represents the daily usage of hair products and calculates the total usage over 14 days. -/
def HairProductUsage (S C H R : ℚ) : Prop :=
  S = 1 ∧
  C = 1/2 * S ∧
  H = 2/3 * S ∧
  R = 1/4 * C ∧
  S * 14 = 14 ∧
  C * 14 = 7 ∧
  H * 14 = 28/3 ∧
  R * 14 = 7/4

/-- Theorem stating the total usage of hair products over 14 days. -/
theorem total_hair_product_usage (S C H R : ℚ) :
  HairProductUsage S C H R →
  S * 14 = 14 ∧ C * 14 = 7 ∧ H * 14 = 28/3 ∧ R * 14 = 7/4 :=
by sorry

end NUMINAMATH_CALUDE_total_hair_product_usage_l2816_281635


namespace NUMINAMATH_CALUDE_matrix_power_difference_l2816_281630

theorem matrix_power_difference (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B = ![![2, 4], ![0, 1]]) : 
  B^20 - 3 • B^19 = ![![-1, 4], ![0, -2]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_difference_l2816_281630


namespace NUMINAMATH_CALUDE_circle_intersection_problem_l2816_281629

theorem circle_intersection_problem (k : ℝ) :
  let center : ℝ × ℝ := ((27 - 3) / 2 + -3, 0)
  let radius : ℝ := (27 - (-3)) / 2
  let circle_equation (x y : ℝ) : Prop := (x - center.1)^2 + (y - center.2)^2 = radius^2
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ circle_equation k y₁ ∧ circle_equation k y₂) →
  (∃ y : ℝ, circle_equation k y ∧ y = 12) →
  k = 3 ∨ k = 21 :=
by sorry


end NUMINAMATH_CALUDE_circle_intersection_problem_l2816_281629


namespace NUMINAMATH_CALUDE_tank_fill_problem_l2816_281647

theorem tank_fill_problem (tank_capacity : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  tank_capacity = 54 →
  added_amount = 9 →
  final_fraction = 9/10 →
  (tank_capacity * final_fraction - added_amount) / tank_capacity = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_problem_l2816_281647


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_zero_l2816_281643

theorem tan_alpha_2_implies_expression_zero (α : Real) (h : Real.tan α = 2) :
  2 * (Real.sin α)^2 - 3 * (Real.sin α) * (Real.cos α) - 2 * (Real.cos α)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_expression_zero_l2816_281643


namespace NUMINAMATH_CALUDE_hair_cut_first_day_l2816_281683

/-- Given that Elizabeth had her hair cut on two consecutive days, with a total of 0.88 inches
    cut off and 0.5 inches cut off on the second day, this theorem proves that 0.38 inches
    were cut off on the first day. -/
theorem hair_cut_first_day (total : ℝ) (second_day : ℝ) (h1 : total = 0.88) (h2 : second_day = 0.5) :
  total - second_day = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_first_day_l2816_281683


namespace NUMINAMATH_CALUDE_largest_divisible_n_l2816_281682

theorem largest_divisible_n : 
  ∀ n : ℕ, n > 5376 → ¬(((n : ℤ)^3 + 200) % (n - 8) = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l2816_281682


namespace NUMINAMATH_CALUDE_inequality_proof_l2816_281607

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^2 ≥ x*y*z*Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2816_281607


namespace NUMINAMATH_CALUDE_x_range_l2816_281619

theorem x_range (x : ℝ) (h1 : 1 / x < 3) (h2 : 1 / x > -4) : x > 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l2816_281619


namespace NUMINAMATH_CALUDE_product_of_solutions_l2816_281687

theorem product_of_solutions (x₁ x₂ : ℝ) (h₁ : x₁ * Real.exp x₁ = Real.exp 2) (h₂ : x₂ * Real.log x₂ = Real.exp 2) :
  x₁ * x₂ = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2816_281687


namespace NUMINAMATH_CALUDE_money_distribution_l2816_281620

theorem money_distribution (a b c : ℝ) : 
  a + b + c = 360 ∧
  a = (1/3) * (b + c) ∧
  b = (2/7) * (a + c) ∧
  a > b
  →
  a - b = 10 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2816_281620


namespace NUMINAMATH_CALUDE_birds_joined_fence_l2816_281613

/-- Given initial numbers of storks and birds on a fence, and the fact that after some birds
    joined there are 2 more birds than storks, prove that 4 birds joined the fence. -/
theorem birds_joined_fence (initial_storks initial_birds : ℕ) 
  (h1 : initial_storks = 5)
  (h2 : initial_birds = 3)
  (h3 : ∃ (joined : ℕ), initial_birds + joined = initial_storks + 2) :
  ∃ (joined : ℕ), joined = 4 ∧ initial_birds + joined = initial_storks + 2 :=
by sorry

end NUMINAMATH_CALUDE_birds_joined_fence_l2816_281613


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_l2816_281676

-- Define variables
variable (a b x y : ℝ)

-- Theorem 1
theorem simplify_expression_1 : 2*a - 3*b + a - 5*b = 3*a - 8*b := by sorry

-- Theorem 2
theorem simplify_expression_2 : (a^2 - 6*a) - 3*(a^2 - 2*a + 1) + 3 = -2*a^2 := by sorry

-- Theorem 3
theorem simplify_expression_3 : 4*(x^2*y - 2*x*y^2) - 3*(-x*y^2 + 2*x^2*y) = -2*x^2*y - 5*x*y^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_l2816_281676


namespace NUMINAMATH_CALUDE_correct_contribution_l2816_281697

/-- Represents the amount spent by each person -/
structure Expenses where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Represents the contribution from Person C to others -/
structure Contribution where
  to_a : ℚ
  to_b : ℚ

def calculate_contribution (e : Expenses) : Contribution :=
  { to_a := 6,
    to_b := 3 }

theorem correct_contribution (e : Expenses) :
  e.b = 12/13 * e.a ∧ 
  e.c = 2/3 * e.b ∧ 
  calculate_contribution e = { to_a := 6, to_b := 3 } :=
by sorry

#check correct_contribution

end NUMINAMATH_CALUDE_correct_contribution_l2816_281697


namespace NUMINAMATH_CALUDE_socorro_training_time_l2816_281681

/-- Calculates the total training time in hours given daily training times and number of days -/
def total_training_time (mult_time : ℕ) (div_time : ℕ) (days : ℕ) (mins_per_hour : ℕ) : ℚ :=
  (mult_time + div_time) * days / mins_per_hour

/-- Proves that Socorro's total training time is 5 hours -/
theorem socorro_training_time :
  total_training_time 10 20 10 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_socorro_training_time_l2816_281681


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2816_281690

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  X^4 + 1 = (X^2 - 3*X + 5) * q + r ∧
  r.degree < (X^2 - 3*X + 5).degree ∧
  r = -3*X - 19 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2816_281690


namespace NUMINAMATH_CALUDE_b_invested_after_six_months_l2816_281657

/-- Represents the investment scenario and calculates when B invested -/
def calculate_b_investment_time (a_investment : ℕ) (b_investment : ℕ) (total_profit : ℕ) (a_profit : ℕ) : ℕ :=
  let a_time := 12
  let b_time := 12 - (a_investment * a_time * total_profit) / (a_profit * (a_investment + b_investment))
  b_time

/-- Theorem stating that B invested 6 months after A, given the problem conditions -/
theorem b_invested_after_six_months :
  calculate_b_investment_time 300 200 100 75 = 6 := by
  sorry

end NUMINAMATH_CALUDE_b_invested_after_six_months_l2816_281657


namespace NUMINAMATH_CALUDE_sequence_problem_l2816_281616

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_sum_a : a 1 + a 3 + a 5 + a 7 + a 9 = 50)
  (h_prod_b : b 4 * b 6 * b 14 * b 16 = 625) :
  (a 2 + a 8) / b 10 = 4 ∨ (a 2 + a 8) / b 10 = -4 :=
sorry

end NUMINAMATH_CALUDE_sequence_problem_l2816_281616


namespace NUMINAMATH_CALUDE_folk_song_competition_probability_l2816_281624

/-- The number of provinces in the competition -/
def num_provinces : ℕ := 6

/-- The number of singers per province -/
def singers_per_province : ℕ := 2

/-- The total number of singers in the competition -/
def total_singers : ℕ := num_provinces * singers_per_province

/-- The number of winners to be selected -/
def num_winners : ℕ := 4

/-- The probability of selecting 4 winners such that exactly two of them are from the same province -/
theorem folk_song_competition_probability :
  (num_provinces.choose 1 * singers_per_province.choose 2 * (total_singers - singers_per_province).choose 1 * (num_provinces - 1).choose 1) / total_singers.choose num_winners = 16 / 33 := by
  sorry

end NUMINAMATH_CALUDE_folk_song_competition_probability_l2816_281624


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l2816_281631

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 25) : 
  a * b = 8 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l2816_281631


namespace NUMINAMATH_CALUDE_weight_replacement_l2816_281640

theorem weight_replacement (total_weight : ℝ) (replaced_weight : ℝ) : 
  (8 : ℝ) * ((total_weight - replaced_weight + 77) / 8 - total_weight / 8) = 1.5 →
  replaced_weight = 65 := by
sorry

end NUMINAMATH_CALUDE_weight_replacement_l2816_281640


namespace NUMINAMATH_CALUDE_train_speed_problem_l2816_281623

/-- The speed of Train A in miles per hour -/
def speed_A : ℝ := 30

/-- The time difference between Train A and Train B's departure in hours -/
def time_diff : ℝ := 2

/-- The distance at which Train B overtakes Train A in miles -/
def overtake_distance : ℝ := 360

/-- The speed of Train B in miles per hour -/
def speed_B : ℝ := 42

theorem train_speed_problem :
  speed_A * (overtake_distance / speed_A) = 
  speed_B * (overtake_distance / speed_B - time_diff) ∧
  speed_B * time_diff + speed_A * time_diff = overtake_distance := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2816_281623


namespace NUMINAMATH_CALUDE_graph_not_in_second_quadrant_implies_a_nonnegative_l2816_281666

-- Define the function f(x) = x^3 - a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a

-- Define the condition that the graph does not pass through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f a x ≤ 0

-- Theorem statement
theorem graph_not_in_second_quadrant_implies_a_nonnegative (a : ℝ) :
  not_in_second_quadrant a → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_graph_not_in_second_quadrant_implies_a_nonnegative_l2816_281666


namespace NUMINAMATH_CALUDE_constant_term_theorem_l2816_281639

theorem constant_term_theorem (m : ℝ) : 
  (∀ x, (x - m) * (x + 7) = x^2 + (7 - m) * x - 7 * m) →
  -7 * m = 14 →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_constant_term_theorem_l2816_281639


namespace NUMINAMATH_CALUDE_max_d_is_three_l2816_281608

/-- Represents a 7-digit number of the form 5d5,22e1 -/
def SevenDigitNumber (d e : Nat) : Nat :=
  5000000 + d * 100000 + 500000 + 22000 + e * 10 + 1

/-- Checks if a number is divisible by 33 -/
def isDivisibleBy33 (n : Nat) : Prop :=
  n % 33 = 0

/-- Checks if d and e are single digits -/
def areSingleDigits (d e : Nat) : Prop :=
  d ≤ 9 ∧ e ≤ 9

/-- The main theorem stating that the maximum value of d is 3 -/
theorem max_d_is_three :
  ∃ (d e : Nat), areSingleDigits d e ∧ 
    isDivisibleBy33 (SevenDigitNumber d e) ∧
    d = 3 ∧
    ∀ (d' e' : Nat), areSingleDigits d' e' → 
      isDivisibleBy33 (SevenDigitNumber d' e') → 
      d' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_d_is_three_l2816_281608


namespace NUMINAMATH_CALUDE_sqrt_three_times_five_to_fourth_l2816_281625

theorem sqrt_three_times_five_to_fourth (x : ℝ) : 
  x = Real.sqrt (5^4 + 5^4 + 5^4) → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_times_five_to_fourth_l2816_281625


namespace NUMINAMATH_CALUDE_cos_squared_plus_sin_double_l2816_281684

open Real

theorem cos_squared_plus_sin_double (α : ℝ) (h : tan α = 2) : 
  cos α ^ 2 + sin (2 * α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_plus_sin_double_l2816_281684


namespace NUMINAMATH_CALUDE_little_league_games_l2816_281669

theorem little_league_games (games_won : ℕ) (games_lost_difference : ℕ) : 
  games_won = 18 → games_lost_difference = 21 → games_won + (games_won + games_lost_difference) = 57 := by
  sorry

end NUMINAMATH_CALUDE_little_league_games_l2816_281669


namespace NUMINAMATH_CALUDE_decimal_23_to_binary_l2816_281646

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec to_binary_helper (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else to_binary_helper (m / 2) ((m % 2) :: acc)
    to_binary_helper n []

theorem decimal_23_to_binary :
  decimal_to_binary 23 = [1, 0, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_23_to_binary_l2816_281646


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l2816_281662

theorem geometric_sequence_product (a b : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ -1 = -1 ∧ a = -1 * r ∧ b = a * r ∧ 2 = b * r) →
  a * b = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l2816_281662


namespace NUMINAMATH_CALUDE_correct_calculation_l2816_281659

theorem correct_calculation (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2816_281659


namespace NUMINAMATH_CALUDE_plot_breadth_is_8_l2816_281691

/-- A rectangular plot with the given properties. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area_is_18_times_breadth : length * breadth = 18 * breadth
  length_breadth_difference : length - breadth = 10

/-- The breadth of the rectangular plot is 8 meters. -/
theorem plot_breadth_is_8 (plot : RectangularPlot) : plot.breadth = 8 := by
  sorry

end NUMINAMATH_CALUDE_plot_breadth_is_8_l2816_281691


namespace NUMINAMATH_CALUDE_triangle_with_sum_of_two_angles_less_than_third_is_obtuse_l2816_281673

theorem triangle_with_sum_of_two_angles_less_than_third_is_obtuse 
  (α β γ : Real) 
  (triangle_angles : α + β + γ = 180) 
  (angle_sum_condition : α + β < γ) : 
  γ > 90 := by
sorry

end NUMINAMATH_CALUDE_triangle_with_sum_of_two_angles_less_than_third_is_obtuse_l2816_281673


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2816_281633

/-- Given an integer n, returns its last two digits as a pair -/
def lastTwoDigits (n : ℤ) : ℤ × ℤ :=
  let tens := (n / 10) % 10
  let ones := n % 10
  (tens, ones)

/-- Given an integer n, returns true if it's divisible by 8 -/
def divisibleBy8 (n : ℤ) : Prop :=
  n % 8 = 0

theorem last_two_digits_product (n : ℤ) :
  divisibleBy8 n ∧ (let (a, b) := lastTwoDigits n; a + b = 14) →
  (let (a, b) := lastTwoDigits n; a * b = 48) :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2816_281633


namespace NUMINAMATH_CALUDE_cos_three_halves_pi_l2816_281674

theorem cos_three_halves_pi : Real.cos (3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_halves_pi_l2816_281674


namespace NUMINAMATH_CALUDE_yellow_peaches_count_l2816_281698

/-- The number of yellow peaches in a basket, given the number of green peaches
    and the difference between green and yellow peaches. -/
def yellow_peaches (green : ℕ) (difference : ℕ) : ℕ :=
  green - difference

/-- Theorem stating that the number of yellow peaches is 6, given the conditions. -/
theorem yellow_peaches_count : yellow_peaches 14 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_yellow_peaches_count_l2816_281698


namespace NUMINAMATH_CALUDE_correct_metal_ratio_l2816_281611

/-- Represents the ratio of two metals in an alloy -/
structure MetalRatio where
  a : ℚ
  b : ℚ

/-- Calculates the cost of an alloy given the ratio of metals and their individual costs -/
def alloyCost (ratio : MetalRatio) (costA costB : ℚ) : ℚ :=
  (ratio.a * costA + ratio.b * costB) / (ratio.a + ratio.b)

/-- Theorem stating the correct ratio of metals to achieve the desired alloy cost -/
theorem correct_metal_ratio :
  let desiredRatio : MetalRatio := ⟨3, 1⟩
  let costA : ℚ := 68
  let costB : ℚ := 96
  let desiredCost : ℚ := 75
  alloyCost desiredRatio costA costB = desiredCost := by sorry

end NUMINAMATH_CALUDE_correct_metal_ratio_l2816_281611


namespace NUMINAMATH_CALUDE_carrot_calories_l2816_281642

/-- The number of calories in a pound of carrots -/
def calories_per_pound_carrots : ℕ := 51

/-- The number of pounds of carrots Tom eats -/
def pounds_carrots : ℕ := 1

/-- The number of pounds of broccoli Tom eats -/
def pounds_broccoli : ℕ := 2

/-- The ratio of calories in broccoli compared to carrots -/
def broccoli_calorie_ratio : ℚ := 1/3

/-- The total number of calories Tom ate -/
def total_calories : ℕ := 85

theorem carrot_calories :
  calories_per_pound_carrots * pounds_carrots +
  (calories_per_pound_carrots : ℚ) * broccoli_calorie_ratio * pounds_broccoli = total_calories := by
  sorry

end NUMINAMATH_CALUDE_carrot_calories_l2816_281642


namespace NUMINAMATH_CALUDE_no_real_m_for_reciprocal_sum_l2816_281679

theorem no_real_m_for_reciprocal_sum (m : ℝ) : ¬ (∃ x₁ x₂ : ℝ,
  (m * x₁^2 - 2*x₁ + m*(m^2 + 1) = 0) ∧
  (m * x₂^2 - 2*x₂ + m*(m^2 + 1) = 0) ∧
  (x₁ ≠ x₂) ∧
  (1/x₁ + 1/x₂ = m)) := by
  sorry

#check no_real_m_for_reciprocal_sum

end NUMINAMATH_CALUDE_no_real_m_for_reciprocal_sum_l2816_281679


namespace NUMINAMATH_CALUDE_problem_statement_l2816_281668

theorem problem_statement (x : ℝ) 
  (h : (4:ℝ)^(2*x) + (2:ℝ)^(-x) + 1 = (129 + 8*Real.sqrt 2) * ((4:ℝ)^x + (2:ℝ)^(-x) - (2:ℝ)^x)) :
  10 * x = 35 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2816_281668


namespace NUMINAMATH_CALUDE_no_solution_when_m_zero_infinite_solutions_when_m_neg_three_unique_solution_when_m_not_zero_and_not_neg_three_l2816_281658

-- Define the system of linear equations
def system (m x y : ℝ) : Prop :=
  m * x + y = -1 ∧ 3 * m * x - m * y = 2 * m + 3

-- Define the determinant of the coefficient matrix
def det_coeff (m : ℝ) : ℝ := -m * (m + 3)

-- Define the determinants for x and y
def det_x (m : ℝ) : ℝ := -m - 3
def det_y (m : ℝ) : ℝ := 2 * m * (m + 3)

-- Theorem for the case when m = 0
theorem no_solution_when_m_zero :
  ¬∃ x y : ℝ, system 0 x y :=
sorry

-- Theorem for the case when m = -3
theorem infinite_solutions_when_m_neg_three :
  ∃ x y : ℝ, system (-3) x y ∧ ∀ t : ℝ, system (-3) (x + t) (y - 3*t) :=
sorry

-- Theorem for the case when m ≠ 0 and m ≠ -3
theorem unique_solution_when_m_not_zero_and_not_neg_three (m : ℝ) (hm : m ≠ 0 ∧ m ≠ -3) :
  ∃! x y : ℝ, system m x y ∧ x = 1/m ∧ y = -2 :=
sorry

end NUMINAMATH_CALUDE_no_solution_when_m_zero_infinite_solutions_when_m_neg_three_unique_solution_when_m_not_zero_and_not_neg_three_l2816_281658


namespace NUMINAMATH_CALUDE_triangle_theorem_l2816_281637

noncomputable section

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  (t.c / (t.a + t.b)) + (sin t.A / (sin t.B + sin t.C)) = 1

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : condition t) 
  (h2 : t.b = sqrt 2) : 
  t.B = π/3 ∧ (∀ (x y : ℝ), x^2 + y^2 ≤ 4) ∧ (∃ (x y : ℝ), x^2 + y^2 = 4) :=
sorry

end

end NUMINAMATH_CALUDE_triangle_theorem_l2816_281637


namespace NUMINAMATH_CALUDE_perpendicular_planes_l2816_281638

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (a b : Line) (α β : Plane) :
  perpendicular a β → 
  parallel a b → 
  contained_in b α → 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l2816_281638


namespace NUMINAMATH_CALUDE_sum_of_matching_positions_is_322_l2816_281654

def array_size : Nat × Nat := (16, 10)

def esther_fill (r c : Nat) : Nat :=
  16 * (r - 1) + c

def frida_fill (r c : Nat) : Nat :=
  10 * (c - 1) + r

def is_same_position (r c : Nat) : Prop :=
  esther_fill r c = frida_fill r c

def sum_of_matching_positions : Nat :=
  (esther_fill 1 1) + (esther_fill 4 6) + (esther_fill 7 11) + (esther_fill 10 16)

theorem sum_of_matching_positions_is_322 :
  sum_of_matching_positions = 322 :=
sorry

end NUMINAMATH_CALUDE_sum_of_matching_positions_is_322_l2816_281654


namespace NUMINAMATH_CALUDE_simplify_expression_l2816_281610

theorem simplify_expression (a b : ℝ) (h : a + b ≠ 1) :
  1 - (1 / (1 + (a + b) / (1 - a - b))) = a + b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2816_281610


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2816_281680

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_first : a 1 = 3) 
  (h_third : a 3 = 5) : 
  a 7 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2816_281680


namespace NUMINAMATH_CALUDE_inverse_of_10_mod_1001_l2816_281695

theorem inverse_of_10_mod_1001 : ∃ x : ℕ, x ∈ Finset.range 1001 ∧ (10 * x) % 1001 = 1 :=
by
  use 901
  sorry

end NUMINAMATH_CALUDE_inverse_of_10_mod_1001_l2816_281695


namespace NUMINAMATH_CALUDE_larger_number_is_72_l2816_281665

theorem larger_number_is_72 (a b : ℝ) : 
  5 * b = 6 * a ∧ b - a = 12 → b = 72 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_is_72_l2816_281665


namespace NUMINAMATH_CALUDE_adults_attending_play_l2816_281645

/-- Proves the number of adults attending a play given ticket prices and total receipts --/
theorem adults_attending_play (adult_price children_price total_receipts total_attendance : ℕ) 
  (h1 : adult_price = 25)
  (h2 : children_price = 15)
  (h3 : total_receipts = 7200)
  (h4 : total_attendance = 400) :
  ∃ (adults children : ℕ), 
    adults + children = total_attendance ∧ 
    adult_price * adults + children_price * children = total_receipts ∧
    adults = 120 := by
  sorry


end NUMINAMATH_CALUDE_adults_attending_play_l2816_281645


namespace NUMINAMATH_CALUDE_ceiling_floor_product_range_l2816_281621

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → (Int.ceil y * Int.floor y = 72) → y ∈ Set.Icc (-9 : ℝ) (-8 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_range_l2816_281621


namespace NUMINAMATH_CALUDE_simplify_expression_l2816_281686

theorem simplify_expression (x : ℝ) : (2*x)^3 + (3*x)*(x^2) = 11*x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2816_281686


namespace NUMINAMATH_CALUDE_M_squared_equals_36_50_times_144_36_and_sum_of_digits_75_l2816_281636

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Definition of M -/
def M : ℕ := sorry

theorem M_squared_equals_36_50_times_144_36_and_sum_of_digits_75 :
  M^2 = 36^50 * 144^36 ∧ sum_of_digits M = 75 := by
  sorry

end NUMINAMATH_CALUDE_M_squared_equals_36_50_times_144_36_and_sum_of_digits_75_l2816_281636


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2816_281606

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 2 * Real.sqrt 3 * x + 1 = 0 → (x - 1/x = 2 * Real.sqrt 2 ∨ x - 1/x = -2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2816_281606


namespace NUMINAMATH_CALUDE_key_arrangement_theorem_l2816_281670

/-- The number of permutations of n elements -/
def totalPermutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of permutations of n elements with exactly one cycle -/
def onePermutation (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of permutations of 10 elements where at least two cycles are present -/
def atLeastTwoCycles : ℕ := totalPermutations 10 - onePermutation 10

/-- The number of permutations of 10 elements with exactly two cycles -/
def exactlyTwoCycles : ℕ :=
  choose 10 1 * Nat.factorial 8 +
  choose 10 2 * Nat.factorial 7 +
  choose 10 3 * Nat.factorial 2 * Nat.factorial 6 +
  choose 10 4 * Nat.factorial 3 * Nat.factorial 5 +
  (choose 10 5 * Nat.factorial 4 * Nat.factorial 4) / 2

theorem key_arrangement_theorem :
  atLeastTwoCycles = 9 * Nat.factorial 9 ∧ exactlyTwoCycles = 1024576 := by sorry

end NUMINAMATH_CALUDE_key_arrangement_theorem_l2816_281670


namespace NUMINAMATH_CALUDE_chess_team_arrangement_count_l2816_281678

def chess_team_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  if num_boys + num_girls ≠ 7 then 0
  else if num_boys ≠ 3 then 0
  else if num_girls ≠ 4 then 0
  else Nat.factorial num_boys * Nat.factorial num_girls

theorem chess_team_arrangement_count :
  chess_team_arrangements 3 4 = 144 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_arrangement_count_l2816_281678


namespace NUMINAMATH_CALUDE_audit_options_l2816_281605

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem audit_options (initial_OR initial_GTU first_week_OR first_week_GTU : ℕ) 
  (h1 : initial_OR = 13)
  (h2 : initial_GTU = 15)
  (h3 : first_week_OR = 2)
  (h4 : first_week_GTU = 3) :
  (choose (initial_OR - first_week_OR) first_week_OR) * 
  (choose (initial_GTU - first_week_GTU) first_week_GTU) = 12100 := by
  sorry

end NUMINAMATH_CALUDE_audit_options_l2816_281605


namespace NUMINAMATH_CALUDE_last_digits_l2816_281699

theorem last_digits (n : ℕ) : 
  (6^811 : ℕ) % 10 = 6 ∧ 
  (2^1000 : ℕ) % 10 = 6 ∧ 
  (3^999 : ℕ) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digits_l2816_281699


namespace NUMINAMATH_CALUDE_i_times_one_plus_i_l2816_281661

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem i_times_one_plus_i : i * (1 + i) = i - 1 := by
  sorry

end NUMINAMATH_CALUDE_i_times_one_plus_i_l2816_281661


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l2816_281689

theorem ceiling_sum_sqrt : ⌈Real.sqrt 50⌉ + ⌈Real.sqrt 150⌉ + ⌈Real.sqrt 250⌉ = 37 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l2816_281689


namespace NUMINAMATH_CALUDE_fred_cards_after_purchase_l2816_281692

/-- The number of baseball cards Fred has after Melanie's purchase -/
def fred_remaining_cards (initial : ℕ) (bought : ℕ) : ℕ :=
  initial - bought

/-- Theorem: Fred has 2 baseball cards left after Melanie's purchase -/
theorem fred_cards_after_purchase :
  fred_remaining_cards 5 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fred_cards_after_purchase_l2816_281692


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2816_281614

/-- Two lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_lines_a_value :
  ∀ a : ℝ,
  let l1 : Line := ⟨2, a, -7⟩
  let l2 : Line := ⟨a - 3, 1, 4⟩
  perpendicular l1 l2 → a = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l2816_281614


namespace NUMINAMATH_CALUDE_rectangle_longest_side_l2816_281644

/-- A rectangle with perimeter 240 feet and area eight times its perimeter has its longest side equal to 101 feet. -/
theorem rectangle_longest_side : ∀ l w : ℝ,
  l > 0 → w > 0 →
  2 * (l + w) = 240 →
  l * w = 8 * 240 →
  max l w = 101 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_longest_side_l2816_281644


namespace NUMINAMATH_CALUDE_shaded_squares_correct_l2816_281651

/-- Given a square grid with odd side length, calculates the number of shaded squares along the two diagonals -/
def shadedSquares (n : ℕ) : ℕ :=
  2 * n - 1

theorem shaded_squares_correct (n : ℕ) (h : Odd n) :
  shadedSquares n = 2 * n - 1 := by
  sorry

#eval shadedSquares 7  -- Expected: 13
#eval shadedSquares 101  -- Expected: 201

end NUMINAMATH_CALUDE_shaded_squares_correct_l2816_281651


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2816_281618

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_of_M_and_N : M ∩ N = Set.Ioo 1 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2816_281618


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l2816_281603

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the circle with diameter AB passing through origin
def circle_AB_origin (xA yA xB yB : ℝ) : Prop := xA*xB + yA*yB = 0

theorem ellipse_and_line_theorem :
  -- Given conditions
  let a : ℝ := Real.sqrt 3
  let e : ℝ := Real.sqrt 6 / 3
  let c : ℝ := e * a

  -- Part 1: Prove the standard equation of ellipse C
  (∀ x y : ℝ, ellipse_C x y ↔ x^2/3 + y^2 = 1) ∧

  -- Part 2: Prove the equation of line l
  (∃ m : ℝ, m = Real.sqrt 3 / 3 ∨ m = -Real.sqrt 3 / 3) ∧
  (∀ m : ℝ, (m = Real.sqrt 3 / 3 ∨ m = -Real.sqrt 3 / 3) →
    (∃ xA yA xB yB : ℝ,
      ellipse_C xA yA ∧ ellipse_C xB yB ∧
      line_l m xA yA ∧ line_l m xB yB ∧
      circle_AB_origin xA yA xB yB)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l2816_281603


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2816_281696

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    where a line passing through its left vertex with slope k 
    intersects the ellipse at a point whose x-coordinate is the 
    distance from the center to the focus, prove that the 
    eccentricity e of the ellipse is between 1/2 and 2/3 
    when k is between 1/3 and 1/2. -/
theorem ellipse_eccentricity_range (a b : ℝ) (k : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : 1/3 < k) (h4 : k < 1/2) :
  let e := Real.sqrt (1 - b^2 / a^2)
  ∃ (x y : ℝ), 
    x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    y = k * (x + a) ∧
    x = a * e ∧
    1/2 < e ∧ e < 2/3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2816_281696


namespace NUMINAMATH_CALUDE_claire_photos_l2816_281634

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 20) :
  claire = 10 := by
  sorry

end NUMINAMATH_CALUDE_claire_photos_l2816_281634


namespace NUMINAMATH_CALUDE_circle_radius_for_equal_areas_l2816_281648

/-- The radius of a circle satisfying the given conditions for a right-angled triangle --/
theorem circle_radius_for_equal_areas (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_side_lengths : a = 6 ∧ b = 8 ∧ c = 10) : 
  ∃ r : ℝ, r^2 = 24 / Real.pi ∧ 
    (π * r^2 = a * b / 2) ∧
    (π * r^2 - a * b / 2 = a * b / 2 - π * r^2) :=
sorry

end NUMINAMATH_CALUDE_circle_radius_for_equal_areas_l2816_281648


namespace NUMINAMATH_CALUDE_least_sum_of_primes_l2816_281685

theorem least_sum_of_primes (p q : ℕ) : 
  Prime p → Prime q → 
  (∀ n : ℕ, n > 0 → (n^(3*p*q) - n) % (3*p*q) = 0) → 
  (∀ p' q' : ℕ, Prime p' → Prime q' → 
    (∀ n : ℕ, n > 0 → (n^(3*p'*q') - n) % (3*p'*q') = 0) → 
    p' + q' ≥ p + q) →
  p + q = 28 := by
sorry

end NUMINAMATH_CALUDE_least_sum_of_primes_l2816_281685


namespace NUMINAMATH_CALUDE_short_story_booklets_l2816_281663

theorem short_story_booklets (pages_per_booklet : ℕ) (total_pages : ℕ) (h1 : pages_per_booklet = 9) (h2 : total_pages = 441) :
  total_pages / pages_per_booklet = 49 := by
  sorry

end NUMINAMATH_CALUDE_short_story_booklets_l2816_281663


namespace NUMINAMATH_CALUDE_mixed_oil_rate_l2816_281677

/-- Given two oils mixed together, calculate the rate of the mixed oil per litre -/
theorem mixed_oil_rate (volume1 volume2 rate1 rate2 : ℚ) 
  (h1 : volume1 = 10)
  (h2 : volume2 = 5)
  (h3 : rate1 = 40)
  (h4 : rate2 = 66) :
  (volume1 * rate1 + volume2 * rate2) / (volume1 + volume2) = 730 / 15 := by
  sorry

end NUMINAMATH_CALUDE_mixed_oil_rate_l2816_281677


namespace NUMINAMATH_CALUDE_sequence_problem_l2816_281653

def arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

theorem sequence_problem (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : arithmetic_sequence 1 a₁ a₂ a₃)
  (h2 : arithmetic_sequence a₁ a₂ a₃ 9)
  (h3 : geometric_sequence (-9) b₁ b₂ b₃)
  (h4 : geometric_sequence b₁ b₂ b₃ (-1)) :
  b₂ / (a₁ + a₃) = -3/10 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l2816_281653


namespace NUMINAMATH_CALUDE_inspector_rejection_l2816_281632

-- Define the rejection rate
def rejection_rate : ℝ := 0.15

-- Define the number of meters examined
def meters_examined : ℝ := 66.67

-- Define the function to calculate the number of rejected meters
def rejected_meters (rate : ℝ) (total : ℝ) : ℝ := rate * total

-- Theorem statement
theorem inspector_rejection :
  rejected_meters rejection_rate meters_examined = 10 := by
  sorry

end NUMINAMATH_CALUDE_inspector_rejection_l2816_281632


namespace NUMINAMATH_CALUDE_minimum_guests_with_both_l2816_281693

theorem minimum_guests_with_both (total : ℕ) 
  (sunglasses : ℕ) (wristbands : ℕ) (both : ℕ) : 
  (3 : ℚ) / 7 * total = sunglasses →
  (4 : ℚ) / 9 * total = wristbands →
  total = sunglasses + wristbands - both →
  total ≥ 63 →
  both ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_minimum_guests_with_both_l2816_281693


namespace NUMINAMATH_CALUDE_no_infinite_line_family_l2816_281650

theorem no_infinite_line_family : ¬ ∃ (k : ℕ → ℝ), 
  (∀ n, k (n + 1) ≥ k n - 1 / k n) ∧ 
  (∀ n, k n * k (n + 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_line_family_l2816_281650


namespace NUMINAMATH_CALUDE_number_of_persons_l2816_281615

theorem number_of_persons (total_amount : ℕ) (amount_per_person : ℕ) 
  (h1 : total_amount = 42900)
  (h2 : amount_per_person = 1950) :
  total_amount / amount_per_person = 22 := by
  sorry

end NUMINAMATH_CALUDE_number_of_persons_l2816_281615


namespace NUMINAMATH_CALUDE_max_vector_norm_l2816_281600

theorem max_vector_norm (θ : ℝ) : 
  (‖(2 * Real.cos θ - Real.sqrt 3, 2 * Real.sin θ + 1)‖ : ℝ) ≤ 4 ∧ 
  ∃ θ₀ : ℝ, ‖(2 * Real.cos θ₀ - Real.sqrt 3, 2 * Real.sin θ₀ + 1)‖ = 4 :=
sorry

end NUMINAMATH_CALUDE_max_vector_norm_l2816_281600


namespace NUMINAMATH_CALUDE_al_sandwich_count_l2816_281652

/-- The number of different types of bread available. -/
def num_bread : ℕ := 5

/-- The number of different types of meat available. -/
def num_meat : ℕ := 6

/-- The number of different types of cheese available. -/
def num_cheese : ℕ := 5

/-- Represents whether French bread is available. -/
def french_bread_available : Prop := True

/-- Represents whether turkey is available. -/
def turkey_available : Prop := True

/-- Represents whether Swiss cheese is available. -/
def swiss_cheese_available : Prop := True

/-- Represents whether white bread is available. -/
def white_bread_available : Prop := True

/-- Represents whether rye bread is available. -/
def rye_bread_available : Prop := True

/-- Represents whether chicken is available. -/
def chicken_available : Prop := True

/-- The number of sandwich combinations with turkey and Swiss cheese. -/
def turkey_swiss_combos : ℕ := num_bread

/-- The number of sandwich combinations with white bread and chicken. -/
def white_chicken_combos : ℕ := num_cheese

/-- The number of sandwich combinations with rye bread and turkey. -/
def rye_turkey_combos : ℕ := num_cheese

/-- The total number of sandwich combinations Al can order. -/
def al_sandwich_options : ℕ := num_bread * num_meat * num_cheese - turkey_swiss_combos - white_chicken_combos - rye_turkey_combos

theorem al_sandwich_count :
  french_bread_available ∧ 
  turkey_available ∧ 
  swiss_cheese_available ∧ 
  white_bread_available ∧
  rye_bread_available ∧
  chicken_available →
  al_sandwich_options = 135 := by
  sorry

end NUMINAMATH_CALUDE_al_sandwich_count_l2816_281652


namespace NUMINAMATH_CALUDE_notebook_and_pen_prices_l2816_281672

def notebook_price : ℝ := 12
def pen_price : ℝ := 6

theorem notebook_and_pen_prices :
  (2 * notebook_price + pen_price = 30) ∧
  (notebook_price = 2 * pen_price) :=
by sorry

end NUMINAMATH_CALUDE_notebook_and_pen_prices_l2816_281672


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2816_281655

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x - 16 * y^2 + 32 * y - 12 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance : ℝ := 2

/-- Theorem: The distance between the vertices of the hyperbola is 2 -/
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_equation x y → vertex_distance = 2 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l2816_281655


namespace NUMINAMATH_CALUDE_circle_distance_problem_l2816_281688

theorem circle_distance_problem (r₁ r₂ d : ℝ) (A B C : ℝ × ℝ) :
  r₁ = 13 →
  r₂ = 30 →
  d = 41 →
  let O₁ : ℝ × ℝ := (0, 0)
  let O₂ : ℝ × ℝ := (d, 0)
  (A.1 - O₂.1)^2 + A.2^2 = r₁^2 →
  A.1 > r₂ →
  (B.1 - O₂.1)^2 + B.2^2 = r₁^2 →
  (C.1 - O₁.1)^2 + C.2^2 = r₂^2 →
  B = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = 12^2 * 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_distance_problem_l2816_281688


namespace NUMINAMATH_CALUDE_point_on_number_line_l2816_281667

theorem point_on_number_line (A : ℝ) : 
  (|A| = 5) ↔ (A = 5 ∨ A = -5) := by sorry

end NUMINAMATH_CALUDE_point_on_number_line_l2816_281667


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2816_281675

/-- A line in the xy-plane represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Checks if two lines are parallel. -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Checks if a point lies on a line. -/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.y_intercept

/-- The given line y = -3x + 7 -/
def given_line : Line :=
  { slope := -3, y_intercept := 7 }

theorem y_intercept_of_parallel_line :
  ∀ (b : Line),
    are_parallel b given_line →
    point_on_line b 5 (-2) →
    b.y_intercept = 13 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2816_281675


namespace NUMINAMATH_CALUDE_equation_solutions_l2816_281628

theorem equation_solutions (m : ℕ+) :
  ∀ x y z : ℕ+, (x^2 + y^2)^m.val = (x * y)^z.val →
  ∃ k n : ℕ+, x = 2^k.val ∧ y = 2^k.val ∧ z = (1 + 2*k.val)*n.val ∧ m = 2*k.val*n.val :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2816_281628


namespace NUMINAMATH_CALUDE_roots_distance_bound_l2816_281664

theorem roots_distance_bound (v w : ℂ) : 
  v ≠ w → 
  (v^401 = 1) → 
  (w^401 = 1) → 
  Complex.abs (v + w) < Real.sqrt (3 + Real.sqrt 5) := by
sorry

end NUMINAMATH_CALUDE_roots_distance_bound_l2816_281664


namespace NUMINAMATH_CALUDE_squares_in_figure_150_l2816_281612

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The sequence of squares for the first four figures -/
def initial_sequence : List ℕ := [1, 7, 19, 37]

theorem squares_in_figure_150 :
  f 150 = 67951 ∧
  (∀ n : Fin 4, f n.val = initial_sequence.get n) :=
sorry

end NUMINAMATH_CALUDE_squares_in_figure_150_l2816_281612


namespace NUMINAMATH_CALUDE_cloth_cost_price_theorem_l2816_281656

/-- Represents the cost price of one meter of cloth given the selling conditions --/
def cost_price_per_meter (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_meters) / total_meters

/-- Theorem stating that under the given conditions, the cost price per meter is 88 --/
theorem cloth_cost_price_theorem (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ)
    (h1 : total_meters = 45)
    (h2 : selling_price = 4500)
    (h3 : profit_per_meter = 12) :
    cost_price_per_meter total_meters selling_price profit_per_meter = 88 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_theorem_l2816_281656


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l2816_281601

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 4 = 4 ∧
  a 3 + a 8 = 5

/-- Theorem stating that a_7 = 1 for the given arithmetic sequence -/
theorem arithmetic_sequence_a7 (a : ℕ → ℚ) (h : ArithmeticSequence a) : a 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l2816_281601
