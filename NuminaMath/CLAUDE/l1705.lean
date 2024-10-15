import Mathlib

namespace NUMINAMATH_CALUDE_fraction_above_line_is_five_sixths_l1705_170524

/-- A square in the coordinate plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- A line in the coordinate plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The fraction of the square's area above a given line -/
def fractionAboveLine (s : Square) (l : Line) : ℝ := sorry

/-- The specific square from the problem -/
def problemSquare : Square :=
  { bottomLeft := (2, 1),
    topRight := (5, 4) }

/-- The specific line from the problem -/
def problemLine : Line :=
  { point1 := (2, 3),
    point2 := (5, 1) }

theorem fraction_above_line_is_five_sixths :
  fractionAboveLine problemSquare problemLine = 5/6 := by sorry

end NUMINAMATH_CALUDE_fraction_above_line_is_five_sixths_l1705_170524


namespace NUMINAMATH_CALUDE_circle_passes_through_M_and_has_same_center_l1705_170563

-- Define the center of the given circle
def center : ℝ × ℝ := (2, -3)

-- Define the point M
def point_M : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle we want to prove
def circle_equation (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = 25

-- Theorem statement
theorem circle_passes_through_M_and_has_same_center :
  -- The circle passes through point M
  circle_equation point_M.1 point_M.2 ∧
  -- The circle has the same center as (x-2)^2 + (y+3)^2 = 16
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = 25 ↔ circle_equation x y :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_M_and_has_same_center_l1705_170563


namespace NUMINAMATH_CALUDE_silver_coin_value_proof_l1705_170561

/-- The value of a silver coin -/
def silver_coin_value : ℝ := 25

theorem silver_coin_value_proof :
  let gold_coin_value : ℝ := 50
  let num_gold_coins : ℕ := 3
  let num_silver_coins : ℕ := 5
  let cash : ℝ := 30
  let total_value : ℝ := 305
  silver_coin_value = (total_value - gold_coin_value * num_gold_coins - cash) / num_silver_coins :=
by
  sorry

end NUMINAMATH_CALUDE_silver_coin_value_proof_l1705_170561


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_x_squared_minus_x_geq_zero_l1705_170508

theorem negation_of_forall_positive_x_squared_minus_x_geq_zero :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_x_squared_minus_x_geq_zero_l1705_170508


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1705_170527

/-- Given a geometric sequence of positive integers where the first term is 5
    and the fourth term is 405, the fifth term is 405. -/
theorem geometric_sequence_fifth_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 5 →                            -- First term is 5
  a 4 = 405 →                          -- Fourth term is 405
  a 5 = 405 :=                         -- Fifth term is 405
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l1705_170527


namespace NUMINAMATH_CALUDE_sector_central_angle_l1705_170542

theorem sector_central_angle (r l : ℝ) (h1 : 2 * r + l = 4) (h2 : (1 / 2) * l * r = 1) :
  l / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1705_170542


namespace NUMINAMATH_CALUDE_shortest_side_of_triangle_l1705_170594

theorem shortest_side_of_triangle (a b c : ℕ) (area : ℕ) : 
  a = 21 →
  a + b + c = 48 →
  area * area = 24 * 3 * (24 - b) * (b - 3) →
  b ≤ c →
  b = 10 :=
sorry

end NUMINAMATH_CALUDE_shortest_side_of_triangle_l1705_170594


namespace NUMINAMATH_CALUDE_square_area_from_two_points_square_area_specific_case_l1705_170558

/-- The area of a square given two points on the same side -/
theorem square_area_from_two_points (x1 y1 x2 y2 : ℝ) (h : x1 = x2) :
  (y1 - y2) ^ 2 = 225 → 
  let side_length := |y1 - y2|
  (side_length ^ 2 : ℝ) = 225 := by
  sorry

/-- The specific case for the given coordinates -/
theorem square_area_specific_case : 
  let x1 : ℝ := 20
  let y1 : ℝ := 20
  let x2 : ℝ := 20
  let y2 : ℝ := 5
  (y1 - y2) ^ 2 = 225 ∧ 
  let side_length := |y1 - y2|
  (side_length ^ 2 : ℝ) = 225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_two_points_square_area_specific_case_l1705_170558


namespace NUMINAMATH_CALUDE_recipe_total_l1705_170564

theorem recipe_total (eggs : ℕ) (flour : ℕ) : 
  eggs = 60 → flour = eggs / 2 → eggs + flour = 90 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_l1705_170564


namespace NUMINAMATH_CALUDE_prime_power_sum_congruence_and_evenness_l1705_170540

theorem prime_power_sum_congruence_and_evenness (p q : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p ≠ q) :
  (p^q + q^p) % (p*q) = (p + q) % (p*q) ∧ 
  (p ≠ 2 → q ≠ 2 → Even ((p^q + q^p) / (p*q))) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_congruence_and_evenness_l1705_170540


namespace NUMINAMATH_CALUDE_inequality_proof_l1705_170577

theorem inequality_proof (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  12 / (w + x + y + z) ≤ 1/(w + x) + 1/(w + y) + 1/(w + z) + 1/(x + y) + 1/(x + z) + 1/(y + z) ∧
  1/(w + x) + 1/(w + y) + 1/(w + z) + 1/(x + y) + 1/(x + z) + 1/(y + z) ≤ 3/4 * (1/w + 1/x + 1/y + 1/z) :=
by sorry


end NUMINAMATH_CALUDE_inequality_proof_l1705_170577


namespace NUMINAMATH_CALUDE_charity_duck_race_money_raised_l1705_170581

/-- The amount of money raised in a charity rubber duck race -/
theorem charity_duck_race_money_raised
  (regular_price : ℚ)
  (large_price : ℚ)
  (regular_sold : ℕ)
  (large_sold : ℕ)
  (h1 : regular_price = 3)
  (h2 : large_price = 5)
  (h3 : regular_sold = 221)
  (h4 : large_sold = 185) :
  regular_price * regular_sold + large_price * large_sold = 1588 :=
by sorry

end NUMINAMATH_CALUDE_charity_duck_race_money_raised_l1705_170581


namespace NUMINAMATH_CALUDE_rug_coverage_area_l1705_170562

/-- Given three rugs with specified overlapping areas, calculate the total floor area covered -/
theorem rug_coverage_area (total_rug_area : ℝ) (two_layer_overlap : ℝ) (three_layer_overlap : ℝ) 
  (h1 : total_rug_area = 204)
  (h2 : two_layer_overlap = 24)
  (h3 : three_layer_overlap = 20) :
  total_rug_area - two_layer_overlap - 2 * three_layer_overlap = 140 := by
  sorry

end NUMINAMATH_CALUDE_rug_coverage_area_l1705_170562


namespace NUMINAMATH_CALUDE_f_lower_bound_l1705_170565

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + 2 * Real.exp (-x) + (a - 2) * x

theorem f_lower_bound (a : ℝ) :
  (∀ x > 0, f a x ≥ (a + 2) * Real.cos x) → a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_f_lower_bound_l1705_170565


namespace NUMINAMATH_CALUDE_five_dice_not_same_probability_l1705_170557

theorem five_dice_not_same_probability : 
  let n : ℕ := 5  -- number of dice
  let s : ℕ := 6  -- number of sides on each die
  let total_outcomes : ℕ := s^n
  let same_number_outcomes : ℕ := s
  let prob_not_same : ℚ := 1 - (same_number_outcomes : ℚ) / total_outcomes
  prob_not_same = 1295 / 1296 :=
by sorry

end NUMINAMATH_CALUDE_five_dice_not_same_probability_l1705_170557


namespace NUMINAMATH_CALUDE_pentagon_square_side_ratio_l1705_170570

/-- The ratio of the side length of a regular pentagon to the side length of a square 
    with the same perimeter -/
theorem pentagon_square_side_ratio : 
  ∀ (pentagon_side square_side : ℝ),
  pentagon_side > 0 → square_side > 0 →
  5 * pentagon_side = 20 →
  4 * square_side = 20 →
  pentagon_side / square_side = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_square_side_ratio_l1705_170570


namespace NUMINAMATH_CALUDE_knicks_win_probability_l1705_170572

/-- The probability of the Bulls winning a single game -/
def p : ℚ := 3/4

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The maximum number of games in the series -/
def max_games : ℕ := 2 * games_to_win - 1

/-- The probability of the Knicks winning the series in exactly 7 games -/
def knicks_win_in_seven : ℚ := 135/4096

theorem knicks_win_probability :
  knicks_win_in_seven = (Nat.choose 6 3 : ℚ) * (1 - p)^3 * p^3 * (1 - p) :=
sorry

end NUMINAMATH_CALUDE_knicks_win_probability_l1705_170572


namespace NUMINAMATH_CALUDE_fourth_power_sum_l1705_170598

theorem fourth_power_sum (a b c : ℝ) 
  (sum_1 : a + b + c = 1)
  (sum_2 : a^2 + b^2 + c^2 = 2)
  (sum_3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 25/6 := by
sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l1705_170598


namespace NUMINAMATH_CALUDE_population_trend_l1705_170589

theorem population_trend (P k : ℝ) (h1 : P > 0) (h2 : -1 < k) (h3 : k < 0) :
  ∀ n : ℕ, (P * (1 + k)^(n + 1)) < (P * (1 + k)^n) := by
  sorry

end NUMINAMATH_CALUDE_population_trend_l1705_170589


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l1705_170573

theorem intersection_points_on_circle :
  ∀ (x y : ℝ), 
    ((x + 2*y = 19 ∨ y + 2*x = 98) ∧ y = 1/x) →
    (x - 34)^2 + (y - 215/4)^2 = 49785/16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l1705_170573


namespace NUMINAMATH_CALUDE_one_hundred_ten_billion_scientific_notation_l1705_170513

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem one_hundred_ten_billion_scientific_notation :
  toScientificNotation 110000000000 = ScientificNotation.mk 1.1 11 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_one_hundred_ten_billion_scientific_notation_l1705_170513


namespace NUMINAMATH_CALUDE_intersection_and_coefficients_l1705_170548

def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def B : Set ℝ := {x | -3 < x ∧ x < 1}

theorem intersection_and_coefficients :
  (A ∩ B = {x | -1 < x ∧ x < 1}) ∧
  (∃ a b : ℝ, (∀ x : ℝ, x ∈ B ↔ 2*x^2 + a*x + b < 0) ∧ a = 3 ∧ b = 4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_coefficients_l1705_170548


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l1705_170537

theorem sqrt_difference_equals_seven_sqrt_two_over_six :
  Real.sqrt (9 / 2) - Real.sqrt (2 / 9) = 7 * Real.sqrt 2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_seven_sqrt_two_over_six_l1705_170537


namespace NUMINAMATH_CALUDE_cantaloupes_total_l1705_170512

/-- The number of cantaloupes grown by Fred -/
def fred_cantaloupes : ℕ := 38

/-- The number of cantaloupes grown by Tim -/
def tim_cantaloupes : ℕ := 44

/-- The total number of cantaloupes grown by Fred and Tim -/
def total_cantaloupes : ℕ := fred_cantaloupes + tim_cantaloupes

theorem cantaloupes_total : total_cantaloupes = 82 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_total_l1705_170512


namespace NUMINAMATH_CALUDE_candies_remaining_l1705_170500

-- Define the number of candies for each color
def red_candies : ℕ := 50
def yellow_candies : ℕ := 3 * red_candies - 35
def blue_candies : ℕ := (2 * yellow_candies) / 3
def green_candies : ℕ := 20
def purple_candies : ℕ := green_candies / 2
def silver_candies : ℕ := 10

-- Define the number of candies Carlos ate
def carlos_ate : ℕ := yellow_candies + green_candies / 2

-- Define the total number of candies
def total_candies : ℕ := red_candies + yellow_candies + blue_candies + green_candies + purple_candies + silver_candies

-- Theorem statement
theorem candies_remaining : total_candies - carlos_ate = 156 := by
  sorry

end NUMINAMATH_CALUDE_candies_remaining_l1705_170500


namespace NUMINAMATH_CALUDE_power_multiplication_l1705_170534

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1705_170534


namespace NUMINAMATH_CALUDE_negative_result_operation_only_A_is_negative_l1705_170516

theorem negative_result_operation : ℤ → Prop :=
  fun x => x < 0

theorem only_A_is_negative :
  negative_result_operation ((-1) + (-3)) ∧
  ¬negative_result_operation (6 - (-3)) ∧
  ¬negative_result_operation ((-3) * (-2)) ∧
  ¬negative_result_operation (0 / (-7)) :=
by
  sorry

end NUMINAMATH_CALUDE_negative_result_operation_only_A_is_negative_l1705_170516


namespace NUMINAMATH_CALUDE_six_students_five_lectures_l1705_170578

/-- The number of ways to assign students to lectures -/
def assignment_count (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: The number of ways to assign 6 students to 5 lectures is 5^6 -/
theorem six_students_five_lectures :
  assignment_count 6 5 = 5^6 := by
  sorry

end NUMINAMATH_CALUDE_six_students_five_lectures_l1705_170578


namespace NUMINAMATH_CALUDE_new_average_age_l1705_170517

theorem new_average_age
  (initial_students : ℕ)
  (initial_average : ℚ)
  (new_student_age : ℕ)
  (h1 : initial_students = 8)
  (h2 : initial_average = 15)
  (h3 : new_student_age = 17) :
  let total_age : ℚ := initial_students * initial_average + new_student_age
  let new_total_students : ℕ := initial_students + 1
  total_age / new_total_students = 137 / 9 :=
by sorry

end NUMINAMATH_CALUDE_new_average_age_l1705_170517


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_rational_square_minus_two_l1705_170514

theorem negation_of_existence (p : ℚ → Prop) : 
  (¬ ∃ x : ℚ, p x) ↔ (∀ x : ℚ, ¬ p x) := by sorry

theorem negation_of_rational_square_minus_two :
  (¬ ∃ x : ℚ, x^2 - 2 = 0) ↔ (∀ x : ℚ, x^2 - 2 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_rational_square_minus_two_l1705_170514


namespace NUMINAMATH_CALUDE_marks_vote_ratio_l1705_170510

theorem marks_vote_ratio (total_voters_first_area : ℕ) (win_percentage : ℚ) (total_votes : ℕ) : 
  total_voters_first_area = 100000 →
  win_percentage = 70 / 100 →
  total_votes = 210000 →
  (total_votes - (total_voters_first_area * win_percentage).floor) / 
  ((total_voters_first_area * win_percentage).floor) = 2 := by
  sorry

end NUMINAMATH_CALUDE_marks_vote_ratio_l1705_170510


namespace NUMINAMATH_CALUDE_jake_weight_proof_l1705_170521

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 152

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 212 - jake_weight

theorem jake_weight_proof :
  (jake_weight - 32 = 2 * sister_weight) ∧
  (jake_weight + sister_weight = 212) →
  jake_weight = 152 :=
by sorry

end NUMINAMATH_CALUDE_jake_weight_proof_l1705_170521


namespace NUMINAMATH_CALUDE_angle_inequality_l1705_170544

theorem angle_inequality (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 2 → x^2 * Real.cos θ - 2*x*(1 - x) + (2 - x)^2 * Real.sin θ > 0) →
  Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_l1705_170544


namespace NUMINAMATH_CALUDE_divisor_problem_l1705_170530

theorem divisor_problem (range_start : Nat) (range_end : Nat) (divisible_count : Nat) : 
  range_start = 10 → 
  range_end = 1000000 → 
  divisible_count = 111110 → 
  ∃ (d : Nat), d = 9 ∧ 
    (∀ n : Nat, range_start ≤ n ∧ n ≤ range_end → 
      (n % d = 0 ↔ ∃ k : Nat, k ≤ divisible_count ∧ n = range_start + (k - 1) * d)) :=
by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1705_170530


namespace NUMINAMATH_CALUDE_nested_square_root_value_l1705_170580

/-- Given that x is a real number satisfying x = √(2 + x), prove that x = 2 -/
theorem nested_square_root_value (x : ℝ) (h : x = Real.sqrt (2 + x)) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l1705_170580


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_reciprocal_squares_l1705_170529

theorem quadratic_roots_sum_of_reciprocal_squares :
  ∀ (r s : ℝ), 
    (2 * r^2 + 3 * r - 5 = 0) →
    (2 * s^2 + 3 * s - 5 = 0) →
    (r ≠ s) →
    (1 / r^2 + 1 / s^2 = 29 / 25) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_reciprocal_squares_l1705_170529


namespace NUMINAMATH_CALUDE_shuffleboard_games_total_l1705_170593

/-- Proves that the total number of games played is 32 given the conditions of the shuffleboard game. -/
theorem shuffleboard_games_total (jerry_wins dave_wins ken_wins : ℕ) 
  (h1 : ken_wins = dave_wins + 5)
  (h2 : dave_wins = jerry_wins + 3)
  (h3 : jerry_wins = 7) : 
  jerry_wins + dave_wins + ken_wins = 32 := by
  sorry

end NUMINAMATH_CALUDE_shuffleboard_games_total_l1705_170593


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1705_170590

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

theorem perpendicular_vectors (t : ℝ) :
  (a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) → t = -5 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1705_170590


namespace NUMINAMATH_CALUDE_ticket_sales_total_l1705_170592

/-- Calculates the total amount collected from ticket sales given the following conditions:
  * Adult ticket cost is $12
  * Child ticket cost is $4
  * Total number of tickets sold is 130
  * Number of adult tickets sold is 40
-/
theorem ticket_sales_total (adult_cost child_cost total_tickets adult_tickets : ℕ) : 
  adult_cost = 12 →
  child_cost = 4 →
  total_tickets = 130 →
  adult_tickets = 40 →
  adult_cost * adult_tickets + child_cost * (total_tickets - adult_tickets) = 840 :=
by
  sorry

#check ticket_sales_total

end NUMINAMATH_CALUDE_ticket_sales_total_l1705_170592


namespace NUMINAMATH_CALUDE_tic_tac_toe_rounds_l1705_170554

theorem tic_tac_toe_rounds (total_rounds : ℕ) (difference : ℕ) (william_wins harry_wins : ℕ) : 
  total_rounds = 15 → 
  difference = 5 → 
  william_wins = harry_wins + difference → 
  william_wins + harry_wins = total_rounds → 
  william_wins = 10 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_rounds_l1705_170554


namespace NUMINAMATH_CALUDE_zeros_in_intervals_l1705_170525

/-- A quadratic function -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem zeros_in_intervals (a b c m n p : ℝ) (h_a : a ≠ 0) (h_order : m < n ∧ n < p) :
  (∃ x y, m < x ∧ x < n ∧ n < y ∧ y < p ∧ 
    quadratic_function a b c x = 0 ∧ 
    quadratic_function a b c y = 0) ↔ 
  (quadratic_function a b c m) * (quadratic_function a b c n) < 0 ∧
  (quadratic_function a b c p) * (quadratic_function a b c n) < 0 :=
sorry

end NUMINAMATH_CALUDE_zeros_in_intervals_l1705_170525


namespace NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_5_6_l1705_170522

theorem tan_alpha_3_implies_fraction_eq_5_6 (α : ℝ) (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_implies_fraction_eq_5_6_l1705_170522


namespace NUMINAMATH_CALUDE_length_a_prime_b_prime_l1705_170591

/-- Given points A, B, and C, where A' and B' are the intersections of lines AC and BC with the line y = x respectively, the length of A'B' is (3√2)/10. -/
theorem length_a_prime_b_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 7) →
  B = (0, 14) →
  C = (3, 5) →
  (A'.1 = A'.2) →  -- A' is on y = x
  (B'.1 = B'.2) →  -- B' is on y = x
  (C.2 - A.2) / (C.1 - A.1) = (A'.2 - A.2) / (A'.1 - A.1) →  -- A' is on line AC
  (C.2 - B.2) / (C.1 - B.1) = (B'.2 - B.2) / (B'.1 - B.1) →  -- B' is on line BC
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = (3 * Real.sqrt 2) / 10 := by
sorry

end NUMINAMATH_CALUDE_length_a_prime_b_prime_l1705_170591


namespace NUMINAMATH_CALUDE_football_club_penalty_kicks_l1705_170511

/-- Calculates the total number of penalty kicks in a football club's shootout contest. -/
def total_penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  (total_players - goalies) * goalies

/-- Theorem stating that for a football club with 25 players including 4 goalies, 
    where each player takes a shot against each goalie, the total number of penalty kicks is 96. -/
theorem football_club_penalty_kicks :
  total_penalty_kicks 25 4 = 96 := by
  sorry

#eval total_penalty_kicks 25 4

end NUMINAMATH_CALUDE_football_club_penalty_kicks_l1705_170511


namespace NUMINAMATH_CALUDE_no_integers_between_sqrt_bounds_l1705_170535

theorem no_integers_between_sqrt_bounds (n : ℕ+) :
  ¬∃ (x y : ℕ+), (Real.sqrt n + Real.sqrt (n + 1) < Real.sqrt x + Real.sqrt y) ∧
                  (Real.sqrt x + Real.sqrt y < Real.sqrt (4 * n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_no_integers_between_sqrt_bounds_l1705_170535


namespace NUMINAMATH_CALUDE_fraction_count_l1705_170546

-- Define a function to check if an expression is a fraction
def is_fraction (expr : String) : Bool :=
  match expr with
  | "1/x" => true
  | "x^2+5x" => false
  | "1/2x" => false
  | "a/(3-2a)" => true
  | "3.14/π" => false
  | _ => false

-- Define the list of expressions
def expressions : List String := ["1/x", "x^2+5x", "1/2x", "a/(3-2a)", "3.14/π"]

-- Theorem statement
theorem fraction_count : (expressions.filter is_fraction).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_count_l1705_170546


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_half_l1705_170553

theorem sin_cos_difference_equals_neg_half : 
  Real.sin (119 * π / 180) * Real.cos (91 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_neg_half_l1705_170553


namespace NUMINAMATH_CALUDE_chicken_farm_proof_l1705_170543

/-- The number of chickens Michael has now -/
def initial_chickens : ℕ := 550

/-- The annual increase in the number of chickens -/
def annual_increase : ℕ := 150

/-- The number of years -/
def years : ℕ := 9

/-- The number of chickens after 9 years -/
def final_chickens : ℕ := 1900

/-- Theorem stating that the initial number of chickens plus the total increase over 9 years equals the final number of chickens -/
theorem chicken_farm_proof : 
  initial_chickens + (annual_increase * years) = final_chickens := by
  sorry


end NUMINAMATH_CALUDE_chicken_farm_proof_l1705_170543


namespace NUMINAMATH_CALUDE_perimeter_of_problem_pentagon_l1705_170585

/-- A pentagon ABCDE with given side lengths and a right angle -/
structure Pentagon :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DE : ℝ)
  (AE : ℝ)
  (right_angle_AED : AE^2 + DE^2 = AB^2 + BC^2 + DE^2)

/-- The perimeter of a pentagon -/
def perimeter (p : Pentagon) : ℝ :=
  p.AB + p.BC + p.CD + p.DE + p.AE

/-- The specific pentagon from the problem -/
def problem_pentagon : Pentagon :=
  { AB := 4
  , BC := 2
  , CD := 2
  , DE := 6
  , AE := 6
  , right_angle_AED := by sorry }

/-- Theorem: The perimeter of the problem pentagon is 14 + 6√2 -/
theorem perimeter_of_problem_pentagon :
  perimeter problem_pentagon = 14 + 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_problem_pentagon_l1705_170585


namespace NUMINAMATH_CALUDE_fruit_basket_count_l1705_170568

/-- The number of ways to choose k items from n identical items -/
def choose_with_repetition (n : ℕ) (k : ℕ) : ℕ := (n + k - 1).choose k

/-- The number of possible fruit baskets given the number of bananas and pears -/
def fruit_baskets (bananas : ℕ) (pears : ℕ) : ℕ :=
  (choose_with_repetition (bananas + 1) 1) * (choose_with_repetition (pears + 1) 1) - 1

theorem fruit_basket_count :
  fruit_baskets 6 9 = 69 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l1705_170568


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l1705_170550

theorem chess_tournament_participants (total_games : ℕ) : total_games = 378 →
  ∃! n : ℕ, n * (n - 1) / 2 = total_games :=
by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l1705_170550


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l1705_170571

-- Define the days of the week
inductive Day : Type
  | Sunday : Day
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day

def next_day (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

def days_after (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ m => next_day (days_after d m)

theorem tomorrow_is_saturday 
  (day_before_yesterday : Day)
  (h : days_after day_before_yesterday 5 = Day.Monday) :
  days_after day_before_yesterday 3 = Day.Saturday :=
by sorry

end NUMINAMATH_CALUDE_tomorrow_is_saturday_l1705_170571


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_calculation_l1705_170559

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width height water_depth : ℝ) : ℝ :=
  length * width + 2 * (length * water_depth) + 2 * (width * water_depth)

/-- Theorem stating that the wet surface area of the given cistern is 387.5 m² -/
theorem cistern_wet_surface_area_calculation :
  cistern_wet_surface_area 15 10 8 4.75 = 387.5 := by
  sorry

#eval cistern_wet_surface_area 15 10 8 4.75

end NUMINAMATH_CALUDE_cistern_wet_surface_area_calculation_l1705_170559


namespace NUMINAMATH_CALUDE_vector_properties_l1705_170596

def a : ℝ × ℝ := (1, 2)
def b (t : ℝ) : ℝ × ℝ := (-4, t)

theorem vector_properties :
  (∀ t : ℝ, (∃ k : ℝ, a = k • b t) → t = -8) ∧
  (∃ t_min : ℝ, ∀ t : ℝ, ‖a - b t‖ ≥ ‖a - b t_min‖ ∧ ‖a - b t_min‖ = 5) ∧
  (∀ t : ℝ, ‖a + b t‖ = ‖a - b t‖ → t = 2) ∧
  (∀ t : ℝ, (a • b t < 0) → t < 2) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l1705_170596


namespace NUMINAMATH_CALUDE_intersection_line_l1705_170518

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line : 
  ∀ (x y : ℝ), circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_l1705_170518


namespace NUMINAMATH_CALUDE_cow_price_problem_l1705_170502

/-- Given the total cost of cows and goats, the number of cows and goats, and the average price of a goat,
    calculate the average price of a cow. -/
def average_cow_price (total_cost : ℕ) (num_cows num_goats : ℕ) (avg_goat_price : ℕ) : ℕ :=
  (total_cost - num_goats * avg_goat_price) / num_cows

/-- Theorem: Given 2 cows and 10 goats with a total cost of 1500 rupees, 
    and an average price of 70 rupees per goat, the average price of a cow is 400 rupees. -/
theorem cow_price_problem : average_cow_price 1500 2 10 70 = 400 := by
  sorry

end NUMINAMATH_CALUDE_cow_price_problem_l1705_170502


namespace NUMINAMATH_CALUDE_min_value_fraction_l1705_170595

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a + b = 2) :
  (∃ (x : ℝ), ∀ (y : ℝ), (3*a - b) / (a^2 + 2*a*b - 3*b^2) ≥ x) ∧
  (∃ (z : ℝ), (3*z - (2-z)) / (z^2 + 2*z*(2-z) - 3*(2-z)^2) = (3 + Real.sqrt 5) / 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1705_170595


namespace NUMINAMATH_CALUDE_two_digit_number_difference_l1705_170575

/-- 
For a two-digit number where:
- The number is 26
- The product of the number and the sum of its digits is 208
Prove that the difference between the unit's digit and the 10's digit is 4.
-/
theorem two_digit_number_difference (n : ℕ) (h1 : n = 26) 
  (h2 : n * (n / 10 + n % 10) = 208) : n % 10 - n / 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_difference_l1705_170575


namespace NUMINAMATH_CALUDE_paper_width_covering_cube_l1705_170574

/-- Given a rectangular piece of paper covering a cube, prove the width of the paper. -/
theorem paper_width_covering_cube 
  (paper_length : ℝ) 
  (cube_volume : ℝ) 
  (h1 : paper_length = 48)
  (h2 : cube_volume = 8) : 
  ∃ (paper_width : ℝ), paper_width = 72 ∧ 
    paper_length * paper_width = 6 * (12 * (cube_volume ^ (1/3)))^2 :=
by sorry

end NUMINAMATH_CALUDE_paper_width_covering_cube_l1705_170574


namespace NUMINAMATH_CALUDE_meet_on_same_side_time_l1705_170588

/-- The time when two points moving on a square meet on the same side for the first time -/
def time_to_meet_on_same_side (side_length : ℝ) (speed_A : ℝ) (speed_B : ℝ) : ℝ :=
  35

/-- Theorem stating that the time to meet on the same side is 35 seconds under given conditions -/
theorem meet_on_same_side_time :
  time_to_meet_on_same_side 100 5 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_meet_on_same_side_time_l1705_170588


namespace NUMINAMATH_CALUDE_simplest_square_root_l1705_170549

theorem simplest_square_root :
  let options : List ℝ := [Real.sqrt 5, Real.sqrt 4, Real.sqrt 12, Real.sqrt (1/2)]
  ∀ x ∈ options, x ≠ Real.sqrt 5 → ∃ y : ℝ, y * y = x ∧ y ≠ x :=
by sorry

end NUMINAMATH_CALUDE_simplest_square_root_l1705_170549


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l1705_170599

theorem quadratic_roots_expression (a b : ℝ) : 
  (a^2 - a - 1 = 0) → (b^2 - b - 1 = 0) → (3*a^2 + 2*b^2 - 3*a - 2*b = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l1705_170599


namespace NUMINAMATH_CALUDE_five_T_three_equals_38_l1705_170532

-- Define the operation T
def T (a b : ℝ) : ℝ := 4 * a + 6 * b

-- Theorem to prove
theorem five_T_three_equals_38 : T 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_five_T_three_equals_38_l1705_170532


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1705_170555

theorem quadratic_solution_difference_squared :
  ∀ d e : ℝ,
  (5 * d^2 + 20 * d - 55 = 0) →
  (5 * e^2 + 20 * e - 55 = 0) →
  (d - e)^2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1705_170555


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1705_170579

theorem linear_equation_solution (m : ℝ) : 
  (3 : ℝ) - m * (1 : ℝ) = 1 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1705_170579


namespace NUMINAMATH_CALUDE_largest_initial_number_l1705_170560

/-- Represents a sequence of five additions -/
structure FiveAdditions (n : ℕ) :=
  (a₁ a₂ a₃ a₄ a₅ : ℕ)
  (sum_eq : n + a₁ + a₂ + a₃ + a₄ + a₅ = 100)
  (not_div₁ : ¬(n % a₁ = 0))
  (not_div₂ : ¬((n + a₁) % a₂ = 0))
  (not_div₃ : ¬((n + a₁ + a₂) % a₃ = 0))
  (not_div₄ : ¬((n + a₁ + a₂ + a₃) % a₄ = 0))
  (not_div₅ : ¬((n + a₁ + a₂ + a₃ + a₄) % a₅ = 0))

/-- The main theorem stating that 89 is the largest initial number -/
theorem largest_initial_number :
  (∃ (f : FiveAdditions 89), True) ∧
  (∀ n > 89, ¬∃ (f : FiveAdditions n), True) :=
sorry

end NUMINAMATH_CALUDE_largest_initial_number_l1705_170560


namespace NUMINAMATH_CALUDE_half_month_days_l1705_170536

/-- Prove that given a 30-day month with specific mean profits, each half of the month contains 15 days -/
theorem half_month_days (total_days : ℕ) (mean_profit : ℚ) (first_half_mean : ℚ) (second_half_mean : ℚ) :
  total_days = 30 ∧ 
  mean_profit = 350 ∧ 
  first_half_mean = 275 ∧ 
  second_half_mean = 425 →
  ∃ (half_days : ℕ), half_days = 15 ∧ total_days = 2 * half_days :=
by sorry

end NUMINAMATH_CALUDE_half_month_days_l1705_170536


namespace NUMINAMATH_CALUDE_simplify_expression_l1705_170541

theorem simplify_expression (a b : ℝ) : 5 * a + 2 * b + (a - 3 * b) = 6 * a - b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1705_170541


namespace NUMINAMATH_CALUDE_notebook_price_l1705_170501

theorem notebook_price :
  ∀ (s n c : ℕ),
  s > 18 →
  s ≤ 36 →
  c > n →
  s * n * c = 990 →
  c = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_price_l1705_170501


namespace NUMINAMATH_CALUDE_steves_matching_socks_l1705_170597

theorem steves_matching_socks (total_socks : ℕ) (mismatching_socks : ℕ) 
  (h1 : total_socks = 25) 
  (h2 : mismatching_socks = 17) : 
  (total_socks - mismatching_socks) / 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_steves_matching_socks_l1705_170597


namespace NUMINAMATH_CALUDE_min_cubes_for_3x9x5_hollow_block_l1705_170519

/-- The minimum number of cubes needed to create a hollow block -/
def min_cubes_for_hollow_block (length width depth : ℕ) : ℕ :=
  length * width * depth - (length - 2) * (width - 2) * (depth - 2)

/-- Theorem stating that the minimum number of cubes for a 3x9x5 hollow block is 114 -/
theorem min_cubes_for_3x9x5_hollow_block :
  min_cubes_for_hollow_block 3 9 5 = 114 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_3x9x5_hollow_block_l1705_170519


namespace NUMINAMATH_CALUDE_problem1_l1705_170576

theorem problem1 (x y : ℝ) : (-3 * x * y)^2 * (4 * x^2) = 36 * x^4 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l1705_170576


namespace NUMINAMATH_CALUDE_prism_faces_l1705_170531

theorem prism_faces (E V : ℕ) (h : E + V = 30) : ∃ (F : ℕ), F = 8 ∧ F + V = E + 2 := by
  sorry

end NUMINAMATH_CALUDE_prism_faces_l1705_170531


namespace NUMINAMATH_CALUDE_physics_class_size_l1705_170523

theorem physics_class_size :
  ∀ (boys_biology girls_biology students_physics : ℕ),
    girls_biology = 3 * boys_biology →
    boys_biology = 25 →
    students_physics = 2 * (boys_biology + girls_biology) →
    students_physics = 200 := by
  sorry

end NUMINAMATH_CALUDE_physics_class_size_l1705_170523


namespace NUMINAMATH_CALUDE_c_most_suitable_l1705_170505

-- Define the structure for an athlete
structure Athlete where
  name : String
  average : ℝ
  variance : ℝ

-- Define the list of athletes
def athletes : List Athlete := [
  ⟨"A", 169, 6.0⟩,
  ⟨"B", 168, 17.3⟩,
  ⟨"C", 169, 5.0⟩,
  ⟨"D", 168, 19.5⟩
]

-- Function to determine if an athlete is suitable
def isSuitable (a : Athlete) : Prop :=
  ∀ b ∈ athletes, 
    a.average ≥ b.average ∧ 
    (a.average = b.average → a.variance ≤ b.variance)

-- Theorem stating that C is the most suitable candidate
theorem c_most_suitable : 
  ∃ c ∈ athletes, c.name = "C" ∧ isSuitable c :=
sorry

end NUMINAMATH_CALUDE_c_most_suitable_l1705_170505


namespace NUMINAMATH_CALUDE_basketball_cricket_students_l1705_170584

theorem basketball_cricket_students (B C : Finset Nat) : 
  (B.card = 7) → 
  (C.card = 8) → 
  ((B ∩ C).card = 5) → 
  ((B ∪ C).card = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_cricket_students_l1705_170584


namespace NUMINAMATH_CALUDE_second_group_men_count_l1705_170567

/-- The work rate of one man -/
def man_rate : ℝ := sorry

/-- The work rate of one woman -/
def woman_rate : ℝ := sorry

/-- The number of men in the second group -/
def x : ℕ := sorry

theorem second_group_men_count : x = 6 :=
  sorry

end NUMINAMATH_CALUDE_second_group_men_count_l1705_170567


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l1705_170547

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical_conversion 
  (x y z : ℝ) 
  (h_x : x = 3) 
  (h_y : y = -3 * Real.sqrt 3) 
  (h_z : z = 4) :
  ∃ (r θ : ℝ), 
    r > 0 ∧ 
    0 ≤ θ ∧ θ < 2 * Real.pi ∧
    r = 6 ∧ 
    θ = 5 * Real.pi / 3 ∧
    r * (Real.cos θ) = x ∧
    r * (Real.sin θ) = y ∧
    z = 4 :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_conversion_l1705_170547


namespace NUMINAMATH_CALUDE_odd_nines_composite_l1705_170582

theorem odd_nines_composite (k : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 10^(2*k) - 9 = a * b :=
sorry

end NUMINAMATH_CALUDE_odd_nines_composite_l1705_170582


namespace NUMINAMATH_CALUDE_cookie_scaling_l1705_170587

/-- Given a recipe for cookies, calculate the required ingredients for a larger batch -/
theorem cookie_scaling (base_cookies : ℕ) (target_cookies : ℕ) 
  (base_flour : ℚ) (base_sugar : ℚ) 
  (target_flour : ℚ) (target_sugar : ℚ) : 
  base_cookies > 0 → 
  (target_flour = (target_cookies : ℚ) / base_cookies * base_flour) ∧ 
  (target_sugar = (target_cookies : ℚ) / base_cookies * base_sugar) →
  (base_cookies = 40 ∧ 
   base_flour = 3 ∧ 
   base_sugar = 1 ∧ 
   target_cookies = 200) →
  (target_flour = 15 ∧ target_sugar = 5) := by
  sorry

end NUMINAMATH_CALUDE_cookie_scaling_l1705_170587


namespace NUMINAMATH_CALUDE_three_dozens_equals_42_l1705_170569

/-- Calculates the total number of flowers a customer receives when buying dozens of flowers with a free flower promotion. -/
def totalFlowers (dozens : ℕ) : ℕ :=
  let boughtFlowers := dozens * 12
  let freeFlowers := dozens * 2
  boughtFlowers + freeFlowers

/-- Theorem stating that buying 3 dozens of flowers results in 42 total flowers. -/
theorem three_dozens_equals_42 :
  totalFlowers 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_three_dozens_equals_42_l1705_170569


namespace NUMINAMATH_CALUDE_probability_not_special_number_l1705_170566

def is_perfect_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a^b = n

def is_power_of_three_halves (n : ℕ) : Prop :=
  ∃ (k : ℕ), (3/2)^k = n

def count_special_numbers : ℕ := 20

theorem probability_not_special_number :
  (200 - count_special_numbers) / 200 = 9 / 10 := by
  sorry

#check probability_not_special_number

end NUMINAMATH_CALUDE_probability_not_special_number_l1705_170566


namespace NUMINAMATH_CALUDE_solution_difference_squared_l1705_170556

theorem solution_difference_squared (α β : ℝ) : 
  α ≠ β ∧ α^2 = 2*α + 1 ∧ β^2 = 2*β + 1 → (α - β)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_solution_difference_squared_l1705_170556


namespace NUMINAMATH_CALUDE_rectangle_diagonal_corners_l1705_170533

/-- Represents a domino on a rectangular grid -/
structure Domino where
  x : ℕ
  y : ℕ
  horizontal : Bool

/-- Represents a diagonal in a domino -/
structure Diagonal where
  domino : Domino
  startCorner : Bool  -- true if the diagonal starts at the top-left or bottom-right corner

/-- Represents a rectangular grid filled with dominoes -/
structure RectangularGrid where
  width : ℕ
  height : ℕ
  dominoes : List Domino
  diagonals : List Diagonal

/-- Check if two diagonals have common endpoints -/
def diagonalsShareEndpoint (d1 d2 : Diagonal) : Bool := sorry

/-- Check if a point is a corner of the rectangle -/
def isRectangleCorner (x y : ℕ) (grid : RectangularGrid) : Bool := sorry

/-- Check if a point is an endpoint of a diagonal -/
def isDiagonalEndpoint (x y : ℕ) (diagonal : Diagonal) : Bool := sorry

/-- The main theorem -/
theorem rectangle_diagonal_corners (grid : RectangularGrid) :
  (∀ d1 d2 : Diagonal, d1 ∈ grid.diagonals → d2 ∈ grid.diagonals → d1 ≠ d2 → ¬(diagonalsShareEndpoint d1 d2)) →
  (∃! (n : ℕ), n = 2 ∧ 
    ∃ (corners : List (ℕ × ℕ)), corners.length = n ∧
      (∀ (x y : ℕ), (x, y) ∈ corners ↔ 
        (isRectangleCorner x y grid ∧ 
         ∃ d : Diagonal, d ∈ grid.diagonals ∧ isDiagonalEndpoint x y d))) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_corners_l1705_170533


namespace NUMINAMATH_CALUDE_smallest_k_for_divisible_difference_l1705_170528

theorem smallest_k_for_divisible_difference : ∃ (k : ℕ), k > 0 ∧
  (∀ (M : Finset ℕ), M ⊆ Finset.range 20 → M.card ≥ k →
    ∃ (a b c d : ℕ), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
      20 ∣ (a - b + c - d)) ∧
  (∀ (k' : ℕ), k' < k →
    ∃ (M : Finset ℕ), M ⊆ Finset.range 20 ∧ M.card = k' ∧
      ∀ (a b c d : ℕ), a ∈ M → b ∈ M → c ∈ M → d ∈ M →
        a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
        ¬(20 ∣ (a - b + c - d))) ∧
  k = 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_divisible_difference_l1705_170528


namespace NUMINAMATH_CALUDE_one_third_of_number_l1705_170520

theorem one_third_of_number (x : ℝ) : 
  (1 / 3 : ℝ) * x = 130.00000000000003 → x = 390.0000000000001 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_number_l1705_170520


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l1705_170526

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dims : BoxDimensions) : ℕ :=
  dims.length * dims.width * dims.height

/-- Calculates the maximum number of smaller boxes that can fit in a larger box -/
def maxBoxesFit (largeBox : BoxDimensions) (smallBox : BoxDimensions) : ℕ :=
  (boxVolume largeBox) / (boxVolume smallBox)

/-- The dimensions of the carton -/
def cartonDims : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDims : BoxDimensions :=
  { length := 7, width := 6, height := 10 }

/-- Theorem stating that the maximum number of soap boxes that can fit in the carton is 150 -/
theorem max_soap_boxes_in_carton :
  maxBoxesFit cartonDims soapBoxDims = 150 := by
  sorry


end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l1705_170526


namespace NUMINAMATH_CALUDE_assembly_line_average_output_l1705_170504

/-- Represents the production data for an assembly line phase -/
structure ProductionPhase where
  cogs_produced : ℕ
  production_rate : ℕ

/-- Calculates the time taken for a production phase in hours -/
def time_taken (phase : ProductionPhase) : ℚ :=
  phase.cogs_produced / phase.production_rate

/-- Calculates the overall average output for two production phases -/
def overall_average_output (phase1 phase2 : ProductionPhase) : ℚ :=
  (phase1.cogs_produced + phase2.cogs_produced) / (time_taken phase1 + time_taken phase2)

/-- Theorem stating that the overall average output is 30 cogs per hour -/
theorem assembly_line_average_output :
  let phase1 : ProductionPhase := { cogs_produced := 60, production_rate := 20 }
  let phase2 : ProductionPhase := { cogs_produced := 60, production_rate := 60 }
  overall_average_output phase1 phase2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_assembly_line_average_output_l1705_170504


namespace NUMINAMATH_CALUDE_square_root_sum_l1705_170552

theorem square_root_sum (x : ℝ) (h : x + x⁻¹ = 3) :
  Real.sqrt x + (Real.sqrt x)⁻¹ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_l1705_170552


namespace NUMINAMATH_CALUDE_stratified_sample_grade12_l1705_170539

/-- Calculates the number of grade 12 students to be selected in a stratified sample -/
theorem stratified_sample_grade12 (total : ℕ) (grade10 : ℕ) (grade11 : ℕ) (sample_size : ℕ) :
  total = 1500 →
  grade10 = 550 →
  grade11 = 450 →
  sample_size = 300 →
  (sample_size * (total - grade10 - grade11)) / total = 100 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_grade12_l1705_170539


namespace NUMINAMATH_CALUDE_intersection_point_of_lines_l1705_170515

theorem intersection_point_of_lines : ∃! p : ℚ × ℚ, 
  (3 * p.2 = -2 * p.1 + 6) ∧ (4 * p.2 = 3 * p.1 - 4) ∧ 
  p = (36/17, 10/17) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_of_lines_l1705_170515


namespace NUMINAMATH_CALUDE_percentage_commutation_l1705_170538

theorem percentage_commutation (n : ℝ) (h : 0.3 * (0.4 * n) = 36) :
  0.4 * (0.3 * n) = 0.3 * (0.4 * n) := by
  sorry

end NUMINAMATH_CALUDE_percentage_commutation_l1705_170538


namespace NUMINAMATH_CALUDE_work_earnings_equality_l1705_170551

theorem work_earnings_equality (t : ℚ) : 
  (t + 3) * (3 * t - 1) = (3 * t - 7) * (t + 4) + 5 → t = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equality_l1705_170551


namespace NUMINAMATH_CALUDE_floor_times_self_equals_108_l1705_170545

theorem floor_times_self_equals_108 :
  ∃! (x : ℝ), (⌊x⌋ : ℝ) * x = 108 ∧ x = 10.8 := by sorry

end NUMINAMATH_CALUDE_floor_times_self_equals_108_l1705_170545


namespace NUMINAMATH_CALUDE_complex_sum_equals_negative_ten_i_l1705_170583

/-- Prove that the sum of complex numbers (5-5i)+(-2-i)-(3+4i) equals -10i -/
theorem complex_sum_equals_negative_ten_i : (5 - 5*I) + (-2 - I) - (3 + 4*I) = -10*I := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_negative_ten_i_l1705_170583


namespace NUMINAMATH_CALUDE_exterior_angles_sum_360_l1705_170509

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  /-- The number of sides in the polygon. -/
  sides : ℕ
  /-- Assumption that the polygon has at least 3 sides. -/
  sides_ge_three : sides ≥ 3

/-- The sum of interior angles of a polygon. -/
def sum_of_interior_angles (p : Polygon) : ℝ := sorry

/-- The sum of exterior angles of a polygon. -/
def sum_of_exterior_angles (p : Polygon) : ℝ := sorry

/-- Theorem: For any polygon, if the sum of its interior angles is 1440°, 
    then the sum of its exterior angles is 360°. -/
theorem exterior_angles_sum_360 (p : Polygon) :
  sum_of_interior_angles p = 1440 → sum_of_exterior_angles p = 360 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angles_sum_360_l1705_170509


namespace NUMINAMATH_CALUDE_interest_rate_proof_l1705_170506

/-- Given a principal amount, time, and the difference between compound and simple interest,
    prove that the interest rate is 25%. -/
theorem interest_rate_proof (P t : ℝ) (diff : ℝ) : 
  P = 3600 → t = 2 → diff = 225 →
  ∃ r : ℝ, r = 25 ∧ 
    P * ((1 + r / 100) ^ t - 1) - (P * r * t / 100) = diff :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_proof_l1705_170506


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l1705_170503

theorem sqrt_fraction_equality : 
  (2 * Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l1705_170503


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1705_170586

theorem polynomial_factorization (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1705_170586


namespace NUMINAMATH_CALUDE_arithmetic_sum_proof_l1705_170507

/-- 
Given an arithmetic sequence with:
- first term a₁ = k² + 1
- common difference d = 1
- number of terms n = 2k + 1

Prove that the sum of the first 2k + 1 terms is k³ + (k + 1)³
-/
theorem arithmetic_sum_proof (k : ℕ) : 
  let a₁ : ℕ := k^2 + 1
  let d : ℕ := 1
  let n : ℕ := 2 * k + 1
  let S := n * (2 * a₁ + (n - 1) * d) / 2
  S = k^3 + (k + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_proof_l1705_170507
