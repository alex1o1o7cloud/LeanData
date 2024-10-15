import Mathlib

namespace NUMINAMATH_CALUDE_percentage_relationship_l736_73641

theorem percentage_relationship (x y z : ℝ) (h1 : y = 0.7 * z) (h2 : x = 0.84 * z) :
  x = y * 1.2 :=
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l736_73641


namespace NUMINAMATH_CALUDE_fraction_power_equality_l736_73695

theorem fraction_power_equality (a b : ℝ) (m : ℤ) (ha : a > 0) (hb : b ≠ 0) :
  (b / a) ^ m = a ^ (-m) * b ^ m := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l736_73695


namespace NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l736_73683

theorem quadratic_reciprocal_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x * y = 1) ↔ c = a :=
sorry

end NUMINAMATH_CALUDE_quadratic_reciprocal_roots_l736_73683


namespace NUMINAMATH_CALUDE_equation_holds_for_all_n_l736_73688

-- Define α as the positive root of x^2 - 1989x - 1 = 0
noncomputable def α : ℝ := (1989 + Real.sqrt (1989^2 + 4)) / 2

-- Define the equation to be proven
def equation (n : ℕ) : Prop :=
  ⌊α * n + 1989 * α * ⌊α * n⌋⌋ = 1989 * n + (1989^2 + 1) * ⌊α * n⌋

-- Theorem statement
theorem equation_holds_for_all_n : ∀ n : ℕ, equation n := by sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_n_l736_73688


namespace NUMINAMATH_CALUDE_darwin_food_expense_l736_73670

theorem darwin_food_expense (initial_amount : ℚ) (gas_fraction : ℚ) (remaining : ℚ) : 
  initial_amount = 600 →
  gas_fraction = 1/3 →
  remaining = 300 →
  (initial_amount - gas_fraction * initial_amount - remaining) / (initial_amount - gas_fraction * initial_amount) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_darwin_food_expense_l736_73670


namespace NUMINAMATH_CALUDE_allowance_theorem_l736_73665

def initial_allowance : ℚ := 12

def first_week_spending (allowance : ℚ) : ℚ := allowance / 3

def second_week_spending (remaining : ℚ) : ℚ := remaining / 4

def final_amount (allowance : ℚ) : ℚ :=
  let after_first_week := allowance - first_week_spending allowance
  after_first_week - second_week_spending after_first_week

theorem allowance_theorem : final_amount initial_allowance = 6 := by
  sorry

end NUMINAMATH_CALUDE_allowance_theorem_l736_73665


namespace NUMINAMATH_CALUDE_kylie_final_coins_l736_73608

/-- Calculates the number of US coins Kylie has left after converting all coins and giving some away --/
def kylie_coins_left (initial_us : ℝ) (euro : ℝ) (canadian : ℝ) (given_away : ℝ) 
  (euro_to_us : ℝ) (canadian_to_us : ℝ) : ℝ :=
  initial_us + euro * euro_to_us + canadian * canadian_to_us - given_away

/-- Theorem stating that Kylie is left with 15.58 US coins --/
theorem kylie_final_coins : 
  kylie_coins_left 15 13 8 21 1.18 0.78 = 15.58 := by
  sorry

end NUMINAMATH_CALUDE_kylie_final_coins_l736_73608


namespace NUMINAMATH_CALUDE_no_nonzero_perfect_square_in_sequence_l736_73680

theorem no_nonzero_perfect_square_in_sequence
  (a b : ℕ → ℤ)
  (h1 : ∀ k, b k = a k + 9)
  (h2 : ∀ k, a (k + 1) = 8 * b k + 8)
  (h3 : ∃ k, a k = 1988 ∨ b k = 1988) :
  ∀ k n, n ≠ 0 → a k ≠ n^2 :=
by sorry

end NUMINAMATH_CALUDE_no_nonzero_perfect_square_in_sequence_l736_73680


namespace NUMINAMATH_CALUDE_josh_marbles_calculation_l736_73684

theorem josh_marbles_calculation (initial_marbles : ℕ) : 
  initial_marbles = 16 → 
  (initial_marbles * 3 * 3 / 4 : ℕ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_calculation_l736_73684


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l736_73610

theorem smallest_right_triangle_area :
  let a : ℝ := 7
  let b : ℝ := 8
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let area : ℝ := (1/2) * a * Real.sqrt (c^2 - a^2)
  area = (7 * Real.sqrt 15) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l736_73610


namespace NUMINAMATH_CALUDE_negation_relationship_l736_73612

theorem negation_relationship (x : ℝ) : 
  (¬(0 < x ∧ x < 2) → ¬(1/x ≥ 1)) ∧ ¬(¬(1/x ≥ 1) → ¬(0 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_negation_relationship_l736_73612


namespace NUMINAMATH_CALUDE_rates_sum_of_squares_l736_73613

/-- Represents the rates for running, bicycling, and roller-skating -/
structure Rates where
  running : ℕ
  bicycling : ℕ
  roller_skating : ℕ

/-- Tom's total distance -/
def tom_distance (r : Rates) : ℕ := 3 * r.running + 4 * r.bicycling + 2 * r.roller_skating

/-- Jerry's total distance -/
def jerry_distance (r : Rates) : ℕ := 3 * r.running + 6 * r.bicycling + 2 * r.roller_skating

/-- Sum of squares of rates -/
def sum_of_squares (r : Rates) : ℕ := r.running^2 + r.bicycling^2 + r.roller_skating^2

theorem rates_sum_of_squares :
  ∃ r : Rates,
    tom_distance r = 104 ∧
    jerry_distance r = 140 ∧
    sum_of_squares r = 440 := by
  sorry

end NUMINAMATH_CALUDE_rates_sum_of_squares_l736_73613


namespace NUMINAMATH_CALUDE_smallest_inexpressible_is_eleven_l736_73673

def expressible (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_inexpressible_is_eleven :
  (∀ m < 11, expressible m) ∧ ¬expressible 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_inexpressible_is_eleven_l736_73673


namespace NUMINAMATH_CALUDE_path_length_is_twelve_l736_73674

/-- A right triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  side_a : ℝ
  side_b : ℝ
  hypotenuse : ℝ
  is_right : side_a^2 + side_b^2 = hypotenuse^2
  side_values : side_a = 9 ∧ side_b = 12 ∧ hypotenuse = 15

/-- A circle rolling inside the triangle -/
structure RollingCircle where
  radius : ℝ
  radius_value : radius = 2

/-- The path traced by the center of the rolling circle -/
def path_length (t : RightTriangle) (c : RollingCircle) : ℝ := 
  t.side_a + t.side_b + t.hypotenuse - 2 * (t.side_a + t.side_b + t.hypotenuse - 6 * c.radius)

/-- Theorem stating that the path length is 12 -/
theorem path_length_is_twelve (t : RightTriangle) (c : RollingCircle) : 
  path_length t c = 12 := by sorry

end NUMINAMATH_CALUDE_path_length_is_twelve_l736_73674


namespace NUMINAMATH_CALUDE_diagonals_bisect_angles_and_parallel_implies_parallelogram_diagonals_bisect_area_implies_parallelogram_l736_73644

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define diagonals
def diagonal1 (q : Quadrilateral) : Line := sorry
def diagonal2 (q : Quadrilateral) : Line := sorry

-- Define the property of diagonals bisecting each other's interior angles
def diagonals_bisect_angles (q : Quadrilateral) : Prop := sorry

-- Define the property of diagonals being parallel
def diagonals_parallel (q : Quadrilateral) : Prop := sorry

-- Define the property of diagonals bisecting the area
def diagonals_bisect_area (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem diagonals_bisect_angles_and_parallel_implies_parallelogram 
  (q : Quadrilateral) (h1 : diagonals_bisect_angles q) (h2 : diagonals_parallel q) : 
  is_parallelogram q := sorry

-- Theorem 2
theorem diagonals_bisect_area_implies_parallelogram 
  (q : Quadrilateral) (h : diagonals_bisect_area q) : 
  is_parallelogram q := sorry

end NUMINAMATH_CALUDE_diagonals_bisect_angles_and_parallel_implies_parallelogram_diagonals_bisect_area_implies_parallelogram_l736_73644


namespace NUMINAMATH_CALUDE_max_b_in_box_l736_73698

theorem max_b_in_box (a b c : ℕ) : 
  a * b * c = 360 →
  1 < c →
  c < b →
  b < a →
  b ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_b_in_box_l736_73698


namespace NUMINAMATH_CALUDE_polygon_sides_l736_73606

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l736_73606


namespace NUMINAMATH_CALUDE_batting_average_increase_l736_73631

theorem batting_average_increase (current_average : ℚ) (matches_played : ℕ) (new_average : ℚ) : 
  current_average = 52 →
  matches_played = 12 →
  new_average = 54 →
  (new_average * (matches_played + 1) - current_average * matches_played : ℚ) = 78 := by
sorry

end NUMINAMATH_CALUDE_batting_average_increase_l736_73631


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l736_73659

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₄ = 17 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l736_73659


namespace NUMINAMATH_CALUDE_goldfish_feeding_l736_73660

/-- Given that one scoop of fish food can feed 8 goldfish, 
    prove that 4 scoops can feed 32 goldfish -/
theorem goldfish_feeding (scoop_capacity : ℕ) (num_scoops : ℕ) : 
  scoop_capacity = 8 → num_scoops = 4 → num_scoops * scoop_capacity = 32 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_feeding_l736_73660


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l736_73634

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) ∧
  (a 2 + a 6) / 2 = 5 ∧
  (a 3 + a 7) / 2 = 7

/-- The general term of the arithmetic sequence satisfying the given conditions -/
theorem arithmetic_sequence_general_term (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  ∀ n : ℕ, a n = 2 * n - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l736_73634


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l736_73681

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Icc 0 2, 
    ∀ y ∈ Set.Icc 0 2, 
      x < y → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l736_73681


namespace NUMINAMATH_CALUDE_cosine_rule_with_ratio_l736_73630

theorem cosine_rule_with_ratio (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ratio : ∃ (k : ℝ), a = 2*k ∧ b = 3*k ∧ c = 4*k) : 
  (a^2 + b^2 - c^2) / (2*a*b) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_rule_with_ratio_l736_73630


namespace NUMINAMATH_CALUDE_divisible_by_four_l736_73625

theorem divisible_by_four (x : Nat) : 
  x < 10 → (3280 + x).mod 4 = 0 ↔ x = 0 ∨ x = 2 ∨ x = 4 ∨ x = 6 ∨ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_four_l736_73625


namespace NUMINAMATH_CALUDE_expression_simplification_l736_73691

theorem expression_simplification :
  -2 * Real.sqrt 2 + 2^(-(1/2 : ℝ)) + 1 / (Real.sqrt 2 + 1) + (Real.sqrt 2 - 1)^(0 : ℝ) = -(Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l736_73691


namespace NUMINAMATH_CALUDE_survey_result_survey_result_proof_l736_73632

theorem survey_result (total_surveyed : ℕ) 
  (electrical_fire_believers : ℕ) 
  (hantavirus_believers : ℕ) : Prop :=
  let electrical_fire_percentage : ℚ := 754 / 1000
  let hantavirus_percentage : ℚ := 523 / 1000
  (electrical_fire_believers : ℚ) / (total_surveyed : ℚ) = electrical_fire_percentage ∧
  (hantavirus_believers : ℚ) / (electrical_fire_believers : ℚ) = hantavirus_percentage ∧
  hantavirus_believers = 31 →
  total_surveyed = 78

theorem survey_result_proof : survey_result 78 59 31 :=
sorry

end NUMINAMATH_CALUDE_survey_result_survey_result_proof_l736_73632


namespace NUMINAMATH_CALUDE_range_of_a_l736_73677

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else Real.log x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) →
  (-1 ≤ a ∧ a < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l736_73677


namespace NUMINAMATH_CALUDE_notebook_boxes_l736_73652

theorem notebook_boxes (notebooks_per_box : ℕ) (total_notebooks : ℕ) (h1 : notebooks_per_box = 9) (h2 : total_notebooks = 27) :
  total_notebooks / notebooks_per_box = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_boxes_l736_73652


namespace NUMINAMATH_CALUDE_min_y_over_x_on_ellipse_l736_73624

theorem min_y_over_x_on_ellipse :
  ∀ x y : ℝ, 4 * (x - 2)^2 + y^2 = 4 →
  ∃ k : ℝ, k = -2/3 * Real.sqrt 3 ∧ ∀ z : ℝ, z = y / x → z ≥ k := by
  sorry

end NUMINAMATH_CALUDE_min_y_over_x_on_ellipse_l736_73624


namespace NUMINAMATH_CALUDE_prob_man_satisfied_correct_expected_satisfied_men_correct_l736_73616

/-- Represents the number of men in the seating arrangement -/
def num_men : ℕ := 50

/-- Represents the number of women in the seating arrangement -/
def num_women : ℕ := 50

/-- Represents the total number of people in the seating arrangement -/
def total_people : ℕ := num_men + num_women

/-- Represents the probability of a specific man being satisfied -/
def prob_man_satisfied : ℚ := 25 / 33

/-- Represents the expected number of satisfied men -/
def expected_satisfied_men : ℚ := 1250 / 33

/-- Theorem stating the probability of a specific man being satisfied -/
theorem prob_man_satisfied_correct : 
  prob_man_satisfied = 1 - (num_men - 1) / (total_people - 1) * (num_men - 2) / (total_people - 2) :=
sorry

/-- Theorem stating the expected number of satisfied men -/
theorem expected_satisfied_men_correct : 
  expected_satisfied_men = num_men * prob_man_satisfied :=
sorry

end NUMINAMATH_CALUDE_prob_man_satisfied_correct_expected_satisfied_men_correct_l736_73616


namespace NUMINAMATH_CALUDE_sally_bread_consumption_l736_73654

theorem sally_bread_consumption :
  let saturday_sandwiches : ℕ := 2
  let sunday_sandwiches : ℕ := 1
  let bread_per_sandwich : ℕ := 2
  let total_sandwiches := saturday_sandwiches + sunday_sandwiches
  let total_bread := total_sandwiches * bread_per_sandwich
  total_bread = 6 := by
  sorry

end NUMINAMATH_CALUDE_sally_bread_consumption_l736_73654


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l736_73667

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculate the distance from a point to a line segment -/
def distanceToLineSegment (p : Point) (a : Point) (b : Point) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem parallelogram_vertex_sum (ABCD : Parallelogram) : 
  ABCD.A = Point.mk (-1) 2 →
  ABCD.B = Point.mk 3 (-6) →
  ABCD.D = Point.mk 7 0 →
  distanceToLineSegment ABCD.C ABCD.A ABCD.D = 3 * distanceToLineSegment ABCD.B ABCD.A ABCD.D →
  ABCD.C.x + ABCD.C.y = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l736_73667


namespace NUMINAMATH_CALUDE_age_difference_l736_73696

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 12) : A - C = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l736_73696


namespace NUMINAMATH_CALUDE_work_completion_time_l736_73639

/-- Given two workers x and y who can complete a work in 10 and 15 days respectively,
    prove that they can complete the work together in 6 days. -/
theorem work_completion_time (x y : ℝ) (hx : x = 1 / 10) (hy : y = 1 / 15) :
  1 / (x + y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l736_73639


namespace NUMINAMATH_CALUDE_west_movement_l736_73650

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (distance : ℤ) (direction : Direction) : ℤ :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- State the theorem
theorem west_movement :
  (∀ (d : ℤ), movement d Direction.East = d) →
  (∀ (d : ℤ), movement d Direction.West = -d) →
  movement 5 Direction.West = -5 := by
  sorry

end NUMINAMATH_CALUDE_west_movement_l736_73650


namespace NUMINAMATH_CALUDE_recreation_spending_ratio_l736_73693

/-- Proves that if wages decrease by 25% and recreation spending decreases from 30% to 20%,
    then the new recreation spending is 50% of the original. -/
theorem recreation_spending_ratio (original_wages : ℝ) (original_wages_positive : original_wages > 0) :
  let new_wages := 0.75 * original_wages
  let original_recreation := 0.3 * original_wages
  let new_recreation := 0.2 * new_wages
  new_recreation / original_recreation = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_recreation_spending_ratio_l736_73693


namespace NUMINAMATH_CALUDE_complex_equation_solution_l736_73604

theorem complex_equation_solution (z : ℂ) :
  (3 + 4*I) * z = 25 → z = 3 - 4*I := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l736_73604


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l736_73657

theorem sufficient_not_necessary (a : ℝ) (h : a > 0) :
  (∀ a, a > 2 → a^a > a^2) ∧
  (∃ a, 0 < a ∧ a < 2 ∧ a^a > a^2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l736_73657


namespace NUMINAMATH_CALUDE_nell_final_baseball_cards_l736_73646

/-- Represents the number of cards Nell has --/
structure Cards where
  initial_baseball : ℕ
  initial_ace : ℕ
  final_ace : ℕ
  difference : ℕ

/-- Calculates the final number of baseball cards Nell has --/
def final_baseball_cards (c : Cards) : ℕ :=
  c.final_ace - c.difference

/-- Theorem stating that Nell's final baseball card count is 111 --/
theorem nell_final_baseball_cards :
  let c : Cards := {
    initial_baseball := 239,
    initial_ace := 38,
    final_ace := 376,
    difference := 265
  }
  final_baseball_cards c = 111 := by
  sorry

end NUMINAMATH_CALUDE_nell_final_baseball_cards_l736_73646


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l736_73626

theorem quadratic_inequality_solution (x : ℝ) :
  (-x^2 - 2*x + 3 ≤ 0) ↔ (x ≤ -3 ∨ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l736_73626


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l736_73607

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + m + 3 = 0 ∧ y^2 + m*y + m + 3 = 0) ↔ 
  (m < -2 ∨ m > 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l736_73607


namespace NUMINAMATH_CALUDE_same_solution_implies_a_plus_b_zero_power_l736_73649

theorem same_solution_implies_a_plus_b_zero_power (a b : ℝ) :
  (∃ (x y : ℝ), 4*x + 3*y = 11 ∧ a*x + b*y = -2 ∧ 3*x - 5*y = 1 ∧ b*x - a*y = 6) →
  (a + b)^2023 = 0 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_plus_b_zero_power_l736_73649


namespace NUMINAMATH_CALUDE_linear_function_characterization_l736_73603

theorem linear_function_characterization (f : ℕ → ℕ) :
  (∀ x y : ℕ, f (x + y) = f x + f y) →
  ∃ a : ℕ, ∀ x : ℕ, f x = a * x :=
by sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l736_73603


namespace NUMINAMATH_CALUDE_smallest_room_width_l736_73640

theorem smallest_room_width 
  (largest_width : ℝ) 
  (largest_length : ℝ) 
  (smallest_length : ℝ) 
  (area_difference : ℝ) :
  largest_width = 45 →
  largest_length = 30 →
  smallest_length = 8 →
  largest_width * largest_length - smallest_length * (largest_width * largest_length - area_difference) / smallest_length = 1230 →
  (largest_width * largest_length - area_difference) / smallest_length = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_room_width_l736_73640


namespace NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l736_73614

theorem smallest_n_for_quadruplets : ∃ (n : ℕ+), 
  (∃! (quad_count : ℕ), quad_count = 154000 ∧ 
    (∃ (S : Finset (ℕ+ × ℕ+ × ℕ+ × ℕ+)), 
      Finset.card S = quad_count ∧
      ∀ (a b c d : ℕ+), (a, b, c, d) ∈ S ↔ 
        (Nat.gcd a.val (Nat.gcd b.val (Nat.gcd c.val d.val)) = 154 ∧
         Nat.lcm a.val (Nat.lcm b.val (Nat.lcm c.val d.val)) = n.val))) ∧
  (∀ (m : ℕ+), m < n →
    ¬∃ (quad_count : ℕ), quad_count = 154000 ∧
      (∃ (S : Finset (ℕ+ × ℕ+ × ℕ+ × ℕ+)),
        Finset.card S = quad_count ∧
        ∀ (a b c d : ℕ+), (a, b, c, d) ∈ S ↔
          (Nat.gcd a.val (Nat.gcd b.val (Nat.gcd c.val d.val)) = 154 ∧
           Nat.lcm a.val (Nat.lcm b.val (Nat.lcm c.val d.val)) = m.val))) ∧
  n = 25520328 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_quadruplets_l736_73614


namespace NUMINAMATH_CALUDE_quadratic_inequality_l736_73672

theorem quadratic_inequality (y : ℝ) : 
  y^2 + 3*y - 54 > 0 ↔ y < -9 ∨ y > 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l736_73672


namespace NUMINAMATH_CALUDE_max_height_is_100_l736_73690

-- Define the height function
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 20

-- Theorem stating that the maximum height is 100
theorem max_height_is_100 : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 100 := by
  sorry

end NUMINAMATH_CALUDE_max_height_is_100_l736_73690


namespace NUMINAMATH_CALUDE_solution_sum_comparison_l736_73605

theorem solution_sum_comparison
  (a a' b b' c c' : ℝ)
  (ha : a ≠ 0)
  (ha' : a' ≠ 0) :
  (c' - b') / a' < (c - b) / a ↔
  (c - b) / a > (c' - b') / a' :=
by sorry

end NUMINAMATH_CALUDE_solution_sum_comparison_l736_73605


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_radii_relation_l736_73609

/-- Theorem about the relationship between radii of spheres in a tetrahedron -/
theorem tetrahedron_sphere_radii_relation 
  (r r_a r_b r_c r_d : ℝ) 
  (S_a S_b S_c S_d V : ℝ) 
  (h_r : r = 3 * V / (S_a + S_b + S_c + S_d))
  (h_r_a : 1 / r_a = (-S_a + S_b + S_c + S_d) / (3 * V))
  (h_r_b : 1 / r_b = (S_a - S_b + S_c + S_d) / (3 * V))
  (h_r_c : 1 / r_c = (S_a + S_b - S_c + S_d) / (3 * V))
  (h_r_d : 1 / r_d = (S_a + S_b + S_c - S_d) / (3 * V))
  (h_positive : r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ r_d > 0) :
  1 / r_a + 1 / r_b + 1 / r_c + 1 / r_d = 2 / r :=
by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_radii_relation_l736_73609


namespace NUMINAMATH_CALUDE_quadratic_factorization_l736_73602

theorem quadratic_factorization (b c : ℝ) :
  (∀ x, x^2 + b*x + c = 0 ↔ x = -2 ∨ x = 3) →
  ∀ x, x^2 + b*x + c = (x + 2) * (x - 3) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l736_73602


namespace NUMINAMATH_CALUDE_roy_julia_age_difference_l736_73682

theorem roy_julia_age_difference :
  ∀ (R J K : ℕ) (x : ℕ),
    R = J + x →  -- Roy is x years older than Julia
    R = K + x / 2 →  -- Roy is half of x years older than Kelly
    R + 4 = 2 * (J + 4) →  -- In 4 years, Roy will be twice as old as Julia
    (R + 4) * (K + 4) = 108 →  -- In 4 years, Roy's age multiplied by Kelly's age would be 108
    x = 6 :=  -- The difference between Roy's and Julia's ages is 6 years
by sorry

end NUMINAMATH_CALUDE_roy_julia_age_difference_l736_73682


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l736_73699

theorem absolute_value_equation_solutions : 
  {x : ℝ | x + 1 = |x + 3| - |x - 1|} = {3, -1, -5} := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l736_73699


namespace NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l736_73658

def P (n : ℕ) : ℚ := 3 / ((n + 3) * (n + 4))

theorem smallest_n_for_P_less_than_threshold : 
  (∃ n : ℕ, P n < 1 / 2010) ∧ 
  (∀ m : ℕ, m < 23 → P m ≥ 1 / 2010) ∧ 
  (P 23 < 1 / 2010) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l736_73658


namespace NUMINAMATH_CALUDE_log_identity_l736_73685

theorem log_identity (a b P : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≠ 1) (ha1 : a ≠ 1) :
  (Real.log P / Real.log a) / (Real.log P / Real.log (a * b)) = 1 + Real.log b / Real.log a :=
by sorry

end NUMINAMATH_CALUDE_log_identity_l736_73685


namespace NUMINAMATH_CALUDE_system_solution_l736_73628

theorem system_solution :
  ∃ (a b c d : ℝ),
    (a + c = 4 ∧
     a * c + b + d = 6 ∧
     a * d + b * c = 5 ∧
     b * d = 2) ∧
    ((a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 1) ∨
     (a = 1 ∧ b = 1 ∧ c = 3 ∧ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l736_73628


namespace NUMINAMATH_CALUDE_partnership_investment_time_l736_73627

/-- Represents the investment and profit scenario of two partners -/
structure PartnershipScenario where
  /-- Ratio of partner p's investment to partner q's investment -/
  investment_ratio_p_q : Rat
  /-- Ratio of partner p's profit to partner q's profit -/
  profit_ratio_p_q : Rat
  /-- Number of months partner p invested -/
  p_investment_time : ℕ
  /-- Number of months partner q invested -/
  q_investment_time : ℕ

/-- Theorem stating the relationship between investment ratios, profit ratios, and investment times -/
theorem partnership_investment_time 
  (scenario : PartnershipScenario) 
  (h1 : scenario.investment_ratio_p_q = 7 / 5)
  (h2 : scenario.profit_ratio_p_q = 7 / 10)
  (h3 : scenario.p_investment_time = 7) :
  scenario.q_investment_time = 14 := by
  sorry

#check partnership_investment_time

end NUMINAMATH_CALUDE_partnership_investment_time_l736_73627


namespace NUMINAMATH_CALUDE_homeless_donation_calculation_l736_73618

theorem homeless_donation_calculation (total amount_first amount_second : ℝ) 
  (h1 : total = 900)
  (h2 : amount_first = 325)
  (h3 : amount_second = 260) :
  total - amount_first - amount_second = 315 :=
by sorry

end NUMINAMATH_CALUDE_homeless_donation_calculation_l736_73618


namespace NUMINAMATH_CALUDE_b_subset_a_iff_a_eq_two_or_three_c_subset_a_iff_m_condition_l736_73601

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + (a-1) = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

-- Theorem 1: B is a subset of A if and only if a = 2 or a = 3
theorem b_subset_a_iff_a_eq_two_or_three :
  ∀ a : ℝ, (∀ x : ℝ, x ∈ B a → x ∈ A) ↔ (a = 2 ∨ a = 3) :=
sorry

-- Theorem 2: C is a subset of A if and only if m = 3 or -2√2 < m < 2√2
theorem c_subset_a_iff_m_condition (m : ℝ) :
  C m ⊆ A ↔ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_b_subset_a_iff_a_eq_two_or_three_c_subset_a_iff_m_condition_l736_73601


namespace NUMINAMATH_CALUDE_combined_age_in_five_years_l736_73689

def amyAge : ℕ := 15
def markAgeDiff : ℕ := 7
def emilyAgeFactor : ℕ := 2
def yearsPassed : ℕ := 5

theorem combined_age_in_five_years :
  (amyAge + yearsPassed) + 
  (amyAge + markAgeDiff + yearsPassed) + 
  (amyAge * emilyAgeFactor + yearsPassed) = 82 := by
  sorry

end NUMINAMATH_CALUDE_combined_age_in_five_years_l736_73689


namespace NUMINAMATH_CALUDE_expression_evaluation_l736_73687

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  ((x * y + 2) * (x * y - 2) - 2 * x^2 * y^2 + 4) / (x * y) = -6 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l736_73687


namespace NUMINAMATH_CALUDE_walter_coins_percentage_l736_73622

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "half-dollar" => 50
  | _ => 0

/-- Calculates the total value of coins in cents -/
def total_value (pennies nickels half_dollars : ℕ) : ℕ :=
  pennies * coin_value "penny" +
  nickels * coin_value "nickel" +
  half_dollars * coin_value "half-dollar"

/-- Converts cents to percentage of a dollar -/
def cents_to_percentage (cents : ℕ) : ℚ :=
  (cents : ℚ) / 100

theorem walter_coins_percentage :
  cents_to_percentage (total_value 3 2 1) = 63 / 100 := by
  sorry

end NUMINAMATH_CALUDE_walter_coins_percentage_l736_73622


namespace NUMINAMATH_CALUDE_capacity_variation_l736_73679

/-- Given positive constants e, R, and r, prove that the function C(n) = en / (R + nr^2) 
    first increases and then decreases as n increases. -/
theorem capacity_variation (e R r : ℝ) (he : e > 0) (hR : R > 0) (hr : r > 0) :
  ∃ n₀ : ℝ, n₀ > 0 ∧
    (∀ n₁ n₂ : ℝ, 0 < n₁ ∧ n₁ < n₂ ∧ n₂ < n₀ → 
      (e * n₁) / (R + n₁ * r^2) < (e * n₂) / (R + n₂ * r^2)) ∧
    (∀ n₁ n₂ : ℝ, n₀ < n₁ ∧ n₁ < n₂ → 
      (e * n₁) / (R + n₁ * r^2) > (e * n₂) / (R + n₂ * r^2)) :=
sorry

end NUMINAMATH_CALUDE_capacity_variation_l736_73679


namespace NUMINAMATH_CALUDE_intersection_M_N_l736_73692

/-- Set M is defined as {x | 0 ≤ x ≤ 1} -/
def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

/-- Set N is defined as {x | |x| ≥ 1} -/
def N : Set ℝ := {x | abs x ≥ 1}

/-- The intersection of sets M and N is equal to the set containing only 1 -/
theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l736_73692


namespace NUMINAMATH_CALUDE_f_range_l736_73620

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x + 4)

-- State the theorem
theorem f_range :
  Set.range f = {y : ℝ | y < 18 ∨ y > 18} :=
by
  sorry


end NUMINAMATH_CALUDE_f_range_l736_73620


namespace NUMINAMATH_CALUDE_smaller_to_larger_base_ratio_l736_73615

/-- An isosceles trapezoid with an inscribed equilateral triangle -/
structure IsoscelesTrapezoidWithTriangle where
  /-- Length of the smaller base (and side of the equilateral triangle) -/
  s : ℝ
  /-- Length of the larger base -/
  b : ℝ
  /-- s is positive -/
  s_pos : 0 < s
  /-- b is positive -/
  b_pos : 0 < b
  /-- The larger base is twice the length of a diagonal of the equilateral triangle -/
  diag_relation : b = 2 * s

/-- The ratio of the smaller base to the larger base is 1/2 -/
theorem smaller_to_larger_base_ratio 
  (t : IsoscelesTrapezoidWithTriangle) : t.s / t.b = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_smaller_to_larger_base_ratio_l736_73615


namespace NUMINAMATH_CALUDE_power_product_simplification_l736_73664

theorem power_product_simplification :
  3000 * (3000 ^ 2999) = 3000 ^ 3000 := by sorry

end NUMINAMATH_CALUDE_power_product_simplification_l736_73664


namespace NUMINAMATH_CALUDE_triangle_pieces_count_l736_73611

/-- The number of rods in the nth row of the triangle -/
def rods_in_row (n : ℕ) : ℕ := 4 * n

/-- The total number of rods in a triangle with n rows -/
def total_rods (n : ℕ) : ℕ := (n * (n + 1) / 2) * 4

/-- The number of connectors in a triangle with n rows of rods -/
def total_connectors (n : ℕ) : ℕ := ((n + 1) * (n + 2)) / 2

/-- The total number of pieces (rods and connectors) in a triangle with n rows -/
def total_pieces (n : ℕ) : ℕ := total_rods n + total_connectors n

theorem triangle_pieces_count :
  total_pieces 10 = 286 := by sorry

end NUMINAMATH_CALUDE_triangle_pieces_count_l736_73611


namespace NUMINAMATH_CALUDE_equal_fractions_sum_l736_73662

theorem equal_fractions_sum (n : ℕ) (sum : ℚ) (fraction : ℚ) :
  n = 450 →
  sum = 1 / 12 →
  n * fraction = sum →
  fraction = 1 / 5400 := by
  sorry

end NUMINAMATH_CALUDE_equal_fractions_sum_l736_73662


namespace NUMINAMATH_CALUDE_point_on_line_l736_73655

-- Define the points
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (4, 3)
def C : ℝ → ℝ × ℝ := λ m ↦ (5, m)

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop :=
  (q.2 - p.2) * (r.1 - q.1) = (r.2 - q.2) * (q.1 - p.1)

-- Theorem statement
theorem point_on_line (m : ℝ) :
  collinear A B (C m) → m = 6 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l736_73655


namespace NUMINAMATH_CALUDE_green_marble_probability_l736_73676

/-- The probability of drawing a green marble from a box with 100 marbles -/
theorem green_marble_probability :
  ∀ (p_white p_red_or_blue p_green : ℝ),
  p_white = 1/4 →
  p_red_or_blue = 0.55 →
  p_white + p_red_or_blue + p_green = 1 →
  p_green = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_green_marble_probability_l736_73676


namespace NUMINAMATH_CALUDE_order_of_numbers_l736_73686

theorem order_of_numbers : 
  20.3 > 1 → 
  0 < 0.32 ∧ 0.32 < 1 → 
  Real.log 0.32 < 0 → 
  Real.log 0.32 < 0.32 ∧ 0.32 < 20.3 := by
sorry

end NUMINAMATH_CALUDE_order_of_numbers_l736_73686


namespace NUMINAMATH_CALUDE_fruit_preference_ratio_l736_73666

theorem fruit_preference_ratio (total_students : ℕ) 
  (cherries_preference : ℕ) (apple_date_ratio : ℕ) (banana_cherry_ratio : ℕ) 
  (h1 : total_students = 780)
  (h2 : cherries_preference = 60)
  (h3 : apple_date_ratio = 2)
  (h4 : banana_cherry_ratio = 3) : 
  (banana_cherry_ratio * cherries_preference) / 
  ((total_students - banana_cherry_ratio * cherries_preference - cherries_preference) / 
   (apple_date_ratio + 1)) = 1 := by
sorry

end NUMINAMATH_CALUDE_fruit_preference_ratio_l736_73666


namespace NUMINAMATH_CALUDE_dress_discount_problem_l736_73663

theorem dress_discount_problem (P : ℝ) (D : ℝ) : 
  P - 61.2 = 4.5 → P * (1 - D) * 1.25 = 61.2 → D = 0.255 := by
sorry

end NUMINAMATH_CALUDE_dress_discount_problem_l736_73663


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l736_73653

-- Define the properties for m and n
def has_two_divisors (x : ℕ) : Prop := (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 2

def has_four_divisors (x : ℕ) : Prop := (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 4

def is_smallest_with_two_divisors (m : ℕ) : Prop :=
  has_two_divisors m ∧ ∀ k < m, ¬has_two_divisors k

def is_largest_under_200_with_four_divisors (n : ℕ) : Prop :=
  n < 200 ∧ has_four_divisors n ∧ ∀ k > n, k < 200 → ¬has_four_divisors k

-- State the theorem
theorem sum_of_special_numbers :
  ∃ (m n : ℕ), is_smallest_with_two_divisors m ∧ is_largest_under_200_with_four_divisors n ∧ m + n = 127 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l736_73653


namespace NUMINAMATH_CALUDE_function_range_l736_73623

/-- The function f(x) = (x^2 - 2x - 3)(x^2 - 2x - 5) has a range of [-1, +∞) -/
theorem function_range (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (x^2 - 2*x - 3) * (x^2 - 2*x - 5)
  ∃ (y : ℝ), y ≥ -1 ∧ ∃ (x : ℝ), f x = y :=
by sorry

end NUMINAMATH_CALUDE_function_range_l736_73623


namespace NUMINAMATH_CALUDE_tangent_lines_through_A_area_of_triangle_AOC_l736_73694

-- Define the circle C
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y + 12 = 0

-- Define point A
def point_A : ℝ × ℝ := (3, 5)

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Define the center of the circle C
def center : ℝ × ℝ := (2, 3)

-- Theorem for the tangent lines
theorem tangent_lines_through_A :
  ∃ (k : ℝ), 
    (∀ x y : ℝ, x = 3 → circle_equation x y) ∧
    (∀ x y : ℝ, y = k*x + (11/4) → circle_equation x y) ∧
    k = 3/4 :=
sorry

-- Theorem for the area of triangle AOC
theorem area_of_triangle_AOC :
  let A := point_A
  let O := origin
  let C := center
  (1/2 : ℝ) * ‖C - O‖ * ‖A - O‖ * (|C.1 * A.2 - C.2 * A.1| / (‖C - O‖ * ‖A - O‖)) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_through_A_area_of_triangle_AOC_l736_73694


namespace NUMINAMATH_CALUDE_sequence_term_proof_l736_73617

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℚ := (2/3) * n^2 - (1/3) * n

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℚ := (4/3) * n - 1

theorem sequence_term_proof (n : ℕ) (h : n > 0) : 
  a n = S n - S (n-1) :=
sorry

end NUMINAMATH_CALUDE_sequence_term_proof_l736_73617


namespace NUMINAMATH_CALUDE_loan_balance_years_l736_73619

theorem loan_balance_years (c V t n : ℝ) (hc : c > 0) (hV : V > 0) (ht : t > -1) :
  V = c / (1 + t)^(3 * n) → n = (Real.log (c / V)) / (3 * Real.log (1 + t)) := by
  sorry

end NUMINAMATH_CALUDE_loan_balance_years_l736_73619


namespace NUMINAMATH_CALUDE_solution_to_equation_l736_73600

theorem solution_to_equation (x y : ℝ) : 
  x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ↔ x = 1/3 ∧ y = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l736_73600


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l736_73638

theorem interest_rate_calculation (principal : ℝ) (interest_paid : ℝ) 
  (h1 : principal = 900) 
  (h2 : interest_paid = 729) : ∃ (rate : ℝ), 
  interest_paid = principal * rate * rate / 100 ∧ rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l736_73638


namespace NUMINAMATH_CALUDE_six_tricycles_l736_73642

/-- Represents the number of children riding each type of vehicle -/
structure VehicleCounts where
  bicycles : ℕ
  tricycles : ℕ
  unicycles : ℕ

/-- The total number of children -/
def total_children : ℕ := 10

/-- The total number of wheels -/
def total_wheels : ℕ := 26

/-- Calculates the total number of children based on vehicle counts -/
def count_children (v : VehicleCounts) : ℕ :=
  v.bicycles + v.tricycles + v.unicycles

/-- Calculates the total number of wheels based on vehicle counts -/
def count_wheels (v : VehicleCounts) : ℕ :=
  2 * v.bicycles + 3 * v.tricycles + v.unicycles

/-- Theorem stating that there are 6 tricycles -/
theorem six_tricycles : 
  ∃ v : VehicleCounts, 
    count_children v = total_children ∧ 
    count_wheels v = total_wheels ∧ 
    v.tricycles = 6 := by
  sorry

end NUMINAMATH_CALUDE_six_tricycles_l736_73642


namespace NUMINAMATH_CALUDE_vector_coefficient_theorem_l736_73678

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points
variable (A B C P O : V)

-- Define the condition that P lies in the plane of triangle ABC
def in_plane (A B C P : V) : Prop := 
  ∃ (α β γ : ℝ), α + β + γ = 1 ∧ P = α • A + β • B + γ • C

-- Define the vector equation
def vector_equation (A B C P O : V) (x : ℝ) : Prop :=
  P - O = (1/2) • (A - O) + (1/3) • (B - O) + x • (C - O)

-- State the theorem
theorem vector_coefficient_theorem 
  (h_plane : in_plane V A B C P)
  (h_eq : vector_equation V A B C P O x) :
  x = 1/6 := by sorry

end NUMINAMATH_CALUDE_vector_coefficient_theorem_l736_73678


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l736_73645

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem line_perpendicular_to_plane 
  (l m : Line) (α : Plane) : 
  perpendicular l α → parallel l m → perpendicular m α := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l736_73645


namespace NUMINAMATH_CALUDE_max_ab_value_l736_73675

theorem max_ab_value (a b : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - b)^2 = 1 ∧ x + 2*y - 1 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ - a)^2 + (y₁ - b)^2 = 1 ∧ 
    (x₂ - a)^2 + (y₂ - b)^2 = 1 ∧ 
    x₁ + 2*y₁ - 1 = 0 ∧ 
    x₂ + 2*y₂ - 1 = 0 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4/5 * Real.sqrt 5)^2) →
  a * b ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l736_73675


namespace NUMINAMATH_CALUDE_confectioner_pastries_l736_73648

theorem confectioner_pastries :
  ∀ (total_pastries : ℕ) 
    (regular_customers : ℕ) 
    (actual_customers : ℕ) 
    (pastry_difference : ℕ),
  regular_customers = 28 →
  actual_customers = 49 →
  pastry_difference = 6 →
  regular_customers * (total_pastries / regular_customers) = 
    actual_customers * (total_pastries / regular_customers - pastry_difference) →
  total_pastries = 1176 :=
by
  sorry

end NUMINAMATH_CALUDE_confectioner_pastries_l736_73648


namespace NUMINAMATH_CALUDE_divisors_product_18_l736_73697

def divisors_product (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem divisors_product_18 : divisors_product 18 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_divisors_product_18_l736_73697


namespace NUMINAMATH_CALUDE_fourth_day_pages_l736_73635

/-- Represents the number of pages read each day -/
structure DailyPages where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Represents the conditions of the book reading problem -/
structure BookReading where
  totalPages : ℕ
  dailyPages : DailyPages
  day1Condition : dailyPages.day1 = 63
  day2Condition : dailyPages.day2 = 2 * dailyPages.day1
  day3Condition : dailyPages.day3 = dailyPages.day2 + 10
  totalCondition : totalPages = dailyPages.day1 + dailyPages.day2 + dailyPages.day3 + dailyPages.day4

/-- Theorem stating that given the conditions, the number of pages read on the fourth day is 29 -/
theorem fourth_day_pages (br : BookReading) (h : br.totalPages = 354) : br.dailyPages.day4 = 29 := by
  sorry

end NUMINAMATH_CALUDE_fourth_day_pages_l736_73635


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l736_73633

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h : is_geometric_sequence a) :
  (is_geometric_sequence (fun n ↦ (a n)^2)) ∧
  (∀ k : ℝ, k ≠ 0 → is_geometric_sequence (fun n ↦ k * a n)) ∧
  (is_geometric_sequence (fun n ↦ 1 / (a n))) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l736_73633


namespace NUMINAMATH_CALUDE_solution_set_l736_73661

def system_solution (x y : ℝ) : Prop :=
  x + y = 20 ∧ Real.log x / Real.log 4 + Real.log y / Real.log 4 = 1 + Real.log 9 / Real.log 4

theorem solution_set : 
  {(x, y) : ℝ × ℝ | system_solution x y} = {(18, 2), (2, 18)} := by sorry

end NUMINAMATH_CALUDE_solution_set_l736_73661


namespace NUMINAMATH_CALUDE_sum_and_subtract_l736_73629

theorem sum_and_subtract : (2345 + 3452 + 4523 + 5234) - 1234 = 14320 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_subtract_l736_73629


namespace NUMINAMATH_CALUDE_treehouse_rope_length_l736_73656

theorem treehouse_rope_length : 
  let rope_lengths : List Nat := [24, 20, 14, 12, 18, 22]
  List.sum rope_lengths = 110 := by
  sorry

end NUMINAMATH_CALUDE_treehouse_rope_length_l736_73656


namespace NUMINAMATH_CALUDE_parabola_equation_l736_73668

/-- Given a point M(5,3) and a parabola y=ax^2 where the distance from M to the axis of symmetry is 6,
    prove that the equation of the parabola is either y = 1/12 x^2 or y = -1/36 x^2 -/
theorem parabola_equation (a : ℝ) (h : |5 + 1/(4*a)| = 6) :
  a = 1/12 ∨ a = -1/36 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l736_73668


namespace NUMINAMATH_CALUDE_set_union_problem_l736_73637

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem set_union_problem (x : ℝ) :
  (∃ y, A y ∩ B y = {9}) →
  (∃ z, A z ∪ B z = {-4, -7, -8, 4, 9}) :=
by sorry

end NUMINAMATH_CALUDE_set_union_problem_l736_73637


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_A_l736_73669

theorem partial_fraction_decomposition_A (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -2 ∧ x ≠ 1 →
    1 / (x^3 - 2*x^2 - 13*x + 10) = A / (x + 2) + B / (x - 1) + C / ((x - 1)^2)) →
  A = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_A_l736_73669


namespace NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l736_73651

/-- Given an arithmetic progression with first term 7, prove that the smallest
    possible value for the third term of the resulting geometric progression is 3.752 -/
theorem smallest_third_term_of_geometric_progression
  (a b c : ℝ)
  (h_arithmetic : ∃ (d : ℝ), a = 7 ∧ b = 7 + d ∧ c = 7 + 2*d)
  (h_geometric : ∃ (r : ℝ), (7 : ℝ) * r = b - 3 ∧ (b - 3) * r = c + 15) :
  ∃ (x : ℝ), (∀ (y : ℝ), (7 : ℝ) * (b - 3) = (b - 3) * (c + 15) → c + 15 ≥ x) ∧ c + 15 ≥ 3.752 :=
sorry

end NUMINAMATH_CALUDE_smallest_third_term_of_geometric_progression_l736_73651


namespace NUMINAMATH_CALUDE_factorization_equality_l736_73621

theorem factorization_equality (a b : ℝ) : a^2 * b - a^3 = a^2 * (b - a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l736_73621


namespace NUMINAMATH_CALUDE_total_interest_after_ten_years_l736_73671

/-- Calculate the total interest after 10 years given the conditions -/
theorem total_interest_after_ten_years
  (P : ℝ) -- Principal amount
  (R : ℝ) -- Interest rate (in percentage per annum)
  (h1 : P * R * 10 / 100 = 600) -- Simple interest on P for 10 years is Rs. 600
  : P * R * 5 / 100 + 3 * P * R * 5 / 100 = 1200 := by
  sorry

#check total_interest_after_ten_years

end NUMINAMATH_CALUDE_total_interest_after_ten_years_l736_73671


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l736_73643

theorem arithmetic_expression_equality : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l736_73643


namespace NUMINAMATH_CALUDE_joe_oranges_count_l736_73647

/-- The number of boxes Joe has for oranges -/
def num_boxes : ℕ := 9

/-- The number of oranges required in each box -/
def oranges_per_box : ℕ := 5

/-- The total number of oranges Joe has -/
def total_oranges : ℕ := num_boxes * oranges_per_box

theorem joe_oranges_count : total_oranges = 45 := by
  sorry

end NUMINAMATH_CALUDE_joe_oranges_count_l736_73647


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n_l736_73636

/-- A positive integer greater than 1 is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ k, 1 < k ∧ k < n ∧ n % k = 0

/-- For all composite integers n, 6 divides n^4 - n and is the largest such divisor. -/
theorem largest_divisor_of_n4_minus_n (n : ℕ) (h : IsComposite n) :
    (6 ∣ n^4 - n) ∧ ∀ m : ℕ, m > 6 → ¬(∀ k : ℕ, IsComposite k → (m ∣ k^4 - k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n_l736_73636
