import Mathlib

namespace NUMINAMATH_CALUDE_range_of_fraction_l1154_115413

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : 2 < b ∧ b < 8) :
  1/8 < a/b ∧ a/b < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_fraction_l1154_115413


namespace NUMINAMATH_CALUDE_tanya_work_days_l1154_115462

/-- Given Sakshi can do a piece of work in 12 days and Tanya is 20% more efficient than Sakshi,
    prove that Tanya can complete the same piece of work in 10 days. -/
theorem tanya_work_days (sakshi_days : ℝ) (tanya_efficiency : ℝ) :
  sakshi_days = 12 →
  tanya_efficiency = 1.2 →
  (sakshi_days / tanya_efficiency) = 10 := by
sorry

end NUMINAMATH_CALUDE_tanya_work_days_l1154_115462


namespace NUMINAMATH_CALUDE_statement_1_statement_2_statement_3_l1154_115486

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (5*a + 1) * x + 4*a + 4

-- Statement 1
theorem statement_1 (a : ℝ) (h : a < -1) : f a 0 < 0 := by sorry

-- Statement 2
theorem statement_2 (a : ℝ) (h : a > 0) : 
  ∃ (y : ℝ), y = 3 ∧ ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → f a x ≤ y := by sorry

-- Statement 3
theorem statement_3 (a : ℝ) (h : a < 0) : 
  f a 2 > f a 3 ∧ f a 3 > f a 4 := by sorry

end NUMINAMATH_CALUDE_statement_1_statement_2_statement_3_l1154_115486


namespace NUMINAMATH_CALUDE_car_ownership_theorem_l1154_115484

/-- The number of cars owned by Cathy, Lindsey, Carol, and Susan -/
def total_cars (cathy lindsey carol susan : ℕ) : ℕ :=
  cathy + lindsey + carol + susan

/-- Theorem stating the total number of cars owned by the four people -/
theorem car_ownership_theorem (cathy lindsey carol susan : ℕ) 
  (h1 : cathy = 5)
  (h2 : lindsey = cathy + 4)
  (h3 : carol = 2 * cathy)
  (h4 : susan = carol - 2) :
  total_cars cathy lindsey carol susan = 32 := by
  sorry

#check car_ownership_theorem

end NUMINAMATH_CALUDE_car_ownership_theorem_l1154_115484


namespace NUMINAMATH_CALUDE_dihedral_angle_cosine_l1154_115420

/-- A regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  -- Add necessary fields here

/-- A line passing through the centers of two circles -/
structure IntersectingLine where
  -- Add necessary fields here

/-- The dihedral angle at the base of the pyramid -/
def dihedralAngle (p : RegularHexagonalPyramid) : ℝ := sorry

/-- Theorem: The cosine of the dihedral angle at the base of a regular hexagonal pyramid
    with the specified intersecting line is equal to sqrt(3/13) -/
theorem dihedral_angle_cosine (p : RegularHexagonalPyramid) (l : IntersectingLine) :
  Real.cos (dihedralAngle p) = Real.sqrt (3 / 13) := by sorry

end NUMINAMATH_CALUDE_dihedral_angle_cosine_l1154_115420


namespace NUMINAMATH_CALUDE_pin_combinations_l1154_115401

/-- The number of unique permutations of a multiset with elements {5, 3, 3, 7} -/
def pinPermutations : ℕ :=
  Nat.factorial 4 / (Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 1)

theorem pin_combinations : pinPermutations = 12 := by
  sorry

end NUMINAMATH_CALUDE_pin_combinations_l1154_115401


namespace NUMINAMATH_CALUDE_board_number_game_l1154_115464

theorem board_number_game (n : ℕ) (h : n = 2009) : 
  let initial_sum := n * (n + 1) / 2
  let initial_remainder := initial_sum % 13
  ∃ (a : ℕ), a ≤ n ∧ (a + 9 + 999) % 13 = initial_remainder ∧ a = 8 :=
sorry

end NUMINAMATH_CALUDE_board_number_game_l1154_115464


namespace NUMINAMATH_CALUDE_only_25_is_five_times_greater_than_last_digit_l1154_115463

def lastDigit (n : Nat) : Nat :=
  n % 10

theorem only_25_is_five_times_greater_than_last_digit :
  ∀ n : Nat, n > 0 → (n = 5 * lastDigit n + lastDigit n) → n = 25 := by
  sorry

end NUMINAMATH_CALUDE_only_25_is_five_times_greater_than_last_digit_l1154_115463


namespace NUMINAMATH_CALUDE_two_trains_meeting_time_l1154_115468

/-- Two trains problem -/
theorem two_trains_meeting_time 
  (distance : ℝ) 
  (fast_speed slow_speed : ℝ) 
  (head_start : ℝ) 
  (h_distance : distance = 270) 
  (h_fast_speed : fast_speed = 120) 
  (h_slow_speed : slow_speed = 75) 
  (h_head_start : head_start = 1) :
  ∃ x : ℝ, slow_speed * head_start + (fast_speed + slow_speed) * x = distance :=
by sorry

end NUMINAMATH_CALUDE_two_trains_meeting_time_l1154_115468


namespace NUMINAMATH_CALUDE_expression_simplification_l1154_115409

theorem expression_simplification (a b c x y : ℝ) (h : c^2*b*x + c*a*y ≠ 0) :
  (c^2*b*x*(a^3*x^3 + 3*a^2*y^2 + b^3*y^3) + c*a*y*(a^3*x^3 + 3*b^3*x^3 + b^3*y^3)) / (c^2*b*x + c*a*y) = 
  a^3*x^3 + 3*a*b^3*x^3 + b^3*y^3 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1154_115409


namespace NUMINAMATH_CALUDE_triangle_inradius_l1154_115471

/-- Given a triangle with perimeter 40 and area 50, prove that its inradius is 2.5 -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : P = 40) 
  (h2 : A = 50) 
  (h3 : A = r * (P / 2)) : r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l1154_115471


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1154_115417

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a^2 < b → -Real.sqrt b < a ∧ a < Real.sqrt b)) ↔
  ((a ≥ Real.sqrt b ∨ a ≤ -Real.sqrt b) → a^2 ≥ b) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1154_115417


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1154_115405

theorem fraction_subtraction : (3 + 6 + 9) / (2 + 4 + 8) - (2 + 4 + 8) / (3 + 6 + 9) = 32 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1154_115405


namespace NUMINAMATH_CALUDE_modular_sum_equivalence_l1154_115496

theorem modular_sum_equivalence : ∃ (x y z : ℤ), 
  (5 * x) % 29 = 1 ∧ 
  (5 * y) % 29 = 1 ∧ 
  (7 * z) % 29 = 1 ∧ 
  (x + y + z) % 29 = 13 := by
  sorry

end NUMINAMATH_CALUDE_modular_sum_equivalence_l1154_115496


namespace NUMINAMATH_CALUDE_digit_count_700_l1154_115403

def count_digit (d : Nat) (n : Nat) : Nat :=
  (n / 100 + 1) * 10

theorem digit_count_700 : 
  (count_digit 9 700 + count_digit 8 700) = 280 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_700_l1154_115403


namespace NUMINAMATH_CALUDE_polynomial_properties_l1154_115498

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial (a b c : ℝ) (x : ℂ) : ℂ :=
  a * x^2 + b * x + c

/-- The polynomial we want to prove about -/
def our_polynomial (x : ℂ) : ℂ :=
  QuadraticPolynomial 2 (-12) 20 x

theorem polynomial_properties :
  (our_polynomial (3 + Complex.I) = 0) ∧
  (∀ x : ℂ, our_polynomial x = 2 * x^2 + (-12) * x + 20) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1154_115498


namespace NUMINAMATH_CALUDE_profit_percentage_l1154_115407

theorem profit_percentage (C S : ℝ) (h : 55 * C = 50 * S) : 
  (S - C) / C * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l1154_115407


namespace NUMINAMATH_CALUDE_average_of_x_and_y_l1154_115477

theorem average_of_x_and_y (x y : ℝ) : 
  (4 + 6.5 + 8 + x + y) / 5 = 18 → (x + y) / 2 = 35.75 := by
  sorry

end NUMINAMATH_CALUDE_average_of_x_and_y_l1154_115477


namespace NUMINAMATH_CALUDE_expression_equals_expected_result_l1154_115453

-- Define the expression
def expression : ℤ := 8 - (-3) + (-5) + (-7)

-- Define the expected result
def expected_result : ℤ := 3 + 8 - 7 - 5

-- Theorem statement
theorem expression_equals_expected_result :
  expression = expected_result :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_expected_result_l1154_115453


namespace NUMINAMATH_CALUDE_polygon_triangle_division_l1154_115410

/-- 
A polygon where the sum of interior angles is twice the sum of exterior angles
can be divided into at most 4 triangles by connecting one vertex to all others.
-/
theorem polygon_triangle_division :
  ∀ (n : ℕ), 
  (n - 2) * 180 = 2 * 360 →
  n - 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_polygon_triangle_division_l1154_115410


namespace NUMINAMATH_CALUDE_small_box_width_l1154_115426

/-- Proves that the width of smaller boxes is 50 cm given the conditions of the problem -/
theorem small_box_width (large_length large_width large_height : ℝ)
                        (small_length small_height : ℝ)
                        (max_small_boxes : ℕ) :
  large_length = 6 →
  large_width = 5 →
  large_height = 4 →
  small_length = 0.6 →
  small_height = 0.4 →
  max_small_boxes = 1000 →
  ∃ (small_width : ℝ),
    small_width = 0.5 ∧
    (max_small_boxes : ℝ) * small_length * small_width * small_height =
    large_length * large_width * large_height :=
by sorry

#check small_box_width

end NUMINAMATH_CALUDE_small_box_width_l1154_115426


namespace NUMINAMATH_CALUDE_black_tiles_imply_total_tiles_l1154_115473

/-- Represents a square floor tiled with congruent square tiles -/
structure TiledFloor where
  side_length : ℕ

/-- Counts the number of black tiles on the diagonals of a square floor -/
def diagonal_black_tiles (floor : TiledFloor) : ℕ :=
  2 * floor.side_length - 1

/-- Counts the number of black tiles in a quarter of the floor -/
def quarter_black_tiles (floor : TiledFloor) : ℕ :=
  (floor.side_length ^ 2) / 4

/-- Calculates the total number of tiles on the floor -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.side_length ^ 2

/-- Theorem stating that if there are 225 black tiles in total, then the total number of tiles is 1024 -/
theorem black_tiles_imply_total_tiles (floor : TiledFloor) :
  diagonal_black_tiles floor + quarter_black_tiles floor = 225 →
  total_tiles floor = 1024 := by
  sorry

end NUMINAMATH_CALUDE_black_tiles_imply_total_tiles_l1154_115473


namespace NUMINAMATH_CALUDE_profit_per_meter_l1154_115455

/-- The profit per meter of cloth given the selling price, quantity sold, and cost price per meter -/
theorem profit_per_meter
  (selling_price : ℕ)
  (quantity : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : selling_price = 4950)
  (h2 : quantity = 75)
  (h3 : cost_price_per_meter = 51) :
  (selling_price - quantity * cost_price_per_meter) / quantity = 15 :=
by sorry

end NUMINAMATH_CALUDE_profit_per_meter_l1154_115455


namespace NUMINAMATH_CALUDE_least_frood_drop_beats_eat_l1154_115488

def frood_drop_score (n : ℕ) : ℕ := n * (n + 1) / 2
def frood_eat_score (n : ℕ) : ℕ := 15 * n

theorem least_frood_drop_beats_eat :
  ∀ k : ℕ, k < 30 → frood_drop_score k ≤ frood_eat_score k ∧
  frood_drop_score 30 > frood_eat_score 30 :=
sorry

end NUMINAMATH_CALUDE_least_frood_drop_beats_eat_l1154_115488


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l1154_115425

theorem max_digits_product_5_4 : 
  ∀ (a b : ℕ), 
  10000 ≤ a ∧ a ≤ 99999 → 
  1000 ≤ b ∧ b ≤ 9999 → 
  a * b < 1000000000 := by
sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l1154_115425


namespace NUMINAMATH_CALUDE_h_zero_iff_b_eq_seven_fifths_l1154_115459

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem stating that h(b) = 0 if and only if b = 7/5 -/
theorem h_zero_iff_b_eq_seven_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = 7 / 5 := by sorry

end NUMINAMATH_CALUDE_h_zero_iff_b_eq_seven_fifths_l1154_115459


namespace NUMINAMATH_CALUDE_basketball_passes_l1154_115427

/-- Represents the number of ways the ball can be with player A after n moves -/
def ball_with_A (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else 2 * ball_with_A (n - 1) + 3 * ball_with_A (n - 2)

/-- The problem statement -/
theorem basketball_passes :
  ball_with_A 7 = 1094 := by
  sorry


end NUMINAMATH_CALUDE_basketball_passes_l1154_115427


namespace NUMINAMATH_CALUDE_sector_area_l1154_115492

theorem sector_area (arc_length : Real) (central_angle : Real) :
  arc_length = π ∧ central_angle = π / 4 →
  let radius := arc_length / central_angle
  let sector_area := (1 / 2) * radius^2 * central_angle
  sector_area = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1154_115492


namespace NUMINAMATH_CALUDE_river_depth_difference_l1154_115475

/-- River depth problem -/
theorem river_depth_difference (depth_may depth_june depth_july : ℝ) : 
  depth_may = 5 →
  depth_july = 45 →
  depth_july = 3 * depth_june →
  depth_june - depth_may = 10 := by
  sorry

end NUMINAMATH_CALUDE_river_depth_difference_l1154_115475


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1154_115440

theorem right_triangle_third_side 
  (a b : ℝ) 
  (h1 : Real.sqrt (a - 3) + |b - 4| = 0) : 
  ∃ c : ℝ, (c = 5 ∨ c = Real.sqrt 7) ∧ 
    ((a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1154_115440


namespace NUMINAMATH_CALUDE_sample_size_example_l1154_115465

/-- Definition of a sample size in a statistical context -/
def sample_size (population : ℕ) (selected : ℕ) : ℕ := selected

/-- Theorem: The sample size for 100 items selected from a population of 5000 is 100 -/
theorem sample_size_example : sample_size 5000 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_example_l1154_115465


namespace NUMINAMATH_CALUDE_counterexample_non_coprime_l1154_115470

theorem counterexample_non_coprime :
  ∃ (a n : ℕ+), (Nat.gcd a.val n.val ≠ 1) ∧ (a.val ^ n.val % n.val ≠ a.val % n.val) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_non_coprime_l1154_115470


namespace NUMINAMATH_CALUDE_expression_factorization_l1154_115472

theorem expression_factorization (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 12 * x^4 + 3) = 12 * (x^6 + 4 * x^4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1154_115472


namespace NUMINAMATH_CALUDE_coin_difference_l1154_115485

/-- Represents the number of coins of each denomination in Tom's collection -/
structure CoinCollection where
  fiveCent : ℚ
  tenCent : ℚ
  twentyCent : ℚ

/-- Conditions for Tom's coin collection -/
def validCollection (c : CoinCollection) : Prop :=
  c.fiveCent + c.tenCent + c.twentyCent = 30 ∧
  c.tenCent = 2 * c.fiveCent ∧
  5 * c.fiveCent + 10 * c.tenCent + 20 * c.twentyCent = 340

/-- The main theorem to prove -/
theorem coin_difference (c : CoinCollection) 
  (h : validCollection c) : c.twentyCent - c.fiveCent = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_coin_difference_l1154_115485


namespace NUMINAMATH_CALUDE_product_units_digit_base_6_l1154_115487

-- Define the base-10 numbers
def a : ℕ := 217
def b : ℕ := 45

-- Define the base of the target representation
def base : ℕ := 6

-- Theorem statement
theorem product_units_digit_base_6 :
  (a * b) % base = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_units_digit_base_6_l1154_115487


namespace NUMINAMATH_CALUDE_shanes_bread_packages_l1154_115454

theorem shanes_bread_packages :
  ∀ (slices_per_bread_package : ℕ) 
    (ham_packages : ℕ) 
    (slices_per_ham_package : ℕ) 
    (bread_slices_per_sandwich : ℕ) 
    (leftover_bread_slices : ℕ),
  slices_per_bread_package = 20 →
  ham_packages = 2 →
  slices_per_ham_package = 8 →
  bread_slices_per_sandwich = 2 →
  leftover_bread_slices = 8 →
  (ham_packages * slices_per_ham_package * bread_slices_per_sandwich + leftover_bread_slices) / slices_per_bread_package = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_shanes_bread_packages_l1154_115454


namespace NUMINAMATH_CALUDE_knights_problem_l1154_115481

/-- Represents the arrangement of knights -/
structure KnightArrangement where
  total : ℕ
  rows : ℕ
  cols : ℕ
  knights_per_row : ℕ
  knights_per_col : ℕ

/-- The conditions of the problem -/
def problem_conditions (k : KnightArrangement) : Prop :=
  k.total = k.rows * k.cols ∧
  k.total - 2 * k.knights_per_row = 24 ∧
  k.total - 2 * k.knights_per_col = 18

/-- The theorem to be proved -/
theorem knights_problem :
  ∀ k : KnightArrangement, problem_conditions k → k.total = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_knights_problem_l1154_115481


namespace NUMINAMATH_CALUDE_quadratic_integer_roots_l1154_115433

theorem quadratic_integer_roots (p : ℕ) (b : ℕ) (hp : Prime p) (hb : b > 0) :
  (∃ x y : ℤ, x^2 - b*x + b*p = 0 ∧ y^2 - b*y + b*p = 0) ↔ b = (p + 1)^2 ∨ b = 4*p :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_roots_l1154_115433


namespace NUMINAMATH_CALUDE_sum_of_coefficients_eq_120_l1154_115497

def binomial_coefficient (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def sum_of_coefficients : ℕ :=
  (Finset.range 8).sum (fun i => binomial_coefficient (i + 2) 2)

theorem sum_of_coefficients_eq_120 : sum_of_coefficients = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_eq_120_l1154_115497


namespace NUMINAMATH_CALUDE_right_triangle_tangent_circles_area_sum_l1154_115419

theorem right_triangle_tangent_circles_area_sum :
  ∀ (r s t : ℝ),
  r > 0 → s > 0 → t > 0 →
  r + s = 6 →
  r + t = 8 →
  s + t = 10 →
  (6 : ℝ)^2 + 8^2 = 10^2 →
  π * (r^2 + s^2 + t^2) = 36 * π := by
sorry

end NUMINAMATH_CALUDE_right_triangle_tangent_circles_area_sum_l1154_115419


namespace NUMINAMATH_CALUDE_bales_stored_is_difference_solution_l1154_115408

/-- The number of bales Tim stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Theorem stating that the number of bales Tim stored is the difference between the final and initial number of bales -/
theorem bales_stored_is_difference (initial_bales final_bales : ℕ) 
  (h : final_bales ≥ initial_bales) :
  bales_stored initial_bales final_bales = final_bales - initial_bales :=
by
  sorry

/-- The solution to the specific problem -/
theorem solution : bales_stored 28 54 = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_bales_stored_is_difference_solution_l1154_115408


namespace NUMINAMATH_CALUDE_exponential_function_through_point_l1154_115494

theorem exponential_function_through_point (a : ℝ) : 
  (∀ x : ℝ, (fun x => a^x) x = a^x) → 
  a^2 = 4 → 
  a > 0 → 
  a ≠ 1 → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_through_point_l1154_115494


namespace NUMINAMATH_CALUDE_f_simplification_f_equality_l1154_115422

def f (x : ℝ) : ℝ := 6 * x^2 + 4 * x - 2 * (x^2 - 1) - 2 * (2 * x + x^2)

theorem f_simplification (x : ℝ) : f x = 2 * x^2 + 2 := by sorry

theorem f_equality : f 3 = f (-3) ∧ f 3 = 20 := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_equality_l1154_115422


namespace NUMINAMATH_CALUDE_erased_odd_number_l1154_115424

/-- The sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ := n^2

/-- The sequence of odd numbers -/
def odd_sequence (n : ℕ) : ℕ := 2*n - 1

theorem erased_odd_number :
  ∃ (n : ℕ) (k : ℕ), k < n ∧ sum_odd_numbers n - odd_sequence k = 1998 →
  odd_sequence k = 27 :=
sorry

end NUMINAMATH_CALUDE_erased_odd_number_l1154_115424


namespace NUMINAMATH_CALUDE_larry_stickers_l1154_115430

/-- The number of stickers Larry loses -/
def lost_stickers : ℕ := 6

/-- The number of stickers Larry ends up with -/
def final_stickers : ℕ := 87

/-- The initial number of stickers Larry had -/
def initial_stickers : ℕ := final_stickers + lost_stickers

theorem larry_stickers : initial_stickers = 93 := by
  sorry

end NUMINAMATH_CALUDE_larry_stickers_l1154_115430


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_sqrt2_minus_half_achievable_min_value_l1154_115411

theorem min_value_of_expression (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  ∀ x y z w, x ≥ 0 ∧ y ≥ 0 ∧ z > 0 ∧ w > 0 ∧ z + w ≥ x + y →
  (b / (c + d)) + (c / (a + b)) ≤ (z / (w + y)) + (w / (x + z)) :=
by sorry

theorem min_value_sqrt2_minus_half (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  (b / (c + d)) + (c / (a + b)) ≥ Real.sqrt 2 - 1/2 :=
by sorry

theorem achievable_min_value (a d b c : ℝ) :
  ∃ a d b c, a ≥ 0 ∧ d ≥ 0 ∧ b > 0 ∧ c > 0 ∧ b + c ≥ a + d ∧
  (b / (c + d)) + (c / (a + b)) = Real.sqrt 2 - 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_sqrt2_minus_half_achievable_min_value_l1154_115411


namespace NUMINAMATH_CALUDE_wrapping_paper_usage_l1154_115466

theorem wrapping_paper_usage 
  (total_used : ℚ) 
  (num_presents : ℕ) 
  (h1 : total_used = 4 / 15) 
  (h2 : num_presents = 5) :
  total_used / num_presents = 4 / 75 :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_usage_l1154_115466


namespace NUMINAMATH_CALUDE_stratified_sample_correct_l1154_115450

/-- Represents the number of students in each category -/
structure StudentPopulation where
  total : ℕ
  junior : ℕ
  undergraduate : ℕ
  graduate : ℕ

/-- Represents the sample size and the number of students to be drawn from each category -/
structure SampleSize where
  total : ℕ
  junior : ℕ
  undergraduate : ℕ
  graduate : ℕ

/-- Calculates the correct sample size for stratified sampling -/
def calculateStratifiedSample (pop : StudentPopulation) (sampleTotal : ℕ) : SampleSize :=
  { total := sampleTotal,
    junior := (sampleTotal * pop.junior) / pop.total,
    undergraduate := (sampleTotal * pop.undergraduate) / pop.total,
    graduate := (sampleTotal * pop.graduate) / pop.total }

/-- Theorem: The calculated stratified sample is correct for the given population -/
theorem stratified_sample_correct (pop : StudentPopulation) (sample : SampleSize) :
  pop.total = 5400 ∧ 
  pop.junior = 1500 ∧ 
  pop.undergraduate = 3000 ∧ 
  pop.graduate = 900 ∧
  sample.total = 180 →
  calculateStratifiedSample pop sample.total = 
    { total := 180, junior := 50, undergraduate := 100, graduate := 30 } := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_correct_l1154_115450


namespace NUMINAMATH_CALUDE_accounting_class_average_score_l1154_115428

/-- The average score for an accounting class --/
def average_score (total_students : ℕ) 
  (day1_percent day2_percent day3_percent : ℚ)
  (day1_score day2_score day3_score : ℚ) : ℚ :=
  (day1_percent * day1_score + day2_percent * day2_score + day3_percent * day3_score) / 1

theorem accounting_class_average_score :
  let total_students : ℕ := 200
  let day1_percent : ℚ := 60 / 100
  let day2_percent : ℚ := 30 / 100
  let day3_percent : ℚ := 10 / 100
  let day1_score : ℚ := 65 / 100
  let day2_score : ℚ := 75 / 100
  let day3_score : ℚ := 95 / 100
  average_score total_students day1_percent day2_percent day3_percent day1_score day2_score day3_score = 71 / 100 := by
  sorry

end NUMINAMATH_CALUDE_accounting_class_average_score_l1154_115428


namespace NUMINAMATH_CALUDE_region_is_rectangle_l1154_115449

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The region defined by the given inequalities -/
def Region : Set Point2D :=
  {p : Point2D | -1 ≤ p.x ∧ p.x ≤ 1 ∧ 2 ≤ p.y ∧ p.y ≤ 4}

/-- Definition of a rectangle in 2D -/
def IsRectangle (S : Set Point2D) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ), x1 < x2 ∧ y1 < y2 ∧
    S = {p : Point2D | x1 ≤ p.x ∧ p.x ≤ x2 ∧ y1 ≤ p.y ∧ p.y ≤ y2}

/-- Theorem: The defined region is a rectangle -/
theorem region_is_rectangle : IsRectangle Region := by
  sorry

end NUMINAMATH_CALUDE_region_is_rectangle_l1154_115449


namespace NUMINAMATH_CALUDE_square_of_binomial_c_value_l1154_115402

theorem square_of_binomial_c_value (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + c = (a * x + b)^2) → c = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_c_value_l1154_115402


namespace NUMINAMATH_CALUDE_angle_range_l1154_115414

theorem angle_range (θ α : Real) : 
  (∃ (x y : Real), x = Real.sin (α - π/3) ∧ y = Real.sqrt 3 ∧ 
    x = Real.sin θ ∧ y = Real.cos θ) →
  Real.sin (2*θ) ≤ 0 →
  -2*π/3 ≤ α ∧ α ≤ π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_range_l1154_115414


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_relation_l1154_115489

/-- A geometric sequence with specific partial sums -/
structure GeometricSequence where
  S : ℝ  -- Sum of first 2 terms
  T : ℝ  -- Sum of first 4 terms
  R : ℝ  -- Sum of first 6 terms

/-- Theorem stating the relation between partial sums of a geometric sequence -/
theorem geometric_sequence_sum_relation (seq : GeometricSequence) :
  seq.S^2 + seq.T^2 = seq.S * (seq.T + seq.R) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_relation_l1154_115489


namespace NUMINAMATH_CALUDE_inequality_proof_l1154_115443

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (h_sum : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1154_115443


namespace NUMINAMATH_CALUDE_cos_42_cos_18_minus_cos_48_sin_18_l1154_115480

theorem cos_42_cos_18_minus_cos_48_sin_18 :
  Real.cos (42 * π / 180) * Real.cos (18 * π / 180) -
  Real.cos (48 * π / 180) * Real.sin (18 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_42_cos_18_minus_cos_48_sin_18_l1154_115480


namespace NUMINAMATH_CALUDE_probability_of_specific_sequence_l1154_115461

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of Jacks in a standard deck -/
def NumJacks : ℕ := 4

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Calculates the probability of drawing the specified sequence of cards -/
def probability_of_sequence : ℚ :=
  (NumKings : ℚ) / StandardDeck *
  (NumHearts - 1) / (StandardDeck - 1) *
  NumJacks / (StandardDeck - 2) *
  (NumSpades - 1) / (StandardDeck - 3) *
  NumQueens / (StandardDeck - 4)

theorem probability_of_specific_sequence :
  probability_of_sequence = 3 / 10125 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_sequence_l1154_115461


namespace NUMINAMATH_CALUDE_constant_d_value_l1154_115421

theorem constant_d_value (x y d : ℝ) 
  (h1 : x / (2 * y) = d / 2)
  (h2 : (7 * x + 4 * y) / (x - 2 * y) = 13) : 
  d = 5 := by
  sorry

end NUMINAMATH_CALUDE_constant_d_value_l1154_115421


namespace NUMINAMATH_CALUDE_ellipse_equation_l1154_115438

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 where a > b > 0,
    foci at (-2, 0) and (2, 0), and the product of slopes of lines from
    the left vertex to the intersection points of the ellipse with the
    circle having diameter F₁F₂ being 1/3, prove that the standard
    equation of the ellipse C is x²/6 + y²/2 = 1. -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∃ (x y : ℝ → ℝ), ∀ t, (x t)^2 / a^2 + (y t)^2 / b^2 = 1) →
  (∃ (F₁ F₂ : ℝ × ℝ), F₁ = (-2, 0) ∧ F₂ = (2, 0)) →
  (∃ (M N : ℝ × ℝ), M.1 > 0 ∧ M.2 > 0 ∧ N.1 < 0 ∧ N.2 > 0) →
  (∃ (A : ℝ × ℝ), A = (-a, 0)) →
  (∃ (m₁ m₂ : ℝ), m₁ * m₂ = 1/3) →
  (x^2 / 6 + y^2 / 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1154_115438


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1154_115404

theorem complex_magnitude_problem (z : ℂ) (h : z / (2 + Complex.I) = Complex.I ^ 2015 + Complex.I ^ 2016) : 
  Complex.abs z = Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1154_115404


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l1154_115437

theorem profit_percent_calculation (selling_price cost_price : ℝ) :
  selling_price = 2524.36 →
  cost_price = 2400 →
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  abs (profit_percent - 5.18) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l1154_115437


namespace NUMINAMATH_CALUDE_linear_equation_m_value_l1154_115482

/-- If (m+1)x + 3y^m = 5 is a linear equation in x and y, then m = 1 -/
theorem linear_equation_m_value (m : ℝ) : 
  (∀ x y : ℝ, ∃ a b c : ℝ, (m + 1) * x + 3 * y^m = a * x + b * y + c) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_m_value_l1154_115482


namespace NUMINAMATH_CALUDE_sequence_general_term_l1154_115436

theorem sequence_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + n) :
  ∀ n, a n = 2 * n :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1154_115436


namespace NUMINAMATH_CALUDE_parallelogram_construction_l1154_115406

-- Define the angle XOY
def Angle (O X Y : Point) : Prop := sorry

-- Define that a point is inside an angle
def InsideAngle (P : Point) (O X Y : Point) : Prop := sorry

-- Define that a point is on a line
def OnLine (P : Point) (A B : Point) : Prop := sorry

-- Define a parallelogram
def Parallelogram (A B C D : Point) : Prop := sorry

-- Define the theorem
theorem parallelogram_construction (O X Y A B : Point) 
  (h1 : Angle O X Y)
  (h2 : InsideAngle A O X Y)
  (h3 : InsideAngle B O X Y) :
  ∃ (C D : Point), 
    OnLine C O X ∧ 
    OnLine D O Y ∧ 
    Parallelogram A B C D :=
sorry

end NUMINAMATH_CALUDE_parallelogram_construction_l1154_115406


namespace NUMINAMATH_CALUDE_darnel_workout_l1154_115499

/-- Darnel's sprinting distances in miles -/
def sprint_distances : List ℝ := [0.8932, 0.7773, 0.9539, 0.5417, 0.6843]

/-- Darnel's jogging distances in miles -/
def jog_distances : List ℝ := [0.7683, 0.4231, 0.5733, 0.625, 0.6549]

/-- The difference between Darnel's total sprinting distance and total jogging distance -/
def sprint_jog_difference : ℝ := sprint_distances.sum - jog_distances.sum

theorem darnel_workout :
  sprint_jog_difference = 0.8058 := by sorry

end NUMINAMATH_CALUDE_darnel_workout_l1154_115499


namespace NUMINAMATH_CALUDE_negate_200_times_minus_one_l1154_115483

/-- Represents the result of negating a number n times -/
def negate_n_times (n : ℕ) : ℤ → ℤ :=
  match n with
  | 0 => id
  | n + 1 => λ x => -(negate_n_times n x)

/-- The theorem states that negating -1 200 times results in -1 -/
theorem negate_200_times_minus_one :
  negate_n_times 200 (-1) = -1 := by sorry

end NUMINAMATH_CALUDE_negate_200_times_minus_one_l1154_115483


namespace NUMINAMATH_CALUDE_shekars_social_studies_score_l1154_115474

/-- Given Shekar's scores in four subjects and his average marks, prove his score in social studies -/
theorem shekars_social_studies_score 
  (math_score science_score english_score biology_score : ℕ)
  (average_score : ℚ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 67)
  (h4 : biology_score = 75)
  (h5 : average_score = 73)
  (h6 : average_score = (math_score + science_score + english_score + biology_score + social_studies_score) / 5) :
  social_studies_score = 82 := by
  sorry

end NUMINAMATH_CALUDE_shekars_social_studies_score_l1154_115474


namespace NUMINAMATH_CALUDE_wire_length_proof_l1154_115446

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 30 ∧ 
  shorter_piece = (3/5) * longer_piece ∧
  total_length = shorter_piece + longer_piece →
  total_length = 80 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l1154_115446


namespace NUMINAMATH_CALUDE_smallest_total_score_l1154_115476

theorem smallest_total_score : 
  ∃ (T : ℕ), T > 0 ∧ 
  (∃ (n m : ℕ), 2 * n + 5 * m = T ∧ (n ≥ m + 3 ∨ m ≥ n + 3)) ∧ 
  (∀ (S : ℕ), S > 0 → S < T → 
    ¬(∃ (n m : ℕ), 2 * n + 5 * m = S ∧ (n ≥ m + 3 ∨ m ≥ n + 3))) ∧
  T = 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_total_score_l1154_115476


namespace NUMINAMATH_CALUDE_converse_of_negative_square_positive_l1154_115460

theorem converse_of_negative_square_positive :
  (∀ x : ℝ, x < 0 → x^2 > 0) →
  (∀ x : ℝ, x^2 > 0 → x < 0) :=
sorry

end NUMINAMATH_CALUDE_converse_of_negative_square_positive_l1154_115460


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1154_115478

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the property of a line passing through a point
def passesThrough (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the property of a line having equal intercepts
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.a = l.b ∨ (l.a = 0 ∧ l.c = 0) ∨ (l.b = 0 ∧ l.c = 0)

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  ∃ (l₁ l₂ : Line2D),
    (passesThrough l₁ ⟨2, 3⟩ ∧ hasEqualIntercepts l₁ ∧ l₁ = ⟨1, 1, -5⟩) ∧
    (passesThrough l₂ ⟨2, 3⟩ ∧ hasEqualIntercepts l₂ ∧ l₂ = ⟨3, -2, 0⟩) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1154_115478


namespace NUMINAMATH_CALUDE_count_satisfying_numbers_l1154_115415

/-- Represents a two-digit number in the dozenal (base 12) system -/
structure DozenalNumber :=
  (tens : Nat)
  (ones : Nat)
  (tens_valid : 1 ≤ tens ∧ tens ≤ 11)
  (ones_valid : ones ≤ 11)

/-- Converts a DozenalNumber to its decimal representation -/
def toDecimal (n : DozenalNumber) : Nat :=
  12 * n.tens + n.ones

/-- Calculates the sum of digits of a DozenalNumber -/
def digitSum (n : DozenalNumber) : Nat :=
  n.tens + n.ones

/-- Checks if a DozenalNumber satisfies the given condition -/
def satisfiesCondition (n : DozenalNumber) : Prop :=
  (toDecimal n - digitSum n) % 12 = 5

theorem count_satisfying_numbers :
  ∃ (numbers : Finset DozenalNumber),
    numbers.card = 12 ∧
    (∀ n : DozenalNumber, n ∈ numbers ↔ satisfiesCondition n) :=
sorry

end NUMINAMATH_CALUDE_count_satisfying_numbers_l1154_115415


namespace NUMINAMATH_CALUDE_sum_of_x_values_l1154_115412

theorem sum_of_x_values (x : ℝ) : 
  (50 < x ∧ x < 150) →
  (Real.cos (2 * x * π / 180))^3 + (Real.cos (6 * x * π / 180))^3 = 
    8 * (Real.cos (4 * x * π / 180))^3 * (Real.cos (x * π / 180))^3 →
  ∃ (s : Finset ℝ), (∀ y ∈ s, 
    (50 < y ∧ y < 150) ∧
    (Real.cos (2 * y * π / 180))^3 + (Real.cos (6 * y * π / 180))^3 = 
      8 * (Real.cos (4 * y * π / 180))^3 * (Real.cos (y * π / 180))^3) ∧
  (s.sum id = 270) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_values_l1154_115412


namespace NUMINAMATH_CALUDE_stating_num_small_triangles_formula_l1154_115416

/-- Represents a triangle with n points inside it -/
structure TriangleWithPoints where
  n : ℕ  -- number of points inside the triangle
  no_collinear : Bool  -- property that no three points are collinear

/-- 
  Calculates the number of small triangles formed in a triangle with n internal points,
  where no three points (including the triangle's vertices) are collinear.
-/
def numSmallTriangles (t : TriangleWithPoints) : ℕ :=
  2 * t.n + 1

/-- 
  Theorem stating that for a triangle with n points inside,
  where no three points are collinear (including the triangle's vertices),
  the number of small triangles formed is 2n + 1.
-/
theorem num_small_triangles_formula (t : TriangleWithPoints) 
  (h : t.no_collinear = true) : 
  numSmallTriangles t = 2 * t.n + 1 := by
  sorry

#eval numSmallTriangles { n := 100, no_collinear := true }

end NUMINAMATH_CALUDE_stating_num_small_triangles_formula_l1154_115416


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1154_115435

/-- A right triangle with perimeter 40 and area 30 has a hypotenuse of length 18.5 -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  a + b + c = 40 →   -- Perimeter condition
  a * b / 2 = 30 →   -- Area condition
  c = 18.5 := by
    sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1154_115435


namespace NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1154_115429

def age_ratio (a_current : ℕ) (b_current : ℕ) : ℚ :=
  (a_current + 20) / (b_current - 20)

theorem age_ratio_is_two_to_one :
  ∀ (a_current b_current : ℕ),
    b_current = 70 →
    a_current = b_current + 10 →
    age_ratio a_current b_current = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_two_to_one_l1154_115429


namespace NUMINAMATH_CALUDE_lecture_duration_l1154_115493

/-- 
Given a lecture that lasts for 2 hours and m minutes, where the positions of the
hour and minute hands on the clock at the end of the lecture are exactly swapped
from their positions at the beginning, this theorem states that the integer part
of m is 46.
-/
theorem lecture_duration (m : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t < 120 + m → 
    (5 * (120 + m - t) = (60 * t) % 360 ∨ 5 * (120 + m - t) = ((60 * t) % 360 + 360) % 360)) →
  Int.floor m = 46 := by
  sorry

end NUMINAMATH_CALUDE_lecture_duration_l1154_115493


namespace NUMINAMATH_CALUDE_sara_final_quarters_l1154_115495

/-- Calculates the final number of quarters Sara has after a series of transactions -/
def sara_quarters (initial : ℕ) (from_dad : ℕ) (spent : ℕ) (dollars_from_mom : ℕ) (quarters_per_dollar : ℕ) : ℕ :=
  initial + from_dad - spent + dollars_from_mom * quarters_per_dollar

/-- Theorem stating that Sara ends up with 63 quarters -/
theorem sara_final_quarters : 
  sara_quarters 21 49 15 2 4 = 63 := by
  sorry

end NUMINAMATH_CALUDE_sara_final_quarters_l1154_115495


namespace NUMINAMATH_CALUDE_physics_marks_correct_l1154_115452

/-- Given a student's marks in four subjects and their average across five subjects,
    calculate the marks in the fifth subject. -/
def calculate_physics_marks (e m c b : ℕ) (avg : ℚ) (n : ℕ) : ℚ :=
  n * avg - (e + m + c + b)

/-- Theorem stating that the calculated physics marks are correct given the problem conditions. -/
theorem physics_marks_correct 
  (e m c b : ℕ) 
  (avg : ℚ) 
  (n : ℕ) 
  (h1 : e = 70) 
  (h2 : m = 60) 
  (h3 : c = 60) 
  (h4 : b = 65) 
  (h5 : avg = 66.6) 
  (h6 : n = 5) : 
  calculate_physics_marks e m c b avg n = 78 := by
sorry

#eval calculate_physics_marks 70 60 60 65 66.6 5

end NUMINAMATH_CALUDE_physics_marks_correct_l1154_115452


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l1154_115458

theorem greatest_integer_inequality : ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l1154_115458


namespace NUMINAMATH_CALUDE_b_31_mod_33_l1154_115490

/-- Definition of b_n as the concatenation of integers from 1 to n --/
def b (n : ℕ) : ℕ :=
  -- This is a placeholder definition. The actual implementation would be more complex.
  sorry

/-- Theorem stating that b_31 mod 33 = 11 --/
theorem b_31_mod_33 : b 31 % 33 = 11 := by
  sorry

end NUMINAMATH_CALUDE_b_31_mod_33_l1154_115490


namespace NUMINAMATH_CALUDE_max_ice_creams_l1154_115441

/-- Given a budget and costs of items, calculate the maximum number of ice creams that can be bought -/
theorem max_ice_creams (budget : ℕ) (pancake_cost ice_cream_cost pancakes_bought : ℕ) : 
  budget = 60 →
  pancake_cost = 5 →
  ice_cream_cost = 8 →
  pancakes_bought = 5 →
  (budget - pancake_cost * pancakes_bought) / ice_cream_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_ice_creams_l1154_115441


namespace NUMINAMATH_CALUDE_teacher_age_proof_l1154_115469

def teacher_age (num_students : ℕ) (student_avg_age : ℕ) (new_avg_age : ℕ) (total_people : ℕ) : ℕ :=
  (new_avg_age * total_people) - (student_avg_age * num_students)

theorem teacher_age_proof :
  teacher_age 23 22 23 24 = 46 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_proof_l1154_115469


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l1154_115456

theorem quadratic_solution_property : 
  ∀ p q : ℝ, 
  (2 * p^2 + 8 * p - 42 = 0) → 
  (2 * q^2 + 8 * q - 42 = 0) → 
  p ≠ q → 
  (p - q + 2)^2 = 144 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l1154_115456


namespace NUMINAMATH_CALUDE_f_of_4_equals_23_l1154_115432

-- Define the function f
def f : ℝ → ℝ := fun x => 2 * (2 * x + 2) + 3

-- State the theorem
theorem f_of_4_equals_23 : f 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_f_of_4_equals_23_l1154_115432


namespace NUMINAMATH_CALUDE_solve_system_l1154_115400

theorem solve_system (a b : ℚ) 
  (eq1 : 2020 * a + 2024 * b = 2028)
  (eq2 : 2022 * a + 2026 * b = 2030) : 
  a - b = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1154_115400


namespace NUMINAMATH_CALUDE_least_number_with_remainder_five_forty_five_satisfies_least_number_is_545_l1154_115439

theorem least_number_with_remainder (n : ℕ) : 
  (n % 12 = 5 ∧ n % 15 = 5 ∧ n % 20 = 5 ∧ n % 54 = 5) → n ≥ 545 := by
  sorry

theorem five_forty_five_satisfies :
  545 % 12 = 5 ∧ 545 % 15 = 5 ∧ 545 % 20 = 5 ∧ 545 % 54 = 5 := by
  sorry

theorem least_number_is_545 : 
  ∃! n : ℕ, (n % 12 = 5 ∧ n % 15 = 5 ∧ n % 20 = 5 ∧ n % 54 = 5) ∧
  ∀ m : ℕ, (m % 12 = 5 ∧ m % 15 = 5 ∧ m % 20 = 5 ∧ m % 54 = 5) → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_five_forty_five_satisfies_least_number_is_545_l1154_115439


namespace NUMINAMATH_CALUDE_backpack_and_weight_difference_l1154_115491

/-- Given the weights of Bridget and Martha, and their combined weight with a backpack,
    prove the weight of the backpack and the weight difference between Bridget and Martha. -/
theorem backpack_and_weight_difference 
  (bridget_weight : ℕ) 
  (martha_weight : ℕ) 
  (combined_weight_with_backpack : ℕ) 
  (h1 : bridget_weight = 39)
  (h2 : martha_weight = 2)
  (h3 : combined_weight_with_backpack = 60) :
  (∃ backpack_weight : ℕ, 
    backpack_weight = combined_weight_with_backpack - (bridget_weight + martha_weight) ∧ 
    backpack_weight = 19) ∧ 
  (bridget_weight - martha_weight = 37) := by
  sorry

end NUMINAMATH_CALUDE_backpack_and_weight_difference_l1154_115491


namespace NUMINAMATH_CALUDE_function_product_l1154_115467

theorem function_product (f : ℕ → ℝ) 
  (h₁ : ∀ n : ℕ, n > 0 → f (n + 3) = (f n - 1) / (f n + 1))
  (h₂ : f 1 ≠ 0)
  (h₃ : f 1 ≠ 1 ∧ f 1 ≠ -1) :
  f 8 * f 2018 = -1 := by
  sorry

end NUMINAMATH_CALUDE_function_product_l1154_115467


namespace NUMINAMATH_CALUDE_negative_integers_abs_not_greater_than_4_l1154_115442

def negativeIntegersWithAbsNotGreaterThan4 : Set ℤ :=
  {x : ℤ | x < 0 ∧ |x| ≤ 4}

theorem negative_integers_abs_not_greater_than_4 :
  negativeIntegersWithAbsNotGreaterThan4 = {-1, -2, -3, -4} := by
  sorry

end NUMINAMATH_CALUDE_negative_integers_abs_not_greater_than_4_l1154_115442


namespace NUMINAMATH_CALUDE_gcf_360_150_l1154_115479

theorem gcf_360_150 : Nat.gcd 360 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_360_150_l1154_115479


namespace NUMINAMATH_CALUDE_tailor_trim_problem_l1154_115451

/-- Given a square cloth with side length 18 feet, if 4 feet are trimmed from two opposite edges
    and x feet are trimmed from the other two edges, resulting in 120 square feet of remaining cloth,
    then x = 6. -/
theorem tailor_trim_problem (x : ℝ) : 
  (18 : ℝ) > 0 ∧ x > 0 ∧ (18 - 4 - 4 : ℝ) * (18 - x) = 120 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_tailor_trim_problem_l1154_115451


namespace NUMINAMATH_CALUDE_inequality_solution_l1154_115423

theorem inequality_solution (x : ℕ) : 
  (x + 3 : ℚ) / (x^2 - 4) - 1 / (x + 2) < 2 * x / (2 * x - x^2) ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1154_115423


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1154_115457

/-- The quadratic function f(x) = x^2 - 16x + p + 3 -/
def f (p : ℝ) (x : ℝ) : ℝ := x^2 - 16*x + p + 3

/-- The function has a zero in the interval [-1, 1] -/
def has_zero_in_interval (p : ℝ) : Prop :=
  ∃ x, x ∈ Set.Icc (-1) 1 ∧ f p x = 0

/-- The range of f(x) when x ∈ [q, 10] is an interval with length 12 - q -/
def range_is_interval_with_length (p q : ℝ) : Prop :=
  ∃ a b, Set.Icc a b = Set.image (f p) (Set.Icc q 10) ∧ b - a = 12 - q

theorem quadratic_function_properties :
  (∀ p, has_zero_in_interval p ↔ -20 ≤ p ∧ p ≤ 12) ∧
  (∃ q, q ≥ 0 ∧ range_is_interval_with_length p q ↔ 
    q = 8 ∨ q = 9 ∨ q = (15 - Real.sqrt 17) / 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1154_115457


namespace NUMINAMATH_CALUDE_one_cow_one_bag_days_l1154_115447

/-- Given that 52 cows eat 104 bags of husk in 78 days, 
    prove that it takes 39 days for one cow to eat one bag of husk. -/
theorem one_cow_one_bag_days (cows : ℕ) (bags : ℕ) (days : ℕ) 
  (h1 : cows = 52) 
  (h2 : bags = 104) 
  (h3 : days = 78) : 
  (bags * days) / (cows * bags) = 39 := by
  sorry

end NUMINAMATH_CALUDE_one_cow_one_bag_days_l1154_115447


namespace NUMINAMATH_CALUDE_min_value_of_sequence_l1154_115418

/-- Given a sequence {a_n} where a₂ = 102 and aₙ₊₁ - aₙ = 4n for n ∈ ℕ*, 
    the minimum value of {aₙ/n} is 26. -/
theorem min_value_of_sequence (a : ℕ → ℝ) : 
  (a 2 = 102) → 
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = 4 * n) → 
  (∃ n₀ : ℕ, n₀ ≥ 1 ∧ a n₀ / n₀ = 26) ∧ 
  (∀ n : ℕ, n ≥ 1 → a n / n ≥ 26) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sequence_l1154_115418


namespace NUMINAMATH_CALUDE_constant_function_if_arithmetic_mean_l1154_115444

def IsArithmeticMean (f : ℤ × ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4

theorem constant_function_if_arithmetic_mean (f : ℤ × ℤ → ℤ) 
  (h1 : ∀ x y : ℤ, f (x, y) > 0)
  (h2 : IsArithmeticMean f) :
  ∃ c : ℤ, ∀ x y : ℤ, f (x, y) = c := by
  sorry

end NUMINAMATH_CALUDE_constant_function_if_arithmetic_mean_l1154_115444


namespace NUMINAMATH_CALUDE_technician_count_l1154_115445

/-- Proves the number of technicians in a workshop given specific salary and worker information --/
theorem technician_count (total_workers : ℕ) (avg_salary_all : ℚ) (avg_salary_tech : ℚ) (avg_salary_non_tech : ℚ) 
  (h1 : total_workers = 12)
  (h2 : avg_salary_all = 9500)
  (h3 : avg_salary_tech = 12000)
  (h4 : avg_salary_non_tech = 6000) :
  ∃ (tech_count : ℕ), tech_count = 7 ∧ tech_count ≤ total_workers :=
by sorry

end NUMINAMATH_CALUDE_technician_count_l1154_115445


namespace NUMINAMATH_CALUDE_smallest_x_y_sum_l1154_115448

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def is_fourth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 4

theorem smallest_x_y_sum :
  ∃! (x y : ℕ), 
    x > 0 ∧ y > 0 ∧
    is_square (720 * x) ∧
    is_fourth_power (720 * y) ∧
    (∀ (x' y' : ℕ), x' > 0 ∧ y' > 0 ∧ 
      is_square (720 * x') ∧ is_fourth_power (720 * y') → 
      x ≤ x' ∧ y ≤ y') ∧
    x + y = 1130 :=
  sorry

end NUMINAMATH_CALUDE_smallest_x_y_sum_l1154_115448


namespace NUMINAMATH_CALUDE_difference_of_squares_l1154_115434

theorem difference_of_squares (a b : ℝ) : (3*a + b) * (3*a - b) = 9*a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1154_115434


namespace NUMINAMATH_CALUDE_odd_prime_condition_l1154_115431

theorem odd_prime_condition (p : ℕ) (h_prime : Nat.Prime p) : 
  (∃! k : ℕ, Even k ∧ k ∣ (14 * p)) → Odd p :=
sorry

end NUMINAMATH_CALUDE_odd_prime_condition_l1154_115431
