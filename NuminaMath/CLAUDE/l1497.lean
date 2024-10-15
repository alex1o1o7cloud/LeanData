import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l1497_149702

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / x + 1 / y = 5) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 1 / y₀ = 5 ∧ 3 * x₀ + 4 * y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1497_149702


namespace NUMINAMATH_CALUDE_first_digit_base9_of_122012_base3_l1497_149778

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 3 * acc) 0

/-- Calculates the first digit of a number in base 9 -/
def firstDigitBase9 (n : Nat) : Nat :=
  if n < 9 then n else firstDigitBase9 (n / 9)

theorem first_digit_base9_of_122012_base3 :
  let y := base3ToBase10 [1, 2, 2, 0, 1, 2]
  firstDigitBase9 y = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_base9_of_122012_base3_l1497_149778


namespace NUMINAMATH_CALUDE_root_power_floor_l1497_149746

theorem root_power_floor (a : ℝ) : 
  a^5 - a^3 + a - 2 = 0 → ⌊a^6⌋ = 3 := by
sorry

end NUMINAMATH_CALUDE_root_power_floor_l1497_149746


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l1497_149708

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define symmetry about y-axis
def symmetric_about_y_axis (p q : Point) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis (a, 3) (4, b) →
  (a + b)^2008 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l1497_149708


namespace NUMINAMATH_CALUDE_sam_initial_nickels_l1497_149732

/-- Given information about Sam's nickels --/
structure SamNickels where
  initial : ℕ  -- Initial number of nickels
  given : ℕ    -- Number of nickels given by dad
  final : ℕ    -- Final number of nickels

/-- Theorem stating the initial number of nickels Sam had --/
theorem sam_initial_nickels (s : SamNickels) (h : s.final = s.initial + s.given) 
  (h_final : s.final = 63) (h_given : s.given = 39) : s.initial = 24 := by
  sorry

#check sam_initial_nickels

end NUMINAMATH_CALUDE_sam_initial_nickels_l1497_149732


namespace NUMINAMATH_CALUDE_square_floor_tiles_l1497_149776

theorem square_floor_tiles (black_tiles : ℕ) (total_tiles : ℕ) : 
  black_tiles = 101 → 
  (∃ (side_length : ℕ), 
    side_length * side_length = total_tiles ∧ 
    2 * side_length - 1 = black_tiles) → 
  total_tiles = 2601 :=
by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l1497_149776


namespace NUMINAMATH_CALUDE_sin_odd_function_phi_l1497_149747

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem sin_odd_function_phi (φ : ℝ) :
  is_odd_function (λ x => Real.sin (x + φ)) → φ = π :=
sorry

end NUMINAMATH_CALUDE_sin_odd_function_phi_l1497_149747


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1497_149725

theorem complex_equation_solution (x y z : ℂ) (h_real : x.im = 0)
  (h_sum : x + y + z = 5)
  (h_prod_sum : x * y + y * z + z * x = 5)
  (h_prod : x * y * z = 5) :
  x = 1 + (4 : ℂ) ^ (1/3 : ℂ) :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1497_149725


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1497_149719

theorem smallest_positive_integer_with_remainders : ∃! x : ℕ+, 
  (x : ℤ) % 4 = 1 ∧ 
  (x : ℤ) % 5 = 2 ∧ 
  (x : ℤ) % 6 = 3 ∧ 
  ∀ y : ℕ+, 
    (y : ℤ) % 4 = 1 → 
    (y : ℤ) % 5 = 2 → 
    (y : ℤ) % 6 = 3 → 
    x ≤ y :=
by
  use 57
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1497_149719


namespace NUMINAMATH_CALUDE_carl_needs_sixty_more_bags_l1497_149701

/-- The number of additional gift bags Carl needs to make for his open house -/
def additional_bags_needed (guaranteed_visitors : ℕ) (possible_visitors : ℕ) (extravagant_bags : ℕ) (average_bags : ℕ) : ℕ :=
  (guaranteed_visitors + possible_visitors) - (extravagant_bags + average_bags)

/-- Theorem stating that Carl needs to make 60 more gift bags -/
theorem carl_needs_sixty_more_bags :
  additional_bags_needed 50 40 10 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_carl_needs_sixty_more_bags_l1497_149701


namespace NUMINAMATH_CALUDE_max_profit_at_85_optimal_selling_price_l1497_149740

/-- Represents the profit function for the item sales --/
def profit (x : ℝ) : ℝ := (10 + x) * (400 - 20 * x) - 500

/-- Theorem stating that the maximum profit is achieved at a selling price of 85 yuan --/
theorem max_profit_at_85 :
  ∃ (x : ℝ), x > 0 ∧ x < 20 ∧
  ∀ (y : ℝ), y > 0 → y < 20 → profit x ≥ profit y ∧
  x + 80 = 85 := by
  sorry

/-- Corollary: The selling price that maximizes profit is 85 yuan --/
theorem optimal_selling_price : 
  ∃ (x : ℝ), x > 0 ∧ x < 20 ∧
  ∀ (y : ℝ), y > 0 → y < 20 → profit x ≥ profit y ∧
  x + 80 = 85 := by
  exact max_profit_at_85

end NUMINAMATH_CALUDE_max_profit_at_85_optimal_selling_price_l1497_149740


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l1497_149794

theorem rectangle_area_diagonal (length width diagonal k : ℝ) : 
  length > 0 →
  width > 0 →
  diagonal > 0 →
  length / width = 5 / 2 →
  diagonal^2 = length^2 + width^2 →
  k = 10 / 29 →
  length * width = k * diagonal^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l1497_149794


namespace NUMINAMATH_CALUDE_factorization_difference_l1497_149721

theorem factorization_difference (a b : ℤ) : 
  (∀ y : ℝ, 5 * y^2 + 3 * y - 44 = (5 * y + a) * (y + b)) → 
  a - b = -15 := by
sorry

end NUMINAMATH_CALUDE_factorization_difference_l1497_149721


namespace NUMINAMATH_CALUDE_mrs_lee_june_percentage_l1497_149714

/-- Represents the Lee family's income structure -/
structure LeeIncome where
  total : ℝ
  mrs_lee : ℝ
  mr_lee : ℝ
  jack : ℝ
  rest : ℝ

/-- Calculates the total income for June based on May's income and the given changes -/
def june_total (may : LeeIncome) : ℝ :=
  1.2 * may.mrs_lee + 1.1 * may.mr_lee + 0.85 * may.jack + may.rest

/-- Theorem stating that Mrs. Lee's earnings in June are between 0% and 60% of the total income -/
theorem mrs_lee_june_percentage (may : LeeIncome)
  (h1 : may.mrs_lee = 0.5 * may.total)
  (h2 : may.total = may.mrs_lee + may.mr_lee + may.jack + may.rest)
  (h3 : may.total > 0) :
  0 < (1.2 * may.mrs_lee) / (june_total may) ∧ (1.2 * may.mrs_lee) / (june_total may) < 0.6 := by
  sorry

end NUMINAMATH_CALUDE_mrs_lee_june_percentage_l1497_149714


namespace NUMINAMATH_CALUDE_painted_area_calculation_l1497_149773

/-- Given a rectangular exhibition space with specific dimensions and border widths,
    calculate the area of the painted region inside the border. -/
theorem painted_area_calculation (total_width total_length border_width_standard border_width_door : ℕ)
    (h1 : total_width = 100)
    (h2 : total_length = 150)
    (h3 : border_width_standard = 15)
    (h4 : border_width_door = 20) :
    (total_width - 2 * border_width_standard) * (total_length - border_width_standard - border_width_door) = 8050 :=
by sorry

end NUMINAMATH_CALUDE_painted_area_calculation_l1497_149773


namespace NUMINAMATH_CALUDE_remainder_512_210_mod_13_l1497_149728

theorem remainder_512_210_mod_13 : 512^210 % 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_512_210_mod_13_l1497_149728


namespace NUMINAMATH_CALUDE_function_with_two_zeros_m_range_l1497_149723

theorem function_with_two_zeros_m_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m < -2 ∨ m > 2 :=
sorry

end NUMINAMATH_CALUDE_function_with_two_zeros_m_range_l1497_149723


namespace NUMINAMATH_CALUDE_unique_quadratic_polynomial_l1497_149754

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  b : ℝ
  c : ℝ

/-- The roots of a quadratic polynomial -/
def roots (p : QuadraticPolynomial) : Set ℝ :=
  {x : ℝ | x^2 + p.b * x + p.c = 0}

/-- The set of coefficients of a quadratic polynomial -/
def coefficients (p : QuadraticPolynomial) : Set ℝ :=
  {1, p.b, p.c}

/-- The theorem stating that there exists exactly one quadratic polynomial
    satisfying the given conditions -/
theorem unique_quadratic_polynomial :
  ∃! p : QuadraticPolynomial, roots p = coefficients p :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_polynomial_l1497_149754


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l1497_149737

theorem quadratic_equation_integer_solutions (k : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ (k^2 - 1) * x^2 - 6 * (3 * k - 1) * x + 72 = 0) ↔ 
  k = 1 ∨ k = 2 ∨ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l1497_149737


namespace NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l1497_149781

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_50th_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_first : a 1 = 7)
  (h_fifteenth : a 15 = 41) :
  a 50 = 126 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_50th_term_l1497_149781


namespace NUMINAMATH_CALUDE_new_average_weight_l1497_149779

theorem new_average_weight (initial_count : ℕ) (initial_avg : ℚ) (new_weight : ℚ) :
  initial_count = 19 →
  initial_avg = 15 →
  new_weight = 13 →
  let total_weight := initial_count * initial_avg + new_weight
  let new_count := initial_count + 1
  new_count * (total_weight / new_count) = 298 :=
by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l1497_149779


namespace NUMINAMATH_CALUDE_exhibit_visit_time_l1497_149783

/-- Represents the time taken by each group to visit the exhibit -/
def group_time (students_per_group : ℕ) (time_per_student : ℕ) : ℕ :=
  students_per_group * time_per_student

/-- Calculates the total time for all groups to visit the exhibit -/
def total_exhibit_time (total_students : ℕ) (num_groups : ℕ) (group_times : List ℕ) : ℕ :=
  let students_per_group := total_students / num_groups
  (group_times.map (group_time students_per_group)).sum

/-- Theorem stating the total time for all groups to visit the exhibit -/
theorem exhibit_visit_time : 
  total_exhibit_time 30 5 [4, 5, 6, 7, 8] = 180 := by
  sorry

#eval total_exhibit_time 30 5 [4, 5, 6, 7, 8]

end NUMINAMATH_CALUDE_exhibit_visit_time_l1497_149783


namespace NUMINAMATH_CALUDE_distance_between_x_intercepts_specific_case_l1497_149706

/-- Two lines in a 2D plane -/
structure TwoLines where
  intersection : ℝ × ℝ
  slope1 : ℝ
  slope2 : ℝ

/-- Calculate the distance between x-intercepts of two lines -/
def distance_between_x_intercepts (lines : TwoLines) : ℝ :=
  sorry

/-- The main theorem -/
theorem distance_between_x_intercepts_specific_case :
  let lines : TwoLines := {
    intersection := (12, 20),
    slope1 := 7/2,
    slope2 := -3/2
  }
  distance_between_x_intercepts lines = 800/21 := by sorry

end NUMINAMATH_CALUDE_distance_between_x_intercepts_specific_case_l1497_149706


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1497_149755

/-- Calculate the interest rate per annum given the principal, amount, and time period. -/
theorem interest_rate_calculation (principal amount : ℕ) (time : ℚ) :
  principal = 1100 →
  amount = 1232 →
  time = 12 / 5 →
  (amount - principal) * 100 / (principal * time) = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1497_149755


namespace NUMINAMATH_CALUDE_inscribed_triangles_inequality_l1497_149780

/-- Two equilateral triangles inscribed in a circle -/
structure InscribedTriangles where
  r : ℝ
  S : ℝ

/-- Theorem: For two equilateral triangles inscribed in a circle with radius r,
    where S is the area of their common part, 2S ≥ √3 r² holds. -/
theorem inscribed_triangles_inequality (t : InscribedTriangles) : 2 * t.S ≥ Real.sqrt 3 * t.r^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangles_inequality_l1497_149780


namespace NUMINAMATH_CALUDE_two_solutions_for_x_squared_minus_y_squared_77_l1497_149777

theorem two_solutions_for_x_squared_minus_y_squared_77 :
  ∃! n : ℕ, n > 0 ∧ 
  (∃ s : Finset (ℕ × ℕ), s.card = n ∧ 
    (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 - p.2^2 = 77) ∧
    (∀ x y : ℕ, x > 0 → y > 0 → x^2 - y^2 = 77 → (x, y) ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_for_x_squared_minus_y_squared_77_l1497_149777


namespace NUMINAMATH_CALUDE_bike_distance_l1497_149730

/-- Theorem: A bike moving at a constant speed of 4 m/s for 8 seconds travels 32 meters. -/
theorem bike_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 4 → time = 8 → distance = speed * time → distance = 32 := by
  sorry

end NUMINAMATH_CALUDE_bike_distance_l1497_149730


namespace NUMINAMATH_CALUDE_at_least_one_less_than_one_l1497_149707

theorem at_least_one_less_than_one (a b c : ℝ) (ha : a < 3) (hb : b < 3) (hc : c < 3) :
  a < 1 ∨ b < 1 ∨ c < 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_one_l1497_149707


namespace NUMINAMATH_CALUDE_sequence_problem_l1497_149718

theorem sequence_problem (a b c : ℤ → ℝ) 
  (h_positive : ∀ n, a n > 0 ∧ b n > 0 ∧ c n > 0)
  (h_a : ∀ n, a n ≥ (b (n+1) + c (n-1)) / 2)
  (h_b : ∀ n, b n ≥ (c (n+1) + a (n-1)) / 2)
  (h_c : ∀ n, c n ≥ (a (n+1) + b (n-1)) / 2)
  (h_init : a 0 = 26 ∧ b 0 = 6 ∧ c 0 = 2004) :
  a 2005 = 2004 ∧ b 2005 = 26 ∧ c 2005 = 6 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1497_149718


namespace NUMINAMATH_CALUDE_unique_f_three_l1497_149729

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem unique_f_three (f : RealFunction) 
  (h : ∀ x y : ℝ, f x * f y - f (x + y) = x - y) : 
  f 3 = -3 := by sorry

end NUMINAMATH_CALUDE_unique_f_three_l1497_149729


namespace NUMINAMATH_CALUDE_seventh_triangular_number_l1497_149784

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The seventh triangular number is 28 -/
theorem seventh_triangular_number : triangular_number 7 = 28 := by sorry

end NUMINAMATH_CALUDE_seventh_triangular_number_l1497_149784


namespace NUMINAMATH_CALUDE_max_value_theorem_l1497_149798

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  (a^2 * b^2) / (a + b) + (a^2 * c^2) / (a + c) + (b^2 * c^2) / (b + c) ≤ 1/6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1497_149798


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_angle_l1497_149770

/-- The equation of a line passing through a given point with a given slope angle -/
theorem line_equation_through_point_with_slope_angle 
  (x₀ y₀ : ℝ) (θ : ℝ) :
  x₀ = Real.sqrt 3 →
  y₀ = 1 →
  θ = π / 3 →
  ∃ (a b c : ℝ), 
    a * Real.sqrt 3 + b * 1 + c = 0 ∧
    a * x + b * y + c = 0 ∧
    a = Real.sqrt 3 ∧
    b = -1 ∧
    c = -2 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_angle_l1497_149770


namespace NUMINAMATH_CALUDE_intersection_of_solutions_range_of_a_l1497_149771

-- Define the conditions
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) : Prop := -x^2 + 5*x - 6 ≥ 0

-- Part 1: Intersection of solutions when a = 1
theorem intersection_of_solutions :
  {x : ℝ | p x 1 ∧ q x} = {x : ℝ | 2 ≤ x ∧ x < 3} :=
sorry

-- Part 2: Range of a for which ¬p ↔ ¬q
theorem range_of_a :
  {a : ℝ | a > 0 ∧ ∀ x, ¬(p x a) ↔ ¬(q x)} = {a : ℝ | 1 < a ∧ a < 2} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_solutions_range_of_a_l1497_149771


namespace NUMINAMATH_CALUDE_f_range_l1497_149735

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x ^ 3 + 2 * Real.sin x ^ 2 - 4 * Real.sin x + 3 * Real.cos x + 3 * Real.cos x ^ 2 - 2) / (Real.sin x - 1)

theorem f_range :
  Set.range (fun (x : ℝ) => f x) = Set.Icc 1 (1 + 3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_f_range_l1497_149735


namespace NUMINAMATH_CALUDE_store_coloring_books_l1497_149745

/-- The number of coloring books sold during the sale -/
def books_sold : ℕ := 39

/-- The number of shelves used after the sale -/
def shelves : ℕ := 9

/-- The number of books on each shelf after the sale -/
def books_per_shelf : ℕ := 9

/-- The initial number of coloring books in stock -/
def initial_stock : ℕ := books_sold + shelves * books_per_shelf

theorem store_coloring_books : initial_stock = 120 := by
  sorry

end NUMINAMATH_CALUDE_store_coloring_books_l1497_149745


namespace NUMINAMATH_CALUDE_replaced_lettuce_cost_is_1_75_l1497_149711

/-- Represents the grocery order with its components -/
structure GroceryOrder where
  originalTotal : ℝ
  tomatoesOld : ℝ
  tomatoesNew : ℝ
  lettuceOld : ℝ
  celeryOld : ℝ
  celeryNew : ℝ
  deliveryAndTip : ℝ
  newTotal : ℝ

/-- The cost of the replaced lettuce given the grocery order details -/
def replacedLettuceCost (order : GroceryOrder) : ℝ :=
  order.lettuceOld + (order.newTotal - order.originalTotal - order.deliveryAndTip) -
  ((order.tomatoesNew - order.tomatoesOld) + (order.celeryNew - order.celeryOld))

/-- Theorem stating that the cost of the replaced lettuce is $1.75 -/
theorem replaced_lettuce_cost_is_1_75 (order : GroceryOrder)
  (h1 : order.originalTotal = 25)
  (h2 : order.tomatoesOld = 0.99)
  (h3 : order.tomatoesNew = 2.20)
  (h4 : order.lettuceOld = 1.00)
  (h5 : order.celeryOld = 1.96)
  (h6 : order.celeryNew = 2.00)
  (h7 : order.deliveryAndTip = 8.00)
  (h8 : order.newTotal = 35) :
  replacedLettuceCost order = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_replaced_lettuce_cost_is_1_75_l1497_149711


namespace NUMINAMATH_CALUDE_p_true_q_false_l1497_149795

theorem p_true_q_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : p ∧ ¬q :=
by
  sorry

end NUMINAMATH_CALUDE_p_true_q_false_l1497_149795


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1497_149785

theorem cubic_equation_roots :
  let a : ℝ := 5
  let b : ℝ := (5 + Real.sqrt 61) / 2
  let f (x : ℝ) := x^3 - 5*x^2 - 9*x + 45
  (f a = 0 ∧ f b = 0 ∧ f (-b) = 0) ∧
  (∃ (r₁ r₂ : ℝ), f r₁ = 0 ∧ f r₂ = 0 ∧ r₁ = -r₂) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1497_149785


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1497_149782

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c > d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1497_149782


namespace NUMINAMATH_CALUDE_fencing_required_l1497_149731

/-- Calculates the fencing required for a rectangular field with one side uncovered -/
theorem fencing_required (length width area : ℝ) (h1 : length = 34) (h2 : area = 680) 
  (h3 : area = length * width) : 2 * width + length = 74 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l1497_149731


namespace NUMINAMATH_CALUDE_lakers_win_probability_l1497_149796

/-- The probability of the Lakers winning a single game -/
def p_lakers : ℚ := 2/3

/-- The probability of the Celtics winning a single game -/
def p_celtics : ℚ := 1/3

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The minimum number of games in the series -/
def min_games : ℕ := 5

/-- The probability of the Lakers winning the NBA Finals given that the series lasts at least 5 games -/
theorem lakers_win_probability : 
  (Finset.sum (Finset.range 3) (λ i => 
    (Nat.choose (games_to_win + i) i) * 
    (p_lakers ^ games_to_win) * 
    (p_celtics ^ i))) = 1040/729 := by sorry

end NUMINAMATH_CALUDE_lakers_win_probability_l1497_149796


namespace NUMINAMATH_CALUDE_election_outcome_depends_on_radicals_l1497_149716

/-- Represents a political group in the election -/
inductive PoliticalGroup
| Socialist
| Republican
| Radical
| Other

/-- Represents the election models -/
inductive ElectionModel
| A
| B

/-- Represents the election system with four political groups -/
structure ElectionSystem where
  groups : Fin 4 → PoliticalGroup
  groupSize : ℕ
  socialistsPrefB : ℕ
  republicansPrefA : ℕ
  radicalSupport : PoliticalGroup

/-- The outcome of the election -/
def electionOutcome (system : ElectionSystem) : ElectionModel :=
  match system.radicalSupport with
  | PoliticalGroup.Socialist => ElectionModel.B
  | PoliticalGroup.Republican => ElectionModel.A
  | _ => sorry -- This case should not occur in our scenario

/-- Theorem stating that the election outcome depends on radicals' support -/
theorem election_outcome_depends_on_radicals (system : ElectionSystem) 
  (h1 : system.socialistsPrefB = system.republicansPrefA)
  (h2 : system.socialistsPrefB > 0) :
  (∃ (support : PoliticalGroup), 
    electionOutcome {system with radicalSupport := support} = ElectionModel.A) ∧
  (∃ (support : PoliticalGroup), 
    electionOutcome {system with radicalSupport := support} = ElectionModel.B) :=
  sorry


end NUMINAMATH_CALUDE_election_outcome_depends_on_radicals_l1497_149716


namespace NUMINAMATH_CALUDE_josh_string_cheese_cost_l1497_149753

/-- The total cost of Josh's string cheese purchase, including tax and discount -/
def total_cost (pack1 pack2 pack3 : ℕ) (price_per_cheese : ℚ) (discount_rate tax_rate : ℚ) : ℚ :=
  let total_cheese := pack1 + pack2 + pack3
  let subtotal := (total_cheese : ℚ) * price_per_cheese
  let discounted_price := subtotal * (1 - discount_rate)
  let tax := discounted_price * tax_rate
  discounted_price + tax

/-- Theorem stating the total cost of Josh's purchase -/
theorem josh_string_cheese_cost :
  total_cost 18 22 24 (10 / 100) (5 / 100) (12 / 100) = (681 / 100) := by
  sorry

end NUMINAMATH_CALUDE_josh_string_cheese_cost_l1497_149753


namespace NUMINAMATH_CALUDE_sqrt_three_expression_l1497_149761

theorem sqrt_three_expression : 
  (Real.sqrt 3 + 2)^2023 * (Real.sqrt 3 - 2)^2024 = -Real.sqrt 3 + 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_expression_l1497_149761


namespace NUMINAMATH_CALUDE_theater_seats_tom_wants_500_seats_l1497_149791

/-- Calculates the number of seats in Tom's theater based on given conditions --/
theorem theater_seats (cost_per_sqft : ℝ) (sqft_per_seat : ℝ) (partner_share : ℝ) (tom_spend : ℝ) : ℝ :=
  let cost_per_seat := cost_per_sqft * sqft_per_seat
  let total_cost := tom_spend / (1 - partner_share)
  total_cost / (3 * cost_per_seat)

/-- Proves that Tom wants 500 seats in his theater --/
theorem tom_wants_500_seats :
  theater_seats 5 12 0.4 54000 = 500 := by
  sorry

end NUMINAMATH_CALUDE_theater_seats_tom_wants_500_seats_l1497_149791


namespace NUMINAMATH_CALUDE_factorization_proof_l1497_149775

theorem factorization_proof (x : ℝ) : 72 * x^5 - 90 * x^9 = 18 * x^5 * (4 - 5 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1497_149775


namespace NUMINAMATH_CALUDE_article_price_calculation_l1497_149704

theorem article_price_calculation (p q : ℝ) : 
  let final_price := 1
  let price_after_increase (x : ℝ) := x * (1 + p / 100)
  let price_after_decrease (y : ℝ) := y * (1 - q / 100)
  let original_price := 10000 / (10000 + 100 * (p - q) - p * q)
  price_after_decrease (price_after_increase original_price) = final_price :=
by sorry

end NUMINAMATH_CALUDE_article_price_calculation_l1497_149704


namespace NUMINAMATH_CALUDE_polyhedron_space_diagonals_l1497_149799

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem: A convex polyhedron Q with 30 vertices, 72 edges, 44 faces 
    (of which 32 are triangular and 12 are quadrilateral) has 339 space diagonals -/
theorem polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 32,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 339 := by
  sorry


end NUMINAMATH_CALUDE_polyhedron_space_diagonals_l1497_149799


namespace NUMINAMATH_CALUDE_tom_new_books_l1497_149772

/-- Calculates the number of new books Tom bought given his initial, sold, and final book counts. -/
def new_books (initial : ℕ) (sold : ℕ) (final : ℕ) : ℕ :=
  final - (initial - sold)

/-- Proves that Tom bought 38 new books given the problem conditions. -/
theorem tom_new_books : new_books 5 4 39 = 38 := by
  sorry

end NUMINAMATH_CALUDE_tom_new_books_l1497_149772


namespace NUMINAMATH_CALUDE_intersection_points_count_l1497_149736

/-- The number of intersection points between y = |3x + 4| and y = -|4x + 3| -/
theorem intersection_points_count : ∃! p : ℝ × ℝ, 
  (p.2 = |3 * p.1 + 4|) ∧ (p.2 = -|4 * p.1 + 3|) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l1497_149736


namespace NUMINAMATH_CALUDE_tangent_line_and_positivity_l1497_149710

open Real

noncomputable def f (a x : ℝ) : ℝ := (x - a) * log x + (1/2) * x

theorem tangent_line_and_positivity (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    (∀ x : ℝ, (f a x₀ - f a x) = (1/2) * (x₀ - x)) → a = 1) ∧
  ((1/(2*exp 1)) < a ∧ a < 2 * sqrt (exp 1) → 
    ∀ x : ℝ, x > 0 → f a x > 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_positivity_l1497_149710


namespace NUMINAMATH_CALUDE_swimming_practice_months_l1497_149774

def total_required_hours : ℕ := 1500
def completed_hours : ℕ := 180
def monthly_practice_hours : ℕ := 220

theorem swimming_practice_months :
  (total_required_hours - completed_hours) / monthly_practice_hours = 6 :=
by sorry

end NUMINAMATH_CALUDE_swimming_practice_months_l1497_149774


namespace NUMINAMATH_CALUDE_decomposable_exponential_linear_cos_decomposable_l1497_149734

-- Define a decomposable function
def Decomposable (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

-- Theorem for part 2
theorem decomposable_exponential_linear (b : ℝ) :
  Decomposable (λ x => 2*x + b + 2^x) → b > -2 :=
by sorry

-- Theorem for part 3
theorem cos_decomposable :
  Decomposable cos :=
by sorry

end NUMINAMATH_CALUDE_decomposable_exponential_linear_cos_decomposable_l1497_149734


namespace NUMINAMATH_CALUDE_total_plums_eq_27_l1497_149727

/-- The number of plums Alyssa picked -/
def alyssas_plums : ℕ := 17

/-- The number of plums Jason picked -/
def jasons_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := alyssas_plums + jasons_plums

theorem total_plums_eq_27 : total_plums = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_plums_eq_27_l1497_149727


namespace NUMINAMATH_CALUDE_solution_difference_l1497_149793

/-- The quadratic equation from the problem -/
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 3*x + 9 = x + 41

/-- The two solutions of the quadratic equation -/
def solutions : Set ℝ :=
  {x : ℝ | quadratic_equation x}

/-- Theorem stating that the positive difference between the two solutions is 12 -/
theorem solution_difference : 
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 12 :=
sorry

end NUMINAMATH_CALUDE_solution_difference_l1497_149793


namespace NUMINAMATH_CALUDE_paige_finished_problems_l1497_149757

/-- Calculates the number of finished homework problems -/
def finished_problems (total : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  total - (remaining_pages * problems_per_page)

/-- Theorem: Paige finished 47 problems -/
theorem paige_finished_problems :
  finished_problems 110 7 9 = 47 := by
  sorry

end NUMINAMATH_CALUDE_paige_finished_problems_l1497_149757


namespace NUMINAMATH_CALUDE_pure_imaginary_equation_l1497_149743

theorem pure_imaginary_equation (z : ℂ) (b : ℝ) : 
  (∃ (a : ℝ), z = a * Complex.I) → 
  (2 - Complex.I) * z = 4 - b * Complex.I → 
  b = -8 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_equation_l1497_149743


namespace NUMINAMATH_CALUDE_sine_function_properties_l1497_149764

noncomputable def f (A ω φ x : ℝ) : ℝ := A * Real.sin (ω * x + φ)

theorem sine_function_properties :
  ∃ (A ω φ : ℝ),
    (f A ω φ 0 = 0) ∧
    (f A ω φ (π/2) = 2) ∧
    (f A ω φ π = 0) ∧
    (f A ω φ (3*π/2) = -2) ∧
    (f A ω φ (2*π) = 0) ∧
    (5*π/3 + π/3 = 2*π) →
    (A = 2) ∧
    (ω = 1/2) ∧
    (φ = 2*π/3) ∧
    (∀ x : ℝ, f A ω φ (x - π/3) = f A ω φ (-x - π/3)) :=
by sorry

end NUMINAMATH_CALUDE_sine_function_properties_l1497_149764


namespace NUMINAMATH_CALUDE_train_distance_difference_l1497_149765

/-- Proves the difference in distance traveled by two trains meeting each other --/
theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 16)
  (h2 : v2 = 21)
  (h3 : total_distance = 444)
  (h4 : v1 > 0)
  (h5 : v2 > 0) :
  let t := total_distance / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  d2 - d1 = 60 := by sorry

end NUMINAMATH_CALUDE_train_distance_difference_l1497_149765


namespace NUMINAMATH_CALUDE_fraction_simplification_l1497_149724

theorem fraction_simplification (x : ℝ) (hx : x ≠ 0) :
  (42 * x^3) / (63 * x^5) = 2 / (3 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1497_149724


namespace NUMINAMATH_CALUDE_house_sale_gain_l1497_149766

/-- Calculates the net gain from selling a house at a profit and buying it back at a loss -/
def netGainFromHouseSale (initialValue : ℝ) (profitPercent : ℝ) (lossPercent : ℝ) : ℝ :=
  let sellingPrice := initialValue * (1 + profitPercent)
  let buybackPrice := sellingPrice * (1 - lossPercent)
  sellingPrice - buybackPrice

/-- Theorem stating that selling a $200,000 house at 15% profit and buying it back at 5% loss results in $11,500 gain -/
theorem house_sale_gain :
  netGainFromHouseSale 200000 0.15 0.05 = 11500 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_gain_l1497_149766


namespace NUMINAMATH_CALUDE_min_sum_dimensions_of_box_l1497_149763

/-- Given a rectangular box with positive integer dimensions and volume 2310,
    the minimum possible sum of its three dimensions is 42. -/
theorem min_sum_dimensions_of_box (l w h : ℕ+) : 
  l * w * h = 2310 → l.val + w.val + h.val ≥ 42 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_of_box_l1497_149763


namespace NUMINAMATH_CALUDE_natural_number_pairs_l1497_149789

theorem natural_number_pairs (a b : ℕ) :
  (∃ k : ℕ, b - 1 = k * (a + 1)) →
  (∃ m : ℕ, a^2 + a + 2 = m * b) →
  ∃ k : ℕ, a = 2 * k ∧ b = 2 * k^2 + 2 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pairs_l1497_149789


namespace NUMINAMATH_CALUDE_impossible_sum_110_l1497_149705

def coin_values : List ℕ := [1, 5, 10, 25, 50]

theorem impossible_sum_110 : 
  ¬ ∃ (coins : List ℕ), 
    coins.length = 6 ∧ 
    (∀ c ∈ coins, c ∈ coin_values) ∧ 
    coins.sum = 110 :=
sorry

end NUMINAMATH_CALUDE_impossible_sum_110_l1497_149705


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_l1497_149769

/-- A regular six-pointed star -/
structure SixPointedStar :=
  (points : Fin 12)

/-- The number of symmetries of a regular six-pointed star -/
def star_symmetries : ℕ := 12

/-- The number of distinct arrangements of 12 different objects on a regular six-pointed star,
    where reflections and rotations are considered equivalent -/
def distinct_arrangements : ℕ := Nat.factorial 12 / star_symmetries

theorem distinct_arrangements_count :
  distinct_arrangements = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_l1497_149769


namespace NUMINAMATH_CALUDE_fruit_amount_proof_l1497_149790

/-- The cost of blueberries in dollars per carton -/
def blueberry_cost : ℚ := 5

/-- The weight of blueberries in ounces per carton -/
def blueberry_weight : ℚ := 6

/-- The cost of raspberries in dollars per carton -/
def raspberry_cost : ℚ := 3

/-- The weight of raspberries in ounces per carton -/
def raspberry_weight : ℚ := 8

/-- The number of batches of muffins -/
def num_batches : ℕ := 4

/-- The total savings in dollars by using raspberries instead of blueberries -/
def total_savings : ℚ := 22

/-- The amount of fruit in ounces required for each batch of muffins -/
def fruit_per_batch : ℚ := 12

theorem fruit_amount_proof :
  (total_savings / (num_batches : ℚ)) / 
  ((blueberry_cost / blueberry_weight) - (raspberry_cost / raspberry_weight)) = fruit_per_batch :=
sorry

end NUMINAMATH_CALUDE_fruit_amount_proof_l1497_149790


namespace NUMINAMATH_CALUDE_max_distinct_numbers_with_prime_triple_sums_l1497_149717

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if the sum of any three numbers in a list is prime -/
def allTripleSumsPrime (l : List ℕ) : Prop :=
  ∀ a b c : ℕ, a ∈ l → b ∈ l → c ∈ l → a ≠ b → b ≠ c → a ≠ c → isPrime (a + b + c)

/-- The theorem stating that the maximum number of distinct natural numbers
    that can be chosen such that the sum of any three of them is prime is 4 -/
theorem max_distinct_numbers_with_prime_triple_sums :
  (∃ l : List ℕ, l.length = 4 ∧ l.Nodup ∧ allTripleSumsPrime l) ∧
  (∀ l : List ℕ, l.length > 4 → ¬(l.Nodup ∧ allTripleSumsPrime l)) :=
sorry

end NUMINAMATH_CALUDE_max_distinct_numbers_with_prime_triple_sums_l1497_149717


namespace NUMINAMATH_CALUDE_georges_walk_l1497_149762

/-- Given that George walks 1 mile to school at 3 mph normally, prove that
    if he walks the first 1/2 mile at 2 mph, he must run the last 1/2 mile
    at 6 mph to arrive at the same time. -/
theorem georges_walk (normal_distance : Real) (normal_speed : Real) 
  (first_half_distance : Real) (first_half_speed : Real) 
  (second_half_distance : Real) (second_half_speed : Real) :
  normal_distance = 1 ∧ 
  normal_speed = 3 ∧ 
  first_half_distance = 1/2 ∧ 
  first_half_speed = 2 ∧ 
  second_half_distance = 1/2 ∧
  normal_distance / normal_speed = 
    first_half_distance / first_half_speed + second_half_distance / second_half_speed →
  second_half_speed = 6 := by
  sorry

#check georges_walk

end NUMINAMATH_CALUDE_georges_walk_l1497_149762


namespace NUMINAMATH_CALUDE_total_spent_equals_621_l1497_149712

/-- The total amount spent by Tate and Peyton on their remaining tickets -/
def total_spent (tate_initial_tickets : ℕ) (tate_initial_price : ℕ) 
  (tate_additional_tickets : ℕ) (tate_additional_price : ℕ)
  (peyton_price : ℕ) : ℕ :=
  let tate_total := tate_initial_tickets * tate_initial_price + 
                    tate_additional_tickets * tate_additional_price
  let peyton_initial_tickets := tate_initial_tickets / 2
  let peyton_remaining_tickets := peyton_initial_tickets - 
                                  (peyton_initial_tickets / 3)
  let peyton_total := peyton_remaining_tickets * peyton_price
  tate_total + peyton_total

/-- Theorem stating the total amount spent by Tate and Peyton -/
theorem total_spent_equals_621 : 
  total_spent 32 14 2 15 13 = 621 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_equals_621_l1497_149712


namespace NUMINAMATH_CALUDE_smallest_square_cover_l1497_149759

/-- The side length of the smallest square that can be covered by 2-by-4 rectangles -/
def smallest_square_side : ℕ := 8

/-- The area of a 2-by-4 rectangle -/
def rectangle_area : ℕ := 2 * 4

/-- The number of 2-by-4 rectangles needed to cover the smallest square -/
def num_rectangles : ℕ := smallest_square_side^2 / rectangle_area

theorem smallest_square_cover :
  (∀ n : ℕ, n < smallest_square_side → n^2 % rectangle_area ≠ 0) ∧
  smallest_square_side^2 % rectangle_area = 0 ∧
  num_rectangles = 8 := by sorry

end NUMINAMATH_CALUDE_smallest_square_cover_l1497_149759


namespace NUMINAMATH_CALUDE_program_output_correct_l1497_149797

def program_execution (initial_A initial_B : ℤ) : ℤ × ℤ × ℤ :=
  let A₁ := if initial_A < 0 then -initial_A else initial_A
  let B₁ := initial_B ^ 2
  let A₂ := A₁ + B₁
  let C  := A₂ - 2 * B₁
  let A₃ := A₂ / C
  let B₂ := B₁ * C + 1
  (A₃, B₂, C)

theorem program_output_correct :
  program_execution (-6) 2 = (5, 9, 2) := by
  sorry

end NUMINAMATH_CALUDE_program_output_correct_l1497_149797


namespace NUMINAMATH_CALUDE_count_divisors_eq_twelve_l1497_149738

/-- The number of natural numbers m such that 2023 ≡ 23 (mod m) -/
def count_divisors : ℕ :=
  (Finset.filter (fun m => m > 23 ∧ 2023 % m = 23) (Finset.range 2024)).card

theorem count_divisors_eq_twelve : count_divisors = 12 := by
  sorry

end NUMINAMATH_CALUDE_count_divisors_eq_twelve_l1497_149738


namespace NUMINAMATH_CALUDE_last_two_digits_33_divisible_by_prime_gt_7_l1497_149715

theorem last_two_digits_33_divisible_by_prime_gt_7 (n : ℕ) :
  (∃ k : ℕ, n = 100 * k + 33) →
  ∃ p : ℕ, p > 7 ∧ Prime p ∧ p ∣ n :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_33_divisible_by_prime_gt_7_l1497_149715


namespace NUMINAMATH_CALUDE_hyperbola_b_value_l1497_149742

-- Define the hyperbola equation
def hyperbola (x y b : ℝ) : Prop := x^2 / 4 - y^2 / b^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := (Real.sqrt 3 * x + y = 0) ∧ (Real.sqrt 3 * x - y = 0)

-- Theorem statement
theorem hyperbola_b_value (b : ℝ) 
  (h1 : b > 0) 
  (h2 : ∀ x y, hyperbola x y b → asymptotes x y) : 
  b = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_b_value_l1497_149742


namespace NUMINAMATH_CALUDE_oil_bill_problem_l1497_149748

/-- The oil bill problem -/
theorem oil_bill_problem (january_bill february_bill : ℚ) 
  (h1 : february_bill / january_bill = 3 / 2)
  (h2 : (february_bill + 30) / january_bill = 5 / 3) :
  january_bill = 180 := by
  sorry

end NUMINAMATH_CALUDE_oil_bill_problem_l1497_149748


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1497_149767

theorem quadratic_factorization (b : ℤ) : 
  (∃ (c d e f : ℤ), 15 * x^2 + b * x + 45 = (c * x + d) * (e * x + f)) → 
  ∃ (k : ℤ), b = 2 * k :=
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1497_149767


namespace NUMINAMATH_CALUDE_anthony_transaction_percentage_l1497_149758

theorem anthony_transaction_percentage (mabel_transactions cal_transactions anthony_transactions jade_transactions : ℕ) :
  mabel_transactions = 90 →
  cal_transactions = (2 : ℚ) / 3 * anthony_transactions →
  jade_transactions = cal_transactions + 16 →
  jade_transactions = 82 →
  (anthony_transactions : ℚ) / mabel_transactions - 1 = (1 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_anthony_transaction_percentage_l1497_149758


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1497_149733

theorem sum_of_squares_of_roots (a b : ℝ) 
  (ha : a^2 - 6*a + 4 = 0) 
  (hb : b^2 - 6*b + 4 = 0) 
  (hab : a ≠ b) : 
  a^2 + b^2 = 28 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1497_149733


namespace NUMINAMATH_CALUDE_pen_sales_revenue_pen_sales_revenue_proof_l1497_149722

theorem pen_sales_revenue : ℝ → Prop :=
  fun total_revenue =>
    ∀ (total_pens : ℕ) (displayed_pens : ℕ) (storeroom_pens : ℕ),
      (displayed_pens : ℝ) = 0.3 * total_pens ∧
      (storeroom_pens : ℝ) = 0.7 * total_pens ∧
      storeroom_pens = 210 ∧
      total_revenue = (displayed_pens : ℝ) * 2 →
      total_revenue = 180

-- The proof is omitted
theorem pen_sales_revenue_proof : pen_sales_revenue 180 := by
  sorry

end NUMINAMATH_CALUDE_pen_sales_revenue_pen_sales_revenue_proof_l1497_149722


namespace NUMINAMATH_CALUDE_g_of_3_equals_79_l1497_149768

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 + 3 * x - 2

-- State the theorem
theorem g_of_3_equals_79 : g 3 = 79 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_79_l1497_149768


namespace NUMINAMATH_CALUDE_equal_charge_at_60_minutes_l1497_149751

/-- United Telephone's base rate in dollars -/
def united_base : ℝ := 9

/-- United Telephone's per-minute charge in dollars -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute charge in dollars -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which both companies charge the same amount -/
def equal_charge_minutes : ℝ := 60

theorem equal_charge_at_60_minutes :
  united_base + united_per_minute * equal_charge_minutes =
  atlantic_base + atlantic_per_minute * equal_charge_minutes :=
by sorry

end NUMINAMATH_CALUDE_equal_charge_at_60_minutes_l1497_149751


namespace NUMINAMATH_CALUDE_four_digit_number_property_l1497_149787

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_valid_n (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 10 % 10 ≠ 0)

def split_n (n : ℕ) : ℕ × ℕ :=
  (n / 100, n % 100)

theorem four_digit_number_property (n : ℕ) 
  (h1 : is_valid_n n) 
  (h2 : let (A, B) := split_n n; is_two_digit A ∧ is_two_digit B)
  (h3 : let (A, B) := split_n n; n % (A * B) = 0) :
  n = 1734 ∨ n = 1352 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_property_l1497_149787


namespace NUMINAMATH_CALUDE_cades_remaining_marbles_l1497_149792

/-- Represents the number of marbles Cade has left after giving some away. -/
def marblesLeft (initial : ℕ) (givenAway : ℕ) : ℕ :=
  initial - givenAway

/-- Theorem stating that Cade's remaining marbles is the difference between his initial marbles and those given away. -/
theorem cades_remaining_marbles (initial : ℕ) (givenAway : ℕ) 
  (h : givenAway ≤ initial) : 
  marblesLeft initial givenAway = initial - givenAway :=
by
  sorry

#eval marblesLeft 87 8  -- Should output 79

end NUMINAMATH_CALUDE_cades_remaining_marbles_l1497_149792


namespace NUMINAMATH_CALUDE_romance_movie_tickets_l1497_149726

theorem romance_movie_tickets (horror_tickets romance_tickets : ℕ) : 
  horror_tickets = 3 * romance_tickets + 18 →
  horror_tickets = 93 →
  romance_tickets = 25 := by
sorry

end NUMINAMATH_CALUDE_romance_movie_tickets_l1497_149726


namespace NUMINAMATH_CALUDE_smallest_c_for_positive_quadratic_l1497_149752

theorem smallest_c_for_positive_quadratic : 
  ∃ c : ℤ, (∀ x : ℝ, x^2 + c*x + 15 > 0) ∧ 
  (∀ d : ℤ, d < c → ∃ x : ℝ, x^2 + d*x + 15 ≤ 0) ∧ 
  c = -7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_for_positive_quadratic_l1497_149752


namespace NUMINAMATH_CALUDE_age_interchange_problem_l1497_149703

theorem age_interchange_problem :
  let valid_pair := λ (t n : ℕ) =>
    t > 30 ∧
    n > 0 ∧
    30 + n < 100 ∧
    t + n < 100 ∧
    (t + n) / 10 = (30 + n) % 10 ∧
    (t + n) % 10 = (30 + n) / 10
  (∃! l : List (ℕ × ℕ), l.length = 21 ∧ ∀ p ∈ l, valid_pair p.1 p.2) :=
by sorry

end NUMINAMATH_CALUDE_age_interchange_problem_l1497_149703


namespace NUMINAMATH_CALUDE_max_value_interval_m_range_l1497_149739

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem max_value_interval_m_range 
  (m : ℝ) 
  (h1 : ∃ (x : ℝ), m < x ∧ x < 8 - m^2 ∧ ∀ (y : ℝ), m < y ∧ y < 8 - m^2 → f y ≤ f x) :
  m ∈ Set.Ioc (-3) (-Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_max_value_interval_m_range_l1497_149739


namespace NUMINAMATH_CALUDE_unique_modular_solution_l1497_149713

theorem unique_modular_solution : ∃! n : ℤ, n ≡ -5678 [ZMOD 10] ∧ 0 ≤ n ∧ n ≤ 9 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l1497_149713


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1497_149786

theorem perfect_square_condition (n : ℕ) : 
  (∃ m : ℕ, n^2 + 3*n = m^2) ↔ n = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1497_149786


namespace NUMINAMATH_CALUDE_blocks_added_l1497_149756

def initial_blocks : ℕ := 35
def final_blocks : ℕ := 65

theorem blocks_added : final_blocks - initial_blocks = 30 := by
  sorry

end NUMINAMATH_CALUDE_blocks_added_l1497_149756


namespace NUMINAMATH_CALUDE_part_one_part_two_l1497_149720

/-- Given a point M(2m+1, m-4) and N(5, 2) in the Cartesian coordinate system -/
def M (m : ℝ) : ℝ × ℝ := (2*m + 1, m - 4)
def N : ℝ × ℝ := (5, 2)

/-- Part 1: If MN is parallel to the x-axis, then M(13, 2) -/
theorem part_one (m : ℝ) : 
  (M m).2 = N.2 → M m = (13, 2) := by sorry

/-- Part 2: If M is 3 units to the right of the y-axis, then M(3, -3) -/
theorem part_two (m : ℝ) :
  (M m).1 = 3 → M m = (3, -3) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1497_149720


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l1497_149709

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {2, 3, 4}

-- Define set N
def N : Finset Nat := {4, 5}

-- Theorem statement
theorem complement_of_union_M_N :
  (U \ (M ∪ N)) = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l1497_149709


namespace NUMINAMATH_CALUDE_tangent_line_and_perpendicular_points_l1497_149750

-- Define the function f(x) = x³ - x - 1
def f (x : ℝ) : ℝ := x^3 - x - 1

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_and_perpendicular_points (x y : ℝ) :
  -- The tangent line at (1, -1) has equation 2x - y - 3 = 0
  (x = 1 ∧ y = -1 → 2 * x - y - 3 = 0) ∧
  -- The points of tangency where the tangent line is perpendicular to y = -1/2x + 3
  -- are (1, -1) and (-1, -1)
  (f' x = 2 → (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_perpendicular_points_l1497_149750


namespace NUMINAMATH_CALUDE_hike_distance_l1497_149760

/-- The distance between two points given specific movement conditions -/
theorem hike_distance (A B C : ℝ × ℝ) : 
  B.1 - A.1 = 5 ∧ 
  B.2 - A.2 = 0 ∧ 
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 64 ∧ 
  C.1 - B.1 = C.2 - B.2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 89 + 40 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hike_distance_l1497_149760


namespace NUMINAMATH_CALUDE_same_side_inequality_l1497_149744

/-- Given that point P (a, b) and point Q (1, 2) are on the same side of the line 3x + 2y - 8 = 0,
    prove that 3a + 2b - 8 > 0 -/
theorem same_side_inequality (a b : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (3*a + 2*b - 8) * (3*1 + 2*2 - 8) = k * (3*a + 2*b - 8)^2) →
  3*a + 2*b - 8 > 0 := by
sorry

end NUMINAMATH_CALUDE_same_side_inequality_l1497_149744


namespace NUMINAMATH_CALUDE_prize_buying_l1497_149700

/-- Given the conditions for prize buying, prove the number of pens and notebooks. -/
theorem prize_buying (x y : ℝ) (h1 : 60 * (x + 2*y) = 50 * (x + 3*y)) 
  (total_budget : ℝ) (h2 : total_budget = 60 * (x + 2*y)) : 
  (total_budget / x = 100) ∧ (total_budget / y = 300) := by
  sorry

end NUMINAMATH_CALUDE_prize_buying_l1497_149700


namespace NUMINAMATH_CALUDE_isosceles_triangle_m_values_l1497_149741

/-- An isosceles triangle with side lengths satisfying a quadratic equation -/
structure IsoscelesTriangle where
  -- The length of side BC
  bc : ℝ
  -- The parameter m in the quadratic equation
  m : ℝ
  -- The roots of the quadratic equation x^2 - 10x + m = 0 represent the lengths of AB and AC
  root1 : ℝ
  root2 : ℝ
  -- Ensure that root1 and root2 are indeed roots of the equation
  eq1 : root1^2 - 10*root1 + m = 0
  eq2 : root2^2 - 10*root2 + m = 0
  -- Ensure that the triangle is isosceles (two sides are equal)
  isosceles : root1 = root2 ∨ (root1 = bc ∧ root2 = 10 - bc) ∨ (root2 = bc ∧ root1 = 10 - bc)
  -- Given condition that BC = 8
  bc_eq_8 : bc = 8

/-- The theorem stating that m is either 16 or 25 -/
theorem isosceles_triangle_m_values (t : IsoscelesTriangle) : t.m = 16 ∨ t.m = 25 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_m_values_l1497_149741


namespace NUMINAMATH_CALUDE_jebbs_take_home_pay_l1497_149788

/-- Calculates the take-home pay after tax -/
def takeHomePay (originalPay : ℝ) (taxRate : ℝ) : ℝ :=
  originalPay * (1 - taxRate)

/-- Proves that Jebb's take-home pay is $585 -/
theorem jebbs_take_home_pay :
  takeHomePay 650 0.1 = 585 := by
  sorry

end NUMINAMATH_CALUDE_jebbs_take_home_pay_l1497_149788


namespace NUMINAMATH_CALUDE_custom_mul_seven_neg_two_l1497_149749

/-- Custom multiplication operation for rational numbers -/
def custom_mul (a b : ℚ) : ℚ := b^2 - a

/-- Theorem stating that 7 * (-2) = -3 under the custom multiplication -/
theorem custom_mul_seven_neg_two : custom_mul 7 (-2) = -3 := by sorry

end NUMINAMATH_CALUDE_custom_mul_seven_neg_two_l1497_149749
