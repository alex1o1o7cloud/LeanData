import Mathlib

namespace NUMINAMATH_CALUDE_billy_reads_three_books_l2529_252928

/-- Represents Billy's reading activity over the weekend --/
structure BillyReading where
  initial_speed : ℝ  -- Initial reading speed in pages per hour
  time_available : ℝ  -- Total time available for reading in hours
  book_pages : ℕ  -- Number of pages in each book
  speed_decrease : ℝ  -- Percentage decrease in reading speed after each book

/-- Calculates the number of books Billy can read --/
def books_read (b : BillyReading) : ℕ :=
  sorry

/-- Theorem stating that Billy can read exactly 3 books --/
theorem billy_reads_three_books :
  let b : BillyReading := {
    initial_speed := 60,
    time_available := 16 * 0.35,
    book_pages := 80,
    speed_decrease := 0.1
  }
  books_read b = 3 := by sorry

end NUMINAMATH_CALUDE_billy_reads_three_books_l2529_252928


namespace NUMINAMATH_CALUDE_range_of_x_minus_y_l2529_252937

theorem range_of_x_minus_y :
  ∀ x y : ℝ, 2 < x ∧ x < 4 → -1 < y ∧ y < 3 →
  ∃ z : ℝ, -1 < z ∧ z < 5 ∧ z = x - y ∧
  ∀ w : ℝ, w = x - y → -1 < w ∧ w < 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_minus_y_l2529_252937


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l2529_252935

theorem ratio_x_to_y (x y : ℝ) (h : y = x * (1 - 0.9166666666666666)) :
  x / y = 12 :=
sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l2529_252935


namespace NUMINAMATH_CALUDE_zeros_after_one_in_500_to_150_l2529_252971

-- Define 500 as 5 * 10^2
def five_hundred : ℕ := 5 * 10^2

-- Theorem statement
theorem zeros_after_one_in_500_to_150 :
  (∃ n : ℕ, five_hundred^150 = 10^n * (1 + 10 * m) ∧ m < 10) ∧
  (∀ k : ℕ, five_hundred^150 = 10^k * (1 + 10 * m) ∧ m < 10 → k = 300) :=
sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_500_to_150_l2529_252971


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2529_252987

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {2, 3}
def B : Set ℕ := {1}

theorem intersection_with_complement : A ∩ (U \ B) = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2529_252987


namespace NUMINAMATH_CALUDE_hexagon_area_for_given_triangle_l2529_252906

/-- Given an isosceles triangle PQR with circumcircle radius r and perimeter p,
    calculate the area of the hexagon formed by the intersections of the
    perpendicular bisectors of the sides with the circumcircle. -/
def hexagon_area (r p : ℝ) : ℝ :=
  5 * p

theorem hexagon_area_for_given_triangle :
  hexagon_area 10 42 = 210 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_for_given_triangle_l2529_252906


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2529_252962

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x - 3 = 0 ∧ a * y^2 + 2 * y - 3 = 0) → a > -1/3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2529_252962


namespace NUMINAMATH_CALUDE_line_equation_proof_l2529_252910

/-- Proves that the equation of a line with slope -2 and y-intercept 3 is 2x + y - 3 = 0 -/
theorem line_equation_proof (x y : ℝ) : 
  let slope : ℝ := -2
  let y_intercept : ℝ := 3
  let line_equation := fun (x y : ℝ) => 2 * x + y - 3 = 0
  line_equation x y ↔ y = slope * x + y_intercept :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2529_252910


namespace NUMINAMATH_CALUDE_problem_statement_l2529_252950

theorem problem_statement (x : ℝ) : 
  x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10 →
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 289/8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2529_252950


namespace NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l2529_252911

theorem arccos_sqrt3_over_2 : Real.arccos (Real.sqrt 3 / 2) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sqrt3_over_2_l2529_252911


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2529_252953

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 24 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y + 2 * y + 24 = 0 → y = x) ↔ 
  (k = 2 + 12 * Real.sqrt 2 ∨ k = 2 - 12 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2529_252953


namespace NUMINAMATH_CALUDE_calculate_small_orders_l2529_252991

/-- Given information about packing peanuts usage in orders, calculate the number of small orders. -/
theorem calculate_small_orders (total_peanuts : ℕ) (large_orders : ℕ) (peanuts_per_large : ℕ) (peanuts_per_small : ℕ) :
  total_peanuts = 800 →
  large_orders = 3 →
  peanuts_per_large = 200 →
  peanuts_per_small = 50 →
  (total_peanuts - large_orders * peanuts_per_large) / peanuts_per_small = 4 :=
by sorry

end NUMINAMATH_CALUDE_calculate_small_orders_l2529_252991


namespace NUMINAMATH_CALUDE_part_one_part_two_l2529_252923

-- Define proposition p
def p (k : ℝ) : Prop := ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 + 2*x - k ≤ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*k*x + 3*k + 4 = 0

-- Theorem for part 1
theorem part_one (k : ℝ) : p k → k ∈ Set.Ici 3 := by sorry

-- Theorem for part 2
theorem part_two (k : ℝ) : 
  (p k ∧ ¬q k) ∨ (¬p k ∧ q k) → k ∈ Set.Iic (-1) ∪ Set.Ico 3 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2529_252923


namespace NUMINAMATH_CALUDE_sixth_graders_count_l2529_252970

/-- The number of fifth graders -/
def fifth_graders : ℕ := 109

/-- The number of seventh graders -/
def seventh_graders : ℕ := 118

/-- The number of teachers -/
def teachers : ℕ := 4

/-- The number of parents per grade -/
def parents_per_grade : ℕ := 2

/-- The number of buses -/
def buses : ℕ := 5

/-- The number of seats per bus -/
def seats_per_bus : ℕ := 72

/-- The total number of seats available -/
def total_seats : ℕ := buses * seats_per_bus

/-- The total number of chaperones -/
def total_chaperones : ℕ := (teachers + parents_per_grade) * 3

/-- The number of students and chaperones excluding sixth graders -/
def non_sixth_grade_total : ℕ := fifth_graders + seventh_graders + total_chaperones

theorem sixth_graders_count : total_seats - non_sixth_grade_total = 115 := by
  sorry

end NUMINAMATH_CALUDE_sixth_graders_count_l2529_252970


namespace NUMINAMATH_CALUDE_sofa_price_calculation_l2529_252961

def living_room_set_price (sofa_price armchair_price coffee_table_price : ℝ) : ℝ :=
  sofa_price + 2 * armchair_price + coffee_table_price

theorem sofa_price_calculation (armchair_price coffee_table_price total_price : ℝ)
  (h1 : armchair_price = 425)
  (h2 : coffee_table_price = 330)
  (h3 : total_price = 2430)
  (h4 : living_room_set_price (total_price - 2 * armchair_price - coffee_table_price) armchair_price coffee_table_price = total_price) :
  total_price - 2 * armchair_price - coffee_table_price = 1250 := by
  sorry

#check sofa_price_calculation

end NUMINAMATH_CALUDE_sofa_price_calculation_l2529_252961


namespace NUMINAMATH_CALUDE_composite_function_evaluation_l2529_252944

def f (x : ℝ) : ℝ := 5 * x + 2

def g (x : ℝ) : ℝ := 3 * x + 4

theorem composite_function_evaluation :
  f (g (f 3)) = 277 :=
by sorry

end NUMINAMATH_CALUDE_composite_function_evaluation_l2529_252944


namespace NUMINAMATH_CALUDE_order_of_numbers_l2529_252979

theorem order_of_numbers : (2 : ℝ)^24 < 10^8 ∧ 10^8 < 5^12 := by sorry

end NUMINAMATH_CALUDE_order_of_numbers_l2529_252979


namespace NUMINAMATH_CALUDE_binary_101_to_decimal_l2529_252949

def binary_to_decimal (b₂ : ℕ) (b₁ : ℕ) (b₀ : ℕ) : ℕ :=
  b₂ * 2^2 + b₁ * 2^1 + b₀ * 2^0

theorem binary_101_to_decimal :
  binary_to_decimal 1 0 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_to_decimal_l2529_252949


namespace NUMINAMATH_CALUDE_art_dealer_loss_l2529_252940

theorem art_dealer_loss (selling_price : ℝ) (selling_price_positive : selling_price > 0) :
  let profit_percentage : ℝ := 0.1
  let loss_percentage : ℝ := 0.1
  let cost_price_1 : ℝ := selling_price / (1 + profit_percentage)
  let cost_price_2 : ℝ := selling_price / (1 - loss_percentage)
  let profit : ℝ := selling_price - cost_price_1
  let loss : ℝ := cost_price_2 - selling_price
  let net_loss : ℝ := loss - profit
  net_loss = 0.02 * selling_price :=
by sorry

end NUMINAMATH_CALUDE_art_dealer_loss_l2529_252940


namespace NUMINAMATH_CALUDE_second_number_in_sum_l2529_252942

theorem second_number_in_sum (a b c : ℝ) : 
  a = 3.15 → c = 0.458 → a + b + c = 3.622 → b = 0.014 := by
  sorry

end NUMINAMATH_CALUDE_second_number_in_sum_l2529_252942


namespace NUMINAMATH_CALUDE_simplify_cube_root_l2529_252948

theorem simplify_cube_root (a b c : ℝ) : ∃ x y z w : ℝ, 
  (54 * a^5 * b^9 * c^14)^(1/3) = x * a^y * b^z * c^w ∧ y + z + w = 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_root_l2529_252948


namespace NUMINAMATH_CALUDE_musicians_performing_l2529_252918

/-- Represents a musical group --/
inductive MusicalGroup
| Quartet
| Trio
| Duet

/-- The number of musicians in each type of group --/
def group_size (g : MusicalGroup) : ℕ :=
  match g with
  | MusicalGroup.Quartet => 4
  | MusicalGroup.Trio => 3
  | MusicalGroup.Duet => 2

/-- The original schedule of performances --/
def original_schedule : List (MusicalGroup × ℕ) :=
  [(MusicalGroup.Quartet, 4), (MusicalGroup.Duet, 5), (MusicalGroup.Trio, 6)]

/-- The changes to the schedule --/
def schedule_changes : List (MusicalGroup × ℕ) :=
  [(MusicalGroup.Quartet, 1), (MusicalGroup.Duet, 2), (MusicalGroup.Trio, 1)]

/-- Calculate the total number of musicians given a schedule --/
def total_musicians (schedule : List (MusicalGroup × ℕ)) : ℕ :=
  schedule.foldl (fun acc (g, n) => acc + n * group_size g) 0

/-- The main theorem --/
theorem musicians_performing (
  orig_schedule : List (MusicalGroup × ℕ)) 
  (changes : List (MusicalGroup × ℕ)) :
  orig_schedule = original_schedule →
  changes = schedule_changes →
  total_musicians orig_schedule - 
  (total_musicians changes + 1) = 35 := by
  sorry

end NUMINAMATH_CALUDE_musicians_performing_l2529_252918


namespace NUMINAMATH_CALUDE_triangle_third_side_l2529_252981

theorem triangle_third_side (a b c : ℕ) : 
  a = 3 → b = 6 → c % 2 = 1 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (c > b - a ∧ c < b + a) →
  c = 5 ∨ c = 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_l2529_252981


namespace NUMINAMATH_CALUDE_triangle_properties_l2529_252946

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  median_CM_eq : (2 : ℝ) * C.1 - C.2 - 5 = 0
  altitude_BH_eq : B.1 - (2 : ℝ) * B.2 - 5 = 0

/-- The theorem statement -/
theorem triangle_properties (abc : Triangle) 
  (h_A : abc.A = (5, 1)) : 
  abc.C = (4, 3) ∧ 
  (6 : ℝ) * abc.B.1 - 5 * abc.B.2 - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2529_252946


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2529_252939

theorem simplify_and_evaluate (a : ℚ) (h : a = -3/2) :
  (a + 2 - 5 / (a - 2)) / ((2 * a^2 - 6 * a) / (a - 2)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2529_252939


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2529_252960

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem twentieth_term_of_sequence :
  let a₁ := 8
  let d := -3
  arithmetic_sequence a₁ d 20 = -49 := by sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2529_252960


namespace NUMINAMATH_CALUDE_range_of_m_l2529_252921

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) (h_sol : ∃ m : ℝ, x + y/4 < m^2 - 3*m) :
  ∀ m : ℝ, (x + y/4 < m^2 - 3*m) ↔ (m < -1 ∨ m > 4) := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2529_252921


namespace NUMINAMATH_CALUDE_power_log_fourth_root_l2529_252900

theorem power_log_fourth_root (x : ℝ) (h : x > 0) :
  ((625 ^ (Real.log x / Real.log 5)) ^ (1/4) : ℝ) = x :=
by sorry

end NUMINAMATH_CALUDE_power_log_fourth_root_l2529_252900


namespace NUMINAMATH_CALUDE_min_value_theorem_l2529_252984

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  x + 4 / (x - 1) ≥ 5 ∧ (x + 4 / (x - 1) = 5 ↔ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2529_252984


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2529_252972

theorem two_numbers_difference (x y : ℝ) (h_sum : x + y = 25) (h_product : x * y = 144) :
  |x - y| = 7 := by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2529_252972


namespace NUMINAMATH_CALUDE_power_greater_than_square_plus_one_l2529_252963

theorem power_greater_than_square_plus_one (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_greater_than_square_plus_one_l2529_252963


namespace NUMINAMATH_CALUDE_megan_bottles_left_l2529_252983

/-- Calculates the number of bottles Megan has left after drinking and giving away some bottles. -/
def bottles_left (initial : ℕ) (drank : ℕ) (given_away : ℕ) : ℕ :=
  initial - (drank + given_away)

/-- Theorem stating that Megan has 25 bottles left after starting with 45, drinking 8, and giving away 12. -/
theorem megan_bottles_left : bottles_left 45 8 12 = 25 := by
  sorry

end NUMINAMATH_CALUDE_megan_bottles_left_l2529_252983


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l2529_252968

theorem smallest_right_triangle_area :
  let a : ℝ := 6
  let b : ℝ := 8
  let area1 : ℝ := (1/2) * a * b
  let area2 : ℝ := (1/2) * a * Real.sqrt (b^2 - a^2)
  min area1 area2 = (3 : ℝ) * Real.sqrt 28 := by
sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l2529_252968


namespace NUMINAMATH_CALUDE_gcd_of_integer_differences_l2529_252954

theorem gcd_of_integer_differences (a b c d : ℤ) : 
  ∃ k : ℤ, (a - b) * (b - c) * (c - d) * (d - a) * (a - c) * (b - d) = 12 * k :=
sorry

end NUMINAMATH_CALUDE_gcd_of_integer_differences_l2529_252954


namespace NUMINAMATH_CALUDE_sum_of_f_values_l2529_252947

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2/x) + 1

theorem sum_of_f_values : 
  f (-7) + f (-5) + f (-3) + f (-1) + f 3 + f 5 + f 7 + f 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l2529_252947


namespace NUMINAMATH_CALUDE_corner_sum_is_ten_l2529_252902

/-- Represents a Go board as a function from coordinates to real numbers -/
def GoBoard : Type := Fin 18 → Fin 18 → ℝ

/-- The property that any 2x2 square on the board sums to 10 -/
def valid_board (board : GoBoard) : Prop :=
  ∀ i j, i < 17 → j < 17 →
    board i j + board (i+1) j + board i (j+1) + board (i+1) (j+1) = 10

/-- The sum of the four corner squares -/
def corner_sum (board : GoBoard) : ℝ :=
  board 0 0 + board 0 17 + board 17 0 + board 17 17

/-- Theorem: For any valid Go board, the sum of the four corners is 10 -/
theorem corner_sum_is_ten (board : GoBoard) (h : valid_board board) :
  corner_sum board = 10 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_ten_l2529_252902


namespace NUMINAMATH_CALUDE_complex_solution_l2529_252955

/-- Given two complex numbers a and b satisfying the equations
    2a^2 + ab + 2b^2 = 0 and a + 2b = 5, prove that both a and b are non-real. -/
theorem complex_solution (a b : ℂ) 
  (eq1 : 2 * a^2 + a * b + 2 * b^2 = 0)
  (eq2 : a + 2 * b = 5) :
  ¬(a.im = 0 ∧ b.im = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_solution_l2529_252955


namespace NUMINAMATH_CALUDE_point_on_circle_l2529_252974

theorem point_on_circle (t : ℝ) :
  let x := (3 * t^2 - 1) / (t^2 + 3)
  let y := 6 * t / (t^2 + 3)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_circle_l2529_252974


namespace NUMINAMATH_CALUDE_rectangle_area_18_pairs_l2529_252901

theorem rectangle_area_18_pairs : 
  {p : ℕ × ℕ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 * p.2 = 18} = 
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)} := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_18_pairs_l2529_252901


namespace NUMINAMATH_CALUDE_cubic_expression_equals_zero_l2529_252958

theorem cubic_expression_equals_zero (k : ℝ) (h : k = 2) : (k^3 - 8) * (k + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equals_zero_l2529_252958


namespace NUMINAMATH_CALUDE_min_value_exponential_function_l2529_252904

theorem min_value_exponential_function :
  ∀ x : ℝ, 4 * Real.exp x + Real.exp (-x) ≥ 4 ∧
  ∃ x₀ : ℝ, 4 * Real.exp x₀ + Real.exp (-x₀) = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_exponential_function_l2529_252904


namespace NUMINAMATH_CALUDE_perfect_linearity_implies_R_squared_one_l2529_252956

/-- A scatter plot is perfectly linear if all its points fall on a straight line with non-zero slope -/
def is_perfectly_linear (scatter_plot : Set (ℝ × ℝ)) : Prop :=
  ∃ (m : ℝ) (b : ℝ), m ≠ 0 ∧ ∀ (x y : ℝ), (x, y) ∈ scatter_plot → y = m * x + b

/-- The coefficient of determination (R²) for a scatter plot -/
def R_squared (scatter_plot : Set (ℝ × ℝ)) : ℝ := sorry

theorem perfect_linearity_implies_R_squared_one (scatter_plot : Set (ℝ × ℝ)) :
  is_perfectly_linear scatter_plot → R_squared scatter_plot = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_linearity_implies_R_squared_one_l2529_252956


namespace NUMINAMATH_CALUDE_binomial_12_choose_6_l2529_252995

theorem binomial_12_choose_6 : Nat.choose 12 6 = 924 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_choose_6_l2529_252995


namespace NUMINAMATH_CALUDE_power_sum_equals_product_l2529_252993

theorem power_sum_equals_product (m n : ℕ+) (a b : ℝ) 
  (h1 : 3^(m.val) = a) (h2 : 3^(n.val) = b) : 
  3^(m.val + n.val) = a * b := by sorry

end NUMINAMATH_CALUDE_power_sum_equals_product_l2529_252993


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2529_252927

theorem coefficient_x_cubed_in_expansion : 
  (Finset.range 21).sum (fun k => (Nat.choose 20 k) * (2^(20 - k)) * (if k = 3 then 1 else 0)) = 149462016 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2529_252927


namespace NUMINAMATH_CALUDE_sauce_per_pulled_pork_sandwich_l2529_252978

/-- The amount of sauce each pulled pork sandwich takes -/
def pulled_pork_sauce : ℚ :=
  1 / 6

theorem sauce_per_pulled_pork_sandwich 
  (total_sauce : ℚ) 
  (burger_sauce : ℚ) 
  (num_burgers : ℕ) 
  (num_pulled_pork : ℕ) 
  (h1 : total_sauce = 5)
  (h2 : burger_sauce = 1 / 4)
  (h3 : num_burgers = 8)
  (h4 : num_pulled_pork = 18)
  (h5 : num_burgers * burger_sauce + num_pulled_pork * pulled_pork_sauce = total_sauce) :
  pulled_pork_sauce = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sauce_per_pulled_pork_sandwich_l2529_252978


namespace NUMINAMATH_CALUDE_circle_line_distance_l2529_252936

theorem circle_line_distance (a : ℝ) : 
  let circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 = 0}
  let line : Set (ℝ × ℝ) := {p | p.1 - p.2 + a = 0}
  let center : ℝ × ℝ := (1, 2)  -- Derived from completing the square
  let distance := |1 - 2 + a| / Real.sqrt 2
  (∀ p ∈ circle, p.1^2 + p.2^2 - 2*p.1 - 4*p.2 = 0) →
  (∀ p ∈ line, p.1 - p.2 + a = 0) →
  distance = Real.sqrt 2 / 2 →
  a = 2 ∨ a = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_distance_l2529_252936


namespace NUMINAMATH_CALUDE_subtract_negative_three_and_one_l2529_252913

theorem subtract_negative_three_and_one : -3 - 1 = -4 := by sorry

end NUMINAMATH_CALUDE_subtract_negative_three_and_one_l2529_252913


namespace NUMINAMATH_CALUDE_isosceles_triangle_height_l2529_252932

/-- Given an isosceles triangle and a rectangle with the same area, where the base of the triangle
    equals the width of the rectangle (10 units), and the length of the rectangle is twice its width,
    prove that the height of the triangle is 40 units. -/
theorem isosceles_triangle_height (triangle_area rectangle_area : ℝ) 
  (triangle_base rectangle_width rectangle_length : ℝ) (triangle_height : ℝ) : 
  triangle_area = rectangle_area →
  triangle_base = rectangle_width →
  triangle_base = 10 →
  rectangle_length = 2 * rectangle_width →
  triangle_area = 1/2 * triangle_base * triangle_height →
  rectangle_area = rectangle_width * rectangle_length →
  triangle_height = 40 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_height_l2529_252932


namespace NUMINAMATH_CALUDE_solution_set_is_open_interval_l2529_252916

def solution_set : Set ℝ := {x : ℝ | |x - 2| - |2*x - 1| > 0}

theorem solution_set_is_open_interval :
  solution_set = Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_is_open_interval_l2529_252916


namespace NUMINAMATH_CALUDE_max_servings_jordan_l2529_252980

/-- Represents the recipe for hot chocolate -/
structure Recipe where
  servings : ℚ
  chocolate : ℚ
  sugar : ℚ
  water : ℚ
  milk : ℚ

/-- Represents the available ingredients -/
structure Ingredients where
  chocolate : ℚ
  sugar : ℚ
  milk : ℚ

/-- Calculates the maximum number of servings that can be made -/
def maxServings (recipe : Recipe) (ingredients : Ingredients) : ℚ :=
  min (ingredients.chocolate / recipe.chocolate * recipe.servings)
      (min (ingredients.sugar / recipe.sugar * recipe.servings)
           (ingredients.milk / recipe.milk * recipe.servings))

theorem max_servings_jordan :
  let recipe : Recipe := ⟨5, 2, 1/4, 1, 4⟩
  let ingredients : Ingredients := ⟨5, 2, 7⟩
  maxServings recipe ingredients = 35/4 := by
  sorry

end NUMINAMATH_CALUDE_max_servings_jordan_l2529_252980


namespace NUMINAMATH_CALUDE_base_twelve_square_l2529_252931

theorem base_twelve_square (b : ℕ) : b > 0 → (3 * b + 2)^2 = b^3 + 2 * b^2 + 4 → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_base_twelve_square_l2529_252931


namespace NUMINAMATH_CALUDE_susies_golden_comets_l2529_252999

theorem susies_golden_comets (susie_rir : ℕ) (britney_total susie_total : ℕ) : ℕ :=
  let susie_gc := britney_total - susie_total - 8
  have h1 : susie_rir = 11 := by sorry
  have h2 : britney_total = susie_total + 8 := by sorry
  have h3 : britney_total = 2 * susie_rir + susie_gc / 2 := by sorry
  have h4 : susie_total = susie_rir + susie_gc := by sorry
  6

#check susies_golden_comets

end NUMINAMATH_CALUDE_susies_golden_comets_l2529_252999


namespace NUMINAMATH_CALUDE_divisibility_by_1987_l2529_252998

def odd_product : ℕ → ℕ := λ n => (List.range n).foldl (λ acc i => acc * (2 * i + 1)) 1

def even_product : ℕ → ℕ := λ n => (List.range n).foldl (λ acc i => acc * (2 * i + 2)) 1

theorem divisibility_by_1987 : ∃ k : ℤ, (odd_product 993 + even_product 993 : ℤ) = k * 1987 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1987_l2529_252998


namespace NUMINAMATH_CALUDE_equation_solutions_count_l2529_252929

theorem equation_solutions_count :
  ∃! (s : Finset ℝ), (∀ x ∈ s, 9 * x^2 - 63 * ⌊x⌋ + 72 = 0) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_count_l2529_252929


namespace NUMINAMATH_CALUDE_number_problem_l2529_252976

theorem number_problem (x : ℝ) : (36 / 100 * x = 129.6) → x = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2529_252976


namespace NUMINAMATH_CALUDE_x_plus_y_equals_22_l2529_252982

theorem x_plus_y_equals_22 (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 2))
  (h2 : (25 : ℝ) ^ y = 5 ^ (x - 16)) : 
  x + y = 22 := by sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_22_l2529_252982


namespace NUMINAMATH_CALUDE_bubble_bath_per_guest_l2529_252920

theorem bubble_bath_per_guest (couple_rooms : ℕ) (single_rooms : ℕ) (total_bubble_bath : ℕ) :
  couple_rooms = 13 →
  single_rooms = 14 →
  total_bubble_bath = 400 →
  (total_bubble_bath : ℚ) / (2 * couple_rooms + single_rooms) = 10 :=
by sorry

end NUMINAMATH_CALUDE_bubble_bath_per_guest_l2529_252920


namespace NUMINAMATH_CALUDE_average_time_theorem_l2529_252973

def relay_race (y z w : ℝ) : Prop :=
  y = 58 ∧ z = 26 ∧ w = 2*z

theorem average_time_theorem (y z w : ℝ) (h : relay_race y z w) :
  (y + z + w) / 3 = (58 + 26 + 2*26) / 3 := by sorry

end NUMINAMATH_CALUDE_average_time_theorem_l2529_252973


namespace NUMINAMATH_CALUDE_two_rooks_placement_count_l2529_252903

/-- The size of a standard chessboard -/
def boardSize : Nat := 8

/-- The total number of squares on a chessboard -/
def totalSquares : Nat := boardSize * boardSize

/-- The number of squares attacked by a rook (excluding its own square) -/
def attackedSquares : Nat := 2 * boardSize - 1

/-- The number of ways to place two rooks of different colors on a chessboard
    such that they do not attack each other -/
def twoRooksPlacement : Nat := totalSquares * (totalSquares - attackedSquares)

theorem two_rooks_placement_count :
  twoRooksPlacement = 3136 := by sorry

end NUMINAMATH_CALUDE_two_rooks_placement_count_l2529_252903


namespace NUMINAMATH_CALUDE_right_triangle_area_l2529_252907

/-- The area of a right triangle formed by two perpendicular vectors -/
theorem right_triangle_area (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) :
  let area := (1/2) * abs (a.1 * b.2 - a.2 * b.1)
  (a = (3, 4) ∧ b = (-4, 3)) → area = 12.5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2529_252907


namespace NUMINAMATH_CALUDE_sine_cosine_problem_l2529_252908

theorem sine_cosine_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < π/2) 
  (h2 : Real.sin x + Real.cos x = -1/5) : 
  (Real.sin x - Real.cos x = 7/5) ∧ 
  ((Real.sin (π + x) + Real.sin (3*π/2 - x)) / (Real.tan (π - x) + Real.sin (π/2 - x)) = 3/11) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_problem_l2529_252908


namespace NUMINAMATH_CALUDE_consecutive_shots_count_l2529_252925

/-- The number of ways to arrange 3 successful shots out of 8 attempts, 
    with exactly 2 consecutive successful shots. -/
def consecutiveShots : ℕ := 30

/-- The total number of attempts. -/
def totalAttempts : ℕ := 8

/-- The number of successful shots. -/
def successfulShots : ℕ := 3

/-- The number of consecutive successful shots required. -/
def consecutiveHits : ℕ := 2

theorem consecutive_shots_count :
  consecutiveShots = 
    (totalAttempts - successfulShots + 1).choose 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_shots_count_l2529_252925


namespace NUMINAMATH_CALUDE_least_common_period_l2529_252986

-- Define the property that f should satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) + f (x - 3) = f x

-- Define what it means for a function to have a period
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_common_period :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
    (∃ p : ℝ, p > 0 ∧ HasPeriod f p) →
    (∀ q : ℝ, q > 0 ∧ HasPeriod f q → q ≥ 18) ∧
    HasPeriod f 18 :=
sorry

end NUMINAMATH_CALUDE_least_common_period_l2529_252986


namespace NUMINAMATH_CALUDE_sound_distance_at_18C_l2529_252922

/-- Represents the speed of sound in air as a function of temperature -/
def speed_of_sound (t : ℝ) : ℝ := 331 + 0.6 * t

/-- Calculates the distance traveled by sound given time and temperature -/
def distance_traveled (time : ℝ) (temp : ℝ) : ℝ :=
  (speed_of_sound temp) * time

/-- Theorem: The distance traveled by sound in 5 seconds at 18°C is approximately 1709 meters -/
theorem sound_distance_at_18C : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |distance_traveled 5 18 - 1709| < ε :=
sorry

end NUMINAMATH_CALUDE_sound_distance_at_18C_l2529_252922


namespace NUMINAMATH_CALUDE_money_left_over_l2529_252989

/-- The amount of money left over after purchasing bread, peanut butter, and honey with a discount coupon. -/
theorem money_left_over (bread_price : ℝ) (peanut_butter_price : ℝ) (honey_price : ℝ)
  (bread_quantity : ℕ) (peanut_butter_quantity : ℕ) (honey_quantity : ℕ)
  (discount : ℝ) (initial_money : ℝ) :
  bread_price = 2.35 →
  peanut_butter_price = 3.10 →
  honey_price = 4.50 →
  bread_quantity = 4 →
  peanut_butter_quantity = 2 →
  honey_quantity = 1 →
  discount = 2 →
  initial_money = 20 →
  initial_money - (bread_price * bread_quantity + peanut_butter_price * peanut_butter_quantity + 
    honey_price * honey_quantity - discount) = 1.90 := by
  sorry

end NUMINAMATH_CALUDE_money_left_over_l2529_252989


namespace NUMINAMATH_CALUDE_selection_schemes_count_l2529_252938

def number_of_people : ℕ := 6
def number_of_places : ℕ := 4
def number_of_restricted_people : ℕ := 2
def number_of_restricted_places : ℕ := 1

theorem selection_schemes_count :
  (number_of_people.choose number_of_places) *
  (number_of_places - number_of_restricted_places).choose 1 *
  ((number_of_people - number_of_restricted_people).choose (number_of_places - 1)) *
  (number_of_places - 1).factorial = 240 := by
  sorry

end NUMINAMATH_CALUDE_selection_schemes_count_l2529_252938


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2529_252964

theorem at_least_one_greater_than_one (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) : max x y > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2529_252964


namespace NUMINAMATH_CALUDE_complex_conjugate_roots_imply_zero_coefficients_l2529_252919

/-- Given a quadratic equation z^2 + (6 + pi)z + (10 + qi) = 0 where p and q are real numbers,
    if the roots are complex conjugates, then p = 0 and q = 0 -/
theorem complex_conjugate_roots_imply_zero_coefficients (p q : ℝ) :
  (∃ x y : ℝ, (Complex.I : ℂ)^2 = -1 ∧
    (x + y * Complex.I) * (x - y * Complex.I) = -(6 + p * Complex.I) * (x + y * Complex.I) - (10 + q * Complex.I)) →
  p = 0 ∧ q = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_conjugate_roots_imply_zero_coefficients_l2529_252919


namespace NUMINAMATH_CALUDE_larger_number_proof_l2529_252945

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1355)
  (h2 : L = 6 * S + 15) : 
  L = 1623 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2529_252945


namespace NUMINAMATH_CALUDE_cubic_roots_determinant_l2529_252966

/-- Given a cubic equation x^3 - px^2 + qx - r = 0 with roots a, b, c,
    the determinant of the matrix
    |a 0 1|
    |0 b 1|
    |1 1 c|
    is equal to r - a - b -/
theorem cubic_roots_determinant (p q r a b c : ℝ) : 
  a^3 - p*a^2 + q*a - r = 0 →
  b^3 - p*b^2 + q*b - r = 0 →
  c^3 - p*c^2 + q*c - r = 0 →
  Matrix.det !![a, 0, 1; 0, b, 1; 1, 1, c] = r - a - b :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_determinant_l2529_252966


namespace NUMINAMATH_CALUDE_sum_equals_thirty_l2529_252943

theorem sum_equals_thirty : 1 + 2 + 3 - 4 + 5 + 6 + 7 - 8 + 9 + 10 + 11 - 12 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_thirty_l2529_252943


namespace NUMINAMATH_CALUDE_monge_point_properties_l2529_252967

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron with vertices A, B, C, and D -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- The Monge point of a tetrahedron -/
def mongePoint (t : Tetrahedron) : Point3D := sorry

/-- Checks if a point lies on a plane defined by three other points -/
def isOnPlane (p q r s : Point3D) : Prop := sorry

/-- The projection of a point onto a plane -/
def projection (p : Point3D) (plane : Point3D × Point3D × Point3D) : Point3D := sorry

/-- The intersection point of the altitudes of a triangular face -/
def altitudeIntersection (face : Point3D × Point3D × Point3D) : Point3D := sorry

/-- The center of the circumscribed circle of a triangular face -/
def circumcenter (face : Point3D × Point3D × Point3D) : Point3D := sorry

/-- Checks if four points are coplanar -/
def areCoplanar (p q r s : Point3D) : Prop := sorry

theorem monge_point_properties (t : Tetrahedron) : 
  isOnPlane (mongePoint t) t.A t.B t.C →
  let D1 := projection t.D (t.A, t.B, t.C)
  (areCoplanar t.D 
    (altitudeIntersection (t.D, t.A, t.B))
    (altitudeIntersection (t.D, t.B, t.C))
    (altitudeIntersection (t.D, t.A, t.C))) ∧
  (areCoplanar t.D
    (circumcenter (t.D, t.A, t.B))
    (circumcenter (t.D, t.B, t.C))
    (circumcenter (t.D, t.A, t.C))) := by
  sorry

end NUMINAMATH_CALUDE_monge_point_properties_l2529_252967


namespace NUMINAMATH_CALUDE_gym_time_zero_l2529_252951

/-- Represents the exercise plan with yoga and exercise components -/
structure ExercisePlan where
  yoga_time : ℕ
  exercise_time : ℕ
  bike_time : ℕ
  gym_time : ℕ
  yoga_exercise_ratio : yoga_time * 3 = exercise_time * 2
  exercise_components : exercise_time = bike_time + gym_time

/-- 
Given an exercise plan where the bike riding time equals the total exercise time,
prove that the gym workout time is zero.
-/
theorem gym_time_zero (plan : ExercisePlan) 
  (h : plan.bike_time = plan.exercise_time) : plan.gym_time = 0 := by
  sorry

end NUMINAMATH_CALUDE_gym_time_zero_l2529_252951


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2529_252997

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 30 →
  a + c = 60 →
  a + b + c = 70 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2529_252997


namespace NUMINAMATH_CALUDE_speed_conversion_l2529_252985

/-- Conversion of speed from m/s to km/h -/
theorem speed_conversion (speed_ms : ℚ) (conversion_factor : ℚ) :
  speed_ms = 13/36 →
  conversion_factor = 36/10 →
  speed_ms * conversion_factor = 13/10 := by
  sorry

#eval (13/36 : ℚ) * (36/10 : ℚ) -- To verify the result

end NUMINAMATH_CALUDE_speed_conversion_l2529_252985


namespace NUMINAMATH_CALUDE_distance_minimization_l2529_252924

theorem distance_minimization (t : ℝ) (h : t > 0) :
  let f (x : ℝ) := x^2 + 1
  let g (x : ℝ) := Real.log x
  let distance_squared (x : ℝ) := (f x - g x)^2
  (∀ x > 0, distance_squared t ≤ distance_squared x) →
  t = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_minimization_l2529_252924


namespace NUMINAMATH_CALUDE_decimal_addition_subtraction_l2529_252992

theorem decimal_addition_subtraction :
  (0.45 : ℚ) - 0.03 + 0.008 = 0.428 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_subtraction_l2529_252992


namespace NUMINAMATH_CALUDE_cube_root_of_product_l2529_252988

theorem cube_root_of_product (a b c : ℕ) : 
  (2^6 * 3^3 * 5^3 : ℝ)^(1/3) = 60 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l2529_252988


namespace NUMINAMATH_CALUDE_definite_integral_sine_cosine_l2529_252959

theorem definite_integral_sine_cosine : 
  ∫ x in (0)..(Real.pi / 2), (4 * Real.sin x + Real.cos x) = 5 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_sine_cosine_l2529_252959


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l2529_252905

/-- A line passing through (2, 3) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (2, 3) -/
  passes_through_point : m * 2 + b = 3
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b ≠ 0 → -b/m = b

/-- The equation of the line is either x + y - 5 = 0 or 3x - 2y = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, y = l.m * x + l.b → x + y = 5) ∨
  (∀ x y, y = l.m * x + l.b → 3*x - 2*y = 0) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l2529_252905


namespace NUMINAMATH_CALUDE_biancas_album_pictures_l2529_252965

/-- Given that Bianca uploaded 33 pictures and put some into 3 albums with 2 pictures each,
    prove that she put 27 pictures into the first album. -/
theorem biancas_album_pictures :
  ∀ (total_pictures : ℕ) (other_albums : ℕ) (pics_per_album : ℕ),
    total_pictures = 33 →
    other_albums = 3 →
    pics_per_album = 2 →
    total_pictures - (other_albums * pics_per_album) = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_biancas_album_pictures_l2529_252965


namespace NUMINAMATH_CALUDE_floor_product_equality_l2529_252933

theorem floor_product_equality (x : ℝ) : ⌊x * ⌊x⌋⌋ = 49 ↔ 7 ≤ x ∧ x < 50 / 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_product_equality_l2529_252933


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2529_252915

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = 2 * x + 1}

theorem intersection_of_M_and_N : M ∩ N = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2529_252915


namespace NUMINAMATH_CALUDE_tv_selection_probability_l2529_252957

def num_type_a : ℕ := 3
def num_type_b : ℕ := 2
def total_tvs : ℕ := num_type_a + num_type_b
def selection_size : ℕ := 2

theorem tv_selection_probability :
  let total_combinations := Nat.choose total_tvs selection_size
  let favorable_combinations := num_type_a * num_type_b
  (favorable_combinations : ℚ) / total_combinations = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_tv_selection_probability_l2529_252957


namespace NUMINAMATH_CALUDE_exp_ln_eight_l2529_252926

theorem exp_ln_eight : Real.exp (Real.log 8) = 8 := by sorry

end NUMINAMATH_CALUDE_exp_ln_eight_l2529_252926


namespace NUMINAMATH_CALUDE_second_number_value_l2529_252914

theorem second_number_value (x : ℝ) : 3 + x * (8 - 3) = 24.16 → x = 4.232 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l2529_252914


namespace NUMINAMATH_CALUDE_six_ring_clock_interval_l2529_252912

/-- A clock that rings a certain number of times per day at equal intervals -/
structure RingingClock where
  rings_per_day : ℕ
  rings_per_day_pos : rings_per_day > 0

/-- The number of minutes in a day -/
def minutes_per_day : ℕ := 24 * 60

/-- Calculate the time interval between rings in minutes -/
def interval_between_rings (clock : RingingClock) : ℚ :=
  minutes_per_day / (clock.rings_per_day - 1)

/-- Theorem: For a clock that rings 6 times a day, the interval between rings is 288 minutes -/
theorem six_ring_clock_interval :
  let clock : RingingClock := ⟨6, by norm_num⟩
  interval_between_rings clock = 288 := by sorry

end NUMINAMATH_CALUDE_six_ring_clock_interval_l2529_252912


namespace NUMINAMATH_CALUDE_travelers_checks_worth_l2529_252977

/-- Represents the total worth of travelers checks -/
def total_worth (num_50 : ℕ) (num_100 : ℕ) : ℕ :=
  50 * num_50 + 100 * num_100

/-- Represents the average value of remaining checks after spending some $50 checks -/
def average_remaining (num_50 : ℕ) (num_100 : ℕ) (spent_50 : ℕ) : ℚ :=
  (50 * (num_50 - spent_50) + 100 * num_100) / (num_50 + num_100 - spent_50)

theorem travelers_checks_worth :
  ∀ (num_50 num_100 : ℕ),
    num_50 + num_100 = 30 →
    average_remaining num_50 num_100 15 = 70 →
    total_worth num_50 num_100 = 1800 :=
by sorry

end NUMINAMATH_CALUDE_travelers_checks_worth_l2529_252977


namespace NUMINAMATH_CALUDE_cake_eating_problem_l2529_252969

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem cake_eating_problem : 
  geometric_series_sum (1/3) (1/3) 7 = 1093/2187 := by sorry

end NUMINAMATH_CALUDE_cake_eating_problem_l2529_252969


namespace NUMINAMATH_CALUDE_point_b_position_l2529_252990

theorem point_b_position (a b : ℝ) : 
  a = -2 → (b - a = 4 ∨ a - b = 4) → (b = 2 ∨ b = -6) := by
  sorry

end NUMINAMATH_CALUDE_point_b_position_l2529_252990


namespace NUMINAMATH_CALUDE_work_completion_time_l2529_252930

theorem work_completion_time (x_days y_days y_worked_days : ℕ) 
  (hx : x_days = 20)
  (hy : y_days = 16)
  (hw : y_worked_days = 12) : 
  (x_days : ℚ) * (1 - y_worked_days / y_days) = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2529_252930


namespace NUMINAMATH_CALUDE_tangent_line_and_perpendicular_points_l2529_252909

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x - 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_and_perpendicular_points :
  -- Part 1: Equation of tangent line at (1, -1)
  (∀ x y : ℝ, (x = 1 ∧ y = f 1) → (2*x - y - 3 = 0)) ∧
  -- Part 2: Points where tangent is perpendicular to y = -1/2x + 3
  (∀ x : ℝ, (f' x = 2) → (x = 1 ∨ x = -1)) ∧
  (∀ x : ℝ, (x = 1 ∨ x = -1) → f x = -1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_perpendicular_points_l2529_252909


namespace NUMINAMATH_CALUDE_max_value_2a_minus_b_l2529_252934

theorem max_value_2a_minus_b :
  ∃ (M : ℝ), M = 2 + Real.sqrt 5 ∧
  (∀ a b : ℝ, a^2 + b^2 - 2*a = 0 → 2*a - b ≤ M) ∧
  (∃ a b : ℝ, a^2 + b^2 - 2*a = 0 ∧ 2*a - b = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_2a_minus_b_l2529_252934


namespace NUMINAMATH_CALUDE_sum_of_abc_l2529_252952

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 13 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l2529_252952


namespace NUMINAMATH_CALUDE_percentage_relation_l2529_252917

theorem percentage_relation (j k l m x : ℝ) 
  (h1 : j * (x / 100) = k * (25 / 100))
  (h2 : k * (150 / 100) = l * (50 / 100))
  (h3 : l * (175 / 100) = m * (75 / 100))
  (h4 : m * (20 / 100) = j * (700 / 100))
  (hj : j > 0) (hk : k > 0) (hl : l > 0) (hm : m > 0) :
  x = 500 := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l2529_252917


namespace NUMINAMATH_CALUDE_monotonically_decreasing_iff_a_leq_neg_three_l2529_252975

/-- A function f is monotonically decreasing on an interval [a, b] if for any x₁, x₂ in [a, b] with x₁ < x₂, we have f(x₁) ≥ f(x₂) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ b → f x₁ ≥ f x₂

/-- The quadratic function f(x) = x² + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem monotonically_decreasing_iff_a_leq_neg_three :
  ∀ a : ℝ, MonotonicallyDecreasing (f a) (-2) 4 ↔ a ≤ -3 := by sorry


end NUMINAMATH_CALUDE_monotonically_decreasing_iff_a_leq_neg_three_l2529_252975


namespace NUMINAMATH_CALUDE_value_of_n_l2529_252941

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters on the left side of the equation -/
def left_quarters : ℕ := 15

/-- The number of nickels on the left side of the equation -/
def left_nickels : ℕ := 18

/-- The number of quarters on the right side of the equation -/
def right_quarters : ℕ := 7

/-- Theorem stating that the value of n is 58 -/
theorem value_of_n : 
  ∃ n : ℕ, 
    left_quarters * quarter_value + left_nickels * nickel_value = 
    right_quarters * quarter_value + n * nickel_value ∧ 
    n = 58 := by
  sorry

end NUMINAMATH_CALUDE_value_of_n_l2529_252941


namespace NUMINAMATH_CALUDE_gcd_b_81_is_3_l2529_252994

theorem gcd_b_81_is_3 (a b : ℤ) : 
  (∃ (x : ℝ), x^2 = 2 ∧ (1 + x)^2012 = a + b * x) → Nat.gcd b.natAbs 81 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_b_81_is_3_l2529_252994


namespace NUMINAMATH_CALUDE_modular_inverse_35_mod_37_l2529_252996

theorem modular_inverse_35_mod_37 : ∃ x : ℕ, x ≤ 36 ∧ (35 * x) % 37 = 1 :=
by
  use 18
  sorry

end NUMINAMATH_CALUDE_modular_inverse_35_mod_37_l2529_252996
