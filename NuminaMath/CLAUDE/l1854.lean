import Mathlib

namespace NUMINAMATH_CALUDE_nasadkas_in_barrel_l1854_185445

/-- The volume of a barrel -/
def barrel : ℝ := sorry

/-- The volume of a nasadka -/
def nasadka : ℝ := sorry

/-- The volume of a bucket -/
def bucket : ℝ := sorry

/-- The first condition: 1 barrel + 20 buckets = 3 barrels -/
axiom condition1 : barrel + 20 * bucket = 3 * barrel

/-- The second condition: 19 barrels + 1 nasadka + 15.5 buckets = 20 barrels + 8 buckets -/
axiom condition2 : 19 * barrel + nasadka + 15.5 * bucket = 20 * barrel + 8 * bucket

/-- The theorem stating that there are 4 nasadkas in a barrel -/
theorem nasadkas_in_barrel : barrel / nasadka = 4 := by sorry

end NUMINAMATH_CALUDE_nasadkas_in_barrel_l1854_185445


namespace NUMINAMATH_CALUDE_probability_of_specific_pairing_l1854_185497

theorem probability_of_specific_pairing (n : ℕ) (h : n = 25) :
  let total_students := n
  let available_partners := n - 1
  (1 : ℚ) / available_partners = 1 / 24 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_specific_pairing_l1854_185497


namespace NUMINAMATH_CALUDE_clementine_baked_72_cookies_l1854_185453

/-- The number of cookies Clementine baked -/
def clementine_cookies : ℕ := 72

/-- The number of cookies Jake baked -/
def jake_cookies : ℕ := 2 * clementine_cookies

/-- The number of cookies Tory baked -/
def tory_cookies : ℕ := (clementine_cookies + jake_cookies) / 2

/-- The price of each cookie in dollars -/
def cookie_price : ℕ := 2

/-- The total amount of money made from selling cookies in dollars -/
def total_money : ℕ := 648

theorem clementine_baked_72_cookies :
  clementine_cookies = 72 ∧
  jake_cookies = 2 * clementine_cookies ∧
  tory_cookies = (clementine_cookies + jake_cookies) / 2 ∧
  cookie_price = 2 ∧
  total_money = 648 ∧
  total_money = cookie_price * (clementine_cookies + jake_cookies + tory_cookies) :=
by sorry

end NUMINAMATH_CALUDE_clementine_baked_72_cookies_l1854_185453


namespace NUMINAMATH_CALUDE_geometric_sequence_term_count_l1854_185440

theorem geometric_sequence_term_count :
  ∀ (a : ℕ → ℚ),
  (∀ k : ℕ, a (k + 1) = a k * (1/2)) →  -- Geometric sequence with q = 1/2
  a 1 = 1/2 →                           -- First term a₁ = 1/2
  (∃ n : ℕ, a n = 1/32) →               -- Some term aₙ = 1/32
  ∃ n : ℕ, n = 5 ∧ a n = 1/32 :=        -- The term count n is 5
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_term_count_l1854_185440


namespace NUMINAMATH_CALUDE_fA_inter_fB_l1854_185447

def f (n : ℕ+) : ℕ := 2 * n + 1

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {3, 4, 5, 6, 7}

def fA : Set ℕ+ := {n : ℕ+ | f n ∈ A}
def fB : Set ℕ+ := {m : ℕ+ | f m ∈ B}

theorem fA_inter_fB : fA ∩ fB = {1, 2} := by sorry

end NUMINAMATH_CALUDE_fA_inter_fB_l1854_185447


namespace NUMINAMATH_CALUDE_number_of_boys_l1854_185470

theorem number_of_boys (total_pupils : ℕ) (number_of_girls : ℕ) 
  (h1 : total_pupils = 485) 
  (h2 : number_of_girls = 232) : 
  total_pupils - number_of_girls = 253 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l1854_185470


namespace NUMINAMATH_CALUDE_two_distinct_real_roots_l1854_185454

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 + x - 2 = m

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ :=
  4 * m + 9

-- Theorem statement
theorem two_distinct_real_roots (m : ℝ) (h : m > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation x₁ m ∧ quadratic_equation x₂ m :=
sorry

end NUMINAMATH_CALUDE_two_distinct_real_roots_l1854_185454


namespace NUMINAMATH_CALUDE_painting_time_calculation_l1854_185416

/-- Given an artist's weekly painting hours and production rate over four weeks,
    calculate the time needed to complete one painting. -/
theorem painting_time_calculation (weekly_hours : ℕ) (paintings_in_four_weeks : ℕ) :
  weekly_hours = 30 →
  paintings_in_four_weeks = 40 →
  (4 * weekly_hours) / paintings_in_four_weeks = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_painting_time_calculation_l1854_185416


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1854_185483

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1854_185483


namespace NUMINAMATH_CALUDE_square_sum_constant_l1854_185496

theorem square_sum_constant (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_constant_l1854_185496


namespace NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l1854_185499

/-- 
Given a cone whose lateral surface is a semicircle with radius a,
prove that the height of the cone is (√3/2)a
-/
theorem cone_height_from_lateral_surface (a : ℝ) (h : a > 0) :
  let slant_height := a
  let base_circumference := π * a
  let base_radius := a / 2
  let height := Real.sqrt ((3 * a^2) / 4)
  height = (Real.sqrt 3 / 2) * a :=
by sorry

end NUMINAMATH_CALUDE_cone_height_from_lateral_surface_l1854_185499


namespace NUMINAMATH_CALUDE_solve_for_y_l1854_185458

theorem solve_for_y (x y : ℝ) (h1 : x + 2*y = 10) (h2 : x = 2) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1854_185458


namespace NUMINAMATH_CALUDE_square_root_equation_l1854_185411

theorem square_root_equation (x : ℝ) : Real.sqrt (x - 3) = 10 → x = 103 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l1854_185411


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l1854_185406

theorem largest_two_digit_prime_factor_of_binomial :
  ∃ (p : ℕ), 
    p.Prime ∧ 
    10 ≤ p ∧ p < 100 ∧
    p ∣ Nat.choose 150 75 ∧
    (∀ q : ℕ, q.Prime → 10 ≤ q → q < 100 → q ∣ Nat.choose 150 75 → q ≤ p) ∧
    (∀ q : ℕ, q > p → ¬(q.Prime ∧ 10 ≤ q ∧ q < 100 ∧ q ∣ Nat.choose 150 75)) :=
by
  sorry

#check largest_two_digit_prime_factor_of_binomial

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l1854_185406


namespace NUMINAMATH_CALUDE_complement_of_A_l1854_185482

universe u

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}

theorem complement_of_A (x : Nat) : x ∈ (U \ A) ↔ x = 3 ∨ x = 4 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1854_185482


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_11_after_change_all_replacements_divisible_by_11_l1854_185463

/-- A function that replaces a digit at a given position in a number with a new digit. -/
def replaceDigit (n : ℕ) (pos : ℕ) (newDigit : ℕ) : ℕ :=
  sorry

/-- A function that checks if a number is divisible by 11. -/
def isDivisibleBy11 (n : ℕ) : Prop :=
  n % 11 = 0

/-- A function that generates all possible numbers after replacing one digit. -/
def allPossibleReplacements (n : ℕ) : List ℕ :=
  sorry

theorem smallest_number_divisible_by_11_after_change : 
  ∀ n : ℕ, n < 909090909 → 
  ∃ m ∈ allPossibleReplacements n, ¬isDivisibleBy11 m :=
by sorry

theorem all_replacements_divisible_by_11 : 
  ∀ m ∈ allPossibleReplacements 909090909, isDivisibleBy11 m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_11_after_change_all_replacements_divisible_by_11_l1854_185463


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1854_185457

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 7 / Real.sqrt 8) = Real.sqrt 70 / 16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1854_185457


namespace NUMINAMATH_CALUDE_evaluate_g_l1854_185460

/-- The function g(x) = 3x^2 - 5x + 8 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 8

/-- Theorem: 3g(2) + 2g(-2) = 90 -/
theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_g_l1854_185460


namespace NUMINAMATH_CALUDE_max_triangles_is_28_l1854_185405

/-- The number of points on the hypotenuse of a right triangle with legs of length 7 -/
def hypotenuse_points : ℕ := 8

/-- The maximum number of triangles that can be formed within the right triangle -/
def max_triangles : ℕ := Nat.choose hypotenuse_points 2

/-- Theorem stating the maximum number of triangles is 28 -/
theorem max_triangles_is_28 : max_triangles = 28 := by sorry

end NUMINAMATH_CALUDE_max_triangles_is_28_l1854_185405


namespace NUMINAMATH_CALUDE_two_point_form_always_valid_two_point_form_works_for_vertical_lines_l1854_185415

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a line passing through two points --/
def line_equation (p1 p2 : Point) (x y : ℝ) : Prop :=
  (y - p1.y) * (p2.x - p1.x) = (x - p1.x) * (p2.y - p1.y)

/-- Theorem: The two-point form of a line equation is always valid --/
theorem two_point_form_always_valid (p1 p2 : Point) (h : p1 ≠ p2) :
  ∃ (l : Line), ∀ (x y : ℝ), (y = l.slope * x + l.intercept) ↔ line_equation p1 p2 x y :=
sorry

/-- Corollary: The two-point form works even for vertical lines --/
theorem two_point_form_works_for_vertical_lines (p1 p2 : Point) (h : p1.x = p2.x) (h' : p1 ≠ p2) :
  ∀ (y : ℝ), ∃ (x : ℝ), line_equation p1 p2 x y :=
sorry

end NUMINAMATH_CALUDE_two_point_form_always_valid_two_point_form_works_for_vertical_lines_l1854_185415


namespace NUMINAMATH_CALUDE_grass_seed_cost_l1854_185475

structure GrassSeed where
  bag5_price : ℝ
  bag10_price : ℝ
  bag25_price : ℝ
  min_purchase : ℝ
  max_purchase : ℝ
  least_cost : ℝ

def is_valid_purchase (gs : GrassSeed) (x : ℕ) (y : ℕ) (z : ℕ) : Prop :=
  5 * x + 10 * y + 25 * z ≥ gs.min_purchase ∧
  5 * x + 10 * y + 25 * z ≤ gs.max_purchase

def total_cost (gs : GrassSeed) (x : ℕ) (y : ℕ) (z : ℕ) : ℝ :=
  gs.bag5_price * x + gs.bag10_price * y + gs.bag25_price * z

theorem grass_seed_cost (gs : GrassSeed) 
  (h1 : gs.bag5_price = 13.80)
  (h2 : gs.bag10_price = 20.43)
  (h3 : gs.min_purchase = 65)
  (h4 : gs.max_purchase = 80)
  (h5 : gs.least_cost = 98.73) :
  gs.bag25_price = 17.01 :=
by sorry

end NUMINAMATH_CALUDE_grass_seed_cost_l1854_185475


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1854_185417

/-- Given a circle with equation x^2 + y^2 - 4x + 2y - 4 = 0, 
    prove that its center is at (2, -1) and its radius is 3. -/
theorem circle_center_and_radius : 
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    (C = (2, -1) ∧ r = 3) ∧ 
    ∀ (x y : ℝ), x^2 + y^2 - 4*x + 2*y - 4 = 0 ↔ (x - C.1)^2 + (y - C.2)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1854_185417


namespace NUMINAMATH_CALUDE_composite_expression_l1854_185465

/-- A positive integer is composite if it can be expressed as a product of two integers
    greater than 1. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- Every composite positive integer can be expressed as xy+xz+yz+1,
    where x, y, and z are positive integers. -/
theorem composite_expression (n : ℕ) (h : IsComposite n) :
    ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ n = x * y + x * z + y * z + 1 := by
  sorry

end NUMINAMATH_CALUDE_composite_expression_l1854_185465


namespace NUMINAMATH_CALUDE_student_marks_average_l1854_185421

/-- Given a student's marks in mathematics, physics, and chemistry, 
    prove that the average of mathematics and chemistry marks is 20. -/
theorem student_marks_average (M P C : ℝ) 
  (h1 : M + P = 20)
  (h2 : C = P + 20) : 
  (M + C) / 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_average_l1854_185421


namespace NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_l1854_185414

theorem fraction_of_fraction_of_fraction (n : ℚ) : n = 72 → (1/2 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * n = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_of_fraction_l1854_185414


namespace NUMINAMATH_CALUDE_ten_percent_increase_l1854_185439

theorem ten_percent_increase (original : ℝ) (increased : ℝ) : 
  original = 600 → increased = original * 1.1 → increased = 660 := by
  sorry

end NUMINAMATH_CALUDE_ten_percent_increase_l1854_185439


namespace NUMINAMATH_CALUDE_power_division_l1854_185432

theorem power_division (a : ℝ) : a^8 / a^2 = a^6 :=
by sorry

end NUMINAMATH_CALUDE_power_division_l1854_185432


namespace NUMINAMATH_CALUDE_regions_for_99_lines_l1854_185423

/-- The number of regions formed by a given number of lines in a plane -/
def num_regions (num_lines : ℕ) : Set ℕ :=
  {n | ∃ (configuration : Type) (f : configuration → ℕ), 
       (∀ c, f c ≤ (num_lines * (num_lines - 1)) / 2 + num_lines + 1) ∧
       (∃ c, f c = n)}

/-- Theorem stating that for 99 lines, the only possible numbers of regions less than 199 are 100 and 198 -/
theorem regions_for_99_lines :
  num_regions 99 ∩ {n | n < 199} = {100, 198} :=
by sorry

end NUMINAMATH_CALUDE_regions_for_99_lines_l1854_185423


namespace NUMINAMATH_CALUDE_existence_of_index_l1854_185408

theorem existence_of_index (a : Fin 7 → ℝ) (h1 : a 1 = 0) (h7 : a 7 = 0) :
  ∃ k : Fin 5, (a k) + (a (k + 2)) ≤ (a (k + 1)) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_index_l1854_185408


namespace NUMINAMATH_CALUDE_lisa_flight_time_l1854_185427

/-- Given a distance of 256 miles and a speed of 32 miles per hour, 
    the time taken to travel this distance is 8 hours. -/
theorem lisa_flight_time : 
  ∀ (distance speed time : ℝ), 
    distance = 256 → 
    speed = 32 → 
    time = distance / speed → 
    time = 8 := by sorry

end NUMINAMATH_CALUDE_lisa_flight_time_l1854_185427


namespace NUMINAMATH_CALUDE_basketball_shot_probability_l1854_185419

theorem basketball_shot_probability (a b c : ℝ) : 
  a ∈ (Set.Ioo 0 1) → 
  b ∈ (Set.Ioo 0 1) → 
  c ∈ (Set.Ioo 0 1) → 
  3 * a + 2 * b = 1 → 
  a * b ≤ 1 / 24 := by
sorry

end NUMINAMATH_CALUDE_basketball_shot_probability_l1854_185419


namespace NUMINAMATH_CALUDE_kellys_supplies_l1854_185456

/-- Calculates the number of supplies left after Kelly's art supply shopping adventure. -/
theorem kellys_supplies (students : ℕ) (paper_per_student : ℕ) (glue_bottles : ℕ) (additional_paper : ℕ) : 
  students = 8 →
  paper_per_student = 3 →
  glue_bottles = 6 →
  additional_paper = 5 →
  ((students * paper_per_student + glue_bottles) / 2 + additional_paper : ℕ) = 20 := by
sorry

end NUMINAMATH_CALUDE_kellys_supplies_l1854_185456


namespace NUMINAMATH_CALUDE_correct_algebraic_notation_l1854_185436

/-- Predicate to check if an expression follows algebraic notation rules -/
def follows_algebraic_notation (expr : String) : Prop :=
  match expr with
  | "7/3 * x^2" => True
  | "a * 1/4" => False
  | "-2 1/6 * p" => False
  | "2y / z" => False
  | _ => False

/-- Theorem stating that 7/3 * x^2 follows algebraic notation rules -/
theorem correct_algebraic_notation :
  follows_algebraic_notation "7/3 * x^2" ∧
  ¬follows_algebraic_notation "a * 1/4" ∧
  ¬follows_algebraic_notation "-2 1/6 * p" ∧
  ¬follows_algebraic_notation "2y / z" :=
sorry

end NUMINAMATH_CALUDE_correct_algebraic_notation_l1854_185436


namespace NUMINAMATH_CALUDE_log_10_50_between_consecutive_integers_sum_l1854_185462

theorem log_10_50_between_consecutive_integers_sum :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < b ∧ a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_log_10_50_between_consecutive_integers_sum_l1854_185462


namespace NUMINAMATH_CALUDE_intersection_sum_l1854_185484

/-- Given two lines that intersect at (2,1), prove that a + b = 5/3 -/
theorem intersection_sum (a b : ℝ) : 
  (2 = (1/3) * 1 + a) →  -- First line equation at (2,1)
  (1 = (1/2) * 2 + b) →  -- Second line equation at (2,1)
  a + b = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1854_185484


namespace NUMINAMATH_CALUDE_no_common_sale_days_l1854_185400

def bookstore_sales : Set Nat :=
  {d | d ≤ 31 ∧ ∃ k, d = 4 * k}

def shoe_store_sales : Set Nat :=
  {d | d ≤ 31 ∧ ∃ n, d = 2 + 8 * n}

theorem no_common_sale_days : bookstore_sales ∩ shoe_store_sales = ∅ := by
  sorry

end NUMINAMATH_CALUDE_no_common_sale_days_l1854_185400


namespace NUMINAMATH_CALUDE_line_points_k_value_l1854_185494

/-- Given a line containing the points (0, 4), (7, k), and (21, -2), prove that k = 2 -/
theorem line_points_k_value (k : ℝ) : 
  (∀ t : ℝ, ∃ x y : ℝ, x = t * 7 ∧ y = t * (k - 4) + 4) → 
  (∃ t : ℝ, 21 = t * 7 ∧ -2 = t * (k - 4) + 4) → 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_points_k_value_l1854_185494


namespace NUMINAMATH_CALUDE_function_inequality_l1854_185430

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 1, (x - 1) * (deriv f x) - f x > 0) :
  (1 / (Real.sqrt 2 - 1)) * f (Real.sqrt 2) < f 2 ∧ f 2 < (1 / 2) * f 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1854_185430


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l1854_185426

theorem sufficient_condition_range (a x : ℝ) : 
  (∀ x, (a ≤ x ∧ x < a + 2) → x ≤ -1) ∧ 
  (∃ x, x ≤ -1 ∧ ¬(a ≤ x ∧ x < a + 2)) →
  a ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l1854_185426


namespace NUMINAMATH_CALUDE_campbell_geometry_qualification_l1854_185446

/-- Represents the minimum score required in the 4th quarter to achieve a given average -/
def min_fourth_quarter_score (q1 q2 q3 : ℚ) (required_avg : ℚ) : ℚ :=
  4 * required_avg - (q1 + q2 + q3)

/-- Theorem: Given Campbell's scores and the required average, the minimum 4th quarter score is 107% -/
theorem campbell_geometry_qualification (campbell_q1 campbell_q2 campbell_q3 : ℚ)
  (h1 : campbell_q1 = 84/100)
  (h2 : campbell_q2 = 79/100)
  (h3 : campbell_q3 = 70/100)
  (required_avg : ℚ)
  (h4 : required_avg = 85/100) :
  min_fourth_quarter_score campbell_q1 campbell_q2 campbell_q3 required_avg = 107/100 := by
sorry

#eval min_fourth_quarter_score (84/100) (79/100) (70/100) (85/100)

end NUMINAMATH_CALUDE_campbell_geometry_qualification_l1854_185446


namespace NUMINAMATH_CALUDE_greatest_possible_average_speed_l1854_185477

def is_palindrome (n : ℕ) : Prop := sorry

def initial_reading : ℕ := 12321
def drive_duration : ℝ := 4
def speed_limit : ℝ := 80
def min_average_speed : ℝ := 60

theorem greatest_possible_average_speed :
  ∀ (final_reading : ℕ),
    is_palindrome initial_reading →
    is_palindrome final_reading →
    final_reading > initial_reading →
    (final_reading - initial_reading : ℝ) / drive_duration > min_average_speed →
    (final_reading - initial_reading : ℝ) / drive_duration ≤ speed_limit →
    (∀ (other_reading : ℕ),
      is_palindrome other_reading →
      other_reading > initial_reading →
      (other_reading - initial_reading : ℝ) / drive_duration > min_average_speed →
      (other_reading - initial_reading : ℝ) / drive_duration ≤ speed_limit →
      (other_reading - initial_reading : ℝ) / drive_duration ≤ (final_reading - initial_reading : ℝ) / drive_duration) →
    (final_reading - initial_reading : ℝ) / drive_duration = 75 :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_average_speed_l1854_185477


namespace NUMINAMATH_CALUDE_tangent_line_condition_l1854_185437

/-- Given a function f(x) = e^x + a*cos(x), if its tangent line at x = 0 passes through (1, 6), then a = 4 -/
theorem tangent_line_condition (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.exp x + a * Real.cos x
  let f' : ℝ → ℝ := λ x ↦ Real.exp x - a * Real.sin x
  (f 0 = 6) ∧ (f' 0 = 1) → a = 4 := by sorry

end NUMINAMATH_CALUDE_tangent_line_condition_l1854_185437


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1854_185451

theorem sum_of_fractions : 
  (1 : ℚ) / 3 + (1 : ℚ) / 2 + (-5 : ℚ) / 6 + (1 : ℚ) / 5 + (1 : ℚ) / 4 + (-9 : ℚ) / 20 + (-5 : ℚ) / 6 = (-5 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1854_185451


namespace NUMINAMATH_CALUDE_bird_wings_problem_l1854_185473

theorem bird_wings_problem :
  ∃! (x y z : ℕ), 2 * x + 4 * y + 3 * z = 70 ∧ x = 2 * y := by
  sorry

end NUMINAMATH_CALUDE_bird_wings_problem_l1854_185473


namespace NUMINAMATH_CALUDE_inequality_implies_not_equal_l1854_185431

theorem inequality_implies_not_equal (a b : ℝ) :
  (a / b + b / a > 2) → (a ≠ b) ∧ ¬(∀ a b : ℝ, a ≠ b → a / b + b / a > 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_implies_not_equal_l1854_185431


namespace NUMINAMATH_CALUDE_sum_of_first_and_third_l1854_185478

theorem sum_of_first_and_third (A B C : ℚ) : 
  A + B + C = 98 →
  A / B = 2 / 3 →
  B / C = 5 / 8 →
  B = 30 →
  A + C = 68 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_and_third_l1854_185478


namespace NUMINAMATH_CALUDE_bowling_average_problem_bowling_average_proof_l1854_185488

theorem bowling_average_problem (initial_average : ℝ) (wickets_last_match : ℕ) 
  (average_decrease : ℝ) (previous_wickets : ℕ) : ℝ :=
  let final_average := initial_average - average_decrease
  let total_wickets := previous_wickets + wickets_last_match
  let previous_runs := initial_average * previous_wickets
  let total_runs := final_average * total_wickets
  total_runs - previous_runs

#check bowling_average_problem 12.4 8 0.4 175 = 26

-- The proof is omitted
theorem bowling_average_proof : 
  bowling_average_problem 12.4 8 0.4 175 = 26 := by sorry

end NUMINAMATH_CALUDE_bowling_average_problem_bowling_average_proof_l1854_185488


namespace NUMINAMATH_CALUDE_percentage_8_years_plus_is_24_percent_l1854_185435

/-- Represents the number of employees for each year range --/
structure EmployeeDistribution :=
  (less_than_2 : ℕ)
  (from_2_to_4 : ℕ)
  (from_4_to_6 : ℕ)
  (from_6_to_8 : ℕ)
  (from_8_to_10 : ℕ)
  (from_10_to_12 : ℕ)
  (from_12_to_14 : ℕ)

/-- Calculates the total number of employees --/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_2 + d.from_2_to_4 + d.from_4_to_6 + d.from_6_to_8 +
  d.from_8_to_10 + d.from_10_to_12 + d.from_12_to_14

/-- Calculates the number of employees with 8 or more years of employment --/
def employees_8_years_plus (d : EmployeeDistribution) : ℕ :=
  d.from_8_to_10 + d.from_10_to_12 + d.from_12_to_14

/-- Calculates the percentage of employees with 8 or more years of employment --/
def percentage_8_years_plus (d : EmployeeDistribution) : ℚ :=
  (employees_8_years_plus d : ℚ) / (total_employees d : ℚ) * 100

/-- Theorem stating that the percentage of employees with 8 or more years of employment is 24% --/
theorem percentage_8_years_plus_is_24_percent (d : EmployeeDistribution)
  (h1 : d.less_than_2 = 4)
  (h2 : d.from_2_to_4 = 6)
  (h3 : d.from_4_to_6 = 5)
  (h4 : d.from_6_to_8 = 4)
  (h5 : d.from_8_to_10 = 3)
  (h6 : d.from_10_to_12 = 2)
  (h7 : d.from_12_to_14 = 1) :
  percentage_8_years_plus d = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_8_years_plus_is_24_percent_l1854_185435


namespace NUMINAMATH_CALUDE_common_difference_is_two_l1854_185401

/-- An arithmetic sequence with sum S_n of first n terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

/-- The common difference of an arithmetic sequence is 2 given the condition -/
theorem common_difference_is_two (seq : ArithmeticSequence) 
    (h : seq.S 5 / 5 - seq.S 2 / 2 = 3) : 
    seq.a 2 - seq.a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_two_l1854_185401


namespace NUMINAMATH_CALUDE_rectangle_area_is_eight_l1854_185471

/-- A square with side length 4 containing two right triangles whose hypotenuses
    are opposite sides of the square. -/
structure SquareWithTriangles where
  side_length : ℝ
  hypotenuse_length : ℝ
  rectangle_width : ℝ
  rectangle_height : ℝ
  h_side_length : side_length = 4
  h_hypotenuse : hypotenuse_length = side_length
  h_right_triangle : rectangle_width ^ 2 + rectangle_height ^ 2 = hypotenuse_length ^ 2
  h_rectangle_dim : rectangle_width + rectangle_height = side_length

/-- The area of the rectangle formed by the intersection of the triangles is 8. -/
theorem rectangle_area_is_eight (s : SquareWithTriangles) : 
  s.rectangle_width * s.rectangle_height = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_is_eight_l1854_185471


namespace NUMINAMATH_CALUDE_a5_value_l1854_185403

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)

theorem a5_value (a : ℕ → ℝ) :
  geometric_sequence a →
  a 3 + a 7 = 3 →
  a 3 * a 7 = 2 →
  a 5 = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_a5_value_l1854_185403


namespace NUMINAMATH_CALUDE_a_3_value_l1854_185474

def a (n : ℕ) : ℚ := (-1)^n * (n : ℚ) / (n + 1)

theorem a_3_value : a 3 = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_a_3_value_l1854_185474


namespace NUMINAMATH_CALUDE_winnie_fell_behind_l1854_185492

/-- The number of repetitions Winnie fell behind --/
def repetitions_fell_behind (yesterday_reps today_reps : ℕ) : ℕ :=
  yesterday_reps - today_reps

/-- Proof that Winnie fell behind by 13 repetitions --/
theorem winnie_fell_behind :
  repetitions_fell_behind 86 73 = 13 := by
  sorry

end NUMINAMATH_CALUDE_winnie_fell_behind_l1854_185492


namespace NUMINAMATH_CALUDE_eliza_ironed_17_clothes_l1854_185498

/-- Calculates the total number of clothes Eliza ironed given the time spent and ironing rates. -/
def total_clothes_ironed (blouse_time : ℕ) (dress_time : ℕ) (blouse_hours : ℕ) (dress_hours : ℕ) : ℕ :=
  let blouses := (blouse_hours * 60) / blouse_time
  let dresses := (dress_hours * 60) / dress_time
  blouses + dresses

/-- Proves that Eliza ironed 17 pieces of clothes given the conditions. -/
theorem eliza_ironed_17_clothes :
  total_clothes_ironed 15 20 2 3 = 17 := by
  sorry

#eval total_clothes_ironed 15 20 2 3

end NUMINAMATH_CALUDE_eliza_ironed_17_clothes_l1854_185498


namespace NUMINAMATH_CALUDE_product_equals_29_l1854_185438

theorem product_equals_29 (a : ℝ) (h : a^2 + a + 1 = 2) : (5 - a) * (6 + a) = 29 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_29_l1854_185438


namespace NUMINAMATH_CALUDE_xias_initial_sticker_count_l1854_185472

/-- Theorem: Xia's initial sticker count
Given that Xia shared 100 stickers with her friends, had 5 sheets of stickers left,
and each sheet contains 10 stickers, prove that she initially had 150 stickers. -/
theorem xias_initial_sticker_count
  (shared_stickers : ℕ)
  (remaining_sheets : ℕ)
  (stickers_per_sheet : ℕ)
  (h1 : shared_stickers = 100)
  (h2 : remaining_sheets = 5)
  (h3 : stickers_per_sheet = 10) :
  shared_stickers + remaining_sheets * stickers_per_sheet = 150 :=
by sorry

end NUMINAMATH_CALUDE_xias_initial_sticker_count_l1854_185472


namespace NUMINAMATH_CALUDE_count_two_repeating_digits_l1854_185469

/-- A four-digit number is a natural number between 1000 and 9999 inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that counts the occurrences of each digit in a four-digit number. -/
def DigitCount (n : ℕ) : ℕ → ℕ := sorry

/-- A four-digit number has exactly two repeating digits if exactly one digit appears twice
    and the other two digits appear once each. -/
def HasExactlyTwoRepeatingDigits (n : ℕ) : Prop :=
  FourDigitNumber n ∧ ∃! d : ℕ, DigitCount n d = 2

/-- The count of four-digit numbers with exactly two repeating digits. -/
def CountTwoRepeatingDigits : ℕ := sorry

/-- The main theorem stating that the count of four-digit numbers with exactly two repeating digits is 2736. -/
theorem count_two_repeating_digits :
  CountTwoRepeatingDigits = 2736 := by sorry

end NUMINAMATH_CALUDE_count_two_repeating_digits_l1854_185469


namespace NUMINAMATH_CALUDE_inequality_proof_l1854_185480

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1854_185480


namespace NUMINAMATH_CALUDE_pipe_ratio_l1854_185402

theorem pipe_ratio (total_length shorter_length : ℕ) 
  (h1 : total_length = 177)
  (h2 : shorter_length = 59)
  (h3 : shorter_length < total_length) :
  (total_length - shorter_length) / shorter_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_pipe_ratio_l1854_185402


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1854_185420

-- Define the arithmetic sequence
def arithmetic_sequence (d : ℝ) (n : ℕ) : ℝ := 1 + (n - 1) * d

-- Define the theorem
theorem arithmetic_sequence_difference
  (d : ℝ) (m n : ℕ) 
  (h1 : d ≠ 0)
  (h2 : arithmetic_sequence d 2 * arithmetic_sequence d 6 = (arithmetic_sequence d 4 - 2)^2)
  (h3 : m > n)
  (h4 : m - n = 10) :
  arithmetic_sequence d m - arithmetic_sequence d n = 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1854_185420


namespace NUMINAMATH_CALUDE_intersection_area_of_specific_rectangles_l1854_185412

/-- Represents a rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ
  angle : ℝ  -- Angle of rotation in radians

/-- Calculates the area of intersection between two rectangles -/
noncomputable def intersectionArea (r1 r2 : Rectangle) : ℝ :=
  sorry

/-- Theorem stating the area of intersection between two specific rectangles -/
theorem intersection_area_of_specific_rectangles :
  let r1 : Rectangle := { width := 4, height := 12, angle := 0 }
  let r2 : Rectangle := { width := 5, height := 10, angle := π/6 }
  intersectionArea r1 r2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_of_specific_rectangles_l1854_185412


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l1854_185444

/-- The line equation as a function of k, x, and y -/
def line_equation (k x y : ℝ) : ℝ := (2*k + 1)*x + (1 - k)*y + 7 - k

/-- The theorem stating that (-2, -5) is a fixed point of the line for all real k -/
theorem fixed_point_theorem :
  ∀ k : ℝ, line_equation k (-2) (-5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l1854_185444


namespace NUMINAMATH_CALUDE_contrapositive_correct_l1854_185450

-- Define a triangle
structure Triangle :=
  (A B C : Point)

-- Define the property of being an isosceles triangle
def isIsosceles (t : Triangle) : Prop := sorry

-- Define the property of having two equal interior angles
def hasTwoEqualAngles (t : Triangle) : Prop := sorry

-- The original statement
def originalStatement (t : Triangle) : Prop :=
  ¬(isIsosceles t) → ¬(hasTwoEqualAngles t)

-- The contrapositive of the original statement
def contrapositive (t : Triangle) : Prop :=
  hasTwoEqualAngles t → isIsosceles t

-- Theorem stating that the contrapositive is correct
theorem contrapositive_correct :
  ∀ t : Triangle, originalStatement t ↔ contrapositive t :=
sorry

end NUMINAMATH_CALUDE_contrapositive_correct_l1854_185450


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l1854_185404

theorem circle_diameter_ratio (C D : ℝ → Prop) (rC rD : ℝ) : 
  (∀ x, C x → D x) →  -- C is within D
  rD = 10 →  -- Diameter of D is 20 cm
  (π * (rD^2 - rC^2)) / (π * rC^2) = 2 →  -- Ratio of shaded area to area of C is 2:1
  2 * rC = 20 * Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l1854_185404


namespace NUMINAMATH_CALUDE_roxanne_sandwiches_l1854_185410

theorem roxanne_sandwiches (lemonade_price : ℚ) (sandwich_price : ℚ) 
  (lemonade_count : ℕ) (paid : ℚ) (change : ℚ) :
  lemonade_price = 2 →
  sandwich_price = 5/2 →
  lemonade_count = 2 →
  paid = 20 →
  change = 11 →
  (paid - change - lemonade_price * lemonade_count) / sandwich_price = 2 :=
by sorry

end NUMINAMATH_CALUDE_roxanne_sandwiches_l1854_185410


namespace NUMINAMATH_CALUDE_duck_cow_problem_l1854_185424

/-- Proves that in a group of ducks and cows, if the total number of legs is 32 more than twice the number of heads, then the number of cows is 16 -/
theorem duck_cow_problem (ducks cows : ℕ) : 
  2 * (ducks + cows) + 32 = 2 * ducks + 4 * cows → cows = 16 := by
  sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l1854_185424


namespace NUMINAMATH_CALUDE_inverse_g_84_l1854_185455

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- State the theorem
theorem inverse_g_84 : g⁻¹ 84 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_g_84_l1854_185455


namespace NUMINAMATH_CALUDE_candy_distribution_l1854_185468

theorem candy_distribution (n : ℕ) (h : n = 30) :
  (min (n % 4) ((4 - n % 4) % 4)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1854_185468


namespace NUMINAMATH_CALUDE_power_equation_solution_l1854_185433

theorem power_equation_solution :
  ∃ x : ℤ, (3 : ℝ)^7 * (3 : ℝ)^x = 81 ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1854_185433


namespace NUMINAMATH_CALUDE_scientific_notation_of_238_billion_l1854_185495

/-- A billion is defined as 10^9 -/
def billion : ℕ := 10^9

/-- The problem statement -/
theorem scientific_notation_of_238_billion :
  (238 : ℝ) * billion = 2.38 * (10 : ℝ)^10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_238_billion_l1854_185495


namespace NUMINAMATH_CALUDE_det_E_equals_25_l1854_185442

/-- A 2x2 matrix representing a dilation by factor 5 centered at the origin -/
def D : Matrix (Fin 2) (Fin 2) ℝ := !![5, 0; 0, 5]

/-- A 2x2 matrix representing a 90-degree counter-clockwise rotation -/
def R : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- The combined transformation matrix E -/
def E : Matrix (Fin 2) (Fin 2) ℝ := R * D

theorem det_E_equals_25 : Matrix.det E = 25 := by
  sorry

end NUMINAMATH_CALUDE_det_E_equals_25_l1854_185442


namespace NUMINAMATH_CALUDE_more_customers_left_than_stayed_l1854_185464

theorem more_customers_left_than_stayed (initial_customers remaining_customers : ℕ) :
  initial_customers = 25 →
  remaining_customers = 7 →
  (initial_customers - remaining_customers) - remaining_customers = 11 := by
  sorry

end NUMINAMATH_CALUDE_more_customers_left_than_stayed_l1854_185464


namespace NUMINAMATH_CALUDE_only_345_is_right_triangle_l1854_185413

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2)

/-- The theorem stating that among the given sets, only (3,4,5) forms a right triangle -/
theorem only_345_is_right_triangle :
  ¬(is_right_triangle 2 3 4) ∧
  (is_right_triangle 3 4 5) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 6 7) :=
by sorry

end NUMINAMATH_CALUDE_only_345_is_right_triangle_l1854_185413


namespace NUMINAMATH_CALUDE_division_problem_l1854_185490

theorem division_problem (L S q : ℕ) : 
  L - S = 1325 → 
  L = 1650 → 
  L = S * q + 5 → 
  q = 5 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1854_185490


namespace NUMINAMATH_CALUDE_movie_spending_ratio_l1854_185479

/-- Proves that the ratio of movie spending to weekly allowance is 1:2 --/
theorem movie_spending_ratio (weekly_allowance car_wash_earnings final_amount : ℕ) :
  weekly_allowance = 8 →
  car_wash_earnings = 8 →
  final_amount = 12 →
  (weekly_allowance + car_wash_earnings - final_amount) / weekly_allowance = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_movie_spending_ratio_l1854_185479


namespace NUMINAMATH_CALUDE_charlies_share_l1854_185441

/-- Represents the share of money each person receives -/
structure Share where
  alice : ℚ
  bond : ℚ
  charlie : ℚ

/-- The conditions of the problem -/
def satisfiesConditions (s : Share) : Prop :=
  s.alice + s.bond + s.charlie = 1105 ∧
  (s.alice - 10) / (s.bond - 20) = 11 / 18 ∧
  (s.alice - 10) / (s.charlie - 15) = 11 / 24

/-- The theorem stating Charlie's share -/
theorem charlies_share :
  ∃ (s : Share), satisfiesConditions s ∧ s.charlie = 495 := by
  sorry


end NUMINAMATH_CALUDE_charlies_share_l1854_185441


namespace NUMINAMATH_CALUDE_function_periodicity_l1854_185459

/-- A function f: ℝ → ℝ satisfying the given property is periodic with period 2a -/
theorem function_periodicity (f : ℝ → ℝ) (a : ℝ) (h_a : a > 0) 
  (h_f : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)) :
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l1854_185459


namespace NUMINAMATH_CALUDE_total_amount_shared_l1854_185476

/-- Represents the amount of money received by each person -/
structure ShareDistribution where
  john : ℕ
  jose : ℕ
  binoy : ℕ

/-- Defines the ratio of money distribution -/
def ratio : Fin 3 → ℕ
  | 0 => 2
  | 1 => 4
  | 2 => 6

/-- Proves that the total amount shared is 12000 given the conditions -/
theorem total_amount_shared (d : ShareDistribution) 
  (h1 : d.john = 2000)
  (h2 : d.jose = 2 * d.john)
  (h3 : d.binoy = 3 * d.john) : 
  d.john + d.jose + d.binoy = 12000 := by
  sorry

#check total_amount_shared

end NUMINAMATH_CALUDE_total_amount_shared_l1854_185476


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1854_185448

theorem arithmetic_calculation : 5 * 7 + 6 * 9 + 13 * 2 + 4 * 6 = 139 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1854_185448


namespace NUMINAMATH_CALUDE_executive_committee_selection_l1854_185409

theorem executive_committee_selection (total_members : ℕ) (committee_size : ℕ) (ineligible_members : ℕ) :
  total_members = 30 →
  committee_size = 5 →
  ineligible_members = 4 →
  Nat.choose (total_members - ineligible_members) committee_size = 60770 := by
sorry

end NUMINAMATH_CALUDE_executive_committee_selection_l1854_185409


namespace NUMINAMATH_CALUDE_range_of_m_l1854_185429

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc 0 1, x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1854_185429


namespace NUMINAMATH_CALUDE_inequality_cube_l1854_185428

theorem inequality_cube (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (a - c)^3 > (b - c)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_cube_l1854_185428


namespace NUMINAMATH_CALUDE_population_growth_rate_l1854_185449

/-- The time it takes for one person to be added to the population, given the rate of population increase. -/
def time_per_person (persons_per_hour : ℕ) : ℚ :=
  (60 * 60) / persons_per_hour

/-- Theorem stating that the time it takes for one person to be added to the population is 15 seconds, 
    given that the population increases by 240 persons in 60 minutes. -/
theorem population_growth_rate : time_per_person 240 = 15 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_l1854_185449


namespace NUMINAMATH_CALUDE_rotation_90_degrees_l1854_185489

def rotate90(z : ℂ) : ℂ := z * Complex.I

theorem rotation_90_degrees :
  rotate90 (-4 - 2 * Complex.I) = 2 - 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_rotation_90_degrees_l1854_185489


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l1854_185434

/-- Given an incident ray y = 2x + 1 reflected by the line y = x, 
    the equation of the reflected ray is x - 2y - 1 = 0 -/
theorem reflected_ray_equation (x y : ℝ) : 
  (y = 2*x + 1) →  -- incident ray
  (y = x) →        -- reflecting line
  (x - 2*y - 1 = 0) -- reflected ray
  := by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l1854_185434


namespace NUMINAMATH_CALUDE_luna_budget_theorem_l1854_185461

/-- Luna's monthly budget problem --/
theorem luna_budget_theorem 
  (house_rental : ℝ) 
  (food : ℝ) 
  (phone : ℝ) 
  (h1 : food = 0.6 * house_rental) 
  (h2 : house_rental + food = 240) 
  (h3 : house_rental + food + phone = 249) :
  phone = 0.1 * food := by
  sorry

end NUMINAMATH_CALUDE_luna_budget_theorem_l1854_185461


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l1854_185452

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The lengths of the four sides of the quadrilateral -/
  sides : Fin 4 → ℝ

/-- The theorem stating the property of the inscribed quadrilateral -/
theorem inscribed_quadrilateral_fourth_side
  (q : InscribedQuadrilateral)
  (h_radius : q.radius = 100 * Real.sqrt 3)
  (h_side1 : q.sides 0 = 100)
  (h_side2 : q.sides 1 = 200)
  (h_side3 : q.sides 2 = 300) :
  q.sides 3 = 450 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_fourth_side_l1854_185452


namespace NUMINAMATH_CALUDE_sum_of_factors_30_l1854_185418

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem sum_of_factors_30 : (factors 30).sum id = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_30_l1854_185418


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1854_185466

/-- Function f(x) = ax² + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

/-- Condition: a ≥ 4 or a ≤ 0 -/
def condition (a : ℝ) : Prop := a ≥ 4 ∨ a ≤ 0

/-- f has zero points -/
def has_zero_points (a : ℝ) : Prop := ∃ x : ℝ, f a x = 0

theorem condition_necessary_not_sufficient :
  (∀ a : ℝ, has_zero_points a → condition a) ∧
  (∃ a : ℝ, condition a ∧ ¬has_zero_points a) :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1854_185466


namespace NUMINAMATH_CALUDE_tan_neg_x_domain_l1854_185422

theorem tan_neg_x_domain :
  {x : ℝ | ∀ n : ℤ, x ≠ -π/2 + n*π} = {x : ℝ | ∃ y : ℝ, y = Real.tan (-x)} :=
by sorry

end NUMINAMATH_CALUDE_tan_neg_x_domain_l1854_185422


namespace NUMINAMATH_CALUDE_profit_range_l1854_185443

/-- Price function for books -/
def C (n : ℕ) : ℕ :=
  if n ≤ 24 then 12 * n
  else if n ≤ 48 then 11 * n
  else 10 * n

/-- Total number of books -/
def total_books : ℕ := 60

/-- Cost per book to the company -/
def cost_per_book : ℕ := 5

/-- Profit function given two people buying books -/
def profit (a b : ℕ) : ℤ :=
  (C a + C b) - (cost_per_book * total_books)

/-- Theorem stating the range of profit -/
theorem profit_range :
  ∀ a b : ℕ,
  a + b = total_books →
  a ≥ 1 →
  b ≥ 1 →
  302 ≤ profit a b ∧ profit a b ≤ 384 :=
sorry

end NUMINAMATH_CALUDE_profit_range_l1854_185443


namespace NUMINAMATH_CALUDE_both_pass_through_origin_l1854_185487

/-- Parabola passing through (0,1) -/
def passes_through_origin (f : ℝ → ℝ) : Prop :=
  f 0 = 1

/-- First parabola -/
def f₁ (x : ℝ) : ℝ := -x^2 + 1

/-- Second parabola -/
def f₂ (x : ℝ) : ℝ := x^2 + 1

/-- Theorem: Both parabolas pass through (0,1) -/
theorem both_pass_through_origin :
  passes_through_origin f₁ ∧ passes_through_origin f₂ := by
  sorry

end NUMINAMATH_CALUDE_both_pass_through_origin_l1854_185487


namespace NUMINAMATH_CALUDE_yard_length_24_trees_18m_spacing_l1854_185481

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_spacing : ℝ) : ℝ :=
  (num_trees - 1) * tree_spacing

/-- Theorem: The length of a yard with 24 trees planted at equal distances,
    with one tree at each end and 18 meters between consecutive trees, is 414 meters. -/
theorem yard_length_24_trees_18m_spacing :
  yard_length 24 18 = 414 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_24_trees_18m_spacing_l1854_185481


namespace NUMINAMATH_CALUDE_system_solution_exists_l1854_185467

theorem system_solution_exists (b : ℝ) : 
  (∃ (a x y : ℝ), 
    x = 7 / b - abs (y + b) ∧ 
    x^2 + y^2 + 96 = -a * (2 * y + a) - 20 * x) ↔ 
  (b ≤ -7/12 ∨ b > 0) :=
sorry

end NUMINAMATH_CALUDE_system_solution_exists_l1854_185467


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l1854_185493

/-- Proves that the remainder yards after 15 marathons is 1500 --/
theorem marathon_remainder_yards :
  let marathons : ℕ := 15
  let miles_per_marathon : ℕ := 26
  let yards_per_marathon : ℕ := 385
  let yards_per_mile : ℕ := 1760
  let total_yards := marathons * (miles_per_marathon * yards_per_mile + yards_per_marathon)
  let full_miles := total_yards / yards_per_mile
  let remainder_yards := total_yards % yards_per_mile
  remainder_yards = 1500 := by sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l1854_185493


namespace NUMINAMATH_CALUDE_sticker_difference_l1854_185425

theorem sticker_difference (karl_stickers : ℕ) (ryan_more_than_karl : ℕ) (total_stickers : ℕ)
  (h1 : karl_stickers = 25)
  (h2 : ryan_more_than_karl = 20)
  (h3 : total_stickers = 105) :
  let ryan_stickers := karl_stickers + ryan_more_than_karl
  let ben_stickers := total_stickers - karl_stickers - ryan_stickers
  ryan_stickers - ben_stickers = 10 := by
sorry

end NUMINAMATH_CALUDE_sticker_difference_l1854_185425


namespace NUMINAMATH_CALUDE_max_profit_theorem_l1854_185486

def profit_A (x : ℕ) : ℚ := 5.06 * x - 0.15 * x^2
def profit_B (x : ℕ) : ℚ := 2 * x

theorem max_profit_theorem :
  ∃ (x : ℕ), x ≤ 15 ∧ 
  (∀ (y : ℕ), y ≤ 15 → 
    profit_A x + profit_B (15 - x) ≥ profit_A y + profit_B (15 - y)) ∧
  profit_A x + profit_B (15 - x) = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l1854_185486


namespace NUMINAMATH_CALUDE_bomb_guaranteed_four_of_a_kind_guaranteed_l1854_185485

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (ranks : ℕ)

/-- Represents the minimum number of cards to draw to ensure a "bomb" -/
def min_cards_for_bomb (d : Deck) : ℕ := d.ranks * (d.suits - 1) + 1

/-- Theorem: Drawing 40 cards from a standard deck guarantees a "bomb" -/
theorem bomb_guaranteed (d : Deck) 
  (h1 : d.total_cards = 52) 
  (h2 : d.suits = 4) 
  (h3 : d.ranks = 13) : 
  min_cards_for_bomb d = 40 := by
sorry

/-- Corollary: Drawing 40 cards guarantees at least four cards of the same rank -/
theorem four_of_a_kind_guaranteed (d : Deck) 
  (h1 : d.total_cards = 52) 
  (h2 : d.suits = 4) 
  (h3 : d.ranks = 13) : 
  ∃ (n : ℕ), n ≤ 40 ∧ (∀ (m : ℕ), m ≥ n → ∃ (r : ℕ), r ≤ d.ranks ∧ 4 ≤ m - (d.ranks - 1) * 3) := by
sorry

end NUMINAMATH_CALUDE_bomb_guaranteed_four_of_a_kind_guaranteed_l1854_185485


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1854_185407

def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1854_185407


namespace NUMINAMATH_CALUDE_mike_afternoon_seeds_l1854_185491

/-- Represents the number of tomato seeds planted by Mike and Ted -/
structure TomatoSeeds where
  mike_morning : ℕ
  ted_morning : ℕ
  mike_afternoon : ℕ
  ted_afternoon : ℕ

/-- The conditions of the tomato planting problem -/
def tomato_planting_conditions (s : TomatoSeeds) : Prop :=
  s.mike_morning = 50 ∧
  s.ted_morning = 2 * s.mike_morning ∧
  s.ted_afternoon = s.mike_afternoon - 20 ∧
  s.mike_morning + s.ted_morning + s.mike_afternoon + s.ted_afternoon = 250

/-- Theorem stating that under the given conditions, Mike planted 60 tomato seeds in the afternoon -/
theorem mike_afternoon_seeds (s : TomatoSeeds) 
  (h : tomato_planting_conditions s) : s.mike_afternoon = 60 := by
  sorry

end NUMINAMATH_CALUDE_mike_afternoon_seeds_l1854_185491
