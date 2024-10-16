import Mathlib

namespace NUMINAMATH_CALUDE_tyler_meal_combinations_correct_l3058_305823

/-- The number of different meal combinations Tyler can choose at a buffet. -/
def tyler_meal_combinations : ℕ := 150

/-- The number of meat options available. -/
def meat_options : ℕ := 3

/-- The number of vegetable options available. -/
def vegetable_options : ℕ := 5

/-- The number of vegetables Tyler must choose. -/
def vegetables_to_choose : ℕ := 3

/-- The number of dessert options available. -/
def dessert_options : ℕ := 5

/-- Theorem stating that the number of meal combinations Tyler can choose is correct. -/
theorem tyler_meal_combinations_correct :
  tyler_meal_combinations = meat_options * (Nat.choose vegetable_options vegetables_to_choose) * dessert_options :=
by sorry

end NUMINAMATH_CALUDE_tyler_meal_combinations_correct_l3058_305823


namespace NUMINAMATH_CALUDE_average_lifespan_of_sampled_products_l3058_305834

/-- Represents a factory producing electronic products -/
structure Factory where
  production_ratio : ℚ
  average_lifespan : ℚ

/-- Calculates the weighted average lifespan of products from multiple factories -/
def weighted_average_lifespan (factories : List Factory) (total_samples : ℕ) : ℚ :=
  let total_ratio := factories.map (λ f => f.production_ratio) |>.sum
  let weighted_sum := factories.map (λ f => f.production_ratio * f.average_lifespan) |>.sum
  weighted_sum / total_ratio

/-- The main theorem proving the average lifespan of sampled products -/
theorem average_lifespan_of_sampled_products : 
  let factories := [
    { production_ratio := 1, average_lifespan := 980 },
    { production_ratio := 2, average_lifespan := 1020 },
    { production_ratio := 1, average_lifespan := 1032 }
  ]
  let total_samples := 100
  weighted_average_lifespan factories total_samples = 1013 := by
  sorry

end NUMINAMATH_CALUDE_average_lifespan_of_sampled_products_l3058_305834


namespace NUMINAMATH_CALUDE_minimal_leasing_cost_l3058_305859

/-- Represents the daily production and cost of equipment types -/
structure EquipmentType where
  productA : ℕ
  productB : ℕ
  cost : ℕ

/-- Represents the company's production requirements -/
structure Requirements where
  minProductA : ℕ
  minProductB : ℕ

/-- Calculates the total production and cost for a given number of days of each equipment type -/
def calculateProduction (typeA : EquipmentType) (typeB : EquipmentType) (daysA : ℕ) (daysB : ℕ) : ℕ × ℕ × ℕ :=
  (daysA * typeA.productA + daysB * typeB.productA,
   daysA * typeB.productB + daysB * typeB.productB,
   daysA * typeA.cost + daysB * typeB.cost)

/-- Checks if the production meets the requirements -/
def meetsRequirements (prod : ℕ × ℕ × ℕ) (req : Requirements) : Prop :=
  prod.1 ≥ req.minProductA ∧ prod.2.1 ≥ req.minProductB

/-- Theorem stating that the minimal leasing cost is 2000 yuan -/
theorem minimal_leasing_cost 
  (typeA : EquipmentType)
  (typeB : EquipmentType)
  (req : Requirements)
  (h1 : typeA.productA = 5)
  (h2 : typeA.productB = 10)
  (h3 : typeA.cost = 200)
  (h4 : typeB.productA = 6)
  (h5 : typeB.productB = 20)
  (h6 : typeB.cost = 300)
  (h7 : req.minProductA = 50)
  (h8 : req.minProductB = 140) :
  ∃ (daysA daysB : ℕ), 
    let prod := calculateProduction typeA typeB daysA daysB
    meetsRequirements prod req ∧ 
    prod.2.2 = 2000 ∧
    (∀ (x y : ℕ), 
      let otherProd := calculateProduction typeA typeB x y
      meetsRequirements otherProd req → otherProd.2.2 ≥ 2000) := by
  sorry


end NUMINAMATH_CALUDE_minimal_leasing_cost_l3058_305859


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_192_l3058_305890

theorem sqrt_sum_equals_sqrt_192 (N : ℕ+) :
  Real.sqrt 12 + Real.sqrt 108 = Real.sqrt N.1 → N.1 = 192 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_192_l3058_305890


namespace NUMINAMATH_CALUDE_negation_implication_geometric_sequence_squared_increasing_l3058_305843

-- Proposition 3
theorem negation_implication (P Q : Prop) :
  (¬(P → Q)) ↔ (P ∧ ¬Q) :=
sorry

-- Proposition 4
theorem geometric_sequence_squared_increasing
  (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = a n * q) (h2 : q > 1) :
  ∀ n, a (n + 1)^2 > a n^2 :=
sorry

end NUMINAMATH_CALUDE_negation_implication_geometric_sequence_squared_increasing_l3058_305843


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3058_305849

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^4 + y^2 = 4*y + 4 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3058_305849


namespace NUMINAMATH_CALUDE_largest_three_digit_number_with_conditions_l3058_305831

theorem largest_three_digit_number_with_conditions : ∃ n : ℕ, 
  (n ≤ 999 ∧ n ≥ 100) ∧ 
  (∃ k : ℕ, n = 7 * k + 2) ∧ 
  (∃ m : ℕ, n = 4 * m + 1) ∧ 
  (∀ x : ℕ, (x ≤ 999 ∧ x ≥ 100) → 
    (∃ k : ℕ, x = 7 * k + 2) → 
    (∃ m : ℕ, x = 4 * m + 1) → 
    x ≤ n) ∧
  n = 989 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_number_with_conditions_l3058_305831


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3058_305850

theorem smallest_n_square_and_cube : 
  (∃ (n : ℕ), n > 0 ∧ 
   (∃ (a : ℕ), 5 * n = a ^ 2) ∧ 
   (∃ (b : ℕ), 3 * n = b ^ 3)) → 
  (∀ (m : ℕ), m > 0 → 
   (∃ (a : ℕ), 5 * m = a ^ 2) → 
   (∃ (b : ℕ), 3 * m = b ^ 3) → 
   m ≥ 1125) ∧
  (∃ (a b : ℕ), 5 * 1125 = a ^ 2 ∧ 3 * 1125 = b ^ 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3058_305850


namespace NUMINAMATH_CALUDE_pear_sales_l3058_305827

theorem pear_sales (morning_sales afternoon_sales total_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 390 →
  afternoon_sales = 260 := by
sorry

end NUMINAMATH_CALUDE_pear_sales_l3058_305827


namespace NUMINAMATH_CALUDE_circle_area_equivalence_l3058_305869

theorem circle_area_equivalence (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 33) (h₂ : r₂ = 24) : 
  (π * r₁^2 - π * r₂^2 = π * r₃^2) → r₃ = 3 * Real.sqrt 57 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equivalence_l3058_305869


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3058_305863

theorem coin_flip_probability (n m k l : ℕ) (h1 : n = 11) (h2 : m = 5) (h3 : k = 7) (h4 : l = 3) :
  let p := (1 : ℚ) / 2
  let total_success_prob := (n.choose k : ℚ) * p^k * (1 - p)^(n - k)
  let monday_success_prob := (m.choose l : ℚ) * p^l * (1 - p)^(m - l)
  let tuesday_success_prob := ((n - m).choose (k - l) : ℚ) * p^(k - l) * (1 - p)^(n - m - (k - l))
  (monday_success_prob * tuesday_success_prob) / total_success_prob = 5 / 11 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3058_305863


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3058_305887

theorem cyclic_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + 
   c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3058_305887


namespace NUMINAMATH_CALUDE_largest_even_number_under_300_l3058_305839

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 2 = 0 ∧ n ≤ 300

theorem largest_even_number_under_300 :
  ∀ n : ℕ, is_valid_number n → n ≤ 298 :=
by
  sorry

#check largest_even_number_under_300

end NUMINAMATH_CALUDE_largest_even_number_under_300_l3058_305839


namespace NUMINAMATH_CALUDE_point_inside_circle_range_l3058_305818

/-- Given that the point (1, 1) is inside the circle (x-a)^2+(y+a)^2=4, 
    prove that the range of a is -1 < a < 1 -/
theorem point_inside_circle_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_range_l3058_305818


namespace NUMINAMATH_CALUDE_xy_plus_x_plus_y_odd_l3058_305894

def S : Set ℕ := {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}

theorem xy_plus_x_plus_y_odd (x y : ℕ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x ≠ y) :
  ¬Even (x * y + x + y) :=
by sorry

end NUMINAMATH_CALUDE_xy_plus_x_plus_y_odd_l3058_305894


namespace NUMINAMATH_CALUDE_min_value_M_min_value_expression_equality_condition_l3058_305847

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem 1: Minimum value of M
theorem min_value_M : 
  (∃ (M : ℝ), ∀ (m : ℝ), (∃ (x₀ : ℝ), f x₀ ≤ m) → M ≤ m) ∧ 
  (∃ (x₀ : ℝ), f x₀ ≤ 2) := by sorry

-- Theorem 2: Minimum value of 1/(2a) + 1/(a+b)
theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3*a + b = 2) :
  1/(2*a) + 1/(a+b) ≥ 2 := by sorry

-- Theorem 3: Equality condition
theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3*a + b = 2) :
  1/(2*a) + 1/(a+b) = 2 ↔ a = 1/2 ∧ b = 1/2 := by sorry

end NUMINAMATH_CALUDE_min_value_M_min_value_expression_equality_condition_l3058_305847


namespace NUMINAMATH_CALUDE_cake_eating_ratio_l3058_305820

theorem cake_eating_ratio (cake_weight : ℝ) (parts : ℕ) (pierre_ate : ℝ) : 
  cake_weight = 400 →
  parts = 8 →
  pierre_ate = 100 →
  (pierre_ate / (cake_weight / parts.cast)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cake_eating_ratio_l3058_305820


namespace NUMINAMATH_CALUDE_heaviest_weight_determinable_l3058_305876

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weight in the geometric progression -/
inductive Weight
  | X
  | XR
  | XR2
  | XR3

/-- Represents a weighing operation -/
def weighing (left : List Weight) (right : List Weight) : WeighingResult :=
  sorry

/-- Determines the heaviest weight using two weighings -/
def findHeaviestWeight (x : ℝ) (r : ℝ) (h1 : x > 0) (h2 : r > 1) : Weight :=
  sorry

theorem heaviest_weight_determinable (x : ℝ) (r : ℝ) (h1 : x > 0) (h2 : r > 1) :
  ∃ (w : Weight), findHeaviestWeight x r h1 h2 = w ∧
    (w = Weight.X ∨ w = Weight.XR ∨ w = Weight.XR2 ∨ w = Weight.XR3) :=
  sorry

end NUMINAMATH_CALUDE_heaviest_weight_determinable_l3058_305876


namespace NUMINAMATH_CALUDE_y_intercept_of_given_line_l3058_305873

/-- A line is defined by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate where the line crosses the y-axis -/
def y_intercept (l : Line) : ℝ :=
  l.slope * (-l.point.1) + l.point.2

/-- The given line has slope 3 and passes through the point (4, 0) -/
def given_line : Line :=
  { slope := 3, point := (4, 0) }

theorem y_intercept_of_given_line :
  y_intercept given_line = -12 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_given_line_l3058_305873


namespace NUMINAMATH_CALUDE_rectangle_area_l3058_305851

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 186) : L * B = 2030 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3058_305851


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersect_A_C_case1_intersect_A_C_case2_intersect_A_C_case3_l3058_305825

open Set Real

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | 1 ≤ x ∧ x < 10} := by sorry

-- Theorem for (ℂR A) ∩ B
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x | 7 ≤ x ∧ x < 10} := by sorry

-- Theorems for A ∩ C in different cases
theorem intersect_A_C_case1 (a : ℝ) (h : a ≤ 1) : A ∩ C a = ∅ := by sorry

theorem intersect_A_C_case2 (a : ℝ) (h : 1 < a ∧ a ≤ 7) : 
  A ∩ C a = {x | 1 ≤ x ∧ x < a} := by sorry

theorem intersect_A_C_case3 (a : ℝ) (h : 7 < a) : 
  A ∩ C a = {x | 1 ≤ x ∧ x < 7} := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersect_A_C_case1_intersect_A_C_case2_intersect_A_C_case3_l3058_305825


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l3058_305870

/-- A quadratic equation is of the form ax^2 + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 = 3x - 2 -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem equation_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_equation_is_quadratic_l3058_305870


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_diff_prod_l3058_305844

theorem repeating_decimal_sum_diff_prod : 
  let repeating_decimal (n : ℕ) := n / 9
  (repeating_decimal 6) + (repeating_decimal 2) - (repeating_decimal 4) * (repeating_decimal 3) = 20 / 27 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_diff_prod_l3058_305844


namespace NUMINAMATH_CALUDE_parallelogram_area_l3058_305886

/-- The area of a parallelogram with base 32 cm and height 18 cm is 576 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 32 → 
  height = 18 → 
  area = base * height → 
  area = 576 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3058_305886


namespace NUMINAMATH_CALUDE_certain_number_is_four_l3058_305804

theorem certain_number_is_four (k : ℝ) (certain_number : ℝ) 
  (h1 : 64 / k = certain_number) 
  (h2 : k = 16) : 
  certain_number = 4 := by
sorry

end NUMINAMATH_CALUDE_certain_number_is_four_l3058_305804


namespace NUMINAMATH_CALUDE_harvard_applicants_l3058_305877

/-- The number of students who choose to attend Harvard University -/
def students_attending : ℕ := 900

/-- The acceptance rate for Harvard University applicants -/
def acceptance_rate : ℚ := 5 / 100

/-- The percentage of accepted students who choose to attend Harvard University -/
def attendance_rate : ℚ := 90 / 100

/-- The number of students who applied to Harvard University -/
def applicants : ℕ := 20000

theorem harvard_applicants :
  (↑students_attending : ℚ) = (↑applicants : ℚ) * acceptance_rate * attendance_rate := by
  sorry

end NUMINAMATH_CALUDE_harvard_applicants_l3058_305877


namespace NUMINAMATH_CALUDE_system_solution_l3058_305815

theorem system_solution (x y : ℝ) (eq1 : 2 * x - y = -1) (eq2 : x + 4 * y = 22) : x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3058_305815


namespace NUMINAMATH_CALUDE_student_marks_l3058_305845

theorem student_marks (total_marks : ℕ) (passing_percentage : ℚ) (failed_by : ℕ) (obtained_marks : ℕ) : 
  total_marks = 400 →
  passing_percentage = 33 / 100 →
  failed_by = 40 →
  obtained_marks = (total_marks * passing_percentage).floor - failed_by →
  obtained_marks = 92 := by
sorry

end NUMINAMATH_CALUDE_student_marks_l3058_305845


namespace NUMINAMATH_CALUDE_gift_purchase_probability_is_correct_l3058_305897

/-- The probability of purchasing gifts from all three stores and still having money left -/
def gift_purchase_probability : ℚ :=
  let initial_amount : ℕ := 5000
  let num_stores : ℕ := 3
  let prices : List ℕ := [1000, 1500, 2000]
  let total_combinations : ℕ := 3^num_stores
  let favorable_cases : ℕ := 17
  favorable_cases / total_combinations

/-- Theorem stating the probability of successful gift purchases -/
theorem gift_purchase_probability_is_correct :
  gift_purchase_probability = 17 / 27 := by sorry

end NUMINAMATH_CALUDE_gift_purchase_probability_is_correct_l3058_305897


namespace NUMINAMATH_CALUDE_bowling_tournament_orders_l3058_305812

/-- A tournament structure with players and games. -/
structure Tournament :=
  (num_players : ℕ)
  (num_games : ℕ)
  (outcomes_per_game : ℕ)

/-- The number of possible prize distribution orders in a tournament. -/
def prize_distribution_orders (t : Tournament) : ℕ := t.outcomes_per_game ^ t.num_games

/-- The specific tournament described in the problem. -/
def bowling_tournament : Tournament :=
  { num_players := 6,
    num_games := 5,
    outcomes_per_game := 2 }

/-- Theorem stating that the number of possible prize distribution orders
    in the bowling tournament is 32. -/
theorem bowling_tournament_orders :
  prize_distribution_orders bowling_tournament = 32 := by
  sorry


end NUMINAMATH_CALUDE_bowling_tournament_orders_l3058_305812


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_l3058_305885

theorem sum_of_last_two_digits (n : ℕ) : (6^15 + 10^15) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_l3058_305885


namespace NUMINAMATH_CALUDE_paiges_pencils_l3058_305842

/-- Paige's pencil problem -/
theorem paiges_pencils (P : ℕ) : 
  P - (P - 15) / 4 + 16 - 12 + 23 = 84 → P = 71 := by
  sorry

end NUMINAMATH_CALUDE_paiges_pencils_l3058_305842


namespace NUMINAMATH_CALUDE_euler_totient_equation_solutions_l3058_305888

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem euler_totient_equation_solutions (a b : ℕ) :
  (a > 0 ∧ b > 0 ∧ 14 * (phi a)^2 - phi (a * b) + 22 * (phi b)^2 = a^2 + b^2) ↔
  (∃ x y : ℕ, a = 30 * 2^x * 3^y ∧ b = 6 * 2^x * 3^y) :=
sorry

end NUMINAMATH_CALUDE_euler_totient_equation_solutions_l3058_305888


namespace NUMINAMATH_CALUDE_expression_simplification_l3058_305819

theorem expression_simplification (y : ℝ) :
  2 * y * (4 * y^2 - 3 * y + 1) - 6 * (y^2 - 3 * y + 4) =
  8 * y^3 - 12 * y^2 + 20 * y - 24 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3058_305819


namespace NUMINAMATH_CALUDE_third_week_cases_new_york_coronavirus_cases_l3058_305805

/-- Proves the number of new coronavirus cases in the third week --/
theorem third_week_cases (first_week : ℕ) (total_cases : ℕ) : ℕ :=
  let second_week := first_week / 2
  let first_two_weeks := first_week + second_week
  total_cases - first_two_weeks

/-- The main theorem that proves the number of new cases in the third week is 2000 --/
theorem new_york_coronavirus_cases : third_week_cases 5000 9500 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_third_week_cases_new_york_coronavirus_cases_l3058_305805


namespace NUMINAMATH_CALUDE_kitchen_tiles_l3058_305817

/-- The number of tiles needed to cover a rectangular floor -/
def tiles_needed (floor_length floor_width tile_area : ℕ) : ℕ :=
  (floor_length * floor_width) / tile_area

/-- Proof that 576 tiles are needed for the given floor and tile specifications -/
theorem kitchen_tiles :
  tiles_needed 48 72 6 = 576 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_tiles_l3058_305817


namespace NUMINAMATH_CALUDE_notebook_cost_l3058_305872

/-- The cost of a notebook and pencil, given their relationship -/
theorem notebook_cost (notebook_cost pencil_cost : ℝ) 
  (total : notebook_cost + pencil_cost = 2.40)
  (difference : notebook_cost = pencil_cost + 2) :
  notebook_cost = 2.20 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l3058_305872


namespace NUMINAMATH_CALUDE_no_two_roots_in_interval_l3058_305852

theorem no_two_roots_in_interval (a b c : ℝ) (ha : a > 0) (hcond : 12 * a + 5 * b + 2 * c > 0) :
  ¬∃ (x y : ℝ), 2 < x ∧ x < 3 ∧ 2 < y ∧ y < 3 ∧
  x ≠ y ∧
  a * x^2 + b * x + c = 0 ∧
  a * y^2 + b * y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_two_roots_in_interval_l3058_305852


namespace NUMINAMATH_CALUDE_tanker_fill_time_l3058_305840

/-- Represents the time it takes to fill a tanker using different pipe configurations -/
structure TankerFill where
  timeB : ℝ  -- Time for pipe B to fill the tanker alone
  timeCombined : ℝ  -- Time to fill the tanker using the combined method
  timeA : ℝ  -- Time for pipe A to fill the tanker alone

/-- Theorem stating the relationship between filling times for pipes A and B -/
theorem tanker_fill_time (tf : TankerFill) (h1 : tf.timeB = 40)
    (h2 : tf.timeCombined = 29.999999999999993) : tf.timeA = 60 := by
  sorry

#check tanker_fill_time

end NUMINAMATH_CALUDE_tanker_fill_time_l3058_305840


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3058_305855

theorem no_integer_solutions : ¬ ∃ (x y z : ℤ), 
  (x^2 - 2*x*y + 3*y^2 - 2*z^2 = 25) ∧ 
  (-x^2 + 4*y*z + 3*z^2 = 55) ∧ 
  (x^2 + 3*x*y - y^2 + 7*z^2 = 130) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3058_305855


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l3058_305882

theorem quadratic_root_theorem (c : ℝ) : 
  (∀ x : ℝ, 2*x^2 + 8*x + c = 0 ↔ x = -2 + Real.sqrt 3 ∨ x = -2 - Real.sqrt 3) →
  c = 13/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l3058_305882


namespace NUMINAMATH_CALUDE_fraction_problem_l3058_305816

theorem fraction_problem (p q : ℚ) (h : p / q = 4 / 5) :
  ∃ x : ℚ, 11 / 7 + x / (2 * q + p) = 2 ∧ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_fraction_problem_l3058_305816


namespace NUMINAMATH_CALUDE_a_range_l3058_305830

-- Define proposition P
def P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 4 > 0

-- Define proposition Q
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a = 0

-- Theorem statement
theorem a_range (a : ℝ) : P a ∧ ¬(Q a) ↔ 1 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3058_305830


namespace NUMINAMATH_CALUDE_matrix_tripler_uniqueness_l3058_305814

theorem matrix_tripler_uniqueness (A M : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ (i j : Fin 2), (M • A) i j = 3 * A i j) ↔ M = ![![3, 0], ![0, 3]] := by
sorry

end NUMINAMATH_CALUDE_matrix_tripler_uniqueness_l3058_305814


namespace NUMINAMATH_CALUDE_square_sum_sqrt_difference_and_sum_l3058_305854

theorem square_sum_sqrt_difference_and_sum (x₁ x₂ : ℝ) :
  x₁ = Real.sqrt 3 - Real.sqrt 2 →
  x₂ = Real.sqrt 3 + Real.sqrt 2 →
  x₁^2 + x₂^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_square_sum_sqrt_difference_and_sum_l3058_305854


namespace NUMINAMATH_CALUDE_equation_solution_l3058_305883

theorem equation_solution (x : ℝ) (a b : ℕ) :
  (x^2 + 5*x + 5/x + 1/x^2 = 40) →
  (x = a + Real.sqrt b) →
  (a > 0 ∧ b > 0) →
  (a + b = 11) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3058_305883


namespace NUMINAMATH_CALUDE_student_competition_assignments_l3058_305892

/-- The number of ways to assign students to competitions -/
def num_assignments (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_competitions ^ num_students

/-- Theorem: For 4 students and 3 competitions, there are 3^4 different assignment outcomes -/
theorem student_competition_assignments :
  num_assignments 4 3 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_student_competition_assignments_l3058_305892


namespace NUMINAMATH_CALUDE_modular_inverse_11_mod_1021_l3058_305857

theorem modular_inverse_11_mod_1021 : ∃ x : ℕ, x ∈ Finset.range 1021 ∧ (11 * x) % 1021 = 1 := by
  use 557
  sorry

end NUMINAMATH_CALUDE_modular_inverse_11_mod_1021_l3058_305857


namespace NUMINAMATH_CALUDE_flyers_made_total_l3058_305841

/-- The total number of flyers made by Jack and Rose for their dog-walking business -/
def total_flyers (jack_handed_out : ℕ) (rose_handed_out : ℕ) (flyers_left : ℕ) : ℕ :=
  jack_handed_out + rose_handed_out + flyers_left

/-- Theorem stating the total number of flyers made by Jack and Rose -/
theorem flyers_made_total :
  total_flyers 120 320 796 = 1236 := by
  sorry

end NUMINAMATH_CALUDE_flyers_made_total_l3058_305841


namespace NUMINAMATH_CALUDE_sample_size_is_176_l3058_305862

/-- Represents the number of students in a stratum -/
structure Stratum where
  size : ℕ

/-- Represents a sample taken from a stratum -/
structure Sample where
  size : ℕ

/-- Calculates the total sample size for stratified sampling -/
def stratifiedSampleSize (male : Stratum) (female : Stratum) (femaleSample : Sample) : ℕ :=
  let maleSampleSize := (male.size * femaleSample.size) / female.size
  maleSampleSize + femaleSample.size

/-- Theorem: The total sample size is 176 given the specified conditions -/
theorem sample_size_is_176
  (male : Stratum)
  (female : Stratum)
  (femaleSample : Sample)
  (h1 : male.size = 1200)
  (h2 : female.size = 1000)
  (h3 : femaleSample.size = 80) :
  stratifiedSampleSize male female femaleSample = 176 := by
  sorry

#check sample_size_is_176

end NUMINAMATH_CALUDE_sample_size_is_176_l3058_305862


namespace NUMINAMATH_CALUDE_room_breadth_calculation_l3058_305868

theorem room_breadth_calculation (room_length : ℝ) (carpet_width_cm : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  room_length = 18 →
  carpet_width_cm = 75 →
  cost_per_meter = 4.50 →
  total_cost = 810 →
  (total_cost / cost_per_meter) / room_length * (carpet_width_cm / 100) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_room_breadth_calculation_l3058_305868


namespace NUMINAMATH_CALUDE_james_payment_is_correct_l3058_305822

/-- Calculates James's payment for stickers given the number of packs, stickers per pack,
    cost per sticker, discount rate, tax rate, and friend's contribution ratio. -/
def james_payment (packs : ℕ) (stickers_per_pack : ℕ) (cost_per_sticker : ℚ)
                  (discount_rate : ℚ) (tax_rate : ℚ) (friend_contribution_ratio : ℚ) : ℚ :=
  let total_cost := packs * stickers_per_pack * cost_per_sticker
  let discounted_cost := total_cost * (1 - discount_rate)
  let taxed_cost := discounted_cost * (1 + tax_rate)
  taxed_cost * (1 - friend_contribution_ratio)

/-- Proves that James's payment is $36.38 given the specific conditions of the problem. -/
theorem james_payment_is_correct :
  james_payment 8 40 (25 / 100) (15 / 100) (7 / 100) (1 / 2) = 3638 / 100 := by
  sorry

end NUMINAMATH_CALUDE_james_payment_is_correct_l3058_305822


namespace NUMINAMATH_CALUDE_distinct_arrangements_eq_twelve_l3058_305889

/-- The number of distinct arrangements of a 4-letter word with one letter repeated twice -/
def distinct_arrangements : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

/-- Theorem stating that the number of distinct arrangements is 12 -/
theorem distinct_arrangements_eq_twelve : distinct_arrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_eq_twelve_l3058_305889


namespace NUMINAMATH_CALUDE_davidsons_class_as_l3058_305826

/-- Proves that given the conditions of the problem, 12 students in Mr. Davidson's class received an 'A' -/
theorem davidsons_class_as (carter_total : ℕ) (carter_as : ℕ) (davidson_total : ℕ) :
  carter_total = 20 →
  carter_as = 8 →
  davidson_total = 30 →
  ∃ davidson_as : ℕ,
    davidson_as * carter_total = carter_as * davidson_total ∧
    davidson_as = 12 :=
by sorry

end NUMINAMATH_CALUDE_davidsons_class_as_l3058_305826


namespace NUMINAMATH_CALUDE_cyclist_speed_calculation_l3058_305811

/-- Given two cyclists, Joann and Fran, this theorem proves the required speed for Fran
    to cover the same distance as Joann in a different amount of time. -/
theorem cyclist_speed_calculation (joann_speed joann_time fran_time : ℝ) 
    (hjs : joann_speed = 15) 
    (hjt : joann_time = 4)
    (hft : fran_time = 5) : 
  joann_speed * joann_time / fran_time = 12 := by
  sorry

#check cyclist_speed_calculation

end NUMINAMATH_CALUDE_cyclist_speed_calculation_l3058_305811


namespace NUMINAMATH_CALUDE_combined_area_of_triangle_and_square_l3058_305837

theorem combined_area_of_triangle_and_square (triangle_area : ℝ) (base_length : ℝ) : 
  triangle_area = 720 → 
  base_length = 40 → 
  (triangle_area = 1/2 * base_length * (triangle_area / (1/2 * base_length))) →
  (base_length^2 + triangle_area = 2320) := by
sorry

end NUMINAMATH_CALUDE_combined_area_of_triangle_and_square_l3058_305837


namespace NUMINAMATH_CALUDE_min_value_theorem_l3058_305821

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (4 * x^2) / (y + 1) + y^2 / (2 * x + 2) ≥ 4/5 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3058_305821


namespace NUMINAMATH_CALUDE_customers_before_correct_l3058_305836

/-- The number of customers before the lunch rush -/
def customers_before : ℝ := 29.0

/-- The number of customers added during the lunch rush -/
def customers_added_lunch : ℝ := 20.0

/-- The number of customers that came in after the lunch rush -/
def customers_after_lunch : ℝ := 34.0

/-- The total number of customers after all additions -/
def total_customers : ℝ := 83.0

/-- Theorem stating that the number of customers before the lunch rush is correct -/
theorem customers_before_correct :
  customers_before + customers_added_lunch + customers_after_lunch = total_customers :=
by sorry

end NUMINAMATH_CALUDE_customers_before_correct_l3058_305836


namespace NUMINAMATH_CALUDE_de_moivre_formula_l3058_305829

theorem de_moivre_formula (x : ℝ) (n : ℕ) (h : x ∈ Set.Ioo 0 (π / 2)) :
  (Complex.exp (Complex.I * x)) ^ n = Complex.exp (Complex.I * (n : ℝ) * x) := by
  sorry

#check de_moivre_formula

end NUMINAMATH_CALUDE_de_moivre_formula_l3058_305829


namespace NUMINAMATH_CALUDE_can_repair_propeller_l3058_305807

/-- Represents the cost of a blade in tugriks -/
def blade_cost : ℕ := 120

/-- Represents the cost of a screw in tugriks -/
def screw_cost : ℕ := 9

/-- Represents the discount threshold in tugriks -/
def discount_threshold : ℕ := 250

/-- Represents the discount rate as a percentage -/
def discount_rate : ℚ := 20 / 100

/-- Represents Karlson's budget in tugriks -/
def budget : ℕ := 360

/-- Calculates the discounted price of an item -/
def apply_discount (price : ℕ) : ℚ :=
  (1 - discount_rate) * price

/-- Theorem stating that Karlson can repair his propeller with his budget -/
theorem can_repair_propeller : ∃ (first_purchase second_purchase : ℕ),
  first_purchase ≥ discount_threshold ∧
  first_purchase + second_purchase ≤ budget ∧
  first_purchase = 2 * blade_cost + 2 * screw_cost ∧
  second_purchase = apply_discount blade_cost :=
sorry

end NUMINAMATH_CALUDE_can_repair_propeller_l3058_305807


namespace NUMINAMATH_CALUDE_fraction_calculation_l3058_305801

theorem fraction_calculation : (1/4 + 1/5) / (3/7 - 1/8) = 126/85 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3058_305801


namespace NUMINAMATH_CALUDE_log_equation_solution_l3058_305871

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 2 + Real.log x / Real.log 8 = 5 → x = 2^(15/4) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3058_305871


namespace NUMINAMATH_CALUDE_positive_quadratic_intervals_l3058_305838

theorem positive_quadratic_intervals (x : ℝ) : 
  (x - 2) * (x + 3) > 0 ↔ x < -3 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_positive_quadratic_intervals_l3058_305838


namespace NUMINAMATH_CALUDE_sin_cos_lt_cos_sin_acute_l3058_305833

theorem sin_cos_lt_cos_sin_acute (x : ℝ) (h : 0 < x ∧ x < π / 2) : 
  Real.sin (Real.cos x) < Real.cos (Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_lt_cos_sin_acute_l3058_305833


namespace NUMINAMATH_CALUDE_quadratic_root_m_value_l3058_305853

theorem quadratic_root_m_value :
  ∀ m : ℝ, ((-1 : ℝ)^2 + m * (-1) + 1 = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_m_value_l3058_305853


namespace NUMINAMATH_CALUDE_trout_division_l3058_305848

theorem trout_division (total_trout : ℕ) (num_people : ℕ) (trout_per_person : ℕ) : 
  total_trout = 52 → num_people = 4 → trout_per_person = total_trout / num_people → trout_per_person = 13 := by
  sorry

end NUMINAMATH_CALUDE_trout_division_l3058_305848


namespace NUMINAMATH_CALUDE_amount_transferred_l3058_305800

def initial_balance : ℕ := 27004
def remaining_balance : ℕ := 26935

theorem amount_transferred : initial_balance - remaining_balance = 69 := by
  sorry

end NUMINAMATH_CALUDE_amount_transferred_l3058_305800


namespace NUMINAMATH_CALUDE_probability_even_product_l3058_305809

def range_start : ℕ := 6
def range_end : ℕ := 18

def is_in_range (n : ℕ) : Prop := range_start ≤ n ∧ n ≤ range_end

def total_integers : ℕ := range_end - range_start + 1

def total_combinations : ℕ := (total_integers * (total_integers - 1)) / 2

def count_even_in_range : ℕ := (range_end - range_start) / 2 + 1

def count_odd_in_range : ℕ := total_integers - count_even_in_range

def combinations_with_odd_product : ℕ := (count_odd_in_range * (count_odd_in_range - 1)) / 2

def combinations_with_even_product : ℕ := total_combinations - combinations_with_odd_product

theorem probability_even_product : 
  (combinations_with_even_product : ℚ) / total_combinations = 9 / 13 := by sorry

end NUMINAMATH_CALUDE_probability_even_product_l3058_305809


namespace NUMINAMATH_CALUDE_volume_increase_when_quadrupled_l3058_305810

/-- Given a cylindrical container, when all its dimensions are quadrupled, 
    its volume increases by a factor of 64. -/
theorem volume_increase_when_quadrupled (r h V : ℝ) :
  V = π * r^2 * h →
  (π * (4*r)^2 * (4*h)) = 64 * V :=
by sorry

end NUMINAMATH_CALUDE_volume_increase_when_quadrupled_l3058_305810


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3058_305856

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  4 - 12 * (x - y) + 9 * (x - y)^2 = (2 - 3*x + 3*y)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (a x : ℝ) :
  2*a*(x^2 + 1)^2 - 8*a*x^2 = 2*a*(x - 1)^2*(x + 1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3058_305856


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3058_305865

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  6 * a^3 + 9 * b^3 + 32 * c^3 + 1 / (4 * a * b * c) ≥ 6 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  6 * a^3 + 9 * b^3 + 32 * c^3 + 1 / (4 * a * b * c) = 6 ↔
  a = (1 : ℝ) / (6 : ℝ)^(1/3) ∧ b = (1 : ℝ) / (9 : ℝ)^(1/3) ∧ c = (1 : ℝ) / (32 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3058_305865


namespace NUMINAMATH_CALUDE_point_inside_circle_l3058_305884

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if a point is inside a circle -/
def isInside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

theorem point_inside_circle (O : Circle) (P : ℝ × ℝ) 
    (h1 : O.radius = 5)
    (h2 : Real.sqrt ((P.1 - O.center.1)^2 + (P.2 - O.center.2)^2) = 4) :
  isInside P O := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l3058_305884


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l3058_305813

/-- Calculates the number of samples to be drawn from a subgroup in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (total_sample_size : ℕ) (subgroup_size : ℕ) : ℕ :=
  (total_sample_size * subgroup_size) / total_population

/-- Theorem: In a stratified sampling scenario with a total population of 1000 and a sample size of 50,
    the number of samples drawn from a subgroup of 200 is equal to 10 -/
theorem stratified_sample_theorem :
  stratified_sample_size 1000 50 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l3058_305813


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l3058_305864

theorem sequence_sum_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S n = (2/3) * (a n) + 1/3) →
  (∀ n : ℕ, a n = (-2)^(n-1)) :=
sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l3058_305864


namespace NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_150_l3058_305835

theorem no_integer_pairs_with_square_diff_150 :
  ¬∃ (m n : ℕ), m ≥ n ∧ m^2 - n^2 = 150 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_pairs_with_square_diff_150_l3058_305835


namespace NUMINAMATH_CALUDE_westpark_teachers_l3058_305881

/-- The number of students at Westpark High School -/
def total_students : ℕ := 900

/-- The number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- The number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- The number of students in each class -/
def students_per_class : ℕ := 25

/-- The number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Calculate the number of teachers required at Westpark High School -/
def calculate_teachers : ℕ := 
  (total_students * classes_per_student / students_per_class + 
   (if (total_students * classes_per_student) % students_per_class = 0 then 0 else 1)) / 
  classes_per_teacher +
  (if ((total_students * classes_per_student / students_per_class + 
       (if (total_students * classes_per_student) % students_per_class = 0 then 0 else 1)) % 
      classes_per_teacher = 0) 
   then 0 
   else 1)

/-- Theorem stating that the number of teachers at Westpark High School is 44 -/
theorem westpark_teachers : calculate_teachers = 44 := by
  sorry

end NUMINAMATH_CALUDE_westpark_teachers_l3058_305881


namespace NUMINAMATH_CALUDE_smallest_product_of_factors_l3058_305898

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_product_of_factors (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  is_factor a 48 → 
  is_factor b 48 → 
  ¬ is_factor (a * b) 48 → 
  (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → is_factor x 48 → is_factor y 48 → 
    ¬ is_factor (x * y) 48 → a * b ≤ x * y) → 
  a * b = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_of_factors_l3058_305898


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3058_305808

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3058_305808


namespace NUMINAMATH_CALUDE_variable_equals_one_l3058_305806

/-- The operator  applied to a real number x -/
def box_operator (x : ℝ) : ℝ := x * (2 - x)

/-- Theorem stating that if y + 1 = (y + 1), then y = 1 -/
theorem variable_equals_one (y : ℝ) (h : y + 1 = box_operator (y + 1)) : y = 1 := by
  sorry

end NUMINAMATH_CALUDE_variable_equals_one_l3058_305806


namespace NUMINAMATH_CALUDE_gcd_lcm_theorem_l3058_305879

theorem gcd_lcm_theorem : 
  (Nat.gcd 42 63 = 21 ∧ Nat.lcm 42 63 = 126) ∧ 
  (Nat.gcd 8 20 = 4 ∧ Nat.lcm 8 20 = 40) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_theorem_l3058_305879


namespace NUMINAMATH_CALUDE_no_integer_solution_a_l3058_305891

theorem no_integer_solution_a (x y : ℤ) : x^2 + y^2 ≠ 2003 := by
  sorry

#check no_integer_solution_a

end NUMINAMATH_CALUDE_no_integer_solution_a_l3058_305891


namespace NUMINAMATH_CALUDE_train_length_l3058_305866

/-- The length of a train given its crossing times over a platform and a signal pole -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 200)
  (h2 : platform_time = 30)
  (h3 : pole_time = 18) :
  ∃ (train_length : ℝ), 
    train_length + platform_length = (train_length / pole_time) * platform_time ∧ 
    train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3058_305866


namespace NUMINAMATH_CALUDE_cube_plus_minus_one_divisible_by_seven_l3058_305860

theorem cube_plus_minus_one_divisible_by_seven (n : ℤ) (h : ¬ 7 ∣ n) :
  7 ∣ (n^3 - 1) ∨ 7 ∣ (n^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_minus_one_divisible_by_seven_l3058_305860


namespace NUMINAMATH_CALUDE_sqrt_eight_plus_abs_sqrt_two_minus_two_plus_neg_half_inv_eq_sqrt_two_l3058_305895

theorem sqrt_eight_plus_abs_sqrt_two_minus_two_plus_neg_half_inv_eq_sqrt_two :
  Real.sqrt 8 + |Real.sqrt 2 - 2| + (-1/2)⁻¹ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_plus_abs_sqrt_two_minus_two_plus_neg_half_inv_eq_sqrt_two_l3058_305895


namespace NUMINAMATH_CALUDE_square_root_expression_simplification_l3058_305867

theorem square_root_expression_simplification :
  (2 + Real.sqrt 3)^2 - Real.sqrt 18 * Real.sqrt (2/3) = 7 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_expression_simplification_l3058_305867


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l3058_305832

def complex_i : ℂ := Complex.I

theorem imaginary_part_of_one_plus_i_to_fifth (h : complex_i ^ 2 = -1) :
  Complex.im ((1 : ℂ) + complex_i) ^ 5 = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_to_fifth_l3058_305832


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3058_305874

theorem max_value_quadratic (x : ℝ) (h : 0 < x ∧ x < 3/2) :
  x * (2 - x) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3058_305874


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l3058_305802

theorem units_digit_of_quotient (h : 7 ∣ (4^2065 + 6^2065)) :
  (4^2065 + 6^2065) / 7 % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l3058_305802


namespace NUMINAMATH_CALUDE_zipline_configurations_count_l3058_305824

/-- The number of stories in each building -/
def n : ℕ := 5

/-- The total number of steps (right + up) -/
def total_steps : ℕ := n + n

/-- The number of ways to string ziplines between two n-story buildings
    satisfying the given conditions -/
def num_zipline_configurations : ℕ := Nat.choose total_steps n

/-- Theorem stating that the number of zipline configurations
    is equal to 252 -/
theorem zipline_configurations_count :
  num_zipline_configurations = 252 := by sorry

end NUMINAMATH_CALUDE_zipline_configurations_count_l3058_305824


namespace NUMINAMATH_CALUDE_circle_to_rectangle_length_l3058_305858

/-- Given a circle with radius R, when divided into equal parts and rearranged to form
    an approximate rectangle with perimeter 20.7 cm, the length of this rectangle is π * R. -/
theorem circle_to_rectangle_length (R : ℝ) (h : (2 * R + 2 * π * R / 2) = 20.7) :
  π * R = (20.7 : ℝ) / 2 - R := by
  sorry

end NUMINAMATH_CALUDE_circle_to_rectangle_length_l3058_305858


namespace NUMINAMATH_CALUDE_perpendicular_line_exists_l3058_305896

-- Define the concept of a line
def Line : Type := sorry

-- Define the concept of a plane
def Plane : Type := sorry

-- Define what it means for a line to be within a plane
def within_plane (l : Line) (p : Plane) : Prop := sorry

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_line_exists (l : Line) (α : Plane) :
  ∃ m : Line, within_plane m α ∧ perpendicular m l := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_exists_l3058_305896


namespace NUMINAMATH_CALUDE_present_age_ratio_l3058_305899

/-- Given Suji's present age and the future ratio of ages, find the present ratio of ages --/
theorem present_age_ratio (suji_age : ℕ) (future_ratio_abi : ℕ) (future_ratio_suji : ℕ) :
  suji_age = 24 →
  (future_ratio_abi : ℚ) / future_ratio_suji = 11 / 9 →
  ∃ (abi_age : ℕ),
    (abi_age + 3 : ℚ) / (suji_age + 3) = (future_ratio_abi : ℚ) / future_ratio_suji ∧
    (abi_age : ℚ) / suji_age = 5 / 4 :=
by sorry

end NUMINAMATH_CALUDE_present_age_ratio_l3058_305899


namespace NUMINAMATH_CALUDE_total_birds_count_l3058_305803

/-- The number of blackbirds in each tree -/
def blackbirds_per_tree : ℕ := 3

/-- The number of trees in the park -/
def number_of_trees : ℕ := 7

/-- The number of magpies in the park -/
def number_of_magpies : ℕ := 13

/-- The total number of birds in the park -/
def total_birds : ℕ := blackbirds_per_tree * number_of_trees + number_of_magpies

theorem total_birds_count : total_birds = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_count_l3058_305803


namespace NUMINAMATH_CALUDE_quartic_root_inequality_l3058_305880

theorem quartic_root_inequality (a b : ℝ) :
  (∃ x : ℝ, x^4 - a*x^3 + 2*x^2 - b*x + 1 = 0) → a^2 + b^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_quartic_root_inequality_l3058_305880


namespace NUMINAMATH_CALUDE_arithmetic_sequence_finite_negative_terms_l3058_305875

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def has_finite_negative_terms (a : ℕ → ℝ) : Prop :=
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → a n ≥ 0

theorem arithmetic_sequence_finite_negative_terms
  (a : ℕ → ℝ) (d : ℝ) (h1 : is_arithmetic_sequence a d)
  (h2 : a 1 < 0) (h3 : d > 0) :
  has_finite_negative_terms a :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_finite_negative_terms_l3058_305875


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l3058_305893

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : Real) 
  (bridge_length : Real) 
  (crossing_time : Real) 
  (h1 : train_length = 150) 
  (h2 : bridge_length = 320) 
  (h3 : crossing_time = 40) : 
  (train_length + bridge_length) / crossing_time = 11.75 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l3058_305893


namespace NUMINAMATH_CALUDE_smallest_norm_w_l3058_305878

theorem smallest_norm_w (w : ℝ × ℝ) (h : ‖w + (4, 2)‖ = 10) :
  ∃ (w_min : ℝ × ℝ), (∀ w' : ℝ × ℝ, ‖w' + (4, 2)‖ = 10 → ‖w_min‖ ≤ ‖w'‖) ∧ ‖w_min‖ = 10 - 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_norm_w_l3058_305878


namespace NUMINAMATH_CALUDE_unique_quadratic_pair_l3058_305846

theorem unique_quadratic_pair : 
  ∃! (b c : ℕ+), 
    (∀ x : ℝ, (x^2 + 2*b*x + c ≤ 0 → x^2 + 2*b*x + c = 0)) ∧ 
    (∀ x : ℝ, (x^2 + 2*c*x + b ≤ 0 → x^2 + 2*c*x + b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_pair_l3058_305846


namespace NUMINAMATH_CALUDE_rectangleAreaStage4_l3058_305828

/-- The area of a rectangle formed by four squares with side lengths
    starting from 2 inches and increasing by 1 inch per stage. -/
def rectangleArea : ℕ → ℕ
| 1 => 2^2
| 2 => 2^2 + 3^2
| 3 => 2^2 + 3^2 + 4^2
| 4 => 2^2 + 3^2 + 4^2 + 5^2
| _ => 0

/-- The area of the rectangle at Stage 4 is 54 square inches. -/
theorem rectangleAreaStage4 : rectangleArea 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_rectangleAreaStage4_l3058_305828


namespace NUMINAMATH_CALUDE_lacy_correct_percentage_l3058_305861

theorem lacy_correct_percentage (x : ℝ) (h : x > 0) :
  let total_problems := 6 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  (correct_problems / total_problems) * 100 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_lacy_correct_percentage_l3058_305861
