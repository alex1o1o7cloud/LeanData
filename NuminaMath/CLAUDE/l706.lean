import Mathlib

namespace NUMINAMATH_CALUDE_corresponding_angles_equal_if_then_form_l706_70690

/-- Two angles are corresponding if they occupy the same relative position when a line intersects two other lines. -/
def are_corresponding (α β : Angle) : Prop := sorry

/-- Rewrite the statement "corresponding angles are equal" in if-then form -/
theorem corresponding_angles_equal_if_then_form :
  (∀ α β : Angle, are_corresponding α β → α = β) ↔
  (∀ α β : Angle, are_corresponding α β → α = β) :=
by sorry

end NUMINAMATH_CALUDE_corresponding_angles_equal_if_then_form_l706_70690


namespace NUMINAMATH_CALUDE_odd_integer_divisor_form_l706_70666

theorem odd_integer_divisor_form (n : ℕ) (hn : Odd n) (x y : ℕ) 
  (hx : x > 0) (hy : y > 0) (heq : (1 : ℚ) / x + (1 : ℚ) / y = (4 : ℚ) / n) :
  ∃ (k : ℕ), ∃ (d : ℕ), d ∣ n ∧ d = 4 * k - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_integer_divisor_form_l706_70666


namespace NUMINAMATH_CALUDE_expression_value_l706_70689

theorem expression_value (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 3)    -- absolute value of m is 3
  : (a + b) / m + c * d + m = 4 ∨ (a + b) / m + c * d + m = -2 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l706_70689


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l706_70612

theorem smallest_x_absolute_value_equation :
  let f : ℝ → ℝ := λ x ↦ |2 * x + 5|
  ∃ x : ℝ, f x = 18 ∧ ∀ y : ℝ, f y = 18 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l706_70612


namespace NUMINAMATH_CALUDE_john_spent_625_l706_70677

/-- The amount John spent on his purchases with a coupon -/
def total_spent (vacuum_cost dishwasher_cost coupon_value : ℕ) : ℕ :=
  vacuum_cost + dishwasher_cost - coupon_value

/-- Theorem stating that John spent $625 on his purchases -/
theorem john_spent_625 :
  total_spent 250 450 75 = 625 := by
  sorry

end NUMINAMATH_CALUDE_john_spent_625_l706_70677


namespace NUMINAMATH_CALUDE_triangle_side_length_l706_70685

def isOnParabola (p : ℝ × ℝ) : Prop := p.2 = -(p.1^2)

def isIsoscelesRightTriangle (p q : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 = q.1^2 + q.2^2 ∧ p.1 * q.1 + p.2 * q.2 = 0

theorem triangle_side_length
  (p q : ℝ × ℝ)
  (h1 : isOnParabola p)
  (h2 : isOnParabola q)
  (h3 : isIsoscelesRightTriangle p q)
  : Real.sqrt (p.1^2 + p.2^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l706_70685


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_multiple_l706_70669

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem consecutive_odd_numbers_multiple (a b c : ℤ) : 
  is_odd a ∧ is_odd b ∧ is_odd c ∧  -- Three odd numbers
  b = a + 2 ∧ c = b + 2 ∧           -- Consecutive
  a = 7 ∧                           -- First number is 7
  ∃ m : ℤ, 8 * a = 3 * c + 5 + m * b -- Equation condition
  → m = 2 :=                        -- Multiple of second number is 2
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_multiple_l706_70669


namespace NUMINAMATH_CALUDE_biography_increase_l706_70606

theorem biography_increase (B : ℝ) (N : ℝ) (h1 : B > 0) (h2 : N > 0) : 
  (0.20 * B + N = 0.32 * (B + N)) → 
  ((N / (0.20 * B)) = 15 / 17) := by
sorry

end NUMINAMATH_CALUDE_biography_increase_l706_70606


namespace NUMINAMATH_CALUDE_zoo_animals_l706_70676

/-- The number of animals in a zoo satisfies certain conditions. -/
theorem zoo_animals (parrots snakes monkeys elephants zebras : ℕ) :
  parrots = 8 →
  snakes = 3 * parrots →
  monkeys = 2 * snakes →
  elephants = (parrots + snakes) / 2 →
  zebras + 35 = monkeys →
  elephants - zebras = 3 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_l706_70676


namespace NUMINAMATH_CALUDE_widget_purchase_l706_70686

theorem widget_purchase (W : ℝ) (h1 : 6 * W = 8 * (W - 2)) : 6 * W = 48 := by
  sorry

end NUMINAMATH_CALUDE_widget_purchase_l706_70686


namespace NUMINAMATH_CALUDE_turtle_count_l706_70621

theorem turtle_count (T : ℕ) : 
  (T + (3 * T - 2)) / 2 = 17 → T = 9 := by
  sorry

end NUMINAMATH_CALUDE_turtle_count_l706_70621


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l706_70682

/-- The coordinates of a point (3, -2) with respect to the origin are (3, -2). -/
theorem point_coordinates_wrt_origin :
  let p : ℝ × ℝ := (3, -2)
  p = (3, -2) :=
by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l706_70682


namespace NUMINAMATH_CALUDE_intersection_chord_length_l706_70673

/-- Given a line and a circle that intersect to form a chord of length √3, 
    prove that the parameter 'a' in the circle equation is 0. -/
theorem intersection_chord_length (a : ℝ) : 
  (∃ (x y : ℝ), (8*x - 6*y - 3 = 0) ∧ 
                (x^2 + y^2 - 2*x + a = 0) ∧ 
                (∃ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) ∧ 
                                (8*x' - 6*y' - 3 = 0) ∧ 
                                (x'^2 + y'^2 - 2*x' + a = 0) ∧ 
                                ((x - x')^2 + (y - y')^2 = 3))) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l706_70673


namespace NUMINAMATH_CALUDE_a_greater_than_c_greater_than_b_l706_70663

theorem a_greater_than_c_greater_than_b :
  let a := 0.6 * Real.exp 0.4
  let b := 2 - Real.log 4
  let c := Real.exp 1 - 2
  a > c ∧ c > b :=
by sorry

end NUMINAMATH_CALUDE_a_greater_than_c_greater_than_b_l706_70663


namespace NUMINAMATH_CALUDE_fourth_term_is_one_l706_70631

def geometric_progression (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem fourth_term_is_one :
  let a₁ := (2 : ℝ)^(1/4)
  let a₂ := (2 : ℝ)^(1/6)
  let a₃ := (2 : ℝ)^(1/12)
  let r := a₂ / a₁
  geometric_progression a₁ r 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_one_l706_70631


namespace NUMINAMATH_CALUDE_smallest_ambiguous_weight_correct_l706_70614

/-- The smallest total weight of kittens for which the number of kittens is not uniquely determined -/
def smallest_ambiguous_weight : ℕ := 480

/-- The total weight of the two lightest kittens -/
def lightest_two_weight : ℕ := 80

/-- The total weight of the four heaviest kittens -/
def heaviest_four_weight : ℕ := 200

/-- Predicate to check if a given total weight allows for a unique determination of the number of kittens -/
def is_uniquely_determined (total_weight : ℕ) : Prop :=
  ∀ n m : ℕ, 
    (n ≠ m) → 
    (∃ (weights_n weights_m : List ℕ),
      (weights_n.length = n ∧ weights_m.length = m) ∧
      (weights_n.sum = total_weight ∧ weights_m.sum = total_weight) ∧
      (weights_n.take 2).sum = lightest_two_weight ∧
      (weights_m.take 2).sum = lightest_two_weight ∧
      (weights_n.reverse.take 4).sum = heaviest_four_weight ∧
      (weights_m.reverse.take 4).sum = heaviest_four_weight) →
    False

theorem smallest_ambiguous_weight_correct :
  (∀ w : ℕ, w < smallest_ambiguous_weight → is_uniquely_determined w) ∧
  ¬is_uniquely_determined smallest_ambiguous_weight :=
sorry

end NUMINAMATH_CALUDE_smallest_ambiguous_weight_correct_l706_70614


namespace NUMINAMATH_CALUDE_rectangle_sides_l706_70607

theorem rectangle_sides (area : ℝ) (perimeter : ℝ) : area = 12 ∧ perimeter = 26 →
  ∃ (length width : ℝ), length * width = area ∧ 2 * (length + width) = perimeter ∧
  ((length = 12 ∧ width = 1) ∨ (length = 1 ∧ width = 12)) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sides_l706_70607


namespace NUMINAMATH_CALUDE_division_value_problem_l706_70604

theorem division_value_problem (x : ℝ) : 
  (1152 / x) - 189 = 3 → x = 6 := by
sorry

end NUMINAMATH_CALUDE_division_value_problem_l706_70604


namespace NUMINAMATH_CALUDE_correct_sample_size_l706_70647

/-- Represents a school in the sampling problem -/
structure School where
  students : ℕ

/-- Represents the sampling data for two schools -/
structure SamplingData where
  schoolA : School
  schoolB : School
  sampleA : ℕ

/-- Calculates the proportional sample size for the second school -/
def calculateSampleB (data : SamplingData) : ℕ :=
  (data.schoolB.students * data.sampleA) / data.schoolA.students

/-- Theorem stating the correct sample size for School B -/
theorem correct_sample_size (data : SamplingData) 
    (h1 : data.schoolA.students = 800)
    (h2 : data.schoolB.students = 500)
    (h3 : data.sampleA = 48) :
  calculateSampleB data = 30 := by
  sorry

#eval calculateSampleB { 
  schoolA := { students := 800 }, 
  schoolB := { students := 500 }, 
  sampleA := 48 
}

end NUMINAMATH_CALUDE_correct_sample_size_l706_70647


namespace NUMINAMATH_CALUDE_fraction_problem_l706_70601

theorem fraction_problem (a b c : ℕ) : 
  a ≤ 10 ∧ b ≤ 10 ∧ c ≤ 10 →
  (3 * a + 2 : ℚ) / 3 = (4 * b + 3 : ℚ) / 4 ∧ 
  (3 * a + 2 : ℚ) / 3 = (5 * c + 3 : ℚ) / 5 →
  (2 * a + b : ℚ) / c = 19 / 4 := by
sorry

#eval (19 : ℚ) / 4  -- This should output 4.75

end NUMINAMATH_CALUDE_fraction_problem_l706_70601


namespace NUMINAMATH_CALUDE_fib_divisibility_spacing_l706_70616

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: Numbers in Fibonacci sequence divisible by m are equally spaced -/
theorem fib_divisibility_spacing (m : ℕ) (h : m > 0) :
  ∃ d : ℕ, d > 0 ∧ ∀ n : ℕ, m ∣ fib n → m ∣ fib (n + d) :=
sorry

end NUMINAMATH_CALUDE_fib_divisibility_spacing_l706_70616


namespace NUMINAMATH_CALUDE_black_squares_count_l706_70630

/-- Represents a checkerboard with side length n -/
structure Checkerboard (n : ℕ) where
  is_corner_black : Bool
  is_alternating : Bool

/-- Counts the number of black squares on a checkerboard -/
def count_black_squares (board : Checkerboard 33) : ℕ :=
  sorry

/-- Theorem: The number of black squares on a 33x33 alternating checkerboard with black corners is 545 -/
theorem black_squares_count : 
  ∀ (board : Checkerboard 33), 
  board.is_corner_black = true → 
  board.is_alternating = true → 
  count_black_squares board = 545 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_count_l706_70630


namespace NUMINAMATH_CALUDE_greatest_power_of_8_dividing_20_factorial_l706_70675

theorem greatest_power_of_8_dividing_20_factorial :
  (∃ n : ℕ+, 8^n.val ∣ Nat.factorial 20 ∧
    ∀ m : ℕ+, 8^m.val ∣ Nat.factorial 20 → m ≤ n) ∧
  (∃ n : ℕ+, n.val = 6 ∧ 8^n.val ∣ Nat.factorial 20 ∧
    ∀ m : ℕ+, 8^m.val ∣ Nat.factorial 20 → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_of_8_dividing_20_factorial_l706_70675


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l706_70620

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  let x := sum_of_integers 10 20
  let y := count_even_integers 10 20
  x + y = 171 := by sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l706_70620


namespace NUMINAMATH_CALUDE_gcf_of_120_180_240_l706_70653

theorem gcf_of_120_180_240 : Nat.gcd 120 (Nat.gcd 180 240) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_120_180_240_l706_70653


namespace NUMINAMATH_CALUDE_light_bulb_cost_exceeds_budget_l706_70697

/-- Represents the cost of light bulbs for Valerie's lamps --/
def light_bulb_cost : ℝ :=
  let small_cost : ℝ := 3 * 8.50
  let large_cost : ℝ := 1 * 14.25
  let medium_cost : ℝ := 2 * 10.75
  let extra_small_cost : ℝ := 4 * 6.25
  small_cost + large_cost + medium_cost + extra_small_cost

/-- Valerie's budget for light bulbs --/
def budget : ℝ := 80

/-- Theorem stating that the total cost of light bulbs exceeds Valerie's budget --/
theorem light_bulb_cost_exceeds_budget : light_bulb_cost > budget := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_cost_exceeds_budget_l706_70697


namespace NUMINAMATH_CALUDE_hexagon_arithmetic_progression_angle_l706_70662

theorem hexagon_arithmetic_progression_angle (a d : ℝ) :
  (∀ i : Fin 6, 0 ≤ i.val → i.val < 6 → 0 < a + i.val * d) →
  (6 * a + 15 * d = 720) →
  ∃ i : Fin 6, a + i.val * d = 240 :=
sorry

end NUMINAMATH_CALUDE_hexagon_arithmetic_progression_angle_l706_70662


namespace NUMINAMATH_CALUDE_distance_difference_l706_70649

/-- The width of a street in Simplifiedtown -/
def street_width : ℝ := 30

/-- The length of one side of a square block in Simplifiedtown -/
def block_side_length : ℝ := 400

/-- The distance Sarah runs from the block's inner edge -/
def sarah_distance : ℝ := 400

/-- The distance Maude runs from the block's inner edge -/
def maude_distance : ℝ := block_side_length + street_width

/-- The theorem stating the difference in distance run by Maude and Sarah -/
theorem distance_difference :
  4 * maude_distance - 4 * sarah_distance = 120 :=
sorry

end NUMINAMATH_CALUDE_distance_difference_l706_70649


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l706_70600

theorem geometric_sequence_inequality (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- a_n is a geometric sequence with common ratio q
  a 2 > a 1 →                   -- a₂ > a₁
  a 1 > 0 →                     -- a₁ > 0
  a 1 + a 3 > 2 * a 2 :=        -- prove: a₁ + a₃ > 2a₂
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l706_70600


namespace NUMINAMATH_CALUDE_angle_sum_identity_l706_70684

theorem angle_sum_identity (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 + 2 * Real.cos α * Real.cos β * Real.cos γ = 1 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_identity_l706_70684


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l706_70624

/-- The set of factors of 48 -/
def factors_of_48 : Set ℕ := {1, 2, 3, 4, 6, 8, 12, 16, 24, 48}

/-- Proposition: The smallest product of two distinct factors of 48 that is not a factor of 48 is 18 -/
theorem smallest_non_factor_product :
  ∃ (x y : ℕ), x ∈ factors_of_48 ∧ y ∈ factors_of_48 ∧ x ≠ y ∧ x * y ∉ factors_of_48 ∧
  x * y = 18 ∧ ∀ (a b : ℕ), a ∈ factors_of_48 → b ∈ factors_of_48 → a ≠ b →
  a * b ∉ factors_of_48 → a * b ≥ 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l706_70624


namespace NUMINAMATH_CALUDE_guilty_cases_count_l706_70628

theorem guilty_cases_count (total : ℕ) (dismissed : ℕ) (delayed : ℕ) : 
  total = 17 →
  dismissed = 2 →
  delayed = 1 →
  (total - dismissed - delayed - (2 * (total - dismissed) / 3)) = 4 := by
sorry

end NUMINAMATH_CALUDE_guilty_cases_count_l706_70628


namespace NUMINAMATH_CALUDE_largest_quantity_l706_70665

theorem largest_quantity (a b c d : ℝ) : 
  (a + 2 = b - 1) ∧ (b - 1 = c + 3) ∧ (c + 3 = d - 4) →
  (d > a) ∧ (d > b) ∧ (d > c) := by
sorry

end NUMINAMATH_CALUDE_largest_quantity_l706_70665


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_profit_function_concave_down_l706_70679

/-- Represents the profit function for a product sale scenario -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 280 * x - 1600

/-- Represents the optimal selling price -/
def optimal_price : ℝ := 14

/-- Represents the maximum profit -/
def max_profit : ℝ := 360

/-- Theorem stating that the optimal price maximizes the profit function -/
theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_price :=
sorry

/-- Theorem stating that the maximum profit is achieved at the optimal price -/
theorem max_profit_at_optimal_price :
  profit_function optimal_price = max_profit :=
sorry

/-- Theorem stating that the profit function is concave down -/
theorem profit_function_concave_down :
  ∀ x y t : ℝ, 0 ≤ t ∧ t ≤ 1 →
  profit_function (t * x + (1 - t) * y) ≥ t * profit_function x + (1 - t) * profit_function y :=
sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_profit_function_concave_down_l706_70679


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l706_70611

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l706_70611


namespace NUMINAMATH_CALUDE_randy_money_problem_l706_70640

theorem randy_money_problem (M : ℝ) : 
  M > 0 →
  (1/4 : ℝ) * (M - 10) = 5 →
  M = 30 := by
sorry

end NUMINAMATH_CALUDE_randy_money_problem_l706_70640


namespace NUMINAMATH_CALUDE_min_value_abc_l706_70622

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 3) :
  a^2 + 8*a*b + 24*b^2 + 16*b*c + 6*c^2 ≥ 54 ∧ 
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' * b' * c' = 3 ∧ 
    a'^2 + 8*a'*b' + 24*b'^2 + 16*b'*c' + 6*c'^2 = 54 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l706_70622


namespace NUMINAMATH_CALUDE_small_box_dimension_l706_70680

/-- Given a large rectangular box and smaller boxes, proves the dimensions of the smaller boxes. -/
theorem small_box_dimension (large_length large_width large_height : ℕ)
                             (small_length small_height : ℕ)
                             (max_boxes : ℕ)
                             (h1 : large_length = 12)
                             (h2 : large_width = 14)
                             (h3 : large_height = 16)
                             (h4 : small_length = 3)
                             (h5 : small_height = 2)
                             (h6 : max_boxes = 64) :
  ∃ (small_width : ℕ), small_width = 7 ∧
    max_boxes * (small_length * small_width * small_height) = 
    large_length * large_width * large_height :=
by sorry

end NUMINAMATH_CALUDE_small_box_dimension_l706_70680


namespace NUMINAMATH_CALUDE_three_number_problem_l706_70626

theorem three_number_problem (a b c : ℝ) 
  (sum_30 : a + b + c = 30)
  (first_twice_sum : a = 2 * (b + c))
  (second_five_third : b = 5 * c)
  (sum_first_third : a + c = 22) :
  a * b * c = 2500 / 9 := by
sorry

end NUMINAMATH_CALUDE_three_number_problem_l706_70626


namespace NUMINAMATH_CALUDE_y_intercept_not_z_l706_70637

/-- For a line ax + by - z = 0 where b ≠ 0, the y-intercept is not equal to z -/
theorem y_intercept_not_z (a b z : ℝ) (h : b ≠ 0) :
  ∃ (y_intercept : ℝ), y_intercept = z / b ∧ y_intercept ≠ z := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_not_z_l706_70637


namespace NUMINAMATH_CALUDE_increasing_sequence_condition_sufficient_condition_not_necessary_condition_l706_70643

theorem increasing_sequence_condition (k : ℝ) : 
  (∀ n : ℕ, n > 1 → (n^2 + k*n) < ((n+1)^2 + k*(n+1))) ↔ k > -3 :=
by sorry

theorem sufficient_condition (k : ℝ) :
  k ≥ -2 → ∀ n : ℕ, n > 1 → (n^2 + k*n) < ((n+1)^2 + k*(n+1)) :=
by sorry

theorem not_necessary_condition :
  ∃ k : ℝ, k < -2 ∧ (∀ n : ℕ, n > 1 → (n^2 + k*n) < ((n+1)^2 + k*(n+1))) :=
by sorry

end NUMINAMATH_CALUDE_increasing_sequence_condition_sufficient_condition_not_necessary_condition_l706_70643


namespace NUMINAMATH_CALUDE_inequality_solution_set_l706_70694

def solution_set (x : ℝ) : Prop := x < 1/3 ∨ x > 2

theorem inequality_solution_set :
  ∀ x : ℝ, (3*x - 1)/(x - 2) > 0 ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l706_70694


namespace NUMINAMATH_CALUDE_money_sharing_problem_l706_70657

/-- Represents the ratio of money distribution among three people -/
structure MoneyRatio :=
  (a : ℕ) (b : ℕ) (c : ℕ)

/-- Represents the money distribution among three people -/
structure MoneyDistribution :=
  (amanda : ℕ) (ben : ℕ) (carlos : ℕ)

/-- Theorem stating that given a money ratio of 2:3:8 and Amanda's share of $30, 
    the total amount shared is $195 -/
theorem money_sharing_problem 
  (ratio : MoneyRatio) 
  (dist : MoneyDistribution) :
  ratio.a = 2 ∧ ratio.b = 3 ∧ ratio.c = 8 ∧ 
  dist.amanda = 30 ∧
  dist.amanda * ratio.b = dist.ben * ratio.a ∧
  dist.amanda * ratio.c = dist.carlos * ratio.a →
  dist.amanda + dist.ben + dist.carlos = 195 :=
by sorry

end NUMINAMATH_CALUDE_money_sharing_problem_l706_70657


namespace NUMINAMATH_CALUDE_boys_playing_both_sports_l706_70638

theorem boys_playing_both_sports (total : ℕ) (basketball : ℕ) (football : ℕ) (neither : ℕ) :
  total = 30 →
  basketball = 18 →
  football = 21 →
  neither = 4 →
  basketball + football - (total - neither) = 13 := by
  sorry

end NUMINAMATH_CALUDE_boys_playing_both_sports_l706_70638


namespace NUMINAMATH_CALUDE_trioball_playing_time_l706_70627

theorem trioball_playing_time (num_children : ℕ) (game_duration : ℕ) (players_per_game : ℕ) :
  num_children = 3 →
  game_duration = 120 →
  players_per_game = 2 →
  ∃ (individual_time : ℕ),
    individual_time * num_children = players_per_game * game_duration ∧
    individual_time = 80 := by
  sorry

end NUMINAMATH_CALUDE_trioball_playing_time_l706_70627


namespace NUMINAMATH_CALUDE_valid_seating_count_l706_70654

/-- Number of seats in a row -/
def num_seats : ℕ := 7

/-- Number of people to be seated -/
def num_people : ℕ := 4

/-- Number of adjacent unoccupied seats -/
def num_adjacent_empty : ℕ := 2

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) (adjacent_empty : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid seating arrangements -/
theorem valid_seating_count :
  seating_arrangements num_seats num_people num_adjacent_empty = 336 :=
sorry

end NUMINAMATH_CALUDE_valid_seating_count_l706_70654


namespace NUMINAMATH_CALUDE_triangle_half_angle_sine_product_l706_70661

theorem triangle_half_angle_sine_product (A B C : ℝ) (h_triangle : A + B + C = π) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_half_angle_sine_product_l706_70661


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l706_70687

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l706_70687


namespace NUMINAMATH_CALUDE_builder_total_payment_l706_70699

/-- Calculates the total amount paid for a purchase of drill bits, hammers, and a toolbox with specific taxes and discounts. -/
def total_amount_paid (drill_bit_sets : ℕ) (drill_bit_price : ℚ) (drill_bit_tax : ℚ)
                      (hammers : ℕ) (hammer_price : ℚ) (hammer_discount : ℚ)
                      (toolbox_price : ℚ) (toolbox_tax : ℚ) : ℚ :=
  let drill_bits_cost := drill_bit_sets * drill_bit_price * (1 + drill_bit_tax)
  let hammers_cost := hammers * hammer_price * (1 - hammer_discount)
  let toolbox_cost := toolbox_price * (1 + toolbox_tax)
  drill_bits_cost + hammers_cost + toolbox_cost

/-- The total amount paid by the builder is $84.55. -/
theorem builder_total_payment :
  total_amount_paid 5 6 (10/100) 3 8 (5/100) 25 (15/100) = 8455/100 := by
  sorry

end NUMINAMATH_CALUDE_builder_total_payment_l706_70699


namespace NUMINAMATH_CALUDE_original_average_from_doubled_l706_70693

theorem original_average_from_doubled (n : ℕ) (A : ℚ) (h1 : n = 10) (h2 : 2 * A = 80) : A = 40 := by
  sorry

end NUMINAMATH_CALUDE_original_average_from_doubled_l706_70693


namespace NUMINAMATH_CALUDE_original_equals_scientific_l706_70618

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 274000000

/-- The scientific notation representation of the original number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 2.74
    exponent := 8
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l706_70618


namespace NUMINAMATH_CALUDE_fifth_invoice_number_l706_70639

/-- Represents the systematic sampling process for invoices -/
def systematicSampling (start : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  start + (n - 1) * interval

/-- Theorem stating that the fifth sampled invoice number is 215 -/
theorem fifth_invoice_number :
  systematicSampling 15 50 5 = 215 := by
  sorry

end NUMINAMATH_CALUDE_fifth_invoice_number_l706_70639


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l706_70668

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2 * Complex.I) / (1 + Complex.I) = Complex.mk a b → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l706_70668


namespace NUMINAMATH_CALUDE_problem_1_l706_70609

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

-- State the theorem
theorem problem_1 (m : ℝ) : (A.compl ∩ B m = ∅) → (m = 1 ∨ m = 2) :=
sorry

end NUMINAMATH_CALUDE_problem_1_l706_70609


namespace NUMINAMATH_CALUDE_sum_of_radii_l706_70636

/-- The sum of radii of circles tangent to x and y axes and externally tangent to a circle at (5,0) with radius 1.5 -/
theorem sum_of_radii : ∃ (r₁ r₂ : ℝ),
  r₁ > 0 ∧ r₂ > 0 ∧
  (r₁ - 5)^2 + r₁^2 = (r₁ + 1.5)^2 ∧
  (r₂ - 5)^2 + r₂^2 = (r₂ + 1.5)^2 ∧
  r₁ + r₂ = 13 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_radii_l706_70636


namespace NUMINAMATH_CALUDE_expression_evaluation_l706_70652

theorem expression_evaluation (c k : ℕ) (h1 : c = 4) (h2 : k = 2) :
  ((c^c - c*(c-1)^c + k)^c : ℕ) = 18974736 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l706_70652


namespace NUMINAMATH_CALUDE_work_rate_comparison_l706_70683

theorem work_rate_comparison (x : ℝ) (work : ℝ) : 
  x > 0 →
  (x + 1) * 21 = x * 28 →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_work_rate_comparison_l706_70683


namespace NUMINAMATH_CALUDE_initial_number_of_boys_l706_70646

theorem initial_number_of_boys (initial_girls : ℕ) (boys_dropped : ℕ) (girls_dropped : ℕ) (remaining_total : ℕ) : 
  initial_girls = 10 →
  boys_dropped = 4 →
  girls_dropped = 3 →
  remaining_total = 17 →
  ∃ initial_boys : ℕ, 
    initial_boys - boys_dropped + (initial_girls - girls_dropped) = remaining_total ∧
    initial_boys = 14 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_of_boys_l706_70646


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l706_70603

theorem smallest_integer_solution (m : ℚ) : 
  (∃ x : ℤ, (3 * (x + 1) - 2 ≤ 4 * (x - 3) + 1) ∧ 
    (∀ y : ℤ, 3 * (y + 1) - 2 ≤ 4 * (y - 3) + 1 → x ≤ y) ∧
    ((1 : ℚ) / 2 * x - m = 5)) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l706_70603


namespace NUMINAMATH_CALUDE_laptop_price_proof_l706_70692

/-- The original sticker price of a laptop -/
def sticker_price : ℝ := 1004

/-- The discount rate at store A -/
def discount_A : ℝ := 0.20

/-- The rebate amount at store A -/
def rebate_A : ℝ := 120

/-- The discount rate at store B -/
def discount_B : ℝ := 0.30

/-- The tax rate applied at both stores -/
def tax_rate : ℝ := 0.07

/-- The price difference between stores A and B -/
def price_difference : ℝ := 21

theorem laptop_price_proof :
  let price_A := (sticker_price * (1 - discount_A) - rebate_A) * (1 + tax_rate)
  let price_B := sticker_price * (1 - discount_B) * (1 + tax_rate)
  price_B - price_A = price_difference :=
by sorry

end NUMINAMATH_CALUDE_laptop_price_proof_l706_70692


namespace NUMINAMATH_CALUDE_problem_solution_l706_70650

theorem problem_solution (x : ℝ) : 
  3 - (1/4)*2 - (1/3)*3 - (1/7)*x = 27 → (10/100) * x = 17.85 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l706_70650


namespace NUMINAMATH_CALUDE_painting_time_equation_l706_70660

/-- Represents the time (in hours) it takes for a person to paint a room alone -/
structure PaintTime where
  hours : ℝ
  hours_positive : hours > 0

/-- Represents the painting scenario with Doug and Dave -/
structure PaintingScenario where
  doug_time : PaintTime
  dave_time : PaintTime
  doug_start_time : ℝ
  dave_join_time : ℝ
  total_time : ℝ
  doug_start_first : doug_start_time = 0
  dave_joins_later : dave_join_time > doug_start_time

/-- The main theorem stating the equation that the total painting time satisfies -/
theorem painting_time_equation (scenario : PaintingScenario) 
  (h1 : scenario.doug_time.hours = 3)
  (h2 : scenario.dave_time.hours = 4)
  (h3 : scenario.dave_join_time = 1) :
  (scenario.total_time - 1) * (7/12 : ℝ) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_equation_l706_70660


namespace NUMINAMATH_CALUDE_train_length_l706_70655

/-- The length of a train given its speed and the time it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * 1000 / 3600 →
  platform_length = 270 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 250 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l706_70655


namespace NUMINAMATH_CALUDE_yellow_highlighters_count_l706_70610

theorem yellow_highlighters_count (yellow pink blue : ℕ) : 
  pink = yellow + 7 →
  blue = pink + 5 →
  yellow + pink + blue = 40 →
  yellow = 7 := by
  sorry

end NUMINAMATH_CALUDE_yellow_highlighters_count_l706_70610


namespace NUMINAMATH_CALUDE_leaf_travel_11_gusts_l706_70645

/-- The net distance traveled by a leaf after a number of wind gusts -/
def leaf_travel (gusts : ℕ) (forward : ℕ) (backward : ℕ) : ℤ :=
  (gusts * forward : ℤ) - (gusts * backward : ℤ)

/-- Theorem: The leaf travels 33 feet after 11 gusts of wind -/
theorem leaf_travel_11_gusts :
  leaf_travel 11 5 2 = 33 := by
  sorry

end NUMINAMATH_CALUDE_leaf_travel_11_gusts_l706_70645


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l706_70691

theorem arithmetic_sequence_count : ∀ (a₁ d aₙ : ℝ) (n : ℕ),
  a₁ = 1.5 ∧ d = 4 ∧ aₙ = 45.5 ∧ aₙ = a₁ + (n - 1) * d →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l706_70691


namespace NUMINAMATH_CALUDE_eighteen_bottles_needed_l706_70613

/-- Calculates the minimum number of small bottles needed to fill a large bottle and a vase -/
def minimum_bottles (small_capacity : ℕ) (large_capacity : ℕ) (vase_capacity : ℕ) : ℕ :=
  let large_bottles := large_capacity / small_capacity
  let remaining_for_vase := vase_capacity
  let vase_bottles := (remaining_for_vase + small_capacity - 1) / small_capacity
  large_bottles + vase_bottles

/-- Theorem stating that 18 small bottles are needed to fill the large bottle and vase -/
theorem eighteen_bottles_needed :
  minimum_bottles 45 675 95 = 18 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_bottles_needed_l706_70613


namespace NUMINAMATH_CALUDE_quadratic_shift_l706_70695

/-- Represents a quadratic function of the form y = (x + a)² + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_horizontal (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a - d, b := f.b }

/-- Shifts a quadratic function vertically -/
def shift_vertical (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b - d }

/-- The main theorem stating that shifting y = (x + 1)² + 3 by 2 units right and 1 unit down
    results in y = (x - 1)² + 2 -/
theorem quadratic_shift :
  let f := QuadraticFunction.mk 1 3
  let g := shift_vertical (shift_horizontal f 2) 1
  g = QuadraticFunction.mk (-1) 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_shift_l706_70695


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l706_70696

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) : 
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 41 / 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l706_70696


namespace NUMINAMATH_CALUDE_store_price_reduction_l706_70634

theorem store_price_reduction (original_price : ℝ) (h_positive : original_price > 0) :
  let first_reduction := 0.12
  let final_percentage := 0.792
  let price_after_first := original_price * (1 - first_reduction)
  let second_reduction := 1 - (final_percentage / (1 - first_reduction))
  second_reduction = 0.1 := by
sorry

end NUMINAMATH_CALUDE_store_price_reduction_l706_70634


namespace NUMINAMATH_CALUDE_rosencrantz_win_probability_value_l706_70632

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the state of the game -/
inductive GameState
| InProgress
| RosencrantzWins
| GuildensternWins

/-- Represents the game rules -/
def game_rules : List CoinFlip → GameState :=
  sorry

/-- The probability of Rosencrantz winning the game -/
def rosencrantz_win_probability : ℚ :=
  sorry

/-- Theorem stating the probability of Rosencrantz winning -/
theorem rosencrantz_win_probability_value :
  rosencrantz_win_probability = (2^2009 - 1) / (3 * 2^2008 - 1) :=
sorry

end NUMINAMATH_CALUDE_rosencrantz_win_probability_value_l706_70632


namespace NUMINAMATH_CALUDE_game_points_proof_l706_70674

def points_earned (total_enemies : ℕ) (points_per_enemy : ℕ) (enemies_not_destroyed : ℕ) : ℕ :=
  (total_enemies - enemies_not_destroyed) * points_per_enemy

theorem game_points_proof :
  points_earned 7 8 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_game_points_proof_l706_70674


namespace NUMINAMATH_CALUDE_sunflower_height_l706_70648

/-- The height of sunflowers from Packet B in inches -/
def height_B : ℝ := 160

/-- The percentage difference between Packet A and Packet B sunflowers -/
def percentage_difference : ℝ := 0.2

/-- The height of sunflowers from Packet A in inches -/
def height_A : ℝ := height_B * (1 + percentage_difference)

theorem sunflower_height : height_A = 192 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_height_l706_70648


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l706_70641

/-- The constant term in the expansion of (2/x + x)^4 is 24 -/
theorem constant_term_binomial_expansion :
  let n : ℕ := 4
  let a : ℚ := 2
  let b : ℚ := 1
  (Nat.choose n (n / 2)) * a^(n / 2) * b^(n / 2) = 24 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l706_70641


namespace NUMINAMATH_CALUDE_perpendicular_equivalence_l706_70698

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line : Line → Line → Prop)

-- Define the intersection of planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_equivalence 
  (α β : Plane) (m n l : Line) 
  (h1 : perp_plane α β) 
  (h2 : intersect α β = l) 
  (h3 : subset m α) 
  (h4 : subset n β) : 
  perp_line m n ↔ (perp_line m l ∨ perp_line n l) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_equivalence_l706_70698


namespace NUMINAMATH_CALUDE_generalized_distributive_laws_l706_70644

variable {α : Type*}
variable {I : Type*}
variable (𝔍 : I → Type*)
variable (A : (i : I) → 𝔍 i → Set α)

def paths (𝔍 : I → Type*) := (i : I) → 𝔍 i

theorem generalized_distributive_laws :
  (⋃ i, ⋂ j, A i j) = (⋂ f : paths 𝔍, ⋃ i, A i (f i)) ∧
  (⋂ i, ⋃ j, A i j) = (⋃ f : paths 𝔍, ⋂ i, A i (f i)) :=
sorry

end NUMINAMATH_CALUDE_generalized_distributive_laws_l706_70644


namespace NUMINAMATH_CALUDE_certain_number_is_three_l706_70608

theorem certain_number_is_three (a b x : ℝ) 
  (h1 : 2 * a = x * b) 
  (h2 : a * b ≠ 0) 
  (h3 : (a / 3) / (b / 2) = 1) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_certain_number_is_three_l706_70608


namespace NUMINAMATH_CALUDE_exactly_two_girls_together_probability_l706_70651

/-- The number of boys in the group -/
def num_boys : ℕ := 2

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to arrange all students -/
def total_arrangements : ℕ := Nat.factorial total_students

/-- The number of ways to arrange students with exactly 2 girls together -/
def favorable_arrangements : ℕ := 
  Nat.choose 3 2 * Nat.factorial 2 * Nat.factorial 3

/-- The probability of exactly 2 out of 3 girls standing next to each other -/
def probability : ℚ := favorable_arrangements / total_arrangements

theorem exactly_two_girls_together_probability : 
  probability = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_exactly_two_girls_together_probability_l706_70651


namespace NUMINAMATH_CALUDE_vector_projection_and_magnitude_l706_70659

/-- Given vectors a and b in R², if the projection of a in its direction is -√2,
    then the second component of b is 4 and the magnitude of b is 2√5. -/
theorem vector_projection_and_magnitude (a b : ℝ × ℝ) :
  a = (1, -1) →
  b.1 = 2 →
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2) = -Real.sqrt 2 →
  b.2 = 4 ∧ Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_and_magnitude_l706_70659


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_l706_70635

/-- Represents the cost function for a caterer -/
structure CatererCost where
  basicFee : ℕ
  perPersonFee : ℕ

/-- Calculates the total cost for a caterer given the number of people -/
def totalCost (c : CatererCost) (people : ℕ) : ℕ :=
  c.basicFee + c.perPersonFee * people

/-- The first caterer's pricing model -/
def caterer1 : CatererCost := { basicFee := 150, perPersonFee := 18 }

/-- The second caterer's pricing model -/
def caterer2 : CatererCost := { basicFee := 250, perPersonFee := 14 }

/-- Theorem stating that for 26 or more people, the second caterer is less expensive -/
theorem second_caterer_cheaper (n : ℕ) (h : n ≥ 26) :
  totalCost caterer2 n < totalCost caterer1 n := by
  sorry


end NUMINAMATH_CALUDE_second_caterer_cheaper_l706_70635


namespace NUMINAMATH_CALUDE_crayons_in_boxes_l706_70672

/-- Given a number of crayons per box and a number of boxes, 
    calculate the total number of crayons -/
def total_crayons (crayons_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  crayons_per_box * num_boxes

/-- Theorem stating that with 8 crayons per box and 10 boxes, 
    the total number of crayons is 80 -/
theorem crayons_in_boxes : total_crayons 8 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_crayons_in_boxes_l706_70672


namespace NUMINAMATH_CALUDE_remainder_1997_pow_2000_mod_7_l706_70625

theorem remainder_1997_pow_2000_mod_7 : 1997^2000 % 7 = 4 := by sorry

end NUMINAMATH_CALUDE_remainder_1997_pow_2000_mod_7_l706_70625


namespace NUMINAMATH_CALUDE_parallel_lines_intersection_l706_70623

/-- Given 9 parallel lines intersected by n parallel lines forming 1008 parallelograms, n must equal 127 -/
theorem parallel_lines_intersection (n : ℕ) : 
  (9 - 1) * (n - 1) = 1008 → n = 127 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_intersection_l706_70623


namespace NUMINAMATH_CALUDE_absolute_value_of_z_l706_70678

theorem absolute_value_of_z (z z₀ : ℂ) : 
  z₀ = 3 + Complex.I ∧ z * z₀ = 3 * z + z₀ → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_l706_70678


namespace NUMINAMATH_CALUDE_min_cube_volume_for_pyramid_l706_70605

/-- Represents a square-based pyramid -/
structure Pyramid where
  height : ℝ
  baseLength : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.sideLength ^ 3

/-- Checks if a pyramid fits inside a cube -/
def pyramidFitsInCube (p : Pyramid) (c : Cube) : Prop :=
  c.sideLength ≥ p.height ∧ c.sideLength ≥ p.baseLength

theorem min_cube_volume_for_pyramid (p : Pyramid) (h1 : p.height = 18) (h2 : p.baseLength = 15) :
  ∃ (c : Cube), pyramidFitsInCube p c ∧ cubeVolume c = 5832 ∧
  ∀ (c' : Cube), pyramidFitsInCube p c' → cubeVolume c' ≥ 5832 := by
  sorry

end NUMINAMATH_CALUDE_min_cube_volume_for_pyramid_l706_70605


namespace NUMINAMATH_CALUDE_john_payment_is_1200_l706_70671

/-- Calculates John's payment for renting a camera -/
def johnPayment (cameraValue : ℝ) (rentalRatePerWeek : ℝ) (rentalWeeks : ℕ) (friendContributionRate : ℝ) : ℝ :=
  let totalRental := cameraValue * rentalRatePerWeek * rentalWeeks
  let friendContribution := totalRental * friendContributionRate
  totalRental - friendContribution

/-- Theorem stating that John's payment is $1200 given the problem conditions -/
theorem john_payment_is_1200 :
  johnPayment 5000 0.1 4 0.4 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_john_payment_is_1200_l706_70671


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l706_70617

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (a₁ d : ℝ) 
  (h_arith : a = arithmeticSequence a₁ d)
  (h_first : a 1 = 5)
  (h_sum : a 6 + a 8 = 58) :
  d = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l706_70617


namespace NUMINAMATH_CALUDE_ratio_of_linear_system_l706_70664

theorem ratio_of_linear_system (x y c d : ℝ) (h1 : 3 * x + 2 * y = c) (h2 : 4 * y - 6 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_linear_system_l706_70664


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l706_70642

theorem min_value_of_sum_of_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b^2) + (b^2 / c^2) + (c^2 / a^2) ≥ 3 ∧
  ((a^2 / b^2) + (b^2 / c^2) + (c^2 / a^2) = 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l706_70642


namespace NUMINAMATH_CALUDE_probability_less_than_5_is_17_18_l706_70619

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point (x,y) in the given square satisfies x + y < 5 --/
def probabilityLessThan5 (s : Square) : ℝ :=
  sorry

/-- The specific square with vertices (0,0), (0,3), (3,3), and (3,0) --/
def specificSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

theorem probability_less_than_5_is_17_18 :
  probabilityLessThan5 specificSquare = 17 / 18 :=
sorry

end NUMINAMATH_CALUDE_probability_less_than_5_is_17_18_l706_70619


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l706_70602

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∀ x y : ℝ, x^2 + y^2 = r^2 → (x + y = r + 1 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      (x' - x)^2 + (y' - y)^2 < δ^2 → 
      ((x'^2 + y'^2 - r^2) * ((x' + y') - (r + 1)) ≥ 0))) →
  r = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l706_70602


namespace NUMINAMATH_CALUDE_problem_statement_l706_70656

noncomputable section

variables (a : ℝ) (x x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := x^2 - a*x
def g (x : ℝ) : ℝ := Real.log x
def h (x : ℝ) : ℝ := f a x + g x

theorem problem_statement :
  (∀ x > 0, f a x ≥ g x) ↔ a ≤ 1 ∧
  ∃ m : ℝ, m = 3/4 - Real.log 2 ∧
    (0 < x₁ ∧ x₁ < 1/2 ∧ 
     h a x₁ - h a x₂ > m ∧
     (∀ m' : ℝ, h a x₁ - h a x₂ > m' → m' ≤ m)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l706_70656


namespace NUMINAMATH_CALUDE_journey_distance_l706_70670

/-- Represents the journey from John's house to the conference center -/
structure Journey where
  initial_speed : ℝ             -- Initial speed in miles per hour
  initial_distance : ℝ          -- Distance covered in the first hour
  late_time : ℝ                 -- Time he would be late if continued at initial speed
  speed_increase : ℝ            -- Increase in speed for the rest of the journey
  early_time : ℝ                -- Time he arrives early after increasing speed

/-- Calculates the total distance of the journey -/
def calculate_distance (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that the distance to the conference center is 191.25 miles -/
theorem journey_distance (j : Journey) 
  (h1 : j.initial_speed = 45)
  (h2 : j.initial_distance = 45)
  (h3 : j.late_time = 0.75)
  (h4 : j.speed_increase = 20)
  (h5 : j.early_time = 0.25) :
  calculate_distance j = 191.25 :=
sorry

end NUMINAMATH_CALUDE_journey_distance_l706_70670


namespace NUMINAMATH_CALUDE_negative_three_below_zero_l706_70629

/-- Represents temperature in Celsius -/
structure Temperature where
  value : ℝ
  unit : String

/-- Defines the concept of opposite meanings for temperatures -/
def oppositeMeaning (t1 t2 : Temperature) : Prop :=
  t1.value = -t2.value ∧ t1.unit = t2.unit

/-- Axiom: If two numbers have opposite meanings, they are respectively called positive and negative -/
axiom positive_negative_opposite (t1 t2 : Temperature) :
  oppositeMeaning t1 t2 → (t1.value > 0 ↔ t2.value < 0)

/-- Given: +10°C represents a temperature of 10°C above zero -/
axiom positive_ten_above_zero :
  ∃ (t : Temperature), t.value = 10 ∧ t.unit = "°C"

/-- Theorem: -3°C represents a temperature of 3°C below zero -/
theorem negative_three_below_zero :
  ∃ (t : Temperature), t.value = -3 ∧ t.unit = "°C" ∧
  ∃ (t_pos : Temperature), oppositeMeaning t t_pos ∧ t_pos.value = 3 :=
sorry

end NUMINAMATH_CALUDE_negative_three_below_zero_l706_70629


namespace NUMINAMATH_CALUDE_ant_final_position_l706_70633

/-- Represents the position of the ant on a 2D plane -/
structure Position where
  x : Int
  y : Int

/-- Represents the direction the ant is facing -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents the state of the ant at any given moment -/
structure AntState where
  pos : Position
  dir : Direction
  moveCount : Nat

/-- The movement function for the ant -/
def move (state : AntState) : AntState :=
  sorry

/-- The main theorem stating the final position of the ant -/
theorem ant_final_position :
  let initial_state : AntState :=
    { pos := { x := -25, y := 25 }
    , dir := Direction.North
    , moveCount := 0
    }
  let final_state := (move^[1010]) initial_state
  final_state.pos = { x := 1491, y := -481 } :=
sorry

end NUMINAMATH_CALUDE_ant_final_position_l706_70633


namespace NUMINAMATH_CALUDE_squats_calculation_l706_70688

/-- 
Proves that if the number of squats increases by 5 each day for four consecutive days, 
and 45 squats are performed on the fourth day, then 30 squats were performed on the first day.
-/
theorem squats_calculation (initial_squats : ℕ) : 
  (∀ (day : ℕ), day < 4 → initial_squats + 5 * day = initial_squats + day * 5) →
  initial_squats + 5 * 3 = 45 →
  initial_squats = 30 := by
  sorry

end NUMINAMATH_CALUDE_squats_calculation_l706_70688


namespace NUMINAMATH_CALUDE_g_increasing_g_geq_h_condition_l706_70667

noncomputable section

-- Define the functions g and h
def g (a : ℝ) (x : ℝ) : ℝ := a * x - a / x - 5 * Real.log x
def h (m : ℝ) (x : ℝ) : ℝ := x^2 - m * x + 4

-- Theorem 1: g(x) is increasing when a > 5/2
theorem g_increasing (a : ℝ) : 
  (∀ x > 0, ∀ y > 0, x < y → g a x < g a y) ↔ a > 5/2 :=
sorry

-- Theorem 2: Condition for g(x₁) ≥ h(x₂) when a = 2
theorem g_geq_h_condition (m : ℝ) :
  (∃ x₁ ∈ Set.Ioo 0 1, ∀ x₂ ∈ Set.Icc 1 2, g 2 x₁ ≥ h m x₂) ↔ 
  m ≥ 8 - 5 * Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_g_increasing_g_geq_h_condition_l706_70667


namespace NUMINAMATH_CALUDE_square_area_on_parabola_and_line_square_area_on_parabola_and_line_is_36_l706_70615

theorem square_area_on_parabola_and_line : ℝ → Prop :=
  fun area =>
    ∃ (x₁ x₂ : ℝ),
      -- The endpoints lie on the parabola y = x^2 + 4x + 3
      8 = x₁^2 + 4*x₁ + 3 ∧
      8 = x₂^2 + 4*x₂ + 3 ∧
      -- The side length is the absolute difference between x-coordinates
      area = (x₁ - x₂)^2 ∧
      -- The area of the square is 36
      area = 36

-- The proof of the theorem
theorem square_area_on_parabola_and_line_is_36 :
  square_area_on_parabola_and_line 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_and_line_square_area_on_parabola_and_line_is_36_l706_70615


namespace NUMINAMATH_CALUDE_initial_men_is_50_l706_70658

/-- Represents a road construction project -/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℝ
  completedLength : ℝ
  completedDays : ℝ
  extraMen : ℕ

/-- Calculates the initial number of men for a given road project -/
def initialMen (project : RoadProject) : ℕ :=
  sorry

/-- The theorem stating that for the given project conditions, the initial number of men is 50 -/
theorem initial_men_is_50 (project : RoadProject) 
  (h1 : project.totalLength = 15)
  (h2 : project.totalDays = 300)
  (h3 : project.completedLength = 2.5)
  (h4 : project.completedDays = 100)
  (h5 : project.extraMen = 75)
  : initialMen project = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_is_50_l706_70658


namespace NUMINAMATH_CALUDE_square_sum_equals_90_l706_70681

theorem square_sum_equals_90 (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_90_l706_70681
