import Mathlib

namespace NUMINAMATH_CALUDE_intersection_values_l1278_127892

-- Define the complex plane
variable (z : ℂ)

-- Define the equation |z - 4| = 3|z + 4|
def equation (z : ℂ) : Prop := Complex.abs (z - 4) = 3 * Complex.abs (z + 4)

-- Define the intersection condition
def intersects_once (k : ℝ) : Prop :=
  ∃! z, equation z ∧ Complex.abs z = k

-- Theorem statement
theorem intersection_values :
  ∀ k, intersects_once k → k = 2 ∨ k = 14 :=
sorry

end NUMINAMATH_CALUDE_intersection_values_l1278_127892


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l1278_127846

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon is a 12-sided polygon -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l1278_127846


namespace NUMINAMATH_CALUDE_profit_per_meter_l1278_127802

/-- Given the selling price and cost price of cloth, calculate the profit per meter -/
theorem profit_per_meter (total_meters : ℕ) (selling_price cost_per_meter : ℚ) :
  total_meters = 85 →
  selling_price = 8925 →
  cost_per_meter = 85 →
  (selling_price - total_meters * cost_per_meter) / total_meters = 20 := by
sorry

end NUMINAMATH_CALUDE_profit_per_meter_l1278_127802


namespace NUMINAMATH_CALUDE_student_count_l1278_127833

/-- Given a group of students where replacing one student changes the average weight,
    this theorem proves the total number of students. -/
theorem student_count
  (avg_decrease : ℝ)  -- The decrease in average weight
  (old_weight : ℝ)    -- Weight of the replaced student
  (new_weight : ℝ)    -- Weight of the new student
  (h1 : avg_decrease = 5)  -- The average weight decreases by 5 kg
  (h2 : old_weight = 86)   -- The replaced student weighs 86 kg
  (h3 : new_weight = 46)   -- The new student weighs 46 kg
  : ℕ := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1278_127833


namespace NUMINAMATH_CALUDE_cubic_expression_value_l1278_127800

theorem cubic_expression_value (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2*x^2 - 12 = -11 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l1278_127800


namespace NUMINAMATH_CALUDE_cement_total_l1278_127894

theorem cement_total (bought : ℕ) (brought : ℕ) (total : ℕ) : 
  bought = 215 → brought = 137 → total = bought + brought → total = 352 := by
  sorry

end NUMINAMATH_CALUDE_cement_total_l1278_127894


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1278_127855

theorem inequality_equivalence (x : ℝ) : (x - 3) / (x^2 + 2*x + 7) ≥ 0 ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1278_127855


namespace NUMINAMATH_CALUDE_number_of_men_in_first_group_l1278_127885

-- Define the number of men in the first group
def M : ℕ := sorry

-- Define the given conditions
def hours_per_day_group1 : ℕ := 10
def earnings_per_week_group1 : ℕ := 1000
def men_group2 : ℕ := 9
def hours_per_day_group2 : ℕ := 6
def earnings_per_week_group2 : ℕ := 1350
def days_per_week : ℕ := 7

-- Theorem to prove
theorem number_of_men_in_first_group :
  (M * hours_per_day_group1 * days_per_week) / earnings_per_week_group1 =
  (men_group2 * hours_per_day_group2 * days_per_week) / earnings_per_week_group2 →
  M = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_in_first_group_l1278_127885


namespace NUMINAMATH_CALUDE_expression_evaluation_l1278_127830

theorem expression_evaluation (a b c : ℝ) (ha : a = 7) (hb : b = 11) (hc : c = 13) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1278_127830


namespace NUMINAMATH_CALUDE_stable_performance_comparison_l1278_127824

/-- Represents a student's performance in standing long jumps --/
structure StudentPerformance where
  average_score : ℝ
  variance : ℝ

/-- Determines if a student's performance is more stable --/
def more_stable (a b : StudentPerformance) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two students with the same average score, 
    the one with lower variance has more stable performance --/
theorem stable_performance_comparison 
  (student_a student_b : StudentPerformance)
  (h_same_average : student_a.average_score = student_b.average_score)
  (h_a_variance : student_a.variance = 0.48)
  (h_b_variance : student_b.variance = 0.53) :
  more_stable student_a student_b :=
by
  sorry

end NUMINAMATH_CALUDE_stable_performance_comparison_l1278_127824


namespace NUMINAMATH_CALUDE_lee_family_concert_cost_is_86_l1278_127844

/-- Represents the cost calculation for the Lee family concert tickets --/
def lee_family_concert_cost : ℝ :=
  let regular_ticket_cost : ℝ := 10
  let booking_fee : ℝ := 1.5
  let youngest_discount : ℝ := 0.4
  let oldest_discount : ℝ := 0.3
  let middle_discount : ℝ := 0.2
  let youngest_count : ℕ := 3
  let oldest_count : ℕ := 3
  let middle_count : ℕ := 4
  let total_tickets : ℕ := youngest_count + oldest_count + middle_count

  let youngest_cost : ℝ := youngest_count * (regular_ticket_cost * (1 - youngest_discount))
  let oldest_cost : ℝ := oldest_count * (regular_ticket_cost * (1 - oldest_discount))
  let middle_cost : ℝ := middle_count * (regular_ticket_cost * (1 - middle_discount))
  
  let total_ticket_cost : ℝ := youngest_cost + oldest_cost + middle_cost
  let total_booking_fees : ℝ := total_tickets * booking_fee

  total_ticket_cost + total_booking_fees

/-- Theorem stating that the total cost for the Lee family concert tickets is $86.00 --/
theorem lee_family_concert_cost_is_86 : lee_family_concert_cost = 86 := by
  sorry

end NUMINAMATH_CALUDE_lee_family_concert_cost_is_86_l1278_127844


namespace NUMINAMATH_CALUDE_power_of_negative_product_l1278_127847

theorem power_of_negative_product (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_product_l1278_127847


namespace NUMINAMATH_CALUDE_production_equation_l1278_127850

/-- Represents the production of machines in a factory --/
structure MachineProduction where
  x : ℝ  -- Actual number of machines produced per day
  original_plan : ℝ  -- Original planned production per day
  increased_production : ℝ  -- Increase in production per day
  time_500 : ℝ  -- Time to produce 500 machines at current rate
  time_300 : ℝ  -- Time to produce 300 machines at original rate

/-- Theorem stating the relationship between production rates and times --/
theorem production_equation (mp : MachineProduction) 
  (h1 : mp.x = mp.original_plan + mp.increased_production)
  (h2 : mp.increased_production = 20)
  (h3 : mp.time_500 = 500 / mp.x)
  (h4 : mp.time_300 = 300 / mp.original_plan)
  (h5 : mp.time_500 = mp.time_300) :
  500 / mp.x = 300 / (mp.x - 20) := by
  sorry

end NUMINAMATH_CALUDE_production_equation_l1278_127850


namespace NUMINAMATH_CALUDE_total_rubber_bands_l1278_127806

theorem total_rubber_bands (harper_bands : ℕ) (difference : ℕ) : 
  harper_bands = 15 → difference = 6 → harper_bands + (harper_bands - difference) = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_rubber_bands_l1278_127806


namespace NUMINAMATH_CALUDE_f_is_linear_l1278_127868

/-- A function representing the total price of masks based on quantity -/
def f (x : ℝ) : ℝ := 0.9 * x

/-- The unit price of a mask in yuan -/
def unit_price : ℝ := 0.9

/-- Theorem stating that f is a linear function -/
theorem f_is_linear : 
  ∃ (m b : ℝ), ∀ x, f x = m * x + b :=
sorry

end NUMINAMATH_CALUDE_f_is_linear_l1278_127868


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1278_127828

theorem geometric_sequence_first_term
  (a : ℕ → ℕ)
  (h1 : a 2 = 3)
  (h2 : a 3 = 9)
  (h3 : a 4 = 27)
  (h4 : a 5 = 81)
  (h5 : a 6 = 243)
  (h_geometric : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n) :
  a 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1278_127828


namespace NUMINAMATH_CALUDE_four_Y_three_equals_negative_eleven_l1278_127812

/-- The Y operation defined for any two real numbers -/
def Y (x y : ℝ) : ℝ := x^2 - 3*x*y + y^2

/-- Theorem stating that 4 Y 3 equals -11 -/
theorem four_Y_three_equals_negative_eleven : Y 4 3 = -11 := by
  sorry

end NUMINAMATH_CALUDE_four_Y_three_equals_negative_eleven_l1278_127812


namespace NUMINAMATH_CALUDE_simplify_expression_l1278_127817

theorem simplify_expression (x : ℝ) : 3 * x^5 * (4 * x^3) = 12 * x^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1278_127817


namespace NUMINAMATH_CALUDE_max_value_negative_x_min_value_greater_than_negative_one_l1278_127860

-- Problem 1
theorem max_value_negative_x (x : ℝ) (hx : x < 0) :
  (x^2 + x + 1) / x ≤ -1 :=
sorry

-- Problem 2
theorem min_value_greater_than_negative_one (x : ℝ) (hx : x > -1) :
  ((x + 5) * (x + 2)) / (x + 1) ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_negative_x_min_value_greater_than_negative_one_l1278_127860


namespace NUMINAMATH_CALUDE_original_class_size_l1278_127829

theorem original_class_size (A B C : ℕ) (N : ℕ) (D : ℕ) :
  A = 40 →
  B = 32 →
  C = 36 →
  D = N * A →
  D + 8 * B = (N + 8) * C →
  N = 8 := by
sorry

end NUMINAMATH_CALUDE_original_class_size_l1278_127829


namespace NUMINAMATH_CALUDE_max_value_condition_l1278_127866

/-- The function f(x) = kx^2 + kx + 1 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + k * x + 1

/-- The maximum value of f(x) on the interval [-2, 2] is 4 -/
def has_max_4 (k : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-2) 2, f k x ≤ 4 ∧ ∃ y ∈ Set.Icc (-2) 2, f k y = 4

/-- The theorem stating that k = 1/2 or k = -12 if and only if
    the maximum value of f(x) on [-2, 2] is 4 -/
theorem max_value_condition (k : ℝ) :
  has_max_4 k ↔ k = 1/2 ∨ k = -12 := by sorry

end NUMINAMATH_CALUDE_max_value_condition_l1278_127866


namespace NUMINAMATH_CALUDE_triangle_area_is_one_l1278_127841

-- Define the complex number z on the unit circle
def z : ℂ :=
  sorry

-- Define the condition |z| = 1
axiom z_on_unit_circle : Complex.abs z = 1

-- Define the vertices of the triangle
def vertex1 : ℂ := z
def vertex2 : ℂ := z^2
def vertex3 : ℂ := z + z^2

-- Define the function to calculate the area of a triangle given three complex points
def triangle_area (a b c : ℂ) : ℝ :=
  sorry

-- State the theorem
theorem triangle_area_is_one :
  triangle_area vertex1 vertex2 vertex3 = 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_is_one_l1278_127841


namespace NUMINAMATH_CALUDE_sum_with_rearrangement_not_1999_nines_sum_with_rearrangement_1010_divisible_by_10_l1278_127826

-- Define a function to represent digit rearrangement
def digitRearrangement (n : ℕ) : ℕ := sorry

-- Define a function to check if a number consists of 1999 nines
def is1999Nines (n : ℕ) : Prop := sorry

-- Part (a)
theorem sum_with_rearrangement_not_1999_nines (n : ℕ) : 
  ¬(is1999Nines (n + digitRearrangement n)) := sorry

-- Part (b)
theorem sum_with_rearrangement_1010_divisible_by_10 (n : ℕ) : 
  n + digitRearrangement n = 1010 → n % 10 = 0 := sorry

end NUMINAMATH_CALUDE_sum_with_rearrangement_not_1999_nines_sum_with_rearrangement_1010_divisible_by_10_l1278_127826


namespace NUMINAMATH_CALUDE_train_length_train_length_approx_l1278_127831

/-- The length of a train given its speed and time to cross a post -/
theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) : ℝ :=
  let speed_m_s := speed_km_hr * 1000 / 3600
  speed_m_s * time_seconds

/-- Theorem stating that a train with speed 40 km/hr crossing a post in 25.2 seconds has a length of approximately 280 meters -/
theorem train_length_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |train_length 40 25.2 - 280| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_approx_l1278_127831


namespace NUMINAMATH_CALUDE_angle_extrema_l1278_127897

/-- The angle formed by the construction described in the problem -/
def constructionAngle (x : Fin n → ℝ) : ℝ :=
  sorry

/-- The theorem stating that the angle is minimal for descending sequences and maximal for ascending sequences -/
theorem angle_extrema (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, x i > 0) :
  (∀ i j, i < j → x i ≥ x j) →
  (∀ y : Fin n → ℝ, (∀ i, y i > 0) → constructionAngle x ≤ constructionAngle y) ∧
  (∀ i j, i < j → x i ≤ x j) →
  (∀ y : Fin n → ℝ, (∀ i, y i > 0) → constructionAngle x ≥ constructionAngle y) :=
sorry

end NUMINAMATH_CALUDE_angle_extrema_l1278_127897


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l1278_127805

def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

theorem max_min_values_of_f :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = min) ∧
    max = 4 ∧ min = -5 :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l1278_127805


namespace NUMINAMATH_CALUDE_infinite_solutions_ratio_l1278_127837

theorem infinite_solutions_ratio (a b c : ℚ) : 
  (∀ x, a * x^2 + b * x + c = (x - 1) * (2 * x + 1)) → 
  a = 2 ∧ b = -1 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_ratio_l1278_127837


namespace NUMINAMATH_CALUDE_product_of_numbers_l1278_127836

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 22) (h2 : x^2 + y^2 = 404) : x * y = 40 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1278_127836


namespace NUMINAMATH_CALUDE_flowers_picked_l1278_127854

/-- Proves that if a person can make 7 bouquets with 8 flowers each after 10 flowers have wilted,
    then they initially picked 66 flowers. -/
theorem flowers_picked (bouquets : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) :
  bouquets = 7 →
  flowers_per_bouquet = 8 →
  wilted_flowers = 10 →
  bouquets * flowers_per_bouquet + wilted_flowers = 66 :=
by sorry

end NUMINAMATH_CALUDE_flowers_picked_l1278_127854


namespace NUMINAMATH_CALUDE_min_value_theorem_l1278_127864

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1/2) :
  (4/a + 1/b) ≥ 18 := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1278_127864


namespace NUMINAMATH_CALUDE_smallest_a1_l1278_127887

theorem smallest_a1 (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_rec : ∀ n > 1, a n = 7 * a (n - 1) - 2 * n) :
  (∀ a₁ : ℝ, (∀ n, a n > 0) → (∀ n > 1, a n = 7 * a (n - 1) - 2 * n) → a₁ ≥ a 1) →
  a 1 = 13 / 18 := by
sorry

end NUMINAMATH_CALUDE_smallest_a1_l1278_127887


namespace NUMINAMATH_CALUDE_range_of_f_range_of_m_l1278_127881

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1| - 1
def g (x : ℝ) : ℝ := -|x + 1| - 4

-- Theorem 1: Range of x for which f(x) ≤ 1
theorem range_of_f (x : ℝ) : f x ≤ 1 ↔ x ∈ Set.Icc (-1) 3 := by sorry

-- Theorem 2: Range of m for which f(x) - g(x) ≥ m + 1 holds for all x
theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ∈ Set.Iic 4 := by sorry

end NUMINAMATH_CALUDE_range_of_f_range_of_m_l1278_127881


namespace NUMINAMATH_CALUDE_emily_team_score_l1278_127813

theorem emily_team_score (total_players : ℕ) (emily_score : ℕ) (other_player_score : ℕ) : 
  total_players = 8 →
  emily_score = 23 →
  other_player_score = 2 →
  emily_score + (total_players - 1) * other_player_score = 37 := by
  sorry

end NUMINAMATH_CALUDE_emily_team_score_l1278_127813


namespace NUMINAMATH_CALUDE_nickels_maximize_value_expected_value_is_7480_l1278_127814

/-- Represents the types of coins --/
inductive Coin
| Quarter
| Nickel
| Dime

/-- Represents the material of a coin --/
inductive Material
| Regular
| Iron

/-- The number of quarters Alice has --/
def initial_quarters : ℕ := 20

/-- The exchange rate for quarters to nickels --/
def quarters_to_nickels : ℕ := 4

/-- The exchange rate for quarters to dimes --/
def quarters_to_dimes : ℕ := 2

/-- The probability of a nickel being iron --/
def iron_nickel_prob : ℚ := 3/10

/-- The probability of a dime being iron --/
def iron_dime_prob : ℚ := 1/10

/-- The value of an iron nickel in cents --/
def iron_nickel_value : ℕ := 300

/-- The value of an iron dime in cents --/
def iron_dime_value : ℕ := 500

/-- The value of a regular nickel in cents --/
def regular_nickel_value : ℕ := 5

/-- The value of a regular dime in cents --/
def regular_dime_value : ℕ := 10

/-- Calculates the expected value of a nickel in cents --/
def expected_nickel_value : ℚ :=
  iron_nickel_prob * iron_nickel_value + (1 - iron_nickel_prob) * regular_nickel_value

/-- Calculates the expected value of a dime in cents --/
def expected_dime_value : ℚ :=
  iron_dime_prob * iron_dime_value + (1 - iron_dime_prob) * regular_dime_value

/-- Theorem stating that exchanging for nickels maximizes expected value --/
theorem nickels_maximize_value :
  expected_nickel_value * quarters_to_nickels > expected_dime_value * quarters_to_dimes :=
sorry

/-- Calculates the total number of nickels Alice can get --/
def total_nickels : ℕ := initial_quarters * quarters_to_nickels

/-- Calculates the expected total value in cents after exchanging for nickels --/
def expected_total_value : ℚ := total_nickels * expected_nickel_value

/-- Theorem stating that the expected total value is 7480 cents ($74.80) --/
theorem expected_value_is_7480 : expected_total_value = 7480 := sorry

end NUMINAMATH_CALUDE_nickels_maximize_value_expected_value_is_7480_l1278_127814


namespace NUMINAMATH_CALUDE_num_triangles_on_circle_l1278_127871

/-- The number of ways to choose k items from n items. -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of points on the circle. -/
def numPoints : ℕ := 10

/-- The number of points needed to form a triangle. -/
def pointsPerTriangle : ℕ := 3

/-- Theorem: The number of triangles that can be formed from 10 points on a circle is 120. -/
theorem num_triangles_on_circle :
  binomial numPoints pointsPerTriangle = 120 := by
  sorry

end NUMINAMATH_CALUDE_num_triangles_on_circle_l1278_127871


namespace NUMINAMATH_CALUDE_subtraction_with_division_l1278_127820

theorem subtraction_with_division : 5100 - (102 / 20.4) = 5095 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_with_division_l1278_127820


namespace NUMINAMATH_CALUDE_problem_statement_l1278_127876

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b + 2 * a + b = 16) :
  (∀ x y : ℝ, x > 0 → y > 0 → x * y + 2 * x + y = 16 → a * b ≥ x * y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x * y + 2 * x + y = 16 → 2 * a + b ≤ 2 * x + y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x * y + 2 * x + y = 16 → 1 / (a + 1) + 1 / (b + 2) ≤ 1 / (x + 1) + 1 / (y + 2)) ∧
  (a * b = 8 ∨ 2 * a + b = 8 ∨ 1 / (a + 1) + 1 / (b + 2) = Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1278_127876


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_positive_square_plus_x_l1278_127865

theorem negation_of_forall_positive (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) := by sorry

theorem negation_of_positive_square_plus_x :
  (¬ ∀ x > 0, x^2 + x > 0) ↔ (∃ x > 0, x^2 + x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_positive_square_plus_x_l1278_127865


namespace NUMINAMATH_CALUDE_inequality_range_l1278_127870

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x > 2*a*x + a) ↔ a ∈ Set.Ioo (-4 : ℝ) (-1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l1278_127870


namespace NUMINAMATH_CALUDE_rotation_180_maps_points_l1278_127801

/-- Rotation of 180° clockwise about the origin in 2D plane -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotation_180_maps_points :
  let C : ℝ × ℝ := (-3, 2)
  let D : ℝ × ℝ := (-2, 5)
  let C' : ℝ × ℝ := (3, -2)
  let D' : ℝ × ℝ := (2, -5)
  rotate180 C = C' ∧ rotate180 D = D' :=
by sorry

end NUMINAMATH_CALUDE_rotation_180_maps_points_l1278_127801


namespace NUMINAMATH_CALUDE_solve_equation_l1278_127834

theorem solve_equation (x : ℝ) : 3 * x + 12 = (1/3) * (7 * x + 42) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1278_127834


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1278_127877

theorem roots_quadratic_equation (m p q c : ℝ) (a b : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + c/b)^2 - p*(a + c/b) + q = 0) →
  ((b + c/a)^2 - p*(b + c/a) + q = 0) →
  (q = 3 + 2*c + c^2/3) :=
by sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1278_127877


namespace NUMINAMATH_CALUDE_larger_number_proof_l1278_127807

theorem larger_number_proof (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  Nat.gcd a b = 84 → Nat.lcm a b = 21 → 4 * a = b → b = 84 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1278_127807


namespace NUMINAMATH_CALUDE_decagon_perimeter_30_l1278_127803

/-- A regular decagon is a polygon with 10 sides of equal length. -/
structure RegularDecagon where
  side_length : ℝ
  sides : Nat
  sides_eq : sides = 10

/-- The perimeter of a polygon is the sum of the lengths of its sides. -/
def perimeter (d : RegularDecagon) : ℝ := d.side_length * d.sides

theorem decagon_perimeter_30 (d : RegularDecagon) (h : d.side_length = 3) : perimeter d = 30 := by
  sorry

end NUMINAMATH_CALUDE_decagon_perimeter_30_l1278_127803


namespace NUMINAMATH_CALUDE_one_is_optimal_l1278_127862

/-- Represents the number of teams that chose a particular number -/
def TeamChoices := ℕ → ℕ

/-- Calculates the score based on the game rules -/
def score (N : ℕ) (choices : TeamChoices) : ℕ :=
  if choices N > N then N else 0

/-- Theorem stating that 1 is the optimal choice -/
theorem one_is_optimal :
  ∀ (N : ℕ) (choices : TeamChoices),
    0 ≤ N ∧ N ≤ 20 →
    score 1 choices ≥ score N choices :=
sorry

end NUMINAMATH_CALUDE_one_is_optimal_l1278_127862


namespace NUMINAMATH_CALUDE_original_number_proof_l1278_127809

theorem original_number_proof : 
  ∃! x : ℕ, 
    (∃ k : ℕ, x + 5 = 23 * k) ∧ 
    (∀ y : ℕ, y < 5 → ∀ m : ℕ, x + y ≠ 23 * m) :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l1278_127809


namespace NUMINAMATH_CALUDE_expected_remaining_bullets_value_l1278_127839

/-- The probability of hitting the target -/
def p : ℝ := 0.6

/-- The total number of bullets -/
def n : ℕ := 4

/-- The expected number of remaining bullets -/
def expected_remaining_bullets : ℝ :=
  (n - 1) * p + (n - 2) * (1 - p) * p + (n - 3) * (1 - p)^2 * p + 0 * (1 - p)^3 * p

/-- Theorem stating the expected number of remaining bullets -/
theorem expected_remaining_bullets_value :
  expected_remaining_bullets = 2.376 := by sorry

end NUMINAMATH_CALUDE_expected_remaining_bullets_value_l1278_127839


namespace NUMINAMATH_CALUDE_no_two_common_tangents_l1278_127819

/-- Two circles in a plane with radii r and 2r -/
structure TwoCircles (r : ℝ) where
  center1 : ℝ × ℝ
  center2 : ℝ × ℝ

/-- The number of common tangents between two circles -/
def numCommonTangents (c : TwoCircles r) : ℕ := sorry

/-- Theorem: It's impossible for two circles with radii r and 2r to have exactly 2 common tangents -/
theorem no_two_common_tangents (r : ℝ) (hr : r > 0) :
  ∀ c : TwoCircles r, numCommonTangents c ≠ 2 := by sorry

end NUMINAMATH_CALUDE_no_two_common_tangents_l1278_127819


namespace NUMINAMATH_CALUDE_intersection_M_N_l1278_127835

-- Define the sets M and N
def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1278_127835


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l1278_127822

/-- 
For a quadratic expression of the form x^2 - 16x + k to be the square of a binomial,
k must equal 64.
-/
theorem quadratic_is_perfect_square (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 16*x + k = (a*x + b)^2) ↔ k = 64 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l1278_127822


namespace NUMINAMATH_CALUDE_ab_greater_ac_l1278_127810

theorem ab_greater_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_ab_greater_ac_l1278_127810


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1278_127821

theorem complex_magnitude_problem (z : ℂ) (h : (3 - 4*I) * z = 4 + 3*I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1278_127821


namespace NUMINAMATH_CALUDE_expression_values_l1278_127853

theorem expression_values (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : (x + y) / z = (y + z) / x) (h2 : (y + z) / x = (z + x) / y) :
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = 8 ∨
  ((x + y) * (y + z) * (z + x)) / (x * y * z) = -1 := by
sorry

end NUMINAMATH_CALUDE_expression_values_l1278_127853


namespace NUMINAMATH_CALUDE_square_difference_306_294_l1278_127849

theorem square_difference_306_294 : 306^2 - 294^2 = 7200 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_306_294_l1278_127849


namespace NUMINAMATH_CALUDE_circle_radius_in_square_l1278_127823

theorem circle_radius_in_square (side_length : ℝ) (l_shape_ratio : ℝ) : 
  side_length = 144 →
  l_shape_ratio = 5/18 →
  let total_area := side_length^2
  let l_shape_area := l_shape_ratio * total_area
  let center_square_area := total_area - 4 * l_shape_area
  let center_square_side := Real.sqrt center_square_area
  let radius := center_square_side / 2
  radius = 61.2 := by sorry

end NUMINAMATH_CALUDE_circle_radius_in_square_l1278_127823


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1278_127886

theorem polygon_interior_angles_sum (n : ℕ) : n ≥ 3 →
  (2 * n - 2) * 180 = 2160 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1278_127886


namespace NUMINAMATH_CALUDE_project_distribution_count_l1278_127875

/-- The number of districts --/
def num_districts : ℕ := 4

/-- The number of projects to be sponsored --/
def num_projects : ℕ := 3

/-- The maximum number of projects allowed in a single district --/
def max_projects_per_district : ℕ := 2

/-- The total number of possible distributions of projects among districts --/
def total_distributions : ℕ := num_districts ^ num_projects

/-- The number of invalid distributions (more than 2 projects in a district) --/
def invalid_distributions : ℕ := num_districts

theorem project_distribution_count :
  (total_distributions - invalid_distributions) = 60 := by
  sorry

end NUMINAMATH_CALUDE_project_distribution_count_l1278_127875


namespace NUMINAMATH_CALUDE_chair_color_probability_l1278_127838

theorem chair_color_probability (black_chairs brown_chairs : ℕ) 
  (h1 : black_chairs = 15) (h2 : brown_chairs = 18) : 
  let total_chairs := black_chairs + brown_chairs
  (black_chairs : ℚ) / total_chairs * ((black_chairs - 1) / (total_chairs - 1)) + 
  (brown_chairs : ℚ) / total_chairs * ((brown_chairs - 1) / (total_chairs - 1)) = 
  (15 : ℚ) / 33 * 14 / 32 + 18 / 33 * 17 / 32 := by
sorry

end NUMINAMATH_CALUDE_chair_color_probability_l1278_127838


namespace NUMINAMATH_CALUDE_A_and_C_work_time_l1278_127832

-- Define work rates
def work_rate_A : ℚ := 1 / 4
def work_rate_B : ℚ := 1 / 12
def work_rate_BC : ℚ := 1 / 3

-- Define the theorem
theorem A_and_C_work_time :
  let work_rate_C : ℚ := work_rate_BC - work_rate_B
  let work_rate_AC : ℚ := work_rate_A + work_rate_C
  (1 : ℚ) / work_rate_AC = 2 := by sorry

end NUMINAMATH_CALUDE_A_and_C_work_time_l1278_127832


namespace NUMINAMATH_CALUDE_perpendicular_plane_line_condition_l1278_127880

-- Define the types for planes and lines
variable (Point : Type) (Vector : Type)
variable (Plane : Type) (Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define what it means for a line to be in a plane
variable (line_in_plane : Line → Plane → Prop)

theorem perpendicular_plane_line_condition 
  (α β : Plane) (m : Line) 
  (h_diff : α ≠ β) 
  (h_m_in_α : line_in_plane m α) :
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m, line_in_plane m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_plane_line_condition_l1278_127880


namespace NUMINAMATH_CALUDE_combined_mixture_ratio_l1278_127848

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℕ
  water : ℕ

/-- Combines two mixtures -/
def combineMixtures (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk, water := m1.water + m2.water }

theorem combined_mixture_ratio : 
  let m1 : Mixture := { milk := 4, water := 2 }
  let m2 : Mixture := { milk := 5, water := 1 }
  let combined := combineMixtures m1 m2
  combined.milk = 9 ∧ combined.water = 3 := by
  sorry

end NUMINAMATH_CALUDE_combined_mixture_ratio_l1278_127848


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l1278_127899

theorem min_perimeter_triangle (d e f : ℕ) : 
  d > 0 → e > 0 → f > 0 →
  (d^2 + e^2 - f^2 : ℚ) / (2 * d * e) = 3 / 5 →
  (d^2 + f^2 - e^2 : ℚ) / (2 * d * f) = 9 / 10 →
  (e^2 + f^2 - d^2 : ℚ) / (2 * e * f) = -1 / 3 →
  d + e + f ≥ 50 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l1278_127899


namespace NUMINAMATH_CALUDE_correct_seating_arrangement_l1278_127816

/-- Represents whether a person is sitting or not -/
inductive Sitting : Type
| yes : Sitting
| no : Sitting

/-- The seating arrangement of individuals M, I, P, and A -/
structure SeatingArrangement :=
  (M : Sitting)
  (I : Sitting)
  (P : Sitting)
  (A : Sitting)

/-- The theorem stating the correct seating arrangement based on the given conditions -/
theorem correct_seating_arrangement :
  ∀ (arrangement : SeatingArrangement),
    arrangement.M = Sitting.no →
    (arrangement.M = Sitting.no → arrangement.I = Sitting.yes) →
    (arrangement.I = Sitting.yes → arrangement.P = Sitting.yes) →
    arrangement.A = Sitting.no →
    (arrangement.P = Sitting.yes ∧ 
     arrangement.I = Sitting.yes ∧ 
     arrangement.M = Sitting.no ∧ 
     arrangement.A = Sitting.no) :=
by sorry


end NUMINAMATH_CALUDE_correct_seating_arrangement_l1278_127816


namespace NUMINAMATH_CALUDE_total_profit_is_63000_l1278_127842

/-- Calculates the total profit earned by two partners based on their investments and one partner's share of the profit. -/
def calculateTotalProfit (tom_investment : ℕ) (tom_months : ℕ) (jose_investment : ℕ) (jose_months : ℕ) (jose_profit : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific investments and Jose's profit share, the total profit is 63000. -/
theorem total_profit_is_63000 :
  calculateTotalProfit 30000 12 45000 10 35000 = 63000 :=
sorry

end NUMINAMATH_CALUDE_total_profit_is_63000_l1278_127842


namespace NUMINAMATH_CALUDE_dog_sled_race_l1278_127898

theorem dog_sled_race (total_sleds : ℕ) (pairs : ℕ) (triples : ℕ) 
  (h1 : total_sleds = 315)
  (h2 : pairs + triples = total_sleds)
  (h3 : (6 * pairs + 2 * triples) * 10 = (2 * pairs + 3 * triples) * 5) :
  pairs = 225 ∧ triples = 90 := by
  sorry

end NUMINAMATH_CALUDE_dog_sled_race_l1278_127898


namespace NUMINAMATH_CALUDE_transformed_curve_equation_l1278_127884

/-- Given a curve and a scaling transformation, prove the equation of the transformed curve -/
theorem transformed_curve_equation (x y x' y' : ℝ) :
  (x^2 / 4 - y^2 = 1) →
  (x' = x / 2) →
  (y' = 2 * y) →
  (x'^2 - y'^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_transformed_curve_equation_l1278_127884


namespace NUMINAMATH_CALUDE_probability_is_two_over_155_l1278_127811

/-- Represents a 5x5x5 cube with two adjacent faces painted red -/
structure PaintedCube :=
  (size : Nat)
  (painted_faces : Nat)

/-- Calculates the number of unit cubes with exactly three painted faces -/
def count_three_painted_faces (cube : PaintedCube) : Nat :=
  1

/-- Calculates the number of unit cubes with no painted faces -/
def count_unpainted_faces (cube : PaintedCube) : Nat :=
  cube.size^3 - (cube.size^2 * 2 - cube.size)

/-- Calculates the total number of ways to choose two unit cubes -/
def total_combinations (cube : PaintedCube) : Nat :=
  (cube.size^3 * (cube.size^3 - 1)) / 2

/-- Calculates the probability of selecting one cube with three painted faces
    and one cube with no painted faces -/
def probability_three_and_none (cube : PaintedCube) : Rat :=
  (count_three_painted_faces cube * count_unpainted_faces cube) / total_combinations cube

/-- The main theorem stating the probability is 2/155 -/
theorem probability_is_two_over_155 (cube : PaintedCube) 
  (h1 : cube.size = 5) 
  (h2 : cube.painted_faces = 2) :
  probability_three_and_none cube = 2 / 155 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_two_over_155_l1278_127811


namespace NUMINAMATH_CALUDE_transaction_outcome_l1278_127856

theorem transaction_outcome : 
  let house_sell := 15000
  let store_sell := 14000
  let vehicle_sell := 18000
  let house_loss_percent := 25
  let store_gain_percent := 16.67
  let vehicle_gain_percent := 12.5
  
  let house_cost := house_sell / (1 - house_loss_percent / 100)
  let store_cost := store_sell / (1 + store_gain_percent / 100)
  let vehicle_cost := vehicle_sell / (1 + vehicle_gain_percent / 100)
  
  let total_cost := house_cost + store_cost + vehicle_cost
  let total_sell := house_sell + store_sell + vehicle_sell
  
  total_cost - total_sell = 1000 := by sorry

end NUMINAMATH_CALUDE_transaction_outcome_l1278_127856


namespace NUMINAMATH_CALUDE_sum_areas_circles_6_8_10_triangle_l1278_127845

/-- Given a 6-8-10 right triangle with vertices as centers of three mutually externally tangent circles,
    the sum of the areas of these circles is 56π. -/
theorem sum_areas_circles_6_8_10_triangle : 
  ∃ (α β γ : ℝ),
    α + β = 6 ∧
    α + γ = 8 ∧
    β + γ = 10 ∧
    α > 0 ∧ β > 0 ∧ γ > 0 →
    π * (α^2 + β^2 + γ^2) = 56 * π := by
  sorry


end NUMINAMATH_CALUDE_sum_areas_circles_6_8_10_triangle_l1278_127845


namespace NUMINAMATH_CALUDE_emma_age_when_sister_is_56_l1278_127859

/-- Emma's current age -/
def emma_age : ℕ := 7

/-- Age difference between Emma and her sister -/
def age_difference : ℕ := 9

/-- Age of Emma's sister when the problem is solved -/
def sister_future_age : ℕ := 56

/-- Emma's age when her sister reaches the future age -/
def emma_future_age : ℕ := emma_age + (sister_future_age - (emma_age + age_difference))

theorem emma_age_when_sister_is_56 : emma_future_age = 47 := by
  sorry

end NUMINAMATH_CALUDE_emma_age_when_sister_is_56_l1278_127859


namespace NUMINAMATH_CALUDE_z_range_l1278_127883

theorem z_range (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x + y = x * y) (hxyz : x + y + z = x * y * z) :
  1 < z ∧ z ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_z_range_l1278_127883


namespace NUMINAMATH_CALUDE_polynomial_ratio_theorem_l1278_127843

/-- The polynomial f(x) = x^2007 + 17x^2006 + 1 -/
def f (x : ℂ) : ℂ := x^2007 + 17*x^2006 + 1

/-- The set of distinct zeros of f -/
def zeros : Finset ℂ := sorry

/-- The polynomial P of degree 2007 -/
noncomputable def P : Polynomial ℂ := sorry

theorem polynomial_ratio_theorem :
  (∀ r ∈ zeros, f r = 0) →
  (Finset.card zeros = 2007) →
  (∀ r ∈ zeros, P.eval (r + 1/r) = 0) →
  (Polynomial.degree P = 2007) →
  P.eval 1 / P.eval (-1) = 289 / 259 := by sorry

end NUMINAMATH_CALUDE_polynomial_ratio_theorem_l1278_127843


namespace NUMINAMATH_CALUDE_female_worker_ants_l1278_127857

theorem female_worker_ants (total_ants : ℕ) (worker_ratio : ℚ) (male_ratio : ℚ) : 
  total_ants = 110 →
  worker_ratio = 1/2 →
  male_ratio = 1/5 →
  ⌊(total_ants : ℚ) * worker_ratio * (1 - male_ratio)⌋ = 44 := by
sorry

end NUMINAMATH_CALUDE_female_worker_ants_l1278_127857


namespace NUMINAMATH_CALUDE_cosA_value_triangle_area_l1278_127893

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Part I
theorem cosA_value (t : Triangle) 
  (h1 : t.a^2 = 3 * t.b * t.c) 
  (h2 : Real.sin t.A = Real.sin t.C) : 
  Real.cos t.A = 1/6 := by
sorry

-- Part II
theorem triangle_area (t : Triangle) 
  (h1 : t.a^2 = 3 * t.b * t.c) 
  (h2 : t.A = π/4) 
  (h3 : t.a = 3) : 
  (1/2) * t.b * t.c * Real.sin t.A = (3/4) * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cosA_value_triangle_area_l1278_127893


namespace NUMINAMATH_CALUDE_plot_length_is_75_l1278_127888

/-- The length of a rectangular plot in meters -/
def length : ℝ := 75

/-- The breadth of a rectangular plot in meters -/
def breadth : ℝ := length - 50

/-- The cost of fencing per meter in rupees -/
def cost_per_meter : ℝ := 26.50

/-- The total cost of fencing in rupees -/
def total_cost : ℝ := 5300

theorem plot_length_is_75 :
  (2 * length + 2 * breadth) * cost_per_meter = total_cost ∧
  length = breadth + 50 ∧
  length = 75 := by sorry

end NUMINAMATH_CALUDE_plot_length_is_75_l1278_127888


namespace NUMINAMATH_CALUDE_mara_bags_count_l1278_127861

/-- Prove that Mara has 12 bags given the conditions of the marble problem -/
theorem mara_bags_count : ∀ (x : ℕ), 
  (x * 2 + 2 = 2 * 13) → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_mara_bags_count_l1278_127861


namespace NUMINAMATH_CALUDE_linda_savings_fraction_l1278_127872

theorem linda_savings_fraction (original_savings : ℚ) (tv_cost : ℚ) 
  (h1 : original_savings = 880)
  (h2 : tv_cost = 220) :
  (original_savings - tv_cost) / original_savings = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_linda_savings_fraction_l1278_127872


namespace NUMINAMATH_CALUDE_factorial_sum_solution_l1278_127815

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_sum_solution :
  ∀ a b c d e f : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
    a > b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e ∧ e ≥ f →
    factorial a = factorial b + factorial c + factorial d + factorial e + factorial f →
    ((a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 5 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ e = 4 ∧ f = 4)) :=
by
  sorry

#check factorial_sum_solution

end NUMINAMATH_CALUDE_factorial_sum_solution_l1278_127815


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l1278_127890

theorem unique_quadratic_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + 2*(b + 1/b)*x + c = 0) → 
  c = 4 := by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l1278_127890


namespace NUMINAMATH_CALUDE_factorization_perfect_square_factorization_difference_of_cubes_l1278_127879

/-- Proves the factorization of a^2 + 2a + 1 -/
theorem factorization_perfect_square (a : ℝ) : a^2 + 2*a + 1 = (a + 1)^2 := by
  sorry

/-- Proves the factorization of a^3 - ab^2 -/
theorem factorization_difference_of_cubes (a b : ℝ) : a^3 - a*b^2 = a*(a + b)*(a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_perfect_square_factorization_difference_of_cubes_l1278_127879


namespace NUMINAMATH_CALUDE_unique_prime_solution_l1278_127840

theorem unique_prime_solution :
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (7 * p * q^2 + p = q^3 + 43 * p^3 + 1) → 
    (p = 2 ∧ q = 7) := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l1278_127840


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1278_127874

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 1 / a 0

/-- Sum of the first n terms of a geometric sequence -/
def sum_n (seq : GeometricSequence) (n : ℕ) : ℝ := sorry

/-- The main theorem -/
theorem geometric_sequence_sum_ratio 
  (seq : GeometricSequence) 
  (h : seq.a 6 = 8 * seq.a 3) : 
  sum_n seq 6 / sum_n seq 3 = 9 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1278_127874


namespace NUMINAMATH_CALUDE_proportion_third_number_l1278_127818

theorem proportion_third_number (y : ℝ) : 
  (0.6 : ℝ) / 0.96 = y / 8 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_third_number_l1278_127818


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1278_127873

-- Equation 1
theorem equation_one_solution (x : ℚ) : 
  (3/2 - 1/(3*x - 1) = 5/(6*x - 2)) ↔ (x = 10/9) :=
sorry

-- Equation 2
theorem equation_two_no_solution : 
  ¬∃ (x : ℚ), (5*x - 4)/(x - 2) = (4*x + 10)/(3*x - 6) - 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l1278_127873


namespace NUMINAMATH_CALUDE_class_size_is_50_l1278_127863

/-- The number of students in class 4(1) -/
def class_size : ℕ := 50

/-- The number of students in the basketball group -/
def basketball_group : ℕ := class_size / 2 + 1

/-- The number of students in the table tennis group -/
def table_tennis_group : ℕ := (class_size - basketball_group) / 2 + 2

/-- The number of students in the chess group -/
def chess_group : ℕ := (class_size - basketball_group - table_tennis_group) / 2 + 3

/-- The number of students in the broadcasting group -/
def broadcasting_group : ℕ := 2

theorem class_size_is_50 :
  class_size = 50 ∧
  basketball_group > class_size / 2 ∧
  table_tennis_group = (class_size - basketball_group) / 2 + 2 ∧
  chess_group = (class_size - basketball_group - table_tennis_group) / 2 + 3 ∧
  broadcasting_group = 2 ∧
  class_size = basketball_group + table_tennis_group + chess_group + broadcasting_group :=
by sorry

end NUMINAMATH_CALUDE_class_size_is_50_l1278_127863


namespace NUMINAMATH_CALUDE_rectangle_circle_square_area_l1278_127808

theorem rectangle_circle_square_area : 
  ∀ (r l b : ℝ), 
    l = (2/5) * r → 
    b = 10 → 
    l * b = 220 → 
    r^2 = 3025 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_square_area_l1278_127808


namespace NUMINAMATH_CALUDE_conference_duration_theorem_l1278_127825

/-- Calculates the duration of a conference in minutes, excluding the lunch break. -/
def conference_duration (total_hours : ℕ) (total_minutes : ℕ) (lunch_break : ℕ) : ℕ :=
  total_hours * 60 + total_minutes - lunch_break

/-- Proves that a conference lasting 8 hours and 40 minutes with a 15-minute lunch break
    has an active session time of 505 minutes. -/
theorem conference_duration_theorem :
  conference_duration 8 40 15 = 505 := by
  sorry

end NUMINAMATH_CALUDE_conference_duration_theorem_l1278_127825


namespace NUMINAMATH_CALUDE_milk_replacement_amount_l1278_127882

/-- Represents the amount of milk removed and replaced with water in each operation -/
def x : ℝ := 9

/-- The capacity of the vessel in litres -/
def vessel_capacity : ℝ := 90

/-- The amount of pure milk remaining after the operations in litres -/
def final_pure_milk : ℝ := 72.9

/-- Theorem stating that the amount of milk removed and replaced with water in each operation is correct -/
theorem milk_replacement_amount : 
  vessel_capacity - x - (vessel_capacity - x) * x / vessel_capacity = final_pure_milk := by
  sorry

end NUMINAMATH_CALUDE_milk_replacement_amount_l1278_127882


namespace NUMINAMATH_CALUDE_misread_division_sum_l1278_127804

theorem misread_division_sum (D : ℕ) (h : D = 56 * 25 + 25) :
  ∃ (q r : ℕ), D = 65 * q + r ∧ r < 65 ∧ q + r = 81 := by
  sorry

end NUMINAMATH_CALUDE_misread_division_sum_l1278_127804


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1278_127878

/-- Given a quadratic function f(x) = x^2 + ax + b, where a and b are real numbers,
    and sets A and B defined as follows:
    A = { x ∈ ℝ | f(x) ≤ 0 }
    B = { x ∈ ℝ | f(f(x)) ≤ 3 }
    If A = B ≠ ∅, then the range of a is [2√3, 6). -/
theorem quadratic_function_range (a b : ℝ) :
  let f := fun x : ℝ => x^2 + a*x + b
  let A := {x : ℝ | f x ≤ 0}
  let B := {x : ℝ | f (f x) ≤ 3}
  A = B ∧ A.Nonempty → a ∈ Set.Icc (2 * Real.sqrt 3) 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1278_127878


namespace NUMINAMATH_CALUDE_fraction_simplification_l1278_127867

theorem fraction_simplification : (252 : ℚ) / 8820 * 21 = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1278_127867


namespace NUMINAMATH_CALUDE_john_reading_capacity_l1278_127895

/-- Represents the reading speed ratio between John and his brother -/
def johnSpeedRatio : ℝ := 1.6

/-- Time taken by John's brother to read one book (in hours) -/
def brotherReadTime : ℝ := 8

/-- Available time for John to read (in hours) -/
def availableTime : ℝ := 15

/-- Number of books John can read in the available time -/
def johnBooksRead : ℕ := 3

theorem john_reading_capacity : 
  ⌊availableTime / (brotherReadTime / johnSpeedRatio)⌋ = johnBooksRead := by
  sorry

end NUMINAMATH_CALUDE_john_reading_capacity_l1278_127895


namespace NUMINAMATH_CALUDE_amber_bronze_selection_l1278_127891

/-- Represents a cell in the grid -/
inductive Cell
| Amber
| Bronze

/-- Represents the grid -/
def Grid (a b : ℕ) := Fin (a + b + 1) → Fin (a + b + 1) → Cell

/-- Counts the number of amber cells in the grid -/
def countAmber (g : Grid a b) : ℕ := sorry

/-- Counts the number of bronze cells in the grid -/
def countBronze (g : Grid a b) : ℕ := sorry

/-- Represents a selection of cells -/
def Selection (a b : ℕ) := Fin (a + b) → Fin (a + b + 1) × Fin (a + b + 1)

/-- Checks if a selection is valid (no two cells in the same row or column) -/
def isValidSelection (s : Selection a b) : Prop := sorry

/-- Counts the number of amber cells in a selection -/
def countAmberInSelection (g : Grid a b) (s : Selection a b) : ℕ := sorry

/-- Counts the number of bronze cells in a selection -/
def countBronzeInSelection (g : Grid a b) (s : Selection a b) : ℕ := sorry

theorem amber_bronze_selection (a b : ℕ) (g : Grid a b) 
  (ha : a > 0) (hb : b > 0)
  (hamber : countAmber g ≥ a^2 + a*b - b)
  (hbronze : countBronze g ≥ b^2 + a*b - a) :
  ∃ (s : Selection a b), 
    isValidSelection s ∧ 
    countAmberInSelection g s = a ∧ 
    countBronzeInSelection g s = b := by
  sorry

end NUMINAMATH_CALUDE_amber_bronze_selection_l1278_127891


namespace NUMINAMATH_CALUDE_custom_mul_identity_l1278_127858

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := 4 * a * b

/-- Theorem stating that if a * x = x for all x, then a = 1/4 -/
theorem custom_mul_identity (a : ℝ) : 
  (∀ x, custom_mul a x = x) → a = (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_identity_l1278_127858


namespace NUMINAMATH_CALUDE_least_three_digit_7_heavy_l1278_127827

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_heavy : 
  (∀ m : ℕ, is_three_digit m → is_7_heavy m → 103 ≤ m) ∧ 
  is_three_digit 103 ∧ 
  is_7_heavy 103 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_7_heavy_l1278_127827


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1278_127851

theorem contrapositive_equivalence (a : ℝ) :
  (¬(a > 1) → ¬(a > 0)) ↔ (a ≤ 1 → a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1278_127851


namespace NUMINAMATH_CALUDE_wood_cutting_l1278_127852

/-- Given a piece of wood that can be sawed into 9 sections of 4 meters each,
    prove that 11 cuts are needed to saw it into 3-meter sections. -/
theorem wood_cutting (wood_length : ℕ) (num_long_sections : ℕ) (long_section_length : ℕ) 
  (short_section_length : ℕ) (h1 : wood_length = num_long_sections * long_section_length)
  (h2 : num_long_sections = 9) (h3 : long_section_length = 4) (h4 : short_section_length = 3) : 
  (wood_length / short_section_length) - 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_wood_cutting_l1278_127852


namespace NUMINAMATH_CALUDE_pink_notebook_cost_l1278_127896

def total_notebooks : ℕ := 4
def green_notebooks : ℕ := 2
def black_notebooks : ℕ := 1
def pink_notebooks : ℕ := 1
def total_cost : ℕ := 45
def black_notebook_cost : ℕ := 15
def green_notebook_cost : ℕ := 10

theorem pink_notebook_cost :
  total_notebooks = green_notebooks + black_notebooks + pink_notebooks →
  total_cost = green_notebooks * green_notebook_cost + black_notebook_cost + pink_notebooks * 10 := by
  sorry

end NUMINAMATH_CALUDE_pink_notebook_cost_l1278_127896


namespace NUMINAMATH_CALUDE_team_a_more_uniform_l1278_127889

/-- Represents a dance team -/
structure DanceTeam where
  name : String
  variance : ℝ

/-- Compares the uniformity of heights between two dance teams -/
def more_uniform_heights (team1 team2 : DanceTeam) : Prop :=
  team1.variance < team2.variance

/-- The problem statement -/
theorem team_a_more_uniform : 
  let team_a : DanceTeam := ⟨"A", 1.5⟩
  let team_b : DanceTeam := ⟨"B", 2.4⟩
  more_uniform_heights team_a team_b := by
  sorry

end NUMINAMATH_CALUDE_team_a_more_uniform_l1278_127889


namespace NUMINAMATH_CALUDE_trapezoid_halving_line_iff_condition_l1278_127869

/-- A trapezoid with bases a and b, and legs c and d. -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  parallel_bases : a ≠ b → a < b

/-- The condition for a line to halve both perimeter and area of a trapezoid. -/
def halvingLineCondition (t : Trapezoid) : Prop :=
  (t.c + t.d) / 2 = (t.a + t.b) / 2 + Real.sqrt ((t.a^2 + t.b^2) / 2) ∨ t.a = t.b

/-- Theorem: A line parallel to the bases halves both perimeter and area of a trapezoid
    if and only if the halving line condition is satisfied. -/
theorem trapezoid_halving_line_iff_condition (t : Trapezoid) :
  ∃ (x : ℝ), 0 < x ∧ x < t.c ∧ x < t.d ∧
    (x + x + t.a + t.b = (t.a + t.b + t.c + t.d) / 2) ∧
    (x * (t.a + t.b) = (t.a + t.b) * t.c / 2) ↔
  halvingLineCondition t :=
sorry

end NUMINAMATH_CALUDE_trapezoid_halving_line_iff_condition_l1278_127869
