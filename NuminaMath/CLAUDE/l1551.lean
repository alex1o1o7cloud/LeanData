import Mathlib

namespace NUMINAMATH_CALUDE_johnny_planks_needed_l1551_155198

/-- Calculates the number of planks needed to build tables. -/
def planks_needed (num_tables : ℕ) (planks_per_leg : ℕ) (legs_per_table : ℕ) (planks_for_surface : ℕ) : ℕ :=
  num_tables * (legs_per_table * planks_per_leg + planks_for_surface)

/-- Theorem: Johnny needs 45 planks to build 5 tables. -/
theorem johnny_planks_needed : 
  planks_needed 5 1 4 5 = 45 := by
sorry

end NUMINAMATH_CALUDE_johnny_planks_needed_l1551_155198


namespace NUMINAMATH_CALUDE_salesman_profit_l1551_155161

/-- Calculates the salesman's profit from backpack sales --/
theorem salesman_profit : 
  let initial_cost : ℚ := 1500
  let import_tax_rate : ℚ := 5 / 100
  let total_cost : ℚ := initial_cost * (1 + import_tax_rate)
  let swap_meet_sales : ℚ := 30 * 22
  let department_store_sales : ℚ := 25 * 35
  let online_sales_regular : ℚ := 10 * 28
  let online_sales_discounted : ℚ := 5 * 28 * (1 - 10 / 100)
  let local_market_sales_1 : ℚ := 10 * 33
  let local_market_sales_2 : ℚ := 5 * 40
  let local_market_sales_3 : ℚ := 15 * 25
  let shipping_expenses : ℚ := 60
  let total_revenue : ℚ := swap_meet_sales + department_store_sales + 
    online_sales_regular + online_sales_discounted + 
    local_market_sales_1 + local_market_sales_2 + local_market_sales_3
  let profit : ℚ := total_revenue - total_cost - shipping_expenses
  profit = 1211 := by sorry

end NUMINAMATH_CALUDE_salesman_profit_l1551_155161


namespace NUMINAMATH_CALUDE_flower_shop_ratio_l1551_155125

/-- Flower shop problem -/
theorem flower_shop_ratio : 
  ∀ (roses lilacs gardenias : ℕ),
  roses = 3 * lilacs →
  lilacs = 10 →
  roses + lilacs + gardenias = 45 →
  gardenias / lilacs = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_ratio_l1551_155125


namespace NUMINAMATH_CALUDE_x_to_y_equals_nine_l1551_155175

theorem x_to_y_equals_nine (x y : ℝ) : y = Real.sqrt (x - 3) + Real.sqrt (3 - x) + 2 → x^y = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_to_y_equals_nine_l1551_155175


namespace NUMINAMATH_CALUDE_complex_addition_result_l1551_155152

theorem complex_addition_result : ∃ z : ℂ, (5 - 3*I + z = -4 + 9*I) ∧ (z = -9 + 12*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_addition_result_l1551_155152


namespace NUMINAMATH_CALUDE_range_of_a_l1551_155146

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a*x > 0) → 
  a < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1551_155146


namespace NUMINAMATH_CALUDE_car_journey_time_l1551_155166

/-- Calculates the total time for a car journey with two segments and a stop -/
theorem car_journey_time (distance1 : ℝ) (speed1 : ℝ) (stop_time : ℝ) (distance2 : ℝ) (speed2 : ℝ) :
  distance1 = 150 ∧ speed1 = 50 ∧ stop_time = 0.5 ∧ distance2 = 200 ∧ speed2 = 75 →
  distance1 / speed1 + stop_time + distance2 / speed2 = 6.17 := by
  sorry

#eval (150 / 50 + 0.5 + 200 / 75 : Float)

end NUMINAMATH_CALUDE_car_journey_time_l1551_155166


namespace NUMINAMATH_CALUDE_triangle_properties_l1551_155153

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : t.c * Real.sin t.C - t.a * Real.sin t.A = (Real.sqrt 3 * t.c - t.b) * Real.sin t.B) :
  -- Part 1: Angle A is 30 degrees (π/6 radians)
  t.A = π / 6 ∧
  -- Part 2: If a = 1, the maximum area is (2 + √3) / 4
  (t.a = 1 → 
    ∃ (S : ℝ), S = (2 + Real.sqrt 3) / 4 ∧ 
    ∀ (S' : ℝ), S' = 1/2 * t.b * t.c * Real.sin t.A → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1551_155153


namespace NUMINAMATH_CALUDE_average_side_length_of_squares_l1551_155159

theorem average_side_length_of_squares (a b c : Real) 
  (ha : a = 25) (hb : b = 64) (hc : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_side_length_of_squares_l1551_155159


namespace NUMINAMATH_CALUDE_board_division_impossibility_l1551_155186

theorem board_division_impossibility : ¬ ∃ (triangle_area : ℚ),
  (63 : ℚ) = 17 * triangle_area ∧
  ∃ (side_length : ℚ), 
    triangle_area = (side_length * side_length * Real.sqrt 3) / 4 ∧
    0 < side_length ∧
    side_length ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_board_division_impossibility_l1551_155186


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l1551_155139

theorem product_remainder_mod_five : ∃ k : ℕ, 2532 * 3646 * 2822 * 3716 * 101 = 5 * k + 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l1551_155139


namespace NUMINAMATH_CALUDE_BD_expression_A_B_D_collinear_l1551_155180

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the non-collinear vectors a and b
variable (a b : V)
variable (h_non_collinear : a ≠ 0 ∧ b ≠ 0 ∧ ¬∃ (r : ℝ), a = r • b)

-- Define the vectors AB, OB, and OD
def AB (a b : V) : V := 2 • a - 8 • b
def OB (a b : V) : V := a + 3 • b
def OD (a b : V) : V := 2 • a - b

-- Statement 1: Express BD in terms of a and b
theorem BD_expression (a b : V) : OD a b - OB a b = a - 4 • b := by sorry

-- Statement 2: Prove that A, B, and D are collinear
theorem A_B_D_collinear (a b : V) : 
  ∃ (r : ℝ), AB a b = r • (OD a b - OB a b) := by sorry

end NUMINAMATH_CALUDE_BD_expression_A_B_D_collinear_l1551_155180


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l1551_155140

theorem bowling_ball_weight :
  ∀ (ball_weight canoe_weight : ℚ),
    9 * ball_weight = 4 * canoe_weight →
    3 * canoe_weight = 112 →
    ball_weight = 448 / 27 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l1551_155140


namespace NUMINAMATH_CALUDE_train_length_approx_100_l1551_155113

/-- Calculates the length of a train given its speed, the time it takes to cross a platform, and the length of the platform. -/
def train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : ℝ :=
  speed * time - platform_length

/-- Theorem stating that a train with given parameters has a length of approximately 100 meters. -/
theorem train_length_approx_100 (speed : ℝ) (time : ℝ) (platform_length : ℝ) 
  (h1 : speed = 60 * 1000 / 3600) -- 60 km/hr converted to m/s
  (h2 : time = 14.998800095992321)
  (h3 : platform_length = 150) :
  ∃ ε > 0, |train_length speed time platform_length - 100| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_approx_100_l1551_155113


namespace NUMINAMATH_CALUDE_chord_length_midway_l1551_155133

theorem chord_length_midway (r : ℝ) (x y : ℝ) : 
  (24 : ℝ) ^ 2 / 4 + x^2 = r^2 →
  (32 : ℝ) ^ 2 / 4 + y^2 = r^2 →
  x + y = 14 →
  let d := (x - y) / 2
  2 * Real.sqrt (r^2 - d^2) = 2 * Real.sqrt 249 := by sorry

end NUMINAMATH_CALUDE_chord_length_midway_l1551_155133


namespace NUMINAMATH_CALUDE_max_consecutive_positive_terms_l1551_155108

/-- A sequence of real numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = a (n - 1) + a (n + 2)

/-- The property that a sequence has k consecutive positive terms starting from index n -/
def HasConsecutivePositiveTerms (a : ℕ → ℝ) (n k : ℕ) : Prop :=
  ∀ i : ℕ, i ∈ Finset.range k → a (n + i) > 0

/-- The main theorem stating that the maximum number of consecutive positive terms is 5 -/
theorem max_consecutive_positive_terms
  (a : ℕ → ℝ) (h : RecurrenceSequence a) :
  (∃ n k : ℕ, k > 5 ∧ HasConsecutivePositiveTerms a n k) → False :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_positive_terms_l1551_155108


namespace NUMINAMATH_CALUDE_skyscraper_anniversary_l1551_155136

/-- Calculates the number of years in the future when it will be 5 years before the 200th anniversary of a skyscraper built 100 years ago. -/
theorem skyscraper_anniversary (years_since_built : ℕ) (years_to_anniversary : ℕ) (years_before_anniversary : ℕ) : 
  years_since_built = 100 →
  years_to_anniversary = 200 →
  years_before_anniversary = 5 →
  years_to_anniversary - years_before_anniversary - years_since_built = 95 :=
by sorry

end NUMINAMATH_CALUDE_skyscraper_anniversary_l1551_155136


namespace NUMINAMATH_CALUDE_function_identity_l1551_155190

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem function_identity (f : ℕ+ → ℕ+) : 
  (∀ m n : ℕ+, is_divisible (m^2 + f n) (m * f m + n)) → 
  (∀ n : ℕ+, f n = n) :=
by sorry

end NUMINAMATH_CALUDE_function_identity_l1551_155190


namespace NUMINAMATH_CALUDE_abc_inequality_l1551_155164

theorem abc_inequality (a b c : ℝ) (ha : |a| < 1) (hb : |b| < 1) (hc : |c| < 1) :
  a * b * c + 2 > a + b + c := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1551_155164


namespace NUMINAMATH_CALUDE_sum_of_possible_e_values_l1551_155144

theorem sum_of_possible_e_values : 
  ∃ (e₁ e₂ : ℝ), (2 * |2 - e₁| = 5) ∧ (2 * |2 - e₂| = 5) ∧ (e₁ + e₂ = 4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_e_values_l1551_155144


namespace NUMINAMATH_CALUDE_cos_negative_seventeen_thirds_pi_l1551_155177

theorem cos_negative_seventeen_thirds_pi : 
  Real.cos (-17/3 * Real.pi) = 1/2 := by sorry

end NUMINAMATH_CALUDE_cos_negative_seventeen_thirds_pi_l1551_155177


namespace NUMINAMATH_CALUDE_minimum_distance_theorem_l1551_155157

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 2

def line_l (x y : ℝ) : Prop := x + 2 * y - 2 * Real.log 2 - 6 = 0

def M (px py qx qy : ℝ) : ℝ := (px - qx)^2 + (py - qy)^2

theorem minimum_distance_theorem (px py qx qy : ℝ) 
  (h1 : f px = py) 
  (h2 : line_l qx qy) : 
  (∃ (min_M : ℝ), ∀ (px' py' qx' qy' : ℝ), 
    f px' = py' → line_l qx' qy' → 
    M px' py' qx' qy' ≥ min_M ∧ 
    min_M = 16/5 ∧
    (M px py qx qy = min_M → qx = 14/5)) := by sorry

end NUMINAMATH_CALUDE_minimum_distance_theorem_l1551_155157


namespace NUMINAMATH_CALUDE_autumn_pencils_l1551_155101

def pencil_count (initial misplaced broken found bought : ℕ) : ℕ :=
  initial - misplaced - broken + found + bought

theorem autumn_pencils : pencil_count 20 7 3 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_autumn_pencils_l1551_155101


namespace NUMINAMATH_CALUDE_bill_left_with_411_l1551_155147

/-- Calculates the amount of money Bill is left with after all transactions and expenses -/
def billsRemainingMoney : ℝ :=
  let merchantA_sale := 8 * 9
  let merchantB_sale := 15 * 11
  let sheriff_fine := 80
  let merchantC_sale := 25 * 8
  let protection_cost := 30
  let passerby_sale := 12 * 7
  
  let total_earnings := merchantA_sale + merchantB_sale + merchantC_sale + passerby_sale
  let total_expenses := sheriff_fine + protection_cost
  
  total_earnings - total_expenses

/-- Theorem stating that Bill is left with $411 after all transactions and expenses -/
theorem bill_left_with_411 : billsRemainingMoney = 411 := by
  sorry

end NUMINAMATH_CALUDE_bill_left_with_411_l1551_155147


namespace NUMINAMATH_CALUDE_max_product_dice_rolls_l1551_155176

theorem max_product_dice_rolls (rolls : List Nat) : 
  rolls.length = 25 → 
  (∀ x ∈ rolls, 1 ≤ x ∧ x ≤ 20) →
  rolls.sum = 70 →
  rolls.prod ≤ (List.replicate 5 2 ++ List.replicate 20 3).prod :=
sorry

end NUMINAMATH_CALUDE_max_product_dice_rolls_l1551_155176


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l1551_155195

/-- A circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The length of the first chord -/
  chord1 : ℝ
  /-- The length of the second chord -/
  chord2 : ℝ
  /-- The length of the third chord -/
  chord3 : ℝ
  /-- The first chord has length 42 -/
  chord1_eq : chord1 = 42
  /-- The second chord has length 42 -/
  chord2_eq : chord2 = 42
  /-- The third chord has length 40 -/
  chord3_eq : chord3 = 40

/-- The theorem stating that the distance between adjacent parallel lines is 3 3/8 -/
theorem parallel_lines_distance (c : CircleWithParallelLines) : c.d = 3 + 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l1551_155195


namespace NUMINAMATH_CALUDE_helens_oranges_l1551_155137

/-- Helen's orange counting problem -/
theorem helens_oranges (initial : ℕ) (from_ann : ℕ) (to_sarah : ℕ) : 
  initial = 9 → from_ann = 29 → to_sarah = 14 → 
  initial + from_ann - to_sarah = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_helens_oranges_l1551_155137


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1551_155126

theorem quadratic_roots_relation (q : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (x₁^2 - 5*x₁ + q = 0) ∧ 
    (x₂^2 - 5*x₂ + q = 0) ∧ 
    (x₃^2 - 7*x₃ + 2*q = 0) ∧ 
    (x₄^2 - 7*x₄ + 2*q = 0) ∧ 
    (x₃ = 2*x₁ ∨ x₃ = 2*x₂ ∨ x₄ = 2*x₁ ∨ x₄ = 2*x₂)) →
  q = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l1551_155126


namespace NUMINAMATH_CALUDE_kostya_bulbs_count_l1551_155118

/-- Function to calculate the number of bulbs after one round of planting -/
def plant_between (n : ℕ) : ℕ := 2 * n - 1

/-- Function to calculate the number of bulbs after three rounds of planting -/
def plant_three_rounds (n : ℕ) : ℕ := plant_between (plant_between (plant_between n))

/-- Theorem stating that if Kostya planted n bulbs and the final count after three rounds is 113, then n must be 15 -/
theorem kostya_bulbs_count : 
  ∀ n : ℕ, plant_three_rounds n = 113 → n = 15 := by
sorry

#eval plant_three_rounds 15  -- Should output 113

end NUMINAMATH_CALUDE_kostya_bulbs_count_l1551_155118


namespace NUMINAMATH_CALUDE_find_N_l1551_155196

theorem find_N (a b c N : ℚ) 
  (sum_eq : a + b + c = 120)
  (a_eq : a - 10 = N)
  (b_eq : 10 * b = N)
  (c_eq : c - 10 = N) :
  N = 1100 / 21 := by
sorry

end NUMINAMATH_CALUDE_find_N_l1551_155196


namespace NUMINAMATH_CALUDE_no_solution_exists_l1551_155106

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_solution_exists : ∀ n : ℕ, n * sum_of_digits n ≠ 100200300 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1551_155106


namespace NUMINAMATH_CALUDE_parallel_line_equation_l1551_155169

/-- A line passing through a point and parallel to another line -/
theorem parallel_line_equation (x y : ℝ) : 
  (x - 2*y + 7 = 0) ↔ 
  (∃ (m b : ℝ), y = m*x + b ∧ m = (1/2) ∧ y = m*(x+1) + 3) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l1551_155169


namespace NUMINAMATH_CALUDE_sample_size_equals_selected_high_school_entrance_exam_sample_size_l1551_155162

/-- Represents a statistical sample --/
structure Sample where
  population : ℕ
  selected : ℕ

/-- Definition of sample size --/
def sampleSize (s : Sample) : ℕ := s.selected

/-- Theorem stating that the sample size is equal to the number of selected students --/
theorem sample_size_equals_selected (s : Sample) 
  (h₁ : s.population = 150000) 
  (h₂ : s.selected = 1000) : 
  sampleSize s = 1000 := by
  sorry

/-- Main theorem proving the sample size for the given problem --/
theorem high_school_entrance_exam_sample_size :
  ∃ s : Sample, s.population = 150000 ∧ s.selected = 1000 ∧ sampleSize s = 1000 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_equals_selected_high_school_entrance_exam_sample_size_l1551_155162


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1551_155194

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l1551_155194


namespace NUMINAMATH_CALUDE_geometric_progression_constant_l1551_155124

theorem geometric_progression_constant (x : ℝ) : 
  (((30 + x) ^ 2 = (10 + x) * (90 + x)) ↔ x = 0) ∧
  (∀ y : ℝ, ((30 + y) ^ 2 = (10 + y) * (90 + y)) → y = 0) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_constant_l1551_155124


namespace NUMINAMATH_CALUDE_line_equation_l1551_155181

/-- A line passing through point (1, 2) with slope √3 has the equation √3x - y + 2 - √3 = 0 -/
theorem line_equation (x y : ℝ) : 
  (y - 2 = Real.sqrt 3 * (x - 1)) ↔ (Real.sqrt 3 * x - y + 2 - Real.sqrt 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l1551_155181


namespace NUMINAMATH_CALUDE_tourists_scientific_correct_l1551_155111

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- The number of tourists per year -/
def tourists_per_year : ℕ := 876000

/-- The scientific notation representation of the number of tourists -/
def tourists_scientific : ScientificNotation where
  coefficient := 8.76
  exponent := 5
  one_le_coeff_lt_ten := by sorry

/-- Theorem stating that the scientific notation representation is correct -/
theorem tourists_scientific_correct : 
  (tourists_scientific.coefficient * (10 : ℝ) ^ tourists_scientific.exponent) = tourists_per_year := by sorry

end NUMINAMATH_CALUDE_tourists_scientific_correct_l1551_155111


namespace NUMINAMATH_CALUDE_wire_length_around_square_field_l1551_155179

theorem wire_length_around_square_field (area : ℝ) (rounds : ℕ) 
  (h1 : area = 69696) 
  (h2 : rounds = 15) : 
  Real.sqrt area * 4 * rounds = 15840 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_around_square_field_l1551_155179


namespace NUMINAMATH_CALUDE_x_minus_y_equals_nine_l1551_155117

theorem x_minus_y_equals_nine (x y : ℕ) (h1 : 3^x * 4^y = 19683) (h2 : x = 9) :
  x - y = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_nine_l1551_155117


namespace NUMINAMATH_CALUDE_sin_300_degrees_l1551_155160

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l1551_155160


namespace NUMINAMATH_CALUDE_sum_longest_altitudes_is_14_l1551_155158

/-- A triangle with sides 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 6
  hb : b = 8
  hc : c = 10
  right_angle : a^2 + b^2 = c^2

/-- The sum of the lengths of the two longest altitudes in the triangle -/
def sum_longest_altitudes (t : RightTriangle) : ℝ := t.a + t.b

/-- Theorem: The sum of the lengths of the two longest altitudes in a triangle 
    with sides 6, 8, and 10 is 14 -/
theorem sum_longest_altitudes_is_14 (t : RightTriangle) : 
  sum_longest_altitudes t = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_longest_altitudes_is_14_l1551_155158


namespace NUMINAMATH_CALUDE_second_number_difference_l1551_155167

theorem second_number_difference (first_number second_number : ℤ) : 
  first_number = 15 →
  second_number = 55 →
  first_number + second_number = 70 →
  second_number - 3 * first_number = 10 := by
sorry

end NUMINAMATH_CALUDE_second_number_difference_l1551_155167


namespace NUMINAMATH_CALUDE_tire_cost_l1551_155155

theorem tire_cost (n : ℕ+) (total_cost battery_cost : ℚ) 
  (h1 : total_cost = 224)
  (h2 : battery_cost = 56) :
  (total_cost - battery_cost) / n = (224 - 56) / n :=
by sorry

end NUMINAMATH_CALUDE_tire_cost_l1551_155155


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l1551_155110

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (1/a + 2/b) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l1551_155110


namespace NUMINAMATH_CALUDE_min_value_of_z_l1551_155131

variable (a b x : ℝ)
variable (h : a ≠ b)

def z (x : ℝ) : ℝ := (x - a)^3 + (x - b)^3

theorem min_value_of_z :
  ∃ (x : ℝ), ∀ (y : ℝ), z a b x ≤ z a b y ↔ x = (a + b) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1551_155131


namespace NUMINAMATH_CALUDE_range_of_m_l1551_155188

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : A ∩ B m = B m → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1551_155188


namespace NUMINAMATH_CALUDE_jason_total_games_l1551_155115

/-- The total number of games Jason will attend over three months -/
def total_games (this_month last_month next_month : ℕ) : ℕ :=
  this_month + last_month + next_month

/-- Theorem stating the total number of games Jason will attend -/
theorem jason_total_games : 
  total_games 11 17 16 = 44 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_games_l1551_155115


namespace NUMINAMATH_CALUDE_consecutive_even_sum_l1551_155104

theorem consecutive_even_sum (n : ℤ) : 
  (∃ (a b c d : ℤ), 
    a = n ∧ 
    b = n + 2 ∧ 
    c = n + 4 ∧ 
    d = n + 6 ∧ 
    c = 14) → 
  (n + (n + 2) + (n + 4) + (n + 6) = 52) := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_l1551_155104


namespace NUMINAMATH_CALUDE_cookie_difference_l1551_155178

def sweet_cookies_initial : ℕ := 37
def salty_cookies_initial : ℕ := 11
def sweet_cookies_eaten : ℕ := 5
def salty_cookies_eaten : ℕ := 2

theorem cookie_difference : sweet_cookies_eaten - salty_cookies_eaten = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l1551_155178


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1551_155182

/-- A circle with equation x^2 + y^2 = m^2 is tangent to the line x + 2y = √(3m) if and only if m = 3/5 -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = m^2 ∧ x + 2*y = Real.sqrt (3*m) ∧ 
    ∀ (x' y' : ℝ), x'^2 + y'^2 = m^2 → x' + 2*y' ≠ Real.sqrt (3*m) ∨ (x' = x ∧ y' = y)) ↔ 
  m = 3/5 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1551_155182


namespace NUMINAMATH_CALUDE_jade_transactions_l1551_155193

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 17 →
  jade = 83 := by
  sorry

end NUMINAMATH_CALUDE_jade_transactions_l1551_155193


namespace NUMINAMATH_CALUDE_range_of_a_for_subset_l1551_155156

/-- The solution set of x^2 - ax - x < 0 -/
def M (a : ℝ) : Set ℝ :=
  {x | x^2 - a*x - x < 0}

/-- The solution set of x^2 - 2x - 3 ≤ 0 -/
def N : Set ℝ :=
  {x | x^2 - 2*x - 3 ≤ 0}

/-- The theorem stating the range of a for which M(a) ⊆ N -/
theorem range_of_a_for_subset : 
  {a : ℝ | M a ⊆ N} = {a : ℝ | -2 ≤ a ∧ a ≤ 2} := by
sorry

end NUMINAMATH_CALUDE_range_of_a_for_subset_l1551_155156


namespace NUMINAMATH_CALUDE_new_average_price_six_toys_average_l1551_155135

def average_price (n : ℕ) (total_cost : ℚ) : ℚ := total_cost / n

theorem new_average_price 
  (n : ℕ) 
  (old_avg : ℚ) 
  (additional_cost : ℚ) : 
  average_price (n + 1) (n * old_avg + additional_cost) = 
    (n * old_avg + additional_cost) / (n + 1) :=
by
  sorry

theorem six_toys_average 
  (dhoni_toys : ℕ) 
  (dhoni_avg : ℚ) 
  (david_toy_price : ℚ) 
  (h1 : dhoni_toys = 5) 
  (h2 : dhoni_avg = 10) 
  (h3 : david_toy_price = 16) :
  average_price (dhoni_toys + 1) (dhoni_toys * dhoni_avg + david_toy_price) = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_new_average_price_six_toys_average_l1551_155135


namespace NUMINAMATH_CALUDE_complex_arithmetic_calculation_l1551_155127

theorem complex_arithmetic_calculation : 
  ((15 - 2) + (4 / (1 / 2)) - (6 * 8)) * (100 - 24) / 38 = -54 := by
sorry

end NUMINAMATH_CALUDE_complex_arithmetic_calculation_l1551_155127


namespace NUMINAMATH_CALUDE_green_apples_count_l1551_155187

theorem green_apples_count (total : ℕ) (red_to_green_ratio : ℕ) 
  (h1 : total = 496) 
  (h2 : red_to_green_ratio = 3) : 
  ∃ green : ℕ, green = 124 ∧ total = green * (red_to_green_ratio + 1) :=
by sorry

end NUMINAMATH_CALUDE_green_apples_count_l1551_155187


namespace NUMINAMATH_CALUDE_ratio_equality_l1551_155191

theorem ratio_equality (a b : ℝ) (h1 : 7 * a = 8 * b) (h2 : a * b ≠ 0) :
  (a / 8) / (b / 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1551_155191


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l1551_155192

theorem polynomial_division_theorem (x : ℝ) : 
  x^5 - 25*x^3 + 13*x^2 - 16*x + 12 = (x - 3) * (x^4 + 3*x^3 - 16*x^2 - 35*x - 121) + (-297) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l1551_155192


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l1551_155189

theorem largest_divisor_of_n_squared_div_72 (n : ℕ) (h : n > 0) (h_div : 72 ∣ n^2) :
  ∀ k : ℕ, k > 12 → ¬(∀ m : ℕ, m > 0 ∧ 72 ∣ m^2 → k ∣ m) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l1551_155189


namespace NUMINAMATH_CALUDE_smallest_valid_n_l1551_155149

def is_valid (n : ℕ) : Prop :=
  ∀ m : ℕ+, ∃ S : Finset ℕ, S ⊆ Finset.range n ∧ 
    (S.prod id : ℕ) ≡ m [ZMOD 100]

theorem smallest_valid_n :
  is_valid 17 ∧ ∀ k < 17, ¬ is_valid k :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l1551_155149


namespace NUMINAMATH_CALUDE_petes_son_age_l1551_155197

/-- Given Pete's current age and the relationship between Pete's and his son's ages in 4 years,
    this theorem proves the current age of Pete's son. -/
theorem petes_son_age (pete_age : ℕ) (h : pete_age = 35) :
  ∃ (son_age : ℕ), son_age = 9 ∧ pete_age + 4 = 3 * (son_age + 4) := by
  sorry

end NUMINAMATH_CALUDE_petes_son_age_l1551_155197


namespace NUMINAMATH_CALUDE_ratio_equivalence_l1551_155129

theorem ratio_equivalence (x y m n : ℚ) 
  (h : (5 * x + 7 * y) / (3 * x + 2 * y) = m / n) :
  (13 * x + 16 * y) / (2 * x + 5 * y) = (2 * m + n) / (m - n) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equivalence_l1551_155129


namespace NUMINAMATH_CALUDE_pages_left_to_read_l1551_155199

/-- Given a book with specified total pages, pages read, daily reading rate, and reading duration,
    calculate the number of pages left to read after the reading period. -/
theorem pages_left_to_read 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (pages_per_day : ℕ) 
  (days : ℕ) 
  (h1 : total_pages = 381) 
  (h2 : pages_read = 149) 
  (h3 : pages_per_day = 20) 
  (h4 : days = 7) :
  total_pages - pages_read - (pages_per_day * days) = 92 := by
  sorry


end NUMINAMATH_CALUDE_pages_left_to_read_l1551_155199


namespace NUMINAMATH_CALUDE_food_drive_problem_l1551_155141

/-- Represents the food drive problem in Ms. Perez's class -/
theorem food_drive_problem (total_students : ℕ) (half_students_12_cans : ℕ) (students_4_cans : ℕ) (total_cans : ℕ) :
  total_students = 30 →
  half_students_12_cans = total_students / 2 →
  students_4_cans = 13 →
  total_cans = 232 →
  half_students_12_cans * 12 + students_4_cans * 4 = total_cans →
  total_students - (half_students_12_cans + students_4_cans) = 2 :=
by sorry

end NUMINAMATH_CALUDE_food_drive_problem_l1551_155141


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1551_155119

theorem complex_equation_solution (i : ℂ) (z : ℂ) 
  (h1 : i * i = -1)
  (h2 : z * (1 - i) = 3 + 2 * i) :
  z = 1/2 + 5/2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1551_155119


namespace NUMINAMATH_CALUDE_impossible_three_quadratics_with_two_roots_l1551_155145

theorem impossible_three_quadratics_with_two_roots :
  ¬ ∃ (a b c : ℝ),
    (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) ∧
    (∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ c * y₁^2 + a * y₁ + b = 0 ∧ c * y₂^2 + a * y₂ + b = 0) ∧
    (∃ (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ b * z₁^2 + c * z₁ + a = 0 ∧ b * z₂^2 + c * z₂ + a = 0) :=
by sorry

end NUMINAMATH_CALUDE_impossible_three_quadratics_with_two_roots_l1551_155145


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_result_l1551_155114

theorem complex_arithmetic_expression_result : 
  let expr := 3034 - ((1002 / 20.04) * (43.8 - 9.2^2) + Real.sqrt 144) / (3.58 * (76 - 8.23^3))
  ∃ ε > 0, abs (expr - 1.17857142857) < ε :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_result_l1551_155114


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l1551_155134

theorem boat_speed_ratio (v : ℝ) (c : ℝ) (d : ℝ) 
  (hv : v = 24) -- Boat speed in still water
  (hc : c = 6)  -- River current speed
  (hd : d = 3)  -- Distance traveled downstream and upstream
  : (2 * d) / ((d / (v + c)) + (d / (v - c))) / v = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l1551_155134


namespace NUMINAMATH_CALUDE_amy_red_balloons_l1551_155103

theorem amy_red_balloons (total green blue : ℕ) (h1 : total = 67) (h2 : green = 17) (h3 : blue = 21) : total - green - blue = 29 := by
  sorry

end NUMINAMATH_CALUDE_amy_red_balloons_l1551_155103


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1551_155150

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  a^4 + b^4 + c^4 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1551_155150


namespace NUMINAMATH_CALUDE_chocolate_sales_l1551_155107

theorem chocolate_sales (C S : ℝ) (n : ℕ) 
  (h1 : 81 * C = n * S)  -- Cost price of 81 chocolates equals selling price of n chocolates
  (h2 : S = 1.8 * C)     -- Selling price is 1.8 times the cost price (derived from 80% gain)
  : n = 45 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_sales_l1551_155107


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l1551_155151

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  (x - 1)^2 = 2 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := by
sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  x^2 - 6*x - 7 = 0 ↔ x = -1 ∨ x = 7 := by
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l1551_155151


namespace NUMINAMATH_CALUDE_diophantine_equation_implication_l1551_155132

-- Define the property of not being a perfect square
def NotPerfectSquare (n : ℤ) : Prop := ∀ m : ℤ, n ≠ m^2

-- Define a nontrivial integer solution
def HasNontrivialSolution (f : ℤ → ℤ → ℤ → ℤ) : Prop :=
  ∃ x y z : ℤ, f x y z = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)

-- Define a nontrivial integer solution for 4 variables
def HasNontrivialSolution4 (f : ℤ → ℤ → ℤ → ℤ → ℤ) : Prop :=
  ∃ x y z w : ℤ, f x y z w = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ w ≠ 0)

theorem diophantine_equation_implication (a b : ℤ) 
  (ha : NotPerfectSquare a) (hb : NotPerfectSquare b)
  (h : HasNontrivialSolution4 (fun x y z w => x^2 - a*y^2 - b*z^2 + a*b*w^2)) :
  HasNontrivialSolution (fun x y z => x^2 - a*y^2 - b*z^2) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_implication_l1551_155132


namespace NUMINAMATH_CALUDE_xyz_value_l1551_155174

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + y * z + z * x) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) :
  x * y * z = 14/3 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1551_155174


namespace NUMINAMATH_CALUDE_work_completion_time_l1551_155116

theorem work_completion_time 
  (total_men : ℕ) 
  (initial_days : ℕ) 
  (absent_men : ℕ) 
  (h1 : total_men = 15)
  (h2 : initial_days = 8)
  (h3 : absent_men = 3) : 
  (total_men * initial_days) / (total_men - absent_men) = 10 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1551_155116


namespace NUMINAMATH_CALUDE_total_scissors_is_86_l1551_155112

/-- Calculates the total number of scissors after changes in two drawers -/
def totalScissorsAfterChanges (
  initialScissors1 : ℕ) (initialScissors2 : ℕ) 
  (addedScissors1 : ℕ) (addedScissors2 : ℕ) : ℕ :=
  (initialScissors1 + addedScissors1) + (initialScissors2 + addedScissors2)

/-- Proves that the total number of scissors after changes is 86 -/
theorem total_scissors_is_86 :
  totalScissorsAfterChanges 39 27 13 7 = 86 := by
  sorry

end NUMINAMATH_CALUDE_total_scissors_is_86_l1551_155112


namespace NUMINAMATH_CALUDE_dog_purchase_cost_l1551_155170

theorem dog_purchase_cost (current_amount additional_amount : ℕ) 
  (h1 : current_amount = 34)
  (h2 : additional_amount = 13) :
  current_amount + additional_amount = 47 := by
  sorry

end NUMINAMATH_CALUDE_dog_purchase_cost_l1551_155170


namespace NUMINAMATH_CALUDE_min_white_surface_fraction_l1551_155100

/-- Represents a cube with given edge length -/
structure Cube where
  edge_length : ℕ

/-- Represents the large cube constructed from smaller cubes -/
structure LargeCube where
  edge_length : ℕ
  small_cubes : ℕ
  red_cubes : ℕ
  white_cubes : ℕ

/-- Calculates the surface area of a cube -/
def surface_area (c : Cube) : ℕ := 6 * c.edge_length^2

/-- Theorem: The minimum fraction of white surface area in the described cube configuration is 1/12 -/
theorem min_white_surface_fraction (lc : LargeCube) 
  (h1 : lc.edge_length = 4)
  (h2 : lc.small_cubes = 64)
  (h3 : lc.red_cubes = 48)
  (h4 : lc.white_cubes = 16) :
  ∃ (white_area : ℕ), 
    white_area ≤ lc.white_cubes ∧ 
    (white_area : ℚ) / (surface_area ⟨lc.edge_length⟩ : ℚ) = 1/12 ∧
    ∀ (other_white_area : ℕ), 
      other_white_area ≤ lc.white_cubes → 
      (other_white_area : ℚ) / (surface_area ⟨lc.edge_length⟩ : ℚ) ≥ 1/12 := by
  sorry

end NUMINAMATH_CALUDE_min_white_surface_fraction_l1551_155100


namespace NUMINAMATH_CALUDE_x_values_l1551_155120

theorem x_values (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 144)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 :=
by sorry

end NUMINAMATH_CALUDE_x_values_l1551_155120


namespace NUMINAMATH_CALUDE_fruit_juice_needed_correct_problem_solution_l1551_155165

/-- Represents the ratio of ingredients in a drink -/
structure DrinkRatio where
  milk : ℚ
  fruit_juice : ℚ

/-- Represents the amount of ingredients in a drink -/
structure DrinkAmount where
  milk : ℚ
  fruit_juice : ℚ

/-- Converts a ratio to normalized form where total parts sum to 1 -/
def normalize_ratio (r : DrinkRatio) : DrinkRatio :=
  let total := r.milk + r.fruit_juice
  { milk := r.milk / total, fruit_juice := r.fruit_juice / total }

/-- Calculates the amount of fruit juice needed to convert drink A to drink B -/
def fruit_juice_needed (amount_A : ℚ) (ratio_A ratio_B : DrinkRatio) : ℚ :=
  let norm_A := normalize_ratio ratio_A
  let norm_B := normalize_ratio ratio_B
  let milk_A := amount_A * norm_A.milk
  let fruit_juice_A := amount_A * norm_A.fruit_juice
  (milk_A - fruit_juice_A) / (norm_B.fruit_juice - norm_B.milk)

/-- Theorem: The amount of fruit juice needed is correct -/
theorem fruit_juice_needed_correct (amount_A : ℚ) (ratio_A ratio_B : DrinkRatio) :
  let juice_needed := fruit_juice_needed amount_A ratio_A ratio_B
  let total_amount := amount_A + juice_needed
  let final_amount := DrinkAmount.mk (amount_A * (normalize_ratio ratio_A).milk) (fruit_juice_needed amount_A ratio_A ratio_B + amount_A * (normalize_ratio ratio_A).fruit_juice)
  final_amount.milk / total_amount = (normalize_ratio ratio_B).milk ∧
  final_amount.fruit_juice / total_amount = (normalize_ratio ratio_B).fruit_juice :=
by sorry

/-- Specific problem instance -/
def drink_A : DrinkRatio := { milk := 4, fruit_juice := 3 }
def drink_B : DrinkRatio := { milk := 3, fruit_juice := 4 }

/-- Theorem: For the given problem, 14 liters of fruit juice are needed -/
theorem problem_solution : 
  fruit_juice_needed 98 drink_A drink_B = 14 :=
by sorry

end NUMINAMATH_CALUDE_fruit_juice_needed_correct_problem_solution_l1551_155165


namespace NUMINAMATH_CALUDE_tenby_position_l1551_155163

def letters : List Char := ['B', 'E', 'N', 'T', 'Y']

def word : String := "TENBY"

def alphabetical_position (w : String) (l : List Char) : ℕ :=
  sorry

theorem tenby_position :
  alphabetical_position word letters = 75 := by
  sorry

end NUMINAMATH_CALUDE_tenby_position_l1551_155163


namespace NUMINAMATH_CALUDE_geometry_propositions_l1551_155105

-- Define the basic types
variable (α β : Plane) (l m : Line)

-- Define the relationships
def perpendicular_to_plane (line : Line) (plane : Plane) : Prop := sorry
def contained_in_plane (line : Line) (plane : Plane) : Prop := sorry
def parallel_planes (plane1 plane2 : Plane) : Prop := sorry
def perpendicular_planes (plane1 plane2 : Plane) : Prop := sorry
def perpendicular_lines (line1 line2 : Line) : Prop := sorry
def parallel_lines (line1 line2 : Line) : Prop := sorry

-- State the theorem
theorem geometry_propositions 
  (h1 : perpendicular_to_plane l α) 
  (h2 : contained_in_plane m β) :
  (parallel_planes α β → perpendicular_lines l m) ∧ 
  ¬(perpendicular_planes α β → parallel_lines l m) ∧
  (parallel_lines l m → perpendicular_planes α β) := by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l1551_155105


namespace NUMINAMATH_CALUDE_total_books_l1551_155109

/-- The total number of books Tim, Sam, and Emma have together is 133. -/
theorem total_books (tim_books sam_books emma_books : ℕ) 
  (h1 : tim_books = 44)
  (h2 : sam_books = 52)
  (h3 : emma_books = 37) : 
  tim_books + sam_books + emma_books = 133 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l1551_155109


namespace NUMINAMATH_CALUDE_hall_reunion_attendance_l1551_155154

theorem hall_reunion_attendance (total : ℕ) (oates : ℕ) (both : ℕ) (hall : ℕ) : 
  total = 150 → oates = 70 → both = 28 → total = oates + hall - both → hall = 108 := by
  sorry

end NUMINAMATH_CALUDE_hall_reunion_attendance_l1551_155154


namespace NUMINAMATH_CALUDE_lines_intersection_l1551_155168

/-- Represents a 2D point or vector -/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Represents a parametric line in 2D -/
structure ParametricLine where
  origin : Point2D
  direction : Point2D

def line1 : ParametricLine := {
  origin := { x := 2, y := 3 },
  direction := { x := 3, y := -1 }
}

def line2 : ParametricLine := {
  origin := { x := 4, y := 1 },
  direction := { x := 1, y := 5 }
}

def intersection : Point2D := {
  x := 26 / 7,
  y := 17 / 7
}

/-- 
  Theorem: The point (26/7, 17/7) is the unique intersection point of the two given lines.
-/
theorem lines_intersection (t u : ℚ) : 
  (∃! p : Point2D, 
    p.x = line1.origin.x + t * line1.direction.x ∧ 
    p.y = line1.origin.y + t * line1.direction.y ∧
    p.x = line2.origin.x + u * line2.direction.x ∧ 
    p.y = line2.origin.y + u * line2.direction.y) ∧
  (intersection.x = line1.origin.x + t * line1.direction.x) ∧
  (intersection.y = line1.origin.y + t * line1.direction.y) ∧
  (intersection.x = line2.origin.x + u * line2.direction.x) ∧
  (intersection.y = line2.origin.y + u * line2.direction.y) :=
by sorry

end NUMINAMATH_CALUDE_lines_intersection_l1551_155168


namespace NUMINAMATH_CALUDE_train_speed_train_speed_is_24_l1551_155143

theorem train_speed (person_speed : ℝ) (overtake_time : ℝ) (train_length : ℝ) : ℝ :=
  let relative_speed := train_length / overtake_time * 3600 / 1000
  relative_speed + person_speed

#check train_speed 4 9 49.999999999999986 = 24

theorem train_speed_is_24 :
  train_speed 4 9 49.999999999999986 = 24 := by sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_is_24_l1551_155143


namespace NUMINAMATH_CALUDE_chord_length_when_m_1_shortest_chord_line_equation_l1551_155102

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 11 = 0

-- Define the line l
def line_l (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Define the chord length function
noncomputable def chord_length (m : ℝ) : ℝ := sorry

-- Define the shortest chord condition
def is_shortest_chord (m : ℝ) : Prop := 
  ∀ m', chord_length m ≤ chord_length m'

-- Theorem 1: Chord length when m = 1
theorem chord_length_when_m_1 : 
  chord_length 1 = 6 * Real.sqrt 13 / 13 := sorry

-- Theorem 2: Equation of line l for shortest chord
theorem shortest_chord_line_equation :
  ∃ m, is_shortest_chord m ∧ 
    ∀ x y, line_l m x y ↔ x - y - 2 = 0 := sorry

end NUMINAMATH_CALUDE_chord_length_when_m_1_shortest_chord_line_equation_l1551_155102


namespace NUMINAMATH_CALUDE_min_value_range_l1551_155138

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- State the theorem
theorem min_value_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 a, f x ≥ f a) → a ∈ Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_min_value_range_l1551_155138


namespace NUMINAMATH_CALUDE_inverse_g_at_neg43_l1551_155148

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x^3 - 3

-- State the theorem
theorem inverse_g_at_neg43 :
  Function.invFun g (-43) = -2 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_at_neg43_l1551_155148


namespace NUMINAMATH_CALUDE_stationery_store_bundles_l1551_155142

/-- Given the number of red and blue sheets of paper and the number of sheets per bundle,
    calculates the maximum number of complete bundles that can be made. -/
def max_bundles (red_sheets blue_sheets sheets_per_bundle : ℕ) : ℕ :=
  (red_sheets + blue_sheets) / sheets_per_bundle

/-- Proves that with 210 red sheets, 473 blue sheets, and 100 sheets per bundle,
    the maximum number of complete bundles is 6. -/
theorem stationery_store_bundles :
  max_bundles 210 473 100 = 6 := by
  sorry

#eval max_bundles 210 473 100

end NUMINAMATH_CALUDE_stationery_store_bundles_l1551_155142


namespace NUMINAMATH_CALUDE_third_set_size_l1551_155185

/-- The number of students in the third set that satisfies the given conditions -/
def third_set_students : ℕ := 60

/-- The pass percentage of the whole set -/
def total_pass_percentage : ℚ := 266 / 300

theorem third_set_size :
  let first_set := 40
  let second_set := 50
  let first_pass_rate := 1
  let second_pass_rate := 9 / 10
  let third_pass_rate := 4 / 5
  (first_set * first_pass_rate + second_set * second_pass_rate + third_set_students * third_pass_rate) /
    (first_set + second_set + third_set_students) = total_pass_percentage := by
  sorry

#check third_set_size

end NUMINAMATH_CALUDE_third_set_size_l1551_155185


namespace NUMINAMATH_CALUDE_range_of_function_l1551_155130

theorem range_of_function (y : ℝ) : 
  (∃ x : ℝ, y = x / (1 + x^2)) ↔ -1/2 ≤ y ∧ y ≤ 1/2 := by
sorry

end NUMINAMATH_CALUDE_range_of_function_l1551_155130


namespace NUMINAMATH_CALUDE_digit_puzzle_l1551_155172

theorem digit_puzzle :
  ∀ (A B C D E F G H M : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0 ∧ F ≠ 0 ∧ G ≠ 0 ∧ H ≠ 0 ∧ M ≠ 0 →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ M →
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ M →
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ M →
    D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ M →
    E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ M →
    F ≠ G ∧ F ≠ H ∧ F ≠ M →
    G ≠ H ∧ G ≠ M →
    H ≠ M →
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10 ∧ M < 10 →
    A + B = 14 →
    M / G = M - F ∧ M - F = H - C →
    D * F = 24 →
    B + E = 16 →
    H = 4 :=
by sorry

end NUMINAMATH_CALUDE_digit_puzzle_l1551_155172


namespace NUMINAMATH_CALUDE_head_start_value_l1551_155123

/-- A race between two runners A and B -/
structure Race where
  length : ℝ
  speed_ratio : ℝ
  head_start : ℝ

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.length = 100 ∧ r.speed_ratio = 2 ∧ r.head_start > 0

/-- Both runners finish at the same time -/
def equal_finish_time (r : Race) : Prop :=
  r.length / r.speed_ratio = (r.length - r.head_start) / 1

theorem head_start_value (r : Race) 
  (h1 : race_conditions r) 
  (h2 : equal_finish_time r) : 
  r.head_start = 50 := by
  sorry

#check head_start_value

end NUMINAMATH_CALUDE_head_start_value_l1551_155123


namespace NUMINAMATH_CALUDE_seashell_count_l1551_155128

def initial_seashells (name : String) : ℕ :=
  match name with
  | "Henry" => 11
  | "John" => 24
  | "Adam" => 17
  | "Leo" => 83 - (11 + 24 + 17)
  | _ => 0

def final_seashells (name : String) : ℕ :=
  match name with
  | "Henry" => initial_seashells "Henry" + 3
  | "John" => initial_seashells "John" - 5
  | "Adam" => initial_seashells "Adam"
  | "Leo" => initial_seashells "Leo" - (initial_seashells "Leo" / 10 * 4) + 5
  | _ => 0

theorem seashell_count :
  final_seashells "Henry" + final_seashells "John" + 
  final_seashells "Adam" + final_seashells "Leo" = 74 :=
by sorry

end NUMINAMATH_CALUDE_seashell_count_l1551_155128


namespace NUMINAMATH_CALUDE_smallest_valid_configuration_l1551_155184

/-- Represents a bench configuration at a concert --/
structure BenchConfiguration where
  M : ℕ  -- Number of bench sections
  adultsPerBench : ℕ  -- Number of adults per bench
  childrenPerBench : ℕ  -- Number of children per bench

/-- Checks if a given bench configuration is valid --/
def isValidConfiguration (config : BenchConfiguration) : Prop :=
  ∃ (adults children : ℕ),
    adults + children = config.M * config.adultsPerBench ∧
    children = 2 * adults ∧
    children ≤ config.M * config.childrenPerBench

/-- The theorem to be proved --/
theorem smallest_valid_configuration :
  ∃ (config : BenchConfiguration),
    config.M = 6 ∧
    config.adultsPerBench = 8 ∧
    config.childrenPerBench = 12 ∧
    isValidConfiguration config ∧
    (∀ (otherConfig : BenchConfiguration),
      otherConfig.adultsPerBench = 8 →
      otherConfig.childrenPerBench = 12 →
      isValidConfiguration otherConfig →
      otherConfig.M ≥ config.M) :=
  sorry

end NUMINAMATH_CALUDE_smallest_valid_configuration_l1551_155184


namespace NUMINAMATH_CALUDE_sum_of_solutions_l1551_155183

theorem sum_of_solutions (x y : ℝ) (h1 : x + 6 * y = 12) (h2 : 3 * x - 2 * y = 8) : x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l1551_155183


namespace NUMINAMATH_CALUDE_total_students_count_l1551_155173

/-- The total number of students in a primary school height survey. -/
def total_students : ℕ := 621

/-- The number of students with heights not exceeding 130 cm. -/
def students_under_130 : ℕ := 99

/-- The average height of students not exceeding 130 cm, in cm. -/
def avg_height_under_130 : ℝ := 122

/-- The number of students with heights not less than 160 cm. -/
def students_over_160 : ℕ := 72

/-- The average height of students not less than 160 cm, in cm. -/
def avg_height_over_160 : ℝ := 163

/-- The average height of students exceeding 130 cm, in cm. -/
def avg_height_130_to_160 : ℝ := 155

/-- The average height of students below 160 cm, in cm. -/
def avg_height_under_160 : ℝ := 148

/-- Theorem stating that given the conditions, the total number of students is 621. -/
theorem total_students_count : total_students = students_under_130 + students_over_160 + 
  (total_students - students_under_130 - students_over_160) :=
by sorry

end NUMINAMATH_CALUDE_total_students_count_l1551_155173


namespace NUMINAMATH_CALUDE_speech_competition_probability_l1551_155122

theorem speech_competition_probability 
  (m n : ℕ) 
  (prob_at_least_one_female : ℝ) 
  (h1 : prob_at_least_one_female = 4/5) :
  1 - prob_at_least_one_female = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_speech_competition_probability_l1551_155122


namespace NUMINAMATH_CALUDE_min_value_expression_l1551_155171

theorem min_value_expression (x y z : ℝ) (h : z = Real.sin x) :
  ∃ (m : ℝ), (∀ (x' y' z' : ℝ), z' = Real.sin x' →
    (y' * Real.cos x' - 2)^2 + (y' + z' + 1)^2 ≥ m) ∧
  m = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1551_155171


namespace NUMINAMATH_CALUDE_curve_properties_l1551_155121

-- Define the curve
def curve (x y : ℝ) : Prop := x * y = 6

-- Define the property of the tangent being bisected
def tangent_bisected (x y : ℝ) : Prop :=
  ∀ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 →
    (curve x y) →
    (a * y = x * b) →
    ((x - 0) ^ 2 + (y - 0) ^ 2 = (a - x) ^ 2 + (0 - y) ^ 2) ∧
    ((x - 0) ^ 2 + (y - 0) ^ 2 = (0 - x) ^ 2 + (b - y) ^ 2)

theorem curve_properties :
  (curve 2 3) ∧
  (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → curve x y → tangent_bisected x y) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l1551_155121
