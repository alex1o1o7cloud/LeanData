import Mathlib

namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_one_l1991_199108

theorem expression_simplification_and_evaluation (x : ℝ) (h : x ≠ 3) :
  (x^2 - 5) / (x - 3) - 4 / (x - 3) = x + 3 :=
by sorry

theorem expression_evaluation_at_one :
  ((1 : ℝ)^2 - 5) / (1 - 3) - 4 / (1 - 3) = 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_expression_evaluation_at_one_l1991_199108


namespace NUMINAMATH_CALUDE_waiter_customers_waiter_customers_proof_l1991_199130

theorem waiter_customers : ℕ → Prop :=
  fun initial_customers =>
    initial_customers - 14 + 36 = 41 →
    initial_customers = 19

-- The proof of the theorem
theorem waiter_customers_proof : ∃ x : ℕ, waiter_customers x :=
  sorry

end NUMINAMATH_CALUDE_waiter_customers_waiter_customers_proof_l1991_199130


namespace NUMINAMATH_CALUDE_password_factorization_l1991_199186

/-- Represents the correspondence between algebraic expressions and words --/
def word_mapping (x y a b : ℝ) : List (ℝ × String) :=
  [(a - b, "学"), (x - y, "我"), (x + y, "爱"), (a + b, "数"),
   (x^2 - y^2, "游"), (a^2 - b^2, "美")]

/-- The main theorem stating the factorization and its word representation --/
theorem password_factorization (x y a b : ℝ) :
  ∃ (result : String),
    ((x^2 - y^2) * a^2 - (x^2 - y^2) * b^2 = (x + y) * (x - y) * (a + b) * (a - b)) ∧
    (result = "我爱数学") ∧
    (∀ (expr : ℝ) (word : String),
      (expr, word) ∈ word_mapping x y a b →
      word ∈ ["我", "爱", "数", "学"]) :=
by
  sorry


end NUMINAMATH_CALUDE_password_factorization_l1991_199186


namespace NUMINAMATH_CALUDE_binomial_inequalities_l1991_199148

theorem binomial_inequalities (n : ℕ) (h : n ≥ 2) :
  (2 : ℝ)^n < (Nat.choose (2*n) n : ℝ) ∧
  (Nat.choose (2*n) n : ℝ) < 4^n ∧
  (Nat.choose (2*n - 1) n : ℝ) < 4^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequalities_l1991_199148


namespace NUMINAMATH_CALUDE_reciprocal_proof_l1991_199151

theorem reciprocal_proof (a b : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_diff : a ≠ b) 
  (h_eq : 1 / (1 + a) + 1 / (1 + b) = 2 / (1 + Real.sqrt (a * b))) : 
  a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_proof_l1991_199151


namespace NUMINAMATH_CALUDE_complex_conjugate_roots_imply_zero_coefficients_l1991_199163

/-- Given a quadratic equation z^2 + (6 + pi)z + (10 + qi) = 0 where p and q are real numbers,
    if the roots are complex conjugates, then p = 0 and q = 0 -/
theorem complex_conjugate_roots_imply_zero_coefficients (p q : ℝ) :
  (∃ x y : ℝ, (Complex.I : ℂ)^2 = -1 ∧
    (x + y * Complex.I) * (x - y * Complex.I) = -(6 + p * Complex.I) * (x + y * Complex.I) - (10 + q * Complex.I)) →
  p = 0 ∧ q = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_conjugate_roots_imply_zero_coefficients_l1991_199163


namespace NUMINAMATH_CALUDE_base5_subtraction_l1991_199113

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The first number in base 5 -/
def num1 : List Nat := [1, 2, 3, 4]

/-- The second number in base 5 -/
def num2 : List Nat := [2, 3, 4]

/-- The expected difference in base 5 -/
def expected_diff : List Nat := [1, 0, 0, 0]

theorem base5_subtraction :
  decimalToBase5 (base5ToDecimal num1 - base5ToDecimal num2) = expected_diff := by
  sorry

end NUMINAMATH_CALUDE_base5_subtraction_l1991_199113


namespace NUMINAMATH_CALUDE_bobby_candy_count_l1991_199115

theorem bobby_candy_count (initial : ℕ) (additional : ℕ) : 
  initial = 26 → additional = 17 → initial + additional = 43 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_count_l1991_199115


namespace NUMINAMATH_CALUDE_largest_number_with_property_l1991_199132

def is_common_multiple (n m k : ℕ) : Prop := n ∣ k ∧ m ∣ k

theorem largest_number_with_property : 
  ∀ n : ℕ, n ≤ 100 → (is_common_multiple 7 8 (n - 2) → n ≤ 58) ∧ 
  (is_common_multiple 7 8 (58 - 2)) :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_property_l1991_199132


namespace NUMINAMATH_CALUDE_area_circular_segment_equilateral_triangle_l1991_199117

/-- The area of a circular segment cut off by one side of an inscribed equilateral triangle -/
theorem area_circular_segment_equilateral_triangle (a : ℝ) (ha : a > 0) :
  let R := a / Real.sqrt 3
  let sector_area := π * R^2 / 3
  let triangle_area := a * R / 4
  sector_area - triangle_area = (a^2 * (4 * π - 3 * Real.sqrt 3)) / 36 := by
  sorry

end NUMINAMATH_CALUDE_area_circular_segment_equilateral_triangle_l1991_199117


namespace NUMINAMATH_CALUDE_subset_sum_theorem_l1991_199119

theorem subset_sum_theorem (A : Finset ℤ) (h_card : A.card = 4) 
  (h_order : ∃ (a₁ a₂ a₃ a₄ : ℤ), A = {a₁, a₂, a₃, a₄} ∧ a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄) 
  (h_subset_sums : (A.powerset.filter (fun s => s.card = 3)).image (fun s => s.sum id) = {-1, 3, 5, 8}) :
  A = {-3, 0, 2, 6} := by
sorry

end NUMINAMATH_CALUDE_subset_sum_theorem_l1991_199119


namespace NUMINAMATH_CALUDE_subtraction_multiplication_equality_l1991_199104

theorem subtraction_multiplication_equality : (2000000000000 - 1234567890123) * 3 = 2296296329631 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_multiplication_equality_l1991_199104


namespace NUMINAMATH_CALUDE_sqrt_sum_of_powers_l1991_199134

theorem sqrt_sum_of_powers : Real.sqrt (5^4 + 5^4 + 5^4 + 2^4) = Real.sqrt 1891 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_of_powers_l1991_199134


namespace NUMINAMATH_CALUDE_two_in_M_l1991_199166

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the complement of M in U
def complementM : Set Nat := {1, 3}

-- Theorem to prove
theorem two_in_M : 2 ∈ (U \ complementM) := by
  sorry

end NUMINAMATH_CALUDE_two_in_M_l1991_199166


namespace NUMINAMATH_CALUDE_additional_cakes_count_l1991_199199

/-- Represents the number of cakes Baker initially made -/
def initial_cakes : ℕ := 62

/-- Represents the number of cakes Baker sold -/
def sold_cakes : ℕ := 144

/-- Represents the number of cakes Baker still has -/
def remaining_cakes : ℕ := 67

/-- Theorem stating the number of additional cakes Baker made -/
theorem additional_cakes_count : 
  ∃ x : ℕ, initial_cakes + x - sold_cakes = remaining_cakes ∧ x = 149 := by
  sorry

end NUMINAMATH_CALUDE_additional_cakes_count_l1991_199199


namespace NUMINAMATH_CALUDE_average_birds_seen_l1991_199124

def marcus_birds : ℕ := 7
def humphrey_birds : ℕ := 11
def darrel_birds : ℕ := 9
def total_watchers : ℕ := 3

theorem average_birds_seen :
  (marcus_birds + humphrey_birds + darrel_birds) / total_watchers = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_birds_seen_l1991_199124


namespace NUMINAMATH_CALUDE_highest_demand_week_sales_total_sales_check_l1991_199194

-- Define the sales for each week
def first_week_sales : ℕ := 20
def second_week_sales : ℕ := 3 * first_week_sales
def third_week_sales : ℕ := 2 * first_week_sales
def fourth_week_sales : ℕ := first_week_sales

-- Define the total sales for the month
def total_sales : ℕ := 300

-- Theorem to prove the highest demand week
theorem highest_demand_week_sales :
  max first_week_sales (max second_week_sales (max third_week_sales fourth_week_sales)) = 60 :=
by sorry

-- Verify that the sum of all weeks' sales equals the total monthly sales
theorem total_sales_check :
  first_week_sales + second_week_sales + third_week_sales + fourth_week_sales = total_sales :=
by sorry

end NUMINAMATH_CALUDE_highest_demand_week_sales_total_sales_check_l1991_199194


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_related_unknowns_l1991_199135

theorem arithmetic_mean_of_related_unknowns (x y z : ℝ) 
  (h : (1 : ℝ) / (x * y) = y / (z - x + 1) ∧ y / (z - x + 1) = 2 / (z + 1)) : 
  x = (y + z) / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_related_unknowns_l1991_199135


namespace NUMINAMATH_CALUDE_f_properties_l1991_199160

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := -1/a + 2/x

theorem f_properties :
  (∀ x > 0, ∀ y > 0, x < y → f a x > f a y) ∧
  (a < 0 → ∀ x > 0, f a x > 0) ∧
  (a > 0 → ∀ x ∈ Set.Ioo 0 (2*a), f a x > 0) ∧
  (a > 0 → ∀ x > 2*a, f a x ≤ 0) ∧
  (a < 0 ∨ a ≥ 1/4 ↔ ∀ x > 0, f a x + 2*x ≥ 0) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l1991_199160


namespace NUMINAMATH_CALUDE_jim_gas_cost_l1991_199171

/-- The total amount spent on gas by Jim -/
def total_gas_cost (nc_gallons : ℕ) (nc_price : ℚ) (va_gallons : ℕ) (va_price_increase : ℚ) : ℚ :=
  (nc_gallons : ℚ) * nc_price + (va_gallons : ℚ) * (nc_price + va_price_increase)

/-- Theorem stating that Jim's total gas cost is $50.00 -/
theorem jim_gas_cost :
  total_gas_cost 10 2 10 1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_jim_gas_cost_l1991_199171


namespace NUMINAMATH_CALUDE_person_savings_l1991_199189

theorem person_savings (income expenditure savings : ℕ) : 
  (income : ℚ) / expenditure = 5 / 4 →
  income = 15000 →
  savings = income - expenditure →
  savings = 3000 := by
sorry

end NUMINAMATH_CALUDE_person_savings_l1991_199189


namespace NUMINAMATH_CALUDE_rohan_farm_size_l1991_199183

/-- Represents the characteristics of Rohan's coconut farm and its earnings -/
structure CoconutFarm where
  trees_per_sqm : ℕ := 2
  coconuts_per_tree : ℕ := 6
  harvest_period_months : ℕ := 3
  coconut_price : ℚ := 1/2
  total_earnings : ℚ := 240
  total_period_months : ℕ := 6

/-- Calculates the size of the coconut farm based on given parameters -/
def farm_size (farm : CoconutFarm) : ℚ :=
  farm.total_earnings / (farm.trees_per_sqm * farm.coconuts_per_tree * farm.coconut_price * (farm.total_period_months / farm.harvest_period_months))

/-- Theorem stating that Rohan's coconut farm size is 20 square meters -/
theorem rohan_farm_size (farm : CoconutFarm) : farm_size farm = 20 := by
  sorry

end NUMINAMATH_CALUDE_rohan_farm_size_l1991_199183


namespace NUMINAMATH_CALUDE_hex_351_equals_849_l1991_199181

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3
  | '4' => 4 | '5' => 5 | '6' => 6 | '7' => 7
  | '8' => 8 | '9' => 9 | 'A' => 10 | 'B' => 11
  | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hex_to_dec c) 0

/-- Theorem: The hexadecimal number 351 is equal to 849 in decimal -/
theorem hex_351_equals_849 : hex_string_to_dec "351" = 849 := by
  sorry

end NUMINAMATH_CALUDE_hex_351_equals_849_l1991_199181


namespace NUMINAMATH_CALUDE_camp_recoloring_l1991_199196

/-- A graph representing friendships in a summer camp -/
structure CampGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  degree_eleven : ∀ v ∈ vertices, (edges.filter (fun e => e.1 = v ∨ e.2 = v)).card = 11
  symmetric : ∀ ⦃a b⦄, (a, b) ∈ edges → (b, a) ∈ edges

/-- A valid coloring of the graph -/
def ValidColoring (G : CampGraph) (coloring : Nat → Fin 7) : Prop :=
  ∀ ⦃a b⦄, (a, b) ∈ G.edges → coloring a ≠ coloring b

theorem camp_recoloring (G : CampGraph) (initial_coloring : Nat → Fin 7)
    (h_valid : ValidColoring G initial_coloring)
    (fixed_vertices : Finset Nat)
    (h_fixed_size : fixed_vertices.card = 100)
    (h_fixed_subset : fixed_vertices ⊆ G.vertices) :
    ∃ (new_coloring : Nat → Fin 7),
      ValidColoring G new_coloring ∧
      (∃ v ∈ G.vertices \ fixed_vertices, new_coloring v ≠ initial_coloring v) ∧
      (∀ v ∈ fixed_vertices, new_coloring v = initial_coloring v) :=
  sorry

end NUMINAMATH_CALUDE_camp_recoloring_l1991_199196


namespace NUMINAMATH_CALUDE_sixth_term_equals_five_l1991_199101

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 6th term of the sequence equals 5 given the conditions -/
theorem sixth_term_equals_five (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 2 + a 6 + a 10 = 15) : 
  a 6 = 5 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_equals_five_l1991_199101


namespace NUMINAMATH_CALUDE_secret_spread_day_l1991_199127

/-- The number of people who know the secret on day n -/
def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2

/-- The day when the secret is known by 6560 people -/
def target_day : ℕ := 8

theorem secret_spread_day : secret_spread target_day = 6560 := by
  sorry

#eval secret_spread target_day

end NUMINAMATH_CALUDE_secret_spread_day_l1991_199127


namespace NUMINAMATH_CALUDE_coefficient_x3_is_negative_five_l1991_199150

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (x+1)(x-2)^3
def coefficient_x3 : ℤ :=
  (1 * (binomial 3 0)) + (-2 * (binomial 3 1))

-- Theorem statement
theorem coefficient_x3_is_negative_five :
  coefficient_x3 = -5 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_is_negative_five_l1991_199150


namespace NUMINAMATH_CALUDE_father_double_son_age_l1991_199102

/-- Represents the ages of a father and son, and the time until the father's age is twice the son's. -/
structure FatherSonAges where
  sonAge : ℕ
  fatherAge : ℕ
  yearsUntilDouble : ℕ

/-- The condition that the father is 25 years older than the son -/
def ageDifference (ages : FatherSonAges) : Prop :=
  ages.fatherAge = ages.sonAge + 25

/-- The condition that after a certain number of years, the father's age will be twice the son's -/
def doubleAgeCondition (ages : FatherSonAges) : Prop :=
  ages.fatherAge + ages.yearsUntilDouble = 2 * (ages.sonAge + ages.yearsUntilDouble)

/-- The main theorem stating that given the initial conditions, it will take 2 years for the father's age to be twice the son's -/
theorem father_double_son_age :
  ∀ (ages : FatherSonAges),
  ages.sonAge = 23 →
  ageDifference ages →
  doubleAgeCondition ages →
  ages.yearsUntilDouble = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_father_double_son_age_l1991_199102


namespace NUMINAMATH_CALUDE_simplify_square_root_sum_l1991_199174

theorem simplify_square_root_sum : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 56) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_root_sum_l1991_199174


namespace NUMINAMATH_CALUDE_lcd_of_fractions_l1991_199156

def fractions : List Nat := [3, 4, 5, 8, 9, 11]

theorem lcd_of_fractions : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 3 4) 5) 8) 9) 11 = 3960 := by
  sorry

end NUMINAMATH_CALUDE_lcd_of_fractions_l1991_199156


namespace NUMINAMATH_CALUDE_car_journey_time_proof_l1991_199190

/-- Represents the speed and distance of a car's journey -/
structure CarJourney where
  speed : ℝ
  distance : ℝ
  time : ℝ

/-- Given two car journeys, proves that the time taken by the second car
    is 4/3 hours under specific conditions -/
theorem car_journey_time_proof
  (m n : CarJourney)
  (h1 : m.time = 4)
  (h2 : n.speed = 3 * m.speed)
  (h3 : n.distance = 3 * m.distance)
  (h4 : m.distance = m.speed * m.time)
  (h5 : n.distance = n.speed * n.time) :
  n.time = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_car_journey_time_proof_l1991_199190


namespace NUMINAMATH_CALUDE_g_1003_fixed_point_l1991_199191

def g₁ (x : ℚ) : ℚ := 1/2 - 4/(4*x+2)

def g (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | 1 => g₁ x
  | n+1 => g₁ (g n x)

theorem g_1003_fixed_point :
  g 1003 (11/2) = 11/2 - 4 := by sorry

end NUMINAMATH_CALUDE_g_1003_fixed_point_l1991_199191


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l1991_199167

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given that the man is 26 years older than his son and the son's current age is 24 years. -/
theorem mans_age_to_sons_age_ratio (sons_current_age : ℕ) (age_difference : ℕ) : 
  sons_current_age = 24 →
  age_difference = 26 →
  (sons_current_age + age_difference + 2) / (sons_current_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l1991_199167


namespace NUMINAMATH_CALUDE_geometric_progression_relation_l1991_199177

/-- Given two geometric progressions, prove that their first terms are related as stated. -/
theorem geometric_progression_relation (a b q : ℝ) (n : ℕ) (h : q ≠ 1) :
  (a * (q^(2*n) - 1)) / (q - 1) = (b * (q^(2*n) - 1)) / (q^2 - 1) →
  b = a + a * q :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_relation_l1991_199177


namespace NUMINAMATH_CALUDE_henley_candy_problem_l1991_199153

theorem henley_candy_problem :
  ∀ (total_candies : ℕ),
    (total_candies : ℚ) * (60 : ℚ) / 100 = 3 * 60 →
    total_candies = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_henley_candy_problem_l1991_199153


namespace NUMINAMATH_CALUDE_dollar_sum_squared_zero_l1991_199109

/-- The dollar operation for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem: For real numbers x and y, (x + y)²$(y + x)² = 0 -/
theorem dollar_sum_squared_zero (x y : ℝ) : dollar ((x + y)^2) ((y + x)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dollar_sum_squared_zero_l1991_199109


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l1991_199128

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 3 ∧ b = 5 ∧ ∀ (x : ℕ), x > 3 ∧ x < b → ¬∃ (y : ℕ), 4 * x + 5 = y ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l1991_199128


namespace NUMINAMATH_CALUDE_leftover_value_is_correct_l1991_199133

/-- Calculate the value of leftover coins after combining and rolling --/
def leftover_value (james_quarters james_dimes emily_quarters emily_dimes : ℕ)
  (quarters_per_roll dimes_per_roll : ℕ) : ℚ :=
  let total_quarters := james_quarters + emily_quarters
  let total_dimes := james_dimes + emily_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * (1 / 4) + (leftover_dimes : ℚ) * (1 / 10)

/-- The main theorem --/
theorem leftover_value_is_correct :
  leftover_value 65 134 103 229 40 50 = 33 / 10 :=
by sorry

end NUMINAMATH_CALUDE_leftover_value_is_correct_l1991_199133


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l1991_199152

theorem x_range_for_inequality (x : ℝ) :
  (∀ a : ℝ, a ∈ Set.Icc 0 2 → a * x^2 + (a + 1) * x + 1 - (3/2) * a < 0) →
  x ∈ Set.Ioo (-2) (-1) := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l1991_199152


namespace NUMINAMATH_CALUDE_floor_to_total_ratio_example_l1991_199114

/-- The ratio of students sitting on the floor to the total number of students -/
def floor_to_total_ratio (total_students floor_students : ℕ) : ℚ :=
  floor_students / total_students

/-- Proof that the ratio of students sitting on the floor to the total number of students is 11/26 -/
theorem floor_to_total_ratio_example : 
  floor_to_total_ratio 26 11 = 11 / 26 := by
  sorry

end NUMINAMATH_CALUDE_floor_to_total_ratio_example_l1991_199114


namespace NUMINAMATH_CALUDE_factorization_equality_l1991_199143

theorem factorization_equality (x y : ℝ) :
  3 * y * (y^2 - 4) + 5 * x * (y^2 - 4) = (3*y + 5*x) * (y + 2) * (y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1991_199143


namespace NUMINAMATH_CALUDE_donation_ratio_l1991_199197

theorem donation_ratio (margo_donation julie_donation : ℕ) 
  (h1 : margo_donation = 4300)
  (h2 : julie_donation = 4700) :
  (julie_donation - margo_donation) / (margo_donation + julie_donation) = 2 / 45 := by
  sorry

end NUMINAMATH_CALUDE_donation_ratio_l1991_199197


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1991_199188

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + 3*x - 4 < 0} = Set.Ioo (-4 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1991_199188


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1991_199192

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f := fun x : ℝ => a * x^2 + b * x + c
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s : ℝ, ∀ x : ℝ, f x = 0 → (∃ y : ℝ, f y = 0 ∧ y ≠ x ∧ x + y = s)) →
  s = -b / a :=
by sorry

theorem sum_of_roots_specific_equation :
  let f := fun x : ℝ => x^2 - 5*x + 6 - 9
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s : ℝ, ∀ x : ℝ, f x = 0 → (∃ y : ℝ, f y = 0 ∧ y ≠ x ∧ x + y = s)) →
  s = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1991_199192


namespace NUMINAMATH_CALUDE_regular_soda_bottles_l1991_199138

theorem regular_soda_bottles (total_bottles : ℕ) (diet_bottles : ℕ) 
  (h1 : total_bottles = 17) 
  (h2 : diet_bottles = 8) : 
  total_bottles - diet_bottles = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_soda_bottles_l1991_199138


namespace NUMINAMATH_CALUDE_imaginaria_city_population_l1991_199147

theorem imaginaria_city_population : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + 225 = b^2 + 1 ∧
  b^2 + 76 = c^2 ∧
  5 ∣ a^2 := by
sorry

end NUMINAMATH_CALUDE_imaginaria_city_population_l1991_199147


namespace NUMINAMATH_CALUDE_gcd_6363_1923_l1991_199161

theorem gcd_6363_1923 : Nat.gcd 6363 1923 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_6363_1923_l1991_199161


namespace NUMINAMATH_CALUDE_escalator_steps_l1991_199168

/-- The number of steps on an escalator between two floors -/
def N : ℕ := 47

/-- The number of steps Jack walks while on the moving escalator -/
def jack_steps : ℕ := 29

/-- The number of steps Jill walks while on the moving escalator -/
def jill_steps : ℕ := 11

/-- Jill's travel time is twice Jack's -/
def time_ratio : ℕ := 2

theorem escalator_steps :
  N - jill_steps = time_ratio * (N - jack_steps) :=
sorry

end NUMINAMATH_CALUDE_escalator_steps_l1991_199168


namespace NUMINAMATH_CALUDE_f5_computation_l1991_199172

/-- A function that represents a boolean operation (AND or OR) -/
def BoolOp : Type := Bool → Bool → Bool

/-- Compute f₅ using only 5 boolean operations -/
def compute_f5 (x₁ x₂ x₃ x₄ x₅ : Bool) (op₁ op₂ op₃ op₄ op₅ : BoolOp) : Bool :=
  let x₆ := op₁ x₁ x₃
  let x₇ := op₂ x₂ x₆
  let x₈ := op₃ x₃ x₅
  let x₉ := op₄ x₄ x₈
  op₅ x₇ x₉

/-- Theorem: f₅ can be computed using only 5 operations of conjunctions and disjunctions -/
theorem f5_computation (x₁ x₂ x₃ x₄ x₅ : Bool) :
  ∃ (op₁ op₂ op₃ op₄ op₅ : BoolOp),
    (∀ a b, op₁ a b = a ∨ b ∨ op₁ a b = a ∧ b) ∧
    (∀ a b, op₂ a b = a ∨ b ∨ op₂ a b = a ∧ b) ∧
    (∀ a b, op₃ a b = a ∨ b ∨ op₃ a b = a ∧ b) ∧
    (∀ a b, op₄ a b = a ∨ b ∨ op₄ a b = a ∧ b) ∧
    (∀ a b, op₅ a b = a ∨ b ∨ op₅ a b = a ∧ b) :=
by
  sorry


end NUMINAMATH_CALUDE_f5_computation_l1991_199172


namespace NUMINAMATH_CALUDE_calculation_proof_l1991_199136

theorem calculation_proof : (-36) / (-1/2 + 1/6 - 1/3) = 54 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1991_199136


namespace NUMINAMATH_CALUDE_game_winning_probability_l1991_199139

/-- A game with consecutive integers from 2 to 2020 -/
def game_range : Set ℕ := {n | 2 ≤ n ∧ n ≤ 2020}

/-- The total number of integers in the game -/
def total_numbers : ℕ := 2019

/-- Two numbers are coprime if their greatest common divisor is 1 -/
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The number of coprime pairs in the game range -/
def coprime_pairs : ℕ := 1010

/-- The probability of winning is the number of coprime pairs divided by the total numbers -/
theorem game_winning_probability :
  (coprime_pairs : ℚ) / total_numbers = 1010 / 2019 := by sorry

end NUMINAMATH_CALUDE_game_winning_probability_l1991_199139


namespace NUMINAMATH_CALUDE_green_eyed_brunettes_l1991_199146

theorem green_eyed_brunettes (total : ℕ) (blueEyedBlondes : ℕ) (brunettes : ℕ) (greenEyed : ℕ) :
  total = 60 →
  blueEyedBlondes = 20 →
  brunettes = 35 →
  greenEyed = 25 →
  ∃ (greenEyedBrunettes : ℕ),
    greenEyedBrunettes = 10 ∧
    greenEyedBrunettes ≤ brunettes ∧
    greenEyedBrunettes ≤ greenEyed ∧
    blueEyedBlondes + (brunettes - greenEyedBrunettes) + greenEyed = total :=
by
  sorry

end NUMINAMATH_CALUDE_green_eyed_brunettes_l1991_199146


namespace NUMINAMATH_CALUDE_function_property_result_l1991_199173

theorem function_property_result (g : ℝ → ℝ) 
    (h : ∀ a c : ℝ, c^3 * g a = a^3 * g c) 
    (h_nonzero : g 3 ≠ 0) : 
  (g 6 - g 2) / g 3 = 208/27 := by
sorry

end NUMINAMATH_CALUDE_function_property_result_l1991_199173


namespace NUMINAMATH_CALUDE_trig_identity_l1991_199122

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos (2 * α) + Real.sin (2 * α)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1991_199122


namespace NUMINAMATH_CALUDE_gustran_facial_cost_l1991_199110

/-- Represents the prices of services at a salon -/
structure SalonPrices where
  haircut : ℕ
  nails : ℕ
  facial : ℕ

/-- Calculates the total cost of services at a salon -/
def totalCost (prices : SalonPrices) : ℕ :=
  prices.haircut + prices.nails + prices.facial

theorem gustran_facial_cost (gustran : SalonPrices) (barbara : SalonPrices) (fancy : SalonPrices)
  (h1 : gustran.haircut = 45)
  (h2 : gustran.nails = 30)
  (h3 : barbara.haircut = 30)
  (h4 : barbara.nails = 40)
  (h5 : barbara.facial = 28)
  (h6 : fancy.haircut = 34)
  (h7 : fancy.nails = 20)
  (h8 : fancy.facial = 30)
  (h9 : totalCost fancy = 84)
  (h10 : totalCost fancy ≤ totalCost barbara)
  (h11 : totalCost fancy ≤ totalCost gustran) :
  gustran.facial = 9 := by
  sorry

end NUMINAMATH_CALUDE_gustran_facial_cost_l1991_199110


namespace NUMINAMATH_CALUDE_sphere_diameter_from_cylinder_l1991_199125

noncomputable def cylinder_volume (d h : ℝ) : ℝ := Real.pi * (d / 2)^2 * h

noncomputable def sphere_volume (d : ℝ) : ℝ := (4 / 3) * Real.pi * (d / 2)^3

theorem sphere_diameter_from_cylinder (cylinder_diameter cylinder_height : ℝ) :
  let total_volume := cylinder_volume cylinder_diameter cylinder_height
  let sphere_count := 9
  let individual_sphere_volume := total_volume / sphere_count
  let sphere_diameter := (6 * individual_sphere_volume / Real.pi)^(1/3)
  cylinder_diameter = 16 ∧ cylinder_height = 12 →
  sphere_diameter = 8 := by
  sorry

#check sphere_diameter_from_cylinder

end NUMINAMATH_CALUDE_sphere_diameter_from_cylinder_l1991_199125


namespace NUMINAMATH_CALUDE_skirt_cut_amount_l1991_199157

/-- The amount cut off the pants in inches -/
def pants_cut : ℝ := 0.5

/-- The additional amount cut off the skirt compared to the pants in inches -/
def additional_skirt_cut : ℝ := 0.25

/-- The total amount cut off the skirt in inches -/
def skirt_cut : ℝ := pants_cut + additional_skirt_cut

theorem skirt_cut_amount : skirt_cut = 0.75 := by sorry

end NUMINAMATH_CALUDE_skirt_cut_amount_l1991_199157


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1991_199111

theorem quadratic_equation_roots (a b : ℝ) (h1 : a ≠ 0) :
  (∃ x : ℝ, a * x^2 = b ∧ x = 2) → (∃ y : ℝ, a * y^2 = b ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1991_199111


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_range_is_zero_l1991_199149

def integers_range : List Int := List.range 11 |>.map (λ x => x - 5)

theorem arithmetic_mean_of_range_is_zero :
  (integers_range.sum : ℚ) / integers_range.length = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_range_is_zero_l1991_199149


namespace NUMINAMATH_CALUDE_cube_sum_ratio_equals_product_ratio_l1991_199129

theorem cube_sum_ratio_equals_product_ratio 
  (a b c d e f : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : d + e + f = 0) 
  (h3 : d * e * f ≠ 0) : 
  (a^3 + b^3 + c^3) / (d^3 + e^3 + f^3) = a * b * c / (d * e * f) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_ratio_equals_product_ratio_l1991_199129


namespace NUMINAMATH_CALUDE_least_number_with_remainder_four_l1991_199159

def is_valid_number (n : ℕ) : Prop :=
  n % 5 = 4 ∧ n % 9 = 4 ∧ n % 12 = 4 ∧ n % 18 = 4

theorem least_number_with_remainder_four :
  is_valid_number 184 ∧ ∀ m : ℕ, m < 184 → ¬ is_valid_number m :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_four_l1991_199159


namespace NUMINAMATH_CALUDE_bakery_storage_ratio_l1991_199116

/-- Bakery storage problem -/
theorem bakery_storage_ratio : 
  ∀ (flour baking_soda : ℝ),
  flour / baking_soda = 10 →
  flour / (baking_soda + 60) = 8 →
  ∃ (sugar : ℝ),
  sugar = 6000 ∧
  sugar / flour = 2.5 :=
by sorry

end NUMINAMATH_CALUDE_bakery_storage_ratio_l1991_199116


namespace NUMINAMATH_CALUDE_possible_to_end_with_80_19_specific_sequence_leads_to_80_19_l1991_199164

/-- Represents the result of a shot --/
inductive ShotResult
  | Success
  | Miss

/-- Applies the effect of a shot to the current amount --/
def applyShot (amount : ℝ) (result : ShotResult) : ℝ :=
  match result with
  | ShotResult.Success => amount * 1.1
  | ShotResult.Miss => amount * 0.9

/-- Theorem stating that it's possible to end up with 80.19 rubles --/
theorem possible_to_end_with_80_19 : ∃ (shots : List ShotResult), 
  shots.foldl applyShot 100 = 80.19 := by
  sorry

/-- Proof that the specific sequence of shots leads to 80.19 rubles --/
theorem specific_sequence_leads_to_80_19 : 
  [ShotResult.Miss, ShotResult.Miss, ShotResult.Miss, ShotResult.Success].foldl applyShot 100 = 80.19 := by
  sorry

end NUMINAMATH_CALUDE_possible_to_end_with_80_19_specific_sequence_leads_to_80_19_l1991_199164


namespace NUMINAMATH_CALUDE_calculation_proof_l1991_199140

theorem calculation_proof :
  ((-1 : ℚ)^2 + 27/4 * (-4) / (-3)^2 = -4) ∧
  ((-36 : ℚ) * (3/4 - 5/6 + 7/9) = -25) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1991_199140


namespace NUMINAMATH_CALUDE_triangle_problem_l1991_199154

/-- Given a triangle ABC with sides a and b that are roots of x^2 - 2√3x + 2 = 0,
    and 2cos(A+B) = 1, prove that angle C is 120° and side AB has length √10 -/
theorem triangle_problem (a b : ℝ) (A B C : ℝ) :
  a^2 - 2 * Real.sqrt 3 * a + 2 = 0 →
  b^2 - 2 * Real.sqrt 3 * b + 2 = 0 →
  2 * Real.cos (A + B) = 1 →
  C = 2 * π / 3 ∧
  (a^2 + b^2 - 2 * a * b * Real.cos C) = 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1991_199154


namespace NUMINAMATH_CALUDE_no_real_distinct_roots_l1991_199144

theorem no_real_distinct_roots (k : ℝ) : 
  ¬∃ (x y : ℝ), x ≠ y ∧ x^2 + 2*k*x + 3*k^2 = 0 ∧ y^2 + 2*k*y + 3*k^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_distinct_roots_l1991_199144


namespace NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l1991_199145

/-- Given a point in spherical coordinates, find its equivalent representation in standard spherical coordinates. -/
theorem spherical_coordinate_equivalence :
  ∀ (ρ θ φ : ℝ),
  ρ > 0 →
  (∃ (k : ℤ), θ = 3 * π / 8 + 2 * π * k) →
  (∃ (m : ℤ), φ = 9 * π / 5 + 2 * π * m) →
  ∃ (θ' φ' : ℝ),
    0 ≤ θ' ∧ θ' < 2 * π ∧
    0 ≤ φ' ∧ φ' ≤ π ∧
    (ρ, θ', φ') = (4, 11 * π / 8, π / 5) :=
by sorry


end NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l1991_199145


namespace NUMINAMATH_CALUDE_seven_consecutive_integers_product_first_57_integers_product_l1991_199106

-- Define a function to calculate the number of trailing zeros
def trailingZeros (n : ℕ) : ℕ := sorry

-- Theorem for seven consecutive integers
theorem seven_consecutive_integers_product (k : ℕ) :
  ∃ m : ℕ, m > 0 ∧ trailingZeros ((k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6) * (k + 7)) ≥ m :=
sorry

-- Theorem for the product of first 57 positive integers
theorem first_57_integers_product :
  trailingZeros (Nat.factorial 57) = 13 :=
sorry

end NUMINAMATH_CALUDE_seven_consecutive_integers_product_first_57_integers_product_l1991_199106


namespace NUMINAMATH_CALUDE_unique_x_with_rational_sums_l1991_199165

theorem unique_x_with_rational_sums (x : ℝ) :
  (∃ a : ℚ, x + Real.sqrt 3 = a) →
  (∃ b : ℚ, x^2 + Real.sqrt 3 = b) →
  x = 1/2 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_unique_x_with_rational_sums_l1991_199165


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1991_199105

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the radius function for circles
variable (radius : Circle → ℝ)

-- Define the distance function between two points
variable (dist : Point → Point → ℝ)

-- Define the "on_circle" predicate
variable (on_circle : Point → Circle → Prop)

-- Define the "intersect" predicate for two circles
variable (intersect : Circle → Circle → Point → Prop)

-- Define the "interior_point" predicate
variable (interior_point : Point → Circle → Prop)

-- Define the "line_intersect" predicate
variable (line_intersect : Point → Point → Circle → Point → Prop)

-- Define the "equilateral" predicate for triangles
variable (equilateral : Point → Point → Point → Prop)

-- Theorem statement
theorem circle_intersection_theorem 
  (k₁ k₂ : Circle) (O A B S T : Point) (r : ℝ) :
  radius k₁ = r →
  on_circle O k₁ →
  intersect k₁ k₂ A →
  intersect k₁ k₂ B →
  interior_point S k₁ →
  line_intersect B S k₁ T →
  equilateral A O S →
  dist T S = r :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1991_199105


namespace NUMINAMATH_CALUDE_smallest_x_value_l1991_199193

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (210 + x)) : 
  2 ≤ x.val :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1991_199193


namespace NUMINAMATH_CALUDE_same_type_ab_squared_and_neg_two_ab_squared_l1991_199158

/-- A polynomial type representing terms of the form c * a^m * b^n where c is a constant -/
structure PolynomialTerm (α : Type*) [CommRing α] where
  coeff : α
  a_exp : ℕ
  b_exp : ℕ

/-- The degree of a polynomial term -/
def PolynomialTerm.degree {α : Type*} [CommRing α] (term : PolynomialTerm α) : ℕ :=
  term.a_exp + term.b_exp

/-- Check if two polynomial terms are of the same type -/
def same_type {α : Type*} [CommRing α] (t1 t2 : PolynomialTerm α) : Prop :=
  t1.a_exp = t2.a_exp ∧ t1.b_exp = t2.b_exp

theorem same_type_ab_squared_and_neg_two_ab_squared 
  {α : Type*} [CommRing α] (a b : α) : 
  let ab_squared : PolynomialTerm α := ⟨1, 1, 2⟩
  let neg_two_ab_squared : PolynomialTerm α := ⟨-2, 1, 2⟩
  same_type ab_squared neg_two_ab_squared ∧ 
  ab_squared.degree = 3 :=
by sorry

end NUMINAMATH_CALUDE_same_type_ab_squared_and_neg_two_ab_squared_l1991_199158


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1991_199137

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.I * Complex.I = -1) →
  (∃ (z : ℂ), z = m * (m + 1) + Complex.I * (m^2 - 1) ∧ z.re = 0 ∧ z.im ≠ 0) →
  m = 0 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1991_199137


namespace NUMINAMATH_CALUDE_complex_number_simplification_l1991_199178

theorem complex_number_simplification :
  (7 - 3 * Complex.I) - 3 * (2 + 4 * Complex.I) + 4 * (1 - Complex.I) = 5 - 19 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l1991_199178


namespace NUMINAMATH_CALUDE_oak_elm_difference_pine_elm_difference_l1991_199107

-- Define the heights of the trees
def elm_height : ℚ := 49/4  -- 12¼ feet
def oak_height : ℚ := 37/2  -- 18½ feet
def pine_height_inches : ℚ := 225  -- 225 inches

-- Convert pine height to feet
def pine_height : ℚ := pine_height_inches / 12

-- Define the theorems to be proved
theorem oak_elm_difference : oak_height - elm_height = 25/4 := by sorry

theorem pine_elm_difference : pine_height - elm_height = 13/2 := by sorry

end NUMINAMATH_CALUDE_oak_elm_difference_pine_elm_difference_l1991_199107


namespace NUMINAMATH_CALUDE_ratio_problem_l1991_199169

theorem ratio_problem (x y : ℝ) (h : (3 * x^2 - y) / (x + y) = 1/2) :
  x / y = 3 / (6 * x - 1) := by sorry

end NUMINAMATH_CALUDE_ratio_problem_l1991_199169


namespace NUMINAMATH_CALUDE_dans_initial_money_l1991_199185

def candy_price : ℕ := 2
def chocolate_price : ℕ := 3

theorem dans_initial_money :
  ∀ (initial_money : ℕ),
  (initial_money = candy_price + chocolate_price) ∧
  (chocolate_price - candy_price = 1) →
  initial_money = 5 := by
sorry

end NUMINAMATH_CALUDE_dans_initial_money_l1991_199185


namespace NUMINAMATH_CALUDE_value_calculation_l1991_199120

theorem value_calculation (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = 1) :
  a + a^3 / b^2 + b^3 / a^2 + b = 2535 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l1991_199120


namespace NUMINAMATH_CALUDE_mirror_country_transfers_l1991_199103

-- Define the type for cities
def City : Type := ℕ

-- Define the type for countries
inductive Country
| Wonderland
| Mirrorland

-- Define a function to represent railroad connections
def connected (country : Country) (city1 city2 : City) : Prop := sorry

-- Define a function to represent the "double" of a city in the other country
def double (city : City) (country : Country) : City := sorry

-- Define the number of transfers needed for a journey
def transfers (country : Country) (start finish : City) : ℕ := sorry

-- State the theorem
theorem mirror_country_transfers 
  (A B : City) 
  (h1 : transfers Country.Wonderland A B ≥ 2) 
  (h2 : ∀ (c1 c2 : City), connected Country.Wonderland c1 c2 ↔ ¬connected Country.Mirrorland (double c1 Country.Mirrorland) (double c2 Country.Mirrorland))
  (h3 : ∀ (c : City), ∃ (d : City), d = double c Country.Mirrorland)
  : ∀ (X Y : City), transfers Country.Mirrorland X Y ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_mirror_country_transfers_l1991_199103


namespace NUMINAMATH_CALUDE_tangent_lines_with_equal_intercepts_l1991_199182

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define a tangent line
def is_tangent_line (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_C x y ∧ a*x + b*y + c = 0 ∧
  ∀ (x' y' : ℝ), circle_C x' y' → a*x' + b*y' + c ≥ 0

-- Define the condition for equal absolute intercepts
def equal_abs_intercepts (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ |c/a| = |c/b|

-- Theorem statement
theorem tangent_lines_with_equal_intercepts :
  ∀ (a b c : ℝ),
    is_tangent_line a b c ∧ equal_abs_intercepts a b c →
    ((a = 1 ∧ b = 1 ∧ c = -3) ∨
     (a = 1 ∧ b = 1 ∧ c = 1) ∨
     (a = 1 ∧ b = -1 ∧ c = -5) ∨
     (a = 1 ∧ b = -1 ∧ c = -1) ∨
     (∃ k : ℝ, k^2 = 10 ∧ a = k ∧ b = -1 ∧ c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_with_equal_intercepts_l1991_199182


namespace NUMINAMATH_CALUDE_approximate_root_l1991_199100

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem approximate_root (hcont : Continuous f) 
  (h1 : f 0.64 < 0) (h2 : f 0.72 > 0) (h3 : f 0.68 < 0) :
  ∃ (x : ℝ), f x = 0 ∧ |x - 0.7| ≤ 0.1 := by
  sorry

end NUMINAMATH_CALUDE_approximate_root_l1991_199100


namespace NUMINAMATH_CALUDE_triangle_area_l1991_199162

theorem triangle_area (a b c : ℝ) (A : ℝ) :
  b^2 - b*c - 2*c^2 = 0 →
  a = Real.sqrt 6 →
  Real.cos A = 7/8 →
  (1/2) * b * c * Real.sin A = Real.sqrt 15 / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1991_199162


namespace NUMINAMATH_CALUDE_complex_fraction_value_l1991_199175

theorem complex_fraction_value (a : ℝ) (z : ℂ) :
  z = (a^2 - 1 : ℂ) + (a + 1 : ℂ) * I →
  z.re = 0 →
  (a + I^2016) / (1 + I) = 1 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l1991_199175


namespace NUMINAMATH_CALUDE_lagoonIslandMales_l1991_199195

/-- Represents the population of alligators on Lagoon Island -/
structure AlligatorPopulation where
  total : ℕ
  males : ℕ
  females : ℕ
  juvenileFemales : ℕ
  adultFemales : ℕ

/-- The conditions of the alligator population on Lagoon Island -/
def lagoonIslandConditions (pop : AlligatorPopulation) : Prop :=
  pop.males = pop.females ∧
  pop.females = pop.juvenileFemales + pop.adultFemales ∧
  pop.juvenileFemales = (2 * pop.females) / 5 ∧
  pop.adultFemales = 15

theorem lagoonIslandMales (pop : AlligatorPopulation) 
  (h : lagoonIslandConditions pop) : pop.males = 25 := by
  sorry

end NUMINAMATH_CALUDE_lagoonIslandMales_l1991_199195


namespace NUMINAMATH_CALUDE_min_total_length_l1991_199198

/-- A set of arcs on a circle -/
structure ArcSet :=
  (n : ℕ)                    -- number of arcs
  (arcs : Fin n → ℝ)         -- length of each arc in degrees
  (total_length : ℝ)         -- total length of all arcs
  (rotation_overlap : ∀ θ : ℝ, ∃ i : Fin n, ∃ j : Fin n, (arcs i + θ) % 360 = arcs j)
                             -- for any rotation, there's an overlap

/-- The minimum total length of arcs in an ArcSet is 360/n -/
theorem min_total_length (F : ArcSet) : F.total_length ≥ 360 / F.n :=
sorry

end NUMINAMATH_CALUDE_min_total_length_l1991_199198


namespace NUMINAMATH_CALUDE_perpendicular_lines_intersection_l1991_199179

/-- Two lines are perpendicular if their slopes multiply to -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- A point (x, y) lies on a line ax + by + c = 0 -/
def point_on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

/-- Main theorem: Given two perpendicular lines and their intersection point, 
    prove that m + n - p = 0 -/
theorem perpendicular_lines_intersection 
  (m n p : ℝ) : 
  perpendicular (m / (-4)) (2 / 5) →
  point_on_line 1 p m 4 (-2) →
  point_on_line 1 p 2 (-5) n →
  m + n - p = 0 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_lines_intersection_l1991_199179


namespace NUMINAMATH_CALUDE_wallet_cost_l1991_199118

theorem wallet_cost (W : ℝ) : 
  W / 2 + 15 + 2 * 15 + 5 = W → W = 100 := by
  sorry

end NUMINAMATH_CALUDE_wallet_cost_l1991_199118


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l1991_199187

theorem quadratic_inequality_and_constraint (a b k : ℝ) : 
  (∀ x, (x < 1 ∨ x > b) ↔ a * x^2 - 3 * x + 2 > 0) →
  (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ k^2 + k + 2) →
  a = 1 ∧ b = 2 ∧ -3 ≤ k ∧ k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l1991_199187


namespace NUMINAMATH_CALUDE_probability_queens_or_jacks_value_l1991_199180

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_jacks : ℕ)
  (num_queens : ℕ)

/-- Calculates the probability of drawing either three queens or at least one jack -/
def probability_queens_or_jacks (d : Deck) : ℚ :=
  sorry

/-- Theorem stating the probability of drawing either three queens or at least one jack -/
theorem probability_queens_or_jacks_value :
  let standard_deck : Deck := ⟨52, 4, 4⟩
  probability_queens_or_jacks standard_deck = 142 / 1105 :=
sorry

end NUMINAMATH_CALUDE_probability_queens_or_jacks_value_l1991_199180


namespace NUMINAMATH_CALUDE_largest_logarithm_l1991_199123

theorem largest_logarithm (h : 0 < Real.log 2 ∧ Real.log 2 < 1) :
  2 * Real.log 2 > Real.log 2 ∧ 
  Real.log 2 > (Real.log 2)^2 ∧ 
  (Real.log 2)^2 > Real.log (Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_logarithm_l1991_199123


namespace NUMINAMATH_CALUDE_triangle_side_range_l1991_199176

theorem triangle_side_range (x : ℝ) : 
  (x > 0) →  -- Ensure positive side lengths
  (x + (x + 1) > (x + 2)) →  -- Triangle inequality
  (x + (x + 1) + (x + 2) ≤ 12) →  -- Perimeter condition
  (1 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1991_199176


namespace NUMINAMATH_CALUDE_abigail_cans_collected_l1991_199142

/-- Given:
  - The total number of cans needed is 100
  - Alyssa has collected 30 cans
  - They still need to collect 27 more cans
  Prove that Abigail has collected 43 cans -/
theorem abigail_cans_collected 
  (total_cans : ℕ) 
  (alyssa_cans : ℕ) 
  (more_cans_needed : ℕ) 
  (h1 : total_cans = 100)
  (h2 : alyssa_cans = 30)
  (h3 : more_cans_needed = 27) :
  total_cans - (alyssa_cans + more_cans_needed) = 43 := by
  sorry

end NUMINAMATH_CALUDE_abigail_cans_collected_l1991_199142


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l1991_199141

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 12) :
  (∀ k l : ℕ+, Nat.gcd k l = 12 → Nat.gcd (10 * k) (15 * l) ≥ 60) ∧
  (∃ a b : ℕ+, Nat.gcd a b = 12 ∧ Nat.gcd (10 * a) (15 * b) = 60) := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l1991_199141


namespace NUMINAMATH_CALUDE_coffee_mixture_cost_theorem_l1991_199155

/-- The cost of the more expensive coffee per pound -/
def expensive_coffee_cost : ℝ := 7.28

/-- The cost of the cheaper coffee per pound -/
def cheaper_coffee_cost : ℝ := 6.42

/-- The amount of cheaper coffee in pounds -/
def cheaper_coffee_amount : ℝ := 7

/-- The amount of expensive coffee in pounds -/
def expensive_coffee_amount : ℝ := 68.25

/-- The price of the mixture per pound -/
def mixture_price : ℝ := 7.20

/-- The total amount of coffee in the mixture -/
def total_coffee_amount : ℝ := cheaper_coffee_amount + expensive_coffee_amount

theorem coffee_mixture_cost_theorem :
  cheaper_coffee_amount * cheaper_coffee_cost +
  expensive_coffee_amount * expensive_coffee_cost =
  total_coffee_amount * mixture_price :=
by sorry

end NUMINAMATH_CALUDE_coffee_mixture_cost_theorem_l1991_199155


namespace NUMINAMATH_CALUDE_two_digit_number_representation_l1991_199131

/-- Represents a two-digit number with specific properties -/
structure TwoDigitNumber where
  units : ℕ
  tens : ℕ
  is_valid : units < 10 ∧ tens < 10
  tens_property : tens = 2 * (units ^ 2)

/-- Theorem stating the representation of the two-digit number -/
theorem two_digit_number_representation (n : TwoDigitNumber) :
  n.tens * 10 + n.units = 20 * (n.units ^ 2) + n.units := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_representation_l1991_199131


namespace NUMINAMATH_CALUDE_symmetry_of_lines_l1991_199126

/-- Given two lines in a 2D plane represented by their equations,
    this function returns true if they are symmetric with respect to the line x+y=0 -/
def are_symmetric_lines (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, l1 x y ↔ l2 (-y) (-x)

/-- The equation of the original line -/
def original_line (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0

/-- The equation of the supposedly symmetric line -/
def symmetric_line (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line
    with respect to the line x+y=0 -/
theorem symmetry_of_lines : are_symmetric_lines original_line symmetric_line :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_lines_l1991_199126


namespace NUMINAMATH_CALUDE_equal_perimeters_rectangle_square_l1991_199170

/-- Given two equal lengths of wire, one formed into a rectangle and one formed into a square,
    the perimeters of the resulting shapes are equal. -/
theorem equal_perimeters_rectangle_square (wire_length : ℝ) (h : wire_length > 0) :
  ∃ (rect_width rect_height square_side : ℝ),
    rect_width > 0 ∧ rect_height > 0 ∧ square_side > 0 ∧
    2 * (rect_width + rect_height) = wire_length ∧
    4 * square_side = wire_length ∧
    2 * (rect_width + rect_height) = 4 * square_side :=
by sorry

end NUMINAMATH_CALUDE_equal_perimeters_rectangle_square_l1991_199170


namespace NUMINAMATH_CALUDE_value_of_b_l1991_199112

theorem value_of_b (a b c : ℝ) 
  (h1 : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
  (h2 : 6 * b * 2 = 4) : 
  b = 15 := by
sorry

end NUMINAMATH_CALUDE_value_of_b_l1991_199112


namespace NUMINAMATH_CALUDE_total_matchsticks_l1991_199121

def boxes : ℕ := 4
def matchboxes_per_box : ℕ := 20
def sticks_per_matchbox : ℕ := 300

theorem total_matchsticks :
  boxes * matchboxes_per_box * sticks_per_matchbox = 24000 :=
by sorry

end NUMINAMATH_CALUDE_total_matchsticks_l1991_199121


namespace NUMINAMATH_CALUDE_spa_nail_polish_problem_l1991_199184

/-- The number of girls who went to the spa -/
def num_girls : ℕ := 8

/-- The number of fingers on each limb -/
def fingers_per_limb : ℕ := 5

/-- The total number of fingers polished -/
def total_fingers_polished : ℕ := 40

/-- The number of limbs polished per girl -/
def limbs_per_girl : ℕ := total_fingers_polished / (num_girls * fingers_per_limb)

theorem spa_nail_polish_problem :
  limbs_per_girl = 1 :=
by sorry

end NUMINAMATH_CALUDE_spa_nail_polish_problem_l1991_199184
