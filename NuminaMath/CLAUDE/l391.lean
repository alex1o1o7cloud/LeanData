import Mathlib

namespace NUMINAMATH_CALUDE_square_measurement_unit_l391_39171

/-- Given a square with sides of length 5 units and an actual area of at least 20.25 square centimeters,
    prove that the length of one unit in this measurement system is 0.9 centimeters. -/
theorem square_measurement_unit (side_length : ℝ) (actual_area : ℝ) :
  side_length = 5 →
  actual_area ≥ 20.25 →
  actual_area = (side_length * 0.9) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_square_measurement_unit_l391_39171


namespace NUMINAMATH_CALUDE_slide_total_boys_l391_39108

theorem slide_total_boys (initial : ℕ) (second : ℕ) (third : ℕ) 
  (h1 : initial = 87) 
  (h2 : second = 46) 
  (h3 : third = 29) : 
  initial + second + third = 162 := by
  sorry

end NUMINAMATH_CALUDE_slide_total_boys_l391_39108


namespace NUMINAMATH_CALUDE_opposite_sign_sum_l391_39141

theorem opposite_sign_sum (x y : ℝ) : (x + 3)^2 + |y - 2| = 0 → (x + y)^y = 1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_sum_l391_39141


namespace NUMINAMATH_CALUDE_stone_game_ratio_l391_39101

/-- The stone game on a blackboard -/
def StoneGame (n : ℕ) : Prop :=
  n ≥ 3 →
  ∀ (s t : ℕ), s > 0 ∧ t > 0 →
  ∃ (q : ℚ), q ≥ 1 ∧ q < n - 1 ∧ (t : ℚ) / s = q

theorem stone_game_ratio (n : ℕ) (h : n ≥ 3) :
  StoneGame n :=
sorry

end NUMINAMATH_CALUDE_stone_game_ratio_l391_39101


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l391_39125

theorem sqrt_difference_inequality (n : ℝ) (hn : n ≥ 0) :
  Real.sqrt (n + 2) - Real.sqrt (n + 1) < Real.sqrt (n + 1) - Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l391_39125


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l391_39136

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l391_39136


namespace NUMINAMATH_CALUDE_angle_BEC_measure_l391_39193

-- Define the geometric configuration
structure GeometricConfig where
  A : Real
  D : Real
  F : Real
  BEC_exists : Bool
  E_above_C : Bool

-- Define the theorem
theorem angle_BEC_measure (config : GeometricConfig) 
  (h1 : config.A = 45)
  (h2 : config.D = 50)
  (h3 : config.F = 55)
  (h4 : config.BEC_exists = true)
  (h5 : config.E_above_C = true) :
  ∃ (BEC : Real), BEC = 10 := by
  sorry

end NUMINAMATH_CALUDE_angle_BEC_measure_l391_39193


namespace NUMINAMATH_CALUDE_number_satisfies_equation_l391_39130

theorem number_satisfies_equation : ∃ x : ℝ, (0.8 * 90 : ℝ) = 0.7 * x + 30 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfies_equation_l391_39130


namespace NUMINAMATH_CALUDE_elvis_editing_time_l391_39187

theorem elvis_editing_time (num_songs : ℕ) (studio_hours : ℕ) (record_time : ℕ) (write_time : ℕ)
  (h1 : num_songs = 10)
  (h2 : studio_hours = 5)
  (h3 : record_time = 12)
  (h4 : write_time = 15) :
  (studio_hours * 60) - (num_songs * write_time + num_songs * record_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_elvis_editing_time_l391_39187


namespace NUMINAMATH_CALUDE_distance_from_displacements_l391_39168

/-- The distance between two points given their net displacements -/
theorem distance_from_displacements (south west : ℝ) :
  south = 20 →
  west = 50 →
  Real.sqrt (south^2 + west^2) = 50 * Real.sqrt 2.9 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_displacements_l391_39168


namespace NUMINAMATH_CALUDE_travel_agency_problem_l391_39154

/-- Represents the travel agency problem --/
theorem travel_agency_problem 
  (seats_per_bus : ℕ) 
  (incomplete_bus_2005 : ℕ) 
  (increase_2006 : ℕ) 
  (h1 : seats_per_bus = 27)
  (h2 : incomplete_bus_2005 = 19)
  (h3 : increase_2006 = 53) :
  ∃ (k : ℕ),
    (seats_per_bus * k + incomplete_bus_2005 + increase_2006) / seats_per_bus - 
    (seats_per_bus * k + incomplete_bus_2005) / seats_per_bus = 2 ∧
    (seats_per_bus * k + incomplete_bus_2005 + increase_2006) % seats_per_bus = 9 :=
by sorry

end NUMINAMATH_CALUDE_travel_agency_problem_l391_39154


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l391_39117

theorem gcd_lcm_product (a b : ℕ) (ha : a = 180) (hb : b = 250) :
  (Nat.gcd a b) * (Nat.lcm a b) = 45000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l391_39117


namespace NUMINAMATH_CALUDE_train_length_is_200_l391_39103

/-- The length of a train that crosses a 200-meter bridge in 10 seconds
    and passes a lamp post on the bridge in 5 seconds. -/
def train_length : ℝ := 200

/-- The length of the bridge in meters. -/
def bridge_length : ℝ := 200

/-- The time taken to cross the bridge in seconds. -/
def bridge_crossing_time : ℝ := 10

/-- The time taken to pass the lamp post in seconds. -/
def lamppost_passing_time : ℝ := 5

/-- Theorem stating that the train length is 200 meters given the conditions. -/
theorem train_length_is_200 :
  train_length = 200 :=
by sorry

end NUMINAMATH_CALUDE_train_length_is_200_l391_39103


namespace NUMINAMATH_CALUDE_bisecting_line_sum_l391_39123

/-- Triangle PQR with vertices P(0, 10), Q(3, 0), and R(10, 0) -/
structure Triangle :=
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)

/-- A line represented by its slope and y-intercept -/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- The line that bisects the area of the triangle -/
def bisecting_line (t : Triangle) : Line :=
  sorry

/-- The theorem to be proved -/
theorem bisecting_line_sum (t : Triangle) :
  let pqr := Triangle.mk (0, 10) (3, 0) (10, 0)
  let l := bisecting_line pqr
  l.slope + l.intercept = -5 :=
sorry

end NUMINAMATH_CALUDE_bisecting_line_sum_l391_39123


namespace NUMINAMATH_CALUDE_number_is_two_l391_39105

theorem number_is_two (x y : ℝ) (n : ℝ) 
  (h1 : n * (x - y) = 4)
  (h2 : 6 * x - 3 * y = 12) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_is_two_l391_39105


namespace NUMINAMATH_CALUDE_power_multiplication_l391_39145

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l391_39145


namespace NUMINAMATH_CALUDE_estimate_products_and_quotients_l391_39157

theorem estimate_products_and_quotients 
  (ε₁ ε₂ ε₃ ε₄ : ℝ) 
  (h₁ : ε₁ > 0) 
  (h₂ : ε₂ > 0) 
  (h₃ : ε₃ > 0) 
  (h₄ : ε₄ > 0) : 
  (|99 * 71 - 7000| ≤ ε₁) ∧ 
  (|25 * 39 - 1000| ≤ ε₂) ∧ 
  (|124 / 3 - 40| ≤ ε₃) ∧ 
  (|398 / 5 - 80| ≤ ε₄) := by
  sorry

end NUMINAMATH_CALUDE_estimate_products_and_quotients_l391_39157


namespace NUMINAMATH_CALUDE_complex_subtraction_l391_39162

theorem complex_subtraction (z₁ z₂ : ℂ) (h₁ : z₁ = 3 + I) (h₂ : z₂ = 2 - I) :
  z₁ - z₂ = 1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l391_39162


namespace NUMINAMATH_CALUDE_triple_a_award_distribution_l391_39134

theorem triple_a_award_distribution (n : Nat) (k : Nat) (h1 : n = 10) (h2 : k = 7) :
  (Nat.choose (n - k + k - 1) (n - k)) = 84 := by
  sorry

end NUMINAMATH_CALUDE_triple_a_award_distribution_l391_39134


namespace NUMINAMATH_CALUDE_buyOneGetOneFreeIsCheaper_finalCostIs216_l391_39159

/-- Represents the total cost of Pauline's purchase with a given discount and sales tax. -/
def totalCost (totalBeforeTax : ℝ) (selectedItemsTotal : ℝ) (discount : ℝ) (salesTaxRate : ℝ) : ℝ :=
  let discountedTotal := totalBeforeTax - selectedItemsTotal * discount
  discountedTotal * (1 + salesTaxRate)

/-- Theorem stating that the Buy One, Get One Free offer is cheaper than the 15% discount offer. -/
theorem buyOneGetOneFreeIsCheaper :
  let totalBeforeTax : ℝ := 250
  let selectedItemsTotal : ℝ := 100
  let remainingItemsTotal : ℝ := totalBeforeTax - selectedItemsTotal
  let discountRate : ℝ := 0.15
  let buyOneGetOneFreeDiscount : ℝ := 0.5
  let salesTaxRate : ℝ := 0.08
  totalCost totalBeforeTax selectedItemsTotal buyOneGetOneFreeDiscount salesTaxRate <
  totalCost totalBeforeTax selectedItemsTotal discountRate salesTaxRate :=
by sorry

/-- Calculates the final cost with the Buy One, Get One Free offer. -/
def finalCost : ℝ :=
  let totalBeforeTax : ℝ := 250
  let selectedItemsTotal : ℝ := 100
  let buyOneGetOneFreeDiscount : ℝ := 0.5
  let salesTaxRate : ℝ := 0.08
  totalCost totalBeforeTax selectedItemsTotal buyOneGetOneFreeDiscount salesTaxRate

/-- Theorem stating that the final cost is $216. -/
theorem finalCostIs216 : finalCost = 216 :=
by sorry

end NUMINAMATH_CALUDE_buyOneGetOneFreeIsCheaper_finalCostIs216_l391_39159


namespace NUMINAMATH_CALUDE_max_value_2x_plus_y_max_value_2x_plus_y_achievable_l391_39195

theorem max_value_2x_plus_y (x y : ℝ) : 
  2 * x - y ≤ 0 → x + y ≤ 3 → x ≥ 0 → 2 * x + y ≤ 4 := by
  sorry

theorem max_value_2x_plus_y_achievable : 
  ∃ x y : ℝ, 2 * x - y ≤ 0 ∧ x + y ≤ 3 ∧ x ≥ 0 ∧ 2 * x + y = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_2x_plus_y_max_value_2x_plus_y_achievable_l391_39195


namespace NUMINAMATH_CALUDE_solve_for_s_l391_39127

theorem solve_for_s (s t : ℚ) 
  (eq1 : 8 * s + 6 * t = 120)
  (eq2 : t - 3 = s) : 
  s = 51 / 7 := by
sorry

end NUMINAMATH_CALUDE_solve_for_s_l391_39127


namespace NUMINAMATH_CALUDE_top_grade_probability_l391_39173

-- Define the probabilities
def prob_B : ℝ := 0.03
def prob_C : ℝ := 0.01

-- Define the theorem
theorem top_grade_probability :
  ∀ (prob_A : ℝ),
  (prob_A + prob_B + prob_C = 1) →
  (prob_A ≥ 0 ∧ prob_A ≤ 1) →
  (prob_A = 0.96) := by
sorry


end NUMINAMATH_CALUDE_top_grade_probability_l391_39173


namespace NUMINAMATH_CALUDE_min_area_rectangle_l391_39184

/-- A rectangle with even integer dimensions and perimeter 120 has a minimum area of 116 -/
theorem min_area_rectangle (l w : ℕ) : 
  Even l → Even w → 
  2 * (l + w) = 120 → 
  ∀ a : ℕ, (Even a.sqrt ∧ Even (60 - a.sqrt) ∧ a = a.sqrt * (60 - a.sqrt)) → 
  116 ≤ a := by
sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l391_39184


namespace NUMINAMATH_CALUDE_relay_race_distance_per_member_l391_39111

theorem relay_race_distance_per_member 
  (total_distance : ℕ) (team_members : ℕ) 
  (h1 : total_distance = 150) 
  (h2 : team_members = 5) : 
  total_distance / team_members = 30 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_distance_per_member_l391_39111


namespace NUMINAMATH_CALUDE_cylinder_water_transfer_l391_39113

theorem cylinder_water_transfer (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let original_volume := π * r^2 * h
  let new_volume := π * (1.25 * r)^2 * (0.72 * h)
  (3/5) * new_volume = 0.675 * original_volume :=
by sorry

end NUMINAMATH_CALUDE_cylinder_water_transfer_l391_39113


namespace NUMINAMATH_CALUDE_trapezoid_with_equal_angles_l391_39164

-- Define a trapezoid
structure Trapezoid :=
  (is_quadrilateral : Bool)
  (has_parallel_sides : Bool)
  (has_nonparallel_sides : Bool)

-- Define properties of a trapezoid
def Trapezoid.is_isosceles (t : Trapezoid) : Prop := sorry
def Trapezoid.is_right_angled (t : Trapezoid) : Prop := sorry
def Trapezoid.has_two_equal_angles (t : Trapezoid) : Prop := sorry

-- Theorem statement
theorem trapezoid_with_equal_angles 
  (t : Trapezoid) 
  (h1 : t.is_quadrilateral = true) 
  (h2 : t.has_parallel_sides = true) 
  (h3 : t.has_nonparallel_sides = true) 
  (h4 : t.has_two_equal_angles) : 
  t.is_isosceles ∨ t.is_right_angled := sorry

end NUMINAMATH_CALUDE_trapezoid_with_equal_angles_l391_39164


namespace NUMINAMATH_CALUDE_A_intersect_B_l391_39177

def A : Set ℝ := {-1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 2}

theorem A_intersect_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l391_39177


namespace NUMINAMATH_CALUDE_min_value_expression_l391_39167

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b^2 + 2) / (a + b) + a^2 / (a * b + 1) ≥ 2 := by sorry

end NUMINAMATH_CALUDE_min_value_expression_l391_39167


namespace NUMINAMATH_CALUDE_betty_stones_count_l391_39132

/-- The number of stones in each bracelet -/
def stones_per_bracelet : ℕ := 14

/-- The number of bracelets Betty can make -/
def number_of_bracelets : ℕ := 10

/-- The total number of stones Betty bought -/
def total_stones : ℕ := stones_per_bracelet * number_of_bracelets

theorem betty_stones_count : total_stones = 140 := by
  sorry

end NUMINAMATH_CALUDE_betty_stones_count_l391_39132


namespace NUMINAMATH_CALUDE_boys_without_calculators_l391_39137

theorem boys_without_calculators (total_students : Nat) (boys : Nat) (students_with_calculators : Nat) (girls_with_calculators : Nat)
  (h1 : total_students = 30)
  (h2 : boys = 20)
  (h3 : students_with_calculators = 25)
  (h4 : girls_with_calculators = 18)
  : total_students - boys - (students_with_calculators - girls_with_calculators) = 13 := by
  sorry

end NUMINAMATH_CALUDE_boys_without_calculators_l391_39137


namespace NUMINAMATH_CALUDE_complex_equation_sum_l391_39148

theorem complex_equation_sum (a t : ℝ) (i : ℂ) : 
  i * i = -1 → a + i = (1 + 2*i) * t*i → t + a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l391_39148


namespace NUMINAMATH_CALUDE_vowels_on_board_l391_39119

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 3

/-- The total number of alphabets written on the board -/
def total_written : ℕ := num_vowels * times_written

theorem vowels_on_board : total_written = 15 := by
  sorry

end NUMINAMATH_CALUDE_vowels_on_board_l391_39119


namespace NUMINAMATH_CALUDE_roll_12_with_8_dice_l391_39158

/-- The number of ways to roll a sum of 12 with 8 fair 6-sided dice -/
def waysToRoll12With8Dice : ℕ := sorry

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice -/
def numDice : ℕ := 8

/-- The target sum -/
def targetSum : ℕ := 12

theorem roll_12_with_8_dice :
  waysToRoll12With8Dice = 330 := by sorry

end NUMINAMATH_CALUDE_roll_12_with_8_dice_l391_39158


namespace NUMINAMATH_CALUDE_solution_set_quadratic_equation_l391_39110

theorem solution_set_quadratic_equation :
  {x : ℝ | x^2 - 3*x + 2 = 0} = {1, 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_equation_l391_39110


namespace NUMINAMATH_CALUDE_third_term_is_16_l391_39189

/-- Geometric sequence with common ratio 2 and sum of first 4 terms equal to 60 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ 
  (a 1 + a 2 + a 3 + a 4 = 60)

/-- The third term of the geometric sequence is 16 -/
theorem third_term_is_16 (a : ℕ → ℝ) (h : geometric_sequence a) : a 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_16_l391_39189


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_solve_linear_equation_l391_39147

-- Equation 1
theorem solve_quadratic_equation (x : ℝ) :
  3 * x^2 + 6 * x - 4 = 0 ↔ x = (-3 + Real.sqrt 21) / 3 ∨ x = (-3 - Real.sqrt 21) / 3 := by
  sorry

-- Equation 2
theorem solve_linear_equation (x : ℝ) :
  3 * x * (2 * x + 1) = 4 * x + 2 ↔ x = -1/2 ∨ x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_solve_linear_equation_l391_39147


namespace NUMINAMATH_CALUDE_divisibility_rule_37_l391_39156

-- Define a function to compute the sum of three-digit groups
def sumOfGroups (n : ℕ) : ℕ := sorry

-- State the theorem
theorem divisibility_rule_37 (n : ℕ) :
  37 ∣ n ↔ 37 ∣ sumOfGroups n := by sorry

end NUMINAMATH_CALUDE_divisibility_rule_37_l391_39156


namespace NUMINAMATH_CALUDE_bonus_allocation_l391_39106

theorem bonus_allocation (bonus : ℚ) (kitchen_fraction : ℚ) (christmas_fraction : ℚ) (leftover : ℚ) 
  (h1 : bonus = 1496)
  (h2 : kitchen_fraction = 1 / 22)
  (h3 : christmas_fraction = 1 / 8)
  (h4 : leftover = 867)
  (h5 : bonus * kitchen_fraction + bonus * christmas_fraction + bonus * (holiday_fraction : ℚ) + leftover = bonus) :
  holiday_fraction = 187 / 748 := by
  sorry

end NUMINAMATH_CALUDE_bonus_allocation_l391_39106


namespace NUMINAMATH_CALUDE_waiter_customers_l391_39140

/-- Calculates the number of customers a waiter has after some tables leave --/
def customers_remaining (initial_tables : Float) (tables_left : Float) (customers_per_table : Float) : Float :=
  (initial_tables - tables_left) * customers_per_table

/-- Theorem: Given the initial conditions, the waiter has 256.0 customers --/
theorem waiter_customers :
  customers_remaining 44.0 12.0 8.0 = 256.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l391_39140


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l391_39129

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l391_39129


namespace NUMINAMATH_CALUDE_log_problem_l391_39109

theorem log_problem (x k : ℝ) : 
  (Real.log 3 / Real.log 9 = x) → 
  (Real.log 81 / Real.log 3 = k * x) → 
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_log_problem_l391_39109


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_l391_39192

theorem same_terminal_side_angle : ∃ (θ : Real), 
  0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
  ∃ (k : ℤ), θ = 2 * k * Real.pi + (-4 * Real.pi / 3) ∧
  θ = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_l391_39192


namespace NUMINAMATH_CALUDE_equation_equality_l391_39181

theorem equation_equality (a b : ℝ) : (a - b)^3 * (b - a)^4 = (a - b)^7 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l391_39181


namespace NUMINAMATH_CALUDE_final_digit_is_two_l391_39163

/-- Represents the state of the board with counts of 0s, 1s, and 2s -/
structure BoardState where
  zeros : ℕ
  ones : ℕ
  twos : ℕ

/-- Represents a valid operation on the board -/
inductive Operation
  | erase_zero_one_add_two
  | erase_one_two_add_zero
  | erase_zero_two_add_one

/-- Applies an operation to a board state -/
def apply_operation (state : BoardState) (op : Operation) : BoardState :=
  match op with
  | Operation.erase_zero_one_add_two => 
      ⟨state.zeros - 1, state.ones - 1, state.twos + 1⟩
  | Operation.erase_one_two_add_zero => 
      ⟨state.zeros + 1, state.ones - 1, state.twos - 1⟩
  | Operation.erase_zero_two_add_one => 
      ⟨state.zeros - 1, state.ones + 1, state.twos - 1⟩

/-- Checks if the board state has only one digit remaining -/
def has_one_digit (state : BoardState) : Prop :=
  (state.zeros = 1 ∧ state.ones = 0 ∧ state.twos = 0) ∨
  (state.zeros = 0 ∧ state.ones = 1 ∧ state.twos = 0) ∨
  (state.zeros = 0 ∧ state.ones = 0 ∧ state.twos = 1)

/-- The main theorem to prove -/
theorem final_digit_is_two 
  (initial : BoardState) 
  (operations : List Operation) 
  (h_final : has_one_digit (operations.foldl apply_operation initial)) :
  (operations.foldl apply_operation initial).twos = 1 :=
sorry

end NUMINAMATH_CALUDE_final_digit_is_two_l391_39163


namespace NUMINAMATH_CALUDE_infinite_even_k_composite_sum_l391_39128

theorem infinite_even_k_composite_sum (t : ℕ+) (p : ℕ) :
  let k := 30 * t + 26
  (∃ n : ℕ+, k = 2 * n) ∧ 
  (Nat.Prime p → ∃ (m n : ℕ+), p^2 + k = m * n ∧ m ≠ 1 ∧ n ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_infinite_even_k_composite_sum_l391_39128


namespace NUMINAMATH_CALUDE_dietitian_excess_calories_l391_39165

/-- Calculates the excess calories consumed given the total lunch calories and the fraction eaten -/
def excess_calories (total_calories : ℕ) (fraction_eaten : ℚ) (recommended_calories : ℕ) : ℤ :=
  ⌊(fraction_eaten * total_calories : ℚ)⌋ - recommended_calories

/-- Proves that eating 3/4 of a 40-calorie lunch exceeds the recommended 25 calories by 5 -/
theorem dietitian_excess_calories :
  excess_calories 40 (3/4 : ℚ) 25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dietitian_excess_calories_l391_39165


namespace NUMINAMATH_CALUDE_crups_are_arogs_and_brafs_l391_39100

-- Define the types for our sets
variable (U : Type) -- Universe set
variable (Arog Braf Crup Dramp : Set U)

-- Define the given conditions
variable (h1 : Arog ⊆ Braf)
variable (h2 : Crup ⊆ Braf)
variable (h3 : Arog ⊆ Dramp)
variable (h4 : Crup ⊆ Dramp)

-- Theorem to prove
theorem crups_are_arogs_and_brafs : Crup ⊆ Arog ∩ Braf :=
sorry

end NUMINAMATH_CALUDE_crups_are_arogs_and_brafs_l391_39100


namespace NUMINAMATH_CALUDE_existence_of_always_different_teams_l391_39176

/-- Represents a team assignment for a single game -/
def GameAssignment := Fin 22 → Bool

/-- Represents the team assignments for all three games -/
def ThreeGamesAssignment := Fin 3 → GameAssignment

theorem existence_of_always_different_teams (games : ThreeGamesAssignment) : 
  ∃ (p1 p2 : Fin 22), p1 ≠ p2 ∧ 
    (∀ (g : Fin 3), games g p1 ≠ games g p2) :=
sorry

end NUMINAMATH_CALUDE_existence_of_always_different_teams_l391_39176


namespace NUMINAMATH_CALUDE_mysterious_number_properties_l391_39112

/-- A positive integer that can be expressed as the difference of the squares of two consecutive even numbers. -/
def MysteriousNumber (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (2*k + 2)^2 - (2*k)^2 ∧ k ≥ 0

theorem mysterious_number_properties :
  (MysteriousNumber 28 ∧ MysteriousNumber 2020) ∧
  (∀ k : ℕ, (2*k + 2)^2 - (2*k)^2 % 4 = 0) ∧
  (∀ k : ℕ, ¬MysteriousNumber ((2*k + 1)^2 - (2*k - 1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_mysterious_number_properties_l391_39112


namespace NUMINAMATH_CALUDE_factor_expression_l391_39107

theorem factor_expression (x : ℝ) : 72 * x^3 - 250 * x^7 = 2 * x^3 * (36 - 125 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l391_39107


namespace NUMINAMATH_CALUDE_red_balls_count_l391_39152

theorem red_balls_count (total : ℕ) (white green yellow purple : ℕ) (prob : ℚ) :
  total = 60 ∧
  white = 22 ∧
  green = 18 ∧
  yellow = 5 ∧
  purple = 9 ∧
  prob = 3/4 ∧
  (white + green + yellow : ℚ) / total = prob →
  total - (white + green + yellow + purple) = 6 :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l391_39152


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l391_39138

theorem quadratic_coefficient (a : ℝ) : 
  (a * (1/2)^2 + 9 * (1/2) - 5 = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l391_39138


namespace NUMINAMATH_CALUDE_f_definition_f_2019_l391_39182

def a_n (n : ℕ) : ℕ := Nat.sqrt n

def b_n (n : ℕ) : ℕ := n - (a_n n)^2

def f (n : ℕ) : ℕ :=
  if b_n n ≤ a_n n then
    (a_n n)^2 + 1
  else if a_n n < b_n n ∧ b_n n ≤ 2 * (a_n n) + 1 then
    (a_n n)^2 + a_n n + 1
  else
    0  -- This case should never occur based on the problem definition

theorem f_definition (n : ℕ) :
  f n = if b_n n ≤ a_n n then
          (a_n n)^2 + 1
        else
          (a_n n)^2 + a_n n + 1 :=
by sorry

theorem f_2019 : f 2019 = 1981 :=
by sorry

end NUMINAMATH_CALUDE_f_definition_f_2019_l391_39182


namespace NUMINAMATH_CALUDE_cuboids_on_diagonal_of_90_cube_l391_39122

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with integer side length -/
structure Cube where
  side : ℕ

/-- Calculates the number of cuboids a diagonal of a cube passes through -/
def cuboids_on_diagonal (cube : Cube) (cuboid : Cuboid) : ℕ :=
  let n1 := cube.side / cuboid.height - 1
  let n2 := cube.side / cuboid.width - 1
  let n3 := cube.side / cuboid.length - 1
  let i12 := cube.side / (cuboid.height * cuboid.width) - 1
  let i23 := cube.side / (cuboid.width * cuboid.length) - 1
  let i13 := cube.side / (cuboid.height * cuboid.length) - 1
  let i123 := cube.side / (cuboid.height * cuboid.width * cuboid.length) - 1
  n1 + n2 + n3 - (i12 + i23 + i13) + i123

/-- The main theorem to be proved -/
theorem cuboids_on_diagonal_of_90_cube (c : Cube) (b : Cuboid) :
  c.side = 90 ∧ b.length = 2 ∧ b.width = 3 ∧ b.height = 5 →
  cuboids_on_diagonal c b = 65 := by
  sorry

end NUMINAMATH_CALUDE_cuboids_on_diagonal_of_90_cube_l391_39122


namespace NUMINAMATH_CALUDE_first_question_percentage_l391_39188

/-- Given a class of students taking a test with two questions, this theorem proves
    the percentage of students who answered the first question correctly. -/
theorem first_question_percentage
  (second_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : second_correct = 65)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 60) :
  ∃ (first_correct : ℝ),
    first_correct = 75 ∧
    first_correct + second_correct - both_correct = 100 - neither_correct :=
by
  sorry


end NUMINAMATH_CALUDE_first_question_percentage_l391_39188


namespace NUMINAMATH_CALUDE_sugar_price_increase_vs_inflation_sugar_price_increase_specific_l391_39166

/-- The percentage by which the rate of increase of sugar price exceeds inflation --/
theorem sugar_price_increase_vs_inflation (initial_price final_price : ℝ) 
  (inflation_rate : ℝ) (years : ℕ) : ℝ :=
  let total_sugar_increase := (final_price - initial_price) / initial_price * 100
  let total_inflation := ((1 + inflation_rate / 100) ^ years - 1) * 100
  total_sugar_increase - total_inflation

/-- Given specific values, prove that the difference is approximately 6.81% --/
theorem sugar_price_increase_specific :
  let initial_price : ℝ := 25
  let final_price : ℝ := 33.0625
  let inflation_rate : ℝ := 12
  let years : ℕ := 2
  abs (sugar_price_increase_vs_inflation initial_price final_price inflation_rate years - 6.81) < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_sugar_price_increase_vs_inflation_sugar_price_increase_specific_l391_39166


namespace NUMINAMATH_CALUDE_differential_equation_solution_l391_39186

/-- The differential equation dy/dx = y^2 has a general solution y = a₀ / (1 - a₀x) -/
theorem differential_equation_solution (x : ℝ) (a₀ : ℝ) :
  let y : ℝ → ℝ := λ x => a₀ / (1 - a₀ * x)
  ∀ x, (deriv y) x = (y x)^2 :=
by sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l391_39186


namespace NUMINAMATH_CALUDE_cloth_sale_calculation_l391_39135

/-- Proves the number of meters of cloth sold given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_calculation (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 4500)
    (h2 : profit_per_meter = 12)
    (h3 : cost_price_per_meter = 88) :
    total_selling_price / (cost_price_per_meter + profit_per_meter) = 45 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_calculation_l391_39135


namespace NUMINAMATH_CALUDE_inequality_proof_l391_39114

theorem inequality_proof : (1/2: ℝ)^(2/3) < (1/2: ℝ)^(1/3) ∧ (1/2: ℝ)^(1/3) < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l391_39114


namespace NUMINAMATH_CALUDE_equation_identity_l391_39191

theorem equation_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end NUMINAMATH_CALUDE_equation_identity_l391_39191


namespace NUMINAMATH_CALUDE_binary_10101_is_21_l391_39194

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + (if bit then 2^i else 0)) 0

theorem binary_10101_is_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_is_21_l391_39194


namespace NUMINAMATH_CALUDE_correct_fruit_baskets_l391_39131

/-- The number of ways to choose from n identical items -/
def chooseFrom (n : ℕ) : ℕ := n + 1

/-- The number of possible fruit baskets given the number of apples and oranges -/
def fruitBaskets (apples oranges : ℕ) : ℕ :=
  chooseFrom apples * chooseFrom oranges - 1

theorem correct_fruit_baskets :
  fruitBaskets 6 8 = 62 := by
  sorry

end NUMINAMATH_CALUDE_correct_fruit_baskets_l391_39131


namespace NUMINAMATH_CALUDE_tangent_triangle_area_l391_39144

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Define the tangent line at (1, -1)
def tangent_line (x : ℝ) : ℝ := -3*x + 2

-- Theorem statement
theorem tangent_triangle_area : 
  let x_intercept : ℝ := 2/3
  let y_intercept : ℝ := tangent_line 0
  let area : ℝ := (1/2) * x_intercept * y_intercept
  (f 1 = -1) ∧ (f' 1 = -3) → area = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_tangent_triangle_area_l391_39144


namespace NUMINAMATH_CALUDE_set_operations_l391_39142

-- Define the universal set U
def U : Set ℤ := {x : ℤ | -2 < x ∧ x < 2}

-- Define set A
def A : Set ℤ := {x : ℤ | x^2 - 5*x - 6 = 0}

-- Define set B
def B : Set ℤ := {x : ℤ | x^2 = 1}

-- Theorem statement
theorem set_operations :
  (A ∪ B = {-1, 1, 6}) ∧
  (A ∩ B = {-1}) ∧
  (U \ (A ∩ B) = {0, 1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l391_39142


namespace NUMINAMATH_CALUDE_jackson_money_l391_39115

/-- Proves that given two people where one has 5 times more money than the other, 
    and together they have $150, the person with more money has $125. -/
theorem jackson_money (williams_money : ℝ) 
  (h1 : williams_money + 5 * williams_money = 150) : 
  5 * williams_money = 125 := by
  sorry

end NUMINAMATH_CALUDE_jackson_money_l391_39115


namespace NUMINAMATH_CALUDE_mike_shortfall_l391_39183

def max_marks : ℕ := 780
def mike_score : ℕ := 212
def passing_percentage : ℚ := 30 / 100

theorem mike_shortfall :
  (↑max_marks * passing_percentage).floor - mike_score = 22 := by
  sorry

end NUMINAMATH_CALUDE_mike_shortfall_l391_39183


namespace NUMINAMATH_CALUDE_final_value_of_A_l391_39172

theorem final_value_of_A (A : Int) : A = 20 → -A + 10 = -10 := by
  sorry

end NUMINAMATH_CALUDE_final_value_of_A_l391_39172


namespace NUMINAMATH_CALUDE_songcheng_visitors_l391_39118

/-- Calculates the total number of visitors to Hangzhou Songcheng on Sunday -/
def total_visitors (morning_visitors : ℕ) (noon_departures : ℕ) (afternoon_increase : ℕ) : ℕ :=
  morning_visitors + (noon_departures + afternoon_increase)

/-- Theorem stating the total number of visitors to Hangzhou Songcheng on Sunday -/
theorem songcheng_visitors :
  total_visitors 500 119 138 = 757 := by
  sorry

end NUMINAMATH_CALUDE_songcheng_visitors_l391_39118


namespace NUMINAMATH_CALUDE_fifth_roots_of_unity_l391_39102

theorem fifth_roots_of_unity (p q r s t m : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * m^4 + q * m^3 + r * m^2 + s * m + t = 0)
  (h2 : q * m^4 + r * m^3 + s * m^2 + t * m + p = 0) :
  m^5 = 1 :=
sorry

end NUMINAMATH_CALUDE_fifth_roots_of_unity_l391_39102


namespace NUMINAMATH_CALUDE_union_complement_equals_set_l391_39139

def I : Set Int := {x | -3 < x ∧ x < 3}
def A : Set Int := {1, 2}
def B : Set Int := {-2, -1, 2}

theorem union_complement_equals_set : A ∪ (I \ B) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_set_l391_39139


namespace NUMINAMATH_CALUDE_sequence_formula_l391_39190

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the properties of the sequence
def PropertyOne (a : Sequence) : Prop :=
  ∀ m n : ℕ, m > n → a (m - n) = a m - a n

def PropertyTwo (a : Sequence) : Prop :=
  ∀ m n : ℕ, m > n → a m > a n

-- State the theorem
theorem sequence_formula (a : Sequence) 
  (h1 : PropertyOne a) (h2 : PropertyTwo a) : 
  ∃ k : ℝ, k > 0 ∧ ∀ n : ℕ, a n = k * n := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l391_39190


namespace NUMINAMATH_CALUDE_distribute_4_3_l391_39133

/-- The number of ways to distribute n distinct objects among k distinct groups,
    where each group must receive at least one object -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 4 distinct objects among 3 distinct groups,
    where each group must receive at least one object, results in 60 different ways -/
theorem distribute_4_3 : distribute 4 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_distribute_4_3_l391_39133


namespace NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l391_39150

/-- The sum of the series ∑_{n=1}^∞ 1/(n(n+3)) is equal to 11/18. -/
theorem sum_reciprocal_n_n_plus_three : 
  (∑' n : ℕ+, (1 : ℝ) / (n * (n + 3))) = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_n_n_plus_three_l391_39150


namespace NUMINAMATH_CALUDE_complex_sum_squares_l391_39124

theorem complex_sum_squares (z : ℂ) (h : Complex.abs (z - (3 - 2*I)) = 3) :
  Complex.abs (z + (1 - I))^2 + Complex.abs (z - (7 - 3*I))^2 = 94 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_squares_l391_39124


namespace NUMINAMATH_CALUDE_base_nine_proof_l391_39198

theorem base_nine_proof (b : ℕ) : 
  (∃ (n : ℕ), n = 144 ∧ 
    n = (b - 2) * (b - 3) + (b - 1) * (b - 3) + (b - 1) * (b - 2)) →
  b = 9 :=
by sorry

end NUMINAMATH_CALUDE_base_nine_proof_l391_39198


namespace NUMINAMATH_CALUDE_total_cost_calculation_l391_39143

/-- Calculates the total cost of beef and vegetables -/
theorem total_cost_calculation (beef_weight : ℝ) (veg_weight : ℝ) (veg_price : ℝ) :
  beef_weight = 4 →
  veg_weight = 6 →
  veg_price = 2 →
  beef_weight * (3 * veg_price) + veg_weight * veg_price = 36 := by
  sorry

#check total_cost_calculation

end NUMINAMATH_CALUDE_total_cost_calculation_l391_39143


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_4_l391_39169

def binary_number : List Bool := [true, true, false, true, false, true, false, false, true, false, true, true]

def last_two_digits (n : List Bool) : (Bool × Bool) :=
  match n.reverse with
  | b0 :: b1 :: _ => (b1, b0)
  | _ => (false, false)  -- Default case, should not occur for valid input

def remainder_mod_4 (digits : Bool × Bool) : Nat :=
  let (b1, b0) := digits
  2 * (if b1 then 1 else 0) + (if b0 then 1 else 0)

theorem remainder_of_binary_div_4 :
  remainder_mod_4 (last_two_digits binary_number) = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_4_l391_39169


namespace NUMINAMATH_CALUDE_total_remaining_is_13589_08_l391_39197

/-- Represents the daily sales and ingredient cost data for Du Chin's meat pie business --/
structure DailyData where
  pies_sold : ℕ
  sales : ℚ
  ingredient_cost : ℚ
  remaining : ℚ

/-- Calculates the daily data for Du Chin's meat pie business over a week --/
def calculate_week_data : List DailyData :=
  let monday_data : DailyData := {
    pies_sold := 200,
    sales := 4000,
    ingredient_cost := 2400,
    remaining := 1600
  }
  let tuesday_data : DailyData := {
    pies_sold := 220,
    sales := 4400,
    ingredient_cost := 2640,
    remaining := 1760
  }
  let wednesday_data : DailyData := {
    pies_sold := 209,
    sales := 4180,
    ingredient_cost := 2376,
    remaining := 1804
  }
  let thursday_data : DailyData := {
    pies_sold := 209,
    sales := 4180,
    ingredient_cost := 2376,
    remaining := 1804
  }
  let friday_data : DailyData := {
    pies_sold := 240,
    sales := 4800,
    ingredient_cost := 2494.80,
    remaining := 2305.20
  }
  let saturday_data : DailyData := {
    pies_sold := 221,
    sales := 4420,
    ingredient_cost := 2370.06,
    remaining := 2049.94
  }
  let sunday_data : DailyData := {
    pies_sold := 232,
    sales := 4640,
    ingredient_cost := 2370.06,
    remaining := 2269.94
  }
  [monday_data, tuesday_data, wednesday_data, thursday_data, friday_data, saturday_data, sunday_data]

/-- Calculates the total remaining money for the week --/
def total_remaining (week_data : List DailyData) : ℚ :=
  week_data.foldl (fun acc day => acc + day.remaining) 0

/-- Theorem stating that the total remaining money for the week is $13589.08 --/
theorem total_remaining_is_13589_08 :
  total_remaining (calculate_week_data) = 13589.08 := by
  sorry


end NUMINAMATH_CALUDE_total_remaining_is_13589_08_l391_39197


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l391_39179

theorem jelly_bean_probability (p_r p_o p_y p_g : ℝ) :
  p_r = 0.1 →
  p_o = 0.4 →
  p_r + p_o + p_y + p_g = 1 →
  p_y + p_g = 0.5 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l391_39179


namespace NUMINAMATH_CALUDE_intersection_implies_m_range_l391_39104

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | x^2 - 4*m*x + 2*m + 6 = 0}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_implies_m_range (m : ℝ) : (A m ∩ B).Nonempty → m ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_range_l391_39104


namespace NUMINAMATH_CALUDE_solution_set_of_decreasing_function_l391_39170

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem solution_set_of_decreasing_function (f : ℝ → ℝ) 
  (h : DecreasingFunction f) : 
  {x : ℝ | f x > f 1} = {x : ℝ | x < 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_decreasing_function_l391_39170


namespace NUMINAMATH_CALUDE_probability_of_valid_roll_l391_39185

/-- A standard six-sided die -/
def Die : Type := Fin 6

/-- The set of possible outcomes when rolling two dice -/
def TwoDiceRoll : Type := Die × Die

/-- The set of valid two-digit numbers between 40 and 50 (inclusive) -/
def ValidNumbers : Set ℕ := {n : ℕ | 40 ≤ n ∧ n ≤ 50}

/-- Function to convert a dice roll to a two-digit number -/
def rollToNumber (roll : TwoDiceRoll) : ℕ :=
  10 * (roll.1.val + 1) + (roll.2.val + 1)

/-- The set of favorable outcomes -/
def FavorableOutcomes : Set TwoDiceRoll :=
  {roll : TwoDiceRoll | rollToNumber roll ∈ ValidNumbers}

/-- Total number of possible outcomes when rolling two dice -/
def TotalOutcomes : ℕ := 36

/-- Number of favorable outcomes -/
def FavorableOutcomesCount : ℕ := 12

/-- Probability of rolling a number between 40 and 50 (inclusive) -/
theorem probability_of_valid_roll :
  (FavorableOutcomesCount : ℚ) / TotalOutcomes = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_valid_roll_l391_39185


namespace NUMINAMATH_CALUDE_license_plate_increase_l391_39120

theorem license_plate_increase : 
  let old_plates := 26 * 10^4
  let new_plates := 26^3 * 10^3
  new_plates / old_plates = 26^2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l391_39120


namespace NUMINAMATH_CALUDE_equal_area_rectangle_width_l391_39121

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: Given two rectangles of equal area, where one rectangle has dimensions 8 by x,
    and the other has dimensions 4 by 30, the value of x is 15 -/
theorem equal_area_rectangle_width :
  ∀ (x : ℝ),
  let r1 := Rectangle.mk 8 x
  let r2 := Rectangle.mk 4 30
  area r1 = area r2 → x = 15 := by
sorry


end NUMINAMATH_CALUDE_equal_area_rectangle_width_l391_39121


namespace NUMINAMATH_CALUDE_range_of_a_l391_39126

/-- A quadratic function y = x^2 + 2(a-1)x + 2 -/
def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The function decreases monotonically on (-∞, 4] -/
def decreases_on_left (a : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ ∧ x₂ ≤ 4 → f a x₁ ≥ f a x₂

/-- The function increases monotonically on [5, +∞) -/
def increases_on_right (a : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 5 ≤ x₁ ∧ x₁ ≤ x₂ → f a x₁ ≤ f a x₂

/-- The range of a given the monotonicity conditions -/
theorem range_of_a (a : ℝ) 
  (h1 : decreases_on_left a) 
  (h2 : increases_on_right a) : 
  -4 ≤ a ∧ a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l391_39126


namespace NUMINAMATH_CALUDE_min_amount_for_house_l391_39155

/-- Calculates the minimum amount needed to buy a house given the original price,
    full payment discount percentage, and deed tax percentage. -/
def min_house_purchase_amount (original_price : ℕ) (discount_percent : ℚ) (deed_tax_percent : ℚ) : ℕ :=
  let discounted_price := (original_price : ℚ) * discount_percent
  let deed_tax := discounted_price * deed_tax_percent
  (discounted_price + deed_tax).ceil.toNat

/-- Proves that the minimum amount needed to buy the house is 311,808 yuan. -/
theorem min_amount_for_house :
  min_house_purchase_amount 320000 (96 / 100) (3 / 200) = 311808 := by
  sorry

#eval min_house_purchase_amount 320000 (96 / 100) (3 / 200)

end NUMINAMATH_CALUDE_min_amount_for_house_l391_39155


namespace NUMINAMATH_CALUDE_water_fraction_after_three_replacements_l391_39199

/-- Represents the fraction of water remaining in a radiator after repeated partial replacements with antifreeze. -/
def waterFractionAfterReplacements (initialVolume : ℚ) (replacementVolume : ℚ) (numReplacements : ℕ) : ℚ :=
  ((initialVolume - replacementVolume) / initialVolume) ^ numReplacements

/-- Theorem stating that after three replacements in a 20-quart radiator, 
    the fraction of water remaining is 27/64. -/
theorem water_fraction_after_three_replacements :
  waterFractionAfterReplacements 20 5 3 = 27 / 64 := by
  sorry

#eval waterFractionAfterReplacements 20 5 3

end NUMINAMATH_CALUDE_water_fraction_after_three_replacements_l391_39199


namespace NUMINAMATH_CALUDE_log_ratio_independence_l391_39146

theorem log_ratio_independence (P K a b : ℝ) 
  (hP : P > 0) (hK : K > 0) (ha : a > 0 ∧ a ≠ 1) (hb : b > 0 ∧ b ≠ 1) : 
  (Real.log P / Real.log a) / (Real.log K / Real.log a) = 
  (Real.log P / Real.log b) / (Real.log K / Real.log b) := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_independence_l391_39146


namespace NUMINAMATH_CALUDE_sundae_price_l391_39175

/-- Proves that the price of each sundae is $1.20 given the specified conditions -/
theorem sundae_price (ice_cream_bars sundaes : ℕ) (total_price ice_cream_price : ℚ) : 
  ice_cream_bars = 125 →
  sundaes = 125 →
  total_price = 225 →
  ice_cream_price = 0.60 →
  (total_price - ice_cream_bars * ice_cream_price) / sundaes = 1.20 := by
sorry

end NUMINAMATH_CALUDE_sundae_price_l391_39175


namespace NUMINAMATH_CALUDE_triangle_properties_l391_39116

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  sum_angles : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The main theorem about properties of triangle ABC -/
theorem triangle_properties (t : Triangle) :
  (t.A - t.B = t.C → ¬(t.A = Real.pi/2 ∧ t.c > t.a ∧ t.c > t.b)) ∧
  (t.a^2 = t.b^2 - t.c^2 → t.B = Real.pi/2) ∧
  (t.A / (t.A + t.B + t.C) = 1/6 ∧ t.B / (t.A + t.B + t.C) = 1/3 ∧ t.C / (t.A + t.B + t.C) = 1/2 → t.C = Real.pi/2) ∧
  (t.a^2 / (t.a^2 + t.b^2 + t.c^2) = 9/50 ∧ t.b^2 / (t.a^2 + t.b^2 + t.c^2) = 16/50 ∧ t.c^2 / (t.a^2 + t.b^2 + t.c^2) = 25/50 → t.a^2 + t.b^2 = t.c^2) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l391_39116


namespace NUMINAMATH_CALUDE_quadratic_minimum_l391_39196

/-- Given a quadratic function f(x) = x^2 - 2x + m with a minimum value of -2 
    on the interval [2, +∞), prove that m = -2. -/
theorem quadratic_minimum (m : ℝ) : 
  (∀ x : ℝ, x ≥ 2 → x^2 - 2*x + m ≥ -2) ∧ 
  (∃ x : ℝ, x ≥ 2 ∧ x^2 - 2*x + m = -2) → 
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l391_39196


namespace NUMINAMATH_CALUDE_greatest_gcd_pentagonal_l391_39160

def P (n : ℕ+) : ℕ := (n : ℕ).succ * n

theorem greatest_gcd_pentagonal (n : ℕ+) : 
  (Nat.gcd (6 * P n) (n.val - 2) : ℕ) ≤ 24 ∧ 
  ∃ m : ℕ+, (Nat.gcd (6 * P m) (m.val - 2) : ℕ) = 24 :=
sorry

end NUMINAMATH_CALUDE_greatest_gcd_pentagonal_l391_39160


namespace NUMINAMATH_CALUDE_probability_is_two_thirty_thirds_l391_39151

/-- A square with side length 3 and 12 equally spaced points on its perimeter -/
structure SquareWithPoints where
  side_length : ℝ
  num_points : ℕ
  points_per_side : ℕ

/-- The probability of selecting two points that are one unit apart -/
def probability_one_unit_apart (s : SquareWithPoints) : ℚ :=
  4 / (s.num_points.choose 2)

/-- The main theorem stating the probability is 2/33 -/
theorem probability_is_two_thirty_thirds :
  let s : SquareWithPoints := {
    side_length := 3,
    num_points := 12,
    points_per_side := 3
  }
  probability_one_unit_apart s = 2 / 33 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_two_thirty_thirds_l391_39151


namespace NUMINAMATH_CALUDE_power_equality_l391_39149

theorem power_equality : (243 : ℕ)^4 = 3^12 * 3^8 := by sorry

end NUMINAMATH_CALUDE_power_equality_l391_39149


namespace NUMINAMATH_CALUDE_absolute_value_equality_l391_39161

theorem absolute_value_equality (y : ℝ) : 
  |y - 3| = |y - 5| → y = 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l391_39161


namespace NUMINAMATH_CALUDE_pie_count_correct_l391_39153

/-- Represents the number of pie slices served in a meal -/
structure MealServing :=
  (apple : ℕ)
  (blueberry : ℕ)
  (cherry : ℕ)
  (pumpkin : ℕ)

/-- Represents the total number of pie slices served over two days -/
structure TotalServing :=
  (apple : ℕ)
  (blueberry : ℕ)
  (cherry : ℕ)
  (pumpkin : ℕ)

def lunch_today : MealServing := ⟨3, 2, 2, 0⟩
def dinner_today : MealServing := ⟨1, 2, 1, 1⟩
def yesterday : MealServing := ⟨8, 8, 0, 0⟩

def total_served : TotalServing := ⟨12, 12, 3, 1⟩

theorem pie_count_correct : 
  lunch_today.apple + dinner_today.apple + yesterday.apple = total_served.apple ∧
  lunch_today.blueberry + dinner_today.blueberry + yesterday.blueberry = total_served.blueberry ∧
  lunch_today.cherry + dinner_today.cherry + yesterday.cherry = total_served.cherry ∧
  lunch_today.pumpkin + dinner_today.pumpkin + yesterday.pumpkin = total_served.pumpkin :=
by sorry

end NUMINAMATH_CALUDE_pie_count_correct_l391_39153


namespace NUMINAMATH_CALUDE_smallest_cube_factor_l391_39180

theorem smallest_cube_factor (n : ℕ) (h : n = 1512) :
  (∃ (y : ℕ), y > 0 ∧ n * 49 = y^3) ∧
  (∀ (x : ℕ), x > 0 ∧ x < 49 → ¬∃ (y : ℕ), y > 0 ∧ n * x = y^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_cube_factor_l391_39180


namespace NUMINAMATH_CALUDE_min_value_inequality_l391_39174

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 ≥ 5 * (5^(1/5) - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l391_39174


namespace NUMINAMATH_CALUDE_triangular_30_and_sum_30_l391_39178

/-- The nth triangular number -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n triangular numbers -/
def sum_triangular (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem triangular_30_and_sum_30 :
  (triangular 30 = 465) ∧ (sum_triangular 30 = 4960) := by sorry

end NUMINAMATH_CALUDE_triangular_30_and_sum_30_l391_39178
