import Mathlib

namespace NUMINAMATH_CALUDE_unique_rectangles_l1661_166133

/-- A rectangle with integer dimensions satisfying area and perimeter conditions -/
structure Rectangle where
  w : ℕ+  -- width
  l : ℕ+  -- length
  area_eq : w * l = 18
  perimeter_eq : 2 * w + 2 * l = 18

/-- The theorem stating that only two rectangles satisfy the conditions -/
theorem unique_rectangles : 
  ∀ r : Rectangle, (r.w = 3 ∧ r.l = 6) ∨ (r.w = 6 ∧ r.l = 3) :=
sorry

end NUMINAMATH_CALUDE_unique_rectangles_l1661_166133


namespace NUMINAMATH_CALUDE_co_presidents_selection_l1661_166155

theorem co_presidents_selection (n : ℕ) (k : ℕ) (h1 : n = 18) (h2 : k = 3) :
  Nat.choose n k = 816 := by
  sorry

end NUMINAMATH_CALUDE_co_presidents_selection_l1661_166155


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_l1661_166145

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical (x y z : ℝ) :
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 ∧ y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  x = 3 ∧ y = -3 * Real.sqrt 3 ∧ z = 2 →
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  (r, θ, z) = (6, 5 * Real.pi / 3, 2) := by
sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_l1661_166145


namespace NUMINAMATH_CALUDE_base_8_6_equivalence_l1661_166189

theorem base_8_6_equivalence :
  ∀ (n : ℕ), n > 0 →
  (∃ (C D : ℕ),
    C < 8 ∧ D < 8 ∧
    D < 6 ∧
    n = 8 * C + D ∧
    n = 6 * D + C) →
  n = 0 :=
by sorry

end NUMINAMATH_CALUDE_base_8_6_equivalence_l1661_166189


namespace NUMINAMATH_CALUDE_weight_of_b_l1661_166126

def weight_problem (a b c : ℝ) : Prop :=
  (a + b + c) / 3 = 45 ∧
  (a + b) / 2 = 40 ∧
  (b + c) / 2 = 43

theorem weight_of_b (a b c : ℝ) (h : weight_problem a b c) : b = 31 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l1661_166126


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1661_166177

theorem quadratic_equation_coefficients :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 4 * x - 1
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c ∧ a = 3 ∧ b = -4 ∧ c = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1661_166177


namespace NUMINAMATH_CALUDE_namjoon_walk_proof_l1661_166105

/-- The additional distance Namjoon walked compared to his usual route -/
def additional_distance (usual_distance initial_walk : ℝ) : ℝ :=
  2 * initial_walk + usual_distance - usual_distance

theorem namjoon_walk_proof (usual_distance initial_walk : ℝ) 
  (h1 : usual_distance = 1.2)
  (h2 : initial_walk = 0.3) :
  additional_distance usual_distance initial_walk = 0.6 := by
  sorry

#eval additional_distance 1.2 0.3

end NUMINAMATH_CALUDE_namjoon_walk_proof_l1661_166105


namespace NUMINAMATH_CALUDE_exponentiation_distributive_multiplication_multiplication_not_distributive_exponentiation_l1661_166127

theorem exponentiation_distributive_multiplication (a b c : ℝ) :
  (a * b) ^ c = a ^ c * b ^ c :=
sorry

theorem multiplication_not_distributive_exponentiation :
  ∃ a b c : ℝ, (a ^ b) * c ≠ (a * c) ^ (b * c) :=
sorry

end NUMINAMATH_CALUDE_exponentiation_distributive_multiplication_multiplication_not_distributive_exponentiation_l1661_166127


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_equals_one_l1661_166123

/-- The function f reaching an extreme value at x = 1 implies a = 1 -/
theorem extreme_value_implies_a_equals_one (a : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 + 2 * Real.sqrt x - 3 * Real.log x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_extreme_value_implies_a_equals_one_l1661_166123


namespace NUMINAMATH_CALUDE_prime_factors_of_n_l1661_166134

theorem prime_factors_of_n (n : ℕ) (h1 : n > 0) (h2 : n < 200) (h3 : ∃ k : ℕ, 14 * n = 60 * k) :
  ∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ∣ n ∧ q ∣ n ∧ r ∣ n :=
sorry

end NUMINAMATH_CALUDE_prime_factors_of_n_l1661_166134


namespace NUMINAMATH_CALUDE_total_oranges_approx_45_l1661_166162

/-- The number of bags of oranges -/
def num_bags : ℝ := 1.956521739

/-- The number of pounds of oranges per bag -/
def pounds_per_bag : ℝ := 23.0

/-- The total pounds of oranges -/
def total_pounds : ℝ := num_bags * pounds_per_bag

/-- Theorem stating that the total pounds of oranges is approximately 45.00 pounds -/
theorem total_oranges_approx_45 :
  ∃ ε > 0, |total_pounds - 45.00| < ε :=
sorry

end NUMINAMATH_CALUDE_total_oranges_approx_45_l1661_166162


namespace NUMINAMATH_CALUDE_shaded_area_of_square_with_circles_l1661_166136

/-- Given a square with side length 24 inches and three circles, each tangent to two sides of the square
    and one adjacent circle, the shaded area (area not covered by the circles) is 576 - 108π square inches. -/
theorem shaded_area_of_square_with_circles (side : ℝ) (circles : ℕ) : 
  side = 24 → circles = 3 → (side^2 - circles * (side/4)^2 * Real.pi) = 576 - 108 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_with_circles_l1661_166136


namespace NUMINAMATH_CALUDE_coefficient_x_squared_eq_five_l1661_166150

/-- The coefficient of x^2 in the expansion of (1/x^2 + x)^5 -/
def coefficient_x_squared : ℕ :=
  (Nat.choose 5 1)

theorem coefficient_x_squared_eq_five : coefficient_x_squared = 5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_eq_five_l1661_166150


namespace NUMINAMATH_CALUDE_function_value_at_50_l1661_166124

theorem function_value_at_50 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3*x + 2) = 9*x^2 - 15*x) :
  f 50 = 146 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_50_l1661_166124


namespace NUMINAMATH_CALUDE_purchasing_plan_comparison_pricing_strategy_comparison_l1661_166120

-- Purchasing plans comparison
theorem purchasing_plan_comparison 
  (a b : ℝ) (m n : ℝ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) (h4 : m > 0) (h5 : n > 0) :
  (2 * a * b) / (a + b) < (a + b) / 2 := by
sorry

-- Pricing strategies comparison
theorem pricing_strategy_comparison 
  (p q : ℝ) (h : p ≠ q) :
  100 * (1 + p) * (1 + q) < 100 * (1 + (p + q) / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_purchasing_plan_comparison_pricing_strategy_comparison_l1661_166120


namespace NUMINAMATH_CALUDE_common_off_days_count_l1661_166151

/-- Charlie's work cycle in days -/
def charlie_cycle : ℕ := 6

/-- Dana's work cycle in days -/
def dana_cycle : ℕ := 7

/-- Total number of days -/
def total_days : ℕ := 1500

/-- Function to calculate the number of common off days -/
def common_off_days (charlie_cycle dana_cycle total_days : ℕ) : ℕ :=
  2 * (total_days / (charlie_cycle.lcm dana_cycle))

/-- Theorem stating that Charlie and Dana have 70 common off days -/
theorem common_off_days_count : 
  common_off_days charlie_cycle dana_cycle total_days = 70 := by
  sorry

end NUMINAMATH_CALUDE_common_off_days_count_l1661_166151


namespace NUMINAMATH_CALUDE_probability_divisible_by_10_and_5_l1661_166158

def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def DivisibleBy10 (n : ℕ) : Prop := n % 10 = 0

def DivisibleBy5 (n : ℕ) : Prop := n % 5 = 0

def CountTwoDigitNumbers : ℕ := 90

def CountTwoDigitDivisibleBy10 : ℕ := 9

theorem probability_divisible_by_10_and_5 :
  (CountTwoDigitDivisibleBy10 : ℚ) / CountTwoDigitNumbers = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_10_and_5_l1661_166158


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1661_166174

theorem quadratic_roots_sum (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 2023 = 0 → 
  x₂^2 + x₂ - 2023 = 0 → 
  x₁^2 + 2*x₁ + x₂ = 2022 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1661_166174


namespace NUMINAMATH_CALUDE_one_positive_root_l1661_166116

def f (x : ℝ) : ℝ := x^4 + 10*x^3 - 2*x^2 + 12*x - 9

theorem one_positive_root :
  ∃! x : ℝ, x > 0 ∧ x < 1 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_positive_root_l1661_166116


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1661_166179

theorem election_winner_percentage (total_votes : ℕ) (majority : ℕ) (winner_percentage : ℚ) : 
  total_votes = 435 →
  majority = 174 →
  winner_percentage = 70 / 100 →
  (winner_percentage * total_votes : ℚ) - ((1 - winner_percentage) * total_votes : ℚ) = majority :=
by sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1661_166179


namespace NUMINAMATH_CALUDE_max_a_for_increasing_f_l1661_166159

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

-- State the theorem
theorem max_a_for_increasing_f :
  ∃ (a : ℝ), a = 1 ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ a → f x₁ < f x₂) ∧
  (∀ a' : ℝ, a' > a → ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ a' ∧ f x₁ ≥ f x₂) :=
sorry

end NUMINAMATH_CALUDE_max_a_for_increasing_f_l1661_166159


namespace NUMINAMATH_CALUDE_unique_scalar_for_vector_equation_l1661_166104

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def cross_product : V → V → V := sorry

theorem unique_scalar_for_vector_equation
  (cross_product : V → V → V)
  (h_cross_product : ∀ (x y z : V) (r : ℝ),
    cross_product (r • x) y = r • cross_product x y ∧
    cross_product x y = -cross_product y x ∧
    cross_product (x + y) z = cross_product x z + cross_product y z) :
  ∃! k : ℝ, ∀ (a b c d : V),
    a + b + c + d = 0 →
    k • (cross_product b a) + cross_product b c + cross_product c a + cross_product d a = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_scalar_for_vector_equation_l1661_166104


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l1661_166144

theorem quadratic_roots_sum_of_squares (a b s p : ℝ) : 
  a^2 + b^2 = 15 → 
  s = a + b → 
  p = a * b → 
  (∀ x, x^2 - s*x + p = 0 ↔ x = a ∨ x = b) → 
  15 = s^2 - 2*p := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l1661_166144


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1661_166166

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 6 - 3*x) → x ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1661_166166


namespace NUMINAMATH_CALUDE_jeans_final_price_l1661_166142

/-- Calculates the final price of jeans after summer and Wednesday discounts --/
theorem jeans_final_price (original_price : ℝ) (summer_discount_percent : ℝ) (wednesday_discount : ℝ) :
  original_price = 49 →
  summer_discount_percent = 50 →
  wednesday_discount = 10 →
  original_price * (1 - summer_discount_percent / 100) - wednesday_discount = 14.5 := by
  sorry

#check jeans_final_price

end NUMINAMATH_CALUDE_jeans_final_price_l1661_166142


namespace NUMINAMATH_CALUDE_diane_age_when_condition_met_l1661_166118

/-- Represents the ages of Diane, Alex, and Allison at the time when the condition is met -/
structure Ages where
  diane : ℕ
  alex : ℕ
  allison : ℕ

/-- Checks if the given ages satisfy the condition -/
def satisfiesCondition (ages : Ages) : Prop :=
  ages.diane = ages.alex / 2 ∧ ages.diane = 2 * ages.allison

/-- Represents the current ages of Diane, Alex, and Allison -/
structure CurrentAges where
  diane : ℕ
  alexPlusAllison : ℕ

/-- Theorem stating that Diane will be 78 when the condition is met -/
theorem diane_age_when_condition_met (current : CurrentAges)
    (h1 : current.diane = 16)
    (h2 : current.alexPlusAllison = 47) :
    ∃ (ages : Ages), satisfiesCondition ages ∧ ages.diane = 78 :=
  sorry

end NUMINAMATH_CALUDE_diane_age_when_condition_met_l1661_166118


namespace NUMINAMATH_CALUDE_matrix_power_four_l1661_166183

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 1, 1]

theorem matrix_power_four : 
  A ^ 4 = !![(-4 : ℝ), 0; 0, -4] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l1661_166183


namespace NUMINAMATH_CALUDE_prove_scotts_golf_score_drop_l1661_166140

def scotts_golf_problem (first_four_average : ℝ) (fifth_round_score : ℝ) : Prop :=
  let first_four_total := first_four_average * 4
  let five_round_total := first_four_total + fifth_round_score
  let new_average := five_round_total / 5
  first_four_average - new_average = 2

theorem prove_scotts_golf_score_drop :
  scotts_golf_problem 78 68 :=
sorry

end NUMINAMATH_CALUDE_prove_scotts_golf_score_drop_l1661_166140


namespace NUMINAMATH_CALUDE_can_transport_goods_l1661_166199

/-- Represents the total weight of goods in tonnes -/
def total_weight : ℝ := 13.5

/-- Represents the maximum weight of goods in a single box in tonnes -/
def max_box_weight : ℝ := 0.35

/-- Represents the number of available trucks -/
def num_trucks : ℕ := 11

/-- Represents the load capacity of each truck in tonnes -/
def truck_capacity : ℝ := 1.5

/-- Theorem stating that the given number of trucks can transport all goods in a single trip -/
theorem can_transport_goods : 
  (num_trucks : ℝ) * truck_capacity ≥ total_weight := by sorry

end NUMINAMATH_CALUDE_can_transport_goods_l1661_166199


namespace NUMINAMATH_CALUDE_power_of_125_two_thirds_l1661_166172

theorem power_of_125_two_thirds : (125 : ℝ) ^ (2/3) = 25 := by sorry

end NUMINAMATH_CALUDE_power_of_125_two_thirds_l1661_166172


namespace NUMINAMATH_CALUDE_alternative_configuration_beats_malfatti_l1661_166114

/-- Given an equilateral triangle with side length 1, the total area of three circles
    in an alternative configuration is greater than the total area of Malfatti circles. -/
theorem alternative_configuration_beats_malfatti :
  let malfatti_area : ℝ := 3 * Real.pi * (2 - Real.sqrt 3) / 8
  let alternative_area : ℝ := 11 * Real.pi / 108
  alternative_area > malfatti_area :=
by sorry

end NUMINAMATH_CALUDE_alternative_configuration_beats_malfatti_l1661_166114


namespace NUMINAMATH_CALUDE_smores_cost_example_l1661_166128

/-- The cost of supplies for S'mores given the number of people, S'mores per person, and cost per set of S'mores. -/
def smoresCost (numPeople : ℕ) (smoresPerPerson : ℕ) (costPerSet : ℚ) (smoresPerSet : ℕ) : ℚ :=
  (numPeople * smoresPerPerson : ℚ) / smoresPerSet * costPerSet

/-- Theorem: The cost of S'mores supplies for 8 people eating 3 S'mores each, where 4 S'mores cost $3, is $18. -/
theorem smores_cost_example : smoresCost 8 3 3 4 = 18 := by
  sorry

#eval smoresCost 8 3 3 4

end NUMINAMATH_CALUDE_smores_cost_example_l1661_166128


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l1661_166100

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 4) (h2 : l^2 + w^2 = d^2) :
  l * w = (20 / 41) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l1661_166100


namespace NUMINAMATH_CALUDE_mod_23_equivalence_l1661_166132

theorem mod_23_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 39548 ≡ n [ZMOD 23] ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_mod_23_equivalence_l1661_166132


namespace NUMINAMATH_CALUDE_min_value_expression_l1661_166141

theorem min_value_expression (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq_one : a + b + c = 1) :
  a + (a * b) ^ (1/3 : ℝ) + (a * b * c) ^ (1/4 : ℝ) ≥ 1/3 + 1/(3 * 3^(1/3 : ℝ)) + 1/(3 * 3^(1/4 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1661_166141


namespace NUMINAMATH_CALUDE_factorial_division_l1661_166131

theorem factorial_division :
  (9 : ℕ).factorial / (4 : ℕ).factorial = 15120 :=
by
  have h1 : (9 : ℕ).factorial = 362880 := by sorry
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1661_166131


namespace NUMINAMATH_CALUDE_eggs_left_after_recovering_capital_l1661_166156

theorem eggs_left_after_recovering_capital 
  (total_eggs : ℕ) 
  (crate_cost_cents : ℕ) 
  (selling_price_cents : ℕ) : ℕ :=
  let eggs_sold := crate_cost_cents / selling_price_cents
  total_eggs - eggs_sold

#check eggs_left_after_recovering_capital 30 500 20 = 5

end NUMINAMATH_CALUDE_eggs_left_after_recovering_capital_l1661_166156


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1661_166102

/-- Given a square with side length y that is divided into a central square
    with side length (y - z) and four congruent rectangles, prove that
    the perimeter of one of these rectangles is 2y. -/
theorem rectangle_perimeter (y z : ℝ) (hz : z < y) :
  let central_side := y - z
  let rect_long_side := y - z
  let rect_short_side := y - (y - z)
  2 * rect_long_side + 2 * rect_short_side = 2 * y :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1661_166102


namespace NUMINAMATH_CALUDE_nested_cube_root_l1661_166187

theorem nested_cube_root (N : ℝ) (h : N > 1) :
  (N * (N * (N * N^(1/3))^(1/3))^(1/3))^(1/3) = N^(40/81) := by
  sorry

end NUMINAMATH_CALUDE_nested_cube_root_l1661_166187


namespace NUMINAMATH_CALUDE_infinite_valid_moves_l1661_166180

-- Define the grid
def InfiniteSquareGrid := ℤ × ℤ

-- Define the directions
inductive Direction
| North
| South
| East
| West

-- Define a car
structure Car where
  position : InfiniteSquareGrid
  direction : Direction

-- Define the state of the grid
structure GridState where
  cars : Finset Car

-- Define a valid move
def validMove (state : GridState) (car : Car) : Prop :=
  car ∈ state.cars ∧
  (∀ other : Car, other ∈ state.cars → other.position ≠ car.position) ∧
  (∀ other : Car, other ∈ state.cars → 
    match car.direction with
    | Direction.North => other.position ≠ (car.position.1, car.position.2 + 1)
    | Direction.South => other.position ≠ (car.position.1, car.position.2 - 1)
    | Direction.East => other.position ≠ (car.position.1 + 1, car.position.2)
    | Direction.West => other.position ≠ (car.position.1 - 1, car.position.2)
  ) ∧
  (∀ other : Car, other ∈ state.cars →
    (car.direction = Direction.East ∧ other.direction = Direction.West → car.position.1 < other.position.1) ∧
    (car.direction = Direction.West ∧ other.direction = Direction.East → car.position.1 > other.position.1) ∧
    (car.direction = Direction.North ∧ other.direction = Direction.South → car.position.2 < other.position.2) ∧
    (car.direction = Direction.South ∧ other.direction = Direction.North → car.position.2 > other.position.2))

-- Define the theorem
theorem infinite_valid_moves (initialState : GridState) : 
  ∃ (moveSequence : ℕ → Car), 
    (∀ n : ℕ, validMove initialState (moveSequence n)) ∧
    (∀ car : Car, car ∈ initialState.cars → ∀ k : ℕ, ∃ n > k, moveSequence n = car) :=
sorry

end NUMINAMATH_CALUDE_infinite_valid_moves_l1661_166180


namespace NUMINAMATH_CALUDE_rational_function_property_l1661_166196

theorem rational_function_property (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 := by sorry

end NUMINAMATH_CALUDE_rational_function_property_l1661_166196


namespace NUMINAMATH_CALUDE_sufficient_condition_for_sum_of_roots_l1661_166121

theorem sufficient_condition_for_sum_of_roots 
  (a b c x₁ x₂ : ℝ) (ha : a ≠ 0) 
  (hroots : x₁ * x₁ + a⁻¹ * b * x₁ + a⁻¹ * c = 0 ∧ 
            x₂ * x₂ + a⁻¹ * b * x₂ + a⁻¹ * c = 0) :
  x₁ + x₂ = -b / a := by
  sorry


end NUMINAMATH_CALUDE_sufficient_condition_for_sum_of_roots_l1661_166121


namespace NUMINAMATH_CALUDE_book_ratio_is_two_to_one_l1661_166170

/-- Represents the number of books Thabo owns in each category -/
structure BookCounts where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- Thabo's book collection satisfies the given conditions -/
def satisfiesConditions (books : BookCounts) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 180 ∧
  books.paperbackNonfiction = books.hardcoverNonfiction + 20 ∧
  books.hardcoverNonfiction = 30

/-- The ratio of paperback fiction to paperback nonfiction books is 2:1 -/
def hasRatioTwoToOne (books : BookCounts) : Prop :=
  2 * books.paperbackNonfiction = books.paperbackFiction

theorem book_ratio_is_two_to_one (books : BookCounts) 
  (h : satisfiesConditions books) : hasRatioTwoToOne books := by
  sorry

#check book_ratio_is_two_to_one

end NUMINAMATH_CALUDE_book_ratio_is_two_to_one_l1661_166170


namespace NUMINAMATH_CALUDE_marbles_sharing_l1661_166198

theorem marbles_sharing (sienna_initial jordan_initial : ℕ)
  (h1 : sienna_initial = 150)
  (h2 : jordan_initial = 90)
  (shared : ℕ)
  (h3 : sienna_initial - shared = 3 * (jordan_initial + shared)) :
  shared = 30 := by
  sorry

end NUMINAMATH_CALUDE_marbles_sharing_l1661_166198


namespace NUMINAMATH_CALUDE_intersection_condition_l1661_166190

theorem intersection_condition (m : ℝ) : 
  let A := {x : ℝ | x^2 - 3*x + 2 = 0}
  let C := {x : ℝ | x^2 - m*x + 2 = 0}
  (A ∩ C = C) → (m = 3 ∨ -2*Real.sqrt 2 < m ∧ m < 2*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_condition_l1661_166190


namespace NUMINAMATH_CALUDE_incorrect_factorization_l1661_166122

theorem incorrect_factorization (x : ℝ) : x^2 + x - 2 ≠ (x - 2) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_factorization_l1661_166122


namespace NUMINAMATH_CALUDE_standing_arrangements_l1661_166130

def num_teachers : ℕ := 2
def num_male_students : ℕ := 3
def num_female_students : ℕ := 4

def total_people : ℕ := num_teachers + num_male_students + num_female_students

theorem standing_arrangements :
  (num_teachers.factorial * -- Permutations of teachers
   num_male_students.factorial * -- Permutations of male students
   num_female_students.factorial * -- Permutations of female students
   (total_people - 2 - num_male_students).choose num_teachers) -- Choosing positions for teachers
  = 1728 := by sorry

end NUMINAMATH_CALUDE_standing_arrangements_l1661_166130


namespace NUMINAMATH_CALUDE_complex_subtraction_l1661_166109

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3*I) (hb : b = 2 + 4*I) :
  a - 3*b = -1 - 15*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1661_166109


namespace NUMINAMATH_CALUDE_fence_repair_problem_l1661_166184

theorem fence_repair_problem : ∃ n : ℕ+, 
  (∃ x y : ℕ, x + y = n ∧ 2 * x + 3 * y = 87) ∧
  (∃ a b : ℕ, a + b = n ∧ 3 * a + 5 * b = 94) :=
by sorry

end NUMINAMATH_CALUDE_fence_repair_problem_l1661_166184


namespace NUMINAMATH_CALUDE_root_of_polynomial_l1661_166193

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 16*x^2 + 4

-- State the theorem
theorem root_of_polynomial :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 16*x^2 + 4) ∧
  -- The polynomial has degree 4
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- √5 + √3 is a root of the polynomial
  p (Real.sqrt 5 + Real.sqrt 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l1661_166193


namespace NUMINAMATH_CALUDE_regular_ngon_parallel_pairs_l1661_166157

/-- Represents a regular n-gon with a connected path visiting each vertex exactly once -/
structure RegularNGonPath (n : ℕ) where
  path : List ℕ
  is_valid : path.length = n ∧ path.toFinset.card = n

/-- Two edges (i, j) and (p, q) are parallel in a regular n-gon if and only if i + j ≡ p + q (mod n) -/
def parallel_edges (n : ℕ) (i j p q : ℕ) : Prop :=
  (i + j) % n = (p + q) % n

/-- Counts the number of parallel pairs in a path -/
def count_parallel_pairs (n : ℕ) (path : RegularNGonPath n) : ℕ :=
  sorry

theorem regular_ngon_parallel_pairs (n : ℕ) (path : RegularNGonPath n) :
  (Even n → count_parallel_pairs n path > 0) ∧
  (Odd n → count_parallel_pairs n path ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_regular_ngon_parallel_pairs_l1661_166157


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_square_imaginary_part_of_one_minus_two_i_squared_l1661_166186

theorem imaginary_part_of_complex_square : ℂ → ℝ
  | ⟨re, im⟩ => im

theorem imaginary_part_of_one_minus_two_i_squared :
  imaginary_part_of_complex_square ((1 - 2 * Complex.I) ^ 2) = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_square_imaginary_part_of_one_minus_two_i_squared_l1661_166186


namespace NUMINAMATH_CALUDE_average_trees_planted_l1661_166125

theorem average_trees_planted (total_students : ℕ) (trees_3 trees_4 trees_5 trees_6 : ℕ) 
  (h1 : total_students = 50)
  (h2 : trees_3 = 20)
  (h3 : trees_4 = 15)
  (h4 : trees_5 = 10)
  (h5 : trees_6 = 5)
  (h6 : trees_3 + trees_4 + trees_5 + trees_6 = total_students) :
  (3 * trees_3 + 4 * trees_4 + 5 * trees_5 + 6 * trees_6) / total_students = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_trees_planted_l1661_166125


namespace NUMINAMATH_CALUDE_vacation_cost_division_l1661_166110

theorem vacation_cost_division (total_cost : ℕ) (cost_difference : ℕ) : 
  total_cost = 720 →
  (total_cost / 4 + cost_difference) * 3 = total_cost →
  cost_difference = 60 →
  3 = total_cost / (total_cost / 4 + cost_difference) :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_division_l1661_166110


namespace NUMINAMATH_CALUDE_odd_digits_base4_233_l1661_166167

/-- Counts the number of odd digits in the base-4 representation of a natural number. -/
def countOddDigitsBase4 (n : ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 233 is 2. -/
theorem odd_digits_base4_233 : countOddDigitsBase4 233 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_233_l1661_166167


namespace NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l1661_166138

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | (x+2)*(4-x) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a+1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) : B ∪ C a = B → a ∈ Set.Icc (-2) 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_range_of_a_l1661_166138


namespace NUMINAMATH_CALUDE_fourth_side_length_l1661_166137

/-- A quadrilateral inscribed in a circle with given side lengths -/
structure InscribedQuadrilateral where
  radius : ℝ
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  radius_positive : radius > 0
  sides_positive : side1 > 0 ∧ side2 > 0 ∧ side3 > 0 ∧ side4 > 0
  inscribed : side1 ≤ 2 * radius ∧ side2 ≤ 2 * radius ∧ side3 ≤ 2 * radius ∧ side4 ≤ 2 * radius

/-- The theorem stating the length of the fourth side -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
    (h1 : q.radius = 250)
    (h2 : q.side1 = 250)
    (h3 : q.side2 = 250)
    (h4 : q.side3 = 100) :
    q.side4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_l1661_166137


namespace NUMINAMATH_CALUDE_option_B_is_inductive_reasoning_l1661_166149

-- Define a sequence
def a : ℕ → ℕ
| 1 => 1
| n => 3 * n - 1

-- Define the sum of the first n terms
def S (n : ℕ) : ℕ := (List.range n).map a |>.sum

-- Define inductive reasoning
def is_inductive_reasoning (process : Prop) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ (∀ k ≤ n, ∃ (result : Prop), process → result)

-- Theorem statement
theorem option_B_is_inductive_reasoning :
  is_inductive_reasoning (∃ (formula : ℕ → ℕ), ∀ n, S n = formula n) :=
sorry

end NUMINAMATH_CALUDE_option_B_is_inductive_reasoning_l1661_166149


namespace NUMINAMATH_CALUDE_certain_number_proof_l1661_166153

theorem certain_number_proof : ∃ x : ℝ, (213 * 16 = 3408 ∧ 16 * x = 340.8) → x = 21.3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1661_166153


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1661_166163

theorem water_tank_capacity (x : ℝ) : 
  (2/3 : ℝ) * x - (1/3 : ℝ) * x = 15 → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1661_166163


namespace NUMINAMATH_CALUDE_davids_english_marks_l1661_166194

/-- Given David's marks in various subjects and his average, prove his marks in English --/
theorem davids_english_marks :
  let math_marks : ℕ := 95
  let physics_marks : ℕ := 82
  let chemistry_marks : ℕ := 97
  let biology_marks : ℕ := 95
  let average_marks : ℕ := 93
  let total_subjects : ℕ := 5
  let total_marks : ℕ := average_marks * total_subjects
  let known_marks_sum : ℕ := math_marks + physics_marks + chemistry_marks + biology_marks
  let english_marks : ℕ := total_marks - known_marks_sum
  english_marks = 96 := by
sorry

end NUMINAMATH_CALUDE_davids_english_marks_l1661_166194


namespace NUMINAMATH_CALUDE_fraction_equality_l1661_166169

theorem fraction_equality (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  1 / x - 1 / y = 2 → (x + x * y - y) / (x - x * y - y) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1661_166169


namespace NUMINAMATH_CALUDE_myrtle_has_three_hens_l1661_166103

/-- The number of hens Myrtle has -/
def num_hens : ℕ := sorry

/-- The number of eggs each hen lays per day -/
def eggs_per_hen_per_day : ℕ := 3

/-- The number of days Myrtle was gone -/
def days_gone : ℕ := 7

/-- The number of eggs the neighbor took -/
def eggs_taken_by_neighbor : ℕ := 12

/-- The number of eggs Myrtle dropped -/
def eggs_dropped : ℕ := 5

/-- The number of eggs Myrtle has remaining -/
def eggs_remaining : ℕ := 46

/-- Theorem stating that Myrtle has 3 hens -/
theorem myrtle_has_three_hens :
  num_hens = 3 :=
by sorry

end NUMINAMATH_CALUDE_myrtle_has_three_hens_l1661_166103


namespace NUMINAMATH_CALUDE_sqrt_mantissa_equality_l1661_166147

theorem sqrt_mantissa_equality (m n : ℕ) (h1 : m ≠ n) (h2 : m > 0) (h3 : n > 0) :
  (∃ (k : ℤ), Real.sqrt m - Real.sqrt n = k) → (∃ (a b : ℕ), m = a^2 ∧ n = b^2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_mantissa_equality_l1661_166147


namespace NUMINAMATH_CALUDE_actual_height_of_boy_l1661_166171

/-- Calculates the actual height of a boy in a class given the following conditions:
  * There are 35 boys in the class
  * The initially calculated average height was 182 cm
  * One boy's height was wrongly written as 166 cm
  * The actual average height is 180 cm
-/
theorem actual_height_of_boy (n : ℕ) (initial_avg : ℝ) (wrong_height : ℝ) (actual_avg : ℝ) :
  n = 35 →
  initial_avg = 182 →
  wrong_height = 166 →
  actual_avg = 180 →
  ∃ (x : ℝ), x = 236 ∧ n * actual_avg = (n * initial_avg - wrong_height + x) :=
by sorry

end NUMINAMATH_CALUDE_actual_height_of_boy_l1661_166171


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l1661_166108

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℝ := 21

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℝ := green_pill_cost - 3

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 819

theorem green_pill_cost_proof :
  green_pill_cost = 21 ∧
  pink_pill_cost = green_pill_cost - 3 ∧
  treatment_days = 21 ∧
  total_cost = 819 ∧
  treatment_days * (green_pill_cost + pink_pill_cost) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_green_pill_cost_proof_l1661_166108


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1661_166182

/-- Given a system of linear equations with a specific k value, 
    prove that xz/y^2 equals a specific constant --/
theorem system_solution_ratio (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (eq1 : x + (16/5)*y + 4*z = 0)
  (eq2 : 3*x + (16/5)*y + z = 0)
  (eq3 : 2*x + 4*y + 3*z = 0) :
  ∃ (c : ℝ), x*z/y^2 = c :=
by sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1661_166182


namespace NUMINAMATH_CALUDE_min_area_triangle_PAB_l1661_166129

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x + 5)^2 + (y - 3)^2 = 2

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Define points A and B
def point_A : ℝ × ℝ := (0, 2)
def point_B : ℝ × ℝ := (-2, 0)

-- Statement of the theorem
theorem min_area_triangle_PAB :
  ∃ (min_area : ℝ),
    (∀ (P : ℝ × ℝ),
      circle_C P.1 P.2 →
      ∀ (area : ℝ),
      area = abs ((P.1 - point_A.1) * (point_B.2 - point_A.2) -
                  (point_B.1 - point_A.1) * (P.2 - point_A.2)) / 2 →
      area ≥ min_area) ∧
    min_area = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_area_triangle_PAB_l1661_166129


namespace NUMINAMATH_CALUDE_block_distance_is_200_l1661_166192

/-- The distance of one time around the block -/
def block_distance : ℝ := sorry

/-- The number of times Johnny runs around the block -/
def johnny_laps : ℕ := 4

/-- The number of times Mickey runs around the block -/
def mickey_laps : ℕ := johnny_laps / 2

/-- The average distance run by Johnny and Mickey -/
def average_distance : ℝ := 600

theorem block_distance_is_200 :
  block_distance = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_block_distance_is_200_l1661_166192


namespace NUMINAMATH_CALUDE_cookie_ratio_l1661_166191

/-- Given a total of 14 bags, 28 cookies, and 2 bags of cookies,
    prove that the ratio of cookies in each bag to the total number of cookies is 1:2 -/
theorem cookie_ratio (total_bags : ℕ) (total_cookies : ℕ) (cookie_bags : ℕ)
  (h1 : total_bags = 14)
  (h2 : total_cookies = 28)
  (h3 : cookie_bags = 2) :
  (total_cookies / cookie_bags) / total_cookies = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_cookie_ratio_l1661_166191


namespace NUMINAMATH_CALUDE_infinitely_many_unlucky_numbers_l1661_166185

/-- A natural number is unlucky if it cannot be represented as x^2 - 1 or y^2 - 1
    for any natural numbers x, y > 1. -/
def isUnlucky (n : ℕ) : Prop :=
  ∀ x y : ℕ, x > 1 ∧ y > 1 → n ≠ x^2 - 1 ∧ n ≠ y^2 - 1

/-- There are infinitely many unlucky numbers. -/
theorem infinitely_many_unlucky_numbers :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ isUnlucky n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_unlucky_numbers_l1661_166185


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l1661_166161

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation 
  (rate : ℝ) (interest : ℝ) (time : ℝ) :
  rate = 4.783950617283951 →
  interest = 155 →
  time = 4 →
  ∃ (principal : ℝ), 
    (principal * rate * time) / 100 = interest ∧ 
    abs (principal - 810.13) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l1661_166161


namespace NUMINAMATH_CALUDE_b_91_mod_49_l1661_166175

/-- Definition of the sequence bₙ -/
def b (n : ℕ) : ℕ := 12^n + 14^n

/-- Theorem stating that b₉₁ mod 49 = 38 -/
theorem b_91_mod_49 : b 91 % 49 = 38 := by
  sorry

end NUMINAMATH_CALUDE_b_91_mod_49_l1661_166175


namespace NUMINAMATH_CALUDE_log_difference_negative_l1661_166154

theorem log_difference_negative (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  Real.log (b - a) < 0 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_negative_l1661_166154


namespace NUMINAMATH_CALUDE_davis_class_groups_l1661_166173

/-- The number of groups in Miss Davis's class -/
def number_of_groups (sticks_per_group : ℕ) (initial_sticks : ℕ) (remaining_sticks : ℕ) : ℕ :=
  (initial_sticks - remaining_sticks) / sticks_per_group

/-- Theorem stating the number of groups in Miss Davis's class -/
theorem davis_class_groups :
  number_of_groups 15 170 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_davis_class_groups_l1661_166173


namespace NUMINAMATH_CALUDE_willy_crayon_count_l1661_166148

/-- The number of crayons Lucy has -/
def lucy_crayons : ℕ := 290

/-- The number of additional crayons Willy has compared to Lucy -/
def additional_crayons : ℕ := 1110

/-- The total number of crayons Willy has -/
def willy_crayons : ℕ := lucy_crayons + additional_crayons

theorem willy_crayon_count : willy_crayons = 1400 := by
  sorry

end NUMINAMATH_CALUDE_willy_crayon_count_l1661_166148


namespace NUMINAMATH_CALUDE_firm_ratio_l1661_166146

theorem firm_ratio (partners associates : ℕ) : 
  partners = 14 ∧ 
  14 * 34 = associates + 35 → 
  (partners : ℚ) / associates = 2 / 63 := by
sorry

end NUMINAMATH_CALUDE_firm_ratio_l1661_166146


namespace NUMINAMATH_CALUDE_unsuitable_temp_l1661_166113

def storage_temp := -18
def temp_range := 2

def is_suitable_temp (temp : Int) : Prop :=
  (storage_temp - temp_range) ≤ temp ∧ temp ≤ (storage_temp + temp_range)

theorem unsuitable_temp :
  ¬(is_suitable_temp (-21)) :=
by
  sorry

end NUMINAMATH_CALUDE_unsuitable_temp_l1661_166113


namespace NUMINAMATH_CALUDE_correct_passengers_off_l1661_166152

/-- Calculates the number of passengers who got off the bus at other stops -/
def passengers_who_got_off (initial : ℕ) (first_stop : ℕ) (other_stops : ℕ) (final : ℕ) : ℕ :=
  initial + first_stop - (final - other_stops)

theorem correct_passengers_off : passengers_who_got_off 50 16 5 49 = 22 := by
  sorry

end NUMINAMATH_CALUDE_correct_passengers_off_l1661_166152


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1661_166111

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) → 
  l * w = 100 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l1661_166111


namespace NUMINAMATH_CALUDE_sunzi_problem_l1661_166178

theorem sunzi_problem : ∃! n : ℕ, 
  100 < n ∧ n < 200 ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sunzi_problem_l1661_166178


namespace NUMINAMATH_CALUDE_courses_difference_count_l1661_166119

/-- The number of available courses -/
def total_courses : ℕ := 4

/-- The number of courses each person chooses -/
def courses_per_person : ℕ := 2

/-- The number of ways to choose courses with at least one difference -/
def ways_with_difference : ℕ := 30

/-- Theorem stating that the number of ways with at least one course different is 30 -/
theorem courses_difference_count :
  (total_courses.choose courses_per_person * courses_per_person.choose courses_per_person) +
  (total_courses.choose 1 * (total_courses - 1).choose 1 * (total_courses - 2).choose 1) =
  ways_with_difference :=
sorry

end NUMINAMATH_CALUDE_courses_difference_count_l1661_166119


namespace NUMINAMATH_CALUDE_y_relationship_l1661_166107

/-- A linear function with slope -2 and y-intercept 5 -/
def f (x : ℝ) : ℝ := -2 * x + 5

/-- Theorem stating the relationship between y-values for specific x-values in the linear function f -/
theorem y_relationship (x₁ y₁ y₂ y₃ : ℝ) 
  (h1 : f x₁ = y₁) 
  (h2 : f (x₁ - 2) = y₂) 
  (h3 : f (x₁ + 3) = y₃) : 
  y₂ > y₁ ∧ y₁ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_relationship_l1661_166107


namespace NUMINAMATH_CALUDE_opposite_of_negative_2022_l1661_166197

-- Define the concept of opposite
def opposite (a : ℤ) : ℤ := -a

-- Theorem statement
theorem opposite_of_negative_2022 : opposite (-2022) = 2022 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2022_l1661_166197


namespace NUMINAMATH_CALUDE_division_problem_l1661_166106

theorem division_problem : (120 : ℝ) / (5 / 2.5) = 60 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1661_166106


namespace NUMINAMATH_CALUDE_probability_product_72_l1661_166135

/-- A function representing the possible outcomes of rolling a standard die -/
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := (standardDie.card) ^ 3

/-- The number of favorable outcomes (combinations that multiply to 72) -/
def favorableOutcomes : ℕ := 6

/-- The probability of rolling three dice such that their product is 72 -/
def probabilityProductIs72 : ℚ := favorableOutcomes / totalOutcomes

theorem probability_product_72 : probabilityProductIs72 = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_probability_product_72_l1661_166135


namespace NUMINAMATH_CALUDE_remainder_theorem_l1661_166188

-- Define the polynomial q(x)
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 5

-- State the theorem
theorem remainder_theorem (D E F : ℝ) :
  (∃ k : ℝ, q D E F x = (x - 2) * k + 15) →
  (∃ m : ℝ, q D E F x = (x + 2) * m + 15) :=
by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1661_166188


namespace NUMINAMATH_CALUDE_sum_difference_theorem_l1661_166160

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_five (n : ℕ) : ℕ :=
  let m := n % 5
  if m < 3 then n - m else n + (5 - m)

def emma_sum (n : ℕ) : ℕ :=
  List.range n |> List.map round_to_nearest_five |> List.sum

theorem sum_difference_theorem :
  sum_to_n 100 - emma_sum 100 = 4750 := by sorry

end NUMINAMATH_CALUDE_sum_difference_theorem_l1661_166160


namespace NUMINAMATH_CALUDE_max_distinct_sum_100_l1661_166195

/-- The sum of the first n positive integers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- A function that checks if a number is the maximum number of distinct positive integers that sum to 100 -/
def is_max_distinct_sum (k : ℕ) : Prop :=
  triangular_sum k ≤ 100 ∧ 
  triangular_sum (k + 1) > 100

theorem max_distinct_sum_100 : is_max_distinct_sum 13 := by
  sorry

#check max_distinct_sum_100

end NUMINAMATH_CALUDE_max_distinct_sum_100_l1661_166195


namespace NUMINAMATH_CALUDE_black_marble_probability_l1661_166181

theorem black_marble_probability :
  let yellow : ℕ := 24
  let blue : ℕ := 18
  let green : ℕ := 12
  let red : ℕ := 8
  let white : ℕ := 7
  let black : ℕ := 3
  let purple : ℕ := 2
  let total : ℕ := yellow + blue + green + red + white + black + purple
  (black : ℚ) / total = 3 / 74 := by sorry

end NUMINAMATH_CALUDE_black_marble_probability_l1661_166181


namespace NUMINAMATH_CALUDE_sum_of_squares_l1661_166165

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_cubes_sevens : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6/7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1661_166165


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l1661_166176

/-- A rectangular prism with different dimensions for length, width, and height. -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0
  different_dimensions : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- The number of pairs of parallel edges in a rectangular prism. -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 12

/-- Theorem stating that a rectangular prism has exactly 12 pairs of parallel edges. -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  parallel_edge_pairs prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l1661_166176


namespace NUMINAMATH_CALUDE_transformation_eventually_repeats_l1661_166117

/-- Represents a transformation step on a sequence of natural numbers -/
def transform (s : List ℕ) : List ℕ :=
  s.map (λ x => s.count x)

/-- Represents the sequence of transformations applied to an initial sequence -/
def transformation_sequence (initial : List ℕ) : ℕ → List ℕ
  | 0 => initial
  | n + 1 => transform (transformation_sequence initial n)

/-- The theorem stating that the transformation sequence will eventually repeat -/
theorem transformation_eventually_repeats (initial : List ℕ) :
  ∃ n : ℕ, transformation_sequence initial n = transformation_sequence initial (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_transformation_eventually_repeats_l1661_166117


namespace NUMINAMATH_CALUDE_weight_of_new_person_l1661_166101

/-- The weight of the new person in a group where the average weight has increased --/
def new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) : ℝ :=
  old_weight + n * avg_increase

/-- Theorem: The weight of the new person in the given scenario is 78.5 kg --/
theorem weight_of_new_person :
  new_person_weight 9 1.5 65 = 78.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l1661_166101


namespace NUMINAMATH_CALUDE_stream_current_rate_l1661_166115

/-- Represents the problem of finding the stream's current rate given rowing conditions. -/
theorem stream_current_rate
  (distance : ℝ)
  (normal_time_diff : ℝ)
  (triple_speed_time_diff : ℝ)
  (h1 : distance = 18)
  (h2 : normal_time_diff = 4)
  (h3 : triple_speed_time_diff = 2)
  (h4 : ∀ (r w : ℝ),
    (distance / (r + w) + normal_time_diff = distance / (r - w)) →
    (distance / (3 * r + w) + triple_speed_time_diff = distance / (3 * r - w)) →
    w = 9 / 8) :
  ∃ (w : ℝ), w = 9 / 8 ∧ 
    (∃ (r : ℝ), 
      (distance / (r + w) + normal_time_diff = distance / (r - w)) ∧
      (distance / (3 * r + w) + triple_speed_time_diff = distance / (3 * r - w))) :=
by
  sorry

end NUMINAMATH_CALUDE_stream_current_rate_l1661_166115


namespace NUMINAMATH_CALUDE_missing_number_proof_l1661_166164

theorem missing_number_proof : ∃ x : ℤ, (4 + 3) + (8 - 3 - x) = 11 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1661_166164


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_l1661_166139

/-- The number of bouncy balls in each pack -/
def ballsPerPack : ℕ := 10

/-- The number of packs of red bouncy balls -/
def redPacks : ℕ := 4

/-- The number of packs of yellow bouncy balls -/
def yellowPacks : ℕ := 8

/-- The number of packs of green bouncy balls -/
def greenPacks : ℕ := 4

/-- The total number of bouncy balls Maggie bought -/
def totalBalls : ℕ := ballsPerPack * (redPacks + yellowPacks + greenPacks)

theorem maggie_bouncy_balls : totalBalls = 160 := by
  sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_l1661_166139


namespace NUMINAMATH_CALUDE_sin_cos_sum_identity_l1661_166143

theorem sin_cos_sum_identity (x y : ℝ) :
  Real.sin (x + y) * Real.cos (2 * y) + Real.cos (x + y) * Real.sin (2 * y) = Real.sin (x + 3 * y) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_identity_l1661_166143


namespace NUMINAMATH_CALUDE_intersection_locus_l1661_166112

theorem intersection_locus (m : ℝ) (x y : ℝ) : 
  (m * x - y + 1 = 0 ∧ x - m * y - 1 = 0) → 
  (x - y = 0 ∨ x - y + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_intersection_locus_l1661_166112


namespace NUMINAMATH_CALUDE_divisible_by_9_digit_sum_l1661_166168

def is_divisible_by_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def digit_sum (n : ℕ) : ℕ := sorry

theorem divisible_by_9_digit_sum (n : ℕ) (h : ℕ) :
  (∃ d : ℕ, n = 10 * d + h) →  -- n has h as its 10's digit
  is_divisible_by_9 n →        -- n is divisible by 9
  h = 1 →                      -- h (10's digit) is 1
  ∃ k : ℕ, digit_sum n = 9 * k -- sum of all digits is multiple of 9
  := by sorry

end NUMINAMATH_CALUDE_divisible_by_9_digit_sum_l1661_166168
