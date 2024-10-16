import Mathlib

namespace NUMINAMATH_CALUDE_system_solvability_l3001_300140

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x * Real.cos a + y * Real.sin a - 2 ≤ 0 ∧
  x^2 + y^2 + 6*x - 2*y - b^2 + 4*b + 6 = 0

-- Define the solution set for b
def solution_set (b : ℝ) : Prop :=
  b ≤ 4 - Real.sqrt 10 ∨ b ≥ Real.sqrt 10

-- Theorem statement
theorem system_solvability (b : ℝ) :
  (∀ a, ∃ x y, system x y a b) ↔ solution_set b := by
  sorry

end NUMINAMATH_CALUDE_system_solvability_l3001_300140


namespace NUMINAMATH_CALUDE_min_value_expression_l3001_300156

theorem min_value_expression (x : ℝ) : (x^2 + 13) / Real.sqrt (x^2 + 7) ≥ 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3001_300156


namespace NUMINAMATH_CALUDE_max_value_a_l3001_300122

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 4 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 150) : 
  a ≤ 8924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 8924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 150 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l3001_300122


namespace NUMINAMATH_CALUDE_power_function_odd_l3001_300167

/-- A power function f(x) = (a-1)x^b passing through the point (a, 1/8) is odd. -/
theorem power_function_odd (a b : ℝ) (h : (a - 1) * a^b = 1/8) : 
  ∀ x ≠ 0, (a - 1) * (-x)^b = -((a - 1) * x^b) :=
by sorry

end NUMINAMATH_CALUDE_power_function_odd_l3001_300167


namespace NUMINAMATH_CALUDE_digit_append_theorem_l3001_300108

theorem digit_append_theorem (n : ℕ) : 
  (∃ d : ℕ, d ≤ 9 ∧ 10 * n + d = 13 * n) ↔ (n = 1 ∨ n = 2 ∨ n = 3) :=
sorry

end NUMINAMATH_CALUDE_digit_append_theorem_l3001_300108


namespace NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l3001_300169

-- Define the repeating decimal 0.3333...
def repeating_decimal : ℚ := 1 / 3

-- Theorem statement
theorem divide_eight_by_repeating_third : 8 / repeating_decimal = 24 := by
  sorry

end NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l3001_300169


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l3001_300151

/-- The measure of one interior angle of a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ := by
  -- Define the number of sides in an octagon
  let n : ℕ := 8

  -- Define the sum of interior angles for an n-sided polygon
  let sum_interior_angles : ℝ := 180 * (n - 2)

  -- Define the measure of one interior angle
  let one_angle : ℝ := sum_interior_angles / n

  -- Prove that one_angle equals 135
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l3001_300151


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3001_300194

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| > a) → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3001_300194


namespace NUMINAMATH_CALUDE_weight_puzzle_l3001_300166

theorem weight_puzzle (w₁ w₂ w₃ w₄ : ℕ) 
  (h1 : w₁ + w₂ = 1700 ∨ w₁ + w₃ = 1700 ∨ w₁ + w₄ = 1700 ∨ w₂ + w₃ = 1700 ∨ w₂ + w₄ = 1700 ∨ w₃ + w₄ = 1700)
  (h2 : w₁ + w₂ = 1870 ∨ w₁ + w₃ = 1870 ∨ w₁ + w₄ = 1870 ∨ w₂ + w₃ = 1870 ∨ w₂ + w₄ = 1870 ∨ w₃ + w₄ = 1870)
  (h3 : w₁ + w₂ = 2110 ∨ w₁ + w₃ = 2110 ∨ w₁ + w₄ = 2110 ∨ w₂ + w₃ = 2110 ∨ w₂ + w₄ = 2110 ∨ w₃ + w₄ = 2110)
  (h4 : w₁ + w₂ = 2330 ∨ w₁ + w₃ = 2330 ∨ w₁ + w₄ = 2330 ∨ w₂ + w₃ = 2330 ∨ w₂ + w₄ = 2330 ∨ w₃ + w₄ = 2330)
  (h5 : w₁ + w₂ = 2500 ∨ w₁ + w₃ = 2500 ∨ w₁ + w₄ = 2500 ∨ w₂ + w₃ = 2500 ∨ w₂ + w₄ = 2500 ∨ w₃ + w₄ = 2500)
  (h_distinct : w₁ ≠ w₂ ∧ w₁ ≠ w₃ ∧ w₁ ≠ w₄ ∧ w₂ ≠ w₃ ∧ w₂ ≠ w₄ ∧ w₃ ≠ w₄) :
  w₁ + w₂ = 2090 ∨ w₁ + w₃ = 2090 ∨ w₁ + w₄ = 2090 ∨ w₂ + w₃ = 2090 ∨ w₂ + w₄ = 2090 ∨ w₃ + w₄ = 2090 :=
by sorry

end NUMINAMATH_CALUDE_weight_puzzle_l3001_300166


namespace NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_16_l3001_300187

theorem fourth_root_81_times_cube_root_27_times_sqrt_16 : 
  (81 : ℝ) ^ (1/4 : ℝ) * (27 : ℝ) ^ (1/3 : ℝ) * (16 : ℝ) ^ (1/2 : ℝ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_81_times_cube_root_27_times_sqrt_16_l3001_300187


namespace NUMINAMATH_CALUDE_a4_square_area_l3001_300159

/-- Represents the properties of an A4 sheet of paper -/
structure A4Sheet where
  length : Real
  width : Real
  ratio_preserved : length / width = length / (2 * width)

theorem a4_square_area (sheet : A4Sheet) (h1 : sheet.length = 29.7) :
  ∃ (area : Real), abs (area - sheet.width ^ 2) < 0.05 ∧ abs (area - 441.0) < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_a4_square_area_l3001_300159


namespace NUMINAMATH_CALUDE_eric_pencil_boxes_l3001_300163

theorem eric_pencil_boxes (pencils_per_box : ℕ) (total_pencils : ℕ) (h1 : pencils_per_box = 9) (h2 : total_pencils = 27) :
  total_pencils / pencils_per_box = 3 :=
by sorry

end NUMINAMATH_CALUDE_eric_pencil_boxes_l3001_300163


namespace NUMINAMATH_CALUDE_kolya_best_strategy_method1_most_advantageous_method2_3_least_advantageous_l3001_300154

/-- Represents the number of nuts Kolya gets in each method -/
structure KolyaNuts (n : ℕ) where
  method1 : ℕ
  method2 : ℕ
  method3 : ℕ

/-- The theorem stating the most and least advantageous methods for Kolya -/
theorem kolya_best_strategy (n : ℕ) (h : n ≥ 2) :
  ∃ (k : KolyaNuts n),
    k.method1 ≥ n + 1 ∧
    k.method2 ≤ n ∧
    k.method3 ≤ n :=
by sorry

/-- Helper function to determine the most advantageous method -/
def most_advantageous (k : KolyaNuts n) : ℕ :=
  max k.method1 (max k.method2 k.method3)

/-- Helper function to determine the least advantageous method -/
def least_advantageous (k : KolyaNuts n) : ℕ :=
  min k.method1 (min k.method2 k.method3)

/-- Theorem stating that method 1 is the most advantageous for Kolya -/
theorem method1_most_advantageous (n : ℕ) (h : n ≥ 2) :
  ∃ (k : KolyaNuts n), most_advantageous k = k.method1 :=
by sorry

/-- Theorem stating that methods 2 and 3 are the least advantageous for Kolya -/
theorem method2_3_least_advantageous (n : ℕ) (h : n ≥ 2) :
  ∃ (k : KolyaNuts n), least_advantageous k = k.method2 ∧ least_advantageous k = k.method3 :=
by sorry

end NUMINAMATH_CALUDE_kolya_best_strategy_method1_most_advantageous_method2_3_least_advantageous_l3001_300154


namespace NUMINAMATH_CALUDE_card_game_result_l3001_300118

/-- Represents the number of cards in each pile -/
structure CardPiles :=
  (left : ℕ)
  (middle : ℕ)
  (right : ℕ)

/-- The card game operations -/
def card_game_operations (initial : CardPiles) : CardPiles :=
  let step1 := initial
  let step2 := CardPiles.mk (step1.left - 2) (step1.middle + 2) step1.right
  let step3 := CardPiles.mk step2.left (step2.middle + 1) (step2.right - 1)
  CardPiles.mk step3.left.succ (step3.middle - step3.left) step3.right

theorem card_game_result (initial : CardPiles) 
  (h1 : initial.left = initial.middle)
  (h2 : initial.middle = initial.right)
  (h3 : initial.left ≥ 2) :
  (card_game_operations initial).middle = 5 :=
sorry

end NUMINAMATH_CALUDE_card_game_result_l3001_300118


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3001_300164

theorem inverse_variation_problem (k : ℝ) (x y : ℝ → ℝ) (h1 : ∀ t, 5 * y t = k / (x t)^2)
  (h2 : y 1 = 16) (h3 : x 1 = 1) : y 8 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3001_300164


namespace NUMINAMATH_CALUDE_time_expression_l3001_300128

/-- Represents the motion of a particle under combined constant accelerations -/
structure ParticleMotion where
  g : ℝ  -- Constant acceleration g
  a : ℝ  -- Additional constant acceleration a
  V₀ : ℝ  -- Initial velocity
  t : ℝ  -- Time
  V : ℝ  -- Final velocity
  S : ℝ  -- Displacement

/-- The final velocity equation for the particle motion -/
def velocity_equation (p : ParticleMotion) : Prop :=
  p.V = (p.g + p.a) * p.t + p.V₀

/-- The displacement equation for the particle motion -/
def displacement_equation (p : ParticleMotion) : Prop :=
  p.S = (1/2) * (p.g + p.a) * p.t^2 + p.V₀ * p.t

/-- Theorem stating that the time can be expressed in terms of S, V, and V₀ -/
theorem time_expression (p : ParticleMotion) 
  (h1 : velocity_equation p) 
  (h2 : displacement_equation p) : 
  p.t = (2 * p.S) / (p.V + p.V₀) := by
  sorry

end NUMINAMATH_CALUDE_time_expression_l3001_300128


namespace NUMINAMATH_CALUDE_tabitha_honey_per_cup_l3001_300150

/-- Proves that Tabitha adds 1 serving of honey per cup of tea -/
theorem tabitha_honey_per_cup :
  ∀ (cups_per_night : ℕ) 
    (container_ounces : ℕ) 
    (servings_per_ounce : ℕ) 
    (nights : ℕ),
  cups_per_night = 2 →
  container_ounces = 16 →
  servings_per_ounce = 6 →
  nights = 48 →
  (container_ounces * servings_per_ounce) / (cups_per_night * nights) = 1 :=
by
  sorry

#check tabitha_honey_per_cup

end NUMINAMATH_CALUDE_tabitha_honey_per_cup_l3001_300150


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l3001_300192

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q :
  (Set.univ \ P) ∩ Q = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l3001_300192


namespace NUMINAMATH_CALUDE_root_product_theorem_l3001_300127

theorem root_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 2 = 0) →
  (b^2 - m*b + 2 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 9/2 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3001_300127


namespace NUMINAMATH_CALUDE_ellipse_locus_and_intercept_range_l3001_300160

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point B
def B : ℝ × ℝ := (0, 1)

-- Define the perpendicularity condition
def perpendicular (P Q : ℝ × ℝ) : Prop :=
  (P.2 - B.2) * (Q.2 - B.2) = -(P.1 - B.1) * (Q.1 - B.1)

-- Define the projection M
def M (P Q : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the perpendicular bisector l
def l (P Q : ℝ × ℝ) : ℝ → ℝ := sorry

-- Define the x-intercept of l
def x_intercept (P Q : ℝ × ℝ) : ℝ := sorry

theorem ellipse_locus_and_intercept_range :
  ∀ (P Q : ℝ × ℝ),
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  P ≠ B →
  Q ≠ B →
  perpendicular P Q →
  (∀ (x y : ℝ), 
    (x, y) = M P Q →
    y ≠ 1 →
    x^2 + (y - 1/5)^2 = (4/5)^2) ∧
  (-9/20 ≤ x_intercept P Q ∧ x_intercept P Q ≤ 9/20) :=
sorry

end NUMINAMATH_CALUDE_ellipse_locus_and_intercept_range_l3001_300160


namespace NUMINAMATH_CALUDE_max_rental_income_l3001_300155

/-- Represents the daily rental income for the construction company. -/
def daily_rental_income (x : ℕ) : ℝ :=
  -200 * x + 80000

/-- The problem statement and proof objective. -/
theorem max_rental_income :
  let total_vehicles : ℕ := 50
  let type_a_vehicles : ℕ := 20
  let type_b_vehicles : ℕ := 30
  let site_a_vehicles : ℕ := 30
  let site_b_vehicles : ℕ := 20
  let site_a_type_a_price : ℝ := 1800
  let site_a_type_b_price : ℝ := 1600
  let site_b_type_a_price : ℝ := 1600
  let site_b_type_b_price : ℝ := 1200
  ∀ x : ℕ, x ≤ type_a_vehicles →
    daily_rental_income x ≤ 80000 ∧
    (∃ x₀ : ℕ, x₀ ≤ type_a_vehicles ∧ daily_rental_income x₀ = 80000) :=
by sorry

#check max_rental_income

end NUMINAMATH_CALUDE_max_rental_income_l3001_300155


namespace NUMINAMATH_CALUDE_sin_graph_shift_l3001_300182

/-- Theorem: Shifting the graph of y = 3sin(2x) to the right by π/16 units 
    results in the graph of y = 3sin(2x - π/8) -/
theorem sin_graph_shift (x : ℝ) : 
  3 * Real.sin (2 * (x - π/16)) = 3 * Real.sin (2 * x - π/8) := by
  sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l3001_300182


namespace NUMINAMATH_CALUDE_polygon_sides_with_45_degree_exterior_angles_l3001_300134

theorem polygon_sides_with_45_degree_exterior_angles :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    exterior_angle = 45 →
    (n : ℝ) * exterior_angle = 360 →
    n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_with_45_degree_exterior_angles_l3001_300134


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3001_300147

theorem smallest_number_with_remainders : ∃! x : ℕ, 
  x > 0 ∧
  x % 45 = 4 ∧
  x % 454 = 45 ∧
  x % 4545 = 454 ∧
  x % 45454 = 4545 ∧
  ∀ y : ℕ, y > 0 ∧ y % 45 = 4 ∧ y % 454 = 45 ∧ y % 4545 = 454 ∧ y % 45454 = 4545 → x ≤ y :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3001_300147


namespace NUMINAMATH_CALUDE_sum_equals_negative_two_thirds_l3001_300180

theorem sum_equals_negative_two_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 4) : 
  a + b + c + d = -2/3 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_negative_two_thirds_l3001_300180


namespace NUMINAMATH_CALUDE_algebraic_equation_proof_l3001_300109

theorem algebraic_equation_proof (a b c : ℝ) 
  (h1 : a^2 + b*c = 14) 
  (h2 : b^2 - 2*b*c = -6) : 
  3*a^2 + 4*b^2 - 5*b*c = 18 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equation_proof_l3001_300109


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3001_300124

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {1, 2}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3001_300124


namespace NUMINAMATH_CALUDE_second_discount_percentage_l3001_300103

theorem second_discount_percentage (initial_price : ℝ) 
  (first_discount second_discount third_discount final_price : ℝ) :
  initial_price = 12000 ∧ 
  first_discount = 20 ∧ 
  third_discount = 5 ∧
  final_price = 7752 ∧
  final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) * (1 - third_discount / 100) →
  second_discount = 15 := by
sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l3001_300103


namespace NUMINAMATH_CALUDE_can_capacity_proof_l3001_300171

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The capacity of the can in liters -/
def canCapacity : ℝ := 36

/-- The amount of milk added in liters -/
def milkAdded : ℝ := 8

theorem can_capacity_proof (initial : CanContents) (final : CanContents) :
  -- Initial ratio of milk to water is 4:3
  initial.milk / initial.water = 4 / 3 →
  -- Final contents after adding milk
  final.milk = initial.milk + milkAdded ∧
  final.water = initial.water →
  -- Can is full after adding milk
  final.milk + final.water = canCapacity →
  -- Final ratio of milk to water is 2:1
  final.milk / final.water = 2 / 1 →
  -- Prove that the capacity of the can is 36 liters
  canCapacity = 36 := by
  sorry


end NUMINAMATH_CALUDE_can_capacity_proof_l3001_300171


namespace NUMINAMATH_CALUDE_abc_sum_mod_11_l3001_300102

theorem abc_sum_mod_11 (a b c : ℕ) : 
  0 < a ∧ a < 11 ∧ 
  0 < b ∧ b < 11 ∧ 
  0 < c ∧ c < 11 ∧ 
  (a * b * c) % 11 = 3 ∧ 
  (8 * c) % 11 = 5 ∧ 
  (a + 3 * b) % 11 = 10 → 
  (a + b + c) % 11 = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_11_l3001_300102


namespace NUMINAMATH_CALUDE_regular_octagon_area_l3001_300143

/-- The area of a regular octagon with side length 2√2 is 16 + 16√2 -/
theorem regular_octagon_area : 
  let s : ℝ := 2 * Real.sqrt 2
  8 * (s^2 / (4 * Real.tan (π/8))) = 16 + 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_area_l3001_300143


namespace NUMINAMATH_CALUDE_binary_11101_is_29_l3001_300190

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldr (fun (i, b) acc => acc + if b then 2^i else 0) 0

/-- The binary representation of the number in question -/
def binary_number : List Bool := [true, false, true, true, true]

/-- Theorem stating that the decimal representation of 11101₂ is 29 -/
theorem binary_11101_is_29 : binary_to_decimal binary_number = 29 := by
  sorry

end NUMINAMATH_CALUDE_binary_11101_is_29_l3001_300190


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l3001_300157

theorem fractional_equation_positive_root (x m : ℝ) : 
  (2 / (x - 2) - (2 * x - m) / (2 - x) = 3) → 
  (x > 0) →
  (m = 6) := by
sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l3001_300157


namespace NUMINAMATH_CALUDE_set_union_problem_l3001_300131

theorem set_union_problem (a b : ℝ) :
  let M : Set ℝ := {a, b}
  let N : Set ℝ := {a + 1, 3}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l3001_300131


namespace NUMINAMATH_CALUDE_car_dealership_shipment_l3001_300178

theorem car_dealership_shipment 
  (initial_cars : ℕ) 
  (initial_silver_percent : ℚ)
  (new_shipment_nonsilver_percent : ℚ)
  (final_silver_percent : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percent = 1/5)
  (h3 : new_shipment_nonsilver_percent = 7/20)
  (h4 : final_silver_percent = 3/10)
  : ∃ (new_shipment : ℕ), 
    (initial_silver_percent * initial_cars + (1 - new_shipment_nonsilver_percent) * new_shipment) / 
    (initial_cars + new_shipment) = final_silver_percent ∧ 
    new_shipment = 11 :=
sorry

end NUMINAMATH_CALUDE_car_dealership_shipment_l3001_300178


namespace NUMINAMATH_CALUDE_pet_store_white_cats_l3001_300188

theorem pet_store_white_cats 
  (total : ℕ) 
  (black : ℕ) 
  (gray : ℕ) 
  (h1 : total = 15) 
  (h2 : black = 10) 
  (h3 : gray = 3) 
  (h4 : ∃ white : ℕ, total = white + black + gray) : 
  ∃ white : ℕ, white = 2 ∧ total = white + black + gray :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_white_cats_l3001_300188


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3001_300193

/-- Given two planar vectors a and b, prove that the cosine of the angle between them is -3/5. -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) : 
  a = (2, 4) → a - 2 • b = (0, 8) → 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = -3/5 := by
  sorry

#check cosine_of_angle_between_vectors

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3001_300193


namespace NUMINAMATH_CALUDE_stream_speed_stream_speed_is_24_l3001_300115

/-- Given a boat with speed in still water and the relationship between upstream and downstream times,
    calculate the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (upstream_time downstream_time : ℝ) : ℝ :=
  let stream_speed := (boat_speed : ℝ) / 3
  have h1 : upstream_time = 2 * downstream_time := by sorry
  have h2 : boat_speed = 72 := by sorry
  have h3 : upstream_time * (boat_speed - stream_speed) = downstream_time * (boat_speed + stream_speed) := by sorry
  stream_speed

/-- The speed of the stream is 24 kmph. -/
theorem stream_speed_is_24 : stream_speed 72 1 0.5 = 24 := by sorry

end NUMINAMATH_CALUDE_stream_speed_stream_speed_is_24_l3001_300115


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l3001_300176

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 1 = 1
  h4 : (a 1) * (a 5) = (a 2) ^ 2

/-- The nth term of the arithmetic sequence is 2n - 1 -/
theorem arithmetic_sequence_nth_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l3001_300176


namespace NUMINAMATH_CALUDE_extra_bananas_l3001_300196

/-- Given the total number of children, the number of absent children, and the planned distribution,
    prove that each present child received 2 extra bananas. -/
theorem extra_bananas (total_children absent_children planned_per_child : ℕ) 
  (h1 : total_children = 660)
  (h2 : absent_children = 330)
  (h3 : planned_per_child = 2) :
  let present_children := total_children - absent_children
  let total_bananas := total_children * planned_per_child
  let actual_per_child := total_bananas / present_children
  actual_per_child - planned_per_child = 2 := by
  sorry

end NUMINAMATH_CALUDE_extra_bananas_l3001_300196


namespace NUMINAMATH_CALUDE_first_term_of_constant_ratio_l3001_300186

def arithmetic_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a + (n - 1 : ℚ) * d) / 2

theorem first_term_of_constant_ratio (d : ℚ) (c : ℚ) :
  d = 5 →
  (∀ n : ℕ, n > 0 → 
    (arithmetic_sum a d (2*n)) / (arithmetic_sum a d n) = c) →
  a = 5/2 :=
sorry

end NUMINAMATH_CALUDE_first_term_of_constant_ratio_l3001_300186


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l3001_300199

theorem perfect_square_quadratic (c : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + 14*x + c = y^2) → c = 49 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l3001_300199


namespace NUMINAMATH_CALUDE_carol_weight_l3001_300117

/-- Given that Alice and Carol have a combined weight of 280 pounds,
    and the difference between Carol's and Alice's weights is one-third of Carol's weight,
    prove that Carol weighs 168 pounds. -/
theorem carol_weight (alice_weight carol_weight : ℝ) 
    (h1 : alice_weight + carol_weight = 280)
    (h2 : carol_weight - alice_weight = carol_weight / 3) :
    carol_weight = 168 := by
  sorry

end NUMINAMATH_CALUDE_carol_weight_l3001_300117


namespace NUMINAMATH_CALUDE_unique_color_for_X_l3001_300149

/-- Represents the four colors used in the grid --/
inductive Color
| Red
| Blue
| Yellow
| Green

/-- Represents a position in the grid --/
structure Position :=
  (x : ℕ)
  (y : ℕ)

/-- Represents the grid --/
def Grid := Position → Option Color

/-- Checks if two positions are adjacent (share a vertex) --/
def adjacent (p1 p2 : Position) : Prop :=
  (p1.x = p2.x ∧ (p1.y = p2.y + 1 ∨ p1.y = p2.y - 1)) ∨
  (p1.y = p2.y ∧ (p1.x = p2.x + 1 ∨ p1.x = p2.x - 1)) ∨
  (p1.x = p2.x + 1 ∧ p1.y = p2.y + 1) ∨
  (p1.x = p2.x - 1 ∧ p1.y = p2.y - 1) ∨
  (p1.x = p2.x + 1 ∧ p1.y = p2.y - 1) ∨
  (p1.x = p2.x - 1 ∧ p1.y = p2.y + 1)

/-- Checks if the grid coloring is valid --/
def valid_coloring (g : Grid) : Prop :=
  ∀ p1 p2 : Position, adjacent p1 p2 →
    (g p1).isSome ∧ (g p2).isSome →
    (g p1 ≠ g p2)

/-- The position of cell X --/
def X : Position := ⟨5, 5⟩

/-- Theorem: There exists a unique color for cell X in a valid 4-color grid --/
theorem unique_color_for_X (g : Grid) (h : valid_coloring g) :
  ∃! c : Color, g X = some c :=
sorry

end NUMINAMATH_CALUDE_unique_color_for_X_l3001_300149


namespace NUMINAMATH_CALUDE_value_of_expression_l3001_300172

theorem value_of_expression (x : ℝ) (h : x^2 - x = 1) : 1 + 2*x - 2*x^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3001_300172


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3001_300168

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 2*x₁ - 9 = 0) ∧ (x₂^2 - 2*x₂ - 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3001_300168


namespace NUMINAMATH_CALUDE_age_problem_l3001_300165

theorem age_problem (P R J M : ℕ) : 
  P = R / 2 →
  R + 12 = J + 12 + 7 →
  J + 12 = 3 * P →
  M + 8 = J + 8 + 9 →
  M + 4 = 2 * (R + 4) →
  P = 5 ∧ R = 10 ∧ J = 3 ∧ M = 24 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3001_300165


namespace NUMINAMATH_CALUDE_gcd_228_1995_base_conversion_l3001_300158

-- Problem 1: GCD of 228 and 1995
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by sorry

-- Problem 2: Base conversion
theorem base_conversion :
  (1 * 3^4 + 1 * 3^3 + 1 * 3^2 + 0 * 3^1 + 2 * 3^0) = (3 * 6^2 + 1 * 6^1 + 5 * 6^0) := by sorry

end NUMINAMATH_CALUDE_gcd_228_1995_base_conversion_l3001_300158


namespace NUMINAMATH_CALUDE_expression_evaluation_l3001_300126

theorem expression_evaluation :
  (15 + 12)^2 - (12^2 + 15^2 + 6 * 15 * 12) = -720 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3001_300126


namespace NUMINAMATH_CALUDE_sum_of_n_values_l3001_300148

theorem sum_of_n_values (n : ℤ) : 
  (∃ (S : Finset ℤ), (∀ m ∈ S, (∃ k : ℤ, 24 = k * (2 * m - 1))) ∧ 
   (∀ m : ℤ, (∃ k : ℤ, 24 = k * (2 * m - 1)) → m ∈ S) ∧ 
   (Finset.sum S id = 3)) := by
sorry

end NUMINAMATH_CALUDE_sum_of_n_values_l3001_300148


namespace NUMINAMATH_CALUDE_linear_function_properties_l3001_300107

/-- Linear function passing through (1, 0) and (0, 2) -/
def linear_function (x : ℝ) : ℝ := -2 * x + 2

theorem linear_function_properties :
  let f := linear_function
  -- The range of y is -4 ≤ y < 6 when -2 < x ≤ 3
  (∀ x : ℝ, -2 < x ∧ x ≤ 3 → -4 ≤ f x ∧ f x < 6) ∧
  -- The point P(m, n) satisfying m - n = 4 has coordinates (2, -2)
  (∃ m n : ℝ, f m = n ∧ m - n = 4 ∧ m = 2 ∧ n = -2) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l3001_300107


namespace NUMINAMATH_CALUDE_max_value_of_f_l3001_300173

def f (x : ℝ) : ℝ := -3 * x^2 + 8

theorem max_value_of_f :
  ∃ (M : ℝ), (∀ (x : ℝ), f x ≤ M) ∧ (∃ (x₀ : ℝ), f x₀ = M) ∧ M = 8 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3001_300173


namespace NUMINAMATH_CALUDE_angle_measure_from_coordinates_l3001_300174

/-- Given an acute angle α and a point A on its terminal side with coordinates (2sin 3, -2cos 3),
    prove that α = 3 - π/2 --/
theorem angle_measure_from_coordinates (α : Real) (A : Real × Real) :
  α > 0 ∧ α < π/2 →  -- α is acute
  A.1 = 2 * Real.sin 3 ∧ A.2 = -2 * Real.cos 3 →  -- coordinates of A
  α = 3 - π/2 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_from_coordinates_l3001_300174


namespace NUMINAMATH_CALUDE_infinitely_many_coprime_phi_m_root_l3001_300120

/-- m-th iteration of Euler's totient function -/
def phi_m (m : ℕ) : ℕ → ℕ :=
  match m with
  | 0 => id
  | m + 1 => phi_m m ∘ Nat.totient

/-- Main theorem -/
theorem infinitely_many_coprime_phi_m_root (a b m k : ℕ) (hk : k ≥ 2) :
  ∃ S : Set ℕ, S.Infinite ∧ ∀ n ∈ S, Nat.gcd (phi_m m n) (Nat.floor ((a * n + b : ℝ) ^ (1 / k))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_coprime_phi_m_root_l3001_300120


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3001_300136

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3001_300136


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l3001_300195

theorem sphere_surface_area_circumscribing_cube (edge_length : ℝ) (surface_area : ℝ) :
  edge_length = 2 →
  surface_area = 4 * Real.pi * (((edge_length ^ 2 + edge_length ^ 2 + edge_length ^ 2) / 4) : ℝ) →
  surface_area = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_cube_l3001_300195


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3001_300177

/-- Given a quadratic inequality ax² + bx + 1 > 0 with solution set {x | -1 < x < 1/3},
    prove that ab = 6 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + b * x + 1 > 0) ↔ (-1 < x ∧ x < 1/3)) → 
  a * b = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3001_300177


namespace NUMINAMATH_CALUDE_five_squared_sum_five_times_l3001_300113

theorem five_squared_sum_five_times : (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) = 125 := by
  sorry

end NUMINAMATH_CALUDE_five_squared_sum_five_times_l3001_300113


namespace NUMINAMATH_CALUDE_drill_bits_total_cost_l3001_300133

/-- Calculates the total amount paid for drill bits including tax -/
theorem drill_bits_total_cost (num_sets : ℕ) (cost_per_set : ℚ) (tax_rate : ℚ) : 
  num_sets = 5 → cost_per_set = 6 → tax_rate = (1/10) → 
  num_sets * cost_per_set * (1 + tax_rate) = 33 := by
  sorry

end NUMINAMATH_CALUDE_drill_bits_total_cost_l3001_300133


namespace NUMINAMATH_CALUDE_part_one_part_two_l3001_300110

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0
def q (x : ℝ) : Prop := 2 < x ∧ x ≤ 5

-- Part 1
theorem part_one (x : ℝ) : 
  p x 1 → q x → x ∈ Set.Ioo 2 4 :=
sorry

-- Part 2
theorem part_two (a : ℝ) : 
  a > 0 → 
  (Set.Ioo 2 5 ⊂ {x | p x a}) → 
  ({x | p x a} ≠ Set.Ioo 2 5) → 
  a ∈ Set.Ioc (5/4) 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3001_300110


namespace NUMINAMATH_CALUDE_unique_prime_in_form_l3001_300161

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

def number_form (A : ℕ) : ℕ := 303100 + A

theorem unique_prime_in_form :
  ∃! A : ℕ, A < 10 ∧ is_prime (number_form A) ∧ number_form A = 303103 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_in_form_l3001_300161


namespace NUMINAMATH_CALUDE_asymptote_angle_is_90_degrees_l3001_300135

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 (b > 0) and eccentricity √2 -/
structure Hyperbola where
  b : ℝ
  h_b_pos : b > 0
  h_ecc : Real.sqrt 2 = Real.sqrt (1 + 1 / b^2)

/-- The angle between the asymptotes of the hyperbola -/
def asymptote_angle (h : Hyperbola) : ℝ := sorry

/-- Theorem stating that the angle between the asymptotes is 90° -/
theorem asymptote_angle_is_90_degrees (h : Hyperbola) :
  asymptote_angle h = 90 * π / 180 := by sorry

end NUMINAMATH_CALUDE_asymptote_angle_is_90_degrees_l3001_300135


namespace NUMINAMATH_CALUDE_scientific_notation_eleven_million_l3001_300111

theorem scientific_notation_eleven_million :
  (11000000 : ℝ) = 1.1 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_eleven_million_l3001_300111


namespace NUMINAMATH_CALUDE_existence_of_divisible_m_l3001_300104

theorem existence_of_divisible_m : ∃ m : ℕ+, (3^100 * m.val + 3^100 - 1) % 1988 = 0 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_divisible_m_l3001_300104


namespace NUMINAMATH_CALUDE_power_of_54_l3001_300132

theorem power_of_54 (a b : ℕ+) (h : (54 : ℕ) ^ a.val = a.val ^ b.val) :
  ∃ k : ℕ, a.val = (54 : ℕ) ^ k := by
sorry

end NUMINAMATH_CALUDE_power_of_54_l3001_300132


namespace NUMINAMATH_CALUDE_negation_of_exists_ellipse_eccentricity_lt_one_l3001_300142

/-- An ellipse is a geometric shape with an eccentricity. -/
structure Ellipse where
  eccentricity : ℝ

/-- The negation of "There exists an ellipse with an eccentricity e < 1" 
    is equivalent to "The eccentricity e ≥ 1 for any ellipse". -/
theorem negation_of_exists_ellipse_eccentricity_lt_one :
  (¬ ∃ (e : Ellipse), e.eccentricity < 1) ↔ (∀ (e : Ellipse), e.eccentricity ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_negation_of_exists_ellipse_eccentricity_lt_one_l3001_300142


namespace NUMINAMATH_CALUDE_divisor_calculation_l3001_300162

theorem divisor_calculation (dividend quotient remainder : ℕ) (h1 : dividend = 76) (h2 : quotient = 4) (h3 : remainder = 8) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 17 := by
  sorry

end NUMINAMATH_CALUDE_divisor_calculation_l3001_300162


namespace NUMINAMATH_CALUDE_root_product_theorem_l3001_300100

theorem root_product_theorem (a b m p : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ∃ r, ((a + 1/b)^2 - p*(a + 1/b) + r = 0) ∧ 
       ((b + 1/a)^2 - p*(b + 1/a) + r = 0) → 
  r = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3001_300100


namespace NUMINAMATH_CALUDE_cats_not_liking_tuna_or_chicken_l3001_300106

theorem cats_not_liking_tuna_or_chicken 
  (total : ℕ) (tuna : ℕ) (chicken : ℕ) (both : ℕ) :
  total = 80 → tuna = 15 → chicken = 60 → both = 10 →
  total - (tuna + chicken - both) = 15 := by
sorry

end NUMINAMATH_CALUDE_cats_not_liking_tuna_or_chicken_l3001_300106


namespace NUMINAMATH_CALUDE_lena_time_to_counter_l3001_300129

/-- The time it takes Lena to reach the counter given her initial movement and remaining distance -/
theorem lena_time_to_counter (initial_distance : ℝ) (initial_time : ℝ) (remaining_distance_meters : ℝ) :
  initial_distance = 40 →
  initial_time = 20 →
  remaining_distance_meters = 100 →
  (remaining_distance_meters * 3.28084) / (initial_distance / initial_time) = 164.042 := by
sorry

end NUMINAMATH_CALUDE_lena_time_to_counter_l3001_300129


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3001_300189

-- Define the arithmetic sequence and its sum
def a (n : ℕ) : ℤ := 2*n - 9
def S (n : ℕ) : ℤ := n^2 - 8*n

-- State the theorem
theorem arithmetic_sequence_properties :
  (a 1 = -7) ∧ 
  (S 3 = -15) ∧ 
  (∀ n : ℕ, S n = n^2 - 8*n) ∧
  (∃ min : ℤ, min = -16 ∧ ∀ n : ℕ, S n ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3001_300189


namespace NUMINAMATH_CALUDE_prob_two_girls_prob_two_girls_five_l3001_300145

/-- The probability of selecting two girls from a club with equal numbers of boys and girls -/
theorem prob_two_girls (n : ℕ) (h : n > 0) : 
  (Nat.choose n 2) / (Nat.choose (2*n) 2) = 2 / 9 :=
sorry

/-- The specific case for a club with 5 girls and 5 boys -/
theorem prob_two_girls_five : 
  (Nat.choose 5 2) / (Nat.choose 10 2) = 2 / 9 :=
sorry

end NUMINAMATH_CALUDE_prob_two_girls_prob_two_girls_five_l3001_300145


namespace NUMINAMATH_CALUDE_even_operations_l3001_300153

-- Define an even integer
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define a perfect square
def is_perfect_square (n : ℤ) : Prop := ∃ k : ℤ, n = k * k

theorem even_operations (n : ℤ) (h : is_even n) :
  (is_even (n^2)) ∧ 
  (∀ m : ℤ, is_even m → is_perfect_square m → is_even (Int.sqrt m)) ∧
  (∀ k : ℤ, ¬(is_even k) → is_even (n * k)) ∧
  (is_even (n^3)) :=
sorry

end NUMINAMATH_CALUDE_even_operations_l3001_300153


namespace NUMINAMATH_CALUDE_younger_son_age_in_30_years_l3001_300185

/-- Given an elder son's age and the age difference between two sons, 
    calculate the younger son's age after a certain number of years. -/
def younger_son_future_age (elder_son_age : ℕ) (age_difference : ℕ) (years_from_now : ℕ) : ℕ :=
  (elder_son_age - age_difference) + years_from_now

theorem younger_son_age_in_30_years :
  younger_son_future_age 40 10 30 = 60 := by
  sorry

end NUMINAMATH_CALUDE_younger_son_age_in_30_years_l3001_300185


namespace NUMINAMATH_CALUDE_cos_30_minus_cos_60_l3001_300170

theorem cos_30_minus_cos_60 : Real.cos (π / 6) - Real.cos (π / 3) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_30_minus_cos_60_l3001_300170


namespace NUMINAMATH_CALUDE_simplify_fraction_l3001_300139

theorem simplify_fraction : (54 : ℚ) / 972 = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3001_300139


namespace NUMINAMATH_CALUDE_xyz_negative_l3001_300141

theorem xyz_negative (a b c x y z : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0)
  (h : |x - a| + |y - b| + |z - c| = 0) : 
  x * y * z < 0 := by
sorry

end NUMINAMATH_CALUDE_xyz_negative_l3001_300141


namespace NUMINAMATH_CALUDE_coin_weighing_possible_l3001_300112

/-- Represents a coin with a weight -/
structure Coin where
  weight : ℕ

/-- Represents the result of a weighing operation -/
inductive WeighResult
  | Equal
  | LeftHeavier
  | RightHeavier

/-- Represents a 2-pan scale that can compare two sets of coins -/
def Scale : (List Coin) → (List Coin) → WeighResult := sorry

/-- The main theorem to be proven -/
theorem coin_weighing_possible (k : ℕ) : 
  ∀ (coins : List Coin), 
    coins.length = 2^k → 
    (∀ a b c : Coin, a ∈ coins → b ∈ coins → c ∈ coins → 
      (a.weight ≠ b.weight ∧ b.weight ≠ c.weight) → a.weight = c.weight) →
    ∃ (measurements : List (List Coin × List Coin)),
      measurements.length ≤ k ∧
      (∀ m ∈ measurements, m.1.length + m.2.length ≤ coins.length) ∧
      (∃ (heavy light : Coin), heavy ∈ coins ∧ light ∈ coins ∧ heavy.weight > light.weight) ∨
      (∀ c1 c2 : Coin, c1 ∈ coins → c2 ∈ coins → c1.weight = c2.weight) := by
  sorry

end NUMINAMATH_CALUDE_coin_weighing_possible_l3001_300112


namespace NUMINAMATH_CALUDE_min_points_on_circle_l3001_300137

theorem min_points_on_circle (n : ℕ) (h : n ≥ 3) :
  let N := if (2*n - 1) % 3 = 0 then n else n - 1
  ∀ (S : Finset (Fin (2*n - 1))),
    S.card ≥ N →
    ∃ (i j : Fin (2*n - 1)), i ∈ S ∧ j ∈ S ∧
      (((j - i : ℤ) + (2*n - 1)) % (2*n - 1) = n ∨
       ((i - j : ℤ) + (2*n - 1)) % (2*n - 1) = n) :=
by sorry

end NUMINAMATH_CALUDE_min_points_on_circle_l3001_300137


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l3001_300197

/-- A line parallel to y = -3x + 6 passing through (3, -2) has y-intercept 7 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b y = -3 * x + b 0) →  -- b is a line with slope -3
  b (-2) = 3 * (-3) + b 0 →     -- b passes through (3, -2)
  b 0 = 7 :=                    -- y-intercept of b is 7
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l3001_300197


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_four_is_five_sixths_l3001_300183

/-- The number of possible outcomes when tossing two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is 4 or less -/
def outcomes_sum_4_or_less : ℕ := 6

/-- The probability of getting a sum greater than four when tossing two dice -/
def prob_sum_greater_than_four : ℚ := 5 / 6

theorem prob_sum_greater_than_four_is_five_sixths :
  prob_sum_greater_than_four = 1 - (outcomes_sum_4_or_less : ℚ) / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_four_is_five_sixths_l3001_300183


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l3001_300130

def is_divisible_by_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 0 → n % d = 0

theorem largest_three_digit_divisible_by_digits :
  ∀ n : ℕ, 800 ≤ n → n ≤ 899 → is_divisible_by_digits n →
  n ≤ 864 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_l3001_300130


namespace NUMINAMATH_CALUDE_snow_cover_probabilities_l3001_300125

theorem snow_cover_probabilities (p : ℝ) (h : p = 0.2) :
  let q := 1 - p
  (q^2 = 0.64) ∧ (1 - q^2 = 0.36) := by
  sorry

end NUMINAMATH_CALUDE_snow_cover_probabilities_l3001_300125


namespace NUMINAMATH_CALUDE_faster_train_speed_l3001_300101

/-- Given two trains moving in the same direction, prove that the speed of the faster train is 90 kmph -/
theorem faster_train_speed 
  (speed_difference : ℝ) 
  (faster_train_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : speed_difference = 36) 
  (h2 : faster_train_length = 435) 
  (h3 : crossing_time = 29) : 
  ∃ (faster_speed slower_speed : ℝ), 
    faster_speed - slower_speed = speed_difference ∧ 
    faster_train_length / crossing_time * 3.6 = speed_difference ∧
    faster_speed = 90 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_l3001_300101


namespace NUMINAMATH_CALUDE_custom_op_example_l3001_300179

-- Define the custom operation ※
def custom_op (a b : ℚ) : ℚ := 4 * b - a

-- Theorem statement
theorem custom_op_example : custom_op (custom_op (-1) 3) 2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l3001_300179


namespace NUMINAMATH_CALUDE_novel_pages_prove_novel_pages_l3001_300138

theorem novel_pages : ℕ → Prop :=
  fun total_pages =>
    let day1_read := total_pages / 6 + 10
    let day1_remaining := total_pages - day1_read
    let day2_read := day1_remaining / 5 + 20
    let day2_remaining := day1_remaining - day2_read
    let day3_read := day2_remaining / 4 + 25
    let day3_remaining := day2_remaining - day3_read
    day3_remaining = 130 → total_pages = 352

theorem prove_novel_pages : novel_pages 352 := by
  sorry

end NUMINAMATH_CALUDE_novel_pages_prove_novel_pages_l3001_300138


namespace NUMINAMATH_CALUDE_orange_juice_percentage_l3001_300184

theorem orange_juice_percentage
  (total_volume : ℝ)
  (watermelon_percentage : ℝ)
  (grape_volume : ℝ)
  (h1 : total_volume = 300)
  (h2 : watermelon_percentage = 40)
  (h3 : grape_volume = 105) :
  (total_volume - watermelon_percentage / 100 * total_volume - grape_volume) / total_volume * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_percentage_l3001_300184


namespace NUMINAMATH_CALUDE_rectangle_width_l3001_300175

/-- 
Given a rectangle where:
  - The length is 3 cm shorter than the width
  - The perimeter is 54 cm
Prove that the width of the rectangle is 15 cm
-/
theorem rectangle_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = width - 3 →
  perimeter = 54 →
  perimeter = 2 * width + 2 * length →
  width = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l3001_300175


namespace NUMINAMATH_CALUDE_line_symmetry_l3001_300116

-- Define a line by its coefficients a, b, and c in the equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define symmetry with respect to x = 1
def symmetricAboutX1 (l1 l2 : Line) : Prop :=
  ∀ x y : ℝ, l1.a * (2 - x) + l1.b * y + l1.c = 0 ↔ l2.a * x + l2.b * y + l2.c = 0

-- Theorem statement
theorem line_symmetry (l1 l2 : Line) :
  l1 = Line.mk 3 (-4) (-3) →
  symmetricAboutX1 l1 l2 →
  l2 = Line.mk 3 4 (-3) := by
  sorry

end NUMINAMATH_CALUDE_line_symmetry_l3001_300116


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3001_300119

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 7 * x + 20 = 0 ↔ x = p + q * I ∨ x = p - q * I) → 
  p + q^2 = 421 / 100 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3001_300119


namespace NUMINAMATH_CALUDE_smallest_natural_number_divisibility_l3001_300181

theorem smallest_natural_number_divisibility : ∃! n : ℕ,
  (∀ m : ℕ, m < n → ¬((m + 2018) % 2020 = 0 ∧ (m + 2020) % 2018 = 0)) ∧
  (n + 2018) % 2020 = 0 ∧
  (n + 2020) % 2018 = 0 ∧
  n = 2030102 := by
  sorry

end NUMINAMATH_CALUDE_smallest_natural_number_divisibility_l3001_300181


namespace NUMINAMATH_CALUDE_circle_inequality_l3001_300191

theorem circle_inequality (c : ℝ) : 
  (∀ x y : ℝ, x^2 + (y-1)^2 = 1 → x + y + c ≥ 0) → 
  c ≥ Real.sqrt 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_circle_inequality_l3001_300191


namespace NUMINAMATH_CALUDE_probability_estimate_l3001_300152

def is_hit (d : Nat) : Bool := d ≥ 2 ∧ d ≤ 9

def group_has_three_hits (g : List Nat) : Bool :=
  (g.filter is_hit).length ≥ 3

def count_successful_groups (groups : List (List Nat)) : Nat :=
  (groups.filter group_has_three_hits).length

theorem probability_estimate (groups : List (List Nat))
  (h1 : groups.length = 20)
  (h2 : ∀ g ∈ groups, g.length = 4)
  (h3 : ∀ g ∈ groups, ∀ d ∈ g, d ≤ 9)
  (h4 : count_successful_groups groups = 15) :
  (count_successful_groups groups : ℚ) / groups.length = 3/4 := by
  sorry

#check probability_estimate

end NUMINAMATH_CALUDE_probability_estimate_l3001_300152


namespace NUMINAMATH_CALUDE_both_false_sufficient_not_necessary_l3001_300121

-- Define simple propositions a and b
variable (a b : Prop)

-- Define the statements
def both_false : Prop := ¬a ∧ ¬b
def either_false : Prop := ¬a ∨ ¬b

-- Theorem statement
theorem both_false_sufficient_not_necessary :
  (both_false a b → either_false a b) ∧
  ¬(either_false a b → both_false a b) :=
sorry

end NUMINAMATH_CALUDE_both_false_sufficient_not_necessary_l3001_300121


namespace NUMINAMATH_CALUDE_complex_magnitude_l3001_300114

theorem complex_magnitude (z : ℂ) (h : (z - Complex.I) * (2 - Complex.I) = Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3001_300114


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l3001_300144

-- Define the function f(x) = x³ - 22 - x
def f (x : ℝ) := x^3 - 22 - x

-- Theorem statement
theorem root_exists_in_interval :
  ∃ x₀ ∈ Set.Ioo 1 2, f x₀ = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_root_exists_in_interval_l3001_300144


namespace NUMINAMATH_CALUDE_add_average_score_theorem_singing_competition_scores_l3001_300146

/-- Represents a set of scores with their statistical properties -/
structure ScoreSet where
  count : ℕ
  average : ℝ
  variance : ℝ

/-- Represents the result after adding a new score -/
structure NewScoreSet where
  new_average : ℝ
  new_variance : ℝ

/-- 
Given a set of scores and a new score, calculates the new average and variance
-/
def add_score (scores : ScoreSet) (new_score : ℝ) : NewScoreSet :=
  sorry

/-- 
Theorem: Adding a score equal to the original average keeps the average the same
and reduces the variance
-/
theorem add_average_score_theorem (scores : ScoreSet) :
  let new_set := add_score scores scores.average
  new_set.new_average = scores.average ∧ new_set.new_variance < scores.variance :=
  sorry

/-- 
Application of the theorem to the specific problem
-/
theorem singing_competition_scores :
  let original_scores : ScoreSet := ⟨8, 5, 3⟩
  let new_set := add_score original_scores 5
  new_set.new_average = 5 ∧ new_set.new_variance < 3 :=
  sorry

end NUMINAMATH_CALUDE_add_average_score_theorem_singing_competition_scores_l3001_300146


namespace NUMINAMATH_CALUDE_kaleb_first_half_score_l3001_300123

/-- Calculates the first half score in a trivia game given the total score and second half score. -/
def first_half_score (total_score second_half_score : ℕ) : ℕ :=
  total_score - second_half_score

/-- Proves that Kaleb's first half score is 43 points given his total score of 66 and second half score of 23. -/
theorem kaleb_first_half_score :
  first_half_score 66 23 = 43 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_first_half_score_l3001_300123


namespace NUMINAMATH_CALUDE_configurations_eq_choose_l3001_300198

/-- The number of k-configurations with m elements from a set of n elements -/
def num_configurations (n k m : ℕ) : ℕ := Nat.choose n k

/-- Theorem stating that the number of k-configurations with m elements 
    from a set of n elements is equal to the binomial coefficient -/
theorem configurations_eq_choose (n k m : ℕ) : 
  num_configurations n k m = Nat.choose n k := by
  sorry

end NUMINAMATH_CALUDE_configurations_eq_choose_l3001_300198


namespace NUMINAMATH_CALUDE_ball_hitting_ground_time_l3001_300105

theorem ball_hitting_ground_time :
  let height (t : ℝ) := -16 * t^2 + 16 * t + 50
  ∃ t : ℝ, t > 0 ∧ height t = 0 ∧ t = (2 + 3 * Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ball_hitting_ground_time_l3001_300105
