import Mathlib

namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l2249_224997

/-- Calculates the total number of people who can ride a Ferris wheel -/
theorem ferris_wheel_capacity 
  (capacity : ℕ)           -- Number of people per ride
  (ride_duration : ℕ)      -- Duration of one ride in minutes
  (operation_time : ℕ) :   -- Total operation time in hours
  capacity * (60 / ride_duration) * operation_time = 1260 :=
by
  sorry

#check ferris_wheel_capacity 70 20 6

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l2249_224997


namespace NUMINAMATH_CALUDE_cantor_set_cardinality_cantor_set_operations_l2249_224929

-- Define the Cantor set
def CantorSet : Set ℝ := sorry

-- Theorem for part (a)
theorem cantor_set_cardinality : Cardinal.mk CantorSet = Cardinal.mk (Set.Icc 0 1) := by sorry

-- Define the sum and difference operations on sets
def setSum (A B : Set ℝ) : Set ℝ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a + b}
def setDiff (A B : Set ℝ) : Set ℝ := {x | ∃ a ∈ A, ∃ b ∈ B, x = a - b}

-- Theorem for part (b)
theorem cantor_set_operations :
  (setSum CantorSet CantorSet = Set.Icc 0 2) ∧
  (setDiff CantorSet CantorSet = Set.Icc (-1) 1) := by sorry

end NUMINAMATH_CALUDE_cantor_set_cardinality_cantor_set_operations_l2249_224929


namespace NUMINAMATH_CALUDE_electronics_store_cost_l2249_224908

/-- Given the cost of 5 MP3 players and 8 headphones is $840, and the cost of one set of headphones
is $30, prove that the cost of 3 MP3 players and 4 headphones is $480. -/
theorem electronics_store_cost (mp3_cost headphones_cost : ℕ) : 
  5 * mp3_cost + 8 * headphones_cost = 840 →
  headphones_cost = 30 →
  3 * mp3_cost + 4 * headphones_cost = 480 := by
  sorry

end NUMINAMATH_CALUDE_electronics_store_cost_l2249_224908


namespace NUMINAMATH_CALUDE_cloth_trimming_l2249_224998

theorem cloth_trimming (x : ℝ) :
  x > 0 →
  (x - 6) * (x - 5) = 120 →
  x = 15 :=
by sorry

end NUMINAMATH_CALUDE_cloth_trimming_l2249_224998


namespace NUMINAMATH_CALUDE_perpendicular_to_same_plane_implies_parallel_l2249_224971

-- Define a plane in 3D space
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define a line in 3D space
def Line : Type := ℝ → ℝ × ℝ × ℝ

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem perpendicular_to_same_plane_implies_parallel 
  (l1 l2 : Line) (p : Plane) 
  (h1 : perpendicular l1 p) (h2 : perpendicular l2 p) : 
  parallel l1 l2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_to_same_plane_implies_parallel_l2249_224971


namespace NUMINAMATH_CALUDE_maria_candy_eaten_l2249_224988

/-- Given that Maria initially had 67 pieces of candy and now has 3 pieces,
    prove that she ate 64 pieces of candy. -/
theorem maria_candy_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 67 → remaining = 3 → eaten = initial - remaining → eaten = 64 := by
  sorry

end NUMINAMATH_CALUDE_maria_candy_eaten_l2249_224988


namespace NUMINAMATH_CALUDE_composite_and_prime_divisors_l2249_224920

/-- Given two distinct positive integers a and b where a, b > 1, and s_n = a^n + b^(n+1) -/
theorem composite_and_prime_divisors (a b : ℕ) (ha : a > 1) (hb : b > 1) (hab : a ≠ b) :
  let s : ℕ → ℕ := fun n => a^n + b^(n+1)
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, ¬ Nat.Prime (s n)) ∧
  (∃ (P : Set ℕ), Set.Infinite P ∧ ∀ p ∈ P, Nat.Prime p ∧ ∃ n, p ∣ s n) := by
  sorry

end NUMINAMATH_CALUDE_composite_and_prime_divisors_l2249_224920


namespace NUMINAMATH_CALUDE_horner_method_multiplications_for_degree_5_l2249_224916

def horner_multiplications (n : ℕ) : ℕ := n

theorem horner_method_multiplications_for_degree_5 :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  let f : ℝ → ℝ := λ x => a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀
  horner_multiplications 5 = 5 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_multiplications_for_degree_5_l2249_224916


namespace NUMINAMATH_CALUDE_larger_integer_is_nine_l2249_224906

theorem larger_integer_is_nine (x y : ℤ) (h_product : x * y = 36) (h_sum : x + y = 13) :
  max x y = 9 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_is_nine_l2249_224906


namespace NUMINAMATH_CALUDE_water_evaporation_per_day_l2249_224983

/-- Given a glass of water with initial amount, evaporation period, and total evaporation percentage,
    calculate the amount of water evaporated per day. -/
theorem water_evaporation_per_day 
  (initial_amount : ℝ) 
  (evaporation_period : ℕ) 
  (evaporation_percentage : ℝ) 
  (h1 : initial_amount = 10)
  (h2 : evaporation_period = 20)
  (h3 : evaporation_percentage = 4) : 
  (initial_amount * evaporation_percentage / 100) / evaporation_period = 0.02 := by
  sorry

#check water_evaporation_per_day

end NUMINAMATH_CALUDE_water_evaporation_per_day_l2249_224983


namespace NUMINAMATH_CALUDE_existence_of_constant_l2249_224993

theorem existence_of_constant : ∃ c : ℝ, c > 0 ∧
  ∀ a b n : ℕ, a > 0 → b > 0 → n > 0 →
  (∀ i j : ℕ, i ≤ n → j ≤ n → Nat.gcd (a + i) (b + j) > 1) →
  min a b > (c * n : ℝ) ^ (n / 2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_constant_l2249_224993


namespace NUMINAMATH_CALUDE_square_area_15m_l2249_224919

theorem square_area_15m (side_length : ℝ) (h : side_length = 15) : 
  side_length * side_length = 225 := by
sorry

end NUMINAMATH_CALUDE_square_area_15m_l2249_224919


namespace NUMINAMATH_CALUDE_uniqueRootIff_l2249_224939

/-- A function that represents the quadratic equation ax^2 + (a-3)x + 1 --/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

/-- Predicate that determines if the graph of f(a, x) intersects the x-axis at only one point --/
def hasUniqueRoot (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem stating that f(a, x) has a unique root if and only if a is 0, 1, or 9 --/
theorem uniqueRootIff (a : ℝ) : hasUniqueRoot a ↔ a = 0 ∨ a = 1 ∨ a = 9 := by
  sorry

end NUMINAMATH_CALUDE_uniqueRootIff_l2249_224939


namespace NUMINAMATH_CALUDE_intersection_sum_l2249_224953

theorem intersection_sum : ∃ (x₁ x₂ : ℝ),
  (x₁^2 = 2*x₁ + 3) ∧
  (x₂^2 = 2*x₂ + 3) ∧
  (x₁ ≠ x₂) ∧
  (x₁ + x₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_sum_l2249_224953


namespace NUMINAMATH_CALUDE_line_angle_theorem_l2249_224940

/-- Given a line with equation (√6 sin θ)x + √3y - 2 = 0 and oblique angle θ ≠ 0, prove θ = 3π/4 -/
theorem line_angle_theorem (θ : Real) (h1 : θ ≠ 0) :
  (∃ (x y : Real), (Real.sqrt 6 * Real.sin θ) * x + Real.sqrt 3 * y - 2 = 0) →
  (∀ (x y : Real), (Real.sqrt 6 * Real.sin θ) * x + Real.sqrt 3 * y - 2 = 0 →
    Real.tan θ = -(Real.sqrt 6 / Real.sqrt 3) * Real.sin θ) →
  θ = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_line_angle_theorem_l2249_224940


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2249_224999

/-- An arithmetic progression with first three terms x - 1, x + 1, and 2x + 3 -/
def arithmetic_progression (x : ℝ) : ℕ → ℝ
| 0 => x - 1
| 1 => x + 1
| 2 => 2*x + 3
| _ => 0  -- We only care about the first three terms

/-- The common difference of the arithmetic progression -/
def common_difference (x : ℝ) : ℝ := arithmetic_progression x 1 - arithmetic_progression x 0

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, 
  (arithmetic_progression x 1 - arithmetic_progression x 0 = common_difference x) ∧
  (arithmetic_progression x 2 - arithmetic_progression x 1 = common_difference x) →
  x = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2249_224999


namespace NUMINAMATH_CALUDE_cost_of_500_pieces_is_10_dollars_l2249_224900

/-- The cost of 500 pieces of gum in dollars -/
def cost_of_500_pieces : ℚ := 10

/-- The cost of 1 piece of gum in cents -/
def cost_per_piece : ℕ := 2

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Theorem: The cost of 500 pieces of gum is 10 dollars -/
theorem cost_of_500_pieces_is_10_dollars :
  cost_of_500_pieces = (500 * cost_per_piece : ℚ) / cents_per_dollar := by
  sorry

end NUMINAMATH_CALUDE_cost_of_500_pieces_is_10_dollars_l2249_224900


namespace NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l2249_224974

theorem least_positive_integer_with_given_remainders : ∃ N : ℕ+,
  (N : ℤ) ≡ 3 [ZMOD 4] ∧
  (N : ℤ) ≡ 4 [ZMOD 5] ∧
  (N : ℤ) ≡ 5 [ZMOD 6] ∧
  (N : ℤ) ≡ 6 [ZMOD 7] ∧
  (N : ℤ) ≡ 10 [ZMOD 11] ∧
  (∀ m : ℕ+, m < N →
    ¬((m : ℤ) ≡ 3 [ZMOD 4] ∧
      (m : ℤ) ≡ 4 [ZMOD 5] ∧
      (m : ℤ) ≡ 5 [ZMOD 6] ∧
      (m : ℤ) ≡ 6 [ZMOD 7] ∧
      (m : ℤ) ≡ 10 [ZMOD 11])) ∧
  N = 4619 :=
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_given_remainders_l2249_224974


namespace NUMINAMATH_CALUDE_cos_alpha_minus_beta_l2249_224973

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : 2 * Real.cos α - Real.cos β = (3 : ℝ) / 2)
  (h2 : 2 * Real.sin α - Real.sin β = 2) :
  Real.cos (α - β) = -(5 : ℝ) / 16 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_beta_l2249_224973


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l2249_224969

/-- The percentage of water in fresh grapes by weight -/
def water_percentage_fresh : ℝ := 90

/-- The percentage of water in dried grapes by weight -/
def water_percentage_dried : ℝ := 20

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 20

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 2.5

theorem water_percentage_in_fresh_grapes :
  water_percentage_fresh = 90 ∧
  (100 - water_percentage_fresh) / 100 * fresh_weight = 
  (100 - water_percentage_dried) / 100 * dried_weight :=
by sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l2249_224969


namespace NUMINAMATH_CALUDE_intersection_A_B_l2249_224921

-- Define set A
def A : Set ℝ := {x | 0 < x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | (x + 1) / (x - 4) ≤ 0}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2249_224921


namespace NUMINAMATH_CALUDE_power_two_gt_sum_powers_l2249_224930

theorem power_two_gt_sum_powers (n : ℕ) (x : ℝ) (h1 : n ≥ 2) (h2 : |x| < 1) :
  2^n > (1 - x)^n + (1 + x)^n := by
  sorry

end NUMINAMATH_CALUDE_power_two_gt_sum_powers_l2249_224930


namespace NUMINAMATH_CALUDE_divisible_by_nine_unique_uphill_divisible_by_nine_l2249_224910

/-- An uphill integer is a positive integer where each digit is strictly greater than the previous digit. -/
def is_uphill (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

/-- A number is divisible by 9 if and only if the sum of its digits is divisible by 9. -/
theorem divisible_by_nine (n : ℕ) : n % 9 = 0 ↔ (n.digits 10).sum % 9 = 0 :=
sorry

/-- There is exactly one uphill integer divisible by 9. -/
theorem unique_uphill_divisible_by_nine : ∃! n : ℕ, is_uphill n ∧ n % 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_nine_unique_uphill_divisible_by_nine_l2249_224910


namespace NUMINAMATH_CALUDE_total_area_form_and_sum_l2249_224968

/-- Represents a rectangular prism with dimensions 1 × 1 × 2 -/
structure RectangularPrism :=
  (length : ℝ := 1)
  (width : ℝ := 1)
  (height : ℝ := 2)

/-- Represents a triangle with vertices from the rectangular prism -/
structure PrismTriangle :=
  (vertices : Fin 3 → Fin 8)

/-- Calculates the area of a PrismTriangle -/
def triangleArea (prism : RectangularPrism) (triangle : PrismTriangle) : ℝ :=
  sorry

/-- The sum of areas of all triangles whose vertices are vertices of the prism -/
def totalTriangleArea (prism : RectangularPrism) : ℝ :=
  sorry

/-- Theorem stating the form of the total area and the sum of m, n, and p -/
theorem total_area_form_and_sum (prism : RectangularPrism) :
  ∃ (m n p : ℕ), totalTriangleArea prism = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 100 :=
sorry

end NUMINAMATH_CALUDE_total_area_form_and_sum_l2249_224968


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2249_224951

/-- Given a rectangle with area 800 cm² and length twice its width, prove its perimeter is 120 cm. -/
theorem rectangle_perimeter (width : ℝ) (length : ℝ) :
  width > 0 →
  length > 0 →
  length = 2 * width →
  width * length = 800 →
  2 * (width + length) = 120 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2249_224951


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l2249_224917

theorem abs_sum_minimum (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 7 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 7 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l2249_224917


namespace NUMINAMATH_CALUDE_bella_galya_distance_l2249_224915

/-- The distance between two houses -/
def distance (house1 house2 : ℕ) : ℕ := sorry

/-- The order of houses along the road -/
def house_order : List String := ["Alya", "Bella", "Valya", "Galya", "Dilya"]

/-- The total distance from a house to all other houses -/
def total_distance (house : String) : ℕ := sorry

theorem bella_galya_distance :
  distance 1 3 = 150 ∧
  house_order = ["Alya", "Bella", "Valya", "Galya", "Dilya"] ∧
  total_distance "Bella" = 700 ∧
  total_distance "Valya" = 600 ∧
  total_distance "Galya" = 650 :=
by sorry

end NUMINAMATH_CALUDE_bella_galya_distance_l2249_224915


namespace NUMINAMATH_CALUDE_minimum_fare_increase_l2249_224980

/-- Represents the fare structure for a taxi service -/
structure FareStructure where
  n : ℝ  -- Total number of passengers
  t : ℝ  -- Base fare
  X : ℝ  -- Fare increase for businessmen

/-- Calculates the total revenue under the given fare structure -/
def totalRevenue (f : FareStructure) : ℝ :=
  0.75 * f.n * f.t + 0.2 * f.n * (f.t + f.X)

/-- Theorem stating the minimum fare increase that doesn't decrease total revenue -/
theorem minimum_fare_increase (f : FareStructure) :
  (∀ X : ℝ, totalRevenue { n := f.n, t := f.t, X := X } ≥ f.n * f.t → X ≥ f.t / 4) ∧
  totalRevenue { n := f.n, t := f.t, X := f.t / 4 } ≥ f.n * f.t :=
by sorry

end NUMINAMATH_CALUDE_minimum_fare_increase_l2249_224980


namespace NUMINAMATH_CALUDE_parking_lot_cars_parking_lot_problem_l2249_224944

theorem parking_lot_cars (total_wheels : ℕ) (num_bikes : ℕ) : ℕ :=
  let car_wheels := 4
  let bike_wheels := 2
  let num_cars := (total_wheels - num_bikes * bike_wheels) / car_wheels
  num_cars

theorem parking_lot_problem :
  parking_lot_cars 44 2 = 10 := by sorry

end NUMINAMATH_CALUDE_parking_lot_cars_parking_lot_problem_l2249_224944


namespace NUMINAMATH_CALUDE_pin_purchase_cost_l2249_224991

/-- The total cost of pins with a discount -/
def total_cost (num_pins : ℕ) (regular_price : ℚ) (discount_percent : ℚ) : ℚ :=
  num_pins * (regular_price * (1 - discount_percent / 100))

/-- Theorem stating the total cost of 10 pins with a 15% discount -/
theorem pin_purchase_cost :
  total_cost 10 20 15 = 170 := by
  sorry

end NUMINAMATH_CALUDE_pin_purchase_cost_l2249_224991


namespace NUMINAMATH_CALUDE_min_value_theorem_l2249_224982

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 1/x' + 1/y' = 1 → 
    1/(x' - 1) + 4/(y' - 1) ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2249_224982


namespace NUMINAMATH_CALUDE_oxford_high_school_population_l2249_224937

/-- The total number of people in Oxford High School -/
def total_people (teachers : ℕ) (principal : ℕ) (classes : ℕ) (students_per_class : ℕ) : ℕ :=
  teachers + principal + (classes * students_per_class)

/-- Theorem: The total number of people in Oxford High School is 349 -/
theorem oxford_high_school_population :
  total_people 48 1 15 20 = 349 := by
  sorry

end NUMINAMATH_CALUDE_oxford_high_school_population_l2249_224937


namespace NUMINAMATH_CALUDE_equation_solution_l2249_224933

theorem equation_solution :
  ∃ y : ℝ, y ≠ 2 ∧ (7 * y / (y - 2) - 5 / (y - 2) = 2 / (y - 2)) ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2249_224933


namespace NUMINAMATH_CALUDE_dogwood_trees_theorem_l2249_224922

def dogwood_trees_problem (initial_trees : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) : ℕ :=
  initial_trees + planted_today + planted_tomorrow

theorem dogwood_trees_theorem (initial_trees planted_today planted_tomorrow : ℕ) :
  dogwood_trees_problem initial_trees planted_today planted_tomorrow =
  initial_trees + planted_today + planted_tomorrow :=
by
  sorry

#eval dogwood_trees_problem 7 5 4

end NUMINAMATH_CALUDE_dogwood_trees_theorem_l2249_224922


namespace NUMINAMATH_CALUDE_square_property_of_natural_numbers_l2249_224918

theorem square_property_of_natural_numbers (a b : ℕ) 
  (h1 : ∃ k : ℕ, a * b = k^2)
  (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m^2) :
  ∃ n : ℕ, 
    2 < n ∧ 
    Even n ∧ 
    ∃ p : ℕ, (a + n) * (b + n) = p^2 := by
  sorry

end NUMINAMATH_CALUDE_square_property_of_natural_numbers_l2249_224918


namespace NUMINAMATH_CALUDE_garrett_granola_bars_l2249_224962

/-- The number of oatmeal raisin granola bars Garrett bought -/
def oatmeal_raisin_bars : ℕ := 6

/-- The number of peanut granola bars Garrett bought -/
def peanut_bars : ℕ := 8

/-- The total number of granola bars Garrett bought -/
def total_bars : ℕ := oatmeal_raisin_bars + peanut_bars

theorem garrett_granola_bars : total_bars = 14 := by sorry

end NUMINAMATH_CALUDE_garrett_granola_bars_l2249_224962


namespace NUMINAMATH_CALUDE_newspaper_spend_l2249_224992

/-- The cost of a weekday newspaper edition -/
def weekday_cost : ℚ := 0.50

/-- The cost of a Sunday newspaper edition -/
def sunday_cost : ℚ := 2.00

/-- The number of weekday editions Hillary buys per week -/
def weekday_editions : ℕ := 3

/-- The number of weeks -/
def weeks : ℕ := 8

/-- Hillary's total newspaper spend over 8 weeks -/
def total_spend : ℚ := weeks * (weekday_editions * weekday_cost + sunday_cost)

theorem newspaper_spend : total_spend = 28 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_spend_l2249_224992


namespace NUMINAMATH_CALUDE_adoption_time_proof_l2249_224947

/-- The number of days required to adopt all puppies -/
def adoptionDays (initialPuppies : ℕ) (additionalPuppies : ℕ) (adoptionRate : ℕ) : ℕ :=
  (initialPuppies + additionalPuppies) / adoptionRate

/-- Theorem stating that it takes 11 days to adopt all puppies under given conditions -/
theorem adoption_time_proof :
  adoptionDays 15 62 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_adoption_time_proof_l2249_224947


namespace NUMINAMATH_CALUDE_least_distinct_values_l2249_224975

/-- Given a list of 2023 positive integers with a unique mode occurring 15 times,
    the least number of distinct values is 145. -/
theorem least_distinct_values (l : List ℕ+) (h1 : l.length = 2023) 
    (h2 : ∃! m, m ∈ l ∧ l.count m = 15) : 
    (∃ (s : Finset ℕ+), s.card = 145 ∧ ∀ x ∈ l, x ∈ s) ∧ 
    (∀ (s : Finset ℕ+), (∀ x ∈ l, x ∈ s) → s.card ≥ 145) :=
sorry

end NUMINAMATH_CALUDE_least_distinct_values_l2249_224975


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2249_224938

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 1, a^2 + 4}
  A ∩ B = {3} → a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2249_224938


namespace NUMINAMATH_CALUDE_flatbread_diameters_exist_l2249_224904

/-- The diameter of the skillet -/
def skillet_diameter : ℕ := 26

/-- Predicate to check if three positive integers satisfy the required conditions -/
def valid_diameters (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x + y + z = skillet_diameter ∧
  x^2 + y^2 + z^2 = 338 ∧
  (x^2 + y^2 + z^2 : ℚ) / 4 = (skillet_diameter^2 : ℚ) / 8

/-- Theorem stating the existence of three positive integers satisfying the conditions -/
theorem flatbread_diameters_exist : ∃ x y z : ℕ, valid_diameters x y z := by
  sorry

end NUMINAMATH_CALUDE_flatbread_diameters_exist_l2249_224904


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2249_224934

-- Define the functions DE, BC, and DB
def DE (x : ℝ) : ℝ := sorry
def BC (x : ℝ) : ℝ := sorry
def DB (x : ℝ) : ℝ := sorry

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | DE x * BC x = DE x * (2 * DB x) ∧ DE x * BC x = 2 * (DE x)^2} = 
  {x : ℝ | 9/4 < x ∧ x < 19/4} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2249_224934


namespace NUMINAMATH_CALUDE_lily_cost_is_four_l2249_224927

/-- Represents the cost structure for wedding reception decorations --/
structure WeddingDecoration where
  numTables : Nat
  tableclothCost : Nat
  placeSettingCost : Nat
  placeSettingsPerTable : Nat
  rosesPerCenterpiece : Nat
  roseCost : Nat
  liliesPerCenterpiece : Nat
  totalCost : Nat

/-- Calculates the cost of each lily given the wedding decoration details --/
def lilyCost (d : WeddingDecoration) : Rat :=
  let tableCostWithoutLilies := d.tableclothCost + 
                                d.placeSettingCost * d.placeSettingsPerTable + 
                                d.rosesPerCenterpiece * d.roseCost
  let totalCostWithoutLilies := d.numTables * tableCostWithoutLilies
  let totalLilyCost := d.totalCost - totalCostWithoutLilies
  let totalLilies := d.numTables * d.liliesPerCenterpiece
  totalLilyCost / totalLilies

/-- Theorem stating that the lily cost for the given wedding decoration is $4 --/
theorem lily_cost_is_four (d : WeddingDecoration) 
  (h1 : d.numTables = 20)
  (h2 : d.tableclothCost = 25)
  (h3 : d.placeSettingCost = 10)
  (h4 : d.placeSettingsPerTable = 4)
  (h5 : d.rosesPerCenterpiece = 10)
  (h6 : d.roseCost = 5)
  (h7 : d.liliesPerCenterpiece = 15)
  (h8 : d.totalCost = 3500) : 
  lilyCost d = 4 := by
  sorry

#eval lilyCost {
  numTables := 20,
  tableclothCost := 25,
  placeSettingCost := 10,
  placeSettingsPerTable := 4,
  rosesPerCenterpiece := 10,
  roseCost := 5,
  liliesPerCenterpiece := 15,
  totalCost := 3500
}

end NUMINAMATH_CALUDE_lily_cost_is_four_l2249_224927


namespace NUMINAMATH_CALUDE_sum_of_xy_is_one_l2249_224986

theorem sum_of_xy_is_one (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^5 + 5*x^3*y + 5*x^2*y^2 + 5*x*y^3 + y^5 = 1) : 
  x + y = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_xy_is_one_l2249_224986


namespace NUMINAMATH_CALUDE_binary_to_base4_conversion_l2249_224936

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem binary_to_base4_conversion :
  decimal_to_base4 (binary_to_decimal [false, true, false, false, true, true, false, true, true, false, true]) = [2, 3, 1, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_base4_conversion_l2249_224936


namespace NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l2249_224984

theorem max_value_of_sum_and_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_and_reciprocal_l2249_224984


namespace NUMINAMATH_CALUDE_absolute_value_equation_l2249_224959

theorem absolute_value_equation (x : ℚ) :
  |6 + x| = |6| + |x| ↔ x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l2249_224959


namespace NUMINAMATH_CALUDE_sequence_perfect_squares_l2249_224978

theorem sequence_perfect_squares (n : ℕ) (h : n > 0) :
  ∃ k : ℕ, (3 * ((10^n - 1) / 9) + 4) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_perfect_squares_l2249_224978


namespace NUMINAMATH_CALUDE_faucet_leak_proof_l2249_224981

/-- Represents a linear function y = kt + b -/
structure LinearFunction where
  k : ℝ
  b : ℝ

/-- The linear function passes through the points (1, 7) and (2, 12) -/
def passesThrough (f : LinearFunction) : Prop :=
  f.k * 1 + f.b = 7 ∧ f.k * 2 + f.b = 12

/-- The value of the function at t = 20 -/
def valueAt20 (f : LinearFunction) : ℝ :=
  f.k * 20 + f.b

/-- The total water leaked in 30 days in milliliters -/
def totalLeaked (f : LinearFunction) : ℝ :=
  f.k * 60 * 24 * 30

theorem faucet_leak_proof (f : LinearFunction) 
  (h : passesThrough f) : 
  f.k = 5 ∧ f.b = 2 ∧ 
  valueAt20 f = 102 ∧ 
  totalLeaked f = 216000 := by
  sorry

#check faucet_leak_proof

end NUMINAMATH_CALUDE_faucet_leak_proof_l2249_224981


namespace NUMINAMATH_CALUDE_book_has_120_pages_l2249_224995

/-- Represents a book reading plan. -/
structure ReadingPlan where
  pagesPerNight : ℕ
  totalDays : ℕ

/-- Calculates the total number of pages in a book given a reading plan. -/
def totalPages (plan : ReadingPlan) : ℕ :=
  plan.pagesPerNight * plan.totalDays

/-- Theorem stating that the book has 120 pages given the specified reading plan. -/
theorem book_has_120_pages :
  ∃ (plan : ReadingPlan),
    plan.pagesPerNight = 12 ∧
    plan.totalDays = 10 ∧
    totalPages plan = 120 := by
  sorry


end NUMINAMATH_CALUDE_book_has_120_pages_l2249_224995


namespace NUMINAMATH_CALUDE_expression_equals_one_l2249_224960

theorem expression_equals_one : (2 * 6) / (12 * 14) * (3 * 12 * 14) / (2 * 6 * 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2249_224960


namespace NUMINAMATH_CALUDE_missing_donuts_percentage_l2249_224945

def initial_donuts : ℕ := 30
def remaining_donuts : ℕ := 9

theorem missing_donuts_percentage :
  (initial_donuts - remaining_donuts : ℚ) / initial_donuts * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_missing_donuts_percentage_l2249_224945


namespace NUMINAMATH_CALUDE_hibiscus_flower_ratio_l2249_224941

/-- Given Mario's hibiscus plants, prove the ratio of flowers on the third to second plant -/
theorem hibiscus_flower_ratio :
  let first_plant_flowers : ℕ := 2
  let second_plant_flowers : ℕ := 2 * first_plant_flowers
  let total_flowers : ℕ := 22
  let third_plant_flowers : ℕ := total_flowers - first_plant_flowers - second_plant_flowers
  third_plant_flowers / second_plant_flowers = 4 := by
sorry

end NUMINAMATH_CALUDE_hibiscus_flower_ratio_l2249_224941


namespace NUMINAMATH_CALUDE_P_divisibility_l2249_224967

/-- The polynomial P(x) -/
def P (a x : ℝ) : ℝ := a^3 * x^5 + (1 - a) * x^4 + (1 + a^3) * x^2 + (1 - 3*a) * x - a^3

/-- The set of values of a for which P(x) is divisible by (x-1) -/
def A : Set ℝ := {a | ∃ q : ℝ → ℝ, ∀ x, P a x = (x - 1) * q x}

theorem P_divisibility :
  A = {1, (-1 + Real.sqrt 13) / 2, (-1 - Real.sqrt 13) / 2} :=
sorry

end NUMINAMATH_CALUDE_P_divisibility_l2249_224967


namespace NUMINAMATH_CALUDE_kittens_per_female_cat_l2249_224950

theorem kittens_per_female_cat 
  (total_adult_cats : ℕ)
  (female_ratio : ℚ)
  (sold_kittens : ℕ)
  (kitten_ratio_after_sale : ℚ)
  (h1 : total_adult_cats = 6)
  (h2 : female_ratio = 1/2)
  (h3 : sold_kittens = 9)
  (h4 : kitten_ratio_after_sale = 67/100) :
  ∃ (kittens_per_female : ℕ),
    kittens_per_female = 7 ∧
    (female_ratio * total_adult_cats : ℚ) * kittens_per_female = 
      (1 - kitten_ratio_after_sale) * 
        ((total_adult_cats : ℚ) / (1 - kitten_ratio_after_sale) - total_adult_cats) +
      sold_kittens :=
by
  sorry

end NUMINAMATH_CALUDE_kittens_per_female_cat_l2249_224950


namespace NUMINAMATH_CALUDE_garden_area_theorem_l2249_224976

/-- The area of a rectangle with a square cut out from each of two different corners -/
def garden_area (length width cut1_side cut2_side : ℝ) : ℝ :=
  length * width - cut1_side^2 - cut2_side^2

/-- Theorem: The area of a 20x18 rectangle with 4x4 and 2x2 squares cut out is 340 sq ft -/
theorem garden_area_theorem :
  garden_area 20 18 4 2 = 340 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_theorem_l2249_224976


namespace NUMINAMATH_CALUDE_smallest_sum_consecutive_primes_div_by_5_l2249_224985

/-- Three consecutive primes with sum divisible by 5 -/
def ConsecutivePrimesWithSumDivBy5 (p q r : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  q = Nat.succ p ∧ r = Nat.succ q ∧
  (p + q + r) % 5 = 0

/-- The smallest sum of three consecutive primes divisible by 5 -/
theorem smallest_sum_consecutive_primes_div_by_5 :
  ∃ (p q r : ℕ), ConsecutivePrimesWithSumDivBy5 p q r ∧
    ∀ (a b c : ℕ), ConsecutivePrimesWithSumDivBy5 a b c → p + q + r ≤ a + b + c ∧
    p + q + r = 10 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_consecutive_primes_div_by_5_l2249_224985


namespace NUMINAMATH_CALUDE_cars_in_driveway_three_cars_in_driveway_l2249_224949

/-- Calculates the number of cars in the driveway given the total number of wheels and the number of wheels for each item. -/
theorem cars_in_driveway (total_wheels : ℕ) (car_wheels bike_wheels trash_can_wheels tricycle_wheels roller_skate_wheels : ℕ)
  (num_bikes num_trash_cans num_tricycles num_roller_skate_pairs : ℕ) : ℕ :=
  let other_wheels := num_bikes * bike_wheels + num_trash_cans * trash_can_wheels +
                      num_tricycles * tricycle_wheels + num_roller_skate_pairs * roller_skate_wheels
  let remaining_wheels := total_wheels - other_wheels
  remaining_wheels / car_wheels

/-- Proves that there are 3 cars in the driveway given the specific conditions. -/
theorem three_cars_in_driveway :
  cars_in_driveway 25 4 2 2 3 4 2 1 1 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cars_in_driveway_three_cars_in_driveway_l2249_224949


namespace NUMINAMATH_CALUDE_three_distinct_zeros_range_l2249_224912

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

-- State the theorem
theorem three_distinct_zeros_range (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) →
  -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_three_distinct_zeros_range_l2249_224912


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2249_224914

-- Define the coefficients of the original quadratic equation
def a : ℝ := 3
def b : ℝ := 4
def c : ℝ := 2

-- Define the roots of the original quadratic equation
def α : ℝ := sorry
def β : ℝ := sorry

-- Define the coefficients of the new quadratic equation
def a' : ℝ := 4
def p : ℝ := sorry
def q : ℝ := sorry

-- State the theorem
theorem quadratic_roots_relation :
  (3 * α^2 + 4 * α + 2 = 0) ∧
  (3 * β^2 + 4 * β + 2 = 0) ∧
  (4 * (2*α + 1)^2 + p * (2*α + 1) + q = 0) ∧
  (4 * (2*β + 1)^2 + p * (2*β + 1) + q = 0) →
  p = 8/3 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2249_224914


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2249_224935

theorem monic_quartic_polynomial_value (q : ℝ → ℝ) : 
  (∃ a b c : ℝ, ∀ x, q x = x^4 + a*x^3 + b*x^2 + c*x + 3) →
  q 0 = 3 →
  q 1 = 4 →
  q 2 = 7 →
  q 3 = 12 →
  q 4 = 43 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2249_224935


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l2249_224963

theorem right_triangle_acute_angles (θ₁ θ₂ : ℝ) : 
  θ₁ = 25 → θ₁ + θ₂ = 90 → θ₂ = 65 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l2249_224963


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2249_224909

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

-- Theorem statement
theorem subset_implies_a_equals_one :
  ∀ a : ℝ, A a ⊆ B a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2249_224909


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2249_224989

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2249_224989


namespace NUMINAMATH_CALUDE_minimum_raft_capacity_l2249_224996

/-- Represents an animal with its weight -/
structure Animal where
  weight : ℕ

/-- Represents the raft with its capacity -/
structure Raft where
  capacity : ℕ

/-- Checks if the raft can carry at least two of the lightest animals -/
def canCarryTwoLightest (r : Raft) (animals : List Animal) : Prop :=
  r.capacity ≥ 2 * (animals.map Animal.weight).minimum

/-- Checks if all animals can be transported using the given raft -/
def canTransportAll (r : Raft) (animals : List Animal) : Prop :=
  canCarryTwoLightest r animals

/-- The main theorem stating the minimum raft capacity -/
theorem minimum_raft_capacity 
  (mice : List Animal)
  (moles : List Animal)
  (hamsters : List Animal)
  (h_mice : mice.length = 5 ∧ ∀ m ∈ mice, m.weight = 70)
  (h_moles : moles.length = 3 ∧ ∀ m ∈ moles, m.weight = 90)
  (h_hamsters : hamsters.length = 4 ∧ ∀ h ∈ hamsters, h.weight = 120)
  : ∃ (r : Raft), r.capacity = 140 ∧ 
    canTransportAll r (mice ++ moles ++ hamsters) ∧
    ∀ (r' : Raft), r'.capacity < 140 → ¬canTransportAll r' (mice ++ moles ++ hamsters) :=
sorry

end NUMINAMATH_CALUDE_minimum_raft_capacity_l2249_224996


namespace NUMINAMATH_CALUDE_sum_of_roots_equal_three_l2249_224952

-- Define the polynomial
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 - 48 * x - 12

-- State the theorem
theorem sum_of_roots_equal_three :
  ∃ (r p q : ℝ), f r = 0 ∧ f p = 0 ∧ f q = 0 ∧ r + p + q = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equal_three_l2249_224952


namespace NUMINAMATH_CALUDE_rug_inner_length_is_four_l2249_224966

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : RectDimensions) : ℝ := d.length * d.width

/-- Represents the three colored regions of the rug -/
structure RugRegions where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Checks if three real numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop := b - a = c - b

theorem rug_inner_length_is_four :
  ∀ (r : RugRegions),
    r.inner.width = 2 →
    r.middle.length = r.inner.length + 4 →
    r.middle.width = r.inner.width + 4 →
    r.outer.length = r.middle.length + 4 →
    r.outer.width = r.middle.width + 4 →
    isArithmeticProgression (area r.inner) (area r.middle - area r.inner) (area r.outer - area r.middle) →
    r.inner.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_four_l2249_224966


namespace NUMINAMATH_CALUDE_estimate_at_25_l2249_224903

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the estimated y value for a given x on a regression line -/
def estimate_y (line : RegressionLine) (x : ℝ) : ℝ :=
  line.slope * x + line.intercept

/-- The specific regression line y = 0.5x - 0.81 -/
def specific_line : RegressionLine :=
  { slope := 0.5, intercept := -0.81 }

/-- Theorem: The estimated y value when x = 25 on the specific regression line is 11.69 -/
theorem estimate_at_25 :
  estimate_y specific_line 25 = 11.69 := by sorry

end NUMINAMATH_CALUDE_estimate_at_25_l2249_224903


namespace NUMINAMATH_CALUDE_D_144_l2249_224994

/-- D(n) represents the number of ways to write a positive integer n as a product of 
    integers strictly greater than 1, where the order of factors matters. -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(144) = 45 -/
theorem D_144 : D 144 = 45 := by sorry

end NUMINAMATH_CALUDE_D_144_l2249_224994


namespace NUMINAMATH_CALUDE_largest_class_proof_l2249_224957

/-- The number of students in the largest class of a school with the following properties:
  - There are 5 classes
  - Each class has 2 students less than the previous class
  - The total number of students is 95
-/
def largest_class : ℕ := 23

theorem largest_class_proof :
  let classes := 5
  let student_difference := 2
  let total_students := 95
  let class_sizes := List.range classes |>.map (λ i => largest_class - i * student_difference)
  classes = 5 ∧
  student_difference = 2 ∧
  total_students = 95 ∧
  class_sizes.sum = total_students ∧
  largest_class ≥ 0 ∧
  (∀ i ∈ class_sizes, i ≥ 0) →
  largest_class = 23 :=
by sorry

end NUMINAMATH_CALUDE_largest_class_proof_l2249_224957


namespace NUMINAMATH_CALUDE_f_range_is_real_l2249_224958

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 4 - Real.cos x * Real.sin x + Real.sin x ^ 4 + Real.tan x

theorem f_range_is_real : Set.range f = Set.univ :=
sorry

end NUMINAMATH_CALUDE_f_range_is_real_l2249_224958


namespace NUMINAMATH_CALUDE_largest_integer_not_exceeding_700pi_l2249_224902

theorem largest_integer_not_exceeding_700pi :
  ⌊700 * Real.pi⌋ = 2199 := by sorry

end NUMINAMATH_CALUDE_largest_integer_not_exceeding_700pi_l2249_224902


namespace NUMINAMATH_CALUDE_blocks_used_for_tower_l2249_224901

theorem blocks_used_for_tower (initial_blocks : ℕ) (blocks_left : ℕ) : 
  initial_blocks = 78 → blocks_left = 59 → initial_blocks - blocks_left = 19 := by
  sorry

end NUMINAMATH_CALUDE_blocks_used_for_tower_l2249_224901


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2249_224913

theorem quadratic_inequality_condition (m : ℝ) :
  (∀ x : ℝ, x > 1 ∧ x < 2 → x^2 + m*x + 4 < 0) ↔ m ≤ -5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2249_224913


namespace NUMINAMATH_CALUDE_regular_polygon_diagonals_l2249_224965

theorem regular_polygon_diagonals (n : ℕ) (h : n > 2) :
  (n * (n - 3) / 2 : ℚ) = 2 * n → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_diagonals_l2249_224965


namespace NUMINAMATH_CALUDE_square_sum_given_conditions_l2249_224932

theorem square_sum_given_conditions (x y : ℝ) 
  (h1 : x^2 + y^2 = 20) 
  (h2 : x * y = 6) : 
  (x + y)^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_sum_given_conditions_l2249_224932


namespace NUMINAMATH_CALUDE_candidate_vote_difference_l2249_224925

theorem candidate_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 8200 →
  candidate_percentage = 35/100 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 2460 := by
  sorry

end NUMINAMATH_CALUDE_candidate_vote_difference_l2249_224925


namespace NUMINAMATH_CALUDE_sunset_delay_theorem_l2249_224948

/-- Calculates the minutes until sunset given the initial sunset time,
    daily sunset delay, days passed, and current time. -/
def minutesUntilSunset (initialSunsetMinutes : ℕ) (dailyDelayMinutes : ℚ)
                       (daysPassed : ℕ) (currentTimeMinutes : ℕ) : ℚ :=
  let newSunsetMinutes : ℚ := initialSunsetMinutes + daysPassed * dailyDelayMinutes
  newSunsetMinutes - currentTimeMinutes

/-- Proves that 40 days after March 1st, at 6:10 PM, 
    there are 38 minutes until sunset. -/
theorem sunset_delay_theorem :
  minutesUntilSunset 1080 1.2 40 1090 = 38 := by
  sorry

#eval minutesUntilSunset 1080 1.2 40 1090

end NUMINAMATH_CALUDE_sunset_delay_theorem_l2249_224948


namespace NUMINAMATH_CALUDE_faye_earnings_l2249_224961

/-- Calculates the earnings from selling necklaces at a garage sale -/
def necklace_earnings (bead_necklaces gem_stone_necklaces price_per_necklace : ℕ) : ℕ :=
  (bead_necklaces + gem_stone_necklaces) * price_per_necklace

/-- Proves that Faye's earnings from selling necklaces are 70 dollars -/
theorem faye_earnings : necklace_earnings 3 7 7 = 70 := by
  sorry

end NUMINAMATH_CALUDE_faye_earnings_l2249_224961


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2249_224911

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + m*y + m = 0 → y = x) ↔ 
  (m = 0 ∨ m = 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2249_224911


namespace NUMINAMATH_CALUDE_farmer_boso_animals_l2249_224946

theorem farmer_boso_animals (a b : ℕ) (h1 : 5 * b = b^(a-5)) (h2 : b = 5) (h3 : a = 7) : ∃ (L : ℕ), L = 3 ∧ 
  (4 * (5 * b) + 2 * (5 * a + 7) + 6 * b^(a-5) = 100 * L + 10 * L + L + 1) :=
sorry

end NUMINAMATH_CALUDE_farmer_boso_animals_l2249_224946


namespace NUMINAMATH_CALUDE_value_of_expression_l2249_224955

theorem value_of_expression (a b : ℤ) (ha : a = -3) (hb : b = 2) : a * (b - 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l2249_224955


namespace NUMINAMATH_CALUDE_building_heights_sum_l2249_224905

/-- The combined height of three buildings with their antennas -/
def combined_height (esb_height esb_antenna wt_height wt_antenna owt_height owt_antenna : ℕ) : ℕ :=
  (esb_height + esb_antenna) + (wt_height + wt_antenna) + (owt_height + owt_antenna)

/-- Theorem stating the combined height of the three buildings -/
theorem building_heights_sum :
  combined_height 1250 204 1450 280 1368 408 = 4960 := by
  sorry

end NUMINAMATH_CALUDE_building_heights_sum_l2249_224905


namespace NUMINAMATH_CALUDE_stock_price_increase_l2249_224970

theorem stock_price_increase (opening_price : ℝ) (increase_percentage : ℝ) : 
  opening_price = 10 → increase_percentage = 0.5 → 
  opening_price * (1 + increase_percentage) = 15 := by
  sorry

#check stock_price_increase

end NUMINAMATH_CALUDE_stock_price_increase_l2249_224970


namespace NUMINAMATH_CALUDE_greatest_x_value_l2249_224979

theorem greatest_x_value (x : ℤ) : 
  (2.134 * (10 : ℝ) ^ (x : ℝ) < 240000) ↔ x ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2249_224979


namespace NUMINAMATH_CALUDE_negative_greater_than_reciprocal_is_proper_fraction_l2249_224990

theorem negative_greater_than_reciprocal_is_proper_fraction (a : ℝ) :
  a < 0 ∧ a > 1 / a → -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_CALUDE_negative_greater_than_reciprocal_is_proper_fraction_l2249_224990


namespace NUMINAMATH_CALUDE_base5_divisible_by_13_l2249_224977

/-- Converts a base 5 number to decimal --/
def base5ToDecimal (a b c d : ℕ) : ℕ :=
  a * 5^3 + b * 5^2 + c * 5 + d

/-- Checks if a number is divisible by 13 --/
def isDivisibleBy13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

theorem base5_divisible_by_13 :
  let y := 2
  let base5Num := base5ToDecimal 2 3 y 2
  isDivisibleBy13 base5Num :=
by sorry

end NUMINAMATH_CALUDE_base5_divisible_by_13_l2249_224977


namespace NUMINAMATH_CALUDE_geometric_sequence_303rd_term_l2249_224924

/-- Represents a geometric sequence -/
def GeometricSequence (a₁ : ℝ) (r : ℝ) := fun (n : ℕ) => a₁ * r ^ (n - 1)

theorem geometric_sequence_303rd_term :
  let seq := GeometricSequence 5 (-2)
  seq 303 = 5 * 2^302 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_303rd_term_l2249_224924


namespace NUMINAMATH_CALUDE_tan_sum_specific_angles_l2249_224943

theorem tan_sum_specific_angles (α β : Real) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) :
  Real.tan (α + β) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_specific_angles_l2249_224943


namespace NUMINAMATH_CALUDE_egg_count_l2249_224942

theorem egg_count (initial_eggs : ℕ) (used_eggs : ℕ) (num_chickens : ℕ) (eggs_per_chicken : ℕ) : 
  initial_eggs = 10 → 
  used_eggs = 5 → 
  num_chickens = 2 → 
  eggs_per_chicken = 3 → 
  initial_eggs - used_eggs + num_chickens * eggs_per_chicken = 11 := by
  sorry

#check egg_count

end NUMINAMATH_CALUDE_egg_count_l2249_224942


namespace NUMINAMATH_CALUDE_inequality_proof_l2249_224928

theorem inequality_proof (p q : ℝ) (m n : ℕ+) 
  (h1 : p ≥ 0) (h2 : q ≥ 0) (h3 : p + q = 1) : 
  (1 - p ^ (m : ℝ)) ^ (n : ℝ) + (1 - q ^ (n : ℝ)) ^ (m : ℝ) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2249_224928


namespace NUMINAMATH_CALUDE_least_positive_integer_modulo_solution_satisfies_congruence_twelve_is_least_positive_solution_l2249_224956

theorem least_positive_integer_modulo (x : ℕ) : x + 3001 ≡ 1723 [ZMOD 15] → x ≥ 12 := by
  sorry

theorem solution_satisfies_congruence : 12 + 3001 ≡ 1723 [ZMOD 15] := by
  sorry

theorem twelve_is_least_positive_solution : ∃! x : ℕ, x + 3001 ≡ 1723 [ZMOD 15] ∧ ∀ y : ℕ, y + 3001 ≡ 1723 [ZMOD 15] → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_modulo_solution_satisfies_congruence_twelve_is_least_positive_solution_l2249_224956


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2249_224931

theorem min_value_trig_expression (α : ℝ) : 
  9 / (Real.sin α)^2 + 1 / (Real.cos α)^2 ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2249_224931


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2249_224987

theorem quadratic_roots_condition (a : ℝ) : 
  (-1 < a ∧ a < 1) → 
  (∃ x₁ x₂ : ℝ, x₁ * x₂ = a - 2 ∧ x₁ + x₂ = -(a + 1) ∧ x₁ > 0 ∧ x₂ < 0) ∧
  ¬(∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ * x₂ = a - 2 ∧ x₁ + x₂ = -(a + 1) ∧ x₁ > 0 ∧ x₂ < 0) → (-1 < a ∧ a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2249_224987


namespace NUMINAMATH_CALUDE_sum_equals_product_implies_two_greater_than_one_l2249_224907

theorem sum_equals_product_implies_two_greater_than_one 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum_prod : a + b + c = a * b * c) : 
  (a > 1 ∧ b > 1) ∨ (a > 1 ∧ c > 1) ∨ (b > 1 ∧ c > 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_product_implies_two_greater_than_one_l2249_224907


namespace NUMINAMATH_CALUDE_pascal_triangle_24th_row_20th_number_l2249_224926

theorem pascal_triangle_24th_row_20th_number : 
  (Nat.choose 24 19) = 42504 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_24th_row_20th_number_l2249_224926


namespace NUMINAMATH_CALUDE_total_swordfish_catch_l2249_224923

/-- The number of times Shelly and Sam go fishing -/
def fishing_trips : ℕ := 5

/-- The number of swordfish Shelly catches each time -/
def shelly_catch : ℕ := 5 - 2

/-- The number of swordfish Sam catches each time -/
def sam_catch : ℕ := shelly_catch - 1

/-- The total number of swordfish caught by Shelly and Sam after their fishing trips -/
def total_catch : ℕ := fishing_trips * (shelly_catch + sam_catch)

theorem total_swordfish_catch : total_catch = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_swordfish_catch_l2249_224923


namespace NUMINAMATH_CALUDE_inequalities_theorem_l2249_224972

theorem inequalities_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a * b) / (a + b) ≤ (a + b) / 2 ∧
  Real.sqrt (a * b) ≤ (a + b) / 2 ∧
  (a + b) / 2 ≤ Real.sqrt ((a^2 + b^2) / 2) ∧
  b^2 / a + a^2 / b ≥ a + b :=
by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l2249_224972


namespace NUMINAMATH_CALUDE_power_of_product_l2249_224964

theorem power_of_product (a b : ℝ) : (a * b^2)^3 = a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2249_224964


namespace NUMINAMATH_CALUDE_arithmetic_fraction_subtraction_l2249_224954

theorem arithmetic_fraction_subtraction :
  (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) - (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) = -9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_fraction_subtraction_l2249_224954
