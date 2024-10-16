import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_identity_l1131_113170

theorem trigonometric_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (170 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1131_113170


namespace NUMINAMATH_CALUDE_reciprocal_difference_decreases_l1131_113134

theorem reciprocal_difference_decreases (n : ℕ) : 
  (1 : ℚ) / n - (1 : ℚ) / (n + 1) = 1 / (n * (n + 1)) ∧
  ∀ m : ℕ, m > n → (1 : ℚ) / m - (1 : ℚ) / (m + 1) < (1 : ℚ) / n - (1 : ℚ) / (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_difference_decreases_l1131_113134


namespace NUMINAMATH_CALUDE_second_die_sides_l1131_113119

theorem second_die_sides (n : ℕ) : 
  n > 0 → 
  (1 : ℚ) / 6 * (1 : ℚ) / n = 0.023809523809523808 → 
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_die_sides_l1131_113119


namespace NUMINAMATH_CALUDE_total_fruits_l1131_113124

theorem total_fruits (apples bananas grapes : ℕ) 
  (h1 : apples = 5) 
  (h2 : bananas = 4) 
  (h3 : grapes = 6) : 
  apples + bananas + grapes = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_l1131_113124


namespace NUMINAMATH_CALUDE_palindromic_four_digit_squares_l1131_113126

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def ends_with_0_4_or_6 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 4 ∨ n % 10 = 6

def satisfies_conditions (n : ℕ) : Prop :=
  is_square n ∧ is_four_digit n ∧ is_palindrome n ∧ ends_with_0_4_or_6 n

theorem palindromic_four_digit_squares :
  ∃! (s : Finset ℕ), s.card = 3 ∧ ∀ n, n ∈ s ↔ satisfies_conditions n :=
sorry

end NUMINAMATH_CALUDE_palindromic_four_digit_squares_l1131_113126


namespace NUMINAMATH_CALUDE_circle_and_reflection_theorem_l1131_113120

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧ 2*a - b - 4 = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (3, 0)
def point_B : ℝ × ℝ := (5, 2)

-- Define the reflection line
def reflection_line (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the point M
def point_M : ℝ × ℝ := (-4, -3)

-- Define the theorem
theorem circle_and_reflection_theorem :
  (circle_C point_A.1 point_A.2) ∧
  (circle_C point_B.1 point_B.2) ∧
  (∃ (x y : ℝ), reflection_line x y ∧ 
    ∃ (t : ℝ), (1 - t) * point_M.1 + t * x = -4 ∧ (1 - t) * point_M.2 + t * y = -3 ∧
    circle_C x y) →
  (∀ (x y : ℝ), circle_C x y ↔ (x - 3)^2 + (y - 2)^2 = 4) ∧
  (∃ (k : ℝ), ∀ (x y : ℝ), (x = 1 ∨ 12*x - 5*y - 52 = 0) ↔ 
    (∃ (t : ℝ), x = (1 - t) * 1 + t * point_M.1 ∧ y = (1 - t) * (-8) + t * point_M.2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_reflection_theorem_l1131_113120


namespace NUMINAMATH_CALUDE_unique_solution_exists_l1131_113196

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 4 * x - 5 * y + 2 * x * y

-- Theorem statement
theorem unique_solution_exists :
  ∃! y : ℝ, star 2 y = 5 := by sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l1131_113196


namespace NUMINAMATH_CALUDE_remaining_savings_l1131_113131

def initial_savings : ℕ := 80
def earrings_cost : ℕ := 23
def necklace_cost : ℕ := 48

theorem remaining_savings : 
  initial_savings - (earrings_cost + necklace_cost) = 9 := by sorry

end NUMINAMATH_CALUDE_remaining_savings_l1131_113131


namespace NUMINAMATH_CALUDE_shane_photos_l1131_113114

theorem shane_photos (total_photos : ℕ) (jan_photos_per_day : ℕ) (jan_days : ℕ) (feb_weeks : ℕ)
  (h1 : total_photos = 146)
  (h2 : jan_photos_per_day = 2)
  (h3 : jan_days = 31)
  (h4 : feb_weeks = 4) :
  (total_photos - jan_photos_per_day * jan_days) / feb_weeks = 21 := by
  sorry

end NUMINAMATH_CALUDE_shane_photos_l1131_113114


namespace NUMINAMATH_CALUDE_sum_always_positive_l1131_113152

-- Define a monotonically increasing odd function
def MonoIncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (hf : MonoIncreasingOddFunction f)
  (ha : ArithmeticSequence a)
  (ha3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l1131_113152


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1131_113185

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1131_113185


namespace NUMINAMATH_CALUDE_log_inequality_implies_x_geq_125_l1131_113129

theorem log_inequality_implies_x_geq_125 (x : ℝ) (h1 : x > 0) 
  (h2 : Real.log x / Real.log 3 ≥ Real.log 5 / Real.log 3 + (2/3) * (Real.log x / Real.log 3)) :
  x ≥ 125 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_implies_x_geq_125_l1131_113129


namespace NUMINAMATH_CALUDE_four_door_room_ways_l1131_113171

/-- The number of ways to enter or exit a room with a given number of doors. -/
def waysToEnterOrExit (numDoors : ℕ) : ℕ := numDoors

/-- The number of different ways to enter and exit a room with a given number of doors. -/
def totalWays (numDoors : ℕ) : ℕ :=
  (waysToEnterOrExit numDoors) * (waysToEnterOrExit numDoors)

/-- Theorem: In a room with four doors, there are 16 different ways to enter and exit. -/
theorem four_door_room_ways :
  totalWays 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_four_door_room_ways_l1131_113171


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1131_113135

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 5) 
  (h2 : a^2 + b^2 = 29) : 
  a * b = 2 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1131_113135


namespace NUMINAMATH_CALUDE_simplified_quadratic_radical_among_given_l1131_113137

def is_simplified_quadratic_radical (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ ¬∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

theorem simplified_quadratic_radical_among_given :
  is_simplified_quadratic_radical (Real.sqrt 6) ∧
  ¬is_simplified_quadratic_radical (Real.sqrt 12) ∧
  ¬is_simplified_quadratic_radical (Real.sqrt 20) ∧
  ¬is_simplified_quadratic_radical (Real.sqrt 32) :=
by sorry

end NUMINAMATH_CALUDE_simplified_quadratic_radical_among_given_l1131_113137


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l1131_113193

theorem pie_crust_flour_calculation (initial_crusts : ℕ) (new_crusts : ℕ) (initial_flour : ℚ) :
  initial_crusts = 36 →
  new_crusts = 24 →
  initial_flour = 1/8 →
  (initial_crusts : ℚ) * initial_flour = (new_crusts : ℚ) * ((3:ℚ)/16) :=
by sorry

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l1131_113193


namespace NUMINAMATH_CALUDE_correct_system_of_equations_l1131_113181

-- Define the number of people and price of goods
variable (x y : ℤ)

-- Define the conditions
def condition1 (x y : ℤ) : Prop := 8 * x - 3 = y
def condition2 (x y : ℤ) : Prop := 7 * x + 4 = y

-- Theorem statement
theorem correct_system_of_equations :
  (∀ x y : ℤ, condition1 x y ∧ condition2 x y →
    (8 * x - 3 = y ∧ 7 * x + 4 = y)) :=
by sorry

end NUMINAMATH_CALUDE_correct_system_of_equations_l1131_113181


namespace NUMINAMATH_CALUDE_solve_system_l1131_113107

theorem solve_system (x y : ℚ) (h1 : 3 * x - y = 9) (h2 : x + 4 * y = 11) : x = 47 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1131_113107


namespace NUMINAMATH_CALUDE_height_weight_correlation_l1131_113197

-- Define the relationship types
inductive Relationship
| Functional
| Correlated
| Unrelated

-- Define the variables
structure Square where
  side : ℝ

structure Vehicle where
  speed : ℝ

structure Person where
  height : ℝ
  weight : ℝ
  eyesight : ℝ

-- Define the relationships between variables
def square_area_perimeter_relation (s : Square) : Relationship :=
  Relationship.Functional

def vehicle_distance_time_relation (v : Vehicle) : Relationship :=
  Relationship.Functional

def person_height_weight_relation (p : Person) : Relationship :=
  Relationship.Correlated

def person_height_eyesight_relation (p : Person) : Relationship :=
  Relationship.Unrelated

-- Theorem statement
theorem height_weight_correlation :
  ∃ (p : Person), person_height_weight_relation p = Relationship.Correlated ∧
    (∀ (s : Square), square_area_perimeter_relation s ≠ Relationship.Correlated) ∧
    (∀ (v : Vehicle), vehicle_distance_time_relation v ≠ Relationship.Correlated) ∧
    (person_height_eyesight_relation p ≠ Relationship.Correlated) :=
  sorry

end NUMINAMATH_CALUDE_height_weight_correlation_l1131_113197


namespace NUMINAMATH_CALUDE_gcd_3_powers_l1131_113191

theorem gcd_3_powers : Nat.gcd (3^1001 - 1) (3^1012 - 1) = 177146 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3_powers_l1131_113191


namespace NUMINAMATH_CALUDE_jones_wardrobe_l1131_113128

/-- The ratio of shirts to pants in Mr. Jones' wardrobe -/
def shirt_to_pants_ratio : ℕ := 6

/-- The number of pants Mr. Jones owns -/
def number_of_pants : ℕ := 40

/-- The total number of pieces of clothes Mr. Jones owns -/
def total_clothes : ℕ := shirt_to_pants_ratio * number_of_pants + number_of_pants

theorem jones_wardrobe : total_clothes = 280 := by
  sorry

end NUMINAMATH_CALUDE_jones_wardrobe_l1131_113128


namespace NUMINAMATH_CALUDE_power_sum_l1131_113176

theorem power_sum (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l1131_113176


namespace NUMINAMATH_CALUDE_quadratic_function_inequality_l1131_113198

theorem quadratic_function_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a < 0) 
  (hx : x₁ < x₂) 
  (hsum : x₁ + x₂ = 0) : 
  2 * a * x₁^2 - a * x₁ + 1 < 2 * a * x₂^2 - a * x₂ + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_inequality_l1131_113198


namespace NUMINAMATH_CALUDE_sum_of_zeros_is_four_l1131_113132

/-- Original parabola function -/
def original_parabola (x : ℝ) : ℝ := (x + 3)^2 - 2

/-- Transformed parabola function -/
def transformed_parabola (x : ℝ) : ℝ := -(x - 2)^2 + 2

/-- The zeros of the transformed parabola -/
def zeros : Set ℝ := {x | transformed_parabola x = 0}

theorem sum_of_zeros_is_four :
  ∃ (a b : ℝ), a ∈ zeros ∧ b ∈ zeros ∧ a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_zeros_is_four_l1131_113132


namespace NUMINAMATH_CALUDE_penny_difference_l1131_113159

theorem penny_difference (kate_pennies john_pennies : ℕ) 
  (h1 : kate_pennies = 223) 
  (h2 : john_pennies = 388) : 
  john_pennies - kate_pennies = 165 := by
sorry

end NUMINAMATH_CALUDE_penny_difference_l1131_113159


namespace NUMINAMATH_CALUDE_dinner_cakes_count_l1131_113110

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := 5

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served over two days -/
def total_cakes : ℕ := 14

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := total_cakes - lunch_cakes - yesterday_cakes

theorem dinner_cakes_count : dinner_cakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cakes_count_l1131_113110


namespace NUMINAMATH_CALUDE_count_possible_D_values_l1131_113173

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if a list of digits are all distinct -/
def all_distinct (digits : List Digit) : Prop :=
  ∀ i j, i ≠ j → digits.get i ≠ digits.get j

/-- Converts a list of digits to a natural number -/
def to_nat (digits : List Digit) : ℕ :=
  digits.foldl (λ acc d => 10 * acc + d.val) 0

/-- The main theorem -/
theorem count_possible_D_values :
  ∃ (possible_D_values : Finset Digit),
    (∀ A B C E D : Digit,
      all_distinct [A, B, C, E, D] →
      to_nat [A, B, C, E, B] + to_nat [B, C, E, D, A] = to_nat [D, B, D, D, D] →
      D ∈ possible_D_values) ∧
    possible_D_values.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_count_possible_D_values_l1131_113173


namespace NUMINAMATH_CALUDE_fraction_inequality_l1131_113175

theorem fraction_inequality (x : ℝ) :
  0 ≤ x ∧ x ≤ 3 →
  (3 * x + 2 < 2 * (5 * x - 4) ↔ 10 / 7 < x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1131_113175


namespace NUMINAMATH_CALUDE_min_seats_for_adjacency_l1131_113102

/-- Represents a row of seats -/
structure SeatRow where
  total_seats : ℕ
  occupied_seats : ℕ

/-- Checks if the next person must sit next to someone -/
def must_sit_next (row : SeatRow) : Prop :=
  ∀ i : ℕ, i < row.total_seats - 1 → (i % 4 = 0 → i < row.occupied_seats * 4)

/-- The main theorem to be proved -/
theorem min_seats_for_adjacency (row : SeatRow) :
  row.total_seats = 150 →
  (∀ r : SeatRow, r.total_seats = 150 → r.occupied_seats < 37 → ¬ must_sit_next r) →
  must_sit_next row →
  row.occupied_seats ≥ 37 :=
sorry

end NUMINAMATH_CALUDE_min_seats_for_adjacency_l1131_113102


namespace NUMINAMATH_CALUDE_amaro_roses_l1131_113164

theorem amaro_roses :
  ∀ (total_roses : ℕ),
  (3 * total_roses / 4 : ℚ) + (3 * total_roses / 16 : ℚ) = 75 →
  total_roses = 80 := by
sorry

end NUMINAMATH_CALUDE_amaro_roses_l1131_113164


namespace NUMINAMATH_CALUDE_backpack_profit_analysis_l1131_113151

/-- Represents the daily profit function for backpack sales -/
def daily_profit (x : ℝ) : ℝ := -x^2 + 90*x - 1800

/-- Represents the daily sales quantity function -/
def sales_quantity (x : ℝ) : ℝ := -x + 60

theorem backpack_profit_analysis 
  (cost_price : ℝ) 
  (price_range : Set ℝ) 
  (max_price : ℝ) 
  (target_profit : ℝ) :
  cost_price = 30 →
  price_range = {x : ℝ | 30 ≤ x ∧ x ≤ 60} →
  max_price = 48 →
  target_profit = 200 →
  (∀ x ∈ price_range, daily_profit x = (x - cost_price) * sales_quantity x) ∧
  (∃ x ∈ price_range, x ≤ max_price ∧ daily_profit x = target_profit ∧ x = 40) ∧
  (∃ x ∈ price_range, ∀ y ∈ price_range, daily_profit x ≥ daily_profit y ∧ 
    x = 45 ∧ daily_profit x = 225) :=
by sorry

end NUMINAMATH_CALUDE_backpack_profit_analysis_l1131_113151


namespace NUMINAMATH_CALUDE_product_integer_part_l1131_113177

theorem product_integer_part : 
  ⌊(1.1 : ℝ) * 1.2 * 1.3 * 1.4 * 1.5 * 1.6⌋ = 1 := by sorry

end NUMINAMATH_CALUDE_product_integer_part_l1131_113177


namespace NUMINAMATH_CALUDE_maxwell_walking_speed_l1131_113143

-- Define the given conditions
def total_distance : ℝ := 65
def maxwell_distance : ℝ := 26
def brad_speed : ℝ := 3

-- Define Maxwell's speed as a variable
def maxwell_speed : ℝ := sorry

-- Theorem to prove
theorem maxwell_walking_speed :
  (maxwell_distance / maxwell_speed = (total_distance - maxwell_distance) / brad_speed) →
  maxwell_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_maxwell_walking_speed_l1131_113143


namespace NUMINAMATH_CALUDE_triangle_properties_l1131_113148

-- Define the triangle ABC
def Triangle (A B C : ℝ) := A + B + C = Real.pi

-- Define the conditions
def ConditionOne (A B C : ℝ) := A + B = 3 * C
def ConditionTwo (A B C : ℝ) := 2 * Real.sin (A - C) = Real.sin B
def ConditionThree := 5

-- Define the height function
def Height (A B C : ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_properties (A B C : ℝ) 
  (h1 : Triangle A B C) 
  (h2 : ConditionOne A B C) 
  (h3 : ConditionTwo A B C) :
  Real.sin A = 3 * Real.sqrt 10 / 10 ∧ 
  Height A B C = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1131_113148


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1131_113162

theorem arithmetic_mean_of_fractions : 
  (3/8 + 5/9) / 2 = 67/144 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1131_113162


namespace NUMINAMATH_CALUDE_tan_negative_seven_pi_fourths_l1131_113166

theorem tan_negative_seven_pi_fourths : Real.tan (-7 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_seven_pi_fourths_l1131_113166


namespace NUMINAMATH_CALUDE_range_of_a_l1131_113174

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 3*a < x ∧ x < a ∧ a < 0}
def B : Set ℝ := {x | x < -4 ∨ x ≥ -2}

-- Define the conditions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

-- State the theorem
theorem range_of_a :
  (∀ x, p x a → q x) ∧ 
  (∃ x, ¬p x a ∧ q x) →
  (a ≤ -4 ∨ (-2/3 ≤ a ∧ a < 0)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1131_113174


namespace NUMINAMATH_CALUDE_smallest_value_complex_sum_l1131_113133

theorem smallest_value_complex_sum (a b c : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_omega_power : ω^4 = 1)
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (∀ (p q r : ℤ), p ≠ q ∧ q ≠ r ∧ p ≠ r → 
      Complex.abs (x + y*ω + z*ω^3) ≤ Complex.abs (p + q*ω + r*ω^3)) ∧
    Complex.abs (x + y*ω + z*ω^3) = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_complex_sum_l1131_113133


namespace NUMINAMATH_CALUDE_wilson_class_blue_eyes_l1131_113165

/-- Represents the class composition -/
structure ClassComposition where
  total : ℕ
  blond_to_blue_ratio : Rat
  both_traits : ℕ
  neither_trait : ℕ

/-- Calculates the number of blue-eyed students -/
def blue_eyed_count (c : ClassComposition) : ℕ :=
  sorry

/-- Theorem stating the number of blue-eyed students in Mrs. Wilson's class -/
theorem wilson_class_blue_eyes :
  let c : ClassComposition := {
    total := 40,
    blond_to_blue_ratio := 3 / 2,
    both_traits := 8,
    neither_trait := 5
  }
  blue_eyed_count c = 18 := by sorry

end NUMINAMATH_CALUDE_wilson_class_blue_eyes_l1131_113165


namespace NUMINAMATH_CALUDE_prob_diff_is_one_third_l1131_113105

/-- The number of marbles of each color in the box -/
def marbles_per_color : ℕ := 1500

/-- The total number of marbles in the box -/
def total_marbles : ℕ := 3 * marbles_per_color

/-- The probability of drawing two marbles of the same color -/
def prob_same_color : ℚ :=
  (3 * (marbles_per_color.choose 2)) / (total_marbles.choose 2)

/-- The probability of drawing two marbles of different colors -/
def prob_diff_color : ℚ :=
  (3 * marbles_per_color * marbles_per_color) / (total_marbles.choose 2)

/-- The theorem stating that the absolute difference between the probabilities is 1/3 -/
theorem prob_diff_is_one_third :
  |prob_same_color - prob_diff_color| = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_diff_is_one_third_l1131_113105


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l1131_113142

theorem binomial_coefficient_n_minus_two (n : ℕ) (hn : n > 0) : 
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_minus_two_l1131_113142


namespace NUMINAMATH_CALUDE_integral_x_squared_minus_x_l1131_113146

theorem integral_x_squared_minus_x : ∫ x in (0)..(2), (x^2 - x) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_minus_x_l1131_113146


namespace NUMINAMATH_CALUDE_income_scientific_notation_l1131_113106

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  one_le_coeff_lt_ten : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Rounds a ScientificNotation to a specified number of significant figures -/
def roundToSignificantFigures (sn : ScientificNotation) (figures : ℕ) : ScientificNotation :=
  sorry

theorem income_scientific_notation :
  let income : ℝ := 31.534 * 1000000000
  let scientific_form := toScientificNotation income
  let rounded_form := roundToSignificantFigures scientific_form 2
  rounded_form = ScientificNotation.mk 3.2 10 sorry :=
sorry

end NUMINAMATH_CALUDE_income_scientific_notation_l1131_113106


namespace NUMINAMATH_CALUDE_no_equal_tuesdays_fridays_l1131_113154

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a 30-day month -/
def Month := Fin 30

/-- Returns the day of the week for a given day in the month, given the starting day -/
def dayOfWeek (startDay : DayOfWeek) (day : Month) : DayOfWeek :=
  sorry

/-- Counts the number of occurrences of a specific day in a 30-day month -/
def countDayOccurrences (startDay : DayOfWeek) (targetDay : DayOfWeek) : Nat :=
  sorry

/-- Theorem: No starting day results in equal Tuesdays and Fridays in a 30-day month -/
theorem no_equal_tuesdays_fridays :
  ∀ (startDay : DayOfWeek),
    countDayOccurrences startDay DayOfWeek.Tuesday ≠ 
    countDayOccurrences startDay DayOfWeek.Friday :=
  sorry

end NUMINAMATH_CALUDE_no_equal_tuesdays_fridays_l1131_113154


namespace NUMINAMATH_CALUDE_fraction_simplification_l1131_113125

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (2 / y + 1 / x) / (1 / x) = 13 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1131_113125


namespace NUMINAMATH_CALUDE_files_remaining_l1131_113190

theorem files_remaining (m v d : ℕ) (hm : m = 4) (hv : v = 21) (hd : d = 23) :
  (m + v) - d = 2 :=
by sorry

end NUMINAMATH_CALUDE_files_remaining_l1131_113190


namespace NUMINAMATH_CALUDE_composite_sum_l1131_113187

theorem composite_sum (x y : ℕ) (h1 : x > 1) (h2 : y > 1) 
  (h3 : ∃ k : ℕ, x^2 + x*y - y = k^2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ x + y + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_sum_l1131_113187


namespace NUMINAMATH_CALUDE_fundraising_contribution_l1131_113130

theorem fundraising_contribution (total_amount : ℕ) (num_participants : ℕ) 
  (h1 : total_amount = 2400) (h2 : num_participants = 9) : 
  (total_amount + num_participants - 1) / num_participants = 267 :=
by
  sorry

#check fundraising_contribution

end NUMINAMATH_CALUDE_fundraising_contribution_l1131_113130


namespace NUMINAMATH_CALUDE_largest_solution_of_quartic_l1131_113195

theorem largest_solution_of_quartic (x : ℝ) : 
  x^4 - 50*x^2 + 625 = 0 → x ≤ 5 ∧ ∃ y, y^4 - 50*y^2 + 625 = 0 ∧ y = 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_quartic_l1131_113195


namespace NUMINAMATH_CALUDE_negation_equivalence_l1131_113153

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ (x₀ + 1) * Real.exp x₀ > 1) ↔
  (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1131_113153


namespace NUMINAMATH_CALUDE_constant_for_one_even_divisor_l1131_113161

theorem constant_for_one_even_divisor (p c : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2) :
  let n := c * p
  (∃! d : ℕ, d ∣ n ∧ Even d) → c = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_for_one_even_divisor_l1131_113161


namespace NUMINAMATH_CALUDE_tetrahedron_existence_l1131_113141

-- Define a tetrahedron type
structure Tetrahedron :=
  (edges : Fin 6 → ℝ)

-- Define the conditions for configuration (a)
def config_a (t : Tetrahedron) : Prop :=
  (∃ i j : Fin 6, i ≠ j ∧ t.edges i < 0.01 ∧ t.edges j < 0.01) ∧
  (∀ k : Fin 6, (t.edges k ≤ 0.01) ∨ (t.edges k > 1000))

-- Define the conditions for configuration (b)
def config_b (t : Tetrahedron) : Prop :=
  (∃ i j k l : Fin 6, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    t.edges i < 0.01 ∧ t.edges j < 0.01 ∧ t.edges k < 0.01 ∧ t.edges l < 0.01) ∧
  (∀ m : Fin 6, (t.edges m < 0.01) ∨ (t.edges m > 1000))

-- Theorem statements
theorem tetrahedron_existence :
  (∃ t : Tetrahedron, config_a t) ∧ (¬ ∃ t : Tetrahedron, config_b t) :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_existence_l1131_113141


namespace NUMINAMATH_CALUDE_min_value_of_f_l1131_113178

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem: x = 3 minimizes the function f(x) = 3x^2 - 18x + 7 -/
theorem min_value_of_f :
  ∀ x : ℝ, f 3 ≤ f x :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1131_113178


namespace NUMINAMATH_CALUDE_triangle_special_angle_l1131_113155

theorem triangle_special_angle (D E F : ℝ) : 
  D + E + F = 180 →  -- sum of angles in a triangle is 180 degrees
  D = E →            -- angles D and E are equal
  F = 2 * D →        -- angle F is twice angle D
  F = 90 :=          -- prove that F is 90 degrees
by sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l1131_113155


namespace NUMINAMATH_CALUDE_mairead_running_distance_l1131_113168

theorem mairead_running_distance (run walk jog : ℝ) : 
  walk = (3/5) * run → 
  jog = 5 * walk → 
  run + walk + jog = 184 → 
  run = 40 := by
  sorry

end NUMINAMATH_CALUDE_mairead_running_distance_l1131_113168


namespace NUMINAMATH_CALUDE_one_intersection_iff_tangent_l1131_113144

-- Define a line
def Line : Type := sorry

-- Define a conic curve
def ConicCurve : Type := sorry

-- Define the property of having only one intersection point
def hasOneIntersectionPoint (l : Line) (c : ConicCurve) : Prop := sorry

-- Define the property of being tangent
def isTangent (l : Line) (c : ConicCurve) : Prop := sorry

-- Theorem stating that having one intersection point is both sufficient and necessary for being tangent
theorem one_intersection_iff_tangent (l : Line) (c : ConicCurve) : 
  hasOneIntersectionPoint l c ↔ isTangent l c := by sorry

end NUMINAMATH_CALUDE_one_intersection_iff_tangent_l1131_113144


namespace NUMINAMATH_CALUDE_second_company_visit_charge_l1131_113194

/-- Paul's Plumbing visit charge -/
def pauls_visit_charge : ℕ := 55

/-- Paul's Plumbing hourly labor charge -/
def pauls_hourly_charge : ℕ := 35

/-- Second company's hourly labor charge -/
def second_hourly_charge : ℕ := 30

/-- Number of labor hours -/
def labor_hours : ℕ := 4

/-- Second company's visit charge -/
def second_visit_charge : ℕ := 75

theorem second_company_visit_charge :
  pauls_visit_charge + labor_hours * pauls_hourly_charge =
  second_visit_charge + labor_hours * second_hourly_charge :=
by sorry

end NUMINAMATH_CALUDE_second_company_visit_charge_l1131_113194


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1131_113149

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a > 0) (k : b > 0) 
  (eq : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) :
  (Real.sin θ)^12 / a^2 + (Real.cos θ)^12 / b^2 = (a^4 + b^4) / (a + b)^6 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1131_113149


namespace NUMINAMATH_CALUDE_additional_boxes_needed_l1131_113189

/-- Calculates the number of additional flooring boxes needed to complete a room -/
theorem additional_boxes_needed
  (room_length : ℝ)
  (room_width : ℝ)
  (area_per_box : ℝ)
  (area_already_covered : ℝ)
  (h1 : room_length = 16)
  (h2 : room_width = 20)
  (h3 : area_per_box = 10)
  (h4 : area_already_covered = 250) :
  ⌈((room_length * room_width - area_already_covered) / area_per_box)⌉ = 7 :=
by sorry

end NUMINAMATH_CALUDE_additional_boxes_needed_l1131_113189


namespace NUMINAMATH_CALUDE_vowel_sequences_count_l1131_113108

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The length of each sequence -/
def sequence_length : ℕ := 5

/-- Calculates the number of five-letter sequences containing at least one of each vowel -/
def vowel_sequences : ℕ :=
  sequence_length^num_vowels - 
  (Nat.choose num_vowels 1) * (num_vowels - 1)^sequence_length +
  (Nat.choose num_vowels 2) * (num_vowels - 2)^sequence_length -
  (Nat.choose num_vowels 3) * (num_vowels - 3)^sequence_length +
  (Nat.choose num_vowels 4) * (num_vowels - 4)^sequence_length

theorem vowel_sequences_count : vowel_sequences = 120 := by
  sorry

end NUMINAMATH_CALUDE_vowel_sequences_count_l1131_113108


namespace NUMINAMATH_CALUDE_distinct_roots_quadratic_l1131_113122

theorem distinct_roots_quadratic (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 - m*x₁ - 2 = 0) ∧ 
  (x₂^2 - m*x₂ - 2 = 0) := by
sorry

end NUMINAMATH_CALUDE_distinct_roots_quadratic_l1131_113122


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1131_113158

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 + 2*m - 8 = 0 → n^2 + 2*n - 8 = 0 → m^2 + 3*m + n = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1131_113158


namespace NUMINAMATH_CALUDE_cd_store_problem_l1131_113109

/-- Represents the total number of CDs in the store -/
def total_cds : ℕ := sorry

/-- Represents the price of expensive CDs -/
def expensive_price : ℕ := 10

/-- Represents the price of cheap CDs -/
def cheap_price : ℕ := 5

/-- Represents the proportion of expensive CDs -/
def expensive_proportion : ℚ := 2/5

/-- Represents the proportion of cheap CDs -/
def cheap_proportion : ℚ := 3/5

/-- Represents the proportion of expensive CDs bought by Prince -/
def expensive_bought_proportion : ℚ := 1/2

/-- Represents the total amount spent by Prince -/
def total_spent : ℕ := 1000

theorem cd_store_problem :
  (expensive_proportion * expensive_bought_proportion * (total_cds : ℚ) * expensive_price) +
  (cheap_proportion * (total_cds : ℚ) * cheap_price) = total_spent ∧
  total_cds = 200 := by sorry

end NUMINAMATH_CALUDE_cd_store_problem_l1131_113109


namespace NUMINAMATH_CALUDE_three_faced_cubes_surface_area_l1131_113123

/-- Represents a cube with painted faces -/
structure PaintedCube where
  side_length : ℝ
  painted_faces : ℕ

/-- The number of smaller cubes with exactly three painted faces in a larger cube -/
def count_three_faced_cubes (cube_side_length : ℝ) : ℕ := 8

/-- The surface area of a cube -/
def surface_area (cube : PaintedCube) : ℝ := 6 * cube.side_length ^ 2

/-- Theorem: The total surface area of cubes with exactly three painted faces in a 9cm cube divided into 1cm cubes is 48cm² -/
theorem three_faced_cubes_surface_area :
  let large_cube_side_length : ℝ := 9
  let small_cube_side_length : ℝ := 1
  let small_cube := PaintedCube.mk small_cube_side_length 3
  let num_three_faced_cubes := count_three_faced_cubes large_cube_side_length
  num_three_faced_cubes * surface_area small_cube = 48 := by
  sorry

end NUMINAMATH_CALUDE_three_faced_cubes_surface_area_l1131_113123


namespace NUMINAMATH_CALUDE_debate_team_boys_l1131_113183

theorem debate_team_boys (girls : ℕ) (groups : ℕ) (group_size : ℕ) (boys : ℕ) : 
  girls = 45 →
  groups = 8 →
  group_size = 7 →
  groups * group_size = girls + boys →
  boys = 11 := by
sorry

end NUMINAMATH_CALUDE_debate_team_boys_l1131_113183


namespace NUMINAMATH_CALUDE_expectation_decreases_variance_increases_l1131_113179

def boxA : ℕ := 1
def boxB : ℕ := 6
def redInB : ℕ := 3

def E (n : ℕ) : ℚ := (n / 2 + 1) / (n + 1)

def D (n : ℕ) : ℚ := E n * (1 - E n)

theorem expectation_decreases_variance_increases :
  ∀ n m : ℕ, 1 ≤ n → n < m → m ≤ 6 →
    (E n > E m) ∧ (D n < D m) := by
  sorry

end NUMINAMATH_CALUDE_expectation_decreases_variance_increases_l1131_113179


namespace NUMINAMATH_CALUDE_forty_third_digit_of_one_thirteenth_l1131_113111

/-- The decimal representation of 1/13 as a sequence of digits after the decimal point -/
def decimalRep : ℕ → Fin 10
  | n => sorry

/-- The length of the repeating sequence in the decimal representation of 1/13 -/
def repeatLength : ℕ := 6

/-- The 43rd digit after the decimal point in the decimal representation of 1/13 is 0 -/
theorem forty_third_digit_of_one_thirteenth : decimalRep 42 = 0 := by sorry

end NUMINAMATH_CALUDE_forty_third_digit_of_one_thirteenth_l1131_113111


namespace NUMINAMATH_CALUDE_equation_solution_l1131_113138

theorem equation_solution : ∃ x : ℚ, (5 + 3.2 * x = 4.4 * x - 30) ∧ (x = 175 / 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1131_113138


namespace NUMINAMATH_CALUDE_negative_distribution_l1131_113136

theorem negative_distribution (m : ℝ) : -(m - 2) = -m + 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_distribution_l1131_113136


namespace NUMINAMATH_CALUDE_intersection_A_B_l1131_113157

def set_A : Set ℝ := {x | ∃ t : ℝ, x = t^2 + 1}
def set_B : Set ℝ := {x | x * (x - 1) = 0}

theorem intersection_A_B :
  set_A ∩ set_B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1131_113157


namespace NUMINAMATH_CALUDE_inequality_proof_l1131_113140

theorem inequality_proof (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  x / (1 + y) + y / (1 + x) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1131_113140


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1131_113147

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-1, 2) ∧
    radius = 2 ∧
    ∀ (x y : ℝ), circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1131_113147


namespace NUMINAMATH_CALUDE_log_function_through_point_l1131_113113

/-- Given a logarithmic function that passes through the point (4, 2), prove that its base is 2 -/
theorem log_function_through_point (f : ℝ → ℝ) (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∀ x > 0, f x = Real.log x / Real.log a) 
  (h4 : f 4 = 2) : 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_function_through_point_l1131_113113


namespace NUMINAMATH_CALUDE_sector_perimeter_to_circumference_ratio_l1131_113117

theorem sector_perimeter_to_circumference_ratio (r : ℝ) (hr : r > 0) :
  let circumference := 2 * π * r
  let sector_arc_length := circumference / 3
  let sector_perimeter := sector_arc_length + 2 * r
  sector_perimeter / circumference = (π + 3) / (3 * π) := by
sorry

end NUMINAMATH_CALUDE_sector_perimeter_to_circumference_ratio_l1131_113117


namespace NUMINAMATH_CALUDE_slide_ratio_problem_l1131_113121

/-- Given that x boys initially went down a slide, y more boys joined them later,
    and the ratio of boys who went down the slide to boys who watched (z) is 5:3,
    prove that z = 21 when x = 22 and y = 13. -/
theorem slide_ratio_problem (x y : ℕ) (z : ℚ) 
    (h1 : x = 22)
    (h2 : y = 13)
    (h3 : (5 : ℚ) / 3 = (x + y : ℚ) / z) : 
  z = 21 := by
  sorry

end NUMINAMATH_CALUDE_slide_ratio_problem_l1131_113121


namespace NUMINAMATH_CALUDE_min_dist_point_on_line_l1131_113101

/-- The point that minimizes the sum of distances to two given points on a line -/
def minDistPoint (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

/-- The line 3x - 4y + 4 = 0 -/
def line (p : ℝ × ℝ) : Prop :=
  3 * p.1 - 4 * p.2 + 4 = 0

theorem min_dist_point_on_line :
  let A : ℝ × ℝ := (-3, 5)
  let B : ℝ × ℝ := (2, 15)
  let P : ℝ × ℝ := (8/3, 3)
  line P ∧
  ∀ Q : ℝ × ℝ, line Q →
    distance P A + distance P B ≤ distance Q A + distance Q B :=
sorry

end NUMINAMATH_CALUDE_min_dist_point_on_line_l1131_113101


namespace NUMINAMATH_CALUDE_marble_distribution_l1131_113116

theorem marble_distribution (total_marbles : ℕ) (initial_group : ℕ) (joining_group : ℕ) : 
  total_marbles = 312 →
  initial_group = 24 →
  (total_marbles / initial_group : ℕ) = ((total_marbles / (initial_group + joining_group)) + 1 : ℕ) →
  joining_group = 2 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l1131_113116


namespace NUMINAMATH_CALUDE_sticks_per_pack_l1131_113145

/-- Represents the number of packs in a carton -/
def packs_per_carton : ℕ := 5

/-- Represents the number of cartons in a brown box -/
def cartons_per_box : ℕ := 4

/-- Represents the total number of sticks in all brown boxes -/
def total_sticks : ℕ := 480

/-- Represents the total number of brown boxes -/
def total_boxes : ℕ := 8

/-- Theorem stating that the number of sticks in each pack is 3 -/
theorem sticks_per_pack : 
  total_sticks / (total_boxes * cartons_per_box * packs_per_carton) = 3 := by
  sorry


end NUMINAMATH_CALUDE_sticks_per_pack_l1131_113145


namespace NUMINAMATH_CALUDE_min_disks_required_l1131_113188

def disk_capacity : ℝ := 1.44

def file_count : ℕ := 40

def file_sizes : List ℝ := [0.95, 0.95, 0.95, 0.95, 0.95] ++ 
                           List.replicate 15 0.65 ++ 
                           List.replicate 20 0.45

def total_file_size : ℝ := file_sizes.sum

theorem min_disks_required : 
  ∀ (arrangement : List (List ℝ)),
    (arrangement.length < 17 → 
     ∃ (disk : List ℝ), disk ∈ arrangement ∧ disk.sum > disk_capacity) ∧
    (∃ (valid_arrangement : List (List ℝ)), 
      valid_arrangement.length = 17 ∧
      valid_arrangement.join.sum = total_file_size ∧
      ∀ (disk : List ℝ), disk ∈ valid_arrangement → disk.sum ≤ disk_capacity) :=
by sorry

end NUMINAMATH_CALUDE_min_disks_required_l1131_113188


namespace NUMINAMATH_CALUDE_mary_shirts_left_l1131_113184

/-- Calculates the number of shirts Mary has left after giving away fractions of each color --/
def shirts_left (blue brown red yellow green : ℕ) : ℕ :=
  let blue_left := blue - (4 * blue / 5)
  let brown_left := brown - (5 * brown / 6)
  let red_left := red - (2 * red / 3)
  let yellow_left := yellow - (3 * yellow / 4)
  let green_left := green - (green / 3)
  blue_left + brown_left + red_left + yellow_left + green_left

/-- The theorem stating that Mary has 45 shirts left --/
theorem mary_shirts_left :
  shirts_left 35 48 27 36 18 = 45 := by sorry

end NUMINAMATH_CALUDE_mary_shirts_left_l1131_113184


namespace NUMINAMATH_CALUDE_unique_solution_log_equation_l1131_113112

theorem unique_solution_log_equation :
  ∃! x : ℝ, (Real.log (2 * x + 1) = Real.log (x^2 - 2)) ∧ (2 * x + 1 > 0) ∧ (x^2 - 2 > 0) ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_log_equation_l1131_113112


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1131_113199

theorem perfect_square_trinomial (a k : ℝ) : 
  (∃ b : ℝ, a^2 + 2*k*a + 9 = (a + b)^2) → (k = 3 ∨ k = -3) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1131_113199


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1131_113139

def silverware_cost : ℝ := 20
def dinner_plates_cost_percentage : ℝ := 0.5

theorem total_cost_calculation :
  let dinner_plates_cost := silverware_cost * dinner_plates_cost_percentage
  let total_cost := silverware_cost + dinner_plates_cost
  total_cost = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1131_113139


namespace NUMINAMATH_CALUDE_fraction_equalities_l1131_113186

theorem fraction_equalities (a b : ℚ) (h : a / b = 5 / 6) : 
  ((a + 2 * b) / b = 17 / 6) ∧
  (b / (2 * a - b) = 3 / 2) ∧
  ((a + 3 * b) / (2 * a) = 23 / 10) ∧
  (a / (3 * b) = 5 / 18) ∧
  ((a - 2 * b) / b = -7 / 6) := by
sorry

end NUMINAMATH_CALUDE_fraction_equalities_l1131_113186


namespace NUMINAMATH_CALUDE_number_problem_l1131_113115

theorem number_problem (x : ℝ) : 4 * x = 166.08 → (x / 4) + 0.48 = 10.86 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1131_113115


namespace NUMINAMATH_CALUDE_algal_bloom_characteristics_l1131_113160

/-- Represents a water body -/
structure WaterBody where
  eutrophic : Bool

/-- Represents an algal bloom -/
structure AlgalBloom where
  growthRate : ℝ
  duration : ℝ

/-- Defines what constitutes rapid growth in a short period -/
def isRapidGrowthInShortPeriod (bloom : AlgalBloom) : Prop :=
  bloom.growthRate > 0 ∧ bloom.duration < 1

/-- Theorem: Algal blooms in eutrophic water bodies are characterized by rapid growth in a short period -/
theorem algal_bloom_characteristics (wb : WaterBody) (bloom : AlgalBloom) :
    wb.eutrophic → isRapidGrowthInShortPeriod bloom := by
  sorry

end NUMINAMATH_CALUDE_algal_bloom_characteristics_l1131_113160


namespace NUMINAMATH_CALUDE_product_equals_sum_implies_y_value_l1131_113156

theorem product_equals_sum_implies_y_value :
  ∀ y : ℚ, (2 * 3 * 5 * y = 2 + 3 + 5 + y) → y = 10 / 29 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_sum_implies_y_value_l1131_113156


namespace NUMINAMATH_CALUDE_equation_solution_l1131_113180

theorem equation_solution : ∃ x : ℝ, x * 400 = 173 * 2400 + 125 * 480 / 60 ∧ x = 1039.3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1131_113180


namespace NUMINAMATH_CALUDE_log_sum_equals_three_l1131_113150

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_sum_equals_three : log10 5 + log10 2 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_l1131_113150


namespace NUMINAMATH_CALUDE_intersection_point_theorem_l1131_113172

/-- Given points A and B, and a point C on the line y=x that intersects AB,
    prove that if AC = 2CB, then the y-coordinate of B is 4. -/
theorem intersection_point_theorem (a : ℝ) : 
  let A : ℝ × ℝ := (7, 1)
  let B : ℝ × ℝ := (1, a)
  let C : ℝ × ℝ := (x, x)
  ∃ x : ℝ, (C.1 - A.1, C.2 - A.2) = 2 • (B.1 - C.1, B.2 - C.2) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_theorem_l1131_113172


namespace NUMINAMATH_CALUDE_union_equals_N_l1131_113169

def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | -3 < x ∧ x < 3}

theorem union_equals_N : M ∪ N = N := by sorry

end NUMINAMATH_CALUDE_union_equals_N_l1131_113169


namespace NUMINAMATH_CALUDE_initial_balloon_count_balloon_package_problem_l1131_113100

theorem initial_balloon_count (num_friends : ℕ) (balloons_given_back : ℕ) (final_balloons_per_friend : ℕ) : ℕ :=
  let initial_balloons_per_friend := final_balloons_per_friend + balloons_given_back
  num_friends * initial_balloons_per_friend

theorem balloon_package_problem :
  initial_balloon_count 5 11 39 = 250 := by
  sorry

end NUMINAMATH_CALUDE_initial_balloon_count_balloon_package_problem_l1131_113100


namespace NUMINAMATH_CALUDE_min_shift_for_monotonic_decrease_l1131_113192

open Real

theorem min_shift_for_monotonic_decrease (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = sin (2*x + 2*m + π/6)) →
  (∀ x ∈ [-π/12, 5*π/12], ∀ y ∈ [-π/12, 5*π/12], x < y → f x > f y) →
  m > 0 →
  m ≥ π/4 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_for_monotonic_decrease_l1131_113192


namespace NUMINAMATH_CALUDE_atlantic_charge_proof_l1131_113127

/-- The base rate for United Telephone in dollars -/
def united_base_rate : ℚ := 9

/-- The additional charge per minute for United Telephone in dollars -/
def united_per_minute : ℚ := 1/4

/-- The base rate for Atlantic Call in dollars -/
def atlantic_base_rate : ℚ := 12

/-- The number of minutes for which the bills are equal -/
def equal_minutes : ℕ := 60

/-- The additional charge per minute for Atlantic Call in dollars -/
def atlantic_per_minute : ℚ := 1/5

theorem atlantic_charge_proof :
  united_base_rate + united_per_minute * equal_minutes =
  atlantic_base_rate + atlantic_per_minute * equal_minutes :=
sorry

end NUMINAMATH_CALUDE_atlantic_charge_proof_l1131_113127


namespace NUMINAMATH_CALUDE_parallelogram_exclusive_properties_l1131_113167

structure Parallelogram where
  sides : Fin 4 → ℝ
  angles : Fin 4 → ℝ
  diagonals : Fin 2 → ℝ
  vertex_midpoint_segments : Fin 4 → ℝ
  has_symmetry_axes : Bool
  is_circumscribable : Bool
  is_inscribable : Bool

def all_sides_equal (p : Parallelogram) : Prop :=
  ∀ i j : Fin 4, p.sides i = p.sides j

def all_angles_equal (p : Parallelogram) : Prop :=
  ∀ i j : Fin 4, p.angles i = p.angles j

def all_diagonals_equal (p : Parallelogram) : Prop :=
  p.diagonals 0 = p.diagonals 1

def all_vertex_midpoint_segments_equal (p : Parallelogram) : Prop :=
  ∀ i j : Fin 4, p.vertex_midpoint_segments i = p.vertex_midpoint_segments j

def vertex_midpoint_segments_perpendicular (p : Parallelogram) : Prop :=
  sorry -- This would require more complex geometry definitions

def vertex_midpoint_segments_intersect (p : Parallelogram) : Prop :=
  sorry -- This would require more complex geometry definitions

theorem parallelogram_exclusive_properties (p : Parallelogram) : 
  ¬(all_sides_equal p ∧ all_angles_equal p) ∧
  ¬(all_sides_equal p ∧ all_diagonals_equal p) ∧
  ¬(all_sides_equal p ∧ all_vertex_midpoint_segments_equal p) ∧
  ¬(all_sides_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_sides_equal p ∧ p.has_symmetry_axes) ∧
  ¬(all_sides_equal p ∧ p.is_circumscribable) ∧
  ¬(all_angles_equal p ∧ all_diagonals_equal p) ∧
  ¬(all_angles_equal p ∧ all_vertex_midpoint_segments_equal p) ∧
  ¬(all_angles_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_angles_equal p ∧ p.has_symmetry_axes) ∧
  ¬(all_angles_equal p ∧ p.is_inscribable) ∧
  ¬(all_diagonals_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_diagonals_equal p ∧ p.is_inscribable) ∧
  ¬(all_vertex_midpoint_segments_equal p ∧ vertex_midpoint_segments_perpendicular p) ∧
  ¬(all_vertex_midpoint_segments_equal p ∧ p.is_inscribable) ∧
  ¬(vertex_midpoint_segments_perpendicular p ∧ p.is_circumscribable) := by
  sorry

#check parallelogram_exclusive_properties

end NUMINAMATH_CALUDE_parallelogram_exclusive_properties_l1131_113167


namespace NUMINAMATH_CALUDE_problem_solution_l1131_113103

def f (x : ℝ) : ℝ := |x - 1|

theorem problem_solution :
  (∃ (m : ℝ), m > 0 ∧
    (∀ x, f (x + 5) ≤ 3 * m ↔ -7 ≤ x ∧ x ≤ -1)) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a^2 + b^2 = 3 →
    2 * a * Real.sqrt (1 + b^2) ≤ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a^2 + b^2 = 3 ∧
    2 * a * Real.sqrt (1 + b^2) = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1131_113103


namespace NUMINAMATH_CALUDE_second_article_loss_percentage_l1131_113104

/-- Proves that the loss percentage on the second article is 10% given the specified conditions --/
theorem second_article_loss_percentage
  (cost_price : ℝ)
  (profit_percent_first : ℝ)
  (net_profit_loss_percent : ℝ)
  (h1 : cost_price = 1000)
  (h2 : profit_percent_first = 10)
  (h3 : net_profit_loss_percent = 99.99999999999946) :
  let selling_price_first := cost_price * (1 + profit_percent_first / 100)
  let total_selling_price := 2 * cost_price * (1 + net_profit_loss_percent / 100)
  let selling_price_second := total_selling_price - selling_price_first
  let loss_second := cost_price - selling_price_second
  loss_second / cost_price * 100 = 10 := by
sorry


end NUMINAMATH_CALUDE_second_article_loss_percentage_l1131_113104


namespace NUMINAMATH_CALUDE_first_day_exceeding_500_l1131_113118

def bacteria_growth (n : ℕ) : ℕ := 5 * 3^n

theorem first_day_exceeding_500 :
  (∀ k < 6, bacteria_growth k ≤ 500) ∧ bacteria_growth 6 > 500 := by sorry

end NUMINAMATH_CALUDE_first_day_exceeding_500_l1131_113118


namespace NUMINAMATH_CALUDE_seven_valid_configurations_l1131_113182

/-- A polygon shape made of congruent squares -/
structure SquarePolygon where
  squares : ℕ
  shape : String

/-- Possible positions to attach an additional square -/
def AttachmentPositions : ℕ := 11

/-- A cube with one face missing requires this many squares -/
def CubeSquares : ℕ := 5

/-- The base cross-shaped polygon -/
def baseCross : SquarePolygon :=
  { squares := 6, shape := "cross" }

/-- Predicate for whether a polygon can form a cube with one face missing -/
def canFormCube (p : SquarePolygon) : Prop := sorry

/-- The number of valid configurations that can form a cube with one face missing -/
def validConfigurations : ℕ := 7

/-- Main theorem: There are exactly 7 valid configurations -/
theorem seven_valid_configurations :
  (∃ (configs : Finset SquarePolygon),
    configs.card = validConfigurations ∧
    (∀ p ∈ configs, p.squares = baseCross.squares + 1 ∧ canFormCube p) ∧
    (∀ p : SquarePolygon, p.squares = baseCross.squares + 1 →
      canFormCube p → p ∈ configs)) := by sorry

end NUMINAMATH_CALUDE_seven_valid_configurations_l1131_113182


namespace NUMINAMATH_CALUDE_largest_square_tile_size_l1131_113163

theorem largest_square_tile_size (length width : ℕ) (h1 : length = 378) (h2 : width = 595) :
  ∃ (tile_size : ℕ), tile_size = Nat.gcd length width ∧ tile_size = 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_tile_size_l1131_113163
