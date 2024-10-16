import Mathlib

namespace NUMINAMATH_CALUDE_milo_run_distance_milo_two_hour_run_l4024_402433

/-- Milo's running speed in miles per hour -/
def milo_run_speed : ℝ := 3

/-- Milo's skateboard speed in miles per hour -/
def milo_skateboard_speed : ℝ := 2 * milo_run_speed

/-- Cory's wheelchair speed in miles per hour -/
def cory_wheelchair_speed : ℝ := 12

theorem milo_run_distance : ℝ → ℝ
  | hours => milo_run_speed * hours

theorem milo_two_hour_run : milo_run_distance 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_milo_run_distance_milo_two_hour_run_l4024_402433


namespace NUMINAMATH_CALUDE_parity_of_expression_l4024_402403

theorem parity_of_expression (o n c : ℤ) 
  (ho : Odd o) (hc : Odd c) : Even (o^2 + n*o + c) := by
  sorry

end NUMINAMATH_CALUDE_parity_of_expression_l4024_402403


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4024_402455

theorem inequality_system_solution (x : ℝ) :
  (2 * (x - 1) ≤ x + 1 ∧ (x + 2) / 2 ≥ (x + 3) / 3) ↔ (0 ≤ x ∧ x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4024_402455


namespace NUMINAMATH_CALUDE_total_trees_l4024_402438

/-- Represents the tree-planting task for a school --/
structure TreePlantingTask where
  total : ℕ
  ninth_grade : ℕ
  eighth_grade : ℕ
  seventh_grade : ℕ

/-- The conditions of the tree-planting task --/
def tree_planting_conditions (a : ℕ) (task : TreePlantingTask) : Prop :=
  task.ninth_grade = task.total / 2 ∧
  task.eighth_grade = (task.total - task.ninth_grade) * 2 / 3 ∧
  task.seventh_grade = a ∧
  task.total = task.ninth_grade + task.eighth_grade + task.seventh_grade

/-- The theorem stating that the total number of trees is 6a --/
theorem total_trees (a : ℕ) (task : TreePlantingTask) 
  (h : tree_planting_conditions a task) : task.total = 6 * a :=
sorry

end NUMINAMATH_CALUDE_total_trees_l4024_402438


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l4024_402486

theorem condition_necessary_not_sufficient :
  (∃ a b : ℝ, a + b ≠ 3 ∧ (a = 1 ∧ b = 2)) ∧
  (∀ a b : ℝ, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  (∃ a b : ℝ, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l4024_402486


namespace NUMINAMATH_CALUDE_complex_product_real_iff_condition_l4024_402439

theorem complex_product_real_iff_condition (a b c d : ℝ) :
  let Z1 : ℂ := Complex.mk a b
  let Z2 : ℂ := Complex.mk c d
  (Z1 * Z2).im = 0 ↔ a * d + b * c = 0 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_iff_condition_l4024_402439


namespace NUMINAMATH_CALUDE_min_operations_needed_l4024_402426

-- Define the type for letters
inductive Letter | A | B | C | D | E | F | G

-- Define the type for positions in the circle
inductive Position | Center | Top | TopRight | BottomRight | Bottom | BottomLeft | TopLeft

-- Define the configuration as a function from Position to Letter
def Configuration := Position → Letter

-- Define the initial configuration
def initial_config : Configuration := sorry

-- Define the final configuration
def final_config : Configuration := sorry

-- Define a valid operation
def valid_operation (c : Configuration) : Configuration := sorry

-- Define the number of operations needed to transform one configuration to another
def operations_needed (start finish : Configuration) : ℕ := sorry

-- The main theorem
theorem min_operations_needed :
  operations_needed initial_config final_config = 3 := by sorry

end NUMINAMATH_CALUDE_min_operations_needed_l4024_402426


namespace NUMINAMATH_CALUDE_sequence_property_l4024_402461

theorem sequence_property (a : ℕ → ℤ) (h1 : a 2 = 4)
  (h2 : ∀ n : ℕ, n ≥ 1 → (a (n + 1) - a n : ℚ) < 2^n + 1/2)
  (h3 : ∀ n : ℕ, n ≥ 1 → (a (n + 2) - a n : ℤ) > 3 * 2^n - 1) :
  a 2018 = 2^2018 :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l4024_402461


namespace NUMINAMATH_CALUDE_lcm_5_7_10_14_l4024_402483

theorem lcm_5_7_10_14 : Nat.lcm 5 (Nat.lcm 7 (Nat.lcm 10 14)) = 70 := by
  sorry

end NUMINAMATH_CALUDE_lcm_5_7_10_14_l4024_402483


namespace NUMINAMATH_CALUDE_sum_properties_l4024_402458

theorem sum_properties (c d : ℤ) 
  (hc : ∃ m : ℤ, c = 6 * m) 
  (hd : ∃ n : ℤ, d = 9 * n) : 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ ¬(Even (x + y))) ∧ 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ ¬(∃ k : ℤ, x + y = 6 * k)) ∧ 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ ¬(∃ k : ℤ, x + y = 9 * k)) ∧ 
  (∃ x y : ℤ, (∃ m : ℤ, x = 6 * m) ∧ (∃ n : ℤ, y = 9 * n) ∧ (∃ k : ℤ, x + y = 9 * k)) :=
by sorry

end NUMINAMATH_CALUDE_sum_properties_l4024_402458


namespace NUMINAMATH_CALUDE_triangle_perimeter_21_l4024_402436

-- Define the triangle
def Triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter of a triangle
def Perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem triangle_perimeter_21 :
  ∀ c : ℝ,
  Triangle 10 3 c →
  (Perimeter 10 3 c = 18 ∨ Perimeter 10 3 c = 19 ∨ Perimeter 10 3 c = 20 ∨ Perimeter 10 3 c = 21) →
  Perimeter 10 3 c = 21 :=
by
  sorry

#check triangle_perimeter_21

end NUMINAMATH_CALUDE_triangle_perimeter_21_l4024_402436


namespace NUMINAMATH_CALUDE_m_range_l4024_402457

theorem m_range (m : ℝ) :
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) ↔ (m ≥ 4 ∨ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_m_range_l4024_402457


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4024_402447

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4024_402447


namespace NUMINAMATH_CALUDE_equal_spacing_ratio_l4024_402410

/-- Given 6 equally spaced points on a number line from 0 to 1, 
    the ratio of the 3rd point's value to the 6th point's value is 0.5 -/
theorem equal_spacing_ratio : 
  ∀ (P Q R S T U : ℝ), 
    0 ≤ P ∧ P < Q ∧ Q < R ∧ R < S ∧ S < T ∧ T < U ∧ U = 1 →
    Q - P = R - Q ∧ R - Q = S - R ∧ S - R = T - S ∧ T - S = U - T →
    R / U = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_spacing_ratio_l4024_402410


namespace NUMINAMATH_CALUDE_root_equation_value_l4024_402488

theorem root_equation_value (a : ℝ) : 
  a^2 - 3*a - 1011 = 0 → 2*a^2 - 6*a + 1 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l4024_402488


namespace NUMINAMATH_CALUDE_freds_walking_speed_l4024_402464

/-- Proves that Fred's walking speed is 4 miles per hour given the initial conditions -/
theorem freds_walking_speed 
  (initial_distance : ℝ) 
  (sams_speed : ℝ) 
  (sams_distance : ℝ) 
  (h1 : initial_distance = 40) 
  (h2 : sams_speed = 4) 
  (h3 : sams_distance = 20) : 
  (initial_distance - sams_distance) / (sams_distance / sams_speed) = 4 :=
by sorry

end NUMINAMATH_CALUDE_freds_walking_speed_l4024_402464


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_III_l4024_402449

/-- A linear function defined by y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents the four quadrants of a 2D coordinate system -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Checks if a point (x, y) is in Quadrant III -/
def isInQuadrantIII (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

/-- Theorem: The linear function y = -2x + 1 does not pass through Quadrant III -/
theorem linear_function_not_in_quadrant_III :
  ∀ x y : ℝ, y = -2 * x + 1 → ¬(isInQuadrantIII x y) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_III_l4024_402449


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_fourths_l4024_402401

theorem reciprocal_of_negative_three_fourths :
  let x : ℚ := -3/4
  let y : ℚ := -4/3
  (x * y = 1) → (y = x⁻¹) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_fourths_l4024_402401


namespace NUMINAMATH_CALUDE_problem_solution_l4024_402418

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0

-- Theorem to prove
theorem problem_solution : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l4024_402418


namespace NUMINAMATH_CALUDE_no_solutions_to_equation_l4024_402434

theorem no_solutions_to_equation :
  ∀ x : ℝ, x ≠ 0 → x ≠ 5 → (2 * x^2 - 10 * x) / (x^2 - 5 * x) ≠ x - 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_to_equation_l4024_402434


namespace NUMINAMATH_CALUDE_ann_age_l4024_402466

/-- Ann's age in years -/
def A : ℕ := sorry

/-- Susan's age in years -/
def S : ℕ := sorry

/-- Ann is 5 years older than Susan -/
axiom age_difference : A = S + 5

/-- The sum of their ages is 27 -/
axiom age_sum : A + S = 27

/-- Prove that Ann is 16 years old -/
theorem ann_age : A = 16 := by sorry

end NUMINAMATH_CALUDE_ann_age_l4024_402466


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l4024_402444

theorem max_value_of_sum_of_squares (x y : ℝ) :
  x^2 + y^2 = 3*x + 8*y → x^2 + y^2 ≤ 73 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_squares_l4024_402444


namespace NUMINAMATH_CALUDE_arithmetic_contains_geometric_l4024_402431

-- Define the arithmetic progression
def arithmetic_progression (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem statement
theorem arithmetic_contains_geometric (a d : ℕ) (h : d > 0) :
  ∃ (r : ℕ) (b : ℕ → ℕ), 
    (∀ n : ℕ, ∃ k : ℕ, b n = arithmetic_progression a d k) ∧
    (∀ n : ℕ, b (n + 1) = r * b n) ∧
    (∀ n : ℕ, b n > 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_contains_geometric_l4024_402431


namespace NUMINAMATH_CALUDE_remaining_gift_cards_value_l4024_402437

/-- Represents the types of gift cards --/
inductive GiftCardType
  | BestBuy
  | Target
  | Walmart
  | Amazon

/-- Represents a gift card with its type and value --/
structure GiftCard where
  type : GiftCardType
  value : Nat

def initial_gift_cards : List GiftCard := [
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.Target, value := 250 },
  { type := GiftCardType.Target, value := 250 },
  { type := GiftCardType.Target, value := 250 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Amazon, value := 1000 },
  { type := GiftCardType.Amazon, value := 1000 }
]

def sent_gift_cards : List GiftCard := [
  { type := GiftCardType.BestBuy, value := 500 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Walmart, value := 100 },
  { type := GiftCardType.Amazon, value := 1000 }
]

theorem remaining_gift_cards_value : 
  (List.sum (initial_gift_cards.map (λ g => g.value)) - 
   List.sum (sent_gift_cards.map (λ g => g.value))) = 4250 := by
  sorry

end NUMINAMATH_CALUDE_remaining_gift_cards_value_l4024_402437


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_five_l4024_402477

theorem smallest_n_multiple_of_five (x y : ℤ) 
  (hx : ∃ k : ℤ, x + 2 = 5 * k) 
  (hy : ∃ k : ℤ, y - 2 = 5 * k) : 
  (∃ n : ℕ+, ∃ m : ℤ, x^2 + x*y + y^2 + n = 5 * m ∧ 
   ∀ k : ℕ+, k < n → ¬∃ m : ℤ, x^2 + x*y + y^2 + k = 5 * m) → 
  (∃ n : ℕ+, n = 1 ∧ ∃ m : ℤ, x^2 + x*y + y^2 + n = 5 * m ∧ 
   ∀ k : ℕ+, k < n → ¬∃ m : ℤ, x^2 + x*y + y^2 + k = 5 * m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_five_l4024_402477


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l4024_402493

theorem max_value_of_fraction (k : ℝ) (h : k > 0) :
  (3 * k^3 + 3 * k) / ((3/2 * k^2 + 14) * (14 * k^2 + 3/2)) ≤ Real.sqrt 21 / 175 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l4024_402493


namespace NUMINAMATH_CALUDE_savings_calculation_l4024_402424

/-- Given an income and an income-to-expenditure ratio, calculate the savings -/
def calculate_savings (income : ℚ) (income_ratio : ℚ) (expenditure_ratio : ℚ) : ℚ :=
  income - (income * expenditure_ratio / income_ratio)

/-- Theorem: For an income of 21000 and an income-to-expenditure ratio of 7:6, the savings is 3000 -/
theorem savings_calculation :
  calculate_savings 21000 7 6 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l4024_402424


namespace NUMINAMATH_CALUDE_remainder_divisibility_l4024_402442

theorem remainder_divisibility (N : ℕ) (h : N > 0) (h1 : N % 60 = 49) : N % 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l4024_402442


namespace NUMINAMATH_CALUDE_lg_2_plus_lg_5_equals_1_l4024_402472

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_2_plus_lg_5_equals_1 : lg 2 + lg 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_lg_2_plus_lg_5_equals_1_l4024_402472


namespace NUMINAMATH_CALUDE_symmetric_curves_l4024_402432

/-- The original curve E -/
def E (x y : ℝ) : Prop :=
  5 * x^2 + 12 * x * y - 22 * x - 12 * y - 19 = 0

/-- The line of symmetry l -/
def l (x y : ℝ) : Prop :=
  x - y + 2 = 0

/-- The symmetric curve E' -/
def E' (x y : ℝ) : Prop :=
  12 * x * y + 5 * y^2 - 78 * y + 45 = 0

/-- Theorem stating that E' is symmetric to E with respect to l -/
theorem symmetric_curves : ∀ (x y x' y' : ℝ),
  l ((x + x') / 2) ((y + y') / 2) →
  E x y ↔ E' x' y' :=
sorry

end NUMINAMATH_CALUDE_symmetric_curves_l4024_402432


namespace NUMINAMATH_CALUDE_goldfinch_percentage_l4024_402453

/-- The number of goldfinches -/
def goldfinches : ℕ := 6

/-- The number of sparrows -/
def sparrows : ℕ := 9

/-- The number of grackles -/
def grackles : ℕ := 5

/-- The total number of birds -/
def total_birds : ℕ := goldfinches + sparrows + grackles

/-- The fraction of goldfinches -/
def goldfinch_fraction : ℚ := goldfinches / total_birds

/-- Theorem: The percentage of goldfinches is 30% -/
theorem goldfinch_percentage :
  goldfinch_fraction * 100 = 30 := by sorry

end NUMINAMATH_CALUDE_goldfinch_percentage_l4024_402453


namespace NUMINAMATH_CALUDE_can_obtain_11_from_1_l4024_402421

/-- Represents the allowed operations on the calculator -/
inductive Operation
  | MultiplyBy3
  | Add3
  | DivideBy3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.MultiplyBy3 => n * 3
  | Operation.Add3 => n + 3
  | Operation.DivideBy3 => if n % 3 = 0 then n / 3 else n

/-- Applies a sequence of operations to a number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- Theorem stating that 11 can be obtained from 1 using the allowed operations -/
theorem can_obtain_11_from_1 : ∃ (ops : List Operation), applyOperations 1 ops = 11 :=
  sorry

end NUMINAMATH_CALUDE_can_obtain_11_from_1_l4024_402421


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l4024_402459

/-- Given a line with y-intercept 3 and slope -3/2, prove that the product of its slope and y-intercept is -9/2 -/
theorem line_slope_intercept_product :
  ∀ (m b : ℚ),
    b = 3 →
    m = -3/2 →
    m * b = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l4024_402459


namespace NUMINAMATH_CALUDE_purple_chip_count_l4024_402468

theorem purple_chip_count (blue green purple red : ℕ) (x : ℕ) :
  blue > 0 → green > 0 → purple > 0 → red > 0 →
  5 < x → x < 11 →
  1^blue * 5^green * x^purple * 11^red = 140800 →
  purple = 1 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_purple_chip_count_l4024_402468


namespace NUMINAMATH_CALUDE_max_green_beads_l4024_402427

/-- A necklace with red, blue, and green beads. -/
structure Necklace :=
  (total : ℕ)
  (red : Finset ℕ)
  (blue : Finset ℕ)
  (green : Finset ℕ)

/-- The necklace satisfies the problem conditions. -/
def ValidNecklace (n : Necklace) : Prop :=
  n.total = 100 ∧
  n.red ∪ n.blue ∪ n.green = Finset.range n.total ∧
  (∀ i : ℕ, ∃ j ∈ n.blue, j % n.total ∈ Finset.range 5 ∪ {n.total - 1, n.total - 2, n.total - 3, n.total - 4}) ∧
  (∀ i : ℕ, ∃ j ∈ n.red, j % n.total ∈ Finset.range 7 ∪ {n.total - 1, n.total - 2, n.total - 3, n.total - 4, n.total - 5, n.total - 6})

/-- The maximum number of green beads in a valid necklace. -/
theorem max_green_beads (n : Necklace) (h : ValidNecklace n) :
  n.green.card ≤ 65 :=
sorry

end NUMINAMATH_CALUDE_max_green_beads_l4024_402427


namespace NUMINAMATH_CALUDE_math_fun_books_count_l4024_402489

theorem math_fun_books_count : ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ 18 * x + 8 * y = 92 ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_math_fun_books_count_l4024_402489


namespace NUMINAMATH_CALUDE_shelves_filled_with_carvings_l4024_402430

def wood_carvings_per_shelf : ℕ := 8
def total_wood_carvings : ℕ := 56

theorem shelves_filled_with_carvings :
  total_wood_carvings / wood_carvings_per_shelf = 7 := by
  sorry

end NUMINAMATH_CALUDE_shelves_filled_with_carvings_l4024_402430


namespace NUMINAMATH_CALUDE_female_turtle_percentage_is_60_l4024_402497

/-- Represents the number of turtles in the lake -/
def total_turtles : ℕ := 100

/-- Represents the fraction of male turtles that have stripes -/
def male_stripe_ratio : ℚ := 1 / 4

/-- Represents the number of baby striped male turtles -/
def baby_striped_males : ℕ := 4

/-- Represents the percentage of adult striped male turtles -/
def adult_striped_male_percentage : ℚ := 60 / 100

/-- Calculates the percentage of female turtles in the lake -/
def female_turtle_percentage : ℚ :=
  let total_striped_males : ℚ := baby_striped_males / (1 - adult_striped_male_percentage)
  let total_males : ℚ := total_striped_males / male_stripe_ratio
  let total_females : ℚ := total_turtles - total_males
  (total_females / total_turtles) * 100

theorem female_turtle_percentage_is_60 :
  female_turtle_percentage = 60 := by sorry

end NUMINAMATH_CALUDE_female_turtle_percentage_is_60_l4024_402497


namespace NUMINAMATH_CALUDE_marnie_bracelets_l4024_402487

def beads_per_bracelet : ℕ := 65

def total_beads : ℕ :=
  5 * 50 + 2 * 100 + 3 * 75 + 4 * 125

theorem marnie_bracelets :
  (total_beads / beads_per_bracelet : ℕ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_marnie_bracelets_l4024_402487


namespace NUMINAMATH_CALUDE_marble_probability_l4024_402406

theorem marble_probability : 
  let green : ℕ := 4
  let white : ℕ := 3
  let red : ℕ := 5
  let blue : ℕ := 6
  let total : ℕ := green + white + red + blue
  let favorable : ℕ := green + white
  (favorable : ℚ) / total = 7 / 18 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l4024_402406


namespace NUMINAMATH_CALUDE_school_count_correct_l4024_402481

/-- Represents the number of primary schools in a town. -/
def num_schools : ℕ := 4

/-- Represents the capacity of the first two schools. -/
def capacity_large : ℕ := 400

/-- Represents the capacity of the other two schools. -/
def capacity_small : ℕ := 340

/-- Represents the total capacity of all schools. -/
def total_capacity : ℕ := 1480

/-- Theorem stating that the number of schools is correct given the capacities. -/
theorem school_count_correct : 
  2 * capacity_large + 2 * capacity_small = total_capacity ∧
  num_schools = 2 + 2 := by sorry

end NUMINAMATH_CALUDE_school_count_correct_l4024_402481


namespace NUMINAMATH_CALUDE_overall_loss_percentage_is_about_2_09_percent_l4024_402435

/-- Represents an appliance with its cost price and profit/loss percentage -/
structure Appliance where
  costPrice : ℕ
  profitLossPercentage : ℤ

/-- Calculates the selling price of an appliance -/
def sellingPrice (a : Appliance) : ℚ :=
  a.costPrice * (1 + a.profitLossPercentage / 100)

/-- The list of appliances with their cost prices and profit/loss percentages -/
def appliances : List Appliance := [
  ⟨15000, -5⟩,
  ⟨8000, 10⟩,
  ⟨12000, -8⟩,
  ⟨10000, 15⟩,
  ⟨5000, 7⟩,
  ⟨20000, -12⟩
]

/-- The total cost price of all appliances -/
def totalCostPrice : ℕ := (appliances.map (·.costPrice)).sum

/-- The total selling price of all appliances -/
def totalSellingPrice : ℚ := (appliances.map sellingPrice).sum

/-- The overall loss percentage -/
def overallLossPercentage : ℚ :=
  (totalCostPrice - totalSellingPrice) / totalCostPrice * 100

/-- Theorem stating that the overall loss percentage is approximately 2.09% -/
theorem overall_loss_percentage_is_about_2_09_percent :
  abs (overallLossPercentage - 2.09) < 0.01 := by sorry

end NUMINAMATH_CALUDE_overall_loss_percentage_is_about_2_09_percent_l4024_402435


namespace NUMINAMATH_CALUDE_tom_crab_price_l4024_402443

/-- A crab seller's weekly income and catch details -/
structure CrabSeller where
  buckets : ℕ
  crabs_per_bucket : ℕ
  days_per_week : ℕ
  weekly_income : ℕ

/-- Calculate the price per crab for a crab seller -/
def price_per_crab (seller : CrabSeller) : ℚ :=
  seller.weekly_income / (seller.buckets * seller.crabs_per_bucket * seller.days_per_week)

/-- Tom's crab selling business -/
def tom : CrabSeller :=
  { buckets := 8
    crabs_per_bucket := 12
    days_per_week := 7
    weekly_income := 3360 }

/-- Theorem stating that Tom sells each crab for $5 -/
theorem tom_crab_price : price_per_crab tom = 5 := by
  sorry


end NUMINAMATH_CALUDE_tom_crab_price_l4024_402443


namespace NUMINAMATH_CALUDE_triangle_count_is_38_l4024_402419

/-- Represents a rectangle with internal divisions as described in the problem -/
structure DividedRectangle where
  -- Add necessary fields to represent the rectangle and its divisions
  -- This is a simplified representation
  height : ℕ
  width : ℕ

/-- Counts the number of triangles in a DividedRectangle -/
def countTriangles (rect : DividedRectangle) : ℕ := sorry

/-- The specific rectangle from the problem -/
def problemRectangle : DividedRectangle := {
  height := 20,
  width := 30
}

/-- Theorem stating that the number of triangles in the problem rectangle is 38 -/
theorem triangle_count_is_38 : countTriangles problemRectangle = 38 := by sorry

end NUMINAMATH_CALUDE_triangle_count_is_38_l4024_402419


namespace NUMINAMATH_CALUDE_distance_to_axes_l4024_402476

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- The distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Theorem stating the distances from point P(3,5) to the x-axis and y-axis -/
theorem distance_to_axes :
  let P : Point := ⟨3, 5⟩
  distanceToXAxis P = 5 ∧ distanceToYAxis P = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_axes_l4024_402476


namespace NUMINAMATH_CALUDE_unique_triple_solution_l4024_402407

theorem unique_triple_solution :
  ∃! (s : Set (ℝ × ℝ × ℝ)), 
    (∀ (x y z : ℝ), (x, y, z) ∈ s ↔ 
      (1 + x^4 ≤ 2*(y - z)^2 ∧
       1 + y^4 ≤ 2*(z - x)^2 ∧
       1 + z^4 ≤ 2*(x - y)^2)) ∧
    (s = {(1, 0, -1), (1, -1, 0), (0, 1, -1), (0, -1, 1), (-1, 1, 0), (-1, 0, 1)}) :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l4024_402407


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4024_402470

theorem inequality_equivalence (x : ℝ) :
  (3 * x - 4 < 9 - 2 * x + |x - 1|) ↔ (x < 3 ∧ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4024_402470


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l4024_402474

theorem angle_terminal_side_point (θ : Real) (a : Real) : 
  (2 * Real.sin (π / 8) ^ 2 - 1, a) ∈ Set.range (λ t : Real => (Real.cos t, Real.sin t)) →
  Real.sin θ = 2 * Real.sqrt 3 * Real.sin (13 * π / 12) * Real.cos (π / 12) →
  a = -Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l4024_402474


namespace NUMINAMATH_CALUDE_ed_marbles_l4024_402471

theorem ed_marbles (doug_initial : ℕ) (ed_more : ℕ) (doug_lost : ℕ) : 
  doug_initial = 22 → ed_more = 5 → doug_lost = 3 →
  doug_initial + ed_more = 27 :=
by sorry

end NUMINAMATH_CALUDE_ed_marbles_l4024_402471


namespace NUMINAMATH_CALUDE_arithmetic_progression_equality_l4024_402495

theorem arithmetic_progression_equality (n : ℕ) 
  (a b : Fin n → ℕ) 
  (h_n : n ≥ 2018) 
  (h_distinct_a : ∀ i j : Fin n, i ≠ j → a i ≠ a j)
  (h_distinct_b : ∀ i j : Fin n, i ≠ j → b i ≠ b j)
  (h_bound_a : ∀ i : Fin n, a i ≤ 5*n)
  (h_bound_b : ∀ i : Fin n, b i ≤ 5*n)
  (h_positive_a : ∀ i : Fin n, a i > 0)
  (h_positive_b : ∀ i : Fin n, b i > 0)
  (h_arithmetic : ∃ d : ℚ, ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) - (a j : ℚ) / (b j : ℚ) = (i.val - j.val : ℚ) * d) :
  ∀ i j : Fin n, (a i : ℚ) / (b i : ℚ) = (a j : ℚ) / (b j : ℚ) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_equality_l4024_402495


namespace NUMINAMATH_CALUDE_square_difference_l4024_402469

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) : 
  (x - y)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l4024_402469


namespace NUMINAMATH_CALUDE_johns_breakfast_calories_l4024_402425

/-- Represents the number of calories in John's breakfast -/
def breakfast_calories : ℝ := 500

/-- Represents the number of calories in John's lunch -/
def lunch_calories : ℝ := 1.25 * breakfast_calories

/-- Represents the number of calories in John's dinner -/
def dinner_calories : ℝ := 2 * lunch_calories

/-- Represents the total number of calories from shakes -/
def shake_calories : ℝ := 3 * 300

/-- Represents the total number of calories John consumes in a day -/
def total_calories : ℝ := 3275

/-- Theorem stating that given the conditions, John's breakfast contains 500 calories -/
theorem johns_breakfast_calories :
  breakfast_calories + lunch_calories + dinner_calories + shake_calories = total_calories :=
by sorry

end NUMINAMATH_CALUDE_johns_breakfast_calories_l4024_402425


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l4024_402475

theorem sum_of_numbers_in_ratio (x y z : ℝ) : 
  y = 2 * x ∧ z = 5 * x ∧ x^2 + y^2 + z^2 = 4320 → x + y + z = 96 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l4024_402475


namespace NUMINAMATH_CALUDE_integral_x_squared_l4024_402452

theorem integral_x_squared : ∫ x in (-1)..1, x^2 = 2/3 := by sorry

end NUMINAMATH_CALUDE_integral_x_squared_l4024_402452


namespace NUMINAMATH_CALUDE_system_solution_l4024_402440

theorem system_solution : 
  ∀ x y : ℝ, 
    (y + Real.sqrt (y - 3*x) + 3*x = 12 ∧ 
     y^2 + y - 3*x - 9*x^2 = 144) ↔ 
    ((x = -24 ∧ y = 72) ∨ (x = -4/3 ∧ y = 12)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4024_402440


namespace NUMINAMATH_CALUDE_percentage_square_root_l4024_402454

theorem percentage_square_root (x : ℝ) : 
  Real.sqrt (x / 100) = 20 → x = 20 := by sorry

end NUMINAMATH_CALUDE_percentage_square_root_l4024_402454


namespace NUMINAMATH_CALUDE_range_start_divisible_by_eleven_l4024_402480

theorem range_start_divisible_by_eleven : ∃ (start : ℕ), 
  (start ≤ 79) ∧ 
  (∃ (a b c d : ℕ), 
    (start = 11 * a) ∧ 
    (start + 11 = 11 * b) ∧ 
    (start + 22 = 11 * c) ∧ 
    (start + 33 = 11 * d) ∧ 
    (start + 33 ≤ 79) ∧
    (start + 44 > 79)) ∧
  (start = 44) := by
sorry

end NUMINAMATH_CALUDE_range_start_divisible_by_eleven_l4024_402480


namespace NUMINAMATH_CALUDE_angle_measure_from_cosine_l4024_402478

theorem angle_measure_from_cosine (A : Real) : 
  0 < A → A < Real.pi / 2 → -- A is acute
  Real.cos A = Real.sqrt 3 / 2 → -- cos A = √3/2
  A = Real.pi / 6 -- A = 30° (π/6 radians)
:= by sorry

end NUMINAMATH_CALUDE_angle_measure_from_cosine_l4024_402478


namespace NUMINAMATH_CALUDE_seahawks_touchdowns_l4024_402409

theorem seahawks_touchdowns 
  (total_points : ℕ)
  (field_goals : ℕ)
  (touchdown_points : ℕ)
  (field_goal_points : ℕ)
  (h1 : total_points = 37)
  (h2 : field_goals = 3)
  (h3 : touchdown_points = 7)
  (h4 : field_goal_points = 3) :
  (total_points - field_goals * field_goal_points) / touchdown_points = 4 := by
  sorry

end NUMINAMATH_CALUDE_seahawks_touchdowns_l4024_402409


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l4024_402499

theorem initial_number_of_persons (n : ℕ) 
  (h1 : (3.5 : ℝ) * n = 28)
  (h2 : (90 : ℝ) - 62 = 28) : 
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l4024_402499


namespace NUMINAMATH_CALUDE_job_completion_time_l4024_402422

/-- Given that 5/8 of a job is completed in 10 days at a constant pace, 
    prove that the entire job will be completed in 16 days. -/
theorem job_completion_time (days_for_part : ℚ) (part_completed : ℚ) (total_days : ℕ) : 
  days_for_part = 10 → part_completed = 5/8 → total_days = 16 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l4024_402422


namespace NUMINAMATH_CALUDE_cos_four_theta_value_l4024_402416

theorem cos_four_theta_value (θ : ℝ) (h : ∑' n, (Real.cos θ)^(2*n) = 8) :
  Real.cos (4 * θ) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_theta_value_l4024_402416


namespace NUMINAMATH_CALUDE_large_triangle_perimeter_l4024_402404

/-- An isosceles triangle with two sides of length 12 and one side of length 14 -/
structure SmallTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : side1 = side2 ∧ side1 = 12 ∧ side3 = 14

/-- A triangle similar to the small triangle with longest side 42 -/
structure LargeTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  similar_to_small : ∃ (k : ℝ), side1 = k * 12 ∧ side2 = k * 12 ∧ side3 = k * 14
  longest_side : side3 = 42

/-- The perimeter of the large triangle is 114 -/
theorem large_triangle_perimeter (small : SmallTriangle) (large : LargeTriangle) :
  large.side1 + large.side2 + large.side3 = 114 := by
  sorry

end NUMINAMATH_CALUDE_large_triangle_perimeter_l4024_402404


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l4024_402412

/-- Given an ellipse and hyperbola with common foci F₁ and F₂, intersecting at point P -/
structure EllipseHyperbolaIntersection where
  /-- The eccentricity of the ellipse -/
  e₁ : ℝ
  /-- The eccentricity of the hyperbola -/
  e₂ : ℝ
  /-- Angle F₁PF₂ -/
  angle_F₁PF₂ : ℝ
  /-- 0 < e₁ < 1 (eccentricity of ellipse) -/
  h₁ : 0 < e₁ ∧ e₁ < 1
  /-- e₂ > 1 (eccentricity of hyperbola) -/
  h₂ : e₂ > 1
  /-- cos ∠F₁PF₂ = 3/5 -/
  h₃ : Real.cos angle_F₁PF₂ = 3/5
  /-- e₂ = 2e₁ -/
  h₄ : e₂ = 2 * e₁

/-- The eccentricity of the ellipse is √10/5 -/
theorem ellipse_eccentricity (eh : EllipseHyperbolaIntersection) : eh.e₁ = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l4024_402412


namespace NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l4024_402451

/-- Proves the relationship between inverse proportionality and percentage changes -/
theorem inverse_proportion_percentage_change 
  (x y x' y' q k : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x * y = k) 
  (h4 : x' * y' = k) 
  (h5 : x' = x * (1 - q / 100)) 
  (h6 : q > 0) 
  (h7 : q < 100) : 
  y' = y * (1 + (100 * q) / (100 - q) / 100) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l4024_402451


namespace NUMINAMATH_CALUDE_floor_length_approx_l4024_402448

/-- Represents the dimensions and painting cost of a rectangular floor. -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ
  paintingCost : ℝ
  paintingRate : ℝ

/-- The length of the floor is 200% more than the breadth. -/
def lengthCondition (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth

/-- The total cost to paint the floor is 624 Rs. -/
def totalCostCondition (floor : RectangularFloor) : Prop :=
  floor.paintingCost = 624

/-- The painting rate is 4 Rs per square meter. -/
def paintingRateCondition (floor : RectangularFloor) : Prop :=
  floor.paintingRate = 4

/-- The theorem stating that the length of the floor is approximately 21.63 meters. -/
theorem floor_length_approx (floor : RectangularFloor) 
  (h1 : lengthCondition floor)
  (h2 : totalCostCondition floor)
  (h3 : paintingRateCondition floor) :
  ∃ ε > 0, |floor.length - 21.63| < ε :=
sorry

end NUMINAMATH_CALUDE_floor_length_approx_l4024_402448


namespace NUMINAMATH_CALUDE_circles_intersection_sum_l4024_402498

/-- Given two circles intersecting at points (1,3) and (m,1), with their centers 
    on the line 2x-y+c=0, prove that m + c = 1 -/
theorem circles_intersection_sum (m c : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    -- Centers of circles lie on the line 2x-y+c=0
    (2 * x₁ - y₁ + c = 0) ∧ 
    (2 * x₂ - y₂ + c = 0) ∧ 
    -- Circles intersect at (1,3) and (m,1)
    ((x₁ - 1)^2 + (y₁ - 3)^2 = (x₁ - m)^2 + (y₁ - 1)^2) ∧
    ((x₂ - 1)^2 + (y₂ - 3)^2 = (x₂ - m)^2 + (y₂ - 1)^2)) →
  m + c = 1 := by
sorry

end NUMINAMATH_CALUDE_circles_intersection_sum_l4024_402498


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4024_402473

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 4 * x + y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 4 * x' + y' = 1 → 1 / x' + 4 / y' ≥ 1 / x + 4 / y) →
  1 / x + 4 / y = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4024_402473


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l4024_402492

/-- The area of the circle represented by the polar equation r = 3 cos θ - 4 sin θ -/
theorem circle_area_from_polar_equation : 
  let r : ℝ → ℝ := λ θ => 3 * Real.cos θ - 4 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (∀ θ, (r θ * Real.cos θ - center.1)^2 + (r θ * Real.sin θ - center.2)^2 = radius^2) ∧
    Real.pi * radius^2 = 25 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l4024_402492


namespace NUMINAMATH_CALUDE_complement_of_union_is_four_l4024_402460

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3}

-- Theorem statement
theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_is_four_l4024_402460


namespace NUMINAMATH_CALUDE_boat_upstream_time_l4024_402494

/-- Proves that the time taken by a boat to cover a distance upstream is 1.5 hours,
    given the conditions of the problem. -/
theorem boat_upstream_time (distance : ℝ) (stream_speed : ℝ) (boat_speed : ℝ) : 
  stream_speed = 3 →
  boat_speed = 15 →
  distance = (boat_speed + stream_speed) * 1 →
  (distance / (boat_speed - stream_speed)) = 1.5 := by
sorry

end NUMINAMATH_CALUDE_boat_upstream_time_l4024_402494


namespace NUMINAMATH_CALUDE_cookie_jar_spending_ratio_l4024_402408

theorem cookie_jar_spending_ratio (initial_amount : ℕ) (doris_spent : ℕ) (final_amount : ℕ) : 
  initial_amount = 21 →
  doris_spent = 6 →
  final_amount = 12 →
  (initial_amount - doris_spent - final_amount) / doris_spent = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_spending_ratio_l4024_402408


namespace NUMINAMATH_CALUDE_hall_length_l4024_402491

/-- Given a rectangular hall where the length is 5 meters more than the breadth
    and the area is 750 square meters, prove that the length is 30 meters. -/
theorem hall_length (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = breadth + 5 →
  area = length * breadth →
  area = 750 →
  length = 30 := by
sorry

end NUMINAMATH_CALUDE_hall_length_l4024_402491


namespace NUMINAMATH_CALUDE_problem_solution_l4024_402414

-- Define p₁
def p₁ : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define p₂
def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - 1 ≥ 0

-- Theorem to prove
theorem problem_solution : (¬p₁) ∨ (¬p₂) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4024_402414


namespace NUMINAMATH_CALUDE_book_club_single_people_count_l4024_402484

/-- Represents a book club with members and book selection turns. -/
structure BookClub where
  total_turns : ℕ  -- Total number of turns per year
  couple_count : ℕ  -- Number of couples in the club
  ron_turns : ℕ  -- Number of turns Ron gets per year

/-- Calculates the number of single people in the book club. -/
def single_people_count (club : BookClub) : ℕ :=
  club.total_turns - (club.couple_count + 1)

/-- Theorem stating that the number of single people in the given book club is 9. -/
theorem book_club_single_people_count :
  ∃ (club : BookClub),
    club.total_turns = 52 / 4 ∧
    club.couple_count = 3 ∧
    club.ron_turns = 4 ∧
    single_people_count club = 9 := by
  sorry

end NUMINAMATH_CALUDE_book_club_single_people_count_l4024_402484


namespace NUMINAMATH_CALUDE_tan_function_property_l4024_402405

open Real

theorem tan_function_property (φ a : ℝ) (h1 : π / 2 < φ) (h2 : φ < 3 * π / 2) : 
  let f := fun x => tan (φ - x)
  (f 0 = 0) → (f (-a) = 1 / 2) → (f (a + π / 4) = -3) := by
  sorry

end NUMINAMATH_CALUDE_tan_function_property_l4024_402405


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l4024_402485

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l4024_402485


namespace NUMINAMATH_CALUDE_total_inflation_time_inflation_time_proof_l4024_402445

/-- Calculates the total time taken to inflate soccer balls -/
theorem total_inflation_time (alexia_time ermias_time leila_time : ℕ)
  (alexia_balls : ℕ) (ermias_extra leila_fewer : ℕ) : ℕ :=
  let alexia_total := alexia_time * alexia_balls
  let ermias_balls := alexia_balls + ermias_extra
  let ermias_total := ermias_time * ermias_balls
  let leila_balls := ermias_balls - leila_fewer
  let leila_total := leila_time * leila_balls
  alexia_total + ermias_total + leila_total

/-- Proves that the total time taken to inflate all soccer balls is 4160 minutes -/
theorem inflation_time_proof :
  total_inflation_time 18 25 30 50 12 5 = 4160 := by
  sorry


end NUMINAMATH_CALUDE_total_inflation_time_inflation_time_proof_l4024_402445


namespace NUMINAMATH_CALUDE_domino_partition_exists_l4024_402462

/-- Represents a domino piece with two numbers -/
structure Domino :=
  (a b : Nat)
  (h1 : a ≤ 6)
  (h2 : b ≤ 6)

/-- The set of all domino pieces in a standard double-six set -/
def dominoSet : Finset Domino :=
  sorry

/-- The sum of points on all domino pieces -/
def totalSum : Nat :=
  sorry

/-- A partition of the domino set into 4 groups -/
def Partition := Fin 4 → Finset Domino

theorem domino_partition_exists :
  ∃ (p : Partition),
    (∀ i j, i ≠ j → Disjoint (p i) (p j)) ∧
    (∀ i, (p i).sum (λ d => d.a + d.b) = 21) ∧
    (∀ d ∈ dominoSet, ∃ i, d ∈ p i) :=
  sorry

end NUMINAMATH_CALUDE_domino_partition_exists_l4024_402462


namespace NUMINAMATH_CALUDE_sqrt_product_equals_140_l4024_402467

theorem sqrt_product_equals_140 :
  Real.sqrt (13 + Real.sqrt (28 + Real.sqrt 281)) *
  Real.sqrt (13 - Real.sqrt (28 + Real.sqrt 281)) *
  Real.sqrt (141 + Real.sqrt 281) = 140 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_140_l4024_402467


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_0_2_l4024_402465

theorem arithmetic_square_root_of_0_2 : ∃ x : ℝ, x^2 = 0.2 ∧ x ≠ 0.02 := by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_0_2_l4024_402465


namespace NUMINAMATH_CALUDE_cubic_equation_root_l4024_402411

theorem cubic_equation_root (a b : ℚ) :
  (2 + Real.sqrt 3 : ℝ) ^ 3 + a * (2 + Real.sqrt 3 : ℝ) ^ 2 + b * (2 + Real.sqrt 3 : ℝ) - 12 = 0 →
  b = -47 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l4024_402411


namespace NUMINAMATH_CALUDE_min_sum_of_product_l4024_402429

theorem min_sum_of_product (a b : ℤ) (h : a * b = 72) : 
  ∀ (x y : ℤ), x * y = 72 → a + b ≤ x + y ∧ ∃ (a₀ b₀ : ℤ), a₀ * b₀ = 72 ∧ a₀ + b₀ = -73 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l4024_402429


namespace NUMINAMATH_CALUDE_triangle_area_l4024_402456

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  b = 6 →
  a = 2 * c →
  B = π / 3 →
  (1 / 2) * a * c * Real.sin B = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4024_402456


namespace NUMINAMATH_CALUDE_buratino_betting_strategy_l4024_402441

theorem buratino_betting_strategy :
  ∃ (x₁ x₂ x₃ y : ℕ+),
    x₁ + x₂ + x₃ + y = 20 ∧
    5 * x₁ + y ≥ 21 ∧
    4 * x₂ + y ≥ 21 ∧
    2 * x₃ + y ≥ 21 :=
by sorry

end NUMINAMATH_CALUDE_buratino_betting_strategy_l4024_402441


namespace NUMINAMATH_CALUDE_sin_greater_cos_range_l4024_402420

theorem sin_greater_cos_range (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  (Real.sin x > Real.cos x ↔ x ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4)) := by
  sorry

end NUMINAMATH_CALUDE_sin_greater_cos_range_l4024_402420


namespace NUMINAMATH_CALUDE_min_a_for_f_nonpositive_l4024_402490

noncomputable def f (a x : ℝ) : ℝ := Real.exp x * (x^3 - 3*x + 3) - a * Real.exp x - x

theorem min_a_for_f_nonpositive :
  (∃ (a : ℝ), ∀ (x : ℝ), x ≥ -2 → f a x ≤ 0) ∧
  (∀ (b : ℝ), b < 1 - 1/Real.exp 1 → ∃ (x : ℝ), x ≥ -2 ∧ f b x > 0) :=
sorry

end NUMINAMATH_CALUDE_min_a_for_f_nonpositive_l4024_402490


namespace NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l4024_402428

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction_division :
  (3 / 4) / (5 / 8) = 6 / 5 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l4024_402428


namespace NUMINAMATH_CALUDE_constant_sum_product_l4024_402400

theorem constant_sum_product (n : Nat) (h : n = 15) : 
  ∃ k : Nat, ∀ (operations : List (Nat × Nat)), 
    operations.length = n - 1 → 
    (∀ (x y : Nat), (x, y) ∈ operations → x ≤ n ∧ y ≤ n) →
    (List.foldl (λ acc (x, y) => acc + x * y * (x + y)) 0 operations) = k ∧ k = 49140 := by
  sorry

end NUMINAMATH_CALUDE_constant_sum_product_l4024_402400


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l4024_402423

/-- Proves that for a rectangular plot with given dimensions and total fencing cost,
    the cost per meter of fencing is as calculated. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 55)
  (h2 : breadth = 45)
  (h3 : length = breadth + 10)
  (h4 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l4024_402423


namespace NUMINAMATH_CALUDE_ice_cream_stacking_l4024_402417

theorem ice_cream_stacking (n : ℕ) (k : ℕ) (h1 : n = 5) (h2 : k = 2) :
  (n! / k!) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_stacking_l4024_402417


namespace NUMINAMATH_CALUDE_max_value_vx_minus_yz_l4024_402415

def A : Set Int := {-3, -2, -1, 0, 1, 2, 3}

theorem max_value_vx_minus_yz :
  ∃ (v x y z : Int), v ∈ A ∧ x ∈ A ∧ y ∈ A ∧ z ∈ A ∧
    v * x - y * z = 6 ∧
    ∀ (v' x' y' z' : Int), v' ∈ A → x' ∈ A → y' ∈ A → z' ∈ A →
      v' * x' - y' * z' ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_vx_minus_yz_l4024_402415


namespace NUMINAMATH_CALUDE_solve_candy_problem_l4024_402479

def candy_problem (debby_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) : Prop :=
  let total_candy := debby_candy + sister_candy
  let eaten_candy := total_candy - remaining_candy
  eaten_candy = 35

theorem solve_candy_problem :
  candy_problem 32 42 39 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l4024_402479


namespace NUMINAMATH_CALUDE_range_of_a_l4024_402402

-- Define the custom operation
def circleMultiply (x y : ℝ) : ℝ := x * (1 - y)

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, circleMultiply (x - a) (x + a) < 2) → 
  -1 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4024_402402


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l4024_402450

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, 5; -2, 9]

theorem matrix_inverse_proof :
  A⁻¹ = !![9/46, -5/46; 2/46, 4/46] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l4024_402450


namespace NUMINAMATH_CALUDE_flag_designs_count_l4024_402413

/-- The number of colors available for the flag. -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag. -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs. -/
def total_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating that the total number of possible flag designs is 27. -/
theorem flag_designs_count : total_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_designs_count_l4024_402413


namespace NUMINAMATH_CALUDE_expansion_sum_zero_l4024_402482

theorem expansion_sum_zero (n k : ℕ) (a b : ℝ) (h1 : n ≥ 2) (h2 : a * b ≠ 0) (h3 : a = k^2 * b) (h4 : k > 0) :
  (n * (a - b)^(n-1) * (-b) + n * (n-1) / 2 * (a - b)^(n-2) * (-b)^2 = 0) →
  n = 2 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_expansion_sum_zero_l4024_402482


namespace NUMINAMATH_CALUDE_triangle_perimeter_l4024_402446

theorem triangle_perimeter (a b : ℝ) (perimeters : List ℝ) : 
  a = 25 → b = 20 → perimeters = [58, 64, 70, 76, 82] →
  ∃ (p : ℝ), p ∈ perimeters ∧ 
  (∀ (x : ℝ), x > 0 ∧ a + b > x ∧ a + x > b ∧ b + x > a → 
    p ≠ a + b + x) ∧
  (∀ (q : ℝ), q ∈ perimeters ∧ q ≠ p → 
    ∃ (y : ℝ), y > 0 ∧ a + b > y ∧ a + y > b ∧ b + y > a ∧ 
    q = a + b + y) := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l4024_402446


namespace NUMINAMATH_CALUDE_probability_seven_tails_l4024_402496

/-- The probability of flipping exactly k tails in n flips of an unfair coin -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of flipping exactly 7 tails in 10 flips of an unfair coin with 2/3 probability of tails -/
theorem probability_seven_tails : 
  binomial_probability 10 7 (2/3) = 5120/19683 := by
  sorry

end NUMINAMATH_CALUDE_probability_seven_tails_l4024_402496


namespace NUMINAMATH_CALUDE_altitude_segment_length_l4024_402463

/-- Represents an acute triangle with two altitudes dividing the sides. -/
structure AcuteTriangleWithAltitudes where
  -- The lengths of the segments created by the altitudes
  a : ℝ
  b : ℝ
  c : ℝ
  y : ℝ
  -- Conditions
  acute : a > 0 ∧ b > 0 ∧ c > 0 ∧ y > 0
  a_val : a = 7
  b_val : b = 4
  c_val : c = 3

/-- The theorem stating that y = 12/7 in the given triangle configuration. -/
theorem altitude_segment_length (t : AcuteTriangleWithAltitudes) : t.y = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_altitude_segment_length_l4024_402463
