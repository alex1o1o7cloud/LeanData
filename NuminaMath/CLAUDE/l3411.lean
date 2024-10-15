import Mathlib

namespace NUMINAMATH_CALUDE_acid_dilution_l3411_341187

/-- Given m ounces of m% acid solution, adding x ounces of water yields (m-10)% solution -/
theorem acid_dilution (m : ℝ) (x : ℝ) (h : m > 25) :
  (m^2 / 100 = (m - 10) / 100 * (m + x)) → x = 10 * m / (m - 10) := by sorry

end NUMINAMATH_CALUDE_acid_dilution_l3411_341187


namespace NUMINAMATH_CALUDE_intersection_line_circle_l3411_341112

/-- Given a line y = 2x + 1 intersecting a circle x^2 + y^2 + ax + 2y + 1 = 0 at points A and B,
    and a line mx + y + 2 = 0 that bisects chord AB perpendicularly, prove that a = 4 -/
theorem intersection_line_circle (a m : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = 2 * A.1 + 1 ∧ B.2 = 2 * B.1 + 1) ∧ 
    (A.1^2 + A.2^2 + a * A.1 + 2 * A.2 + 1 = 0 ∧ 
     B.1^2 + B.2^2 + a * B.1 + 2 * B.2 + 1 = 0) ∧
    (∃ C : ℝ × ℝ, C ∈ Set.Icc A B ∧ 
      m * C.1 + C.2 + 2 = 0 ∧
      (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0)) →
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_circle_l3411_341112


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l3411_341122

theorem units_digit_of_expression : 
  (2 * 21 * 2019 + 2^5 - 4^3) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l3411_341122


namespace NUMINAMATH_CALUDE_peggy_record_count_l3411_341166

/-- The number of records Peggy has -/
def num_records : ℕ := 200

/-- The price Sammy offers for each record -/
def sammy_price : ℚ := 4

/-- The price Bryan offers for each record he's interested in -/
def bryan_interested_price : ℚ := 6

/-- The price Bryan offers for each record he's not interested in -/
def bryan_not_interested_price : ℚ := 1

/-- The difference in profit between Sammy's and Bryan's deals -/
def profit_difference : ℚ := 100

theorem peggy_record_count :
  (sammy_price * num_records) - 
  ((bryan_interested_price * (num_records / 2)) + (bryan_not_interested_price * (num_records / 2))) = 
  profit_difference :=
sorry

end NUMINAMATH_CALUDE_peggy_record_count_l3411_341166


namespace NUMINAMATH_CALUDE_jason_born_1981_l3411_341155

/-- The year of the first AMC 8 competition -/
def first_amc8_year : ℕ := 1985

/-- The number of the AMC 8 competition Jason participated in -/
def jason_amc8_number : ℕ := 10

/-- Jason's age when he participated in the AMC 8 -/
def jason_age : ℕ := 13

/-- Calculates the year of a given AMC 8 competition -/
def amc8_year (n : ℕ) : ℕ := first_amc8_year + n - 1

/-- Jason's birth year -/
def jason_birth_year : ℕ := amc8_year jason_amc8_number - jason_age

theorem jason_born_1981 : jason_birth_year = 1981 := by
  sorry

end NUMINAMATH_CALUDE_jason_born_1981_l3411_341155


namespace NUMINAMATH_CALUDE_distance_traveled_l3411_341114

theorem distance_traveled (initial_speed : ℝ) (increased_speed : ℝ) (additional_distance : ℝ) :
  initial_speed = 10 →
  increased_speed = 15 →
  additional_distance = 15 →
  ∃ (actual_distance : ℝ) (time : ℝ),
    actual_distance = initial_speed * time ∧
    actual_distance + additional_distance = increased_speed * time ∧
    actual_distance = 30 :=
by sorry

end NUMINAMATH_CALUDE_distance_traveled_l3411_341114


namespace NUMINAMATH_CALUDE_gnuff_tutoring_time_l3411_341127

/-- Calculates the number of minutes tutored given the total amount paid, flat rate, and per-minute rate. -/
def minutes_tutored (total_amount : ℕ) (flat_rate : ℕ) (per_minute_rate : ℕ) : ℕ :=
  (total_amount - flat_rate) / per_minute_rate

/-- Theorem stating that given the specific rates and total amount, the number of minutes tutored is 18. -/
theorem gnuff_tutoring_time :
  minutes_tutored 146 20 7 = 18 := by
  sorry

#eval minutes_tutored 146 20 7

end NUMINAMATH_CALUDE_gnuff_tutoring_time_l3411_341127


namespace NUMINAMATH_CALUDE_pattern_c_cannot_fold_l3411_341110

/-- Represents a pattern of squares with folding lines -/
structure SquarePattern where
  squares : Finset (ℝ × ℝ)  -- Set of coordinates for squares
  foldLines : Finset ((ℝ × ℝ) × (ℝ × ℝ))  -- Set of folding lines

/-- Represents the set of all possible patterns -/
def AllPatterns : Finset SquarePattern := sorry

/-- Predicate to check if a pattern can be folded into a cube without overlap -/
def canFoldIntoCube (p : SquarePattern) : Prop := sorry

/-- The specific Pattern C -/
def PatternC : SquarePattern := sorry

/-- Theorem stating that Pattern C is the only pattern that cannot be folded into a cube -/
theorem pattern_c_cannot_fold :
  PatternC ∈ AllPatterns ∧
  ¬(canFoldIntoCube PatternC) ∧
  ∀ p ∈ AllPatterns, p ≠ PatternC → canFoldIntoCube p :=
sorry

end NUMINAMATH_CALUDE_pattern_c_cannot_fold_l3411_341110


namespace NUMINAMATH_CALUDE_lcm_minus_gcd_equals_34_l3411_341100

theorem lcm_minus_gcd_equals_34 : Nat.lcm 40 8 - Nat.gcd 24 54 = 34 := by
  sorry

end NUMINAMATH_CALUDE_lcm_minus_gcd_equals_34_l3411_341100


namespace NUMINAMATH_CALUDE_pants_price_decrease_percentage_l3411_341138

/-- Proves that the percentage decrease in selling price is 20% given the conditions --/
theorem pants_price_decrease_percentage (purchase_price : ℝ) (markup_percentage : ℝ) (gross_profit : ℝ) :
  purchase_price = 210 →
  markup_percentage = 0.25 →
  gross_profit = 14 →
  let original_price := purchase_price / (1 - markup_percentage)
  let final_price := purchase_price + gross_profit
  let price_decrease := original_price - final_price
  let percentage_decrease := (price_decrease / original_price) * 100
  percentage_decrease = 20 := by
  sorry

end NUMINAMATH_CALUDE_pants_price_decrease_percentage_l3411_341138


namespace NUMINAMATH_CALUDE_johns_age_to_tonyas_age_ratio_l3411_341144

/-- Proves that the ratio of John's age to Tonya's age is 1:2 given the specified conditions --/
theorem johns_age_to_tonyas_age_ratio :
  ∀ (john mary tonya : ℕ),
    john = 2 * mary →
    tonya = 60 →
    (john + mary + tonya) / 3 = 35 →
    john / tonya = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_to_tonyas_age_ratio_l3411_341144


namespace NUMINAMATH_CALUDE_negation_of_odd_function_implication_l3411_341140

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem negation_of_odd_function_implication :
  (¬ (is_odd f → is_odd (λ x => f (-x)))) ↔ (is_odd f → ¬ is_odd (λ x => f (-x))) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_odd_function_implication_l3411_341140


namespace NUMINAMATH_CALUDE_train_journey_distance_l3411_341113

/-- Calculates the total distance traveled by a train with increasing speed -/
def train_distance (initial_speed : ℕ) (speed_increase : ℕ) (hours : ℕ) : ℕ :=
  hours * (2 * initial_speed + (hours - 1) * speed_increase) / 2

/-- Theorem: A train traveling for 11 hours, with an initial speed of 10 miles/hr
    and increasing its speed by 10 miles/hr each hour, travels a total of 660 miles -/
theorem train_journey_distance :
  train_distance 10 10 11 = 660 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_distance_l3411_341113


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3411_341134

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₂ + a₁₂ = 32, prove that a₃ + a₁₁ = 32 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : ArithmeticSequence a) (h2 : a 2 + a 12 = 32) :
  a 3 + a 11 = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3411_341134


namespace NUMINAMATH_CALUDE_symmetry_implies_axis_l3411_341149

/-- A function g is symmetric about x = 1.5 if g(x) = g(3-x) for all x -/
def IsSymmetricAbout1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for a function g if 
    for every point (x, g(x)) on the graph, (3-x, g(x)) is also on the graph -/
def IsAxisOfSymmetry1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

theorem symmetry_implies_axis (g : ℝ → ℝ) :
  IsSymmetricAbout1_5 g → IsAxisOfSymmetry1_5 g := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_axis_l3411_341149


namespace NUMINAMATH_CALUDE_unique_divisor_remainder_l3411_341147

theorem unique_divisor_remainder : ∃! (d r : ℤ),
  (1210 % d = r) ∧
  (1690 % d = r) ∧
  (2670 % d = r) ∧
  (d > 0) ∧
  (0 ≤ r) ∧
  (r < d) ∧
  (d - 4*r = -20) := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_remainder_l3411_341147


namespace NUMINAMATH_CALUDE_blocks_added_l3411_341109

theorem blocks_added (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 86) 
  (h2 : final_blocks = 95) : 
  final_blocks - initial_blocks = 9 := by
  sorry

end NUMINAMATH_CALUDE_blocks_added_l3411_341109


namespace NUMINAMATH_CALUDE_apple_orange_cost_l3411_341162

theorem apple_orange_cost (apple_cost orange_cost : ℚ) 
  (eq1 : 2 * apple_cost + 3 * orange_cost = 6)
  (eq2 : 4 * apple_cost + 7 * orange_cost = 13) :
  16 * apple_cost + 23 * orange_cost = 47 := by
  sorry

end NUMINAMATH_CALUDE_apple_orange_cost_l3411_341162


namespace NUMINAMATH_CALUDE_tangent_line_quadratic_l3411_341178

/-- Given a quadratic function f(x) = x² + ax + b, if the tangent line
    to f at x = 0 is x - y + 1 = 0, then a = 1 and b = 1 -/
theorem tangent_line_quadratic (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  let f' : ℝ → ℝ := λ x ↦ 2*x + a
  (∀ x, f' x = (deriv f) x) →
  (f' 0 = 1) →
  (f 0 = 1) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_quadratic_l3411_341178


namespace NUMINAMATH_CALUDE_calculation_proof_l3411_341168

theorem calculation_proof : 
  Real.sqrt 3 * (Real.sqrt 3 + 2) - 2 * Real.tan (60 * π / 180) + (-1) ^ 2023 = 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l3411_341168


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3411_341161

theorem polynomial_expansion (t : ℝ) : 
  (3 * t^3 - 4 * t + 1) * (4 * t^2 - 5 * t + 3) = 
  12 * t^5 - 15 * t^4 - 7 * t^3 + 24 * t^2 - 17 * t + 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3411_341161


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l3411_341153

-- Define the conditions
theorem log_equality_implies_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 8 = Real.log q / Real.log 18) ∧
  (Real.log p / Real.log 8 = Real.log (p + q) / Real.log 32) →
  q / p = (4 + Real.sqrt 41) / 5 := by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l3411_341153


namespace NUMINAMATH_CALUDE_smallest_special_number_l3411_341172

theorem smallest_special_number : ∃ (n : ℕ), 
  (100 ≤ n ∧ n < 1000) ∧ 
  (∃ (k : ℕ), n = 2 * k) ∧
  (∃ (k : ℕ), n + 1 = 3 * k) ∧
  (∃ (k : ℕ), n + 2 = 4 * k) ∧
  (∃ (k : ℕ), n + 3 = 5 * k) ∧
  (∃ (k : ℕ), n + 4 = 6 * k) ∧
  (∀ m : ℕ, m < n → 
    ¬((100 ≤ m ∧ m < 1000) ∧ 
      (∃ (k : ℕ), m = 2 * k) ∧
      (∃ (k : ℕ), m + 1 = 3 * k) ∧
      (∃ (k : ℕ), m + 2 = 4 * k) ∧
      (∃ (k : ℕ), m + 3 = 5 * k) ∧
      (∃ (k : ℕ), m + 4 = 6 * k))) ∧
  n = 122 :=
by sorry

end NUMINAMATH_CALUDE_smallest_special_number_l3411_341172


namespace NUMINAMATH_CALUDE_base_conversion_difference_l3411_341111

/-- Convert a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_conversion_difference : 
  let base_6_num := [0, 1, 2, 3, 4]  -- 43210 in base 6 (least significant digit first)
  let base_7_num := [0, 1, 2, 3]     -- 3210 in base 7 (least significant digit first)
  to_base_10 base_6_num 6 - to_base_10 base_7_num 7 = 4776 := by
  sorry


end NUMINAMATH_CALUDE_base_conversion_difference_l3411_341111


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3411_341192

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 2*a-1, a^2+1}
  A ∩ B = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3411_341192


namespace NUMINAMATH_CALUDE_walking_distance_l3411_341136

theorem walking_distance (original_speed : ℝ) (increased_speed : ℝ) (additional_distance : ℝ) (actual_distance : ℝ) : 
  original_speed = 12 →
  increased_speed = 20 →
  increased_speed * (actual_distance / original_speed) = original_speed * (actual_distance / original_speed) + additional_distance →
  additional_distance = 24 →
  actual_distance = 36 := by
sorry

end NUMINAMATH_CALUDE_walking_distance_l3411_341136


namespace NUMINAMATH_CALUDE_invisible_dots_count_l3411_341197

/-- The sum of numbers on a single die -/
def dieFaceSum : ℕ := 21

/-- The number of dice -/
def numDice : ℕ := 5

/-- The visible numbers on the dice -/
def visibleNumbers : List ℕ := [1, 1, 2, 2, 3, 3, 4, 5, 6, 6]

/-- The theorem stating that the total number of dots not visible is 72 -/
theorem invisible_dots_count : 
  numDice * dieFaceSum - visibleNumbers.sum = 72 := by sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l3411_341197


namespace NUMINAMATH_CALUDE_vegetarian_eaters_count_family_total_check_l3411_341160

/-- Represents the eating habits distribution in a family -/
structure FamilyEatingHabits where
  total : Nat
  onlyVegetarian : Nat
  onlyNonVegetarian : Nat
  both : Nat
  pescatarian : Nat
  vegan : Nat

/-- Calculates the number of people eating vegetarian food -/
def vegetarianEaters (habits : FamilyEatingHabits) : Nat :=
  habits.onlyVegetarian + habits.both + habits.vegan

/-- The given family's eating habits -/
def familyHabits : FamilyEatingHabits := {
  total := 40
  onlyVegetarian := 16
  onlyNonVegetarian := 12
  both := 8
  pescatarian := 3
  vegan := 1
}

/-- Theorem: The number of vegetarian eaters in the family is 25 -/
theorem vegetarian_eaters_count :
  vegetarianEaters familyHabits = 25 := by
  sorry

/-- Theorem: The sum of all eating habit categories equals the total family members -/
theorem family_total_check :
  familyHabits.onlyVegetarian + familyHabits.onlyNonVegetarian + familyHabits.both +
  familyHabits.pescatarian + familyHabits.vegan = familyHabits.total := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_count_family_total_check_l3411_341160


namespace NUMINAMATH_CALUDE_a_to_b_value_l3411_341101

theorem a_to_b_value (a b : ℝ) (h : Real.sqrt (a + 2) + (b - 3)^2 = 0) : a^b = -8 := by
  sorry

end NUMINAMATH_CALUDE_a_to_b_value_l3411_341101


namespace NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l3411_341150

/-- Given a geometric sequence with positive terms and common ratio q,
    if 3a₁, (1/2)a₃, 2a₂ form an arithmetic sequence, then q = 3 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence with ratio q
  q > 0 →  -- q is positive
  2 * ((1/2) * a 3) = 3 * a 1 + 2 * a 2 →  -- arithmetic sequence condition
  q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_ratio_l3411_341150


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l3411_341103

-- Part 1
def positive_integer_solutions (x : ℕ) : Prop :=
  4 * (x + 2) < 18 + 2 * x

theorem solution_set_part1 :
  {x : ℕ | positive_integer_solutions x} = {1, 2, 3, 4} :=
sorry

-- Part 2
def inequality_system (x : ℝ) : Prop :=
  5 * x + 2 ≥ 4 * x + 1 ∧ (x + 1) / 4 > (x - 3) / 2 + 1

theorem solution_set_part2 :
  {x : ℝ | inequality_system x} = {x : ℝ | -1 ≤ x ∧ x < 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l3411_341103


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_869_l3411_341183

theorem sqrt_product_plus_one_equals_869 : 
  Real.sqrt ((31 * 30 * 29 * 28) + 1) = 869 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_869_l3411_341183


namespace NUMINAMATH_CALUDE_perfect_square_property_l3411_341175

theorem perfect_square_property (x y p : ℕ+) (hp : Nat.Prime p.val) 
  (h : 4 * x.val^2 + 8 * y.val^2 + (2 * x.val - 3 * y.val) * p.val - 12 * x.val * y.val = 0) :
  ∃ (n : ℕ), 4 * y.val + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_property_l3411_341175


namespace NUMINAMATH_CALUDE_gcd_of_1975_and_2625_l3411_341156

theorem gcd_of_1975_and_2625 : Nat.gcd 1975 2625 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_1975_and_2625_l3411_341156


namespace NUMINAMATH_CALUDE_line_equation_proof_l3411_341165

/-- A line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (l : Line2D) (p : Point2D) : Prop :=
  l.a * p.x + l.b * p.y = l.c

/-- Check if a vector is normal to a line -/
def isNormalVector (l : Line2D) (v : Vector2D) : Prop :=
  l.a * v.x + l.b * v.y = 0

/-- The main theorem -/
theorem line_equation_proof (l : Line2D) (A : Point2D) (n : Vector2D) :
  l.a = 2 ∧ l.b = 1 ∧ l.c = 2 →
  A.x = 1 ∧ A.y = 0 →
  n.x = 2 ∧ n.y = -1 →
  pointOnLine l A ∧ isNormalVector l n := by
  sorry

#check line_equation_proof

end NUMINAMATH_CALUDE_line_equation_proof_l3411_341165


namespace NUMINAMATH_CALUDE_fraction_equality_l3411_341179

theorem fraction_equality (a b : ℚ) (h : a / b = 3 / 2) : (a + b) / a = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3411_341179


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3411_341194

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) / (x + 2) ≤ 0 ↔ -2 < x ∧ x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3411_341194


namespace NUMINAMATH_CALUDE_east_north_not_opposite_forward_backward_opposite_main_theorem_l3411_341176

/-- Represents a direction of movement --/
inductive Direction
  | Forward
  | Backward
  | East
  | North

/-- Represents a quantity with a value and a direction --/
structure Quantity where
  value : ℝ
  direction : Direction

/-- Defines when two quantities are opposite --/
def are_opposite (q1 q2 : Quantity) : Prop :=
  (q1.value = q2.value) ∧
  ((q1.direction = Direction.Forward ∧ q2.direction = Direction.Backward) ∨
   (q1.direction = Direction.Backward ∧ q2.direction = Direction.Forward))

/-- Theorem stating that east and north movements are not opposite --/
theorem east_north_not_opposite :
  ¬(are_opposite
      { value := 10, direction := Direction.East }
      { value := 10, direction := Direction.North }) :=
by
  sorry

/-- Theorem stating that forward and backward movements are opposite --/
theorem forward_backward_opposite :
  are_opposite
    { value := 5, direction := Direction.Forward }
    { value := 5, direction := Direction.Backward } :=
by
  sorry

/-- Main theorem proving that east and north movements are not opposite,
    while forward and backward movements are opposite --/
theorem main_theorem :
  (¬(are_opposite
      { value := 10, direction := Direction.East }
      { value := 10, direction := Direction.North })) ∧
  (are_opposite
    { value := 5, direction := Direction.Forward }
    { value := 5, direction := Direction.Backward }) :=
by
  sorry

end NUMINAMATH_CALUDE_east_north_not_opposite_forward_backward_opposite_main_theorem_l3411_341176


namespace NUMINAMATH_CALUDE_number_of_cartons_l3411_341182

/-- Represents the number of boxes in a carton -/
def boxes_per_carton : ℕ := 12

/-- Represents the number of packs in a box -/
def packs_per_box : ℕ := 10

/-- Represents the price of a pack in dollars -/
def price_per_pack : ℕ := 1

/-- Represents the total cost for all cartons in dollars -/
def total_cost : ℕ := 1440

/-- Theorem stating that the number of cartons is 12 -/
theorem number_of_cartons : 
  (total_cost : ℚ) / (boxes_per_carton * packs_per_box * price_per_pack) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cartons_l3411_341182


namespace NUMINAMATH_CALUDE_range_start_number_l3411_341137

theorem range_start_number (n : ℕ) (h1 : n ≤ 79) (h2 : n % 11 = 0) 
  (h3 : ∀ k, k ∈ Finset.range 5 → (n - k * 11) % 11 = 0) : n - 4 * 11 = 33 :=
sorry

end NUMINAMATH_CALUDE_range_start_number_l3411_341137


namespace NUMINAMATH_CALUDE_fixed_point_range_l3411_341198

/-- The problem statement translated to Lean 4 --/
theorem fixed_point_range (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) (hm1 : m ≠ 1) :
  (∃ (x y : ℝ), (2 * a * x - b * y + 14 = 0) ∧ 
                (y = m^(x + 1) + 1) ∧ 
                ((x - a + 1)^2 + (y + b - 2)^2 ≤ 25)) →
  (3 / 4 : ℝ) ≤ b / a ∧ b / a ≤ (4 / 3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_range_l3411_341198


namespace NUMINAMATH_CALUDE_largest_perimeter_l3411_341164

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℤ

/-- Checks if the given lengths can form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℝ :=
  t.side1 + t.side2 + t.side3

/-- Theorem stating the largest possible perimeter for the given triangle --/
theorem largest_perimeter :
  ∃ (t : Triangle), t.side1 = 8 ∧ t.side2 = 12 ∧ is_valid_triangle t ∧
  ∀ (t' : Triangle), t'.side1 = 8 ∧ t'.side2 = 12 ∧ is_valid_triangle t' →
  perimeter t ≥ perimeter t' ∧ perimeter t = 39 :=
sorry

end NUMINAMATH_CALUDE_largest_perimeter_l3411_341164


namespace NUMINAMATH_CALUDE_cone_surface_area_l3411_341135

theorem cone_surface_area (slant_height : ℝ) (base_circumference : ℝ) :
  slant_height = 2 →
  base_circumference = 2 * Real.pi →
  π * (base_circumference / (2 * π)) * (base_circumference / (2 * π) + slant_height) = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3411_341135


namespace NUMINAMATH_CALUDE_speeding_ticket_theorem_l3411_341104

/-- Represents the percentage of motorists who receive speeding tickets -/
def ticket_percentage : ℝ := 10

/-- Represents the percentage of motorists who exceed the speed limit -/
def exceed_limit_percentage : ℝ := 16.666666666666664

/-- Theorem stating that 40% of motorists who exceed the speed limit do not receive speeding tickets -/
theorem speeding_ticket_theorem :
  (exceed_limit_percentage - ticket_percentage) / exceed_limit_percentage * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_speeding_ticket_theorem_l3411_341104


namespace NUMINAMATH_CALUDE_rectangular_plot_shorter_side_l3411_341123

theorem rectangular_plot_shorter_side
  (width : ℝ)
  (num_poles : ℕ)
  (pole_distance : ℝ)
  (h1 : width = 50)
  (h2 : num_poles = 32)
  (h3 : pole_distance = 5)
  : ∃ (length : ℝ), length = 27.5 ∧ 2 * (length + width) = (num_poles - 1 : ℝ) * pole_distance :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_shorter_side_l3411_341123


namespace NUMINAMATH_CALUDE_factors_of_2310_l3411_341139

theorem factors_of_2310 : Finset.card (Nat.divisors 2310) = 32 := by sorry

end NUMINAMATH_CALUDE_factors_of_2310_l3411_341139


namespace NUMINAMATH_CALUDE_scissors_freedom_theorem_l3411_341118

/-- Represents the state of the rope and scissors system -/
structure RopeScissorsState where
  loopThroughScissors : Bool
  ropeEndsFixed : Bool
  noKnotsUntied : Bool

/-- Represents a single manipulation of the rope -/
inductive RopeManipulation
  | PullLoop
  | PassLoopAroundEnds
  | ReverseDirection

/-- Defines a sequence of rope manipulations -/
def ManipulationSequence := List RopeManipulation

/-- Predicate to check if a manipulation sequence frees the scissors -/
def freesScissors (seq : ManipulationSequence) : Prop := sorry

/-- The main theorem stating that there exists a sequence of manipulations that frees the scissors -/
theorem scissors_freedom_theorem (initialState : RopeScissorsState) 
  (h1 : initialState.loopThroughScissors = true)
  (h2 : initialState.ropeEndsFixed = true)
  (h3 : initialState.noKnotsUntied = true) :
  ∃ (seq : ManipulationSequence), freesScissors seq := by
  sorry


end NUMINAMATH_CALUDE_scissors_freedom_theorem_l3411_341118


namespace NUMINAMATH_CALUDE_geometric_mean_of_two_and_six_l3411_341185

theorem geometric_mean_of_two_and_six :
  ∃ (x : ℝ), x^2 = 2 * 6 ∧ (x = 2 * Real.sqrt 3 ∨ x = -2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_of_two_and_six_l3411_341185


namespace NUMINAMATH_CALUDE_unique_prime_generator_l3411_341184

theorem unique_prime_generator : ∃! p : ℕ, Prime (p + 10) ∧ Prime (p + 14) :=
  ⟨3, 
    by {
      sorry -- Proof that 3 satisfies the conditions
    },
    by {
      sorry -- Proof that 3 is the only natural number satisfying the conditions
    }
  ⟩

end NUMINAMATH_CALUDE_unique_prime_generator_l3411_341184


namespace NUMINAMATH_CALUDE_vertex_not_at_minus_two_one_l3411_341141

/-- Represents a parabola in the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The given parabola y = -2(x-2)^2 + 1 --/
def givenParabola : Parabola := { a := -2, h := 2, k := 1 }

/-- The vertex of a parabola --/
def vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

/-- Theorem stating that the vertex of the given parabola is not at (-2,1) --/
theorem vertex_not_at_minus_two_one :
  vertex givenParabola ≠ (-2, 1) := by
  sorry

end NUMINAMATH_CALUDE_vertex_not_at_minus_two_one_l3411_341141


namespace NUMINAMATH_CALUDE_halloween_cleanup_time_l3411_341189

theorem halloween_cleanup_time
  (egg_cleanup_time : ℕ)
  (tp_cleanup_time : ℕ)
  (egg_count : ℕ)
  (tp_count : ℕ)
  (h1 : egg_cleanup_time = 15)  -- 15 seconds per egg
  (h2 : tp_cleanup_time = 30)   -- 30 minutes per roll of toilet paper
  (h3 : egg_count = 60)         -- 60 eggs
  (h4 : tp_count = 7)           -- 7 rolls of toilet paper
  : (egg_count * egg_cleanup_time) / 60 + tp_count * tp_cleanup_time = 225 := by
  sorry

#check halloween_cleanup_time

end NUMINAMATH_CALUDE_halloween_cleanup_time_l3411_341189


namespace NUMINAMATH_CALUDE_circumcenter_equidistant_l3411_341128

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem: The circumcenter is equidistant from all vertices
theorem circumcenter_equidistant (t : Triangle) :
  distance (circumcenter t) t.A = distance (circumcenter t) t.B ∧
  distance (circumcenter t) t.B = distance (circumcenter t) t.C :=
sorry

end NUMINAMATH_CALUDE_circumcenter_equidistant_l3411_341128


namespace NUMINAMATH_CALUDE_int_roots_count_l3411_341196

/-- A polynomial of degree 4 with integer coefficients -/
structure IntPoly4 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The number of integer roots of a polynomial, counting multiplicity -/
def num_int_roots (p : IntPoly4) : ℕ := sorry

/-- Theorem stating the possible values for the number of integer roots -/
theorem int_roots_count (p : IntPoly4) : 
  num_int_roots p = 0 ∨ num_int_roots p = 1 ∨ num_int_roots p = 2 ∨ num_int_roots p = 4 :=
sorry

end NUMINAMATH_CALUDE_int_roots_count_l3411_341196


namespace NUMINAMATH_CALUDE_joes_haircuts_l3411_341181

/-- The number of women's haircuts Joe did -/
def womens_haircuts : ℕ := sorry

/-- The time it takes to cut a woman's hair in minutes -/
def womens_haircut_time : ℕ := 50

/-- The time it takes to cut a man's hair in minutes -/
def mens_haircut_time : ℕ := 15

/-- The time it takes to cut a kid's hair in minutes -/
def kids_haircut_time : ℕ := 25

/-- The number of men's haircuts Joe did -/
def mens_haircuts : ℕ := 2

/-- The number of kids' haircuts Joe did -/
def kids_haircuts : ℕ := 3

/-- The total time Joe spent cutting hair in minutes -/
def total_time : ℕ := 255

theorem joes_haircuts : womens_haircuts = 3 := by sorry

end NUMINAMATH_CALUDE_joes_haircuts_l3411_341181


namespace NUMINAMATH_CALUDE_tetrahedron_in_spheres_l3411_341115

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron defined by four points -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a sphere defined by its center and radius -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Checks if a point is inside a sphere -/
def isInSphere (p : Point3D) (s : Sphere) : Prop := sorry

/-- Checks if a point is inside a tetrahedron -/
def isInTetrahedron (p : Point3D) (t : Tetrahedron) : Prop := sorry

/-- Creates a sphere with diameter AB -/
def sphereAB (t : Tetrahedron) : Sphere := sorry

/-- Creates a sphere with diameter AC -/
def sphereAC (t : Tetrahedron) : Sphere := sorry

/-- Creates a sphere with diameter AD -/
def sphereAD (t : Tetrahedron) : Sphere := sorry

/-- The main theorem: every point in the tetrahedron is in at least one of the three spheres -/
theorem tetrahedron_in_spheres (t : Tetrahedron) (p : Point3D) :
  isInTetrahedron p t →
  (isInSphere p (sphereAB t) ∨ isInSphere p (sphereAC t) ∨ isInSphere p (sphereAD t)) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_in_spheres_l3411_341115


namespace NUMINAMATH_CALUDE_exists_real_a_sqrt3_minus_a_real_l3411_341186

theorem exists_real_a_sqrt3_minus_a_real : ∃ a : ℝ, ∃ b : ℝ, b = Real.sqrt 3 - a := by
  sorry

end NUMINAMATH_CALUDE_exists_real_a_sqrt3_minus_a_real_l3411_341186


namespace NUMINAMATH_CALUDE_twelve_switches_four_connections_l3411_341151

/-- The number of connections in a network of switches where each switch connects to a fixed number of others. -/
def connections (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a network of 12 switches, where each switch is directly connected to exactly 4 other switches, the total number of connections is 24. -/
theorem twelve_switches_four_connections :
  connections 12 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_twelve_switches_four_connections_l3411_341151


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l3411_341171

theorem max_value_of_trigonometric_function :
  let f : ℝ → ℝ := λ x => Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)
  let S : Set ℝ := {x | -5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 3}
  ∃ x₀ ∈ S, ∀ x ∈ S, f x ≤ f x₀ ∧ f x₀ = 11 * Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l3411_341171


namespace NUMINAMATH_CALUDE_square_sum_lower_bound_l3411_341107

theorem square_sum_lower_bound (a b : ℝ) 
  (h1 : a^3 - b^3 = 2) 
  (h2 : a^5 - b^5 ≥ 4) : 
  a^2 + b^2 ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_lower_bound_l3411_341107


namespace NUMINAMATH_CALUDE_article_selling_price_l3411_341131

theorem article_selling_price (cost_price : ℝ) (selling_price : ℝ) : 
  (selling_price - cost_price = cost_price - 448) → 
  (768 = 1.2 * cost_price) → 
  selling_price = 832 := by
sorry

end NUMINAMATH_CALUDE_article_selling_price_l3411_341131


namespace NUMINAMATH_CALUDE_sum_of_products_l3411_341121

theorem sum_of_products : 5 * 12 + 7 * 15 + 13 * 4 + 6 * 9 = 271 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_l3411_341121


namespace NUMINAMATH_CALUDE_donation_ratio_l3411_341169

theorem donation_ratio (shirts pants shorts : ℕ) : 
  shirts = 4 →
  pants = 2 * shirts →
  shirts + pants + shorts = 16 →
  shorts * 2 = pants :=
by
  sorry

end NUMINAMATH_CALUDE_donation_ratio_l3411_341169


namespace NUMINAMATH_CALUDE_gaeun_taller_than_nana_l3411_341116

/-- Nana's height in meters -/
def nana_height_m : ℝ := 1.618

/-- Gaeun's height in centimeters -/
def gaeun_height_cm : ℝ := 162.3

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem gaeun_taller_than_nana : 
  gaeun_height_cm > nana_height_m * m_to_cm := by sorry

end NUMINAMATH_CALUDE_gaeun_taller_than_nana_l3411_341116


namespace NUMINAMATH_CALUDE_base_4_divisible_by_19_l3411_341124

def base_4_to_decimal (a b c d : ℕ) : ℕ := a * 4^3 + b * 4^2 + c * 4 + d

theorem base_4_divisible_by_19 :
  ∃! x : ℕ, x < 4 ∧ 19 ∣ base_4_to_decimal 2 1 x 2 :=
by
  sorry

end NUMINAMATH_CALUDE_base_4_divisible_by_19_l3411_341124


namespace NUMINAMATH_CALUDE_first_die_sides_l3411_341148

theorem first_die_sides (p : ℝ) (n : ℕ) : 
  p = 0.023809523809523808 →  -- Given probability
  p = 1 / (n * 7) →           -- Probability formula
  n = 6                       -- Number of sides on first die
:= by sorry

end NUMINAMATH_CALUDE_first_die_sides_l3411_341148


namespace NUMINAMATH_CALUDE_decagon_circle_intersection_undecagon_no_circle_intersection_l3411_341108

/-- Represents a polygon with n sides -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Represents a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Constructs circles on each side of a polygon -/
def constructCircles {n : ℕ} (p : Polygon n) : Fin n → Circle :=
  sorry

/-- Checks if a point is a common intersection of all circles -/
def isCommonIntersection {n : ℕ} (p : Polygon n) (circles : Fin n → Circle) (point : ℝ × ℝ) : Prop :=
  sorry

/-- Checks if a point is a vertex of the polygon -/
def isVertex {n : ℕ} (p : Polygon n) (point : ℝ × ℝ) : Prop :=
  sorry

theorem decagon_circle_intersection :
  ∃ (p : Polygon 10) (point : ℝ × ℝ),
    isCommonIntersection p (constructCircles p) point ∧
    ¬isVertex p point :=
  sorry

theorem undecagon_no_circle_intersection :
  ∀ (p : Polygon 11) (point : ℝ × ℝ),
    isCommonIntersection p (constructCircles p) point →
    isVertex p point :=
  sorry

end NUMINAMATH_CALUDE_decagon_circle_intersection_undecagon_no_circle_intersection_l3411_341108


namespace NUMINAMATH_CALUDE_average_molar_mass_of_compound_l3411_341120

/-- Given a compound where 4 moles weigh 672 grams, prove that its average molar mass is 168 grams/mole -/
theorem average_molar_mass_of_compound (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 672)
  (h2 : num_moles = 4) :
  total_weight / num_moles = 168 := by
  sorry

end NUMINAMATH_CALUDE_average_molar_mass_of_compound_l3411_341120


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l3411_341106

/-- Given that selling 11 balls at Rs. 720 results in a loss equal to the cost price of 5 balls,
    prove that the cost price of one ball is Rs. 120. -/
theorem cost_price_of_ball (cost : ℕ) : 
  (11 * cost - 720 = 5 * cost) → cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l3411_341106


namespace NUMINAMATH_CALUDE_range_of_a_l3411_341167

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 2) ↔ ((x - a)^2 < 1)) ↔ 
  (1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3411_341167


namespace NUMINAMATH_CALUDE_a_2023_coordinates_l3411_341142

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Returns the conjugate point of a given point -/
def conjugate (p : Point) : Point :=
  { x := -p.y + 1, y := p.x + 1 }

/-- Returns the nth point in the sequence starting from A₁ -/
def nthPoint (n : ℕ) : Point :=
  match n % 4 with
  | 1 => { x := 3, y := 1 }
  | 2 => { x := 0, y := 4 }
  | 3 => { x := -3, y := 1 }
  | _ => { x := 0, y := -2 }

theorem a_2023_coordinates : nthPoint 2023 = { x := -3, y := 1 } := by
  sorry

end NUMINAMATH_CALUDE_a_2023_coordinates_l3411_341142


namespace NUMINAMATH_CALUDE_officer_selection_theorem_l3411_341130

def total_members : ℕ := 18
def officer_positions : ℕ := 6
def past_officers : ℕ := 8

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem officer_selection_theorem : 
  choose total_members officer_positions - 
  (choose (total_members - past_officers) officer_positions + 
   past_officers * choose (total_members - past_officers) (officer_positions - 1)) = 16338 :=
sorry

end NUMINAMATH_CALUDE_officer_selection_theorem_l3411_341130


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3411_341159

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + x < 0 ↔ -1/2 < x ∧ x < 1/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3411_341159


namespace NUMINAMATH_CALUDE_increasing_on_zero_one_iff_decreasing_on_three_four_l3411_341133

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem increasing_on_zero_one_iff_decreasing_on_three_four
  (f : ℝ → ℝ) (h1 : is_even_function f) (h2 : has_period f 2) :
  is_increasing_on f 0 1 ↔ is_decreasing_on f 3 4 :=
sorry

end NUMINAMATH_CALUDE_increasing_on_zero_one_iff_decreasing_on_three_four_l3411_341133


namespace NUMINAMATH_CALUDE_floor_sum_example_l3411_341190

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l3411_341190


namespace NUMINAMATH_CALUDE_student_selection_methods_l3411_341163

def total_students : ℕ := 8
def num_boys : ℕ := 6
def num_girls : ℕ := 2
def students_to_select : ℕ := 4
def boys_to_select : ℕ := 3
def girls_to_select : ℕ := 1

theorem student_selection_methods :
  (Nat.choose num_boys boys_to_select) * (Nat.choose num_girls girls_to_select) = 40 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_methods_l3411_341163


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3411_341143

/-- Given that y varies inversely as x², prove that x = 2 when y = 8, 
    given that y = 2 when x = 4. -/
theorem inverse_variation_problem (y x : ℝ) (h : x > 0) : 
  (∃ (k : ℝ), ∀ (x : ℝ), x > 0 → y * x^2 = k) → 
  (2 * 4^2 = 8 * x^2) →
  (y = 8) →
  (x = 2) := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3411_341143


namespace NUMINAMATH_CALUDE_vaishali_hats_l3411_341145

/-- The number of hats with 4 stripes each that Vaishali has -/
def hats_with_four_stripes : ℕ :=
  let three_stripe_hats := 4
  let three_stripe_count := 3
  let no_stripe_hats := 6
  let five_stripe_hats := 2
  let five_stripe_count := 5
  let total_stripes := 34
  let remaining_stripes := total_stripes - 
    (three_stripe_hats * three_stripe_count + 
     no_stripe_hats * 0 + 
     five_stripe_hats * five_stripe_count)
  remaining_stripes / 4

theorem vaishali_hats : hats_with_four_stripes = 3 := by
  sorry

end NUMINAMATH_CALUDE_vaishali_hats_l3411_341145


namespace NUMINAMATH_CALUDE_man_work_time_l3411_341119

/-- Represents the time taken to complete a piece of work -/
structure WorkTime where
  days : ℝ
  days_pos : days > 0

/-- Represents the rate at which work is completed -/
def WorkRate := ℝ

theorem man_work_time (total_work : ℝ) 
  (h_total_work_pos : total_work > 0)
  (combined_time : WorkTime) 
  (son_time : WorkTime) 
  (h_combined : combined_time.days = 3)
  (h_son : son_time.days = 7.5) :
  ∃ (man_time : WorkTime), man_time.days = 5 :=
sorry

end NUMINAMATH_CALUDE_man_work_time_l3411_341119


namespace NUMINAMATH_CALUDE_x_plus_y_value_l3411_341157

theorem x_plus_y_value (x y : Real) 
  (eq1 : x + Real.sin y = 2023)
  (eq2 : x + 2023 * Real.cos y = 2022)
  (y_range : π/2 ≤ y ∧ y ≤ π) :
  x + y = 2022 + π/2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l3411_341157


namespace NUMINAMATH_CALUDE_expression_value_at_two_l3411_341173

theorem expression_value_at_two :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 4)
  f 2 = 6 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_two_l3411_341173


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l3411_341132

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 7*a + 7 = 0) → (b^2 - 7*b + 7 = 0) → a^2 + b^2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l3411_341132


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l3411_341191

theorem repeating_decimal_sum_difference (x y z : ℚ) : 
  (x = 246 / 999) → 
  (y = 135 / 999) → 
  (z = 579 / 999) → 
  x - y + z = 230 / 333 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l3411_341191


namespace NUMINAMATH_CALUDE_diego_fruit_weight_l3411_341199

/-- The total weight of fruit Diego can carry in his bookbag -/
def total_weight (watermelon grapes oranges apples : ℕ) : ℕ :=
  watermelon + grapes + oranges + apples

/-- Theorem stating the total weight of fruit Diego can carry -/
theorem diego_fruit_weight :
  ∃ (watermelon grapes oranges apples : ℕ),
    watermelon = 1 ∧ grapes = 1 ∧ oranges = 1 ∧ apples = 17 ∧
    total_weight watermelon grapes oranges apples = 20 := by
  sorry

end NUMINAMATH_CALUDE_diego_fruit_weight_l3411_341199


namespace NUMINAMATH_CALUDE_triangle_max_area_l3411_341170

/-- Given a triangle ABC with side lengths a, b, c, where c = 1,
    and area S = (a^2 + b^2 - 1) / 4,
    prove that the maximum value of S is (√2 + 1) / 4 -/
theorem triangle_max_area (a b : ℝ) (h_c : c = 1) 
  (h_area : (a^2 + b^2 - 1) / 4 = (1/2) * a * b * Real.sin C) :
  (∃ (S : ℝ), S = (a^2 + b^2 - 1) / 4 ∧ 
    (∀ (S' : ℝ), S' = (a'^2 + b'^2 - 1) / 4 → S' ≤ S)) →
  (a^2 + b^2 - 1) / 4 ≤ (Real.sqrt 2 + 1) / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3411_341170


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l3411_341152

/-- Represents the structure of the cube after modifications -/
structure ModifiedCube where
  initialSize : Nat
  smallCubeSize : Nat
  removedCubes : Nat
  remainingCubes : Nat

/-- Calculates the surface area of the modified cube structure -/
def surfaceArea (cube : ModifiedCube) : Nat :=
  sorry

/-- Theorem stating that the surface area of the specific modified cube is 2820 -/
theorem modified_cube_surface_area :
  let cube : ModifiedCube := {
    initialSize := 12,
    smallCubeSize := 3,
    removedCubes := 14,
    remainingCubes := 50
  }
  surfaceArea cube = 2820 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l3411_341152


namespace NUMINAMATH_CALUDE_selection_method1_selection_method2_selection_method3_selection_method4_l3411_341174

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of athletes -/
def total_athletes : ℕ := 10

/-- The number of male athletes -/
def male_athletes : ℕ := 6

/-- The number of female athletes -/
def female_athletes : ℕ := 4

/-- The number of athletes to be selected -/
def selected_athletes : ℕ := 5

/-- The number of ways to select 3 males and 2 females -/
theorem selection_method1 : choose male_athletes 3 * choose female_athletes 2 = 120 := sorry

/-- The number of ways to select with at least one captain participating -/
theorem selection_method2 : 2 * choose 8 4 + choose 8 3 = 196 := sorry

/-- The number of ways to select with at least one female athlete -/
theorem selection_method3 : choose total_athletes selected_athletes - choose male_athletes selected_athletes = 246 := sorry

/-- The number of ways to select with both a captain and at least one female athlete -/
theorem selection_method4 : choose 9 4 + choose 8 4 - choose 5 4 = 191 := sorry

end NUMINAMATH_CALUDE_selection_method1_selection_method2_selection_method3_selection_method4_l3411_341174


namespace NUMINAMATH_CALUDE_sin_450_degrees_l3411_341129

theorem sin_450_degrees : Real.sin (450 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_450_degrees_l3411_341129


namespace NUMINAMATH_CALUDE_new_customers_calculation_l3411_341154

theorem new_customers_calculation (initial_customers final_customers : ℕ) :
  initial_customers = 3 →
  final_customers = 8 →
  final_customers - initial_customers = 5 := by
  sorry

end NUMINAMATH_CALUDE_new_customers_calculation_l3411_341154


namespace NUMINAMATH_CALUDE_line_equation_proof_l3411_341125

-- Define the point A
def A : ℝ × ℝ := (-1, 4)

-- Define the x-intercept
def x_intercept : ℝ := 3

-- Theorem statement
theorem line_equation_proof :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ (x + y - 3 = 0)) ∧ 
    (A.2 = m * A.1 + b) ∧
    (0 = m * x_intercept + b) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3411_341125


namespace NUMINAMATH_CALUDE_parabola_passes_origin_l3411_341117

/-- A parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

/-- The origin point (0, 0) -/
def origin : Point :=
  { x := 0, y := 0 }

/-- Check if a point lies on a parabola -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = para.f p.x

/-- The given parabola y = (x+2)^2 -/
def given_parabola : Parabola :=
  { f := λ x => (x + 2)^2 }

/-- Theorem: Rightward translation by 2 units makes the parabola pass through the origin -/
theorem parabola_passes_origin :
  ∃ (p : Point), lies_on (translate p 2 0) given_parabola ∧ p = origin := by
  sorry

end NUMINAMATH_CALUDE_parabola_passes_origin_l3411_341117


namespace NUMINAMATH_CALUDE_sum_of_distinct_divisors_of_2000_l3411_341180

def divisors_of_2000 : List ℕ := [1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000]

def is_sum_of_distinct_divisors (n : ℕ) : Prop :=
  ∃ (subset : List ℕ), subset.Nodup ∧ subset.Subset divisors_of_2000 ∧ subset.sum = n

theorem sum_of_distinct_divisors_of_2000 :
  ∀ n : ℕ, n > 0 ∧ n < 2000 → is_sum_of_distinct_divisors n :=
sorry

end NUMINAMATH_CALUDE_sum_of_distinct_divisors_of_2000_l3411_341180


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3411_341102

theorem complex_number_in_fourth_quadrant (a : ℝ) :
  let z : ℂ := (a^2 - 4*a + 5) - 6*Complex.I
  (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3411_341102


namespace NUMINAMATH_CALUDE_attendance_ratio_l3411_341177

/-- Proves that given the charges on three days and the average charge, 
    the ratio of attendance on these days is 4:1:5. -/
theorem attendance_ratio 
  (charge1 charge2 charge3 avg_charge : ℚ)
  (h1 : charge1 = 15)
  (h2 : charge2 = 15/2)
  (h3 : charge3 = 5/2)
  (h4 : avg_charge = 5)
  (x y z : ℚ) -- attendance on day 1, 2, and 3 respectively
  (h5 : (charge1 * x + charge2 * y + charge3 * z) / (x + y + z) = avg_charge) :
  ∃ (k : ℚ), k > 0 ∧ x = 4*k ∧ y = k ∧ z = 5*k := by
sorry


end NUMINAMATH_CALUDE_attendance_ratio_l3411_341177


namespace NUMINAMATH_CALUDE_one_third_of_36_l3411_341193

theorem one_third_of_36 : (1 / 3 : ℚ) * 36 = 12 := by sorry

end NUMINAMATH_CALUDE_one_third_of_36_l3411_341193


namespace NUMINAMATH_CALUDE_kendall_driving_distance_l3411_341188

/-- The distance Kendall drove with her mother in miles -/
def mother_distance : ℝ := 0.17

/-- The distance Kendall drove with her father in miles -/
def father_distance : ℝ := 0.5

/-- The distance Kendall drove with her friend in miles -/
def friend_distance : ℝ := 0.68

/-- The conversion factor from miles to kilometers -/
def mile_to_km : ℝ := 1.60934

/-- The total distance Kendall drove in kilometers -/
def total_distance_km : ℝ := (mother_distance + father_distance + friend_distance) * mile_to_km

theorem kendall_driving_distance :
  ∃ ε > 0, |total_distance_km - 2.17| < ε :=
sorry

end NUMINAMATH_CALUDE_kendall_driving_distance_l3411_341188


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3411_341126

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) ↔ (0 < a ∧ a < 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3411_341126


namespace NUMINAMATH_CALUDE_triangle_side_length_l3411_341146

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (a + b + c) * (b + c - a) = 3 * b * c →
  a = Real.sqrt 3 →
  Real.tan B = Real.sqrt 2 / 4 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B →
  c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C →
  b = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3411_341146


namespace NUMINAMATH_CALUDE_four_double_prime_value_l3411_341195

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- State the theorem
theorem four_double_prime_value : prime (prime 4) = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_double_prime_value_l3411_341195


namespace NUMINAMATH_CALUDE_pencils_left_l3411_341105

/-- Given two boxes of pencils with fourteen pencils each, prove that after giving away six pencils, the number of pencils left is 22. -/
theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given_away : ℕ) : 
  boxes = 2 → pencils_per_box = 14 → pencils_given_away = 6 →
  boxes * pencils_per_box - pencils_given_away = 22 := by
sorry

end NUMINAMATH_CALUDE_pencils_left_l3411_341105


namespace NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_59048_l3411_341158

/-- The greatest prime divisor of a natural number -/
def greatest_prime_divisor (n : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the digits of the greatest prime divisor of 59048 is 7 -/
theorem sum_digits_greatest_prime_divisor_59048 :
  sum_of_digits (greatest_prime_divisor 59048) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_digits_greatest_prime_divisor_59048_l3411_341158
