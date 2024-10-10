import Mathlib

namespace curve_L_properties_l2514_251496

/-- Definition of the curve L -/
def L (p : ℕ) (x y : ℤ) : Prop := 4 * y^2 = (x - p) * p

/-- A prime number is odd if it's not equal to 2 -/
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p ≠ 2

theorem curve_L_properties (p : ℕ) (hp : is_odd_prime p) :
  (∃ (S : Set (ℤ × ℤ)), Set.Infinite S ∧ ∀ (x y : ℤ), (x, y) ∈ S → y ≠ 0 ∧ L p x y) ∧
  (∀ (x y : ℤ), L p x y → ¬ ∃ (d : ℤ), d^2 = x^2 + y^2) :=
sorry

end curve_L_properties_l2514_251496


namespace tangent_line_equation_l2514_251434

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (m = f' 1) ∧
    (f 1 = m * 1 + b) ∧
    (m * 1 - f 1 + b = 0) ∧
    (m = 4 ∧ b = -2) := by
  sorry


end tangent_line_equation_l2514_251434


namespace toy_problem_solution_l2514_251435

/-- Represents the problem of Mia and her mom putting toys in a box -/
def ToyProblem (totalToys : ℕ) (putIn : ℕ) (takeOut : ℕ) (cycleTime : ℚ) : Prop :=
  let netIncrease := putIn - takeOut
  let cycles := (totalToys - 1) / netIncrease + 1
  cycles * cycleTime / 60 = 12.5

/-- The theorem statement for the toy problem -/
theorem toy_problem_solution :
  ToyProblem 50 5 3 (30 / 60) :=
sorry

end toy_problem_solution_l2514_251435


namespace prob_more_twos_than_fives_correct_l2514_251419

def num_dice : ℕ := 5
def num_sides : ℕ := 6

def prob_more_twos_than_fives : ℚ := 2721 / 7776

theorem prob_more_twos_than_fives_correct :
  let total_outcomes := num_sides ^ num_dice
  let equal_twos_and_fives := 2334
  (1 / 2) * (1 - equal_twos_and_fives / total_outcomes) = prob_more_twos_than_fives :=
by sorry

end prob_more_twos_than_fives_correct_l2514_251419


namespace degree_of_5x_cubed_plus_9_to_10_l2514_251464

/-- The degree of a polynomial of the form (ax³ + b)ⁿ where a and b are constants and n is a positive integer -/
def degree_of_cubic_plus_constant_to_power (a b : ℝ) (n : ℕ+) : ℕ :=
  3 * n

/-- Theorem stating that the degree of (5x³ + 9)¹⁰ is 30 -/
theorem degree_of_5x_cubed_plus_9_to_10 :
  degree_of_cubic_plus_constant_to_power 5 9 10 = 30 := by
  sorry

end degree_of_5x_cubed_plus_9_to_10_l2514_251464


namespace coin_count_l2514_251427

theorem coin_count (num_25_cent num_10_cent : ℕ) : 
  num_25_cent = 17 → num_10_cent = 17 → num_25_cent + num_10_cent = 34 := by
  sorry

end coin_count_l2514_251427


namespace max_ratio_two_digit_mean_50_l2514_251470

/-- Two-digit positive integer -/
def TwoDigitPositiveInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The mean of two numbers is 50 -/
def MeanIs50 (x y : ℕ) : Prop := (x + y) / 2 = 50

theorem max_ratio_two_digit_mean_50 :
  ∃ (x y : ℕ), TwoDigitPositiveInt x ∧ TwoDigitPositiveInt y ∧ MeanIs50 x y ∧
    ∀ (a b : ℕ), TwoDigitPositiveInt a → TwoDigitPositiveInt b → MeanIs50 a b →
      (a : ℚ) / b ≤ (x : ℚ) / y ∧ (x : ℚ) / y = 99 := by
  sorry

end max_ratio_two_digit_mean_50_l2514_251470


namespace base_conversion_2200_to_base9_l2514_251497

-- Define a function to convert a base 9 number to base 10
def base9ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

-- Theorem statement
theorem base_conversion_2200_to_base9 :
  base9ToBase10 [4, 1, 0, 3] = 2200 := by
  sorry

end base_conversion_2200_to_base9_l2514_251497


namespace student_selection_probability_l2514_251489

theorem student_selection_probability : 
  let total_students : ℕ := 4
  let selected_students : ℕ := 2
  let target_group : ℕ := 2
  let favorable_outcomes : ℕ := target_group * (total_students - target_group)
  let total_outcomes : ℕ := Nat.choose total_students selected_students
  (favorable_outcomes : ℚ) / total_outcomes = 2 / 3 := by
sorry

end student_selection_probability_l2514_251489


namespace root_implies_inequality_l2514_251474

theorem root_implies_inequality (a b : ℝ) 
  (h : (a + b + a) * (a + b + b) = 9) : a * b ≤ 1 := by
  sorry

end root_implies_inequality_l2514_251474


namespace marys_age_l2514_251428

/-- Given that Suzy is 20 years old now and in four years she will be twice Mary's age,
    prove that Mary is currently 8 years old. -/
theorem marys_age (suzy_age : ℕ) (mary_age : ℕ) : 
  suzy_age = 20 → 
  (suzy_age + 4 = 2 * (mary_age + 4)) → 
  mary_age = 8 := by
sorry

end marys_age_l2514_251428


namespace expression_value_l2514_251437

theorem expression_value : 
  (121^2 - 19^2) / (91^2 - 13^2) * ((91 - 13)*(91 + 13)) / ((121 - 19)*(121 + 19)) = 1 := by
  sorry

end expression_value_l2514_251437


namespace f_minus_g_greater_than_two_l2514_251456

noncomputable def f (x : ℝ) : ℝ := (2 - x^3) * Real.exp x

noncomputable def g (x : ℝ) : ℝ := Real.log x / x

theorem f_minus_g_greater_than_two (x : ℝ) (h : x ∈ Set.Ioo 0 1) : f x - g x > 2 := by
  sorry

end f_minus_g_greater_than_two_l2514_251456


namespace sine_phase_shift_specific_sine_phase_shift_l2514_251401

/-- The phase shift of a sine function y = A * sin(B * x + C) is -C/B -/
theorem sine_phase_shift (A B C : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ A * Real.sin (B * x + C)
  let phase_shift := -C / B
  ∀ x, f (x + phase_shift) = A * Real.sin (B * x)
  := by sorry

/-- The phase shift of y = 3 * sin(4x + π/4) is -π/16 -/
theorem specific_sine_phase_shift : 
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin (4 * x + π/4)
  let phase_shift := -π/16
  ∀ x, f (x + phase_shift) = 3 * Real.sin (4 * x)
  := by sorry

end sine_phase_shift_specific_sine_phase_shift_l2514_251401


namespace intersection_line_equation_l2514_251457

-- Define the two given lines
def l₁ (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def l₂ (x y : ℝ) : Prop := y = 1 - x

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := l₁ x y ∧ l₂ x y

-- Define the line passing through the intersection point and origin
def target_line (x y : ℝ) : Prop := 3 * x + 2 * y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∃ (x₀ y₀ : ℝ), intersection_point x₀ y₀ ∧
  ∀ (x y : ℝ), (x = 0 ∧ y = 0) ∨ (x = x₀ ∧ y = y₀) → target_line x y :=
sorry

end intersection_line_equation_l2514_251457


namespace motel_pricing_l2514_251482

/-- Represents the motel's pricing structure and guest stays. -/
structure MotelStay where
  flatFee : ℝ
  regularRate : ℝ
  discountRate : ℝ
  markStay : ℕ
  markCost : ℝ
  lucyStay : ℕ
  lucyCost : ℝ

/-- The motel's pricing satisfies the given conditions. -/
def validPricing (m : MotelStay) : Prop :=
  m.discountRate = 0.8 * m.regularRate ∧
  m.markStay = 5 ∧
  m.lucyStay = 7 ∧
  m.markCost = m.flatFee + 3 * m.regularRate + 2 * m.discountRate ∧
  m.lucyCost = m.flatFee + 3 * m.regularRate + 4 * m.discountRate ∧
  m.markCost = 310 ∧
  m.lucyCost = 410

/-- The theorem stating the correct flat fee and regular rate. -/
theorem motel_pricing (m : MotelStay) (h : validPricing m) :
  m.flatFee = 22.5 ∧ m.regularRate = 62.5 := by
  sorry

end motel_pricing_l2514_251482


namespace function_properties_l2514_251472

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + 2

theorem function_properties :
  -- Part 1: When a = 0, the zero of the function is x = 2
  (∃ x : ℝ, f 0 x = 0 ∧ x = 2) ∧
  
  -- Part 2: When a = 1, the range of m for solutions in [1,3] is [-1/4, 2]
  (∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 1 3 ∧ f 1 x = m) ↔ m ∈ Set.Icc (-1/4) 2) ∧
  
  -- Part 3: When a > 0, the solution set of f(x) > 0
  (∀ a : ℝ, a > 0 →
    (a = 1/2 → {x : ℝ | f a x > 0} = {x : ℝ | x ≠ 2}) ∧
    (0 < a ∧ a < 1/2 → {x : ℝ | f a x > 0} = {x : ℝ | x > 1/a ∨ x < 2}) ∧
    (a > 1/2 → {x : ℝ | f a x > 0} = {x : ℝ | x < 1/a ∨ x > 2}))
  := by sorry


end function_properties_l2514_251472


namespace f_sum_equals_negative_two_l2514_251426

def f (x : ℝ) : ℝ := x^3 - x - 1

theorem f_sum_equals_negative_two : 
  f 2023 + (deriv f) 2023 + f (-2023) - (deriv f) (-2023) = -2 := by
  sorry

end f_sum_equals_negative_two_l2514_251426


namespace min_value_sequence_l2514_251439

theorem min_value_sequence (a : ℕ → ℝ) (h1 : a 1 = 25) 
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n) : 
  ∀ n : ℕ, n ≥ 1 → a n / n ≥ 9 ∧ ∃ m : ℕ, m ≥ 1 ∧ a m / m = 9 :=
sorry

end min_value_sequence_l2514_251439


namespace quadratic_inequality_l2514_251433

theorem quadratic_inequality (x : ℝ) : 
  3 * x^2 + 2 * x - 3 > 10 - 2 * x ↔ x < (-2 - Real.sqrt 43) / 3 ∨ x > (-2 + Real.sqrt 43) / 3 := by
sorry

end quadratic_inequality_l2514_251433


namespace girls_together_arrangements_girls_separate_arrangements_l2514_251414

/-- The number of boys in the lineup -/
def num_boys : ℕ := 4

/-- The number of girls in the lineup -/
def num_girls : ℕ := 3

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of ways to arrange the lineup with girls together -/
def arrangements_girls_together : ℕ := 720

/-- The number of ways to arrange the lineup with no two girls together -/
def arrangements_girls_separate : ℕ := 1440

/-- Theorem stating the number of arrangements with girls together -/
theorem girls_together_arrangements :
  (num_girls.factorial * (num_boys + 1).factorial) = arrangements_girls_together := by sorry

/-- Theorem stating the number of arrangements with no two girls together -/
theorem girls_separate_arrangements :
  (num_boys.factorial * (Nat.choose (num_boys + 1) num_girls) * num_girls.factorial) = arrangements_girls_separate := by sorry

end girls_together_arrangements_girls_separate_arrangements_l2514_251414


namespace triangle_area_is_one_l2514_251413

/-- The area of a triangle bounded by the x-axis and two lines -/
def triangleArea (line1 : ℝ → ℝ → Prop) (line2 : ℝ → ℝ → Prop) : ℝ :=
  1 -- We define the area as 1 based on the problem statement

/-- The first line equation: y - 2x = 2 -/
def line1 (x y : ℝ) : Prop :=
  y - 2*x = 2

/-- The second line equation: 2y - x = 1 -/
def line2 (x y : ℝ) : Prop :=
  2*y - x = 1

/-- Theorem stating that the area of the triangle is 1 -/
theorem triangle_area_is_one :
  triangleArea line1 line2 = 1 := by
  sorry


end triangle_area_is_one_l2514_251413


namespace only_one_divides_power_minus_one_l2514_251480

theorem only_one_divides_power_minus_one :
  ∀ n : ℕ+, n ∣ (2^n.val - 1) → n = 1 := by
  sorry

end only_one_divides_power_minus_one_l2514_251480


namespace polynomial_simplification_l2514_251479

theorem polynomial_simplification (x : ℝ) : (x^2 - 4) * (x - 2) * (x + 2) = x^4 - 8*x^2 + 16 := by
  sorry

end polynomial_simplification_l2514_251479


namespace all_lamps_on_iff_even_l2514_251445

/-- Represents the state of a lamp (on or off) -/
inductive LampState
| On : LampState
| Off : LampState

/-- Represents a grid of lamps -/
def LampGrid (n : ℕ) := Fin n → Fin n → LampState

/-- Function to toggle a lamp state -/
def toggleLamp : LampState → LampState
| LampState.On => LampState.Off
| LampState.Off => LampState.On

/-- Function to press a switch at position (i, j) -/
def pressSwitch (grid : LampGrid n) (i j : Fin n) : LampGrid n :=
  fun x y => if x = i ∨ y = j then toggleLamp (grid x y) else grid x y

/-- Predicate to check if all lamps are on -/
def allLampsOn (grid : LampGrid n) : Prop :=
  ∀ i j, grid i j = LampState.On

/-- Main theorem: It's possible to achieve all lamps on iff n is even -/
theorem all_lamps_on_iff_even (n : ℕ) :
  (∀ (initialGrid : LampGrid n), ∃ (switches : List (Fin n × Fin n)),
    allLampsOn (switches.foldl (fun g (i, j) => pressSwitch g i j) initialGrid)) ↔
  Even n :=
sorry

end all_lamps_on_iff_even_l2514_251445


namespace block_with_t_hole_difference_l2514_251421

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents the dimensions and position of a T-shaped hole -/
structure THole where
  height : ℕ
  length : ℕ
  width : ℕ
  distanceFromFront : ℕ

/-- Calculates the number of cubes needed to create a block with a T-shaped hole -/
def cubesNeededWithHole (block : BlockDimensions) (hole : THole) : ℕ :=
  block.length * block.width * block.depth - (hole.height * hole.length + hole.width - 1)

/-- Theorem stating that a 7x7x6 block with the given T-shaped hole requires 3 fewer cubes -/
theorem block_with_t_hole_difference :
  let block := BlockDimensions.mk 7 7 6
  let hole := THole.mk 1 3 2 3
  block.length * block.width * block.depth - cubesNeededWithHole block hole = 3 := by
  sorry


end block_with_t_hole_difference_l2514_251421


namespace candies_in_packet_l2514_251447

/-- The number of candies in a packet of candy -/
def candies_per_packet : ℕ := 18

/-- The number of packets Bobby buys -/
def num_packets : ℕ := 2

/-- The number of days per week Bobby eats 2 candies -/
def days_eating_two : ℕ := 5

/-- The number of days per week Bobby eats 1 candy -/
def days_eating_one : ℕ := 2

/-- The number of weeks it takes to finish the packets -/
def weeks_to_finish : ℕ := 3

/-- Theorem stating the number of candies in a packet -/
theorem candies_in_packet :
  candies_per_packet * num_packets = 
  (days_eating_two * 2 + days_eating_one * 1) * weeks_to_finish :=
by sorry

end candies_in_packet_l2514_251447


namespace containers_per_truck_is_160_l2514_251458

/-- The number of trucks with 20 boxes each -/
def trucks_with_20_boxes : ℕ := 7

/-- The number of trucks with 12 boxes each -/
def trucks_with_12_boxes : ℕ := 5

/-- The number of boxes on trucks with 20 boxes -/
def boxes_on_20_box_trucks : ℕ := 20

/-- The number of boxes on trucks with 12 boxes -/
def boxes_on_12_box_trucks : ℕ := 12

/-- The number of containers of oil in each box -/
def containers_per_box : ℕ := 8

/-- The number of trucks for redistribution -/
def redistribution_trucks : ℕ := 10

/-- The total number of containers of oil -/
def total_containers : ℕ := 
  (trucks_with_20_boxes * boxes_on_20_box_trucks + 
   trucks_with_12_boxes * boxes_on_12_box_trucks) * containers_per_box

/-- The number of containers per truck after redistribution -/
def containers_per_truck : ℕ := total_containers / redistribution_trucks

theorem containers_per_truck_is_160 : containers_per_truck = 160 := by
  sorry

end containers_per_truck_is_160_l2514_251458


namespace zero_in_interval_l2514_251461

-- Define the function f(x) = x^5 + x - 3
def f (x : ℝ) : ℝ := x^5 + x - 3

-- Theorem statement
theorem zero_in_interval :
  ∃! x : ℝ, x ∈ Set.Icc 1 2 ∧ f x = 0 := by sorry

end zero_in_interval_l2514_251461


namespace pattern_symmetries_l2514_251454

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  point : Point
  direction : Point

/-- Represents the pattern on the line -/
structure Pattern where
  line : Line
  unit_length : ℝ  -- Length of one repeating unit

/-- Represents a rigid motion transformation -/
inductive RigidMotion
  | Rotation (center : Point) (angle : ℝ)
  | Translation (direction : Point) (distance : ℝ)
  | Reflection (line : Line)

/-- Checks if a transformation preserves the pattern -/
def preserves_pattern (p : Pattern) (t : RigidMotion) : Prop :=
  sorry

theorem pattern_symmetries (p : Pattern) :
  (∃ (center : Point), preserves_pattern p (RigidMotion.Rotation center (2 * π / 3))) ∧
  (∃ (center : Point), preserves_pattern p (RigidMotion.Rotation center (4 * π / 3))) ∧
  (preserves_pattern p (RigidMotion.Translation p.line.direction p.unit_length)) ∧
  (preserves_pattern p (RigidMotion.Reflection p.line)) ∧
  (∃ (perp_line : Line), 
    (perp_line.direction.x * p.line.direction.x + perp_line.direction.y * p.line.direction.y = 0) ∧
    preserves_pattern p (RigidMotion.Reflection perp_line)) :=
by sorry

end pattern_symmetries_l2514_251454


namespace complex_multiplication_l2514_251475

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_multiplication :
  (1 + i) * i = -1 + i :=
sorry

end complex_multiplication_l2514_251475


namespace quadratic_function_property_l2514_251498

/-- Given a quadratic function f(x) = x^2 - 2x + 3, if f(m) = f(n) where m ≠ n, 
    then f(m + n) = 3 -/
theorem quadratic_function_property (m n : ℝ) (h : m ≠ n) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + 3
  f m = f n → f (m + n) = 3 := by
sorry

end quadratic_function_property_l2514_251498


namespace equation_solution_l2514_251466

theorem equation_solution (x y : ℝ) (h : x - y / 2 = 1) : y = 2 * x - 2 := by
  sorry

end equation_solution_l2514_251466


namespace second_prime_range_l2514_251406

theorem second_prime_range (p q : ℕ) (hp : Prime p) (hq : Prime q) : 
  15 < p * q ∧ p * q ≤ 70 ∧ 2 < p ∧ p < 6 ∧ p * q = 69 → q = 23 := by
  sorry

end second_prime_range_l2514_251406


namespace digit_multiplication_theorem_l2514_251488

/-- A function that checks if a number is a digit (0-9) -/
def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- A function that converts a three-digit number to its decimal representation -/
def three_digit_to_decimal (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

/-- A function that converts a four-digit number to its decimal representation -/
def four_digit_to_decimal (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

theorem digit_multiplication_theorem (A B C D : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D)
  (h_multiplication : three_digit_to_decimal A B C * D = four_digit_to_decimal A B C D) :
  C + D = 5 := by
  sorry

end digit_multiplication_theorem_l2514_251488


namespace can_guess_number_l2514_251484

theorem can_guess_number (n : Nat) (q : Nat) : n ≤ 1000 → q = 10 → 2^q ≥ n → True := by
  sorry

end can_guess_number_l2514_251484


namespace minimize_sum_of_distances_l2514_251425

/-- Given points P, Q, and R in ℝ², if R is chosen to minimize the sum of distances |PR| + |RQ|, then R lies on the line segment PQ. -/
theorem minimize_sum_of_distances (P Q R : ℝ × ℝ) :
  P = (-2, -2) →
  Q = (0, -1) →
  R.1 = 2 →
  (∀ S : ℝ × ℝ, dist P R + dist R Q ≤ dist P S + dist S Q) →
  R.2 = 0 := by sorry


end minimize_sum_of_distances_l2514_251425


namespace green_mandm_probability_l2514_251455

structure MandMJar :=
  (green : ℕ) (red : ℕ) (blue : ℕ) (orange : ℕ) (yellow : ℕ) (purple : ℕ) (brown : ℕ) (pink : ℕ)

def initial_jar : MandMJar :=
  ⟨35, 25, 10, 15, 0, 0, 0, 0⟩

def carter_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green - 20, jar.red - 8, jar.blue, jar.orange, jar.yellow, jar.purple, jar.brown, jar.pink⟩

def sister_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red / 2, jar.blue, jar.orange, jar.yellow + 14, jar.purple, jar.brown, jar.pink⟩

def alex_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red, jar.blue, 0, jar.yellow - 3, jar.purple + 8, jar.brown, jar.pink⟩

def cousin_eats (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red, 0, jar.orange, jar.yellow, jar.purple, jar.brown + 10, jar.pink⟩

def sister_adds_pink (jar : MandMJar) : MandMJar :=
  ⟨jar.green, jar.red, jar.blue, jar.orange, jar.yellow, jar.purple, jar.brown, jar.pink + 10⟩

def total_mandms (jar : MandMJar) : ℕ :=
  jar.green + jar.red + jar.blue + jar.orange + jar.yellow + jar.purple + jar.brown + jar.pink

theorem green_mandm_probability :
  let final_jar := sister_adds_pink (cousin_eats (alex_eats (sister_eats (carter_eats initial_jar))))
  (final_jar.green : ℚ) / ((total_mandms final_jar - 1) : ℚ) = 15 / 61 := by sorry

end green_mandm_probability_l2514_251455


namespace taxi_cost_proof_l2514_251446

/-- The cost per mile for a taxi ride to the airport -/
def cost_per_mile : ℚ := 5 / 14

/-- Mike's distance in miles -/
def mike_distance : ℚ := 28

theorem taxi_cost_proof :
  ∀ (x : ℚ),
  (2.5 + x * mike_distance = 2.5 + 5 + x * 14) →
  x = cost_per_mile := by
  sorry

end taxi_cost_proof_l2514_251446


namespace max_n_theorem_l2514_251486

/-- Represents a convex polygon with interior points -/
structure ConvexPolygonWithInteriorPoints where
  n : ℕ  -- number of vertices in the polygon
  interior_points : ℕ  -- number of interior points
  no_collinear : Bool  -- no three points are collinear

/-- Calculates the number of triangles formed in a polygon with interior points -/
def num_triangles (p : ConvexPolygonWithInteriorPoints) : ℕ :=
  p.n + p.interior_points + 198

/-- The maximum value of n for which no more than 300 triangles are formed -/
def max_n_for_300_triangles : ℕ := 102

/-- Theorem stating the maximum value of n for which no more than 300 triangles are formed -/
theorem max_n_theorem (p : ConvexPolygonWithInteriorPoints) 
    (h1 : p.interior_points = 100)
    (h2 : p.no_collinear = true) :
    (∀ m : ℕ, m > max_n_for_300_triangles → num_triangles { n := m, interior_points := 100, no_collinear := true } > 300) ∧
    num_triangles { n := max_n_for_300_triangles, interior_points := 100, no_collinear := true } ≤ 300 :=
  sorry

end max_n_theorem_l2514_251486


namespace equilateral_triangle_perimeter_l2514_251424

/-- 
Given an equilateral triangle where one of its sides is also a side of an isosceles triangle,
this theorem proves that if the isosceles triangle has a perimeter of 65 and a base of 25,
then the perimeter of the equilateral triangle is 60.
-/
theorem equilateral_triangle_perimeter 
  (s : ℝ) 
  (h_isosceles_perimeter : s + s + 25 = 65) 
  (h_equilateral_side : s > 0) : 
  3 * s = 60 := by
sorry

end equilateral_triangle_perimeter_l2514_251424


namespace black_balls_count_l2514_251405

theorem black_balls_count (total : ℕ) (red : ℕ) (prob_white : ℚ) 
  (h_total : total = 100)
  (h_red : red = 30)
  (h_prob_white : prob_white = 47/100) :
  total - red - (total * prob_white).num = 23 := by
sorry

end black_balls_count_l2514_251405


namespace quadratic_curve_coefficient_l2514_251402

theorem quadratic_curve_coefficient (p q y1 y2 : ℝ) : 
  (y1 = p + q + 5) →
  (y2 = p - q + 5) →
  (y1 + y2 = 14) →
  p = 2 := by
sorry

end quadratic_curve_coefficient_l2514_251402


namespace f_at_2_equals_neg_26_l2514_251471

/-- Given a function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem f_at_2_equals_neg_26 (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = x^5 + a*x^3 + b*x - 8)
    (h2 : f (-2) = 10) : 
  f 2 = -26 := by
sorry

end f_at_2_equals_neg_26_l2514_251471


namespace sum_of_digits_multiple_of_990_l2514_251404

/-- Given a six-digit number 123abc that is a multiple of 990, 
    prove that the sum of its hundreds, tens, and units digits (a + b + c) is 12 -/
theorem sum_of_digits_multiple_of_990 (a b c : ℕ) : 
  (0 < a) → (a < 10) →
  (0 ≤ b) → (b < 10) →
  (0 ≤ c) → (c < 10) →
  (123000 + 100 * a + 10 * b + c) % 990 = 0 →
  a + b + c = 12 := by
sorry

end sum_of_digits_multiple_of_990_l2514_251404


namespace min_value_sum_reciprocals_l2514_251411

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
  (pos_s : 0 < s) (pos_t : 0 < t) (pos_u : 0 < u)
  (sum_eq_8 : p + q + r + s + t + u = 8) :
  2/p + 4/q + 9/r + 16/s + 25/t + 36/u ≥ 98 := by
  sorry

end min_value_sum_reciprocals_l2514_251411


namespace pennies_indeterminate_l2514_251436

/-- Represents the number of coins Sandy has -/
structure SandyCoins where
  pennies : ℕ
  nickels : ℕ

/-- Represents the state of Sandy's coins before and after her dad's borrowing -/
structure SandyState where
  initial : SandyCoins
  borrowed_nickels : ℕ
  remaining : SandyCoins

/-- Defines the conditions of the problem -/
def valid_state (s : SandyState) : Prop :=
  s.initial.nickels = 31 ∧
  s.borrowed_nickels = 20 ∧
  s.remaining.nickels = 11 ∧
  s.initial.nickels = s.remaining.nickels + s.borrowed_nickels

/-- Theorem stating that the initial number of pennies cannot be determined -/
theorem pennies_indeterminate (s1 s2 : SandyState) :
  valid_state s1 → valid_state s2 → s1.initial.pennies ≠ s2.initial.pennies → True := by
  sorry

end pennies_indeterminate_l2514_251436


namespace johns_pool_depth_l2514_251432

theorem johns_pool_depth (sarah_depth john_depth : ℕ) : 
  sarah_depth = 5 →
  john_depth = 2 * sarah_depth + 5 →
  john_depth = 15 := by
  sorry

end johns_pool_depth_l2514_251432


namespace intersection_when_m_is_2_subset_iff_m_leq_1_l2514_251495

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = Real.sqrt (3 - 2*x) ∧ x ∈ Set.Icc (-13/2) (3/2)}
def B (m : ℝ) : Set ℝ := Set.Icc (1 - m) (m + 1)

-- Statement 1: When m = 2, A ∩ B = [0, 3]
theorem intersection_when_m_is_2 : A ∩ B 2 = Set.Icc 0 3 := by sorry

-- Statement 2: B ⊆ A if and only if m ≤ 1
theorem subset_iff_m_leq_1 : ∀ m, B m ⊆ A ↔ m ≤ 1 := by sorry

end intersection_when_m_is_2_subset_iff_m_leq_1_l2514_251495


namespace total_tabs_is_300_l2514_251477

/-- Calculates the total number of tabs opened across all browsers --/
def totalTabs (numBrowsers : ℕ) (windowsPerBrowser : ℕ) (initialTabsPerWindow : ℕ) (additionalTabsPerTwelve : ℕ) : ℕ :=
  let tabsPerWindow := initialTabsPerWindow + additionalTabsPerTwelve
  let tabsPerBrowser := tabsPerWindow * windowsPerBrowser
  tabsPerBrowser * numBrowsers

/-- Proves that the total number of tabs is 300 given the specified conditions --/
theorem total_tabs_is_300 :
  totalTabs 4 5 12 3 = 300 := by
  sorry

end total_tabs_is_300_l2514_251477


namespace average_extra_chores_l2514_251431

/-- Proves that given the specified conditions, the average number of extra chores per week is 15 -/
theorem average_extra_chores
  (fixed_allowance : ℝ)
  (extra_chore_pay : ℝ)
  (total_weeks : ℕ)
  (total_earned : ℝ)
  (h1 : fixed_allowance = 20)
  (h2 : extra_chore_pay = 1.5)
  (h3 : total_weeks = 10)
  (h4 : total_earned = 425) :
  (total_earned / total_weeks - fixed_allowance) / extra_chore_pay = 15 := by
  sorry

#check average_extra_chores

end average_extra_chores_l2514_251431


namespace distribution_problem_l2514_251410

theorem distribution_problem (total_amount : ℕ) (first_group : ℕ) (difference : ℕ) (second_group : ℕ) :
  total_amount = 5040 →
  first_group = 14 →
  difference = 80 →
  (total_amount / first_group) = (total_amount / second_group + difference) →
  second_group = 18 := by
sorry

end distribution_problem_l2514_251410


namespace parabola_translation_l2514_251478

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola -/
def original_parabola : Parabola := { a := 1, b := -2, c := 4 }

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + d }

/-- Translates a parabola horizontally -/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b - 2 * p.a * d, c := p.c + p.a * d^2 - p.b * d }

/-- The resulting parabola after translation -/
def translated_parabola : Parabola :=
  translate_horizontal (translate_vertical original_parabola 3) 1

theorem parabola_translation :
  translated_parabola = { a := 1, b := -4, c := 10 } := by
  sorry

end parabola_translation_l2514_251478


namespace inverse_proportion_problem_l2514_251440

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 52) (h3 : x = 3 * y) :
  ∃ y_new : ℝ, ((-10 : ℝ) * y_new = k) ∧ (y_new = -50.7) := by
  sorry

end inverse_proportion_problem_l2514_251440


namespace double_reflection_of_F_l2514_251460

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_over_y_equals_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem double_reflection_of_F :
  let F : ℝ × ℝ := (5, 2)
  let F' := reflect_over_y_axis F
  let F'' := reflect_over_y_equals_x F'
  F'' = (2, -5) := by sorry

end double_reflection_of_F_l2514_251460


namespace hour_hand_path_l2514_251481

/-- The number of times the hour hand covers its path in a day -/
def coverages_per_day : ℕ := 2

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of hours for one full rotation of the hour hand -/
def hours_per_rotation : ℕ := 12

/-- The path covered by the hour hand in one rotation, in degrees -/
def path_per_rotation : ℝ := 360

theorem hour_hand_path :
  path_per_rotation = 360 :=
sorry

end hour_hand_path_l2514_251481


namespace average_of_sequence_l2514_251465

theorem average_of_sequence (z : ℝ) : 
  let sequence := [0, 3*z, 6*z, 12*z, 24*z]
  (sequence.sum / sequence.length : ℝ) = 9*z := by
sorry

end average_of_sequence_l2514_251465


namespace money_division_l2514_251494

theorem money_division (total : ℚ) (a b c : ℚ) 
  (h1 : total = 527)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) :
  b = 93 := by
  sorry

end money_division_l2514_251494


namespace students_like_both_sports_l2514_251469

/-- The number of students who like basketball -/
def B : ℕ := 10

/-- The number of students who like cricket -/
def C : ℕ := 8

/-- The number of students who like either basketball or cricket or both -/
def B_union_C : ℕ := 14

/-- The number of students who like both basketball and cricket -/
def B_intersect_C : ℕ := B + C - B_union_C

theorem students_like_both_sports : B_intersect_C = 4 := by
  sorry

end students_like_both_sports_l2514_251469


namespace set_membership_implies_value_l2514_251441

theorem set_membership_implies_value (m : ℤ) : 3 ∈ ({1, m + 2} : Set ℤ) → m = 1 := by
  sorry

end set_membership_implies_value_l2514_251441


namespace ratio_of_system_l2514_251449

theorem ratio_of_system (x y a b : ℝ) (h1 : 4 * x - 2 * y = a) (h2 : 6 * y - 12 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 3 := by
  sorry

end ratio_of_system_l2514_251449


namespace yella_computer_usage_l2514_251490

/-- Yella's computer usage problem -/
theorem yella_computer_usage 
  (last_week_usage : ℕ) 
  (this_week_first_4_days : ℕ) 
  (this_week_last_3_days : ℕ) 
  (next_week_weekday_classes : ℕ) 
  (next_week_weekday_gaming : ℕ) 
  (next_week_weekend_usage : ℕ) : 
  last_week_usage = 91 →
  this_week_first_4_days = 8 →
  this_week_last_3_days = 10 →
  next_week_weekday_classes = 5 →
  next_week_weekday_gaming = 3 →
  next_week_weekend_usage = 12 →
  (last_week_usage - (4 * this_week_first_4_days + 3 * this_week_last_3_days) = 29) ∧
  (last_week_usage - (5 * (next_week_weekday_classes + next_week_weekday_gaming) + 2 * next_week_weekend_usage) = 27) :=
by sorry

end yella_computer_usage_l2514_251490


namespace revenue_maximized_at_optimal_price_l2514_251422

/-- Revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (145 - 7 * p)

/-- The optimal price that maximizes revenue --/
def optimal_price : ℕ := 10

theorem revenue_maximized_at_optimal_price :
  ∀ p : ℕ, p ≤ 30 → R p ≤ R optimal_price := by
  sorry

end revenue_maximized_at_optimal_price_l2514_251422


namespace parrot_female_fraction_l2514_251438

theorem parrot_female_fraction (total_birds : ℝ) (female_parrot_fraction : ℝ) : 
  (3 / 5 : ℝ) * total_birds +                   -- number of parrots
  (2 / 5 : ℝ) * total_birds =                   -- number of toucans
  total_birds ∧                                 -- total number of birds
  (3 / 4 : ℝ) * ((2 / 5 : ℝ) * total_birds) +   -- number of female toucans
  female_parrot_fraction * ((3 / 5 : ℝ) * total_birds) = -- number of female parrots
  (1 / 2 : ℝ) * total_birds →                   -- total number of female birds
  female_parrot_fraction = (1 / 3 : ℝ) :=
by sorry

end parrot_female_fraction_l2514_251438


namespace candidate_a_republican_voters_l2514_251448

theorem candidate_a_republican_voters (total : ℝ) (h_total_pos : total > 0) : 
  let dem_percent : ℝ := 0.7
  let rep_percent : ℝ := 1 - dem_percent
  let dem_for_a_percent : ℝ := 0.8
  let total_for_a_percent : ℝ := 0.65
  let rep_for_a_percent : ℝ := 
    (total_for_a_percent - dem_percent * dem_for_a_percent) / rep_percent
  rep_for_a_percent = 0.3 := by
sorry

end candidate_a_republican_voters_l2514_251448


namespace maya_lifting_improvement_l2514_251416

theorem maya_lifting_improvement (america_initial : ℕ) (america_peak : ℕ) : 
  america_initial = 240 →
  america_peak = 300 →
  (america_peak / 2 : ℕ) - (america_initial / 4 : ℕ) = 90 := by
  sorry

end maya_lifting_improvement_l2514_251416


namespace T_100_value_l2514_251400

/-- The original sequence a_n -/
def a (n : ℕ) : ℕ := 2^(n-1)

/-- The number of inserted terms between a_k and a_{k+1} -/
def inserted_count (k : ℕ) : ℕ := k

/-- The value of inserted terms between a_k and a_{k+1} -/
def inserted_value (k : ℕ) : ℤ := (-1)^k * k

/-- The sum of the first n terms of the new sequence b_n -/
noncomputable def T (n : ℕ) : ℤ := sorry

/-- The theorem to prove -/
theorem T_100_value : T 100 = 8152 := by sorry

end T_100_value_l2514_251400


namespace min_value_of_f_l2514_251485

/-- The function f(x) = (x^2 + ax - 1)e^(x-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 
  ((2*x + a) * Real.exp (x - 1)) + ((x^2 + a*x - 1) * Real.exp (x - 1))

theorem min_value_of_f (a : ℝ) :
  (f_deriv a (-2) = 0) →  -- x = -2 is an extremum point
  (∃ x, f a x = -1) ∧ (∀ y, f a y ≥ -1) := by
  sorry

end min_value_of_f_l2514_251485


namespace charity_ticket_revenue_l2514_251442

/-- Represents the revenue from full-price tickets in a charity event -/
def revenue_full_price (full_price : ℝ) (num_full_price : ℝ) : ℝ :=
  full_price * num_full_price

/-- Represents the revenue from discounted tickets in a charity event -/
def revenue_discounted (full_price : ℝ) (num_discounted : ℝ) : ℝ :=
  0.75 * full_price * num_discounted

/-- Theorem stating that the revenue from full-price tickets can be determined -/
theorem charity_ticket_revenue 
  (full_price : ℝ) 
  (num_full_price num_discounted : ℝ) 
  (h1 : num_full_price + num_discounted = 150)
  (h2 : revenue_full_price full_price num_full_price + 
        revenue_discounted full_price num_discounted = 2250)
  : ∃ (r : ℝ), revenue_full_price full_price num_full_price = r :=
by
  sorry


end charity_ticket_revenue_l2514_251442


namespace min_value_exponential_sum_l2514_251463

theorem min_value_exponential_sum (x y : ℝ) (h : 2 * x + 3 * y = 6) :
  ∃ (m : ℝ), m = 16 ∧ ∀ a b, 2 * a + 3 * b = 6 → 4^a + 8^b ≥ m :=
sorry

end min_value_exponential_sum_l2514_251463


namespace football_players_l2514_251459

theorem football_players (total : ℕ) (cricket : ℕ) (neither : ℕ) (both : ℕ) 
  (h1 : total = 410)
  (h2 : cricket = 175)
  (h3 : neither = 50)
  (h4 : both = 140) :
  total - cricket + both - neither = 375 := by
  sorry

end football_players_l2514_251459


namespace power_product_equality_l2514_251467

theorem power_product_equality : (-0.25)^2014 * (-4)^2015 = -4 := by
  sorry

end power_product_equality_l2514_251467


namespace tax_reduction_l2514_251403

theorem tax_reduction (T C X : ℝ) (h1 : T > 0) (h2 : C > 0) (h3 : X > 0) : 
  (T * (1 - X / 100) * (C * 1.2) = 0.84 * (T * C)) → X = 30 := by
  sorry

end tax_reduction_l2514_251403


namespace odd_digits_base4_523_l2514_251443

/-- Represents a digit in base 4 --/
def Base4Digit := Fin 4

/-- Converts a natural number to its base 4 representation --/
def toBase4 (n : ℕ) : List Base4Digit := sorry

/-- Checks if a Base4Digit is odd --/
def isOddBase4Digit (d : Base4Digit) : Bool := sorry

/-- Counts the number of odd digits in a list of Base4Digits --/
def countOddDigits (digits : List Base4Digit) : ℕ := sorry

theorem odd_digits_base4_523 :
  countOddDigits (toBase4 523) = 2 := by sorry

end odd_digits_base4_523_l2514_251443


namespace oil_volume_in_tank_l2514_251429

/-- The volume of oil in a cylindrical tank with given dimensions and mixture ratio -/
theorem oil_volume_in_tank (tank_height : ℝ) (tank_diameter : ℝ) (fill_percentage : ℝ) 
  (oil_ratio : ℝ) (water_ratio : ℝ) (h_height : tank_height = 8) 
  (h_diameter : tank_diameter = 3) (h_fill : fill_percentage = 0.75) 
  (h_ratio : oil_ratio / (oil_ratio + water_ratio) = 3 / 10) : 
  (fill_percentage * π * (tank_diameter / 2)^2 * tank_height) * (oil_ratio / (oil_ratio + water_ratio)) = 4.05 * π := by
sorry

end oil_volume_in_tank_l2514_251429


namespace instantaneous_velocity_at_4_seconds_l2514_251418

-- Define the displacement function
def s (t : ℝ) : ℝ := 3 * t^2 + t + 4

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 6 * t + 1

-- State the theorem
theorem instantaneous_velocity_at_4_seconds :
  v 4 = 25 := by sorry

end instantaneous_velocity_at_4_seconds_l2514_251418


namespace unique_two_digit_number_l2514_251499

theorem unique_two_digit_number : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧
  (∃ a b : ℕ, a ≠ b ∧ a < 10 ∧ b < 10 ∧ n = 10 * a + b) ∧
  n^2 = (n / 10 + n % 10)^3 :=
by
  -- The proof goes here
  sorry

end unique_two_digit_number_l2514_251499


namespace daniel_correct_answers_l2514_251453

/-- Represents a mathematics competition --/
structure MathCompetition where
  total_problems : ℕ
  points_correct : ℕ
  points_incorrect : ℤ

/-- Represents a contestant's performance --/
structure ContestantPerformance where
  correct_answers : ℕ
  incorrect_answers : ℕ
  total_score : ℤ

/-- The specific competition Daniel participated in --/
def danielCompetition : MathCompetition :=
  { total_problems := 12
  , points_correct := 4
  , points_incorrect := -3 }

/-- Calculates the score based on correct and incorrect answers --/
def calculateScore (comp : MathCompetition) (perf : ContestantPerformance) : ℤ :=
  (comp.points_correct : ℤ) * perf.correct_answers + comp.points_incorrect * perf.incorrect_answers

/-- Theorem stating that Daniel must have answered 9 questions correctly --/
theorem daniel_correct_answers (comp : MathCompetition) (perf : ContestantPerformance) :
    comp = danielCompetition →
    perf.correct_answers + perf.incorrect_answers = comp.total_problems →
    calculateScore comp perf = 21 →
    perf.correct_answers = 9 := by
  sorry

end daniel_correct_answers_l2514_251453


namespace oblique_projection_preserves_parallelogram_l2514_251462

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram in 2D space -/
structure Parallelogram where
  a : Point2D
  b : Point2D
  c : Point2D
  d : Point2D

/-- Represents an oblique projection transformation -/
def ObliqueProjection := Point2D → Point2D

/-- Checks if four points form a parallelogram -/
def isParallelogram (a b c d : Point2D) : Prop := sorry

/-- The theorem stating that oblique projection preserves parallelograms -/
theorem oblique_projection_preserves_parallelogram 
  (p : Parallelogram) (proj : ObliqueProjection) :
  let p' := Parallelogram.mk 
    (proj p.a) (proj p.b) (proj p.c) (proj p.d)
  isParallelogram p'.a p'.b p'.c p'.d := by sorry

end oblique_projection_preserves_parallelogram_l2514_251462


namespace box_balls_count_l2514_251444

theorem box_balls_count : ∃ x : ℕ, (x > 20 ∧ x < 30 ∧ x - 20 = 30 - x) ∧ x = 25 := by
  sorry

end box_balls_count_l2514_251444


namespace common_integer_root_l2514_251409

theorem common_integer_root (a : ℤ) : 
  (∃ x : ℤ, a * x + a = 7 ∧ 3 * x - a = 17) ↔ a = 1 :=
by sorry

end common_integer_root_l2514_251409


namespace systematic_sampling_fourth_element_l2514_251487

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : ℕ → ℕ :=
  fun n => start + (n - 1) * (total / sampleSize)

theorem systematic_sampling_fourth_element :
  let total := 800
  let sampleSize := 50
  let interval := total / sampleSize
  let start := 7
  (systematicSample total sampleSize start 4 = 55) ∧ 
  (49 ≤ systematicSample total sampleSize start 4) ∧ 
  (systematicSample total sampleSize start 4 ≤ 64) := by
  sorry

end systematic_sampling_fourth_element_l2514_251487


namespace max_n_for_factorization_l2514_251415

theorem max_n_for_factorization : 
  (∃ (n : ℤ), ∀ (x : ℝ), ∃ (A B : ℤ), 
    6 * x^2 + n * x + 144 = (6 * x + A) * (x + B)) ∧
  (∀ (m : ℤ), m > 865 → 
    ¬∃ (A B : ℤ), ∀ (x : ℝ), 6 * x^2 + m * x + 144 = (6 * x + A) * (x + B)) :=
by sorry

end max_n_for_factorization_l2514_251415


namespace line_equation_proof_l2514_251468

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- The projection point P(-2,1) -/
def projection_point : Point := ⟨-2, 1⟩

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The line passing through two points -/
def line_through_points (p1 p2 : Point) : Line :=
  let a := p2.y - p1.y
  let b := p1.x - p2.x
  let c := p2.x * p1.y - p1.x * p2.y
  ⟨a, b, c⟩

/-- The line perpendicular to a given line passing through a point -/
def perpendicular_line_through_point (l : Line) (p : Point) : Line :=
  ⟨l.b, -l.a, l.a * p.y - l.b * p.x⟩

theorem line_equation_proof (L : Line) : 
  (point_on_line projection_point L) ∧ 
  (perpendicular L (line_through_points origin projection_point)) →
  L = ⟨2, -1, 5⟩ := by
  sorry


end line_equation_proof_l2514_251468


namespace _l2514_251430

def smallest_angle_theorem (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b c : V) (ha : ‖a‖ = 3) (hb : ‖b‖ = 2) (hc : ‖c‖ = 5) (habc : a + b + c = 0) :
  Real.arccos (inner a c / (‖a‖ * ‖c‖)) = π := by sorry

end _l2514_251430


namespace weaving_sum_l2514_251423

/-- The sum of an arithmetic sequence with first term 5, common difference 16/29, and 30 terms -/
theorem weaving_sum : 
  let a₁ : ℚ := 5
  let d : ℚ := 16 / 29
  let n : ℕ := 30
  (n : ℚ) * a₁ + (n * (n - 1) : ℚ) / 2 * d = 390 := by
  sorry

end weaving_sum_l2514_251423


namespace max_abs_sum_quadratic_coeff_l2514_251452

theorem max_abs_sum_quadratic_coeff (a b c : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → |a*x^2 + b*x + c| ≤ 1) → 
  |a| + |b| + |c| ≤ 3 ∧ ∃ a' b' c' : ℝ, 
    (∀ x : ℝ, |x| ≤ 1 → |a'*x^2 + b'*x + c'| ≤ 1) ∧ 
    |a'| + |b'| + |c'| = 3 :=
sorry

end max_abs_sum_quadratic_coeff_l2514_251452


namespace combination_equality_l2514_251420

theorem combination_equality (x : ℕ) : (Nat.choose 9 x = Nat.choose 9 (2*x - 3)) → (x = 3 ∨ x = 4) :=
by sorry

end combination_equality_l2514_251420


namespace system_solution_ratio_l2514_251417

theorem system_solution_ratio (x y a b : ℝ) 
  (h1 : 6 * x - 4 * y = a)
  (h2 : 6 * y - 9 * x = b)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hb : b ≠ 0) :
  a / b = -2 / 3 := by
sorry

end system_solution_ratio_l2514_251417


namespace brown_ball_weight_calculation_l2514_251491

/-- The weight of the blue ball in pounds -/
def blue_ball_weight : ℝ := 6

/-- The total weight of both balls in pounds -/
def total_weight : ℝ := 9.12

/-- The weight of the brown ball in pounds -/
def brown_ball_weight : ℝ := total_weight - blue_ball_weight

theorem brown_ball_weight_calculation :
  brown_ball_weight = 3.12 := by sorry

end brown_ball_weight_calculation_l2514_251491


namespace multiply_1307_by_1307_l2514_251451

theorem multiply_1307_by_1307 : 1307 * 1307 = 1709249 := by
  sorry

end multiply_1307_by_1307_l2514_251451


namespace second_square_size_l2514_251493

/-- Represents a square on the board -/
structure Square :=
  (size : Nat)
  (position : Nat × Nat)

/-- Represents the board configuration -/
def BoardConfiguration := List Square

/-- Checks if a given configuration covers the entire 10x10 board -/
def covers_board (config : BoardConfiguration) : Prop := sorry

/-- Checks if all squares in the configuration have different sizes -/
def all_different_sizes (config : BoardConfiguration) : Prop := sorry

/-- Checks if the last two squares in the configuration are 5x5 and 4x4 -/
def last_two_squares_correct (config : BoardConfiguration) : Prop := sorry

/-- Checks if the second square in the configuration is 8x8 -/
def second_square_is_8x8 (config : BoardConfiguration) : Prop := sorry

theorem second_square_size (config : BoardConfiguration) :
  config.length = 6 →
  covers_board config →
  all_different_sizes config →
  last_two_squares_correct config →
  second_square_is_8x8 config :=
sorry

end second_square_size_l2514_251493


namespace min_rubles_to_win_l2514_251476

/-- Represents the state of the game machine -/
structure GameState :=
  (score : ℕ)
  (rubles_spent : ℕ)

/-- Defines the possible moves in the game -/
inductive Move
| insert_one : Move
| insert_two : Move

/-- Applies a move to the current game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.insert_one => ⟨state.score + 1, state.rubles_spent + 1⟩
  | Move.insert_two => ⟨state.score * 2, state.rubles_spent + 2⟩

/-- Checks if the game state is valid (score ≤ 50) -/
def is_valid_state (state : GameState) : Prop :=
  state.score ≤ 50

/-- Checks if the game is won (score = 50) -/
def is_winning_state (state : GameState) : Prop :=
  state.score = 50

/-- The main theorem to prove -/
theorem min_rubles_to_win :
  ∃ (moves : List Move),
    let final_state := moves.foldl apply_move ⟨0, 0⟩
    is_valid_state final_state ∧
    is_winning_state final_state ∧
    final_state.rubles_spent = 11 ∧
    (∀ (other_moves : List Move),
      let other_final_state := other_moves.foldl apply_move ⟨0, 0⟩
      is_valid_state other_final_state →
      is_winning_state other_final_state →
      other_final_state.rubles_spent ≥ 11) :=
sorry

end min_rubles_to_win_l2514_251476


namespace sum_of_fourth_powers_l2514_251492

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares : a^2 + b^2 + c^2 = 0.1) : 
  a^4 + b^4 + c^4 = 0.005 := by
  sorry

end sum_of_fourth_powers_l2514_251492


namespace zero_point_existence_l2514_251450

theorem zero_point_existence (a : ℝ) :
  a < -2 → 
  (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) ∧
  (¬ ∀ a : ℝ, (∃ x ∈ Set.Icc (-1) 2, a * x + 3 = 0) → a < -2) :=
sorry

end zero_point_existence_l2514_251450


namespace intersection_of_perpendicular_tangents_on_parabola_l2514_251483

/-- Given a parabola y = 4x and two points on it with perpendicular tangents,
    prove that the x-coordinate of the intersection of these tangents is -1. -/
theorem intersection_of_perpendicular_tangents_on_parabola
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : y₁ = 4 * x₁)
  (h₂ : y₂ = 4 * x₂)
  (h_perp : (4 / y₁) * (4 / y₂) = -1) :
  ∃ (x y : ℝ), x = -1 ∧ 
    y = (4 / y₁) * x + y₁ / 2 ∧
    y = (4 / y₂) * x + y₂ / 2 :=
sorry

end intersection_of_perpendicular_tangents_on_parabola_l2514_251483


namespace product_consecutive_integers_square_l2514_251412

theorem product_consecutive_integers_square (x : ℤ) :
  ∃ (y : ℤ), x * (x + 1) * (x + 2) = y^2 ↔ x = 0 ∨ x = -1 ∨ x = -2 := by
  sorry

end product_consecutive_integers_square_l2514_251412


namespace unique_congruence_in_range_l2514_251473

theorem unique_congruence_in_range : ∃! n : ℤ, 10 ≤ n ∧ n ≤ 15 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end unique_congruence_in_range_l2514_251473


namespace discussions_probability_l2514_251408

def word := "DISCUSSIONS"

theorem discussions_probability : 
  let total_arrangements := Nat.factorial 11 / (Nat.factorial 4 * Nat.factorial 2)
  let favorable_arrangements := Nat.factorial 8 / Nat.factorial 2
  (favorable_arrangements : ℚ) / total_arrangements = 4 / 165 := by
  sorry

end discussions_probability_l2514_251408


namespace tennis_tournament_l2514_251407

theorem tennis_tournament (n : ℕ) : 
  (∃ (total_matches : ℕ) (women_wins men_wins : ℕ),
    -- Total number of players
    (n + (2*n + 1) = 3*n + 1) ∧
    -- Total matches calculation
    (total_matches = (3*n + 1) * (3*n) / 2 + 2*n) ∧
    -- Ratio of wins
    (3 * men_wins = 2 * women_wins) ∧
    -- Total wins equal total matches
    (women_wins + men_wins = total_matches) ∧
    -- n is a positive integer
    (n > 0)) →
  n = 2 :=
by sorry

end tennis_tournament_l2514_251407
