import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2925_292504

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0
  sum_property : ∀ n, S n = n * (a 0 + a (n-1)) / 2

/-- Theorem: For an arithmetic sequence with S_3 = 9 and S_6 = 36, S_9 = 81 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h3 : seq.S 3 = 9) (h6 : seq.S 6 = 36) : seq.S 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2925_292504


namespace NUMINAMATH_CALUDE_carries_strawberry_harvest_l2925_292513

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the expected strawberry harvest based on garden dimensions and planting information -/
def expectedStrawberryHarvest (dimensions : GardenDimensions) (plantsPerSquareFoot : ℝ) (strawberriesPerPlant : ℝ) : ℝ :=
  dimensions.length * dimensions.width * plantsPerSquareFoot * strawberriesPerPlant

/-- Theorem stating that Carrie's garden will yield 1920 strawberries -/
theorem carries_strawberry_harvest :
  let dimensions : GardenDimensions := { length := 6, width := 8 }
  let plantsPerSquareFoot : ℝ := 4
  let strawberriesPerPlant : ℝ := 10
  expectedStrawberryHarvest dimensions plantsPerSquareFoot strawberriesPerPlant = 1920 := by
  sorry

end NUMINAMATH_CALUDE_carries_strawberry_harvest_l2925_292513


namespace NUMINAMATH_CALUDE_no_adjacent_birch_probability_l2925_292501

def num_maple : ℕ := 5
def num_oak : ℕ := 4
def num_birch : ℕ := 6
def total_trees : ℕ := num_maple + num_oak + num_birch

def probability_no_adjacent_birch : ℚ :=
  (Nat.choose (num_maple + num_oak + 1) num_birch) / (Nat.choose total_trees num_birch)

theorem no_adjacent_birch_probability :
  probability_no_adjacent_birch = 2 / 45 :=
sorry

end NUMINAMATH_CALUDE_no_adjacent_birch_probability_l2925_292501


namespace NUMINAMATH_CALUDE_drum_fill_time_l2925_292569

/-- The time to fill a cylindrical drum with varying rain rate -/
theorem drum_fill_time (initial_rate : ℝ) (area : ℝ) (depth : ℝ) :
  let rate := fun t : ℝ => initial_rate * t^2
  let volume := area * depth
  let fill_time := (volume * 3 / (5 * initial_rate))^(1/3)
  fill_time^3 = volume * 3 / (5 * initial_rate) :=
by sorry

end NUMINAMATH_CALUDE_drum_fill_time_l2925_292569


namespace NUMINAMATH_CALUDE_min_trees_chopped_l2925_292531

def trees_per_sharpening : ℕ := 13
def cost_per_sharpening : ℕ := 5
def total_sharpening_cost : ℕ := 35

theorem min_trees_chopped :
  ∃ (n : ℕ), n ≥ 91 ∧ n ≥ (total_sharpening_cost / cost_per_sharpening) * trees_per_sharpening :=
by sorry

end NUMINAMATH_CALUDE_min_trees_chopped_l2925_292531


namespace NUMINAMATH_CALUDE_sum_remainder_mod_9_l2925_292548

theorem sum_remainder_mod_9 : (7150 + 7152 + 7154 + 7156 + 7158) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_9_l2925_292548


namespace NUMINAMATH_CALUDE_train_length_proof_l2925_292551

-- Define the given conditions
def faster_train_speed : ℝ := 42
def slower_train_speed : ℝ := 36
def passing_time : ℝ := 36

-- Define the theorem
theorem train_length_proof :
  let relative_speed := faster_train_speed - slower_train_speed
  let speed_in_mps := relative_speed * (5 / 18)
  let distance := speed_in_mps * passing_time
  let train_length := distance / 2
  train_length = 30 := by sorry

end NUMINAMATH_CALUDE_train_length_proof_l2925_292551


namespace NUMINAMATH_CALUDE_sequence_sum_formula_l2925_292527

/-- Given a sequence a with a₁ = 1 and Sₙ = n² * aₙ for all positive integers n,
    prove that the sum of the first n terms Sₙ is equal to 2n / (n+1). -/
theorem sequence_sum_formula (a : ℕ → ℚ) (S : ℕ → ℚ) :
  a 1 = 1 →
  (∀ n : ℕ, n > 0 → S n = n^2 * a n) →
  ∀ n : ℕ, n > 0 → S n = 2 * n / (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_formula_l2925_292527


namespace NUMINAMATH_CALUDE_chocolate_chip_calculation_l2925_292590

/-- Represents the number of cups of chocolate chips per batch in the recipe -/
def cups_per_batch : ℝ := 2.0

/-- Represents the number of batches that can be made with the available chocolate chips -/
def number_of_batches : ℝ := 11.5

/-- Calculates the total number of cups of chocolate chips -/
def total_chocolate_chips : ℝ := cups_per_batch * number_of_batches

/-- Proves that the total number of cups of chocolate chips is 23 -/
theorem chocolate_chip_calculation : total_chocolate_chips = 23 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_calculation_l2925_292590


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2925_292593

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  (∀ n, b (n + 1) > b n) →  -- increasing sequence
  (∃ d : ℤ, ∀ n, b (n + 1) - b n = d) →  -- arithmetic sequence
  b 4 * b 5 = 18 →
  b 3 * b 6 = -80 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2925_292593


namespace NUMINAMATH_CALUDE_benny_seashells_l2925_292558

theorem benny_seashells (initial_seashells given_away remaining : ℕ) : 
  initial_seashells = 66 → given_away = 52 → remaining = 14 →
  initial_seashells - given_away = remaining := by sorry

end NUMINAMATH_CALUDE_benny_seashells_l2925_292558


namespace NUMINAMATH_CALUDE_problem_solution_l2925_292502

theorem problem_solution (x y : ℝ) 
  (h1 : (1/2) * (x - 2)^3 + 32 = 0)
  (h2 : 3*x - 2*y = 6^2) :
  Real.sqrt (x^2 - y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2925_292502


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2925_292586

/-- A line passing through the origin -/
structure OriginLine where
  slope : ℝ

/-- The intersection point of two lines -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the problem setup -/
def TriangleSetup (l : OriginLine) : Prop :=
  ∃ (p q : Point),
    -- The line intersects x = 1
    p.x = 1 ∧ p.y = -l.slope
    -- The line intersects y = 1 + (√2/2)x
    ∧ q.x = 1 ∧ q.y = 1 + (Real.sqrt 2 / 2)
    -- The three lines form an equilateral triangle
    ∧ (p.x - 0)^2 + (p.y - 0)^2 = (q.x - p.x)^2 + (q.y - p.y)^2
    ∧ (q.x - 0)^2 + (q.y - 0)^2 = (q.x - p.x)^2 + (q.y - p.y)^2

/-- The main theorem -/
theorem triangle_perimeter (l : OriginLine) :
  TriangleSetup l → (3 : ℝ) + 3 * Real.sqrt 2 = 
    let p := Point.mk 1 (-l.slope)
    let q := Point.mk 1 (1 + Real.sqrt 2 / 2)
    3 * Real.sqrt ((q.x - p.x)^2 + (q.y - p.y)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_perimeter_l2925_292586


namespace NUMINAMATH_CALUDE_trigonometric_equality_l2925_292563

theorem trigonometric_equality : 
  1 / Real.cos (40 * π / 180) - 2 * Real.sqrt 3 / Real.sin (40 * π / 180) = -4 * Real.tan (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l2925_292563


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2925_292534

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem multiplication_puzzle (a b : ℕ) (ha : is_digit a) (hb : is_digit b) 
  (h_mult : (30 + a) * (10 * b + 4) = 156) : a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2925_292534


namespace NUMINAMATH_CALUDE_existence_of_small_difference_l2925_292524

theorem existence_of_small_difference (a : Fin 101 → ℝ)
  (h_increasing : ∀ i j, i < j → a i < a j)
  (h_bound : a 100 - a 0 ≤ 1000) :
  ∃ i j, i < j ∧ 0 < a j - a i ∧ a j - a i ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_small_difference_l2925_292524


namespace NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l2925_292571

theorem remainder_2468135792_mod_101 : 2468135792 % 101 = 52 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2468135792_mod_101_l2925_292571


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_l2925_292598

theorem right_triangle_leg_sum (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive lengths
  b = a + 2 →              -- one leg is 2 units longer
  a^2 + b^2 = c^2 →        -- Pythagorean theorem
  c = 29 →                 -- hypotenuse is 29 units
  a + b = 40 :=            -- sum of legs is 40
by sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_l2925_292598


namespace NUMINAMATH_CALUDE_difference_of_ones_and_zeros_237_l2925_292578

def base_2_representation (n : Nat) : List Nat :=
  sorry

def count_zeros (l : List Nat) : Nat :=
  sorry

def count_ones (l : List Nat) : Nat :=
  sorry

theorem difference_of_ones_and_zeros_237 :
  let binary_237 := base_2_representation 237
  let x := count_zeros binary_237
  let y := count_ones binary_237
  y - x = 6 := by sorry

end NUMINAMATH_CALUDE_difference_of_ones_and_zeros_237_l2925_292578


namespace NUMINAMATH_CALUDE_correct_log_values_l2925_292514

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define variables a, b, c
variable (a b c : ℝ)

-- Define the given correct logarithmic values
axiom log_3 : log 3 = 2*a - b
axiom log_5 : log 5 = a + c
axiom log_9 : log 9 = 4*a - 2*b
axiom log_0_27 : log 0.27 = 6*a - 3*b - 2
axiom log_8 : log 8 = 3 - 3*a - 3*c
axiom log_6 : log 6 = 1 + a - b - c

-- State the theorem to be proved
theorem correct_log_values :
  log 1.5 = 3*a - b + c - 1 ∧ log 7 = 2*b + c :=
sorry

end NUMINAMATH_CALUDE_correct_log_values_l2925_292514


namespace NUMINAMATH_CALUDE_root_product_equation_l2925_292543

theorem root_product_equation (x₁ x₂ x₃ : ℝ) : 
  (Real.sqrt 2023 * x₁^3 - 4047 * x₁^2 + 3 = 0) →
  (Real.sqrt 2023 * x₂^3 - 4047 * x₂^2 + 3 = 0) →
  (Real.sqrt 2023 * x₃^3 - 4047 * x₃^2 + 3 = 0) →
  x₁ < x₂ → x₂ < x₃ →
  x₂ * (x₁ + x₃) = 4046 := by
sorry

end NUMINAMATH_CALUDE_root_product_equation_l2925_292543


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_five_equals_sqrt_fifteen_l2925_292592

theorem sqrt_product (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt (a * b) = Real.sqrt a * Real.sqrt b :=
by sorry

theorem sqrt_three_times_sqrt_five_equals_sqrt_fifteen :
  Real.sqrt 3 * Real.sqrt 5 = Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_three_times_sqrt_five_equals_sqrt_fifteen_l2925_292592


namespace NUMINAMATH_CALUDE_lime_juice_per_lime_l2925_292537

-- Define the variables and constants
def tablespoons_per_mocktail : ℚ := 1
def days : ℕ := 30
def limes_per_dollar : ℚ := 3
def dollars_spent : ℚ := 5

-- Define the theorem
theorem lime_juice_per_lime :
  let total_tablespoons := tablespoons_per_mocktail * days
  let total_limes := limes_per_dollar * dollars_spent
  let juice_per_lime := total_tablespoons / total_limes
  juice_per_lime = 2 := by
sorry


end NUMINAMATH_CALUDE_lime_juice_per_lime_l2925_292537


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l2925_292596

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : m ≥ 1000 ∧ m < 10000) 
  (h2 : m % 2 = 0) (h3 : m % 221 = 0) : 
  ∃ (d : ℕ), d ∣ m ∧ d > 221 ∧ d ≤ 442 ∧ ∀ (x : ℕ), x ∣ m → x > 221 → x ≥ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l2925_292596


namespace NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_mean_l2925_292507

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- The 4th term is the geometric mean of the 2nd and 5th terms -/
def geometric_mean_condition (a : ℕ → ℝ) : Prop :=
  a 4 ^ 2 = a 2 * a 5

/-- Main theorem: If a is an arithmetic sequence with common difference 2
    and the 4th term is the geometric mean of the 2nd and 5th terms,
    then the 2nd term is -8 -/
theorem arithmetic_sequence_with_geometric_mean
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : geometric_mean_condition a) :
  a 2 = -8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_with_geometric_mean_l2925_292507


namespace NUMINAMATH_CALUDE_monthly_calendar_sum_l2925_292568

theorem monthly_calendar_sum : ∃ (x : ℕ), 
  8 ≤ x ∧ x ≤ 24 ∧ 
  ∃ (k : ℕ), (x - 7) + x + (x + 7) = 3 * k ∧ 
  (x - 7) + x + (x + 7) = 33 := by
  sorry

end NUMINAMATH_CALUDE_monthly_calendar_sum_l2925_292568


namespace NUMINAMATH_CALUDE_special_number_divisibility_l2925_292505

/-- Represents a 4-digit number with the given properties -/
structure SpecialNumber where
  value : Nat
  is_four_digit : value ≥ 1000 ∧ value < 10000
  has_three_unique_digits : ∃ (a b c : Nat), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ((value / 1000 = a ∧ (value / 100) % 10 = a) ∨
     (value / 1000 = a ∧ (value / 10) % 10 = a) ∨
     (value / 1000 = a ∧ value % 10 = a) ∨
     ((value / 100) % 10 = a ∧ (value / 10) % 10 = a) ∨
     ((value / 100) % 10 = a ∧ value % 10 = a) ∨
     ((value / 10) % 10 = a ∧ value % 10 = a)) ∧
    value = a * 1000 + b * 100 + c * 10 + (if value / 1000 = a then b else a)

/-- Mrs. Smith's age is the last two digits of the special number -/
def mrs_smith_age (n : SpecialNumber) : Nat := n.value % 100

/-- The ages of Mrs. Smith's children -/
def children_ages : Finset Nat := Finset.range 12 \ {0}

theorem special_number_divisibility (n : SpecialNumber) :
  ∃ (x : Nat), x ∈ children_ages ∧ ¬(n.value % x = 0) ∧
  ∀ (y : Nat), y ∈ children_ages ∧ y ≠ x → n.value % y = 0 →
  x = 3 := by sorry

#check special_number_divisibility

end NUMINAMATH_CALUDE_special_number_divisibility_l2925_292505


namespace NUMINAMATH_CALUDE_jackson_money_l2925_292550

/-- The amount of money each person has -/
structure Money where
  williams : ℝ
  jackson : ℝ
  lucy : ℝ
  ethan : ℝ

/-- The conditions of the problem -/
def problem_conditions (m : Money) : Prop :=
  m.jackson = 7 * m.williams ∧
  m.lucy = 3 * m.williams ∧
  m.ethan = m.lucy + 20 ∧
  m.williams + m.jackson + m.lucy + m.ethan = 600

/-- The theorem stating Jackson's money amount -/
theorem jackson_money (m : Money) (h : problem_conditions m) : 
  m.jackson = 7 * (600 - 20) / 14 := by
  sorry

end NUMINAMATH_CALUDE_jackson_money_l2925_292550


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2925_292591

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  isArithmeticSequence a →
  a 1 + 2 * a 8 + a 15 = 96 →
  2 * a 9 - a 10 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2925_292591


namespace NUMINAMATH_CALUDE_compound_proposition_truth_l2925_292587

theorem compound_proposition_truth (p q : Prop) 
  (hp : p) (hq : ¬q) : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_truth_l2925_292587


namespace NUMINAMATH_CALUDE_expression_evaluation_l2925_292517

theorem expression_evaluation :
  let a : ℤ := 2
  let b : ℤ := -1
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = -14 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2925_292517


namespace NUMINAMATH_CALUDE_overlapping_area_is_64_l2925_292546

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side_length : ℝ)

/-- Represents the rotation of a sheet -/
inductive Rotation
  | NoRotation
  | Rotate45
  | Rotate90

/-- Represents the configuration of three sheets -/
structure SheetConfiguration :=
  (bottom : Sheet)
  (middle : Sheet)
  (top : Sheet)
  (middle_rotation : Rotation)
  (top_rotation : Rotation)

/-- Calculates the area of the overlapping polygon -/
def overlapping_area (config : SheetConfiguration) : ℝ :=
  sorry

/-- Theorem stating that the overlapping area is 64 for the given configuration -/
theorem overlapping_area_is_64 :
  ∀ (config : SheetConfiguration),
    config.bottom.side_length = 8 ∧
    config.middle.side_length = 8 ∧
    config.top.side_length = 8 ∧
    config.middle_rotation = Rotation.Rotate45 ∧
    config.top_rotation = Rotation.Rotate90 →
    overlapping_area config = 64 :=
  sorry

end NUMINAMATH_CALUDE_overlapping_area_is_64_l2925_292546


namespace NUMINAMATH_CALUDE_church_attendance_l2925_292525

theorem church_attendance (total people : ℕ) (children : ℕ) (female_adults : ℕ) :
  total = 200 →
  children = 80 →
  female_adults = 60 →
  total = children + female_adults + (total - children - female_adults) →
  total - children - female_adults = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_church_attendance_l2925_292525


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l2925_292510

/-- Given that two dozen apples cost $15.60, prove that four dozen apples at the same rate will cost $31.20. -/
theorem apple_cost_calculation (cost_two_dozen : ℝ) (h : cost_two_dozen = 15.60) :
  let cost_per_dozen : ℝ := cost_two_dozen / 2
  let cost_four_dozen : ℝ := 4 * cost_per_dozen
  cost_four_dozen = 31.20 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l2925_292510


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l2925_292554

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 4 = 4 ∧
  a 3 + a 8 = 5

/-- Theorem stating that a_7 = 1 for the given arithmetic sequence -/
theorem arithmetic_sequence_a7 (a : ℕ → ℚ) (h : ArithmeticSequence a) : a 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l2925_292554


namespace NUMINAMATH_CALUDE_brochure_calculation_l2925_292518

/-- Calculates the number of brochures created by a printing press given specific conditions -/
theorem brochure_calculation (single_page_spreads : ℕ) 
  (h1 : single_page_spreads = 20)
  (h2 : ∀ n : ℕ, n = single_page_spreads → 2 * n = number_of_double_page_spreads)
  (h3 : ∀ n : ℕ, n = total_spread_pages → n / 4 = number_of_ad_blocks)
  (h4 : ∀ n : ℕ, n = number_of_ad_blocks → 4 * n = total_ads)
  (h5 : ∀ n : ℕ, n = total_ads → n / 4 = ad_pages)
  (h6 : ∀ n : ℕ, n = total_pages → n / 5 = number_of_brochures)
  : number_of_brochures = 25 := by
  sorry

#check brochure_calculation

end NUMINAMATH_CALUDE_brochure_calculation_l2925_292518


namespace NUMINAMATH_CALUDE_range_of_t_l2925_292544

theorem range_of_t (a b t : ℝ) (h1 : a^2 + a*b + b^2 = 1) (h2 : t = a*b - a^2 - b^2) :
  -3 ≤ t ∧ t ≤ -1/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_l2925_292544


namespace NUMINAMATH_CALUDE_point_on_number_line_l2925_292532

theorem point_on_number_line (A : ℝ) : 
  (|A| = 5) ↔ (A = 5 ∨ A = -5) := by sorry

end NUMINAMATH_CALUDE_point_on_number_line_l2925_292532


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2925_292549

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2925_292549


namespace NUMINAMATH_CALUDE_alice_bob_meet_l2925_292533

/-- The number of points on the circle -/
def n : ℕ := 18

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 11

/-- Bob's extra skip every second turn -/
def bob_extra : ℕ := 1

/-- The number of turns after which Alice and Bob meet -/
def meeting_turns : ℕ := 36

/-- Function to calculate position on the circle after a number of moves -/
def position (start : ℕ) (moves : ℕ) : ℕ :=
  (start + moves - 1) % n + 1

/-- Alice's position after a given number of turns -/
def alice_position (turns : ℕ) : ℕ :=
  position n (alice_move * turns)

/-- Bob's position after a given number of turns -/
def bob_position (turns : ℕ) : ℕ :=
  position n (n * turns - bob_move * turns - bob_extra * (turns / 2))

/-- Theorem stating that Alice and Bob meet after the specified number of turns -/
theorem alice_bob_meet : alice_position meeting_turns = bob_position meeting_turns := by
  sorry


end NUMINAMATH_CALUDE_alice_bob_meet_l2925_292533


namespace NUMINAMATH_CALUDE_visitor_difference_l2925_292599

/-- The number of paintings in Buckingham Palace -/
def paintings : ℕ := 39

/-- The number of visitors on the current day -/
def visitors_current : ℕ := 661

/-- The number of visitors on the previous day -/
def visitors_previous : ℕ := 600

/-- Theorem: The difference in visitors between the current day and the previous day is 61 -/
theorem visitor_difference : visitors_current - visitors_previous = 61 := by
  sorry

end NUMINAMATH_CALUDE_visitor_difference_l2925_292599


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l2925_292545

theorem max_value_on_ellipse :
  ∀ x y : ℝ, (x^2 / 6 + y^2 / 4 = 1) →
  ∃ (max : ℝ), (∀ x' y' : ℝ, (x'^2 / 6 + y'^2 / 4 = 1) → x' + 2*y' ≤ max) ∧
  max = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l2925_292545


namespace NUMINAMATH_CALUDE_coin_equality_l2925_292575

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Theorem stating that if 25 quarters and 15 dimes equal 15 quarters and n nickels, then n = 80 -/
theorem coin_equality (n : ℕ) : 
  25 * quarter_value + 15 * dime_value = 15 * quarter_value + n * nickel_value → n = 80 := by
  sorry


end NUMINAMATH_CALUDE_coin_equality_l2925_292575


namespace NUMINAMATH_CALUDE_sum_of_other_digits_l2925_292540

def is_form_76h4 (n : ℕ) : Prop :=
  ∃ h : ℕ, n = 7000 + 600 + 10 * h + 4

theorem sum_of_other_digits (n : ℕ) (h : ℕ) :
  is_form_76h4 n →
  h = 1 →
  n % 9 = 0 →
  (7 + 6 + 4 : ℕ) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_other_digits_l2925_292540


namespace NUMINAMATH_CALUDE_spherical_triangle_area_l2925_292553

/-- The area of a spherical triangle formed by the intersection of a sphere with a trihedral angle -/
theorem spherical_triangle_area 
  (R : ℝ) 
  (α β γ : ℝ) 
  (h_positive : R > 0)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_vertex_center : True)  -- Represents the condition that the vertex coincides with the sphere's center
  : ∃ (S_Δ : ℝ), S_Δ = R^2 * (α + β + γ - Real.pi) :=
sorry

end NUMINAMATH_CALUDE_spherical_triangle_area_l2925_292553


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l2925_292581

theorem right_triangle_leg_square (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = a + 2 →        -- Given condition
  b^2 = 4*(a + 1) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l2925_292581


namespace NUMINAMATH_CALUDE_car_trip_distance_l2925_292523

theorem car_trip_distance (D : ℝ) : 
  (D / 2 : ℝ) + (D / 2 / 4 : ℝ) + 105 = D → D = 280 :=
by sorry

end NUMINAMATH_CALUDE_car_trip_distance_l2925_292523


namespace NUMINAMATH_CALUDE_no_seven_flip_l2925_292519

/-- A function that returns the reverse of the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Definition of a k-flip number -/
def isKFlip (k : ℕ) (n : ℕ) : Prop :=
  k * n = reverseDigits n

/-- Theorem: There is no 7-flip integer -/
theorem no_seven_flip : ¬∃ (n : ℕ), n > 0 ∧ isKFlip 7 n := by sorry

end NUMINAMATH_CALUDE_no_seven_flip_l2925_292519


namespace NUMINAMATH_CALUDE_tenth_root_unity_sum_l2925_292579

theorem tenth_root_unity_sum (z : ℂ) : 
  z = Complex.exp (3 * Real.pi * Complex.I / 5) →
  z / (1 + z^2) + z^3 / (1 + z^6) + z^5 / (1 + z^10) = (z + z^3 - 1/2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_tenth_root_unity_sum_l2925_292579


namespace NUMINAMATH_CALUDE_angle_of_inclination_l2925_292564

/-- The angle of inclination of the line x + √3 y - 5 = 0 is 150° -/
theorem angle_of_inclination (x y : ℝ) : 
  x + Real.sqrt 3 * y - 5 = 0 → 
  ∃ θ : ℝ, θ = 150 * π / 180 ∧ 
    Real.tan θ = -(1 / Real.sqrt 3) ∧
    0 ≤ θ ∧ θ < π := by
  sorry

end NUMINAMATH_CALUDE_angle_of_inclination_l2925_292564


namespace NUMINAMATH_CALUDE_sin_15_cos_15_double_l2925_292536

theorem sin_15_cos_15_double : 2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_double_l2925_292536


namespace NUMINAMATH_CALUDE_greatest_x_value_l2925_292580

theorem greatest_x_value (x : ℝ) : 
  (x^2 - x - 30) / (x - 6) = 2 / (x + 4) → x ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2925_292580


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2925_292516

-- Define a structure for a rectangular solid
structure RectangularSolid where
  a : ℕ
  b : ℕ
  c : ℕ

-- Define primality
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define volume
def volume (solid : RectangularSolid) : ℕ := solid.a * solid.b * solid.c

-- Define surface area
def surfaceArea (solid : RectangularSolid) : ℕ :=
  2 * (solid.a * solid.b + solid.b * solid.c + solid.c * solid.a)

-- The main theorem
theorem rectangular_solid_surface_area :
  ∀ solid : RectangularSolid,
    isPrime solid.a ∧ isPrime solid.b ∧ isPrime solid.c →
    volume solid = 221 →
    surfaceArea solid = 502 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2925_292516


namespace NUMINAMATH_CALUDE_sandwich_cost_l2925_292582

/-- Proves that the cost of each sandwich is $5 --/
theorem sandwich_cost (num_sandwiches : ℕ) (paid : ℕ) (change : ℕ) :
  num_sandwiches = 3 ∧ paid = 20 ∧ change = 5 →
  (paid - change) / num_sandwiches = 5 :=
by
  sorry

#check sandwich_cost

end NUMINAMATH_CALUDE_sandwich_cost_l2925_292582


namespace NUMINAMATH_CALUDE_derivative_of_f_l2925_292560

noncomputable def f (x : ℝ) : ℝ := (Real.tan (Real.log 2) * Real.sin (19 * x)^2) / (19 * Real.cos (38 * x))

theorem derivative_of_f (x : ℝ) :
  deriv f x = (Real.tan (Real.log 2)^2 * Real.tan (38 * x)) / Real.cos (38 * x) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l2925_292560


namespace NUMINAMATH_CALUDE_race_distance_proof_l2925_292572

/-- The distance of the race where B beats C -/
def race_distance : ℝ := 800

theorem race_distance_proof :
  ∀ (v_a v_b v_c : ℝ),  -- speeds of A, B, and C
  v_a > 0 ∧ v_b > 0 ∧ v_c > 0 →  -- positive speeds
  (1000 / v_a = 900 / v_b) →  -- A beats B by 100m in 1000m race
  (race_distance / v_b = (race_distance - 100) / v_c) →  -- B beats C by 100m in race_distance
  (1000 / v_a = 787.5 / v_c) →  -- A beats C by 212.5m in 1000m race
  race_distance = 800 := by
sorry

end NUMINAMATH_CALUDE_race_distance_proof_l2925_292572


namespace NUMINAMATH_CALUDE_runner_problem_l2925_292556

/-- Proves that given the conditions of the runner's problem, the time taken for the second half is 10 hours -/
theorem runner_problem (v : ℝ) (h1 : v > 0) : 
  (40 / v = 20 / v + 5) → (40 / (v / 2) = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_runner_problem_l2925_292556


namespace NUMINAMATH_CALUDE_incorrect_reasonings_l2925_292521

-- Define the type for analogical reasoning
inductive AnalogicalReasoning
  | addition_subtraction
  | vector_complex_square
  | quadratic_equation
  | geometric_addition

-- Define a function to check if a reasoning is correct
def is_correct_reasoning (r : AnalogicalReasoning) : Prop :=
  match r with
  | AnalogicalReasoning.addition_subtraction => True
  | AnalogicalReasoning.vector_complex_square => False
  | AnalogicalReasoning.quadratic_equation => False
  | AnalogicalReasoning.geometric_addition => True

-- Theorem statement
theorem incorrect_reasonings :
  ∃ (incorrect : List AnalogicalReasoning),
    incorrect.length = 2 ∧
    (∀ r ∈ incorrect, ¬(is_correct_reasoning r)) ∧
    (∀ r, r ∉ incorrect → is_correct_reasoning r) :=
  sorry

end NUMINAMATH_CALUDE_incorrect_reasonings_l2925_292521


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2017_l2925_292528

/-- An arithmetic sequence satisfying the given condition -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a n + 2 * a (n + 1) + 3 * a (n + 2) = 6 * n + 22

/-- The 2017th term of the arithmetic sequence is 6058/3 -/
theorem arithmetic_sequence_2017 (a : ℕ → ℚ) (h : ArithmeticSequence a) :
  a 2017 = 6058 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2017_l2925_292528


namespace NUMINAMATH_CALUDE_hilt_pies_theorem_l2925_292508

/-- The total number of pies Mrs. Hilt needs to bake -/
def total_pies (pecan_pies apple_pies : ℝ) (factor : ℝ) : ℝ :=
  (pecan_pies + apple_pies) * factor

/-- Theorem: Given the initial number of pecan pies (16.0) and apple pies (14.0),
    and a multiplication factor (5.0), the total number of pies Mrs. Hilt
    needs to bake is 150.0. -/
theorem hilt_pies_theorem :
  total_pies 16.0 14.0 5.0 = 150.0 := by
  sorry

end NUMINAMATH_CALUDE_hilt_pies_theorem_l2925_292508


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2925_292503

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2925_292503


namespace NUMINAMATH_CALUDE_count_nondegenerate_triangles_l2925_292539

/-- A point in the integer grid -/
structure GridPoint where
  s : Nat
  t : Nat
  s_bound : s ≤ 4
  t_bound : t ≤ 4

/-- A triangle represented by three grid points -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p2.s - p1.s) * (p3.t - p1.t) = (p3.s - p1.s) * (p2.t - p1.t)

/-- Predicate to check if a triangle is nondegenerate -/
def nondegenerate (t : GridTriangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all valid grid points -/
def gridPoints : Finset GridPoint :=
  sorry

/-- The set of all possible triangles formed by grid points -/
def allTriangles : Finset GridTriangle :=
  sorry

/-- The set of all nondegenerate triangles -/
def nondegenerateTriangles : Finset GridTriangle :=
  sorry

theorem count_nondegenerate_triangles :
  Finset.card nondegenerateTriangles = 2170 :=
sorry

end NUMINAMATH_CALUDE_count_nondegenerate_triangles_l2925_292539


namespace NUMINAMATH_CALUDE_shifted_quadratic_sum_l2925_292570

/-- Given a quadratic function f(x) = 3x^2 + 2x + 5, shifting it 5 units to the left
    results in a new quadratic function g(x) = ax^2 + bx + c.
    This theorem proves that a + b + c = 125. -/
theorem shifted_quadratic_sum (a b c : ℝ) : 
  (∀ x, 3 * (x + 5)^2 + 2 * (x + 5) + 5 = a * x^2 + b * x + c) →
  a + b + c = 125 := by
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_sum_l2925_292570


namespace NUMINAMATH_CALUDE_age_difference_l2925_292597

theorem age_difference (A B C : ℕ) (h1 : C = A - 18) : A + B - (B + C) = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2925_292597


namespace NUMINAMATH_CALUDE_abc_inequality_l2925_292552

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2925_292552


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2925_292585

-- Define the set A
def A : Set ℝ := {-1, 0, 2}

-- Define the set B as a function of a
def B (a : ℝ) : Set ℝ := {2^a}

-- Theorem statement
theorem subset_implies_a_equals_one (a : ℝ) (h : B a ⊆ A) : a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l2925_292585


namespace NUMINAMATH_CALUDE_f_always_positive_l2925_292511

def f (x : ℝ) : ℝ := x^8 - x^5 + x^2 - x + 1

theorem f_always_positive : ∀ x : ℝ, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_always_positive_l2925_292511


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2925_292542

theorem absolute_value_equation_solutions (x : ℝ) :
  |5 * x - 4| = 29 ↔ x = -5 ∨ x = 33/5 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2925_292542


namespace NUMINAMATH_CALUDE_star_equality_implies_x_eq_five_l2925_292583

/-- Binary operation ★ on ordered pairs of integers -/
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

theorem star_equality_implies_x_eq_five :
  ∀ y : ℤ, star 4 5 1 1 = star x y 2 3 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_implies_x_eq_five_l2925_292583


namespace NUMINAMATH_CALUDE_probability_divisible_by_15_l2925_292566

/-- The set of digits used to form the six-digit number -/
def digits : Finset Nat := {1, 2, 3, 4, 5, 9}

/-- The number of digits -/
def n : Nat := 6

/-- The condition for divisibility by 15 -/
def divisible_by_15 (num : Nat) : Prop := num % 15 = 0

/-- The set of all possible six-digit numbers formed by the given digits -/
def all_numbers : Finset Nat := sorry

/-- The set of all six-digit numbers formed by the given digits that are divisible by 15 -/
def divisible_numbers : Finset Nat := sorry

/-- The probability of a randomly selected six-digit number being divisible by 15 -/
theorem probability_divisible_by_15 : 
  (Finset.card divisible_numbers : ℚ) / (Finset.card all_numbers : ℚ) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_divisible_by_15_l2925_292566


namespace NUMINAMATH_CALUDE_arc_length_calculation_l2925_292520

theorem arc_length_calculation (r α : Real) (h1 : r = π) (h2 : α = 2 * π / 3) :
  r * α = (2 / 3) * π^2 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_calculation_l2925_292520


namespace NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l2925_292506

theorem quadratic_real_roots_k_range (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) → k ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_k_range_l2925_292506


namespace NUMINAMATH_CALUDE_volume_ratio_is_twenty_l2925_292595

-- Define the dimensions of the shapes
def cube_edge : ℝ := 1  -- 1 meter
def cuboid_width : ℝ := 0.5  -- 50 cm in meters
def cuboid_length : ℝ := 0.5  -- 50 cm in meters
def cuboid_height : ℝ := 0.2  -- 20 cm in meters

-- Define the volume functions
def cube_volume (edge : ℝ) : ℝ := edge ^ 3
def cuboid_volume (width length height : ℝ) : ℝ := width * length * height

-- Theorem statement
theorem volume_ratio_is_twenty :
  (cube_volume cube_edge) / (cuboid_volume cuboid_width cuboid_length cuboid_height) = 20 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_is_twenty_l2925_292595


namespace NUMINAMATH_CALUDE_parallel_vectors_x_coord_l2925_292559

/-- Given three points A, B, and C in a plane, where vector AB is parallel to vector BC,
    prove that the x-coordinate of point C is 1. -/
theorem parallel_vectors_x_coord (A B C : ℝ × ℝ) : 
  A = (0, -3) → B = (3, 3) → C.2 = -1 → 
  (∃ k : ℝ, k ≠ 0 ∧ (B.1 - A.1, B.2 - A.2) = k • (C.1 - B.1, C.2 - B.2)) →
  C.1 = 1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_coord_l2925_292559


namespace NUMINAMATH_CALUDE_triomino_corner_reachability_l2925_292538

/-- Represents an L-triomino on a board -/
structure Triomino where
  center : Nat × Nat
  leg1 : Nat × Nat
  leg2 : Nat × Nat

/-- Represents a board of size m × n -/
structure Board (m n : Nat) where
  triomino : Triomino

/-- Defines a valid initial position of the triomino -/
def initial_position (m n : Nat) : Board m n :=
  { triomino := { center := (0, 0), leg1 := (0, 1), leg2 := (1, 0) } }

/-- Defines a valid rotation of the triomino -/
def can_rotate (b : Board m n) : Prop :=
  ∃ new_position : Triomino, true  -- We assume any rotation is possible

/-- Defines if a triomino can reach the bottom right corner -/
def can_reach_corner (m n : Nat) : Prop :=
  ∃ final_position : Board m n, 
    final_position.triomino.center = (m - 1, n - 1)

/-- The main theorem to be proved -/
theorem triomino_corner_reachability (m n : Nat) :
  can_reach_corner m n ↔ m % 2 = 1 ∧ n % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_triomino_corner_reachability_l2925_292538


namespace NUMINAMATH_CALUDE_harolds_leftover_money_l2925_292565

/-- Harold's financial situation --/
def harolds_finances (income rent car_payment groceries : ℚ) : Prop :=
  let utilities := car_payment / 2
  let total_expenses := rent + car_payment + utilities + groceries
  let remaining := income - total_expenses
  let retirement := remaining / 2
  let left_after_retirement := remaining - retirement
  income = 2500 ∧ 
  rent = 700 ∧ 
  car_payment = 300 ∧ 
  groceries = 50 ∧ 
  left_after_retirement = 650

theorem harolds_leftover_money :
  ∃ (income rent car_payment groceries : ℚ),
    harolds_finances income rent car_payment groceries :=
sorry

end NUMINAMATH_CALUDE_harolds_leftover_money_l2925_292565


namespace NUMINAMATH_CALUDE_certain_expression_proof_l2925_292500

theorem certain_expression_proof (a b X : ℝ) 
  (h1 : a / b = 4 / 3) 
  (h2 : (3 * a + 2 * b) / X = 3) : 
  X = 2 * b := by
sorry

end NUMINAMATH_CALUDE_certain_expression_proof_l2925_292500


namespace NUMINAMATH_CALUDE_honey_purchase_cost_l2925_292562

def honey_problem (bulk_price min_spend tax_rate excess_pounds : ℕ) : Prop :=
  let min_pounds : ℕ := min_spend / bulk_price
  let total_pounds : ℕ := min_pounds + excess_pounds
  let pre_tax_cost : ℕ := total_pounds * bulk_price
  let tax_amount : ℕ := total_pounds * tax_rate
  let total_cost : ℕ := pre_tax_cost + tax_amount
  total_cost = 240

theorem honey_purchase_cost :
  honey_problem 5 40 1 32 := by sorry

end NUMINAMATH_CALUDE_honey_purchase_cost_l2925_292562


namespace NUMINAMATH_CALUDE_little_john_height_l2925_292567

/-- Conversion factor from centimeters to meters -/
def cm_to_m : ℝ := 0.01

/-- Conversion factor from millimeters to meters -/
def mm_to_m : ℝ := 0.001

/-- Little John's height in meters, centimeters, and millimeters -/
def height_m : ℝ := 2
def height_cm : ℝ := 8
def height_mm : ℝ := 3

/-- Theorem stating that Little John's height in meters is 2.083 -/
theorem little_john_height : 
  height_m + height_cm * cm_to_m + height_mm * mm_to_m = 2.083 := by
  sorry

end NUMINAMATH_CALUDE_little_john_height_l2925_292567


namespace NUMINAMATH_CALUDE_exam_students_count_l2925_292509

theorem exam_students_count (total_average : ℝ) (excluded_average : ℝ) (new_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 40) 
  (h3 : new_average = 90) (h4 : excluded_count = 5) : 
  ∃ (n : ℕ), n = 25 ∧ 
    (n : ℝ) * total_average = 
      ((n - excluded_count) : ℝ) * new_average + (excluded_count : ℝ) * excluded_average :=
by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l2925_292509


namespace NUMINAMATH_CALUDE_determinant_scaling_l2925_292576

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = 7 →
  Matrix.det ![![3*x, 3*y], ![3*z, 3*w]] = 63 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l2925_292576


namespace NUMINAMATH_CALUDE_complex_magnitude_l2925_292577

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2925_292577


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2925_292512

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 7) % 12 = 0 ∧
  (n - 7) % 16 = 0 ∧
  (n - 7) % 18 = 0 ∧
  (n - 7) % 21 = 0 ∧
  (n - 7) % 28 = 0 ∧
  (n - 7) % 35 = 0 ∧
  (n - 7) % 39 = 0

theorem smallest_number_divisible_by_all :
  is_divisible_by_all 65527 ∧
  ∀ m : ℕ, m < 65527 → ¬is_divisible_by_all m :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2925_292512


namespace NUMINAMATH_CALUDE_total_pears_picked_l2925_292574

theorem total_pears_picked (jason_pears keith_pears mike_pears : ℕ)
  (h1 : jason_pears = 46)
  (h2 : keith_pears = 47)
  (h3 : mike_pears = 12) :
  jason_pears + keith_pears + mike_pears = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l2925_292574


namespace NUMINAMATH_CALUDE_subtracted_value_l2925_292584

theorem subtracted_value (n v : ℝ) (h1 : n = -10) (h2 : 2 * n - v = -12) : v = -8 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l2925_292584


namespace NUMINAMATH_CALUDE_square_covering_theorem_l2925_292589

theorem square_covering_theorem (l : ℕ) (h1 : l > 0) : 
  (∃ n : ℕ, n > 0 ∧ 2 * n^2 = 8 * l^2 / 9 ∧ l^2 < 2 * (n + 1)^2) ↔ 
  l ∈ ({3, 6, 9, 12, 15, 18, 21, 24} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_square_covering_theorem_l2925_292589


namespace NUMINAMATH_CALUDE_apples_left_after_pie_l2925_292555

def apples_left (initial : ℝ) (contribution : ℝ) (pie_requirement : ℝ) : ℝ :=
  initial + contribution - pie_requirement

theorem apples_left_after_pie : apples_left 10 5 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_after_pie_l2925_292555


namespace NUMINAMATH_CALUDE_trajectory_and_fixed_point_l2925_292541

-- Define the plane
variable (P : ℝ × ℝ)

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the line l
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the foot of the perpendicular Q
def Q (P : ℝ × ℝ) : ℝ × ℝ := (-1, P.2)

-- Define the dot product of 2D vectors
def dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem trajectory_and_fixed_point (P : ℝ × ℝ) : 
  (dot (P.1 + 1, P.2) (2, -P.2) = dot (P.1 - 1, P.2) (-2, P.2)) →
  (∃ (C : Set (ℝ × ℝ)) (E : ℝ × ℝ), 
    C = {p : ℝ × ℝ | p.2^2 = 4 * p.1} ∧
    E = (1, 0) ∧
    (∀ (k m : ℝ), 
      let M := (m^2, 2*m)
      let N := (-1, -1/m + m)
      (M ∈ C ∧ N.1 = -1) →
      (∃ (r : ℝ), (M.1 - E.1)^2 + (M.2 - E.2)^2 = r^2 ∧
                  (N.1 - E.1)^2 + (N.2 - E.2)^2 = r^2))) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_fixed_point_l2925_292541


namespace NUMINAMATH_CALUDE_average_of_quadratic_solutions_l2925_292529

/-- Given a quadratic equation ax² - 4ax + b = 0 with two real solutions,
    prove that the average of these solutions is 2. -/
theorem average_of_quadratic_solutions (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 - 4 * a * x + b
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) → (x₁ + x₂) / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_of_quadratic_solutions_l2925_292529


namespace NUMINAMATH_CALUDE_expand_expression_l2925_292535

theorem expand_expression (x : ℝ) : 6 * (x - 3) * (x^2 + 4*x + 16) = 6*x^3 + 6*x^2 + 24*x - 288 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2925_292535


namespace NUMINAMATH_CALUDE_least_divisible_by_1_to_9_halved_l2925_292557

def is_divisible_by_range (n : ℕ) (a b : ℕ) : Prop :=
  ∀ k : ℕ, a ≤ k → k ≤ b → k ∣ n

theorem least_divisible_by_1_to_9_halved :
  ∃ l : ℕ, (∀ m : ℕ, is_divisible_by_range m 1 9 → l ≤ m) ∧
           is_divisible_by_range l 1 9 ∧
           l / 2 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_least_divisible_by_1_to_9_halved_l2925_292557


namespace NUMINAMATH_CALUDE_fraction_product_equals_one_l2925_292515

theorem fraction_product_equals_one : 
  (4 + 6 + 8) / (3 + 5 + 7) * (3 + 5 + 7) / (4 + 6 + 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_one_l2925_292515


namespace NUMINAMATH_CALUDE_total_visitors_proof_l2925_292530

/-- The total number of visitors over two days at a tourist attraction -/
def total_visitors (m n : ℕ) : ℕ :=
  2 * m + n + 1000

/-- Theorem: The total number of visitors over two days is 2m + n + 1000 -/
theorem total_visitors_proof (m n : ℕ) : 
  total_visitors m n = 2 * m + n + 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_visitors_proof_l2925_292530


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l2925_292573

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x + y = 1

-- Define the condition for point P
def P_on_C₂ (P : ℝ × ℝ) : Prop := C₂ P.1 P.2

-- Define the condition for point R
def R_on_C₁ (R : ℝ × ℝ) : Prop := C₁ R.1 R.2

-- Define the condition that R is on OP
def R_on_OP (O P R : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 0 ∧ R.1 = t * P.1 ∧ R.2 = t * P.2

-- Define the condition for point Q
def Q_condition (O P Q R : ℝ × ℝ) : Prop :=
  (Q.1^2 + Q.2^2) * (P.1^2 + P.2^2) = (R.1^2 + R.2^2)^2

-- The main theorem
theorem trajectory_of_Q (O P Q R : ℝ × ℝ) :
  O = (0, 0) →
  P_on_C₂ P →
  R_on_C₁ R →
  R_on_OP O P R →
  Q_condition O P Q R →
  (Q.1 - 1/2)^2 + (Q.2 - 1/2)^2 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l2925_292573


namespace NUMINAMATH_CALUDE_complete_square_sum_l2925_292547

/-- 
Given a quadratic equation x^2 - 6x + 5 = 0, when rewritten in the form (x + d)^2 = e 
where d and e are integers, prove that d + e = 1
-/
theorem complete_square_sum (d e : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + d)^2 = e) → d + e = 1 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l2925_292547


namespace NUMINAMATH_CALUDE_extracurricular_materials_choice_l2925_292526

/-- The number of ways to choose r items from n items -/
def choose (n : ℕ) (r : ℕ) : ℕ := Nat.choose n r

/-- The number of ways to arrange r items from n items -/
def arrange (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The total number of extracurricular reading materials -/
def totalMaterials : ℕ := 6

/-- The number of materials each student chooses -/
def materialsPerStudent : ℕ := 2

/-- The number of common materials between students -/
def commonMaterials : ℕ := 1

theorem extracurricular_materials_choice :
  (choose totalMaterials commonMaterials) *
  (arrange (totalMaterials - commonMaterials) (materialsPerStudent - commonMaterials)) = 120 := by
  sorry


end NUMINAMATH_CALUDE_extracurricular_materials_choice_l2925_292526


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2925_292522

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : 
  |x - y| = 22 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2925_292522


namespace NUMINAMATH_CALUDE_divide_fractions_and_mixed_number_l2925_292588

theorem divide_fractions_and_mixed_number :
  (5 : ℚ) / 6 / (1 + 3 / 9) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_divide_fractions_and_mixed_number_l2925_292588


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2925_292594

theorem smallest_integer_with_remainder_one (k : ℕ) : k = 400 ↔ 
  (k > 1) ∧ 
  (k % 19 = 1) ∧ 
  (k % 7 = 1) ∧ 
  (k % 3 = 1) ∧ 
  (∀ m : ℕ, m > 1 → m % 19 = 1 → m % 7 = 1 → m % 3 = 1 → k ≤ m) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2925_292594


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2925_292561

theorem sum_of_solutions (y : ℝ) : (∃ y₁ y₂ : ℝ, y₁ + 16 / y₁ = 12 ∧ y₂ + 16 / y₂ = 12 ∧ y₁ ≠ y₂ ∧ y₁ + y₂ = 12) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2925_292561
