import Mathlib

namespace NUMINAMATH_CALUDE_profit_maximization_l4155_415502

/-- Profit function given price x -/
def profit (x : ℝ) : ℝ := (x - 40) * (300 - (x - 60) * 10)

/-- The price that maximizes profit -/
def optimal_price : ℝ := 65

/-- The maximum profit achieved -/
def max_profit : ℝ := 6250

theorem profit_maximization :
  (∀ x : ℝ, profit x ≤ profit optimal_price) ∧
  profit optimal_price = max_profit := by
  sorry

#check profit_maximization

end NUMINAMATH_CALUDE_profit_maximization_l4155_415502


namespace NUMINAMATH_CALUDE_quadratic_always_positive_range_l4155_415542

theorem quadratic_always_positive_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_range_l4155_415542


namespace NUMINAMATH_CALUDE_intersection_implies_a_zero_l4155_415523

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {a^2, a+1, -1}
def B (a : ℝ) : Set ℝ := {2*a-1, |a-2|, 3*a^2+4}

-- Theorem statement
theorem intersection_implies_a_zero (a : ℝ) : A a ∩ B a = {-1} → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_zero_l4155_415523


namespace NUMINAMATH_CALUDE_part_one_part_two_l4155_415596

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + 4*a*x + 2*a + 6

-- Define the function g
def g (a : ℝ) : ℝ := 2 - a * |a + 3|

-- Part 1
theorem part_one (a : ℝ) : (∀ y ≥ 0, ∃ x, f a x = y) ∧ (∀ x, f a x ≥ 0) → a = 3/2 := by sorry

-- Part 2
theorem part_two (a : ℝ) : 
  (∀ x, f a x ≥ 0) → 
  (∀ y ∈ Set.Icc (-19/4) (-2), ∃ a ∈ Set.Icc (-1) (3/2), g a = y) ∧ 
  (∀ a ∈ Set.Icc (-1) (3/2), g a ∈ Set.Icc (-19/4) (-2)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4155_415596


namespace NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l4155_415522

theorem sufficient_condition_for_quadratic_inequality (a : ℝ) :
  (a ≥ 3) →
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 - x - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 - x - a ≤ 0) → a ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_quadratic_inequality_l4155_415522


namespace NUMINAMATH_CALUDE_smallest_square_area_l4155_415552

/-- The smallest area of a square containing non-overlapping 1x4 and 2x5 rectangles -/
theorem smallest_square_area (r1_width r1_height r2_width r2_height : ℕ) 
  (h1 : r1_width = 1 ∧ r1_height = 4)
  (h2 : r2_width = 2 ∧ r2_height = 5)
  (h_no_overlap : True)  -- Represents the non-overlapping condition
  (h_parallel : True)    -- Represents the parallel sides condition
  : ∃ (s : ℕ), s^2 = 81 ∧ ∀ (t : ℕ), (t ≥ r1_width ∧ t ≥ r1_height ∧ t ≥ r2_width ∧ t ≥ r2_height) → t^2 ≥ s^2 := by
  sorry

#check smallest_square_area

end NUMINAMATH_CALUDE_smallest_square_area_l4155_415552


namespace NUMINAMATH_CALUDE_choose_cooks_count_l4155_415520

def total_people : ℕ := 10
def cooks_needed : ℕ := 3

theorem choose_cooks_count : Nat.choose total_people cooks_needed = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_cooks_count_l4155_415520


namespace NUMINAMATH_CALUDE_cosine_increasing_interval_l4155_415517

theorem cosine_increasing_interval (a : Real) : 
  (∀ x₁ x₂ : Real, -π ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ a → Real.cos x₁ < Real.cos x₂) → 
  a ∈ Set.Ioc (-π) 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_increasing_interval_l4155_415517


namespace NUMINAMATH_CALUDE_equation_solution_l4155_415573

theorem equation_solution :
  ∃ x : ℝ, (2*x + 1)/3 - (5*x - 1)/6 = 1 ∧ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4155_415573


namespace NUMINAMATH_CALUDE_digit_105_of_7_19th_l4155_415516

/-- The decimal representation of 7/19 has a repeating cycle of length 18 -/
def decimal_cycle_length : ℕ := 18

/-- The repeating decimal representation of 7/19 -/
def decimal_rep : List ℕ := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]

/-- The 105th digit after the decimal point in the decimal representation of 7/19 is 7 -/
theorem digit_105_of_7_19th : decimal_rep[(105 - 1) % decimal_cycle_length] = 7 := by
  sorry

end NUMINAMATH_CALUDE_digit_105_of_7_19th_l4155_415516


namespace NUMINAMATH_CALUDE_special_numbers_theorem_l4155_415513

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  90 ≤ n ∧ n ≤ 150 ∧ digit_sum (digit_sum n) = 1

theorem special_numbers_theorem : 
  {n : ℕ | satisfies_condition n} = {91, 100, 109, 118, 127, 136, 145} := by
  sorry

end NUMINAMATH_CALUDE_special_numbers_theorem_l4155_415513


namespace NUMINAMATH_CALUDE_min_computers_to_purchase_l4155_415545

/-- Represents the problem of finding the minimum number of computers to purchase --/
theorem min_computers_to_purchase (total_devices : ℕ) (computer_cost whiteboard_cost max_cost : ℚ) :
  total_devices = 30 →
  computer_cost = 1/2 →
  whiteboard_cost = 3/2 →
  max_cost = 30 →
  ∃ (min_computers : ℕ),
    min_computers = 15 ∧
    ∀ (x : ℕ),
      x < 15 →
      (x : ℚ) * computer_cost + (total_devices - x : ℚ) * whiteboard_cost > max_cost :=
by sorry

end NUMINAMATH_CALUDE_min_computers_to_purchase_l4155_415545


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4155_415550

theorem inequality_equivalence (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 8) ≤ 7 / 3 ↔ -8 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4155_415550


namespace NUMINAMATH_CALUDE_correct_arrangements_l4155_415540

/-- Represents a student with a grade -/
structure Student where
  grade : Nat

/-- Represents a car with students -/
structure Car where
  students : Finset Student

/-- The total number of students -/
def totalStudents : Nat := 8

/-- The number of grades -/
def numGrades : Nat := 4

/-- The number of students per grade -/
def studentsPerGrade : Nat := 2

/-- The number of students per car -/
def studentsPerCar : Nat := 4

/-- Twin sisters from first grade -/
def twinSisters : Finset Student := sorry

/-- All students -/
def allStudents : Finset Student := sorry

/-- Checks if a car has exactly two students from the same grade -/
def hasTwoSameGrade (car : Car) : Prop := sorry

/-- The number of ways to arrange students in car A -/
def numArrangements : Nat := sorry

/-- Main theorem -/
theorem correct_arrangements :
  numArrangements = 24 := by sorry

end NUMINAMATH_CALUDE_correct_arrangements_l4155_415540


namespace NUMINAMATH_CALUDE_complex_equation_sum_l4155_415539

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 →
  (a + 2 * i) / i = b - i →
  a + b = 3 := by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l4155_415539


namespace NUMINAMATH_CALUDE_perfect_cube_implies_one_l4155_415580

theorem perfect_cube_implies_one (a : ℕ) 
  (h : ∀ n : ℕ, ∃ k : ℕ, 4 * (a^n + 1) = k^3) : 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_implies_one_l4155_415580


namespace NUMINAMATH_CALUDE_product_125_sum_31_l4155_415527

theorem product_125_sum_31 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 125 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 31 := by
sorry

end NUMINAMATH_CALUDE_product_125_sum_31_l4155_415527


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4155_415524

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 + x - 12 ≤ 0 ∧ x ≥ -4 → -4 ≤ x ∧ x ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4155_415524


namespace NUMINAMATH_CALUDE_smiths_bakery_pies_smiths_bakery_pies_proof_l4155_415577

theorem smiths_bakery_pies : ℕ → ℕ → Prop :=
  fun mcgees_pies smiths_pies =>
    mcgees_pies = 16 →
    smiths_pies = mcgees_pies^2 + mcgees_pies^2 / 2 →
    smiths_pies = 384

-- The proof would go here, but we're skipping it as requested
theorem smiths_bakery_pies_proof : smiths_bakery_pies 16 384 := by
  sorry

end NUMINAMATH_CALUDE_smiths_bakery_pies_smiths_bakery_pies_proof_l4155_415577


namespace NUMINAMATH_CALUDE_orange_basket_problem_l4155_415581

theorem orange_basket_problem (N : ℕ) : 
  N % 10 = 2 → N % 12 = 0 → N = 72 := by
  sorry

end NUMINAMATH_CALUDE_orange_basket_problem_l4155_415581


namespace NUMINAMATH_CALUDE_one_point_three_six_billion_scientific_notation_l4155_415505

/-- Proves that 1.36 billion is equal to 1.36 × 10^9 -/
theorem one_point_three_six_billion_scientific_notation :
  (1.36 : ℝ) * (10 ^ 9 : ℝ) = 1.36e9 := by sorry

end NUMINAMATH_CALUDE_one_point_three_six_billion_scientific_notation_l4155_415505


namespace NUMINAMATH_CALUDE_boltons_class_size_l4155_415535

theorem boltons_class_size :
  ∀ (S : ℚ),
  (2 / 5 : ℚ) * S + (1 / 3 : ℚ) * ((3 / 5 : ℚ) * S) + ((3 / 5 : ℚ) * S - (1 / 3 : ℚ) * ((3 / 5 : ℚ) * S)) = S →
  (2 / 5 : ℚ) * S + ((3 / 5 : ℚ) * S - (1 / 3 : ℚ) * ((3 / 5 : ℚ) * S)) = 20 →
  S = 25 := by
  sorry

end NUMINAMATH_CALUDE_boltons_class_size_l4155_415535


namespace NUMINAMATH_CALUDE_short_sleeve_shirts_count_proof_short_sleeve_shirts_l4155_415564

theorem short_sleeve_shirts_count : ℕ → ℕ → ℕ → Prop :=
  fun total_shirts long_sleeve_shirts short_sleeve_shirts =>
    total_shirts = long_sleeve_shirts + short_sleeve_shirts →
    total_shirts = 30 →
    long_sleeve_shirts = 21 →
    short_sleeve_shirts = 8

-- The proof is omitted
theorem proof_short_sleeve_shirts : short_sleeve_shirts_count 30 21 8 := by
  sorry

end NUMINAMATH_CALUDE_short_sleeve_shirts_count_proof_short_sleeve_shirts_l4155_415564


namespace NUMINAMATH_CALUDE_equal_sum_product_square_diff_l4155_415510

theorem equal_sum_product_square_diff : ∃ (x y : ℝ),
  (x + y = x * y) ∧ (x + y = x^2 - y^2) ∧
  ((x = (3 + Real.sqrt 5) / 2 ∧ y = (1 + Real.sqrt 5) / 2) ∨
   (x = (3 - Real.sqrt 5) / 2 ∧ y = (1 - Real.sqrt 5) / 2) ∨
   (x = 0 ∧ y = 0)) :=
by sorry


end NUMINAMATH_CALUDE_equal_sum_product_square_diff_l4155_415510


namespace NUMINAMATH_CALUDE_upward_shift_quadratic_l4155_415574

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := -x^2

/-- The amount of upward shift -/
def shift : ℝ := 2

/-- The shifted function -/
def g (x : ℝ) : ℝ := f x + shift

theorem upward_shift_quadratic :
  ∀ x : ℝ, g x = -(x^2) + 2 := by
  sorry

end NUMINAMATH_CALUDE_upward_shift_quadratic_l4155_415574


namespace NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l4155_415519

theorem range_of_a_for_false_proposition :
  {a : ℝ | ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + 2*a*x₀ + 2*a + 3 < 0} = Set.Ioi (-1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_false_proposition_l4155_415519


namespace NUMINAMATH_CALUDE_circle_lattice_point_uniqueness_l4155_415585

theorem circle_lattice_point_uniqueness (r : ℝ) (hr : r > 0) :
  ∃! (x y : ℤ), (↑x - Real.sqrt 2)^2 + (↑y - 1/3)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_lattice_point_uniqueness_l4155_415585


namespace NUMINAMATH_CALUDE_nancy_crayons_l4155_415555

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 15

/-- The number of packs Nancy bought -/
def packs_bought : ℕ := 41

/-- The total number of crayons Nancy bought -/
def total_crayons : ℕ := crayons_per_pack * packs_bought

theorem nancy_crayons : total_crayons = 615 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crayons_l4155_415555


namespace NUMINAMATH_CALUDE_ratio_part_to_whole_l4155_415525

theorem ratio_part_to_whole (N : ℝ) (x : ℝ) 
  (h1 : (1/4) * x * (2/5) * N = 14)
  (h2 : (2/5) * N = 168) : 
  x / N = 2/5 := by
sorry

end NUMINAMATH_CALUDE_ratio_part_to_whole_l4155_415525


namespace NUMINAMATH_CALUDE_smallest_two_digit_number_with_conditions_l4155_415544

theorem smallest_two_digit_number_with_conditions : ∃ n : ℕ,
  (n ≥ 10 ∧ n < 100) ∧  -- two-digit number
  (n % 3 = 0) ∧         -- divisible by 3
  (n % 4 = 0) ∧         -- divisible by 4
  (n % 5 = 4) ∧         -- remainder 4 when divided by 5
  (∀ m : ℕ, (m ≥ 10 ∧ m < 100) ∧ (m % 3 = 0) ∧ (m % 4 = 0) ∧ (m % 5 = 4) → n ≤ m) ∧
  n = 24 :=
by
  sorry

#check smallest_two_digit_number_with_conditions

end NUMINAMATH_CALUDE_smallest_two_digit_number_with_conditions_l4155_415544


namespace NUMINAMATH_CALUDE_puzzle_e_count_l4155_415534

/-- Represents the types of puzzle pieces -/
inductive PieceType
| A  -- Corner piece
| B  -- Edge piece
| C  -- Special edge piece
| D  -- Internal piece with 3 indentations
| E  -- Internal piece with 2 indentations

/-- Structure representing a rectangular puzzle -/
structure Puzzle where
  width : ℕ
  height : ℕ
  total_pieces : ℕ
  a_count : ℕ
  b_count : ℕ
  c_count : ℕ
  d_count : ℕ
  balance_equation : 2 * a_count + b_count + c_count + 3 * d_count = 2 * b_count + 2 * c_count + d_count

/-- Theorem stating the number of E-type pieces in the puzzle -/
theorem puzzle_e_count (p : Puzzle) 
  (h_dim : p.width = 23 ∧ p.height = 37)
  (h_total : p.total_pieces = 851)
  (h_a : p.a_count = 4)
  (h_b : p.b_count = 108)
  (h_c : p.c_count = 4)
  (h_d : p.d_count = 52) :
  p.total_pieces - (p.a_count + p.b_count + p.c_count + p.d_count) = 683 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_e_count_l4155_415534


namespace NUMINAMATH_CALUDE_horner_method_eval_l4155_415579

def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

theorem horner_method_eval :
  f (-4) = 220 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_eval_l4155_415579


namespace NUMINAMATH_CALUDE_antoinette_weight_l4155_415562

/-- Proves that Antoinette weighs 79 kilograms given the conditions of the problem -/
theorem antoinette_weight :
  ∀ (rupert antoinette charles : ℝ),
  antoinette = 2 * rupert - 7 →
  charles = (antoinette + rupert) / 2 + 5 →
  rupert + antoinette + charles = 145 →
  antoinette = 79 := by
sorry

end NUMINAMATH_CALUDE_antoinette_weight_l4155_415562


namespace NUMINAMATH_CALUDE_ship_distance_theorem_l4155_415586

/-- A function representing the square of the distance of a ship from an island over time. -/
def distance_squared (t : ℝ) : ℝ := 36 * t^2 - 84 * t + 49

/-- The theorem stating the distances at specific times given the initial conditions. -/
theorem ship_distance_theorem :
  (distance_squared 0 = 49) ∧
  (distance_squared 2 = 25) ∧
  (distance_squared 3 = 121) →
  (Real.sqrt (distance_squared 1) = 1) ∧
  (Real.sqrt (distance_squared 4) = 17) := by
  sorry

#check ship_distance_theorem

end NUMINAMATH_CALUDE_ship_distance_theorem_l4155_415586


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l4155_415531

theorem solution_set_of_inequality (x : ℝ) : 
  (Set.Ioo (-2 : ℝ) (-1/3 : ℝ)).Nonempty ∧ 
  (∀ y ∈ Set.Ioo (-2 : ℝ) (-1/3 : ℝ), (2*y - 1) / (3*y + 1) > 1) ∧
  (∀ z : ℝ, z ∉ Set.Ioo (-2 : ℝ) (-1/3 : ℝ) → (2*z - 1) / (3*z + 1) ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l4155_415531


namespace NUMINAMATH_CALUDE_least_number_with_remainder_four_l4155_415521

theorem least_number_with_remainder_four (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → 
    (m % 6 ≠ 4 ∨ m % 9 ≠ 4 ∨ m % 12 ≠ 4 ∨ m % 18 ≠ 4)) ∧
  n % 6 = 4 ∧ n % 9 = 4 ∧ n % 12 = 4 ∧ n % 18 = 4 → 
  n = 40 := by
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_four_l4155_415521


namespace NUMINAMATH_CALUDE_jason_shorts_expenditure_l4155_415532

theorem jason_shorts_expenditure (total : ℝ) (jacket : ℝ) (shorts : ℝ) : 
  total = 19.02 → jacket = 4.74 → total = jacket + shorts → shorts = 14.28 := by
  sorry

end NUMINAMATH_CALUDE_jason_shorts_expenditure_l4155_415532


namespace NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l4155_415537

/-- The number of positive integer divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer has exactly 12 divisors -/
def has_twelve_divisors (n : ℕ+) : Prop :=
  num_divisors n = 12

theorem smallest_with_twelve_divisors :
  ∃ (n : ℕ+), has_twelve_divisors n ∧ ∀ (m : ℕ+), has_twelve_divisors m → n ≤ m :=
by
  use 72
  sorry

end NUMINAMATH_CALUDE_smallest_with_twelve_divisors_l4155_415537


namespace NUMINAMATH_CALUDE_tangent_slope_three_points_point_on_curve_l4155_415591

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_slope_three_points (x : ℝ) :
  (f' x = 3) ↔ (x = 1 ∨ x = -1) :=
sorry

theorem point_on_curve (x : ℝ) :
  (f' x = 3 ∧ f x = x^3) ↔ ((x = 1 ∧ f x = 1) ∨ (x = -1 ∧ f x = -1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_three_points_point_on_curve_l4155_415591


namespace NUMINAMATH_CALUDE_kaydence_age_is_twelve_l4155_415508

/-- Represents the ages of family members and the total family age -/
structure Family where
  total_age : ℕ
  father_age : ℕ
  mother_age : ℕ
  brother_age : ℕ
  sister_age : ℕ

/-- Calculates Kaydence's age based on the family's ages -/
def kaydence_age (f : Family) : ℕ :=
  f.total_age - (f.father_age + f.mother_age + f.brother_age + f.sister_age)

/-- Theorem stating that Kaydence's age is 12 given the family conditions -/
theorem kaydence_age_is_twelve :
  ∀ (f : Family),
    f.total_age = 200 →
    f.father_age = 60 →
    f.mother_age = f.father_age - 2 →
    f.brother_age = f.father_age / 2 →
    f.sister_age = 40 →
    kaydence_age f = 12 := by
  sorry

end NUMINAMATH_CALUDE_kaydence_age_is_twelve_l4155_415508


namespace NUMINAMATH_CALUDE_total_distance_traveled_l4155_415592

/-- Converts kilometers per hour to miles per hour -/
def kph_to_mph (kph : ℝ) : ℝ := kph * 0.621371

/-- Calculates distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem total_distance_traveled :
  let walk_time : ℝ := 90 / 60  -- 90 minutes in hours
  let walk_speed : ℝ := 3       -- 3 mph
  let rest_time : ℝ := 15 / 60  -- 15 minutes in hours
  let cycle_time : ℝ := 45 / 60 -- 45 minutes in hours
  let cycle_speed : ℝ := kph_to_mph 20 -- 20 kph converted to mph
  let total_time : ℝ := 2.5     -- 2 hours and 30 minutes
  let walk_distance := distance walk_speed walk_time
  let cycle_distance := distance cycle_speed cycle_time
  let total_distance := walk_distance + cycle_distance
  ∃ ε > 0, |total_distance - 13.82| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l4155_415592


namespace NUMINAMATH_CALUDE_tangent_property_of_sine_equation_l4155_415557

theorem tangent_property_of_sine_equation (k : ℝ) (α β : ℝ) :
  (∃ (k : ℝ), k > 0 ∧
    (∀ x : ℝ, x ∈ Set.Ioo 0 Real.pi → (|Real.sin x| / x = k ↔ x = α ∨ x = β)) ∧
    α ∈ Set.Ioo 0 Real.pi ∧
    β ∈ Set.Ioo 0 Real.pi ∧
    α < β) →
  Real.tan (β + Real.pi / 4) = (1 + β) / (1 - β) :=
by sorry

end NUMINAMATH_CALUDE_tangent_property_of_sine_equation_l4155_415557


namespace NUMINAMATH_CALUDE_emma_numbers_l4155_415528

theorem emma_numbers (x y : ℤ) : 
  4 * x + 3 * y = 140 → (x = 20 ∨ y = 20) → x = 20 ∧ y = 20 := by
sorry

end NUMINAMATH_CALUDE_emma_numbers_l4155_415528


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l4155_415556

theorem smallest_number_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧
  (n % 13 = 12) ∧
  (n % 11 = 10) ∧
  (n % 7 = 6) ∧
  (n % 5 = 4) ∧
  (n % 3 = 2) ∧
  (∀ m : ℕ, m > 0 → 
    (m % 13 = 12) ∧
    (m % 11 = 10) ∧
    (m % 7 = 6) ∧
    (m % 5 = 4) ∧
    (m % 3 = 2) → 
    n ≤ m) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l4155_415556


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_less_than_neg_one_l4155_415566

-- Define the function f(x) = ax + 1
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 1

-- State the theorem
theorem unique_solution_implies_a_less_than_neg_one :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_less_than_neg_one_l4155_415566


namespace NUMINAMATH_CALUDE_existence_of_abc_l4155_415551

def S (x : ℕ) : ℕ := (x.digits 10).sum

theorem existence_of_abc : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  S (a + b) < 5 ∧ 
  S (b + c) < 5 ∧ 
  S (c + a) < 5 ∧ 
  S (a + b + c) > 50 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_abc_l4155_415551


namespace NUMINAMATH_CALUDE_cinema_selection_is_systematic_sampling_l4155_415529

/-- Represents a cinema with a specific number of rows and seats per row. -/
structure Cinema where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a sampling method. -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | WithReplacement

/-- Represents the selection of seats in a cinema. -/
structure SeatSelection where
  cinema : Cinema
  seatNumber : Nat

/-- Determines if a sampling method is systematic based on the seat selection. -/
def isSystematicSampling (selection : SeatSelection) : Prop :=
  selection.cinema.rows > 0 ∧
  selection.cinema.seatsPerRow > 0 ∧
  selection.seatNumber < selection.cinema.seatsPerRow

/-- Theorem stating that the given seat selection is an example of systematic sampling. -/
theorem cinema_selection_is_systematic_sampling 
  (cinema : Cinema)
  (selection : SeatSelection)
  (h1 : cinema.rows = 50)
  (h2 : cinema.seatsPerRow = 60)
  (h3 : selection.seatNumber = 18)
  (h4 : selection.cinema = cinema) :
  isSystematicSampling selection ∧ 
  SamplingMethod.Systematic = SamplingMethod.Systematic :=
sorry


end NUMINAMATH_CALUDE_cinema_selection_is_systematic_sampling_l4155_415529


namespace NUMINAMATH_CALUDE_camp_attendance_outside_county_attendance_l4155_415587

theorem camp_attendance (lawrence_camp : ℕ) (lawrence_home : ℕ) (lawrence_total : ℕ)
  (h1 : lawrence_camp = 610769)
  (h2 : lawrence_home = 590796)
  (h3 : lawrence_total = 1201565)
  (h4 : lawrence_total = lawrence_camp + lawrence_home) :
  lawrence_camp = lawrence_total - lawrence_home :=
by sorry

theorem outside_county_attendance (lawrence_camp : ℕ) (lawrence_home : ℕ) (lawrence_total : ℕ)
  (h1 : lawrence_camp = 610769)
  (h2 : lawrence_home = 590796)
  (h3 : lawrence_total = 1201565)
  (h4 : lawrence_total = lawrence_camp + lawrence_home) :
  0 = lawrence_camp - (lawrence_total - lawrence_home) :=
by sorry

end NUMINAMATH_CALUDE_camp_attendance_outside_county_attendance_l4155_415587


namespace NUMINAMATH_CALUDE_train_length_calculation_l4155_415507

/-- Calculates the length of a train given its speed, bridge length, and time to cross the bridge. -/
theorem train_length_calculation (train_speed : Real) (bridge_length : Real) (crossing_time : Real) :
  let train_speed_ms : Real := train_speed * (1000 / 3600)
  let total_distance : Real := train_speed_ms * crossing_time
  let train_length : Real := total_distance - bridge_length
  train_speed = 45 ∧ bridge_length = 219.03 ∧ crossing_time = 30 →
  train_length = 155.97 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l4155_415507


namespace NUMINAMATH_CALUDE_point_in_region_t_range_l4155_415569

/-- Given a point (1, t) in the region represented by x - y + 1 > 0, 
    the range of values for t is t < 2 -/
theorem point_in_region_t_range (t : ℝ) : 
  (1 : ℝ) - t + 1 > 0 → t < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_t_range_l4155_415569


namespace NUMINAMATH_CALUDE_max_winner_number_l4155_415595

/-- Represents a wrestler in the tournament -/
structure Wrestler :=
  (number : ℕ)

/-- The tournament setup -/
def Tournament :=
  { wrestlers : Finset Wrestler // wrestlers.card = 512 }

/-- Predicate for the winning condition in a match -/
def wins (w1 w2 : Wrestler) : Prop :=
  w1.number < w2.number ∧ w2.number - w1.number > 2

/-- The winner of the tournament -/
def tournamentWinner (t : Tournament) : Wrestler :=
  sorry

/-- Theorem stating the maximum possible qualification number of the winner -/
theorem max_winner_number (t : Tournament) : 
  (tournamentWinner t).number ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_max_winner_number_l4155_415595


namespace NUMINAMATH_CALUDE_bee_count_l4155_415584

theorem bee_count (initial_bees : ℕ) (incoming_bees : ℕ) : 
  initial_bees = 16 → incoming_bees = 8 → initial_bees + incoming_bees = 24 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l4155_415584


namespace NUMINAMATH_CALUDE_combined_weight_theorem_l4155_415536

/-- The combined weight that Rodney, Roger, and Ron can lift -/
def combinedWeight (rodney roger ron : ℕ) : ℕ := rodney + roger + ron

/-- Theorem stating the combined weight that Rodney, Roger, and Ron can lift -/
theorem combined_weight_theorem :
  ∀ (ron : ℕ),
  let roger := 4 * ron - 7
  let rodney := 2 * roger
  rodney = 146 →
  combinedWeight rodney roger ron = 239 := by
sorry

end NUMINAMATH_CALUDE_combined_weight_theorem_l4155_415536


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l4155_415593

theorem isosceles_triangle_quadratic_roots (k : ℝ) : 
  (∃ (a b : ℝ), 
    -- a and b are the roots of the quadratic equation
    a^2 - 12*a + k = 0 ∧ 
    b^2 - 12*b + k = 0 ∧ 
    -- a and b are equal (isosceles triangle)
    a = b ∧ 
    -- triangle inequality
    3 + a > b ∧ 3 + b > a ∧ a + b > 3 ∧
    -- one side is 3
    3 > 0) → 
  k = 36 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l4155_415593


namespace NUMINAMATH_CALUDE_rectangular_prism_dimensions_l4155_415546

/-- Represents the dimensions of a rectangular prism -/
structure Dimensions where
  m : ℕ+
  n : ℕ+
  r : ℕ+
  h_order : m ≤ n ∧ n ≤ r

/-- Calculates the number of cubes with no red faces -/
def k₀ (d : Dimensions) : ℤ :=
  (d.m.val - 2) * (d.n.val - 2) * (d.r.val - 2)

/-- Calculates the number of cubes with one red face -/
def k₁ (d : Dimensions) : ℤ :=
  2 * ((d.m.val - 2) * (d.n.val - 2) + (d.m.val - 2) * (d.r.val - 2) + (d.n.val - 2) * (d.r.val - 2))

/-- Calculates the number of cubes with two red faces -/
def k₂ (d : Dimensions) : ℤ :=
  4 * (d.m.val - 2 + d.n.val - 2 + d.r.val - 2)

/-- The main theorem stating the possible dimensions of the rectangular prism -/
theorem rectangular_prism_dimensions (d : Dimensions) :
  k₀ d + k₂ d - k₁ d = 1985 →
  (d.m.val = 5 ∧ d.n.val = 7 ∧ d.r.val = 663) ∨
  (d.m.val = 5 ∧ d.n.val = 5 ∧ d.r.val = 1981) ∨
  (d.m.val = 3 ∧ d.n.val = 3 ∧ d.r.val = 1981) ∨
  (d.m.val = 1 ∧ d.n.val = 7 ∧ d.r.val = 399) ∨
  (d.m.val = 1 ∧ d.n.val = 3 ∧ d.r.val = 1987) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_prism_dimensions_l4155_415546


namespace NUMINAMATH_CALUDE_nested_squares_difference_l4155_415578

/-- Given four nested squares with side lengths S₁ > S₂ > S₃ > S₄,
    where the differences between consecutive square side lengths are 11, 5, and 13 (from largest to smallest),
    prove that S₁ - S₄ = 29. -/
theorem nested_squares_difference (S₁ S₂ S₃ S₄ : ℝ) 
  (h₁ : S₁ = S₂ + 11)
  (h₂ : S₂ = S₃ + 5)
  (h₃ : S₃ = S₄ + 13) :
  S₁ - S₄ = 29 := by
  sorry

end NUMINAMATH_CALUDE_nested_squares_difference_l4155_415578


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l4155_415504

theorem polygon_sides_from_angle_sum (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 900 → (n - 2) * 180 = sum_angles → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l4155_415504


namespace NUMINAMATH_CALUDE_geometric_series_sum_l4155_415512

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (h : r ≠ 1) :
  let S := (a * (r^n - 1)) / (r - 1)
  a = -1 → r = -3 → n = 8 → S = 1640 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l4155_415512


namespace NUMINAMATH_CALUDE_initial_students_count_l4155_415575

theorem initial_students_count (initial_avg : ℝ) (new_student_weight : ℝ) (new_avg : ℝ) :
  initial_avg = 28 →
  new_student_weight = 4 →
  new_avg = 27.2 →
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 29 :=
by sorry

end NUMINAMATH_CALUDE_initial_students_count_l4155_415575


namespace NUMINAMATH_CALUDE_rectangle_ratio_l4155_415560

theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0) : 
  (s + 2*y = 3*s) →  -- Outer square side length
  (x + s = 3*s) →    -- Perpendicular arrangement
  ((3*s)^2 = 9*s^2)  -- Area ratio
  → x / y = 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l4155_415560


namespace NUMINAMATH_CALUDE_population_growth_duration_l4155_415547

/-- Proves that given specific population growth rates and a total net increase,
    the duration of the period is 24 hours. -/
theorem population_growth_duration :
  let birth_rate : ℕ := 3  -- people per second
  let death_rate : ℕ := 1  -- people per second
  let net_increase_rate : ℕ := birth_rate - death_rate
  let total_net_increase : ℕ := 172800
  let duration_seconds : ℕ := total_net_increase / net_increase_rate
  let seconds_per_hour : ℕ := 3600
  duration_seconds / seconds_per_hour = 24 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_duration_l4155_415547


namespace NUMINAMATH_CALUDE_remainder_of_2615_base12_div_9_l4155_415582

/-- Converts a base-12 digit to its decimal equivalent -/
def base12ToDecimal (digit : ℕ) : ℕ := digit

/-- Calculates the decimal value of a base-12 number given its digits -/
def base12Value (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  base12ToDecimal d₃ * 12^3 + base12ToDecimal d₂ * 12^2 + 
  base12ToDecimal d₁ * 12^1 + base12ToDecimal d₀ * 12^0

/-- The base-12 number 2615₁₂ -/
def num : ℕ := base12Value 2 6 1 5

theorem remainder_of_2615_base12_div_9 :
  num % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_of_2615_base12_div_9_l4155_415582


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l4155_415518

theorem smallest_integer_with_given_remainders :
  ∃ (a : ℕ), a > 0 ∧ a % 8 = 6 ∧ a % 9 = 5 ∧
  ∀ (b : ℕ), b > 0 → b % 8 = 6 → b % 9 = 5 → a ≤ b :=
by
  use 14
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l4155_415518


namespace NUMINAMATH_CALUDE_chinese_chess_draw_probability_l4155_415565

/-- Given the probabilities of winning and not losing for player A in Chinese chess,
    calculate the probability of a draw between player A and player B. -/
theorem chinese_chess_draw_probability
  (prob_win : ℝ) (prob_not_lose : ℝ)
  (h_win : prob_win = 0.4)
  (h_not_lose : prob_not_lose = 0.9) :
  prob_not_lose - prob_win = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chinese_chess_draw_probability_l4155_415565


namespace NUMINAMATH_CALUDE_x_value_when_y_is_one_l4155_415572

theorem x_value_when_y_is_one (x y : ℝ) : 
  y = 2 / (4 * x + 2) → y = 1 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_one_l4155_415572


namespace NUMINAMATH_CALUDE_ammonium_hydroxide_formation_l4155_415594

/-- Represents a chemical compound in a reaction --/
structure Compound where
  name : String
  moles : ℚ

/-- Represents a chemical reaction --/
structure Reaction where
  reactants : List Compound
  products : List Compound

/-- Finds the number of moles of a specific compound in a list of compounds --/
def findMoles (compounds : List Compound) (name : String) : ℚ :=
  match compounds.find? (fun c => c.name = name) with
  | some compound => compound.moles
  | none => 0

/-- The chemical reaction --/
def reaction : Reaction :=
  { reactants := [
      { name := "NH4Cl", moles := 1 },
      { name := "NaOH", moles := 1 }
    ],
    products := [
      { name := "NH4OH", moles := 1 },
      { name := "NaCl", moles := 1 }
    ]
  }

theorem ammonium_hydroxide_formation :
  findMoles reaction.products "NH4OH" = 1 :=
by sorry

end NUMINAMATH_CALUDE_ammonium_hydroxide_formation_l4155_415594


namespace NUMINAMATH_CALUDE_five_fridays_in_october_implies_five_mondays_in_november_l4155_415576

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

def october_has_five_fridays (year : Nat) : Prop :=
  ∃ dates : List Date,
    dates.length = 5 ∧
    ∀ d ∈ dates, d.dayOfWeek = DayOfWeek.Friday ∧ d.day ≤ 31

def november_has_five_mondays (year : Nat) : Prop :=
  ∃ dates : List Date,
    dates.length = 5 ∧
    ∀ d ∈ dates, d.dayOfWeek = DayOfWeek.Monday ∧ d.day ≤ 30

theorem five_fridays_in_october_implies_five_mondays_in_november (year : Nat) :
  october_has_five_fridays year → november_has_five_mondays year :=
by
  sorry


end NUMINAMATH_CALUDE_five_fridays_in_october_implies_five_mondays_in_november_l4155_415576


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l4155_415503

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 2*x < 0}

-- State the theorem
theorem A_intersect_B_eq_open_interval :
  A ∩ B = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l4155_415503


namespace NUMINAMATH_CALUDE_jump_rope_cost_l4155_415548

/-- The cost of Dalton's desired items --/
structure ItemCosts where
  board_game : ℕ
  playground_ball : ℕ
  jump_rope : ℕ

/-- Dalton's available money and additional need --/
structure DaltonMoney where
  allowance : ℕ
  uncle_gift : ℕ
  additional_need : ℕ

/-- Theorem: Given the costs of items and Dalton's available money, 
    prove that the jump rope costs $7 --/
theorem jump_rope_cost 
  (costs : ItemCosts) 
  (money : DaltonMoney) 
  (h1 : costs.board_game = 12)
  (h2 : costs.playground_ball = 4)
  (h3 : money.allowance = 6)
  (h4 : money.uncle_gift = 13)
  (h5 : money.additional_need = 4)
  (h6 : costs.board_game + costs.playground_ball + costs.jump_rope = 
        money.allowance + money.uncle_gift + money.additional_need) :
  costs.jump_rope = 7 := by
  sorry


end NUMINAMATH_CALUDE_jump_rope_cost_l4155_415548


namespace NUMINAMATH_CALUDE_cost_price_is_47_5_l4155_415553

/-- Given an article with a marked price and discount rate, calculates the cost price -/
def calculate_cost_price (marked_price : ℚ) (discount_rate : ℚ) (profit_rate : ℚ) : ℚ :=
  let selling_price := marked_price * (1 - discount_rate)
  selling_price / (1 + profit_rate)

/-- Theorem stating that the cost price of the article is 47.5 given the conditions -/
theorem cost_price_is_47_5 :
  let marked_price : ℚ := 74.21875
  let discount_rate : ℚ := 0.20
  let profit_rate : ℚ := 0.25
  calculate_cost_price marked_price discount_rate profit_rate = 47.5 := by
  sorry

#eval calculate_cost_price 74.21875 0.20 0.25

end NUMINAMATH_CALUDE_cost_price_is_47_5_l4155_415553


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4155_415533

/-- The quadratic equation 3x^2 - 4x + 1 = 0 has two distinct real roots -/
theorem quadratic_equation_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 3 * x₁^2 - 4 * x₁ + 1 = 0 ∧ 3 * x₂^2 - 4 * x₂ + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4155_415533


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l4155_415583

theorem arithmetic_mean_problem (a₁ a₂ a₃ a₄ a₅ a₆ A : ℝ) 
  (h_mean : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = A)
  (h_first_four : (a₁ + a₂ + a₃ + a₄) / 4 = A + 10)
  (h_last_four : (a₃ + a₄ + a₅ + a₆) / 4 = A - 7) :
  (a₁ + a₂ + a₅ + a₆) / 4 = A - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l4155_415583


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4155_415526

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a3 : a 3 = 3/2) 
  (h_S3 : (a 1) + (a 2) + (a 3) = 9/2) :
  a 1 / a 0 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4155_415526


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l4155_415554

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) : 
  total_clips = 81 → num_boxes = 9 → clips_per_box = total_clips / num_boxes → clips_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l4155_415554


namespace NUMINAMATH_CALUDE_special_triangle_ac_length_l4155_415501

/-- A triangle ABC with a point D on side AC, satisfying specific conditions -/
structure SpecialTriangle where
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point D on side AC -/
  D : ℝ × ℝ
  /-- AB is greater than BC -/
  ab_gt_bc : dist A B > dist B C
  /-- BC equals 6 -/
  bc_eq_six : dist B C = 6
  /-- BD equals 7 -/
  bd_eq_seven : dist B D = 7
  /-- Triangle ABD is isosceles -/
  abd_isosceles : dist A B = dist A D ∨ dist A B = dist B D
  /-- Triangle BCD is isosceles -/
  bcd_isosceles : dist B C = dist C D ∨ dist B D = dist C D
  /-- D lies on AC -/
  d_on_ac : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = ((1 - t) • A.1 + t • C.1, (1 - t) • A.2 + t • C.2)

/-- The length of AC in the special triangle is 13 -/
theorem special_triangle_ac_length (t : SpecialTriangle) : dist t.A t.C = 13 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_ac_length_l4155_415501


namespace NUMINAMATH_CALUDE_tammy_haircuts_l4155_415515

/-- The number of paid haircuts required to get a free haircut -/
def haircuts_for_free : ℕ := 14

/-- The number of free haircuts Tammy has already received -/
def free_haircuts_received : ℕ := 5

/-- The number of haircuts Tammy needs for her next free one -/
def haircuts_until_next_free : ℕ := 5

/-- The total number of haircuts Tammy has gotten -/
def total_haircuts : ℕ := 79

theorem tammy_haircuts :
  total_haircuts = 
    (free_haircuts_received * haircuts_for_free) + 
    (haircuts_for_free - haircuts_until_next_free) :=
by sorry

end NUMINAMATH_CALUDE_tammy_haircuts_l4155_415515


namespace NUMINAMATH_CALUDE_cubic_factorization_l4155_415598

theorem cubic_factorization (x : ℝ) : x^3 - 6*x^2 + 9*x = x*(x-3)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l4155_415598


namespace NUMINAMATH_CALUDE_probabilities_sum_to_one_l4155_415561

def p₁ : ℝ := 0.22
def p₂ : ℝ := 0.31
def p₃ : ℝ := 0.47

theorem probabilities_sum_to_one : p₁ + p₂ + p₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_probabilities_sum_to_one_l4155_415561


namespace NUMINAMATH_CALUDE_rachel_bought_seven_chairs_l4155_415509

/-- Calculates the number of chairs Rachel bought given the number of tables,
    time spent per furniture piece, and total time spent. -/
def chairs_bought (num_tables : ℕ) (time_per_piece : ℕ) (total_time : ℕ) : ℕ :=
  (total_time - num_tables * time_per_piece) / time_per_piece

/-- Theorem stating that Rachel bought 7 chairs given the problem conditions. -/
theorem rachel_bought_seven_chairs :
  chairs_bought 3 4 40 = 7 := by
  sorry

end NUMINAMATH_CALUDE_rachel_bought_seven_chairs_l4155_415509


namespace NUMINAMATH_CALUDE_parabola_translation_l4155_415543

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The equation of a parabola in vertex form -/
def Parabola.equation (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * (x - p.h)^2 + p.k

/-- Vertical translation of a parabola -/
def verticalTranslate (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + dy }

theorem parabola_translation (x y : ℝ) :
  let p := Parabola.mk 3 0 0
  let p_translated := verticalTranslate p 3
  Parabola.equation p_translated x y ↔ y = 3 * x^2 + 3 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l4155_415543


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l4155_415530

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l4155_415530


namespace NUMINAMATH_CALUDE_octagon_side_length_l4155_415511

theorem octagon_side_length (square_side : ℝ) (h : square_side = 1) :
  let octagon_side := square_side - 2 * ((square_side * (1 - 1 / Real.sqrt 2)) / 2)
  octagon_side = 1 - Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_side_length_l4155_415511


namespace NUMINAMATH_CALUDE_total_handshakes_l4155_415588

-- Define the number of people in each group
def group_a : Nat := 25  -- people who all know each other
def group_b : Nat := 10  -- people who know no one
def group_c : Nat := 5   -- people who only know each other

-- Define the total number of people
def total_people : Nat := group_a + group_b + group_c

-- Define the function to calculate handshakes between two groups
def handshakes_between (group1 : Nat) (group2 : Nat) : Nat := group1 * group2

-- Define the function to calculate handshakes within a group
def handshakes_within (group : Nat) : Nat := group * (group - 1) / 2

-- Theorem statement
theorem total_handshakes : 
  handshakes_between group_a group_b + 
  handshakes_between group_a group_c + 
  handshakes_between group_b group_c + 
  handshakes_within group_b = 470 := by
  sorry

end NUMINAMATH_CALUDE_total_handshakes_l4155_415588


namespace NUMINAMATH_CALUDE_largest_number_l4155_415549

def a : ℚ := 24680 + 1 / 13579
def b : ℚ := 24680 - 1 / 13579
def c : ℚ := 24680 * (1 / 13579)
def d : ℚ := 24680 / (1 / 13579)
def e : ℚ := 24680.13579

theorem largest_number : d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l4155_415549


namespace NUMINAMATH_CALUDE_water_bottles_per_day_l4155_415570

theorem water_bottles_per_day 
  (total_bottles : ℕ) 
  (total_days : ℕ) 
  (h1 : total_bottles = 28) 
  (h2 : total_days = 4) 
  (h3 : total_days ≠ 0) : 
  total_bottles / total_days = 7 := by
sorry

end NUMINAMATH_CALUDE_water_bottles_per_day_l4155_415570


namespace NUMINAMATH_CALUDE_carol_nickels_l4155_415500

/-- Represents the contents of Carol's piggy bank -/
structure PiggyBank where
  quarters : ℕ
  nickels : ℕ
  total_cents : ℕ
  nickel_quarter_diff : nickels = quarters + 7
  total_value : total_cents = 5 * nickels + 25 * quarters

/-- Theorem stating that Carol has 21 nickels in her piggy bank -/
theorem carol_nickels (bank : PiggyBank) (h : bank.total_cents = 455) : bank.nickels = 21 := by
  sorry

end NUMINAMATH_CALUDE_carol_nickels_l4155_415500


namespace NUMINAMATH_CALUDE_sum_of_ages_l4155_415590

/-- Given the ages and relationships of Beckett, Olaf, Shannen, and Jack, prove that the sum of their ages is 71 years. -/
theorem sum_of_ages (beckett_age olaf_age shannen_age jack_age : ℕ) : 
  beckett_age = 12 →
  olaf_age = beckett_age + 3 →
  shannen_age = olaf_age - 2 →
  jack_age = 2 * shannen_age + 5 →
  beckett_age + olaf_age + shannen_age + jack_age = 71 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l4155_415590


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l4155_415559

theorem jake_and_sister_weight (jake_weight : ℕ) (sister_weight : ℕ) : 
  jake_weight = 108 →
  jake_weight - 12 = 2 * sister_weight →
  jake_weight + sister_weight = 156 :=
by sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l4155_415559


namespace NUMINAMATH_CALUDE_octagon_diagonals_l4155_415589

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon is a polygon with 8 vertices -/
def octagon_vertices : ℕ := 8

theorem octagon_diagonals :
  num_diagonals octagon_vertices = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l4155_415589


namespace NUMINAMATH_CALUDE_closest_to_fraction_l4155_415506

def options : List ℝ := [500, 1000, 2000, 2100, 4000]

theorem closest_to_fraction (options : List ℝ) :
  2100 = (options.filter (λ x => ∀ y ∈ options, |850 / 0.42 - x| ≤ |850 / 0.42 - y|)).head! :=
by sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l4155_415506


namespace NUMINAMATH_CALUDE_lillian_candy_count_l4155_415538

theorem lillian_candy_count (initial_candies : ℕ) (additional_candies : ℕ) : 
  initial_candies = 88 → additional_candies = 5 → initial_candies + additional_candies = 93 := by
  sorry

end NUMINAMATH_CALUDE_lillian_candy_count_l4155_415538


namespace NUMINAMATH_CALUDE_a_minus_b_value_l4155_415568

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 4) (h2 : b^2 = 9) (h3 : a/b > 0) :
  a - b = 1 ∨ a - b = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l4155_415568


namespace NUMINAMATH_CALUDE_factory_shutdown_probabilities_l4155_415597

/-- The number of factories -/
def num_factories : ℕ := 5

/-- The number of days in a week -/
def num_days : ℕ := 7

/-- The probability of all factories choosing Sunday to shut down -/
def prob_all_sunday : ℚ := 1 / 7^num_factories

/-- The probability of at least two factories choosing the same day to shut down -/
def prob_at_least_two_same : ℚ := 1 - (num_days.factorial / (num_days - num_factories).factorial) / 7^num_factories

theorem factory_shutdown_probabilities :
  (prob_all_sunday = 1 / 16807) ∧
  (prob_at_least_two_same = 2041 / 2401) := by
  sorry


end NUMINAMATH_CALUDE_factory_shutdown_probabilities_l4155_415597


namespace NUMINAMATH_CALUDE_sequence_properties_l4155_415571

def sequence_a (n : ℕ) : ℝ := 2 * n - 1

def sum_S (n : ℕ) : ℝ := n^2

theorem sequence_properties :
  (∀ n : ℕ, n > 0 → 4 * (sum_S n) = (sequence_a n + 1)^2) →
  (∀ n : ℕ, n > 0 → sequence_a n = 2 * n - 1) ∧
  (sequence_a 1 = 1) ∧
  (sum_S 20 = 400) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l4155_415571


namespace NUMINAMATH_CALUDE_prime_power_sum_congruence_and_evenness_l4155_415599

theorem prime_power_sum_congruence_and_evenness (p q : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p ≠ q) :
  (p^q + q^p) % (p*q) = (p + q) % (p*q) ∧ 
  (p ≠ 2 → q ≠ 2 → Even ((p^q + q^p) / (p*q))) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_sum_congruence_and_evenness_l4155_415599


namespace NUMINAMATH_CALUDE_angle_C_is_120_degrees_l4155_415567

theorem angle_C_is_120_degrees 
  (A B : ℝ) 
  (m : ℝ × ℝ) 
  (n : ℝ × ℝ) 
  (h1 : m = (Real.sqrt 3 * Real.sin A, Real.sin B))
  (h2 : n = (Real.cos B, Real.sqrt 3 * Real.cos A))
  (h3 : m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B))
  : ∃ C : ℝ, C = 2 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_angle_C_is_120_degrees_l4155_415567


namespace NUMINAMATH_CALUDE_first_sample_in_systematic_sampling_l4155_415541

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun n => firstSample + (n - 1) * (total / sampleSize)

theorem first_sample_in_systematic_sampling
  (total : ℕ) (sampleSize : ℕ) (fourthSample : ℕ) 
  (h1 : total = 800)
  (h2 : sampleSize = 80)
  (h3 : fourthSample = 39) :
  ∃ firstSample : ℕ, 
    firstSample ∈ Finset.range 10 ∧ 
    systematicSample total sampleSize firstSample 4 = fourthSample ∧
    firstSample = 9 :=
by sorry

end NUMINAMATH_CALUDE_first_sample_in_systematic_sampling_l4155_415541


namespace NUMINAMATH_CALUDE_kindergarten_lineup_probability_l4155_415558

theorem kindergarten_lineup_probability :
  let total_children : ℕ := 20
  let num_girls : ℕ := 11
  let num_boys : ℕ := 9
  let favorable_arrangements := Nat.choose 14 9 + 6 * Nat.choose 13 8
  let total_arrangements := Nat.choose total_children num_boys
  (favorable_arrangements : ℚ) / total_arrangements =
    (Nat.choose 14 9 + 6 * Nat.choose 13 8 : ℚ) / Nat.choose 20 9 :=
by sorry

end NUMINAMATH_CALUDE_kindergarten_lineup_probability_l4155_415558


namespace NUMINAMATH_CALUDE_albert_sequence_theorem_l4155_415563

/-- Represents the sequence of positive integers starting with 1 or 2 in increasing order -/
def albert_sequence : ℕ → ℕ := sorry

/-- Returns the nth digit in Albert's sequence -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- The three-digit number formed by the 1498th, 1499th, and 1500th digits -/
def target_number : ℕ := 100 * (nth_digit 1498) + 10 * (nth_digit 1499) + (nth_digit 1500)

theorem albert_sequence_theorem : target_number = 121 := by sorry

end NUMINAMATH_CALUDE_albert_sequence_theorem_l4155_415563


namespace NUMINAMATH_CALUDE_school_journey_time_l4155_415514

/-- The time for a journey to school, given specific conditions about forgetting an item -/
theorem school_journey_time : ∃ (t : ℝ), 
  (t > 0) ∧ 
  (t - 6 > 0) ∧
  (∃ (x : ℝ), x > 0 ∧ x = t / 5) ∧
  ((9/5) * t = t + 2) ∧ 
  (t = 20) := by
  sorry

end NUMINAMATH_CALUDE_school_journey_time_l4155_415514
