import Mathlib

namespace NUMINAMATH_CALUDE_M_properties_l1104_110430

-- Define the operation M
def M : ℚ → ℚ
| n => if (↑n : ℚ).den = 1 
       then (↑n : ℚ).num - 3 
       else -(1 / ((↑n : ℚ).den ^ 2))

-- Theorem statement
theorem M_properties : 
  (M 28 * M (1/5) = -1) ∧ 
  (-1 / M 39 / (-M (1/6)) = -1) := by
  sorry

end NUMINAMATH_CALUDE_M_properties_l1104_110430


namespace NUMINAMATH_CALUDE_jan_math_problem_l1104_110479

-- Define the operation of rounding to the nearest ten
def roundToNearestTen (x : ℤ) : ℤ :=
  10 * ((x + 5) / 10)

-- Theorem statement
theorem jan_math_problem :
  roundToNearestTen (83 - 29 + 58) = 110 := by
  sorry

end NUMINAMATH_CALUDE_jan_math_problem_l1104_110479


namespace NUMINAMATH_CALUDE_custom_operation_equation_l1104_110451

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 4 * a - b

-- State the theorem
theorem custom_operation_equation :
  ∃ x : ℝ, (star 4 (star 3 x) = 2) ∧ (x = -2) := by sorry

end NUMINAMATH_CALUDE_custom_operation_equation_l1104_110451


namespace NUMINAMATH_CALUDE_probability_of_valid_pair_l1104_110435

def ball_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

def is_valid_pair (x y : ℕ) : Bool :=
  x ∈ ball_numbers ∧ y ∈ ball_numbers ∧ Even (x * y) ∧ x * y > 14

def valid_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (λ p => is_valid_pair p.1 p.2) (ball_numbers.product ball_numbers)

theorem probability_of_valid_pair :
  (valid_pairs.card : ℚ) / (ball_numbers.card * ball_numbers.card : ℚ) = 16 / 49 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_valid_pair_l1104_110435


namespace NUMINAMATH_CALUDE_water_balloon_packs_l1104_110434

/-- Represents the number of packs of water balloons --/
def num_own_packs : ℕ := 3

/-- Represents the number of balloons in each pack --/
def balloons_per_pack : ℕ := 6

/-- Represents the number of neighbor's packs used --/
def num_neighbor_packs : ℕ := 2

/-- Represents the extra balloons Milly takes --/
def extra_balloons : ℕ := 7

/-- Represents the number of balloons Floretta is left with --/
def floretta_balloons : ℕ := 8

theorem water_balloon_packs :
  num_own_packs * balloons_per_pack + num_neighbor_packs * balloons_per_pack =
  2 * (floretta_balloons + extra_balloons) :=
sorry

end NUMINAMATH_CALUDE_water_balloon_packs_l1104_110434


namespace NUMINAMATH_CALUDE_no_super_sudoku_l1104_110446

/-- Represents a 9x9 grid of integers -/
def Grid := Fin 9 → Fin 9 → Fin 9

/-- Checks if a given row contains each number 1-9 exactly once -/
def validRow (g : Grid) (row : Fin 9) : Prop :=
  ∀ n : Fin 9, ∃! col : Fin 9, g row col = n

/-- Checks if a given column contains each number 1-9 exactly once -/
def validColumn (g : Grid) (col : Fin 9) : Prop :=
  ∀ n : Fin 9, ∃! row : Fin 9, g row col = n

/-- Checks if a given 3x3 subsquare contains each number 1-9 exactly once -/
def validSubsquare (g : Grid) (startRow startCol : Fin 3) : Prop :=
  ∀ n : Fin 9, ∃! (row col : Fin 3), g (3 * startRow + row) (3 * startCol + col) = n

/-- Defines a super-sudoku grid -/
def isSuperSudoku (g : Grid) : Prop :=
  (∀ row : Fin 9, validRow g row) ∧
  (∀ col : Fin 9, validColumn g col) ∧
  (∀ startRow startCol : Fin 3, validSubsquare g startRow startCol)

/-- Theorem: There are no possible super-sudoku grids -/
theorem no_super_sudoku : ¬∃ g : Grid, isSuperSudoku g := by
  sorry

end NUMINAMATH_CALUDE_no_super_sudoku_l1104_110446


namespace NUMINAMATH_CALUDE_max_area_of_nonoverlapping_triangle_l1104_110423

/-- A triangle on a coordinate plane -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Check if two triangles overlap -/
def overlap (t1 t2 : Triangle) : Prop := sorry

/-- Translation of a triangle by an integer vector -/
def translate (t : Triangle) (v : ℤ × ℤ) : Triangle := sorry

/-- Area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- A triangle is valid if its translations by integer vectors do not overlap -/
def valid_triangle (t : Triangle) : Prop :=
  ∀ v : ℤ × ℤ, ¬(overlap t (translate t v))

theorem max_area_of_nonoverlapping_triangle :
  ∃ (t : Triangle), valid_triangle t ∧ area t = 2/3 ∧
  ∀ (t' : Triangle), valid_triangle t' → area t' ≤ 2/3 := by sorry

end NUMINAMATH_CALUDE_max_area_of_nonoverlapping_triangle_l1104_110423


namespace NUMINAMATH_CALUDE_find_n_l1104_110485

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem find_n : ∃ n : ℕ, n * factorial (n + 1) + factorial (n + 1) = 5040 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1104_110485


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1104_110478

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_3 + a_5 = 12, then a_4 = 6 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) (h_sum : a 3 + a 5 = 12) : 
    a 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1104_110478


namespace NUMINAMATH_CALUDE_complex_multiplication_l1104_110450

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (2 + i) * (3 + i) = 5 + 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1104_110450


namespace NUMINAMATH_CALUDE_extended_tile_ratio_l1104_110442

theorem extended_tile_ratio (initial_black : ℕ) (initial_white : ℕ) 
  (h1 : initial_black = 7)
  (h2 : initial_white = 18)
  (h3 : initial_black + initial_white = 25) :
  let side_length : ℕ := (initial_black + initial_white).sqrt
  let extended_side_length : ℕ := side_length + 2
  let extended_black : ℕ := initial_black + 4 * side_length + 4
  let extended_white : ℕ := initial_white
  (extended_black : ℚ) / extended_white = 31 / 18 := by
sorry

end NUMINAMATH_CALUDE_extended_tile_ratio_l1104_110442


namespace NUMINAMATH_CALUDE_range_of_roots_difference_l1104_110482

-- Define the function g
def g (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- Define the derivative of g as f
def f (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- State the theorem
theorem range_of_roots_difference
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hsum : a + 2 * b + 3 * c = 0)
  (hpos : f a b c 0 * f a b c 1 > 0)
  (x₁ x₂ : ℝ)
  (hroot₁ : f a b c x₁ = 0)
  (hroot₂ : f a b c x₂ = 0) :
  ∃ y, y ∈ Set.Icc 0 (2/3) ∧ |x₁ - x₂| = y :=
sorry

end NUMINAMATH_CALUDE_range_of_roots_difference_l1104_110482


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1104_110400

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The derivative of the function -/
def HasDerivative (f : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = 2 * x + 2

/-- The function has two equal real roots -/
def HasEqualRoots (f : ℝ → ℝ) : Prop :=
  ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r) ∧ (deriv f r = 0)

/-- The main theorem -/
theorem quadratic_function_theorem (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f) 
  (h2 : HasDerivative f) 
  (h3 : HasEqualRoots f) : 
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1104_110400


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_eight_l1104_110426

theorem intersection_nonempty_implies_a_geq_neg_eight (a : ℝ) : 
  (∃ x : ℝ, x ∈ {x | 1 ≤ x ∧ x ≤ 2} ∩ {x | x^2 + 2*x + a ≥ 0}) → a ≥ -8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_geq_neg_eight_l1104_110426


namespace NUMINAMATH_CALUDE_intersection_and_union_for_negative_one_intersection_equals_B_iff_l1104_110411

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a+2}

theorem intersection_and_union_for_negative_one :
  (A ∩ B (-1) = {x | -2 ≤ x ∧ x ≤ -1}) ∧
  (A ∪ B (-1) = {x | x ≤ 1 ∨ x ≥ 5}) := by sorry

theorem intersection_equals_B_iff :
  ∀ a : ℝ, A ∩ B a = B a ↔ a ≤ -3 ∨ a > 2 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_for_negative_one_intersection_equals_B_iff_l1104_110411


namespace NUMINAMATH_CALUDE_periodic_decimal_sum_l1104_110427

/-- The sum of 0.3̅, 0.0̅4̅, and 0.0̅0̅5̅ is equal to 14/37 -/
theorem periodic_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) = 14 / 37 := by sorry

end NUMINAMATH_CALUDE_periodic_decimal_sum_l1104_110427


namespace NUMINAMATH_CALUDE_gold_coin_percentage_is_55_25_percent_l1104_110489

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  bead_percent : Real
  gold_coin_percent_of_coins : Real

/-- Calculates the percentage of gold coins in the urn -/
def gold_coin_percent (urn : UrnComposition) : Real :=
  (1 - urn.bead_percent) * urn.gold_coin_percent_of_coins

/-- Theorem stating that the percentage of gold coins in the urn is 55.25% -/
theorem gold_coin_percentage_is_55_25_percent 
  (urn : UrnComposition) 
  (h1 : urn.bead_percent = 0.15) 
  (h2 : urn.gold_coin_percent_of_coins = 0.65) : 
  gold_coin_percent urn = 0.5525 := by
  sorry

#eval gold_coin_percent { bead_percent := 0.15, gold_coin_percent_of_coins := 0.65 }

end NUMINAMATH_CALUDE_gold_coin_percentage_is_55_25_percent_l1104_110489


namespace NUMINAMATH_CALUDE_triangle_side_sum_l1104_110494

theorem triangle_side_sum (a b c : ℝ) (h_angles : a = 30 ∧ b = 45 ∧ c = 105) 
  (h_sum : a + b + c = 180) (h_side : ∃ side : ℝ, side = 6 * Real.sqrt 2) : 
  ∃ (x y : ℝ), x + y = 18 + 6 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l1104_110494


namespace NUMINAMATH_CALUDE_number_of_boys_l1104_110497

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 100 → 
  boys + girls = total → 
  girls = boys → 
  boys = 50 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l1104_110497


namespace NUMINAMATH_CALUDE_point_not_on_line_l1104_110403

theorem point_not_on_line (a c : ℝ) (h : a * c > 0) :
  ¬(∃ (x y : ℝ), x = 2500 ∧ y = 0 ∧ y = a * x + c) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l1104_110403


namespace NUMINAMATH_CALUDE_meena_cookie_sales_l1104_110460

/-- The number of dozens of cookies Meena sold to Mr. Stone -/
def cookies_sold_to_mr_stone (total_dozens : ℕ) (brock_cookies : ℕ) (katy_multiplier : ℕ) (cookies_left : ℕ) : ℕ :=
  let total_cookies := total_dozens * 12
  let katy_cookies := brock_cookies * katy_multiplier
  let sold_cookies := total_cookies - cookies_left
  let mr_stone_cookies := sold_cookies - (brock_cookies + katy_cookies)
  mr_stone_cookies / 12

theorem meena_cookie_sales : 
  cookies_sold_to_mr_stone 5 7 2 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_meena_cookie_sales_l1104_110460


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_l1104_110415

theorem smallest_five_digit_multiple : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit number
  n % 9 = 0 ∧                 -- multiple of 9
  n % 6 = 0 ∧                 -- multiple of 6
  n % 2 = 0 ∧                 -- multiple of 2
  (∀ m : ℕ, 
    (m ≥ 10000 ∧ m < 100000) ∧ 
    m % 9 = 0 ∧ 
    m % 6 = 0 ∧ 
    m % 2 = 0 → 
    n ≤ m) ∧
  n = 10008 := by
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_l1104_110415


namespace NUMINAMATH_CALUDE_symmetric_line_correct_l1104_110418

/-- Given two lines in a plane, this function returns the equation of the line 
    that is symmetric to the first line with respect to the second line. -/
def symmetricLine (l₁ l₂ : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  sorry

/-- The first given line l₁ -/
def l₁ : ℝ → ℝ → Prop :=
  fun x y ↦ 3 * x - y - 3 = 0

/-- The second given line l₂ -/
def l₂ : ℝ → ℝ → Prop :=
  fun x y ↦ x + y - 1 = 0

/-- The expected symmetric line l₃ -/
def l₃ : ℝ → ℝ → Prop :=
  fun x y ↦ x - 3 * y - 1 = 0

theorem symmetric_line_correct :
  symmetricLine l₁ l₂ = l₃ := by sorry

end NUMINAMATH_CALUDE_symmetric_line_correct_l1104_110418


namespace NUMINAMATH_CALUDE_min_operations_to_500_l1104_110425

/-- Represents the available operations on the calculator --/
inductive Operation
  | addOne
  | subOne
  | mulTwo

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.addOne => n + 1
  | Operation.subOne => if n > 0 then n - 1 else 0
  | Operation.mulTwo => n * 2

/-- Checks if a sequence of operations contains all three operation types --/
def containsAllOperations (ops : List Operation) : Prop :=
  Operation.addOne ∈ ops ∧ Operation.subOne ∈ ops ∧ Operation.mulTwo ∈ ops

/-- Applies a sequence of operations to a starting number --/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Theorem: The minimum number of operations to reach 500 from 1 is 13 --/
theorem min_operations_to_500 :
  (∃ (ops : List Operation),
    applyOperations 1 ops = 500 ∧
    containsAllOperations ops ∧
    ops.length = 13) ∧
  (∀ (ops : List Operation),
    applyOperations 1 ops = 500 →
    containsAllOperations ops →
    ops.length ≥ 13) :=
  sorry

end NUMINAMATH_CALUDE_min_operations_to_500_l1104_110425


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l1104_110492

theorem binomial_coefficient_problem (h1 : Nat.choose 18 11 = 31824)
                                     (h2 : Nat.choose 18 12 = 18564)
                                     (h3 : Nat.choose 20 13 = 77520) :
  Nat.choose 19 13 = 27132 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l1104_110492


namespace NUMINAMATH_CALUDE_poplar_tree_count_l1104_110410

theorem poplar_tree_count : ∃ (poplar willow : ℕ),
  poplar + willow = 120 ∧ poplar + 10 = willow ∧ poplar = 55 := by
  sorry

end NUMINAMATH_CALUDE_poplar_tree_count_l1104_110410


namespace NUMINAMATH_CALUDE_line_intercepts_minimum_sum_l1104_110404

theorem line_intercepts_minimum_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_point : a + b = a * b) : 
  (a / b + b / a) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = a₀ * b₀ ∧ a₀ / b₀ + b₀ / a₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_minimum_sum_l1104_110404


namespace NUMINAMATH_CALUDE_no_natural_pair_satisfies_condition_l1104_110412

theorem no_natural_pair_satisfies_condition : 
  ¬∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ (b^a ∣ a^b - 1) :=
by sorry

end NUMINAMATH_CALUDE_no_natural_pair_satisfies_condition_l1104_110412


namespace NUMINAMATH_CALUDE_mary_saw_256_snakes_l1104_110401

/-- The number of breeding balls -/
def num_breeding_balls : Nat := 7

/-- The number of snakes in each breeding ball -/
def snakes_in_balls : List Nat := [15, 20, 25, 30, 35, 40, 45]

/-- The number of extra pairs of snakes -/
def extra_pairs : Nat := 23

/-- The total number of snakes Mary saw -/
def total_snakes : Nat := (List.sum snakes_in_balls) + (2 * extra_pairs)

theorem mary_saw_256_snakes :
  total_snakes = 256 := by sorry

end NUMINAMATH_CALUDE_mary_saw_256_snakes_l1104_110401


namespace NUMINAMATH_CALUDE_abs_neg_sqrt_two_l1104_110488

theorem abs_neg_sqrt_two : |(-Real.sqrt 2)| = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_abs_neg_sqrt_two_l1104_110488


namespace NUMINAMATH_CALUDE_newspaper_photos_l1104_110455

/-- The total number of photos in a newspaper with specified page types -/
def total_photos (pages_with_two_photos pages_with_three_photos : ℕ) : ℕ :=
  2 * pages_with_two_photos + 3 * pages_with_three_photos

/-- Theorem stating that the total number of photos in the newspaper is 51 -/
theorem newspaper_photos : total_photos 12 9 = 51 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_photos_l1104_110455


namespace NUMINAMATH_CALUDE_friend_team_assignment_l1104_110462

theorem friend_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k ^ n = 65536 :=
sorry

end NUMINAMATH_CALUDE_friend_team_assignment_l1104_110462


namespace NUMINAMATH_CALUDE_remainder_div_nine_l1104_110441

theorem remainder_div_nine (n : ℕ) (h : n % 18 = 11) : n % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_div_nine_l1104_110441


namespace NUMINAMATH_CALUDE_bryan_spent_1500_l1104_110493

/-- The total amount spent by Bryan on t-shirts and pants -/
def total_spent (num_tshirts : ℕ) (tshirt_price : ℕ) (num_pants : ℕ) (pants_price : ℕ) : ℕ :=
  num_tshirts * tshirt_price + num_pants * pants_price

/-- Theorem: Bryan spent $1500 on 5 t-shirts at $100 each and 4 pairs of pants at $250 each -/
theorem bryan_spent_1500 : total_spent 5 100 4 250 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_bryan_spent_1500_l1104_110493


namespace NUMINAMATH_CALUDE_square_coverage_l1104_110447

theorem square_coverage (unit_square_area : ℝ) (large_square_side : ℝ) :
  unit_square_area = 1 →
  large_square_side = 5 / 4 →
  3 * unit_square_area ≥ large_square_side ^ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_square_coverage_l1104_110447


namespace NUMINAMATH_CALUDE_pet_store_bird_count_l1104_110465

theorem pet_store_bird_count :
  ∀ (num_cages : ℕ) (parrots_per_cage : ℕ) (parakeets_per_cage : ℕ),
    num_cages = 8 →
    parrots_per_cage = 2 →
    parakeets_per_cage = 7 →
    num_cages * (parrots_per_cage + parakeets_per_cage) = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_store_bird_count_l1104_110465


namespace NUMINAMATH_CALUDE_square_root_three_expansion_special_case_square_root_three_simplify_square_root_expression_l1104_110409

-- Part 1
theorem square_root_three_expansion (a b m n : ℕ+) :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2 →
  a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem special_case_square_root_three (a m n : ℕ+) :
  a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2 →
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem simplify_square_root_expression :
  Real.sqrt (25 + 4 * Real.sqrt 6) = 5 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_special_case_square_root_three_simplify_square_root_expression_l1104_110409


namespace NUMINAMATH_CALUDE_inequality_proof_l1104_110417

theorem inequality_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  3 * ((x^2 / y^2) + (y^2 / x^2)) - 8 * ((x / y) + (y / x)) + 10 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1104_110417


namespace NUMINAMATH_CALUDE_marius_monica_difference_l1104_110471

/-- The number of subjects taken by students Millie, Monica, and Marius. -/
structure SubjectCounts where
  millie : ℕ
  monica : ℕ
  marius : ℕ

/-- The conditions of the problem. -/
def problem_conditions (counts : SubjectCounts) : Prop :=
  counts.millie = counts.marius + 3 ∧
  counts.marius > counts.monica ∧
  counts.monica = 10 ∧
  counts.millie + counts.monica + counts.marius = 41

/-- The theorem stating that Marius takes 4 more subjects than Monica. -/
theorem marius_monica_difference (counts : SubjectCounts) 
  (h : problem_conditions counts) : counts.marius - counts.monica = 4 := by
  sorry

end NUMINAMATH_CALUDE_marius_monica_difference_l1104_110471


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l1104_110496

def f (x : ℝ) := x^3 - 3*x + 1

theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-3) 0, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = max) ∧
    (∀ x ∈ Set.Icc (-3) 0, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-3) 0, f x = min) ∧
    max = 3 ∧ min = -17 :=
by sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l1104_110496


namespace NUMINAMATH_CALUDE_student_count_l1104_110406

theorem student_count (average_decrease : ℝ) (weight_difference : ℝ) : 
  average_decrease = 8 → weight_difference = 32 → (weight_difference / average_decrease : ℝ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1104_110406


namespace NUMINAMATH_CALUDE_billys_restaurant_bill_l1104_110416

/-- The total bill for a group at Billy's Restaurant -/
def total_bill (adults children meal_cost : ℕ) : ℕ :=
  (adults + children) * meal_cost

/-- Theorem: The total bill for a group of 2 adults and 5 children, 
    with each meal costing 3 dollars, is 21 dollars -/
theorem billys_restaurant_bill : total_bill 2 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_billys_restaurant_bill_l1104_110416


namespace NUMINAMATH_CALUDE_unique_triple_existence_l1104_110466

theorem unique_triple_existence : 
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  (1 / a = b + c) ∧ (1 / b = a + c) ∧ (1 / c = a + b) := by
sorry

end NUMINAMATH_CALUDE_unique_triple_existence_l1104_110466


namespace NUMINAMATH_CALUDE_quarterback_sacks_l1104_110486

theorem quarterback_sacks (total_attempts : ℕ) (no_throw_percentage : ℚ) (sack_ratio : ℚ) :
  total_attempts = 80 →
  no_throw_percentage = 30 / 100 →
  sack_ratio = 1 / 2 →
  ↑(total_attempts : ℕ) * no_throw_percentage * sack_ratio = 12 :=
by sorry

end NUMINAMATH_CALUDE_quarterback_sacks_l1104_110486


namespace NUMINAMATH_CALUDE_matrix_product_equality_l1104_110495

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 4]

theorem matrix_product_equality :
  A * B = !![17, -5; 16, -20] := by sorry

end NUMINAMATH_CALUDE_matrix_product_equality_l1104_110495


namespace NUMINAMATH_CALUDE_equilateral_hyperbola_properties_l1104_110449

/-- An equilateral hyperbola passing through point A(3,-1) with its axes of symmetry lying on the coordinate axes -/
def equilateral_hyperbola (x y : ℝ) : Prop :=
  x^2/8 - y^2/8 = 1

theorem equilateral_hyperbola_properties :
  -- The hyperbola passes through point A(3,-1)
  equilateral_hyperbola 3 (-1) ∧
  -- The axes of symmetry lie on the coordinate axes (implied by the equation form)
  ∀ (x y : ℝ), equilateral_hyperbola x y ↔ equilateral_hyperbola (-x) y ∧
  ∀ (x y : ℝ), equilateral_hyperbola x y ↔ equilateral_hyperbola x (-y) ∧
  -- The hyperbola is equilateral (asymptotes are perpendicular)
  ∃ (a : ℝ), a > 0 ∧ ∀ (x y : ℝ), equilateral_hyperbola x y ↔ x^2/a^2 - y^2/a^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_hyperbola_properties_l1104_110449


namespace NUMINAMATH_CALUDE_certain_number_proof_l1104_110422

theorem certain_number_proof : ∃! x : ℕ, (x - 16) % 37 = 0 ∧ (x - 16) / 37 = 23 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1104_110422


namespace NUMINAMATH_CALUDE_minimal_disks_is_16_l1104_110407

-- Define the problem parameters
def total_files : ℕ := 42
def disk_capacity : ℚ := 2.88
def large_files : ℕ := 8
def medium_files : ℕ := 16
def large_file_size : ℚ := 1.6
def medium_file_size : ℚ := 1
def small_file_size : ℚ := 0.5

-- Define the function to calculate the minimal number of disks
def minimal_disks : ℕ := sorry

-- State the theorem
theorem minimal_disks_is_16 : minimal_disks = 16 := by sorry

end NUMINAMATH_CALUDE_minimal_disks_is_16_l1104_110407


namespace NUMINAMATH_CALUDE_profit_formula_l1104_110419

/-- Represents the cost and pricing structure of a shop selling bundles -/
structure ShopBundle where
  water_cost : ℚ  -- Cost of water bottle in dollars
  fruit_cost : ℚ  -- Cost of fruit in dollars
  snack_cost : ℚ  -- Cost of snack in dollars (unknown)
  regular_price : ℚ  -- Regular selling price of a bundle
  fifth_bundle_price : ℚ  -- Price of every 5th bundle
  water_per_bundle : ℕ  -- Number of water bottles per bundle
  fruit_per_bundle : ℕ  -- Number of fruits per bundle
  snack_per_bundle : ℕ  -- Number of snacks per regular bundle
  extra_snack : ℕ  -- Extra snacks given in 5th bundle

/-- Calculates the total profit for 5 bundles given the shop's pricing structure -/
def total_profit_five_bundles (shop : ShopBundle) : ℚ :=
  let regular_cost := shop.water_cost * shop.water_per_bundle +
                      shop.fruit_cost * shop.fruit_per_bundle +
                      shop.snack_cost * shop.snack_per_bundle
  let fifth_bundle_cost := shop.water_cost * shop.water_per_bundle +
                           shop.fruit_cost * shop.fruit_per_bundle +
                           shop.snack_cost * (shop.snack_per_bundle + shop.extra_snack)
  let regular_profit := shop.regular_price - regular_cost
  let fifth_bundle_profit := shop.fifth_bundle_price - fifth_bundle_cost
  4 * regular_profit + fifth_bundle_profit

/-- Theorem stating that the total profit for 5 bundles can be expressed as 15.40 - 16S -/
theorem profit_formula (shop : ShopBundle)
  (h1 : shop.water_cost = 0.5)
  (h2 : shop.fruit_cost = 0.25)
  (h3 : shop.regular_price = 4.6)
  (h4 : shop.fifth_bundle_price = 2)
  (h5 : shop.water_per_bundle = 1)
  (h6 : shop.fruit_per_bundle = 2)
  (h7 : shop.snack_per_bundle = 3)
  (h8 : shop.extra_snack = 1) :
  total_profit_five_bundles shop = 15.4 - 16 * shop.snack_cost := by
  sorry

end NUMINAMATH_CALUDE_profit_formula_l1104_110419


namespace NUMINAMATH_CALUDE_complex_3_minus_i_in_fourth_quadrant_l1104_110469

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative -/
def in_fourth_quadrant (z : ℂ) : Prop := 0 < z.re ∧ z.im < 0

/-- The complex number 3 - i is in the fourth quadrant -/
theorem complex_3_minus_i_in_fourth_quadrant : 
  in_fourth_quadrant (3 - I) := by sorry

end NUMINAMATH_CALUDE_complex_3_minus_i_in_fourth_quadrant_l1104_110469


namespace NUMINAMATH_CALUDE_max_planes_for_10_points_l1104_110440

/-- The number of points in space -/
def n : ℕ := 10

/-- The number of points required to determine a plane -/
def k : ℕ := 3

/-- Assumption that no three points are collinear -/
axiom no_collinear : True

/-- The maximum number of planes determined by n points in space -/
def max_planes (n : ℕ) : ℕ := Nat.choose n k

theorem max_planes_for_10_points : max_planes n = 120 := by sorry

end NUMINAMATH_CALUDE_max_planes_for_10_points_l1104_110440


namespace NUMINAMATH_CALUDE_robs_planned_reading_time_l1104_110484

/-- Proves that Rob's planned reading time was 3 hours given the conditions -/
theorem robs_planned_reading_time 
  (pages_read : ℕ) 
  (reading_rate : ℚ)  -- pages per minute
  (actual_time_ratio : ℚ) :
  pages_read = 9 →
  reading_rate = 1 / 15 →
  actual_time_ratio = 3 / 4 →
  (pages_read / reading_rate) / actual_time_ratio / 60 = 3 := by
sorry

end NUMINAMATH_CALUDE_robs_planned_reading_time_l1104_110484


namespace NUMINAMATH_CALUDE_min_spheres_to_cover_unit_cylinder_l1104_110491

/-- Represents a cylinder with given height and base radius -/
structure Cylinder where
  height : ℝ
  baseRadius : ℝ

/-- Represents a sphere with given radius -/
structure Sphere where
  radius : ℝ

/-- Function to determine the minimum number of spheres needed to cover a cylinder -/
def minSpheresToCoverCylinder (c : Cylinder) (s : Sphere) : ℕ :=
  sorry

/-- Theorem stating that a cylinder with height 1 and base radius 1 requires at least 3 unit spheres to cover it -/
theorem min_spheres_to_cover_unit_cylinder :
  let c := Cylinder.mk 1 1
  let s := Sphere.mk 1
  minSpheresToCoverCylinder c s = 3 :=
sorry

end NUMINAMATH_CALUDE_min_spheres_to_cover_unit_cylinder_l1104_110491


namespace NUMINAMATH_CALUDE_matrix_multiplication_l1104_110433

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 4]

theorem matrix_multiplication :
  A * B = !![17, -5; 16, -20] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_l1104_110433


namespace NUMINAMATH_CALUDE_chord_squared_sum_l1104_110477

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit
def O : Point := Unit.unit -- Center of the circle
def A : Point := Unit.unit
def B : Point := Unit.unit
def C : Point := Unit.unit
def D : Point := Unit.unit
def E : Point := Unit.unit

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the distance function
def distance : Point → Point → ℝ := sorry

-- Define the angle function
def angle : Point → Point → Point → ℝ := sorry

-- State the theorem
theorem chord_squared_sum (c : Circle) :
  distance O A = radius ∧
  distance O B = radius ∧
  distance A B = 2 * radius ∧
  distance B E = 3 ∧
  angle A E C = π / 3 →
  (distance C E)^2 + (distance D E)^2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_chord_squared_sum_l1104_110477


namespace NUMINAMATH_CALUDE_inequality_proofs_l1104_110487

theorem inequality_proofs (x : ℝ) :
  (x^2 - x - 2 ≥ 0 ∧ Real.sqrt (x^2 - x - 2) ≤ 2*x → x ≥ 2) ∧
  (x^2 - x - 2 ≥ 0 ∧ Real.sqrt (x^2 - x - 2) ≥ 2*x → x ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l1104_110487


namespace NUMINAMATH_CALUDE_fraction_change_with_addition_l1104_110437

theorem fraction_change_with_addition (a b n : ℕ) (h_b_pos : b > 0) :
  (a / b < 1 → (a + n) / (b + n) > a / b) ∧
  (a / b > 1 → (a + n) / (b + n) < a / b) := by
sorry

end NUMINAMATH_CALUDE_fraction_change_with_addition_l1104_110437


namespace NUMINAMATH_CALUDE_value_of_x_l1104_110448

theorem value_of_x (w u v x : ℤ) 
  (hw : w = 50)
  (hv : v = 3 * w + 30)
  (hu : u = v - 15)
  (hx : x = 2 * u + 12) : 
  x = 342 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1104_110448


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l1104_110444

/-- For a geometric progression with first term b₁, common ratio q, n-th term bₙ, and sum of first n terms Sₙ, 
    the ratio (Sₙ - bₙ) / (Sₙ - b₁) is equal to 1/q for all q -/
theorem geometric_progression_ratio (n : ℕ) (b₁ q : ℝ) : 
  let bₙ := b₁ * q^(n - 1)
  let Sₙ := if q ≠ 1 then b₁ * (q^n - 1) / (q - 1) else n * b₁
  (Sₙ - bₙ) / (Sₙ - b₁) = 1 / q :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l1104_110444


namespace NUMINAMATH_CALUDE_min_packs_for_130_cans_l1104_110454

/-- Represents the number of cans in each pack type -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans for a given pack size -/
def cansInPack (p : PackSize) : ℕ :=
  match p with
  | .small => 8
  | .medium => 15
  | .large => 30

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : ℕ :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a combination -/
def totalPacks (c : PackCombination) : ℕ :=
  c.small + c.medium + c.large

/-- Theorem: The minimum number of packs to buy 130 cans is 6 -/
theorem min_packs_for_130_cans :
  ∃ (c : PackCombination),
    totalCans c = 130 ∧
    totalPacks c = 6 ∧
    (∀ (d : PackCombination), totalCans d = 130 → totalPacks d ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_min_packs_for_130_cans_l1104_110454


namespace NUMINAMATH_CALUDE_conference_handshakes_l1104_110428

/-- The number of handshakes in a conference of n people where each person
    shakes hands exactly once with every other person. -/
def handshakes (n : ℕ) : ℕ := n.choose 2

/-- Theorem stating that in a conference of 10 people where each person
    shakes hands exactly once with every other person, the total number
    of handshakes is 45. -/
theorem conference_handshakes :
  handshakes 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1104_110428


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1104_110470

/-- Given vectors a and b in ℝ³, where a is parallel to b, prove that the magnitude of b is 3√6 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ × ℝ) : 
  a = (-1, 2, 1) → 
  b.1 = 3 → 
  ∃ (k : ℝ), b = k • a → 
  ‖b‖ = 3 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1104_110470


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l1104_110431

theorem coefficient_x_squared_expansion : 
  let f : ℕ → ℕ → ℕ := fun n k => Nat.choose n k
  let g : ℕ → ℤ := fun n => (-1)^n
  (f 3 0) * (f 4 2) + (f 3 1) * (f 4 1) * (g 1) + (f 3 2) * 2^2 * (f 4 0) = -6 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l1104_110431


namespace NUMINAMATH_CALUDE_fixed_tangent_circle_l1104_110468

-- Define the main circle
def main_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the property of the chords
def chord_property (OA OB : ℝ) : Prop := OA * OB = 2

-- Define the tangent circle
def tangent_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Theorem statement
theorem fixed_tangent_circle 
  (O A B : ℝ × ℝ) 
  (hA : main_circle A.1 A.2) 
  (hB : main_circle B.1 B.2)
  (hOA : O = (0, 0))
  (hchord : chord_property (Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2)) 
                           (Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2)))
  (hAB : A ≠ B) :
  ∃ (P : ℝ × ℝ), tangent_circle P.1 P.2 ∧ 
    (P.1 - A.1) * (B.2 - A.2) = (P.2 - A.2) * (B.1 - A.1) :=
sorry

end NUMINAMATH_CALUDE_fixed_tangent_circle_l1104_110468


namespace NUMINAMATH_CALUDE_intersection_points_count_l1104_110472

-- Define the two equations
def equation1 (x y : ℝ) : Prop := (x - 2*y + 3) * (4*x + y - 5) = 0
def equation2 (x y : ℝ) : Prop := (x + 2*y - 3) * (3*x - 4*y + 6) = 0

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define what it means for a point to satisfy both equations
def satisfiesBothEquations (p : Point) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

-- Statement of the theorem
theorem intersection_points_count :
  ∃ (s : Finset Point), (∀ p ∈ s, satisfiesBothEquations p) ∧ s.card = 3 ∧
  (∀ p : Point, satisfiesBothEquations p → p ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l1104_110472


namespace NUMINAMATH_CALUDE_right_angled_triangle_sides_l1104_110480

theorem right_angled_triangle_sides : 
  (∃ (a b c : ℕ), (a = 5 ∧ b = 3 ∧ c = 4) ∧ a^2 = b^2 + c^2) ∧
  (∀ (a b c : ℕ), (a = 2 ∧ b = 3 ∧ c = 4) → a^2 ≠ b^2 + c^2) ∧
  (∀ (a b c : ℕ), (a = 4 ∧ b = 6 ∧ c = 9) → a^2 ≠ b^2 + c^2) ∧
  (∀ (a b c : ℕ), (a = 5 ∧ b = 11 ∧ c = 13) → a^2 ≠ b^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_sides_l1104_110480


namespace NUMINAMATH_CALUDE_triangle_intersection_theorem_l1104_110499

/-- A triangle in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Checks if a triangle is acute-angled -/
def isAcute (t : Triangle) : Prop := sorry

/-- Constructs the next triangle from the given triangle -/
def nextTriangle (t : Triangle) : Triangle := sorry

/-- Counts the number of intersection points between two triangles -/
def intersectionPoints (t1 t2 : Triangle) : ℕ := sorry

/-- The main theorem -/
theorem triangle_intersection_theorem (A₀B₀C₀ : Triangle) (h : isAcute A₀B₀C₀) :
  ∀ n : ℕ, intersectionPoints ((nextTriangle^[n]) A₀B₀C₀) ((nextTriangle^[n+1]) A₀B₀C₀) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_intersection_theorem_l1104_110499


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1104_110483

theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![x, 1]
  let b : Fin 2 → ℝ := ![1, -1]
  (∃ (k : ℝ), a = k • b) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1104_110483


namespace NUMINAMATH_CALUDE_limit_of_sequence_l1104_110438

def a (n : ℕ) : ℚ := (4 * n - 1) / (2 * n + 1)

theorem limit_of_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l1104_110438


namespace NUMINAMATH_CALUDE_hill_height_correct_l1104_110424

/-- The height of the hill in feet -/
def hill_height : ℝ := 900

/-- The uphill speed in feet per second -/
def uphill_speed : ℝ := 9

/-- The downhill speed in feet per second -/
def downhill_speed : ℝ := 12

/-- The total time to run up and down the hill in seconds -/
def total_time : ℝ := 175

/-- Theorem stating that the given hill height satisfies the conditions -/
theorem hill_height_correct : 
  hill_height / uphill_speed + hill_height / downhill_speed = total_time :=
sorry

end NUMINAMATH_CALUDE_hill_height_correct_l1104_110424


namespace NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l1104_110498

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem tangent_parallel_to_x_axis :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ (deriv f x₀ = 0) ∧ (f x₀ = 1 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_x_axis_l1104_110498


namespace NUMINAMATH_CALUDE_impossible_cover_l1104_110457

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangular piece -/
structure Piece :=
  (width : ℕ)
  (height : ℕ)

/-- Checks if a board can be completely covered by pieces without overlapping or sticking out -/
def can_cover (b : Board) (p : Piece) : Prop :=
  ∃ (arrangement : ℕ), 
    (arrangement * p.width * p.height = b.rows * b.cols) ∧ 
    (b.rows % p.height = 0) ∧ 
    (b.cols % p.width = 0)

/-- The main theorem stating that specific boards cannot be covered by specific pieces -/
theorem impossible_cover : 
  ¬(can_cover (Board.mk 6 6) (Piece.mk 1 4)) ∧ 
  ¬(can_cover (Board.mk 12 9) (Piece.mk 2 2)) :=
sorry

end NUMINAMATH_CALUDE_impossible_cover_l1104_110457


namespace NUMINAMATH_CALUDE_chocolate_cost_proof_l1104_110490

def candy_bar_cost : ℝ := 2
def chocolate_cost_difference : ℝ := 1

theorem chocolate_cost_proof :
  let chocolate_cost := candy_bar_cost + chocolate_cost_difference
  chocolate_cost = 3 := by sorry

end NUMINAMATH_CALUDE_chocolate_cost_proof_l1104_110490


namespace NUMINAMATH_CALUDE_candy_inconsistency_l1104_110420

theorem candy_inconsistency :
  ¬∃ (K Y N B : ℕ),
    K + Y + N = 120 ∧
    N + B = 103 ∧
    K + Y + B = 152 :=
by sorry

end NUMINAMATH_CALUDE_candy_inconsistency_l1104_110420


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1104_110408

theorem arithmetic_sequence_middle_term (a b c : ℤ) : 
  (a = 3^2 ∧ c = 3^4 ∧ b - a = c - b) → b = 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l1104_110408


namespace NUMINAMATH_CALUDE_sin_ratio_minus_sqrt3_over_sin_l1104_110443

theorem sin_ratio_minus_sqrt3_over_sin : 
  (Real.sin (80 * π / 180)) / (Real.sin (20 * π / 180)) - 
  (Real.sqrt 3) / (2 * Real.sin (80 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_ratio_minus_sqrt3_over_sin_l1104_110443


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1104_110467

theorem right_triangle_perimeter : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- all sides are positive integers
  b = 4 ∧                   -- one leg measures 4
  a^2 + b^2 = c^2 ∧         -- right-angled triangle (Pythagorean theorem)
  a + b + c = 12            -- perimeter is 12
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1104_110467


namespace NUMINAMATH_CALUDE_f_properties_l1104_110432

noncomputable section

def e : ℝ := Real.exp 1

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < e then -x^3 + x^2 else a * Real.log x

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ ∀ z : ℝ, f a z = 0 → z = x ∨ z = y) ∧
  (∃ M N : ℝ × ℝ,
    let (xM, yM) := M
    let (xN, yN) := N
    xM > 0 ∧ xN < 0 ∧
    yM = f a xM ∧ yN = f a (-xN) ∧
    xM * xN + yM * yN = 0 ∧
    (xM - xN) * yM = xM * (yM - yN) ∧
    0 < a ∧ a ≤ 1 / (e + 1)) ∧
  (∀ a' : ℝ, a' ≤ 0 ∨ a' > 1 / (e + 1) →
    ¬∃ M N : ℝ × ℝ,
      let (xM, yM) := M
      let (xN, yN) := N
      xM > 0 ∧ xN < 0 ∧
      yM = f a' xM ∧ yN = f a' (-xN) ∧
      xM * xN + yM * yN = 0 ∧
      (xM - xN) * yM = xM * (yM - yN)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1104_110432


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l1104_110456

theorem fractional_equation_solution :
  ∃ (x : ℝ), (1 / x = 2 / (x + 3)) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l1104_110456


namespace NUMINAMATH_CALUDE_cookies_per_bag_l1104_110445

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 75) (h2 : num_bags = 25) :
  total_cookies / num_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l1104_110445


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1104_110402

-- Define a triangle with side lengths a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality_ab : a < b + c
  triangle_inequality_bc : b < a + c
  triangle_inequality_ca : c < a + b

-- Define what it means for an angle to be acute in terms of side lengths
def is_angle_A_acute (t : Triangle) : Prop :=
  t.b ^ 2 + t.c ^ 2 > t.a ^ 2

-- Define the condition a ≤ (b + c) / 2
def condition (t : Triangle) : Prop :=
  t.a ≤ (t.b + t.c) / 2

-- Theorem statement
theorem condition_sufficient_not_necessary :
  (∀ t : Triangle, condition t → is_angle_A_acute t) ∧
  ¬(∀ t : Triangle, is_angle_A_acute t → condition t) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1104_110402


namespace NUMINAMATH_CALUDE_basketball_game_theorem_l1104_110452

/-- Represents the scores of a team in a four-quarter basketball game -/
structure GameScores where
  q1 : ℕ
  q2 : ℕ
  q3 : ℕ
  q4 : ℕ

/-- Checks if the given scores form an arithmetic sequence -/
def is_arithmetic (s : GameScores) : Prop :=
  ∃ (a d : ℕ), s.q1 = a ∧ s.q2 = a + d ∧ s.q3 = a + 2*d ∧ s.q4 = a + 3*d

/-- Checks if the given scores form a geometric sequence -/
def is_geometric (s : GameScores) : Prop :=
  ∃ (b r : ℕ), r > 1 ∧ s.q1 = b ∧ s.q2 = b * r ∧ s.q3 = b * r^2 ∧ s.q4 = b * r^3

/-- Calculates the total score for a team -/
def total_score (s : GameScores) : ℕ := s.q1 + s.q2 + s.q3 + s.q4

/-- Calculates the first half score for a team -/
def first_half_score (s : GameScores) : ℕ := s.q1 + s.q2

/-- The main theorem stating the conditions and the result to be proved -/
theorem basketball_game_theorem (team1 team2 : GameScores) : 
  is_arithmetic team1 →
  is_geometric team2 →
  total_score team1 = total_score team2 + 2 →
  total_score team1 ≤ 100 →
  total_score team2 ≤ 100 →
  first_half_score team1 + first_half_score team2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_theorem_l1104_110452


namespace NUMINAMATH_CALUDE_unique_solution_fifth_power_equation_l1104_110405

theorem unique_solution_fifth_power_equation :
  ∃! (x : ℝ), x ≠ 0 ∧ (5 * x)^10 = (10 * x)^5 :=
by
  -- The unique solution is x = 2/5
  use 2/5
  sorry

end NUMINAMATH_CALUDE_unique_solution_fifth_power_equation_l1104_110405


namespace NUMINAMATH_CALUDE_abc_value_for_factored_polynomial_l1104_110458

/-- If a polynomial ax^2 + bx + c can be factored as (x-1)(x-2), then abc = -6 -/
theorem abc_value_for_factored_polynomial (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = (x - 1) * (x - 2)) →
  a * b * c = -6 := by
  sorry

end NUMINAMATH_CALUDE_abc_value_for_factored_polynomial_l1104_110458


namespace NUMINAMATH_CALUDE_sum_in_terms_of_x_l1104_110414

theorem sum_in_terms_of_x (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) :
  x + y + z = 16 * x := by
sorry

end NUMINAMATH_CALUDE_sum_in_terms_of_x_l1104_110414


namespace NUMINAMATH_CALUDE_root_in_interval_l1104_110459

noncomputable def f (x : ℝ) := Real.exp x + x - 2

theorem root_in_interval : ∃ x ∈ Set.Ioo 0 1, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1104_110459


namespace NUMINAMATH_CALUDE_class_grade_average_l1104_110474

theorem class_grade_average (N : ℕ) (X : ℝ) : 
  (X * N + 45 * (2 * N)) / (3 * N) = 48 → X = 54 := by
  sorry

end NUMINAMATH_CALUDE_class_grade_average_l1104_110474


namespace NUMINAMATH_CALUDE_middle_number_is_seven_l1104_110481

/-- Given three consecutive integers where the sums of these integers taken in pairs are 18, 20, and 23, prove that the middle number is 7. -/
theorem middle_number_is_seven (x : ℤ) 
  (h1 : x + (x + 1) = 18) 
  (h2 : x + (x + 2) = 20) 
  (h3 : (x + 1) + (x + 2) = 23) : 
  x + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_is_seven_l1104_110481


namespace NUMINAMATH_CALUDE_ellipse_sum_bound_l1104_110439

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Theorem statement
theorem ellipse_sum_bound :
  ∀ x y : ℝ, ellipse x y → -4 ≤ x + 2*y ∧ x + 2*y ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_sum_bound_l1104_110439


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1104_110413

-- Define the conditions
def p (x : ℝ) : Prop := (x - 1) / (x + 3) ≥ 0
def q (x : ℝ) : Prop := 5 * x - 6 > x^2

-- Define the relationship between ¬p and ¬q
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (¬P → ¬Q) ∧ ¬(¬Q → ¬P)

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  sufficient_not_necessary (∃ x, p x) (∃ x, q x) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l1104_110413


namespace NUMINAMATH_CALUDE_town_population_males_l1104_110476

theorem town_population_males (total_population : ℕ) (num_segments : ℕ) (male_segments : ℕ) :
  total_population = 800 →
  num_segments = 4 →
  male_segments = 1 →
  2 * (total_population / num_segments * male_segments) = total_population →
  total_population / num_segments * male_segments = 400 :=
by sorry

end NUMINAMATH_CALUDE_town_population_males_l1104_110476


namespace NUMINAMATH_CALUDE_ryan_has_30_stickers_l1104_110475

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := 30

/-- The number of stickers Steven has -/
def steven_stickers : ℕ := 3 * ryan_stickers

/-- The number of stickers Terry has -/
def terry_stickers : ℕ := steven_stickers + 20

/-- The total number of stickers -/
def total_stickers : ℕ := 230

theorem ryan_has_30_stickers :
  ryan_stickers + steven_stickers + terry_stickers = total_stickers ∧
  ryan_stickers = 30 := by sorry

end NUMINAMATH_CALUDE_ryan_has_30_stickers_l1104_110475


namespace NUMINAMATH_CALUDE_sum_coordinates_reflection_over_x_axis_l1104_110461

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflect a point over the x-axis -/
def reflectOverXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

/-- Sum of all coordinate values of two points -/
def sumCoordinates (p1 p2 : Point2D) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

/-- Theorem: The sum of coordinates of a point (5, y) and its reflection over x-axis is 10 -/
theorem sum_coordinates_reflection_over_x_axis (y : ℝ) :
  let c : Point2D := { x := 5, y := y }
  let d : Point2D := reflectOverXAxis c
  sumCoordinates c d = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_coordinates_reflection_over_x_axis_l1104_110461


namespace NUMINAMATH_CALUDE_angle_at_intersection_point_l1104_110473

/-- In a 3x3 grid, given points A, B, C, D, and E where AB and CD intersect at E, 
    prove that the angle at E is 45 degrees. -/
theorem angle_at_intersection_point (A B C D E : ℝ × ℝ) : 
  A = (0, 0) → 
  B = (3, 3) → 
  C = (0, 3) → 
  D = (3, 0) → 
  (E.1 - A.1) / (E.2 - A.2) = (B.1 - A.1) / (B.2 - A.2) →  -- E is on line AB
  (E.1 - C.1) / (E.2 - C.2) = (D.1 - C.1) / (D.2 - C.2) →  -- E is on line CD
  Real.arctan ((B.2 - A.2) / (B.1 - A.1) - (D.2 - C.2) / (D.1 - C.1)) / 
    (1 + (B.2 - A.2) / (B.1 - A.1) * (D.2 - C.2) / (D.1 - C.1)) * (180 / Real.pi) = 45 :=
by sorry

end NUMINAMATH_CALUDE_angle_at_intersection_point_l1104_110473


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l1104_110421

/-- The number of sides in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem pentadecagon_triangles : num_triangles = 455 := by
  sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l1104_110421


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l1104_110464

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a corner cube to be removed -/
structure CornerCubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of the modified cube -/
def modifiedCubeSurfaceArea (originalCube : CubeDimensions) (cornerCube : CornerCubeDimensions) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the modified cube is 96 sq.cm -/
theorem modified_cube_surface_area :
  let originalCube : CubeDimensions := ⟨4, 4, 4⟩
  let cornerCube : CornerCubeDimensions := ⟨1, 1, 1⟩
  modifiedCubeSurfaceArea originalCube cornerCube = 96 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l1104_110464


namespace NUMINAMATH_CALUDE_parabola_focus_l1104_110463

/-- The focus of the parabola y² = -8x is at the point (-2, 0) -/
theorem parabola_focus (x y : ℝ) : 
  y^2 = -8*x → (x + 2)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l1104_110463


namespace NUMINAMATH_CALUDE_vector_equality_transitivity_l1104_110429

variable {V : Type*} [AddCommGroup V]

theorem vector_equality_transitivity (a b c : V) :
  a = b → b = c → a = c := by
  sorry

end NUMINAMATH_CALUDE_vector_equality_transitivity_l1104_110429


namespace NUMINAMATH_CALUDE_samuel_coaching_discontinue_date_l1104_110436

/-- Represents a date in a non-leap year -/
structure Date where
  month : Nat
  day : Nat

/-- Calculates the number of days from January 1st to a given date in a non-leap year -/
def daysFromNewYear (d : Date) : Nat :=
  sorry

/-- The date Samuel discontinued coaching -/
def discontinueDate : Date :=
  { month := 11, day := 3 }

theorem samuel_coaching_discontinue_date 
  (totalCost : Nat) 
  (dailyCharge : Nat) 
  (nonLeapYear : Bool) :
  totalCost = 7038 →
  dailyCharge = 23 →
  nonLeapYear = true →
  daysFromNewYear discontinueDate = totalCost / dailyCharge :=
by sorry

end NUMINAMATH_CALUDE_samuel_coaching_discontinue_date_l1104_110436


namespace NUMINAMATH_CALUDE_trigonometric_equation_l1104_110453

theorem trigonometric_equation (x : Real) :
  2 * Real.cos x - 3 * Real.sin x = 2 →
  Real.sin x + 3 * Real.cos x = 3 ∨ Real.sin x + 3 * Real.cos x = -31/13 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l1104_110453
