import Mathlib

namespace NUMINAMATH_CALUDE_quadruple_solution_l3958_395869

theorem quadruple_solution (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a * b * c * d = 1)
  (h2 : a^2012 + 2012 * b = 2012 * c + d^2012)
  (h3 : 2012 * a + b^2012 = c^2012 + 2012 * d) :
  ∃ t : ℝ, t > 0 ∧ a = t ∧ b = 1/t ∧ c = 1/t ∧ d = t :=
sorry

end NUMINAMATH_CALUDE_quadruple_solution_l3958_395869


namespace NUMINAMATH_CALUDE_exact_sunny_days_probability_l3958_395817

def num_days : ℕ := 5
def sunny_prob : ℚ := 2/5
def desired_sunny_days : ℕ := 2

theorem exact_sunny_days_probability :
  (num_days.choose desired_sunny_days : ℚ) * sunny_prob ^ desired_sunny_days * (1 - sunny_prob) ^ (num_days - desired_sunny_days) = 4320/15625 := by
  sorry

end NUMINAMATH_CALUDE_exact_sunny_days_probability_l3958_395817


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l3958_395803

-- Define the type of real-valued functions
def RealFunction := ℝ → ℝ

-- Define the properties of f
def IsStrictlyIncreasing (f : RealFunction) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

def HasInverse (f g : RealFunction) : Prop :=
  (∀ x : ℝ, f (g x) = x) ∧ (∀ x : ℝ, g (f x) = x)

def SatisfiesEquation (f g : RealFunction) : Prop :=
  ∀ x : ℝ, f x + g x = 2 * x

-- Main theorem
theorem unique_function_satisfying_conditions :
  ∃! f : RealFunction,
    IsStrictlyIncreasing f ∧
    (∃ g : RealFunction, HasInverse f g ∧ SatisfiesEquation f g) ∧
    (∀ x : ℝ, f x = x) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l3958_395803


namespace NUMINAMATH_CALUDE_dot_position_after_operations_l3958_395893

-- Define a square
structure Square where
  side : ℝ
  side_pos : side > 0

-- Define a point (for the dot)
structure Point where
  x : ℝ
  y : ℝ

-- Define the operations
def fold_diagonal (s : Square) (p : Point) : Point := sorry

def rotate_90_clockwise (s : Square) (p : Point) : Point := sorry

def unfold (s : Square) (p : Point) : Point := sorry

-- Theorem statement
theorem dot_position_after_operations (s : Square) : 
  let initial_dot : Point := ⟨s.side, s.side⟩
  let folded_dot := fold_diagonal s initial_dot
  let rotated_dot := rotate_90_clockwise s folded_dot
  let final_dot := unfold s rotated_dot
  final_dot.x > s.side / 2 ∧ final_dot.y < s.side / 2 := by sorry

end NUMINAMATH_CALUDE_dot_position_after_operations_l3958_395893


namespace NUMINAMATH_CALUDE_integral_x_squared_zero_to_one_l3958_395874

theorem integral_x_squared_zero_to_one :
  ∫ x in (0 : ℝ)..(1 : ℝ), x^2 = (1 : ℝ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_zero_to_one_l3958_395874


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3958_395804

/-- Given that the solution set of x² - ax - b < 0 is (2, 3), 
    prove that the solution set of bx² - ax - 1 > 0 is (-1/2, -1/3) -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) :
  ∀ x, b*x^2 - a*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3958_395804


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3958_395820

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3958_395820


namespace NUMINAMATH_CALUDE_sum_diagonal_blocks_420_eq_2517_l3958_395819

/-- Given a 420 × 420 square grid tiled with 1 × 2 blocks, this function calculates
    the sum of all possible values for the total number of blocks
    that the two diagonals pass through. -/
def sum_diagonal_blocks_420 : ℕ :=
  let grid_size : ℕ := 420
  let diagonal_squares : ℕ := 2 * grid_size
  let non_center_squares : ℕ := diagonal_squares - 4
  let non_center_blocks : ℕ := non_center_squares
  let min_center_blocks : ℕ := 2
  let max_center_blocks : ℕ := 4
  (non_center_blocks + min_center_blocks) +
  (non_center_blocks + min_center_blocks + 1) +
  (non_center_blocks + max_center_blocks)

theorem sum_diagonal_blocks_420_eq_2517 :
  sum_diagonal_blocks_420 = 2517 := by
  sorry

end NUMINAMATH_CALUDE_sum_diagonal_blocks_420_eq_2517_l3958_395819


namespace NUMINAMATH_CALUDE_pet_fee_calculation_l3958_395805

-- Define the given constants
def daily_rate : ℚ := 125
def stay_duration_days : ℕ := 14
def service_fee_rate : ℚ := 0.2
def security_deposit_rate : ℚ := 0.5
def security_deposit : ℚ := 1110

-- Define the pet fee
def pet_fee : ℚ := 120

-- Theorem statement
theorem pet_fee_calculation :
  let base_cost := daily_rate * stay_duration_days
  let service_fee := service_fee_rate * base_cost
  let total_without_pet_fee := base_cost + service_fee
  let total_with_pet_fee := security_deposit / security_deposit_rate
  total_with_pet_fee - total_without_pet_fee = pet_fee := by
  sorry


end NUMINAMATH_CALUDE_pet_fee_calculation_l3958_395805


namespace NUMINAMATH_CALUDE_train_length_l3958_395860

/-- Calculates the length of a train given its speed and the time and distance it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (5/18) → 
  platform_length = 250 → 
  crossing_time = 30 → 
  (train_speed * crossing_time) - platform_length = 350 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3958_395860


namespace NUMINAMATH_CALUDE_fourth_score_proof_l3958_395810

/-- Given four test scores with an average of 94, where three scores are known to be 85, 100, and 94,
    prove that the fourth score must be 97. -/
theorem fourth_score_proof (score1 score2 score3 score4 : ℕ) : 
  score1 = 85 → score2 = 100 → score3 = 94 → 
  (score1 + score2 + score3 + score4) / 4 = 94 →
  score4 = 97 := by sorry

end NUMINAMATH_CALUDE_fourth_score_proof_l3958_395810


namespace NUMINAMATH_CALUDE_plane_equation_correct_l3958_395816

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Parametric representation of a plane -/
def parametricPlane (s t : ℝ) : Point3D :=
  { x := 2 + 2*s - t
    y := 4 - 2*s
    z := 5 - 3*s + 3*t }

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The plane equation we want to prove -/
def targetPlane : Plane :=
  { a := 2
    b := 1
    c := -1
    d := -3 }

theorem plane_equation_correct :
  ∀ s t : ℝ, pointOnPlane targetPlane (parametricPlane s t) := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l3958_395816


namespace NUMINAMATH_CALUDE_jerry_has_36_stickers_l3958_395898

/-- The number of stickers each person has -/
structure StickerCount where
  fred : ℕ
  george : ℕ
  jerry : ℕ

/-- The conditions of the problem -/
def sticker_problem (s : StickerCount) : Prop :=
  s.fred = 18 ∧
  s.george = s.fred - 6 ∧
  s.jerry = 3 * s.george

/-- The theorem stating Jerry has 36 stickers -/
theorem jerry_has_36_stickers (s : StickerCount) (h : sticker_problem s) : s.jerry = 36 := by
  sorry

end NUMINAMATH_CALUDE_jerry_has_36_stickers_l3958_395898


namespace NUMINAMATH_CALUDE_range_of_g_l3958_395885

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.cos x ^ 4 + Real.sin x ^ 2 ∧ Real.cos x ^ 4 + Real.sin x ^ 2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l3958_395885


namespace NUMINAMATH_CALUDE_functional_equation_implies_linearity_l3958_395807

theorem functional_equation_implies_linearity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x^2 - f x * f y + f y^2)) :
  ∀ x : ℝ, f (2005 * x) = 2005 * f x :=
sorry

end NUMINAMATH_CALUDE_functional_equation_implies_linearity_l3958_395807


namespace NUMINAMATH_CALUDE_triangle_altitude_after_base_extension_l3958_395832

theorem triangle_altitude_after_base_extension (area : ℝ) (new_base : ℝ) (h : area = 800) (h_base : new_base = 50) :
  let new_altitude := 2 * area / new_base
  new_altitude = 32 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_after_base_extension_l3958_395832


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l3958_395851

theorem min_triangles_to_cover (side_large : ℝ) (side_small : ℝ) : 
  side_large = 12 → side_small = 1 → 
  (side_large / side_small) ^ 2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l3958_395851


namespace NUMINAMATH_CALUDE_arccos_one_eq_zero_l3958_395848

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_eq_zero_l3958_395848


namespace NUMINAMATH_CALUDE_parallel_vectors_sin_cos_product_l3958_395847

/-- 
Given two vectors in the plane, a = (4, 3) and b = (sin α, cos α),
prove that if a is parallel to b, then sin α * cos α = 12/25.
-/
theorem parallel_vectors_sin_cos_product (α : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ (4, 3) = k • (Real.sin α, Real.cos α)) → 
  Real.sin α * Real.cos α = 12/25 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sin_cos_product_l3958_395847


namespace NUMINAMATH_CALUDE_no_odd_pieces_all_diagonals_black_squares_count_equivalence_l3958_395818

/-- Represents a chess piece on a chessboard --/
structure ChessPiece where
  position : Nat × Nat
  color : Bool

/-- Represents a chessboard with pieces --/
def Chessboard := List ChessPiece

/-- Represents a diagonal on a chessboard --/
inductive Diagonal
| A1H8 : Nat → Diagonal  -- Diagonals parallel to a1-h8
| A8H1 : Nat → Diagonal  -- Diagonals parallel to a8-h1

/-- Returns the number of pieces on a given diagonal --/
def piecesOnDiagonal (board : Chessboard) (diag : Diagonal) : Nat :=
  sorry

/-- Checks if a number is odd --/
def isOdd (n : Nat) : Bool :=
  n % 2 = 1

/-- Main theorem: It's impossible to have an odd number of pieces on all 30 diagonals --/
theorem no_odd_pieces_all_diagonals (board : Chessboard) : 
  ¬(∀ (d : Diagonal), isOdd (piecesOnDiagonal board d)) :=
by
  sorry

/-- Helper function to count pieces on black squares along a1-h8 diagonals --/
def countBlackSquaresA1H8 (board : Chessboard) : Nat :=
  sorry

/-- Helper function to count pieces on black squares along a8-h1 diagonals --/
def countBlackSquaresA8H1 (board : Chessboard) : Nat :=
  sorry

/-- Theorem: The two ways of counting pieces on black squares are equivalent --/
theorem black_squares_count_equivalence (board : Chessboard) :
  countBlackSquaresA1H8 board = countBlackSquaresA8H1 board :=
by
  sorry

end NUMINAMATH_CALUDE_no_odd_pieces_all_diagonals_black_squares_count_equivalence_l3958_395818


namespace NUMINAMATH_CALUDE_trajectory_and_m_value_l3958_395886

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 7

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 7

-- Define the line that intersects C
def intersecting_line (x y m : ℝ) : Prop := x + y - m = 0

-- Define the property that a circle passes through the origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem trajectory_and_m_value :
  ∀ (x₀ y₀ x y x₁ y₁ x₂ y₂ m : ℝ),
  (3/2, 0) = ((x₀ + x)/2, (y₀ + y)/2) →  -- A is midpoint of BM
  circle_O x₀ y₀ →  -- B is on circle O
  trajectory_C x y →  -- M is on trajectory C
  intersecting_line x₁ y₁ m →  -- P is on the intersecting line
  intersecting_line x₂ y₂ m →  -- Q is on the intersecting line
  trajectory_C x₁ y₁ →  -- P is on trajectory C
  trajectory_C x₂ y₂ →  -- Q is on trajectory C
  circle_through_origin x₁ y₁ x₂ y₂ →  -- Circle with PQ as diameter passes through origin
  (∀ x y, trajectory_C x y ↔ (x - 3)^2 + y^2 = 7) ∧  -- Trajectory equation is correct
  (m = 1 ∨ m = 2)  -- m value is correct
  := by sorry

end NUMINAMATH_CALUDE_trajectory_and_m_value_l3958_395886


namespace NUMINAMATH_CALUDE_car_tire_usage_l3958_395815

/-- Represents the usage of tires on a car -/
structure TireUsage where
  total_tires : ℕ
  active_tires : ℕ
  total_miles : ℕ
  miles_per_tire : ℕ

/-- Calculates the miles each tire is used given the total miles driven and number of tires -/
def calculate_miles_per_tire (usage : TireUsage) : Prop :=
  usage.miles_per_tire = usage.total_miles * usage.active_tires / usage.total_tires

/-- Theorem stating that for a car with 5 tires, 4 of which are used at any time, 
    each tire is used for 40,000 miles over a total of 50,000 miles driven -/
theorem car_tire_usage :
  ∀ (usage : TireUsage), 
    usage.total_tires = 5 →
    usage.active_tires = 4 →
    usage.total_miles = 50000 →
    calculate_miles_per_tire usage →
    usage.miles_per_tire = 40000 :=
sorry

end NUMINAMATH_CALUDE_car_tire_usage_l3958_395815


namespace NUMINAMATH_CALUDE_product_change_l3958_395849

theorem product_change (a b : ℝ) (h : (a - 3) * (b + 3) - a * b = 900) : 
  a * b - (a + 3) * (b - 3) = 918 := by
sorry

end NUMINAMATH_CALUDE_product_change_l3958_395849


namespace NUMINAMATH_CALUDE_min_sum_squares_l3958_395854

/-- B-neighborhood of A is defined as the solution set of |x-A| < B -/
def neighborhood (A B : ℝ) := {x : ℝ | |x - A| < B}

theorem min_sum_squares (a b : ℝ) : 
  neighborhood (a + b - 3) (a + b) = Set.Ioo (-3 : ℝ) 3 → 
  ∃ (min : ℝ), min = 9/2 ∧ ∀ x y : ℝ, x^2 + y^2 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3958_395854


namespace NUMINAMATH_CALUDE_saltwater_volume_l3958_395843

/-- Proves that the initial volume of a saltwater solution is 160 gallons --/
theorem saltwater_volume : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (0.20 * x / x = 1/5) ∧ 
  ((0.20 * x + 16) / (3/4 * x + 24) = 1/3) ∧ 
  (x = 160) := by
  sorry

end NUMINAMATH_CALUDE_saltwater_volume_l3958_395843


namespace NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l3958_395812

/-- A conic section defined by the equation x^2 + my^2 = 1 -/
structure Conic (m : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2 + m*y^2 = 1

/-- Definition of a hyperbola -/
def IsHyperbola (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ↔ x^2 + m*y^2 = 1

/-- Theorem: The equation x^2 + my^2 = 1 represents a hyperbola if and only if m < 0 -/
theorem hyperbola_iff_m_negative (m : ℝ) : IsHyperbola m ↔ m < 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l3958_395812


namespace NUMINAMATH_CALUDE_construction_materials_cost_l3958_395878

/-- The total cost of construction materials for Mr. Zander -/
def total_cost (cement_bags : ℕ) (cement_price : ℕ) (sand_lorries : ℕ) (sand_tons_per_lorry : ℕ) (sand_price : ℕ) : ℕ :=
  cement_bags * cement_price + sand_lorries * sand_tons_per_lorry * sand_price

/-- Theorem stating that the total cost of construction materials for Mr. Zander is $13,000 -/
theorem construction_materials_cost :
  total_cost 500 10 20 10 40 = 13000 := by
  sorry

end NUMINAMATH_CALUDE_construction_materials_cost_l3958_395878


namespace NUMINAMATH_CALUDE_least_four_digit_divisible_l3958_395865

/-- A function that checks if a number has all different digits -/
def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- A function that checks if a number is divisible by another number -/
def is_divisible_by (n m : ℕ) : Prop :=
  n % m = 0

theorem least_four_digit_divisible :
  ∀ n : ℕ,
    1000 ≤ n                          -- four-digit number
    → n < 10000                       -- four-digit number
    → has_different_digits n          -- all digits are different
    → is_divisible_by n 1             -- divisible by 1
    → is_divisible_by n 2             -- divisible by 2
    → is_divisible_by n 4             -- divisible by 4
    → is_divisible_by n 8             -- divisible by 8
    → 1248 ≤ n                        -- 1248 is the least such number
  := by sorry

end NUMINAMATH_CALUDE_least_four_digit_divisible_l3958_395865


namespace NUMINAMATH_CALUDE_expression_factorization_l3958_395883

theorem expression_factorization (x : ℝ) :
  (4 * x^3 - 64 * x^2 + 52) - (-3 * x^3 - 2 * x^2 + 52) = x^2 * (7 * x - 62) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3958_395883


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3958_395825

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 = 2 → a 2 + a 5 = 13 → a 5 + a 6 + a 7 = 33 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3958_395825


namespace NUMINAMATH_CALUDE_no_dapper_numbers_l3958_395841

/-- A two-digit positive integer is 'dapper' if it equals the sum of its nonzero tens digit and the cube of its units digit. -/
def is_dapper (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ (a b : ℕ), n = 10 * a + b ∧ a ≠ 0 ∧ n = a + b^3

/-- There are no two-digit positive integers that are 'dapper'. -/
theorem no_dapper_numbers : ¬∃ (n : ℕ), is_dapper n := by
  sorry

#check no_dapper_numbers

end NUMINAMATH_CALUDE_no_dapper_numbers_l3958_395841


namespace NUMINAMATH_CALUDE_max_type_a_mascots_l3958_395838

/-- Represents the mascot types -/
inductive MascotType
| A
| B

/-- Represents the mascot purchase scenario -/
structure MascotPurchase where
  totalMascots : ℕ
  totalCost : ℕ
  unitPriceA : ℕ
  unitPriceB : ℕ
  newBudget : ℕ
  newTotalMascots : ℕ

/-- Conditions for the mascot purchase -/
def validMascotPurchase (mp : MascotPurchase) : Prop :=
  mp.totalMascots = 110 ∧
  mp.totalCost = 6000 ∧
  mp.unitPriceA = (6 * mp.unitPriceB) / 5 ∧
  mp.totalCost = mp.totalMascots / 2 * (mp.unitPriceA + mp.unitPriceB) ∧
  mp.newBudget = 16800 ∧
  mp.newTotalMascots = 300

/-- Theorem: The maximum number of type A mascots that can be purchased in the second round is 180 -/
theorem max_type_a_mascots (mp : MascotPurchase) (h : validMascotPurchase mp) :
  ∀ n : ℕ, n ≤ mp.newTotalMascots → n * mp.unitPriceA + (mp.newTotalMascots - n) * mp.unitPriceB ≤ mp.newBudget →
  n ≤ 180 :=
sorry

end NUMINAMATH_CALUDE_max_type_a_mascots_l3958_395838


namespace NUMINAMATH_CALUDE_remainder_13_pow_2011_mod_100_l3958_395864

theorem remainder_13_pow_2011_mod_100 : 13^2011 % 100 = 37 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13_pow_2011_mod_100_l3958_395864


namespace NUMINAMATH_CALUDE_sum_of_roots_is_18_l3958_395868

def is_symmetric_about_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 + x) = f (3 - x)

def has_six_distinct_roots_in_arithmetic_sequence (f : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ) (d : ℝ),
    r₁ < r₂ ∧ r₂ < r₃ ∧ r₃ < r₄ ∧ r₄ < r₅ ∧ r₅ < r₆ ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ f r₄ = 0 ∧ f r₅ = 0 ∧ f r₆ = 0 ∧
    r₂ - r₁ = d ∧ r₃ - r₂ = d ∧ r₄ - r₃ = d ∧ r₅ - r₄ = d ∧ r₆ - r₅ = d

theorem sum_of_roots_is_18 (f : ℝ → ℝ) 
    (h_sym : is_symmetric_about_3 f)
    (h_roots : has_six_distinct_roots_in_arithmetic_sequence f) :
    ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ),
      f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ f r₄ = 0 ∧ f r₅ = 0 ∧ f r₆ = 0 ∧
      r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_18_l3958_395868


namespace NUMINAMATH_CALUDE_pairing_probability_l3958_395813

/-- Represents a student in the classroom -/
structure Student :=
  (name : String)

/-- The probability of a specific event occurring in a random pairing scenario -/
def probability_of_pairing (total_students : ℕ) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / (total_students - 1)

/-- The classroom setup -/
def classroom_setup : Prop :=
  ∃ (students : Finset Student) (margo irma jess kurt : Student),
    students.card = 50 ∧
    margo ∈ students ∧
    irma ∈ students ∧
    jess ∈ students ∧
    kurt ∈ students ∧
    margo ≠ irma ∧ margo ≠ jess ∧ margo ≠ kurt

theorem pairing_probability (h : classroom_setup) :
  probability_of_pairing 50 3 = 3 / 49 := by
  sorry

end NUMINAMATH_CALUDE_pairing_probability_l3958_395813


namespace NUMINAMATH_CALUDE_unique_favorite_number_l3958_395870

def is_favorite_number (n : ℕ) : Prop :=
  80 < n ∧ n ≤ 130 ∧
  n % 13 = 0 ∧
  n % 3 ≠ 0 ∧
  (n / 100 + (n / 10) % 10 + n % 10) % 4 = 0

theorem unique_favorite_number : ∃! n, is_favorite_number n :=
  sorry

end NUMINAMATH_CALUDE_unique_favorite_number_l3958_395870


namespace NUMINAMATH_CALUDE_ice_cream_consumption_l3958_395897

theorem ice_cream_consumption (friday : ℝ) (saturday : ℝ) (sunday : ℝ) (monday : ℝ) :
  friday = 3.25 →
  saturday = 2.5 →
  sunday = 1.75 →
  monday = 0.5 →
  friday + saturday + sunday + monday + (2 * monday) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l3958_395897


namespace NUMINAMATH_CALUDE_square_root_16_l3958_395830

theorem square_root_16 (x : ℝ) : (x + 1)^2 = 16 → x = 3 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_16_l3958_395830


namespace NUMINAMATH_CALUDE_quadratic_rational_root_parity_l3958_395863

theorem quadratic_rational_root_parity (a b c : ℤ) (h_a : a ≠ 0) :
  (∃ (p q : ℤ), q ≠ 0 ∧ a * (p / q)^2 + b * (p / q) + c = 0) →
  (Even b ∨ Even c) →
  ¬(Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_parity_l3958_395863


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l3958_395806

theorem sine_cosine_inequality (n : ℕ+) (x : ℝ) : 
  (Real.sin (2 * x))^(n : ℝ) + (Real.sin x^(n : ℝ) - Real.cos x^(n : ℝ))^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l3958_395806


namespace NUMINAMATH_CALUDE_distinct_prime_factors_30_factorial_l3958_395859

/-- The number of distinct prime factors of 30! -/
def num_distinct_prime_factors_30_factorial : ℕ := 10

/-- Theorem stating that the number of distinct prime factors of 30! is 10 -/
theorem distinct_prime_factors_30_factorial :
  num_distinct_prime_factors_30_factorial = 10 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_30_factorial_l3958_395859


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3958_395881

theorem sum_of_fractions : 
  (1 : ℚ) / 3 + (1 : ℚ) / 2 + (-5 : ℚ) / 6 + (1 : ℚ) / 5 + (1 : ℚ) / 4 + (-9 : ℚ) / 20 + (-9 : ℚ) / 20 = (-9 : ℚ) / 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3958_395881


namespace NUMINAMATH_CALUDE_farm_field_problem_l3958_395861

/-- Represents the problem of calculating the farm field area and initial work plan --/
theorem farm_field_problem (planned_daily_rate : ℕ) (actual_daily_rate : ℕ) (extra_days : ℕ) (area_left : ℕ) 
  (h1 : planned_daily_rate = 90)
  (h2 : actual_daily_rate = 85)
  (h3 : extra_days = 2)
  (h4 : area_left = 40) :
  ∃ (total_area : ℕ) (initial_days : ℕ),
    total_area = 3780 ∧ 
    initial_days = 42 ∧
    planned_daily_rate * initial_days = total_area ∧
    actual_daily_rate * (initial_days + extra_days) + area_left = total_area :=
by
  sorry

end NUMINAMATH_CALUDE_farm_field_problem_l3958_395861


namespace NUMINAMATH_CALUDE_trig_inequality_l3958_395892

theorem trig_inequality (θ : Real) (h1 : 0 < θ) (h2 : θ < Real.pi / 4) :
  Real.sin θ ^ 2 < Real.cos θ ^ 2 ∧ Real.cos θ ^ 2 < (Real.cos θ / Real.sin θ) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_inequality_l3958_395892


namespace NUMINAMATH_CALUDE_exponential_sum_conjugate_l3958_395802

theorem exponential_sum_conjugate (α β : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) = -1/3 + 5/8 * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) = -1/3 - 5/8 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_exponential_sum_conjugate_l3958_395802


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_two_l3958_395894

/-- Given two nonconstant geometric sequences with different common ratios,
    if a specific condition is met, then the sum of their common ratios is 2. -/
theorem sum_of_common_ratios_is_two
  (k : ℝ)
  (a₂ a₃ b₂ b₃ : ℝ)
  (ha : a₂ ≠ k ∧ a₃ ≠ a₂)  -- First sequence is nonconstant
  (hb : b₂ ≠ k ∧ b₃ ≠ b₂)  -- Second sequence is nonconstant
  (hseq₁ : ∃ p : ℝ, p ≠ 1 ∧ a₂ = k * p ∧ a₃ = k * p^2)  -- First sequence is geometric
  (hseq₂ : ∃ r : ℝ, r ≠ 1 ∧ b₂ = k * r ∧ b₃ = k * r^2)  -- Second sequence is geometric
  (hdiff : ∀ p r, (a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2) → p ≠ r)  -- Different common ratios
  (hcond : a₃ - b₃ = 2 * (a₂ - b₂))  -- Given condition
  : ∃ p r : ℝ, (a₂ = k * p ∧ a₃ = k * p^2 ∧ b₂ = k * r ∧ b₃ = k * r^2) ∧ p + r = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_two_l3958_395894


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3958_395882

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k + 1) * x^2 + 4 * x - 1 = 0) ↔ (k ≥ -5 ∧ k ≠ -1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3958_395882


namespace NUMINAMATH_CALUDE_trisha_walk_distance_l3958_395888

/-- The total distance Trisha walked during her vacation in New York City -/
def total_distance (hotel_to_postcard postcard_to_tshirt tshirt_to_hotel : ℝ) : ℝ :=
  hotel_to_postcard + postcard_to_tshirt + tshirt_to_hotel

/-- Theorem stating that Trisha walked 0.89 miles in total -/
theorem trisha_walk_distance :
  total_distance 0.11 0.11 0.67 = 0.89 := by
  sorry

end NUMINAMATH_CALUDE_trisha_walk_distance_l3958_395888


namespace NUMINAMATH_CALUDE_height_equality_l3958_395866

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  p : ℝ -- semiperimeter
  ha : ℝ -- height corresponding to side a

-- State the theorem
theorem height_equality (t : Triangle) : 
  t.ha = (2 * (t.p - t.a) * Real.cos (t.β / 2) * Real.cos (t.γ / 2)) / Real.cos (t.α / 2) ∧
  t.ha = (2 * (t.p - t.b) * Real.sin (t.β / 2) * Real.cos (t.γ / 2)) / Real.sin (t.α / 2) := by
  sorry

end NUMINAMATH_CALUDE_height_equality_l3958_395866


namespace NUMINAMATH_CALUDE_final_mango_distribution_l3958_395856

/-- Represents the state of mango distribution among friends in a circle. -/
structure MangoDistribution :=
  (friends : ℕ)
  (mangos : List ℕ)

/-- Defines the rules for sharing mangos. -/
def share (d : MangoDistribution) : MangoDistribution :=
  sorry

/-- Defines the rules for eating mangos. -/
def eat (d : MangoDistribution) : MangoDistribution :=
  sorry

/-- Checks if any further actions (sharing or eating) are possible. -/
def canContinue (d : MangoDistribution) : Bool :=
  sorry

/-- Applies sharing and eating rules until no further actions are possible. -/
def applyRulesUntilStable (d : MangoDistribution) : MangoDistribution :=
  sorry

/-- Counts the number of people with mangos in the final distribution. -/
def countPeopleWithMangos (d : MangoDistribution) : ℕ :=
  sorry

/-- Main theorem stating that exactly 8 people will have mangos at the end. -/
theorem final_mango_distribution
  (initial : MangoDistribution)
  (h1 : initial.friends = 100)
  (h2 : initial.mangos = [2019] ++ List.replicate 99 0) :
  countPeopleWithMangos (applyRulesUntilStable initial) = 8 :=
sorry

end NUMINAMATH_CALUDE_final_mango_distribution_l3958_395856


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3958_395828

theorem geometric_sequence_fourth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 1024) 
  (h2 : a * r^5 = 32) : 
  a * r^3 = 128 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l3958_395828


namespace NUMINAMATH_CALUDE_distribute_5_3_l3958_395880

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there are 5 ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l3958_395880


namespace NUMINAMATH_CALUDE_sum_of_divisor_and_quotient_l3958_395831

/-- Given a valid vertical division, prove that the sum of the divisor and quotient is 723. -/
theorem sum_of_divisor_and_quotient : 
  ∀ (D Q : ℕ), 
  (D = 581) →  -- Divisor condition
  (Q = 142) →  -- Quotient condition
  (D + Q = 723) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisor_and_quotient_l3958_395831


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3958_395887

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) = 140 * n → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3958_395887


namespace NUMINAMATH_CALUDE_divisible_by_30_l3958_395827

theorem divisible_by_30 (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_30_l3958_395827


namespace NUMINAMATH_CALUDE_gracies_height_l3958_395808

/-- Given the heights of Griffin, Grayson, and Gracie, prove Gracie's height -/
theorem gracies_height 
  (griffin_height : ℕ) 
  (grayson_height : ℕ) 
  (gracie_height : ℕ)
  (h1 : griffin_height = 61)
  (h2 : grayson_height = griffin_height + 2)
  (h3 : gracie_height = grayson_height - 7) : 
  gracie_height = 56 := by sorry

end NUMINAMATH_CALUDE_gracies_height_l3958_395808


namespace NUMINAMATH_CALUDE_triangle_inequalities_l3958_395835

theorem triangle_inequalities (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p := (a + b + c) / 2
  (2 * (a + b + c) * (a^2 + b^2 + c^2) ≥ 3 * (a^3 + b^3 + c^3 + 3 * a * b * c)) ∧
  ((a + b + c)^3 ≤ 5 * (b * c * (b + c) + c * a * (c + a) + a * b * (a + b)) - 3 * a * b * c) ∧
  (a * b * c < a^2 * (p - a) + b^2 * (p - b) + c^2 * (p - c) ∧
    a^2 * (p - a) + b^2 * (p - b) + c^2 * (p - c) ≤ 3/2 * a * b * c) ∧
  (1 < Real.cos (π - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) +
       Real.cos (π - Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) +
       Real.cos (π - Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) ∧
   Real.cos (π - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) +
       Real.cos (π - Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) +
       Real.cos (π - Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) ≤ 3/2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l3958_395835


namespace NUMINAMATH_CALUDE_division_problem_l3958_395840

theorem division_problem : (786 * 74) / 30 = 1938.8 := by sorry

end NUMINAMATH_CALUDE_division_problem_l3958_395840


namespace NUMINAMATH_CALUDE_inner_prism_volume_l3958_395884

theorem inner_prism_volume (w l h : ℕ+) : 
  w * l * h = 128 ↔ (w : ℕ) * (l : ℕ) * (h : ℕ) = 128 := by sorry

end NUMINAMATH_CALUDE_inner_prism_volume_l3958_395884


namespace NUMINAMATH_CALUDE_count_valid_numbers_l3958_395899

/-- The number of ways to choose 2 items from 3 -/
def choose_3_2 : ℕ := 3

/-- The number of ways to arrange 2 items -/
def arrange_2 : ℕ := 2

/-- The number of ways to insert 2 items into 3 gaps -/
def insert_2_into_3 : ℕ := 6

/-- The number of valid five-digit numbers -/
def valid_numbers : ℕ := choose_3_2 * arrange_2 * arrange_2 * insert_2_into_3

theorem count_valid_numbers : valid_numbers = 72 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l3958_395899


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3958_395822

theorem simple_interest_problem (interest : ℚ) (rate : ℚ) (time : ℚ) (principal : ℚ) : 
  interest = 4016.25 →
  rate = 9 / 100 →
  time = 5 →
  principal = 8925 →
  interest = principal * rate * time :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3958_395822


namespace NUMINAMATH_CALUDE_vanessa_score_l3958_395800

/-- Calculates Vanessa's score in a basketball game -/
theorem vanessa_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) : 
  total_score = 72 → other_players = 7 → avg_score = 6 →
  total_score - (other_players * avg_score) = 30 := by
sorry

end NUMINAMATH_CALUDE_vanessa_score_l3958_395800


namespace NUMINAMATH_CALUDE_hyperbola_tangent_equation_l3958_395858

-- Define the ellipse
def is_ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the hyperbola
def is_hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the tangent line for the ellipse
def is_ellipse_tangent (x y x₀ y₀ a b : ℝ) : Prop := (x₀ * x / a^2) + (y₀ * y / b^2) = 1

-- Define the tangent line for the hyperbola
def is_hyperbola_tangent (x y x₀ y₀ a b : ℝ) : Prop := (x₀ * x / a^2) - (y₀ * y / b^2) = 1

-- State the theorem
theorem hyperbola_tangent_equation (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_point : is_hyperbola x₀ y₀ a b) :
  is_hyperbola_tangent x y x₀ y₀ a b :=
sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_equation_l3958_395858


namespace NUMINAMATH_CALUDE_triangle_inequality_check_l3958_395836

theorem triangle_inequality_check (rods : Fin 100 → ℝ) 
  (h_sorted : ∀ i j : Fin 100, i ≤ j → rods i ≤ rods j) :
  (∀ i j k : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    rods i + rods j > rods k) ↔ 
  (rods 98 + rods 99 > rods 100) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_check_l3958_395836


namespace NUMINAMATH_CALUDE_addition_preserves_inequality_l3958_395814

theorem addition_preserves_inequality (a b c d : ℝ) : a < b → c < d → a + c < b + d := by
  sorry

end NUMINAMATH_CALUDE_addition_preserves_inequality_l3958_395814


namespace NUMINAMATH_CALUDE_name_length_difference_l3958_395811

/-- Given that Elida has 5 letters in her name and the total of 10 times the average number
    of letters in both names is 65, prove that Adrianna's name has 3 more letters than Elida's name. -/
theorem name_length_difference (elida_length : ℕ) (adrianna_length : ℕ) : 
  elida_length = 5 →
  10 * ((elida_length + adrianna_length) / 2) = 65 →
  adrianna_length = elida_length + 3 := by
sorry


end NUMINAMATH_CALUDE_name_length_difference_l3958_395811


namespace NUMINAMATH_CALUDE_circle_configuration_theorem_l3958_395823

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of three circles C, D, and E -/
structure CircleConfiguration where
  C : Circle
  D : Circle
  E : Circle
  C_radius_is_4 : C.radius = 4
  D_internally_tangent_to_C : True  -- This is a placeholder for the tangency condition
  E_internally_tangent_to_C : True  -- This is a placeholder for the tangency condition
  E_externally_tangent_to_D : True  -- This is a placeholder for the tangency condition
  E_tangent_to_diameter : True      -- This is a placeholder for the tangency condition
  D_radius_twice_E : D.radius = 2 * E.radius

theorem circle_configuration_theorem (config : CircleConfiguration) 
  (p q : ℕ) (h : config.D.radius = Real.sqrt p - q) : 
  p + q = 259 := by
  sorry

end NUMINAMATH_CALUDE_circle_configuration_theorem_l3958_395823


namespace NUMINAMATH_CALUDE_abcdefg_over_defghij_l3958_395876

theorem abcdefg_over_defghij (a b c d e f g h i j : ℚ)
  (h1 : a / b = -7 / 3)
  (h2 : b / c = -5 / 2)
  (h3 : c / d = 2)
  (h4 : d / e = -3 / 2)
  (h5 : e / f = 4 / 3)
  (h6 : f / g = -1 / 4)
  (h7 : g / h = 3 / -5)
  (h8 : i ≠ 0) -- Additional hypothesis to avoid division by zero
  : a * b * c * d * e * f * g / (d * e * f * g * h * i * j) = (-21 / 16) * (c / i) := by
  sorry

end NUMINAMATH_CALUDE_abcdefg_over_defghij_l3958_395876


namespace NUMINAMATH_CALUDE_coin_toss_probability_l3958_395829

/-- The number of coin tosses -/
def n : ℕ := 5

/-- The number of heads -/
def k : ℕ := 4

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def binomial_probability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1/2)^n

theorem coin_toss_probability :
  binomial_probability n k = 5/32 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l3958_395829


namespace NUMINAMATH_CALUDE_meat_for_hamburgers_l3958_395824

/-- Given that Rachelle uses 4 pounds of meat to make 10 hamburgers,
    prove that she needs 12 pounds of meat to make 30 hamburgers. -/
theorem meat_for_hamburgers (meat_for_10 : ℝ) (hamburgers_for_10 : ℕ)
    (meat_for_30 : ℝ) (hamburgers_for_30 : ℕ) :
    meat_for_10 = 4 ∧ hamburgers_for_10 = 10 ∧ hamburgers_for_30 = 30 →
    meat_for_30 = 12 := by
  sorry

end NUMINAMATH_CALUDE_meat_for_hamburgers_l3958_395824


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_values_l3958_395839

theorem quadratic_root_implies_a_values (a : ℝ) : 
  ((-2)^2 + (3/2) * a * (-2) - a^2 = 0) → (a = 1 ∨ a = -4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_values_l3958_395839


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l3958_395801

/-- A function that returns the number of divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with 8 zeros -/
def ends_with_8_zeros (n : ℕ) : Prop := sorry

/-- The theorem to be proved -/
theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ≠ b ∧
    ends_with_8_zeros a ∧
    ends_with_8_zeros b ∧
    num_divisors a = 90 ∧
    num_divisors b = 90 ∧
    a + b = 700000000 := by sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l3958_395801


namespace NUMINAMATH_CALUDE_inequality_solution_l3958_395873

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3958_395873


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l3958_395845

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem f_strictly_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l3958_395845


namespace NUMINAMATH_CALUDE_smallest_divisible_by_10_11_12_l3958_395875

theorem smallest_divisible_by_10_11_12 : ∃ (n : ℕ), n > 0 ∧ 
  10 ∣ n ∧ 11 ∣ n ∧ 12 ∣ n ∧ 
  ∀ (m : ℕ), m > 0 ∧ 10 ∣ m ∧ 11 ∣ m ∧ 12 ∣ m → n ≤ m :=
by
  use 660
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_10_11_12_l3958_395875


namespace NUMINAMATH_CALUDE_sum_of_distances_l3958_395834

/-- Given points A, B, and D in a coordinate plane, prove that the sum of distances AD and BD is 2√5 + √130 -/
theorem sum_of_distances (A B D : ℝ × ℝ) : 
  A = (15, 0) → B = (0, 5) → D = (4, 3) → 
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 2 * Real.sqrt 5 + Real.sqrt 130 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_l3958_395834


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_specific_ellipse_eccentricity_l3958_395855

/-- The eccentricity of an ellipse with equation x²/a² + y²/b² = 1 is √(1 - b²/a²) -/
theorem ellipse_eccentricity (a b : ℝ) (h : 0 < b ∧ b < a) :
  let e := Real.sqrt (1 - b^2 / a^2)
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  e = Real.sqrt (1 - b^2 / a^2) :=
sorry

/-- The eccentricity of the ellipse x²/9 + y² = 1 is 2√2/3 -/
theorem specific_ellipse_eccentricity :
  let e := Real.sqrt (1 - 1^2 / 3^2)
  (∀ x y : ℝ, x^2 / 9 + y^2 = 1) →
  e = 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_specific_ellipse_eccentricity_l3958_395855


namespace NUMINAMATH_CALUDE_maria_evan_age_sum_maria_evan_age_sum_proof_l3958_395889

theorem maria_evan_age_sum : ℕ → ℕ → Prop :=
  fun maria_age evan_age =>
    (maria_age = evan_age + 7) →
    (maria_age + 10 = 3 * (evan_age - 5)) →
    (maria_age + evan_age = 39)

-- The proof is omitted
theorem maria_evan_age_sum_proof : ∃ (maria_age evan_age : ℕ), maria_evan_age_sum maria_age evan_age :=
  sorry

end NUMINAMATH_CALUDE_maria_evan_age_sum_maria_evan_age_sum_proof_l3958_395889


namespace NUMINAMATH_CALUDE_two_identical_digits_in_2_pow_30_l3958_395867

theorem two_identical_digits_in_2_pow_30 :
  ∃ (d : ℕ) (i j : ℕ), i ≠ j ∧ i < 10 ∧ j < 10 ∧
  (2^30 / 10^i) % 10 = d ∧ (2^30 / 10^j) % 10 = d :=
by
  have h1 : 2^30 > 10^9 := sorry
  have h2 : 2^30 < 8 * 10^9 := sorry
  have pigeonhole : ∀ (n m : ℕ), n > m → 
    ∃ (k : ℕ), k < n ∧ (∃ (i j : ℕ), i < m ∧ j < m ∧ i ≠ j ∧
    (n / 10^i) % 10 = k ∧ (n / 10^j) % 10 = k) := sorry
  sorry


end NUMINAMATH_CALUDE_two_identical_digits_in_2_pow_30_l3958_395867


namespace NUMINAMATH_CALUDE_vector_addition_theorem_l3958_395877

/-- Given vectors a and b, prove that 2a + b equals the specified result -/
theorem vector_addition_theorem (a b : ℝ × ℝ × ℝ) :
  a = (1, 2, -3) →
  b = (5, -7, 8) →
  (2 : ℝ) • a + b = (7, -3, 2) := by sorry

end NUMINAMATH_CALUDE_vector_addition_theorem_l3958_395877


namespace NUMINAMATH_CALUDE_sqrt_x_plus_five_equals_two_l3958_395852

theorem sqrt_x_plus_five_equals_two (x : ℝ) (h : x = -1) : Real.sqrt (x + 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_five_equals_two_l3958_395852


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3958_395833

theorem equal_roots_quadratic (p : ℝ) : 
  (∃! p, ∀ x, x^2 - (p+1)*x + p = 0 → (∃! r, x = r)) := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3958_395833


namespace NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l3958_395842

theorem midpoint_x_coordinate_sum (a b c : ℝ) :
  let vertex_sum := a + b + c
  let midpoint1 := (a + b) / 2
  let midpoint2 := (a + c) / 2
  let midpoint3 := (b + c) / 2
  midpoint1 + midpoint2 + midpoint3 = vertex_sum := by
sorry

end NUMINAMATH_CALUDE_midpoint_x_coordinate_sum_l3958_395842


namespace NUMINAMATH_CALUDE_p_difference_qr_l3958_395871

theorem p_difference_qr (p q r : ℕ) : 
  p = 56 → 
  q = p / 8 →
  r = p / 8 →
  p - (q + r) = 42 := by
sorry

end NUMINAMATH_CALUDE_p_difference_qr_l3958_395871


namespace NUMINAMATH_CALUDE_divisor_problem_l3958_395862

theorem divisor_problem (d : ℕ+) : 
  (∃ n : ℕ, n % d = 3 ∧ (2 * n) % d = 2) → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3958_395862


namespace NUMINAMATH_CALUDE_tuesday_rainfall_l3958_395846

/-- Given that it rained 0.9 inches on Monday and Tuesday's rainfall was 0.7 inches less than Monday's,
    prove that it rained 0.2 inches on Tuesday. -/
theorem tuesday_rainfall (monday_rain : ℝ) (tuesday_difference : ℝ) 
  (h1 : monday_rain = 0.9)
  (h2 : tuesday_difference = 0.7) :
  monday_rain - tuesday_difference = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_rainfall_l3958_395846


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3958_395850

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + (x + 1/2) * f (1 - x) = 1

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
    (f 0 = 2 ∧ f 1 = -2) ∧
    (∀ x ≠ 1/2, f x = 2 / (1 - 2*x)) ∧
    f (1/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3958_395850


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_one_range_of_a_l3958_395844

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| - |x + 1|

-- Part I
theorem solution_set_for_a_eq_one :
  {x : ℝ | f 1 x ≤ x^2 - x} = {x : ℝ | x ≤ -1 ∨ x ≥ 0} :=
sorry

-- Part II
theorem range_of_a (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : 2*m + n = 1) :
  (∀ x, f a x ≤ 1/m + 2/n) → -9 ≤ a ∧ a ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_one_range_of_a_l3958_395844


namespace NUMINAMATH_CALUDE_quadratic_form_completion_constant_term_value_l3958_395809

theorem quadratic_form_completion (x : ℝ) : 
  x^2 - 6*x = (x - 3)^2 - 9 :=
sorry

theorem constant_term_value : 
  ∃ k, ∀ x, x^2 - 6*x = (x - 3)^2 + k ∧ k = -9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_form_completion_constant_term_value_l3958_395809


namespace NUMINAMATH_CALUDE_quadratic_completion_l3958_395857

theorem quadratic_completion (b : ℝ) (n : ℝ) : 
  (∀ x, x^2 + b*x + (1/5 : ℝ) = (x + n)^2 + (1/20 : ℝ)) → b < 0 → b = -Real.sqrt (3/5)
:= by sorry

end NUMINAMATH_CALUDE_quadratic_completion_l3958_395857


namespace NUMINAMATH_CALUDE_max_fewer_cards_l3958_395890

/-- The set of digits that remain valid when flipped upside down -/
def valid_flip_digits : Finset ℕ := {1, 6, 8, 9}

/-- The set of digits that can be used in the tens place for reversible numbers -/
def valid_tens_digits : Finset ℕ := {0, 1, 6, 8, 9}

/-- The set of digits that can be used in the tens place for symmetrical numbers -/
def symmetrical_tens_digits : Finset ℕ := {0, 1, 8}

/-- The total number of three-digit numbers -/
def total_numbers : ℕ := 900

/-- The number of reversible three-digit numbers -/
def reversible_numbers : ℕ := (valid_tens_digits.card) * (valid_flip_digits.card) * (valid_flip_digits.card)

/-- The number of symmetrical three-digit numbers -/
def symmetrical_numbers : ℕ := (symmetrical_tens_digits.card) * (valid_flip_digits.card)

/-- The maximum number of cards needed considering reversible and symmetrical numbers -/
def max_cards_needed : ℕ := symmetrical_numbers + ((reversible_numbers - symmetrical_numbers) / 2)

/-- The theorem stating the maximum number of fewer cards that need to be printed -/
theorem max_fewer_cards : total_numbers - max_cards_needed = 854 := by sorry

end NUMINAMATH_CALUDE_max_fewer_cards_l3958_395890


namespace NUMINAMATH_CALUDE_sqrt_problems_l3958_395853

-- Define the arithmetic square root
noncomputable def arithmeticSqrt (x : ℝ) : ℝ := Real.sqrt x

-- Define the square root function that returns a set
def squareRoot (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

theorem sqrt_problems :
  (∀ x : ℝ, x > 0 → arithmeticSqrt x ≥ 0) ∧
  (squareRoot 81 = {9, -9}) ∧
  (|2 - Real.sqrt 5| = Real.sqrt 5 - 2) ∧
  (Real.sqrt (4/121) = 2/11) ∧
  (2 * Real.sqrt 3 - 5 * Real.sqrt 3 = -3 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_problems_l3958_395853


namespace NUMINAMATH_CALUDE_quadratic_roots_l3958_395821

theorem quadratic_roots (d : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + d = 0 ↔ x = (3 + Real.sqrt d) / 2 ∨ x = (3 - Real.sqrt d) / 2) →
  d = 9/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3958_395821


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l3958_395872

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l3958_395872


namespace NUMINAMATH_CALUDE_largest_x_floor_div_l3958_395879

theorem largest_x_floor_div : ∃ (x : ℝ), x = 63/8 ∧ 
  (∀ (y : ℝ), y > x → ⌊y⌋/y ≠ 8/9) ∧ ⌊x⌋/x = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_div_l3958_395879


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3958_395896

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x - 4| + |x + 3| < a) → a > 7 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3958_395896


namespace NUMINAMATH_CALUDE_fraction_value_at_three_l3958_395826

theorem fraction_value_at_three :
  let x : ℝ := 3
  (x^8 + 16*x^4 + 64) / (x^4 + 8) = 89 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_at_three_l3958_395826


namespace NUMINAMATH_CALUDE_product_to_standard_form_l3958_395895

theorem product_to_standard_form (x : ℝ) : 
  (x - 1) * (x + 3) * (x + 5) = x^3 + 7*x^2 + 7*x - 15 := by
  sorry

end NUMINAMATH_CALUDE_product_to_standard_form_l3958_395895


namespace NUMINAMATH_CALUDE_calculate_matches_played_rahul_matches_played_l3958_395891

/-- 
Given a cricketer's current batting average and the change in average after scoring in an additional match,
this theorem calculates the number of matches played before the additional match.
-/
theorem calculate_matches_played (current_average : ℚ) (additional_runs : ℕ) (new_average : ℚ) : ℕ :=
  let m := (additional_runs - new_average) / (new_average - current_average)
  m.num.toNat

/--
Proves that Rahul has played 5 matches given his current batting average and the change after an additional match.
-/
theorem rahul_matches_played : calculate_matches_played 51 69 54 = 5 := by
  sorry

end NUMINAMATH_CALUDE_calculate_matches_played_rahul_matches_played_l3958_395891


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l3958_395837

/-- The height of a tree after a given number of years, given that it triples its height every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem stating that if a tree reaches 81 feet after 4 years of tripling its height annually, 
    then its height after 2 years is 9 feet -/
theorem tree_height_after_two_years 
  (h : ∃ initial_height : ℝ, tree_height initial_height 4 = 81) : 
  ∃ initial_height : ℝ, tree_height initial_height 2 = 9 :=
sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l3958_395837
