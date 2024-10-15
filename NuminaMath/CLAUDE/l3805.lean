import Mathlib

namespace NUMINAMATH_CALUDE_smallest_number_l3805_380590

theorem smallest_number (a b c d : ℚ) (ha : a = 0) (hb : b = -2/3) (hc : c = 1) (hd : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3805_380590


namespace NUMINAMATH_CALUDE_quadratic_radical_condition_l3805_380551

theorem quadratic_radical_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 3) ↔ x ≥ -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_radical_condition_l3805_380551


namespace NUMINAMATH_CALUDE_sum_of_squares_l3805_380576

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3805_380576


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l3805_380520

/-- For a normal distribution with mean 14.5 and standard deviation 1.7,
    the value that is exactly 2 standard deviations less than the mean is 11.1. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (h1 : μ = 14.5) (h2 : σ = 1.7) :
  μ - 2 * σ = 11.1 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l3805_380520


namespace NUMINAMATH_CALUDE_x_minus_y_equals_two_l3805_380534

theorem x_minus_y_equals_two (x y : ℝ) 
  (sum_eq : x + y = 6) 
  (diff_squares_eq : x^2 - y^2 = 12) : 
  x - y = 2 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_two_l3805_380534


namespace NUMINAMATH_CALUDE_second_tract_width_l3805_380569

/-- Given two rectangular tracts of land, prove that the width of the second tract is 630 meters -/
theorem second_tract_width (length1 width1 length2 combined_area : ℝ)
  (h1 : length1 = 300)
  (h2 : width1 = 500)
  (h3 : length2 = 250)
  (h4 : combined_area = 307500)
  (h5 : combined_area = length1 * width1 + length2 * (combined_area - length1 * width1) / length2) :
  (combined_area - length1 * width1) / length2 = 630 := by
  sorry

end NUMINAMATH_CALUDE_second_tract_width_l3805_380569


namespace NUMINAMATH_CALUDE_quadratic_equation_m_range_l3805_380523

/-- Given a quadratic equation (m-1)x² + x + 1 = 0 with real roots, 
    prove that the range of m is m ≤ 5/4 and m ≠ 1 -/
theorem quadratic_equation_m_range (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + x + 1 = 0) → 
  (m ≤ 5/4 ∧ m ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_range_l3805_380523


namespace NUMINAMATH_CALUDE_max_red_balls_l3805_380532

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Yellow
  | Green

/-- The number marked on a ball of a given color -/
def ballNumber (c : BallColor) : Nat :=
  match c with
  | BallColor.Red => 4
  | BallColor.Yellow => 5
  | BallColor.Green => 6

/-- The total number of balls drawn -/
def totalBalls : Nat := 8

/-- The sum of numbers on all drawn balls -/
def totalSum : Nat := 39

/-- A configuration of drawn balls -/
structure BallConfiguration where
  red : Nat
  yellow : Nat
  green : Nat
  sum_eq : red + yellow + green = totalBalls
  number_sum_eq : red * ballNumber BallColor.Red + 
                  yellow * ballNumber BallColor.Yellow + 
                  green * ballNumber BallColor.Green = totalSum

/-- The maximum number of red balls in any valid configuration is 4 -/
theorem max_red_balls : 
  ∀ (config : BallConfiguration), config.red ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_red_balls_l3805_380532


namespace NUMINAMATH_CALUDE_main_theorem_l3805_380575

/-- Definition of the function f --/
def f (a b k : ℤ) : ℤ := a * k^3 + b * k

/-- Definition of n-good --/
def is_n_good (a b n : ℤ) : Prop :=
  ∀ m k : ℤ, n ∣ (f a b k - f a b m) → n ∣ (k - m)

/-- Definition of very good --/
def is_very_good (a b : ℤ) : Prop :=
  ∀ n : ℤ, ∃ m : ℤ, m > n ∧ is_n_good a b m

/-- Main theorem --/
theorem main_theorem :
  (is_n_good 1 (-51^2) 51 ∧ ¬ is_very_good 1 (-51^2)) ∧
  (∀ a b : ℤ, is_n_good a b 2013 → is_very_good a b) := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l3805_380575


namespace NUMINAMATH_CALUDE_sector_radius_l3805_380501

theorem sector_radius (θ : ℝ) (L : ℝ) (R : ℝ) :
  θ = 60 → L = π → L = (θ * π * R) / 180 → R = 3 :=
by sorry

end NUMINAMATH_CALUDE_sector_radius_l3805_380501


namespace NUMINAMATH_CALUDE_unique_integer_point_implies_c_value_l3805_380586

/-- The x-coordinate of the first point -/
def x1 : ℚ := 22

/-- The y-coordinate of the first point -/
def y1 : ℚ := 38/3

/-- The y-coordinate of the second point -/
def y2 : ℚ := 53/3

/-- The number of integer points on the line segment -/
def num_integer_points : ℕ := 1

/-- The x-coordinate of the second point -/
def c : ℚ := 23

theorem unique_integer_point_implies_c_value :
  (∃! p : ℤ × ℤ, (x1 : ℚ) < p.1 ∧ p.1 < c ∧
    (p.2 : ℚ) = y1 + (y2 - y1) / (c - x1) * ((p.1 : ℚ) - x1)) →
  c = 23 := by sorry

end NUMINAMATH_CALUDE_unique_integer_point_implies_c_value_l3805_380586


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_12_l3805_380588

def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) - a n = 1

def geometricMean (x y z : ℚ) : Prop :=
  z * z = x * y

def sumOfTerms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_12 (a : ℕ → ℚ) :
  arithmeticSequence a →
  geometricMean (a 3) (a 11) (a 6) →
  sumOfTerms a 12 = 96 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_12_l3805_380588


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3805_380500

theorem repeating_decimal_sum : 
  (234 : ℚ) / 999 - (567 : ℚ) / 999 + (891 : ℚ) / 999 = (186 : ℚ) / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3805_380500


namespace NUMINAMATH_CALUDE_lateral_surface_area_cylinder_l3805_380536

/-- The lateral surface area of a cylinder with radius 1 and height 2 is 4π. -/
theorem lateral_surface_area_cylinder : 
  let r : ℝ := 1
  let h : ℝ := 2
  2 * Real.pi * r * h = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_lateral_surface_area_cylinder_l3805_380536


namespace NUMINAMATH_CALUDE_parabola_vertex_range_l3805_380591

/-- Represents a parabola with equation y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The vertex of a parabola -/
structure Vertex where
  s : ℝ
  t : ℝ

theorem parabola_vertex_range 
  (p : Parabola) 
  (v : Vertex) 
  (y₁ y₂ : ℝ)
  (h1 : p.a * (-2)^2 + p.b * (-2) + p.c = y₁)
  (h2 : p.a * 4^2 + p.b * 4 + p.c = y₂)
  (h3 : y₁ > y₂)
  (h4 : y₂ > v.t)
  : v.s > 1 ∧ v.s ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_range_l3805_380591


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3805_380511

/-- Given that the coefficients of the first three terms in the expansion of (x + 1/(2x))^n form an arithmetic sequence,
    prove that the coefficient of the x^4 term in the expansion is 7. -/
theorem binomial_expansion_coefficient (n : ℕ) : 
  (∃ d : ℚ, (1 : ℚ) = (n.choose 0 : ℚ) ∧ 
             (1/2 : ℚ) * (n.choose 1 : ℚ) = (n.choose 0 : ℚ) + d ∧ 
             (1/4 : ℚ) * (n.choose 2 : ℚ) = (n.choose 0 : ℚ) + 2*d) → 
  (1/4 : ℚ) * (n.choose 4 : ℚ) = 7 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l3805_380511


namespace NUMINAMATH_CALUDE_no_house_spirits_l3805_380595

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (HouseSpirit : U → Prop)
variable (LovesMischief : U → Prop)
variable (LovesCleanlinessAndOrder : U → Prop)

-- State the theorem
theorem no_house_spirits
  (h1 : ∀ x, HouseSpirit x → LovesMischief x)
  (h2 : ∀ x, HouseSpirit x → LovesCleanlinessAndOrder x)
  (h3 : ∀ x, LovesCleanlinessAndOrder x → ¬LovesMischief x) :
  ¬∃ x, HouseSpirit x :=
by sorry

end NUMINAMATH_CALUDE_no_house_spirits_l3805_380595


namespace NUMINAMATH_CALUDE_max_reflections_theorem_l3805_380577

/-- The angle between two lines in degrees -/
def angle_between_lines : ℝ := 6

/-- The maximum number of reflections before perpendicular incidence -/
def max_reflections : ℕ := 15

/-- Theorem: Given the angle between two lines is 6°, the maximum number of reflections
    before perpendicular incidence is 15 -/
theorem max_reflections_theorem (angle : ℝ) (n : ℕ) 
  (h1 : angle = angle_between_lines)
  (h2 : n = max_reflections) :
  n * angle = 90 ∧ ∀ m : ℕ, m > n → m * angle > 90 := by
  sorry

#check max_reflections_theorem

end NUMINAMATH_CALUDE_max_reflections_theorem_l3805_380577


namespace NUMINAMATH_CALUDE_ren_faire_amulet_sales_l3805_380554

/-- Represents the problem of calculating amulets sold per day at a Ren Faire --/
theorem ren_faire_amulet_sales (selling_price : ℕ) (cost_price : ℕ) (revenue_share : ℚ)
  (total_days : ℕ) (total_profit : ℕ) :
  selling_price = 40 →
  cost_price = 30 →
  revenue_share = 1/10 →
  total_days = 2 →
  total_profit = 300 →
  (selling_price - cost_price - (revenue_share * selling_price)) * total_days * 
    (total_profit / ((selling_price - cost_price - (revenue_share * selling_price)) * total_days)) = 25 :=
by sorry

end NUMINAMATH_CALUDE_ren_faire_amulet_sales_l3805_380554


namespace NUMINAMATH_CALUDE_max_value_of_g_l3805_380538

-- Define the function g(x)
def g (x : ℝ) : ℝ := 9*x - 2*x^3

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l3805_380538


namespace NUMINAMATH_CALUDE_min_value_expression_l3805_380546

theorem min_value_expression (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 10)
  (hb : 1 ≤ b ∧ b ≤ 10)
  (hc : 1 ≤ c ∧ c ≤ 10)
  (hbc : b < c) :
  4 ≤ 3*a - a*b + a*c ∧ ∃ (a' b' c' : ℕ), 
    1 ≤ a' ∧ a' ≤ 10 ∧
    1 ≤ b' ∧ b' ≤ 10 ∧
    1 ≤ c' ∧ c' ≤ 10 ∧
    b' < c' ∧
    3*a' - a'*b' + a'*c' = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3805_380546


namespace NUMINAMATH_CALUDE_product_difference_sum_l3805_380558

theorem product_difference_sum : 
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧ 
    a * b = 50 ∧ 
    (max a b - min a b) = 5 → 
    a + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_sum_l3805_380558


namespace NUMINAMATH_CALUDE_solution_set_part_I_solution_part_II_l3805_380512

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Part I
theorem solution_set_part_I :
  {x : ℝ | f 1 x ≥ 3 * x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} := by sorry

-- Part II
theorem solution_part_II (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -3}) → a = 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_I_solution_part_II_l3805_380512


namespace NUMINAMATH_CALUDE_fern_leaves_count_l3805_380598

/-- The number of ferns Karen hangs -/
def num_ferns : ℕ := 6

/-- The number of fronds each fern has -/
def fronds_per_fern : ℕ := 7

/-- The number of leaves each frond has -/
def leaves_per_frond : ℕ := 30

/-- The total number of leaves on all ferns -/
def total_leaves : ℕ := num_ferns * fronds_per_fern * leaves_per_frond

theorem fern_leaves_count : total_leaves = 1260 := by
  sorry

end NUMINAMATH_CALUDE_fern_leaves_count_l3805_380598


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3805_380597

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 3 + a 5 = 7) →
  (a 5 + a 7 + a 9 = 28) →
  (a 9 + a 11 + a 13 = 112) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3805_380597


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l3805_380592

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l₂ (a x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) :
  (∀ x y, l₁ a x y ∧ l₂ a x y → (a * 1 + 2 * (a - 1) = 0)) →
  a = 2/3 :=
sorry

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) :
  (∀ x y, l₁ a x y ∧ l₂ a x y → (a / 1 = 2 / (a - 1) ∧ a / 1 ≠ 6 / (a^2 - 1))) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l3805_380592


namespace NUMINAMATH_CALUDE_fraction_simplification_l3805_380549

theorem fraction_simplification (x y : ℝ) (h : x ≠ y) :
  (x + y) / (x - y) - (2 * y) / (x - y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3805_380549


namespace NUMINAMATH_CALUDE_hapok_guarantee_l3805_380578

/-- Represents the coin division game between Hapok and Glazok -/
structure CoinGame where
  total_coins : ℕ
  max_handfuls : ℕ

/-- Represents a strategy for Hapok -/
def HapokStrategy := ℕ → ℕ

/-- Represents a strategy for Glazok -/
def GlazokStrategy := ℕ → Bool

/-- The outcome of the game given strategies for both players -/
def gameOutcome (game : CoinGame) (hapok_strat : HapokStrategy) (glazok_strat : GlazokStrategy) : ℕ := sorry

/-- Hapok's guaranteed minimum coins -/
def hapokGuaranteedCoins (game : CoinGame) : ℕ := sorry

/-- The main theorem stating Hapok can guarantee at least 46 coins -/
theorem hapok_guarantee (game : CoinGame) (h1 : game.total_coins = 100) (h2 : game.max_handfuls = 9) :
  hapokGuaranteedCoins game ≥ 46 := sorry

end NUMINAMATH_CALUDE_hapok_guarantee_l3805_380578


namespace NUMINAMATH_CALUDE_solution_equivalence_l3805_380524

/-- Given prime numbers p and q with p < q, the positive integer solutions (x, y) to 
    1/x + 1/y = 1/p - 1/q are equivalent to the positive integer solutions of 
    ((q - p)x - pq)((q - p)y - pq) = p^2q^2 -/
theorem solution_equivalence (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p < q) :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  (1 / x + 1 / y = 1 / p - 1 / q) ↔ 
  ((q - p) * x - p * q) * ((q - p) * y - p * q) = p^2 * q^2 :=
by sorry

end NUMINAMATH_CALUDE_solution_equivalence_l3805_380524


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l3805_380579

/-- Represents a cube composed of unit cubes -/
structure LargeCube where
  size : ℕ
  total_units : ℕ

/-- Represents a plane intersecting the large cube -/
structure IntersectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

/-- Counts the number of unit cubes intersected by the plane -/
def count_intersected_cubes (cube : LargeCube) (plane : IntersectingPlane) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem intersected_cubes_count 
  (cube : LargeCube) 
  (plane : IntersectingPlane) 
  (h1 : cube.size = 4) 
  (h2 : cube.total_units = 64) 
  (h3 : plane.perpendicular_to_diagonal) 
  (h4 : plane.bisects_diagonal) : 
  count_intersected_cubes cube plane = 56 :=
sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l3805_380579


namespace NUMINAMATH_CALUDE_remainder_98_pow_50_mod_50_l3805_380522

theorem remainder_98_pow_50_mod_50 : 98^50 % 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_98_pow_50_mod_50_l3805_380522


namespace NUMINAMATH_CALUDE_distribute_5_3_l3805_380568

/-- The number of ways to distribute indistinguishable objects into distinguishable containers -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 21 ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l3805_380568


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3805_380572

theorem infinite_geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = (1/4 : ℝ)) 
  (h2 : S = 80) 
  (h3 : S = a / (1 - r)) : 
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3805_380572


namespace NUMINAMATH_CALUDE_value_of_E_l3805_380587

theorem value_of_E (a b c : ℝ) (h1 : a ≠ b) (h2 : a^2 * (b + c) = 2023) (h3 : b^2 * (c + a) = 2023) :
  c^2 * (a + b) = 2023 := by
sorry

end NUMINAMATH_CALUDE_value_of_E_l3805_380587


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3805_380563

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) 
  (h_arith : ArithmeticSequence a)
  (h_sum1 : a 2 + a 3 = 4)
  (h_sum2 : a 4 + a 5 = 6) :
  a 9 + a 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3805_380563


namespace NUMINAMATH_CALUDE_lunch_break_duration_l3805_380533

-- Define the painting rates and lunch break
structure PaintingData where
  paula_rate : ℝ
  helpers_rate : ℝ
  lunch_break : ℝ

-- Define the workday durations
def monday_duration : ℝ := 9
def tuesday_duration : ℝ := 7
def wednesday_duration : ℝ := 12

-- Define the portions painted each day
def monday_portion : ℝ := 0.6
def tuesday_portion : ℝ := 0.3
def wednesday_portion : ℝ := 0.1

-- Theorem statement
theorem lunch_break_duration (d : PaintingData) : 
  (monday_duration - d.lunch_break) * (d.paula_rate + d.helpers_rate) = monday_portion ∧
  (tuesday_duration - d.lunch_break) * d.helpers_rate = tuesday_portion ∧
  (wednesday_duration - d.lunch_break) * d.paula_rate = wednesday_portion →
  d.lunch_break = 1 := by
  sorry

#check lunch_break_duration

end NUMINAMATH_CALUDE_lunch_break_duration_l3805_380533


namespace NUMINAMATH_CALUDE_triplet_sum_not_seven_l3805_380518

theorem triplet_sum_not_seven : 
  let triplet_A := (3/2, 4/3, 13/6)
  let triplet_B := (4, -3, 6)
  let triplet_C := (2.5, 3.1, 1.4)
  let triplet_D := (7.4, -9.4, 9.0)
  let triplet_E := (-3/4, -9/4, 8)
  
  let sum_A := triplet_A.1 + triplet_A.2.1 + triplet_A.2.2
  let sum_B := triplet_B.1 + triplet_B.2.1 + triplet_B.2.2
  let sum_C := triplet_C.1 + triplet_C.2.1 + triplet_C.2.2
  let sum_D := triplet_D.1 + triplet_D.2.1 + triplet_D.2.2
  let sum_E := triplet_E.1 + triplet_E.2.1 + triplet_E.2.2
  
  (sum_A ≠ 7 ∧ sum_E ≠ 7) ∧ (sum_B = 7 ∧ sum_C = 7 ∧ sum_D = 7) := by
  sorry

end NUMINAMATH_CALUDE_triplet_sum_not_seven_l3805_380518


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3805_380584

theorem sum_of_cubes (x y z : ℝ) 
  (sum_eq : x + y + z = 7)
  (sum_prod_eq : x*y + x*z + y*z = 9)
  (prod_eq : x*y*z = -18) : 
  x^3 + y^3 + z^3 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3805_380584


namespace NUMINAMATH_CALUDE_base2_to_base4_conversion_l3805_380529

def base2_to_decimal (n : List Bool) : ℕ :=
  n.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List (Fin 4) :=
      if m = 0 then [] else (m % 4) :: aux (m / 4)
    aux n |>.reverse

theorem base2_to_base4_conversion :
  let base2 : List Bool := [true, false, true, false, true, false, true, false, true]
  let base4 : List (Fin 4) := [1, 1, 1, 1, 1]
  decimal_to_base4 (base2_to_decimal base2) = base4 := by
  sorry

end NUMINAMATH_CALUDE_base2_to_base4_conversion_l3805_380529


namespace NUMINAMATH_CALUDE_range_of_s_l3805_380525

-- Define a composite positive integer not divisible by 3
def IsCompositeNotDivisibleBy3 (n : ℕ) : Prop :=
  n > 1 ∧ ¬ (∃ k : ℕ, n = k * k) ∧ ¬ (3 ∣ n)

-- Define the function s
def s (n : ℕ) (h : IsCompositeNotDivisibleBy3 n) : ℕ :=
  sorry -- Implementation of s is not required for the statement

-- The main theorem
theorem range_of_s :
  ∀ m : ℤ, m > 3 ↔ ∃ (n : ℕ) (h : IsCompositeNotDivisibleBy3 n), s n h = m :=
sorry

end NUMINAMATH_CALUDE_range_of_s_l3805_380525


namespace NUMINAMATH_CALUDE_stratified_sampling_senior_high_l3805_380570

theorem stratified_sampling_senior_high (total_students : ℕ) (senior_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 1800)
  (h2 : senior_students = 600)
  (h3 : sample_size = 180) :
  (senior_students * sample_size) / total_students = 60 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_senior_high_l3805_380570


namespace NUMINAMATH_CALUDE_rachel_homework_l3805_380594

theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 7 →
  math_pages = reading_pages + 4 →
  reading_pages = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_l3805_380594


namespace NUMINAMATH_CALUDE_two_correct_statements_l3805_380553

theorem two_correct_statements :
  let statement1 := ∀ a b : ℝ, (a > 0 ∧ b < 0) ∨ (a < 0 ∧ b > 0) → a + b = 0
  let statement2 := ∀ a : ℝ, -a < 0
  let statement3 := ∀ n : ℤ, n ≠ 0
  let statement4 := ∀ a b : ℝ, |a| > |b| → |a| > |b - 0|
  let statement5 := ∀ a : ℝ, a ≠ 0 → |a| > 0
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ statement4 ∧ statement5) :=
by
  sorry

end NUMINAMATH_CALUDE_two_correct_statements_l3805_380553


namespace NUMINAMATH_CALUDE_repeating_decimal_interval_l3805_380544

def is_repeating_decimal_of_period (n : ℕ) (p : ℕ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ n ∣ (10^p - 1) ∧ ∀ (q : ℕ), q < p → ¬(n ∣ (10^q - 1))

theorem repeating_decimal_interval :
  ∀ (n : ℕ),
    n > 0 →
    n < 2000 →
    is_repeating_decimal_of_period n 4 →
    is_repeating_decimal_of_period (n + 4) 6 →
    801 ≤ n ∧ n ≤ 1200 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_interval_l3805_380544


namespace NUMINAMATH_CALUDE_product_of_primes_summing_to_91_l3805_380527

theorem product_of_primes_summing_to_91 (p q : ℕ) : 
  Prime p → Prime q → p + q = 91 → p * q = 178 := by
  sorry

end NUMINAMATH_CALUDE_product_of_primes_summing_to_91_l3805_380527


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l3805_380561

theorem magnitude_of_complex_number : 
  Complex.abs ((1 + Complex.I)^2 / (1 - 2 * Complex.I)) = 2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l3805_380561


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3805_380564

theorem interest_rate_calculation (initial_amount : ℝ) (final_amount : ℝ) 
  (second_year_rate : ℝ) (first_year_rate : ℝ) : 
  initial_amount = 6000 ∧ 
  final_amount = 6552 ∧ 
  second_year_rate = 0.05 ∧
  first_year_rate = 0.04 →
  final_amount = initial_amount + 
    (initial_amount * first_year_rate) + 
    ((initial_amount + initial_amount * first_year_rate) * second_year_rate) :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3805_380564


namespace NUMINAMATH_CALUDE_ratatouille_price_proof_l3805_380580

def ratatouille_problem (eggplant_weight : ℝ) (zucchini_weight : ℝ) 
  (tomato_price : ℝ) (tomato_weight : ℝ)
  (onion_price : ℝ) (onion_weight : ℝ)
  (basil_price : ℝ) (basil_weight : ℝ)
  (total_quarts : ℝ) (price_per_quart : ℝ) : Prop :=
  let total_weight := eggplant_weight + zucchini_weight
  let other_ingredients_cost := tomato_price * tomato_weight + 
                                onion_price * onion_weight + 
                                basil_price * basil_weight * 2
  let total_cost := total_quarts * price_per_quart
  let eggplant_zucchini_cost := total_cost - other_ingredients_cost
  let price_per_pound := eggplant_zucchini_cost / total_weight
  price_per_pound = 2

theorem ratatouille_price_proof :
  ratatouille_problem 5 4 3.5 4 1 3 2.5 1 4 10 := by
  sorry

end NUMINAMATH_CALUDE_ratatouille_price_proof_l3805_380580


namespace NUMINAMATH_CALUDE_squares_sum_difference_l3805_380540

theorem squares_sum_difference : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 241 := by
  sorry

end NUMINAMATH_CALUDE_squares_sum_difference_l3805_380540


namespace NUMINAMATH_CALUDE_jamies_father_age_ratio_l3805_380505

/-- The year of Jamie's 10th birthday -/
def birth_year : ℕ := 2010

/-- Jamie's age on his 10th birthday -/
def jamie_initial_age : ℕ := 10

/-- The ratio of Jamie's father's age to Jamie's age on Jamie's 10th birthday -/
def initial_age_ratio : ℕ := 5

/-- The year when Jamie's father's age is twice Jamie's age -/
def target_year : ℕ := 2040

/-- The ratio of Jamie's father's age to Jamie's age in the target year -/
def target_age_ratio : ℕ := 2

theorem jamies_father_age_ratio :
  target_year = birth_year + (initial_age_ratio - target_age_ratio) * jamie_initial_age := by
  sorry

#check jamies_father_age_ratio

end NUMINAMATH_CALUDE_jamies_father_age_ratio_l3805_380505


namespace NUMINAMATH_CALUDE_fraction_problem_l3805_380582

theorem fraction_problem (N : ℝ) (f : ℝ) (h1 : N = 12) (h2 : 1 + f * N = 0.75 * N) : f = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3805_380582


namespace NUMINAMATH_CALUDE_a_minus_b_equals_1790_l3805_380516

/-- Prove that A - B = 1790 given the definitions of A and B -/
theorem a_minus_b_equals_1790 :
  let A := 1 * 1000 + 16 * 100 + 28 * 10
  let B := 355 + 3 * 245
  A - B = 1790 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_1790_l3805_380516


namespace NUMINAMATH_CALUDE_student_calculation_correct_result_problem_statement_l3805_380509

def repeating_decimal (c d : ℕ) : ℚ :=
  1 + (10 * c + d : ℚ) / 99

theorem student_calculation (c d : ℕ) : ℚ :=
  74 * (1 + (c : ℚ) / 10 + (d : ℚ) / 100) + 3

theorem correct_result (c d : ℕ) : ℚ :=
  74 * repeating_decimal c d + 3

theorem problem_statement (c d : ℕ) : 
  correct_result c d - student_calculation c d = 1.2 → c = 1 ∧ d = 6 :=
sorry

end NUMINAMATH_CALUDE_student_calculation_correct_result_problem_statement_l3805_380509


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3805_380556

theorem quadratic_equation_properties (k m : ℝ) 
  (h : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (1 + 2*k^2)*x₁^2 - 4*k*m*x₁ + 2*m^2 - 2 = 0 
                   ∧ (1 + 2*k^2)*x₂^2 - 4*k*m*x₂ + 2*m^2 - 2 = 0) : 
  m^2 < 1 + 2*k^2 
  ∧ (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (1 + 2*k^2)*x₁^2 - 4*k*m*x₁ + 2*m^2 - 2 = 0 
                 → (1 + 2*k^2)*x₂^2 - 4*k*m*x₂ + 2*m^2 - 2 = 0 
                 → x₁*x₂ < 2)
  ∧ (∃ S : ℝ → ℝ, (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (1 + 2*k^2)*x₁^2 - 4*k*m*x₁ + 2*m^2 - 2 = 0 
                                       → (1 + 2*k^2)*x₂^2 - 4*k*m*x₂ + 2*m^2 - 2 = 0 
                                       → S m = |m| * Real.sqrt ((x₁ + x₂)^2 - 4*x₁*x₂))
     ∧ (∀ m : ℝ, S m ≤ Real.sqrt 2)
     ∧ (∃ m : ℝ, S m = Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3805_380556


namespace NUMINAMATH_CALUDE_class_size_l3805_380581

/-- The number of girls in Jungkook's class -/
def num_girls : ℕ := 9

/-- The number of boys in Jungkook's class -/
def num_boys : ℕ := 16

/-- The total number of students in Jungkook's class -/
def total_students : ℕ := num_girls + num_boys

theorem class_size : total_students = 25 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3805_380581


namespace NUMINAMATH_CALUDE_dodecagon_area_times_hundred_l3805_380508

/-- The area of a regular dodecagon inscribed in a unit circle -/
def dodecagonArea : ℝ := 3

/-- 100 times the area of a regular dodecagon inscribed in a unit circle -/
def hundredTimesDodecagonArea : ℝ := 100 * dodecagonArea

theorem dodecagon_area_times_hundred : hundredTimesDodecagonArea = 300 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_area_times_hundred_l3805_380508


namespace NUMINAMATH_CALUDE_jiale_pricing_correct_l3805_380557

/-- Represents the pricing and discount options for teapots and teacups -/
structure TeaSetPricing where
  teapot_price : ℝ
  teacup_price : ℝ
  option1 : ℝ → ℝ  -- Cost function for Option 1
  option2 : ℝ → ℝ  -- Cost function for Option 2

/-- The specific pricing structure for Jiale Supermarket -/
def jiale_pricing : TeaSetPricing :=
  { teapot_price := 90
    teacup_price := 25
    option1 := λ x => 25 * x + 325
    option2 := λ x => 22.5 * x + 405 }

/-- Theorem stating the correctness of the cost calculations -/
theorem jiale_pricing_correct (x : ℝ) (h : x > 5) :
  let p := jiale_pricing
  p.option1 x = 25 * x + 325 ∧ p.option2 x = 22.5 * x + 405 := by
  sorry

#check jiale_pricing_correct

end NUMINAMATH_CALUDE_jiale_pricing_correct_l3805_380557


namespace NUMINAMATH_CALUDE_transformation_maps_curve_to_ellipse_l3805_380571

/-- The transformation that maps a curve to an ellipse -/
def transformation (x' y' : ℝ) : ℝ × ℝ :=
  (2 * x', y')

/-- The original curve equation -/
def original_curve (x y : ℝ) : Prop :=
  y^2 = 4

/-- The transformed ellipse equation -/
def transformed_ellipse (x' y' : ℝ) : Prop :=
  x'^2 + y'^2 / 4 = 1

/-- Theorem stating that the transformation maps the original curve to the ellipse -/
theorem transformation_maps_curve_to_ellipse :
  ∀ x' y', original_curve (transformation x' y').1 (transformation x' y').2 ↔ transformed_ellipse x' y' :=
sorry

end NUMINAMATH_CALUDE_transformation_maps_curve_to_ellipse_l3805_380571


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3805_380574

theorem simplify_and_evaluate (x : ℝ) (h : x = 4) : 
  (2 - 2*x/(x-2)) / ((x^2 - 4) / (x^2 - 4*x + 4)) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3805_380574


namespace NUMINAMATH_CALUDE_cases_in_1990_l3805_380521

/-- Calculates the number of disease cases in a given year, assuming a linear decrease from 1960 to 2000 --/
def diseaseCases (year : ℕ) : ℕ :=
  let initialCases : ℕ := 600000
  let finalCases : ℕ := 600
  let initialYear : ℕ := 1960
  let finalYear : ℕ := 2000
  let totalYears : ℕ := finalYear - initialYear
  let yearlyDecrease : ℕ := (initialCases - finalCases) / totalYears
  initialCases - yearlyDecrease * (year - initialYear)

theorem cases_in_1990 :
  diseaseCases 1990 = 150450 := by
  sorry

end NUMINAMATH_CALUDE_cases_in_1990_l3805_380521


namespace NUMINAMATH_CALUDE_minimum_words_for_90_percent_l3805_380510

/-- Represents the French exam vocabulary test -/
structure FrenchExam where
  total_words : ℕ
  learned_words : ℕ
  score_threshold : ℚ

/-- Calculate the score for a given exam -/
def calculate_score (exam : FrenchExam) : ℚ :=
  (exam.learned_words + (exam.total_words - exam.learned_words) / 10) / exam.total_words

/-- Theorem stating the minimum number of words to learn for a 90% score -/
theorem minimum_words_for_90_percent (exam : FrenchExam) 
    (h1 : exam.total_words = 800)
    (h2 : exam.score_threshold = 9/10) :
    (∀ n : ℕ, n < 712 → calculate_score ⟨exam.total_words, n, exam.score_threshold⟩ < exam.score_threshold) ∧
    calculate_score ⟨exam.total_words, 712, exam.score_threshold⟩ ≥ exam.score_threshold :=
  sorry


end NUMINAMATH_CALUDE_minimum_words_for_90_percent_l3805_380510


namespace NUMINAMATH_CALUDE_cubic_sum_zero_l3805_380515

theorem cubic_sum_zero (a b c : ℝ) : 
  a + b + c = 0 → a^3 + a^2*c - a*b*c + b^2*c + b^3 = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_zero_l3805_380515


namespace NUMINAMATH_CALUDE_basketball_chess_fans_l3805_380517

/-- The number of students who like basketball or chess given the following conditions:
  * 40% of students like basketball
  * 10% of students like chess
  * 250 students were interviewed
-/
theorem basketball_chess_fans (total_students : ℕ) (basketball_percent : ℚ) (chess_percent : ℚ) :
  total_students = 250 →
  basketball_percent = 40 / 100 →
  chess_percent = 10 / 100 →
  (basketball_percent + chess_percent) * total_students = 125 := by
sorry

end NUMINAMATH_CALUDE_basketball_chess_fans_l3805_380517


namespace NUMINAMATH_CALUDE_gasoline_quantity_reduction_l3805_380550

/-- Proves that a 25% price increase and 10% budget increase results in a 12% reduction in quantity --/
theorem gasoline_quantity_reduction 
  (P : ℝ) -- Original price of gasoline
  (Q : ℝ) -- Original quantity of gasoline
  (h1 : P > 0) -- Assumption: Price is positive
  (h2 : Q > 0) -- Assumption: Quantity is positive
  : 
  let new_price := 1.25 * P -- 25% price increase
  let new_budget := 1.10 * (P * Q) -- 10% budget increase
  let new_quantity := new_budget / new_price -- New quantity calculation
  (1 - new_quantity / Q) * 100 = 12 -- Percentage reduction in quantity
  := by sorry

end NUMINAMATH_CALUDE_gasoline_quantity_reduction_l3805_380550


namespace NUMINAMATH_CALUDE_pizzas_served_during_lunch_l3805_380566

theorem pizzas_served_during_lunch (total_pizzas dinner_pizzas lunch_pizzas : ℕ) : 
  total_pizzas = 15 → 
  dinner_pizzas = 6 → 
  lunch_pizzas = total_pizzas - dinner_pizzas → 
  lunch_pizzas = 9 := by
sorry

end NUMINAMATH_CALUDE_pizzas_served_during_lunch_l3805_380566


namespace NUMINAMATH_CALUDE_square_area_7m_l3805_380547

theorem square_area_7m (side_length : ℝ) (area : ℝ) : 
  side_length = 7 → area = side_length ^ 2 → area = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_area_7m_l3805_380547


namespace NUMINAMATH_CALUDE_least_number_of_radios_l3805_380542

/-- Represents the problem of finding the least number of radios bought by a dealer. -/
theorem least_number_of_radios (n d : ℕ) (h_d_pos : d > 0) : 
  (∃ (d : ℕ), d > 0 ∧ 
    10 * n - 30 - (3 * d) / (2 * n) = 80 ∧ 
    ∀ m : ℕ, m < n → ¬(∃ (d' : ℕ), d' > 0 ∧ 10 * m - 30 - (3 * d') / (2 * m) = 80)) →
  n = 11 :=
sorry

end NUMINAMATH_CALUDE_least_number_of_radios_l3805_380542


namespace NUMINAMATH_CALUDE_curve_C_equation_l3805_380535

/-- The equation of curve C -/
def curve_C (a : ℝ) (x y : ℝ) : Prop :=
  a * x^2 + a * y^2 - 2 * a^2 * x - 4 * y = 0

/-- The equation of line l -/
def line_l (x y : ℝ) : Prop :=
  y = -2 * x + 4

/-- M and N are distinct intersection points of curve C and line l -/
def intersection_points (a : ℝ) (M N : ℝ × ℝ) : Prop :=
  M ≠ N ∧ curve_C a M.1 M.2 ∧ curve_C a N.1 N.2 ∧ line_l M.1 M.2 ∧ line_l N.1 N.2

/-- The distance from origin O to M is equal to the distance from O to N -/
def equal_distances (M N : ℝ × ℝ) : Prop :=
  M.1^2 + M.2^2 = N.1^2 + N.2^2

theorem curve_C_equation (a : ℝ) (M N : ℝ × ℝ) :
  a ≠ 0 →
  intersection_points a M N →
  equal_distances M N →
  ∃ x y : ℝ, x^2 + y^2 - 4*x - 2*y = 0 :=
by sorry

end NUMINAMATH_CALUDE_curve_C_equation_l3805_380535


namespace NUMINAMATH_CALUDE_ticket_price_reduction_l3805_380573

theorem ticket_price_reduction 
  (original_price : ℚ)
  (sold_increase_ratio : ℚ)
  (revenue_increase_ratio : ℚ)
  (price_reduction : ℚ) :
  original_price = 50 →
  sold_increase_ratio = 1/3 →
  revenue_increase_ratio = 1/4 →
  (original_price - price_reduction) * (1 + sold_increase_ratio) = original_price * (1 + revenue_increase_ratio) →
  price_reduction = 25/2 := by
sorry

end NUMINAMATH_CALUDE_ticket_price_reduction_l3805_380573


namespace NUMINAMATH_CALUDE_round_trip_time_l3805_380593

/-- Calculates the total time for a round trip on a river given the rower's speed, river speed, and distance. -/
theorem round_trip_time (rower_speed river_speed distance : ℝ) 
  (h1 : rower_speed = 6)
  (h2 : river_speed = 2)
  (h3 : distance = 2.67) : 
  (distance / (rower_speed - river_speed)) + (distance / (rower_speed + river_speed)) = 1.00125 := by
  sorry

#eval (2.67 / (6 - 2)) + (2.67 / (6 + 2))

end NUMINAMATH_CALUDE_round_trip_time_l3805_380593


namespace NUMINAMATH_CALUDE_sequence_ratio_l3805_380539

-- Define arithmetic sequence
def is_arithmetic_sequence (s : Fin 4 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 3, s (i + 1) - s i = d

-- Define geometric sequence
def is_geometric_sequence (s : Fin 5 → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : Fin 4, s (i + 1) / s i = r

theorem sequence_ratio :
  ∀ a₁ a₂ b₁ b₂ b₃ : ℝ,
  let s₁ : Fin 4 → ℝ := ![1, a₁, a₂, 9]
  let s₂ : Fin 5 → ℝ := ![1, b₁, b₂, b₃, 9]
  is_arithmetic_sequence s₁ →
  is_geometric_sequence s₂ →
  b₂ / (a₁ + a₂) = 3/10 :=
by sorry

end NUMINAMATH_CALUDE_sequence_ratio_l3805_380539


namespace NUMINAMATH_CALUDE_k_range_for_three_elements_l3805_380567

def P (k : ℝ) : Set ℕ := {x : ℕ | 2 < x ∧ x < k}

theorem k_range_for_three_elements (k : ℝ) :
  (∃ (a b c : ℕ), P k = {a, b, c} ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c) →
  5 < k ∧ k ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_k_range_for_three_elements_l3805_380567


namespace NUMINAMATH_CALUDE_unique_special_number_l3805_380530

/-- A three-digit number ending with 2 that, when the 2 is moved to the front,
    results in a number 18 greater than the original. -/
def special_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 10 = 2 ∧  -- ends with 2
  200 + (n / 10) = n + 18  -- moving 2 to front increases by 18

theorem unique_special_number :
  ∃! n : ℕ, special_number n ∧ n = 202 :=
sorry

end NUMINAMATH_CALUDE_unique_special_number_l3805_380530


namespace NUMINAMATH_CALUDE_turtles_on_log_l3805_380528

theorem turtles_on_log (initial_turtles : ℕ) : initial_turtles = 50 → 226 = initial_turtles + (7 * initial_turtles - 6) - (3 * (initial_turtles + (7 * initial_turtles - 6)) / 7) := by
  sorry

end NUMINAMATH_CALUDE_turtles_on_log_l3805_380528


namespace NUMINAMATH_CALUDE_rice_and_grain_separation_l3805_380537

/-- Represents the amount of rice in dan -/
def total_rice : ℕ := 1536

/-- Represents the sample size in grains -/
def sample_size : ℕ := 256

/-- Represents the number of mixed grain in the sample -/
def mixed_grain_sample : ℕ := 18

/-- Calculates the amount of mixed grain in the entire batch -/
def mixed_grain_total : ℕ := total_rice * mixed_grain_sample / sample_size

theorem rice_and_grain_separation :
  mixed_grain_total = 108 := by
  sorry

end NUMINAMATH_CALUDE_rice_and_grain_separation_l3805_380537


namespace NUMINAMATH_CALUDE_sections_after_five_lines_l3805_380513

/-- The number of sections in a rectangle after drawing n line segments,
    where each line increases the number of sections by its sequence order. -/
def sections (n : ℕ) : ℕ :=
  1 + (List.range n).sum

/-- Theorem: After drawing 5 line segments in a rectangle that initially has 1 section,
    where each new line segment increases the number of sections by its sequence order,
    the final number of sections is 16. -/
theorem sections_after_five_lines :
  sections 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_sections_after_five_lines_l3805_380513


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3805_380503

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequenceWithSum where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1  -- Arithmetic sequence property
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2  -- Sum formula

/-- The main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequenceWithSum) 
  (h1 : seq.a 8 - seq.a 5 = 9)
  (h2 : seq.S 8 - seq.S 5 = 66) :
  seq.a 33 = 100 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3805_380503


namespace NUMINAMATH_CALUDE_function_properties_l3805_380559

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

theorem function_properties :
  (∀ x ≠ 0, f x = x^2 + 1/x) →
  f 1 = 2 →
  (¬ (∀ x ≠ 0, f (-x) = f x) ∧ ¬ (∀ x ≠ 0, f (-x) = -f x)) ∧
  (∀ x y, 2 ≤ x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3805_380559


namespace NUMINAMATH_CALUDE_problem_solution_l3805_380555

theorem problem_solution (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 3 * t + 6) 
  (h3 : x = -6) : 
  y = 19.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3805_380555


namespace NUMINAMATH_CALUDE_triple_base_and_exponent_l3805_380543

theorem triple_base_and_exponent (a b : ℝ) (y : ℝ) (h1 : b ≠ 0) :
  (3 * a) ^ (3 * b) = a ^ b * y ^ b → y = 27 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triple_base_and_exponent_l3805_380543


namespace NUMINAMATH_CALUDE_jose_land_share_l3805_380526

def total_land_area : ℝ := 20000
def num_siblings : ℕ := 4

theorem jose_land_share :
  let total_people := num_siblings + 1
  let share := total_land_area / total_people
  share = 4000 := by
  sorry

end NUMINAMATH_CALUDE_jose_land_share_l3805_380526


namespace NUMINAMATH_CALUDE_ab_value_l3805_380583

theorem ab_value (a b c : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 27) 
  (h3 : a + b + c = 10) : 
  a * b = 9 := by
sorry

end NUMINAMATH_CALUDE_ab_value_l3805_380583


namespace NUMINAMATH_CALUDE_min_colors_2016_board_l3805_380545

/-- A color assignment for a square board. -/
def ColorAssignment (n : ℕ) := Fin n → Fin n → ℕ

/-- Predicate for a valid coloring of a square board. -/
def ValidColoring (n k : ℕ) (c : ColorAssignment n) : Prop :=
  -- One diagonal is colored with the first color
  (∀ i, c i i = 0) ∧
  -- Symmetric cells have the same color
  (∀ i j, c i j = c j i) ∧
  -- Cells in the same row on different sides of the diagonal have different colors
  (∀ i j₁ j₂, i < j₁ ∧ j₂ < i → c i j₁ ≠ c i j₂)

/-- Theorem stating the minimum number of colors needed for a 2016 × 2016 board. -/
theorem min_colors_2016_board :
  (∃ (c : ColorAssignment 2016), ValidColoring 2016 11 c) ∧
  (∀ k < 11, ¬ ∃ (c : ColorAssignment 2016), ValidColoring 2016 k c) :=
sorry

end NUMINAMATH_CALUDE_min_colors_2016_board_l3805_380545


namespace NUMINAMATH_CALUDE_complex_multiplication_l3805_380506

theorem complex_multiplication (z₁ z₂ : ℂ) (h₁ : z₁ = 1 - I) (h₂ : z₂ = 2 + I) :
  z₁ * z₂ = 3 - I := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3805_380506


namespace NUMINAMATH_CALUDE_orthogonal_vectors_sum_magnitude_l3805_380599

/-- Prove that given planar vectors a and b, where a and b are orthogonal, 
    a = (-1, 1), and |b| = 1, |a + 2b| = √6. -/
theorem orthogonal_vectors_sum_magnitude 
  (a b : ℝ × ℝ) 
  (h_orthogonal : a.1 * b.1 + a.2 * b.2 = 0)
  (h_a : a = (-1, 1))
  (h_b_norm : Real.sqrt (b.1^2 + b.2^2) = 1) :
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_sum_magnitude_l3805_380599


namespace NUMINAMATH_CALUDE_parabola_point_order_l3805_380562

/-- The parabola function -/
def f (x : ℝ) : ℝ := -2 * (x + 1)^2 - 1

/-- Point A -/
def A : ℝ × ℝ := (-3, f (-3))

/-- Point B -/
def B : ℝ × ℝ := (-2, f (-2))

/-- Point C -/
def C : ℝ × ℝ := (2, f 2)

theorem parabola_point_order :
  A.2 < B.2 ∧ C.2 < A.2 := by sorry

end NUMINAMATH_CALUDE_parabola_point_order_l3805_380562


namespace NUMINAMATH_CALUDE_dice_probability_l3805_380596

def red_die := Finset.range 6
def blue_die := Finset.range 6

def event_M (x : ℕ) : Prop := x % 3 = 0 ∧ x ∈ red_die
def event_N (x y : ℕ) : Prop := x + y > 8 ∧ x ∈ red_die ∧ y ∈ blue_die

def P_MN : ℚ := 5 / 36
def P_M : ℚ := 1 / 3

theorem dice_probability : (P_MN / P_M) = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l3805_380596


namespace NUMINAMATH_CALUDE_time_interval_for_population_change_l3805_380560

/-- Proves that given the specified birth and death rates and net population increase,
    the time interval is 2 seconds. -/
theorem time_interval_for_population_change (t : ℝ) : 
  (5 : ℝ) / t - (3 : ℝ) / t > 0 →  -- Ensure positive net change
  (5 - 3) * (86400 / t) = 86400 →
  t = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_interval_for_population_change_l3805_380560


namespace NUMINAMATH_CALUDE_upper_bound_of_expression_l3805_380519

theorem upper_bound_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  -1/(2*a) - 2/b ≤ -9/2 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 1 ∧ -1/(2*a₀) - 2/b₀ = -9/2 :=
by sorry

end NUMINAMATH_CALUDE_upper_bound_of_expression_l3805_380519


namespace NUMINAMATH_CALUDE_krtecek_return_distance_l3805_380502

/-- Represents a direction in 2D space -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a movement with distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Calculates the net displacement in centimeters for a list of movements -/
def netDisplacement (movements : List Movement) : ℝ × ℝ :=
  sorry

/-- Calculates the distance to the starting point given a net displacement -/
def distanceToStart (displacement : ℝ × ℝ) : ℝ :=
  sorry

/-- The list of Krteček's movements -/
def krtecekMovements : List Movement := [
  ⟨500, Direction.North⟩,
  ⟨230, Direction.West⟩,
  ⟨150, Direction.South⟩,
  ⟨370, Direction.West⟩,
  ⟨620, Direction.South⟩,
  ⟨53, Direction.East⟩,
  ⟨270, Direction.North⟩
]

theorem krtecek_return_distance :
  distanceToStart (netDisplacement krtecekMovements) = 547 := by
  sorry

end NUMINAMATH_CALUDE_krtecek_return_distance_l3805_380502


namespace NUMINAMATH_CALUDE_matthews_income_l3805_380585

/-- Represents the state income tax calculation function -/
def state_tax (q : ℝ) (income : ℝ) : ℝ :=
  0.01 * q * 50000 + 0.01 * (q + 3) * (income - 50000)

/-- Represents the condition that the total tax is (q + 0.5)% of the total income -/
def tax_condition (q : ℝ) (income : ℝ) : Prop :=
  state_tax q income = 0.01 * (q + 0.5) * income

/-- Theorem stating that given the tax calculation method and condition, 
    Matthew's annual income is $60000 -/
theorem matthews_income (q : ℝ) : 
  ∃ (income : ℝ), tax_condition q income ∧ income = 60000 := by
  sorry

end NUMINAMATH_CALUDE_matthews_income_l3805_380585


namespace NUMINAMATH_CALUDE_percentage_equality_l3805_380504

theorem percentage_equality (x y : ℝ) (h1 : 2.5 * x = 0.75 * y) (h2 : x = 20) : y = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l3805_380504


namespace NUMINAMATH_CALUDE_dog_cleaner_amount_l3805_380531

/-- The amount of cleaner used for a cat stain in ounces -/
def cat_cleaner : ℝ := 4

/-- The amount of cleaner used for a rabbit stain in ounces -/
def rabbit_cleaner : ℝ := 1

/-- The total amount of cleaner used for all stains in ounces -/
def total_cleaner : ℝ := 49

/-- The number of dog stains -/
def num_dogs : ℕ := 6

/-- The number of cat stains -/
def num_cats : ℕ := 3

/-- The number of rabbit stains -/
def num_rabbits : ℕ := 1

/-- The amount of cleaner used for a dog stain in ounces -/
def dog_cleaner : ℝ := 6

theorem dog_cleaner_amount :
  num_dogs * dog_cleaner + num_cats * cat_cleaner + num_rabbits * rabbit_cleaner = total_cleaner :=
by sorry

end NUMINAMATH_CALUDE_dog_cleaner_amount_l3805_380531


namespace NUMINAMATH_CALUDE_sugar_price_inflation_rate_l3805_380552

/-- Proves that the inflation rate is 12% given the conditions of sugar price increase --/
theorem sugar_price_inflation_rate 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (sugar_rate_increase : ℝ → ℝ) 
  (inflation_rate : ℝ) :
  initial_price = 25 →
  final_price = 33.0625 →
  (∀ x, sugar_rate_increase x = x + 0.03) →
  initial_price * (1 + sugar_rate_increase inflation_rate)^2 = final_price →
  inflation_rate = 0.12 := by
sorry

end NUMINAMATH_CALUDE_sugar_price_inflation_rate_l3805_380552


namespace NUMINAMATH_CALUDE_ceiling_tiling_count_l3805_380548

/-- Represents a rectangular region -/
structure Rectangle :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a tile -/
structure Tile :=
  (length : ℕ)
  (width : ℕ)

/-- Counts the number of ways to tile a rectangle with given tiles -/
def count_tilings (r : Rectangle) (t : Tile) : ℕ :=
  sorry

/-- Counts the number of ways to tile a rectangle with a beam -/
def count_tilings_with_beam (r : Rectangle) (t : Tile) (beam_pos : ℕ) : ℕ :=
  sorry

theorem ceiling_tiling_count :
  let ceiling := Rectangle.mk 6 4
  let tile := Tile.mk 2 1
  let beam_pos := 2
  count_tilings_with_beam ceiling tile beam_pos = 180 :=
sorry

end NUMINAMATH_CALUDE_ceiling_tiling_count_l3805_380548


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l3805_380565

theorem consecutive_integers_average (highest : ℕ) (h : highest = 36) :
  let set := List.range 7
  let numbers := set.map (λ i => highest - (6 - i))
  (numbers.sum : ℚ) / 7 = 33 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l3805_380565


namespace NUMINAMATH_CALUDE_power_equality_l3805_380589

theorem power_equality (y : ℝ) (h : (10 : ℝ) ^ (4 * y) = 100) : (10 : ℝ) ^ (y / 2) = (10 : ℝ) ^ (1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3805_380589


namespace NUMINAMATH_CALUDE_unique_divisor_with_remainders_l3805_380514

theorem unique_divisor_with_remainders :
  ∃! b : ℕ, b > 1 ∧ 826 % b = 7 ∧ 4373 % b = 8 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_divisor_with_remainders_l3805_380514


namespace NUMINAMATH_CALUDE_lcm_28_72_l3805_380507

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end NUMINAMATH_CALUDE_lcm_28_72_l3805_380507


namespace NUMINAMATH_CALUDE_signup_ways_eq_81_l3805_380541

/-- The number of interest groups available --/
def num_groups : ℕ := 3

/-- The number of students signing up --/
def num_students : ℕ := 4

/-- The number of ways students can sign up for interest groups --/
def signup_ways : ℕ := num_groups ^ num_students

/-- Theorem: The number of ways four students can sign up for one of three interest groups is 81 --/
theorem signup_ways_eq_81 : signup_ways = 81 := by
  sorry

end NUMINAMATH_CALUDE_signup_ways_eq_81_l3805_380541
