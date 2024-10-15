import Mathlib

namespace NUMINAMATH_CALUDE_dice_roll_probability_l1505_150529

/-- The probability of rolling a specific number on a standard six-sided die -/
def prob_single_roll : ℚ := 1 / 6

/-- The probability of not rolling a 1 on a single die -/
def prob_not_one : ℚ := 5 / 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (numbers between 10 and 20 inclusive) -/
def favorable_outcomes : ℕ := 11

theorem dice_roll_probability : 
  (1 : ℚ) - prob_not_one * prob_not_one = favorable_outcomes / total_outcomes := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1505_150529


namespace NUMINAMATH_CALUDE_y_sqrt_x_plus_one_l1505_150575

theorem y_sqrt_x_plus_one (x y k : ℝ) : 
  (y * (Real.sqrt x + 1) = k) →
  (x = 1 ∧ y = 5 → k = 10) ∧
  (y = 2 → x = 16) := by
sorry

end NUMINAMATH_CALUDE_y_sqrt_x_plus_one_l1505_150575


namespace NUMINAMATH_CALUDE_sum_of_squares_is_384_l1505_150521

/-- Represents the rates of cycling, jogging, and swimming -/
structure Rates where
  cycling : ℕ
  jogging : ℕ
  swimming : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (r : Rates) : Prop :=
  -- Rates are even
  r.cycling % 2 = 0 ∧ r.jogging % 2 = 0 ∧ r.swimming % 2 = 0 ∧
  -- Ed's distance equation
  3 * r.cycling + 4 * r.jogging + 2 * r.swimming = 88 ∧
  -- Sue's distance equation
  2 * r.cycling + 3 * r.jogging + 4 * r.swimming = 104

/-- The theorem to be proved -/
theorem sum_of_squares_is_384 :
  ∃ r : Rates, satisfies_conditions r ∧ 
    r.cycling^2 + r.jogging^2 + r.swimming^2 = 384 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_is_384_l1505_150521


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1505_150592

theorem quadratic_one_root (k : ℝ) : k > 0 ∧ (∃! x : ℝ, x^2 + 6*k*x + 9*k = 0) ↔ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1505_150592


namespace NUMINAMATH_CALUDE_extreme_value_condition_l1505_150564

/-- The cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a+6)

/-- Theorem stating the condition for f to have extreme values on ℝ -/
theorem extreme_value_condition (a : ℝ) : 
  (∃ x : ℝ, (f' a x = 0 ∧ ∀ y : ℝ, f' a y = 0 → y = x) → False) ↔ (a > 6 ∨ a < -3) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l1505_150564


namespace NUMINAMATH_CALUDE_sequence_properties_l1505_150504

def S (n : ℕ+) : ℤ := 3 * n - 2 * n^2

def a (n : ℕ+) : ℤ := -4 * n + 5

theorem sequence_properties :
  ∀ n : ℕ+,
  (∀ k : ℕ+, k ≤ n → S k - S (k-1) = a k) ∧
  S n ≥ n * a n :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1505_150504


namespace NUMINAMATH_CALUDE_combination_permutation_ratio_l1505_150534

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem combination_permutation_ratio (x y : ℕ) (h : y > x) :
  (binomial_coefficient y x : ℚ) / (binomial_coefficient (y + 2) x : ℚ) = 1 / 3 ∧
  (permutation y x : ℚ) / (binomial_coefficient y x : ℚ) = 24 ↔
  x = 4 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_combination_permutation_ratio_l1505_150534


namespace NUMINAMATH_CALUDE_distance_sum_between_22_and_23_l1505_150570

/-- Given points A, B, and D in a 2D plane, prove that the sum of distances AD and BD 
    is between 22 and 23. -/
theorem distance_sum_between_22_and_23 :
  let A : ℝ × ℝ := (15, 0)
  let B : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (6, 8)
  let distance (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  22 < distance A D + distance B D ∧ distance A D + distance B D < 23 := by
  sorry


end NUMINAMATH_CALUDE_distance_sum_between_22_and_23_l1505_150570


namespace NUMINAMATH_CALUDE_max_safe_sages_is_82_l1505_150569

/-- Represents a train with a given number of wagons. -/
structure Train :=
  (num_wagons : ℕ)

/-- Represents the journey details. -/
structure Journey :=
  (start_station : ℕ)
  (end_station : ℕ)
  (controller_start : ℕ)
  (controller_move_interval : ℕ)

/-- Represents the movement capabilities of sages. -/
structure SageMovement :=
  (max_move : ℕ)

/-- Represents the visibility range of sages. -/
structure SageVisibility :=
  (range : ℕ)

/-- Represents the maximum number of sages that can avoid controllers. -/
def max_safe_sages (t : Train) (j : Journey) (sm : SageMovement) (sv : SageVisibility) : ℕ :=
  82

/-- Theorem stating that 82 is the maximum number of sages that can avoid controllers. -/
theorem max_safe_sages_is_82 
  (t : Train) 
  (j : Journey) 
  (sm : SageMovement) 
  (sv : SageVisibility) : 
  max_safe_sages t j sm sv = 82 :=
by sorry

end NUMINAMATH_CALUDE_max_safe_sages_is_82_l1505_150569


namespace NUMINAMATH_CALUDE_greatest_common_divisor_and_sum_of_digits_l1505_150554

def numbers : List Nat := [23115, 34365, 83197, 153589]

def differences (nums : List Nat) : List Nat :=
  List.map (λ (pair : Nat × Nat) => pair.2 - pair.1) (List.zip nums (List.tail nums))

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem greatest_common_divisor_and_sum_of_digits :
  let diffs := differences numbers
  let n := diffs.foldl Nat.gcd (diffs.head!)
  n = 1582 ∧ sumOfDigits n = 16 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_and_sum_of_digits_l1505_150554


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1505_150517

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 2| + |x - 3| ≤ a) → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1505_150517


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_factorials_l1505_150533

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem greatest_prime_factor_of_sum_factorials :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (factorial 15 + factorial 18) ∧
  ∀ q : ℕ, Nat.Prime q → q ∣ (factorial 15 + factorial 18) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_factorials_l1505_150533


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l1505_150538

/-- The perimeter of a regular hexagon inscribed in a circle -/
theorem hexagon_perimeter (r : ℝ) (h : r = 10) : 
  6 * (2 * r * Real.sin (π / 6)) = 60 := by
  sorry

#check hexagon_perimeter

end NUMINAMATH_CALUDE_hexagon_perimeter_l1505_150538


namespace NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1505_150540

theorem lcm_from_product_and_hcf (a b : ℕ+) :
  a * b = 84942 ∧ Nat.gcd a b = 33 → Nat.lcm a b = 2574 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_hcf_l1505_150540


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l1505_150582

theorem cubic_root_sum_squares (p q r : ℝ) (x y z : ℝ) : 
  (x^3 - p*x^2 + q*x - r = 0) → 
  (y^3 - p*y^2 + q*y - r = 0) → 
  (z^3 - p*z^2 + q*z - r = 0) → 
  (x + y + z = p) →
  (x*y + x*z + y*z = q) →
  x^2 + y^2 + z^2 = p^2 - 2*q := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l1505_150582


namespace NUMINAMATH_CALUDE_cake_cutting_l1505_150532

theorem cake_cutting (cake_side : ℝ) (num_pieces : ℕ) : 
  cake_side = 15 → num_pieces = 9 → 
  ∃ (piece_side : ℝ), piece_side = 5 ∧ 
  cake_side = piece_side * Real.sqrt (num_pieces : ℝ) := by
sorry

end NUMINAMATH_CALUDE_cake_cutting_l1505_150532


namespace NUMINAMATH_CALUDE_squares_in_figure_50_l1505_150597

/-- The function representing the number of squares in the nth figure -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- Theorem stating that the 50th figure has 7651 squares -/
theorem squares_in_figure_50 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 50 = 7651 := by
  sorry

#eval f 50  -- This will evaluate f(50) and should output 7651

end NUMINAMATH_CALUDE_squares_in_figure_50_l1505_150597


namespace NUMINAMATH_CALUDE_return_speed_calculation_l1505_150576

theorem return_speed_calculation (total_distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) 
  (h1 : total_distance = 300) 
  (h2 : outbound_speed = 75) 
  (h3 : average_speed = 50) :
  ∃ inbound_speed : ℝ, 
    inbound_speed = 37.5 ∧ 
    average_speed = total_distance / (total_distance / (2 * outbound_speed) + total_distance / (2 * inbound_speed)) := by
  sorry

end NUMINAMATH_CALUDE_return_speed_calculation_l1505_150576


namespace NUMINAMATH_CALUDE_problem_solution_l1505_150519

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log x - x + 1

theorem problem_solution :
  (∃! a : ℝ, ∀ x > 0, f a x ≤ 0) ∧
  (∀ x ∈ Set.Ioo 0 (Real.pi / 2), Real.exp x * Real.sin x - x > f 1 x) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1505_150519


namespace NUMINAMATH_CALUDE_photo_arrangements_l1505_150562

/-- The number of male students -/
def num_male_students : ℕ := 4

/-- The number of female students -/
def num_female_students : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male_students + num_female_students

theorem photo_arrangements :
  /- (1) Arrangements with male student A at one of the ends -/
  (∃ (n : ℕ), n = 1440 ∧ 
    n = 2 * (Nat.factorial (total_students - 1))) ∧
  /- (2) Arrangements where female students B and C are not next to each other -/
  (∃ (m : ℕ), m = 3600 ∧ 
    m = (Nat.factorial (total_students - 2)) * (total_students * (total_students - 1) / 2)) ∧
  /- (3) Arrangements where female student B is not at the ends and C is not in the middle -/
  (∃ (k : ℕ), k = 3120 ∧ 
    k = (Nat.factorial (total_students - 2)) * 4 + (Nat.factorial (total_students - 2)) * 4 * 5) :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l1505_150562


namespace NUMINAMATH_CALUDE_zero_variance_median_equals_mean_l1505_150527

-- Define a sample as a finite multiset of real numbers
def Sample := Multiset ℝ

-- Define the variance of a sample
def variance (s : Sample) : ℝ := sorry

-- Define the median of a sample
def median (s : Sample) : ℝ := sorry

-- Define the mean of a sample
def mean (s : Sample) : ℝ := sorry

-- Theorem statement
theorem zero_variance_median_equals_mean (s : Sample) (a : ℝ) :
  variance s = 0 ∧ median s = a → mean s = a := by sorry

end NUMINAMATH_CALUDE_zero_variance_median_equals_mean_l1505_150527


namespace NUMINAMATH_CALUDE_solve_system_l1505_150528

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 3 * q = 10) 
  (eq2 : 3 * p + 5 * q = 20) : 
  q = 35 / 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1505_150528


namespace NUMINAMATH_CALUDE_eleventh_tenth_square_difference_l1505_150566

/-- The side length of the nth square in the sequence -/
def squareSideLength (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- The number of tiles in the nth square -/
def squareTiles (n : ℕ) : ℕ := (squareSideLength n) ^ 2

/-- The difference in tiles between the nth and (n-1)th squares -/
def tileDifference (n : ℕ) : ℕ := squareTiles n - squareTiles (n - 1)

theorem eleventh_tenth_square_difference :
  tileDifference 11 = 88 := by sorry

end NUMINAMATH_CALUDE_eleventh_tenth_square_difference_l1505_150566


namespace NUMINAMATH_CALUDE_divisor_between_l1505_150558

theorem divisor_between (n a b : ℕ) (hn : n > 8) (ha : a > 0) (hb : b > 0) 
  (hab : a < b) (hdiv_a : a ∣ n) (hdiv_b : b ∣ n) (heq : n = a^2 + b) : 
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end NUMINAMATH_CALUDE_divisor_between_l1505_150558


namespace NUMINAMATH_CALUDE_odd_digits_345_base5_l1505_150522

/-- Counts the number of odd digits in a base-5 number -/
def countOddDigitsBase5 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-5 -/
def toBase5 (n : ℕ) : ℕ := sorry

theorem odd_digits_345_base5 :
  countOddDigitsBase5 (toBase5 345) = 2 := by sorry

end NUMINAMATH_CALUDE_odd_digits_345_base5_l1505_150522


namespace NUMINAMATH_CALUDE_equation_seven_solutions_l1505_150507

-- Define the equation
def equation (a x : ℝ) : Prop :=
  Real.sin (Real.sqrt (a^2 - x^2 - 2*x - 1)) = 0.5

-- Define the number of distinct solutions
def has_seven_distinct_solutions (a : ℝ) : Prop :=
  ∃ (s : Finset ℝ), s.card = 7 ∧ (∀ x ∈ s, equation a x) ∧
    (∀ x : ℝ, equation a x → x ∈ s)

-- State the theorem
theorem equation_seven_solutions :
  ∀ a : ℝ, has_seven_distinct_solutions a ↔ a = 17 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_seven_solutions_l1505_150507


namespace NUMINAMATH_CALUDE_johns_butterfly_jars_l1505_150594

/-- The number of caterpillars in each jar -/
def caterpillars_per_jar : ℕ := 10

/-- The percentage of caterpillars that fail to become butterflies -/
def failure_rate : ℚ := 40 / 100

/-- The price of each butterfly in dollars -/
def price_per_butterfly : ℕ := 3

/-- The total amount made from selling butterflies in dollars -/
def total_amount : ℕ := 72

/-- The number of jars John has -/
def number_of_jars : ℕ := 4

theorem johns_butterfly_jars :
  let butterflies_per_jar := caterpillars_per_jar * (1 - failure_rate)
  let revenue_per_jar := butterflies_per_jar * price_per_butterfly
  total_amount / revenue_per_jar = number_of_jars := by sorry

end NUMINAMATH_CALUDE_johns_butterfly_jars_l1505_150594


namespace NUMINAMATH_CALUDE_product_xy_equals_one_l1505_150541

theorem product_xy_equals_one (x y : ℝ) 
  (h : (Real.sqrt (x^2 + 1) - x + 1) * (Real.sqrt (y^2 + 1) - y + 1) = 2) : 
  x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_xy_equals_one_l1505_150541


namespace NUMINAMATH_CALUDE_train_length_l1505_150531

/-- The length of a train given its speed, platform length, and crossing time --/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5 / 18) →
  platform_length = 230 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 290 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1505_150531


namespace NUMINAMATH_CALUDE_triangle_CSE_is_equilateral_l1505_150598

-- Define the circle k
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the chord AB
def Chord (k : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ k ∧ B ∈ k

-- Define the perpendicular bisector
def PerpendicularBisector (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X : ℝ × ℝ | (X.1 - P.1)^2 + (X.2 - P.2)^2 = (X.1 - Q.1)^2 + (X.2 - Q.2)^2}

-- Define the line through two points
def LineThroughPoints (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X : ℝ × ℝ | (X.2 - P.2) * (Q.1 - P.1) = (X.1 - P.1) * (Q.2 - P.2)}

theorem triangle_CSE_is_equilateral
  (k : Set (ℝ × ℝ))
  (r : ℝ)
  (A B S C D E : ℝ × ℝ)
  (h1 : k = Circle (0, 0) r)
  (h2 : Chord k A B)
  (h3 : S ∈ LineThroughPoints A B)
  (h4 : (S.1 - A.1)^2 + (S.2 - A.2)^2 = r^2)
  (h5 : (B.1 - A.1)^2 + (B.2 - A.2)^2 > r^2)
  (h6 : C ∈ k ∧ C ∈ PerpendicularBisector B S)
  (h7 : D ∈ k ∧ D ∈ PerpendicularBisector B S)
  (h8 : E ∈ k ∧ E ∈ LineThroughPoints D S) :
  (C.1 - S.1)^2 + (C.2 - S.2)^2 = (C.1 - E.1)^2 + (C.2 - E.2)^2 ∧
  (C.1 - S.1)^2 + (C.2 - S.2)^2 = (E.1 - S.1)^2 + (E.2 - S.2)^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_CSE_is_equilateral_l1505_150598


namespace NUMINAMATH_CALUDE_segment_point_difference_l1505_150565

/-- Given a line segment PQ with endpoints P(6,-2) and Q(-3,10), and a point R(a,b) on PQ such that
    the distance from P to R is one-third the distance from P to Q, prove that b-a = -1. -/
theorem segment_point_difference (a b : ℝ) : 
  let p : ℝ × ℝ := (6, -2)
  let q : ℝ × ℝ := (-3, 10)
  let r : ℝ × ℝ := (a, b)
  (r.1 - p.1) / (q.1 - p.1) = (r.2 - p.2) / (q.2 - p.2) ∧  -- R is on PQ
  (r.1 - p.1)^2 + (r.2 - p.2)^2 = (1/9) * ((q.1 - p.1)^2 + (q.2 - p.2)^2) -- PR = (1/3)PQ
  →
  b - a = -1 := by
sorry

end NUMINAMATH_CALUDE_segment_point_difference_l1505_150565


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l1505_150588

/-- Given that points (-3,-1) and (4,-6) are on opposite sides of the line 3x-2y-a=0,
    the range of values for a is (-7, 24). -/
theorem opposite_sides_line_range (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ = -3 ∧ y₁ = -1 ∧ x₂ = 4 ∧ y₂ = -6 ∧ 
    (3*x₁ - 2*y₁ - a) * (3*x₂ - 2*y₂ - a) < 0) ↔ 
  -7 < a ∧ a < 24 :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l1505_150588


namespace NUMINAMATH_CALUDE_fixed_point_symmetric_coordinates_l1505_150500

/-- Given a line that always passes through a fixed point P and P is symmetric about x + y = 0, prove P's coordinates -/
theorem fixed_point_symmetric_coordinates :
  ∀ (k : ℝ), 
  (∃ (P : ℝ × ℝ), ∀ (x y : ℝ), k * x - y + k - 2 = 0 → (x, y) = P) →
  (∃ (P' : ℝ × ℝ), 
    (P'.1 + P'.2 = 0) ∧ 
    (P'.1 - P.1)^2 + (P'.2 - P.2)^2 = 2 * ((P.1 + P.2) / 2)^2) →
  P = (2, 1) :=
by sorry


end NUMINAMATH_CALUDE_fixed_point_symmetric_coordinates_l1505_150500


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_81_l1505_150579

theorem factor_t_squared_minus_81 (t : ℝ) : t^2 - 81 = (t - 9) * (t + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_81_l1505_150579


namespace NUMINAMATH_CALUDE_chessboard_tiling_impossible_l1505_150530

/-- Represents a chessboard tile -/
inductive Tile
| L
| T

/-- Represents the color distribution of a tile placement -/
structure ColorDistribution :=
  (black : ℕ)
  (white : ℕ)

/-- The color distribution of an L-tile -/
def l_tile_distribution : ColorDistribution :=
  ⟨2, 2⟩

/-- The possible color distributions of a T-tile -/
def t_tile_distributions : List ColorDistribution :=
  [⟨3, 1⟩, ⟨1, 3⟩]

/-- The number of squares on the chessboard -/
def chessboard_squares : ℕ := 64

/-- The number of black squares on the chessboard -/
def chessboard_black_squares : ℕ := 32

/-- The number of white squares on the chessboard -/
def chessboard_white_squares : ℕ := 32

/-- The number of L-tiles -/
def num_l_tiles : ℕ := 15

/-- The number of T-tiles -/
def num_t_tiles : ℕ := 1

theorem chessboard_tiling_impossible :
  ∀ (t_dist : ColorDistribution),
    t_dist ∈ t_tile_distributions →
    (num_l_tiles * l_tile_distribution.black + t_dist.black ≠ chessboard_black_squares ∨
     num_l_tiles * l_tile_distribution.white + t_dist.white ≠ chessboard_white_squares) :=
by sorry

end NUMINAMATH_CALUDE_chessboard_tiling_impossible_l1505_150530


namespace NUMINAMATH_CALUDE_incorrect_operator_is_second_l1505_150535

def original_expression : List Int := [3, 5, -7, 9, -11, 13, -15, 17]

def calculate (expr : List Int) : Int :=
  expr.foldl (· + ·) 0

def flip_operator (expr : List Int) (index : Nat) : List Int :=
  expr.mapIdx (fun i x => if i == index then -x else x)

theorem incorrect_operator_is_second :
  ∃ (i : Nat), i < original_expression.length ∧
    calculate (flip_operator original_expression i) = -4 ∧
    i = 1 ∧
    ∀ (j : Nat), j < original_expression.length → j ≠ i →
      calculate (flip_operator original_expression j) ≠ -4 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_operator_is_second_l1505_150535


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1505_150525

theorem fraction_subtraction_simplification :
  ∃ (a b : ℚ), a = 9/19 ∧ b = 5/57 ∧ a - b = 22/57 ∧ (∀ (c d : ℤ), c ≠ 0 → 22/57 = c/d → (c = 22 ∧ d = 57 ∨ c = -22 ∧ d = -57)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1505_150525


namespace NUMINAMATH_CALUDE_cubic_factorization_l1505_150550

theorem cubic_factorization (x : ℝ) : x^3 - 16*x = x*(x+4)*(x-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1505_150550


namespace NUMINAMATH_CALUDE_largest_reciprocal_l1505_150567

theorem largest_reciprocal (a b c d e : ℝ) (ha : a = 1/4) (hb : b = 3/8) (hc : c = 1/2) (hd : d = 4) (he : e = 1000) :
  (1 / a > 1 / b) ∧ (1 / a > 1 / c) ∧ (1 / a > 1 / d) ∧ (1 / a > 1 / e) := by
  sorry

#check largest_reciprocal

end NUMINAMATH_CALUDE_largest_reciprocal_l1505_150567


namespace NUMINAMATH_CALUDE_pump_fill_time_l1505_150509

/-- The time it takes for the pump to fill the tank without the leak -/
def pump_time : ℝ := 2

/-- The time it takes for the pump and leak together to fill the tank -/
def combined_time : ℝ := 2.8

/-- The time it takes for the leak to empty the full tank -/
def leak_time : ℝ := 7

theorem pump_fill_time :
  (1 / pump_time) - (1 / leak_time) = (1 / combined_time) :=
by sorry

end NUMINAMATH_CALUDE_pump_fill_time_l1505_150509


namespace NUMINAMATH_CALUDE_colored_points_theorem_l1505_150571

theorem colored_points_theorem (r b g : ℕ) (d_rb d_rg d_bg : ℝ) : 
  r + b + g = 15 →
  (r : ℝ) * (b : ℝ) * d_rb = 51 →
  (r : ℝ) * (g : ℝ) * d_rg = 39 →
  (b : ℝ) * (g : ℝ) * d_bg = 1 →
  d_rb > 0 →
  d_rg > 0 →
  d_bg > 0 →
  ((r = 13 ∧ b = 1 ∧ g = 1) ∨ (r = 8 ∧ b = 4 ∧ g = 3)) := by
sorry

end NUMINAMATH_CALUDE_colored_points_theorem_l1505_150571


namespace NUMINAMATH_CALUDE_max_salary_specific_team_l1505_150514

/-- Represents a basketball team -/
structure BasketballTeam where
  players : Nat
  minSalary : Nat
  salaryCap : Nat

/-- Calculates the maximum possible salary for a single player in a basketball team -/
def maxPlayerSalary (team : BasketballTeam) : Nat :=
  team.salaryCap - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player
    in a specific basketball team configuration -/
theorem max_salary_specific_team :
  let team : BasketballTeam := {
    players := 25,
    minSalary := 18000,
    salaryCap := 900000
  }
  maxPlayerSalary team = 468000 := by
  sorry

#eval maxPlayerSalary {players := 25, minSalary := 18000, salaryCap := 900000}

end NUMINAMATH_CALUDE_max_salary_specific_team_l1505_150514


namespace NUMINAMATH_CALUDE_balance_forces_l1505_150503

/-- A force is represented by a pair of real numbers -/
def Force : Type := ℝ × ℝ

/-- Addition of forces -/
def add_forces (f1 f2 : Force) : Force :=
  (f1.1 + f2.1, f1.2 + f2.2)

/-- The zero force -/
def zero_force : Force := (0, 0)

/-- Two forces are balanced by a third force if their sum is the zero force -/
def balances (f1 f2 f3 : Force) : Prop :=
  add_forces (add_forces f1 f2) f3 = zero_force

theorem balance_forces :
  let f1 : Force := (1, 1)
  let f2 : Force := (2, 3)
  let f3 : Force := (-3, -4)
  balances f1 f2 f3 := by sorry

end NUMINAMATH_CALUDE_balance_forces_l1505_150503


namespace NUMINAMATH_CALUDE_system_solution_l1505_150568

theorem system_solution (x y : ℝ) (eq1 : 2 * x + y = 6) (eq2 : x + 2 * y = 3) : x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1505_150568


namespace NUMINAMATH_CALUDE_remaining_bottles_l1505_150581

theorem remaining_bottles (small_initial big_initial : ℕ) 
  (small_percent big_percent : ℚ) : 
  small_initial = 6000 →
  big_initial = 10000 →
  small_percent = 12 / 100 →
  big_percent = 15 / 100 →
  (small_initial - small_initial * small_percent).floor +
  (big_initial - big_initial * big_percent).floor = 13780 := by
  sorry

end NUMINAMATH_CALUDE_remaining_bottles_l1505_150581


namespace NUMINAMATH_CALUDE_total_movie_time_is_172_l1505_150524

/-- Represents a segment of movie watching, including the time spent watching and rewinding --/
structure MovieSegment where
  watchTime : ℕ
  rewindTime : ℕ

/-- Calculates the total time for a movie segment --/
def segmentTime (segment : MovieSegment) : ℕ :=
  segment.watchTime + segment.rewindTime

/-- The sequence of movie segments as described in the problem --/
def movieSegments : List MovieSegment := [
  ⟨30, 5⟩,
  ⟨20, 7⟩,
  ⟨10, 12⟩,
  ⟨15, 8⟩,
  ⟨25, 15⟩,
  ⟨15, 10⟩
]

/-- Theorem stating that the total time to watch the movie is 172 minutes --/
theorem total_movie_time_is_172 :
  (movieSegments.map segmentTime).sum = 172 := by
  sorry

end NUMINAMATH_CALUDE_total_movie_time_is_172_l1505_150524


namespace NUMINAMATH_CALUDE_angstadt_student_count_l1505_150556

/-- Given that:
  1. Half of Mr. Angstadt's students are enrolled in Statistics.
  2. 90% of the students in Statistics are seniors.
  3. There are 54 seniors enrolled in Statistics.
  Prove that Mr. Angstadt has 120 students throughout the school day. -/
theorem angstadt_student_count :
  ∀ (total_students stats_students seniors : ℕ),
  stats_students = total_students / 2 →
  seniors = (90 * stats_students) / 100 →
  seniors = 54 →
  total_students = 120 :=
by sorry

end NUMINAMATH_CALUDE_angstadt_student_count_l1505_150556


namespace NUMINAMATH_CALUDE_exists_problem_solved_by_all_l1505_150577

/-- Represents a problem on the exam -/
def Problem : Type := ℕ

/-- Represents a student in the class -/
def Student : Type := ℕ

/-- Given n students and 2^(n-1) problems, if for each pair of distinct problems
    there is at least one student who has solved both and at least one student
    who has solved one but not the other, then there exists a problem solved by
    all n students. -/
theorem exists_problem_solved_by_all
  (n : ℕ)
  (problems : Finset Problem)
  (students : Finset Student)
  (solved : Problem → Student → Prop)
  (h_num_students : students.card = n)
  (h_num_problems : problems.card = 2^(n-1))
  (h_solved_both : ∀ p q : Problem, p ≠ q →
    ∃ s : Student, solved p s ∧ solved q s)
  (h_solved_one_not_other : ∀ p q : Problem, p ≠ q →
    ∃ s : Student, (solved p s ∧ ¬solved q s) ∨ (solved q s ∧ ¬solved p s)) :
  ∃ p : Problem, ∀ s : Student, solved p s :=
sorry

end NUMINAMATH_CALUDE_exists_problem_solved_by_all_l1505_150577


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1505_150589

theorem complex_equation_solution (z : ℂ) : 4 + 2 * Complex.I * z = 2 - 6 * Complex.I * z ↔ z = Complex.I / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1505_150589


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l1505_150560

/-- Converts a base 2 number to base 10 --/
def binary_to_decimal (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

/-- Converts a base 10 number to base 8 --/
def decimal_to_octal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_to_octal_conversion :
  decimal_to_octal (binary_to_decimal [true, true, false, true, true, false, true, true, false]) = [6, 6, 6] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l1505_150560


namespace NUMINAMATH_CALUDE_distinct_z_values_l1505_150526

def is_valid_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def swap_digits (x : ℕ) : ℕ :=
  let a := x / 100
  let b := (x / 10) % 10
  let c := x % 10
  100 * b + 10 * a + c

def z (x : ℕ) : ℕ := Int.natAbs (x - swap_digits x)

theorem distinct_z_values (x : ℕ) (hx : is_valid_number x) : 
  ∃ (S : Finset ℕ), (∀ n, n ∈ S ↔ ∃ y, is_valid_number y ∧ z y = n) ∧ Finset.card S = 9 :=
sorry

end NUMINAMATH_CALUDE_distinct_z_values_l1505_150526


namespace NUMINAMATH_CALUDE_sum_divisible_by_143_l1505_150578

theorem sum_divisible_by_143 : ∃ k : ℕ, (1000 * 1001) / 2 = 143 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_143_l1505_150578


namespace NUMINAMATH_CALUDE_lune_area_minus_square_l1505_150512

theorem lune_area_minus_square (r1 r2 s : ℝ) : r1 = 2 → r2 = 1 → s = 1 →
  (π * r1^2 / 2 - π * r2^2 / 2) - s^2 = 3 * π / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_lune_area_minus_square_l1505_150512


namespace NUMINAMATH_CALUDE_f_monotone_increasing_min_m_value_l1505_150555

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1/x) * log x

theorem f_monotone_increasing :
  StrictMono f := by sorry

theorem min_m_value (m : ℝ) :
  (∀ x > 0, (2 * f x - m) / (exp (m * x)) ≤ m) ↔ m ≥ 2/exp 1 := by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_min_m_value_l1505_150555


namespace NUMINAMATH_CALUDE_cube_volume_doubling_l1505_150506

theorem cube_volume_doubling (a : ℝ) (h : a > 0) :
  (2 * a) ^ 3 = 8 * a ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_doubling_l1505_150506


namespace NUMINAMATH_CALUDE_integer_root_theorem_l1505_150557

def polynomial (x b : ℤ) : ℤ := x^3 + 4*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

def valid_b_values : Set ℤ := {-177, -62, -35, -25, -18, -17, 9, 16, 27, 48, 144, 1296}

theorem integer_root_theorem :
  ∀ b : ℤ, has_integer_root b ↔ b ∈ valid_b_values :=
by sorry

end NUMINAMATH_CALUDE_integer_root_theorem_l1505_150557


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_circles_l1505_150515

/-- The area of the shaded region formed by the intersection of a rectangle and two circles -/
theorem shaded_area_rectangle_circles (rectangle_width : ℝ) (rectangle_height : ℝ) (circle_radius : ℝ) : 
  rectangle_width = 12 →
  rectangle_height = 10 →
  circle_radius = 3 →
  let rectangle_area := rectangle_width * rectangle_height
  let circle_area := π * circle_radius^2
  let shaded_area := rectangle_area - 2 * circle_area
  shaded_area = 120 - 18 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_circles_l1505_150515


namespace NUMINAMATH_CALUDE_equal_interest_l1505_150584

def interest_rate_1_1 : ℝ := 0.07
def interest_rate_1_2 : ℝ := 0.10
def interest_rate_2_1 : ℝ := 0.05
def interest_rate_2_2 : ℝ := 0.12

def principal_1 : ℝ := 600
def principal_2 : ℝ := 800

def time_1_1 : ℕ := 3
def time_2_1 : ℕ := 2
def time_2_2 : ℕ := 3

def total_interest_2 : ℝ := principal_2 * interest_rate_2_1 * time_2_1 + principal_2 * interest_rate_2_2 * time_2_2

theorem equal_interest (n : ℕ) : 
  principal_1 * interest_rate_1_1 * time_1_1 + principal_1 * interest_rate_1_2 * n = total_interest_2 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_equal_interest_l1505_150584


namespace NUMINAMATH_CALUDE_cuts_through_examples_l1505_150510

/-- A line cuts through a curve at a point if it's tangent to the curve at that point
    and the curve lies on both sides of the line near that point. -/
def cuts_through (l : ℝ → ℝ) (c : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  (∀ x, l x = c x → x = p.1) ∧  -- l is tangent to c at p
  (∃ ε > 0, ∀ x, |x - p.1| < ε → 
    ((c x > l x ∧ x < p.1) ∨ (c x < l x ∧ x > p.1) ∨
     (c x < l x ∧ x < p.1) ∨ (c x > l x ∧ x > p.1)))

theorem cuts_through_examples :
  (cuts_through (λ _ ↦ 0) (λ x ↦ x^3) (0, 0)) ∧
  (cuts_through (λ x ↦ x) Real.sin (0, 0)) ∧
  (cuts_through (λ x ↦ x) Real.tan (0, 0)) :=
sorry

end NUMINAMATH_CALUDE_cuts_through_examples_l1505_150510


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1505_150545

theorem quadratic_equation_solution (y : ℝ) : 
  y^2 + 6*y + 8 = -(y + 4)*(y + 6) ↔ y = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1505_150545


namespace NUMINAMATH_CALUDE_greatest_x_value_l1505_150505

theorem greatest_x_value (x : ℤ) (h : (3.134 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 31000) :
  x ≤ 3 ∧ ∃ y : ℤ, y > 3 → (3.134 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 31000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1505_150505


namespace NUMINAMATH_CALUDE_sum_of_squares_l1505_150549

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 2*b = 7)
  (eq2 : b^2 + 4*c = -7)
  (eq3 : c^2 + 6*a = -14) :
  a^2 + b^2 + c^2 = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1505_150549


namespace NUMINAMATH_CALUDE_part_one_part_two_l1505_150539

noncomputable section

open Real

-- Define the function f
def f (a b x : ℝ) : ℝ := a * log x - b * x^2

-- Define the tangent line equation
def tangent_line (x : ℝ) : ℝ := -3 * x + 2 * log 2 + 2

-- Part 1: Prove that a = 2 and b = 1
theorem part_one : 
  ∀ a b : ℝ, (∀ x : ℝ, f a b x = f a b 2 + (x - 2) * (-3)) → a = 2 ∧ b = 1 := 
by sorry

-- Define the function h for part 2
def h (x m : ℝ) : ℝ := 2 * log x - x^2 + m

-- Part 2: Prove the range of m
theorem part_two : 
  ∀ m : ℝ, (∃ x y : ℝ, 1/exp 1 ≤ x ∧ x < y ∧ y ≤ exp 1 ∧ h x m = 0 ∧ h y m = 0) 
  → 1 < m ∧ m ≤ 1/(exp 1)^2 + 2 := 
by sorry

end

end NUMINAMATH_CALUDE_part_one_part_two_l1505_150539


namespace NUMINAMATH_CALUDE_shop_b_better_l1505_150580

/-- Represents a costume rental shop -/
structure Shop where
  name : String
  base_price : ℕ
  discount_rate : ℚ
  discount_threshold : ℕ
  additional_discount : ℕ

/-- Calculates the number of sets that can be rented from a shop given a budget -/
def sets_rentable (shop : Shop) (budget : ℕ) : ℚ :=
  if budget / shop.base_price > shop.discount_threshold
  then (budget + shop.additional_discount) / (shop.base_price * (1 - shop.discount_rate))
  else budget / shop.base_price

/-- The main theorem proving Shop B offers more sets than Shop A -/
theorem shop_b_better (shop_a shop_b : Shop) (budget : ℕ) :
  shop_a.name = "A" →
  shop_b.name = "B" →
  shop_b.base_price = shop_a.base_price + 10 →
  400 / shop_a.base_price = 500 / shop_b.base_price →
  shop_b.discount_rate = 1/5 →
  shop_b.discount_threshold = 100 →
  shop_b.additional_discount = 200 →
  budget = 5000 →
  sets_rentable shop_b budget > sets_rentable shop_a budget :=
by
  sorry

end NUMINAMATH_CALUDE_shop_b_better_l1505_150580


namespace NUMINAMATH_CALUDE_teal_color_survey_l1505_150553

theorem teal_color_survey (total : ℕ) (green : ℕ) (both : ℕ) (neither : ℕ) :
  total = 150 →
  green = 90 →
  both = 40 →
  neither = 20 →
  ∃ blue : ℕ, blue = 80 ∧ blue + green - both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_teal_color_survey_l1505_150553


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1505_150552

-- Problem 1
theorem simplify_expression_1 (a : ℝ) : 2 * a^2 - 3*a - 5*a^2 + 6*a = -3*a^2 + 3*a := by
  sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) : 2*(a-1) - (2*a-3) + 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1505_150552


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l1505_150585

theorem smallest_five_digit_multiple_of_18 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 18 ∣ n → 10008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_18_l1505_150585


namespace NUMINAMATH_CALUDE_min_correct_answers_l1505_150544

/-- Represents the scoring system and conditions of the IQ test -/
structure IQTest where
  total_questions : ℕ
  correct_points : ℕ
  wrong_points : ℕ
  unanswered : ℕ
  min_score : ℕ

/-- Calculates the score based on the number of correct answers -/
def calculate_score (test : IQTest) (correct_answers : ℕ) : ℤ :=
  (correct_answers : ℤ) * test.correct_points - 
  ((test.total_questions - test.unanswered - correct_answers) : ℤ) * test.wrong_points

/-- Theorem stating the minimum number of correct answers needed to achieve the minimum score -/
theorem min_correct_answers (test : IQTest) : 
  test.total_questions = 20 ∧ 
  test.correct_points = 5 ∧ 
  test.wrong_points = 2 ∧ 
  test.unanswered = 2 ∧ 
  test.min_score = 60 →
  (∀ x : ℕ, x < 14 → calculate_score test x < test.min_score) ∧
  calculate_score test 14 ≥ test.min_score := by
  sorry


end NUMINAMATH_CALUDE_min_correct_answers_l1505_150544


namespace NUMINAMATH_CALUDE_f_fixed_points_l1505_150511

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem f_fixed_points : {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_f_fixed_points_l1505_150511


namespace NUMINAMATH_CALUDE_impossibility_of_forming_parallelepiped_l1505_150561

/-- Represents the dimensions of a rectangular parallelepiped -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Checks if a parallelepiped can be formed from smaller parallelepipeds -/
def can_form_parallelepiped (large : Dimensions) (small : Dimensions) : Prop :=
  ∃ (n : ℕ), 
    n * (small.length * small.width * small.height) = large.length * large.width * large.height ∧
    ∀ (face : ℕ), face ∈ 
      [large.length * large.width, large.width * large.height, large.length * large.height] →
      ∃ (a b : ℕ), a * small.length * small.width + b * small.length * small.height + 
                   (n - a - b) * small.width * small.height = face

theorem impossibility_of_forming_parallelepiped : 
  ¬ can_form_parallelepiped 
    (Dimensions.mk 3 4 5) 
    (Dimensions.mk 2 2 1) := by
  sorry

end NUMINAMATH_CALUDE_impossibility_of_forming_parallelepiped_l1505_150561


namespace NUMINAMATH_CALUDE_money_split_l1505_150587

theorem money_split (total : ℝ) (ratio_small : ℕ) (ratio_large : ℕ) (smaller_share : ℝ) :
  total = 125 →
  ratio_small = 2 →
  ratio_large = 3 →
  smaller_share = (ratio_small : ℝ) / ((ratio_small : ℝ) + (ratio_large : ℝ)) * total →
  smaller_share = 50 := by
sorry

end NUMINAMATH_CALUDE_money_split_l1505_150587


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1505_150599

theorem inequality_solution_set (x : ℝ) :
  x ≠ -2 ∧ x ≠ 9/2 →
  ((x + 1) / (x + 2) > (3 * x + 4) / (2 * x + 9)) ↔
  (-9/2 ≤ x ∧ x ≤ -2) ∨ ((1 - Real.sqrt 5) / 2 < x ∧ x < (1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1505_150599


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l1505_150551

/-- The number of friends Alex has -/
def num_friends : ℕ := 15

/-- The initial number of coins Alex has -/
def initial_coins : ℕ := 100

/-- The minimum number of coins needed to distribute to friends -/
def min_coins_needed : ℕ := (num_friends * (num_friends + 1)) / 2

/-- The number of additional coins needed -/
def additional_coins_needed : ℕ := min_coins_needed - initial_coins

theorem alex_coin_distribution :
  additional_coins_needed = 20 := by sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l1505_150551


namespace NUMINAMATH_CALUDE_kate_emily_hair_ratio_l1505_150516

/-- The ratio of hair lengths -/
def hair_length_ratio (kate_length emily_length : ℕ) : ℚ :=
  kate_length / emily_length

/-- Theorem stating the ratio of Kate's hair length to Emily's hair length -/
theorem kate_emily_hair_ratio :
  let logan_length : ℕ := 20
  let emily_length : ℕ := logan_length + 6
  let kate_length : ℕ := 7
  hair_length_ratio kate_length emily_length = 7 / 26 := by
sorry

end NUMINAMATH_CALUDE_kate_emily_hair_ratio_l1505_150516


namespace NUMINAMATH_CALUDE_selfie_count_l1505_150559

theorem selfie_count (last_year this_year : ℕ) : 
  (this_year : ℚ) / last_year = 17 / 10 →
  this_year - last_year = 630 →
  last_year + this_year = 2430 :=
by
  sorry

end NUMINAMATH_CALUDE_selfie_count_l1505_150559


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1505_150572

theorem parallel_vectors_x_value (x : ℝ) 
  (h1 : x > 0) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (h2 : a = (8 + x/2, x)) 
  (h3 : b = (x + 1, 2)) 
  (h4 : ∃ (k : ℝ), a = k • b) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1505_150572


namespace NUMINAMATH_CALUDE_binomial_coefficient_18_8_l1505_150543

theorem binomial_coefficient_18_8 (h1 : Nat.choose 16 6 = 8008)
                                  (h2 : Nat.choose 16 7 = 11440)
                                  (h3 : Nat.choose 16 8 = 12870) :
  Nat.choose 18 8 = 43758 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_18_8_l1505_150543


namespace NUMINAMATH_CALUDE_m_squared_divisors_l1505_150523

/-- A number with exactly 4 divisors -/
def HasFourDivisors (m : ℕ) : Prop :=
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 4

/-- The number of divisors of a natural number -/
def NumberOfDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem m_squared_divisors (m : ℕ) (h : HasFourDivisors m) : 
  NumberOfDivisors (m^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_m_squared_divisors_l1505_150523


namespace NUMINAMATH_CALUDE_watch_sale_loss_percentage_l1505_150596

/-- Proves that the loss percentage is 10% for a watch sale scenario --/
theorem watch_sale_loss_percentage 
  (cost_price : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 1200)
  (h2 : selling_price < cost_price)
  (h3 : selling_price + 180 = cost_price * 1.05) :
  (cost_price - selling_price) / cost_price * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_watch_sale_loss_percentage_l1505_150596


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1505_150518

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), x = 2 + (35 / 100) * (1 / (1 - 1/100)^n)) →
  x = 233 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l1505_150518


namespace NUMINAMATH_CALUDE_perfect_square_properties_l1505_150548

theorem perfect_square_properties (a : ℕ) :
  (∀ n : ℕ, n > 0 → a ∈ ({1, 2, 4} : Set ℕ) → ¬∃ m : ℕ, n * (a + n) = m^2) ∧
  ((∃ k : ℕ, k ≥ 3 ∧ a = 2^k) → ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, n * (a + n) = m^2) ∧
  (a ∉ ({1, 2, 4} : Set ℕ) → ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, n * (a + n) = m^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_properties_l1505_150548


namespace NUMINAMATH_CALUDE_set_operations_l1505_150595

-- Define the universal set U
def U : Set ℕ := {n : ℕ | n % 2 = 0 ∧ n ≤ 10}

-- Define set A
def A : Set ℕ := {0, 2, 4, 6}

-- Define set B
def B : Set ℕ := {x : ℕ | x ∈ A ∧ x < 4}

theorem set_operations :
  (Set.compl A) = {8, 10} ∧
  (A ∩ (Set.compl B)) = {4, 6} := by
  sorry

#check set_operations

end NUMINAMATH_CALUDE_set_operations_l1505_150595


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l1505_150513

theorem angle_sum_around_point (y : ℝ) : 
  (6*y + 3*y + 4*y + 2*y = 360) → y = 24 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l1505_150513


namespace NUMINAMATH_CALUDE_alexander_pencil_count_l1505_150520

/-- The number of pencils Alexander uses for all exhibitions -/
def total_pencils (initial_pictures : ℕ) (new_galleries : ℕ) (pictures_per_new_gallery : ℕ) 
  (pencils_per_picture : ℕ) (pencils_for_signing : ℕ) : ℕ :=
  let total_pictures := initial_pictures + new_galleries * pictures_per_new_gallery
  let pencils_for_drawing := total_pictures * pencils_per_picture
  let total_exhibitions := 1 + new_galleries
  let pencils_for_all_signings := total_exhibitions * pencils_for_signing
  pencils_for_drawing + pencils_for_all_signings

/-- Theorem stating that Alexander uses 88 pencils in total -/
theorem alexander_pencil_count : 
  total_pencils 9 5 2 4 2 = 88 := by
  sorry


end NUMINAMATH_CALUDE_alexander_pencil_count_l1505_150520


namespace NUMINAMATH_CALUDE_range_of_m_l1505_150574

-- Define the propositions p and q
def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the condition that ¬p is necessary but not sufficient for ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  (∀ x, ¬(q x m) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q x m)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (m > 0 ∧ not_p_necessary_not_sufficient_for_not_q m) ↔ m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1505_150574


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l1505_150590

/-- Proves that mixing 100 mL of 10% alcohol solution with 300 mL of 30% alcohol solution
    results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 100
  let x_concentration : ℝ := 0.1
  let y_volume : ℝ := 300
  let y_concentration : ℝ := 0.3
  let target_concentration : ℝ := 0.25
  let total_volume := x_volume + y_volume
  let total_alcohol := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = target_concentration := by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l1505_150590


namespace NUMINAMATH_CALUDE_proposition_p_q_range_l1505_150537

theorem proposition_p_q_range (x a : ℝ) : 
  (∀ x, x^2 ≤ 5*x - 4 → x^2 - (a + 2)*x + 2*a ≤ 0) ∧ 
  (∃ x, x^2 ≤ 5*x - 4 ∧ x^2 - (a + 2)*x + 2*a > 0) →
  1 ≤ a ∧ a ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_proposition_p_q_range_l1505_150537


namespace NUMINAMATH_CALUDE_triangle_inequality_l1505_150573

theorem triangle_inequality (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^3 / c^3 + b^3 / c^3 + 3 * a * b / c^2 > 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1505_150573


namespace NUMINAMATH_CALUDE_manuscript_revision_problem_l1505_150583

/-- The number of pages revised twice in a manuscript -/
def pages_revised_twice (total_pages : ℕ) (pages_revised_once : ℕ) (total_cost : ℕ) : ℕ :=
  (total_cost - (5 * total_pages + 3 * pages_revised_once)) / 6

/-- Theorem stating the number of pages revised twice -/
theorem manuscript_revision_problem (total_pages : ℕ) (pages_revised_once : ℕ) (total_cost : ℕ) 
  (h1 : total_pages = 200)
  (h2 : pages_revised_once = 80)
  (h3 : total_cost = 1360) :
  pages_revised_twice total_pages pages_revised_once total_cost = 20 := by
sorry

#eval pages_revised_twice 200 80 1360

end NUMINAMATH_CALUDE_manuscript_revision_problem_l1505_150583


namespace NUMINAMATH_CALUDE_inequality_proof_l1505_150508

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1505_150508


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l1505_150502

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- The original point P -/
def P : ℝ × ℝ := (3, -5)

theorem reflection_across_y_axis :
  reflect_y_axis P = (-3, -5) := by sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l1505_150502


namespace NUMINAMATH_CALUDE_max_min_product_l1505_150542

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 15) (hprod : x*y + y*z + z*x = 45) :
  ∃ (m : ℝ), m = min (x*y) (min (y*z) (z*x)) ∧ m ≤ 17.5 ∧
  ∀ (m' : ℝ), m' = min (x*y) (min (y*z) (z*x)) → m' ≤ 17.5 := by
sorry

end NUMINAMATH_CALUDE_max_min_product_l1505_150542


namespace NUMINAMATH_CALUDE_smallest_x_value_l1505_150501

theorem smallest_x_value : 
  let f (x : ℚ) := 7 * (4 * x^2 + 4 * x + 5) - x * (4 * x - 35)
  ∃ (x : ℚ), f x = 0 ∧ ∀ (y : ℚ), f y = 0 → x ≤ y ∧ x = -5/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1505_150501


namespace NUMINAMATH_CALUDE_arrange_products_eq_eight_l1505_150586

/-- The number of ways to arrange 4 different products in a row,
    with products A and B both to the left of product C -/
def arrange_products : ℕ :=
  let n : ℕ := 4  -- Total number of products
  let ways_to_arrange_AB : ℕ := 2  -- Number of ways to arrange A and B
  let positions_for_last_product : ℕ := 4  -- Possible positions for the last product
  ways_to_arrange_AB * positions_for_last_product

/-- Theorem stating that the number of arrangements is 8 -/
theorem arrange_products_eq_eight : arrange_products = 8 := by
  sorry

end NUMINAMATH_CALUDE_arrange_products_eq_eight_l1505_150586


namespace NUMINAMATH_CALUDE_fractional_equation_range_l1505_150563

theorem fractional_equation_range (x a : ℝ) : 
  (2 * x - a) / (x + 1) = 1 → x > 0 → a > -1 := by sorry

end NUMINAMATH_CALUDE_fractional_equation_range_l1505_150563


namespace NUMINAMATH_CALUDE_crates_needed_is_fifteen_l1505_150591

/-- Calculates the number of crates needed to load items in a warehouse --/
def calculate_crates (crate_capacity : ℕ) (nail_bags : ℕ) (nail_weight : ℕ) 
  (hammer_bags : ℕ) (hammer_weight : ℕ) (plank_bags : ℕ) (plank_weight : ℕ) 
  (left_out_weight : ℕ) : ℕ :=
  let total_weight := nail_bags * nail_weight + hammer_bags * hammer_weight + plank_bags * plank_weight
  let loadable_weight := total_weight - left_out_weight
  (loadable_weight + crate_capacity - 1) / crate_capacity

/-- Theorem stating that given the problem conditions, 15 crates are needed --/
theorem crates_needed_is_fifteen :
  calculate_crates 20 4 5 12 5 10 30 80 = 15 := by
  sorry

end NUMINAMATH_CALUDE_crates_needed_is_fifteen_l1505_150591


namespace NUMINAMATH_CALUDE_journey_equation_correct_l1505_150593

/-- Represents a journey with a stop -/
structure Journey where
  rate_before : ℝ  -- rate before stop in km/h
  rate_after : ℝ   -- rate after stop in km/h
  stop_time : ℝ    -- stop time in hours
  total_time : ℝ   -- total journey time in hours
  total_distance : ℝ -- total distance traveled in km

/-- The equation for the journey is correct -/
theorem journey_equation_correct (j : Journey) 
  (h1 : j.rate_before = 80)
  (h2 : j.rate_after = 100)
  (h3 : j.stop_time = 1/3)
  (h4 : j.total_time = 3)
  (h5 : j.total_distance = 250) :
  ∃ t : ℝ, t ≥ 0 ∧ t ≤ j.total_time - j.stop_time ∧ 
  j.rate_before * t + j.rate_after * (j.total_time - j.stop_time - t) = j.total_distance :=
sorry

end NUMINAMATH_CALUDE_journey_equation_correct_l1505_150593


namespace NUMINAMATH_CALUDE_smallest_other_integer_l1505_150536

theorem smallest_other_integer (a b x : ℕ) : 
  a > 0 → b > 0 → x > 0 → a = 72 → 
  Nat.gcd a b = x + 6 → 
  Nat.lcm a b = 2 * x * (x + 6) → 
  b ≥ 24 ∧ (∃ (y : ℕ), y > 0 ∧ y + 6 ∣ 72 ∧ 
    Nat.gcd 72 24 = y + 6 ∧ 
    Nat.lcm 72 24 = 2 * y * (y + 6)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l1505_150536


namespace NUMINAMATH_CALUDE_duck_profit_l1505_150547

/-- Calculates the profit from buying and selling ducks -/
theorem duck_profit
  (num_ducks : ℕ)
  (cost_per_duck : ℝ)
  (weight_per_duck : ℝ)
  (sell_price_per_pound : ℝ)
  (h1 : num_ducks = 30)
  (h2 : cost_per_duck = 10)
  (h3 : weight_per_duck = 4)
  (h4 : sell_price_per_pound = 5) :
  let total_cost := num_ducks * cost_per_duck
  let total_weight := num_ducks * weight_per_duck
  let total_revenue := total_weight * sell_price_per_pound
  total_revenue - total_cost = 300 :=
by sorry

end NUMINAMATH_CALUDE_duck_profit_l1505_150547


namespace NUMINAMATH_CALUDE_savings_ratio_first_year_l1505_150546

/-- Represents the financial situation of a person over two years -/
structure FinancialSituation where
  firstYearIncome : ℝ
  firstYearSavingsRatio : ℝ
  incomeIncrease : ℝ
  savingsIncrease : ℝ

/-- The theorem stating the savings ratio in the first year -/
theorem savings_ratio_first_year 
  (fs : FinancialSituation)
  (h1 : fs.incomeIncrease = 0.3)
  (h2 : fs.savingsIncrease = 1.0)
  (h3 : fs.firstYearIncome > 0)
  (h4 : 0 ≤ fs.firstYearSavingsRatio ∧ fs.firstYearSavingsRatio ≤ 1) :
  let firstYearExpenditure := fs.firstYearIncome * (1 - fs.firstYearSavingsRatio)
  let secondYearIncome := fs.firstYearIncome * (1 + fs.incomeIncrease)
  let secondYearSavings := fs.firstYearIncome * fs.firstYearSavingsRatio * (1 + fs.savingsIncrease)
  let secondYearExpenditure := secondYearIncome - secondYearSavings
  firstYearExpenditure + secondYearExpenditure = 2 * firstYearExpenditure →
  fs.firstYearSavingsRatio = 0.3 := by
sorry

end NUMINAMATH_CALUDE_savings_ratio_first_year_l1505_150546
