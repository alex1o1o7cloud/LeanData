import Mathlib

namespace NUMINAMATH_CALUDE_fred_tim_marbles_comparison_l3962_396213

theorem fred_tim_marbles_comparison :
  let fred_marbles : ℕ := 110
  let tim_marbles : ℕ := 5
  (fred_marbles / tim_marbles : ℚ) = 22 :=
by sorry

end NUMINAMATH_CALUDE_fred_tim_marbles_comparison_l3962_396213


namespace NUMINAMATH_CALUDE_shirt_to_pants_ratio_l3962_396287

/-- Proves that the ratio of the price of the shirt to the price of the pants is 3:4 given the conditions of the problem. -/
theorem shirt_to_pants_ratio (total_cost pants_price shoes_price shirt_price : ℕ) : 
  total_cost = 340 →
  pants_price = 120 →
  shoes_price = pants_price + 10 →
  shirt_price = total_cost - pants_price - shoes_price →
  (shirt_price : ℚ) / pants_price = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_shirt_to_pants_ratio_l3962_396287


namespace NUMINAMATH_CALUDE_inscribed_pentagon_area_l3962_396264

/-- An equilateral triangle -/
structure EquilateralTriangle :=
  (A B C : ℝ × ℝ)
  (side_length : ℝ)
  (is_equilateral : side_length = 2)

/-- An equilateral pentagon inscribed in a triangle -/
structure InscribedPentagon (T : EquilateralTriangle) :=
  (A M N P Q : ℝ × ℝ)
  (is_equilateral : ∀ X Y, (X, Y) ∈ [(A, M), (M, N), (N, P), (P, Q), (Q, A)] → 
    Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = Real.sqrt ((A.1 - M.1)^2 + (A.2 - M.2)^2))
  (M_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * T.A.1 + (1 - t) * T.B.1, t * T.A.2 + (1 - t) * T.B.2))
  (Q_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (t * T.A.1 + (1 - t) * T.C.1, t * T.A.2 + (1 - t) * T.C.2))
  (N_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ N = (t * T.B.1 + (1 - t) * T.C.1, t * T.B.2 + (1 - t) * T.C.2))
  (P_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * T.B.1 + (1 - t) * T.C.1, t * T.B.2 + (1 - t) * T.C.2))
  (has_symmetry : N.1 - T.B.1 = T.C.1 - P.1 ∧ N.2 - T.B.2 = T.C.2 - P.2)

/-- The area of a polygon given its vertices -/
def polygon_area (vertices : List (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem inscribed_pentagon_area (T : EquilateralTriangle) (P : InscribedPentagon T) :
  polygon_area [P.A, P.M, P.N, P.P, P.Q] = 48 - 27 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_pentagon_area_l3962_396264


namespace NUMINAMATH_CALUDE_constant_sum_of_roots_l3962_396227

theorem constant_sum_of_roots (b x : ℝ) (h : (6 / b) < x ∧ x < (10 / b)) :
  Real.sqrt (x^2 - 2*x + 1) + Real.sqrt (x^2 - 6*x + 9) = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_sum_of_roots_l3962_396227


namespace NUMINAMATH_CALUDE_y_derivative_l3962_396283

noncomputable def y (x : ℝ) : ℝ := 3 * (Real.sin x / Real.cos x ^ 2) + 2 * (Real.sin x / Real.cos x ^ 4)

theorem y_derivative (x : ℝ) :
  deriv y x = (3 + 3 * Real.sin x ^ 2) / Real.cos x ^ 3 + (2 - 6 * Real.sin x ^ 2) / Real.cos x ^ 5 :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l3962_396283


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_factorials_l3962_396240

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def last_two_digits (n : ℕ) : ℕ := n % 100

def sum_factorials : ℕ := List.range 31 |>.map (fun i => factorial (6 + 3 * i)) |>.sum

theorem last_two_digits_of_sum_factorials :
  last_two_digits sum_factorials = 20 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_sum_factorials_l3962_396240


namespace NUMINAMATH_CALUDE_romanian_sequence_swaps_l3962_396215

/-- Represents a Romanian sequence -/
def RomanianSequence (n : ℕ) := { s : List Char // s.length = 3*n ∧ s.count 'I' = n ∧ s.count 'M' = n ∧ s.count 'O' = n }

/-- The minimum number of swaps required to transform one sequence into another -/
def minSwaps (s1 s2 : List Char) : ℕ := sorry

theorem romanian_sequence_swaps (n : ℕ) :
  ∀ (X : RomanianSequence n), ∃ (Y : RomanianSequence n), minSwaps X.val Y.val ≥ (3 * n^2) / 2 := by sorry

end NUMINAMATH_CALUDE_romanian_sequence_swaps_l3962_396215


namespace NUMINAMATH_CALUDE_polynomial_value_equivalence_l3962_396241

theorem polynomial_value_equivalence (a : ℝ) : 
  2 * a^2 + 3 * a + 1 = 6 → -6 * a^2 - 9 * a + 8 = -7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_equivalence_l3962_396241


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l3962_396267

/-- Given a man's rowing speeds with and against a stream, calculate his rate in still water. -/
theorem mans_rate_in_still_water 
  (speed_with_stream : ℝ) 
  (speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 16) 
  (h2 : speed_against_stream = 12) : 
  (speed_with_stream + speed_against_stream) / 2 = 14 := by
  sorry

#check mans_rate_in_still_water

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l3962_396267


namespace NUMINAMATH_CALUDE_nested_square_root_value_l3962_396251

theorem nested_square_root_value :
  ∀ y : ℝ, y = Real.sqrt (4 + y) → y = (1 + Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l3962_396251


namespace NUMINAMATH_CALUDE_hall_wallpaper_expenditure_l3962_396252

/-- Calculates the total expenditure for covering the walls and ceiling of a rectangular hall with wallpaper. -/
def total_expenditure (length width height cost_per_sqm : ℚ) : ℚ :=
  let wall_area := 2 * (length * height + width * height)
  let ceiling_area := length * width
  let total_area := wall_area + ceiling_area
  total_area * cost_per_sqm

/-- Theorem stating that the total expenditure for covering a 30m x 25m x 10m hall with wallpaper costing Rs. 75 per square meter is Rs. 138,750. -/
theorem hall_wallpaper_expenditure :
  total_expenditure 30 25 10 75 = 138750 := by
  sorry

end NUMINAMATH_CALUDE_hall_wallpaper_expenditure_l3962_396252


namespace NUMINAMATH_CALUDE_locus_is_straight_line_l3962_396270

-- Define the fixed point A
def A : ℝ × ℝ := (1, 1)

-- Define the line l
def l (x y : ℝ) : Prop := x + y - 2 = 0

-- Define the locus of points equidistant from A and l
def locus (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  (x - A.1)^2 + (y - A.2)^2 = (x + y - 2)^2 / 2

-- Theorem statement
theorem locus_is_straight_line :
  ∃ (a b c : ℝ), ∀ (x y : ℝ), locus (x, y) ↔ a*x + b*y + c = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_is_straight_line_l3962_396270


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_planes_perpendicular_to_line_are_parallel_l3962_396232

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields

structure Plane3D where
  -- Add necessary fields

-- Define perpendicularity and parallelism
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

def perpendicular_plane_line (p : Plane3D) (l : Line3D) : Prop :=
  sorry

def parallel_planes (p1 p2 : Plane3D) : Prop :=
  sorry

-- Theorem 1: Two lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_plane_are_parallel
  (l1 l2 : Line3D) (p : Plane3D)
  (h1 : perpendicular_line_plane l1 p)
  (h2 : perpendicular_line_plane l2 p) :
  parallel_lines l1 l2 :=
sorry

-- Theorem 2: Two planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_line_are_parallel
  (p1 p2 : Plane3D) (l : Line3D)
  (h1 : perpendicular_plane_line p1 l)
  (h2 : perpendicular_plane_line p2 l) :
  parallel_planes p1 p2 :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_planes_perpendicular_to_line_are_parallel_l3962_396232


namespace NUMINAMATH_CALUDE_gray_area_calculation_l3962_396257

theorem gray_area_calculation (black_area : ℝ) (width1 height1 width2 height2 : ℝ) :
  black_area = 37 ∧ 
  width1 = 8 ∧ height1 = 10 ∧ 
  width2 = 12 ∧ height2 = 9 →
  width2 * height2 - (width1 * height1 - black_area) = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_gray_area_calculation_l3962_396257


namespace NUMINAMATH_CALUDE_antonov_remaining_packs_l3962_396265

/-- The number of packs Antonov has after giving one pack to his sister -/
def remaining_packs (initial_candies : ℕ) (candies_per_pack : ℕ) : ℕ :=
  (initial_candies - candies_per_pack) / candies_per_pack

/-- Theorem stating that Antonov has 2 packs remaining -/
theorem antonov_remaining_packs :
  remaining_packs 60 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_antonov_remaining_packs_l3962_396265


namespace NUMINAMATH_CALUDE_unique_divisor_sequence_l3962_396294

theorem unique_divisor_sequence : ∃! (x y z w : ℕ), 
  x > 1 ∧ y > 1 ∧ z > 1 ∧ w > 1 ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
  x % y = 0 ∧ y % z = 0 ∧ z % w = 0 ∧
  x + y + z + w = 329 ∧
  x = 231 ∧ y = 77 ∧ z = 14 ∧ w = 7 := by
sorry

end NUMINAMATH_CALUDE_unique_divisor_sequence_l3962_396294


namespace NUMINAMATH_CALUDE_least_divisible_by_second_primes_l3962_396271

/-- The second set of four consecutive prime numbers -/
def second_consecutive_primes : Finset Nat := {11, 13, 17, 19}

/-- The product of the second set of four consecutive prime numbers -/
def product_of_primes : Nat := 46219

/-- Theorem stating that the product of the second set of four consecutive primes
    is the least positive whole number divisible by all of them -/
theorem least_divisible_by_second_primes :
  (∀ p ∈ second_consecutive_primes, product_of_primes % p = 0) ∧
  (∀ n : Nat, 0 < n ∧ n < product_of_primes →
    ∃ p ∈ second_consecutive_primes, n % p ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_least_divisible_by_second_primes_l3962_396271


namespace NUMINAMATH_CALUDE_integral_x_over_sqrt_5_minus_x_l3962_396222

theorem integral_x_over_sqrt_5_minus_x (x : ℝ) :
  HasDerivAt (λ x => (2/3) * (5 - x)^(3/2) - 10 * (5 - x)^(1/2)) 
             (x / (5 - x)^(1/2)) 
             x :=
sorry

end NUMINAMATH_CALUDE_integral_x_over_sqrt_5_minus_x_l3962_396222


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3962_396291

-- Define repeating decimals
def repeating_decimal_6 : ℚ := 2/3
def repeating_decimal_2 : ℚ := 2/9
def repeating_decimal_4 : ℚ := 4/9

-- Theorem statement
theorem repeating_decimal_sum :
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3962_396291


namespace NUMINAMATH_CALUDE_mater_cost_percentage_l3962_396236

theorem mater_cost_percentage (lightning_cost sally_cost mater_cost : ℝ) :
  lightning_cost = 140000 →
  sally_cost = 3 * mater_cost →
  sally_cost = 42000 →
  mater_cost / lightning_cost = 0.1 := by
sorry

end NUMINAMATH_CALUDE_mater_cost_percentage_l3962_396236


namespace NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_l3962_396219

theorem smallest_b_for_quadratic_inequality :
  ∀ b : ℝ, b^2 - 16*b + 55 ≥ 0 → b ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_l3962_396219


namespace NUMINAMATH_CALUDE_ten_valid_n_l3962_396206

-- Define the expression
def E (n : ℤ) : ℤ := 4 * n + 7

-- Define the property for valid n
def is_valid (n : ℤ) : Prop := 1 < E n ∧ E n < 40

-- State the theorem
theorem ten_valid_n : (∃! (s : Finset ℤ), s.card = 10 ∧ ∀ n, n ∈ s ↔ is_valid n) :=
sorry

end NUMINAMATH_CALUDE_ten_valid_n_l3962_396206


namespace NUMINAMATH_CALUDE_sum_of_powers_equals_six_to_power_three_l3962_396281

theorem sum_of_powers_equals_six_to_power_three :
  3^3 + 4^3 + 5^3 = 6^3 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_equals_six_to_power_three_l3962_396281


namespace NUMINAMATH_CALUDE_square_perimeter_l3962_396233

theorem square_perimeter (rectangle_length : ℝ) (rectangle_width : ℝ) 
  (h1 : rectangle_length = 125)
  (h2 : rectangle_width = 64)
  (h3 : ∃ square_side : ℝ, square_side^2 = 5 * rectangle_length * rectangle_width) :
  ∃ square_perimeter : ℝ, square_perimeter = 800 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_l3962_396233


namespace NUMINAMATH_CALUDE_relationship_between_A_B_C_l3962_396266

-- Define the variables and functions
variable (a : ℝ)
def A : ℝ := 2 * a - 7
def B : ℝ := a^2 - 4 * a + 3
def C : ℝ := a^2 + 6 * a - 28

-- Theorem statement
theorem relationship_between_A_B_C (h : a > 2) :
  (B a - A a > 0) ∧
  (∀ x, 2 < x ∧ x < 3 → C x - A x < 0) ∧
  (C 3 - A 3 = 0) ∧
  (∀ y, y > 3 → C y - A y > 0) := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_A_B_C_l3962_396266


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3962_396256

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on x and y axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = -l.c / l.b

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line2D),
    pointOnLine ⟨1, 2⟩ l1 ∧
    pointOnLine ⟨1, 2⟩ l2 ∧
    equalIntercepts l1 ∧
    equalIntercepts l2 ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -3) ∨ (l2.a = 2 ∧ l2.b = -1 ∧ l2.c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l3962_396256


namespace NUMINAMATH_CALUDE_gcf_of_lcm_equals_15_l3962_396290

-- Define GCF (Greatest Common Factor)
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM (Least Common Multiple)
def LCM (c d : ℕ) : ℕ := Nat.lcm c d

-- Theorem statement
theorem gcf_of_lcm_equals_15 : GCF (LCM 9 15) (LCM 10 21) = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_lcm_equals_15_l3962_396290


namespace NUMINAMATH_CALUDE_value_k_std_dev_below_mean_l3962_396228

-- Define the properties of the normal distribution
def mean : ℝ := 12
def std_dev : ℝ := 1.2

-- Define the range for k
def k_range (k : ℝ) : Prop := 2 < k ∧ k < 3 ∧ k ≠ ⌊k⌋

-- Theorem statement
theorem value_k_std_dev_below_mean (k : ℝ) (h : k_range k) :
  ∃ (value : ℝ), value = mean - k * std_dev :=
sorry

end NUMINAMATH_CALUDE_value_k_std_dev_below_mean_l3962_396228


namespace NUMINAMATH_CALUDE_kaylee_age_l3962_396268

/-- Given that in 7 years, Kaylee will be 3 times as old as Matt is now,
    and Matt is currently 5 years old, prove that Kaylee is currently 8 years old. -/
theorem kaylee_age (matt_age : ℕ) (kaylee_age : ℕ) :
  matt_age = 5 →
  kaylee_age + 7 = 3 * matt_age →
  kaylee_age = 8 := by
sorry

end NUMINAMATH_CALUDE_kaylee_age_l3962_396268


namespace NUMINAMATH_CALUDE_max_perimeter_l3962_396280

-- Define the triangle sides
def side1 : ℝ := 7
def side2 : ℝ := 8

-- Define the third side as a natural number
def x : ℕ → ℝ := λ n => n

-- Define the triangle inequality
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- The theorem to prove
theorem max_perimeter :
  ∃ n : ℕ, (is_valid_triangle side1 side2 (x n)) ∧
  (∀ m : ℕ, is_valid_triangle side1 side2 (x m) →
    perimeter side1 side2 (x n) ≥ perimeter side1 side2 (x m)) ∧
  perimeter side1 side2 (x n) = 29 :=
sorry

end NUMINAMATH_CALUDE_max_perimeter_l3962_396280


namespace NUMINAMATH_CALUDE_coefficient_a4b3c2_in_expansion_l3962_396288

theorem coefficient_a4b3c2_in_expansion (a b c : ℕ) : 
  (Nat.choose 9 5) * (Nat.choose 5 2) = 1260 := by sorry

end NUMINAMATH_CALUDE_coefficient_a4b3c2_in_expansion_l3962_396288


namespace NUMINAMATH_CALUDE_perpendicular_chords_sum_bounds_l3962_396273

/-- Given a circle with radius R and an interior point P at distance kR from the center,
    where 0 ≤ k ≤ 1, the sum of the lengths of two perpendicular chords passing through P
    is bounded above by 2R√(2(1 - k²)) and below by 0. -/
theorem perpendicular_chords_sum_bounds (R k : ℝ) (h_R_pos : R > 0) (h_k_range : 0 ≤ k ∧ k ≤ 1) :
  ∃ (chord_sum : ℝ), 0 ≤ chord_sum ∧ chord_sum ≤ 2 * R * Real.sqrt (2 * (1 - k^2)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_chords_sum_bounds_l3962_396273


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3962_396299

theorem inequality_solution_set (θ : ℝ) (x : ℝ) :
  (|x + Real.cos θ ^ 2| ≤ Real.sin θ ^ 2) ↔ (-1 ≤ x ∧ x ≤ -Real.cos (2 * θ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3962_396299


namespace NUMINAMATH_CALUDE_min_abs_w_l3962_396249

theorem min_abs_w (w : ℂ) (h : Complex.abs (w - 2*I) + Complex.abs (w + 3) = 6) :
  ∃ (min_abs : ℝ), min_abs = 1 ∧ ∀ (z : ℂ), Complex.abs (w - 2*I) + Complex.abs (w + 3) = 6 → Complex.abs z ≥ min_abs :=
sorry

end NUMINAMATH_CALUDE_min_abs_w_l3962_396249


namespace NUMINAMATH_CALUDE_exists_u_floor_power_minus_n_even_l3962_396229

theorem exists_u_floor_power_minus_n_even :
  ∃ u : ℝ, u > 0 ∧ ∀ n : ℕ, n > 0 → ∃ k : ℤ, 
    (⌊u^n⌋ : ℤ) - n = 2 * k := by sorry

end NUMINAMATH_CALUDE_exists_u_floor_power_minus_n_even_l3962_396229


namespace NUMINAMATH_CALUDE_amy_total_distance_l3962_396276

/-- Calculates the total distance Amy biked over two days given the conditions. -/
def total_distance (yesterday : ℕ) (less_than_twice : ℕ) : ℕ :=
  yesterday + (2 * yesterday - less_than_twice)

/-- Proves that Amy biked 33 miles in total over two days. -/
theorem amy_total_distance :
  total_distance 12 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_amy_total_distance_l3962_396276


namespace NUMINAMATH_CALUDE_prism_cone_properties_l3962_396286

/-- Regular triangular prism with a point T on edge BB₁ forming a cone --/
structure PrismWithCone where
  -- Base edge length of the prism
  a : ℝ
  -- Height of the prism
  h : ℝ
  -- Distance BT
  bt : ℝ
  -- Distance B₁T
  b₁t : ℝ
  -- Constraint on BT:B₁T ratio
  h_ratio : bt / b₁t = 2 / 3
  -- Constraint on prism height
  h_height : h = 5

/-- Theorem about the ratio of prism height to base edge and cone volume --/
theorem prism_cone_properties (p : PrismWithCone) :
  -- 1. Ratio of prism height to base edge is √5
  p.h / p.a = Real.sqrt 5 ∧
  -- 2. Volume of the cone
  ∃ (v : ℝ), v = (180 * Real.pi * Real.sqrt 3) / (23 * Real.sqrt 23) := by
  sorry

end NUMINAMATH_CALUDE_prism_cone_properties_l3962_396286


namespace NUMINAMATH_CALUDE_john_uber_profit_l3962_396259

/-- Calculates the profit from driving Uber given the income, initial car cost, and trade-in value. -/
def uberProfit (income : ℕ) (carCost : ℕ) (tradeInValue : ℕ) : ℕ :=
  income - (carCost - tradeInValue)

/-- Proves that John's profit from driving Uber is $18,000 given the specified conditions. -/
theorem john_uber_profit :
  let income : ℕ := 30000
  let carCost : ℕ := 18000
  let tradeInValue : ℕ := 6000
  uberProfit income carCost tradeInValue = 18000 := by
  sorry

#eval uberProfit 30000 18000 6000

end NUMINAMATH_CALUDE_john_uber_profit_l3962_396259


namespace NUMINAMATH_CALUDE_complete_square_k_value_l3962_396237

/-- A quadratic expression can be factored using the complete square formula if and only if
    it can be written in the form (x + a)^2 for some real number a. --/
def is_complete_square (k : ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x^2 + k*x + 9 = (x + a)^2

/-- If x^2 + kx + 9 can be factored using the complete square formula,
    then k = 6 or k = -6. --/
theorem complete_square_k_value (k : ℝ) :
  is_complete_square k → k = 6 ∨ k = -6 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_k_value_l3962_396237


namespace NUMINAMATH_CALUDE_range_of_m_l3962_396282

theorem range_of_m (x y m : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 2 3, y^2 - x*y - m*x^2 ≤ 0) →
  m ∈ Set.Ioi 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3962_396282


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3962_396285

theorem quadratic_always_positive (b c : ℤ) 
  (h : ∀ x : ℤ, (x^2 : ℤ) + b*x + c > 0) : 
  b^2 - 4*c ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3962_396285


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3962_396250

theorem inequality_equivalence (x : ℝ) : (x - 3) / (x^2 + 4*x + 10) ≥ 0 ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3962_396250


namespace NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l3962_396242

theorem six_digit_multiple_of_nine : ∃ (n : ℕ), 456786 = 9 * n := by
  sorry

end NUMINAMATH_CALUDE_six_digit_multiple_of_nine_l3962_396242


namespace NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l3962_396235

theorem sum_of_roots_squared_equation (x : ℝ) :
  (∀ x, (x - 4)^2 = 16 ↔ x = 8 ∨ x = 0) →
  (∃ a b : ℝ, (a - 4)^2 = 16 ∧ (b - 4)^2 = 16 ∧ a + b = 8) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_squared_equation_l3962_396235


namespace NUMINAMATH_CALUDE_least_positive_angle_phi_l3962_396295

theorem least_positive_angle_phi : 
  ∃ φ : Real, φ > 0 ∧ φ ≤ π/2 ∧ 
  Real.cos (15 * π/180) = Real.sin (45 * π/180) + Real.sin φ ∧
  ∀ ψ : Real, ψ > 0 ∧ ψ < φ → 
    Real.cos (15 * π/180) ≠ Real.sin (45 * π/180) + Real.sin ψ ∧
  φ = 15 * π/180 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_angle_phi_l3962_396295


namespace NUMINAMATH_CALUDE_afternoon_sales_proof_l3962_396217

/-- A salesman sells pears in the morning and afternoon. -/
structure PearSales where
  morning : ℝ
  afternoon : ℝ

/-- The total amount of pears sold in a day. -/
def total_sales (s : PearSales) : ℝ := s.morning + s.afternoon

/-- Theorem: Given a salesman who sold twice as much pears in the afternoon than in the morning,
    and sold 390 kilograms in total that day, the amount sold in the afternoon is 260 kilograms. -/
theorem afternoon_sales_proof (s : PearSales) 
    (h1 : s.afternoon = 2 * s.morning) 
    (h2 : total_sales s = 390) : 
    s.afternoon = 260 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_sales_proof_l3962_396217


namespace NUMINAMATH_CALUDE_correct_possible_values_l3962_396245

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The set of possible values for 'a' -/
def PossibleValues : Set ℝ := {1/3, 3, -6}

/-- Function to count the number of intersection points between three lines -/
def countIntersections (l1 l2 l3 : Line) : ℕ := sorry

/-- Theorem stating that the set of possible values of 'a' is correct -/
theorem correct_possible_values :
  ∀ a : ℝ,
  (∃ l3 : Line,
    l3.a = a ∧ l3.b = 3 ∧ l3.c = -5 ∧
    countIntersections ⟨1, 1, 1⟩ ⟨2, -1, 8⟩ l3 ≤ 2) ↔
  a ∈ PossibleValues :=
sorry

end NUMINAMATH_CALUDE_correct_possible_values_l3962_396245


namespace NUMINAMATH_CALUDE_class_composition_l3962_396238

theorem class_composition (total : ℕ) (girls boys : ℕ) : 
  girls = (6 : ℚ) / 10 * total →
  (girls - 1 : ℚ) / (total - 3) = 25 / 40 →
  girls = 21 ∧ boys = 14 := by
  sorry

end NUMINAMATH_CALUDE_class_composition_l3962_396238


namespace NUMINAMATH_CALUDE_points_divisible_by_ten_l3962_396203

/-- A configuration of points on a circle satisfying certain distance conditions -/
structure PointConfiguration where
  n : ℕ
  circle_length : ℕ
  distance_one : ∀ i : Fin n, ∃! j : Fin n, i ≠ j ∧ (i.val - j.val) % circle_length = 1
  distance_two : ∀ i : Fin n, ∃! j : Fin n, i ≠ j ∧ (i.val - j.val) % circle_length = 2

/-- Theorem stating that for a specific configuration, n is divisible by 10 -/
theorem points_divisible_by_ten (config : PointConfiguration) 
  (h_length : config.circle_length = 15) : 
  10 ∣ config.n :=
sorry

end NUMINAMATH_CALUDE_points_divisible_by_ten_l3962_396203


namespace NUMINAMATH_CALUDE_set_difference_equals_interval_l3962_396216

def M : Set ℝ := {x | x^2 + x - 12 ≤ 0}

def N : Set ℝ := {y | ∃ x, y = 3^x ∧ x ≤ 1}

theorem set_difference_equals_interval :
  {x | x ∈ M ∧ x ∉ N} = Set.Ico (-4) 0 := by sorry

end NUMINAMATH_CALUDE_set_difference_equals_interval_l3962_396216


namespace NUMINAMATH_CALUDE_not_equivalent_to_0_0000042_l3962_396205

theorem not_equivalent_to_0_0000042 : ¬ (2.1 * 10^(-6) = 0.0000042) :=
by
  have h1 : 0.0000042 = 4.2 * 10^(-6) := by sorry
  sorry

end NUMINAMATH_CALUDE_not_equivalent_to_0_0000042_l3962_396205


namespace NUMINAMATH_CALUDE_renatas_final_balance_l3962_396296

/-- Represents the balance and transactions of Renata's day --/
def renatas_day (initial_amount : ℚ) (charity_donation : ℚ) (prize_pounds : ℚ) 
  (slot_loss_euros : ℚ) (slot_loss_pounds : ℚ) (slot_loss_dollars : ℚ)
  (sunglasses_euros : ℚ) (water_pounds : ℚ) (lottery_ticket : ℚ) (lottery_prize : ℚ)
  (meal_euros : ℚ) (coffee_euros : ℚ) : ℚ :=
  let pound_to_dollar : ℚ := 1.35
  let euro_to_dollar : ℚ := 1.10
  let sunglasses_discount : ℚ := 0.20
  let meal_discount : ℚ := 0.30
  
  let balance1 := initial_amount - charity_donation
  let balance2 := balance1 + prize_pounds * pound_to_dollar
  let balance3 := balance2 - slot_loss_euros * euro_to_dollar
  let balance4 := balance3 - slot_loss_pounds * pound_to_dollar
  let balance5 := balance4 - slot_loss_dollars
  let balance6 := balance5 - sunglasses_euros * (1 - sunglasses_discount) * euro_to_dollar
  let balance7 := balance6 - water_pounds * pound_to_dollar
  let balance8 := balance7 - lottery_ticket
  let balance9 := balance8 + lottery_prize
  let lunch_cost := (meal_euros * (1 - meal_discount) + coffee_euros) * euro_to_dollar
  balance9 - lunch_cost / 2

/-- Theorem stating that Renata's final balance is $35.95 --/
theorem renatas_final_balance :
  renatas_day 50 10 50 30 20 15 15 1 1 30 10 3 = 35.95 := by sorry

end NUMINAMATH_CALUDE_renatas_final_balance_l3962_396296


namespace NUMINAMATH_CALUDE_oatmeal_boxes_sold_problem_l3962_396275

/-- The number of oatmeal biscuit boxes sold to the neighbor -/
def oatmeal_boxes_sold (total_boxes : ℕ) (lemon_boxes : ℕ) (chocolate_boxes : ℕ) (boxes_to_sell : ℕ) : ℕ :=
  total_boxes - lemon_boxes - chocolate_boxes - boxes_to_sell

theorem oatmeal_boxes_sold_problem (total_boxes : ℕ) (lemon_boxes : ℕ) (chocolate_boxes : ℕ) (boxes_to_sell : ℕ)
  (h1 : total_boxes = 33)
  (h2 : lemon_boxes = 12)
  (h3 : chocolate_boxes = 5)
  (h4 : boxes_to_sell = 12) :
  oatmeal_boxes_sold total_boxes lemon_boxes chocolate_boxes boxes_to_sell = 4 := by
  sorry

end NUMINAMATH_CALUDE_oatmeal_boxes_sold_problem_l3962_396275


namespace NUMINAMATH_CALUDE_intersection_M_N_l3962_396211

def M : Set ℝ := {x | -4 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - x - 6 < 0}

theorem intersection_M_N : M ∩ N = {x | -2 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3962_396211


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3962_396209

/-- Given a quadratic equation 2x² = 5x - 3, prove that when converted to the general form ax² + bx + c = 0,
    if the coefficient of x² (a) is 2, then the coefficient of x (b) is -5. -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, 2*x^2 = 5*x - 3) →  -- original equation
  (∀ x, a*x^2 + b*x + c = 0) →  -- general form
  a = 2 →
  b = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3962_396209


namespace NUMINAMATH_CALUDE_power_plus_one_not_divisible_by_power_minus_one_l3962_396223

theorem power_plus_one_not_divisible_by_power_minus_one (x y : ℕ) (h : y > 2) :
  ¬ (2^y - 1 ∣ 2^x + 1) := by
  sorry

end NUMINAMATH_CALUDE_power_plus_one_not_divisible_by_power_minus_one_l3962_396223


namespace NUMINAMATH_CALUDE_infinitely_many_non_representable_l3962_396255

/-- A number is representable if it can be written as p + n^(2k) for some prime p and integers n, k -/
def Representable (m : ℕ) : Prop :=
  ∃ (p n k : ℕ), Prime p ∧ m = p + n^(2*k)

/-- The theorem stating that there are infinitely many non-representable numbers -/
theorem infinitely_many_non_representable :
  ∀ N : ℕ, ∃ m : ℕ, m > N ∧ ¬(Representable m) := by sorry

end NUMINAMATH_CALUDE_infinitely_many_non_representable_l3962_396255


namespace NUMINAMATH_CALUDE_complex_magnitude_l3962_396204

theorem complex_magnitude (z : ℂ) (h : z = 4 + 3 * I) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3962_396204


namespace NUMINAMATH_CALUDE_tooth_fairy_total_l3962_396234

/-- The total number of baby teeth a child has. -/
def totalTeeth : ℕ := 20

/-- The number of teeth lost or swallowed. -/
def lostTeeth : ℕ := 2

/-- The amount received for the first tooth. -/
def firstToothAmount : ℕ := 20

/-- The amount received for each subsequent tooth. -/
def regularToothAmount : ℕ := 2

/-- The total amount received from the tooth fairy. -/
def totalAmount : ℕ := firstToothAmount + regularToothAmount * (totalTeeth - lostTeeth - 1)

theorem tooth_fairy_total : totalAmount = 54 := by
  sorry

end NUMINAMATH_CALUDE_tooth_fairy_total_l3962_396234


namespace NUMINAMATH_CALUDE_max_min_f_l3962_396218

def f (x y : ℝ) : ℝ := 3 * |x + y| + |4 * y + 9| + |7 * y - 3 * x - 18|

theorem max_min_f :
  ∀ x y : ℝ, x^2 + y^2 ≤ 5 →
  (∀ x' y' : ℝ, x'^2 + y'^2 ≤ 5 → f x' y' ≤ f x y) → f x y = 27 + 6 * Real.sqrt 5 ∧
  (∀ x' y' : ℝ, x'^2 + y'^2 ≤ 5 → f x y ≤ f x' y') → f x y = 27 - 3 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_max_min_f_l3962_396218


namespace NUMINAMATH_CALUDE_unique_m_value_l3962_396225

/-- Given a set A and a real number m, proves that m = 3 is the only valid solution -/
theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_m_value_l3962_396225


namespace NUMINAMATH_CALUDE_reciprocal_of_lcm_l3962_396207

def a : ℕ := 24
def b : ℕ := 195

theorem reciprocal_of_lcm (a b : ℕ) : (1 : ℚ) / (Nat.lcm a b) = 1 / 1560 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_lcm_l3962_396207


namespace NUMINAMATH_CALUDE_cable_length_l3962_396248

/-- The length of the curve defined by the intersection of a plane and a sphere --/
theorem cable_length (x y z : ℝ) : 
  x + y + z = 10 → 
  x * y + y * z + x * z = -22 → 
  (∃ (l : ℝ), l = 4 * Real.pi * Real.sqrt (83 / 3) ∧ 
   l = 2 * Real.pi * Real.sqrt (144 - (10^2 / 3))) :=
by sorry

end NUMINAMATH_CALUDE_cable_length_l3962_396248


namespace NUMINAMATH_CALUDE_flower_bee_difference_l3962_396279

def number_of_flowers : ℕ := 5
def number_of_bees : ℕ := 3

theorem flower_bee_difference : 
  number_of_flowers - number_of_bees = 2 := by sorry

end NUMINAMATH_CALUDE_flower_bee_difference_l3962_396279


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l3962_396214

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Predicate for a parabola having a vertical axis of symmetry -/
def has_vertical_axis_of_symmetry (p : Parabola) : Prop :=
  ∃ h : ℝ, ∀ x y : ℝ, p.y_coord (h + x) = p.y_coord (h - x)

theorem parabola_coefficient_sum (p : Parabola) :
  p.y_coord (-3) = 4 →  -- vertex condition
  has_vertical_axis_of_symmetry p →  -- vertical axis of symmetry
  p.y_coord (-1) = 16 →  -- point condition
  p.a + p.b + p.c = 52 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l3962_396214


namespace NUMINAMATH_CALUDE_cat_cafe_cool_cats_cat_cafe_cool_cats_proof_l3962_396278

theorem cat_cafe_cool_cats : ℕ → ℕ → ℕ → Prop :=
  fun cool paw meow =>
    paw = 2 * cool ∧
    meow = 3 * paw ∧
    meow + paw = 40 →
    cool = 5

-- Proof
theorem cat_cafe_cool_cats_proof : cat_cafe_cool_cats 5 10 30 :=
by
  sorry

end NUMINAMATH_CALUDE_cat_cafe_cool_cats_cat_cafe_cool_cats_proof_l3962_396278


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l3962_396244

/-- A function satisfying the given functional equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 6 * x^2 * f y + 2 * x^2 * y^2

theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfyingFunction f ∧ f = fun x ↦ x^2 :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l3962_396244


namespace NUMINAMATH_CALUDE_upstream_journey_distance_l3962_396253

/-- Calculates the effective speed of a boat traveling upstream -/
def effectiveSpeed (boatSpeed currentSpeed : ℝ) : ℝ :=
  boatSpeed - currentSpeed

/-- Calculates the distance traveled in one hour given the effective speed -/
def distanceTraveled (effectiveSpeed : ℝ) : ℝ :=
  effectiveSpeed * 1

theorem upstream_journey_distance 
  (boatSpeed : ℝ) 
  (currentSpeed1 currentSpeed2 currentSpeed3 : ℝ) 
  (h1 : boatSpeed = 50)
  (h2 : currentSpeed1 = 10)
  (h3 : currentSpeed2 = 20)
  (h4 : currentSpeed3 = 15) :
  distanceTraveled (effectiveSpeed boatSpeed currentSpeed1) +
  distanceTraveled (effectiveSpeed boatSpeed currentSpeed2) +
  distanceTraveled (effectiveSpeed boatSpeed currentSpeed3) = 105 := by
  sorry

end NUMINAMATH_CALUDE_upstream_journey_distance_l3962_396253


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l3962_396269

theorem imaginary_part_of_complex_product (i : ℂ) :
  i * i = -1 →
  (Complex.im ((2 - 3 * i) * i) = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_product_l3962_396269


namespace NUMINAMATH_CALUDE_hyperbola_and_line_equations_l3962_396239

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 49 + y^2 / 24 = 1

-- Define the asymptotes of the hyperbola
def asymptotes (x y : ℝ) : Prop := y = 4/3 * x ∨ y = -4/3 * x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the line passing through the right focus
def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 5 * Real.sqrt 3 = 0

-- State the theorem
theorem hyperbola_and_line_equations :
  (∀ x y : ℝ, ellipse x y → asymptotes x y → hyperbola x y) ∧
  (∀ x y : ℝ, ellipse x y → line x y) := by sorry

end NUMINAMATH_CALUDE_hyperbola_and_line_equations_l3962_396239


namespace NUMINAMATH_CALUDE_sum_of_constants_l3962_396261

/-- Given a function y = a + b/x, prove that a + b = 11 under specific conditions -/
theorem sum_of_constants (a b : ℝ) : 
  (2 = a + b/(-2)) → 
  (6 = a + b/(-6)) → 
  (10 = a + b/(-3)) → 
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l3962_396261


namespace NUMINAMATH_CALUDE_peter_bought_nine_kilos_of_tomatoes_l3962_396272

/-- Represents the purchase of groceries by Peter -/
structure Groceries where
  initialMoney : ℕ
  potatoPrice : ℕ
  potatoKilos : ℕ
  tomatoPrice : ℕ
  cucumberPrice : ℕ
  cucumberKilos : ℕ
  bananaPrice : ℕ
  bananaKilos : ℕ
  remainingMoney : ℕ

/-- Calculates the number of kilos of tomatoes bought -/
def tomatoKilos (g : Groceries) : ℕ :=
  (g.initialMoney - g.remainingMoney - 
   (g.potatoPrice * g.potatoKilos + 
    g.cucumberPrice * g.cucumberKilos + 
    g.bananaPrice * g.bananaKilos)) / g.tomatoPrice

/-- Theorem stating that Peter bought 9 kilos of tomatoes -/
theorem peter_bought_nine_kilos_of_tomatoes (g : Groceries) 
  (h1 : g.initialMoney = 500)
  (h2 : g.potatoPrice = 2)
  (h3 : g.potatoKilos = 6)
  (h4 : g.tomatoPrice = 3)
  (h5 : g.cucumberPrice = 4)
  (h6 : g.cucumberKilos = 5)
  (h7 : g.bananaPrice = 5)
  (h8 : g.bananaKilos = 3)
  (h9 : g.remainingMoney = 426) :
  tomatoKilos g = 9 := by
  sorry

end NUMINAMATH_CALUDE_peter_bought_nine_kilos_of_tomatoes_l3962_396272


namespace NUMINAMATH_CALUDE_unique_solution_ABCD_l3962_396231

/-- Represents a base-5 number with two digits --/
def Base5TwoDigit (a b : Nat) : Nat := 5 * a + b

/-- Represents a base-5 number with one digit --/
def Base5OneDigit (a : Nat) : Nat := a

/-- Represents a base-5 number with two identical digits --/
def Base5TwoSameDigit (a : Nat) : Nat := 5 * a + a

theorem unique_solution_ABCD :
  ∀ A B C D : Nat,
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0) →
  (A < 5 ∧ B < 5 ∧ C < 5 ∧ D < 5) →
  (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) →
  (Base5TwoDigit A B + Base5OneDigit C = Base5TwoDigit D 0) →
  (Base5TwoDigit A B + Base5TwoDigit B A = Base5TwoSameDigit D) →
  A = 4 ∧ B = 1 ∧ C = 4 ∧ D = 4 := by
  sorry

#check unique_solution_ABCD

end NUMINAMATH_CALUDE_unique_solution_ABCD_l3962_396231


namespace NUMINAMATH_CALUDE_train_length_l3962_396297

/-- The length of a train given its crossing times over two platforms -/
theorem train_length (t1 t2 p1 p2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) (h3 : p1 > 0) (h4 : p2 > 0)
  (h5 : (L + p1) / t1 = (L + p2) / t2) : L = 100 :=
by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3962_396297


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3962_396254

/-- Given that Ashley's age is 8 and the sum of Ashley's and Mary's ages is 22,
    prove that the ratio of Ashley's age to Mary's age is 4:7. -/
theorem age_ratio_proof (ashley_age mary_age : ℕ) : 
  ashley_age = 8 → ashley_age + mary_age = 22 → ashley_age * 7 = mary_age * 4 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3962_396254


namespace NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_adjustment_l3962_396274

/-- Proves that a 60% price increase and 20% quantity decrease results in a 28% revenue increase -/
theorem revenue_change_after_price_and_quantity_adjustment 
  (P Q : ℝ) 
  (P_new : ℝ := 1.60 * P) 
  (Q_new : ℝ := 0.80 * Q) 
  (h_P : P > 0) 
  (h_Q : Q > 0) : 
  (P_new * Q_new) / (P * Q) = 1.28 := by
  sorry

end NUMINAMATH_CALUDE_revenue_change_after_price_and_quantity_adjustment_l3962_396274


namespace NUMINAMATH_CALUDE_primitive_roots_existence_l3962_396201

theorem primitive_roots_existence (p : Nat) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ x : Nat, IsPrimitiveRoot x p ∧ IsPrimitiveRoot (4 * x) p :=
sorry

end NUMINAMATH_CALUDE_primitive_roots_existence_l3962_396201


namespace NUMINAMATH_CALUDE_book_difference_proof_l3962_396263

def old_town_books : ℕ := 750
def riverview_books : ℕ := 1240
def downtown_books : ℕ := 1800
def eastside_books : ℕ := 1620

def library_books : List ℕ := [old_town_books, riverview_books, downtown_books, eastside_books]

theorem book_difference_proof :
  (List.maximum library_books).get! - (List.minimum library_books).get! = 1050 := by
  sorry

end NUMINAMATH_CALUDE_book_difference_proof_l3962_396263


namespace NUMINAMATH_CALUDE_scientific_notation_of_1300000_l3962_396200

theorem scientific_notation_of_1300000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1300000 = a * (10 : ℝ) ^ n ∧ a = 1.3 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1300000_l3962_396200


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_eccentricity_2_l3962_396243

/-- Prove that for a hyperbola with equation x²/a² - y²/b² = 1 where a > 0, b > 0, 
    and eccentricity e = 2, the equations of its asymptotes are y = ±√3x. -/
theorem hyperbola_asymptotes_eccentricity_2 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (he : (a^2 + b^2).sqrt / a = 2) : 
  ∃ (k : ℝ), k = Real.sqrt 3 ∧ 
    (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (y = k*x ∨ y = -k*x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_eccentricity_2_l3962_396243


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3962_396208

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^2015 + 1 = q * (X^8 - X^6 + X^4 - X^2 + 1) + (-X^5 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3962_396208


namespace NUMINAMATH_CALUDE_symmetry_point_xOy_l3962_396277

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xOy plane in 3D space -/
def xOyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xOy plane -/
def symmetryXOy (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetry_point_xOy :
  let P : Point3D := { x := -3, y := 2, z := -1 }
  symmetryXOy P = { x := -3, y := 2, z := 1 } := by
  sorry

end NUMINAMATH_CALUDE_symmetry_point_xOy_l3962_396277


namespace NUMINAMATH_CALUDE_brad_read_more_books_l3962_396292

/-- Proves that Brad read 4 more books than William across two months --/
theorem brad_read_more_books (william_last_month : ℕ) (brad_this_month : ℕ) : 
  william_last_month = 6 →
  brad_this_month = 8 →
  (3 * william_last_month + brad_this_month) - (william_last_month + 2 * brad_this_month) = 4 := by
sorry

end NUMINAMATH_CALUDE_brad_read_more_books_l3962_396292


namespace NUMINAMATH_CALUDE_game_winning_strategy_l3962_396260

def game_move (k : ℕ) : Set ℕ := {k + 1, 2 * k}

def is_winning_position (n : ℕ) : Prop :=
  ∃ (k c : ℕ), n = 2^(2*k+1) + 2*c ∧ c < 2^k

theorem game_winning_strategy (n : ℕ) (h : n > 1) :
  (∀ k, k ∈ game_move 2 → k ≤ n) →
  (is_winning_position n ↔ 
    ∃ (strategy : ℕ → ℕ), 
      (∀ m, m < n → strategy m ∈ game_move m) ∧
      (∀ m, m < n → strategy (strategy m) > n)) :=
sorry

end NUMINAMATH_CALUDE_game_winning_strategy_l3962_396260


namespace NUMINAMATH_CALUDE_value_of_a_l3962_396226

theorem value_of_a (a : ℝ) (h : a + a/4 = 5/2) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3962_396226


namespace NUMINAMATH_CALUDE_smallest_integer_S_n_l3962_396246

def K' : ℚ := 137 / 60

def S (n : ℕ) : ℚ := n * 5^(n-1) * K' + 1

theorem smallest_integer_S_n : 
  (∀ m : ℕ, m > 0 → m < 12 → ¬(S m).isInt) ∧ (S 12).isInt :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_S_n_l3962_396246


namespace NUMINAMATH_CALUDE_ram_ravi_selection_probability_l3962_396289

theorem ram_ravi_selection_probability 
  (p_ram : ℝ) 
  (p_both : ℝ) 
  (h1 : p_ram = 1/7)
  (h2 : p_both = 0.02857142857142857) :
  ∃ (p_ravi : ℝ), p_ravi = 0.2 ∧ p_both = p_ram * p_ravi :=
by sorry

end NUMINAMATH_CALUDE_ram_ravi_selection_probability_l3962_396289


namespace NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l3962_396284

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1/4)) = -16286/16384 := by
  sorry

end NUMINAMATH_CALUDE_cos_eight_arccos_one_fourth_l3962_396284


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3962_396224

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x - 2
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 3 ∧ x₂ = 1 - Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3962_396224


namespace NUMINAMATH_CALUDE_find_number_multiplied_by_9999_l3962_396262

theorem find_number_multiplied_by_9999 : ∃ x : ℕ, x * 9999 = 724797420 ∧ x = 72480 := by
  sorry

end NUMINAMATH_CALUDE_find_number_multiplied_by_9999_l3962_396262


namespace NUMINAMATH_CALUDE_main_age_is_46_l3962_396247

/-- Represents the ages of four siblings -/
structure Ages where
  main : ℕ
  brother : ℕ
  sister : ℕ
  youngest : ℕ

/-- Checks if the given ages satisfy all the conditions -/
def satisfiesConditions (ages : Ages) : Prop :=
  let futureAges := Ages.mk (ages.main + 10) (ages.brother + 10) (ages.sister + 10) (ages.youngest + 10)
  futureAges.main + futureAges.brother + futureAges.sister + futureAges.youngest = 88 ∧
  futureAges.main = 2 * futureAges.brother ∧
  futureAges.main = 3 * futureAges.sister ∧
  futureAges.main = 4 * futureAges.youngest ∧
  ages.brother = ages.sister + 3 ∧
  ages.sister = 2 * ages.youngest ∧
  ages.youngest = 4

theorem main_age_is_46 :
  ∃ (ages : Ages), satisfiesConditions ages ∧ ages.main = 46 := by
  sorry

end NUMINAMATH_CALUDE_main_age_is_46_l3962_396247


namespace NUMINAMATH_CALUDE_condition_relationship_l3962_396220

theorem condition_relationship (x₁ x₂ : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 4 ∧ x₂ > 4 → x₁ + x₂ > 8 ∧ x₁ * x₂ > 16) ∧
  (∃ x₁ x₂ : ℝ, x₁ + x₂ > 8 ∧ x₁ * x₂ > 16 ∧ ¬(x₁ > 4 ∧ x₂ > 4)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l3962_396220


namespace NUMINAMATH_CALUDE_bill_difference_l3962_396202

theorem bill_difference (mike_tip joe_tip : ℝ) (mike_percent joe_percent : ℝ) 
  (h1 : mike_tip = 5)
  (h2 : joe_tip = 10)
  (h3 : mike_percent = 20)
  (h4 : joe_percent = 25)
  (h5 : mike_tip = mike_percent / 100 * mike_bill)
  (h6 : joe_tip = joe_percent / 100 * joe_bill) :
  |mike_bill - joe_bill| = 15 :=
sorry

end NUMINAMATH_CALUDE_bill_difference_l3962_396202


namespace NUMINAMATH_CALUDE_count_sequences_l3962_396210

/-- The number of finite sequences of k positive integers that sum to n -/
def T (n k : ℕ) : ℕ := sorry

/-- The main theorem stating that T(n,k) equals (n-1 choose k-1) for 1 ≤ k < n -/
theorem count_sequences (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  T n k = Nat.choose (n - 1) (k - 1) := by sorry

end NUMINAMATH_CALUDE_count_sequences_l3962_396210


namespace NUMINAMATH_CALUDE_binomial_divisibility_sequence_l3962_396293

theorem binomial_divisibility_sequence :
  ∃ n : ℕ, n > 2003 ∧ ∀ i j : ℕ, 0 ≤ i → i < j → j ≤ 2003 → (n.choose i ∣ n.choose j) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_sequence_l3962_396293


namespace NUMINAMATH_CALUDE_calculate_expression_l3962_396212

theorem calculate_expression : 
  3 / Real.sqrt 3 - (Real.pi + Real.sqrt 3) ^ 0 - Real.sqrt 27 + |Real.sqrt 3 - 2| = -3 * Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3962_396212


namespace NUMINAMATH_CALUDE_olivia_paper_usage_l3962_396230

/-- The number of pieces of paper Olivia initially had -/
def initial_pieces : ℕ := 81

/-- The number of pieces of paper Olivia has left -/
def remaining_pieces : ℕ := 25

/-- The number of pieces of paper Olivia used -/
def used_pieces : ℕ := initial_pieces - remaining_pieces

theorem olivia_paper_usage :
  used_pieces = 56 :=
sorry

end NUMINAMATH_CALUDE_olivia_paper_usage_l3962_396230


namespace NUMINAMATH_CALUDE_course_total_hours_l3962_396258

/-- Calculates the total hours spent on a course given the course duration and weekly schedule. -/
def total_course_hours (weeks : ℕ) (class_hours_1 class_hours_2 class_hours_3 homework_hours : ℕ) : ℕ :=
  weeks * (class_hours_1 + class_hours_2 + class_hours_3 + homework_hours)

/-- Proves that a 24-week course with the given weekly schedule results in 336 total hours. -/
theorem course_total_hours :
  total_course_hours 24 3 3 4 4 = 336 := by
  sorry

end NUMINAMATH_CALUDE_course_total_hours_l3962_396258


namespace NUMINAMATH_CALUDE_tim_pay_per_task_l3962_396298

/-- Represents the pay per task for Tim's work --/
def pay_per_task (tasks_per_day : ℕ) (days_per_week : ℕ) (weekly_pay : ℚ) : ℚ :=
  weekly_pay / (tasks_per_day * days_per_week)

/-- Theorem stating that Tim's pay per task is $1.20 --/
theorem tim_pay_per_task :
  pay_per_task 100 6 720 = 1.20 := by
  sorry

end NUMINAMATH_CALUDE_tim_pay_per_task_l3962_396298


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l3962_396221

/-- The ring toss game made this amount in the first 44 days -/
def first_period_earnings : ℕ := 382

/-- The ring toss game made this amount in the remaining 10 days -/
def second_period_earnings : ℕ := 374

/-- The total earnings of the ring toss game -/
def total_earnings : ℕ := first_period_earnings + second_period_earnings

theorem ring_toss_earnings : total_earnings = 756 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l3962_396221
