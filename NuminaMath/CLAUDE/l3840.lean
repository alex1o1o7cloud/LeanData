import Mathlib

namespace NUMINAMATH_CALUDE_gcd_b_always_one_l3840_384037

def b (n : ℕ) : ℤ := (8^n - 1) / 7

theorem gcd_b_always_one (n : ℕ) : Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_b_always_one_l3840_384037


namespace NUMINAMATH_CALUDE_kimberly_skittles_bought_l3840_384091

/-- The number of Skittles Kimberly bought -/
def skittles_bought : ℕ := sorry

/-- Kimberly's initial number of Skittles -/
def initial_skittles : ℕ := 5

/-- Kimberly's total number of Skittles after buying more -/
def total_skittles : ℕ := 12

theorem kimberly_skittles_bought :
  skittles_bought = total_skittles - initial_skittles :=
sorry

end NUMINAMATH_CALUDE_kimberly_skittles_bought_l3840_384091


namespace NUMINAMATH_CALUDE_triangle_side_length_proof_l3840_384098

noncomputable def triangle_side_length 
  (AB AC : ℝ) 
  (angle_BAC angle_ABC : ℝ) : ℝ :=
let BC := Real.sqrt 149
BC

theorem triangle_side_length_proof 
  (AB AC : ℝ) 
  (angle_BAC angle_ABC : ℝ) 
  (h1 : AB = 7) 
  (h2 : AC = 10) 
  (h3 : angle_BAC = 40) 
  (h4 : angle_ABC = 50) :
  triangle_side_length AB AC angle_BAC angle_ABC = Real.sqrt 149 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_proof_l3840_384098


namespace NUMINAMATH_CALUDE_adams_house_number_range_l3840_384093

/-- Represents a range of house numbers -/
structure Range where
  lower : Nat
  upper : Nat
  valid : lower ≤ upper

/-- Checks if two ranges overlap -/
def overlaps (r1 r2 : Range) : Prop :=
  (r1.lower ≤ r2.upper ∧ r2.lower ≤ r1.upper) ∨
  (r2.lower ≤ r1.upper ∧ r1.lower ≤ r2.upper)

/-- The given ranges -/
def rangeA : Range := ⟨123, 213, by sorry⟩
def rangeB : Range := ⟨132, 231, by sorry⟩
def rangeC : Range := ⟨123, 312, by sorry⟩
def rangeD : Range := ⟨231, 312, by sorry⟩
def rangeE : Range := ⟨312, 321, by sorry⟩

/-- All ranges except E -/
def otherRanges : List Range := [rangeA, rangeB, rangeC, rangeD]

theorem adams_house_number_range :
  (∀ r ∈ otherRanges, ∃ r' ∈ otherRanges, r ≠ r' ∧ overlaps r r') ∧
  (∀ r ∈ otherRanges, ¬overlaps r rangeE) :=
by sorry

end NUMINAMATH_CALUDE_adams_house_number_range_l3840_384093


namespace NUMINAMATH_CALUDE_factorization_xy_squared_minus_16x_l3840_384090

theorem factorization_xy_squared_minus_16x (x y : ℝ) : x * y^2 - 16 * x = x * (y + 4) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_xy_squared_minus_16x_l3840_384090


namespace NUMINAMATH_CALUDE_line_direction_vector_l3840_384079

/-- Given a line passing through two points and its direction vector, prove the value of 'a' -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (a : ℝ) : 
  p1 = (-3, 7) → p2 = (2, -1) → ∃ k : ℝ, k ≠ 0 ∧ k • (p2.1 - p1.1, p2.2 - p1.2) = (a, -2) → a = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l3840_384079


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l3840_384063

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l3840_384063


namespace NUMINAMATH_CALUDE_max_value_is_120_l3840_384055

def is_valid_assignment (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  b ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  c ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  d ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)

def expression (a b c d : ℕ) : ℚ :=
  (a : ℚ) / ((b : ℚ) / ((c * d : ℚ)))

theorem max_value_is_120 :
  ∀ a b c d : ℕ, is_valid_assignment a b c d →
    expression a b c d ≤ 120 :=
by sorry

end NUMINAMATH_CALUDE_max_value_is_120_l3840_384055


namespace NUMINAMATH_CALUDE_distance_to_midpoint_is_12_l3840_384067

/-- An isosceles triangle DEF with given side lengths -/
structure IsoscelesTriangleDEF where
  /-- The length of side DE -/
  de : ℝ
  /-- The length of side DF -/
  df : ℝ
  /-- The length of side EF -/
  ef : ℝ
  /-- DE and DF are equal -/
  de_eq_df : de = df
  /-- DE is 13 units -/
  de_is_13 : de = 13
  /-- EF is 10 units -/
  ef_is_10 : ef = 10

/-- The distance from D to the midpoint of EF in the isosceles triangle DEF -/
def distanceToMidpoint (t : IsoscelesTriangleDEF) : ℝ :=
  sorry

/-- Theorem stating that the distance from D to the midpoint of EF is 12 units -/
theorem distance_to_midpoint_is_12 (t : IsoscelesTriangleDEF) :
  distanceToMidpoint t = 12 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_midpoint_is_12_l3840_384067


namespace NUMINAMATH_CALUDE_angle_ratio_is_one_fourth_l3840_384051

-- Define the triangle ABC
variable (A B C : Point) (ABC : Triangle A B C)

-- Define the points P and Q
variable (P Q : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- BP and BQ bisect angle ABC
axiom bp_bisects : angle A B P = angle P B C
axiom bq_bisects : angle A B Q = angle Q B C

-- BM is the bisector of angle PBQ
variable (M : Point)
axiom bm_bisects : angle P B M = angle M B Q

-- Theorem statement
theorem angle_ratio_is_one_fourth :
  (angle M B Q) / (angle A B Q) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_angle_ratio_is_one_fourth_l3840_384051


namespace NUMINAMATH_CALUDE_operations_result_l3840_384011

-- Define operation S
def S (a b : ℤ) : ℤ := 4*a + 6*b

-- Define operation T
def T (a b : ℤ) : ℤ := 5*a + 3*b

-- Theorem to prove
theorem operations_result : (S 6 3 = 42) ∧ (T 6 3 = 39) := by
  sorry

end NUMINAMATH_CALUDE_operations_result_l3840_384011


namespace NUMINAMATH_CALUDE_f_increasing_and_odd_l3840_384025

def f (x : ℝ) : ℝ := x^3

theorem f_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_and_odd_l3840_384025


namespace NUMINAMATH_CALUDE_biography_increase_l3840_384040

theorem biography_increase (B : ℝ) (N : ℝ) (h1 : N > 0) (h2 : B > 0)
  (h3 : 0.20 * B + N = 0.30 * (B + N)) :
  (N / (0.20 * B)) * 100 = 100 / 1.4 := by
  sorry

end NUMINAMATH_CALUDE_biography_increase_l3840_384040


namespace NUMINAMATH_CALUDE_arctg_sum_implies_product_sum_l3840_384094

/-- Given that arctg x + arctg y + arctg z = π/2, prove that xy + yz + zx = 1 -/
theorem arctg_sum_implies_product_sum (x y z : ℝ) 
  (h : Real.arctan x + Real.arctan y + Real.arctan z = π / 2) : 
  x * y + y * z + x * z = 1 := by
  sorry

end NUMINAMATH_CALUDE_arctg_sum_implies_product_sum_l3840_384094


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l3840_384013

theorem indefinite_integral_proof (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  let f := fun (x : ℝ) => (x^3 - 6*x^2 + 11*x - 10) / ((x+2)*(x-2)^3)
  let F := fun (x : ℝ) => Real.log (abs (x+2)) + 1 / (2*(x-2)^2)
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l3840_384013


namespace NUMINAMATH_CALUDE_polynomial_value_at_three_l3840_384024

theorem polynomial_value_at_three : 
  let x : ℝ := 3
  (x^5 : ℝ) - 7 * (x^3 : ℝ) = 54 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_three_l3840_384024


namespace NUMINAMATH_CALUDE_prime_factors_count_l3840_384068

theorem prime_factors_count : 
  let expression := [(2, 25), (3, 17), (5, 11), (7, 8), (11, 4), (13, 3)]
  (expression.map (λ (p : ℕ × ℕ) => p.2)).sum = 68 := by sorry

end NUMINAMATH_CALUDE_prime_factors_count_l3840_384068


namespace NUMINAMATH_CALUDE_overlapping_segment_length_l3840_384000

theorem overlapping_segment_length (tape_length : ℝ) (total_length : ℝ) (num_tapes : ℕ) :
  tape_length = 250 →
  total_length = 925 →
  num_tapes = 4 →
  ∃ (overlap_length : ℝ),
    overlap_length * (num_tapes - 1) = num_tapes * tape_length - total_length ∧
    overlap_length = 25 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_segment_length_l3840_384000


namespace NUMINAMATH_CALUDE_min_slope_tangent_line_l3840_384032

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

-- Theorem statement
theorem min_slope_tangent_line :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, f' x ≥ f' (-1)) ∧
    (f' (-1) = 3) ∧
    (f (-1) = -5) ∧
    (a * x + b * y + c = 0) ∧
    (a / b = f' (-1)) ∧
    ((-1) * a + (-5) * b + c = 0) ∧
    (a = 3 ∧ b = -1 ∧ c = -2) :=
by sorry

end NUMINAMATH_CALUDE_min_slope_tangent_line_l3840_384032


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l3840_384035

-- Define the given line
def given_line : ℝ → ℝ → Prop :=
  λ x y => x + 2 * y - 1 = 0

-- Define the point that the desired line passes through
def point : ℝ × ℝ := (2, 0)

-- Define the equation of the desired line
def desired_line : ℝ → ℝ → Prop :=
  λ x y => x + 2 * y - 2 = 0

-- Theorem statement
theorem line_through_point_parallel_to_given :
  (desired_line point.1 point.2) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, desired_line x y ↔ given_line (x + k) (y + k/2)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l3840_384035


namespace NUMINAMATH_CALUDE_marblesPerJar_eq_five_l3840_384065

/-- The number of marbles in each jar, given the conditions of the problem -/
def marblesPerJar : ℕ :=
  let numJars : ℕ := 16
  let numPots : ℕ := numJars / 2
  let totalMarbles : ℕ := 200
  let marblesPerPot : ℕ → ℕ := fun x ↦ 3 * x
  (totalMarbles / (numJars + numPots * 3))

theorem marblesPerJar_eq_five : marblesPerJar = 5 := by
  sorry

end NUMINAMATH_CALUDE_marblesPerJar_eq_five_l3840_384065


namespace NUMINAMATH_CALUDE_product_digits_sum_l3840_384015

/-- Converts a base-9 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 9 + d) 0

/-- Converts a base-10 number to base-9 --/
def toBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
  aux n []

/-- Sums the digits of a number represented as a list of digits --/
def sumDigits (digits : List Nat) : Nat :=
  digits.sum

theorem product_digits_sum :
  let a := [1, 2, 5]  -- 125 in base 9
  let b := [3, 3]     -- 33 in base 9
  let product := toBase10 a * toBase10 b
  sumDigits (toBase9 product) = 16 := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_l3840_384015


namespace NUMINAMATH_CALUDE_wedding_cost_theorem_l3840_384010

/-- Calculates the total cost of a wedding given the venue cost, cost per guest, 
    John's desired number of guests, and the percentage increase desired by John's wife. -/
def wedding_cost (venue_cost : ℕ) (cost_per_guest : ℕ) (john_guests : ℕ) (wife_increase_percent : ℕ) : ℕ :=
  let total_guests := john_guests + john_guests * wife_increase_percent / 100
  venue_cost + cost_per_guest * total_guests

/-- Proves that the total cost of the wedding is $50,000 given the specified conditions. -/
theorem wedding_cost_theorem : 
  wedding_cost 10000 500 50 60 = 50000 := by
  sorry

#eval wedding_cost 10000 500 50 60

end NUMINAMATH_CALUDE_wedding_cost_theorem_l3840_384010


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l3840_384030

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 4*x = x*(x + 2)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l3840_384030


namespace NUMINAMATH_CALUDE_strudel_price_calculation_l3840_384007

/-- Calculates the final price of a strudel after two 50% increases and a 50% decrease -/
def finalPrice (initialPrice : ℝ) : ℝ :=
  initialPrice * 1.5 * 1.5 * 0.5

/-- Theorem stating that the final price of a strudel is 90 rubles -/
theorem strudel_price_calculation :
  finalPrice 80 = 90 := by
  sorry

end NUMINAMATH_CALUDE_strudel_price_calculation_l3840_384007


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3840_384029

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ (y : ℝ), y^2 = x - 2) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_meaningful_l3840_384029


namespace NUMINAMATH_CALUDE_equal_areas_imply_all_equal_l3840_384064

-- Define a square
structure Square where
  side : ℝ
  area : ℝ
  area_eq : area = side * side

-- Define the four parts of the square
structure SquareParts where
  square : Square
  part1 : ℝ
  part2 : ℝ
  part3 : ℝ
  part4 : ℝ
  sum_eq_area : part1 + part2 + part3 + part4 = square.area

-- Define the perpendicular lines
structure PerpendicularLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ
  perpendicular : ∀ x y, line1 x * line2 y = -1

-- Theorem statement
theorem equal_areas_imply_all_equal (sq : Square) (parts : SquareParts) (lines : PerpendicularLines)
  (h1 : parts.square = sq)
  (h2 : parts.part1 = parts.part2)
  (h3 : parts.part2 = parts.part3)
  (h4 : ∃ x y, x ∈ Set.Icc 0 sq.side ∧ y ∈ Set.Icc 0 sq.side ∧ 
       lines.line1 x = lines.line2 y) :
  parts.part1 = parts.part2 ∧ parts.part2 = parts.part3 ∧ parts.part3 = parts.part4 :=
by sorry

end NUMINAMATH_CALUDE_equal_areas_imply_all_equal_l3840_384064


namespace NUMINAMATH_CALUDE_probability_of_divisor_of_12_l3840_384016

/-- An 8-sided die numbered from 1 to 8 -/
def Die := Finset.range 8

/-- The set of divisors of 12 that are less than or equal to 8 -/
def DivisorsOf12 : Finset ℕ := {1, 2, 3, 4, 6}

/-- The probability of rolling a divisor of 12 on an 8-sided die -/
def probability : ℚ := (DivisorsOf12.card : ℚ) / (Die.card : ℚ)

theorem probability_of_divisor_of_12 : probability = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_divisor_of_12_l3840_384016


namespace NUMINAMATH_CALUDE_miles_on_wednesday_l3840_384053

/-- Represents the miles run by Mrs. Hilt on different days of the week -/
structure RunningWeek where
  monday : ℕ
  wednesday : ℕ
  friday : ℕ
  total : ℕ

/-- Theorem stating that Mrs. Hilt ran 2 miles on Wednesday -/
theorem miles_on_wednesday (week : RunningWeek) 
  (h1 : week.monday = 3)
  (h2 : week.friday = 7)
  (h3 : week.total = 12)
  : week.wednesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_miles_on_wednesday_l3840_384053


namespace NUMINAMATH_CALUDE_star_five_three_l3840_384033

def star (a b : ℝ) : ℝ := 4 * a + 6 * b

theorem star_five_three : star 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l3840_384033


namespace NUMINAMATH_CALUDE_no_thirty_consecutive_zeros_l3840_384070

/-- For any natural number n, the last 100 digits of 5^n do not contain 30 consecutive zeros. -/
theorem no_thirty_consecutive_zeros (n : ℕ) : 
  ¬ (∃ k : ℕ, k + 29 < 100 ∧ ∀ i : ℕ, i < 30 → (5^n / 10^k) % 10^(100-k) % 10 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_thirty_consecutive_zeros_l3840_384070


namespace NUMINAMATH_CALUDE_sin_2theta_from_exp_l3840_384038

theorem sin_2theta_from_exp (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5) : 
  Real.sin (2 * θ) = 12 * Real.sqrt 2 / 25 := by
sorry

end NUMINAMATH_CALUDE_sin_2theta_from_exp_l3840_384038


namespace NUMINAMATH_CALUDE_log_and_exp_problem_l3840_384054

theorem log_and_exp_problem :
  (Real.log 9 / Real.log 3 = 2) ∧
  (∀ a : ℝ, a = Real.log 3 / Real.log 4 → 2^a = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_log_and_exp_problem_l3840_384054


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3840_384096

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 2 * x^2 - 5 * x + 2
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = 1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_solution_l3840_384096


namespace NUMINAMATH_CALUDE_max_pieces_is_18_l3840_384045

/-- Represents the size of a square cake piece -/
inductive PieceSize
  | Small : PieceSize  -- 2" x 2"
  | Medium : PieceSize -- 4" x 4"
  | Large : PieceSize  -- 6" x 6"

/-- Represents a configuration of cake pieces -/
structure CakeConfiguration where
  small_pieces : Nat
  medium_pieces : Nat
  large_pieces : Nat

/-- Checks if a given configuration fits within a 20" x 20" cake -/
def fits_in_cake (config : CakeConfiguration) : Prop :=
  2 * config.small_pieces + 4 * config.medium_pieces + 6 * config.large_pieces ≤ 400

/-- The maximum number of pieces that can be cut from the cake -/
def max_pieces : Nat := 18

/-- Theorem stating that the maximum number of pieces is 18 -/
theorem max_pieces_is_18 :
  ∀ (config : CakeConfiguration),
    fits_in_cake config →
    config.small_pieces + config.medium_pieces + config.large_pieces ≤ max_pieces :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_is_18_l3840_384045


namespace NUMINAMATH_CALUDE_range_of_negative_values_l3840_384020

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is increasing on (-∞, 0) if f(x) ≤ f(y) for all x < y < 0 -/
def IncreasingOnNegative (f : ℝ → ℝ) : Prop := ∀ x y, x < y → y < 0 → f x ≤ f y

theorem range_of_negative_values (f : ℝ → ℝ) 
  (h_even : IsEven f) 
  (h_inc_neg : IncreasingOnNegative f) 
  (h_f2 : f 2 = 0) :
  ∀ x, f x < 0 ↔ x < -2 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l3840_384020


namespace NUMINAMATH_CALUDE_cookie_distribution_l3840_384026

theorem cookie_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) :
  total_cookies = 24 →
  num_people = 6 →
  cookies_per_person = total_cookies / num_people →
  cookies_per_person = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l3840_384026


namespace NUMINAMATH_CALUDE_initial_men_correct_l3840_384041

/-- The number of men initially working on the digging project. -/
def initial_men : ℕ := 55

/-- The number of hours worked per day in the initial condition. -/
def initial_hours : ℕ := 8

/-- The depth dug in meters in the initial condition. -/
def initial_depth : ℕ := 30

/-- The number of hours worked per day in the new condition. -/
def new_hours : ℕ := 6

/-- The depth to be dug in meters in the new condition. -/
def new_depth : ℕ := 50

/-- The additional number of men needed for the new condition. -/
def extra_men : ℕ := 11

/-- Theorem stating that the initial number of men is correct given the conditions. -/
theorem initial_men_correct :
  initial_men * initial_hours * initial_depth = (initial_men + extra_men) * new_hours * new_depth :=
by sorry

end NUMINAMATH_CALUDE_initial_men_correct_l3840_384041


namespace NUMINAMATH_CALUDE_one_linear_two_var_l3840_384046

-- Define the structure of an expression
inductive Expression
  | Linear (a b : ℝ) (c : ℝ) : Expression  -- ax + by = c
  | Rational (a : ℝ) (b : ℝ) (c : ℝ) : Expression  -- x + b/y = c
  | ThreeVar (a b c : ℝ) (d : ℝ) : Expression  -- ax + by + cz = d
  | Incomplete (a b : ℝ) : Expression  -- ax + by
  | Inequality (a b : ℝ) (c : ℝ) : Expression  -- ax - by > c

def is_linear_two_var (e : Expression) : Bool :=
  match e with
  | Expression.Linear _ _ _ => true
  | _ => false

def count_linear_two_var (exprs : List Expression) : Nat :=
  exprs.filter is_linear_two_var |>.length

def given_expressions : List Expression :=
  [Expression.Linear 2 (-3) 5,
   Expression.Rational 1 3 6,
   Expression.ThreeVar 3 (-1) 2 0,
   Expression.Incomplete 2 4,
   Expression.Inequality 5 (-1) 0]

theorem one_linear_two_var :
  count_linear_two_var given_expressions = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_linear_two_var_l3840_384046


namespace NUMINAMATH_CALUDE_linear_function_range_l3840_384001

theorem linear_function_range (x y : ℝ) :
  y = -2 * x + 3 →
  y ≤ 6 →
  x ≥ -3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_range_l3840_384001


namespace NUMINAMATH_CALUDE_solution_to_equation_l3840_384014

theorem solution_to_equation : ∃ (x y : ℤ), x + 3 * y = 7 ∧ x = -2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3840_384014


namespace NUMINAMATH_CALUDE_bakery_combinations_l3840_384074

/-- The number of ways to distribute n identical items among k groups -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of roll types -/
def num_roll_types : ℕ := 4

/-- The number of remaining rolls to distribute -/
def remaining_rolls : ℕ := 2

theorem bakery_combinations :
  distribute remaining_rolls num_roll_types = 10 := by
  sorry

end NUMINAMATH_CALUDE_bakery_combinations_l3840_384074


namespace NUMINAMATH_CALUDE_sphere_volume_from_inscribed_cube_l3840_384023

theorem sphere_volume_from_inscribed_cube (s : ℝ) (h : s > 0) :
  let cube_surface_area := 6 * s^2
  let cube_diagonal := s * Real.sqrt 3
  let sphere_radius := cube_diagonal / 2
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius^3
  cube_surface_area = 24 → sphere_volume = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_inscribed_cube_l3840_384023


namespace NUMINAMATH_CALUDE_increasing_interval_of_f_l3840_384059

noncomputable def f (x : ℝ) := x - Real.exp x

theorem increasing_interval_of_f :
  {x : ℝ | ∀ y, x < y → f x < f y} = Set.Iio 0 :=
sorry

end NUMINAMATH_CALUDE_increasing_interval_of_f_l3840_384059


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3840_384028

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem perpendicular_line_equation :
  ∃ (x0 y0 : ℝ), intersection_point x0 y0 ∧
  ∃ (m : ℝ), perpendicular m (-2) ∧
  ∀ (x y : ℝ), y - y0 = m * (x - x0) ↔ x - 2 * y + 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3840_384028


namespace NUMINAMATH_CALUDE_complex_to_exponential_l3840_384058

theorem complex_to_exponential (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → z = 2 * Complex.exp (Complex.I * (Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_complex_to_exponential_l3840_384058


namespace NUMINAMATH_CALUDE_cone_angle_cosine_l3840_384056

/-- Given a cone whose side surface unfolds into a sector with central angle 4π/3 and radius 18 cm,
    prove that the cosine of the angle between the slant height and the base is 2/3 -/
theorem cone_angle_cosine (θ : Real) (l r : Real) : 
  θ = 4 / 3 * π → 
  l = 18 → 
  θ = 2 * π * r / l → 
  r / l = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_cone_angle_cosine_l3840_384056


namespace NUMINAMATH_CALUDE_raffle_probabilities_l3840_384099

/-- Represents a raffle with a fixed number of participants, white balls, and black balls -/
structure Raffle :=
  (participants : ℕ)
  (white_balls : ℕ)
  (black_balls : ℕ)

/-- Calculates the probability of a participant winning based on their position in the drawing order -/
noncomputable def win_probability (r : Raffle) (position : ℕ) : ℚ :=
  sorry

theorem raffle_probabilities :
  let raffle1 := Raffle.mk 4 3 1
  let raffle2 := Raffle.mk 4 6 2
  (win_probability raffle1 1 = 1/4) ∧ 
  (win_probability raffle1 4 = 1/4) ∧
  (win_probability raffle2 1 = 5/14) ∧
  (win_probability raffle2 4 = 1/7) :=
by sorry

end NUMINAMATH_CALUDE_raffle_probabilities_l3840_384099


namespace NUMINAMATH_CALUDE_fib_100_div_5_l3840_384006

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The Fibonacci sequence modulo 5 repeats every 20 terms -/
axiom fib_mod_5_period : ∀ n : ℕ, fib (n + 20) % 5 = fib n % 5

theorem fib_100_div_5 : 5 ∣ fib 100 := by
  sorry

end NUMINAMATH_CALUDE_fib_100_div_5_l3840_384006


namespace NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l3840_384092

theorem base_b_not_divisible_by_five (b : ℤ) : b ∈ ({5, 6, 7, 8, 10} : Set ℤ) →
  (b^2 * (3*b - 2) % 5 ≠ 0 ↔ b = 8) := by
  sorry

end NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l3840_384092


namespace NUMINAMATH_CALUDE_prob_red_black_correct_l3840_384082

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (h_total : total_cards = 52)
  (h_red : red_cards = 26)
  (h_black : black_cards = 26)
  (h_sum : red_cards + black_cards = total_cards)

/-- The probability of drawing one red card and one black card in the first two draws -/
def prob_red_black (d : Deck) : ℚ :=
  26 / 51

theorem prob_red_black_correct (d : Deck) : 
  prob_red_black d = 26 / 51 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_black_correct_l3840_384082


namespace NUMINAMATH_CALUDE_cubic_minus_four_ab_squared_factorization_l3840_384087

theorem cubic_minus_four_ab_squared_factorization (a b : ℝ) :
  a^3 - 4*a*b^2 = a*(a+2*b)*(a-2*b) := by sorry

end NUMINAMATH_CALUDE_cubic_minus_four_ab_squared_factorization_l3840_384087


namespace NUMINAMATH_CALUDE_fruit_sales_theorem_l3840_384031

/-- The standard weight of a batch of fruits in kilograms -/
def standard_weight : ℕ := 30

/-- The weight deviations from the standard weight -/
def weight_deviations : List ℤ := [9, -10, -5, 6, -7, -6, 7, 10]

/-- The price per kilogram on the first day in yuan -/
def price_per_kg : ℕ := 10

/-- The discount rate for the second day as a rational number -/
def discount_rate : ℚ := 1/10

theorem fruit_sales_theorem :
  let total_weight := (List.sum weight_deviations + standard_weight * weight_deviations.length : ℤ)
  let first_day_sales := (price_per_kg * (total_weight / 2) : ℚ)
  let second_day_sales := (price_per_kg * (1 - discount_rate) * (total_weight - total_weight / 2) : ℚ)
  total_weight = 244 ∧ (first_day_sales + second_day_sales : ℚ) = 2318 := by
  sorry

end NUMINAMATH_CALUDE_fruit_sales_theorem_l3840_384031


namespace NUMINAMATH_CALUDE_salary_increase_l3840_384009

theorem salary_increase (S : ℝ) (h1 : S > 0) : 
  0.08 * (S + S * (10 / 100)) = 1.4667 * (0.06 * S) := by
  sorry

#check salary_increase

end NUMINAMATH_CALUDE_salary_increase_l3840_384009


namespace NUMINAMATH_CALUDE_triple_sum_product_equation_l3840_384036

theorem triple_sum_product_equation (x y z : ℕ+) : 
  x ≤ y ∧ y ≤ z ∧ x * y + y * z + z * x - x * y * z = 2 ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 2 ∧ y = 3 ∧ z = 4) := by
  sorry

end NUMINAMATH_CALUDE_triple_sum_product_equation_l3840_384036


namespace NUMINAMATH_CALUDE_die_throw_outcomes_l3840_384027

/-- Represents the number of sides on a fair cubic die -/
def numSides : ℕ := 6

/-- Represents the number of throws -/
def numThrows : ℕ := 4

/-- Represents the number of different outcomes required to stop -/
def differentOutcomes : ℕ := 3

/-- Calculates the total number of different outcomes for the die throws -/
def totalOutcomes : ℕ := numSides * (numSides - 1) * (numSides - 2) * differentOutcomes

theorem die_throw_outcomes :
  totalOutcomes = 270 :=
sorry

end NUMINAMATH_CALUDE_die_throw_outcomes_l3840_384027


namespace NUMINAMATH_CALUDE_total_books_l3840_384073

-- Define the number of books for each person
def sam_books : ℕ := 110

-- Joan has twice as many books as Sam
def joan_books : ℕ := 2 * sam_books

-- Tom has half the number of books as Joan
def tom_books : ℕ := joan_books / 2

-- Alice has 3 times the number of books Tom has
def alice_books : ℕ := 3 * tom_books

-- Theorem statement
theorem total_books : sam_books + joan_books + tom_books + alice_books = 770 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l3840_384073


namespace NUMINAMATH_CALUDE_shaded_squares_area_sum_square_division_problem_l3840_384034

theorem shaded_squares_area_sum (initial_area : ℝ) (ratio : ℝ) :
  initial_area > 0 →
  ratio > 0 →
  ratio < 1 →
  let series_sum := initial_area / (1 - ratio)
  series_sum = initial_area / (1 - ratio) :=
by sorry

theorem square_division_problem :
  let initial_side_length : ℝ := 8
  let initial_area : ℝ := initial_side_length ^ 2
  let ratio : ℝ := 1 / 4
  let series_sum := initial_area / (1 - ratio)
  series_sum = 64 / 3 :=
by sorry

end NUMINAMATH_CALUDE_shaded_squares_area_sum_square_division_problem_l3840_384034


namespace NUMINAMATH_CALUDE_coat_drive_total_l3840_384072

theorem coat_drive_total (high_school_coats : ℕ) (elementary_school_coats : ℕ) 
  (h1 : high_school_coats = 6922)
  (h2 : elementary_school_coats = 2515) :
  high_school_coats + elementary_school_coats = 9437 := by
  sorry

end NUMINAMATH_CALUDE_coat_drive_total_l3840_384072


namespace NUMINAMATH_CALUDE_multiplier_value_l3840_384078

theorem multiplier_value (n : ℕ) (increase : ℕ) (result : ℕ) : 
  n = 14 → increase = 196 → result = 15 → n * result = n + increase :=
by
  sorry

end NUMINAMATH_CALUDE_multiplier_value_l3840_384078


namespace NUMINAMATH_CALUDE_isosceles_tetrahedron_ratio_bounds_l3840_384003

/-- An isosceles tetrahedron with edge lengths a, b, and c. -/
structure IsoscelesTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The circumradius of the tetrahedron. -/
noncomputable def R (t : IsoscelesTetrahedron) : ℝ :=
  sorry

/-- The circumradius of the base triangle. -/
noncomputable def r (t : IsoscelesTetrahedron) : ℝ :=
  sorry

/-- The theorem stating the bounds of the ratio r/R. -/
theorem isosceles_tetrahedron_ratio_bounds (t : IsoscelesTetrahedron) :
    2 * Real.sqrt 2 / 3 ≤ r t / R t ∧ r t / R t < 1 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_tetrahedron_ratio_bounds_l3840_384003


namespace NUMINAMATH_CALUDE_bottles_per_case_l3840_384083

/-- The number of bottles a case can hold, given the total daily production and number of cases required. -/
theorem bottles_per_case (total_bottles : ℕ) (total_cases : ℕ) 
  (h1 : total_bottles = 72000) 
  (h2 : total_cases = 8000) : 
  total_bottles / total_cases = 9 := by
sorry

end NUMINAMATH_CALUDE_bottles_per_case_l3840_384083


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l3840_384050

-- Define the vectors
def a : ℝ × ℝ := (2, 5)
def b : ℝ → ℝ × ℝ := λ y ↦ (1, y)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem parallel_vectors_y_value :
  parallel a (b y) → y = 5/2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l3840_384050


namespace NUMINAMATH_CALUDE_base8_to_base6_conversion_l3840_384039

-- Define a function to convert from base 8 to base 10
def base8ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

-- Define a function to convert from base 10 to base 6
def base10ToBase6 (n : ℕ) : ℕ :=
  (n / 216) * 1000 + ((n / 36) % 6) * 100 + ((n / 6) % 6) * 10 + (n % 6)

-- Theorem statement
theorem base8_to_base6_conversion :
  base10ToBase6 (base8ToBase10 753) = 2135 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base6_conversion_l3840_384039


namespace NUMINAMATH_CALUDE_fraction_simplification_l3840_384043

theorem fraction_simplification (a : ℝ) (h : a ≠ 5) :
  (a^2 - 5*a) / (a - 5) = a := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3840_384043


namespace NUMINAMATH_CALUDE_vector_relationships_l3840_384052

/-- Given vector a and unit vector b, prove their parallel and perpendicular relationships -/
theorem vector_relationships (a b : ℝ × ℝ) :
  a = (3, 4) →
  norm b = 1 →
  (b.1 * a.2 = b.2 * a.1 → b = (3/5, 4/5) ∨ b = (-3/5, -4/5)) ∧
  (b.1 * a.1 + b.2 * a.2 = 0 → b = (-4/5, 3/5) ∨ b = (4/5, -3/5)) :=
by sorry

end NUMINAMATH_CALUDE_vector_relationships_l3840_384052


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_def_l3840_384088

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (a : ℝ) : ℝ := Real.sqrt a

-- State the theorem
theorem arithmetic_sqrt_def (a : ℝ) (h : 0 < a) : 
  arithmetic_sqrt a = Real.sqrt a := by sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_def_l3840_384088


namespace NUMINAMATH_CALUDE_paint_cans_used_l3840_384048

/-- Represents the number of rooms that can be painted with a given amount of paint -/
structure PaintCapacity where
  rooms : ℕ

/-- Represents the number of paint cans -/
structure PaintCans where
  count : ℕ

/-- The painting scenario -/
structure PaintingScenario where
  initialCapacity : PaintCapacity
  lostCans : PaintCans
  finalCapacity : PaintCapacity

/-- The theorem to prove -/
theorem paint_cans_used (scenario : PaintingScenario) 
  (h1 : scenario.initialCapacity.rooms = 40)
  (h2 : scenario.lostCans.count = 5)
  (h3 : scenario.finalCapacity.rooms = 30) :
  ∃ (usedCans : PaintCans), usedCans.count = 15 ∧ 
    usedCans.count * (scenario.initialCapacity.rooms / (scenario.initialCapacity.rooms - scenario.finalCapacity.rooms)) = scenario.finalCapacity.rooms :=
by sorry

end NUMINAMATH_CALUDE_paint_cans_used_l3840_384048


namespace NUMINAMATH_CALUDE_equal_roots_condition_l3840_384076

theorem equal_roots_condition (m : ℝ) : 
  (∃ (x : ℝ), (x * (x - 3) - (m + 2)) / ((x - 2) * (m - 2)) = x / m) ∧ 
  (∀ (x y : ℝ), (x * (x - 3) - (m + 2)) / ((x - 2) * (m - 2)) = x / m ∧ 
                 (y * (y - 3) - (m + 2)) / ((y - 2) * (m - 2)) = y / m 
                 → x = y) ↔ 
  m = (-7 + Real.sqrt 2) / 2 ∨ m = (-7 - Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l3840_384076


namespace NUMINAMATH_CALUDE_one_prime_between_90_and_100_l3840_384047

theorem one_prime_between_90_and_100 : 
  ∃! p, Prime p ∧ 90 < p ∧ p < 100 := by
sorry

end NUMINAMATH_CALUDE_one_prime_between_90_and_100_l3840_384047


namespace NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l3840_384061

/-- The maximum y-coordinate of a point on the graph of r = sin 3θ is 9/8 -/
theorem max_y_coordinate_sin_3theta (θ : Real) :
  let r := Real.sin (3 * θ)
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  ∀ y', y' = r * Real.sin θ → y' ≤ 9/8 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l3840_384061


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_l3840_384049

theorem sum_of_last_two_digits (n : ℕ) : (7^25 + 13^25) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_l3840_384049


namespace NUMINAMATH_CALUDE_lemon_square_price_is_correct_l3840_384021

/-- Represents the price of a lemon square -/
def lemon_square_price : ℝ := 2

/-- The number of brownies sold -/
def brownies_sold : ℕ := 4

/-- The price of each brownie -/
def brownie_price : ℝ := 3

/-- The number of lemon squares sold -/
def lemon_squares_sold : ℕ := 5

/-- The number of cookies to be sold -/
def cookies_to_sell : ℕ := 7

/-- The price of each cookie -/
def cookie_price : ℝ := 4

/-- The total revenue goal -/
def total_revenue_goal : ℝ := 50

theorem lemon_square_price_is_correct :
  (brownies_sold : ℝ) * brownie_price +
  (lemon_squares_sold : ℝ) * lemon_square_price +
  (cookies_to_sell : ℝ) * cookie_price =
  total_revenue_goal :=
by sorry

end NUMINAMATH_CALUDE_lemon_square_price_is_correct_l3840_384021


namespace NUMINAMATH_CALUDE_economic_output_equals_scientific_notation_l3840_384081

/-- Represents the economic output in yuan -/
def economic_output : ℝ := 4500 * 1000000000

/-- The scientific notation representation of the economic output -/
def scientific_notation : ℝ := 4.5 * (10 ^ 12)

/-- Theorem stating that the economic output is equal to its scientific notation representation -/
theorem economic_output_equals_scientific_notation : 
  economic_output = scientific_notation := by sorry

end NUMINAMATH_CALUDE_economic_output_equals_scientific_notation_l3840_384081


namespace NUMINAMATH_CALUDE_expression_equivalence_l3840_384002

theorem expression_equivalence (a : ℝ) : 
  (a^2 + a - 2) / (a^2 + 3*a + 2) * (5 * (a + 1)^2) = 5*a^2 - 5 :=
by sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3840_384002


namespace NUMINAMATH_CALUDE_unique_prime_103207_l3840_384086

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_prime_103207 :
  (¬ is_prime 103201) ∧
  (¬ is_prime 103202) ∧
  (¬ is_prime 103203) ∧
  (is_prime 103207) ∧
  (¬ is_prime 103209) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_103207_l3840_384086


namespace NUMINAMATH_CALUDE_common_chord_length_l3840_384019

theorem common_chord_length (r : ℝ) (d : ℝ) (h1 : r = 12) (h2 : d = 16) :
  let chord_length := 2 * Real.sqrt (r^2 - (d/2)^2)
  chord_length = 8 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_common_chord_length_l3840_384019


namespace NUMINAMATH_CALUDE_mushroom_picking_ratio_l3840_384089

/-- Proves the ratio of mushrooms picked on the last day to the second day -/
theorem mushroom_picking_ratio : 
  ∀ (total_mushrooms first_day_revenue second_day_picked price_per_mushroom : ℕ),
  total_mushrooms = 65 →
  first_day_revenue = 58 →
  second_day_picked = 12 →
  price_per_mushroom = 2 →
  (total_mushrooms - first_day_revenue / price_per_mushroom - second_day_picked) / second_day_picked = 2 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_picking_ratio_l3840_384089


namespace NUMINAMATH_CALUDE_hardcover_nonfiction_count_l3840_384060

/-- Represents the number of books of each type --/
structure BookCounts where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ
  hardcoverFiction : ℕ

/-- The total number of books --/
def totalBooks : ℕ := 10000

/-- Conditions for the book counts --/
def validBookCounts (bc : BookCounts) : Prop :=
  bc.paperbackFiction + bc.paperbackNonfiction + bc.hardcoverNonfiction + bc.hardcoverFiction = totalBooks ∧
  bc.paperbackNonfiction = bc.hardcoverNonfiction + 100 ∧
  bc.paperbackFiction * 3 = bc.hardcoverFiction * 5 ∧
  bc.hardcoverFiction = totalBooks / 100 * 12 ∧
  bc.paperbackNonfiction + bc.hardcoverNonfiction = totalBooks / 100 * 30

theorem hardcover_nonfiction_count (bc : BookCounts) (h : validBookCounts bc) : 
  bc.hardcoverNonfiction = 1450 := by
  sorry

end NUMINAMATH_CALUDE_hardcover_nonfiction_count_l3840_384060


namespace NUMINAMATH_CALUDE_sugar_per_larger_cookie_l3840_384012

/-- Proves that if 40 cookies each use 1/8 cup of sugar, and the same total amount of sugar
    is used to make 25 larger cookies, then each larger cookie will contain 1/5 cup of sugar. -/
theorem sugar_per_larger_cookie :
  let small_cookies : ℕ := 40
  let large_cookies : ℕ := 25
  let sugar_per_small : ℚ := 1 / 8
  let total_sugar : ℚ := small_cookies * sugar_per_small
  let sugar_per_large : ℚ := total_sugar / large_cookies
  sugar_per_large = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sugar_per_larger_cookie_l3840_384012


namespace NUMINAMATH_CALUDE_matthew_initial_cakes_l3840_384071

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 29

/-- The number of friends Matthew gave crackers and cakes to -/
def num_friends : ℕ := 2

/-- The number of cakes each person ate -/
def cakes_eaten_per_person : ℕ := 15

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := initial_crackers + num_friends * cakes_eaten_per_person

theorem matthew_initial_cakes : initial_cakes = 59 := by sorry

end NUMINAMATH_CALUDE_matthew_initial_cakes_l3840_384071


namespace NUMINAMATH_CALUDE_ryan_quiz_goal_l3840_384095

theorem ryan_quiz_goal (total_quizzes : ℕ) (goal_percentage : ℚ)
  (mid_year_quizzes : ℕ) (mid_year_as : ℕ) (h1 : total_quizzes = 60)
  (h2 : goal_percentage = 3/4) (h3 : mid_year_quizzes = 40) (h4 : mid_year_as = 30) :
  ∃ (max_lower_grade : ℕ),
    max_lower_grade = 5 ∧
    (mid_year_as + (total_quizzes - mid_year_quizzes - max_lower_grade) : ℚ) / total_quizzes ≥ goal_percentage :=
by sorry

end NUMINAMATH_CALUDE_ryan_quiz_goal_l3840_384095


namespace NUMINAMATH_CALUDE_vanessa_picked_17_carrots_l3840_384062

/-- The number of carrots Vanessa picked -/
def vanessas_carrots (good_carrots bad_carrots moms_carrots : ℕ) : ℕ :=
  good_carrots + bad_carrots - moms_carrots

/-- Proof that Vanessa picked 17 carrots -/
theorem vanessa_picked_17_carrots :
  vanessas_carrots 24 7 14 = 17 := by
  sorry

end NUMINAMATH_CALUDE_vanessa_picked_17_carrots_l3840_384062


namespace NUMINAMATH_CALUDE_escalator_travel_time_l3840_384057

/-- Proves that a person walking on a moving escalator takes 10 seconds to cover the entire length -/
theorem escalator_travel_time
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (person_speed : ℝ)
  (h1 : escalator_speed = 11)
  (h2 : escalator_length = 140)
  (h3 : person_speed = 3) :
  escalator_length / (escalator_speed + person_speed) = 10 := by
  sorry


end NUMINAMATH_CALUDE_escalator_travel_time_l3840_384057


namespace NUMINAMATH_CALUDE_photograph_perimeter_l3840_384075

theorem photograph_perimeter (w h m : ℝ) 
  (area_with_2inch_border : (w + 4) * (h + 4) = m)
  (area_with_4inch_border : (w + 8) * (h + 8) = m + 94)
  : 2 * (w + h) = 23 := by
  sorry

end NUMINAMATH_CALUDE_photograph_perimeter_l3840_384075


namespace NUMINAMATH_CALUDE_population_growth_percentage_l3840_384080

theorem population_growth_percentage (a b c : ℝ) : 
  let growth_factor_1 := 1 + a / 100
  let growth_factor_2 := 1 + b / 100
  let growth_factor_3 := 1 + c / 100
  let total_growth := growth_factor_1 * growth_factor_2 * growth_factor_3
  (total_growth - 1) * 100 = a + b + c + (a * b + a * c + b * c) / 100 + a * b * c / 10000 := by
sorry

end NUMINAMATH_CALUDE_population_growth_percentage_l3840_384080


namespace NUMINAMATH_CALUDE_cans_display_rows_l3840_384004

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 2 * n + 1

/-- The total number of cans in a display with n rows -/
def total_cans (n : ℕ) : ℕ := n * (n + 2)

/-- The number of rows in the display -/
def num_rows : ℕ := 12

theorem cans_display_rows :
  (cans_in_row 1 = 3) ∧
  (∀ n : ℕ, n > 0 → cans_in_row (n + 1) = cans_in_row n + 2) ∧
  (total_cans num_rows = 169) ∧
  (∀ m : ℕ, m ≠ num_rows → total_cans m ≠ 169) :=
by sorry

end NUMINAMATH_CALUDE_cans_display_rows_l3840_384004


namespace NUMINAMATH_CALUDE_square_root_of_81_l3840_384084

theorem square_root_of_81 : 
  {x : ℝ | x^2 = 81} = {9, -9} := by sorry

end NUMINAMATH_CALUDE_square_root_of_81_l3840_384084


namespace NUMINAMATH_CALUDE_division_value_problem_l3840_384066

theorem division_value_problem (x : ℝ) : (3 / x) * 12 = 9 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_value_problem_l3840_384066


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3840_384085

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 9 → 1/a < 1/9) ∧ 
  (∃ a, 1/a < 1/9 ∧ ¬(a > 9)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3840_384085


namespace NUMINAMATH_CALUDE_trapezoid_longer_base_l3840_384008

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  midline : ℝ
  midline_difference : ℝ
  longer_base : ℝ
  shorter_base : ℝ

/-- The theorem stating the properties of the specific trapezoid -/
theorem trapezoid_longer_base 
  (t : Trapezoid) 
  (h1 : t.midline = 10)
  (h2 : t.midline_difference = 3)
  (h3 : t.midline = (t.longer_base + t.shorter_base) / 2)
  (h4 : t.midline_difference = (t.longer_base - t.shorter_base) / 2) :
  t.longer_base = 13 := by
    sorry


end NUMINAMATH_CALUDE_trapezoid_longer_base_l3840_384008


namespace NUMINAMATH_CALUDE_data_median_and_variance_l3840_384022

def data : List ℝ := [5, 9, 8, 8, 10]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem data_median_and_variance :
  median data = 8 ∧ variance data = 2.8 := by sorry

end NUMINAMATH_CALUDE_data_median_and_variance_l3840_384022


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_l3840_384097

def repeating_decimal : ℚ := 0.145145145

theorem repeating_decimal_fraction :
  repeating_decimal = 145 / 999 ∧
  145 + 999 = 1144 := by
  sorry

#check repeating_decimal_fraction

end NUMINAMATH_CALUDE_repeating_decimal_fraction_l3840_384097


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3840_384017

theorem complex_equation_solution (z : ℂ) : 
  Complex.abs z - 2 * z = -1 + 8 * Complex.I → 
  z = 3 - 4 * Complex.I ∨ z = -5/3 - 4 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3840_384017


namespace NUMINAMATH_CALUDE_paco_cookies_l3840_384044

/-- Calculates the total number of cookies Paco has after buying cookies with a promotion --/
def total_cookies (initial : ℕ) (eaten : ℕ) (bought : ℕ) : ℕ :=
  let remaining := initial - eaten
  let free := 2 * bought
  let from_bakery := bought + free
  remaining + from_bakery

/-- Proves that Paco ends up with 149 cookies given the initial conditions --/
theorem paco_cookies : total_cookies 40 2 37 = 149 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l3840_384044


namespace NUMINAMATH_CALUDE_conical_funnel_area_l3840_384005

/-- The area of cardboard required for a conical funnel -/
theorem conical_funnel_area (slant_height : ℝ) (base_circumference : ℝ) 
  (h1 : slant_height = 6)
  (h2 : base_circumference = 6 * Real.pi) : 
  (1 / 2 : ℝ) * base_circumference * slant_height = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_conical_funnel_area_l3840_384005


namespace NUMINAMATH_CALUDE_trishSellPriceIs150Cents_l3840_384077

/-- The price at which Trish sells each stuffed animal -/
def trishSellPrice (barbaraStuffedAnimals : ℕ) (barbaraSellPrice : ℚ) (totalDonation : ℚ) : ℚ :=
  let trishStuffedAnimals := 2 * barbaraStuffedAnimals
  let barbaraContribution := barbaraStuffedAnimals * barbaraSellPrice
  let trishContribution := totalDonation - barbaraContribution
  trishContribution / trishStuffedAnimals

theorem trishSellPriceIs150Cents 
  (barbaraStuffedAnimals : ℕ) 
  (barbaraSellPrice : ℚ) 
  (totalDonation : ℚ) 
  (h1 : barbaraStuffedAnimals = 9)
  (h2 : barbaraSellPrice = 2)
  (h3 : totalDonation = 45) :
  trishSellPrice barbaraStuffedAnimals barbaraSellPrice totalDonation = 3/2 := by
  sorry

#eval trishSellPrice 9 2 45

end NUMINAMATH_CALUDE_trishSellPriceIs150Cents_l3840_384077


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l3840_384069

theorem third_root_of_cubic (a b : ℝ) : 
  (∀ x : ℝ, a * x^3 + (a + 2*b) * x^2 + (b - 3*a) * x + (8 - a) = 0 ↔ x = -2 ∨ x = 3 ∨ x = 4/3) →
  ∃ x : ℝ, x ≠ -2 ∧ x ≠ 3 ∧ a * x^3 + (a + 2*b) * x^2 + (b - 3*a) * x + (8 - a) = 0 ∧ x = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l3840_384069


namespace NUMINAMATH_CALUDE_calculation_proof_l3840_384042

theorem calculation_proof : (-3/4 - 5/9 + 7/12) / (-1/36) = 26 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3840_384042


namespace NUMINAMATH_CALUDE_ratio_equality_l3840_384018

theorem ratio_equality (a b c d : ℝ) 
  (h1 : a = 5 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 6 * d) : 
  (a + b * c) / (c + d * b) = 3 * (5 + 6 * d) / (1 + 3 * d) := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3840_384018
