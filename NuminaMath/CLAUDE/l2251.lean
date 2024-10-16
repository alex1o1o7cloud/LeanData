import Mathlib

namespace NUMINAMATH_CALUDE_three_numbers_average_l2251_225119

theorem three_numbers_average (a b c : ℝ) 
  (h1 : a + (b + c) / 2 = 65)
  (h2 : b + (a + c) / 2 = 69)
  (h3 : c + (a + b) / 2 = 76)
  : (a + b + c) / 3 = 35 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_average_l2251_225119


namespace NUMINAMATH_CALUDE_parabola_min_distance_l2251_225175

/-- Represents a parabola y^2 = 2px with p > 0 -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- The minimum distance from any point on the parabola to its focus is 1 -/
def min_distance_to_focus (para : Parabola) : Prop :=
  ∃ (x y : ℝ), y^2 = 2 * para.p * x ∧ 1 ≤ Real.sqrt ((x - para.p/2)^2 + y^2)

/-- If the minimum distance from any point on the parabola to the focus is 1, then p = 2 -/
theorem parabola_min_distance (para : Parabola) 
    (h_min : min_distance_to_focus para) : para.p = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_min_distance_l2251_225175


namespace NUMINAMATH_CALUDE_emerald_puzzle_l2251_225100

theorem emerald_puzzle :
  ∃ n : ℕ,
    n > 0 ∧
    n % 8 = 5 ∧
    n % 7 = 6 ∧
    (∀ m : ℕ, m > 0 ∧ m % 8 = 5 ∧ m % 7 = 6 → n ≤ m) ∧
    n % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_emerald_puzzle_l2251_225100


namespace NUMINAMATH_CALUDE_min_buses_for_535_students_l2251_225191

/-- The minimum number of buses needed to transport a given number of students -/
def min_buses (capacity : ℕ) (students : ℕ) : ℕ :=
  (students + capacity - 1) / capacity

/-- Theorem: Given a bus capacity of 45 students and 535 students to transport,
    the minimum number of buses needed is 12 -/
theorem min_buses_for_535_students :
  min_buses 45 535 = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_buses_for_535_students_l2251_225191


namespace NUMINAMATH_CALUDE_positive_reals_inequality_l2251_225152

theorem positive_reals_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^3 = x - y) : x^2 + 4*y^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequality_l2251_225152


namespace NUMINAMATH_CALUDE_same_prime_divisors_same_outcome_l2251_225129

/-- The number game as described in the problem -/
def NumberGame (k : ℕ) (n : ℕ) : Prop :=
  k > 2 ∧ n ≥ k

/-- A number is good if Banana has a winning strategy -/
def IsGood (k : ℕ) (n : ℕ) : Prop :=
  NumberGame k n ∧ sorry -- Definition of good number

/-- Two numbers have the same prime divisors up to k -/
def SamePrimeDivisorsUpTo (k : ℕ) (n n' : ℕ) : Prop :=
  ∀ p : ℕ, p ≤ k → Prime p → (p ∣ n ↔ p ∣ n')

/-- Main theorem: numbers with same prime divisors up to k have the same game outcome -/
theorem same_prime_divisors_same_outcome (k : ℕ) (n n' : ℕ) :
  NumberGame k n → NumberGame k n' → SamePrimeDivisorsUpTo k n n' →
  (IsGood k n ↔ IsGood k n') :=
sorry

end NUMINAMATH_CALUDE_same_prime_divisors_same_outcome_l2251_225129


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2251_225198

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^6 + 2*x^5 - 3*x^4 - 4*x^3 + 5*x^2 - 6*x + 7 = 
  (x^2 - 1) * (x - 2) * q + (-3*x^2 - 8*x + 13) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2251_225198


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l2251_225128

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the arithmetic sequence property
def arithmetic_sequence_property (a : ℕ → ℝ) (q : ℝ) :=
  (3 / 2) * (a 1 * q) = (2 * a 0 + a 0 * q^2) / 2

-- Theorem statement
theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : arithmetic_sequence_property a q) :
  q = 1 ∨ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l2251_225128


namespace NUMINAMATH_CALUDE_unique_perfect_square_solution_l2251_225122

/-- A number is a perfect square if it's equal to some integer squared. -/
def IsPerfectSquare (x : ℤ) : Prop := ∃ k : ℤ, x = k^2

/-- The theorem states that 125 is the only positive integer n such that
    both 20n and 5n + 275 are perfect squares. -/
theorem unique_perfect_square_solution :
  ∀ n : ℕ+, (IsPerfectSquare (20 * n.val)) ∧ (IsPerfectSquare (5 * n.val + 275)) ↔ n = 125 := by
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_solution_l2251_225122


namespace NUMINAMATH_CALUDE_right_triangle_circumcircle_area_relation_l2251_225126

/-- Represents a right triangle with its circumscribed circle -/
structure RightTriangleWithCircumcircle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  area_A : ℝ
  area_B : ℝ
  area_C : ℝ

/-- The theorem to be proved -/
theorem right_triangle_circumcircle_area_relation 
  (t : RightTriangleWithCircumcircle)
  (h_right_triangle : t.side1^2 + t.side2^2 = t.hypotenuse^2)
  (h_sides : t.side1 = 15 ∧ t.side2 = 36 ∧ t.hypotenuse = 39)
  (h_largest_C : t.area_C ≥ t.area_A ∧ t.area_C ≥ t.area_B)
  (h_non_negative : t.area_A ≥ 0 ∧ t.area_B ≥ 0 ∧ t.area_C ≥ 0)
  : t.area_A + t.area_B + 270 = t.area_C := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circumcircle_area_relation_l2251_225126


namespace NUMINAMATH_CALUDE_problem_solution_l2251_225144

theorem problem_solution (x : ℝ) : 
  x > 0 → -- x is positive
  x * (x / 100) = 9 → -- x% of x is 9
  ∃ k : ℤ, x = 3 * k → -- x is a multiple of 3
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2251_225144


namespace NUMINAMATH_CALUDE_complex_square_one_minus_i_l2251_225123

theorem complex_square_one_minus_i :
  (1 - Complex.I) ^ 2 = -2 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_square_one_minus_i_l2251_225123


namespace NUMINAMATH_CALUDE_class_average_weight_l2251_225134

theorem class_average_weight (n₁ : ℕ) (w₁ : ℝ) (n₂ : ℕ) (w₂ : ℝ) :
  n₁ = 16 →
  w₁ = 50.25 →
  n₂ = 8 →
  w₂ = 45.15 →
  (n₁ * w₁ + n₂ * w₂) / (n₁ + n₂ : ℝ) = 48.55 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l2251_225134


namespace NUMINAMATH_CALUDE_candles_used_l2251_225199

/-- Given a candle that lasts 8 nights when burned for 1 hour per night,
    calculate the number of candles used when burned for 2 hours per night for 24 nights -/
theorem candles_used
  (nights_per_candle : ℕ)
  (hours_per_night : ℕ)
  (total_nights : ℕ)
  (h1 : nights_per_candle = 8)
  (h2 : hours_per_night = 2)
  (h3 : total_nights = 24) :
  (total_nights * hours_per_night) / (nights_per_candle * 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_candles_used_l2251_225199


namespace NUMINAMATH_CALUDE_repeating_decimal_47_l2251_225194

/-- The repeating decimal 0.474747... is equal to 47/99 -/
theorem repeating_decimal_47 : ∀ (x : ℚ), (∃ (n : ℕ), x * 10^n = ⌊x * 10^n⌋ + 0.47) → x = 47 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_47_l2251_225194


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l2251_225108

def num_white_socks : ℕ := 4
def num_brown_socks : ℕ := 4
def num_blue_socks : ℕ := 4
def num_red_socks : ℕ := 4

def total_socks : ℕ := num_white_socks + num_brown_socks + num_blue_socks + num_red_socks

theorem sock_pair_combinations :
  (num_red_socks * num_white_socks) + 
  (num_red_socks * num_brown_socks) + 
  (num_red_socks * num_blue_socks) = 48 :=
by sorry

end NUMINAMATH_CALUDE_sock_pair_combinations_l2251_225108


namespace NUMINAMATH_CALUDE_right_angle_vector_proof_l2251_225176

/-- Given two vectors OA and OB in a 2D Cartesian coordinate system, 
    where OA forms a right angle with AB, prove that the y-coordinate of OA is 5. -/
theorem right_angle_vector_proof (t : ℝ) : 
  let OA : ℝ × ℝ := (-1, t)
  let OB : ℝ × ℝ := (2, 2)
  let AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)
  (AB.1 * OB.1 + AB.2 * OB.2 = 0) → t = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_angle_vector_proof_l2251_225176


namespace NUMINAMATH_CALUDE_sum_of_coefficients_factorization_l2251_225151

theorem sum_of_coefficients_factorization (x y : ℝ) : 
  (∃ a b c d e f g h i j : ℤ, 
    27 * x^9 - 512 * y^9 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + i*x*y + j*y^2) ∧
    a + b + c + d + e + f + g + h + i + j = 32) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_factorization_l2251_225151


namespace NUMINAMATH_CALUDE_pond_length_l2251_225154

/-- The length of a rectangular pond given its width, depth, and volume. -/
theorem pond_length (width : ℝ) (depth : ℝ) (volume : ℝ) 
  (h_width : width = 15)
  (h_depth : depth = 5)
  (h_volume : volume = 1500)
  : volume / (width * depth) = 20 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l2251_225154


namespace NUMINAMATH_CALUDE_apple_picking_ratio_l2251_225135

theorem apple_picking_ratio :
  ∀ (first_hour second_hour third_hour : ℕ),
    first_hour = 66 →
    second_hour = 2 * first_hour →
    first_hour + second_hour + third_hour = 220 →
    third_hour * 3 = first_hour :=
by
  sorry

end NUMINAMATH_CALUDE_apple_picking_ratio_l2251_225135


namespace NUMINAMATH_CALUDE_certain_number_divisor_l2251_225193

theorem certain_number_divisor : ∃ n : ℕ, 
  n > 1 ∧ 
  n < 509 - 5 ∧ 
  (509 - 5) % n = 0 ∧ 
  ∀ m : ℕ, m > n → m < 509 - 5 → (509 - 5) % m ≠ 0 ∧
  ∀ k : ℕ, k < 5 → (509 - k) % n ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_divisor_l2251_225193


namespace NUMINAMATH_CALUDE_calculation_proof_l2251_225117

theorem calculation_proof :
  (1/2 + (-2/3) - 4/7 + (-1/2) - 1/3 = -11/7) ∧
  (-7^2 + 2*(-3)^2 - (-6)/((-1/3)^2) = 23) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2251_225117


namespace NUMINAMATH_CALUDE_inequality_proof_l2251_225160

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_sum : x + y + z = 1) :
  (1 / x^2 + x) * (1 / y^2 + y) * (1 / z^2 + z) ≥ (28/3)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2251_225160


namespace NUMINAMATH_CALUDE_total_time_calculation_l2251_225166

-- Define the time spent sharpening the knife
def sharpening_time : ℕ := 10

-- Define the multiplier for peeling time
def peeling_multiplier : ℕ := 3

-- Theorem to prove
theorem total_time_calculation :
  sharpening_time + peeling_multiplier * sharpening_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_time_calculation_l2251_225166


namespace NUMINAMATH_CALUDE_collinear_points_imply_a_value_l2251_225190

/-- Given three points A, B, and C in the plane, 
    this function returns true if they are collinear. -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  (C.2 - A.2) * (B.1 - A.1) = (B.2 - A.2) * (C.1 - A.1)

theorem collinear_points_imply_a_value : 
  ∀ a : ℝ, collinear (3, 2) (-2, a) (8, 12) → a = -8 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_imply_a_value_l2251_225190


namespace NUMINAMATH_CALUDE_cow_field_total_l2251_225143

/-- Represents the number of cows in a field with specific conditions -/
structure CowField where
  male : ℕ
  female : ℕ
  spotted_female : ℕ
  horned_male : ℕ

/-- The conditions of the cow field problem -/
def cow_field_conditions (field : CowField) : Prop :=
  field.female = 2 * field.male ∧
  field.spotted_female = field.female / 2 ∧
  field.horned_male = field.male / 2 ∧
  field.spotted_female = field.horned_male + 50

/-- The theorem stating that a cow field satisfying the given conditions has 300 cows in total -/
theorem cow_field_total (field : CowField) (h : cow_field_conditions field) :
  field.male + field.female = 300 :=
by sorry

end NUMINAMATH_CALUDE_cow_field_total_l2251_225143


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2251_225183

/-- An isosceles triangle with sides 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun a b c p => 
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Two sides are 9, one side is 4
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (b = c) →  -- Isosceles condition
    (a + b + c = p) →  -- Definition of perimeter
    p = 22  -- The perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : 
  ∃ (a b c p : ℝ), isosceles_triangle_perimeter a b c p :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l2251_225183


namespace NUMINAMATH_CALUDE_sqrt_six_irrational_l2251_225161

theorem sqrt_six_irrational : Irrational (Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_irrational_l2251_225161


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2251_225103

def complex_one_plus_i : ℂ := Complex.mk 1 1

theorem complex_equation_solution (a b : ℝ) : 
  let z : ℂ := complex_one_plus_i
  (z^2 + a*z + b) / (z^2 - z + 1) = 1 - Complex.I → a = -1 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2251_225103


namespace NUMINAMATH_CALUDE_sally_pokemon_cards_l2251_225132

theorem sally_pokemon_cards (initial : ℕ) (dan_gift : ℕ) (sally_bought : ℕ) : 
  initial = 27 → dan_gift = 41 → sally_bought = 20 → 
  initial + dan_gift + sally_bought = 88 := by
sorry

end NUMINAMATH_CALUDE_sally_pokemon_cards_l2251_225132


namespace NUMINAMATH_CALUDE_function_property_l2251_225150

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_property (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : ∀ x, f (x - 3/2) = f (x + 1/2))
  (h3 : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
sorry

end NUMINAMATH_CALUDE_function_property_l2251_225150


namespace NUMINAMATH_CALUDE_right_triangle_angle_measure_l2251_225159

/-- In a right triangle ABC where angle C is 90° and tan A is √3, angle A measures 60°. -/
theorem right_triangle_angle_measure (A B C : Real) (h1 : A + B + C = 180) 
  (h2 : C = 90) (h3 : Real.tan A = Real.sqrt 3) : A = 60 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_measure_l2251_225159


namespace NUMINAMATH_CALUDE_mikes_total_work_hours_l2251_225112

/-- Calculates the total hours worked given a work schedule --/
def totalHoursWorked (hours_per_day1 hours_per_day2 hours_per_day3 : ℕ) 
                     (days1 days2 days3 : ℕ) : ℕ :=
  hours_per_day1 * days1 + hours_per_day2 * days2 + hours_per_day3 * days3

/-- Proves that Mike's total work hours is 93 --/
theorem mikes_total_work_hours :
  totalHoursWorked 3 4 5 5 7 10 = 93 := by
  sorry

#eval totalHoursWorked 3 4 5 5 7 10

end NUMINAMATH_CALUDE_mikes_total_work_hours_l2251_225112


namespace NUMINAMATH_CALUDE_max_min_difference_l2251_225153

def f (x : ℝ) : ℝ := x^3 - 3*x - 1

theorem max_min_difference (M N : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f x ≤ M) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = M) ∧
  (∀ x ∈ Set.Icc (-3) 2, N ≤ f x) ∧
  (∃ x ∈ Set.Icc (-3) 2, f x = N) →
  M - N = 20 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_l2251_225153


namespace NUMINAMATH_CALUDE_range_of_a_l2251_225149

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2251_225149


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_expression_l2251_225158

theorem isosceles_right_triangle_expression (a : ℝ) (h : a > 0) :
  (a + 2 * a) / Real.sqrt (a^2 + a^2) = 3 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_expression_l2251_225158


namespace NUMINAMATH_CALUDE_perfect_square_bc_l2251_225138

theorem perfect_square_bc (a b c : ℕ) 
  (h : (a^2 / (a^2 + b^2) : ℚ) + (c^2 / (a^2 + c^2) : ℚ) = 2 * c / (b + c)) : 
  ∃ k : ℕ, b * c = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_bc_l2251_225138


namespace NUMINAMATH_CALUDE_sum_of_roots_l2251_225192

theorem sum_of_roots (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) : 
  α + β = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2251_225192


namespace NUMINAMATH_CALUDE_two_pow_2016_days_from_thursday_is_friday_l2251_225140

-- Define the days of the week
inductive Day : Type
  | monday : Day
  | tuesday : Day
  | wednesday : Day
  | thursday : Day
  | friday : Day
  | saturday : Day
  | sunday : Day

def next_day (d : Day) : Day :=
  match d with
  | Day.monday => Day.tuesday
  | Day.tuesday => Day.wednesday
  | Day.wednesday => Day.thursday
  | Day.thursday => Day.friday
  | Day.friday => Day.saturday
  | Day.saturday => Day.sunday
  | Day.sunday => Day.monday

def days_from_now (start : Day) (n : ℕ) : Day :=
  match n with
  | 0 => start
  | n + 1 => next_day (days_from_now start n)

theorem two_pow_2016_days_from_thursday_is_friday :
  days_from_now Day.thursday (2^2016) = Day.friday :=
sorry

end NUMINAMATH_CALUDE_two_pow_2016_days_from_thursday_is_friday_l2251_225140


namespace NUMINAMATH_CALUDE_logarithm_expression_equals_two_l2251_225177

theorem logarithm_expression_equals_two :
  (Real.log 243 / Real.log 3) / (Real.log 3 / Real.log 81) -
  (Real.log 729 / Real.log 3) / (Real.log 3 / Real.log 27) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equals_two_l2251_225177


namespace NUMINAMATH_CALUDE_omega_identity_l2251_225124

theorem omega_identity (ω : ℂ) (h : ω = -1/2 + Complex.I * (Real.sqrt 3) / 2) :
  1 + ω = -1/ω := by sorry

end NUMINAMATH_CALUDE_omega_identity_l2251_225124


namespace NUMINAMATH_CALUDE_vanya_more_heads_probability_vanya_more_heads_probability_is_half_l2251_225136

/-- The probability that Vanya gets more heads than Tanya when Vanya flips a coin n+1 times and Tanya flips a coin n times. -/
theorem vanya_more_heads_probability (n : ℕ) : ℝ :=
  let vanya_flips := n + 1
  let tanya_flips := n
  let prob_vanya_more_heads := (1 : ℝ) / 2
  prob_vanya_more_heads

/-- Proof that the probability of Vanya getting more heads than Tanya is 1/2. -/
theorem vanya_more_heads_probability_is_half (n : ℕ) :
  vanya_more_heads_probability n = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_vanya_more_heads_probability_vanya_more_heads_probability_is_half_l2251_225136


namespace NUMINAMATH_CALUDE_train_speed_problem_l2251_225187

theorem train_speed_problem (v : ℝ) : 
  v > 0 → -- The speed of the second train is positive
  (∃ t : ℝ, t > 0 ∧ -- There exists a positive time t
    16 * t + v * t = 444 ∧ -- Total distance traveled equals the distance between stations
    v * t = 16 * t + 60) -- The second train travels 60 km more than the first
  → v = 21 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2251_225187


namespace NUMINAMATH_CALUDE_truck_trailer_weights_l2251_225101

/-- Given the weights of trucks and trailers, prove their specific values -/
theorem truck_trailer_weights :
  ∀ (W_A W_B W_A' W_B' : ℝ),
    W_A + W_A' = 9000 →
    W_B + W_B' = 11000 →
    W_A' = 0.5 * W_A - 400 →
    W_B' = 0.4 * W_B + 500 →
    W_B = W_A + 2000 →
    W_A = 5500 ∧ W_B = 7500 ∧ W_A' = 2350 ∧ W_B' = 3500 := by
  sorry

end NUMINAMATH_CALUDE_truck_trailer_weights_l2251_225101


namespace NUMINAMATH_CALUDE_percentage_problem_l2251_225137

/-- Proves that the percentage is 50% given the problem conditions -/
theorem percentage_problem (x : ℝ) : 
  (x / 100) * 150 = 75 / 100 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2251_225137


namespace NUMINAMATH_CALUDE_platform_length_calculation_l2251_225179

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is
    approximately 350.13 meters. -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 39 →
  time_pole = 18 →
  ∃ (platform_length : ℝ), abs (platform_length - 350.13) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l2251_225179


namespace NUMINAMATH_CALUDE_curve_crosses_at_point_l2251_225145

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := 3 * t^2 + 1

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 6 * t^2 + 4

/-- The curve crosses itself if there exist two distinct real values of t that yield the same point -/
def curve_crosses_itself : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ x a = x b ∧ y a = y b

/-- The point where the curve crosses itself -/
def crossing_point : ℝ × ℝ := (109, -428)

/-- Theorem stating that the curve crosses itself at the specified point -/
theorem curve_crosses_at_point : 
  curve_crosses_itself ∧ 
  ∃ a b : ℝ, a ≠ b ∧ x a = crossing_point.1 ∧ y a = crossing_point.2 ∧
                    x b = crossing_point.1 ∧ y b = crossing_point.2 :=
sorry

end NUMINAMATH_CALUDE_curve_crosses_at_point_l2251_225145


namespace NUMINAMATH_CALUDE_reverse_divisibility_implies_divides_99_l2251_225174

/-- Given a natural number, return the number formed by reversing its digits -/
def reverse_digits (n : ℕ) : ℕ := sorry

/-- A natural number k has the property that if it divides n, it also divides the reverse of n -/
def has_reverse_divisibility_property (k : ℕ) : Prop :=
  ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n

theorem reverse_divisibility_implies_divides_99 (k : ℕ) :
  has_reverse_divisibility_property k → 99 ∣ k := by sorry

end NUMINAMATH_CALUDE_reverse_divisibility_implies_divides_99_l2251_225174


namespace NUMINAMATH_CALUDE_triangle_inequalities_l2251_225120

-- Define a triangle with heights and an internal point
structure Triangle :=
  (h₁ h₂ h₃ u v w : ℝ)
  (h₁_pos : h₁ > 0)
  (h₂_pos : h₂ > 0)
  (h₃_pos : h₃ > 0)
  (u_pos : u > 0)
  (v_pos : v > 0)
  (w_pos : w > 0)

-- Theorem statement
theorem triangle_inequalities (t : Triangle) :
  (t.h₁ / t.u + t.h₂ / t.v + t.h₃ / t.w ≥ 9) ∧
  (t.h₁ * t.h₂ * t.h₃ ≥ 27 * t.u * t.v * t.w) ∧
  ((t.h₁ - t.u) * (t.h₂ - t.v) * (t.h₃ - t.w) ≥ 8 * t.u * t.v * t.w) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l2251_225120


namespace NUMINAMATH_CALUDE_smallest_b_value_l2251_225118

theorem smallest_b_value (a b : ℝ) (h1 : 2 < a) (h2 : a < b) (h3 : a + b = 7)
  (h4 : ¬ (2 + a > b ∧ 2 + b > a ∧ a + b > 2)) : 
  (∀ b' : ℝ, (∃ a' : ℝ, 2 < a' ∧ a' < b' ∧ a' + b' = 7 ∧
    ¬ (2 + a' > b' ∧ 2 + b' > a' ∧ a' + b' > 2)) → b' ≥ 9/2) ∧ b = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l2251_225118


namespace NUMINAMATH_CALUDE_vacation_cost_l2251_225164

/-- If a total cost C divided among 3 people is $40 more per person than if divided among 4 people, then C equals $480. -/
theorem vacation_cost (C : ℚ) : C / 3 - C / 4 = 40 → C = 480 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l2251_225164


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2251_225169

/-- The polynomial division theorem for z^2023 - 1 divided by z^2 + z + 1 -/
theorem polynomial_division_remainder (z : ℂ) : ∃ (Q R : ℂ → ℂ),
  z^2023 - 1 = (z^2 + z + 1) * Q z + R z ∧ 
  (∀ x, R x = -x - 1) ∧
  (∃ a b, ∀ x, R x = a * x + b) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2251_225169


namespace NUMINAMATH_CALUDE_rectangle_least_area_l2251_225142

theorem rectangle_least_area (l w : ℕ) (h1 : l > 0) (h2 : w > 0) (h3 : 2 * (l + w) = 100) : 
  l * w ≥ 49 := by
sorry

end NUMINAMATH_CALUDE_rectangle_least_area_l2251_225142


namespace NUMINAMATH_CALUDE_average_marks_bcd_e_is_48_l2251_225104

def average_marks_bcd_e (a b c d e : ℕ) : Prop :=
  -- The average marks of a, b, c is 48
  (a + b + c) / 3 = 48 ∧
  -- When d joins, the average becomes 47
  (a + b + c + d) / 4 = 47 ∧
  -- E has 3 more marks than d
  e = d + 3 ∧
  -- The marks of a is 43
  a = 43 →
  -- The average marks of b, c, d, e is 48
  (b + c + d + e) / 4 = 48

theorem average_marks_bcd_e_is_48 : 
  ∀ (a b c d e : ℕ), average_marks_bcd_e a b c d e :=
sorry

end NUMINAMATH_CALUDE_average_marks_bcd_e_is_48_l2251_225104


namespace NUMINAMATH_CALUDE_x_plus_y_equals_three_l2251_225110

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

variable (a b c p : V)
variable (x y : ℝ)

-- {a, b, c} is a basis of the space
variable (h1 : LinearIndependent ℝ ![a, b, c])
variable (h2 : Submodule.span ℝ {a, b, c} = ⊤)

-- p = 3a + b + c
variable (h3 : p = 3 • a + b + c)

-- {a+b, a-b, c} is another basis of the space
variable (h4 : LinearIndependent ℝ ![a + b, a - b, c])
variable (h5 : Submodule.span ℝ {a + b, a - b, c} = ⊤)

-- p = x(a+b) + y(a-b) + c
variable (h6 : p = x • (a + b) + y • (a - b) + c)

theorem x_plus_y_equals_three : x + y = 3 := by sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_three_l2251_225110


namespace NUMINAMATH_CALUDE_disjoint_sets_cardinality_relation_l2251_225130

theorem disjoint_sets_cardinality_relation 
  (a b : ℕ+) 
  (A B : Finset ℤ) 
  (h_disjoint : Disjoint A B)
  (h_membership : ∀ i ∈ A ∪ B, (i + a) ∈ A ∨ (i - b) ∈ B) :
  a * A.card = b * B.card := by
  sorry

end NUMINAMATH_CALUDE_disjoint_sets_cardinality_relation_l2251_225130


namespace NUMINAMATH_CALUDE_complex_number_problem_l2251_225170

theorem complex_number_problem (m : ℝ) (z : ℂ) :
  let z₁ : ℂ := m * (m - 1) + (m - 1) * Complex.I
  (z₁.re = 0 ∧ z₁.im ≠ 0) →
  (3 + z₁) * z = 4 + 2 * Complex.I →
  m = 0 ∧ z = 1 + Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2251_225170


namespace NUMINAMATH_CALUDE_bob_weighs_165_l2251_225131

/-- Bob's weight given the conditions -/
def bobs_weight (jim_weight bob_weight : ℕ) : Prop :=
  (jim_weight + bob_weight = 220) ∧ 
  (bob_weight - jim_weight = 2 * jim_weight) ∧
  (bob_weight = 165)

/-- Theorem stating that Bob's weight is 165 pounds given the conditions -/
theorem bob_weighs_165 :
  ∃ (jim_weight bob_weight : ℕ), bobs_weight jim_weight bob_weight :=
by
  sorry

end NUMINAMATH_CALUDE_bob_weighs_165_l2251_225131


namespace NUMINAMATH_CALUDE_kilmer_park_tree_height_l2251_225156

/-- Calculates the height of a tree in inches after a given number of years -/
def tree_height_in_inches (initial_height : ℕ) (annual_growth : ℕ) (years : ℕ) : ℕ :=
  (initial_height + annual_growth * years) * 12

/-- Theorem: The height of the tree in Kilmer Park after 8 years is 1104 inches -/
theorem kilmer_park_tree_height : tree_height_in_inches 52 5 8 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_kilmer_park_tree_height_l2251_225156


namespace NUMINAMATH_CALUDE_vegan_soy_free_dishes_l2251_225196

theorem vegan_soy_free_dishes 
  (total_dishes : ℕ) 
  (vegan_ratio : ℚ) 
  (soy_ratio : ℚ) 
  (h1 : vegan_ratio = 1 / 3) 
  (h2 : soy_ratio = 5 / 6) : 
  ↑total_dishes * vegan_ratio * (1 - soy_ratio) = ↑total_dishes * (1 / 18) := by
sorry

end NUMINAMATH_CALUDE_vegan_soy_free_dishes_l2251_225196


namespace NUMINAMATH_CALUDE_expression_simplification_l2251_225127

theorem expression_simplification (x : ℝ) (h : x^2 + x - 6 = 0) :
  (x - 1) / ((2 / (x - 1)) - 1) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2251_225127


namespace NUMINAMATH_CALUDE_unique_sequence_l2251_225141

theorem unique_sequence (n : ℕ) (h : n > 1) :
  ∃! (x : ℕ → ℕ), 
    (∀ k, k ∈ Finset.range (n - 1) → x k > 0) ∧ 
    (∀ i j, i < j ∧ j < n - 1 → x i < x j) ∧
    (∀ i, i ∈ Finset.range (n - 1) → x i + x (n - 1 - i) = 2 * n) ∧
    (∀ i j, i ∈ Finset.range (n - 1) ∧ j ∈ Finset.range (n - 1) ∧ x i + x j < 2 * n → 
      ∃ k, k ∈ Finset.range (n - 1) ∧ x i + x j = x k) ∧
    (∀ k, k ∈ Finset.range (n - 1) → x k = 2 * (k + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_l2251_225141


namespace NUMINAMATH_CALUDE_square_vector_problem_l2251_225147

theorem square_vector_problem (a b c : ℝ × ℝ) : 
  (∀ x : ℝ × ℝ, ‖x‖ = 1 → ‖x + x‖ = ‖a‖) →  -- side length is 1
  ‖a‖ = 1 →                                -- |a| = 1 (side length)
  ‖c‖ = Real.sqrt 2 →                      -- |c| = √2 (diagonal)
  a + b = c →                              -- vector addition
  ‖b - a - c‖ = 2 := by sorry

end NUMINAMATH_CALUDE_square_vector_problem_l2251_225147


namespace NUMINAMATH_CALUDE_dividend_calculation_l2251_225178

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 14)
  (h2 : quotient = 9)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 131 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2251_225178


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2251_225185

open Real

-- Define the property p
def property_p (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + π) = -f x

-- Define the property q
def property_q (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 2*π) = f x

-- Theorem: p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ f : ℝ → ℝ, property_p f → property_q f) ∧
  (∃ f : ℝ → ℝ, property_q f ∧ ¬property_p f) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2251_225185


namespace NUMINAMATH_CALUDE_gym_guests_first_hour_l2251_225102

/-- The number of guests who entered the gym in the first hour -/
def first_hour_guests : ℕ := 50

/-- The total number of towels available -/
def total_towels : ℕ := 300

/-- The number of hours the gym is open -/
def open_hours : ℕ := 4

/-- The increase rate for the second hour -/
def second_hour_rate : ℚ := 1.2

/-- The increase rate for the third hour -/
def third_hour_rate : ℚ := 1.25

/-- The increase rate for the fourth hour -/
def fourth_hour_rate : ℚ := 4/3

/-- The number of towels that need to be washed at the end of the day -/
def towels_to_wash : ℕ := 285

theorem gym_guests_first_hour :
  first_hour_guests * (1 + second_hour_rate + second_hour_rate * third_hour_rate +
    second_hour_rate * third_hour_rate * fourth_hour_rate) = towels_to_wash :=
sorry

end NUMINAMATH_CALUDE_gym_guests_first_hour_l2251_225102


namespace NUMINAMATH_CALUDE_largest_four_digit_number_with_conditions_l2251_225121

theorem largest_four_digit_number_with_conditions : 
  ∀ n : ℕ, 
  n ≤ 9999 ∧ n ≥ 1000 ∧ 
  ∃ k : ℕ, n = 11 * k + 2 ∧
  ∃ m : ℕ, n = 7 * m + 4 
  → n ≤ 9979 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_number_with_conditions_l2251_225121


namespace NUMINAMATH_CALUDE_student_calculation_difference_l2251_225189

theorem student_calculation_difference : 
  let number : ℝ := 80.00000000000003
  let correct_answer := number * (4/5 : ℝ)
  let student_answer := number / (4/5 : ℝ)
  student_answer - correct_answer = 36.0000000000000175 := by
sorry

end NUMINAMATH_CALUDE_student_calculation_difference_l2251_225189


namespace NUMINAMATH_CALUDE_solution_to_linear_equation_l2251_225180

theorem solution_to_linear_equation :
  let x : ℝ := 4
  let y : ℝ := 2
  2 * x - y = 6 :=
by sorry

end NUMINAMATH_CALUDE_solution_to_linear_equation_l2251_225180


namespace NUMINAMATH_CALUDE_pencils_to_library_l2251_225106

theorem pencils_to_library (total_pencils : Nat) (num_classrooms : Nat) 
    (h1 : total_pencils = 935) 
    (h2 : num_classrooms = 9) : 
  total_pencils % num_classrooms = 8 := by
  sorry

end NUMINAMATH_CALUDE_pencils_to_library_l2251_225106


namespace NUMINAMATH_CALUDE_valentines_calculation_l2251_225186

/-- The number of Valentines Mrs. Franklin initially had -/
def initial_valentines : ℕ := 58

/-- The number of Valentines Mrs. Franklin gave to her students -/
def given_valentines : ℕ := 42

/-- The number of Valentines Mrs. Franklin has now -/
def remaining_valentines : ℕ := initial_valentines - given_valentines

theorem valentines_calculation : remaining_valentines = 16 := by
  sorry

end NUMINAMATH_CALUDE_valentines_calculation_l2251_225186


namespace NUMINAMATH_CALUDE_quadratic_fraction_value_l2251_225157

theorem quadratic_fraction_value (x : ℝ) (h : x^2 - 3*x + 1 = 0) : 
  x^2 / (x^4 + x^2 + 1) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_fraction_value_l2251_225157


namespace NUMINAMATH_CALUDE_polynomial_characterization_l2251_225195

-- Define a real polynomial
def RealPolynomial := ℝ → ℝ

-- Define the condition for a, b, c
def SatisfiesCondition (a b c : ℝ) : Prop := a * b + b * c + c * a = 0

-- Define the property that P must satisfy
def SatisfiesProperty (P : RealPolynomial) : Prop :=
  ∀ a b c : ℝ, SatisfiesCondition a b c →
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)

-- Define the form of the polynomial we want to prove
def IsQuarticQuadratic (P : RealPolynomial) : Prop :=
  ∃ α β : ℝ, ∀ x : ℝ, P x = α * x^4 + β * x^2

-- The main theorem
theorem polynomial_characterization (P : RealPolynomial) :
  SatisfiesProperty P → IsQuarticQuadratic P :=
sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l2251_225195


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2251_225163

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 4 + a 6 = 3 →
  a 1 + a 3 + a 5 + a 7 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2251_225163


namespace NUMINAMATH_CALUDE_calculator_probability_l2251_225155

/-- The probability of a specific number M appearing on the display
    when starting from a number N, where M < N -/
def prob_appear (N M : ℕ) : ℚ :=
  if M < N then 1 / (M + 1 : ℚ) else 0

/-- The probability of all numbers in a list appearing on the display
    when starting from a given number -/
def prob_all_appear (start : ℕ) (numbers : List ℕ) : ℚ :=
  numbers.foldl (fun acc n => acc * prob_appear start n) 1

theorem calculator_probability :
  prob_all_appear 2003 [1000, 100, 10, 1] = 1 / 2224222 := by
  sorry

end NUMINAMATH_CALUDE_calculator_probability_l2251_225155


namespace NUMINAMATH_CALUDE_unique_prime_sum_and_difference_l2251_225165

theorem unique_prime_sum_and_difference : 
  ∃! p : ℕ, 
    Nat.Prime p ∧ 
    (∃ q r : ℕ, Nat.Prime q ∧ Nat.Prime r ∧ p = q + r) ∧
    (∃ s t : ℕ, Nat.Prime s ∧ Nat.Prime t ∧ p = s - t) ∧
    p = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_sum_and_difference_l2251_225165


namespace NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_prob_at_least_one_multiple_of_four_l2251_225181

/-- The probability of selecting at least one multiple of 4 when randomly choosing 3 integers from 1 to 50 (inclusive) -/
theorem probability_at_least_one_multiple_of_four : ℚ :=
  28051 / 50000

/-- The set of integers from 1 to 50 -/
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 50}

/-- The number of elements in set S -/
def S_size : ℕ := 50

/-- The set of multiples of 4 in S -/
def M : Set ℕ := {n ∈ S | n % 4 = 0}

/-- The number of elements in set M -/
def M_size : ℕ := 12

/-- The probability of selecting a number that is not a multiple of 4 -/
def p_not_multiple_of_four : ℚ := (S_size - M_size) / S_size

/-- The probability of selecting three numbers, none of which are multiples of 4 -/
def p_none_multiple_of_four : ℚ := p_not_multiple_of_four ^ 3

/-- Theorem: The probability of selecting at least one multiple of 4 when randomly choosing 3 integers from 1 to 50 (inclusive) is 28051/50000 -/
theorem prob_at_least_one_multiple_of_four :
  1 - p_none_multiple_of_four = probability_at_least_one_multiple_of_four :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_prob_at_least_one_multiple_of_four_l2251_225181


namespace NUMINAMATH_CALUDE_cups_per_girl_l2251_225109

theorem cups_per_girl (total_students : ℕ) (boys : ℕ) (cups_per_boy : ℕ) (total_cups : ℕ) :
  total_students = 30 →
  boys = 10 →
  cups_per_boy = 5 →
  total_cups = 90 →
  2 * boys = total_students - boys →
  (total_students - boys) * (total_cups - boys * cups_per_boy) / (total_students - boys) = 2 :=
by sorry

end NUMINAMATH_CALUDE_cups_per_girl_l2251_225109


namespace NUMINAMATH_CALUDE_cafeteria_pies_l2251_225111

theorem cafeteria_pies (initial_apples handed_out apples_per_pie : ℕ) 
  (h1 : initial_apples = 86)
  (h2 : handed_out = 30)
  (h3 : apples_per_pie = 8) :
  (initial_apples - handed_out) / apples_per_pie = 7 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l2251_225111


namespace NUMINAMATH_CALUDE_smallest_x_congruence_l2251_225172

theorem smallest_x_congruence :
  ∃ (x : ℕ), x > 0 ∧ (725 * x) % 35 = (1165 * x) % 35 ∧
  ∀ (y : ℕ), y > 0 → (725 * y) % 35 = (1165 * y) % 35 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_congruence_l2251_225172


namespace NUMINAMATH_CALUDE_grandmother_truth_lies_consistent_solution_l2251_225184

-- Define the type for grandmothers
inductive Grandmother
| Emilia
| Leonie
| Gabrielle

-- Define a function to represent the number of grandchildren for each grandmother
def grandchildren : Grandmother → ℕ
| Grandmother.Emilia => 8
| Grandmother.Leonie => 7
| Grandmother.Gabrielle => 10

-- Define a function to represent the statements made by each grandmother
def statements : Grandmother → List (Grandmother → Bool)
| Grandmother.Emilia => [
    fun g => grandchildren g = 7,
    fun g => grandchildren g = 8,
    fun g => grandchildren Grandmother.Gabrielle = 10
  ]
| Grandmother.Leonie => [
    fun g => grandchildren Grandmother.Emilia = 8,
    fun g => grandchildren g = 6,
    fun g => grandchildren g = 7
  ]
| Grandmother.Gabrielle => [
    fun g => grandchildren Grandmother.Emilia = 7,
    fun g => grandchildren g = 9,
    fun g => grandchildren g = 10
  ]

-- Define a function to count true statements for each grandmother
def countTrueStatements (g : Grandmother) : ℕ :=
  (statements g).filter (fun s => s g) |>.length

-- Theorem stating that each grandmother tells the truth twice and lies once
theorem grandmother_truth_lies :
  ∀ g : Grandmother, countTrueStatements g = 2 :=
sorry

-- Main theorem proving the consistency of the solution
theorem consistent_solution :
  (grandchildren Grandmother.Emilia = 8) ∧
  (grandchildren Grandmother.Leonie = 7) ∧
  (grandchildren Grandmother.Gabrielle = 10) :=
sorry

end NUMINAMATH_CALUDE_grandmother_truth_lies_consistent_solution_l2251_225184


namespace NUMINAMATH_CALUDE_max_two_greater_than_half_l2251_225139

theorem max_two_greater_than_half (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_distinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  let values := [Real.sin α * Real.cos β, Real.sin β * Real.cos γ, Real.sin γ * Real.cos α]
  (values.filter (λ x => x > 1/2)).length ≤ 2 := by
sorry

end NUMINAMATH_CALUDE_max_two_greater_than_half_l2251_225139


namespace NUMINAMATH_CALUDE_not_adjacent_in_sorted_consecutive_l2251_225171

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def sorted_by_digit_sum (a b : ℕ) : Prop :=
  (sum_of_digits a < sum_of_digits b) ∨ 
  (sum_of_digits a = sum_of_digits b ∧ a ≤ b)

theorem not_adjacent_in_sorted_consecutive (start : ℕ) : 
  ¬ ∃ i : ℕ, i < 99 ∧ 
    (sorted_by_digit_sum (start + i) 2010 ∧ sorted_by_digit_sum 2010 2011 ∧ sorted_by_digit_sum 2011 (start + (i + 1))) ∨
    (sorted_by_digit_sum (start + i) 2011 ∧ sorted_by_digit_sum 2011 2010 ∧ sorted_by_digit_sum 2010 (start + (i + 1))) :=
sorry

end NUMINAMATH_CALUDE_not_adjacent_in_sorted_consecutive_l2251_225171


namespace NUMINAMATH_CALUDE_line_parameterization_l2251_225182

/-- Given a line y = (3/2)x - 25 parameterized by (x,y) = (f(t), 15t - 7),
    prove that f(t) = 10t + 12 is the correct parameterization for x. -/
theorem line_parameterization (f : ℝ → ℝ) :
  (∀ t : ℝ, (3/2) * f t - 25 = 15 * t - 7) →
  f = λ t => 10 * t + 12 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l2251_225182


namespace NUMINAMATH_CALUDE_charcoal_drawings_count_l2251_225197

theorem charcoal_drawings_count (total : ℕ) (colored_pencil : ℕ) (blending_marker : ℕ) 
  (h1 : total = 25)
  (h2 : colored_pencil = 14)
  (h3 : blending_marker = 7) :
  total - (colored_pencil + blending_marker) = 4 := by
  sorry

end NUMINAMATH_CALUDE_charcoal_drawings_count_l2251_225197


namespace NUMINAMATH_CALUDE_band_repertoire_average_l2251_225173

theorem band_repertoire_average (total_songs : ℕ) (first_set : ℕ) (second_set : ℕ) (encore : ℕ) : 
  total_songs = 30 →
  first_set = 5 →
  second_set = 7 →
  encore = 2 →
  (total_songs - (first_set + second_set + encore)) / 2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_band_repertoire_average_l2251_225173


namespace NUMINAMATH_CALUDE_franzia_carlo_rosi_ratio_l2251_225133

/-- Represents the ages of three wine brands and their relationships -/
structure WineAges where
  franzia : ℕ
  carlo_rosi : ℕ
  twin_valley : ℕ
  carlo_rosi_multiple : ℕ
  carlo_rosi_is_40 : carlo_rosi = 40
  carlo_rosi_four_times_twin_valley : carlo_rosi = 4 * twin_valley
  total_age_170 : franzia + carlo_rosi + twin_valley = 170

/-- Theorem stating the ratio of Franzia's age to Carlo Rosi's age is 3:1 -/
theorem franzia_carlo_rosi_ratio (w : WineAges) : w.franzia / w.carlo_rosi = 3 := by
  sorry

end NUMINAMATH_CALUDE_franzia_carlo_rosi_ratio_l2251_225133


namespace NUMINAMATH_CALUDE_gcd_98_75_l2251_225148

theorem gcd_98_75 : Nat.gcd 98 75 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_75_l2251_225148


namespace NUMINAMATH_CALUDE_two_lines_at_45_degrees_l2251_225162

/-- The equation represents two lines that intersect at a 45° angle when k = 80 -/
theorem two_lines_at_45_degrees (x y : ℝ) :
  let k : ℝ := 80
  let equation := x^2 + x*y - 6*y^2 - 20*x - 20*y + k
  ∃ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, equation = 0 ↔ (l₁ x y ∨ l₂ x y)) ∧
    (∃ x₀ y₀, l₁ x₀ y₀ ∧ l₂ x₀ y₀) ∧
    (∀ x₀ y₀, l₁ x₀ y₀ ∧ l₂ x₀ y₀ → 
      ∃ (v₁ v₂ : ℝ × ℝ),
        (v₁.1 ≠ 0 ∨ v₁.2 ≠ 0) ∧
        (v₂.1 ≠ 0 ∨ v₂.2 ≠ 0) ∧
        (v₁.1 * v₂.1 + v₁.2 * v₂.2) / (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2)) = Real.cos (π/4)) :=
by sorry


end NUMINAMATH_CALUDE_two_lines_at_45_degrees_l2251_225162


namespace NUMINAMATH_CALUDE_cubic_factorization_l2251_225167

theorem cubic_factorization (a : ℝ) : a^3 + 2*a^2 + a = a*(a+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2251_225167


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2251_225113

theorem container_volume_ratio 
  (container1 container2 container3 : ℝ) 
  (h1 : 3/5 * container1 = 2/3 * container2) 
  (h2 : 2/3 * container2 - 1/2 * container3 = 1/2 * container3) 
  (h3 : container1 > 0) 
  (h4 : container2 > 0) 
  (h5 : container3 > 0) : 
  container2 / container3 = 2/3 := by
sorry


end NUMINAMATH_CALUDE_container_volume_ratio_l2251_225113


namespace NUMINAMATH_CALUDE_stockholm_to_malmo_via_gothenburg_l2251_225146

/-- Represents a distance on a map --/
structure MapDistance :=
  (cm : ℝ)

/-- Represents a real-world distance --/
structure RealDistance :=
  (km : ℝ)

/-- Represents a map scale --/
structure MapScale :=
  (km_per_cm : ℝ)

/-- Converts a map distance to a real distance given a scale --/
def convert_distance (md : MapDistance) (scale : MapScale) : RealDistance :=
  ⟨md.cm * scale.km_per_cm⟩

/-- Adds two real distances --/
def add_distances (d1 d2 : RealDistance) : RealDistance :=
  ⟨d1.km + d2.km⟩

theorem stockholm_to_malmo_via_gothenburg 
  (stockholm_gothenburg : MapDistance)
  (gothenburg_malmo : MapDistance)
  (scale : MapScale)
  (h1 : stockholm_gothenburg.cm = 120)
  (h2 : gothenburg_malmo.cm = 150)
  (h3 : scale.km_per_cm = 20) :
  (add_distances 
    (convert_distance stockholm_gothenburg scale)
    (convert_distance gothenburg_malmo scale)).km = 5400 :=
by
  sorry

end NUMINAMATH_CALUDE_stockholm_to_malmo_via_gothenburg_l2251_225146


namespace NUMINAMATH_CALUDE_opposite_of_negative_sqrt_3_squared_l2251_225125

theorem opposite_of_negative_sqrt_3_squared : -((-Real.sqrt 3)^2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_sqrt_3_squared_l2251_225125


namespace NUMINAMATH_CALUDE_abs_z_equals_2_sqrt_2_l2251_225105

def z : ℂ := Complex.I - 2 * Complex.I^2 + 3 * Complex.I^3

theorem abs_z_equals_2_sqrt_2 : Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_2_sqrt_2_l2251_225105


namespace NUMINAMATH_CALUDE_symmetric_polynomial_property_l2251_225115

theorem symmetric_polynomial_property (p q r : ℝ) :
  let f := λ x : ℝ => p * x^7 + q * x^3 + r * x - 5
  f (-6) = 3 → f 6 = -13 := by
sorry

end NUMINAMATH_CALUDE_symmetric_polynomial_property_l2251_225115


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l2251_225188

/-- The number of digits in a and b -/
def n : ℕ := 1984

/-- The integer a consisting of n nines in base 10 -/
def a : ℕ := (10^n - 1) / 9

/-- The integer b consisting of n fives in base 10 -/
def b : ℕ := (5 * (10^n - 1)) / 9

/-- Function to calculate the sum of digits of a natural number in base 10 -/
def sumOfDigits (k : ℕ) : ℕ :=
  if k < 10 then k else k % 10 + sumOfDigits (k / 10)

/-- Theorem stating that the sum of digits of 9ab is 27779 -/
theorem sum_of_digits_9ab : sumOfDigits (9 * a * b) = 27779 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l2251_225188


namespace NUMINAMATH_CALUDE_unique_digit_multiplication_l2251_225114

theorem unique_digit_multiplication (B : ℕ) : 
  (B < 10) →                           -- B is a single digit
  (B2 : ℕ) →                           -- B2 is a natural number
  (B2 = 10 * B + 2) →                  -- B2 is a two-digit number ending in 2
  (7 * B < 100) →                      -- 7B is a two-digit number
  (B2 * (70 + B) = 6396) →             -- The multiplication equation
  (B = 8) := by sorry

end NUMINAMATH_CALUDE_unique_digit_multiplication_l2251_225114


namespace NUMINAMATH_CALUDE_smaller_number_problem_l2251_225116

theorem smaller_number_problem (x y : ℝ) 
  (eq1 : 3 * x - y = 20) 
  (eq2 : x + y = 48) : 
  min x y = 17 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l2251_225116


namespace NUMINAMATH_CALUDE_departmental_store_average_salary_l2251_225168

def average_salary (num_managers : ℕ) (num_associates : ℕ) (avg_salary_managers : ℚ) (avg_salary_associates : ℚ) : ℚ :=
  let total_salary := num_managers * avg_salary_managers + num_associates * avg_salary_associates
  let total_employees := num_managers + num_associates
  total_salary / total_employees

theorem departmental_store_average_salary :
  average_salary 9 18 1300 12000 = 8433.33 := by
  sorry

end NUMINAMATH_CALUDE_departmental_store_average_salary_l2251_225168


namespace NUMINAMATH_CALUDE_cd_product_value_l2251_225107

/-- An equilateral triangle with vertices at (0,0), (c,17), and (d,43) -/
structure EquilateralTriangle where
  c : ℝ
  d : ℝ

/-- The product of c and d in the equilateral triangle -/
def cd_product (triangle : EquilateralTriangle) : ℝ := triangle.c * triangle.d

/-- Theorem stating that the product cd equals -1689/24 for the given equilateral triangle -/
theorem cd_product_value (triangle : EquilateralTriangle) :
  cd_product triangle = -1689 / 24 := by sorry

end NUMINAMATH_CALUDE_cd_product_value_l2251_225107
