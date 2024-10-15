import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_l252_25297

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 14)
  (eq2 : b^2 + 5*c = -13)
  (eq3 : c^2 + 7*a = -26) :
  a^2 + b^2 + c^2 = 20.75 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l252_25297


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l252_25282

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l252_25282


namespace NUMINAMATH_CALUDE_consecutive_lucky_tickets_exist_l252_25216

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is lucky if the sum of its digits is divisible by 7 -/
def is_lucky (n : ℕ) : Prop := sum_of_digits n % 7 = 0

/-- There exist two consecutive lucky bus ticket numbers -/
theorem consecutive_lucky_tickets_exist : ∃ n : ℕ, is_lucky n ∧ is_lucky (n + 1) := by sorry

end NUMINAMATH_CALUDE_consecutive_lucky_tickets_exist_l252_25216


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l252_25270

theorem cubic_sum_over_product (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) (sum_squares : a^2 + b^2 + c^2 = 3) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l252_25270


namespace NUMINAMATH_CALUDE_number_of_stoplights_l252_25283

-- Define the number of stoplights
variable (n : ℕ)

-- Define the time for the first route with all green lights
def green_time : ℕ := 10

-- Define the additional time for each red light
def red_light_delay : ℕ := 3

-- Define the time for the second route
def second_route_time : ℕ := 14

-- Define the additional time when all lights are red compared to the second route
def all_red_additional_time : ℕ := 5

-- Theorem statement
theorem number_of_stoplights :
  (green_time + n * red_light_delay = second_route_time + all_red_additional_time) →
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_stoplights_l252_25283


namespace NUMINAMATH_CALUDE_angle_cosine_equivalence_l252_25252

theorem angle_cosine_equivalence (A B : Real) (hA : 0 < A ∧ A < Real.pi) (hB : 0 < B ∧ B < Real.pi) :
  A > B ↔ Real.cos A < Real.cos B := by
  sorry

end NUMINAMATH_CALUDE_angle_cosine_equivalence_l252_25252


namespace NUMINAMATH_CALUDE_adas_original_seat_was_two_l252_25293

/-- Represents the seats in the theater --/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends sitting in the theater --/
inductive Friend
| Ada
| Bea
| Ceci
| Dee
| Edie
| Faye

/-- Represents the state of the seating arrangement --/
structure SeatingArrangement where
  seats : Fin 6 → Option Friend

/-- Defines a valid initial seating arrangement --/
def validInitialArrangement (arr : SeatingArrangement) : Prop :=
  ∃ (emptySlot : Fin 6), 
    (∀ i : Fin 6, i ≠ emptySlot → arr.seats i ≠ none) ∧
    (arr.seats emptySlot = none) ∧
    (∃ (ada bea ceci dee edie faye : Fin 6), 
      ada ≠ bea ∧ ada ≠ ceci ∧ ada ≠ dee ∧ ada ≠ edie ∧ ada ≠ faye ∧
      bea ≠ ceci ∧ bea ≠ dee ∧ bea ≠ edie ∧ bea ≠ faye ∧
      ceci ≠ dee ∧ ceci ≠ edie ∧ ceci ≠ faye ∧
      dee ≠ edie ∧ dee ≠ faye ∧
      edie ≠ faye ∧
      arr.seats ada = some Friend.Ada ∧
      arr.seats bea = some Friend.Bea ∧
      arr.seats ceci = some Friend.Ceci ∧
      arr.seats dee = some Friend.Dee ∧
      arr.seats edie = some Friend.Edie ∧
      arr.seats faye = some Friend.Faye)

/-- Defines the final seating arrangement after movements --/
def finalArrangement (initial : SeatingArrangement) (final : SeatingArrangement) : Prop :=
  ∃ (bea bea' ceci ceci' dee dee' edie edie' : Fin 6),
    initial.seats bea = some Friend.Bea ∧
    initial.seats ceci = some Friend.Ceci ∧
    initial.seats dee = some Friend.Dee ∧
    initial.seats edie = some Friend.Edie ∧
    bea' = (bea + 3) % 6 ∧
    ceci' = (ceci + 2) % 6 ∧
    dee' ≠ dee ∧ edie' ≠ edie ∧
    final.seats bea' = some Friend.Bea ∧
    final.seats ceci' = some Friend.Ceci ∧
    final.seats dee' = some Friend.Dee ∧
    final.seats edie' = some Friend.Edie ∧
    (final.seats 0 = none ∨ final.seats 5 = none)

/-- Theorem: Ada's original seat was Seat 2 --/
theorem adas_original_seat_was_two 
  (initial final : SeatingArrangement)
  (h_initial : validInitialArrangement initial)
  (h_final : finalArrangement initial final) :
  ∃ (ada : Fin 6), initial.seats ada = some Friend.Ada ∧ ada = 1 := by
  sorry

end NUMINAMATH_CALUDE_adas_original_seat_was_two_l252_25293


namespace NUMINAMATH_CALUDE_vacant_seats_l252_25235

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 60/100) : 
  (1 - filled_percentage) * total_seats = 240 := by
sorry

end NUMINAMATH_CALUDE_vacant_seats_l252_25235


namespace NUMINAMATH_CALUDE_book_gain_percent_l252_25285

theorem book_gain_percent (MP : ℝ) (CP : ℝ) (SP : ℝ) : 
  CP = 0.64 * MP →
  SP = 0.84 * MP →
  (SP - CP) / CP * 100 = 31.25 :=
by sorry

end NUMINAMATH_CALUDE_book_gain_percent_l252_25285


namespace NUMINAMATH_CALUDE_ellipse_m_range_l252_25299

/-- Represents an ellipse with equation x^2/m^2 + y^2/(2+m) = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ x y : ℝ, x^2/m^2 + y^2/(2+m) = 1

/-- Condition for foci on x-axis -/
def foci_on_x_axis (m : ℝ) : Prop := m^2 > 2 + m

/-- The range of m for which the ellipse is valid and has foci on x-axis -/
def valid_m_range (m : ℝ) : Prop := (m > 2 ∨ (-2 < m ∧ m < -1))

theorem ellipse_m_range (m : ℝ) (e : Ellipse m) :
  foci_on_x_axis m → valid_m_range m :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l252_25299


namespace NUMINAMATH_CALUDE_share_ratio_l252_25287

/-- Proves that the ratio of B's share to C's share is 3:2 given the problem conditions -/
theorem share_ratio (total amount : ℕ) (a_share b_share c_share : ℕ) :
  amount = 544 →
  a_share = 384 →
  b_share = 96 →
  c_share = 64 →
  amount = a_share + b_share + c_share →
  a_share = (2 : ℚ) / 3 * b_share →
  b_share / c_share = (3 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l252_25287


namespace NUMINAMATH_CALUDE_gravitational_force_on_space_station_l252_25273

/-- Gravitational force calculation -/
theorem gravitational_force_on_space_station 
  (inverse_square_law : ∀ (d : ℝ) (f : ℝ), f * d^2 = (400 : ℝ) * 6000^2)
  (earth_surface_distance : ℝ := 6000)
  (earth_surface_force : ℝ := 400)
  (space_station_distance : ℝ := 360000) :
  (earth_surface_force * earth_surface_distance^2) / space_station_distance^2 = 1/9 := by
sorry

end NUMINAMATH_CALUDE_gravitational_force_on_space_station_l252_25273


namespace NUMINAMATH_CALUDE_solution_set_f_less_g_plus_a_range_of_a_for_f_plus_g_greater_a_squared_l252_25231

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |x - 2| + a
def g (x : ℝ) : ℝ := |x + 4|

-- Theorem for part I
theorem solution_set_f_less_g_plus_a (a : ℝ) :
  {x : ℝ | f x a < g x + a} = {x : ℝ | x > -1} :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_plus_g_greater_a_squared :
  {a : ℝ | ∀ x, f x a + g x > a^2} = {a : ℝ | -2 < a ∧ a < 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_g_plus_a_range_of_a_for_f_plus_g_greater_a_squared_l252_25231


namespace NUMINAMATH_CALUDE_six_digit_square_numbers_l252_25296

/-- Represents a 6-digit number as a tuple of its digits -/
def SixDigitNumber := (Nat × Nat × Nat × Nat × Nat × Nat)

/-- Converts a 6-digit number tuple to its numerical value -/
def toNumber (n : SixDigitNumber) : Nat :=
  match n with
  | (a, b, c, d, e, f) => 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

/-- Extracts the last three digits of a 6-digit number tuple -/
def lastThreeDigits (n : SixDigitNumber) : Nat :=
  match n with
  | (_, _, _, d, e, f) => 100 * d + 10 * e + f

/-- Checks if a given 6-digit number satisfies the condition (abcdef) = (def)^2 -/
def satisfiesCondition (n : SixDigitNumber) : Prop :=
  toNumber n = (lastThreeDigits n) ^ 2

theorem six_digit_square_numbers :
  ∀ n : SixDigitNumber,
    satisfiesCondition n →
    (toNumber n = 390625 ∨ toNumber n = 141376) :=
by sorry

end NUMINAMATH_CALUDE_six_digit_square_numbers_l252_25296


namespace NUMINAMATH_CALUDE_car_speed_l252_25288

/-- Proves that if a car travels 1 km in 5 seconds more than it would take at 90 km/hour, then its speed is 80 km/hour. -/
theorem car_speed (v : ℝ) (h : v > 0) : 
  (3600 / v) = (3600 / 90) + 5 → v = 80 := by
sorry

end NUMINAMATH_CALUDE_car_speed_l252_25288


namespace NUMINAMATH_CALUDE_prairie_total_area_l252_25234

/-- The total area of a prairie given the area covered by a dust storm and the area left untouched. -/
theorem prairie_total_area (dust_covered : ℕ) (untouched : ℕ) 
  (h1 : dust_covered = 64535) 
  (h2 : untouched = 522) : 
  dust_covered + untouched = 65057 := by
  sorry

end NUMINAMATH_CALUDE_prairie_total_area_l252_25234


namespace NUMINAMATH_CALUDE_parabola_through_point_2_4_l252_25243

/-- A parabola passing through the point (2, 4) can be represented by either y² = 8x or x² = y -/
theorem parabola_through_point_2_4 :
  ∃ (f : ℝ → ℝ), (f 2 = 4 ∧ (∀ x y : ℝ, y = f x ↔ (y^2 = 8*x ∨ x^2 = y))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_through_point_2_4_l252_25243


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l252_25274

theorem sum_of_a_and_b (a b : ℝ) 
  (ha : |a| = 5)
  (hb : |b| = 3)
  (hab : |a - b| = b - a) :
  a + b = -2 ∨ a + b = -8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l252_25274


namespace NUMINAMATH_CALUDE_monotonic_functional_equation_solution_l252_25200

-- Define a monotonic function f from real numbers to real numbers
def monotonic_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y ∨ ∀ x y, x ≤ y → f x ≥ f y

-- Define the functional equation
def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

-- Theorem statement
theorem monotonic_functional_equation_solution :
  ∀ f : ℝ → ℝ, monotonic_function f → functional_equation f →
  ∃ a : ℝ, (a > 1 ∨ 0 < a ∧ a < 1) ∧ ∀ x, f x = a^x :=
sorry

end NUMINAMATH_CALUDE_monotonic_functional_equation_solution_l252_25200


namespace NUMINAMATH_CALUDE_gcf_of_72_90_120_l252_25230

theorem gcf_of_72_90_120 : Nat.gcd 72 (Nat.gcd 90 120) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_72_90_120_l252_25230


namespace NUMINAMATH_CALUDE_complex_modulus_power_l252_25255

theorem complex_modulus_power : 
  Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_power_l252_25255


namespace NUMINAMATH_CALUDE_alok_payment_l252_25251

/-- Represents the order and prices of items in Alok's purchase --/
structure AlokOrder where
  chapati_quantity : ℕ
  rice_quantity : ℕ
  vegetable_quantity : ℕ
  icecream_quantity : ℕ
  chapati_price : ℕ
  rice_price : ℕ
  vegetable_price : ℕ

/-- Calculates the total cost of Alok's order --/
def total_cost (order : AlokOrder) : ℕ :=
  order.chapati_quantity * order.chapati_price +
  order.rice_quantity * order.rice_price +
  order.vegetable_quantity * order.vegetable_price

/-- Theorem stating that Alok's total payment is 811 --/
theorem alok_payment (order : AlokOrder)
  (h1 : order.chapati_quantity = 16)
  (h2 : order.rice_quantity = 5)
  (h3 : order.vegetable_quantity = 7)
  (h4 : order.icecream_quantity = 6)
  (h5 : order.chapati_price = 6)
  (h6 : order.rice_price = 45)
  (h7 : order.vegetable_price = 70) :
  total_cost order = 811 := by
  sorry

end NUMINAMATH_CALUDE_alok_payment_l252_25251


namespace NUMINAMATH_CALUDE_square_between_squares_l252_25218

theorem square_between_squares (n k l m : ℕ) :
  m^2 < n ∧ n < (m+1)^2 ∧ n - k = m^2 ∧ n + l = (m+1)^2 →
  ∃ p : ℕ, n - k * l = p^2 := by
sorry

end NUMINAMATH_CALUDE_square_between_squares_l252_25218


namespace NUMINAMATH_CALUDE_alphabet_proof_main_theorem_l252_25292

/-- Represents an alphabet with letters containing dots and/or straight lines -/
structure Alphabet where
  total : ℕ
  both : ℕ
  line_only : ℕ
  dot_only : ℕ
  h_total : total = both + line_only + dot_only
  h_all_types : total > 0

/-- The specific alphabet described in the problem -/
def problem_alphabet : Alphabet where
  total := 40
  both := 9
  line_only := 24
  dot_only := 7
  h_total := by rfl
  h_all_types := by norm_num

/-- Theorem stating that the problem_alphabet satisfies the given conditions -/
theorem alphabet_proof : 
  ∃ (a : Alphabet), 
    a.total = 40 ∧ 
    a.both = 9 ∧ 
    a.line_only = 24 ∧ 
    a.dot_only = 7 :=
by
  use problem_alphabet
  simp [problem_alphabet]

/-- Main theorem to prove -/
theorem main_theorem (a : Alphabet) 
  (h1 : a.total = 40)
  (h2 : a.both = 9)
  (h3 : a.line_only = 24) :
  a.dot_only = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_alphabet_proof_main_theorem_l252_25292


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l252_25205

theorem triangle_side_ratio (a b c k : ℝ) 
  (triangle_sides : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (side_ratio : a^2 + b^2 = k * c^2) : 
  k > 0.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l252_25205


namespace NUMINAMATH_CALUDE_range_of_a_l252_25254

theorem range_of_a (a b : ℝ) 
  (h1 : 0 ≤ a - b ∧ a - b ≤ 1) 
  (h2 : 1 ≤ a + b ∧ a + b ≤ 4) : 
  1/2 ≤ a ∧ a ≤ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l252_25254


namespace NUMINAMATH_CALUDE_expression_evaluation_l252_25262

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -2
  (3*x + 2*y) * (3*x - 2*y) - 5*x*(x - y) - (2*x - y)^2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l252_25262


namespace NUMINAMATH_CALUDE_exactly_two_support_probability_l252_25279

theorem exactly_two_support_probability (p : ℝ) (h : p = 0.6) :
  let q := 1 - p
  3 * p^2 * q = 0.432 := by sorry

end NUMINAMATH_CALUDE_exactly_two_support_probability_l252_25279


namespace NUMINAMATH_CALUDE_geometric_sequence_logarithm_l252_25212

theorem geometric_sequence_logarithm (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = -Real.sqrt 2 * a n) :
  Real.log (a 2017)^2 - Real.log (a 2016)^2 = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_logarithm_l252_25212


namespace NUMINAMATH_CALUDE_remainder_sum_equality_l252_25204

/-- The remainder sum function -/
def r (n : ℕ) : ℕ := (Finset.range n).sum (λ i => n % (i + 1))

/-- Theorem: The remainder sum of 2^k - 1 equals the remainder sum of 2^k for all k ≥ 1 -/
theorem remainder_sum_equality (k : ℕ) (hk : k ≥ 1) : r (2^k - 1) = r (2^k) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_equality_l252_25204


namespace NUMINAMATH_CALUDE_chord_length_no_intersection_tangent_two_intersections_one_intersection_l252_25224

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12*x

-- Define the line y = 2x - 6
def line1 (x y : ℝ) : Prop := y = 2*x - 6

-- Define the line y = kx + 1
def line2 (k x y : ℝ) : Prop := y = k*x + 1

-- Theorem for the chord length
theorem chord_length : ∃ (x1 y1 x2 y2 : ℝ),
  parabola x1 y1 ∧ parabola x2 y2 ∧ 
  line1 x1 y1 ∧ line1 x2 y2 ∧
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2 : ℝ) = 15 := by sorry

-- Theorems for the positional relationships
theorem no_intersection (k : ℝ) : 
  k > 3 → ¬∃ (x y : ℝ), parabola x y ∧ line2 k x y := by sorry

theorem tangent : 
  ∃! (x y : ℝ), parabola x y ∧ line2 3 x y := by sorry

theorem two_intersections (k : ℝ) : 
  k < 3 ∧ k ≠ 0 → ∃ (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧ parabola x1 y1 ∧ parabola x2 y2 ∧ 
    line2 k x1 y1 ∧ line2 k x2 y2 := by sorry

theorem one_intersection : 
  ∃! (x y : ℝ), parabola x y ∧ line2 0 x y := by sorry

end NUMINAMATH_CALUDE_chord_length_no_intersection_tangent_two_intersections_one_intersection_l252_25224


namespace NUMINAMATH_CALUDE_sunghoon_scores_l252_25281

theorem sunghoon_scores (korean math english : ℝ) 
  (h1 : korean / math = 1.2) 
  (h2 : math / english = 5/6) : 
  korean / english = 1 := by
  sorry

end NUMINAMATH_CALUDE_sunghoon_scores_l252_25281


namespace NUMINAMATH_CALUDE_linear_system_solution_l252_25278

theorem linear_system_solution :
  ∀ x y : ℚ,
  (2 * x + y = 6) →
  (x + 2 * y = 5) →
  ((x + y) / 3 = 11 / 9) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l252_25278


namespace NUMINAMATH_CALUDE_no_infinite_power_arithmetic_progression_l252_25298

/-- Represents a term in the sequence of the form a^b -/
def PowerTerm := Nat → Nat

/-- Represents an arithmetic progression -/
def ArithmeticProgression (f : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, f (n + 1) = f n + d

/-- A function that checks if a number is of the form a^b with a, b positive integers and b ≥ 2 -/
def IsPowerForm (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b ≥ 2 ∧ n = a^b

/-- The main theorem stating that no infinite non-constant arithmetic progression
    exists where each term is of the form a^b with a, b positive integers and b ≥ 2 -/
theorem no_infinite_power_arithmetic_progression :
  ¬∃ f : PowerTerm, ArithmeticProgression f ∧
    (∀ n, IsPowerForm (f n)) ∧
    (∃ d : ℕ, d > 0 ∧ ∀ n : ℕ, f (n + 1) = f n + d) :=
sorry

end NUMINAMATH_CALUDE_no_infinite_power_arithmetic_progression_l252_25298


namespace NUMINAMATH_CALUDE_polynomial_less_than_factorial_l252_25258

theorem polynomial_less_than_factorial (A B C : ℝ) : 
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (A * n^2 + B * n + C : ℝ) < n! :=
sorry

end NUMINAMATH_CALUDE_polynomial_less_than_factorial_l252_25258


namespace NUMINAMATH_CALUDE_direct_square_variation_theorem_l252_25268

/-- A function representing direct variation with the square of x -/
def direct_square_variation (k : ℝ) (x : ℝ) : ℝ := k * x^2

theorem direct_square_variation_theorem (y : ℝ → ℝ) :
  (∃ k : ℝ, ∀ x, y x = direct_square_variation k x) →
  y 3 = 18 →
  y 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_direct_square_variation_theorem_l252_25268


namespace NUMINAMATH_CALUDE_extreme_value_conditions_l252_25272

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extreme_value_conditions (a b : ℝ) :
  f a b 1 = 10 ∧ 
  (deriv (f a b)) 1 = 0 →
  a = 4 ∧ b = -11 := by sorry

end NUMINAMATH_CALUDE_extreme_value_conditions_l252_25272


namespace NUMINAMATH_CALUDE_power_of_power_eq_expanded_power_l252_25210

theorem power_of_power_eq_expanded_power (x : ℝ) : (2 * x^2)^3 = 8 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_eq_expanded_power_l252_25210


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l252_25249

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (a^2 - 2*a : ℝ) + (a^2 - a - 2 : ℝ) * I
  (z.re = 0) → (a = 0 ∨ a = 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l252_25249


namespace NUMINAMATH_CALUDE_gcd_117_182_f_neg_one_l252_25203

-- Define the polynomial f(x)
def f (x : ℤ) : ℤ := 1 - 9*x + 8*x^2 - 4*x^4 + 5*x^5 + 3*x^6

-- Theorem for the GCD of 117 and 182
theorem gcd_117_182 : Nat.gcd 117 182 = 13 := by sorry

-- Theorem for the value of f(-1)
theorem f_neg_one : f (-1) = 12 := by sorry

end NUMINAMATH_CALUDE_gcd_117_182_f_neg_one_l252_25203


namespace NUMINAMATH_CALUDE_triangle_area_l252_25209

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 9√3/14 when a = 3, b = 2c, and A = 2π/3 -/
theorem triangle_area (a b c : ℝ) (A : ℝ) :
  a = 3 →
  b = 2 * c →
  A = 2 * Real.pi / 3 →
  (1 / 2 : ℝ) * b * c * Real.sin A = 9 * Real.sqrt 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l252_25209


namespace NUMINAMATH_CALUDE_f_value_at_negative_five_pi_thirds_l252_25264

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_value_at_negative_five_pi_thirds 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f π)
  (h_cos : ∀ x ∈ Set.Icc (-π/2) 0, f x = Real.cos x) :
  f (-5*π/3) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_negative_five_pi_thirds_l252_25264


namespace NUMINAMATH_CALUDE_officer_selection_count_l252_25225

/-- The number of members in the club -/
def club_size : ℕ := 12

/-- The number of officer positions to be filled -/
def officer_positions : ℕ := 5

/-- The number of ways to select distinct officers from the club members -/
def officer_selection_ways : ℕ := club_size * (club_size - 1) * (club_size - 2) * (club_size - 3) * (club_size - 4)

/-- Theorem stating that the number of ways to select officers is 95,040 -/
theorem officer_selection_count :
  officer_selection_ways = 95040 :=
by sorry

end NUMINAMATH_CALUDE_officer_selection_count_l252_25225


namespace NUMINAMATH_CALUDE_average_income_of_A_and_B_l252_25239

/-- Given the average monthly incomes of different pairs of people and the income of one person,
    prove that the average monthly income of A and B is 5050. -/
theorem average_income_of_A_and_B (A B C : ℕ) : 
  A = 4000 →
  (B + C) / 2 = 6250 →
  (A + C) / 2 = 5200 →
  (A + B) / 2 = 5050 := by
  sorry


end NUMINAMATH_CALUDE_average_income_of_A_and_B_l252_25239


namespace NUMINAMATH_CALUDE_certain_number_solution_l252_25237

theorem certain_number_solution : 
  ∃ x : ℝ, (5100 - (102 / x) = 5095) ∧ (x = 20.4) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_solution_l252_25237


namespace NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l252_25291

theorem right_triangle_median_to_hypotenuse (DE DF : ℝ) :
  DE = 15 →
  DF = 20 →
  let EF := Real.sqrt (DE^2 + DF^2)
  let median := EF / 2
  median = 12.5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_median_to_hypotenuse_l252_25291


namespace NUMINAMATH_CALUDE_two_digit_product_sum_l252_25213

theorem two_digit_product_sum : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 3024 ∧ 
  a + b = 120 := by
sorry

end NUMINAMATH_CALUDE_two_digit_product_sum_l252_25213


namespace NUMINAMATH_CALUDE_trigonometric_equality_l252_25266

theorem trigonometric_equality (α : ℝ) :
  (2 * Real.cos (π/6 - 2*α) - Real.sqrt 3 * Real.sin (5*π/2 - 2*α)) /
  (Real.cos (9*π/2 - 2*α) + 2 * Real.cos (π/6 + 2*α)) =
  Real.tan (2*α) / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l252_25266


namespace NUMINAMATH_CALUDE_log_relation_l252_25217

theorem log_relation (y : ℝ) (m : ℝ) : 
  Real.log 5 / Real.log 9 = y → Real.log 125 / Real.log 3 = m * y → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l252_25217


namespace NUMINAMATH_CALUDE_nine_to_power_2023_div_3_l252_25219

theorem nine_to_power_2023_div_3 (n : ℕ) : n = 9^2023 → n / 3 = 3^4045 :=
by
  sorry

end NUMINAMATH_CALUDE_nine_to_power_2023_div_3_l252_25219


namespace NUMINAMATH_CALUDE_product_of_roots_l252_25242

theorem product_of_roots (x : ℝ) : (x - 1) * (x + 4) = 22 → ∃ y : ℝ, (y - 1) * (y + 4) = 22 ∧ x * y = -26 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l252_25242


namespace NUMINAMATH_CALUDE_max_volume_pyramid_l252_25260

/-- A triangular prism with vertices A, B, C, A₁, B₁, C₁ -/
structure TriangularPrism where
  volume : ℝ
  AA₁ : ℝ
  BB₁ : ℝ
  CC₁ : ℝ

/-- Points M, N, K on edges AA₁, BB₁, CC₁ respectively -/
structure PrismPoints (prism : TriangularPrism) where
  M : ℝ
  N : ℝ
  K : ℝ
  h_M : M ≤ prism.AA₁
  h_N : N ≤ prism.BB₁
  h_K : K ≤ prism.CC₁

/-- Theorem stating the maximum volume of pyramid MNKP -/
theorem max_volume_pyramid (prism : TriangularPrism) (points : PrismPoints prism) :
  prism.volume = 35 →
  points.M / prism.AA₁ = 5 / 6 →
  points.N / prism.BB₁ = 6 / 7 →
  points.K / prism.CC₁ = 2 / 3 →
  (∃ (P : ℝ), (P ≥ 0 ∧ P ≤ prism.AA₁) ∨ (P ≥ 0 ∧ P ≤ prism.BB₁) ∨ (P ≥ 0 ∧ P ≤ prism.CC₁)) →
  ∃ (pyramid_volume : ℝ), pyramid_volume ≤ 10 ∧ 
    ∀ (other_volume : ℝ), other_volume ≤ pyramid_volume := by
  sorry

end NUMINAMATH_CALUDE_max_volume_pyramid_l252_25260


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l252_25236

/-- Calculates the total cost of plastering a rectangular tank's walls and bottom. -/
def plasteringCost (length width depth : ℝ) (costPerSqm : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * costPerSqm

/-- Proves that the plastering cost for a tank with given dimensions is 223.2 rupees. -/
theorem tank_plastering_cost :
  let length : ℝ := 25
  let width : ℝ := 12
  let depth : ℝ := 6
  let costPerSqm : ℝ := 0.30  -- 30 paise = 0.30 rupees
  plasteringCost length width depth costPerSqm = 223.2 := by
  sorry

#eval plasteringCost 25 12 6 0.30

end NUMINAMATH_CALUDE_tank_plastering_cost_l252_25236


namespace NUMINAMATH_CALUDE_initial_production_rate_l252_25245

/-- Proves that the initial production rate is 15 cogs per hour given the problem conditions --/
theorem initial_production_rate : 
  ∀ (initial_rate : ℝ),
  (∃ (initial_time : ℝ),
    initial_rate * initial_time = 60 ∧  -- Initial order production
    initial_time + 1 = 120 / 24 ∧       -- Total time equation
    (60 + 60) / (initial_time + 1) = 24 -- Average output equation
  ) → initial_rate = 15 := by
  sorry


end NUMINAMATH_CALUDE_initial_production_rate_l252_25245


namespace NUMINAMATH_CALUDE_extended_box_with_hemispheres_volume_l252_25202

/-- The volume of a region formed by extending a rectangular parallelepiped and adding hemispheres at its vertices -/
theorem extended_box_with_hemispheres_volume 
  (l w h : ℝ) 
  (hl : l = 5) 
  (hw : w = 6) 
  (hh : h = 7) 
  (extension : ℝ) 
  (hemisphere_radius : ℝ) 
  (he : extension = 2) 
  (hr : hemisphere_radius = 2) : 
  (l + 2 * extension) * (w + 2 * extension) * (h + 2 * extension) + 
  8 * ((2 / 3) * π * hemisphere_radius^3) = 
  990 + (128 / 3) * π :=
sorry

end NUMINAMATH_CALUDE_extended_box_with_hemispheres_volume_l252_25202


namespace NUMINAMATH_CALUDE_inequality_system_solution_l252_25232

theorem inequality_system_solution (x : ℝ) :
  x > -6 - 2*x ∧ x ≤ (3 + x) / 4 → -2 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l252_25232


namespace NUMINAMATH_CALUDE_sphere_volume_with_diameter_10_l252_25271

/-- The volume of a sphere with diameter 10 meters is 500/3 * π cubic meters. -/
theorem sphere_volume_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let volume : ℝ := (4 / 3) * π * radius^3
  volume = (500 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_with_diameter_10_l252_25271


namespace NUMINAMATH_CALUDE_fourth_number_value_l252_25214

theorem fourth_number_value (a b c d e f g : ℝ) 
  (h1 : (a + b + c + d) / 4 = 4)
  (h2 : (d + e + f + g) / 4 = 4)
  (h3 : (a + b + c + d + e + f + g) / 7 = 3) :
  d = 11 := by sorry

end NUMINAMATH_CALUDE_fourth_number_value_l252_25214


namespace NUMINAMATH_CALUDE_intersection_points_on_circle_l252_25227

/-- The parabolas y = (x - 1)^2 and x - 2 = (y + 1)^2 intersect at four points. 
    These points lie on a circle with radius squared equal to 1/4. -/
theorem intersection_points_on_circle : 
  ∃ (c : ℝ × ℝ) (r : ℝ),
    (∀ (p : ℝ × ℝ), 
      (p.2 = (p.1 - 1)^2 ∧ p.1 - 2 = (p.2 + 1)^2) → 
      (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    r^2 = (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_on_circle_l252_25227


namespace NUMINAMATH_CALUDE_radius_of_Q_l252_25263

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
axiom P : Circle
axiom Q : Circle
axiom R : Circle
axiom S : Circle

-- Define the conditions
axiom externally_tangent : P.radius + Q.radius = dist P.center Q.center ∧
                           P.radius + R.radius = dist P.center R.center ∧
                           Q.radius + R.radius = dist Q.center R.center

axiom internally_tangent : S.radius = P.radius + dist P.center S.center ∧
                           S.radius = Q.radius + dist Q.center S.center ∧
                           S.radius = R.radius + dist R.center S.center

axiom Q_R_congruent : Q.radius = R.radius

axiom P_radius : P.radius = 2

axiom P_through_S_center : dist P.center S.center = P.radius

-- Theorem to prove
theorem radius_of_Q : Q.radius = 16/9 := by sorry

end NUMINAMATH_CALUDE_radius_of_Q_l252_25263


namespace NUMINAMATH_CALUDE_sixth_number_is_52_when_i_7_l252_25206

/-- Represents the systematic sampling function for a population of 100 individuals -/
def systematicSample (i : Nat) (k : Nat) : Nat :=
  let drawnNumber := i + k
  if drawnNumber ≥ 10 then drawnNumber - 10 else drawnNumber

/-- Theorem stating that the 6th number drawn is 52 when i=7 -/
theorem sixth_number_is_52_when_i_7 :
  ∀ (populationSize : Nat) (segmentCount : Nat) (sampleSize : Nat) (i : Nat),
    populationSize = 100 →
    segmentCount = 10 →
    sampleSize = 10 →
    i = 7 →
    systematicSample i 5 = 52 := by
  sorry

end NUMINAMATH_CALUDE_sixth_number_is_52_when_i_7_l252_25206


namespace NUMINAMATH_CALUDE_curve_classification_l252_25233

structure Curve where
  m : ℝ
  n : ℝ

def isEllipse (c : Curve) : Prop :=
  c.m > c.n ∧ c.n > 0

def isHyperbola (c : Curve) : Prop :=
  c.m * c.n < 0

def isTwoLines (c : Curve) : Prop :=
  c.m = 0 ∧ c.n > 0

theorem curve_classification (c : Curve) :
  (isEllipse c → ∃ foci : ℝ × ℝ, foci.1 = 0) ∧
  (isHyperbola c → ∃ k : ℝ, k^2 = -c.m / c.n) ∧
  (isTwoLines c → ∃ y₁ y₂ : ℝ, y₁ = -y₂ ∧ y₁^2 = 1 / c.n) :=
sorry

end NUMINAMATH_CALUDE_curve_classification_l252_25233


namespace NUMINAMATH_CALUDE_complex_number_sum_of_parts_l252_25226

theorem complex_number_sum_of_parts (m : ℝ) : 
  let z : ℂ := m / (1 - Complex.I) + (1 - Complex.I) / 2 * Complex.I
  (z.re + z.im = 1) → m = 1 := by sorry

end NUMINAMATH_CALUDE_complex_number_sum_of_parts_l252_25226


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_nonnegative_reals_l252_25222

open Set

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x, y = 2 * x}
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Define the intersection set
def intersection_set : Set ℝ := {y | y ≥ 0}

-- Theorem statement
theorem A_intersect_B_equals_nonnegative_reals : A ∩ B = intersection_set := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_nonnegative_reals_l252_25222


namespace NUMINAMATH_CALUDE_farm_tax_collection_l252_25220

/-- Represents the farm tax collection scenario in a village -/
structure FarmTaxScenario where
  /-- Total cultivated land in the village -/
  total_cultivated_land : ℝ
  /-- Tax rate applied to taxable land -/
  tax_rate : ℝ
  /-- Proportion of cultivated land that is taxable (60%) -/
  taxable_land_ratio : ℝ
  /-- Mr. William's tax payment -/
  william_tax_payment : ℝ
  /-- Proportion of Mr. William's taxable land to total taxable land (16%) -/
  william_land_ratio : ℝ

/-- Calculates the total farm tax collected from the village -/
def total_farm_tax (scenario : FarmTaxScenario) : ℝ :=
  scenario.total_cultivated_land * scenario.taxable_land_ratio * scenario.tax_rate

/-- Theorem stating that the total farm tax collected is $3000 -/
theorem farm_tax_collection (scenario : FarmTaxScenario) 
  (h1 : scenario.taxable_land_ratio = 0.6)
  (h2 : scenario.william_tax_payment = 480)
  (h3 : scenario.william_land_ratio = 0.16) :
  total_farm_tax scenario = 3000 := by
  sorry

#check farm_tax_collection

end NUMINAMATH_CALUDE_farm_tax_collection_l252_25220


namespace NUMINAMATH_CALUDE_total_vegetarian_count_l252_25229

/-- Represents the dietary preferences in a family -/
structure DietaryPreferences where
  only_vegetarian : ℕ
  only_non_vegetarian : ℕ
  both_veg_and_non_veg : ℕ
  vegan : ℕ
  vegan_and_vegetarian : ℕ
  gluten_free_from_both : ℕ

/-- Calculates the total number of people eating vegetarian food -/
def total_vegetarian (d : DietaryPreferences) : ℕ :=
  d.only_vegetarian + d.both_veg_and_non_veg + (d.vegan - d.vegan_and_vegetarian)

/-- Theorem stating the total number of people eating vegetarian food -/
theorem total_vegetarian_count (d : DietaryPreferences)
  (h1 : d.only_vegetarian = 15)
  (h2 : d.only_non_vegetarian = 8)
  (h3 : d.both_veg_and_non_veg = 11)
  (h4 : d.vegan = 5)
  (h5 : d.vegan_and_vegetarian = 3)
  (h6 : d.gluten_free_from_both = 2)
  : total_vegetarian d = 28 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetarian_count_l252_25229


namespace NUMINAMATH_CALUDE_inequality_solution_l252_25280

theorem inequality_solution (x : ℝ) : 
  (4 ≤ x^2 - 3*x - 6 ∧ x^2 - 3*x - 6 ≤ 2*x + 8) ↔ 
  ((5 ≤ x ∧ x ≤ 7) ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l252_25280


namespace NUMINAMATH_CALUDE_mn_gcd_lcm_equation_l252_25221

theorem mn_gcd_lcm_equation (m n : ℕ+) :
  m * n = (Nat.gcd m n)^2 + Nat.lcm m n →
  (m = 2 ∧ n = 4) ∨ (m = 4 ∧ n = 2) := by
  sorry

end NUMINAMATH_CALUDE_mn_gcd_lcm_equation_l252_25221


namespace NUMINAMATH_CALUDE_equal_candies_after_sharing_l252_25211

/-- The number of candies Minyoung and Taehyung should have to be equal -/
def target_candies (total_candies : ℕ) : ℕ :=
  total_candies / 2

/-- The number of candies Taehyung should take from Minyoung -/
def candies_to_take (minyoung_candies taehyung_candies : ℕ) : ℕ :=
  (minyoung_candies + taehyung_candies) / 2 - taehyung_candies

theorem equal_candies_after_sharing 
  (minyoung_initial : ℕ) 
  (taehyung_initial : ℕ) 
  (h1 : minyoung_initial = 9) 
  (h2 : taehyung_initial = 3) :
  let candies_taken := candies_to_take minyoung_initial taehyung_initial
  minyoung_initial - candies_taken = taehyung_initial + candies_taken ∧
  candies_taken = 3 :=
by sorry

#eval candies_to_take 9 3

end NUMINAMATH_CALUDE_equal_candies_after_sharing_l252_25211


namespace NUMINAMATH_CALUDE_not_perfect_square_l252_25290

theorem not_perfect_square : ∃ (n : ℕ), n = 6^2041 ∧
  (∀ (m : ℕ), m^2 ≠ n) ∧
  (∃ (a : ℕ), 3^2040 = a^2) ∧
  (∃ (b : ℕ), 7^2042 = b^2) ∧
  (∃ (c : ℕ), 8^2043 = c^2) ∧
  (∃ (d : ℕ), 9^2044 = d^2) :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_l252_25290


namespace NUMINAMATH_CALUDE_animal_permutations_l252_25265

/-- The number of animals excluding Rat and Snake -/
def n : ℕ := 4

/-- The total number of animals -/
def total_animals : ℕ := 6

/-- Theorem stating that the number of permutations of n distinct objects
    is equal to n factorial, where n is the number of animals excluding
    Rat and Snake -/
theorem animal_permutations :
  (Finset.range n).card.factorial = 24 :=
sorry

end NUMINAMATH_CALUDE_animal_permutations_l252_25265


namespace NUMINAMATH_CALUDE_vintik_shpuntik_journey_l252_25250

/-- The problem of Vintik and Shpuntik's journey to school -/
theorem vintik_shpuntik_journey 
  (distance : ℝ) 
  (vintik_scooter_speed : ℝ) 
  (walking_speed : ℝ) 
  (h_distance : distance = 6) 
  (h_vintik_scooter : vintik_scooter_speed = 10) 
  (h_walking : walking_speed = 5) :
  ∃ (shpuntik_bicycle_speed : ℝ),
    -- Vintik's journey
    ∃ (vintik_time : ℝ),
      vintik_time * (vintik_scooter_speed / 2 + walking_speed / 2) = distance ∧
    -- Shpuntik's journey
    (distance / 2) / shpuntik_bicycle_speed + (distance / 2) / walking_speed = vintik_time ∧
    -- Shpuntik's bicycle speed
    shpuntik_bicycle_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_vintik_shpuntik_journey_l252_25250


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l252_25223

theorem express_y_in_terms_of_x (x y : ℝ) :
  4 * x - y = 7 → y = 4 * x - 7 := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l252_25223


namespace NUMINAMATH_CALUDE_fraction_simplification_l252_25246

theorem fraction_simplification {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (a^(2*b) * b^(3*a)) / (b^(2*b) * a^(3*a)) = (a/b)^(2*b - 3*a) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l252_25246


namespace NUMINAMATH_CALUDE_line_slope_and_point_l252_25261

/-- Given two points P and Q in a coordinate plane, if the slope of the line
    through P and Q is -5/4, then the y-coordinate of Q is -2. Additionally,
    if R is a point on this line and is horizontally 6 units to the right of Q,
    then R has coordinates (11, -9.5). -/
theorem line_slope_and_point (P Q R : ℝ × ℝ) : 
  P = (-3, 8) →
  Q.1 = 5 →
  (Q.2 - P.2) / (Q.1 - P.1) = -5/4 →
  R.1 = Q.1 + 6 →
  (R.2 - Q.2) / (R.1 - Q.1) = -5/4 →
  Q.2 = -2 ∧ R = (11, -9.5) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_and_point_l252_25261


namespace NUMINAMATH_CALUDE_triangle_properties_l252_25269

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A))
  (h2 : t.A = 2 * t.B)
  (h3 : t.A + t.B + t.C = Real.pi) :  -- Triangle angle sum property
  t.C = 5 * Real.pi / 8 ∧ 2 * t.a ^ 2 = t.b ^ 2 + t.c ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l252_25269


namespace NUMINAMATH_CALUDE_trivia_team_selection_l252_25215

theorem trivia_team_selection (total_students : ℕ) (num_groups : ℕ) (students_per_group : ℕ) 
  (h1 : total_students = 36)
  (h2 : num_groups = 3)
  (h3 : students_per_group = 9) :
  total_students - (num_groups * students_per_group) = 9 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_selection_l252_25215


namespace NUMINAMATH_CALUDE_no_solutions_to_equation_l252_25284

theorem no_solutions_to_equation : 
  ¬∃ (x : ℝ), |x - 1| = |2*x - 4| + |x - 5| := by
sorry

end NUMINAMATH_CALUDE_no_solutions_to_equation_l252_25284


namespace NUMINAMATH_CALUDE_united_charge_per_minute_is_correct_l252_25241

/-- Additional charge per minute for United Telephone -/
def united_charge_per_minute : ℚ := 25 / 100

/-- Base rate for United Telephone -/
def united_base_rate : ℚ := 7

/-- Base rate for Atlantic Call -/
def atlantic_base_rate : ℚ := 12

/-- Additional charge per minute for Atlantic Call -/
def atlantic_charge_per_minute : ℚ := 1 / 5

/-- Number of minutes for which the bills are equal -/
def equal_minutes : ℕ := 100

theorem united_charge_per_minute_is_correct :
  united_base_rate + equal_minutes * united_charge_per_minute =
  atlantic_base_rate + equal_minutes * atlantic_charge_per_minute :=
by sorry

end NUMINAMATH_CALUDE_united_charge_per_minute_is_correct_l252_25241


namespace NUMINAMATH_CALUDE_equation_solution_l252_25259

theorem equation_solution :
  ∃ y : ℚ, (y + 1/3 = 3/8 - 1/4) ∧ (y = -5/24) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l252_25259


namespace NUMINAMATH_CALUDE_vincent_book_expenditure_l252_25228

def animal_books : ℕ := 10
def space_books : ℕ := 1
def train_books : ℕ := 3
def book_cost : ℕ := 16

theorem vincent_book_expenditure :
  (animal_books + space_books + train_books) * book_cost = 224 := by
  sorry

end NUMINAMATH_CALUDE_vincent_book_expenditure_l252_25228


namespace NUMINAMATH_CALUDE_wall_ratio_l252_25240

/-- Given a wall with specific dimensions, prove that the ratio of its length to its height is 7:1 -/
theorem wall_ratio (w h l : ℝ) : 
  w = 3 →                 -- width is 3 meters
  h = 6 * w →             -- height is 6 times the width
  w * h * l = 6804 →      -- volume is 6804 cubic meters
  l / h = 7 := by
sorry

end NUMINAMATH_CALUDE_wall_ratio_l252_25240


namespace NUMINAMATH_CALUDE_ball_count_theorem_l252_25244

theorem ball_count_theorem (B W : ℕ) (h1 : W = 3 * B) 
  (h2 : 5 * B + W = 2 * (B + W)) : 
  B + 5 * W = 4 * (B + W) := by
sorry

end NUMINAMATH_CALUDE_ball_count_theorem_l252_25244


namespace NUMINAMATH_CALUDE_percentage_of_workday_in_meetings_l252_25277

/-- Represents the duration of a workday in hours -/
def workday_hours : ℕ := 9

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 45

/-- Calculates the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 2 * first_meeting_minutes

/-- Calculates the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Calculates the total workday time in minutes -/
def total_workday_minutes : ℕ := workday_hours * 60

/-- Theorem stating that the percentage of the workday spent in meetings is 25% -/
theorem percentage_of_workday_in_meetings : 
  (total_meeting_minutes : ℚ) / (total_workday_minutes : ℚ) * 100 = 25 := by
  sorry


end NUMINAMATH_CALUDE_percentage_of_workday_in_meetings_l252_25277


namespace NUMINAMATH_CALUDE_no_integer_solutions_l252_25247

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), 
    (x^6 + x^3 + x^3*y + y = 147^157) ∧ 
    (x^3 + x^3*y + y^2 + y + z^9 = 157^147) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l252_25247


namespace NUMINAMATH_CALUDE_fuel_consumption_problem_l252_25289

/-- The fuel consumption problem for an aviation engineer --/
theorem fuel_consumption_problem 
  (fuel_per_person : ℝ) 
  (fuel_per_bag : ℝ) 
  (num_passengers : ℕ) 
  (num_crew : ℕ) 
  (bags_per_person : ℕ) 
  (total_fuel : ℝ) 
  (trip_distance : ℝ) 
  (h1 : fuel_per_person = 3)
  (h2 : fuel_per_bag = 2)
  (h3 : num_passengers = 30)
  (h4 : num_crew = 5)
  (h5 : bags_per_person = 2)
  (h6 : total_fuel = 106000)
  (h7 : trip_distance = 400) :
  let total_people := num_passengers + num_crew
  let total_bags := total_people * bags_per_person
  let additional_fuel_per_mile := total_people * fuel_per_person + total_bags * fuel_per_bag
  let total_fuel_per_mile := total_fuel / trip_distance
  total_fuel_per_mile - additional_fuel_per_mile = 20 := by
  sorry

end NUMINAMATH_CALUDE_fuel_consumption_problem_l252_25289


namespace NUMINAMATH_CALUDE_absolute_sum_diff_equal_implies_product_zero_l252_25201

theorem absolute_sum_diff_equal_implies_product_zero (a b : ℝ) :
  |a + b| = |a - b| → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_sum_diff_equal_implies_product_zero_l252_25201


namespace NUMINAMATH_CALUDE_ash_cloud_radius_l252_25267

theorem ash_cloud_radius (height : ℝ) (diameter_ratio : ℝ) : 
  height = 300 → diameter_ratio = 18 → (diameter_ratio * height) / 2 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_ash_cloud_radius_l252_25267


namespace NUMINAMATH_CALUDE_probability_rain_at_least_one_day_l252_25257

/-- The probability of rain on at least one day given independent probabilities for each day -/
theorem probability_rain_at_least_one_day 
  (p_friday p_saturday p_sunday : ℝ) 
  (h_friday : p_friday = 0.3)
  (h_saturday : p_saturday = 0.45)
  (h_sunday : p_sunday = 0.55)
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - p_friday) * (1 - p_saturday) * (1 - p_sunday) = 0.82675 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_at_least_one_day_l252_25257


namespace NUMINAMATH_CALUDE_cylinder_equation_l252_25238

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Definition of a cylinder in cylindrical coordinates -/
def IsCylinder (S : Set CylindricalPoint) (c : ℝ) : Prop :=
  ∀ p : CylindricalPoint, p ∈ S ↔ p.r = c

theorem cylinder_equation (c : ℝ) (h : c > 0) :
  IsCylinder {p : CylindricalPoint | p.r = c} c :=
by sorry

end NUMINAMATH_CALUDE_cylinder_equation_l252_25238


namespace NUMINAMATH_CALUDE_log_equation_solution_l252_25248

theorem log_equation_solution (x : ℝ) : Real.log (729 : ℝ) / Real.log (3 * x) = x → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l252_25248


namespace NUMINAMATH_CALUDE_worker_count_l252_25275

theorem worker_count (total : ℕ) (extra_total : ℕ) (extra_per_worker : ℕ) : 
  (total = 300000) → 
  (extra_total = 375000) → 
  (extra_per_worker = 50) → 
  (∃ (w : ℕ), w * (extra_total / w - total / w) = extra_per_worker ∧ w = 1500) :=
by sorry

end NUMINAMATH_CALUDE_worker_count_l252_25275


namespace NUMINAMATH_CALUDE_walk_time_calculation_l252_25207

/-- The time it takes Mark to walk into the courthouse each day -/
def walk_time : ℕ := sorry

/-- The number of work days in a week -/
def work_days : ℕ := 5

/-- The time it takes to find parking each day -/
def parking_time : ℕ := 5

/-- The time it takes to get through the metal detector on crowded days -/
def crowded_detector_time : ℕ := 30

/-- The time it takes to get through the metal detector on less crowded days -/
def less_crowded_detector_time : ℕ := 10

/-- The number of crowded days per week -/
def crowded_days : ℕ := 2

/-- The number of less crowded days per week -/
def less_crowded_days : ℕ := 3

/-- The total time spent on all activities in a week -/
def total_weekly_time : ℕ := 130

theorem walk_time_calculation : 
  walk_time = 3 ∧
  work_days * (parking_time + walk_time) + 
  crowded_days * crowded_detector_time +
  less_crowded_days * less_crowded_detector_time = 
  total_weekly_time :=
sorry

end NUMINAMATH_CALUDE_walk_time_calculation_l252_25207


namespace NUMINAMATH_CALUDE_mandatoryQuestions_eq_13_l252_25295

/-- Represents a math competition with mandatory and optional questions -/
structure MathCompetition where
  totalQuestions : ℕ
  correctAnswers : ℕ
  totalScore : ℕ
  mandatoryCorrectPoints : ℕ
  mandatoryIncorrectPoints : ℕ
  optionalCorrectPoints : ℕ

/-- Calculates the number of mandatory questions in the competition -/
def mandatoryQuestions (comp : MathCompetition) : ℕ :=
  sorry

/-- Theorem stating that the number of mandatory questions is 13 -/
theorem mandatoryQuestions_eq_13 (comp : MathCompetition) 
  (h1 : comp.totalQuestions = 25)
  (h2 : comp.correctAnswers = 15)
  (h3 : comp.totalScore = 49)
  (h4 : comp.mandatoryCorrectPoints = 3)
  (h5 : comp.mandatoryIncorrectPoints = 2)
  (h6 : comp.optionalCorrectPoints = 5) :
  mandatoryQuestions comp = 13 := by
  sorry

end NUMINAMATH_CALUDE_mandatoryQuestions_eq_13_l252_25295


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l252_25276

theorem root_exists_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo (1/2 : ℝ) 1 ∧ Real.exp x = 1/x := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l252_25276


namespace NUMINAMATH_CALUDE_first_group_size_correct_l252_25286

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 63

/-- The number of days the first group takes to repair the road -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group takes to repair the road -/
def second_group_days : ℕ := 21

/-- The number of hours per day the second group works -/
def second_group_hours : ℕ := 6

/-- The theorem stating that the first group size is correct -/
theorem first_group_size_correct :
  first_group_size * first_group_days * first_group_hours =
  second_group_size * second_group_days * second_group_hours :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_correct_l252_25286


namespace NUMINAMATH_CALUDE_inequality_system_solution_l252_25294

theorem inequality_system_solution :
  ∃ (x y : ℝ), 
    (13 * x^2 - 4 * x * y + 4 * y^2 ≤ 2) ∧
    (2 * x - 4 * y ≤ -3) ∧
    (x = -1/3) ∧ (y = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l252_25294


namespace NUMINAMATH_CALUDE_ab_power_2023_l252_25253

theorem ab_power_2023 (a b : ℝ) (h : |a - 2| + (b + 1/2)^2 = 0) : (a * b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ab_power_2023_l252_25253


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l252_25208

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 34 →
  (1/2) * a * b = 24 →
  a^2 + b^2 = c^2 →
  c = 62/4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l252_25208


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l252_25256

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l252_25256
