import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l232_23268

/-- An isosceles triangle with side lengths 4, 9, and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c => 
    (a = 4 ∧ b = 9 ∧ c = 9) →  -- Two sides are 9, one side is 4
    (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
    (b = c) →  -- Isosceles condition
    (a + b + c = 22)  -- Perimeter is 22

#check isosceles_triangle_perimeter

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : 
  ∃ (a b c : ℝ), isosceles_triangle_perimeter a b c :=
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l232_23268


namespace NUMINAMATH_CALUDE_emily_game_lives_l232_23294

def game_lives (initial : ℕ) (lost : ℕ) (final : ℕ) : ℕ :=
  final - (initial - lost)

theorem emily_game_lives :
  game_lives 42 25 41 = 24 := by
  sorry

end NUMINAMATH_CALUDE_emily_game_lives_l232_23294


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_13_l232_23278

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 7 ∨ d = 1

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digit_sum_13 :
  ∀ n : ℕ, is_valid_number n → digit_sum n = 13 → n ≤ 7111111 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_13_l232_23278


namespace NUMINAMATH_CALUDE_triple_transmission_more_accurate_main_theorem_l232_23207

/-- Probability of correctly decoding 0 in single transmission -/
def single_transmission_prob (α : ℝ) : ℝ := 1 - α

/-- Probability of correctly decoding 0 in triple transmission -/
def triple_transmission_prob (α : ℝ) : ℝ := 3 * α * (1 - α)^2 + (1 - α)^3

/-- Theorem stating that triple transmission is more accurate than single for sending 0 when 0 < α < 0.5 -/
theorem triple_transmission_more_accurate (α : ℝ) 
  (h1 : 0 < α) (h2 : α < 0.5) : 
  triple_transmission_prob α > single_transmission_prob α := by
  sorry

/-- Main theorem combining all conditions and the result -/
theorem main_theorem (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < 0.5) (h3 : 0 < β) (h4 : β < 1) :
  triple_transmission_prob α > single_transmission_prob α ∧
  single_transmission_prob α = 1 - α ∧
  triple_transmission_prob α = 3 * α * (1 - α)^2 + (1 - α)^3 := by
  sorry

end NUMINAMATH_CALUDE_triple_transmission_more_accurate_main_theorem_l232_23207


namespace NUMINAMATH_CALUDE_real_part_of_z_l232_23258

def complex_number_z : ℂ → Prop :=
  λ z ↦ z * Complex.I = 2 * Complex.I

theorem real_part_of_z (z : ℂ) (h : complex_number_z z) :
  z.re = 3/2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l232_23258


namespace NUMINAMATH_CALUDE_log_decreasing_implies_a_range_l232_23264

/-- A function f is decreasing on an interval [a, b] if for any x, y in [a, b] with x < y, f(x) > f(y) -/
def DecreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x > f y

/-- The logarithm function with base a -/
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_decreasing_implies_a_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  DecreasingOn (fun x => log a (5 - a * x)) 1 3 → 1 < a ∧ a < 5/3 := by
  sorry

end NUMINAMATH_CALUDE_log_decreasing_implies_a_range_l232_23264


namespace NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l232_23233

theorem polygon_sides_from_diagonals (d : ℕ) (h : d = 44) : ∃ n : ℕ, n ≥ 3 ∧ d = n * (n - 3) / 2 ∧ n = 11 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l232_23233


namespace NUMINAMATH_CALUDE_opposite_gold_is_black_l232_23263

-- Define the set of colors
inductive Color
  | Blue
  | Orange
  | Yellow
  | Black
  | Silver
  | Gold

-- Define a cube face
structure Face where
  color : Color

-- Define a cube
structure Cube where
  top : Face
  bottom : Face
  front : Face
  back : Face
  right : Face
  left : Face

-- Define the three views of the cube
def view1 (c : Cube) : Prop :=
  c.top.color = Color.Black ∧ c.front.color = Color.Blue ∧ c.right.color = Color.Orange

def view2 (c : Cube) : Prop :=
  c.top.color = Color.Black ∧ c.front.color = Color.Yellow ∧ c.right.color = Color.Orange

def view3 (c : Cube) : Prop :=
  c.top.color = Color.Black ∧ c.front.color = Color.Silver ∧ c.right.color = Color.Orange

-- Theorem statement
theorem opposite_gold_is_black (c : Cube) :
  view1 c → view2 c → view3 c → c.bottom.color = Color.Gold → c.top.color = Color.Black :=
by sorry

end NUMINAMATH_CALUDE_opposite_gold_is_black_l232_23263


namespace NUMINAMATH_CALUDE_six_digit_number_property_l232_23248

/-- Represents a six-digit number in the form 1ABCDE -/
def SixDigitNumber (a b c d e : Nat) : Nat :=
  100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem six_digit_number_property 
  (a b c d e : Nat) 
  (h1 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) 
  (h2 : SixDigitNumber a b c d e * 3 = SixDigitNumber b c d e a) : 
  a + b + c + d + e = 26 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_property_l232_23248


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l232_23290

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l232_23290


namespace NUMINAMATH_CALUDE_election_votes_theorem_l232_23288

theorem election_votes_theorem (total_votes : ℕ) (winner_votes second_votes third_votes : ℕ) :
  (winner_votes : ℚ) = 45 / 100 * total_votes ∧
  (second_votes : ℚ) = 35 / 100 * total_votes ∧
  winner_votes = second_votes + 150 ∧
  winner_votes + second_votes + third_votes = total_votes →
  total_votes = 1500 ∧ winner_votes = 675 ∧ second_votes = 525 ∧ third_votes = 300 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l232_23288


namespace NUMINAMATH_CALUDE_function_proof_l232_23260

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) := a * x^3 + b * x^2 - 3 * x

-- Define the derivative of f
def f' (a b : ℝ) (x : ℝ) := 3 * a * x^2 + 2 * b * x - 3

-- Define the function g
def g (a b : ℝ) (x : ℝ) := (1/3) * f a b x - 6 * Real.log x

-- Define the curve y = xf(x)
def curve (a b : ℝ) (x : ℝ) := x * (f a b x)

theorem function_proof (a b : ℝ) :
  (∀ x, f' a b x = f' a b (-x)) →  -- f' is even
  f' a b 1 = 0 →                   -- f'(1) = 0
  (∃ c, ∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
    |g a b x₁ - g a b x₂| ≤ c) →   -- |g(x₁) - g(x₂)| ≤ c for x₁, x₂ ∈ [1, 2]
  (∀ x, f a b x = x^3 - 3*x) ∧     -- f(x) = x³ - 3x
  (∃ c_min, c_min = -4/3 + 6 * Real.log 2 ∧ 
    ∀ c', (∀ x₁ x₂, x₁ ∈ Set.Icc 1 2 → x₂ ∈ Set.Icc 1 2 → 
      |g a b x₁ - g a b x₂| ≤ c') → c' ≥ c_min) ∧  -- Minimum value of c
  (∃ s : Set ℝ, s = {4, 3/4 - 4 * Real.sqrt 2} ∧ 
    ∀ m, m ∈ s ↔ (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      curve a b x₁ = m * x₁ - 2 * m ∧
      curve a b x₂ = m * x₂ - 2 * m ∧
      curve a b x₃ = m * x₃ - 2 * m)) -- Set of m values for three tangent lines
  := by sorry

end NUMINAMATH_CALUDE_function_proof_l232_23260


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l232_23230

theorem fixed_point_on_line (m : ℝ) : (2 * m + 1) * 3 + (m + 1) * 1 - 7 * m - 4 = 0 := by
  sorry

#check fixed_point_on_line

end NUMINAMATH_CALUDE_fixed_point_on_line_l232_23230


namespace NUMINAMATH_CALUDE_inequality_solution_condition_l232_23293

theorem inequality_solution_condition (a : ℝ) : 
  (∃! x y : ℤ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ 
    (∀ z : ℤ, z < 0 → ((z + a) / 2 ≥ 1) ↔ (z = x ∨ z = y))) 
  → 4 ≤ a ∧ a < 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_condition_l232_23293


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l232_23209

def f (x : ℕ) : ℕ := 3 * x + 2

theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ, (3^100 * m + (3^100 - 1)) % 1988 = 0 := by
sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l232_23209


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l232_23204

theorem arithmetic_calculations :
  ((-6) - 3 + (-7) - (-2) = -14) ∧
  ((-1)^2023 + 5 * (-2) - 12 / (-4) = -8) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l232_23204


namespace NUMINAMATH_CALUDE_parallel_tangents_ordinates_l232_23236

/-- The curve function y = x³ - 3x² + 6x + 2 -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 6

theorem parallel_tangents_ordinates (P Q : ℝ × ℝ) :
  P.2 = f P.1 →
  Q.2 = f Q.1 →
  f' P.1 = f' Q.1 →
  P.2 = 1 →
  Q.2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_ordinates_l232_23236


namespace NUMINAMATH_CALUDE_fourth_roll_five_probability_l232_23225

-- Define the probabilities for each die
def fair_die_prob : ℚ := 1 / 6
def biased_die_five_prob : ℚ := 1 / 2
def biased_die_other_prob : ℚ := 1 / 10

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define the number of rolls
def num_rolls : ℕ := 4

-- Define the probability of choosing each die
def choose_prob : ℚ := 1 / 2

-- Theorem statement
theorem fourth_roll_five_probability :
  let p_fair := fair_die_prob ^ 3 * choose_prob
  let p_biased := biased_die_five_prob ^ 3 * choose_prob
  let p_fair_given_three_fives := p_fair / (p_fair + p_biased)
  let p_biased_given_three_fives := p_biased / (p_fair + p_biased)
  p_fair_given_three_fives * fair_die_prob + p_biased_given_three_fives * biased_die_five_prob = 41 / 84 :=
by sorry

end NUMINAMATH_CALUDE_fourth_roll_five_probability_l232_23225


namespace NUMINAMATH_CALUDE_faulty_engine_sampling_l232_23202

/-- Given a set of 33 items where 8 are faulty, this theorem proves:
    1. The probability of identifying all faulty items by sampling 32 items
    2. The expected number of samplings required to identify all faulty items -/
theorem faulty_engine_sampling (n : Nat) (k : Nat) (h1 : n = 33) (h2 : k = 8) :
  let p := Nat.choose (n - 1) (k - 1) / Nat.choose n k
  let e := (n * k) / (n - k + 1)
  (p = 25 / 132) ∧ (e = 272 / 9) := by
  sorry

#check faulty_engine_sampling

end NUMINAMATH_CALUDE_faulty_engine_sampling_l232_23202


namespace NUMINAMATH_CALUDE_two_roots_k_range_l232_23210

theorem two_roots_k_range (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, f x = x * Real.exp (-2 * x) + k) →
  (∃! x₁ x₂, x₁ ∈ Set.Ioo (-2 : ℝ) 2 ∧ x₂ ∈ Set.Ioo (-2 : ℝ) 2 ∧ x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) →
  k ∈ Set.Ioo (-(1 / (2 * Real.exp 1))) (-(2 / Real.exp 4)) :=
by sorry

end NUMINAMATH_CALUDE_two_roots_k_range_l232_23210


namespace NUMINAMATH_CALUDE_hiker_total_distance_l232_23282

-- Define the hiker's walking parameters
def day1_distance : ℕ := 18
def day1_speed : ℕ := 3
def day2_speed_increase : ℕ := 1
def day3_speed : ℕ := 5
def day3_hours : ℕ := 6

-- Theorem to prove
theorem hiker_total_distance :
  let day1_hours : ℕ := day1_distance / day1_speed
  let day2_hours : ℕ := day1_hours - 1
  let day2_speed : ℕ := day1_speed + day2_speed_increase
  let day2_distance : ℕ := day2_speed * day2_hours
  let day3_distance : ℕ := day3_speed * day3_hours
  day1_distance + day2_distance + day3_distance = 68 := by
  sorry


end NUMINAMATH_CALUDE_hiker_total_distance_l232_23282


namespace NUMINAMATH_CALUDE_sum_odd_integers_less_than_100_l232_23216

theorem sum_odd_integers_less_than_100 : 
  (Finset.filter (fun n => n % 2 = 1) (Finset.range 100)).sum id = 2500 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_less_than_100_l232_23216


namespace NUMINAMATH_CALUDE_range_of_f_l232_23231

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l232_23231


namespace NUMINAMATH_CALUDE_smallest_result_l232_23297

def S : Finset ℕ := {4, 5, 7, 11, 13, 17}

theorem smallest_result (a b c : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (x y z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (x + y) * z = 48 ∧ (a + b) * c ≥ 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_result_l232_23297


namespace NUMINAMATH_CALUDE_jacoby_lottery_ticket_cost_l232_23245

def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_winnings : ℕ := 500
def sister_gift : ℕ := 500
def sisters_count : ℕ := 2
def remaining_needed : ℕ := 3214

theorem jacoby_lottery_ticket_cost :
  let job_earnings := hourly_wage * hours_worked
  let cookie_earnings := cookie_price * cookies_sold
  let gifts := sister_gift * sisters_count
  let total_earned := job_earnings + cookie_earnings + lottery_winnings + gifts
  let actual_total := trip_cost - remaining_needed
  total_earned - actual_total = 10
  := by sorry

end NUMINAMATH_CALUDE_jacoby_lottery_ticket_cost_l232_23245


namespace NUMINAMATH_CALUDE_ratio_sum_equality_l232_23266

theorem ratio_sum_equality (a b c x y z : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h_abc : a^2 + b^2 + c^2 = 49)
  (h_xyz : x^2 + y^2 + z^2 = 64)
  (h_dot : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_equality_l232_23266


namespace NUMINAMATH_CALUDE_cubic_discriminant_l232_23224

theorem cubic_discriminant (p q : ℝ) (x₁ x₂ x₃ : ℝ) : 
  x₁^3 + p*x₁ + q = 0 → 
  x₂^3 + p*x₂ + q = 0 → 
  x₃^3 + p*x₃ + q = 0 → 
  (x₁ - x₂)^2 * (x₂ - x₃)^2 * (x₃ - x₁)^2 = -4*p^3 - 27*q^2 := by
sorry

end NUMINAMATH_CALUDE_cubic_discriminant_l232_23224


namespace NUMINAMATH_CALUDE_exist_three_quadratic_polynomials_l232_23298

/-- A quadratic polynomial with real coefficients -/
def QuadraticPolynomial := ℝ → ℝ

/-- A quadratic polynomial has two distinct real roots -/
def HasTwoDistinctRealRoots (p : QuadraticPolynomial) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ p r₁ = 0 ∧ p r₂ = 0

/-- A quadratic polynomial has no real roots -/
def HasNoRealRoots (p : QuadraticPolynomial) : Prop :=
  ∀ x : ℝ, p x ≠ 0

/-- Theorem: There exist three quadratic polynomials satisfying the given conditions -/
theorem exist_three_quadratic_polynomials :
  ∃ (P₁ P₂ P₃ : QuadraticPolynomial),
    HasTwoDistinctRealRoots P₁ ∧
    HasTwoDistinctRealRoots P₂ ∧
    HasTwoDistinctRealRoots P₃ ∧
    HasNoRealRoots (λ x => P₁ x + P₂ x) ∧
    HasNoRealRoots (λ x => P₁ x + P₃ x) ∧
    HasNoRealRoots (λ x => P₂ x + P₃ x) := by
  sorry

end NUMINAMATH_CALUDE_exist_three_quadratic_polynomials_l232_23298


namespace NUMINAMATH_CALUDE_unique_function_satisfying_condition_l232_23255

open Function Real

theorem unique_function_satisfying_condition :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2 ∧ f = id := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_condition_l232_23255


namespace NUMINAMATH_CALUDE_temperature_difference_l232_23214

theorem temperature_difference (low high : ℤ) (h1 : low = -2) (h2 : high = 5) :
  high - low = 7 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l232_23214


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l232_23206

theorem fraction_zero_implies_x_negative_three (x : ℝ) :
  (x^2 - 9) / (x - 3) = 0 ∧ x - 3 ≠ 0 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_three_l232_23206


namespace NUMINAMATH_CALUDE_caleb_gallons_per_trip_l232_23259

/-- Prove that Caleb adds 7 gallons per trip to fill a pool --/
theorem caleb_gallons_per_trip 
  (pool_capacity : ℕ) 
  (cynthia_gallons : ℕ) 
  (total_trips : ℕ) 
  (h1 : pool_capacity = 105)
  (h2 : cynthia_gallons = 8)
  (h3 : total_trips = 7)
  : ∃ (caleb_gallons : ℕ), 
    caleb_gallons * total_trips + cynthia_gallons * total_trips = pool_capacity ∧ 
    caleb_gallons = 7 := by
  sorry

end NUMINAMATH_CALUDE_caleb_gallons_per_trip_l232_23259


namespace NUMINAMATH_CALUDE_stamps_collection_theorem_l232_23287

def kylie_stamps : ℕ := 34
def nelly_stamps_difference : ℕ := 44

def total_stamps : ℕ := kylie_stamps + (kylie_stamps + nelly_stamps_difference)

theorem stamps_collection_theorem : total_stamps = 112 := by
  sorry

end NUMINAMATH_CALUDE_stamps_collection_theorem_l232_23287


namespace NUMINAMATH_CALUDE_gcd_of_sum_and_reversed_l232_23221

def arithmetic_sequence (a k : ℕ) : Fin 4 → ℕ
  | 0 => a
  | 1 => a + k
  | 2 => a + 2*k
  | 3 => a + 3*k

def four_digit_number (digits : Fin 4 → ℕ) : ℕ :=
  1000 * (digits 0) + 100 * (digits 1) + 10 * (digits 2) + (digits 3)

def reversed_four_digit_number (digits : Fin 4 → ℕ) : ℕ :=
  1000 * (digits 3) + 100 * (digits 2) + 10 * (digits 1) + (digits 0)

theorem gcd_of_sum_and_reversed (a k : ℕ) :
  ∀ (n : ℕ), n = four_digit_number (arithmetic_sequence a k) + 
              reversed_four_digit_number (arithmetic_sequence a k) + k →
  Nat.gcd n 2 = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_sum_and_reversed_l232_23221


namespace NUMINAMATH_CALUDE_product_binary1011_ternary212_eq_253_l232_23262

/-- Converts a list of digits in a given base to its decimal representation -/
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

/-- The binary representation of 1011 -/
def binary1011 : List Nat := [1, 0, 1, 1]

/-- The base-3 representation of 212 -/
def ternary212 : List Nat := [2, 1, 2]

theorem product_binary1011_ternary212_eq_253 :
  (toDecimal binary1011 2) * (toDecimal ternary212 3) = 253 := by
  sorry

end NUMINAMATH_CALUDE_product_binary1011_ternary212_eq_253_l232_23262


namespace NUMINAMATH_CALUDE_distance_product_zero_l232_23237

-- Define the curve C
def C : Set (ℝ × ℝ) := {(x, y) | x^2 / 36 + y^2 / 16 = 1}

-- Define the line l
def l : Set (ℝ × ℝ) := {(x, y) | ∃ t : ℝ, x = 1 - t/2 ∧ y = 1 + (Real.sqrt 3 * t)/2}

-- Define point P
def P : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem distance_product_zero (A B : ℝ × ℝ) 
  (hA : A ∈ C ∩ l) (hB : B ∈ C ∩ l) (hAB : A ≠ B) :
  ‖P‖ * ‖P - B‖ = 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_product_zero_l232_23237


namespace NUMINAMATH_CALUDE_stream_speed_l232_23269

/-- Given a boat traveling downstream, this theorem proves the speed of the stream. -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 24 →
  downstream_distance = 84 →
  downstream_time = 3 →
  ∃ stream_speed : ℝ, stream_speed = 4 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l232_23269


namespace NUMINAMATH_CALUDE_at_most_one_point_inside_plane_l232_23212

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Checks if a point is on a line -/
def isPointOnLine (p : Point3D) (l : Line3D) : Prop := sorry

/-- Checks if a point is on a plane -/
def isPointOnPlane (p : Point3D) (pl : Plane3D) : Prop := sorry

/-- Checks if a point is outside a plane -/
def isPointOutsidePlane (p : Point3D) (pl : Plane3D) : Prop := 
  ¬(isPointOnPlane p pl)

/-- The main theorem -/
theorem at_most_one_point_inside_plane 
  (l : Line3D) (pl : Plane3D) 
  (p1 p2 : Point3D) 
  (h1 : isPointOnLine p1 l) 
  (h2 : isPointOnLine p2 l) 
  (h3 : isPointOutsidePlane p1 pl) 
  (h4 : isPointOutsidePlane p2 pl) : 
  ∃! p, isPointOnLine p l ∧ isPointOnPlane p pl :=
sorry

end NUMINAMATH_CALUDE_at_most_one_point_inside_plane_l232_23212


namespace NUMINAMATH_CALUDE_smallest_valid_number_l232_23203

def starts_with_19 (n : ℕ) : Prop :=
  ∃ k : ℕ, n ≥ 19 * 10^k ∧ n < 20 * 10^k

def ends_with_89 (n : ℕ) : Prop :=
  n % 100 = 89

def is_valid_number (n : ℕ) : Prop :=
  starts_with_19 (n^2) ∧ ends_with_89 (n^2)

theorem smallest_valid_number :
  is_valid_number 1383 ∧ ∀ m : ℕ, m < 1383 → ¬(is_valid_number m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l232_23203


namespace NUMINAMATH_CALUDE_minimize_reciprocal_sum_l232_23270

theorem minimize_reciprocal_sum (a b : ℕ+) (h : 4 * a.val + b.val = 30) :
  (1 : ℚ) / a.val + (1 : ℚ) / b.val ≥ (1 : ℚ) / 5 + (1 : ℚ) / 10 :=
sorry

end NUMINAMATH_CALUDE_minimize_reciprocal_sum_l232_23270


namespace NUMINAMATH_CALUDE_equation_roots_l232_23285

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => (x - 3)^2 - (x - 3)
  (f 3 = 0 ∧ f 4 = 0) ∧ ∀ x : ℝ, f x = 0 → (x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l232_23285


namespace NUMINAMATH_CALUDE_cubic_root_sum_l232_23275

theorem cubic_root_sum (a b c : ℝ) : 
  (a^3 - 2*a^2 - a + 2 = 0) →
  (b^3 - 2*b^2 - b + 2 = 0) →
  (c^3 - 2*c^2 - c + 2 = 0) →
  a*(b-c)^2 + b*(c-a)^2 + c*(a-b)^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l232_23275


namespace NUMINAMATH_CALUDE_factorization_2x_cubed_minus_8x_l232_23208

theorem factorization_2x_cubed_minus_8x (x : ℝ) : 
  2 * x^3 - 8 * x = 2 * x * (x - 2) * (x + 2) := by
sorry


end NUMINAMATH_CALUDE_factorization_2x_cubed_minus_8x_l232_23208


namespace NUMINAMATH_CALUDE_largest_n_for_divisibility_l232_23246

theorem largest_n_for_divisibility (n p q : ℕ+) : 
  (n.val ^ 3 + p.val) % (n.val + q.val) = 0 → 
  n.val ≤ 3060 ∧ 
  (n.val = 3060 → p.val = 300 ∧ q.val = 15) :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_divisibility_l232_23246


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l232_23241

/-- The cost per page for the first typing of a manuscript -/
def first_typing_cost (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) 
  (revision_cost : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost - revision_cost * (pages_revised_once + 2 * pages_revised_twice)) / total_pages

theorem manuscript_typing_cost :
  first_typing_cost 200 80 20 3 1360 = 5 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_typing_cost_l232_23241


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l232_23257

theorem quadratic_solution_product (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (2*a - 5) * (3*b - 4) = 47 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l232_23257


namespace NUMINAMATH_CALUDE_choir_members_l232_23218

theorem choir_members (n : ℕ) : 
  50 ≤ n ∧ n ≤ 150 ∧ 
  n % 6 = 4 ∧ 
  n % 10 = 4 → 
  n = 64 ∨ n = 94 ∨ n = 124 := by
sorry

end NUMINAMATH_CALUDE_choir_members_l232_23218


namespace NUMINAMATH_CALUDE_halloween_candy_duration_l232_23244

/-- Calculates the number of full days candy will last given initial amounts, trades, losses, and daily consumption. -/
def candy_duration (neighbors : ℕ) (sister : ℕ) (traded : ℕ) (lost : ℕ) (daily_consumption : ℕ) : ℕ :=
  ((neighbors + sister - traded - lost) / daily_consumption : ℕ)

/-- Theorem stating that under the given conditions, the candy will last for 23 full days. -/
theorem halloween_candy_duration :
  candy_duration 75 130 25 15 7 = 23 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_duration_l232_23244


namespace NUMINAMATH_CALUDE_odd_prime_sum_of_squares_l232_23240

theorem odd_prime_sum_of_squares (p : ℕ) (hp : Nat.Prime p) (hodd : Odd p) :
  (∃ (a b : ℕ+), a.val^2 + b.val^2 = p) ↔ p % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_sum_of_squares_l232_23240


namespace NUMINAMATH_CALUDE_car_distance_proof_l232_23267

/-- Proves that a car traveling at 162 km/h for 5 hours covers a distance of 810 km -/
theorem car_distance_proof (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 162 → time = 5 → distance = speed * time → distance = 810 := by
sorry

end NUMINAMATH_CALUDE_car_distance_proof_l232_23267


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l232_23296

theorem partial_fraction_decomposition (N₁ N₂ : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (60 * x - 46) / (x^2 - 5*x + 6) = N₁ / (x - 2) + N₂ / (x - 3)) →
  N₁ * N₂ = -1036 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l232_23296


namespace NUMINAMATH_CALUDE_total_cars_theorem_l232_23228

/-- Calculates the total number of cars at the end of the play -/
def total_cars_at_end (front_cars : ℕ) (back_multiplier : ℕ) (additional_cars : ℕ) : ℕ :=
  front_cars + (back_multiplier * front_cars) + additional_cars

/-- Theorem: Given the initial conditions, the total number of cars at the end of the play is 600 -/
theorem total_cars_theorem : total_cars_at_end 100 2 300 = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_theorem_l232_23228


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l232_23261

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i / (2 + i)) = ((1 : ℂ) + 2 * i) / 5 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l232_23261


namespace NUMINAMATH_CALUDE_march_greatest_drop_l232_23213

/-- Represents the months from January to August --/
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August

/-- Price change for each month --/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January => -1.00
  | Month.February => 1.50
  | Month.March => -3.00
  | Month.April => 2.00
  | Month.May => -0.75
  | Month.June => 1.00
  | Month.July => -2.50
  | Month.August => -2.00

/-- Definition of a price drop --/
def is_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- Theorem: March has the greatest monthly drop in price --/
theorem march_greatest_drop :
  ∀ m : Month, is_price_drop m → price_change Month.March ≤ price_change m :=
by sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l232_23213


namespace NUMINAMATH_CALUDE_probability_is_half_l232_23277

/-- An equilateral triangle divided by two medians -/
structure TriangleWithMedians where
  /-- The number of regions formed by drawing two medians in an equilateral triangle -/
  total_regions : ℕ
  /-- The number of shaded regions -/
  shaded_regions : ℕ
  /-- The total number of regions is 6 -/
  h_total : total_regions = 6
  /-- The number of shaded regions is 3 -/
  h_shaded : shaded_regions = 3

/-- The probability of a point landing in a shaded region -/
def probability (t : TriangleWithMedians) : ℚ :=
  t.shaded_regions / t.total_regions

theorem probability_is_half (t : TriangleWithMedians) :
  probability t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l232_23277


namespace NUMINAMATH_CALUDE_collinear_points_on_cubic_curve_l232_23243

/-- Three points on a cubic curve that are collinear satisfy a specific relation -/
theorem collinear_points_on_cubic_curve
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h_curve₁ : y₁^2 = x₁^3)
  (h_curve₂ : y₂^2 = x₂^3)
  (h_curve₃ : y₃^2 = x₃^3)
  (h_collinear : (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁))
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁)
  (h_nonzero : y₁ ≠ 0 ∧ y₂ ≠ 0 ∧ y₃ ≠ 0) :
  x₁ / y₁ + x₂ / y₂ + x₃ / y₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_collinear_points_on_cubic_curve_l232_23243


namespace NUMINAMATH_CALUDE_max_d_is_two_l232_23284

def a (n : ℕ) : ℕ := 101 + (n + 1)^2 + 3*n

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_is_two : 
  (∃ (n : ℕ), d n = 2) ∧ (∀ (n : ℕ), d n ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_max_d_is_two_l232_23284


namespace NUMINAMATH_CALUDE_difference_is_nine_l232_23220

theorem difference_is_nine (a b c d : ℝ) 
  (h1 : ∃ x, a - b = c + d + x)
  (h2 : a + b = c - d - 3)
  (h3 : a - c = 3) :
  (a - b) - (c + d) = 9 := by
  sorry

end NUMINAMATH_CALUDE_difference_is_nine_l232_23220


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l232_23200

theorem two_digit_reverse_sum (x y n : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ x = 10 * a + b ∧ y = 10 * b + a) →  -- y is obtained by reversing the digits of x
  x^2 + y^2 = n^2 →  -- x^2 + y^2 = n^2
  x + y + n = 132 := by
sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l232_23200


namespace NUMINAMATH_CALUDE_m_returns_to_original_position_min_steps_to_return_l232_23232

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the position of point M on side AB
def PositionM (t : Triangle) (a : ℝ) : ℝ × ℝ :=
  (a * t.A.1 + (1 - a) * t.B.1, a * t.A.2 + (1 - a) * t.B.2)

-- Define the movement of point M
def MoveM (t : Triangle) (pos : ℝ × ℝ) (step : ℕ) : ℝ × ℝ :=
  sorry

-- Theorem: M returns to its original position
theorem m_returns_to_original_position (t : Triangle) (a : ℝ) :
  ∃ n : ℕ, MoveM t (PositionM t a) n = PositionM t a :=
sorry

-- Theorem: Minimum number of steps for M to return
theorem min_steps_to_return (t : Triangle) (a : ℝ) :
  (a = 1/2 ∧ (∃ n : ℕ, n = 3 ∧ MoveM t (PositionM t a) n = PositionM t a ∧
    ∀ m : ℕ, m < n → MoveM t (PositionM t a) m ≠ PositionM t a)) ∨
  (a ≠ 1/2 ∧ (∃ n : ℕ, n = 6 ∧ MoveM t (PositionM t a) n = PositionM t a ∧
    ∀ m : ℕ, m < n → MoveM t (PositionM t a) m ≠ PositionM t a)) :=
sorry

end NUMINAMATH_CALUDE_m_returns_to_original_position_min_steps_to_return_l232_23232


namespace NUMINAMATH_CALUDE_trig_identity_l232_23283

theorem trig_identity : 
  (2 * Real.sin (46 * π / 180) - Real.sqrt 3 * Real.cos (74 * π / 180)) / Real.cos (16 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l232_23283


namespace NUMINAMATH_CALUDE_condition_relationship_l232_23234

theorem condition_relationship :
  (∀ x : ℝ, (0 < x ∧ x < 5) → (|x - 2| < 3)) ∧
  (∃ x : ℝ, (|x - 2| < 3) ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l232_23234


namespace NUMINAMATH_CALUDE_complex_root_magnitude_l232_23273

theorem complex_root_magnitude (n : ℕ) (a : ℝ) (z : ℂ) 
  (h1 : n ≥ 2) 
  (h2 : 0 < a) 
  (h3 : a < (n + 1 : ℝ) / (n - 1 : ℝ)) 
  (h4 : z^(n+1) - a * z^n + a * z - 1 = 0) : 
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_root_magnitude_l232_23273


namespace NUMINAMATH_CALUDE_different_elements_same_image_single_element_unique_image_elements_without_preimage_l232_23223

-- Define the mapping f from A to B
variable {A B : Type}
variable (f : A → B)

-- Statement 1: Different elements in A can have the same image in B
theorem different_elements_same_image :
  ∃ (x y : A), x ≠ y ∧ f x = f y :=
sorry

-- Statement 2: A single element in A cannot have different images in B
theorem single_element_unique_image :
  ∀ (x : A) (y z : B), f x = y ∧ f x = z → y = z :=
sorry

-- Statement 3: There can be elements in B that do not have a pre-image in A
theorem elements_without_preimage :
  ∃ (y : B), ∀ (x : A), f x ≠ y :=
sorry

end NUMINAMATH_CALUDE_different_elements_same_image_single_element_unique_image_elements_without_preimage_l232_23223


namespace NUMINAMATH_CALUDE_max_profit_difference_l232_23249

def total_records : ℕ := 300

def sammy_offer : ℕ → ℚ := λ n => 4 * n

def bryan_offer : ℕ → ℚ := λ n => 6 * (2/3 * n) + 1 * (1/3 * n)

def christine_offer : ℕ → ℚ := λ n => 10 * 30 + 3 * (n - 30)

theorem max_profit_difference (n : ℕ) (h : n = total_records) : 
  max (abs (sammy_offer n - bryan_offer n))
      (max (abs (sammy_offer n - christine_offer n))
           (abs (bryan_offer n - christine_offer n)))
  = 190 :=
sorry

end NUMINAMATH_CALUDE_max_profit_difference_l232_23249


namespace NUMINAMATH_CALUDE_parabola_directrix_l232_23226

/-- Given a parabola with equation x² = 4y, its directrix has equation y = -1 -/
theorem parabola_directrix (x y : ℝ) : x^2 = 4*y → (∃ (k : ℝ), k = -1 ∧ y = k) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l232_23226


namespace NUMINAMATH_CALUDE_original_denominator_proof_l232_23281

theorem original_denominator_proof (d : ℤ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = 2 / 5 →
  d = 18 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l232_23281


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l232_23252

theorem units_digit_of_quotient (n : ℕ) : 
  (4^1993 + 5^1993) % 3 = 0 ∧ ((4^1993 + 5^1993) / 3) % 10 = 3 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l232_23252


namespace NUMINAMATH_CALUDE_percentage_of_temporary_workers_l232_23205

theorem percentage_of_temporary_workers
  (total_workers : ℕ)
  (technician_ratio : ℚ)
  (non_technician_ratio : ℚ)
  (permanent_technician_ratio : ℚ)
  (permanent_non_technician_ratio : ℚ)
  (h1 : technician_ratio = 9/10)
  (h2 : non_technician_ratio = 1/10)
  (h3 : permanent_technician_ratio = 9/10)
  (h4 : permanent_non_technician_ratio = 1/10)
  (h5 : technician_ratio + non_technician_ratio = 1) :
  let permanent_workers := (technician_ratio * permanent_technician_ratio +
                            non_technician_ratio * permanent_non_technician_ratio) * total_workers
  let temporary_workers := total_workers - permanent_workers
  (temporary_workers : ℚ) / (total_workers : ℚ) = 18/100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_temporary_workers_l232_23205


namespace NUMINAMATH_CALUDE_circle_area_l232_23242

theorem circle_area (r : ℝ) (h : 2 * π * r = 18 * π) : π * r^2 = 81 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_l232_23242


namespace NUMINAMATH_CALUDE_range_of_m_l232_23250

/-- For the equation m/(x-2) = 3 with positive solutions for x, 
    the range of m is {m ∈ ℝ | m > -6 and m ≠ 0} -/
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ m / (x - 2) = 3) ↔ m > -6 ∧ m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l232_23250


namespace NUMINAMATH_CALUDE_det_A_l232_23254

def A : Matrix (Fin 3) (Fin 3) ℤ := !![3, 1, -2; 8, 5, -4; 3, 3, 7]

def A' : Matrix (Fin 3) (Fin 3) ℤ := 
  Matrix.of (λ i j => 
    if i = 0 then A i j
    else A i j - A 0 j)

theorem det_A'_eq_55 : Matrix.det A' = 55 := by sorry

end NUMINAMATH_CALUDE_det_A_l232_23254


namespace NUMINAMATH_CALUDE_triangle_theorem_l232_23217

-- Define the triangle ABC
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) :
  Triangle a b c →
  b^2 * c * Real.cos C + c^2 * b * Real.cos B = a * b^2 + a * c^2 - a^3 →
  (A = Real.pi / 3 ∧
   (b + c = 2 → ∀ a' : ℝ, Triangle a' b c → a' ≥ 1)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l232_23217


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l232_23292

/-- A quadratic function f(x) = ax^2 + 2bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_order : c > b ∧ b > a

/-- The graph of f passes through (1, 0) -/
def passes_through_one_zero (f : QuadraticFunction) : Prop :=
  f.a + 2 * f.b + f.c = 0

/-- The graph of f intersects with y = -a -/
def intersects_neg_a (f : QuadraticFunction) : Prop :=
  ∃ x : ℝ, f.a * x^2 + 2 * f.b * x + f.c = -f.a

/-- The ratio b/a is in [0, 1) -/
def ratio_in_range (f : QuadraticFunction) : Prop :=
  0 ≤ f.b / f.a ∧ f.b / f.a < 1

/-- Line segments AB, BC, CD form an obtuse triangle -/
def forms_obtuse_triangle (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  AB + CD > BC ∧ AB^2 + CD^2 < BC^2

/-- The ratio b/a is in the specified range -/
def ratio_in_specific_range (f : QuadraticFunction) : Prop :=
  -1 + 4/21 < f.b / f.a ∧ f.b / f.a < -1 + Real.sqrt 15 / 3

theorem quadratic_function_properties (f : QuadraticFunction)
    (h_pass : passes_through_one_zero f)
    (h_intersect : intersects_neg_a f) :
  ratio_in_range f ∧
  (∀ A B C D : ℝ × ℝ, forms_obtuse_triangle A B C D →
    ratio_in_specific_range f) :=
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l232_23292


namespace NUMINAMATH_CALUDE_permutation_identities_l232_23295

def A (n m : ℕ) : ℕ := (n :: List.range m).prod

theorem permutation_identities :
  (∀ n m : ℕ, A (n + 1) (m + 1) - A n m = n^2 * A (n - 1) (m - 1)) ∧
  (∀ n m : ℕ, A n m = n * A (n - 1) (m - 1)) := by
  sorry

end NUMINAMATH_CALUDE_permutation_identities_l232_23295


namespace NUMINAMATH_CALUDE_investment_time_period_l232_23274

/-- 
Given:
- principal: The sum invested (in rupees)
- rate_difference: The difference in interest rates (as a decimal)
- interest_difference: The additional interest earned due to the higher rate (in rupees)

Proves that the time period for which the sum is invested is 2 years.
-/
theorem investment_time_period 
  (principal : ℝ) 
  (rate_difference : ℝ) 
  (interest_difference : ℝ) 
  (h1 : principal = 14000)
  (h2 : rate_difference = 0.03)
  (h3 : interest_difference = 840) :
  principal * rate_difference * 2 = interest_difference := by
  sorry

end NUMINAMATH_CALUDE_investment_time_period_l232_23274


namespace NUMINAMATH_CALUDE_largest_common_term_l232_23276

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def is_common_term (x : ℤ) (a₁ d₁ a₂ d₂ : ℤ) : Prop :=
  ∃ n m : ℕ, arithmetic_sequence a₁ d₁ n = x ∧ arithmetic_sequence a₂ d₂ m = x

theorem largest_common_term :
  ∃ x : ℤ, x ≤ 150 ∧ is_common_term x 1 8 5 9 ∧
    ∀ y : ℤ, y ≤ 150 → is_common_term y 1 8 5 9 → y ≤ x :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l232_23276


namespace NUMINAMATH_CALUDE_sum_of_roots_l232_23239

theorem sum_of_roots (k : ℝ) (a₁ a₂ : ℝ) (h₁ : a₁ ≠ a₂) 
  (h₂ : 5 * a₁^2 + k * a₁ - 2 = 0) (h₃ : 5 * a₂^2 + k * a₂ - 2 = 0) : 
  a₁ + a₂ = -k / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l232_23239


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l232_23253

/-- Given two squares ABCD and DEFG where CE = 14 and AG = 2, prove that the sum of their areas is 100 -/
theorem sum_of_square_areas (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 14 → a - b = 2 → a^2 + b^2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l232_23253


namespace NUMINAMATH_CALUDE_inequality_relation_l232_23235

theorem inequality_relation (a b : ℝ) : 
  ¬(∀ a b : ℝ, a > b → 1/a < 1/b) ∧ ¬(∀ a b : ℝ, 1/a < 1/b → a > b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relation_l232_23235


namespace NUMINAMATH_CALUDE_find_divisor_l232_23286

theorem find_divisor (n : ℕ) (added : ℕ) (divisor : ℕ) : 
  (n + added) % divisor = 0 ∧ 
  (∀ m : ℕ, m < added → (n + m) % divisor ≠ 0) →
  divisor = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l232_23286


namespace NUMINAMATH_CALUDE_center_of_mass_theorem_l232_23238

/-- Represents the center of mass coordinates for an n × n chessboard -/
structure CenterOfMass where
  x : ℚ
  y : ℚ

/-- Calculates the center of mass for the sum-1 rule -/
def centerOfMassSumRule (n : ℕ) : CenterOfMass :=
  { x := ((n + 1) * (7 * n - 1)) / (12 * n),
    y := ((n + 1) * (7 * n - 1)) / (12 * n) }

/-- Calculates the center of mass for the product rule -/
def centerOfMassProductRule (n : ℕ) : CenterOfMass :=
  { x := (2 * n + 1) / 3,
    y := (2 * n + 1) / 3 }

/-- Theorem stating the correctness of the center of mass calculations -/
theorem center_of_mass_theorem (n : ℕ) :
  (centerOfMassSumRule n).x = ((n + 1) * (7 * n - 1)) / (12 * n) ∧
  (centerOfMassSumRule n).y = ((n + 1) * (7 * n - 1)) / (12 * n) ∧
  (centerOfMassProductRule n).x = (2 * n + 1) / 3 ∧
  (centerOfMassProductRule n).y = (2 * n + 1) / 3 := by
  sorry

end NUMINAMATH_CALUDE_center_of_mass_theorem_l232_23238


namespace NUMINAMATH_CALUDE_range_of_a_l232_23280

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 4*x else 4*x - x^2

theorem range_of_a (a : ℝ) (h : f (a - 2) + f a > 0) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l232_23280


namespace NUMINAMATH_CALUDE_product_of_place_values_l232_23215

/-- The place value of a digit in a decimal number -/
def placeValue (digit : ℕ) (position : ℤ) : ℚ :=
  (digit : ℚ) * (10 : ℚ) ^ position

/-- The numeral under consideration -/
def numeral : ℚ := 7804830.88

/-- The product of place values of the three 8's in the numeral -/
def productOfPlaceValues : ℚ :=
  placeValue 8 5 * placeValue 8 1 * placeValue 8 (-2)

theorem product_of_place_values :
  productOfPlaceValues = 5120000 := by sorry

end NUMINAMATH_CALUDE_product_of_place_values_l232_23215


namespace NUMINAMATH_CALUDE_largest_among_rationals_l232_23256

theorem largest_among_rationals : 
  let numbers : List ℚ := [-2/3, -2, -1, -5]
  (∀ x ∈ numbers, x ≤ -2/3) ∧ (-2/3 ∈ numbers) := by
  sorry

end NUMINAMATH_CALUDE_largest_among_rationals_l232_23256


namespace NUMINAMATH_CALUDE_mikaela_personal_needs_fraction_l232_23227

/-- Calculates the fraction of total earnings spent on personal needs --/
def fraction_spent_on_personal_needs (hourly_rate : ℚ) (first_month_hours : ℕ) (second_month_additional_hours : ℕ) (amount_saved : ℚ) : ℚ :=
  let first_month_earnings := hourly_rate * first_month_hours
  let second_month_hours := first_month_hours + second_month_additional_hours
  let second_month_earnings := hourly_rate * second_month_hours
  let total_earnings := first_month_earnings + second_month_earnings
  let amount_spent := total_earnings - amount_saved
  amount_spent / total_earnings

/-- Proves that Mikaela spent 4/5 of her total earnings on personal needs --/
theorem mikaela_personal_needs_fraction :
  fraction_spent_on_personal_needs 10 35 5 150 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_mikaela_personal_needs_fraction_l232_23227


namespace NUMINAMATH_CALUDE_megans_earnings_l232_23251

/-- Calculates the total earnings for a given work schedule and hourly rate -/
def total_earnings (hours_per_day : ℕ) (hourly_rate : ℚ) (days_per_month : ℕ) (months : ℕ) : ℚ :=
  hours_per_day * hourly_rate * days_per_month * months

/-- Proves that Megan's total earnings for two months equal $2400 -/
theorem megans_earnings :
  let hours_per_day : ℕ := 8
  let hourly_rate : ℚ := 15/2
  let days_per_month : ℕ := 20
  let months : ℕ := 2
  total_earnings hours_per_day hourly_rate days_per_month months = 2400 := by
  sorry

#eval total_earnings 8 (15/2) 20 2

end NUMINAMATH_CALUDE_megans_earnings_l232_23251


namespace NUMINAMATH_CALUDE_max_value_trigonometric_expression_l232_23289

theorem max_value_trigonometric_expression (θ : Real) 
  (h : 0 < θ ∧ θ < π / 2) : 
  ∃ (max : Real), max = 4 * Real.sqrt 2 ∧ 
    ∀ φ, 0 < φ ∧ φ < π / 2 → 
      3 * Real.sin φ + 2 * Real.cos φ + 1 / Real.cos φ ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_trigonometric_expression_l232_23289


namespace NUMINAMATH_CALUDE_binomial_12_10_l232_23211

theorem binomial_12_10 : Nat.choose 12 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_10_l232_23211


namespace NUMINAMATH_CALUDE_kenya_peanuts_l232_23265

theorem kenya_peanuts (jose_peanuts : ℕ) (kenya_more : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_more = 48) : 
  jose_peanuts + kenya_more = 133 := by
sorry

end NUMINAMATH_CALUDE_kenya_peanuts_l232_23265


namespace NUMINAMATH_CALUDE_minimum_boxes_required_l232_23271

structure BoxType where
  capacity : ℕ
  quantity : ℕ

def total_brochures : ℕ := 10000

def small_box : BoxType := ⟨50, 40⟩
def medium_box : BoxType := ⟨200, 25⟩
def large_box : BoxType := ⟨500, 10⟩

def box_types : List BoxType := [small_box, medium_box, large_box]

def can_ship (boxes : List (BoxType × ℕ)) : Prop :=
  (boxes.map (λ (b, n) => b.capacity * n)).sum ≥ total_brochures

theorem minimum_boxes_required :
  ∃ (boxes : List (BoxType × ℕ)),
    (boxes.map Prod.snd).sum = 35 ∧
    can_ship boxes ∧
    ∀ (other_boxes : List (BoxType × ℕ)),
      can_ship other_boxes →
      (other_boxes.map Prod.snd).sum ≥ 35 :=
sorry

end NUMINAMATH_CALUDE_minimum_boxes_required_l232_23271


namespace NUMINAMATH_CALUDE_g_range_l232_23229

def g (x : ℝ) := x^2 - 2*x

theorem g_range :
  ∀ x ∈ Set.Icc 0 3, -1 ≤ g x ∧ g x ≤ 3 ∧
  (∃ x₁ ∈ Set.Icc 0 3, g x₁ = -1) ∧
  (∃ x₂ ∈ Set.Icc 0 3, g x₂ = 3) :=
sorry

end NUMINAMATH_CALUDE_g_range_l232_23229


namespace NUMINAMATH_CALUDE_compacted_space_calculation_l232_23279

/-- The number of cans Nick has -/
def num_cans : ℕ := 100

/-- The space each can takes up before compaction (in square inches) -/
def initial_space : ℝ := 30

/-- The percentage of space each can takes up after compaction -/
def compaction_ratio : ℝ := 0.35

/-- The total space occupied by all cans after compaction (in square inches) -/
def total_compacted_space : ℝ := num_cans * (initial_space * compaction_ratio)

theorem compacted_space_calculation :
  total_compacted_space = 1050 := by sorry

end NUMINAMATH_CALUDE_compacted_space_calculation_l232_23279


namespace NUMINAMATH_CALUDE_bianca_next_day_miles_l232_23201

/-- The number of miles Bianca ran on the first day -/
def first_day_miles : ℕ := 8

/-- The total number of miles Bianca ran over two days -/
def total_miles : ℕ := 12

/-- The number of miles Bianca ran on the next day -/
def next_day_miles : ℕ := total_miles - first_day_miles

theorem bianca_next_day_miles :
  next_day_miles = 4 :=
by sorry

end NUMINAMATH_CALUDE_bianca_next_day_miles_l232_23201


namespace NUMINAMATH_CALUDE_max_c_value_l232_23219

-- Define the function f
def f (a b c d x : ℝ) : ℝ := a * x^3 + 2 * b * x^2 + 3 * c * x + 4 * d

-- Define the conditions
def is_valid_function (a b c d : ℝ) : Prop :=
  a < 0 ∧ c > 0 ∧
  (∀ x, f a b c d x = -f a b c d (-x)) ∧
  (∀ x ∈ Set.Icc 0 1, f a b c d x ∈ Set.Icc 0 1)

-- Theorem statement
theorem max_c_value (a b c d : ℝ) (h : is_valid_function a b c d) :
  c ≤ Real.sqrt 3 / 2 ∧ ∃ a₀ b₀ d₀, is_valid_function a₀ b₀ (Real.sqrt 3 / 2) d₀ :=
sorry

end NUMINAMATH_CALUDE_max_c_value_l232_23219


namespace NUMINAMATH_CALUDE_employee_pay_l232_23247

theorem employee_pay (total : ℝ) (x y : ℝ) (h1 : total = 616) (h2 : x = 1.2 * y) (h3 : total = x + y) : y = 280 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l232_23247


namespace NUMINAMATH_CALUDE_brick_width_calculation_l232_23291

/-- Proves that the width of each brick is 11.25 cm given the wall and brick dimensions and the number of bricks needed. -/
theorem brick_width_calculation (brick_length : ℝ) (brick_height : ℝ) (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 25 →
  brick_height = 6 →
  wall_length = 750 →
  wall_width = 600 →
  wall_height = 22.5 →
  num_bricks = 6000 →
  ∃ (brick_width : ℝ), brick_width = 11.25 ∧ 
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l232_23291


namespace NUMINAMATH_CALUDE_equation_solution_l232_23222

theorem equation_solution : 
  ∃ (x : ℝ), x ≥ 0 ∧ 2021 * x = 2022 * (x^2021)^(1/2022) - 1 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l232_23222


namespace NUMINAMATH_CALUDE_biography_percentage_before_purchase_l232_23299

theorem biography_percentage_before_purchase
  (increase_rate : ℝ)
  (final_percentage : ℝ)
  (h_increase : increase_rate = 0.8823529411764707)
  (h_final : final_percentage = 0.32)
  : ∃ (initial_percentage : ℝ),
    initial_percentage * (1 + increase_rate) = final_percentage ∧
    initial_percentage = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_biography_percentage_before_purchase_l232_23299


namespace NUMINAMATH_CALUDE_at_least_one_even_difference_l232_23272

theorem at_least_one_even_difference (n : ℕ) (a b : Fin (2*n+1) → ℤ) 
  (h : ∃ σ : Equiv.Perm (Fin (2*n+1)), ∀ i, b i = a (σ i)) :
  ∃ k : Fin (2*n+1), Even (a k - b k) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_even_difference_l232_23272
