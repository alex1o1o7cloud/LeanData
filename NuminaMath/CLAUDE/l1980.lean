import Mathlib

namespace NUMINAMATH_CALUDE_hexagonal_prism_vertices_l1980_198045

/-- A hexagonal prism is a three-dimensional geometric shape with hexagonal bases -/
structure HexagonalPrism :=
  (base : Nat)
  (height : Nat)

/-- The number of vertices in a hexagonal prism -/
def num_vertices (prism : HexagonalPrism) : Nat :=
  12

/-- Theorem: A hexagonal prism has 12 vertices -/
theorem hexagonal_prism_vertices (prism : HexagonalPrism) :
  num_vertices prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_vertices_l1980_198045


namespace NUMINAMATH_CALUDE_room_number_unit_digit_l1980_198032

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def divisible_by_seven (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def contains_digit_nine (n : ℕ) : Prop := ∃ a b : ℕ, n = 10 * a + 9 * b ∧ b ≤ 1

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def satisfies_three_conditions (n : ℕ) : Prop :=
  (is_prime n ∧ is_even n ∧ divisible_by_seven n) ∨
  (is_prime n ∧ is_even n ∧ contains_digit_nine n) ∨
  (is_prime n ∧ divisible_by_seven n ∧ contains_digit_nine n) ∨
  (is_even n ∧ divisible_by_seven n ∧ contains_digit_nine n)

theorem room_number_unit_digit :
  ∃ n : ℕ, is_two_digit n ∧ satisfies_three_conditions n ∧ n % 10 = 8 :=
sorry

end NUMINAMATH_CALUDE_room_number_unit_digit_l1980_198032


namespace NUMINAMATH_CALUDE_slope_extension_l1980_198002

theorem slope_extension (slope_length : ℝ) (initial_angle final_angle : ℝ) 
  (h1 : slope_length = 1)
  (h2 : initial_angle = 20 * π / 180)
  (h3 : final_angle = 10 * π / 180) :
  let extension := slope_length
  extension = 1 := by sorry

end NUMINAMATH_CALUDE_slope_extension_l1980_198002


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_8_l1980_198097

theorem largest_four_digit_divisible_by_8 : 
  ∀ n : ℕ, n ≤ 9999 → n ≥ 1000 → n % 8 = 0 → n ≤ 9992 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_8_l1980_198097


namespace NUMINAMATH_CALUDE_equal_area_line_coeff_sum_l1980_198053

/-- A region formed by eight unit circles packed in the first quadrant --/
def R : Set (ℝ × ℝ) :=
  sorry

/-- A line with slope 3 that divides R into two equal areas --/
def l : Set (ℝ × ℝ) :=
  sorry

/-- The line l expressed in the form ax = by + c --/
def line_equation (a b c : ℕ) : Prop :=
  ∀ x y, (x, y) ∈ l ↔ a * x = b * y + c

/-- The coefficients a, b, and c are positive integers with gcd 1 --/
def coeff_constraints (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ Nat.gcd a (Nat.gcd b c) = 1

theorem equal_area_line_coeff_sum :
  ∃ a b c : ℕ,
    line_equation a b c ∧
    coeff_constraints a b c ∧
    a^2 + b^2 + c^2 = 65 :=
sorry

end NUMINAMATH_CALUDE_equal_area_line_coeff_sum_l1980_198053


namespace NUMINAMATH_CALUDE_approximate_cost_price_of_toy_l1980_198059

/-- The cost price of a toy given the selling conditions --/
def cost_price_of_toy (num_toys : ℕ) (total_selling_price : ℚ) (gain_in_toys : ℕ) : ℚ :=
  let selling_price_per_toy := total_selling_price / num_toys
  let x := selling_price_per_toy * num_toys / (num_toys + gain_in_toys)
  x

/-- Theorem stating the approximate cost price of a toy under given conditions --/
theorem approximate_cost_price_of_toy :
  let calculated_price := cost_price_of_toy 18 27300 3
  ⌊calculated_price⌋ = 1300 := by sorry

end NUMINAMATH_CALUDE_approximate_cost_price_of_toy_l1980_198059


namespace NUMINAMATH_CALUDE_third_wall_length_l1980_198083

/-- Calculates the length of the third wall in a hall of mirrors. -/
theorem third_wall_length
  (total_glass : ℝ)
  (wall1_length wall1_height : ℝ)
  (wall2_length wall2_height : ℝ)
  (wall3_height : ℝ)
  (h1 : total_glass = 960)
  (h2 : wall1_length = 30 ∧ wall1_height = 12)
  (h3 : wall2_length = 30 ∧ wall2_height = 12)
  (h4 : wall3_height = 12)
  : ∃ (wall3_length : ℝ),
    total_glass = wall1_length * wall1_height + wall2_length * wall2_height + wall3_length * wall3_height
    ∧ wall3_length = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_third_wall_length_l1980_198083


namespace NUMINAMATH_CALUDE_opponent_total_score_l1980_198023

def team_scores : List ℕ := [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def one_run_losses : ℕ := 3
def two_run_losses : ℕ := 2

def remaining_games (total_games : ℕ) : ℕ :=
  total_games - one_run_losses - two_run_losses

theorem opponent_total_score :
  ∃ (one_loss_scores two_loss_scores three_times_scores : List ℕ),
    (one_loss_scores.length = one_run_losses) ∧
    (two_loss_scores.length = two_run_losses) ∧
    (three_times_scores.length = remaining_games team_scores.length) ∧
    (∀ (s : ℕ), s ∈ one_loss_scores → s ∈ team_scores) ∧
    (∀ (s : ℕ), s ∈ two_loss_scores → s ∈ team_scores) ∧
    (∀ (s : ℕ), s ∈ three_times_scores → s ∈ team_scores) ∧
    (∀ (s : ℕ), s ∈ one_loss_scores → ∃ (o : ℕ), o = s + 1) ∧
    (∀ (s : ℕ), s ∈ two_loss_scores → ∃ (o : ℕ), o = s + 2) ∧
    (∀ (s : ℕ), s ∈ three_times_scores → ∃ (o : ℕ), o = s / 3) ∧
    (one_loss_scores.sum + two_loss_scores.sum + three_times_scores.sum = 42) :=
by
  sorry

end NUMINAMATH_CALUDE_opponent_total_score_l1980_198023


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l1980_198095

theorem max_distance_circle_to_line : 
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let line := {p : ℝ × ℝ | p.1 = 2}
  ∀ p ∈ circle, ∀ q ∈ line, 
    ∃ r ∈ circle, Real.sqrt ((r.1 - q.1)^2 + (r.2 - q.2)^2) = 3 ∧
    ∀ s ∈ circle, Real.sqrt ((s.1 - q.1)^2 + (s.2 - q.2)^2) ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l1980_198095


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l1980_198064

theorem intersection_sum_zero (x₁ x₂ : ℝ) :
  (x₁^2 + 9^2 = 169) →
  (x₂^2 + 9^2 = 169) →
  x₁ ≠ x₂ →
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l1980_198064


namespace NUMINAMATH_CALUDE_factorization_equality_l1980_198026

theorem factorization_equality (x y : ℝ) : 
  x^2 - 2*x - 2*y^2 + 4*y - x*y = (x - 2*y)*(x + y - 2) := by
sorry

end NUMINAMATH_CALUDE_factorization_equality_l1980_198026


namespace NUMINAMATH_CALUDE_cubic_function_property_l1980_198072

theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x - 3
  f (-2) = 7 → f 2 = -13 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1980_198072


namespace NUMINAMATH_CALUDE_special_subset_characterization_l1980_198027

/-- The highest power of 2 that divides a natural number -/
def v2 (n : ℕ) : ℕ := sorry

/-- Theorem: Characterization of elements in special subsets of {1, ..., 2n} -/
theorem special_subset_characterization (n : ℕ) (c : ℕ) (hc : c ∈ Finset.range (2 * n + 1)) :
  (∃ (A : Finset ℕ), A ⊆ Finset.range (2 * n + 1) ∧ 
   A.card = n ∧ 
   c ∈ A ∧ 
   ∀ (x y : ℕ), x ∈ A → y ∈ A → x ≠ y → ¬(x ∣ y)) ↔ 
  (c : ℚ) > n * (2/3)^(v2 c + 1) :=
sorry

end NUMINAMATH_CALUDE_special_subset_characterization_l1980_198027


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1980_198089

theorem quadratic_one_solution (k : ℝ) : 
  (∃! x : ℝ, k * x^2 + 4 * x + 4 = 0) → k = 0 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1980_198089


namespace NUMINAMATH_CALUDE_quadratic_product_property_l1980_198000

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0

/-- The discriminant of a quadratic polynomial -/
def discriminant (p : QuadraticPolynomial) : ℤ :=
  p.b^2 - 4 * p.a * p.c

/-- Predicate for a quadratic polynomial having distinct roots -/
def has_distinct_roots (p : QuadraticPolynomial) : Prop :=
  discriminant p ≠ 0

/-- The product of the roots of a quadratic polynomial -/
def root_product (p : QuadraticPolynomial) : ℚ :=
  (p.c : ℚ) / (p.a : ℚ)

/-- The product of the coefficients of a quadratic polynomial -/
def coeff_product (p : QuadraticPolynomial) : ℤ :=
  p.a * p.b * p.c

theorem quadratic_product_property (p : QuadraticPolynomial) 
  (h_distinct : has_distinct_roots p)
  (h_product : (coeff_product p : ℚ) = root_product p) :
  ∃ (n : ℤ), n < 0 ∧ coeff_product p = n :=
by sorry

end NUMINAMATH_CALUDE_quadratic_product_property_l1980_198000


namespace NUMINAMATH_CALUDE_other_number_proof_l1980_198086

theorem other_number_proof (a b : ℕ+) : 
  Nat.lcm a b = 2520 →
  Nat.gcd a b = 12 →
  a = 240 →
  b = 126 := by
sorry

end NUMINAMATH_CALUDE_other_number_proof_l1980_198086


namespace NUMINAMATH_CALUDE_arcade_spending_correct_l1980_198062

/-- Calculates the total money spent by John at the arcade --/
def arcade_spending (total_time : ℕ) (break_time : ℕ) (rate1 : ℚ) (interval1 : ℕ)
  (rate2 : ℚ) (interval2 : ℕ) (rate3 : ℚ) (interval3 : ℕ) (rate4 : ℚ) (interval4 : ℕ) : ℚ :=
  let play_time := total_time - break_time
  let hour1 := (60 / interval1) * rate1
  let hour2 := (60 / interval2) * rate2
  let hour3 := (60 / interval3) * rate3
  let hour4 := ((play_time - 180) / interval4) * rate4
  hour1 + hour2 + hour3 + hour4

theorem arcade_spending_correct :
  arcade_spending 275 50 0.5 4 0.75 5 1 3 1.25 7 = 42.75 := by
  sorry

end NUMINAMATH_CALUDE_arcade_spending_correct_l1980_198062


namespace NUMINAMATH_CALUDE_chess_points_theorem_l1980_198011

theorem chess_points_theorem :
  ∃! (s : Finset ℕ), s.card = 2 ∧
  (∀ x ∈ s, ∃ n : ℕ, 11 * n + x * (100 - n) = 800) ∧
  (3 ∈ s ∧ 4 ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_chess_points_theorem_l1980_198011


namespace NUMINAMATH_CALUDE_train_length_l1980_198055

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (cross_time : ℝ) : 
  speed_kmph = 72 → cross_time = 7 → speed_kmph * (1000 / 3600) * cross_time = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1980_198055


namespace NUMINAMATH_CALUDE_sqrt_difference_less_than_sqrt_of_difference_l1980_198082

theorem sqrt_difference_less_than_sqrt_of_difference 
  (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_less_than_sqrt_of_difference_l1980_198082


namespace NUMINAMATH_CALUDE_gcd_8369_4087_2159_l1980_198076

theorem gcd_8369_4087_2159 : Nat.gcd 8369 (Nat.gcd 4087 2159) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8369_4087_2159_l1980_198076


namespace NUMINAMATH_CALUDE_parabola_translation_up_l1980_198020

/-- Represents a parabola in the form y = ax² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Translates a parabola vertically by a given amount -/
def translate_vertical (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, b := p.b + dy }

theorem parabola_translation_up :
  let original := Parabola.mk 3 0
  let translated := translate_vertical original 1
  translated = Parabola.mk 3 1 := by sorry

end NUMINAMATH_CALUDE_parabola_translation_up_l1980_198020


namespace NUMINAMATH_CALUDE_unique_prime_triple_l1980_198096

theorem unique_prime_triple : 
  ∃! (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    (∃ k : ℤ, (p^2 + 2*q : ℤ) = k * (q + r : ℤ)) ∧
    (∃ l : ℤ, (q^2 + 9*r : ℤ) = l * (r + p : ℤ)) ∧
    (∃ m : ℤ, (r^2 + 3*p : ℤ) = m * (p + q : ℤ)) ∧
    p = 2 ∧ q = 3 ∧ r = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l1980_198096


namespace NUMINAMATH_CALUDE_u_converges_to_zero_l1980_198046

open Real

variable (f : ℝ → ℝ)
variable (u : ℕ → ℝ)

-- f is non-decreasing
axiom f_nondecreasing : ∀ x y, x ≤ y → f x ≤ f y

-- f(y) - f(x) < y - x for all real numbers x and y > x
axiom f_contractive : ∀ x y, x < y → f y - f x < y - x

-- Recurrence relation for u
axiom u_recurrence : ∀ n : ℕ, u (n + 2) = f (u (n + 1)) - f (u n)

-- Theorem to prove
theorem u_converges_to_zero : 
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |u n| < ε) :=
sorry

end NUMINAMATH_CALUDE_u_converges_to_zero_l1980_198046


namespace NUMINAMATH_CALUDE_square_area_is_36_l1980_198094

/-- A square in the coordinate plane with specific y-coordinates -/
structure SquareInPlane where
  -- Define the y-coordinates of the vertices
  y1 : ℝ := 0
  y2 : ℝ := 3
  y3 : ℝ := 0
  y4 : ℝ := -3

/-- The area of the square -/
def squareArea (s : SquareInPlane) : ℝ := 36

/-- Theorem: The area of the square with given y-coordinates is 36 -/
theorem square_area_is_36 (s : SquareInPlane) : squareArea s = 36 := by
  sorry


end NUMINAMATH_CALUDE_square_area_is_36_l1980_198094


namespace NUMINAMATH_CALUDE_exists_abs_leq_zero_l1980_198009

theorem exists_abs_leq_zero : ∃ x : ℝ, |x| ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_abs_leq_zero_l1980_198009


namespace NUMINAMATH_CALUDE_odometer_sum_squares_l1980_198067

/-- Represents a 3-digit number abc where a, b, c are single digits --/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  a_positive : a ≥ 1
  sum_constraint : a + b + c = 8

/-- Represents the odometer readings for Denise's trip --/
structure OdometerReadings where
  initial : ThreeDigitNumber
  final : ThreeDigitNumber
  final_swap : final.a = initial.b ∧ final.b = initial.a ∧ final.c = initial.c

/-- Represents Denise's trip --/
structure Trip where
  readings : OdometerReadings
  hours : ℕ
  hours_positive : hours > 0
  speed : ℕ
  speed_eq : speed = 48
  distance_constraint : 90 * (readings.initial.b - readings.initial.a) = hours * speed

theorem odometer_sum_squares (t : Trip) : 
  t.readings.initial.a ^ 2 + t.readings.initial.b ^ 2 + t.readings.initial.c ^ 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_odometer_sum_squares_l1980_198067


namespace NUMINAMATH_CALUDE_intersection_distance_l1980_198013

/-- The distance between the intersection points of the line y = 1 - x and the circle x^2 + y^2 = 8 is equal to √30 -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), 
    (A.1^2 + A.2^2 = 8) ∧ 
    (B.1^2 + B.2^2 = 8) ∧ 
    (A.2 = 1 - A.1) ∧ 
    (B.2 = 1 - B.1) ∧ 
    (A ≠ B) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 30) :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l1980_198013


namespace NUMINAMATH_CALUDE_train_length_calculation_l1980_198031

/-- The length of a train given its speed and time to cross a point -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 360 →
  time_s = 0.9999200063994881 →
  ∃ (length_m : ℝ), abs (length_m - 99.992) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1980_198031


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_abs_sqrt_square_domain_l1980_198099

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

theorem abs_sqrt_square_domain : Set.univ = {x : ℝ | ∃ y, y = Real.sqrt (x^2)} := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_abs_sqrt_square_domain_l1980_198099


namespace NUMINAMATH_CALUDE_downstream_distance_l1980_198048

/-- Calculates the distance traveled downstream by a boat -/
theorem downstream_distance 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 24) 
  (h2 : stream_speed = 4) 
  (h3 : time = 2) : 
  boat_speed * time + stream_speed * time = 56 := by
  sorry

#check downstream_distance

end NUMINAMATH_CALUDE_downstream_distance_l1980_198048


namespace NUMINAMATH_CALUDE_milk_price_is_three_l1980_198033

/-- Represents the milk and butter selling scenario --/
structure MilkButterScenario where
  num_cows : ℕ
  milk_per_cow : ℕ
  num_customers : ℕ
  milk_per_customer : ℕ
  butter_sticks_per_gallon : ℕ
  butter_price_per_stick : ℚ
  total_earnings : ℚ

/-- Calculates the price per gallon of milk --/
def price_per_gallon (scenario : MilkButterScenario) : ℚ :=
  let total_milk := scenario.num_cows * scenario.milk_per_cow
  let sold_milk := scenario.num_customers * scenario.milk_per_customer
  let remaining_milk := total_milk - sold_milk
  let butter_sticks := remaining_milk * scenario.butter_sticks_per_gallon
  let butter_earnings := butter_sticks * scenario.butter_price_per_stick
  let milk_earnings := scenario.total_earnings - butter_earnings
  milk_earnings / sold_milk

/-- Theorem stating that the price per gallon of milk is $3 --/
theorem milk_price_is_three (scenario : MilkButterScenario) 
  (h1 : scenario.num_cows = 12)
  (h2 : scenario.milk_per_cow = 4)
  (h3 : scenario.num_customers = 6)
  (h4 : scenario.milk_per_customer = 6)
  (h5 : scenario.butter_sticks_per_gallon = 2)
  (h6 : scenario.butter_price_per_stick = 3/2)
  (h7 : scenario.total_earnings = 144) :
  price_per_gallon scenario = 3 := by
  sorry

end NUMINAMATH_CALUDE_milk_price_is_three_l1980_198033


namespace NUMINAMATH_CALUDE_inverse_proportional_m_range_l1980_198068

/-- Given an inverse proportional function y = (1 - 2m) / x with two points
    A(x₁, y₁) and B(x₂, y₂) on its graph, where x₁ < 0 < x₂ and y₁ < y₂,
    prove that the range of m is m < 1/2. -/
theorem inverse_proportional_m_range (m x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = (1 - 2*m) / x₁)
  (h2 : y₂ = (1 - 2*m) / x₂)
  (h3 : x₁ < 0)
  (h4 : 0 < x₂)
  (h5 : y₁ < y₂) :
  m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_inverse_proportional_m_range_l1980_198068


namespace NUMINAMATH_CALUDE_chord_longer_than_arc_l1980_198058

theorem chord_longer_than_arc (R : ℝ) (h : R > 0) :
  let angle := 60 * π / 180
  let arc_length := angle * R
  let new_radius := 1.05 * R
  let chord_length := 2 * new_radius * Real.sin (angle / 2)
  chord_length > arc_length := by sorry

end NUMINAMATH_CALUDE_chord_longer_than_arc_l1980_198058


namespace NUMINAMATH_CALUDE_tanning_salon_revenue_l1980_198049

/-- Revenue calculation for a tanning salon --/
theorem tanning_salon_revenue :
  let first_visit_cost : ℕ := 10
  let subsequent_visit_cost : ℕ := 8
  let total_customers : ℕ := 100
  let second_visit_customers : ℕ := 30
  let third_visit_customers : ℕ := 10
  
  let first_visit_revenue := first_visit_cost * total_customers
  let second_visit_revenue := subsequent_visit_cost * second_visit_customers
  let third_visit_revenue := subsequent_visit_cost * third_visit_customers
  
  first_visit_revenue + second_visit_revenue + third_visit_revenue = 1320 :=
by
  sorry


end NUMINAMATH_CALUDE_tanning_salon_revenue_l1980_198049


namespace NUMINAMATH_CALUDE_thought_number_is_729_l1980_198081

/-- 
Given a three-digit number, if each of the numbers 109, 704, and 124 
matches it exactly in one digit place, then the number is 729.
-/
theorem thought_number_is_729 (x : ℕ) : 
  (100 ≤ x ∧ x < 1000) → 
  (∃! d : ℕ, d < 3 ∧ (x / 10^d % 10 = 109 / 10^d % 10)) →
  (∃! d : ℕ, d < 3 ∧ (x / 10^d % 10 = 704 / 10^d % 10)) →
  (∃! d : ℕ, d < 3 ∧ (x / 10^d % 10 = 124 / 10^d % 10)) →
  x = 729 := by
sorry


end NUMINAMATH_CALUDE_thought_number_is_729_l1980_198081


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l1980_198088

/-- 
Given a canoe that rows downstream at 12 km/hr and a stream with a speed of 2 km/hr,
this theorem proves that the speed of the canoe when rowing upstream is 8 km/hr.
-/
theorem canoe_upstream_speed :
  let downstream_speed : ℝ := 12
  let stream_speed : ℝ := 2
  let canoe_speed : ℝ := downstream_speed - stream_speed
  let upstream_speed : ℝ := canoe_speed - stream_speed
  upstream_speed = 8 := by sorry

end NUMINAMATH_CALUDE_canoe_upstream_speed_l1980_198088


namespace NUMINAMATH_CALUDE_find_number_l1980_198074

theorem find_number : ∃ x : ℝ, 13 * x - 272 = 105 ∧ x = 29 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1980_198074


namespace NUMINAMATH_CALUDE_unique_solution_conditions_l1980_198051

theorem unique_solution_conditions (n p : ℕ) :
  (∃! (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + p * y = n ∧ x + y = p ^ z) ↔
  (p > 1 ∧ 
   (n - 1) % (p - 1) = 0 ∧
   ∀ k : ℕ, n ≠ p ^ k) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_conditions_l1980_198051


namespace NUMINAMATH_CALUDE_undefined_expression_l1980_198035

theorem undefined_expression (x : ℝ) : 
  (x^2 - 22*x + 121 = 0) ↔ (x = 11) := by sorry

#check undefined_expression

end NUMINAMATH_CALUDE_undefined_expression_l1980_198035


namespace NUMINAMATH_CALUDE_sum_of_trapezoid_areas_l1980_198079

/-- Represents a trapezoid with four side lengths --/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the area of a trapezoid given its side lengths --/
def trapezoidArea (t : Trapezoid) : ℝ := sorry

/-- Generates all valid trapezoid configurations with given side lengths --/
def validConfigurations (sides : List ℝ) : List Trapezoid := sorry

/-- Theorem: The sum of areas of all valid trapezoids with side lengths 4, 6, 8, and 10 
    is equal to a specific value --/
theorem sum_of_trapezoid_areas :
  let sides := [4, 6, 8, 10]
  let validTraps := validConfigurations sides
  let areas := validTraps.map trapezoidArea
  ∃ (total : ℝ), areas.sum = total := by sorry

end NUMINAMATH_CALUDE_sum_of_trapezoid_areas_l1980_198079


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_upper_bound_l1980_198061

theorem quadratic_inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 3 → a ≤ x^2 - 4*x) →
  a ≤ -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_upper_bound_l1980_198061


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1980_198070

theorem sufficient_not_necessary (a : ℝ) :
  (a > 9 → (1 / a) < (1 / 9)) ∧
  ∃ b : ℝ, (1 / b) < (1 / 9) ∧ b ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1980_198070


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l1980_198075

/-- The number of ways to distribute indistinguishable balls among distinguishable boxes -/
def distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls among 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l1980_198075


namespace NUMINAMATH_CALUDE_keyboard_cost_l1980_198028

/-- 
Given:
- 15 keyboards and 25 printers cost $2050 in total
- One printer costs $70

Prove that the cost of one keyboard is $20
-/
theorem keyboard_cost : 
  ∀ (keyboard_cost : ℝ),
  (15 * keyboard_cost + 25 * 70 = 2050) →
  keyboard_cost = 20 := by
sorry

end NUMINAMATH_CALUDE_keyboard_cost_l1980_198028


namespace NUMINAMATH_CALUDE_angela_january_sleep_l1980_198014

/-- The number of hours Angela slept per night in December -/
def december_sleep_hours : ℝ := 6.5

/-- The number of days in December -/
def december_days : ℕ := 31

/-- The additional hours Angela slept in January compared to December -/
def january_additional_sleep : ℝ := 62

/-- The number of days in January -/
def january_days : ℕ := 31

/-- Calculates the total hours Angela slept in December -/
def december_total_sleep : ℝ := december_sleep_hours * december_days

/-- Calculates the total hours Angela slept in January -/
def january_total_sleep : ℝ := december_total_sleep + january_additional_sleep

/-- Theorem stating that Angela slept 8.5 hours per night in January -/
theorem angela_january_sleep :
  january_total_sleep / january_days = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_angela_january_sleep_l1980_198014


namespace NUMINAMATH_CALUDE_x_axis_condition_l1980_198007

/-- A line in the 2D plane represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The x-axis is a line where y = 0 for all x -/
def is_x_axis (l : Line) : Prop :=
  ∀ x y : ℝ, l.A * x + l.B * y + l.C = 0 ↔ y = 0

/-- If a line is the x-axis, then B ≠ 0 and A = C = 0 -/
theorem x_axis_condition (l : Line) :
  is_x_axis l → l.B ≠ 0 ∧ l.A = 0 ∧ l.C = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_axis_condition_l1980_198007


namespace NUMINAMATH_CALUDE_vectors_are_coplanar_l1980_198073

def a : ℝ × ℝ × ℝ := (1, -2, 6)
def b : ℝ × ℝ × ℝ := (1, 0, 1)
def c : ℝ × ℝ × ℝ := (2, -6, 17)

def coplanar (u v w : ℝ × ℝ × ℝ) : Prop :=
  let (u₁, u₂, u₃) := u
  let (v₁, v₂, v₃) := v
  let (w₁, w₂, w₃) := w
  (u₁ * (v₂ * w₃ - v₃ * w₂) - u₂ * (v₁ * w₃ - v₃ * w₁) + u₃ * (v₁ * w₂ - v₂ * w₁)) = 0

theorem vectors_are_coplanar : coplanar a b c := by
  sorry

end NUMINAMATH_CALUDE_vectors_are_coplanar_l1980_198073


namespace NUMINAMATH_CALUDE_chord_equation_l1980_198043

theorem chord_equation (x y : ℝ) :
  (x^2 + y^2 - 2*x = 0) →  -- Circle equation
  (∃ (t : ℝ), x = 1/2 + t ∧ y = 1/2 + t) →  -- Midpoint condition
  (x - y = 0) :=  -- Line equation
by sorry

end NUMINAMATH_CALUDE_chord_equation_l1980_198043


namespace NUMINAMATH_CALUDE_xyz_value_l1980_198078

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19) :
  x * y * z = 10 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1980_198078


namespace NUMINAMATH_CALUDE_function_intersection_theorem_l1980_198093

noncomputable def f (a b x : ℝ) : ℝ := (a * Real.log x + b) / x

noncomputable def g (a x : ℝ) : ℝ := a + 2 - x - 2 / x

def has_extremum_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

def exactly_one_intersection (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x ≤ b ∧ f x = g x

theorem function_intersection_theorem (a b : ℝ) :
  a ≤ 2 →
  a ≠ 0 →
  has_extremum_at (f a b) (1 / Real.exp 1) →
  (a = -1 ∨ a < -2 / Real.log 2 ∨ (0 < a ∧ a ≤ 2)) ↔
  exactly_one_intersection (f a b) (g a) 0 2 :=
sorry

end NUMINAMATH_CALUDE_function_intersection_theorem_l1980_198093


namespace NUMINAMATH_CALUDE_characterize_M_l1980_198038

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 1 = 0}

-- Define the set M
def M : Set ℝ := {m : ℝ | A ∩ B m = B m}

-- Theorem statement
theorem characterize_M : M = {0, 1/2, 1/3} := by sorry

end NUMINAMATH_CALUDE_characterize_M_l1980_198038


namespace NUMINAMATH_CALUDE_triangle_is_obtuse_l1980_198019

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * cos(C) = b + (2/3) * c, then ABC is an obtuse triangle -/
theorem triangle_is_obtuse (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a * Real.cos C = b + (2/3) * c →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  π / 2 < A := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_obtuse_l1980_198019


namespace NUMINAMATH_CALUDE_unspent_portion_after_transfer_l1980_198066

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Calculates the unspent portion of a credit card's limit after a balance transfer -/
def unspentPortionAfterTransfer (gold : CreditCard) (platinum : CreditCard) : ℝ :=
  sorry

/-- Theorem stating the unspent portion of the platinum card's limit after transfer -/
theorem unspent_portion_after_transfer
  (gold : CreditCard)
  (platinum : CreditCard)
  (h1 : platinum.limit = 2 * gold.limit)
  (h2 : ∃ X : ℝ, gold.balance = X * gold.limit)
  (h3 : platinum.balance = (1 / 7) * platinum.limit) :
  unspentPortionAfterTransfer gold platinum = (12 - 7 * (gold.balance / gold.limit)) / 14 :=
  sorry

end NUMINAMATH_CALUDE_unspent_portion_after_transfer_l1980_198066


namespace NUMINAMATH_CALUDE_least_prime_factor_of_7_4_minus_7_3_l1980_198085

theorem least_prime_factor_of_7_4_minus_7_3 :
  Nat.minFac (7^4 - 7^3) = 2 := by
sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_7_4_minus_7_3_l1980_198085


namespace NUMINAMATH_CALUDE_bread_lasts_three_days_l1980_198039

/-- Represents the number of days bread will last for a household -/
def days_bread_lasts (
  household_members : ℕ)
  (breakfast_slices_per_member : ℕ)
  (snack_slices_per_member : ℕ)
  (slices_per_loaf : ℕ)
  (number_of_loaves : ℕ) : ℕ :=
  let total_slices := number_of_loaves * slices_per_loaf
  let daily_consumption := household_members * (breakfast_slices_per_member + snack_slices_per_member)
  total_slices / daily_consumption

/-- Theorem stating that 5 loaves of bread will last 3 days for a family of 4 -/
theorem bread_lasts_three_days :
  days_bread_lasts 4 3 2 12 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bread_lasts_three_days_l1980_198039


namespace NUMINAMATH_CALUDE_probability_two_fours_eight_dice_l1980_198008

theorem probability_two_fours_eight_dice : 
  let n : ℕ := 8  -- number of dice
  let k : ℕ := 2  -- number of successes (showing 4)
  let p : ℚ := 1 / 6  -- probability of rolling a 4 on a single die
  Nat.choose n k * p^k * (1 - p)^(n - k) = (28 * 15625 : ℚ) / 279936 := by
sorry

end NUMINAMATH_CALUDE_probability_two_fours_eight_dice_l1980_198008


namespace NUMINAMATH_CALUDE_cylinder_radius_problem_l1980_198021

/-- 
Given a cylinder with height 4 inches, if increasing the radius by 4 inches 
results in the same volume as tripling the height, then the original radius 
is 2 + 2√3 inches.
-/
theorem cylinder_radius_problem (r : ℝ) : 
  (π * (r + 4)^2 * 4 = π * r^2 * 12) → 
  r = 2 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_problem_l1980_198021


namespace NUMINAMATH_CALUDE_amanda_kitchen_upgrade_l1980_198030

/-- The total cost of Amanda's kitchen upgrade after applying discounts -/
def kitchen_upgrade_cost (cabinet_knobs : ℕ) (knob_price : ℚ) (drawer_pulls : ℕ) (pull_price : ℚ) 
  (knob_discount : ℚ) (pull_discount : ℚ) : ℚ :=
  let knob_total := cabinet_knobs * knob_price
  let pull_total := drawer_pulls * pull_price
  let discounted_knob_total := knob_total * (1 - knob_discount)
  let discounted_pull_total := pull_total * (1 - pull_discount)
  discounted_knob_total + discounted_pull_total

/-- Amanda's kitchen upgrade cost is $67.70 -/
theorem amanda_kitchen_upgrade : 
  kitchen_upgrade_cost 18 (5/2) 8 4 (1/10) (3/20) = 677/10 := by
  sorry

end NUMINAMATH_CALUDE_amanda_kitchen_upgrade_l1980_198030


namespace NUMINAMATH_CALUDE_f_of_three_equals_seven_l1980_198050

/-- Given a function f(x) = x^7 - ax^5 + bx^3 + cx + 2 where f(-3) = -3, prove that f(3) = 7 -/
theorem f_of_three_equals_seven 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^7 - a*x^5 + b*x^3 + c*x + 2)
  (h2 : f (-3) = -3) :
  f 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_f_of_three_equals_seven_l1980_198050


namespace NUMINAMATH_CALUDE_min_value_theorem_l1980_198010

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3*x*y) :
  2*x + y ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 3*x₀*y₀ ∧ 2*x₀ + y₀ = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1980_198010


namespace NUMINAMATH_CALUDE_intersecting_line_passes_through_fixed_point_l1980_198084

/-- A parabola defined by y² = 4x passing through (1, 2) -/
structure Parabola where
  eq : ℝ → ℝ → Prop
  passes_through : eq 1 2

/-- A line intersecting the parabola at two points -/
structure IntersectingLine (p : Parabola) where
  slope : ℝ
  y_intercept : ℝ
  intersects_parabola : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    p.eq x₁ y₁ ∧ p.eq x₂ y₂ ∧
    x₁ = slope * y₁ + y_intercept ∧
    x₂ = slope * y₂ + y_intercept ∧
    y₁ * y₂ = -4

/-- The theorem to be proved -/
theorem intersecting_line_passes_through_fixed_point (p : Parabola) (l : IntersectingLine p) :
  ∃ (x y : ℝ), x = l.slope * y + l.y_intercept ∧ x = 1 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_intersecting_line_passes_through_fixed_point_l1980_198084


namespace NUMINAMATH_CALUDE_saree_price_calculation_l1980_198054

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.2) * (1 - 0.15) = 231.2 → P = 340 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_calculation_l1980_198054


namespace NUMINAMATH_CALUDE_workout_total_weight_l1980_198006

-- Define the workout structure
structure Workout :=
  (weight : ℕ)
  (reps : ℕ)

-- Define the workout session
def chest_workout : Workout := ⟨90, 8⟩
def back_workout : Workout := ⟨70, 10⟩
def leg_workout : Workout := ⟨130, 6⟩

-- Calculate total weight for a single workout
def total_weight (w : Workout) : ℕ := w.weight * w.reps

-- Calculate grand total weight for the entire session
def grand_total : ℕ :=
  total_weight chest_workout + total_weight back_workout + total_weight leg_workout

-- Theorem to prove
theorem workout_total_weight :
  grand_total = 2200 :=
by sorry

end NUMINAMATH_CALUDE_workout_total_weight_l1980_198006


namespace NUMINAMATH_CALUDE_mean_equality_problem_l1980_198044

theorem mean_equality_problem (z : ℝ) : 
  (7 + 11 + 5 + 9) / 4 = (15 + z) / 2 → z = 1 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l1980_198044


namespace NUMINAMATH_CALUDE_root_square_condition_l1980_198015

theorem root_square_condition (a : ℚ) : 
  (∃ x y : ℚ, x^2 - (15/4)*x + a^3 = 0 ∧ y^2 - (15/4)*y + a^3 = 0 ∧ x = y^2) ↔ 
  (a = 3/2 ∨ a = -5/2) := by
sorry

end NUMINAMATH_CALUDE_root_square_condition_l1980_198015


namespace NUMINAMATH_CALUDE_construction_equation_correct_l1980_198052

/-- Represents a construction project with a work stoppage -/
structure ConstructionProject where
  totalLength : ℝ
  originalDailyRate : ℝ
  workStoppageDays : ℝ
  increasedDailyRate : ℝ

/-- The equation correctly represents the construction project situation -/
theorem construction_equation_correct (project : ConstructionProject) 
  (h1 : project.totalLength = 2000)
  (h2 : project.workStoppageDays = 3)
  (h3 : project.increasedDailyRate = project.originalDailyRate + 40) :
  project.totalLength / project.originalDailyRate - 
  project.totalLength / project.increasedDailyRate = 
  project.workStoppageDays := by
sorry

end NUMINAMATH_CALUDE_construction_equation_correct_l1980_198052


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2root2_l1980_198022

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + 1)⁻¹ + (y + 1)⁻¹ = 1 → a + 2*b ≤ x + 2*y :=
by
  sorry

theorem min_value_is_2root2 (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  a + 2*b = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2root2_l1980_198022


namespace NUMINAMATH_CALUDE_cube_division_l1980_198042

theorem cube_division (n : ℕ) (small_edge : ℝ) : 
  12 / n = small_edge ∧ 
  n * 6 * small_edge^2 = 8 * 6 * 12^2 → 
  n^3 = 512 ∧ small_edge = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_cube_division_l1980_198042


namespace NUMINAMATH_CALUDE_color_distance_existence_l1980_198034

-- Define the color type
inductive Color
| Red
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- The main theorem
theorem color_distance_existence (x : ℝ) (h : x > 0) :
  ∃ (c : Color), ∃ (p1 p2 : Point), coloring p1 = c ∧ coloring p2 = c ∧ distance p1 p2 = x := by
  sorry

end NUMINAMATH_CALUDE_color_distance_existence_l1980_198034


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l1980_198029

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  num_tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to a configuration -/
def add_tiles (initial : TileConfiguration) (added : ℕ) : TileConfiguration :=
  { num_tiles := initial.num_tiles + added,
    perimeter := initial.perimeter + added }

/-- Theorem statement -/
theorem perimeter_after_adding_tiles 
  (initial : TileConfiguration)
  (h1 : initial.num_tiles = 8)
  (h2 : initial.perimeter = 16)
  (added : ℕ)
  (h3 : added = 3) :
  ∃ (final : TileConfiguration),
    final = add_tiles initial added ∧ 
    final.perimeter = 19 :=
sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l1980_198029


namespace NUMINAMATH_CALUDE_probability_x_more_points_than_y_in_given_tournament_l1980_198047

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  num_games_per_team : Nat
  win_probability : ℚ
  x_beats_y_first : Bool

/-- The probability that team X finishes with more points than team Y -/
def probability_x_more_points_than_y (tournament : SoccerTournament) : ℚ :=
  sorry

/-- The specific tournament described in the problem -/
def given_tournament : SoccerTournament where
  num_teams := 8
  num_games_per_team := 7
  win_probability := 1/2
  x_beats_y_first := true

theorem probability_x_more_points_than_y_in_given_tournament :
  probability_x_more_points_than_y given_tournament = 610/1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_more_points_than_y_in_given_tournament_l1980_198047


namespace NUMINAMATH_CALUDE_student_arrangement_count_l1980_198060

/-- Represents the number of students -/
def n : ℕ := 5

/-- Represents the condition that two specific students must be together -/
def must_be_together : Prop := True

/-- Represents the condition that two specific students cannot be together -/
def cannot_be_together : Prop := True

/-- The number of ways to arrange the students -/
def arrangement_count : ℕ := 24

/-- Theorem stating that the number of arrangements satisfying the conditions is 24 -/
theorem student_arrangement_count :
  must_be_together → cannot_be_together → arrangement_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l1980_198060


namespace NUMINAMATH_CALUDE_distance_to_school_proof_l1980_198036

/-- The distance from Layla's house to the high school -/
def distance_to_school : ℝ := 3

theorem distance_to_school_proof :
  ∀ (total_distance : ℝ),
  (2 * distance_to_school + 4 = total_distance) →
  (total_distance = 10) →
  distance_to_school = 3 := by
sorry

end NUMINAMATH_CALUDE_distance_to_school_proof_l1980_198036


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l1980_198017

/-- Given vectors a and b, prove that the magnitude of 2a + b is 5√2 -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) : 
  a = (3, 2) → b = (-1, 1) → ‖(2 • a) + b‖ = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l1980_198017


namespace NUMINAMATH_CALUDE_correct_reasoning_combination_l1980_198005

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical

-- Define the direction of reasoning
inductive ReasoningDirection
| PartToWhole
| GeneralToSpecific
| SpecificToSpecific

-- Define a function that maps a reasoning type to its direction
def reasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.PartToWhole
  | ReasoningType.Deductive => ReasoningDirection.GeneralToSpecific
  | ReasoningType.Analogical => ReasoningDirection.SpecificToSpecific

-- Theorem stating that the correct combination is (①③⑤)
theorem correct_reasoning_combination :
  (reasoningDirection ReasoningType.Inductive = ReasoningDirection.PartToWhole) ∧
  (reasoningDirection ReasoningType.Deductive = ReasoningDirection.GeneralToSpecific) ∧
  (reasoningDirection ReasoningType.Analogical = ReasoningDirection.SpecificToSpecific) :=
sorry

end NUMINAMATH_CALUDE_correct_reasoning_combination_l1980_198005


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l1980_198098

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem first_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a5 : a 5 = 9) 
  (h_a3_a2 : 2 * a 3 = a 2 + 6) : 
  a 1 = -3 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l1980_198098


namespace NUMINAMATH_CALUDE_abc_inequality_l1980_198041

theorem abc_inequality (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : (a : ℚ) / (b + c^2 : ℚ) = (a + c^2 : ℚ) / b) :
  a + b + c ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1980_198041


namespace NUMINAMATH_CALUDE_cookie_distribution_theorem_l1980_198063

/-- Represents the distribution of cookies in boxes -/
def CookieDistribution := List Nat

/-- Represents the process of taking cookies from boxes and placing them on plates -/
def distributeCookies (boxes : CookieDistribution) : List Nat :=
  let maxCookies := boxes.foldl max 0
  List.range maxCookies |>.map (fun i => boxes.filter (· > i) |>.length)

theorem cookie_distribution_theorem (boxes : CookieDistribution) :
  (boxes.toFinset |>.card) = ((distributeCookies boxes).toFinset |>.card) := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_theorem_l1980_198063


namespace NUMINAMATH_CALUDE_function_characterization_l1980_198092

def is_valid_function (f : ℕ+ → ℕ+) : Prop :=
  ∀ n : ℕ+, (n : ℕ)^3 - (n : ℕ)^2 ≤ (f n : ℕ) * (f (f n) : ℕ)^2 ∧ 
             (f n : ℕ) * (f (f n) : ℕ)^2 ≤ (n : ℕ)^3 + (n : ℕ)^2

theorem function_characterization (f : ℕ+ → ℕ+) (h : is_valid_function f) : 
  ∀ n : ℕ+, f n = n - 1 ∨ f n = n ∨ f n = n + 1 :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l1980_198092


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1980_198087

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def has_equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ (-l.c / l.a = -l.c / l.b)

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line),
    (passes_through l1 ⟨1, 2⟩ ∧ has_equal_intercepts l1) ∧
    (passes_through l2 ⟨1, 2⟩ ∧ has_equal_intercepts l2) ∧
    ((l1.a = 2 ∧ l1.b = -1 ∧ l1.c = 0) ∨ (l2.a = 1 ∧ l2.b = 1 ∧ l2.c = -3)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1980_198087


namespace NUMINAMATH_CALUDE_hcf_of_three_numbers_l1980_198018

theorem hcf_of_three_numbers (a b c : ℕ+) :
  (Nat.lcm (Nat.lcm a.val b.val) c.val = 1200) →
  (a.val * b.val * c.val = 108000) →
  (Nat.gcd (Nat.gcd a.val b.val) c.val = 90) := by
sorry

end NUMINAMATH_CALUDE_hcf_of_three_numbers_l1980_198018


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1980_198025

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a > 0 → a * b ≥ 0) ↔ (∃ (a b : ℝ), a > 0 ∧ a * b < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1980_198025


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1980_198040

theorem solution_set_inequality (x : ℝ) : 
  (1 - |x|) * (1 + x) > 0 ↔ x < 1 ∧ x ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1980_198040


namespace NUMINAMATH_CALUDE_cone_height_ratio_l1980_198012

/-- Represents a cone with height, slant height, and central angle of unfolded lateral surface -/
structure Cone where
  height : ℝ
  slant_height : ℝ
  central_angle : ℝ

/-- The theorem statement -/
theorem cone_height_ratio (A B : Cone) :
  A.slant_height = B.slant_height →
  A.central_angle + B.central_angle = 2 * Real.pi →
  A.central_angle * A.slant_height^2 / (B.central_angle * B.slant_height^2) = 2 →
  A.height / B.height = Real.sqrt 10 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l1980_198012


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1980_198057

/-- Proves that given a selling price of 400 Rs. and a profit percentage of 25%, the cost price is 320 Rs. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 400 →
  profit_percentage = 25 →
  selling_price = (1 + profit_percentage / 100) * 320 :=
by
  sorry

#check cost_price_calculation

end NUMINAMATH_CALUDE_cost_price_calculation_l1980_198057


namespace NUMINAMATH_CALUDE_extreme_value_theorem_l1980_198004

theorem extreme_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 5 * x * y) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 5 * a * b → 4 * x + 3 * y ≥ 4 * a + 3 * b ∧ 4 * x + 3 * y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_theorem_l1980_198004


namespace NUMINAMATH_CALUDE_total_beads_count_l1980_198077

/-- Represents the number of beads of each color in Sue's necklace --/
structure BeadCounts where
  purple : ℕ
  blue : ℕ
  green : ℕ
  red : ℕ

/-- Defines the conditions for Sue's necklace --/
def necklace_conditions (counts : BeadCounts) : Prop :=
  counts.purple = 7 ∧
  counts.blue = 2 * counts.purple ∧
  counts.green = counts.blue + 11 ∧
  counts.red = counts.green / 2 ∧
  (counts.purple + counts.blue + counts.green + counts.red) % 2 = 0

/-- The theorem to be proved --/
theorem total_beads_count (counts : BeadCounts) 
  (h : necklace_conditions counts) : 
  counts.purple + counts.blue + counts.green + counts.red = 58 := by
  sorry

end NUMINAMATH_CALUDE_total_beads_count_l1980_198077


namespace NUMINAMATH_CALUDE_student_number_problem_l1980_198080

theorem student_number_problem (x : ℝ) : 2 * x - 152 = 102 → x = 127 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1980_198080


namespace NUMINAMATH_CALUDE_papers_printed_proof_l1980_198016

theorem papers_printed_proof :
  let presses1 : ℕ := 40
  let presses2 : ℕ := 30
  let time1 : ℝ := 12
  let time2 : ℝ := 15.999999999999998
  let rate : ℝ := (presses2 * time2) / (presses1 * time1)
  presses1 * rate * time1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_papers_printed_proof_l1980_198016


namespace NUMINAMATH_CALUDE_smallest_a_value_l1980_198091

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b)
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  ∀ a' : ℝ, (0 ≤ a' ∧ (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x))) → a ≤ a' → a = 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1980_198091


namespace NUMINAMATH_CALUDE_cricketer_average_score_l1980_198065

theorem cricketer_average_score (total_matches : ℕ) (overall_average : ℚ) 
  (last_matches : ℕ) (last_average : ℚ) (some_average : ℚ) :
  total_matches = 10 →
  overall_average = 389/10 →
  last_matches = 4 →
  last_average = 137/4 →
  some_average = 42 →
  ∃ (x : ℕ), x + last_matches = total_matches ∧ 
    (x : ℚ) * some_average + (last_matches : ℚ) * last_average = (total_matches : ℚ) * overall_average ∧
    x = 6 :=
by sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l1980_198065


namespace NUMINAMATH_CALUDE_plane_flight_time_l1980_198003

/-- Given a plane flying between two cities, prove that the return trip takes 84 minutes -/
theorem plane_flight_time (d : ℝ) (p : ℝ) (w : ℝ) :
  d > 0 ∧ p > 0 ∧ w > 0 ∧ p > w → -- Positive distance, plane speed, and wind speed
  d / (p - w) = 96 → -- Trip against wind takes 96 minutes
  d / (p + w) = d / p - 6 → -- Return trip is 6 minutes less than in still air
  d / (p + w) = 84 := by
  sorry

end NUMINAMATH_CALUDE_plane_flight_time_l1980_198003


namespace NUMINAMATH_CALUDE_matrix_cube_computation_l1980_198056

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube_computation :
  A ^ 3 = !![3, -6; 6, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_computation_l1980_198056


namespace NUMINAMATH_CALUDE_square_sum_xy_l1980_198090

theorem square_sum_xy (x y a c : ℝ) (h1 : x * y = a) (h2 : 1 / x^2 + 1 / y^2 = c) :
  (x + y)^2 = a * c^2 + 2 * a := by
  sorry

end NUMINAMATH_CALUDE_square_sum_xy_l1980_198090


namespace NUMINAMATH_CALUDE_preimage_of_2_neg2_l1980_198071

-- Define the mapping f
def f (x y : ℝ) : ℝ × ℝ := (x + y, x^2 - y)

-- Define the theorem
theorem preimage_of_2_neg2 :
  ∃ (x y : ℝ), x ≥ 0 ∧ f x y = (2, -2) ∧ (x, y) = (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_2_neg2_l1980_198071


namespace NUMINAMATH_CALUDE_final_cat_count_l1980_198024

def initial_siamese : ℝ := 13.5
def initial_house : ℝ := 5.25
def cats_added : ℝ := 10.75
def cats_discounted : ℝ := 0.5

theorem final_cat_count :
  initial_siamese + initial_house + cats_added - cats_discounted = 29 := by
  sorry

end NUMINAMATH_CALUDE_final_cat_count_l1980_198024


namespace NUMINAMATH_CALUDE_rectangle_area_l1980_198037

/-- Given a rectangle divided into 18 congruent squares, where the length is three times
    the width and the diagonal of one small square is 5 cm, the area of the entire
    rectangular region is 112.5 square cm. -/
theorem rectangle_area (n m : ℕ) (s : ℝ) : 
  n * m = 18 →
  n = 2 * m →
  s^2 + s^2 = 5^2 →
  (n * s) * (m * s) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1980_198037


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1980_198001

theorem quadratic_equation_solution (x : ℝ) : 9 * x^2 - 4 = 0 ↔ x = 2/3 ∨ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1980_198001


namespace NUMINAMATH_CALUDE_missing_number_proof_l1980_198069

theorem missing_number_proof (x : ℝ) (y : ℝ) : 
  (12 + x + y + 78 + 104) / 5 = 62 ∧ 
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 → 
  y = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1980_198069
