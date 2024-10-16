import Mathlib

namespace NUMINAMATH_CALUDE_tangent_intersection_l673_67393

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_intersection (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 1 ∧ f a x₁ = a + 1) ∧
    (x₂ = -1 ∧ f a x₂ = -a - 1) ∧
    (∀ x : ℝ, f a x = (f' a x₁) * x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_l673_67393


namespace NUMINAMATH_CALUDE_books_read_total_l673_67303

def total_books (megan kelcie john greg alice : ℝ) : ℝ :=
  megan + kelcie + john + greg + alice

theorem books_read_total :
  ∀ (megan kelcie john greg alice : ℝ),
    megan = 45 →
    kelcie = megan / 3 →
    john = kelcie + 7 →
    greg = 2 * john + 11 →
    alice = 2.5 * greg - 10 →
    total_books megan kelcie john greg alice = 264.5 :=
by
  sorry

end NUMINAMATH_CALUDE_books_read_total_l673_67303


namespace NUMINAMATH_CALUDE_infinite_solutions_exist_l673_67347

theorem infinite_solutions_exist (a b c d : ℝ) : 
  ((2*a + 16*b) + (3*c - 8*d)) / 2 = 74 →
  4*a + 6*b = 9*c - 12*d →
  ∃ (f : ℝ → ℝ → ℝ), b = f a d ∧ f a d = -a/21 - 2*d/7 :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_exist_l673_67347


namespace NUMINAMATH_CALUDE_orange_grape_ratio_l673_67382

/-- Given the number of orange and grape sweets, and the number of sweets per tray,
    calculate the ratio of orange to grape sweets in each tray. -/
def sweetRatio (orange : Nat) (grape : Nat) (perTray : Nat) : Rat :=
  (orange / perTray) / (grape / perTray)

/-- Theorem stating that for 36 orange sweets and 44 grape sweets,
    when divided into trays of 4, the ratio is 9/11. -/
theorem orange_grape_ratio :
  sweetRatio 36 44 4 = 9 / 11 := by
  sorry

end NUMINAMATH_CALUDE_orange_grape_ratio_l673_67382


namespace NUMINAMATH_CALUDE_root_relationship_l673_67341

theorem root_relationship (m n a b : ℝ) : 
  (∀ x, 3 - (x - m) * (x - n) = 0 ↔ x = a ∨ x = b) →
  a < m ∧ m < n ∧ n < b :=
sorry

end NUMINAMATH_CALUDE_root_relationship_l673_67341


namespace NUMINAMATH_CALUDE_backpack_price_relation_l673_67310

theorem backpack_price_relation (x : ℝ) : x > 0 →
  (810 : ℝ) / (x + 20) = (600 : ℝ) / x * (1 - 0.1) := by
  sorry

end NUMINAMATH_CALUDE_backpack_price_relation_l673_67310


namespace NUMINAMATH_CALUDE_complex_vector_magnitude_l673_67370

/-- Given two complex numbers, prove that the magnitude of their difference is √29 -/
theorem complex_vector_magnitude (z1 z2 : ℂ) : 
  z1 = 1 - 2*I ∧ z2 = -1 + 3*I → Complex.abs (z2 - z1) = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_complex_vector_magnitude_l673_67370


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l673_67344

theorem sin_cos_difference_equals_half : 
  Real.sin (137 * π / 180) * Real.cos (13 * π / 180) - 
  Real.cos (43 * π / 180) * Real.sin (13 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l673_67344


namespace NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l673_67383

theorem abs_eq_neg_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_neg_implies_nonpositive_l673_67383


namespace NUMINAMATH_CALUDE_digit_207_is_8_l673_67384

/-- The decimal representation of 3/7 as a sequence of digits -/
def decimal_rep_3_7 : ℕ → Fin 10
  | n => sorry

/-- The length of the repeating sequence in the decimal representation of 3/7 -/
def repeat_length : ℕ := 6

/-- The 207th digit beyond the decimal point in the decimal representation of 3/7 -/
def digit_207 : Fin 10 := decimal_rep_3_7 206

theorem digit_207_is_8 : digit_207 = 8 := by sorry

end NUMINAMATH_CALUDE_digit_207_is_8_l673_67384


namespace NUMINAMATH_CALUDE_quadrangular_prism_angles_l673_67360

/-- A quadrangular prism with specific geometric properties -/
structure QuadrangularPrism where
  -- Base angles
  angleASB : ℝ
  angleDCS : ℝ
  -- Dihedral angle between SAD and SBC
  dihedralAngle : ℝ

/-- The theorem stating the possible angle measures in the quadrangular prism -/
theorem quadrangular_prism_angles (prism : QuadrangularPrism)
  (h1 : prism.angleASB = π/6)  -- 30°
  (h2 : prism.angleDCS = π/4)  -- 45°
  (h3 : prism.dihedralAngle = π/3)  -- 60°
  : (∃ (angleBSC angleASD : ℝ),
      (angleBSC = π/2 ∧ angleASD = π - Real.arccos (Real.sqrt 3 / 2)) ∨
      (angleBSC = Real.arccos (2 * Real.sqrt 2 / 3) ∧ 
       angleASD = Real.arccos (5 * Real.sqrt 3 / 9))) := by
  sorry


end NUMINAMATH_CALUDE_quadrangular_prism_angles_l673_67360


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l673_67311

theorem polynomial_product_sum (x : ℝ) : ∃ (a b c d e : ℝ),
  (2 * x^3 - 3 * x^2 + 5 * x - 1) * (8 - 3 * x) = 
    a * x^4 + b * x^3 + c * x^2 + d * x + e ∧
  16 * a + 8 * b + 4 * c + 2 * d + e = 26 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l673_67311


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l673_67320

def polynomial (a₂ : ℤ) (x : ℤ) : ℤ := x^3 + a₂ * x^2 - 7*x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (a₂ : ℤ) :
  ∀ x : ℤ, polynomial a₂ x = 0 → x ∈ possible_roots :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l673_67320


namespace NUMINAMATH_CALUDE_calculation_result_l673_67329

theorem calculation_result : (481 + 426)^2 - 4 * 481 * 426 = 3505 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l673_67329


namespace NUMINAMATH_CALUDE_problem_solution_l673_67396

/-- Equation I: 2x + y + z = 47, where x, y, z are positive integers -/
def equation_I (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ 2 * x + y + z = 47

/-- Equation II: 2x + y + z + w = 47, where x, y, z, w are positive integers -/
def equation_II (x y z w : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ 2 * x + y + z + w = 47

/-- Consecutive integers -/
def consecutive (a b c : ℕ) : Prop :=
  b = a + 1 ∧ c = a + 2

/-- Four consecutive integers -/
def consecutive_four (a b c d : ℕ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3

/-- Consecutive even integers -/
def consecutive_even (a b c : ℕ) : Prop :=
  ∃ k : ℕ, a = 2 * k ∧ b = 2 * (k + 1) ∧ c = 2 * (k + 2)

/-- Four consecutive even integers -/
def consecutive_even_four (a b c d : ℕ) : Prop :=
  ∃ k : ℕ, a = 2 * k ∧ b = 2 * (k + 1) ∧ c = 2 * (k + 2) ∧ d = 2 * (k + 3)

/-- Four consecutive odd integers -/
def consecutive_odd_four (a b c d : ℕ) : Prop :=
  ∃ k : ℕ, a = 2 * k + 1 ∧ b = 2 * k + 3 ∧ c = 2 * k + 5 ∧ d = 2 * k + 7

theorem problem_solution :
  (∃ x y z : ℕ, equation_I x y z ∧ consecutive x y z) ∧
  (∃ x y z w : ℕ, equation_II x y z w ∧ consecutive_four x y z w) ∧
  (¬ ∃ x y z : ℕ, equation_I x y z ∧ consecutive_even x y z) ∧
  (¬ ∃ x y z w : ℕ, equation_II x y z w ∧ consecutive_even_four x y z w) ∧
  (¬ ∃ x y z w : ℕ, equation_II x y z w ∧ consecutive_odd_four x y z w) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l673_67396


namespace NUMINAMATH_CALUDE_x2y_plus_1_is_third_degree_binomial_l673_67358

/-- A binomial is a polynomial with exactly two terms. -/
def is_binomial (p : Polynomial ℝ) : Prop :=
  p.support.card = 2

/-- The degree of a polynomial is the highest degree of any of its terms. -/
def polynomial_degree (p : Polynomial ℝ) : ℕ := p.natDegree

/-- A third-degree polynomial has a degree of 3. -/
def is_third_degree (p : Polynomial ℝ) : Prop :=
  polynomial_degree p = 3

theorem x2y_plus_1_is_third_degree_binomial :
  let p : Polynomial ℝ := X^2 * Y + 1
  is_binomial p ∧ is_third_degree p :=
sorry

end NUMINAMATH_CALUDE_x2y_plus_1_is_third_degree_binomial_l673_67358


namespace NUMINAMATH_CALUDE_fence_perimeter_is_262_l673_67324

/-- Calculates the outer perimeter of a rectangular fence with given specifications -/
def calculate_fence_perimeter (total_posts : ℕ) (post_width : ℚ) (post_spacing : ℕ) 
  (aspect_ratio : ℚ) : ℚ :=
  let width_posts := total_posts / (3 : ℕ)
  let length_posts := 2 * width_posts
  let width := (width_posts - 1) * post_spacing + width_posts * post_width
  let length := (length_posts - 1) * post_spacing + length_posts * post_width
  2 * (width + length)

/-- The outer perimeter of the fence with given specifications is 262 feet -/
theorem fence_perimeter_is_262 : 
  calculate_fence_perimeter 32 (1/2) 6 2 = 262 := by
  sorry

end NUMINAMATH_CALUDE_fence_perimeter_is_262_l673_67324


namespace NUMINAMATH_CALUDE_bob_local_tax_cents_l673_67366

/-- Bob's hourly wage in dollars -/
def bob_hourly_wage : ℝ := 25

/-- Local tax rate as a decimal -/
def local_tax_rate : ℝ := 0.025

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℝ := 100

/-- Theorem: The amount of Bob's hourly wage used for local taxes is 62.5 cents -/
theorem bob_local_tax_cents : 
  bob_hourly_wage * local_tax_rate * dollars_to_cents = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_bob_local_tax_cents_l673_67366


namespace NUMINAMATH_CALUDE_circle_chord_intersection_area_l673_67314

theorem circle_chord_intersection_area (r : ℝ) (chord_length : ℝ) (intersection_dist : ℝ)
  (h_r : r = 30)
  (h_chord : chord_length = 50)
  (h_dist : intersection_dist = 14) :
  ∃ (m n d : ℕ), 
    (0 < m) ∧ (0 < n) ∧ (0 < d) ∧
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ d)) ∧
    (∃ (area : ℝ), area = m * Real.pi - n * Real.sqrt d) ∧
    (m + n + d = 162) :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_intersection_area_l673_67314


namespace NUMINAMATH_CALUDE_parking_lot_cars_l673_67305

theorem parking_lot_cars (initial_cars : ℕ) (cars_left : ℕ) (extra_cars_entered : ℕ) :
  initial_cars = 80 →
  cars_left = 13 →
  extra_cars_entered = 5 →
  initial_cars - cars_left + (cars_left + extra_cars_entered) = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l673_67305


namespace NUMINAMATH_CALUDE_monochromatic_sequence_exists_l673_67386

/-- A color can be either red or blue -/
inductive Color
| red
| blue

/-- A coloring function assigns a color to each positive integer -/
def Coloring := ℕ+ → Color

/-- An infinite sequence of positive integers -/
def InfiniteSequence := ℕ → ℕ+

theorem monochromatic_sequence_exists (c : Coloring) :
  ∃ (seq : InfiniteSequence) (color : Color),
    (∀ n : ℕ, seq n < seq (n + 1)) ∧
    (∀ n : ℕ, ∃ k : ℕ+, 2 * k = seq n + seq (n + 1)) ∧
    (∀ n : ℕ, c (seq n) = color ∧ c k = color) :=
sorry

end NUMINAMATH_CALUDE_monochromatic_sequence_exists_l673_67386


namespace NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l673_67345

/-- Represents the fishing schedule in the coastal village --/
structure FishingSchedule where
  daily : Nat
  everyOtherDay : Nat
  everyThreeDay : Nat
  yesterday : Nat
  today : Nat

/-- Calculates the number of people fishing tomorrow given the fishing schedule --/
def fishersTomorrow (schedule : FishingSchedule) : Nat :=
  sorry

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow --/
theorem fifteen_fishers_tomorrow :
  let schedule := FishingSchedule.mk 7 8 3 12 10
  fishersTomorrow schedule = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_fishers_tomorrow_l673_67345


namespace NUMINAMATH_CALUDE_profit_percentage_invariance_l673_67338

theorem profit_percentage_invariance 
  (cost_price : ℝ) 
  (discount_percentage : ℝ) 
  (final_profit_percentage : ℝ) 
  (discount_percentage_pos : 0 < discount_percentage) 
  (discount_percentage_lt_100 : discount_percentage < 100) 
  (final_profit_percentage_pos : 0 < final_profit_percentage) :
  let selling_price := cost_price * (1 + final_profit_percentage / 100)
  let discounted_price := selling_price * (1 - discount_percentage / 100)
  let profit_without_discount := (selling_price - cost_price) / cost_price * 100
  profit_without_discount = final_profit_percentage := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_invariance_l673_67338


namespace NUMINAMATH_CALUDE_trig_identity_degrees_l673_67337

theorem trig_identity_degrees : 
  Real.sin ((-1200 : ℝ) * π / 180) * Real.cos ((1290 : ℝ) * π / 180) + 
  Real.cos ((-1020 : ℝ) * π / 180) * Real.sin ((-1050 : ℝ) * π / 180) = 1 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_degrees_l673_67337


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l673_67361

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 + m - 1001 = 0) → 
  (n^2 + n - 1001 = 0) → 
  m^2 + 2*m + n = 1000 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l673_67361


namespace NUMINAMATH_CALUDE_rectangle_equal_angles_l673_67355

/-- A rectangle in a 2D plane -/
structure Rectangle where
  a : ℝ  -- width
  b : ℝ  -- height
  pos_a : 0 < a
  pos_b : 0 < b

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define the angle between three points -/
def angle (A B C : Point) : ℝ := sorry

/-- Theorem: The set of points P between parallel lines AB and CD of a rectangle
    such that ∠APB = ∠CPD is the line y = b/2 -/
theorem rectangle_equal_angles (rect : Rectangle) :
  ∀ P : Point,
    0 ≤ P.y ∧ P.y ≤ rect.b →
    (angle ⟨0, 0⟩ P ⟨rect.a, 0⟩ = angle ⟨rect.a, rect.b⟩ P ⟨0, rect.b⟩) ↔
    P.y = rect.b / 2 :=
  sorry

end NUMINAMATH_CALUDE_rectangle_equal_angles_l673_67355


namespace NUMINAMATH_CALUDE_rons_current_age_l673_67356

theorem rons_current_age (maurice_current_age : ℕ) (years_from_now : ℕ) :
  maurice_current_age = 7 →
  years_from_now = 5 →
  ∃ (ron_current_age : ℕ),
    ron_current_age + years_from_now = 4 * (maurice_current_age + years_from_now) ∧
    ron_current_age = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_rons_current_age_l673_67356


namespace NUMINAMATH_CALUDE_smallest_altitude_bound_l673_67313

/-- Given a triangle with an inscribed circle of radius 1, 
    the smallest altitude of the triangle is less than or equal to 3. -/
theorem smallest_altitude_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) (r : ℝ) (hr : r = 1) :
  let s := (a + b + c) / 2
  let area := s * r
  let ha := 2 * area / a
  let hb := 2 * area / b
  let hc := 2 * area / c
  min ha (min hb hc) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_altitude_bound_l673_67313


namespace NUMINAMATH_CALUDE_probability_of_sum_seven_l673_67336

-- Define the two dice
def die1 : Finset ℕ := {1, 2, 3, 4, 5, 6}
def die2 : Finset ℕ := {2, 3, 4, 5, 6, 7}

-- Define the total outcomes
def total_outcomes : ℕ := die1.card * die2.card

-- Define the favorable outcomes (pairs that sum to 7)
def favorable_outcomes : Finset (ℕ × ℕ) := 
  {(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)}

-- Theorem statement
theorem probability_of_sum_seven :
  (favorable_outcomes.card : ℚ) / total_outcomes = 1 / 6 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_sum_seven_l673_67336


namespace NUMINAMATH_CALUDE_max_value_trigonometric_function_l673_67351

open Real

theorem max_value_trigonometric_function :
  ∃ (M : ℝ), M = 3 - 2 * sqrt 2 ∧
  ∀ θ : ℝ, 0 < θ → θ < π / 2 →
  (1 / sin θ - 1) * (1 / cos θ - 1) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_trigonometric_function_l673_67351


namespace NUMINAMATH_CALUDE_least_divisible_n_divisors_l673_67388

theorem least_divisible_n_divisors (n : ℕ) : 
  (∀ k < n, ¬(3^3 * 5^5 * 7^7 ∣ (149^k - 2^k))) →
  (3^3 * 5^5 * 7^7 ∣ (149^n - 2^n)) →
  (∀ m : ℕ, m > n → ¬(3^3 * 5^5 * 7^7 ∣ (149^m - 2^m))) →
  Nat.card {d : ℕ | d ∣ n} = 270 :=
sorry

end NUMINAMATH_CALUDE_least_divisible_n_divisors_l673_67388


namespace NUMINAMATH_CALUDE_partner_c_profit_share_l673_67300

/-- Given the investments of partners A, B, and C, and the total profit,
    calculate C's share of the profit. -/
theorem partner_c_profit_share
  (invest_a invest_b invest_c total_profit : ℝ)
  (h1 : invest_a = 3 * invest_b)
  (h2 : invest_a = 2 / 3 * invest_c)
  (h3 : total_profit = 66000) :
  (invest_c / (invest_a + invest_b + invest_c)) * total_profit = (9 / 17) * 66000 :=
by sorry

end NUMINAMATH_CALUDE_partner_c_profit_share_l673_67300


namespace NUMINAMATH_CALUDE_competition_participants_l673_67371

theorem competition_participants : ∀ (initial : ℕ),
  (initial : ℚ) * (1 - 0.6) * (1 / 4) = 30 →
  initial = 300 := by
sorry

end NUMINAMATH_CALUDE_competition_participants_l673_67371


namespace NUMINAMATH_CALUDE_average_temperature_l673_67375

def temperature_day1 : ℤ := -14
def temperature_day2 : ℤ := -8
def temperature_day3 : ℤ := 1
def num_days : ℕ := 3

theorem average_temperature :
  (temperature_day1 + temperature_day2 + temperature_day3) / num_days = -7 :=
by sorry

end NUMINAMATH_CALUDE_average_temperature_l673_67375


namespace NUMINAMATH_CALUDE_custom_operation_equation_l673_67331

-- Define the custom operation *
def star (a b : ℤ) : ℤ := 2 * a + b

-- State the theorem
theorem custom_operation_equation :
  ∃ x : ℤ, star 3 (star 4 x) = -1 ∧ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equation_l673_67331


namespace NUMINAMATH_CALUDE_valid_seating_count_l673_67327

/-- Number of seats in a row -/
def num_seats : ℕ := 7

/-- Number of people to be seated -/
def num_people : ℕ := 4

/-- Number of adjacent unoccupied seats -/
def num_adjacent_empty : ℕ := 2

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (seats : ℕ) (people : ℕ) (adjacent_empty : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid seating arrangements -/
theorem valid_seating_count :
  seating_arrangements num_seats num_people num_adjacent_empty = 336 :=
sorry

end NUMINAMATH_CALUDE_valid_seating_count_l673_67327


namespace NUMINAMATH_CALUDE_greatest_x_value_l673_67343

theorem greatest_x_value : 
  let f (x : ℝ) := ((5*x - 20) / (4*x - 5))^2 + (5*x - 20) / (4*x - 5)
  ∃ (x_max : ℝ), x_max = 50/29 ∧ 
    (∀ (x : ℝ), f x = 18 → x ≤ x_max) ∧
    (f x_max = 18) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l673_67343


namespace NUMINAMATH_CALUDE_polynomial_identity_l673_67323

theorem polynomial_identity : ∀ x : ℝ, 
  (x^2 + 3*x + 2) * (x + 3) = (x + 1) * (x^2 + 5*x + 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l673_67323


namespace NUMINAMATH_CALUDE_min_distance_to_line_l673_67378

-- Define the right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ c > a ∧ c > b ∧ a^2 + b^2 = c^2

-- Define the point (m, n) on the line ax + by + c = 0
def point_on_line (a b c m n : ℝ) : Prop :=
  a * m + b * n + c = 0

-- Theorem statement
theorem min_distance_to_line (a b c m n : ℝ) 
  (h1 : is_right_triangle a b c) 
  (h2 : point_on_line a b c m n) : 
  m^2 + n^2 ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l673_67378


namespace NUMINAMATH_CALUDE_f_minimum_at_cos2x_neg_half_l673_67332

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * Real.cos (2 * x) + 16) - Real.sin x ^ 2

theorem f_minimum_at_cos2x_neg_half :
  ∀ x : ℝ, f x ≥ 0 ∧ (f x = 0 ↔ Real.cos (2 * x) = -1/2) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_at_cos2x_neg_half_l673_67332


namespace NUMINAMATH_CALUDE_remainder_problem_l673_67333

theorem remainder_problem (x : ℤ) : x % 82 = 5 → (x + 7) % 41 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l673_67333


namespace NUMINAMATH_CALUDE_statement_A_statement_B_statement_C_statement_D_l673_67325

-- Define the curve C
def C (m n x y : ℝ) : Prop := m * x^2 + n * y^2 = 1

-- Statement A
theorem statement_A (m n : ℝ) (h1 : n > m) (h2 : m > 0) :
  ¬ (∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ∀ x y : ℝ, C m n x y ↔ (x^2 / a^2 + y^2 / b^2 = 1) ∧
    (∃ c : ℝ, c > 0 ∧ a^2 = b^2 + c^2)) :=
sorry

-- Statement B
theorem statement_B (n : ℝ) (h : n > 0) :
  ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ ∀ x : ℝ, C 0 n x y1 ∧ C 0 n x y2 :=
sorry

-- Statement C
theorem statement_C (m n : ℝ) (h : m * n < 0) :
  ∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, C m n x y →
    (y - k * x) * (y + k * x) ≤ 0 ∧ k^2 = -m/n :=
sorry

-- Statement D
theorem statement_D (n : ℝ) (h : n > 0) :
  ¬ (∀ x y : ℝ, C n n x y ↔ x^2 + y^2 = n) :=
sorry

end NUMINAMATH_CALUDE_statement_A_statement_B_statement_C_statement_D_l673_67325


namespace NUMINAMATH_CALUDE_sphere_radius_in_cube_l673_67301

/-- The radius of spheres packed in a cube -/
theorem sphere_radius_in_cube (n : ℕ) (side_length : ℝ) (radius : ℝ) : 
  n = 8 →  -- There are 8 spheres
  side_length = 2 →  -- The cube has side length 2
  radius > 0 →  -- The radius is positive
  (2 * radius = side_length / 2 + radius) →  -- Condition for spheres to be tangent
  radius = 1 := by sorry

end NUMINAMATH_CALUDE_sphere_radius_in_cube_l673_67301


namespace NUMINAMATH_CALUDE_copper_carbonate_molecular_weight_l673_67381

/-- The molecular weight of Copper(II) carbonate for a given number of moles -/
def molecular_weight (moles : ℝ) : ℝ := sorry

/-- Theorem: The molecular weight of one mole of Copper(II) carbonate is 124 grams/mole -/
theorem copper_carbonate_molecular_weight :
  molecular_weight 1 = 124 :=
by
  have h : molecular_weight 8 = 992 := sorry
  sorry

end NUMINAMATH_CALUDE_copper_carbonate_molecular_weight_l673_67381


namespace NUMINAMATH_CALUDE_solve_linear_equation_l673_67372

theorem solve_linear_equation (x : ℝ) (h : x - 3*x + 5*x = 150) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l673_67372


namespace NUMINAMATH_CALUDE_odd_function_condition_l673_67357

/-- Given a > 1, f(x) = (a^x / (a^x - 1)) + m is an odd function if and only if m = -1/2 -/
theorem odd_function_condition (a : ℝ) (h : a > 1) :
  ∃ m : ℝ, ∀ x : ℝ, x ≠ 0 →
    (fun x : ℝ => (a^x / (a^x - 1)) + m) x = -((fun x : ℝ => (a^x / (a^x - 1)) + m) (-x)) ↔
    m = -1/2 := by sorry

end NUMINAMATH_CALUDE_odd_function_condition_l673_67357


namespace NUMINAMATH_CALUDE_shaded_area_proof_l673_67317

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem shaded_area_proof : U \ (A ∪ B) = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l673_67317


namespace NUMINAMATH_CALUDE_max_daily_net_income_l673_67377

/-- Represents the daily rental fee for each electric car -/
def x : ℕ → ℕ := fun n => n

/-- Represents the daily net income from renting out electric cars -/
def y : ℕ → ℤ
| n =>
  if 60 ≤ n ∧ n ≤ 90 then
    750 * n - 1700
  else if 90 < n ∧ n ≤ 300 then
    -3 * n * n + 1020 * n - 1700
  else
    0

/-- The theorem stating the maximum daily net income and the corresponding rental fee -/
theorem max_daily_net_income :
  ∃ (n : ℕ), 60 ≤ n ∧ n ≤ 300 ∧ y n = 85000 ∧ n = 170 ∧
  ∀ (m : ℕ), 60 ≤ m ∧ m ≤ 300 → y m ≤ y n :=
sorry

end NUMINAMATH_CALUDE_max_daily_net_income_l673_67377


namespace NUMINAMATH_CALUDE_f_min_at_two_l673_67390

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem f_min_at_two :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 2 :=
sorry

end NUMINAMATH_CALUDE_f_min_at_two_l673_67390


namespace NUMINAMATH_CALUDE_hyperbola_equation_l673_67339

/-- Given a hyperbola with the general equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is x - √3 y = 0 and one of its foci is on the directrix
    of the parabola y² = -4x, then its equation is 4/3 x² - 4y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (asymptote : ∀ x y : ℝ, x - Real.sqrt 3 * y = 0 → x^2 / a^2 - y^2 / b^2 = 1)
  (focus_on_directrix : ∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ y^2 = -4*x ∧ x = 1) :
  ∀ x y : ℝ, 4/3 * x^2 - 4 * y^2 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l673_67339


namespace NUMINAMATH_CALUDE_no_solution_arctan_equation_l673_67395

theorem no_solution_arctan_equation :
  ¬ ∃ (x : ℝ), x > 0 ∧ Real.arctan (1 / x^2) + Real.arctan (1 / x^4) = π / 4 := by
sorry

end NUMINAMATH_CALUDE_no_solution_arctan_equation_l673_67395


namespace NUMINAMATH_CALUDE_line_polar_equation_l673_67374

-- Define the line in Cartesian coordinates
def line (x y : ℝ) : Prop := (Real.sqrt 3 / 3) * x - y = 0

-- Define the polar coordinates
def polar_coords (ρ θ : ℝ) (x y : ℝ) : Prop :=
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ ρ ≥ 0

-- State the theorem
theorem line_polar_equation :
  ∀ ρ θ x y : ℝ,
  polar_coords ρ θ x y →
  line x y →
  (θ = π / 6 ∨ θ = 7 * π / 6) :=
sorry

end NUMINAMATH_CALUDE_line_polar_equation_l673_67374


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_multiple_l673_67380

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem consecutive_odd_numbers_multiple (a b c : ℤ) : 
  is_odd a ∧ is_odd b ∧ is_odd c ∧  -- Three odd numbers
  b = a + 2 ∧ c = b + 2 ∧           -- Consecutive
  a = 7 ∧                           -- First number is 7
  ∃ m : ℤ, 8 * a = 3 * c + 5 + m * b -- Equation condition
  → m = 2 :=                        -- Multiple of second number is 2
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_multiple_l673_67380


namespace NUMINAMATH_CALUDE_integer_root_count_l673_67350

theorem integer_root_count : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, ∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) ∧ S.card = 12 :=
sorry

end NUMINAMATH_CALUDE_integer_root_count_l673_67350


namespace NUMINAMATH_CALUDE_unique_positive_solution_l673_67399

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l673_67399


namespace NUMINAMATH_CALUDE_uncovered_area_square_in_square_l673_67315

theorem uncovered_area_square_in_square (large_side : ℝ) (small_side : ℝ) :
  large_side = 10 →
  small_side = 4 →
  large_side ^ 2 - small_side ^ 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_square_in_square_l673_67315


namespace NUMINAMATH_CALUDE_hannah_easter_eggs_l673_67322

theorem hannah_easter_eggs (total : ℕ) (h : total = 63) :
  ∃ (helen : ℕ) (hannah : ℕ),
    hannah = 2 * helen ∧
    hannah + helen = total ∧
    hannah = 42 := by
  sorry

end NUMINAMATH_CALUDE_hannah_easter_eggs_l673_67322


namespace NUMINAMATH_CALUDE_calculate_x_l673_67353

theorem calculate_x : ∀ (w y z x : ℕ),
  w = 90 →
  z = w + 25 →
  y = z + 15 →
  x = y + 7 →
  x = 137 := by
  sorry

end NUMINAMATH_CALUDE_calculate_x_l673_67353


namespace NUMINAMATH_CALUDE_train_length_speed_relation_l673_67312

/-- Represents the properties of a train and its movement. -/
structure Train where
  length : ℝ
  speed : ℝ
  platform_crossing_time : ℝ
  pole_crossing_time : ℝ

/-- Theorem stating the relationship between train length and speed. -/
theorem train_length_speed_relation (t : Train) 
  (h1 : t.platform_crossing_time = 40)
  (h2 : t.pole_crossing_time = 20)
  (h3 : t.length = t.speed * t.pole_crossing_time)
  (h4 : 2 * t.length = t.speed * t.platform_crossing_time) :
  t.length = 20 * t.speed :=
by sorry

end NUMINAMATH_CALUDE_train_length_speed_relation_l673_67312


namespace NUMINAMATH_CALUDE_triangle_perimeter_l673_67363

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A →
  a = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l673_67363


namespace NUMINAMATH_CALUDE_one_pair_probability_l673_67318

/-- Represents the number of socks -/
def total_socks : ℕ := 10

/-- Represents the number of colors -/
def num_colors : ℕ := 5

/-- Represents the number of socks per color -/
def socks_per_color : ℕ := 2

/-- Represents the number of socks to be selected -/
def socks_selected : ℕ := 4

/-- Calculates the probability of selecting exactly one pair of socks of the same color -/
def prob_one_pair : ℚ := 4 / 7

/-- Proves that the probability of selecting exactly one pair of socks of the same color
    when randomly choosing 4 socks from a set of 10 socks (2 of each of 5 colors) is 4/7 -/
theorem one_pair_probability : 
  total_socks = num_colors * socks_per_color ∧ 
  socks_selected = 4 → 
  prob_one_pair = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_one_pair_probability_l673_67318


namespace NUMINAMATH_CALUDE_order_of_abc_l673_67335

theorem order_of_abc : ∀ (a b c : ℝ),
  a = 0.1 * Real.exp 0.1 →
  b = 1 / 9 →
  c = -Real.log 0.9 →
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_order_of_abc_l673_67335


namespace NUMINAMATH_CALUDE_subtractions_to_additions_theorem_l673_67319

-- Define the original expression
def original_expression : List ℤ := [6, -3, 7, -2]

-- Define the operation of changing subtractions to additions
def change_subtractions_to_additions (expr : List ℤ) : List ℤ :=
  expr.map (λ x => if x < 0 then -x else x)

-- Define the result of the operation
def result_expression : List ℤ := [6, -3, 7, -2]

-- State the theorem
theorem subtractions_to_additions_theorem :
  change_subtractions_to_additions original_expression = result_expression :=
sorry

end NUMINAMATH_CALUDE_subtractions_to_additions_theorem_l673_67319


namespace NUMINAMATH_CALUDE_favorite_fruit_apples_l673_67308

theorem favorite_fruit_apples (total students_oranges students_pears students_strawberries : ℕ) 
  (h1 : total = 450)
  (h2 : students_oranges = 70)
  (h3 : students_pears = 120)
  (h4 : students_strawberries = 113) :
  total - (students_oranges + students_pears + students_strawberries) = 147 := by
  sorry

end NUMINAMATH_CALUDE_favorite_fruit_apples_l673_67308


namespace NUMINAMATH_CALUDE_jaydee_typing_time_l673_67348

/-- Calculates the time needed to type a research paper given specific conditions. -/
def time_to_type_paper (words_per_minute : ℕ) (break_interval : ℕ) (break_duration : ℕ) 
  (words_per_mistake : ℕ) (mistake_correction_time : ℕ) (total_words : ℕ) : ℕ :=
  let typing_time := (total_words + words_per_minute - 1) / words_per_minute
  let breaks := typing_time / break_interval
  let break_time := breaks * break_duration
  let mistakes := (total_words + words_per_mistake - 1) / words_per_mistake
  let correction_time := mistakes * mistake_correction_time
  let total_minutes := typing_time + break_time + correction_time
  (total_minutes + 59) / 60

/-- Theorem stating that Jaydee will take 6 hours to type the research paper. -/
theorem jaydee_typing_time : 
  time_to_type_paper 32 25 5 100 1 7125 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jaydee_typing_time_l673_67348


namespace NUMINAMATH_CALUDE_brendan_recharge_ratio_l673_67340

/-- Represents the financial data for Brendan's June earnings and expenses -/
structure FinancialData where
  totalEarnings : ℕ
  carCost : ℕ
  remainingMoney : ℕ

/-- Calculates the amount recharged on the debit card -/
def amountRecharged (data : FinancialData) : ℕ :=
  data.totalEarnings - data.carCost - data.remainingMoney

/-- Represents a ratio as a pair of natural numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Calculates the ratio of recharged amount to total earnings -/
def rechargeRatio (data : FinancialData) : Ratio :=
  let recharged := amountRecharged data
  let gcd := Nat.gcd recharged data.totalEarnings
  { numerator := recharged / gcd, denominator := data.totalEarnings / gcd }

/-- Theorem stating that Brendan's recharge ratio is 1:2 -/
theorem brendan_recharge_ratio :
  let data : FinancialData := { totalEarnings := 5000, carCost := 1500, remainingMoney := 1000 }
  let ratio := rechargeRatio data
  ratio.numerator = 1 ∧ ratio.denominator = 2 := by sorry


end NUMINAMATH_CALUDE_brendan_recharge_ratio_l673_67340


namespace NUMINAMATH_CALUDE_four_digit_number_property_l673_67376

theorem four_digit_number_property : ∃ (a b c d : ℕ), 
  (a ≥ 1 ∧ a ≤ 9) ∧ 
  (b ≥ 0 ∧ b ≤ 9) ∧ 
  (c ≥ 0 ∧ c ≤ 9) ∧ 
  (d ≥ 0 ∧ d ≤ 9) ∧ 
  (a * 1000 + b * 100 + c * 10 + d ≥ 1000) ∧
  (a * 1000 + b * 100 + c * 10 + d ≤ 9999) ∧
  ((a + b + c + d) * (a * b * c * d) = 3990) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_property_l673_67376


namespace NUMINAMATH_CALUDE_cube_edge_length_equality_l673_67307

theorem cube_edge_length_equality (a : ℝ) : 
  let parallelepiped_volume : ℝ := 2 * 3 * 6
  let parallelepiped_surface_area : ℝ := 2 * (2 * 3 + 3 * 6 + 2 * 6)
  let cube_volume : ℝ := a^3
  let cube_surface_area : ℝ := 6 * a^2
  (parallelepiped_volume / cube_volume = parallelepiped_surface_area / cube_surface_area) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_length_equality_l673_67307


namespace NUMINAMATH_CALUDE_unique_digit_divisibility_l673_67302

theorem unique_digit_divisibility : ∃! (B : ℕ), B < 10 ∧ 45 % B = 0 ∧ (451 * 10 + B * 1 + 7) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_divisibility_l673_67302


namespace NUMINAMATH_CALUDE_train_cost_XY_is_900_l673_67342

/-- Represents the cost of a train journey in dollars -/
def train_cost (distance : ℝ) : ℝ := 0.20 * distance

/-- The cities and their distances -/
structure Cities where
  XY : ℝ
  XZ : ℝ

/-- The problem setup -/
def piravena_journey : Cities where
  XY := 4500
  XZ := 4000

theorem train_cost_XY_is_900 :
  train_cost piravena_journey.XY = 900 := by sorry

end NUMINAMATH_CALUDE_train_cost_XY_is_900_l673_67342


namespace NUMINAMATH_CALUDE_min_value_a_l673_67365

theorem min_value_a : 
  (∀ (x y : ℝ), x > 0 → y > 0 → x + Real.sqrt (x * y) ≤ a * (x + y)) → 
  a ≥ (Real.sqrt 2 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_a_l673_67365


namespace NUMINAMATH_CALUDE_correct_number_of_choices_l673_67346

/-- Represents the number of junior boys or girls -/
def num_juniors : ℕ := 7

/-- Represents the number of senior boys or girls -/
def num_seniors : ℕ := 8

/-- Represents the number of genders (boys and girls) -/
def num_genders : ℕ := 2

/-- Calculates the number of ways to choose a president and vice-president -/
def ways_to_choose_leaders : ℕ :=
  num_genders * (num_juniors * num_seniors + num_seniors * num_juniors)

/-- Theorem stating that the number of ways to choose leaders is 224 -/
theorem correct_number_of_choices : ways_to_choose_leaders = 224 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_choices_l673_67346


namespace NUMINAMATH_CALUDE_woodworker_solution_l673_67359

/-- Represents the number of furniture items made by a woodworker. -/
structure FurnitureCount where
  chairs : ℕ
  tables : ℕ
  cabinets : ℕ

/-- Calculates the total number of legs used for a given furniture count. -/
def totalLegs (f : FurnitureCount) : ℕ :=
  4 * f.chairs + 4 * f.tables + 2 * f.cabinets

/-- The woodworker's furniture count satisfies the given conditions. -/
def isSolution (f : FurnitureCount) : Prop :=
  f.chairs = 6 ∧ f.cabinets = 4 ∧ totalLegs f = 80

theorem woodworker_solution :
  ∃ f : FurnitureCount, isSolution f ∧ f.tables = 12 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_solution_l673_67359


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l673_67397

theorem greatest_two_digit_multiple_of_17 : ∀ n : ℕ, 
  n ≤ 99 → n ≥ 10 → n % 17 = 0 → n ≤ 85 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l673_67397


namespace NUMINAMATH_CALUDE_unique_solution_iff_p_eq_neg_four_thirds_l673_67354

/-- The equation has exactly one solution if and only if p = -4/3 -/
theorem unique_solution_iff_p_eq_neg_four_thirds :
  (∃! x : ℝ, (2 * x + 3) / (p * x - 2) = x) ↔ p = -4/3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_p_eq_neg_four_thirds_l673_67354


namespace NUMINAMATH_CALUDE_other_communities_count_l673_67328

theorem other_communities_count (total_boys : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) :
  total_boys = 850 →
  muslim_percent = 34/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  (total_boys : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 238 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l673_67328


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l673_67392

theorem factorization_difference_of_squares (a b : ℝ) : a^2 * b^2 - 9 = (a*b + 3) * (a*b - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l673_67392


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l673_67389

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x^3 + 4*x^2 + 2*x + 1) * (y^3 + 4*y^2 + 2*y + 1) * (z^3 + 4*z^2 + 2*z + 1)) / (x*y*z) ≥ 1331 :=
by sorry

theorem min_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  ((x^3 + 4*x^2 + 2*x + 1) * (y^3 + 4*y^2 + 2*y + 1) * (z^3 + 4*z^2 + 2*z + 1)) / (x*y*z) = 1331 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l673_67389


namespace NUMINAMATH_CALUDE_least_difference_nm_l673_67385

/-- Given a triangle ABC with sides AB = x+6, AC = 4x, BC = x+12, prove that the least possible 
    value of n-m is 2.5, where m and n are defined such that 1.5 < x < 4, m = 1.5, and n = 4. -/
theorem least_difference_nm (x : ℝ) (m n : ℝ) : 
  x > 0 ∧ 
  (x + 6) + 4*x > (x + 12) ∧
  (x + 6) + (x + 12) > 4*x ∧
  4*x + (x + 12) > (x + 6) ∧
  x + 12 > x + 6 ∧
  x + 12 > 4*x ∧
  m = 1.5 ∧
  n = 4 ∧
  1.5 < x ∧
  x < 4 →
  n - m = 2.5 := by
sorry

end NUMINAMATH_CALUDE_least_difference_nm_l673_67385


namespace NUMINAMATH_CALUDE_polynomial_factorization_l673_67364

theorem polynomial_factorization (x : ℝ) :
  x^4 + 2021*x^2 + 2020*x + 2021 = (x^2 + x + 1)*(x^2 - x + 2021) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l673_67364


namespace NUMINAMATH_CALUDE_inequality_proof_l673_67330

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a^4 / (4*a^4 + b^4 + c^4)) + (b^4 / (a^4 + 4*b^4 + c^4)) + (c^4 / (a^4 + b^4 + 4*c^4)) ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l673_67330


namespace NUMINAMATH_CALUDE_smallest_multiple_l673_67398

theorem smallest_multiple : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (∃ (k : ℕ), a = 5 * k) ∧ 
  (∃ (m : ℕ), a + 1 = 7 * m) ∧ 
  (∃ (n : ℕ), a + 2 = 9 * n) ∧ 
  (∃ (p : ℕ), a + 3 = 11 * p) ∧ 
  (∀ (b : ℕ), 
    (b > 0) ∧ 
    (∃ (k : ℕ), b = 5 * k) ∧ 
    (∃ (m : ℕ), b + 1 = 7 * m) ∧ 
    (∃ (n : ℕ), b + 2 = 9 * n) ∧ 
    (∃ (p : ℕ), b + 3 = 11 * p) 
    → b ≥ a) ∧
  a = 1735 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l673_67398


namespace NUMINAMATH_CALUDE_no_divisible_by_nine_l673_67316

def base_n_number (n : ℕ) : ℕ := 3 + 2*n + 1*n^2 + 0*n^3 + 3*n^4 + 2*n^5

theorem no_divisible_by_nine :
  ∀ n : ℕ, 2 ≤ n → n ≤ 100 → ¬(base_n_number n % 9 = 0) := by
sorry

end NUMINAMATH_CALUDE_no_divisible_by_nine_l673_67316


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l673_67304

def complex (a b : ℝ) := a + b * Complex.I

theorem condition_necessary_not_sufficient :
  ∃ a b : ℝ, (complex a b)^2 = 2 * Complex.I ∧ (a ≠ 1 ∨ b ≠ 1) ∧
  ∀ a b : ℝ, (complex a b)^2 = 2 * Complex.I → a = 1 ∧ b = 1 :=
sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l673_67304


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l673_67306

theorem not_all_perfect_squares (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧ ¬∃ (k : ℕ), a * b - 1 = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l673_67306


namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l673_67309

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = -3)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -346 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l673_67309


namespace NUMINAMATH_CALUDE_cubic_function_properties_l673_67334

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_function_properties :
  ∀ (a b c : ℝ),
  (∀ x : ℝ, f a b c x ≤ f a b c (-1)) ∧  -- Maximum at x = -1
  (f a b c (-1) = 7) ∧                   -- Maximum value is 7
  (∀ x : ℝ, f a b c x ≥ f a b c 3) →     -- Minimum at x = 3
  (a = -3 ∧ b = -9 ∧ c = 2 ∧ f a b c 3 = -25) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l673_67334


namespace NUMINAMATH_CALUDE_heap_sheet_count_l673_67373

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The number of sheets in a bunch -/
def sheets_per_bunch : ℕ := 4

/-- The number of sheets in a bundle -/
def sheets_per_bundle : ℕ := 2

/-- The total number of sheets removed -/
def total_sheets_removed : ℕ := 114

/-- The number of sheets in a heap -/
def sheets_per_heap : ℕ := 20

theorem heap_sheet_count :
  sheets_per_heap = 
    (total_sheets_removed - 
      (colored_bundles * sheets_per_bundle + 
       white_bunches * sheets_per_bunch)) / scrap_heaps :=
by sorry

end NUMINAMATH_CALUDE_heap_sheet_count_l673_67373


namespace NUMINAMATH_CALUDE_inequality_condition_l673_67394

theorem inequality_condition (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 2| + |x + 1| < b) ↔ b > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_condition_l673_67394


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l673_67368

/-- Given a sphere O with radius R and a plane perpendicular to a radius OP at its midpoint M,
    intersecting the sphere to form a circle O₁, the volume ratio of the sphere with O₁ as its
    great circle to sphere O is 3/8 * √3. -/
theorem sphere_volume_ratio (R : ℝ) (h : R > 0) : 
  let r := R * (Real.sqrt 3 / 2)
  (4 / 3 * Real.pi * r^3) / (4 / 3 * Real.pi * R^3) = 3 / 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l673_67368


namespace NUMINAMATH_CALUDE_four_squares_power_of_two_l673_67321

def count_four_squares (n : ℕ) : ℕ :=
  if n % 2 = 0 then 1 else 0

theorem four_squares_power_of_two (n : ℕ) :
  count_four_squares n = (Nat.card {(a, b, c, d) : ℕ × ℕ × ℕ × ℕ | a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a^2 + b^2 + c^2 + d^2 = 2^n}) :=
sorry

end NUMINAMATH_CALUDE_four_squares_power_of_two_l673_67321


namespace NUMINAMATH_CALUDE_six_fold_application_of_f_on_four_l673_67369

noncomputable def f (x : ℝ) : ℝ := -1 / x

theorem six_fold_application_of_f_on_four (h : ∀ (x : ℝ), x ≠ 0 → f x = -1 / x) :
  f (f (f (f (f (f 4))))) = 4 :=
by sorry

end NUMINAMATH_CALUDE_six_fold_application_of_f_on_four_l673_67369


namespace NUMINAMATH_CALUDE_blue_face_probability_l673_67349

/-- A cube with colored faces -/
structure ColoredCube where
  blue_faces : ℕ
  red_faces : ℕ

/-- The probability of rolling a specific color on a colored cube -/
def roll_probability (cube : ColoredCube) (color : String) : ℚ :=
  match color with
  | "blue" => cube.blue_faces / (cube.blue_faces + cube.red_faces)
  | "red" => cube.red_faces / (cube.blue_faces + cube.red_faces)
  | _ => 0

/-- Theorem: The probability of rolling a blue face on a cube with 3 blue faces and 3 red faces is 1/2 -/
theorem blue_face_probability :
  ∀ (cube : ColoredCube),
    cube.blue_faces = 3 →
    cube.red_faces = 3 →
    roll_probability cube "blue" = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_face_probability_l673_67349


namespace NUMINAMATH_CALUDE_intersection_segment_length_l673_67387

/-- Line l in the Cartesian coordinate system -/
def line_l (x y : ℝ) : Prop := x + y = 3

/-- Curve C in the Cartesian coordinate system -/
def curve_C (x y : ℝ) : Prop := y = (x - 3)^2

/-- The intersection points of line l and curve C -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | line_l p.1 p.2 ∧ curve_C p.1 p.2}

/-- Theorem stating that the length of the line segment between 
    the intersection points of line l and curve C is √2 -/
theorem intersection_segment_length : 
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ 
  A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l673_67387


namespace NUMINAMATH_CALUDE_jacks_total_yen_l673_67391

/-- Represents the amount of money in different currencies -/
structure Money where
  pounds : ℕ
  euros : ℕ
  yen : ℕ

/-- Represents currency exchange rates -/
structure ExchangeRates where
  pounds_per_euro : ℕ
  yen_per_pound : ℕ

/-- Calculates the total amount in yen given initial amounts and exchange rates -/
def total_in_yen (initial : Money) (rates : ExchangeRates) : ℕ :=
  (initial.pounds + initial.euros * rates.pounds_per_euro) * rates.yen_per_pound + initial.yen

/-- Theorem stating that Jack's total amount in yen is 9400 -/
theorem jacks_total_yen :
  let initial : Money := { pounds := 42, euros := 11, yen := 3000 }
  let rates : ExchangeRates := { pounds_per_euro := 2, yen_per_pound := 100 }
  total_in_yen initial rates = 9400 := by
  sorry


end NUMINAMATH_CALUDE_jacks_total_yen_l673_67391


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l673_67379

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2 * Complex.I) / (1 + Complex.I) = Complex.mk a b → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l673_67379


namespace NUMINAMATH_CALUDE_concert_attendees_l673_67367

theorem concert_attendees :
  let num_buses : ℕ := 8
  let students_per_bus : ℕ := 45
  let chaperones_per_bus : List ℕ := [2, 3, 4, 5, 3, 4, 2, 6]
  let total_students : ℕ := num_buses * students_per_bus
  let total_chaperones : ℕ := chaperones_per_bus.sum
  let total_attendees : ℕ := total_students + total_chaperones
  total_attendees = 389 := by
  sorry


end NUMINAMATH_CALUDE_concert_attendees_l673_67367


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l673_67352

theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x - a = 0 ∧ y^2 - 2*y - a = 0) ↔ a > -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l673_67352


namespace NUMINAMATH_CALUDE_score_difference_l673_67362

/-- Represents the test scores of three students -/
structure TestScores where
  meghan : ℕ
  jose : ℕ
  alisson : ℕ

/-- The properties of the test and scores -/
def ValidTestScores (s : TestScores) : Prop :=
  let totalQuestions : ℕ := 50
  let marksPerQuestion : ℕ := 2
  let maxScore : ℕ := totalQuestions * marksPerQuestion
  let wrongQuestions : ℕ := 5
  (s.jose = maxScore - wrongQuestions * marksPerQuestion) ∧ 
  (s.jose = s.alisson + 40) ∧
  (s.meghan + s.jose + s.alisson = 210) ∧
  (s.meghan < s.jose)

/-- The theorem stating the difference between Jose's and Meghan's scores -/
theorem score_difference (s : TestScores) (h : ValidTestScores s) : 
  s.jose - s.meghan = 20 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_l673_67362


namespace NUMINAMATH_CALUDE_gcf_of_120_180_240_l673_67326

theorem gcf_of_120_180_240 : Nat.gcd 120 (Nat.gcd 180 240) = 60 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_120_180_240_l673_67326
