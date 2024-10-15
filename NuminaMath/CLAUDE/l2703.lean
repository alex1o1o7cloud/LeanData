import Mathlib

namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l2703_270374

theorem sum_and_ratio_to_difference (a b : ℝ) 
  (h1 : a + b = 500) 
  (h2 : a / b = 0.8) : 
  b - a = 100 / 1.8 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l2703_270374


namespace NUMINAMATH_CALUDE_trapezoid_perimeters_l2703_270325

/-- A trapezoid with given measurements -/
structure Trapezoid where
  longerBase : ℝ
  height : ℝ
  leg1 : ℝ
  leg2 : ℝ

/-- The set of possible perimeters for a given trapezoid -/
def possiblePerimeters (t : Trapezoid) : Set ℝ :=
  {p | ∃ shorterBase : ℝ, 
    p = t.longerBase + t.leg1 + t.leg2 + shorterBase ∧
    shorterBase > 0 ∧
    (shorterBase = t.longerBase - Real.sqrt (t.leg1^2 - t.height^2) - Real.sqrt (t.leg2^2 - t.height^2) ∨
     shorterBase = t.longerBase + Real.sqrt (t.leg1^2 - t.height^2) - Real.sqrt (t.leg2^2 - t.height^2))}

/-- The theorem to be proved -/
theorem trapezoid_perimeters (t : Trapezoid) 
  (h1 : t.longerBase = 30)
  (h2 : t.height = 24)
  (h3 : t.leg1 = 25)
  (h4 : t.leg2 = 30) :
  possiblePerimeters t = {90, 104} := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeters_l2703_270325


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_4_range_of_a_for_f_geq_abs_a_plus_1_l2703_270353

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + 2 * |x - 1|

-- Theorem for part I
theorem solution_set_f_less_than_4 :
  {x : ℝ | f x < 4} = {x : ℝ | -1 < x ∧ x < 5/3} := by sorry

-- Theorem for part II
theorem range_of_a_for_f_geq_abs_a_plus_1 :
  (∀ x : ℝ, f x ≥ |a + 1|) ↔ -3 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_4_range_of_a_for_f_geq_abs_a_plus_1_l2703_270353


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2703_270381

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Checks if a point (x, y) lies on the quadratic function -/
def QuadraticFunction.passesThrough (f : QuadraticFunction) (x y : ℝ) : Prop :=
  f.evaluate x = y

theorem quadratic_function_theorem (f : QuadraticFunction) :
  f.c = -3 →
  f.passesThrough 2 (-3) →
  f.passesThrough (-1) 0 →
  (f.a = 1 ∧ f.b = -2) ∧
  (∃ k : ℝ, k = 4 ∧
    (∀ x : ℝ, (f.evaluate x + k = 0) → (∀ y : ℝ, y ≠ x → f.evaluate y + k ≠ 0))) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2703_270381


namespace NUMINAMATH_CALUDE_jillian_oranges_l2703_270359

/-- Given that Jillian divides oranges into pieces for her friends, 
    this theorem proves the number of oranges she had. -/
theorem jillian_oranges 
  (pieces_per_orange : ℕ) 
  (pieces_per_friend : ℕ) 
  (num_friends : ℕ) 
  (h1 : pieces_per_orange = 10) 
  (h2 : pieces_per_friend = 4) 
  (h3 : num_friends = 200) : 
  (num_friends * pieces_per_friend) / pieces_per_orange = 80 := by
  sorry

end NUMINAMATH_CALUDE_jillian_oranges_l2703_270359


namespace NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l2703_270345

/-- The volume of a cylinder formed by rotating a square about its horizontal line of symmetry. -/
theorem volume_cylinder_from_square_rotation (side_length : ℝ) (h_positive : side_length > 0) :
  let radius : ℝ := side_length / 2
  let height : ℝ := side_length
  let volume : ℝ := π * radius ^ 2 * height
  side_length = 10 → volume = 250 * π := by sorry

end NUMINAMATH_CALUDE_volume_cylinder_from_square_rotation_l2703_270345


namespace NUMINAMATH_CALUDE_remaining_amount_for_seat_and_tape_l2703_270314

def initial_amount : ℕ := 60
def frame_cost : ℕ := 15
def wheel_cost : ℕ := 25

theorem remaining_amount_for_seat_and_tape : 
  initial_amount - (frame_cost + wheel_cost) = 20 := by
sorry

end NUMINAMATH_CALUDE_remaining_amount_for_seat_and_tape_l2703_270314


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l2703_270316

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  let C : Matrix (Fin 2) (Fin 2) ℤ := !![23, -7; 24, -16]
  A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l2703_270316


namespace NUMINAMATH_CALUDE_sum_of_first_45_terms_l2703_270375

def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := 3*n - 1

def c (n : ℕ) : ℕ := a n + b n

def S (n : ℕ) : ℕ := (2^n - 1) + n * (3*n + 1) / 2 - (2 + 8 + 32)

theorem sum_of_first_45_terms : S 45 = 2^45 - 3017 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_45_terms_l2703_270375


namespace NUMINAMATH_CALUDE_linear_function_properties_l2703_270370

/-- A linear function passing through points (3,5) and (-4,-9) -/
def f (x : ℝ) : ℝ := 2 * x - 1

theorem linear_function_properties :
  (∃ k b : ℝ, ∀ x, f x = k * x + b) ∧
  f 3 = 5 ∧
  f (-4) = -9 ∧
  f 0 = -1 ∧
  f (1/2) = 0 ∧
  (1/2 * |f 0|) / 2 = 1/4 ∧
  (∀ a : ℝ, f a = 2 → a = 3/2) :=
sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2703_270370


namespace NUMINAMATH_CALUDE_expression_equals_x_power_44_l2703_270355

def numerator_sequence (n : ℕ) : ℕ := 2 * n + 1

def denominator_sequence (n : ℕ) : ℕ := 4 * n

def numerator_sum (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => numerator_sequence (i + 1))

def denominator_sum (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => denominator_sequence (i + 1))

theorem expression_equals_x_power_44 (x : ℝ) (hx : x = 3) :
  (x ^ numerator_sum 14) / (x ^ denominator_sum 9) = x ^ 44 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_x_power_44_l2703_270355


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2703_270330

/-- Given a geometric sequence {aₙ} with sum of first n terms Sₙ, 
    if S₆/S₃ = -19/8 and a₄ - a₂ = -15/8, then a₃ = 9/4 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℚ)  -- The geometric sequence
  (S : ℕ → ℚ)  -- The sum function
  (h1 : ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1)))  -- Definition of sum for geometric sequence
  (h2 : S 6 / S 3 = -19/8)  -- Given condition
  (h3 : a 4 - a 2 = -15/8)  -- Given condition
  : a 3 = 9/4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2703_270330


namespace NUMINAMATH_CALUDE_probability_theorem_l2703_270397

def team_sizes : List Nat := [6, 9, 10]
def co_captains_per_team : Nat := 3
def members_selected : Nat := 3

def probability_all_co_captains (sizes : List Nat) (co_captains : Nat) (selected : Nat) : ℚ :=
  let team_probabilities := sizes.map (λ n => (co_captains.factorial * (n - co_captains).choose (selected - co_captains)) / n.choose selected)
  (1 / sizes.length) * team_probabilities.sum

theorem probability_theorem :
  probability_all_co_captains team_sizes co_captains_per_team members_selected = 177 / 12600 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2703_270397


namespace NUMINAMATH_CALUDE_rectangular_hyperbola_real_axis_length_l2703_270309

/-- Given a rectangular hyperbola C centered at the origin with foci on the x-axis,
    if C intersects the line x = -4 at two points with a vertical distance of 4√3,
    then the length of the real axis of C is 4. -/
theorem rectangular_hyperbola_real_axis_length
  (C : Set (ℝ × ℝ))
  (h1 : ∃ (a : ℝ), a > 0 ∧ C = {(x, y) | x^2 - y^2 = a^2})
  (h2 : ∃ (y1 y2 : ℝ), ((-4, y1) ∈ C ∧ (-4, y2) ∈ C ∧ |y1 - y2| = 4 * Real.sqrt 3)) :
  ∃ (a : ℝ), a > 0 ∧ C = {(x, y) | x^2 - y^2 = a^2} ∧ 2 * a = 4 :=
sorry

end NUMINAMATH_CALUDE_rectangular_hyperbola_real_axis_length_l2703_270309


namespace NUMINAMATH_CALUDE_john_ray_difference_l2703_270350

/-- The number of chickens each person took -/
structure ChickenCount where
  john : ℕ
  mary : ℕ
  ray : ℕ

/-- The conditions of the chicken distribution -/
def valid_distribution (c : ChickenCount) : Prop :=
  c.john = c.mary + 5 ∧
  c.ray = c.mary - 6 ∧
  c.ray = 10

/-- The theorem stating the difference between John's and Ray's chicken count -/
theorem john_ray_difference (c : ChickenCount) (h : valid_distribution c) : 
  c.john - c.ray = 11 := by
  sorry

end NUMINAMATH_CALUDE_john_ray_difference_l2703_270350


namespace NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_for_ab_zero_l2703_270399

theorem a_zero_sufficient_not_necessary_for_ab_zero :
  (∃ a b : ℝ, a = 0 → a * b = 0) ∧
  (∃ a b : ℝ, a * b = 0 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_a_zero_sufficient_not_necessary_for_ab_zero_l2703_270399


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2703_270379

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l2703_270379


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2703_270332

def A : Set ℝ := {x | x^2 - 16 < 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

theorem intersection_of_A_and_B : 
  A ∩ B = {x | -4 < x ∧ x < 1 ∨ 3 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2703_270332


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l2703_270324

theorem geometric_to_arithmetic_sequence (a b c : ℝ) (x y z : ℝ) :
  (10 ^ a = x) →
  (10 ^ b = y) →
  (10 ^ c = z) →
  (∃ r : ℝ, y = x * r ∧ z = y * r) →  -- geometric sequence condition
  ∃ d : ℝ, b - a = d ∧ c - b = d  -- arithmetic sequence condition
:= by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l2703_270324


namespace NUMINAMATH_CALUDE_min_c_value_l2703_270369

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! (x y : ℝ), 2 * x + y = 2025 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1013 :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l2703_270369


namespace NUMINAMATH_CALUDE_total_is_600_l2703_270368

/-- Represents the shares of money for three individuals -/
structure Shares :=
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)

/-- The conditions of the money division problem -/
def SatisfiesConditions (s : Shares) : Prop :=
  s.a = (2/3) * (s.b + s.c) ∧
  s.b = (6/9) * (s.a + s.c) ∧
  s.a = 240

/-- The theorem stating that the total amount is 600 given the conditions -/
theorem total_is_600 (s : Shares) (h : SatisfiesConditions s) :
  s.a + s.b + s.c = 600 := by
  sorry

#check total_is_600

end NUMINAMATH_CALUDE_total_is_600_l2703_270368


namespace NUMINAMATH_CALUDE_product_set_sum_l2703_270347

theorem product_set_sum (a₁ a₂ a₃ a₄ : ℚ) :
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Finset ℚ) =
  {-24, -2, -3/2, -1/8, 1, 3} →
  (a₁ + a₂ + a₃ + a₄ = 9/4) ∨ (a₁ + a₂ + a₃ + a₄ = -9/4) := by
  sorry

end NUMINAMATH_CALUDE_product_set_sum_l2703_270347


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2703_270343

def p (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2703_270343


namespace NUMINAMATH_CALUDE_integer_pair_property_l2703_270387

theorem integer_pair_property (x y : ℤ) (h : x > y) :
  (x * y - (x + y) = Nat.gcd x.natAbs y.natAbs + Nat.lcm x.natAbs y.natAbs) ↔
  ((x = 6 ∧ y = 3) ∨ 
   (x = 6 ∧ y = 4) ∨ 
   (∃ t : ℕ, x = 1 + t ∧ y = -t) ∨
   (∃ t : ℕ, x = 2 ∧ y = -2 * t)) := by
sorry

end NUMINAMATH_CALUDE_integer_pair_property_l2703_270387


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l2703_270336

/-- Calculates the total cost of concert tickets for a group of friends --/
theorem concert_ticket_cost :
  let normal_price : ℚ := 50
  let website_tickets : ℕ := 3
  let scalper_tickets : ℕ := 4
  let scalper_price_multiplier : ℚ := 2.5
  let scalper_discount : ℚ := 15
  let service_fee_rate : ℚ := 0.1
  let discount_ticket1_rate : ℚ := 0.6
  let discount_ticket2_rate : ℚ := 0.75

  let website_cost : ℚ := normal_price * website_tickets
  let website_fee : ℚ := website_cost * service_fee_rate
  let total_website_cost : ℚ := website_cost + website_fee

  let scalper_cost : ℚ := normal_price * scalper_tickets * scalper_price_multiplier - scalper_discount
  let scalper_fee : ℚ := scalper_cost * service_fee_rate
  let total_scalper_cost : ℚ := scalper_cost + scalper_fee

  let discount_ticket1_cost : ℚ := normal_price * discount_ticket1_rate
  let discount_ticket2_cost : ℚ := normal_price * discount_ticket2_rate
  let total_discount_cost : ℚ := discount_ticket1_cost + discount_ticket2_cost

  let total_cost : ℚ := total_website_cost + total_scalper_cost + total_discount_cost

  total_cost = 766
  := by sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l2703_270336


namespace NUMINAMATH_CALUDE_cuboid_properties_l2703_270312

/-- Represents a cuboid with given length, width, and height -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total length of edges of a cuboid -/
def totalEdgeLength (c : Cuboid) : ℝ :=
  4 * (c.length + c.width + c.height)

/-- Calculates the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.width * c.height + c.height * c.length)

/-- Theorem stating the correctness of totalEdgeLength and surfaceArea functions -/
theorem cuboid_properties (c : Cuboid) :
  (totalEdgeLength c = 4 * (c.length + c.width + c.height)) ∧
  (surfaceArea c = 2 * (c.length * c.width + c.width * c.height + c.height * c.length)) := by
  sorry

end NUMINAMATH_CALUDE_cuboid_properties_l2703_270312


namespace NUMINAMATH_CALUDE_greatest_power_sum_l2703_270333

/-- Given positive integers c and d where d > 1, if c^d is the greatest possible value less than 800, then c + d = 30 -/
theorem greatest_power_sum (c d : ℕ) (hc : c > 0) (hd : d > 1) 
  (h : ∀ (x y : ℕ), x > 0 → y > 1 → x^y < 800 → c^d ≥ x^y) : c + d = 30 := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_sum_l2703_270333


namespace NUMINAMATH_CALUDE_red_candles_count_l2703_270360

/-- Given the ratio of red candles to blue candles and the number of blue candles,
    calculate the number of red candles. -/
theorem red_candles_count (blue_candles : ℕ) (ratio_red : ℕ) (ratio_blue : ℕ) 
    (h1 : blue_candles = 27) 
    (h2 : ratio_red = 5) 
    (h3 : ratio_blue = 3) : ℕ :=
  45

#check red_candles_count

end NUMINAMATH_CALUDE_red_candles_count_l2703_270360


namespace NUMINAMATH_CALUDE_worker_idle_days_l2703_270364

/-- Proves that given the specified conditions, the number of idle days is 38 --/
theorem worker_idle_days 
  (total_days : ℕ) 
  (pay_per_working_day : ℕ) 
  (forfeit_per_idle_day : ℕ) 
  (total_amount : ℕ) 
  (h1 : total_days = 60)
  (h2 : pay_per_working_day = 30)
  (h3 : forfeit_per_idle_day = 5)
  (h4 : total_amount = 500) :
  ∃ (idle_days : ℕ), 
    idle_days = 38 ∧ 
    idle_days + (total_days - idle_days) = total_days ∧
    pay_per_working_day * (total_days - idle_days) - forfeit_per_idle_day * idle_days = total_amount :=
by
  sorry

end NUMINAMATH_CALUDE_worker_idle_days_l2703_270364


namespace NUMINAMATH_CALUDE_senate_democrats_count_l2703_270390

/-- Given the conditions of the House of Representatives and Senate composition,
    prove that the number of Democrats in the Senate is 55. -/
theorem senate_democrats_count : 
  ∀ (house_total house_dem house_rep senate_total senate_dem senate_rep : ℕ),
  house_total = 434 →
  house_total = house_dem + house_rep →
  house_rep = house_dem + 30 →
  senate_total = 100 →
  senate_total = senate_dem + senate_rep →
  5 * senate_rep = 4 * senate_dem →
  senate_dem = 55 := by
  sorry

end NUMINAMATH_CALUDE_senate_democrats_count_l2703_270390


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2703_270310

/-- Given a geometric sequence {a_n} with a_1 = 3 and a_1 + a_3 + a_5 = 21, 
    prove that a_3 + a_5 + a_7 = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 3)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : a 1 + a 3 + a 5 = 21) :
  a 3 + a 5 + a 7 = 42 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2703_270310


namespace NUMINAMATH_CALUDE_no_identical_lines_l2703_270372

-- Define the equations of the lines
def line1 (a d x y : ℝ) : Prop := 4 * x + a * y + d = 0
def line2 (d x y : ℝ) : Prop := d * x - 3 * y + 15 = 0

-- Define what it means for the lines to be identical
def identical_lines (a d : ℝ) : Prop :=
  ∀ x y : ℝ, line1 a d x y ↔ line2 d x y

-- Theorem statement
theorem no_identical_lines : ¬∃ a d : ℝ, identical_lines a d :=
sorry

end NUMINAMATH_CALUDE_no_identical_lines_l2703_270372


namespace NUMINAMATH_CALUDE_razorback_tshirt_sales_l2703_270356

theorem razorback_tshirt_sales 
  (revenue_per_tshirt : ℕ) 
  (total_tshirts : ℕ) 
  (revenue_one_game : ℕ) 
  (h1 : revenue_per_tshirt = 98)
  (h2 : total_tshirts = 163)
  (h3 : revenue_one_game = 8722) :
  ∃ (arkansas_tshirts : ℕ), 
    arkansas_tshirts * revenue_per_tshirt = revenue_one_game ∧
    arkansas_tshirts ≤ total_tshirts ∧
    arkansas_tshirts = 89 :=
by sorry

end NUMINAMATH_CALUDE_razorback_tshirt_sales_l2703_270356


namespace NUMINAMATH_CALUDE_multiply_and_distribute_l2703_270329

theorem multiply_and_distribute (m : ℝ) : (4*m + 1) * (2*m) = 8*m^2 + 2*m := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_distribute_l2703_270329


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2703_270396

-- Define the quadratic function f
def f (x : ℝ) : ℝ := x^2 + x - 2

-- Define the theorem
theorem quadratic_function_theorem :
  (∀ x : ℝ, f x < 0 ↔ -2 < x ∧ x < 1) ∧ 
  f 0 = -2 ∧
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (∀ x : ℝ, f x = x^2 + x - 2) ∧
  (∀ m : ℝ, (∀ θ : ℝ, f (Real.cos θ) ≤ Real.sqrt 2 * Real.sin (θ + Real.pi / 4) + m * Real.sin θ) ↔ 
    -3 ≤ m ∧ m ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2703_270396


namespace NUMINAMATH_CALUDE_max_value_is_12_l2703_270338

/-- Represents an arithmetic expression using the given operations and numbers -/
inductive Expr
  | num : ℕ → Expr
  | add : Expr → Expr → Expr
  | div : Expr → Expr → Expr
  | mul : Expr → Expr → Expr

/-- Evaluates an arithmetic expression -/
def eval : Expr → ℚ
  | Expr.num n => n
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.div e1 e2 => eval e1 / eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2

/-- Checks if an expression uses the given numbers in order -/
def usesNumbers (e : Expr) (nums : List ℕ) : Prop := sorry

/-- Counts the number of times each operation is used in an expression -/
def countOps (e : Expr) : (ℕ × ℕ × ℕ) := sorry

/-- Checks if an expression uses at most one pair of parentheses -/
def atMostOneParenthesis (e : Expr) : Prop := sorry

/-- The main theorem statement -/
theorem max_value_is_12 :
  ∀ e : Expr,
    usesNumbers e [7, 2, 3, 4] →
    countOps e = (1, 1, 1) →
    atMostOneParenthesis e →
    eval e ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_is_12_l2703_270338


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l2703_270385

/-- Given two circles C₁ and C₂ defined by equations (x-12)²+y²=25 and (x+18)²+y²=64 respectively,
    the length of the shortest line segment RS tangent to C₁ at R and C₂ at S is 339/13. -/
theorem shortest_tangent_length (C₁ C₂ : Set (ℝ × ℝ)) (R S : ℝ × ℝ) :
  C₁ = {p : ℝ × ℝ | (p.1 - 12)^2 + p.2^2 = 25} →
  C₂ = {p : ℝ × ℝ | (p.1 + 18)^2 + p.2^2 = 64} →
  R ∈ C₁ →
  S ∈ C₂ →
  (∀ p ∈ C₁, (R.1 - p.1) * (R.1 - 12) + (R.2 - p.2) * R.2 = 0) →
  (∀ p ∈ C₂, (S.1 - p.1) * (S.1 + 18) + (S.2 - p.2) * S.2 = 0) →
  (∀ T U : ℝ × ℝ, T ∈ C₁ → U ∈ C₂ → 
    (∀ q ∈ C₁, (T.1 - q.1) * (T.1 - 12) + (T.2 - q.2) * T.2 = 0) →
    (∀ q ∈ C₂, (U.1 - q.1) * (U.1 + 18) + (U.2 - q.2) * U.2 = 0) →
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) ≤ Real.sqrt ((T.1 - U.1)^2 + (T.2 - U.2)^2)) →
  Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 339 / 13 :=
by sorry

end NUMINAMATH_CALUDE_shortest_tangent_length_l2703_270385


namespace NUMINAMATH_CALUDE_bus_speed_l2703_270388

/-- Calculates the speed of a bus in kilometers per hour (kmph) given distance and time -/
theorem bus_speed (distance : Real) (time : Real) (conversion_factor : Real) : 
  distance = 900.072 ∧ time = 30 ∧ conversion_factor = 3.6 →
  (distance / time) * conversion_factor = 108.00864 := by
  sorry

#check bus_speed

end NUMINAMATH_CALUDE_bus_speed_l2703_270388


namespace NUMINAMATH_CALUDE_image_of_two_is_five_l2703_270349

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- State the theorem
theorem image_of_two_is_five : f 2 = 5 := by sorry

end NUMINAMATH_CALUDE_image_of_two_is_five_l2703_270349


namespace NUMINAMATH_CALUDE_doors_per_apartment_l2703_270383

/-- Proves that the number of doors per apartment is 7, given the specifications of the apartment buildings and total doors needed. -/
theorem doors_per_apartment 
  (num_buildings : ℕ) 
  (floors_per_building : ℕ) 
  (apartments_per_floor : ℕ) 
  (total_doors : ℕ) 
  (h1 : num_buildings = 2)
  (h2 : floors_per_building = 12)
  (h3 : apartments_per_floor = 6)
  (h4 : total_doors = 1008) :
  total_doors / (num_buildings * floors_per_building * apartments_per_floor) = 7 := by
  sorry

#check doors_per_apartment

end NUMINAMATH_CALUDE_doors_per_apartment_l2703_270383


namespace NUMINAMATH_CALUDE_complex_product_simplification_l2703_270394

theorem complex_product_simplification :
  let i : ℂ := Complex.I
  ((4 - 3*i) - (2 + 5*i)) * (2*i) = 16 + 4*i := by sorry

end NUMINAMATH_CALUDE_complex_product_simplification_l2703_270394


namespace NUMINAMATH_CALUDE_factorization_of_2x_cubed_minus_8x_l2703_270308

theorem factorization_of_2x_cubed_minus_8x (x : ℝ) : 2*x^3 - 8*x = 2*x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x_cubed_minus_8x_l2703_270308


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_68_l2703_270358

theorem smallest_n_divisible_by_68 :
  ∃ (n : ℕ), n^2 + 14*n + 13 ≡ 0 [MOD 68] ∧
  (∀ (m : ℕ), m < n → ¬(m^2 + 14*m + 13 ≡ 0 [MOD 68])) ∧
  n = 21 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_68_l2703_270358


namespace NUMINAMATH_CALUDE_money_conditions_l2703_270313

theorem money_conditions (a b : ℝ) 
  (h1 : 6 * a - b = 45)
  (h2 : 4 * a + b > 60) : 
  a > 10.5 ∧ b > 18 := by
sorry

end NUMINAMATH_CALUDE_money_conditions_l2703_270313


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l2703_270346

theorem trigonometric_expression_equality : 
  4 * Real.sin (80 * π / 180) - Real.cos (10 * π / 180) / Real.sin (10 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l2703_270346


namespace NUMINAMATH_CALUDE_max_value_constraint_l2703_270393

theorem max_value_constraint (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) :
  x + 2*y + 2*z ≤ 15 := by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2703_270393


namespace NUMINAMATH_CALUDE_only_negative_four_has_no_sqrt_l2703_270305

-- Define the set of numbers we're considering
def numbers : Set ℝ := {-4, 0, 0.5, 2}

-- Define what it means for a real number to have a square root
def has_sqrt (x : ℝ) : Prop := ∃ y : ℝ, y^2 = x

-- State the theorem
theorem only_negative_four_has_no_sqrt :
  ∀ x ∈ numbers, ¬(has_sqrt x) ↔ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_only_negative_four_has_no_sqrt_l2703_270305


namespace NUMINAMATH_CALUDE_alicia_tax_deduction_l2703_270366

/-- Represents Alicia's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℚ := 25 / 1000

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tax deduction in cents -/
def tax_deduction (wage : ℚ) (rate : ℚ) : ℚ :=
  dollars_to_cents (wage * rate)

theorem alicia_tax_deduction :
  tax_deduction hourly_wage tax_rate = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_alicia_tax_deduction_l2703_270366


namespace NUMINAMATH_CALUDE_mary_anne_sparkling_water_l2703_270384

-- Define the cost per bottle
def cost_per_bottle : ℚ := 2

-- Define the total spent per year
def total_spent_per_year : ℚ := 146

-- Define the number of days in a year
def days_per_year : ℕ := 365

-- Define the fraction of a bottle drunk each night
def fraction_per_night : ℚ := 1 / 5

-- Theorem statement
theorem mary_anne_sparkling_water :
  fraction_per_night * (days_per_year : ℚ) = total_spent_per_year / cost_per_bottle :=
sorry

end NUMINAMATH_CALUDE_mary_anne_sparkling_water_l2703_270384


namespace NUMINAMATH_CALUDE_range_of_y_minus_x_l2703_270380

-- Define the triangle ABC and points D and P
variable (A B C D P : ℝ × ℝ)

-- Define vectors
def vec (X Y : ℝ × ℝ) : ℝ × ℝ := (Y.1 - X.1, Y.2 - X.2)

-- Conditions
variable (h1 : vec D C = (2 * (vec A D).1, 2 * (vec A D).2))
variable (h2 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * B.1 + (1 - t) * D.1, t * B.2 + (1 - t) * D.2))
variable (h3 : ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ vec A P = (x * (vec A B).1 + y * (vec A C).1, x * (vec A B).2 + y * (vec A C).2))

-- Theorem statement
theorem range_of_y_minus_x :
  ∃ S : Set ℝ, S = {z | ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    vec A P = (x * (vec A B).1 + y * (vec A C).1, x * (vec A B).2 + y * (vec A C).2) ∧
    z = y - x} ∧
  S = {z | -1 < z ∧ z < 1/3} :=
sorry

end NUMINAMATH_CALUDE_range_of_y_minus_x_l2703_270380


namespace NUMINAMATH_CALUDE_rotate_180_equals_optionC_l2703_270351

/-- Represents a geometric shape --/
structure Shape :=
  (id : ℕ)

/-- Represents a rotation operation --/
def rotate (s : Shape) (angle : ℝ) : Shape :=
  { id := s.id }

/-- The original T-like shape --/
def original : Shape :=
  { id := 0 }

/-- Option C from the problem --/
def optionC : Shape :=
  { id := 1 }

/-- Theorem stating that rotating the original shape 180 degrees results in option C --/
theorem rotate_180_equals_optionC : 
  rotate original 180 = optionC := by
  sorry

end NUMINAMATH_CALUDE_rotate_180_equals_optionC_l2703_270351


namespace NUMINAMATH_CALUDE_proper_subset_of_A_l2703_270322

def A : Set ℝ := {x | x^2 < 5*x}

theorem proper_subset_of_A : Set.Subset (Set.Ioo 1 5) A ∧ (Set.Ioo 1 5) ≠ A := by
  sorry

end NUMINAMATH_CALUDE_proper_subset_of_A_l2703_270322


namespace NUMINAMATH_CALUDE_triangle_height_l2703_270371

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) :
  area = 46 →
  base = 10 →
  area = (base * height) / 2 →
  height = 9.2 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l2703_270371


namespace NUMINAMATH_CALUDE_philippe_can_win_l2703_270377

/-- Represents a game state with cards remaining and sums for each player -/
structure GameState :=
  (remaining : Finset Nat)
  (philippe_sum : Nat)
  (emmanuel_sum : Nat)

/-- The initial game state -/
def initial_state : GameState :=
  { remaining := Finset.range 2018,
    philippe_sum := 0,
    emmanuel_sum := 0 }

/-- A strategy is a function that selects a card from the remaining set -/
def Strategy := (GameState → Nat)

/-- Applies a strategy to a game state, returning the new state -/
def apply_strategy (s : Strategy) (g : GameState) : GameState :=
  let card := s g
  { remaining := g.remaining.erase card,
    philippe_sum := g.philippe_sum + card,
    emmanuel_sum := g.emmanuel_sum }

/-- Plays the game to completion using the given strategies -/
def play_game (philippe_strategy : Strategy) (emmanuel_strategy : Strategy) : GameState :=
  sorry

/-- Theorem stating that Philippe can always win -/
theorem philippe_can_win :
  ∃ (philippe_strategy : Strategy),
    ∀ (emmanuel_strategy : Strategy),
      let final_state := play_game philippe_strategy emmanuel_strategy
      Even final_state.philippe_sum ∧ Odd final_state.emmanuel_sum :=
sorry

end NUMINAMATH_CALUDE_philippe_can_win_l2703_270377


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l2703_270357

theorem square_perimeter_relation (perimeter_C : ℝ) (area_ratio : ℝ) : 
  perimeter_C = 40 →
  area_ratio = 1/3 →
  let side_C := perimeter_C / 4
  let area_C := side_C ^ 2
  let area_D := area_ratio * area_C
  let side_D := Real.sqrt area_D
  let perimeter_D := 4 * side_D
  perimeter_D = (40 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l2703_270357


namespace NUMINAMATH_CALUDE_cost_per_candy_bar_l2703_270392

-- Define the given conditions
def boxes_sold : ℕ := 5
def candy_bars_per_box : ℕ := 10
def selling_price_per_bar : ℚ := 3/2  -- $1.50 as a rational number
def total_profit : ℚ := 25

-- Define the theorem
theorem cost_per_candy_bar :
  let total_bars := boxes_sold * candy_bars_per_box
  let total_revenue := total_bars * selling_price_per_bar
  let total_cost := total_revenue - total_profit
  total_cost / total_bars = 1 := by sorry

end NUMINAMATH_CALUDE_cost_per_candy_bar_l2703_270392


namespace NUMINAMATH_CALUDE_geometric_sequence_12th_term_l2703_270311

/-- Given a geometric sequence where the 5th term is 5 and the 8th term is 40, 
    the 12th term is 640. -/
theorem geometric_sequence_12th_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_5th : a 5 = 5) 
  (h_8th : a 8 = 40) : 
  a 12 = 640 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_12th_term_l2703_270311


namespace NUMINAMATH_CALUDE_negation_equivalence_l2703_270317

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 1 ≤ 2*x) ↔ (∀ x : ℝ, x^2 + 1 > 2*x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2703_270317


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l2703_270306

theorem simplify_complex_fraction : 
  1 / ((1 / (Real.sqrt 3 + 2)) + (2 / (Real.sqrt 5 - 2))) = Real.sqrt 3 - 2 * Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l2703_270306


namespace NUMINAMATH_CALUDE_two_digit_primes_from_set_l2703_270365

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def digit_set : Finset ℕ := {3, 5, 8, 9}

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def formed_from_set (n : ℕ) : Prop :=
  is_two_digit n ∧
  (n / 10) ∈ digit_set ∧
  (n % 10) ∈ digit_set ∧
  (n / 10) ≠ (n % 10)

theorem two_digit_primes_from_set :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_prime n ∧ formed_from_set n) ∧
    (∀ n, is_prime n ∧ formed_from_set n → n ∈ s) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_two_digit_primes_from_set_l2703_270365


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l2703_270373

theorem arithmetic_sequence_squares (a b c : ℝ) 
  (h : ∃ (d : ℝ), (1 / (b + c)) - (1 / (a + b)) = (1 / (c + a)) - (1 / (b + c))) :
  ∃ (k : ℝ), b^2 - a^2 = c^2 - b^2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l2703_270373


namespace NUMINAMATH_CALUDE_combined_temp_range_l2703_270395

-- Define the temperature ranges for each vegetable type
def type_a_range : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }
def type_b_range : Set ℝ := { x | 3 ≤ x ∧ x ≤ 8 }

-- Define the combined suitable temperature range
def combined_range : Set ℝ := type_a_range ∩ type_b_range

-- Theorem stating the combined suitable temperature range
theorem combined_temp_range : 
  combined_range = { x | 3 ≤ x ∧ x ≤ 5 } := by sorry

end NUMINAMATH_CALUDE_combined_temp_range_l2703_270395


namespace NUMINAMATH_CALUDE_problem_solution_l2703_270326

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_xyz : x * y * z = 1)
  (h_x_z : x + 1 / z = 7)
  (h_y_x : y + 1 / x = 20) :
  z + 1 / y = 29 / 139 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2703_270326


namespace NUMINAMATH_CALUDE_intersection_A_B_when_m_1_sufficient_necessary_condition_l2703_270321

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 7*x + 6 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x + 1 - m^2 ≤ 0}

-- Part 1: Intersection of A and B when m = 1
theorem intersection_A_B_when_m_1 : A ∩ B 1 = {x | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2: Condition for x ∈ A to be sufficient and necessary for x ∈ B
theorem sufficient_necessary_condition (m : ℝ) :
  (∀ x, x ∈ A ↔ x ∈ B m) ↔ m ≥ 5 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_m_1_sufficient_necessary_condition_l2703_270321


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l2703_270382

theorem loss_percentage_calculation (CP : ℝ) :
  CP > 0 ∧ 
  240 = CP * (1 + 0.20) ∧ 
  170 < CP 
  → 
  (CP - 170) / CP * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l2703_270382


namespace NUMINAMATH_CALUDE_jamie_yellow_balls_l2703_270331

theorem jamie_yellow_balls (initial_red : ℕ) (total_after : ℕ) : 
  initial_red = 16 →
  total_after = 74 →
  (initial_red - 6) + (2 * initial_red) + (total_after - ((initial_red - 6) + (2 * initial_red))) = total_after :=
by
  sorry

end NUMINAMATH_CALUDE_jamie_yellow_balls_l2703_270331


namespace NUMINAMATH_CALUDE_kimikos_age_l2703_270391

theorem kimikos_age (kimiko omi arlette : ℝ) 
  (h1 : omi = 2 * kimiko)
  (h2 : arlette = 3/4 * kimiko)
  (h3 : (kimiko + omi + arlette) / 3 = 35) :
  kimiko = 28 := by
  sorry

end NUMINAMATH_CALUDE_kimikos_age_l2703_270391


namespace NUMINAMATH_CALUDE_absolute_value_five_minus_sqrt_eleven_l2703_270328

theorem absolute_value_five_minus_sqrt_eleven : |5 - Real.sqrt 11| = 1.683 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_five_minus_sqrt_eleven_l2703_270328


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l2703_270348

theorem polynomial_multiplication :
  ∀ x : ℝ, (5 * x + 3) * (2 * x - 4 + x^2) = 5 * x^3 + 13 * x^2 - 14 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l2703_270348


namespace NUMINAMATH_CALUDE_purple_ring_weight_l2703_270315

/-- The weight of the purple ring given the weights of other rings and the total weight -/
theorem purple_ring_weight 
  (orange_weight : ℝ) 
  (white_weight : ℝ) 
  (total_weight : ℝ) 
  (h_orange : orange_weight = 0.08333333333333333)
  (h_white : white_weight = 0.4166666666666667)
  (h_total : total_weight = 0.8333333333333334) :
  total_weight - (orange_weight + white_weight) = 0.3333333333333334 := by
sorry

end NUMINAMATH_CALUDE_purple_ring_weight_l2703_270315


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2703_270352

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2703_270352


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2703_270337

theorem inequality_solution_set (a b : ℝ) (h : b ≠ 0) :
  ¬(∀ x : ℝ, ax > b ↔ x < -b/a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2703_270337


namespace NUMINAMATH_CALUDE_problem_solution_l2703_270339

theorem problem_solution : 18 * ((150 / 3) + (40 / 5) + (16 / 32) + 2) = 1089 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2703_270339


namespace NUMINAMATH_CALUDE_no_rational_roots_odd_coefficients_l2703_270327

theorem no_rational_roots_odd_coefficients (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_odd_coefficients_l2703_270327


namespace NUMINAMATH_CALUDE_mary_final_cards_l2703_270361

/-- Calculates the final number of baseball cards Mary has after a series of transactions -/
def final_card_count (initial_cards torn_cards fred_cards bought_cards lost_cards lisa_trade_in lisa_trade_out alex_trade_in alex_trade_out : ℕ) : ℕ :=
  initial_cards - torn_cards + fred_cards + bought_cards - lost_cards - lisa_trade_in + lisa_trade_out - alex_trade_in + alex_trade_out

/-- Theorem stating that Mary ends up with 70 baseball cards -/
theorem mary_final_cards : 
  final_card_count 18 8 26 40 5 3 4 7 5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_mary_final_cards_l2703_270361


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2703_270389

theorem cubic_root_sum (a b c : ℕ+) :
  let x : ℝ := (Real.rpow a (1/3 : ℝ) + Real.rpow b (1/3 : ℝ) + 2) / c
  27 * x^3 - 6 * x^2 - 6 * x - 2 = 0 →
  a + b + c = 75 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2703_270389


namespace NUMINAMATH_CALUDE_number_operation_l2703_270386

theorem number_operation (x : ℝ) : (x - 5) / 7 = 7 → (x - 4) / 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_l2703_270386


namespace NUMINAMATH_CALUDE_field_path_area_and_cost_l2703_270302

/-- Represents the dimensions of a rectangular field with a path around it -/
structure FieldWithPath where
  field_length : ℝ
  field_width : ℝ
  path_width : ℝ

/-- Calculates the area of the path around a rectangular field -/
def path_area (f : FieldWithPath) : ℝ :=
  (f.field_length + 2 * f.path_width) * (f.field_width + 2 * f.path_width) - f.field_length * f.field_width

/-- Calculates the cost of constructing the path given a cost per square meter -/
def path_cost (f : FieldWithPath) (cost_per_sqm : ℝ) : ℝ :=
  path_area f * cost_per_sqm

/-- Theorem stating the area of the path and its construction cost for the given field -/
theorem field_path_area_and_cost :
  let f : FieldWithPath := { field_length := 75, field_width := 40, path_width := 2.5 }
  path_area f = 600 ∧ path_cost f 2 = 1200 := by sorry

end NUMINAMATH_CALUDE_field_path_area_and_cost_l2703_270302


namespace NUMINAMATH_CALUDE_aunt_gift_amount_l2703_270304

theorem aunt_gift_amount (jade_initial : ℕ) (julia_initial : ℕ) (total_after : ℕ) 
    (h1 : jade_initial = 38)
    (h2 : julia_initial = jade_initial / 2)
    (h3 : total_after = 97)
    (h4 : ∃ (gift : ℕ), total_after = jade_initial + julia_initial + 2 * gift) :
  ∃ (gift : ℕ), gift = 20 ∧ total_after = jade_initial + julia_initial + 2 * gift :=
by sorry

end NUMINAMATH_CALUDE_aunt_gift_amount_l2703_270304


namespace NUMINAMATH_CALUDE_cubic_inequality_l2703_270367

theorem cubic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 * b + b^3 * c + c^3 * a - a^2 * b * c - b^2 * c * a - c^2 * a * b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2703_270367


namespace NUMINAMATH_CALUDE_neil_initial_games_l2703_270376

theorem neil_initial_games (henry_initial : ℕ) (games_given : ℕ) (neil_initial : ℕ) :
  henry_initial = 58 →
  games_given = 6 →
  henry_initial - games_given = 4 * (neil_initial + games_given) →
  neil_initial = 7 := by
sorry

end NUMINAMATH_CALUDE_neil_initial_games_l2703_270376


namespace NUMINAMATH_CALUDE_king_of_diamonds_in_top_two_l2703_270307

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (suits : ℕ)
  (ranks : ℕ)
  (jokers : ℕ)

/-- The probability of an event occurring -/
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

/-- Theorem stating the probability of the King of Diamonds being one of the top two cards -/
theorem king_of_diamonds_in_top_two (d : Deck) 
  (h1 : d.total_cards = 54)
  (h2 : d.suits = 4)
  (h3 : d.ranks = 13)
  (h4 : d.jokers = 2) :
  probability 2 d.total_cards = 1 / 27 := by
  sorry

#check king_of_diamonds_in_top_two

end NUMINAMATH_CALUDE_king_of_diamonds_in_top_two_l2703_270307


namespace NUMINAMATH_CALUDE_james_club_expenditure_l2703_270320

/-- Calculate the total amount James spent at the club --/
theorem james_club_expenditure :
  let entry_fee : ℕ := 20
  let rounds_for_friends : ℕ := 2
  let num_friends : ℕ := 5
  let drinks_for_self : ℕ := 6
  let drink_cost : ℕ := 6
  let food_cost : ℕ := 14
  let tip_percentage : ℚ := 30 / 100

  let drinks_cost : ℕ := rounds_for_friends * num_friends * drink_cost + drinks_for_self * drink_cost
  let subtotal : ℕ := entry_fee + drinks_cost + food_cost
  let tip : ℕ := (tip_percentage * (drinks_cost + food_cost)).num.toNat
  let total_spent : ℕ := subtotal + tip

  total_spent = 163 := by sorry

end NUMINAMATH_CALUDE_james_club_expenditure_l2703_270320


namespace NUMINAMATH_CALUDE_notebook_cost_l2703_270303

theorem notebook_cost (initial_money : ℕ) (notebooks_bought : ℕ) (books_bought : ℕ) 
  (book_cost : ℕ) (money_left : ℕ) :
  initial_money = 56 →
  notebooks_bought = 7 →
  books_bought = 2 →
  book_cost = 7 →
  money_left = 14 →
  ∃ (notebook_cost : ℕ), 
    notebook_cost * notebooks_bought + book_cost * books_bought = initial_money - money_left ∧
    notebook_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2703_270303


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2703_270300

theorem decimal_to_fraction :
  (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2703_270300


namespace NUMINAMATH_CALUDE_no_x_squared_term_l2703_270341

theorem no_x_squared_term (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + 5) * (-2*x) - 6*x^2 = -2*x^3 - 10*x) ↔ a = -3 :=
by sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l2703_270341


namespace NUMINAMATH_CALUDE_checkerboard_fraction_sum_l2703_270363

/-- The number of squares in a n×n grid -/
def squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of rectangles in a (n+1)×(n+1) grid -/
def rectangles (n : ℕ) : ℕ := (n * (n + 1) / 2)^2

theorem checkerboard_fraction_sum :
  let s := squares 7
  let r := rectangles 7
  let g := Nat.gcd s r
  (s / g) + (r / g) = 33 := by sorry

end NUMINAMATH_CALUDE_checkerboard_fraction_sum_l2703_270363


namespace NUMINAMATH_CALUDE_rainwater_farm_l2703_270378

theorem rainwater_farm (cows goats chickens : ℕ) : 
  cows = 9 →
  goats = 4 * cows →
  goats = 2 * chickens →
  chickens = 18 := by
sorry

end NUMINAMATH_CALUDE_rainwater_farm_l2703_270378


namespace NUMINAMATH_CALUDE_words_with_vowels_count_l2703_270398

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels : Finset Char := {'A', 'E'}
def consonants : Finset Char := alphabet \ vowels
def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def consonant_words : Nat := consonants.card ^ word_length
def words_with_vowels : Nat := total_words - consonant_words

theorem words_with_vowels_count : words_with_vowels = 6752 := by sorry

end NUMINAMATH_CALUDE_words_with_vowels_count_l2703_270398


namespace NUMINAMATH_CALUDE_absent_children_count_l2703_270301

/-- Given a school with a total of 700 children, where each child was supposed to get 2 bananas,
    but due to absences, each present child got 4 bananas instead,
    prove that the number of absent children is 350. -/
theorem absent_children_count (total_children : ℕ) (bananas_per_child_original : ℕ) 
    (bananas_per_child_actual : ℕ) (absent_children : ℕ) : 
    total_children = 700 → 
    bananas_per_child_original = 2 →
    bananas_per_child_actual = 4 →
    absent_children = total_children - (total_children * bananas_per_child_original) / bananas_per_child_actual →
    absent_children = 350 := by
  sorry

end NUMINAMATH_CALUDE_absent_children_count_l2703_270301


namespace NUMINAMATH_CALUDE_gcd_140_396_l2703_270342

theorem gcd_140_396 : Nat.gcd 140 396 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_140_396_l2703_270342


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l2703_270362

theorem consecutive_even_integers_sum (y : ℤ) : 
  y % 2 = 0 ∧ 
  (y + 2) % 2 = 0 ∧ 
  y = 2 * (y + 2) → 
  y + (y + 2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l2703_270362


namespace NUMINAMATH_CALUDE_minimum_distance_to_exponential_curve_l2703_270354

open Real

theorem minimum_distance_to_exponential_curve (a : ℝ) :
  (∃ x₀ : ℝ, (x₀ - a)^2 + (exp x₀ - a)^2 ≤ 1/2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_distance_to_exponential_curve_l2703_270354


namespace NUMINAMATH_CALUDE_absolute_value_integral_l2703_270344

theorem absolute_value_integral : ∫ x in (-1)..2, |x| = 5/2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_integral_l2703_270344


namespace NUMINAMATH_CALUDE_price_reduction_equation_l2703_270334

/-- Given an original price and a final price after two equal percentage reductions,
    this theorem states the equation relating the reduction percentage to the prices. -/
theorem price_reduction_equation (original_price final_price : ℝ) (x : ℝ) 
  (h1 : original_price = 60)
  (h2 : final_price = 48.6)
  (h3 : x > 0 ∧ x < 1) :
  original_price * (1 - x)^2 = final_price := by
sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l2703_270334


namespace NUMINAMATH_CALUDE_calculate_expression_l2703_270319

theorem calculate_expression : (8^3 / 8^2) * 2^6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2703_270319


namespace NUMINAMATH_CALUDE_oranges_from_joyce_calculation_l2703_270340

/-- Represents the number of oranges Clarence has initially. -/
def initial_oranges : ℕ := 5

/-- Represents the total number of oranges Clarence has after receiving some from Joyce. -/
def total_oranges : ℕ := 8

/-- Represents the number of oranges Joyce gave to Clarence. -/
def oranges_from_joyce : ℕ := total_oranges - initial_oranges

/-- Proves that the number of oranges Joyce gave to Clarence is equal to the difference
    between Clarence's total oranges after receiving from Joyce and Clarence's initial oranges. -/
theorem oranges_from_joyce_calculation :
  oranges_from_joyce = total_oranges - initial_oranges :=
by sorry

end NUMINAMATH_CALUDE_oranges_from_joyce_calculation_l2703_270340


namespace NUMINAMATH_CALUDE_money_distribution_l2703_270318

theorem money_distribution (ram gopal krishan : ℚ) : 
  (ram / gopal = 7 / 17) →
  (gopal / krishan = 7 / 17) →
  (ram = 490) →
  (krishan = 2890) :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l2703_270318


namespace NUMINAMATH_CALUDE_sum_bounds_and_range_l2703_270323

open Real

theorem sum_bounds_and_range (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let S := a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)
  (1 < S ∧ S < 2) ∧ ∀ x, 1 < x → x < 2 → ∃ a' b' c' d', 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 
    x = a' / (a' + b' + d') + b' / (a' + b' + c') + c' / (b' + c' + d') + d' / (a' + c' + d') :=
by sorry

end NUMINAMATH_CALUDE_sum_bounds_and_range_l2703_270323


namespace NUMINAMATH_CALUDE_correct_survey_order_l2703_270335

/-- Represents the steps in conducting a survey --/
inductive SurveyStep
  | CreateQuestionnaire
  | OrganizeResults
  | DrawPieChart
  | AnalyzeResults

/-- Defines the correct order of survey steps --/
def correct_order : List SurveyStep :=
  [SurveyStep.CreateQuestionnaire, SurveyStep.OrganizeResults, 
   SurveyStep.DrawPieChart, SurveyStep.AnalyzeResults]

/-- Theorem stating that the defined order is correct for determining the most popular club activity --/
theorem correct_survey_order : 
  correct_order = [SurveyStep.CreateQuestionnaire, SurveyStep.OrganizeResults, 
                   SurveyStep.DrawPieChart, SurveyStep.AnalyzeResults] := by
  sorry

end NUMINAMATH_CALUDE_correct_survey_order_l2703_270335
