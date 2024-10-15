import Mathlib

namespace NUMINAMATH_CALUDE_find_Y_l943_94376

theorem find_Y : ∃ Y : ℚ, (19 + Y / 151) * 151 = 2912 → Y = 43 := by
  sorry

end NUMINAMATH_CALUDE_find_Y_l943_94376


namespace NUMINAMATH_CALUDE_power_in_denominator_l943_94300

theorem power_in_denominator (x : ℕ) : (10 ^ 655 / 10 ^ x = 100000) → x = 650 := by
  sorry

end NUMINAMATH_CALUDE_power_in_denominator_l943_94300


namespace NUMINAMATH_CALUDE_perpendicular_planes_necessary_not_sufficient_l943_94330

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perp_line_plane (m : Line) (α : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two planes are perpendicular -/
def perp_plane_plane (α β : Plane) : Prop := sorry

/-- Definition of necessary but not sufficient condition -/
def necessary_not_sufficient (P Q : Prop) : Prop :=
  (Q → P) ∧ ¬(P → Q)

theorem perpendicular_planes_necessary_not_sufficient 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) (h2 : α ≠ β) 
  (h3 : perp_line_plane m α) (h4 : line_in_plane n β) :
  necessary_not_sufficient (perp_plane_plane α β) (parallel m n) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_planes_necessary_not_sufficient_l943_94330


namespace NUMINAMATH_CALUDE_total_area_smaller_than_4pi_R_squared_l943_94375

variable (R x y z : ℝ)

/-- Three circles with radii x, y, and z touch each other externally -/
axiom circles_touch_externally : True

/-- The centers of the three circles lie on a fourth circle with radius R -/
axiom centers_on_fourth_circle : True

/-- The radius R of the fourth circle is related to x, y, and z by Heron's formula -/
axiom heron_formula : R = (x + y) * (y + z) * (z + x) / (4 * Real.sqrt ((x + y + z) * x * y * z))

/-- The total area of the three circle disks is smaller than 4πR² -/
theorem total_area_smaller_than_4pi_R_squared :
  x^2 + y^2 + z^2 < 4 * R^2 := by sorry

end NUMINAMATH_CALUDE_total_area_smaller_than_4pi_R_squared_l943_94375


namespace NUMINAMATH_CALUDE_largest_two_digit_divisible_by_six_ending_in_four_l943_94323

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

theorem largest_two_digit_divisible_by_six_ending_in_four :
  ∀ n : ℕ, is_two_digit n → n % 6 = 0 → ends_in_four n → n ≤ 84 :=
sorry

end NUMINAMATH_CALUDE_largest_two_digit_divisible_by_six_ending_in_four_l943_94323


namespace NUMINAMATH_CALUDE_widest_opening_is_f₃_l943_94306

/-- The quadratic function with the widest opening -/
def widest_opening (f₁ f₂ f₃ f₄ : ℝ → ℝ) : Prop :=
  ∃ (a₁ a₂ a₃ a₄ : ℝ),
    (∀ x, f₁ x = -10 * x^2) ∧
    (∀ x, f₂ x = 2 * x^2) ∧
    (∀ x, f₃ x = (1/100) * x^2) ∧
    (∀ x, f₄ x = -x^2) ∧
    (abs a₃ < abs a₄ ∧ abs a₄ < abs a₂ ∧ abs a₂ < abs a₁)

/-- Theorem stating that f₃ has the widest opening -/
theorem widest_opening_is_f₃ (f₁ f₂ f₃ f₄ : ℝ → ℝ) :
  widest_opening f₁ f₂ f₃ f₄ → (∀ x, f₃ x = (1/100) * x^2) := by
  sorry

end NUMINAMATH_CALUDE_widest_opening_is_f₃_l943_94306


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l943_94364

theorem greatest_divisor_with_remainders : 
  Nat.gcd (976543 - 7) (897623 - 11) = 4 := by sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l943_94364


namespace NUMINAMATH_CALUDE_triangle_side_difference_l943_94386

theorem triangle_side_difference (x : ℕ) : 
  (x + 8 > 10) ∧ (x + 10 > 8) ∧ (8 + 10 > x) →
  (∃ (max min : ℕ), 
    (∀ y : ℕ, (y + 8 > 10) ∧ (y + 10 > 8) ∧ (8 + 10 > y) → y ≤ max ∧ y ≥ min) ∧
    (max - min = 14)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_difference_l943_94386


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l943_94305

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l943_94305


namespace NUMINAMATH_CALUDE_twenty_mps_equals_72_kmph_l943_94361

/-- Conversion from meters per second to kilometers per hour -/
def mps_to_kmph (speed_mps : ℝ) : ℝ :=
  speed_mps * 3.6

/-- Theorem: 20 mps is equal to 72 kmph -/
theorem twenty_mps_equals_72_kmph :
  mps_to_kmph 20 = 72 := by
  sorry

#eval mps_to_kmph 20

end NUMINAMATH_CALUDE_twenty_mps_equals_72_kmph_l943_94361


namespace NUMINAMATH_CALUDE_binary_1101011_equals_107_l943_94366

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101011_equals_107 :
  binary_to_decimal [true, true, false, true, false, true, true] = 107 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101011_equals_107_l943_94366


namespace NUMINAMATH_CALUDE_one_fifth_equals_point_two_l943_94302

theorem one_fifth_equals_point_two : (1 : ℚ) / 5 = (2 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_equals_point_two_l943_94302


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l943_94344

theorem complex_arithmetic_expression : -1^2009 * (-3) + 1 - 2^2 * 3 + (1 - 2^2) / 3 + (1 - 2 * 3)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l943_94344


namespace NUMINAMATH_CALUDE_rectangle_breadth_l943_94327

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) : 
  square_area = 16 → 
  rectangle_area = 220 → 
  ∃ (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ),
    circle_radius = Real.sqrt square_area ∧
    rectangle_length = 5 * circle_radius ∧
    rectangle_area = rectangle_length * rectangle_breadth ∧
    rectangle_breadth = 11 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l943_94327


namespace NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_1_any_4m_plus_1_has_prime_factor_4k_plus_1_infinitely_many_primes_4k_plus_1_from_divisibility_l943_94317

theorem infinitely_many_primes_4k_plus_1 :
  ∀ (S : Set Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 1) →
  (∀ n, ∃ p ∈ S, p > n) :=
by
  sorry

theorem any_4m_plus_1_has_prime_factor_4k_plus_1 :
  ∀ m : Nat, ∃ p : Nat, Nat.Prime p ∧ (∃ k : Nat, p = 4*k + 1) ∧ p ∣ (4*m + 1) :=
by
  sorry

theorem infinitely_many_primes_4k_plus_1_from_divisibility 
  (h : ∀ m : Nat, ∃ p : Nat, Nat.Prime p ∧ (∃ k : Nat, p = 4*k + 1) ∧ p ∣ (4*m + 1)) :
  ∀ (S : Set Nat), (∀ p ∈ S, Nat.Prime p ∧ ∃ k, p = 4*k + 1) →
  (∀ n, ∃ p ∈ S, p > n) :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_4k_plus_1_any_4m_plus_1_has_prime_factor_4k_plus_1_infinitely_many_primes_4k_plus_1_from_divisibility_l943_94317


namespace NUMINAMATH_CALUDE_min_value_and_inequality_inequality_holds_l943_94328

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1 - x^2) / x^2

theorem min_value_and_inequality (a : ℝ) :
  (∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f a x ≥ f a x_min ∧ f a x_min = 0) ↔ a = 2 :=
sorry

theorem inequality_holds (x : ℝ) (h : x > 0) : f 2 x ≥ 1 / x - Real.exp (1 - x) :=
sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_inequality_holds_l943_94328


namespace NUMINAMATH_CALUDE_quadratic_minimum_l943_94346

theorem quadratic_minimum (x : ℝ) : x^2 - 6*x + 5 ≥ -4 ∧ ∃ y : ℝ, y^2 - 6*y + 5 = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l943_94346


namespace NUMINAMATH_CALUDE_coin_stack_solution_l943_94385

/-- Thickness of a nickel in millimeters -/
def nickel_thickness : ℚ := 2.05

/-- Thickness of a quarter in millimeters -/
def quarter_thickness : ℚ := 1.65

/-- Height of the stack in millimeters -/
def stack_height : ℚ := 16.5

theorem coin_stack_solution :
  ∃! (n q : ℕ), 
    n * nickel_thickness + q * quarter_thickness = stack_height ∧
    n + q = 9 := by sorry

end NUMINAMATH_CALUDE_coin_stack_solution_l943_94385


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l943_94347

theorem necessary_but_not_sufficient :
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x - 3) < 0)) ∧
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l943_94347


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l943_94360

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (x : ℕ) : ℝ := 60 + 8 * x

/-- Gamma T-Shirt Company's pricing function -/
def gamma_cost (x : ℕ) : ℝ := 12 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme : ℕ := 16

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme < gamma_cost min_shirts_for_acme ∧
  ∀ n : ℕ, n < min_shirts_for_acme → acme_cost n ≥ gamma_cost n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l943_94360


namespace NUMINAMATH_CALUDE_monomial_degree_implies_a_value_l943_94373

/-- Given that (a-2)x^2y^(|a|+1) is a monomial of degree 5 in x and y, prove that a = -2 -/
theorem monomial_degree_implies_a_value (a : ℤ) : 
  (∃ (x y : ℚ), (a - 2) * x^2 * y^(|a| + 1) ≠ 0) ∧ 
  (2 + (|a| + 1) = 5) → 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_monomial_degree_implies_a_value_l943_94373


namespace NUMINAMATH_CALUDE_circle_center_on_line_l943_94355

/-- Given a circle x^2 + y^2 + Dx + Ey = 0 with center on the line x + y = l, prove D + E = -2 -/
theorem circle_center_on_line (D E l : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + D*x + E*y = 0 ∧ x + y = l) → D + E = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_on_line_l943_94355


namespace NUMINAMATH_CALUDE_multiple_of_2007_cube_difference_l943_94359

theorem multiple_of_2007_cube_difference (k : ℕ+) :
  (∃ a : ℤ, ∃ m : ℤ, (a + k.val : ℤ)^3 - a^3 = 2007 * m) ↔ ∃ n : ℕ, k.val = 669 * n :=
sorry

end NUMINAMATH_CALUDE_multiple_of_2007_cube_difference_l943_94359


namespace NUMINAMATH_CALUDE_square_in_triangle_angle_sum_l943_94358

/-- The sum of angles in a triangle --/
def triangle_angle_sum : ℝ := 180

/-- The interior angle of an equilateral triangle --/
def equilateral_triangle_angle : ℝ := 60

/-- The sum of angles on a straight line --/
def straight_line_angle_sum : ℝ := 180

/-- The angle of a right angle (in a square) --/
def right_angle : ℝ := 90

/-- Configuration of a square inscribed in an equilateral triangle --/
structure SquareInTriangle where
  x : ℝ  -- Angle between square side and triangle side
  y : ℝ  -- Angle between square side and triangle side
  p : ℝ  -- Complementary angle to x in the triangle
  q : ℝ  -- Complementary angle to y in the triangle

/-- Theorem: The sum of x and y in the SquareInTriangle configuration is 150° --/
theorem square_in_triangle_angle_sum (config : SquareInTriangle) : 
  config.x + config.y = 150 := by
  sorry

end NUMINAMATH_CALUDE_square_in_triangle_angle_sum_l943_94358


namespace NUMINAMATH_CALUDE_eggs_removed_l943_94333

theorem eggs_removed (original : ℕ) (remaining : ℕ) (removed : ℕ) : 
  original = 27 → remaining = 20 → removed = original - remaining → removed = 7 := by
sorry

end NUMINAMATH_CALUDE_eggs_removed_l943_94333


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l943_94351

/-- Represents a parabola in the form y = (x - h)² + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Shifts a point horizontally -/
def shift_horizontal (p : Point) (shift : ℝ) : Point :=
  { x := p.x + shift, y := p.y }

/-- Shifts a point vertically -/
def shift_vertical (p : Point) (shift : ℝ) : Point :=
  { x := p.x, y := p.y + shift }

/-- The vertex of a parabola -/
def vertex (p : Parabola) : Point :=
  { x := p.h, y := p.k }

theorem parabola_shift_theorem (p : Parabola) :
  p.h = 2 ∧ p.k = 3 →
  (shift_vertical (shift_horizontal (vertex p) (-3)) (-5)) = { x := -1, y := -2 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l943_94351


namespace NUMINAMATH_CALUDE_judy_shopping_cost_l943_94303

-- Define the prices and quantities
def carrot_price : ℝ := 1.50
def carrot_quantity : ℕ := 6
def milk_price : ℝ := 3.50
def milk_quantity : ℕ := 4
def pineapple_price : ℝ := 5.00
def pineapple_quantity : ℕ := 3
def pineapple_discount : ℝ := 0.25
def flour_price : ℝ := 6.00
def flour_quantity : ℕ := 3
def flour_discount : ℝ := 0.10
def ice_cream_price : ℝ := 8.00
def coupon_value : ℝ := 10.00
def coupon_threshold : ℝ := 50.00

-- Define the theorem
theorem judy_shopping_cost :
  let carrot_total := carrot_price * carrot_quantity
  let milk_total := milk_price * milk_quantity
  let pineapple_total := pineapple_price * (1 - pineapple_discount) * pineapple_quantity
  let flour_total := flour_price * (1 - flour_discount) * flour_quantity
  let subtotal := carrot_total + milk_total + pineapple_total + flour_total + ice_cream_price
  let final_total := if subtotal ≥ coupon_threshold then subtotal - coupon_value else subtotal
  final_total = 48.45 := by sorry

end NUMINAMATH_CALUDE_judy_shopping_cost_l943_94303


namespace NUMINAMATH_CALUDE_reverse_digit_increase_l943_94308

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a + b + c = 10 ∧
    b = a + c ∧
    n = 253

theorem reverse_digit_increase (n : ℕ) (h : is_valid_number n) :
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    100 * c + 10 * b + a - n = 99 := by
  sorry

end NUMINAMATH_CALUDE_reverse_digit_increase_l943_94308


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l943_94354

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

theorem sixth_term_of_geometric_sequence (a₁ a₄ : ℝ) (h₁ : a₁ = 8) (h₂ : a₄ = 64) :
  ∃ r : ℝ, geometric_sequence a₁ r 4 = a₄ ∧ geometric_sequence a₁ r 6 = 256 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l943_94354


namespace NUMINAMATH_CALUDE_volleyball_tournament_l943_94377

theorem volleyball_tournament (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_tournament_l943_94377


namespace NUMINAMATH_CALUDE_power_of_two_equation_l943_94319

theorem power_of_two_equation (y : ℤ) : (1 / 8 : ℚ) * 2^36 = 2^y → y = 33 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l943_94319


namespace NUMINAMATH_CALUDE_tangent_line_coefficients_l943_94398

/-- Given a curve y = x^2 + ax + b with a tangent line at (1, b) with equation x - y + 1 = 0,
    prove that a = -1 and b = 2 -/
theorem tangent_line_coefficients (a b : ℝ) : 
  (∀ x y : ℝ, y = x^2 + a*x + b) →
  (∃ y : ℝ, y = 1^2 + a*1 + b) →
  (∀ x y : ℝ, y = 1^2 + a*1 + b → x - y + 1 = 0 → x = 1) →
  a = -1 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_coefficients_l943_94398


namespace NUMINAMATH_CALUDE_sum_of_squares_130_l943_94314

theorem sum_of_squares_130 : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a > 0 ∧ 
  b > 0 ∧ 
  a^2 + b^2 = 130 ∧ 
  a + b = 16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_130_l943_94314


namespace NUMINAMATH_CALUDE_fiftiethTermIs346_l943_94350

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

/-- The 50th term of the specific arithmetic sequence -/
def fiftiethTerm : ℤ := arithmeticSequenceTerm 3 7 50

theorem fiftiethTermIs346 : fiftiethTerm = 346 := by
  sorry

end NUMINAMATH_CALUDE_fiftiethTermIs346_l943_94350


namespace NUMINAMATH_CALUDE_triangle_inradius_inequality_l943_94313

-- Define a triangle with sides a, b, c and inradius r
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  -- Ensure that a, b, c form a valid triangle
  triangle_inequality : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b
  -- Ensure that r is positive
  positive_inradius : r > 0

-- State the theorem
theorem triangle_inradius_inequality (t : Triangle) :
  1 / t.a^2 + 1 / t.b^2 + 1 / t.c^2 ≤ 1 / (4 * t.r^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_inequality_l943_94313


namespace NUMINAMATH_CALUDE_parking_lot_tires_l943_94329

/-- Calculates the total number of tires in a parking lot with various vehicles -/
def total_tires (cars motorcycles trucks bicycles unicycles strollers : ℕ) 
  (cars_extra_tire bicycles_flat : ℕ) (unicycles_extra : ℕ) : ℕ :=
  -- 4-wheel drive cars
  (cars * 5 + cars_extra_tire) + 
  -- Motorcycles
  (motorcycles * 4) + 
  -- 6-wheel trucks
  (trucks * 7) + 
  -- Bicycles
  (bicycles * 2 - bicycles_flat) + 
  -- Unicycles
  (unicycles + unicycles_extra) + 
  -- Baby strollers
  (strollers * 4)

/-- Theorem stating the total number of tires in the parking lot -/
theorem parking_lot_tires : 
  total_tires 30 20 10 5 3 2 4 3 1 = 323 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_tires_l943_94329


namespace NUMINAMATH_CALUDE_base_seven_digits_of_2000_l943_94310

theorem base_seven_digits_of_2000 : ∃ n : ℕ, 
  (7^(n-1) ≤ 2000) ∧ (2000 < 7^n) ∧ (n = 4) := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_2000_l943_94310


namespace NUMINAMATH_CALUDE_train_average_speed_l943_94389

theorem train_average_speed :
  let distance1 : ℝ := 290
  let time1 : ℝ := 4.5
  let distance2 : ℝ := 400
  let time2 : ℝ := 5.5
  let total_distance : ℝ := distance1 + distance2
  let total_time : ℝ := time1 + time2
  let average_speed : ℝ := total_distance / total_time
  average_speed = 69 := by sorry

end NUMINAMATH_CALUDE_train_average_speed_l943_94389


namespace NUMINAMATH_CALUDE_possible_distances_l943_94322

theorem possible_distances (p q r s t : ℝ) 
  (h1 : |p - q| = 3)
  (h2 : |q - r| = 4)
  (h3 : |r - s| = 5)
  (h4 : |s - t| = 6) :
  ∃ (S : Set ℝ), S = {0, 2, 4, 6, 8, 10, 12, 18} ∧ |p - t| ∈ S :=
by sorry

end NUMINAMATH_CALUDE_possible_distances_l943_94322


namespace NUMINAMATH_CALUDE_line_vector_at_t_one_l943_94368

/-- A parameterized line in 2D space -/
structure ParameterizedLine where
  vector : ℝ → Fin 2 → ℝ

/-- The theorem stating the properties of the given line and the vector to be proved -/
theorem line_vector_at_t_one
  (L : ParameterizedLine)
  (h1 : L.vector 4 = ![2, 5])
  (h2 : L.vector 5 = ![4, -3]) :
  L.vector 1 = ![8, -19] := by
  sorry

end NUMINAMATH_CALUDE_line_vector_at_t_one_l943_94368


namespace NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_digits_of_large_number_l943_94382

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number has exactly ten billion digits -/
def has_ten_billion_digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_sum_of_digits_of_large_number 
  (A : ℕ) 
  (h1 : has_ten_billion_digits A) 
  (h2 : A % 9 = 0) : 
  let B := sum_of_digits A
  let C := sum_of_digits B
  sum_of_digits C = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_sum_of_digits_of_large_number_l943_94382


namespace NUMINAMATH_CALUDE_exist_two_N_l943_94304

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the condition for point M
def M_condition (x y : ℝ) : Prop :=
  Real.sqrt ((x+1)^2 + y^2) + Real.sqrt ((x-1)^2 + y^2) = 2 * Real.sqrt 2

-- Define the line l
def line_l (x : ℝ) : Prop := x = -1/2

-- Define the property that N is the midpoint of AB
def is_midpoint (N A B : ℝ × ℝ) : Prop :=
  N.1 = (A.1 + B.1) / 2 ∧ N.2 = (A.2 + B.2) / 2

-- Define the property that PQ is perpendicular bisector of AB
def is_perp_bisector (P Q A B : ℝ × ℝ) : Prop :=
  (P.1 - Q.1) * (A.1 - B.1) + (P.2 - Q.2) * (A.2 - B.2) = 0 ∧
  (P.1 + Q.1) / 2 = (A.1 + B.1) / 2 ∧
  (P.2 + Q.2) / 2 = (A.2 + B.2) / 2

-- Define the property that (1,0) is on the circle with diameter PQ
def on_circle_PQ (P Q : ℝ × ℝ) : Prop :=
  (1 - P.1) * (1 - Q.1) + (-P.2) * (-Q.2) = 0

-- Main theorem
theorem exist_two_N :
  ∃ N1 N2 : ℝ × ℝ,
    N1 ≠ N2 ∧
    line_l N1.1 ∧ line_l N2.1 ∧
    (∃ A B P Q : ℝ × ℝ,
      E A.1 A.2 ∧ E B.1 B.2 ∧ E P.1 P.2 ∧ E Q.1 Q.2 ∧
      is_midpoint N1 A B ∧
      is_perp_bisector P Q A B ∧
      on_circle_PQ P Q) ∧
    (∃ A B P Q : ℝ × ℝ,
      E A.1 A.2 ∧ E B.1 B.2 ∧ E P.1 P.2 ∧ E Q.1 Q.2 ∧
      is_midpoint N2 A B ∧
      is_perp_bisector P Q A B ∧
      on_circle_PQ P Q) ∧
    N1 = (-1/2, Real.sqrt 19 / 19) ∧
    N2 = (-1/2, -Real.sqrt 19 / 19) ∧
    (∀ N : ℝ × ℝ,
      line_l N.1 →
      (∃ A B P Q : ℝ × ℝ,
        E A.1 A.2 ∧ E B.1 B.2 ∧ E P.1 P.2 ∧ E Q.1 Q.2 ∧
        is_midpoint N A B ∧
        is_perp_bisector P Q A B ∧
        on_circle_PQ P Q) →
      N = N1 ∨ N = N2) :=
sorry

end NUMINAMATH_CALUDE_exist_two_N_l943_94304


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l943_94341

/-- The number of terms in the arithmetic sequence 2.5, 7.5, 12.5, ..., 57.5, 62.5 -/
def sequenceLength : ℕ := 13

/-- The first term of the sequence -/
def firstTerm : ℚ := 2.5

/-- The last term of the sequence -/
def lastTerm : ℚ := 62.5

/-- The common difference of the sequence -/
def commonDifference : ℚ := 5

theorem arithmetic_sequence_length :
  sequenceLength = (lastTerm - firstTerm) / commonDifference + 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l943_94341


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l943_94311

theorem fraction_inequality_solution_set :
  {x : ℝ | (x - 1) / (1 - 2*x) ≥ 0} = Set.Ioo (1/2) 1 ∪ {1} :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l943_94311


namespace NUMINAMATH_CALUDE_geometric_series_sum_proof_l943_94348

/-- The sum of the infinite geometric series 5/3 - 5/6 + 5/18 - 5/54 + ... -/
def geometric_series_sum : ℚ := 10/9

/-- The first term of the geometric series -/
def a : ℚ := 5/3

/-- The common ratio of the geometric series -/
def r : ℚ := -1/2

theorem geometric_series_sum_proof :
  geometric_series_sum = a / (1 - r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_proof_l943_94348


namespace NUMINAMATH_CALUDE_inequality_implies_a_bounds_l943_94345

-- Define the operation ⊕
def circleplus (x y : ℝ) : ℝ := (x + 3) * (y - 1)

-- State the theorem
theorem inequality_implies_a_bounds :
  (∀ x : ℝ, circleplus (x - a) (x + a) > -16) → -2 < a ∧ a < 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_a_bounds_l943_94345


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_difference_l943_94390

theorem cube_root_equation_solution_difference : ∃ x₁ x₂ : ℝ,
  (x₁ ≠ x₂) ∧
  ((9 - x₁^2 / 4)^(1/3 : ℝ) = -3) ∧
  ((9 - x₂^2 / 4)^(1/3 : ℝ) = -3) ∧
  (abs (x₁ - x₂) = 24) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_difference_l943_94390


namespace NUMINAMATH_CALUDE_at_least_one_positive_discriminant_l943_94352

theorem at_least_one_positive_discriminant 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (4 * b^2 - 4 * a * c > 0) ∨ 
  (4 * c^2 - 4 * a * b > 0) ∨ 
  (4 * a^2 - 4 * b * c > 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_positive_discriminant_l943_94352


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l943_94301

theorem cube_sum_divisibility (x y z : ℤ) (h : x^3 + y^3 = z^3) :
  3 ∣ x ∨ 3 ∣ y ∨ 3 ∣ z := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l943_94301


namespace NUMINAMATH_CALUDE_ivan_nails_purchase_l943_94396

-- Define the cost of nails per 100 grams in each store
def cost_store1 : ℝ := 180
def cost_store2 : ℝ := 120

-- Define the amount Ivan was short in the first store
def short_amount : ℝ := 1430

-- Define the change Ivan received in the second store
def change_amount : ℝ := 490

-- Define the function to calculate the cost of nails in kilograms
def cost_per_kg (cost_per_100g : ℝ) : ℝ := cost_per_100g * 10

-- Define the amount of nails Ivan bought in kilograms
def nails_bought : ℝ := 3.2

-- Theorem statement
theorem ivan_nails_purchase :
  (cost_per_kg cost_store1 * nails_bought - (cost_per_kg cost_store2 * nails_bought + change_amount) = short_amount) ∧
  (nails_bought = 3.2) :=
by sorry

end NUMINAMATH_CALUDE_ivan_nails_purchase_l943_94396


namespace NUMINAMATH_CALUDE_initial_bacteria_count_l943_94399

/-- The number of bacteria after a given number of doubling periods -/
def bacteria_count (initial_count : ℕ) (periods : ℕ) : ℕ :=
  initial_count * 2^periods

theorem initial_bacteria_count :
  ∃ (initial_count : ℕ),
    bacteria_count initial_count 8 = 262144 ∧
    initial_count = 1024 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_bacteria_count_l943_94399


namespace NUMINAMATH_CALUDE_negative_four_cubed_equality_l943_94315

theorem negative_four_cubed_equality : (-4)^3 = -4^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_four_cubed_equality_l943_94315


namespace NUMINAMATH_CALUDE_average_of_numbers_l943_94367

def number1 : Nat := 8642097531
def number2 : Nat := 6420875319
def number3 : Nat := 4208653197
def number4 : Nat := 2086431975
def number5 : Nat := 864219753

def numbers : List Nat := [number1, number2, number3, number4, number5]

theorem average_of_numbers : 
  (numbers.sum / numbers.length : Rat) = 4444455555 := by sorry

end NUMINAMATH_CALUDE_average_of_numbers_l943_94367


namespace NUMINAMATH_CALUDE_village_population_l943_94356

/-- Given a village population with specific demographic percentages,
    calculate the total population. -/
theorem village_population (adult_percentage : ℝ) (adult_women_percentage : ℝ)
    (adult_women_count : ℕ) :
    adult_percentage = 0.9 →
    adult_women_percentage = 0.6 →
    adult_women_count = 21600 →
    ∃ total_population : ℕ,
      total_population = 40000 ∧
      (adult_percentage * adult_women_percentage * total_population : ℝ) = adult_women_count :=
by
  sorry

end NUMINAMATH_CALUDE_village_population_l943_94356


namespace NUMINAMATH_CALUDE_cube_root_of_negative_27_l943_94307

theorem cube_root_of_negative_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_27_l943_94307


namespace NUMINAMATH_CALUDE_min_value_of_expression_min_value_is_nine_min_value_achieved_l943_94357

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_perp : m * n + (1 - n) * 1 = 0) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y + (1 - y) * 1 = 0 → 1/x + 4*y ≥ 1/m + 4*n :=
by sorry

theorem min_value_is_nine (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_perp : m * n + (1 - n) * 1 = 0) : 
  1/m + 4*n ≥ 9 :=
by sorry

theorem min_value_achieved (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_perp : m * n + (1 - n) * 1 = 0) : 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y + (1 - y) * 1 = 0 ∧ 1/x + 4*y = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_min_value_is_nine_min_value_achieved_l943_94357


namespace NUMINAMATH_CALUDE_multiplier_problem_l943_94370

theorem multiplier_problem (a b : ℝ) (h1 : 4 * a = b) (h2 : b = 30) (h3 : 40 * a * b = 1800) :
  ∃ m : ℝ, m * b = 30 ∧ m = 5 := by
sorry

end NUMINAMATH_CALUDE_multiplier_problem_l943_94370


namespace NUMINAMATH_CALUDE_shirt_tie_outfits_l943_94365

theorem shirt_tie_outfits (num_shirts : ℕ) (num_ties : ℕ) 
  (h1 : num_shirts = 6) (h2 : num_ties = 5) : 
  num_shirts * num_ties = 30 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_outfits_l943_94365


namespace NUMINAMATH_CALUDE_sum_angles_less_than_1100_l943_94393

/-- Represents the angle measurement scenario with a car and a fence -/
structure AngleMeasurement where
  carSpeed : ℝ  -- Car speed in km/h
  fenceLength : ℝ  -- Fence length in meters
  measurementInterval : ℝ  -- Measurement interval in seconds

/-- Calculates the sum of angles measured -/
def sumOfAngles (scenario : AngleMeasurement) : ℝ :=
  sorry  -- Proof omitted

/-- Theorem stating that the sum of angles is less than 1100 degrees -/
theorem sum_angles_less_than_1100 (scenario : AngleMeasurement) 
  (h1 : scenario.carSpeed = 60)
  (h2 : scenario.fenceLength = 100)
  (h3 : scenario.measurementInterval = 1) :
  sumOfAngles scenario < 1100 := by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_sum_angles_less_than_1100_l943_94393


namespace NUMINAMATH_CALUDE_integral_even_odd_functions_l943_94340

open Set
open Interval
open MeasureTheory
open Measure

/-- A function f is even on [-a,a] -/
def IsEven (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a > 0 ∧ ∀ x ∈ Icc (-a) a, f (-x) = f x

/-- A function f is odd on [-a,a] -/
def IsOdd (f : ℝ → ℝ) (a : ℝ) : Prop :=
  a > 0 ∧ ∀ x ∈ Icc (-a) a, f (-x) = -f x

theorem integral_even_odd_functions (f : ℝ → ℝ) (a : ℝ) :
  (IsEven f a → ∫ x in Icc (-a) a, f x = 2 * ∫ x in Icc 0 a, f x) ∧
  (IsOdd f a → ∫ x in Icc (-a) a, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_integral_even_odd_functions_l943_94340


namespace NUMINAMATH_CALUDE_rectangle_length_l943_94338

theorem rectangle_length (L B : ℝ) (h1 : L / B = 25 / 16) (h2 : L * B = 200^2) : L = 250 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l943_94338


namespace NUMINAMATH_CALUDE_marble_game_winner_l943_94339

/-- Represents the distribution of marbles in a single game -/
structure GameDistribution where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the total marbles each player has after all games -/
structure FinalMarbles where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The main theorem statement -/
theorem marble_game_winner
  (p q r : ℕ)
  (h_p_lt_q : p < q)
  (h_q_lt_r : q < r)
  (h_p_pos : 0 < p)
  (h_sum : p + q + r = 13)
  (final_marbles : FinalMarbles)
  (h_final_a : final_marbles.a = 20)
  (h_final_b : final_marbles.b = 10)
  (h_final_c : final_marbles.c = 9)
  (h_b_last : ∃ (g1 g2 : GameDistribution), g1.b + g2.b + r = 10)
  : ∃ (g1 g2 : GameDistribution),
    g1.c = q ∧ g2.c ≠ q ∧
    g1.a + g2.a + final_marbles.a - 20 = p + q + r ∧
    g1.b + g2.b + final_marbles.b - 10 = p + q + r ∧
    g1.c + g2.c + final_marbles.c - 9 = p + q + r :=
sorry

end NUMINAMATH_CALUDE_marble_game_winner_l943_94339


namespace NUMINAMATH_CALUDE_elena_and_alex_money_l943_94316

theorem elena_and_alex_money : (5 : ℚ) / 6 + (7 : ℚ) / 15 = (13 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_elena_and_alex_money_l943_94316


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_17_l943_94392

theorem least_five_digit_congruent_to_6_mod_17 :
  (∀ n : ℕ, 10000 ≤ n ∧ n < 10017 → ¬(n % 17 = 6)) ∧
  10017 % 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_6_mod_17_l943_94392


namespace NUMINAMATH_CALUDE_lcm_16_24_l943_94320

theorem lcm_16_24 : Nat.lcm 16 24 = 48 := by
  sorry

end NUMINAMATH_CALUDE_lcm_16_24_l943_94320


namespace NUMINAMATH_CALUDE_last_digit_of_expression_l943_94363

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_expression : last_digit (287 * 287 + 269 * 269 - 2 * 287 * 269) = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_expression_l943_94363


namespace NUMINAMATH_CALUDE_range_of_a_l943_94394

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x ∈ Set.Ioo (-1) 1 ∧ a * x^2 - 1 ≥ 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l943_94394


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l943_94321

/-- The area of a triangle with base 10 and height 5 is 25 -/
theorem triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 10 ∧ height = 5 → area = (base * height) / 2 → area = 25

#check triangle_area

theorem triangle_area_proof : triangle_area 10 5 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_proof_l943_94321


namespace NUMINAMATH_CALUDE_total_molecular_weight_l943_94395

/-- Atomic weight in g/mol -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "Ca" => 40.08
  | "I"  => 126.90
  | "Na" => 22.99
  | "Cl" => 35.45
  | "K"  => 39.10
  | "S"  => 32.06
  | "O"  => 16.00
  | _    => 0  -- Default case

/-- Molecular weight of a compound in g/mol -/
def molecular_weight (compound : String) : ℝ :=
  match compound with
  | "CaI2" => atomic_weight "Ca" + 2 * atomic_weight "I"
  | "NaCl" => atomic_weight "Na" + atomic_weight "Cl"
  | "K2SO4" => 2 * atomic_weight "K" + atomic_weight "S" + 4 * atomic_weight "O"
  | _      => 0  -- Default case

/-- Total weight of a given number of moles of a compound in grams -/
def total_weight (compound : String) (moles : ℝ) : ℝ :=
  moles * molecular_weight compound

/-- Theorem: The total molecular weight of 10 moles of CaI2, 7 moles of NaCl, and 15 moles of K2SO4 is 5961.78 grams -/
theorem total_molecular_weight : 
  total_weight "CaI2" 10 + total_weight "NaCl" 7 + total_weight "K2SO4" 15 = 5961.78 := by
  sorry

end NUMINAMATH_CALUDE_total_molecular_weight_l943_94395


namespace NUMINAMATH_CALUDE_jane_has_nine_cans_l943_94325

/-- The number of sunflower seeds Jane has -/
def total_seeds : ℕ := 54

/-- The number of seeds Jane places in each can -/
def seeds_per_can : ℕ := 6

/-- The number of cans Jane has -/
def number_of_cans : ℕ := total_seeds / seeds_per_can

/-- Proof that Jane has 9 cans -/
theorem jane_has_nine_cans : number_of_cans = 9 := by
  sorry

end NUMINAMATH_CALUDE_jane_has_nine_cans_l943_94325


namespace NUMINAMATH_CALUDE_four_at_seven_l943_94336

-- Define the binary operation @
def binaryOp (a b : ℤ) : ℤ := 4 * a - 2 * b

-- Theorem statement
theorem four_at_seven : binaryOp 4 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_at_seven_l943_94336


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l943_94387

theorem slower_speed_calculation (distance : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  distance = 50 →
  faster_speed = 14 →
  extra_distance = 20 →
  ∃ slower_speed : ℝ,
    slower_speed > 0 ∧
    distance / slower_speed = distance / faster_speed + extra_distance / faster_speed ∧
    slower_speed = 10 :=
by sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l943_94387


namespace NUMINAMATH_CALUDE_truncated_tetrahedron_edge_count_l943_94335

/-- A tetrahedron with truncated vertices -/
structure TruncatedTetrahedron where
  /-- The number of truncated vertices -/
  truncatedVertices : ℕ
  /-- Assertion that all vertices are truncated -/
  all_truncated : truncatedVertices = 4
  /-- Assertion that truncations are distinct and non-intersecting -/
  distinct_truncations : True

/-- The number of edges in a truncated tetrahedron -/
def edgeCount (t : TruncatedTetrahedron) : ℕ := sorry

/-- Theorem stating that a truncated tetrahedron has 18 edges -/
theorem truncated_tetrahedron_edge_count (t : TruncatedTetrahedron) : 
  edgeCount t = 18 := by sorry

end NUMINAMATH_CALUDE_truncated_tetrahedron_edge_count_l943_94335


namespace NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l943_94371

/-- Given a line in vector form, prove it's equivalent to slope-intercept form -/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 4) + (5 : ℝ) * (y - 1) = 0 ↔ 
  y = -(2/5 : ℝ) * x + (13/5 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_line_vector_to_slope_intercept_l943_94371


namespace NUMINAMATH_CALUDE_sequence_growth_l943_94326

/-- A sequence of integers satisfying the given conditions -/
def Sequence (a : ℕ → ℤ) : Prop :=
  a 1 > a 0 ∧ a 1 > 0 ∧ ∀ r : ℕ, r ≤ 98 → a (r + 2) = 3 * a (r + 1) - 2 * a r

/-- The main theorem -/
theorem sequence_growth (a : ℕ → ℤ) (h : Sequence a) : a 100 > 2^99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_growth_l943_94326


namespace NUMINAMATH_CALUDE_max_sum_xyz_l943_94332

theorem max_sum_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_sum_xyz_l943_94332


namespace NUMINAMATH_CALUDE_paint_problem_solution_l943_94369

def paint_problem (total_paint : ℚ) (second_week_fraction : ℚ) (total_used : ℚ) (first_week_fraction : ℚ) : Prop :=
  total_paint > 0 ∧
  second_week_fraction > 0 ∧ second_week_fraction < 1 ∧
  total_used > 0 ∧ total_used < total_paint ∧
  first_week_fraction > 0 ∧ first_week_fraction < 1 ∧
  first_week_fraction * total_paint + second_week_fraction * (total_paint - first_week_fraction * total_paint) = total_used

theorem paint_problem_solution :
  paint_problem 360 (1/3) 180 (1/4) := by
  sorry

end NUMINAMATH_CALUDE_paint_problem_solution_l943_94369


namespace NUMINAMATH_CALUDE_tennis_uniform_numbers_l943_94381

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem tennis_uniform_numbers 
  (e f g h : ℕ) 
  (e_birthday today_date g_birthday : ℕ)
  (h_two_digit_prime : is_two_digit_prime e ∧ is_two_digit_prime f ∧ is_two_digit_prime g ∧ is_two_digit_prime h)
  (h_sum_all : e + f + g + h = e_birthday)
  (h_sum_ef : e + f = today_date)
  (h_sum_gf : g + f = g_birthday)
  (h_sum_hg : h + g = e_birthday) :
  h = 19 := by
  sorry

end NUMINAMATH_CALUDE_tennis_uniform_numbers_l943_94381


namespace NUMINAMATH_CALUDE_tabitha_honey_nights_l943_94353

/-- The number of nights Tabitha can enjoy honey in her tea before bed -/
def honey_nights (servings_per_cup : ℕ) (cups_per_night : ℕ) (container_size : ℕ) (servings_per_ounce : ℕ) : ℕ :=
  (container_size * servings_per_ounce) / (servings_per_cup * cups_per_night)

/-- Theorem stating that Tabitha can enjoy honey in her tea for 48 nights before bed -/
theorem tabitha_honey_nights :
  honey_nights 1 2 16 6 = 48 := by sorry

end NUMINAMATH_CALUDE_tabitha_honey_nights_l943_94353


namespace NUMINAMATH_CALUDE_ball_diameter_l943_94362

theorem ball_diameter (h : Real) (d : Real) (r : Real) : 
  h = 2 → d = 8 → r^2 = (d/2)^2 + (r - h)^2 → 2*r = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_diameter_l943_94362


namespace NUMINAMATH_CALUDE_solution_set_implies_m_range_l943_94331

theorem solution_set_implies_m_range :
  (∀ x : ℝ, x^2 + m*x + 1 > 0) → m ∈ Set.Ioo (-2 : ℝ) 2 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_range_l943_94331


namespace NUMINAMATH_CALUDE_candy_division_l943_94383

theorem candy_division (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (hpq : p < q) (hqr : q < r) :
  (∃ n : ℕ, n > 0 ∧ n * (r + q - 2 * p) = 39) →
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x + y + (r - p) = 10) →
  (∃ z : ℕ, z > 0 ∧ 18 - 3 * p = 9) →
  (p = 3 ∧ q = 6 ∧ r = 13) := by
sorry

end NUMINAMATH_CALUDE_candy_division_l943_94383


namespace NUMINAMATH_CALUDE_same_terminal_side_angles_l943_94372

/-- 
Theorem: The angles -675° and -315° are the only angles in the range [-720°, 0°) 
that have the same terminal side as 45°.
-/
theorem same_terminal_side_angles : 
  ∀ θ : ℝ, -720 ≤ θ ∧ θ < 0 → 
  (∃ k : ℤ, θ = 45 + 360 * k) ↔ (θ = -675 ∨ θ = -315) := by
  sorry


end NUMINAMATH_CALUDE_same_terminal_side_angles_l943_94372


namespace NUMINAMATH_CALUDE_worker_a_time_l943_94309

theorem worker_a_time (b_time : ℝ) (combined_time : ℝ) (a_time : ℝ) : 
  b_time = 10 →
  combined_time = 4.444444444444445 →
  (1 / a_time + 1 / b_time = 1 / combined_time) →
  a_time = 8 := by
    sorry

end NUMINAMATH_CALUDE_worker_a_time_l943_94309


namespace NUMINAMATH_CALUDE_triangle_inequality_l943_94349

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a * b + b * c + c * a ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l943_94349


namespace NUMINAMATH_CALUDE_return_journey_speed_l943_94391

/-- Calculates the average speed of a return journey given the conditions of the problem -/
theorem return_journey_speed 
  (morning_time : ℝ) 
  (evening_time : ℝ) 
  (morning_speed : ℝ) 
  (h1 : morning_time = 1) 
  (h2 : evening_time = 1.5) 
  (h3 : morning_speed = 30) : 
  (morning_speed * morning_time) / evening_time = 20 :=
by
  sorry

#check return_journey_speed

end NUMINAMATH_CALUDE_return_journey_speed_l943_94391


namespace NUMINAMATH_CALUDE_perpendicular_slope_l943_94374

/-- Given a line with equation 5x - 2y = 10, the slope of a perpendicular line is -2/5 -/
theorem perpendicular_slope (x y : ℝ) :
  (5 * x - 2 * y = 10) → 
  (slope_of_perpendicular_line : ℝ) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l943_94374


namespace NUMINAMATH_CALUDE_ribbon_shortage_l943_94324

theorem ribbon_shortage (total_ribbon : ℝ) (num_gifts : ℕ) (ribbon_per_gift : ℝ) (ribbon_per_bow : ℝ) :
  total_ribbon = 18 →
  num_gifts = 6 →
  ribbon_per_gift = 2 →
  ribbon_per_bow = 1.5 →
  total_ribbon - (num_gifts * ribbon_per_gift + num_gifts * ribbon_per_bow) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_shortage_l943_94324


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l943_94343

/-- The repeating decimal 0.51246246246... -/
def repeating_decimal : ℚ := 
  51246 / 100000 + (246 / 100000) * (1 / (1 - 1 / 1000))

/-- The fraction representation -/
def fraction : ℚ := 511734 / 99900

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l943_94343


namespace NUMINAMATH_CALUDE_min_value_of_sum_absolute_differences_l943_94334

theorem min_value_of_sum_absolute_differences (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, |x - 2| + |x - 3| + |x - 4| < a) → a > 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_absolute_differences_l943_94334


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l943_94388

theorem min_value_of_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / x + 9 / y) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l943_94388


namespace NUMINAMATH_CALUDE_olympic_mascot_problem_l943_94318

theorem olympic_mascot_problem (total_items wholesale_cost : ℕ) 
  (wholesale_price_A wholesale_price_B : ℕ) 
  (retail_price_A retail_price_B : ℕ) (min_profit : ℕ) :
  total_items = 100 ∧ 
  wholesale_cost = 5650 ∧
  wholesale_price_A = 60 ∧ 
  wholesale_price_B = 50 ∧
  retail_price_A = 80 ∧
  retail_price_B = 60 ∧
  min_profit = 1400 →
  (∃ (num_A num_B : ℕ),
    num_A + num_B = total_items ∧
    num_A * wholesale_price_A + num_B * wholesale_price_B = wholesale_cost ∧
    num_A = 65 ∧ num_B = 35) ∧
  (∃ (min_A : ℕ),
    min_A ≥ 40 ∧
    ∀ (num_A : ℕ),
      num_A ≥ min_A →
      (num_A * (retail_price_A - wholesale_price_A) + 
       (total_items - num_A) * (retail_price_B - wholesale_price_B)) ≥ min_profit) :=
by sorry

end NUMINAMATH_CALUDE_olympic_mascot_problem_l943_94318


namespace NUMINAMATH_CALUDE_min_n_is_minimum_l943_94337

/-- The minimum positive integer n for which the expansion of (2x - 1/∛x)^n contains a constant term -/
def min_n : ℕ := 4

/-- Predicate to check if the expansion of (2x - 1/∛x)^n contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, n = 4 * r / 3 ∧ r > 0

/-- Theorem stating that min_n is the minimum positive integer satisfying the condition -/
theorem min_n_is_minimum :
  (∀ k : ℕ, k > 0 ∧ k < min_n → ¬(has_constant_term k)) ∧
  has_constant_term min_n :=
sorry

end NUMINAMATH_CALUDE_min_n_is_minimum_l943_94337


namespace NUMINAMATH_CALUDE_unpainted_cubes_count_l943_94384

/-- Represents a large cube composed of unit cubes -/
structure LargeCube where
  side_length : ℕ
  total_units : ℕ
  painted_on_opposite_faces : ℕ
  painted_on_other_faces : ℕ

/-- Calculates the number of unpainted unit cubes in the large cube -/
def unpainted_cubes (c : LargeCube) : ℕ :=
  c.total_units - (2 * c.painted_on_opposite_faces + 4 * c.painted_on_other_faces - 8)

/-- The theorem to be proved -/
theorem unpainted_cubes_count (c : LargeCube) 
  (h1 : c.side_length = 6)
  (h2 : c.total_units = 216)
  (h3 : c.painted_on_opposite_faces = 16)
  (h4 : c.painted_on_other_faces = 9) :
  unpainted_cubes c = 156 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_count_l943_94384


namespace NUMINAMATH_CALUDE_west_distance_notation_l943_94378

-- Define a type for direction
inductive Direction
| East
| West

-- Define a function to convert distance and direction to a signed number
def signedDistance (distance : ℝ) (direction : Direction) : ℝ :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- State the theorem
theorem west_distance_notation :
  signedDistance 6 Direction.East = 6 →
  signedDistance 10 Direction.West = -10 := by
  sorry

end NUMINAMATH_CALUDE_west_distance_notation_l943_94378


namespace NUMINAMATH_CALUDE_typing_competition_equation_l943_94397

/-- Prove that in a typing competition where A types x characters per minute and 
    B types (x-10) characters per minute, if A types 900 characters and B types 840 characters 
    in the same amount of time, then the equation 900/x = 840/(x-10) holds. -/
theorem typing_competition_equation (x : ℝ) 
    (hx : x > 10) -- Ensure x - 10 is positive
    (hA : 900 / x = 840 / (x - 10)) : -- Time taken by A equals time taken by B
  900 / x = 840 / (x - 10) := by
  sorry

end NUMINAMATH_CALUDE_typing_competition_equation_l943_94397


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l943_94380

/-- Given two arithmetic sequences, prove the ratio of their 5th terms -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℝ) (S T : ℕ → ℝ) :
  (∀ n, S n / T n = (7 * n) / (n + 3)) →  -- Given condition
  (∀ n, S n = (n / 2) * (a 1 + a n)) →    -- Definition of S_n for arithmetic sequence
  (∀ n, T n = (n / 2) * (b 1 + b n)) →    -- Definition of T_n for arithmetic sequence
  a 5 / b 5 = 21 / 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l943_94380


namespace NUMINAMATH_CALUDE_geometric_series_sum_l943_94379

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum : 
  let a : ℚ := 1/5
  let r : ℚ := -1/3
  let n : ℕ := 6
  geometric_sum a r n = 182/1215 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l943_94379


namespace NUMINAMATH_CALUDE_no_prime_for_expression_l943_94312

theorem no_prime_for_expression (m n : ℕ) : 
  ¬(Nat.Prime (n^2 + 2018*m*n + 2019*m + n - 2019*m^2)) := by
sorry

end NUMINAMATH_CALUDE_no_prime_for_expression_l943_94312


namespace NUMINAMATH_CALUDE_lcm_of_6_10_15_l943_94342

theorem lcm_of_6_10_15 : Nat.lcm (Nat.lcm 6 10) 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_6_10_15_l943_94342
