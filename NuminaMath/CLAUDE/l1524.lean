import Mathlib

namespace NUMINAMATH_CALUDE_fixed_point_of_parabola_family_l1524_152488

/-- The fixed point of the family of parabolas y = 4x^2 + 2tx - 3t is (1.5, 9) -/
theorem fixed_point_of_parabola_family (t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 + 2 * t * x - 3 * t
  f 1.5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabola_family_l1524_152488


namespace NUMINAMATH_CALUDE_altitude_from_C_to_AB_l1524_152494

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 5)
def B : ℝ × ℝ := (1, -2)
def C : ℝ × ℝ := (-6, 4)

-- Define the equation of the altitude
def altitude_equation (x y : ℝ) : Prop := 7 * x - 6 * y + 30 = 0

-- Theorem statement
theorem altitude_from_C_to_AB :
  ∀ x y : ℝ, altitude_equation x y ↔ 
  (x - C.1) * (B.1 - A.1) + (y - C.2) * (B.2 - A.2) = 0 ∧
  (x, y) ≠ C :=
sorry

end NUMINAMATH_CALUDE_altitude_from_C_to_AB_l1524_152494


namespace NUMINAMATH_CALUDE_rational_solution_quadratic_l1524_152416

theorem rational_solution_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + 4 * k = 0) ↔ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_rational_solution_quadratic_l1524_152416


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1524_152462

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ (x y : ℤ), f (x + y) = f x + f y + 2) :
  ∃ (a : ℤ), ∀ (x : ℤ), f x = a * x - 2 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1524_152462


namespace NUMINAMATH_CALUDE_tangent_line_m_value_l1524_152492

/-- The curve function f(x) = x^3 + x - 1 -/
def f (x : ℝ) : ℝ := x^3 + x - 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_line_m_value :
  let p₁ : ℝ × ℝ := (1, f 1)
  let slope : ℝ := f' 1
  let p₂ : ℝ × ℝ := (2, m)
  (∀ m : ℝ, (m - p₁.2) = slope * (p₂.1 - p₁.1) → m = 5) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_m_value_l1524_152492


namespace NUMINAMATH_CALUDE_row_6_seat_16_notation_l1524_152483

/-- Represents a seat in a movie theater -/
structure Seat :=
  (row : ℕ)
  (number : ℕ)

/-- The format for denoting a seat -/
def seatNotation (s : Seat) : ℕ × ℕ := (s.row, s.number)

/-- Given condition: "row 10, seat 3" is denoted as (10,3) -/
axiom example_seat : seatNotation { row := 10, number := 3 } = (10, 3)

/-- Theorem: "row 6, seat 16" is denoted as (6,16) -/
theorem row_6_seat_16_notation :
  seatNotation { row := 6, number := 16 } = (6, 16) := by
  sorry


end NUMINAMATH_CALUDE_row_6_seat_16_notation_l1524_152483


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1524_152460

theorem sufficient_not_necessary_condition : 
  (∀ x : ℝ, x ≥ 3 → (x - 2) ≥ 0) ∧ 
  (∃ x : ℝ, (x - 2) ≥ 0 ∧ ¬(x ≥ 3)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1524_152460


namespace NUMINAMATH_CALUDE_point_trajectory_l1524_152419

/-- The trajectory of a point M(x,y) satisfying a specific distance condition -/
theorem point_trajectory (x y : ℝ) (h : Real.sqrt ((x + 5)^2 + y^2) - Real.sqrt ((x - 5)^2 + y^2) = 8) (hx : x > 0) :
  x^2 / 16 - y^2 / 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_trajectory_l1524_152419


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l1524_152468

/-- Given a rectangular box Q inscribed in a sphere of radius r,
    if the surface area of Q is 672 and the sum of the lengths of its 12 edges is 168,
    then r = √273 -/
theorem inscribed_box_radius (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  2 * (a * b + b * c + a * c) = 672 →
  4 * (a + b + c) = 168 →
  (2 * r) ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 →
  r = Real.sqrt 273 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l1524_152468


namespace NUMINAMATH_CALUDE_average_of_seventeen_numbers_l1524_152491

-- Define the problem parameters
def total_count : ℕ := 17
def first_nine_avg : ℚ := 56
def last_nine_avg : ℚ := 63
def ninth_number : ℚ := 68

-- Theorem statement
theorem average_of_seventeen_numbers :
  let first_nine_sum := 9 * first_nine_avg
  let last_nine_sum := 9 * last_nine_avg
  let total_sum := first_nine_sum + last_nine_sum - ninth_number
  total_sum / total_count = 59 := by
sorry

end NUMINAMATH_CALUDE_average_of_seventeen_numbers_l1524_152491


namespace NUMINAMATH_CALUDE_compare_expressions_l1524_152478

theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (a^2 + b^2) ≤ 2 * (a^3 + b^3) := by
  sorry

end NUMINAMATH_CALUDE_compare_expressions_l1524_152478


namespace NUMINAMATH_CALUDE_square_of_negative_product_l1524_152477

theorem square_of_negative_product (a b : ℝ) : (-3 * a^2 * b)^2 = 9 * a^4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l1524_152477


namespace NUMINAMATH_CALUDE_log_equation_solution_l1524_152451

theorem log_equation_solution (y : ℝ) (h : y > 0) : 
  Real.log y^3 / Real.log 3 + Real.log y / Real.log (1/3) = 6 → y = 27 := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1524_152451


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_100111011_base6_l1524_152493

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- Finds all divisors of a number -/
def divisors (n : ℕ) : List ℕ := sorry

theorem largest_prime_divisor_of_100111011_base6 :
  let n := base6ToBase10 100111011
  ∃ (d : ℕ), d ∈ divisors n ∧ isPrime d ∧ d = 181 ∧ ∀ (p : ℕ), p ∈ divisors n → isPrime p → p ≤ d :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_100111011_base6_l1524_152493


namespace NUMINAMATH_CALUDE_water_bottles_taken_out_l1524_152402

theorem water_bottles_taken_out (red : ℕ) (black : ℕ) (blue : ℕ) (remaining : ℕ) :
  red = 2 → black = 3 → blue = 4 → remaining = 4 →
  red + black + blue - remaining = 5 :=
by sorry

end NUMINAMATH_CALUDE_water_bottles_taken_out_l1524_152402


namespace NUMINAMATH_CALUDE_sum_in_base7_l1524_152410

/-- Converts a list of digits in base 7 to a natural number -/
def fromBase7 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a natural number to a list of digits in base 7 -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem sum_in_base7 :
  fromBase7 [2, 4, 5] + fromBase7 [5, 4, 3] = fromBase7 [1, 1, 2, 1] :=
by sorry

end NUMINAMATH_CALUDE_sum_in_base7_l1524_152410


namespace NUMINAMATH_CALUDE_curve_is_ellipse_with_foci_on_y_axis_l1524_152430

-- Define the angle α in radians
def α : ℝ := sorry

-- Define the conditions on α
axiom α_pos : 0 < α
axiom α_less_than_pi_half : α < Real.pi / 2

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  x^2 + y^2 * (Real.cos α) = 1

-- Define what it means for a curve to be an ellipse with foci on the y-axis
def is_ellipse_with_foci_on_y_axis (eq : (ℝ → ℝ → Prop)) : Prop := sorry

-- State the theorem
theorem curve_is_ellipse_with_foci_on_y_axis :
  is_ellipse_with_foci_on_y_axis curve_equation := by sorry

end NUMINAMATH_CALUDE_curve_is_ellipse_with_foci_on_y_axis_l1524_152430


namespace NUMINAMATH_CALUDE_larger_number_proof_l1524_152434

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 59) 
  (h2 : Nat.lcm a b = 12272) (h3 : 13 ∣ Nat.lcm a b) (h4 : 16 ∣ Nat.lcm a b) :
  max a b = 944 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1524_152434


namespace NUMINAMATH_CALUDE_newspapers_julie_can_print_l1524_152444

-- Define the given conditions
def boxes : ℕ := 2
def packages_per_box : ℕ := 5
def sheets_per_package : ℕ := 250
def sheets_per_newspaper : ℕ := 25

-- Define the theorem
theorem newspapers_julie_can_print :
  (boxes * packages_per_box * sheets_per_package) / sheets_per_newspaper = 100 := by
  sorry

end NUMINAMATH_CALUDE_newspapers_julie_can_print_l1524_152444


namespace NUMINAMATH_CALUDE_less_than_preserved_subtraction_l1524_152454

theorem less_than_preserved_subtraction (a b : ℝ) : a < b → a - 1 < b - 1 := by
  sorry

end NUMINAMATH_CALUDE_less_than_preserved_subtraction_l1524_152454


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l1524_152471

theorem no_solution_to_equation :
  ¬∃ s : ℝ, (s^2 - 6*s + 8) / (s^2 - 9*s + 20) = (s^2 - 3*s - 18) / (s^2 - 2*s - 15) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l1524_152471


namespace NUMINAMATH_CALUDE_fish_difference_l1524_152401

theorem fish_difference (goldfish : ℕ) (angelfish : ℕ) (guppies : ℕ) : 
  goldfish = 8 →
  guppies = 2 * angelfish →
  goldfish + angelfish + guppies = 44 →
  angelfish - goldfish = 4 := by
sorry

end NUMINAMATH_CALUDE_fish_difference_l1524_152401


namespace NUMINAMATH_CALUDE_coefficient_x4_l1524_152487

/-- The coefficient of x^4 in the simplified form of 5(x^4 - 3x^2) + 3(2x^3 - x^4 + 4x^6) - (6x^2 - 2x^4) is 4 -/
theorem coefficient_x4 (x : ℝ) : 
  let expr := 5*(x^4 - 3*x^2) + 3*(2*x^3 - x^4 + 4*x^6) - (6*x^2 - 2*x^4)
  ∃ (a b c d e : ℝ), expr = 4*x^4 + a*x^6 + b*x^3 + c*x^2 + d*x + e :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x4_l1524_152487


namespace NUMINAMATH_CALUDE_logan_hair_length_l1524_152490

/-- Given information about hair lengths of Kate, Emily, and Logan, prove Logan's hair length. -/
theorem logan_hair_length (kate_length emily_length logan_length : ℝ) 
  (h1 : kate_length = 7)
  (h2 : kate_length = emily_length / 2)
  (h3 : emily_length = logan_length + 6) :
  logan_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_logan_hair_length_l1524_152490


namespace NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1524_152446

theorem sin_product_equals_one_sixteenth :
  Real.sin (12 * π / 180) * Real.sin (36 * π / 180) * Real.sin (72 * π / 180) * Real.sin (84 * π / 180) = 1 / 16 :=
by sorry

end NUMINAMATH_CALUDE_sin_product_equals_one_sixteenth_l1524_152446


namespace NUMINAMATH_CALUDE_f_min_at_x_min_l1524_152420

/-- The quadratic function f(x) = x^2 - 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 36

/-- The point where the minimum of f occurs -/
def x_min : ℝ := 6

theorem f_min_at_x_min :
  ∀ x : ℝ, f x ≥ f x_min :=
sorry

end NUMINAMATH_CALUDE_f_min_at_x_min_l1524_152420


namespace NUMINAMATH_CALUDE_omitted_angle_measure_l1524_152411

theorem omitted_angle_measure (n : ℕ) (sum_without_one : ℝ) : 
  n ≥ 3 → 
  sum_without_one = 1958 → 
  (n - 2) * 180 - sum_without_one = 22 :=
by sorry

end NUMINAMATH_CALUDE_omitted_angle_measure_l1524_152411


namespace NUMINAMATH_CALUDE_parabola_directrix_distance_l1524_152459

/-- For a parabola y = mx^2, if the distance from the origin to the directrix is 2, then m = ± 1/8 -/
theorem parabola_directrix_distance (m : ℝ) : 
  (∀ x y : ℝ, y = m * x^2) →  -- Condition 1: Parabola equation
  (∃ d : ℝ, d = -1 / (4 * m) ∧ |d| = 2) →  -- Condition 2: Distance to directrix
  m = 1/8 ∨ m = -1/8 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_distance_l1524_152459


namespace NUMINAMATH_CALUDE_minimum_value_of_function_l1524_152404

theorem minimum_value_of_function (x : ℝ) (h : x > 0) : 
  (∀ y : ℝ, y > 0 → 4 / y + y ≥ 4) ∧ (∃ z : ℝ, z > 0 ∧ 4 / z + z = 4) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_function_l1524_152404


namespace NUMINAMATH_CALUDE_saleems_baskets_l1524_152473

theorem saleems_baskets (initial_avg_cost : ℝ) (additional_basket_cost : ℝ) (new_avg_cost : ℝ) :
  initial_avg_cost = 4 →
  additional_basket_cost = 8 →
  new_avg_cost = 4.8 →
  ∃ x : ℕ, x > 0 ∧ 
    (x * initial_avg_cost + additional_basket_cost) / (x + 1 : ℝ) = new_avg_cost ∧
    x = 4 :=
by sorry

end NUMINAMATH_CALUDE_saleems_baskets_l1524_152473


namespace NUMINAMATH_CALUDE_sports_field_dimensions_l1524_152400

/-- The dimensions of a rectangular sports field with a surrounding path -/
theorem sports_field_dimensions (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ x : ℝ,
    x > 0 ∧
    x * (x + b) = (x + 2*a) * (x + b + 2*a) - x * (x + b) ∧
    x = (Real.sqrt (b^2 + 32*a^2) - b + 4*a) / 2 ∧
    x + b = (Real.sqrt (b^2 + 32*a^2) + b + 4*a) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sports_field_dimensions_l1524_152400


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1524_152461

/-- Represents the repeating decimal 0.3̄6 -/
def repeating_decimal : ℚ := 0.3666666666666667

/-- The fraction we want to prove equal to the repeating decimal -/
def target_fraction : ℚ := 11 / 30

/-- Theorem stating that the repeating decimal 0.3̄6 is equal to 11/30 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l1524_152461


namespace NUMINAMATH_CALUDE_vector_dot_product_result_l1524_152418

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

theorem vector_dot_product_result :
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_result_l1524_152418


namespace NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_max_coefficient_terms_sqrt_inequality_l1524_152474

-- Part 1
theorem binomial_expansion_arithmetic_sequence (n : ℕ) :
  (∃ r : ℚ, r ≠ 0 ∧ 
    n.choose 0 + (1/4) * n.choose 2 = 2 * (1/2) * n.choose 1) →
  n = 8 :=
sorry

-- Part 2
theorem max_coefficient_terms (n : ℕ) (x : ℝ) :
  n = 8 →
  ∃ c : ℝ, c > 0 ∧
    (∀ k : ℕ, k ≤ n → 
      c * x^(5 : ℝ) ≥ (1/(2^k : ℝ)) * n.choose k * x^((n - k : ℝ)/2)) ∧
    (∀ k : ℕ, k ≤ n → 
      c * x^(7/2 : ℝ) ≥ (1/(2^k : ℝ)) * n.choose k * x^((n - k : ℝ)/2)) :=
sorry

-- Part 3
theorem sqrt_inequality (a : ℝ) :
  a > 1 →
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt a - Real.sqrt (a - 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_expansion_arithmetic_sequence_max_coefficient_terms_sqrt_inequality_l1524_152474


namespace NUMINAMATH_CALUDE_circle_properties_l1524_152484

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 9
def C₂ (x y : ℝ) : Prop := (x-1)^2 + (y+1)^2 = 16

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

theorem circle_properties :
  -- 1. The equation of the line passing through the centers of C₁ and C₂ is y = -x
  (∃ m b : ℝ, ∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = -1) → y = m * x + b) ∧
  (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = -1) → y = -x) ∧
  -- 2. The circles intersect and the length of their common chord is √94/2
  (∃ x₁ y₁ x₂ y₂ : ℝ, C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
    ((x₂ - x₁)^2 + (y₂ - y₁)^2)^(1/2) = (94)^(1/2)/2) ∧
  -- 3. There exist exactly 4 points on C₂ that are at a distance of 2 from the line y = x
  (∃! (a b c d : ℝ × ℝ), 
    C₂ a.1 a.2 ∧ C₂ b.1 b.2 ∧ C₂ c.1 c.2 ∧ C₂ d.1 d.2 ∧
    (∀ x y : ℝ, line_y_eq_x x y → 
      ((a.1 - x)^2 + (a.2 - y)^2)^(1/2) = 2 ∧
      ((b.1 - x)^2 + (b.2 - y)^2)^(1/2) = 2 ∧
      ((c.1 - x)^2 + (c.2 - y)^2)^(1/2) = 2 ∧
      ((d.1 - x)^2 + (d.2 - y)^2)^(1/2) = 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l1524_152484


namespace NUMINAMATH_CALUDE_original_number_is_nine_l1524_152442

theorem original_number_is_nine (x : ℝ) : (x - 5) / 4 = (x - 4) / 5 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_nine_l1524_152442


namespace NUMINAMATH_CALUDE_line_proof_circle_proof_l1524_152476

-- Define the line
def line_equation (x y : ℝ) : Prop := y = x + 2

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x + 2)^2 + (y - 3)^2 = 4

-- Theorem for the line
theorem line_proof (x y : ℝ) (h1 : line_equation x y) :
  y = x + 2 := by sorry

-- Theorem for the circle
theorem circle_proof (x y : ℝ) (h2 : circle_equation x y) :
  (x + 2)^2 + (y - 3)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_line_proof_circle_proof_l1524_152476


namespace NUMINAMATH_CALUDE_right_triangle_leg_construction_l1524_152405

theorem right_triangle_leg_construction (c m : ℝ) (h_positive : c > 0) :
  ∃ (a b p : ℝ),
    a > 0 ∧ b > 0 ∧ p > 0 ∧
    a^2 + b^2 = c^2 ∧
    a^2 - b^2 = 4 * m^2 ∧
    p = (c * (1 + Real.sqrt 5)) / 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_leg_construction_l1524_152405


namespace NUMINAMATH_CALUDE_projection_vector_proof_l1524_152431

/-- Given two vectors projected onto the same vector, prove the resulting projection vector. -/
theorem projection_vector_proof (v : ℝ × ℝ) : 
  let a : ℝ × ℝ := (-6, 2)
  let b : ℝ × ℝ := (3, 4)
  let p : ℝ × ℝ := (-12/17, 54/17)
  (∃ (t : ℝ), p = a + t • (b - a)) ∧ 
  (p.1 * (b.1 - a.1) + p.2 * (b.2 - a.2) = 0) := by
sorry


end NUMINAMATH_CALUDE_projection_vector_proof_l1524_152431


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1524_152437

theorem z_in_first_quadrant : 
  ∀ z : ℂ, z / (1 + Complex.I) = 2 - Complex.I → 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = 1 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1524_152437


namespace NUMINAMATH_CALUDE_cameron_fruit_arrangements_l1524_152480

/-- The number of ways to arrange n objects, where there are k groups of indistinguishable objects with sizes a₁, a₂, ..., aₖ -/
def multinomial (n : ℕ) (a : List ℕ) : ℕ :=
  Nat.factorial n / (a.map Nat.factorial).prod

/-- The number of ways Cameron can eat his fruit -/
def cameronFruitArrangements : ℕ :=
  multinomial 9 [4, 3, 2]

theorem cameron_fruit_arrangements :
  cameronFruitArrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_cameron_fruit_arrangements_l1524_152480


namespace NUMINAMATH_CALUDE_pencil_distribution_l1524_152409

theorem pencil_distribution (total_pencils : ℕ) (pencils_per_student : ℕ) (h1 : total_pencils = 42) (h2 : pencils_per_student = 3) :
  total_pencils / pencils_per_student = 14 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1524_152409


namespace NUMINAMATH_CALUDE_train_departure_sequences_l1524_152433

theorem train_departure_sequences :
  let total_trains : ℕ := 6
  let trains_per_group : ℕ := 3
  let num_special_trains : ℕ := 2  -- G1 and G2
  let num_regular_trains : ℕ := total_trains - num_special_trains

  -- Number of ways to choose trains for G1's group (excluding G1 itself)
  let group_formations : ℕ := Nat.choose num_regular_trains (trains_per_group - 1)

  -- Number of permutations for each group
  let group_permutations : ℕ := Nat.factorial trains_per_group

  -- Total number of departure sequences
  group_formations * group_permutations * group_permutations = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_train_departure_sequences_l1524_152433


namespace NUMINAMATH_CALUDE_selling_price_loss_percentage_l1524_152450

theorem selling_price_loss_percentage (cost_price : ℝ) 
  (h : cost_price > 0) : 
  let selling_price_100 := 40 * cost_price
  let cost_price_100 := 100 * cost_price
  (selling_price_100 / cost_price_100) * 100 = 40 → 
  ((cost_price_100 - selling_price_100) / cost_price_100) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_selling_price_loss_percentage_l1524_152450


namespace NUMINAMATH_CALUDE_initial_fish_caught_per_day_l1524_152443

-- Define the initial colony size
def initial_colony_size : ℕ := sorry

-- Define the colony size after the first year (doubled)
def first_year_size : ℕ := 2 * initial_colony_size

-- Define the colony size after the second year (tripled from first year)
def second_year_size : ℕ := 3 * first_year_size

-- Define the current colony size (after third year)
def current_colony_size : ℕ := 1077

-- Define the increase in the third year
def third_year_increase : ℕ := 129

-- Define the fish consumption per penguin per day
def fish_per_penguin : ℚ := 3/2

-- Theorem stating the initial number of fish caught per day
theorem initial_fish_caught_per_day :
  (initial_colony_size : ℚ) * fish_per_penguin = 237 :=
by sorry

end NUMINAMATH_CALUDE_initial_fish_caught_per_day_l1524_152443


namespace NUMINAMATH_CALUDE_cone_volume_approximation_l1524_152453

theorem cone_volume_approximation (L h : ℝ) (h1 : L > 0) (h2 : h > 0) :
  (1 / 75 : ℝ) * L^2 * h = (1 / 3 : ℝ) * ((25 / 4) / 4) * L^2 * h := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_approximation_l1524_152453


namespace NUMINAMATH_CALUDE_sufficient_implies_necessary_l1524_152413

theorem sufficient_implies_necessary (A B : Prop) :
  (A → B) → (¬B → ¬A) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_implies_necessary_l1524_152413


namespace NUMINAMATH_CALUDE_grid_recoloring_theorem_l1524_152427

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents the grid -/
def Grid := Fin 99 → Fin 99 → Color

/-- Represents a row or column index -/
def Index := Fin 99

/-- Represents a recoloring operation -/
inductive RecolorOp
| Row (i : Index)
| Col (j : Index)

/-- Applies a recoloring operation to a grid -/
def applyRecolor (g : Grid) (op : RecolorOp) : Grid :=
  sorry

/-- Checks if all cells in the grid have the same color -/
def isMonochromatic (g : Grid) : Prop :=
  sorry

/-- The main theorem -/
theorem grid_recoloring_theorem (g : Grid) :
  ∃ (ops : List RecolorOp), isMonochromatic (ops.foldl applyRecolor g) :=
sorry

end NUMINAMATH_CALUDE_grid_recoloring_theorem_l1524_152427


namespace NUMINAMATH_CALUDE_dividend_calculation_l1524_152496

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 19)
  (h2 : quotient = 7)
  (h3 : remainder = 6) :
  divisor * quotient + remainder = 139 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1524_152496


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_product_l1524_152482

theorem quadratic_equation_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 8 ∧ x * y = 9) →
  m + n = 51 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_product_l1524_152482


namespace NUMINAMATH_CALUDE_curve_transformation_l1524_152432

/-- Given a scaling transformation and the equation of the transformed curve,
    prove the equation of the original curve. -/
theorem curve_transformation (x y x' y' : ℝ) :
  (x' = 5 * x) →
  (y' = 3 * y) →
  (x'^2 + 4 * y'^2 = 1) →
  (25 * x^2 + 36 * y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l1524_152432


namespace NUMINAMATH_CALUDE_m_le_n_l1524_152457

theorem m_le_n (a b : ℝ) : 
  let m := (6^a) / (36^(a+1) + 1)
  let n := (1/3) * b^2 - b + 5/6
  m ≤ n := by sorry

end NUMINAMATH_CALUDE_m_le_n_l1524_152457


namespace NUMINAMATH_CALUDE_parallelogram_area_32_15_l1524_152422

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 32 cm and height 15 cm is 480 square centimeters -/
theorem parallelogram_area_32_15 :
  parallelogram_area 32 15 = 480 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_32_15_l1524_152422


namespace NUMINAMATH_CALUDE_count_four_digit_with_three_is_1000_l1524_152408

/-- The count of four-digit positive integers with the thousands digit 3 -/
def count_four_digit_with_three : ℕ :=
  (List.range 10).length * (List.range 10).length * (List.range 10).length

/-- Theorem: The count of four-digit positive integers with the thousands digit 3 is 1000 -/
theorem count_four_digit_with_three_is_1000 :
  count_four_digit_with_three = 1000 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_with_three_is_1000_l1524_152408


namespace NUMINAMATH_CALUDE_tangent_line_inclination_range_l1524_152464

open Real

theorem tangent_line_inclination_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 4) :
  let f := fun x => Real.log x + x / b
  let α := Real.arctan (((1 / a) + (1 / b)) : ℝ)
  π / 4 ≤ α ∧ α < π / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_range_l1524_152464


namespace NUMINAMATH_CALUDE_circle_properties_l1524_152449

noncomputable def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 1

theorem circle_properties :
  ∃ (c : ℝ × ℝ),
    (c.1 = c.2) ∧  -- Center is on the line y = x
    (∀ x y : ℝ, circle_equation x y → (x - c.1)^2 + (y - c.2)^2 = 1) ∧  -- Equation represents a circle
    (circle_equation 1 0) ∧  -- Circle passes through (1,0)
    (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → ¬(circle_equation x 0)) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1524_152449


namespace NUMINAMATH_CALUDE_part_one_part_two_part_two_converse_l1524_152407

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 3 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x < 1 ∨ x > 6}

-- Part 1
theorem part_one : A 3 ∩ (Set.univ \ B) = {x | 1 ≤ x ∧ x ≤ 5} := by sorry

-- Part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : A a ∩ B = ∅) :
  a ∈ {x | 0 < x ∧ x ≤ 2} := by sorry

theorem part_two_converse (a : ℝ) (h : a ∈ {x | 0 < x ∧ x ≤ 2}) :
  a > 0 ∧ A a ∩ B = ∅ := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_two_converse_l1524_152407


namespace NUMINAMATH_CALUDE_oil_price_relation_l1524_152465

/-- Represents a right circular cylinder -/
structure Cylinder where
  height : ℝ
  radius : ℝ

/-- The price of oil in a cylinder when filled to a certain capacity -/
def oilPrice (c : Cylinder) (fillRatio : ℝ) : ℝ := sorry

theorem oil_price_relation (x y : Cylinder) :
  y.height = 4 * x.height →
  y.radius = 4 * x.radius →
  oilPrice y 0.5 = 64 →
  oilPrice x 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_relation_l1524_152465


namespace NUMINAMATH_CALUDE_total_sharks_count_l1524_152497

/-- The number of sharks at Newport Beach -/
def newport_sharks : ℕ := 22

/-- The number of sharks at Dana Point beach -/
def dana_point_sharks : ℕ := 4 * newport_sharks

/-- The total number of sharks on both beaches -/
def total_sharks : ℕ := newport_sharks + dana_point_sharks

theorem total_sharks_count : total_sharks = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_sharks_count_l1524_152497


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1524_152429

theorem partial_fraction_decomposition (x : ℝ) (h : x ≠ 0) :
  (-2 * x^2 + 5 * x - 6) / (x^3 + 2 * x) = 
  (-3 : ℝ) / x + (x + 5) / (x^2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1524_152429


namespace NUMINAMATH_CALUDE_earl_went_up_seven_floors_l1524_152489

/-- Represents the number of floors in the building -/
def total_floors : ℕ := 20

/-- Represents Earl's initial floor -/
def initial_floor : ℕ := 1

/-- Represents the number of floors Earl goes up initially -/
def first_up : ℕ := 5

/-- Represents the number of floors Earl goes down -/
def down : ℕ := 2

/-- Represents the number of floors Earl is away from the top after his final movement -/
def floors_from_top : ℕ := 9

/-- Calculates the number of floors Earl went up the second time -/
def second_up : ℕ := total_floors - floors_from_top - (initial_floor + first_up - down)

/-- Theorem stating that Earl went up 7 floors the second time -/
theorem earl_went_up_seven_floors : second_up = 7 := by sorry

end NUMINAMATH_CALUDE_earl_went_up_seven_floors_l1524_152489


namespace NUMINAMATH_CALUDE_proportional_function_quadrants_l1524_152469

/-- A proportional function passing through the second and fourth quadrants has a negative coefficient. -/
theorem proportional_function_quadrants (k : ℝ) :
  (∀ x y : ℝ, y = k * x →
    ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) →
  k < 0 := by
  sorry

end NUMINAMATH_CALUDE_proportional_function_quadrants_l1524_152469


namespace NUMINAMATH_CALUDE_meadow_trees_l1524_152455

/-- Represents the number of trees around the meadow. -/
def num_trees : ℕ := sorry

/-- Represents Serezha's count of a specific tree. -/
def serezha_count1 : ℕ := 20

/-- Represents Misha's count of the same tree as serezha_count1. -/
def misha_count1 : ℕ := 7

/-- Represents Serezha's count of another specific tree. -/
def serezha_count2 : ℕ := 7

/-- Represents Misha's count of the same tree as serezha_count2. -/
def misha_count2 : ℕ := 94

/-- The theorem stating that the number of trees around the meadow is 100. -/
theorem meadow_trees : num_trees = 100 := by sorry

end NUMINAMATH_CALUDE_meadow_trees_l1524_152455


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1524_152479

theorem gcd_of_three_numbers : Nat.gcd 8247 (Nat.gcd 13619 29826) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1524_152479


namespace NUMINAMATH_CALUDE_rotated_solid_properties_l1524_152421

/-- A right-angled triangle with sides 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angled : a^2 + b^2 = c^2
  side_a : a = 3
  side_b : b = 4
  side_c : c = 5

/-- The solid formed by rotating the triangle around its hypotenuse -/
def RotatedSolid (t : RightTriangle) : Prop :=
  ∃ (surface_area volume : ℝ),
    surface_area = 84/5 * Real.pi ∧
    volume = 48/5 * Real.pi

/-- Theorem stating the surface area and volume of the rotated solid -/
theorem rotated_solid_properties (t : RightTriangle) :
  RotatedSolid t := by sorry

end NUMINAMATH_CALUDE_rotated_solid_properties_l1524_152421


namespace NUMINAMATH_CALUDE_max_min_diff_abs_sum_ratio_l1524_152456

/-- The difference between the maximum and minimum values of |a + b| / (|a| + |b|) for nonzero real numbers a and b is 1. -/
theorem max_min_diff_abs_sum_ratio : ∃ (m' M' : ℝ),
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → m' ≤ |a + b| / (|a| + |b|) ∧ |a + b| / (|a| + |b|) ≤ M') ∧
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ |a + b| / (|a| + |b|) = m') ∧
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ |a + b| / (|a| + |b|) = M') ∧
  M' - m' = 1 := by
sorry

end NUMINAMATH_CALUDE_max_min_diff_abs_sum_ratio_l1524_152456


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1524_152495

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^24 + 8^15) ∧ ∀ q, Nat.Prime q → q ∣ (3^24 + 8^15) → p ≤ q := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1524_152495


namespace NUMINAMATH_CALUDE_willow_football_time_l1524_152440

/-- Proves that Willow played football for 60 minutes given the conditions -/
theorem willow_football_time :
  ∀ (total_time basketball_time football_time : ℕ),
  total_time = 120 →
  basketball_time = 60 →
  total_time = basketball_time + football_time →
  football_time = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_willow_football_time_l1524_152440


namespace NUMINAMATH_CALUDE_prop_2_correct_prop_4_correct_prop_1_not_necessarily_true_prop_3_not_necessarily_true_l1524_152423

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Proposition 2
theorem prop_2_correct 
  (a b : Line) (α : Plane) 
  (h1 : parallel_line_plane a α) 
  (h2 : perpendicular_line_plane b α) : 
  perpendicular_lines a b :=
sorry

-- Proposition 4
theorem prop_4_correct 
  (a : Line) (α β : Plane) 
  (h1 : perpendicular_line_plane a α) 
  (h2 : parallel_line_plane a β) : 
  perpendicular_planes α β :=
sorry

-- Proposition 1 is not necessarily true
theorem prop_1_not_necessarily_true :
  ¬ ∀ (a b : Line) (α β : Plane),
    parallel_line_plane a α → parallel_line_plane b β → parallel_lines a b :=
sorry

-- Proposition 3 is not necessarily true
theorem prop_3_not_necessarily_true :
  ¬ ∀ (a b : Line) (α : Plane),
    parallel_lines a b → parallel_line_plane b α → parallel_line_plane a α :=
sorry

end NUMINAMATH_CALUDE_prop_2_correct_prop_4_correct_prop_1_not_necessarily_true_prop_3_not_necessarily_true_l1524_152423


namespace NUMINAMATH_CALUDE_common_internal_tangent_length_l1524_152439

theorem common_internal_tangent_length
  (center_distance : ℝ)
  (small_radius : ℝ)
  (large_radius : ℝ)
  (h1 : center_distance = 41)
  (h2 : small_radius = 4)
  (h3 : large_radius = 5) :
  Real.sqrt (center_distance^2 - (small_radius + large_radius)^2) = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_common_internal_tangent_length_l1524_152439


namespace NUMINAMATH_CALUDE_expression_value_l1524_152428

theorem expression_value (a : ℝ) (h : a = -3) : 
  (3 * a⁻¹ + a⁻¹ / 3) / a = 10 / 27 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1524_152428


namespace NUMINAMATH_CALUDE_oreo_distribution_l1524_152445

/-- The number of Oreos Jordan has -/
def jordans_oreos : ℕ := 11

/-- The number of Oreos James has -/
def james_oreos (j : ℕ) : ℕ := 2 * j + 3

/-- The total number of Oreos -/
def total_oreos : ℕ := 36

theorem oreo_distribution : 
  james_oreos jordans_oreos + jordans_oreos = total_oreos :=
by sorry

end NUMINAMATH_CALUDE_oreo_distribution_l1524_152445


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l1524_152470

theorem max_value_theorem (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 29 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 1) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l1524_152470


namespace NUMINAMATH_CALUDE_cubic_factorization_l1524_152412

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1524_152412


namespace NUMINAMATH_CALUDE_parallelogram_area_l1524_152424

/-- Given two 2D vectors u and z, this theorem proves that the area of the parallelogram
    formed by u and z + u is 3. -/
theorem parallelogram_area (u z : Fin 2 → ℝ) (hu : u = ![4, -1]) (hz : z = ![9, -3]) :
  let z' := z + u
  abs (u 0 * z' 1 - u 1 * z' 0) = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l1524_152424


namespace NUMINAMATH_CALUDE_sum_of_squares_l1524_152447

theorem sum_of_squares (x y z a b c k : ℝ) 
  (h1 : x * y = k * a)
  (h2 : x * z = b)
  (h3 : y * z = c)
  (h4 : k ≠ 0)
  (h5 : x ≠ 0)
  (h6 : y ≠ 0)
  (h7 : z ≠ 0)
  (h8 : a ≠ 0)
  (h9 : b ≠ 0)
  (h10 : c ≠ 0) :
  x^2 + y^2 + z^2 = (k * (a * b + a * c + b * c)) / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1524_152447


namespace NUMINAMATH_CALUDE_equilateral_triangle_to_trapezoid_l1524_152485

/-- Represents a paper shape that can be folded -/
structure PaperShape where
  vertices : ℕ
  layers : ℕ

/-- Represents the process of folding a paper shape -/
def fold (initial : PaperShape) (final : PaperShape) : Prop :=
  ∃ (steps : ℕ), steps > 0 ∧ final.layers ≥ initial.layers

/-- An equilateral triangle -/
def equilateralTriangle : PaperShape :=
  { vertices := 3, layers := 1 }

/-- A trapezoid -/
def trapezoid : PaperShape :=
  { vertices := 4, layers := 3 }

theorem equilateral_triangle_to_trapezoid :
  fold equilateralTriangle trapezoid :=
sorry

#check equilateral_triangle_to_trapezoid

end NUMINAMATH_CALUDE_equilateral_triangle_to_trapezoid_l1524_152485


namespace NUMINAMATH_CALUDE_oates_reunion_attendees_l1524_152435

/-- The number of people attending the Oates reunion -/
def oates_attendees : ℕ := 50

/-- The number of people attending the Hall reunion -/
def hall_attendees : ℕ := 62

/-- The number of people attending both reunions -/
def both_attendees : ℕ := 12

/-- The total number of guests at the hotel -/
def total_guests : ℕ := 100

theorem oates_reunion_attendees :
  oates_attendees + hall_attendees - both_attendees = total_guests :=
by sorry

end NUMINAMATH_CALUDE_oates_reunion_attendees_l1524_152435


namespace NUMINAMATH_CALUDE_burger_nonfiller_percentage_l1524_152426

/-- Given a burger with a total weight and filler weight, calculate the percentage that is not filler -/
theorem burger_nonfiller_percentage
  (total_weight : ℝ)
  (filler_weight : ℝ)
  (h1 : total_weight = 180)
  (h2 : filler_weight = 45)
  : (total_weight - filler_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_burger_nonfiller_percentage_l1524_152426


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1524_152499

-- Define the ellipse C
def ellipse_C (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola E
def parabola_E (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the foci
def F₁ : ℝ × ℝ := (-1, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define point P
structure Point_P where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_C a b x y
  on_parabola : parabola_E x y
  first_quadrant : x > 0 ∧ y > 0

-- Define the tangent line PF₁
def tangent_line (P : Point_P) (x y : ℝ) : Prop :=
  y = (P.y / (P.x + 1)) * (x + 1)

theorem ellipse_major_axis_length
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (P : Point_P)
  (h3 : tangent_line P P.x P.y) :
  2 * a = 2 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1524_152499


namespace NUMINAMATH_CALUDE_line_t_value_l1524_152475

/-- A line passing through points (2, 8), (6, 20), (10, 32), and (35, t) -/
structure Line where
  -- Define the slope of the line
  slope : ℝ
  -- Define the y-intercept of the line
  y_intercept : ℝ
  -- Ensure the line passes through (2, 8)
  point1_condition : 8 = slope * 2 + y_intercept
  -- Ensure the line passes through (6, 20)
  point2_condition : 20 = slope * 6 + y_intercept
  -- Ensure the line passes through (10, 32)
  point3_condition : 32 = slope * 10 + y_intercept

/-- The t-value for the point (35, t) on the line -/
def t_value (l : Line) : ℝ := l.slope * 35 + l.y_intercept

theorem line_t_value : ∀ l : Line, t_value l = 107 := by sorry

end NUMINAMATH_CALUDE_line_t_value_l1524_152475


namespace NUMINAMATH_CALUDE_cos_shift_equivalence_l1524_152415

open Real

theorem cos_shift_equivalence (x : ℝ) :
  cos (2 * (x + π / 6) - π / 3) = cos (2 * x) :=
by sorry

end NUMINAMATH_CALUDE_cos_shift_equivalence_l1524_152415


namespace NUMINAMATH_CALUDE_fraction_who_say_dislike_but_like_l1524_152417

/-- Represents the student population at Gateway Academy -/
structure StudentPopulation where
  total : ℝ
  like_skating : ℝ
  dislike_skating : ℝ
  say_like_actually_like : ℝ
  say_dislike_actually_like : ℝ
  say_like_actually_dislike : ℝ
  say_dislike_actually_dislike : ℝ

/-- The conditions of the problem -/
def gateway_academy (pop : StudentPopulation) : Prop :=
  pop.total > 0 ∧
  pop.like_skating = 0.4 * pop.total ∧
  pop.dislike_skating = 0.6 * pop.total ∧
  pop.say_like_actually_like = 0.7 * pop.like_skating ∧
  pop.say_dislike_actually_like = 0.3 * pop.like_skating ∧
  pop.say_like_actually_dislike = 0.2 * pop.dislike_skating ∧
  pop.say_dislike_actually_dislike = 0.8 * pop.dislike_skating

/-- The theorem to be proved -/
theorem fraction_who_say_dislike_but_like (pop : StudentPopulation) 
  (h : gateway_academy pop) : 
  pop.say_dislike_actually_like / (pop.say_dislike_actually_like + pop.say_dislike_actually_dislike) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_who_say_dislike_but_like_l1524_152417


namespace NUMINAMATH_CALUDE_wasted_meat_pounds_l1524_152448

def minimum_wage : ℝ := 8
def fruit_veg_cost_per_pound : ℝ := 4
def fruit_veg_wasted : ℝ := 15
def bread_cost_per_pound : ℝ := 1.5
def bread_wasted : ℝ := 60
def janitor_normal_wage : ℝ := 10
def janitor_hours : ℝ := 10
def meat_cost_per_pound : ℝ := 5
def james_work_hours : ℝ := 50

def total_cost : ℝ := james_work_hours * minimum_wage

def fruit_veg_cost : ℝ := fruit_veg_cost_per_pound * fruit_veg_wasted
def bread_cost : ℝ := bread_cost_per_pound * bread_wasted
def janitor_cost : ℝ := janitor_normal_wage * 1.5 * janitor_hours

def known_costs : ℝ := fruit_veg_cost + bread_cost + janitor_cost
def meat_cost : ℝ := total_cost - known_costs

theorem wasted_meat_pounds : meat_cost / meat_cost_per_pound = 20 := by
  sorry

end NUMINAMATH_CALUDE_wasted_meat_pounds_l1524_152448


namespace NUMINAMATH_CALUDE_A_intersect_B_is_empty_l1524_152425

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem A_intersect_B_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_empty_l1524_152425


namespace NUMINAMATH_CALUDE_sqrt_eleven_between_integers_l1524_152472

theorem sqrt_eleven_between_integers (a : ℤ) : 3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4 →
  (↑a < Real.sqrt 11 ∧ Real.sqrt 11 < ↑a + 1) ↔ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eleven_between_integers_l1524_152472


namespace NUMINAMATH_CALUDE_intersection_is_origin_l1524_152463

/-- The line l in Cartesian coordinates -/
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * x

/-- The curve C in Cartesian coordinates -/
def curve_C (x y : ℝ) : Prop := y = (1/2) * x^2

/-- The intersection point of line l and curve C -/
def intersection_point : ℝ × ℝ := (0, 0)

/-- Theorem stating that the intersection point is (0, 0) -/
theorem intersection_is_origin :
  line_l (intersection_point.1) (intersection_point.2) ∧
  curve_C (intersection_point.1) (intersection_point.2) :=
sorry

end NUMINAMATH_CALUDE_intersection_is_origin_l1524_152463


namespace NUMINAMATH_CALUDE_coffee_table_price_l1524_152467

theorem coffee_table_price 
  (sofa_price : ℕ) 
  (armchair_price : ℕ) 
  (num_armchairs : ℕ) 
  (total_invoice : ℕ) 
  (h1 : sofa_price = 1250)
  (h2 : armchair_price = 425)
  (h3 : num_armchairs = 2)
  (h4 : total_invoice = 2430) :
  total_invoice - (sofa_price + num_armchairs * armchair_price) = 330 := by
sorry

end NUMINAMATH_CALUDE_coffee_table_price_l1524_152467


namespace NUMINAMATH_CALUDE_craig_commission_l1524_152486

/-- Calculates the total commission for Craig's appliance sales. -/
def total_commission (
  refrigerator_base : ℝ)
  (refrigerator_rate : ℝ)
  (washing_machine_base : ℝ)
  (washing_machine_rate : ℝ)
  (oven_base : ℝ)
  (oven_rate : ℝ)
  (refrigerator_count : ℕ)
  (refrigerator_total_price : ℝ)
  (washing_machine_count : ℕ)
  (washing_machine_total_price : ℝ)
  (oven_count : ℕ)
  (oven_total_price : ℝ) : ℝ :=
  (refrigerator_count * (refrigerator_base + refrigerator_rate * refrigerator_total_price)) +
  (washing_machine_count * (washing_machine_base + washing_machine_rate * washing_machine_total_price)) +
  (oven_count * (oven_base + oven_rate * oven_total_price))

/-- Craig's total commission for the week is $5620.20. -/
theorem craig_commission :
  total_commission 75 0.08 50 0.10 60 0.12 3 5280 4 2140 5 4620 = 5620.20 := by
  sorry

end NUMINAMATH_CALUDE_craig_commission_l1524_152486


namespace NUMINAMATH_CALUDE_negation_equivalence_l1524_152406

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 + 1 > 2*x) ↔ (∃ x : ℝ, x^2 + 1 ≤ 2*x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1524_152406


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l1524_152438

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 300670

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 3.0067
    exponent := 5
    h1 := by sorry }

/-- Theorem stating that the proposed notation correctly represents the original number -/
theorem correct_scientific_notation :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = original_number := by
  sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l1524_152438


namespace NUMINAMATH_CALUDE_deshaun_summer_reading_l1524_152403

theorem deshaun_summer_reading 
  (summer_break_days : ℕ) 
  (avg_pages_per_book : ℕ) 
  (closest_person_percentage : ℚ) 
  (second_person_pages_per_day : ℕ) 
  (h1 : summer_break_days = 80)
  (h2 : avg_pages_per_book = 320)
  (h3 : closest_person_percentage = 3/4)
  (h4 : second_person_pages_per_day = 180) :
  ∃ (books_read : ℕ), books_read = 60 ∧ 
    (books_read * avg_pages_per_book : ℚ) = 
      (second_person_pages_per_day * summer_break_days : ℚ) / closest_person_percentage :=
by sorry

end NUMINAMATH_CALUDE_deshaun_summer_reading_l1524_152403


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1524_152458

/-- A rectangular solid with prime edge lengths and volume 385 has surface area 334 -/
theorem rectangular_solid_surface_area : 
  ∀ (l w h : ℕ), 
    Prime l → Prime w → Prime h →
    l * w * h = 385 →
    2 * (l * w + l * h + w * h) = 334 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1524_152458


namespace NUMINAMATH_CALUDE_min_n_for_geometric_sum_l1524_152466

theorem min_n_for_geometric_sum (n : ℕ) : 
  (∀ k : ℕ, k < n → (2^(k+1) - 1) ≤ 128) ∧ 
  (2^(n+1) - 1) > 128 → 
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_min_n_for_geometric_sum_l1524_152466


namespace NUMINAMATH_CALUDE_composition_ratio_l1524_152481

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem composition_ratio :
  (f (g (f 1))) / (g (f (g 1))) = -23 / 5 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l1524_152481


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1524_152414

/-- Two arithmetic sequences and their sums -/
structure ArithmeticSequencePair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum of first n terms of a
  T : ℕ → ℚ  -- Sum of first n terms of b

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequencePair)
  (h : ∀ n : ℕ, seq.S n / seq.T n = (n + 3) / (2 * n + 1)) :
  seq.a 6 / seq.b 6 = 14 / 23 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1524_152414


namespace NUMINAMATH_CALUDE_smallest_six_digit_number_divisible_by_3_4_5_l1524_152436

def is_divisible_by (n m : Nat) : Prop := n % m = 0

theorem smallest_six_digit_number_divisible_by_3_4_5 :
  ∀ n : Nat,
    325000 ≤ n ∧ n < 326000 →
    is_divisible_by n 3 ∧ is_divisible_by n 4 ∧ is_divisible_by n 5 →
    325020 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_six_digit_number_divisible_by_3_4_5_l1524_152436


namespace NUMINAMATH_CALUDE_alyssa_toy_spending_l1524_152498

/-- The amount Alyssa spent on a football -/
def football_cost : ℚ := 571/100

/-- The amount Alyssa spent on marbles -/
def marbles_cost : ℚ := 659/100

/-- The total amount Alyssa spent on toys -/
def total_cost : ℚ := football_cost + marbles_cost

theorem alyssa_toy_spending :
  total_cost = 1230/100 := by sorry

end NUMINAMATH_CALUDE_alyssa_toy_spending_l1524_152498


namespace NUMINAMATH_CALUDE_mario_orange_consumption_l1524_152441

/-- Represents the amount of fruit eaten by each person in ounces -/
structure FruitConsumption where
  mario : ℕ
  lydia : ℕ
  nicolai : ℕ

/-- Converts pounds to ounces -/
def poundsToOunces (pounds : ℕ) : ℕ := pounds * 16

/-- Theorem: Given the conditions, Mario ate 8 ounces of oranges -/
theorem mario_orange_consumption (total : ℕ) (fc : FruitConsumption) 
  (h1 : poundsToOunces total = fc.mario + fc.lydia + fc.nicolai)
  (h2 : total = 8)
  (h3 : fc.lydia = 24)
  (h4 : fc.nicolai = poundsToOunces 6) :
  fc.mario = 8 := by
  sorry

#check mario_orange_consumption

end NUMINAMATH_CALUDE_mario_orange_consumption_l1524_152441


namespace NUMINAMATH_CALUDE_fraction_equality_l1524_152452

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : a/b + (a+6*b)/(b+6*a) = 3) :
  a/b = (8 + Real.sqrt 46)/6 ∨ a/b = (8 - Real.sqrt 46)/6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1524_152452
