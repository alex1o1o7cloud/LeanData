import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l3381_338173

/-- Given an arithmetic sequence {aₙ} with sum of first n terms Sₙ,
    if a₄/a₈ = 2/3, then S₇/S₁₅ = 14/45 -/
theorem arithmetic_sequence_sum_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n : ℚ) / 2 * (a 1 + a n))
  (h_ratio : a 4 / a 8 = 2 / 3) :
  S 7 / S 15 = 14 / 45 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l3381_338173


namespace NUMINAMATH_CALUDE_first_digit_base8_725_l3381_338164

/-- The first digit of the base 8 representation of a natural number -/
def firstDigitBase8 (n : ℕ) : ℕ :=
  sorry

/-- The base 10 number we're converting -/
def base10Number : ℕ := 725

/-- Theorem stating that the first digit of 725 in base 8 is 1 -/
theorem first_digit_base8_725 : firstDigitBase8 base10Number = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_base8_725_l3381_338164


namespace NUMINAMATH_CALUDE_toy_value_proof_l3381_338178

theorem toy_value_proof (total_toys : ℕ) (total_value : ℕ) (special_toy_value : ℕ) :
  total_toys = 9 →
  total_value = 52 →
  special_toy_value = 12 →
  ∃ (other_toy_value : ℕ),
    (total_toys - 1) * other_toy_value + special_toy_value = total_value ∧
    other_toy_value = 5 := by
  sorry

end NUMINAMATH_CALUDE_toy_value_proof_l3381_338178


namespace NUMINAMATH_CALUDE_min_tokens_correct_l3381_338152

/-- The minimum number of tokens required to fill an n × m grid -/
def min_tokens (n m : ℕ) : ℕ :=
  if n % 2 = 0 ∨ m % 2 = 0 then
    (n + 1) / 2 + (m + 1) / 2
  else
    (n + 1) / 2 + (m + 1) / 2 - 1

/-- A function that determines if a grid can be filled given initial token placement -/
def can_fill_grid (n m : ℕ) (initial_tokens : Finset (ℕ × ℕ)) : Prop :=
  sorry

theorem min_tokens_correct (n m : ℕ) :
  ∀ (k : ℕ), k < min_tokens n m →
    ¬∃ (initial_tokens : Finset (ℕ × ℕ)),
      initial_tokens.card = k ∧
      can_fill_grid n m initial_tokens :=
  sorry

end NUMINAMATH_CALUDE_min_tokens_correct_l3381_338152


namespace NUMINAMATH_CALUDE_min_value_of_function_l3381_338170

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  (x^2 + 7*x + 10) / (x + 1) ≥ 9 ∧
  ∃ y : ℝ, y > -1 ∧ (y^2 + 7*y + 10) / (y + 1) = 9 :=
by
  sorry

#check min_value_of_function

end NUMINAMATH_CALUDE_min_value_of_function_l3381_338170


namespace NUMINAMATH_CALUDE_congruence_implies_b_zero_l3381_338172

theorem congruence_implies_b_zero (a b c m : ℤ) (h_m : m > 1) 
  (h_cong : ∀ n : ℕ, (a^n + b*n + c) % m = 0) : 
  b % m = 0 ∧ (b^2) % m = 0 := by
  sorry

end NUMINAMATH_CALUDE_congruence_implies_b_zero_l3381_338172


namespace NUMINAMATH_CALUDE_fraction_transformation_l3381_338187

theorem fraction_transformation (x : ℚ) : 
  (3 + 2*x) / (4 + 3*x) = 5/9 → x = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l3381_338187


namespace NUMINAMATH_CALUDE_sin_product_equals_neg_two_fifths_l3381_338133

theorem sin_product_equals_neg_two_fifths (θ : Real) (h : Real.tan θ = 2) :
  Real.sin θ * Real.sin (3 * Real.pi / 2 + θ) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_neg_two_fifths_l3381_338133


namespace NUMINAMATH_CALUDE_basketball_probability_l3381_338189

theorem basketball_probability (free_throw high_school pro : ℝ) 
  (h1 : free_throw = 4/5)
  (h2 : high_school = 1/2)
  (h3 : pro = 1/3) :
  1 - (1 - free_throw) * (1 - high_school) * (1 - pro) = 14/15 := by
  sorry

end NUMINAMATH_CALUDE_basketball_probability_l3381_338189


namespace NUMINAMATH_CALUDE_triangle_problem_l3381_338135

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (2 * Real.sqrt 3 / 3 * b * c * Real.sin A = b^2 + c^2 - a^2) →
  (c = 5) →
  (Real.cos B = 1 / 7) →
  (A = π / 3 ∧ b = 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3381_338135


namespace NUMINAMATH_CALUDE_regression_line_intercept_l3381_338142

/-- Given a regression line with slope 1.23 passing through (4, 5), prove its y-intercept is 0.08 -/
theorem regression_line_intercept (slope : ℝ) (x₀ y₀ : ℝ) (h1 : slope = 1.23) (h2 : x₀ = 4) (h3 : y₀ = 5) :
  y₀ = slope * x₀ + 0.08 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_intercept_l3381_338142


namespace NUMINAMATH_CALUDE_relay_race_average_time_l3381_338180

/-- Calculates the average time per leg in a two-leg relay race -/
def average_time_per_leg (time_y time_z : ℕ) : ℚ :=
  (time_y + time_z : ℚ) / 2

/-- Theorem: The average time per leg for the given relay race is 42 seconds -/
theorem relay_race_average_time :
  average_time_per_leg 58 26 = 42 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_average_time_l3381_338180


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_a_nonzero_b_zero_l3381_338126

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_iff_a_nonzero_b_zero (a b : ℝ) :
  is_purely_imaginary (Complex.mk b (a)) ↔ a ≠ 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_a_nonzero_b_zero_l3381_338126


namespace NUMINAMATH_CALUDE_expression_equality_l3381_338129

theorem expression_equality : 
  (Real.sqrt 12) / 2 + |Real.sqrt 3 - 2| - Real.tan (π / 3) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3381_338129


namespace NUMINAMATH_CALUDE_ramp_cost_is_2950_l3381_338125

/-- Calculate the total cost of installing a ramp --/
def total_ramp_cost (permit_cost : ℝ) (contractor_hourly_rate : ℝ) 
  (contractor_days : ℕ) (contractor_hours_per_day : ℕ) 
  (inspector_discount_percent : ℝ) : ℝ :=
  let contractor_total_hours := contractor_days * contractor_hours_per_day
  let contractor_cost := contractor_hourly_rate * contractor_total_hours
  let inspector_cost := contractor_cost * (1 - inspector_discount_percent)
  permit_cost + contractor_cost + inspector_cost

/-- Theorem stating the total cost of installing a ramp is $2950 --/
theorem ramp_cost_is_2950 : 
  total_ramp_cost 250 150 3 5 0.8 = 2950 := by
  sorry

end NUMINAMATH_CALUDE_ramp_cost_is_2950_l3381_338125


namespace NUMINAMATH_CALUDE_second_half_speed_l3381_338163

/-- Proves that given a journey of 224 km completed in 10 hours, where the first half is traveled at 21 km/hr, the speed of the second half of the journey is 24 km/hr. -/
theorem second_half_speed (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) 
    (h1 : total_distance = 224)
    (h2 : total_time = 10)
    (h3 : first_half_speed = 21) : 
  (total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed) = 24 := by
  sorry

#check second_half_speed

end NUMINAMATH_CALUDE_second_half_speed_l3381_338163


namespace NUMINAMATH_CALUDE_fifteenth_row_seats_l3381_338158

/-- Represents the number of seats in a given row of the sports palace. -/
def seats_in_row (n : ℕ) : ℕ := 5 + 2 * (n - 1)

/-- Theorem stating that the 15th row of the sports palace has 33 seats. -/
theorem fifteenth_row_seats : seats_in_row 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_row_seats_l3381_338158


namespace NUMINAMATH_CALUDE_line_through_three_points_l3381_338155

/-- A line contains the points (-2, 7), (7, k), and (21, 4). This theorem proves that k = 134/23. -/
theorem line_through_three_points (k : ℚ) : 
  (∃ (m b : ℚ), 
    (7 : ℚ) = m * (-2 : ℚ) + b ∧ 
    k = m * (7 : ℚ) + b ∧ 
    (4 : ℚ) = m * (21 : ℚ) + b) → 
  k = 134 / 23 := by
sorry

end NUMINAMATH_CALUDE_line_through_three_points_l3381_338155


namespace NUMINAMATH_CALUDE_triangle_perimeter_from_excircle_radii_l3381_338114

theorem triangle_perimeter_from_excircle_radii (a b c : ℝ) (ra rb rc : ℝ) :
  ra = 3 ∧ rb = 10 ∧ rc = 15 →
  ra > 0 ∧ rb > 0 ∧ rc > 0 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  (b + c - a) / 2 = ra ∧ (a + c - b) / 2 = rb ∧ (a + b - c) / 2 = rc →
  a + b + c = 30 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_from_excircle_radii_l3381_338114


namespace NUMINAMATH_CALUDE_rowing_distance_calculation_l3381_338130

/-- Represents the problem of calculating the distance to a destination given rowing conditions. -/
theorem rowing_distance_calculation 
  (rowing_speed : ℝ) 
  (current_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : rowing_speed = 10) 
  (h2 : current_speed = 2) 
  (h3 : total_time = 5) : 
  ∃ (distance : ℝ), distance = 24 ∧ 
    distance / (rowing_speed + current_speed) + 
    distance / (rowing_speed - current_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_rowing_distance_calculation_l3381_338130


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3381_338185

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  a 2 = 8 →                     -- Given condition
  a 5 = 64 →                    -- Given condition
  q = 2 :=                      -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3381_338185


namespace NUMINAMATH_CALUDE_cyclic_fraction_inequality_l3381_338100

theorem cyclic_fraction_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z := by
  sorry

end NUMINAMATH_CALUDE_cyclic_fraction_inequality_l3381_338100


namespace NUMINAMATH_CALUDE_factorial_calculation_l3381_338123

theorem factorial_calculation : (Nat.factorial 15) / ((Nat.factorial 7) * (Nat.factorial 8)) * 2 = 1286 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l3381_338123


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_384_l3381_338120

/-- Represents the number of executives -/
def num_executives : ℕ := 5

/-- Represents the total number of people (executives + partners) -/
def total_people : ℕ := 2 * num_executives

/-- Calculates the number of distinct seating arrangements -/
def seating_arrangements : ℕ :=
  (List.range num_executives).foldl (λ acc i => acc * (total_people - 2 * i)) 1 / total_people

/-- Theorem stating that the number of distinct seating arrangements is 384 -/
theorem seating_arrangements_eq_384 : seating_arrangements = 384 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_eq_384_l3381_338120


namespace NUMINAMATH_CALUDE_remainder_of_product_l3381_338118

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => a₁ + i * d)

def product_of_list (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

theorem remainder_of_product (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 3 ∧ d = 10 ∧ n = 20 →
  (product_of_list (arithmetic_sequence a₁ d n)) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_l3381_338118


namespace NUMINAMATH_CALUDE_difference_of_squares_l3381_338121

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3381_338121


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3381_338141

def is_valid (A : ℕ+) : Prop :=
  ∃ (a b : ℕ), 
    A = 2^a * 3^b ∧
    (a + 1) * (b + 1) = 3 * a * b

theorem smallest_valid_number : 
  is_valid 12 ∧ ∀ A : ℕ+, A < 12 → ¬is_valid A :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3381_338141


namespace NUMINAMATH_CALUDE_triangle_congruence_problem_l3381_338165

theorem triangle_congruence_problem (x y z : ℝ) : 
  (x + y + z = 3) →
  (z + 6 = 2*y - z) →
  (x + 8*z = y + 2) →
  x^2 + y^2 + z^2 = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_congruence_problem_l3381_338165


namespace NUMINAMATH_CALUDE_intersection_A_B_C_subset_intersection_A_B_l3381_338188

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 2*x - 8 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 3 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 3*a*x + 2*a^2 < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 4} := by sorry

-- Theorem for the range of a such that C is a subset of A ∩ B
theorem C_subset_intersection_A_B (a : ℝ) : 
  C a ⊆ (A ∩ B) ↔ (a = 0 ∨ (1 ≤ a ∧ a ≤ 2)) := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_C_subset_intersection_A_B_l3381_338188


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3381_338190

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  (Real.sqrt (a.1^2 + a.2^2) = 4) →
  (Real.sqrt (b.1^2 + b.2^2) = 3) →
  (angle_between a b = Real.pi / 3) →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 37 :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3381_338190


namespace NUMINAMATH_CALUDE_delegate_seating_probability_l3381_338117

/-- Represents the number of delegates -/
def total_delegates : ℕ := 12

/-- Represents the number of countries -/
def num_countries : ℕ := 3

/-- Represents the number of delegates per country -/
def delegates_per_country : ℕ := 4

/-- Calculates the probability of each delegate sitting next to at least one delegate from another country -/
def seating_probability : ℚ :=
  221 / 231

/-- Theorem stating that the probability of each delegate sitting next to at least one delegate 
    from another country is 221/231 -/
theorem delegate_seating_probability :
  seating_probability = 221 / 231 := by sorry

end NUMINAMATH_CALUDE_delegate_seating_probability_l3381_338117


namespace NUMINAMATH_CALUDE_square_reciprocal_sum_equality_l3381_338107

theorem square_reciprocal_sum_equality (n m k : ℕ+) : 
  (1 : ℚ) / n.val^2 + (1 : ℚ) / m.val^2 = (k : ℚ) / (n.val^2 + m.val^2) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_reciprocal_sum_equality_l3381_338107


namespace NUMINAMATH_CALUDE_projection_vector_l3381_338102

/-- Given a vector b and the dot product of vectors a and b, 
    prove that the projection of a onto b is as calculated. -/
theorem projection_vector (a b : ℝ × ℝ) (h : a • b = 10) 
    (hb : b = (3, 4)) : 
  (a • b / (b • b)) • b = (6/5, 8/5) := by
  sorry

end NUMINAMATH_CALUDE_projection_vector_l3381_338102


namespace NUMINAMATH_CALUDE_binary_1010_to_decimal_l3381_338166

/-- Converts a list of binary digits to its decimal representation. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1010₂ -/
def binary_1010 : List Bool := [false, true, false, true]

/-- Theorem stating that the decimal representation of 1010₂ is 10 -/
theorem binary_1010_to_decimal :
  binary_to_decimal binary_1010 = 10 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010_to_decimal_l3381_338166


namespace NUMINAMATH_CALUDE_smallest_perfect_cube_divisor_l3381_338153

theorem smallest_perfect_cube_divisor
  (p q r : ℕ)
  (hp : Prime p)
  (hq : Prime q)
  (hr : Prime r)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)
  (h1_not_prime : ¬ Prime 1)
  (n : ℕ)
  (hn : n = p * q^3 * r^6) :
  ∃ (m : ℕ), m^3 = p^3 * q^3 * r^6 ∧
    ∀ (k : ℕ), (k^3 ≥ n) → (k^3 ≥ m^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_cube_divisor_l3381_338153


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3381_338162

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a coloring function type
def Coloring := ℤ × ℤ → Color

-- Define a rectangle type
structure Rectangle where
  x1 : ℤ
  y1 : ℤ
  x2 : ℤ
  y2 : ℤ
  h_x : x1 < x2
  h_y : y1 < y2

-- State the theorem
theorem monochromatic_rectangle_exists (c : Coloring) :
  ∃ (r : Rectangle), 
    c (r.x1, r.y1) = c (r.x1, r.y2) ∧
    c (r.x1, r.y1) = c (r.x2, r.y1) ∧
    c (r.x1, r.y1) = c (r.x2, r.y2) :=
by sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l3381_338162


namespace NUMINAMATH_CALUDE_abs_ratio_equals_sqrt_eleven_sevenths_l3381_338106

theorem abs_ratio_equals_sqrt_eleven_sevenths (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + b^2 = 9*a*b) : 
  |((a+b)/(a-b))| = Real.sqrt (11/7) := by
sorry

end NUMINAMATH_CALUDE_abs_ratio_equals_sqrt_eleven_sevenths_l3381_338106


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3381_338157

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem parallel_lines_a_value :
  ∀ a : ℝ,
  let l1 : Line := { a := a, b := 2, c := 6 }
  let l2 : Line := { a := 1, b := a - 1, c := 3 }
  parallel l1 l2 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3381_338157


namespace NUMINAMATH_CALUDE_water_evaporation_rate_l3381_338167

/-- Given a bowl with water that experiences evaporation over time, 
    calculate the amount of water evaporated per day. -/
theorem water_evaporation_rate 
  (initial_water : ℝ) 
  (evaporation_period : ℕ) 
  (evaporation_percentage : ℝ) 
  (h1 : initial_water = 10)
  (h2 : evaporation_period = 50)
  (h3 : evaporation_percentage = 0.03)
  : (initial_water * evaporation_percentage) / evaporation_period = 0.06 := by
  sorry


end NUMINAMATH_CALUDE_water_evaporation_rate_l3381_338167


namespace NUMINAMATH_CALUDE_room_area_calculation_l3381_338169

/-- The area of a rectangular room with width 8 feet and length 1.5 feet is 12 square feet. -/
theorem room_area_calculation (width length area : ℝ) : 
  width = 8 → length = 1.5 → area = width * length → area = 12 := by
sorry

end NUMINAMATH_CALUDE_room_area_calculation_l3381_338169


namespace NUMINAMATH_CALUDE_trig_simplification_l3381_338144

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.cos x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3381_338144


namespace NUMINAMATH_CALUDE_port_distance_equation_l3381_338181

/-- The distance between two ports satisfies a specific equation based on ship and river speeds --/
theorem port_distance_equation (ship_speed : ℝ) (current_speed : ℝ) (time_difference : ℝ) 
  (h1 : ship_speed = 26)
  (h2 : current_speed = 2)
  (h3 : time_difference = 3) :
  ∃ x : ℝ, x / (ship_speed + current_speed) = x / (ship_speed - current_speed) - time_difference :=
by sorry

end NUMINAMATH_CALUDE_port_distance_equation_l3381_338181


namespace NUMINAMATH_CALUDE_top_layer_blocks_l3381_338127

/-- Represents a four-layer pyramid with a specific block distribution -/
structure Pyramid :=
  (top : ℕ)  -- Number of blocks in the top layer

/-- The total number of blocks in the pyramid -/
def Pyramid.total (p : Pyramid) : ℕ :=
  p.top + 3 * p.top + 9 * p.top + 27 * p.top

theorem top_layer_blocks (p : Pyramid) :
  p.total = 40 → p.top = 1 := by
  sorry

#check top_layer_blocks

end NUMINAMATH_CALUDE_top_layer_blocks_l3381_338127


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_eq_4_div_sqrt5_l3381_338108

noncomputable def hyperbola_real_axis_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (P : ℝ × ℝ) 
  (hP : P.1^2 / a^2 - P.2^2 / b^2 = 1) 
  (hP_right : P.1 > 0) 
  (A B : ℝ × ℝ) 
  (hA : A.1 > 0 ∧ A.2 > 0) 
  (hB : B.1 > 0 ∧ B.2 < 0) 
  (hAP_PB : (A.1 - P.1, A.2 - P.2) = (-1/3) • (B.1 - P.1, B.2 - P.2)) 
  (hAOB_area : (1/2) * abs (A.1 * B.2 - A.2 * B.1) = 2 * b) : 
  ℝ :=
2 * a

theorem hyperbola_real_axis_length_eq_4_div_sqrt5 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (P : ℝ × ℝ) 
  (hP : P.1^2 / a^2 - P.2^2 / b^2 = 1) 
  (hP_right : P.1 > 0) 
  (A B : ℝ × ℝ) 
  (hA : A.1 > 0 ∧ A.2 > 0) 
  (hB : B.1 > 0 ∧ B.2 < 0) 
  (hAP_PB : (A.1 - P.1, A.2 - P.2) = (-1/3) • (B.1 - P.1, B.2 - P.2)) 
  (hAOB_area : (1/2) * abs (A.1 * B.2 - A.2 * B.1) = 2 * b) : 
  hyperbola_real_axis_length a b ha hb P hP hP_right A B hA hB hAP_PB hAOB_area = 4 / Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_eq_4_div_sqrt5_l3381_338108


namespace NUMINAMATH_CALUDE_randy_block_difference_l3381_338145

/-- Randy's block building problem -/
theorem randy_block_difference :
  ∀ (total_blocks house_blocks tower_blocks : ℕ),
    total_blocks = 90 →
    house_blocks = 89 →
    tower_blocks = 63 →
    house_blocks - tower_blocks = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_randy_block_difference_l3381_338145


namespace NUMINAMATH_CALUDE_vector_BC_l3381_338124

def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (3, 2)
def AC : ℝ × ℝ := (-4, -3)

theorem vector_BC : (B.1 - A.1 + AC.1, B.2 - A.2 + AC.2) = (-7, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_l3381_338124


namespace NUMINAMATH_CALUDE_five_three_bar_equals_sixteen_thirds_l3381_338116

/-- Represents a repeating decimal with an integer part and a repeating fractional part. -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number. -/
def repeatingDecimalToRational (x : RepeatingDecimal) : ℚ :=
  x.integerPart + (x.repeatingPart : ℚ) / 9

/-- The repeating decimal 5.3̄ -/
def five_three_bar : RepeatingDecimal :=
  { integerPart := 5, repeatingPart := 3 }

/-- Theorem: The repeating decimal 5.3̄ is equal to 16/3 -/
theorem five_three_bar_equals_sixteen_thirds :
  repeatingDecimalToRational five_three_bar = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_five_three_bar_equals_sixteen_thirds_l3381_338116


namespace NUMINAMATH_CALUDE_distance_between_points_l3381_338160

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 7)
  let p2 : ℝ × ℝ := (3, -2)
  dist p1 p2 = 9 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3381_338160


namespace NUMINAMATH_CALUDE_identical_lines_condition_no_identical_lines_l3381_338149

/-- Two lines are identical if and only if they have the same slope and y-intercept -/
theorem identical_lines_condition (a b : ℝ) : 
  (∀ x y : ℝ, 2*x + a*y + b = 0 ↔ b*x - 3*y + 15 = 0) ↔ 
  ((-2/a = b/3) ∧ (-b/a = -5)) :=
sorry

/-- There are no real pairs (a, b) such that the lines 2x + ay + b = 0 and bx - 3y + 15 = 0 have the same graph -/
theorem no_identical_lines : ¬∃ a b : ℝ, ∀ x y : ℝ, 2*x + a*y + b = 0 ↔ b*x - 3*y + 15 = 0 :=
sorry

end NUMINAMATH_CALUDE_identical_lines_condition_no_identical_lines_l3381_338149


namespace NUMINAMATH_CALUDE_points_on_line_l3381_338132

/-- Given a line defined by x = (y^2 / 3) - (2 / 5), if three points (m, n), (m + p, n + 9), and (m + q, n + 18) lie on this line, then p = 6n + 27 and q = 12n + 108 -/
theorem points_on_line (m n p q : ℝ) : 
  (m = n^2 / 3 - 2 / 5) →
  (m + p = (n + 9)^2 / 3 - 2 / 5) →
  (m + q = (n + 18)^2 / 3 - 2 / 5) →
  (p = 6 * n + 27 ∧ q = 12 * n + 108) := by
sorry

end NUMINAMATH_CALUDE_points_on_line_l3381_338132


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l3381_338146

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (a b : ℝ), x^2 - 6*x + 7 = 0 ↔ (x + a)^2 = b ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l3381_338146


namespace NUMINAMATH_CALUDE_binomial_expansion_m_value_l3381_338111

/-- Given a binomial expansion (mx+1)^n where the 5th term has the largest
    coefficient and the coefficient of x^3 is 448, prove that m = 2 -/
theorem binomial_expansion_m_value (m : ℝ) (n : ℕ) : 
  (∃ k : ℕ, k = 5 ∧ 
    ∀ j : ℕ, j ≤ n + 1 → Nat.choose n (j - 1) * m^(j - 1) ≤ Nat.choose n (k - 1) * m^(k - 1)) ∧
  Nat.choose n 3 * m^3 = 448 →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_m_value_l3381_338111


namespace NUMINAMATH_CALUDE_train_speed_l3381_338184

def train_length : ℝ := 100
def tunnel_length : ℝ := 2300
def time_seconds : ℝ := 120

theorem train_speed :
  let total_distance := tunnel_length + train_length
  let speed_ms := total_distance / time_seconds
  let speed_kmh := speed_ms * 3.6
  speed_kmh = 72 := by sorry

end NUMINAMATH_CALUDE_train_speed_l3381_338184


namespace NUMINAMATH_CALUDE_rhombus_area_l3381_338159

/-- A rhombus with side length √113 and diagonal difference 8 has area 194. -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 113 →
  diag_diff = 8 →
  area = (Real.sqrt 210)^2 - 4^2 →
  area = 194 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l3381_338159


namespace NUMINAMATH_CALUDE_tom_fruit_purchase_l3381_338194

/-- Given Tom's purchase of apples and mangoes, prove the amount of mangoes bought -/
theorem tom_fruit_purchase (apple_kg : ℕ) (apple_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) 
  (h1 : apple_kg = 8)
  (h2 : apple_rate = 70)
  (h3 : mango_rate = 70)
  (h4 : total_paid = 1190) :
  (total_paid - apple_kg * apple_rate) / mango_rate = 9 := by
  sorry

end NUMINAMATH_CALUDE_tom_fruit_purchase_l3381_338194


namespace NUMINAMATH_CALUDE_larger_integer_value_l3381_338134

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * b = 189) :
  a = 21 ∨ b = 21 :=
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l3381_338134


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l3381_338177

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

/-- Given vectors a and b, with a parallel to b, prove that y = 7 -/
theorem parallel_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) 
    (ha : a = (2, 3)) 
    (hb : b = (4, -1 + y)) 
    (h_parallel : parallel a b) : 
  y = 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l3381_338177


namespace NUMINAMATH_CALUDE_range_of_f_l3381_338199

-- Define the function f(x) = |x| - 4
def f (x : ℝ) : ℝ := |x| - 4

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -4} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3381_338199


namespace NUMINAMATH_CALUDE_smallest_number_problem_l3381_338128

theorem smallest_number_problem (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  (a + b + c) / 3 = 30 →
  b = 31 →
  a ≤ b ∧ b ≤ c →
  c = b + 6 →
  a = 22 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_problem_l3381_338128


namespace NUMINAMATH_CALUDE_book_sales_l3381_338143

theorem book_sales (wednesday_sales : ℕ) : 
  wednesday_sales + 3 * wednesday_sales + 3 * wednesday_sales / 5 = 69 → 
  wednesday_sales = 15 := by
sorry

end NUMINAMATH_CALUDE_book_sales_l3381_338143


namespace NUMINAMATH_CALUDE_problem_statement_l3381_338168

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the problem statement
theorem problem_statement (a b : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_equation : 2 * (Real.sqrt (log10 a) + Real.sqrt (log10 b)) + log10 (Real.sqrt a) + log10 (Real.sqrt b) = 108)
  (h_int_sqrt_log_a : ∃ m : ℕ, Real.sqrt (log10 a) = m)
  (h_int_sqrt_log_b : ∃ n : ℕ, Real.sqrt (log10 b) = n)
  (h_int_log_sqrt_a : ∃ k : ℕ, log10 (Real.sqrt a) = k)
  (h_int_log_sqrt_b : ∃ l : ℕ, log10 (Real.sqrt b) = l) :
  a * b = 10^116 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3381_338168


namespace NUMINAMATH_CALUDE_swimming_pool_length_l3381_338112

theorem swimming_pool_length 
  (width : ℝ) 
  (water_removed : ℝ) 
  (water_level_lowered : ℝ) 
  (cubic_foot_to_gallon : ℝ) :
  width = 20 →
  water_removed = 4500 →
  water_level_lowered = 0.5 →
  cubic_foot_to_gallon = 7.5 →
  ∃ (length : ℝ), length = 60 ∧ 
    water_removed / cubic_foot_to_gallon = length * width * water_level_lowered :=
by
  sorry

end NUMINAMATH_CALUDE_swimming_pool_length_l3381_338112


namespace NUMINAMATH_CALUDE_line_passes_first_third_quadrants_iff_positive_slope_l3381_338119

/-- A line passes through the first and third quadrants if and only if its slope is positive -/
theorem line_passes_first_third_quadrants_iff_positive_slope (k : ℝ) :
  (k ≠ 0 ∧ ∀ x y : ℝ, y = k * x → (x > 0 → y > 0) ∧ (x < 0 → y < 0)) ↔ k > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_first_third_quadrants_iff_positive_slope_l3381_338119


namespace NUMINAMATH_CALUDE_existence_of_additive_approximation_l3381_338115

/-- Given a function f: ℝ → ℝ satisfying |f(x+y) - f(x) - f(y)| ≤ 1 for all x, y ∈ ℝ,
    there exists an additive function g: ℝ → ℝ such that |f(x) - g(x)| ≤ 1 for all x ∈ ℝ. -/
theorem existence_of_additive_approximation (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, |f (x + y) - f x - f y| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x y : ℝ, g (x + y) = g x + g y) ∧ 
    (∀ x : ℝ, |f x - g x| ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_additive_approximation_l3381_338115


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l3381_338110

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 0 < x ∧ x < 4}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 3} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 4} := by sorry

-- Theorem for the range of a when B is a subset of C
theorem range_of_a (h : B ⊆ C a) : a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_range_of_a_l3381_338110


namespace NUMINAMATH_CALUDE_ellipse_segment_length_l3381_338131

/-- The length of segment AB for a given ellipse -/
theorem ellipse_segment_length : 
  ∀ (x y : ℝ), 
  (x^2 / 25 + y^2 / 16 = 1) →  -- Ellipse equation
  (∃ (a b : ℝ), 
    (a^2 / 25 + b^2 / 16 = 1) ∧  -- Points A and B satisfy ellipse equation
    (a = 3) ∧  -- x-coordinate of right focus
    (b = 16/5 ∨ b = -16/5)) →  -- y-coordinates of intersection points
  (16/5 - (-16/5) = 32/5) :=  -- Length of segment AB
by
  sorry

end NUMINAMATH_CALUDE_ellipse_segment_length_l3381_338131


namespace NUMINAMATH_CALUDE_room_occupancy_l3381_338154

theorem room_occupancy (chairs : ℕ) (people : ℕ) : 
  (2 : ℚ) / 3 * people = (3 : ℚ) / 4 * chairs ∧ 
  chairs - (3 : ℚ) / 4 * chairs = 6 →
  people = 27 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l3381_338154


namespace NUMINAMATH_CALUDE_units_digit_of_3_power_2020_l3381_338179

theorem units_digit_of_3_power_2020 : ∃ n : ℕ, 3^2020 ≡ 1 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_3_power_2020_l3381_338179


namespace NUMINAMATH_CALUDE_inequality_proof_l3381_338191

theorem inequality_proof (x : ℝ) : x > -4/3 → 3 - 1/(3*x + 4) < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3381_338191


namespace NUMINAMATH_CALUDE_min_sum_of_primes_l3381_338198

theorem min_sum_of_primes (p q : ℕ) : 
  p > 1 → q > 1 → Nat.Prime p → Nat.Prime q → 
  17 * (p + 1) = 21 * (q + 1) → 
  (∀ p' q' : ℕ, p' > 1 → q' > 1 → Nat.Prime p' → Nat.Prime q' → 
    17 * (p' + 1) = 21 * (q' + 1) → p + q ≤ p' + q') → 
  p + q = 70 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_l3381_338198


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l3381_338140

theorem two_digit_number_interchange (a b k : ℕ) (h1 : a ≥ 1 ∧ a ≤ 9) (h2 : b ≤ 9) 
  (h3 : 10 * a + b = k * (a + b)) :
  10 * b + a = (11 - k) * (a + b) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l3381_338140


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3381_338175

/-- The parabola is defined by the equation y = -x^2 + 3x - 4 -/
def parabola (x y : ℝ) : Prop := y = -x^2 + 3*x - 4

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (x y : ℝ) : Prop := x = 0

theorem parabola_y_axis_intersection :
  ∃ (x y : ℝ), parabola x y ∧ on_y_axis x y ∧ x = 0 ∧ y = -4 :=
sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3381_338175


namespace NUMINAMATH_CALUDE_total_sum_lent_l3381_338195

/-- Proves that the total sum lent is 2665 given the specified conditions --/
theorem total_sum_lent (first_part second_part : ℕ) : 
  second_part = 1640 →
  (first_part * 8 * 3) = (second_part * 3 * 5) →
  first_part + second_part = 2665 := by
  sorry

#check total_sum_lent

end NUMINAMATH_CALUDE_total_sum_lent_l3381_338195


namespace NUMINAMATH_CALUDE_time_for_A_to_reach_B_l3381_338183

-- Define the total distance between points A and B
variable (S : ℝ) 

-- Define the speeds of A and B
variable (v_A v_B : ℝ)

-- Define the time when B catches up to A for the first time
variable (t : ℝ)

-- Theorem statement
theorem time_for_A_to_reach_B 
  (h1 : v_A * (t + 48/60) = v_B * t) 
  (h2 : v_A * (t + 48/60) = 2/3 * S) 
  (h3 : v_A * (t + 48/60 + 1/2 * t + 6/60) + 6/60 * v_B = S) 
  : (108 : ℝ) - (96 : ℝ) = 12 := by
  sorry


end NUMINAMATH_CALUDE_time_for_A_to_reach_B_l3381_338183


namespace NUMINAMATH_CALUDE_triangle_problem_l3381_338148

theorem triangle_problem (a b c A B C : Real) (h1 : (2*b - c) * Real.cos A = a * Real.cos C)
  (h2 : a = Real.sqrt 13) (h3 : b + c = 5) :
  A = π/3 ∧ (1/2 * b * c * Real.sin A = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3381_338148


namespace NUMINAMATH_CALUDE_domain_of_g_l3381_338150

noncomputable def g (x : ℝ) : ℝ := 1 / ⌊x^2 - 8*x + 18⌋

theorem domain_of_g : Set.range g = Set.univ :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l3381_338150


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3381_338176

theorem average_of_remaining_numbers 
  (total : ℕ) 
  (subset : ℕ) 
  (remaining : ℕ) 
  (total_sum : ℚ) 
  (subset_sum : ℚ) 
  (h1 : total = 5) 
  (h2 : subset = 3) 
  (h3 : remaining = total - subset) 
  (h4 : total_sum / total = 6) 
  (h5 : subset_sum / subset = 4) : 
  (total_sum - subset_sum) / remaining = 9 := by
sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3381_338176


namespace NUMINAMATH_CALUDE_inequality_addition_l3381_338161

theorem inequality_addition {a b c d : ℝ} (hab : a > b) (hcd : c > d) (hc : c ≠ 0) (hd : d ≠ 0) :
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l3381_338161


namespace NUMINAMATH_CALUDE_tax_growth_equation_l3381_338174

/-- Represents the average annual growth rate of taxes paid by a company over two years -/
def average_annual_growth_rate (initial_tax : ℝ) (final_tax : ℝ) (years : ℕ) (x : ℝ) : Prop :=
  initial_tax * (1 + x)^years = final_tax

/-- Theorem stating that the equation 40(1+x)^2 = 48.4 correctly represents the average annual growth rate -/
theorem tax_growth_equation (x : ℝ) :
  average_annual_growth_rate 40 48.4 2 x ↔ 40 * (1 + x)^2 = 48.4 :=
by sorry

end NUMINAMATH_CALUDE_tax_growth_equation_l3381_338174


namespace NUMINAMATH_CALUDE_alvin_coconut_trees_l3381_338197

/-- The number of coconuts each tree yields -/
def coconuts_per_tree : ℕ := 5

/-- The price of each coconut in dollars -/
def price_per_coconut : ℕ := 3

/-- The amount Alvin needs to earn in dollars -/
def target_earnings : ℕ := 90

/-- The number of coconut trees Alvin needs to harvest -/
def trees_to_harvest : ℕ := 6

theorem alvin_coconut_trees :
  trees_to_harvest * coconuts_per_tree * price_per_coconut = target_earnings :=
sorry

end NUMINAMATH_CALUDE_alvin_coconut_trees_l3381_338197


namespace NUMINAMATH_CALUDE_lunks_needed_for_two_dozen_oranges_l3381_338113

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks_rate : ℚ := 4 / 2

/-- Exchange rate between kunks and oranges -/
def kunks_to_oranges_rate : ℚ := 3 / 6

/-- Number of oranges in two dozen -/
def two_dozen : ℕ := 24

/-- The number of lunks required to purchase two dozen oranges -/
def lunks_for_two_dozen : ℕ := 24

theorem lunks_needed_for_two_dozen_oranges :
  (two_dozen : ℚ) / kunks_to_oranges_rate * lunks_to_kunks_rate = lunks_for_two_dozen := by
  sorry

end NUMINAMATH_CALUDE_lunks_needed_for_two_dozen_oranges_l3381_338113


namespace NUMINAMATH_CALUDE_right_triangle_area_l3381_338196

theorem right_triangle_area (a b : ℝ) (h : Real.sqrt (a - 5) + (b - 4)^2 = 0) :
  let area := (1/2) * a * b
  area = 6 ∨ area = 10 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3381_338196


namespace NUMINAMATH_CALUDE_least_tiles_required_l3381_338136

def room_length : ℕ := 544
def room_width : ℕ := 374

theorem least_tiles_required (length width : ℕ) (h1 : length = room_length) (h2 : width = room_width) :
  ∃ (tile_size : ℕ), 
    tile_size > 0 ∧
    length % tile_size = 0 ∧
    width % tile_size = 0 ∧
    (length / tile_size) * (width / tile_size) = 176 :=
sorry

end NUMINAMATH_CALUDE_least_tiles_required_l3381_338136


namespace NUMINAMATH_CALUDE_amount_ratio_problem_l3381_338101

theorem amount_ratio_problem (total amount_p amount_q amount_r : ℚ) : 
  total = 1210 →
  amount_p + amount_q + amount_r = total →
  amount_p / amount_q = 5 / 4 →
  amount_r = 400 →
  amount_q / amount_r = 9 / 10 := by
sorry

end NUMINAMATH_CALUDE_amount_ratio_problem_l3381_338101


namespace NUMINAMATH_CALUDE_angle_equality_l3381_338151

theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (20 * π / 180) = Real.cos θ + 2 * Real.sin θ) : 
  θ = 10 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l3381_338151


namespace NUMINAMATH_CALUDE_longest_line_segment_in_pie_slice_l3381_338109

theorem longest_line_segment_in_pie_slice (d : ℝ) (n : ℕ) (h_d : d = 16) (h_n : n = 4) : 
  let r := d / 2
  let θ := 2 * Real.pi / n
  let m := 2 * r * Real.sin (θ / 2)
  m ^ 2 = 128 := by sorry

end NUMINAMATH_CALUDE_longest_line_segment_in_pie_slice_l3381_338109


namespace NUMINAMATH_CALUDE_multiple_of_p_capital_l3381_338104

theorem multiple_of_p_capital (P Q R : ℚ) (total_profit : ℚ) 
  (h1 : ∃ x : ℚ, x * P = 6 * Q)
  (h2 : ∃ x : ℚ, x * P = 10 * R)
  (h3 : total_profit = 4650)
  (h4 : R * total_profit / (P + Q + R) = 900) :
  ∃ x : ℚ, x * P = 10 * R ∧ x = 10 := by sorry

end NUMINAMATH_CALUDE_multiple_of_p_capital_l3381_338104


namespace NUMINAMATH_CALUDE_apple_price_is_two_l3381_338137

/-- The cost of items in Fabian's shopping basket -/
def shopping_cost (apple_price : ℝ) : ℝ :=
  5 * apple_price +  -- 5 kg of apples
  3 * (apple_price - 1) +  -- 3 packs of sugar
  0.5 * 6  -- 500g of walnuts

/-- Theorem: The price of apples is $2 per kg -/
theorem apple_price_is_two :
  ∃ (apple_price : ℝ), apple_price = 2 ∧ shopping_cost apple_price = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_apple_price_is_two_l3381_338137


namespace NUMINAMATH_CALUDE_value_difference_l3381_338147

theorem value_difference (n : ℝ) (h : n = 40) : 
  (n * 1.25) - (n * 0.7) = 22 := by
  sorry

end NUMINAMATH_CALUDE_value_difference_l3381_338147


namespace NUMINAMATH_CALUDE_arccos_negative_half_l3381_338138

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_arccos_negative_half_l3381_338138


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3381_338156

/-- Given two regular polygons with equal perimeters, where one has 50 sides
    and its side length is three times the other's, prove that the number of
    sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →  -- Ensure positive side length
  50 * (3 * s) = n * s →  -- Equal perimeters
  n = 150 := by
  sorry


end NUMINAMATH_CALUDE_second_polygon_sides_l3381_338156


namespace NUMINAMATH_CALUDE_smallest_good_is_correct_l3381_338139

/-- The operation described in the problem -/
def operation (n : ℕ) : ℕ :=
  (n / 10) + 2 * (n % 10)

/-- A number is 'good' if it's unchanged by the operation -/
def is_good (n : ℕ) : Prop :=
  operation n = n

/-- The smallest 'good' number -/
def smallest_good : ℕ :=
  10^99 + 1

theorem smallest_good_is_correct :
  is_good smallest_good ∧ ∀ m : ℕ, m < smallest_good → ¬ is_good m :=
sorry

end NUMINAMATH_CALUDE_smallest_good_is_correct_l3381_338139


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l3381_338171

theorem invalid_votes_percentage 
  (total_votes : ℕ) 
  (candidate_a_percentage : ℚ) 
  (candidate_a_votes : ℕ) 
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 75 / 100)
  (h3 : candidate_a_votes = 357000) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 := by
sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l3381_338171


namespace NUMINAMATH_CALUDE_exponent_calculation_l3381_338182

theorem exponent_calculation : 3^3 * 5^3 * 3^5 * 5^5 = 15^8 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l3381_338182


namespace NUMINAMATH_CALUDE_correct_distribution_l3381_338186

/-- Represents the amount of logs contributed by each person -/
structure Contribution where
  troykina : ℕ
  pyatorkina : ℕ
  bestoplivny : ℕ

/-- Represents the payment made by Bestoplivny in kopecks -/
def bestoplivny_payment : ℕ := 80

/-- Calculates the fair distribution of the payment -/
def calculate_distribution (c : Contribution) : ℕ × ℕ := sorry

/-- Theorem stating the correct distribution of the payment -/
theorem correct_distribution (c : Contribution) 
  (h1 : c.troykina = 3)
  (h2 : c.pyatorkina = 5)
  (h3 : c.bestoplivny = 0) :
  calculate_distribution c = (10, 70) := by sorry

end NUMINAMATH_CALUDE_correct_distribution_l3381_338186


namespace NUMINAMATH_CALUDE_min_bags_l3381_338192

theorem min_bags (total_objects : ℕ) (red_boxes blue_boxes : ℕ) 
  (objects_per_red : ℕ) (objects_per_blue : ℕ) :
  total_objects = 731 ∧ 
  red_boxes = 17 ∧ 
  blue_boxes = 43 ∧ 
  objects_per_red = 43 ∧ 
  objects_per_blue = 17 →
  ∃ (n : ℕ), n > 0 ∧ 
    (∃ (a b : ℕ), a ≤ red_boxes ∧ b ≤ blue_boxes ∧ 
      objects_per_red * a + objects_per_blue * b = total_objects) ∧
    (∀ (m : ℕ), m > 0 ∧ 
      (∃ (a b : ℕ), a ≤ red_boxes ∧ b ≤ blue_boxes ∧ 
        objects_per_red * a + objects_per_blue * b = total_objects) → 
      n ≤ m) ∧
    n = 17 := by
  sorry

end NUMINAMATH_CALUDE_min_bags_l3381_338192


namespace NUMINAMATH_CALUDE_barbara_winning_condition_l3381_338122

/-- The game rules for Alberto and Barbara --/
structure GameRules where
  alberto_choice : ℕ → ℕ
  barbara_choice : ℕ → ℕ → ℕ
  alberto_move : ℕ → ℕ
  max_moves : ℕ

/-- Barbara's winning condition --/
def barbara_wins (n : ℕ) (rules : GameRules) : Prop :=
  ∃ (strategy : ℕ → ℕ), ∀ (alberto_plays : ℕ → ℕ),
    ∃ (m : ℕ), m ≤ rules.max_moves ∧
    (strategy (alberto_plays m)) = n

/-- The main theorem --/
theorem barbara_winning_condition (n : ℕ) (h : n > 1) :
  (∃ (rules : GameRules), barbara_wins n rules) ↔ (∃ (k : ℕ), n = 6 * k) :=
sorry

end NUMINAMATH_CALUDE_barbara_winning_condition_l3381_338122


namespace NUMINAMATH_CALUDE_total_crayons_l3381_338105

/-- Given a box of crayons where there are four times as many red crayons as blue crayons,
    and there are 3 blue crayons, prove that the total number of crayons is 15. -/
theorem total_crayons (blue : ℕ) (red : ℕ) (h1 : blue = 3) (h2 : red = 4 * blue) :
  blue + red = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l3381_338105


namespace NUMINAMATH_CALUDE_f_properties_l3381_338103

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

theorem f_properties :
  (∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y) ∧
  (∀ x : ℝ, x ≠ 0 → f x = f (-x)) ∧
  (∀ x : ℝ, x > 0 → deriv f x > 0) := by sorry

end NUMINAMATH_CALUDE_f_properties_l3381_338103


namespace NUMINAMATH_CALUDE_lte_lemma_largest_power_of_two_dividing_difference_l3381_338193

-- Define the valuation function v_2
def v_2 (n : ℕ) : ℕ := sorry

-- Define the Lifting The Exponent Lemma
theorem lte_lemma (a b : ℕ) (h : Odd a ∧ Odd b) :
  v_2 (a^4 - b^4) = v_2 (a - b) + v_2 4 + v_2 (a + b) - 1 := sorry

-- Main theorem
theorem largest_power_of_two_dividing_difference :
  ∃ k : ℕ, k = 7 ∧ 2^k = (Nat.gcd (17^4 - 15^4) (2^64)) := by sorry

end NUMINAMATH_CALUDE_lte_lemma_largest_power_of_two_dividing_difference_l3381_338193
