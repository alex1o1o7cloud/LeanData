import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_forall_square_leq_l2739_273950

theorem negation_of_forall_square_leq (P : ℝ → Prop) : 
  (¬ ∀ x > 1, x^2 ≤ x) ↔ (∃ x > 1, x^2 > x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_square_leq_l2739_273950


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l2739_273966

/-- A regular triangular pyramid -/
structure RegularTriangularPyramid where
  /-- The dihedral angle between two adjacent faces -/
  α : Real
  /-- The distance from the center of the base to an edge of the lateral face -/
  d : Real

/-- The volume of a regular triangular pyramid -/
noncomputable def volume (p : RegularTriangularPyramid) : Real :=
  (9 * Real.tan p.α ^ 3) / (4 * Real.sqrt (3 * Real.tan p.α ^ 2 - 1))

theorem regular_triangular_pyramid_volume 
  (p : RegularTriangularPyramid) 
  (h1 : p.d = 1) 
  : volume p = (9 * Real.tan p.α ^ 3) / (4 * Real.sqrt (3 * Real.tan p.α ^ 2 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l2739_273966


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2739_273961

theorem gcd_of_three_numbers : Nat.gcd 12903 (Nat.gcd 18239 37422) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2739_273961


namespace NUMINAMATH_CALUDE_magic_king_episodes_l2739_273936

theorem magic_king_episodes (total_seasons : ℕ) 
  (first_half_episodes : ℕ) (second_half_episodes : ℕ) : 
  total_seasons = 10 ∧ 
  first_half_episodes = 20 ∧ 
  second_half_episodes = 25 →
  (total_seasons / 2 * first_half_episodes) + 
  (total_seasons / 2 * second_half_episodes) = 225 := by
  sorry

end NUMINAMATH_CALUDE_magic_king_episodes_l2739_273936


namespace NUMINAMATH_CALUDE_carol_peanuts_l2739_273937

theorem carol_peanuts (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 2 → received = 5 → total = initial + received → total = 7 := by
sorry

end NUMINAMATH_CALUDE_carol_peanuts_l2739_273937


namespace NUMINAMATH_CALUDE_root_expression_value_l2739_273928

theorem root_expression_value (r s : ℝ) : 
  (3 * r^2 + 4 * r - 18 = 0) →
  (3 * s^2 + 4 * s - 18 = 0) →
  r ≠ s →
  (3 * r^3 - 3 * s^3) / (r - s) = 70/3 := by
sorry

end NUMINAMATH_CALUDE_root_expression_value_l2739_273928


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l2739_273992

/-- An ellipse with equation 9x^2 + 25y^2 = 225 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 9 * p.1^2 + 25 * p.2^2 = 225}

/-- A point on the ellipse -/
def PointOnEllipse (p : ℝ × ℝ) : Prop :=
  p ∈ Ellipse

/-- An equilateral triangle -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

/-- The triangle is inscribed in the ellipse -/
def TriangleInscribed (t : EquilateralTriangle) : Prop :=
  PointOnEllipse t.A ∧ PointOnEllipse t.B ∧ PointOnEllipse t.C

/-- One vertex is at (5/3, 0) -/
def VertexAtGivenPoint (t : EquilateralTriangle) : Prop :=
  t.A = (5/3, 0) ∨ t.B = (5/3, 0) ∨ t.C = (5/3, 0)

/-- One altitude is contained in the x-axis -/
def AltitudeOnXAxis (t : EquilateralTriangle) : Prop :=
  (t.A.2 = 0 ∧ t.B.2 = -t.C.2) ∨ (t.B.2 = 0 ∧ t.A.2 = -t.C.2) ∨ (t.C.2 = 0 ∧ t.A.2 = -t.B.2)

/-- The main theorem -/
theorem equilateral_triangle_side_length_squared 
  (t : EquilateralTriangle) 
  (h1 : TriangleInscribed t) 
  (h2 : VertexAtGivenPoint t) 
  (h3 : AltitudeOnXAxis t) : 
  (t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2 = 1475/196 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_squared_l2739_273992


namespace NUMINAMATH_CALUDE_product_of_polynomials_l2739_273994

theorem product_of_polynomials (d p q : ℝ) : 
  (4 * d^3 + 2 * d^2 - 5 * d + p) * (6 * d^2 + q * d - 3) = 
  24 * d^5 + q * d^4 - 33 * d^3 - 15 * d^2 + q * d - 15 → 
  p + q = 12.5 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l2739_273994


namespace NUMINAMATH_CALUDE_total_marbles_in_jar_l2739_273946

def ben_marbles : ℕ := 56
def leo_marbles_difference : ℕ := 20

theorem total_marbles_in_jar : 
  ben_marbles + (ben_marbles + leo_marbles_difference) = 132 := by
sorry

end NUMINAMATH_CALUDE_total_marbles_in_jar_l2739_273946


namespace NUMINAMATH_CALUDE_jake_and_kendra_weight_l2739_273997

/-- Calculates the combined weight of Jake and Kendra given Jake's current weight and the condition about their weight relation after Jake loses 8 pounds. -/
def combinedWeight (jakeWeight : ℕ) : ℕ :=
  let kendraWeight := (jakeWeight - 8) / 2
  jakeWeight + kendraWeight

/-- Theorem stating that given Jake's current weight of 196 pounds and the condition about their weight relation, the combined weight of Jake and Kendra is 290 pounds. -/
theorem jake_and_kendra_weight : combinedWeight 196 = 290 := by
  sorry

#eval combinedWeight 196

end NUMINAMATH_CALUDE_jake_and_kendra_weight_l2739_273997


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2739_273920

/-- An isosceles triangle with two sides of length 6 cm and perimeter 20 cm has a base of length 8 cm. -/
theorem isosceles_triangle_base_length 
  (side_length : ℝ) 
  (perimeter : ℝ) 
  (h1 : side_length = 6) 
  (h2 : perimeter = 20) : 
  perimeter - 2 * side_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l2739_273920


namespace NUMINAMATH_CALUDE_drawer_probability_verify_drawer_probability_l2739_273980

/-- The probability of selecting one shirt, one pair of shorts, and one pair of socks
    from a drawer with 6 shirts, 7 pairs of shorts, and 8 pairs of socks
    when randomly removing three articles of clothing. -/
theorem drawer_probability : ℕ → ℕ → ℕ → ℚ
  | 6, 7, 8 => 168/665
  | _, _, _ => 0

/-- Verifies that the probability is correct for the given problem. -/
theorem verify_drawer_probability :
  drawer_probability 6 7 8 = 168/665 := by sorry

end NUMINAMATH_CALUDE_drawer_probability_verify_drawer_probability_l2739_273980


namespace NUMINAMATH_CALUDE_camp_girls_count_l2739_273941

theorem camp_girls_count (total : ℕ) (difference : ℕ) (girls : ℕ) : 
  total = 133 → difference = 33 → girls + (girls + difference) = total → girls = 50 := by
sorry

end NUMINAMATH_CALUDE_camp_girls_count_l2739_273941


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l2739_273905

-- Define proposition p
def p : Prop := ∀ x : ℝ, (Real.exp x > 1) → (x > 0)

-- Define proposition q
def q : Prop := ∀ x : ℝ, (|x - 3| > 1) → (x > 4)

-- Theorem to prove
theorem p_or_q_is_true : p ∨ q := by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l2739_273905


namespace NUMINAMATH_CALUDE_luncheon_attendance_l2739_273906

/-- A luncheon problem -/
theorem luncheon_attendance (total_invited : ℕ) (tables_needed : ℕ) (capacity_per_table : ℕ)
  (h1 : total_invited = 45)
  (h2 : tables_needed = 5)
  (h3 : capacity_per_table = 2) :
  total_invited - (tables_needed * capacity_per_table) = 35 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_attendance_l2739_273906


namespace NUMINAMATH_CALUDE_largest_k_dividing_factorial_l2739_273954

theorem largest_k_dividing_factorial (n : ℕ) (h : n = 2520) :
  (∃ k : ℕ, k = 629 ∧ 
   (∀ m : ℕ, n^k ∣ n! ∧ 
   (m > k → ¬(n^m ∣ n!)))) :=
sorry

end NUMINAMATH_CALUDE_largest_k_dividing_factorial_l2739_273954


namespace NUMINAMATH_CALUDE_power_multiplication_l2739_273989

theorem power_multiplication (a : ℝ) : a^3 * a^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2739_273989


namespace NUMINAMATH_CALUDE_abc_value_l2739_273901

theorem abc_value (a b c : ℂ) 
  (eq1 : a * b + 5 * b = -20)
  (eq2 : b * c + 5 * c = -20)
  (eq3 : c * a + 5 * a = -20) :
  a * b * c = 100 := by
  sorry

end NUMINAMATH_CALUDE_abc_value_l2739_273901


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2739_273918

theorem coefficient_x_cubed_in_expansion : 
  (Finset.range 21).sum (fun k => (Nat.choose 20 k) * (2^(20 - k)) * (if k = 3 then 1 else 0)) = 149462016 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l2739_273918


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l2739_273964

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_specific :
  arithmetic_sequence_sum (-3) 7 6 = 87 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_specific_l2739_273964


namespace NUMINAMATH_CALUDE_egg_weight_probability_l2739_273938

/-- Given that the probability of an egg's weight being less than 30 grams is 0.30,
    prove that the probability of its weight being not less than 30 grams is 0.70. -/
theorem egg_weight_probability (p_less_than_30 : ℝ) (h1 : p_less_than_30 = 0.30) :
  1 - p_less_than_30 = 0.70 := by
  sorry

end NUMINAMATH_CALUDE_egg_weight_probability_l2739_273938


namespace NUMINAMATH_CALUDE_inverse_g_equals_five_l2739_273971

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 3

-- State the theorem
theorem inverse_g_equals_five (x : ℝ) : g (g⁻¹ x) = x → g⁻¹ x = 5 → x = 503 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_equals_five_l2739_273971


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l2739_273963

/-- The quadratic equation x^2 + (m-1)x + 1 = 0 has solutions in [0,2] if and only if m < -1 -/
theorem quadratic_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ x^2 + (m-1)*x + 1 = 0) ↔ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l2739_273963


namespace NUMINAMATH_CALUDE_fountain_pen_price_l2739_273960

theorem fountain_pen_price (num_fountain_pens : ℕ) (num_mechanical_pencils : ℕ)
  (total_cost : ℚ) (avg_price_mechanical_pencil : ℚ) :
  num_fountain_pens = 450 →
  num_mechanical_pencils = 3750 →
  total_cost = 11250 →
  avg_price_mechanical_pencil = 2.25 →
  (total_cost - (num_mechanical_pencils : ℚ) * avg_price_mechanical_pencil) / (num_fountain_pens : ℚ) = 6.25 := by
sorry

end NUMINAMATH_CALUDE_fountain_pen_price_l2739_273960


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2739_273979

theorem trigonometric_identity (c d : ℝ) (θ : ℝ) 
  (h : (Real.sin θ)^2 / c + (Real.cos θ)^2 / d = 1 / (c + d)) 
  (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : d ≠ 1) :
  (Real.sin θ)^4 / c^2 + (Real.cos θ)^4 / d^2 = 2 * (c - d)^2 / (c^2 * d^2 * (d - 1)^2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2739_273979


namespace NUMINAMATH_CALUDE_percentage_equivalence_l2739_273996

theorem percentage_equivalence (x : ℝ) (h : (30/100) * ((15/100) * x) = 27) :
  (15/100) * ((30/100) * x) = 27 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equivalence_l2739_273996


namespace NUMINAMATH_CALUDE_triangle_inequality_l2739_273944

theorem triangle_inequality (a b c r s : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 →
  s = (a + b + c) / 2 →
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  1 / (s - a)^2 + 1 / (s - b)^2 + 1 / (s - c)^2 ≥ 1 / r^2 := by
sorry


end NUMINAMATH_CALUDE_triangle_inequality_l2739_273944


namespace NUMINAMATH_CALUDE_strawberries_left_l2739_273939

/-- Given 3.5 baskets of strawberries, with 50 strawberries per basket,
    distributed equally among 24 girls, prove that 7 strawberries are left. -/
theorem strawberries_left (baskets : ℚ) (strawberries_per_basket : ℕ) (girls : ℕ) :
  baskets = 3.5 ∧ strawberries_per_basket = 50 ∧ girls = 24 →
  (baskets * strawberries_per_basket : ℚ) - (↑girls * ↑⌊(baskets * strawberries_per_basket) / girls⌋) = 7 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_left_l2739_273939


namespace NUMINAMATH_CALUDE_x_cubed_plus_x_cubed_l2739_273978

theorem x_cubed_plus_x_cubed (x : ℝ) (h : x > 0) : 
  (x^3 + x^3 = 2*x^3) ∧ 
  (x^3 + x^3 ≠ x^6) ∧ 
  (x^3 + x^3 ≠ (3*x)^3) ∧ 
  (x^3 + x^3 ≠ (x^3)^2) :=
by sorry

end NUMINAMATH_CALUDE_x_cubed_plus_x_cubed_l2739_273978


namespace NUMINAMATH_CALUDE_total_profit_is_100_l2739_273993

/-- Calculates the total profit given investments, time periods, and A's share --/
def calculate_total_profit (a_investment : ℕ) (a_months : ℕ) (b_investment : ℕ) (b_months : ℕ) (a_share : ℕ) : ℕ :=
  let a_weight := a_investment * a_months
  let b_weight := b_investment * b_months
  let total_weight := a_weight + b_weight
  let part_value := a_share * total_weight / a_weight
  part_value

theorem total_profit_is_100 :
  calculate_total_profit 150 12 200 6 60 = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_100_l2739_273993


namespace NUMINAMATH_CALUDE_fraction_addition_l2739_273902

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2739_273902


namespace NUMINAMATH_CALUDE_smallest_Y_value_l2739_273983

/-- A function that checks if a positive integer consists only of 0s and 1s -/
def onlyZerosAndOnes (n : ℕ+) : Prop := sorry

/-- The theorem stating the smallest possible value of Y -/
theorem smallest_Y_value (S : ℕ+) (hS : onlyZerosAndOnes S) (hDiv : 18 ∣ S) :
  (S / 18 : ℕ) ≥ 6172839500 :=
sorry

end NUMINAMATH_CALUDE_smallest_Y_value_l2739_273983


namespace NUMINAMATH_CALUDE_parabola_coefficients_l2739_273929

/-- A parabola with vertex (4, -1), vertical axis of symmetry, and passing through (0, -5) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : a ≠ 0 → -b / (2 * a) = 4
  vertex_y : a * 4^2 + b * 4 + c = -1
  symmetry : a ≠ 0 → -b / (2 * a) = 4
  point : a * 0^2 + b * 0 + c = -5

theorem parabola_coefficients (p : Parabola) : p.a = -1/4 ∧ p.b = 2 ∧ p.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l2739_273929


namespace NUMINAMATH_CALUDE_integer_fraction_condition_l2739_273982

theorem integer_fraction_condition (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℚ) / (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1) = k.val) ↔
  (∃ l : ℕ+, (a = l ∧ b = 2 * l) ∨ (a = 8 * l.val ^ 4 - l ∧ b = 2 * l)) :=
by sorry

end NUMINAMATH_CALUDE_integer_fraction_condition_l2739_273982


namespace NUMINAMATH_CALUDE_bike_ride_distance_l2739_273976

theorem bike_ride_distance (first_hour second_hour third_hour : ℝ) : 
  second_hour = first_hour * 1.2 →
  third_hour = second_hour * 1.25 →
  first_hour + second_hour + third_hour = 74 →
  second_hour = 24 := by
sorry

end NUMINAMATH_CALUDE_bike_ride_distance_l2739_273976


namespace NUMINAMATH_CALUDE_conference_center_distance_l2739_273949

theorem conference_center_distance
  (initial_speed : ℝ)
  (initial_distance : ℝ)
  (late_time : ℝ)
  (speed_increase : ℝ)
  (early_time : ℝ)
  (h1 : initial_speed = 40)
  (h2 : initial_distance = 40)
  (h3 : late_time = 1.5)
  (h4 : speed_increase = 20)
  (h5 : early_time = 0.25)
  : ∃ (total_distance : ℝ), total_distance = 310 :=
by
  sorry

end NUMINAMATH_CALUDE_conference_center_distance_l2739_273949


namespace NUMINAMATH_CALUDE_l_shaped_area_is_23_l2739_273952

-- Define the side lengths
def large_square_side : ℝ := 8
def medium_square_side : ℝ := 4
def small_square_side : ℝ := 3

-- Define the areas
def large_square_area : ℝ := large_square_side ^ 2
def medium_square_area : ℝ := medium_square_side ^ 2
def small_square_area : ℝ := small_square_side ^ 2

-- Define the L-shaped area
def l_shaped_area : ℝ := large_square_area - (2 * medium_square_area + small_square_area)

-- Theorem statement
theorem l_shaped_area_is_23 : l_shaped_area = 23 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_is_23_l2739_273952


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l2739_273913

theorem sin_cos_sum_equals_sqrt2_over_2 :
  Real.sin (63 * π / 180) * Real.cos (18 * π / 180) +
  Real.cos (63 * π / 180) * Real.cos (108 * π / 180) =
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt2_over_2_l2739_273913


namespace NUMINAMATH_CALUDE_x_sixth_plus_inverse_l2739_273965

theorem x_sixth_plus_inverse (x : ℝ) (h : x + 1/x = 7) : x^6 + 1/x^6 = 103682 := by
  sorry

end NUMINAMATH_CALUDE_x_sixth_plus_inverse_l2739_273965


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2739_273925

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_fifth_term : a 5 = 10)
  (h_sum_first_three : a 1 + a 2 + a 3 = 3) :
  a 8 = 19 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l2739_273925


namespace NUMINAMATH_CALUDE_cubic_inequality_l2739_273914

theorem cubic_inequality (x : ℝ) : x^3 - 10*x^2 > -25*x ↔ (0 < x ∧ x < 5) ∨ x > 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l2739_273914


namespace NUMINAMATH_CALUDE_lcm_18_24_l2739_273926

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l2739_273926


namespace NUMINAMATH_CALUDE_square_sum_fifteen_l2739_273945

theorem square_sum_fifteen (x y : ℝ) 
  (h1 : y + 4 = (x - 2)^2) 
  (h2 : x + 4 = (y - 2)^2) 
  (h3 : x ≠ y) : 
  x^2 + y^2 = 15 := by
sorry

end NUMINAMATH_CALUDE_square_sum_fifteen_l2739_273945


namespace NUMINAMATH_CALUDE_sum_of_abc_l2739_273927

theorem sum_of_abc (a b c : ℕ+) 
  (h1 : a.val * b.val + c.val = 47)
  (h2 : b.val * c.val + a.val = 47)
  (h3 : a.val * c.val + b.val = 47) :
  a.val + b.val + c.val = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l2739_273927


namespace NUMINAMATH_CALUDE_f_divisible_by_factorial_l2739_273930

def f : ℕ → ℕ → ℕ
  | 0, 0 => 1
  | 0, _ => 0
  | _, 0 => 0
  | n+1, k+1 => (n+1) * (f (n+1) k + f n k)

theorem f_divisible_by_factorial (n k : ℕ) : 
  ∃ m : ℤ, f n k = n! * m := by sorry

end NUMINAMATH_CALUDE_f_divisible_by_factorial_l2739_273930


namespace NUMINAMATH_CALUDE_prob_at_least_9_is_0_7_l2739_273932

/-- A shooter has probabilities of scoring different points in one shot. -/
structure Shooter where
  prob_10 : ℝ  -- Probability of scoring 10 points
  prob_9 : ℝ   -- Probability of scoring 9 points
  prob_8_or_less : ℝ  -- Probability of scoring 8 or fewer points
  sum_to_one : prob_10 + prob_9 + prob_8_or_less = 1  -- Probabilities sum to 1

/-- The probability of scoring at least 9 points is the sum of probabilities of scoring 10 and 9 points. -/
def prob_at_least_9 (s : Shooter) : ℝ := s.prob_10 + s.prob_9

/-- Given the probabilities for a shooter, prove that the probability of scoring at least 9 points is 0.7. -/
theorem prob_at_least_9_is_0_7 (s : Shooter) 
    (h1 : s.prob_10 = 0.4) 
    (h2 : s.prob_9 = 0.3) 
    (h3 : s.prob_8_or_less = 0.3) : 
  prob_at_least_9 s = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_9_is_0_7_l2739_273932


namespace NUMINAMATH_CALUDE_compute_expression_l2739_273957

theorem compute_expression : 9 * (-5) - (7 * -2) + (8 * -6) = -79 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2739_273957


namespace NUMINAMATH_CALUDE_divisibility_relation_l2739_273985

theorem divisibility_relation (p a b n : ℕ) : 
  p ≥ 3 → 
  Nat.Prime p → 
  Nat.Coprime a b → 
  p ∣ (a^(2^n) + b^(2^n)) → 
  2^(n+1) ∣ (p-1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_relation_l2739_273985


namespace NUMINAMATH_CALUDE_sequence_property_l2739_273962

theorem sequence_property (a : ℕ → ℝ) :
  a 2 = 2 ∧ (∀ n : ℕ, n ≥ 2 → a (n + 1) - a n - 1 = 0) →
  ∀ n : ℕ, n ≥ 2 → a n = n :=
by sorry

end NUMINAMATH_CALUDE_sequence_property_l2739_273962


namespace NUMINAMATH_CALUDE_extended_segment_coordinates_l2739_273933

/-- Given points A and B, and a point C on the line extension of AB such that BC = 1/2 * AB, 
    prove that C has the specified coordinates. -/
theorem extended_segment_coordinates (A B C : ℝ × ℝ) : 
  A = (3, -3) → 
  B = (15, 3) → 
  C - B = (1/2 : ℝ) • (B - A) → 
  C = (21, 6) := by
sorry

end NUMINAMATH_CALUDE_extended_segment_coordinates_l2739_273933


namespace NUMINAMATH_CALUDE_total_interest_calculation_l2739_273934

/-- Calculate total interest over 10 years with principal trebling after 5 years -/
theorem total_interest_calculation (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 600) : 
  P * R * 5 / 100 + 3 * P * R * 5 / 100 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_calculation_l2739_273934


namespace NUMINAMATH_CALUDE_unique_solution_2011_l2739_273907

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The main theorem -/
theorem unique_solution_2011 :
  ∃! n : ℕ, n + sum_of_digits n = 2011 ∧ n = 1991 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_2011_l2739_273907


namespace NUMINAMATH_CALUDE_second_smallest_odd_number_l2739_273948

/-- Given a sequence of four consecutive odd numbers whose sum is 112,
    the second smallest number in this sequence is 27. -/
theorem second_smallest_odd_number : ∀ (a b c d : ℤ),
  (∃ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5 ∧ d = 2*n + 7) →  -- consecutive odd numbers
  (a + b + c + d = 112) →                                            -- sum is 112
  b = 27                                                             -- second smallest is 27
:= by sorry

end NUMINAMATH_CALUDE_second_smallest_odd_number_l2739_273948


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2739_273900

theorem weight_of_new_person (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 66 →
  avg_increase = 2.5 →
  ∃ (new_weight : ℝ), new_weight = replaced_weight + initial_count * avg_increase :=
by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l2739_273900


namespace NUMINAMATH_CALUDE_billy_reads_three_books_l2739_273919

/-- Represents Billy's reading activity over the weekend --/
structure BillyReading where
  initial_speed : ℝ  -- Initial reading speed in pages per hour
  time_available : ℝ  -- Total time available for reading in hours
  book_pages : ℕ  -- Number of pages in each book
  speed_decrease : ℝ  -- Percentage decrease in reading speed after each book

/-- Calculates the number of books Billy can read --/
def books_read (b : BillyReading) : ℕ :=
  sorry

/-- Theorem stating that Billy can read exactly 3 books --/
theorem billy_reads_three_books :
  let b : BillyReading := {
    initial_speed := 60,
    time_available := 16 * 0.35,
    book_pages := 80,
    speed_decrease := 0.1
  }
  books_read b = 3 := by sorry

end NUMINAMATH_CALUDE_billy_reads_three_books_l2739_273919


namespace NUMINAMATH_CALUDE_cost_price_calculation_l2739_273924

theorem cost_price_calculation (markup_percentage : ℝ) (discount_percentage : ℝ) (profit : ℝ) : 
  markup_percentage = 0.2 →
  discount_percentage = 0.1 →
  profit = 40 →
  ∃ (cost_price : ℝ), 
    cost_price * (1 + markup_percentage) * (1 - discount_percentage) - cost_price = profit ∧
    cost_price = 500 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l2739_273924


namespace NUMINAMATH_CALUDE_gcd_45123_32768_l2739_273911

theorem gcd_45123_32768 : Nat.gcd 45123 32768 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45123_32768_l2739_273911


namespace NUMINAMATH_CALUDE_ellipse_b_value_l2739_273969

/-- Define an ellipse with foci F1 and F2, and a point P on the ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  F1 : ℝ × ℝ
  F2 : ℝ × ℝ
  P : ℝ × ℝ
  h1 : a > b
  h2 : b > 0
  h3 : P.1^2 / a^2 + P.2^2 / b^2 = 1  -- P is on the ellipse

/-- The dot product of PF1 and PF2 is zero -/
def orthogonal_foci (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0

/-- The area of triangle PF1F2 is 9 -/
def triangle_area (e : Ellipse) : Prop :=
  let PF1 := (e.F1.1 - e.P.1, e.F1.2 - e.P.2)
  let PF2 := (e.F2.1 - e.P.1, e.F2.2 - e.P.2)
  abs (PF1.1 * PF2.2 - PF1.2 * PF2.1) / 2 = 9

/-- Main theorem: If the foci are orthogonal from P and the triangle area is 9, then b = 3 -/
theorem ellipse_b_value (e : Ellipse) 
  (h_orth : orthogonal_foci e) (h_area : triangle_area e) : e.b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_b_value_l2739_273969


namespace NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l2739_273908

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection :
  let line : ℝ → ℝ → Prop := λ x y ↦ 2 * x - y + 1 = 0
  let circle : ℝ → ℝ → Prop := λ x y ↦ (x - 1)^2 + (y - 1)^2 = 1
  ∃ (A B : ℝ × ℝ),
    line A.1 A.2 ∧ line B.1 B.2 ∧
    circle A.1 A.2 ∧ circle B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l2739_273908


namespace NUMINAMATH_CALUDE_trigonometric_identity_proof_l2739_273972

theorem trigonometric_identity_proof (x : ℝ) : 
  Real.sin (x + Real.pi / 3) + 2 * Real.sin (x - Real.pi / 3) - Real.sqrt 3 * Real.cos (2 * Real.pi / 3 - x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_proof_l2739_273972


namespace NUMINAMATH_CALUDE_number_puzzle_l2739_273984

theorem number_puzzle : ∃ N : ℚ, N = 90 ∧ 3 + (1/2) * (1/3) * (1/5) * N = (1/15) * N := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2739_273984


namespace NUMINAMATH_CALUDE_cos_equation_solutions_l2739_273959

theorem cos_equation_solutions :
  ∃! (S : Finset ℝ), 
    (∀ x ∈ S, x ∈ Set.Icc 0 Real.pi ∧ Real.cos (7 * x) = Real.cos (5 * x)) ∧
    S.card = 7 :=
sorry

end NUMINAMATH_CALUDE_cos_equation_solutions_l2739_273959


namespace NUMINAMATH_CALUDE_max_table_sum_l2739_273917

def numbers : List ℕ := [3, 5, 7, 11, 17, 19]

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ 
  d ∈ numbers ∧ e ∈ numbers ∧ f ∈ numbers ∧
  (a = b ∧ b = c) ∨ (d = e ∧ e = f)

def table_sum (a b c d e f : ℕ) : ℕ :=
  a*d + a*e + a*f + b*d + b*e + b*f + c*d + c*e + c*f

theorem max_table_sum :
  ∀ a b c d e f : ℕ,
    is_valid_arrangement a b c d e f →
    table_sum a b c d e f ≤ 1995 ∧
    (∃ a b c d e f : ℕ, 
      is_valid_arrangement a b c d e f ∧ 
      table_sum a b c d e f = 1995 ∧
      (a = 19 ∧ b = 19 ∧ c = 19) ∨ (d = 19 ∧ e = 19 ∧ f = 19)) := by
  sorry

end NUMINAMATH_CALUDE_max_table_sum_l2739_273917


namespace NUMINAMATH_CALUDE_consecutive_sum_39_l2739_273915

theorem consecutive_sum_39 (n m : ℕ) : 
  m = n + 1 → n + m = 39 → m = 20 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_39_l2739_273915


namespace NUMINAMATH_CALUDE_ron_book_picks_l2739_273923

/-- Represents a book club with its properties --/
structure BookClub where
  members : ℕ
  weekly_meetings : ℕ
  holiday_breaks : ℕ
  guest_picks : ℕ
  leap_year_extra_meeting : ℕ

/-- Calculates the number of times a member gets to pick a book --/
def picks_per_member (club : BookClub) (is_leap_year : Bool) : ℕ :=
  let total_meetings := club.weekly_meetings - club.holiday_breaks + (if is_leap_year then club.leap_year_extra_meeting else 0)
  let member_picks := total_meetings - club.guest_picks - (if is_leap_year then 1 else 0)
  member_picks / club.members

/-- Theorem stating that Ron gets to pick 3 books in both leap and non-leap years --/
theorem ron_book_picks (club : BookClub) 
    (h1 : club.members = 13)
    (h2 : club.weekly_meetings = 52)
    (h3 : club.holiday_breaks = 5)
    (h4 : club.guest_picks = 6)
    (h5 : club.leap_year_extra_meeting = 1) : 
    picks_per_member club false = 3 ∧ picks_per_member club true = 3 := by
  sorry

end NUMINAMATH_CALUDE_ron_book_picks_l2739_273923


namespace NUMINAMATH_CALUDE_solve_for_x_l2739_273943

theorem solve_for_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2739_273943


namespace NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l2739_273955

theorem angle_with_special_supplement_complement : ∃ (x : ℝ), 
  0 < x ∧ x < 90 ∧ 
  (180 - x) = 4 * (90 - x) + 15 ∧
  x = 65 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_special_supplement_complement_l2739_273955


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2739_273977

theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  a = (2, 0) → 
  ‖b‖ = 1 → 
  a • b = 0 → 
  ‖a + 2 • b‖ = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2739_273977


namespace NUMINAMATH_CALUDE_xy_value_l2739_273970

theorem xy_value (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = 35/12 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2739_273970


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l2739_273931

theorem fraction_zero_solution (x : ℝ) : 
  (x + 2) / (2 * x - 4) = 0 ↔ x = -2 ∧ 2 * x - 4 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_solution_l2739_273931


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_divisible_by_three_l2739_273942

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

theorem sum_of_three_consecutive_odd_divisible_by_three : 
  ∀ (a b c : ℕ), 
    (is_odd a ∧ is_odd b ∧ is_odd c) → 
    (a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0) →
    (∃ k, b = a + 6*k + 6 ∧ c = b + 6) →
    (c = 27) →
    (a + b + c = 63) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_divisible_by_three_l2739_273942


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_three_even_three_odd_count_six_digit_numbers_with_three_even_three_odd_l2739_273912

theorem six_digit_numbers_with_three_even_three_odd : ℕ :=
  let first_digit_choices := 9
  let position_choices := Nat.choose 5 2
  let same_parity_fill := 5^2
  let opposite_parity_fill := 2^3
  first_digit_choices * position_choices * same_parity_fill * opposite_parity_fill

theorem count_six_digit_numbers_with_three_even_three_odd :
  six_digit_numbers_with_three_even_three_odd = 90000 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_three_even_three_odd_count_six_digit_numbers_with_three_even_three_odd_l2739_273912


namespace NUMINAMATH_CALUDE_angle_b_is_sixty_degrees_triangle_is_equilateral_l2739_273987

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Add necessary conditions
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  area_formula : S = (1/2) * a * c * Real.sin B

-- Theorem 1
theorem angle_b_is_sixty_degrees (t : Triangle) 
  (h : t.a^2 + t.c^2 = t.b^2 + t.a * t.c) : 
  t.B = π/3 := by sorry

-- Theorem 2
theorem triangle_is_equilateral (t : Triangle)
  (h1 : t.a^2 + t.c^2 = t.b^2 + t.a * t.c)
  (h2 : t.b = 2)
  (h3 : t.S = Real.sqrt 3) :
  t.a = t.b ∧ t.b = t.c := by sorry

end NUMINAMATH_CALUDE_angle_b_is_sixty_degrees_triangle_is_equilateral_l2739_273987


namespace NUMINAMATH_CALUDE_star_calculation_l2739_273940

-- Define the ☆ operation
def star (a b : ℚ) : ℚ := a - b + 1

-- Theorem to prove
theorem star_calculation : (star (star 2 3) 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l2739_273940


namespace NUMINAMATH_CALUDE_group_earnings_l2739_273988

/-- Represents the wage of a man in rupees -/
def man_wage : ℕ := 6

/-- Represents the number of men in the group -/
def num_men : ℕ := 5

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 8

/-- Represents the number of women in the group (unknown) -/
def num_women : ℕ := sorry

/-- The total amount earned by the group -/
def total_amount : ℕ := 3 * (num_men * man_wage)

theorem group_earnings : 
  total_amount = 90 := by sorry

end NUMINAMATH_CALUDE_group_earnings_l2739_273988


namespace NUMINAMATH_CALUDE_problem_statement_l2739_273916

theorem problem_statement (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : 1 / a^3 = 512 / b^3 ∧ 1 / a^3 = 125 / c^3 ∧ 1 / a^3 = d / (a + b + c)^3) : 
  d = 2744 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2739_273916


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2739_273991

theorem smallest_positive_integer_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 2) ∧ 
  (n % 6 = 3) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 6 = 3 → m ≥ n) ∧
  n = 21 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l2739_273991


namespace NUMINAMATH_CALUDE_largest_movable_n_l2739_273968

/-- Represents the rules for moving cards between boxes -/
structure CardMoveRules where
  /-- A card can be placed in an empty box -/
  place_in_empty : Bool
  /-- A card can be placed on top of a card with a number one greater than its own -/
  place_on_greater : Bool

/-- Represents the configuration of card boxes -/
structure BoxConfiguration where
  /-- Number of blue boxes -/
  k : Nat
  /-- Number of cards (2n) -/
  card_count : Nat
  /-- Rules for moving cards -/
  move_rules : CardMoveRules

/-- Determines if all cards can be moved to blue boxes given a configuration -/
def can_move_all_cards (config : BoxConfiguration) : Prop :=
  ∃ (final_state : List (List Nat)), 
    final_state.length = config.k ∧ 
    final_state.all (λ box => box.length > 0) ∧
    final_state.join.toFinset = Finset.range config.card_count

/-- The main theorem stating the largest possible n for which all cards can be moved -/
theorem largest_movable_n (k : Nat) (h : k > 1) :
  ∀ n : Nat, (
    let config := BoxConfiguration.mk k (2 * n) 
      { place_in_empty := true, place_on_greater := true }
    can_move_all_cards config ↔ n ≤ k - 1
  ) := by sorry

end NUMINAMATH_CALUDE_largest_movable_n_l2739_273968


namespace NUMINAMATH_CALUDE_rabbits_in_park_l2739_273967

theorem rabbits_in_park (cage_rabbits : ℕ) (park_rabbits : ℕ) : 
  cage_rabbits = 13 →
  cage_rabbits + 7 = park_rabbits / 3 →
  park_rabbits = 60 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_in_park_l2739_273967


namespace NUMINAMATH_CALUDE_smallest_AAB_value_l2739_273904

/-- Represents a digit (1 to 9) -/
def Digit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Represents a two-digit number AB -/
def TwoDigitNumber (A B : Digit) : ℕ := 10 * A.val + B.val

/-- Represents a three-digit number AAB -/
def ThreeDigitNumber (A B : Digit) : ℕ := 100 * A.val + 10 * A.val + B.val

/-- The main theorem -/
theorem smallest_AAB_value :
  ∀ (A B : Digit),
    A ≠ B →
    TwoDigitNumber A B = (ThreeDigitNumber A B) / 7 →
    ∀ (A' B' : Digit),
      A' ≠ B' →
      TwoDigitNumber A' B' = (ThreeDigitNumber A' B') / 7 →
      ThreeDigitNumber A B ≤ ThreeDigitNumber A' B' →
      ThreeDigitNumber A B = 664 :=
sorry

end NUMINAMATH_CALUDE_smallest_AAB_value_l2739_273904


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2739_273999

theorem quadratic_inequality_empty_solution_set (m : ℝ) :
  (∀ x : ℝ, (m + 1) * x^2 - m * x + m ≥ 0) ↔ m ≥ 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_set_l2739_273999


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2739_273903

theorem condition_sufficient_not_necessary :
  (∀ k : ℤ, Real.sin (2 * k * Real.pi + Real.pi / 4) = Real.sqrt 2 / 2) ∧
  (∃ x : ℝ, Real.sin x = Real.sqrt 2 / 2 ∧ ∀ k : ℤ, x ≠ 2 * k * Real.pi + Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2739_273903


namespace NUMINAMATH_CALUDE_multiplication_table_even_fraction_l2739_273986

/-- The size of the multiplication table (16 in this case) -/
def table_size : ℕ := 16

/-- A number is even if it's divisible by 2 -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- The count of even numbers in the range [0, table_size - 1] -/
def even_count : ℕ := (table_size + 1) / 2

/-- The count of odd numbers in the range [0, table_size - 1] -/
def odd_count : ℕ := table_size - even_count

/-- The total number of entries in the multiplication table -/
def total_entries : ℕ := table_size * table_size

/-- The number of entries where both factors are odd -/
def odd_entries : ℕ := odd_count * odd_count

/-- The number of entries where at least one factor is even -/
def even_entries : ℕ := total_entries - odd_entries

/-- The fraction of even entries in the multiplication table -/
def even_fraction : ℚ := even_entries / total_entries

theorem multiplication_table_even_fraction :
  even_fraction = 3/4 := by sorry

end NUMINAMATH_CALUDE_multiplication_table_even_fraction_l2739_273986


namespace NUMINAMATH_CALUDE_bathtub_guests_l2739_273956

/-- Proves that given a bathtub with 10 liters capacity, after 3 guests use 1.5 liters each
    and 1 guest uses 1.75 liters, the remaining water can be used by exactly 3 more guests
    if each uses 1.25 liters. -/
theorem bathtub_guests (bathtub_capacity : ℝ) (guests_1 : ℕ) (water_1 : ℝ)
                        (guests_2 : ℕ) (water_2 : ℝ) (water_per_remaining_guest : ℝ) :
  bathtub_capacity = 10 →
  guests_1 = 3 →
  water_1 = 1.5 →
  guests_2 = 1 →
  water_2 = 1.75 →
  water_per_remaining_guest = 1.25 →
  (bathtub_capacity - (guests_1 * water_1 + guests_2 * water_2)) / water_per_remaining_guest = 3 :=
by sorry

end NUMINAMATH_CALUDE_bathtub_guests_l2739_273956


namespace NUMINAMATH_CALUDE_total_cans_l2739_273974

def bag1 : ℕ := 5
def bag2 : ℕ := 7
def bag3 : ℕ := 12
def bag4 : ℕ := 4
def bag5 : ℕ := 8
def bag6 : ℕ := 10

theorem total_cans : bag1 + bag2 + bag3 + bag4 + bag5 + bag6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_l2739_273974


namespace NUMINAMATH_CALUDE_package_weight_sum_l2739_273935

theorem package_weight_sum (x y z : ℝ) 
  (h1 : x + y = 112)
  (h2 : y + z = 118)
  (h3 : z + x = 120) :
  x + y + z = 175 := by
sorry

end NUMINAMATH_CALUDE_package_weight_sum_l2739_273935


namespace NUMINAMATH_CALUDE_linear_system_solution_l2739_273973

/-- Given a system of linear equations 2x + my = 5 and nx - 3y = 2,
    if the augmented matrix transforms to [[1, 0, 3], [0, 1, 1]],
    then m/n = -3/5 -/
theorem linear_system_solution (m n : ℚ) : 
  (∃ x y : ℚ, 2*x + m*y = 5 ∧ n*x - 3*y = 2) →
  (∃ x y : ℚ, x = 3 ∧ y = 1) →
  m/n = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l2739_273973


namespace NUMINAMATH_CALUDE_semicircle_perimeter_specific_semicircle_perimeter_l2739_273990

/-- The perimeter of a semi-circle with radius r is equal to π * r + 2 * r -/
theorem semicircle_perimeter (r : ℝ) (h : r > 0) :
  let perimeter := π * r + 2 * r
  perimeter = π * r + 2 * r :=
by sorry

/-- The perimeter of a semi-circle with radius 6.7 cm is approximately 34.45 cm -/
theorem specific_semicircle_perimeter :
  let r : ℝ := 6.7
  let perimeter := π * r + 2 * r
  ∃ ε > 0, |perimeter - 34.45| < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_specific_semicircle_perimeter_l2739_273990


namespace NUMINAMATH_CALUDE_circle_and_line_theorem_l2739_273975

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line on which the center of C lies
def CenterLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 - p.2 - 2 = 0}

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)

-- Define point P
def P : ℝ × ℝ := (-3, 3)

-- Define the x-axis
def XAxis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

-- Define a line given its equation ax + by + c = 0
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

theorem circle_and_line_theorem :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center ∈ CenterLine ∧
    A ∈ Circle center radius ∧
    B ∈ Circle center radius ∧
    (∃ (a b c : ℝ),
      (Line a b c = {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 3 = 0} ∨
       Line a b c = {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 + 3 = 0}) ∧
      P ∈ Line a b c ∧
      (∃ (q : ℝ × ℝ), q ∈ XAxis ∧ q ∈ Line a b c) ∧
      (∃ (t : ℝ × ℝ), t ∈ Circle center radius ∧ t ∈ Line a b c)) ∧
    center = (2, 2) ∧
    radius = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_theorem_l2739_273975


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2739_273958

theorem decimal_to_fraction : 
  (1.45 : ℚ) = 29 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2739_273958


namespace NUMINAMATH_CALUDE_polynomial_roots_magnitude_l2739_273953

theorem polynomial_roots_magnitude (c : ℂ) : 
  (∃ (Q : ℂ → ℂ), 
    Q = (fun x => (x^2 - 3*x + 3) * (x^2 - c*x + 9) * (x^2 - 5*x + 15)) ∧
    (∃ (r1 r2 r3 : ℂ), 
      r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧
      (∀ x : ℂ, Q x = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3))) →
  Complex.abs c = 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_magnitude_l2739_273953


namespace NUMINAMATH_CALUDE_baking_time_proof_l2739_273947

/-- Alice's pie-baking time in minutes -/
def alice_time : ℕ := 5

/-- Bob's pie-baking time in minutes -/
def bob_time : ℕ := 6

/-- The time period in which Alice bakes 2 more pies than Bob -/
def time_period : ℕ := 60

theorem baking_time_proof :
  (time_period / alice_time : ℚ) = (time_period / bob_time : ℚ) + 2 :=
by sorry

end NUMINAMATH_CALUDE_baking_time_proof_l2739_273947


namespace NUMINAMATH_CALUDE_billy_brad_weight_difference_l2739_273910

-- Define the weights as natural numbers
def carl_weight : ℕ := 145
def billy_weight : ℕ := 159

-- Define Brad's weight in terms of Carl's
def brad_weight : ℕ := carl_weight + 5

-- State the theorem
theorem billy_brad_weight_difference :
  billy_weight - brad_weight = 9 :=
by sorry

end NUMINAMATH_CALUDE_billy_brad_weight_difference_l2739_273910


namespace NUMINAMATH_CALUDE_ned_bomb_diffusion_l2739_273909

/-- Represents the problem of Ned racing to deactivate a time bomb --/
def bomb_diffusion_problem (total_flights : ℕ) (seconds_per_flight : ℕ) (bomb_timer : ℕ) (time_spent : ℕ) : Prop :=
  let total_time := total_flights * seconds_per_flight
  let remaining_time := total_time - time_spent
  let time_left := bomb_timer - remaining_time
  time_left = 84

/-- Theorem stating that Ned will have 84 seconds to diffuse the bomb --/
theorem ned_bomb_diffusion :
  bomb_diffusion_problem 40 13 58 273 := by
  sorry

#check ned_bomb_diffusion

end NUMINAMATH_CALUDE_ned_bomb_diffusion_l2739_273909


namespace NUMINAMATH_CALUDE_expression_evaluation_l2739_273951

theorem expression_evaluation :
  (5^500 + 6^501)^2 - (5^500 - 6^501)^2 = 24 * 30^500 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2739_273951


namespace NUMINAMATH_CALUDE_batch_size_proof_l2739_273995

/-- The number of parts in the batch -/
def total_parts : ℕ := 1150

/-- The fraction of work A completes when cooperating with the master -/
def a_work_fraction : ℚ := 1/5

/-- The fraction of work B completes when cooperating with the master -/
def b_work_fraction : ℚ := 2/5

/-- The number of fewer parts B processes when A joins -/
def b_fewer_parts : ℕ := 60

theorem batch_size_proof :
  (b_work_fraction * total_parts : ℚ) - 
  ((1 - a_work_fraction - b_work_fraction) / 
   (1 + (1 - a_work_fraction - b_work_fraction) / a_work_fraction) * total_parts : ℚ) = 
  b_fewer_parts := by sorry

end NUMINAMATH_CALUDE_batch_size_proof_l2739_273995


namespace NUMINAMATH_CALUDE_minimum_balls_to_draw_thirty_eight_sufficient_l2739_273922

/-- Represents the number of balls of each color in the bag -/
structure BagContents :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (other : ℕ)

/-- Represents the configuration of drawn balls -/
structure DrawnBalls :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (other : ℕ)

/-- Check if a given configuration of drawn balls satisfies the condition -/
def satisfiesCondition (drawn : DrawnBalls) : Prop :=
  drawn.red ≥ 10 ∨ drawn.blue ≥ 10 ∨ drawn.yellow ≥ 10

/-- Check if it's possible to draw a given configuration from the bag -/
def canDraw (bag : BagContents) (drawn : DrawnBalls) : Prop :=
  drawn.red ≤ bag.red ∧
  drawn.blue ≤ bag.blue ∧
  drawn.yellow ≤ bag.yellow ∧
  drawn.other ≤ bag.other ∧
  drawn.red + drawn.blue + drawn.yellow + drawn.other ≤ bag.red + bag.blue + bag.yellow + bag.other

theorem minimum_balls_to_draw (bag : BagContents)
  (h1 : bag.red = 20)
  (h2 : bag.blue = 20)
  (h3 : bag.yellow = 20)
  (h4 : bag.other = 10) :
  ∀ n : ℕ, n < 38 →
    ∃ drawn : DrawnBalls, canDraw bag drawn ∧ ¬satisfiesCondition drawn ∧ drawn.red + drawn.blue + drawn.yellow + drawn.other = n :=
by sorry

theorem thirty_eight_sufficient (bag : BagContents)
  (h1 : bag.red = 20)
  (h2 : bag.blue = 20)
  (h3 : bag.yellow = 20)
  (h4 : bag.other = 10) :
  ∀ drawn : DrawnBalls, canDraw bag drawn → drawn.red + drawn.blue + drawn.yellow + drawn.other = 38 →
    satisfiesCondition drawn :=
by sorry

end NUMINAMATH_CALUDE_minimum_balls_to_draw_thirty_eight_sufficient_l2739_273922


namespace NUMINAMATH_CALUDE_joseph_cards_l2739_273998

/-- Calculates the total number of cards Joseph had initially -/
def total_cards (num_students : ℕ) (cards_per_student : ℕ) (cards_left : ℕ) : ℕ :=
  num_students * cards_per_student + cards_left

/-- Proves that Joseph had 357 cards initially -/
theorem joseph_cards : total_cards 15 23 12 = 357 := by
  sorry

end NUMINAMATH_CALUDE_joseph_cards_l2739_273998


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_product_l2739_273981

theorem pure_imaginary_complex_product (a : ℝ) :
  let z : ℂ := (1 - 2*I) * (a - I) * I
  (z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_product_l2739_273981


namespace NUMINAMATH_CALUDE_polygon_with_120_degree_angles_is_hexagon_l2739_273921

theorem polygon_with_120_degree_angles_is_hexagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 120 →
    (n - 2) * 180 = n * interior_angle →
    n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_120_degree_angles_is_hexagon_l2739_273921
