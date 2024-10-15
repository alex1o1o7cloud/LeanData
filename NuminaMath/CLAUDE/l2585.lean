import Mathlib

namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l2585_258572

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 36 = 0 ∧ x ≠ 4 → x = 9 := by sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l2585_258572


namespace NUMINAMATH_CALUDE_c_minus_a_equals_40_l2585_258568

theorem c_minus_a_equals_40
  (a b c d e : ℝ)
  (h1 : (a + b) / 2 = 40)
  (h2 : (b + c) / 2 = 60)
  (h3 : (d + e) / 2 = 80)
  (h4 : (a * b * d) = (b * c * e)) :
  c - a = 40 := by
  sorry

end NUMINAMATH_CALUDE_c_minus_a_equals_40_l2585_258568


namespace NUMINAMATH_CALUDE_average_of_list_l2585_258577

def number_list : List Nat := [55, 48, 507, 2, 684, 42]

theorem average_of_list (list : List Nat) : 
  (list.sum / list.length : ℚ) = 223 :=
by sorry

end NUMINAMATH_CALUDE_average_of_list_l2585_258577


namespace NUMINAMATH_CALUDE_blender_price_difference_l2585_258551

def in_store_price : ℚ := 75.99
def tv_payment : ℚ := 17.99
def shipping_fee : ℚ := 6.50
def handling_charge : ℚ := 2.50

theorem blender_price_difference :
  (4 * tv_payment + shipping_fee + handling_charge - in_store_price) * 100 = 497 := by
  sorry

end NUMINAMATH_CALUDE_blender_price_difference_l2585_258551


namespace NUMINAMATH_CALUDE_circle_equation_l2585_258500

/-- Given a circle with center (0, -2) and a chord intercepted by the line 2x - y + 3 = 0
    with length 4√5, prove that the equation of the circle is x² + (y+2)² = 25. -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (0, -2)
  let chord_line (x y : ℝ) := 2 * x - y + 3 = 0
  let chord_length : ℝ := 4 * Real.sqrt 5
  ∃ (r : ℝ), r > 0 ∧
    (∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 ↔
      x^2 + (y + 2)^2 = 25) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_equation_l2585_258500


namespace NUMINAMATH_CALUDE_total_time_to_school_l2585_258567

def time_to_gate : ℕ := 15
def time_gate_to_building : ℕ := 6
def time_building_to_room : ℕ := 9

theorem total_time_to_school :
  time_to_gate + time_gate_to_building + time_building_to_room = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_time_to_school_l2585_258567


namespace NUMINAMATH_CALUDE_democrat_ratio_l2585_258566

theorem democrat_ratio (total_participants : ℕ) 
  (female_participants male_participants : ℕ)
  (female_democrats male_democrats : ℕ) :
  total_participants = 720 →
  female_participants + male_participants = total_participants →
  female_democrats = female_participants / 2 →
  male_democrats = male_participants / 4 →
  female_democrats = 120 →
  (female_democrats + male_democrats) * 3 = total_participants :=
by sorry

end NUMINAMATH_CALUDE_democrat_ratio_l2585_258566


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2585_258583

theorem rationalize_denominator :
  ∃ (A B C D E F G H I : ℤ),
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
      (A * Real.sqrt B + C * Real.sqrt D + E * Real.sqrt F + G * Real.sqrt H) / I ∧
    I > 0 ∧
    A = 3 ∧ B = 3 ∧ C = 9 ∧ D = 5 ∧ E = -9 ∧ F = 11 ∧ G = 6 ∧ H = 33 ∧ I = 51 :=
by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2585_258583


namespace NUMINAMATH_CALUDE_equivalence_of_statements_l2585_258557

variable (P Q : Prop)

theorem equivalence_of_statements :
  (P → Q) ↔ (¬Q → ¬P) ∧ (¬P ∨ Q) :=
sorry

end NUMINAMATH_CALUDE_equivalence_of_statements_l2585_258557


namespace NUMINAMATH_CALUDE_min_value_of_function_l2585_258507

theorem min_value_of_function (x : ℝ) (h : x > 2) :
  (x^2 - 4*x + 8) / (x - 2) ≥ 4 ∧ ∃ y > 2, (y^2 - 4*y + 8) / (y - 2) = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2585_258507


namespace NUMINAMATH_CALUDE_sum_of_squared_distances_bounded_l2585_258562

/-- A point on the perimeter of a unit square -/
structure PerimeterPoint where
  x : Real
  y : Real
  on_perimeter : (x = 0 ∨ x = 1 ∨ y = 0 ∨ y = 1) ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1

/-- Four points on the perimeter of a unit square, in order -/
structure FourPoints where
  A : PerimeterPoint
  B : PerimeterPoint
  C : PerimeterPoint
  D : PerimeterPoint
  in_order : (A.x ≤ B.x ∧ A.y ≥ B.y) ∧ (B.x ≤ C.x ∧ B.y ≤ C.y) ∧ (C.x ≥ D.x ∧ C.y ≤ D.y) ∧ (D.x ≤ A.x ∧ D.y ≤ A.y)
  each_side_has_point : (A.x = 0 ∨ B.x = 0 ∨ C.x = 0 ∨ D.x = 0) ∧
                        (A.x = 1 ∨ B.x = 1 ∨ C.x = 1 ∨ D.x = 1) ∧
                        (A.y = 0 ∨ B.y = 0 ∨ C.y = 0 ∨ D.y = 0) ∧
                        (A.y = 1 ∨ B.y = 1 ∨ C.y = 1 ∨ D.y = 1)

/-- The squared distance between two points -/
def squared_distance (p1 p2 : PerimeterPoint) : Real :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- The theorem to be proved -/
theorem sum_of_squared_distances_bounded (points : FourPoints) :
  2 ≤ squared_distance points.A points.B +
      squared_distance points.B points.C +
      squared_distance points.C points.D +
      squared_distance points.D points.A
  ∧
  squared_distance points.A points.B +
  squared_distance points.B points.C +
  squared_distance points.C points.D +
  squared_distance points.D points.A ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_distances_bounded_l2585_258562


namespace NUMINAMATH_CALUDE_different_color_pairs_count_l2585_258516

/- Given a drawer with distinguishable socks: -/
def white_socks : ℕ := 6
def brown_socks : ℕ := 5
def blue_socks : ℕ := 4

/- Define the function to calculate the number of ways to choose two socks of different colors -/
def different_color_pairs : ℕ :=
  white_socks * brown_socks +
  brown_socks * blue_socks +
  white_socks * blue_socks

/- The theorem to prove -/
theorem different_color_pairs_count : different_color_pairs = 74 := by
  sorry

end NUMINAMATH_CALUDE_different_color_pairs_count_l2585_258516


namespace NUMINAMATH_CALUDE_sixtieth_pair_l2585_258501

/-- Definition of the sequence of integer pairs -/
def sequence_pair : ℕ → ℕ × ℕ
| 0 => (1, 1)
| n + 1 => 
  let (a, b) := sequence_pair n
  if a = 1 then (b + 1, 1) else (a - 1, b + 1)

/-- The 60th pair in the sequence is (5, 7) -/
theorem sixtieth_pair : sequence_pair 59 = (5, 7) := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_pair_l2585_258501


namespace NUMINAMATH_CALUDE_max_quarters_is_19_l2585_258513

/-- Represents the number of each coin type in the piggy bank -/
structure CoinCount where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Checks if the given coin count satisfies the problem conditions -/
def isValidCoinCount (c : CoinCount) : Prop :=
  c.nickels > 0 ∧ c.dimes > 0 ∧ c.quarters > 0 ∧
  c.nickels + c.dimes + c.quarters = 120 ∧
  5 * c.nickels + 10 * c.dimes + 25 * c.quarters = 1000

/-- Theorem stating that 19 is the maximum number of quarters possible -/
theorem max_quarters_is_19 :
  ∀ c : CoinCount, isValidCoinCount c → c.quarters ≤ 19 :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_is_19_l2585_258513


namespace NUMINAMATH_CALUDE_intersection_M_N_l2585_258588

def M : Set ℝ := { x | x^2 ≥ 1 }
def N : Set ℝ := { y | ∃ x, y = 3*x^2 + 1 }

theorem intersection_M_N : M ∩ N = { x | x ≥ 1 ∨ x ≤ -1 } := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2585_258588


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l2585_258504

theorem tax_free_items_cost 
  (total_paid : ℝ) 
  (sales_tax : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_paid = 40) 
  (h2 : sales_tax = 1.28) 
  (h3 : tax_rate = 0.08) : 
  total_paid - (sales_tax / tax_rate + sales_tax) = 22.72 := by
sorry

end NUMINAMATH_CALUDE_tax_free_items_cost_l2585_258504


namespace NUMINAMATH_CALUDE_total_counts_for_week_l2585_258542

/-- Represents the number of times Carla counts each item on a given day -/
structure DailyCounts where
  tiles : Nat
  books : Nat
  chairs : Nat

/-- The week's counting activities -/
def week : List DailyCounts := [
  ⟨1, 1, 0⟩,  -- Monday
  ⟨2, 3, 0⟩,  -- Tuesday
  ⟨0, 0, 4⟩,  -- Wednesday
  ⟨3, 0, 2⟩,  -- Thursday
  ⟨1, 2, 3⟩   -- Friday
]

/-- Calculates the total number of counts for a day -/
def totalCountsForDay (day : DailyCounts) : Nat :=
  day.tiles + day.books + day.chairs

/-- Theorem stating that the total number of counts for the week is 22 -/
theorem total_counts_for_week : (week.map totalCountsForDay).sum = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_counts_for_week_l2585_258542


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l2585_258555

theorem quadratic_complete_square (x : ℝ) : ∃ (p q : ℝ), 
  (4 * x^2 + 8 * x - 448 = 0) ↔ ((x + p)^2 = q) ∧ q = 113 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l2585_258555


namespace NUMINAMATH_CALUDE_similar_right_triangle_longest_side_l2585_258571

theorem similar_right_triangle_longest_side
  (a b c : ℝ)
  (h_right : a^2 + b^2 = c^2)
  (h_sides : a = 8 ∧ b = 15 ∧ c = 17)
  (k : ℝ)
  (h_perimeter : k * (a + b + c) = 160)
  : k * c = 68 :=
by sorry

end NUMINAMATH_CALUDE_similar_right_triangle_longest_side_l2585_258571


namespace NUMINAMATH_CALUDE_find_number_l2585_258536

theorem find_number (a b : ℕ+) (hcf : Nat.gcd a b = 12) (lcm : Nat.lcm a b = 396) (hb : b = 198) : a = 24 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2585_258536


namespace NUMINAMATH_CALUDE_count_eight_digit_numbers_product_7000_l2585_258565

/-- The number of eight-digit numbers whose digits multiply to 7000 -/
def eight_digit_numbers_with_product_7000 : ℕ := 5600

/-- The prime factorization of 7000 -/
def prime_factorization_7000 : List ℕ := [7, 2, 2, 2, 5, 5, 5]

theorem count_eight_digit_numbers_product_7000 :
  eight_digit_numbers_with_product_7000 = 5600 := by
  sorry

end NUMINAMATH_CALUDE_count_eight_digit_numbers_product_7000_l2585_258565


namespace NUMINAMATH_CALUDE_male_attendees_on_time_l2585_258527

/-- Proves that the fraction of male attendees who arrived on time is 0.875 -/
theorem male_attendees_on_time (total_attendees : ℝ) : 
  let male_attendees := (3/5 : ℝ) * total_attendees
  let female_attendees := (2/5 : ℝ) * total_attendees
  let on_time_female := (9/10 : ℝ) * female_attendees
  let not_on_time := 0.115 * total_attendees
  let on_time := total_attendees - not_on_time
  ∃ (on_time_male : ℝ), 
    on_time_male + on_time_female = on_time ∧ 
    on_time_male / male_attendees = 0.875 :=
by sorry

end NUMINAMATH_CALUDE_male_attendees_on_time_l2585_258527


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2585_258569

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^3 - 3*x + 1) - 7*(x^3 - x^2 + 3*x - 4) = 8*x^4 - 7*x^3 + x^2 - 19*x + 28 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2585_258569


namespace NUMINAMATH_CALUDE_gcd_459_357_l2585_258560

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2585_258560


namespace NUMINAMATH_CALUDE_fraction_equality_l2585_258549

theorem fraction_equality (a b : ℝ) : (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2585_258549


namespace NUMINAMATH_CALUDE_nuts_distribution_l2585_258552

/-- The number of ways to distribute n identical objects into k distinct groups -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of nuts to be distributed -/
def num_nuts : ℕ := 9

/-- The number of pockets -/
def num_pockets : ℕ := 3

theorem nuts_distribution :
  distribute num_nuts num_pockets = 55 := by sorry

end NUMINAMATH_CALUDE_nuts_distribution_l2585_258552


namespace NUMINAMATH_CALUDE_m_range_l2585_258595

def p (x : ℝ) : Prop := |1 - (x - 2)/3| ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem m_range (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → ¬(p x)) →
  (∃ x, q x m ∧ ¬(p x)) →
  m ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_m_range_l2585_258595


namespace NUMINAMATH_CALUDE_triangle_angle_constraint_l2585_258589

/-- 
Given a triangle ABC with the conditions:
1) 5 * sin(A) + 2 * cos(B) = 5
2) 2 * sin(B) + 5 * cos(A) = 2

This theorem states that either:
a) The triangle is degenerate with angle C = 180°, or
b) There is no solution for a non-degenerate triangle.
-/
theorem triangle_angle_constraint (A B C : ℝ) : 
  (5 * Real.sin A + 2 * Real.cos B = 5) →
  (2 * Real.sin B + 5 * Real.cos A = 2) →
  (A + B + C = Real.pi) →
  ((C = Real.pi ∧ (A = 0 ∨ B = 0)) ∨ 
   ∀ A B C, ¬(A > 0 ∧ B > 0 ∧ C > 0)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_constraint_l2585_258589


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extreme_l2585_258570

open Real

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem f_monotonicity_and_extreme :
  (∀ x y, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) ∧
  (∀ x, f x ≤ f 1) ∧
  (f 1 = 1 / Real.exp 1) := by
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extreme_l2585_258570


namespace NUMINAMATH_CALUDE_intersection_line_hyperbola_l2585_258593

theorem intersection_line_hyperbola (a : ℝ) :
  (∃ A B : ℝ × ℝ, 
    (A.2 = a * A.1 + 1 ∧ 3 * A.1^2 - A.2^2 = 1) ∧
    (B.2 = a * B.1 + 1 ∧ 3 * B.1^2 - B.2^2 = 1) ∧
    A ≠ B) →
  (∃ A B : ℝ × ℝ, 
    (A.2 = a * A.1 + 1 ∧ 3 * A.1^2 - A.2^2 = 1) ∧
    (B.2 = a * B.1 + 1 ∧ 3 * B.1^2 - B.2^2 = 1) ∧
    A ≠ B ∧
    A.1 * B.1 + A.2 * B.2 = 0) →
  a = 1 ∨ a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_line_hyperbola_l2585_258593


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2585_258511

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_in_second_quadrant :
  let z : ℂ := Complex.mk (-2) 1
  is_in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2585_258511


namespace NUMINAMATH_CALUDE_square_sum_ge_product_sum_l2585_258526

theorem square_sum_ge_product_sum (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + a*c := by
  sorry

end NUMINAMATH_CALUDE_square_sum_ge_product_sum_l2585_258526


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2585_258531

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^3 + 3*X^2 = (X^2 + 4*X + 2) * q + (-X^2 - 2*X) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2585_258531


namespace NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l2585_258508

theorem sum_of_x_solutions_is_zero (y : ℝ) (h1 : y = 10) (h2 : ∃ x : ℝ, x^2 + y^2 = 169) : 
  ∃ x1 x2 : ℝ, x1^2 + y^2 = 169 ∧ x2^2 + y^2 = 169 ∧ x1 + x2 = 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x_solutions_is_zero_l2585_258508


namespace NUMINAMATH_CALUDE_rectangle_length_calculation_l2585_258584

theorem rectangle_length_calculation (rectangle_width square_width area_difference : ℝ) : 
  rectangle_width = 6 →
  square_width = 5 →
  rectangle_width * (32 / rectangle_width) - square_width * square_width = area_difference →
  area_difference = 7 →
  32 / rectangle_width = 32 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_calculation_l2585_258584


namespace NUMINAMATH_CALUDE_specific_polyhedron_volume_l2585_258548

/-- Represents a polyhedron formed by folding a specific figure -/
structure Polyhedron where
  /-- Number of isosceles right triangles in the figure -/
  num_triangles : Nat
  /-- Number of squares in the figure -/
  num_squares : Nat
  /-- Number of regular hexagons in the figure -/
  num_hexagons : Nat
  /-- Side length of the isosceles right triangles -/
  triangle_side : ℝ
  /-- Side length of the squares -/
  square_side : ℝ
  /-- Side length of the regular hexagon -/
  hexagon_side : ℝ

/-- Calculates the volume of the polyhedron -/
def volume (p : Polyhedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific polyhedron -/
theorem specific_polyhedron_volume :
  let p : Polyhedron := {
    num_triangles := 3,
    num_squares := 3,
    num_hexagons := 1,
    triangle_side := 2,
    square_side := 2,
    hexagon_side := Real.sqrt 8
  }
  volume p = 47 / 6 := by
  sorry

end NUMINAMATH_CALUDE_specific_polyhedron_volume_l2585_258548


namespace NUMINAMATH_CALUDE_tv_purchase_hours_l2585_258535

/-- The number of additional hours needed to buy a TV given the TV cost, hourly wage, and weekly work hours. -/
def additional_hours_needed (tv_cost : ℕ) (hourly_wage : ℕ) (weekly_work_hours : ℕ) : ℕ :=
  let monthly_work_hours := weekly_work_hours * 4
  let monthly_earnings := monthly_work_hours * hourly_wage
  let additional_amount_needed := tv_cost - monthly_earnings
  additional_amount_needed / hourly_wage

/-- Theorem stating that given a TV cost of $1700, an hourly wage of $10, and a 30-hour workweek, 
    the additional hours needed to buy the TV is 50. -/
theorem tv_purchase_hours : additional_hours_needed 1700 10 30 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tv_purchase_hours_l2585_258535


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l2585_258587

theorem consecutive_odd_numbers (x : ℤ) : 
  (∃ (y z : ℤ), y = x + 2 ∧ z = x + 4 ∧ 
   Odd x ∧ Odd y ∧ Odd z ∧
   11 * x = 3 * (x + 4) + 16 + 4 * (x + 2)) → 
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_l2585_258587


namespace NUMINAMATH_CALUDE_symmetric_graphs_intersection_l2585_258537

noncomputable def f (a b x : ℝ) : ℝ := 2*a + 1/(x-b)

theorem symmetric_graphs_intersection (a b c d : ℝ) :
  (∃! x, f a b x = f c d x) ↔ (a - c) * (b - d) = 2 := by sorry

end NUMINAMATH_CALUDE_symmetric_graphs_intersection_l2585_258537


namespace NUMINAMATH_CALUDE_power_difference_inequality_l2585_258505

theorem power_difference_inequality (n : ℕ) (a b : ℝ) 
  (hn : n > 1) (hab : a > b) (hb : b > 0) :
  (a^n - b^n) * (1/b^(n-1) - 1/a^(n-1)) > 4*n*(n-1)*(Real.sqrt a - Real.sqrt b)^2 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_inequality_l2585_258505


namespace NUMINAMATH_CALUDE_factorial_ratio_2016_l2585_258574

-- Define factorial
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_ratio_2016 :
  (factorial 2016)^2 / (factorial 2015 * factorial 2017) = 2016 / 2017 :=
by sorry

end NUMINAMATH_CALUDE_factorial_ratio_2016_l2585_258574


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2585_258502

theorem three_numbers_sum (a b c : ℝ) :
  a ≤ b → b ≤ c →
  b = 10 →
  (a + b + c) / 3 = a + 20 →
  (a + b + c) / 3 = c - 10 →
  a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2585_258502


namespace NUMINAMATH_CALUDE_cubic_fraction_inequality_l2585_258573

theorem cubic_fraction_inequality (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  1/2 ≤ (a^3 + b^3) / (a^2 + b^2) ∧ (a^3 + b^3) / (a^2 + b^2) ≤ 1 ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1/2 ↔ a = 1/2 ∧ b = 1/2) ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1 ↔ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0)) :=
sorry

end NUMINAMATH_CALUDE_cubic_fraction_inequality_l2585_258573


namespace NUMINAMATH_CALUDE_factor_sum_problem_l2585_258545

theorem factor_sum_problem (N : ℕ) 
  (h1 : N > 0)
  (h2 : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ∣ N ∧ b ∣ N ∧ a + b = 4 ∧ ∀ (x : ℕ), x > 0 → x ∣ N → x ≥ a ∧ x ≥ b)
  (h3 : ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c ∣ N ∧ d ∣ N ∧ c + d = 204 ∧ ∀ (x : ℕ), x > 0 → x ∣ N → x ≤ c ∧ x ≤ d) :
  N = 153 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_problem_l2585_258545


namespace NUMINAMATH_CALUDE_stu_has_four_books_l2585_258533

/-- Given the number of books for Elmo, Laura, and Stu, we define their relationships --/
def book_relation (elmo laura stu : ℕ) : Prop :=
  elmo = 3 * laura ∧ laura = 2 * stu ∧ elmo = 24

/-- Theorem stating that if the book relation holds, then Stu has 4 books --/
theorem stu_has_four_books (elmo laura stu : ℕ) :
  book_relation elmo laura stu → stu = 4 := by
  sorry

end NUMINAMATH_CALUDE_stu_has_four_books_l2585_258533


namespace NUMINAMATH_CALUDE_cos_theta_plus_5pi_over_6_l2585_258553

theorem cos_theta_plus_5pi_over_6 (θ : Real) (h1 : 0 < θ) (h2 : θ < π / 2) 
  (h3 : Real.sin (θ / 2 + π / 6) = 4 / 5) : 
  Real.cos (θ + 5 * π / 6) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_theta_plus_5pi_over_6_l2585_258553


namespace NUMINAMATH_CALUDE_expression_evaluation_l2585_258517

theorem expression_evaluation : 12 - 5 * 3^2 + 8 / 2 - 7 + 4^2 = -20 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2585_258517


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2585_258506

/-- The minimum distance from the origin to the line 3x + 4y - 20 = 0 is 4 -/
theorem min_distance_to_line : ∃ (d : ℝ),
  (∀ (a b : ℝ), 3 * a + 4 * b - 20 = 0 → a^2 + b^2 ≥ d^2) ∧
  (∃ (a b : ℝ), 3 * a + 4 * b - 20 = 0 ∧ a^2 + b^2 = d^2) ∧
  d = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2585_258506


namespace NUMINAMATH_CALUDE_poncelet_theorem_l2585_258532

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle type
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define an incircle
def incircle (t : Triangle) : Circle := sorry

-- Function to check if a point lies on a circle
def lies_on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Theorem statement
theorem poncelet_theorem 
  (ABC DEF : Triangle) 
  (common_incircle : incircle ABC = incircle DEF)
  (c : Circle)
  (A_on_c : lies_on_circle ABC.A c)
  (B_on_c : lies_on_circle ABC.B c)
  (C_on_c : lies_on_circle ABC.C c)
  (D_on_c : lies_on_circle DEF.A c)
  (E_on_c : lies_on_circle DEF.B c) :
  lies_on_circle DEF.C c := by
  sorry


end NUMINAMATH_CALUDE_poncelet_theorem_l2585_258532


namespace NUMINAMATH_CALUDE_assignment_methods_count_l2585_258540

def number_of_departments : ℕ := 5
def number_of_graduates : ℕ := 4
def departments_to_fill : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))
def permute (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

def assignment_methods : ℕ := 
  (choose number_of_departments departments_to_fill) * 
  (choose number_of_graduates 2) * 
  (permute departments_to_fill departments_to_fill)

theorem assignment_methods_count : assignment_methods = 360 := by
  sorry

end NUMINAMATH_CALUDE_assignment_methods_count_l2585_258540


namespace NUMINAMATH_CALUDE_sum_smallest_largest_primes_l2585_258534

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def primes_between (a b : ℕ) : Set ℕ :=
  {n : ℕ | a < n ∧ n < b ∧ is_prime n}

theorem sum_smallest_largest_primes :
  let P := primes_between 50 100
  ∃ (p q : ℕ), p ∈ P ∧ q ∈ P ∧
    (∀ x ∈ P, p ≤ x) ∧
    (∀ x ∈ P, x ≤ q) ∧
    p + q = 150 :=
sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_primes_l2585_258534


namespace NUMINAMATH_CALUDE_translation_of_complex_plane_l2585_258591

open Complex

theorem translation_of_complex_plane (t : ℂ → ℂ) :
  (t (-3 + 3*I) = -8 - 2*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = z + w) →
  (t (-2 + 6*I) = -7 + I) :=
by sorry

end NUMINAMATH_CALUDE_translation_of_complex_plane_l2585_258591


namespace NUMINAMATH_CALUDE_solve_system_l2585_258541

theorem solve_system (s t : ℤ) (eq1 : 11 * s + 7 * t = 160) (eq2 : s = 2 * t + 4) : t = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2585_258541


namespace NUMINAMATH_CALUDE_batsman_average_l2585_258596

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_total = (16 : ℕ) * previous_average ∧ 
  (previous_total + 56) / 17 = previous_average + 3 →
  (previous_total + 56) / 17 = 8 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l2585_258596


namespace NUMINAMATH_CALUDE_max_a_cubic_function_l2585_258564

/-- Given a cubic function f(x) = ax^3 + bx^2 + cx + d with a ≠ 0,
    and |f'(x)| ≤ 1 for 0 ≤ x ≤ 1, the maximum value of a is 8/3. -/
theorem max_a_cubic_function (a b c d : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → |3 * a * x^2 + 2 * b * x + c| ≤ 1) →
  a ≤ 8/3 :=
by sorry

end NUMINAMATH_CALUDE_max_a_cubic_function_l2585_258564


namespace NUMINAMATH_CALUDE_total_water_consumption_is_417_total_water_consumption_proof_l2585_258594

/-- Represents a washing machine with water consumption rates for different wash types -/
structure WashingMachine where
  heavy_wash : ℕ
  regular_wash : ℕ
  light_wash : ℕ

/-- Calculates the total water consumption for a washing machine -/
def water_consumption (m : WashingMachine) (heavy regular light bleach : ℕ) : ℕ :=
  m.heavy_wash * heavy + m.regular_wash * regular + m.light_wash * (light + bleach)

/-- Theorem: The total water consumption for all machines is 417 gallons -/
theorem total_water_consumption_is_417 : ℕ :=
  let machine_a : WashingMachine := ⟨25, 15, 3⟩
  let machine_b : WashingMachine := ⟨20, 12, 2⟩
  let machine_c : WashingMachine := ⟨30, 18, 4⟩
  
  let total_consumption :=
    water_consumption machine_a 3 4 2 4 +
    water_consumption machine_b 2 3 1 3 +
    water_consumption machine_c 4 2 1 5

  417

theorem total_water_consumption_proof :
  (let machine_a : WashingMachine := ⟨25, 15, 3⟩
   let machine_b : WashingMachine := ⟨20, 12, 2⟩
   let machine_c : WashingMachine := ⟨30, 18, 4⟩
   
   let total_consumption :=
     water_consumption machine_a 3 4 2 4 +
     water_consumption machine_b 2 3 1 3 +
     water_consumption machine_c 4 2 1 5

   total_consumption) = 417 := by
  sorry

end NUMINAMATH_CALUDE_total_water_consumption_is_417_total_water_consumption_proof_l2585_258594


namespace NUMINAMATH_CALUDE_regular_hexagon_side_length_l2585_258503

/-- A regular hexagon with a diagonal of 18 inches has sides of 9 inches. -/
theorem regular_hexagon_side_length :
  ∀ (diagonal side : ℝ),
  diagonal = 18 →
  diagonal = 2 * side →
  side = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_side_length_l2585_258503


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2585_258576

open Real

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  ∃ (min_val : Real), min_val = (11 * Real.sqrt 2) / 2 ∧
  ∀ θ', 0 < θ' ∧ θ' < π/2 →
    3 * cos θ' + 2 / sin θ' + 2 * Real.sqrt 2 * tan θ' ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2585_258576


namespace NUMINAMATH_CALUDE_no_solution_exists_l2585_258556

/-- P(n) denotes the greatest prime factor of n -/
def greatest_prime_factor (n : ℕ) : ℕ := sorry

/-- Theorem: There are no positive integers n > 1 such that both 
    P(n) = √n and P(n+36) = √(n+36) -/
theorem no_solution_exists : ¬ ∃ (n : ℕ), n > 1 ∧ 
  (greatest_prime_factor n = Nat.sqrt n) ∧ 
  (greatest_prime_factor (n + 36) = Nat.sqrt (n + 36)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2585_258556


namespace NUMINAMATH_CALUDE_product_of_distinct_nonzero_reals_l2585_258530

theorem product_of_distinct_nonzero_reals (x y : ℝ) : 
  x ≠ 0 → y ≠ 0 → x ≠ y → x + 3 / x = y + 3 / y → x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_nonzero_reals_l2585_258530


namespace NUMINAMATH_CALUDE_min_steps_to_remove_zeros_l2585_258512

/-- Represents the state of the blackboard -/
structure BoardState where
  zeros : Nat
  ones : Nat

/-- Defines a step operation on the board state -/
def step (s : BoardState) : BoardState :=
  { zeros := s.zeros - 1, ones := s.ones + 1 }

/-- Theorem: Minimum steps to remove all zeroes -/
theorem min_steps_to_remove_zeros (initial : BoardState) 
  (h1 : initial.zeros = 150) 
  (h2 : initial.ones = 151) : 
  ∃ (n : Nat), n = 150 ∧ (step^[n] initial).zeros = 0 :=
sorry

end NUMINAMATH_CALUDE_min_steps_to_remove_zeros_l2585_258512


namespace NUMINAMATH_CALUDE_thirteen_y_minus_x_equals_one_l2585_258546

theorem thirteen_y_minus_x_equals_one (x y : ℤ) 
  (h1 : x > 0) 
  (h2 : x = 11 * y + 4) 
  (h3 : 2 * x = 8 * (3 * y) + 3) : 
  13 * y - x = 1 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_y_minus_x_equals_one_l2585_258546


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2585_258558

theorem necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2585_258558


namespace NUMINAMATH_CALUDE_paul_final_stock_l2585_258528

def pencils_per_day : ℕ := 100
def work_days_per_week : ℕ := 5
def initial_stock : ℕ := 80
def pencils_sold : ℕ := 350

def final_stock : ℕ := initial_stock + (pencils_per_day * work_days_per_week) - pencils_sold

theorem paul_final_stock : final_stock = 230 := by sorry

end NUMINAMATH_CALUDE_paul_final_stock_l2585_258528


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2585_258599

/-- The probability of getting exactly k successes in n independent trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The number of coin flips -/
def n : ℕ := 3

/-- The number of times we want the coin to land tails up -/
def k : ℕ := 2

/-- The probability of the coin landing tails up on a single flip -/
def p : ℝ := 0.5

theorem coin_flip_probability : binomial_probability n k p = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2585_258599


namespace NUMINAMATH_CALUDE_professor_percentage_l2585_258586

theorem professor_percentage (total : ℝ) (women_percent : ℝ) (tenured_percent : ℝ) (men_tenured_percent : ℝ) :
  women_percent = 70 →
  tenured_percent = 70 →
  men_tenured_percent = 50 →
  let women := total * (women_percent / 100)
  let tenured := total * (tenured_percent / 100)
  let men := total - women
  let men_tenured := men * (men_tenured_percent / 100)
  let women_tenured := tenured - men_tenured
  let women_or_tenured := women + tenured - women_tenured
  (women_or_tenured / total) * 100 = 85 := by
  sorry

end NUMINAMATH_CALUDE_professor_percentage_l2585_258586


namespace NUMINAMATH_CALUDE_equation_solutions_l2585_258543

theorem equation_solutions : 
  let equation := fun x : ℝ => x^2 * (x + 1)^2 + x^2 - 3 * (x + 1)^2
  ∀ x : ℝ, equation x = 0 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2585_258543


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_315_l2585_258598

/-- The sum of the digits in the binary representation of 315 is 6. -/
theorem sum_of_binary_digits_315 : 
  (Nat.digits 2 315).sum = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_315_l2585_258598


namespace NUMINAMATH_CALUDE_intersection_cardinality_l2585_258518

def M : Finset ℕ := {1, 2, 4, 6, 8}
def N : Finset ℕ := {1, 2, 3, 5, 6, 7}

theorem intersection_cardinality : Finset.card (M ∩ N) = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_cardinality_l2585_258518


namespace NUMINAMATH_CALUDE_probability_two_slate_rocks_l2585_258554

/-- The probability of selecting two slate rocks without replacement from a collection of rocks -/
theorem probability_two_slate_rocks (slate pumice granite : ℕ) 
  (h_slate : slate = 14)
  (h_pumice : pumice = 20)
  (h_granite : granite = 10) :
  let total := slate + pumice + granite
  (slate : ℚ) / total * ((slate - 1) : ℚ) / (total - 1) = 13 / 1892 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_slate_rocks_l2585_258554


namespace NUMINAMATH_CALUDE_stock_price_calculation_l2585_258585

/-- Calculates the final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

/-- Theorem stating that given the specific conditions, the final stock price is $262.5 -/
theorem stock_price_calculation :
  final_stock_price 150 1.5 0.3 = 262.5 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_l2585_258585


namespace NUMINAMATH_CALUDE_quadratic_sequence_problem_l2585_258590

theorem quadratic_sequence_problem (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (eq1 : y₁ + 4*y₂ + 9*y₃ + 16*y₄ + 25*y₅ = 3)
  (eq2 : 4*y₁ + 9*y₂ + 16*y₃ + 25*y₄ + 36*y₅ = 20)
  (eq3 : 9*y₁ + 16*y₂ + 25*y₃ + 36*y₄ + 49*y₅ = 150) :
  16*y₁ + 25*y₂ + 36*y₃ + 49*y₄ + 64*y₅ = 336 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_problem_l2585_258590


namespace NUMINAMATH_CALUDE_job_completion_time_l2585_258515

/-- Proves that given the conditions of the problem, A takes 30 days to complete the job alone. -/
theorem job_completion_time (x : ℝ) (h1 : x > 0) (h2 : 10 * (1 / x + 1 / 40) = 0.5833333333333334) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2585_258515


namespace NUMINAMATH_CALUDE_cone_base_circumference_l2585_258510

/-- The circumference of the base of a right circular cone formed by removing a 180° sector from a circle with radius 6 inches is equal to 6π. -/
theorem cone_base_circumference (r : ℝ) (h : r = 6) :
  let original_circumference := 2 * π * r
  let removed_sector_angle := π  -- 180° in radians
  let full_circle_angle := 2 * π  -- 360° in radians
  let base_circumference := (removed_sector_angle / full_circle_angle) * original_circumference
  base_circumference = 6 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l2585_258510


namespace NUMINAMATH_CALUDE_monomial_same_type_l2585_258561

/-- A structure representing a monomial with coefficients in ℤ -/
structure Monomial :=
  (coeff : ℤ)
  (m_exp : ℕ)
  (n_exp : ℕ)

/-- Two monomials are of the same type if they have the same variables with the same exponents -/
def same_type (a b : Monomial) : Prop :=
  a.m_exp = b.m_exp ∧ a.n_exp = b.n_exp

/-- The monomial -2mn^2 -/
def monomial1 : Monomial :=
  { coeff := -2, m_exp := 1, n_exp := 2 }

/-- The monomial mn^2 -/
def monomial2 : Monomial :=
  { coeff := 1, m_exp := 1, n_exp := 2 }

theorem monomial_same_type : same_type monomial1 monomial2 := by
  sorry

end NUMINAMATH_CALUDE_monomial_same_type_l2585_258561


namespace NUMINAMATH_CALUDE_nelly_babysitting_nights_l2585_258514

/-- The number of nights Nelly needs to babysit to afford pizza for herself and her friends -/
def nights_to_babysit (friends : ℕ) (pizza_cost : ℕ) (people_per_pizza : ℕ) (earnings_per_night : ℕ) : ℕ :=
  let total_people := friends + 1
  let pizzas_needed := (total_people + people_per_pizza - 1) / people_per_pizza
  let total_cost := pizzas_needed * pizza_cost
  (total_cost + earnings_per_night - 1) / earnings_per_night

/-- Theorem stating that Nelly needs to babysit for 15 nights to afford pizza for herself and 14 friends -/
theorem nelly_babysitting_nights :
  nights_to_babysit 14 12 3 4 = 15 := by
  sorry


end NUMINAMATH_CALUDE_nelly_babysitting_nights_l2585_258514


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l2585_258519

-- Define the total number of employees
def total_employees : ℕ := 200

-- Define the number of groups
def num_groups : ℕ := 40

-- Define the size of each group
def group_size : ℕ := 5

-- Define the group from which the known number is drawn
def known_group : ℕ := 5

-- Define the known number drawn
def known_number : ℕ := 23

-- Define the target group
def target_group : ℕ := 10

-- Theorem statement
theorem systematic_sampling_result :
  -- Ensure the total number of employees is divisible by the number of groups
  total_employees = num_groups * group_size →
  -- Ensure the known number is within the range of the known group
  known_number > (known_group - 1) * group_size ∧ known_number ≤ known_group * group_size →
  -- Prove that the number drawn from the target group is 48
  ∃ (n : ℕ), n = (target_group - 1) * group_size + (known_number - (known_group - 1) * group_size) ∧ n = 48 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l2585_258519


namespace NUMINAMATH_CALUDE_allen_blocks_count_l2585_258538

/-- The number of blocks per color -/
def blocks_per_color : ℕ := 7

/-- The number of colors used -/
def colors_used : ℕ := 7

/-- The total number of blocks -/
def total_blocks : ℕ := blocks_per_color * colors_used

theorem allen_blocks_count : total_blocks = 49 := by
  sorry

end NUMINAMATH_CALUDE_allen_blocks_count_l2585_258538


namespace NUMINAMATH_CALUDE_croissants_leftover_l2585_258597

theorem croissants_leftover (total : Nat) (neighbors : Nat) (h1 : total = 59) (h2 : neighbors = 8) :
  total % neighbors = 3 := by
  sorry

end NUMINAMATH_CALUDE_croissants_leftover_l2585_258597


namespace NUMINAMATH_CALUDE_courses_last_year_is_six_l2585_258520

-- Define the number of courses taken last year
def courses_last_year : ℕ := 6

-- Define the average grade last year
def avg_grade_last_year : ℝ := 100

-- Define the number of courses taken the year before
def courses_year_before : ℕ := 5

-- Define the average grade for the year before
def avg_grade_year_before : ℝ := 50

-- Define the average grade for the entire two-year period
def avg_grade_two_years : ℝ := 77

-- Theorem statement
theorem courses_last_year_is_six :
  ((courses_year_before * avg_grade_year_before + courses_last_year * avg_grade_last_year) / 
   (courses_year_before + courses_last_year : ℝ) = avg_grade_two_years) ∧
  (courses_last_year = 6) :=
sorry

end NUMINAMATH_CALUDE_courses_last_year_is_six_l2585_258520


namespace NUMINAMATH_CALUDE_therapy_charge_theorem_l2585_258529

/-- Represents the pricing structure of a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ
  additional_hour : ℕ
  first_hour_premium : first_hour = additional_hour + 20

/-- Calculates the total charge for a given number of therapy hours. -/
def total_charge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  pricing.first_hour + (hours - 1) * pricing.additional_hour

/-- Theorem stating the total charge for 3 hours of therapy given the conditions. -/
theorem therapy_charge_theorem (pricing : TherapyPricing) 
  (h1 : total_charge pricing 5 = 300) : 
  total_charge pricing 3 = 188 := by
  sorry

end NUMINAMATH_CALUDE_therapy_charge_theorem_l2585_258529


namespace NUMINAMATH_CALUDE_december_gas_consumption_l2585_258592

/-- Gas fee structure and consumption for a user in December --/
structure GasConsumption where
  baseRate : ℝ  -- Rate for the first 60 cubic meters
  excessRate : ℝ  -- Rate for consumption above 60 cubic meters
  baseVolume : ℝ  -- Volume threshold for base rate
  averageCost : ℝ  -- Average cost per cubic meter for the user
  consumption : ℝ  -- Total gas consumption

/-- The gas consumption satisfies the given fee structure and average cost --/
def validConsumption (g : GasConsumption) : Prop :=
  g.baseRate * g.baseVolume + g.excessRate * (g.consumption - g.baseVolume) = g.averageCost * g.consumption

/-- Theorem stating that given the fee structure and average cost, 
    the gas consumption in December was 100 cubic meters --/
theorem december_gas_consumption :
  ∃ (g : GasConsumption), 
    g.baseRate = 1 ∧ 
    g.excessRate = 1.5 ∧ 
    g.baseVolume = 60 ∧ 
    g.averageCost = 1.2 ∧ 
    g.consumption = 100 ∧
    validConsumption g :=
  sorry

end NUMINAMATH_CALUDE_december_gas_consumption_l2585_258592


namespace NUMINAMATH_CALUDE_watch_cost_price_l2585_258582

theorem watch_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (additional_amount : ℝ) :
  loss_percentage = 10 →
  gain_percentage = 5 →
  additional_amount = 180 →
  ∃ (cost_price : ℝ),
    cost_price * (1 - loss_percentage / 100) + additional_amount = cost_price * (1 + gain_percentage / 100) ∧
    cost_price = 1200 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2585_258582


namespace NUMINAMATH_CALUDE_solution_product_l2585_258522

theorem solution_product (p q : ℝ) : 
  (p - 7) * (2 * p + 10) = p^2 - 13 * p + 36 →
  (q - 7) * (2 * q + 10) = q^2 - 13 * q + 36 →
  p ≠ q →
  (p - 2) * (q - 2) = -84 := by
  sorry

end NUMINAMATH_CALUDE_solution_product_l2585_258522


namespace NUMINAMATH_CALUDE_max_value_problem_l2585_258578

theorem max_value_problem (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l2585_258578


namespace NUMINAMATH_CALUDE_project_completion_time_l2585_258544

/-- The number of days person A takes to complete the project alone -/
def days_A : ℝ := 45

/-- The number of days person B takes to complete the project alone -/
def days_B : ℝ := 30

/-- The number of days person B works alone initially -/
def initial_days_B : ℝ := 22

/-- The total number of days to complete the project -/
def total_days : ℝ := 34

theorem project_completion_time :
  (total_days - initial_days_B) / days_A + initial_days_B / days_B = 1 := by sorry

end NUMINAMATH_CALUDE_project_completion_time_l2585_258544


namespace NUMINAMATH_CALUDE_line_curve_properties_l2585_258525

/-- Line passing through a point with a given direction vector -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Curve defined by an equation -/
def Curve := (ℝ × ℝ) → Prop

def line_l : Line := { point := (1, 0), direction := (2, -1) }

def curve_C : Curve := fun (x, y) ↦ x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- Distance from a point to a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Line) : ℝ := sorry

/-- Check if a line intersects a curve -/
def intersects (l : Line) (c : Curve) : Prop := sorry

/-- Length of the chord formed by the intersection of a line and a curve -/
def chord_length (l : Line) (c : Curve) : ℝ := sorry

theorem line_curve_properties :
  let origin := (0, 0)
  distance_point_to_line origin line_l = 1 / Real.sqrt 5 ∧
  intersects line_l curve_C ∧
  chord_length line_l curve_C = 2 * Real.sqrt 145 / 5 := by sorry

end NUMINAMATH_CALUDE_line_curve_properties_l2585_258525


namespace NUMINAMATH_CALUDE_balls_in_boxes_theorem_l2585_258563

def number_of_ways (n m k : ℕ) : ℕ :=
  Nat.choose n m * Nat.choose m k * Nat.factorial k

theorem balls_in_boxes_theorem : number_of_ways 5 4 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_theorem_l2585_258563


namespace NUMINAMATH_CALUDE_half_angle_quadrant_l2585_258539

def is_second_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

def is_first_or_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi < α ∧ α < k * Real.pi + Real.pi / 2

theorem half_angle_quadrant (α : Real) :
  is_second_quadrant α → is_first_or_third_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_half_angle_quadrant_l2585_258539


namespace NUMINAMATH_CALUDE_anthonys_remaining_pencils_l2585_258509

/-- Represents the number of pencils Anthony has initially -/
def initial_pencils : ℝ := 56.0

/-- Represents the number of pencils Anthony gives to Kathryn -/
def pencils_given : ℝ := 9.5

/-- Theorem stating that Anthony's remaining pencils equal the initial amount minus the amount given away -/
theorem anthonys_remaining_pencils : 
  initial_pencils - pencils_given = 46.5 := by sorry

end NUMINAMATH_CALUDE_anthonys_remaining_pencils_l2585_258509


namespace NUMINAMATH_CALUDE_lego_sales_triple_pieces_l2585_258547

/-- Represents the number of Lego pieces sold for each type --/
structure LegoSales where
  single : ℕ
  double : ℕ
  triple : ℕ
  quadruple : ℕ

/-- Calculates the total earnings in cents from Lego sales --/
def totalEarnings (sales : LegoSales) : ℕ :=
  sales.single * 1 + sales.double * 2 + sales.triple * 3 + sales.quadruple * 4

/-- The main theorem to prove --/
theorem lego_sales_triple_pieces : 
  ∃ (sales : LegoSales), 
    sales.single = 100 ∧ 
    sales.double = 45 ∧ 
    sales.quadruple = 165 ∧ 
    totalEarnings sales = 1000 ∧ 
    sales.triple = 50 := by
  sorry


end NUMINAMATH_CALUDE_lego_sales_triple_pieces_l2585_258547


namespace NUMINAMATH_CALUDE_extremum_values_l2585_258550

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_values (a b : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x - 1| < ε → f a b x ≤ f a b 1) ∧
  (∃ (δ : ℝ), ∀ (x : ℝ), |x - 1| < δ → f a b x ≥ f a b 1) ∧
  f a b 1 = 10 →
  a = 4 ∧ b = -11 := by sorry

end NUMINAMATH_CALUDE_extremum_values_l2585_258550


namespace NUMINAMATH_CALUDE_brothers_age_ratio_l2585_258523

theorem brothers_age_ratio :
  ∀ (rick_age oldest_age middle_age smallest_age youngest_age : ℕ),
    rick_age = 15 →
    oldest_age = 2 * rick_age →
    middle_age = oldest_age / 3 →
    youngest_age = 3 →
    smallest_age = youngest_age + 2 →
    (smallest_age : ℚ) / (middle_age : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_ratio_l2585_258523


namespace NUMINAMATH_CALUDE_complement_of_at_most_one_hit_l2585_258575

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcome of two consecutive shots -/
def TwoShotOutcome := ShotOutcome × ShotOutcome

/-- The event "at most one shot hits the target" -/
def atMostOneHit (outcome : TwoShotOutcome) : Prop :=
  match outcome with
  | (ShotOutcome.Hit, ShotOutcome.Miss) => True
  | (ShotOutcome.Miss, ShotOutcome.Hit) => True
  | (ShotOutcome.Miss, ShotOutcome.Miss) => True
  | _ => False

/-- The event "both shots hit the target" -/
def bothHit (outcome : TwoShotOutcome) : Prop :=
  outcome = (ShotOutcome.Hit, ShotOutcome.Hit)

theorem complement_of_at_most_one_hit :
  ∀ (outcome : TwoShotOutcome), ¬(atMostOneHit outcome) ↔ bothHit outcome := by
  sorry

end NUMINAMATH_CALUDE_complement_of_at_most_one_hit_l2585_258575


namespace NUMINAMATH_CALUDE_parallel_tangents_intersection_l2585_258580

theorem parallel_tangents_intersection (x₀ : ℝ) : 
  (∃ (k : ℝ), (2 * x₀ = k) ∧ (-3 * x₀^2 = k)) → (x₀ = 0 ∨ x₀ = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_parallel_tangents_intersection_l2585_258580


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l2585_258581

/-- Parabola type -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 4*p*x

/-- Line type -/
structure Line where
  m : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y = m*x + b

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem statement -/
theorem parabola_intersection_theorem (C : Parabola) (a : ℝ) (l : Line) 
  (A B A' : Point) :
  C.p = 3 →  -- This ensures y^2 = 12x
  a < 0 →
  A.y^2 = 12*A.x →
  B.y^2 = 12*B.x →
  l.eq A.x A.y →
  l.eq B.x B.y →
  l.eq a 0 →
  A'.x = A.x →
  A'.y = -A.y →
  ∃ (l' : Line), l'.eq A'.x A'.y ∧ l'.eq B.x B.y ∧ l'.eq (-a) 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l2585_258581


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2585_258524

theorem fraction_inequality_solution_set :
  {x : ℝ | (x - 2) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2585_258524


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_x_in_range_l2585_258579

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - 1| + |x - 2*a|

-- Part I
theorem solution_set_when_a_is_one :
  let a := 1
  ∀ x, f x a ≤ 3 ↔ x ∈ Set.Icc 0 2 :=
sorry

-- Part II
theorem a_value_when_x_in_range :
  (∀ x ∈ Set.Icc 1 2, f x a ≤ 3) → a = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_a_value_when_x_in_range_l2585_258579


namespace NUMINAMATH_CALUDE_rotate_point_A_l2585_258521

/-- Rotate a point 180 degrees about the origin -/
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem rotate_point_A : 
  let A : ℝ × ℝ := (-4, 1)
  rotate180 A = (4, -1) := by
sorry

end NUMINAMATH_CALUDE_rotate_point_A_l2585_258521


namespace NUMINAMATH_CALUDE_log_equation_solution_l2585_258559

/-- Given a > 0, prove that the solution to log_√2(x - a) = 1 + log_2 x is x = a + 1 + √(2a + 1) -/
theorem log_equation_solution (a : ℝ) (ha : a > 0) :
  ∃! x : ℝ, x > a ∧ Real.log (x - a) / Real.log (Real.sqrt 2) = 1 + Real.log x / Real.log 2 ∧
  x = a + 1 + Real.sqrt (2 * a + 1) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2585_258559
