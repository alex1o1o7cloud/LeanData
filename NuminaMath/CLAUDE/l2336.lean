import Mathlib

namespace NUMINAMATH_CALUDE_power_calculation_l2336_233615

theorem power_calculation : 4^2009 * (-0.25)^2008 - 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2336_233615


namespace NUMINAMATH_CALUDE_average_value_sequence_l2336_233653

theorem average_value_sequence (x : ℝ) : 
  let sequence := [0, 3*x, 6*x, 12*x, 24*x]
  (sequence.sum / sequence.length : ℝ) = 9*x := by
  sorry

end NUMINAMATH_CALUDE_average_value_sequence_l2336_233653


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2336_233660

def M : Set ℕ := {1, 2, 5}
def N : Set ℕ := {x | x ≤ 2}

theorem intersection_of_M_and_N : M ∩ N = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2336_233660


namespace NUMINAMATH_CALUDE_division_remainder_l2336_233647

theorem division_remainder : ∃ (A B : ℕ), 26 = 4 * A + B ∧ B < 4 ∧ B = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l2336_233647


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2336_233697

/-- The complex number z = i(1+i) is located in the second quadrant of the complex plane. -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := Complex.I * (1 + Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l2336_233697


namespace NUMINAMATH_CALUDE_inscribed_triangle_ratio_l2336_233624

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a triangle -/
structure Triangle where
  p : Point
  q : Point
  r : Point

theorem inscribed_triangle_ratio (a : ℝ) (b : ℝ) (c : ℝ) (e : Ellipse) (t : Triangle) :
  e.a = a ∧ e.b = b ∧
  c = (3/5) * a ∧
  t.q = Point.mk 0 b ∧
  t.p.y = t.r.y ∧
  t.p.x = -c ∧ t.r.x = c ∧
  (t.p.x^2 / a^2) + (t.p.y^2 / b^2) = 1 ∧
  (t.q.x^2 / a^2) + (t.q.y^2 / b^2) = 1 ∧
  (t.r.x^2 / a^2) + (t.r.y^2 / b^2) = 1 ∧
  2 * c = 0.6 * a →
  (Real.sqrt ((t.p.x - t.q.x)^2 + (t.p.y - t.q.y)^2)) / (t.r.x - t.p.x) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_ratio_l2336_233624


namespace NUMINAMATH_CALUDE_square_side_length_l2336_233601

theorem square_side_length (d : ℝ) (h : d = Real.sqrt 8) :
  ∃ s : ℝ, s > 0 ∧ s * Real.sqrt 2 = d ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2336_233601


namespace NUMINAMATH_CALUDE_total_cats_is_31_l2336_233634

/-- The number of cats owned by Jamie, Gordon, Hawkeye, and Natasha -/
def total_cats : ℕ :=
  let jamie_persian := 4
  let jamie_maine_coon := 2
  let gordon_persian := jamie_persian / 2
  let gordon_maine_coon := jamie_maine_coon + 1
  let hawkeye_persian := 0
  let hawkeye_maine_coon := gordon_maine_coon * 2
  let natasha_persian := 3
  let natasha_maine_coon := jamie_maine_coon + gordon_maine_coon + hawkeye_maine_coon
  jamie_persian + jamie_maine_coon +
  gordon_persian + gordon_maine_coon +
  hawkeye_persian + hawkeye_maine_coon +
  natasha_persian + natasha_maine_coon

theorem total_cats_is_31 : total_cats = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_cats_is_31_l2336_233634


namespace NUMINAMATH_CALUDE_difference_not_necessarily_periodic_l2336_233625

-- Define a periodic function
def Periodic (f : ℝ → ℝ) : Prop :=
  (∃ x y, f x ≠ f y) ∧ 
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x)

-- Define functions g and h with their respective periods
def g_periodic (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 6) = g x

def h_periodic (h : ℝ → ℝ) : Prop :=
  ∀ x, h (x + 2 * Real.pi) = h x

-- Theorem statement
theorem difference_not_necessarily_periodic 
  (g h : ℝ → ℝ) 
  (hg : g_periodic g) 
  (hh : h_periodic h) :
  ¬ (∀ f : ℝ → ℝ, f = g - h → Periodic f) :=
sorry

end NUMINAMATH_CALUDE_difference_not_necessarily_periodic_l2336_233625


namespace NUMINAMATH_CALUDE_racket_price_proof_l2336_233694

/-- Given the total cost of items and the costs of sneakers and sports outfit, 
    prove that the price of the tennis racket is the difference between the total 
    cost and the sum of the other items' costs. -/
theorem racket_price_proof (total_cost sneakers_cost outfit_cost : ℕ) 
    (h1 : total_cost = 750)
    (h2 : sneakers_cost = 200)
    (h3 : outfit_cost = 250) : 
    total_cost - (sneakers_cost + outfit_cost) = 300 := by
  sorry

#check racket_price_proof

end NUMINAMATH_CALUDE_racket_price_proof_l2336_233694


namespace NUMINAMATH_CALUDE_sqrt_sum_expression_l2336_233656

theorem sqrt_sum_expression (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_minimal : ∀ (a' b' c' : ℕ), a' > 0 → b' > 0 → c' > 0 → 
    (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 7 + 1 / Real.sqrt 7) * c' = a' * Real.sqrt 3 + b' * Real.sqrt 7 
    → c ≤ c')
  (h_equality : (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 7 + 1 / Real.sqrt 7) * c = a * Real.sqrt 3 + b * Real.sqrt 7) :
  a + b + c = 73 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_expression_l2336_233656


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l2336_233690

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem fiftieth_term_of_sequence : arithmetic_sequence 2 3 50 = 149 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l2336_233690


namespace NUMINAMATH_CALUDE_G_1000_units_digit_l2336_233671

/-- Modified Fermat number -/
def G (n : ℕ) : ℕ := 3^(3^n) + 2

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

theorem G_1000_units_digit :
  unitsDigit (G 1000) = 5 := by
  sorry

end NUMINAMATH_CALUDE_G_1000_units_digit_l2336_233671


namespace NUMINAMATH_CALUDE_total_differential_arcctg_l2336_233680

noncomputable def z (x y : ℝ) : ℝ := Real.arctan (y / x)

theorem total_differential_arcctg (x y dx dy : ℝ) (hx : x = 1) (hy : y = 3) (hdx : dx = 0.01) (hdy : dy = -0.05) :
  let dz := -(y / (x^2 + y^2)) * dx + (x / (x^2 + y^2)) * dy
  dz = -0.008 := by
  sorry

end NUMINAMATH_CALUDE_total_differential_arcctg_l2336_233680


namespace NUMINAMATH_CALUDE_oil_price_reduction_l2336_233672

/-- Prove that the amount spent on oil is 1500 given the conditions --/
theorem oil_price_reduction (original_price reduced_price amount_spent : ℝ) 
  (h1 : reduced_price = original_price * (1 - 0.2))
  (h2 : reduced_price = 30)
  (h3 : amount_spent / reduced_price - amount_spent / original_price = 10) :
  amount_spent = 1500 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l2336_233672


namespace NUMINAMATH_CALUDE_range_of_fraction_l2336_233696

theorem range_of_fraction (x y : ℝ) (h : x + Real.sqrt (1 - y^2) = 0) :
  ∃ (a b : ℝ), a = -Real.sqrt 3 / 3 ∧ b = Real.sqrt 3 / 3 ∧
  ∀ (z : ℝ), (∃ (x' y' : ℝ), x' + Real.sqrt (1 - y'^2) = 0 ∧ z = y' / (x' - 2)) →
  a ≤ z ∧ z ≤ b :=
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l2336_233696


namespace NUMINAMATH_CALUDE_ball_attendees_l2336_233695

theorem ball_attendees :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendees_l2336_233695


namespace NUMINAMATH_CALUDE_book_purchase_equation_l2336_233673

theorem book_purchase_equation (x : ℝ) : x > 0 →
  (∀ y : ℝ, y = x + 8 → y > 0) →
  (15000 : ℝ) / (x + 8) = (12000 : ℝ) / x :=
by
  sorry

end NUMINAMATH_CALUDE_book_purchase_equation_l2336_233673


namespace NUMINAMATH_CALUDE_jessie_points_l2336_233627

def total_points : ℕ := 311
def other_players_points : ℕ := 188
def num_equal_scorers : ℕ := 3

theorem jessie_points : 
  (total_points - other_players_points) / num_equal_scorers = 41 := by
  sorry

end NUMINAMATH_CALUDE_jessie_points_l2336_233627


namespace NUMINAMATH_CALUDE_sum_of_squares_101_to_200_l2336_233610

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_101_to_200 :
  sum_of_squares 200 - sum_of_squares 100 = 2348350 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_101_to_200_l2336_233610


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_tank_l2336_233645

/-- The area of a stripe on a cylindrical tank -/
theorem stripe_area_on_cylindrical_tank
  (diameter : ℝ)
  (stripe_width : ℝ)
  (revolutions : ℝ)
  (h_diameter : diameter = 20)
  (h_stripe_width : stripe_width = 4)
  (h_revolutions : revolutions = 3) :
  stripe_width * revolutions * (π * diameter) = 240 * π :=
by sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_tank_l2336_233645


namespace NUMINAMATH_CALUDE_total_travel_options_l2336_233665

/-- The number of train options from location A to location B -/
def train_options : ℕ := 3

/-- The number of ferry options from location B to location C -/
def ferry_options : ℕ := 2

/-- The number of direct flight options from location A to location C -/
def flight_options : ℕ := 2

/-- The total number of travel options from location A to location C -/
def total_options : ℕ := train_options * ferry_options + flight_options

theorem total_travel_options : total_options = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_options_l2336_233665


namespace NUMINAMATH_CALUDE_minimal_divisible_number_l2336_233679

theorem minimal_divisible_number : ∃! n : ℕ,
  2007000 ≤ n ∧ n < 2008000 ∧
  n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧
  (∀ m : ℕ, 2007000 ≤ m ∧ m < n → (m % 3 ≠ 0 ∨ m % 5 ≠ 0 ∨ m % 7 ≠ 0)) ∧
  n = 2007075 :=
sorry

end NUMINAMATH_CALUDE_minimal_divisible_number_l2336_233679


namespace NUMINAMATH_CALUDE_trigonometric_product_transformation_l2336_233617

theorem trigonometric_product_transformation (α : ℝ) :
  4.66 * Real.sin (5 * π / 2 + 4 * α) - Real.sin (5 * π / 2 + 2 * α) ^ 6 + Real.cos (7 * π / 2 - 2 * α) ^ 6 = 
  (1 / 8) * Real.sin (4 * α) * Real.sin (8 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_transformation_l2336_233617


namespace NUMINAMATH_CALUDE_modulo_seventeen_residue_l2336_233602

theorem modulo_seventeen_residue : (352 + 6 * 68 + 8 * 221 + 3 * 34 + 5 * 17) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_modulo_seventeen_residue_l2336_233602


namespace NUMINAMATH_CALUDE_point_coordinates_l2336_233651

/-- A point in the second quadrant with given distances from axes has coordinates (-1, 2) -/
theorem point_coordinates (P : ℝ × ℝ) :
  (P.1 < 0 ∧ P.2 > 0) →  -- P is in the second quadrant
  |P.2| = 2 →            -- Distance from P to x-axis is 2
  |P.1| = 1 →            -- Distance from P to y-axis is 1
  P = (-1, 2) :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l2336_233651


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2336_233626

/-- A cubic polynomial with rational coefficients -/
def CubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + p*x^2 + q*x + r
  where
  p := -(a + b + c)
  q := a*b + b*c + c*a
  r := -a*b*c

theorem cubic_roots_sum (a b c : ℝ) :
  (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c < 1) →
  CubicPolynomial a b c 0 = -1/8 →
  (∃ r : ℝ, b = a*r ∧ c = a*r^2) →
  (∑' k, (a^k + b^k + c^k)) = 9/2 →
  a + b + c = 19/12 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2336_233626


namespace NUMINAMATH_CALUDE_largest_common_term_l2336_233658

theorem largest_common_term (n m : ℕ) : 
  (147 = 2 + 5 * n) ∧ 
  (147 = 3 + 8 * m) ∧ 
  (147 ≤ 150) ∧ 
  (∀ k : ℕ, k > 147 → k ≤ 150 → (k - 2) % 5 ≠ 0 ∨ (k - 3) % 8 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l2336_233658


namespace NUMINAMATH_CALUDE_unique_number_property_l2336_233611

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l2336_233611


namespace NUMINAMATH_CALUDE_even_student_schools_count_l2336_233608

/-- Represents a school with its student count -/
structure School where
  name : String
  students : ℕ

/-- Checks if a number is even -/
def isEven (n : ℕ) : Bool :=
  n % 2 = 0

/-- Counts the number of schools with an even number of students -/
def countEvenStudentSchools (schools : List School) : ℕ :=
  (schools.filter (fun s => isEven s.students)).length

/-- The main theorem -/
theorem even_student_schools_count :
  let schools : List School := [
    ⟨"A", 786⟩,
    ⟨"B", 777⟩,
    ⟨"C", 762⟩,
    ⟨"D", 819⟩,
    ⟨"E", 493⟩
  ]
  countEvenStudentSchools schools = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_student_schools_count_l2336_233608


namespace NUMINAMATH_CALUDE_sqrt_ratio_equality_l2336_233622

theorem sqrt_ratio_equality : 
  (Real.sqrt (3^2 + 4^2)) / (Real.sqrt (25 + 16)) = (5 * Real.sqrt 41) / 41 := by
sorry

end NUMINAMATH_CALUDE_sqrt_ratio_equality_l2336_233622


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2336_233650

/-- The minimum distance from the origin to the line 2x - y + 1 = 0 is √5/5 -/
theorem min_distance_to_line : 
  let line := {(x, y) : ℝ × ℝ | 2 * x - y + 1 = 0}
  ∃ (d : ℝ), d = Real.sqrt 5 / 5 ∧ 
    (∀ (P : ℝ × ℝ), P ∈ line → d ≤ Real.sqrt (P.1^2 + P.2^2)) ∧
    (∃ (P : ℝ × ℝ), P ∈ line ∧ d = Real.sqrt (P.1^2 + P.2^2)) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_to_line_l2336_233650


namespace NUMINAMATH_CALUDE_simplify_expression_l2336_233613

theorem simplify_expression : 8 * (15 / 4) * (-56 / 45) = -112 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2336_233613


namespace NUMINAMATH_CALUDE_function_equality_l2336_233661

theorem function_equality (f : ℤ → ℤ) :
  (∀ a b : ℤ, f (a^2 + b^2) + f (a * b) = f a ^ 2 + f b + 1) →
  (∀ a : ℤ, f a = 1) := by
sorry

end NUMINAMATH_CALUDE_function_equality_l2336_233661


namespace NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l2336_233655

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {2, 3}
def B : Set Nat := {3, 5}

theorem intersection_complement_equals_singleton : A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l2336_233655


namespace NUMINAMATH_CALUDE_triangle_areas_l2336_233648

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point O (intersection of altitudes)
def O : ℝ × ℝ := sorry

-- Define the points P, Q, R on the sides of the triangle
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry
def R : ℝ × ℝ := sorry

-- Define the given conditions
axiom parallel_RP_AC : sorry
axiom AC_length : sorry
axiom sin_ABC : sorry

-- Define the areas of triangles ABC and ROC
noncomputable def area_ABC (t : Triangle) : ℝ := sorry
noncomputable def area_ROC (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_areas (t : Triangle) :
  (area_ABC t = 16/3 ∧ area_ROC t = 21/25) ∨
  (area_ABC t = 3 ∧ area_ROC t = 112/75) :=
sorry

end NUMINAMATH_CALUDE_triangle_areas_l2336_233648


namespace NUMINAMATH_CALUDE_base_7_multiplication_l2336_233684

/-- Converts a number from base 7 to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a number from base 10 to base 7 --/
def to_base_7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Theorem statement --/
theorem base_7_multiplication :
  to_base_7 (to_base_10 [4, 2, 3] * to_base_10 [3]) = [5, 0, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_7_multiplication_l2336_233684


namespace NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l2336_233691

theorem smallest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 15*n + 56 ≤ 0 ∧ ∀ (m : ℤ), m^2 - 15*m + 56 ≤ 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_quadratic_inequality_l2336_233691


namespace NUMINAMATH_CALUDE_extremum_at_one_decreasing_when_a_geq_two_monotonicity_when_a_lt_two_l2336_233637

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := (2 - a) * x - 2 * Real.log x

def f_deriv (x : ℝ) : ℝ := 2 - a - 2 / x

theorem extremum_at_one (h : f_deriv a 1 = 0) : a = 0 := by sorry

theorem decreasing_when_a_geq_two (h : a ≥ 2) : 
  ∀ x > 0, f_deriv a x < 0 := by sorry

theorem monotonicity_when_a_lt_two (h : a < 2) :
  (∀ x ∈ Set.Ioo 0 (2 / (2 - a)), f_deriv a x < 0) ∧
  (∀ x ∈ Set.Ioi (2 / (2 - a)), f_deriv a x > 0) := by sorry

end NUMINAMATH_CALUDE_extremum_at_one_decreasing_when_a_geq_two_monotonicity_when_a_lt_two_l2336_233637


namespace NUMINAMATH_CALUDE_fraction_addition_l2336_233664

theorem fraction_addition (m n : ℚ) (h : m / n = 3 / 7) : (m + n) / n = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2336_233664


namespace NUMINAMATH_CALUDE_marks_speed_l2336_233632

/-- Given a distance of 24 miles and a time of 4 hours, prove that the speed is 6 miles per hour. -/
theorem marks_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 24 ∧ time = 4 ∧ speed = distance / time → speed = 6 := by
  sorry

end NUMINAMATH_CALUDE_marks_speed_l2336_233632


namespace NUMINAMATH_CALUDE_odd_prime_divisibility_l2336_233600

theorem odd_prime_divisibility (p a b c : ℤ) : 
  Prime p → 
  Odd p → 
  (p ∣ a^2023 + b^2023) → 
  (p ∣ b^2024 + c^2024) → 
  (p ∣ a^2025 + c^2025) → 
  (p ∣ a) ∧ (p ∣ b) ∧ (p ∣ c) := by
sorry

end NUMINAMATH_CALUDE_odd_prime_divisibility_l2336_233600


namespace NUMINAMATH_CALUDE_min_value_inequality_l2336_233662

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

-- State the theorem
theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ a + b + c = m) →
  a^2 + b^2 + c^2 ≥ 16/3 := by
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2336_233662


namespace NUMINAMATH_CALUDE_x_value_proof_l2336_233654

theorem x_value_proof (x : ℝ) (h1 : x^2 - 3*x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2336_233654


namespace NUMINAMATH_CALUDE_largest_multiple_of_nine_below_negative_seventy_l2336_233663

theorem largest_multiple_of_nine_below_negative_seventy :
  ∀ n : ℤ, n % 9 = 0 ∧ n < -70 → n ≤ -72 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_nine_below_negative_seventy_l2336_233663


namespace NUMINAMATH_CALUDE_joes_lift_weight_l2336_233674

theorem joes_lift_weight (first_lift second_lift : ℕ) : 
  first_lift + second_lift = 600 →
  2 * first_lift = second_lift + 300 →
  first_lift = 300 := by
  sorry

end NUMINAMATH_CALUDE_joes_lift_weight_l2336_233674


namespace NUMINAMATH_CALUDE_power_inequality_l2336_233621

theorem power_inequality (a b : ℝ) (n : ℕ+) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : Odd b) 
  (h4 : (b ^ n.val) ∣ (a ^ n.val - 1)) : 
  a ^ (⌊b⌋) > 3 ^ n.val / n.val :=
sorry

end NUMINAMATH_CALUDE_power_inequality_l2336_233621


namespace NUMINAMATH_CALUDE_quadratic_point_ordering_l2336_233689

/-- A quadratic function of the form y = -(x+1)² + c -/
def quadratic_function (c : ℝ) (x : ℝ) : ℝ := -(x + 1)^2 + c

theorem quadratic_point_ordering (c : ℝ) :
  let y₁ := quadratic_function c (-13/4)
  let y₂ := quadratic_function c (-1)
  let y₃ := quadratic_function c 0
  y₁ < y₃ ∧ y₃ < y₂ := by sorry

end NUMINAMATH_CALUDE_quadratic_point_ordering_l2336_233689


namespace NUMINAMATH_CALUDE_binomial_probability_two_successes_l2336_233619

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- Theorem: For a random variable ξ following a binomial distribution B(6, 1/3),
    the probability P(ξ = 2) is equal to 80/243 -/
theorem binomial_probability_two_successes :
  binomial_pmf 6 (1/3) 2 = 80/243 :=
sorry

end NUMINAMATH_CALUDE_binomial_probability_two_successes_l2336_233619


namespace NUMINAMATH_CALUDE_abs_equation_unique_solution_l2336_233620

theorem abs_equation_unique_solution :
  ∃! x : ℝ, |x - 9| = |x + 3| := by
sorry

end NUMINAMATH_CALUDE_abs_equation_unique_solution_l2336_233620


namespace NUMINAMATH_CALUDE_season_games_count_l2336_233657

/-- The number of teams in the league -/
def num_teams : ℕ := 12

/-- The number of times each team plays every other team -/
def games_per_matchup : ℕ := 2

/-- The number of non-league games each team plays -/
def non_league_games : ℕ := 5

/-- The total number of games in a season -/
def total_games : ℕ := (num_teams * (num_teams - 1) / 2) * games_per_matchup + num_teams * non_league_games

theorem season_games_count : total_games = 192 := by
  sorry

end NUMINAMATH_CALUDE_season_games_count_l2336_233657


namespace NUMINAMATH_CALUDE_cyclist_speed_l2336_233677

theorem cyclist_speed (distance : ℝ) (time_difference : ℝ) : 
  distance = 96 →
  time_difference = 16 →
  ∃ (speed : ℝ), 
    speed > 0 ∧
    distance / (speed - 4) = distance / (1.5 * speed) + time_difference ∧
    speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_l2336_233677


namespace NUMINAMATH_CALUDE_video_votes_l2336_233681

theorem video_votes (score : ℤ) (like_percentage : ℚ) : 
  score = 130 ∧ like_percentage = 70 / 100 → 
  ∃ total_votes : ℕ, 
    (like_percentage * total_votes : ℚ) - ((1 - like_percentage) * total_votes : ℚ) = score ∧
    total_votes = 325 := by
  sorry

end NUMINAMATH_CALUDE_video_votes_l2336_233681


namespace NUMINAMATH_CALUDE_evaluate_expression_l2336_233652

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/4) (hy : y = 3/4) (hz : z = -8) :
  x^2 * y^3 * z^2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2336_233652


namespace NUMINAMATH_CALUDE_power_23_mod_5_l2336_233614

theorem power_23_mod_5 : 2^23 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_23_mod_5_l2336_233614


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2336_233636

/-- The surface area of a cube inscribed in a sphere, which is itself inscribed in another cube --/
theorem inscribed_cube_surface_area (outer_cube_area : ℝ) : 
  outer_cube_area = 150 →
  ∃ (inner_cube_area : ℝ), inner_cube_area = 50 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l2336_233636


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2336_233670

-- Problem 1
theorem problem_1 (x : ℝ) (h : x^2 - 2*x = 5) : 2*x^2 - 4*x + 2023 = 2033 := by
  sorry

-- Problem 2
theorem problem_2 (m n : ℝ) (h : m - n = -3) : 2*(m-n) - m + n + 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2336_233670


namespace NUMINAMATH_CALUDE_equal_lengths_implies_k_value_l2336_233675

theorem equal_lengths_implies_k_value (AB AC : ℝ) (k : ℝ) :
  AB = AC → AB = 8 → AC = 5 - k → k = -3 := by
  sorry

end NUMINAMATH_CALUDE_equal_lengths_implies_k_value_l2336_233675


namespace NUMINAMATH_CALUDE_geometric_sequence_transformation_l2336_233635

/-- Given a geometric sequence {a_n} with common ratio q (q ≠ 1),
    prove that the sequence {b_n} defined as b_n = a_{3n-2} + a_{3n-1} + a_{3n}
    is a geometric sequence with common ratio q^3. -/
theorem geometric_sequence_transformation (q : ℝ) (hq : q ≠ 1) (a : ℕ → ℝ) 
    (h_geom : ∀ n : ℕ, a (n + 1) = q * a n) :
  let b : ℕ → ℝ := λ n ↦ a (3 * n - 2) + a (3 * n - 1) + a (3 * n)
  ∀ n : ℕ, b (n + 1) = q^3 * b n := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_transformation_l2336_233635


namespace NUMINAMATH_CALUDE_flatbread_division_l2336_233666

-- Define a planar region
def PlanarRegion : Type := Set (ℝ × ℝ)

-- Define the area of a planar region
noncomputable def area (R : PlanarRegion) : ℝ := sorry

-- Define a line in 2D space
def Line : Type := Set (ℝ × ℝ)

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the division of a planar region by two lines
def divide (R : PlanarRegion) (l1 l2 : Line) : List PlanarRegion := sorry

-- Theorem statement
theorem flatbread_division (R : PlanarRegion) (P : ℝ) (h : area R = P) :
  ∃ (l1 l2 : Line), perpendicular l1 l2 ∧ 
    ∀ (part : PlanarRegion), part ∈ divide R l1 l2 → area part = P / 4 := by
  sorry

end NUMINAMATH_CALUDE_flatbread_division_l2336_233666


namespace NUMINAMATH_CALUDE_square_sum_diff_product_l2336_233646

theorem square_sum_diff_product (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ((a + b)^2 - (a - b)^2)^2 / (a * b)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_diff_product_l2336_233646


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2336_233616

/-- The hyperbola C with parameter m -/
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

/-- The asymptote of the hyperbola C -/
def asymptote (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + m * y = 0

/-- The focal length of a hyperbola -/
def focal_length (m : ℝ) : ℝ := sorry

theorem hyperbola_focal_length (m : ℝ) (h1 : m > 0) :
  (∀ x y : ℝ, hyperbola m x y ↔ asymptote m x y) →
  focal_length m = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2336_233616


namespace NUMINAMATH_CALUDE_unique_phone_number_l2336_233640

/-- A six-digit number -/
def SixDigitNumber := { n : ℕ // 100000 ≤ n ∧ n < 1000000 }

/-- The set of divisors we're interested in -/
def Divisors : Finset ℕ := {3, 4, 7, 9, 11, 13}

/-- The property that a number gives the same remainder when divided by all numbers in Divisors -/
def SameRemainder (n : ℕ) : Prop :=
  ∃ r, ∀ d ∈ Divisors, n % d = r

/-- The main theorem -/
theorem unique_phone_number :
  ∃! (T : SixDigitNumber),
    Odd T.val ∧
    (T.val / 100000 = 7) ∧
    ((T.val / 100) % 10 = 2) ∧
    SameRemainder T.val ∧
    T.val = 720721 := by
  sorry

#check unique_phone_number

end NUMINAMATH_CALUDE_unique_phone_number_l2336_233640


namespace NUMINAMATH_CALUDE_pipeline_theorem_l2336_233667

/-- Represents the pipeline construction problem -/
structure PipelineConstruction where
  total_length : ℝ
  daily_increase : ℝ
  days_ahead : ℝ
  actual_daily_length : ℝ

/-- The equation describing the pipeline construction problem -/
def pipeline_equation (p : PipelineConstruction) : Prop :=
  p.total_length / (p.actual_daily_length - p.daily_increase) -
  p.total_length / p.actual_daily_length = p.days_ahead

/-- Theorem stating that the equation holds for the given parameters -/
theorem pipeline_theorem (p : PipelineConstruction)
  (h1 : p.total_length = 4000)
  (h2 : p.daily_increase = 10)
  (h3 : p.days_ahead = 20) :
  pipeline_equation p :=
sorry

end NUMINAMATH_CALUDE_pipeline_theorem_l2336_233667


namespace NUMINAMATH_CALUDE_patrick_less_than_twice_greg_l2336_233659

def homework_hours (jacob greg patrick : ℕ) : Prop :=
  jacob = 18 ∧ 
  greg = jacob - 6 ∧ 
  jacob + greg + patrick = 50

theorem patrick_less_than_twice_greg : 
  ∀ jacob greg patrick : ℕ, 
  homework_hours jacob greg patrick → 
  2 * greg - patrick = 4 := by
sorry

end NUMINAMATH_CALUDE_patrick_less_than_twice_greg_l2336_233659


namespace NUMINAMATH_CALUDE_remaining_money_is_130_l2336_233629

/-- Given an initial amount of money, calculate the remaining amount after spending on books and DVDs -/
def remaining_money (initial : ℚ) : ℚ :=
  let after_books := initial - (1/4 * initial + 10)
  let after_dvds := after_books - (2/5 * after_books + 8)
  after_dvds

/-- Theorem: Given $320 initially, the remaining money after buying books and DVDs is $130 -/
theorem remaining_money_is_130 : remaining_money 320 = 130 := by
  sorry

#eval remaining_money 320

end NUMINAMATH_CALUDE_remaining_money_is_130_l2336_233629


namespace NUMINAMATH_CALUDE_f_2016_value_l2336_233623

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem f_2016_value (f : ℝ → ℝ) 
  (h1 : f 1 = 1/4)
  (h2 : functional_equation f) : 
  f 2016 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_f_2016_value_l2336_233623


namespace NUMINAMATH_CALUDE_line_triangle_area_theorem_l2336_233688

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Checks if a line forms a triangle with the coordinate axes -/
def formsTriangle (l : Line) : Prop :=
  l.b ≠ 0 ∧ l.b / l.m < 0

/-- Calculates the area of the triangle formed by a line and the coordinate axes -/
noncomputable def triangleArea (l : Line) : ℝ :=
  abs (l.b * (l.b / l.m)) / 2

/-- The main theorem -/
theorem line_triangle_area_theorem (k : ℝ) :
  let l : Line := { m := -2, b := k }
  formsTriangle l ∧ triangleArea l = 4 → k = 4 ∨ k = -4 := by
  sorry

end NUMINAMATH_CALUDE_line_triangle_area_theorem_l2336_233688


namespace NUMINAMATH_CALUDE_farm_animals_count_l2336_233698

theorem farm_animals_count (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 300) 
  (h2 : total_legs = 688) : 
  ∃ (ducks cows : ℕ), 
    ducks + cows = total_animals ∧ 
    2 * ducks + 4 * cows = total_legs ∧ 
    ducks = 256 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_count_l2336_233698


namespace NUMINAMATH_CALUDE_point_slope_problem_l2336_233649

/-- If m > 0 and the points (m, 4) and (2, m) lie on a line with slope m², then m = 2. -/
theorem point_slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = m^2) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_slope_problem_l2336_233649


namespace NUMINAMATH_CALUDE_cricket_bat_price_l2336_233612

theorem cricket_bat_price (cost_price_A : ℝ) (profit_A_percent : ℝ) (profit_B_percent : ℝ) : 
  cost_price_A = 156 →
  profit_A_percent = 20 →
  profit_B_percent = 25 →
  let selling_price_B := cost_price_A * (1 + profit_A_percent / 100)
  let selling_price_C := selling_price_B * (1 + profit_B_percent / 100)
  selling_price_C = 234 :=
by sorry

end NUMINAMATH_CALUDE_cricket_bat_price_l2336_233612


namespace NUMINAMATH_CALUDE_factory_wage_problem_l2336_233682

/-- Proves that the hourly rate for the remaining employees is $17 given the problem conditions -/
theorem factory_wage_problem (total_employees : ℕ) (employees_at_12 : ℕ) (employees_at_14 : ℕ)
  (shift_length : ℕ) (total_cost : ℕ) :
  total_employees = 300 →
  employees_at_12 = 200 →
  employees_at_14 = 40 →
  shift_length = 8 →
  total_cost = 31840 →
  let remaining_employees := total_employees - (employees_at_12 + employees_at_14)
  let remaining_cost := total_cost - (employees_at_12 * 12 * shift_length + employees_at_14 * 14 * shift_length)
  remaining_cost / (remaining_employees * shift_length) = 17 := by
  sorry

#check factory_wage_problem

end NUMINAMATH_CALUDE_factory_wage_problem_l2336_233682


namespace NUMINAMATH_CALUDE_condition1_condition2_f_satisfies_conditions_l2336_233630

/-- A function satisfying the given conditions -/
def f (x : ℝ) := -3 * x

/-- The first condition: f(x) + f(-x) = 0 for all x ∈ ℝ -/
theorem condition1 : ∀ x : ℝ, f x + f (-x) = 0 := by
  sorry

/-- The second condition: f(x + t) - f(x) < 0 for all x ∈ ℝ and t > 0 -/
theorem condition2 : ∀ x t : ℝ, t > 0 → f (x + t) - f x < 0 := by
  sorry

/-- The main theorem: f satisfies both conditions -/
theorem f_satisfies_conditions : 
  (∀ x : ℝ, f x + f (-x) = 0) ∧ 
  (∀ x t : ℝ, t > 0 → f (x + t) - f x < 0) := by
  sorry

end NUMINAMATH_CALUDE_condition1_condition2_f_satisfies_conditions_l2336_233630


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l2336_233687

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (70/31, 135/31)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 8*x - 3*y = 5

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 5*x + 2*y = 20

theorem intersection_point_is_solution :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y ∧
  ∀ x' y', line1 x' y' ∧ line2 x' y' → x' = x ∧ y' = y := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l2336_233687


namespace NUMINAMATH_CALUDE_intersection_area_is_sqrt_k_l2336_233669

/-- Regular tetrahedron with edge length 5 -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_eq : edge_length = 5

/-- Plane passing through specific points of a regular tetrahedron -/
structure IntersectionPlane (t : RegularTetrahedron) where
  -- Midpoint of edge VA
  point_R : ℝ × ℝ × ℝ
  -- Midpoint of edge AB
  point_S : ℝ × ℝ × ℝ
  -- Point one-third from C to B
  point_T : ℝ × ℝ × ℝ

/-- Area of the intersection between the tetrahedron and the plane -/
def intersection_area (t : RegularTetrahedron) (p : IntersectionPlane t) : ℝ := sorry

/-- The theorem to be proved -/
theorem intersection_area_is_sqrt_k (t : RegularTetrahedron) (p : IntersectionPlane t) :
  ∃ k : ℝ, k > 0 ∧ intersection_area t p = Real.sqrt k := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_sqrt_k_l2336_233669


namespace NUMINAMATH_CALUDE_program1_output_program2_output_l2336_233604

-- Define a type to represent the state of the program
structure ProgramState where
  a : Int
  b : Int
  c : Int

-- Function to simulate the first program
def program1 (initial : ProgramState) : ProgramState :=
  { a := initial.b
  , b := initial.c
  , c := initial.c }

-- Function to simulate the second program
def program2 (initial : ProgramState) : ProgramState :=
  { a := initial.b
  , b := initial.c
  , c := initial.b }

-- Theorem for the first program
theorem program1_output :
  let initial := ProgramState.mk 3 (-5) 8
  let final := program1 initial
  final.a = -5 ∧ final.b = 8 ∧ final.c = 8 := by sorry

-- Theorem for the second program
theorem program2_output :
  let initial := ProgramState.mk 3 (-5) 8
  let final := program2 initial
  final.a = -5 ∧ final.b = 8 ∧ final.c = -5 := by sorry

end NUMINAMATH_CALUDE_program1_output_program2_output_l2336_233604


namespace NUMINAMATH_CALUDE_no_double_application_increment_l2336_233692

theorem no_double_application_increment (f : ℕ → ℕ) : ∃ n : ℕ, n > 0 ∧ f (f n) ≠ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_increment_l2336_233692


namespace NUMINAMATH_CALUDE_tims_score_is_2352_l2336_233683

-- Define the first 8 prime numbers
def first_8_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the product of the first 8 prime numbers
def prime_product : Nat := first_8_primes.prod

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Define N as the sum of digits in the product of the first 8 prime numbers
def N : Nat := sum_of_digits prime_product

-- Define Tim's score as the sum of the first N even numbers
def tims_score : Nat := N * (N + 1)

-- The theorem to prove
theorem tims_score_is_2352 : tims_score = 2352 := by sorry

end NUMINAMATH_CALUDE_tims_score_is_2352_l2336_233683


namespace NUMINAMATH_CALUDE_prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l2336_233631

/- Define the number of white and black balls -/
def num_white : ℕ := 4
def num_black : ℕ := 2
def total_balls : ℕ := num_white + num_black

/- Define the number of draws -/
def num_draws : ℕ := 3

/- Theorem for drawing without replacement -/
theorem prob_at_least_one_black_without_replacement :
  let total_ways := Nat.choose total_balls num_draws
  let all_white_ways := Nat.choose num_white num_draws
  (1 : ℚ) - (all_white_ways : ℚ) / (total_ways : ℚ) = 4/5 := by sorry

/- Theorem for drawing with replacement -/
theorem prob_exactly_one_black_with_replacement :
  let total_ways := total_balls ^ num_draws
  let one_black_ways := num_draws * num_black * (num_white ^ (num_draws - 1))
  (one_black_ways : ℚ) / (total_ways : ℚ) = 4/9 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l2336_233631


namespace NUMINAMATH_CALUDE_f_three_equals_three_l2336_233606

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_three_equals_three :
  (∀ x, f (2 * x - 1) = x + 1) → f 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_three_equals_three_l2336_233606


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l2336_233618

theorem circle_center_and_radius :
  ∀ (x y : ℝ), 4*x^2 - 8*x + 4*y^2 + 24*y + 28 = 0 ↔ 
  (x - 1)^2 + (y + 3)^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l2336_233618


namespace NUMINAMATH_CALUDE_max_digit_sum_is_24_l2336_233668

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60

/-- Calculates the sum of digits for a given natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time in 24-hour format -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum possible sum of digits in a 24-hour format display -/
def maxDigitSum : Nat := 24

/-- Theorem stating that the maximum sum of digits in a 24-hour format display is 24 -/
theorem max_digit_sum_is_24 :
  ∀ t : Time24, timeDigitSum t ≤ maxDigitSum :=
by
  sorry

#check max_digit_sum_is_24

end NUMINAMATH_CALUDE_max_digit_sum_is_24_l2336_233668


namespace NUMINAMATH_CALUDE_sufficiency_not_necessity_l2336_233686

theorem sufficiency_not_necessity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b = 2 → a * b ≤ 1) ∧
  ∃ (c d : ℝ), 0 < c ∧ 0 < d ∧ c * d ≤ 1 ∧ c + d ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_sufficiency_not_necessity_l2336_233686


namespace NUMINAMATH_CALUDE_maggie_bouncy_balls_indeterminate_l2336_233678

theorem maggie_bouncy_balls_indeterminate 
  (yellow_packs : ℝ) 
  (green_packs_given : ℝ) 
  (balls_per_pack : ℝ) 
  (total_kept : ℕ) 
  (h1 : yellow_packs = 8.0)
  (h2 : green_packs_given = 4.0)
  (h3 : balls_per_pack = 10.0)
  (h4 : total_kept = 80)
  (h5 : yellow_packs * balls_per_pack = total_kept) :
  ∃ (x y : ℝ), x ≠ y ∧ 
    (yellow_packs * balls_per_pack - green_packs_given * balls_per_pack + x * balls_per_pack = total_kept) ∧
    (yellow_packs * balls_per_pack - green_packs_given * balls_per_pack + y * balls_per_pack = total_kept) :=
by sorry

end NUMINAMATH_CALUDE_maggie_bouncy_balls_indeterminate_l2336_233678


namespace NUMINAMATH_CALUDE_two_digit_product_2210_l2336_233676

theorem two_digit_product_2210 (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 2210 →
  min a b = 26 := by
sorry

end NUMINAMATH_CALUDE_two_digit_product_2210_l2336_233676


namespace NUMINAMATH_CALUDE_red_balls_count_l2336_233628

theorem red_balls_count (total : ℕ) (p_black : ℚ) (p_at_least_one_white : ℚ) :
  total = 10 ∧ 
  p_black = 2/5 ∧ 
  p_at_least_one_white = 7/9 →
  ∃ (black white red : ℕ), 
    black + white + red = total ∧
    black = 4 ∧
    red = 1 ∧
    (black : ℚ) / total = p_black ∧
    1 - (Nat.choose (black + red) 2 : ℚ) / (Nat.choose total 2) = p_at_least_one_white :=
by
  sorry

#check red_balls_count

end NUMINAMATH_CALUDE_red_balls_count_l2336_233628


namespace NUMINAMATH_CALUDE_volunteers_distribution_l2336_233641

/-- The number of ways to distribute n volunteers into k schools,
    with each school receiving at least one volunteer. -/
def distribute_volunteers (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 75 volunteers into 3 schools,
    with each school receiving at least one volunteer, is equal to 150. -/
theorem volunteers_distribution :
  distribute_volunteers 75 3 = 150 := by sorry

end NUMINAMATH_CALUDE_volunteers_distribution_l2336_233641


namespace NUMINAMATH_CALUDE_carla_counted_books_thrice_l2336_233644

/-- Represents the counting scenario for Carla on Monday and Tuesday -/
structure CarlaCounting where
  monday_tiles : ℕ
  monday_books : ℕ
  tuesday_total : ℕ

/-- Calculates the number of times Carla counted the books on Tuesday -/
def books_count_tuesday (c : CarlaCounting) : ℕ :=
  let tuesday_tiles := c.monday_tiles * 2
  let tuesday_books := c.tuesday_total - tuesday_tiles
  tuesday_books / c.monday_books

/-- Theorem stating that given the conditions, Carla counted the books 3 times on Tuesday -/
theorem carla_counted_books_thrice (c : CarlaCounting) 
  (h1 : c.monday_tiles = 38)
  (h2 : c.monday_books = 75)
  (h3 : c.tuesday_total = 301) :
  books_count_tuesday c = 3 := by
  sorry

end NUMINAMATH_CALUDE_carla_counted_books_thrice_l2336_233644


namespace NUMINAMATH_CALUDE_geometric_sequence_S5_l2336_233639

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_S5 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  (a 3 + a 4) / (a 1 + a 2) = 4 →
  ∃ S5 : ℝ, (S5 = 31 ∨ S5 = 11) ∧ S5 = (a 1 + a 2 + a 3 + a 4 + a 5) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_S5_l2336_233639


namespace NUMINAMATH_CALUDE_circle_area_increase_l2336_233685

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l2336_233685


namespace NUMINAMATH_CALUDE_series_sum_equals_257_l2336_233609

def series_sum : ℕ := by
  -- Define the ranges for n, m, and p
  let n_range := Finset.range 12
  let m_range := Finset.range 3
  let p_range := Finset.range 2

  -- Define the summation functions
  let f_n (n : ℕ) := 2 * (n + 1)
  let f_m (m : ℕ) := 3 * (2 * m + 3)
  let f_p (p : ℕ) := 4 * (4 * p + 2)

  -- Calculate the sum
  exact (n_range.sum f_n) + (m_range.sum f_m) + (p_range.sum f_p)

theorem series_sum_equals_257 : series_sum = 257 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_257_l2336_233609


namespace NUMINAMATH_CALUDE_some_number_proof_l2336_233693

def total_prime_factors (n : ℕ) : ℕ := sorry

theorem some_number_proof (x : ℕ) :
  total_prime_factors (x * 11^13 * 7^5) = 29 → x = 2^11 := by
  sorry

end NUMINAMATH_CALUDE_some_number_proof_l2336_233693


namespace NUMINAMATH_CALUDE_tangent_sum_twelve_eighteen_equals_sqrt_three_over_three_l2336_233643

theorem tangent_sum_twelve_eighteen_equals_sqrt_three_over_three :
  (Real.tan (12 * π / 180) + Real.tan (18 * π / 180)) / 
  (1 - Real.tan (12 * π / 180) * Real.tan (18 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_twelve_eighteen_equals_sqrt_three_over_three_l2336_233643


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_to_zero_l2336_233605

theorem sqrt_two_minus_one_to_zero : (Real.sqrt 2 - 1) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_to_zero_l2336_233605


namespace NUMINAMATH_CALUDE_folded_paper_distance_l2336_233638

/-- Given a square sheet of paper with area 18 cm², when folded so that a corner point A
    rests on the diagonal making the visible black area equal to the visible white area,
    the distance from A to its original position is 2√6 cm. -/
theorem folded_paper_distance (s : ℝ) (x : ℝ) :
  s^2 = 18 →
  (1/2) * x^2 = 18 - x^2 →
  Real.sqrt (2 * x^2) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_distance_l2336_233638


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2336_233699

theorem function_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x * (2 * Real.log a - Real.log x) ≤ a) →
  (0 < a ∧ a ≤ Real.exp (-1)) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_range_l2336_233699


namespace NUMINAMATH_CALUDE_quadratic_real_equal_roots_l2336_233607

theorem quadratic_real_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 2 * x + 6 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y + 2 * y + 6 = 0 → y = x) ↔ 
  (m = 2 - 6 * Real.sqrt 2 ∨ m = 2 + 6 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_equal_roots_l2336_233607


namespace NUMINAMATH_CALUDE_preimage_of_four_l2336_233603

def f (x : ℝ) : ℝ := x^2

theorem preimage_of_four (x : ℝ) : f x = 4 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_four_l2336_233603


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1242_l2336_233633

theorem sum_of_largest_and_smallest_prime_factors_of_1242 :
  ∃ (smallest largest : Nat),
    smallest.Prime ∧
    largest.Prime ∧
    smallest ∣ 1242 ∧
    largest ∣ 1242 ∧
    (∀ p : Nat, p.Prime → p ∣ 1242 → p ≤ largest) ∧
    (∀ p : Nat, p.Prime → p ∣ 1242 → p ≥ smallest) ∧
    smallest + largest = 25 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1242_l2336_233633


namespace NUMINAMATH_CALUDE_oil_volume_in_liters_l2336_233642

def bottle_volume : ℝ := 200
def num_bottles : ℕ := 20
def ml_per_liter : ℝ := 1000

theorem oil_volume_in_liters :
  (bottle_volume * num_bottles) / ml_per_liter = 4 := by
  sorry

end NUMINAMATH_CALUDE_oil_volume_in_liters_l2336_233642
