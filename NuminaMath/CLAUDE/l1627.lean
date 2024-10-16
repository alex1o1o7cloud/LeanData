import Mathlib

namespace NUMINAMATH_CALUDE_max_a_is_correct_l1627_162759

/-- The quadratic function f(x) = -x^2 + 2x - 2 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

/-- The maximum value of a for which f(x) is increasing when x ≤ a -/
def max_a : ℝ := 1

theorem max_a_is_correct :
  ∀ a : ℝ, (∀ x y : ℝ, x ≤ y → y ≤ a → f x ≤ f y) → a ≤ max_a :=
by sorry

end NUMINAMATH_CALUDE_max_a_is_correct_l1627_162759


namespace NUMINAMATH_CALUDE_seat_distribution_correct_l1627_162798

/-- Represents the total number of seats on the airplane -/
def total_seats : ℕ := 90

/-- Represents the number of seats in First Class -/
def first_class_seats : ℕ := 36

/-- Represents the proportion of seats in Business Class -/
def business_class_proportion : ℚ := 1/5

/-- Represents the proportion of seats in Premium Economy -/
def premium_economy_proportion : ℚ := 2/5

/-- Theorem stating that the given seat distribution is correct -/
theorem seat_distribution_correct : 
  first_class_seats + 
  (business_class_proportion * total_seats).floor + 
  (premium_economy_proportion * total_seats).floor + 
  (total_seats - first_class_seats - 
   (business_class_proportion * total_seats).floor - 
   (premium_economy_proportion * total_seats).floor) = total_seats :=
by sorry

end NUMINAMATH_CALUDE_seat_distribution_correct_l1627_162798


namespace NUMINAMATH_CALUDE_range_of_m_m_value_for_specific_chord_length_m_value_for_perpendicular_chords_l1627_162714

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

-- Theorem 1: Range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → m < 5 :=
sorry

-- Theorem 2: Value of m when |MN| = 4√5/5
theorem m_value_for_specific_chord_length :
  ∃ m : ℝ, 
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      circle_equation x₁ y₁ m ∧ 
      circle_equation x₂ y₂ m ∧
      line_equation x₁ y₁ ∧ 
      line_equation x₂ y₂ ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4*Real.sqrt 5/5)^2) ∧
    m = 4 :=
sorry

-- Theorem 3: Value of m when OM ⊥ ON
theorem m_value_for_perpendicular_chords :
  ∃ m : ℝ, 
    (∃ x₁ y₁ x₂ y₂ : ℝ,
      circle_equation x₁ y₁ m ∧ 
      circle_equation x₂ y₂ m ∧
      line_equation x₁ y₁ ∧ 
      line_equation x₂ y₂ ∧
      x₁*x₂ + y₁*y₂ = 0) ∧
    m = 8/5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_m_value_for_specific_chord_length_m_value_for_perpendicular_chords_l1627_162714


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l1627_162773

/-- The percentage of water in fresh grapes by weight -/
def water_percentage_fresh : ℝ := 90

/-- The percentage of water in dried grapes by weight -/
def water_percentage_dried : ℝ := 20

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 20

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 2.5

theorem water_percentage_in_fresh_grapes :
  water_percentage_fresh = 90 ∧
  (100 - water_percentage_fresh) / 100 * fresh_weight = 
  (100 - water_percentage_dried) / 100 * dried_weight :=
by sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l1627_162773


namespace NUMINAMATH_CALUDE_percentage_equality_l1627_162728

theorem percentage_equality : (0.1 / 100) * 12356 = 12.356000000000002 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1627_162728


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1627_162739

-- Define the functions DE, BC, and DB
def DE (x : ℝ) : ℝ := sorry
def BC (x : ℝ) : ℝ := sorry
def DB (x : ℝ) : ℝ := sorry

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | DE x * BC x = DE x * (2 * DB x) ∧ DE x * BC x = 2 * (DE x)^2} = 
  {x : ℝ | 9/4 < x ∧ x < 19/4} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1627_162739


namespace NUMINAMATH_CALUDE_missy_dog_yells_l1627_162797

/-- Represents the number of times Missy yells at her dogs -/
structure DogYells where
  obedient : ℕ
  stubborn : ℕ
  total : ℕ

/-- Theorem: If Missy yells at the obedient dog 12 times and yells at both dogs combined 60 times,
    then she yells at the stubborn dog 4 times for every one time she yells at the obedient dog -/
theorem missy_dog_yells (d : DogYells) 
    (h1 : d.obedient = 12)
    (h2 : d.total = 60)
    (h3 : d.total = d.obedient + d.stubborn) :
    d.stubborn = 4 * d.obedient := by
  sorry

#check missy_dog_yells

end NUMINAMATH_CALUDE_missy_dog_yells_l1627_162797


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l1627_162785

theorem sum_of_special_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x * y = 50 * (x + y)) (h2 : x * y = 75 * (x - y)) :
  x + y = 360 := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l1627_162785


namespace NUMINAMATH_CALUDE_sin_n_equals_cos_276_l1627_162745

theorem sin_n_equals_cos_276 :
  ∃ n : ℤ, -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.cos (276 * π / 180) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_sin_n_equals_cos_276_l1627_162745


namespace NUMINAMATH_CALUDE_calculation_proof_l1627_162712

theorem calculation_proof :
  ((-1/2) * (-8) + (-6) = -2) ∧
  (-1^4 - 2 / (-1/3) - |-9| = -4) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l1627_162712


namespace NUMINAMATH_CALUDE_tangent_circles_theorem_l1627_162711

/-- Two circles in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a plane -/
def Point : Type := ℝ × ℝ

/-- Predicate to check if two circles are tangent -/
def are_tangent (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a point is on the common tangent of two circles -/
def on_common_tangent (p : Point) (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if the common tangent is perpendicular to the line joining the centers -/
def perpendicular_to_center_line (p : Point) (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a circle is tangent to another circle -/
def is_tangent_to (c1 c2 : Circle) : Prop := sorry

/-- Predicate to check if a point is on a circle -/
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

theorem tangent_circles_theorem 
  (c1 c2 : Circle) 
  (p : Point) 
  (h1 : are_tangent c1 c2)
  (h2 : on_common_tangent p c1 c2)
  (h3 : perpendicular_to_center_line p c1 c2) :
  ∃! (s1 s2 : Circle), 
    s1 ≠ s2 ∧ 
    is_tangent_to s1 c1 ∧ 
    is_tangent_to s1 c2 ∧ 
    point_on_circle p s1 ∧
    is_tangent_to s2 c1 ∧ 
    is_tangent_to s2 c2 ∧ 
    point_on_circle p s2 :=
  sorry

end NUMINAMATH_CALUDE_tangent_circles_theorem_l1627_162711


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1627_162702

theorem trigonometric_inequality (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 
  2 * Real.cos x ≤ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ∧
  Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ≤ Real.sqrt 2 →
  Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1627_162702


namespace NUMINAMATH_CALUDE_tangent_line_circle_l1627_162794

/-- The line 4x - 3y = 0 is tangent to the circle x^2 + y^2 - 2x + ay + 1 = 0 if and only if a = -1 or a = 4 -/
theorem tangent_line_circle (a : ℝ) : 
  (∀ x y : ℝ, (4 * x - 3 * y = 0 ∧ x^2 + y^2 - 2*x + a*y + 1 = 0) → 
    (∀ x' y' : ℝ, x'^2 + y'^2 - 2*x' + a*y' + 1 = 0 → (x = x' ∧ y = y'))) ↔ 
  (a = -1 ∨ a = 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l1627_162794


namespace NUMINAMATH_CALUDE_sum_equality_seven_eight_l1627_162705

theorem sum_equality_seven_eight (S : Finset ℤ) (h : S.card = 15) :
  {s | ∃ (T : Finset ℤ), T ⊆ S ∧ T.card = 7 ∧ s = T.sum id} =
  {s | ∃ (T : Finset ℤ), T ⊆ S ∧ T.card = 8 ∧ s = T.sum id} :=
by sorry

end NUMINAMATH_CALUDE_sum_equality_seven_eight_l1627_162705


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1627_162724

theorem sum_of_coefficients (a b c d e f g h j k : ℤ) :
  (∀ x y : ℝ, 27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) →
  a + b + c + d + e + f + g + h + j + k = 92 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1627_162724


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1627_162778

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  1 / x + 1 / y = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1627_162778


namespace NUMINAMATH_CALUDE_k_value_l1627_162713

theorem k_value (θ : Real) (k : Real) 
  (h1 : k = (3 * Real.sin θ + 5 * Real.cos θ) / (2 * Real.sin θ + Real.cos θ))
  (h2 : Real.tan θ = 3) : k = 2 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l1627_162713


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l1627_162757

theorem right_triangle_acute_angles (θ₁ θ₂ : ℝ) : 
  θ₁ = 25 → θ₁ + θ₂ = 90 → θ₂ = 65 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l1627_162757


namespace NUMINAMATH_CALUDE_rectangle_area_sum_l1627_162726

theorem rectangle_area_sum (a b : ℤ) (h1 : a > b) (h2 : b > 1) : 
  (2 * (a - b).natAbs * (a + b).natAbs = 50) → a + b = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_sum_l1627_162726


namespace NUMINAMATH_CALUDE_milan_bill_cost_l1627_162787

/-- Calculates the total cost of a long distance phone bill -/
def long_distance_bill_cost (monthly_fee : ℚ) (cost_per_minute : ℚ) (minutes_used : ℕ) : ℚ :=
  monthly_fee + cost_per_minute * minutes_used

/-- Proves that Milan's long distance bill cost is $23.36 -/
theorem milan_bill_cost :
  let monthly_fee : ℚ := 2
  let cost_per_minute : ℚ := 12 / 100
  let minutes_used : ℕ := 178
  long_distance_bill_cost monthly_fee cost_per_minute minutes_used = 2336 / 100 := by
  sorry

end NUMINAMATH_CALUDE_milan_bill_cost_l1627_162787


namespace NUMINAMATH_CALUDE_pauline_cars_l1627_162747

theorem pauline_cars (total : ℕ) (regular_percent : ℚ) (truck_percent : ℚ) 
  (h_total : total = 125)
  (h_regular : regular_percent = 64/100)
  (h_truck : truck_percent = 8/100) :
  (total : ℚ) * (1 - regular_percent - truck_percent) = 35 := by
  sorry

end NUMINAMATH_CALUDE_pauline_cars_l1627_162747


namespace NUMINAMATH_CALUDE_cups_needed_for_six_cookies_l1627_162738

/-- The number of cups in a quart -/
def cups_per_quart : ℚ := 4

/-- The number of cookies that can be baked with 3 quarts of milk -/
def cookies_per_three_quarts : ℚ := 18

/-- The number of cookies we want to bake -/
def target_cookies : ℚ := 6

/-- The number of cups of milk needed to bake the target number of cookies -/
def cups_needed : ℚ := (3 * cups_per_quart * target_cookies) / cookies_per_three_quarts

theorem cups_needed_for_six_cookies :
  cups_needed = 4 := by sorry

end NUMINAMATH_CALUDE_cups_needed_for_six_cookies_l1627_162738


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1627_162732

/-- Theorem: For a hyperbola with equation x² - y²/b² = 1 where b > 0,
    if one of its asymptotes has equation y = 3x, then b = 3. -/
theorem hyperbola_asymptote (b : ℝ) (hb : b > 0) :
  (∃ x y : ℝ, x^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = 3 * x ∧ x^2 - y^2 / b^2 = 1) →
  b = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1627_162732


namespace NUMINAMATH_CALUDE_ada_original_seat_l1627_162792

-- Define the number of seats
def num_seats : ℕ := 6

-- Define the movements of friends
def bea_move : ℤ := 3
def ceci_move : ℤ := 1
def dee_move : ℤ := -2
def edie_move : ℤ := -1

-- Define Ada's final position
def ada_final_seat : ℕ := 2

-- Theorem statement
theorem ada_original_seat :
  let net_displacement := bea_move + ceci_move + dee_move + edie_move
  net_displacement = 1 →
  ∃ (ada_original : ℕ), 
    ada_original > 0 ∧ 
    ada_original ≤ num_seats ∧
    ada_original - ada_final_seat = 1 := by
  sorry

end NUMINAMATH_CALUDE_ada_original_seat_l1627_162792


namespace NUMINAMATH_CALUDE_max_value_exponents_l1627_162762

theorem max_value_exponents (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : a < 1) :
  (a^b : ℝ) = max (a^b) (max (b^a) (max (a^a) (b^b))) := by sorry

end NUMINAMATH_CALUDE_max_value_exponents_l1627_162762


namespace NUMINAMATH_CALUDE_no_intersection_l1627_162743

-- Define the two functions
def f (x : ℝ) : ℝ := |3 * x + 6|
def g (x : ℝ) : ℝ := -|4 * x - 3|

-- Theorem stating that there are no intersection points
theorem no_intersection :
  ¬ ∃ (x y : ℝ), f x = y ∧ g x = y :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l1627_162743


namespace NUMINAMATH_CALUDE_second_term_value_l1627_162735

theorem second_term_value (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A / B = 3 / 4) (h4 : (A + 10) / (B + 10) = 4 / 5) : B = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_term_value_l1627_162735


namespace NUMINAMATH_CALUDE_sqrt_x4_eq_x2_l1627_162706

theorem sqrt_x4_eq_x2 : ∀ x : ℝ, Real.sqrt (x^4) = x^2 := by sorry

end NUMINAMATH_CALUDE_sqrt_x4_eq_x2_l1627_162706


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1627_162779

theorem min_value_of_expression (x : ℝ) :
  let f := λ x : ℝ => Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((x - 1)^2 + (x - 1)^2)
  (∀ x, f x ≥ 1) ∧ (∃ x, f x = 1) := by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1627_162779


namespace NUMINAMATH_CALUDE_smallest_k_and_largest_base_l1627_162790

theorem smallest_k_and_largest_base : ∃ (b : ℕ), 
  (64 ^ 7 > b ^ 20) ∧ 
  (∀ (x : ℕ), x > b → 64 ^ 7 ≤ x ^ 20) ∧ 
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_and_largest_base_l1627_162790


namespace NUMINAMATH_CALUDE_first_investment_rate_l1627_162760

/-- Proves that given the conditions, the interest rate of the first investment is 10% --/
theorem first_investment_rate (total_investment : ℝ) (second_investment : ℝ) (second_rate : ℝ) (income_difference : ℝ) :
  total_investment = 2000 →
  second_investment = 650 →
  second_rate = 0.08 →
  income_difference = 83 →
  ∃ (first_rate : ℝ),
    first_rate * (total_investment - second_investment) - second_rate * second_investment = income_difference ∧
    first_rate = 0.10 :=
by sorry

end NUMINAMATH_CALUDE_first_investment_rate_l1627_162760


namespace NUMINAMATH_CALUDE_equidistant_points_in_quadrants_I_II_l1627_162719

/-- A point (x, y) on the line 4x + 6y = 18 that is equidistant from both coordinate axes -/
def EquidistantPoint (x y : ℝ) : Prop :=
  4 * x + 6 * y = 18 ∧ |x| = |y|

/-- A point (x, y) is in quadrant I -/
def InQuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- A point (x, y) is in quadrant II -/
def InQuadrantII (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- A point (x, y) is in quadrant III -/
def InQuadrantIII (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- A point (x, y) is in quadrant IV -/
def InQuadrantIV (x y : ℝ) : Prop := x > 0 ∧ y < 0

theorem equidistant_points_in_quadrants_I_II :
  ∀ x y : ℝ, EquidistantPoint x y →
  (InQuadrantI x y ∨ InQuadrantII x y) ∧
  ¬(InQuadrantIII x y ∨ InQuadrantIV x y) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_points_in_quadrants_I_II_l1627_162719


namespace NUMINAMATH_CALUDE_z_value_theorem_l1627_162791

theorem z_value_theorem (z w : ℝ) (hz : z ≠ 0) (hw : w ≠ 0)
  (h1 : z + 1 / w = 15) (h2 : w^2 + 1 / z = 3) : z = 44 / 3 := by
  sorry

end NUMINAMATH_CALUDE_z_value_theorem_l1627_162791


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1627_162796

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * Real.sqrt 5 = 2 * Real.sqrt (a^2 + b^2)) →
  (b / a = 1 / 2) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 4 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1627_162796


namespace NUMINAMATH_CALUDE_table_rotation_l1627_162763

theorem table_rotation (table_length table_width : ℝ) (S : ℕ) : 
  table_length = 9 →
  table_width = 12 →
  S = ⌈(table_length^2 + table_width^2).sqrt⌉ →
  S = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_table_rotation_l1627_162763


namespace NUMINAMATH_CALUDE_water_bottles_needed_l1627_162749

theorem water_bottles_needed (people : ℕ) (trip_hours : ℕ) (bottles_per_person_per_hour : ℚ) : 
  people = 10 → trip_hours = 24 → bottles_per_person_per_hour = 1/2 →
  (people : ℚ) * trip_hours * bottles_per_person_per_hour = 120 :=
by sorry

end NUMINAMATH_CALUDE_water_bottles_needed_l1627_162749


namespace NUMINAMATH_CALUDE_tangent_line_slope_l1627_162710

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x

/-- The derivative of f -/
noncomputable def f' (x : ℝ) : ℝ := Real.exp x

/-- A point on the curve where the tangent line passes through -/
noncomputable def x₀ : ℝ := 1

/-- The slope of the tangent line -/
noncomputable def k : ℝ := f' x₀

theorem tangent_line_slope : k = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l1627_162710


namespace NUMINAMATH_CALUDE_value_of_expression_l1627_162752

theorem value_of_expression (a b : ℤ) (ha : a = -3) (hb : b = 2) : a * (b - 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1627_162752


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l1627_162740

theorem monic_quartic_polynomial_value (q : ℝ → ℝ) : 
  (∃ a b c : ℝ, ∀ x, q x = x^4 + a*x^3 + b*x^2 + c*x + 3) →
  q 0 = 3 →
  q 1 = 4 →
  q 2 = 7 →
  q 3 = 12 →
  q 4 = 43 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l1627_162740


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1627_162768

theorem closest_integer_to_cube_root : ∃ (n : ℤ), 
  n = 10 ∧ ∀ (m : ℤ), |m - (7^3 + 9^3)^(1/3)| ≥ |n - (7^3 + 9^3)^(1/3)| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1627_162768


namespace NUMINAMATH_CALUDE_john_experience_theorem_l1627_162795

/-- Represents the years of experience for each person -/
structure Experience where
  james : ℕ
  john : ℕ
  mike : ℕ

/-- The conditions of the problem -/
def problem_conditions (e : Experience) : Prop :=
  e.james = 20 ∧
  e.john - 8 = 2 * (e.james - 8) ∧
  e.james + e.john + e.mike = 68

/-- John's experience when Mike started -/
def john_experience_when_mike_started (e : Experience) : ℕ :=
  e.john - e.mike

/-- The theorem to prove -/
theorem john_experience_theorem (e : Experience) :
  problem_conditions e → john_experience_when_mike_started e = 16 := by
  sorry

end NUMINAMATH_CALUDE_john_experience_theorem_l1627_162795


namespace NUMINAMATH_CALUDE_problem_solution_l1627_162751

def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x < 4 ↔ x ∈ Set.Ioo (-2) 2) ∧
  (∃ x : ℝ, f x - |a - 1| < 0) ↔ a ∈ Set.Iio (-1) ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1627_162751


namespace NUMINAMATH_CALUDE_solution_system_l1627_162799

theorem solution_system (x y : ℝ) 
  (eq1 : ⌊x⌋ + (y - ⌊y⌋) = 7.2)
  (eq2 : (x - ⌊x⌋) + ⌊y⌋ = 10.3) : 
  |x - y| = 2.9 := by
  sorry

end NUMINAMATH_CALUDE_solution_system_l1627_162799


namespace NUMINAMATH_CALUDE_sequence_split_equal_sum_l1627_162720

theorem sequence_split_equal_sum (p : ℕ) (h_prime : Nat.Prime p) :
  (∃ (b : ℕ) (S : ℕ) (splits : List (List ℕ)),
    b > 1 ∧
    splits.length = b ∧
    (∀ l ∈ splits, l.sum = S) ∧
    splits.join = List.range p) ↔ p = 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_split_equal_sum_l1627_162720


namespace NUMINAMATH_CALUDE_pages_copied_l1627_162770

/-- Given the cost of 7 cents for 5 pages, prove that $35 allows copying 2500 pages. -/
theorem pages_copied (cost_per_5_pages : ℚ) (total_dollars : ℚ) : 
  cost_per_5_pages = 7 / 100 → 
  total_dollars = 35 → 
  (total_dollars * 100 * 5) / cost_per_5_pages = 2500 := by
  sorry

end NUMINAMATH_CALUDE_pages_copied_l1627_162770


namespace NUMINAMATH_CALUDE_dolphin_training_hours_l1627_162756

/-- Calculates the number of hours each trainer spends training dolphins -/
def trainer_hours (num_dolphins : ℕ) (hours_per_dolphin : ℕ) (num_trainers : ℕ) : ℕ :=
  (num_dolphins * hours_per_dolphin) / num_trainers

theorem dolphin_training_hours :
  trainer_hours 4 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_dolphin_training_hours_l1627_162756


namespace NUMINAMATH_CALUDE_trig_problem_l1627_162776

theorem trig_problem (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : Real.tan α - 1 / Real.tan α = -3/2) : 
  Real.tan α = -2 ∧ 
  (Real.cos (3*π/2 + α) - Real.cos (π - α)) / Real.sin (π/2 - α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l1627_162776


namespace NUMINAMATH_CALUDE_expression_equals_one_l1627_162717

theorem expression_equals_one : (2 * 6) / (12 * 14) * (3 * 12 * 14) / (2 * 6 * 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1627_162717


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1627_162777

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 4 - x^2 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop := y = 2*x ∨ y = -2*x

-- Theorem statement
theorem hyperbola_asymptote :
  ∀ x y : ℝ, hyperbola_equation x y → (∃ X Y : ℝ, X ≠ 0 ∧ asymptote_equation X Y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1627_162777


namespace NUMINAMATH_CALUDE_equal_piece_length_l1627_162761

theorem equal_piece_length 
  (total_length_cm : ℕ) 
  (total_pieces : ℕ) 
  (equal_pieces : ℕ) 
  (remaining_piece_length_mm : ℕ) :
  total_length_cm = 1165 →
  total_pieces = 154 →
  equal_pieces = 150 →
  remaining_piece_length_mm = 100 →
  ∃ (equal_piece_length_mm : ℕ),
    equal_piece_length_mm = 75 ∧
    total_length_cm * 10 = equal_piece_length_mm * equal_pieces + 
      remaining_piece_length_mm * (total_pieces - equal_pieces) :=
by sorry

end NUMINAMATH_CALUDE_equal_piece_length_l1627_162761


namespace NUMINAMATH_CALUDE_tangent_line_at_one_max_value_condition_a_range_condition_l1627_162772

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x

-- Define e as the base of natural logarithms
def e : ℝ := Real.exp 1

-- Theorem 1: Tangent line equation when a = 1
theorem tangent_line_at_one (x y : ℝ) :
  f 1 1 = 1 → 2 * x - y - 1 = 0 ↔ y - 1 = 2 * (x - 1) :=
sorry

-- Theorem 2: Value of a when maximum of f(x) is -2
theorem max_value_condition (a : ℝ) :
  (∃ x > 0, ∀ y > 0, f a x ≥ f a y) ∧ (∃ x > 0, f a x = -2) → a = -e :=
sorry

-- Theorem 3: Range of a when a < 0 and f(x) ≤ g(x) for x ∈ [1,e]
theorem a_range_condition (a : ℝ) :
  a < 0 ∧ (∀ x ∈ Set.Icc 1 e, f a x ≤ g a x) →
  a ∈ Set.Icc ((1 - 2*e) / (e^2 - e)) 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_max_value_condition_a_range_condition_l1627_162772


namespace NUMINAMATH_CALUDE_min_people_liking_both_l1627_162708

theorem min_people_liking_both (total : ℕ) (mozart : ℕ) (beethoven : ℕ)
  (h1 : total = 150)
  (h2 : mozart = 130)
  (h3 : beethoven = 110)
  (h4 : mozart ≤ total)
  (h5 : beethoven ≤ total) :
  mozart + beethoven - total ≤ (min mozart beethoven) ∧
  (min mozart beethoven) = 90 :=
by sorry

end NUMINAMATH_CALUDE_min_people_liking_both_l1627_162708


namespace NUMINAMATH_CALUDE_value_swap_l1627_162701

theorem value_swap (a b : ℕ) (h1 : a = 1) (h2 : b = 2) :
  let c := a
  let a' := b
  let b' := c
  (a', b', c) = (2, 1, 1) := by sorry

end NUMINAMATH_CALUDE_value_swap_l1627_162701


namespace NUMINAMATH_CALUDE_max_value_theorem_l1627_162734

theorem max_value_theorem (x y z : ℝ) (h : 3 * x + 4 * y + 2 * z = 12) :
  ∃ (max : ℝ), max = 3 ∧ ∀ (a b c : ℝ), 3 * a + 4 * b + 2 * c = 12 →
    a^2 * b + a^2 * c + b * c^2 ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1627_162734


namespace NUMINAMATH_CALUDE_black_squares_56th_row_l1627_162767

/-- Represents the number of squares in a row of the geometric pattern -/
def squares_in_row (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- Represents the number of black squares in a row of the geometric pattern -/
def black_squares_in_row (n : ℕ) : ℕ := (squares_in_row n - 1) / 2

/-- Theorem stating that the 56th row contains 56 black squares -/
theorem black_squares_56th_row : black_squares_in_row 56 = 56 := by
  sorry


end NUMINAMATH_CALUDE_black_squares_56th_row_l1627_162767


namespace NUMINAMATH_CALUDE_special_quadrilateral_area_sum_l1627_162721

/-- A convex quadrilateral with specific side lengths and angle -/
structure ConvexQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  angleCDA : ℝ
  convex : AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0
  angleCondition : 0 < angleCDA ∧ angleCDA < π

/-- The area of the quadrilateral can be expressed in the form √a + b√c -/
def hasSpecialAreaForm (q : ConvexQuadrilateral) (a b c : ℕ) : Prop :=
  ∃ (area : ℝ), area = Real.sqrt a + b * Real.sqrt c ∧
  area = q.AB * q.BC * Real.sin q.angleCDA / 2 + q.CD * q.DA * Real.sin q.angleCDA / 2 ∧
  ∀ k : ℕ, k > 1 → (k * k ∣ a → k = 1) ∧ (k * k ∣ c → k = 1)

/-- Main theorem -/
theorem special_quadrilateral_area_sum (q : ConvexQuadrilateral) 
    (h1 : q.AB = 8) (h2 : q.BC = 4) (h3 : q.CD = 10) (h4 : q.DA = 10) 
    (h5 : q.angleCDA = π/3) (a b c : ℕ) (h6 : hasSpecialAreaForm q a b c) : 
    a + b + c = 259 := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_area_sum_l1627_162721


namespace NUMINAMATH_CALUDE_mersenne_divisibility_l1627_162730

theorem mersenne_divisibility (n : ℕ+) :
  (∃ m : ℕ+, (2^n.val - 1) ∣ (m.val^2 + 81)) ↔ ∃ k : ℕ, n.val = 2^k :=
sorry

end NUMINAMATH_CALUDE_mersenne_divisibility_l1627_162730


namespace NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l1627_162718

theorem circle_radius_from_longest_chord (c : ℝ) (h : c > 0) :
  (∃ (r : ℝ), r > 0 ∧ 2 * r = c) → c / 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_longest_chord_l1627_162718


namespace NUMINAMATH_CALUDE_total_area_is_60_l1627_162788

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The four rectangles that compose the figure -/
def rectangles : List Rectangle := [
  { width := 5, height := 5 },
  { width := 5, height := 3 },
  { width := 5, height := 2 },
  { width := 5, height := 2 }
]

/-- Theorem: The total area of the figure is 60 square units -/
theorem total_area_is_60 : 
  (rectangles.map Rectangle.area).sum = 60 := by sorry

end NUMINAMATH_CALUDE_total_area_is_60_l1627_162788


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l1627_162729

/-- A rectangular park with sides in ratio 3:2 and area 5766 sq m -/
structure RectangularPark where
  length : ℝ
  width : ℝ
  ratio_condition : length = 3/2 * width
  area_condition : length * width = 5766

/-- The cost of fencing in paise per meter -/
def fencing_cost_paise : ℝ := 50

/-- Theorem stating the cost of fencing the park -/
theorem fencing_cost_theorem (park : RectangularPark) : 
  (2 * (park.length + park.width) * fencing_cost_paise) / 100 = 155 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_theorem_l1627_162729


namespace NUMINAMATH_CALUDE_wood_piece_weight_relation_l1627_162781

/-- Represents a square piece of wood -/
structure WoodPiece where
  sideLength : ℝ
  weight : ℝ

/-- Theorem stating the relationship between two square wood pieces -/
theorem wood_piece_weight_relation 
  (piece1 piece2 : WoodPiece)
  (h1 : piece1.sideLength = 4)
  (h2 : piece1.weight = 16)
  (h3 : piece2.sideLength = 6)
  : piece2.weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_wood_piece_weight_relation_l1627_162781


namespace NUMINAMATH_CALUDE_largest_class_proof_l1627_162754

/-- The number of students in the largest class of a school with the following properties:
  - There are 5 classes
  - Each class has 2 students less than the previous class
  - The total number of students is 95
-/
def largest_class : ℕ := 23

theorem largest_class_proof :
  let classes := 5
  let student_difference := 2
  let total_students := 95
  let class_sizes := List.range classes |>.map (λ i => largest_class - i * student_difference)
  classes = 5 ∧
  student_difference = 2 ∧
  total_students = 95 ∧
  class_sizes.sum = total_students ∧
  largest_class ≥ 0 ∧
  (∀ i ∈ class_sizes, i ≥ 0) →
  largest_class = 23 :=
by sorry

end NUMINAMATH_CALUDE_largest_class_proof_l1627_162754


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1627_162727

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  1 / x + 1 / y = 3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1627_162727


namespace NUMINAMATH_CALUDE_triangle_c_coordinates_l1627_162715

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Defines the Euler line of a triangle -/
def Triangle.eulerLine (t : Triangle) : Line :=
  { a := 1, b := -1, c := 2 }

/-- Theorem: If triangle ABC has vertices A(2,0) and B(0,4), and its Euler line 
    is x-y+2=0, then the coordinates of C must be (-4,0) -/
theorem triangle_c_coordinates (t : Triangle) : 
  t.A = { x := 2, y := 0 } →
  t.B = { x := 0, y := 4 } →
  (t.eulerLine = { a := 1, b := -1, c := 2 }) →
  t.C = { x := -4, y := 0 } :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_c_coordinates_l1627_162715


namespace NUMINAMATH_CALUDE_math_club_probability_l1627_162786

def club_sizes : List Nat := [6, 9, 10]
def co_presidents_per_club : Nat := 3
def members_selected : Nat := 4

def probability_two_copresidents (n : Nat) : Rat :=
  (Nat.choose co_presidents_per_club 2 * Nat.choose (n - co_presidents_per_club) 2) /
  Nat.choose n members_selected

theorem math_club_probability : 
  (1 / 3 : Rat) * (club_sizes.map probability_two_copresidents).sum = 44 / 105 := by
  sorry

end NUMINAMATH_CALUDE_math_club_probability_l1627_162786


namespace NUMINAMATH_CALUDE_root_product_one_l1627_162764

theorem root_product_one (b c : ℝ) (hb : b > 0) (hc : c > 0) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, 
    (x₁^2 + 2*b*x₁ + c = 0) ∧ 
    (x₂^2 + 2*b*x₂ + c = 0) ∧ 
    (x₃^2 + 2*c*x₃ + b = 0) ∧ 
    (x₄^2 + 2*c*x₄ + b = 0) ∧ 
    (x₁ * x₂ * x₃ * x₄ = 1)) → 
  b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_product_one_l1627_162764


namespace NUMINAMATH_CALUDE_calculate_individual_tip_l1627_162723

/-- Calculates the individual tip amount for a group dining out -/
theorem calculate_individual_tip (julie_order : ℚ) (letitia_order : ℚ) (anton_order : ℚ) 
  (tip_rate : ℚ) (h1 : julie_order = 10) (h2 : letitia_order = 20) (h3 : anton_order = 30) 
  (h4 : tip_rate = 0.2) : 
  (julie_order + letitia_order + anton_order) * tip_rate / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_individual_tip_l1627_162723


namespace NUMINAMATH_CALUDE_britney_lemon_tea_l1627_162736

/-- The number of people sharing the lemon tea -/
def number_of_people : ℕ := 5

/-- The number of cups each person gets -/
def cups_per_person : ℕ := 2

/-- The total number of cups of lemon tea Britney brewed -/
def total_cups : ℕ := number_of_people * cups_per_person

theorem britney_lemon_tea : total_cups = 10 := by
  sorry

end NUMINAMATH_CALUDE_britney_lemon_tea_l1627_162736


namespace NUMINAMATH_CALUDE_smallest_a1_l1627_162722

/-- Given a sequence of positive real numbers {aₙ} where aₙ = 15aₙ₋₁ - 2n for all n > 1,
    the smallest possible value of a₁ is 29/98. -/
theorem smallest_a1 (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n > 1, a n = 15 * a (n - 1) - 2 * n) →
  ∀ x, (∀ n, a n > 0) → (∀ n > 1, a n = 15 * a (n - 1) - 2 * n) → a 1 ≥ x →
  x ≤ 29 / 98 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a1_l1627_162722


namespace NUMINAMATH_CALUDE_min_beans_betty_buys_l1627_162775

/-- The minimum number of pounds of beans Betty could buy given the conditions on rice and beans -/
theorem min_beans_betty_buys (r b : ℝ) 
  (h1 : r ≥ 4 + 2 * b) 
  (h2 : r ≤ 3 * b) : 
  b ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_beans_betty_buys_l1627_162775


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_singleton_two_l1627_162737

def A : Set ℝ := {-1, 2, 3}
def B : Set ℝ := {x : ℝ | x * (x - 3) < 0}

theorem A_intersect_B_eq_singleton_two : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_singleton_two_l1627_162737


namespace NUMINAMATH_CALUDE_problem_statement_l1627_162789

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (a * b ≤ 1) ∧ (2^a + 2^b ≥ 2 * Real.sqrt 2) ∧ (1/a + 4/b ≥ 9/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1627_162789


namespace NUMINAMATH_CALUDE_regular_dodecagon_diagonal_sum_l1627_162742

/-- A regular dodecagon -/
structure RegularDodecagon where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  radius_pos : radius > 0

/-- A diagonal in a regular dodecagon -/
inductive Diagonal
  | D1_7  -- Diagonal from vertex 1 to vertex 7
  | D1_3  -- Diagonal from vertex 1 to vertex 3
  | D1_11 -- Diagonal from vertex 1 to vertex 11

/-- The length of a diagonal in a regular dodecagon -/
def diagonalLength (d : RegularDodecagon) (diag : Diagonal) : ℝ :=
  match diag with
  | Diagonal.D1_7  => 2 * d.radius
  | Diagonal.D1_3  => d.radius
  | Diagonal.D1_11 => d.radius

/-- Theorem: In a regular dodecagon, there exist three diagonals where 
    the length of one diagonal equals the sum of the lengths of the other two -/
theorem regular_dodecagon_diagonal_sum (d : RegularDodecagon) :
  ∃ (d1 d2 d3 : Diagonal), 
    diagonalLength d d1 = diagonalLength d d2 + diagonalLength d d3 :=
sorry


end NUMINAMATH_CALUDE_regular_dodecagon_diagonal_sum_l1627_162742


namespace NUMINAMATH_CALUDE_base_h_equation_solution_l1627_162709

/-- Represents a number in base h --/
def BaseH (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

/-- The theorem statement --/
theorem base_h_equation_solution :
  ∃ (h : Nat), h > 1 ∧ 
    BaseH [8, 3, 7, 4] h + BaseH [6, 9, 2, 5] h = BaseH [1, 5, 3, 0, 9] h ∧
    h = 9 := by
  sorry

end NUMINAMATH_CALUDE_base_h_equation_solution_l1627_162709


namespace NUMINAMATH_CALUDE_larger_number_proof_l1627_162733

/-- Given two positive integers with the specified HCF and LCM factors, 
    prove that the larger of the two numbers is 3289 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf_condition : Nat.gcd a b = 23)
  (lcm_condition : ∃ k : ℕ+, Nat.lcm a b = 23 * 11 * 13 * 15^2 * k) :
  max a b = 3289 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1627_162733


namespace NUMINAMATH_CALUDE_least_positive_integer_modulo_solution_satisfies_congruence_twelve_is_least_positive_solution_l1627_162753

theorem least_positive_integer_modulo (x : ℕ) : x + 3001 ≡ 1723 [ZMOD 15] → x ≥ 12 := by
  sorry

theorem solution_satisfies_congruence : 12 + 3001 ≡ 1723 [ZMOD 15] := by
  sorry

theorem twelve_is_least_positive_solution : ∃! x : ℕ, x + 3001 ≡ 1723 [ZMOD 15] ∧ ∀ y : ℕ, y + 3001 ≡ 1723 [ZMOD 15] → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_modulo_solution_satisfies_congruence_twelve_is_least_positive_solution_l1627_162753


namespace NUMINAMATH_CALUDE_m_range_l1627_162716

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, x ≤ -1 → (m^2 - m) * 2^x - (1/2)^x < 1) → 
  -2 < m ∧ m < 3 := by
sorry

end NUMINAMATH_CALUDE_m_range_l1627_162716


namespace NUMINAMATH_CALUDE_quadratic_equations_properties_l1627_162700

theorem quadratic_equations_properties (b c : ℤ) (x₁ x₂ x₁' x₂' : ℤ) :
  (x₁^2 + b*x₁ + c = 0) →
  (x₂^2 + b*x₂ + c = 0) →
  (x₁'^2 + c*x₁' + b = 0) →
  (x₂'^2 + c*x₂' + b = 0) →
  (x₁ * x₂ > 0) →
  (x₁' * x₂' > 0) →
  (x₁ + x₂ = -b) →
  (x₁ * x₂ = c) →
  (x₁' + x₂' = -c) →
  (x₁' * x₂' = b) →
  (x₁ < 0 ∧ x₂ < 0) ∧
  (b - 1 ≤ c ∧ c ≤ b + 1) ∧
  ((b = 5 ∧ c = 6) ∨ (b = 6 ∧ c = 5) ∨ (b = 4 ∧ c = 4)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_properties_l1627_162700


namespace NUMINAMATH_CALUDE_division_problem_l1627_162746

theorem division_problem (dividend : Nat) (divisor : Nat) (remainder : Nat) (quotient : Nat) :
  dividend = 161 →
  divisor = 16 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 10 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1627_162746


namespace NUMINAMATH_CALUDE_greatest_number_jo_thinking_l1627_162783

theorem greatest_number_jo_thinking : ∃ n : ℕ,
  n < 100 ∧
  (∃ k : ℕ, n = 5 * k - 2) ∧
  (∃ m : ℕ, n = 9 * m - 4) ∧
  (∀ x : ℕ, x < 100 ∧ (∃ k : ℕ, x = 5 * k - 2) ∧ (∃ m : ℕ, x = 9 * m - 4) → x ≤ n) ∧
  n = 68 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_jo_thinking_l1627_162783


namespace NUMINAMATH_CALUDE_slope_range_l1627_162774

-- Define the circle F
def circle_F (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the trajectory of point P
def trajectory_P (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y^2 = 4*x) ∨ (x < 0 ∧ y = 0)

-- Define the line l
def line_l (k m x y : ℝ) : Prop := y = k*x + m

-- Define the condition for points A and B
def condition_AB (xA yA xB yB : ℝ) : Prop :=
  xA * xB + yA * yB = -4 ∧
  4 * Real.sqrt 6 ≤ Real.sqrt ((xB - xA)^2 + (yB - yA)^2) ∧
  Real.sqrt ((xB - xA)^2 + (yB - yA)^2) ≤ 4 * Real.sqrt 30

-- Theorem statement
theorem slope_range (k : ℝ) :
  (∃ m xA yA xB yB,
    circle_F 1 0 ∧
    trajectory_P xA yA ∧ trajectory_P xB yB ∧
    line_l k m xA yA ∧ line_l k m xB yB ∧
    condition_AB xA yA xB yB ∧
    xA > 0 ∧ xB > 0 ∧ xA ≠ xB) →
  (k ∈ Set.Icc (-1) (-1/2) ∨ k ∈ Set.Icc (1/2) 1) :=
sorry

end NUMINAMATH_CALUDE_slope_range_l1627_162774


namespace NUMINAMATH_CALUDE_equation_equivalence_l1627_162769

-- Define the original equation
def original_equation (x y : ℝ) : Prop := 2 * x - 3 * y - 4 = 0

-- Define the intercept form
def intercept_form (x y : ℝ) : Prop := x / 2 + y / (-4/3) = 1

-- Theorem stating the equivalence of the two forms
theorem equation_equivalence :
  ∀ x y : ℝ, original_equation x y ↔ intercept_form x y :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1627_162769


namespace NUMINAMATH_CALUDE_hugo_first_roll_four_given_win_l1627_162793

-- Define the number of players
def num_players : ℕ := 5

-- Define the number of sides on the die
def die_sides : ℕ := 6

-- Define Hugo's winning probability
def hugo_win_prob : ℚ := 1 / num_players

-- Define the probability of rolling a 4
def roll_four_prob : ℚ := 1 / die_sides

-- Define the probability of Hugo winning given he rolled a 4
def hugo_win_given_four_prob : ℚ := 256 / 1296

-- Theorem to prove
theorem hugo_first_roll_four_given_win (
  num_players : ℕ) (die_sides : ℕ) (hugo_win_prob : ℚ) 
  (roll_four_prob : ℚ) (hugo_win_given_four_prob : ℚ) :
  num_players = 5 ∧ die_sides = 6 ∧ 
  hugo_win_prob = 1 / num_players ∧
  roll_four_prob = 1 / die_sides ∧
  hugo_win_given_four_prob = 256 / 1296 →
  (roll_four_prob * hugo_win_given_four_prob) / hugo_win_prob = 40 / 243 :=
by sorry

end NUMINAMATH_CALUDE_hugo_first_roll_four_given_win_l1627_162793


namespace NUMINAMATH_CALUDE_cheryl_mms_after_dinner_l1627_162782

/-- The number of m&m's Cheryl had at the beginning -/
def initial_mms : ℕ := 25

/-- The number of m&m's Cheryl ate after lunch -/
def after_lunch : ℕ := 7

/-- The number of m&m's Cheryl gave to her sister -/
def given_to_sister : ℕ := 13

/-- The number of m&m's Cheryl ate after dinner -/
def after_dinner : ℕ := 5

theorem cheryl_mms_after_dinner : 
  initial_mms - after_lunch - after_dinner - given_to_sister = 0 :=
by sorry

end NUMINAMATH_CALUDE_cheryl_mms_after_dinner_l1627_162782


namespace NUMINAMATH_CALUDE_lisas_number_l1627_162766

theorem lisas_number : ∃ n : ℕ, 
  (1000 < n ∧ n < 3000) ∧ 
  150 ∣ n ∧ 
  45 ∣ n ∧ 
  (∀ m : ℕ, (1000 < m ∧ m < 3000) ∧ 150 ∣ m ∧ 45 ∣ m → n ≤ m) ∧
  n = 1350 := by sorry

end NUMINAMATH_CALUDE_lisas_number_l1627_162766


namespace NUMINAMATH_CALUDE_existence_of_property_P_one_third_l1627_162784

-- Define the property P(m) for a function f on an interval
def has_property_P (f : ℝ → ℝ) (m : ℝ) (D : Set ℝ) : Prop :=
  ∃ x₀ ∈ D, f x₀ = f (x₀ + m) ∧ x₀ + m ∈ D

-- Theorem statement
theorem existence_of_property_P_one_third
  (f : ℝ → ℝ) (h_cont : Continuous f) (h_eq : f 0 = f 2) :
  ∃ x₀ ∈ Set.Icc 0 (5/3), f x₀ = f (x₀ + 1/3) :=
by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_existence_of_property_P_one_third_l1627_162784


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_one_l1627_162704

/-- The equation of a line passing through (-1, -1) with slope 1 is y = x -/
theorem line_equation_through_point_with_slope_one :
  ∀ (x y : ℝ), (y + 1 = 1 * (x + 1)) ↔ (y = x) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_one_l1627_162704


namespace NUMINAMATH_CALUDE_min_value_theorem_l1627_162758

theorem min_value_theorem (a : ℝ) :
  ∃ (min : ℝ), min = 8 ∧
  ∀ (x y : ℝ), x > 0 → y = -x^2 + 3 * Real.log x →
  (a - x)^2 + (a + 2 - y)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1627_162758


namespace NUMINAMATH_CALUDE_fractional_part_An_bounds_l1627_162744

theorem fractional_part_An_bounds (n : ℕ+) :
  let An := (49 * n.val ^ 2 + 0.35 * n.val : ℝ).sqrt
  0.024 < An - ⌊An⌋ ∧ An - ⌊An⌋ < 0.025 := by
  sorry

end NUMINAMATH_CALUDE_fractional_part_An_bounds_l1627_162744


namespace NUMINAMATH_CALUDE_average_running_time_l1627_162725

/-- Represents the average running time for students in a specific grade --/
structure GradeRunningTime where
  grade : Nat
  avgTime : ℚ

/-- Represents the number of students in each grade relative to the number of fifth graders --/
structure GradeRatio where
  grade : Nat
  ratio : ℚ

theorem average_running_time 
  (third_grade : GradeRunningTime)
  (fourth_grade : GradeRunningTime)
  (fifth_grade : GradeRunningTime)
  (third_ratio : GradeRatio)
  (fourth_ratio : GradeRatio)
  (h1 : third_grade.grade = 3 ∧ third_grade.avgTime = 14)
  (h2 : fourth_grade.grade = 4 ∧ fourth_grade.avgTime = 18)
  (h3 : fifth_grade.grade = 5 ∧ fifth_grade.avgTime = 11)
  (h4 : third_ratio.grade = 3 ∧ third_ratio.ratio = 3)
  (h5 : fourth_ratio.grade = 4 ∧ fourth_ratio.ratio = 3/2)
  : (third_grade.avgTime * third_ratio.ratio + 
     fourth_grade.avgTime * fourth_ratio.ratio + 
     fifth_grade.avgTime) / 
    (third_ratio.ratio + fourth_ratio.ratio + 1) = 160/11 := by
  sorry

end NUMINAMATH_CALUDE_average_running_time_l1627_162725


namespace NUMINAMATH_CALUDE_f_comp_three_roots_l1627_162765

/-- A quadratic function with a parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- The theorem stating the condition for f(f(x)) to have exactly 3 distinct real roots -/
theorem f_comp_three_roots :
  ∀ c : ℝ, (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    f_comp c r₁ = 0 ∧ f_comp c r₂ = 0 ∧ f_comp c r₃ = 0) ↔ 
  c = (11 - Real.sqrt 13) / 2 :=
sorry

end NUMINAMATH_CALUDE_f_comp_three_roots_l1627_162765


namespace NUMINAMATH_CALUDE_total_pencils_is_60_l1627_162755

/-- The total number of pencils owned by 5 children -/
def total_pencils : ℕ :=
  let child1 := 6
  let child2 := 9
  let child3 := 12
  let child4 := 15
  let child5 := 18
  child1 + child2 + child3 + child4 + child5

/-- Theorem stating that the total number of pencils is 60 -/
theorem total_pencils_is_60 : total_pencils = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_is_60_l1627_162755


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1627_162771

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 2 + Complex.I) : z = 1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1627_162771


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l1627_162748

def meat_quantity : ℕ := 2
def meat_price_per_kg : ℕ := 82
def initial_money : ℕ := 180

theorem money_left_after_purchase : 
  initial_money - (meat_quantity * meat_price_per_kg) = 16 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l1627_162748


namespace NUMINAMATH_CALUDE_sphere_radius_l1627_162731

theorem sphere_radius (d h : ℝ) (h1 : d = 30) (h2 : h = 10) : 
  ∃ r : ℝ, r^2 = d^2 / 4 + h^2 ∧ r = 5 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_sphere_radius_l1627_162731


namespace NUMINAMATH_CALUDE_nanometers_to_meters_l1627_162703

-- Define the conversion factors
def nanometer_to_millimeter : ℝ := 1e-6
def millimeter_to_meter : ℝ := 1e-3

-- Define the given length in nanometers
def length_in_nanometers : ℝ := 3e10

-- State the theorem
theorem nanometers_to_meters :
  length_in_nanometers * nanometer_to_millimeter * millimeter_to_meter = 30 := by
  sorry

end NUMINAMATH_CALUDE_nanometers_to_meters_l1627_162703


namespace NUMINAMATH_CALUDE_binary_to_base4_conversion_l1627_162741

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem binary_to_base4_conversion :
  decimal_to_base4 (binary_to_decimal [false, true, false, false, true, true, false, true, true, false, true]) = [2, 3, 1, 2, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_base4_conversion_l1627_162741


namespace NUMINAMATH_CALUDE_bags_weight_after_removal_l1627_162707

/-- 
Given a bag of sugar weighing 16 kg and a bag of salt weighing 30 kg,
if 4 kg is removed from their combined weight, the resulting weight is 42 kg.
-/
theorem bags_weight_after_removal (sugar_weight salt_weight removal_weight : ℕ) 
  (h1 : sugar_weight = 16)
  (h2 : salt_weight = 30)
  (h3 : removal_weight = 4) :
  sugar_weight + salt_weight - removal_weight = 42 :=
by sorry

end NUMINAMATH_CALUDE_bags_weight_after_removal_l1627_162707


namespace NUMINAMATH_CALUDE_initial_average_is_16_l1627_162750

def initial_average_problem (A : ℝ) : Prop :=
  -- Define the sum of 6 initial observations
  let initial_sum := 6 * A
  -- Define the sum of 7 observations after adding the new one
  let new_sum := initial_sum + 9
  -- The new average is A - 1
  new_sum / 7 = A - 1

theorem initial_average_is_16 :
  ∃ A : ℝ, initial_average_problem A ∧ A = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_is_16_l1627_162750


namespace NUMINAMATH_CALUDE_alternating_sequence_sum_l1627_162780

def sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  let last_term := a₁ + (n - 1) * d
  let sum_of_pairs := ((n - 1) / 2) * (a₁ + last_term - d)
  if n % 2 = 0 then sum_of_pairs else sum_of_pairs + last_term

theorem alternating_sequence_sum :
  sequence_sum 2 3 19 = 29 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sequence_sum_l1627_162780
