import Mathlib

namespace NUMINAMATH_CALUDE_fliers_remaining_l1870_187077

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ) : 
  total = 2000 →
  morning_fraction = 1 / 10 →
  afternoon_fraction = 1 / 4 →
  (total - total * morning_fraction) * (1 - afternoon_fraction) = 1350 := by
sorry

end NUMINAMATH_CALUDE_fliers_remaining_l1870_187077


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1870_187015

theorem arithmetic_computation : 3 + 8 * 3 - 4 + 2^3 * 5 / 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1870_187015


namespace NUMINAMATH_CALUDE_inequality_solution_l1870_187036

/-- Given that the solution of the inequality 2x^2 - 6x + 4 < 0 is 1 < x < b, prove that b = 2 -/
theorem inequality_solution (b : ℝ) 
  (h : ∀ x : ℝ, 1 < x ∧ x < b ↔ 2 * x^2 - 6 * x + 4 < 0) : 
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1870_187036


namespace NUMINAMATH_CALUDE_solve_for_S_l1870_187056

theorem solve_for_S : ∃ S : ℚ, (1/2 : ℚ) * (1/7 : ℚ) * S = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ S = 70 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_S_l1870_187056


namespace NUMINAMATH_CALUDE_solution_sum_equals_23_l1870_187048

theorem solution_sum_equals_23 (x y a b c d : ℝ) : 
  (x + y = 5) →
  (2 * x * y = 5) →
  (∃ (sign : Bool), x = (a + if sign then b * Real.sqrt c else -b * Real.sqrt c) / d) →
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (d > 0) →
  (∀ k : ℕ, k > 1 → ¬(∃ (m n : ℤ), a * k = m * d ∧ b * k = n * d)) →
  (a + b + c + d = 23) := by
sorry

end NUMINAMATH_CALUDE_solution_sum_equals_23_l1870_187048


namespace NUMINAMATH_CALUDE_tv_price_calculation_l1870_187016

/-- The actual selling price of a television set given its cost price,
    markup percentage, and discount percentage. -/
def actual_selling_price (cost_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  cost_price * (1 + markup_percent) * discount_percent

/-- Theorem stating that for a television with cost price 'a',
    25% markup, and 70% discount, the actual selling price is 70%(1+25%)a. -/
theorem tv_price_calculation (a : ℝ) :
  actual_selling_price a 0.25 0.7 = 0.7 * (1 + 0.25) * a := by
  sorry

#check tv_price_calculation

end NUMINAMATH_CALUDE_tv_price_calculation_l1870_187016


namespace NUMINAMATH_CALUDE_remaining_payment_remaining_payment_specific_l1870_187079

/-- Given a product with a deposit, sales tax, and discount, calculate the remaining amount to be paid. -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) : ℝ :=
  let full_price := deposit / deposit_percentage
  let discounted_price := full_price * (1 - discount_rate)
  let final_price := discounted_price * (1 + sales_tax_rate)
  final_price - deposit

/-- Prove that the remaining payment for a product with given conditions is $733.20 -/
theorem remaining_payment_specific : 
  remaining_payment 80 0.1 0.07 0.05 = 733.20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_remaining_payment_specific_l1870_187079


namespace NUMINAMATH_CALUDE_palindrome_square_base_l1870_187007

theorem palindrome_square_base (r : ℕ) (x : ℕ) (p : ℕ) (h_r : r > 3) :
  let q := 2 * p
  let x_base_r := p * r^3 + p * r^2 + q * r + q
  let x_squared_base_r := x_base_r^2
  (∃ (a b c : ℕ), x_squared_base_r = a * r^6 + b * r^5 + c * r^4 + c * r^3 + c * r^2 + b * r + a) →
  (∃ (n : ℕ), n > 1 ∧ r = 3 * n^2) :=
by sorry

end NUMINAMATH_CALUDE_palindrome_square_base_l1870_187007


namespace NUMINAMATH_CALUDE_product_sum_in_base_l1870_187050

/-- Given a base b, convert a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Given a base b, convert a number from base 10 to base b -/
def fromBase10 (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Check if a number is valid in a given base -/
def isValidInBase (n : ℕ) (b : ℕ) : Prop := sorry

theorem product_sum_in_base (b : ℕ) : 
  (b > 1) →
  (isValidInBase 14 b) →
  (isValidInBase 17 b) →
  (isValidInBase 18 b) →
  (isValidInBase 4356 b) →
  (toBase10 14 b * toBase10 17 b * toBase10 18 b = toBase10 4356 b) →
  (fromBase10 (toBase10 14 b + toBase10 17 b + toBase10 18 b) b = 39) :=
by sorry

end NUMINAMATH_CALUDE_product_sum_in_base_l1870_187050


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l1870_187052

theorem smallest_integer_solution (x : ℤ) : 
  (∀ y : ℤ, |y^2 - 5*y + 6| = 14 → x ≤ y) ↔ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l1870_187052


namespace NUMINAMATH_CALUDE_quadratic_coefficient_positive_l1870_187039

theorem quadratic_coefficient_positive
  (a b c n : ℤ)
  (h_a_nonzero : a ≠ 0)
  (p : ℤ → ℤ)
  (h_p : ∀ x, p x = a * x^2 + b * x + c)
  (h_ineq : n < p n ∧ p n < p (p n) ∧ p (p n) < p (p (p n))) :
  0 < a :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_positive_l1870_187039


namespace NUMINAMATH_CALUDE_eight_solutions_of_g_fourth_composition_l1870_187002

/-- The function g(x) = x^2 - 3x -/
def g (x : ℝ) : ℝ := x^2 - 3*x

/-- The theorem stating that there are exactly 8 distinct real numbers d such that g(g(g(g(d)))) = 2 -/
theorem eight_solutions_of_g_fourth_composition :
  ∃! (s : Finset ℝ), (∀ d ∈ s, g (g (g (g d))) = 2) ∧ s.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_solutions_of_g_fourth_composition_l1870_187002


namespace NUMINAMATH_CALUDE_fixed_point_satisfies_equation_fixed_point_is_unique_l1870_187001

/-- The line equation passing through a fixed point for all real values of a -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x - y + 2 * a + 1 = 0

/-- The fixed point coordinates -/
def fixed_point : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the fixed point satisfies the line equation for all real a -/
theorem fixed_point_satisfies_equation :
  ∀ (a : ℝ), line_equation a (fixed_point.1) (fixed_point.2) :=
by sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_is_unique :
  ∀ (x y : ℝ), (∀ (a : ℝ), line_equation a x y) → (x, y) = fixed_point :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_satisfies_equation_fixed_point_is_unique_l1870_187001


namespace NUMINAMATH_CALUDE_harmonic_sets_theorem_l1870_187045

-- Define a circle
class Circle where
  -- Add any necessary properties for a circle

-- Define a point on a circle
class PointOnCircle (c : Circle) where
  -- Add any necessary properties for a point on a circle

-- Define a line
class Line where
  -- Add any necessary properties for a line

-- Define the property of lines intersecting at a single point
def intersectAtSinglePoint (l1 l2 l3 l4 : Line) : Prop :=
  sorry

-- Define a harmonic set of points
def isHarmonic {c : Circle} (A B C D : PointOnCircle c) : Prop :=
  sorry

-- Define the line connecting two points
def connectingLine {c : Circle} (P Q : PointOnCircle c) : Line :=
  sorry

theorem harmonic_sets_theorem
  {c : Circle}
  (A B C D A₁ B₁ C₁ D₁ : PointOnCircle c)
  (h_intersect : intersectAtSinglePoint
    (connectingLine A A₁)
    (connectingLine B B₁)
    (connectingLine C C₁)
    (connectingLine D D₁))
  (h_harmonic : isHarmonic A B C D ∨ isHarmonic A₁ B₁ C₁ D₁) :
  isHarmonic A B C D ∧ isHarmonic A₁ B₁ C₁ D₁ :=
sorry

end NUMINAMATH_CALUDE_harmonic_sets_theorem_l1870_187045


namespace NUMINAMATH_CALUDE_magic_square_sum_l1870_187027

/-- Represents a 3x3 magic square with given values and variables -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  sum : ℕ
  row1_eq : sum = 20 + e + 18
  row2_eq : sum = 15 + c + d
  row3_eq : sum = a + 25 + b
  col1_eq : sum = 20 + 15 + a
  col2_eq : sum = e + c + 25
  col3_eq : sum = 18 + d + b
  diag1_eq : sum = 20 + c + b
  diag2_eq : sum = a + c + 18

/-- Theorem: In the given magic square, d + e = 42 -/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 42 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l1870_187027


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l1870_187006

theorem rectangle_width_length_ratio (w : ℝ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 32 → w / 10 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l1870_187006


namespace NUMINAMATH_CALUDE_m_range_l1870_187054

-- Define the sets A and B
def A : Set ℝ := {x | x^2 < 16}
def B (m : ℝ) : Set ℝ := {x | x < m}

-- State the theorem
theorem m_range (m : ℝ) : A ∩ B m = A → m ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_m_range_l1870_187054


namespace NUMINAMATH_CALUDE_sandwich_change_l1870_187000

/-- Calculates the change received when buying a number of items at a given price and paying with a certain amount. -/
def calculate_change (num_items : ℕ) (price_per_item : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_items * price_per_item)

/-- Proves that buying 3 items at $5 each, paid with a $20 bill, results in $5 change. -/
theorem sandwich_change : calculate_change 3 5 20 = 5 := by
  sorry

#eval calculate_change 3 5 20

end NUMINAMATH_CALUDE_sandwich_change_l1870_187000


namespace NUMINAMATH_CALUDE_sisters_name_length_sisters_name_length_is_five_l1870_187035

theorem sisters_name_length (jonathan_first_name_length : ℕ) 
                             (jonathan_surname_length : ℕ) 
                             (sister_surname_length : ℕ) 
                             (total_letters : ℕ) : ℕ :=
  let jonathan_full_name_length := jonathan_first_name_length + jonathan_surname_length
  let sister_first_name_length := total_letters - jonathan_full_name_length - sister_surname_length
  sister_first_name_length

theorem sisters_name_length_is_five : 
  sisters_name_length 8 10 10 33 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sisters_name_length_sisters_name_length_is_five_l1870_187035


namespace NUMINAMATH_CALUDE_star_vertex_angle_formula_l1870_187089

/-- The angle measure at the vertices of a star formed by extending the sides of a regular n-sided polygon -/
def starVertexAngle (n : ℕ) : ℚ :=
  (n - 4) * 180 / n

/-- Theorem stating the angle measure at the vertices of a star formed by extending the sides of a regular n-sided polygon -/
theorem star_vertex_angle_formula (n : ℕ) (h : n > 2) :
  starVertexAngle n = (n - 4) * 180 / n :=
by sorry

end NUMINAMATH_CALUDE_star_vertex_angle_formula_l1870_187089


namespace NUMINAMATH_CALUDE_total_quantities_l1870_187004

theorem total_quantities (average : ℝ) (average_three : ℝ) (average_two : ℝ) : 
  average = 11 → average_three = 4 → average_two = 21.5 → 
  ∃ (n : ℕ), n = 5 ∧ 
    (n : ℝ) * average = 3 * average_three + 2 * average_two := by
  sorry

end NUMINAMATH_CALUDE_total_quantities_l1870_187004


namespace NUMINAMATH_CALUDE_sarah_trucks_l1870_187080

/-- The number of trucks Sarah had initially -/
def initial_trucks : ℕ := 51

/-- The number of trucks Sarah gave to Jeff -/
def trucks_given : ℕ := 13

/-- The number of trucks Sarah has now -/
def remaining_trucks : ℕ := initial_trucks - trucks_given

theorem sarah_trucks : remaining_trucks = 38 := by
  sorry

end NUMINAMATH_CALUDE_sarah_trucks_l1870_187080


namespace NUMINAMATH_CALUDE_parallelogram_angle_difference_l1870_187071

theorem parallelogram_angle_difference (a b : ℝ) : 
  a = 70 → -- smaller angle is 70 degrees
  a + b = 180 → -- adjacent angles are supplementary
  b - a = 40 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_angle_difference_l1870_187071


namespace NUMINAMATH_CALUDE_map_distance_conversion_l1870_187010

/-- Proves that given a map scale where 312 inches represents 136 km,
    a point 25 inches away on the map corresponds to approximately 10.9 km
    in actual distance. -/
theorem map_distance_conversion
  (map_distance : ℝ) (actual_distance : ℝ) (point_on_map : ℝ)
  (h1 : map_distance = 312)
  (h2 : actual_distance = 136)
  (h3 : point_on_map = 25) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧
  abs ((actual_distance / map_distance) * point_on_map - 10.9) < ε :=
sorry

end NUMINAMATH_CALUDE_map_distance_conversion_l1870_187010


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1870_187060

/-- The lateral surface area of a cone with base radius 3 and slant height 5 is 15π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 3  -- radius of the base
  let s : ℝ := 5  -- slant height
  let lateral_area := π * r * s  -- formula for lateral surface area of a cone
  lateral_area = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1870_187060


namespace NUMINAMATH_CALUDE_no_odd_faced_odd_edged_polyhedron_l1870_187070

/-- Represents a face of a polyhedron -/
structure Face where
  edges : Nat
  odd_edges : Odd edges

/-- Represents a polyhedron -/
structure Polyhedron where
  faces : List Face
  odd_faces : Odd faces.length

/-- Theorem stating that a polyhedron with an odd number of faces, 
    each having an odd number of edges, cannot exist -/
theorem no_odd_faced_odd_edged_polyhedron : 
  ¬ ∃ (p : Polyhedron), True := by sorry

end NUMINAMATH_CALUDE_no_odd_faced_odd_edged_polyhedron_l1870_187070


namespace NUMINAMATH_CALUDE_painting_selections_l1870_187096

/-- The number of traditional Chinese paintings -/
def traditional_paintings : Nat := 5

/-- The number of oil paintings -/
def oil_paintings : Nat := 2

/-- The number of watercolor paintings -/
def watercolor_paintings : Nat := 7

/-- The number of ways to choose one painting from each category -/
def one_from_each : Nat := traditional_paintings * oil_paintings * watercolor_paintings

/-- The number of ways to choose two paintings of different types -/
def two_different_types : Nat := 
  traditional_paintings * oil_paintings + 
  traditional_paintings * watercolor_paintings + 
  oil_paintings * watercolor_paintings

theorem painting_selections :
  one_from_each = 70 ∧ two_different_types = 59 := by sorry

end NUMINAMATH_CALUDE_painting_selections_l1870_187096


namespace NUMINAMATH_CALUDE_pi_half_irrational_l1870_187069

theorem pi_half_irrational : Irrational (Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l1870_187069


namespace NUMINAMATH_CALUDE_power_function_values_l1870_187022

-- Define the power function f
def f (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem power_function_values :
  (f 3 = 9) →
  (f 2 = 4) ∧ (∀ x, f (2*x + 1) = 4*x^2 + 4*x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_values_l1870_187022


namespace NUMINAMATH_CALUDE_complex_square_i_positive_l1870_187020

theorem complex_square_i_positive (a : ℝ) 
  (h : (Complex.I * (a + Complex.I)^2).re > 0 ∧ (Complex.I * (a + Complex.I)^2).im = 0) : 
  a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_square_i_positive_l1870_187020


namespace NUMINAMATH_CALUDE_intersection_and_union_subset_condition_l1870_187081

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 1}
def B : Set ℝ := {x | -3 ≤ x - 1 ∧ x - 1 ≤ 2}

-- Define set M
def M (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < 2*a + 2}

-- Theorem for part 1
theorem intersection_and_union :
  (A ∩ B = {x | 1 < x ∧ x ≤ 3}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x ≤ 1 ∨ x > 3}) := by sorry

-- Theorem for part 2
theorem subset_condition (a : ℝ) :
  M a ⊆ A ↔ a ≤ -3 ∨ a > 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_subset_condition_l1870_187081


namespace NUMINAMATH_CALUDE_sunscreen_discount_percentage_l1870_187005

/-- Calculate the discount percentage for Juanita's sunscreen purchase -/
theorem sunscreen_discount_percentage : 
  let bottles_per_year : ℕ := 12
  let cost_per_bottle : ℚ := 30
  let discounted_total_cost : ℚ := 252
  let original_total_cost : ℚ := bottles_per_year * cost_per_bottle
  let discount_amount : ℚ := original_total_cost - discounted_total_cost
  let discount_percentage : ℚ := (discount_amount / original_total_cost) * 100
  discount_percentage = 30 := by sorry

end NUMINAMATH_CALUDE_sunscreen_discount_percentage_l1870_187005


namespace NUMINAMATH_CALUDE_constant_term_is_180_l1870_187067

/-- The binomial expansion of (√x + 2/x²)^10 has its largest coefficient in the sixth term -/
axiom largest_coeff_sixth_term : ∃ k, k = 6 ∧ ∀ j, j ≠ k → 
  Nat.choose 10 (k-1) * 2^(k-1) ≥ Nat.choose 10 (j-1) * 2^(j-1)

/-- The constant term in the expansion of (√x + 2/x²)^10 -/
def constant_term : ℕ := Nat.choose 10 2 * 2^2

theorem constant_term_is_180 : constant_term = 180 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_180_l1870_187067


namespace NUMINAMATH_CALUDE_correct_algorithm_statement_l1870_187021

-- Define the concept of an algorithm
def Algorithm : Type := Unit

-- Define the property of being correct for an algorithm
def is_correct (a : Algorithm) : Prop := sorry

-- Define the property of yielding a definite result
def yields_definite_result (a : Algorithm) : Prop := sorry

-- Define the property of ending within a finite number of steps
def ends_in_finite_steps (a : Algorithm) : Prop := sorry

-- Define the property of having clear and unambiguous steps
def has_clear_steps (a : Algorithm) : Prop := sorry

-- Define the property of being unique for solving a certain type of problem
def is_unique_for_problem (a : Algorithm) : Prop := sorry

-- Theorem stating that the only correct statement is (2)
theorem correct_algorithm_statement :
  ∀ (a : Algorithm),
    is_correct a →
    (yields_definite_result a ∧
    ¬(¬(has_clear_steps a)) ∧
    ¬(is_unique_for_problem a) ∧
    ¬(ends_in_finite_steps a)) :=
by sorry

end NUMINAMATH_CALUDE_correct_algorithm_statement_l1870_187021


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l1870_187011

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^(2/3) + m * x + 1

-- State the theorem
theorem f_monotone_increasing (m : ℝ) :
  (∀ x, f m x = f m (-x)) →  -- f is an even function
  ∀ x ≥ 0, ∀ y ≥ x, f m x ≤ f m y :=  -- f is monotonically increasing on [0, +∞)
by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l1870_187011


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1870_187019

theorem quadratic_function_properties (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : a^2 + 2*a*c + c^2 < b^2) 
  (h3 : ∀ t : ℝ, a*(t+2)^2 + b*(t+2) + c = a*(-t+2)^2 + b*(-t+2) + c) 
  (h4 : a*(-2)^2 + b*(-2) + c = 2) :
  (∃ axis : ℝ, axis = 2 ∧ 
    ∀ x : ℝ, a*x^2 + b*x + c = a*(2*axis - x)^2 + b*(2*axis - x) + c) ∧ 
  (2/15 < a ∧ a < 2/7) := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1870_187019


namespace NUMINAMATH_CALUDE_polygon_with_1080_degrees_is_octagon_l1870_187068

/-- A polygon with interior angles summing to 1080° has 8 sides. -/
theorem polygon_with_1080_degrees_is_octagon :
  ∀ n : ℕ, (n - 2) * 180 = 1080 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_1080_degrees_is_octagon_l1870_187068


namespace NUMINAMATH_CALUDE_root_inequality_l1870_187018

theorem root_inequality (x₀ : ℝ) (h : x₀ > 0) (hroot : Real.log x₀ - 1 / x₀ = 0) :
  2^x₀ > x₀^(1/2) ∧ x₀^(1/2) > Real.log x₀ := by
  sorry

end NUMINAMATH_CALUDE_root_inequality_l1870_187018


namespace NUMINAMATH_CALUDE_milk_per_serving_in_cups_l1870_187053

/-- Proof that the amount of milk required per serving is 0.5 cups -/
theorem milk_per_serving_in_cups : 
  let ml_per_cup : ℝ := 250
  let total_people : ℕ := 8
  let servings_per_person : ℕ := 2
  let milk_cartons : ℕ := 2
  let ml_per_carton : ℝ := 1000
  
  let total_milk : ℝ := milk_cartons * ml_per_carton
  let total_servings : ℕ := total_people * servings_per_person
  let ml_per_serving : ℝ := total_milk / total_servings
  let cups_per_serving : ℝ := ml_per_serving / ml_per_cup

  cups_per_serving = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_milk_per_serving_in_cups_l1870_187053


namespace NUMINAMATH_CALUDE_volume_equality_l1870_187097

/-- The volume of the solid obtained by rotating the region bounded by x² = 4y, x² = -4y, x = 4, and x = -4 about the y-axis -/
def V₁ : ℝ := sorry

/-- The volume of the solid obtained by rotating the region defined by x² + y² ≤ 16, x² + (y-2)² ≥ 4, and x² + (y+2)² ≥ 4 about the y-axis -/
def V₂ : ℝ := sorry

/-- Theorem stating that V₁ equals V₂ -/
theorem volume_equality : V₁ = V₂ := by sorry

end NUMINAMATH_CALUDE_volume_equality_l1870_187097


namespace NUMINAMATH_CALUDE_sphere_cube_volume_comparison_l1870_187061

theorem sphere_cube_volume_comparison :
  ∀ (r a : ℝ), r > 0 → a > 0 →
  4 * π * r^2 = 6 * a^2 →
  (4/3) * π * r^3 > a^3 := by
sorry

end NUMINAMATH_CALUDE_sphere_cube_volume_comparison_l1870_187061


namespace NUMINAMATH_CALUDE_password_probability_l1870_187034

/-- Represents the probability of using password A in week k -/
def P (k : ℕ) : ℚ :=
  3/4 * (-1/3)^(k-1) + 1/4

/-- The problem statement -/
theorem password_probability : P 7 = 61/243 := by
  sorry

end NUMINAMATH_CALUDE_password_probability_l1870_187034


namespace NUMINAMATH_CALUDE_cube_edge_sum_l1870_187040

theorem cube_edge_sum (surface_area : ℝ) (h : surface_area = 150) :
  let side_length := Real.sqrt (surface_area / 6)
  12 * side_length = 60 := by sorry

end NUMINAMATH_CALUDE_cube_edge_sum_l1870_187040


namespace NUMINAMATH_CALUDE_kids_at_camp_l1870_187084

theorem kids_at_camp (total_kids : ℕ) (kids_at_home : ℕ) (h1 : total_kids = 1059955) (h2 : kids_at_home = 495718) :
  total_kids - kids_at_home = 564237 := by
  sorry

end NUMINAMATH_CALUDE_kids_at_camp_l1870_187084


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1870_187037

theorem sqrt_equation_solution :
  ∀ y : ℚ, (Real.sqrt (8 * y) / Real.sqrt (6 * (y - 2)) = 3) → y = 54 / 23 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1870_187037


namespace NUMINAMATH_CALUDE_cone_lateral_area_l1870_187044

/-- The lateral area of a cone with slant height 8 cm and base diameter 6 cm is 24π cm² -/
theorem cone_lateral_area (slant_height : ℝ) (base_diameter : ℝ) :
  slant_height = 8 →
  base_diameter = 6 →
  (1 / 2 : ℝ) * π * base_diameter * slant_height = 24 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_area_l1870_187044


namespace NUMINAMATH_CALUDE_stadium_attendance_l1870_187091

theorem stadium_attendance (total_start : ℕ) (girls_start : ℕ) 
  (h1 : total_start = 600)
  (h2 : girls_start = 240)
  (h3 : girls_start ≤ total_start) :
  let boys_start := total_start - girls_start
  let boys_left := boys_start / 4
  let girls_left := girls_start / 8
  let remaining := total_start - boys_left - girls_left
  remaining = 480 := by sorry

end NUMINAMATH_CALUDE_stadium_attendance_l1870_187091


namespace NUMINAMATH_CALUDE_restaurant_group_cost_l1870_187095

/-- Represents the cost structure and group composition at a restaurant -/
structure RestaurantGroup where
  adult_meal_cost : ℚ
  adult_drink_cost : ℚ
  adult_dessert_cost : ℚ
  kid_meal_cost : ℚ
  kid_drink_cost : ℚ
  kid_dessert_cost : ℚ
  total_people : ℕ
  num_kids : ℕ

/-- Calculates the total cost for a restaurant group -/
def total_cost (g : RestaurantGroup) : ℚ :=
  let num_adults := g.total_people - g.num_kids
  let adult_cost := num_adults * (g.adult_meal_cost + g.adult_drink_cost + g.adult_dessert_cost)
  let kid_cost := g.num_kids * (g.kid_meal_cost + g.kid_drink_cost + g.kid_dessert_cost)
  adult_cost + kid_cost

/-- Theorem stating that the total cost for the given group is $87.50 -/
theorem restaurant_group_cost :
  let g : RestaurantGroup := {
    adult_meal_cost := 7
    adult_drink_cost := 4
    adult_dessert_cost := 3
    kid_meal_cost := 0
    kid_drink_cost := 2
    kid_dessert_cost := 3/2
    total_people := 13
    num_kids := 9
  }
  total_cost g = 175/2 := by sorry

end NUMINAMATH_CALUDE_restaurant_group_cost_l1870_187095


namespace NUMINAMATH_CALUDE_fraction_value_l1870_187057

theorem fraction_value : (2024 - 1935)^2 / 225 = 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1870_187057


namespace NUMINAMATH_CALUDE_power_function_not_through_origin_l1870_187013

theorem power_function_not_through_origin (n : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (n^2 - 3*n + 3) * x^(n^2 - n - 2) ≠ 0) →
  n = 1 ∨ n = 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_not_through_origin_l1870_187013


namespace NUMINAMATH_CALUDE_min_flash_drives_l1870_187072

theorem min_flash_drives (total_files : ℕ) (drive_capacity : ℚ)
  (files_0_9MB : ℕ) (files_0_8MB : ℕ) (files_0_6MB : ℕ) :
  total_files = files_0_9MB + files_0_8MB + files_0_6MB →
  drive_capacity = 2.88 →
  files_0_9MB = 5 →
  files_0_8MB = 18 →
  files_0_6MB = 17 →
  (∃ min_drives : ℕ, 
    min_drives = 13 ∧
    min_drives * drive_capacity ≥ 
      (files_0_9MB * 0.9 + files_0_8MB * 0.8 + files_0_6MB * 0.6) ∧
    ∀ n : ℕ, n < min_drives → 
      n * drive_capacity < 
        (files_0_9MB * 0.9 + files_0_8MB * 0.8 + files_0_6MB * 0.6)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_flash_drives_l1870_187072


namespace NUMINAMATH_CALUDE_diamond_two_neg_five_l1870_187049

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a * b^2 - b + 1

-- Theorem statement
theorem diamond_two_neg_five : diamond 2 (-5) = 56 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_neg_five_l1870_187049


namespace NUMINAMATH_CALUDE_water_bottle_shortage_l1870_187099

/-- Represents the water bottle consumption during a soccer match --/
structure WaterBottleConsumption where
  initial_bottles : ℕ
  first_break_players : ℕ
  first_break_bottles_per_player : ℕ
  second_break_players : ℕ
  second_break_bottles_per_player : ℕ
  second_break_extra_bottles : ℕ
  third_break_players : ℕ
  third_break_bottles_per_player : ℕ

/-- Calculates the shortage of water bottles after the match --/
def calculate_shortage (consumption : WaterBottleConsumption) : ℤ :=
  let total_used := 
    consumption.first_break_players * consumption.first_break_bottles_per_player +
    consumption.second_break_players * consumption.second_break_bottles_per_player +
    consumption.second_break_extra_bottles +
    consumption.third_break_players * consumption.third_break_bottles_per_player
  consumption.initial_bottles - total_used

/-- Theorem stating that there is a shortage of 4 bottles given the match conditions --/
theorem water_bottle_shortage : 
  ∃ (consumption : WaterBottleConsumption), 
    consumption.initial_bottles = 48 ∧
    consumption.first_break_players = 11 ∧
    consumption.first_break_bottles_per_player = 2 ∧
    consumption.second_break_players = 14 ∧
    consumption.second_break_bottles_per_player = 1 ∧
    consumption.second_break_extra_bottles = 4 ∧
    consumption.third_break_players = 12 ∧
    consumption.third_break_bottles_per_player = 1 ∧
    calculate_shortage consumption = -4 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_shortage_l1870_187099


namespace NUMINAMATH_CALUDE_permutation_exists_16_no_permutation_exists_15_l1870_187047

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_valid_permutation (perm : List ℕ) (max_sum : ℕ) : Prop :=
  perm.length = numbers.length ∧
  perm.toFinset = numbers.toFinset ∧
  ∀ i, i + 2 < perm.length → perm[i]! + perm[i+1]! + perm[i+2]! ≤ max_sum

theorem permutation_exists_16 : ∃ perm, is_valid_permutation perm 16 :=
sorry

theorem no_permutation_exists_15 : ¬∃ perm, is_valid_permutation perm 15 :=
sorry

end NUMINAMATH_CALUDE_permutation_exists_16_no_permutation_exists_15_l1870_187047


namespace NUMINAMATH_CALUDE_total_ants_count_l1870_187032

/-- The total number of ants employed for all tasks in the construction site. -/
def total_ants : ℕ :=
  let red_carrying := 413
  let black_carrying := 487
  let yellow_carrying := 360
  let red_digging := 356
  let black_digging := 518
  let green_digging := 250
  let red_assembling := 298
  let black_assembling := 392
  let blue_assembling := 200
  let black_food := black_carrying / 4
  red_carrying + black_carrying + yellow_carrying +
  red_digging + black_digging + green_digging +
  red_assembling + black_assembling + blue_assembling -
  black_food

/-- Theorem stating that the total number of ants employed for all tasks is 3153. -/
theorem total_ants_count : total_ants = 3153 := by
  sorry

end NUMINAMATH_CALUDE_total_ants_count_l1870_187032


namespace NUMINAMATH_CALUDE_probability_at_least_one_vowel_l1870_187086

/-- The probability of picking at least one vowel from two sets of letters -/
theorem probability_at_least_one_vowel (set1 set2 : Finset Char) 
  (vowels1 vowels2 : Finset Char) : 
  set1.card = 6 →
  set2.card = 6 →
  vowels1 ⊆ set1 →
  vowels2 ⊆ set2 →
  vowels1.card = 2 →
  vowels2.card = 1 →
  (set1.card * set2.card : ℚ)⁻¹ * 
    ((vowels1.card * set2.card) + (set1.card - vowels1.card) * vowels2.card) = 1/2 := by
  sorry

#check probability_at_least_one_vowel

end NUMINAMATH_CALUDE_probability_at_least_one_vowel_l1870_187086


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1870_187046

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1870_187046


namespace NUMINAMATH_CALUDE_three_zeros_sin_minus_one_l1870_187026

/-- The function f(x) = sin(ωx) - 1 has exactly 3 zeros in [0, 2π] iff ω ∈ [9/4, 13/4) -/
theorem three_zeros_sin_minus_one (ω : ℝ) : ω > 0 →
  (∃! (s : Finset ℝ), s.card = 3 ∧ (∀ x ∈ s, x ∈ Set.Icc 0 (2 * Real.pi) ∧ Real.sin (ω * x) = 1)) ↔
  ω ∈ Set.Icc (9 / 4) (13 / 4) := by
  sorry

#check three_zeros_sin_minus_one

end NUMINAMATH_CALUDE_three_zeros_sin_minus_one_l1870_187026


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l1870_187055

/-- Given a sequence {a_n} with S_n being the sum of its first n terms,
    if S_n^2 - 2S_n - a_nS_n + 1 = 0 for all positive integers n,
    then S_n = n / (n + 1) for all positive integers n. -/
theorem sequence_sum_theorem (a : ℕ+ → ℚ) (S : ℕ+ → ℚ)
    (h : ∀ n : ℕ+, S n ^ 2 - 2 * S n - a n * S n + 1 = 0) :
  ∀ n : ℕ+, S n = n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l1870_187055


namespace NUMINAMATH_CALUDE_zero_in_A_l1870_187025

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : (0 : ℝ) ∈ A := by sorry

end NUMINAMATH_CALUDE_zero_in_A_l1870_187025


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1870_187042

theorem inequality_system_solution_set :
  let S := {x : ℝ | (2/3 * (2*x + 5) > 2) ∧ (x - 2 < 0)}
  S = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1870_187042


namespace NUMINAMATH_CALUDE_fruit_punch_total_l1870_187076

theorem fruit_punch_total (orange_punch : ℝ) (cherry_punch : ℝ) (apple_juice : ℝ) : 
  orange_punch = 4.5 →
  cherry_punch = 2 * orange_punch →
  apple_juice = cherry_punch - 1.5 →
  orange_punch + cherry_punch + apple_juice = 21 := by
  sorry

end NUMINAMATH_CALUDE_fruit_punch_total_l1870_187076


namespace NUMINAMATH_CALUDE_mary_potatoes_l1870_187058

theorem mary_potatoes (initial_potatoes : ℕ) : 
  initial_potatoes - 3 = 5 → initial_potatoes = 8 := by
  sorry

end NUMINAMATH_CALUDE_mary_potatoes_l1870_187058


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l1870_187078

/-- A quadratic equation x^2 + bx + 25 = 0 has at least one real root if and only if b ∈ (-∞, -10] ∪ [10, ∞) -/
theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l1870_187078


namespace NUMINAMATH_CALUDE_range_of_g_range_of_a_l1870_187033

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 1| - |x - 2|

-- Theorem for the range of g(x)
theorem range_of_g : Set.range g = Set.Icc (-1 : ℝ) 1 := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, g x < a^2 + a + 1) ↔ (a < -1 ∨ a > 1) := by sorry

end NUMINAMATH_CALUDE_range_of_g_range_of_a_l1870_187033


namespace NUMINAMATH_CALUDE_family_ages_solution_l1870_187098

/-- Represents the ages of a family with two parents and two children -/
structure FamilyAges where
  father : ℕ
  mother : ℕ
  older_son : ℕ
  younger_son : ℕ

/-- The conditions of the family ages problem -/
def family_ages_conditions (ages : FamilyAges) : Prop :=
  ages.father = ages.mother + 3 ∧
  ages.older_son = ages.younger_son + 4 ∧
  ages.father + ages.mother + ages.older_son + ages.younger_son = 81 ∧
  ages.father + ages.mother + ages.older_son + max (ages.younger_son - 5) 0 = 62

/-- The theorem stating the solution to the family ages problem -/
theorem family_ages_solution :
  ∃ (ages : FamilyAges), family_ages_conditions ages ∧
    ages.father = 36 ∧ ages.mother = 33 ∧ ages.older_son = 8 ∧ ages.younger_son = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_family_ages_solution_l1870_187098


namespace NUMINAMATH_CALUDE_new_average_after_deductions_l1870_187041

def consecutive_integers (start : ℤ) : List ℤ :=
  List.range 10 |>.map (fun i => start + i)

def deduct_sequence (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => (n - 1) - i)

theorem new_average_after_deductions (start : ℤ) :
  let original := consecutive_integers start
  let deductions := deduct_sequence 10
  let new_list := List.zipWith (· - ·) original deductions
  (List.sum original) / 10 = 11 →
  (List.sum new_list) / 10 = 7 := by
sorry

end NUMINAMATH_CALUDE_new_average_after_deductions_l1870_187041


namespace NUMINAMATH_CALUDE_tiffany_bag_difference_l1870_187051

/-- Calculates the difference in bags between Tuesday and Monday after giving away some bags -/
def bagDifference (mondayBags tuesdayFound givenAway : ℕ) : ℕ :=
  (mondayBags + tuesdayFound - givenAway) - mondayBags

theorem tiffany_bag_difference :
  bagDifference 7 12 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_bag_difference_l1870_187051


namespace NUMINAMATH_CALUDE_rational_sum_theorem_l1870_187059

theorem rational_sum_theorem (a b c : ℚ) 
  (h1 : a * b * c < 0) 
  (h2 : a + b + c = 0) : 
  (a - b - c) / abs a + (b - c - a) / abs b + (c - a - b) / abs c = 2 := by
  sorry

end NUMINAMATH_CALUDE_rational_sum_theorem_l1870_187059


namespace NUMINAMATH_CALUDE_fraction_representation_of_naturals_l1870_187009

theorem fraction_representation_of_naturals (n : ℕ) :
  ∃ x y : ℕ, n = x^3 / y^4 :=
sorry

end NUMINAMATH_CALUDE_fraction_representation_of_naturals_l1870_187009


namespace NUMINAMATH_CALUDE_real_root_of_complex_quadratic_l1870_187030

theorem real_root_of_complex_quadratic (k : ℝ) (a : ℝ) :
  (∃ x : ℂ, x^2 + (k + 2*I)*x + (2 : ℂ) + k*I = 0) →
  (a^2 + (k + 2*I)*a + (2 : ℂ) + k*I = 0) →
  (a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_real_root_of_complex_quadratic_l1870_187030


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1870_187075

theorem fraction_sum_equality : (2 : ℚ) / 15 + 4 / 20 + 5 / 45 = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1870_187075


namespace NUMINAMATH_CALUDE_ellipse_properties_l1870_187029

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of point M -/
def point_M : ℝ × ℝ := (0, 2)

/-- Theorem stating the properties of the ellipse and its related points -/
theorem ellipse_properties :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  (∀ x y, ellipse_C x y a b ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ x y, (∃ x₁ y₁, ellipse_C x₁ y₁ a b ∧ x = (x₁ + point_M.1) / 2 ∧ y = (y₁ + point_M.2) / 2) ↔
    x^2 / 2 + (y - 1)^2 = 1) ∧
  (∀ k₁ k₂ x₁ y₁ x₂ y₂,
    ellipse_C x₁ y₁ a b ∧ ellipse_C x₂ y₂ a b ∧
    k₁ = (y₁ - point_M.2) / (x₁ - point_M.1) ∧
    k₂ = (y₂ - point_M.2) / (x₂ - point_M.1) ∧
    k₁ + k₂ = 8 →
    ∃ t, (1 - t) * x₁ + t * x₂ = -1/2 ∧ (1 - t) * y₁ + t * y₂ = -2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1870_187029


namespace NUMINAMATH_CALUDE_partition_condition_l1870_187090

/-- A partition of ℕ* into n sets satisfying the given conditions -/
structure Partition (a : ℝ) where
  n : ℕ+
  sets : Fin n → Set ℕ+
  disjoint : ∀ i j, i ≠ j → Disjoint (sets i) (sets j)
  cover : (⋃ i, sets i) = Set.univ
  infinite : ∀ i, Set.Infinite (sets i)
  difference : ∀ i x y, x ∈ sets i → y ∈ sets i → x > y → x - y ≥ a ^ (i : ℕ)

/-- The main theorem -/
theorem partition_condition (a : ℝ) : 
  (∃ p : Partition a, True) → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_partition_condition_l1870_187090


namespace NUMINAMATH_CALUDE_family_c_members_l1870_187012

/-- Represents the number of members in each family in Indira Nagar --/
structure FamilyMembers where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat

/-- The initial number of family members before some left for the hostel --/
def initial_members : FamilyMembers := {
  a := 7,
  b := 8,
  c := 10,  -- This is what we want to prove
  d := 13,
  e := 6,
  f := 10
}

/-- The number of family members after one member from each family left for the hostel --/
def members_after_hostel (fm : FamilyMembers) : FamilyMembers :=
  { a := fm.a - 1,
    b := fm.b - 1,
    c := fm.c - 1,
    d := fm.d - 1,
    e := fm.e - 1,
    f := fm.f - 1 }

/-- The total number of families --/
def num_families : Nat := 6

/-- Theorem stating that the initial number of members in family c was 10 --/
theorem family_c_members :
  (members_after_hostel initial_members).a +
  (members_after_hostel initial_members).b +
  (members_after_hostel initial_members).c +
  (members_after_hostel initial_members).d +
  (members_after_hostel initial_members).e +
  (members_after_hostel initial_members).f =
  8 * num_families :=
by sorry

end NUMINAMATH_CALUDE_family_c_members_l1870_187012


namespace NUMINAMATH_CALUDE_only_prime_alternating_base14_l1870_187094

/-- Represents a number in base 14 with alternating 1s and 0s -/
def alternating_base14 (n : ℕ) : ℕ :=
  (14^(2*n) - 1) / 195

/-- Checks if a number is prime -/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, 1 < m → m < p → ¬(p % m = 0)

theorem only_prime_alternating_base14 :
  ∀ n : ℕ, is_prime (alternating_base14 n) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_only_prime_alternating_base14_l1870_187094


namespace NUMINAMATH_CALUDE_sum_squares_first_12_base6_l1870_187064

-- Define a function to convert a number from base 10 to base 6
def toBase6 (n : ℕ) : List ℕ := sorry

-- Define a function to square a number
def square (n : ℕ) : ℕ := n * n

-- Define a function to sum a list of numbers in base 6
def sumBase6 (list : List (List ℕ)) : List ℕ := sorry

-- Main theorem
theorem sum_squares_first_12_base6 : 
  sumBase6 (List.map (λ n => toBase6 (square n)) (List.range 12)) = [5, 1, 5, 0, 1] := by sorry

end NUMINAMATH_CALUDE_sum_squares_first_12_base6_l1870_187064


namespace NUMINAMATH_CALUDE_sum_of_squares_l1870_187093

theorem sum_of_squares (a b : ℕ+) (h : a.val^2 + 2*a.val*b.val - 3*b.val^2 - 41 = 0) : 
  a.val^2 + b.val^2 = 221 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1870_187093


namespace NUMINAMATH_CALUDE_simultaneous_inequalities_solution_l1870_187074

theorem simultaneous_inequalities_solution (x : ℝ) :
  (x^2 - 8*x + 12 < 0 ∧ 2*x - 4 > 0) ↔ (x > 2 ∧ x < 6) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_inequalities_solution_l1870_187074


namespace NUMINAMATH_CALUDE_anthony_jim_shoe_difference_l1870_187024

-- Define the number of shoe pairs for each person
def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

-- Theorem statement
theorem anthony_jim_shoe_difference :
  anthony_shoes - jim_shoes = 2 := by
  sorry

end NUMINAMATH_CALUDE_anthony_jim_shoe_difference_l1870_187024


namespace NUMINAMATH_CALUDE_deepak_age_l1870_187085

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1870_187085


namespace NUMINAMATH_CALUDE_power_of_three_mod_ten_l1870_187073

theorem power_of_three_mod_ten : 3^19 % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_power_of_three_mod_ten_l1870_187073


namespace NUMINAMATH_CALUDE_solve_equation_l1870_187062

theorem solve_equation (x : ℚ) (h : x - 3*x + 5*x = 200) : x = 200/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1870_187062


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l1870_187066

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_lines 
  (m n : Line) (α β : Plane) :
  perpendicular m α → 
  parallel_lines m n → 
  parallel_line_plane n β → 
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_lines_l1870_187066


namespace NUMINAMATH_CALUDE_expression_value_l1870_187063

theorem expression_value (a b c d : ℤ) 
  (ha : a = 15) (hb : b = 19) (hc : c = 3) (hd : d = 2) : 
  (a - (b - c)) - ((a - b) - c + d) = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1870_187063


namespace NUMINAMATH_CALUDE_haley_music_files_l1870_187083

theorem haley_music_files :
  ∀ (initial_music_files : ℕ),
    initial_music_files + 42 - 11 = 58 →
    initial_music_files = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_haley_music_files_l1870_187083


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1870_187087

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_9 < 0 and a_1 + a_18 > 0, then a_10 > 0 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a9_neg : a 9 < 0)
  (h_sum_pos : a 1 + a 18 > 0) : 
  a 10 > 0 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1870_187087


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l1870_187008

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem monotonic_increasing_interval (ω : ℝ) (h_pos : ω > 0) (h_period : ∀ x : ℝ, f ω (x + π / ω) = f ω x) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * π - π / 12) (k * π + 5 * π / 12)) := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l1870_187008


namespace NUMINAMATH_CALUDE_traces_bag_weight_is_two_l1870_187092

/-- The weight of one of Trace's shopping bags -/
def traces_bag_weight (
  trace_bags : ℕ
  ) (gordon_bags : ℕ
  ) (gordon_bag1_weight : ℕ
  ) (gordon_bag2_weight : ℕ
  ) (lola_bags : ℕ
  ) : ℕ :=
  sorry

theorem traces_bag_weight_is_two :
  ∀ (trace_bags : ℕ)
    (gordon_bags : ℕ)
    (gordon_bag1_weight : ℕ)
    (gordon_bag2_weight : ℕ)
    (lola_bags : ℕ),
  trace_bags = 5 →
  gordon_bags = 2 →
  gordon_bag1_weight = 3 →
  gordon_bag2_weight = 7 →
  lola_bags = 4 →
  trace_bags * traces_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags = 
    gordon_bag1_weight + gordon_bag2_weight →
  (gordon_bag1_weight + gordon_bag2_weight) / (3 * lola_bags) = 
    (gordon_bag1_weight + gordon_bag2_weight) / 3 - 1 →
  traces_bag_weight trace_bags gordon_bags gordon_bag1_weight gordon_bag2_weight lola_bags = 2 :=
by sorry

end NUMINAMATH_CALUDE_traces_bag_weight_is_two_l1870_187092


namespace NUMINAMATH_CALUDE_probability_of_five_ones_l1870_187014

def num_dice : ℕ := 15
def num_ones : ℕ := 5
def sides_on_die : ℕ := 6

theorem probability_of_five_ones :
  (Nat.choose num_dice num_ones : ℚ) * (1 / sides_on_die : ℚ)^num_ones * (1 - 1 / sides_on_die : ℚ)^(num_dice - num_ones) =
  (Nat.choose 15 5 : ℚ) * (1 / 6 : ℚ)^5 * (5 / 6 : ℚ)^10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_five_ones_l1870_187014


namespace NUMINAMATH_CALUDE_arcade_spend_example_l1870_187082

/-- Calculates the total amount spent at an arcade given the time spent and cost per interval. -/
def arcade_spend (hours : ℕ) (cost_per_interval : ℚ) (interval_minutes : ℕ) : ℚ :=
  let total_minutes : ℕ := hours * 60
  let num_intervals : ℕ := total_minutes / interval_minutes
  ↑num_intervals * cost_per_interval

/-- Proves that spending 3 hours at an arcade using $0.50 every 6 minutes results in a total spend of $15. -/
theorem arcade_spend_example : arcade_spend 3 (1/2) 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arcade_spend_example_l1870_187082


namespace NUMINAMATH_CALUDE_prime_square_mod_240_l1870_187003

theorem prime_square_mod_240 (p : Nat) (h_prime : Nat.Prime p) (h_gt_5 : p > 5) :
  ∃ (r₁ r₂ : Nat), r₁ ≠ r₂ ∧ r₁ < 240 ∧ r₂ < 240 ∧
  ∀ (q : Nat), Nat.Prime q → q > 5 → (q^2 % 240 = r₁ ∨ q^2 % 240 = r₂) :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_240_l1870_187003


namespace NUMINAMATH_CALUDE_amanda_grass_seed_bags_l1870_187023

/-- The number of bags of grass seed needed for a specific lot -/
def grassSeedBags (lotLength lotWidth concreteLength concreteWidth bagCoverage : ℕ) : ℕ :=
  let totalArea := lotLength * lotWidth
  let concreteArea := concreteLength * concreteWidth
  let grassArea := totalArea - concreteArea
  (grassArea + bagCoverage - 1) / bagCoverage

theorem amanda_grass_seed_bags :
  grassSeedBags 120 60 40 40 56 = 100 := by
  sorry

end NUMINAMATH_CALUDE_amanda_grass_seed_bags_l1870_187023


namespace NUMINAMATH_CALUDE_f_at_2_l1870_187065

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem f_at_2 : f 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l1870_187065


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1870_187017

theorem not_sufficient_nor_necessary (x y : ℝ) : 
  (∃ a b : ℝ, a > b ∧ ¬(|a| > |b|)) ∧ 
  (∃ c d : ℝ, |c| > |d| ∧ ¬(c > d)) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1870_187017


namespace NUMINAMATH_CALUDE_all_propositions_false_l1870_187088

-- Define the basic geometric objects
variable (Point : Type) [AddCommGroup Point] [Module ℝ Point]
variable (Line : Type)
variable (Plane : Type)

-- Define the geometric relations
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (intersection_plane : Plane → Plane → Line)

-- Theorem statement
theorem all_propositions_false :
  (∀ (a b : Line) (p : Plane), parallel_line a b → line_in_plane b p → parallel_line_plane a p) = False ∧
  (∀ (a b : Line) (α : Plane), parallel_line_plane a α → parallel_line_plane b α → parallel_line a b) = False ∧
  (∀ (a b : Line) (α β : Plane), parallel_line_plane a α → parallel_line_plane b β → perpendicular_plane α β → perpendicular_line a b) = False ∧
  (∀ (a b : Line) (α β : Plane), intersection_plane α β = a → parallel_line_plane b α → parallel_line b a) = False :=
sorry

end NUMINAMATH_CALUDE_all_propositions_false_l1870_187088


namespace NUMINAMATH_CALUDE_zack_classroom_count_l1870_187043

/-- The number of students in each classroom -/
structure ClassroomCounts where
  tina : ℕ
  maura : ℕ
  zack : ℕ

/-- The conditions of the problem -/
def classroom_conditions (c : ClassroomCounts) : Prop :=
  c.tina = c.maura ∧
  c.zack = (c.tina + c.maura) / 2 ∧
  c.tina + c.maura + c.zack = 69

/-- The theorem to prove -/
theorem zack_classroom_count (c : ClassroomCounts) :
  classroom_conditions c → c.zack = 23 := by
  sorry


end NUMINAMATH_CALUDE_zack_classroom_count_l1870_187043


namespace NUMINAMATH_CALUDE_student_weight_loss_l1870_187028

/-- The weight the student needs to lose to weigh twice as much as his sister -/
def weight_to_lose (total_weight student_weight : ℝ) : ℝ :=
  let sister_weight := total_weight - student_weight
  student_weight - 2 * sister_weight

theorem student_weight_loss (total_weight student_weight : ℝ) 
  (h1 : total_weight = 110)
  (h2 : student_weight = 75) :
  weight_to_lose total_weight student_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_student_weight_loss_l1870_187028


namespace NUMINAMATH_CALUDE_probability_both_truth_l1870_187031

theorem probability_both_truth (prob_A prob_B : ℝ) 
  (h_A : prob_A = 0.7) 
  (h_B : prob_B = 0.6) : 
  prob_A * prob_B = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_truth_l1870_187031


namespace NUMINAMATH_CALUDE_family_age_problem_l1870_187038

/-- Represents the ages of family members at different points in time -/
structure FamilyAges where
  grandpa : ℕ
  dad : ℕ
  xiaoming : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages1 ages2 : FamilyAges) : Prop :=
  ages1.grandpa = 2 * ages1.dad ∧
  ages1.xiaoming = 1 ∧
  ages2.dad = 8 * ages2.xiaoming ∧
  ages2.grandpa = 61

/-- The theorem to be proved -/
theorem family_age_problem (ages1 ages2 : FamilyAges) 
  (h : problem_conditions ages1 ages2) : 
  (∃ (ages3 : FamilyAges), 
    ages3.grandpa - ages3.xiaoming = 57 ∧ 
    ages3.grandpa = 20 * ages3.xiaoming ∧ 
    ages3.dad = 31) :=
sorry

end NUMINAMATH_CALUDE_family_age_problem_l1870_187038
