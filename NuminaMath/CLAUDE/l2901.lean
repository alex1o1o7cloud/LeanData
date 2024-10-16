import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2901_290185

/-- An arithmetic sequence is defined by its first term, common difference, and last term. -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ
  last : ℤ

/-- The number of terms in an arithmetic sequence. -/
def numTerms (seq : ArithmeticSequence) : ℤ :=
  (seq.last - seq.first) / seq.diff + 1

/-- Theorem: The arithmetic sequence with first term 13, common difference 3, and last term 73 has exactly 21 terms. -/
theorem arithmetic_sequence_terms : 
  let seq := ArithmeticSequence.mk 13 3 73
  numTerms seq = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l2901_290185


namespace NUMINAMATH_CALUDE_animals_remaining_l2901_290131

theorem animals_remaining (cows dogs : ℕ) : 
  cows = 2 * dogs →
  cows = 184 →
  (184 - 184 / 4) + (dogs - 3 * dogs / 4) = 161 := by
sorry

end NUMINAMATH_CALUDE_animals_remaining_l2901_290131


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2901_290167

theorem simplify_square_roots : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 49) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2901_290167


namespace NUMINAMATH_CALUDE_wage_increase_l2901_290107

theorem wage_increase (original_wage : ℝ) (increase_percentage : ℝ) (new_wage : ℝ) :
  original_wage = 20 →
  increase_percentage = 40 →
  new_wage = original_wage * (1 + increase_percentage / 100) →
  new_wage = 28 := by
sorry

end NUMINAMATH_CALUDE_wage_increase_l2901_290107


namespace NUMINAMATH_CALUDE_largest_x_quadratic_inequality_l2901_290140

theorem largest_x_quadratic_inequality :
  ∀ x : ℝ, x^2 - 10*x + 24 ≤ 0 → x ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_quadratic_inequality_l2901_290140


namespace NUMINAMATH_CALUDE_cos_a_minus_b_l2901_290171

theorem cos_a_minus_b (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
sorry

end NUMINAMATH_CALUDE_cos_a_minus_b_l2901_290171


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2901_290100

theorem product_of_three_numbers (a b c m : ℝ) 
  (sum_eq : a + b + c = 195)
  (m_eq_8a : m = 8 * a)
  (m_eq_b_minus_10 : m = b - 10)
  (m_eq_c_plus_10 : m = c + 10)
  (a_smallest : a < b ∧ a < c) :
  a * b * c = 95922 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2901_290100


namespace NUMINAMATH_CALUDE_decimal_multiplication_l2901_290182

theorem decimal_multiplication : (0.8 : ℝ) * 0.12 = 0.096 := by
  sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l2901_290182


namespace NUMINAMATH_CALUDE_line_segment_length_l2901_290123

/-- The length of a line segment with endpoints (1, 2) and (8, 6) is √65 -/
theorem line_segment_length : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (8, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 65 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l2901_290123


namespace NUMINAMATH_CALUDE_davids_english_marks_l2901_290195

/-- Represents the marks obtained in each subject --/
structure Marks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average of a list of natural numbers --/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

/-- Theorem stating that David's marks in English are 76 --/
theorem davids_english_marks (marks : Marks) :
  marks.mathematics = 65 →
  marks.physics = 82 →
  marks.chemistry = 67 →
  marks.biology = 85 →
  average [marks.english, marks.mathematics, marks.physics, marks.chemistry, marks.biology] = 75 →
  marks.english = 76 := by
  sorry


end NUMINAMATH_CALUDE_davids_english_marks_l2901_290195


namespace NUMINAMATH_CALUDE_circle_parabola_intersection_l2901_290183

/-- The circle and parabola intersect at exactly one point if and only if b = 1/4 -/
theorem circle_parabola_intersection (b : ℝ) : 
  (∃! p : ℝ × ℝ, p.1^2 + p.2^2 = 4*b^2 ∧ p.2 = p.1^2 - 2*b) ↔ b = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_circle_parabola_intersection_l2901_290183


namespace NUMINAMATH_CALUDE_customer_flow_solution_l2901_290130

/-- Represents the customer flow in a restaurant --/
def customer_flow (x y z : ℕ) : Prop :=
  let initial_customers : ℕ := 3
  let final_customers : ℕ := 8
  x = 2 * z ∧
  y = x - 3 ∧
  initial_customers + x + y - z = final_customers

/-- Theorem stating the solution to the customer flow problem --/
theorem customer_flow_solution :
  customer_flow 6 3 3 ∧ 6 + 3 = 9 := by
  sorry

#check customer_flow_solution

end NUMINAMATH_CALUDE_customer_flow_solution_l2901_290130


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2901_290169

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 9, 3]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -39/14, 51/14]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2901_290169


namespace NUMINAMATH_CALUDE_abc_inequality_l2901_290108

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_abc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2901_290108


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l2901_290105

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 4 →  -- angles are in ratio 5:4
  b = 80 :=  -- smaller angle is 80°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l2901_290105


namespace NUMINAMATH_CALUDE_fraction_difference_equality_l2901_290134

theorem fraction_difference_equality (x y : ℝ) : 
  let P := x^2 + y^2
  let Q := x - y
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = 4 * x * y / ((x^2 + y^2)^2 - (x - y)^2) :=
by sorry

end NUMINAMATH_CALUDE_fraction_difference_equality_l2901_290134


namespace NUMINAMATH_CALUDE_bookstore_comparison_l2901_290154

/-- Represents the amount to be paid at Bookstore A -/
def bookstore_A (x : ℝ) : ℝ := 0.8 * x

/-- Represents the amount to be paid at Bookstore B -/
def bookstore_B (x : ℝ) : ℝ := 0.6 * x + 40

theorem bookstore_comparison (x : ℝ) (h : x > 100) :
  (bookstore_A x < bookstore_B x ↔ x < 200) ∧
  (bookstore_A x > bookstore_B x ↔ x > 200) ∧
  (bookstore_A x = bookstore_B x ↔ x = 200) := by
  sorry

#check bookstore_comparison

end NUMINAMATH_CALUDE_bookstore_comparison_l2901_290154


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2901_290127

theorem unique_solution_for_equation :
  ∃! n : ℚ, (1 : ℚ) / (n + 2) + 2 / (n + 2) + n / (n + 2) = 3 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2901_290127


namespace NUMINAMATH_CALUDE_at_least_one_not_in_area_l2901_290152

theorem at_least_one_not_in_area (p q : Prop) : 
  (¬p ∨ ¬q) ↔ (∃ trainee, trainee = "A" ∧ ¬p ∨ trainee = "B" ∧ ¬q) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_not_in_area_l2901_290152


namespace NUMINAMATH_CALUDE_triangle_side_length_l2901_290149

/-- A triangle with circumradius 1 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (circumradius : ℝ)
  (h_circumradius : circumradius = 1)

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The circumcircle of a triangle -/
def circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- The circle passing through two points and the orthocenter -/
def circle_through_points_and_orthocenter (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- The center of a circle -/
def center (c : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem triangle_side_length (t : Triangle) :
  center (circle_through_points_and_orthocenter t) ∈ circumcircle t →
  distance t.A t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2901_290149


namespace NUMINAMATH_CALUDE_cherry_pies_count_l2901_290145

/-- Given a total number of pies and a ratio for distribution among three types,
    calculate the number of pies of the third type. -/
def calculate_third_type_pies (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) : ℕ :=
  let ratio_sum := ratio1 + ratio2 + ratio3
  let pies_per_part := total / ratio_sum
  ratio3 * pies_per_part

/-- Theorem stating that given 40 pies distributed in the ratio 2:5:3,
    the number of cherry pies (third type) is 12. -/
theorem cherry_pies_count :
  calculate_third_type_pies 40 2 5 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pies_count_l2901_290145


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l2901_290187

theorem complex_magnitude_one (z : ℂ) (h : (1 - z) / (1 + z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l2901_290187


namespace NUMINAMATH_CALUDE_base10_to_base12_144_l2901_290153

/-- Converts a digit to its base 12 representation -/
def toBase12Digit (n : ℕ) : String :=
  if n < 10 then toString n
  else if n = 10 then "A"
  else if n = 11 then "B"
  else ""

/-- Converts a number from base 10 to base 12 -/
def toBase12 (n : ℕ) : String :=
  let d1 := n / 12
  let d0 := n % 12
  toBase12Digit d1 ++ toBase12Digit d0

theorem base10_to_base12_144 :
  toBase12 144 = "B10" := by sorry

end NUMINAMATH_CALUDE_base10_to_base12_144_l2901_290153


namespace NUMINAMATH_CALUDE_unknown_number_value_l2901_290162

theorem unknown_number_value (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 45 * 25) : x = 49 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_value_l2901_290162


namespace NUMINAMATH_CALUDE_reflection_about_y_eq_neg_x_l2901_290163

def reflect_point (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem reflection_about_y_eq_neg_x (x y : ℝ) :
  reflect_point 4 (-3) = (3, -4) := by
  sorry

end NUMINAMATH_CALUDE_reflection_about_y_eq_neg_x_l2901_290163


namespace NUMINAMATH_CALUDE_part_one_part_two_l2901_290142

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |2*x + a|

-- Part I
theorem part_one : 
  {x : ℝ | f 1 x ≥ 5} = {x : ℝ | x ≤ -4/3 ∨ x ≥ 2} := by sorry

-- Part II
theorem part_two : 
  (∃ x₀ : ℝ, f a x₀ + |x₀ - 2| < 3) → -7 < a ∧ a < -1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2901_290142


namespace NUMINAMATH_CALUDE_initial_salmon_count_l2901_290125

theorem initial_salmon_count (current_count : ℕ) (increase_factor : ℕ) (initial_count : ℕ) : 
  current_count = 5500 →
  increase_factor = 10 →
  current_count = (increase_factor + 1) * initial_count →
  initial_count = 550 := by
  sorry

end NUMINAMATH_CALUDE_initial_salmon_count_l2901_290125


namespace NUMINAMATH_CALUDE_max_value_quadratic_function_l2901_290126

theorem max_value_quadratic_function :
  let f : ℝ → ℝ := fun x ↦ -x^2 + 2*x + 1
  ∃ (m : ℝ), m = 2 ∧ ∀ x, f x ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_function_l2901_290126


namespace NUMINAMATH_CALUDE_andrews_age_l2901_290114

theorem andrews_age :
  ∀ (a g : ℕ), 
    g = 15 * a →  -- Grandfather's age is fifteen times Andrew's age
    g - a = 70 →  -- Grandfather was 70 years old when Andrew was born
    a = 5         -- Andrew's age is 5
  := by sorry

end NUMINAMATH_CALUDE_andrews_age_l2901_290114


namespace NUMINAMATH_CALUDE_complete_square_equation_l2901_290158

theorem complete_square_equation : ∃ (a b c : ℤ), 
  (a > 0) ∧ 
  (∀ x : ℝ, 64 * x^2 + 80 * x - 81 = 0 ↔ (a * x + b)^2 = c) ∧
  (a = 8 ∧ b = 5 ∧ c = 106) := by
  sorry

end NUMINAMATH_CALUDE_complete_square_equation_l2901_290158


namespace NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l2901_290151

theorem sphere_radius_when_area_equals_volume (r : ℝ) (h : r > 0) :
  (4 * Real.pi * r^2) = (4/3 * Real.pi * r^3) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_when_area_equals_volume_l2901_290151


namespace NUMINAMATH_CALUDE_f_is_direct_proportion_l2901_290159

def f (x : ℝ) : ℝ := 3 * x

theorem f_is_direct_proportion : 
  (∀ x : ℝ, f x = 3 * x) ∧ 
  (f 0 = 0) ∧ 
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → f x / x = f y / y) := by
  sorry

end NUMINAMATH_CALUDE_f_is_direct_proportion_l2901_290159


namespace NUMINAMATH_CALUDE_equidistant_line_equations_l2901_290156

/-- A line passing through (1, 2) and equidistant from (0, 0) and (3, 1) -/
structure EquidistantLine where
  -- Coefficients of the line equation ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through (1, 2)
  passes_through : a + 2 * b + c = 0
  -- The line is equidistant from (0, 0) and (3, 1)
  equidistant : (c^2) / (a^2 + b^2) = (3*a + b + c)^2 / (a^2 + b^2)

/-- Theorem stating the two possible equations of the equidistant line -/
theorem equidistant_line_equations : 
  ∀ (l : EquidistantLine), (l.a = 1 ∧ l.b = -3 ∧ l.c = 5) ∨ (l.a = 3 ∧ l.b = 1 ∧ l.c = -5) :=
by sorry

end NUMINAMATH_CALUDE_equidistant_line_equations_l2901_290156


namespace NUMINAMATH_CALUDE_reasoning_is_analogical_l2901_290113

/-- A type representing different reasoning methods -/
inductive ReasoningMethod
  | Inductive
  | Analogical
  | Deductive
  | None

/-- A circle with radius R -/
structure Circle (R : ℝ) where
  radius : R > 0

/-- A rectangle inscribed in a circle -/
structure InscribedRectangle (R : ℝ) extends Circle R where
  width : ℝ
  height : ℝ
  inscribed : width^2 + height^2 ≤ 4 * R^2

/-- A sphere with radius R -/
structure Sphere (R : ℝ) where
  radius : R > 0

/-- A rectangular solid inscribed in a sphere -/
structure InscribedRectangularSolid (R : ℝ) extends Sphere R where
  length : ℝ
  width : ℝ
  height : ℝ
  inscribed : length^2 + width^2 + height^2 ≤ 4 * R^2

/-- Theorem about maximum area rectangle in a circle -/
axiom max_area_square_in_circle (R : ℝ) :
  ∀ (rect : InscribedRectangle R), rect.width * rect.height ≤ 2 * R^2

/-- The reasoning method used to deduce the theorem about cubes in spheres -/
def reasoning_method : ReasoningMethod := by sorry

/-- The main theorem stating that the reasoning method is analogical -/
theorem reasoning_is_analogical :
  reasoning_method = ReasoningMethod.Analogical := by sorry

end NUMINAMATH_CALUDE_reasoning_is_analogical_l2901_290113


namespace NUMINAMATH_CALUDE_distribute_five_students_three_classes_l2901_290170

/-- The number of ways to distribute students into classes -/
def distribute_students (total_students : ℕ) (num_classes : ℕ) (pre_assigned : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of distributions for the given problem -/
theorem distribute_five_students_three_classes : 
  distribute_students 5 3 1 = 56 := by sorry

end NUMINAMATH_CALUDE_distribute_five_students_three_classes_l2901_290170


namespace NUMINAMATH_CALUDE_negative_x_to_negative_k_is_positive_l2901_290102

theorem negative_x_to_negative_k_is_positive
  (x : ℝ) (k : ℤ) (hx : x < 0) (hk : k > 0) :
  -x^(-k) > 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_x_to_negative_k_is_positive_l2901_290102


namespace NUMINAMATH_CALUDE_reflection_sum_coordinates_l2901_290128

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflect a point over the y-axis -/
def reflect_over_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The sum of coordinates of two points -/
def sum_of_coordinates (p1 p2 : Point) : ℝ :=
  p1.x + p1.y + p2.x + p2.y

theorem reflection_sum_coordinates :
  let C : Point := { x := 3, y := 8 }
  let D : Point := reflect_over_y_axis C
  sum_of_coordinates C D = 16 := by
  sorry

end NUMINAMATH_CALUDE_reflection_sum_coordinates_l2901_290128


namespace NUMINAMATH_CALUDE_average_weight_increase_l2901_290166

/-- Proves that the increase in average weight when including a teacher is 400 grams -/
theorem average_weight_increase (num_students : Nat) (avg_weight_students : ℝ) (teacher_weight : ℝ) :
  num_students = 24 →
  avg_weight_students = 35 →
  teacher_weight = 45 →
  ((num_students + 1) * ((num_students * avg_weight_students + teacher_weight) / (num_students + 1)) -
   (num_students * avg_weight_students)) * 1000 = 400 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2901_290166


namespace NUMINAMATH_CALUDE_fresh_fruits_count_l2901_290155

/-- Calculates the number of fresh fruits left after sales and spoilage --/
def freshFruitsLeft (initialPineapples initialCoconuts soldPineapples soldCoconuts rottenPineapples spoiledCoconutPercentage : ℕ) : ℕ :=
  let remainingPineapples := initialPineapples - soldPineapples
  let freshPineapples := remainingPineapples - rottenPineapples
  let remainingCoconuts := initialCoconuts - soldCoconuts
  let spoiledCoconuts := (remainingCoconuts * spoiledCoconutPercentage + 99) / 100  -- Round up
  let freshCoconuts := remainingCoconuts - spoiledCoconuts
  freshPineapples + freshCoconuts

/-- Theorem stating that the total number of fresh pineapples and coconuts left is 92 --/
theorem fresh_fruits_count :
  freshFruitsLeft 120 75 52 38 11 5 = 92 := by
  sorry

end NUMINAMATH_CALUDE_fresh_fruits_count_l2901_290155


namespace NUMINAMATH_CALUDE_temperature_difference_l2901_290144

/-- Given the highest and lowest temperatures on a certain day in Xianning,
    prove that the temperature difference is 5°C. -/
theorem temperature_difference (lowest highest : ℝ) 
  (h_lowest : lowest = -3)
  (h_highest : highest = 2) :
  highest - lowest = 5 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l2901_290144


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_five_l2901_290103

theorem complex_magnitude_equals_five (t : ℝ) :
  Complex.abs (1 + 2 * t * Complex.I) = 5 ↔ t = Real.sqrt 6 ∨ t = -Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_five_l2901_290103


namespace NUMINAMATH_CALUDE_present_age_of_B_l2901_290132

-- Define the ages of A and B as natural numbers
variable (A B : ℕ)

-- Define the conditions
def condition1 : Prop := A + 10 = 2 * (B - 10)
def condition2 : Prop := A = B + 7

-- Theorem statement
theorem present_age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 37 := by
  sorry

end NUMINAMATH_CALUDE_present_age_of_B_l2901_290132


namespace NUMINAMATH_CALUDE_school_girls_count_l2901_290197

theorem school_girls_count (total_pupils : ℕ) (girl_boy_difference : ℕ) :
  total_pupils = 1455 →
  girl_boy_difference = 281 →
  ∃ (boys girls : ℕ),
    boys + girls = total_pupils ∧
    girls = boys + girl_boy_difference ∧
    girls = 868 := by
  sorry

end NUMINAMATH_CALUDE_school_girls_count_l2901_290197


namespace NUMINAMATH_CALUDE_not_all_nonnegative_l2901_290117

theorem not_all_nonnegative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
  sorry

end NUMINAMATH_CALUDE_not_all_nonnegative_l2901_290117


namespace NUMINAMATH_CALUDE_gcd_1515_600_l2901_290116

theorem gcd_1515_600 : Nat.gcd 1515 600 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1515_600_l2901_290116


namespace NUMINAMATH_CALUDE_dividend_calculation_l2901_290139

theorem dividend_calculation (remainder quotient divisor : ℕ) 
  (h1 : remainder = 1)
  (h2 : quotient = 54)
  (h3 : divisor = 4) :
  divisor * quotient + remainder = 217 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2901_290139


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l2901_290189

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 0) :
  Complex.abs (a + b + c) = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l2901_290189


namespace NUMINAMATH_CALUDE_fatima_phone_probability_l2901_290106

def first_three_digits : List ℕ := [295, 296, 299]
def base_last_four : List ℕ := [1, 6, 7]

def possible_numbers : ℕ := sorry

theorem fatima_phone_probability :
  (1 : ℚ) / possible_numbers = (1 : ℚ) / 72 := by sorry

end NUMINAMATH_CALUDE_fatima_phone_probability_l2901_290106


namespace NUMINAMATH_CALUDE_percentage_difference_l2901_290165

theorem percentage_difference (x y : ℝ) (h : x = 3 * y) :
  (x - y) / x * 100 = 200 / 3 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l2901_290165


namespace NUMINAMATH_CALUDE_average_speed_two_part_journey_l2901_290150

theorem average_speed_two_part_journey 
  (total_distance : ℝ) 
  (first_part_ratio : ℝ) 
  (first_part_speed : ℝ) 
  (second_part_speed : ℝ) 
  (h1 : first_part_ratio = 0.35) 
  (h2 : first_part_speed = 35) 
  (h3 : second_part_speed = 65) 
  (h4 : first_part_ratio > 0 ∧ first_part_ratio < 1) :
  let second_part_ratio := 1 - first_part_ratio
  let first_part_time := (first_part_ratio * total_distance) / first_part_speed
  let second_part_time := (second_part_ratio * total_distance) / second_part_speed
  let total_time := first_part_time + second_part_time
  let average_speed := total_distance / total_time
  average_speed = 50 := by sorry

end NUMINAMATH_CALUDE_average_speed_two_part_journey_l2901_290150


namespace NUMINAMATH_CALUDE_phone_extension_permutations_l2901_290111

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of digits in John's phone extension -/
def num_digits : ℕ := 5

/-- Theorem: The number of permutations of 5 distinct objects is 120 -/
theorem phone_extension_permutations : permutations num_digits = 120 := by
  sorry

end NUMINAMATH_CALUDE_phone_extension_permutations_l2901_290111


namespace NUMINAMATH_CALUDE_pollen_diameter_scientific_notation_l2901_290141

/-- Expresses a given number in scientific notation -/
def scientific_notation (n : ℝ) : ℝ × ℤ :=
  sorry

theorem pollen_diameter_scientific_notation :
  scientific_notation 0.0000021 = (2.1, -6) :=
sorry

end NUMINAMATH_CALUDE_pollen_diameter_scientific_notation_l2901_290141


namespace NUMINAMATH_CALUDE_baseball_cards_equality_l2901_290190

theorem baseball_cards_equality (J M C : ℕ) : 
  C = 20 → 
  M = C - 6 → 
  J + M + C = 48 → 
  J = M := by sorry

end NUMINAMATH_CALUDE_baseball_cards_equality_l2901_290190


namespace NUMINAMATH_CALUDE_one_chief_physician_probability_l2901_290179

theorem one_chief_physician_probability 
  (total_male_doctors : ℕ) 
  (total_female_doctors : ℕ) 
  (male_chief_physicians : ℕ) 
  (female_chief_physicians : ℕ) 
  (selected_male_doctors : ℕ) 
  (selected_female_doctors : ℕ) :
  total_male_doctors = 4 →
  total_female_doctors = 5 →
  male_chief_physicians = 1 →
  female_chief_physicians = 1 →
  selected_male_doctors = 3 →
  selected_female_doctors = 2 →
  (Nat.choose total_male_doctors selected_male_doctors *
   Nat.choose total_female_doctors selected_female_doctors -
   Nat.choose (total_male_doctors - male_chief_physicians) selected_male_doctors *
   Nat.choose (total_female_doctors - female_chief_physicians) selected_female_doctors -
   Nat.choose (total_male_doctors - male_chief_physicians) (selected_male_doctors - 1) *
   Nat.choose (total_female_doctors - female_chief_physicians) selected_female_doctors -
   Nat.choose total_male_doctors selected_male_doctors *
   Nat.choose (total_female_doctors - female_chief_physicians) (selected_female_doctors - 1)) /
  (Nat.choose total_male_doctors selected_male_doctors *
   Nat.choose total_female_doctors selected_female_doctors) = 6 / 17 := by
  sorry

end NUMINAMATH_CALUDE_one_chief_physician_probability_l2901_290179


namespace NUMINAMATH_CALUDE_scale_division_theorem_l2901_290112

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 6 * 12 + 8

/-- Represents the number of parts the scale is divided into -/
def num_parts : ℕ := 4

/-- Represents the length of each part in inches -/
def part_length : ℕ := scale_length / num_parts

/-- Proves that each part of the scale is 20 inches (1 foot 8 inches) long -/
theorem scale_division_theorem : part_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_theorem_l2901_290112


namespace NUMINAMATH_CALUDE_stating_pyramid_levels_for_1023_toothpicks_l2901_290109

/-- Represents the number of toothpicks in a pyramid level. -/
def toothpicks_in_level (n : ℕ) : ℕ := 2^(n - 1)

/-- Represents the total number of toothpicks used up to a given level. -/
def total_toothpicks (n : ℕ) : ℕ := 2^n - 1

/-- 
Theorem stating that a pyramid with 1023 toothpicks has 10 levels,
where each level doubles the number of toothpicks from the previous level.
-/
theorem pyramid_levels_for_1023_toothpicks : 
  ∃ n : ℕ, n = 10 ∧ total_toothpicks n = 1023 := by
  sorry


end NUMINAMATH_CALUDE_stating_pyramid_levels_for_1023_toothpicks_l2901_290109


namespace NUMINAMATH_CALUDE_pagoda_lanterns_sum_l2901_290181

/-- Represents a pagoda with lanterns -/
structure Pagoda where
  layers : ℕ
  top_lanterns : ℕ
  total_lanterns : ℕ

/-- Calculates the number of lanterns on the bottom layer of the pagoda -/
def bottom_lanterns (p : Pagoda) : ℕ := p.top_lanterns * 2^(p.layers - 1)

/-- Calculates the sum of lanterns on all layers of the pagoda -/
def sum_lanterns (p : Pagoda) : ℕ := p.top_lanterns * (2^p.layers - 1)

/-- Theorem: For a 7-layer pagoda with lanterns doubling from top to bottom and
    a total of 381 lanterns, the sum of lanterns on the top and bottom layers is 195 -/
theorem pagoda_lanterns_sum :
  ∀ (p : Pagoda), p.layers = 7 → p.total_lanterns = 381 → sum_lanterns p = p.total_lanterns →
  p.top_lanterns + bottom_lanterns p = 195 :=
sorry

end NUMINAMATH_CALUDE_pagoda_lanterns_sum_l2901_290181


namespace NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_quadratic_inequality_l2901_290115

theorem x_negative_necessary_not_sufficient_for_quadratic_inequality :
  (∀ x : ℝ, x^2 + x < 0 → x < 0) ∧
  (∃ x : ℝ, x < 0 ∧ x^2 + x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_quadratic_inequality_l2901_290115


namespace NUMINAMATH_CALUDE_football_scoring_problem_l2901_290135

/-- Represents the football scoring problem with Gina and Tom -/
theorem football_scoring_problem 
  (gina_day1 : ℕ) 
  (tom_day1 : ℕ) 
  (tom_day2 : ℕ) 
  (gina_day2 : ℕ) 
  (h1 : gina_day1 = 2)
  (h2 : tom_day1 = gina_day1 + 3)
  (h3 : tom_day2 = 6)
  (h4 : gina_day2 < tom_day2)
  (h5 : gina_day1 + tom_day1 + gina_day2 + tom_day2 = 17) :
  tom_day2 - gina_day2 = 2 := by
sorry

end NUMINAMATH_CALUDE_football_scoring_problem_l2901_290135


namespace NUMINAMATH_CALUDE_three_isosceles_right_triangles_l2901_290161

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 2 * x^2 + 4 * x - y^2 = 0

-- Define an isosceles right triangle with O as the right angle
def isosceles_right_triangle (A B : ℝ × ℝ) : Prop :=
  let (xA, yA) := A
  let (xB, yB) := B
  xA * xB + yA * yB = 0 ∧ xA^2 + yA^2 = xB^2 + yB^2

-- Main theorem
theorem three_isosceles_right_triangles :
  ∃ (S : Finset (ℝ × ℝ)),
    Finset.card S = 3 ∧
    (∀ A ∈ S, hyperbola A.1 A.2) ∧
    (∀ A B, A ∈ S → B ∈ S → A ≠ B → isosceles_right_triangle A B) ∧
    (∀ A B, hyperbola A.1 A.2 → hyperbola B.1 B.2 → 
      isosceles_right_triangle A B → (A ∈ S ∧ B ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_three_isosceles_right_triangles_l2901_290161


namespace NUMINAMATH_CALUDE_johns_age_l2901_290122

theorem johns_age (age : ℕ) : 
  (age + 9 = 3 * (age - 11)) → age = 21 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l2901_290122


namespace NUMINAMATH_CALUDE_sum_greater_than_four_l2901_290172

theorem sum_greater_than_four (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (h : 1/a + 1/b = 1) : a + b > 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_four_l2901_290172


namespace NUMINAMATH_CALUDE_cos_2theta_value_l2901_290157

theorem cos_2theta_value (θ : ℝ) (h : Real.tan (θ + π/4) = (1/2) * Real.tan θ - 7/2) : 
  Real.cos (2 * θ) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2theta_value_l2901_290157


namespace NUMINAMATH_CALUDE_fourth_student_in_sample_l2901_290143

def systematic_sample (total_students : ℕ) (sample_size : ℕ) (sample : Finset ℕ) : Prop :=
  sample.card = sample_size ∧
  ∃ k : ℕ, ∀ i ∈ sample, ∃ j : ℕ, i = 1 + j * (total_students / sample_size)

theorem fourth_student_in_sample 
  (total_students : ℕ) (sample_size : ℕ) (sample : Finset ℕ) 
  (h1 : total_students = 52)
  (h2 : sample_size = 4)
  (h3 : 3 ∈ sample)
  (h4 : 29 ∈ sample)
  (h5 : 42 ∈ sample)
  (h6 : systematic_sample total_students sample_size sample) :
  16 ∈ sample :=
sorry

end NUMINAMATH_CALUDE_fourth_student_in_sample_l2901_290143


namespace NUMINAMATH_CALUDE_exists_equal_shift_eval_l2901_290137

-- Define the type for polynomials of degree 2014
def Poly2014 := Polynomial ℝ

-- Define what it means for a polynomial to be monic of degree 2014
def is_monic_2014 (p : Poly2014) : Prop :=
  p.degree = 2014 ∧ p.leadingCoeff = 1

-- Define the theorem
theorem exists_equal_shift_eval
  (P Q : Poly2014)
  (h_monic_P : is_monic_2014 P)
  (h_monic_Q : is_monic_2014 Q)
  (h_not_equal : ∀ x : ℝ, P.eval x ≠ Q.eval x) :
  ∃ x : ℝ, P.eval (x - 1) = Q.eval (x + 1) := by
sorry

end NUMINAMATH_CALUDE_exists_equal_shift_eval_l2901_290137


namespace NUMINAMATH_CALUDE_extremum_values_of_e_l2901_290180

theorem extremum_values_of_e (a b c d e : ℝ) 
  (h1 : 3*a + 2*b - c + 4*d + Real.sqrt 133 * e = Real.sqrt 133)
  (h2 : 2*a^2 + 3*b^2 + 3*c^2 + d^2 + 6*e^2 = 60) :
  ∃ (e_min e_max : ℝ), 
    e_min = (1 - Real.sqrt 19) / 2 ∧ 
    e_max = (1 + Real.sqrt 19) / 2 ∧
    e_min ≤ e ∧ e ≤ e_max ∧
    (e = e_min ∨ e = e_max → 
      ∃ (k : ℝ), a = 3*k/8 ∧ b = k/6 ∧ c = -k/12 ∧ d = k) :=
by sorry

end NUMINAMATH_CALUDE_extremum_values_of_e_l2901_290180


namespace NUMINAMATH_CALUDE_transfer_equation_l2901_290121

def location_A : ℕ := 232
def location_B : ℕ := 146

theorem transfer_equation (x : ℤ) : 
  (location_A : ℤ) + x = 3 * ((location_B : ℤ) - x) ↔ 
  (location_A : ℤ) + x = 3 * ((location_B : ℤ) - x) :=
by sorry

end NUMINAMATH_CALUDE_transfer_equation_l2901_290121


namespace NUMINAMATH_CALUDE_charity_ticket_sales_l2901_290136

theorem charity_ticket_sales (full_price_tickets half_price_tickets : ℕ) 
  (full_price half_price : ℚ) : 
  full_price_tickets + half_price_tickets = 160 →
  full_price_tickets * full_price + half_price_tickets * half_price = 2400 →
  half_price = full_price / 2 →
  full_price_tickets * full_price = 960 := by
sorry

end NUMINAMATH_CALUDE_charity_ticket_sales_l2901_290136


namespace NUMINAMATH_CALUDE_sugar_water_and_triangle_inequality_l2901_290160

theorem sugar_water_and_triangle_inequality 
  (a b m : ℝ) 
  (hab : b > a) (ha : a > 0) (hm : m > 0) 
  (A B C : ℝ) 
  (hABC : A > 0 ∧ B > 0 ∧ C > 0) 
  (hAcute : A < B + C ∧ B < C + A ∧ C < A + B) : 
  (a / b < (a + m) / (b + m)) ∧ 
  (A / (B + C) + B / (C + A) + C / (A + B) < 2) := by
  sorry

end NUMINAMATH_CALUDE_sugar_water_and_triangle_inequality_l2901_290160


namespace NUMINAMATH_CALUDE_other_leg_length_l2901_290168

/-- Given a right triangle with one leg of length 5 and hypotenuse of length 11,
    the length of the other leg is 4√6. -/
theorem other_leg_length (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
    (h_leg : a = 5) (h_hyp : c = 11) : b = 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_other_leg_length_l2901_290168


namespace NUMINAMATH_CALUDE_combined_drying_time_l2901_290192

-- Define the driers' capacities and individual drying times
def drier1_capacity : ℚ := 1/2
def drier2_capacity : ℚ := 3/4
def drier3_capacity : ℚ := 1

def drier1_time : ℚ := 24
def drier2_time : ℚ := 2
def drier3_time : ℚ := 8

-- Define the combined drying rate
def combined_rate : ℚ := 
  drier1_capacity / drier1_time + 
  drier2_capacity / drier2_time + 
  drier3_capacity / drier3_time

-- Theorem statement
theorem combined_drying_time : 
  1 / combined_rate = 3/2 := by sorry

end NUMINAMATH_CALUDE_combined_drying_time_l2901_290192


namespace NUMINAMATH_CALUDE_pie_slices_problem_l2901_290110

theorem pie_slices_problem (initial_slices : ℕ) : 
  initial_slices > 0 →
  (initial_slices - 1 : ℚ) / 2 - 1 = 5 / 2 →
  initial_slices = 8 := by
sorry

end NUMINAMATH_CALUDE_pie_slices_problem_l2901_290110


namespace NUMINAMATH_CALUDE_valid_sequence_count_l2901_290186

/-- Represents the number of valid sequences of length n -/
def S (n : ℕ) : ℕ :=
  sorry

/-- Represents the number of valid sequences of length n ending with A -/
def A (n : ℕ) : ℕ :=
  sorry

theorem valid_sequence_count : S 2015 % 12 = 8 := by
  sorry

end NUMINAMATH_CALUDE_valid_sequence_count_l2901_290186


namespace NUMINAMATH_CALUDE_train_problem_solution_l2901_290120

/-- Represents the train problem scenario -/
structure TrainProblem where
  totalDistance : ℝ
  trainBTime : ℝ
  meetingPointA : ℝ
  trainATime : ℝ

/-- The solution to the train problem -/
def solveTrain (p : TrainProblem) : Prop :=
  p.totalDistance = 125 ∧
  p.trainBTime = 8 ∧
  p.meetingPointA = 50 ∧
  p.trainATime = 12

/-- Theorem stating that the solution satisfies the problem conditions -/
theorem train_problem_solution :
  ∀ (p : TrainProblem),
    p.totalDistance = 125 ∧
    p.trainBTime = 8 ∧
    p.meetingPointA = 50 →
    solveTrain p :=
by
  sorry

#check train_problem_solution

end NUMINAMATH_CALUDE_train_problem_solution_l2901_290120


namespace NUMINAMATH_CALUDE_grid_paths_7x3_l2901_290164

theorem grid_paths_7x3 : 
  let m : ℕ := 7  -- width of the grid
  let n : ℕ := 3  -- height of the grid
  (Nat.choose (m + n) n) = 120 := by
sorry

end NUMINAMATH_CALUDE_grid_paths_7x3_l2901_290164


namespace NUMINAMATH_CALUDE_triangle_segment_length_l2901_290138

/-- Triangle ABC with points D and E on BC -/
structure TriangleABC where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Length of side CA -/
  CA : ℝ
  /-- Length of CD -/
  CD : ℝ
  /-- Ratio of BE to EC -/
  BE_EC_ratio : ℝ
  /-- Equality of angles BAE and CAD -/
  angle_equality : Bool

/-- The main theorem -/
theorem triangle_segment_length 
  (t : TriangleABC) 
  (h1 : t.AB = 12) 
  (h2 : t.BC = 16) 
  (h3 : t.CA = 15) 
  (h4 : t.CD = 5) 
  (h5 : t.BE_EC_ratio = 3) 
  (h6 : t.angle_equality = true) : 
  ∃ (BE : ℝ), BE = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_segment_length_l2901_290138


namespace NUMINAMATH_CALUDE_division_multiplication_result_l2901_290104

theorem division_multiplication_result : 
  let number := 5
  let intermediate := number / 6
  let result := intermediate * 12
  result = 10 := by sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l2901_290104


namespace NUMINAMATH_CALUDE_remainder_invariant_can_reach_43_from_3_cannot_reach_43_from_5_l2901_290175

/-- The set of allowed operations in Daniel's game -/
inductive GameOperation
  | AddFour    : GameOperation
  | MultiplyFour : GameOperation
  | Square     : GameOperation

/-- Apply a single game operation to a number -/
def applyOperation (n : ℤ) (op : GameOperation) : ℤ :=
  match op with
  | GameOperation.AddFour    => n + 4
  | GameOperation.MultiplyFour => n * 4
  | GameOperation.Square     => n * n

/-- Apply a sequence of game operations to a number -/
def applyOperations (n : ℤ) (ops : List GameOperation) : ℤ :=
  ops.foldl applyOperation n

/-- Proposition: Starting from a number with remainder 1 when divided by 4,
    any resulting number will have remainder 0 or 1 -/
theorem remainder_invariant (n : ℤ) (ops : List GameOperation) :
  n % 4 = 1 → (applyOperations n ops) % 4 = 0 ∨ (applyOperations n ops) % 4 = 1 :=
sorry

/-- Proposition: It's possible to obtain 43 from 3 using allowed operations -/
theorem can_reach_43_from_3 : ∃ (ops : List GameOperation), applyOperations 3 ops = 43 :=
sorry

/-- Proposition: It's impossible to obtain 43 from 5 using allowed operations -/
theorem cannot_reach_43_from_5 : ¬ ∃ (ops : List GameOperation), applyOperations 5 ops = 43 :=
sorry

end NUMINAMATH_CALUDE_remainder_invariant_can_reach_43_from_3_cannot_reach_43_from_5_l2901_290175


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2901_290184

theorem smaller_number_proof (x y : ℕ+) : 
  (x * y : ℕ) = 323 → 
  (x : ℕ) = (y : ℕ) + 2 → 
  y = 17 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2901_290184


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2901_290199

theorem quadratic_factorization (m : ℝ) : m^2 - 14*m + 49 = (m - 7)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2901_290199


namespace NUMINAMATH_CALUDE_children_playing_both_sports_l2901_290198

/-- Given a class of children with the following properties:
  * The total number of children is 38
  * 19 children play tennis
  * 21 children play squash
  * 10 children play neither sport
  Then, the number of children who play both sports is 12 -/
theorem children_playing_both_sports
  (total : ℕ) (tennis : ℕ) (squash : ℕ) (neither : ℕ) (both : ℕ)
  (h1 : total = 38)
  (h2 : tennis = 19)
  (h3 : squash = 21)
  (h4 : neither = 10)
  (h5 : total = tennis + squash - both + neither) :
  both = 12 := by
sorry

end NUMINAMATH_CALUDE_children_playing_both_sports_l2901_290198


namespace NUMINAMATH_CALUDE_tangent_point_coordinates_l2901_290173

theorem tangent_point_coordinates : 
  ∀ x y : ℝ, 
    y = Real.exp (-x) →                        -- Point P(x,y) is on the curve y = e^(-x)
    (- Real.exp (-x)) = -2 →                   -- Tangent line is parallel to 2x + y + 1 = 0
    x = -Real.log 2 ∧ y = 2 := by              -- Coordinates of P are (-ln2, 2)
  sorry

end NUMINAMATH_CALUDE_tangent_point_coordinates_l2901_290173


namespace NUMINAMATH_CALUDE_equation_solution_l2901_290178

theorem equation_solution (y : ℝ) : 
  ∃ x : ℝ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 ∧ x = -21/38 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2901_290178


namespace NUMINAMATH_CALUDE_two_angles_not_unique_l2901_290194

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ -- length of one leg
  b : ℝ -- length of the other leg
  c : ℝ -- length of the hypotenuse
  angle_A : ℝ -- one acute angle in radians
  angle_B : ℝ -- the other acute angle in radians
  right_angle : c^2 = a^2 + b^2 -- Pythagorean theorem
  acute_angles : angle_A > 0 ∧ angle_A < π/2 ∧ angle_B > 0 ∧ angle_B < π/2
  angle_sum : angle_A + angle_B = π/2 -- sum of acute angles in a right triangle

-- Theorem stating that two acute angles do not uniquely determine a right-angled triangle
theorem two_angles_not_unique (angle1 angle2 : ℝ) 
  (h1 : angle1 > 0 ∧ angle1 < π/2) 
  (h2 : angle2 > 0 ∧ angle2 < π/2) 
  (h3 : angle1 + angle2 = π/2) : 
  ∃ t1 t2 : RightTriangle, t1 ≠ t2 ∧ t1.angle_A = angle1 ∧ t1.angle_B = angle2 ∧
                           t2.angle_A = angle1 ∧ t2.angle_B = angle2 :=
sorry

end NUMINAMATH_CALUDE_two_angles_not_unique_l2901_290194


namespace NUMINAMATH_CALUDE_system_solution_l2901_290133

theorem system_solution : ∃! (x y : ℝ), 
  (x^2 * y + x * y^2 + 3*x + 3*y + 24 = 0) ∧ 
  (x^3 * y - x * y^3 + 3*x^2 - 3*y^2 - 48 = 0) ∧
  (x = -3) ∧ (y = -1) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2901_290133


namespace NUMINAMATH_CALUDE_tea_mixture_price_l2901_290147

/-- The price of the first variety of tea in Rs per kg -/
def price_first : ℝ := 126

/-- The price of the second variety of tea in Rs per kg -/
def price_second : ℝ := 135

/-- The price of the third variety of tea in Rs per kg -/
def price_third : ℝ := 175.5

/-- The price of the mixture in Rs per kg -/
def price_mixture : ℝ := 153

/-- The ratio of the first variety in the mixture -/
def ratio_first : ℝ := 1

/-- The ratio of the second variety in the mixture -/
def ratio_second : ℝ := 1

/-- The ratio of the third variety in the mixture -/
def ratio_third : ℝ := 2

/-- The total ratio sum -/
def ratio_total : ℝ := ratio_first + ratio_second + ratio_third

theorem tea_mixture_price :
  (ratio_first * price_first + ratio_second * price_second + ratio_third * price_third) / ratio_total = price_mixture := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l2901_290147


namespace NUMINAMATH_CALUDE_red_paint_amount_l2901_290177

/-- Given a paint mixture with a ratio of red to white as 5:7, 
    if 21 quarts of white paint are used, then 15 quarts of red paint should be used. -/
theorem red_paint_amount (red white : ℚ) : 
  (red / white = 5 / 7) → (white = 21) → (red = 15) := by
  sorry

end NUMINAMATH_CALUDE_red_paint_amount_l2901_290177


namespace NUMINAMATH_CALUDE_better_value_is_16_cents_per_ounce_l2901_290196

/-- Represents a box of macaroni and cheese -/
structure MacaroniBox where
  weight : ℕ  -- weight in ounces
  price : ℕ   -- price in cents

/-- Calculates the price per ounce for a given box -/
def pricePerOunce (box : MacaroniBox) : ℚ :=
  box.price / box.weight

/-- Finds the box with the lowest price per ounce -/
def bestValue (box1 box2 : MacaroniBox) : MacaroniBox :=
  if pricePerOunce box1 ≤ pricePerOunce box2 then box1 else box2

theorem better_value_is_16_cents_per_ounce :
  let largerBox : MacaroniBox := ⟨30, 480⟩
  let smallerBox : MacaroniBox := ⟨20, 340⟩
  pricePerOunce (bestValue largerBox smallerBox) = 16 / 1 := by
  sorry

end NUMINAMATH_CALUDE_better_value_is_16_cents_per_ounce_l2901_290196


namespace NUMINAMATH_CALUDE_probability_not_same_intersection_is_two_thirds_l2901_290146

/-- Represents the number of officers -/
def num_officers : ℕ := 3

/-- Represents the number of intersections -/
def num_intersections : ℕ := 2

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := (num_officers.choose 2) * 2

/-- The number of arrangements where two specific officers are at the same intersection -/
def same_intersection_arrangements : ℕ := 2

/-- The probability that two specific officers are not at the same intersection -/
def probability_not_same_intersection : ℚ := 1 - (same_intersection_arrangements : ℚ) / total_arrangements

theorem probability_not_same_intersection_is_two_thirds :
  probability_not_same_intersection = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_not_same_intersection_is_two_thirds_l2901_290146


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2901_290119

-- Define the complex number z
def z : ℂ := (2 - Complex.I) * Complex.I

-- Theorem statement
theorem z_in_first_quadrant : Real.sign (z.re) = 1 ∧ Real.sign (z.im) = 1 :=
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2901_290119


namespace NUMINAMATH_CALUDE_pf_length_l2901_290129

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) where
  right_angled : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0
  pq_length : Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) = 3
  pr_length : Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) = 3 * Real.sqrt 3

-- Define the altitude PL and median RM
def altitude (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry
def median (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the intersection point F
def intersectionPoint (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem pf_length (P Q R : ℝ × ℝ) (t : Triangle P Q R) :
  let F := intersectionPoint P Q R
  Real.sqrt ((F.1 - P.1)^2 + (F.2 - P.2)^2) = 0.857 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_pf_length_l2901_290129


namespace NUMINAMATH_CALUDE_quadratic_y_intercept_l2901_290191

/-- The function f(x) = -(x-1)^2 + 2 intersects the y-axis at the point (0, 1) -/
theorem quadratic_y_intercept :
  let f : ℝ → ℝ := fun x ↦ -(x - 1)^2 + 2
  f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_y_intercept_l2901_290191


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2901_290188

/-- 
Given an arithmetic sequence with first term a₁ and non-zero common difference d,
if the 1st, 6th, and 21st terms form a geometric sequence,
then the common ratio of this geometric sequence is 3.
-/
theorem arithmetic_geometric_sequence_ratio 
  (a₁ : ℝ) (d : ℝ) (h : d ≠ 0) : 
  (a₁ + 5 * d)^2 = a₁ * (a₁ + 20 * d) → 
  (a₁ + 5 * d) / a₁ = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2901_290188


namespace NUMINAMATH_CALUDE_pen_more_expensive_than_two_notebooks_l2901_290118

-- Define the variables
variable (T : ℝ) -- cost of one notebook
variable (R : ℝ) -- cost of one pen
variable (C : ℝ) -- cost of one pencil

-- Define the conditions
axiom condition1 : T + R + C = 120
axiom condition2 : 5 * T + 2 * R + 3 * C = 350

-- State the theorem
theorem pen_more_expensive_than_two_notebooks : R > 2 * T := by
  sorry

end NUMINAMATH_CALUDE_pen_more_expensive_than_two_notebooks_l2901_290118


namespace NUMINAMATH_CALUDE_three_number_problem_l2901_290174

theorem three_number_problem (a b c : ℝ) :
  a + b + c = 114 →
  b^2 = a * c →
  ∃ d : ℝ, b = a + 3*d ∧ c = a + 24*d →
  ((a = 38 ∧ b = 38 ∧ c = 38) ∨ (a = 2 ∧ b = 14 ∧ c = 98)) :=
by sorry

end NUMINAMATH_CALUDE_three_number_problem_l2901_290174


namespace NUMINAMATH_CALUDE_percentage_difference_l2901_290101

theorem percentage_difference : 
  (60 * 80 / 100) - (25 * 4 / 5) = 28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2901_290101


namespace NUMINAMATH_CALUDE_circle_center_sum_l2901_290124

/-- The sum of the x and y coordinates of the center of a circle
    described by the equation x^2 + y^2 = 6x + 6y - 30 is equal to 6 -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 6*x + 6*y - 30) → ∃ h k : ℝ, (h + k = 6 ∧ (x - h)^2 + (y - k)^2 = (x^2 + y^2 - 6*x - 6*y + 30)) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2901_290124


namespace NUMINAMATH_CALUDE_number_of_trees_l2901_290148

/-- The number of trees around the house. -/
def n : ℕ := 118

/-- The difference between Alexander's and Timur's starting points. -/
def start_diff : ℕ := 33 - 12

/-- The theorem stating the number of trees around the house. -/
theorem number_of_trees :
  ∃ k : ℕ, n + k = 105 - 12 + 8 ∧ start_diff = 33 - 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_trees_l2901_290148


namespace NUMINAMATH_CALUDE_determinant_2x2_matrix_l2901_290193

theorem determinant_2x2_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, 3; -1, 2]
  Matrix.det A = 17 := by
sorry

end NUMINAMATH_CALUDE_determinant_2x2_matrix_l2901_290193


namespace NUMINAMATH_CALUDE_subset_with_sum_property_l2901_290176

theorem subset_with_sum_property (Y : Finset ℕ+) (n : ℕ) (hn : Y.card = n) :
  ∃ B : Finset ℕ+, B ⊆ Y ∧ B.card > n / 3 ∧
    ∀ u v : ℕ+, u ∈ B → v ∈ B → (u + v : ℕ+) ∉ B := by
  sorry

end NUMINAMATH_CALUDE_subset_with_sum_property_l2901_290176
