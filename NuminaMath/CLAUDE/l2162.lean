import Mathlib

namespace NUMINAMATH_CALUDE_small_triangle_area_ratio_l2162_216279

/-- Represents a right triangle divided into a square and two smaller right triangles -/
structure DividedRightTriangle where
  /-- Area of the square -/
  square_area : ℝ
  /-- Area of the first small right triangle -/
  small_triangle1_area : ℝ
  /-- Area of the second small right triangle -/
  small_triangle2_area : ℝ
  /-- The first small triangle's area is n times the square's area -/
  small_triangle1_prop : small_triangle1_area = square_area * n
  /-- The square and two small triangles form a right triangle -/
  forms_right_triangle : square_area + small_triangle1_area + small_triangle2_area > 0

/-- 
If one small right triangle has an area n times the square's area, 
then the other small right triangle has an area 1/(4n) times the square's area 
-/
theorem small_triangle_area_ratio 
  (t : DividedRightTriangle) (n : ℝ) (hn : n > 0) :
  t.small_triangle2_area / t.square_area = 1 / (4 * n) := by
  sorry

end NUMINAMATH_CALUDE_small_triangle_area_ratio_l2162_216279


namespace NUMINAMATH_CALUDE_least_value_property_l2162_216217

/-- Represents a 3-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_three_digit : hundreds ≥ 1 ∧ hundreds ≤ 9

/-- The value of a 3-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The sum of digits of a 3-digit number -/
def ThreeDigitNumber.digit_sum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.units

/-- Predicate for the difference between hundreds and tens being 8 -/
def digit_difference_eight (n : ThreeDigitNumber) : Prop :=
  n.tens - n.hundreds = 8 ∨ n.hundreds - n.tens = 8

theorem least_value_property (k : ThreeDigitNumber) 
  (h : digit_difference_eight k) :
  ∃ (min_k : ThreeDigitNumber), 
    digit_difference_eight min_k ∧
    ∀ (k' : ThreeDigitNumber), digit_difference_eight k' → 
      min_k.value ≤ k'.value ∧
      min_k.value = 19 * min_k.digit_sum :=
  sorry

end NUMINAMATH_CALUDE_least_value_property_l2162_216217


namespace NUMINAMATH_CALUDE_sum_a_b_equals_eleven_l2162_216267

theorem sum_a_b_equals_eleven (a b c d : ℝ) 
  (h1 : b + c = 9)
  (h2 : c + d = 3)
  (h3 : a + d = 5) :
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_eleven_l2162_216267


namespace NUMINAMATH_CALUDE_cream_ratio_l2162_216202

-- Define the initial quantities
def initial_coffee : ℚ := 20
def john_drink : ℚ := 3
def john_cream : ℚ := 4
def jane_cream : ℚ := 3
def jane_drink : ℚ := 5

-- Calculate the amounts of cream in each cup
def john_cream_amount : ℚ := john_cream

def jane_mixture : ℚ := initial_coffee + jane_cream
def jane_cream_ratio : ℚ := jane_cream / jane_mixture
def jane_remaining : ℚ := jane_mixture - jane_drink
def jane_cream_amount : ℚ := jane_cream_ratio * jane_remaining

-- State the theorem
theorem cream_ratio : john_cream_amount / jane_cream_amount = 46 / 27 := by
  sorry

end NUMINAMATH_CALUDE_cream_ratio_l2162_216202


namespace NUMINAMATH_CALUDE_industrial_lubricants_allocation_l2162_216231

theorem industrial_lubricants_allocation :
  let total_degrees : ℝ := 360
  let total_percentage : ℝ := 100
  let microphotonics : ℝ := 14
  let home_electronics : ℝ := 24
  let food_additives : ℝ := 15
  let genetically_modified_microorganisms : ℝ := 19
  let astrophysics_degrees : ℝ := 72
  let known_sectors := microphotonics + home_electronics + food_additives + genetically_modified_microorganisms
  let astrophysics_percentage := (astrophysics_degrees / total_degrees) * total_percentage
  let industrial_lubricants := total_percentage - known_sectors - astrophysics_percentage
  industrial_lubricants = 8 := by
sorry

end NUMINAMATH_CALUDE_industrial_lubricants_allocation_l2162_216231


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2162_216281

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2162_216281


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l2162_216229

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def are_opposite (q : Quadrilateral) (v1 v2 : ℝ × ℝ) : Prop := sorry

def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

def angle_measure (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_side_length 
  (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_opposite : are_opposite q q.A q.C)
  (h_BC : side_length q.B q.C = 4)
  (h_ADC : angle_measure q.A q.D q.C = π / 3)
  (h_BAD : angle_measure q.B q.A q.D = π / 2)
  (h_area : area q = (side_length q.A q.B * side_length q.C q.D + 
                      side_length q.B q.C * side_length q.A q.D) / 2) :
  side_length q.C q.D = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_side_length_l2162_216229


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2162_216204

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 →
  q^3 - 8*q^2 + 10*q - 3 = 0 →
  r^3 - 8*r^2 + 10*r - 3 = 0 →
  p / (q*r + 2) + q / (p*r + 2) + r / (p*q + 2) = 8/5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2162_216204


namespace NUMINAMATH_CALUDE_hyperbola_properties_l2162_216211

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- A point on the hyperbola -/
structure PointOnHyperbola (h : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1
  not_vertex : x ≠ a ∧ x ≠ -a

/-- The theorem stating properties of the hyperbola -/
theorem hyperbola_properties (a b : ℝ) (h : Hyperbola a b) (M : PointOnHyperbola h)
  (slope_product : (M.y / (M.x + a)) * (-M.y / (M.x - a)) = 144/25)
  (focus_asymptote_distance : b * (a^2 + b^2)^(1/2) / a = 12) :
  (∃ (e : ℝ), e = 13/5 ∧ e^2 = 1 + b^2/a^2) ∧
  (a = 5 ∧ b = 12) := by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l2162_216211


namespace NUMINAMATH_CALUDE_smallest_number_l2162_216205

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def binary : List Nat := [1, 0, 1, 0, 1, 1]
def ternary : List Nat := [1, 2, 1, 0]
def octal : List Nat := [1, 1, 0]
def duodecimal : List Nat := [6, 8]

theorem smallest_number : 
  to_decimal ternary 3 ≤ to_decimal binary 2 ∧
  to_decimal ternary 3 ≤ to_decimal octal 8 ∧
  to_decimal ternary 3 ≤ to_decimal duodecimal 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2162_216205


namespace NUMINAMATH_CALUDE_odd_sum_even_equivalence_l2162_216215

theorem odd_sum_even_equivalence (x y : ℤ) :
  (Odd x ∧ Odd y → Even (x + y)) ↔ (¬Even (x + y) → ¬(Odd x ∧ Odd y)) := by sorry

end NUMINAMATH_CALUDE_odd_sum_even_equivalence_l2162_216215


namespace NUMINAMATH_CALUDE_bisection_termination_condition_l2162_216200

/-- The bisection method termination condition -/
def is_termination_condition (x₁ x₂ e : ℝ) : Prop :=
  |x₁ - x₂| < e

/-- Theorem stating that the correct termination condition for the bisection method is |x₁ - x₂| < e -/
theorem bisection_termination_condition (x₁ x₂ e : ℝ) (h : e > 0) :
  is_termination_condition x₁ x₂ e ↔ |x₁ - x₂| < e := by sorry

end NUMINAMATH_CALUDE_bisection_termination_condition_l2162_216200


namespace NUMINAMATH_CALUDE_product_of_numbers_l2162_216242

theorem product_of_numbers (x y : ℝ) : 
  x - y = 7 → x^2 + y^2 = 85 → x * y = 18 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2162_216242


namespace NUMINAMATH_CALUDE_parallel_transitivity_l2162_216270

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)
variable (notContained : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity
  (a b : Line) (α : Plane)
  (h1 : parallelLine a b)
  (h2 : parallelLinePlane a α)
  (h3 : notContained b α) :
  parallelLinePlane b α :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l2162_216270


namespace NUMINAMATH_CALUDE_m_div_60_eq_483840_l2162_216249

/-- The smallest positive integer that is a multiple of 60 and has exactly 96 positive integral divisors -/
def m : ℕ := sorry

/-- m is a multiple of 60 -/
axiom m_multiple_of_60 : 60 ∣ m

/-- m has exactly 96 positive integral divisors -/
axiom m_divisors_count : (Finset.filter (· ∣ m) (Finset.range m)).card = 96

/-- m is the smallest such number -/
axiom m_smallest : ∀ k : ℕ, k < m → ¬(60 ∣ k ∧ (Finset.filter (· ∣ k) (Finset.range k)).card = 96)

/-- The main theorem -/
theorem m_div_60_eq_483840 : m / 60 = 483840 := sorry

end NUMINAMATH_CALUDE_m_div_60_eq_483840_l2162_216249


namespace NUMINAMATH_CALUDE_senior_class_college_attendance_l2162_216248

theorem senior_class_college_attendance 
  (total_boys : ℕ) 
  (total_girls : ℕ) 
  (boys_not_attended_percent : ℚ) 
  (total_attended_percent : ℚ) :
  total_boys = 300 →
  total_girls = 240 →
  boys_not_attended_percent = 30 / 100 →
  total_attended_percent = 70 / 100 →
  (total_girls - (total_attended_percent * (total_boys + total_girls) - 
    (1 - boys_not_attended_percent) * total_boys)) / total_girls = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_senior_class_college_attendance_l2162_216248


namespace NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l2162_216246

theorem least_positive_t_for_geometric_progression :
  ∃ (t : ℝ) (α : ℝ),
    0 < α ∧ α < Real.pi / 2 ∧
    (∃ (r : ℝ),
      Real.arcsin (Real.sin α) * r = Real.arcsin (Real.sin (3 * α)) ∧
      Real.arcsin (Real.sin (3 * α)) * r = Real.arcsin (Real.sin (5 * α)) ∧
      Real.arcsin (Real.sin (5 * α)) * r = Real.arcsin (Real.sin (t * α))) ∧
    (∀ (t' : ℝ) (α' : ℝ),
      0 < α' ∧ α' < Real.pi / 2 →
      (∃ (r' : ℝ),
        Real.arcsin (Real.sin α') * r' = Real.arcsin (Real.sin (3 * α')) ∧
        Real.arcsin (Real.sin (3 * α')) * r' = Real.arcsin (Real.sin (5 * α')) ∧
        Real.arcsin (Real.sin (5 * α')) * r' = Real.arcsin (Real.sin (t' * α'))) →
      t ≤ t') ∧
    t = 27 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l2162_216246


namespace NUMINAMATH_CALUDE_composite_ratio_l2162_216221

def first_seven_composites : List Nat := [4, 6, 8, 9, 10, 12, 14]
def next_seven_composites : List Nat := [15, 16, 18, 20, 21, 22, 24]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

theorem composite_ratio :
  (product_of_list first_seven_composites) / 
  (product_of_list next_seven_composites) = 1 / 264 := by
  sorry

end NUMINAMATH_CALUDE_composite_ratio_l2162_216221


namespace NUMINAMATH_CALUDE_product_mod_25_l2162_216297

theorem product_mod_25 (n : ℕ) : 
  (105 * 86 * 97 ≡ n [ZMOD 25]) → 
  (0 ≤ n ∧ n < 25) → 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_product_mod_25_l2162_216297


namespace NUMINAMATH_CALUDE_cos_pi_minus_alpha_l2162_216252

theorem cos_pi_minus_alpha (α : Real) (h : Real.sin (π / 2 + α) = 1 / 7) :
  Real.cos (π - α) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_alpha_l2162_216252


namespace NUMINAMATH_CALUDE_positive_integer_divisibility_l2162_216265

theorem positive_integer_divisibility (n : ℕ) :
  (n + 2009 ∣ n^2 + 2009) ∧ (n + 2010 ∣ n^2 + 2010) → n = 0 ∨ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_divisibility_l2162_216265


namespace NUMINAMATH_CALUDE_ruble_exchange_impossible_l2162_216216

theorem ruble_exchange_impossible : ¬ ∃ (x y z : ℕ), 
  x + y + z = 10 ∧ x + 3*y + 5*z = 25 := by sorry

end NUMINAMATH_CALUDE_ruble_exchange_impossible_l2162_216216


namespace NUMINAMATH_CALUDE_average_difference_l2162_216299

theorem average_difference (x : ℚ) : 
  (10 + 80 + x) / 3 = (20 + 40 + 60) / 3 - 5 ↔ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l2162_216299


namespace NUMINAMATH_CALUDE_triangle_side_length_l2162_216255

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  c = 3 →
  C = π / 3 →
  a = 2 * b →
  b = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2162_216255


namespace NUMINAMATH_CALUDE_binomial_coefficient_condition_l2162_216222

theorem binomial_coefficient_condition (a : ℚ) : 
  (Finset.range 8).sum (fun k => (Nat.choose 7 k) * a^(7-k) * 1^k) = (a + 1)^7 ∧ 
  (Nat.choose 7 6) * a * 1^6 = 1 → 
  a = 1/7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_condition_l2162_216222


namespace NUMINAMATH_CALUDE_max_value_on_line_l2162_216240

/-- Given a point A(3,1) on the line mx + ny + 1 = 0, where mn > 0, 
    the maximum value of 3/m + 1/n is -16. -/
theorem max_value_on_line (m n : ℝ) : 
  (3 * m + n = -1) → 
  (m * n > 0) → 
  (∀ k l : ℝ, (3 * k + l = -1) → (k * l > 0) → (3 / m + 1 / n ≥ 3 / k + 1 / l)) →
  3 / m + 1 / n = -16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_line_l2162_216240


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2162_216226

theorem inequality_solution_set (a : ℝ) :
  (∀ x : ℝ, (a * x) / (x - 1) < (a - 1) / (x - 1)) ↔
  (a > 0 ∧ (a - 1) / a < x ∧ x < 1) ∨
  (a = 0 ∧ x < 1) ∨
  (a < 0 ∧ (x > (a - 1) / a ∨ x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2162_216226


namespace NUMINAMATH_CALUDE_mutually_exclusive_pairs_count_l2162_216289

-- Define the total number of volunteers
def total_volunteers : ℕ := 7

-- Define the number of male and female volunteers
def male_volunteers : ℕ := 4
def female_volunteers : ℕ := 3

-- Define the number of selected volunteers
def selected_volunteers : ℕ := 2

-- Define the events
def event1 : Prop := False  -- Logically inconsistent event
def event2 : Prop := True   -- At least 1 female and all females
def event3 : Prop := True   -- At least 1 male and at least 1 female
def event4 : Prop := True   -- At least 1 female and all males

-- Define a function to count mutually exclusive pairs
def count_mutually_exclusive_pairs (events : List Prop) : ℕ := 1

-- Theorem statement
theorem mutually_exclusive_pairs_count :
  count_mutually_exclusive_pairs [event1, event2, event3, event4] = 1 := by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_pairs_count_l2162_216289


namespace NUMINAMATH_CALUDE_unpainted_cubes_5x5x5_l2162_216292

/-- Given a cube of size n x n x n, where the outer layer is painted,
    calculate the number of unpainted inner cubes. -/
def unpaintedCubes (n : ℕ) : ℕ :=
  (n - 2)^3

/-- The number of unpainted cubes in a 5x5x5 painted cube is 27. -/
theorem unpainted_cubes_5x5x5 :
  unpaintedCubes 5 = 27 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_5x5x5_l2162_216292


namespace NUMINAMATH_CALUDE_salad_dressing_weight_is_700_l2162_216234

/-- Calculates the weight of salad dressing given bowl capacity, oil and vinegar proportions, and their densities. -/
def salad_dressing_weight (bowl_capacity : ℝ) (oil_proportion : ℝ) (vinegar_proportion : ℝ) (oil_density : ℝ) (vinegar_density : ℝ) : ℝ :=
  (bowl_capacity * oil_proportion * oil_density) + (bowl_capacity * vinegar_proportion * vinegar_density)

/-- Theorem stating that the weight of the salad dressing is 700 grams given the specified conditions. -/
theorem salad_dressing_weight_is_700 :
  salad_dressing_weight 150 (2/3) (1/3) 5 4 = 700 := by
  sorry

end NUMINAMATH_CALUDE_salad_dressing_weight_is_700_l2162_216234


namespace NUMINAMATH_CALUDE_zeros_after_decimal_for_one_over_twelve_to_twelve_l2162_216219

-- Define the function to count zeros after decimal point
def count_zeros_after_decimal (x : ℚ) : ℕ :=
  sorry

-- Theorem statement
theorem zeros_after_decimal_for_one_over_twelve_to_twelve :
  count_zeros_after_decimal (1 / (12^12)) = 11 :=
sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_for_one_over_twelve_to_twelve_l2162_216219


namespace NUMINAMATH_CALUDE_root_of_polynomial_l2162_216278

theorem root_of_polynomial (x : ℝ) : x^5 - 2*x^4 - x^2 + 2*x - 3 = 0 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l2162_216278


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2162_216275

theorem rectangle_area_increase (L W : ℝ) (h_positive : L > 0 ∧ W > 0) :
  let original_area := L * W
  let new_area := (1.1 * L) * (1.1 * W)
  (new_area - original_area) / original_area = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2162_216275


namespace NUMINAMATH_CALUDE_equality_implies_a_equals_two_l2162_216261

theorem equality_implies_a_equals_two (a : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x + a = (x + 1)*(x + 2)) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_equality_implies_a_equals_two_l2162_216261


namespace NUMINAMATH_CALUDE_parabola_directrix_l2162_216260

/-- The directrix of a parabola y = ax^2 where a < 0 -/
def directrix_equation (a : ℝ) : ℝ → Prop :=
  fun y => y = -1 / (4 * a)

/-- The parabola equation y = ax^2 -/
def parabola_equation (a : ℝ) : ℝ → ℝ → Prop :=
  fun x y => y = a * x^2

theorem parabola_directrix (a : ℝ) (h : a < 0) :
  ∃ y, directrix_equation a y ∧
    ∀ x, parabola_equation a x y →
      ∃ p, p > 0 ∧ (x^2 = 4 * p * y) ∧ (y = -p) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2162_216260


namespace NUMINAMATH_CALUDE_det_equation_roots_l2162_216286

/-- The determinant equation has either one or three real roots -/
theorem det_equation_roots (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let det := fun x => x * (x * x + a * a) + c * (b * x + a * b) - b * (a * c - b * x)
  ∃ (n : Fin 2), (n = 0 ∧ (∃! x, det x = d)) ∨ (n = 1 ∧ (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ det x = d ∧ det y = d ∧ det z = d)) :=
by sorry

end NUMINAMATH_CALUDE_det_equation_roots_l2162_216286


namespace NUMINAMATH_CALUDE_larger_number_proof_l2162_216228

theorem larger_number_proof (x y : ℝ) (h1 : 4 * y = 7 * x) (h2 : y - x = 12) : y = 28 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2162_216228


namespace NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l2162_216288

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailing_zeros_100_factorial : trailingZeros 100 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_100_factorial_l2162_216288


namespace NUMINAMATH_CALUDE_popton_bus_toes_l2162_216268

/-- Represents a race of beings on planet Popton -/
inductive Race
| Hoopit
| Neglart

/-- Number of hands for each race -/
def hands (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5

/-- Number of toes per hand for each race -/
def toesPerHand (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of toes for an individual of a given race -/
def toesPerIndividual (r : Race) : ℕ :=
  (hands r) * (toesPerHand r)

/-- Number of students of each race on the bus -/
def studentsOnBus (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8

/-- Total number of toes on the Popton school bus -/
def totalToesOnBus : ℕ :=
  (toesPerIndividual Race.Hoopit) * (studentsOnBus Race.Hoopit) +
  (toesPerIndividual Race.Neglart) * (studentsOnBus Race.Neglart)

/-- Theorem: The total number of toes on the Popton school bus is 164 -/
theorem popton_bus_toes : totalToesOnBus = 164 := by
  sorry

end NUMINAMATH_CALUDE_popton_bus_toes_l2162_216268


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l2162_216206

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 70) : 
  c - a = 50 := by
sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l2162_216206


namespace NUMINAMATH_CALUDE_fraction_product_square_l2162_216212

theorem fraction_product_square : (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_square_l2162_216212


namespace NUMINAMATH_CALUDE_bingo_first_column_count_l2162_216285

/-- The number of ways to choose 5 distinct numbers from 1 to 15 -/
def bingo_first_column : ℕ :=
  (15 * 14 * 13 * 12 * 11)

/-- Theorem: The number of distinct possibilities for the first column
    of a MODIFIED SHORT BINGO card is 360360 -/
theorem bingo_first_column_count : bingo_first_column = 360360 := by
  sorry

end NUMINAMATH_CALUDE_bingo_first_column_count_l2162_216285


namespace NUMINAMATH_CALUDE_octagon_all_equal_l2162_216232

/-- Represents an octagon with numbers at each vertex -/
structure Octagon :=
  (vertices : Fin 8 → ℝ)

/-- Condition that each vertex number is the mean of its adjacent vertices -/
def mean_condition (o : Octagon) : Prop :=
  ∀ i : Fin 8, o.vertices i = (o.vertices (i - 1) + o.vertices (i + 1)) / 2

/-- Theorem stating that all vertex numbers must be equal -/
theorem octagon_all_equal (o : Octagon) (h : mean_condition o) : 
  ∀ i j : Fin 8, o.vertices i = o.vertices j :=
sorry

end NUMINAMATH_CALUDE_octagon_all_equal_l2162_216232


namespace NUMINAMATH_CALUDE_square_floor_area_l2162_216210

theorem square_floor_area (rug_length : ℝ) (rug_width : ℝ) (uncovered_fraction : ℝ) :
  rug_length = 2 →
  rug_width = 7 →
  uncovered_fraction = 0.78125 →
  ∃ (floor_side : ℝ),
    floor_side ^ 2 = 64 ∧
    rug_length * rug_width = (1 - uncovered_fraction) * floor_side ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_square_floor_area_l2162_216210


namespace NUMINAMATH_CALUDE_julie_lawns_mowed_l2162_216273

def bike_cost : ℕ := 2345
def initial_savings : ℕ := 1500
def newspapers_delivered : ℕ := 600
def newspaper_pay : ℚ := 0.4
def dogs_walked : ℕ := 24
def dog_walking_pay : ℕ := 15
def lawn_mowing_pay : ℕ := 20
def money_left : ℕ := 155

def total_earned (lawns_mowed : ℕ) : ℚ :=
  initial_savings + newspapers_delivered * newspaper_pay + dogs_walked * dog_walking_pay + lawns_mowed * lawn_mowing_pay

theorem julie_lawns_mowed :
  ∃ (lawns_mowed : ℕ), total_earned lawns_mowed = bike_cost + money_left ∧ lawns_mowed = 20 :=
sorry

end NUMINAMATH_CALUDE_julie_lawns_mowed_l2162_216273


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_882_l2162_216238

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_882 :
  largest_perfect_square_factor 882 = 441 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_882_l2162_216238


namespace NUMINAMATH_CALUDE_johns_next_birthday_l2162_216253

/-- Represents the ages of John, Emily, and Lucas -/
structure Ages where
  john : ℝ
  emily : ℝ
  lucas : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.john = 1.25 * ages.emily ∧
  ages.emily = 0.7 * ages.lucas ∧
  ages.john + ages.emily + ages.lucas = 32

/-- The main theorem -/
theorem johns_next_birthday (ages : Ages) 
  (h : satisfies_conditions ages) : 
  ⌈ages.john⌉ = 11 := by
  sorry


end NUMINAMATH_CALUDE_johns_next_birthday_l2162_216253


namespace NUMINAMATH_CALUDE_triangle_side_length_l2162_216208

theorem triangle_side_length (a c area : ℝ) (ha : a = 1) (hc : c = 7) (harea : area = 5) :
  let h := 2 * area / c
  let b := Real.sqrt ((a^2 + h^2) : ℝ)
  b = Real.sqrt 149 / 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2162_216208


namespace NUMINAMATH_CALUDE_bowl_score_theorem_l2162_216280

def noa_score : ℕ := 30

def phillip_score (noa : ℕ) : ℕ := 2 * noa

def total_score (noa phillip : ℕ) : ℕ := noa + phillip

theorem bowl_score_theorem : 
  total_score noa_score (phillip_score noa_score) = 90 := by
  sorry

end NUMINAMATH_CALUDE_bowl_score_theorem_l2162_216280


namespace NUMINAMATH_CALUDE_first_part_value_l2162_216243

theorem first_part_value (x y : ℝ) : 
  x + y = 24 → 
  7 * x + 5 * y = 146 → 
  x = 13 := by
sorry

end NUMINAMATH_CALUDE_first_part_value_l2162_216243


namespace NUMINAMATH_CALUDE_shifted_line_equation_l2162_216207

/-- Represents a line in the Cartesian coordinate system -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shifts a line vertically by a given amount -/
def vertical_shift (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + shift }

/-- The theorem stating that shifting y = -2x up by 5 units results in y = -2x + 5 -/
theorem shifted_line_equation :
  let original_line : Line := { slope := -2, intercept := 0 }
  let shifted_line := vertical_shift original_line 5
  shifted_line = { slope := -2, intercept := 5 } := by sorry

end NUMINAMATH_CALUDE_shifted_line_equation_l2162_216207


namespace NUMINAMATH_CALUDE_unique_coefficient_exists_l2162_216245

theorem unique_coefficient_exists (x y : ℝ) 
  (eq1 : 4 * x + y = 8) 
  (eq2 : 3 * x - 4 * y = 5) : 
  ∃! a : ℝ, a * x - 3 * y = 23 := by
sorry

end NUMINAMATH_CALUDE_unique_coefficient_exists_l2162_216245


namespace NUMINAMATH_CALUDE_angle_around_point_l2162_216223

/-- Given a point in a plane with four angles around it, where three of the angles are equal (x°) and the fourth is 140°, prove that x = 220/3. -/
theorem angle_around_point (x : ℚ) : 
  (3 * x + 140 = 360) → x = 220 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_around_point_l2162_216223


namespace NUMINAMATH_CALUDE_work_completion_ratio_l2162_216263

/-- Given that A can finish a work in 18 days and that A and B working together
    can finish 1/6 of the work in a day, prove that the ratio of the time taken
    by B to finish the work alone to the time taken by A is 1/2. -/
theorem work_completion_ratio (a b : ℝ) (ha : a = 18) 
    (hab : 1 / a + 1 / b = 1 / 6) : b / a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_ratio_l2162_216263


namespace NUMINAMATH_CALUDE_new_person_weight_l2162_216291

theorem new_person_weight (original_count : ℕ) (original_average : ℝ) (leaving_weight : ℝ) (average_increase : ℝ) :
  original_count = 20 →
  leaving_weight = 92 →
  average_increase = 4.5 →
  (original_count * (original_average + average_increase) - (original_count - 1) * original_average) = 182 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2162_216291


namespace NUMINAMATH_CALUDE_two_distinct_roots_range_l2162_216269

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + (m+3)

-- Define the discriminant of the quadratic function
def discriminant (m : ℝ) : ℝ := m^2 - 4*(m+3)

-- Theorem statement
theorem two_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ m < -2 ∨ m > 6 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_range_l2162_216269


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l2162_216244

theorem binomial_square_coefficient (b : ℚ) : 
  (∃ t u : ℚ, ∀ x, bx^2 + 18*x + 16 = (t*x + u)^2) → b = 81/16 := by
sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l2162_216244


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l2162_216230

theorem least_sum_of_bases (a b : ℕ+) : 
  (4 * a.val + 7 = 7 * b.val + 4) →  -- 47 in base a equals 74 in base b
  (∀ (x y : ℕ+), (4 * x.val + 7 = 7 * y.val + 4) → (x.val + y.val ≥ a.val + b.val)) →
  (a.val + b.val = 24) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l2162_216230


namespace NUMINAMATH_CALUDE_double_area_square_exists_l2162_216284

/-- A point on the grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A square on the grid --/
structure GridSquare where
  a : GridPoint
  b : GridPoint
  c : GridPoint
  d : GridPoint

/-- The area of a grid square --/
def area (s : GridSquare) : ℕ := sorry

/-- A square is legal if its vertices are grid points --/
def is_legal (s : GridSquare) : Prop := sorry

theorem double_area_square_exists (n : ℕ) (h : ∃ s : GridSquare, is_legal s ∧ area s = n) :
  ∃ t : GridSquare, is_legal t ∧ area t = 2 * n := by sorry

end NUMINAMATH_CALUDE_double_area_square_exists_l2162_216284


namespace NUMINAMATH_CALUDE_mountain_distances_l2162_216272

/-- Given a mountainous region with points A, B, and C, where:
    - The horizontal projection of BC is 2400 m
    - Peak B is 800 m higher than C
    - The elevation angle of AB is 20°
    - The elevation angle of AC is 2°
    - The angle between AB' and AC' (where B' and C' are horizontal projections) is 60°
    
    This theorem states that:
    - The horizontal projection of AB is approximately 2426 m
    - The horizontal projection of AC is approximately 2374 m
    - The height difference between B and A is approximately 883.2 m
-/
theorem mountain_distances (BC_proj : ℝ) (B_height_diff : ℝ) (AB_angle : ℝ) (AC_angle : ℝ) (ABC_angle : ℝ)
  (h_BC_proj : BC_proj = 2400)
  (h_B_height_diff : B_height_diff = 800)
  (h_AB_angle : AB_angle = 20 * π / 180)
  (h_AC_angle : AC_angle = 2 * π / 180)
  (h_ABC_angle : ABC_angle = 60 * π / 180) :
  ∃ (AB_proj AC_proj BA_height : ℝ),
    (abs (AB_proj - 2426) < 1) ∧
    (abs (AC_proj - 2374) < 1) ∧
    (abs (BA_height - 883.2) < 0.1) := by
  sorry

end NUMINAMATH_CALUDE_mountain_distances_l2162_216272


namespace NUMINAMATH_CALUDE_optimal_seating_l2162_216259

/-- Represents the conference hall seating problem --/
def ConferenceSeating (total_chairs : ℕ) (chairs_per_row : ℕ) (attendees : ℕ) : Prop :=
  ∃ (chairs_to_remove : ℕ),
    let remaining_chairs := total_chairs - chairs_to_remove
    remaining_chairs % chairs_per_row = 0 ∧
    remaining_chairs ≥ attendees ∧
    ∀ (n : ℕ), n < chairs_to_remove →
      (total_chairs - n) % chairs_per_row ≠ 0 ∨
      (total_chairs - n) < attendees ∨
      (total_chairs - n) - attendees > remaining_chairs - attendees

theorem optimal_seating :
  ConferenceSeating 156 13 100 ∧
  (∃ (chairs_to_remove : ℕ), chairs_to_remove = 52 ∧
    let remaining_chairs := 156 - chairs_to_remove
    remaining_chairs % 13 = 0 ∧
    remaining_chairs ≥ 100 ∧
    ∀ (n : ℕ), n < chairs_to_remove →
      (156 - n) % 13 ≠ 0 ∨
      (156 - n) < 100 ∨
      (156 - n) - 100 > remaining_chairs - 100) :=
by sorry

end NUMINAMATH_CALUDE_optimal_seating_l2162_216259


namespace NUMINAMATH_CALUDE_two_truthful_students_l2162_216233

/-- Represents the performance of a student in the exam -/
inductive Performance
| Good
| NotGood

/-- Represents a student -/
inductive Student
| A
| B
| C
| D

/-- The statement made by each student -/
def statement (s : Student) (performances : Student → Performance) : Prop :=
  match s with
  | Student.A => ∀ s, performances s = Performance.NotGood
  | Student.B => ∃ s, performances s = Performance.Good
  | Student.C => performances Student.B = Performance.NotGood ∨ performances Student.D = Performance.NotGood
  | Student.D => performances Student.D = Performance.NotGood

/-- Checks if a student's statement is true -/
def isTruthful (s : Student) (performances : Student → Performance) : Prop :=
  statement s performances

theorem two_truthful_students :
  ∃ (performances : Student → Performance),
    (isTruthful Student.B performances ∧ isTruthful Student.C performances) ∧
    (¬isTruthful Student.A performances ∧ ¬isTruthful Student.D performances) ∧
    (∀ (s1 s2 : Student), isTruthful s1 performances ∧ isTruthful s2 performances ∧ s1 ≠ s2 →
      ∀ (s : Student), s ≠ s1 ∧ s ≠ s2 → ¬isTruthful s performances) :=
  sorry

end NUMINAMATH_CALUDE_two_truthful_students_l2162_216233


namespace NUMINAMATH_CALUDE_money_distribution_l2162_216227

theorem money_distribution (total : ℝ) (a b c : ℝ) : 
  total = 1080 →
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a + b + c = total →
  a > b →
  a - b = 30 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l2162_216227


namespace NUMINAMATH_CALUDE_prob_heads_and_five_l2162_216282

/-- The probability of getting heads on a fair coin flip -/
def prob_heads : ℚ := 1 / 2

/-- The probability of rolling a 5 on a regular eight-sided die -/
def prob_five : ℚ := 1 / 8

/-- The events (coin flip and die roll) are independent -/
axiom events_independent : True

theorem prob_heads_and_five : prob_heads * prob_five = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_heads_and_five_l2162_216282


namespace NUMINAMATH_CALUDE_pie_division_l2162_216298

theorem pie_division (total_pie : ℚ) (num_people : ℕ) (individual_share : ℚ) : 
  total_pie = 5/8 →
  num_people = 4 →
  individual_share = total_pie / num_people →
  individual_share = 5/32 := by
sorry

end NUMINAMATH_CALUDE_pie_division_l2162_216298


namespace NUMINAMATH_CALUDE_no_real_solutions_l2162_216203

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2162_216203


namespace NUMINAMATH_CALUDE_chord_length_l2162_216290

-- Define the circle C
def circle_C (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x + 4*a*y + 5*a^2 - 25 = 0

-- Define line l₁
def line_l1 (x y : ℝ) : Prop :=
  x + y + 2 = 0

-- Define line l₂
def line_l2 (x y : ℝ) : Prop :=
  3*x + 4*y - 5 = 0

-- Define the center of the circle
def center (a : ℝ) : ℝ × ℝ :=
  (a, -2*a)

-- State that the center of circle C lies on line l₁
axiom center_on_l1 (a : ℝ) :
  line_l1 (center a).1 (center a).2

-- Theorem: The length of the chord formed by intersecting circle C with line l₂ is 8
theorem chord_length : ℝ := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l2162_216290


namespace NUMINAMATH_CALUDE_simplify_expression_l2162_216241

theorem simplify_expression : (625 : ℝ) ^ (1/4) * (400 : ℝ) ^ (1/2) = 100 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2162_216241


namespace NUMINAMATH_CALUDE_product_of_odd_numbers_not_always_composite_l2162_216220

theorem product_of_odd_numbers_not_always_composite :
  ∃ (a b : ℕ), 
    (a % 2 = 1) ∧ 
    (b % 2 = 1) ∧ 
    ¬(∃ (x : ℕ), 1 < x ∧ x < a * b ∧ (a * b) % x = 0) :=
by sorry

end NUMINAMATH_CALUDE_product_of_odd_numbers_not_always_composite_l2162_216220


namespace NUMINAMATH_CALUDE_min_value_expression_l2162_216201

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  x^2 + 4*x*y + 9*y^2 + 8*y*z + 3*z^2 ≥ 18 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 ∧
    x₀^2 + 4*x₀*y₀ + 9*y₀^2 + 8*y₀*z₀ + 3*z₀^2 = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2162_216201


namespace NUMINAMATH_CALUDE_no_positive_solutions_l2162_216237

theorem no_positive_solutions :
  ¬∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^3 + y^3 + z^3 = x + y + z ∧
  x^2 + y^2 + z^2 = x*y*z :=
by sorry

end NUMINAMATH_CALUDE_no_positive_solutions_l2162_216237


namespace NUMINAMATH_CALUDE_divisors_of_3b_plus_18_l2162_216224

theorem divisors_of_3b_plus_18 (a b : ℤ) (h : 4 * b = 10 - 2 * a) :
  (∀ d : ℤ, d ∈ ({1, 2, 3, 6} : Set ℤ) → d ∣ (3 * b + 18)) ∧
  (∃ a b : ℤ, 4 * b = 10 - 2 * a ∧ (¬(4 ∣ (3 * b + 18)) ∨ ¬(5 ∣ (3 * b + 18)) ∨
                                   ¬(7 ∣ (3 * b + 18)) ∨ ¬(8 ∣ (3 * b + 18)))) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_3b_plus_18_l2162_216224


namespace NUMINAMATH_CALUDE_quarters_count_l2162_216296

/-- Represents the types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a collection of coins --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  dimes : Nat
  quarters : Nat
  total_coins : pennies + nickels + dimes + quarters = 11
  at_least_one : pennies ≥ 1 ∧ nickels ≥ 1 ∧ dimes ≥ 1 ∧ quarters ≥ 1
  total_value : pennies * coinValue Coin.Penny +
                nickels * coinValue Coin.Nickel +
                dimes * coinValue Coin.Dime +
                quarters * coinValue Coin.Quarter = 132

theorem quarters_count (cc : CoinCollection) : cc.quarters = 3 := by
  sorry

end NUMINAMATH_CALUDE_quarters_count_l2162_216296


namespace NUMINAMATH_CALUDE_morning_snowfall_l2162_216247

/-- Given the total snowfall and afternoon snowfall in Yardley, 
    prove that the morning snowfall is the difference between them. -/
theorem morning_snowfall (total : ℝ) (afternoon : ℝ) 
  (h1 : total = 0.625) (h2 : afternoon = 0.5) : 
  total - afternoon = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_morning_snowfall_l2162_216247


namespace NUMINAMATH_CALUDE_continuity_at_two_l2162_216256

noncomputable def f (x : ℝ) : ℝ := (x^4 - 16) / (x^3 - 2*x^2)

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |f x - 8| < ε :=
sorry

end NUMINAMATH_CALUDE_continuity_at_two_l2162_216256


namespace NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l2162_216236

/-- A polygon in the figure --/
inductive Polygon
| EquilateralTriangle
| Square
| RegularHexagon

/-- The figure consisting of multiple polygons --/
structure Figure where
  polygons : List Polygon
  triangleSideLength : ℝ
  squareSideLength : ℝ
  hexagonSideLength : ℝ

/-- The polyhedron formed by folding the figure --/
structure Polyhedron where
  figure : Figure

/-- Calculate the volume of the polyhedron --/
def calculateVolume (p : Polyhedron) : ℝ :=
  sorry

/-- The theorem stating that the volume of the specific polyhedron is 8 --/
theorem volume_of_specific_polyhedron :
  let fig : Figure := {
    polygons := [Polygon.EquilateralTriangle, Polygon.EquilateralTriangle, Polygon.EquilateralTriangle,
                 Polygon.Square, Polygon.Square, Polygon.Square,
                 Polygon.RegularHexagon],
    triangleSideLength := 2,
    squareSideLength := 2,
    hexagonSideLength := 1
  }
  let poly : Polyhedron := { figure := fig }
  calculateVolume poly = 8 :=
sorry

end NUMINAMATH_CALUDE_volume_of_specific_polyhedron_l2162_216236


namespace NUMINAMATH_CALUDE_average_weight_b_c_l2162_216276

/-- Given the weights of three people a, b, and c, prove that the average weight of b and c is 45 kg -/
theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →   -- The average weight of a, b, and c is 45 kg
  (a + b) / 2 = 40 →       -- The average weight of a and b is 40 kg
  b = 35 →                 -- The weight of b is 35 kg
  (b + c) / 2 = 45 :=      -- The average weight of b and c is 45 kg
by sorry

end NUMINAMATH_CALUDE_average_weight_b_c_l2162_216276


namespace NUMINAMATH_CALUDE_special_inequality_l2162_216264

/-- The equation x^2 - 4x + |a-3| = 0 has real roots with respect to x -/
def has_real_roots (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 4*x + |a-3| = 0

/-- The inequality t^2 - 2at + 12 < 0 holds for all a in [-1, 7] -/
def inequality_holds (t : ℝ) : Prop :=
  ∀ a : ℝ, -1 ≤ a ∧ a ≤ 7 → t^2 - 2*a*t + 12 < 0

theorem special_inequality (t : ℝ) :
  (∃ a : ℝ, has_real_roots a) →
  inequality_holds t →
  3 < t ∧ t < 4 := by
  sorry

end NUMINAMATH_CALUDE_special_inequality_l2162_216264


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l2162_216251

theorem units_digit_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (35^87 + 93^49) % 10 = n :=
  by sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l2162_216251


namespace NUMINAMATH_CALUDE_circumscribing_circle_diameter_l2162_216225

/-- The diameter of a circle circumscribing six equal, mutually tangent circles -/
theorem circumscribing_circle_diameter (r : ℝ) (h : r = 4) : 
  let small_circle_radius : ℝ := r
  let small_circles_count : ℕ := 6
  let large_circle_diameter : ℝ := 2 * (2 * small_circle_radius + small_circle_radius)
  large_circle_diameter = 24 := by sorry

end NUMINAMATH_CALUDE_circumscribing_circle_diameter_l2162_216225


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l2162_216239

/-- Given ratios between w, x, y, and z, prove the ratio of w to y -/
theorem ratio_w_to_y 
  (h1 : ∃ (k : ℚ), w = (5/2) * k ∧ x = k) 
  (h2 : ∃ (m : ℚ), y = 4 * m ∧ z = m) 
  (h3 : ∃ (n : ℚ), z = (1/8) * n ∧ x = n) : 
  w = 5 * y := by sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l2162_216239


namespace NUMINAMATH_CALUDE_cm_to_m_sq_dm_to_sq_m_min_to_hour_g_to_kg_seven_cm_to_m_thirtyfive_sq_dm_to_sq_m_fortyfive_min_to_hour_twothousandfivehundred_g_to_kg_l2162_216250

-- Define conversion rates
def cm_per_m : ℚ := 100
def sq_dm_per_sq_m : ℚ := 100
def min_per_hour : ℚ := 60
def g_per_kg : ℚ := 1000

-- Theorems to prove
theorem cm_to_m (x : ℚ) : x / cm_per_m = x / 100 := by sorry

theorem sq_dm_to_sq_m (x : ℚ) : x / sq_dm_per_sq_m = x / 100 := by sorry

theorem min_to_hour (x : ℚ) : x / min_per_hour = x / 60 := by sorry

theorem g_to_kg (x : ℚ) : x / g_per_kg = x / 1000 := by sorry

-- Specific conversions
theorem seven_cm_to_m : 7 / cm_per_m = 7 / 100 := by sorry

theorem thirtyfive_sq_dm_to_sq_m : 35 / sq_dm_per_sq_m = 7 / 20 := by sorry

theorem fortyfive_min_to_hour : 45 / min_per_hour = 3 / 4 := by sorry

theorem twothousandfivehundred_g_to_kg : 2500 / g_per_kg = 5 / 2 := by sorry

end NUMINAMATH_CALUDE_cm_to_m_sq_dm_to_sq_m_min_to_hour_g_to_kg_seven_cm_to_m_thirtyfive_sq_dm_to_sq_m_fortyfive_min_to_hour_twothousandfivehundred_g_to_kg_l2162_216250


namespace NUMINAMATH_CALUDE_range_of_a_l2162_216218

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + Real.exp x - 1 / Real.exp x

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) : -1 ≤ a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2162_216218


namespace NUMINAMATH_CALUDE_dot_product_OM_ON_l2162_216283

/-- Regular triangle OAB with side length 1 -/
structure RegularTriangle where
  O : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  is_regular : sorry
  side_length : sorry

/-- Points M and N divide AB into three equal parts -/
def divide_side (t : RegularTriangle) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

/-- Vector representation -/
def vec (p q : ℝ × ℝ) : ℝ × ℝ :=
  (q.1 - p.1, q.2 - p.2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Main theorem -/
theorem dot_product_OM_ON (t : RegularTriangle) : 
  let (M, N) := divide_side t
  let m := vec t.O M
  let n := vec t.O N
  dot_product m n = 1/6 := by
    sorry

end NUMINAMATH_CALUDE_dot_product_OM_ON_l2162_216283


namespace NUMINAMATH_CALUDE_lady_arrangements_proof_l2162_216295

def num_gentlemen : ℕ := 6
def num_ladies : ℕ := 3
def total_positions : ℕ := 9

def valid_arrangements : ℕ := 129600

theorem lady_arrangements_proof :
  (num_gentlemen + num_ladies = total_positions) →
  (valid_arrangements = num_gentlemen.factorial * (num_gentlemen + 1).choose num_ladies) :=
by sorry

end NUMINAMATH_CALUDE_lady_arrangements_proof_l2162_216295


namespace NUMINAMATH_CALUDE_monthly_production_l2162_216214

/-- Represents the number of computers produced in a given time period -/
structure ComputerProduction where
  rate : ℝ  -- Computers produced per 30-minute interval
  days : ℕ  -- Number of days in the time period

/-- Calculates the total number of computers produced in the given time period -/
def totalComputers (prod : ComputerProduction) : ℝ :=
  (prod.rate * (prod.days * 24 * 2 : ℝ))

/-- Theorem stating that a factory producing 6.25 computers every 30 minutes
    for 28 days will produce 8400 computers -/
theorem monthly_production :
  totalComputers ⟨6.25, 28⟩ = 8400 := by sorry

end NUMINAMATH_CALUDE_monthly_production_l2162_216214


namespace NUMINAMATH_CALUDE_david_profit_l2162_216209

/-- Represents the discount percentage based on the number of sacks bought -/
def discount_percentage (num_sacks : ℕ) : ℚ :=
  if num_sacks ≤ 10 then 2/100
  else if num_sacks ≤ 20 then 4/100
  else 5/100

/-- Calculates the total cost of buying sacks with discount -/
def total_cost (num_sacks : ℕ) (price_per_sack : ℚ) : ℚ :=
  num_sacks * price_per_sack * (1 - discount_percentage num_sacks)

/-- Calculates the total selling price for a given number of days and price per kg -/
def selling_price (kg_per_day : ℚ) (price_per_kg : ℚ) (num_days : ℕ) : ℚ :=
  kg_per_day * price_per_kg * num_days

/-- Calculates the total selling price for the week -/
def total_selling_price (kg_per_day : ℚ) : ℚ :=
  selling_price kg_per_day 1.20 3 +
  selling_price kg_per_day 1.30 2 +
  selling_price kg_per_day 1.25 2

/-- Calculates the profit after tax -/
def profit_after_tax (total_selling : ℚ) (total_cost : ℚ) (tax_rate : ℚ) : ℚ :=
  total_selling * (1 - tax_rate) - total_cost

/-- Theorem stating David's profit for the week -/
theorem david_profit :
  let num_sacks : ℕ := 25
  let price_per_sack : ℚ := 50
  let sack_weight : ℚ := 50
  let total_kg : ℚ := num_sacks * sack_weight
  let kg_per_day : ℚ := total_kg / 7
  let tax_rate : ℚ := 12/100
  profit_after_tax
    (total_selling_price kg_per_day)
    (total_cost num_sacks price_per_sack)
    tax_rate = 179.62 := by
  sorry


end NUMINAMATH_CALUDE_david_profit_l2162_216209


namespace NUMINAMATH_CALUDE_power_four_times_four_equals_eight_l2162_216294

theorem power_four_times_four_equals_eight (a : ℝ) : a ^ 4 * a ^ 4 = a ^ 8 := by
  sorry

end NUMINAMATH_CALUDE_power_four_times_four_equals_eight_l2162_216294


namespace NUMINAMATH_CALUDE_smallest_cube_with_specific_digits_l2162_216277

/-- Returns the first n digits of a natural number -/
def firstNDigits (n : ℕ) (x : ℕ) : ℕ := sorry

/-- Returns the last n digits of a natural number -/
def lastNDigits (n : ℕ) (x : ℕ) : ℕ := sorry

/-- Checks if the first n digits of a natural number are all 1 -/
def firstNDigitsAreOne (n : ℕ) (x : ℕ) : Prop :=
  firstNDigits n x = 10^n - 1

/-- Checks if the last n digits of a natural number are all 1 -/
def lastNDigitsAreOne (n : ℕ) (x : ℕ) : Prop :=
  lastNDigits n x = 10^n - 1

theorem smallest_cube_with_specific_digits :
  ∀ x : ℕ, x ≥ 1038471 →
    (firstNDigitsAreOne 3 (x^3) ∧ lastNDigitsAreOne 4 (x^3)) →
    x = 1038471 := by sorry

end NUMINAMATH_CALUDE_smallest_cube_with_specific_digits_l2162_216277


namespace NUMINAMATH_CALUDE_age_difference_l2162_216287

theorem age_difference (son_age man_age : ℕ) : 
  son_age = 30 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2162_216287


namespace NUMINAMATH_CALUDE_tall_blonde_is_swedish_l2162_216274

/-- Represents the nationality of a racer -/
inductive Nationality
| Italian
| Swedish

/-- Represents the physical characteristics of a racer -/
structure Characteristics where
  height : Bool  -- true for tall, false for short
  hair : Bool    -- true for blonde, false for brunette

/-- Represents a racer -/
structure Racer where
  nationality : Nationality
  characteristics : Characteristics

def is_tall_blonde (r : Racer) : Prop :=
  r.characteristics.height ∧ r.characteristics.hair

def is_short_brunette (r : Racer) : Prop :=
  ¬r.characteristics.height ∧ ¬r.characteristics.hair

theorem tall_blonde_is_swedish (racers : Finset Racer) : 
  (∀ r : Racer, r ∈ racers → (is_tall_blonde r → r.nationality = Nationality.Swedish)) :=
by
  sorry

#check tall_blonde_is_swedish

end NUMINAMATH_CALUDE_tall_blonde_is_swedish_l2162_216274


namespace NUMINAMATH_CALUDE_identify_tricksters_l2162_216257

/-- Represents an inhabitant of the village -/
inductive Inhabitant
| Knight
| Trickster

/-- The village with its inhabitants -/
structure Village where
  inhabitants : Fin 65 → Inhabitant
  trickster_count : Nat
  knight_count : Nat
  trickster_count_eq : trickster_count = 2
  knight_count_eq : knight_count = 63
  total_count_eq : trickster_count + knight_count = 65

/-- A question asked to an inhabitant about a group of inhabitants -/
def Question := List (Fin 65) → Bool

/-- The result of asking questions to identify tricksters -/
structure IdentificationResult where
  questions_asked : Nat
  tricksters_found : List (Fin 65)
  all_tricksters_found : tricksters_found.length = 2

/-- The main theorem stating that tricksters can be identified with no more than 30 questions -/
theorem identify_tricksters (v : Village) : 
  ∃ (strategy : List Question), 
    ∃ (result : IdentificationResult), 
      result.questions_asked ≤ 30 ∧ 
      (∀ i : Fin 65, v.inhabitants i = Inhabitant.Trickster ↔ i ∈ result.tricksters_found) :=
sorry

end NUMINAMATH_CALUDE_identify_tricksters_l2162_216257


namespace NUMINAMATH_CALUDE_detergent_in_altered_solution_l2162_216271

/-- Represents the ratio of components in a cleaning solution -/
structure CleaningSolution where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- Calculates the amount of detergent in the altered solution -/
def altered_detergent_amount (original : CleaningSolution) (water_amount : ℚ) : ℚ :=
  let new_bleach := original.bleach * 3
  let new_detergent := original.detergent
  let new_water := original.water * 2
  let total_parts := new_bleach + new_detergent + new_water
  (new_detergent / total_parts) * water_amount

/-- Theorem stating the amount of detergent in the altered solution -/
theorem detergent_in_altered_solution 
  (original : CleaningSolution)
  (h1 : original.bleach = 2)
  (h2 : original.detergent = 25)
  (h3 : original.water = 100)
  (water_amount : ℚ)
  (h4 : water_amount = 300) :
  altered_detergent_amount original water_amount = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_detergent_in_altered_solution_l2162_216271


namespace NUMINAMATH_CALUDE_pole_length_reduction_l2162_216235

theorem pole_length_reduction (original_length current_length : ℝ) 
  (h1 : original_length = 20)
  (h2 : current_length = 14) :
  (original_length - current_length) / original_length * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_pole_length_reduction_l2162_216235


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l2162_216254

theorem fraction_ratio_equality : ∃ x : ℚ, (x / (2/6) = (3/4) / (1/2)) ∧ (x = 2/9) := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l2162_216254


namespace NUMINAMATH_CALUDE_inequality_range_l2162_216258

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x) ↔ 
  (-2 < m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l2162_216258


namespace NUMINAMATH_CALUDE_fill_container_l2162_216266

/-- The capacity of a standard jar in milliliters -/
def standard_jar_capacity : ℕ := 60

/-- The capacity of the big container in milliliters -/
def big_container_capacity : ℕ := 840

/-- The minimum number of standard jars needed to fill the big container -/
def min_jars_needed : ℕ := 14

theorem fill_container :
  min_jars_needed = (big_container_capacity + standard_jar_capacity - 1) / standard_jar_capacity :=
by sorry

end NUMINAMATH_CALUDE_fill_container_l2162_216266


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2162_216213

theorem arithmetic_sequence_sum_ratio (a₁ d : ℚ) : 
  let S : ℕ → ℚ := λ n => n * a₁ + n * (n - 1) / 2 * d
  (S 3) / (S 7) = 1 / 3 → (S 6) / (S 7) = 17 / 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2162_216213


namespace NUMINAMATH_CALUDE_jennas_number_l2162_216262

theorem jennas_number (x : ℝ) : 3 * ((3 * x + 20) - 5) = 225 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_jennas_number_l2162_216262


namespace NUMINAMATH_CALUDE_perfect_square_theorem_l2162_216293

theorem perfect_square_theorem (a b c d : ℤ) : 
  d = (a + Real.rpow 2 (1/3 : ℝ) * b + Real.rpow 4 (1/3 : ℝ) * c)^2 → 
  ∃ k : ℤ, d = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_theorem_l2162_216293
