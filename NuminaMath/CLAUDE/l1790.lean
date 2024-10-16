import Mathlib

namespace NUMINAMATH_CALUDE_mechanic_bill_calculation_l1790_179078

/-- Given a mechanic's hourly rate, parts cost, and hours worked, calculate the total bill -/
def total_bill (hourly_rate : ℕ) (parts_cost : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_rate * hours_worked + parts_cost

/-- Theorem: The total bill for a 5-hour job with $45/hour rate and $225 parts cost is $450 -/
theorem mechanic_bill_calculation :
  total_bill 45 225 5 = 450 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_bill_calculation_l1790_179078


namespace NUMINAMATH_CALUDE_inequality_satisfied_by_five_integers_l1790_179013

theorem inequality_satisfied_by_five_integers :
  ∃! (S : Finset ℤ), (∀ n ∈ S, Real.sqrt (n + 1 : ℝ) ≤ Real.sqrt (5 * n - 7 : ℝ) ∧
                               Real.sqrt (5 * n - 7 : ℝ) < Real.sqrt (3 * n + 6 : ℝ)) ∧
                     S.card = 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_satisfied_by_five_integers_l1790_179013


namespace NUMINAMATH_CALUDE_girls_not_participating_count_l1790_179012

/-- Represents the number of students in an extracurricular activity -/
structure Activity where
  total : ℕ
  boys : ℕ
  girls : ℕ

/-- Represents the school's student body and activities -/
structure School where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  soccer : Activity
  basketball : Activity
  chess : Activity
  math : Activity
  glee : Activity
  absent_boys : ℕ
  absent_girls : ℕ

/-- The number of girls not participating in any extracurricular activities -/
def girls_not_participating (s : School) : ℕ :=
  s.total_girls - s.soccer.girls - s.basketball.girls - s.chess.girls - s.math.girls - s.glee.girls - s.absent_girls

theorem girls_not_participating_count (s : School) :
  s.total_students = 800 ∧
  s.total_boys = 420 ∧
  s.total_girls = 380 ∧
  s.soccer.total = 320 ∧
  s.soccer.boys = 224 ∧
  s.basketball.total = 280 ∧
  s.basketball.girls = 182 ∧
  s.chess.total = 70 ∧
  s.chess.boys = 56 ∧
  s.math.total = 50 ∧
  s.math.boys = 25 ∧
  s.math.girls = 25 ∧
  s.absent_boys = 21 ∧
  s.absent_girls = 30 →
  girls_not_participating s = 33 := by
  sorry


end NUMINAMATH_CALUDE_girls_not_participating_count_l1790_179012


namespace NUMINAMATH_CALUDE_camel_cost_l1790_179016

/-- Proves that the cost of one camel is 5600 given the specified conditions --/
theorem camel_cost (camel horse ox elephant : ℕ → ℚ) 
  (h1 : 10 * camel 1 = 24 * horse 1)
  (h2 : 16 * horse 1 = 4 * ox 1)
  (h3 : 6 * ox 1 = 4 * elephant 1)
  (h4 : 10 * elephant 1 = 140000) : 
  camel 1 = 5600 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l1790_179016


namespace NUMINAMATH_CALUDE_triangle_point_C_l1790_179040

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

def isMedian (t : Triangle) (M : Point) : Prop :=
  M.x = (t.A.x + t.B.x) / 2 ∧ M.y = (t.A.y + t.B.y) / 2

def isAngleBisector (t : Triangle) (L : Point) : Prop :=
  -- We can't define this precisely without more geometric functions,
  -- so we'll leave it as an axiom for now
  True

theorem triangle_point_C (t : Triangle) (M L : Point) :
  t.A = Point.mk 2 8 →
  M = Point.mk 4 11 →
  L = Point.mk 6 6 →
  isMedian t M →
  isAngleBisector t L →
  t.C = Point.mk 14 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_point_C_l1790_179040


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_l1790_179069

-- Define the conditions
def condition_p (t : ℝ) : Prop := ∀ x : ℝ, (1/2) * x^2 - t*x + 1/2 > 0

def condition_q (t a : ℝ) : Prop := t^2 - (a-1)*t - a < 0

-- Theorem 1
theorem theorem_1 (t : ℝ) : condition_p t → -1 < t ∧ t < 1 := by sorry

-- Theorem 2
theorem theorem_2 (a : ℝ) : 
  (∀ t : ℝ, condition_p t → condition_q t a) ∧ 
  (∃ t : ℝ, condition_q t a ∧ ¬condition_p t) → 
  a > 1 := by sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_l1790_179069


namespace NUMINAMATH_CALUDE_line_perp_plane_condition_l1790_179087

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the relation of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- The theorem statement
theorem line_perp_plane_condition 
  (m n : Line) (α β : Plane) 
  (h1 : perp_planes α β)
  (h2 : intersect α β = m)
  (h3 : contained_in n α) :
  perp_line_plane n β ↔ perp_lines n m :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_condition_l1790_179087


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l1790_179091

def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_in_second_quadrant :
  let z : ℂ := -2 + I
  second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l1790_179091


namespace NUMINAMATH_CALUDE_sin_15_cos_15_value_l1790_179084

theorem sin_15_cos_15_value : (1/4) * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_cos_15_value_l1790_179084


namespace NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_three_l1790_179011

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given that z = (a^2 - 9) + (a + 3)i is a pure imaginary number, prove that a = 3. -/
theorem pure_imaginary_implies_a_eq_three (a : ℝ) 
    (h : IsPureImaginary ((a^2 - 9) + (a + 3)*I)) : a = 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_implies_a_eq_three_l1790_179011


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l1790_179079

theorem expression_equals_negative_one (x y : ℝ) 
  (hx : x ≠ 0) (hxy : x ≠ 2*y ∧ x ≠ -2*y) : 
  (x / (x + 2*y) + 2*y / (x - 2*y)) / (2*y / (x + 2*y) - x / (x - 2*y)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l1790_179079


namespace NUMINAMATH_CALUDE_nathans_blanket_temp_l1790_179066

def initial_temp : ℝ := 50
def type_a_effect : ℝ := 2
def type_b_effect : ℝ := 3
def total_type_a : ℕ := 8
def used_type_a : ℕ := total_type_a / 2
def total_type_b : ℕ := 6

theorem nathans_blanket_temp :
  initial_temp + (used_type_a : ℝ) * type_a_effect + (total_type_b : ℝ) * type_b_effect = 76 := by
  sorry

end NUMINAMATH_CALUDE_nathans_blanket_temp_l1790_179066


namespace NUMINAMATH_CALUDE_sundae_cost_theorem_l1790_179047

/-- The cost of ice cream in dollars -/
def ice_cream_cost : ℚ := 2

/-- The cost of one topping in dollars -/
def topping_cost : ℚ := 1/2

/-- The number of toppings on the sundae -/
def num_toppings : ℕ := 10

/-- The total cost of a sundae with given number of toppings -/
def sundae_cost (ice_cream : ℚ) (topping : ℚ) (num : ℕ) : ℚ :=
  ice_cream + topping * num

theorem sundae_cost_theorem :
  sundae_cost ice_cream_cost topping_cost num_toppings = 7 := by
  sorry

end NUMINAMATH_CALUDE_sundae_cost_theorem_l1790_179047


namespace NUMINAMATH_CALUDE_day_after_tomorrow_l1790_179050

/-- Represents days of the week -/
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns the next day -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

/-- Returns the previous day -/
def prevDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Saturday
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday

theorem day_after_tomorrow (today : Day) :
  (nextDay (nextDay today) = Day.Saturday) → (today = Day.Thursday) →
  (prevDay (nextDay (nextDay today)) = Day.Friday) :=
by
  sorry


end NUMINAMATH_CALUDE_day_after_tomorrow_l1790_179050


namespace NUMINAMATH_CALUDE_class_size_from_marking_error_l1790_179022

/-- The number of pupils in a class where a marking error occurred -/
def number_of_pupils : ℕ := 16

/-- The incorrect mark entered for a pupil -/
def incorrect_mark : ℕ := 73

/-- The correct mark for the pupil -/
def correct_mark : ℕ := 65

/-- The increase in class average due to the error -/
def average_increase : ℚ := 1/2

theorem class_size_from_marking_error :
  (incorrect_mark - correct_mark : ℚ) = number_of_pupils * average_increase :=
sorry

end NUMINAMATH_CALUDE_class_size_from_marking_error_l1790_179022


namespace NUMINAMATH_CALUDE_quadratic_real_roots_quadratic_integer_roots_l1790_179092

/-- The quadratic equation kx^2 + (k+1)x + (k-1) = 0 -/
def quadratic_equation (k x : ℝ) : Prop :=
  k * x^2 + (k + 1) * x + (k - 1) = 0

/-- The set of k values for which the equation has real roots -/
def real_roots_set : Set ℝ :=
  {k | (3 - 2 * Real.sqrt 3) / 3 ≤ k ∧ k ≤ (3 + 2 * Real.sqrt 3) / 3}

/-- The set of k values for which the equation has integer roots -/
def integer_roots_set : Set ℝ :=
  {0, 1, -1/7}

theorem quadratic_real_roots :
  ∀ k : ℝ, (∃ x : ℝ, quadratic_equation k x) ↔ k ∈ real_roots_set :=
sorry

theorem quadratic_integer_roots :
  ∀ k : ℝ, (∃ x : ℤ, quadratic_equation k (x : ℝ)) ↔ k ∈ integer_roots_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_quadratic_integer_roots_l1790_179092


namespace NUMINAMATH_CALUDE_wall_height_calculation_l1790_179095

/-- Calculates the height of a wall given its dimensions and the number and size of bricks used --/
theorem wall_height_calculation (brick_length brick_width brick_height : ℝ)
  (wall_length wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 20 →
  brick_width = 10 →
  brick_height = 7.5 →
  wall_length = 27 →
  wall_width = 2 →
  num_bricks = 27000 →
  ∃ (wall_height : ℝ), wall_height = 0.75 ∧
    wall_length * wall_width * wall_height = (brick_length * brick_width * brick_height * num_bricks) / 1000000 := by
  sorry

#check wall_height_calculation

end NUMINAMATH_CALUDE_wall_height_calculation_l1790_179095


namespace NUMINAMATH_CALUDE_inequality_and_minimum_l1790_179083

theorem inequality_and_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y + 1/(x*y) ≥ 8) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_l1790_179083


namespace NUMINAMATH_CALUDE_recurring_decimal_calculation_l1790_179060

theorem recurring_decimal_calculation : ∀ (x y : ℚ),
  x = 1/3 → y = 1 → (8 * x) / y = 8/3 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_calculation_l1790_179060


namespace NUMINAMATH_CALUDE_cost_per_item_l1790_179089

theorem cost_per_item (total_customers : ℕ) (purchase_percentage : ℚ) (total_profit : ℚ) : 
  total_customers = 100 → 
  purchase_percentage = 80 / 100 → 
  total_profit = 1000 → 
  total_profit / (total_customers * purchase_percentage) = 25 / 2 := by
sorry

end NUMINAMATH_CALUDE_cost_per_item_l1790_179089


namespace NUMINAMATH_CALUDE_unique_triple_solution_l1790_179046

theorem unique_triple_solution : 
  ∃! (n p q : ℕ), n ≥ 2 ∧ n^p + n^q = n^2010 ∧ p = 2009 ∧ q = 2009 ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l1790_179046


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l1790_179081

/-- Given a quadratic equation x^2 - 3x + k = 0 with one root being 4,
    prove that the other root is -1 and k = -4 -/
theorem quadratic_equation_proof (k : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + k = 0 ∧ x = 4) → 
  (∃ y : ℝ, y^2 - 3*y + k = 0 ∧ y = -1) ∧ k = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l1790_179081


namespace NUMINAMATH_CALUDE_proposition_logic_l1790_179054

theorem proposition_logic (p q : Prop) : 
  (((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q))) ∧
  ((¬(p ∧ q) → p) ∧ ¬(p → ¬(p ∧ q))) := by
  sorry

end NUMINAMATH_CALUDE_proposition_logic_l1790_179054


namespace NUMINAMATH_CALUDE_barycentric_coordinate_properties_l1790_179039

/-- Barycentric coordinates in a tetrahedron -/
structure BarycentricCoord :=
  (x₁ x₂ x₃ x₄ : ℝ)
  (sum_to_one : x₁ + x₂ + x₃ + x₄ = 1)

/-- The tetrahedron A₁A₂A₃A₄ -/
structure Tetrahedron :=
  (A₁ A₂ A₃ A₄ : BarycentricCoord)

/-- A point lies on line A₁A₂ iff x₃ = 0 and x₄ = 0 -/
def lies_on_line_A₁A₂ (t : Tetrahedron) (p : BarycentricCoord) : Prop :=
  p.x₃ = 0 ∧ p.x₄ = 0

/-- A point lies on plane A₁A₂A₃ iff x₄ = 0 -/
def lies_on_plane_A₁A₂A₃ (t : Tetrahedron) (p : BarycentricCoord) : Prop :=
  p.x₄ = 0

/-- A point lies on the plane through A₃A₄ parallel to A₁A₂ iff x₁ = -x₂ and x₃ + x₄ = 1 -/
def lies_on_plane_parallel_A₁A₂_through_A₃A₄ (t : Tetrahedron) (p : BarycentricCoord) : Prop :=
  p.x₁ = -p.x₂ ∧ p.x₃ + p.x₄ = 1

theorem barycentric_coordinate_properties (t : Tetrahedron) (p : BarycentricCoord) :
  (lies_on_line_A₁A₂ t p ↔ p.x₃ = 0 ∧ p.x₄ = 0) ∧
  (lies_on_plane_A₁A₂A₃ t p ↔ p.x₄ = 0) ∧
  (lies_on_plane_parallel_A₁A₂_through_A₃A₄ t p ↔ p.x₁ = -p.x₂ ∧ p.x₃ + p.x₄ = 1) := by
  sorry

end NUMINAMATH_CALUDE_barycentric_coordinate_properties_l1790_179039


namespace NUMINAMATH_CALUDE_logarithm_simplification_l1790_179036

theorem logarithm_simplification 
  (p q r s y z : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (hy : y > 0) (hz : z > 0) : 
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log (p * z / (s * y)) = Real.log (y / z) :=
sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l1790_179036


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1790_179058

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 9 * x₁ + 7 = 0) → 
  (2 * x₂^2 - 9 * x₂ + 7 = 0) → 
  x₁^2 + x₂^2 = 53/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1790_179058


namespace NUMINAMATH_CALUDE_point_on_number_line_l1790_179062

theorem point_on_number_line (A : ℝ) : (|A| = 3) ↔ (A = 3 ∨ A = -3) := by
  sorry

end NUMINAMATH_CALUDE_point_on_number_line_l1790_179062


namespace NUMINAMATH_CALUDE_rth_term_of_sequence_l1790_179008

-- Define the sum function for the arithmetic progression
def S (n : ℕ) : ℕ := 5 * n + 4 * n^2

-- Define the rth term of the sequence
def a_r (r : ℕ) : ℕ := S r - S (r - 1)

-- Theorem statement
theorem rth_term_of_sequence (r : ℕ) (h : r > 0) : a_r r = 8 * r + 1 := by
  sorry

end NUMINAMATH_CALUDE_rth_term_of_sequence_l1790_179008


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l1790_179010

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l1790_179010


namespace NUMINAMATH_CALUDE_light_bulb_probability_l1790_179059

theorem light_bulb_probability (pass_rate : ℝ) (h1 : 0 ≤ pass_rate ∧ pass_rate ≤ 1) :
  pass_rate = 0.99 → 
  ∃ (P : Set ℝ → ℝ), 
    (∀ A, 0 ≤ P A ∧ P A ≤ 1) ∧ 
    (P ∅ = 0) ∧ 
    (P univ = 1) ∧
    P {x | x ≤ pass_rate} = 0.99 :=
by sorry

end NUMINAMATH_CALUDE_light_bulb_probability_l1790_179059


namespace NUMINAMATH_CALUDE_min_value_of_geometric_sequence_l1790_179038

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem min_value_of_geometric_sequence (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_condition : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  ∃ min_value : ℝ, min_value = 54 ∧ 
  ∀ x : ℝ, (∃ a : ℕ → ℝ, geometric_sequence a ∧ 
    (∀ n : ℕ, a n > 0) ∧ 
    2 * a 4 + a 3 - 2 * a 2 - a 1 = 8 ∧
    2 * a 8 + a 7 = x) → x ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_min_value_of_geometric_sequence_l1790_179038


namespace NUMINAMATH_CALUDE_P_neither_sufficient_nor_necessary_for_Q_l1790_179017

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := x^2 + (a-2)*x + 2*a - 8

-- Define the condition P
def condition_P (a : ℝ) : Prop := -1 < a ∧ a < 1

-- Define the condition Q
def condition_Q (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ < 0 ∧
    quadratic_equation a x₁ = 0 ∧ quadratic_equation a x₂ = 0

-- Theorem stating that P is neither sufficient nor necessary for Q
theorem P_neither_sufficient_nor_necessary_for_Q :
  (¬∀ a : ℝ, condition_P a → condition_Q a) ∧
  (¬∀ a : ℝ, condition_Q a → condition_P a) :=
sorry

end NUMINAMATH_CALUDE_P_neither_sufficient_nor_necessary_for_Q_l1790_179017


namespace NUMINAMATH_CALUDE_shortest_distance_to_mount_fuji_l1790_179048

theorem shortest_distance_to_mount_fuji (a b c h : ℝ) : 
  a = 60 → b = 45 → c^2 = a^2 + b^2 → h * c = a * b → h = 36 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_to_mount_fuji_l1790_179048


namespace NUMINAMATH_CALUDE_no_other_products_of_three_primes_l1790_179007

/-- The reverse of a natural number -/
def reverse (n : ℕ) : ℕ := sorry

/-- Predicate for a number being the product of exactly three distinct primes -/
def isProductOfThreeDistinctPrimes (n : ℕ) : Prop := sorry

theorem no_other_products_of_three_primes : 
  let original := 2017
  let reversed := 7102
  -- 7102 is the reverse of 2017
  reverse original = reversed →
  -- 7102 is the product of three distinct primes p, q, and r
  ∃ (p q r : ℕ), isProductOfThreeDistinctPrimes reversed ∧ 
                 reversed = p * q * r ∧ 
                 p ≠ q ∧ p ≠ r ∧ q ≠ r →
  -- There are no other positive integers that are products of three distinct primes 
  -- summing to the same value as p + q + r
  ¬∃ (n : ℕ), n ≠ reversed ∧ 
              isProductOfThreeDistinctPrimes n ∧
              (∃ (p1 p2 p3 : ℕ), n = p1 * p2 * p3 ∧ 
                                 p1 + p2 + p3 = p + q + r ∧
                                 p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3) :=
by sorry

end NUMINAMATH_CALUDE_no_other_products_of_three_primes_l1790_179007


namespace NUMINAMATH_CALUDE_ping_pong_game_ratio_l1790_179045

/-- Given that Frankie and Carla played 30 games of ping pong,
    and Carla won 20 games, prove that the ratio of games
    Frankie won to games Carla won is 1:2. -/
theorem ping_pong_game_ratio :
  let total_games : ℕ := 30
  let carla_wins : ℕ := 20
  let frankie_wins : ℕ := total_games - carla_wins
  (frankie_wins : ℚ) / (carla_wins : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_game_ratio_l1790_179045


namespace NUMINAMATH_CALUDE_y_percentage_more_than_z_l1790_179030

/-- Proves that given the conditions, y gets 20% more than z -/
theorem y_percentage_more_than_z (total : ℝ) (z_share : ℝ) (x_more_than_y : ℝ) :
  total = 1480 →
  z_share = 400 →
  x_more_than_y = 0.25 →
  (((total - z_share) / (2 + x_more_than_y) - z_share) / z_share) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_y_percentage_more_than_z_l1790_179030


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_cos_squared_alpha_l1790_179031

theorem sin_2alpha_minus_cos_squared_alpha (α : Real) (h : Real.tan α = 2) :
  Real.sin (2 * α) - Real.cos α ^ 2 = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_cos_squared_alpha_l1790_179031


namespace NUMINAMATH_CALUDE_equation_roots_range_l1790_179077

theorem equation_roots_range (n : ℕ) (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   2*n - 1 < x₁ ∧ x₁ ≤ 2*n + 1 ∧
   2*n - 1 < x₂ ∧ x₂ ≤ 2*n + 1 ∧
   |x₁ - 2*n| = k * Real.sqrt x₁ ∧
   |x₂ - 2*n| = k * Real.sqrt x₂) →
  (0 < k ∧ k ≤ 1 / Real.sqrt (2*n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_range_l1790_179077


namespace NUMINAMATH_CALUDE_total_distance_after_five_days_l1790_179019

/-- The total distance run by Peter and Andrew after 5 days -/
def total_distance (andrew_distance : ℕ) (peter_extra : ℕ) (days : ℕ) : ℕ :=
  (andrew_distance + peter_extra + andrew_distance) * days

/-- Theorem stating the total distance run by Peter and Andrew after 5 days -/
theorem total_distance_after_five_days :
  total_distance 2 3 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_after_five_days_l1790_179019


namespace NUMINAMATH_CALUDE_area_relationship_l1790_179076

/-- A right triangle with sides 18, 24, and 30 -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right : side1^2 + side2^2 = hypotenuse^2
  side1_eq : side1 = 18
  side2_eq : side2 = 24
  hypotenuse_eq : hypotenuse = 30

/-- Areas of non-triangular regions in a circumscribed circle -/
structure CircleAreas where
  D : ℝ
  E : ℝ
  F : ℝ
  F_largest : F ≥ D ∧ F ≥ E

/-- Theorem stating the relationship between areas D, E, F, and the triangle area -/
theorem area_relationship (t : RightTriangle) (areas : CircleAreas) :
  areas.D + areas.E + 216 = areas.F := by
  sorry

end NUMINAMATH_CALUDE_area_relationship_l1790_179076


namespace NUMINAMATH_CALUDE_correct_admin_in_sample_l1790_179071

/-- Represents the composition of staff in a school -/
structure StaffComposition where
  total : ℕ
  administrative : ℕ
  teaching : ℕ
  teaching_support : ℕ

/-- Represents a stratified sample from the staff -/
structure StratifiedSample where
  size : ℕ
  administrative : ℕ

/-- Calculates the correct number of administrative personnel in a stratified sample -/
def calculate_admin_in_sample (staff : StaffComposition) (sample : StratifiedSample) : ℕ :=
  (staff.administrative * sample.size) / staff.total

/-- The theorem to be proved -/
theorem correct_admin_in_sample (staff : StaffComposition) (sample : StratifiedSample) : 
  staff.total = 200 →
  staff.administrative = 24 →
  staff.teaching = 10 * staff.teaching_support →
  sample.size = 50 →
  calculate_admin_in_sample staff sample = 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_admin_in_sample_l1790_179071


namespace NUMINAMATH_CALUDE_uv_length_in_triangle_l1790_179015

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)
  (xy_length : Real)
  (xz_length : Real)
  (yz_length : Real)

-- Define the angle bisector points S and T
structure AngleBisectorPoints :=
  (S : ℝ × ℝ)
  (T : ℝ × ℝ)

-- Define the perpendicular feet U and V
structure PerpendicularFeet :=
  (U : ℝ × ℝ)
  (V : ℝ × ℝ)

-- Define the theorem
theorem uv_length_in_triangle (t : Triangle) (ab : AngleBisectorPoints) (pf : PerpendicularFeet) :
  t.xy_length = 140 ∧ t.xz_length = 130 ∧ t.yz_length = 150 →
  -- S is on the angle bisector of angle X and YZ
  -- T is on the angle bisector of angle Y and XZ
  -- U is the foot of the perpendicular from Z to YT
  -- V is the foot of the perpendicular from Z to XS
  Real.sqrt ((pf.U.1 - pf.V.1)^2 + (pf.U.2 - pf.V.2)^2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_uv_length_in_triangle_l1790_179015


namespace NUMINAMATH_CALUDE_farmer_crops_after_pest_destruction_l1790_179006

-- Define the constants
def corn_cobs_per_row : ℕ := 9
def potatoes_per_row : ℕ := 30
def corn_rows : ℕ := 10
def potato_rows : ℕ := 5
def pest_destruction_ratio : ℚ := 1/2

-- Define the theorem
theorem farmer_crops_after_pest_destruction :
  (corn_rows * corn_cobs_per_row + potato_rows * potatoes_per_row) * pest_destruction_ratio = 120 := by
  sorry

end NUMINAMATH_CALUDE_farmer_crops_after_pest_destruction_l1790_179006


namespace NUMINAMATH_CALUDE_max_m_value_l1790_179055

theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → x^2 - x + 1/2 ≥ m) → 
  m ≤ 1/4 := by
sorry

end NUMINAMATH_CALUDE_max_m_value_l1790_179055


namespace NUMINAMATH_CALUDE_factors_of_210_l1790_179009

def number_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem factors_of_210 : number_of_factors 210 = 16 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_210_l1790_179009


namespace NUMINAMATH_CALUDE_dot_only_count_l1790_179086

/-- Represents an alphabet with letters containing dots and straight lines -/
structure Alphabet :=
  (total : ℕ)
  (dot_and_line : ℕ)
  (line_only : ℕ)
  (all_have_dot_or_line : Prop)

/-- The number of letters containing a dot but not a straight line -/
def dot_only (α : Alphabet) : ℕ :=
  α.total - α.dot_and_line - α.line_only

/-- Theorem stating the number of letters with only a dot in the given alphabet -/
theorem dot_only_count (α : Alphabet) 
  (h1 : α.total = 40)
  (h2 : α.dot_and_line = 13)
  (h3 : α.line_only = 24) :
  dot_only α = 16 := by
  sorry

end NUMINAMATH_CALUDE_dot_only_count_l1790_179086


namespace NUMINAMATH_CALUDE_vojta_sum_problem_l1790_179023

theorem vojta_sum_problem (S A B C : ℕ) : 
  S + 10 * B + C = 2224 →
  S + 10 * A + B = 2198 →
  S + 10 * A + C = 2204 →
  A < 10 →
  B < 10 →
  C < 10 →
  S + 100 * A + 10 * B + C = 2324 :=
by sorry

end NUMINAMATH_CALUDE_vojta_sum_problem_l1790_179023


namespace NUMINAMATH_CALUDE_alex_age_l1790_179042

theorem alex_age (charlie_age alex_age : ℕ) : 
  charlie_age = 2 * alex_age + 8 → 
  charlie_age = 22 → 
  alex_age = 7 := by
sorry

end NUMINAMATH_CALUDE_alex_age_l1790_179042


namespace NUMINAMATH_CALUDE_range_of_a_l1790_179026

def P (a : ℝ) : Set ℝ := {x | a - 4 < x ∧ x < a + 4}

def Q : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem range_of_a :
  (∀ x : ℝ, x ∈ Q → x ∈ P a) → -1 ≤ a ∧ a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1790_179026


namespace NUMINAMATH_CALUDE_square_congruence_mod_four_l1790_179029

theorem square_congruence_mod_four (n : ℤ) : (n^2) % 4 = 0 ∨ (n^2) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_congruence_mod_four_l1790_179029


namespace NUMINAMATH_CALUDE_line_points_b_plus_one_l1790_179037

/-- Given a line y = 0.75x + 1 and three points on the line, prove that b + 1 = 5 -/
theorem line_points_b_plus_one (b a : ℝ) : 
  (b = 0.75 * 4 + 1) →  -- Point (4, b) on the line
  (5 = 0.75 * a + 1) →  -- Point (a, 5) on the line
  (b + 1 = 0.75 * a + 1) →  -- Point (a, b + 1) on the line
  b + 1 = 5 :=
by sorry

end NUMINAMATH_CALUDE_line_points_b_plus_one_l1790_179037


namespace NUMINAMATH_CALUDE_product_representation_l1790_179044

theorem product_representation (a : ℝ) (p : ℕ+) 
  (h1 : 12345 * 6789 = a * (10 : ℝ)^(p : ℝ))
  (h2 : 1 ≤ a ∧ a < 10) :
  p = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_representation_l1790_179044


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l1790_179033

/-- Represents a systematic sample from a range of products. -/
structure SystematicSample where
  total_products : Nat
  sample_size : Nat
  start : Nat
  step : Nat

/-- Generates a systematic sample. -/
def generateSample (s : SystematicSample) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.step)

/-- Checks if a sample is valid for the given total number of products. -/
def isValidSample (sample : List Nat) (total_products : Nat) : Prop :=
  sample.all (· < total_products) ∧ sample.length > 0 ∧ sample.Nodup

/-- Theorem: The correct systematic sample for 50 products with 5 samples is [1, 11, 21, 31, 41]. -/
theorem correct_systematic_sample :
  let sample := [1, 11, 21, 31, 41]
  let s : SystematicSample := {
    total_products := 50,
    sample_size := 5,
    start := 1,
    step := 10
  }
  generateSample s = sample ∧ isValidSample sample s.total_products := by
  sorry


end NUMINAMATH_CALUDE_correct_systematic_sample_l1790_179033


namespace NUMINAMATH_CALUDE_exists_scores_with_median_16_l1790_179090

/-- Represents a set of basketball scores -/
def BasketballScores := List ℕ

/-- Calculates the median of a list of natural numbers -/
def median (scores : BasketballScores) : ℚ :=
  sorry

/-- Theorem: There exists a set of basketball scores with a median of 16 -/
theorem exists_scores_with_median_16 : 
  ∃ (scores : BasketballScores), median scores = 16 := by
  sorry

end NUMINAMATH_CALUDE_exists_scores_with_median_16_l1790_179090


namespace NUMINAMATH_CALUDE_rhombus_property_l1790_179064

structure Rhombus (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D : V)
  (is_rhombus : (B - A) = (C - B) ∧ (C - B) = (D - C) ∧ (D - C) = (A - D))

theorem rhombus_property {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (ABCD : Rhombus V) (E F P Q : V) :
  (∃ t : ℝ, E = ABCD.A + t • (ABCD.B - ABCD.A)) →
  (∃ s : ℝ, F = ABCD.A + s • (ABCD.D - ABCD.A)) →
  (ABCD.A - E = ABCD.D - F) →
  (∃ u : ℝ, P = ABCD.B + u • (ABCD.C - ABCD.B)) →
  (∃ v : ℝ, P = ABCD.D + v • (E - ABCD.D)) →
  (∃ w : ℝ, Q = ABCD.C + w • (ABCD.D - ABCD.C)) →
  (∃ x : ℝ, Q = ABCD.B + x • (F - ABCD.B)) →
  (∃ y z : ℝ, P - E = y • (P - ABCD.D) ∧ Q - F = z • (Q - ABCD.B) ∧ y + z = 1) ∧
  (∃ a : ℝ, ABCD.A - P = a • (Q - P)) :=
sorry

end NUMINAMATH_CALUDE_rhombus_property_l1790_179064


namespace NUMINAMATH_CALUDE_equation_solutions_l1790_179094

theorem equation_solutions (n : ℕ+) : 
  (∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    s.card = 10 ∧ 
    ∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ 3*x + 3*y + 2*z = n) ↔ 
  n = 17 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1790_179094


namespace NUMINAMATH_CALUDE_bart_firewood_needs_l1790_179051

/-- The number of pieces of firewood obtained from one tree -/
def pieces_per_tree : ℕ := 75

/-- The number of logs burned per day -/
def logs_per_day : ℕ := 5

/-- The number of days from November 1 through February 28 -/
def total_days : ℕ := 120

/-- The number of trees Bart needs to cut down -/
def trees_needed : ℕ := 8

theorem bart_firewood_needs :
  trees_needed = (total_days * logs_per_day + pieces_per_tree - 1) / pieces_per_tree :=
by sorry

end NUMINAMATH_CALUDE_bart_firewood_needs_l1790_179051


namespace NUMINAMATH_CALUDE_smallest_k_sum_digits_l1790_179097

/-- Sum of digits function -/
def s (n : ℕ) : ℕ := sorry

/-- Theorem stating that 9999 is the smallest positive integer k satisfying the condition -/
theorem smallest_k_sum_digits : 
  (∀ m : ℕ, m ∈ Finset.range 2014 → s ((m + 1) * 9999) = s 9999) ∧ 
  (∀ k : ℕ, k < 9999 → ∃ m : ℕ, m ∈ Finset.range 2014 ∧ s ((m + 1) * k) ≠ s k) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_sum_digits_l1790_179097


namespace NUMINAMATH_CALUDE_least_x_for_even_prime_fraction_l1790_179027

theorem least_x_for_even_prime_fraction (x p : ℕ) : 
  x > 0 → 
  Nat.Prime p → 
  (x / (12 * p) = 2) → 
  (∀ y : ℕ, y > 0 → Nat.Prime p → (y / (12 * p) = 2) → y ≥ x) → 
  x = 48 := by
sorry

end NUMINAMATH_CALUDE_least_x_for_even_prime_fraction_l1790_179027


namespace NUMINAMATH_CALUDE_complex_power_sum_l1790_179057

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^3000 + 1/z^3000 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1790_179057


namespace NUMINAMATH_CALUDE_card_cost_correct_l1790_179018

/-- The cost of cards in the first box -/
def cost_box1 : ℝ := 1.25

/-- The cost of cards in the second box -/
def cost_box2 : ℝ := 1.75

/-- The number of cards bought from each box -/
def cards_per_box : ℕ := 6

/-- The total amount spent -/
def total_spent : ℝ := 18

/-- Theorem stating that the cost of cards in the first box is correct -/
theorem card_cost_correct : 
  cost_box1 * cards_per_box + cost_box2 * cards_per_box = total_spent := by
  sorry

end NUMINAMATH_CALUDE_card_cost_correct_l1790_179018


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1790_179000

-- System 1
theorem system_one_solution (x y : ℝ) :
  (3 * x + 2 * y = 10 ∧ x / 2 - (y + 1) / 3 = 1) →
  (x = 3 ∧ y = 1/2) :=
by sorry

-- System 2
theorem system_two_solution (x y : ℝ) :
  (4 * x - 5 * y = 3 ∧ (x - 2 * y) / 0.4 = 0.6) →
  (x = 1.6 ∧ y = 0.68) :=
by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l1790_179000


namespace NUMINAMATH_CALUDE_expression_evaluation_l1790_179002

theorem expression_evaluation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / (a * b) - (a^2 + a * b) / (a^2 + b^2) = 
  (a^4 + b^4 + a^2 * b^2 - a^2 * b - a * b^2) / (a * b * (a^2 + b^2)) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1790_179002


namespace NUMINAMATH_CALUDE_committee_selection_count_l1790_179020

theorem committee_selection_count : Nat.choose 12 7 = 792 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_count_l1790_179020


namespace NUMINAMATH_CALUDE_min_value_implies_a_l1790_179080

/-- Given a function f(x) = 4x + a/x where x > 0 and a > 0,
    if the function takes its minimum value at x = 2,
    then a = 16 -/
theorem min_value_implies_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, x > 0 → 4*x + a/x ≥ 4*2 + a/2) →
  (∀ x : ℝ, x > 0 → x ≠ 2 → 4*x + a/x > 4*2 + a/2) →
  a = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l1790_179080


namespace NUMINAMATH_CALUDE_arrangement_from_combination_l1790_179025

theorem arrangement_from_combination (n : ℕ) (h1 : n ≥ 2) (h2 : Nat.choose n 2 = 15) : 
  n * (n - 1) = 30 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_from_combination_l1790_179025


namespace NUMINAMATH_CALUDE_wood_length_after_sawing_l1790_179088

theorem wood_length_after_sawing (original_length saw_length : Real) 
  (h1 : original_length = 0.41)
  (h2 : saw_length = 0.33) :
  original_length - saw_length = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_after_sawing_l1790_179088


namespace NUMINAMATH_CALUDE_sum_trailing_zeros_15_factorial_l1790_179072

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def trailingZerosBase10 (n : ℕ) : ℕ := 
  (List.range 5).foldl (fun acc i => acc + n / (5 ^ (i + 1))) 0

def trailingZerosBase12 (n : ℕ) : ℕ := 
  min 
    ((List.range 2).foldl (fun acc i => acc + n / (3 ^ (i + 1))) 0)
    ((List.range 3).foldl (fun acc i => acc + n / (2 ^ (i + 1))) 0 / 2)

theorem sum_trailing_zeros_15_factorial : 
  trailingZerosBase12 (factorial 15) + trailingZerosBase10 (factorial 15) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_trailing_zeros_15_factorial_l1790_179072


namespace NUMINAMATH_CALUDE_sum_of_zeros_infimum_l1790_179073

noncomputable section

def f (x : ℝ) : ℝ := if x ≥ 1 then Real.log x else 1 - x / 2

def F (m : ℝ) (x : ℝ) : ℝ := f (f x + 1) + m

theorem sum_of_zeros_infimum (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F m x₁ = 0 ∧ F m x₂ = 0) →
  ∃ s : ℝ, s = 4 - 2 * Real.log 2 ∧ ∀ x₁ x₂ : ℝ, F m x₁ = 0 → F m x₂ = 0 → x₁ + x₂ ≥ s :=
sorry

end NUMINAMATH_CALUDE_sum_of_zeros_infimum_l1790_179073


namespace NUMINAMATH_CALUDE_smaller_octagon_area_ratio_l1790_179098

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The smaller octagon formed by connecting midpoints of sides of a regular octagon -/
def smallerOctagon (oct : RegularOctagon) : RegularOctagon := sorry

/-- The area of a regular octagon -/
def area (oct : RegularOctagon) : ℝ := sorry

/-- Theorem: The area of the smaller octagon is half the area of the larger octagon -/
theorem smaller_octagon_area_ratio (oct : RegularOctagon) : 
  area (smallerOctagon oct) = (1/2 : ℝ) * area oct := by sorry

end NUMINAMATH_CALUDE_smaller_octagon_area_ratio_l1790_179098


namespace NUMINAMATH_CALUDE_function_composition_equality_l1790_179005

theorem function_composition_equality (b : ℚ) : 
  let p : ℚ → ℚ := λ x => 3 * x - 5
  let q : ℚ → ℚ := λ x => 4 * x - b
  p (q 3) = 9 → b = 22 / 3 := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1790_179005


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l1790_179096

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (vote_difference : ℕ) 
  (candidate_percentage : ℚ) :
  total_votes = 7900 →
  vote_difference = 2370 →
  candidate_percentage = total_votes.cast / 100 * 
    ((total_votes.cast - vote_difference.cast) / (2 * total_votes.cast)) →
  candidate_percentage = 35 := by
sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l1790_179096


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l1790_179052

theorem half_abs_diff_squares_21_19 : 
  (1 / 2 : ℝ) * |21^2 - 19^2| = 40 := by sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_21_19_l1790_179052


namespace NUMINAMATH_CALUDE_a_5_equals_5_l1790_179041

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  h1 : a 3 + a 11 = 18  -- Condition 1
  h2 : (a 1 + a 2 + a 3) = -3  -- Condition 2 (S₃ = -3)

/-- The theorem stating that a₅ = 5 for the given arithmetic sequence -/
theorem a_5_equals_5 (seq : ArithmeticSequence) : seq.a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_a_5_equals_5_l1790_179041


namespace NUMINAMATH_CALUDE_no_first_quadrant_intersection_l1790_179085

/-- A linear function y = -3x + m -/
def linear_function (x : ℝ) (m : ℝ) : ℝ := -3 * x + m

/-- The first quadrant of the coordinate plane -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

theorem no_first_quadrant_intersection :
  ∀ x y : ℝ, first_quadrant x y → linear_function x (-1) ≠ y := by
  sorry

end NUMINAMATH_CALUDE_no_first_quadrant_intersection_l1790_179085


namespace NUMINAMATH_CALUDE_intersection_points_line_l1790_179053

theorem intersection_points_line (s : ℝ) :
  let x : ℝ := (41 * s + 13) / 11
  let y : ℝ := -(2 * s + 6) / 11
  (2 * x - 3 * y = 8 * s + 4) ∧ 
  (x + 4 * y = 3 * s - 1) →
  y = (-22 * x + 272) / 451 := by
sorry

end NUMINAMATH_CALUDE_intersection_points_line_l1790_179053


namespace NUMINAMATH_CALUDE_complex_ratio_range_l1790_179001

theorem complex_ratio_range (x y : ℝ) :
  let z : ℂ := x + y * Complex.I
  let ratio := (z + 1) / (z + 2)
  (ratio.re / ratio.im = Real.sqrt 3) →
  (y / x ∈ Set.Icc ((Real.sqrt 3 * -3 - 4 * Real.sqrt 2) / 5) ((Real.sqrt 3 * -3 + 4 * Real.sqrt 2) / 5)) :=
by sorry

end NUMINAMATH_CALUDE_complex_ratio_range_l1790_179001


namespace NUMINAMATH_CALUDE_min_jellybeans_correct_l1790_179082

/-- The smallest number of jellybeans Alex should buy -/
def min_jellybeans : ℕ := 134

/-- Theorem stating that min_jellybeans is the smallest number satisfying the conditions -/
theorem min_jellybeans_correct :
  (min_jellybeans ≥ 120) ∧
  (min_jellybeans % 15 = 14) ∧
  (∀ n : ℕ, n ≥ 120 → n % 15 = 14 → n ≥ min_jellybeans) :=
by sorry

end NUMINAMATH_CALUDE_min_jellybeans_correct_l1790_179082


namespace NUMINAMATH_CALUDE_power_multiplication_l1790_179065

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1790_179065


namespace NUMINAMATH_CALUDE_topsoil_cost_theorem_l1790_179014

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The amount of topsoil in cubic yards -/
def amount_in_cubic_yards : ℝ := 7

/-- The cost of topsoil for a given amount of cubic yards -/
def topsoil_cost (amount : ℝ) : ℝ :=
  amount * cubic_yards_to_cubic_feet * cost_per_cubic_foot

theorem topsoil_cost_theorem :
  topsoil_cost amount_in_cubic_yards = 1512 := by
  sorry

end NUMINAMATH_CALUDE_topsoil_cost_theorem_l1790_179014


namespace NUMINAMATH_CALUDE_reverse_product_92565_l1790_179068

def is_reverse (a b : ℕ) : Prop :=
  (Nat.digits 10 a).reverse = Nat.digits 10 b

theorem reverse_product_92565 :
  ∃! (a b : ℕ), a < b ∧ is_reverse a b ∧ a * b = 92565 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_reverse_product_92565_l1790_179068


namespace NUMINAMATH_CALUDE_highest_power_prime_factorial_l1790_179067

def highest_power_of_prime (p n : ℕ) : ℕ := sorry

def sum_of_floor_divisions (p n : ℕ) : ℕ := sorry

theorem highest_power_prime_factorial (p n : ℕ) (h_prime : Nat.Prime p) :
  ∃ k : ℕ, p ^ k ≤ n ∧ n < p ^ (k + 1) ∧
  highest_power_of_prime p n = sum_of_floor_divisions p n :=
sorry

end NUMINAMATH_CALUDE_highest_power_prime_factorial_l1790_179067


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l1790_179004

theorem geometric_arithmetic_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence condition
  (a 1 + a 3 = 2 * (2 * a 2)) →  -- arithmetic sequence condition
  q = 2 - Real.sqrt 3 ∨ q = 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_l1790_179004


namespace NUMINAMATH_CALUDE_three_by_four_grid_squares_l1790_179099

/-- A structure representing a grid of squares -/
structure SquareGrid where
  rows : Nat
  cols : Nat
  total_small_squares : Nat

/-- Function to count the total number of squares in a grid -/
def count_total_squares (grid : SquareGrid) : Nat :=
  sorry

/-- Theorem stating that a 3x4 grid of 12 small squares contains 17 total squares -/
theorem three_by_four_grid_squares :
  let grid := SquareGrid.mk 3 4 12
  count_total_squares grid = 17 :=
by sorry

end NUMINAMATH_CALUDE_three_by_four_grid_squares_l1790_179099


namespace NUMINAMATH_CALUDE_squared_sum_equals_20_75_l1790_179028

theorem squared_sum_equals_20_75 
  (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 9) 
  (eq2 : b^2 + 5*c = -8) 
  (eq3 : c^2 + 7*a = -18) : 
  a^2 + b^2 + c^2 = 20.75 := by
  sorry

end NUMINAMATH_CALUDE_squared_sum_equals_20_75_l1790_179028


namespace NUMINAMATH_CALUDE_lemonade_problem_l1790_179035

theorem lemonade_problem (V : ℝ) 
  (h1 : V > 0)
  (h2 : V / 10 = V - 2 * (V / 5))
  (h3 : V / 8 = V - 2 * (V / 5 + V / 20))
  (h4 : V / 3 = V - 2 * (V / 5 + V / 20 + 5 * V / 12)) :
  V / 6 = V - (V / 3) / 2 := by sorry

end NUMINAMATH_CALUDE_lemonade_problem_l1790_179035


namespace NUMINAMATH_CALUDE_project_time_calculation_l1790_179021

/-- Calculates the remaining time for writing a report given the total time available and time spent on research and proposal. -/
def remaining_time (total_time research_time proposal_time : ℕ) : ℕ :=
  total_time - (research_time + proposal_time)

/-- Proves that given 20 hours total, 10 hours for research, and 2 hours for proposal, 
    the remaining time for writing the report is 8 hours. -/
theorem project_time_calculation :
  remaining_time 20 10 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_project_time_calculation_l1790_179021


namespace NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l1790_179093

/-- Given a parabola y^2 = -2px where p > 0, if its directrix is tangent to the circle (x-5)^2 + y^2 = 25, then p = 20 -/
theorem parabola_directrix_tangent_circle (p : ℝ) : 
  p > 0 →
  (∃ x y : ℝ, y^2 = -2*p*x) →
  (∃ x : ℝ, x = p/2 ∧ (x-5)^2 = 25) →
  p = 20 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_tangent_circle_l1790_179093


namespace NUMINAMATH_CALUDE_charity_raffle_winnings_l1790_179024

theorem charity_raffle_winnings (winnings : ℝ) : 
  (winnings / 2 - 2 = 55) → winnings = 114 := by
  sorry

end NUMINAMATH_CALUDE_charity_raffle_winnings_l1790_179024


namespace NUMINAMATH_CALUDE_vehicle_value_fraction_l1790_179003

theorem vehicle_value_fraction (value_this_year value_last_year : ℚ) 
  (h1 : value_this_year = 16000)
  (h2 : value_last_year = 20000) :
  value_this_year / value_last_year = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_vehicle_value_fraction_l1790_179003


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1790_179074

theorem arithmetic_mean_problem (x y : ℝ) : 
  (8 + x + 21 + y + 14 + 11) / 6 = 15 → x + y = 36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1790_179074


namespace NUMINAMATH_CALUDE_polygon_equal_sides_different_angles_l1790_179034

-- Define a polygon type
inductive Polygon
| Triangle
| Quadrilateral
| Pentagon

-- Function to check if a polygon can have all sides equal and all angles different
def canHaveEqualSidesAndDifferentAngles (p : Polygon) : Prop :=
  match p with
  | Polygon.Triangle => False
  | Polygon.Quadrilateral => False
  | Polygon.Pentagon => True

-- Theorem statement
theorem polygon_equal_sides_different_angles :
  (canHaveEqualSidesAndDifferentAngles Polygon.Triangle = False) ∧
  (canHaveEqualSidesAndDifferentAngles Polygon.Quadrilateral = False) ∧
  (canHaveEqualSidesAndDifferentAngles Polygon.Pentagon = True) := by
  sorry

#check polygon_equal_sides_different_angles

end NUMINAMATH_CALUDE_polygon_equal_sides_different_angles_l1790_179034


namespace NUMINAMATH_CALUDE_sufficient_conditions_for_quadratic_inequality_l1790_179070

theorem sufficient_conditions_for_quadratic_inequality :
  (∀ x, x < 4 → x^2 - 2*x - 8 < 0) ∨
  (∀ x, 0 < x ∧ x < 4 → x^2 - 2*x - 8 < 0) ∨
  (∀ x, -2 < x ∧ x < 4 → x^2 - 2*x - 8 < 0) ∨
  (∀ x, -2 < x ∧ x < 3 → x^2 - 2*x - 8 < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_conditions_for_quadratic_inequality_l1790_179070


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1790_179063

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 1)) ↔ (∃ x₀ : ℝ, x₀^2 + 1 < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1790_179063


namespace NUMINAMATH_CALUDE_boys_not_adjacent_or_ends_probability_l1790_179043

/-- The number of boys in the lineup -/
def num_boys : ℕ := 2

/-- The number of girls in the lineup -/
def num_girls : ℕ := 4

/-- The total number of people in the lineup -/
def total_people : ℕ := num_boys + num_girls

/-- The number of spaces between girls where boys can be placed -/
def available_spaces : ℕ := num_girls - 1

/-- The probability that the boys are neither adjacent nor at the ends in a lineup of boys and girls -/
theorem boys_not_adjacent_or_ends_probability :
  (num_boys.factorial * num_girls.factorial * available_spaces.choose num_boys) / total_people.factorial = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_boys_not_adjacent_or_ends_probability_l1790_179043


namespace NUMINAMATH_CALUDE_mitchell_has_30_pencils_l1790_179032

/-- The number of pencils Antonio has -/
def antonio_pencils : ℕ := sorry

/-- The number of pencils Mitchell has -/
def mitchell_pencils : ℕ := antonio_pencils + 6

/-- The total number of pencils Mitchell and Antonio have together -/
def total_pencils : ℕ := 54

theorem mitchell_has_30_pencils :
  mitchell_pencils = 30 :=
by
  sorry

#check mitchell_has_30_pencils

end NUMINAMATH_CALUDE_mitchell_has_30_pencils_l1790_179032


namespace NUMINAMATH_CALUDE_triangle_angle_zero_l1790_179061

theorem triangle_angle_zero (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  C = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_zero_l1790_179061


namespace NUMINAMATH_CALUDE_xyz_sum_and_inequality_l1790_179049

theorem xyz_sum_and_inequality (x y z : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_not_all_equal : ¬(x = y ∧ y = z))
  (h_equation : x^3 + y^3 + z^3 - 3*x*y*z - 3*(x^2 + y^2 + z^2 - x*y - y*z - z*x) = 0) :
  (x + y + z = 3) ∧ (x^2*(1 + y) + y^2*(1 + z) + z^2*(1 + x) > 6) := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_and_inequality_l1790_179049


namespace NUMINAMATH_CALUDE_opposite_sides_line_parameter_range_l1790_179075

/-- Given two points on opposite sides of a line, determine the range of the line's parameter --/
theorem opposite_sides_line_parameter_range :
  ∀ (m : ℝ),
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ = 3 ∧ y₁ = 1 ∧ x₂ = -4 ∧ y₂ = 6 ∧
    (3 * x₁ - 2 * y₁ + m) * (3 * x₂ - 2 * y₂ + m) < 0) →
  7 < m ∧ m < 24 :=
by sorry


end NUMINAMATH_CALUDE_opposite_sides_line_parameter_range_l1790_179075


namespace NUMINAMATH_CALUDE_constant_molecular_weight_l1790_179056

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight : ℝ := 3264

/-- The number of moles of the compound -/
def number_of_moles : ℝ := 8

/-- Theorem: The molecular weight of a compound remains constant regardless of the number of moles -/
theorem constant_molecular_weight : 
  molecular_weight = molecular_weight * (number_of_moles / number_of_moles) :=
by sorry

end NUMINAMATH_CALUDE_constant_molecular_weight_l1790_179056
