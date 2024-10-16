import Mathlib

namespace NUMINAMATH_CALUDE_parabola_directrix_quadratic_roots_as_eccentricities_l1534_153462

-- Define the parabola
def parabola (x y : ℝ) : Prop := x = 2 * y^2

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 - 5 * x + 2 = 0

-- Theorem for the parabola directrix
theorem parabola_directrix : ∃ (p : ℝ), ∀ (x y : ℝ), parabola x y → (x = -1/8 ↔ x = -p) := by sorry

-- Theorem for the quadratic equation roots as eccentricities
theorem quadratic_roots_as_eccentricities :
  ∃ (e₁ e₂ : ℝ), quadratic_equation e₁ ∧ quadratic_equation e₂ ∧
  (0 < e₁ ∧ e₁ < 1) ∧ (e₂ > 1) := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_quadratic_roots_as_eccentricities_l1534_153462


namespace NUMINAMATH_CALUDE_water_bottles_remaining_l1534_153412

/-- Calculates the number of bottles remaining after two days of consumption --/
def bottlesRemaining (initialBottles : ℕ) : ℕ :=
  let firstDayRemaining := initialBottles - 
    (initialBottles / 4 + initialBottles / 6 + initialBottles / 8)
  let fatherSecondDay := firstDayRemaining / 5
  let motherSecondDay := (firstDayRemaining - fatherSecondDay) / 7
  let sonSecondDay := (firstDayRemaining - fatherSecondDay - motherSecondDay) / 9
  let daughterSecondDay := (firstDayRemaining - fatherSecondDay - motherSecondDay - sonSecondDay) / 9
  firstDayRemaining - (fatherSecondDay + motherSecondDay + sonSecondDay + daughterSecondDay)

theorem water_bottles_remaining (initialBottles : ℕ) :
  initialBottles = 48 → bottlesRemaining initialBottles = 14 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_remaining_l1534_153412


namespace NUMINAMATH_CALUDE_race_winner_distance_l1534_153401

theorem race_winner_distance (catrina_distance : ℝ) (catrina_time : ℝ) 
  (sedra_distance : ℝ) (sedra_time : ℝ) (race_distance : ℝ) :
  catrina_distance = 100 ∧ 
  catrina_time = 10 ∧ 
  sedra_distance = 400 ∧ 
  sedra_time = 44 ∧ 
  race_distance = 1000 →
  let catrina_speed := catrina_distance / catrina_time
  let sedra_speed := sedra_distance / sedra_time
  let catrina_race_time := race_distance / catrina_speed
  let sedra_race_distance := sedra_speed * catrina_race_time
  race_distance - sedra_race_distance = 91 :=
by sorry

end NUMINAMATH_CALUDE_race_winner_distance_l1534_153401


namespace NUMINAMATH_CALUDE_abs_negative_two_l1534_153403

theorem abs_negative_two : abs (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_two_l1534_153403


namespace NUMINAMATH_CALUDE_complex_roots_of_quadratic_l1534_153457

theorem complex_roots_of_quadratic (a b : ℝ) : 
  (Complex.I + 1) ^ 2 + a * (Complex.I + 1) + b = 0 → 
  (a = -2 ∧ b = 2) ∧ (Complex.I - 1) ^ 2 + a * (Complex.I - 1) + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_of_quadratic_l1534_153457


namespace NUMINAMATH_CALUDE_average_mark_five_subjects_l1534_153463

/-- Given a student's marks in six subjects, prove that the average mark for five subjects
    (excluding physics) is 70, when the total marks are 350 more than the physics marks. -/
theorem average_mark_five_subjects (physics_mark : ℕ) : 
  let total_marks : ℕ := physics_mark + 350
  let remaining_marks : ℕ := total_marks - physics_mark
  let num_subjects : ℕ := 5
  (remaining_marks : ℚ) / num_subjects = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_mark_five_subjects_l1534_153463


namespace NUMINAMATH_CALUDE_polynomial_solutions_l1534_153411

def f (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_solutions (a b c d : ℝ) :
  (f a b c d 4 = 102) →
  (f a b c d 3 = 102) →
  (f a b c d (-3) = 102) →
  (f a b c d (-4) = 102) →
  ({x : ℝ | f a b c d x = 246} = {0, 5, -5}) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_solutions_l1534_153411


namespace NUMINAMATH_CALUDE_jacket_savings_percentage_l1534_153466

/-- Calculates the percentage saved on a purchase given the original price and total savings. -/
def percentage_saved (original_price savings : ℚ) : ℚ :=
  (savings / original_price) * 100

/-- Proves that the total percentage saved on a jacket purchase is 22.5% given the specified conditions. -/
theorem jacket_savings_percentage :
  let original_price : ℚ := 160
  let store_discount : ℚ := 20
  let coupon_savings : ℚ := 16
  let total_savings : ℚ := store_discount + coupon_savings
  percentage_saved original_price total_savings = 22.5 := by
  sorry


end NUMINAMATH_CALUDE_jacket_savings_percentage_l1534_153466


namespace NUMINAMATH_CALUDE_angle_372_in_first_quadrant_l1534_153496

/-- An angle is in the first quadrant if it is between 0° and 90° (exclusive) when reduced to the range [0°, 360°) -/
def is_in_first_quadrant (angle : ℝ) : Prop :=
  0 ≤ (angle % 360) ∧ (angle % 360) < 90

/-- Theorem: An angle of 372° is located in the first quadrant -/
theorem angle_372_in_first_quadrant :
  is_in_first_quadrant 372 := by
  sorry


end NUMINAMATH_CALUDE_angle_372_in_first_quadrant_l1534_153496


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_l1534_153425

theorem smallest_integer_with_remainder (n : ℕ) : n = 169 →
  n > 16 ∧
  n % 6 = 1 ∧
  n % 7 = 1 ∧
  n % 8 = 1 ∧
  ∀ m : ℕ, m > 16 ∧ m % 6 = 1 ∧ m % 7 = 1 ∧ m % 8 = 1 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_l1534_153425


namespace NUMINAMATH_CALUDE_squarefree_primes_property_l1534_153445

theorem squarefree_primes_property : 
  {p : ℕ | Nat.Prime p ∧ p ≥ 3 ∧ 
    ∀ q : ℕ, Nat.Prime q → q < p → 
      Squarefree (p - p / q * q)} = {5, 7, 13} := by sorry

end NUMINAMATH_CALUDE_squarefree_primes_property_l1534_153445


namespace NUMINAMATH_CALUDE_two_face_cubes_4x4x4_l1534_153402

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of unit cubes with exactly two faces on the surface of a cuboid -/
def count_two_face_cubes (c : Cuboid) : ℕ :=
  12 * (c.length - 2)

/-- Theorem: A 4x4x4 cuboid has 24 unit cubes with exactly two faces on its surface -/
theorem two_face_cubes_4x4x4 :
  let c : Cuboid := ⟨4, 4, 4⟩
  count_two_face_cubes c = 24 := by
  sorry

#eval count_two_face_cubes ⟨4, 4, 4⟩

end NUMINAMATH_CALUDE_two_face_cubes_4x4x4_l1534_153402


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2_sqrt_2_equality_condition_l1534_153417

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 → a + 2 * b ≤ x + 2 * y :=
by sorry

theorem min_value_is_2_sqrt_2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  a + 2 * b ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  a + 2 * b = 2 * Real.sqrt 2 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2_sqrt_2_equality_condition_l1534_153417


namespace NUMINAMATH_CALUDE_problem_statement_l1534_153464

theorem problem_statement : Real.rpow 81 0.25 * Real.rpow 81 0.2 = 9 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1534_153464


namespace NUMINAMATH_CALUDE_max_sum_under_constraint_l1534_153414

theorem max_sum_under_constraint (a b c : ℝ) :
  a^2 + 4*b^2 + 9*c^2 - 2*a - 12*b + 6*c + 2 = 0 →
  a + b + c ≤ 17/3 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_under_constraint_l1534_153414


namespace NUMINAMATH_CALUDE_root_sum_theorem_l1534_153476

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 8*x^2 + 10*x - 3

-- Define the roots
axiom p : ℝ
axiom q : ℝ
axiom r : ℝ

-- Axioms stating that p, q, and r are roots of f
axiom p_root : f p = 0
axiom q_root : f q = 0
axiom r_root : f r = 0

-- The theorem to prove
theorem root_sum_theorem :
  p / (q * r + 2) + q / (p * r + 2) + r / (p * q + 2) = 38 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l1534_153476


namespace NUMINAMATH_CALUDE_consecutive_even_sum_l1534_153474

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_consecutive_even (a b c : ℕ) : Prop :=
  is_even a ∧ is_even b ∧ is_even c ∧ b = a + 2 ∧ c = b + 2

def has_valid_digits (n : ℕ) : Prop :=
  n ≥ 20000 ∧ n < 30000 ∧
  n % 10 = 0 ∧
  (n / 10000 : ℕ) = 2 ∧
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧
  ((n / 10) % 10 ≠ (n / 1000) % 10) ∧
  ((n / 100) % 10 ≠ (n / 1000) % 10)

theorem consecutive_even_sum (a b c : ℕ) :
  is_consecutive_even a b c →
  has_valid_digits (a * b * c) →
  a + b + c = 84 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_l1534_153474


namespace NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l1534_153415

/-- Triangle with positive integer side lengths --/
structure Triangle :=
  (side1 : ℕ+) (side2 : ℕ+) (side3 : ℕ+)

/-- Isosceles triangle with two equal sides --/
def IsoscelesTriangle (t : Triangle) : Prop :=
  t.side1 = t.side2

/-- Point J is the intersection of angle bisectors of ∠Q and ∠R --/
def HasIntersectionJ (t : Triangle) : Prop :=
  ∃ j : ℝ × ℝ, true  -- We don't need to specify the exact conditions for J

/-- Length of QJ is 10 --/
def QJLength (t : Triangle) : Prop :=
  ∃ qj : ℝ, qj = 10

/-- Perimeter of a triangle --/
def Perimeter (t : Triangle) : ℕ :=
  t.side1.val + t.side2.val + t.side3.val

/-- The main theorem --/
theorem smallest_perimeter_isosceles_triangle :
  ∀ t : Triangle,
    IsoscelesTriangle t →
    HasIntersectionJ t →
    QJLength t →
    (∀ t' : Triangle,
      IsoscelesTriangle t' →
      HasIntersectionJ t' →
      QJLength t' →
      Perimeter t ≤ Perimeter t') →
    Perimeter t = 120 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l1534_153415


namespace NUMINAMATH_CALUDE_line_slope_proportionality_l1534_153444

/-- Given a line where an increase of 3 units in x corresponds to an increase of 7 units in y,
    prove that an increase of 9 units in x results in an increase of 21 units in y. -/
theorem line_slope_proportionality (f : ℝ → ℝ) (x : ℝ) :
  (f (x + 3) - f x = 7) → (f (x + 9) - f x = 21) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_proportionality_l1534_153444


namespace NUMINAMATH_CALUDE_max_value_inequality_l1534_153478

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 2*y)^2 / (x^2 + y^2) ≤ 9/2 := by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1534_153478


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l1534_153404

/-- Given two rectangles of equal area, where one rectangle measures 12 inches by 15 inches
    and the other has a length of 9 inches, prove that the width of the second rectangle is 20 inches. -/
theorem equal_area_rectangles_width (area carol_length carol_width jordan_length jordan_width : ℝ) :
  area = carol_length * carol_width →
  area = jordan_length * jordan_width →
  carol_length = 12 →
  carol_width = 15 →
  jordan_length = 9 →
  jordan_width = 20 := by
sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l1534_153404


namespace NUMINAMATH_CALUDE_p_iff_a_in_range_exactly_one_true_iff_a_in_range_l1534_153485

-- Define the propositions and conditions
def p (a : ℝ) : Prop := ∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - (a+2)*x + 2*a = 0

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, 
  x₁^2 - 2*m*x₁ - 3 = 0 ∧ 
  x₂^2 - 2*m*x₂ - 3 = 0 ∧ 
  x₁ ≠ x₂

def inequality_holds (a m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, q m → a^2 - 3*a ≥ |x₁ - x₂|

-- State the theorems
theorem p_iff_a_in_range (a : ℝ) : 
  p a ↔ -1 < a ∧ a < 1 :=
sorry

theorem exactly_one_true_iff_a_in_range (a : ℝ) : 
  (p a ∧ ∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → ¬(q m ∧ inequality_holds a m)) ∨
  (¬p a ∧ ∃ m : ℝ, -1 ≤ m ∧ m ≤ 1 ∧ q m ∧ inequality_holds a m)
  ↔ 
  a < 1 ∨ a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_p_iff_a_in_range_exactly_one_true_iff_a_in_range_l1534_153485


namespace NUMINAMATH_CALUDE_exchange_equality_l1534_153491

theorem exchange_equality (a₁ b₁ a₂ b₂ : ℝ) 
  (h1 : a₁^2 + b₁^2 = 1)
  (h2 : a₂^2 + b₂^2 = 1)
  (h3 : a₁*a₂ + b₁*b₂ = 0) :
  (a₁^2 + a₂^2 = 1) ∧ (b₁^2 + b₂^2 = 1) ∧ (a₁*b₁ + a₂*b₂ = 0) := by
sorry

end NUMINAMATH_CALUDE_exchange_equality_l1534_153491


namespace NUMINAMATH_CALUDE_no_real_solutions_existence_of_zero_product_ellipses_same_foci_l1534_153454

-- Statement 1
theorem no_real_solutions : ∀ x : ℝ, x^2 - 3*x + 3 ≠ 0 := by sorry

-- Statement 2
theorem existence_of_zero_product : ∃ x y : ℝ, x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0 := by sorry

-- Statement 3
def ellipse1 (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

def ellipse2 (k x y : ℝ) : Prop := x^2/(25-k) + y^2/(9-k) = 1

def has_same_foci (k : ℝ) : Prop :=
  ∃ a b : ℝ, (∀ x y : ℝ, ellipse1 x y ↔ (x-a)^2/25 + (x+a)^2/25 + y^2/9 = 1) ∧
             (∀ x y : ℝ, ellipse2 k x y ↔ (x-b)^2/(25-k) + (x+b)^2/(25-k) + y^2/(9-k) = 1) ∧
             a = b

theorem ellipses_same_foci : ∀ k : ℝ, 9 < k → k < 25 → has_same_foci k := by sorry

end NUMINAMATH_CALUDE_no_real_solutions_existence_of_zero_product_ellipses_same_foci_l1534_153454


namespace NUMINAMATH_CALUDE_z_value_l1534_153434

theorem z_value (x y z : ℚ) 
  (eq1 : 3 * x^2 + 2 * x * y * z - y^3 + 11 = z)
  (eq2 : x = 2)
  (eq3 : y = 3) : 
  z = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_z_value_l1534_153434


namespace NUMINAMATH_CALUDE_original_pencils_count_l1534_153433

/-- The number of pencils originally in the drawer -/
def original_pencils : ℕ := sorry

/-- The number of pencils Joan added to the drawer -/
def added_pencils : ℕ := 27

/-- The total number of pencils after Joan's addition -/
def total_pencils : ℕ := 60

/-- Theorem stating that the original number of pencils was 33 -/
theorem original_pencils_count : original_pencils = 33 :=
by
  sorry

#check original_pencils_count

end NUMINAMATH_CALUDE_original_pencils_count_l1534_153433


namespace NUMINAMATH_CALUDE_vasya_no_purchase_days_l1534_153446

/-- Represents Vasya's purchases over 15 school days -/
structure VasyaPurchases where
  marshmallow_days : ℕ -- Days buying 9 marshmallows
  meatpie_days : ℕ -- Days buying 2 meat pies
  combo_days : ℕ -- Days buying 4 marshmallows and 1 meat pie
  nothing_days : ℕ -- Days buying nothing

/-- Theorem stating the number of days Vasya didn't buy anything -/
theorem vasya_no_purchase_days (p : VasyaPurchases) : 
  p.marshmallow_days + p.meatpie_days + p.combo_days + p.nothing_days = 15 → 
  9 * p.marshmallow_days + 4 * p.combo_days = 30 →
  2 * p.meatpie_days + p.combo_days = 9 →
  p.nothing_days = 7 := by
  sorry

#check vasya_no_purchase_days

end NUMINAMATH_CALUDE_vasya_no_purchase_days_l1534_153446


namespace NUMINAMATH_CALUDE_congruence_problem_l1534_153492

theorem congruence_problem (N : ℕ) (h1 : N > 1) 
  (h2 : 69 ≡ 90 [MOD N]) (h3 : 90 ≡ 125 [MOD N]) : 
  81 ≡ 4 [MOD N] := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l1534_153492


namespace NUMINAMATH_CALUDE_book_pages_count_l1534_153481

/-- Represents the number of pages Bill reads on a given day -/
def pagesReadOnDay (day : ℕ) : ℕ := 10 + 2 * (day - 1)

/-- Represents the total number of pages Bill has read up to a given day -/
def totalPagesRead (days : ℕ) : ℕ := (days * (pagesReadOnDay 1 + pagesReadOnDay days)) / 2

theorem book_pages_count :
  ∀ (total_days : ℕ) (reading_days : ℕ),
  total_days = 14 →
  reading_days = total_days - 2 →
  (totalPagesRead reading_days : ℚ) = (3/4) * (336 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_book_pages_count_l1534_153481


namespace NUMINAMATH_CALUDE_sequence_inequality_l1534_153436

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h1 : a 0 = 0 ∧ a (n + 1) = 0)
  (h2 : ∀ k : ℕ, k ≥ 1 ∧ k ≤ n → |a (k - 1) - 2 * a k + a (k + 1)| ≤ 1) :
  ∀ k : ℕ, k ≤ n + 1 → |a k| ≤ k * (n + 1 - k) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1534_153436


namespace NUMINAMATH_CALUDE_interval_change_l1534_153471

/-- Represents the interval between buses on a circular route -/
def interval (num_buses : ℕ) (total_time : ℕ) : ℚ :=
  total_time / num_buses

/-- The theorem stating the relationship between intervals for 2 and 3 buses -/
theorem interval_change (total_time : ℕ) :
  total_time = 2 * 21 →
  interval 2 total_time = 21 →
  interval 3 total_time = 14 := by
  sorry

#eval interval 3 42  -- Should output 14

end NUMINAMATH_CALUDE_interval_change_l1534_153471


namespace NUMINAMATH_CALUDE_factorization_p1_factorization_p2_l1534_153423

-- Define the polynomials
def p1 (x : ℝ) : ℝ := (x^2 - 2*x - 1) * (x^2 - 2*x + 3) + 4
def p2 (x : ℝ) : ℝ := (x^2 + 6*x) * (x^2 + 6*x + 18) + 81

-- State the theorems
theorem factorization_p1 : ∀ x : ℝ, p1 x = (x - 1)^4 := by sorry

theorem factorization_p2 : ∀ x : ℝ, p2 x = (x + 3)^4 := by sorry

end NUMINAMATH_CALUDE_factorization_p1_factorization_p2_l1534_153423


namespace NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l1534_153443

/-- A conic section in the xy-plane -/
structure ConicSection where
  equation : ℝ → ℝ → Prop

/-- A hyperbola in the xy-plane -/
structure Hyperbola extends ConicSection

/-- The specific conic section given by the equation x^2 + my^2 = 1 -/
def specific_conic (m : ℝ) : ConicSection where
  equation := fun x y => x^2 + m*y^2 = 1

theorem hyperbola_iff_m_negative (m : ℝ) :
  ∃ (h : Hyperbola), h.equation = (specific_conic m).equation ↔ m < 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l1534_153443


namespace NUMINAMATH_CALUDE_circle_packing_problem_l1534_153461

theorem circle_packing_problem :
  ∃ (n : ℕ),
    n > 0 ∧
    n^2 = ((n + 14) * (n + 15)) / 2 ∧
    n^2 = 1225 :=
by sorry

end NUMINAMATH_CALUDE_circle_packing_problem_l1534_153461


namespace NUMINAMATH_CALUDE_real_condition_pure_imaginary_condition_fourth_quadrant_condition_l1534_153400

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 2*a - 3) (a^2 + a - 12)

-- (I) z is a real number iff a = -4 or a = 3
theorem real_condition (a : ℝ) : z a = Complex.mk (z a).re 0 ↔ a = -4 ∨ a = 3 := by sorry

-- (II) z is a pure imaginary number iff a = -1
theorem pure_imaginary_condition (a : ℝ) : z a = Complex.mk 0 (z a).im ∧ (z a).im ≠ 0 ↔ a = -1 := by sorry

-- (III) z is in the fourth quadrant iff -4 < a < -1
theorem fourth_quadrant_condition (a : ℝ) : (z a).re > 0 ∧ (z a).im < 0 ↔ -4 < a ∧ a < -1 := by sorry

end NUMINAMATH_CALUDE_real_condition_pure_imaginary_condition_fourth_quadrant_condition_l1534_153400


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1534_153499

theorem triangle_angle_calculation (A B C : ℝ) : 
  A = 88 → B - C = 20 → A + B + C = 180 → C = 36 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1534_153499


namespace NUMINAMATH_CALUDE_hat_price_theorem_l1534_153493

theorem hat_price_theorem (final_price : ℚ) 
  (h1 : final_price = 8)
  (h2 : ∃ original_price : ℚ, 
    final_price = original_price * (1/5) * (1 + 1/5)) : 
  ∃ original_price : ℚ, original_price = 100/3 ∧ 
    final_price = original_price * (1/5) * (1 + 1/5) := by
  sorry

end NUMINAMATH_CALUDE_hat_price_theorem_l1534_153493


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l1534_153495

/-- The function f(x) = ax - 3 + 3 always passes through the point (3, 4) for any real number a. -/
theorem fixed_point_of_exponential_translation (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x - 3 + 3
  f 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_translation_l1534_153495


namespace NUMINAMATH_CALUDE_smallest_difference_36k_5m_l1534_153498

theorem smallest_difference_36k_5m :
  (∀ k m : ℕ+, 36^k.val - 5^m.val ≥ 11) ∧
  (∃ k m : ℕ+, 36^k.val - 5^m.val = 11) :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_36k_5m_l1534_153498


namespace NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1534_153432

-- Define the ellipse E
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line that B and F lie on
def line_BF (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

theorem ellipse_and_line_theorem :
  ∀ a b : ℝ,
  a > b ∧ b > 0 →
  (∃ xB yB xF yF : ℝ,
    ellipse a b xB yB ∧
    ellipse a b xF yF ∧
    line_BF xB yB ∧
    line_BF xF yF ∧
    yB > yF) →
  (∀ x y : ℝ, ellipse a b x y ↔ x^2 / 8 + y^2 / 4 = 1) ∧
  (∀ x₁ y₁ x₂ y₂ : ℝ,
    ellipse a b x₁ y₁ ∧
    ellipse a b x₂ y₂ ∧
    is_midpoint x₁ y₁ x₂ y₂ (-1) 1 →
    x₁ - 2*y₁ + 3 = 0 ∧
    x₂ - 2*y₂ + 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_theorem_l1534_153432


namespace NUMINAMATH_CALUDE_packet_weight_difference_l1534_153429

theorem packet_weight_difference (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  (b + c + d + e) / 4 = 79 →
  a = 75 →
  e - d = 3 :=
by sorry

end NUMINAMATH_CALUDE_packet_weight_difference_l1534_153429


namespace NUMINAMATH_CALUDE_six_couples_handshakes_l1534_153413

/-- The number of handshakes in a gathering of couples where each person
    shakes hands with everyone except their spouse -/
def handshakes (n : ℕ) : ℕ :=
  let total_people := 2 * n
  let total_potential_handshakes := total_people * (total_people - 1) / 2
  total_potential_handshakes - n

theorem six_couples_handshakes :
  handshakes 6 = 60 := by sorry

end NUMINAMATH_CALUDE_six_couples_handshakes_l1534_153413


namespace NUMINAMATH_CALUDE_ellipse_properties_l1534_153426

/-- Given an ellipse with equation x²/100 + y²/36 = 1, prove that its major axis length is 20 and eccentricity is 4/5 -/
theorem ellipse_properties (x y : ℝ) :
  x^2 / 100 + y^2 / 36 = 1 →
  ∃ (a b c : ℝ),
    a = 10 ∧
    b = 6 ∧
    c^2 = a^2 - b^2 ∧
    2 * a = 20 ∧
    c / a = 4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1534_153426


namespace NUMINAMATH_CALUDE_exam_score_problem_l1534_153431

/-- Given an exam with 150 questions, where correct answers score 5 marks,
    wrong answers lose 2 marks, and the total score is 370,
    prove that the number of correctly answered questions is 95. -/
theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ)
  (h_total : total_questions = 150)
  (h_correct : correct_score = 5)
  (h_wrong : wrong_score = -2)
  (h_score : total_score = 370) :
  ∃ (correct_answers : ℕ),
    correct_answers = 95 ∧
    correct_answers ≤ total_questions ∧
    (correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score) :=
by sorry


end NUMINAMATH_CALUDE_exam_score_problem_l1534_153431


namespace NUMINAMATH_CALUDE_y_satisfies_differential_equation_l1534_153408

-- Define the function y
noncomputable def y (x : ℝ) : ℝ :=
  Real.sqrt ((Real.log ((1 + Real.exp x) / 2))^2 + 1)

-- State the theorem
theorem y_satisfies_differential_equation (x : ℝ) :
  (1 + Real.exp x) * y x * (deriv y x) = Real.exp x := by
  sorry

end NUMINAMATH_CALUDE_y_satisfies_differential_equation_l1534_153408


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1534_153465

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1534_153465


namespace NUMINAMATH_CALUDE_hallway_width_proof_l1534_153422

/-- Proves that the width of a hallway is 4 feet given the specified conditions -/
theorem hallway_width_proof (total_area : Real) (central_length : Real) (central_width : Real) (hallway_length : Real) :
  total_area = 124 ∧ 
  central_length = 10 ∧ 
  central_width = 10 ∧ 
  hallway_length = 6 → 
  (total_area - central_length * central_width) / hallway_length = 4 := by
sorry

end NUMINAMATH_CALUDE_hallway_width_proof_l1534_153422


namespace NUMINAMATH_CALUDE_first_player_wins_l1534_153480

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a player in the game --/
inductive Player | First | Second

/-- Represents a move in the game --/
structure Move :=
  (top_left : ℕ × ℕ)
  (size : ℕ)

/-- The game state --/
structure GameState :=
  (grid : Grid)
  (current_player : Player)
  (moves : List Move)

/-- Checks if a move is valid --/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  sorry

/-- Applies a move to the game state --/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over --/
def is_game_over (state : GameState) : Prop :=
  sorry

/-- Determines the winner of the game --/
def winner (state : GameState) : Option Player :=
  sorry

/-- Represents a strategy for playing the game --/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for a player --/
def is_winning_strategy (strategy : Strategy) (player : Player) : Prop :=
  sorry

/-- The main theorem: there exists a winning strategy for the first player --/
theorem first_player_wins :
  ∃ (strategy : Strategy), is_winning_strategy strategy Player.First :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l1534_153480


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l1534_153405

theorem root_difference_implies_k_value (k : ℝ) :
  (∀ x y : ℝ, x^2 + k*x + 10 = 0 ∧ y^2 - k*y + 10 = 0 ∧ y = x + 3) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l1534_153405


namespace NUMINAMATH_CALUDE_jakes_weight_ratio_l1534_153456

/-- Proves that the ratio of Jake's weight after losing 8 pounds to his sister's weight is 2:1 -/
theorem jakes_weight_ratio :
  let jake_current_weight : ℕ := 188
  let total_weight : ℕ := 278
  let weight_loss : ℕ := 8
  let jake_new_weight : ℕ := jake_current_weight - weight_loss
  let sister_weight : ℕ := total_weight - jake_current_weight
  (jake_new_weight : ℚ) / (sister_weight : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_jakes_weight_ratio_l1534_153456


namespace NUMINAMATH_CALUDE_carls_gift_bags_l1534_153416

/-- Represents the gift bag distribution problem at Carl's open house. -/
theorem carls_gift_bags (total_visitors : ℕ) (extravagant_bags : ℕ) (additional_bags : ℕ) :
  total_visitors = 90 →
  extravagant_bags = 10 →
  additional_bags = 60 →
  total_visitors - (extravagant_bags + additional_bags) = 30 := by
  sorry

#check carls_gift_bags

end NUMINAMATH_CALUDE_carls_gift_bags_l1534_153416


namespace NUMINAMATH_CALUDE_f_geq_kx_implies_k_range_l1534_153475

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x^2 - 3 * x else Real.exp x + Real.exp 2

-- State the theorem
theorem f_geq_kx_implies_k_range :
  (∀ x : ℝ, f x ≥ k * x) → -3 ≤ k ∧ k ≤ Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_f_geq_kx_implies_k_range_l1534_153475


namespace NUMINAMATH_CALUDE_circle_to_bar_graph_correspondence_l1534_153420

/-- Represents the proportions of a circle graph -/
structure CircleGraph where
  white : ℝ
  black : ℝ
  gray : ℝ
  sum_to_one : white + black + gray = 1
  white_twice_others : white = 2 * black ∧ white = 2 * gray
  black_gray_equal : black = gray

/-- Represents the heights of bars in a bar graph -/
structure BarGraph where
  white : ℝ
  black : ℝ
  gray : ℝ

/-- Theorem stating that a bar graph correctly represents a circle graph -/
theorem circle_to_bar_graph_correspondence (cg : CircleGraph) (bg : BarGraph) :
  (bg.white = 2 * bg.black ∧ bg.white = 2 * bg.gray) ∧ bg.black = bg.gray :=
by sorry

end NUMINAMATH_CALUDE_circle_to_bar_graph_correspondence_l1534_153420


namespace NUMINAMATH_CALUDE_sum_plus_even_count_equals_1811_l1534_153442

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_plus_even_count_equals_1811 :
  sum_of_integers 10 60 + count_even_integers 10 60 = 1811 := by
  sorry

end NUMINAMATH_CALUDE_sum_plus_even_count_equals_1811_l1534_153442


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_106_l1534_153440

theorem last_three_digits_of_7_to_106 : ∃ n : ℕ, 7^106 ≡ 321 [ZMOD 1000] :=
by sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_106_l1534_153440


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1534_153484

theorem perpendicular_lines_b_value (b : ℝ) : 
  let v1 : Fin 2 → ℝ := ![4, -1]
  let v2 : Fin 2 → ℝ := ![b, 8]
  (∀ i, v1 i * v2 i = 0) → b = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1534_153484


namespace NUMINAMATH_CALUDE_mikes_video_games_l1534_153488

theorem mikes_video_games :
  ∀ (total_games working_games nonworking_games : ℕ) 
    (price_per_game total_earnings : ℕ),
  nonworking_games = 8 →
  price_per_game = 7 →
  total_earnings = 56 →
  working_games * price_per_game = total_earnings →
  total_games = working_games + nonworking_games →
  total_games = 16 := by
sorry

end NUMINAMATH_CALUDE_mikes_video_games_l1534_153488


namespace NUMINAMATH_CALUDE_equation_transformation_l1534_153490

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 - x^3 - 2*x^2 - x + 1 = 0 ↔ x^2 * (y^2 - y - 4) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l1534_153490


namespace NUMINAMATH_CALUDE_integer_points_in_triangle_DEF_l1534_153452

/-- Represents a right triangle with integer side lengths -/
structure RightTriangle where
  leg1 : ℕ
  leg2 : ℕ

/-- Counts the number of integer coordinate points in and on a right triangle -/
def count_integer_points (t : RightTriangle) : ℕ :=
  sorry

/-- The specific right triangle with legs 15 and 20 -/
def triangle_DEF : RightTriangle :=
  { leg1 := 15, leg2 := 20 }

/-- Theorem stating that the number of integer coordinate points in triangle_DEF is 181 -/
theorem integer_points_in_triangle_DEF : 
  count_integer_points triangle_DEF = 181 := by
  sorry

end NUMINAMATH_CALUDE_integer_points_in_triangle_DEF_l1534_153452


namespace NUMINAMATH_CALUDE_hike_vans_count_l1534_153421

/-- Calculates the number of vans required for a hike --/
def calculate_vans (total_people : ℕ) (cars : ℕ) (taxis : ℕ) 
  (people_per_car : ℕ) (people_per_taxi : ℕ) (people_per_van : ℕ) : ℕ :=
  let people_in_cars_and_taxis := cars * people_per_car + taxis * people_per_taxi
  let people_in_vans := total_people - people_in_cars_and_taxis
  (people_in_vans + people_per_van - 1) / people_per_van

theorem hike_vans_count : 
  calculate_vans 58 3 6 4 6 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hike_vans_count_l1534_153421


namespace NUMINAMATH_CALUDE_f_intersects_positive_y_axis_l1534_153486

-- Define the function f(x) = x + 1
def f (x : ℝ) : ℝ := x + 1

-- Theorem stating that f intersects the y-axis at a point with positive y-coordinate
theorem f_intersects_positive_y_axis : ∃ (y : ℝ), y > 0 ∧ f 0 = y := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_positive_y_axis_l1534_153486


namespace NUMINAMATH_CALUDE_abs_gt_one_necessary_not_sufficient_product_nonzero_iff_both_nonzero_l1534_153477

-- Theorem for Option A
theorem abs_gt_one_necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) :=
sorry

-- Theorem for Option C
theorem product_nonzero_iff_both_nonzero (a b : ℝ) :
  a * b ≠ 0 ↔ a ≠ 0 ∧ b ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_abs_gt_one_necessary_not_sufficient_product_nonzero_iff_both_nonzero_l1534_153477


namespace NUMINAMATH_CALUDE_prob_ice_given_ski_l1534_153407

/-- The probability that a high school student likes ice skating -/
def P_ice_skating : ℝ := 0.6

/-- The probability that a high school student likes skiing -/
def P_skiing : ℝ := 0.5

/-- The probability that a high school student likes either ice skating or skiing -/
def P_ice_or_ski : ℝ := 0.7

/-- The probability that a high school student likes both ice skating and skiing -/
def P_ice_and_ski : ℝ := P_ice_skating + P_skiing - P_ice_or_ski

theorem prob_ice_given_ski :
  P_ice_and_ski / P_skiing = 0.8 := by sorry

end NUMINAMATH_CALUDE_prob_ice_given_ski_l1534_153407


namespace NUMINAMATH_CALUDE_inequality_proof_l1534_153472

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ (3 / 2) * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1534_153472


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l1534_153455

theorem complex_magnitude_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l1534_153455


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1534_153482

theorem roots_of_quadratic_equation :
  let equation := fun (x : ℂ) => x^2 + 4
  ∃ (r₁ r₂ : ℂ), r₁ = -2*I ∧ r₂ = 2*I ∧ equation r₁ = 0 ∧ equation r₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l1534_153482


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l1534_153450

-- Define the triangle ABC and point D
variable (A B C D : EuclideanPlane)

-- Define the conditions
def on_line_segment (D A C : EuclideanPlane) : Prop := sorry

-- Angle measures in degrees
def angle_measure (p q r : EuclideanPlane) : ℝ := sorry

-- Sum of angles around a point
def angle_sum_around_point (p : EuclideanPlane) : ℝ := sorry

-- Theorem statement
theorem angle_ABC_measure
  (h1 : on_line_segment D A C)
  (h2 : angle_measure A B D = 70)
  (h3 : angle_sum_around_point B = 200)
  (h4 : angle_measure C B D = 60) :
  angle_measure A B C = 70 := by sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l1534_153450


namespace NUMINAMATH_CALUDE_monkeys_on_different_ladders_l1534_153497

/-- Represents a ladder in the system -/
structure Ladder where
  id : Nat

/-- Represents a monkey in the system -/
structure Monkey where
  id : Nat
  currentLadder : Ladder

/-- Represents a rope connecting two ladders -/
structure Rope where
  ladder1 : Ladder
  ladder2 : Ladder
  height1 : Nat
  height2 : Nat

/-- Represents the state of the system -/
structure MonkeyLadderSystem where
  n : Nat
  ladders : List Ladder
  monkeys : List Monkey
  ropes : List Rope

/-- Predicate to check if all monkeys are on different ladders -/
def allMonkeysOnDifferentLadders (system : MonkeyLadderSystem) : Prop :=
  ∀ m1 m2 : Monkey, m1 ∈ system.monkeys → m2 ∈ system.monkeys → m1 ≠ m2 →
    m1.currentLadder ≠ m2.currentLadder

/-- The main theorem stating that all monkeys end up on different ladders -/
theorem monkeys_on_different_ladders (system : MonkeyLadderSystem) 
    (h1 : system.n > 0)
    (h2 : system.ladders.length = system.n)
    (h3 : system.monkeys.length = system.n)
    (h4 : ∀ m : Monkey, m ∈ system.monkeys → m.currentLadder ∈ system.ladders)
    (h5 : ∀ r : Rope, r ∈ system.ropes → r.ladder1 ∈ system.ladders ∧ r.ladder2 ∈ system.ladders)
    (h6 : ∀ r : Rope, r ∈ system.ropes → r.ladder1 ≠ r.ladder2)
    (h7 : ∀ r1 r2 : Rope, r1 ∈ system.ropes → r2 ∈ system.ropes → r1 ≠ r2 →
      (r1.ladder1 = r2.ladder1 → r1.height1 ≠ r2.height1) ∧
      (r1.ladder2 = r2.ladder2 → r1.height2 ≠ r2.height2))
    : allMonkeysOnDifferentLadders system :=
  sorry

end NUMINAMATH_CALUDE_monkeys_on_different_ladders_l1534_153497


namespace NUMINAMATH_CALUDE_park_area_l1534_153448

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure RectangularPark where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- The cost of fencing in pence per meter -/
def fencing_cost_per_meter : ℝ := 40

/-- The total cost of fencing in dollars -/
def total_fencing_cost : ℝ := 100

theorem park_area (park : RectangularPark) : 
  (2 * (park.length + park.width) * fencing_cost_per_meter / 100 = total_fencing_cost) →
  (park.length * park.width = 3750) := by
  sorry

end NUMINAMATH_CALUDE_park_area_l1534_153448


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1534_153468

theorem arithmetic_calculations :
  ((-7 + 13 - 6 + 20 = 20) ∧
   (-2^3 + (2 - 3) - 2 * (-1)^2023 = -7)) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1534_153468


namespace NUMINAMATH_CALUDE_square_binomial_coefficient_l1534_153430

theorem square_binomial_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 8 * x + 16 = (r * x + s)^2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_coefficient_l1534_153430


namespace NUMINAMATH_CALUDE_H_constant_l1534_153451

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem H_constant : ∀ x : ℝ, H x = 5 := by sorry

end NUMINAMATH_CALUDE_H_constant_l1534_153451


namespace NUMINAMATH_CALUDE_rhombus_area_l1534_153406

/-- The area of a rhombus with side length 2 and an angle of 45 degrees between adjacent sides is 2√2 -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 2) (h2 : θ = π / 4) :
  s * s * Real.sin θ = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l1534_153406


namespace NUMINAMATH_CALUDE_composite_numbers_l1534_153479

theorem composite_numbers (n : ℕ) (h : n > 0) : 
  (∃ k : ℕ, k > 1 ∧ k < 2 * 2^(2^(2*n)) + 1 ∧ (2 * 2^(2^(2*n)) + 1) % k = 0) ∧ 
  (∃ m : ℕ, m > 1 ∧ m < 3 * 2^(2*n) + 1 ∧ (3 * 2^(2*n) + 1) % m = 0) := by
sorry

end NUMINAMATH_CALUDE_composite_numbers_l1534_153479


namespace NUMINAMATH_CALUDE_max_value_sum_ratios_l1534_153427

theorem max_value_sum_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ 2*a) :
  b/a + c/b + a/c ≤ 7/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_ratios_l1534_153427


namespace NUMINAMATH_CALUDE_vertex_distance_is_five_l1534_153428

/-- The equation of the curve -/
def curve_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y - 1| = 5

/-- The y-coordinate of the upper vertex -/
def upper_vertex_y : ℝ := 3

/-- The y-coordinate of the lower vertex -/
def lower_vertex_y : ℝ := -2

/-- The distance between the vertices -/
def vertex_distance : ℝ := |upper_vertex_y - lower_vertex_y|

theorem vertex_distance_is_five :
  vertex_distance = 5 :=
by sorry

end NUMINAMATH_CALUDE_vertex_distance_is_five_l1534_153428


namespace NUMINAMATH_CALUDE_equation_solutions_l1534_153424

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define our equation
def equation (x : ℝ) : Prop := (floor x : ℝ) * (x^2 + 1) = x^3

-- Theorem statement
theorem equation_solutions :
  (∀ k : ℕ, ∃! x : ℝ, k ≤ x ∧ x < k + 1 ∧ equation x) ∧
  (∀ x : ℝ, x > 0 → equation x → ¬ (∃ q : ℚ, (q : ℝ) = x)) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1534_153424


namespace NUMINAMATH_CALUDE_machining_defective_rate_l1534_153458

theorem machining_defective_rate 
  (p1 p2 p3 : ℚ) 
  (h1 : p1 = 1 / 70)
  (h2 : p2 = 1 / 69)
  (h3 : p3 = 1 / 68)
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - p1) * (1 - p2) * (1 - p3) = 3 / 70 :=
sorry

end NUMINAMATH_CALUDE_machining_defective_rate_l1534_153458


namespace NUMINAMATH_CALUDE_winnie_balloons_distribution_l1534_153435

/-- Represents the number of balloons Winnie has left after distribution -/
def balloonsLeft (red white green chartreuse friends : ℕ) : ℕ :=
  (red + white + green + chartreuse) % friends

/-- Proves that Winnie has no balloons left after distribution -/
theorem winnie_balloons_distribution 
  (red : ℕ) (white : ℕ) (green : ℕ) (chartreuse : ℕ) (friends : ℕ)
  (h_red : red = 24)
  (h_white : white = 36)
  (h_green : green = 70)
  (h_chartreuse : chartreuse = 90)
  (h_friends : friends = 10) :
  balloonsLeft red white green chartreuse friends = 0 := by
  sorry

#eval balloonsLeft 24 36 70 90 10

end NUMINAMATH_CALUDE_winnie_balloons_distribution_l1534_153435


namespace NUMINAMATH_CALUDE_max_true_statements_l1534_153489

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^2 ∧ x^2 < 1),
    (x^2 > 1),
    (-1 < x ∧ x < 0),
    (0 < x ∧ x < 1),
    (0 < x - Real.sqrt x ∧ x - Real.sqrt x < 1)
  ]
  ¬∃ (s : Finset (Fin 5)), s.card > 3 ∧ (∀ i ∈ s, statements[i.val]) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l1534_153489


namespace NUMINAMATH_CALUDE_smallest_coprime_to_210_l1534_153470

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_coprime_to_210 :
  ∀ x : ℕ, x > 1 → x < 11 → ¬(is_relatively_prime x 210) ∧ is_relatively_prime 11 210 :=
by sorry

end NUMINAMATH_CALUDE_smallest_coprime_to_210_l1534_153470


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1534_153469

noncomputable def f (x : ℝ) : ℝ := Real.exp (2*x + 1) - 3*x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 * Real.exp 1 - 3 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1534_153469


namespace NUMINAMATH_CALUDE_divisibility_by_forty_l1534_153409

theorem divisibility_by_forty (p : ℕ) (h_prime : Prime p) (h_ge_seven : p ≥ 7) :
  (∃ q : ℕ, Prime q ∧ q ≥ 7 ∧ 40 ∣ (q^2 - 1)) ∧
  (∃ r : ℕ, Prime r ∧ r ≥ 7 ∧ ¬(40 ∣ (r^2 - 1))) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_forty_l1534_153409


namespace NUMINAMATH_CALUDE_second_concert_attendance_l1534_153453

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (additional_attendees : ℕ) 
  (h1 : first_concert = 65899) 
  (h2 : additional_attendees = 119) : 
  first_concert + additional_attendees = 66018 := by
  sorry

end NUMINAMATH_CALUDE_second_concert_attendance_l1534_153453


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1534_153437

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → 
  Nat.gcd a b = 3 → 
  Nat.lcm a b = 36 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1534_153437


namespace NUMINAMATH_CALUDE_airplane_faster_than_driving_l1534_153410

/-- Proves that taking an airplane is 90 minutes faster than driving for a job interview --/
theorem airplane_faster_than_driving :
  let driving_time_minutes : ℕ := 3 * 60 + 15
  let drive_to_airport : ℕ := 10
  let wait_to_board : ℕ := 20
  let flight_time : ℕ := driving_time_minutes / 3
  let get_off_plane : ℕ := 10
  let total_airplane_time : ℕ := drive_to_airport + wait_to_board + flight_time + get_off_plane
  driving_time_minutes - total_airplane_time = 90 := by
  sorry

end NUMINAMATH_CALUDE_airplane_faster_than_driving_l1534_153410


namespace NUMINAMATH_CALUDE_tennis_ball_ratio_l1534_153419

/-- Given the number of tennis balls for Lily, Brian, and Frodo, prove the ratio of Brian's to Frodo's tennis balls -/
theorem tennis_ball_ratio :
  ∀ (lily_balls brian_balls frodo_balls : ℕ),
    lily_balls = 3 →
    brian_balls = 22 →
    frodo_balls = lily_balls + 8 →
    (brian_balls : ℚ) / (frodo_balls : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_tennis_ball_ratio_l1534_153419


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l1534_153473

theorem square_difference_fourth_power : (7^2 - 6^2)^4 = 28561 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l1534_153473


namespace NUMINAMATH_CALUDE_consecutive_decreasing_difference_l1534_153418

/-- Represents a three-digit number with consecutive decreasing digits -/
structure ConsecutiveDecreasingNumber where
  x : ℕ
  h1 : x ≥ 1
  h2 : x ≤ 7

/-- Calculates the value of a three-digit number given its digits -/
def number_value (n : ConsecutiveDecreasingNumber) : ℕ :=
  100 * (n.x + 2) + 10 * (n.x + 1) + n.x

/-- Calculates the value of the reversed three-digit number given its digits -/
def reversed_value (n : ConsecutiveDecreasingNumber) : ℕ :=
  100 * n.x + 10 * (n.x + 1) + (n.x + 2)

/-- Theorem stating that the difference between a three-digit number with consecutive 
    decreasing digits and its reverse is always 198 -/
theorem consecutive_decreasing_difference 
  (n : ConsecutiveDecreasingNumber) : 
  number_value n - reversed_value n = 198 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_decreasing_difference_l1534_153418


namespace NUMINAMATH_CALUDE_slope_range_for_inclination_angle_l1534_153441

theorem slope_range_for_inclination_angle (θ : Real) (k : Real) :
  (π / 3 ≤ θ ∧ θ ≤ 3 * π / 4) →
  k = Real.tan θ →
  k ∈ Set.Iic (-1) ∪ Set.Ici (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_slope_range_for_inclination_angle_l1534_153441


namespace NUMINAMATH_CALUDE_ellipse_equation_l1534_153467

/-- Given an ellipse with semi-major axis a, semi-minor axis b, eccentricity √3/3,
    and a triangle formed by two points on the ellipse and one focus with perimeter 4√3,
    prove that the standard equation of the ellipse is x²/3 + y²/2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∃ c : ℝ, c / a = Real.sqrt 3 / 3 ∧ 
   a^2 = b^2 + c^2 ∧
   4 * a = 4 * Real.sqrt 3) →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 3 + y^2 / 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1534_153467


namespace NUMINAMATH_CALUDE_max_true_statements_l1534_153494

theorem max_true_statements (y : ℝ) : 
  let statement1 := 0 < y^3 ∧ y^3 < 1
  let statement2 := y^3 > 1
  let statement3 := -1 < y ∧ y < 0
  let statement4 := 0 < y ∧ y < 1
  let statement5 := 0 < y^2 - y^3 ∧ y^2 - y^3 < 1
  (∃ y : ℝ, (statement1 ∧ statement4 ∧ statement5)) ∧
  (∀ y : ℝ, ¬(statement1 ∧ statement2 ∧ statement3 ∧ statement4) ∧
            ¬(statement1 ∧ statement2 ∧ statement3 ∧ statement5) ∧
            ¬(statement1 ∧ statement2 ∧ statement4 ∧ statement5) ∧
            ¬(statement1 ∧ statement3 ∧ statement4 ∧ statement5) ∧
            ¬(statement2 ∧ statement3 ∧ statement4 ∧ statement5)) :=
by
  sorry

end NUMINAMATH_CALUDE_max_true_statements_l1534_153494


namespace NUMINAMATH_CALUDE_line_circle_intersection_l1534_153487

/-- The line kx - 2y + 1 = 0 always intersects the circle x^2 + (y-1)^2 = 1 for any real k -/
theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), k * x - 2 * y + 1 = 0 ∧ x^2 + (y - 1)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l1534_153487


namespace NUMINAMATH_CALUDE_marble_probability_l1534_153483

theorem marble_probability (red green white blue : ℕ) 
  (h_red : red = 5)
  (h_green : green = 4)
  (h_white : white = 12)
  (h_blue : blue = 2) :
  let total := red + green + white + blue
  (red / total) * (blue / (total - 1)) = 5 / 253 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l1534_153483


namespace NUMINAMATH_CALUDE_alpha_value_l1534_153460

theorem alpha_value (f : ℝ → ℝ) (α : ℝ) 
  (h1 : ∀ x, f x = 4 / (1 - x)) 
  (h2 : f α = 2) : 
  α = -1 := by
sorry

end NUMINAMATH_CALUDE_alpha_value_l1534_153460


namespace NUMINAMATH_CALUDE_cubic_roots_problem_l1534_153439

/-- A cubic polynomial with coefficient c and constant term d -/
def cubic (c d : ℝ) (x : ℝ) : ℝ := x^3 + c*x + d

theorem cubic_roots_problem (c d : ℝ) (u v : ℝ) :
  (∃ w, cubic c d u = 0 ∧ cubic c d v = 0 ∧ cubic c d w = 0) ∧
  (∃ w', cubic c (d + 300) (u + 5) = 0 ∧ cubic c (d + 300) (v - 4) = 0 ∧ cubic c (d + 300) w' = 0) →
  d = -616 ∨ d = 1575 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_problem_l1534_153439


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l1534_153438

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem fiftieth_term_of_sequence : arithmetic_sequence 2 4 50 = 198 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l1534_153438


namespace NUMINAMATH_CALUDE_specific_pyramid_lateral_area_l1534_153449

/-- Represents a pyramid with a parallelogram base -/
structure Pyramid :=
  (base_side1 : ℝ)
  (base_side2 : ℝ)
  (base_area : ℝ)
  (height : ℝ)

/-- Calculates the lateral surface area of a pyramid -/
def lateral_surface_area (p : Pyramid) : ℝ :=
  sorry

/-- Theorem stating the lateral surface area of the specific pyramid -/
theorem specific_pyramid_lateral_area :
  let p : Pyramid := { 
    base_side1 := 10,
    base_side2 := 18,
    base_area := 90,
    height := 6
  }
  lateral_surface_area p = 192 := by sorry

end NUMINAMATH_CALUDE_specific_pyramid_lateral_area_l1534_153449


namespace NUMINAMATH_CALUDE_prep_school_cost_per_semester_l1534_153459

/-- The cost per semester for John's son's prep school -/
def cost_per_semester (total_cost : ℕ) (years : ℕ) (semesters_per_year : ℕ) : ℕ :=
  total_cost / (years * semesters_per_year)

/-- Proof that the cost per semester is $20,000 -/
theorem prep_school_cost_per_semester :
  cost_per_semester 520000 13 2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_prep_school_cost_per_semester_l1534_153459


namespace NUMINAMATH_CALUDE_rotation_composition_is_translation_l1534_153447

-- Define a plane figure
def PlaneFigure : Type := sorry

-- Define a point in the plane
def Point : Type := sorry

-- Define a rotation transformation
def rotate (center : Point) (angle : ℝ) (figure : PlaneFigure) : PlaneFigure := sorry

-- Define a translation transformation
def translate (displacement : Point) (figure : PlaneFigure) : PlaneFigure := sorry

-- Define composition of transformations
def compose (t1 t2 : PlaneFigure → PlaneFigure) : PlaneFigure → PlaneFigure := sorry

theorem rotation_composition_is_translation 
  (F : PlaneFigure) (O O₁ : Point) (α : ℝ) :
  ∃ d : Point, compose (rotate O α) (rotate O₁ (-α)) F = translate d F :=
sorry

end NUMINAMATH_CALUDE_rotation_composition_is_translation_l1534_153447
