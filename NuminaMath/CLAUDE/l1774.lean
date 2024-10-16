import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_complex_expression_l1774_177494

theorem max_value_of_complex_expression (z : ℂ) (h : Complex.abs z = 1) :
  ∃ (M : ℝ), M = 4 ∧ ∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w + 2 * Real.sqrt 2 + Complex.I) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_complex_expression_l1774_177494


namespace NUMINAMATH_CALUDE_class_arrangement_probability_l1774_177427

/-- The number of classes in a school day -/
def num_classes : ℕ := 6

/-- The total number of possible arrangements of classes -/
def total_arrangements : ℕ := num_classes.factorial

/-- The number of arrangements where Mathematics is not the last class
    and Physical Education is not the first class -/
def valid_arrangements : ℕ :=
  (num_classes - 1).factorial + (num_classes - 2) * (num_classes - 2) * (num_classes - 2).factorial

/-- The probability that Mathematics is not the last class
    and Physical Education is not the first class -/
theorem class_arrangement_probability :
  (valid_arrangements : ℚ) / total_arrangements = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_class_arrangement_probability_l1774_177427


namespace NUMINAMATH_CALUDE_bicycle_count_l1774_177439

/-- Represents the number of acrobats, elephants, and bicycles in a parade. -/
structure ParadeCount where
  acrobats : ℕ
  elephants : ℕ
  bicycles : ℕ

/-- Checks if the given parade count satisfies the conditions of the problem. -/
def isValidParadeCount (count : ParadeCount) : Prop :=
  count.acrobats + count.elephants = 25 ∧
  2 * count.acrobats + 4 * count.elephants + 2 * count.bicycles = 68

/-- Theorem stating that there are 9 bicycles in the parade. -/
theorem bicycle_count : ∃ (count : ParadeCount), isValidParadeCount count ∧ count.bicycles = 9 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_count_l1774_177439


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_247_l1774_177461

theorem smallest_n_divisible_by_247 :
  ∀ n : ℕ, n > 0 ∧ n < 37 → ¬(247 ∣ n * (n + 1) * (n + 2)) ∧ (247 ∣ 37 * 38 * 39) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_247_l1774_177461


namespace NUMINAMATH_CALUDE_smallest_gcd_bc_l1774_177421

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 294) (h2 : Nat.gcd a c = 1155) :
  (∀ b' c' : ℕ+, Nat.gcd a b' = 294 → Nat.gcd a c' = 1155 → Nat.gcd b c ≤ Nat.gcd b' c') ∧
  Nat.gcd b c = 21 :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_bc_l1774_177421


namespace NUMINAMATH_CALUDE_binomial_variance_example_l1774_177417

/-- The variance of a binomial distribution with 100 trials and 0.02 probability of success is 1.96 -/
theorem binomial_variance_example :
  let n : ℕ := 100
  let p : ℝ := 0.02
  let q : ℝ := 1 - p
  let variance : ℝ := n * p * q
  variance = 1.96 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_example_l1774_177417


namespace NUMINAMATH_CALUDE_range_of_a_l1774_177485

-- Define the sets S and T
def S : Set ℝ := {x | |x - 1| + |x + 2| > 5}
def T (a : ℝ) : Set ℝ := {x | |x - a| ≤ 4}

-- State the theorem
theorem range_of_a (a : ℝ) : S ∪ T a = Set.univ → -2 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1774_177485


namespace NUMINAMATH_CALUDE_birth_year_proof_l1774_177496

/-- A person born in the first half of the 19th century whose age was x in the year x^2 was born in 1806 -/
theorem birth_year_proof (x : ℕ) (h1 : 1800 < x^2) (h2 : x^2 < 1850) (h3 : x^2 - x = 1806) : 
  x^2 - x = 1806 := by sorry

end NUMINAMATH_CALUDE_birth_year_proof_l1774_177496


namespace NUMINAMATH_CALUDE_digit_count_theorem_l1774_177458

def digits : Finset Nat := {0, 1, 2, 3, 4, 5, 6}

def four_digit_no_repeat (d : Finset Nat) : Nat :=
  (d.filter (· ≠ 0)).card * (d.card - 1) * (d.card - 2) * (d.card - 3)

def four_digit_even_no_repeat (d : Finset Nat) : Nat :=
  (d.filter (· % 2 = 0)).card * (d.filter (· ≠ 0)).card * (d.card - 2) * (d.card - 3)

def four_digit_div5_no_repeat (d : Finset Nat) : Nat :=
  (d.filter (· % 5 = 0)).card * (d.filter (· ≠ 0)).card * (d.card - 2) * (d.card - 3)

theorem digit_count_theorem :
  four_digit_no_repeat digits = 720 ∧
  four_digit_even_no_repeat digits = 420 ∧
  four_digit_div5_no_repeat digits = 220 := by
  sorry

end NUMINAMATH_CALUDE_digit_count_theorem_l1774_177458


namespace NUMINAMATH_CALUDE_range_of_t_l1774_177452

theorem range_of_t (x t : ℝ) : 
  (∀ x, (1 < x ∧ x ≤ 4) → |x - t| < 1) →
  (2 ≤ t ∧ t ≤ 3) := by
sorry

end NUMINAMATH_CALUDE_range_of_t_l1774_177452


namespace NUMINAMATH_CALUDE_angle_C_measure_l1774_177435

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real

-- State the theorem
theorem angle_C_measure (abc : Triangle) (h : abc.A + abc.B = 80) : abc.C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l1774_177435


namespace NUMINAMATH_CALUDE_impossible_conditions_l1774_177454

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  (xc - xa) * (yb - ya) = (xb - xa) * (yc - ya) ∧  -- collinearity check
  (xb - xa)^2 + (yb - ya)^2 = 144 ∧                -- AB = 12
  (xc - xb) * (xa - xb) + (yc - yb) * (ya - yb) = 0 -- ∠ABC = 90°

-- Define a point inside the triangle
def InsideTriangle (P : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  (xp - xa) * (yb - ya) < (xb - xa) * (yp - ya) ∧
  (xp - xb) * (yc - yb) < (xc - xb) * (yp - yb) ∧
  (xp - xc) * (ya - yc) < (xa - xc) * (yp - yc)

-- Define the point D on AC
def PointOnAC (D : ℝ × ℝ) (A C : ℝ × ℝ) : Prop :=
  let (xd, yd) := D
  let (xa, ya) := A
  let (xc, yc) := C
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ xd = xa + t * (xc - xa) ∧ yd = ya + t * (yc - ya)

-- Define P being on BD
def POnBD (P : ℝ × ℝ) (B D : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xb, yb) := B
  let (xd, yd) := D
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ xp = xb + t * (xd - xb) ∧ yp = yb + t * (yd - yb)

-- Define BD > 6√2
def BDGreaterThan6Sqrt2 (B D : ℝ × ℝ) : Prop :=
  let (xb, yb) := B
  let (xd, yd) := D
  (xd - xb)^2 + (yd - yb)^2 > 72

-- Define P above the median of BC
def PAboveMedianBC (P : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  let (xp, yp) := P
  let (xa, ya) := A
  let (xb, yb) := B
  let (xc, yc) := C
  let xm := (xb + xc) / 2
  let ym := (yb + yc) / 2
  (xp - xa) * (ym - ya) > (xm - xa) * (yp - ya)

theorem impossible_conditions (A B C : ℝ × ℝ) (h : Triangle A B C) :
  ¬∃ (P D : ℝ × ℝ), 
    InsideTriangle P A B C ∧ 
    PointOnAC D A C ∧ 
    POnBD P B D ∧ 
    BDGreaterThan6Sqrt2 B D ∧ 
    PAboveMedianBC P A B C :=
  sorry

end NUMINAMATH_CALUDE_impossible_conditions_l1774_177454


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l1774_177462

theorem quadratic_roots_sum (a b : ℝ) : 
  (a^2 + a - 2024 = 0) → (b^2 + b - 2024 = 0) → (a^2 + 2*a + b = 2023) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l1774_177462


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1774_177442

/-- The dividend polynomial -/
def P (x : ℝ) : ℝ := 5*x^4 - 12*x^3 + 3*x^2 - x - 30

/-- The divisor polynomial -/
def D (x : ℝ) : ℝ := x^2 - 1

/-- The remainder polynomial -/
def R (x : ℝ) : ℝ := -13*x - 22

theorem polynomial_division_remainder :
  ∃ (Q : ℝ → ℝ), ∀ x, P x = D x * Q x + R x :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1774_177442


namespace NUMINAMATH_CALUDE_workers_total_earning_approx_1480_l1774_177411

/-- Represents the daily wage and work days of a worker -/
structure Worker where
  dailyWage : ℝ
  workDays : ℕ

/-- Calculates the total earning of a worker -/
def totalEarning (w : Worker) : ℝ :=
  w.dailyWage * w.workDays

/-- Theorem stating that the total earning of the three workers is approximately 1480 -/
theorem workers_total_earning_approx_1480 
  (a b c : Worker)
  (h_a_days : a.workDays = 16)
  (h_b_days : b.workDays = 9)
  (h_c_days : c.workDays = 4)
  (h_c_wage : c.dailyWage = 71.15384615384615)
  (h_wage_ratio : a.dailyWage / c.dailyWage = 3 / 5 ∧ b.dailyWage / c.dailyWage = 4 / 5) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
    abs ((totalEarning a + totalEarning b + totalEarning c) - 1480) < ε :=
sorry

end NUMINAMATH_CALUDE_workers_total_earning_approx_1480_l1774_177411


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1774_177447

theorem sqrt_inequality (x : ℝ) (h1 : x > 0) (h2 : Real.sqrt (x + 4) < 3 * x) : 
  x > (1 + Real.sqrt 145) / 18 := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1774_177447


namespace NUMINAMATH_CALUDE_smallest_fraction_l1774_177403

theorem smallest_fraction :
  let f1 := 5 / 12
  let f2 := 7 / 17
  let f3 := 20 / 41
  let f4 := 125 / 252
  let f5 := 155 / 312
  f2 ≤ f1 ∧ f2 ≤ f3 ∧ f2 ≤ f4 ∧ f2 ≤ f5 :=
by
  sorry

#check smallest_fraction

end NUMINAMATH_CALUDE_smallest_fraction_l1774_177403


namespace NUMINAMATH_CALUDE_circle_ratio_after_increase_l1774_177409

theorem circle_ratio_after_increase (r : ℝ) : 
  let new_radius : ℝ := r + 1
  let new_circumference : ℝ := 2 * Real.pi * new_radius
  let new_diameter : ℝ := 2 * new_radius
  new_circumference / new_diameter = Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_increase_l1774_177409


namespace NUMINAMATH_CALUDE_solve_equation_l1774_177449

theorem solve_equation (x : ℚ) : x / 4 - x - 3 / 6 = 1 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1774_177449


namespace NUMINAMATH_CALUDE_area_of_triangle_AGE_l1774_177479

/-- Square ABCD with side length 5 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 5) ∧ B = (0, 0) ∧ C = (5, 0) ∧ D = (5, 5))

/-- Point E on side BC -/
def E : ℝ × ℝ := (2, 0)

/-- Point G on diagonal BD -/
def G : ℝ × ℝ := sorry

/-- Circumscribed circle of triangle ABE -/
def circle_ABE (sq : Square) : Set (ℝ × ℝ) := sorry

/-- Area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem area_of_triangle_AGE (sq : Square) :
  G ∈ circle_ABE sq →
  G.1 + G.2 = 5 →
  triangle_area sq.A G E = 54.5 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AGE_l1774_177479


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1774_177443

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem arithmetic_sequence_difference (a₁_A a₁_B d_A d_B : ℝ) (n : ℕ) :
  a₁_A = 20 ∧ a₁_B = 40 ∧ d_A = 12 ∧ d_B = -12 ∧ n = 51 →
  |arithmetic_sequence a₁_A d_A n - arithmetic_sequence a₁_B d_B n| = 1180 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l1774_177443


namespace NUMINAMATH_CALUDE_positive_integer_pairs_l1774_177414

theorem positive_integer_pairs : 
  ∀ (a b : ℕ+), 
    (∃ (k : ℕ+), k * a = b^4 + 1) → 
    (∃ (l : ℕ+), l * b = a^4 + 1) → 
    (Int.floor (Real.sqrt a.val) = Int.floor (Real.sqrt b.val)) → 
    ((a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1)) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_l1774_177414


namespace NUMINAMATH_CALUDE_vector_magnitude_cos_sin_l1774_177423

theorem vector_magnitude_cos_sin (x : ℝ) : 
  let a : Fin 2 → ℝ := ![Real.cos x, Real.sin x]
  ‖a‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_cos_sin_l1774_177423


namespace NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l1774_177482

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x^2 = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f :=
sorry

end NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l1774_177482


namespace NUMINAMATH_CALUDE_defective_units_percentage_l1774_177412

theorem defective_units_percentage
  (shipped_defective_ratio : Real)
  (total_shipped_defective_ratio : Real)
  (h1 : shipped_defective_ratio = 0.04)
  (h2 : total_shipped_defective_ratio = 0.0024) :
  ∃ (defective_ratio : Real),
    defective_ratio = 0.06 ∧
    shipped_defective_ratio * defective_ratio = total_shipped_defective_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l1774_177412


namespace NUMINAMATH_CALUDE_intersection_forms_right_triangle_l1774_177404

/-- Given a line and a parabola that intersect at two points, prove that these points and the origin form a right triangle -/
theorem intersection_forms_right_triangle (m : ℝ) :
  ∃ (x₁ x₂ : ℝ),
    -- The points satisfy the line equation
    (m * x₁ - x₁^2 + 1 = 0) ∧ (m * x₂ - x₂^2 + 1 = 0) ∧
    -- The points are distinct
    (x₁ ≠ x₂) →
    -- The triangle formed by (0,0), (x₁, x₁^2), and (x₂, x₂^2) is right-angled
    (x₁ * x₂ + x₁^2 * x₂^2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_intersection_forms_right_triangle_l1774_177404


namespace NUMINAMATH_CALUDE_rectangle_area_change_l1774_177445

theorem rectangle_area_change (L W x : ℝ) (h_positive : L > 0 ∧ W > 0) : 
  L * (1 + x / 100) * W * (1 - x / 100) = 1.01 * L * W → x = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l1774_177445


namespace NUMINAMATH_CALUDE_fraction_equality_l1774_177416

theorem fraction_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  (a + b) / a = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1774_177416


namespace NUMINAMATH_CALUDE_zilla_savings_theorem_l1774_177437

/-- Represents Zilla's monthly financial breakdown -/
structure ZillaFinances where
  total_earnings : ℝ
  rent_percentage : ℝ
  rent_amount : ℝ
  other_expenses_percentage : ℝ

/-- Calculates Zilla's savings based on her financial breakdown -/
def calculate_savings (z : ZillaFinances) : ℝ :=
  z.total_earnings - (z.rent_amount + z.total_earnings * z.other_expenses_percentage)

/-- Theorem stating Zilla's savings amount to $817 -/
theorem zilla_savings_theorem (z : ZillaFinances) 
  (h1 : z.rent_percentage = 0.07)
  (h2 : z.other_expenses_percentage = 0.5)
  (h3 : z.rent_amount = 133)
  (h4 : z.rent_amount = z.total_earnings * z.rent_percentage) :
  calculate_savings z = 817 := by
  sorry

#eval calculate_savings { total_earnings := 1900, rent_percentage := 0.07, rent_amount := 133, other_expenses_percentage := 0.5 }

end NUMINAMATH_CALUDE_zilla_savings_theorem_l1774_177437


namespace NUMINAMATH_CALUDE_expression_simplification_l1774_177405

theorem expression_simplification (a : ℤ) (n : ℕ) (h : n ≠ 1) :
  (a^(3*n) / (a^n - 1) + 1 / (a^n + 1)) - (a^(2*n) / (a^n + 1) + 1 / (a^n - 1)) = a^(2*n) + 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1774_177405


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_a_greater_than_two_l1774_177438

theorem sufficient_condition_implies_a_greater_than_two (a : ℝ) :
  (∀ x, -2 < x ∧ x < -1 → (a + x) * (1 + x) < 0) ∧
  (∃ x, ((a + x) * (1 + x) < 0) ∧ (x ≤ -2 ∨ x ≥ -1))
  → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_a_greater_than_two_l1774_177438


namespace NUMINAMATH_CALUDE_tenth_term_is_110_l1774_177451

/-- Define the sequence of small stars -/
def smallStars (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem: The 10th term of the sequence is 110 -/
theorem tenth_term_is_110 : smallStars 10 = 110 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_110_l1774_177451


namespace NUMINAMATH_CALUDE_meal_combinations_count_l1774_177473

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The index of the restricted item -/
def restricted_item : ℕ := 10

/-- A function that calculates the number of valid meal combinations -/
def valid_combinations (n : ℕ) (r : ℕ) : ℕ :=
  n * n - 1

/-- Theorem stating that the number of valid meal combinations is 224 -/
theorem meal_combinations_count :
  valid_combinations menu_items restricted_item = 224 := by
  sorry

#eval valid_combinations menu_items restricted_item

end NUMINAMATH_CALUDE_meal_combinations_count_l1774_177473


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1774_177455

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 3 = 7 →                     -- Condition 1
  (a 1 + a 2 + a 3 = 21) →      -- Condition 2 (S_3 = 21)
  q = -0.5 ∨ q = 1 :=           -- Conclusion
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1774_177455


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_attained_l1774_177444

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 := by sorry

theorem min_value_attained (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ (1 / a₀ + 3 / b₀) = 16 := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_attained_l1774_177444


namespace NUMINAMATH_CALUDE_quadratic_residue_product_l1774_177431

theorem quadratic_residue_product (p a b : ℤ) (hp : Prime p) (ha : ¬ p ∣ a) (hb : ¬ p ∣ b) :
  (∃ x : ℤ, x^2 ≡ a [ZMOD p]) → (∃ y : ℤ, y^2 ≡ b [ZMOD p]) →
  (∃ z : ℤ, z^2 ≡ a * b [ZMOD p]) := by
sorry

end NUMINAMATH_CALUDE_quadratic_residue_product_l1774_177431


namespace NUMINAMATH_CALUDE_system_solution_l1774_177424

theorem system_solution (a b : ℝ) 
  (h1 : 2 * a * 3 + 3 * 4 = 18) 
  (h2 : -(3) + 5 * b * 4 = 17) : 
  ∃ (x y : ℝ), 2 * a * (x + y) + 3 * (x - y) = 18 ∧ 
               (x + y) - 5 * b * (x - y) = -17 ∧ 
               x = 3.5 ∧ y = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1774_177424


namespace NUMINAMATH_CALUDE_inequality_problem_l1774_177497

theorem inequality_problem (a b x y : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h_sum : a + b = 1) : 
  (a*x + b*y) * (b*x + a*y) ≥ x*y := by
sorry

end NUMINAMATH_CALUDE_inequality_problem_l1774_177497


namespace NUMINAMATH_CALUDE_pseudoprime_propagation_l1774_177402

/-- A number n is a pseudoprime to base 2 if 2^n ≡ 2 (mod n) --/
def is_pseudoprime_base_2 (n : ℕ) : Prop :=
  2^n % n = 2 % n

theorem pseudoprime_propagation (n : ℕ) (h : is_pseudoprime_base_2 n) :
  is_pseudoprime_base_2 (2^n - 1) :=
sorry

end NUMINAMATH_CALUDE_pseudoprime_propagation_l1774_177402


namespace NUMINAMATH_CALUDE_parabola_c_value_l1774_177469

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) 
    (point_condition : p.x_coord 0 = 1)
    (vertex_condition : p.x_coord 2 = 3 ∧ (∀ y, p.x_coord y ≤ p.x_coord 2)) :
  p.c = 1 := by
sorry


end NUMINAMATH_CALUDE_parabola_c_value_l1774_177469


namespace NUMINAMATH_CALUDE_log_expression_equality_l1774_177429

theorem log_expression_equality (a b : ℝ) (ha : a = Real.log 8) (hb : b = Real.log 25) :
  5^(a/b) + 2^(b/a) = Real.sqrt 8 + 5^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l1774_177429


namespace NUMINAMATH_CALUDE_sarah_trucks_left_l1774_177486

/-- The number of trucks Sarah has left after giving some away -/
def trucks_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Sarah has 38 trucks left after starting with 51 and giving away 13 -/
theorem sarah_trucks_left :
  trucks_left 51 13 = 38 := by
  sorry

end NUMINAMATH_CALUDE_sarah_trucks_left_l1774_177486


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1774_177470

theorem polygon_interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 1260) →
  (180 * ((n + 3) - 2) = 1800) := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l1774_177470


namespace NUMINAMATH_CALUDE_parentheses_removal_l1774_177464

theorem parentheses_removal (a b : ℝ) : -(-a + b - 1) = a - b + 1 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l1774_177464


namespace NUMINAMATH_CALUDE_inequality_problem_l1774_177493

/-- Given an inequality with parameter a, prove that a = 8 and find the solution set -/
theorem inequality_problem (a : ℝ) : 
  (∀ x : ℝ, |x^2 - 4*x + a| + |x - 3| ≤ 5 → x ≤ 3) ∧ 
  (∃ x : ℝ, x = 3 ∧ |x^2 - 4*x + a| + |x - 3| = 5) →
  a = 8 ∧ ∀ x : ℝ, (|x^2 - 4*x + a| + |x - 3| ≤ 5 ↔ 2 ≤ x ∧ x ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_inequality_problem_l1774_177493


namespace NUMINAMATH_CALUDE_alcohol_percentage_first_vessel_l1774_177426

theorem alcohol_percentage_first_vessel
  (vessel1_capacity : ℝ)
  (vessel2_capacity : ℝ)
  (vessel2_alcohol_percentage : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (final_mixture_concentration : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel2_capacity = 6)
  (h3 : vessel2_alcohol_percentage = 50)
  (h4 : total_liquid = 8)
  (h5 : final_vessel_capacity = 10)
  (h6 : final_mixture_concentration = 37)
  : ∃ (vessel1_alcohol_percentage : ℝ),
    vessel1_alcohol_percentage = 35 ∧
    (vessel1_alcohol_percentage / 100 * vessel1_capacity +
     vessel2_alcohol_percentage / 100 * vessel2_capacity =
     final_mixture_concentration / 100 * final_vessel_capacity) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_first_vessel_l1774_177426


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l1774_177440

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (base_diameter : ℝ) (h : base_diameter = 12 * Real.sqrt 3) :
  let cone_height : ℝ := base_diameter / 2
  let sphere_radius : ℝ := 3 * Real.sqrt 6 - 3 * Real.sqrt 3
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (4 / 3) * Real.pi * (3 * Real.sqrt 6 - 3 * Real.sqrt 3) ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l1774_177440


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1774_177450

def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : arithmeticSequence a)
  (h2 : a 2 + a 9 + a 12 - a 14 + a 20 - a 7 = 8) :
  a 9 - (1/4) * a 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1774_177450


namespace NUMINAMATH_CALUDE_dealership_sales_prediction_l1774_177468

/-- Represents the sales prediction for a car dealership -/
structure SalesPrediction where
  sportsCarsRatio : ℕ
  sedansRatio : ℕ
  predictedSportsCars : ℕ

/-- Calculates the expected sedan sales and total vehicles needed -/
def calculateSales (pred : SalesPrediction) : ℕ × ℕ :=
  let expectedSedans := pred.predictedSportsCars * pred.sedansRatio / pred.sportsCarsRatio
  let totalVehicles := pred.predictedSportsCars + expectedSedans
  (expectedSedans, totalVehicles)

/-- Theorem stating the expected sales for the given scenario -/
theorem dealership_sales_prediction :
  let pred : SalesPrediction := {
    sportsCarsRatio := 3,
    sedansRatio := 5,
    predictedSportsCars := 36
  }
  calculateSales pred = (60, 96) := by
  sorry

end NUMINAMATH_CALUDE_dealership_sales_prediction_l1774_177468


namespace NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l1774_177459

/-- Given three positive integers in the ratio 2:3:5 with LCM 180, prove their sum is 60 -/
theorem sum_of_numbers_with_lcm_and_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a * 3 = b * 2 →
  a * 5 = c * 2 →
  Nat.lcm (Nat.lcm a b) c = 180 →
  a + b + c = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_with_lcm_and_ratio_l1774_177459


namespace NUMINAMATH_CALUDE_solve_system_l1774_177425

theorem solve_system (a b c d : ℚ)
  (eq1 : a = 2 * b + c)
  (eq2 : b = 2 * c + d)
  (eq3 : 2 * c = d + a - 1)
  (eq4 : d = a - c) :
  b = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1774_177425


namespace NUMINAMATH_CALUDE_travel_ways_l1774_177490

theorem travel_ways (ways_AB ways_BC : ℕ) (h1 : ways_AB = 3) (h2 : ways_BC = 2) : 
  ways_AB * ways_BC = 6 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_l1774_177490


namespace NUMINAMATH_CALUDE_two_bedroom_units_l1774_177448

theorem two_bedroom_units (total_units : ℕ) (one_bedroom_cost two_bedroom_cost : ℕ) (total_cost : ℕ)
  (h1 : total_units = 12)
  (h2 : one_bedroom_cost = 360)
  (h3 : two_bedroom_cost = 450)
  (h4 : total_cost = 4950)
  (h5 : ∃ (x y : ℕ), x + y = total_units ∧ x * one_bedroom_cost + y * two_bedroom_cost = total_cost) :
  ∃ (y : ℕ), y = 7 ∧ ∃ (x : ℕ), x + y = total_units ∧ x * one_bedroom_cost + y * two_bedroom_cost = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_two_bedroom_units_l1774_177448


namespace NUMINAMATH_CALUDE_ark5_ensures_metabolic_energy_needs_l1774_177436

-- Define the enzyme Ark5
structure Ark5 where
  activity : Bool

-- Define cancer cells
structure CancerCell where
  energy_balanced : Bool
  proliferating : Bool
  alive : Bool

-- Define the effect of Ark5 on cancer cells
def ark5_effect (a : Ark5) (c : CancerCell) : CancerCell :=
  { energy_balanced := a.activity
  , proliferating := true
  , alive := a.activity }

-- Theorem statement
theorem ark5_ensures_metabolic_energy_needs :
  ∀ (a : Ark5) (c : CancerCell),
    (¬a.activity → ¬c.energy_balanced) ∧
    (¬a.activity → c.proliferating) ∧
    (¬a.activity → ¬c.alive) →
    (a.activity → c.energy_balanced) :=
sorry

end NUMINAMATH_CALUDE_ark5_ensures_metabolic_energy_needs_l1774_177436


namespace NUMINAMATH_CALUDE_f_properties_l1774_177489

def f (x : ℝ) := x^2 + x - 2

theorem f_properties :
  (∀ y : ℝ, y ∈ Set.Icc (-1) 1 → ∃ x : ℝ, x ∈ Set.Ico (-1) 1 ∧ f x > f y) ∧
  (∃ x : ℝ, x ∈ Set.Ico (-1) 1 ∧ f x = -9/4 ∧ ∀ y : ℝ, y ∈ Set.Ico (-1) 1 → f y ≥ f x) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1774_177489


namespace NUMINAMATH_CALUDE_expected_zeroes_l1774_177499

/-- Represents the probability of getting heads on the unfair coin. -/
def A : ℚ := 1/5

/-- Represents the length of the generated string. -/
def B : ℕ := 4

/-- Represents the expected number of zeroes in the string. -/
def C : ℚ := B/2

/-- Proves that the expected number of zeroes in the string is half its length,
    and that for the given probabilities, the string length is 4. -/
theorem expected_zeroes :
  C = B/2 ∧ A = 3*B/((B+1)*(B+2)*2) ∧ B = (4 - A*B)/(4*A) := by
  sorry

#eval C  -- Should output 2

end NUMINAMATH_CALUDE_expected_zeroes_l1774_177499


namespace NUMINAMATH_CALUDE_orange_purchase_price_l1774_177480

/-- The price of oranges per 3 pounds -/
def price_per_3_pounds : ℝ := 3

/-- The weight of oranges purchased in pounds -/
def weight_purchased : ℝ := 18

/-- The discount rate applied for purchases over 15 pounds -/
def discount_rate : ℝ := 0.05

/-- The minimum weight for discount eligibility in pounds -/
def discount_threshold : ℝ := 15

/-- The final price paid by the customer for the oranges -/
def final_price : ℝ := 17.10

theorem orange_purchase_price :
  weight_purchased > discount_threshold →
  final_price = (weight_purchased / 3 * price_per_3_pounds) * (1 - discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_orange_purchase_price_l1774_177480


namespace NUMINAMATH_CALUDE_initial_cars_correct_l1774_177481

/-- Represents the car dealership scenario -/
structure CarDealership where
  initialCars : ℕ
  initialSilverPercent : ℚ
  newShipment : ℕ
  newNonSilverPercent : ℚ
  finalSilverPercent : ℚ

/-- The car dealership scenario with given conditions -/
def scenario : CarDealership :=
  { initialCars := 360,  -- This is what we want to prove
    initialSilverPercent := 15 / 100,
    newShipment := 80,
    newNonSilverPercent := 30 / 100,
    finalSilverPercent := 25 / 100 }

/-- Theorem stating that the initial number of cars is correct given the conditions -/
theorem initial_cars_correct (d : CarDealership) : 
  d.initialCars = scenario.initialCars →
  d.initialSilverPercent = scenario.initialSilverPercent →
  d.newShipment = scenario.newShipment →
  d.newNonSilverPercent = scenario.newNonSilverPercent →
  d.finalSilverPercent = scenario.finalSilverPercent →
  d.finalSilverPercent * (d.initialCars + d.newShipment) = 
    d.initialSilverPercent * d.initialCars + (1 - d.newNonSilverPercent) * d.newShipment :=
by sorry

#check initial_cars_correct

end NUMINAMATH_CALUDE_initial_cars_correct_l1774_177481


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1774_177407

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The axis of symmetry for a function f is a vertical line x = a such that
    f(a + x) = f(a - x) for all x in the domain of f -/
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) :
  IsEven (fun x ↦ f (x + 1)) → AxisOfSymmetry f 1 := by sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1774_177407


namespace NUMINAMATH_CALUDE_gcd_problem_l1774_177483

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 431) :
  Int.gcd (8 * b^2 + 63 * b + 143) (4 * b + 17) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1774_177483


namespace NUMINAMATH_CALUDE_files_per_folder_l1774_177413

theorem files_per_folder (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) :
  initial_files = 93 →
  deleted_files = 21 →
  num_folders = 9 →
  (initial_files - deleted_files) / num_folders = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_files_per_folder_l1774_177413


namespace NUMINAMATH_CALUDE_divides_prime_expression_l1774_177484

theorem divides_prime_expression (p : Nat) (h1 : p.Prime) (h2 : p > 3) :
  (42 * p) ∣ (3^p - 2^p - 1) := by
  sorry

end NUMINAMATH_CALUDE_divides_prime_expression_l1774_177484


namespace NUMINAMATH_CALUDE_officers_selection_count_l1774_177446

/-- The number of ways to select officers from a group -/
def select_officers (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k * Nat.factorial k

/-- Theorem: Selecting 3 officers from 4 people results in 24 ways -/
theorem officers_selection_count :
  select_officers 4 3 = 24 := by
  sorry

#eval select_officers 4 3  -- This should output 24

end NUMINAMATH_CALUDE_officers_selection_count_l1774_177446


namespace NUMINAMATH_CALUDE_f_properties_l1774_177415

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - 1/2) * x^2 + log x

theorem f_properties :
  (∀ m : ℝ, (∃ x₀ : ℝ, x₀ ∈ Set.Icc 1 (exp 1) ∧ f 1 x₀ ≤ m) ↔ m ∈ Set.Ici (1/2)) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 1 → f a x < 2 * a * x) ↔ a ∈ Set.Icc (-1/2) (1/2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1774_177415


namespace NUMINAMATH_CALUDE_xiao_zhao_grade_l1774_177401

/-- Calculates the final grade based on component scores and weights -/
def calculate_grade (class_score : ℝ) (midterm_score : ℝ) (final_score : ℝ) 
  (class_weight : ℝ) (midterm_weight : ℝ) (final_weight : ℝ) : ℝ :=
  class_score * class_weight + midterm_score * midterm_weight + final_score * final_weight

/-- Theorem stating that Xiao Zhao's physical education grade is 44.5 -/
theorem xiao_zhao_grade : 
  let max_score : ℝ := 50
  let class_weight : ℝ := 0.3
  let midterm_weight : ℝ := 0.2
  let final_weight : ℝ := 0.5
  let class_score : ℝ := 40
  let midterm_score : ℝ := 50
  let final_score : ℝ := 45
  calculate_grade class_score midterm_score final_score class_weight midterm_weight final_weight = 44.5 := by
  sorry

end NUMINAMATH_CALUDE_xiao_zhao_grade_l1774_177401


namespace NUMINAMATH_CALUDE_complex_division_simplification_l1774_177457

theorem complex_division_simplification :
  let z₁ : ℂ := 5 + 3 * I
  let z₂ : ℂ := 2 + I
  z₁ / z₂ = 13/5 + (1/5) * I :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l1774_177457


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l1774_177428

theorem solution_set_abs_inequality :
  {x : ℝ | 1 < |x + 2| ∧ |x + 2| < 5} = {x : ℝ | -7 < x ∧ x < -3} ∪ {x : ℝ | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l1774_177428


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1774_177466

theorem binomial_expansion_coefficient (m : ℝ) : 
  m > 0 → 
  (Nat.choose 5 2 * m^2 = Nat.choose 5 1 * m + 30) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1774_177466


namespace NUMINAMATH_CALUDE_inverse_of_AB_l1774_177408

def A : Matrix (Fin 2) (Fin 2) ℚ := !![1, 0; 0, 2]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![1, 1/2; 0, 1]

theorem inverse_of_AB :
  (A * B)⁻¹ = !![1, -1; 0, 1/2] := by sorry

end NUMINAMATH_CALUDE_inverse_of_AB_l1774_177408


namespace NUMINAMATH_CALUDE_x_coordinate_of_first_point_l1774_177422

/-- Given two points on a line, prove the x-coordinate of the first point -/
theorem x_coordinate_of_first_point 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  m = 2 * n + 5 := by
sorry

end NUMINAMATH_CALUDE_x_coordinate_of_first_point_l1774_177422


namespace NUMINAMATH_CALUDE_sasha_work_hours_l1774_177400

/-- Calculates the number of hours Sasha worked given her question completion rate,
    total number of questions, and remaining questions. -/
def hours_worked (completion_rate : ℕ) (total_questions : ℕ) (remaining_questions : ℕ) : ℚ :=
  (total_questions - remaining_questions) / completion_rate

/-- Proves that Sasha worked for 2 hours given the problem conditions. -/
theorem sasha_work_hours :
  let completion_rate : ℕ := 15
  let total_questions : ℕ := 60
  let remaining_questions : ℕ := 30
  hours_worked completion_rate total_questions remaining_questions = 2 := by
  sorry

end NUMINAMATH_CALUDE_sasha_work_hours_l1774_177400


namespace NUMINAMATH_CALUDE_integer_root_values_l1774_177441

def polynomial (x b : ℤ) : ℤ := x^4 + 4*x^3 + 2*x^2 + b*x + 12

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x b = 0

theorem integer_root_values :
  {b : ℤ | has_integer_root b} = {-34, -19, -10, -9, -3, 2, 4, 6, 8, 11} :=
sorry

end NUMINAMATH_CALUDE_integer_root_values_l1774_177441


namespace NUMINAMATH_CALUDE_problem_statement_l1774_177476

theorem problem_statement : 
  (∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≥ 0) ∧ ¬(∀ a b : ℝ, a < b → 1/a > 1/b) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1774_177476


namespace NUMINAMATH_CALUDE_dog_food_cost_l1774_177492

-- Define the given constants
def puppy_cost : ℚ := 10
def days : ℕ := 21
def food_per_day : ℚ := 1/3
def food_per_bag : ℚ := 7/2
def total_cost : ℚ := 14

-- Define the theorem
theorem dog_food_cost :
  let total_food := days * food_per_day
  let bags_needed := total_food / food_per_bag
  let food_cost := total_cost - puppy_cost
  food_cost / bags_needed = 2 := by
sorry

end NUMINAMATH_CALUDE_dog_food_cost_l1774_177492


namespace NUMINAMATH_CALUDE_parrot_count_l1774_177472

theorem parrot_count (total_birds : ℕ) (remaining_parrots : ℕ) (remaining_crow : ℕ) 
  (h1 : total_birds = 13)
  (h2 : remaining_parrots = 2)
  (h3 : remaining_crow = 1)
  (h4 : ∃ (x : ℕ), total_birds = remaining_parrots + remaining_crow + 2 * x) :
  ∃ (initial_parrots : ℕ), initial_parrots = 7 ∧ 
    ∃ (initial_crows : ℕ), initial_crows + initial_parrots = total_birds :=
by
  sorry

end NUMINAMATH_CALUDE_parrot_count_l1774_177472


namespace NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l1774_177478

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l1774_177478


namespace NUMINAMATH_CALUDE_edward_money_theorem_l1774_177430

/-- Calculates Edward's earnings from mowing lawns --/
def lawn_earnings (small medium large : ℕ) : ℕ :=
  8 * small + 12 * medium + 15 * large

/-- Calculates Edward's earnings from cleaning gardens --/
def garden_earnings (gardens : ℕ) : ℕ :=
  if gardens = 0 then 0
  else if gardens = 1 then 10
  else if gardens = 2 then 22
  else 22 + 15 * (gardens - 2)

/-- Calculates Edward's total earnings --/
def total_earnings (small medium large gardens : ℕ) : ℕ :=
  lawn_earnings small medium large + garden_earnings gardens

/-- Calculates Edward's final amount of money --/
def edward_final_money (small medium large gardens savings fuel_cost rental_cost : ℕ) : ℕ :=
  total_earnings small medium large gardens + savings - (fuel_cost + rental_cost)

theorem edward_money_theorem :
  edward_final_money 3 1 1 5 7 10 15 = 100 := by
  sorry

#eval edward_final_money 3 1 1 5 7 10 15

end NUMINAMATH_CALUDE_edward_money_theorem_l1774_177430


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1774_177433

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex nonagon has 27 distinct diagonals -/
theorem nonagon_diagonals : diagonals_in_nonagon = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1774_177433


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1774_177434

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f := λ x : ℝ => a^(x-1) + 4
  f 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1774_177434


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l1774_177432

theorem pure_imaginary_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = y * Complex.I) : 
  p / q = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l1774_177432


namespace NUMINAMATH_CALUDE_orange_juice_concentrate_size_l1774_177487

/-- The size of a can of orange juice concentrate in ounces -/
def concentrate_size : ℝ := 420

/-- The number of servings to be prepared -/
def num_servings : ℕ := 280

/-- The size of each serving in ounces -/
def serving_size : ℝ := 6

/-- The ratio of water cans to concentrate cans -/
def water_to_concentrate_ratio : ℝ := 3

theorem orange_juice_concentrate_size :
  concentrate_size * (1 + water_to_concentrate_ratio) * num_servings = serving_size * num_servings :=
sorry

end NUMINAMATH_CALUDE_orange_juice_concentrate_size_l1774_177487


namespace NUMINAMATH_CALUDE_cookie_comparison_l1774_177475

theorem cookie_comparison (a b c : ℕ) (ha : a = 7) (hb : b = 8) (hc : c = 5) :
  (1 : ℚ) / c > (1 : ℚ) / a ∧ (1 : ℚ) / c > (1 : ℚ) / b :=
sorry

end NUMINAMATH_CALUDE_cookie_comparison_l1774_177475


namespace NUMINAMATH_CALUDE_set_operations_and_range_l1774_177474

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | 5 - a < x ∧ x < a}

-- Theorem statement
theorem set_operations_and_range :
  (A ∪ B = {x : ℝ | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | 7 ≤ x ∧ x < 10}) ∧
  (∀ a : ℝ, C a ⊆ B → a ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l1774_177474


namespace NUMINAMATH_CALUDE_virginia_march_rainfall_l1774_177453

/-- Calculates the rainfall in March given the rainfall amounts for April, May, June, July, and the average rainfall for 5 months. -/
def march_rainfall (april may june july average : ℝ) : ℝ :=
  5 * average - (april + may + june + july)

/-- Theorem stating that the rainfall in March was 3.79 inches given the specified conditions. -/
theorem virginia_march_rainfall :
  let april : ℝ := 4.5
  let may : ℝ := 3.95
  let june : ℝ := 3.09
  let july : ℝ := 4.67
  let average : ℝ := 4
  march_rainfall april may june july average = 3.79 := by
  sorry

end NUMINAMATH_CALUDE_virginia_march_rainfall_l1774_177453


namespace NUMINAMATH_CALUDE_circle_radius_theorem_l1774_177406

theorem circle_radius_theorem (r : ℝ) (h : r > 0) : 3 * (2 * Real.pi * r) = Real.pi * r^2 → r = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_theorem_l1774_177406


namespace NUMINAMATH_CALUDE_solve_equation_l1774_177477

theorem solve_equation (r : ℚ) : 3 * (r - 7) = 4 * (2 - 2 * r) + 4 → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1774_177477


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_range_l1774_177419

theorem quadratic_inequality_empty_solution_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + a < 0) ↔ a < -3/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_range_l1774_177419


namespace NUMINAMATH_CALUDE_largest_square_area_l1774_177488

theorem largest_square_area (X Y Z : ℝ) (h_right_angle : X^2 + Y^2 = Z^2)
  (h_equal_sides : X = Y) (h_sum_areas : X^2 + Y^2 + Z^2 + (2*Y)^2 = 650) :
  Z^2 = 650/3 := by
sorry

end NUMINAMATH_CALUDE_largest_square_area_l1774_177488


namespace NUMINAMATH_CALUDE_cone_height_ratio_l1774_177495

theorem cone_height_ratio (base_circumference : Real) (original_height : Real) (shorter_volume : Real) :
  base_circumference = 20 * Real.pi →
  original_height = 40 →
  shorter_volume = 160 * Real.pi →
  ∃ (shorter_height : Real),
    (1 / 3) * Real.pi * ((base_circumference / (2 * Real.pi)) ^ 2) * shorter_height = shorter_volume ∧
    shorter_height / original_height = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l1774_177495


namespace NUMINAMATH_CALUDE_balloon_tanks_l1774_177465

theorem balloon_tanks (num_balloons : ℕ) (air_per_balloon : ℕ) (tank_capacity : ℕ) :
  num_balloons = 1000 →
  air_per_balloon = 10 →
  tank_capacity = 500 →
  (num_balloons * air_per_balloon + tank_capacity - 1) / tank_capacity = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_balloon_tanks_l1774_177465


namespace NUMINAMATH_CALUDE_cubic_yards_and_feet_conversion_l1774_177460

/-- Conversion factor from cubic yards to cubic feet -/
def cubic_yard_to_cubic_feet : ℝ := 27

/-- The problem statement -/
theorem cubic_yards_and_feet_conversion 
  (cubic_yards : ℝ) (additional_cubic_feet : ℝ) : 
  cubic_yards * cubic_yard_to_cubic_feet + additional_cubic_feet = 139 ↔ 
  cubic_yards = 5 ∧ additional_cubic_feet = 4 :=
by sorry

end NUMINAMATH_CALUDE_cubic_yards_and_feet_conversion_l1774_177460


namespace NUMINAMATH_CALUDE_decimal_point_shift_l1774_177498

theorem decimal_point_shift (x : ℝ) : 10 * x = x + 37.89 → 100 * x = 421 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_shift_l1774_177498


namespace NUMINAMATH_CALUDE_refrigerator_part_payment_l1774_177463

/-- Given a refrigerator purchase where a part payment of 25% has been made
    and $2625 remains to be paid (representing 75% of the total cost),
    prove that the part payment is equal to $875. -/
theorem refrigerator_part_payment
  (total_cost : ℝ)
  (part_payment_percentage : ℝ)
  (remaining_payment : ℝ)
  (remaining_percentage : ℝ)
  (h1 : part_payment_percentage = 0.25)
  (h2 : remaining_payment = 2625)
  (h3 : remaining_percentage = 0.75)
  (h4 : remaining_payment = remaining_percentage * total_cost) :
  part_payment_percentage * total_cost = 875 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_part_payment_l1774_177463


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1774_177456

theorem other_root_of_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + k * x = 5 ∧ x = 3) → 
  (∃ y : ℝ, 3 * y^2 + k * y = 5 ∧ y = -5/9) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1774_177456


namespace NUMINAMATH_CALUDE_subset_implies_a_leq_two_l1774_177471

theorem subset_implies_a_leq_two (a : ℝ) : 
  let A : Set ℝ := {x | x ≥ a}
  let B : Set ℝ := {x | |x - 3| < 1}
  B ⊆ A → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_leq_two_l1774_177471


namespace NUMINAMATH_CALUDE_satisfying_function_is_identity_l1774_177420

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧
  (f 1 = 1) ∧
  (∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem satisfying_function_is_identity (f : ℝ → ℝ) (hf : SatisfyingFunction f) :
  ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_CALUDE_satisfying_function_is_identity_l1774_177420


namespace NUMINAMATH_CALUDE_trivia_team_distribution_l1774_177467

theorem trivia_team_distribution (total : ℕ) (not_picked : ℕ) (groups : ℕ) : 
  total = 36 → not_picked = 9 → groups = 3 → 
  (total - not_picked) / groups = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_trivia_team_distribution_l1774_177467


namespace NUMINAMATH_CALUDE_rainfall_volume_calculation_l1774_177410

-- Define the rainfall in centimeters
def rainfall_cm : ℝ := 5

-- Define the ground area in hectares
def ground_area_hectares : ℝ := 1.5

-- Define the conversion factor from hectares to square meters
def hectares_to_sqm : ℝ := 10000

-- Define the conversion factor from centimeters to meters
def cm_to_m : ℝ := 0.01

-- Theorem statement
theorem rainfall_volume_calculation :
  let rainfall_m := rainfall_cm * cm_to_m
  let ground_area_sqm := ground_area_hectares * hectares_to_sqm
  rainfall_m * ground_area_sqm = 750 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_volume_calculation_l1774_177410


namespace NUMINAMATH_CALUDE_power_sum_equals_two_l1774_177418

theorem power_sum_equals_two : (1 : ℤ)^10 + (-1 : ℤ)^8 + (-1 : ℤ)^7 + (1 : ℤ)^5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_two_l1774_177418


namespace NUMINAMATH_CALUDE_board_numbers_theorem_l1774_177491

def pairwise_sums : List ℕ := [5, 8, 9, 13, 14, 14, 15, 17, 18, 23]

def is_valid_set (s : List ℕ) : Prop :=
  s.length = 5 ∧
  (List.map (λ (x, y) => x + y) (s.product s)).filter (λ x => x ∉ s) = pairwise_sums

theorem board_numbers_theorem :
  ∃ (s : List ℕ), is_valid_set s ∧ s.prod = 4752 := by sorry

end NUMINAMATH_CALUDE_board_numbers_theorem_l1774_177491
