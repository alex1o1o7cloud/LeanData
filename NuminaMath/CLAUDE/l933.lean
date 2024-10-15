import Mathlib

namespace NUMINAMATH_CALUDE_square_difference_l933_93356

theorem square_difference (n : ℕ) (h : (n + 1)^2 = n^2 + 2*n + 1) :
  n^2 - (n - 1)^2 = 2*n - 1 := by
  sorry

#check square_difference 50

end NUMINAMATH_CALUDE_square_difference_l933_93356


namespace NUMINAMATH_CALUDE_largest_gcd_of_sum_780_l933_93322

theorem largest_gcd_of_sum_780 :
  ∃ (x y : ℕ+), x + y = 780 ∧ 
  ∀ (a b : ℕ+), a + b = 780 → Nat.gcd x y ≥ Nat.gcd a b ∧
  Nat.gcd x y = 390 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_of_sum_780_l933_93322


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l933_93395

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 4 = 0) → 
  (x₂^2 + 2*x₂ - 4 = 0) → 
  (x₁ + x₂ = -2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l933_93395


namespace NUMINAMATH_CALUDE_cooler_capacity_sum_l933_93378

theorem cooler_capacity_sum (c1 c2 c3 : ℝ) : 
  c1 = 100 →
  c2 = c1 + c1 * 0.5 →
  c3 = c2 / 2 →
  c1 + c2 + c3 = 325 := by
sorry

end NUMINAMATH_CALUDE_cooler_capacity_sum_l933_93378


namespace NUMINAMATH_CALUDE_x_squared_coefficient_in_expansion_l933_93345

/-- The coefficient of x² in the expansion of (2+x)(1-2x)^5 is 70 -/
theorem x_squared_coefficient_in_expansion : Int := by
  sorry

end NUMINAMATH_CALUDE_x_squared_coefficient_in_expansion_l933_93345


namespace NUMINAMATH_CALUDE_product_of_large_numbers_l933_93329

theorem product_of_large_numbers : (300000 : ℕ) * 300000 * 3 = 270000000000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_large_numbers_l933_93329


namespace NUMINAMATH_CALUDE_sameTerminalSideAs315_eq_l933_93382

/-- The set of angles with the same terminal side as 315° -/
def sameTerminalSideAs315 : Set ℝ :=
  {α | ∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 4}

/-- Theorem stating that the set of angles with the same terminal side as 315° 
    is equal to {α | α = 2kπ - π/4, k ∈ ℤ} -/
theorem sameTerminalSideAs315_eq : 
  sameTerminalSideAs315 = {α | ∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 4} := by
  sorry


end NUMINAMATH_CALUDE_sameTerminalSideAs315_eq_l933_93382


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l933_93300

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l933_93300


namespace NUMINAMATH_CALUDE_roger_birthday_money_l933_93371

/-- Calculates the amount of birthday money Roger received -/
def birthday_money (initial_amount spent_amount final_amount : ℤ) : ℤ :=
  final_amount - initial_amount + spent_amount

/-- Proves that Roger received 28 dollars for his birthday -/
theorem roger_birthday_money :
  birthday_money 16 25 19 = 28 := by
  sorry

end NUMINAMATH_CALUDE_roger_birthday_money_l933_93371


namespace NUMINAMATH_CALUDE_sum_six_odd_squares_not_2020_l933_93369

theorem sum_six_odd_squares_not_2020 : ¬ ∃ (a b c d e f : ℤ),
  (2 * a + 1)^2 + (2 * b + 1)^2 + (2 * c + 1)^2 + 
  (2 * d + 1)^2 + (2 * e + 1)^2 + (2 * f + 1)^2 = 2020 :=
by sorry

end NUMINAMATH_CALUDE_sum_six_odd_squares_not_2020_l933_93369


namespace NUMINAMATH_CALUDE_polynomial_simplification_l933_93338

theorem polynomial_simplification (x : ℝ) : 
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 4*x^3 - 6*x^3 + 8*x^3 = 
  -3 + 23*x - x^2 + 6*x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l933_93338


namespace NUMINAMATH_CALUDE_intersection_points_on_ellipse_l933_93334

/-- The points of intersection of two parametric lines lie on an ellipse -/
theorem intersection_points_on_ellipse (s : ℝ) : 
  ∃ (a b : ℝ) (h : a > 0 ∧ b > 0), 
    ∀ (x y : ℝ), 
      (s * x - 3 * y - 4 * s = 0 ∧ x - 3 * s * y + 4 = 0) → 
      (x^2 / a^2 + y^2 / b^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_intersection_points_on_ellipse_l933_93334


namespace NUMINAMATH_CALUDE_sum_of_squares_is_45_l933_93385

/-- Represents the ages of Alice, Bob, and Charlie -/
structure Ages where
  alice : ℕ
  bob : ℕ
  charlie : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  (3 * ages.alice + 2 * ages.bob = 4 * ages.charlie) ∧
  (3 * ages.charlie^2 = 4 * ages.alice^2 + 2 * ages.bob^2) ∧
  (Nat.gcd ages.alice ages.bob = 1) ∧
  (Nat.gcd ages.alice ages.charlie = 1) ∧
  (Nat.gcd ages.bob ages.charlie = 1)

/-- The theorem to be proved -/
theorem sum_of_squares_is_45 (ages : Ages) :
  satisfies_conditions ages →
  ages.alice^2 + ages.bob^2 + ages.charlie^2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_is_45_l933_93385


namespace NUMINAMATH_CALUDE_correct_ordering_l933_93310

theorem correct_ordering (m n p q : ℝ) 
  (h1 : m < n) 
  (h2 : p < q) 
  (h3 : (p - m) * (p - n) < 0) 
  (h4 : (q - m) * (q - n) < 0) : 
  m < p ∧ p < q ∧ q < n := by
  sorry

end NUMINAMATH_CALUDE_correct_ordering_l933_93310


namespace NUMINAMATH_CALUDE_no_extrema_on_open_interval_l933_93380

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem no_extrema_on_open_interval : 
  ¬ (∃ (x : ℝ), x ∈ Set.Ioo (-1) 1 ∧ (∀ (y : ℝ), y ∈ Set.Ioo (-1) 1 → f y ≤ f x)) ∧
  ¬ (∃ (x : ℝ), x ∈ Set.Ioo (-1) 1 ∧ (∀ (y : ℝ), y ∈ Set.Ioo (-1) 1 → f y ≥ f x)) :=
by sorry

end NUMINAMATH_CALUDE_no_extrema_on_open_interval_l933_93380


namespace NUMINAMATH_CALUDE_trigonometric_sum_equals_three_plus_sqrt_three_l933_93312

theorem trigonometric_sum_equals_three_plus_sqrt_three :
  let sin_30 : ℝ := 1/2
  let cos_30 : ℝ := Real.sqrt 3 / 2
  let tan_30 : ℝ := sin_30 / cos_30
  3 * tan_30 + 6 * sin_30 = 3 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equals_three_plus_sqrt_three_l933_93312


namespace NUMINAMATH_CALUDE_mary_has_five_candies_l933_93394

/-- The number of candies Mary has on Halloween -/
def marys_candies (bob_candies sue_candies john_candies sam_candies total_candies : ℕ) : ℕ :=
  total_candies - (bob_candies + sue_candies + john_candies + sam_candies)

/-- Theorem: Mary has 5 candies given the Halloween candy distribution -/
theorem mary_has_five_candies :
  marys_candies 10 20 5 10 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_five_candies_l933_93394


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l933_93342

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im (i^2 / (2*i - 1)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l933_93342


namespace NUMINAMATH_CALUDE_steven_has_16_apples_l933_93335

/-- Represents the number of fruits a person has -/
structure FruitCount where
  peaches : ℕ
  apples : ℕ

/-- Given information about Steven and Jake's fruit counts -/
def steven_jake_fruits : Prop :=
  ∃ (steven jake : FruitCount),
    steven.peaches = 17 ∧
    steven.peaches = steven.apples + 1 ∧
    jake.peaches + 6 = steven.peaches ∧
    jake.apples = steven.apples + 8

/-- Theorem stating that Steven has 16 apples -/
theorem steven_has_16_apples :
  steven_jake_fruits → ∃ (steven : FruitCount), steven.apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_steven_has_16_apples_l933_93335


namespace NUMINAMATH_CALUDE_salary_adjustment_l933_93333

theorem salary_adjustment (original_salary : ℝ) (original_salary_positive : original_salary > 0) :
  let reduced_salary := original_salary * 0.9
  (reduced_salary * (1 + 100/9 * 0.01) : ℝ) = original_salary := by
  sorry

end NUMINAMATH_CALUDE_salary_adjustment_l933_93333


namespace NUMINAMATH_CALUDE_polynomial_division_l933_93393

theorem polynomial_division (x : ℝ) (h : x ≠ 0) :
  (6 * x^4 - 8 * x^3) / (-2 * x^2) = -3 * x^2 + 4 * x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l933_93393


namespace NUMINAMATH_CALUDE_largest_divisor_of_m_l933_93343

theorem largest_divisor_of_m (m : ℕ) (h1 : m > 0) (h2 : 54 ∣ m^2) : 
  ∃ (d : ℕ), d ∣ m ∧ d = 9 ∧ ∀ (k : ℕ), k ∣ m → k ≤ d :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_m_l933_93343


namespace NUMINAMATH_CALUDE_age_difference_proof_l933_93372

def age_difference (a b c : ℕ) : ℕ := (a + b) - (b + c)

theorem age_difference_proof (a b c : ℕ) (h : c = a - 11) :
  age_difference a b c = 11 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l933_93372


namespace NUMINAMATH_CALUDE_sum_difference_of_squares_l933_93368

theorem sum_difference_of_squares (n : ℤ) : ∃ a b c d : ℤ, n = a^2 + b^2 - c^2 - d^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_of_squares_l933_93368


namespace NUMINAMATH_CALUDE_sale_price_calculation_l933_93361

theorem sale_price_calculation (original_price : ℝ) :
  let increased_price := original_price * 1.3
  let sale_price := increased_price * 0.9
  sale_price = original_price * 1.17 :=
by sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l933_93361


namespace NUMINAMATH_CALUDE_sqrt_sum_quotient_simplification_l933_93328

theorem sqrt_sum_quotient_simplification :
  (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt 75 = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_quotient_simplification_l933_93328


namespace NUMINAMATH_CALUDE_ellipse_equation_l933_93367

theorem ellipse_equation (A B C : ℝ × ℝ) (h1 : A = (-2, 0)) (h2 : B = (2, 0)) (h3 : C.1^2 + C.2^2 = 5) 
  (h4 : (C.1 - A.1) * (B.2 - A.2) = (C.2 - A.2) * (B.1 - A.1)) :
  ∃ (x y : ℝ), x^2/4 + 3*y^2/4 = 1 ∧ x^2 + y^2 = C.1^2 + C.2^2 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ellipse_equation_l933_93367


namespace NUMINAMATH_CALUDE_value_of_x_minus_y_l933_93303

theorem value_of_x_minus_y (x y z : ℝ) 
  (eq1 : 3 * x - 5 * y = 5)
  (eq2 : x / (x + y) = 5 / 7)
  (eq3 : x + z * y = 10) :
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_minus_y_l933_93303


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l933_93307

/-- An isosceles triangle with perimeter 20 and leg length 7 has a base length of 6 -/
theorem isosceles_triangle_base_length : ∀ (base leg : ℝ),
  leg = 7 → base + 2 * leg = 20 → base = 6 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l933_93307


namespace NUMINAMATH_CALUDE_smallest_triangle_side_l933_93379

-- Define the triangle sides
def a : ℕ := 7
def b : ℕ := 11

-- Define the triangle inequality
def is_triangle (x y z : ℕ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Define the property we want to prove
def smallest_side (s : ℕ) : Prop :=
  is_triangle a b s ∧ ∀ t : ℕ, t < s → ¬(is_triangle a b t)

-- The theorem to prove
theorem smallest_triangle_side : smallest_side 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_side_l933_93379


namespace NUMINAMATH_CALUDE_vector_sum_simplification_l933_93317

variable {V : Type*} [AddCommGroup V]
variable (A B C D : V)

theorem vector_sum_simplification :
  (B - A) + (A - C) + (D - B) = D - C :=
by sorry

end NUMINAMATH_CALUDE_vector_sum_simplification_l933_93317


namespace NUMINAMATH_CALUDE_f_properties_l933_93363

noncomputable section

/-- The function f(x) = (ax+b)e^x -/
def f (a b x : ℝ) : ℝ := (a * x + b) * Real.exp x

/-- The condition that f has an extremum at x = -1 -/
def has_extremum_at_neg_one (a b : ℝ) : Prop :=
  ∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b (-1) ≥ f a b x

/-- The condition that f(x) ≥ x^2 + 2x - 1 for x ≥ -1 -/
def satisfies_inequality (a b : ℝ) : Prop :=
  ∀ x ≥ -1, f a b x ≥ x^2 + 2*x - 1

/-- The main theorem -/
theorem f_properties (a b : ℝ) 
  (h1 : has_extremum_at_neg_one a b)
  (h2 : satisfies_inequality a b) :
  b = 0 ∧ 2 / Real.exp 1 ≤ a ∧ a ≤ 2 * Real.exp 1 :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l933_93363


namespace NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coordinate_l933_93355

/-- Given an equilateral triangle with two vertices at (1, 10) and (9, 10),
    and the third vertex in the first quadrant, 
    prove that the y-coordinate of the third vertex is 10 + 4√3 -/
theorem equilateral_triangle_third_vertex_y_coordinate 
  (A B C : ℝ × ℝ) : 
  A = (1, 10) → 
  B = (9, 10) → 
  C.1 ≥ 0 → 
  C.2 ≥ 0 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 → 
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 → 
  C.2 = 10 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_third_vertex_y_coordinate_l933_93355


namespace NUMINAMATH_CALUDE_church_capacity_l933_93373

/-- Calculates the total number of people that can sit in a church when it's full -/
theorem church_capacity (rows : ℕ) (chairs_per_row : ℕ) (people_per_chair : ℕ) : 
  rows = 20 → chairs_per_row = 6 → people_per_chair = 5 → 
  rows * chairs_per_row * people_per_chair = 600 := by
  sorry

#check church_capacity

end NUMINAMATH_CALUDE_church_capacity_l933_93373


namespace NUMINAMATH_CALUDE_complement_union_problem_l933_93349

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem complement_union_problem : (U \ A) ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_problem_l933_93349


namespace NUMINAMATH_CALUDE_complex_equation_sum_l933_93311

theorem complex_equation_sum (a b : ℝ) :
  (a + 2 * Complex.I) * Complex.I = b + Complex.I →
  ∃ (result : ℝ), a + b = result :=
sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l933_93311


namespace NUMINAMATH_CALUDE_average_xyz_l933_93397

theorem average_xyz (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) : 
  (x + y + z) / 3 = 8 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_xyz_l933_93397


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l933_93313

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ 
   x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) → 
  m > 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l933_93313


namespace NUMINAMATH_CALUDE_problem_statement_l933_93331

theorem problem_statement (a b c d : ℕ+) (r : ℚ) 
  (h1 : r = 1 - (a : ℚ) / b - (c : ℚ) / d)
  (h2 : a + c ≤ 1993)
  (h3 : r > 0) :
  r > 1 / (1993 : ℚ)^3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l933_93331


namespace NUMINAMATH_CALUDE_division_remainder_proof_l933_93351

theorem division_remainder_proof (D R r : ℕ) : 
  D = 12 * 42 + R →
  D = 21 * 24 + r →
  0 ≤ r →
  r < 21 →
  r = 0 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l933_93351


namespace NUMINAMATH_CALUDE_complex_equation_solution_l933_93325

theorem complex_equation_solution (z : ℂ) : (z - 2*Complex.I) * (2 - Complex.I) = 5 → z = 2 + 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l933_93325


namespace NUMINAMATH_CALUDE_system_solution_l933_93357

theorem system_solution (x y : ℝ) 
  (h1 : x * y = 6)
  (h2 : x^2 * y + x * y^2 + x + y = 63) :
  x^2 + y^2 = 69 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l933_93357


namespace NUMINAMATH_CALUDE_cliffs_rock_collection_l933_93336

theorem cliffs_rock_collection (sedimentary : ℕ) (igneous : ℕ) : 
  igneous = sedimentary / 2 →
  (2 : ℕ) * (igneous / 3) = 40 →
  sedimentary + igneous = 180 := by
sorry

end NUMINAMATH_CALUDE_cliffs_rock_collection_l933_93336


namespace NUMINAMATH_CALUDE_min_value_theorem_l933_93389

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (ht : t > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p * q * r * s = 16)
  (h2 : t * u * v * w = 25)
  (h3 : p * t = q * u)
  (h4 : p * t = r * v)
  (h5 : p * t = s * w) :
  (∀ x : ℝ, (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 80) ∧
  (∃ x : ℝ, (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 = 80) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l933_93389


namespace NUMINAMATH_CALUDE_race_time_calculation_l933_93364

/-- Represents a race between two runners A and B -/
structure Race where
  length : ℝ  -- Race length in meters
  lead_distance : ℝ  -- Distance by which A beats B
  lead_time : ℝ  -- Time by which A beats B
  a_time : ℝ  -- Time taken by A to complete the race

/-- Theorem stating that for the given race conditions, A's time is 5.25 seconds -/
theorem race_time_calculation (race : Race) 
  (h1 : race.length = 80)
  (h2 : race.lead_distance = 56)
  (h3 : race.lead_time = 7) :
  race.a_time = 5.25 := by
  sorry

#check race_time_calculation

end NUMINAMATH_CALUDE_race_time_calculation_l933_93364


namespace NUMINAMATH_CALUDE_multiply_by_seven_l933_93365

theorem multiply_by_seven (x : ℝ) (h : 8 * x = 64) : 7 * x = 56 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_seven_l933_93365


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l933_93302

theorem multiplication_addition_equality : 26 * 33 + 67 * 26 = 2600 := by sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l933_93302


namespace NUMINAMATH_CALUDE_sum_is_zero_l933_93308

/-- Given a finite subset M of real numbers with more than 2 elements,
    if for each element the absolute value is at least as large as
    the absolute value of the sum of the other elements,
    then the sum of all elements in M is zero. -/
theorem sum_is_zero (M : Finset ℝ) (h_size : 2 < M.card) :
  (∀ a ∈ M, |a| ≥ |M.sum id - a|) → M.sum id = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_zero_l933_93308


namespace NUMINAMATH_CALUDE_value_of_expression_l933_93390

theorem value_of_expression (x y : ℝ) 
  (h1 : x^2 - x*y = 12) 
  (h2 : y^2 - x*y = 15) : 
  2*(x-y)^2 - 3 = 51 := by
sorry

end NUMINAMATH_CALUDE_value_of_expression_l933_93390


namespace NUMINAMATH_CALUDE_smallest_a2_l933_93304

def sequence_property (a : ℕ → ℝ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ a 2 > 0 ∧
  ∀ n ∈ Finset.range 7, a (n + 2) * a n * a (n - 1) = a (n + 2) + a n + a (n - 1)

def no_extension (a : ℕ → ℝ) : Prop :=
  ∀ x : ℝ, x * a 8 * a 7 ≠ x + a 8 + a 7

theorem smallest_a2 (a : ℕ → ℝ) (h1 : sequence_property a) (h2 : no_extension a) :
  a 2 = Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_a2_l933_93304


namespace NUMINAMATH_CALUDE_roots_relation_l933_93387

-- Define the polynomials
def f (x : ℝ) : ℝ := x^3 + 5*x^2 + 6*x - 8
def g (x u v w : ℝ) : ℝ := x^3 + u*x^2 + v*x + w

-- Define the theorem
theorem roots_relation (p q r u v w : ℝ) : 
  (f p = 0 ∧ f q = 0 ∧ f r = 0) → 
  (g (p+q) u v w = 0 ∧ g (q+r) u v w = 0 ∧ g (r+p) u v w = 0) →
  w = 8 := by
sorry

end NUMINAMATH_CALUDE_roots_relation_l933_93387


namespace NUMINAMATH_CALUDE_ernie_circles_l933_93381

theorem ernie_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) 
  (ali_circles : ℕ) (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) 
  (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) : 
  (total_boxes - ali_circles * ali_boxes_per_circle) / ernie_boxes_per_circle = 4 := by
  sorry

end NUMINAMATH_CALUDE_ernie_circles_l933_93381


namespace NUMINAMATH_CALUDE_sqrt_28_div_sqrt_7_l933_93376

theorem sqrt_28_div_sqrt_7 : Real.sqrt 28 / Real.sqrt 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_28_div_sqrt_7_l933_93376


namespace NUMINAMATH_CALUDE_donation_distribution_l933_93323

theorem donation_distribution (total : ℝ) (contingency : ℝ) : 
  total = 240 →
  contingency = 30 →
  (3 : ℝ) / 8 * total = total - (1 / 3 * total) - (1 / 4 * (total - 1 / 3 * total)) - contingency :=
by sorry

end NUMINAMATH_CALUDE_donation_distribution_l933_93323


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l933_93309

/-- Represents the number of times fruits are taken out -/
def x : ℕ := sorry

/-- The original number of apples -/
def initial_apples : ℕ := 3 * x + 1

/-- The original number of oranges -/
def initial_oranges : ℕ := 4 * x + 12

/-- The condition that the number of oranges is twice that of apples -/
axiom orange_apple_ratio : initial_oranges = 2 * initial_apples

theorem fruit_basket_problem : x = 5 := by sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l933_93309


namespace NUMINAMATH_CALUDE_yue_bao_scientific_notation_l933_93341

theorem yue_bao_scientific_notation : 5853 = 5.853 * (10 ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_yue_bao_scientific_notation_l933_93341


namespace NUMINAMATH_CALUDE_square_measurement_error_l933_93344

theorem square_measurement_error (area_error : Real) (side_error : Real) : 
  area_error = 8.16 → side_error = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_measurement_error_l933_93344


namespace NUMINAMATH_CALUDE_line_intersects_y_axis_l933_93332

/-- A line passing through two given points intersects the y-axis at (0, 0) -/
theorem line_intersects_y_axis (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁ = 3 ∧ y₁ = 9)
  (h₂ : x₂ = -7 ∧ y₂ = -21) :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) ∧
    0 = m * 0 + b :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_y_axis_l933_93332


namespace NUMINAMATH_CALUDE_b_100_mod_50_l933_93352

/-- Define the sequence b_n = 7^n + 9^n -/
def b (n : ℕ) : ℕ := 7^n + 9^n

/-- Theorem: b_100 ≡ 2 (mod 50) -/
theorem b_100_mod_50 : b 100 ≡ 2 [MOD 50] := by
  sorry

end NUMINAMATH_CALUDE_b_100_mod_50_l933_93352


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l933_93391

theorem quadratic_roots_property (α β : ℝ) : 
  (α^2 + α - 1 = 0) → 
  (β^2 + β - 1 = 0) → 
  (α ≠ β) →
  α^4 - 3*β = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l933_93391


namespace NUMINAMATH_CALUDE_lcm_prime_sum_l933_93320

theorem lcm_prime_sum (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x > y → Nat.lcm x y = 10 → 2 * x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_lcm_prime_sum_l933_93320


namespace NUMINAMATH_CALUDE_average_difference_l933_93350

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 7 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l933_93350


namespace NUMINAMATH_CALUDE_greatest_n_product_consecutive_odds_l933_93346

theorem greatest_n_product_consecutive_odds : 
  ∃ (n : ℕ), n = 899 ∧ 
  n < 1000 ∧ 
  (∃ (m : ℤ), 4 * n^3 - 3 * n = (2 * m - 1) * (2 * m + 1)) ∧
  (∀ (k : ℕ), k < 1000 → k > n → 
    ¬∃ (m : ℤ), 4 * k^3 - 3 * k = (2 * m - 1) * (2 * m + 1)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_n_product_consecutive_odds_l933_93346


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_N_l933_93396

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2 - 2}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 2}

-- Statement to prove
theorem M_intersect_N_equals_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_N_l933_93396


namespace NUMINAMATH_CALUDE_count_integers_in_list_integers_in_list_D_l933_93377

def consecutive_integers (start : Int) (count : Nat) : List Int :=
  List.range count |>.map (fun i => start + i)

theorem count_integers_in_list (start : Int) (positive_range : Nat) : 
  let list := consecutive_integers start (positive_range + start.natAbs + 1)
  list.length = positive_range + start.natAbs + 1 :=
by sorry

-- The main theorem
theorem integers_in_list_D : 
  let start := -4
  let positive_range := 6
  let list_D := consecutive_integers start (positive_range + start.natAbs + 1)
  list_D.length = 12 :=
by sorry

end NUMINAMATH_CALUDE_count_integers_in_list_integers_in_list_D_l933_93377


namespace NUMINAMATH_CALUDE_probability_one_from_each_name_l933_93339

def total_cards : ℕ := 12
def alice_letters : ℕ := 5
def bob_letters : ℕ := 7

theorem probability_one_from_each_name :
  let prob_alice_then_bob := (alice_letters : ℚ) / total_cards * bob_letters / (total_cards - 1)
  let prob_bob_then_alice := (bob_letters : ℚ) / total_cards * alice_letters / (total_cards - 1)
  prob_alice_then_bob + prob_bob_then_alice = 35 / 66 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_from_each_name_l933_93339


namespace NUMINAMATH_CALUDE_mikas_height_mikas_height_is_70_l933_93360

/-- Proves that Mika's current height is 70 inches given the problem conditions -/
theorem mikas_height (original_height : ℝ) (sheas_growth_rate : ℝ) (mikas_growth_ratio : ℝ) 
  (sheas_current_height : ℝ) : ℝ :=
  let sheas_growth := sheas_current_height - original_height
  let mikas_growth := mikas_growth_ratio * sheas_growth
  original_height + mikas_growth
where
  -- Shea and Mika were originally the same height
  original_height_positive : 0 < original_height := by sorry
  -- Shea has grown by 25%
  sheas_growth_rate_def : sheas_growth_rate = 0.25 := by sorry
  -- Mika has grown two-thirds as many inches as Shea
  mikas_growth_ratio_def : mikas_growth_ratio = 2/3 := by sorry
  -- Shea is now 75 inches tall
  sheas_current_height_def : sheas_current_height = 75 := by sorry
  -- Shea's current height is 25% more than the original height
  sheas_growth_equation : sheas_current_height = original_height * (1 + sheas_growth_rate) := by sorry

theorem mikas_height_is_70 : mikas_height 60 0.25 (2/3) 75 = 70 := by sorry

end NUMINAMATH_CALUDE_mikas_height_mikas_height_is_70_l933_93360


namespace NUMINAMATH_CALUDE_complex_equation_solution_l933_93347

theorem complex_equation_solution (z : ℂ) : (1 + 2*I)*z = 5 → z = 1 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l933_93347


namespace NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l933_93388

theorem tan_sum_pi_twelfths : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 * Real.sqrt 2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_twelfths_l933_93388


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l933_93384

/-- Given two rectangles of equal area, where one rectangle has dimensions 6 inches by 50 inches,
    and the other has a width of 20 inches, prove that the length of the second rectangle is 15 inches. -/
theorem equal_area_rectangles (area : ℝ) (length_jordan width_jordan width_carol : ℝ) :
  area = length_jordan * width_jordan →
  length_jordan = 6 →
  width_jordan = 50 →
  width_carol = 20 →
  ∃ length_carol : ℝ, area = length_carol * width_carol ∧ length_carol = 15 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l933_93384


namespace NUMINAMATH_CALUDE_school_boys_count_l933_93353

/-- The number of girls in the school -/
def num_girls : ℕ := 34

/-- The difference between the number of boys and girls -/
def difference : ℕ := 807

/-- The number of boys in the school -/
def num_boys : ℕ := num_girls + difference

theorem school_boys_count : num_boys = 841 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l933_93353


namespace NUMINAMATH_CALUDE_investment_profit_ratio_l933_93315

/-- Represents the profit ratio between two investors based on their capital and investment duration. -/
def profit_ratio (capital_a capital_b : ℕ) (duration_a duration_b : ℚ) : ℚ × ℚ :=
  let contribution_a := capital_a * duration_a
  let contribution_b := capital_b * duration_b
  (contribution_a, contribution_b)

/-- Theorem stating that given the specified investments and durations, the profit ratio is 2:1. -/
theorem investment_profit_ratio :
  let (ratio_a, ratio_b) := profit_ratio 27000 36000 12 (9/2)
  ratio_a / ratio_b = 2 := by
  sorry

end NUMINAMATH_CALUDE_investment_profit_ratio_l933_93315


namespace NUMINAMATH_CALUDE_equal_expressions_l933_93327

theorem equal_expressions : 10006 - 8008 = 10000 - 8002 := by sorry

end NUMINAMATH_CALUDE_equal_expressions_l933_93327


namespace NUMINAMATH_CALUDE_statue_weight_calculation_l933_93362

/-- The weight of a marble statue after three weeks of carving --/
def final_statue_weight (initial_weight : ℝ) : ℝ :=
  let week1_remainder := initial_weight * (1 - 0.28)
  let week2_remainder := week1_remainder * (1 - 0.18)
  let week3_remainder := week2_remainder * (1 - 0.20)
  week3_remainder

/-- Theorem stating the final weight of the statue --/
theorem statue_weight_calculation :
  final_statue_weight 180 = 85.0176 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_calculation_l933_93362


namespace NUMINAMATH_CALUDE_bizarre_coin_expected_value_l933_93301

/-- A bizarre weighted coin with three possible outcomes -/
inductive CoinOutcome
| Heads
| Tails
| Edge

/-- The probability of each outcome for the bizarre weighted coin -/
def probability (outcome : CoinOutcome) : ℚ :=
  match outcome with
  | CoinOutcome.Heads => 1/4
  | CoinOutcome.Tails => 1/2
  | CoinOutcome.Edge => 1/4

/-- The payoff for each outcome of the bizarre weighted coin -/
def payoff (outcome : CoinOutcome) : ℤ :=
  match outcome with
  | CoinOutcome.Heads => 1
  | CoinOutcome.Tails => 3
  | CoinOutcome.Edge => -8

/-- The expected value of flipping the bizarre weighted coin -/
def expected_value : ℚ :=
  (probability CoinOutcome.Heads * payoff CoinOutcome.Heads) +
  (probability CoinOutcome.Tails * payoff CoinOutcome.Tails) +
  (probability CoinOutcome.Edge * payoff CoinOutcome.Edge)

/-- Theorem stating that the expected value of flipping the bizarre weighted coin is -1/4 -/
theorem bizarre_coin_expected_value :
  expected_value = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_bizarre_coin_expected_value_l933_93301


namespace NUMINAMATH_CALUDE_isosceles_triangle_l933_93366

theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a + b = Real.tan (C / 2) * (a * Real.tan A + b * Real.tan B) →
  A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l933_93366


namespace NUMINAMATH_CALUDE_max_value_constraint_l933_93330

theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2*x^2 - 3*x*y + 4*y^2 = 15) : 
  3*x^2 + 2*x*y + y^2 ≤ 50*Real.sqrt 3 + 65 := by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l933_93330


namespace NUMINAMATH_CALUDE_first_car_speed_l933_93318

theorem first_car_speed (v : ℝ) (h1 : v > 0) : 
  v * 2.25 * 4 = 720 → v * 1.25 = 100 := by
  sorry

end NUMINAMATH_CALUDE_first_car_speed_l933_93318


namespace NUMINAMATH_CALUDE_greatest_root_of_g_l933_93354

def g (x : ℝ) : ℝ := 12 * x^5 - 24 * x^3 + 9 * x

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt (3/2) ∧
  g r = 0 ∧
  ∀ x, g x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_g_l933_93354


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l933_93348

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The second term of a sequence. -/
def second_term (a : ℕ → ℝ) : ℝ := a 1

theorem arithmetic_sequence_second_term
  (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) (h_sum : a 0 + a 2 = 8) :
  second_term a = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l933_93348


namespace NUMINAMATH_CALUDE_seven_is_target_digit_l933_93399

/-- The numeral we're examining -/
def numeral : ℕ := 657903

/-- The difference between local value and face value -/
def difference : ℕ := 6993

/-- Function to get the local value of a digit in a specific place -/
def localValue (digit : ℕ) (place : ℕ) : ℕ := digit * (10 ^ place)

/-- Function to get the face value of a digit -/
def faceValue (digit : ℕ) : ℕ := digit

/-- Theorem stating that 7 is the only digit in the numeral with the given difference -/
theorem seven_is_target_digit :
  ∃! d : ℕ, d < 10 ∧ 
    (∃ p : ℕ, p < 6 ∧ 
      (numeral / (10 ^ p)) % 10 = d ∧
      localValue d p - faceValue d = difference) :=
sorry

end NUMINAMATH_CALUDE_seven_is_target_digit_l933_93399


namespace NUMINAMATH_CALUDE_flower_bed_fraction_l933_93340

-- Define the yard and flower beds
def yard_length : ℝ := 25
def yard_width : ℝ := 5
def flower_bed_area : ℝ := 50

-- Define the theorem
theorem flower_bed_fraction :
  let total_yard_area := yard_length * yard_width
  let total_flower_bed_area := 2 * flower_bed_area
  (total_flower_bed_area / total_yard_area) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_l933_93340


namespace NUMINAMATH_CALUDE_divisibility_problem_specific_divisibility_problem_l933_93370

theorem divisibility_problem (n : ℕ) (d : ℕ) (h : d > 0) :
  ∃ (x y : ℕ),
    (x = (d - n % d) % d) ∧
    (y = n % d) ∧
    ((n + x) % d = 0) ∧
    ((n - y) % d = 0) ∧
    (∀ x' : ℕ, x' < x → (n + x') % d ≠ 0) ∧
    (∀ y' : ℕ, y' < y → (n - y') % d ≠ 0) :=
by sorry

-- Specific instance for the given problem
theorem specific_divisibility_problem :
  ∃ (x y : ℕ),
    (x = 10) ∧
    (y = 27) ∧
    ((1100 + x) % 37 = 0) ∧
    ((1100 - y) % 37 = 0) ∧
    (∀ x' : ℕ, x' < x → (1100 + x') % 37 ≠ 0) ∧
    (∀ y' : ℕ, y' < y → (1100 - y') % 37 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_specific_divisibility_problem_l933_93370


namespace NUMINAMATH_CALUDE_no_valid_prime_pairs_l933_93321

theorem no_valid_prime_pairs : 
  ∀ a b : ℕ, 
    Prime a → 
    Prime b → 
    b > a → 
    (a - 8) * (b - 8) = 64 → 
    False :=
by
  sorry

end NUMINAMATH_CALUDE_no_valid_prime_pairs_l933_93321


namespace NUMINAMATH_CALUDE_inequality_proof_l933_93319

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a + b = 2) :
  a * b < 1 ∧ 1 < (a^2 + b^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l933_93319


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l933_93358

theorem square_of_binomial_constant (p : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + p = (a * x + b)^2) → p = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l933_93358


namespace NUMINAMATH_CALUDE_smallest_n_for_m_independent_same_color_lines_l933_93314

/-- A two-coloring of a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Fin 2

/-- Predicate for m lines of the same color with no common endpoints -/
def HasMIndependentSameColorLines (c : TwoColoring n) (m : ℕ) : Prop :=
  ∃ (edges : Fin m → Fin n × Fin n),
    (∀ i j, i ≠ j → (edges i).1 ≠ (edges j).1 ∧ (edges i).1 ≠ (edges j).2 ∧
                    (edges i).2 ≠ (edges j).1 ∧ (edges i).2 ≠ (edges j).2) ∧
    (∀ i j, c (edges i).1 (edges i).2 = c (edges j).1 (edges j).2)

/-- The main theorem -/
theorem smallest_n_for_m_independent_same_color_lines (m : ℕ) :
  (∀ n, n ≥ 3 * m - 1 → ∀ c : TwoColoring n, HasMIndependentSameColorLines c m) ∧
  (∀ n, n < 3 * m - 1 → ∃ c : TwoColoring n, ¬HasMIndependentSameColorLines c m) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_m_independent_same_color_lines_l933_93314


namespace NUMINAMATH_CALUDE_least_six_digit_congruent_to_3_mod_17_l933_93383

theorem least_six_digit_congruent_to_3_mod_17 : ∃ (n : ℕ), 
  (n ≥ 100000 ∧ n < 1000000) ∧ 
  n % 17 = 3 ∧
  ∀ (m : ℕ), (m ≥ 100000 ∧ m < 1000000 ∧ m % 17 = 3) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_six_digit_congruent_to_3_mod_17_l933_93383


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l933_93337

/-- Given that the coefficient of x^3y^3 in the expansion of (x+ay)^6 is -160, prove that a = -2 -/
theorem binomial_coefficient_problem (a : ℝ) : 
  (Nat.choose 6 3 : ℝ) * a^3 = -160 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l933_93337


namespace NUMINAMATH_CALUDE_runners_meet_count_l933_93326

-- Constants
def total_time : ℝ := 45
def odell_speed : ℝ := 260
def odell_radius : ℝ := 70
def kershaw_speed : ℝ := 320
def kershaw_radius : ℝ := 80
def kershaw_delay : ℝ := 5

-- Theorem statement
theorem runners_meet_count :
  let odell_angular_speed := odell_speed / odell_radius
  let kershaw_angular_speed := kershaw_speed / kershaw_radius
  let relative_angular_speed := odell_angular_speed + kershaw_angular_speed
  let effective_time := total_time - kershaw_delay
  let meet_count := ⌊(effective_time * relative_angular_speed) / (2 * Real.pi)⌋
  meet_count = 49 := by
  sorry

end NUMINAMATH_CALUDE_runners_meet_count_l933_93326


namespace NUMINAMATH_CALUDE_square_side_length_l933_93306

theorem square_side_length (perimeter : ℝ) (area : ℝ) (h1 : perimeter = 44) (h2 : area = 121) :
  ∃ (side : ℝ), side * 4 = perimeter ∧ side * side = area ∧ side = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l933_93306


namespace NUMINAMATH_CALUDE_max_m_value_l933_93359

theorem max_m_value (m : ℝ) (h1 : m > 1) 
  (h2 : ∃ x : ℝ, x ∈ Set.Icc (-2) 0 ∧ x^2 + 2*m*x + m^2 - m ≤ 0) : 
  (∀ n : ℝ, (n > 1 ∧ ∃ y : ℝ, y ∈ Set.Icc (-2) 0 ∧ y^2 + 2*n*y + n^2 - n ≤ 0) → n ≤ m) →
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l933_93359


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l933_93398

def total_players : ℕ := 15
def lineup_size : ℕ := 5
def preselected_players : ℕ := 3

theorem starting_lineup_combinations : 
  Nat.choose (total_players - preselected_players) (lineup_size - preselected_players) = 66 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l933_93398


namespace NUMINAMATH_CALUDE_gym_cost_theorem_l933_93392

/-- Calculates the total cost for gym memberships and personal training for one year -/
def total_gym_cost (cheap_monthly : ℝ) (cheap_signup : ℝ) (cheap_maintenance : ℝ)
                   (expensive_monthly_factor : ℝ) (expensive_signup_months : ℝ) (expensive_maintenance : ℝ)
                   (signup_discount : ℝ) (cheap_pt_base : ℝ) (cheap_pt_discount : ℝ)
                   (expensive_pt_base : ℝ) (expensive_pt_discount : ℝ) : ℝ :=
  let cheap_total := cheap_monthly * 12 + cheap_signup * (1 - signup_discount) + cheap_maintenance +
                     (cheap_pt_base * 10 + cheap_pt_base * (1 - cheap_pt_discount) * 10)
  let expensive_monthly := cheap_monthly * expensive_monthly_factor
  let expensive_total := expensive_monthly * 12 + (expensive_monthly * expensive_signup_months) * (1 - signup_discount) +
                         expensive_maintenance + (expensive_pt_base * 5 + expensive_pt_base * (1 - expensive_pt_discount) * 10)
  cheap_total + expensive_total

/-- The theorem states that the total gym cost for the given parameters is $1780.50 -/
theorem gym_cost_theorem :
  total_gym_cost 10 50 30 3 4 60 0.1 25 0.2 45 0.15 = 1780.50 := by
  sorry

end NUMINAMATH_CALUDE_gym_cost_theorem_l933_93392


namespace NUMINAMATH_CALUDE_staircase_extension_l933_93386

def toothpicks_for_step (n : ℕ) : ℕ := 12 + 2 * (n - 5)

theorem staircase_extension : 
  (toothpicks_for_step 5) + (toothpicks_for_step 6) = 26 :=
by sorry

end NUMINAMATH_CALUDE_staircase_extension_l933_93386


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l933_93374

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n ≥ 2, a n - a (n - 1) = 2) ∧ (a 1 = 1)

theorem tenth_term_of_sequence (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 19 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l933_93374


namespace NUMINAMATH_CALUDE_product_digit_sum_l933_93324

/-- Represents a 101-digit number with alternating digits --/
def AlternatingDigitNumber (a b : ℕ) : ℕ := sorry

/-- The first 101-digit number: 1010101...010101 --/
def num1 : ℕ := AlternatingDigitNumber 1 0

/-- The second 101-digit number: 7070707...070707 --/
def num2 : ℕ := AlternatingDigitNumber 7 0

/-- Returns the hundreds digit of a natural number --/
def hundredsDigit (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a natural number --/
def unitsDigit (n : ℕ) : ℕ := sorry

theorem product_digit_sum :
  hundredsDigit (num1 * num2) + unitsDigit (num1 * num2) = 10 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l933_93324


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l933_93316

theorem rectangular_prism_diagonal (l w h : ℝ) (hl : l = 15) (hw : w = 25) (hh : h = 12) :
  Real.sqrt (l^2 + w^2 + h^2) = Real.sqrt 994 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l933_93316


namespace NUMINAMATH_CALUDE_quadrilateral_property_l933_93375

-- Define the quadrilateral ABCD
variable (A B C D : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- State the theorem
theorem quadrilateral_property (h1 : angle A D C = 135)
  (h2 : angle A D B - angle A B D = 2 * angle D A B)
  (h3 : angle A D B - angle A B D = 4 * angle C B D)
  (h4 : distance B C = Real.sqrt 2 * distance C D) :
  distance A B = distance B C + distance A D := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_property_l933_93375


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l933_93305

theorem quadratic_coefficient (b : ℝ) : 
  ((-14 : ℝ)^2 + b * (-14) + 49 = 0) → b = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l933_93305
