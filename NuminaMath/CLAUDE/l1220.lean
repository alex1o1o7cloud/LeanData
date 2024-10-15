import Mathlib

namespace NUMINAMATH_CALUDE_shorter_tree_height_l1220_122007

theorem shorter_tree_height (h1 h2 : ℝ) : 
  h2 = h1 + 20 →  -- One tree is 20 feet taller
  h1 / h2 = 5 / 7 →  -- Heights are in ratio 5:7
  h1 + h2 = 240 →  -- Sum of heights is 240 feet
  h1 = 110 :=  -- Shorter tree is 110 feet tall
by sorry

end NUMINAMATH_CALUDE_shorter_tree_height_l1220_122007


namespace NUMINAMATH_CALUDE_regular_pay_limit_l1220_122040

/-- The problem of finding the limit for regular pay. -/
theorem regular_pay_limit (regular_rate : ℝ) (overtime_rate : ℝ) (total_pay : ℝ) (overtime_hours : ℝ) :
  regular_rate = 3 →
  overtime_rate = 2 * regular_rate →
  total_pay = 186 →
  overtime_hours = 11 →
  ∃ (regular_hours : ℝ),
    regular_hours * regular_rate + overtime_hours * overtime_rate = total_pay ∧
    regular_hours = 40 :=
by sorry

end NUMINAMATH_CALUDE_regular_pay_limit_l1220_122040


namespace NUMINAMATH_CALUDE_curve_intersects_all_planes_l1220_122052

/-- A smooth curve in ℝ³ -/
def C : ℝ → ℝ × ℝ × ℝ := fun t ↦ (t, t^3, t^5)

/-- Definition of a plane in ℝ³ -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  not_all_zero : A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0

/-- The theorem stating that the curve C intersects every plane -/
theorem curve_intersects_all_planes :
  ∀ (p : Plane), ∃ (t : ℝ), 
    let (x, y, z) := C t
    p.A * x + p.B * y + p.C * z + p.D = 0 := by
  sorry


end NUMINAMATH_CALUDE_curve_intersects_all_planes_l1220_122052


namespace NUMINAMATH_CALUDE_vasyas_birthday_l1220_122042

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

theorem vasyas_birthday (today : DayOfWeek) 
  (h1 : next_day (next_day today) = DayOfWeek.Sunday) -- Sunday is the day after tomorrow
  (h2 : ∃ birthday : DayOfWeek, next_day birthday = today) -- Today is the day after Vasya's birthday
  : ∃ birthday : DayOfWeek, birthday = DayOfWeek.Thursday := by
  sorry

end NUMINAMATH_CALUDE_vasyas_birthday_l1220_122042


namespace NUMINAMATH_CALUDE_fraction_simplification_l1220_122071

theorem fraction_simplification (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) (h3 : a ≠ 2) :
  (a + 1) / (a^2 - 1) / ((a^2 - 4) / (a^2 + a - 2)) - (1 - a) / (a - 2) = a / (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1220_122071


namespace NUMINAMATH_CALUDE_at_least_one_negative_l1220_122008

theorem at_least_one_negative (a b : ℝ) (h : a + b < 0) :
  a < 0 ∨ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_negative_l1220_122008


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1220_122061

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h_sum : x + y = 3 * x * y) (h_diff : x - y = 1) :
  1 / x + 1 / y = Real.sqrt 13 + 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1220_122061


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l1220_122022

/-- Given a line segment CD with midpoint N and endpoint C, proves that the product of D's coordinates is 39 -/
theorem midpoint_coordinate_product (C N D : ℝ × ℝ) : 
  C = (5, 3) → N = (4, 8) → N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → D.1 * D.2 = 39 := by
  sorry

#check midpoint_coordinate_product

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l1220_122022


namespace NUMINAMATH_CALUDE_equation_equivalence_l1220_122077

theorem equation_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 3 * x = 7 * y) : 
  x / 7 = y / 3 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1220_122077


namespace NUMINAMATH_CALUDE_dalmatian_spots_l1220_122079

theorem dalmatian_spots (b p : ℕ) (h1 : b = 2 * p - 1) (h2 : b + p = 59) : b = 39 := by
  sorry

end NUMINAMATH_CALUDE_dalmatian_spots_l1220_122079


namespace NUMINAMATH_CALUDE_sum_of_roots_l1220_122078

theorem sum_of_roots (a b : ℝ) : 
  a ≠ b → a * (a - 6) = 7 → b * (b - 6) = 7 → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1220_122078


namespace NUMINAMATH_CALUDE_f_monotone_intervals_f_B_range_l1220_122058

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sin x ^ 2 + 1 / 2

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

theorem f_monotone_intervals (k : ℤ) :
  is_monotone_increasing f (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) := by sorry

theorem f_B_range (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < Real.pi / 2 →
  b * Real.cos (2 * A) = b * Real.cos A - a * Real.sin B →
  ∃ x, f B = x ∧ -Real.sqrt 2 / 2 ≤ x ∧ x ≤ Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_f_monotone_intervals_f_B_range_l1220_122058


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1220_122021

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 - i) / (1 + i) = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1220_122021


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1220_122006

theorem complex_modulus_problem : 
  Complex.abs ((1 + Complex.I) * (2 - Complex.I)) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1220_122006


namespace NUMINAMATH_CALUDE_equality_check_l1220_122044

theorem equality_check : 
  ((-2 : ℤ)^3 ≠ -2 * 3) ∧ 
  (2^3 ≠ 3^2) ∧ 
  ((-2 : ℤ)^3 = -2^3) ∧ 
  (-3^2 ≠ (-3)^2) := by
  sorry

end NUMINAMATH_CALUDE_equality_check_l1220_122044


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1220_122035

theorem square_area_from_diagonal (diagonal : ℝ) (area : ℝ) : 
  diagonal = 12 * Real.sqrt 2 → area = 144 → 
  diagonal^2 / 2 = area := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1220_122035


namespace NUMINAMATH_CALUDE_money_distribution_l1220_122018

/-- Given a total amount of money and the fraction one person has relative to the others,
    calculate how much money that person has. -/
theorem money_distribution (total : ℕ) (fraction : ℚ) (person_amount : ℕ) : 
  total = 7000 →
  fraction = 2 / 3 →
  person_amount = total * (fraction / (1 + fraction)) →
  person_amount = 2800 := by
  sorry

#check money_distribution

end NUMINAMATH_CALUDE_money_distribution_l1220_122018


namespace NUMINAMATH_CALUDE_square_EFGH_area_l1220_122050

theorem square_EFGH_area : 
  ∀ (original_side_length : ℝ) (EFGH_side_length : ℝ),
  original_side_length = 8 →
  EFGH_side_length = original_side_length + 2 * (original_side_length / 2) →
  EFGH_side_length^2 = 256 :=
by sorry

end NUMINAMATH_CALUDE_square_EFGH_area_l1220_122050


namespace NUMINAMATH_CALUDE_ant_distance_theorem_l1220_122095

def ant_movement : List (ℝ × ℝ) := [(-7, 0), (0, 5), (3, 0), (0, -2), (9, 0), (0, -2), (-1, 0), (0, -1)]

def total_displacement (movements : List (ℝ × ℝ)) : ℝ × ℝ :=
  movements.foldl (λ (acc : ℝ × ℝ) (move : ℝ × ℝ) => (acc.1 + move.1, acc.2 + move.2)) (0, 0)

theorem ant_distance_theorem :
  let final_position := total_displacement ant_movement
  Real.sqrt (final_position.1 ^ 2 + final_position.2 ^ 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ant_distance_theorem_l1220_122095


namespace NUMINAMATH_CALUDE_angle_point_cosine_l1220_122086

/-- Given an angle α and a real number a, proves that if the terminal side of α
    passes through point P(3a, 4) and cos α = -3/5, then a = -1. -/
theorem angle_point_cosine (α : Real) (a : Real) : 
  (∃ r : Real, r > 0 ∧ 3 * a = r * Real.cos α ∧ 4 = r * Real.sin α) → 
  Real.cos α = -3/5 → 
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_angle_point_cosine_l1220_122086


namespace NUMINAMATH_CALUDE_lilys_books_l1220_122046

theorem lilys_books (books_last_month : ℕ) : 
  books_last_month + 2 * books_last_month = 12 → books_last_month = 4 := by
  sorry

end NUMINAMATH_CALUDE_lilys_books_l1220_122046


namespace NUMINAMATH_CALUDE_rectangle_square_cut_l1220_122088

theorem rectangle_square_cut (m n : ℕ) (hm : m > 2) (hn : n > 2) :
  (m - 2) * (n - 2) = 8 ↔
  (2 * (m + n) - 4 = m * n) ∧ (m * n - 4 = 2 * (m + n)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_square_cut_l1220_122088


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l1220_122069

/-- An isosceles trapezoid with the given measurements --/
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longerBase : ℝ

/-- The area of an isosceles trapezoid --/
def trapezoidArea (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- The theorem stating the area of the specific trapezoid --/
theorem specific_trapezoid_area : 
  let t : IsoscelesTrapezoid := { 
    leg := 20,
    diagonal := 25,
    longerBase := 30
  }
  abs (trapezoidArea t - 315.82) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l1220_122069


namespace NUMINAMATH_CALUDE_equation_solution_l1220_122036

theorem equation_solution (x : ℝ) : 
  (3 / (x^2 + x) - x^2 = 2 + x) → (2*x^2 + 2*x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1220_122036


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1220_122055

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a := by sorry

theorem sum_of_roots_specific_quadratic :
  let r₁ := (5 + Real.sqrt 1) / 2
  let r₂ := (5 - Real.sqrt 1) / 2
  r₁ + r₂ = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l1220_122055


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l1220_122030

theorem quadratic_equation_condition (m : ℝ) : 
  (|m - 1| = 2 ∧ m + 1 ≠ 0) ↔ m = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l1220_122030


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l1220_122023

-- Part 1
theorem calculation_proof :
  |Real.sqrt 5 - 3| + (1/2)⁻¹ - Real.sqrt 20 + Real.sqrt 3 * Real.cos (30 * π / 180) = 13/2 - 3 * Real.sqrt 5 := by
  sorry

-- Part 2
theorem inequality_system_solution (x : ℝ) :
  (-3 * (x - 2) ≥ 4 - x ∧ (1 + 2*x) / 3 > x - 1) ↔ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l1220_122023


namespace NUMINAMATH_CALUDE_speech_competition_selection_l1220_122056

def total_students : Nat := 9
def num_boys : Nat := 5
def num_girls : Nat := 4
def students_to_select : Nat := 4

def selection_methods : Nat := sorry

theorem speech_competition_selection :
  (total_students = num_boys + num_girls) →
  (students_to_select ≤ total_students) →
  (selection_methods = 86) := by sorry

end NUMINAMATH_CALUDE_speech_competition_selection_l1220_122056


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l1220_122066

-- Define the quadratic polynomial
def q (x : ℚ) : ℚ := (29 * x^2 - 44 * x + 135) / 15

-- State the theorem
theorem quadratic_polynomial_satisfies_conditions :
  q (-1) = 6 ∧ q 2 = 1 ∧ q 4 = 17 := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l1220_122066


namespace NUMINAMATH_CALUDE_clever_calculation_l1220_122041

theorem clever_calculation :
  (1978 + 250 + 1022 + 750 = 4000) ∧
  (454 + 999 * 999 + 545 = 999000) ∧
  (999 + 998 + 997 + 996 + 1004 + 1003 + 1002 + 1001 = 8000) := by
  sorry

end NUMINAMATH_CALUDE_clever_calculation_l1220_122041


namespace NUMINAMATH_CALUDE_power_division_equality_l1220_122092

theorem power_division_equality (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l1220_122092


namespace NUMINAMATH_CALUDE_triangle_area_234_l1220_122049

theorem triangle_area_234 : 
  let a := 2
  let b := 3
  let c := 4
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = (3 * Real.sqrt 15) / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_234_l1220_122049


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1220_122085

theorem least_addition_for_divisibility : 
  ∃ (x : ℕ), (1056 + x) % 23 = 0 ∧ ∀ (y : ℕ), y < x → (1056 + y) % 23 ≠ 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1220_122085


namespace NUMINAMATH_CALUDE_ant_journey_l1220_122011

-- Define the plane and points A and B
variable (Plane : Type) (A B : Plane)

-- Define the distance functions from A and B
variable (distA distB : ℝ → ℝ)

-- Define the conditions
variable (h1 : distA 7 = 5)
variable (h2 : distB 7 = 3)
variable (h3 : distB 0 = 0)
variable (h4 : distA 0 = 4)

-- Define the distance between A and B
def dist_AB : ℝ := 4

-- Define the theorem
theorem ant_journey :
  (∃ t1 t2 : ℝ, t1 ≠ t2 ∧ 0 ≤ t1 ∧ t1 ≤ 9 ∧ 0 ≤ t2 ∧ t2 ≤ 9 ∧ distA t1 = distB t1 ∧ distA t2 = distB t2) ∧
  (dist_AB = 4) ∧
  (∃ t1 t2 t3 : ℝ, t1 ≠ t2 ∧ t2 ≠ t3 ∧ t1 ≠ t3 ∧ 
    0 ≤ t1 ∧ t1 ≤ 9 ∧ 0 ≤ t2 ∧ t2 ≤ 9 ∧ 0 ≤ t3 ∧ t3 ≤ 9 ∧
    distA t1 + distB t1 = dist_AB ∧
    distA t2 + distB t2 = dist_AB ∧
    distA t3 + distB t3 = dist_AB) ∧
  (∃ d : ℝ, d = 8 ∧ 
    d = |distA 3 - distA 0| + |distA 5 - distA 3| + |distA 7 - distA 5| + |distA 9 - distA 7|) :=
by sorry

end NUMINAMATH_CALUDE_ant_journey_l1220_122011


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l1220_122001

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 10 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y + 2 * y + 10 = 0 → y = x) ↔ 
  (k = 2 - 2 * Real.sqrt 30 ∨ k = -2 - 2 * Real.sqrt 30) :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l1220_122001


namespace NUMINAMATH_CALUDE_min_score_for_average_l1220_122004

def total_tests : ℕ := 7
def max_score : ℕ := 100
def target_average : ℕ := 80

def first_four_scores : List ℕ := [82, 90, 78, 85]

theorem min_score_for_average (scores : List ℕ) 
  (h1 : scores.length = 4)
  (h2 : ∀ s ∈ scores, s ≤ max_score) :
  ∃ (x y z : ℕ),
    x ≤ max_score ∧ y ≤ max_score ∧ z ≤ max_score ∧
    (scores.sum + x + y + z) / total_tests = target_average ∧
    (∀ a b c : ℕ, 
      a ≤ max_score → b ≤ max_score → c ≤ max_score →
      (scores.sum + a + b + c) / total_tests = target_average →
      min x y ≤ min a b ∧ min x y ≤ c) ∧
    25 = min x (min y z) := by
  sorry

#check min_score_for_average first_four_scores

end NUMINAMATH_CALUDE_min_score_for_average_l1220_122004


namespace NUMINAMATH_CALUDE_tangents_divide_plane_l1220_122032

/-- The number of regions created by n lines in a plane --/
def num_regions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => num_regions k + (k + 1)

/-- Theorem: 7 distinct tangents to a circle divide the plane into 29 regions --/
theorem tangents_divide_plane : num_regions 7 = 29 := by
  sorry

/-- Lemma: The number of regions for n tangents follows the recursive formula R(n) = R(n-1) + n --/
lemma regions_recursive_formula (n : ℕ) : num_regions (n + 1) = num_regions n + (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_tangents_divide_plane_l1220_122032


namespace NUMINAMATH_CALUDE_periodic_coloring_divides_l1220_122099

/-- A coloring of the integers -/
def Coloring := ℤ → Bool

/-- A coloring is t-periodic if it repeats every t steps -/
def isPeriodic (c : Coloring) (t : ℕ) : Prop :=
  ∀ x : ℤ, c x = c (x + t)

/-- For a given x, exactly one of x + a₁, ..., x + aₙ is colored -/
def hasUniqueColoredSum (c : Coloring) (a : Fin n → ℕ) : Prop :=
  ∀ x : ℤ, ∃! i : Fin n, c (x + a i)

theorem periodic_coloring_divides (n : ℕ) (t : ℕ) (a : Fin n → ℕ) (h_a : StrictMono a) 
    (c : Coloring) (h_periodic : isPeriodic c t) (h_unique : hasUniqueColoredSum c a) : 
    n ∣ t := by sorry

end NUMINAMATH_CALUDE_periodic_coloring_divides_l1220_122099


namespace NUMINAMATH_CALUDE_T_is_Y_shape_l1220_122089

/-- The set T of points (x, y) in the coordinate plane -/
def T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (5 = x + 3 ∧ y - 6 < 5) ∨
               (5 = y - 6 ∧ x + 3 < 5) ∨
               (x + 3 = y - 6 ∧ 5 < x + 3)}

/-- The common start point of the "Y" shape -/
def commonPoint : ℝ × ℝ := (2, 11)

/-- The vertical line segment of the "Y" shape -/
def verticalSegment : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 2 ∧ p.2 < 11}

/-- The horizontal line segment of the "Y" shape -/
def horizontalSegment : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 11 ∧ p.1 < 2}

/-- The diagonal ray of the "Y" shape -/
def diagonalRay : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 + 9 ∧ p.1 > 2}

theorem T_is_Y_shape :
  T = verticalSegment ∪ horizontalSegment ∪ diagonalRay ∧
  commonPoint ∈ T ∧
  commonPoint ∈ verticalSegment ∧
  commonPoint ∈ horizontalSegment ∧
  commonPoint ∈ diagonalRay :=
sorry

end NUMINAMATH_CALUDE_T_is_Y_shape_l1220_122089


namespace NUMINAMATH_CALUDE_cos_2alpha_special_value_l1220_122028

theorem cos_2alpha_special_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.sin (α - π/4) = 1/3) : 
  Real.cos (2*α) = -4*Real.sqrt 2/9 := by
sorry

end NUMINAMATH_CALUDE_cos_2alpha_special_value_l1220_122028


namespace NUMINAMATH_CALUDE_gcd_85_100_l1220_122076

theorem gcd_85_100 : Nat.gcd 85 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_85_100_l1220_122076


namespace NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l1220_122031

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l1220_122031


namespace NUMINAMATH_CALUDE_cardinality_of_star_product_l1220_122082

def P : Finset ℕ := {3, 4, 5}
def Q : Finset ℕ := {4, 5, 6, 7}

def star_product (P Q : Finset ℕ) : Finset (ℕ × ℕ) :=
  Finset.product P Q

theorem cardinality_of_star_product :
  Finset.card (star_product P Q) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cardinality_of_star_product_l1220_122082


namespace NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l1220_122048

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_8_terms :
  let a : ℚ := 1/2
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a r n = 9840/6561 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_first_8_terms_l1220_122048


namespace NUMINAMATH_CALUDE_collinear_dots_probability_l1220_122010

/-- The number of dots in each row and column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots to be chosen -/
def chosenDots : ℕ := 4

/-- The number of ways to choose 4 collinear dots from horizontal or vertical lines -/
def horizontalVerticalSets : ℕ := 2 * gridSize

/-- The number of ways to choose 4 collinear dots from diagonal lines -/
def diagonalSets : ℕ := 2 * (Nat.choose gridSize chosenDots)

/-- The total number of ways to choose 4 collinear dots -/
def totalCollinearSets : ℕ := horizontalVerticalSets + diagonalSets

/-- Theorem: The probability of selecting 4 collinear dots from a 5x5 grid 
    when choosing 4 dots at random is 4/2530 -/
theorem collinear_dots_probability : 
  (totalCollinearSets : ℚ) / (Nat.choose totalDots chosenDots) = 4 / 2530 := by
  sorry

end NUMINAMATH_CALUDE_collinear_dots_probability_l1220_122010


namespace NUMINAMATH_CALUDE_xy_max_value_l1220_122081

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 2) :
  xy ≤ (1 : ℝ) / 2 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 2 ∧ x₀ * y₀ = (1 : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_xy_max_value_l1220_122081


namespace NUMINAMATH_CALUDE_lawn_mowing_difference_l1220_122067

/-- The difference between spring and summer lawn mowing counts -/
theorem lawn_mowing_difference (spring_count summer_count : ℕ) 
  (h1 : spring_count = 8) 
  (h2 : summer_count = 5) : 
  spring_count - summer_count = 3 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_difference_l1220_122067


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1220_122068

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 1 → x^2 + x - 2 > 0) ∧
  (∃ x, x^2 + x - 2 > 0 ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1220_122068


namespace NUMINAMATH_CALUDE_f_monotonic_range_l1220_122072

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 3

-- Define the property of being monotonic on an interval
def IsMonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ f x > f y)

-- Theorem statement
theorem f_monotonic_range (m : ℝ) :
  IsMonotonicOn f m (m + 4) → m ∈ Set.Iic (-5) ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_f_monotonic_range_l1220_122072


namespace NUMINAMATH_CALUDE_expression_value_l1220_122097

theorem expression_value : 
  (7 - (540 : ℚ) / 9) - (5 - (330 : ℚ) * 2 / 11) + (2 - (260 : ℚ) * 3 / 13) = -56 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1220_122097


namespace NUMINAMATH_CALUDE_log_one_half_decreasing_l1220_122084

-- Define the logarithm function with base a
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our specific function f(x) = log_(1/2)(x)
noncomputable def f (x : ℝ) : ℝ := log (1/2) x

-- State the theorem
theorem log_one_half_decreasing :
  0 < (1/2 : ℝ) ∧ (1/2 : ℝ) < 1 →
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x < y → f y < f x :=
sorry

end NUMINAMATH_CALUDE_log_one_half_decreasing_l1220_122084


namespace NUMINAMATH_CALUDE_range_of_a_l1220_122063

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x) →
  (∃ x : ℝ, x^2 + 4*x + a = 0) →
  a ∈ Set.Icc (Real.exp 1) 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1220_122063


namespace NUMINAMATH_CALUDE_triangle_theorem_l1220_122024

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) :
  (t.A = 2 * t.B → t.C = 5 * Real.pi / 8) ∧ 
  (2 * t.a^2 = t.b^2 + t.c^2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1220_122024


namespace NUMINAMATH_CALUDE_math_club_minimum_size_l1220_122074

theorem math_club_minimum_size :
  ∀ (boys girls : ℕ),
  (boys : ℝ) / (boys + girls : ℝ) > 0.6 →
  girls = 5 →
  boys + girls ≥ 13 ∧
  ∀ (total : ℕ), total < 13 →
    ¬(∃ (b g : ℕ), b + g = total ∧ (b : ℝ) / (total : ℝ) > 0.6 ∧ g = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_math_club_minimum_size_l1220_122074


namespace NUMINAMATH_CALUDE_function_increasing_decreasing_implies_m_range_l1220_122059

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

-- State the theorem
theorem function_increasing_decreasing_implies_m_range :
  ∀ m : ℝ, 
  (∀ x ≥ 2, ∀ y ≥ 2, x < y → f m x < f m y) ∧ 
  (∀ x ≤ 1, ∀ y ≤ 1, x < y → f m x > f m y) →
  8 ≤ m ∧ m ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_function_increasing_decreasing_implies_m_range_l1220_122059


namespace NUMINAMATH_CALUDE_complement_union_problem_l1220_122017

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 2}
def B : Set Int := {-2, 2}

theorem complement_union_problem : (U \ A) ∪ B = {-2, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_problem_l1220_122017


namespace NUMINAMATH_CALUDE_radio_profit_percentage_is_approximately_6_8_percent_l1220_122026

/-- Calculates the profit percentage for a radio sale given the following parameters:
    * initial_cost: The initial cost of the radio
    * overhead: Overhead expenses
    * purchase_tax_rate: Purchase tax rate
    * luxury_tax_rate: Luxury tax rate
    * exchange_discount_rate: Exchange offer discount rate
    * sales_tax_rate: Sales tax rate
    * selling_price: Final selling price
-/
def calculate_profit_percentage (
  initial_cost : ℝ
  ) (overhead : ℝ
  ) (purchase_tax_rate : ℝ
  ) (luxury_tax_rate : ℝ
  ) (exchange_discount_rate : ℝ
  ) (sales_tax_rate : ℝ
  ) (selling_price : ℝ
  ) : ℝ :=
  sorry

/-- The profit percentage for the radio sale is approximately 6.8% -/
theorem radio_profit_percentage_is_approximately_6_8_percent :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |calculate_profit_percentage 225 28 0.08 0.05 0.10 0.12 300 - 6.8| < ε :=
sorry

end NUMINAMATH_CALUDE_radio_profit_percentage_is_approximately_6_8_percent_l1220_122026


namespace NUMINAMATH_CALUDE_triangle_properties_l1220_122065

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ) -- angles
  (a b c : ℝ) -- opposite sides

-- Define the conditions and theorems
theorem triangle_properties (t : Triangle) 
  (h1 : t.b = Real.sqrt 7)
  (h2 : Real.sin t.A = Real.sqrt 3 * Real.sin t.C) :
  (t.B = π / 6 → Real.sin t.B = Real.sin t.C) ∧ 
  (t.B > π / 2 ∧ Real.cos (2 * t.B) = 1 / 2 → 
    t.a * Real.sin t.C = Real.sqrt 21 / 14) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_triangle_properties_l1220_122065


namespace NUMINAMATH_CALUDE_pauls_vertical_distance_l1220_122012

/-- The number of feet Paul travels vertically in a week -/
def vertical_distance_per_week (story : ℕ) (trips_per_day : ℕ) (days_per_week : ℕ) (feet_per_story : ℕ) : ℕ :=
  2 * story * trips_per_day * days_per_week * feet_per_story

/-- Theorem stating the total vertical distance Paul travels in a week -/
theorem pauls_vertical_distance :
  vertical_distance_per_week 5 3 7 10 = 2100 := by
  sorry

#eval vertical_distance_per_week 5 3 7 10

end NUMINAMATH_CALUDE_pauls_vertical_distance_l1220_122012


namespace NUMINAMATH_CALUDE_chicken_multiple_l1220_122005

theorem chicken_multiple (total chickens : ℕ) (colten_chickens : ℕ) (m : ℕ) : 
  total = 383 →
  colten_chickens = 37 →
  (∃ (quentin skylar : ℕ), 
    quentin + skylar + colten_chickens = total ∧
    quentin = 2 * skylar + 25 ∧
    skylar = m * colten_chickens - 4) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_chicken_multiple_l1220_122005


namespace NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l1220_122075

def cannot_form_triangle (a b c : ℝ) : Prop :=
  a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

theorem min_cut_length_for_non_triangle : ∃ (x : ℝ),
  (x > 0) ∧
  (cannot_form_triangle (9 - x) (12 - x) (15 - x)) ∧
  (∀ y, 0 < y ∧ y < x → ¬(cannot_form_triangle (9 - y) (12 - y) (15 - y))) ∧
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l1220_122075


namespace NUMINAMATH_CALUDE_min_cuts_for_cube_division_l1220_122098

/-- Represents a three-dimensional cube --/
structure Cube where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the process of cutting a cube --/
def cut_cube (initial : Cube) (final_size : ℕ) (allow_rearrange : Bool) : ℕ :=
  sorry

/-- Theorem: The minimum number of cuts to divide a 3x3x3 cube into 27 1x1x1 cubes is 6 --/
theorem min_cuts_for_cube_division :
  let initial_cube : Cube := ⟨3, 3, 3⟩
  let final_size : ℕ := 1
  let num_final_cubes : ℕ := 27
  let allow_rearrange : Bool := true
  (cut_cube initial_cube final_size allow_rearrange = 6) ∧
  (∀ n : ℕ, n < 6 → cut_cube initial_cube final_size allow_rearrange ≠ n) :=
by sorry

end NUMINAMATH_CALUDE_min_cuts_for_cube_division_l1220_122098


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1220_122025

theorem quadratic_inequality_equivalence (a : ℝ) : 
  (∀ x : ℝ, x^2 - x - 2 < 0 ↔ -2 < x ∧ x < a) ↔ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1220_122025


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l1220_122015

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem f_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l1220_122015


namespace NUMINAMATH_CALUDE_age_problem_l1220_122014

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 42 →
  b = 16 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1220_122014


namespace NUMINAMATH_CALUDE_square_circle_union_area_l1220_122057

/-- The area of the union of a square with side length 12 and a circle with radius 6 
    centered at the center of the square is equal to 144. -/
theorem square_circle_union_area : 
  let square_side : ℝ := 12
  let circle_radius : ℝ := 6
  let square_area := square_side ^ 2
  let circle_area := π * circle_radius ^ 2
  square_area = circle_area + 144 := by
  sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l1220_122057


namespace NUMINAMATH_CALUDE_project_hours_difference_l1220_122045

theorem project_hours_difference (total_hours : ℕ) (kate_hours : ℕ) : 
  total_hours = 117 →
  2 * kate_hours + kate_hours + 6 * kate_hours = total_hours →
  6 * kate_hours - kate_hours = 65 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l1220_122045


namespace NUMINAMATH_CALUDE_cube_root_fraction_equality_l1220_122087

theorem cube_root_fraction_equality : 
  (((5 : ℝ) / 6 * 20.25) ^ (1/3 : ℝ)) = (3 * (5 ^ (2/3 : ℝ))) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_fraction_equality_l1220_122087


namespace NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_174_l1220_122062

-- Define the function f
def f (a : ℝ) : ℝ := 3 * a - 6

-- Define the function F
def F (a b : ℝ) : ℝ := 2 * b^2 + 3 * a

-- Theorem statement
theorem F_of_4_f_of_5_equals_174 : F 4 (f 5) = 174 := by
  sorry

end NUMINAMATH_CALUDE_F_of_4_f_of_5_equals_174_l1220_122062


namespace NUMINAMATH_CALUDE_certain_number_problem_l1220_122073

theorem certain_number_problem : ∃ x : ℝ, 0.12 * x - 0.1 * 14.2 = 1.484 ∧ x = 24.2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1220_122073


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l1220_122047

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := (x - 6)^2 + y^2 = 64

-- Define the point P
def P : ℝ × ℝ := (2, 0)

-- Define the point Q (center of the given circle)
def Q : ℝ × ℝ := (6, 0)

-- Define a circle passing through P and tangent to the given circle
def passingCircle (a b r : ℝ) : Prop :=
  (a - P.1)^2 + (b - P.2)^2 = r^2 ∧
  ∃ (x y : ℝ), givenCircle x y ∧ (a - x)^2 + (b - y)^2 = r^2 ∧
  (a - Q.1)^2 + (b - Q.2)^2 = (8 - r)^2

-- Define the locus of centers
def locus (a b : ℝ) : Prop :=
  ∃ (r : ℝ), passingCircle a b r

-- Theorem statement
theorem locus_is_ellipse :
  ∀ (a b : ℝ), locus a b ↔ 
    (a - P.1)^2 + (b - P.2)^2 + (a - Q.1)^2 + (b - Q.2)^2 = 8^2 :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l1220_122047


namespace NUMINAMATH_CALUDE_infinitely_many_all_off_infinitely_many_never_all_off_l1220_122016

-- Define the lamp state as a list of booleans
def LampState := List Bool

-- Define the state modification function
def modifyState (state : LampState) : LampState :=
  sorry

-- Define the initial state
def initialState (n : Nat) : LampState :=
  sorry

-- Define a predicate to check if all lamps are off
def allLampsOff (state : LampState) : Prop :=
  sorry

-- Define a function to evolve the state
def evolveState (n : Nat) (steps : Nat) : LampState :=
  sorry

theorem infinitely_many_all_off :
  ∃ S : Set Nat, (∀ n ∈ S, n ≥ 2) ∧ Set.Infinite S ∧
  ∀ n ∈ S, ∃ k : Nat, allLampsOff (evolveState n k) :=
sorry

theorem infinitely_many_never_all_off :
  ∃ T : Set Nat, (∀ n ∈ T, n ≥ 2) ∧ Set.Infinite T ∧
  ∀ n ∈ T, ∀ k : Nat, ¬(allLampsOff (evolveState n k)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_all_off_infinitely_many_never_all_off_l1220_122016


namespace NUMINAMATH_CALUDE_cubic_fraction_sum_l1220_122053

theorem cubic_fraction_sum (a b : ℝ) (h1 : |a| ≠ |b|) 
  (h2 : (a + b) / (a - b) + (a - b) / (a + b) = 6) :
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_sum_l1220_122053


namespace NUMINAMATH_CALUDE_book_sale_discount_l1220_122002

/-- Calculates the discount percentage for a book sale --/
theorem book_sale_discount (cost : ℝ) (markup_percent : ℝ) (profit_percent : ℝ) 
  (h_cost : cost = 50)
  (h_markup : markup_percent = 30)
  (h_profit : profit_percent = 17) : 
  let marked_price := cost * (1 + markup_percent / 100)
  let selling_price := cost * (1 + profit_percent / 100)
  let discount := marked_price - selling_price
  (discount / marked_price) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_discount_l1220_122002


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l1220_122009

/-- The polynomial Q(x) -/
def Q (b : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + b*x + 5

/-- Theorem: If x - 5 is a factor of Q(x), then b = -41 -/
theorem factor_implies_b_value (b : ℝ) : 
  (∀ x, Q b x = 0 ↔ x = 5) → b = -41 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l1220_122009


namespace NUMINAMATH_CALUDE_fraction_value_l1220_122070

theorem fraction_value (a b : ℝ) (h : a + 1/b = 2/a + 2*b ∧ a + 1/b ≠ 0) : a/b = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1220_122070


namespace NUMINAMATH_CALUDE_sum_floor_equality_l1220_122093

theorem sum_floor_equality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + b^2 = 2008) (h2 : c^2 + d^2 = 2008) (h3 : a * c = 1000) (h4 : b * d = 1000) :
  ⌊a + b + c + d⌋ = 126 := by
  sorry

end NUMINAMATH_CALUDE_sum_floor_equality_l1220_122093


namespace NUMINAMATH_CALUDE_factorization_proof_l1220_122051

theorem factorization_proof (x : ℝ) : -2 * x^2 + 18 = -2 * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1220_122051


namespace NUMINAMATH_CALUDE_symmetry_implies_line_equation_l1220_122000

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y + 4 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 2 = 0

-- Define symmetry with respect to a line
def symmetric_wrt_line (circle1 circle2 line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x2 y2 ∧ 
  (∃ (x y : ℝ), line x y ∧ 
    ((x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2) ∧
    ((x1 + x2) / 2 = x) ∧ ((y1 + y2) / 2 = y))

-- Theorem statement
theorem symmetry_implies_line_equation : 
  symmetric_wrt_line circle_O circle_C line_l :=
sorry

end NUMINAMATH_CALUDE_symmetry_implies_line_equation_l1220_122000


namespace NUMINAMATH_CALUDE_right_triangle_cone_rotation_l1220_122091

/-- Given a right triangle with legs a and b, if rotating about leg a produces a cone
    with volume 800π cm³ and rotating about leg b produces a cone with volume 1920π cm³,
    then the hypotenuse length is 26 cm. -/
theorem right_triangle_cone_rotation (a b : ℝ) :
  a > 0 ∧ b > 0 →
  (1 / 3 : ℝ) * Real.pi * a * b^2 = 800 * Real.pi →
  (1 / 3 : ℝ) * Real.pi * b * a^2 = 1920 * Real.pi →
  Real.sqrt (a^2 + b^2) = 26 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_cone_rotation_l1220_122091


namespace NUMINAMATH_CALUDE_apples_per_pie_l1220_122060

/-- Given a box of apples, calculate the weight of apples needed per pie -/
theorem apples_per_pie (total_weight : ℝ) (num_pies : ℕ) : 
  total_weight = 120 → num_pies = 15 → (total_weight / 2) / num_pies = 4 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l1220_122060


namespace NUMINAMATH_CALUDE_lcm_812_smallest_lcm_812_24_smallest_lcm_812_672_l1220_122094

theorem lcm_812_smallest (n : ℕ) : n > 0 ∧ Nat.lcm 812 n = 672 → n ≥ 24 := by
  sorry

theorem lcm_812_24 : Nat.lcm 812 24 = 672 := by
  sorry

theorem smallest_lcm_812_672 : ∃ (n : ℕ), n > 0 ∧ Nat.lcm 812 n = 672 ∧ ∀ (m : ℕ), m > 0 → Nat.lcm 812 m = 672 → m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_lcm_812_smallest_lcm_812_24_smallest_lcm_812_672_l1220_122094


namespace NUMINAMATH_CALUDE_student_fraction_mistake_l1220_122037

theorem student_fraction_mistake (n : ℚ) (correct_fraction : ℚ) (student_fraction : ℚ) :
  n = 288 →
  correct_fraction = 5 / 16 →
  student_fraction * n = correct_fraction * n + 150 →
  student_fraction = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_student_fraction_mistake_l1220_122037


namespace NUMINAMATH_CALUDE_triangle_properties_l1220_122043

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 1 ∧ Real.cos t.C + (2 * t.a + t.c) * Real.cos t.B = 0

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.B = 2 * Real.pi / 3 ∧ 
  (∀ (s : ℝ), s = 1/2 * t.a * t.c * Real.sin t.B → s ≤ Real.sqrt 3 / 12) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l1220_122043


namespace NUMINAMATH_CALUDE_least_α_is_correct_l1220_122039

/-- An isosceles triangle with two equal angles α° and a third angle β° -/
structure IsoscelesTriangle where
  α : ℕ
  β : ℕ
  is_isosceles : α + α + β = 180
  α_prime : Nat.Prime α
  β_prime : Nat.Prime β
  α_ne_β : α ≠ β

/-- The least possible value of α in an isosceles triangle where α and β are distinct primes -/
def least_α : ℕ := 41

theorem least_α_is_correct (t : IsoscelesTriangle) : t.α ≥ least_α := by
  sorry

end NUMINAMATH_CALUDE_least_α_is_correct_l1220_122039


namespace NUMINAMATH_CALUDE_simplify_fraction_l1220_122083

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (75 * b^3) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1220_122083


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l1220_122080

/-- A function that checks if three numbers form a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- Theorem stating that (9, 12, 15) is the only Pythagorean triple among the given options -/
theorem pythagorean_triple_identification :
  (¬ is_pythagorean_triple 3 4 5) ∧
  (¬ is_pythagorean_triple 3 4 7) ∧
  (is_pythagorean_triple 9 12 15) :=
sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l1220_122080


namespace NUMINAMATH_CALUDE_james_initial_milk_l1220_122096

def ounces_drank : ℕ := 13
def ounces_per_gallon : ℕ := 128
def ounces_left : ℕ := 371

def initial_gallons : ℚ :=
  (ounces_left + ounces_drank) / ounces_per_gallon

theorem james_initial_milk : initial_gallons = 3 := by
  sorry

end NUMINAMATH_CALUDE_james_initial_milk_l1220_122096


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1220_122090

theorem quadratic_equation_result (x : ℝ) : 
  2 * x^2 - 5 = 11 → 
  (4 * x^2 + 4 * x + 1 = 33 + 8 * Real.sqrt 2) ∨ 
  (4 * x^2 + 4 * x + 1 = 33 - 8 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1220_122090


namespace NUMINAMATH_CALUDE_ellipse_transformation_l1220_122003

/-- Given an ellipse with equation x²/6 + y² = 1, prove that compressing
    the x-coordinates to 1/2 of their original value and stretching the
    y-coordinates to twice their original value results in a curve with
    equation 2x²/3 + y²/4 = 1. -/
theorem ellipse_transformation (x y : ℝ) :
  (x^2 / 6 + y^2 = 1) →
  (∃ x' y' : ℝ, x' = x / 2 ∧ y' = 2 * y ∧ 2 * x'^2 / 3 + y'^2 / 4 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_transformation_l1220_122003


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1220_122029

theorem arithmetic_calculation : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1220_122029


namespace NUMINAMATH_CALUDE_power_equation_solution_l1220_122020

theorem power_equation_solution : ∃ x : ℕ, (5 ^ 5) * (9 ^ 3) = 3 * (15 ^ x) ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1220_122020


namespace NUMINAMATH_CALUDE_inequality_proof_l1220_122038

theorem inequality_proof (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 0) :
  a^2 > a*b ∧ a*b > a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1220_122038


namespace NUMINAMATH_CALUDE_complex_sum_equals_two_l1220_122054

theorem complex_sum_equals_two (z : ℂ) (h : z^7 = 1) (h2 : z = Complex.exp (2 * Real.pi * Complex.I / 7)) : 
  (z^2 / (1 + z^3)) + (z^4 / (1 + z^6)) + (z^6 / (1 + z^9)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_two_l1220_122054


namespace NUMINAMATH_CALUDE_belle_treats_cost_l1220_122019

/-- The cost of feeding Belle her treats for a week -/
def cost_per_week : ℚ :=
  let biscuits_per_day : ℕ := 4
  let bones_per_day : ℕ := 2
  let biscuit_cost : ℚ := 1/4
  let bone_cost : ℚ := 1
  let days_per_week : ℕ := 7
  (biscuits_per_day * biscuit_cost + bones_per_day * bone_cost) * days_per_week

/-- Theorem stating that the cost of feeding Belle her treats for a week is $21 -/
theorem belle_treats_cost : cost_per_week = 21 := by
  sorry

end NUMINAMATH_CALUDE_belle_treats_cost_l1220_122019


namespace NUMINAMATH_CALUDE_N_is_composite_l1220_122064

/-- The number formed by k+1 ones and k zeros in between -/
def N (k : ℕ) : ℕ := 10^(k+1) + 1

/-- Theorem stating that N(k) is composite for k > 1 -/
theorem N_is_composite (k : ℕ) (h : k > 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N k = a * b :=
sorry

end NUMINAMATH_CALUDE_N_is_composite_l1220_122064


namespace NUMINAMATH_CALUDE_remainder_theorem_l1220_122034

theorem remainder_theorem (k : ℕ) 
  (h1 : k > 0) 
  (h2 : k < 39) 
  (h3 : k % 5 = 2) 
  (h4 : k % 6 = 5) : 
  k % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1220_122034


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1220_122033

theorem negation_of_proposition :
  (¬(∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0)) ↔
  (∀ a b : ℝ, a^2 + b^2 = 0 → a ≠ 0 ∨ b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1220_122033


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1220_122027

theorem quadratic_factorization (c d : ℤ) : 
  (∀ x, 25 * x^2 - 160 * x - 144 = (5 * x + c) * (5 * x + d)) → 
  c + 2 * d = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1220_122027


namespace NUMINAMATH_CALUDE_triangle_ratio_equals_two_l1220_122013

noncomputable def triangle_ratio (A B C : ℝ) (a b c : ℝ) : ℝ :=
  (a + b - c) / (Real.sin A + Real.sin B - Real.sin C)

theorem triangle_ratio_equals_two (A B C : ℝ) (a b c : ℝ) 
  (h1 : A = π / 3)  -- 60° in radians
  (h2 : a = Real.sqrt 3) :
  triangle_ratio A B C a b c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_equals_two_l1220_122013
