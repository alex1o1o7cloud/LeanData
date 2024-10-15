import Mathlib

namespace NUMINAMATH_CALUDE_soccer_game_water_consumption_l3978_397803

/-- Proves that the number of water bottles consumed is 4, given the initial quantities,
    remaining bottles, and the relationship between water and soda consumption. -/
theorem soccer_game_water_consumption
  (initial_water : Nat)
  (initial_soda : Nat)
  (remaining_bottles : Nat)
  (h1 : initial_water = 12)
  (h2 : initial_soda = 34)
  (h3 : remaining_bottles = 30)
  (h4 : ∀ w s, w + s = initial_water + initial_soda - remaining_bottles → s = 3 * w) :
  initial_water + initial_soda - remaining_bottles - 3 * (initial_water + initial_soda - remaining_bottles) / 4 = 4 :=
by sorry

end NUMINAMATH_CALUDE_soccer_game_water_consumption_l3978_397803


namespace NUMINAMATH_CALUDE_overlap_difference_l3978_397857

def total_students : ℕ := 232
def geometry_students : ℕ := 144
def biology_students : ℕ := 119

theorem overlap_difference :
  (min geometry_students biology_students) - 
  (geometry_students + biology_students - total_students) = 88 :=
by sorry

end NUMINAMATH_CALUDE_overlap_difference_l3978_397857


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3978_397855

theorem inequality_equivalence (x : ℝ) : 
  (2 < (x - 1)⁻¹ ∧ (x - 1)⁻¹ < 3 ∧ x ≠ 1) ↔ (4/3 < x ∧ x < 3/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3978_397855


namespace NUMINAMATH_CALUDE_circle_intersection_radius_range_l3978_397866

/-- The range of r values for which there are two points P satisfying the given conditions -/
theorem circle_intersection_radius_range :
  ∀ (r : ℝ),
  (∃ (P₁ P₂ : ℝ × ℝ),
    P₁ ≠ P₂ ∧
    ((P₁.1 - 3)^2 + (P₁.2 - 4)^2 = r^2) ∧
    ((P₂.1 - 3)^2 + (P₂.2 - 4)^2 = r^2) ∧
    ((P₁.1 + 2)^2 + P₁.2^2 + (P₁.1 - 2)^2 + P₁.2^2 = 40) ∧
    ((P₂.1 + 2)^2 + P₂.2^2 + (P₂.1 - 2)^2 + P₂.2^2 = 40)) ↔
  (1 < r ∧ r < 9) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_radius_range_l3978_397866


namespace NUMINAMATH_CALUDE_M_subset_N_l3978_397821

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | x ≤ 1}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l3978_397821


namespace NUMINAMATH_CALUDE_function_range_properties_l3978_397894

open Set

/-- Given a function f with maximum M and minimum m on [a, b], prove the following statements -/
theorem function_range_properties
  (f : ℝ → ℝ) (a b M m : ℝ) (h_max : ∀ x ∈ Icc a b, f x ≤ M) (h_min : ∀ x ∈ Icc a b, m ≤ f x) :
  (∀ p, (∀ x ∈ Icc a b, p ≤ f x) → p ∈ Iic m) ∧
  (∀ p, (∃ x ∈ Icc a b, p = f x) → p ∈ Icc m M) ∧
  (∀ p, (∃ x ∈ Icc a b, p ≤ f x) → p ∈ Iic M) :=
by sorry


end NUMINAMATH_CALUDE_function_range_properties_l3978_397894


namespace NUMINAMATH_CALUDE_function_inequality_l3978_397882

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_critical : deriv f 1 = 0) 
  (h_condition : ∀ x : ℝ, (x - 1) * (deriv f x) > 0) : 
  f 0 + f 2 > 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3978_397882


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l3978_397865

theorem trigonometric_inequality : 
  let a := 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180)
  let b := (2 * Real.tan (76 * π / 180)) / (1 + Real.tan (76 * π / 180) ^ 2)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l3978_397865


namespace NUMINAMATH_CALUDE_stating_at_least_two_different_selections_l3978_397844

/-- The number of available courses -/
def num_courses : ℕ := 6

/-- The number of courses each student must choose -/
def courses_per_student : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- 
Theorem stating that the number of ways in which at least two out of three students 
can select different combinations of 2 courses from a set of 6 courses is equal to 2520.
-/
theorem at_least_two_different_selections : 
  (choose num_courses courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student * 
   choose num_courses courses_per_student) * num_students - 
  (choose num_courses courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student) * num_students + 
  (choose num_courses courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student * 
   choose (num_courses - 2 * courses_per_student) courses_per_student) = 2520 :=
by sorry

end NUMINAMATH_CALUDE_stating_at_least_two_different_selections_l3978_397844


namespace NUMINAMATH_CALUDE_linear_equation_exponent_l3978_397815

theorem linear_equation_exponent (k : ℝ) : 
  (∀ x, ∃ a b : ℝ, x^(2*k - 1) + 2 = a*x + b) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_exponent_l3978_397815


namespace NUMINAMATH_CALUDE_geometric_sum_eight_terms_l3978_397819

theorem geometric_sum_eight_terms : 
  let a : ℕ := 2
  let r : ℕ := 2
  let n : ℕ := 8
  a * (r^n - 1) / (r - 1) = 510 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_eight_terms_l3978_397819


namespace NUMINAMATH_CALUDE_circumcenter_coordinates_l3978_397818

/-- A quadrilateral in 2D space -/
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The circumcenter of a quadrilateral -/
def circumcenter (q : Quadrilateral) : ℝ × ℝ := sorry

/-- A quadrilateral is inscribed in a circle if its circumcenter exists -/
def isInscribed (q : Quadrilateral) : Prop :=
  ∃ c : ℝ × ℝ, c = circumcenter q

theorem circumcenter_coordinates (q : Quadrilateral) (h : isInscribed q) :
  circumcenter q = (6, 1) := by sorry

end NUMINAMATH_CALUDE_circumcenter_coordinates_l3978_397818


namespace NUMINAMATH_CALUDE_mrs_hilt_total_distance_l3978_397851

/-- Calculate the total distance walked by Mrs. Hilt -/
def total_distance (
  water_fountain_dist : ℝ)
  (main_office_dist : ℝ)
  (teachers_lounge_dist : ℝ)
  (water_fountain_increase : ℝ)
  (main_office_increase : ℝ)
  (teachers_lounge_increase : ℝ)
  (water_fountain_visits : ℕ)
  (main_office_visits : ℕ)
  (teachers_lounge_visits : ℕ) : ℝ :=
  let water_fountain_return := water_fountain_dist * (1 + water_fountain_increase)
  let main_office_return := main_office_dist * (1 + main_office_increase)
  let teachers_lounge_return := teachers_lounge_dist * (1 + teachers_lounge_increase)
  (water_fountain_dist + water_fountain_return) * water_fountain_visits +
  (main_office_dist + main_office_return) * main_office_visits +
  (teachers_lounge_dist + teachers_lounge_return) * teachers_lounge_visits

/-- Theorem stating that Mrs. Hilt's total walking distance is 699 feet -/
theorem mrs_hilt_total_distance :
  total_distance 30 50 35 0.15 0.10 0.20 4 2 3 = 699 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_total_distance_l3978_397851


namespace NUMINAMATH_CALUDE_slope_equals_one_implies_m_equals_one_l3978_397877

/-- Given two points M(-2, m) and N(m, 4), if the slope of the line passing through M and N
    is equal to 1, then m = 1. -/
theorem slope_equals_one_implies_m_equals_one (m : ℝ) : 
  (4 - m) / (m - (-2)) = 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_slope_equals_one_implies_m_equals_one_l3978_397877


namespace NUMINAMATH_CALUDE_complex_modulus_one_l3978_397807

theorem complex_modulus_one (z : ℂ) (h : 3 * z^6 + 2 * Complex.I * z^5 - 2 * z - 3 * Complex.I = 0) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l3978_397807


namespace NUMINAMATH_CALUDE_sector_angle_measure_l3978_397872

/-- Given a circular sector with arc length and area both equal to 5,
    prove that the radian measure of its central angle is 5/2 -/
theorem sector_angle_measure (r : ℝ) (α : ℝ) 
    (h1 : α * r = 5)  -- arc length formula
    (h2 : (1/2) * α * r^2 = 5)  -- sector area formula
    : α = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_measure_l3978_397872


namespace NUMINAMATH_CALUDE_encoded_CDE_value_l3978_397892

/-- Represents the digits in the base 7 encoding system -/
inductive Digit
  | A | B | C | D | E | F | G

/-- Represents a number in the base 7 encoding system -/
def EncodedNumber := List Digit

/-- Converts an EncodedNumber to its base 10 representation -/
def to_base_10 : EncodedNumber → ℕ := sorry

/-- Checks if two EncodedNumbers are consecutive -/
def are_consecutive (a b : EncodedNumber) : Prop := sorry

/-- The main theorem -/
theorem encoded_CDE_value :
  ∃ (bcg bcf bad : EncodedNumber),
    (are_consecutive bcg bcf) ∧
    (are_consecutive bcf bad) ∧
    bcg = [Digit.B, Digit.C, Digit.G] ∧
    bcf = [Digit.B, Digit.C, Digit.F] ∧
    bad = [Digit.B, Digit.A, Digit.D] →
    to_base_10 [Digit.C, Digit.D, Digit.E] = 329 := by
  sorry

end NUMINAMATH_CALUDE_encoded_CDE_value_l3978_397892


namespace NUMINAMATH_CALUDE_north_movement_representation_l3978_397826

/-- Represents the direction of movement -/
inductive Direction
  | North
  | South

/-- Represents a movement with distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Converts a movement to its numerical representation -/
def movementToMeters (m : Movement) : ℝ :=
  match m.direction with
  | Direction.North => m.distance
  | Direction.South => -m.distance

theorem north_movement_representation (d : ℝ) (h : d > 0) :
  let southMovement : Movement := ⟨d, Direction.South⟩
  let northMovement : Movement := ⟨d, Direction.North⟩
  movementToMeters southMovement = -d →
  movementToMeters northMovement = d :=
by sorry

end NUMINAMATH_CALUDE_north_movement_representation_l3978_397826


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3978_397899

theorem trigonometric_equation_solution (x : ℝ) : 
  3 - 7 * (Real.cos x)^2 * Real.sin x - 3 * (Real.sin x)^3 = 0 ↔ 
  (∃ k : ℤ, x = π / 2 + 2 * k * π) ∨ 
  (∃ k : ℤ, x = (-1)^k * π / 6 + k * π) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3978_397899


namespace NUMINAMATH_CALUDE_max_triangle_area_in_ellipse_l3978_397833

/-- The maximum area of a triangle inscribed in an ellipse with semi-axes a and b -/
theorem max_triangle_area_in_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (A : ℝ), A = a * b * (3 * Real.sqrt 3) / 4 ∧
  ∀ (triangle_area : ℝ), triangle_area ≤ A :=
sorry

end NUMINAMATH_CALUDE_max_triangle_area_in_ellipse_l3978_397833


namespace NUMINAMATH_CALUDE_sock_ratio_l3978_397861

/-- The ratio of black socks to blue socks in Mr. Lin's original order --/
theorem sock_ratio : 
  ∀ (x y : ℕ), 
  (x = 4) → -- Mr. Lin ordered 4 pairs of black socks
  ((2 * x + y) * 3 = (2 * y + x) * 2) → -- 50% increase when swapped
  (y = 4 * x) -- Ratio of black to blue is 1:4
  := by sorry

end NUMINAMATH_CALUDE_sock_ratio_l3978_397861


namespace NUMINAMATH_CALUDE_salary_calculation_l3978_397811

/-- Represents the number of turbans given as part of the yearly salary -/
def turbans_per_year : ℕ := sorry

/-- The price of a turban in rupees -/
def turban_price : ℕ := 110

/-- The base salary in rupees for a full year -/
def base_salary : ℕ := 90

/-- The amount in rupees received by the servant after 9 months -/
def received_amount : ℕ := 40

/-- The number of months the servant worked -/
def months_worked : ℕ := 9

/-- The total number of months in a year -/
def months_in_year : ℕ := 12

theorem salary_calculation :
  (months_worked : ℚ) / months_in_year * (base_salary + turbans_per_year * turban_price) =
  received_amount + turban_price ∧ turbans_per_year = 1 := by sorry

end NUMINAMATH_CALUDE_salary_calculation_l3978_397811


namespace NUMINAMATH_CALUDE_polynomial_equality_l3978_397873

theorem polynomial_equality (d e c : ℝ) : 
  (∀ x : ℝ, (6 * x^2 - 5 * x + 10/3) * (d * x^2 + e * x + c) = 
    18 * x^4 - 5 * x^3 + 15 * x^2 - (50/3) * x + 45/3) → 
  c = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3978_397873


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3978_397806

theorem line_passes_through_fixed_point 
  (a b c : ℝ) 
  (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_sum : 1/a + 1/b = 1/c) : 
  ∃ (x y : ℝ), x/a + y/b = 1 ∧ x = c ∧ y = c :=
by sorry


end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3978_397806


namespace NUMINAMATH_CALUDE_compound_composition_l3978_397862

/-- Atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Carbon atoms in the compound -/
def num_C : ℕ := 6

/-- Number of Oxygen atoms in the compound -/
def num_O : ℕ := 7

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 192

/-- Calculates the number of Hydrogen atoms in the compound -/
def num_H : ℕ := 8

theorem compound_composition :
  (num_C : ℝ) * atomic_weight_C + (num_O : ℝ) * atomic_weight_O + (num_H : ℝ) * atomic_weight_H = molecular_weight := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l3978_397862


namespace NUMINAMATH_CALUDE_consecutive_product_111222_l3978_397817

theorem consecutive_product_111222 (b : ℕ) :
  b * (b + 1) = 111222 → b = 333 := by sorry

end NUMINAMATH_CALUDE_consecutive_product_111222_l3978_397817


namespace NUMINAMATH_CALUDE_defective_pen_count_l3978_397854

theorem defective_pen_count (total_pens : ℕ) (prob_non_defective : ℚ) : 
  total_pens = 12 →
  prob_non_defective = 6/11 →
  (∃ (non_defective : ℕ), 
    (non_defective : ℚ) / total_pens * ((non_defective - 1) : ℚ) / (total_pens - 1) = prob_non_defective ∧
    total_pens - non_defective = 1) :=
by sorry

end NUMINAMATH_CALUDE_defective_pen_count_l3978_397854


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l3978_397814

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5) →
  a₀ + a₂ + a₄ = -16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l3978_397814


namespace NUMINAMATH_CALUDE_casket_inscription_proof_l3978_397869

/-- Represents a craftsman who can make caskets -/
inductive Craftsman
| Bellini
| Cellini
| CelliniSon

/-- Represents a casket with an inscription -/
structure Casket where
  maker : Craftsman
  inscription : String

/-- Determines if an inscription is true for a pair of caskets -/
def isInscriptionTrue (c1 c2 : Casket) (inscription : String) : Prop :=
  match inscription with
  | "At least one of these boxes was made by Cellini's son" =>
    c1.maker = Craftsman.CelliniSon ∨ c2.maker = Craftsman.CelliniSon
  | _ => False

/-- Cellini's son never engraves true statements -/
axiom celliniSonFalsity (c : Casket) :
  c.maker = Craftsman.CelliniSon → ¬(isInscriptionTrue c c c.inscription)

/-- The inscription that solves the problem -/
def problemInscription : String :=
  "At least one of these boxes was made by Cellini's son"

theorem casket_inscription_proof :
  ∃ (c1 c2 : Casket),
    (c1.inscription = problemInscription) ∧
    (c2.inscription = problemInscription) ∧
    (c1.maker = c2.maker) ∧
    (c1.maker = Craftsman.Bellini ∨ c1.maker = Craftsman.Cellini) ∧
    (¬∃ (c : Casket), c.inscription = problemInscription →
      c.maker = Craftsman.Bellini ∨ c.maker = Craftsman.Cellini) ∧
    (∀ (c : Casket), c.inscription = problemInscription →
      ¬(c.maker = Craftsman.Bellini ∨ c.maker = Craftsman.Cellini)) :=
by
  sorry


end NUMINAMATH_CALUDE_casket_inscription_proof_l3978_397869


namespace NUMINAMATH_CALUDE_inequality_proof_l3978_397837

theorem inequality_proof (a r : ℝ) (n : ℕ) 
  (ha : a ≥ -2) (hr : r ≥ 0) (hn : n ≥ 1) :
  r^(2*n) + a*r^n + 1 ≥ (1 - r)^(2*n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3978_397837


namespace NUMINAMATH_CALUDE_monotonic_increasing_iff_b_range_l3978_397841

/-- The function y = (1/3)x³ + bx² + (b+2)x + 3 is monotonically increasing on ℝ 
    if and only if b < -1 or b > 2 -/
theorem monotonic_increasing_iff_b_range (b : ℝ) : 
  (∀ x : ℝ, StrictMono (fun x => (1/3) * x^3 + b * x^2 + (b + 2) * x + 3)) ↔ 
  (b < -1 ∨ b > 2) := by
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_iff_b_range_l3978_397841


namespace NUMINAMATH_CALUDE_harkamal_purchase_amount_l3978_397824

/-- The total amount paid by Harkamal for grapes and mangoes -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Harkamal paid 1135 for his purchase -/
theorem harkamal_purchase_amount :
  total_amount_paid 8 80 9 55 = 1135 := by
  sorry


end NUMINAMATH_CALUDE_harkamal_purchase_amount_l3978_397824


namespace NUMINAMATH_CALUDE_haley_cousins_count_l3978_397867

/-- The number of origami papers Haley has to give away -/
def total_papers : ℕ := 48

/-- The number of origami papers each cousin receives -/
def papers_per_cousin : ℕ := 8

/-- Haley's number of cousins -/
def num_cousins : ℕ := total_papers / papers_per_cousin

theorem haley_cousins_count : num_cousins = 6 := by
  sorry

end NUMINAMATH_CALUDE_haley_cousins_count_l3978_397867


namespace NUMINAMATH_CALUDE_smallest_numbers_with_special_property_l3978_397835

theorem smallest_numbers_with_special_property :
  ∃ (a b : ℕ), a > b ∧ 
    (∃ (k : ℕ), a^2 - b^2 = k^3) ∧
    (∃ (m : ℕ), a^3 - b^3 = m^2) ∧
    (∀ (x y : ℕ), x > y → 
      (∃ (k : ℕ), x^2 - y^2 = k^3) →
      (∃ (m : ℕ), x^3 - y^3 = m^2) →
      (x > a ∨ (x = a ∧ y ≥ b))) ∧
    a = 10 ∧ b = 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_numbers_with_special_property_l3978_397835


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3978_397828

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 11 + 7 * Complex.I) : 
  z = 7 - 11 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3978_397828


namespace NUMINAMATH_CALUDE_absent_percentage_l3978_397810

theorem absent_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ)
  (h_total : total_students = 180)
  (h_boys : boys = 100)
  (h_girls : girls = 80)
  (h_sum : total_students = boys + girls)
  (absent_boys_fraction : ℚ)
  (absent_girls_fraction : ℚ)
  (h_absent_boys : absent_boys_fraction = 1 / 5)
  (h_absent_girls : absent_girls_fraction = 1 / 4) :
  (absent_boys_fraction * boys + absent_girls_fraction * girls) / total_students = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_absent_percentage_l3978_397810


namespace NUMINAMATH_CALUDE_probability_from_odds_probability_3_5_odds_l3978_397888

/-- Given odds of a:b in favor of an event, the probability of the event occurring is a/(a+b) -/
theorem probability_from_odds (a b : ℕ) (h : a > 0 ∧ b > 0) :
  let odds := a / b
  let probability := a / (a + b)
  probability = odds / (1 + odds) :=
by sorry

/-- The probability of an event with odds 3:5 in its favor is 3/8 -/
theorem probability_3_5_odds :
  let a := 3
  let b := 5
  let probability := a / (a + b)
  probability = 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_probability_from_odds_probability_3_5_odds_l3978_397888


namespace NUMINAMATH_CALUDE_dot_product_theorem_l3978_397853

def a : ℝ × ℝ := (1, 3)

theorem dot_product_theorem (b : ℝ × ℝ) 
  (h1 : Real.sqrt ((b.1 - 1)^2 + (b.2 - 3)^2) = Real.sqrt 10)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 2) :
  (2 * a.1 + b.1) * (a.1 - b.1) + (2 * a.2 + b.2) * (a.2 - b.2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l3978_397853


namespace NUMINAMATH_CALUDE_shark_sightings_problem_l3978_397834

/-- Shark sightings problem -/
theorem shark_sightings_problem 
  (daytona : ℕ) 
  (cape_may long_beach santa_cruz : ℕ) :
  daytona = 26 ∧
  daytona = 3 * cape_may + 5 ∧
  long_beach = 2 * cape_may ∧
  long_beach = daytona - 4 ∧
  santa_cruz = cape_may + long_beach + 3 ∧
  santa_cruz = daytona - 9 →
  cape_may = 7 ∧ long_beach = 22 ∧ santa_cruz = 32 := by
sorry

end NUMINAMATH_CALUDE_shark_sightings_problem_l3978_397834


namespace NUMINAMATH_CALUDE_horner_method_value_l3978_397887

def horner_polynomial (x : ℝ) : ℝ := (((-6 * x + 5) * x + 0) * x + 2) * x + 6

theorem horner_method_value :
  horner_polynomial 3 = -115 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_value_l3978_397887


namespace NUMINAMATH_CALUDE_jacob_rental_cost_l3978_397813

/-- Calculates the total cost of renting a car given the daily rate, per-mile rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℚ) (mile_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mile_rate * miles

/-- Proves that Jacob's total car rental cost is $237.5 given the specified conditions. -/
theorem jacob_rental_cost :
  let daily_rate : ℚ := 30
  let mile_rate : ℚ := 1/4
  let rental_days : ℕ := 5
  let miles_driven : ℕ := 350
  total_rental_cost daily_rate mile_rate rental_days miles_driven = 237.5 := by
sorry

end NUMINAMATH_CALUDE_jacob_rental_cost_l3978_397813


namespace NUMINAMATH_CALUDE_total_workers_is_18_l3978_397881

/-- Represents the total number of workers in a workshop -/
def total_workers : ℕ := sorry

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := 6

/-- Represents the average salary of all workers in the workshop -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians in the workshop -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technicians in the workshop -/
def avg_salary_non_technicians : ℕ := 6000

/-- Theorem stating that given the conditions, the total number of workers is 18 -/
theorem total_workers_is_18 :
  (total_workers * avg_salary_all = 
    num_technicians * avg_salary_technicians + 
    (total_workers - num_technicians) * avg_salary_non_technicians) →
  total_workers = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_workers_is_18_l3978_397881


namespace NUMINAMATH_CALUDE_cyclists_speed_l3978_397836

/-- Proves that the cyclist's speed is 11 miles per hour given the problem conditions --/
theorem cyclists_speed (hiker_speed : ℝ) (cyclist_travel_time : ℝ) (hiker_catch_up_time : ℝ) :
  hiker_speed = 4 →
  cyclist_travel_time = 5 / 60 →
  hiker_catch_up_time = 13.75 / 60 →
  ∃ (cyclist_speed : ℝ), cyclist_speed = 11 :=
by
  sorry

#check cyclists_speed

end NUMINAMATH_CALUDE_cyclists_speed_l3978_397836


namespace NUMINAMATH_CALUDE_tax_saving_theorem_l3978_397816

theorem tax_saving_theorem (old_rate new_rate : ℝ) (saving : ℝ) (income : ℝ) : 
  old_rate = 0.45 → 
  new_rate = 0.30 → 
  saving = 7200 → 
  (old_rate - new_rate) * income = saving → 
  income = 48000 := by
sorry

end NUMINAMATH_CALUDE_tax_saving_theorem_l3978_397816


namespace NUMINAMATH_CALUDE_leading_coefficient_of_f_l3978_397878

/-- Given a polynomial f satisfying f(x + 1) - f(x) = 6x + 4 for all x,
    prove that the leading coefficient of f is 3. -/
theorem leading_coefficient_of_f (f : ℝ → ℝ) :
  (∀ x, f (x + 1) - f x = 6 * x + 4) →
  ∃ c, ∀ x, f x = 3 * x^2 + x + c :=
sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_f_l3978_397878


namespace NUMINAMATH_CALUDE_tracy_candies_l3978_397812

theorem tracy_candies (x : ℕ) : 
  (∃ (y : ℕ), x = 4 * y) →  -- x is divisible by 4
  (∃ (z : ℕ), (3 * x) / 4 = 3 * z) →  -- (3/4)x is divisible by 3
  (7 ≤ x / 2 - 24) →  -- lower bound after brother takes candies
  (x / 2 - 24 ≤ 11) →  -- upper bound after brother takes candies
  (x = 72 ∨ x = 76) :=
by sorry

end NUMINAMATH_CALUDE_tracy_candies_l3978_397812


namespace NUMINAMATH_CALUDE_complexity_not_greater_for_power_of_two_exists_number_with_greater_or_equal_complexity_l3978_397876

/-- The complexity of an integer is the number of factors in its prime factorization -/
def complexity (n : ℕ) : ℕ := sorry

/-- For n = 2^k, all numbers between n and 2n have complexity not greater than that of n -/
theorem complexity_not_greater_for_power_of_two (k : ℕ) :
  ∀ m : ℕ, 2^k ≤ m → m ≤ 2^(k+1) → complexity m ≤ complexity (2^k) := by sorry

/-- For any n > 1, there exists at least one number between n and 2n with complexity greater than or equal to that of n -/
theorem exists_number_with_greater_or_equal_complexity (n : ℕ) (h : n > 1) :
  ∃ m : ℕ, n < m ∧ m < 2*n ∧ complexity m ≥ complexity n := by sorry

end NUMINAMATH_CALUDE_complexity_not_greater_for_power_of_two_exists_number_with_greater_or_equal_complexity_l3978_397876


namespace NUMINAMATH_CALUDE_sequence_pattern_l3978_397875

def sequence_sum (a b : ℕ) : ℕ := a + b - 1

theorem sequence_pattern : 
  (sequence_sum 6 7 = 12) ∧
  (sequence_sum 8 9 = 16) ∧
  (sequence_sum 5 6 = 10) ∧
  (sequence_sum 7 8 = 14) →
  sequence_sum 3 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_sequence_pattern_l3978_397875


namespace NUMINAMATH_CALUDE_set_membership_implies_x_values_l3978_397856

theorem set_membership_implies_x_values (A : Set ℝ) (x : ℝ) :
  A = {2, 4, x^2 - x} → 6 ∈ A → x = 3 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_x_values_l3978_397856


namespace NUMINAMATH_CALUDE_ensemble_size_l3978_397890

/-- Represents the "Sunshine" ensemble --/
structure Ensemble where
  violin_players : ℕ
  bass_players : ℕ
  violin_avg_age : ℝ
  bass_avg_age : ℝ

/-- Represents the ensemble after Igor's switch --/
structure EnsembleAfterSwitch where
  violin_players : ℕ
  bass_players : ℕ
  violin_avg_age : ℝ
  bass_avg_age : ℝ

/-- Theorem stating the size of the ensemble --/
theorem ensemble_size (e : Ensemble) (e_after : EnsembleAfterSwitch) : 
  e.violin_players + e.bass_players = 23 :=
by
  have h1 : e.violin_avg_age = 22 := by sorry
  have h2 : e.bass_avg_age = 45 := by sorry
  have h3 : e_after.violin_players = e.violin_players + 1 := by sorry
  have h4 : e_after.bass_players = e.bass_players - 1 := by sorry
  have h5 : e_after.violin_avg_age = e.violin_avg_age + 1 := by sorry
  have h6 : e_after.bass_avg_age = e.bass_avg_age + 1 := by sorry
  sorry

#check ensemble_size

end NUMINAMATH_CALUDE_ensemble_size_l3978_397890


namespace NUMINAMATH_CALUDE_sequence_general_term_l3978_397870

theorem sequence_general_term (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) :
  a 1 = 1 ∧
  (∀ n : ℕ+, S n = (n + 2 : ℚ) / 3 * a n) →
  ∀ n : ℕ+, a n = (n * (n + 1) : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3978_397870


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3978_397849

theorem sufficient_not_necessary (x : ℝ) : 
  (1 / x > 2 → x < 1 / 2) ∧ ¬(x < 1 / 2 → 1 / x > 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3978_397849


namespace NUMINAMATH_CALUDE_chad_bbq_ice_cost_l3978_397879

/-- The cost of ice for Chad's BBQ --/
def ice_cost (people : ℕ) (ice_per_person : ℕ) (pack_size : ℕ) (cost_per_pack : ℚ) : ℚ :=
  let total_ice := people * ice_per_person
  let packs_needed := (total_ice + pack_size - 1) / pack_size  -- Ceiling division
  packs_needed * cost_per_pack

/-- Theorem stating the cost of ice for Chad's BBQ --/
theorem chad_bbq_ice_cost :
  ice_cost 15 2 10 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_chad_bbq_ice_cost_l3978_397879


namespace NUMINAMATH_CALUDE_project_completion_time_l3978_397895

/-- Proves that the total time to complete a project is 15 days given the specified conditions. -/
theorem project_completion_time 
  (a_rate : ℝ) 
  (b_rate : ℝ) 
  (a_quit_before : ℝ) 
  (h1 : a_rate = 1 / 20) 
  (h2 : b_rate = 1 / 30) 
  (h3 : a_quit_before = 5) : 
  ∃ (total_time : ℝ), total_time = 15 ∧ 
    (total_time - a_quit_before) * (a_rate + b_rate) + 
    a_quit_before * b_rate = 1 :=
sorry

end NUMINAMATH_CALUDE_project_completion_time_l3978_397895


namespace NUMINAMATH_CALUDE_first_book_has_200_words_l3978_397897

/-- The number of words in Jenny's first book --/
def first_book_words : ℕ := sorry

/-- The number of words Jenny can read per hour --/
def reading_speed : ℕ := 100

/-- The number of words in the second book --/
def second_book_words : ℕ := 400

/-- The number of words in the third book --/
def third_book_words : ℕ := 300

/-- The number of days Jenny plans to read --/
def reading_days : ℕ := 10

/-- The average number of minutes Jenny spends reading per day --/
def daily_reading_minutes : ℕ := 54

/-- Theorem stating that the first book has 200 words --/
theorem first_book_has_200_words :
  first_book_words = 200 := by sorry

end NUMINAMATH_CALUDE_first_book_has_200_words_l3978_397897


namespace NUMINAMATH_CALUDE_train_journey_time_l3978_397809

theorem train_journey_time (S T D : ℝ) (h1 : D = S * T) (h2 : D = (S / 2) * (T + 4)) :
  T + 4 = 8 := by sorry

end NUMINAMATH_CALUDE_train_journey_time_l3978_397809


namespace NUMINAMATH_CALUDE_infinitely_many_representable_terms_l3978_397848

-- Define the sequence type
def PositiveIntegerSequence := ℕ → ℕ+

-- Define the property that the sequence is strictly increasing
def StrictlyIncreasing (a : PositiveIntegerSequence) : Prop :=
  ∀ k, a k < a (k + 1)

-- State the theorem
theorem infinitely_many_representable_terms 
  (a : PositiveIntegerSequence) 
  (h : StrictlyIncreasing a) : 
  ∃ S : Set ℕ, (Set.Infinite S) ∧ 
    (∀ m ∈ S, ∃ (p q x y : ℕ), 
      p ≠ q ∧ 
      x > 0 ∧ 
      y > 0 ∧ 
      (a m : ℕ) = x * (a p : ℕ) + y * (a q : ℕ)) :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_representable_terms_l3978_397848


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l3978_397889

theorem greatest_integer_with_gcd_six (n : ℕ) : 
  (n < 50 ∧ Nat.gcd n 18 = 6 ∧ ∀ m : ℕ, m < 50 ∧ Nat.gcd m 18 = 6 → m ≤ n) → n = 42 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_l3978_397889


namespace NUMINAMATH_CALUDE_area_of_triangle_l3978_397858

noncomputable def m (x y : ℝ) : ℝ × ℝ := (2 * Real.cos x, y - 2 * Real.sqrt 3 * Real.sin x * Real.cos x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (1, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + 1

theorem area_of_triangle (x y a b c : ℝ) : 
  (∃ k : ℝ, m x y = k • n x) → 
  f (c / 2) = 3 → 
  c = 2 * Real.sqrt 6 → 
  a + b = 6 → 
  (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_area_of_triangle_l3978_397858


namespace NUMINAMATH_CALUDE_arthurs_walk_distance_l3978_397898

/-- Represents Arthur's walk in blocks -/
structure ArthursWalk where
  east : ℕ
  north : ℕ
  south : ℕ
  west : ℕ

/-- Calculates the total distance of Arthur's walk in miles -/
def total_distance (walk : ArthursWalk) : ℚ :=
  (walk.east + walk.north + walk.south + walk.west) * (1 / 4)

/-- Theorem: Arthur's specific walk equals 6.5 miles -/
theorem arthurs_walk_distance :
  let walk : ArthursWalk := { east := 8, north := 10, south := 3, west := 5 }
  total_distance walk = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_walk_distance_l3978_397898


namespace NUMINAMATH_CALUDE_nina_running_distance_l3978_397832

theorem nina_running_distance (total : ℝ) (first : ℝ) (second_known : ℝ) 
  (h1 : total = 0.83)
  (h2 : first = 0.08)
  (h3 : second_known = 0.08) :
  total - (first + second_known) = 0.67 := by
  sorry

end NUMINAMATH_CALUDE_nina_running_distance_l3978_397832


namespace NUMINAMATH_CALUDE_odd_periodic_function_sum_l3978_397859

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has period T if f(x + T) = f(x) for all x -/
def HasPeriod (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem odd_periodic_function_sum (f : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_period : HasPeriod f 4) :
  f 2005 + f 2006 + f 2007 = f 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_sum_l3978_397859


namespace NUMINAMATH_CALUDE_tangent_line_count_possibilities_l3978_397839

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ
  radius_pos : radius > 0

/-- Counts the number of distinct values in a list of natural numbers -/
def countDistinctValues (list : List ℕ) : ℕ :=
  (list.toFinset).card

/-- The possible numbers of tangent lines for two non-overlapping circles -/
def possibleTangentLineCounts : List ℕ := [0, 3, 4]

/-- Theorem stating that for two non-overlapping circles with radii 5 and 8,
    the number of possible distinct values for the count of tangent lines is 3 -/
theorem tangent_line_count_possibilities (circle1 circle2 : Circle)
    (h1 : circle1.radius = 5)
    (h2 : circle2.radius = 8)
    (h_non_overlap : circle1 ≠ circle2) :
    countDistinctValues possibleTangentLineCounts = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_count_possibilities_l3978_397839


namespace NUMINAMATH_CALUDE_wardrobe_cost_calculation_l3978_397804

def wardrobe_cost (skirt_price blouse_price jacket_price pant_price : ℝ)
  (skirt_discount jacket_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let skirt_total := 4 * (skirt_price * (1 - skirt_discount))
  let blouse_total := 6 * blouse_price
  let jacket_total := 2 * (jacket_price - jacket_discount)
  let pant_total := 2 * pant_price + 0.5 * pant_price
  let subtotal := skirt_total + blouse_total + jacket_total + pant_total
  subtotal * (1 + tax_rate)

theorem wardrobe_cost_calculation :
  wardrobe_cost 25 18 45 35 0.1 5 0.07 = 391.09 := by
  sorry

end NUMINAMATH_CALUDE_wardrobe_cost_calculation_l3978_397804


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l3978_397805

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 1 ∨ x < -5} :=
sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (11/2)*t} = {t : ℝ | 1/2 ≤ t ∧ t ≤ 5} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l3978_397805


namespace NUMINAMATH_CALUDE_place_value_decomposition_l3978_397823

theorem place_value_decomposition :
  (286 = 200 + 80 + 6) ∧
  (7560 = 7000 + 500 + 60) ∧
  (2048 = 2000 + 40 + 8) ∧
  (8009 = 8000 + 9) ∧
  (3070 = 3000 + 70) := by
  sorry

end NUMINAMATH_CALUDE_place_value_decomposition_l3978_397823


namespace NUMINAMATH_CALUDE_derivative_of_exp_x_squared_minus_one_l3978_397820

theorem derivative_of_exp_x_squared_minus_one (x : ℝ) :
  deriv (λ x => Real.exp (x^2 - 1)) x = 2 * x * Real.exp (x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_exp_x_squared_minus_one_l3978_397820


namespace NUMINAMATH_CALUDE_equal_area_polygons_equidecomposable_l3978_397852

-- Define a polygon as a set of points in the plane
def Polygon : Type := Set (ℝ × ℝ)

-- Define the concept of area for a polygon
noncomputable def area (P : Polygon) : ℝ := sorry

-- Define equidecomposability
def equidecomposable (P Q : Polygon) : Prop := sorry

-- Theorem statement
theorem equal_area_polygons_equidecomposable (P Q : Polygon) :
  area P = area Q → equidecomposable P Q := by sorry

end NUMINAMATH_CALUDE_equal_area_polygons_equidecomposable_l3978_397852


namespace NUMINAMATH_CALUDE_parallelogram_area_24_16_l3978_397880

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 24 cm and height 16 cm is 384 square centimeters -/
theorem parallelogram_area_24_16 : parallelogram_area 24 16 = 384 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_24_16_l3978_397880


namespace NUMINAMATH_CALUDE_divisibility_of_cube_difference_l3978_397822

theorem divisibility_of_cube_difference (a b c : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime c → 
  c ∣ (a + b) → c ∣ (a * b) → 
  c ∣ (a^3 - b^3) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_cube_difference_l3978_397822


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_l3978_397860

/-- Represents the daily rental cost of a canoe in dollars -/
def canoe_cost : ℚ := 11

/-- Represents the daily rental cost of a kayak in dollars -/
def kayak_cost : ℚ := 16

/-- Represents the ratio of canoes to kayaks rented -/
def rental_ratio : ℚ := 4 / 3

/-- Represents the total revenue in dollars -/
def total_revenue : ℚ := 460

/-- Represents the number of kayaks rented -/
def kayaks : ℕ := 15

/-- Represents the number of canoes rented -/
def canoes : ℕ := 20

theorem canoe_kayak_difference :
  canoes - kayaks = 5 ∧
  canoe_cost * canoes + kayak_cost * kayaks = total_revenue ∧
  (canoes : ℚ) / kayaks = rental_ratio := by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_difference_l3978_397860


namespace NUMINAMATH_CALUDE_prob_A_plus_B_complement_l3978_397871

-- Define the sample space
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define event A
def A : Finset Nat := {2, 4}

-- Define event B
def B : Finset Nat := {1, 2, 3, 4, 5}

-- Define the complement of B
def B_complement : Finset Nat := Ω \ B

-- Define the probability measure
def P (E : Finset Nat) : Rat := (E.card : Rat) / (Ω.card : Rat)

-- State the theorem
theorem prob_A_plus_B_complement : P (A ∪ B_complement) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_plus_B_complement_l3978_397871


namespace NUMINAMATH_CALUDE_interior_angle_sum_regular_polygon_l3978_397846

theorem interior_angle_sum_regular_polygon (n : ℕ) (h : n > 2) :
  let exterior_angle : ℝ := 20
  let interior_angle_sum : ℝ := (n - 2) * 180
  (360 / exterior_angle = n) →
  interior_angle_sum = 2880 :=
by sorry

end NUMINAMATH_CALUDE_interior_angle_sum_regular_polygon_l3978_397846


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l3978_397891

/-- Represents a cyclic quadrilateral ABCD with angles α, β, γ, and ω -/
structure CyclicQuadrilateral where
  α : ℝ
  β : ℝ
  γ : ℝ
  ω : ℝ
  sum_180 : α + β + γ + ω = 180

/-- Theorem: In a cyclic quadrilateral ABCD, if α = c, β = 43°, γ = 59°, and ω = d, then d = 42° -/
theorem cyclic_quadrilateral_angle (q : CyclicQuadrilateral) (h1 : q.α = 36) (h2 : q.β = 43) (h3 : q.γ = 59) : q.ω = 42 := by
  sorry

#check cyclic_quadrilateral_angle

end NUMINAMATH_CALUDE_cyclic_quadrilateral_angle_l3978_397891


namespace NUMINAMATH_CALUDE_number_from_percentage_l3978_397886

theorem number_from_percentage (x : ℝ) : 0.15 * 0.30 * 0.50 * x = 117 → x = 5200 := by
  sorry

end NUMINAMATH_CALUDE_number_from_percentage_l3978_397886


namespace NUMINAMATH_CALUDE_line_translation_distance_l3978_397840

/-- Two lines in a 2D Cartesian coordinate system -/
structure Line2D where
  slope : ℝ
  intercept : ℝ

/-- The vertical distance between two parallel lines -/
def vertical_distance (l1 l2 : Line2D) : ℝ :=
  l2.intercept - l1.intercept

/-- Theorem: The vertical distance between l1 and l2 is 6 units -/
theorem line_translation_distance :
  let l1 : Line2D := { slope := -2, intercept := -2 }
  let l2 : Line2D := { slope := -2, intercept := 4 }
  vertical_distance l1 l2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_line_translation_distance_l3978_397840


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l3978_397830

/-- Represents the total number of handshakes -/
def total_handshakes : ℕ := 435

/-- Calculates the number of handshakes between players given the number of players -/
def player_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the number of handshakes the coach had -/
def coach_handshakes (n : ℕ) : ℕ := total_handshakes - player_handshakes n

theorem min_coach_handshakes :
  ∃ (n : ℕ), n > 1 ∧ player_handshakes n ≤ total_handshakes ∧ coach_handshakes n = 0 :=
sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l3978_397830


namespace NUMINAMATH_CALUDE_mixed_gender_more_likely_l3978_397825

def child_gender := Bool

def prob_all_same_gender (n : ℕ) : ℚ :=
  (1 / 2) ^ n

def prob_mixed_gender (n : ℕ) : ℚ :=
  1 - prob_all_same_gender n

theorem mixed_gender_more_likely (n : ℕ) (h : n = 3) :
  prob_mixed_gender n > prob_all_same_gender n :=
sorry

end NUMINAMATH_CALUDE_mixed_gender_more_likely_l3978_397825


namespace NUMINAMATH_CALUDE_programmer_debug_time_l3978_397802

/-- Proves that given a 48-hour work week, where 1/4 of the time is spent on flow charts
    and 3/8 on coding, the remaining time spent on debugging is 18 hours. -/
theorem programmer_debug_time (total_hours : ℝ) (flow_chart_fraction : ℝ) (coding_fraction : ℝ) :
  total_hours = 48 →
  flow_chart_fraction = 1/4 →
  coding_fraction = 3/8 →
  total_hours * (1 - flow_chart_fraction - coding_fraction) = 18 :=
by sorry

end NUMINAMATH_CALUDE_programmer_debug_time_l3978_397802


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3978_397829

theorem min_value_of_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (2 * a * (-1) - b * 2 + 2 = 0) → 
  (∀ x y : ℝ, 2 * a * x - b * y + 2 = 0 → x^2 + y^2 + 2*x - 4*y + 1 = 0) →
  (∀ c d : ℝ, c > 0 → d > 0 → (2 * c * (-1) - d * 2 + 2 = 0) → 1/a + 1/b ≤ 1/c + 1/d) →
  1/a + 1/b = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3978_397829


namespace NUMINAMATH_CALUDE_even_function_property_l3978_397831

theorem even_function_property (f : ℝ → ℝ) :
  (∀ x, f x = f (-x)) →  -- f is even
  (∀ x > 0, f x = 10^x) →  -- f(x) = 10^x for x > 0
  (∀ x < 0, f x = 10^(-x)) := by  -- f(x) = 10^(-x) for x < 0
sorry

end NUMINAMATH_CALUDE_even_function_property_l3978_397831


namespace NUMINAMATH_CALUDE_binomial_variance_three_fourths_l3978_397850

def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_variance_three_fourths (p : ℝ) 
  (h1 : 0 ≤ p) (h2 : p ≤ 1) 
  (h3 : binomial_variance 3 p = 3/4) : p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_three_fourths_l3978_397850


namespace NUMINAMATH_CALUDE_fibonacci_inequality_l3978_397842

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_inequality (n : ℕ) (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (fibonacci n / fibonacci (n - 1) : ℚ) < (a / b : ℚ) →
  (a / b : ℚ) < (fibonacci (n + 1) / fibonacci n : ℚ) →
  b ≥ fibonacci (n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_fibonacci_inequality_l3978_397842


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l3978_397838

/-- Calculates the average speed of a car trip given the following conditions:
  * The total trip duration is 6 hours
  * The car travels at an average speed of 75 mph for the first 4 hours
  * The car travels at an average speed of 60 mph for the remaining hours
-/
theorem car_trip_average_speed : 
  let total_time : ℝ := 6
  let first_part_time : ℝ := 4
  let second_part_time : ℝ := total_time - first_part_time
  let first_part_speed : ℝ := 75
  let second_part_speed : ℝ := 60
  let total_distance : ℝ := first_part_speed * first_part_time + second_part_speed * second_part_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 70 := by sorry

end NUMINAMATH_CALUDE_car_trip_average_speed_l3978_397838


namespace NUMINAMATH_CALUDE_all_hanging_pieces_equal_l3978_397800

/-- Represents a square table covered by a square tablecloth -/
structure TableWithCloth where
  table_side : ℝ
  cloth_side : ℝ
  hanging_piece : ℝ → ℝ → ℝ
  no_corner_covered : cloth_side > table_side
  no_overlap : cloth_side ≤ table_side + 2 * (hanging_piece 0 0)
  adjacent_equal : ∀ (i j : Fin 4), (i.val + 1) % 4 = j.val → 
    hanging_piece i.val 0 = hanging_piece j.val 0

/-- All four hanging pieces of the tablecloth are equal -/
theorem all_hanging_pieces_equal (t : TableWithCloth) : 
  ∀ (i j : Fin 4), t.hanging_piece i.val 0 = t.hanging_piece j.val 0 := by
  sorry

end NUMINAMATH_CALUDE_all_hanging_pieces_equal_l3978_397800


namespace NUMINAMATH_CALUDE_cube_pyramid_equal_volume_l3978_397863

/-- Given a cube with edge length 6 and a square-based pyramid with base edge length 10,
    if their volumes are equal, then the height of the pyramid is 162/25. -/
theorem cube_pyramid_equal_volume (h : ℚ) : 
  (6 : ℚ)^3 = (1/3 : ℚ) * 10^2 * h → h = 162/25 := by
  sorry

end NUMINAMATH_CALUDE_cube_pyramid_equal_volume_l3978_397863


namespace NUMINAMATH_CALUDE_final_bill_calculation_l3978_397874

def original_bill : ℝ := 400
def late_charge_rate : ℝ := 0.02

def final_amount : ℝ := original_bill * (1 + late_charge_rate)^3

theorem final_bill_calculation : 
  ∃ (ε : ℝ), abs (final_amount - 424.48) < ε ∧ ε > 0 :=
by sorry

end NUMINAMATH_CALUDE_final_bill_calculation_l3978_397874


namespace NUMINAMATH_CALUDE_shark_sightings_total_l3978_397868

/-- The number of shark sightings in Daytona Beach -/
def daytona_beach_sightings : ℕ := sorry

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 24

/-- Cape May has 8 less than double the number of shark sightings of Daytona Beach -/
axiom cape_may_relation : cape_may_sightings = 2 * daytona_beach_sightings - 8

/-- The total number of shark sightings in both locations -/
def total_sightings : ℕ := cape_may_sightings + daytona_beach_sightings

theorem shark_sightings_total : total_sightings = 40 := by
  sorry

end NUMINAMATH_CALUDE_shark_sightings_total_l3978_397868


namespace NUMINAMATH_CALUDE_min_cuts_for_eleven_days_max_rings_for_n_cuts_l3978_397896

/-- Represents a chain of rings -/
structure Chain where
  rings : ℕ

/-- Represents a stay at the inn -/
structure Stay where
  days : ℕ

/-- Calculates the minimum number of cuts required for a given chain and stay -/
def minCuts (chain : Chain) (stay : Stay) : ℕ :=
  sorry

/-- Calculates the maximum number of rings in a chain for a given number of cuts -/
def maxRings (cuts : ℕ) : ℕ :=
  sorry

theorem min_cuts_for_eleven_days (chain : Chain) (stay : Stay) :
  chain.rings = 11 → stay.days = 11 → minCuts chain stay = 2 :=
  sorry

theorem max_rings_for_n_cuts (n : ℕ) :
  maxRings n = (n + 1) * 2^n - 1 :=
  sorry

end NUMINAMATH_CALUDE_min_cuts_for_eleven_days_max_rings_for_n_cuts_l3978_397896


namespace NUMINAMATH_CALUDE_problem_statement_l3978_397847

theorem problem_statement (a b c d x : ℤ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : x = -1)  -- x is the largest negative integer
  : x^2 - (a + b - c * d)^2012 + (-c * d)^2011 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3978_397847


namespace NUMINAMATH_CALUDE_martha_blocks_l3978_397883

/-- The number of blocks Martha ends with is equal to her initial blocks plus the blocks she finds -/
theorem martha_blocks (initial_blocks found_blocks : ℕ) :
  initial_blocks + found_blocks = initial_blocks + found_blocks :=
by sorry

#check martha_blocks 4 80

end NUMINAMATH_CALUDE_martha_blocks_l3978_397883


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l3978_397827

theorem smallest_right_triangle_area (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let area1 := (1/2) * a * b
  let c := Real.sqrt (b^2 - a^2)
  let area2 := (1/2) * a * c
  min area1 area2 = 6 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l3978_397827


namespace NUMINAMATH_CALUDE_equation_solution_l3978_397885

theorem equation_solution (x y : ℚ) 
  (eq1 : 4 * x + y = 20) 
  (eq2 : x + 2 * y = 17) : 
  5 * x^2 + 18 * x * y + 5 * y^2 = 696 + 5/7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3978_397885


namespace NUMINAMATH_CALUDE_oak_grove_books_after_donations_l3978_397864

/-- Represents the number of books in Oak Grove libraries -/
structure OakGroveLibraries where
  public_library : ℕ
  school_libraries : ℕ
  community_center : ℕ

/-- Calculates the total number of books after donations -/
def total_books_after_donations (libs : OakGroveLibraries) (public_donation : ℕ) (community_donation : ℕ) : ℕ :=
  libs.public_library + libs.school_libraries + libs.community_center - public_donation - community_donation

/-- Theorem stating the total number of books after donations -/
theorem oak_grove_books_after_donations :
  let initial_libraries : OakGroveLibraries := {
    public_library := 1986,
    school_libraries := 5106,
    community_center := 3462
  }
  let public_donation : ℕ := 235
  let community_donation : ℕ := 328
  total_books_after_donations initial_libraries public_donation community_donation = 9991 := by
  sorry


end NUMINAMATH_CALUDE_oak_grove_books_after_donations_l3978_397864


namespace NUMINAMATH_CALUDE_number_of_routes_l3978_397843

def grid_size : ℕ := 3

def total_moves : ℕ := 2 * grid_size

def right_moves : ℕ := grid_size

def down_moves : ℕ := grid_size

theorem number_of_routes : Nat.choose total_moves right_moves = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_routes_l3978_397843


namespace NUMINAMATH_CALUDE_square_equation_solution_l3978_397808

theorem square_equation_solution :
  ∃! x : ℤ, (2020 + x)^2 = x^2 :=
by
  use -1010
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3978_397808


namespace NUMINAMATH_CALUDE_prism_volume_l3978_397801

/-- A right rectangular prism with face areas 15, 20, and 24 square inches has a volume of 60 cubic inches. -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 15) (h2 : w * h = 20) (h3 : l * h = 24) :
  l * w * h = 60 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l3978_397801


namespace NUMINAMATH_CALUDE_min_a_for_monotonic_odd_function_l3978_397893

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x > 0 then Real.exp x + a
  else if x < 0 then -(Real.exp (-x) + a)
  else 0

-- State the theorem
theorem min_a_for_monotonic_odd_function :
  ∀ a : ℝ, 
  (∀ x : ℝ, f a x = -f a (-x)) → -- f is odd
  (∀ x y : ℝ, x < y → f a x ≤ f a y) → -- f is monotonic
  a ≥ -1 ∧ 
  ∀ b : ℝ, (∀ x : ℝ, f b x = -f b (-x)) → 
            (∀ x y : ℝ, x < y → f b x ≤ f b y) → 
            b ≥ a :=
by sorry

end NUMINAMATH_CALUDE_min_a_for_monotonic_odd_function_l3978_397893


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3978_397845

theorem simplify_trig_expression :
  let sin30 : ℝ := 1 / 2
  let cos30 : ℝ := Real.sqrt 3 / 2
  ∀ (sin10 sin20 cos10 : ℝ),
    (sin10 + sin20 * cos30) / (cos10 - sin20 * sin30) = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3978_397845


namespace NUMINAMATH_CALUDE_equation_simplification_l3978_397884

theorem equation_simplification :
  (Real.sqrt ((7 : ℝ)^2 + 24^2)) / (Real.sqrt (49 + 16)) = (25 * Real.sqrt 65) / 65 := by
  sorry

end NUMINAMATH_CALUDE_equation_simplification_l3978_397884
