import Mathlib

namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1346_134647

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 2 → 1 / x' + 1 / y' ≥ 1 / x + 1 / y) →
  1 / x + 1 / y = 3 / 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1346_134647


namespace NUMINAMATH_CALUDE_prime_solution_existence_l1346_134677

theorem prime_solution_existence (p : ℕ) : 
  Prime p ↔ (p = 2 ∨ p = 3 ∨ p = 7) ∧ 
  (∃ x y : ℕ+, x * (y^2 - p) + y * (x^2 - p) = 5 * p) :=
sorry

end NUMINAMATH_CALUDE_prime_solution_existence_l1346_134677


namespace NUMINAMATH_CALUDE_number_line_position_l1346_134635

/-- Given a number line where the distance from 0 to 25 is divided into 5 equal steps,
    the position after 4 steps from 0 is 20. -/
theorem number_line_position (total_distance : ℝ) (total_steps : ℕ) (steps_taken : ℕ) :
  total_distance = 25 ∧ total_steps = 5 ∧ steps_taken = 4 →
  (total_distance / total_steps) * steps_taken = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_line_position_l1346_134635


namespace NUMINAMATH_CALUDE_moles_of_HCN_l1346_134662

-- Define the reaction components
structure Reaction where
  CuSO4 : ℝ
  HCN : ℝ
  Cu_CN_2 : ℝ
  H2SO4 : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.CuSO4 = r.Cu_CN_2 ∧ r.HCN = 4 * r.CuSO4 ∧ r.H2SO4 = r.CuSO4

-- Define the given conditions
def given_conditions (r : Reaction) : Prop :=
  r.CuSO4 = 1 ∧ r.Cu_CN_2 = 1

-- Theorem to prove
theorem moles_of_HCN (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : given_conditions r) : 
  r.HCN = 4 :=
sorry

end NUMINAMATH_CALUDE_moles_of_HCN_l1346_134662


namespace NUMINAMATH_CALUDE_director_sphere_theorem_l1346_134688

/-- The surface S: ax^2 + by^2 + cz^2 = 1 -/
def S (a b c : ℝ) (x y z : ℝ) : Prop :=
  a * x^2 + b * y^2 + c * z^2 = 1

/-- The director sphere K: x^2 + y^2 + z^2 = 1/a + 1/b + 1/c -/
def K (a b c : ℝ) (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 = 1/a + 1/b + 1/c

/-- A plane tangent to S at point (u, v, w) -/
def tangent_plane (a b c : ℝ) (u v w x y z : ℝ) : Prop :=
  a * u * x + b * v * y + c * w * z = 1

/-- Three mutually perpendicular planes -/
def perpendicular_planes (p₁ q₁ r₁ p₂ q₂ r₂ p₃ q₃ r₃ : ℝ) : Prop :=
  p₁ * p₂ + q₁ * q₂ + r₁ * r₂ = 0 ∧
  p₁ * p₃ + q₁ * q₃ + r₁ * r₃ = 0 ∧
  p₂ * p₃ + q₂ * q₃ + r₂ * r₃ = 0

theorem director_sphere_theorem (a b c : ℝ) (x₀ y₀ z₀ : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ u₁ v₁ w₁ u₂ v₂ w₂ u₃ v₃ w₃ : ℝ,
    S a b c u₁ v₁ w₁ ∧ S a b c u₂ v₂ w₂ ∧ S a b c u₃ v₃ w₃ ∧
    tangent_plane a b c u₁ v₁ w₁ x₀ y₀ z₀ ∧
    tangent_plane a b c u₂ v₂ w₂ x₀ y₀ z₀ ∧
    tangent_plane a b c u₃ v₃ w₃ x₀ y₀ z₀ ∧
    perpendicular_planes 
      (a * u₁) (b * v₁) (c * w₁)
      (a * u₂) (b * v₂) (c * w₂)
      (a * u₃) (b * v₃) (c * w₃)) →
  K a b c x₀ y₀ z₀ := by
  sorry

end NUMINAMATH_CALUDE_director_sphere_theorem_l1346_134688


namespace NUMINAMATH_CALUDE_count_pairs_theorem_l1346_134669

def X : Finset Nat := Finset.range 10

def intersection_set : Finset Nat := {5, 7, 8}

def count_valid_pairs (X : Finset Nat) (intersection_set : Finset Nat) : Nat :=
  let remaining_elements := X \ intersection_set
  3^(remaining_elements.card) - 1

theorem count_pairs_theorem (X : Finset Nat) (intersection_set : Finset Nat) :
  X = Finset.range 10 →
  intersection_set = {5, 7, 8} →
  count_valid_pairs X intersection_set = 2186 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_theorem_l1346_134669


namespace NUMINAMATH_CALUDE_living_room_count_l1346_134695

/-- The number of people in a house. -/
def total_people : ℕ := 15

/-- The number of people in the bedroom. -/
def bedroom_people : ℕ := 7

/-- The number of people in the living room. -/
def living_room_people : ℕ := total_people - bedroom_people

/-- Theorem stating that the number of people in the living room is 8. -/
theorem living_room_count : living_room_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_living_room_count_l1346_134695


namespace NUMINAMATH_CALUDE_set_A_properties_l1346_134606

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem set_A_properties :
  (A = {-2, 2}) ∧ (2 ∈ A) ∧ (-2 ∈ A) := by
  sorry

end NUMINAMATH_CALUDE_set_A_properties_l1346_134606


namespace NUMINAMATH_CALUDE_min_value_of_a_l1346_134674

-- Define the set of x values
def X : Set ℝ := { x | 0 < x ∧ x ≤ 1/2 }

-- Define the inequality condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x ∈ X, x^2 + a*x + 1 ≥ 0

-- State the theorem
theorem min_value_of_a :
  (∃ a_min : ℝ, inequality_holds a_min ∧
    ∀ a : ℝ, inequality_holds a → a ≥ a_min) ∧
  (∀ a_min : ℝ, (inequality_holds a_min ∧
    ∀ a : ℝ, inequality_holds a → a ≥ a_min) →
    a_min = -5/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l1346_134674


namespace NUMINAMATH_CALUDE_hotel_reunion_attendees_l1346_134686

theorem hotel_reunion_attendees (total_guests dates_attendees hall_attendees : ℕ) 
  (h1 : total_guests = 50)
  (h2 : dates_attendees = 50)
  (h3 : hall_attendees = 60)
  (h4 : ∀ g, g ≤ total_guests → (g ≤ dates_attendees ∨ g ≤ hall_attendees)) :
  dates_attendees + hall_attendees - total_guests = 60 := by
  sorry

end NUMINAMATH_CALUDE_hotel_reunion_attendees_l1346_134686


namespace NUMINAMATH_CALUDE_max_profit_at_90_l1346_134650

noncomputable section

-- Define the cost function
def cost (x : ℝ) : ℝ :=
  if x < 90 then 0.5 * x^2 + 60 * x + 5
  else 121 * x + 8100 / x - 2180 + 5

-- Define the revenue function
def revenue (x : ℝ) : ℝ := 1.2 * x

-- Define the profit function
def profit (x : ℝ) : ℝ := revenue x - cost x

-- Theorem statement
theorem max_profit_at_90 :
  ∀ x > 0, profit x ≤ profit 90 ∧ profit 90 = 1500 := by sorry

end

end NUMINAMATH_CALUDE_max_profit_at_90_l1346_134650


namespace NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l1346_134632

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | |x - 1| < 1}

def B : Set ℝ := {x | x < 1 ∨ x ≥ 4}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {x : ℝ | 0 < x ∧ x < 4} :=
by sorry

end NUMINAMATH_CALUDE_union_of_A_and_complement_of_B_l1346_134632


namespace NUMINAMATH_CALUDE_g_neg_one_value_l1346_134693

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of f being an odd function when combined with x^2
def isOddWithSquare (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- State the theorem
theorem g_neg_one_value (f : ℝ → ℝ) 
  (h1 : isOddWithSquare f) 
  (h2 : f 1 = 1) : 
  g f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_one_value_l1346_134693


namespace NUMINAMATH_CALUDE_inequality_solution_l1346_134689

theorem inequality_solution (x : ℝ) :
  3 * x^2 - 2 * x ≥ 9 ↔ x ≤ -1 ∨ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1346_134689


namespace NUMINAMATH_CALUDE_quadrilateral_circumcenter_l1346_134623

-- Define the types of quadrilaterals
inductive Quadrilateral
  | Square
  | NonSquareRectangle
  | NonSquareRhombus
  | Parallelogram
  | IsoscelesTrapezoid

-- Define a function to check if a quadrilateral has a circumcenter
def hasCircumcenter (q : Quadrilateral) : Prop :=
  match q with
  | Quadrilateral.Square => True
  | Quadrilateral.NonSquareRectangle => True
  | Quadrilateral.NonSquareRhombus => False
  | Quadrilateral.Parallelogram => False
  | Quadrilateral.IsoscelesTrapezoid => True

-- Theorem stating which quadrilaterals have a circumcenter
theorem quadrilateral_circumcenter :
  ∀ q : Quadrilateral,
    hasCircumcenter q ↔
      (q = Quadrilateral.Square ∨
       q = Quadrilateral.NonSquareRectangle ∨
       q = Quadrilateral.IsoscelesTrapezoid) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_circumcenter_l1346_134623


namespace NUMINAMATH_CALUDE_solution_set_l1346_134661

theorem solution_set (m : ℤ) 
  (h1 : ∃! (x : ℤ), |2*x - m| ≤ 1 ∧ x = 2) :
  {x : ℝ | |x - 1| + |x - 3| ≥ m} = 
    {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l1346_134661


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1346_134672

theorem min_value_expression (x : ℝ) : 
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 2 * Real.sqrt 5 :=
by sorry

theorem equality_condition : 
  ∃ x : ℝ, Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l1346_134672


namespace NUMINAMATH_CALUDE_smallest_integer_solution_minus_four_is_smallest_l1346_134685

theorem smallest_integer_solution (x : ℤ) : (7 - 3 * x < 22) ↔ (x ≥ -4) :=
  sorry

theorem minus_four_is_smallest : ∀ y : ℤ, (7 - 3 * y < 22) → y ≥ -4 :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_minus_four_is_smallest_l1346_134685


namespace NUMINAMATH_CALUDE_alloy_composition_l1346_134619

theorem alloy_composition (m₁ m₂ m₃ m₄ : ℝ) 
  (total_mass : m₁ + m₂ + m₃ + m₄ = 20)
  (first_second_relation : m₁ = 1.5 * m₂)
  (second_third_ratio : m₂ = (3/4) * m₃)
  (third_fourth_ratio : m₃ = (5/6) * m₄) :
  m₄ = 960 / 123 := by
  sorry

end NUMINAMATH_CALUDE_alloy_composition_l1346_134619


namespace NUMINAMATH_CALUDE_triangle_with_integer_altitudes_and_prime_inradius_l1346_134679

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Calculates the semi-perimeter of a triangle -/
def semiPerimeter (t : Triangle) : ℚ :=
  (t.a.val + t.b.val + t.c.val) / 2

/-- Calculates the area of a triangle using Heron's formula -/
def area (t : Triangle) : ℚ :=
  let s := semiPerimeter t
  (s * (s - t.a.val) * (s - t.b.val) * (s - t.c.val)).sqrt

/-- Calculates the inradius of a triangle -/
def inradius (t : Triangle) : ℚ :=
  area t / semiPerimeter t

/-- Calculates the altitude to side a of a triangle -/
def altitudeA (t : Triangle) : ℚ :=
  2 * area t / t.a.val

/-- Calculates the altitude to side b of a triangle -/
def altitudeB (t : Triangle) : ℚ :=
  2 * area t / t.b.val

/-- Calculates the altitude to side c of a triangle -/
def altitudeC (t : Triangle) : ℚ :=
  2 * area t / t.c.val

/-- States that a number is prime -/
def isPrime (n : ℕ) : Prop :=
  Nat.Prime n

theorem triangle_with_integer_altitudes_and_prime_inradius :
  ∃ (t : Triangle),
    t.a = 13 ∧ t.b = 14 ∧ t.c = 15 ∧
    (altitudeA t).isInt ∧ (altitudeB t).isInt ∧ (altitudeC t).isInt ∧
    ∃ (r : ℕ), (inradius t) = r ∧ isPrime r :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_integer_altitudes_and_prime_inradius_l1346_134679


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1346_134605

theorem complex_number_quadrant : ∃ (z : ℂ), z = (25 / (3 - 4*I)) * I ∧ (z.re < 0 ∧ z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1346_134605


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l1346_134666

theorem discount_percentage_proof (num_toys : ℕ) (cost_per_toy : ℚ) (total_paid : ℚ) :
  num_toys = 5 →
  cost_per_toy = 3 →
  total_paid = 12 →
  (1 - total_paid / (num_toys * cost_per_toy)) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l1346_134666


namespace NUMINAMATH_CALUDE_number_problem_l1346_134668

theorem number_problem (x y : ℝ) 
  (h1 : (40 / 100) * x = (30 / 100) * 50)
  (h2 : (60 / 100) * x = (45 / 100) * y) :
  x = 37.5 ∧ y = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1346_134668


namespace NUMINAMATH_CALUDE_table_seats_l1346_134624

/-- The number of people sitting at the table -/
def n : ℕ := 10

/-- The sum of seeds taken in the first round -/
def first_round_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of seeds taken in the second round -/
def second_round_sum (n : ℕ) : ℕ := n * (n + 1) / 2 + n^2

/-- The theorem stating that n = 10 satisfies the conditions -/
theorem table_seats : 
  (second_round_sum n - first_round_sum n = 100) ∧ 
  (∀ m : ℕ, second_round_sum m - first_round_sum m = 100 → m = n) := by
  sorry

#check table_seats

end NUMINAMATH_CALUDE_table_seats_l1346_134624


namespace NUMINAMATH_CALUDE_perpendicular_line_through_B_l1346_134639

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point B
def point_B : ℝ × ℝ := (3, 0)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Theorem statement
theorem perpendicular_line_through_B :
  (perpendicular_line point_B.1 point_B.2) ∧
  (∀ (x y : ℝ), perpendicular_line x y → given_line x y → 
    (x - point_B.1) * (x - point_B.1) + (y - point_B.2) * (y - point_B.2) ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_B_l1346_134639


namespace NUMINAMATH_CALUDE_intersection_range_l1346_134683

def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (9 - p.1^2) ∧ p.2 ≠ 0}

def N (b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = p.1 + b}

theorem intersection_range (b : ℝ) (h : (M ∩ N b).Nonempty) : b ∈ Set.Ioo (-3) (3 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_range_l1346_134683


namespace NUMINAMATH_CALUDE_parabola_intersection_point_l1346_134681

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.evaluate (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_intersection_point (p : Parabola) (h1 : p.a = 1 ∧ p.b = -2 ∧ p.c = -3)
    (h2 : p.evaluate (-1) = 0) :
    p.evaluate 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_point_l1346_134681


namespace NUMINAMATH_CALUDE_speed_increase_ratio_l1346_134649

theorem speed_increase_ratio (v : ℝ) (h : (v + 2) / v = 2.5) :
  (v + 4) / v = 4 := by
  sorry

end NUMINAMATH_CALUDE_speed_increase_ratio_l1346_134649


namespace NUMINAMATH_CALUDE_intersection_line_canonical_equations_l1346_134604

/-- The canonical equations of the line formed by the intersection of two planes -/
theorem intersection_line_canonical_equations 
  (plane1 : ℝ → ℝ → ℝ → Prop) 
  (plane2 : ℝ → ℝ → ℝ → Prop) 
  (canonical_eq : ℝ → ℝ → ℝ → Prop) : 
  (∀ x y z, plane1 x y z ↔ 3*x + 3*y - 2*z - 1 = 0) →
  (∀ x y z, plane2 x y z ↔ 2*x - 3*y + z + 6 = 0) →
  (∀ x y z, canonical_eq x y z ↔ (x + 1)/(-3) = (y - 4/3)/(-7) ∧ (y - 4/3)/(-7) = z/(-15)) →
  ∀ x y z, (plane1 x y z ∧ plane2 x y z) ↔ canonical_eq x y z :=
sorry

end NUMINAMATH_CALUDE_intersection_line_canonical_equations_l1346_134604


namespace NUMINAMATH_CALUDE_inequality_proof_l1346_134609

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  x^2 / (1 + x + x*y*z) + y^2 / (1 + y + x*y*z) + z^2 / (1 + z + x*y*z) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1346_134609


namespace NUMINAMATH_CALUDE_cube_triangle_area_sum_l1346_134682

/-- Represents a 2x2x2 cube -/
structure Cube :=
  (side : ℝ)
  (is_two_by_two_by_two : side = 2)

/-- Represents a triangle with vertices on the cube -/
structure CubeTriangle :=
  (cube : Cube)
  (vertices : Fin 3 → Fin 8)

/-- The area of a triangle on the cube -/
noncomputable def triangleArea (t : CubeTriangle) : ℝ := sorry

/-- The sum of areas of all triangles on the cube -/
noncomputable def totalArea (c : Cube) : ℝ := sorry

/-- The representation of the total area in the form m + √n + √p -/
structure AreaRepresentation (c : Cube) :=
  (m n p : ℕ)
  (total_area_eq : totalArea c = m + Real.sqrt n + Real.sqrt p)

theorem cube_triangle_area_sum (c : Cube) (rep : AreaRepresentation c) :
  rep.m + rep.n + rep.p = 11584 := by sorry

end NUMINAMATH_CALUDE_cube_triangle_area_sum_l1346_134682


namespace NUMINAMATH_CALUDE_maize_donated_amount_l1346_134643

/-- The amount of maize donated to Alfred -/
def maize_donated (
  stored_per_month : ℕ)  -- Amount of maize stored per month
  (months : ℕ)           -- Number of months
  (stolen : ℕ)           -- Amount of maize stolen
  (final_amount : ℕ)     -- Final amount of maize after 2 years
  : ℕ :=
  final_amount - (stored_per_month * months - stolen)

/-- Theorem stating the amount of maize donated to Alfred -/
theorem maize_donated_amount :
  maize_donated 1 24 5 27 = 8 := by
  sorry

end NUMINAMATH_CALUDE_maize_donated_amount_l1346_134643


namespace NUMINAMATH_CALUDE_largest_even_three_digit_number_with_conditions_l1346_134691

theorem largest_even_three_digit_number_with_conditions :
  ∃ (x : ℕ), 
    x = 972 ∧
    x % 2 = 0 ∧
    100 ≤ x ∧ x < 1000 ∧
    x % 5 = 2 ∧
    Nat.gcd 30 (Nat.gcd x 15) = 3 ∧
    ∀ (y : ℕ), 
      y % 2 = 0 → 
      100 ≤ y → y < 1000 → 
      y % 5 = 2 → 
      Nat.gcd 30 (Nat.gcd y 15) = 3 → 
      y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_even_three_digit_number_with_conditions_l1346_134691


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1346_134628

theorem nested_fraction_evaluation :
  1 / (1 + 1 / (2 + 1 / (1 + 1 / 4))) = 14 / 19 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1346_134628


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1346_134641

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1346_134641


namespace NUMINAMATH_CALUDE_integer_between_sqrt2_and_sqrt12_l1346_134667

theorem integer_between_sqrt2_and_sqrt12 (a : ℤ) : 
  (Real.sqrt 2 < a) ∧ (a < Real.sqrt 12) → (a = 2 ∨ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt2_and_sqrt12_l1346_134667


namespace NUMINAMATH_CALUDE_repeating_decimal_limit_l1346_134621

/-- Define the sequence of partial sums for 0.9999... -/
def partialSum (n : ℕ) : ℚ := 1 - (1 / 10 ^ n)

/-- Theorem: The limit of the sequence of partial sums for 0.9999... is 1 -/
theorem repeating_decimal_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |partialSum n - 1| < ε :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_limit_l1346_134621


namespace NUMINAMATH_CALUDE_a_zero_sufficient_a_zero_not_necessary_l1346_134690

def f (a b x : ℝ) : ℝ := x^2 + a * abs x + b

-- Sufficient condition
theorem a_zero_sufficient (a b : ℝ) :
  a = 0 → ∀ x, f a b x = f a b (-x) :=
sorry

-- Not necessary condition
theorem a_zero_not_necessary :
  ∃ a b : ℝ, a ≠ 0 ∧ (∀ x, f a b x = f a b (-x)) :=
sorry

end NUMINAMATH_CALUDE_a_zero_sufficient_a_zero_not_necessary_l1346_134690


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1346_134612

theorem no_integer_solutions : ¬∃ (k : ℕ+) (x : ℤ), 3 * (k : ℤ) * x - 18 = 5 * (k : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1346_134612


namespace NUMINAMATH_CALUDE_triples_divisible_by_1000_l1346_134622

/-- The number of ordered triples (a,b,c) in {1, ..., 2016}³ such that a² + b² + c² ≡ 0 (mod 2017) is divisible by 1000. -/
theorem triples_divisible_by_1000 : ∃ N : ℕ,
  (N = (Finset.filter (fun (t : ℕ × ℕ × ℕ) =>
    let (a, b, c) := t
    1 ≤ a ∧ a ≤ 2016 ∧
    1 ≤ b ∧ b ≤ 2016 ∧
    1 ≤ c ∧ c ≤ 2016 ∧
    (a^2 + b^2 + c^2) % 2017 = 0)
    (Finset.product (Finset.range 2016) (Finset.product (Finset.range 2016) (Finset.range 2016)))).card) ∧
  N % 1000 = 0 :=
by sorry

end NUMINAMATH_CALUDE_triples_divisible_by_1000_l1346_134622


namespace NUMINAMATH_CALUDE_sine_product_ratio_equals_one_l1346_134602

theorem sine_product_ratio_equals_one :
  let d : ℝ := 2 * Real.pi / 15
  (Real.sin (4 * d) * Real.sin (6 * d) * Real.sin (8 * d) * Real.sin (10 * d) * Real.sin (12 * d)) /
  (Real.sin (2 * d) * Real.sin (3 * d) * Real.sin (4 * d) * Real.sin (5 * d) * Real.sin (6 * d)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_sine_product_ratio_equals_one_l1346_134602


namespace NUMINAMATH_CALUDE_f_inequality_l1346_134676

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_inequality (hf : Differentiable ℝ f) 
  (h : ∀ x, f x > deriv f x) : 
  f 2013 < Real.exp 2013 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1346_134676


namespace NUMINAMATH_CALUDE_count_valid_numbers_l1346_134634

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 100 ∧ n % sum_of_digits n = 0

theorem count_valid_numbers : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid n) ∧ S.card = 24 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l1346_134634


namespace NUMINAMATH_CALUDE_smallest_common_factor_l1346_134680

theorem smallest_common_factor (n : ℕ) : 
  (∀ m : ℕ, 0 < m → m < 7 → ¬(∃ k : ℕ, k > 1 ∧ k ∣ (8*m - 3) ∧ k ∣ (5*m + 4))) ∧
  (∃ k : ℕ, k > 1 ∧ k ∣ (8*7 - 3) ∧ k ∣ (5*7 + 4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l1346_134680


namespace NUMINAMATH_CALUDE_custom_op_example_l1346_134657

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := a - 2 * b

-- State the theorem
theorem custom_op_example : custom_op 2 (-3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l1346_134657


namespace NUMINAMATH_CALUDE_temp_difference_l1346_134699

-- Define the temperatures
def southern_temp : Int := -7
def northern_temp : Int := -15

-- State the theorem
theorem temp_difference : southern_temp - northern_temp = 8 := by
  sorry

end NUMINAMATH_CALUDE_temp_difference_l1346_134699


namespace NUMINAMATH_CALUDE_derivative_limit_relation_l1346_134608

theorem derivative_limit_relation (f : ℝ → ℝ) (x₀ : ℝ) (h : HasDerivAt f 2 x₀) :
  Filter.Tendsto (fun k => (f (x₀ - k) - f x₀) / (2 * k)) (Filter.atTop.comap (fun k => 1 / k)) (nhds (-1)) := by
  sorry

end NUMINAMATH_CALUDE_derivative_limit_relation_l1346_134608


namespace NUMINAMATH_CALUDE_unique_point_property_l1346_134611

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (-2, 0)

-- Define the point P
def P : ℝ → ℝ × ℝ := λ p => (p, 0)

-- Define a chord passing through the focus
def chord (m : ℝ) (x : ℝ) : ℝ := m * x + 2 * m

-- Define the angle equality condition
def angle_equality (p : ℝ) : Prop :=
  ∀ m : ℝ, ∃ A B : ℝ × ℝ,
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    A.2 = chord m A.1 ∧
    B.2 = chord m B.1 ∧
    (A.2 - 0) / (A.1 - p) = -(B.2 - 0) / (B.1 - p)

-- Theorem statement
theorem unique_point_property :
  ∃! p : ℝ, p > 0 ∧ angle_equality p :=
sorry

end NUMINAMATH_CALUDE_unique_point_property_l1346_134611


namespace NUMINAMATH_CALUDE_product_of_sums_l1346_134671

theorem product_of_sums : (-1-2-3-4-5-6-7-8-9-10) * (1-2+3-4+5-6+7-8+9-10) = 275 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_l1346_134671


namespace NUMINAMATH_CALUDE_triangle_and_circle_problem_l1346_134625

-- Define the curve C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the circle C₂
def C₂ (x y : ℝ) : Prop := x^2 + (y + Real.sqrt 3)^2 = 1

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (on_C₁ : C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₁ C.1 C.2)
  (counterclockwise : sorry)  -- We would need to define this properly
  (A_coord : A = (2, 0))

-- State the theorem
theorem triangle_and_circle_problem (ABC : Triangle) :
  ABC.B = (-1, Real.sqrt 3) ∧
  ABC.C = (-1, -Real.sqrt 3) ∧
  ∀ (P : ℝ × ℝ), C₂ P.1 P.2 →
    8 ≤ ((P.1 - ABC.B.1)^2 + (P.2 - ABC.B.2)^2) +
        ((P.1 - ABC.C.1)^2 + (P.2 - ABC.C.2)^2) ∧
    ((P.1 - ABC.B.1)^2 + (P.2 - ABC.B.2)^2) +
    ((P.1 - ABC.C.1)^2 + (P.2 - ABC.C.2)^2) ≤ 24 :=
sorry

end NUMINAMATH_CALUDE_triangle_and_circle_problem_l1346_134625


namespace NUMINAMATH_CALUDE_isabella_babysitting_afternoons_l1346_134629

/-- Calculates the number of afternoons Isabella babysits per week -/
def babysitting_afternoons (hourly_rate : ℚ) (hours_per_day : ℚ) (total_weeks : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings / total_weeks) / (hourly_rate * hours_per_day)

/-- Proves that Isabella babysits 6 afternoons per week -/
theorem isabella_babysitting_afternoons :
  babysitting_afternoons 5 5 7 1050 = 6 := by
  sorry

end NUMINAMATH_CALUDE_isabella_babysitting_afternoons_l1346_134629


namespace NUMINAMATH_CALUDE_wall_ratio_l1346_134618

theorem wall_ratio (width height length volume : ℝ) :
  width = 4 →
  height = 6 * width →
  volume = width * height * length →
  volume = 16128 →
  length / height = 7 := by
sorry

end NUMINAMATH_CALUDE_wall_ratio_l1346_134618


namespace NUMINAMATH_CALUDE_twentyFirstTerm_l1346_134652

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a d n : ℕ) : ℕ :=
  a + (n - 1) * d

/-- Theorem: The 21st term of an arithmetic progression with first term 3 and common difference 5 is 103 -/
theorem twentyFirstTerm :
  arithmeticProgressionTerm 3 5 21 = 103 := by
  sorry

end NUMINAMATH_CALUDE_twentyFirstTerm_l1346_134652


namespace NUMINAMATH_CALUDE_six_digit_palindrome_divisibility_l1346_134631

theorem six_digit_palindrome_divisibility (a b : Nat) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≤ 9) :
  let ab := 10 * a + b
  let ababab := 100000 * ab + 1000 * ab + ab
  ∃ (k1 k2 k3 k4 : Nat), ababab = 101 * k1 ∧ ababab = 7 * k2 ∧ ababab = 11 * k3 ∧ ababab = 13 * k4 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_palindrome_divisibility_l1346_134631


namespace NUMINAMATH_CALUDE_alligator_population_after_one_year_l1346_134665

/-- The number of alligators after a given number of doubling periods -/
def alligator_population (initial_population : ℕ) (doubling_periods : ℕ) : ℕ :=
  initial_population * 2^doubling_periods

/-- Theorem: After one year (two doubling periods), 
    the alligator population will be 16 given an initial population of 4 -/
theorem alligator_population_after_one_year :
  alligator_population 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_alligator_population_after_one_year_l1346_134665


namespace NUMINAMATH_CALUDE_negative_number_identification_l1346_134659

theorem negative_number_identification (a b c d : ℝ) 
  (ha : a = -6) (hb : b = 0) (hc : c = 0.2) (hd : d = 3) :
  a < 0 ∧ b ≥ 0 ∧ c > 0 ∧ d > 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_number_identification_l1346_134659


namespace NUMINAMATH_CALUDE_non_square_sequence_2003_l1346_134620

/-- The sequence of positive integers with perfect squares removed -/
def non_square_sequence : ℕ → ℕ := sorry

/-- The 2003rd term of the non-square sequence -/
def term_2003 : ℕ := non_square_sequence 2003

theorem non_square_sequence_2003 : term_2003 = 2048 := by sorry

end NUMINAMATH_CALUDE_non_square_sequence_2003_l1346_134620


namespace NUMINAMATH_CALUDE_stream_speed_l1346_134697

/-- Prove that the speed of the stream is 4 km/hr given the conditions of the boat problem -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (h1 : boat_speed = 24)
  (h2 : distance = 56) (h3 : time = 2) : 
  (distance / time - boat_speed) = 4 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1346_134697


namespace NUMINAMATH_CALUDE_min_value_expression_l1346_134694

theorem min_value_expression (x : ℝ) :
  ∃ (min : ℝ), min = -6480.25 ∧
  ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1346_134694


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1346_134654

/-- Proves that the speed of a train is approximately 80 km/hr given specific conditions -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) (man_speed_kmh : Real) :
  train_length = 220 →
  crossing_time = 10.999120070394369 →
  man_speed_kmh = 8 →
  ∃ (train_speed_kmh : Real), abs (train_speed_kmh - 80) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_calculation_l1346_134654


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1346_134673

theorem sqrt_product_equality : Real.sqrt 72 * Real.sqrt 18 * Real.sqrt 8 = 72 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1346_134673


namespace NUMINAMATH_CALUDE_distance_from_origin_l1346_134692

theorem distance_from_origin (x y n : ℝ) : 
  x > 1 →
  y = 8 →
  (x - 1)^2 + (y - 6)^2 = 12^2 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (205 + 2 * Real.sqrt 140) :=
by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l1346_134692


namespace NUMINAMATH_CALUDE_jade_stone_volume_sum_l1346_134660

-- Define the weights per cubic inch
def jade_weight_per_cubic_inch : ℝ := 7
def stone_weight_per_cubic_inch : ℝ := 6

-- Define the edge length of the cubic stone
def edge_length : ℝ := 3

-- Define the total weight in taels
def total_weight : ℝ := 176

-- Theorem statement
theorem jade_stone_volume_sum (x y : ℝ) 
  (h1 : x + y = total_weight) 
  (h2 : x ≥ 0) 
  (h3 : y ≥ 0) : 
  x / jade_weight_per_cubic_inch + y / stone_weight_per_cubic_inch = edge_length ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_jade_stone_volume_sum_l1346_134660


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l1346_134630

theorem simplify_sqrt_difference : 
  (Real.sqrt 800 / Real.sqrt 50) - (Real.sqrt 288 / Real.sqrt 72) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l1346_134630


namespace NUMINAMATH_CALUDE_museum_trip_cost_l1346_134656

/-- The total cost of entrance tickets for a group of students and teachers -/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : ℕ :=
  (num_students + num_teachers) * ticket_price

/-- Theorem: The total cost for 20 students and 3 teachers with $5 tickets is $115 -/
theorem museum_trip_cost : total_cost 20 3 5 = 115 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_cost_l1346_134656


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1346_134678

/-- A quadratic function that opens upwards and passes through (0,1) -/
def QuadraticFunction (a b : ℝ) (h : a > 0) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + 1

theorem quadratic_function_properties (a b : ℝ) (h : a > 0) :
  (QuadraticFunction a b h) 0 = 1 ∧
  ∀ x y : ℝ, x < y → (QuadraticFunction a b h) x < (QuadraticFunction a b h) y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1346_134678


namespace NUMINAMATH_CALUDE_cindys_calculation_l1346_134640

theorem cindys_calculation (x : ℝ) : (x - 7) / 5 = 57 → (x - 5) / 7 = 41 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l1346_134640


namespace NUMINAMATH_CALUDE_fraction_problem_l1346_134651

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4 : ℝ) * (1/3 : ℝ) * F * N = 35)
  (h2 : (40/100 : ℝ) * N = 420) : 
  F = 2/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l1346_134651


namespace NUMINAMATH_CALUDE_polynomial_factorization_sum_l1346_134687

theorem polynomial_factorization_sum (a₁ a₂ c₁ b₂ c₂ : ℝ) 
  (h : ∀ x : ℝ, x^5 - x^4 + x^3 - x^2 + x - 1 = (x^3 + a₁*x^2 + a₂*x + c₁)*(x^2 + b₂*x + c₂)) :
  a₁*c₁ + b₂*c₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_sum_l1346_134687


namespace NUMINAMATH_CALUDE_tank_fill_time_main_theorem_l1346_134698

-- Define the fill rates of the pipes
def fill_rate_1 : ℚ := 1 / 18
def fill_rate_2 : ℚ := 1 / 60
def empty_rate : ℚ := 1 / 45

-- Define the combined rate
def combined_rate : ℚ := fill_rate_1 + fill_rate_2 - empty_rate

-- Theorem statement
theorem tank_fill_time :
  combined_rate = 1 / 20 := by sorry

-- Time to fill the tank
def fill_time : ℚ := 1 / combined_rate

-- Main theorem
theorem main_theorem :
  fill_time = 20 := by sorry

end NUMINAMATH_CALUDE_tank_fill_time_main_theorem_l1346_134698


namespace NUMINAMATH_CALUDE_min_odd_in_A_P_l1346_134642

/-- A polynomial of degree 8 -/
def Polynomial8 : Type := ℝ → ℝ

/-- The set A_P for a polynomial P -/
def A_P (P : Polynomial8) (c : ℝ) : Set ℝ :=
  {x : ℝ | P x = c}

/-- Theorem: For any polynomial P of degree 8, if 8 is in A_P, then A_P contains at least one odd number -/
theorem min_odd_in_A_P (P : Polynomial8) (h : 8 ∈ A_P P (P 8)) :
  ∃ (x : ℝ), x ∈ A_P P (P 8) ∧ ∃ (n : ℤ), x = 2 * n + 1 :=
sorry

end NUMINAMATH_CALUDE_min_odd_in_A_P_l1346_134642


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1346_134645

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 30

def divisors_of_30 : Set ℤ := {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial b₂ b₁ x = 0} = divisors_of_30 := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1346_134645


namespace NUMINAMATH_CALUDE_bulls_win_in_seven_games_l1346_134637

def probability_heat_wins : ℚ := 3/4

def games_to_win : ℕ := 4

def total_games : ℕ := 7

theorem bulls_win_in_seven_games :
  let probability_bulls_wins := 1 - probability_heat_wins
  let combinations := Nat.choose (total_games - 1) (games_to_win - 1)
  let probability_tied_after_six := combinations * (probability_bulls_wins ^ (games_to_win - 1)) * (probability_heat_wins ^ (games_to_win - 1))
  let probability_bulls_win_last := probability_bulls_wins
  probability_tied_after_six * probability_bulls_win_last = 540/16384 := by
  sorry

end NUMINAMATH_CALUDE_bulls_win_in_seven_games_l1346_134637


namespace NUMINAMATH_CALUDE_license_plate_theorem_l1346_134663

def vowels : Nat := 5
def consonants : Nat := 21
def odd_digits : Nat := 5
def even_digits : Nat := 5

def license_plate_count : Nat :=
  (vowels^2 + consonants^2) * odd_digits * even_digits^2

theorem license_plate_theorem :
  license_plate_count = 58250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l1346_134663


namespace NUMINAMATH_CALUDE_largest_b_value_l1346_134653

theorem largest_b_value (b : ℚ) (h : (3*b + 7) * (b - 2) = 9*b) : b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_b_value_l1346_134653


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l1346_134614

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + (a - 1) * x^2 + 48 * (a - 2) * x + b

-- State the theorem
theorem f_monotone_decreasing (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →  -- Symmetry about the origin
  (∀ x ∈ Set.Icc (-4 : ℝ) (4 : ℝ), ∀ y ∈ Set.Icc (-4 : ℝ) (4 : ℝ), x ≤ y → f a b x ≥ f a b y) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l1346_134614


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1346_134633

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 30 + (5 : ℚ) / 300 + (7 : ℚ) / 3000 = 0.119 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l1346_134633


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1346_134670

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 1 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 1 → y = x) ↔ 
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1346_134670


namespace NUMINAMATH_CALUDE_completing_square_proof_l1346_134610

theorem completing_square_proof (x : ℝ) : 
  x^2 - 4*x - 3 = 0 ↔ (x - 2)^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_proof_l1346_134610


namespace NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l1346_134616

/- Define the factorial function -/
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

/- Define a function to get the largest power of 2 that divides n! -/
def largestPowerOf2DividingFactorial (n : ℕ) : ℕ :=
  (n / 2) + (n / 4) + (n / 8) + (n / 16) + (n / 32)

/- Define a function to get the ones digit of a number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

/- Theorem statement -/
theorem ones_digit_of_largest_power_of_two_dividing_32_factorial :
  onesDigit (2^(largestPowerOf2DividingFactorial 32)) = 8 := by
  sorry


end NUMINAMATH_CALUDE_ones_digit_of_largest_power_of_two_dividing_32_factorial_l1346_134616


namespace NUMINAMATH_CALUDE_field_breadth_is_50_l1346_134627

/-- Proves that the breadth of a field is 50 meters given specific conditions -/
theorem field_breadth_is_50 (field_length : ℝ) (tank_length tank_width tank_depth : ℝ) 
  (field_rise : ℝ) (b : ℝ) : 
  field_length = 90 →
  tank_length = 25 →
  tank_width = 20 →
  tank_depth = 4 →
  field_rise = 0.5 →
  tank_length * tank_width * tank_depth = (field_length * b - tank_length * tank_width) * field_rise →
  b = 50 := by
  sorry

end NUMINAMATH_CALUDE_field_breadth_is_50_l1346_134627


namespace NUMINAMATH_CALUDE_johnny_wage_l1346_134603

/-- Given a total earning and hours worked, calculates the hourly wage -/
def hourly_wage (total_earning : ℚ) (hours_worked : ℚ) : ℚ :=
  total_earning / hours_worked

theorem johnny_wage :
  let total_earning : ℚ := 33/2  -- $16.5 represented as a rational number
  let hours_worked : ℚ := 2
  hourly_wage total_earning hours_worked = 33/4  -- $8.25 represented as a rational number
:= by sorry

end NUMINAMATH_CALUDE_johnny_wage_l1346_134603


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l1346_134613

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that 3(2-i) + i(3+2i) = 4 -/
theorem simplify_complex_expression : 3 * (2 - i) + i * (3 + 2 * i) = (4 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l1346_134613


namespace NUMINAMATH_CALUDE_paul_spent_three_tickets_l1346_134601

/-- Represents the number of tickets Paul spent on the Ferris wheel -/
def tickets_spent (initial : ℕ) (left : ℕ) : ℕ := initial - left

/-- Theorem stating that Paul spent 3 tickets on the Ferris wheel -/
theorem paul_spent_three_tickets :
  tickets_spent 11 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_paul_spent_three_tickets_l1346_134601


namespace NUMINAMATH_CALUDE_equation_has_integer_solution_l1346_134600

theorem equation_has_integer_solution : ∃ (x y : ℤ), x^2 - 2 = 7*y := by
  sorry

end NUMINAMATH_CALUDE_equation_has_integer_solution_l1346_134600


namespace NUMINAMATH_CALUDE_repeating_decimal_value_l1346_134675

/-- The decimal representation 0.7overline{23}15 as a rational number -/
def repeating_decimal : ℚ := 62519 / 66000

/-- Theorem stating that 0.7overline{23}15 is equal to 62519/66000 -/
theorem repeating_decimal_value : repeating_decimal = 0.7 + 0.015 + (23 : ℚ) / 990 := by
  sorry

#eval repeating_decimal

end NUMINAMATH_CALUDE_repeating_decimal_value_l1346_134675


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1346_134636

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∃ r, ∀ n, a (n + 1) = r * a n)
  (h_arith : a 1 + 2 * a 2 = a 3) :
  (a 9 + a 10) / (a 9 + a 8) = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1346_134636


namespace NUMINAMATH_CALUDE_integral_evaluation_l1346_134644

open Set
open MeasureTheory
open Interval

theorem integral_evaluation :
  ∫ x in (-2)..2, (x^2 * Real.sin x + Real.sqrt (4 - x^2)) = 2 * Real.pi :=
by
  have h1 : ∫ x in (-2)..2, x^2 * Real.sin x = 0 := sorry
  have h2 : ∫ x in (-2)..2, Real.sqrt (4 - x^2) = 2 * Real.pi := sorry
  sorry

end NUMINAMATH_CALUDE_integral_evaluation_l1346_134644


namespace NUMINAMATH_CALUDE_sequence_problem_l1346_134664

/-- Given a sequence {aₙ} that satisfies the recurrence relation
    aₙ₊₁/(n+1) = aₙ/n for all n, and a₅ = 15, prove that a₈ = 24. -/
theorem sequence_problem (a : ℕ → ℚ)
    (h1 : ∀ n, a (n + 1) / (n + 1) = a n / n)
    (h2 : a 5 = 15) :
    a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1346_134664


namespace NUMINAMATH_CALUDE_salad_dressing_total_l1346_134658

/-- Given a ratio of ingredients and the amount of one ingredient, 
    calculate the total amount of all ingredients. -/
def total_ingredients (ratio : List Nat) (water_amount : Nat) : Nat :=
  let total_parts := ratio.sum
  let part_size := water_amount / ratio.head!
  part_size * total_parts

/-- Theorem: For a salad dressing with a water:olive oil:salt ratio of 3:2:1 
    and 15 cups of water, the total amount of ingredients is 30 cups. -/
theorem salad_dressing_total : 
  total_ingredients [3, 2, 1] 15 = 30 := by
  sorry

#eval total_ingredients [3, 2, 1] 15

end NUMINAMATH_CALUDE_salad_dressing_total_l1346_134658


namespace NUMINAMATH_CALUDE_arithmetic_sequence_25th_term_l1346_134617

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.a₁ + (n - 1) * seq.d

theorem arithmetic_sequence_25th_term
  (seq : ArithmeticSequence)
  (h₃ : seq.nthTerm 3 = 7)
  (h₁₈ : seq.nthTerm 18 = 37) :
  seq.nthTerm 25 = 51 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_25th_term_l1346_134617


namespace NUMINAMATH_CALUDE_waiter_customer_count_l1346_134638

theorem waiter_customer_count (initial : Float) (lunch_rush : Float) (later : Float) :
  initial = 29.0 → lunch_rush = 20.0 → later = 34.0 →
  initial + lunch_rush + later = 83.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customer_count_l1346_134638


namespace NUMINAMATH_CALUDE_range_of_f_l1346_134696

def f (x : ℝ) : ℝ := x^2 - 4*x

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ f x = y} = Set.Icc (-4) 21 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1346_134696


namespace NUMINAMATH_CALUDE_division_problem_l1346_134684

theorem division_problem (n : ℕ) : n / 4 = 12 → n / 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1346_134684


namespace NUMINAMATH_CALUDE_no_simple_condition_for_equality_l1346_134648

/-- There is no simple general condition for when a + b + c² = (a+b)(a+c) for all real numbers a, b, and c. -/
theorem no_simple_condition_for_equality (a b c : ℝ) : 
  ¬ ∃ (simple_condition : Prop), simple_condition ↔ (a + b + c^2 = (a+b)*(a+c)) :=
sorry

end NUMINAMATH_CALUDE_no_simple_condition_for_equality_l1346_134648


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l1346_134615

theorem distinct_prime_factors_count (n : Nat) : n = 85 * 87 * 91 * 94 →
  Finset.card (Nat.factorization n).support = 8 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l1346_134615


namespace NUMINAMATH_CALUDE_no_cards_below_threshold_l1346_134626

def jungkook_card : ℚ := 0.8
def yoongi_card : ℚ := 1/2
def yoojeong_card : ℚ := 0.9
def yuna_card : ℚ := 1/3

def threshold : ℚ := 0.3

def count_below_threshold (cards : List ℚ) : ℕ :=
  (cards.filter (· < threshold)).length

theorem no_cards_below_threshold :
  count_below_threshold [jungkook_card, yoongi_card, yoojeong_card, yuna_card] = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_cards_below_threshold_l1346_134626


namespace NUMINAMATH_CALUDE_milk_processing_profit_comparison_l1346_134607

/-- Represents the profit calculation for a milk processing factory --/
theorem milk_processing_profit_comparison :
  let total_milk : ℝ := 9
  let fresh_milk_profit : ℝ := 500
  let yogurt_profit : ℝ := 1200
  let milk_slice_profit : ℝ := 2000
  let yogurt_capacity : ℝ := 3
  let milk_slice_capacity : ℝ := 1
  let processing_days : ℝ := 4

  let plan1_profit := milk_slice_capacity * processing_days * milk_slice_profit + 
                      (total_milk - milk_slice_capacity * processing_days) * fresh_milk_profit

  let plan2_milk_slice : ℝ := 1.5
  let plan2_yogurt : ℝ := 7.5
  let plan2_profit := plan2_milk_slice * milk_slice_profit + plan2_yogurt * yogurt_profit

  plan2_profit > plan1_profit ∧ 
  plan2_milk_slice + plan2_yogurt = total_milk ∧
  plan2_milk_slice / milk_slice_capacity + plan2_yogurt / yogurt_capacity = processing_days :=
by sorry

end NUMINAMATH_CALUDE_milk_processing_profit_comparison_l1346_134607


namespace NUMINAMATH_CALUDE_hydropump_volume_l1346_134646

/-- Represents the rate of water pumping in gallons per hour -/
def pump_rate : ℝ := 600

/-- Represents the time in hours -/
def pump_time : ℝ := 1.5

/-- Represents the volume of water pumped in gallons -/
def water_volume : ℝ := pump_rate * pump_time

theorem hydropump_volume : water_volume = 900 := by
  sorry

end NUMINAMATH_CALUDE_hydropump_volume_l1346_134646


namespace NUMINAMATH_CALUDE_f_at_2_l1346_134655

def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 1

theorem f_at_2 : f 2 = 15 := by sorry

end NUMINAMATH_CALUDE_f_at_2_l1346_134655
