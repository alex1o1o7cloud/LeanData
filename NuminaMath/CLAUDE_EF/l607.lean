import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_parabola_l607_60727

/-- The parabola equation -/
def parabola (x y n : ℝ) : Prop := y = x^2 + 4*x + n

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  ((x1 - x2)^2 + (y1 - y2)^2).sqrt

/-- P is the closest point to A on the parabola -/
def is_closest_point (m n : ℝ) : Prop :=
  ∀ x y : ℝ, parabola x y n → distance m 3 (-2) 0 ≤ distance x y (-2) 0

theorem closest_point_on_parabola (m n : ℝ) :
  parabola m 3 n → is_closest_point m n → m + n = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_parabola_l607_60727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_mass_proof_l607_60786

/-- The mass of each weight in grams -/
def weight_mass : ℝ := 400

/-- The distance the pencil is moved for the second balance in cm -/
def first_shift : ℝ := 9

/-- The additional distance the pencil is moved for the third balance in cm -/
def second_shift : ℝ := 5

/-- The mass of the ball in grams -/
def ball_mass : ℝ := 600

theorem ball_mass_proof :
  ∀ (l₁ l₂ : ℝ),
  l₁ > 0 ∧ l₂ > 0 →
  ball_mass * l₁ = weight_mass * l₂ →
  ball_mass * (l₁ + first_shift) = 2 * weight_mass * (l₂ - first_shift) →
  ball_mass * (l₁ + first_shift + second_shift) = 3 * weight_mass * (l₂ - first_shift - second_shift) →
  ball_mass = 600 := by
  sorry

#check ball_mass_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_mass_proof_l607_60786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_multiple_of_three_l607_60743

theorem expression_multiple_of_three (a b : ℕ) :
  ∃ k : ℤ, (Real.sqrt 5 - Real.sqrt 2) * (Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ)) = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_multiple_of_three_l607_60743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l607_60728

theorem system_solution :
  ∀ x y : ℝ,
  (x > 0 ∧ y > 0) →
  (Real.log y / Real.log x - Real.log x / Real.log y = 8/3 ∧ x * y = 16) →
  ((x = 8 ∧ y = 2) ∨ (x = 1/4 ∧ y = 64)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l607_60728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l607_60749

noncomputable def f (x : ℝ) := Real.cos x - Real.cos (x + Real.pi / 2)

theorem f_properties : 
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ 
   ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧ 
  (∃ x_max, f x_max = Real.sqrt 2 ∧ ∀ x, f x ≤ Real.sqrt 2) ∧
  (∃ x_min, f x_min = -Real.sqrt 2 ∧ ∀ x, f x ≥ -Real.sqrt 2) ∧
  (∀ k : ℤ, f (2 * k * Real.pi + Real.pi / 4) = Real.sqrt 2) ∧
  (∀ k : ℤ, f (2 * k * Real.pi - 3 * Real.pi / 4) = -Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l607_60749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_iff_even_l607_60754

/-- A valid grid arrangement for a given size n -/
def ValidArrangement (n : ℕ) (grid : Fin n → Fin n → ℕ) : Prop :=
  -- n is greater than 1
  n > 1 ∧
  -- All numbers from 1 to n^2 are present in the grid
  (∀ k, 1 ≤ k ∧ k ≤ n^2 → ∃ i j, grid i j = k) ∧
  -- Every two consecutive numbers are in adjacent cells
  (∀ k, 1 ≤ k ∧ k < n^2 → ∃ i₁ j₁ i₂ j₂, 
    grid i₁ j₁ = k ∧ grid i₂ j₂ = k + 1 ∧ 
    ((i₁ = i₂ ∧ (j₁.val + 1 = j₂.val ∨ j₂.val + 1 = j₁.val)) ∨ 
     (j₁ = j₂ ∧ (i₁.val + 1 = i₂.val ∨ i₂.val + 1 = i₁.val)))) ∧
  -- Numbers with the same remainder when divided by n are in different rows and columns
  (∀ i₁ j₁ i₂ j₂, grid i₁ j₁ % n = grid i₂ j₂ % n → 
    (i₁ ≠ i₂ ∧ j₁ ≠ j₂) ∨ (i₁ = i₂ ∧ j₁ = j₂))

/-- The main theorem stating that a valid arrangement exists if and only if n is even -/
theorem valid_arrangement_iff_even (n : ℕ) : 
  (∃ grid : Fin n → Fin n → ℕ, ValidArrangement n grid) ↔ Even n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangement_iff_even_l607_60754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sts_income_tax_lowest_l607_60715

/-- Represents the financial data of a business -/
structure FinancialData where
  annual_income : ℝ
  annual_expenses : ℝ
  expense_payment_ratio : ℝ
  insurance_contributions : ℝ

/-- Represents the tax rates for different taxation systems -/
structure TaxRates where
  main_system : ℝ
  sts_income : ℝ
  sts_income_minus_expenses : ℝ
  minimum_tax : ℝ

/-- Calculates the tax under the main taxation system -/
noncomputable def main_system_tax (data : FinancialData) (rates : TaxRates) : ℝ :=
  (data.annual_income - data.annual_expenses) * rates.main_system

/-- Calculates the tax under the STS - Income system -/
noncomputable def sts_income_tax (data : FinancialData) (rates : TaxRates) : ℝ :=
  let base_tax := data.annual_income * rates.sts_income
  let reduction := min (0.5 * base_tax) (data.insurance_contributions * data.expense_payment_ratio)
  base_tax - reduction

/-- Calculates the tax under the STS - Income minus expenses system -/
noncomputable def sts_income_minus_expenses_tax (data : FinancialData) (rates : TaxRates) : ℝ :=
  max ((data.annual_income - data.annual_expenses * data.expense_payment_ratio) * rates.sts_income_minus_expenses)
      (data.annual_income * rates.minimum_tax)

/-- Theorem: STS - Income tax is the lowest among the three taxation systems -/
theorem sts_income_tax_lowest (data : FinancialData) (rates : TaxRates) :
  sts_income_tax data rates ≤ min (main_system_tax data rates) (sts_income_minus_expenses_tax data rates) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sts_income_tax_lowest_l607_60715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_universal_knower_l607_60717

/-- Represents a company with a specific knowledge relationship between people -/
structure Company (n : ℕ) where
  people : Finset (Fin (2*n+1))
  knows : Fin (2*n+1) → Fin (2*n+1) → Prop
  all_people : people.card = 2*n+1
  knowledge_condition : ∀ (S : Finset (Fin (2*n+1))), S.card = n → 
    ∃ (p : Fin (2*n+1)), p ∉ S ∧ (∀ q, q ∈ S → knows p q)

/-- The main theorem stating that there exists a person who knows everyone -/
theorem exists_universal_knower (n : ℕ) (company : Company n) : 
  ∃ (p : Fin (2*n+1)), ∀ (q : Fin (2*n+1)), p ≠ q → company.knows p q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_universal_knower_l607_60717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_percentage_l607_60761

/-- Proves that the percentage reduction in prices was 20% --/
theorem price_reduction_percentage (shirt_price jacket_price : ℝ) 
  (num_shirts num_jackets : ℕ) (total_paid : ℝ) 
  (h1 : shirt_price = 60)
  (h2 : jacket_price = 90)
  (h3 : num_shirts = 5)
  (h4 : num_jackets = 10)
  (h5 : total_paid = 960)
  : (1 - total_paid / (shirt_price * num_shirts + jacket_price * num_jackets)) * 100 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_percentage_l607_60761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_power_function_l607_60776

-- Define the power function
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 5*m + 7) * x^(m^2 - 6)

-- State the theorem
theorem monotonic_increasing_power_function :
  ∀ m : ℝ, (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) → m = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_power_function_l607_60776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wronskian_exponential_l607_60726

noncomputable def y₁ (x k₁ : ℝ) : ℝ := Real.exp (k₁ * x)
noncomputable def y₂ (x k₂ : ℝ) : ℝ := Real.exp (k₂ * x)
noncomputable def y₃ (x k₃ : ℝ) : ℝ := Real.exp (k₃ * x)

noncomputable def wronskian (f g h : ℝ → ℝ) (x : ℝ) : ℝ :=
  let f' := deriv f
  let g' := deriv g
  let h' := deriv h
  let f'' := deriv f'
  let g'' := deriv g'
  let h'' := deriv h'
  Matrix.det !![f x, g x, h x;
                 f' x, g' x, h' x;
                 f'' x, g'' x, h'' x]

theorem wronskian_exponential (x k₁ k₂ k₃ : ℝ) :
  wronskian (y₁ · k₁) (y₂ · k₂) (y₃ · k₃) x =
  Real.exp ((k₁ + k₂ + k₃) * x) * (k₂ - k₁) * (k₃ - k₁) * (k₃ - k₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wronskian_exponential_l607_60726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l607_60745

theorem triangle_side_length (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  Real.sin A = 1 / 3 ∧
  b = Real.sqrt 3 * Real.sin B →
  a = Real.sqrt 3 / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l607_60745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equal_width_is_diameter_l607_60714

/-- A curve of constant width -/
class ConstantWidthCurve (α : Type*) [MetricSpace α] where
  width : ℝ
  is_constant_width : ∀ (x y : α), dist x y ≤ width

/-- A chord of a curve -/
def Chord (α : Type*) [MetricSpace α] (curve : Set α) (a b : α) :=
  a ∈ curve ∧ b ∈ curve

/-- A diameter of a curve -/
def Diameter (α : Type*) [MetricSpace α] (curve : Set α) (a b : α) :=
  Chord α curve a b ∧ ∀ (x y : α), x ∈ curve → y ∈ curve → dist x y ≤ dist a b

theorem chord_equal_width_is_diameter
  {α : Type*} [MetricSpace α] [c : ConstantWidthCurve α] (curve : Set α) (a b : α) :
  Chord α curve a b → dist a b = c.width → Diameter α curve a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equal_width_is_diameter_l607_60714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_and_height_l607_60762

/-- Definition of a tetrahedron with given vertices -/
structure Tetrahedron where
  A₁ : ℝ × ℝ × ℝ
  A₂ : ℝ × ℝ × ℝ
  A₃ : ℝ × ℝ × ℝ
  A₄ : ℝ × ℝ × ℝ

/-- Calculate the volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Calculate the height from A₄ to the face A₁A₂A₃ -/
def tetrahedronHeight (t : Tetrahedron) : ℝ :=
  sorry

theorem tetrahedron_volume_and_height :
  let t : Tetrahedron := {
    A₁ := (-4, 2, 6),
    A₂ := (2, -3, 0),
    A₃ := (-10, 5, 8),
    A₄ := (-5, 2, -4)
  }
  volume t = 56 / 3 ∧ tetrahedronHeight t = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_and_height_l607_60762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_adjustment_l607_60725

/-- Given a price that has undergone specific percentage changes, calculate the percentage decrease needed to return to the original price. -/
theorem price_adjustment (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_changes := initial_price * 1.25 * 0.85 * 1.1
  let required_decrease := (price_after_changes - initial_price) / price_after_changes
  abs (required_decrease - 0.1443) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_adjustment_l607_60725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_six_divisor_count_l607_60706

theorem remainder_six_divisor_count : ∃! n : ℕ, n > 6 ∧ 53 % n = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_six_divisor_count_l607_60706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ps_divided_by_qr_equals_sqrt_three_l607_60752

/-- A triangle is equilateral if all its sides have equal length -/
def IsEquilateral (A B C : ℝ × ℝ) : Prop :=
  ‖B - A‖ = ‖C - B‖ ∧ ‖C - B‖ = ‖A - C‖

/-- Given two equilateral triangles PQR and QRS sharing side QR, 
    prove that the ratio of PS to QR is √3 -/
theorem ps_divided_by_qr_equals_sqrt_three 
  (P Q R S : ℝ × ℝ) -- Points in 2D plane
  (h1 : IsEquilateral P Q R) -- PQR is equilateral
  (h2 : IsEquilateral Q R S) -- QRS is equilateral
  : ‖S - P‖ / ‖R - Q‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ps_divided_by_qr_equals_sqrt_three_l607_60752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l607_60767

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / (x - 1)

-- State the theorem
theorem f_properties :
  -- f is decreasing on (1, +∞)
  (∀ x y : ℝ, 1 < x ∧ x < y → f x > f y) ∧
  -- f(x) > 3/x for all x > 1
  (∀ x : ℝ, x > 1 → f x > 3 / x) ∧
  -- 3 is the largest integer k for which f(x) > k/x holds
  (∀ k : ℕ, (∀ x : ℝ, x > 1 → f x > (k : ℝ) / x) → k ≤ 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l607_60767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l607_60718

/-- A point in a 2D Euclidean space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Function to calculate the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Function to check if a point is inside a triangle -/
def isInside (F : Point) (t : Triangle) : Prop := sorry

/-- Function to check if sides of a triangle subtend the same angles at a point -/
def subtendSameAngles (F : Point) (t : Triangle) : Prop := sorry

/-- Function to find the intersection point of two lines -/
noncomputable def lineIntersection (A B C D : Point) : Point := sorry

/-- The main theorem -/
theorem triangle_inequality (t : Triangle) (F : Point) :
  isInside F t →
  subtendSameAngles F t →
  let D := lineIntersection t.B F t.A t.C
  let E := lineIntersection t.C F t.A t.B
  distance t.A t.B + distance t.A t.C ≥ 4 * distance D E := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l607_60718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_max_value_a_l607_60798

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - k) * Real.exp x

-- Theorem for the minimum value of f in [0,1]
theorem min_value_f (k : ℝ) :
  ∃ (m : ℝ), ∀ x ∈ Set.Icc 0 1, f k x ≥ m ∧
   ((k ≤ 1 → m = -k) ∧
   (1 < k ∧ k < 2 → m = -Real.exp (k - 1)) ∧
   (k ≥ 2 → m = (1 - k) * Real.exp 1)) :=
by sorry

-- Theorem for the maximum value of a
theorem max_value_a :
  (∀ a : ℤ, (∀ x : ℝ, x > 0 → x * Real.exp x - a * Real.exp x + a > 0) →
   a ≤ 3) ∧
  (∀ x : ℝ, x > 0 → x * Real.exp x - 3 * Real.exp x + 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_max_value_a_l607_60798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_theorem_l607_60769

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Distance between two points in ℝ² -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Predicate for two circles intersecting -/
def circles_intersect (c1 c2 : Circle) : Prop :=
  let d := distance c1.center c2.center
  d < c1.radius + c2.radius ∧ d > |c1.radius - c2.radius|

/-- First circle: x^2 + (y-2)^2 = 1 -/
def circle1 : Circle :=
  { center := (0, 2), radius := 1 }

/-- Second circle: x^2 + y^2 + 4x + 2y - 11 = 0 -/
def circle2 : Circle :=
  { center := (-2, -1), radius := 4 }

/-- Theorem stating that the two given circles intersect -/
theorem circles_intersect_theorem : circles_intersect circle1 circle2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_theorem_l607_60769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_values_l607_60789

theorem theta_values (θ : Real) (h1 : 0 ≤ θ ∧ θ ≤ π/2) 
  (h2 : |Real.cos θ * Real.sin θ + Real.sin θ * Real.cos θ - 1| / Real.sqrt ((Real.sin θ)^2 + (Real.cos θ)^2) = 1/2) :
  θ = π/12 ∨ θ = 5*π/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_values_l607_60789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l607_60778

/-- A polynomial P satisfying P(2x) = P'(x) P''(x) for all real x must be of the form
    P(x) = (4/9)x³ + (4/3)x² + cx for some constant c. -/
theorem polynomial_functional_equation (P : Polynomial ℝ) :
  (∀ x : ℝ, P.eval (2 * x) = (P.derivative.eval x) * (P.derivative.derivative.eval x)) →
  ∃ c : ℝ, P = Polynomial.monomial 3 (4/9) + Polynomial.monomial 2 (4/3) + Polynomial.monomial 1 c :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_functional_equation_l607_60778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_school_distance_l607_60770

/-- The distance from Xiaoming's home to school -/
noncomputable def distance : ℝ := 4200

/-- The time it takes Xiaoming to walk to school at 60 meters per minute -/
noncomputable def time_fast : ℝ := (distance / 60) - 10

/-- The time it takes Xiaoming to walk to school at 50 meters per minute -/
noncomputable def time_slow : ℝ := (distance / 50) + 4

/-- Theorem stating that the distance from Xiaoming's home to school is 4200 meters -/
theorem xiaoming_school_distance :
  distance = 4200 ∧ time_fast * 60 = distance ∧ time_slow * 50 = distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaoming_school_distance_l607_60770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_attendance_l607_60791

theorem conference_attendance
  (total_participants : ℕ)
  (constant_attendance : Fin 3 → Finset ℕ)
  (first_not_return : Finset ℕ)
  (second_only : Finset ℕ)
  (third_new : Finset ℕ)
  (h_total : total_participants = 300)
  (h_constant : ∀ i : Fin 3, (constant_attendance i).card = (constant_attendance 0).card)
  (h_first_not_return : first_not_return.card = (constant_attendance 0).card / 2)
  (h_second_only : second_only.card = (constant_attendance 1).card / 3)
  (h_third_new : third_new.card = (constant_attendance 2).card / 4)
  (h_at_least_one : ∀ p, p < total_participants → 
    p ∈ constant_attendance 0 ∨ p ∈ constant_attendance 1 ∨ p ∈ constant_attendance 2) :
  ((constant_attendance 0).card = 156 ∧ 
   (constant_attendance 0 ∩ constant_attendance 1 ∩ constant_attendance 2).card = 37) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_attendance_l607_60791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_vet_costs_dog_vet_costs_proof_l607_60747

theorem dog_vet_costs (num_appointments : ℕ) (appointment_cost : ℕ) 
  (insurance_cost : ℕ) (insurance_coverage : ℚ) : Prop :=
  let first_appointment_cost := appointment_cost
  let insurance_covered_appointments := num_appointments - 1
  let discounted_appointment_cost := appointment_cost - (appointment_cost * insurance_coverage).floor
  let total_discounted_cost := insurance_covered_appointments * discounted_appointment_cost
  let total_cost := first_appointment_cost + total_discounted_cost + insurance_cost
  
  -- Theorem statement
  (num_appointments = 3 ∧ 
  appointment_cost = 400 ∧ 
  insurance_cost = 100 ∧ 
  insurance_coverage = 4/5) → 
  total_cost = 660

-- Proof
theorem dog_vet_costs_proof : dog_vet_costs 3 400 100 (4/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_vet_costs_dog_vet_costs_proof_l607_60747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l607_60758

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2) / h.a

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) (F A B : Point)
  (h_right_focus : F.x > 0 ∧ F.y = 0)
  (h_asymptote : A.y = (h.b / h.a) * A.x ∧ B.y = -(h.b / h.a) * B.x)
  (h_perpendicular : (A.y - F.y) * (A.x - F.x) = -(h.a / h.b) * (A.x - F.x)^2)
  (h_vector_relation : 2 * (F.x - A.x) = B.x - F.x ∧ 2 * (F.y - A.y) = B.y - F.y) :
  eccentricity h = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l607_60758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_garden_area_ratio_l607_60788

theorem circular_garden_area_ratio (r : ℝ) (h : r > 0) :
  (π * r^2) / (π * (2*r)^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_garden_area_ratio_l607_60788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_quadratic_root_coefficients_l607_60704

-- Part 1
noncomputable def z : ℂ := (1 + 2*Complex.I) / (3 - 4*Complex.I)

theorem modulus_of_z : Complex.abs z = Real.sqrt 5 / 5 := by sorry

-- Part 2
theorem quadratic_root_coefficients (p q : ℝ) 
  (h : (2 - 3*Complex.I)^2 + p*(2 - 3*Complex.I) + q = 0) : 
  p = 4 ∧ q = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_quadratic_root_coefficients_l607_60704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_ending_888_l607_60748

theorem smallest_cube_ending_888 :
  ∀ n : ℕ, n > 0 → n^3 % 1000 = 888 → n ≥ 192 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cube_ending_888_l607_60748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l607_60731

/-- Given a cone with lateral surface area 3π and central angle 2π/3, its volume is (2√2π)/3 -/
theorem cone_volume (l r h : ℝ) : 
  (1/2 * (2*Real.pi/3) * l^2 = 3*Real.pi) → 
  (2*Real.pi*r = 2*Real.pi/3 * l) → 
  (h^2 = l^2 - r^2) → 
  (1/3 * Real.pi * r^2 * h = 2*Real.sqrt 2*Real.pi/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l607_60731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_of_curve_C_polar_coordinates_of_T_l607_60710

-- Define the curve C
noncomputable def curve_C (α : Real) : Real × Real :=
  (Real.cos α, 1 + Real.sin α)

-- Define the domain of α
def α_domain : Set Real :=
  { α | -Real.pi/2 ≤ α ∧ α ≤ Real.pi/2 }

-- Define the polar equation
noncomputable def polar_equation (θ : Real) : Real :=
  2 * Real.sin θ

-- Define the domain of θ
def θ_domain : Set Real :=
  { θ | 0 ≤ θ ∧ θ ≤ Real.pi/2 }

-- Theorem 1: Polar equation of curve C
theorem polar_equation_of_curve_C :
  ∀ (α : Real), α ∈ α_domain →
  ∃ (θ : Real), θ ∈ θ_domain ∧
  (polar_equation θ * Real.cos θ, polar_equation θ * Real.sin θ) = curve_C α :=
sorry

-- Theorem 2: Polar coordinates of point T
theorem polar_coordinates_of_T :
  ∃ (T : Real × Real), T ∈ (Set.range curve_C) ∧
  (T.1^2 + T.2^2 = 3) ∧
  T = (Real.sqrt 3 * Real.cos (Real.pi/3), Real.sqrt 3 * Real.sin (Real.pi/3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_of_curve_C_polar_coordinates_of_T_l607_60710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_twelve_cards_l607_60785

def probability_empty_bag (n : ℕ) : ℚ :=
  if n ≤ 2 then 1
  else (Finset.range (n - 2)).prod (fun k => (3 : ℚ) / (2 * (k + 3) - 1)) * (3 : ℚ) / 5

theorem probability_twelve_cards :
  probability_empty_bag 6 = 9 / 385 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_twelve_cards_l607_60785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_sum_is_three_l607_60740

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a three-digit positive integer -/
structure ThreeDigitInt where
  hundreds : Digit
  tens : Digit
  ones : Digit
  pos : hundreds.val ≠ 0

/-- Represents a two-digit positive integer -/
structure TwoDigitInt where
  tens : Digit
  ones : Digit
  pos : tens.val ≠ 0

/-- The sum of two integers -/
def sum (a : ThreeDigitInt) (b : TwoDigitInt) : Nat :=
  a.hundreds.val * 100 + a.tens.val * 10 + a.ones.val +
  b.tens.val * 10 + b.ones.val

/-- The sum of the digits of a three-digit number -/
def digitSum (n : Nat) : Nat :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

/-- All digits in the two numbers are different -/
def allDigitsDifferent (a : ThreeDigitInt) (b : TwoDigitInt) : Prop :=
  a.hundreds ≠ a.tens ∧ a.hundreds ≠ a.ones ∧ a.hundreds ≠ b.tens ∧ a.hundreds ≠ b.ones ∧
  a.tens ≠ a.ones ∧ a.tens ≠ b.tens ∧ a.tens ≠ b.ones ∧
  a.ones ≠ b.tens ∧ a.ones ≠ b.ones ∧
  b.tens ≠ b.ones

theorem smallest_digit_sum_is_three
  (a : ThreeDigitInt) (b : TwoDigitInt)
  (h1 : allDigitsDifferent a b)
  (h2 : sum a b < 1000) :
  ∃ (a' : ThreeDigitInt) (b' : TwoDigitInt),
    allDigitsDifferent a' b' ∧
    sum a' b' < 1000 ∧
    digitSum (sum a' b') = 3 ∧
    ∀ (a'' : ThreeDigitInt) (b'' : TwoDigitInt),
      allDigitsDifferent a'' b'' →
      sum a'' b'' < 1000 →
      digitSum (sum a'' b'') ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_sum_is_three_l607_60740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sine_graph_l607_60780

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 3)

theorem shift_sine_graph (x : ℝ) : g x = f (x + Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sine_graph_l607_60780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l607_60735

/-- Given a function f, prove that the area between y = 2f(2x-1) and the x-axis 
    from x=0 to x=2.5 is equal to the integral of f from 0 to 5 -/
theorem area_equality (f : ℝ → ℝ) (h : ∫ x in (0 : ℝ)..(5 : ℝ), f x = 8) :
  ∫ x in (0 : ℝ)..(5 : ℝ)/2, 2 * f (2*x - 1) = ∫ x in (0 : ℝ)..(5 : ℝ), f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_equality_l607_60735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harold_money_l607_60711

theorem harold_money (d e f g h : ℤ) : 
  d + e + f + g + h = 72 →
  abs (d - e) = 15 →
  abs (e - f) = 9 →
  abs (f - g) = 7 →
  abs (g - h) = 6 →
  abs (h - d) = 13 →
  h = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_harold_money_l607_60711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_is_48_leo_final_is_correct_l607_60733

/-- The total amount of money Leo and Ryan have together -/
def total_amount : ℚ := 48

/-- Ryan's share of the total amount -/
def ryan_share : ℚ := (2 : ℚ) / 3 * total_amount

/-- Leo's share of the total amount -/
def leo_share : ℚ := total_amount - ryan_share

/-- The amount Ryan owed Leo -/
def ryan_debt : ℚ := 10

/-- The amount Leo owed Ryan -/
def leo_debt : ℚ := 7

/-- Leo's final amount after settling debts -/
def leo_final : ℚ := 19

theorem total_is_48 : total_amount = 48 := by
  -- The proof is trivial since we defined total_amount as 48
  rfl

theorem leo_final_is_correct : leo_share + ryan_debt - leo_debt = leo_final := by
  -- We'll use 'sorry' to skip the actual proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_is_48_leo_final_is_correct_l607_60733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_inequality_l607_60732

theorem sin_cos_sum_inequality (α β a b : Real) : 
  0 < α → α < β → β < π/4 → 
  Real.sin α + Real.cos α = a → 
  Real.sin β + Real.cos β = b → 
  a < b := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_inequality_l607_60732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_properties_l607_60792

theorem logarithm_properties (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_a_gt_1 : a > 1)
  (h_log_ab : (Real.log a) / (Real.log 3) = (Real.log b) / (Real.log 5))
  (h_log_bc : (Real.log b) / (Real.log 3) = (Real.log c) / (Real.log 5)) :
  ((Real.log b) / (Real.log a) = (Real.log 5) / (Real.log 3)) ∧
  (a * c > b ^ 2) ∧
  ((2:ℝ) ^ a + (2:ℝ) ^ c > (2:ℝ) ^ (b + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_properties_l607_60792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_third_l607_60793

/-- The area enclosed by the curves y = x^2 and y = √x -/
noncomputable def area_between_curves : ℝ := ∫ x in (0 : ℝ)..1, (Real.sqrt x - x^2)

/-- Theorem stating that the area enclosed by the curves y = x^2 and y = √x is 1/3 -/
theorem area_is_one_third : area_between_curves = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_one_third_l607_60793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l607_60700

/-- The function f(x) = x^3 - (3/2)x^2 + a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - (3/2) * x^2 + a

theorem min_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ 3) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 3) →
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = (1/2 : ℝ) ∧
  ∀ y ∈ Set.Icc (-1 : ℝ) 1, f a y ≥ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l607_60700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_max_l607_60746

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where the area S = (1/2) * c^2 and ab = √2, 
    prove that the maximum value of a^2 + b^2 + c^2 is 4. -/
theorem triangle_side_sum_max (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (1/2) * c^2 = (1/2) * a * b * Real.sin C →
  a * b = Real.sqrt 2 →
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    (1/2) * z^2 = (1/2) * x * y * Real.sin C ∧
    x * y = Real.sqrt 2 ∧
    x^2 + y^2 + z^2 > a^2 + b^2 + c^2) →
  a^2 + b^2 + c^2 ≤ 4 :=
by
  sorry

#check triangle_side_sum_max

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_max_l607_60746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inverse_shifted_l607_60779

theorem sine_inverse_shifted (x y : ℝ) :
  y = Real.sin x →
  (1992 + 1/2) * Real.pi ≤ x ∧ x ≤ (1993 + 1/2) * Real.pi →
  x = 1993 * Real.pi + Real.arcsin y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inverse_shifted_l607_60779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_squared_in_second_quadrant_l607_60766

noncomputable def z : ℂ := Complex.exp (Complex.I * (Real.pi / 3))

theorem z_squared_in_second_quadrant : 
  let z_squared := z^2
  Complex.re z_squared < 0 ∧ Complex.im z_squared > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_squared_in_second_quadrant_l607_60766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l₃_l607_60757

/-- Line l₁ with equation 4x - 3y = 2 -/
def l₁ (x y : ℝ) : Prop := 4 * x - 3 * y = 2

/-- Point P -/
def P : ℝ × ℝ := (-2, -3)

/-- Line l₂ with equation y = 2 -/
def l₂ (y : ℝ) : Prop := y = 2

/-- Point Q is the intersection of l₁ and l₂ -/
def Q : ℝ × ℝ := (2, 2)

/-- Line l₃ passes through P and has positive slope -/
def l₃ (m : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = m * (x - P.1) ∧ m > 0

/-- Point R is the intersection of l₂ and l₃ -/
noncomputable def R (m : ℝ) : ℝ × ℝ :=
  (P.1 + (2 - P.2) / m, 2)

/-- The area of triangle PQR is 6 -/
def triangle_area (m : ℝ) : Prop :=
  abs ((P.1 * (Q.2 - (R m).2) + Q.1 * ((R m).2 - P.2) + (R m).1 * (P.2 - Q.2)) / 2) = 6

theorem slope_of_l₃ :
  ∃ m : ℝ, (l₃ m P.1 P.2 ∧ triangle_area m) → (m = 25/32 ∨ m = 25/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_l₃_l607_60757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l607_60734

/-- Represents the properties of a water tank system -/
structure TankSystem where
  capacity : ℝ
  leakTime : ℝ
  inletRate : ℝ

/-- Calculates the time it takes for the tank to empty with both inlet and leak -/
noncomputable def emptyingTime (t : TankSystem) : ℝ :=
  t.capacity / (t.capacity / t.leakTime - t.inletRate * 60)

/-- Theorem stating that for the given tank system, the emptying time is 8 hours -/
theorem tank_emptying_time :
  let t : TankSystem := {
    capacity := 5760,
    leakTime := 6,
    inletRate := 4
  }
  emptyingTime t = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_emptying_time_l607_60734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_statements_false_l607_60796

/-- Represents a statement on the card -/
inductive Statement
| One
| Two
| Three
| Four

/-- Represents the truth value of a statement -/
def isFalse : Statement → Prop := sorry

/-- The total number of statements on the card -/
def totalStatements : Nat := 4

/-- The number of false statements on the card -/
def falseStatementCount : Nat := 3

/-- Theorem stating that exactly three statements on the card are false -/
theorem three_statements_false :
  (∃ (s1 s2 s3 : Statement), s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧
    isFalse s1 ∧ isFalse s2 ∧ isFalse s3) ∧
  (∀ (s : Statement), isFalse s → 
    ∃ (t1 t2 : Statement), t1 ≠ t2 ∧ s ≠ t1 ∧ s ≠ t2 ∧
    isFalse t1 ∧ isFalse t2) ∧
  (∃ (s : Statement), ¬isFalse s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_statements_false_l607_60796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overtake_time_approx_l607_60763

/-- The time it takes for person B to overtake person A -/
noncomputable def overtake_time (speed_a speed_b : ℝ) (head_start : ℝ) : ℝ :=
  let distance_a := speed_a * head_start
  let relative_speed := speed_b - speed_a
  distance_a / relative_speed

/-- The theorem stating the time it takes for B to overtake A -/
theorem overtake_time_approx :
  let speed_a := (4 : ℝ)
  let speed_b := (4.555555555555555 : ℝ)
  let head_start := (0.5 : ℝ)
  abs (overtake_time speed_a speed_b head_start - 3.57) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overtake_time_approx_l607_60763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlational_relationships_l607_60708

-- Define the concept of a relationship
structure Relationship :=
  (factor1 : String)
  (factor2 : String)

-- Define what it means for a relationship to be correlational
def is_correlational (r : Relationship) : Prop := 
  (r.factor1 ≠ r.factor2) ∧ 
  (∃ (x : ℚ), 0 < x ∧ x < 1)

-- Define the relationships
def learning_attitude_performance : Relationship := ⟨"student's learning attitude", "academic performance"⟩
def teaching_quality_performance : Relationship := ⟨"teacher's teaching quality", "students' academic performance"⟩
def height_performance : Relationship := ⟨"student's height", "academic performance"⟩
def economic_conditions_performance : Relationship := ⟨"family economic conditions", "students' academic performance"⟩

-- State the theorem
theorem correlational_relationships :
  is_correlational learning_attitude_performance ∧
  is_correlational teaching_quality_performance ∧
  ¬is_correlational height_performance ∧
  ¬is_correlational economic_conditions_performance :=
by
  sorry

#check correlational_relationships

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlational_relationships_l607_60708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_catches_cara_jim_catches_cara_in_minutes_l607_60768

/-- Represents the running scenario of Jim and Cara -/
structure RunningScenario where
  jim_speed : ℝ
  cara_speed : ℝ
  initial_run_time : ℝ
  jim_stretch_time : ℝ

/-- Calculates the time it takes Jim to catch up to Cara -/
noncomputable def catch_up_time (scenario : RunningScenario) : ℝ :=
  let distance_gap := scenario.cara_speed * scenario.jim_stretch_time
  let speed_difference := scenario.jim_speed - scenario.cara_speed
  distance_gap / speed_difference

/-- Theorem stating that Jim will catch up to Cara in 1.5 hours -/
theorem jim_catches_cara (scenario : RunningScenario) 
  (h1 : scenario.jim_speed = 6)
  (h2 : scenario.cara_speed = 5)
  (h3 : scenario.initial_run_time = 0.5)
  (h4 : scenario.jim_stretch_time = 0.3) :
  catch_up_time scenario = 1.5 := by
  sorry

/-- Converts the catch-up time from hours to minutes -/
noncomputable def catch_up_time_minutes (scenario : RunningScenario) : ℝ :=
  catch_up_time scenario * 60

/-- Theorem stating that Jim will catch up to Cara in 90 minutes -/
theorem jim_catches_cara_in_minutes (scenario : RunningScenario) 
  (h1 : scenario.jim_speed = 6)
  (h2 : scenario.cara_speed = 5)
  (h3 : scenario.initial_run_time = 0.5)
  (h4 : scenario.jim_stretch_time = 0.3) :
  catch_up_time_minutes scenario = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_catches_cara_jim_catches_cara_in_minutes_l607_60768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l607_60783

noncomputable def f (x : ℝ) := Real.sin (-2 * x) + Real.cos (-2 * x)

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    (T > 0) ∧
    (∀ S, S > 0 → (∀ x, f (x + S) = f x) → T ≤ S) ∧
    (∀ x ∈ Set.Icc (-Real.pi/8) (3*Real.pi/8), ∀ y ∈ Set.Icc (-Real.pi/8) (3*Real.pi/8), x < y → f y < f x) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≥ -Real.sqrt 2) ∧
    (f (3*Real.pi/8) = -Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l607_60783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l607_60797

/-- The parabola y^2 = 4x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

/-- The point A -/
def A : ℝ × ℝ := (0, -1)

/-- The line x = -1 -/
def line (p : ℝ × ℝ) : Prop := p.1 = -1

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Distance from a point to the line x = -1 -/
def distanceToLine (p : ℝ × ℝ) : ℝ := abs (p.1 + 1)

/-- Sum of distances from a point to A and to the line -/
noncomputable def sumOfDistances (p : ℝ × ℝ) : ℝ := distance p A + distanceToLine p

theorem min_sum_of_distances :
  ∃ (m : ℝ), ∀ (p : ℝ × ℝ), parabola p → sumOfDistances p ≥ m ∧ 
  ∃ (q : ℝ × ℝ), parabola q ∧ sumOfDistances q = m ∧ m = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l607_60797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_weight_l607_60729

/-- The weight of the mixture of two brands of vegetable ghee -/
theorem mixture_weight (weight_a weight_b : ℝ) (ratio_a ratio_b total_volume : ℝ) : 
  weight_a = 900 →
  weight_b = 850 →
  ratio_a = 3 →
  ratio_b = 2 →
  total_volume = 4 →
  ((ratio_a / (ratio_a + ratio_b)) * total_volume * weight_a + 
   (ratio_b / (ratio_a + ratio_b)) * total_volume * weight_b) / 1000 = 3.52 := by
  intros h1 h2 h3 h4 h5
  sorry

#check mixture_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_weight_l607_60729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l607_60781

/-- The parabola y = x^2 - 4x + 4 -/
noncomputable def parabola (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- The line y = 2x - 3 -/
noncomputable def line (x : ℝ) : ℝ := 2*x - 3

/-- The distance between a point (x, parabola x) on the parabola and the line -/
noncomputable def distance_to_line (x : ℝ) : ℝ :=
  abs (2*x - parabola x - 3) / Real.sqrt 5

theorem shortest_distance :
  ∃ (d : ℝ), d = 4 / Real.sqrt 5 ∧
  ∀ (x : ℝ), distance_to_line x ≥ d := by
  sorry

#check shortest_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l607_60781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_geometric_mean_l607_60739

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * a 1 + (n : ℝ) * (n - 1 : ℝ) / 2 * (a n - a 1) / (n - 1 : ℝ)

noncomputable def geometric_mean (x y : ℝ) : ℝ :=
  Real.sqrt (x * y)

theorem arithmetic_sequence_geometric_mean
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum9 : sum_arithmetic_sequence a 9 = -36)
  (h_sum13 : sum_arithmetic_sequence a 13 = -104) :
  geometric_mean (a 5) (a 7) = 4 * Real.sqrt 2 ∨
  geometric_mean (a 5) (a 7) = -4 * Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_geometric_mean_l607_60739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_equality_l607_60723

noncomputable def a (n : ℕ) : ℝ := (Real.sin ((2 ^ n * Real.pi) / 18)) ^ 2

theorem smallest_n_for_equality : 
  (∀ n : ℕ, a n = (Real.sin (Real.pi / 18)) ^ 2 → n ≥ 12) ∧ 
  a 12 = (Real.sin (Real.pi / 18)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_equality_l607_60723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_min_angle_le_60_sqrt_inequality_l607_60771

-- Problem 1
theorem triangle_min_angle_le_60 : 
  ∀ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c → a + b + c = 180 → 
  min a (min b c) ≤ 60 := by sorry

-- Problem 2
theorem sqrt_inequality (n : ℝ) (hn : n ≥ 0) : 
  Real.sqrt (n + 2) - Real.sqrt (n + 1) ≤ Real.sqrt (n + 1) - Real.sqrt n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_min_angle_le_60_sqrt_inequality_l607_60771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l607_60720

/-- Calculates the annual interest rate given the principal, simple interest, and time period. -/
noncomputable def calculate_interest_rate (principal : ℝ) (simple_interest : ℝ) (time : ℝ) : ℝ :=
  (simple_interest / (principal * time)) * 100

/-- Theorem stating that for the given conditions, the interest rate is approximately 3.5% -/
theorem interest_rate_calculation (principal : ℝ) (simple_interest : ℝ) (time : ℝ)
  (h1 : principal = 666.67)
  (h2 : simple_interest = 70)
  (h3 : time = 3) :
  abs (calculate_interest_rate principal simple_interest time - 3.5) < 0.01 := by
  sorry

/-- Compute the result using rational approximations -/
def approximate_interest_rate : ℚ :=
  (70 / (666.67 * 3)) * 100

#eval approximate_interest_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l607_60720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_l607_60712

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: Given a trapezium with parallel sides of 10 cm and 18 cm, and an area of 140.00014 cm²,
    the distance between the parallel sides is 10.00001 cm. -/
theorem trapezium_height (a b area h : ℝ) 
  (ha : a = 10) 
  (hb : b = 18) 
  (harea : area = 140.00014) 
  (h_def : trapeziumArea a b h = area) : 
  h = 10.00001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_height_l607_60712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_faster_longer_l607_60777

noncomputable section

-- Define the lengths of the segments
variable (a b c d : ℝ)
-- Define the speeds of the airplanes
variable (v₁ v₂ v₃ v₄ : ℝ)

-- Assumptions
axiom lengths_different : a > b ∧ b > c ∧ c > d
axiom speeds_positive : v₁ > 0 ∧ v₂ > 0 ∧ v₃ > 0 ∧ v₄ > 0

-- Define the total travel time function
def total_time (s₁ s₂ s₃ s₄ : ℝ) : ℝ :=
  a / s₁ + b / s₂ + c / s₃ + d / s₄

-- Theorem: The total travel time is minimized when faster planes are used for longer segments
theorem min_time_faster_longer (s₁ s₂ s₃ s₄ : ℝ) 
  (h₁ : s₁ > 0) (h₂ : s₂ > 0) (h₃ : s₃ > 0) (h₄ : s₄ > 0) :
  v₁ ≥ v₂ ∧ v₂ ≥ v₃ ∧ v₃ ≥ v₄ →
  total_time v₁ v₂ v₃ v₄ ≤ total_time s₁ s₂ s₃ s₄ :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_faster_longer_l607_60777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_plus_pi_4_l607_60773

theorem tan_2alpha_plus_pi_4 (α : Real) (h1 : α ∈ Set.Ioo (π/2) π) (h2 : Real.sin α = Real.sqrt 5 / 5) :
  Real.tan (2*α + π/4) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2alpha_plus_pi_4_l607_60773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_inscribed_cube_formula_l607_60755

/-- A cube inscribed in a hemisphere with radius R -/
structure InscribedCube (R : ℝ) where
  /-- Four vertices of the cube lie on the base of the hemisphere -/
  vertices_on_base : Prop
  /-- Four vertices of the cube lie on the spherical surface of the hemisphere -/
  vertices_on_surface : Prop

/-- The volume of a cube inscribed in a hemisphere -/
noncomputable def volume_inscribed_cube (R : ℝ) (cube : InscribedCube R) : ℝ :=
  (2 * R^3 * Real.sqrt 6) / 9

/-- Theorem: The volume of a cube inscribed in a hemisphere of radius R is (2 * R^3 * √6) / 9 -/
theorem volume_inscribed_cube_formula (R : ℝ) (cube : InscribedCube R) :
  volume_inscribed_cube R cube = (2 * R^3 * Real.sqrt 6) / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_inscribed_cube_formula_l607_60755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_inequality_implies_a_bound_l607_60787

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + (1 + a) * Real.exp (-x)

-- Part 1: Prove that if f is even, then a = 0
theorem even_function_implies_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 :=
by sorry

-- Part 2: Prove that if f(x) ≥ a+1 for all x > 0, then a ≤ 3
theorem inequality_implies_a_bound (a : ℝ) :
  (∀ x > 0, f a x ≥ a + 1) → a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_inequality_implies_a_bound_l607_60787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l607_60790

-- Define the complex-valued function representing the left side of the equation
noncomputable def f (x : ℂ) : ℂ := 
  (2 * x^3 + 6 * x^2 * Complex.I * Real.sqrt 3 + 12 * x + 4 * Complex.I * Real.sqrt 3) + 
  (2 * x + 2 * Complex.I * Real.sqrt 3)

-- Define the set of solutions
noncomputable def solutions : Set ℂ := 
  {-Complex.I * Real.sqrt 3, -Complex.I * Real.sqrt 3 + 1, -Complex.I * Real.sqrt 3 - 1}

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℂ, f x = 0 ↔ x ∈ solutions := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l607_60790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l607_60701

/-- Calculates the value of an investment after a given time period with simple interest -/
def simple_interest_value (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Calculates the value of an investment after a given time period with compound interest -/
noncomputable def compound_interest_value (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The difference in value between two investments after two years -/
theorem investment_difference : 
  let fund_a_principal : ℝ := 2000
  let fund_a_rate : ℝ := 0.12
  let fund_b_principal : ℝ := 1000
  let fund_b_rate : ℝ := 0.30
  let time : ℝ := 2
  ∃ ε > 0, |simple_interest_value fund_a_principal fund_a_rate time - 
  compound_interest_value fund_b_principal fund_b_rate time - 550| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_difference_l607_60701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l607_60775

/-- An inverse proportion function with parameter k -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := (k - 2) / x

/-- Predicate to check if a point (x, y) is in the second or fourth quadrant -/
def in_second_or_fourth_quadrant (x y : ℝ) : Prop :=
  (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)

/-- Theorem stating that if the graph of y = (k-2)/x lies in the second and fourth quadrants, then k < 2 -/
theorem inverse_proportion_quadrants (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → in_second_or_fourth_quadrant x (inverse_proportion k x)) →
  k < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l607_60775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_empty_l607_60716

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {3, 4, 5, 6, 7, 8}

theorem complement_intersection_empty : (U \ A) ∩ (U \ B) = ∅ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_intersection_empty_l607_60716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_same_score_l607_60765

def test_score (correct incorrect : ℕ) : ℤ :=
  6 + 4 * correct - incorrect

def possible_scores : Finset ℤ :=
  Finset.image (λ p : ℕ × ℕ => test_score p.1 p.2)
    (Finset.filter (λ p : ℕ × ℕ => p.1 + p.2 ≤ 6) (Finset.product (Finset.range 7) (Finset.range 7)))

theorem at_least_three_same_score (num_students : ℕ) (h : num_students = 51) :
  ∃ (score : ℤ), (Finset.filter (λ s => s = score) possible_scores).card ≥ 3 := by
  sorry

#eval possible_scores.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_same_score_l607_60765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l607_60760

/-- The radius of spheres placed in the cone -/
noncomputable def sphere_radius : ℝ := Real.sqrt 24

/-- The arrangement of spheres in the cone -/
structure SphereArrangement where
  /-- The three spheres touch each other externally -/
  spheres_touch : Bool
  /-- Two spheres touch the lateral surface and base of the cone -/
  two_spheres_touch_cone : Bool
  /-- The third sphere touches the lateral surface at a point in the same plane as the centers -/
  third_sphere_touch : Bool

/-- The cone containing the spheres -/
structure Cone where
  /-- The radius of the base -/
  base_radius : ℝ
  /-- The height of the cone -/
  height : ℝ
  /-- The radius of the base is equal to the height -/
  radius_equals_height : base_radius = height
  /-- The arrangement of spheres in the cone -/
  sphere_arrangement : SphereArrangement

/-- The theorem stating the radius of the base of the cone -/
theorem cone_base_radius (c : Cone) 
  (h1 : c.sphere_arrangement.spheres_touch)
  (h2 : c.sphere_arrangement.two_spheres_touch_cone)
  (h3 : c.sphere_arrangement.third_sphere_touch) : 
  c.base_radius = sphere_radius * (7 + 4 * Real.sqrt 3 + 2 * Real.sqrt 6) / (2 * Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_radius_l607_60760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l607_60713

theorem cubic_equation_roots (m n : ℝ) : 
  (∃ α β : ℝ, 
    (∀ x : ℝ, x^3 + m*x^2 - 3*x + n = 0 ↔ x = α ∨ x = β) ∧ 
    α^2 + 2*β^2 = 6 ∧ 
    β > 0 ∧ 
    (∃ k : ℕ, k ≥ 2 ∧ (Multiset.count β {α, β, β} = k)))
  → m = 0 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_equation_roots_l607_60713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_minimum_value_one_l607_60737

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x + 1/2) + 2 / (2 * x + 1)

-- Theorem for part (I)
theorem f_monotone_increasing (a : ℝ) :
  (∀ x > 0, Monotone (fun x => f a x)) ↔ a ≥ 2 := by sorry

-- Theorem for part (II)
theorem f_minimum_value_one :
  ∃! a : ℝ, ∀ x > 0, f a x ≥ 1 ∧ ∃ y > 0, f a y = 1 ∧ a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_f_minimum_value_one_l607_60737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_m_with_property_P_l607_60751

def fibonacci (a b : ℕ) : ℕ → ℕ
  | 0 => a
  | 1 => b
  | (n + 2) => fibonacci a b (n + 1) + fibonacci a b n

def has_property_P (m : ℕ) (a b : ℕ) : Prop :=
  ∃ N : ℕ, ∀ k > N, ¬∃ x : ℕ, x^2 = 1 + m * (fibonacci a b k) * (fibonacci a b (k + 2))

theorem infinitely_many_m_with_property_P (a b : ℕ) (h : a < b) :
  ∀ n : ℕ, ∃ m > n, has_property_P m a b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_m_with_property_P_l607_60751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_nine_is_zero_l607_60719

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- The first, third, and fourth terms form a geometric sequence -/
def geometric_subsequence (a : ℕ → ℝ) : Prop :=
  a 3 ^ 2 = a 1 * a 4

/-- Sum of the first n terms of the sequence -/
noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a 1 + (n - 1 : ℝ) * 2)

/-- The main theorem: S₉ = 0 for the given sequence -/
theorem sum_nine_is_zero (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_geom : geometric_subsequence a) : 
  S a 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_nine_is_zero_l607_60719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_proof_l607_60795

noncomputable def basic_salary : ℝ := 1250
noncomputable def commission_rate : ℝ := 0.10
noncomputable def total_sales : ℝ := 23600
noncomputable def monthly_expenses : ℝ := 2888

noncomputable def commission : ℝ := commission_rate * total_sales
noncomputable def total_earnings : ℝ := basic_salary + commission
noncomputable def savings : ℝ := total_earnings - monthly_expenses
noncomputable def savings_percentage : ℝ := (savings / total_earnings) * 100

theorem savings_percentage_proof : 
  ∃ ε > 0, |savings_percentage - 22.16| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_percentage_proof_l607_60795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_free_throw_success_rate_increase_l607_60744

/-- Calculates the increase in free throw success rate given initial and additional attempts -/
theorem free_throw_success_rate_increase 
  (initial_success : ℕ) 
  (initial_attempts : ℕ) 
  (additional_attempts : ℕ) 
  (additional_success_rate : ℚ) : 
  let new_success := initial_success + (additional_success_rate * additional_attempts).floor
  let total_attempts := initial_attempts + additional_attempts
  let initial_rate := (initial_success : ℚ) / initial_attempts
  let new_rate := (new_success : ℚ) / total_attempts
  let increase := new_rate - initial_rate
  ⌊increase * 100⌋₊ = 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_free_throw_success_rate_increase_l607_60744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_external_correct_max_area_internal_correct_l607_60721

/-- Two circles touching at point A with diameters d₁ and d₂ --/
structure TouchingCircles where
  d₁ : ℝ
  d₂ : ℝ
  h₁ : d₁ > 0
  h₂ : d₂ > 0

/-- Quadrilateral BDCE formed by points on the circles --/
def Quadrilateral (c : TouchingCircles) :=
  {BDCE : Set (ℝ × ℝ) // BDCE.ncard = 4}

/-- The area of the quadrilateral BDCE --/
noncomputable def area (c : TouchingCircles) (q : Quadrilateral c) : ℝ := sorry

/-- The maximum area of the quadrilateral for external tangency --/
noncomputable def max_area_external (c : TouchingCircles) : ℝ :=
  (c.d₁ + c.d₂)^2 / 4

/-- The maximum area of the quadrilateral for internal tangency --/
noncomputable def max_area_internal (c : TouchingCircles) : ℝ :=
  (c.d₂^2 - c.d₁^2) / 4

/-- Theorem: The maximum area of the quadrilateral for external tangency --/
theorem max_area_external_correct (c : TouchingCircles) :
  ∃ q : Quadrilateral c, area c q ≤ max_area_external c ∧
  ∃ q' : Quadrilateral c, area c q' = max_area_external c := by
  sorry

/-- Theorem: The maximum area of the quadrilateral for internal tangency --/
theorem max_area_internal_correct (c : TouchingCircles) :
  ∃ q : Quadrilateral c, area c q ≤ max_area_internal c ∧
  ∃ q' : Quadrilateral c, area c q' = max_area_internal c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_external_correct_max_area_internal_correct_l607_60721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_10_range_of_t_for_f_geq_4_over_t_plus_2_l607_60730

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

-- Theorem for part I
theorem solution_set_f_geq_10 :
  {x : ℝ | f x ≥ 10} = Set.Iic (-3) ∪ Set.Ici 7 :=
sorry

-- Theorem for part II
theorem range_of_t_for_f_geq_4_over_t_plus_2 :
  {t : ℝ | ∀ x : ℝ, f x ≥ 4/t + 2} = Set.Iio 0 ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_geq_10_range_of_t_for_f_geq_4_over_t_plus_2_l607_60730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_lambda_l607_60736

theorem sine_equation_lambda (lambda : ℝ) : 
  2 * Real.sin (77 * π / 180) - Real.sin (17 * π / 180) = lambda * Real.sin (73 * π / 180) → 
  lambda = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_lambda_l607_60736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_supremum_l607_60759

theorem inequality_supremum : 
  (∀ x : ℝ, |x + 2| + |x - 1| ≥ 3) ∧ 
  (∀ ε > 0, ∃ x : ℝ, |x + 2| + |x - 1| < 3 + ε) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_supremum_l607_60759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l607_60738

noncomputable def sequence_formula (n : ℕ+) : ℝ := Real.sqrt (6 * n - 3)

theorem sequence_proof (n : ℕ+) : 
  (n = 1 → sequence_formula n = Real.sqrt 3) ∧
  (n = 2 → sequence_formula n = 3) ∧
  (n = 3 → sequence_formula n = Real.sqrt 15) ∧
  (n = 4 → sequence_formula n = Real.sqrt 21) ∧
  (n = 5 → sequence_formula n = 3 * Real.sqrt 3) ∧
  (∀ (m : ℕ+), sequence_formula m = Real.sqrt (6 * m - 3)) :=
by
  sorry

#check sequence_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_proof_l607_60738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_integer_and_divisible_by_m_l607_60764

-- Define the function f
def f (c : ℤ) (q : ℕ) (a : ℕ → ℤ) (n : ℕ) (x : ℕ) : ℤ :=
  c * q^x + (Finset.range (n+1)).sum (λ i ↦ a i * x^i)

-- State the theorem
theorem f_integer_and_divisible_by_m 
  (q : ℕ) (c : ℤ) (a : ℕ → ℤ) (n : ℕ) (m : ℕ) (hm : m ≥ 1) :
  (∀ k : ℕ, k ≤ n + 1 → ∃ z : ℤ, f c q a n k = z) →
  (∀ k : ℕ, k ≤ n + 1 → ∃ z : ℤ, f c q a n k = m * z) →
  ∀ x : ℕ, ∃ z : ℤ, f c q a n x = z ∧ ∃ w : ℤ, f c q a n x = m * w :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_integer_and_divisible_by_m_l607_60764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_proof_l607_60782

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define the left vertex
def A : ℝ × ℝ := (-2, 0)

-- Define the point P
noncomputable def P : ℝ × ℝ := (4, Real.sqrt 15)

-- Define the centroid G
noncomputable def G (P : ℝ × ℝ) : ℝ × ℝ := ((P.1 + F₁.1 + F₂.1) / 3, (P.2 + F₁.2 + F₂.2) / 3)

-- Define the incenter I (we don't need to define its exact position for this problem)
noncomputable def I (P : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the line l
def line_l (x : ℝ) : ℝ := -2 * x + 6

-- Define the theorem
theorem hyperbola_proof :
  -- Part 1: P exists on the hyperbola in the first quadrant and IG ∥ F₁F₂
  hyperbola P.1 P.2 ∧ P.1 > 0 ∧ P.2 > 0 ∧
  ∃ (t : ℝ), G P = I P + t • (F₂.1 - F₁.1, F₂.2 - F₁.2) ∧
  -- Part 2: The equation of line l is y = -2x + 6
  ∀ (x : ℝ), line_l x = -2 * x + 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_proof_l607_60782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_pi_minus_theta_l607_60772

theorem sin_two_pi_minus_theta (θ : ℝ) 
  (h1 : 3 * (Real.cos θ)^2 = Real.tan θ + 3) 
  (h2 : ∀ k : ℤ, θ ≠ k * π) : 
  Real.sin (2 * (π - θ)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_pi_minus_theta_l607_60772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l607_60724

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi/3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l607_60724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_determinable_cards_l607_60709

/-- Represents a card with a unique number -/
structure Card where
  number : ℕ
  unique : True

/-- Represents the game state -/
structure GameState where
  cards : Finset Card
  cardCount : Nat
  h_card_count : cardCount = 2013

/-- Represents a move in the game -/
def Move (gs : GameState) := 
  { selectedCards : Finset Card // selectedCards.card = 10 ∧ selectedCards ⊆ gs.cards }

/-- Represents the response to a move -/
def Response (gs : GameState) (m : Move gs) := 
  { c : Card // c ∈ m.val }

/-- Represents a strategy for determining card numbers -/
def Strategy (gs : GameState) := 
  Π (m : Move gs), Response gs m → Option (Finset Card)

/-- Predicate to check if a card's number is known -/
def isKnown (c : Card) : Prop := True  -- Placeholder, replace with actual logic

/-- The main theorem: The maximum number of determinable cards is 1986 -/
theorem max_determinable_cards (gs : GameState) :
  ∃ (s : Strategy gs), ∀ (determined : Finset Card), 
    (∀ c ∈ determined, isKnown c) → determined.card ≤ 1986 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_determinable_cards_l607_60709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rawhide_bone_cost_rawhide_bone_cost_is_one_l607_60707

/-- The cost of each rawhide bone, given Belle's treat consumption and costs -/
theorem rawhide_bone_cost : ℝ := 1

/-- Proof that the rawhide bone cost is $1, given the conditions -/
theorem rawhide_bone_cost_is_one : 
  ∃ (bone_cost : ℝ), bone_cost = 1 ∧ 
  (7 : ℝ) * (4 * 0.25 + 2 * bone_cost) = 21 :=
by
  use 1
  apply And.intro
  · rfl
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rawhide_bone_cost_rawhide_bone_cost_is_one_l607_60707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_sixteen_is_min_k_l607_60705

-- Define the function f
noncomputable def f (k : ℕ) (x : ℝ) : ℝ :=
  (Real.sin (k * x / 10))^4 + (Real.cos (k * x / 10))^4

-- Define the property that the function's range over any unit interval is equal to its entire range
def range_property (k : ℕ) : Prop :=
  ∀ a : ℝ, Set.range (fun x => f k x) = Set.range (fun x => f k (x + a))

-- State the theorem
theorem min_k_value :
  ∃ k₀ : ℕ, k₀ > 0 ∧ range_property k₀ ∧ ∀ k : ℕ, k > 0 → range_property k → k₀ ≤ k :=
sorry

-- Proof that 16 is the minimum value of k
theorem sixteen_is_min_k : min_k_value.choose = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_value_sixteen_is_min_k_l607_60705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_projections_l607_60741

-- Define vectors a and b as elements of ℝ²
def a : ℝ × ℝ := sorry
def b : ℝ × ℝ := sorry

-- Define the projection function
def proj (v : ℝ × ℝ) (w : ℝ × ℝ) : ℝ × ℝ := sorry

theorem orthogonal_projections 
  (h1 : a.1 * b.1 + a.2 * b.2 = 0) -- a and b are orthogonal
  (h2 : proj a (4, -2) = (-2/5, -4/5)) : -- Given projection of (4, -2) onto a
  proj b (4, -2) = (22/5, -6/5) := -- Desired projection of (4, -2) onto b
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_projections_l607_60741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_constant_l607_60750

noncomputable def line (x : ℝ) : ℝ := (3/2) * x + 3

noncomputable def vector_on_line (a : ℝ) : ℝ × ℝ := (a, line a)

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * w.1 + v.2 * w.2
  let magnitude_squared := w.1 * w.1 + w.2 * w.2
  let scalar := dot_product / magnitude_squared
  (scalar * w.1, scalar * w.2)

theorem projection_constant (w : ℝ × ℝ) :
  ∀ a : ℝ, projection (vector_on_line a) w = (-18/13, 12/13) := by
  sorry

#check projection_constant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_constant_l607_60750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_intersect_at_most_twice_l607_60703

-- Define the ellipse structure
structure Ellipse where
  foci : Fin 2 → ℝ × ℝ
  majorAxis : ℝ

-- Define a function to check if a point is on an ellipse
def isOnEllipse (e : Ellipse) (p : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p.1 - (e.foci 0).1)^2 + (p.2 - (e.foci 0).2)^2)
  let d2 := Real.sqrt ((p.1 - (e.foci 1).1)^2 + (p.2 - (e.foci 1).2)^2)
  d1 + d2 = e.majorAxis

-- Define the theorem
theorem ellipses_intersect_at_most_twice
  (e1 e2 : Ellipse)
  (h_diff : e1 ≠ e2)
  (h_focus : e1.foci 0 = e2.foci 0) :
  ∃ (n : Nat), n ≤ 2 ∧
  (∀ (p : ℝ × ℝ), isOnEllipse e1 p ∧ isOnEllipse e2 p → 
   ∃ (points : Fin n → ℝ × ℝ), ∀ (i : Fin n), isOnEllipse e1 (points i) ∧ isOnEllipse e2 (points i)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipses_intersect_at_most_twice_l607_60703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_bounds_l607_60794

-- Define the recursive function g
noncomputable def g : ℕ → ℝ
| 0 => 0
| 1 => 0
| 2 => 2 * Real.log 10
| (n+3) => Real.log (n + 3 + g (n+2))

-- Define B as g(2050)
noncomputable def B : ℝ := g 2050

-- Theorem statement
theorem B_bounds : Real.log 2053 < B ∧ B < Real.log 2054 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_bounds_l607_60794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l607_60753

-- Define the quadratic function
noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

-- Define the x-coordinate of the vertex
noncomputable def vertex_x : ℝ := 9 / (2 * 3)

-- Theorem statement
theorem quadratic_properties :
  vertex_x > 0 ∧ vertex_x = (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l607_60753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l607_60722

-- Define the constants
noncomputable def a : ℝ := (2 : ℝ) ^ (1/10 : ℝ)
noncomputable def b : ℝ := Real.log (5/2)
noncomputable def c : ℝ := Real.log (9/10) / Real.log 3

-- State the theorem
theorem size_relationship : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_relationship_l607_60722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l607_60702

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := (Real.sqrt 3 / 4) * (t.a^2 + t.b^2 - t.c^2)

theorem triangle_properties (t : Triangle) 
  (h1 : 0 < t.a ∧ 0 < t.b ∧ 0 < t.c)
  (h2 : t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b)
  (h3 : t.A > 0 ∧ t.B > 0 ∧ t.C > 0)
  (h4 : t.A + t.B + t.C = Real.pi)
  (h5 : area t = (Real.sqrt 3 / 4) * (t.a^2 + t.b^2 - t.c^2)) :
  (t.C = Real.pi / 3) ∧ 
  (t.a + t.b = 4 → 
    (6 ≤ t.a + t.b + t.c ∧ t.a + t.b + t.c < 8) ∧
    (area t ≤ Real.sqrt 3 ∧ 
     ∃ (t' : Triangle), area t' = Real.sqrt 3 ∧ t'.a + t'.b = 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l607_60702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_two_girls_mutually_exclusive_not_complementary_l607_60756

/-- Represents the number of girls selected -/
inductive GirlsSelected
  | Zero
  | One
  | Two

/-- The sample space of all possible outcomes when selecting 2 students from 2 boys and 2 girls -/
def SampleSpace := Fin 6

/-- Event representing exactly one girl being selected -/
def ExactlyOneGirl : Set SampleSpace := sorry

/-- Event representing exactly two girls being selected -/
def ExactlyTwoGirls : Set SampleSpace := sorry

/-- The probability measure on the sample space -/
noncomputable def P : Set SampleSpace → ℝ := sorry

theorem exactly_one_two_girls_mutually_exclusive_not_complementary :
  (ExactlyOneGirl ∩ ExactlyTwoGirls = ∅) ∧
  (ExactlyOneGirl ∪ ExactlyTwoGirls ≠ Set.univ) := by
  sorry

#check exactly_one_two_girls_mutually_exclusive_not_complementary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_two_girls_mutually_exclusive_not_complementary_l607_60756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_satisfying_lcm_conditions_l607_60774

theorem unique_m_satisfying_lcm_conditions :
  ∀ m : ℕ, 
    m > 0 → 
    Nat.lcm 40 m = 120 → 
    Nat.lcm m 45 = 180 → 
    m = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_m_satisfying_lcm_conditions_l607_60774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_even_g_l607_60784

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos (x / 3), Real.sqrt 3 * Real.cos (x / 3))
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sin (x / 3), Real.cos (x / 3))

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (3 * x + φ)

theorem min_phi_for_even_g :
  ∀ φ > 0, (∀ x, g φ x = g φ (-x)) → φ ≥ π / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_phi_for_even_g_l607_60784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_womans_speed_is_25_l607_60742

/-- The woman's traveling speed in miles per hour -/
noncomputable def womans_speed : ℝ := 25

/-- The man's constant walking speed in miles per hour -/
noncomputable def mans_speed : ℝ := 5

/-- The time in hours the woman waits after stopping -/
noncomputable def waiting_time : ℝ := 20 / 60

/-- The time in hours the woman travels before stopping -/
noncomputable def travel_time : ℝ := 5 / 60

/-- The total time in hours from when the woman passes the man until he catches up -/
noncomputable def total_time : ℝ := travel_time + waiting_time

theorem womans_speed_is_25 : 
  womans_speed * travel_time = mans_speed * total_time := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_womans_speed_is_25_l607_60742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l607_60799

noncomputable def f (x : Real) : Real := -2 * (Real.cos x)^2 - 2 * Real.sin x + 9/2

theorem min_value_of_f :
  ∃ (m : Real), m = 2 ∧ ∀ x ∈ Set.Icc (π/6) (2*π/3), f x ≥ m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l607_60799
