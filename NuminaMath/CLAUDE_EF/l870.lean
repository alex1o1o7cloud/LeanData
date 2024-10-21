import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_13pi_l870_87005

noncomputable section

def point := ℝ × ℝ

def C : point := (-2, 3)
def D : point := (4, -1)

noncomputable def distance (p q : point) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

noncomputable def diameter := distance C D

noncomputable def radius := diameter / 2

def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circle_area_is_13pi :
  circle_area radius = 13 * Real.pi := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_13pi_l870_87005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l870_87009

-- Define the circle A
def circle_A : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 20}

-- Define line l₁
def line_l₁ (x y : ℝ) : Prop :=
  x + 2*y + 7 = 0

-- Define point B
def point_B : ℝ × ℝ := (-4, 0)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + 4)

theorem circle_and_line_problem :
  ∃ (M N : ℝ × ℝ) (k : ℝ),
    (∃ (p : ℝ × ℝ), p ∈ circle_A ∧ line_l₁ p.1 p.2) ∧
    (M ∈ circle_A ∧ N ∈ circle_A) ∧
    (line_l k M.1 M.2 ∧ line_l k N.1 N.2) ∧
    line_l k point_B.1 point_B.2 ∧
    distance M N = 2 * Real.sqrt 11 →
    (∀ (x y : ℝ), (x + 1)^2 + (y - 2)^2 = 20 ↔ (x, y) ∈ circle_A) ∧
    ((∃ (x y : ℝ), x = -4 ∧ line_l k x y) ∨
     (∃ (x y : ℝ), 5*x + 12*y + 20 = 0 ∧ line_l k x y)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l870_87009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_supply_and_max_revenue_l870_87056

/-- Market supply and demand model with taxation -/
structure MarketModel where
  demandSlope : ℝ
  demandIntercept : ℝ
  supplySlope : ℝ
  supplyIntercept : ℝ
  taxRate : ℝ

/-- Calculate the equilibrium quantity given a market model -/
noncomputable def equilibriumQuantity (model : MarketModel) : ℝ :=
  (model.demandIntercept - model.supplyIntercept + model.taxRate * model.demandSlope) /
    (model.supplySlope - model.demandSlope)

/-- Calculate the tax revenue given a market model -/
noncomputable def taxRevenue (model : MarketModel) : ℝ :=
  model.taxRate * equilibriumQuantity model

/-- Theorem stating the market supply function and maximum tax revenue -/
theorem market_supply_and_max_revenue :
  ∃ (model : MarketModel),
    model.demandSlope = -4 ∧
    model.demandIntercept = 688 ∧
    model.supplySlope = 6 ∧
    model.supplyIntercept = -312 ∧
    (∀ t, taxRevenue { demandSlope := -4, demandIntercept := 688, 
                       supplySlope := 6, supplyIntercept := -312, taxRate := t } ≤ 8640) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_supply_and_max_revenue_l870_87056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_upper_bound_l870_87045

open Real

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := (log x + (x - b)^2) / x

-- State the theorem
theorem b_upper_bound (b : ℝ) :
  (∃ x ∈ Set.Icc (1/2 : ℝ) 2, f b x + x * (deriv (f b) x) > 0) →
  b < 9/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_upper_bound_l870_87045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_42_63518_to_nearest_tenth_l870_87002

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The statement that rounding 42.63518 to the nearest tenth equals 42.6 -/
theorem round_42_63518_to_nearest_tenth :
  roundToNearestTenth 42.63518 = 42.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_42_63518_to_nearest_tenth_l870_87002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_cost_l870_87093

theorem car_rental_cost 
  (daily_rate : ℝ) 
  (mile_rate : ℝ) 
  (days : ℕ) 
  (miles : ℕ) 
  (h1 : daily_rate = 25) 
  (h2 : mile_rate = 0.2) 
  (h3 : days = 4) 
  (h4 : miles = 400) : 
  daily_rate * (days : ℝ) + mile_rate * (miles : ℝ) = 180 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_rental_cost_l870_87093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l870_87079

noncomputable def f (x : ℝ) := x - 3 + Real.log x / Real.log 3

theorem root_exists_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 3 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l870_87079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_church_tower_height_calculation_l870_87077

/-- The height of the church tower in feet -/
noncomputable def church_tower_height : ℝ := Real.sqrt 57500

/-- The height of the catholic tower in feet -/
def catholic_tower_height : ℝ := 200

/-- The distance between the two towers in feet -/
def distance_between_towers : ℝ := 350

/-- The distance from the church tower to the grain in feet -/
def distance_to_grain : ℝ := 150

theorem church_tower_height_calculation :
  church_tower_height^2 = 
    catholic_tower_height^2 + 
    (distance_between_towers - distance_to_grain)^2 - 
    distance_to_grain^2 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval church_tower_height

end NUMINAMATH_CALUDE_ERRORFEEDBACK_church_tower_height_calculation_l870_87077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_orthogonality_and_angle_l870_87015

/-- Given vectors m and n, prove cos(2α) = -3/5 and β = π/4 -/
theorem vector_orthogonality_and_angle (α β : ℝ) 
  (h_alpha : α ∈ Set.Ioo 0 (π/2))
  (h_beta : β ∈ Set.Ioo 0 (π/2))
  (h_perp : (Real.cos α) * 2 + (-1) * (Real.sin α) = 0)  -- m ⊥ n condition
  (h_sin : Real.sin (α - β) = Real.sqrt 10 / 10) : 
  Real.cos (2 * α) = -3/5 ∧ β = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_orthogonality_and_angle_l870_87015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_when_f_is_zero_range_of_f_in_interval_l870_87053

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin x * Real.cos (x + Real.pi / 4)

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  BC : ℝ
  AB : ℝ

-- Theorem for part I
theorem angle_B_when_f_is_zero (t : Triangle) 
  (h1 : t.BC = 2) 
  (h2 : t.AB = Real.sqrt 2) 
  (h3 : f (t.A - Real.pi / 4) = 0) : 
  t.B = Real.pi / 4 ∨ t.B = 7 * Real.pi / 12 := by
  sorry

-- Theorem for part II
theorem range_of_f_in_interval : 
  Set.Icc (f (Real.pi / 2)) (f (17 * Real.pi / 24)) = 
  Set.Icc (-Real.sqrt 2 - 1) (-2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_when_f_is_zero_range_of_f_in_interval_l870_87053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_volume_l870_87066

/-- A regular tetrahedron with edge length 1 -/
structure RegularTetrahedron where
  edge_length : ℝ
  is_regular : edge_length = 1

/-- The circumscribed sphere of a regular tetrahedron -/
def circumscribed_sphere (t : RegularTetrahedron) : ℝ → Prop := sorry

/-- The volume of a sphere given its radius -/
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4/3) * Real.pi * radius^3

/-- Theorem: The volume of the circumscribed sphere of a regular tetrahedron with edge length 1 is (√6/8)π -/
theorem circumscribed_sphere_volume (t : RegularTetrahedron) :
  ∃ (r : ℝ), circumscribed_sphere t r ∧ sphere_volume r = (Real.sqrt 6 / 8) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_volume_l870_87066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l870_87060

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- Line 1: x - y - 1 = 0 -/
def line1 (x y : ℝ) : Prop := x - y - 1 = 0

/-- Line 2: x - y + 1 = 0 -/
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

theorem distance_between_lines :
  distance_between_parallel_lines 1 (-1) (-1) 1 = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l870_87060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_values_l870_87059

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relationships between lines and planes
axiom parallel_line_plane : Line → Plane → Prop
axiom parallel_plane_plane : Plane → Plane → Prop
axiom parallel_line_line : Line → Line → Prop
axiom perpendicular_line_plane : Line → Plane → Prop
axiom perpendicular_plane_plane : Plane → Plane → Prop
axiom perpendicular_line_line : Line → Line → Prop

-- Define the propositions
def proposition1 (l : Line) (α β : Plane) : Prop :=
  parallel_line_plane l β → parallel_plane_plane α β → parallel_line_plane l α

def proposition2 (l m n : Line) : Prop :=
  parallel_line_line l n → parallel_line_line m n → parallel_line_line l m

def proposition3 (l : Line) (α β : Plane) : Prop :=
  perpendicular_plane_plane α β → parallel_line_plane l α → perpendicular_line_plane l β

def proposition4 (l m : Line) (α β : Plane) : Prop :=
  perpendicular_line_plane l α → perpendicular_line_plane m β → perpendicular_plane_plane α β → perpendicular_line_line l m

-- Theorem statement
theorem propositions_truth_values (l m n : Line) (α β γ : Plane) :
  (¬ ∀ l α β, proposition1 l α β) ∧
  (∀ l m n, proposition2 l m n) ∧
  (¬ ∀ l α β, proposition3 l α β) ∧
  (∀ l m α β, proposition4 l m α β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_values_l870_87059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_removal_l870_87030

theorem average_after_removal (numbers : Finset ℕ) (sum : ℕ) : 
  numbers.card = 12 →
  sum = numbers.sum id →
  sum / 12 = 90 →
  80 ∈ numbers →
  90 ∈ numbers →
  let remaining := numbers.erase 80 |>.erase 90
  (remaining.sum id) / (remaining.card) = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_after_removal_l870_87030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_equals_target_l870_87037

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ -3 then (3/2) * x + 4.5
  else if x ≤ -1 then (1/2) * x + 1/2
  else if x ≤ 1 then 2 * x + 1
  else if x ≤ 3 then (-1/2) * x + 3.5
  else 2 * x - 4

noncomputable def x₁ : ℝ := -4/3
noncomputable def x₂ : ℝ := 3/4
noncomputable def x₃ : ℝ := 13/4

theorem sum_of_x_coordinates_equals_target : 
  x₁ + x₂ + x₃ = -4/3 + 3/4 + 13/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_coordinates_equals_target_l870_87037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_valued_implies_rational_coeffs_l870_87057

/-- A polynomial with complex coefficients -/
noncomputable def ComplexPolynomial : Type := ℕ → ℂ

/-- Evaluation of a polynomial at a point -/
noncomputable def eval (P : ComplexPolynomial) (x : ℂ) : ℂ :=
  ∑' n, (P n) * x^n

/-- A polynomial is rational-valued on rationals if its evaluation at any rational number is rational -/
def rational_valued_on_rationals (P : ComplexPolynomial) : Prop :=
  ∀ q : ℚ, ∃ r : ℚ, eval P (↑q) = r

/-- The coefficients of a polynomial -/
def coefficients (P : ComplexPolynomial) : Set ℂ :=
  {c : ℂ | ∃ n : ℕ, P n = c}

theorem rational_valued_implies_rational_coeffs (P : ComplexPolynomial) 
  (h : rational_valued_on_rationals P) : 
  ∀ c ∈ coefficients P, ∃ r : ℚ, c = r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_valued_implies_rational_coeffs_l870_87057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_S_l870_87092

-- Define the set S
def S : Set ℂ := {z | -2 ≤ z.re ∧ z.re ≤ 2 ∧ -2 ≤ z.im ∧ z.im ≤ 2}

-- Define the transformation
noncomputable def transform (z : ℂ) : ℂ := (1/2 + Complex.I/2) * z

-- Theorem statement
theorem transform_stays_in_S :
  ∀ z ∈ S, transform z ∈ S :=
by
  sorry

#check transform_stays_in_S

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_stays_in_S_l870_87092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_geometric_sequence_a_general_formula_l870_87038

/-- A sequence satisfying the given recurrence relation -/
noncomputable def a : ℕ → ℝ
  | 0 => 0  -- Adding a case for 0 to avoid missing cases error
  | 1 => 1
  | n + 1 => 2 * (n + 1) / n * a n

/-- The sequence b_n defined as a_n / n -/
noncomputable def b (n : ℕ) : ℝ :=
  if n = 0 then 0 else a n / n

/-- Theorem stating that b_n is a geometric sequence with common ratio 2 -/
theorem b_is_geometric_sequence : ∀ n : ℕ, n ≥ 1 → b (n + 1) = 2 * b n := by
  sorry

/-- Theorem stating the general formula for a_n -/
theorem a_general_formula : ∀ n : ℕ, n ≥ 1 → a n = n * 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_geometric_sequence_a_general_formula_l870_87038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l870_87007

/-- Given a geometric sequence with positive terms satisfying certain conditions,
    prove that the minimum value of a specific expression is 12√3. -/
theorem geometric_sequence_minimum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  (2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) →  -- given condition
  (∃ m : ℝ, m = 12 * Real.sqrt 3 ∧ ∀ q > 0, 2 * a 5 + a 4 ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l870_87007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_7_factorial_plus_8_factorial_l870_87069

theorem largest_prime_factor_of_7_factorial_plus_8_factorial : 
  (Nat.factors (Nat.factorial 7 + Nat.factorial 8)).maximum? = some 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_7_factorial_plus_8_factorial_l870_87069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l870_87082

/-- Represents the compound interest problem with varying rates and transactions --/
theorem compound_interest_problem :
  ∃ (P : ℝ),
    (let A1 := P * (1 + 0.09);
     let A2 := (A1 + 500) * (1 + 0.10);
     let A3 := (A2 - 300) * (1 + 0.08);
     let A4 := A3 * (1 + 0.08);
     let A5 := A4 * (1 + 0.09);
     A5 = 1120) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_problem_l870_87082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l870_87067

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x - 3) + Real.sqrt (5 - x)

theorem f_domain : Set.Ioo 3 5 = {x : ℝ | f x ∈ Set.univ} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l870_87067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l870_87040

theorem lambda_range (l : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, Real.sqrt (x * (x^2 + 8) * (8 - x)) < l * (x + 1)) → 
  l ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l870_87040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_values_l870_87044

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x)^2, Real.sin x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_vectors_x_values (x : ℝ) 
  (h1 : 0 < x) (h2 : x < Real.pi) (h3 : perpendicular (m x) (n x)) :
  x = Real.pi / 2 ∨ x = 3 * Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_values_l870_87044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_variance_l870_87021

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- The variance of five terms in an arithmetic sequence -/
noncomputable def variance_five_terms (a : ℕ → ℝ) : ℝ :=
  (1 / 5) * ((a 1 - a 5)^2 + (a 3 - a 5)^2 + (a 5 - a 5)^2 + (a 7 - a 5)^2 + (a 9 - a 5)^2)

theorem arithmetic_sequence_variance (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d → variance_five_terms a = 8 → d = 1 ∨ d = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_variance_l870_87021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_half_l870_87064

theorem angle_sum_is_pi_half (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi/2) 
  (h2 : 0 < β ∧ β < Real.pi/2) 
  (h3 : Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β)) : 
  α + β = Real.pi/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_half_l870_87064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l870_87008

noncomputable def f (a : ℝ) (x : ℝ) := Real.log (a * x^2 - x + 1 / (16 * a))

def p (a : ℝ) : Prop := ∀ x, ∃ y, f a x = y

def q (a : ℝ) : Prop := ∀ x > 0, Real.sqrt (2 * x + 1) < 1 + a * x

theorem range_of_a (h1 : ∀ a, p a ∨ q a) (h2 : ∀ a, ¬(p a ∧ q a)) :
  ∃ a, 1 ≤ a ∧ a ≤ 2 ∧ ∀ b, (1 ≤ b ∧ b ≤ 2) → (p b ∨ q b) ∧ ¬(p b ∧ q b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l870_87008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_lateral_area_l870_87058

/-- The lateral surface area of a regular triangular pyramid -/
noncomputable def lateral_surface_area (a : ℝ) : ℝ :=
  (a^2 * Real.sqrt 39) / 4

/-- Theorem: The lateral surface area of a regular triangular pyramid with base side length a
    and lateral edge making a 60° angle with the base is (a² √39) / 4 -/
theorem regular_triangular_pyramid_lateral_area (a : ℝ) (h : a > 0) :
  let base_side_length := a
  let lateral_edge_angle := 60
  lateral_surface_area base_side_length = (a^2 * Real.sqrt 39) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_triangular_pyramid_lateral_area_l870_87058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_equilateral_triangle_l870_87047

/-- The area of a circle circumscribed about an equilateral triangle -/
theorem circle_area_equilateral_triangle (s : ℝ) (h : s = 12) : 
  π * (s * Real.sqrt 3 / 3)^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_equilateral_triangle_l870_87047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clarissa_copies_l870_87018

-- Define the constants
def cost_per_page : ℚ := 1/20  -- 0.05 as a rational number
def cost_per_binding : ℚ := 5
def pages_per_manuscript : ℕ := 400
def total_cost : ℚ := 250

-- Define the function to calculate the number of copies
def number_of_copies : ℕ :=
  (total_cost / (cost_per_page * pages_per_manuscript + cost_per_binding)).floor.toNat

-- Theorem statement
theorem clarissa_copies : number_of_copies = 10 := by
  -- Unfold the definition of number_of_copies
  unfold number_of_copies
  -- Simplify the arithmetic expressions
  simp [cost_per_page, cost_per_binding, pages_per_manuscript, total_cost]
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clarissa_copies_l870_87018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_difference_l870_87041

theorem number_difference (a b : ℤ) (h1 : a + b = 18350) 
  (h2 : b % 5 = 0) (h3 : (b / 10) * 10 + 5 = a) : 
  |a - b| = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_difference_l870_87041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l870_87029

theorem divisibility_property (k : ℕ) (h1 : k ≥ 1) (h2 : Nat.Coprime k 6) :
  ∃ n : ℕ, k ∣ (2^n + 3^n + 6^n - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l870_87029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operations_on_S_l870_87096

def S : Set ℤ := {n : ℤ | n = 0 ∨ ∃ k : ℤ, n = 2 * k}

theorem operations_on_S :
  (∀ a b : ℤ, a ∈ S → b ∈ S → (a + b) ∈ S) ∧
  (∀ a b : ℤ, a ∈ S → b ∈ S → (a - b) ∈ S) ∧
  (∀ a b : ℤ, a ∈ S → b ∈ S → (a * b) ∈ S) ∧
  (∃ a b : ℤ, a ∈ S ∧ b ∈ S ∧ a ≠ 0 ∧ (b / a) ∉ S) ∧
  (∃ a b : ℤ, a ∈ S ∧ b ∈ S ∧ ((a + b) / 2) ∉ S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_operations_on_S_l870_87096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l870_87099

theorem vector_problem (α β : ℝ) 
  (hα : 0 < α)
  (hβ : α < β)
  (hπ : β < π / 2)
  (a b : ℝ × ℝ)
  (ha : a = (Real.cos α, Real.sin α))
  (hb : b = (Real.cos β, Real.sin β))
  (hperp : (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2) = 0)
  (hdot : a.1 * b.1 + a.2 * b.2 = 4 / 5)
  (htanβ : Real.tan β = 2) :
  Real.tan α = 1 / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l870_87099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_9_1_l870_87081

/-- Represents the minimum number of weighings required to find a fake pearl -/
def min_weighings (total_pearls : ℕ) (fake_pearls : ℕ) : ℕ := sorry

/-- The properties of the pearl problem -/
axiom pearl_properties :
  (∃ (n : ℕ), min_weighings 9 1 = n) ∧
  (∀ (n : ℕ), min_weighings 9 1 ≤ n → ∃ (strategy : Unit), True) ∧
  (∀ (n : ℕ), n < min_weighings 9 1 → ¬∃ (strategy : Unit), True)

/-- Theorem stating that the minimum number of weighings for 9 pearls with 1 fake is 2 -/
theorem min_weighings_9_1 : min_weighings 9 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_weighings_9_1_l870_87081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_ten_congruent_to_one_mod_five_l870_87032

theorem sum_of_first_ten_congruent_to_one_mod_five : 
  (Finset.filter (fun n => n ≥ 2 ∧ n % 5 = 1) (Finset.range 47)).sum id = 235 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_ten_congruent_to_one_mod_five_l870_87032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l870_87001

noncomputable section

/-- Curve C in parametric form -/
def curve_C (θ : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos θ, 2 * Real.sin θ)

/-- Line l in general form -/
def line_l (x y : ℝ) : Prop := 2 * x - y - 6 = 0

/-- Distance from a point to line l -/
def distance_to_line (x y : ℝ) : ℝ :=
  abs (2 * x - y - 6) / Real.sqrt 7

/-- The point with maximum distance -/
def max_point : ℝ × ℝ := (3/2, 1/2)

/-- The maximum distance -/
def max_distance : ℝ := 10 * Real.sqrt 7 / 7

theorem max_distance_point :
  ∀ θ : ℝ, distance_to_line (curve_C θ).1 (curve_C θ).2 ≤ distance_to_line max_point.1 max_point.2 ∧
  distance_to_line max_point.1 max_point.2 = max_distance := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l870_87001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_f_is_inverse_of_g_l870_87050

-- Define the original function g
noncomputable def g (x : ℝ) : ℝ := Real.log (x - 1)

-- Define the proposed inverse function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1

-- State the theorem
theorem inverse_functions (x : ℝ) (hx : x > 1) :
  f (g x) = x ∧ g (f x) = x :=
by sorry

-- Define the domains
def dom_g : Set ℝ := {x | x > 1}
def dom_f : Set ℝ := Set.univ

-- State that f is the inverse of g
theorem f_is_inverse_of_g :
  Function.LeftInverse f g ∧ Function.RightInverse f g ∧
  Function.Injective f ∧ Function.Surjective g ∧
  Set.range g = dom_f ∧ Set.range f = dom_g :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_functions_f_is_inverse_of_g_l870_87050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_theorem_l870_87014

-- Define the curve C in polar coordinates
noncomputable def curve_C (a : ℝ) (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * a * Real.sin (θ + Real.pi / 4)

-- Define the curve C in Cartesian coordinates
def curve_C_cartesian (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a)^2 = 2 * a^2

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x + 1)

-- Define the point P
def point_P : ℝ × ℝ := (-1, 0)

-- Define the theorem
theorem curve_intersection_theorem (a : ℝ) :
  a > 0 →
  (∃ (M N : ℝ × ℝ),
    curve_C_cartesian a M.1 M.2 ∧
    curve_C_cartesian a N.1 N.2 ∧
    line_l M.1 M.2 ∧
    line_l N.1 N.2 ∧
    Real.sqrt ((M.1 - point_P.1)^2 + (M.2 - point_P.2)^2) +
    Real.sqrt ((N.1 - point_P.1)^2 + (N.2 - point_P.2)^2) = 5) →
  a = 2 * Real.sqrt 3 - 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_theorem_l870_87014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l870_87013

def spinner1 : List ℕ := [2, 3, 5, 7, 10]
def spinner2 : List ℕ := [6, 9, 11, 14]

def total_outcomes : ℕ := (spinner1.length : ℕ) * (spinner2.length : ℕ)

def is_even (n : ℕ) : Bool := n % 2 = 0

def even_product_outcomes : ℕ :=
  (spinner1.filter is_even).length * spinner2.length +
  (spinner1.filter (fun x => ¬is_even x)).length * (spinner2.filter is_even).length

theorem spinner_probability :
  (even_product_outcomes : ℚ) / (total_outcomes : ℚ) = 7 / 10 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l870_87013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_to_l_l870_87065

/-- The curve C obtained by compressing the y-coordinate of each point on the circle x^2 + y^2 = 4 to half of its original value -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}

/-- The line l obtained by rotating the line 3x - 2y - 8 = 0 counterclockwise by 90 degrees around the origin -/
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 + 3 * p.2 - 8 = 0}

/-- The distance function from a point to a line -/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |2 * p.1 + 3 * p.2 - 8| / Real.sqrt 13

/-- The maximum distance from any point on curve C to line l is √13 -/
theorem max_distance_C_to_l : 
  ∃ (p : ℝ × ℝ), p ∈ C ∧ ∀ (q : ℝ × ℝ), q ∈ C → distance_to_line q ≤ distance_to_line p ∧ distance_to_line p = Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C_to_l_l870_87065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_reciprocals_l870_87028

theorem min_sum_reciprocals (x y a b : ℝ) : 
  (∀ x y : ℝ, |x + y| + |x - y| = 2) → 
  a > 0 → 
  b > 0 → 
  (∀ x y : ℝ, 4 * a * x + b * y ≤ 1) → 
  (∃ x y : ℝ, 4 * a * x + b * y = 1) → 
  1 / a + 1 / b ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_reciprocals_l870_87028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l870_87083

theorem trigonometric_simplification (x : ℝ) (h1 : Real.sin x ≠ 0) (h2 : 1 + Real.cos x ≠ 0) :
  (Real.sin x / (1 + Real.cos x)) + ((1 + Real.cos x) / Real.sin x) = 2 * (1 / Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l870_87083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_coins_count_l870_87078

/-- Proves that given a collection of dimes and nickels with a total value of $3.10,
    where the number of dimes is 26, the total number of coins is 36. -/
theorem steve_coins_count :
  let total_value : ℚ := 310 / 100  -- $3.10 in decimal form
  let dime_value : ℚ := 10 / 100    -- $0.10 in decimal form
  let nickel_value : ℚ := 5 / 100   -- $0.05 in decimal form
  let num_dimes : ℕ := 26

  let dimes_value : ℚ := num_dimes * dime_value
  let nickels_value : ℚ := total_value - dimes_value
  let num_nickels : ℕ := Int.toNat ((nickels_value / nickel_value).num)

  num_dimes + num_nickels = 36 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_steve_coins_count_l870_87078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teeth_brushing_time_l870_87080

/-- Proves that brushing teeth 3 times a day for 2 minutes each time, over 30 days, results in 3 hours total brushing time -/
theorem teeth_brushing_time 
  (brushings_per_day : ℕ)
  (minutes_per_brushing : ℕ)
  (days : ℕ)
  (minutes_per_hour : ℕ) : 
  brushings_per_day = 3 →
  minutes_per_brushing = 2 →
  days = 30 →
  minutes_per_hour = 60 →
  (brushings_per_day * minutes_per_brushing * days) / minutes_per_hour = 3 := by
  sorry

#check teeth_brushing_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teeth_brushing_time_l870_87080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_is_circle_l870_87042

/-- The equation of the cookie's boundary -/
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 5 = 2*x + 6*y

/-- The radius of the cookie -/
noncomputable def cookie_radius : ℝ := Real.sqrt 5

/-- Theorem stating that the cookie equation represents a circle -/
theorem cookie_is_circle :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    cookie_equation x y ↔ (x - h)^2 + (y - k)^2 = cookie_radius^2 :=
by
  -- Introduce the center coordinates
  let h := 1
  let k := 3
  
  -- Prove the existence of h and k
  use h, k
  
  -- Prove the equivalence for all x and y
  intro x y
  
  -- Expand the definitions and simplify
  simp [cookie_equation, cookie_radius]
  
  -- Algebraic manipulation (this step would normally require more detailed proof)
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_is_circle_l870_87042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_returns_to_initial_l870_87035

/-- Represents the sequence of price changes -/
noncomputable def price_changes (x : ℝ) : List ℝ := [x / 100, -0.1, 0.2, -0.15]

/-- Applies a single price change to the current price -/
noncomputable def apply_change (price : ℝ) (change : ℝ) : ℝ :=
  price * (1 + change)

/-- Applies a list of price changes to the initial price -/
noncomputable def apply_changes (initial_price : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl apply_change initial_price

/-- Theorem stating that the price returns to its initial value if and only if x = 9 -/
theorem price_returns_to_initial (x : ℝ) :
  apply_changes 100 (price_changes x) = 100 ↔ x = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_returns_to_initial_l870_87035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_x_values_exist_l870_87019

/-- The distance from a point (x, y) to the line ax + by + c = 0 -/
noncomputable def distanceToLine (x y a b c : ℝ) : ℝ :=
  |a*x + b*y + c| / Real.sqrt (a^2 + b^2)

/-- A point (x, y) satisfies the equal distance condition if its distances to
    the x-axis, y-axis, and the line x + y = 2 are all equal -/
def satisfiesEqualDistance (x y : ℝ) : Prop :=
  y = x ∧ x = distanceToLine x y 1 1 (-2)

/-- There exist at least two distinct values of x that satisfy the equal distance condition -/
theorem multiple_x_values_exist : ∃ x₁ x₂ y₁ y₂ : ℝ, 
  x₁ ≠ x₂ ∧ 
  satisfiesEqualDistance x₁ y₁ ∧ 
  satisfiesEqualDistance x₂ y₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_x_values_exist_l870_87019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drilled_sphere_equiv_smaller_sphere_l870_87048

/-- The volume of a sphere with a cylindrical hole drilled through its center -/
noncomputable def drilled_sphere_volume (sphere_radius : ℝ) (hole_radius : ℝ) : ℝ :=
  (4 / 3) * Real.pi * sphere_radius^3 - 2 * Real.pi * hole_radius^2 * (sphere_radius^2 - hole_radius^2).sqrt

theorem drilled_sphere_equiv_smaller_sphere 
  (original_radius : ℝ) (hole_radius : ℝ) (new_radius : ℝ)
  (h1 : original_radius = 13)
  (h2 : hole_radius = 5)
  (h3 : new_radius = 12) :
  drilled_sphere_volume original_radius hole_radius = (4 / 3) * Real.pi * new_radius^3 := by
  sorry

#check drilled_sphere_equiv_smaller_sphere

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drilled_sphere_equiv_smaller_sphere_l870_87048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_l870_87051

/-- Proves the distance traveled downstream given upstream conditions -/
theorem downstream_distance 
  (stream_speed : ℝ) 
  (upstream_distance : ℝ) 
  (journey_time : ℝ) 
  (h1 : stream_speed = 3) 
  (h2 : upstream_distance = 60) 
  (h3 : journey_time = 4) : 
  (((upstream_distance / journey_time) + stream_speed) * journey_time) - 
  (stream_speed * journey_time) = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_l870_87051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_c_max_unique_maximum_l870_87054

/-- The quadratic function we want to maximize -/
noncomputable def f (c : ℝ) : ℝ := (1/3) * c^2 - 7*c + 2

/-- The critical point where the maximum occurs -/
noncomputable def c_max : ℝ := 21/2

/-- Theorem stating that f(c_max) is the maximum value of f -/
theorem max_value_at_c_max :
  ∀ c : ℝ, f c ≤ f c_max := by
  sorry

/-- Theorem stating that the maximum at c_max is unique -/
theorem unique_maximum :
  ∀ c : ℝ, c ≠ c_max → f c < f c_max := by
  sorry

#check max_value_at_c_max
#check unique_maximum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_c_max_unique_maximum_l870_87054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_of_cow_l870_87089

/-- Given the total cost of cows and goats, and the average price of a goat,
    calculate the average price of a cow. -/
theorem average_price_of_cow 
  (total_cost : ℕ) 
  (num_cows : ℕ) 
  (num_goats : ℕ) 
  (avg_price_goat : ℕ) 
  (h1 : total_cost = 1500)
  (h2 : num_cows = 2)
  (h3 : num_goats = 10)
  (h4 : avg_price_goat = 70) : 
  (total_cost - num_goats * avg_price_goat) / num_cows = 400 := by
  sorry

#check average_price_of_cow

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_price_of_cow_l870_87089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l870_87097

noncomputable def f (x : Real) : Real := Real.sin x - x

theorem max_value_of_f :
  ∃ (m : Real), m = Real.pi/2 - 1 ∧
  ∀ (x : Real), -Real.pi/2 ≤ x ∧ x ≤ Real.pi/2 → f x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l870_87097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l870_87090

-- Define the new operation ⊕
noncomputable def oplus (a b : ℝ) : ℝ :=
  if a ≥ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  (oplus 1 x) * x - (oplus 2 x)

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-2 : ℝ) 2 ∧ f x = 2 ∧ ∀ (y : ℝ), y ∈ Set.Icc (-2 : ℝ) 2 → f y ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l870_87090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_selection_theorem_l870_87006

/-- The number of ways to choose a committee under specific conditions -/
def committee_selection_count : ℕ := 41

/-- The total number of people in the group -/
def total_people : ℕ := 9

/-- The size of the committee to be formed -/
def committee_size : ℕ := 5

/-- Represents whether two specific people must serve together or not at all -/
def must_serve_together_or_not_at_all : Prop := True

/-- Represents that two specific people refuse to serve with each other -/
def refuse_to_serve_together : Prop := True

/-- Theorem stating the number of ways to choose the committee under given conditions -/
theorem committee_selection_theorem 
  (h1 : must_serve_together_or_not_at_all)
  (h2 : refuse_to_serve_together) :
  (Nat.choose total_people committee_size -
   (Nat.choose (total_people - 2) (committee_size - 2) + 
    Nat.choose (total_people - 2) committee_size)) = committee_selection_count := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_selection_theorem_l870_87006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l870_87033

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Add case for 0
  | 1 => 1
  | n + 2 => (1/16) * (1 + 4 * a (n + 1) + Real.sqrt (1 + 24 * a (n + 1)))

theorem a_general_term (n : ℕ) : 
  a n = 1/3 + (1/2)^n + (2/3) * (1/4)^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l870_87033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_mean_equals_60_l870_87010

noncomputable def regression_line (x : ℝ) : ℝ := 1.5 * x + 45

def x_values : List ℝ := [1, 7, 10, 13, 19]

noncomputable def x_mean : ℝ := (x_values.sum) / x_values.length

theorem y_mean_equals_60 : regression_line x_mean = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_mean_equals_60_l870_87010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_for_given_wire_l870_87084

noncomputable def sphere_radius (wire_radius : ℝ) (wire_length : ℝ) : ℝ :=
  let wire_volume := Real.pi * wire_radius^2 * wire_length
  (3 * wire_volume / (4 * Real.pi))^(1/3)

theorem sphere_radius_for_given_wire : 
  sphere_radius 4 144 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_for_given_wire_l870_87084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_cost_is_8775_l870_87055

/-- Represents the dimensions and costs for paving a trapezoidal room floor. -/
structure RoomPavingData where
  shorter_base : ℚ
  longer_base : ℚ
  height : ℚ
  cost_a : ℚ
  cost_b : ℚ
  area_ratio_a : ℚ
  area_ratio_b : ℚ

/-- Calculates the total cost of paving a trapezoidal room floor. -/
def totalPavingCost (data : RoomPavingData) : ℚ :=
  let total_area := (data.shorter_base + data.longer_base) * data.height / 2
  let area_a := total_area * data.area_ratio_a
  let area_b := total_area * data.area_ratio_b
  area_a * data.cost_a + area_b * data.cost_b

/-- Theorem stating that the total paving cost for the given room is 8775. -/
theorem paving_cost_is_8775 (data : RoomPavingData)
    (h1 : data.shorter_base = 11/2)
    (h2 : data.longer_base = 13/2)
    (h3 : data.height = 15/4)
    (h4 : data.cost_a = 350)
    (h5 : data.cost_b = 450)
    (h6 : data.area_ratio_a = 3/5)
    (h7 : data.area_ratio_b = 2/5) :
    totalPavingCost data = 8775 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paving_cost_is_8775_l870_87055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_vertical_line_inclination_angle_x_eq_neg_one_l870_87020

/-- Represents the inclination angle of a line -/
def InclinationAngle (f : ℝ → ℝ → Prop) : ℝ := sorry

/-- Checks if a point (x, y) is on a line -/
def IsOnLine (f : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := f p.1 p.2

/-- The inclination angle of a vertical line is 90 degrees. -/
theorem inclination_angle_vertical_line :
  ∀ (a : ℝ), (∀ (x y : ℝ), x = a → IsOnLine (fun x y => x = a) (x, y)) → InclinationAngle (fun x y => x = a) = 90 := by
  sorry

/-- The line x = -1 is vertical. -/
def line_x_eq_neg_one (x y : ℝ) : Prop :=
  x = -1

/-- The inclination angle of the line x = -1 is 90 degrees. -/
theorem inclination_angle_x_eq_neg_one :
  InclinationAngle line_x_eq_neg_one = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_vertical_line_inclination_angle_x_eq_neg_one_l870_87020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_inequality_l870_87098

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x - 1 else 1 - x

-- State the theorem
theorem solution_set_of_f_inequality :
  (∀ x, f x = f (|x|)) →  -- f is even
  (∀ x ≥ 0, f x = x - 1) →  -- definition of f for non-negative x
  {x : ℝ | f (x - 1) < 0} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_inequality_l870_87098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l870_87031

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let w := (x * y * z) ^ (1/3)
  (1 + x/y) * (1 + y/z) * (1 + z/x) ≥ 2 + 2 * (x + y + z) / w := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l870_87031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tangent_circle_l870_87004

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := x - y - 4 = 0

/-- The given circle equation -/
def given_circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y = 0

/-- The equation of the circle we want to prove -/
def target_circle_eq (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 2

/-- Theorem stating that the target circle has the smallest radius and is tangent to both the line and the given circle -/
theorem smallest_tangent_circle :
  ∃ (x y : ℝ), 
    (∃ (x₀ y₀ : ℝ), line_eq x₀ y₀ ∧ target_circle_eq x₀ y₀) ∧ 
    (∃ (x₁ y₁ : ℝ), given_circle_eq x₁ y₁ ∧ target_circle_eq x₁ y₁) ∧
    (∀ (r : ℝ) (x₂ y₂ : ℝ), 
      ((x₂ - x)^2 + (y₂ - y)^2 = r^2) → 
      ((∃ (x₃ y₃ : ℝ), line_eq x₃ y₃ ∧ (x₃ - x₂)^2 + (y₃ - y₂)^2 = r^2) ∧
       (∃ (x₄ y₄ : ℝ), given_circle_eq x₄ y₄ ∧ (x₄ - x₂)^2 + (y₄ - y₂)^2 = r^2)) →
      r ≥ Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tangent_circle_l870_87004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_l870_87074

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x - 4 else -2 * x - 4

-- State the theorem
theorem solution_set_f (x : ℝ) :
  (f (x - 2) > 0) ↔ (x < 0 ∨ x > 4) :=
by sorry

-- Define the evenness property of f
axiom f_even (x : ℝ) : f (-x) = f x

-- State that f(x) = 2x - 4 for x ≥ 0
axiom f_def_nonneg (x : ℝ) : x ≥ 0 → f x = 2 * x - 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_l870_87074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l870_87052

theorem platform_length 
  (train_length : ℝ) 
  (platform_crossing_time : ℝ) 
  (pole_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 39)
  (h3 : pole_crossing_time = 24)
  : (train_length * platform_crossing_time / pole_crossing_time) - train_length = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l870_87052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l870_87026

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 - 3 * Real.sin x + 1

-- Define the domain
def domain : Set ℝ := {x | Real.pi/6 ≤ x ∧ x ≤ 5*Real.pi/6}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -1/8 ≤ y ∧ y ≤ 0} := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l870_87026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l870_87027

/-- The time (in seconds) it takes for a train to cross a bridge -/
noncomputable def time_to_cross (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  total_distance / train_speed_mps

/-- Theorem stating that a train 165 meters long, running at 54 kmph, 
    takes 55 seconds to cross a bridge 660 meters in length -/
theorem train_crossing_time :
  time_to_cross 165 54 660 = 55 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_cross 165 54 660

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l870_87027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l870_87061

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) / (x - 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ 1 ∧ x ≠ 2}

-- Theorem stating that the domain of f is [1,2) ∪ (2,+∞)
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l870_87061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_score_combinations_eq_four_l870_87062

/-- Represents the number of different ways to score 19 points in 14 matches -/
def score_combinations : ℕ :=
  Finset.card (Finset.filter 
    (fun p : ℕ × ℕ × ℕ => 
      p.1 + p.2.1 + p.2.2 = 14 ∧ 
      3 * p.1 + p.2.1 = 19 ∧ 
      p.1 ≥ 0 ∧ p.2.1 ≥ 0 ∧ p.2.2 ≥ 0)
    (Finset.product (Finset.range 15) (Finset.product (Finset.range 15) (Finset.range 15))))

/-- Theorem stating that there are exactly 4 ways to score 19 points in 14 matches -/
theorem score_combinations_eq_four : score_combinations = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_score_combinations_eq_four_l870_87062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elvis_writing_time_l870_87094

/-- Represents the time spent on Elvis's album production --/
structure AlbumProduction where
  total_songs : ℕ
  studio_time : ℕ  -- in minutes
  record_time_per_song : ℕ  -- in minutes
  collab_songs : ℕ
  extra_collab_time : ℕ  -- in minutes
  edit_time : ℕ  -- in minutes
  mix_master_time : ℕ  -- in minutes

/-- Calculates the time spent writing each song --/
def time_per_song (a : AlbumProduction) : ℚ :=
  let total_record_time := a.total_songs * a.record_time_per_song + a.collab_songs * a.extra_collab_time
  let total_production_time := total_record_time + a.edit_time + a.mix_master_time
  let total_writing_time := a.studio_time - total_production_time
  (total_writing_time : ℚ) / a.total_songs

/-- Theorem stating that given Elvis's album production parameters, the time spent writing each song is approximately 8.33 minutes --/
theorem elvis_writing_time :
  let elvis_album := AlbumProduction.mk 15 (9 * 60) 18 4 10 45 60
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |time_per_song elvis_album - 8.33| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elvis_writing_time_l870_87094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_valid_configuration_l870_87039

/-- Represents a circular arrangement of 6 people -/
def CircularArrangement := Fin 6

/-- A standard six-sided die -/
def Die := Fin 6

/-- A function representing the roll of each person -/
def RollConfiguration := CircularArrangement → Die

/-- Two positions are adjacent or opposite in the circular arrangement -/
def adjacent_or_opposite (a b : CircularArrangement) : Prop :=
  (a.val + 1 = b.val) ∨ (b.val + 1 = a.val) ∨ (a.val + 3 = b.val) ∨ (b.val + 3 = a.val)

/-- A roll configuration is valid if no adjacent or opposite people have the same roll -/
def valid_configuration (roll : RollConfiguration) : Prop :=
  ∀ a b : CircularArrangement, adjacent_or_opposite a b → roll a ≠ roll b

/-- The total number of possible roll configurations -/
def total_configurations : ℕ := 6^6

/-- The number of valid roll configurations -/
def valid_configurations : ℕ := 6 * 5 * 5 * 4 * 5 * 4

theorem probability_of_valid_configuration :
  (valid_configurations : ℚ) / total_configurations = 125 / 972 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_valid_configuration_l870_87039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l870_87088

def u : Fin 3 → ℝ := ![3, 0, -2]
def v : Fin 3 → ℝ := ![1, -4, 2]

theorem angle_between_vectors (ε : ℝ) (h : ε > 0) :
  ∃ θ : ℝ, Real.cos θ = (-1 : ℝ) / Real.sqrt 273 ∧ 
    |θ * (180 / Real.pi) - 95.74| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l870_87088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_coordinate_is_4_l870_87049

/-- Line l: x + y - 4 = 0 -/
def line_l (x y : ℝ) : Prop := x + y - 4 = 0

/-- Circle: x^2 + y^2 = 4 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Point A lies on line l -/
def point_on_line (x y : ℝ) : Prop := line_l x y

/-- Two points B and C exist on the circle -/
def points_on_circle (xb yb xc yc : ℝ) : Prop := unit_circle xb yb ∧ unit_circle xc yc

/-- Angle BAC = 60° -/
noncomputable def angle_bac_60 (xa ya xb yb xc yc : ℝ) : Prop := 
  ∃ (angle : ℝ), angle = 60 ∧ Real.cos angle = (xb - xa) * (xc - xa) + (yb - ya) * (yc - ya) / 
    (Real.sqrt ((xb - xa)^2 + (yb - ya)^2) * Real.sqrt ((xc - xa)^2 + (yc - ya)^2))

/-- The maximum possible x-coordinate of point A is 4 -/
theorem max_x_coordinate_is_4 (xa ya xb yb xc yc : ℝ) :
  point_on_line xa ya →
  points_on_circle xb yb xc yc →
  angle_bac_60 xa ya xb yb xc yc →
  ∃ (max_x : ℝ), max_x = 4 ∧ ∀ (x : ℝ), point_on_line x ya → x ≤ max_x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_coordinate_is_4_l870_87049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_PAB_l870_87024

/-- The line on which point P moves -/
def line (x y : ℝ) : Prop := 2 * x + y + 4 = 0

/-- The circle C -/
def circleC (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

/-- The distance from a point (x, y) to the center of the circle -/
noncomputable def distToCenter (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- The perimeter of triangle PAB given the distance d from P to the center of the circle -/
noncomputable def perimeterPAB (d : ℝ) : ℝ := 2 * Real.sqrt (d^2 - 1) + 2 * Real.sqrt (1 - 1/d^2)

/-- The minimum perimeter of triangle PAB -/
noncomputable def minPerimeter : ℝ := 4 + 4 * Real.sqrt 5 / 5

theorem min_perimeter_PAB :
  ∀ x y : ℝ, line x y →
  ∀ a b : ℝ, circleC a b →
  ∀ d : ℝ, d ≥ Real.sqrt 5 →
  perimeterPAB d ≥ minPerimeter :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_PAB_l870_87024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l870_87073

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1 : ℚ) * seq.d)

theorem arithmetic_sequence_ratio (seq : ArithmeticSequence) :
  S seq 6 / S seq 3 = 3 → S seq 12 / S seq 9 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l870_87073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_common_denominator_l870_87022

noncomputable section

variable (x y : ℝ)
variable (h1 : x ≠ 0)
variable (h2 : y ≠ 0)

def fraction1 (x y : ℝ) : ℝ := 1 / (x^2 * y)
def fraction2 (x y : ℝ) : ℝ := 2 / (x * y^2)

def simplestCommonDenominator (x y : ℝ) : ℝ := x^2 * y^2

theorem simplest_common_denominator (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) :
  simplestCommonDenominator x y = x^2 * y^2 ∧
  (∃ (a b : ℝ), a * fraction1 x y + b * fraction2 x y = 1 / simplestCommonDenominator x y) ∧
  (∀ (z : ℝ), (∃ (a b : ℝ), a * fraction1 x y + b * fraction2 x y = 1 / z) → z ≥ simplestCommonDenominator x y) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_common_denominator_l870_87022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_touches_circumcircle_l870_87063

/-- A right-angled triangle with vertices A, B, and C, where C is the right angle -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 0

/-- The circumcenter of a right-angled triangle -/
noncomputable def circumcenter (t : RightTriangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1) / 2, (t.A.2 + t.B.2) / 2)

/-- The radius of the circumcircle of a right-angled triangle -/
noncomputable def circumradius (t : RightTriangle) : ℝ :=
  Real.sqrt (((t.A.1 - t.B.1) / 2)^2 + ((t.A.2 - t.B.2) / 2)^2)

/-- The incenter of a right-angled triangle -/
noncomputable def incenter (t : RightTriangle) : ℝ × ℝ :=
  let a := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let b := Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)
  ((a * t.A.1 + b * t.B.1) / (a + b), (a * t.A.2 + b * t.B.2) / (a + b))

/-- The radius of the incircle of a right-angled triangle -/
noncomputable def inradius (t : RightTriangle) : ℝ :=
  let a := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
  let b := Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)
  let c := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
  (a + b - c) / 2

/-- The homothety transformation with center C and ratio 2 -/
def homothety (t : RightTriangle) (p : ℝ × ℝ) : ℝ × ℝ :=
  (2 * (p.1 - t.C.1) + t.C.1, 2 * (p.2 - t.C.2) + t.C.2)

/-- The main theorem -/
theorem incircle_touches_circumcircle (t : RightTriangle) :
  let O := circumcenter t
  let I := incenter t
  let I' := homothety t I
  let R := circumradius t
  let r := inradius t
  Real.sqrt ((O.1 - I'.1)^2 + (O.2 - I'.2)^2) = R - 2 * r := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_touches_circumcircle_l870_87063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_categorize_numbers_l870_87068

def given_numbers : List ℚ := [-4, 1, -3/5, 0, 22/7, 3, -5, -1/3]

def is_non_negative_integer (q : ℚ) : Bool :=
  q ≥ 0 && q.den = 1

def is_negative_fraction (q : ℚ) : Bool :=
  q < 0 && q.den ≠ 1

theorem categorize_numbers :
  (given_numbers.filter is_non_negative_integer).toFinset = {1, 0, 3} ∧
  (given_numbers.filter is_negative_fraction).toFinset = {-3/5, -1/3} := by
  sorry

#eval given_numbers.filter is_non_negative_integer
#eval given_numbers.filter is_negative_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_categorize_numbers_l870_87068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l870_87000

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2014 = 0

-- Define the inclination angle of a line
noncomputable def inclination_angle (m : ℝ) : ℝ := Real.arctan m

-- Theorem statement
theorem line_inclination_angle :
  ∃ (m : ℝ), (∀ x y, line_equation x y → y = m * x + 2014 / (Real.sqrt 3)) ∧
  inclination_angle m = π / 6 := by
  -- We use 'sorry' to skip the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l870_87000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_is_28000_l870_87072

structure Position where
  title : String
  count : Nat
  salary : Nat
deriving Inhabited

def company_data : List Position := [
  ⟨"CEO", 1, 140000⟩,
  ⟨"Senior Vice-President", 4, 95000⟩,
  ⟨"Manager", 12, 80000⟩,
  ⟨"Senior Consultant", 8, 55000⟩,
  ⟨"Consultant", 49, 28000⟩
]

def total_employees : Nat := (company_data.map Position.count).sum

def median_position : Nat := (total_employees + 1) / 2

def cumulative_count (n : Nat) : Nat :=
  (company_data.take n).map Position.count |>.sum

theorem median_salary_is_28000 :
  ∃ i : Nat, i < company_data.length ∧
    cumulative_count i < median_position ∧
    median_position ≤ cumulative_count (i + 1) ∧
    (company_data.get! i).salary = 28000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_salary_is_28000_l870_87072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_plane_angles_lt_two_pi_sum_dihedral_angles_gt_pi_l870_87076

/-- A trihedral angle -/
structure TrihedralAngle where
  /-- The three plane angles of the trihedral angle -/
  planeAngles : Fin 3 → ℝ
  /-- The three dihedral angles of the trihedral angle -/
  dihedralAngles : Fin 3 → ℝ
  /-- All angles are positive -/
  all_positive : ∀ i, planeAngles i > 0 ∧ dihedralAngles i > 0

/-- The sum of plane angles in a trihedral angle is less than 2π -/
theorem sum_plane_angles_lt_two_pi (t : TrihedralAngle) :
  (Finset.sum Finset.univ t.planeAngles) < 2 * Real.pi := by
  sorry

/-- The sum of dihedral angles in a trihedral angle is greater than π -/
theorem sum_dihedral_angles_gt_pi (t : TrihedralAngle) :
  (Finset.sum Finset.univ t.dihedralAngles) > Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_plane_angles_lt_two_pi_sum_dihedral_angles_gt_pi_l870_87076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_three_digit_numbers_l870_87087

def is_valid_pair (a b : ℕ) : Prop :=
  100 ≤ a ∧ a < 1000 ∧
  100 ≤ b ∧ b < 1000 ∧
  (a % 2 = 0 ∨ b % 2 = 0) ∧
  ∃ (d1 d2 d3 d4 d5 d6 : ℕ),
    Finset.toSet {d1, d2, d3, d4, d5, d6} = Finset.toSet {1, 2, 3, 7, 8, 9} ∧
    a = 100 * d1 + 10 * d2 + d3 ∧
    b = 100 * d4 + 10 * d5 + d6

theorem smallest_sum_of_three_digit_numbers :
  ∀ a b : ℕ, is_valid_pair a b → a + b ≥ 561 := by
  sorry

#check smallest_sum_of_three_digit_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_three_digit_numbers_l870_87087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_segment_vector_representation_l870_87043

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the points C, D, and Q
variable (C D Q : V)

-- Define the ratio condition
def ratio_condition (C D Q : V) : Prop :=
  ∃ (t : ℝ), t > 0 ∧ Q - C = (9/2 : ℝ) • (D - C)

-- Theorem statement
theorem extended_segment_vector_representation
  (h : ratio_condition C D Q) :
  ∃ (x y : ℝ), x = -2/5 ∧ y = 7/5 ∧ Q = x • C + y • D :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_segment_vector_representation_l870_87043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_probability_l870_87036

/-- Represents a pyramid with a square base and four identical triangular faces -/
structure Pyramid where
  base : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- The probability of a point lying within one of the spheres -/
noncomputable def probability_in_spheres (r : ℝ) : ℝ :=
  (5 * (4 / 3) * Real.pi * r^3) / ((4 / 3) * Real.pi * (3 * r)^3)

theorem pyramid_sphere_probability (p : Pyramid) (s_inscribed : Sphere) (s_circumscribed : Sphere) 
  (h1 : s_circumscribed.radius = 3 * s_inscribed.radius)
  (h2 : ∀ (s_external : Sphere), s_external.radius = s_inscribed.radius) :
  probability_in_spheres s_inscribed.radius = 5 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_sphere_probability_l870_87036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_product_less_than_one_l870_87023

def prime_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ p => Nat.Prime p ∧ p ∣ n)

def a (n : ℕ) : ℚ :=
  if n < 2 then 0 else (prime_divisors n).sum (λ p => 1 / (p : ℚ))

theorem sum_a_product_less_than_one (N : ℕ) (h : N ≥ 2) :
  (Finset.range (N - 1)).sum (λ k => (Finset.range k).prod (λ i => a (i + 2))) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_a_product_less_than_one_l870_87023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l870_87091

/-- Trapezoid with an inscribed circle -/
structure TrapezoidWithCircle where
  AB : ℝ
  CD : ℝ
  radius : ℝ
  arc_angle : ℝ
  tangent_to_AB : Bool
  tangent_to_BC : Bool
  tangent_to_DA : Bool

/-- The area of a trapezoid with an inscribed circle satisfying specific conditions -/
noncomputable def trapezoid_area (t : TrapezoidWithCircle) : ℝ :=
  (t.AB + t.CD) * (t.radius + t.radius * Real.sin (t.arc_angle / 2)) / 2

/-- Theorem stating the area of the specific trapezoid -/
theorem specific_trapezoid_area : 
  let t : TrapezoidWithCircle := {
    AB := 10,
    CD := 15,
    radius := 6,
    arc_angle := 120 * π / 180,
    tangent_to_AB := true,
    tangent_to_BC := true,
    tangent_to_DA := true
  }
  trapezoid_area t = 225 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l870_87091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l870_87012

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 24) / Real.log (1/3)

-- State the theorem
theorem f_monotone_increasing :
  StrictMonoOn f (Set.Iio (-4)) :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l870_87012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l870_87075

/-- A geometric sequence with positive first term -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h1 : ∀ n, a (n + 1) = a n * q
  h2 : a 1 > 0

/-- "a_1 < a_3" is a necessary but not sufficient condition for "a_3 < a_6" -/
theorem necessary_not_sufficient :
  (∀ seq : GeometricSequence, seq.a 3 < seq.a 6 → seq.a 1 < seq.a 3) ∧
  (∃ seq : GeometricSequence, seq.a 1 < seq.a 3 ∧ seq.a 3 ≥ seq.a 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_not_sufficient_l870_87075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l870_87046

/-- Represents a trapezium with given dimensions -/
structure Trapezium where
  side1 : ℝ
  side2 : ℝ
  height : ℝ

/-- Calculates the area of a trapezium -/
noncomputable def area (t : Trapezium) : ℝ :=
  (t.side1 + t.side2) * t.height / 2

theorem trapezium_properties (t : Trapezium) 
  (h1 : t.side1 = 24)
  (h2 : t.side2 = 18)
  (h3 : t.height = 15) :
  area t = 315 ∧ t.height = 15 := by
  sorry

#check trapezium_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_properties_l870_87046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_preference_percentage_l870_87034

-- Define the number of students preferring each pet type
def dogs : ℕ := 80
def cats : ℕ := 70
def fish : ℕ := 50
def birds : ℕ := 30
def rabbits : ℕ := 40

-- Define the total number of students
def total : ℕ := dogs + cats + fish + birds + rabbits

-- Define the percentage of students preferring fish
def fish_percentage : ℚ := (fish : ℚ) / (total : ℚ) * 100

-- Theorem to prove
theorem fish_preference_percentage :
  18 ≤ fish_percentage ∧ fish_percentage < 19 := by
  sorry

#eval fish_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fish_preference_percentage_l870_87034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l870_87086

/-- Triangle ABC with sides a, b, c corresponding to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector m -/
def m (t : Triangle) : ℝ × ℝ := (2 * t.b - t.c, t.a)

/-- Vector n -/
noncomputable def n (t : Triangle) : ℝ × ℝ := (Real.cos t.C, Real.cos t.A)

/-- Vectors are parallel -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Dot product of vectors AB and AC -/
noncomputable def dotProduct (t : Triangle) : ℝ := t.b * t.c * Real.cos t.A

theorem triangle_properties (t : Triangle) :
  parallel (m t) (n t) →
  dotProduct t = 4 →
  (t.A = Real.pi / 3 ∧ ∃ (a_min : ℝ), a_min = 2 * Real.sqrt 2 ∧ ∀ (a' : ℝ), a' ≥ a_min) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l870_87086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_one_eighteenth_l870_87003

/-- Fibonacci-like sequence defined by b₁ = 2, b₂ = 3, and b_(n+2) = b_(n+1) + b_n for n ≥ 1 -/
def b : ℕ → ℚ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 3
  | (n + 3) => b (n + 2) + b (n + 1)

/-- The sum S = ∑(n=1 to ∞) b_n / 3^(n+2) -/
noncomputable def S : ℚ := ∑' n, b n / (3 : ℚ)^(n + 2)

/-- The sum S equals 1/18 -/
theorem sum_equals_one_eighteenth : S = 1 / 18 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_one_eighteenth_l870_87003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_set_divisibility_l870_87095

theorem prime_set_divisibility (p : ℕ) (hp : p.Prime) (hp5 : p > 5) :
  ∃ x y : ℕ, x ∈ {z | ∃ n : ℕ, z = p - n^2 ∧ n^2 < p} ∧
             y ∈ {z | ∃ n : ℕ, z = p - n^2 ∧ n^2 < p} ∧
             x ≠ y ∧ x ≠ 1 ∧ x ∣ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_set_divisibility_l870_87095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_not_exceed_half_triangle_area_l870_87016

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a square
structure Square where
  M : ℝ × ℝ
  N : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ

-- Function to calculate the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

-- Function to calculate the area of a square
noncomputable def squareArea (s : Square) : ℝ := sorry

-- Function to check if a square is inside a triangle
def isSquareInTriangle (s : Square) (t : Triangle) : Prop := sorry

-- Theorem statement
theorem square_area_not_exceed_half_triangle_area (t : Triangle) (s : Square) :
  isSquareInTriangle s t → squareArea s ≤ (1/2) * triangleArea t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_not_exceed_half_triangle_area_l870_87016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_x_axis_coords_l870_87070

/-- Given a point M in ℝ³, return its symmetric point with respect to the x-axis -/
def symmetric_point_x_axis (M : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (M.1, -M.2.1, -M.2.2)

theorem symmetry_x_axis_coords :
  let M : ℝ × ℝ × ℝ := (-1, -2, 3)
  symmetric_point_x_axis M = (-1, 2, -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_x_axis_coords_l870_87070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_oclock_strikes_l870_87017

/-- Represents a clock that strikes at regular intervals -/
structure StrikingClock where
  /-- Time taken to complete all strikes -/
  total_time : ℚ
  /-- Number of strikes -/
  num_strikes : ℕ

/-- Calculates the time taken for a given number of strikes -/
def time_for_strikes (clock : StrikingClock) (strikes : ℕ) : ℚ :=
  (clock.total_time / (clock.num_strikes - 1 : ℚ)) * ((strikes - 1) : ℚ)

theorem twelve_oclock_strikes (clock : StrikingClock) 
  (h1 : clock.total_time = 8)
  (h2 : clock.num_strikes = 5) :
  time_for_strikes clock 12 = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_oclock_strikes_l870_87017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selected_numbers_sum_l870_87025

theorem selected_numbers_sum (n : ℕ) (S : Finset ℕ) : 
  n ≥ 5 → 
  (∀ x, x ∈ S → x < n) → 
  S.card > (n + 1) / 2 → 
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ a + b = c := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selected_numbers_sum_l870_87025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cotangent_cos_squared_l870_87011

theorem tan_cotangent_cos_squared (x : ℝ) (h : Real.cos x ≠ 0) : 
  (Real.tan x + 1 / Real.tan x) * Real.cos x ^ 2 = 1 / Real.tan x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cotangent_cos_squared_l870_87011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_max_distance_l870_87071

-- Define the curves
noncomputable def C₁ (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)

noncomputable def C₂ (θ : ℝ) : ℝ := 2 * Real.sin θ

noncomputable def C₃ (θ : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.cos θ

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {(0, 0), (Real.sqrt 3 / 2, 3 / 2)}

-- Define the distance between A and B
noncomputable def AB (α : ℝ) : ℝ :=
  |2 * Real.sin α - 2 * Real.sqrt 3 * Real.cos α|

-- Statement of the theorem
theorem intersection_and_max_distance
  (t : ℝ) (α : ℝ) (h₁ : t ≠ 0) (h₂ : 0 ≤ α ∧ α < π) :
  (∀ θ : ℝ, (C₂ θ * Real.cos θ, C₂ θ * Real.sin θ) ∈ intersection_points ∧
            (C₃ θ * Real.cos θ, C₃ θ * Real.sin θ) ∈ intersection_points) ∧
  (∃ α_max : ℝ, ∀ α : ℝ, AB α ≤ AB α_max ∧ AB α_max = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_max_distance_l870_87071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l870_87085

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 10*y + 25 = 0

-- Define the center and radius of circle1
def center1 : ℝ × ℝ := (1, 1)
def radius1 : ℝ := 1

-- Define the center and radius of circle2
def center2 : ℝ × ℝ := (4, 5)
def radius2 : ℝ := 4

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)

-- Theorem: The circles are externally tangent
theorem circles_externally_tangent : distance_between_centers = radius1 + radius2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l870_87085
