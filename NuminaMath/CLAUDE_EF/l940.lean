import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l940_94051

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then x^2 - 4
  else if x > 2 then 2*x
  else 0  -- This else case is added to make the function total

theorem solution_exists (x₀ : ℝ) (h : f x₀ = -2) : x₀ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l940_94051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l940_94031

-- Define the function f
noncomputable def f (m n x : ℝ) : ℝ := (m * x + n) / (x^2 + 1)

-- State the theorem
theorem function_properties
  (m n : ℝ)
  (h_odd : ∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f m n (-x) = -f m n x)
  (h_f_1 : f m n 1 = 1) :
  (m = 2 ∧ n = 0) ∧
  (∀ x y, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x < y → f m n x < f m n y) ∧
  (Set.Icc 0 1 = {a : ℝ | f m n (a - 1) + f m n (a^2 - 1) < 0}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l940_94031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saber_toothed_frog_tails_l940_94037

/-- Represents the number of tails each Saber-toothed Frog tadpole has -/
def x : ℕ := sorry

/-- Represents the number of Triassic Discoglossus tadpoles -/
def n : ℕ := sorry

/-- Represents the number of Saber-toothed Frog tadpoles -/
def k : ℕ := sorry

/-- The total number of legs is 100 -/
axiom total_legs : 5 * n + 4 * k = 100

/-- The total number of tails is 64 -/
axiom total_tails : n + x * k = 64

/-- Theorem stating that each Saber-toothed Frog tadpole has 3 tails -/
theorem saber_toothed_frog_tails : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saber_toothed_frog_tails_l940_94037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l940_94041

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define point A
def point_A : ℝ × ℝ := (-1, 8)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∀ P : ℝ × ℝ, parabola P.1 P.2 →
    ∃ m : ℝ, ∀ Q : ℝ × ℝ, parabola Q.1 Q.2 →
      distance P point_A + distance P focus ≥ m ∧
      m = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l940_94041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_gallons_required_l940_94007

/-- The number of pillars to be painted -/
def num_pillars : ℕ := 20

/-- The height of each pillar in feet -/
def pillar_height : ℝ := 25

/-- The diameter of each pillar in feet -/
def pillar_diameter : ℝ := 8

/-- The area that one gallon of paint covers in square feet -/
def paint_coverage : ℝ := 400

/-- Calculates the lateral surface area of a right cylinder -/
noncomputable def lateral_surface_area (radius height : ℝ) : ℝ :=
  2 * Real.pi * radius * height

/-- Calculates the total paint required in gallons -/
noncomputable def total_paint_required (num_pillars : ℕ) (radius height coverage : ℝ) : ℝ :=
  (num_pillars : ℝ) * lateral_surface_area radius height / coverage

/-- Rounds up a real number to the nearest integer -/
noncomputable def ceil (x : ℝ) : ℤ := ⌈x⌉

theorem paint_gallons_required :
  ceil (total_paint_required num_pillars (pillar_diameter / 2) pillar_height paint_coverage) = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_gallons_required_l940_94007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_4_08_l940_94074

/-- The time (in minutes) it takes for two people walking in opposite directions 
    on a circular track to meet for the first time. -/
noncomputable def meetingTime (trackCircumference : ℝ) (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  trackCircumference / ((speed1 + speed2) * 1000 / 60)

/-- Theorem stating that two people walking in opposite directions on a 
    circular track with given speeds will meet after approximately 4.08 minutes. -/
theorem meeting_time_approx_4_08 :
  let trackCircumference : ℝ := 561
  let speed1 : ℝ := 4.5  -- km/hr
  let speed2 : ℝ := 3.75 -- km/hr
  abs (meetingTime trackCircumference speed1 speed2 - 4.08) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval meetingTime 561 4.5 3.75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_time_approx_4_08_l940_94074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_temperature_formula_temperature_after_100_replacements_replacements_to_reach_28_degrees_l940_94043

/-- Represents the temperature of water in a bathtub after n replacements -/
noncomputable def water_temperature (m v t d : ℝ) (n : ℕ) : ℝ :=
  ((m - v) / m) ^ n * t + (1 - ((m - v) / m) ^ n) * d

/-- Theorem stating the formula for water temperature after n replacements -/
theorem water_temperature_formula (m v t d : ℝ) (n : ℕ) :
  water_temperature m v t d n = ((m - v) / m) ^ n * t + (1 - ((m - v) / m) ^ n) * d :=
by sorry

/-- Theorem for part a of the problem -/
theorem temperature_after_100_replacements :
  ∃ temp : ℝ, abs (water_temperature 40001 201 60 10 100 - temp) < 0.01 ∧ 30.12 < temp ∧ temp < 30.14 :=
by sorry

/-- Theorem for part b of the problem -/
theorem replacements_to_reach_28_degrees :
  ∃ n : ℕ, abs (water_temperature 40001 201 40 15 n - 28) < 0.01 ∧ 128 ≤ n ∧ n ≤ 130 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_temperature_formula_temperature_after_100_replacements_replacements_to_reach_28_degrees_l940_94043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l940_94063

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Point on the ellipse -/
noncomputable def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse a b (Real.sqrt 2) 1

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Line l passing through Q(1, 0) -/
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  x = m * y + 1

/-- Point P -/
def point_p : ℝ × ℝ := (4, 3)

/-- Function to calculate k₁ · k₂ -/
noncomputable def k1k2 (m : ℝ) : ℝ := sorry

/-- Theorem stating the equations of the ellipse and the line -/
theorem ellipse_and_line_equations
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : point_on_ellipse a b) 
  (h4 : eccentricity a b = Real.sqrt 2 / 2) :
  ∃ (m : ℝ), 
    (ellipse 2 (Real.sqrt 2) = ellipse a b) ∧ 
    (line_l m = λ x y => x - y - 1 = 0) ∧
    (∀ m' : ℝ, k1k2 m' ≤ k1k2 m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l940_94063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midrange_sales_increase_l940_94034

structure Product where
  name : String
  quality : ℕ
  price : ℕ

def is_overpriced (p : Product) : Prop := p.price > p.quality

def is_economy (p : Product) : Prop := p.price < p.quality

def is_midrange (p : Product) : Prop := p.price = p.quality

def sales_increase (p : Product) (display : List Product) : Prop :=
  ∃ (q : Product) (r : Product), 
    q ∈ display ∧ r ∈ display ∧
    is_overpriced q ∧ is_economy r ∧
    is_midrange p

theorem midrange_sales_increase (a b c : Product) 
  (h1 : is_overpriced a) 
  (h2 : is_midrange b) 
  (h3 : is_economy c) :
  sales_increase b [a, b, c] :=
by
  use a, c
  simp [sales_increase, is_overpriced, is_economy, is_midrange]
  exact ⟨h1, h3, h2⟩

#check midrange_sales_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midrange_sales_increase_l940_94034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_factorial_l940_94000

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

theorem min_sum_of_product_factorial (x y z w : ℕ+) 
  (h : (x : ℕ) * y * z * w = factorial 12) : 
  (x : ℕ) + y + z + w ≥ 365 ∧ ∃ (a b c d : ℕ+), (a : ℕ) * b * c * d = factorial 12 ∧ (a : ℕ) + b + c + d = 365 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_product_factorial_l940_94000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_trapezoid_perimeter_l940_94097

/-- An isosceles trapezoid circumscribed around a circle. -/
structure CircumscribedTrapezoid where
  /-- The radius of the inscribed circle. -/
  R : ℝ
  /-- The acute angle at the base of the trapezoid. -/
  α : ℝ
  /-- Assumption that R is positive. -/
  R_pos : 0 < R
  /-- Assumption that α is an acute angle. -/
  α_acute : 0 < α ∧ α < π / 2

/-- The perimeter of a circumscribed isosceles trapezoid. -/
noncomputable def perimeter (t : CircumscribedTrapezoid) : ℝ := 8 * t.R / Real.sin t.α

/-- Theorem stating that the perimeter of a circumscribed isosceles trapezoid
    is equal to 8R / sin(α). -/
theorem circumscribed_trapezoid_perimeter (t : CircumscribedTrapezoid) :
  perimeter t = 8 * t.R / Real.sin t.α := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_trapezoid_perimeter_l940_94097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_true_l940_94006

-- Define the function f and its derivative f'
def f : ℝ → ℝ := sorry
def f' : ℝ → ℝ := sorry

-- Define the conditions
axiom f'_even : ∀ x : ℝ, f' (-x) = f' x
axiom f_sum_condition : ∀ x : ℝ, f (2*x) + f (2-2*x) = 3

-- Define the theorem to be proved
theorem not_necessarily_true : 
  ¬(∀ (f f' : ℝ → ℝ), (∀ x : ℝ, f' (-x) = f' x) → (∀ x : ℝ, f (2*x) + f (2-2*x) = 3) → 
    (∀ x : ℝ, f' (f (1-x)) = f' (f (1+x)))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_true_l940_94006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l940_94056

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (2*a + 2) * x + (2*a + 1) * Real.log x

theorem function_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : 1/2 ≤ a ∧ a ≤ 2) 
  (hx₁ : 1 ≤ x₁ ∧ x₁ ≤ 2) 
  (hx₂ : 1 ≤ x₂ ∧ x₂ ≤ 2) 
  (hx_neq : x₁ ≠ x₂) :
  ∃ (l : ℝ), l ≥ 6 ∧ |f a x₁ - f a x₂| < l * |1/x₁ - 1/x₂| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l940_94056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_killers_required_l940_94026

/-- Represents a killer at the convention -/
structure Killer :=
  (id : Nat)
  (victims : Finset Nat)

/-- The total number of participants at the convention -/
def totalParticipants : Nat := 1000

/-- Predicate to check if a killer's actions are valid according to the rules -/
def validKiller (k : Killer) : Prop :=
  k.id ≤ totalParticipants ∧
  k.id > 1 ∧
  k.victims.card ≤ k.id ∧
  ∀ v ∈ k.victims, v > k.id

/-- The set of all participants except number 1 -/
def allVictims : Finset Nat :=
  Finset.range (totalParticipants + 1) \ {1}

/-- Theorem stating the minimum number of killers required -/
theorem min_killers_required :
  ∃ (killers : Finset Killer),
    killers.card = 10 ∧
    (∀ k ∈ killers, validKiller k) ∧
    (killers.biUnion (λ k => k.victims) = allVictims) ∧
    (∀ (smallerSet : Finset Killer),
      smallerSet.card < 10 →
      (∀ k ∈ smallerSet, validKiller k) →
      (smallerSet.biUnion (λ k => k.victims) ≠ allVictims)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_killers_required_l940_94026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_sphere_surface_area_l940_94085

theorem quarter_sphere_surface_area (base_area : ℝ) (h : base_area = 144 * Real.pi) :
  ∃ (total_area : ℝ), total_area = 1008 * Real.pi ∧
  total_area = (3 / 4) * 4 * Real.pi * (base_area / (Real.pi / 4)) + base_area :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quarter_sphere_surface_area_l940_94085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l940_94089

theorem trigonometric_problem (α β : Real) 
  (h1 : α ∈ Set.Ioo (π/2) π)
  (h2 : Real.sin (α/2) + Real.cos (α/2) = Real.sqrt 6 / 2)
  (h3 : Real.sin (α - β) = -3/5)
  (h4 : β ∈ Set.Ioo (π/2) π) : 
  Real.tan (α + π/4) = (15 - 7 * Real.sqrt 3) / 13 ∧ 
  Real.cos β = -(4 * Real.sqrt 3 + 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l940_94089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_radius_maximizes_distance_l940_94069

/-- Given two circles, one fixed with radius 1 and center at (0,0), and another with variable radius r 
    and center at (c,0), where c > 1 and 0 ≤ r ≤ c - 1, this function calculates the radius r that 
    maximizes the distance of the tangent point on the variable circle from their common symmetry axis. -/
noncomputable def optimal_radius (c : ℝ) : ℝ :=
  -3/4 + Real.sqrt (c^2/2 + 1/16)

/-- Theorem stating that the optimal_radius function gives the correct result -/
theorem optimal_radius_maximizes_distance (c : ℝ) (h1 : c > 1) :
  let r := optimal_radius c
  (0 ≤ r ∧ r ≤ c - 1) ∧
  ∀ (s : ℝ), 0 ≤ s ∧ s ≤ c - 1 →
    let t := s / c * Real.sqrt (c^2 - (1 + s)^2)
    let t_opt := r / c * Real.sqrt (c^2 - (1 + r)^2)
    t ≤ t_opt := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_radius_maximizes_distance_l940_94069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supporters_with_all_items_eq_two_l940_94014

/-- The number of seats in the stadium -/
def stadium_capacity : ℕ := 5000

/-- The interval for receiving a t-shirt -/
def t_shirt_interval : ℕ := 25

/-- The interval for receiving a hat -/
def hat_interval : ℕ := 40

/-- The interval for receiving a scarf -/
def scarf_interval : ℕ := 90

/-- Function to calculate the number of supporters receiving all three items -/
def supporters_with_all_items : ℕ :=
  (stadium_capacity / Nat.lcm t_shirt_interval (Nat.lcm hat_interval scarf_interval))

theorem supporters_with_all_items_eq_two : supporters_with_all_items = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supporters_with_all_items_eq_two_l940_94014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_diagonal_l940_94018

theorem square_area_from_diagonal (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let d := 2*a - b
  (d^2 / 2) = (fun s => s^2) ((2*a - b) / Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_diagonal_l940_94018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_cos_3phi_curve_l940_94078

noncomputable def r (φ : Real) : Real := Real.cos (3 * φ)

noncomputable def polar_area (f : Real → Real) (a b : Real) : Real :=
  ∫ x in a..b, (1/2) * (f x)^2

theorem area_of_cos_3phi_curve :
  6 * (polar_area r 0 (π/6)) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_cos_3phi_curve_l940_94078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_product_simplification_l940_94001

open Real

theorem tangent_product_simplification :
  (∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) →
  Real.tan (π/4) = 1 →
  (1 + Real.tan (π/6)) * (1 + Real.tan (π/12)) = 2 := by
  intros tan_sum_formula tan_45_eq_1
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_product_simplification_l940_94001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_and_min_area_l940_94046

-- Define the ellipse C1
def C1 (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define line l1 (implicitly through F1 and perpendicular to x-axis)
def l1 (x : ℝ) : Prop := x = -2

-- Define line l2 (implicitly perpendicular to l1)
def l2 (x y : ℝ) : Prop := ∃ (p : ℝ), y = p ∧ x ≠ -2

-- Define point M
def M (x y : ℝ) : Prop :=
  ∃ (px py : ℝ), l2 px py ∧ 
  (x - px)^2 + (y - py)^2 = (x - F2.fst)^2 + (y - F2.snd)^2

-- Define the trajectory C2
def C2 (x y : ℝ) : Prop := y^2 = 8 * x

-- Define perpendicular lines through F2
def perpendicularLines (k : ℝ) (x y : ℝ) : Prop :=
  (y = k * (x - F2.fst)) ∨ (y = -1/k * (x - F2.fst))

-- Define area of quadrilateral (placeholder function)
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  sorry -- Actual implementation would go here

-- Theorem statement
theorem ellipse_parabola_and_min_area :
  (∀ x y, M x y → C2 x y) ∧
  (∃ min_area : ℝ, min_area = 64/9 ∧
    ∀ k : ℝ, ∀ A B C D : ℝ × ℝ,
      perpendicularLines k A.1 A.2 ∧
      perpendicularLines k B.1 B.2 ∧
      perpendicularLines k C.1 C.2 ∧
      perpendicularLines k D.1 D.2 ∧
      C1 A.1 A.2 ∧ C1 B.1 B.2 ∧ C1 C.1 C.2 ∧ C1 D.1 D.2 →
      area_quadrilateral A B C D ≥ min_area) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_and_min_area_l940_94046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_locations_l940_94039

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the area of a triangle given its three vertices -/
noncomputable def triangleArea (a b c : Point) : ℝ :=
  abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2

/-- Checks if a triangle is a right triangle -/
def isRightTriangle (a b c : Point) : Prop :=
  let ab := distance a b
  let bc := distance b c
  let ca := distance c a
  ab^2 + bc^2 = ca^2 ∨ bc^2 + ca^2 = ab^2 ∨ ca^2 + ab^2 = bc^2

/-- The main theorem to be proved -/
theorem right_triangle_locations (a b : Point) (h : distance a b = 10) :
  (∃ (s : Finset Point), s.card = 8 ∧
    (∀ c ∈ s, isRightTriangle a b c ∧ triangleArea a b c = 20) ∧
    (∀ c, isRightTriangle a b c ∧ triangleArea a b c = 20 → c ∈ s)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_locations_l940_94039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_store_jelly_types_l940_94024

/-- Represents the types of jelly sold in the grocery store -/
inductive JellyType
  | Grape
  | Strawberry
  | Raspberry
  | Plum
deriving Fintype, Repr

/-- Represents the number of jars sold for each jelly type -/
def jars_sold (jelly : JellyType) : ℕ :=
  match jelly with
  | JellyType.Plum => 6
  | JellyType.Strawberry => 18
  | _ => 0  -- We don't know the exact numbers for Grape and Raspberry

/-- The conditions given in the problem -/
axiom grape_strawberry_relation : jars_sold JellyType.Grape = 2 * jars_sold JellyType.Strawberry
axiom raspberry_plum_relation : jars_sold JellyType.Raspberry = 2 * jars_sold JellyType.Plum
axiom raspberry_grape_relation : (3 : ℕ) * jars_sold JellyType.Raspberry = jars_sold JellyType.Grape

/-- The theorem to prove -/
theorem grocery_store_jelly_types : Fintype.card JellyType = 4 := by
  rfl

#eval Fintype.card JellyType

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_store_jelly_types_l940_94024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_function_properties_l940_94087

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x + Real.pi / 3)

theorem tangent_function_properties :
  -- 1. For any two distinct roots of f(x) = 3, their absolute difference is at least π/2
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ = 3 → f x₂ = 3 → |x₁ - x₂| ≥ Real.pi / 2) ∧
  -- 2. (-π/6, 0) is a symmetry center of f
  (∀ x : ℝ, f (x - Real.pi / 6) = -f (-x - Real.pi / 6)) ∧
  -- 3. All points (1/4kπ - π/6, 0) for k ∈ ℤ are symmetry centers of f
  (∀ k : ℤ, ∀ x : ℝ, f (x - (Real.pi / 4 * ↑k - Real.pi / 6)) = -f (-x - (Real.pi / 4 * ↑k - Real.pi / 6))) ∧
  -- 4. f does not have any symmetry axes
  (¬ ∃ a : ℝ, ∀ x : ℝ, f (a + x) = f (a - x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_function_properties_l940_94087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_33_terms_equals_231_l940_94082

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
structure ArithmeticSequence where
  firstTerm : ℚ
  commonDifference : ℚ

/-- The nth term of an arithmetic sequence. -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.firstTerm + (n - 1 : ℚ) * seq.commonDifference

/-- The sum of the first n terms of an arithmetic sequence. -/
def sumFirstNTerms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.firstTerm + (n - 1 : ℚ) * seq.commonDifference)

/-- Theorem: For an arithmetic sequence where the sum of its 7th, 11th, 20th, and 30th terms is 28,
    the sum of its first 33 terms is 231. -/
theorem sum_33_terms_equals_231 (seq : ArithmeticSequence) 
    (h : nthTerm seq 7 + nthTerm seq 11 + nthTerm seq 20 + nthTerm seq 30 = 28) :
  sumFirstNTerms seq 33 = 231 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_33_terms_equals_231_l940_94082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parents_ages_l940_94096

/-- Given Naughty's age, calculate his parents' ages -/
theorem parents_ages (x : ℕ) : ℕ × ℕ := by
  /- Naughty's age is x -/
  let naughty_age := x

  /- Father is 27 years older than Naughty -/
  let father_age := naughty_age + 27

  /- Mother's age is 4 times Naughty's age -/
  let mother_age := 4 * naughty_age

  /- Return the ages of father and mother -/
  exact (father_age, mother_age)

#check parents_ages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parents_ages_l940_94096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_squares_if_triple_product_square_l940_94027

theorem all_squares_if_triple_product_square (S : Finset ℕ+) :
  S.card = 2000 →
  (∀ a b c, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → ∃ k : ℕ+, a * b * c = k ^ 2) →
  ∀ x, x ∈ S → ∃ y : ℕ+, x = y ^ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_squares_if_triple_product_square_l940_94027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celine_erasers_l940_94095

/-- The number of erasers collected by Gabriel -/
def gabriel : ℕ := sorry

/-- The number of erasers collected by Celine -/
def celine : ℕ := 2 * gabriel

/-- The number of erasers collected by Julian -/
def julian : ℕ := 2 * celine

/-- The total number of erasers collected -/
def total : ℕ := gabriel + celine + julian

theorem celine_erasers :
  total = 35 → celine = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_celine_erasers_l940_94095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_at_x_3_l940_94016

-- Define the lines and point M
noncomputable def line1 (x : ℝ) : ℝ := 3 * x - 3
def line2 : ℝ := -1
def M : ℝ × ℝ := (1, 2)

-- Define point B as the intersection of the two lines
noncomputable def B : ℝ × ℝ := (line2, line1 line2)

-- Define the general equation of line AC
noncomputable def lineAC (k : ℝ) (x : ℝ) : ℝ := k * x + 2 - k

-- Define point A
noncomputable def A (k : ℝ) : ℝ × ℝ := ((5 - k) / (3 - k), line1 ((5 - k) / (3 - k)))

-- Define point C
noncomputable def C (k : ℝ) : ℝ × ℝ := (line2, lineAC k line2)

-- Define the area of triangle ABC as a function of k
noncomputable def areaABC (k : ℝ) : ℝ := 2 * (k - 4)^2 / (3 - k)

theorem min_area_at_x_3 :
  ∃ (k : ℝ), (A k).1 = 3 ∧ 
  ∀ (k' : ℝ), k' ≠ k → areaABC k ≤ areaABC k' := by
  -- Proof goes here
  sorry

#check min_area_at_x_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_at_x_3_l940_94016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_calculation_l940_94017

-- Define the given values
noncomputable def investment : ℝ := 7000
noncomputable def interest_rate : ℝ := 10.5
noncomputable def market_value : ℝ := 96.97222222222223
noncomputable def brokerage_fee : ℝ := 0.25

-- Define the calculation steps
noncomputable def effective_market_value : ℝ := market_value - (market_value * brokerage_fee / 100)
noncomputable def face_value : ℝ := (investment / effective_market_value) * 100
noncomputable def income : ℝ := (face_value * interest_rate) / 100

-- State the theorem
theorem investment_income_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |income - 759.50| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_income_calculation_l940_94017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_m_value_l940_94003

def a : ℝ × ℝ := (-2, 3)
def b (m : ℝ) : ℝ × ℝ := (3, m)

theorem orthogonal_vectors_m_value (m : ℝ) :
  (a.1 * (b m).1 + a.2 * (b m).2 = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_m_value_l940_94003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l940_94030

-- Define the given values
noncomputable def a : ℝ := Real.log 10 / Real.log 4
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.sqrt 2

-- State the theorem
theorem order_of_values : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_values_l940_94030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_beautiful_g_is_beautiful_l940_94061

-- Define the concept of a beautiful function
def is_beautiful_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, ∃ y ∈ D, f y = -f x

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := 1 / (x + 2)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos x

-- Define the domains
def D_f : Set ℝ := {x : ℝ | x ≠ -2}
def D_g : Set ℝ := Set.univ

-- Theorem statements
theorem f_is_beautiful : is_beautiful_function f D_f := by sorry

theorem g_is_beautiful : is_beautiful_function g D_g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_beautiful_g_is_beautiful_l940_94061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l940_94092

noncomputable def f (x : ℝ) : ℝ := 1/x - Real.log x / Real.log 2

theorem root_exists_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 := by
  sorry

#check root_exists_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_exists_in_interval_l940_94092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l940_94011

theorem cost_price_calculation (discount : ℝ) (profit_percentage : ℝ) (markup_percentage : ℝ) :
  discount = 50 →
  profit_percentage = 0.20 →
  markup_percentage = 0.4778 →
  ∃ (cost_price : ℝ), abs (cost_price - 179.98) < 0.01 ∧
    (1 + markup_percentage) * cost_price - discount = (1 + profit_percentage) * cost_price :=
by
  intros h_discount h_profit h_markup
  use 179.98
  constructor
  · norm_num
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l940_94011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_l940_94002

/-- Given points P(-3,0) and Q(3,6) in the xy-plane, 
    prove that the value of m that minimizes PR + RQ for point R(1,m) is 4 -/
theorem minimize_distance_sum (P Q R : ℝ × ℝ) : 
  P = (-3, 0) → 
  Q = (3, 6) → 
  R.1 = 1 → 
  (∀ m : ℝ, Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) 
           ≤ Real.sqrt ((1 - P.1)^2 + (m - P.2)^2) + Real.sqrt ((Q.1 - 1)^2 + (Q.2 - m)^2)) → 
  R.2 = 4 := by
  sorry

#check minimize_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_distance_sum_l940_94002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l940_94083

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h : (x / e.a)^2 + (y / e.b)^2 = 1

theorem ellipse_eccentricity_range (e : Ellipse) 
  (p : PointOnEllipse e) 
  (h : p.x^2 - e.a * p.x + p.y^2 = 0) : 
  Real.sqrt 2 / 2 < eccentricity e ∧ eccentricity e < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l940_94083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_single_student_time_two_students_min_time_l940_94012

-- Define the processes
inductive Process : Type
| A | B | C | D | E | F | G

-- Define the time required for each process
def processTime : Process → Nat
| Process.A => 9
| Process.B => 9
| Process.C => 7
| Process.D => 9
| Process.E => 7
| Process.F => 10
| Process.G => 2

-- Define the process order constraints
def processOrder (p q : Process) : Prop :=
  (p = Process.A ∧ (q = Process.C ∨ q = Process.D)) ∨
  (p = Process.B ∧ q = Process.E) ∨
  (p = Process.D ∧ q = Process.E) ∨
  (p = Process.C ∧ q = Process.F) ∨
  (p = Process.D ∧ q = Process.F)

-- Theorem for single student time
theorem single_student_time :
  (List.map processTime [Process.A, Process.B, Process.C, Process.D, Process.E, Process.F, Process.G]).sum = 53 := by sorry

-- Theorem for minimum time with two students
theorem two_students_min_time : ∃ (schedule : List (Process × Nat)),
  (∀ p, p ∈ [Process.A, Process.B, Process.C, Process.D, Process.E, Process.F, Process.G] → ∃ t, (p, t) ∈ schedule) ∧
  (∀ (p q : Process) (t₁ t₂ : Nat), (p, t₁) ∈ schedule → (q, t₂) ∈ schedule → processOrder p q → t₁ + processTime p ≤ t₂) ∧
  (∀ (p : Process) (t : Nat), (p, t) ∈ schedule → t + processTime p ≤ 28) ∧
  (∃ (p : Process) (t : Nat), (p, t) ∈ schedule ∧ t + processTime p = 28) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_single_student_time_two_students_min_time_l940_94012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l940_94015

/-- Given an arithmetic sequence {a_n} with common difference d and a geometric sequence {b_n},
    we define a function f and prove properties about a₆, b_n, and T_n. -/
theorem sequence_properties
  (a b : ℕ → ℚ) (d : ℚ) (f : ℚ → ℚ) (T : ℕ → ℚ) :
  (∀ n, a (n + 1) - a n = d) →  -- a_n is arithmetic with difference d
  (d ≠ 0) →
  (∀ n, b (n + 1) / b n = b 2 / b 1) →  -- b_n is geometric
  (f = λ x ↦ b 1 * x^2 + b 2 * x + b 3) →
  (f 0 = -4) →  -- y-intercept is -4
  (∃ x, ∀ y, f y ≤ f x ∧ f x = a 6 - 7/2) →  -- maximum value is a₆ - 7/2
  (f (a 2 + a 8) = f (a 3 + a 11)) →
  (a 2 = -7/2) →
  (a 6 = 1/2) ∧
  (∀ n, b n = -(-2)^(n-1)) ∧
  (∀ n, T n = -4*n / (9*(2*n-9))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l940_94015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tod_driving_time_l940_94044

/-- Calculates the total driving time given distances and speed -/
noncomputable def driving_time (north_distance : ℝ) (west_distance : ℝ) (speed : ℝ) : ℝ :=
  (north_distance + west_distance) / speed

/-- Proves that Tod's driving time is 6 hours -/
theorem tod_driving_time :
  driving_time 55 95 25 = 6 := by
  -- Unfold the definition of driving_time
  unfold driving_time
  -- Simplify the arithmetic
  simp [add_div]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tod_driving_time_l940_94044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l940_94084

theorem sqrt_inequality : Real.sqrt 8 - Real.sqrt 6 < Real.sqrt 5 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l940_94084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_parrots_count_l940_94073

theorem red_parrots_count (total : ℕ) (green_ratio : ℚ) (red_ratio : ℚ) : 
  total = 120 → 
  green_ratio = 5 / 8 → 
  red_ratio = 1 / 3 → 
  (total - (green_ratio * ↑total).floor) * red_ratio = 15 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_parrots_count_l940_94073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_neg_2_l940_94057

theorem square_of_3x_plus_4_when_x_is_neg_2 :
  (3 * (-2 : ℝ) + 4)^2 = 4 := by
  -- Evaluate the expression inside the parenthesis
  have h1 : 3 * (-2 : ℝ) + 4 = -2 := by
    ring
  
  -- Square the result
  calc
    (3 * (-2 : ℝ) + 4)^2 = (-2)^2 := by rw [h1]
    _ = 4 := by ring
  
  -- QED


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_of_3x_plus_4_when_x_is_neg_2_l940_94057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_circle_minimum_distance_l940_94032

/-- Apollonius Circle Theorem -/
theorem apollonius_circle_minimum_distance
  (M : ℝ × ℝ)  -- Moving point on the Apollonius circle
  (P : ℝ × ℝ)  -- Fixed point P
  (Q : ℝ × ℝ)  -- Fixed point Q
  (B : ℝ × ℝ)  -- Point B
  (h1 : M.1^2 + M.2^2 = 1/4)  -- Equation of the Apollonius circle
  (h2 : Q.2 = 0)  -- Q is on the x-axis
  (h3 : P = (-1, 0))  -- Coordinates of P
  (h4 : dist M Q / dist M P = 1/2)  -- λ = 1/2
  (h5 : B = (1, 2))  -- Coordinates of B
  : ∃ (M : ℝ × ℝ), (1/2 * dist M P + dist M B) = Real.sqrt 89 / 4 ∧
    ∀ (N : ℝ × ℝ), N.1^2 + N.2^2 = 1/4 → 1/2 * dist N P + dist N B ≥ Real.sqrt 89 / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonius_circle_minimum_distance_l940_94032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_placement_exists_l940_94070

-- Define a color type
inductive Color
| A | B | C | D | E | F

-- Define a face type
structure Face where
  color : Color

-- Define a cube type
structure Cube where
  faces : Fin 6 → Face

-- Define a box type
structure Box where
  faces : Fin 6 → Face

-- Define a placement type
structure Placement where
  cube_face_to_box_face : Fin 6 → Fin 6

-- Define a function to check if a placement is valid
def is_valid_placement (c : Cube) (b : Box) (p : Placement) : Prop :=
  ∀ (i : Fin 6), (c.faces i).color ≠ (b.faces (p.cube_face_to_box_face i)).color

-- Theorem statement
theorem cube_placement_exists (c : Cube) (b : Box) 
  (h1 : ∀ (i j : Fin 6), i ≠ j → (c.faces i).color ≠ (c.faces j).color)
  (h2 : ∀ (i j : Fin 6), i ≠ j → (b.faces i).color ≠ (b.faces j).color) :
  ∃ (p : Placement), is_valid_placement c b p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_placement_exists_l940_94070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_x_axis_l940_94033

/-- Given a line with equation ax + by + c = 0, 
    returns the equation of the line symmetric to it with respect to the x-axis -/
def symmetricLineXAxis (a b c : ℝ) : ℝ → ℝ → Prop := λ x y ↦ a * x - b * y + c = 0

theorem symmetric_line_x_axis :
  let original_line := λ x y ↦ 3 * x - 4 * y + 5 = 0
  let symmetric_line := symmetricLineXAxis 3 (-4) 5
  ∀ x y, symmetric_line x y ↔ 3 * x + 4 * y + 5 = 0 := by
  sorry

#check symmetric_line_x_axis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_x_axis_l940_94033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_surface_area_fraction_l940_94066

/-- Represents a cube with its edge length and number of smaller cubes -/
structure Cube where
  edge_length : ℕ
  num_small_cubes : ℕ

/-- Represents the color distribution of smaller cubes -/
structure ColorDistribution where
  num_white : ℕ
  num_black : ℕ

/-- Helper function to calculate total surface area of a cube -/
def total_surface_area (c : Cube) : ℕ :=
  6 * c.edge_length * c.edge_length

/-- Helper function to calculate white surface area -/
def white_surface_area (c : Cube) (cd : ColorDistribution) : ℕ :=
  total_surface_area c - cd.num_black

/-- Helper function representing the number of black cubes on each edge -/
def edge_black_cubes : ℕ := 2

/-- Theorem stating the fraction of white surface area -/
theorem white_surface_area_fraction 
  (c : Cube) 
  (cd : ColorDistribution) 
  (h1 : c.edge_length = 4)
  (h2 : c.num_small_cubes = 64)
  (h3 : cd.num_white = 48)
  (h4 : cd.num_black = 16)
  (h5 : cd.num_white + cd.num_black = c.num_small_cubes)
  (h6 : edge_black_cubes = 2) :
  (white_surface_area c cd : ℚ) / (total_surface_area c : ℚ) = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_surface_area_fraction_l940_94066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_integers_congruent_to_two_mod_four_l940_94098

theorem two_digit_integers_congruent_to_two_mod_four : 
  (Finset.filter (fun n => n % 4 = 2) (Finset.range 100 \ Finset.range 10)).card = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_integers_congruent_to_two_mod_four_l940_94098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_total_bananas_l940_94064

/-- Represents the number of bananas taken by each monkey -/
structure BananaCounts where
  b1 : ℕ
  b2 : ℕ
  b3 : ℕ

/-- Checks if the banana distribution is valid according to the problem conditions -/
def is_valid_distribution (bc : BananaCounts) : Prop :=
  -- Each monkey keeps a fraction and divides the rest equally
  (3 * bc.b1) % 4 = 0 ∧
  bc.b1 % 8 = 0 ∧
  bc.b2 % 4 = 0 ∧
  bc.b2 % 8 = 0 ∧
  bc.b3 % 12 = 0 ∧
  -- The final ratio of bananas for the three monkeys is 3:2:1
  (3 * (bc.b1 / 4 : ℚ) + (bc.b2 / 8 : ℚ) + (11 * bc.b3 / 24 : ℚ)) =
    3 * ((bc.b1 / 8 : ℚ) + (bc.b2 / 4 : ℚ) + (11 * bc.b3 / 24 : ℚ)) ∧
  ((bc.b1 / 8 : ℚ) + (bc.b2 / 4 : ℚ) + (11 * bc.b3 / 24 : ℚ)) =
    2 * ((bc.b1 / 8 : ℚ) + (3 * bc.b2 / 8 : ℚ) + (bc.b3 / 12 : ℚ))

/-- The theorem stating the least possible total number of bananas -/
theorem least_total_bananas :
  ∀ bc : BananaCounts, is_valid_distribution bc →
    bc.b1 + bc.b2 + bc.b3 ≥ 408 := by
  sorry

#check least_total_bananas

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_total_bananas_l940_94064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_mean_and_variance_l940_94049

noncomputable def mean (data : List ℝ) : ℝ := (data.sum) / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let μ := mean data
  (data.map (fun x => (x - μ)^2)).sum / data.length

theorem transformed_mean_and_variance
  (data : List ℝ)
  (h_mean : mean data = 2)
  (h_var : variance data = 3) :
  let new_data := data.map (fun x => 2 * x + 5)
  (mean new_data = 9 ∧ variance new_data = 12) := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_mean_and_variance_l940_94049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_max_ratio_sine_angle_l940_94072

/-- Represents a cone with given slant height -/
structure Cone where
  slantHeight : ℝ
  radius : ℝ
  height : ℝ

/-- The lateral surface area of a cone -/
noncomputable def lateralSurfaceArea (c : Cone) : ℝ :=
  Real.pi * c.radius * c.slantHeight

/-- The volume of a cone -/
noncomputable def volume (c : Cone) : ℝ :=
  (1/3) * Real.pi * c.radius^2 * c.height

/-- The ratio of volume to lateral surface area -/
noncomputable def volumeToSurfaceRatio (c : Cone) : ℝ :=
  volume c / lateralSurfaceArea c

/-- The sine of the angle between the slant height and the base -/
noncomputable def sineAngle (c : Cone) : ℝ :=
  c.height / c.slantHeight

theorem cone_max_ratio_sine_angle (c : Cone) 
  (h1 : c.slantHeight = 3)
  (h2 : c.height^2 + c.radius^2 = c.slantHeight^2)
  (h3 : ∀ c' : Cone, c'.slantHeight = 3 → volumeToSurfaceRatio c' ≤ volumeToSurfaceRatio c) :
  sineAngle c = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_max_ratio_sine_angle_l940_94072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_angle_sum_l940_94019

theorem polygon_angle_sum (n : ℕ) :
  (n - 1) * 180 + 360 = (n + 1) * 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_angle_sum_l940_94019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_two_l940_94045

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - x else x + 1

theorem f_at_two : f 2 = 2 := by
  -- Unfold the definition of f
  unfold f
  -- Since 2 ≥ 0, we use the first case of the if-then-else
  simp [if_pos (show 2 ≥ 0 by norm_num)]
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_two_l940_94045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_one_l940_94094

/-- A polynomial with nonnegative integer coefficients less than 100 -/
def NonNegIntPoly (P : ℝ → ℝ) : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ : ℕ),
    (∀ i : Fin 5, (Vector.ofFn ![a₀, a₁, a₂, a₃, a₄]).get i < 100) ∧
    (∀ x : ℝ, P x = a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀)

/-- The main theorem -/
theorem polynomial_value_at_one
    (P : ℝ → ℝ)
    (h_nonneg : NonNegIntPoly P)
    (h_10 : P 10 = 331633)
    (h_neg10 : P (-10) = 273373) :
  P 1 = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_one_l940_94094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_square_l940_94068

-- Define the hyperbola and circle
def hyperbola (x y : ℝ) : Prop := x * y = 18
def our_circle (x y : ℝ) : Prop := x^2 + y^2 = 36

-- Define a point of intersection
def is_intersection_point (p : ℝ × ℝ) : Prop :=
  hyperbola p.1 p.2 ∧ our_circle p.1 p.2

-- Define the set of all intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | is_intersection_point p}

-- Define a square
def is_square (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c d : ℝ × ℝ), s = {a, b, c, d} ∧
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∧
    (b.1 - c.1)^2 + (b.2 - c.2)^2 = (c.1 - d.1)^2 + (c.2 - d.2)^2 ∧
    (c.1 - d.1)^2 + (c.2 - d.2)^2 = (d.1 - a.1)^2 + (d.2 - a.2)^2 ∧
    (a.1 - c.1)^2 + (a.2 - c.2)^2 = (b.1 - d.1)^2 + (b.2 - d.2)^2

-- Theorem statement
theorem intersection_forms_square :
  is_square intersection_points :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_forms_square_l940_94068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_20_l940_94081

/-- The angle between the hour and minute hands of a clock at a given time --/
noncomputable def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  |60 * (hour % 12 + (minute : ℝ) / 60) - 11 * minute| / 2

/-- Theorem: At 3:20, the angle between the hour and minute hands of a clock is 20° --/
theorem angle_at_3_20 : clock_angle 3 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_at_3_20_l940_94081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_equivalence_l940_94058

theorem operation_equivalence (x : ℝ) (h : x ≠ 0) : 
  abs ((((x / ((8 ^ (2/3 : ℝ)) / 5)) * ((12 : ℝ) / 19) ^ (1/3)) ^ (3/5)) / x - 1.0412) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_operation_equivalence_l940_94058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_specific_existence_l940_94022

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ p x) ↔ (∀ x : ℝ, x ≥ 0 → ¬ p x) :=
by sorry

theorem negation_of_specific_existence :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ (2 : ℝ)^x = 5) ↔ (∀ x : ℝ, x ≥ 0 → (2 : ℝ)^x ≠ 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_specific_existence_l940_94022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_addition_theorem_l940_94048

/-- Represents an octal digit (0-7) -/
def OctalDigit := Fin 8

/-- Represents an octal number as a list of octal digits -/
def OctalNumber := List OctalDigit

/-- Addition operation for octal numbers -/
def octal_add : OctalNumber → OctalNumber → OctalNumber :=
  sorry

/-- Conversion from an octal number to its decimal representation -/
def octal_to_decimal : OctalNumber → ℕ :=
  sorry

/-- Helper function to convert a natural number to an OctalDigit -/
def nat_to_octal_digit (n : ℕ) : OctalDigit :=
  ⟨n % 8, by
    apply Nat.mod_lt
    exact Nat.zero_lt_succ 7⟩

theorem octal_addition_theorem :
  let a : OctalNumber := [nat_to_octal_digit 4, nat_to_octal_digit 5, nat_to_octal_digit 2]
  let b : OctalNumber := [nat_to_octal_digit 1, nat_to_octal_digit 6, nat_to_octal_digit 4]
  let result : OctalNumber := [nat_to_octal_digit 6, nat_to_octal_digit 3, nat_to_octal_digit 6]
  octal_add a b = result :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_addition_theorem_l940_94048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_height_theorem_l940_94047

/-- Represents an L-trimino tile --/
structure LTrimino where
  orientation : Fin 8

/-- Represents the resulting polygon after k L-trimino tiles have fallen --/
structure Polygon where
  tiles : List LTrimino
  width : ℕ
  height : ℚ

/-- The probability that two consecutive L-trimino tiles form a block of height 3 --/
def blockProbability : ℚ := 1 / 4

/-- Calculates the expected height of the polygon --/
noncomputable def expectedHeight (k : ℕ) : ℚ := (7 * k + 1) / 4

/-- Theorem stating the expected height of the polygon --/
theorem expected_height_theorem (k : ℕ) (p : Polygon) :
  p.width = 2 → p.tiles.length = k → expectedHeight k = p.height := by
  sorry

#check expected_height_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_height_theorem_l940_94047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_circumscribed_circle_area_l940_94093

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units -/
noncomputable def circumscribed_circle_area (s : ℝ) : ℝ :=
  let h := (Real.sqrt 3 / 2) * s  -- Altitude of the equilateral triangle
  let R := (2 / 3) * h            -- Radius of the circumscribed circle
  Real.pi * R^2                   -- Area of the circle

/-- Theorem stating that the area of the circumscribed circle is 48π square units -/
theorem equilateral_triangle_circumscribed_circle_area :
  circumscribed_circle_area 12 = 48 * Real.pi := by
  sorry

-- Remove the #eval line as it's not computable
-- #eval circumscribed_circle_area 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_circumscribed_circle_area_l940_94093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_plays_win_probability_s_plays_win_probability_l940_94008

/-- The probability of the best play winning with a majority of votes in a competition -/
noncomputable def bestPlayWinProbability (n : ℕ) (s : ℕ) : ℝ :=
  1 - (1/2)^((s-1)*n)

/-- Theorem: The probability of the best play winning with a majority of votes
    for two plays is 1 - (1/2)^n -/
theorem two_plays_win_probability (n : ℕ) :
  bestPlayWinProbability n 2 = 1 - (1/2)^n := by
  sorry

/-- Theorem: The probability of the best play winning with a majority of votes
    for s plays is 1 - (1/2)^((s-1)*n) -/
theorem s_plays_win_probability (n s : ℕ) :
  bestPlayWinProbability n s = 1 - (1/2)^((s-1)*n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_plays_win_probability_s_plays_win_probability_l940_94008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radius_l940_94054

-- Define Circle as a structure instead of a type synonym
structure Circle (α : Type*) [NormedAddCommGroup α] where
  center : α
  radius : ℝ

-- Define the necessary predicates
def is_inscribed_in_semicircle (c : Circle ℝ) (R : ℝ) : Prop := sorry

def circles_touch_externally (c₁ c₂ : Circle ℝ) : Prop := sorry

def circle_touches_diameter (c : Circle ℝ) (R : ℝ) : Prop := sorry

-- Main theorem
theorem inscribed_circles_radius (R : ℝ) (h_positive : R > 0) :
  (∃ (c₁ c₂ : Circle ℝ),
    (c₁.radius = (Real.sqrt 2 - 1) * R ∧ c₂.radius = (Real.sqrt 2 - 1) * R) ∧
    (is_inscribed_in_semicircle c₁ R ∧ is_inscribed_in_semicircle c₂ R) ∧
    (circles_touch_externally c₁ c₂) ∧
    (circle_touches_diameter c₁ R ∧ circle_touches_diameter c₂ R)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radius_l940_94054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l940_94038

theorem equation_solution : 
  {x : ℝ | (2 : ℝ)^(2*x) - 6 * (2 : ℝ)^x + 8 = 0} = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l940_94038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_triangle_FMN_l940_94077

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point is on the ellipse -/
def isOnEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculate the perimeter of a triangle given three points -/
noncomputable def trianglePerimeter (p1 p2 p3 : Point) : ℝ :=
  distance p1 p2 + distance p2 p3 + distance p3 p1

/-- Theorem: Maximum perimeter of triangle FMN -/
theorem max_perimeter_triangle_FMN (e : Ellipse) (F M N : Point) :
  e.a = 10 → e.b = 8 →
  isOnEllipse e M → isOnEllipse e N →
  F.x = -6 → F.y = 0 →
  ∃ (max_perimeter : ℝ), max_perimeter = 40 ∧
    ∀ (M' N' : Point), isOnEllipse e M' → isOnEllipse e N' →
      trianglePerimeter F M' N' ≤ max_perimeter :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_triangle_FMN_l940_94077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l940_94062

theorem trigonometric_problem (α β : ℝ) 
  (h1 : Real.sin α = 3/5)
  (h2 : α ∈ Set.Ioo (π/2) π)
  (h3 : β ∈ Set.Ioo 0 (π/2))
  (h4 : Real.cos (α - β) = 1/3) :
  (Real.tan (α + π/4) = 1/7) ∧ 
  (Real.cos β = (6 * Real.sqrt 2 - 4) / 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l940_94062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_derivative_positive_l940_94060

open Real

-- Define the functions f, g, and h
noncomputable def f (a x : ℝ) : ℝ := 1/2 * x^2 - (a + 1) * x + 2 * (a - 1) * log x

noncomputable def g (a x : ℝ) : ℝ := -3/2 * x^2 + x + (4 - 2*a) * log x

noncomputable def h (a x : ℝ) : ℝ := f a x + g a x

-- State the theorem
theorem h_derivative_positive (a x₁ x₂ : ℝ) 
  (h_pos : 0 < x₁ ∧ 0 < x₂)
  (h_root₁ : h a x₁ = 0)
  (h_root₂ : h a x₂ = 0)
  (h_order : x₁ < x₂ ∧ x₂ < 4 * x₁) :
  deriv (h a) ((2 * x₁ + x₂) / 3) > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_derivative_positive_l940_94060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l940_94021

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 5)^2 + (y - 3)^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 9 = 0

-- Define the center and radius of each circle
def center1 : ℝ × ℝ := (5, 3)
def radius1 : ℝ := 3

def center2 : ℝ × ℝ := (2, -1)
noncomputable def radius2 : ℝ := Real.sqrt 14

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)

-- Theorem stating that the circles intersect
theorem circles_intersect : 
  distance_between_centers > abs (radius1 - radius2) ∧ 
  distance_between_centers < radius1 + radius2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l940_94021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_problem_l940_94035

def taxi_movements : List Int := [5, -3, -5, 4, -8, 6, -4]
def charge_per_km : Int := 2

theorem taxi_problem :
  let final_position := taxi_movements.sum
  let total_distance := taxi_movements.map (fun x => x.natAbs) |>.sum
  let turnover := total_distance * charge_per_km
  (final_position = -5 ∧ 
   total_distance = 35 ∧ 
   turnover = 70) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_problem_l940_94035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_valid_k_l940_94040

/-- Represents an operation on two rational numbers -/
inductive Operation
  | Add : Operation
  | Sub : Operation
  | SubRev : Operation
  | Mul : Operation
  | Div : Operation
  | DivRev : Operation

/-- Applies an operation to two rational numbers -/
def applyOperation (op : Operation) (a b : ℚ) : Option ℚ :=
  match op with
  | Operation.Add => some (a + b)
  | Operation.Sub => some (a - b)
  | Operation.SubRev => some (b - a)
  | Operation.Mul => some (a * b)
  | Operation.Div => if b ≠ 0 then some (a / b) else none
  | Operation.DivRev => if a ≠ 0 then some (b / a) else none

/-- Represents a sequence of n-1 operations -/
def OperationSequence (n : ℕ) := Fin (n - 1) → Operation

/-- Applies a sequence of operations to a list of rationals -/
def applySequence (n : ℕ) (seq : OperationSequence n) (list : List ℚ) : Option ℚ :=
  sorry

/-- The set of non-negative integers k for which n! can be obtained -/
def validK (n : ℕ) : Set ℕ :=
  {k : ℕ | ∃ (seq : OperationSequence n),
    applySequence n seq (List.range n |>.map (fun i => (k : ℚ) + (i + 1))) = some (n.factorial : ℚ)}

/-- The main theorem -/
theorem finite_valid_k (n : ℕ) (h : n > 100) : Set.Finite (validK n) := by
  sorry

#check finite_valid_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_finite_valid_k_l940_94040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quintic_not_all_real_roots_l940_94005

/-- Given a quintic polynomial with coefficients satisfying 2a₄² < 5a₅a₃,
    not all of its roots are real. -/
theorem quintic_not_all_real_roots
  (a₅ a₄ a₃ a₂ a₁ a₀ : ℝ) 
  (h : 2 * a₄^2 < 5 * a₅ * a₃) :
  ∃ (x : ℂ), (a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀ = 0) ∧ (x.im ≠ 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quintic_not_all_real_roots_l940_94005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_six_balls_two_boxes_l940_94055

/-- The number of ways to distribute n distinguishable balls into 2 indistinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => if k * 2 = n then (n.choose k) / 2 else n.choose k)

/-- Theorem stating that there are 32 ways to distribute 6 distinguishable balls into 2 indistinguishable boxes -/
theorem distribute_six_balls_two_boxes :
  distribute_balls 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_six_balls_two_boxes_l940_94055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l940_94075

noncomputable section

/-- Definition of the ellipse and its properties -/
def Ellipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)

/-- Definition of eccentricity -/
noncomputable def Eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Definition of rhombus area -/
def RhombusArea (a b : ℝ) : ℝ := 4 * a * b

/-- Definition of dot product -/
def DotProduct (x1 y1 x2 y2 : ℝ) : ℝ := x1 * x2 + y1 * y2

/-- Main theorem -/
theorem ellipse_properties :
  ∀ a b : ℝ,
  Ellipse a b →
  Eccentricity a b = Real.sqrt 3 / 2 →
  RhombusArea a b = 4 →
  (∃ x1 y1 y0 : ℝ,
    x1^2 / 4 + y1^2 = 1 ∧
    DotProduct (-2) (-y0) x1 (y1 - y0) = 4 ∧
    (y0 = 2 * Real.sqrt 2 ∨ y0 = -2 * Real.sqrt 2 ∨
     y0 = 2 * Real.sqrt 14 / 5 ∨ y0 = -2 * Real.sqrt 14 / 5)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l940_94075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_modular_property_l940_94088

theorem unique_modular_property (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b) % c = (a * c) % b ∧ 
  (a * c) % b = (b * c) % a ∧ 
  (b * c) % a = (a * b) % c →
  Finset.toSet {a, b, c} = Finset.toSet {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_modular_property_l940_94088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chords_cosine_l940_94090

theorem circle_chords_cosine (r : ℝ) (α β : ℝ) : 
  r > 0 ∧ 
  α > 0 ∧ β > 0 ∧ 
  α + β < π ∧
  2 * r * Real.sin (α/2) = 2 ∧
  2 * r * Real.sin (β/2) = 3 ∧
  2 * r * Real.sin ((α + β)/2) = 4 →
  Real.cos α = 17/32 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chords_cosine_l940_94090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_solution_l940_94020

theorem smallest_angle_solution (x : ℝ) : 
  (∀ y : ℝ, y > 0 ∧ y < x → Real.sin (4 * y) * Real.sin (6 * y) ≠ Real.cos (4 * y) * Real.cos (6 * y)) ∧
  Real.sin (4 * x) * Real.sin (6 * x) = Real.cos (4 * x) * Real.cos (6 * x) ∧
  x > 0 →
  x = 9 * Real.pi / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_solution_l940_94020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_cost_minimization_l940_94029

/-- The annual energy consumption cost function (in ten thousand yuan) --/
noncomputable def C (k : ℝ) (x : ℝ) : ℝ := k / (3 * x + 5)

/-- The total cost function over 20 years (in ten thousand yuan) --/
noncomputable def f (x : ℝ) : ℝ := 6 * x + 20 * C 40 x

/-- Theorem stating the minimum value of f(x) and where it occurs --/
theorem insulation_cost_minimization :
  ∃ (min_cost : ℝ) (min_thickness : ℝ),
    (∀ x, 0 ≤ x → x ≤ 10 → f x ≥ min_cost) ∧
    f min_thickness = min_cost ∧
    min_cost = 70 ∧
    min_thickness = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_insulation_cost_minimization_l940_94029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_consecutive_multiples_of_four_l940_94079

/-- The greatest number that always divides the product of 3 consecutive multiples of 4 is 192 -/
theorem greatest_divisor_consecutive_multiples_of_four :
  ∀ n : ℕ, 
  (∃ m : ℕ, (4 * n) * (4 * (n + 1)) * (4 * (n + 2)) = 192 * m) ∧ 
  (∀ k > 192, ∃ n : ℕ, ¬(k ∣ (4 * n) * (4 * (n + 1)) * (4 * (n + 2)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_divisor_consecutive_multiples_of_four_l940_94079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_abc_l940_94052

/-- The maximum area of triangle ABC given the specified conditions -/
theorem max_area_triangle_abc (A B C : ℝ × ℝ) (O : Set (ℝ × ℝ)) : 
  A = (0, 3) →
  B ∈ O →
  C ∈ O →
  O = {p | p.1^2 + p.2^2 = 25} →
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) →
  ∃ (area : ℝ), area = (25 + 3 * Real.sqrt 41) / 2 ∧
    ∀ (other_area : ℝ), 
      abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2 ≤ other_area →
      other_area ≤ area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_abc_l940_94052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_area_l940_94023

/-- A parabola passing through points (-1,0) and (5,0) with vertex P -/
structure Parabola where
  b : ℝ
  c : ℝ
  passes_through_A : -(-1)^2 + b*(-1) + c = 0
  passes_through_B : -(5)^2 + b*5 + c = 0

/-- The vertex of the parabola -/
noncomputable def vertex (p : Parabola) : ℝ × ℝ := (p.b / 2, -((p.b / 2)^2) + p.b * (p.b / 2) + p.c)

theorem parabola_equation_and_area (p : Parabola) : 
  (∀ x, -x^2 + p.b*x + p.c = -x^2 + 4*x + 5) ∧ 
  (let P := vertex p
   (1/2) * (5 - (-1)) * P.2 = 27) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_and_area_l940_94023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_value_l940_94071

theorem greatest_x_value (x : ℕ) (h : Nat.lcm x (Nat.lcm 15 21) = 105) : 
  x ≤ 105 ∧ ∃ y : ℕ, y = 105 ∧ Nat.lcm y (Nat.lcm 15 21) = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_x_value_l940_94071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_equals_1_over_65536_l940_94050

/-- Represents a 4x4 grid of squares --/
def Grid := Fin 4 → Fin 4 → Bool

/-- Probability of a square being black initially --/
noncomputable def initial_black_prob : ℝ := 1 / 2

/-- Rotates a position 90 degrees clockwise --/
def rotate (i j : Fin 4) : Fin 4 × Fin 4 := sorry

/-- Determines if a square should be black after rotation --/
def is_black_after_rotation (g : Grid) (i j : Fin 4) : Bool := sorry

/-- Probability of the grid being entirely black after rotation --/
noncomputable def prob_all_black_after_rotation (g : Grid) : ℝ := sorry

/-- The main theorem stating the probability of the grid being entirely black after the operation --/
theorem prob_all_black_equals_1_over_65536 : 
  ∀ g : Grid, prob_all_black_after_rotation g = 1 / 65536 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_black_equals_1_over_65536_l940_94050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l940_94013

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | x^2 - 7*x + 10 ≤ 0}

theorem union_of_A_and_B : A ∪ B = Set.Icc (-1) 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l940_94013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l940_94053

/-- Calculate simple interest -/
noncomputable def simpleInterest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Calculate compound interest -/
noncomputable def compoundInterest (principal rate time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating the ratio of simple interest to compound interest -/
theorem interest_ratio :
  (simpleInterest 3225 8 5) / (compoundInterest 8000 15 2) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l940_94053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_white_or_red_ball_l940_94025

theorem probability_white_or_red_ball 
  (a b c : ℕ) -- number of white, red, and yellow balls
  (h : a + b + c > 0) -- ensure total number of balls is positive
  : (a + b : ℚ) / (a + b + c) = probability_of_white_or_red :=
by
  sorry -- skip the proof for now

where
  probability_of_white_or_red : ℚ := (a + b : ℚ) / (a + b + c)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_white_or_red_ball_l940_94025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l940_94086

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Check if a point lies on a hyperbola -/
def onHyperbola (p : Point) (h : Hyperbola) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Origin point -/
def O : Point := ⟨0, 0⟩

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2

/-- Theorem statement -/
theorem hyperbola_triangle_area 
  (h : Hyperbola) 
  (p : Point) 
  (f1 f2 : Point) 
  (h_eq : h.a^2 - h.b^2 = 1) 
  (p_on_h : onHyperbola p h) 
  (p_dist : distance O p = 2) 
  (f_def : distance f1 f2 = 2 * Real.sqrt (h.a^2 + h.b^2)) :
  triangleArea p f1 f2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_triangle_area_l940_94086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_statistics_l940_94042

/-- Represents the types of cycles in the race -/
inductive CycleType
| Unicycle
| Bicycle
| Tricycle
| Quadricycle

/-- Returns the number of wheels for a given cycle type -/
def wheels (c : CycleType) : Nat :=
  match c with
  | .Unicycle => 1
  | .Bicycle => 2
  | .Tricycle => 3
  | .Quadricycle => 4

/-- Returns the cost of a given cycle type -/
def cost (c : CycleType) : Nat :=
  match c with
  | .Unicycle => 50
  | .Bicycle => 100
  | .Tricycle => 150
  | .Quadricycle => 200

/-- Represents the distribution of cycle types in the race -/
def cycleDistribution : List (CycleType × Nat) :=
  [(CycleType.Unicycle, 8), (CycleType.Bicycle, 12), (CycleType.Tricycle, 16), (CycleType.Quadricycle, 4)]

theorem race_statistics :
  (List.sum (cycleDistribution.map (λ (c, n) => n)) = 40) ∧
  (List.sum (cycleDistribution.map (λ (c, n) => wheels c * n)) = 96) ∧
  (List.sum (cycleDistribution.map (λ (c, n) => cost c * n)) = 4800) := by
  sorry

#check race_statistics

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_statistics_l940_94042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_line_l940_94059

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0

-- Define the line
def lineEq (x y : ℝ) : Prop := x - y + 2 = 0

-- Define the center of the circle
def centerPoint : ℝ × ℝ := (-1, -1)

-- State the theorem
theorem distance_from_center_to_line :
  let (cx, cy) := centerPoint
  ∃ d : ℝ, d = |cx - cy + 2| / Real.sqrt 2 ∧ d = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_line_l940_94059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l940_94076

theorem solution_set_inequality :
  {x : ℝ | (x + 3) * (6 - x) ≥ 0} = Set.Icc (-3 : ℝ) 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l940_94076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_bought_at_bookstore_is_five_l940_94009

/-- Represents the number of books in Mary's mystery book library --/
def BookInventory : Type := ℕ

/-- Calculates the number of books bought at the bookstore --/
def books_bought_at_bookstore (initial : ℕ) 
                              (book_club : ℕ) 
                              (yard_sales : ℕ)
                              (from_daughter : ℕ)
                              (from_mother : ℕ)
                              (donated : ℕ)
                              (sold : ℕ)
                              (final : ℕ) : ℕ :=
  final - (initial + book_club + yard_sales + from_daughter + from_mother - donated - sold)

/-- Theorem stating the number of books bought at the bookstore --/
theorem books_bought_at_bookstore_is_five :
  books_bought_at_bookstore 72 12 2 1 4 12 3 81 = 5 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_bought_at_bookstore_is_five_l940_94009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l940_94099

theorem salary_increase_percentage (S : ℝ) (P : ℝ) : 
  S > 0 → 
  S * (1 + P / 100) * (85 / 100) = S * (1 + 2.25 / 100) → 
  abs (P - 20.35) < 0.01 := by
  sorry

#eval (20.352941176 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salary_increase_percentage_l940_94099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l940_94028

theorem equation_solutions :
  let f (x : ℝ) := (3 + Real.sqrt 5) ^ x ^ (1/4) + (3 - Real.sqrt 5) ^ x ^ (1/4)
  ∀ x : ℝ, f x = 2 * Real.sqrt 3 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l940_94028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_equation_l940_94004

/-- A quadratic function with specific properties -/
noncomputable def p (x : ℝ) : ℝ := -12/5 * x^2 + 48/5

/-- The reciprocal of p has vertical asymptotes at x = -2 and x = 2 -/
axiom asymptotes : ∀ (x : ℝ), x ≠ -2 ∧ x ≠ 2 → (1 / p x) ≠ 0

/-- p(-3) equals 12 -/
axiom p_at_neg_three : p (-3) = 12

/-- Theorem: p(x) is equal to -12/5 * x^2 + 48/5 -/
theorem p_equation : ∀ (x : ℝ), p x = -12/5 * x^2 + 48/5 := by
  intro x
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_equation_l940_94004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_cosine_values_l940_94010

theorem tangent_cosine_values (α : ℝ) 
  (h1 : Real.tan (π/4 + α) = 1/7) 
  (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.tan α = -3/4 ∧ Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_cosine_values_l940_94010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_people_per_acre_approx_l940_94067

/-- Represents the population of the country -/
def population : ℕ := 300000000

/-- Represents the area of the country in square kilometers -/
def area : ℕ := 4000000

/-- Represents the number of acres in one square kilometer -/
def acres_per_sq_km : ℚ := 2471 / 10

/-- Calculates the average number of people per acre -/
noncomputable def avg_people_per_acre : ℝ :=
  (population : ℝ) / ((area : ℝ) * (acres_per_sq_km : ℝ))

/-- Theorem stating that the average number of people per acre is approximately 0.3 -/
theorem avg_people_per_acre_approx :
  |avg_people_per_acre - 0.3| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_people_per_acre_approx_l940_94067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brandon_job_applications_l940_94091

theorem brandon_job_applications (total_businesses : ℕ) 
  (h1 : total_businesses = 72) 
  (h2 : total_businesses % 2 = 0) 
  (h3 : total_businesses % 3 = 0) : ℕ := by
  let fired := total_businesses / 2
  let quit := total_businesses / 3
  let available := fired - quit
  have : available = 12 := by
    -- Proof steps would go here
    sorry
  exact 12


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brandon_job_applications_l940_94091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_when_k_is_one_exists_positive_k_max_k_is_three_l940_94080

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x * Real.log x - k * (x - 1)

-- Theorem 1
theorem extreme_value_when_k_is_one :
  ∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f 1 x_min ≤ f 1 x ∧ f 1 x_min = 0 :=
sorry

-- Theorem 2
theorem exists_positive_k :
  ∃ (k : ℕ), ∀ (x : ℝ), x > 1 → f (k : ℝ) x + x > 0 :=
sorry

-- Theorem 3
theorem max_k_is_three :
  (∃ (k : ℕ), ∀ (x : ℝ), x > 1 → f (k : ℝ) x + x > 0) ∧
  (∀ (k : ℕ), (∀ (x : ℝ), x > 1 → f (k : ℝ) x + x > 0) → k ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_when_k_is_one_exists_positive_k_max_k_is_three_l940_94080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_negative_one_point_two_l940_94065

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_of_negative_one_point_two :
  floor (-1.2) = -2 := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_negative_one_point_two_l940_94065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_last_student_l940_94036

/-- Represents a student in the circle. -/
inductive Student
| Alice
| Ben
| Carl
| Dave

/-- Checks if a number contains 5 as a digit or is a multiple of 5. -/
def isEliminationNumber (n : Nat) : Bool :=
  n % 5 = 0 || n.repr.contains '5'

/-- Simulates the counting process and returns the last student remaining. -/
def lastStudent : Student :=
  sorry

/-- Theorem stating that Dave is the last student remaining in the circle. -/
theorem dave_last_student : lastStudent = Student.Dave := by
  sorry

#eval isEliminationNumber 15  -- Should return true
#eval isEliminationNumber 7   -- Should return false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dave_last_student_l940_94036
