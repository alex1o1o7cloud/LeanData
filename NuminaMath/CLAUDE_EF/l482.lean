import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_money_calculation_l482_48222

/-- The amount of money Gwen received for her birthday -/
def birthday_money : ℕ := 7

/-- The amount of money Gwen spent -/
def spent : ℕ := 2

/-- The amount of money Gwen has left -/
def remaining : ℕ := 5

/-- Theorem stating that the birthday money is equal to the sum of spent and remaining money -/
theorem birthday_money_calculation : birthday_money = spent + remaining := by
  rfl

#eval birthday_money
#eval spent
#eval remaining

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_money_calculation_l482_48222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_trapezoid_altitudes_sum_l482_48250

theorem right_angled_trapezoid_altitudes_sum : 
  let line (x y : ℝ) : Prop := 9 * x + 6 * y = 54
  let y_axis (x y : ℝ) : Prop := x = 0
  let y_four (x y : ℝ) : Prop := y = 4
  let distance (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : ℝ := 
    |9 * p.1 + 6 * p.2 - 54| / Real.sqrt 117
  (distance (0, 0) line + distance (0, 4) line) = 28 * Real.sqrt 13 / 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angled_trapezoid_altitudes_sum_l482_48250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l482_48226

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) + 1

theorem range_of_f :
  let S := Set.range f
  S = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l482_48226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l482_48254

theorem purely_imaginary_condition (a : ℝ) : 
  (∀ z : ℂ, z = Complex.mk (a^2 - 1) (a - 1) → z.re = 0 ∧ z.im ≠ 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l482_48254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_l482_48240

/-- The edge length of the inscribed regular tetrahedron -/
def tetrahedron_edge : ℝ := 4

/-- The radius of the circumscribed sphere -/
noncomputable def sphere_radius : ℝ := Real.sqrt 6

/-- The theorem stating the area of the circular cross-section -/
theorem cross_section_area :
  let r := Real.sqrt (sphere_radius ^ 2 - (sphere_radius / 3) ^ 2)
  (π * r ^ 2) = 16 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_l482_48240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_area_calculation_l482_48274

/-- The area of land that can make a profit of $500 every 3 months, given the conditions of the problem -/
noncomputable def profit_area (total_land : ℝ) (num_sons : ℕ) (yearly_profit_per_son : ℝ) (quarterly_profit : ℝ) : ℝ :=
  let land_per_son := total_land / (num_sons : ℝ)
  let yearly_profit_from_area := quarterly_profit * 4
  let profit_ratio := yearly_profit_per_son / yearly_profit_from_area
  (land_per_son / profit_ratio) * 10000

theorem profit_area_calculation :
  profit_area 3 8 10000 500 = 750 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_area_calculation_l482_48274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l482_48208

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x)

-- State the theorem
theorem domain_of_f :
  (∀ y ∈ Set.range f, y < 0) →
  {x : ℝ | ∃ y, f x = y} = Set.Ioo 0 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l482_48208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l482_48219

/-- Given a right triangle with one leg of 16 inches and the opposite angle of 45°,
    the hypotenuse is 16√2 inches. -/
theorem right_triangle_hypotenuse (leg1 : ℝ) (angle : ℝ) (hypotenuse : ℝ)
  (h1 : angle = 45)
  (h2 : leg1 = 16)
  (h3 : hypotenuse = leg1 * Real.sqrt 2) :
  hypotenuse = 16 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l482_48219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l482_48239

open Real

-- Define the functions
noncomputable def f₁ (x : ℝ) := tan x
noncomputable def f₂ (x : ℝ) := sin (abs x)
noncomputable def f₃ (x : ℝ) := abs (sin x)
noncomputable def f₄ (x : ℝ) := abs (cos x)

-- Define the properties
def has_period_pi (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + π) = f x

def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- Theorem statement
theorem function_properties :
  (has_period_pi f₁ ∧ increasing_on_interval f₁ 0 (π/2)) ∧
  (has_period_pi f₃ ∧ increasing_on_interval f₃ 0 (π/2)) ∧
  ¬(has_period_pi f₂ ∧ increasing_on_interval f₂ 0 (π/2)) ∧
  ¬(has_period_pi f₄ ∧ increasing_on_interval f₄ 0 (π/2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l482_48239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_CDFE_l482_48299

/-- The area of quadrilateral CDFE in a rectangle ABCD with AB = 2, AD = 1, and AE = AF = x -/
noncomputable def area_CDFE (x : ℝ) : ℝ := (1/2) * x * (3 - 2*x)

theorem max_area_CDFE :
  ∃ (x_max : ℝ), x_max = 3/4 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → area_CDFE x ≤ area_CDFE x_max ∧
  area_CDFE x_max = 9/16 := by
  sorry

#check max_area_CDFE

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_CDFE_l482_48299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos6_l482_48277

theorem min_sin6_plus_2cos6 :
  ∀ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ 2/3 ∧
  ∃ y : ℝ, Real.sin y ^ 6 + 2 * Real.cos y ^ 6 = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos6_l482_48277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowing_rate_is_four_percent_l482_48294

/-- Calculates the borrowing rate given the principal, borrowing period, lending rate, and yearly gain -/
noncomputable def calculate_borrowing_rate (principal : ℝ) (period : ℝ) (lending_rate : ℝ) (yearly_gain : ℝ) : ℝ :=
  let total_gain := yearly_gain * period
  let lending_interest := principal * lending_rate * period / 100
  let borrowing_interest := lending_interest - total_gain
  (borrowing_interest * 100) / (principal * period)

/-- Theorem stating that given the specific conditions, the borrowing rate is 4% -/
theorem borrowing_rate_is_four_percent (principal : ℝ) (period : ℝ) (lending_rate : ℝ) (yearly_gain : ℝ)
    (h1 : principal = 5000)
    (h2 : period = 2)
    (h3 : lending_rate = 25 / 4)
    (h4 : yearly_gain = 112.5) :
  calculate_borrowing_rate principal period lending_rate yearly_gain = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_borrowing_rate_is_four_percent_l482_48294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_equal_angles_implies_antiparallelogram_l482_48262

/-- A quadrilateral in the Euclidean plane. -/
structure Quadrilateral (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] :=
  (A B C D : P)

/-- An anti-parallelogram is a quadrilateral where opposite sides are equal but not parallel. -/
def is_antiparallelogram {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (q : Quadrilateral P) : Prop :=
  (dist q.A q.B = dist q.C q.D) ∧ 
  (dist q.A q.D = dist q.B q.C) ∧ 
  ¬(∃ (k : ℝ), q.A - q.B = k • (q.C - q.D)) ∧ 
  ¬(∃ (k : ℝ), q.A - q.D = k • (q.B - q.C))

/-- The diagonals of a quadrilateral. -/
def diagonals {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (q : Quadrilateral P) : 
  (P × P) × (P × P) := ((q.A, q.C), (q.B, q.D))

/-- The angle between two vectors. -/
noncomputable def angle_between {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (v w : P) : ℝ :=
  Real.arccos ((inner v w) / (norm v * norm w))

/-- The condition that opposite sides form equal angles with the diagonals. -/
def opposite_sides_equal_angles_with_diagonals {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] 
  (q : Quadrilateral P) : Prop :=
  let ((AC, BD)) := diagonals q
  angle_between (q.A - AC.1) (q.D - AC.1) = angle_between (q.B - AC.2) (q.C - AC.2) ∧
  angle_between (q.A - BD.1) (q.D - BD.2) = angle_between (q.B - BD.1) (q.C - BD.2)

/-- The main theorem: if opposite sides form equal angles with the diagonals,
    then the quadrilateral is an anti-parallelogram. -/
theorem opposite_sides_equal_angles_implies_antiparallelogram 
  {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (q : Quadrilateral P) :
  opposite_sides_equal_angles_with_diagonals q → is_antiparallelogram q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_sides_equal_angles_implies_antiparallelogram_l482_48262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_a_range_valid_a_subset_open_unit_interval_l482_48292

-- Define the function representing the inequality
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - Real.log x / Real.log a

-- State the theorem
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (1/3), f a x < 0) → a ∈ Set.Icc (1/27) 1 := by
  sorry

-- Define the set of valid 'a' values
def valid_a_set : Set ℝ := Set.Icc (1/27) 1

-- State that the valid_a_set is a subset of (0, 1)
theorem valid_a_subset_open_unit_interval :
  valid_a_set ⊆ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_a_range_valid_a_subset_open_unit_interval_l482_48292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l482_48270

-- Define the function g(x) as noncomputable
noncomputable def g (x : ℝ) : ℝ := (3*x - 4)*(x + 2)/(x - 1)

-- State the theorem
theorem inequality_solution :
  {x : ℝ | g x ≤ 0} = Set.Ioc (-2) (4/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l482_48270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_triangle_after_cuts_l482_48281

/-- Represents a triangle with its three angles in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180

/-- Defines the original triangle with angles 20°, 20°, 140° -/
def original_triangle : Triangle where
  angle1 := 20
  angle2 := 20
  angle3 := 140
  sum_180 := by norm_num

/-- Checks if a triangle is similar to the original triangle -/
def is_similar_to_original (t : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t.angle1 = k * original_triangle.angle1 ∧
    t.angle2 = k * original_triangle.angle2 ∧
    t.angle3 = k * original_triangle.angle3

/-- Represents the process of cutting a triangle along an angle bisector -/
noncomputable def cut_along_bisector (t : Triangle) : Triangle × Triangle :=
  sorry

/-- Theorem stating that it's impossible to obtain a similar triangle after cuts -/
theorem no_similar_triangle_after_cuts :
  ¬ ∃ (n : ℕ) (cuts : ℕ → Triangle → Triangle × Triangle),
    ∃ (t : Triangle), 
      (cuts 0 original_triangle).1 = t ∨ (cuts 0 original_triangle).2 = t ∧
      (∀ i : ℕ, i > 0 → i ≤ n →
        (cuts i t).1 = t ∨ (cuts i t).2 = t) ∧
      is_similar_to_original t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_similar_triangle_after_cuts_l482_48281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_y_axis_l482_48279

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := Real.log x / Real.log 2
def f₂ (x : ℝ) : ℝ := x^2
noncomputable def f₃ (x : ℝ) : ℝ := 2^(abs x)
noncomputable def f₄ (x : ℝ) : ℝ := Real.arcsin x

-- Define evenness
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem symmetry_about_y_axis :
  ¬(is_even f₁) ∧ (is_even f₂) ∧ (is_even f₃) ∧ ¬(is_even f₄) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_y_axis_l482_48279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_puzzle_l482_48251

/-- Represents the possible types of inhabitants -/
inductive InhabitantType
  | Knight
  | Liar

/-- Represents the inhabitants of the island -/
structure Inhabitant where
  name : String
  type : InhabitantType

/-- Function to determine if a statement is true based on the inhabitant type -/
def is_statement_true (i : Inhabitant) (statement : Prop) : Prop :=
  match i.type with
  | InhabitantType.Knight => statement
  | InhabitantType.Liar => ¬statement

/-- The main theorem to prove -/
theorem island_puzzle (A B C : Inhabitant) :
  (B.name = "B") →
  (C.name = "C") →
  (is_statement_true B (∃ (x : Inhabitant), x.name = "A" ∧ x.type = InhabitantType.Liar) = False) →
  (is_statement_true C (B.type = InhabitantType.Liar) = True) →
  (B.type = InhabitantType.Liar ∧ C.type = InhabitantType.Knight) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_puzzle_l482_48251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_length_l482_48246

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  C : ℝ

-- Define our specific triangle
noncomputable def ourTriangle : Triangle where
  a := 3
  b := 5
  c := 7  -- We'll prove this is correct
  C := 120 * (Real.pi / 180)  -- Convert to radians

-- State the theorem
theorem side_c_length (t : Triangle) (h1 : t = ourTriangle) : 
  t.c = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_length_l482_48246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l482_48266

/-- Calculates the principal given the final amount, interest rate, and time period. -/
noncomputable def calculate_principal (amount : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  amount / (1 + rate * time)

/-- Theorem stating that given the specified conditions, the calculated principal is approximately 1892.86. -/
theorem principal_calculation (amount : ℝ) (rate : ℝ) (time : ℝ) 
  (h_amount : amount = 2120)
  (h_rate : rate = 0.05)
  (h_time : time = 2.4) : 
  ∃ (p : ℝ), abs (calculate_principal amount rate time - p) < 0.01 ∧ p = 1892.86 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l482_48266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisection_trajectory_l482_48217

-- Define the circles
def circle_C1 (a b : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 6
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 3 = 0

-- Define the bisection property
def bisects_circumference (C1 : ℝ → ℝ → ℝ → ℝ → Prop) (C2 : ℝ → ℝ → Prop) : Prop :=
  ∀ a b, (∀ x y, C1 a b x y → C2 x y) → 
    ∃ l : ℝ → ℝ → Prop, (∀ x y, l x y ↔ C1 a b x y ∧ C2 x y) ∧
    (∃ x₀ y₀, C2 x₀ y₀ ∧ l x₀ y₀)

-- Theorem statement
theorem circle_bisection_trajectory :
  bisects_circumference circle_C1 circle_C2 →
  ∀ a b, (∃ x y, circle_C1 a b x y) → a^2 + b^2 + 2*a + 2*b + 1 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_bisection_trajectory_l482_48217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l482_48231

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the eccentricity of the ellipse
noncomputable def eccentricity : ℝ := Real.sqrt 3 / 2

-- Define the maximum product of distances from a point on the ellipse to the foci
def max_product_distances : ℝ := 4

-- Theorem statement
theorem ellipse_properties :
  (∀ x y, ellipse x y → ∃ f₁ f₂, (x - f₁)^2 + y^2 + (x - f₂)^2 + y^2 = 16) ∧
  (∀ x y, ellipse x y → ∃ f₁ f₂, ((x - f₁)^2 + y^2) * ((x - f₂)^2 + y^2) ≤ max_product_distances^2) ∧
  (eccentricity^2 = 3/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l482_48231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_a_equals_three_l482_48296

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- State the theorem
theorem prove_a_equals_three (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x y, x < y → f a x < f a y)
  (h4 : (f a 1 - f a (-1)) / (f a 2 - f a (-2)) = 3/10) :
  a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_a_equals_three_l482_48296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_ratio_l482_48267

/-- The volume of a right circular cylinder -/
noncomputable def cylinderVolume (height : ℝ) (circumference : ℝ) : ℝ :=
  (height * circumference^2) / (4 * Real.pi)

/-- The ratio of volumes of two cylinders -/
noncomputable def volumeRatio (h1 c1 h2 c2 : ℝ) : ℝ :=
  cylinderVolume h1 c1 / cylinderVolume h2 c2

theorem tank_volume_ratio :
  volumeRatio 5 4 8 10 = 1/10 := by
  -- Unfold the definitions
  unfold volumeRatio cylinderVolume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_ratio_l482_48267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_coloring_l482_48232

/-- 
Given coprime natural numbers a and b where a, b > 2, 
this theorem states that the point (ab + a + b) / 2 on the line ax + by = z 
satisfies the condition that for any integer c, exactly one of the equations 
ax + by = c and ax + by = ab + a + b - c has a solution in natural numbers x and y.
-/
theorem symmetric_coloring (a b : ℕ) (h1 : Nat.Coprime a b) (h2 : a > 2) (h3 : b > 2) :
  ∀ c : ℤ, (∃ x y : ℕ, a * x + b * y = c) ≠ 
           (∃ x y : ℕ, a * x + b * y = a * b + a + b - c) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_coloring_l482_48232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_ranges_l482_48268

-- Define constants and variables
def distance_AB : ℝ := 10
def max_speed_express : ℝ := 150

-- Define train speeds (in km/min)
variable (c₁ c₂ c₃ : ℝ)

-- Define conditions
def condition_1 (c₁ c₂ : ℝ) : Prop := 5 / c₁ + 5 / c₂ = 15
def condition_2 (c₂ c₃ : ℝ) : Prop := 5 / c₂ + 5 / c₃ = 11
def condition_3 (c₁ c₂ : ℝ) : Prop := c₂ ≤ c₁
def condition_4 (c₃ : ℝ) : Prop := c₃ ≤ max_speed_express / 60

-- Define speed ranges
def speed_range_passenger (c₁ : ℝ) : Prop := 2/3 ≤ c₁ ∧ c₁ ≤ 5/6
def speed_range_freight (c₂ : ℝ) : Prop := 5/9 ≤ c₂ ∧ c₂ ≤ 2/3
def speed_range_express (c₃ : ℝ) : Prop := 10/7 ≤ c₃ ∧ c₃ ≤ 5/2

-- Theorem statement
theorem train_speed_ranges (c₁ c₂ c₃ : ℝ) :
  condition_1 c₁ c₂ → condition_2 c₂ c₃ → condition_3 c₁ c₂ → condition_4 c₃ →
  speed_range_passenger c₁ ∧ speed_range_freight c₂ ∧ speed_range_express c₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_ranges_l482_48268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l482_48218

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the centers and radii
def center₁ : ℝ × ℝ := (0, 0)
def center₂ : ℝ × ℝ := (2, 0)
def radius₁ : ℝ := 1
def radius₂ : ℝ := 2

-- Define the common chord
def common_chord (x : ℝ) : Prop := x = 1/4

-- Define the length of AB
noncomputable def length_AB : ℝ := Real.sqrt 15 / 2

theorem circle_properties :
  (∀ x y, C₁ x y ↔ (x - center₁.1)^2 + (y - center₁.2)^2 = radius₁^2) ∧
  (∀ x y, C₂ x y ↔ (x - center₂.1)^2 + (y - center₂.2)^2 = radius₂^2) ∧
  Real.sqrt ((center₂.1 - center₁.1)^2 + (center₂.2 - center₁.2)^2) = 2 ∧
  (∀ x y, C₁ x y ∧ C₂ x y → common_chord x) ∧
  (∃ A B : ℝ × ℝ, C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = length_AB) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l482_48218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_e_minus_3_l482_48298

-- Define e as a constant (base of natural logarithms)
noncomputable def e : ℝ := Real.exp 1

-- Theorem statement
theorem floor_e_minus_3 : ⌊e - 3⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_e_minus_3_l482_48298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_increase_l482_48233

def increase_average (current_innings : ℕ) (current_average : ℚ) (target_increase : ℚ) : ℕ :=
  let current_total := current_innings * current_average
  let new_innings := current_innings + 1
  let new_average := current_average + target_increase
  let new_total := new_innings * new_average
  (new_total - current_total).ceil.toNat

theorem cricket_average_increase 
  (current_innings : ℕ) (current_average : ℚ) (target_increase : ℚ) 
  (h1 : current_innings = 10) 
  (h2 : current_average = 32) 
  (h3 : target_increase = 4) : 
  increase_average current_innings current_average target_increase = 76 := by
  sorry

#eval increase_average 10 32 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_increase_l482_48233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l482_48285

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the upper vertex
def upper_vertex (a b : ℝ) : ℝ × ℝ := (0, b)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define an isosceles right-angled triangle
def is_isosceles_right_triangle (p q r : ℝ × ℝ) : Prop :=
  let pq := (q.1 - p.1, q.2 - p.2)
  let pr := (r.1 - p.1, r.2 - p.2)
  pq.1 * pr.1 + pq.2 * pr.2 = 0 ∧
  pq.1^2 + pq.2^2 = pr.1^2 + pr.2^2

-- Define the line
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the orthocenter condition
def is_orthocenter (h p q r : ℝ × ℝ) : Prop :=
  (h.1 - p.1) * (q.1 - r.1) + (h.2 - p.2) * (q.2 - r.2) = 0 ∧
  (h.1 - q.1) * (p.1 - r.1) + (h.2 - q.2) * (p.2 - r.2) = 0 ∧
  (h.1 - r.1) * (p.1 - q.1) + (h.2 - r.2) * (p.2 - q.2) = 0

theorem ellipse_and_line_theorem (a b : ℝ) 
  (h_pos : a > b ∧ b > 0)
  (h_triangle : is_isosceles_right_triangle origin (upper_vertex a b) focus) :
  (∀ x y, ellipse a b x y ↔ ellipse (Real.sqrt 2) 1 x y) ∧
  (∃ p q : ℝ × ℝ, 
    ellipse (Real.sqrt 2) 1 p.1 p.2 ∧
    ellipse (Real.sqrt 2) 1 q.1 q.2 ∧
    line (-4/3) p.1 p.2 ∧
    line (-4/3) q.1 q.2 ∧
    is_orthocenter focus p q (upper_vertex (Real.sqrt 2) 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l482_48285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l482_48202

-- Define the ceiling function A
noncomputable def A (x : ℝ) : ℤ := Int.ceil x

-- State the theorem
theorem range_of_x (x : ℝ) (h1 : x > 0) (h2 : A (2 * x * (A x)) = 5) :
  x > 1 ∧ x ≤ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l482_48202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_and_cone_volumes_l482_48257

/-- Theorem about cylinder and cone volumes -/
theorem cylinder_and_cone_volumes
  (base_area : ℝ) (height : ℝ)
  (h_base : base_area = 72)
  (h_height : height = 6) :
  base_area * height = 432 ∧ (1 / 3 : ℝ) * base_area * height = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_and_cone_volumes_l482_48257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_nonpositive_range_l482_48221

-- Define the function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 + Real.log x) / x - a

-- Part 1: Monotonicity of f when a = 0
theorem f_monotonicity (x : ℝ) (hx : x > 0) :
  (∀ y ∈ Set.Ioo 0 1, f y 0 < f x 0 → y < x) ∧
  (∀ y ∈ Set.Ioi 1, f y 0 < f x 0 → y > x) := by
  sorry

-- Part 2: Range of a for which f(x) ≤ 0 holds for all x > 0
theorem f_nonpositive_range (a : ℝ) :
  (∀ x > 0, f x a ≤ 0) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_f_nonpositive_range_l482_48221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_multiplied_by_six_l482_48249

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  θ : ℝ

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * t.a * t.b * Real.sin t.θ

-- Define the new triangle after transformation
def newTriangle (t : Triangle) : Triangle where
  a := 3 * t.a
  b := 2 * t.b
  θ := t.θ

-- Theorem statement
theorem area_multiplied_by_six (t : Triangle) :
  area (newTriangle t) = 6 * area t := by
  -- Unfold the definitions
  unfold area newTriangle
  -- Simplify the expressions
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_multiplied_by_six_l482_48249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_l482_48259

/-- ω is a nonreal complex cube root of unity -/
noncomputable def ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

/-- The set of ordered pairs (a,b) of integers such that |a ω - b| = 1 -/
def S : Set (ℤ × ℤ) := {p : ℤ × ℤ | Complex.abs (p.1 • ω - p.2) = 1}

/-- The cardinality of S is 6 -/
theorem count_pairs : Finset.card (Finset.filter (fun p => Complex.abs (p.1 • ω - p.2) = 1) (Finset.product (Finset.range 3) (Finset.range 3))) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_l482_48259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_combination_l482_48242

open Real

/-- Given two vectors in 2D space with specific magnitudes and angle between them,
    prove that the magnitude of their linear combination is equal to 2. -/
theorem magnitude_of_vector_combination (a b : ℝ × ℝ) :
  let angle := 30 * π / 180
  ‖a‖ = 2 →
  ‖b‖ = Real.sqrt 3 →
  a.1 * b.1 + a.2 * b.2 = ‖a‖ * ‖b‖ * cos angle →
  ‖a - 2 • b‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_combination_l482_48242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_properties_l482_48205

def is_permutation (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ∈ Finset.range 2016 → ∃! k : ℕ, k < 2016 ∧ a k = n + 1

def sum_consecutive (a : ℕ → ℕ) (m n : ℕ) : ℕ :=
  Finset.sum (Finset.range m) (λ i => a (n + i))

theorem permutation_properties (a : ℕ → ℕ) (h : is_permutation a) :
  (∀ n, n < 2009 → sum_consecutive a 8 n ≤ 16100) ∧
  (∃ a', is_permutation a' ∧ ∀ k, k < 252 → sum_consecutive a' 8 (8*k) = 8068) ∧
  (¬ ∃ a', is_permutation a' ∧ ∀ k, k < 672 → sum_consecutive a' 3 (3*k) = (sum_consecutive a' 3 0)) ∧
  (∀ n, n < 2009 → sum_consecutive a 8 n ≤ 8068) →
  (∃ n, n < 2009 ∧ sum_consecutive a 8 n = 8068) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_properties_l482_48205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_of_two_l482_48288

def number : ℕ := 20231222

def count_digit (n : ℕ) (d : ℕ) : ℕ :=
  (Nat.digits 10 n).filter (· = d) |>.length

def total_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).length

theorem frequency_of_two :
  (count_digit number 2 : ℚ) / (total_digits number : ℚ) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frequency_of_two_l482_48288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_C_range_of_μ_l482_48220

-- Define the parametric equations of line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 - (Real.sqrt 3 / 2) * t, 1 + (1 / 2) * t)

-- Define the polar equation of curve C
noncomputable def curve_C (θ : ℝ) : ℝ :=
  4 * Real.cos (θ - Real.pi / 6)

-- Theorem for the Cartesian equation of curve C
theorem cartesian_equation_C :
  ∀ x y : ℝ, (x - Real.sqrt 3)^2 + (y - 1)^2 = 4 ↔
  ∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ := by
  sorry

-- Define μ
noncomputable def μ (x y : ℝ) : ℝ := Real.sqrt 3 * x + y

-- Theorem for the range of μ
theorem range_of_μ :
  ∀ t : ℝ, -2 ≤ t ∧ t ≤ 2 →
  let (x, y) := line_l t
  2 ≤ μ x y ∧ μ x y ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_C_range_of_μ_l482_48220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_roots_relation_k_values_l482_48214

/-- The quadratic equation x² + (2k+1)x + k² + k = 0 -/
def quadratic_equation (k x : ℝ) : Prop :=
  x^2 + (2*k + 1)*x + k^2 + k = 0

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ :=
  (2*k + 1)^2 - 4*(k^2 + k)

/-- The roots of the quadratic equation satisfy x₁ + x₂ = x₁x₂ - 1 -/
def roots_relation (x₁ x₂ : ℝ) : Prop :=
  x₁ + x₂ = x₁*x₂ - 1

theorem quadratic_equation_roots (k : ℝ) :
  (∀ x, quadratic_equation k x → x ∈ Set.univ) ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ quadratic_equation k x₁ ∧ quadratic_equation k x₂) :=
sorry

theorem roots_relation_k_values :
  ∀ k x₁ x₂,
    quadratic_equation k x₁ →
    quadratic_equation k x₂ →
    roots_relation x₁ x₂ →
    (k = 0 ∨ k = -3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_roots_relation_k_values_l482_48214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_points_existence_l482_48271

/-- The function f(x) = (1/3)x³ - x² --/
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2

/-- The derivative of f(x) --/
noncomputable def f' (x : ℝ) : ℝ := x^2 - 2*x

/-- Theorem stating the equivalence between the existence of two mean value points and the range of b --/
theorem mean_value_points_existence (b : ℝ) :
  (∃ m₁ m₂ : ℝ, 0 < m₁ ∧ m₁ < m₂ ∧ m₂ < b ∧
    f b - f 0 = f' m₁ * b ∧
    f b - f 0 = f' m₂ * b ∧
    ∀ m : ℝ, 0 < m ∧ m < b ∧ f b - f 0 = f' m * b → m = m₁ ∨ m = m₂) ↔
  (3/2 < b ∧ b < 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_value_points_existence_l482_48271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player1_wins_l482_48203

/-- The maximum number allowed in the game -/
def max_num : ℕ := 10

/-- The set of natural numbers from 1 to max_num -/
def game_numbers : Finset ℕ := Finset.range max_num

/-- A function to check if a number is a valid move given the current board state -/
def is_valid_move (board : Finset ℕ) (n : ℕ) : Prop :=
  n ∈ game_numbers ∧ n ∉ board ∧ ∀ m ∈ board, ¬(n ∣ m)

/-- The winning strategy for Player 1 -/
noncomputable def winning_strategy (board : Finset ℕ) : ℕ :=
  if 6 ∉ board then 6
  else if 5 ∉ board ∧ (∀ m ∈ board, ¬(5 ∣ m)) then 5
  else if 9 ∉ board ∧ (∀ m ∈ board, ¬(9 ∣ m)) then 9
  else 10

/-- Theorem stating that Player 1 has a winning strategy -/
theorem player1_wins :
  ∃ (strategy : Finset ℕ → ℕ),
    ∀ (board : Finset ℕ),
      is_valid_move board (strategy board) →
      ∀ (n : ℕ), is_valid_move (insert (strategy board) board) n →
        ∃ (m : ℕ), is_valid_move (insert n (insert (strategy board) board)) m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player1_wins_l482_48203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_once_all_same_l482_48237

noncomputable def num_dice : ℕ := 10
noncomputable def num_rolls : ℕ := 5
noncomputable def num_faces : ℝ := 6

noncomputable def prob_all_same_face : ℝ := (1 / num_faces) ^ num_dice

noncomputable def prob_not_all_same_face : ℝ := 1 - prob_all_same_face

noncomputable def prob_never_all_same : ℝ := prob_not_all_same_face ^ num_rolls

theorem prob_at_least_once_all_same (num_dice : ℕ) (num_rolls : ℕ) (num_faces : ℝ) :
  1 - (1 - (1 / num_faces) ^ num_dice) ^ num_rolls =
  1 - prob_never_all_same :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_at_least_once_all_same_l482_48237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_employment_percentage_E_approx_l482_48216

/-- The percentage of employed population in town X -/
noncomputable def E : ℝ :=
  0.46 / (1 - 0.28125)

/-- The percentage of employed males in the population -/
def employed_males : ℝ := 0.46

/-- The percentage of females among employed people -/
def employed_females_ratio : ℝ := 0.28125

theorem employment_percentage :
  abs (E - 0.6397) < 0.0001 := by
  sorry

-- We can't use #eval for noncomputable definitions
-- Instead, we can state a theorem about the approximate value of E
theorem E_approx : abs (E - 0.6397) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_employment_percentage_E_approx_l482_48216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_product_l482_48252

-- Define the ellipse properties
structure Ellipse (O A B C D F : ℝ × ℝ) : Prop where
  -- O is the center
  -- AB is the major axis
  -- CD is the minor axis
  -- F is a focus
  -- We don't need to explicitly define these properties for this problem

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the diameter of the incircle of a triangle
noncomputable def incircleDiameter (p q r : ℝ × ℝ) : ℝ :=
  -- We don't need to define this explicitly for the problem
  0 -- Placeholder value

-- State the theorem
theorem ellipse_product (O A B C D F : ℝ × ℝ) :
  Ellipse O A B C D F →
  distance O F = 8 →
  incircleDiameter O C F = 6 →
  distance A B * distance C D = 175 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_product_l482_48252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_component_upper_bound_l482_48263

/-- The cost per component for a computer manufacturer -/
def cost_per_component : ℝ := 0 -- We'll define it as 0 initially

/-- The shipping cost per unit -/
def shipping_cost : ℝ := 7

/-- The fixed monthly costs -/
def fixed_costs : ℝ := 16500

/-- The monthly production and sales volume -/
def monthly_volume : ℕ := 150

/-- The lowest selling price for break-even -/
def lowest_price : ℝ := 198.33

/-- Theorem stating that the cost per component is at most $81.33 -/
theorem cost_per_component_upper_bound :
  cost_per_component ≤ 81.33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_component_upper_bound_l482_48263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l482_48204

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

def A : Set ℝ := {x | -1 < x ∧ x < 1}

theorem problem_solution :
  ∀ a : ℝ,
  (∀ x : ℝ, x ∈ Set.Ioo a (a + 1) → x ∈ A) →
  (a ∈ Set.Icc (-1) 0) ∧
  (∀ x ∈ A, f (-x) = -f x) ∧
  (∃ x ∈ A, f (-x) ≠ f x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l482_48204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l482_48215

noncomputable def f (x : ℝ) := Real.cos x ^ 4 + Real.sin x ^ 2

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l482_48215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l482_48245

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) : 
  (t.a - t.c) * (t.a + t.c) * Real.sin t.C = t.c * (t.b - t.c) * Real.sin t.B →
  (1/2) * t.b * t.c * Real.sin t.A = Real.sqrt 3 →
  Real.sin t.B * Real.sin t.C = 1/4 →
  t.A = π/3 ∧ t.a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l482_48245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drug_effect_not_significant_l482_48200

noncomputable def total_patients : ℝ := 200
noncomputable def new_drug_patients : ℝ := 100
noncomputable def no_drug_patients : ℝ := 100
noncomputable def cured_with_drug : ℝ := 60
noncomputable def not_cured_without_drug : ℝ := 50

noncomputable def k_squared (n a b c d : ℝ) : ℝ :=
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

def critical_value : ℝ := 2.706

theorem drug_effect_not_significant : 
  let a : ℝ := cured_with_drug
  let b : ℝ := new_drug_patients - cured_with_drug
  let c : ℝ := no_drug_patients - not_cured_without_drug
  let d : ℝ := not_cured_without_drug
  let n : ℝ := total_patients
  k_squared n a b c d < critical_value := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drug_effect_not_significant_l482_48200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_simplification_l482_48223

theorem sqrt_sum_simplification : Real.sqrt 27 - Real.sqrt 12 + Real.sqrt 48 = 5 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_simplification_l482_48223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_g_g_eq_4_l482_48256

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -2 then 2
  else if x ≤ 2 then x^2
  else 5

-- State the theorem
theorem two_solutions_for_g_g_eq_4 :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x, x ∈ s ↔ g (g x) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_for_g_g_eq_4_l482_48256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l482_48278

/-- The circle with center (1, 3) and radius 4 -/
def Circle : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 - 3)^2 = 16}

/-- The line x - y + t = 0 -/
def Line (t : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 - p.2 + t = 0}

/-- The chord length is 4√2 -/
noncomputable def ChordLength : ℝ := 4 * Real.sqrt 2

/-- The theorem stating the value of t -/
theorem circle_line_intersection (t : ℝ) :
  (∃ p q : ℝ × ℝ, p ∈ Circle ∧ q ∈ Circle ∧ 
   p ∈ Line t ∧ q ∈ Line t ∧ 
   ((p.1 - q.1)^2 + (p.2 - q.2)^2) = ChordLength^2) →
  t = -2 ∨ t = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l482_48278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_product_equality_l482_48209

/-- A function that checks if three numbers are pairwise coprime -/
def pairwise_coprime (x y z : ℕ) : Prop :=
  Nat.Coprime x y ∧ Nat.Coprime y z ∧ Nat.Coprime z x

/-- The main theorem -/
theorem coprime_product_equality :
  ∀ x y z t : ℕ+,
  pairwise_coprime x.val y.val z.val →
  (x + y) * (y + z) * (z + x) = x * y * z * t →
  ((x, y, z, t) = (1, 1, 1, 8) ∨ (x, y, z, t) = (1, 1, 2, 9) ∨ (x, y, z, t) = (1, 2, 3, 10)) :=
by
  sorry

#check coprime_product_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_product_equality_l482_48209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_solution_l482_48211

/-- The set of real numbers x satisfying (1/4)^(x-1) > 16 is equal to the interval (-∞, -1). -/
theorem exponential_inequality_solution :
  {x : ℝ | (1 / 4 : ℝ)^(x - 1) > 16} = Set.Iio (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_inequality_solution_l482_48211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_range_exact_l482_48210

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  (arccos (x/3))^2 + Real.pi * arcsin (x/3) - (arcsin (x/3))^2 - (Real.pi^2/12) * (x^2 - 3*x + 9)

-- State the theorem
theorem f_range : 
  ∀ y ∈ Set.range f, -3*Real.pi^2/4 ≤ y ∧ y ≤ Real.pi^2/2 :=
by sorry

-- Define the domain of f
def f_domain : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 3}

-- State that the range is exactly [-3π^2/4, π^2/2]
theorem f_range_exact : 
  Set.range f = {y : ℝ | -3*Real.pi^2/4 ≤ y ∧ y ≤ Real.pi^2/2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_range_exact_l482_48210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_five_even_digits_position_l482_48206

/-- Represents the sequence of digits formed by concatenating natural numbers. -/
def digitSequence : ℕ → ℕ := sorry

/-- Returns true if the digit at position n in the sequence is even. -/
def isEvenDigit (n : ℕ) : Prop :=
  digitSequence n % 2 = 0

/-- The position of the first digit of the first occurrence of five consecutive even digits. -/
def firstFiveEvenDigitsPosition : ℕ := 490

/-- Theorem stating that the position of the first digit of the first occurrence of five consecutive even digits is 490. -/
theorem first_five_even_digits_position :
  (∀ k < firstFiveEvenDigitsPosition, ¬(∀ i < 5, isEvenDigit (k + i))) ∧
  (∀ i < 5, isEvenDigit (firstFiveEvenDigitsPosition + i)) :=
by sorry

#check first_five_even_digits_position

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_five_even_digits_position_l482_48206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l482_48235

/-- Calculates simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculates compound interest (yearly compounding) -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem: If the difference between compound and simple interest is 150,
    then the principal is 15000 -/
theorem interest_difference_implies_principal
  (principal rate time difference : ℝ)
  (h_rate : rate = 10)
  (h_time : time = 2)
  (h_diff : compound_interest principal rate time - simple_interest principal rate time = difference)
  (h_difference : difference = 150) :
  principal = 15000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l482_48235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l482_48291

theorem tan_half_sum (a b : ℝ) 
  (h1 : Real.cos a + Real.cos b = 1/3)
  (h2 : Real.sin a + Real.sin b = 4/13) : 
  Real.tan ((a + b)/2) = 12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_sum_l482_48291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l482_48227

/-- The y-intercept of a line is the point where the line intersects the y-axis. -/
noncomputable def y_intercept (a b c : ℝ) : ℝ × ℝ := (0, c / b)

/-- The line equation is in the form ax + by = c, where a, b, and c are real numbers and b ≠ 0. -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  b_nonzero : b ≠ 0

theorem y_intercept_of_line (l : Line) 
  (h1 : l.a = 7)
  (h2 : l.b = 3)
  (h3 : l.c = 21) : 
  y_intercept l.a l.b l.c = (0, 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l482_48227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_when_a_zero_f_max_value_l482_48258

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + Real.sqrt (1 - a)| - |x - Real.sqrt a|

-- Theorem 1: When a = 0, f(x) ≥ 0 if and only if x ∈ [-1/2, +∞)
theorem f_nonnegative_when_a_zero (x : ℝ) :
  f 0 x ≥ 0 ↔ x ≥ -1/2 := by sorry

-- Theorem 2: For any a ∈ [0,1], the maximum value of f(x) is 1
theorem f_max_value (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  ∃ (x : ℝ), ∀ (y : ℝ), f a x ≥ f a y ∧ f a x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_when_a_zero_f_max_value_l482_48258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_celine_erasers_l482_48241

theorem celine_erasers (gabriel_erasers celine_erasers julian_erasers : ℕ) 
  (h1 : gabriel_erasers * 2 = celine_erasers)
  (h2 : celine_erasers * 2 = julian_erasers)
  (h3 : gabriel_erasers + celine_erasers + julian_erasers = 35) :
  celine_erasers = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_celine_erasers_l482_48241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l482_48265

/-- A sequence a : ℕ → ℝ is an arithmetic sequence if the difference between
    any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : IsArithmeticSequence a) :
  a 2 + a 8 = 15 - a 5 → a 5 = 5 := by
  intro h1
  have h2 : a 2 + a 8 = 2 * a 5 := by
    sorry -- Proof that in an arithmetic sequence, a₂ + a₈ = 2a₅
  rw [h2] at h1
  have h3 : 2 * a 5 = 15 - a 5 := h1
  linarith

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l482_48265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_four_l482_48247

/-- A rhombus with an inscribed circle and a square inscribed in that circle. -/
structure RhombusCircleSquare where
  /-- Side length of the rhombus -/
  a : ℝ
  /-- Acute angle of the rhombus in radians -/
  angle : ℝ
  /-- The acute angle is 30° (π/6 radians) -/
  angle_eq : angle = Real.pi / 6
  /-- The side length is positive -/
  a_pos : a > 0

/-- The ratio of the area of the rhombus to the area of the inscribed square -/
noncomputable def areaRatio (rcs : RhombusCircleSquare) : ℝ :=
  (rcs.a^2 * Real.sin rcs.angle) / ((rcs.a * Real.sin (rcs.angle / 2) * Real.cos (rcs.angle / 2))^2 / 2)

/-- The ratio of the areas is 4 -/
theorem area_ratio_is_four (rcs : RhombusCircleSquare) : areaRatio rcs = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_four_l482_48247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_driving_kids_weekend_time_l482_48289

/-- Calculates the time spent driving kids on each weekend day given the total weekly driving time, daily commute time, work days, and weekend days. -/
noncomputable def time_driving_kids_per_weekend_day (total_weekly_driving_time : ℝ) (daily_commute_time : ℝ) (work_days : ℝ) (weekend_days : ℝ) : ℝ :=
  (total_weekly_driving_time - daily_commute_time * work_days) / weekend_days

/-- Theorem stating that given the specified conditions, the time spent driving kids each weekend day is 2 hours. -/
theorem driving_kids_weekend_time :
  let total_weekly_driving_time : ℝ := 9
  let daily_commute_time : ℝ := 1
  let work_days : ℝ := 5
  let weekend_days : ℝ := 2
  time_driving_kids_per_weekend_day total_weekly_driving_time daily_commute_time work_days weekend_days = 2 := by
  -- Unfold the definition and simplify
  unfold time_driving_kids_per_weekend_day
  -- Perform the calculation
  simp [add_div, sub_div]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_driving_kids_weekend_time_l482_48289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_satisfy_conditions_l482_48280

theorem two_pairs_satisfy_conditions : ∃! n : Nat, 
  n = (Finset.filter (fun p : Nat × Nat => 
    let (a, b) := p
    a > 0 ∧ b > 0 ∧ 
    a + b = 667 ∧ 
    Nat.lcm a b = 120 * Nat.gcd a b) (Finset.range 668 ×ˢ Finset.range 668)).card ∧ 
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pairs_satisfy_conditions_l482_48280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_increases_by_4_7_or_9_l482_48297

def move_first_digit_to_end (n : ℕ) : ℕ :=
  let s := toString n
  if s.length > 1 then
    let rest := s.drop 1
    let first := s.take 1
    (rest ++ first).toNat!
  else n

theorem no_integer_increases_by_4_7_or_9 :
  ∀ n : ℕ, n ≠ 0 →
    (move_first_digit_to_end n ≠ 4 * n) ∧
    (move_first_digit_to_end n ≠ 7 * n) ∧
    (move_first_digit_to_end n ≠ 9 * n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_increases_by_4_7_or_9_l482_48297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maintenance_check_increase_l482_48243

/-- Calculates the percentage increase between two values -/
noncomputable def percentageIncrease (original : ℝ) (new : ℝ) : ℝ :=
  (new - original) / original * 100

theorem maintenance_check_increase :
  let originalInterval : ℝ := 30
  let newInterval : ℝ := 45
  percentageIncrease originalInterval newInterval = 50 := by
  -- Unfold the definition of percentageIncrease
  unfold percentageIncrease
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maintenance_check_increase_l482_48243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_journey_time_l482_48264

/-- Represents the journey from Vasya's home to school -/
structure Journey where
  distance : ℝ
  speed : ℝ

/-- Represents the time taken for a journey -/
noncomputable def journey_time (j : Journey) : ℝ := j.distance / j.speed

/-- Monday's journey -/
noncomputable def monday_journey (j : Journey) : ℝ := journey_time j

/-- Tuesday's journey -/
noncomputable def tuesday_journey (j : Journey) : ℝ :=
  (j.distance / 2) / j.speed + 1/12 + (j.distance / 2) / (2 * j.speed)

/-- The theorem stating that Vasya's journey time is 20 minutes -/
theorem vasya_journey_time (j : Journey) :
  monday_journey j = tuesday_journey j → journey_time j = 1/3 := by
  sorry

#check vasya_journey_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_journey_time_l482_48264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_area_is_30_figure_set_is_triangle_triangle_area_is_30_l482_48269

/-- The equation defining the figure -/
def figure_equation (x y : ℝ) : Prop :=
  |5*x| + |12*y| + |60 - 5*x - 12*y| = 60

/-- The set of points satisfying the equation -/
def figure_set : Set (ℝ × ℝ) :=
  {p | figure_equation p.1 p.2}

/-- The area of the figure -/
noncomputable def figure_area : ℝ := 30 -- We define it directly as 30

/-- Theorem stating that the area of the figure is 30 -/
theorem figure_area_is_30 : figure_area = 30 := by
  -- The proof is trivial since we defined figure_area as 30
  rfl

/-- Theorem proving that the figure_set is indeed a triangle -/
theorem figure_set_is_triangle : 
  figure_set = {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ 5*p.1 + 12*p.2 ≤ 60} := by
  sorry

/-- Theorem proving that the area of the triangle is indeed 30 -/
theorem triangle_area_is_30 : 
  MeasureTheory.volume {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ 5*p.1 + 12*p.2 ≤ 60} = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_figure_area_is_30_figure_set_is_triangle_triangle_area_is_30_l482_48269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_factorial_l482_48295

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def is_factor (a b : ℕ) : Bool := b % a = 0

def count_factors (n : ℕ) (set : List ℕ) : ℕ :=
  (set.filter (λ x => is_factor x n)).length

theorem probability_factor_of_factorial :
  let set := List.range 120
  let fac6 := factorial 6
  (count_factors fac6 set : ℚ) / set.length = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_factor_of_factorial_l482_48295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_variance_after_adding_point_l482_48213

noncomputable def variance (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  (data.map (fun x => (x - mean)^2)).sum / data.length

theorem new_variance_after_adding_point
  (original_data : List ℝ)
  (h1 : original_data.length = 7)
  (h2 : (original_data.sum / original_data.length) = 5)
  (h3 : variance original_data = 4)
  (new_point : ℝ)
  (h4 : new_point = 5) :
  variance (new_point :: original_data) = 7/2 := by
  sorry

#check new_variance_after_adding_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_variance_after_adding_point_l482_48213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_calls_for_spies_solution_for_2017_spies_l482_48207

/-- The minimum number of telephone calls needed for n spies to share all parts of a secret code -/
def min_calls (n : ℕ) : ℕ :=
  2 * n - 4

/-- The actual minimum number of calls needed (axiom) -/
axiom minimum_calls_needed : ℕ → ℕ

/-- Theorem stating the minimum number of calls for n spies to share all parts of a secret code -/
theorem min_calls_for_spies (n : ℕ) (h : n ≥ 4) :
  min_calls n = minimum_calls_needed n :=
by sorry

/-- Solution for the specific case of 2017 spies -/
theorem solution_for_2017_spies :
  min_calls 2017 = 4030 :=
by
  rfl  -- This is a reflexivity proof, as it's a direct calculation


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_calls_for_spies_solution_for_2017_spies_l482_48207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_l482_48275

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.exp x

-- Define the derivative of f(x)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x + a + 1)

-- Theorem statement
theorem perpendicular_tangents (a : ℝ) : 
  (f' a (-1)) * (f' a 1) = -1 ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_l482_48275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l482_48225

-- Define the side length of the larger square
def large_square_side (y : ℝ) : ℝ := 2 * y

-- Define the side length of the smaller square
def small_square_side (y : ℝ) : ℝ := y

-- Define the base and height of the right triangle
def triangle_base_height (y : ℝ) : ℝ := large_square_side y - small_square_side y

-- Define the hypotenuse of the right triangle
noncomputable def triangle_hypotenuse (y : ℝ) : ℝ := y * Real.sqrt 2

-- Theorem statement
theorem triangle_perimeter (y : ℝ) (h : y > 0) :
  triangle_base_height y + triangle_base_height y + triangle_hypotenuse y = 2 * y + y * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l482_48225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greg_ppo_reward_l482_48290

/-- The maximum reward for ProcGen -/
noncomputable def max_procgen_reward : ℝ := 240

/-- The maximum reward for the challenging version of CoinRun -/
noncomputable def max_coinrun_reward : ℝ := max_procgen_reward / 2

/-- The percentage of reward obtained by Greg's PPO algorithm -/
noncomputable def ppo_performance : ℝ := 0.9

/-- The reward obtained by Greg's PPO algorithm -/
noncomputable def ppo_reward : ℝ := ppo_performance * max_coinrun_reward

/-- Theorem stating that the reward obtained by Greg's PPO algorithm is 108 -/
theorem greg_ppo_reward : ppo_reward = 108 := by
  -- Unfold definitions
  unfold ppo_reward
  unfold max_coinrun_reward
  unfold max_procgen_reward
  unfold ppo_performance
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greg_ppo_reward_l482_48290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_sum_squared_bound_l482_48287

/-- Given a circle with radius r and any point C on its circumference,
    if s is the sum of the distances from C to the endpoints of a diameter,
    then s^2 ≤ 8r^2 -/
theorem circle_diameter_sum_squared_bound (r : ℝ) (C A B : EuclideanSpace ℝ (Fin 2)) (s : ℝ) 
    (h1 : r > 0)
    (h2 : dist C A + dist C B = s)
    (h3 : dist A B = 2 * r)
    (h4 : dist (C - A) (0 : EuclideanSpace ℝ (Fin 2)) = r) :
  s^2 ≤ 8 * r^2 := by
  sorry

/-- A point is on a circle with center O and radius r -/
def onCircle (P O : EuclideanSpace ℝ (Fin 2)) (r : ℝ) : Prop :=
  dist (P - O) 0 = r

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_diameter_sum_squared_bound_l482_48287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l482_48248

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 2^x + 2^(a*x + b)

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
    (f a b 1 = 5/2) ∧
    (f a b 2 = 17/4) ∧
    (a = -1) ∧
    (b = 0) ∧
    (∀ x y : ℝ, x < y ∧ y ≤ 0 → f a b x > f a b y) ∧
    (∀ x : ℝ, f a b x ≥ 2) ∧
    (∃ x : ℝ, f a b x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l482_48248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l482_48212

/-- The function f(x) = sin(ω * x + φ) -/
noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

/-- The theorem stating the maximum value of ω given the conditions -/
theorem max_omega_value (ω φ : ℝ) : 
  ω > 0 → 
  |φ| ≤ π / 2 → 
  f ω φ (-π / 4) = 0 → 
  (∀ x, f ω φ (x + π / 4) = f ω φ (-x + π / 4)) → 
  (∀ x y, π / 18 < x → x < y → y < 5 * π / 36 → (f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y)) → 
  ω ≤ 9 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l482_48212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l482_48272

-- Define the efficiency of Sakshi as the base
def sakshi_efficiency : ℚ := 1

-- Define Tanya's efficiency relative to Sakshi
def tanya_efficiency : ℚ := 5/4 * sakshi_efficiency

-- Define Mikaela's efficiency relative to Sakshi
def mikaela_efficiency : ℚ := 1/2 * sakshi_efficiency

-- Define the time it takes Sakshi to complete the work
def sakshi_time : ℚ := 20

-- Define the work rate of each person
noncomputable def work_rate (efficiency : ℚ) (time : ℚ) : ℚ := efficiency / time

-- Define the combined work rate
noncomputable def combined_work_rate : ℚ :=
  work_rate sakshi_efficiency sakshi_time +
  work_rate tanya_efficiency sakshi_time +
  work_rate mikaela_efficiency sakshi_time

-- Theorem to prove
theorem work_completion_time :
  1 / combined_work_rate = 80 / 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l482_48272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_28_l482_48260

/-- Represents the time taken to travel one mile on a given day -/
def time_per_mile (day : Nat) : Nat :=
  6 + 3 * (day - 1)

/-- Calculates the distance traveled in one hour given the time per mile -/
def distance_traveled (time_per_mile : Nat) : Nat :=
  60 / time_per_mile

/-- The total distance traveled over five days -/
def total_distance : Nat :=
  (List.range 5).map (fun day => distance_traveled (time_per_mile (day + 1))) |>.sum

theorem total_distance_is_28 :
  total_distance = 28 := by
  sorry

#eval total_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_is_28_l482_48260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y3_in_expansion_l482_48234

def binomial_expansion (a b : ℕ → ℕ) (n : ℕ) : ℕ → ℕ → ℕ :=
  λ i j => sorry  -- Placeholder implementation

theorem coefficient_x3y3_in_expansion :
  let expansion := binomial_expansion (λ i => if i = 0 then 1 else if i = 1 then 1 else if i = 2 then 1 else 0)
                                      (λ i => if i = 1 then 1 else 0)
                                      5
  expansion 3 3 = 20 := by
  sorry  -- Placeholder proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y3_in_expansion_l482_48234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_P_to_A_l482_48224

-- Define the triangle ABC
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  right_angle_at_A : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0

-- Define the intersection point P of angle bisectors
noncomputable def P (triangle : RightTriangle) : ℝ × ℝ := sorry

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem distance_from_P_to_A (triangle : RightTriangle) :
  distance (P triangle) triangle.C = Real.sqrt 80000 →
  distance (P triangle) triangle.A = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_P_to_A_l482_48224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_difference_identity_l482_48282

theorem cos_sin_difference_identity (x y : ℝ) : 
  Real.cos (x + y) * Real.sin x - Real.sin (x + y) * Real.cos x = Real.sin (x + y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_difference_identity_l482_48282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l482_48253

/-- The perimeter of an isosceles triangle with a side shared with an equilateral triangle -/
theorem isosceles_triangle_perimeter 
  (eq_triangle_perimeter : ℝ)
  (isosceles_base : ℝ)
  (isosceles_perimeter : ℝ)
  (h1 : eq_triangle_perimeter = 60)
  (h2 : isosceles_base = 25)
  (h3 : isosceles_perimeter = (eq_triangle_perimeter / 3) * 2 + isosceles_base) :
  isosceles_perimeter = 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l482_48253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_identical_lines_l482_48244

/-- Two lines in the xy-plane -/
structure TwoLines where
  b : ℝ
  c : ℝ

/-- First line equation -/
def line1 (lines : TwoLines) (x y : ℝ) : Prop :=
  3 * x + lines.b * y + lines.c = 0

/-- Second line equation -/
def line2 (lines : TwoLines) (x y : ℝ) : Prop :=
  lines.c * x - 2 * y + 12 = 0

/-- The lines are identical if they have the same slope and y-intercept -/
def are_identical (lines : TwoLines) : Prop :=
  ∃ (m k : ℝ), (∀ x y, line1 lines x y ↔ y = m * x + k) ∧
               (∀ x y, line2 lines x y ↔ y = m * x + k)

/-- There are exactly two pairs (b, c) that make the lines identical -/
theorem two_identical_lines : ∃! (s : Finset (ℝ × ℝ)), 
  (∀ p ∈ s, are_identical ⟨p.1, p.2⟩) ∧ s.card = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_identical_lines_l482_48244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_when_sin_B_cos_C_minimum_l482_48229

-- Define an obtuse triangle ABC
structure ObtuseTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  obtuse : A + B + C = π ∧ max A (max B C) > π/2

-- Define the theorem
theorem angle_B_when_sin_B_cos_C_minimum (abc : ObtuseTriangle) : 
  (Real.sin abc.A^2 + (Real.sqrt 3 / 6) * Real.sin (2 * abc.A) = 1) →
  (∀ x, Real.sin x * Real.cos (π - abc.A - x) ≥ Real.sin abc.B * Real.cos abc.C) →
  abc.B = π/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_when_sin_B_cos_C_minimum_l482_48229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_cartesian_equation_max_area_triangle_ABP_l482_48284

noncomputable section

-- Define the polar coordinate system
def polar_to_cartesian (ρ : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define points A and B in polar coordinates
def A : ℝ × ℝ := polar_to_cartesian (3 * Real.sqrt 3) (Real.pi / 2)
def B : ℝ × ℝ := polar_to_cartesian 3 (Real.pi / 3)

-- Define circle C in polar coordinates
def circle_C (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Statement 1: Standard equation of circle C in Cartesian coordinates
theorem circle_C_cartesian_equation (x y : ℝ) :
  (x, y) ∈ {p : ℝ × ℝ | ∃ θ, p = polar_to_cartesian (circle_C θ) θ} ↔
  (x - 1)^2 + y^2 = 1 := by
  sorry

-- Statement 2: Maximum area of triangle ABP
theorem max_area_triangle_ABP :
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) *
   (1 + Real.sqrt 3) / 2 : ℝ) = (3 * Real.sqrt 3 + 3) / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_cartesian_equation_max_area_triangle_ABP_l482_48284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_conclusions_l482_48283

noncomputable section

-- Define the line
def line (x y : ℝ) : Prop := 2*x - 3*y + 1 = 0

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin (2*x - Real.pi/3)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem correct_conclusions :
  -- Conclusion 3
  (∀ (a b : ℝ), (line 1 0 ∧ ¬line a b) → (3*b - 2*a > 1)) ∧
  -- Conclusion 4
  (∃ (φ : ℝ), φ > 0 ∧ 
    is_even (λ x ↦ f (x - φ)) ∧
    (∀ (ψ : ℝ), ψ > 0 ∧ is_even (λ x ↦ f (x - ψ)) → φ ≤ ψ) ∧
    φ = Real.pi/12) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_conclusions_l482_48283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_eq_neg_two_l482_48276

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ a * x + 2 * y + a + 3 = 0
def l₂ (a : ℝ) : ℝ → ℝ → Prop := λ x y ↦ x + (a + 1) * y + 4 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- Theorem statement
theorem parallel_lines_imply_a_eq_neg_two :
  ∀ a : ℝ, parallel (l₁ a) (l₂ a) → a = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_eq_neg_two_l482_48276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_intersecting_range_l482_48236

/-- Two parallel lines -/
def line1 (x y : ℝ) : Prop := x - y - 2 = 0
def line2 (x y : ℝ) : Prop := x - y + 3 = 0

/-- Circle C -/
def circleC (x y b : ℝ) : Prop := (x + 1)^2 + y^2 = b^2

/-- The relationship is "parallel and intersecting" -/
def parallel_intersecting (b : ℝ) : Prop :=
  ∃ (x y : ℝ), (line1 x y ∨ line2 x y) ∧ circleC x y b

/-- The range of b -/
def b_range (b : ℝ) : Prop :=
  (b > Real.sqrt 2 ∧ b < (3 * Real.sqrt 2) / 2) ∨
  (b > (3 * Real.sqrt 2) / 2)

theorem parallel_intersecting_range :
  ∀ b > 0, parallel_intersecting b ↔ b_range b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_intersecting_range_l482_48236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l482_48230

/-- Calculate the length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 60 →
  crossing_time = 45 →
  ∃ (bridge_length : ℝ), (bridge_length ≥ 500.14 ∧ bridge_length ≤ 500.16 ∧ bridge_length > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l482_48230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l482_48273

theorem cosine_sum_product_form : ∃ (a b c d : ℕ+),
  (∀ x : ℝ, Real.cos (2*x) + Real.cos (4*x) + Real.cos (8*x) + Real.cos (10*x) = 
    (a : ℝ) * Real.cos (b*x) * Real.cos (c*x) * Real.cos (d*x)) ∧
  a + b + c + d = 14 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_product_form_l482_48273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_saramago_readers_fraction_l482_48293

/-- The fraction of workers who have read Saramago's book -/
def s : ℚ := sorry

/-- The total number of workers -/
def total_workers : ℕ := 72

/-- The number of workers who have read both books -/
def read_both : ℕ := 4

/-- The fraction of workers who have read Kureishi's book -/
def kureishi_readers : ℚ := 5/8

theorem saramago_readers_fraction :
  (s * total_workers : ℚ) - read_both +  -- Workers who read only Saramago
  (kureishi_readers * total_workers - read_both) +  -- Workers who read only Kureishi
  read_both +  -- Workers who read both
  ((s * total_workers : ℚ) - read_both - 1) =  -- Workers who read neither
  total_workers ∧ s = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_saramago_readers_fraction_l482_48293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l482_48286

/-- An ellipse with center at the origin, major to minor axis ratio of 2:1, and one focus at (0, -2) -/
structure Ellipse where
  /-- Ratio of major to minor axis -/
  axis_ratio : ℝ
  /-- Distance from center to focus -/
  c : ℝ
  /-- Semi-major axis length -/
  a : ℝ
  /-- Semi-minor axis length -/
  b : ℝ
  /-- Axis ratio condition -/
  axis_ratio_cond : axis_ratio = 2
  /-- Focus position condition -/
  focus_cond : c = 2
  /-- Relationship between a and b -/
  ab_relation : a = 2 * b
  /-- Relationship between a, b, and c -/
  abc_relation : c^2 = a^2 - b^2

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.b^2 + y^2 / e.a^2 = 1

/-- Theorem stating the eccentricity and standard equation of the ellipse -/
theorem ellipse_properties (e : Ellipse) :
  eccentricity e = Real.sqrt 3 / 2 ∧
  ∀ x y, standard_equation e x y ↔ x^2 / (4/3) + y^2 / (16/3) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l482_48286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_l482_48201

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x) + 1 / Real.sqrt x

-- Define the domain
def domain : Set ℝ := { x | 0 < x ∧ x ≤ 1 }

-- Theorem statement
theorem function_domain : 
  ∀ x : ℝ, f x ∈ Set.range f ↔ x ∈ domain := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_l482_48201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sum_property_l482_48228

noncomputable section

variable (x₁ x₂ x₃ y₁ y₂ y₃ a b : ℝ)

/-- The average of three real numbers -/
def average (x y z : ℝ) : ℝ := (x + y + z) / 3

/-- Theorem stating the property of average of sums -/
theorem average_sum_property 
  (h1 : average x₁ x₂ x₃ = a) 
  (h2 : average y₁ y₂ y₃ = b) : 
  average (3*x₁ + y₁) (3*x₂ + y₂) (3*x₃ + y₃) = 3*a + b := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_sum_property_l482_48228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_describes_parabola_l482_48238

/-- The equation |y-3| = √((x+4)² + (y-1)²) describes a parabola -/
theorem equation_describes_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
    ∀ (x y : ℝ), (|y - 3| = Real.sqrt ((x + 4)^2 + (y - 1)^2)) ↔
      (y = a * x^2 + b * x + c) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_describes_parabola_l482_48238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_origin_l482_48261

-- Define the point P
def P : ℝ × ℝ := (3, 4)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem distance_P_to_origin : distance P O = 5 := by
  -- Unfold the definitions
  unfold distance P O
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_P_to_origin_l482_48261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l482_48255

theorem sin_cos_sum (α β p q : ℝ) 
  (h1 : Real.sin α + Real.sin β = p) 
  (h2 : Real.cos α + Real.cos β = q) 
  (h3 : p ≠ 0 ∨ q ≠ 0) : 
  Real.sin (α + β) = (2 * p * q) / (p^2 + q^2) ∧ 
  Real.cos (α + β) = (q^2 - p^2) / (q^2 + p^2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_l482_48255
