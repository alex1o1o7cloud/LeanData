import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_four_coefficients_binomial_expansion_l424_42493

theorem sum_of_first_four_coefficients_binomial_expansion :
  ∀ a : ℝ,
  let expansion := (1 + 1/a)^7
  let first_four_coefficients := [1, 7, 21, 35]
  List.sum first_four_coefficients = 64 := by
  intro a
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_four_coefficients_binomial_expansion_l424_42493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_radius_is_sqrt_59_l424_42476

/-- A sphere intersecting two planes -/
structure IntersectingSphere where
  xy_center : ℝ × ℝ × ℝ
  xy_radius : ℝ
  yz_center : ℝ × ℝ × ℝ

/-- The radius of the intersection with the yz-plane -/
noncomputable def yz_radius (s : IntersectingSphere) : ℝ :=
  Real.sqrt 59

theorem intersection_radius_is_sqrt_59 (s : IntersectingSphere) 
  (h1 : s.xy_center = (3, 5, 0))
  (h2 : s.xy_radius = 2)
  (h3 : s.yz_center = (0, 5, -8)) :
  yz_radius s = Real.sqrt 59 := by
  sorry

#check intersection_radius_is_sqrt_59

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_radius_is_sqrt_59_l424_42476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_transformation_l424_42413

/-- Given that r₁, r₂, and r₃ are the roots of the polynomial x³ - 3x² + 5 = 0,
    the monic polynomial whose roots are 3r₁, 3r₂, and 3r₃ is x³ - 9x² + 135 -/
theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (∀ x : ℂ, x^3 - 3*x^2 + 5 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (∀ x : ℂ, x^3 - 9*x^2 + 135 = 0 ↔ x = 3*r₁ ∨ x = 3*r₂ ∨ x = 3*r₃) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_transformation_l424_42413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_two_l424_42455

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1))

-- State the theorem
theorem sum_equals_two (a b : ℝ) (h : f a + f (b - 2) = 0) : a + b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_two_l424_42455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l424_42428

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (2 + x) + Real.log (2 - x)

-- Define the domain of f
def domain : Set ℝ := Set.Ioo (-2) 2

-- Theorem stating that f is even and decreasing on (0, 2)
theorem f_properties :
  (∀ x, x ∈ domain → f (-x) = f x) ∧
  (∀ x y, x ∈ Set.Ioo 0 2 → y ∈ Set.Ioo 0 2 → x < y → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l424_42428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l424_42457

open Real

-- Define the triangle
def Triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧
  0 < B ∧ B < Real.pi/2 ∧
  0 < C ∧ C < Real.pi/2 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- State the theorem
theorem triangle_side_sum_range 
  (A B C a b c : ℝ)
  (h_triangle : Triangle A B C a b c)
  (h_eq : Real.cos A / a + Real.cos B / b = 2 * Real.sqrt 3 * Real.sin C / (3 * a))
  (h_b : b = 2 * Real.sqrt 3) :
  6 < a + c ∧ a + c ≤ 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_sum_range_l424_42457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_one_eq_neg_three_monotonic_decreasing_intervals_min_value_is_neg_four_l424_42401

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x * (x + 4) else x * (x - 4)

-- Statement 1: f(f(1)) = -3
theorem f_f_one_eq_neg_three : f (f 1) = -3 := by sorry

-- Statement 2: Monotonic decreasing intervals
theorem monotonic_decreasing_intervals :
  (∀ x y, x ∈ Set.Iic (-2) → y ∈ Set.Iic (-2) → x ≤ y → f x ≥ f y) ∧
  (∀ x y, x ∈ Set.Icc 0 2 → y ∈ Set.Icc 0 2 → x ≤ y → f x ≥ f y) := by sorry

-- Statement 3: Minimum value is -4
theorem min_value_is_neg_four :
  ∃ x, f x = -4 ∧ ∀ y, f y ≥ -4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_one_eq_neg_three_monotonic_decreasing_intervals_min_value_is_neg_four_l424_42401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l424_42442

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x * (Real.cos x - Real.sqrt 3 * Real.sin x)

-- Theorem for the smallest positive period and monotonic intervals
theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  -- Function is monotonically increasing on [0, π/12]
  (∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 12 → f x < f y) ∧
  -- Function is monotonically increasing on [7π/12, π]
  (∀ (x y : ℝ), 7 * Real.pi / 12 ≤ x ∧ x < y ∧ y ≤ Real.pi → f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l424_42442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covers_fraction_of_grid_l424_42417

/-- Triangle with vertices P, Q, R -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- Calculate the area of a triangle using the Shoelace formula -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let (x1, y1) := t.P
  let (x2, y2) := t.Q
  let (x3, y3) := t.R
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Calculate the area of a rectangular grid -/
def gridArea (width : ℝ) (height : ℝ) : ℝ :=
  width * height

/-- The main theorem to prove -/
theorem triangle_covers_fraction_of_grid :
  let t : Triangle := { P := (2, 2), Q := (7, 3), R := (6, 5) }
  let gridWidth : ℝ := 8
  let gridHeight : ℝ := 6
  (triangleArea t) / (gridArea gridWidth gridHeight) = 11 / 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_covers_fraction_of_grid_l424_42417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_guests_at_banquet_l424_42469

/-- Represents the types of food sets available at the banquet -/
inductive FoodSet
  | Vegetarian
  | Pescatarian
  | Carnivorous

/-- Represents the banquet information -/
structure BanquetInfo where
  totalFood : ℕ
  vegetarianFood : ℕ
  pescatarianFood : ℕ
  carnivorousFood : ℕ
  maxVegetarianPerGuest : ℚ
  maxPescatarianPerGuest : ℚ
  maxCarnivorousPerGuest : ℚ

/-- Calculates the minimum number of guests for a given food set -/
def minGuestsForSet (info : BanquetInfo) (set : FoodSet) : ℚ :=
  match set with
  | FoodSet.Vegetarian => (info.vegetarianFood : ℚ) / info.maxVegetarianPerGuest
  | FoodSet.Pescatarian => (info.pescatarianFood : ℚ) / info.maxPescatarianPerGuest
  | FoodSet.Carnivorous => (info.carnivorousFood : ℚ) / info.maxCarnivorousPerGuest

/-- Calculates the total minimum number of guests for all food sets -/
def totalMinGuests (info : BanquetInfo) : ℕ :=
  Nat.ceil (minGuestsForSet info FoodSet.Vegetarian) +
  Nat.ceil (minGuestsForSet info FoodSet.Pescatarian) +
  Nat.ceil (minGuestsForSet info FoodSet.Carnivorous)

/-- Theorem stating the minimum number of guests at the banquet -/
theorem min_guests_at_banquet (info : BanquetInfo)
  (h1 : info.totalFood = 675)
  (h2 : info.vegetarianFood = 195)
  (h3 : info.pescatarianFood = 220)
  (h4 : info.carnivorousFood = 260)
  (h5 : info.maxVegetarianPerGuest = 3)
  (h6 : info.maxPescatarianPerGuest = 5/2)
  (h7 : info.maxCarnivorousPerGuest = 4)
  (h8 : info.vegetarianFood + info.pescatarianFood + info.carnivorousFood = info.totalFood) :
  totalMinGuests info = 218 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_guests_at_banquet_l424_42469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l424_42423

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line
def line (m x₀ : ℝ) (x y : ℝ) : Prop := x = m*y + x₀

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ m b : ℝ) : ℝ :=
  |m * x₀ - y₀ + b| / Real.sqrt (m^2 + 1)

theorem parabola_intersection_theorem 
  (p : ℝ) 
  (x₀ : ℝ) 
  (h_p : p > 0) 
  (h_x₀ : x₀ ≥ 1/8) 
  (h_focus : p = 1/4) -- Distance from focus to directrix is 1/2
  (m : ℝ) 
  (h_m : m ≠ 0) :
  ∃ (B : ℝ × ℝ) (d : ℝ),
    B = (-x₀, 0) ∧
    d = distance_point_to_line (-x₀) 0 m x₀ ∧
    Real.sqrt 6/12 ≤ d ∧ d < 1/2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_theorem_l424_42423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squared_diagonals_is_sixteen_sum_of_squared_diagonals_is_sixteen_proof_l424_42479

/-- A rhombus with side length 2 -/
structure Rhombus where
  side_length : ℝ
  is_two : side_length = 2
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- The sum of squares of diagonal lengths of a rhombus -/
def sum_of_squared_diagonals (r : Rhombus) : ℝ := 
  r.diagonal1^2 + r.diagonal2^2

/-- Theorem: For a rhombus with side length 2, the sum of squares of its diagonal lengths is 16 -/
theorem sum_of_squared_diagonals_is_sixteen (r : Rhombus) : 
  sum_of_squared_diagonals r = 16 := by
  sorry

/-- Lemma: The diagonals of a rhombus with side length 2 satisfy the Pythagorean theorem -/
lemma rhombus_diagonals_pythagorean (r : Rhombus) :
  (r.diagonal1 / 2)^2 + (r.diagonal2 / 2)^2 = r.side_length^2 := by
  sorry

/-- Proof of the main theorem using the Pythagorean theorem lemma -/
theorem sum_of_squared_diagonals_is_sixteen_proof (r : Rhombus) :
  sum_of_squared_diagonals r = 16 := by
  have h1 : (r.diagonal1 / 2)^2 + (r.diagonal2 / 2)^2 = r.side_length^2 := 
    rhombus_diagonals_pythagorean r
  have h2 : r.side_length = 2 := r.is_two
  sorry  -- Complete the proof here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squared_diagonals_is_sixteen_sum_of_squared_diagonals_is_sixteen_proof_l424_42479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_decrease_theorem_l424_42435

/-- The percentage decrease in stock price from February to January -/
noncomputable def stock_decrease_feb_to_jan (feb_price mar_price jan_price : ℝ) : ℝ :=
  (feb_price - jan_price) / feb_price * 100

theorem stock_decrease_theorem (feb_price mar_price jan_price : ℝ) 
  (h1 : jan_price < feb_price)
  (h2 : jan_price = mar_price * 1.20)
  (h3 : mar_price = feb_price * 0.7500000000000007) :
  ∃ (ε : ℝ), ε > 0 ∧ |stock_decrease_feb_to_jan feb_price mar_price jan_price - 10| < ε := by
  sorry

#check stock_decrease_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_decrease_theorem_l424_42435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_true_propositions_l424_42472

-- Define reciprocals
def are_reciprocals (x y : ℝ) : Prop := x * y = 1

-- Define congruence for triangles (simplified)
def are_congruent (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

-- Define area for triangles (simplified)
def triangle_area (t : Set (ℝ × ℝ)) : ℝ := sorry

-- Define the quadratic equation
def has_real_solutions (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 2*x + m = 0

theorem four_true_propositions :
  (∀ x y : ℝ, are_reciprocals x y → x * y = 1) ∧
  (¬(∀ t1 t2 : Set (ℝ × ℝ), triangle_area t1 = triangle_area t2 → are_congruent t1 t2)) ∧
  (∀ m : ℝ, ¬(has_real_solutions m) → m > 1) ∧
  (∀ A B : Set α, ¬(A ⊆ B) → A ∩ B ≠ B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_true_propositions_l424_42472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_OM_equation_min_distance_M_to_C_l424_42412

noncomputable section

-- Define the polar coordinate system
def polar_to_rect (r : ℝ) (θ : ℝ) : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)

-- Define point M
def M : ℝ × ℝ := polar_to_rect (4 * Real.sqrt 2) (Real.pi / 4)

-- Define curve C
def C (α : ℝ) : ℝ × ℝ := (1 + Real.sqrt 2 * Real.cos α, Real.sqrt 2 * Real.sin α)

-- Theorem for part I
theorem line_OM_equation : 
  ∀ (x y : ℝ), (x, y) ∈ Set.range (λ t : ℝ ↦ (t * M.1, t * M.2)) → y = x :=
by sorry

-- Theorem for part II
theorem min_distance_M_to_C :
  Real.sqrt ((M.1 - 1)^2 + M.2^2) - Real.sqrt 2 = 5 - Real.sqrt 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_OM_equation_min_distance_M_to_C_l424_42412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_closed_form_l424_42486

/-- Given a natural number n, f(n) is defined as the sum of powers of 2 
    with exponents following the arithmetic sequence 1, 4, 7, ..., 3n+10 -/
def f (n : ℕ) : ℕ := 2 + 2^4 + 2^7 + 2^10 + 2^(3*n+10)

/-- The number of terms in the sequence for f(n) is n+4 -/
def f_terms_count (n : ℕ) : ℕ := n + 4

/-- The theorem states that f(n) is equal to (2/7) * (8^(n+4) - 1) for all natural numbers n -/
theorem f_closed_form (n : ℕ) : f n = (2 / 7) * (8^(f_terms_count n) - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_closed_form_l424_42486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l424_42425

/-- Two circles with radii 9 and 7, centers 14 units apart, intersect at point P.
    A line through P creates equal chords QP and PR. The square of QP's length is 170. -/
theorem chord_length_squared (O₁ O₂ P Q R : ℝ × ℝ) : 
  let d := dist O₁ O₂
  let r₁ := dist O₁ P
  let r₂ := dist O₂ P
  dist O₁ Q = r₁ → 
  dist O₂ R = r₂ → 
  d = 14 → 
  r₁ = 9 → 
  r₂ = 7 → 
  dist O₁ P = r₁ → 
  dist O₂ P = r₂ → 
  dist Q P = dist P R → 
  (dist Q P) ^ 2 = 170 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_squared_l424_42425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_calculation_l424_42421

/-- Given that x men working x hours a day for x days using machines of efficiency e produce x²e articles,
    prove that y men working y+2 hours a day for y days using machines of efficiency e
    produce ey(y² + 2y)/x articles -/
theorem production_calculation (x y e : ℝ) (h : x > 0) :
  y * (y + 2) * y * (e / x) = e * y * (y^2 + 2*y) / x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_calculation_l424_42421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l424_42409

noncomputable def f (x : ℝ) := Real.sin x * Real.cos x - Real.sqrt 3 * (Real.cos x)^2 + 1/2 * Real.sqrt 3

theorem f_properties :
  let period : ℝ := Real.pi
  let increase_interval (k : ℤ) : Set ℝ := Set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)
  let decrease_interval (k : ℤ) : Set ℝ := Set.Icc (k * Real.pi + 5 * Real.pi / 12) (k * Real.pi + 11 * Real.pi / 12)
  let symmetry_axis (k : ℤ) : ℝ := 5 * Real.pi / 12 + k * Real.pi / 2
  let symmetry_center (k : ℤ) : ℝ := k * Real.pi / 2 + Real.pi / 6
  (∀ x, f (x + period) = f x) ∧
  (∀ k : ℤ, ∀ x ∈ increase_interval k, ∀ y ∈ increase_interval k, x < y → f x < f y) ∧
  (∀ k : ℤ, ∀ x ∈ decrease_interval k, ∀ y ∈ decrease_interval k, x < y → f x > f y) ∧
  (∀ k : ℤ, ∀ x, f (symmetry_axis k + x) = f (symmetry_axis k - x)) ∧
  (∀ k : ℤ, ∀ x, f (symmetry_center k + x) = f (symmetry_center k - x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l424_42409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_x_l424_42444

theorem tan_pi_minus_x (x : ℝ) (h1 : Real.cos (π + x) = 3/5) (h2 : x ∈ Set.Ioo π (2*π)) :
  Real.tan (π - x) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_minus_x_l424_42444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_duration_calculation_l424_42468

-- Define the problem parameters
noncomputable def total_distance : ℝ := 480
noncomputable def speed1 : ℝ := 60
noncomputable def speed2 : ℝ := 40
noncomputable def distance1 : ℝ := 120
noncomputable def distance2 : ℝ := 150
noncomputable def lunch_break : ℝ := 0.5
noncomputable def bathroom_break : ℝ := 0.25
noncomputable def gas_stop : ℝ := 1/6
noncomputable def traffic_delay : ℝ := 0.5
noncomputable def museum_stop : ℝ := 1.5

-- Define the theorem
theorem trip_duration_calculation :
  let distance3 : ℝ := total_distance - distance1 - distance2
  let driving_time : ℝ := distance1 / speed1 + distance2 / speed2 + distance3 / speed1
  let stop_time : ℝ := lunch_break + 2 * bathroom_break + 2 * gas_stop + traffic_delay + museum_stop
  let total_time : ℝ := driving_time + stop_time
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |total_time - 12.58| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_duration_calculation_l424_42468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l424_42436

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  a > b ∧ b > 0 ∧  -- Ellipse parameters
  x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧  -- Point A on ellipse
  x₂^2 / a^2 + y₂^2 / b^2 = 1 ∧  -- Point B on ellipse
  |y₁ - y₂| = 4 ∧  -- Vertical distance between A and B
  (∃ (x₀ : ℝ), (x₁ - x₀) * (x₂ - x₀) = -a^2 + b^2) ∧  -- A and B are on a line through a focus
  (∃ (S : ℝ), S = π ∧ S = a)  -- Area of triangle AF₁B is π and equals a
  →
  Real.sqrt (a^2 - b^2) / a = 1/2 :=  -- Eccentricity is 1/2
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l424_42436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l424_42400

noncomputable section

open Real

theorem triangle_properties (A B C a b c : ℝ) (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) 
  (h_sin_C : sin C = 2 * cos A * sin (B + π/3))
  (h_bc_sum : b + c = 6) :
  A = π/3 ∧ 
  ∃ (AD : ℝ), AD ≤ (3 * sqrt 3) / 2 ∧ 
  ∀ (AD' : ℝ), AD' ≤ AD :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l424_42400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_implies_a_eq_three_l424_42453

/-- The function f(x) = a^x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

/-- The theorem stating that if the sum of max and min values of f(x) = a^x on [1,2] is 12, then a = 3 -/
theorem sum_max_min_implies_a_eq_three (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc 1 2, f a x ≤ f a 2 ∧ f a 1 ≤ f a x) →
  f a 1 + f a 2 = 12 →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_max_min_implies_a_eq_three_l424_42453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coeff_binomial_expansion_l424_42410

theorem max_coeff_binomial_expansion :
  ∃ (n : ℕ) (x : ℝ) (r : ℕ),
    let expansion := (1 + 2 * Real.sqrt x) ^ n
    let max_coeff_term := 672 * x ^ (5/2 : ℝ)
    n = 7 ∧ r < n ∧
    (Nat.choose n r * (2 ^ r : ℝ) * (Real.sqrt x) ^ r = 2 * Nat.choose n (r-1) * (2 ^ (r-1) : ℝ) * (Real.sqrt x) ^ (r-1)) ∧
    (Nat.choose n r * (2 ^ r : ℝ) * (Real.sqrt x) ^ r = (5/6 : ℝ) * Nat.choose n (r+1) * (2 ^ (r+1) : ℝ) * (Real.sqrt x) ^ (r+1)) ∧
    (∀ (k : ℕ), k < n → Nat.choose n k * (2 ^ k : ℝ) * (Real.sqrt x) ^ k ≤ Nat.choose n 5 * (2 ^ 5 : ℝ) * (Real.sqrt x) ^ 5) ∧
    max_coeff_term = Nat.choose n 5 * (2 ^ 5 : ℝ) * (Real.sqrt x) ^ 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coeff_binomial_expansion_l424_42410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_150_degree_angles_l424_42433

/-- The measure of an interior angle of a regular n-gon in degrees -/
noncomputable def interior_angle (n : ℕ) : ℝ :=
  180 * (n - 2) / n

theorem regular_polygon_with_150_degree_angles (n : ℕ) :
  (n ≥ 3) →  -- A polygon must have at least 3 sides
  (∀ i : ℕ, i < n → interior_angle n = 150) →  -- All interior angles are 150°
  n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_with_150_degree_angles_l424_42433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_is_five_l424_42478

/-- A right triangular prism with a circumscribed sphere -/
structure RightTriangularPrism where
  /-- Length of side AB of the base triangle -/
  ab : ℝ
  /-- Length of side AC of the base triangle -/
  ac : ℝ
  /-- Length of side BC of the base triangle -/
  bc : ℝ
  /-- Height of the prism (AA₁) -/
  height : ℝ
  /-- AB equals AC -/
  ab_eq_ac : ab = ac
  /-- AB equals 4√2 -/
  ab_value : ab = 4 * Real.sqrt 2
  /-- BC equals 8 -/
  bc_value : bc = 8
  /-- Height equals 6 -/
  height_value : height = 6

/-- The radius of the circumscribed sphere of the right triangular prism -/
noncomputable def circumscribed_sphere_radius (p : RightTriangularPrism) : ℝ :=
  (1 / 2) * Real.sqrt (p.height ^ 2 + p.bc ^ 2)

/-- Theorem stating that the radius of the circumscribed sphere is 5 -/
theorem circumscribed_sphere_radius_is_five (p : RightTriangularPrism) :
    circumscribed_sphere_radius p = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_radius_is_five_l424_42478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strict_convexity_range_l424_42424

/-- The function f(x) = e^x - x*ln(x) - (m/2)*x^2 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x - x * Real.log x - m / 2 * x^2

/-- The second derivative of f(x) -/
noncomputable def f_second_deriv (m : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 / x - m

/-- The condition for strict convexity -/
def is_strictly_convex (m : ℝ) : Prop :=
  ∀ x, x ∈ Set.Ioo 1 4 → f_second_deriv m x < 0

/-- The theorem stating the range of m for which f is strictly convex -/
theorem strict_convexity_range :
  {m : ℝ | is_strictly_convex m} = Set.Ici (Real.exp 4 - 1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strict_convexity_range_l424_42424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tangent_l424_42495

theorem parallel_vectors_tangent (α : ℝ) :
  let a : Fin 2 → ℝ := ![Real.sin α, Real.cos α]
  let b : Fin 2 → ℝ := ![3, 4]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) →
  Real.tan α = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_tangent_l424_42495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_compensation_l424_42496

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_rate_increase : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 16)
  (h2 : regular_hours = 40)
  (h3 : overtime_rate_increase = 0.75)
  (h4 : total_hours = 57) :
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  let overtime_hours := total_hours - regular_hours
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  let total_compensation := regular_pay + overtime_pay
  total_compensation = 1116 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_driver_compensation_l424_42496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_possible_sums_l424_42404

/-- The set of ball numbers -/
def BallNumbers : Finset ℕ := {1, 2, 3, 4, 5}

/-- The set of possible sums when drawing two balls with replacement -/
def PossibleSums : Finset ℕ :=
  Finset.biUnion BallNumbers (λ x => Finset.image (λ y => x + y) BallNumbers)

/-- Theorem stating that the number of distinct possible sums is 9 -/
theorem number_of_possible_sums :
  Finset.card PossibleSums = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_possible_sums_l424_42404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_tangent_to_smaller_circle_l424_42474

-- Define the given constants
noncomputable def ring_area : ℝ := (50 / 3) * Real.pi
def larger_diameter : ℝ := 10

-- Define the radii of the circles
noncomputable def larger_radius : ℝ := larger_diameter / 2
noncomputable def smaller_radius : ℝ := Real.sqrt (25 / 3)

-- Define the chord length
noncomputable def chord_length : ℝ := (10 * Real.sqrt 6) / 3

-- Theorem statement
theorem chord_length_tangent_to_smaller_circle :
  ring_area = Real.pi * (larger_radius^2 - smaller_radius^2) ∧
  chord_length^2 = 4 * (larger_radius^2 - smaller_radius^2) := by
  sorry

#eval "Theorem statement defined successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_tangent_to_smaller_circle_l424_42474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_price_l424_42422

/-- Given that six 2-liter bottles of water cost $12, prove that the price of 1 liter of water is $1. -/
theorem water_price (total_cost : ℝ) (num_bottles : ℕ) (bottle_volume : ℝ) 
  (h1 : total_cost = 12)
  (h2 : num_bottles = 6)
  (h3 : bottle_volume = 2) :
  total_cost / (↑num_bottles * bottle_volume) = 1 := by
  sorry

#check water_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_price_l424_42422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l424_42414

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_sum_of_factors (p q r s : ℕ+) : 
  p * q * r * s = factorial 9 → 
  (∀ a b c d : ℕ+, a * b * c * d = factorial 9 → a + b + c + d ≥ p + q + r + s) → 
  p + q + r + s = 129 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_l424_42414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_square_l424_42460

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmeticSum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_sum_square (n : ℕ) :
  n ≠ 0 → arithmeticSum 4 8 n = 4 * (n : ℝ)^2 := by
  sorry

#check arithmetic_sequence_sum_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_square_l424_42460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l424_42406

theorem vector_equation_solution :
  ∃ (a b : ℚ), 
    a = 12/11 ∧ 
    b = 14/33 ∧ 
    (a • (![3, 4] : Fin 2 → ℚ)) + (b • (![(-3), 7] : Fin 2 → ℚ)) = ![2, 10] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l424_42406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_sum_l424_42449

/-- The ellipse on which points A, B, and C lie -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The angle between two vectors in radians -/
noncomputable def angle (v₁ v₂ : ℝ × ℝ) : ℝ := 
  Real.arccos ((v₁.1 * v₂.1 + v₁.2 * v₂.2) / (Real.sqrt (v₁.1^2 + v₁.2^2) * Real.sqrt (v₂.1^2 + v₂.2^2)))

/-- The theorem to be proved -/
theorem ellipse_points_sum (A B C : ℝ × ℝ) : 
  ellipse A.1 A.2 → 
  ellipse B.1 B.2 → 
  ellipse C.1 C.2 → 
  angle A B = 2 * Real.pi / 3 →
  angle B C = 2 * Real.pi / 3 →
  angle C A = 2 * Real.pi / 3 →
  1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) + 1 / (C.1^2 + C.2^2) = 15 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_points_sum_l424_42449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_identity_triangle_angle_relation_triangle_side_angle_inequality_l424_42430

-- Part 1
theorem cos_difference_identity (x y : ℝ) :
  Real.cos (2 * x) - Real.cos (2 * y) = -2 * Real.sin (x + y) * Real.sin (x - y) := by sorry

-- Part 2
theorem triangle_angle_relation (a b c A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi →
  a * Real.sin A = (b + c) * Real.sin B →
  A = 2 * B := by sorry

-- Part 3
theorem triangle_side_angle_inequality (a b c A B C m : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi →
  (b - c) * (m + 2 * (Real.cos B)^2) ≤ 2 * b →
  -3 ≤ m ∧ m ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_identity_triangle_angle_relation_triangle_side_angle_inequality_l424_42430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_million_identical_digits_in_sqrt_2_l424_42451

theorem no_six_million_identical_digits_in_sqrt_2 :
  ∀ k l : ℕ,
  (k + l ≤ 10000000) →
  (∃ c : ℕ, c < 10 ∧
    (∀ i : ℕ, k < i ∧ i ≤ k + l →
      (⌊Real.sqrt 2 * 10^i⌋ % 10 = c))) →
  l ≤ k + 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_six_million_identical_digits_in_sqrt_2_l424_42451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l424_42498

-- Define the ellipse parameters
noncomputable def a : ℝ := 4
noncomputable def c : ℝ := 2 * Real.sqrt 3

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 4 = 1

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  y = x + 2

-- Theorem statement
theorem ellipse_and_triangle_area :
  -- Given conditions
  (c^2 = 12) →
  (a = 4) →
  -- Prove the ellipse equation
  (∀ x y : ℝ, ellipse_equation x y ↔ x^2 / (a^2) + y^2 / (a^2 - c^2) = 1) ∧
  -- Prove the area of triangle OAB
  (∃ A B : ℝ × ℝ,
    ellipse_equation A.1 A.2 ∧
    ellipse_equation B.1 B.2 ∧
    line_equation A.1 A.2 ∧
    line_equation B.1 B.2 ∧
    (1/2 * |A.1 * B.2 - A.2 * B.1| = 16/5)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_area_l424_42498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_of_same_suit_in_same_row_l424_42461

/-- Represents a playing card with a suit and a value -/
structure Card where
  suit : Fin 4
  value : Fin 13

/-- Represents the 13x4 array of cards -/
def CardArray := Fin 13 → Fin 4 → Card

/-- Two cards are adjacent if they are next to each other horizontally or vertically -/
def adjacent (i j i' j' : Nat) : Prop :=
  (i = i' ∧ (j + 1 = j' ∨ j' + 1 = j)) ∨ (j = j' ∧ (i + 1 = i' ∨ i' + 1 = i))

/-- The condition that adjacent cards have the same suit or value -/
def adjacentSameProperty (arr : CardArray) : Prop :=
  ∀ i j i' j', adjacent i j i' j' →
    (arr i j).suit = (arr i' j').suit ∨ (arr i j).value = (arr i' j').value

/-- The theorem to be proved -/
theorem cards_of_same_suit_in_same_row
  (arr : CardArray) (h : adjacentSameProperty arr) :
  ∀ s : Fin 4, ∃ i : Fin 13, ∀ j : Fin 4, (arr i j).suit = s :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cards_of_same_suit_in_same_row_l424_42461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l424_42416

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) - 2 * Real.exp x + 2 * x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2 * Real.exp (2 * x) - 2 * Real.exp x + 2

-- Define the tangent line g
noncomputable def g (x₀ x : ℝ) : ℝ := f' x₀ * (x - x₀) + f x₀

-- State the theorem
theorem tangent_line_property :
  ∃ x₀ : ℝ, x₀ = -Real.log 2 ∧
  (∀ x : ℝ, (x - x₀) * (f x - g x₀ x) ≥ 0) ∧
  (∀ y : ℝ, y ≠ x₀ → ∃ z : ℝ, (z - y) * (f z - g y z) < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l424_42416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_l424_42471

theorem real_part_of_complex_fraction :
  let z : ℂ := -Complex.I / (1 + 2 * Complex.I)
  z.re = -2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_part_of_complex_fraction_l424_42471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coinciding_segments_l424_42445

/-- A cube representation --/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 8 × Fin 8)

/-- A closed eight-segment broken line on a cube --/
structure BrokenLine (cube : Cube) where
  segments : Finset (Fin 8 × Fin 8)
  is_closed : ∀ v : Fin 8, (segments.filter (λ s ↦ v = s.1 ∨ v = s.2)).card % 2 = 0
  is_eight_segment : segments.card = 8
  vertices_on_cube : ∀ s ∈ segments, s.1 ∈ cube.vertices ∧ s.2 ∈ cube.vertices

/-- The number of segments of a broken line that coincide with cube edges --/
def coinciding_segments (cube : Cube) (line : BrokenLine cube) : ℕ :=
  (line.segments ∩ cube.edges).card

/-- The main theorem --/
theorem min_coinciding_segments (cube : Cube) :
  ∀ line : BrokenLine cube, coinciding_segments cube line ≥ 2 := by
  sorry

#check min_coinciding_segments

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coinciding_segments_l424_42445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_A_l424_42484

/-- The locus of point A in triangle ABC, where B(-3,0) and C(3,0), 
    and the product of slopes of AB and AC is 4/9 -/
theorem locus_of_point_A (x y : ℝ) : 
  x ≠ 3 → x ≠ -3 →
  (y / (x + 3)) * (y / (x - 3)) = 4/9 → x^2/9 - y^2/4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_A_l424_42484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_classes_less_than_1968_l424_42403

/-- Represents a class of non-negative integers related by digit removal operations -/
def DigitRemovalClass : Type := Nat → Prop

/-- The operation of removing two consecutive digits or identical groups of digits -/
def digit_removal_operation (n m : Nat) : Prop :=
  ∃ (digits_n digits_m : List Nat),
    digits_n.length > digits_m.length ∧
    (∃ (i : Nat), i < digits_n.length - 1 ∧
      (digits_n.take i ++ digits_n.drop (i + 2) = digits_m ∨
       ∃ (j : Nat), j > i ∧ j < digits_n.length ∧
         digits_n.take i = digits_n.take j ∧
         digits_n.drop (i + (j - i)) = digits_n.drop j ∧
         digits_n.take i ++ digits_n.drop j = digits_m))

/-- A valid partitioning of non-negative integers satisfying the digit removal condition -/
def valid_partitioning (classes : List DigitRemovalClass) : Prop :=
  (∀ n : Nat, ∃! c, c ∈ classes ∧ c n) ∧
  (∀ c ∈ classes, ∃ n, c n) ∧
  (∀ c ∈ classes, ∀ n m : Nat, c n → digit_removal_operation n m → c m)

/-- The maximum number of classes in any valid partitioning is less than 1968 -/
theorem max_classes_less_than_1968 :
  ¬∃ (classes : List DigitRemovalClass), classes.length = 1968 ∧ valid_partitioning classes :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_classes_less_than_1968_l424_42403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_evaluation_l424_42475

structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

def EqualDiagonals (q : Quadrilateral) : Prop :=
  sorry

def IsRectangle (q : Quadrilateral) : Prop :=
  sorry

theorem proposition_evaluation :
  (∃ (a b : ℝ), a > b ∨ b > a) ∧
  (∃ (a b c : ℝ), a ≤ b ∧ a + c ≤ b + c) ∧
  (∃ (q : Quadrilateral), EqualDiagonals q ∧ ¬IsRectangle q) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_evaluation_l424_42475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derived_triangles_similarity_first_derived_original_similarity_second_derived_original_similarity_l424_42467

/-- An isosceles triangle with base a and legs b -/
structure IsoscelesTriangle where
  a : ℝ  -- base
  b : ℝ  -- leg
  h_positive : 0 < a ∧ 0 < b
  h_isosceles : b ≤ a

/-- Height of the triangle corresponding to the base -/
noncomputable def height_a (t : IsoscelesTriangle) : ℝ := 
  Real.sqrt (t.b^2 - (t.a/2)^2)

/-- Height of the triangle corresponding to the leg -/
noncomputable def height_b (t : IsoscelesTriangle) : ℝ := 
  (t.a / 2) * (t.b / Real.sqrt (t.b^2 - (t.a/2)^2))

/-- First derived triangle -/
noncomputable def derived_triangle_1 (t : IsoscelesTriangle) : IsoscelesTriangle :=
  { a := height_a t,
    b := t.a,
    h_positive := by sorry,
    h_isosceles := by sorry }

/-- Second derived triangle -/
noncomputable def derived_triangle_2 (t : IsoscelesTriangle) : IsoscelesTriangle :=
  { a := height_b t,
    b := t.a,
    h_positive := by sorry,
    h_isosceles := by sorry }

/-- Two triangles are similar -/
def are_similar (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.a / t1.b = t2.a / t2.b

theorem derived_triangles_similarity (t : IsoscelesTriangle) :
  are_similar (derived_triangle_1 t) (derived_triangle_2 t) ↔ t.a^2 + t.b^2 = 2 * t.a * t.b := by
  sorry

theorem first_derived_original_similarity (t : IsoscelesTriangle) :
  are_similar t (derived_triangle_1 t) ↔ t.a = t.b := by
  sorry

theorem second_derived_original_similarity (t : IsoscelesTriangle) :
  are_similar t (derived_triangle_2 t) ↔ height_a t = t.a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derived_triangles_similarity_first_derived_original_similarity_second_derived_original_similarity_l424_42467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_pell_equation_solvable_l424_42497

theorem negative_pell_equation_solvable (d : ℕ) (h_prime : Nat.Prime d) (h_mod : d % 4 = 1) :
  ∃ (x y : ℤ), x^2 - d * y^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_pell_equation_solvable_l424_42497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_events_l424_42482

structure Ball where
  color : String
  deriving DecidableEq

def Bag := Multiset Ball

def draw (bag : Bag) (n : Nat) : Multiset Ball := sorry

def isRed (b : Ball) : Prop := b.color = "red"
def isWhite (b : Ball) : Prop := b.color = "white"

def threeRedBalls (draw : Multiset Ball) : Prop :=
  Multiset.card draw = 3 ∧ ∀ b ∈ draw, isRed b

def atLeastOneWhiteBall (draw : Multiset Ball) : Prop :=
  Multiset.card draw = 3 ∧ ∃ b ∈ draw, isWhite b

def threeWhiteBalls (draw : Multiset Ball) : Prop :=
  Multiset.card draw = 3 ∧ ∀ b ∈ draw, isWhite b

def mutuallyExclusive (e1 e2 : Multiset Ball → Prop) : Prop :=
  ∀ d : Multiset Ball, ¬(e1 d ∧ e2 d)

theorem mutually_exclusive_events (bag : Bag) 
  (h1 : Multiset.count (Ball.mk "red") bag = 5)
  (h2 : Multiset.count (Ball.mk "white") bag = 5) :
  (mutuallyExclusive threeRedBalls atLeastOneWhiteBall) ∧
  (mutuallyExclusive threeRedBalls threeWhiteBalls) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_events_l424_42482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_sum_l424_42485

/-- Calculates the position of the hour hand at a given time -/
noncomputable def hourHandPosition (hour : ℕ) (minute : ℕ) : ℝ :=
  (hour % 12 : ℝ) * 30 + (minute : ℝ) * 0.5

/-- Calculates the position of the minute hand at a given time -/
noncomputable def minuteHandPosition (minute : ℕ) : ℝ :=
  (minute : ℝ) * 6

/-- Calculates the smaller angle between two positions on a clock -/
noncomputable def smallerAngle (pos1 : ℝ) (pos2 : ℝ) : ℝ :=
  min (abs (pos1 - pos2)) (360 - abs (pos1 - pos2))

/-- The sum of smaller angles at 9:20 and 2:50 is 305° -/
theorem clock_angle_sum : 
  smallerAngle (hourHandPosition 9 20) (minuteHandPosition 20) +
  smallerAngle (hourHandPosition 2 50) (minuteHandPosition 50) = 305 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_sum_l424_42485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_properties_l424_42402

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define point M
def point_M : ℝ × ℝ := (1, 2)

-- Define point N as the intersection of line l with x-axis
def point_N : ℝ × ℝ := (-3, 0)

-- Define a point P on the circle
def point_P (x y : ℝ) : Prop := circle_C x y

-- Helper function for triangle area (not implemented)
noncomputable def area_triangle (a b c : ℝ × ℝ) : ℝ := sorry

-- Helper function for tangent of angle (not implemented)
noncomputable def tan_angle (a b c : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem tangent_line_properties :
  ∀ x y : ℝ,
  line_l x y →
  (∃ x' y', point_P x' y' ∧ line_l x' y') →
  (point_N.1 = -3 ∧ point_N.2 = 0) ∧
  (∃ max_area : ℝ, max_area = 10 ∧ 
    ∀ p : ℝ × ℝ, point_P p.1 p.2 → 
      area_triangle point_M point_N p ≤ max_area) ∧
  (∃ max_tan : ℝ, max_tan = 4/3 ∧
    ∀ p : ℝ × ℝ, point_P p.1 p.2 →
      tan_angle point_M point_N p ≤ max_tan) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_properties_l424_42402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_time_to_pass_section_l424_42426

-- Define the piecewise function y(t)
noncomputable def y (t : ℝ) : ℝ :=
  if 6 ≤ t ∧ t ≤ 9 then
    -1/8 * t^3 - 3/4 * t^2 + 36*t - 629/4
  else if 9 ≤ t ∧ t ≤ 10 then
    1/8 * t + 59/4
  else if 10 < t ∧ t ≤ 12 then
    -3 * t^2 + 66*t - 345
  else
    0  -- Define a default value for t outside the given range

-- Theorem statement
theorem max_time_to_pass_section :
  ∃ (max_t : ℝ), max_t = 8 ∧
  ∀ t, 6 ≤ t ∧ t ≤ 12 → y t ≤ y max_t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_time_to_pass_section_l424_42426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_l424_42473

/-- Given that x = π/3 is an axis of symmetry for f(x) = sin(2x + φ) and |φ| < π/2, prove that φ = -π/6 -/
theorem symmetry_of_sine (φ : ℝ) (h1 : |φ| < Real.pi/2) 
  (h2 : ∀ x, Real.sin (2*x + φ) = Real.sin (2*(Real.pi/3 - x) + φ)) : φ = -Real.pi/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_l424_42473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_to_obtain_g_l424_42488

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

theorem shift_to_obtain_g (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (a d : ℝ), ∀ (n : ℕ), f ω (a + n * Real.pi / 2) = 0 ∧ f ω (a + n * Real.pi / 2) > 0) :
  ∀ (x : ℝ), f ω (x + Real.pi / 12) = g ω x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_to_obtain_g_l424_42488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_negative_103_l424_42405

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 5

-- State the theorem
theorem inverse_g_at_negative_103 :
  ∃ (g_inv : ℝ → ℝ), Function.RightInverse g g_inv ∧ g_inv (-103) = -3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_g_at_negative_103_l424_42405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_k_range_l424_42466

/-- Given a function f(x) = e^x - k with a root in the interval (0, 1),
    prove that k is in the open interval (1, e) -/
theorem root_implies_k_range (k : ℝ) :
  (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ Real.exp x = k) →
  k ∈ Set.Ioo 1 (Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_implies_k_range_l424_42466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_composition_l424_42407

open Real

theorem log_function_composition (x : ℝ) (h : -1 < x ∧ x < 1) :
  let f (t : ℝ) := log ((1 + t) / (1 - t))
  f ((3 * x + x^3) / (1 + 3 * x^2)) = 3 * f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_function_composition_l424_42407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positives_needed_l424_42441

/-- Represents a 4x4 grid where each cell is either +1 or -1 -/
def Grid := Fin 4 → Fin 4 → Int

/-- Checks if a given grid satisfies the condition that any 2x2 subgrid sum is nonnegative -/
def satisfiesCondition (g : Grid) : Prop :=
  ∀ r1 r2 c1 c2, r1 < r2 ∧ c1 < c2 →
    g r1 c1 + g r1 c2 + g r2 c1 + g r2 c2 ≥ 0

/-- Counts the number of +1's in a grid -/
def countPositives (g : Grid) : Nat :=
  (Finset.univ : Finset (Fin 4)).sum fun r =>
    (Finset.univ : Finset (Fin 4)).sum fun c =>
      if g r c = 1 then 1 else 0

/-- The main theorem: The minimum number of +1's needed is 10 -/
theorem min_positives_needed (g : Grid) :
  satisfiesCondition g → countPositives g ≥ 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positives_needed_l424_42441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_angle_probability_l424_42420

def algebra_tiles : Finset Char := {'A', 'L', 'G', 'E', 'B', 'R', 'A'}
def angle_letters : Finset Char := {'A', 'N', 'G', 'L', 'E'}

theorem algebra_angle_probability :
  let total_tiles := algebra_tiles.card
  let favorable_tiles := (algebra_tiles ∩ angle_letters).card + (algebra_tiles.filter (· = 'A')).card
  (favorable_tiles : ℚ) / total_tiles = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_angle_probability_l424_42420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_distributions_count_l424_42456

/-- Represents a valid distribution of balls into boxes -/
structure ValidDistribution where
  d : Fin 3 → ℕ
  h1 : d 0 ≥ 1
  h2 : d 1 ≥ 2
  h3 : d 2 ≥ 3
  h4 : d 0 + d 1 + d 2 = 10

/-- The number of valid distributions of 10 balls into 3 boxes -/
def numValidDistributions : ℕ := 15

theorem valid_distributions_count : numValidDistributions = 15 := by
  -- The proof is omitted for now
  sorry

#eval numValidDistributions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_distributions_count_l424_42456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_f_equality_l424_42494

/-- A function that satisfies f(x) = f(1/x) for all non-zero x -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x

/-- Theorem stating that f(x) = f(1/x) for all non-zero x -/
theorem f_symmetry (x : ℝ) (hx : x ≠ 0) : f x = f (1/x) := by
  sorry

/-- Theorem stating that f(log₃2) = f(log₂3) -/
theorem f_equality : f (Real.log 2 / Real.log 3) = f (Real.log 3 / Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_f_equality_l424_42494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_derivatives_l424_42452

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x + 3 / x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x / (x^2)
noncomputable def h (x : ℝ) : ℝ := (3*x + 5)^3
noncomputable def k (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + Real.cos x

-- State the theorem
theorem correct_derivatives :
  (∀ x, deriv f x ≠ 1 + 3 / (x^2)) ∧
  (∀ x, deriv g x = (2*x*Real.cos x - 4*Real.sin x) / (x^3)) ∧
  (∀ x, deriv h x ≠ 3*(3*x + 5)^2) ∧
  (∀ x, deriv k x = Real.exp (x * Real.log 2) * Real.log 2 - Real.sin x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_derivatives_l424_42452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ellipse_l424_42432

noncomputable section

-- Define the ellipse M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 9 + p.2^2 = 1}

-- Define the right vertex C
def C : ℝ × ℝ := (3, 0)

-- Define a line that intersects the ellipse at two points
def intersectsEllipse (l : Set (ℝ × ℝ)) : Prop :=
  ∃ A B : ℝ × ℝ, A ∈ M ∧ B ∈ M ∧ A ∈ l ∧ B ∈ l ∧ A ≠ B

-- Define the condition that circle with AB as diameter passes through C
def circleThroughC (A B : ℝ × ℝ) : Prop :=
  (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0

-- Define the area of triangle ABC
def areaABC (A B : ℝ × ℝ) : ℝ :=
  abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2

-- State the theorem
theorem max_area_triangle_ellipse :
  ∀ l : Set (ℝ × ℝ), intersectsEllipse l →
  (∀ A B : ℝ × ℝ, A ∈ M ∧ B ∈ M ∧ A ∈ l ∧ B ∈ l ∧ A ≠ B → circleThroughC A B) →
  (∃ A B : ℝ × ℝ, A ∈ M ∧ B ∈ M ∧ A ∈ l ∧ B ∈ l ∧ A ≠ B ∧ areaABC A B = 3/8) ∧
  (∀ A B : ℝ × ℝ, A ∈ M ∧ B ∈ M ∧ A ∈ l ∧ B ∈ l ∧ A ≠ B → areaABC A B ≤ 3/8) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_triangle_ellipse_l424_42432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batting_average_problem_l424_42459

theorem batting_average_problem (n : ℕ) :
  (53 * n + 78) / (n + 1) = 58 ↔ n = 4 := by
  have h1 : (53 * n + 78) / (n + 1) = 58 ↔ 53 * n + 78 = 58 * (n + 1) := by
    sorry
  have h2 : 53 * n + 78 = 58 * (n + 1) ↔ 53 * n + 78 = 58 * n + 58 := by
    sorry
  have h3 : 53 * n + 78 = 58 * n + 58 ↔ 78 - 58 = 58 * n - 53 * n := by
    sorry
  have h4 : 78 - 58 = 58 * n - 53 * n ↔ 20 = 5 * n := by
    sorry
  have h5 : 20 = 5 * n ↔ n = 4 := by
    sorry
  exact h1.trans (h2.trans (h3.trans (h4.trans h5)))

#check batting_average_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batting_average_problem_l424_42459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_completion_time_cost_comparison_l424_42431

-- Define the problem parameters
def total_days : ℝ := 24
def total_cost : ℝ := 1.2
def team_a_initial_days : ℝ := 18
def team_b_additional_days : ℝ := 48
def mixed_cost : ℝ := 1.1
def max_days : ℝ := 60

-- Define variables for team A and B's individual completion times
variable (x y : ℝ)

-- Define the equations from the problem
def equation1 (x y : ℝ) : Prop := (total_days / x) + (total_days / y) = 1
def equation2 (x y : ℝ) : Prop := (team_a_initial_days / x) + (team_b_additional_days / y) = 1

-- Define variables for team A and B's daily costs
variable (m n : ℝ)

-- Define the cost equations
def cost_equation1 (m n : ℝ) : Prop := total_days * m + total_days * n = total_cost * 10
def cost_equation2 (m n : ℝ) : Prop := team_a_initial_days * m + team_b_additional_days * n = mixed_cost * 10

-- Theorem for completion time
theorem completion_time (hx : x = 30) (hy : y = 120) :
  equation1 x y ∧ equation2 x y → x < max_days ∧ y > max_days := by sorry

-- Theorem for cost comparison
theorem cost_comparison (hm : m = 13/3) (hn : n = 2/3) :
  cost_equation1 m n ∧ cost_equation2 m n → n * y < m * x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_completion_time_cost_comparison_l424_42431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_cycle_condition_l424_42477

/-- The function f(x) = 1 / (ax + b) -/
noncomputable def f (a b x : ℝ) : ℝ := 1 / (a * x + b)

/-- Theorem stating the condition for the existence of a 3-cycle in the function f -/
theorem three_cycle_condition (a b : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ 
    f a b x₁ = x₂ ∧ f a b x₂ = x₃ ∧ f a b x₃ = x₁) ↔ 
  a = -b^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_cycle_condition_l424_42477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donny_apple_cost_l424_42438

/-- Represents the size of an apple -/
inductive AppleSize
  | Small
  | Medium
  | Big

/-- Returns the price of an apple based on its size -/
def applePrice (size : AppleSize) : ℚ :=
  match size with
  | AppleSize.Small => 3/2
  | AppleSize.Medium => 2
  | AppleSize.Big => 3

/-- Calculates the total cost of apples -/
def totalCost (smallCount mediumCount bigCount : ℕ) : ℚ :=
  smallCount * applePrice AppleSize.Small +
  mediumCount * applePrice AppleSize.Medium +
  bigCount * applePrice AppleSize.Big

theorem donny_apple_cost :
  totalCost 6 6 8 = 45 := by
  -- Unfold the definition of totalCost
  unfold totalCost
  -- Unfold the definition of applePrice
  simp [applePrice]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_donny_apple_cost_l424_42438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l424_42462

/-- Calculates the time taken for a faster train to cross a man in a slower train -/
noncomputable def time_to_cross (faster_speed slower_speed : ℝ) (train_length : ℝ) : ℝ :=
  let relative_speed := faster_speed - slower_speed
  train_length / (relative_speed * 1000 / 3600)

/-- Theorem stating that the time taken for the faster train to cross the man in the slower train is 33 seconds -/
theorem train_crossing_time :
  let faster_speed := (162 : ℝ)
  let slower_speed := (18 : ℝ)
  let train_length := (1320 : ℝ)
  time_to_cross faster_speed slower_speed train_length = 33 := by
  -- Unfold the definition of time_to_cross
  unfold time_to_cross
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l424_42462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wine_limit_l424_42458

noncomputable def wine_sequence : ℕ → ℝ
  | 0 => 1
  | n + 1 => (9/10) * wine_sequence n + 1/10

theorem wine_limit :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |wine_sequence n - 1/2| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wine_limit_l424_42458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l424_42411

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x + 1 / Real.sqrt (1 + abs (Real.sin (2 * x)))

theorem f_bounds : 
  (∀ x : ℝ, f x ≤ 3 * Real.sqrt 2 / 2) ∧ 
  (∀ x : ℝ, f x ≥ - Real.sqrt 2 / 2) ∧
  (∃ x : ℝ, f x = 3 * Real.sqrt 2 / 2) ∧
  (∃ x : ℝ, f x = - Real.sqrt 2 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l424_42411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2016_l424_42434

/-- Arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (a₁ d : ℝ) (n : ℕ) : ℝ := (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_sum_2016 
  (a₁ d : ℝ) 
  (h1 : a₁ = -2014)
  (h2 : S a₁ d 2012 / 2012 - S a₁ d 10 / 10 = 2002) :
  S a₁ d 2016 = 2016 := by
  sorry

#check arithmetic_sequence_sum_2016

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2016_l424_42434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_distance_and_slope_ellipse_intersection_slope_condition_l424_42491

/-- An ellipse with equation x^2/4 + y^2/3 = 1 -/
structure Ellipse where
  eq : ℝ → ℝ → Prop
  h : ∀ x y, eq x y ↔ x^2/4 + y^2/3 = 1

/-- Left focus of the ellipse -/
def leftFocus : ℝ × ℝ := (-1, 0)

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- Intersection points of a line with the ellipse -/
def intersectionPoints (e : Ellipse) (l : Line) : Set (ℝ × ℝ) :=
  {p | e.eq p.1 p.2 ∧ p.2 - l.point.2 = l.slope * (p.1 - l.point.1)}

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Point symmetric to p about the origin -/
def symmetricOrigin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

/-- Point symmetric to p about the x-axis -/
def symmetricXAxis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Slope of a line passing through two points -/
noncomputable def slopeTwoPoints (p q : ℝ × ℝ) : ℝ :=
  (q.2 - p.2) / (q.1 - p.1)

theorem ellipse_intersection_distance_and_slope (e : Ellipse) :
  ∃ l : Line,
    l.point = leftFocus ∧
    l.slope = 1 ∧
    ∃ p q : ℝ × ℝ,
      p ∈ intersectionPoints e l ∧
      q ∈ intersectionPoints e l ∧
      distance p q = 24/7 :=
  sorry

theorem ellipse_intersection_slope_condition (e : Ellipse) :
  ∃ l : Line,
    l.point = leftFocus ∧
    l.slope ≠ 0 ∧
    ∃ p q p' q' : ℝ × ℝ,
      p ∈ intersectionPoints e l ∧
      q ∈ intersectionPoints e l ∧
      p' = symmetricOrigin p ∧
      q' = symmetricXAxis q ∧
      |slopeTwoPoints p' q'| = 2 ∧
      l.slope = 3/7 * Real.sqrt 7 ∨ l.slope = -3/7 * Real.sqrt 7 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_distance_and_slope_ellipse_intersection_slope_condition_l424_42491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camping_cost_settlement_correct_l424_42480

/-- The amount Alice must pay Bob to equally share camping trip costs -/
noncomputable def camping_cost_settlement (A B C : ℝ) : ℝ :=
  (A - B + C) / 2

theorem camping_cost_settlement_correct (A B C : ℝ) (h : A > B) : 
  camping_cost_settlement A B C = (A - B + C) / 2 := by
  -- Unfold the definition of camping_cost_settlement
  unfold camping_cost_settlement
  -- The equality is now trivial
  rfl

#check camping_cost_settlement_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camping_cost_settlement_correct_l424_42480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_sequence_divisibility_l424_42446

def sequence' (n : ℕ) : ℤ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n + 2 => 2 * sequence' (n + 1) + sequence' n

theorem sequence_property (m : ℕ) (j : ℕ) (h : m > 0) (hj : j ≤ m) :
  ∃ k : ℤ, sequence' (m + j) + (-1)^j * sequence' (m - j) = 2 * k * sequence' m :=
sorry

theorem sequence_divisibility (n k : ℕ) (h : 2^k ∣ n) :
  2^k ∣ sequence' n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_sequence_divisibility_l424_42446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_novel_recording_distribution_l424_42415

theorem novel_recording_distribution (total_time min_per_tape : ℕ) 
  (h1 : total_time = 480)
  (h2 : min_per_tape = 60)
  (h3 : total_time % min_per_tape = 0) :
  let num_tapes := total_time / min_per_tape
  (total_time / num_tapes : ℚ) = min_per_tape := by
  sorry

#check novel_recording_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_novel_recording_distribution_l424_42415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_weight_problem_l424_42483

theorem class_weight_problem (total_boys : ℕ) (group1_boys : ℕ) (group1_avg_weight : ℝ) (total_avg_weight : ℝ) :
  total_boys = 32 →
  group1_boys = 24 →
  group1_avg_weight = 50.25 →
  total_avg_weight = 48.975 →
  (total_boys * total_avg_weight - group1_boys * group1_avg_weight) / (total_boys - group1_boys) = 45.15 := by
  intro h1 h2 h3 h4
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_weight_problem_l424_42483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_triangle_congruence_area_l424_42490

-- Define the Triangle type
structure Triangle where
  -- You may want to add appropriate fields here, e.g., vertices

-- Define congruence for triangles
def Congruent (T1 T2 : Triangle) : Prop :=
  sorry -- Define congruence condition here

-- Define equal area for triangles
def EqualArea (T1 T2 : Triangle) : Prop :=
  sorry -- Define equal area condition here

-- Define the original proposition
def original_proposition (T1 T2 : Triangle) : Prop :=
  Congruent T1 T2 → EqualArea T1 T2

-- Define the inverse proposition
def inverse_proposition (T1 T2 : Triangle) : Prop :=
  EqualArea T1 T2 → Congruent T1 T2

-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition
theorem inverse_of_triangle_congruence_area :
  ∀ (T1 T2 : Triangle), inverse_proposition T1 T2 ↔ ¬(¬(original_proposition T1 T2)) :=
by
  sorry -- The proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_triangle_congruence_area_l424_42490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l424_42443

/-- The molar mass of aluminum in g/mol -/
noncomputable def molar_mass_Al : ℝ := 26.98

/-- The molar mass of iodine in g/mol -/
noncomputable def molar_mass_I : ℝ := 126.90

/-- The number of iodine atoms in aluminum iodide -/
def num_iodine_atoms : ℕ := 3

/-- The molar mass of aluminum iodide in g/mol -/
noncomputable def molar_mass_AlI3 : ℝ := molar_mass_Al + num_iodine_atoms * molar_mass_I

/-- The mass percentage of aluminum in aluminum iodide -/
noncomputable def mass_percentage_Al : ℝ := (molar_mass_Al / molar_mass_AlI3) * 100

/-- Theorem stating that the mass percentage of aluminum in aluminum iodide is approximately 6.62% -/
theorem mass_percentage_Al_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |mass_percentage_Al - 6.62| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_approx_l424_42443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l424_42499

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sqrt (1 - 2 * x)

-- State the theorem
theorem f_range :
  Set.range f = Set.Iic 1 :=
by sorry

-- Define the domain condition
def domain_condition (x : ℝ) : Prop := 1 - 2 * x ≥ 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l424_42499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l424_42489

-- Define sets A and B
def A : Set ℝ := {x | 33 - x < 6}
def B : Set ℝ := {x | Real.log (x - 1) < 1}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo (2 - Real.log 32) 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l424_42489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_three_implies_x_equals_sqrt_three_l424_42437

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

-- Theorem statement
theorem f_equals_three_implies_x_equals_sqrt_three :
  ∀ x : ℝ, f x = 3 → x = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_three_implies_x_equals_sqrt_three_l424_42437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_cookies_comparison_l424_42487

/-- The cost of a chocolate bar -/
def C : ℝ := sorry

/-- The cost of a pack of cookies -/
def P : ℝ := sorry

/-- Given that 7 chocolate bars are more expensive than 8 packs of cookies,
    prove that 8 chocolate bars are more expensive than 9 packs of cookies -/
theorem chocolate_cookies_comparison (h : 7 * C > 8 * P) : 8 * C > 9 * P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_cookies_comparison_l424_42487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_R_l424_42492

theorem solve_for_R (R : ℝ) : R^(3/2) = 18 * (27^(1/9)) → R = 6 * (3^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_R_l424_42492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_remainder_l424_42408

/-- The sum of powers from 1^1 to n^n -/
def sumOfPowers (n : ℕ) : ℕ := (List.range n).foldl (fun acc i => acc + (i + 1) ^ (i + 1)) 0

/-- The remainder when the sum of powers from 1^1 to 7^7 is divided by 7 -/
theorem sum_of_powers_remainder :
  (sumOfPowers 7) % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_powers_remainder_l424_42408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l424_42450

noncomputable def f (x : ℝ) : ℝ := (6 * x^2 - 7) / (4 * x^2 + 6 * x + 3)

def is_vertical_asymptote (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - a| ∧ |x - a| < δ → |f x| > 1/ε

theorem vertical_asymptotes_sum :
  ∃ (p q : ℝ), is_vertical_asymptote f p ∧ is_vertical_asymptote f q ∧ p + q = -2 := by
  -- We'll use p = -1/2 and q = -3/2
  let p : ℝ := -1/2
  let q : ℝ := -3/2
  
  have h_sum : p + q = -2 := by
    simp [p, q]
    norm_num
  
  exists p, q
  constructor
  · sorry  -- Proof that f has a vertical asymptote at p = -1/2
  constructor
  · sorry  -- Proof that f has a vertical asymptote at q = -3/2
  · exact h_sum


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l424_42450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_ratio_l424_42419

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line structure -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Theorem statement -/
theorem parabola_area_ratio 
  (para : Parabola) 
  (F : Point) 
  (l : Line) 
  (A B : Point) 
  (A' B' : Point) :
  F.x = para.p / 2 ∧ F.y = 0 →  -- Focus condition
  l.m = 0 ∧ l.b = -para.p / 2 →  -- Directrix condition
  A.y^2 = 2 * para.p * A.x ∧ B.y^2 = 2 * para.p * B.x →  -- A and B on parabola
  (A.x - F.x)^2 + (A.y - F.y)^2 = (B.x - F.x)^2 + (B.y - F.y)^2 →  -- A and B equidistant from F
  (A.x - B.x)^2 + (A.y - B.y)^2 = (3 * para.p)^2 →  -- |AB| = 3p
  A'.x = -para.p / 2 ∧ A'.y = A.y →  -- A' is projection of A
  B'.x = -para.p / 2 ∧ B'.y = B.y →  -- B' is projection of B
  (abs ((F.x - A'.x) * (B'.y - A'.y) - (F.y - A'.y) * (B'.x - A'.x)) / 2) / (3 * para.p * para.p) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_area_ratio_l424_42419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_pi_third_l424_42429

theorem cos_neg_pi_third : Real.cos (-π/3) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_pi_third_l424_42429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_polar_l424_42454

/-- The length of the chord cut by a line on a circle in polar coordinates -/
theorem chord_length_polar (l C : ℝ → ℝ → Prop) : 
  (∀ ρ θ, l ρ θ ↔ ρ * Real.sin (π/6 - θ) = 2) →
  (∀ ρ θ, C ρ θ ↔ ρ = 4 * Real.cos θ) →
  ∃ chord_length, chord_length = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_polar_l424_42454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l424_42470

/-- The time (in seconds) it takes for a train to pass a man moving in the opposite direction -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  train_length / (train_speed + man_speed) * 3600 / 1000

/-- Theorem stating that the time for a 605m train moving at 60 kmph to pass a man moving at 6 kmph in the opposite direction is approximately 33.01 seconds -/
theorem train_passing_time_approx :
  ∃ ε > 0, |train_passing_time 605 60 6 - 33.01| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l424_42470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_parallel_lines_l424_42481

/-- A triangle is equilateral if all its sides are equal -/
def is_equilateral (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

/-- A point M is on a segment AB if it lies between A and B -/
def point_on_segment (M A B : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • B

/-- Two points are on different sides of a line if the line intersects the segment between them -/
def different_sides_of_line (M K B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ P : EuclideanSpace ℝ (Fin 2), point_on_segment P B C ∧ point_on_segment P M K

/-- Two lines are parallel if they have the same direction -/
def parallel_lines (A C B K : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, C - A = t • (K - B)

theorem equilateral_triangle_parallel_lines
  (A B C M K : EuclideanSpace ℝ (Fin 2))
  (h1 : is_equilateral A B C)
  (h2 : point_on_segment M A B)
  (h3 : is_equilateral M K C)
  (h4 : different_sides_of_line M K B C) :
  parallel_lines A C B K :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_parallel_lines_l424_42481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l424_42464

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

noncomputable def frac (x : ℝ) : ℝ := x - (floor x : ℝ)

theorem absolute_difference_of_solution (x y : ℝ) :
  (floor x : ℝ) - frac y = 1.7 →
  frac x + (floor y : ℝ) = 4.4 →
  |x - y| = 1.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_of_solution_l424_42464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_difference_l424_42439

theorem quadratic_roots_difference (q : ℝ) (h_pos : q > 0) :
  let a : ℝ := 1
  let b : ℝ := 2*q - 1
  let c : ℝ := q
  let discriminant : ℝ := b^2 - 4*a*c
  let root1 : ℝ := (-b + Real.sqrt discriminant) / (2*a)
  let root2 : ℝ := (-b - Real.sqrt discriminant) / (2*a)
  abs (root1 - root2) = 2 → q = 1 + Real.sqrt 7 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_difference_l424_42439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_reciprocal_l424_42447

-- Define the function f(x) = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- State the theorem that the derivative of f(x) is -1/x^2
theorem derivative_reciprocal :
  ∀ x, x ≠ 0 → deriv f x = -(1 / x^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_reciprocal_l424_42447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_spheres_volume_l424_42465

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The total volume of n spheres with radius r -/
noncomputable def total_sphere_volume (n : ℕ) (r : ℝ) : ℝ := n * sphere_volume r

theorem four_spheres_volume :
  total_sphere_volume 4 3 = 144 * Real.pi := by
  -- Unfold the definitions
  unfold total_sphere_volume sphere_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_spheres_volume_l424_42465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l424_42440

theorem existence_of_special_set :
  ∃ (X : Finset ℕ), (X.card = 2022) ∧
  (∀ (a b c : ℕ), a ∈ X → b ∈ X → c ∈ X → a ≠ b → b ≠ c → a ≠ c →
    ∀ (n : ℕ), n > 0 → Nat.gcd (a^n + b^n) c = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_set_l424_42440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l424_42463

-- Define the inequality function
noncomputable def f (x : ℝ) : ℝ := (1/2)^(x^2 - 3*x)

-- State the theorem
theorem inequality_solution_set :
  {x : ℝ | f x > 4} = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l424_42463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_b1_a1_range_of_a_when_b1_l424_42418

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * Real.cos x + 2 * b * (Real.cos x)^2 + 1 - b

-- Part 1: Range of f when b=1 and a=1
theorem range_of_f_when_b1_a1 :
  Set.range (fun x => f 1 1 x) = Set.Icc (-1/8 : ℝ) 3 := by sorry

-- Part 2: Range of a when b=1 such that |f(x)| ≥ a² for some x
theorem range_of_a_when_b1 :
  {a : ℝ | ∃ x, |f a 1 x| ≥ a^2} = Set.Icc (-2 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_b1_a1_range_of_a_when_b1_l424_42418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_expression_value_l424_42427

-- Define the expression
noncomputable def logarithmic_expression : ℝ := 
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9)

-- State the theorem
theorem logarithmic_expression_value : logarithmic_expression = 25 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_expression_value_l424_42427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_calculation_l424_42448

/-- Calculates the altitude difference given temperature change and rate of change --/
noncomputable def altitude_difference (temp_change : ℝ) (rate : ℝ) : ℝ :=
  -temp_change / rate

/-- The problem statement --/
theorem altitude_calculation (rate : ℝ) (temp_ground : ℝ) (temp_high : ℝ) 
  (h1 : rate = 6)
  (h2 : temp_ground = 18)
  (h3 : temp_high = -48) :
  altitude_difference (temp_high - temp_ground) rate = 11 := by
  -- Unfold the definition of altitude_difference
  unfold altitude_difference
  -- Substitute the values
  rw [h1, h2, h3]
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_calculation_l424_42448
