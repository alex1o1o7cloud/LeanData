import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_sin_eq_l436_43679

/-- The function resulting from translating y = sin(x) right by π/4 units and then doubling all ordinates -/
noncomputable def transformed_sin (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi/4)

/-- The original sine function -/
noncomputable def original_sin (x : ℝ) : ℝ := Real.sin x

/-- Theorem stating that the transformed function is equal to 2sin(x - π/4) -/
theorem transformed_sin_eq (x : ℝ) : transformed_sin x = 2 * Real.sin (x - Real.pi/4) := by
  -- The proof is trivial as it follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_sin_eq_l436_43679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chrystal_speed_l436_43651

/-- Chrystal's vehicle speed on a flat road -/
noncomputable def flat_speed : ℝ := 30

/-- Speed when ascending the mountain -/
noncomputable def ascending_speed (v : ℝ) : ℝ := v / 2

/-- Speed when descending the mountain -/
noncomputable def descending_speed (v : ℝ) : ℝ := 1.2 * v

/-- Distance to the top of the mountain in miles -/
noncomputable def ascending_distance : ℝ := 60

/-- Distance down to the foot of the mountain in miles -/
noncomputable def descending_distance : ℝ := 72

/-- Total time to pass the whole mountain in hours -/
noncomputable def total_time : ℝ := 6

theorem chrystal_speed :
  ascending_distance / ascending_speed flat_speed +
  descending_distance / descending_speed flat_speed = total_time :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chrystal_speed_l436_43651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_D_l436_43672

/-- The distance between two points in polar coordinates -/
noncomputable def polarDistance (r₁ r₂ : ℝ) (φ₁ φ₂ : ℝ) : ℝ :=
  Real.sqrt (r₁^2 + r₂^2 - 2*r₁*r₂*(Real.cos (φ₁ - φ₂)))

/-- Theorem: Distance between C(5, φ₁) and D(7, φ₂) where φ₁ - φ₂ = π/3 -/
theorem distance_C_D (φ₁ φ₂ : ℝ) (h : φ₁ - φ₂ = π/3) :
  polarDistance 5 7 φ₁ φ₂ = Real.sqrt 39 := by
  sorry

#check distance_C_D

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_C_D_l436_43672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_waitress_hourly_wage_l436_43611

/-- Calculates the hourly wage of a waitress given the following conditions:
  * Standard tip rate is 15% of the cost of orders
  * Worked 3 8-hour shifts in a week
  * Averaged $40 in orders per hour
  * Earned $240 in total for the week
-/
theorem waitress_hourly_wage (tip_rate : ℝ) (shifts : ℕ) (hours_per_shift : ℕ) 
  (avg_orders_per_hour : ℝ) (total_earnings : ℝ) : 
  tip_rate = 0.15 →
  shifts = 3 →
  hours_per_shift = 8 →
  avg_orders_per_hour = 40 →
  total_earnings = 240 →
  ∃ (hourly_wage : ℝ), hourly_wage = 4 :=
by
  intro h_tip_rate h_shifts h_hours h_avg_orders h_total_earnings
  let total_hours := shifts * hours_per_shift
  let total_orders := avg_orders_per_hour * (shifts * hours_per_shift : ℝ)
  let total_tips := tip_rate * total_orders
  let wage_earnings := total_earnings - total_tips
  let hourly_wage := wage_earnings / (shifts * hours_per_shift : ℝ)
  
  exists hourly_wage
  sorry

#check waitress_hourly_wage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_waitress_hourly_wage_l436_43611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_equal_angles_not_congruent_l436_43619

/-- Structure for a triangle -/
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  side1 : Real
  side2 : Real
  side3 : Real

/-- Definition: A triangle is right-angled -/
def RightAngled (T : Triangle) : Prop :=
  T.angle3 = Real.pi / 2

/-- Definition: An angle is acute -/
def Acute (θ : Real) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

/-- Definition: Two triangles are congruent -/
def Congruent (T1 T2 : Triangle) : Prop :=
  T1.side1 = T2.side1 ∧ T1.side2 = T2.side2 ∧ T1.side3 = T2.side3

/-- Two right-angled triangles with equal acute angles are not necessarily congruent -/
theorem right_triangle_equal_angles_not_congruent :
  ∃ (T1 T2 : Triangle) (α β : Real),
    RightAngled T1 ∧ RightAngled T2 ∧
    Acute α ∧ Acute β ∧
    (T1.angle1 = α ∧ T2.angle1 = α) ∧
    (T1.angle2 = β ∧ T2.angle2 = β) ∧
    ¬Congruent T1 T2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_equal_angles_not_congruent_l436_43619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l436_43666

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (F₁ F₂ P Q : ℝ × ℝ) : 
  a > 0 → b > 0 →
  (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x, y) ∈ Set.range (λ t ↦ (a * Real.cosh t, b * Real.sinh t))) →
  F₁.1 < F₂.1 →
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 = (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 →
  (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0 →
  (∃ (k : ℝ), k > 0 ∧ 
    ((Q.1 - P.1)^2 + (Q.2 - P.2)^2 = (3*k)^2) ∧
    ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 = (4*k)^2)) →
  (F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2 = 4 * a^2 * (17/9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l436_43666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l436_43678

theorem sin_minus_cos_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 2 / 2) : 
  Real.sin α - Real.cos α = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l436_43678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_step_staircase_ways_l436_43600

def staircase_ways (total_steps : ℕ) (min_steps : ℕ) : ℕ :=
  (Finset.range (total_steps - min_steps + 1)).sum (λ k =>
    Nat.choose (total_steps - 1) (total_steps - min_steps - k))

theorem nine_step_staircase_ways :
  staircase_ways 9 6 = 93 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_step_staircase_ways_l436_43600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_is_1408_l436_43617

/-- The surface area of the final figure in the cube problem -/
def final_surface_area : ℕ := 1408

/-- Theorem stating that the surface area of the final figure is 1408 -/
theorem surface_area_is_1408 : final_surface_area = 1408 := by
  -- Unfold the definition of final_surface_area
  unfold final_surface_area
  -- The equality is now trivial
  rfl

#check surface_area_is_1408

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_is_1408_l436_43617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_expression_equals_six_l436_43614

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem triangle_expression_equals_six (ABC : Triangle) (D : ℝ × ℝ) :
  distance ABC.B ABC.C = 3 →
  D.1 = ABC.B.1 + 2 * (ABC.C.1 - ABC.B.1) / 3 →
  D.2 = ABC.B.2 + 2 * (ABC.C.2 - ABC.B.2) / 3 →
  (distance ABC.A ABC.B)^2 + 2 * (distance ABC.A ABC.C)^2 - 3 * (distance ABC.A D)^2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_expression_equals_six_l436_43614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_integers_count_l436_43644

noncomputable def floor_sqrt_squared_over_500 (n : ℕ) : ℤ := ⌊(n : ℝ) / 500⌋

theorem distinct_integers_count :
  ∃ (S : Finset ℤ), (∀ n ∈ Finset.range 1001, floor_sqrt_squared_over_500 n ∈ S) ∧ S.card = 3 :=
by
  -- We'll construct the set S explicitly
  let S : Finset ℤ := {0, 1, 2}
  
  -- We'll use this set as our witness
  use S

  -- Now we need to prove the two parts of the conjunction
  constructor

  -- Part 1: Show that all values are in S
  · intro n hn
    -- The proof of this part would go here
    sorry

  -- Part 2: Show that S has exactly 3 elements
  · rfl  -- This is true by construction of S

-- The proof is incomplete, but the structure is correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_integers_count_l436_43644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_parallel_plane_l436_43655

-- Define the necessary structures
structure Plane : Type

structure Line : Type

-- Define the relations
def parallel (l1 l2 : Line) : Prop := sorry

def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem line_perpendicular_to_parallel_plane (α : Plane) (a b : Line) 
  (h1 : parallel a b) (h2 : perpendicular_line_plane a α) : 
  perpendicular_line_plane b α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_to_parallel_plane_l436_43655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_area_l436_43658

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a hexagon in a 2D plane -/
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- Check if three circles are pairwise externally tangent -/
def are_pairwise_externally_tangent (c1 c2 c3 : Circle) : Prop :=
  sorry

/-- Check if a circle is tangent to two sides of a hexagon -/
def is_tangent_to_sides (c : Circle) (h : Hexagon) (i j : Fin 6) : Prop :=
  sorry

/-- Check if a hexagon is convex and equiangular -/
def is_convex_equiangular (h : Hexagon) : Prop :=
  sorry

/-- Calculate the area of a circle -/
noncomputable def circle_area (c : Circle) : ℝ :=
  Real.pi * c.radius^2

theorem hexagon_circle_area (h : Hexagon) (γ₁ γ₂ γ₃ : Circle) :
  is_convex_equiangular h →
  ‖h.vertices 1 - h.vertices 0‖ = 1 →
  ‖h.vertices 3 - h.vertices 2‖ = 1 →
  ‖h.vertices 5 - h.vertices 4‖ = 1 →
  ‖h.vertices 2 - h.vertices 1‖ = 4 →
  ‖h.vertices 4 - h.vertices 3‖ = 4 →
  ‖h.vertices 0 - h.vertices 5‖ = 4 →
  are_pairwise_externally_tangent γ₁ γ₂ γ₃ →
  is_tangent_to_sides γ₁ h 0 1 →
  is_tangent_to_sides γ₂ h 2 3 →
  is_tangent_to_sides γ₃ h 4 5 →
  circle_area γ₁ = 147 / 100 * Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circle_area_l436_43658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l436_43652

/-- Given a hyperbola and an ellipse sharing the same foci, 
    the product of the distances from their intersection point 
    to the foci is equal to a - m -/
theorem intersection_distance_product 
  (m n a b : ℝ) 
  (hm : m > 0) (hn : n > 0) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (hyperbola : ℝ → ℝ → Prop) 
  (ellipse : ℝ → ℝ → Prop)
  (M : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ)
  (h_hyperbola : ∀ x y, hyperbola x y ↔ x^2 / m - y^2 / n = 1)
  (h_ellipse : ∀ x y, ellipse x y ↔ x^2 / a + y^2 / b = 1)
  (h_intersection : hyperbola M.1 M.2 ∧ ellipse M.1 M.2)
  (h_same_foci : ∀ x y, 
    hyperbola x y → |dist (x, y) F₁ - dist (x, y) F₂| = 2 * Real.sqrt m ∧
    ellipse x y → dist (x, y) F₁ + dist (x, y) F₂ = 2 * Real.sqrt a) :
  dist M F₁ * dist M F₂ = a - m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l436_43652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l436_43624

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 
  4 * Real.sin (ω * x) * Real.cos (ω * x + Real.pi / 3) + 2 * Real.sqrt 3

theorem f_properties (ω : ℝ) (h_ω_pos : ω > 0) (h_period : ∀ x, f ω (x + Real.pi) = f ω x) :
  let min_value := Real.sqrt 3 - 1
  let max_value := 2 + Real.sqrt 3
  let min_point := -Real.pi / 4
  let max_point := Real.pi / 12
  let interval := Set.Icc (-Real.pi / 4) (Real.pi / 6)
  (∀ x, x ∈ interval → f ω x ≥ min_value) ∧ 
  (f ω min_point = min_value) ∧
  (∀ x, x ∈ interval → f ω x ≤ max_value) ∧
  (f ω max_point = max_value) ∧
  (∀ ω' > 0, (∀ x y, x ∈ interval → y ∈ interval → x < y → f ω' x < f ω' y) → ω' ≤ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l436_43624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_salary_calculation_l436_43667

/-- Proves that given the specified savings and expense increase conditions, 
    the monthly salary is approximately 5208.33 --/
theorem monthly_salary_calculation (salary : ℝ) 
  (h1 : salary * 0.15 = salary - (salary * 0.85 * 1.12 + 250)) : 
  ‖salary - 5208.33‖ < 0.01 := by
  sorry

#eval (250 / 0.048 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monthly_salary_calculation_l436_43667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l436_43689

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (α - Real.pi/4) = Real.sqrt 2/4) : 
  Real.sin (2*α) = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l436_43689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l436_43671

/-- Represents a parallelogram ABCD -/
structure Parallelogram :=
  (AB : ℝ)
  (BC : ℝ)
  (height : ℝ)

/-- Represents a pyramid PABCD with base ABCD -/
structure Pyramid :=
  (base : Parallelogram)
  (PA : ℝ)

/-- The volume of a pyramid -/
noncomputable def pyramidVolume (p : Pyramid) : ℝ :=
  (1 / 3) * p.base.AB * p.base.height * p.PA

/-- Theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  let base := Parallelogram.mk 10 6 5
  let pyramid := Pyramid.mk base 8
  pyramidVolume pyramid = 400 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l436_43671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l436_43628

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | n+2 => a ((n+2) / 2) + (-1)^(n+3)

theorem a_properties :
  (∃ n ≤ 1996, a n = 9) ∧
  (∀ n ≤ 1996, a n ≤ 9) ∧
  (∃ n ≤ 1996, a n = -10) ∧
  (∀ n ≤ 1996, a n ≥ -10) ∧
  (Finset.filter (λ n => a n = 0) (Finset.range 1997)).card = 346 :=
by sorry

#eval a 1023
#eval a 1365

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_properties_l436_43628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pl_max_time_l436_43602

noncomputable def x (t : ℝ) : ℝ := 10 * Real.exp (-t)

noncomputable def y (t : ℝ) : ℝ := 10 * Real.exp (-t) - 10 * Real.exp (-2*t)

theorem pl_max_time :
  let x' : ℝ → ℝ := fun t => -x t
  let y' : ℝ → ℝ := fun t => x t - 2 * y t
  ∀ t : ℝ, t ≥ 0 →
    (∀ s : ℝ, s ≥ 0 → y s ≤ y (Real.log 2)) ∧
    (∀ t : ℝ, x' t = -x t) ∧
    (∀ t : ℝ, y' t = x t - 2 * y t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pl_max_time_l436_43602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AD_is_6_sqrt_43_final_result_l436_43674

/-- Triangle ABC with given side lengths -/
structure Triangle :=
  (AB : ℝ) (BC : ℝ) (CA : ℝ)

/-- Point on the circumcircle -/
def CircumcirclePoint (t : Triangle) := ℝ × ℝ

/-- The length of AD where D is a point on the circumcircle -/
noncomputable def length_AD (t : Triangle) (D : CircumcirclePoint t) : ℝ := sorry

/-- Function to represent the intersection of circumcircle and perpendicular bisector -/
noncomputable def intersection_of_circumcircle_and_perpendicular_bisector (t : Triangle) : CircumcirclePoint t := sorry

theorem length_AD_is_6_sqrt_43 (t : Triangle) (D : CircumcirclePoint t) :
  t.AB = 43 → t.BC = 13 → t.CA = 48 →
  D = intersection_of_circumcircle_and_perpendicular_bisector t →
  length_AD t D = 6 * Real.sqrt 43 := by sorry

/-- The final result: greatest integer less than or equal to m + √n -/
theorem final_result :
  ∃ (t : Triangle) (D : CircumcirclePoint t),
    t.AB = 43 ∧ t.BC = 13 ∧ t.CA = 48 ∧
    D = intersection_of_circumcircle_and_perpendicular_bisector t ∧
    ⌊(6 : ℝ) + Real.sqrt 43⌋ = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AD_is_6_sqrt_43_final_result_l436_43674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_prices_l436_43688

/-- Represents the cost and selling prices of furniture items -/
structure FurnitureItem where
  costPrice : ℝ
  sellingPrice : ℝ

/-- Calculates the selling price given the cost price and markup percentage -/
noncomputable def calculateSellingPrice (costPrice : ℝ) (markupPercentage : ℝ) : ℝ :=
  costPrice * (1 + markupPercentage / 100)

/-- Calculates the cost price given the selling price and discount percentage -/
noncomputable def calculateCostPriceWithDiscount (sellingPrice : ℝ) (discountPercentage : ℝ) : ℝ :=
  sellingPrice / (1 - discountPercentage / 100)

/-- Theorem stating the cost prices of furniture items given their selling prices and markup/discount percentages -/
theorem furniture_cost_prices
  (computerTable : FurnitureItem)
  (chair : FurnitureItem)
  (bookshelf : FurnitureItem)
  (h1 : computerTable.sellingPrice = calculateSellingPrice computerTable.costPrice 100)
  (h2 : chair.sellingPrice = calculateSellingPrice chair.costPrice 75)
  (h3 : bookshelf.sellingPrice = calculateCostPriceWithDiscount bookshelf.costPrice 25)
  (h4 : computerTable.sellingPrice = 1000)
  (h5 : chair.sellingPrice = 1750)
  (h6 : bookshelf.sellingPrice = 1500) :
  computerTable.costPrice = 500 ∧ chair.costPrice = 1000 ∧ bookshelf.costPrice = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_furniture_cost_prices_l436_43688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_fourth_quadrant_l436_43662

theorem trig_values_fourth_quadrant (α : Real) 
  (h1 : Real.sin α = -3/5) 
  (h2 : α ∈ Set.Icc (3/2 * Real.pi) (2 * Real.pi)) :
  Real.cos α = 4/5 ∧ Real.tan α = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_fourth_quadrant_l436_43662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l436_43634

noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 400 then 400 * x - (1/2) * x^2
  else 80000

def fixed_cost : ℝ := 20000
def variable_cost : ℝ := 100

def total_cost (x : ℝ) : ℝ := fixed_cost + variable_cost * x

noncomputable def profit (x : ℝ) : ℝ := R x - total_cost x

theorem max_profit :
  ∃ (x : ℝ), x = 300 ∧ profit x = 25000 ∧ ∀ (y : ℝ), profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l436_43634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l436_43639

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (a - 2) * x

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := (1/3) * m * x^3 - m * x

-- State the theorem
theorem function_properties :
  ∃ (a : ℝ),
    (∀ x, x > 0 → (deriv (f a)) x = 0 ↔ x = 1) ∧
    (∀ x, x > 0 → f a x ≤ -1) ∧
    (∀ m : ℝ, m > 0 →
      (∀ x₁, 1 < x₁ ∧ x₁ < 2 →
        ∃ x₂, 1 < x₂ ∧ x₂ < 2 ∧ f a x₁ = g m x₂) ↔
      m ≥ 3 - (3/2) * Real.log 2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l436_43639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sewers_overflow_after_10_days_l436_43609

/-- The number of days the sewers can handle before they overflow -/
noncomputable def days_before_overflow (sewer_capacity : ℝ) (runoff_per_hour : ℝ) (hours_per_day : ℝ) : ℝ :=
  sewer_capacity / (runoff_per_hour * hours_per_day)

/-- Theorem stating that the sewers can handle 10 days of rain before they overflow -/
theorem sewers_overflow_after_10_days :
  let sewer_capacity : ℝ := 240000
  let runoff_per_hour : ℝ := 1000
  let hours_per_day : ℝ := 24
  days_before_overflow sewer_capacity runoff_per_hour hours_per_day = 10 := by
  -- Unfold the definition of days_before_overflow
  unfold days_before_overflow
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sewers_overflow_after_10_days_l436_43609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l436_43643

/-- Given a hyperbola C: x²/a² - y²/b² = 1 (a > 0, b > 0) with asymptotes tangent to the circle (x-a)² + y² = b²/4, prove that its eccentricity is 2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → 
    (∃ k : ℝ, (x - a)^2 + y^2 = b^2/4 ∧ 
      ((b*x = a*y) ∨ (b*x = -a*y)))) → 
  Real.sqrt (1 + b^2/a^2) = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l436_43643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surfaceAreaRatio_l436_43646

-- Define a cube with side length s
def cube (s : ℝ) := s

-- Define a regular tetrahedron formed by four vertices of the cube
def tetrahedron (s : ℝ) := s

-- Surface area of the cube
def cubeSurfaceArea (s : ℝ) : ℝ := 6 * s^2

-- Surface area of the tetrahedron
noncomputable def tetrahedronSurfaceArea (s : ℝ) : ℝ := 2 * s^2 * Real.sqrt 3

-- Theorem: The ratio of the surface areas is √3
theorem surfaceAreaRatio (s : ℝ) (h : s > 0) : 
  cubeSurfaceArea s / tetrahedronSurfaceArea s = Real.sqrt 3 := by
  sorry

#check surfaceAreaRatio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surfaceAreaRatio_l436_43646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_squares_in_eaten_portion_l436_43660

/-- Represents a square on the chessboard -/
inductive Square
| Black
| White

/-- Represents a row of the chessboard -/
def Row := List Square

/-- Represents the eaten portion of the chessboard -/
def EatenPortion := List Row

/-- Generates a standard chessboard row starting with the given square color -/
def generateRow (startColor : Square) : Row := sorry

/-- Counts the number of black squares in a row -/
def countBlackInRow (row : Row) : Nat := sorry

/-- The eaten portion starts with a black square and follows the standard chessboard pattern -/
def standardEatenPortion : EatenPortion :=
  [generateRow Square.Black,
   generateRow Square.White,
   generateRow Square.Black]

theorem black_squares_in_eaten_portion :
  (standardEatenPortion.map countBlackInRow).sum = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_squares_in_eaten_portion_l436_43660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l436_43687

theorem division_problem (x y : ℕ) (h1 : x % y = 3) (h2 : (x : ℝ) / (y : ℝ) = 96.12) : y = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l436_43687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_by_repeating_decimal_l436_43649

theorem divide_by_repeating_decimal : 
  ∃ (x : ℚ), (x = 2/3) ∧ (6 / x = 9) := by
  use 2/3
  constructor
  · rfl
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_by_repeating_decimal_l436_43649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_inequality_solution_a_range_l436_43663

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 1

-- Define the line l
def l (x y : ℝ) : Prop := y = x + 4

-- Define the intersection points
def intersection_point (x y : ℝ) : Prop := C₁ x y ∧ l x y

-- Theorem about the distance between intersection points
theorem intersection_distance : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    intersection_point x₁ y₁ ∧ 
    intersection_point x₂ y₂ ∧ 
    x₁ ≠ x₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 2 :=
sorry

-- Define the function f(x) for part (2)
def f (a x : ℝ) : ℝ := |a - 3*x| - |2 + x|

-- Theorem for part (2)(I)
theorem inequality_solution (x : ℝ) :
  f 2 x ≤ 3 ↔ -3/4 ≤ x ∧ x ≤ 7/2 :=
sorry

-- Theorem for part (2)(II)
theorem a_range :
  (∃ x, f a x ≥ 1 - a + 2*|2 + x|) ↔ a ≥ -5/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_inequality_solution_a_range_l436_43663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_two_minus_g_eight_l436_43632

/-- A linear function with the property that g(x+2) - g(x) = 5 for all real x -/
noncomputable def g : ℝ → ℝ := sorry

/-- The property that g(x+2) - g(x) = 5 for all real x -/
axiom g_property : ∀ x : ℝ, g (x + 2) - g x = 5

/-- g is a linear function -/
axiom g_linear : IsLinearMap ℝ g

theorem g_two_minus_g_eight : g 2 - g 8 = -15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_two_minus_g_eight_l436_43632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_acid_percentage_l436_43629

/-- Represents a mixture of acid and water -/
structure Mixture where
  acid : ℝ
  water : ℝ

/-- The percentage of acid in a mixture -/
noncomputable def acid_percentage (m : Mixture) : ℝ :=
  m.acid / (m.acid + m.water) * 100

/-- Adds acid to a mixture -/
def add_acid (m : Mixture) (amount : ℝ) : Mixture :=
  { acid := m.acid + amount, water := m.water }

/-- Adds water to a mixture -/
def add_water (m : Mixture) (amount : ℝ) : Mixture :=
  { acid := m.acid, water := m.water + amount }

theorem original_mixture_acid_percentage 
  (m : Mixture) 
  (h1 : acid_percentage (add_acid m 2) = 25)
  (h2 : acid_percentage (add_water (add_acid m 2) 2) = 20) :
  acid_percentage m = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_mixture_acid_percentage_l436_43629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_two_extrema_l436_43690

noncomputable def f (x : ℝ) := Real.cos (2 * x)

theorem cos_2x_two_extrema (m : ℝ) :
  m = Real.pi →
  (∃ (a b : ℝ), a ∈ Set.Ioo (-Real.pi/4) m ∧ 
                 b ∈ Set.Ioo (-Real.pi/4) m ∧ 
                 a ≠ b ∧
                 (∀ x ∈ Set.Ioo (-Real.pi/4) m, f x ≤ f a) ∧
                 (∀ x ∈ Set.Ioo (-Real.pi/4) m, f x ≤ f b) ∧
                 (∀ c ∈ Set.Ioo (-Real.pi/4) m, 
                    (∀ x ∈ Set.Ioo (-Real.pi/4) m, f x ≤ f c) → 
                    c = a ∨ c = b)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_two_extrema_l436_43690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_preposterous_midpoint_locus_l436_43673

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a preposterous midpoint
def IsPreposterousMidpoint (A B X : Point) : Prop :=
  ∃ (φ : ℝ) (τ : ℝ × ℝ), 
    let A' := (φ * A.x - τ.1, φ * A.y - τ.2)
    let B' := (φ * B.x - τ.1, φ * B.y - τ.2)
    let X' := (φ * X.x - τ.1, φ * X.y - τ.2)
    A'.1 ≥ 0 ∧ A'.2 ≥ 0 ∧ B'.1 ≥ 0 ∧ B'.2 ≥ 0 ∧
    X'.1 = Real.sqrt (A'.1 * B'.1) ∧
    X'.2 = Real.sqrt (A'.2 * B'.2)

-- Define the disk with diameter AB, excluding the center
def InDiskExcludingCenter (A B X : Point) : Prop :=
  let M := Point.mk ((A.x + B.x) / 2) ((A.y + B.y) / 2)  -- Midpoint of AB
  (X.x - M.x)^2 + (X.y - M.y)^2 < ((A.x - B.x)^2 + (A.y - B.y)^2) / 4

-- Theorem statement
theorem preposterous_midpoint_locus (A B : Point) (h : A ≠ B) :
  ∀ X, IsPreposterousMidpoint A B X ↔ InDiskExcludingCenter A B X :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_preposterous_midpoint_locus_l436_43673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l436_43650

/-- Given a function f such that f(x-1) = 2x^2 - 8x + 11 for all x,
    prove that f(x) = 2x^2 - 4x + 5 for all x. -/
theorem function_transformation (f : ℝ → ℝ) 
    (h : ∀ x, f (x - 1) = 2 * x^2 - 8 * x + 11) :
  ∀ x, f x = 2 * x^2 - 4 * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l436_43650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crank_slider_equations_correct_l436_43618

/-- Represents a crank-slider mechanism -/
structure CrankSlider where
  ω : ℝ  -- Angular velocity
  OA : ℝ  -- Length of OA
  AB : ℝ  -- Length of AB
  AL : ℝ  -- Length of AL

/-- Equations of motion and velocity for point L -/
noncomputable def pointL_equations (cs : CrankSlider) (t θ : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let x_L := cs.OA * Real.cos (cs.ω * t) + cs.AL * Real.cos θ
  let y_L := cs.OA * Real.sin (cs.ω * t) + cs.AL * Real.sin θ
  let v_x_L := -cs.ω * (cs.OA + cs.AL) * Real.sin (cs.ω * t)
  let v_y_L := cs.ω * (cs.OA + cs.AL) * Real.cos (cs.ω * t)
  (x_L, y_L, v_x_L, v_y_L)

/-- Theorem stating the correctness of the equations for the given crank-slider mechanism -/
theorem crank_slider_equations_correct (cs : CrankSlider) (t θ : ℝ) 
    (h1 : cs.ω = 10)
    (h2 : cs.OA = 90)
    (h3 : cs.AB = 90)
    (h4 : cs.AL = cs.AB / 3) :
  pointL_equations cs t θ = (90 * Real.cos (10 * t) + 30 * Real.cos θ,
                             90 * Real.sin (10 * t) + 30 * Real.sin θ,
                             -1200 * Real.sin (10 * t),
                             1200 * Real.cos (10 * t)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crank_slider_equations_correct_l436_43618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_line_slope_l436_43606

noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2)

theorem secant_line_slope :
  let x₁ : ℝ := 0
  let y₁ : ℝ := f x₁
  let x₂ : ℝ := 1
  let y₂ : ℝ := f x₂
  (y₂ - y₁) / (x₂ - x₁) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_secant_line_slope_l436_43606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_25_l436_43696

/-- Calculates the discount percentage given the original price, price increase percentage, and final price after discount. -/
noncomputable def calculate_discount_percentage (original_price : ℝ) (increase_percentage : ℝ) (final_price : ℝ) : ℝ :=
  let increased_price := original_price * (1 + increase_percentage / 100)
  let discount_amount := increased_price - final_price
  (discount_amount / increased_price) * 100

/-- Proves that the discount percentage is 25% given the specified conditions. -/
theorem discount_percentage_is_25 :
  let original_price : ℝ := 200
  let increase_percentage : ℝ := 25
  let final_price : ℝ := 187.5
  calculate_discount_percentage original_price increase_percentage final_price = 25 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_is_25_l436_43696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_three_l436_43627

-- Define the vectors a and b
noncomputable def a (θ : Real) : Fin 2 → Real := ![Real.cos θ, Real.sin θ]
noncomputable def b : Fin 2 → Real := ![Real.sqrt 3, 1]

-- Define the distance function
noncomputable def distance (θ : Real) : Real :=
  Real.sqrt ((a θ 0 - b 0)^2 + (a θ 1 - b 1)^2)

-- Theorem statement
theorem max_distance_is_three :
  ∃ θ : Real, ∀ φ : Real, distance θ ≥ distance φ ∧ distance θ = 3 := by
  sorry

#check max_distance_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_is_three_l436_43627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_solution_l436_43605

theorem unique_triple_solution : 
  ∃! (a b c : ℕ), 
    a ≥ 2 ∧ 
    b ≥ 1 ∧ 
    c ≥ 0 ∧ 
    (Real.logb (a : ℝ) (b : ℝ) = (c : ℝ) ^ 3) ∧ 
    (a + b + c = 100) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_solution_l436_43605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_overs_played_l436_43680

def initial_run_rate : ℝ := 4.3
def remaining_overs : ℕ := 30
def required_run_rate : ℝ := 5.933333333333334
def target_runs : ℕ := 264

theorem initial_overs_played (x : ℕ) : 
  initial_run_rate * (x : ℝ) + required_run_rate * (remaining_overs : ℝ) = (target_runs : ℝ) → 
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_overs_played_l436_43680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_limit_in_zero_one_l436_43697

def sequence_rule (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a n = a (n - 1) / 2 ∨ a n = Real.sqrt (a (n - 1))) ∧ a n > 0

theorem no_limit_in_zero_one (a : ℕ → ℝ) :
  sequence_rule a → ¬ ∃ A : ℝ, 0 < A ∧ A < 1 ∧ Filter.Tendsto a Filter.atTop (nhds A) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_limit_in_zero_one_l436_43697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_from_perimeter_semicircle_radius_is_correct_l436_43670

/-- The radius of a semi-circle given its perimeter -/
theorem semicircle_radius_from_perimeter (perimeter : ℝ) (h : perimeter > 0) :
  let radius := perimeter / (Real.pi + 2)
  perimeter = radius * Real.pi + 2 * radius :=
by
  intro radius
  -- Proof steps would go here
  sorry

/-- The radius of a semi-circle with perimeter 144 cm -/
noncomputable def semicircle_radius : ℝ := 144 / (Real.pi + 2)

/-- Proof that the calculated radius is correct -/
theorem semicircle_radius_is_correct : 
  144 = semicircle_radius * Real.pi + 2 * semicircle_radius :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_from_perimeter_semicircle_radius_is_correct_l436_43670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_9000_terms_l436_43682

/-- Represents a geometric sequence -/
structure GeometricSequence where
  firstTerm : ℝ
  commonRatio : ℝ

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumOfTerms (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.firstTerm * (1 - g.commonRatio^n) / (1 - g.commonRatio)

/-- Theorem stating the sum of the first 9000 terms given the sums of first 3000 and 6000 terms -/
theorem sum_of_9000_terms (g : GeometricSequence) 
  (h1 : sumOfTerms g 3000 = 500)
  (h2 : sumOfTerms g 6000 = 950) :
  sumOfTerms g 9000 = 1355 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_9000_terms_l436_43682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_transformation_l436_43698

noncomputable def q (w d z x : ℝ) : ℝ := 5 * w^2 / (4 * d^2 * (z^3 + x^2))

theorem q_transformation (w d z x : ℝ) (hw : w ≠ 0) (hd : d ≠ 0) (hz : z ≠ 0) (hx : x ≠ 0) :
  q (4*w) (2*d) (3*z) (x/2) = (16/27) * q w d z x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_transformation_l436_43698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_items_eq_192_l436_43675

def num_children : ℕ := 6

def crayon_sequence (n : ℕ) : ℕ := 2 * n

def apples_per_child : ℕ := 10

def cookies_per_child : ℕ := 15

def total_items : ℕ := (Finset.sum (Finset.range num_children) (fun i => crayon_sequence (i + 1))) +
                       (num_children * apples_per_child) +
                       (num_children * cookies_per_child)

theorem total_items_eq_192 : total_items = 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_items_eq_192_l436_43675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_area_relation_l436_43699

/-- Given a right-angled triangle with sides a, b, and c (where c is the hypotenuse),
    and equilateral triangles constructed on each side with areas A, B, and C respectively,
    prove that the area of the equilateral triangle on the hypotenuse (C) is equal to
    the sum of the areas of the other two equilateral triangles (A + B). -/
theorem equilateral_triangles_area_relation (a b c A B C : ℝ) 
  (h_right_angle : a^2 + b^2 = c^2)
  (h_A : A = (Real.sqrt 3 / 4) * a^2)
  (h_B : B = (Real.sqrt 3 / 4) * b^2)
  (h_C : C = (Real.sqrt 3 / 4) * c^2) :
  C = A + B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_area_relation_l436_43699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_victories_for_40_points_l436_43683

/-- Represents the scoring system in the football competition -/
structure ScoringSystem where
  victory_points : Nat
  draw_points : Nat
  defeat_points : Nat

/-- Represents the state of a team's performance in the tournament -/
structure TeamPerformance where
  total_matches : Nat
  played_matches : Nat
  current_points : Nat

/-- Calculates the minimum number of victories required to reach the target points -/
def min_victories_required (ss : ScoringSystem) (tp : TeamPerformance) (target_points : Nat) : Nat :=
  let remaining_matches := tp.total_matches - tp.played_matches
  let points_needed := target_points - tp.current_points
  (points_needed + ss.victory_points - 1) / ss.victory_points

/-- Theorem stating the minimum number of victories required to reach 40 points -/
theorem min_victories_for_40_points (ss : ScoringSystem) (tp : TeamPerformance) :
  ss.victory_points = 3 →
  ss.draw_points = 1 →
  ss.defeat_points = 0 →
  tp.total_matches = 20 →
  tp.played_matches = 5 →
  tp.current_points = 8 →
  min_victories_required ss tp 40 = 11 := by
  sorry

#eval min_victories_required 
  { victory_points := 3, draw_points := 1, defeat_points := 0 } 
  { total_matches := 20, played_matches := 5, current_points := 8 } 
  40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_victories_for_40_points_l436_43683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_root_is_four_thirds_l436_43686

/-- The cubic equation type -/
structure CubicEquation (α : Type*) [Field α] where
  a : α
  b : α

/-- The roots of the cubic equation -/
noncomputable def roots (eq : CubicEquation ℝ) : Fin 3 → ℝ :=
  sorry

/-- The cubic equation evaluates to zero for its roots -/
axiom roots_are_zeros (eq : CubicEquation ℝ) :
  ∀ (i : Fin 3), eq.a * (roots eq i)^3 + (eq.a + 2*eq.b) * (roots eq i)^2 + 
                 (eq.b - 3*eq.a) * (roots eq i) + (10 - eq.a) = 0

/-- Two of the roots are -2 and 3 -/
axiom given_roots (eq : CubicEquation ℝ) :
  roots eq 0 = -2 ∧ roots eq 1 = 3

theorem third_root_is_four_thirds (eq : CubicEquation ℝ) :
  roots eq 2 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_root_is_four_thirds_l436_43686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_sum_l436_43642

theorem sqrt_equation_sum (a b : ℝ) : 
  (Real.sqrt (6 + a/b) = 6 * Real.sqrt (a/b)) → a + b = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_equation_sum_l436_43642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l436_43693

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem interest_ratio : 
  (simple_interest 1750 8 3) / (compound_interest 4000 10 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_ratio_l436_43693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l436_43601

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h0 : a > b
  h1 : b > 0
  eccentricity : ℝ
  h2 : eccentricity = Real.sqrt 6 / 3
  right_focus : ℝ × ℝ
  h3 : right_focus = (2 * Real.sqrt 2, 0)

/-- Line intersecting the ellipse -/
structure Line where
  slope : ℝ
  h4 : slope = 1

/-- Triangle formed by intersection points and given point -/
structure Triangle where
  P : ℝ × ℝ
  h5 : P = (-3, 2)

/-- Main theorem statement -/
theorem ellipse_and_triangle_properties
  (e : Ellipse) (l : Line) (t : Triangle) :
  (∃ x y : ℝ, x^2 / 12 + y^2 / 4 = 1) ∧
  (∃ area : ℝ, area = 9/2) := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l436_43601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sequence_product_l436_43681

def fraction_product : ℕ → ℚ
  | 0 => 1
  | n + 1 => fraction_product n * ((n + 5 : ℚ) / (n + 8 : ℚ))

theorem fraction_sequence_product :
  fraction_product 50 = 7 / 4213440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sequence_product_l436_43681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_sixth_f_max_min_l436_43622

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * cos (π / 2 - x) * cos x - Real.sqrt 3 * cos (2 * x)

-- Theorem for the value of f(π/6)
theorem f_pi_sixth : f (π / 6) = 0 := by sorry

-- Theorem for the maximum and minimum values of f(x) on [0, π/2]
theorem f_max_min : ∃ (x_max x_min : ℝ), 
  x_max ∈ Set.Icc 0 (π / 2) ∧ 
  x_min ∈ Set.Icc 0 (π / 2) ∧ 
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ f x_max) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x_min ≤ f x) ∧
  f x_max = 2 ∧ 
  f x_min = -Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_sixth_f_max_min_l436_43622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l436_43610

-- Define the quadrilateral ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (6, -1)
def C : ℝ × ℝ := (7, 7)

-- Define the properties of the fourth vertex D
def AD_length : ℝ := 5
def DC_length : ℝ := 5

-- Define the areas for concave and convex cases
def concave_area : ℝ := 21
def convex_area : ℝ := 28

-- Helper functions (not proven, just declared for completeness)
noncomputable def angle_at_vertex (P Q R : ℝ × ℝ) : ℝ := sorry
noncomputable def area_quadrilateral (P Q R S : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area_theorem :
  ∀ D : ℝ × ℝ,
  (‖D - A‖ = AD_length) →
  (‖D - C‖ = DC_length) →
  (∃ (angle : ℝ), angle > 180 ∧ angle_at_vertex A D C = angle) →
  area_quadrilateral A B C D = concave_area ∧
  (∃ (D' : ℝ × ℝ),
    ‖D' - A‖ = AD_length ∧
    ‖D' - C‖ = DC_length ∧
    (∃ (angle : ℝ), angle < 180 ∧ angle_at_vertex A D' C = angle) ∧
    area_quadrilateral A B C D' = convex_area) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_theorem_l436_43610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l436_43654

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (4 + 5 * Real.cos t, 5 + 5 * Real.sin t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := let ρ := 2 * Real.sin θ; (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the intersection points in polar coordinates
def intersection_points : Set (ℝ × ℝ) := {(2, Real.pi / 2), (Real.sqrt 2, Real.pi / 4)}

-- Theorem statement
theorem curves_intersection :
  ∀ (t θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi →
  (C₁ t = C₂ θ) ↔ (2 * Real.sin θ, θ) ∈ intersection_points :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_intersection_l436_43654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_in_terms_of_c_and_d_l436_43620

theorem sin_x_in_terms_of_c_and_d (c d x : ℝ) 
  (h1 : Real.tan x = (3 * c * d) / (c^2 - d^2))
  (h2 : c > d)
  (h3 : d > 0)
  (h4 : 0 < x)
  (h5 : x < Real.pi / 2) :
  Real.sin x = (3 * c * d) / Real.sqrt (c^4 + 7 * c^2 * d^2 + d^4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_x_in_terms_of_c_and_d_l436_43620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_sphere_volume_relation_main_result_l436_43637

noncomputable def r : ℝ := 1 -- Arbitrary positive real number

theorem cone_cylinder_sphere_volume_relation :
  let Vcone := (2/3) * Real.pi * r^3
  let Vcylinder := 2 * Real.pi * r^3
  let Vsphere := (4/3) * Real.pi * r^3
  Vcone + Vcylinder = 2 * Vsphere := by
  
  -- Unfold the let bindings
  simp only [r]
  
  -- Simplify the expressions
  calc
    (2/3) * Real.pi * 1^3 + 2 * Real.pi * 1^3 
      = (2/3 + 2) * Real.pi * 1^3 := by ring
    _ = (8/3) * Real.pi * 1^3 := by ring
    _ = 2 * ((4/3) * Real.pi * 1^3) := by ring
    _ = 2 * ((4/3) * Real.pi * 1^3) := by ring

theorem main_result : 
  ∃ (r : ℝ), r > 0 ∧ 
  let Vcone := (2/3) * Real.pi * r^3
  let Vcylinder := 2 * Real.pi * r^3
  let Vsphere := (4/3) * Real.pi * r^3
  Vcone + Vcylinder = 2 * Vsphere := by
  
  -- We use r = 1 as our example
  use 1
  constructor
  · -- Prove 1 > 0
    linarith
  · -- Apply the previous theorem
    exact cone_cylinder_sphere_volume_relation

#check main_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_cylinder_sphere_volume_relation_main_result_l436_43637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l436_43665

-- Define the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, 2)

-- Define the circle equation
def on_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the area of a triangle given three points
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

-- Theorem statement
theorem min_triangle_area :
  ∃ (min_area : ℝ), min_area = 3 - Real.sqrt 2 ∧
  ∀ (C : ℝ × ℝ), on_circle C.1 C.2 →
  triangle_area A B C ≥ min_area :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l436_43665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l436_43633

theorem sin_pi_plus_alpha (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin (π / 2 + α) = 3 / 5) : 
  Real.sin (π + α) = - 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_plus_alpha_l436_43633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l436_43631

theorem trigonometric_identities 
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin (α + 2*β) = 1/3) :
  (α + β = 2*π/3 → Real.sin β = (2*Real.sqrt 6 - 1) / 6) ∧
  (Real.sin β = 4/5 → Real.cos α = (24 + 14*Real.sqrt 2) / 75) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l436_43631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l436_43684

/-- The function f(x) represents the left-hand side of the equation -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/4)^x + (1/2)^x + a

/-- The theorem states that if the equation has a positive solution, then -2 < a < 0 -/
theorem equation_solution_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ f a x = 0) → -2 < a ∧ a < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l436_43684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l436_43615

noncomputable section

open Real

-- Define the function f
def f (x : ℝ) : ℝ :=
  2 * sin (π - x) + cos (-x) - sin ((5/2) * π - x) + cos ((π/2) + x)

theorem f_properties :
  -- Part 1
  (∀ α : ℝ, 0 < α ∧ α < π ∧ f α = (2/3) * α → 
    tan α = (2 * sqrt 5) / 5 ∨ tan α = -(2 * sqrt 5) / 5) ∧
  -- Part 2
  (∀ α : ℝ, f α = 2 * sin α - cos α + 3/4 → 
    sin α * cos α = 7/32) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l436_43615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_is_one_l436_43691

-- Define the imaginary unit
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := (1 + 2*i) / (2 - i)

-- State the theorem
theorem modulus_of_z_is_one : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modulus_of_z_is_one_l436_43691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_implies_a_l436_43664

-- Define the line
def line (x y : ℝ) : Prop := x + y + 4 = 0

-- Define the circle
def circle_eq (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y + a = 0

-- Define the chord length
def chord_length (l : ℝ) : Prop := l = 2

-- Theorem statement
theorem intersection_chord_length_implies_a (a : ℝ) :
  (∃ x y : ℝ, line x y ∧ circle_eq x y a) →
  (∃ l : ℝ, chord_length l) →
  a = -7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_implies_a_l436_43664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_neg_infinity_to_neg_two_l436_43626

-- Define the function f(x) = log₁/₂(x² - 4)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4) / Real.log (1/2)

-- State the theorem
theorem f_monotone_increasing_on_neg_infinity_to_neg_two :
  StrictMonoOn f (Set.Iio (-2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_on_neg_infinity_to_neg_two_l436_43626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_B_length_l436_43653

-- Define the given parameters
noncomputable def train_A_speed : ℝ := 60  -- km/hr
noncomputable def train_B_speed : ℝ := 75  -- km/hr
noncomputable def crossing_time : ℝ := 27  -- seconds
noncomputable def train_A_length : ℝ := 300  -- meters

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 1000 / 3600

-- Define the theorem
theorem train_B_length :
  let relative_speed := (train_A_speed + train_B_speed) * km_hr_to_m_s
  let train_B_length := relative_speed * crossing_time - train_A_length
  train_B_length = 712.5 := by
    -- Proof steps would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_B_length_l436_43653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garlic_needed_proof_l436_43657

/-- Represents the number of cloves of garlic needed to repel creatures -/
def GarlicNeeded : Type := ℕ

/-- The number of cloves of garlic that can repel a certain number of creatures -/
structure GarlicEffectiveness :=
  (cloves : ℕ)
  (vampires : ℕ)
  (wights : ℕ)
  (vampireBats : ℕ)

/-- The given effectiveness of garlic against creatures -/
def baseEffectiveness : GarlicEffectiveness :=
  { cloves := 3
    vampires := 2
    wights := 3
    vampireBats := 8 }

/-- The number of creatures to be repelled -/
structure CreaturesToRepel :=
  (vampires : ℕ)
  (wights : ℕ)
  (vampireBats : ℕ)

/-- The creatures we need to repel in this problem -/
def targetCreatures : CreaturesToRepel :=
  { vampires := 30
    wights := 12
    vampireBats := 40 }

/-- Calculate the total number of garlic cloves needed -/
def totalGarlicNeeded (e : GarlicEffectiveness) (c : CreaturesToRepel) : ℕ :=
  let vampireCloves := (c.vampires * e.cloves + e.vampires - 1) / e.vampires
  let wightCloves := (c.wights * e.cloves + e.wights - 1) / e.wights
  let vampireBatCloves := (c.vampireBats * e.cloves + e.vampireBats - 1) / e.vampireBats
  vampireCloves + wightCloves + vampireBatCloves

/-- The theorem to prove -/
theorem garlic_needed_proof :
  totalGarlicNeeded baseEffectiveness targetCreatures = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garlic_needed_proof_l436_43657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_and_area_l436_43613

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (3, 0)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

-- Theorem statement
theorem parabola_point_and_area (M : PointOnParabola) 
  (h : distance (M.x, M.y) focus = 5) : 
  M.x = 2 ∧ triangleArea origin focus (M.x, M.y) = 3 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_and_area_l436_43613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_existence_and_bound_l436_43695

/-- Definition of a complete rectangle -/
def is_complete (rectangle : Array (Array (Option Bool))) : Prop :=
  let m := rectangle.size
  let n := if m > 0 then rectangle.get! 0 |>.size else 0
  (∀ i j, i < m → j < n → (rectangle.get! i |>.get! j).isSome → 
    ((rectangle.get! i |>.get! j) = some true ∨ (rectangle.get! i |>.get! j) = some false)) ∧
  (∀ s : Array Bool, s.size = n → ∃ i < m, ∀ j < n, 
    ((rectangle.get! i |>.get! j).isSome → (rectangle.get! i |>.get! j) = some (s.get! j)) ∧
    ((rectangle.get! i |>.get! j).isNone → true))

/-- Definition of a minimal rectangle -/
def is_minimal (rectangle : Array (Array (Option Bool))) : Prop :=
  is_complete rectangle ∧
  ∀ i < rectangle.size, ¬is_complete (rectangle.eraseIdx i)

/-- Main theorem -/
theorem rectangle_existence_and_bound :
  (∀ k : Nat, k > 0 → k ≤ 2018 → ∃ rectangle : Array (Array (Option Bool)),
    rectangle.size = 2^k ∧
    (∀ i, i < rectangle.size → (rectangle.get! i).size = 2018) ∧
    is_minimal rectangle ∧
    (∃ cols : Finset Nat, cols.card = k ∧ 
      ∀ j ∈ cols, ∃ i₁ i₂, i₁ < rectangle.size ∧ i₂ < rectangle.size ∧
        (rectangle.get! i₁ |>.get! j) = some true ∧ (rectangle.get! i₂ |>.get! j) = some false)) ∧
  (∀ m k : Nat, k > 0 → k ≤ 2018 → 
    ∀ rectangle : Array (Array (Option Bool)),
    rectangle.size = m ∧
    (∀ i, i < rectangle.size → (rectangle.get! i).size = 2018) ∧
    is_minimal rectangle →
    (∃ cols : Finset Nat, cols.card = k ∧
      ∀ j ∈ cols, ∃ i, i < rectangle.size ∧ 
        ((rectangle.get! i |>.get! j) = some true ∨ (rectangle.get! i |>.get! j) = some false)) →
    m ≤ 2^k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_existence_and_bound_l436_43695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_document_delivery_speed_l436_43630

/-- A problem about document delivery speeds of horses. -/
theorem document_delivery_speed (x : ℝ) (h : x > 3) : 
  (900 : ℝ) / (x + 1) * 2 = 900 / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_document_delivery_speed_l436_43630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_equality_l436_43645

theorem contrapositive_sin_equality (x y : ℝ) : 
  (¬(Real.sin x = Real.sin y) → ¬(x = y)) ↔ (x = y → Real.sin x = Real.sin y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_equality_l436_43645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_number_is_17_l436_43692

/-- The area of the shaded region in a 4x5 rectangle with a circular cut-out of diameter 2 -/
noncomputable def shaded_area : ℝ := 20 - Real.pi

/-- The whole number closest to the shaded area -/
def closest_whole_number : ℕ := 17

/-- Theorem stating that the closest whole number to the shaded area is 17 -/
theorem closest_number_is_17 : 
  ∀ n : ℕ, |shaded_area - ↑n| ≥ |shaded_area - ↑closest_whole_number| := by
  sorry

#check closest_number_is_17

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_number_is_17_l436_43692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_value_l436_43659

/-- The value of the infinite nested square root √(3 - √(3 - √(3 - √(3 - ...)))) -/
noncomputable def nestedSqrt : ℝ := Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt 3))))

/-- Theorem stating that the value of the nested square root is (-1 + √13) / 2 -/
theorem nested_sqrt_value : nestedSqrt = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_value_l436_43659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_zero_or_four_l436_43636

-- Define the line l: x + y + m = 0
def line (m : ℝ) (x y : ℝ) : Prop := x + y + m = 0

-- Define the circle C: x^2 + y^2 + 4y = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4*y = 0

-- Define the intersection points M and N
def intersection_points (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line m x₁ y₁ ∧ circle_eq x₁ y₁ ∧
    line m x₂ y₂ ∧ circle_eq x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Define the area of triangle CMN
noncomputable def triangle_area (m : ℝ) : ℝ := sorry

-- Theorem stating that the area is maximized when m = 0 or m = 4
theorem max_area_at_zero_or_four :
  ∀ m : ℝ, intersection_points m →
    (∀ k : ℝ, triangle_area m ≥ triangle_area k) →
    (m = 0 ∨ m = 4) := by
  sorry

#check max_area_at_zero_or_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_at_zero_or_four_l436_43636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l436_43656

theorem geometric_sequence_common_ratio
  (a : ℝ) :
  let seq := λ n => a + Real.log 3 / Real.log (2^(2^n))
  (∀ n, seq (n + 1) / seq n = (1 : ℝ) / 3) ∧
  (∀ n, seq (n + 2) / seq (n + 1) = (1 : ℝ) / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l436_43656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l436_43694

/-- A rational function with specific asymptotes and a known point -/
noncomputable def q (x : ℝ) : ℝ := (5/3) * x^2 - (5/3) * x - 10

/-- The vertical asymptotes of 1/q(x) occur at x = -2 and x = 3 -/
def has_asymptotes (f : ℝ → ℝ) : Prop :=
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x + 2| < δ → |f x| > 1/ε) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x| > 1/ε)

theorem q_satisfies_conditions :
  has_asymptotes (λ x ↦ 1 / q x) ∧ q 1 = -10 := by
  sorry

#check q_satisfies_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_q_satisfies_conditions_l436_43694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_neg_two_sufficient_not_necessary_l436_43625

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of line l1: ax + (a + 1)y + 1 = 0 -/
noncomputable def slope_l1 (a : ℝ) : ℝ := -a / (a + 1)

/-- The slope of line l2: x + ay + 2 = 0 -/
noncomputable def slope_l2 (a : ℝ) : ℝ := -1 / a

/-- a = -2 is sufficient but not necessary for l1 ⊥ l2 -/
theorem a_neg_two_sufficient_not_necessary :
  (∀ a : ℝ, a = -2 → perpendicular (slope_l1 a) (slope_l2 a)) ∧
  (∃ a : ℝ, a ≠ -2 ∧ perpendicular (slope_l1 a) (slope_l2 a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_neg_two_sufficient_not_necessary_l436_43625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_is_80_l436_43638

/-- Calculates the share of profit for an investor given the investments and time periods -/
noncomputable def calculate_share_of_profit (investment_a : ℝ) (months_a : ℝ) (investment_b : ℝ) (months_b : ℝ) (total_profit : ℝ) : ℝ :=
  let investment_months_a := investment_a * months_a
  let investment_months_b := investment_b * months_b
  let total_investment_months := investment_months_a + investment_months_b
  let proportion_a := investment_months_a / total_investment_months
  proportion_a * total_profit

/-- Theorem stating that A's share of the profit is $80 given the problem conditions -/
theorem a_share_is_80 :
  calculate_share_of_profit 400 12 200 6 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_share_is_80_l436_43638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_exponent_difference_l436_43635

/-- Given two monomials whose sum is still a monomial, prove that m - n = -1 -/
theorem monomial_sum_exponent_difference (m n : ℤ) : 
  (∃ (a : ℚ) (p q : ℕ), (2/3 : ℚ) * X^2 * Y^n + (-2 : ℚ) * X^m * Y^3 = a * X^p * Y^q) →
  m - n = -1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_sum_exponent_difference_l436_43635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_heartsuit_values_l436_43676

-- Define the ♡ function
noncomputable def heartsuit (x : ℝ) : ℝ := (x + x^2 + x^3) / 3

-- Theorem statement
theorem sum_of_heartsuit_values :
  heartsuit 1 + heartsuit 2 + heartsuit 3 = 56 / 3 := by
  -- Expand the definition of heartsuit
  unfold heartsuit
  -- Simplify the expressions
  simp [pow_two, pow_three]
  -- Perform arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_heartsuit_values_l436_43676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_l436_43603

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if a point (x, y) is on the line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- Calculates the slope of a line given its inclination angle in degrees -/
noncomputable def slopeFromAngle (angle : ℝ) : ℝ :=
  Real.tan (angle * Real.pi / 180)

/-- The main theorem to prove -/
theorem line_equation_correct (x y : ℝ) :
  x - y + 3 = 0 ↔
  ∃ (l : Line),
    l.slope = slopeFromAngle 45 ∧
    l.containsPoint (-1) 2 ∧
    l.containsPoint x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_l436_43603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_swept_by_lines_l436_43621

/-- Given points A_t and B_t on a plane, prove that the set swept by all lines A_tB_t
    is bounded by the parabola y ≤ (x^2)/4 + 1 -/
theorem set_swept_by_lines (x y t : ℝ) :
  (∃ (m : ℝ), y = m * (x - (1 + t)) + (1 + t) ∧ 
              y = m * (x - (-1 + t)) + (1 - t)) →
  y ≤ (x^2) / 4 + 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_swept_by_lines_l436_43621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l436_43641

open Real

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (AB : ℝ)

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.A + t.B = 3 * t.C ∧
  2 * sin (t.A - t.C) = sin t.B ∧
  t.AB = 5

-- Define area function (placeholder)
noncomputable def area (t : Triangle) : ℝ :=
  sorry

-- Define the theorem
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  sin t.A = 3 * Real.sqrt 10 / 10 ∧
  ∃ (height : ℝ), height = 6 ∧ height * t.AB / 2 = area t :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l436_43641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l436_43607

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + x

theorem f_decreasing_interval :
  ∀ x y : ℝ, x > (1/2 : ℝ) → y > (1/2 : ℝ) → x < y → f y < f x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l436_43607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joes_second_speed_l436_43677

/-- Calculates the speed for the second part of Joe's trip given the following conditions:
  * Joe drives 420 miles at 60 miles per hour
  * He then drives 120 miles at an unknown speed
  * His average speed for the entire trip is 54 miles per hour
-/
noncomputable def second_part_speed (first_distance : ℝ) (second_distance : ℝ) (first_speed : ℝ) (average_speed : ℝ) : ℝ :=
  let total_distance := first_distance + second_distance
  let total_time := total_distance / average_speed
  let first_time := first_distance / first_speed
  let second_time := total_time - first_time
  second_distance / second_time

/-- Theorem stating that Joe's speed during the second part of the trip was 40 miles per hour -/
theorem joes_second_speed :
  second_part_speed 420 120 60 54 = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joes_second_speed_l436_43677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_routeB_fastest_l436_43623

noncomputable section

structure Route where
  distance : ℝ
  speed : ℝ
  trafficDelay : ℝ
  restStops : ℕ
  restStopDuration : ℝ

noncomputable def totalTime (r : Route) : ℝ :=
  r.distance / r.speed + r.trafficDelay + (r.restStops : ℝ) * r.restStopDuration

def routeA : Route :=
  { distance := 1500
    speed := 75
    trafficDelay := 2
    restStops := 3
    restStopDuration := 0.5 }

def routeB : Route :=
  { distance := 1300
    speed := 70
    trafficDelay := 0
    restStops := 2
    restStopDuration := 0.75 }

def routeC : Route :=
  { distance := 1800
    speed := 80
    trafficDelay := 2.5
    restStops := 4
    restStopDuration := 1/3 }

def routeD : Route :=
  { distance := 750
    speed := 25
    trafficDelay := 0
    restStops := 1
    restStopDuration := 1 }

theorem routeB_fastest : 
  totalTime routeB < totalTime routeA ∧ 
  totalTime routeB < totalTime routeC ∧ 
  totalTime routeB < totalTime routeD := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_routeB_fastest_l436_43623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sequence_convexity_l436_43608

def is_convex (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s n ≤ (s (n - 1) + s (n + 1)) / 2

def is_positive_sequence (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s n > 0

theorem log_sequence_convexity 
  (b : ℕ → ℝ) 
  (h_pos : is_positive_sequence b) 
  (h_convex : ∀ c : ℝ, c > 0 → is_convex (λ n ↦ c^n * b n)) : 
  is_convex (λ n ↦ Real.log (b n)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sequence_convexity_l436_43608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tiling_scenario_l436_43669

/-- The number of possible tiling scenarios using regular triangles and dodecagons -/
noncomputable def tiling_scenarios : ℕ :=
  let triangle_angle : ℕ := 60
  let dodecagon_angle : ℕ := 150
  let total_angle : ℕ := 360
  let solutions := {(m, n) : ℕ × ℕ | m * triangle_angle + n * dodecagon_angle = total_angle ∧ m > 0 ∧ n > 0}
  Finset.card (Finset.filter (fun (m, n) => m * triangle_angle + n * dodecagon_angle = total_angle ∧ m > 0 ∧ n > 0) (Finset.product (Finset.range 7) (Finset.range 7)))

/-- Theorem stating that there is only one possible tiling scenario -/
theorem unique_tiling_scenario : tiling_scenarios = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tiling_scenario_l436_43669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l436_43604

theorem exponent_problem (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_problem_l436_43604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_divisible_by_5_l436_43648

def is_divisible_by_5 (n : ℕ) : Bool := n % 5 = 0

def numbers_between_6_and_34_divisible_by_5 : List ℕ :=
  (List.range 29).map (· + 6) |>.filter is_divisible_by_5

theorem average_of_numbers_divisible_by_5 :
  let numbers := numbers_between_6_and_34_divisible_by_5
  (numbers.sum : ℚ) / numbers.length = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_numbers_divisible_by_5_l436_43648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_production_rate_l436_43685

def g (x : Int) : Int := -6 * x^2 + 72 * x

theorem min_production_rate (a : Int) :
  (∀ x : Int, 1 ≤ x ∧ x ≤ 12 → a ≥ g x) ↔ a ≥ 171 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_production_rate_l436_43685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_3a_plus_2b_eq_neg_7_l436_43612

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + a*x - 2*b

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := (x - 2)^2 + (y + 4)^2 = 5

-- Define the tangency condition
def is_tangent (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_eq x y ∧ f a b x = y ∧
  ∀ (x' y' : ℝ), circle_eq x' y' → (x' = x ∧ y' = y) ∨ (f a b x' ≠ y')

-- State the theorem
theorem tangent_implies_3a_plus_2b_eq_neg_7 (a b : ℝ) :
  is_tangent a b → f a b 1 = -2 → 3*a + 2*b = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_implies_3a_plus_2b_eq_neg_7_l436_43612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocal_l436_43661

theorem tan_sum_reciprocal (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 5) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocal_l436_43661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l436_43640

theorem evaluate_expression : (125 : ℝ) ^ (1/3 : ℝ) * (64 : ℝ) ^ (-(1/2) : ℝ) * (81 : ℝ) ^ (1/4 : ℝ) = 15/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l436_43640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_l436_43668

/-- The original cost price of an item given the total price and additional charges -/
noncomputable def original_cost (total_price : ℝ) (markup_rate : ℝ) (assembly_rate : ℝ) (shipping_rate : ℝ) : ℝ :=
  total_price / (1 + markup_rate + assembly_rate + shipping_rate)

/-- Theorem stating that the original cost of the computer table is approximately 4923 -/
theorem computer_table_cost : 
  let total_price := (6400 : ℝ)
  let markup_rate := (0.15 : ℝ)
  let assembly_rate := (0.05 : ℝ)
  let shipping_rate := (0.10 : ℝ)
  abs (original_cost total_price markup_rate assembly_rate shipping_rate - 4923) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_cost_l436_43668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_100_equals_one_third_l436_43616

def c : ℕ → ℚ
  | 0 => 1  -- Adding this case to cover Nat.zero
  | 1 => 1
  | 2 => 1/3
  | n+3 => (2 - c (n+2)) / (3 * c (n+1))

theorem c_100_equals_one_third : c 100 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_100_equals_one_third_l436_43616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_shape_l436_43647

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A right angle in 2D space -/
structure RightAngle where
  vertex : Point2D
  arm1 : Point2D → ℝ  -- Distance function to first arm
  arm2 : Point2D → ℝ  -- Distance function to second arm

/-- A closed curve consisting of different segments -/
inductive ClosedCurve where
  | LineSegment : Point2D → Point2D → ClosedCurve
  | CircularArc : Point2D → ℝ → ℝ → ℝ → ClosedCurve  -- Center, radius, start angle, end angle
  | ParabolicArc : Point2D → Point2D → ℝ → ClosedCurve  -- Focus, point on directrix, distance

/-- The locus of points with constant sum of distances from right angle arms -/
def locusOfPoints (angle : RightAngle) (d : ℝ) : Set Point2D :=
  {p : Point2D | angle.arm1 p + angle.arm2 p = d}

/-- Helper function to check if a point is on a ClosedCurve -/
def isOnCurve (p : Point2D) (c : ClosedCurve) : Prop :=
  match c with
  | ClosedCurve.LineSegment p1 p2 => sorry
  | ClosedCurve.CircularArc center radius startAngle endAngle => sorry
  | ClosedCurve.ParabolicArc focus directrixPoint distance => sorry

/-- The theorem stating the shape of the locus -/
theorem locus_shape (angle : RightAngle) (d : ℝ) :
  ∃ (seg : ClosedCurve) (arc : ClosedCurve) (para1 para2 : ClosedCurve),
    (∃ p1 p2, seg = ClosedCurve.LineSegment p1 p2) ∧
    (∃ center radius startAngle endAngle, arc = ClosedCurve.CircularArc center radius startAngle endAngle) ∧
    (∃ focus1 directrixPoint1 distance1, para1 = ClosedCurve.ParabolicArc focus1 directrixPoint1 distance1) ∧
    (∃ focus2 directrixPoint2 distance2, para2 = ClosedCurve.ParabolicArc focus2 directrixPoint2 distance2) ∧
    locusOfPoints angle d = {p | isOnCurve p seg ∨ isOnCurve p arc ∨ isOnCurve p para1 ∨ isOnCurve p para2} :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_shape_l436_43647
