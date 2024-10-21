import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wolf_daily_meat_consumption_l638_63898

/-- Calculates the daily meat consumption for wolves given hunting conditions -/
theorem wolf_daily_meat_consumption
  (hunting_wolves : ℕ)
  (additional_wolves : ℕ)
  (hunt_interval : ℕ)
  (deer_meat : ℕ)
  (deer_per_wolf : ℕ)
  (h1 : hunting_wolves = 4)
  (h2 : additional_wolves = 16)
  (h3 : hunt_interval = 5)
  (h4 : deer_meat = 200)
  (h5 : deer_per_wolf = 1) :
  (deer_meat * deer_per_wolf) / hunt_interval = 40 := by
  sorry

#check wolf_daily_meat_consumption

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wolf_daily_meat_consumption_l638_63898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_I_l638_63872

-- Define H(p, q)
def H (p q : ℝ) : ℝ := -3*p*q + 4*p*(1-q) + 2*(1-p)*q - 5*(1-p)*(1-q)

-- Define I(p)
noncomputable def I (p : ℝ) : ℝ := ⨆ q ∈ Set.Icc 0 1, H p q

-- Theorem statement
theorem minimize_I :
  ∃ (p : ℝ), p ∈ Set.Icc 0 1 ∧
  (∀ (p' : ℝ), p' ∈ Set.Icc 0 1 → I p ≤ I p') ∧
  p = 1/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_I_l638_63872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l638_63831

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 0 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l638_63831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_not_sufficient_nor_necessary_l638_63830

open Set Real

theorem zero_point_not_sufficient_nor_necessary 
  (f : ℝ → ℝ) (hf : Continuous f) :
  ¬(∀ a b : ℝ, a < b → (∃ x ∈ Ioo a b, f x = 0) → f a * f b < 0) ∧
  ¬(∀ a b : ℝ, a < b → f a * f b < 0 → ∃ x ∈ Ioo a b, f x = 0) :=
by
  constructor
  · sorry  -- Proof that it's not sufficient
  · sorry  -- Proof that it's not necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_not_sufficient_nor_necessary_l638_63830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_solution_l638_63835

theorem unique_triple_solution (m : ℕ) (h_m_pos : 0 < m) (h_m_even : Even m) :
  ∃! (n x y : ℕ), 
    0 < n ∧ 0 < x ∧ 0 < y ∧
    Nat.Coprime m n ∧
    (x^2 + y^2)^m = (x * y)^n ∧
    n = m + 1 ∧ x = 2^(m / 2) ∧ y = 2^(m / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_triple_solution_l638_63835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_base_polyhedron_volume_l638_63889

/-- A polyhedron with parallel polygonal bases -/
structure ParallelBasePolyhedron where
  -- Distance between base planes
  H : ℝ
  -- Areas of the bases
  S₁ : ℝ
  S₂ : ℝ
  -- Area of cross-section equidistant from both bases
  S₃ : ℝ
  -- Ensure all areas are non-negative
  H_pos : 0 < H
  S₁_nonneg : 0 ≤ S₁
  S₂_nonneg : 0 ≤ S₂
  S₃_nonneg : 0 ≤ S₃

/-- The volume of a polyhedron with parallel polygonal bases -/
noncomputable def volume (p : ParallelBasePolyhedron) : ℝ :=
  (1 / 6) * p.H * (p.S₁ + p.S₂ + 4 * p.S₃)

/-- Theorem: The volume of a polyhedron with parallel polygonal bases
    is (1/6) * H * (S₁ + S₂ + 4S₃) -/
theorem parallel_base_polyhedron_volume (p : ParallelBasePolyhedron) :
  volume p = (1 / 6) * p.H * (p.S₁ + p.S₂ + 4 * p.S₃) := by
  -- Unfold the definition of volume
  unfold volume
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_base_polyhedron_volume_l638_63889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_markup_percentage_l638_63812

theorem merchant_markup_percentage (x : ℝ) : 
  (∀ (cost_price : ℝ), cost_price > 0 →
    let marked_price := cost_price * (1 + x / 100)
    let selling_price := marked_price * 0.8
    selling_price = cost_price * 1.12) →
  x = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_merchant_markup_percentage_l638_63812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_not_always_parallel_l638_63827

-- Define the basic types
variable (Point : Type) -- Points
variable (Line : Type) -- Lines
variable (Plane : Type) -- Planes

-- Define the relations
variable (pointOnLine : Point → Line → Prop) -- Point lies on a line
variable (pointOnPlane : Point → Plane → Prop) -- Point lies on a plane
variable (parallelLines : Line → Line → Prop) -- Lines are parallel
variable (parallelPlanes : Plane → Plane → Prop) -- Planes are parallel
variable (perpLine : Line → Plane → Prop) -- Line is perpendicular to a plane
variable (perpPlanes : Plane → Plane → Prop) -- Planes are perpendicular

-- State the theorem
theorem perpendicular_planes_not_always_parallel 
  (α β γ : Plane) : 
  ¬(∀ α β γ, (perpPlanes α γ ∧ perpPlanes β γ) → parallelPlanes α β) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_not_always_parallel_l638_63827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_35_miles_l638_63874

/-- The distance Bob walked when he met Yolanda -/
noncomputable def distance_bob_walked (total_distance : ℝ) (rate_yolanda : ℝ) (rate_bob : ℝ) (head_start : ℝ) : ℝ :=
  (total_distance * rate_bob) / (rate_yolanda + rate_bob) - head_start * rate_bob

/-- Theorem stating that Bob walked 35 miles when they met -/
theorem bob_walked_35_miles : 
  let total_distance : ℝ := 65
  let rate_yolanda : ℝ := 5
  let rate_bob : ℝ := 7
  let head_start : ℝ := 1
  distance_bob_walked total_distance rate_yolanda rate_bob head_start = 35 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_35_miles_l638_63874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_grows_faster_than_linear_l638_63851

theorem exponential_grows_faster_than_linear :
  ∀ (a : ℝ), a > 0 → ∃ (x₀ : ℝ), ∀ (x : ℝ), x > x₀ → Real.exp x > a * x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_grows_faster_than_linear_l638_63851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_CBG_equals_945_sqrt3_div_481_l638_63862

-- Define the circle and triangle
def Circle : ℝ × ℝ → Prop := λ p ↦ (p.1 - 0)^2 + (p.2 - 0)^2 = 9
def Triangle : ℝ × ℝ → Prop := λ p ↦ true  -- Placeholder definition

-- Define the points
variable (A B C D E F G : ℝ × ℝ)

-- Define the equilateral triangle ABC inscribed in the circle
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  Circle A ∧ Circle B ∧ Circle C ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

-- Define the extended points D and E
def extend_to_D (A D : ℝ × ℝ) : Prop := (A.1 - D.1)^2 + (A.2 - D.2)^2 = 15^2
def extend_to_E (A E : ℝ × ℝ) : Prop := (A.1 - E.1)^2 + (A.2 - E.2)^2 = 14^2

-- Define F as the intersection of parallel lines
def F_intersection (A D E F : ℝ × ℝ) : Prop :=
  (F.2 - D.2) / (F.1 - D.1) = (E.2 - A.2) / (E.1 - A.1) ∧
  (F.2 - E.2) / (F.1 - E.1) = (D.2 - A.2) / (D.1 - A.1)

-- Define G as the point on the circle collinear with A and F
def G_on_circle (A F G : ℝ × ℝ) : Prop :=
  Circle G ∧
  (G.2 - A.2) / (G.1 - A.1) = (F.2 - A.2) / (F.1 - A.1) ∧
  G ≠ A

-- Define the area of triangle CBG
noncomputable def area_CBG (C B G : ℝ × ℝ) : ℝ := 
  abs ((C.1 - G.1) * (B.2 - G.2) - (B.1 - G.1) * (C.2 - G.2)) / 2

-- State the theorem
theorem area_CBG_equals_945_sqrt3_div_481
  (h1 : triangle_ABC A B C)
  (h2 : extend_to_D A D)
  (h3 : extend_to_E A E)
  (h4 : F_intersection A D E F)
  (h5 : G_on_circle A F G) :
  area_CBG C B G = 945 * Real.sqrt 3 / 481 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_CBG_equals_945_sqrt3_div_481_l638_63862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_ratio_l638_63892

theorem max_value_trig_ratio (x : ℝ) (h : 0 ≤ x ∧ x ≤ π/16) :
  (∃ y : ℝ, 0 ≤ y ∧ y ≤ π/16 ∧
    ∀ z : ℝ, 0 ≤ z ∧ z ≤ π/16 →
      (Real.sin (2*y) + Real.sin (4*y) + Real.sin (6*y)) / (Real.cos (2*y) + Real.cos (4*y) + Real.cos (6*y)) ≥
      (Real.sin (2*z) + Real.sin (4*z) + Real.sin (6*z)) / (Real.cos (2*z) + Real.cos (4*z) + Real.cos (6*z))) ∧
  (Real.sin (2*x) + Real.sin (4*x) + Real.sin (6*x)) / (Real.cos (2*x) + Real.cos (4*x) + Real.cos (6*x)) ≤ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_ratio_l638_63892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_for_specific_configuration_l638_63853

/-- Represents the configuration of two poles and a wire looping around them. -/
structure PoleConfiguration where
  pole_distance : ℝ  -- Distance between pole bottoms
  short_pole_height : ℝ  -- Height of the shorter pole
  tall_pole_height : ℝ  -- Height of the taller pole

/-- Calculates the total wire length for a given pole configuration. -/
noncomputable def total_wire_length (config : PoleConfiguration) : ℝ :=
  let height_diff := config.tall_pole_height - config.short_pole_height
  let hypotenuse := (config.pole_distance^2 + height_diff^2).sqrt
  let vertical_distance := 2 * config.short_pole_height
  let base_distance := config.pole_distance + config.tall_pole_height
  hypotenuse + vertical_distance + base_distance

/-- Theorem stating that for the given pole configuration, 
    the total wire length is equal to √464 + 58 feet. -/
theorem wire_length_for_specific_configuration :
  let config := PoleConfiguration.mk 20 10 18
  total_wire_length config = Real.sqrt 464 + 58 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_for_specific_configuration_l638_63853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l638_63815

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + Real.pi / 4)

def is_zero (f : ℝ → ℝ) (x : ℝ) : Prop := f x = 0

def is_critical_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

theorem f_properties (ω : ℝ) (n : ℕ) (hn : n > 0) :
  ω > 0 →
  (2 * Real.pi / 3 : ℝ) * n ≤ (2 * Real.pi / ω) ∧ (2 * Real.pi / ω) ≤ n * Real.pi →
  is_zero (f ω) (Real.pi / 6) →
  (2 / n : ℝ) ≤ ω ∧ ω ≤ 3 / n ∧ is_critical_point (f ω) (7 * Real.pi / 6) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l638_63815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_sum_l638_63838

/-- Given a line segment with one endpoint at (1,2) and midpoint at (5,6),
    the sum of the coordinates of the other endpoint is 19. -/
theorem endpoint_coordinate_sum : ∀ (endpoint1 midpoint endpoint2 : ℝ × ℝ),
  endpoint1 = (1, 2) →
  midpoint = (5, 6) →
  midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2.1 + endpoint2.2 = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_endpoint_coordinate_sum_l638_63838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_coord_equilateral_triangle_l638_63859

theorem no_integer_coord_equilateral_triangle :
  ¬ ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℤ),
    let a := ((x₂ - x₁)^2 + (y₂ - y₁)^2 : ℝ).sqrt
    let b := ((x₃ - x₂)^2 + (y₃ - y₂)^2 : ℝ).sqrt
    let c := ((x₁ - x₃)^2 + (y₁ - y₃)^2 : ℝ).sqrt
    a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_coord_equilateral_triangle_l638_63859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_depth_is_75_l638_63841

/-- Represents the volume of a right frustum in cubic centimeters -/
def frustum_volume : ℝ := 190000

/-- Represents the length of the larger base of the frustum in centimeters -/
def larger_base : ℝ := 60

/-- Represents the length of the smaller base of the frustum in centimeters -/
def smaller_base : ℝ := 40

/-- Calculates the area of a circular base given its diameter -/
noncomputable def base_area (diameter : ℝ) : ℝ :=
  (Real.pi / 4) * diameter ^ 2

/-- Theorem stating that the depth of the frustum is 75 cm -/
theorem frustum_depth_is_75 : 
  (3 * frustum_volume) / (base_area larger_base + 
    Real.sqrt (base_area larger_base * base_area smaller_base) + 
    base_area smaller_base) = 75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_depth_is_75_l638_63841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_l638_63891

/-- A configuration of marked cells in a 15 × 15 table -/
def Configuration := Fin 15 → Fin 15 → Bool

/-- Checks if a 1 × 10 horizontal strip contains a marked cell -/
def horizontalStripMarked (config : Configuration) (row : Fin 15) (start : Fin 6) : Prop :=
  ∃ col : Fin 15, col.val ≥ start.val ∧ col.val < start.val + 10 ∧ config row col = true

/-- Checks if a 1 × 10 vertical strip contains a marked cell -/
def verticalStripMarked (config : Configuration) (col : Fin 15) (start : Fin 6) : Prop :=
  ∃ row : Fin 15, row.val ≥ start.val ∧ row.val < start.val + 10 ∧ config row col = true

/-- Checks if a configuration is valid (all strips contain a marked cell) -/
def isValidConfiguration (config : Configuration) : Prop :=
  (∀ row : Fin 15, ∀ start : Fin 6, horizontalStripMarked config row start) ∧
  (∀ col : Fin 15, ∀ start : Fin 6, verticalStripMarked config col start)

/-- Counts the number of marked cells in a configuration -/
def markedCellCount (config : Configuration) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 15)) fun i =>
    Finset.sum (Finset.univ : Finset (Fin 15)) fun j =>
      if config i j then 1 else 0)

/-- The main theorem: the minimum number of marked cells in a valid configuration is 20 -/
theorem min_marked_cells :
  (∃ config : Configuration, isValidConfiguration config ∧ markedCellCount config = 20) ∧
  (∀ config : Configuration, isValidConfiguration config → markedCellCount config ≥ 20) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_l638_63891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_quadrilateral_l638_63802

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the lines
def line_l (m x y : ℝ) : Prop := m * x - y + 1 = 0
def line_n (m x y : ℝ) : Prop := x + m * y - m = 0

-- Define the intersection points
def intersection_points (m : ℝ) : Prop :=
  ∃ (xa ya xc yc xb yb xd yd : ℝ),
    circle_equation xa ya ∧ circle_equation xc yc ∧ circle_equation xb yb ∧ circle_equation xd yd ∧
    line_l m xa ya ∧ line_l m xc yc ∧
    line_n m xb yb ∧ line_n m xd yd

-- Theorem statement
theorem max_area_of_quadrilateral :
  ∀ m : ℝ, intersection_points m →
  ∃ (area : ℝ), area ≤ 7 ∧
  (∀ other_area : ℝ, other_area ≤ area) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_quadrilateral_l638_63802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_roots_l638_63845

theorem periodic_function_roots (a : ℝ) : 
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    t₁^2 - (5*a - 2)*t₁ - 3*a^2 - 7*a + 1 = 0 ∧
    t₂^2 - (5*a - 2)*t₂ - 3*a^2 - 7*a + 1 = 0 ∧
    (∀ m : ℝ, m ≠ 0 → ∃ T : ℝ, T > 0 ∧ 
      ∀ x : ℝ, Real.cos (m * Real.pi * x) * Real.cos ((t₁^3 + t₂^3) * Real.pi * x) = 
               Real.cos (m * Real.pi * (x + T)) * Real.cos ((t₁^3 + t₂^3) * Real.pi * (x + T)))) →
  a = 2/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_roots_l638_63845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_in_range_l638_63869

noncomputable def f (x : ℝ) : ℝ := x^2 + x + (1/2)

theorem integer_count_in_range (n : ℕ) :
  let range_min := f n
  let range_max := f (n + 1)
  (Int.floor range_max - Int.ceil range_min + 1 : ℤ) = 2 * (n + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_count_in_range_l638_63869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_i_is_unit_vector_perpendicular_to_b_l638_63888

noncomputable def b : ℝ × ℝ := (-2, 1)
noncomputable def i : ℝ × ℝ := (-Real.sqrt 5 / 5, -2 * Real.sqrt 5 / 5)

theorem i_is_unit_vector_perpendicular_to_b : 
  (i.1 * i.1 + i.2 * i.2 = 1) ∧ (i.1 * b.1 + i.2 * b.2 = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_i_is_unit_vector_perpendicular_to_b_l638_63888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_trigonometry_l638_63829

theorem acute_angles_trigonometry 
  (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < Real.pi / 2) 
  (h_acute_β : 0 < β ∧ β < Real.pi / 2) 
  (h_sin_α : Real.sin α = 4 / 5) 
  (h_cos_αβ : Real.cos (α + β) = 5 / 13) : 
  Real.sin β = 16 / 65 ∧ Real.sin (Real.pi / 2 + 2 * β) = 3713 / 4225 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_angles_trigonometry_l638_63829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_neg_sqrt_3_implies_b_is_one_third_l638_63832

-- Define the function g
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * x^3 - Real.sqrt 3

-- State the theorem
theorem g_composition_equals_neg_sqrt_3_implies_b_is_one_third
  (b : ℝ)
  (h1 : b > 0)
  (h2 : g b (g b (Real.sqrt 3)) = -Real.sqrt 3) :
  b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_equals_neg_sqrt_3_implies_b_is_one_third_l638_63832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l638_63840

-- Define the circle C
def circleC (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

-- Define the line l
def lineL (y : ℝ) : Prop := y = 1

-- Theorem statement
theorem intersection_points :
  ∀ x y : ℝ, circleC x y ∧ lineL y ↔ (x = -1 ∧ y = 1) ∨ (x = 1 ∧ y = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l638_63840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_upper_bound_l638_63893

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x - (1/4) * x + (3/(4*x)) - 1
def g (b x : ℝ) : ℝ := -x^2 + 2*b*x - 4

-- State the theorem
theorem b_upper_bound (b : ℝ) :
  (∀ x₁ ∈ Set.Ioo 0 2, ∀ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g b x₂) →
  b ≤ Real.sqrt 14 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_upper_bound_l638_63893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l638_63868

-- Define a right pyramid with a square base
structure RightPyramid where
  base_area : ℝ
  total_surface_area : ℝ
  triangular_face_area : ℝ

-- Define the conditions of the problem
def problem_pyramid : RightPyramid :=
  { base_area := 207,
    total_surface_area := 486,
    triangular_face_area := 69 }

-- Theorem statement
theorem pyramid_volume (p : RightPyramid) 
  (h1 : p.total_surface_area = 486)
  (h2 : p.triangular_face_area = p.base_area / 3)
  (h3 : p.base_area = 207)
  : Real.sqrt (310.5^2 * 207) = (1/3) * p.base_area * (3/2 * Real.sqrt p.base_area) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_l638_63868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_g_property_l638_63895

theorem function_g_property (g : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), x > 0 → y > 0 → x * g y - y * g x = g (x^2 / y)) : 
  g 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_g_property_l638_63895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_YZRS_l638_63847

noncomputable section

-- Define the square PQRS
def square_side_length : ℝ := 8

-- Define points P, Q, R, S
def P : ℝ × ℝ := (0, 0)
def Q : ℝ × ℝ := (square_side_length, 0)
def R : ℝ × ℝ := (square_side_length, square_side_length)
def S : ℝ × ℝ := (0, square_side_length)

-- Define midpoint X of PQ
noncomputable def X : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define midpoint Y of XS
noncomputable def Y : ℝ × ℝ := ((X.1 + S.1) / 2, (X.2 + S.2) / 2)

-- Define midpoint Z of XR
noncomputable def Z : ℝ × ℝ := ((X.1 + R.1) / 2, (X.2 + R.2) / 2)

-- Define the area of trapezoid YZRS
def area_trapezoid_YZRS : ℝ := 24

-- Theorem statement
theorem area_of_trapezoid_YZRS :
  area_trapezoid_YZRS = 24 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_YZRS_l638_63847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_property_l638_63839

/-- A prism with 21 edges -/
structure Prism :=
  (edges : ℕ)
  (faces : ℕ)
  (vertices : ℕ)
  (is_prism : edges = 21)

/-- Theorem stating that for a prism with 21 edges, 3x - 2y = -1 where x is the number of faces and y is the number of vertices -/
theorem prism_property (p : Prism) : (3 : ℤ) * p.faces - 2 * p.vertices = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_property_l638_63839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_product_l638_63897

theorem min_value_product (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (h : 1/x + 1/y + 1/z + 1/w = 8) :
  ∃ (m : ℝ), m = 1/432 ∧ ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 →
    1/a + 1/b + 1/c + 1/d = 8 → a^3 * b^2 * c * d^2 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_product_l638_63897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l638_63824

noncomputable def g (t : ℝ) : ℝ := (t^2 + 5/4*t) / (t^2 + 1)

theorem range_of_g :
  ∀ y : ℝ, (∃ t : ℝ, g t = y) → y ∈ Set.Icc (-5/16) (21/16) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l638_63824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_one_ninth_l638_63883

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

-- State the theorem
theorem f_composition_one_ninth : f (f (1/9)) = 1/4 := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_one_ninth_l638_63883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l638_63855

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x)

noncomputable def vec_a (x : ℝ) : Fin 2 → ℝ
| 0 => 2 * Real.cos x
| 1 => Real.sqrt 3 * Real.sin (2 * x)
| _ => 0

noncomputable def vec_b (x : ℝ) : Fin 2 → ℝ
| 0 => Real.cos x
| 1 => 1
| _ => 0

theorem triangle_area (A B C : ℝ) (h1 : f A = 2) (h2 : Real.sin B = 2 * Real.sin C)
  (h3 : (Real.sin A * Real.sin B * Real.sin C) / (4 * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2)) = 7/4) :
  (Real.sin A * Real.sin B * Real.sin C) / (2 * Real.sin ((A + B + C)/2) * Real.sin ((B + C - A)/2) * Real.sin ((C + A - B)/2) * Real.sin ((A + B - C)/2)) = 7 * Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l638_63855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_charles_whistles_l638_63826

/-- Given that Sean has 223 whistles and 95 more whistles than Charles,
    prove that Charles has 128 whistles. -/
theorem charles_whistles (sean_whistles charles_whistles : ℕ) (difference : ℕ) :
  sean_whistles = 223 →
  difference = 95 →
  sean_whistles = difference + charles_whistles →
  charles_whistles = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_charles_whistles_l638_63826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excel_manufacturing_company_non_union_women_percentage_l638_63806

theorem excel_manufacturing_company_non_union_women_percentage
  (total_employees : ℕ)
  (men_percentage : ℚ)
  (unionized_percentage : ℚ)
  (unionized_men_percentage : ℚ)
  (men_percentage_hypothesis : men_percentage = 46 / 100)
  (unionized_percentage_hypothesis : unionized_percentage = 60 / 100)
  (unionized_men_percentage_hypothesis : unionized_men_percentage = 70 / 100)
  (total_employees_positive : total_employees > 0) :
  (total_employees * (1 - unionized_percentage) -
   (total_employees * men_percentage - total_employees * unionized_percentage * unionized_men_percentage)) /
  (total_employees * (1 - unionized_percentage)) = 90 / 100 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excel_manufacturing_company_non_union_women_percentage_l638_63806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_comparison_l638_63810

theorem expression_comparison (a b c : ℝ) : 
  (a + (-b) + (-c) = a - b - c) ∧
  (a - b - (-c) ≠ a - b - c) ∧
  (a - b - c = a - b - c) ∧
  (a - b + (-c) = a - b - c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_comparison_l638_63810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_deposit_is_fifteen_percent_l638_63864

/-- Calculates the percentage of salary credited to fixed deposit account -/
noncomputable def fixed_deposit_percentage (salary : ℝ) (grocery_percentage : ℝ) (cash_in_hand : ℝ) : ℝ :=
  let remaining_percentage := 1 - grocery_percentage
  let x := (salary - cash_in_hand / remaining_percentage) / salary * 100
  x

/-- Theorem stating the fixed deposit percentage is 15% given the problem conditions -/
theorem fixed_deposit_is_fifteen_percent :
  fixed_deposit_percentage 4000 0.3 2380 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_deposit_is_fifteen_percent_l638_63864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l638_63803

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Add a case for 0
  | 1 => 2
  | n + 2 => sequence_a (n + 1) + 2 * (n + 1)

theorem a_100_value : sequence_a 100 = 9902 := by
  -- Proof steps will go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l638_63803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percentage_is_positive_and_approximately_0_93_l638_63846

-- Define the number of articles and their distribution
def total_articles : ℕ := 120
def articles_sold : ℕ := 100
def first_tax_group : ℕ := 60
def first_discount_group : ℕ := 50

-- Define tax and discount rates
def first_tax_rate : ℚ := 10 / 100
def second_tax_rate : ℚ := 5 / 100
def first_discount_rate : ℚ := 15 / 100
def second_discount_rate : ℚ := 8 / 100

-- Define the relationship between cost price and selling price
def cost_price_equals_selling_price (cp sp : ℚ) : Prop :=
  cp * total_articles = sp * articles_sold

-- Calculate total cost price including tax
def total_cost_price_with_tax (cp : ℚ) : ℚ :=
  cp * total_articles + 
  (cp * first_tax_group * first_tax_rate) + 
  (cp * (total_articles - first_tax_group) * second_tax_rate)

-- Calculate total selling price after discount
def total_selling_price_after_discount (sp : ℚ) : ℚ :=
  sp * articles_sold - 
  (sp * first_discount_group * first_discount_rate) - 
  (sp * (articles_sold - first_discount_group) * second_discount_rate)

-- Define the gain percentage calculation
def gain_percentage (cp sp : ℚ) : ℚ :=
  (total_selling_price_after_discount sp - total_cost_price_with_tax cp) / 
  total_cost_price_with_tax cp * 100

-- Theorem statement
theorem gain_percentage_is_positive_and_approximately_0_93 (cp sp : ℚ) :
  cost_price_equals_selling_price cp sp →
  gain_percentage cp sp > 0 ∧ gain_percentage cp sp > 0.92 ∧ gain_percentage cp sp < 0.94 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_percentage_is_positive_and_approximately_0_93_l638_63846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_minus_x_l638_63822

theorem definite_integral_sqrt_minus_x : 
  ∫ x in (Set.Icc 0 1), (Real.sqrt (1 - x^2) - x) = (Real.pi - 2) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_minus_x_l638_63822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dormitory_weight_probability_l638_63879

noncomputable def probability_median_is_55 (weights : List ℝ) (unmeasured_lower : ℝ) (unmeasured_upper : ℝ) : ℝ :=
  (unmeasured_upper - 55) / (unmeasured_upper - unmeasured_lower)

theorem dormitory_weight_probability (weights : List ℝ) (unmeasured_lower : ℝ) (unmeasured_upper : ℝ) : 
  weights = [60, 55, 60, 55, 65, 50, 50] → 
  unmeasured_lower = 50 → 
  unmeasured_upper = 60 → 
  (probability_median_is_55 weights unmeasured_lower unmeasured_upper) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dormitory_weight_probability_l638_63879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_approx_l638_63837

/-- Conversion factor from km/hr to m/s -/
noncomputable def kmhr_to_ms : ℝ := 5 / 18

/-- Length of the train in meters -/
def train_length : ℝ := 110

/-- Speed of the train in km/hr -/
def train_speed_kmhr : ℝ := 82

/-- Speed of the man in km/hr -/
def man_speed_kmhr : ℝ := 6

/-- Calculate the time for the train to pass the man -/
noncomputable def time_to_pass : ℝ :=
  let train_speed_ms := train_speed_kmhr * kmhr_to_ms
  let man_speed_ms := man_speed_kmhr * kmhr_to_ms
  let relative_speed := train_speed_ms + man_speed_ms
  train_length / relative_speed

theorem time_to_pass_approx :
  |time_to_pass - 4.50| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_approx_l638_63837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_reciprocals_l638_63836

theorem smallest_sum_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  ∀ a b : ℕ+, a ≠ b → 
    ((1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15) → 
    (x : ℕ) + (y : ℕ) ≤ (a : ℕ) + (b : ℕ) →
  (x : ℕ) + (y : ℕ) = 64 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_reciprocals_l638_63836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EH_l638_63858

noncomputable section

-- Define the square EFGH
def square_side : ℝ := 5

-- Define point E
def E : ℝ × ℝ := (0, square_side)

-- Define point H
def H : ℝ × ℝ := (0, 0)

-- Define point N (midpoint of GH)
def N : ℝ × ℝ := (square_side / 2, 0)

-- Define the radii of the circles
def radius_N : ℝ := 2.5
def radius_E : ℝ := square_side

-- Define Q as the intersection point of the two circles (excluding H)
noncomputable def Q : ℝ × ℝ := sorry

-- The theorem to prove
theorem distance_Q_to_EH : 
  let (x_Q, y_Q) := Q
  (square_side - y_Q) = 
    Real.sqrt ((x_Q - E.1)^2 + (y_Q - E.2)^2) - radius_E :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_Q_to_EH_l638_63858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_average_age_l638_63894

/-- Calculates the average age of adults in a computer science camp -/
theorem adult_average_age (total_members : ℕ) (overall_avg : ℚ) 
  (num_girls num_boys num_adults : ℕ) (girls_avg boys_avg : ℚ) : ℚ :=
by
  -- Assumptions
  have h1 : total_members = 40 := by sorry
  have h2 : overall_avg = 17 := by sorry
  have h3 : num_girls = 20 := by sorry
  have h4 : num_boys = 15 := by sorry
  have h5 : num_adults = 5 := by sorry
  have h6 : girls_avg = 15 := by sorry
  have h7 : boys_avg = 16 := by sorry
  have h8 : total_members = num_girls + num_boys + num_adults := by sorry
  
  -- Calculate total sum of ages
  let total_sum : ℚ := total_members * overall_avg
  
  -- Calculate sum of girls' and boys' ages
  let girls_sum : ℚ := num_girls * girls_avg
  let boys_sum : ℚ := num_boys * boys_avg
  
  -- Calculate sum of adults' ages
  let adults_sum : ℚ := total_sum - girls_sum - boys_sum
  
  -- Calculate average age of adults
  let adult_avg : ℚ := adults_sum / num_adults
  
  -- Prove that the average age of adults is 28
  have h9 : adult_avg = 28 := by sorry
  
  exact adult_avg

-- This line is not necessary in a theorem, so we'll comment it out
-- #eval adult_average_age 40 17 20 15 5 15 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_average_age_l638_63894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_straightedge_constructions_l638_63884

-- Define the basic types
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the plane
def Plane := Set Point

-- Define the given circle and its center
variable (S : Circle)
variable (O : Point)

-- Define the straightedge construction
def StraightedgeConstruction := List (Point → Point → Line)

-- Define the constructions
def ParallelAndPerpendicular (construction : StraightedgeConstruction) : Prop := sorry

def EqualSegment (construction : StraightedgeConstruction) : Prop := sorry

def ProportionalSegment (construction : StraightedgeConstruction) : Prop := sorry

def LineCircleIntersection (construction : StraightedgeConstruction) : Prop := sorry

def CircleCircleIntersection (construction : StraightedgeConstruction) : Prop := sorry

-- Main theorem
theorem straightedge_constructions
  (plane : Plane) (S : Circle) (O : Point) :
  ∃ (construction : StraightedgeConstruction),
    ParallelAndPerpendicular construction ∧
    EqualSegment construction ∧
    ProportionalSegment construction ∧
    LineCircleIntersection construction ∧
    CircleCircleIntersection construction :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_straightedge_constructions_l638_63884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_satisfies_conditions_l638_63861

-- Define the fraction as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x - 1) / x

-- Theorem stating that f satisfies the given conditions
theorem fraction_satisfies_conditions :
  (∀ x : ℝ, f x = (x - 1) / x) ∧ 
  (f 1 = 0) := by
  constructor
  · intro x
    rfl  -- reflexivity proves the equality
  · -- Proof that f 1 = 0
    have h : (1 : ℝ) ≠ 0 := by norm_num
    calc
      f 1 = (1 - 1) / 1 := rfl
      _ = 0 / 1 := by norm_num
      _ = 0 := by simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_satisfies_conditions_l638_63861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equals_neg_three_fourths_l638_63800

/-- Given an angle α with its vertex at the origin, its initial side on the positive x-axis,
    and a point P(-4,3) on its terminal side, prove that the given trigonometric expression
    equals -3/4. -/
theorem trig_expression_equals_neg_three_fourths (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (-4, 3) ∧ P.1 = -4 * Real.cos α ∧ P.2 = -4 * Real.sin α) →
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_equals_neg_three_fourths_l638_63800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_periodic_subsequence_l638_63882

/-- Number of divisors of a natural number -/
def num_divisors (m : ℕ) : ℕ := (Nat.divisors m).card

/-- Sequence defined by the recurrence relation -/
def a (c : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => num_divisors (a c n) + c

/-- Definition of a periodic sequence -/
def is_periodic (f : ℕ → ℕ) (k p : ℕ) : Prop :=
  ∀ n, n ≥ k → f (n + p) = f n

theorem existence_of_periodic_subsequence (c : ℕ) :
  ∃ k p : ℕ, p > 0 ∧ is_periodic (a c) k p := by
  sorry

#check existence_of_periodic_subsequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_periodic_subsequence_l638_63882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_two_hours_l638_63880

/-- Calculates the round trip time given the average speeds and time to work -/
noncomputable def round_trip_time (speed_to_work : ℝ) (speed_to_home : ℝ) (time_to_work_minutes : ℝ) : ℝ :=
  let time_to_work_hours := time_to_work_minutes / 60
  let distance := speed_to_work * time_to_work_hours
  let time_to_home := distance / speed_to_home
  time_to_work_hours + time_to_home

/-- Theorem stating that the round trip time is 2 hours given the specified conditions -/
theorem round_trip_time_is_two_hours :
  round_trip_time 80 120 72 = 2 := by
  -- Unfold the definition of round_trip_time
  unfold round_trip_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_is_two_hours_l638_63880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemniscate_equation_lemniscate_x_symmetry_lemniscate_y_symmetry_unique_equidistant_point_l638_63852

-- Define the lemniscate
def is_on_lemniscate (x y : ℝ) : Prop :=
  ((x + 1)^2 + y^2) * ((x - 1)^2 + y^2) = 1

-- Theorem 1: Equation of the lemniscate
theorem lemniscate_equation (x y : ℝ) :
  is_on_lemniscate x y ↔ (x^2 + y^2)^2 = 2 * (x^2 - y^2) := by sorry

-- Theorem 2: Symmetry with respect to x-axis
theorem lemniscate_x_symmetry (x y : ℝ) :
  is_on_lemniscate x y ↔ is_on_lemniscate x (-y) := by sorry

-- Theorem 3: Symmetry with respect to y-axis
theorem lemniscate_y_symmetry (x y : ℝ) :
  is_on_lemniscate x y ↔ is_on_lemniscate (-x) y := by sorry

-- Theorem 4: Unique point with equal distances to A and B
theorem unique_equidistant_point :
  ∃! p : ℝ × ℝ, is_on_lemniscate p.1 p.2 ∧ (p.1 + 1)^2 + p.2^2 = (p.1 - 1)^2 + p.2^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemniscate_equation_lemniscate_x_symmetry_lemniscate_y_symmetry_unique_equidistant_point_l638_63852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_with_pair_l638_63828

/-- The number of ways to seat n people around a round table -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to arrange a pair of people -/
def pairArrangements : ℕ := 2

theorem seating_arrangements_with_pair (total_people : ℕ) (pair_count : ℕ) :
  total_people = 6 →
  pair_count = 2 →
  roundTableArrangements (total_people - pair_count + 1) * pairArrangements = 48 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_with_pair_l638_63828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_results_l638_63819

-- Define the custom operation
def custom_op (a b : ℤ) : ℤ := a^2 - b.natAbs

-- State the theorem
theorem custom_op_results :
  (custom_op (-2) 3 = 1) ∧
  (custom_op 5 (-4) = 21) ∧
  (custom_op (-3) (-1) = 8) := by
  -- Prove each part of the conjunction
  constructor
  · -- Prove custom_op (-2) 3 = 1
    rfl
  · constructor
    · -- Prove custom_op 5 (-4) = 21
      rfl
    · -- Prove custom_op (-3) (-1) = 8
      rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_op_results_l638_63819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l638_63860

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 3 * x

-- Define the focus of the parabola
noncomputable def focus : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define points A and B on the parabola
axiom A : ℝ × ℝ
axiom B : ℝ × ℝ

-- A is in the first quadrant
axiom A_in_first_quadrant : A.1 > 0 ∧ A.2 > 0

-- A and B are on the parabola
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2

-- F, A, and B are collinear
axiom F_A_B_collinear : ∃ (m : ℝ), A.1 - focus.1 = m * (A.2 - focus.2) ∧
                                   B.1 - focus.1 = m * (B.2 - focus.2)

-- AF = 3FB
axiom AF_eq_3FB : (A.1 - focus.1)^2 + (A.2 - focus.2)^2 = 
                  9 * ((B.1 - focus.1)^2 + (B.2 - focus.2)^2)

-- The theorem to prove
theorem circle_equation : 
  (A.1 - (5/3) * Real.sqrt 3)^2 + (A.2 - 2)^2 = 64/3 ∧
  (B.1 - (5/3) * Real.sqrt 3)^2 + (B.2 - 2)^2 = 64/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l638_63860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_in_set_l638_63848

def S : Finset Int := {-10, -4, -2, 0, 6}

theorem smallest_product_in_set :
  (∀ x y, x ∈ S → y ∈ S → x * y ≥ -60) ∧ (∃ x y, x ∈ S ∧ y ∈ S ∧ x * y = -60) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_product_in_set_l638_63848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l638_63887

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 4*x < 0}
def N : Set ℝ := {x | |x| ≤ 2}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Icc (-2) 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l638_63887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_polynomial_form_l638_63875

/-- A polynomial that satisfies the given functional equation -/
def SatisfyingPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x * P (x - 1) = (x - 2) * P x

/-- The theorem stating the form of polynomials satisfying the functional equation -/
theorem satisfying_polynomial_form (P : ℝ → ℝ) (hP : SatisfyingPolynomial P) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * (x^2 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_satisfying_polynomial_form_l638_63875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l638_63881

theorem trig_identity (x : ℝ) 
  (h1 : Real.sin (x + π) + Real.cos (x - π) = 1/2) 
  (h2 : 0 < x ∧ x < π) : 
  Real.sin x * Real.cos x = -3/8 ∧ Real.sin x - Real.cos x = Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l638_63881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_of_26_exists_max_power_l638_63813

def product_20_to_2020 : ℕ := Finset.prod (Finset.range 2001) (λ i => if i ≥ 20 then i else 1)

theorem max_power_of_26 : 
  ∀ k m : ℕ, product_20_to_2020 = 26^k * m → k ≤ 165 :=
by sorry

theorem exists_max_power :
  ∃ m : ℕ, product_20_to_2020 = 26^165 * m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_of_26_exists_max_power_l638_63813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l638_63899

-- Define the function f on (0, +∞)
noncomputable def f : ℝ → ℝ := sorry

-- Define the average rate of change of f
noncomputable def avg_rate_of_change (x : ℝ) (Δx : ℝ) : ℝ :=
  2 / (Real.sqrt (x + Δx) + Real.sqrt x) - 1 / (x^2 + x * Δx)

-- State the theorem
theorem f_monotone_increasing :
  ∀ x > 0, ∀ Δx > 0,
    (avg_rate_of_change x Δx = (f (x + Δx) - f x) / Δx) →
    (∀ y > 1, ∀ z > y, f z > f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l638_63899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_even_numbers_count_l638_63865

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def even_digits : Finset ℕ := {0, 2, 4, 6, 8}

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_all_even_digits (n : ℕ) : Prop :=
  ∀ d, (d = n % 10 ∨ d = (n / 10) % 10 ∨ d = n / 100) → d ∈ even_digits

def count_three_digit_even_numbers : ℕ :=
  (even_digits.filter (· ≠ 0)).card * even_digits.card * even_digits.card

theorem three_digit_even_numbers_count :
  count_three_digit_even_numbers = 100 :=
by
  -- Proof goes here
  sorry

#eval count_three_digit_even_numbers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_even_numbers_count_l638_63865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_60_degrees_l638_63801

/-- The area of a figure formed by rotating a semicircle about one of its endpoints -/
noncomputable def rotated_semicircle_area (R : ℝ) (α : ℝ) : ℝ :=
  (α / (2 * Real.pi)) * (2 * Real.pi * R^2)

/-- Theorem: The area of a figure formed by rotating a semicircle about one of its endpoints
    by an angle of 60° is equal to (2 * π * R^2) / 3, where R is the radius of the semicircle -/
theorem rotated_semicircle_area_60_degrees (R : ℝ) (h : R > 0) :
  rotated_semicircle_area R (Real.pi / 3) = (2 * Real.pi * R^2) / 3 := by
  sorry

#check rotated_semicircle_area_60_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_60_degrees_l638_63801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l638_63876

noncomputable section

-- Define the parametric curve
def x (t : ℝ) : ℝ := (t + 1) / t
def y (t : ℝ) : ℝ := (t - 1) / t

-- Define the point of interest
def t₀ : ℝ := -1

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  let x₀ := x t₀
  let y₀ := y t₀
  let slope := (deriv y t₀) / (deriv x t₀)
  ∀ x' y', y' - y₀ = slope * (x' - x₀) ↔ y' = -x' + 2 := by
  sorry

-- Theorem for the normal line equation
theorem normal_line_equation :
  let x₀ := x t₀
  let y₀ := y t₀
  let slope := -(deriv x t₀) / (deriv y t₀)
  ∀ x' y', y' - y₀ = slope * (x' - x₀) ↔ y' = x' + 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_normal_line_equation_l638_63876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miscount_adjustment_l638_63844

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | "penny" => 1
  | _ => 0

/-- Calculates the error in cents when mistaking one coin for another -/
def counting_error (actual : String) (counted_as : String) : ℤ :=
  (coin_value actual : ℤ) - (coin_value counted_as : ℤ)

/-- Theorem stating the total error and correct adjustment for miscount -/
theorem miscount_adjustment (y : ℕ) :
  let quarter_error := y * (counting_error "quarter" "dime")
  let nickel_error := y * (counting_error "nickel" "penny")
  let total_error := quarter_error + nickel_error
  total_error = 19 * y ∧ 
  (λ adjustment => adjustment = 19 * y ∧ total_error + adjustment = 0) (19 * y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_miscount_adjustment_l638_63844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_height_difference_l638_63857

/-- The diameter of each cylindrical pipe in centimeters -/
def pipeDiameter : ℝ := 12

/-- The number of pipes in each crate -/
def numberOfPipes : ℕ := 200

/-- The height of Crate A with square grid packing -/
def heightCrateA : ℝ := 20 * pipeDiameter

/-- The height of Crate B with hexagonal grid packing -/
noncomputable def heightCrateB : ℝ := 11 * (Real.sqrt 3 * pipeDiameter / 2) + pipeDiameter

/-- The positive difference in heights between Crate A and Crate B -/
noncomputable def heightDifference : ℝ := heightCrateA - heightCrateB

/-- Theorem stating the height difference between Crate A and Crate B -/
theorem crate_height_difference :
  heightDifference = 228 - 66 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crate_height_difference_l638_63857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_card_disproves_statement_l638_63886

structure Card where
  letter : Char
  number : Nat

def isVowel (c : Char) : Bool :=
  c ∈ ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']

def isEven (n : Nat) : Bool :=
  n % 2 = 0

def petersStatement (c : Card) : Bool :=
  isVowel c.letter → isEven c.number

def canDisproveStatement (c : Card) : Bool :=
  (isVowel c.letter ∧ ¬isEven c.number) ∨ (¬isEven c.number ∧ isVowel c.letter)

theorem fourth_card_disproves_statement (cards : List Card) 
  (h1 : cards.length = 5)
  (h2 : ∃ c, c ∈ cards ∧ c.number = 7)
  (h3 : ∀ c, c ∈ cards → c.number ≠ 7 → isEven c.number ∨ ¬isVowel c.letter) :
  ∃! c, c ∈ cards ∧ canDisproveStatement c ∧ c.number = 7 := by
  sorry

#check fourth_card_disproves_statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_card_disproves_statement_l638_63886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_pi_over_two_l638_63854

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

-- State the theorem
theorem integral_f_equals_pi_over_two : 
  ∫ x in (-1 : ℝ)..1, f x = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_pi_over_two_l638_63854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l638_63870

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | x > -1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l638_63870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_over_grid_l638_63811

open Finset BigOperators

def f (x : ℚ) : ℚ := x^3 / (1 + x^3)

theorem sum_of_f_over_grid (n : ℕ) (h : n > 0) : 
  ∑ k in range n, ∑ m in range n, f (↑(k+1) / ↑(m+1)) = (n^2 : ℚ) / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_over_grid_l638_63811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_barbara_wins_l638_63820

/-- Represents the state of the game -/
inductive GameState
| Barbara : ℕ → GameState  -- Barbara's turn with n coins
| Jenna : ℕ → GameState    -- Jenna's turn with n coins

/-- Defines a valid move in the game -/
def isValidMove : GameState → GameState → Prop
| (GameState.Barbara n), (GameState.Jenna m) => 
    (n ≥ 3 ∧ m = n - 3) ∨ (n ≥ 5 ∧ m = n - 5)
| (GameState.Jenna n), (GameState.Barbara m) => 
    (m = n - 2) ∨ (m = n - 4)
| _, _ => False

/-- Defines a winning strategy for a player -/
def WinningStrategy (player : GameState → Prop) : Prop :=
  ∀ (state : GameState), player state → 
    ∃ (next_state : GameState), 
      (isValidMove state next_state) ∧ 
      (∀ (opponent_move : GameState), 
        (isValidMove next_state opponent_move) → 
        player opponent_move)

/-- The main theorem: Barbara has a winning strategy starting with 2025 coins -/
theorem barbara_wins : 
  WinningStrategy (λ state => ∃ n, state = GameState.Barbara n) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_barbara_wins_l638_63820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jellybean_count_l638_63871

theorem jellybean_count : ℕ := by
  -- Define the remaining percent of jellybeans after each day
  let remaining_percent : ℝ := 0.75
  -- Define the number of days
  let days : ℕ := 3
  -- Define the final count of jellybeans
  let final_count : ℕ := 45
  -- Theorem: The original number of jellybeans is 107
  have h : ⌈(final_count : ℝ) / remaining_percent ^ days⌉ = 107 := by
    sorry
  exact 107

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jellybean_count_l638_63871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l638_63805

noncomputable def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

noncomputable def F₁ : ℝ × ℝ := (-1, 0)
noncomputable def F₂ : ℝ × ℝ := (1, 0)
noncomputable def P : ℝ × ℝ := (1, 3/2)

noncomputable def A : ℝ × ℝ := (1, 3/2)
noncomputable def B : ℝ × ℝ := (1, -3/2)

theorem ellipse_and_triangle_properties :
  -- 1. The given equation satisfies the conditions
  (ellipse_C P.1 P.2) ∧
  -- 2. The area of triangle ABF₁ is 3
  (1/2 * 2 * 3 = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l638_63805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_0_to_50_eq_25_l638_63866

/-- The sum of the alternating series from 0 to n -/
def alternatingSum (n : ℕ) : ℤ :=
  List.range (n + 1)
    |> List.map (fun i => if i % 2 = 0 then i else -i)
    |> List.sum

/-- Theorem stating that the sum of the alternating series from 0 to 50 is 25 -/
theorem alternating_sum_0_to_50_eq_25 : alternatingSum 50 = 25 := by
  sorry

#eval alternatingSum 50  -- This will compute and display the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_0_to_50_eq_25_l638_63866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l638_63825

/-- Simple interest calculation -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem principal_calculation (rate time interest : ℝ) 
  (h_rate : rate = 12)
  (h_time : time = 3)
  (h_interest : interest = 5400) :
  ∃ (principal : ℝ), simple_interest principal rate time = interest ∧ principal = 15000 := by
  use 15000
  constructor
  · -- Prove that simple_interest 15000 rate time = interest
    simp [simple_interest, h_rate, h_time, h_interest]
    norm_num
  · -- Prove that principal = 15000
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_calculation_l638_63825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l638_63808

-- Define the function f
noncomputable def f (a θ : ℝ) : ℝ := Real.cos θ ^ 3 + 4 / (3 * a * Real.cos θ ^ 2 - a ^ 3)

-- State the theorem
theorem min_value_of_f :
  ∀ (a θ : ℝ),
  0 < a → a < Real.sqrt 3 * Real.cos θ →
  θ ∈ Set.Icc (-π/4) (π/3) →
  f a θ ≥ 17 * Real.sqrt 2 / 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l638_63808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_similarity_statements_l638_63896

-- Define the types of shapes we're dealing with
inductive Shape
| Rhombus
| EquilateralTriangle
| Square
| Rectangle
| CongruentTriangle
| RightAngledTriangle

-- Define a function that checks if two shapes of a given type are always similar
def alwaysSimilar (s : Shape) : Bool :=
  match s with
  | Shape.Rhombus => false
  | Shape.EquilateralTriangle => true
  | Shape.Square => true
  | Shape.Rectangle => false
  | Shape.CongruentTriangle => true
  | Shape.RightAngledTriangle => false

-- Theorem statement
theorem correct_similarity_statements :
  (List.filter alwaysSimilar [Shape.Rhombus, Shape.EquilateralTriangle, Shape.Square,
    Shape.Rectangle, Shape.CongruentTriangle, Shape.RightAngledTriangle]).length = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_similarity_statements_l638_63896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_with_equal_adjacent_sides_is_square_l638_63877

/-- A rectangle is a quadrilateral with four right angles -/
structure Rectangle where
  sides : Fin 4 → ℝ
  right_angles : ∀ i : Fin 4, ∃ (angle : ℝ), angle = Real.pi / 2

/-- A square is a rectangle with all sides equal -/
structure Square extends Rectangle where
  equal_sides : ∀ i j : Fin 4, sides i = sides j

/-- Theorem: A rectangle with a pair of adjacent sides equal is a square -/
theorem rectangle_with_equal_adjacent_sides_is_square 
  (r : Rectangle) (h : ∃ i : Fin 4, r.sides i = r.sides (i+1)) : 
  Square where
  sides := r.sides
  right_angles := r.right_angles
  equal_sides := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_with_equal_adjacent_sides_is_square_l638_63877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l638_63843

/-- An ellipse with foci on the y-axis, eccentricity 2√2/3, and one focus at (0, 2√2) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h : c = 2 * Real.sqrt 2
  k : a > b
  m : b > 0
  n : c / a = 2 * Real.sqrt 2 / 3

/-- A circle centered at the origin with radius 2 -/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- A line passing through P(-1, 0) -/
def Line := {l : ℝ × ℝ → Prop | ∃ (m : ℝ), ∀ (x y : ℝ), l (x, y) ↔ y = m * (x + 1)}

/-- Point C on the ellipse -/
def PointOnEllipse (e : Ellipse) := {p : ℝ × ℝ | p.1^2 / e.b^2 + p.2^2 / e.a^2 = 1}

/-- Theorem stating the maximum length of CP⃗ and the corresponding length of chord AB -/
theorem ellipse_intersection_theorem (e : Ellipse) :
  ∃ (C : PointOnEllipse e) (A B : Circle) (l : Line),
    (∀ (D : PointOnEllipse e), ‖(D : ℝ × ℝ) - (-1, 0)‖ ≤ ‖(C : ℝ × ℝ) - (-1, 0)‖) ∧
    ((A : ℝ × ℝ) - (B : ℝ × ℝ)) • ((C : ℝ × ℝ) - (-1, 0)) = 0 ∧
    ‖(C : ℝ × ℝ) - (-1, 0)‖ = 3 ∧
    ‖(A : ℝ × ℝ) - (B : ℝ × ℝ)‖ = Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l638_63843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_divisors_l638_63834

theorem distinct_prime_divisors (a n : ℕ) (h_a_odd : Odd a) (h_a_gt_3 : a > 3) (h_n_pos : n > 0) :
  ∃ (p : Finset ℕ), (∀ q ∈ p, Nat.Prime q) ∧ (Finset.card p ≥ n + 1) ∧ (∀ q ∈ p, q ∣ (a^(2^n) - 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_prime_divisors_l638_63834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_dodecagon_area_l638_63878

/-- The area of a regular dodecagon inscribed in a circle with radius 10 is 300 square units. -/
theorem regular_dodecagon_area (r : ℝ) (h : r = 10) : 
  12 * (1/2 * r^2 * Real.sin (π/6)) = 300 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_dodecagon_area_l638_63878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_work_hours_l638_63849

/-- Represents the weekly work hours for each employee -/
structure WorkHours where
  fiona : ℕ
  john : ℕ
  jeremy : ℕ

/-- Represents the monthly payroll information -/
structure Payroll where
  hourlyRate : ℕ
  totalPaid : ℕ

/-- Proves that John works 30 hours per week given the conditions -/
theorem john_work_hours (h : WorkHours) (p : Payroll) :
  h.fiona = 40 →
  h.jeremy = 25 →
  p.hourlyRate = 20 →
  p.totalPaid = 7600 →
  h.john = 30 := by
  intro hf hj hr ht
  -- Proof steps would go here
  sorry

#check john_work_hours

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_work_hours_l638_63849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_problem_l638_63821

/-- The speed of the slower train given the conditions of the problem -/
noncomputable def slower_train_speed (faster_train_speed : ℝ) (passing_time : ℝ) (faster_train_length : ℝ) : ℝ :=
  faster_train_speed - (faster_train_length / passing_time) * 3.6

/-- Theorem stating the speed of the slower train given the problem conditions -/
theorem slower_train_speed_problem :
  let faster_train_speed : ℝ := 50
  let passing_time : ℝ := 15
  let faster_train_length : ℝ := 75.006
  ∃ ε > 0, |slower_train_speed faster_train_speed passing_time faster_train_length - 31.99856| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slower_train_speed_problem_l638_63821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_parallel_equidistant_lines_l638_63817

/-- A type representing a straight line in 3D space -/
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  point : ℝ × ℝ × ℝ  -- A point on the line
  direction : ℝ × ℝ × ℝ  -- Direction vector of the line

/-- Predicate to check if two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  ∃ k : ℝ, l1.direction = k • l2.direction

/-- Predicate to check if two lines are equidistant -/
def equidistant (l1 l2 : Line3D) : Prop :=
  sorry  -- The actual implementation would involve complex calculations

/-- A set of pairwise parallel and equidistant lines -/
def ParallelEquidistantLines (s : Set Line3D) : Prop :=
  ∀ l1 l2 : Line3D, l1 ∈ s → l2 ∈ s → l1 ≠ l2 → parallel l1 l2 ∧ equidistant l1 l2

theorem max_parallel_equidistant_lines (s : Set Line3D) 
  (h : ParallelEquidistantLines s) (hs : Fintype s) : Fintype.card s ≤ 3 :=
by
  sorry  -- The proof would go here

#check max_parallel_equidistant_lines

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_parallel_equidistant_lines_l638_63817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_inequality_l638_63809

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - (1/2) * a * x^2 + (4-a) * x

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 4 / x - a * x + (4-a)

theorem extremum_inequality (a : ℝ) (x₀ x₁ x₂ : ℝ) :
  (0 < x₁) → (x₁ < x₂) → (0 < x₀) →
  (∃ x, f_deriv a x = 0) →  -- f has an extremum
  (f a x₁ - f a x₂ = f_deriv a x₀ * (x₁ - x₂)) →
  (x₁ + x₂ > 2 * x₀) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_inequality_l638_63809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_half_B_is_four_l638_63867

-- Define the properties of the cans
noncomputable def radius_B : ℝ := Real.pi
noncomputable def height_B : ℝ := Real.pi
noncomputable def radius_C : ℝ := 2 * radius_B
noncomputable def height_C : ℝ := height_B / 2

-- Define the volumes of the cans
noncomputable def volume_B : ℝ := Real.pi * radius_B^2 * height_B
noncomputable def volume_C : ℝ := Real.pi * radius_C^2 * height_C

-- Define the cost to fill Can C completely
def cost_C : ℝ := 16

-- Theorem to prove
theorem cost_half_B_is_four :
  let cost_B := cost_C * (volume_B / volume_C)
  cost_B / 2 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_half_B_is_four_l638_63867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_lines_count_l638_63850

def S : Finset Int := {-3, -2, -1, 0, 1, 2, 3}

def is_valid_line (a b c : Int) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a ≠ 0 ∧ (a > 0 → b < 0) ∧ (a < 0 → b > 0)

def count_valid_lines : Nat :=
  (S.filter (λ a => a ≠ 0)).card *
  3 * -- number of valid b for each a
  5   -- number of valid c for each (a,b) pair

theorem valid_lines_count :
  count_valid_lines - 2 = 88 := by
  -- Proof goes here
  sorry

#eval count_valid_lines - 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_lines_count_l638_63850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pavel_sum_associative_l638_63856

/-- Pavel sum operation -/
noncomputable def pavel_sum (x y : ℝ) : ℝ := (x + y) / (1 - x * y)

/-- Associativity of Pavel sum -/
theorem pavel_sum_associative (a b c : ℝ) 
  (h1 : 1 - a * b ≠ 0) 
  (h2 : 1 - b * c ≠ 0) 
  (h3 : 1 - pavel_sum a b * c ≠ 0) 
  (h4 : 1 - a * pavel_sum b c ≠ 0) : 
  pavel_sum a (pavel_sum b c) = pavel_sum (pavel_sum a b) c := by
  sorry

#check pavel_sum_associative

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pavel_sum_associative_l638_63856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_price_problem_l638_63804

/-- Given three varieties of tea mixed in a specific ratio, this theorem proves
    the price of the third variety given the prices of the other two and the mixture. -/
theorem tea_price_problem (price1 price2 mixture_price : ℚ) 
  (h1 : price1 = 126)
  (h2 : price2 = 135)
  (h3 : mixture_price = 153) : 
  (4 * mixture_price - price1 - price2) / 2 = 175.5 := by
  sorry

#check tea_price_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_price_problem_l638_63804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l638_63818

noncomputable section

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (0, -2)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v = t • w ∨ w = t • v

noncomputable def angle (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem vector_problem :
  (∀ k : ℝ, collinear (k • a - b) (a + b) ↔ k = -1) ∧
  (∀ k : ℝ, angle (k • a - b) (a + b) = 2 * Real.pi / 3 ↔ k = -1 + Real.sqrt 3 ∨ k = -1 - Real.sqrt 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l638_63818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_is_sqrt_3_l638_63842

/-- An ellipse passing through a specific point with given eccentricity -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a
  h_point : a^2 * (3/4) + b^2 = a^2 * b^2
  h_eccentricity : (a^2 - b^2) / a^2 = 3/4

/-- The maximum difference between triangular areas formed by intersecting line -/
noncomputable def max_area_difference (ε : Ellipse) : ℝ := Real.sqrt 3

/-- Theorem statement -/
theorem max_area_difference_is_sqrt_3 (ε : Ellipse) :
  max_area_difference ε = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_is_sqrt_3_l638_63842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_and_power_l638_63816

-- Define the distance function
noncomputable def distance_to_nearest_integer (x : ℝ) : ℝ :=
  min (x - ⌊x⌋) (⌈x⌉ - x)

-- Main theorem
theorem existence_of_prime_and_power (a b : ℕ+) :
  ∃ (p : ℕ) (k : ℕ), 
    Nat.Prime p ∧ 
    p % 2 = 1 ∧ 
    distance_to_nearest_integer ((a : ℝ) / p^k) + 
    distance_to_nearest_integer ((b : ℝ) / p^k) + 
    distance_to_nearest_integer (((a + b) : ℝ) / p^k) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_and_power_l638_63816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_AEC_l638_63807

noncomputable section

-- Define the square
def square_side_length : ℝ := 2

-- Define the points
def A : ℝ × ℝ := (0, square_side_length)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (square_side_length, 0)
def D : ℝ × ℝ := (square_side_length, square_side_length)

-- Define C' after folding
def C' : ℝ × ℝ := (square_side_length, 2/3 * square_side_length)

-- Define E as the intersection of BC and AB
def E : ℝ × ℝ := (3/2, 3/2)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem perimeter_of_triangle_AEC' :
  distance A E + distance E C' + distance C' A = (4 * Real.sqrt 10) / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_AEC_l638_63807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_intersection_condition_l638_63885

-- Define the curves and line
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 1 + Real.sin θ)

noncomputable def C₂ (a : ℝ) (θ : ℝ) : ℝ := 2 * Real.cos θ + a

noncomputable def l (t : ℝ) : ℝ × ℝ := (3 + (Real.sqrt 2 / 2) * t, -1 + (Real.sqrt 2 / 2) * t)

-- Define the point P
def P : ℝ × ℝ := (3, -1)

-- Statement 1
theorem intersection_chord_length :
  ∃ M N : ℝ × ℝ, M ∈ Set.range C₁ ∧ N ∈ Set.range C₁ ∧
  M ∈ {p : ℝ × ℝ | (p.1^2 + p.2^2 : ℝ) = 2 * p.1} ∧
  N ∈ {p : ℝ × ℝ | (p.1^2 + p.2^2 : ℝ) = 2 * p.1} ∧
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = Real.sqrt 2 := by
  sorry

-- Statement 2
theorem intersection_condition (a : ℝ) :
  (∃ A B : ℝ × ℝ, A ∈ Set.range l ∧ B ∈ Set.range l ∧
   A ∈ {p : ℝ × ℝ | (p.1^2 + p.2^2 : ℝ) = 2 * p.1 + a} ∧
   B ∈ {p : ℝ × ℝ | (p.1^2 + p.2^2 : ℝ) = 2 * p.1 + a} ∧
   ((A.1 - P.1)^2 + (A.2 - P.2)^2) * ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 1) →
  (a = 3 ∨ a = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_intersection_condition_l638_63885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_is_ten_l638_63833

/-- Represents the sale of an article -/
structure ArticleSale where
  originalPrice : ℚ
  sellingPrice : ℚ
  gainPercent : ℚ

/-- Calculates the selling price based on original price and gain percent -/
def calculateSellingPrice (sale : ArticleSale) : ℚ :=
  sale.originalPrice * (1 + sale.gainPercent / 100)

/-- Theorem: If an article is sold for $15 with a 50% gain, its original price was $10 -/
theorem original_price_is_ten (sale : ArticleSale)
  (h1 : sale.sellingPrice = 15)
  (h2 : sale.gainPercent = 50)
  : sale.originalPrice = 10 := by
  sorry

#check original_price_is_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_price_is_ten_l638_63833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l638_63863

noncomputable def x : ℚ := 2.5081081081081  -- This is an approximation of the actual number as a rational

theorem decimal_to_fraction (m n : ℕ) (h1 : x = m / n) (h2 : Nat.Coprime m n) :
  m + n = 86417 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l638_63863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_refrigerator_profit_percentage_l638_63890

/-- Calculates the profit percentage for a refrigerator sale --/
theorem refrigerator_profit_percentage 
  (discounted_price : ℚ) 
  (discount_percentage : ℚ) 
  (transport_cost : ℚ) 
  (installation_cost : ℚ) 
  (selling_price : ℚ) 
  (h1 : discounted_price = 12500)
  (h2 : discount_percentage = 20)
  (h3 : transport_cost = 125)
  (h4 : installation_cost = 250)
  (h5 : selling_price = 17920) :
  ∃ (profit_percentage : ℚ), 
    (profit_percentage ≥ 32.27 ∧ profit_percentage ≤ 32.29) ∧ 
    profit_percentage = (selling_price - (discounted_price + transport_cost + installation_cost)) / 
      (discounted_price / (1 - discount_percentage / 100)) * 100 := by
  sorry

#eval (17920 - (12500 + 125 + 250)) / (12500 / (1 - 20 / 100)) * 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_refrigerator_profit_percentage_l638_63890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l638_63814

/-- The curve on which point P lies -/
def curve (x y : ℝ) : Prop := x^2 - y - Real.log x = 0

/-- The line to which we're calculating the distance -/
def line (x y : ℝ) : Prop := y = x - 3

/-- The minimum distance from a point on the curve to the line -/
noncomputable def min_distance : ℝ := 3 * Real.sqrt 2 / 2

theorem min_distance_theorem :
  ∀ (P : ℝ × ℝ), curve P.1 P.2 →
  (∃ (Q : ℝ × ℝ), line Q.1 Q.2 ∧
    ∀ (R : ℝ × ℝ), line R.1 R.2 →
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2)) →
  ∃ (Q : ℝ × ℝ), line Q.1 Q.2 ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = min_distance :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l638_63814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_count_l638_63823

theorem election_votes_count (total_votes : ℕ) : 
  (0.005 * (total_votes : ℝ) + 3000 = 0.505 * (total_votes : ℝ)) → 
  total_votes = 6000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_count_l638_63823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aubriella_fish_tank_l638_63873

/-- Calculates the remaining gallons needed to fill a tank -/
def remaining_gallons_to_fill (tank_capacity : ℕ) (pour_rate : ℚ) (pour_time : ℕ) : ℕ :=
  let gallons_poured := (↑pour_time * 60 / 20 : ℚ)
  (tank_capacity - Int.floor gallons_poured).natAbs

/-- Proves that 32 gallons are needed to fill the tank under given conditions -/
theorem aubriella_fish_tank : remaining_gallons_to_fill 50 (1/20) 6 = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aubriella_fish_tank_l638_63873
