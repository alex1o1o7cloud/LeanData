import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shortest_side_on_cube_l908_90855

/-- A point on the face of a unit cube -/
structure CubeFacePoint where
  x : ℝ
  y : ℝ
  z : ℝ
  on_face : (x = 0 ∨ x = 1) ∨ (y = 0 ∨ y = 1) ∨ (z = 0 ∨ z = 1)

/-- Distance between two points on the face of a unit cube -/
noncomputable def distance (p q : CubeFacePoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- The shortest side of a triangle formed by three points on the face of a unit cube -/
noncomputable def shortest_side (p q r : CubeFacePoint) : ℝ :=
  min (distance p q) (min (distance q r) (distance r p))

/-- The maximum possible length of the shortest side of a triangle on a unit cube -/
theorem max_shortest_side_on_cube : 
  (∃ (p q r : CubeFacePoint), shortest_side p q r = Real.sqrt 2) ∧
  (∀ (p q r : CubeFacePoint), shortest_side p q r ≤ Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_shortest_side_on_cube_l908_90855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anands_investment_l908_90829

/-- Calculates Anand's investment in a business partnership --/
theorem anands_investment (deepak_investment total_profit deepak_profit : ℚ) :
  deepak_investment = 3200 →
  total_profit = 1380 →
  deepak_profit = 810.28 →
  ∃ anand_investment : ℚ,
    anand_investment * deepak_profit = deepak_investment * (total_profit - deepak_profit) ∧
    abs (anand_investment - 2250.24) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anands_investment_l908_90829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_in_M_l908_90820

-- Define the property that characterizes set M
def belongs_to_M (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

-- Define the given functions
noncomputable def f1 : ℝ → ℝ := λ x => x
noncomputable def f2 : ℝ → ℝ := λ x => 2^x
noncomputable def f3 : ℝ → ℝ := λ x => Real.log (x^2 + 2) / Real.log 2
noncomputable def f4 : ℝ → ℝ := λ x => Real.cos (Real.pi * x)

-- State the theorem
theorem functions_in_M :
  belongs_to_M f2 ∧ belongs_to_M f4 ∧ ¬belongs_to_M f1 ∧ ¬belongs_to_M f3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_in_M_l908_90820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_ratio_l908_90801

theorem sphere_volume_ratio (r₁ r₂ r₃ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₃ > 0) 
  (h_ratio : (r₁, r₂, r₃) = (1, 2, 3)) :
  (4/3 * Real.pi * r₃^3) = 3 * ((4/3 * Real.pi * r₁^3) + (4/3 * Real.pi * r₂^3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_volume_ratio_l908_90801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l908_90871

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 6) * Real.sin (2 * x) - 1 / 4

theorem symmetry_center_of_f :
  ∃ (k : ℤ), (∀ (x : ℝ), f (7 * Real.pi / 24 + x) = -f (7 * Real.pi / 24 - x)) ∧
  (∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧ ∀ (x : ℝ), |x| < δ → |f (7 * Real.pi / 24 + x) + f (7 * Real.pi / 24 - x)| < ε) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_f_l908_90871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_l_is_50_l908_90890

/-- The speed of vehicle l in km/h -/
noncomputable def speed_l : ℝ := sorry

/-- The speed of vehicle k in km/h -/
noncomputable def speed_k : ℝ := sorry

/-- The time l travels in hours -/
noncomputable def time_l : ℝ := sorry

/-- The time k travels in hours -/
noncomputable def time_k : ℝ := sorry

/-- The total distance between l and k in km -/
noncomputable def total_distance : ℝ := sorry

theorem speed_of_l_is_50 
  (h1 : speed_k = 1.5 * speed_l)  -- k is 50% faster than l
  (h2 : time_l = 3)               -- l travels for 3 hours (9 a.m. to 12 p.m.)
  (h3 : time_k = 2)               -- k travels for 2 hours (10 a.m. to 12 p.m.)
  (h4 : total_distance = 300)     -- l and k are 300 kms apart
  (h5 : speed_l * time_l + speed_k * time_k = total_distance)  -- They meet when traveling in opposite directions
  : speed_l = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_of_l_is_50_l908_90890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_l908_90879

/-- The radius of the bus wheel in centimeters -/
def wheel_radius : ℚ := 175

/-- The speed of the bus in kilometers per hour -/
def bus_speed : ℚ := 66

/-- The number of centimeters in a kilometer -/
def cm_per_km : ℚ := 100000

/-- The number of minutes in an hour -/
def minutes_per_hour : ℚ := 60

/-- The approximate value of pi -/
def π_approx : ℚ := 314159 / 100000

/-- Calculates the revolutions per minute of the bus wheel -/
def revolutions_per_minute (r : ℚ) (v : ℚ) : ℚ :=
  let circumference := 2 * π_approx * r
  let speed_cm_per_minute := v * cm_per_km / minutes_per_hour
  speed_cm_per_minute / circumference

/-- Theorem stating that the revolutions per minute of the bus wheel is approximately 100 -/
theorem bus_wheel_rpm :
  ∃ ε : ℚ, ε > 0 ∧ |revolutions_per_minute wheel_radius bus_speed - 100| < ε :=
by
  sorry

#eval revolutions_per_minute wheel_radius bus_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_wheel_rpm_l908_90879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_sales_calculation_l908_90821

def initial_stock : ℕ := 900
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def friday_sales : ℕ := 135
def unsold_percentage : ℚ := 55333333333333336 / 100000000000000000

theorem thursday_sales_calculation :
  ∃ (thursday_sales : ℕ),
    thursday_sales = initial_stock - 
    (monday_sales + tuesday_sales + wednesday_sales + friday_sales + 
    (unsold_percentage * initial_stock).floor) ∧
    thursday_sales = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thursday_sales_calculation_l908_90821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_max_a_for_increasing_g_l908_90850

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.log x + 2 * x - 3

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x - a) * Real.log x + a * (x - 1) / x

-- Statement for part I
theorem f_has_unique_zero :
  ∃! x : ℝ, x ≥ 1 ∧ f x = 0 :=
sorry

-- Statement for part II
theorem max_a_for_increasing_g :
  (∀ x : ℝ, x ≥ 1 → Monotone (g 6)) ∧
  ¬(∃ a : ℤ, a > 6 ∧ (∀ x : ℝ, x ≥ 1 → Monotone (g (↑a)))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_max_a_for_increasing_g_l908_90850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_60_deg_radius_12_l908_90826

/-- The arc length of a circle segment given its radius and central angle in degrees -/
noncomputable def arcLength (radius : ℝ) (centralAngleDeg : ℝ) : ℝ :=
  radius * (centralAngleDeg * (Real.pi / 180))

theorem arc_length_60_deg_radius_12 :
  arcLength 12 60 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_60_deg_radius_12_l908_90826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_abs_abs_minus_x_l908_90818

theorem abs_abs_abs_minus_x (x : ℤ) (h : x = -2023) :
  (|(|x - x| - |x|)| - x) = 4046 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_abs_abs_minus_x_l908_90818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_for_inequality_l908_90888

theorem theta_range_for_inequality (θ : ℝ) : 
  (∀ x ∈ Set.Icc (0 : ℝ) 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔ 
  ∃ k : ℤ, θ ∈ Set.Ioo ((2 * k : ℝ) * π + π / 12) ((2 * k : ℝ) * π + 5 * π / 12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_range_for_inequality_l908_90888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_lambda_range_l908_90839

/-- A function that is symmetric about the origin -/
def SymmetricAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function that is increasing on an interval -/
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

/-- The theorem statement -/
theorem symmetric_functions_lambda_range :
  ∀ (f g : ℝ → ℝ) (lambda : ℝ),
  SymmetricAboutOrigin f →
  SymmetricAboutOrigin g →
  (∀ x, f x = x^2 + 2*x) →
  let h := fun x ↦ g x - lambda * f x + 1
  IncreasingOn h (-1) 1 ↔ lambda ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_functions_lambda_range_l908_90839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_2_l908_90862

noncomputable def f (ω b : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4) + b

noncomputable def T (ω : ℝ) : ℝ := 2 * Real.pi / ω

theorem f_value_at_pi_over_2 (ω b : ℝ) :
  ω > 0 →
  2 * Real.pi / 3 < T ω →
  T ω < Real.pi →
  (∀ x, f ω b (x + 3 * Real.pi / 2) = 4 - f ω b x) →
  f ω b (Real.pi / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_pi_over_2_l908_90862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_value_l908_90870

theorem cos_two_alpha_value (α β : ℝ) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.tan β = 4/3 →
  Real.cos (α + β) = 0 →
  Real.cos (2*α) = 7/25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_alpha_value_l908_90870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l908_90819

/-- The time taken for three workers to complete a job together, given their individual completion times. -/
noncomputable def combined_time (t_a t_b t_c : ℝ) : ℝ :=
  1 / (1 / t_a + 1 / t_b + 1 / t_c)

/-- Theorem stating that workers with individual completion times of 10, 12, and 15 hours
    will take 4 hours to complete the job when working together. -/
theorem workers_combined_time :
  combined_time 10 12 15 = 4 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_workers_combined_time_l908_90819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_reciprocal_lengths_l908_90893

-- Define the ellipse Γ
def Γ : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1}

-- Define the left focus F₁
def F₁ : ℝ × ℝ := (-1, 0)

-- Define a function to calculate the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem constant_sum_of_reciprocal_lengths :
  ∀ (A B C D : ℝ × ℝ),
  A ∈ Γ → B ∈ Γ → C ∈ Γ → D ∈ Γ →
  (A.1 - F₁.1) * (C.1 - F₁.1) + (A.2 - F₁.2) * (C.2 - F₁.2) = 0 →
  (B.1 - F₁.1) * (D.1 - F₁.1) + (B.2 - F₁.2) * (D.2 - F₁.2) = 0 →
  1 / distance A B + 1 / distance C D = (3/4) * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_of_reciprocal_lengths_l908_90893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_a_range_l908_90866

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.exp x + 1

-- Theorem for the maximum value of f
theorem f_max_value :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 2 * Real.log 2 - 1 :=
sorry

-- Theorem for the range of a
theorem a_range (x : ℝ) (hx : x ∈ Set.Ioo 0 1) :
  ∀ (a : ℝ), (a * f x < Real.tan x) ↔ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_a_range_l908_90866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tetrahedron_volume_l908_90858

/-- A cube with alternately colored vertices -/
structure ColoredCube where
  sideLength : ℝ
  vertexColors : Fin 8 → Bool  -- True for blue, False for red

/-- The volume of a tetrahedron given its base area and height -/
noncomputable def tetrahedronVolume (baseArea height : ℝ) : ℝ := (1/3) * baseArea * height

/-- The theorem stating the volume of the blue tetrahedron in the colored cube -/
theorem blue_tetrahedron_volume (cube : ColoredCube) 
  (h1 : cube.sideLength = 10)
  (h2 : ∀ (face : Fin 6), ∃ (v1 v2 : Fin 8), 
    v1 ≠ v2 ∧ cube.vertexColors v1 ≠ cube.vertexColors v2) :
  ∃ (blueTetrahedron : Finset (Fin 8)),
    (∀ v ∈ blueTetrahedron, cube.vertexColors v = true) ∧
    (blueTetrahedron.card = 4) ∧
    (tetrahedronVolume ((cube.sideLength^2) / 2) cube.sideLength = 1000 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_tetrahedron_volume_l908_90858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marshmallows_l908_90894

def marshmallow_challenge (haley michael brandon sofia lucas : ℕ) : Prop :=
  haley = 8 ∧
  michael = 3 * haley ∧
  brandon = michael / 2 ∧
  sofia = 2 * (haley + brandon) ∧
  lucas = (haley + michael + brandon) / 3 + Int.floor (Real.sqrt (sofia : ℝ))

theorem total_marshmallows :
  ∀ haley michael brandon sofia lucas,
    marshmallow_challenge haley michael brandon sofia lucas →
    haley + michael + brandon + sofia + lucas = 104 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_marshmallows_l908_90894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_S_value_l908_90842

noncomputable def P (t : ℝ) (X : ℝ) : ℝ :=
  (1/t) * X^4 + (1 - 10/t) * X^3 - 2 * X^2 + (2*t)^(1/3) * X + Real.arctan t

-- Define S directly as a function of t
def S (t : ℝ) : ℝ :=
  t^2 - 16*t + 100

-- State the theorem
theorem min_S_value (t : ℝ) (h : t > 0) : 
  |S t| ≥ 36 ∧ (t = 8 → |S t| = 36) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_S_value_l908_90842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l908_90896

noncomputable def A : ℝ × ℝ := (3, 15)
noncomputable def B : ℝ × ℝ := (15, 0)
noncomputable def C (p : ℝ) : ℝ × ℝ := (0, p)

noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((P.1 - R.1) * (Q.2 - R.2) - (Q.1 - R.1) * (P.2 - R.2))

theorem triangle_abc_area (p : ℝ) :
  triangleArea A B (C p) = 36 → p = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_area_l908_90896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_sum_l908_90802

theorem cosine_power_sum (n : ℕ) (x θ : ℝ) (h : x + x⁻¹ = 2 * Real.cos θ) :
  x^n + (x⁻¹)^n = 2 * Real.cos (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_sum_l908_90802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l908_90895

/-- The function f(x) as described in the problem -/
noncomputable def f (w φ x : ℝ) : ℝ := 2 * Real.sin (w * x + φ) + 1

/-- The theorem statement -/
theorem phi_range (w φ : ℝ) :
  w > 0 →
  |φ| ≤ π / 2 →
  (∃ x₁ x₂, x₂ - x₁ = π ∧ f w φ x₁ = 3 ∧ f w φ x₂ = 3) →
  (∀ x, π / 24 < x → x < π / 3 → f w φ x > 2) →
  π / 12 ≤ φ ∧ φ ≤ π / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l908_90895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_common_denominator_l908_90828

-- Define the fractions
noncomputable def f1 (x : ℝ) : ℝ := 2 * x / (x - 2)
noncomputable def f2 (x : ℝ) : ℝ := 3 / (x^2 - 2*x)

-- Define the proposed common denominator
noncomputable def common_denominator (x : ℝ) : ℝ := x * (x - 2)

-- Statement to prove
theorem simplest_common_denominator (x : ℝ) (h : x ≠ 2 ∧ x ≠ 0) : 
  ∃ (k1 k2 : ℝ), f1 x = k1 / common_denominator x ∧ 
                 f2 x = k2 / common_denominator x ∧
                 ∀ (d : ℝ), (∃ (m1 m2 : ℝ), f1 x = m1 / d ∧ f2 x = m2 / d) → 
                   ∃ (c : ℝ), d = c * common_denominator x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_common_denominator_l908_90828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_sine_cosine_product_l908_90864

/-- The function f(x) = a^(x+1) + 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x+1) + 1

/-- The theorem stating the relationship between the function and sin α cos α -/
theorem fixed_point_sine_cosine_product (a : ℝ) (α : ℝ) :
  a > 0 ∧ a ≠ 1 ∧ 
  f a (-1) = 2 ∧
  (∃ (r : ℝ), r > 0 ∧ r * (Real.cos α) = -1 ∧ r * (Real.sin α) = 2) →
  Real.sin α * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_sine_cosine_product_l908_90864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_and_inequality_range_l908_90853

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := ((x - 1) / (x + 1)) ^ 2

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := (Real.sqrt x + 1) / (1 - Real.sqrt x)

-- State the theorem
theorem inverse_function_and_inequality_range :
  (∀ x > 1, f x = ((x - 1) / (x + 1)) ^ 2) →
  (∀ x ∈ Set.Ioo 0 1, f_inv x = (Real.sqrt x + 1) / (1 - Real.sqrt x)) →
  (∀ x ∈ Set.Icc (1/4) (1/2), ∀ a : ℝ, 
    (1 - Real.sqrt x) * f_inv x > a * (a - Real.sqrt x)) →
  ∃ a_min a_max : ℝ, a_min = -1 ∧ a_max = 3/2 ∧ 
    ∀ a : ℝ, a_min < a ∧ a < a_max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_and_inequality_range_l908_90853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_depth_is_five_l908_90880

/-- The depth of a rectangular pond given its length, width, and volume. -/
noncomputable def pondDepth (length width volume : ℝ) : ℝ :=
  volume / (length * width)

/-- Theorem stating that the depth of a rectangular pond with given dimensions is 5 meters. -/
theorem pond_depth_is_five :
  let length : ℝ := 20
  let width : ℝ := 12
  let volume : ℝ := 1200
  pondDepth length width volume = 5 := by
  -- Unfold the definition of pondDepth
  unfold pondDepth
  -- Simplify the expression
  simp
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_depth_is_five_l908_90880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_quadratic_l908_90830

theorem min_value_quadratic :
  ∀ (x y : ℝ), y = x^2 + 16*x + 20 → y ≥ -44 ∧ ∃ x₀ : ℝ, x₀^2 + 16*x₀ + 20 = -44 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_quadratic_l908_90830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_square_specific_chord_length_square_l908_90835

/-- Given two circles with radii r₁ and r₂, whose centers are d units apart,
    if a line is drawn through their intersection point P such that it creates
    equal chords QP and PR in each circle, then the square of the length of QP
    is approximately (200 + 200 * (d² - r₁² - r₂²) / (2 * r₁ * r₂)). -/
theorem chord_length_square (r₁ r₂ d : ℝ) (hr₁ : r₁ > 0) (hr₂ : r₂ > 0) (hd : d > 0) :
  let x := (200 + 200 * (d^2 - r₁^2 - r₂^2) / (2 * r₁ * r₂))
  ∃ (QP : ℝ), abs (QP^2 - x) < 1 ∧
    ∃ (P : ℝ × ℝ), ∃ (O₁ O₂ : ℝ × ℝ),
      ‖O₁ - P‖ = r₁ ∧
      ‖O₂ - P‖ = r₂ ∧
      ‖O₁ - O₂‖ = d ∧
      ∃ (Q R : ℝ × ℝ), ‖Q - P‖ = ‖R - P‖ ∧ abs (‖Q - P‖^2 - x) < 1 :=
by sorry

/-- For the specific case where r₁ = 10, r₂ = 7, and d = 15,
    the square of the chord length is approximately 309. -/
theorem specific_chord_length_square :
  ∃ (QP : ℝ), abs (QP^2 - 309) < 1 ∧
    ∃ (P : ℝ × ℝ), ∃ (O₁ O₂ : ℝ × ℝ),
      ‖O₁ - P‖ = 10 ∧
      ‖O₂ - P‖ = 7 ∧
      ‖O₁ - O₂‖ = 15 ∧
      ∃ (Q R : ℝ × ℝ), ‖Q - P‖ = ‖R - P‖ ∧ abs (‖Q - P‖^2 - 309) < 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_square_specific_chord_length_square_l908_90835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_shower_heads_l908_90854

/-- The number of students using the shower room. -/
def num_students : ℕ := 100

/-- The preheating time per shower head in minutes. -/
def preheating_time : ℕ := 3

/-- The allocated showering time for each group in minutes. -/
def showering_time : ℕ := 12

/-- The total heating time as a function of the number of shower heads. -/
noncomputable def total_heating_time (x : ℕ) : ℝ :=
  (preheating_time : ℝ) * x + showering_time * (num_students : ℝ) / x

/-- The optimal number of shower heads that minimizes the total heating time. -/
def optimal_showers : ℕ := 20

/-- Theorem stating that the optimal number of shower heads minimizes the total heating time. -/
theorem optimal_shower_heads :
  ∀ x : ℕ, x > 0 → x ≠ optimal_showers →
    total_heating_time x > total_heating_time optimal_showers :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_shower_heads_l908_90854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_given_condition_l908_90805

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then
    (3 * (1 - 2)^x) / (2^x + 1)
  else
    -(1/4) * (x^3 + 3*x)

theorem range_of_x_given_condition (h : ∀ (x m : ℝ), -3 ≤ m ∧ m ≤ 2 → f (m*x - 1) + f x > 0) :
  ∃ (a b : ℝ), a = -1/2 ∧ b = 1/3 ∧ ∀ (x : ℝ), a < x ∧ x < b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_given_condition_l908_90805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_vector_ratio_l908_90891

-- For statement B
theorem vector_parallel (a b : ℝ × ℝ) :
  a = (1, 3) ∧ a - b = (-1, -3) → ∃ k : ℝ, b = k • a :=
sorry

-- For statement C
theorem vector_ratio (O A B C : ℝ × ℝ) :
  O ≠ A ∧ O ≠ B ∧ O ≠ C ∧ ¬(∃ t : ℝ × ℝ → ℝ, t O = 0 ∧ t A = t B ∧ t B = t C) ∧
  (A - O) - 3 • (B - O) + 2 • (C - O) = (0, 0) →
  ‖A - B‖ / ‖B - C‖ = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_vector_ratio_l908_90891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_groups_l908_90849

/-- The probability of two specific students being in different groups when six students are evenly divided into two groups -/
theorem probability_different_groups (n k : ℕ) : 
  n = 6 → k = 2 → (Nat.choose n k * Nat.choose (n - k) k) / (Nat.choose n (n / 2)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_groups_l908_90849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_diagonal_l908_90899

theorem rectangle_area_diagonal (length width perimeter : ℝ) 
  (h1 : length / width = 5 / 2) 
  (h2 : 2 * (length + width) = perimeter) 
  (h3 : perimeter = 42) : 
  (length * width) / (length^2 + width^2) = 10 / 29 := by
  sorry

#check rectangle_area_diagonal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_diagonal_l908_90899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l908_90837

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x + a
noncomputable def h (b : ℝ) (x : ℝ) : ℝ := x + 25/x + b

-- Theorem for part (1)
theorem part_one (a : ℝ) :
  (∀ x : ℝ, g a x ≠ 0) → a > 4 :=
by sorry

-- Theorem for part (2)
theorem part_two (b : ℝ) :
  (∀ x₁ : ℝ, x₁ ∈ Set.Icc 1 4 → ∃ x₂ : ℝ, x₂ ∈ Set.Ioo 1 5 ∧ f x₁ = h b x₂) →
  -21 < b ∧ b ≤ -14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l908_90837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_given_parabola_l908_90836

/-- A parabola is defined by its equation y = ax^2 + c -/
structure Parabola where
  a : ℝ
  c : ℝ

/-- The directrix of a parabola is a line parallel to its axis of symmetry -/
noncomputable def directrix (p : Parabola) : ℝ := -1 / (4 * p.a) + p.c

/-- The given parabola y = 4x^2 + 4 -/
def given_parabola : Parabola := { a := 4, c := 4 }

/-- Theorem: The directrix of the parabola y = 4x^2 + 4 is y = 63/16 -/
theorem directrix_of_given_parabola :
  directrix given_parabola = 63 / 16 := by
  -- Unfold the definitions
  unfold directrix
  unfold given_parabola
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_directrix_of_given_parabola_l908_90836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_star_more_in_row_than_col_l908_90877

/-- Represents a cell in the grid -/
structure Cell (M N : ℕ) where
  row : Fin M
  col : Fin N

/-- Represents the grid with stars -/
structure StarGrid (M N : ℕ) where
  hasStarRed : Cell M N → Bool
  rowHasStar : ∀ r, ∃ c, hasStarRed ⟨r, c⟩
  colHasStar : ∀ c, ∃ r, hasStarRed ⟨r, c⟩

/-- Count stars in a row -/
def countStarsInRow (grid : StarGrid M N) (r : Fin M) : ℕ :=
  (Finset.filter (fun c => grid.hasStarRed ⟨r, c⟩) (Finset.univ : Finset (Fin N))).card

/-- Count stars in a column -/
def countStarsInCol (grid : StarGrid M N) (c : Fin N) : ℕ :=
  (Finset.filter (fun r => grid.hasStarRed ⟨r, c⟩) (Finset.univ : Finset (Fin M))).card

/-- Main theorem -/
theorem exists_star_more_in_row_than_col {M N : ℕ} (h : N > M) (grid : StarGrid M N) :
  ∃ cell : Cell M N, grid.hasStarRed cell ∧ countStarsInRow grid cell.row > countStarsInCol grid cell.col := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_star_more_in_row_than_col_l908_90877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l908_90852

theorem polynomial_factorization :
  ∀ x : Polynomial ℤ,
  x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l908_90852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_happiness_survey_l908_90869

theorem happiness_survey (n k : ℕ) (h_n : n > 0) (h_k : k > 0) :
  let answers : Fin k → Fin n → Bool := λ _ _ ↦ sorry
  let same_answer (d₁ d₂ : Fin k) := 
    (Finset.filter (λ i ↦ answers d₁ i = answers d₂ i) (Finset.univ : Finset (Fin n))).card = n / 2
  let equal_yes_no (p : Fin n) := 
    (Finset.filter (λ d ↦ answers d p) (Finset.univ : Finset (Fin k))).card = k / 2
  ∀ d₁ d₂, d₁ ≠ d₂ → same_answer d₁ d₂ →
    (Finset.filter equal_yes_no (Finset.univ : Finset (Fin n))).card ≤ n - n / k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_happiness_survey_l908_90869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_max_value_l908_90885

theorem trigonometric_sum_max_value :
  ∃ (M : ℝ), M = 3 ∧ 
  ∀ (θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ),
    Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + Real.cos θ₃ * Real.sin θ₄ + 
    Real.cos θ₄ * Real.sin θ₅ + Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁ ≤ M ∧
  ∃ (θ₁' θ₂' θ₃' θ₄' θ₅' θ₆' : ℝ),
    Real.cos θ₁' * Real.sin θ₂' + Real.cos θ₂' * Real.sin θ₃' + Real.cos θ₃' * Real.sin θ₄' + 
    Real.cos θ₄' * Real.sin θ₅' + Real.cos θ₅' * Real.sin θ₆' + Real.cos θ₆' * Real.sin θ₁' = M :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_max_value_l908_90885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l908_90827

/-- Distance formula from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (A B C x₀ y₀ : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- Distance formula from a point (x₀, y₀, z₀) to a plane Ax + By + Cz + D = 0 -/
noncomputable def distance_point_to_plane (A B C D x₀ y₀ z₀ : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C * z₀ + D| / Real.sqrt (A^2 + B^2 + C^2)

theorem distance_to_plane (x₀ y₀ z₀ : ℝ) (h : x₀ = 0 ∧ y₀ = 1 ∧ z₀ = 3) :
  distance_point_to_plane 1 2 3 3 x₀ y₀ z₀ = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l908_90827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lineA_has_largest_inclination_l908_90857

-- Define a structure for a line
structure Line where
  slope : Option ℝ
  intercept : ℝ

-- Define the angle of inclination function
noncomputable def angleOfInclination (l : Line) : ℝ :=
  match l.slope with
  | none => Real.pi / 2  -- 90° for vertical lines
  | some m => 
      if m < 0 then
        Real.pi - Real.arctan (-m)  -- For negative slopes
      else
        Real.arctan m  -- For non-negative slopes

-- Define the four lines
def lineA : Line := { slope := some (-1), intercept := 1 }
def lineB : Line := { slope := some 1, intercept := 1 }
def lineC : Line := { slope := some 2, intercept := 1 }
def lineD : Line := { slope := none, intercept := 1 }

-- State the theorem
theorem lineA_has_largest_inclination :
  ∀ l ∈ [lineA, lineB, lineC, lineD], 
    angleOfInclination lineA ≥ angleOfInclination l := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lineA_has_largest_inclination_l908_90857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l908_90800

theorem tan_double_angle (α : Real) :
  (Real.sin (5 * Real.pi / 6), Real.cos (5 * Real.pi / 6)) = (Real.sin α, Real.cos α) →
  Real.tan (2 * α) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l908_90800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l908_90848

noncomputable def f (a ω x : ℝ) : ℝ := 2 * a * Real.sin (ω * x) * Real.cos (ω * x) + 2 * Real.sqrt 3 * (Real.cos (ω * x))^2 - Real.sqrt 3

theorem function_properties (a ω : ℝ) (ha : a > 0) (hω : ω > 0) 
  (hmax : ∀ x, f a ω x ≤ 2) 
  (hperiod : ∀ x, f a ω (x + π) = f a ω x) 
  (hsmallest_period : ∀ p, p > 0 ∧ (∀ x, f a ω (x + p) = f a ω x) → p ≥ π) :
  (∀ x, f a ω x = 2 * Real.sin (2 * x + π / 3)) ∧
  (∀ k, ∃ x, x = π / 12 + k * π / 2 ∧ ∀ y, f a ω (x + y) = f a ω (x - y)) ∧
  (∀ k x, k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12 → 
    ∀ y, x ≤ y ∧ y ≤ k * π + π / 12 → f a ω x ≤ f a ω y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l908_90848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_a5_b5_l908_90872

/-- Two arithmetic sequences satisfying the given condition -/
def arithmetic_sequences (a b : ℕ → ℚ) : Prop :=
  ∀ n : ℕ+, (Finset.range n).sum a / (Finset.range n).sum b = (7 * n + 1) / (n + 2)

/-- Theorem stating that for arithmetic sequences satisfying the condition, a₅ / b₅ = 64/11 -/
theorem ratio_a5_b5 (a b : ℕ → ℚ) (h : arithmetic_sequences a b) : 
  a 5 / b 5 = 64/11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_a5_b5_l908_90872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisor_theorem_l908_90860

theorem common_divisor_theorem (k n : ℕ) 
  (h1 : (k : ℤ) - n > 1 ∨ (n : ℤ) - k > 1) 
  (h2 : ∃ m : ℤ, 4 * k * n + 1 = (k + n) * m) : 
  ∃ d : ℕ, d > 1 ∧ d ∣ (2 * n - 1) ∧ d ∣ (2 * k + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_divisor_theorem_l908_90860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_seven_l908_90897

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the main theorem
theorem f_of_three_equals_seven :
  (∀ x : ℝ, x ≠ 0 → f (x + 1/x) = x^2 + 1/x^2) →
  f 3 = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_seven_l908_90897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_l908_90881

noncomputable def f (x : ℝ) : ℝ := (Real.cos (2 * x)) / (Real.sin x + Real.cos x)

theorem f_domain_and_range :
  (∀ x : ℝ, f x ≠ 0 → ∃ k : ℤ, x ≠ k * π - π / 4) ∧
  (∀ y : ℝ, y ∈ Set.Ioo (-Real.sqrt 2) (Real.sqrt 2) → ∃ x : ℝ, f x = y) ∧
  (∀ x : ℝ, f x ∈ Set.Ioo (-Real.sqrt 2) (Real.sqrt 2)) :=
by
  sorry

#check f_domain_and_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_l908_90881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solvability_l908_90863

theorem equation_solvability (n : ℕ) :
  (∃ (a b : ℤ), 3 * a^2 - b^2 = (2018 : ℤ)^n) ↔ n % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solvability_l908_90863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_out_of_place_white_cells_l908_90813

/-- Represents a cell on the board -/
structure Cell where
  x : Fin 10
  y : Fin 10
  color : Bool  -- true for white, false for black

/-- The board is a 10x10 grid of cells -/
def Board := Array (Array Cell)

/-- Checks if two cells are neighbors -/
def isNeighbor (c1 c2 : Cell) : Bool :=
  (c1.x = c2.x && (c1.y = c2.y + 1 || c1.y + 1 = c2.y)) ||
  ((c1.x = c2.x + 1 || c1.x + 1 = c2.x) && c1.y = c2.y) ||
  ((c1.x = c2.x + 1 || c1.x + 1 = c2.x) && (c1.y = c2.y + 1 || c1.y + 1 = c2.y))

/-- Checks if a cell is out of place -/
def isOutOfPlace (board : Board) (cell : Cell) : Bool :=
  let differentColorNeighbors := board.foldl (fun count row =>
    count + row.foldl (fun innerCount c =>
      if isNeighbor cell c && c.color ≠ cell.color then innerCount + 1 else innerCount
    ) 0
  ) 0
  differentColorNeighbors ≥ 7

/-- Counts the number of out of place white cells -/
def countOutOfPlaceWhiteCells (board : Board) : Nat :=
  board.foldl (fun count row =>
    count + row.foldl (fun innerCount cell =>
      if cell.color && isOutOfPlace board cell then innerCount + 1 else innerCount
    ) 0
  ) 0

/-- The main theorem stating the maximum number of out of place white cells -/
theorem max_out_of_place_white_cells (board : Board) :
  countOutOfPlaceWhiteCells board ≤ 26 := by
  sorry

#check max_out_of_place_white_cells

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_out_of_place_white_cells_l908_90813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l908_90803

noncomputable def m : ℝ × ℝ := (Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

theorem vector_problem (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (m.1 * (n x).1 + m.2 * (n x).2 = 0 → Real.tan x = 1) ∧
  (Real.cos (Real.pi / 3) = (m.1 * (n x).1 + m.2 * (n x).2) / (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt ((n x).1^2 + (n x).2^2)) → x = 5 * Real.pi / 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l908_90803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_ratio_and_hcf_l908_90807

theorem lcm_of_ratio_and_hcf (a b : ℕ) (h_ratio : 4 * a = 3 * b)
  (h_hcf : Nat.gcd a b = 3) : Nat.lcm a b = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_ratio_and_hcf_l908_90807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_deriv_2005_l908_90865

open Real

/-- The sequence of functions defined by repeated differentiation of sin x -/
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => sin
  | n + 1 => λ x => deriv (f n) x

/-- Theorem stating that the 2005th derivative of sin x is cos x -/
theorem sin_deriv_2005 : f 2005 = cos := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_deriv_2005_l908_90865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_l908_90887

-- Define the fuel cost function
noncomputable def fuel_cost (v : ℝ) : ℝ := 35 * (v / 10) ^ 3

-- Define the total cost function
noncomputable def total_cost (v : ℝ) : ℝ := fuel_cost v + 560

-- Define the cost per kilometer function
noncomputable def cost_per_km (v : ℝ) : ℝ := total_cost v / v

-- Theorem statement
theorem optimal_speed :
  ∃ (v : ℝ), v > 0 ∧ v ≤ 25 ∧
  ∀ (u : ℝ), u > 0 → u ≤ 25 → cost_per_km v ≤ cost_per_km u ∧
  v = 20 := by
  sorry

#check optimal_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_l908_90887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sergios_farm_price_per_kg_l908_90831

/-- Represents the production and sale of fruits on Mr. Sergio's farm -/
structure FruitFarm where
  mango_produce : ℕ
  apple_produce : ℕ
  orange_produce : ℕ
  total_amount : ℕ

/-- The conditions of Mr. Sergio's fruit farm -/
def sergios_farm : FruitFarm where
  mango_produce := 400
  apple_produce := 2 * 400
  orange_produce := 400 + 200
  total_amount := 90000

/-- The price per kg of fruits on Mr. Sergio's farm -/
noncomputable def price_per_kg (farm : FruitFarm) : ℚ :=
  farm.total_amount / (farm.mango_produce + farm.apple_produce + farm.orange_produce)

/-- Theorem stating that the price per kg of fruits on Mr. Sergio's farm is $50 -/
theorem sergios_farm_price_per_kg :
  price_per_kg sergios_farm = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sergios_farm_price_per_kg_l908_90831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_min_value_equality_x_range_l908_90883

/-- The minimum value of the given expression is 6 -/
theorem min_value_theorem (a b : ℝ) (ha : a ≠ 0) :
  (|3*a + 2*b| + |3*a - 2*b|) / |a| ≥ 6 :=
sorry

/-- The equality case for the minimum value -/
theorem min_value_equality (a b : ℝ) (ha : a ≠ 0) :
  (|3*a + 2*b| + |3*a - 2*b|) / |a| = 6 ↔ (3*a + 2*b) * (3*a - 2*b) ≥ 0 :=
sorry

/-- The range of x given the inequality -/
theorem x_range (x : ℝ) :
  (∀ (a b : ℝ) (ha : a ≠ 0), |3*a + 2*b| + |3*a - 2*b| ≥ |a| * (|2 + x| + |2 - x|)) ↔
  x ∈ Set.Icc (-3) 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_min_value_equality_x_range_l908_90883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_chinese_sector_area_l908_90806

/-- Represents a circular sector --/
structure CircularSector where
  circumference : ℝ
  diameter : ℝ

/-- Calculates the area of a circular sector --/
noncomputable def sectorArea (s : CircularSector) : ℝ :=
  (1/2) * s.circumference * (s.diameter / 2)

/-- Theorem stating that a circular sector with circumference 30 and diameter 16 has an area of 120 --/
theorem ancient_chinese_sector_area :
  let s : CircularSector := { circumference := 30, diameter := 16 }
  sectorArea s = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_chinese_sector_area_l908_90806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l908_90834

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_arithmetic (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_10 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arithmetic : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_condition : a 4^2 + a 5^2 = a 6^2 + a 7^2) :
  sum_arithmetic a 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_10_l908_90834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_approximation_l908_90840

/-- Given a number line with specific properties, prove that C best approximates x^2 -/
theorem x_squared_approximation
  (x : ℝ)
  (A B C D E : ℝ)
  (h1 : -1 < x)
  (h2 : x < 0)
  (h3 : 0 < C)
  (h4 : C < 1)
  (h5 : A < B ∧ B < C ∧ C < D ∧ D < E) :
  ∀ y ∈ ({A, B, D, E} : Set ℝ), |x^2 - C| ≤ |x^2 - y| :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_approximation_l908_90840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_l908_90811

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 4*x + 4

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 4

-- Theorem statement
theorem f_extrema_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 28/3 ∧ min = -4/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_on_interval_l908_90811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_perpendicular_l908_90856

-- Define the points
variable (A B C D A₁ B₁ C₁ D₁ P Q R S : ℝ × ℝ)

-- Define the squares
def is_square (A B C D : ℝ × ℝ) : Prop :=
  ∃ (side : ℝ), side > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = side^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = side^2 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = side^2 ∧
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = side^2

-- Define equally oriented squares
noncomputable def equally_oriented (A B C D A₁ B₁ C₁ D₁ : ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), 
    B.1 - A.1 = (B₁.1 - A₁.1) * Real.cos θ - (B₁.2 - A₁.2) * Real.sin θ ∧
    B.2 - A.2 = (B₁.1 - A₁.1) * Real.sin θ + (B₁.2 - A₁.2) * Real.cos θ

-- Define perpendicular bisector
def perp_bisector (A B M : ℝ × ℝ) : Prop :=
  (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) ∧
  (B.1 - A.1) * (M.1 - A.1) + (B.2 - A.2) * (M.2 - A.2) = 0

-- Define perpendicularity
def perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (R.1 - P.1) * (S.1 - Q.1) + (R.2 - P.2) * (S.2 - Q.2) = 0

-- Theorem statement
theorem intersection_points_perpendicular 
  (h1 : is_square A B C D) 
  (h2 : is_square A₁ B₁ C₁ D₁)
  (h3 : equally_oriented A B C D A₁ B₁ C₁ D₁)
  (h4 : ∃ M₁, perp_bisector A A₁ M₁ ∧ perp_bisector A B P)
  (h5 : ∃ M₂, perp_bisector B B₁ M₂ ∧ perp_bisector B C Q)
  (h6 : ∃ M₃, perp_bisector C C₁ M₃ ∧ perp_bisector C D R)
  (h7 : ∃ M₄, perp_bisector D D₁ M₄ ∧ perp_bisector D A S) :
  perpendicular P Q R S :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_perpendicular_l908_90856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l908_90844

-- Define the function type
def PositiveRealFunction := ℝ → ℝ

-- Define the conditions
def satisfies_condition1 (f : PositiveRealFunction) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → f (x * f y) = y * f x

def satisfies_condition2 (f : PositiveRealFunction) : Prop :=
  Filter.Tendsto f Filter.atTop (nhds 0)

-- State the theorem
theorem unique_function_satisfying_conditions
  (f : PositiveRealFunction)
  (h1 : satisfies_condition1 f)
  (h2 : satisfies_condition2 f) :
  ∀ x : ℝ, x > 0 → f x = 1 / x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l908_90844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_set_l908_90816

noncomputable def median (s : Finset ℝ) : ℝ := sorry

theorem median_of_set (a : ℤ) (b : ℝ) (h1 : a ≠ 0) (h2 : b > 0) (h3 : a * b^3 = Real.log b / Real.log 10) :
  median {0, 1, (a : ℝ), b, b^2} = b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_set_l908_90816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_BC_l908_90873

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the triangle ABC
structure Triangle where
  B : ℝ × ℝ
  C : ℝ × ℝ

def triangleABC (t : Triangle) : Prop :=
  -- A is at the origin
  (parabola 0 = 0) ∧
  -- B and C are on the parabola
  (t.B.2 = parabola t.B.1) ∧
  (t.C.2 = parabola t.C.1) ∧
  -- BC is parallel to x-axis
  (t.B.2 = t.C.2) ∧
  -- Area of triangle is 128
  (abs ((t.C.1 - t.B.1) * t.B.2 / 2) = 128)

-- Theorem statement
theorem length_of_BC (t : Triangle) (h : triangleABC t) : 
  abs (t.C.1 - t.B.1) = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_BC_l908_90873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l908_90845

/-- Represents a rhombus with given side length and shorter diagonal -/
structure Rhombus where
  side_length : ℝ
  shorter_diagonal : ℝ

/-- Calculates the length of the longer diagonal of a rhombus -/
noncomputable def longer_diagonal (r : Rhombus) : ℝ :=
  2 * Real.sqrt (r.side_length ^ 2 - (r.shorter_diagonal / 2) ^ 2)

/-- Theorem: In a rhombus with side length 65 and shorter diagonal 72, the longer diagonal is 108 -/
theorem rhombus_longer_diagonal :
  let r : Rhombus := { side_length := 65, shorter_diagonal := 72 }
  longer_diagonal r = 108 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_longer_diagonal_l908_90845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l908_90824

theorem problem_solution : 
  ((Real.sqrt 3 - 2)^2 + Real.sqrt 27 = 7 - Real.sqrt 3) ∧
  ((Real.sqrt 6 - 2) * (Real.sqrt 6 + 2) - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l908_90824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_difference_is_fifty_percent_l908_90810

/-- The hourly wage of Mike in dollars -/
noncomputable def mike_wage : ℝ := 14

/-- The hourly wage of Phil in dollars -/
noncomputable def phil_wage : ℝ := 7

/-- The percentage difference in hourly wages between Mike and Phil -/
noncomputable def wage_difference_percentage : ℝ := (mike_wage - phil_wage) / mike_wage * 100

theorem wage_difference_is_fifty_percent :
  wage_difference_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wage_difference_is_fifty_percent_l908_90810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_carlos_distance_l908_90832

/-- Represents a person's movement rate in miles per minute -/
structure Rate where
  miles : ℚ
  minutes : ℚ
  valid : minutes > 0

/-- Calculates the distance traveled given a rate and time -/
def distance (r : Rate) (time : ℚ) : ℚ :=
  (r.miles / r.minutes) * time

theorem mia_carlos_distance (mia_rate : Rate) (carlos_rate : Rate) (time : ℚ) :
  mia_rate.miles = 1 ∧ mia_rate.minutes = 20 ∧
  carlos_rate.miles = 3 ∧ carlos_rate.minutes = 40 ∧
  time = 120 →
  distance mia_rate time + distance carlos_rate time = 15 := by
  sorry

#eval distance { miles := 1, minutes := 20, valid := by norm_num } 120
#eval distance { miles := 3, minutes := 40, valid := by norm_num } 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mia_carlos_distance_l908_90832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l908_90833

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

-- Define set B
def B : Set ℝ := {x | |x| ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l908_90833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_curves_l908_90815

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.exp (1 / x) / (x ^ 2)

-- State the theorem
theorem area_bounded_by_curves : 
  ∃ (S : ℝ), S = ∫ x in (1)..(2), f x ∧ S = Real.exp 1 - Real.sqrt (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_curves_l908_90815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l908_90809

noncomputable def f (x : ℝ) := x + 4/x
noncomputable def g (x a : ℝ) := 2^x + a

theorem problem_statement (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 1, ∃ x₂ ∈ Set.Icc 2 3, f x₁ ≥ g x₂ a) →
  a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l908_90809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_with_lcm_conditions_l908_90898

theorem count_triples_with_lcm_conditions : 
  ∃! n : ℕ, n = (Finset.filter (fun (x, y, z) => 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    Nat.lcm x y = 108 ∧ 
    Nat.lcm x z = 240 ∧ 
    Nat.lcm y z = 360
  ) (Finset.product (Finset.range 361) (Finset.product (Finset.range 361) (Finset.range 361)))).card
  ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_triples_with_lcm_conditions_l908_90898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l908_90868

/-- The probability that two points chosen independently at random on the sides of a square
    with side length 2 have a straight-line distance of at least 1 -/
noncomputable def probability_distance_at_least_one (T : Set (ℝ × ℝ)) : ℝ :=
  (15 - Real.pi) / 8

/-- The square T with side length 2 -/
def square_T : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ (p.1 = 0 ∨ p.1 = 2 ∨ p.2 = 0 ∨ p.2 = 2)}

theorem probability_theorem :
  probability_distance_at_least_one square_T = (15 - Real.pi) / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l908_90868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_bottom_width_l908_90861

/-- Represents the trapezoidal cross-section of a stream -/
structure StreamCrossSection where
  top_width : ℝ
  bottom_width : ℝ
  depth : ℝ
  area : ℝ

/-- Calculates the area of a trapezoidal cross-section -/
noncomputable def trapezoid_area (s : StreamCrossSection) : ℝ :=
  (s.top_width + s.bottom_width) * s.depth / 2

/-- Theorem: The bottom width of the stream is 6 meters -/
theorem stream_bottom_width (s : StreamCrossSection) 
    (h1 : s.top_width = 10)
    (h2 : s.depth = 80)
    (h3 : s.area = 640)
    (h4 : s.area = trapezoid_area s) : 
  s.bottom_width = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_bottom_width_l908_90861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_mixture_total_l908_90867

/-- Given a mixture of blue, green, and white paint in a 1:2:5 ratio,
    where 6 gallons of green paint are used, prove that the total
    amount of paint used is 24 gallons. -/
theorem paint_mixture_total (blue green white total : ℚ) : 
  blue + green + white = total →
  (blue : ℚ) / green = 1 / 2 →
  (white : ℚ) / green = 5 / 2 →
  green = 6 →
  total = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_mixture_total_l908_90867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l908_90804

/-- The area of a right triangle with base 3 and height 9 is 13.5 square units. -/
theorem triangle_area (base height : Real) 
  (h1 : base = 3) (h2 : height = 9) :
  (1 / 2) * base * height = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l908_90804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_equation_l908_90874

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define the line
def line_eq (x y : ℝ) : Prop := x + 2*y - 2 = 0

-- Define the point P
def P (t : ℝ) : ℝ × ℝ := (t, 0)

-- Define the chord length
def chord_length : ℝ := 4

theorem unique_line_equation (t : ℝ) (h_t : t > 0) :
  ∃! l : ℝ → ℝ → Prop, 
    (∃ x y, l x y ∧ P t = (x, y)) ∧ 
    (∃ a b c d, l a b ∧ l c d ∧ circle_eq a b ∧ circle_eq c d ∧ 
      (a - c)^2 + (b - d)^2 = chord_length^2) ∧
    (∀ x y, l x y ↔ line_eq x y) :=
  sorry

#check unique_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_equation_l908_90874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_implies_coefficients_l908_90892

theorem polynomial_root_implies_coefficients 
  (a b : ℝ) 
  (root : ℂ)
  (h : root = (2 : ℂ) - 3 * Complex.I) 
  (is_root : root ^ 3 + a * root ^ 2 + b * root - (6 : ℂ) = 0) : 
  a = -46/13 ∧ b = 193/13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_root_implies_coefficients_l908_90892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_wait_is_40_seconds_l908_90808

/-- The duration of the green light in minutes -/
noncomputable def green_duration : ℝ := 1

/-- The duration of the red light in minutes -/
noncomputable def red_duration : ℝ := 2

/-- The total cycle time of the traffic light in minutes -/
noncomputable def cycle_time : ℝ := green_duration + red_duration

/-- The probability of arriving during the green light -/
noncomputable def prob_green : ℝ := green_duration / cycle_time

/-- The expected waiting time during the green light in minutes -/
noncomputable def expected_wait_green : ℝ := 0

/-- The expected waiting time during the red light in minutes -/
noncomputable def expected_wait_red : ℝ := red_duration / 2

/-- The total expected waiting time in minutes -/
noncomputable def total_expected_wait : ℝ :=
  expected_wait_green * prob_green + expected_wait_red * (1 - prob_green)

/-- The average waiting time in seconds -/
noncomputable def average_wait_seconds : ℝ := total_expected_wait * 60

theorem average_wait_is_40_seconds :
  average_wait_seconds = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_wait_is_40_seconds_l908_90808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olivia_cookie_baggies_l908_90812

theorem olivia_cookie_baggies (chocolate_chip : ℕ) (oatmeal : ℕ) 
  (h1 : chocolate_chip = 33) (h2 : oatmeal = 67) :
  Nat.gcd chocolate_chip oatmeal = 1 :=
by
  rw [h1, h2]
  norm_num

#eval Nat.gcd 33 67

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olivia_cookie_baggies_l908_90812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l908_90838

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (a : ℝ) (x : ℝ) := -x^2 + a*x - 2

def tangent_line (x : ℝ) := x - 1

noncomputable def common_points (a : ℝ) :=
  let Δ := a^2 - 2*a - 3
  if Δ > 0 then 2
  else if Δ = 0 then 1
  else 0

noncomputable def h (x : ℝ) := x + 2/x + Real.log x

theorem problem_solution :
  (∀ a, common_points a = 
    if a < -1 ∨ a > 3 then 2
    else if a = -1 ∨ a = 3 then 1
    else 0) ∧
  (∀ a, (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), 
    (f x - g a x = 0 → ∃ y ≠ x, f y - g a y = 0)) ↔ 
    3 < a ∧ a ≤ Real.exp 1 + 2/Real.exp 1 + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l908_90838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l908_90851

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (2 : ℝ)^x ≤ 0) ↔ (∀ x : ℝ, (2 : ℝ)^x > 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l908_90851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_distribution_cdf_l908_90847

-- Define the random variable X
def X : ℝ → ℝ := sorry

-- Define the cumulative distribution function F(x)
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ -3 then 0
  else if x < 2 then (x + 3) / 5
  else 1

-- Theorem statement
theorem uniform_distribution_cdf :
  (∀ x, X x ∈ Set.Icc (-3) 2) →  -- X is uniformly distributed on [-3, 2]
  (∀ x, F x = 
    if x ≤ -3 then 0
    else if x < 2 then (x + 3) / 5
    else 1) :=
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_distribution_cdf_l908_90847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l908_90886

theorem complex_modulus_problem (z : ℂ) :
  (1 - Complex.I * Real.sqrt 3 * z) / (1 + Complex.I * Real.sqrt 3 * z) = Complex.I →
  Complex.abs z = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l908_90886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_three_roots_sum_zero_l908_90843

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -(f x)

/-- A real number r is a root of function f if f(r) = 0 -/
def IsRoot (f : ℝ → ℝ) (r : ℝ) : Prop :=
  f r = 0

theorem odd_function_three_roots_sum_zero
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (x₁ x₂ x₃ : ℝ)
  (h_roots : IsRoot f x₁ ∧ IsRoot f x₂ ∧ IsRoot f x₃)
  (h_exactly_three : ∀ x, IsRoot f x → x = x₁ ∨ x = x₂ ∨ x = x₃) :
  x₁ + x₂ + x₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_three_roots_sum_zero_l908_90843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_region_l908_90846

/-- The area of a region bounded by three arcs of circles with given properties -/
theorem area_of_special_region :
  ∀ (r : ℝ) (θ : ℝ),
  r = 4 →
  θ = π / 3 →
  ∃ (A : ℝ),
    A = 3 * (r^2 * θ / 2 - r^2 * Real.sin θ / 2) ∧
    A = 16 * Real.sqrt 3 - 8 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_special_region_l908_90846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l908_90814

theorem log_equation_solution (x : ℝ) (h : Real.log 128 / Real.log x = 7/3) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l908_90814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l908_90823

theorem count_integer_solutions : 
  ∃ (S : Finset ℤ), (∀ x : ℤ, x ∈ S ↔ |x - 3| ≤ 7) ∧ Finset.card S = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integer_solutions_l908_90823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_and_last_l908_90841

def numbers : List Int := [-3, 1, 5, 8, 11, 13]

def isValidArrangement (arr : List Int) : Prop :=
  arr.length = 6 ∧
  arr.toFinset = numbers.toFinset ∧
  (∃ i, i ∈ [1, 2, 3] ∧ arr[i]! = 13) ∧
  (∃ i, i ∈ [2, 3, 4, 5] ∧ arr[i]! = -3) ∧
  (arr.head? ∉ [some 5, some 8] ∧ arr.getLast? ∉ [some 5, some 8])

theorem sum_of_first_and_last (arr : List Int) :
  isValidArrangement arr → arr.head?.getD 0 + arr.getLast?.getD 0 = 12 := by
  sorry

#check sum_of_first_and_last

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_and_last_l908_90841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l908_90878

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem cosine_of_angle_between_vectors 
  (a b : V) 
  (ha : ‖a‖ = 5) 
  (hb : ‖b‖ = 7) 
  (hab : ‖a + b‖ = 10) : 
  (inner a b) / (‖a‖ * ‖b‖) = 13 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_of_angle_between_vectors_l908_90878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_validity_l908_90859

/-- The line equation y = 3x - 5 -/
def line_equation (x y : ℝ) : Prop := y = 3 * x - 5

/-- A parameterization of a line -/
structure Parameterization where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Check if a point (x, y) satisfies the line equation -/
noncomputable def point_on_line (p : ℝ × ℝ) : Prop :=
  line_equation p.1 p.2

/-- Check if a vector is a valid direction vector for the line -/
def valid_direction (v : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k, 3 * k)

/-- Check if a parameterization is valid for the line -/
noncomputable def valid_parameterization (param : Parameterization) : Prop :=
  point_on_line param.point ∧ valid_direction param.direction

/-- The given parameterizations -/
noncomputable def param_A : Parameterization := ⟨(0, -5), (1, 3)⟩
noncomputable def param_B : Parameterization := ⟨(5/3, 0), (-3, -1)⟩
noncomputable def param_C : Parameterization := ⟨(2, 1), (9, 3)⟩
noncomputable def param_D : Parameterization := ⟨(3, -2), (3/2, 3)⟩
noncomputable def param_E : Parameterization := ⟨(-5, -20), (1/15, 1/5)⟩

theorem parameterization_validity :
  valid_parameterization param_A ∧
  valid_parameterization param_B ∧
  valid_parameterization param_E ∧
  ¬valid_parameterization param_C ∧
  ¬valid_parameterization param_D := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parameterization_validity_l908_90859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_family_ice_cream_cost_l908_90825

/-- Represents the cost of ice cream scoops for the Martin family's order -/
def ice_cream_cost (kiddie_price regular_price double_price regular_count kiddie_count double_count : ℕ) : ℕ :=
  regular_price * regular_count + kiddie_price * kiddie_count + double_price * double_count

theorem martin_family_ice_cream_cost :
  ice_cream_cost 3 4 6 2 2 3 = 32 := by
  -- Unfold the definition of ice_cream_cost
  unfold ice_cream_cost
  -- Evaluate the arithmetic expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_family_ice_cream_cost_l908_90825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l908_90817

noncomputable def equation1 (c d x : ℝ) : ℝ := (x + c) * (x + d) * (x - 10) / ((x - 5) ^ 2)
noncomputable def equation2 (c d x : ℝ) : ℝ := (x + 3 * c) * (x - 4) * (x - 8) / ((x + d) * (x - 10))

theorem problem_solution (c d : ℝ) 
  (h1 : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    equation1 c d x = 0 ∧ equation1 c d y = 0 ∧ equation1 c d z = 0)
  (h2 : ∃! x : ℝ, equation2 c d x = 0) :
  100 * c + d = 141.3 := by
  sorry

#eval Float.toString (100 * (4/3) + 8)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l908_90817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_2_l908_90884

/-- The radius of the circle formed by points with spherical coordinates (2, θ, π/4) -/
noncomputable def circle_radius : ℝ :=
  Real.sqrt 2

/-- Theorem: The radius of the circle formed by points with spherical coordinates (2, θ, π/4) is √2 -/
theorem circle_radius_is_sqrt_2 :
  let ρ : ℝ := 2
  let φ : ℝ := π / 4
  let r : ℝ := Real.sqrt (ρ^2 * Real.sin φ * Real.sin φ)
  r = circle_radius := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_2_l908_90884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_monotone_l908_90882

/-- A function F that takes two real inputs and returns a real output. -/
noncomputable def F : ℝ → ℝ → ℝ := sorry

/-- A continuous, nonconstant real function satisfying the functional equation. -/
noncomputable def f : ℝ → ℝ := sorry

/-- f is continuous -/
axiom f_continuous : Continuous f

/-- f is nonconstant -/
axiom f_nonconstant : ∃ x y, f x ≠ f y

/-- Functional equation: f(x + y) = F(f(x), f(y)) for all real x and y -/
axiom functional_equation : ∀ x y, f (x + y) = F (f x) (f y)

/-- Theorem: f is strictly monotone -/
theorem f_strictly_monotone : StrictMono f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_monotone_l908_90882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distance_minimizes_total_fee_land_fee_inverse_proportional_inventory_fee_directly_proportional_fees_at_10km_l908_90875

/-- The distance from the warehouse to the station that minimizes the total fee -/
noncomputable def optimal_distance : ℝ := 5

/-- The monthly land occupation fee as a function of distance -/
noncomputable def land_fee (x : ℝ) : ℝ := 40 / x

/-- The monthly inventory fee as a function of distance -/
noncomputable def inventory_fee (x : ℝ) : ℝ := (8 / 5) * x

/-- The total monthly fee as a function of distance -/
noncomputable def total_fee (x : ℝ) : ℝ := land_fee x + inventory_fee x

theorem optimal_distance_minimizes_total_fee :
  ∀ x > 0, total_fee optimal_distance ≤ total_fee x := by
  sorry

theorem land_fee_inverse_proportional :
  ∀ x y, x > 0 → y > 0 → land_fee x * x = land_fee y * y := by
  sorry

theorem inventory_fee_directly_proportional :
  ∀ x y, x > 0 → y > 0 → inventory_fee x / x = inventory_fee y / y := by
  sorry

theorem fees_at_10km :
  land_fee 10 = 4 ∧ inventory_fee 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_distance_minimizes_total_fee_land_fee_inverse_proportional_inventory_fee_directly_proportional_fees_at_10km_l908_90875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_collinear_l908_90822

/-- Given a vector a = (1, 1, 0), prove that e = (√2/2, √2/2, 0) is a unit vector collinear with a -/
theorem unit_vector_collinear (a e : Fin 3 → ℝ) : 
  a = ![1, 1, 0] → 
  e = ![Real.sqrt 2 / 2, Real.sqrt 2 / 2, 0] → 
  (∃ (k : ℝ), e = k • a) ∧ 
  Real.sqrt (e 0^2 + e 1^2 + e 2^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_collinear_l908_90822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l908_90889

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set of f(x) > 0
def solution_set (a : ℝ) : Set ℝ := {x | 1/2 < x ∧ x < 2}

-- Define convexity
def is_convex (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ D → x₂ ∈ D → (f x₁ + f x₂) / 2 ≤ f ((x₁ + x₂) / 2)

-- Theorem statement
theorem problem_solution (a : ℝ) :
  (∀ x, f a x > 0 ↔ x ∈ solution_set a) →
  (∀ x, a * x^2 + 5 * x + a^2 - 1 < 0 ↔ x > 3 ∨ x < -1/2) ∧
  is_convex (f (-2)) Set.univ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l908_90889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solutions_l908_90876

theorem cubic_root_equation_solutions :
  ∀ x : ℝ, (((3 + Real.sqrt 5) ^ x) ^ (1/3) + ((3 - Real.sqrt 5) ^ x) ^ (1/3) = 6) ↔ (x = 3 ∨ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solutions_l908_90876
