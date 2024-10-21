import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_properties_l429_42985

theorem equation_roots_properties (θ : ℝ) (m : ℝ) : 
  θ ∈ Set.Ioo 0 (2 * Real.pi) →
  (2 * (Real.sin θ)^2 - (Real.sqrt 3 + 1) * Real.sin θ + m = 0) →
  (2 * (Real.cos θ)^2 - (Real.sqrt 3 + 1) * Real.cos θ + m = 0) →
  ((Real.sin θ)^2 / (Real.sin θ - Real.cos θ) + 
   (Real.cos θ)^2 / (Real.cos θ - Real.sin θ) = (Real.sqrt 3 + 1) / 2) ∧
  (m = Real.sqrt 3 / 2) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_properties_l429_42985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_specific_pyramid_l429_42987

/-- Regular pyramid with square base -/
structure RegularPyramid where
  base_side : ℝ
  height : ℝ

/-- Lateral surface area of a regular pyramid -/
noncomputable def lateral_surface_area (p : RegularPyramid) : ℝ :=
  4 * (1/2 * p.base_side * Real.sqrt (1/4 * p.base_side^2 + p.height^2))

/-- Theorem: The lateral surface area of a regular pyramid with base side 2 and height 3 is 4√10 -/
theorem lateral_surface_area_specific_pyramid :
  lateral_surface_area ⟨2, 3⟩ = 4 * Real.sqrt 10 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval lateral_surface_area ⟨2, 3⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_specific_pyramid_l429_42987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l429_42939

/-- Given a geometric sequence with positive common ratio, 
    prove that if a_3 * a_9 = 2 * (a_5)^2 and a_2 = 1, then a_1 = √2/2 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  q > 0 →  -- Common ratio is positive
  a 3 * a 9 = 2 * (a 5)^2 →  -- Given condition
  a 2 = 1 →  -- Given condition
  a 1 = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l429_42939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_square_equals_three_digit_l429_42915

/-- Represents a digit (0-9) -/
def Digit := Fin 10

theorem two_digit_square_equals_three_digit (a b c : Digit) : 
  b.val = 1 →
  (10 * a.val + b.val : ℕ)^2 = (100 * c.val + 10 * c.val + b.val : ℕ) →
  (100 * c.val + 10 * c.val + b.val : ℕ) > 300 →
  a ≠ b →
  a ≠ c →
  b ≠ c →
  a.val = 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_square_equals_three_digit_l429_42915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_non_overlapping_regions_count_l429_42919

/-- A structure representing a configuration of lines on a plane -/
structure LineConfiguration where
  n : ℕ
  line : ℕ → Set (Real × Real)
  not_parallel : ∀ i j, i ≠ j → ¬ (∀ x y, (x, y) ∈ line i ↔ (x, y) ∈ line j)
  no_triple_intersection : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    ¬ (∃ p, p ∈ line i ∧ p ∈ line j ∧ p ∈ line k)

/-- The number of intersection points for n lines -/
def intersection_points (config : LineConfiguration) : ℕ :=
  config.n * (config.n - 1) / 2

/-- The number of non-overlapping regions formed by n lines -/
def non_overlapping_regions (config : LineConfiguration) : ℕ :=
  config.n * (config.n + 1) / 2 + 1

/-- Theorem stating the number of intersection points -/
theorem intersection_points_count (config : LineConfiguration) :
  intersection_points config = config.n * (config.n - 1) / 2 := by
  sorry

/-- Theorem stating the number of non-overlapping regions -/
theorem non_overlapping_regions_count (config : LineConfiguration) :
  non_overlapping_regions config = config.n * (config.n + 1) / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_non_overlapping_regions_count_l429_42919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tims_payment_l429_42954

noncomputable def mri_cost : ℝ := 1200
noncomputable def exam_time_minutes : ℝ := 30
noncomputable def doctor_hourly_rate : ℝ := 300
noncomputable def additional_fee : ℝ := 150
noncomputable def insurance_coverage_percent : ℝ := 80

noncomputable def total_cost : ℝ := mri_cost + (doctor_hourly_rate * exam_time_minutes / 60) + additional_fee
noncomputable def insurance_coverage : ℝ := total_cost * insurance_coverage_percent / 100

theorem tims_payment :
  total_cost - insurance_coverage = 300 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tims_payment_l429_42954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l429_42973

noncomputable def f (x : ℝ) := Real.exp (-(x - 1)^2)

theorem f_inequality : f (Real.sqrt 3 / 2) > f (Real.sqrt 6 / 2) ∧ f (Real.sqrt 6 / 2) > f (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l429_42973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l429_42998

open Real Matrix

-- Define the rotation matrix
noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

-- Define the identity matrix
def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0],
    ![0, 1]]

-- Theorem statement
theorem smallest_rotation_power : 
  (∃ (n : ℕ), n > 0 ∧ (rotation_matrix (240 * π / 180))^n = identity_matrix) ∧
  (∀ (m : ℕ), 0 < m ∧ m < 3 → (rotation_matrix (240 * π / 180))^m ≠ identity_matrix) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_rotation_power_l429_42998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l429_42908

theorem sum_remainder (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x % 20 = 7 → y % 20 = 11 → z % 20 = 15 → (x + y + z) % 20 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l429_42908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PEF_l429_42902

/-- Ellipse with given properties -/
structure Ellipse :=
  (A B E F P : ℝ × ℝ)
  (on_major_axis : A.1 = B.1 ∧ A.2 = B.2)
  (is_focus_E : True)  -- placeholder for focus property
  (is_focus_F : True)  -- placeholder for focus property
  (major_axis_length : dist A B = 4)
  (focus_distance : dist A F = 2 + Real.sqrt 3)
  (point_on_ellipse : True)  -- placeholder for P being on the ellipse
  (distance_product : dist P E * dist P F = 2)

/-- Theorem statement -/
theorem area_of_triangle_PEF (Γ : Ellipse) : 
  Real.sqrt ((dist Γ.P Γ.E)^2 * (dist Γ.P Γ.F)^2 - 
    ((dist Γ.P Γ.E)^2 + (dist Γ.P Γ.F)^2 - (dist Γ.E Γ.F)^2)^2 / 4) / 2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PEF_l429_42902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_uses_half_kg_l429_42999

/-- Represents the weight used by the dealer -/
noncomputable def dealer_weight : ℝ := sorry

/-- The dealer claims to sell at cost price -/
axiom claimed_cost_price : True

/-- The dealer makes a 100% profit -/
axiom profit_percentage : ℝ 

/-- The profit percentage is 100 -/
axiom profit_is_100 : profit_percentage = 100

/-- The relationship between the claimed weight, actual weight, and profit -/
axiom profit_relation : 
  dealer_weight * (1 + profit_percentage / 100) = 1

theorem dealer_uses_half_kg : dealer_weight = 0.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_uses_half_kg_l429_42999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l429_42912

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^2 - 5*x + 6)*(x-3)*(x+2)
  ∀ x : ℝ, f x = 0 ↔ x = 2 ∨ x = 3 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_equation_l429_42912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_feeding_cost_l429_42947

-- Define the given constants
noncomputable def total_weight : ℚ := 30
noncomputable def food_ratio : ℚ := 2
noncomputable def ounces_per_jar : ℚ := 15
noncomputable def cost_per_jar : ℚ := 2

-- Define the function to calculate the cost
noncomputable def calculate_cost (weight : ℚ) (ratio : ℚ) (jar_size : ℚ) (jar_cost : ℚ) : ℚ :=
  let total_food := weight * ratio
  let jars_needed := total_food / jar_size
  (Rat.ceil jars_needed) * jar_cost

-- State the theorem
theorem turtle_feeding_cost :
  calculate_cost total_weight food_ratio ounces_per_jar cost_per_jar = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_feeding_cost_l429_42947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l429_42980

def is_arithmetic_progression (seq : List ℚ) : Prop :=
  seq.length ≥ 3 ∧
  ∀ i j k, i + 1 = j ∧ j + 1 = k ∧ k < seq.length →
    seq.get! j - seq.get! i = seq.get! k - seq.get! j

theorem arithmetic_sequence_solution :
  ∀ (x : ℚ),
  is_arithmetic_progression [3/4, 2*x - 3, 7*x] →
  x = -9/4 :=
by
  intro x h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_solution_l429_42980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_l429_42926

def is_consequence_of (A B : Prop) : Prop :=
  B → A

def is_necessary_condition_for (A B : Prop) : Prop :=
  is_consequence_of A B

theorem necessary_condition (A B : Prop) : 
  (is_necessary_condition_for A B) ↔ (B → A) := by
  unfold is_necessary_condition_for
  unfold is_consequence_of
  simp
  
#check necessary_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_l429_42926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digit_occurrence_l429_42959

def room_numbers : List ℕ := (List.range 26).map (λ x => x + 300) ++ (List.range 26).map (λ x => x + 400)

def digit_count (d : ℕ) : ℕ := (room_numbers.map (λ n => n.repr.count (Char.ofNat (d + 48)))).sum

theorem max_digit_occurrence : ∃ d : ℕ, d < 10 ∧ digit_count d = 26 ∧ ∀ d' : ℕ, d' < 10 → digit_count d' ≤ 26 := by
  sorry

#eval room_numbers
#eval (List.range 10).map digit_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digit_occurrence_l429_42959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l429_42964

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / Real.sqrt (x - 7)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l429_42964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l429_42968

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 1 else -1

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  f x / x^2

-- Theorem statement
theorem g_properties :
  (∀ x ≠ 0, g (-x) = -g x) ∧                    -- g is an odd function
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → g x₂ < g x₁) ∧ -- g is decreasing on (-∞, 0)
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → g x₂ < g x₁)   -- g is decreasing on (0, +∞)
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l429_42968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_complex_l429_42945

/-- Given complex numbers a, b, c forming an isosceles right triangle with equal sides of length 12,
    and |a + b + c| = 24, prove that |ab + ac + bc| = 144√2 -/
theorem isosceles_right_triangle_complex (a b c : ℂ) : 
  (∃ (x y : ℂ), Complex.abs (x - y) = 12 ∧ Complex.abs (x - c) = 12 ∧ Complex.abs (y - c) = 12 ∧ Complex.arg (x - y) = π/2) →
  Complex.abs (a + b + c) = 24 →
  Complex.abs (a*b + a*c + b*c) = 144 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_complex_l429_42945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l429_42905

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 2 * Real.cos (x / 2) ^ 2) * Real.cos (x + Real.pi / 2)

theorem problem_solution (a : ℝ) (α : ℝ) 
  (h1 : f a (Real.pi / 2) = 0)
  (h2 : α > Real.pi / 2 ∧ α < Real.pi)
  (h3 : f a (α / 2) = -2 / 5) :
  a = -1 ∧ Real.cos (Real.pi / 6 - 2 * α) = (-7 * Real.sqrt 3 - 24) / 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l429_42905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_dozen_price_is_30_l429_42993

/-- The price Chloe charged for half a dozen strawberries -/
def half_dozen_price (cost_per_dozen : ℚ) (total_dozens_sold : ℕ) (total_profit : ℚ) : ℚ :=
  ((cost_per_dozen * total_dozens_sold + total_profit) / total_dozens_sold) / 2

/-- Theorem stating the price for half a dozen strawberries -/
theorem half_dozen_price_is_30 :
  half_dozen_price 50 50 500 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_dozen_price_is_30_l429_42993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_zero_l429_42927

variable (A B : ℝ)

def f (x : ℝ) : ℝ := A * x^2 + B
def g (x : ℝ) : ℝ := B * x^2 + A

theorem sum_of_coefficients_is_zero 
  (h1 : A ≠ B) 
  (h2 : ∀ x, f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A + B = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_zero_l429_42927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_is_correct_l429_42932

/-- Represents the dimensions and properties of a water-filled cistern -/
structure Cistern where
  length : ℝ
  width : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the depth of water in a cistern given its dimensions and wet surface area -/
noncomputable def waterDepth (c : Cistern) : ℝ :=
  (c.wetSurfaceArea - c.length * c.width) / (2 * (c.length + c.width))

/-- Theorem stating that for a cistern with given dimensions, the water depth is 1.25 meters -/
theorem water_depth_is_correct (c : Cistern) 
    (h_length : c.length = 4)
    (h_width : c.width = 2)
    (h_wetSurfaceArea : c.wetSurfaceArea = 23) :
  waterDepth c = 1.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_is_correct_l429_42932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_power_twelve_l429_42950

noncomputable def z : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2

theorem z_power_twelve : z^12 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_power_twelve_l429_42950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_40_cents_l429_42970

structure Coin where
  value : ℚ
  is_heads : Bool

def penny : Coin := ⟨1/100, false⟩
def nickel : Coin := ⟨5/100, false⟩
def dime : Coin := ⟨10/100, false⟩
def quarter : Coin := ⟨25/100, false⟩
def half_dollar : Coin := ⟨50/100, false⟩

def coin_set : List Coin := [penny, nickel, dime, quarter, half_dollar]

def total_value (coins : List Coin) : ℚ :=
  (coins.filter (λ c => c.is_heads)).foldl (λ acc c => acc + c.value) 0

def is_at_least_40_cents (coins : List Coin) : Bool :=
  total_value coins ≥ 40/100

def all_outcomes : List (List Coin) :=
  List.mapM (λ c => [{ c with is_heads := true }, { c with is_heads := false }]) coin_set

def successful_outcomes : List (List Coin) :=
  all_outcomes.filter is_at_least_40_cents

theorem probability_at_least_40_cents :
  (successful_outcomes.length : ℚ) / (all_outcomes.length : ℚ) = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_40_cents_l429_42970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_speeds_and_zero_differences_l429_42963

/-- Represents a runner's performance --/
structure Runner where
  distance : ℝ  -- Distance in miles
  time : ℝ      -- Time in hours
  speed : ℝ     -- Speed in miles per hour

/-- Calculates the average speed given distance and time --/
noncomputable def averageSpeed (d : ℝ) (t : ℝ) : ℝ := d / t

theorem equal_speeds_and_zero_differences 
  (jim : Runner) 
  (frank : Runner) 
  (susan : Runner) 
  (h_jim : jim = ⟨16, 2, averageSpeed 16 2⟩) 
  (h_frank : frank = ⟨20, 2.5, averageSpeed 20 2.5⟩) 
  (h_susan : susan = ⟨12, 1.5, averageSpeed 12 1.5⟩) : 
  jim.speed = frank.speed ∧ 
  jim.speed = susan.speed ∧ 
  frank.speed - jim.speed = 0 ∧ 
  susan.speed - jim.speed = 0 := by
  sorry

#check equal_speeds_and_zero_differences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_speeds_and_zero_differences_l429_42963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l429_42920

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_abcd : a * b * c * d = 1) : 
  (1 / Real.sqrt (1/2 + a + a*b + a*b*c)) + 
  (1 / Real.sqrt (1/2 + b + b*c + b*c*d)) + 
  (1 / Real.sqrt (1/2 + c + c*d + c*d*a)) + 
  (1 / Real.sqrt (1/2 + d + d*a + d*a*b)) ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l429_42920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sign_and_angle_C_l429_42977

/-- Triangle properties and f function -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  R : ℝ
  r : ℝ
  hab : a ≤ b
  hbc : b ≤ c
  hR : R > 0
  hr : r > 0

/-- Definition of f -/
def f (t : Triangle) : ℝ := t.a + t.b - 2*t.R - 2*t.r

/-- Theorem stating the relationship between f and angle C -/
theorem f_sign_and_angle_C (t : Triangle) (C : ℝ) (hC : 0 < C ∧ C < π) :
  (f t > 0 ↔ 0 < C ∧ C < π/2) ∧
  (f t = 0 ↔ C = π/2) ∧
  (f t < 0 ↔ π/2 < C ∧ C < π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sign_and_angle_C_l429_42977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parallel_chords_l429_42982

-- Define the circle and chord properties
def circle_radius : ℝ := 10
def chord_distance : ℝ := 6

-- Define the angle subtended by the chord at the center
noncomputable def theta : ℝ := 2 * Real.arccos (chord_distance / (2 * circle_radius))

-- Define the area between the chords
noncomputable def area_between_chords : ℝ := 100 * theta - 6 * Real.sqrt 91

-- Theorem statement
theorem area_between_parallel_chords :
  area_between_chords = 100 * theta - 6 * Real.sqrt 91 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parallel_chords_l429_42982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_91_l429_42937

/-- Calculates the downstream distance given swimming conditions -/
noncomputable def downstream_distance (upstream_distance : ℝ) (swim_time : ℝ) (still_water_speed : ℝ) : ℝ :=
  let stream_speed := (still_water_speed * swim_time - upstream_distance) / swim_time
  (still_water_speed + stream_speed) * swim_time

/-- Proves that the downstream distance is 91 km given the specified conditions -/
theorem downstream_distance_is_91 :
  downstream_distance 21 7 8 = 91 := by
  -- Unfold the definition of downstream_distance
  unfold downstream_distance
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_91_l429_42937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_correct_l429_42900

/-- The equation of a hyperbola in the form (ax + b)^2/c - (dy + e)^2/f = 1 -/
def HyperbolaEquation (a b c d e f : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (a * x + b)^2 / c - (d * y + e)^2 / f = 1

/-- The center of a hyperbola given by the equation (ax + b)^2/c - (dy + e)^2/f = 1 -/
noncomputable def HyperbolaCenter (a b c d e f : ℝ) : ℝ × ℝ :=
  (- b / a, e / d)

theorem hyperbola_center_correct :
  let h := HyperbolaEquation 4 8 36 3 (-6) 25
  let center := HyperbolaCenter 4 8 36 3 (-6) 25
  center = (-2, 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_center_correct_l429_42900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_theorem_l429_42914

/-- Define a line in 2D space using parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Define a line in 2D space using a general equation --/
structure GeneralLine where
  equation : ℝ → ℝ → Prop

/-- Define a point in 2D space --/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem intersection_and_distance_theorem 
  (l1 : ParametricLine) 
  (l2 : GeneralLine) 
  (Q : Point) :
  (l1.x t = 1 + t ∧ l1.y t = -5 + Real.sqrt 3 * t) →
  (l2.equation x y ↔ x - y - 2 * Real.sqrt 3 = 0) →
  Q.x = 1 ∧ Q.y = -5 →
  ∃ (P : Point),
    (P.x = 1 + 2 * Real.sqrt 3 ∧ P.y = 1) ∧
    distance P Q = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_theorem_l429_42914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l429_42942

noncomputable def f (x : ℝ) : ℝ := x / (x - 2)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l429_42942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equality_l429_42903

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_union_equality : (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equality_l429_42903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l429_42943

def f (x : ℝ) : ℝ := 6 * x^4 - 35 * x^3 + 62 * x^2 - 35 * x + 6

theorem roots_of_polynomial :
  ∀ x : ℝ, f x = 0 ↔ x = 2 ∨ x = 3 ∨ x = (1/2 : ℝ) ∨ x = (1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_polynomial_l429_42943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l429_42969

theorem power_equality (n b : ℝ) : n = 2^(0.15 : ℝ) → n^b = 8 → b = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_l429_42969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l429_42907

def sequence_a : ℕ → ℚ
| 0 => 1
| n + 1 => ((n + 1)^2 + (n + 1)) * sequence_a n / (3 * sequence_a n + (n + 1)^2 + (n + 1))

theorem sequence_a_general_term (n : ℕ) : 
  sequence_a n = (n + 1) / (4 * (n + 1) - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l429_42907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_v1_value_l429_42906

def f (x : ℝ) : ℝ := 4*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

def horner_v1 (a₅ a₄ : ℝ) (x : ℝ) : ℝ := a₅ * x + a₄

theorem horner_v1_value :
  let x : ℝ := 5
  let a₅ : ℝ := 4
  let a₄ : ℝ := 2
  horner_v1 a₅ a₄ x = 22 :=
by
  unfold horner_v1
  simp
  norm_num

#eval horner_v1 4 2 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_v1_value_l429_42906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_king_diagonal_moves_even_l429_42909

/-- Represents a chessboard traversal by a king -/
structure KingTraversal where
  /-- The size of the chessboard (8 for standard 8x8 board) -/
  board_size : Nat
  /-- The sequence of moves made by the king -/
  moves : List (Nat × Nat)
  /-- Ensures the board is 8x8 -/
  board_is_8x8 : board_size = 8
  /-- Ensures each square is visited exactly once -/
  visits_each_square_once : moves.length = board_size * board_size
  /-- Ensures the king returns to the starting square -/
  returns_to_start : moves.head? = moves.get? (moves.length - 1)

/-- Counts the number of diagonal moves in a king's traversal -/
def countDiagonalMoves (traversal : KingTraversal) : Nat :=
  sorry

/-- Theorem stating that the number of diagonal moves in a valid king's traversal is even -/
theorem king_diagonal_moves_even (traversal : KingTraversal) : 
  Even (countDiagonalMoves traversal) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_king_diagonal_moves_even_l429_42909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_theta_l429_42981

noncomputable def f (x θ : Real) : Real := Real.sin (2 * x + θ) + Real.sqrt 3 * Real.cos (2 * x + θ)

theorem odd_decreasing_function_theta (θ : Real) 
  (h1 : θ > 0) (h2 : θ < Real.pi) 
  (h3 : ∀ x, f x θ = -f (-x) θ) -- odd function condition
  (h4 : ∀ x y, -Real.pi/4 ≤ x ∧ x < y ∧ y ≤ 0 → f x θ > f y θ) -- decreasing condition
  : θ = 2 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_decreasing_function_theta_l429_42981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_on_parabola_l429_42940

/-- Given an equilateral triangle AOB with O as the origin and vertices A and B 
    on the parabola y^2 = 3x, the side length of the triangle is 6√3. -/
theorem equilateral_triangle_on_parabola : 
  ∀ (A B : ℝ × ℝ),
  let O := (0, 0)
  -- A and B lie on the parabola y^2 = 3x
  (A.2)^2 = 3 * A.1 →
  (B.2)^2 = 3 * B.1 →
  -- AOB is an equilateral triangle
  dist O A = dist A B →
  dist O B = dist A B →
  dist O A = dist O B →
  -- The side length is 6√3
  dist O A = 6 * Real.sqrt 3 :=
by sorry

/-- Helper function to calculate the distance between two points -/
noncomputable def dist (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_on_parabola_l429_42940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_took_five_hours_l429_42988

/-- Represents a mechanic's job with hourly rate, parts cost, and total bill -/
structure MechanicJob where
  hourlyRate : ℚ
  partsCost : ℚ
  totalBill : ℚ

/-- Calculates the number of hours worked given a MechanicJob -/
def hoursWorked (job : MechanicJob) : ℚ :=
  (job.totalBill - job.partsCost) / job.hourlyRate

/-- Theorem stating that for the given conditions, the job took 5 hours -/
theorem job_took_five_hours (job : MechanicJob) 
  (h1 : job.hourlyRate = 45)
  (h2 : job.partsCost = 225)
  (h3 : job.totalBill = 450) : 
  hoursWorked job = 5 := by
  sorry

def main : IO Unit := do
  let job : MechanicJob := { hourlyRate := 45, partsCost := 225, totalBill := 450 }
  IO.println s!"Hours worked: {hoursWorked job}"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_took_five_hours_l429_42988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_area_relation_l429_42961

/-- A truncated pyramid with the given properties -/
structure TruncatedPyramid where
  S₁ : ℝ  -- Area of the larger base
  S₂ : ℝ  -- Area of the smaller base
  S : ℝ   -- Area of the lateral surface
  dividable : Prop  -- The pyramid can be divided into two truncated pyramids
  inscribable : Prop  -- Each resulting truncated pyramid can inscribe a sphere

/-- The theorem to be proven -/
theorem truncated_pyramid_area_relation (p : TruncatedPyramid) :
  p.dividable → p.inscribable →
  p.S = (Real.sqrt p.S₁ + Real.sqrt p.S₂) * (p.S₁^(1/4) + p.S₂^(1/4))^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_pyramid_area_relation_l429_42961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_comparison_l429_42983

noncomputable def scheme_a (m n : ℝ) : ℝ := (1 + m / 100) * (1 + n / 100)
noncomputable def scheme_b (m n : ℝ) : ℝ := (1 + n / 100) * (1 + m / 100)
noncomputable def scheme_c (m n : ℝ) : ℝ := (1 + (m + n) / 200) * (1 + (m + n) / 200)

theorem price_comparison (m n : ℝ) (h : m > n) (h' : n > 0) :
  scheme_a m n = scheme_b m n ∧ scheme_c m n > scheme_a m n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_comparison_l429_42983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l429_42974

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem solution_set_of_inequality (x : ℝ) :
  (∀ n : ℕ+, floor x = n ↔ n ≤ x ∧ x < n + 1) →
  (Set.Icc 2 8 : Set ℝ) = {x | 4 * (floor x)^2 - 36 * (floor x) + 45 < 0} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l429_42974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l429_42924

-- Define the function f(x) = sin x - log x (base 10)
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.log x / Real.log 10

-- State the theorem
theorem f_has_three_zeros :
  ∃ (a b c : ℝ), a ∈ Set.Ioo (0 : ℝ) (Real.pi : ℝ) ∧
                  b ∈ Set.Ioo (Real.pi : ℝ) (2 * Real.pi : ℝ) ∧
                  c ∈ Set.Ioo (2 * Real.pi : ℝ) (10 : ℝ) ∧
                  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
                  (∀ x, x > 0 → f x = 0 → x = a ∨ x = b ∨ x = c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l429_42924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_run_while_watching_movies_l429_42911

/-- Calculates the distance run while watching movies on a treadmill -/
theorem distance_run_while_watching_movies
  (running_speed : ℝ)  -- Running speed in minutes per mile
  (movie_duration : ℝ)  -- Duration of each movie in hours
  (num_movies : ℕ)  -- Number of movies watched
  (h1 : running_speed = 12)  -- Paul runs a mile in 12 minutes
  (h2 : movie_duration = 1.5)  -- Each movie is 1.5 hours long
  (h3 : num_movies = 2)  -- Paul watches two movies
  : (num_movies : ℝ) * movie_duration * 60 / running_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_run_while_watching_movies_l429_42911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_integral_part_a_limit_integral_part_b_l429_42949

/-- For any real-valued function f, we define the integral of f(x) from a to b -/
noncomputable def integral (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

theorem limit_integral_part_a :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
  |n * integral (λ x ↦ ((1 - x) / (1 + x))^n) 0 1 - (1/2)| < ε := by
  sorry

theorem limit_integral_part_b (k : ℕ) (hk : k ≥ 1) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
  |n^(k+1) * integral (λ x ↦ ((1 - x) / (1 + x))^n * x^k) 0 1 - (Nat.factorial k / 2^(k+1))| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_integral_part_a_limit_integral_part_b_l429_42949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l429_42958

-- Define the constants and functions
noncomputable def α : ℝ := 3 + 2 * Real.sqrt 2
noncomputable def x : ℝ := α ^ 500
noncomputable def n : ℤ := ⌊x⌋
noncomputable def f : ℝ := x - n

-- State the theorem
theorem x_times_one_minus_f_equals_one : x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_times_one_minus_f_equals_one_l429_42958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_castle_halls_bound_max_halls_is_three_l429_42901

/-- Represents a hall in the castle -/
def Hall := ℕ

/-- Represents the assignment of rooms to halls -/
def RoomAssignment := ℕ → Hall

/-- Checks if two rooms are in the same hall -/
def SameHall (assignment : RoomAssignment) (n m : ℕ) : Prop :=
  assignment n = assignment m

/-- The condition that room n is in the same hall as rooms 3n+1 and n+10 -/
def RoomRelation (assignment : RoomAssignment) : Prop :=
  ∀ n : ℕ, SameHall assignment n (3*n+1) ∧ SameHall assignment n (n+10)

/-- The number of distinct halls in the assignment -/
noncomputable def NumHalls (assignment : RoomAssignment) : ℕ :=
  Finset.card (Finset.range 3)  -- We use range 3 as a placeholder for the actual set of halls

theorem castle_halls_bound (assignment : RoomAssignment) 
  (h : RoomRelation assignment) : NumHalls assignment ≤ 3 := by
  sorry

/-- The maximum number of distinct halls is 3 -/
theorem max_halls_is_three : 
  ∃ (assignment : RoomAssignment), RoomRelation assignment ∧ NumHalls assignment = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_castle_halls_bound_max_halls_is_three_l429_42901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l429_42984

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition that a + b + c is an integer -/
def Parabola.sumIsInteger (p : Parabola) : Prop :=
  ∃ n : ℤ, p.a + p.b + p.c = n

/-- The vertex of the parabola -/
noncomputable def Parabola.vertex (p : Parabola) : ℝ × ℝ :=
  (1/4, -9/8)

theorem smallest_a_value (p : Parabola) 
    (h1 : p.a > 0)
    (h2 : p.sumIsInteger)
    (h3 : p.vertex = (1/4, -9/8)) :
    p.a ≥ 2/9 ∧ ∃ p' : Parabola, p'.a = 2/9 ∧ p'.sumIsInteger ∧ p'.vertex = (1/4, -9/8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_value_l429_42984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_is_ten_percent_l429_42931

/-- Calculates the loss percentage for a watch sale given the cost price,
    gain percentage, and price increase for a profitable sale. -/
noncomputable def calculate_loss_percentage (cost_price : ℝ) (gain_percentage : ℝ) (price_increase : ℝ) : ℝ :=
  let selling_price_with_gain := cost_price * (1 + gain_percentage / 100)
  let selling_price_with_loss := selling_price_with_gain - price_increase
  let loss := cost_price - selling_price_with_loss
  (loss / cost_price) * 100

/-- Theorem stating that the loss percentage is 10% given the problem conditions. -/
theorem loss_percentage_is_ten_percent :
  calculate_loss_percentage 1500 5 225 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_is_ten_percent_l429_42931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_tileable_contains_ab_domino_l429_42944

/-- A rectangle is (a,b)-tileable if it can be covered by a×b dominoes without gaps or overlaps. -/
def is_ab_tileable (a b m n : ℕ) : Prop :=
  ∃ (tiling : ℕ → ℕ → Bool), ∀ i j, tiling i j → 
    (i < m ∧ j < n) ∧ 
    (∃ k l, (k, l) ∈ [(0, 0), (0, b), (a, 0), (a, b)] ∧ 
      tiling (i + k) (j + l) = tiling i j)

/-- If a rectangle is (a,b)-tileable, then its dimensions are at least a and b. -/
theorem ab_tileable_contains_ab_domino (a b m n : ℕ) (h : a > 0 ∧ b > 0) :
  is_ab_tileable a b m n → m ≥ a ∧ n ≥ b :=
by
  intro h_tileable
  sorry -- The proof is omitted for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_tileable_contains_ab_domino_l429_42944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_age_is_34_l429_42986

/-- Represents a class of students -/
structure StudentClass where
  originalAverage : ℝ
  newAverage : ℝ
  ageDifference : ℝ

/-- Calculates the average age of new students given a StudentClass -/
def newStudentAverage (c : StudentClass) : ℝ :=
  c.originalAverage - c.ageDifference

/-- Theorem stating the average age of new students is 34 given the conditions -/
theorem new_student_age_is_34 (c : StudentClass) 
  (h1 : c.originalAverage = 40)
  (h2 : c.newAverage = c.originalAverage - 4)
  (h3 : c.ageDifference = 6) : 
  newStudentAverage c = 34 := by
  -- Unfold the definition of newStudentAverage
  unfold newStudentAverage
  -- Use the given hypotheses
  rw [h3]
  rw [h1]
  -- Perform the calculation
  norm_num

#check new_student_age_is_34

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_student_age_is_34_l429_42986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_range_l429_42996

theorem complex_range (a : ℝ) (Z₁ Z₂ : ℂ) 
  (h1 : Complex.abs Z₁ = a) 
  (h2 : Complex.abs Z₂ = 1) 
  (h3 : Z₁ * Z₂ = -a) : 
  ∃ x : ℝ, Z₁ - a * Z₂ = x ∧ -2 * a ≤ x ∧ x ≤ 2 * a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_range_l429_42996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cooking_probability_l429_42918

/-- Represents the set of courses available for selection -/
inductive Courses
| planting
| cooking
| pottery
| carpentry

/-- Represents the probability measure on the set of courses -/
noncomputable def P : Courses → ℝ
| _ => 1 / 4

/-- Represents the "cooking" course -/
def cooking : Courses := Courses.cooking

theorem cooking_probability :
  P cooking = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cooking_probability_l429_42918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l429_42992

theorem derivative_at_two (f : ℝ → ℝ) (h : ∀ x, f x = x^2 * (deriv f 1) - 3*x) :
  deriv f 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l429_42992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_events_exhaustive_events_l429_42933

-- Define the sample space
def SampleSpace := Finset (Fin 10)

-- Define the event A: at least 2 defective products are drawn
def EventA (s : SampleSpace) : Prop := s.card ≥ 2

-- Define the complementary event: at most 1 defective product is drawn
def ComplementEventA (s : SampleSpace) : Prop := s.card ≤ 1

-- Theorem: EventA and ComplementEventA are mutually exclusive
theorem mutually_exclusive_events : 
  ∀ s : SampleSpace, ¬(EventA s ∧ ComplementEventA s) := by
  intro s
  push_neg
  simp [EventA, ComplementEventA]
  sorry

-- Theorem: EventA and ComplementEventA are exhaustive
theorem exhaustive_events : 
  ∀ s : SampleSpace, EventA s ∨ ComplementEventA s := by
  intro s
  simp [EventA, ComplementEventA]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_events_exhaustive_events_l429_42933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_one_root_in_interval_l429_42951

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 3 * x^2 + 1

-- State the theorem
theorem exact_one_root_in_interval :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
by
  sorry

#check exact_one_root_in_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_one_root_in_interval_l429_42951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_product_l429_42957

/-- Two hyperbolas with foci on the x-axis -/
structure Hyperbolas where
  C₁ : Set (ℝ × ℝ)
  C₂ : Set (ℝ × ℝ)

/-- Asymptotic lines of the hyperbolas -/
def asymptoticLines (h : Hyperbolas) (k₁ k₂ : ℝ) : Prop :=
  ∃ (f₁ f₂ : ℝ → ℝ), 
    (∀ x, f₁ x = k₁ * x ∨ f₁ x = -k₁ * x) ∧ 
    (∀ x, f₂ x = k₂ * x ∨ f₂ x = -k₂ * x) ∧
    (∀ ε > 0, ∃ M > 0, ∀ x, |x| > M → 
      (∃ p ∈ h.C₁, dist (x, f₁ x) p < ε) ∧ 
      (∃ q ∈ h.C₂, dist (x, f₂ x) q < ε))

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (C : Set (ℝ × ℝ)) : ℝ :=
  sorry  -- Definition of eccentricity

/-- Theorem: Minimum value of e₁e₂ is 2 -/
theorem min_eccentricity_product (h : Hyperbolas) (k₁ k₂ : ℝ) :
  asymptoticLines h k₁ k₂ → k₁ * k₂ = 1 →
  ∀ ε > 0, ∃ e₁ e₂ : ℝ, 
    e₁ = eccentricity h.C₁ ∧ 
    e₂ = eccentricity h.C₂ ∧ 
    e₁ * e₂ ≥ 2 ∧
    e₁ * e₂ < 2 + ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_eccentricity_product_l429_42957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequalities_l429_42946

theorem trigonometric_inequalities :
  (Real.sin (35 * π / 180) > 0) ∧
  (Real.cos (167 * π / 180) < 0) ∧
  (Real.tan 3 < 0) ∧
  (1 / Real.tan (-1.5) < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequalities_l429_42946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_equality_l429_42904

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

-- State the theorem
theorem piecewise_function_equality (a : ℝ) :
  f a (1 - a) = f a (1 + a) → a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_equality_l429_42904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_theorem_l429_42990

/-- Represents the walking speed of Linda in miles per hour -/
noncomputable def linda_speed : ℝ := 2

/-- Represents the jogging speed of Tom in miles per hour -/
noncomputable def tom_speed : ℝ := 7

/-- Represents the time difference in hours between when Linda starts walking and Tom starts jogging -/
noncomputable def time_difference : ℝ := 1

/-- Calculates the time it takes Tom to cover a given distance -/
noncomputable def tom_time (distance : ℝ) : ℝ := distance / tom_speed

/-- Calculates Linda's distance after a given time -/
noncomputable def linda_distance (time : ℝ) : ℝ := linda_speed * time

/-- Theorem stating the difference in time for Tom to cover half and twice Linda's distance -/
theorem time_difference_theorem : 
  let half_distance := linda_distance time_difference / 2
  let double_distance := linda_distance time_difference * 2
  let time_diff := tom_time double_distance - tom_time half_distance
  abs (time_diff * 60 - 25.72) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_difference_theorem_l429_42990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_is_twenty_percent_l429_42966

/-- Calculates the second discount percentage given the original price, first discount percentage, and final price -/
noncomputable def second_discount_percentage (original_price first_discount_percent final_price : ℝ) : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_percent / 100)
  let second_discount_amount := price_after_first_discount - final_price
  (second_discount_amount / price_after_first_discount) * 100

theorem second_discount_is_twenty_percent :
  second_discount_percentage 149.99999999999997 10 108 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_discount_is_twenty_percent_l429_42966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l429_42997

/-- The number of permutations of the digits 6, 0, 0, 6, 3 that form a 5-digit number not starting or ending with 0 -/
def validPermutations : ℕ := 9

/-- Theorem stating that the number of valid permutations is 9 -/
theorem valid_permutations_count :
  validPermutations = 9 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_permutations_count_l429_42997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_hyperbola_l429_42991

-- Define the parametric equations for x and y
noncomputable def x (u : ℝ) : ℝ := 2 * (Real.exp u + Real.exp (-u))
noncomputable def y (u : ℝ) : ℝ := 4 * (Real.exp u - Real.exp (-u))

-- State the theorem
theorem points_form_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ u : ℝ, (x u)^2 / a^2 - (y u)^2 / b^2 = 1 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_hyperbola_l429_42991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_height_l429_42976

theorem inscribed_cylinder_height (r_cylinder r_sphere : ℝ) 
  (h_cylinder_radius : r_cylinder = 3)
  (h_sphere_radius : r_sphere = 7)
  (h_positive : 0 < r_cylinder ∧ 0 < r_sphere)
  (h_cylinder_in_sphere : r_cylinder ≤ r_sphere) :
  2 * Real.sqrt (r_sphere^2 - r_cylinder^2) = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_height_l429_42976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probability_l429_42953

-- Define the probability_of_hitting_pattern function
noncomputable def probability_of_hitting_pattern (accuracy : ℝ) (total_shots : ℕ) (hits : ℕ) (consecutive_hits : ℕ) : ℝ :=
  sorry -- The actual implementation would depend on how we define this probability

theorem shooting_probability (accuracy : ℝ) (total_shots hits consecutive_hits : ℕ) :
  accuracy = 0.6 →
  total_shots = 8 →
  hits = 5 →
  consecutive_hits = 4 →
  (Nat.descFactorial 4 2) * (accuracy ^ hits) * ((1 - accuracy) ^ (total_shots - hits)) =
    probability_of_hitting_pattern accuracy total_shots hits consecutive_hits :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probability_l429_42953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_triangle_area_l429_42995

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -x^2 + 8*x - 12

/-- The area of a triangle given three points -/
noncomputable def triangleArea (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  (1/2) * abs (x₁*y₂ + x₂*y₃ + x₃*y₁ - y₁*x₂ - y₂*x₃ - y₃*x₁)

theorem largest_triangle_area :
  ∀ p q : ℝ,
  1 ≤ p → p ≤ 4 →
  parabola 1 0 →
  parabola 4 3 →
  parabola p q →
  (∀ r s : ℝ, 1 ≤ r → r ≤ 4 → parabola r s →
    triangleArea 1 0 4 3 r s ≤ triangleArea 1 0 4 3 p q) →
  triangleArea 1 0 4 3 p q = 15/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_triangle_area_l429_42995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l429_42916

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x^3 - Real.sqrt 3 * x + 2

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - Real.sqrt 3

-- Theorem statement
theorem tangent_slope_range :
  Set.range f' = Set.Ioi (-Real.sqrt 3) := by
  sorry

#check tangent_slope_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_range_l429_42916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_food_price_calculation_l429_42972

/-- The actual price of the food before tax, tip, and discounts -/
def actual_price : ℝ := 160.06

/-- The total amount spent -/
def total_spent : ℝ := 198.60

/-- The tip percentage -/
def tip_percent : ℝ := 0.20

/-- The sales tax percentage -/
def sales_tax_percent : ℝ := 0.10

/-- The membership discount percentage -/
def discount_percent : ℝ := 0.15

/-- The percentage of the bill eligible for discount -/
def discounted_portion : ℝ := 0.40

/-- The percentage of the bill not eligible for discount -/
def non_discounted_portion : ℝ := 0.60

theorem food_price_calculation :
  let discounted_price := actual_price * discounted_portion * (1 - discount_percent)
  let non_discounted_price := actual_price * non_discounted_portion
  let price_after_discount := discounted_price + non_discounted_price
  let price_with_tax := price_after_discount * (1 + sales_tax_percent)
  let final_price := price_with_tax * (1 + tip_percent)
  ∃ ε > 0, |final_price - total_spent| < ε :=
by
  sorry

#eval actual_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_food_price_calculation_l429_42972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l429_42935

/-- The binomial expansion of (3/√x - x)^n -/
noncomputable def binomial_expansion (x : ℝ) (n : ℕ) := (3 / Real.sqrt x - x) ^ n

/-- The sum of coefficients of all terms in the expansion -/
def sum_of_coefficients (n : ℕ) := (3 - 1) ^ n

/-- The sum of binomial coefficients of all terms -/
def sum_of_binomial_coefficients (n : ℕ) := 2 ^ n

/-- The coefficient of x^r in the expansion -/
def coefficient (n r : ℕ) : ℤ := 
  (Nat.choose n r) * (-1)^r * 3^(n-r)

theorem coefficient_of_x_squared (n : ℕ) :
  sum_of_coefficients n + sum_of_binomial_coefficients n = 64 →
  coefficient n 3 = -90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_l429_42935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_binary_by_8_main_theorem_l429_42917

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def last_three_digits (b : List Bool) : List Bool :=
  b.reverse.take 3

theorem remainder_of_binary_by_8 (b : List Bool) :
  binary_to_decimal b % 8 = binary_to_decimal (last_three_digits b) := by
  sorry

theorem main_theorem :
  binary_to_decimal [true, false, true, true, false, true, false, true, true, false, true, false] % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_binary_by_8_main_theorem_l429_42917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bracket_is_mean_l429_42975

/-- A function that takes a finite list of real numbers and returns a real number -/
def bracket (X : List ℝ) : ℝ := sorry

/-- The bracket function is invariant under permutations -/
axiom bracket_perm (X Y : List ℝ) : Y.Perm X → bracket Y = bracket X

/-- The bracket function satisfies the translation property -/
axiom bracket_translation (X : List ℝ) (α : ℝ) :
  bracket (X.map (· + α)) = bracket X + α

/-- The bracket function is odd -/
axiom bracket_odd (X : List ℝ) : bracket (X.map (·⁻¹)) = -(bracket X)

/-- The bracket function satisfies the replacement property -/
axiom bracket_replacement (X : List ℝ) (x : ℝ) :
  bracket (List.replicate X.length (bracket X) ++ [x]) = bracket (X ++ [x])

/-- The main theorem: the bracket function is equal to the arithmetic mean -/
theorem bracket_is_mean (X : List ℝ) :
  bracket X = (X.sum / X.length : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bracket_is_mean_l429_42975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_triangle_l429_42941

-- Define a point
structure Point where
  x : Real
  y : Real

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define a square
structure Square where
  B' : Point
  C' : Point
  D' : Point
  E' : Point

-- Define a line segment
def LineSegment (a b : Point) : Set Point :=
  {p : Point | ∃ t : Real, 0 ≤ t ∧ t ≤ 1 ∧ p.x = (1 - t) * a.x + t * b.x ∧ p.y = (1 - t) * a.y + t * b.y}

-- Define adjacency for square vertices
def adjacent (p q : Point) : Prop :=
  (p.x - q.x)^2 + (p.y - q.y)^2 = 1 -- Assuming unit square for simplicity

-- Define the theorem
theorem square_in_triangle (t : Triangle) : 
  ∃ s : Square,
    (s.B' ∈ LineSegment t.A t.B) ∧
    (s.C' ∈ LineSegment t.A t.C) ∧
    (s.D' ∈ LineSegment t.B t.C) ∧
    (s.E' ∈ LineSegment t.B t.C) ∧
    (adjacent s.D' s.E') :=
by
  sorry -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_in_triangle_l429_42941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_DE_in_right_triangle_l429_42952

-- Define the right triangle DEF
noncomputable def RightTriangleDEF (DE EF DF : ℝ) : Prop :=
  DE ^ 2 + EF ^ 2 = DF ^ 2

-- Define the cosine of angle F
noncomputable def CosF (DE EF : ℝ) : ℝ := DE / EF

-- State the theorem
theorem length_DE_in_right_triangle :
  ∀ DE EF DF : ℝ,
  RightTriangleDEF DE EF DF →
  CosF DE EF = (5 * Real.sqrt 34) / 34 →
  EF = Real.sqrt 34 →
  DE = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_DE_in_right_triangle_l429_42952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_spending_l429_42994

/-- The total amount John spent at the supermarket --/
noncomputable def total_spent : ℚ := 65

/-- The fraction of money spent on fresh fruits and vegetables --/
def fruits_veg_fraction : ℚ := 1/5

/-- The fraction of money spent on meat products --/
def meat_fraction : ℚ := 1/3

/-- The fraction of money spent on bakery products --/
def bakery_fraction : ℚ := 1/10

/-- The fraction of money spent on dairy products --/
def dairy_fraction : ℚ := 1/6

/-- The amount spent on candy and magazine --/
noncomputable def candy_magazine_amount : ℚ := 13

/-- The cost of the magazine --/
noncomputable def magazine_cost : ℚ := 4

theorem supermarket_spending :
  fruits_veg_fraction * total_spent +
  meat_fraction * total_spent +
  bakery_fraction * total_spent +
  dairy_fraction * total_spent +
  candy_magazine_amount = total_spent :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supermarket_spending_l429_42994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_minimized_l429_42962

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points and line
variable (A B P Q : V)

-- Define positive numbers a and b
variable (a b : ℝ) (ha : 0 < a) (hb : 0 < b)

-- Define the line PQ
def line_PQ (t : ℝ) : V := (1 - t) • P + t • Q

-- Define the point M on line PQ
variable (M : V) (hM : ∃ t, M = line_PQ V P Q t)

-- Define the distance function
noncomputable def distance (X Y : V) : ℝ := ‖X - Y‖

-- Define the sum function
noncomputable def sum_function (M : V) : ℝ := b * distance V A M + a * distance V B M

-- Define the angle function
noncomputable def cos_angle (X Y Z : V) : ℝ := 
  inner (Y - X) (Z - X) / (‖Y - X‖ * ‖Z - X‖)

-- State the theorem
theorem sum_minimized :
  (∀ X, (∃ t, X = line_PQ V P Q t) → sum_function V A B a b M ≤ sum_function V A B a b X) ↔ 
  cos_angle V A M P / cos_angle V B M Q = a / b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_minimized_l429_42962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l429_42965

theorem train_speed_problem (train1_length train2_length : ℝ)
  (train2_speed : ℝ) (clearing_time : ℝ) :
  train1_length = 111 →
  train2_length = 165 →
  train2_speed = 90 →
  clearing_time = 6.623470122390208 →
  let total_distance := (train1_length + train2_length) / 1000
  let time_in_hours := clearing_time / 3600
  let relative_speed := total_distance / time_in_hours
  let train1_speed := relative_speed - train2_speed
  train1_speed = 60 := by
    sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l429_42965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l429_42921

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x - Real.pi / 4) + 3

theorem smallest_positive_period 
  (ω : ℝ) 
  (h1 : 1 < ω ∧ ω < 2) 
  (h2 : ∀ x, f ω (x + Real.pi) = f ω (Real.pi - x)) : 
  ∃ T, T > 0 ∧ (∀ x, f ω (x + T) = f ω x) ∧ 
  (∀ S, S > 0 ∧ (∀ x, f ω (x + S) = f ω x) → T ≤ S) ∧
  T = 8 * Real.pi / 5 := by
  sorry

#check smallest_positive_period

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_l429_42921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_reciprocals_l429_42967

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x+1) - 2

-- State the theorem
theorem min_value_sum_reciprocals (a m n : ℝ) : 
  a > 0 → a ≠ 1 → 
  f a (-1) = -1 →
  m * (-1) + n * (-1) + 2 = 0 →
  m * n > 0 →
  (∀ x y, m * x + n * y + 2 = 0 → f a x = y) →
  (∀ p q, p > 0 → q > 0 → m * (-1) + n * (-1) + 2 = 0 → p * (-1) + q * (-1) + 2 = 0 → 1/m + 1/n ≤ 1/p + 1/q) →
  1/m + 1/n = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_reciprocals_l429_42967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_prime_or_power_of_two_l429_42979

theorem arithmetic_sequence_prime_or_power_of_two (n : ℕ) (h_n : n > 6) :
  (∃ d : ℕ, d > 0 ∧
    (∀ a b : ℕ, 0 < a ∧ a < n ∧ 0 < b ∧ b < n ∧ Nat.Coprime a n ∧ Nat.Coprime b n →
      ∃ k : ℤ, (a : ℤ) - (b : ℤ) = k * (d : ℤ))) →
  Nat.Prime n ∨ (∃ m : ℕ, n = 2^m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_prime_or_power_of_two_l429_42979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_question_one_question_two_l429_42936

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | (x + a) * (x - 1) ≤ 0}
def N : Set ℝ := {x | 4 * x^2 - 4 * x - 3 < 0}

-- Define the complement of M in R
def notMR (a : ℝ) : Set ℝ := {x | x ≤ -a ∨ x ≥ 1}

-- Theorem for question 1
theorem question_one (a : ℝ) (h : a > 0) : 
  M a ∪ N = {x : ℝ | -2 ≤ x ∧ x < 3/2} → a = 2 := by sorry

-- Theorem for question 2
theorem question_two (a : ℝ) (h : a > 0) : 
  N ∪ notMR a = Set.univ → 0 < a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_question_one_question_two_l429_42936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l429_42928

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- A geometric sequence with its sum sequence -/
structure GeometricSequence where
  b : ℕ → ℝ
  T : ℕ → ℝ
  q : ℝ
  is_geometric : ∀ n, b (n + 1) = b n * q
  first_term : b 1 = 1
  sum_formula : ∀ n, T n = (1 - q^n) / (1 - q)

/-- The main theorem -/
theorem sequence_problem (as : ArithmeticSequence) (gs : GeometricSequence)
    (h1 : as.S 2 = as.a 3)
    (h2 : as.S 2 = gs.b 3)
    (h3 : ∃ r, as.a 3 = as.a 1 * r ∧ gs.b 4 = as.a 3 * r)
    (h4 : ∀ n, 2 * as.S n - n * as.a n = b + Real.log (2 * gs.T n + 1) / Real.log a) :
    (∀ n, as.a n = 3 * n ∧ gs.b n = 3^(n - 1)) ∧ a = 27 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l429_42928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_through_common_points_l429_42956

/-- A conic in the 2D plane --/
structure Conic where
  F : ℝ → ℝ → ℝ

/-- Two conics intersect at four points --/
def four_common_points (Γ₁ Γ₂ : Conic) : Prop :=
  ∃ (p₁ p₂ p₃ p₄ : ℝ × ℝ),
    Γ₁.F p₁.1 p₁.2 = 0 ∧ Γ₂.F p₁.1 p₁.2 = 0 ∧
    Γ₁.F p₂.1 p₂.2 = 0 ∧ Γ₂.F p₂.1 p₂.2 = 0 ∧
    Γ₁.F p₃.1 p₃.2 = 0 ∧ Γ₂.F p₃.1 p₃.2 = 0 ∧
    Γ₁.F p₄.1 p₄.2 = 0 ∧ Γ₂.F p₄.1 p₄.2 = 0 ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄

/-- Any conic passing through the four common points can be expressed as a linear combination --/
theorem conic_through_common_points (Γ₁ Γ₂ : Conic) (h : four_common_points Γ₁ Γ₂) :
  ∀ Γ : Conic, (∀ p : ℝ × ℝ, Γ₁.F p.1 p.2 = 0 ∧ Γ₂.F p.1 p.2 = 0 → Γ.F p.1 p.2 = 0) →
  ∃ l m : ℝ, ∀ x y : ℝ, Γ.F x y = l * Γ₁.F x y + m * Γ₂.F x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_through_common_points_l429_42956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rectangle_intersection_l429_42955

/-- A rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Predicate for a square rectangle -/
def Rectangle.is_square (r : Rectangle) : Prop := r.width = r.height

/-- Predicate for perpendicular sides -/
def Rectangle.perpendicular (side1 side2 : ℝ) : Prop := sorry

/-- The intersection of two rectangles -/
def Rectangle.intersection (r1 r2 : Rectangle) : Rectangle := sorry

theorem square_rectangle_intersection 
  (ABCD WXYZ : Rectangle) (AP : ℝ) :
  Rectangle.is_square ABCD ∧ 
  ABCD.width = 8 ∧
  WXYZ.width = 12 ∧ 
  WXYZ.height = 8 ∧
  Rectangle.perpendicular ABCD.height WXYZ.width ∧
  Rectangle.area (Rectangle.intersection ABCD WXYZ) = (1/2) * Rectangle.area WXYZ →
  AP = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_rectangle_intersection_l429_42955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_gt_median_l429_42978

/-- Density function type -/
def DensityFunction := ℝ → ℝ

/-- Properties of the density function -/
structure DensityProperties (f : DensityFunction) (a b : ℝ) :=
  (zero_outside : ∀ x, x < a ∨ x ≥ b → f x = 0)
  (positive_inside : ∀ x, a ≤ x ∧ x < b → f x > 0)
  (continuous : Continuous f)
  (decreasing : ∀ x y, a ≤ x ∧ x < y ∧ y < b → f x ≥ f y)

/-- Expected value of a random variable -/
noncomputable def ExpectedValue (X : ℝ → ℝ) (f : DensityFunction) : ℝ :=
  sorry

/-- Median of a random variable -/
noncomputable def Median (X : ℝ → ℝ) (f : DensityFunction) : ℝ :=
  sorry

/-- Theorem: Expected value is greater than median -/
theorem expected_value_gt_median
  (X : ℝ → ℝ) (f : DensityFunction) (a b : ℝ)
  (h : DensityProperties f a b) :
  ExpectedValue X f > Median X f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_gt_median_l429_42978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_equals_72_l429_42913

theorem factorial_ratio_equals_72 (n : ℕ) : (n + 2).factorial / n.factorial = 72 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_ratio_equals_72_l429_42913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_betting_problem_l429_42930

theorem betting_problem (X : ℝ) : 
  X > 0 → 
  X * (3/4)^3 = 37 → 
  ∃ ε > 0, |X - 87.70| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_betting_problem_l429_42930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_inequality_l429_42948

open Real

-- Define an acute triangle
def is_acute_triangle (α β γ : Real) : Prop :=
  0 < α ∧ α < Real.pi/2 ∧
  0 < β ∧ β < Real.pi/2 ∧
  0 < γ ∧ γ < Real.pi/2 ∧
  α + β + γ = Real.pi

-- Define an obtuse triangle
def is_obtuse_triangle (δ ε ζ : Real) : Prop :=
  0 < δ ∧ δ < Real.pi ∧
  0 < ε ∧ ε < Real.pi ∧
  0 < ζ ∧ ζ < Real.pi ∧
  Real.pi/2 < max δ (max ε ζ) ∧
  δ + ε + ζ = Real.pi

-- Theorem statement
theorem tangent_sum_inequality
  (α β γ δ ε ζ : Real)
  (h_acute : is_acute_triangle α β γ)
  (h_obtuse : is_obtuse_triangle δ ε ζ) :
  Real.tan α + Real.tan β + Real.tan γ ≠ Real.tan δ + Real.tan ε + Real.tan ζ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sum_inequality_l429_42948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_cut_unit_cube_l429_42934

/-- The height of a unit cube with a corner cut off, when placed on the cut face. -/
noncomputable def heightOfCutCube : ℝ :=
  2 * Real.sqrt 3 / 3

/-- Represents a cube. -/
structure Cube where
  sideLength : ℝ

/-- Represents a cut face of a cube. -/
structure CutFace where
  cube : Cube

/-- Create a unit cube. -/
def Cube.unit : Cube :=
  ⟨1⟩

/-- Create a cut face through adjacent vertices of a cube. -/
def CutFace.throughAdjacentVertices (cube : Cube) : CutFace :=
  ⟨cube⟩

/-- Calculate the height of a cube when placed on a cut face. -/
noncomputable def Cube.heightWhenPlacedOn (cube : Cube) (cutFace : CutFace) : ℝ :=
  heightOfCutCube

/-- Theorem stating the height of a unit cube with a corner cut off, when placed on the cut face. -/
theorem height_of_cut_unit_cube :
  let cube := Cube.unit
  let cutFace := CutFace.throughAdjacentVertices cube
  heightOfCutCube = Cube.heightWhenPlacedOn cube cutFace :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_of_cut_unit_cube_l429_42934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_XYZ_l429_42922

-- Define the triangle ABC
variable (A B C : ℂ)

-- Define the centers of squares
noncomputable def X : ℂ := (B + C) / 2 - (C - B) * Complex.I / 2
noncomputable def Y : ℂ := (C + A) / 2 - (A - C) * Complex.I / 2
noncomputable def Z : ℂ := (A + B) / 2 - (B - A) * Complex.I / 2

-- Define the given lengths
def AX_length : ℝ := 6
def BY_length : ℝ := 7
def CA_length : ℝ := 8

-- State the theorem
theorem area_of_triangle_XYZ :
  let triangle_area := (AX_length * BY_length) / 2
  triangle_area = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_XYZ_l429_42922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_distance_to_x_axis_l429_42910

theorem hyperbola_point_distance_to_x_axis 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (F₁ F₂ : ℝ × ℝ) 
  (h₁ : (P.1^2 / a^2) - P.2^2 = 1)  -- P is on the hyperbola
  (h₂ : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0)  -- PF₁ ⟂ PF₂
  (h₃ : F₁.2 = 0 ∧ F₂.2 = 0)  -- F₁ and F₂ are on the x-axis
  (h₄ : (F₁.1 - F₂.1)^2 / (4 * a^2) = 5/4)  -- eccentricity is √5/2
  : |P.2| = Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_distance_to_x_axis_l429_42910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_face_sum_l429_42989

def Cube := Fin 8 → ℕ

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def isValidCube (c : Cube) : Prop :=
  (∀ i : Fin 8, c i < 8) ∧
  (∀ i j : Fin 8, i ≠ j → c i ≠ c j) ∧
  (∀ e : Fin 12, isPrime (c (edge e).fst + c (edge e).snd))
where
  edge : Fin 12 → Fin 8 × Fin 8 := sorry

def faceSum (c : Cube) (f : Fin 6) : ℕ :=
  (face f).foldl (· + c ·) 0
where
  face : Fin 6 → List (Fin 8) := sorry

theorem max_face_sum (c : Cube) (h : isValidCube c) :
  ∃ f : Fin 6, faceSum c f = 18 ∧ ∀ f' : Fin 6, faceSum c f' ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_face_sum_l429_42989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_count_l429_42960

/-- Proves the number of new students who joined the class --/
theorem new_students_count 
  (original_average new_students_average average_decrease : ℝ) 
  (original_strength : ℕ) 
  (h1 : original_average = 40)
  (h2 : new_students_average = 32)
  (h3 : average_decrease = 4)
  (h4 : original_strength = 15)
  : ∃ (x : ℕ), 
    (original_strength * original_average + x * new_students_average) / (original_strength + x) = 
    original_average - average_decrease ∧ 
    x = 15 := by
  sorry

#check new_students_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_count_l429_42960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parabola_intersection_l429_42938

/-- The circle equation -/
def circle_eq (x y b : ℝ) : Prop := x^2 + y^2 = b^2

/-- The parabola equation -/
def parabola_eq (x y b : ℝ) : Prop := y = x^2 - 2*b

/-- The number of intersection points between the circle and parabola -/
noncomputable def intersection_count (b : ℝ) : ℕ := sorry

/-- Theorem stating the condition for exactly three intersection points -/
theorem circle_parabola_intersection (b : ℝ) :
  intersection_count b = 3 ↔ b > 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parabola_intersection_l429_42938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l429_42923

def sequence_a : ℕ → ℚ
  | 0 => 1/2  -- Add a case for 0 to cover all natural numbers
  | n + 1 => (3 * sequence_a n) / (sequence_a n + 3)

theorem sequence_a_formula (n : ℕ) :
  sequence_a n = 3 / (n + 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l429_42923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adoption_time_for_puppies_l429_42925

/-- The number of days needed to adopt all puppies -/
def adoption_days (initial_puppies : ℕ) (additional_puppies : ℕ) (adoptions_per_day : ℕ) : ℕ :=
  (((initial_puppies + additional_puppies : ℚ) / adoptions_per_day).ceil).toNat

/-- Theorem stating that it takes 12 days to adopt all puppies under given conditions -/
theorem adoption_time_for_puppies :
  adoption_days 35 48 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_adoption_time_for_puppies_l429_42925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_problem_l429_42971

theorem set_intersection_problem : Set ℕ := by
  -- Define sets A and B
  let A : Set ℕ := {0, 1, 2, 3, 4, 5}
  let B : Set ℕ := {2, 4, 6}
  
  -- State and prove the theorem
  have : A ∩ B = {2, 4} := by
    -- The proof goes here
    sorry
  
  -- Return the result
  exact {2, 4}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_problem_l429_42971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l429_42929

noncomputable def f (x θ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + θ) + Real.cos (2 * x + θ)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.cos (2 * x - 2 * Real.pi / 3)

theorem min_value_g (θ : ℝ) (h : 0 < θ ∧ θ < Real.pi) :
  ∃ (x : ℝ), -Real.pi/4 ≤ x ∧ x ≤ Real.pi/6 ∧
  (∀ (y : ℝ), -Real.pi/4 ≤ y ∧ y ≤ Real.pi/6 → g x ≤ g y) ∧
  g x = -2 := by
  sorry

#check min_value_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_g_l429_42929
