import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yolas_past_weight_theorem_weight_conditions_l316_31699

/-- Yola's weight 2 years ago given current weights and a known difference -/
def yolas_past_weight (yola_current : ℕ) (wanda_difference : ℕ) (D : ℕ) : ℕ :=
  yola_current + wanda_difference - D

theorem yolas_past_weight_theorem (yola_current : ℕ) (wanda_difference : ℕ) (D : ℕ) :
  yolas_past_weight yola_current wanda_difference D = yola_current + wanda_difference - D :=
by
  -- Unfold the definition of yolas_past_weight
  unfold yolas_past_weight
  -- The equality holds by definition
  rfl

/-- Assertions about the given conditions -/
theorem weight_conditions :
  ∃ (yola_current wanda_difference D : ℕ),
    yola_current = 220 ∧
    wanda_difference = 30 ∧
    D = (yola_current + wanda_difference) - yolas_past_weight yola_current wanda_difference D :=
by
  -- We use 'sorry' to skip the proof of existence
  sorry

#check yolas_past_weight
#check yolas_past_weight_theorem
#check weight_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yolas_past_weight_theorem_weight_conditions_l316_31699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_2_l316_31659

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 2*y + 8 = 0

/-- The radius of the circle -/
noncomputable def CircleRadius : ℝ := Real.sqrt 2

/-- Theorem stating that the given equation represents a circle with radius sqrt(2) -/
theorem circle_radius_is_sqrt_2 :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = CircleRadius^2 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_2_l316_31659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_to_bike_speed_ratio_l316_31657

noncomputable def tractor_distance : ℝ := 575
noncomputable def tractor_time : ℝ := 23
noncomputable def car_distance : ℝ := 540
noncomputable def car_time : ℝ := 6

noncomputable def tractor_speed : ℝ := tractor_distance / tractor_time
noncomputable def bike_speed : ℝ := 2 * tractor_speed
noncomputable def car_speed : ℝ := car_distance / car_time

theorem car_to_bike_speed_ratio :
  car_speed / bike_speed = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_to_bike_speed_ratio_l316_31657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_perpendicular_to_ground_l316_31650

-- Define the ground as a plane
def ground : Set (Fin 3 → ℝ) := sorry

-- Define the flagpole as a line
def flagpole : Set (Fin 3 → ℝ) := sorry

-- Define the property of being vertical
def is_vertical (line : Set (Fin 3 → ℝ)) : Prop := sorry

-- Define the property of being perpendicular to a plane
def perpendicular_to_plane (line : Set (Fin 3 → ℝ)) (plane : Set (Fin 3 → ℝ)) : Prop := sorry

-- Theorem statement
theorem flagpole_perpendicular_to_ground :
  is_vertical flagpole → perpendicular_to_plane flagpole ground := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_perpendicular_to_ground_l316_31650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l316_31630

theorem vector_collinearity (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0)) :
  (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt (a.1^2 + a.2^2) - Real.sqrt (b.1^2 + b.2^2)) →
  ∃ k : ℝ, a = (k * b.1, k * b.2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l316_31630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_6_8346_l316_31696

/-- Rounds a number to the nearest hundredth -/
noncomputable def roundToHundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

/-- The original number to be rounded -/
def originalNumber : ℝ := 6.8346

theorem round_to_hundredth_6_8346 :
  roundToHundredth originalNumber = 6.83 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_6_8346_l316_31696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_sectors_area_l316_31651

-- Define the radius and angles
def radius : ℝ := 10
def angle1 : ℝ := 90
def angle2 : ℝ := 45

-- Define the area of a circular sector
noncomputable def sectorArea (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * Real.pi * r^2

-- Theorem statement
theorem combined_sectors_area :
  sectorArea radius angle1 + sectorArea radius angle2 = 37.5 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_sectors_area_l316_31651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_theorem_l316_31636

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  ∃ (a b c d : ℕ), ({a, b, c, d} : Finset ℕ) = {0, 3, 4, 8} ∧
  n = 1000 * a + 100 * b + 10 * c + d

def difference (a b : ℕ) : ℕ := max a b - min a b

theorem greatest_difference_theorem :
  ∃ (max_diff : ℕ),
    max_diff = 5382 ∧
    ∀ (a b : ℕ), is_valid_number a → is_valid_number b →
      difference a b ≤ max_diff := by
  sorry

#check greatest_difference_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_theorem_l316_31636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alyssa_flyers_l316_31673

/-- Proves that Alyssa passed out 67 flyers given the distribution of flyers among friends. -/
theorem alyssa_flyers (total : ℕ) (ryan : ℕ) (scott : ℕ) (belinda_percent : ℚ) 
  (h1 : total = 200)
  (h2 : ryan = 42)
  (h3 : scott = 51)
  (h4 : belinda_percent = 1/5)
  (h5 : belinda_percent * ↑total = (total : ℚ) * belinda_percent) : 
  total - (ryan + scott + (belinda_percent * ↑total).floor) = 67 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alyssa_flyers_l316_31673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l316_31670

-- Define the necessary functions and types
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry
noncomputable def eccentricity (h : Set (ℝ × ℝ)) : ℝ := sorry

theorem hyperbola_eccentricity (a b c : ℝ) (F₁ F₂ P : ℝ × ℝ) (h : Set (ℝ × ℝ)) :
  a > 0 →
  b > 0 →
  (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x, y) ∈ h) →
  F₁ = (-c, 0) →
  F₂ = (c, 0) →
  P ∈ {P : ℝ × ℝ | P ∈ h ∧ P.1 > 0} →
  angle F₁ P F₂ = 60 * π / 180 →
  area_triangle F₁ P F₂ = Real.sqrt 3 * a * c →
  eccentricity h = (1 + Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l316_31670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l316_31616

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Proves that a train 150 meters long, traveling at 36 kmph, will take 30 seconds to cross a bridge 150 meters long -/
theorem train_crossing_bridge : train_crossing_time 150 150 36 = 30 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_crossing_time 150 150 36

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l316_31616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l316_31656

theorem inequality_solution_set (x : ℝ) : (9 : ℝ)^x > (3 : ℝ)^(x-2) ↔ x > -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l316_31656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_imaginary_axis_length_l316_31668

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The length of the imaginary axis of a hyperbola -/
def imaginary_axis_length (h : Hyperbola) : ℝ := 2 * h.b

theorem hyperbola_imaginary_axis_length 
  (h : Hyperbola) 
  (h_distances : ∃ (p : ℝ × ℝ), p.1^2 / h.a^2 - p.2^2 / h.b^2 = 1 ∧ 
    ∃ (f1 f2 : ℝ × ℝ), 
      (p.1 - f1.1)^2 + (p.2 - f1.2)^2 = 100 ∧ 
      (p.1 - f2.1)^2 + (p.2 - f2.2)^2 = 16)
  (h_eccentricity : eccentricity h = 2) :
  imaginary_axis_length h = 6 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_imaginary_axis_length_l316_31668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l316_31611

/-- Given an ellipse with equation x²/m + y²/3 = 1, focus on x-axis, and eccentricity 1/2, m = 4 -/
theorem ellipse_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m + y^2 / 3 = 1) →  -- Ellipse equation
  (∃ c : ℝ, c > 0 ∧ c < Real.sqrt m ∧ 
    ∀ x y : ℝ, x^2 / m + y^2 / 3 = 1 → 
    (x + c)^2 / m + y^2 / 3 = 1 ∨ (x - c)^2 / m + y^2 / 3 = 1) →  -- Focus on x-axis
  (Real.sqrt (m - 3) / Real.sqrt m = 1 / 2) →  -- Eccentricity is 1/2
  m = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l316_31611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tyson_lake_speed_l316_31614

/-- Tyson's swimming speeds and race details -/
structure SwimmerData where
  ocean_speed : ℚ
  total_races : ℕ
  race_distance : ℚ
  total_time : ℚ

/-- Calculate Tyson's lake speed given his race data -/
noncomputable def calculate_lake_speed (data : SwimmerData) : ℚ :=
  let lake_races := data.total_races / 2
  let ocean_races := data.total_races / 2
  let lake_distance := (lake_races : ℚ) * data.race_distance
  let ocean_distance := (ocean_races : ℚ) * data.race_distance
  let ocean_time := ocean_distance / data.ocean_speed
  let lake_time := data.total_time - ocean_time
  lake_distance / lake_time

/-- Theorem stating that Tyson's lake speed is 3 mph given the provided data -/
theorem tyson_lake_speed :
  let data : SwimmerData := {
    ocean_speed := 5/2,
    total_races := 10,
    race_distance := 3,
    total_time := 11
  }
  calculate_lake_speed data = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tyson_lake_speed_l316_31614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_cardinality_S_l316_31602

def digit_sum (n : ℕ) : ℕ := sorry

def S : Set ℕ := {n | digit_sum n = 15 ∧ 0 ≤ n ∧ n < 10^8}

noncomputable def S_finset : Finset ℕ := sorry

theorem digit_sum_of_cardinality_S : digit_sum (Finset.card S_finset) = 33 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_of_cardinality_S_l316_31602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inclination_range_l316_31689

noncomputable def f (x : ℝ) := Real.cos x

theorem tangent_inclination_range :
  ∀ x α : ℝ,
  (0 ≤ α ∧ α < π ∧ α ≠ π / 2) →
  (Real.tan α = -(Real.sin x)) →
  (α ∈ Set.Icc 0 (π / 4) ∪ Set.Ico (3 * π / 4) π) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inclination_range_l316_31689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l316_31687

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2

-- Define the point on the curve
def point : ℝ × ℝ := (1, 0)

-- Define the slope of the tangent line at the point
noncomputable def tangent_slope : ℝ := curve_derivative point.fst

-- Define the slope of the perpendicular line
noncomputable def perpendicular_slope : ℝ := -1 / tangent_slope

-- Define the equation of the line
noncomputable def line_equation (x : ℝ) : ℝ := perpendicular_slope * (x - point.fst) + point.snd

-- Theorem statement
theorem perpendicular_line_equation :
  ∀ x : ℝ, line_equation x = -1/3 * x + 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l316_31687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l316_31648

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧ 
    ∀ (S : ℝ), S > 0 ∧ (∀ (x : ℝ), f (x + S) = f x) → T ≤ S) ∧
  (∀ (x : ℝ), -π/4 ≤ x ∧ x ≤ π/6 → f x ≤ 3) ∧
  (∀ (x : ℝ), -π/4 ≤ x ∧ x ≤ π/6 → f x ≥ 1 - Real.sqrt 3) ∧
  (∃ (x : ℝ), -π/4 ≤ x ∧ x ≤ π/6 ∧ f x = 3) ∧
  (∃ (x : ℝ), -π/4 ≤ x ∧ x ≤ π/6 ∧ f x = 1 - Real.sqrt 3) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l316_31648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_and_average_l316_31665

noncomputable def y (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 5 then 26 * (x : ℝ) - 56
  else if 5 < x ∧ x ≤ 12 then 210 - 20 * (x : ℝ)
  else 0

noncomputable def w (x : ℕ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 5 then 13 * (x : ℝ) - 43
  else if 5 < x ∧ x ≤ 12 then -10 * (x : ℝ) + 200 - 640 / (x : ℝ)
  else 0

theorem max_profit_and_average :
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 12 → y x ≤ y 6) ∧
  y 6 = 90 ∧
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 12 → w x ≤ w 8) := by
  sorry

#check max_profit_and_average

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_and_average_l316_31665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_seven_fifths_l316_31660

noncomputable section

/-- The curve in polar coordinates -/
def curve (θ : ℝ) : ℝ := Real.sqrt 2 * Real.cos (θ + Real.pi/4)

/-- The line in parametric form -/
def line (t : ℝ) : ℝ × ℝ := (1 + 4*t, -1 - 3*t)

/-- The length of the chord cut by the line from the curve -/
def chord_length : ℝ := 7/5

/-- Theorem stating that the chord length is 7/5 -/
theorem chord_length_is_seven_fifths :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧
  (∃ (θ₁ θ₂ : ℝ), 
    (curve θ₁ * Real.cos θ₁, curve θ₁ * Real.sin θ₁) = line t₁ ∧
    (curve θ₂ * Real.cos θ₂, curve θ₂ * Real.sin θ₂) = line t₂ ∧
    Real.sqrt ((line t₁).1 - (line t₂).1)^2 + ((line t₁).2 - (line t₂).2)^2 = chord_length) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_seven_fifths_l316_31660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_side_length_l316_31693

-- Define a triangle with three different integer side lengths and perimeter 24
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a < b
  h2 : b < c
  h3 : a + b + c = 24

-- Define the triangle inequality
def satisfies_triangle_inequality (t : Triangle) : Prop :=
  (t.a + t.b > t.c) ∧
  (t.a + t.c > t.b) ∧
  (t.b + t.c > t.a)

-- Theorem statement
theorem max_side_length (t : Triangle) (h : satisfies_triangle_inequality t) :
  t.c ≤ 11 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_side_length_l316_31693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_total_distance_l316_31685

noncomputable def circle_radius : ℝ := 50

def num_children : ℕ := 8

def distance_to_adjacent (r : ℝ) : ℝ := r

noncomputable def distance_to_skip_one (r : ℝ) : ℝ := r * Real.sqrt 2

noncomputable def distance_to_skip_two (r : ℝ) : ℝ := r * Real.sqrt (2 + 2 * Real.sqrt 2)

def distance_to_opposite (r : ℝ) : ℝ := 2 * r

noncomputable def total_distance_per_child (r : ℝ) : ℝ :=
  2 * distance_to_skip_one r + 2 * distance_to_skip_two r + distance_to_opposite r

theorem least_total_distance :
  num_children * total_distance_per_child circle_radius =
    800 * Real.sqrt 2 + 800 * Real.sqrt (2 + 2 * Real.sqrt 2) + 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_total_distance_l316_31685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_minimum_triangle_area_minimum_area_at_negative_half_unique_minimum_l316_31662

noncomputable section

-- Define the family of lines
def line (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 - 2 * k = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (2, 1)

-- Define the area of triangle AOB
noncomputable def triangle_area (k : ℝ) : ℝ := (1/2) * (2 - 1/k) * (1 - 2*k)

theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line k (fixed_point.1) (fixed_point.2) := by sorry

theorem minimum_triangle_area :
  ∀ k : ℝ, k < 0 → triangle_area k ≥ 4 := by sorry

theorem minimum_area_at_negative_half :
  ∃ k : ℝ, k = -1/2 ∧ triangle_area k = 4 := by sorry

theorem unique_minimum :
  ∀ k : ℝ, k < 0 → (triangle_area k = 4 ↔ k = -1/2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_minimum_triangle_area_minimum_area_at_negative_half_unique_minimum_l316_31662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_conditions_l316_31663

noncomputable section

/-- A curve with equation y = (px^2 + qx + r) / (sx^2 + tx + u) -/
def curve (p q r s t u : ℝ) (x : ℝ) : ℝ :=
  (p * x^2 + q * x + r) / (s * x^2 + t * x + u)

/-- The line y = x -/
def line (x : ℝ) : ℝ := x

/-- Symmetry about y = x means that if (a, b) is on the curve, then (b, a) is also on the curve -/
def symmetricAboutYEqX (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f a = b → f b = a

theorem curve_symmetry_conditions 
  (p q r s t u : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0) :
  symmetricAboutYEqX (curve p q r s t u) → p = s ∧ q = t ∧ r = u := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_symmetry_conditions_l316_31663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_condition_l316_31617

/-- Given a parabola y = x^2 and a point (0, c) where c > 0, if there exist two chords of length 2 
    passing through (0, c), then c can be any positive real number. -/
theorem parabola_chord_length_condition (c : ℝ) : c > 0 → 
  (∃ (k₁ k₂ : ℝ), k₁ ≠ k₂ ∧ 
    (∀ (x y : ℝ), y = x^2 → (y - c = k₁ * x → (x^2 + (y - c)^2 = 4)) ∧
                            (y - c = k₂ * x → (x^2 + (y - c)^2 = 4)))) →
  c ∈ Set.Ioi 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_condition_l316_31617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_with_equilateral_face_l316_31619

-- Define the regular octagon
def regular_octagon (s : ℝ) : Set (ℝ × ℝ) :=
  sorry

-- Define the right pyramid
def right_pyramid (base : Set (ℝ × ℝ)) (height : ℝ) : Set (ℝ × ℝ × ℝ) :=
  sorry

-- Define the volume of a pyramid
noncomputable def pyramid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- Theorem statement
theorem pyramid_volume_with_equilateral_face :
  ∀ (s : ℝ),
  s = 10 →
  let base := regular_octagon s
  let base_area := 200 * Real.sqrt 2
  let height := 5 * Real.sqrt 3
  let pyramid := right_pyramid base height
  pyramid_volume base_area height = (1000 * Real.sqrt 6) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_with_equilateral_face_l316_31619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l316_31625

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the evenness of f
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the condition on f''(x)cos(x) + f(x)sin(x)
def satisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 (Real.pi/2), deriv^[2] f x * Real.cos x + f x * Real.sin x > 0

-- State the theorem
theorem f_inequality (h1 : isEven f) (h2 : satisfiesCondition f) :
  f (Real.pi/6) < Real.sqrt 3 * f (Real.pi/3) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l316_31625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_second_derivative_extrema_l316_31652

open Real Set

/-- Given a function f(x) = x^4 * cos(x) + mx^2 + 2x where m ∈ ℝ,
    if the maximum value of f''(x) on [-4, 4] is 16,
    then the minimum value of f''(x) on [-4, 4] is -12 -/
theorem function_second_derivative_extrema 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ x, f x = x^4 * cos x + m * x^2 + 2 * x) 
  (h2 : ∃ (c : ℝ), c ∈ Icc (-4) 4 ∧ ∀ x ∈ Icc (-4) 4, (deriv^[2] f) x ≤ (deriv^[2] f) c)
  (h3 : ∃ (c : ℝ), c ∈ Icc (-4) 4 ∧ (deriv^[2] f) c = 16) :
  ∃ (d : ℝ), d ∈ Icc (-4) 4 ∧ (deriv^[2] f) d = -12 ∧ 
  ∀ x ∈ Icc (-4) 4, (deriv^[2] f) x ≥ (deriv^[2] f) d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_second_derivative_extrema_l316_31652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_50th_term_l316_31626

/-- A sequence defined by f(n) = 2f(n-1) + 4 with f(1) = 0 -/
def f : ℕ → ℤ
  | 0 => 0  -- Add this case for n = 0
  | 1 => 0
  | n + 2 => 2 * f (n + 1) + 4

/-- The 50th term of the sequence f is equal to 2^51 - 4 -/
theorem f_50th_term : f 50 = 2^51 - 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_50th_term_l316_31626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_with_restriction_l316_31628

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def total_arrangements (n : ℕ) : ℕ := 
  factorial n

def arrangements_with_pair_together (n : ℕ) : ℕ := 
  factorial (n - 1) * factorial 2

theorem seating_arrangements_with_restriction : 
  total_arrangements 8 - arrangements_with_pair_together 8 = 30240 :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#eval total_arrangements 8 - arrangements_with_pair_together 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seating_arrangements_with_restriction_l316_31628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l316_31677

/-- Definition of the ellipse E -/
noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- The main theorem -/
theorem ellipse_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0)
    (h3 : ellipse a b 1 (3/2))
    (h4 : eccentricity a b = 1/2) :
  (∀ x y, ellipse 2 (Real.sqrt 3) x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  ∃ k m : ℝ, ∀ x y,
    ellipse 2 (Real.sqrt 3) x y ∧
    y = k * x + m ∧
    x ≠ 2 ∧
    (∃ x1 y1 x2 y2 : ℝ,
      ellipse 2 (Real.sqrt 3) x1 y1 ∧
      ellipse 2 (Real.sqrt 3) x2 y2 ∧
      y1 = k * x1 + m ∧
      y2 = k * x2 + m ∧
      (x1 - x2)^2 + (y1 - y2)^2 = (x1 - 2)^2 + y1^2 + (x2 - 2)^2 + y2^2) →
    x = 2/7 ∧ y = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l316_31677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_interval_l316_31672

def g (x : ℝ) : ℝ := x * (2 - x)

theorem g_increasing_interval :
  ∀ a b : ℝ, a < b →
    (∀ x, x < 1 → g x < g (x + (b - a))) ∧
    (∀ x, x > 1 → g x > g (x + (b - a))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_interval_l316_31672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_unfolding_theorem_l316_31690

-- Define a square
structure Square where
  side : ℝ
  area : ℝ := side * side

-- Define a cube
structure Cube where
  edge : ℝ
  faces : Fin 6 → Square
  face_congruence : ∀ i j : Fin 6, (faces i).side = (faces j).side

-- Define an unfolded shape
structure UnfoldedShape where
  squares : Fin 6 → Square
  connected : Prop

-- Define a function to check if an unfolded shape can form a cube
def can_form_cube (shape : UnfoldedShape) : Prop :=
  ∃ (cube : Cube), (∀ i : Fin 6, shape.squares i = cube.faces i) ∧ shape.connected

-- Define the specific shapes from Figures 5a and 5b
noncomputable def shape_5a : UnfoldedShape := sorry
noncomputable def shape_5b : UnfoldedShape := sorry

-- Theorem statement
theorem cube_unfolding_theorem :
  ∃ (cube : Cube),
    can_form_cube shape_5a ∧
    can_form_cube shape_5b ∧
    shape_5a ≠ shape_5b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_unfolding_theorem_l316_31690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fraction_integer_l316_31694

def a : ℕ → ℕ
  | 0 => 1  -- Add this case to cover Nat.zero
  | 1 => 1
  | n + 1 => if n % 2 = 1 then 2 * a n else a n + 1

def S : ℕ → ℕ
  | 0 => 0
  | n + 1 => S n + a (n + 1)

theorem sequence_fraction_integer (n : ℕ) :
  n > 0 ∧ (∃ k : ℕ, S n = k * a n) ↔ n = 1 ∨ n = 3 ∨ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_fraction_integer_l316_31694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l316_31623

theorem binomial_expansion_coefficient (x : ℝ) : 
  let expansion := (x + 2 / x^2)^6
  let fourth_term_coefficient := Nat.choose 6 3 * 2^3
  fourth_term_coefficient = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l316_31623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_squared_l316_31695

/-- An ellipse with the given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on the ellipse -/
structure Point (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The focal distance of the ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ := Real.sqrt (e.a^2 - e.b^2)

/-- The left focus of the ellipse -/
noncomputable def left_focus (e : Ellipse) : ℝ × ℝ := (-focal_distance e, 0)

/-- The right focus of the ellipse -/
noncomputable def right_focus (e : Ellipse) : ℝ × ℝ := (focal_distance e, 0)

/-- Theorem: Given an ellipse with the specified properties, its eccentricity squared equals 2 - √3 -/
theorem ellipse_eccentricity_squared (e : Ellipse) (P : Point e) (Q : ℝ × ℝ) :
  (P.x = focal_distance e) →
  (P.y = e.b^2 / e.a) →
  (Q.1 = focal_distance e / 3) →
  (Q.2 = 2 * e.b^2 / (3 * e.a)) →
  (eccentricity e)^2 = 2 - Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_squared_l316_31695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_slope_l316_31618

/-- A line passing through (-1, 0) intersecting a circle --/
structure IntersectingLine where
  slope : ℝ
  passes_through_neg_one_zero : slope * (-1 + 1) = 0

/-- The circle x^2 + y^2 - 4x = 0 --/
def Circle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 - 4*p.1 = 0}

/-- The center of the circle --/
def CircleCenter : ℝ × ℝ := (2, 0)

/-- The radius of the circle --/
def CircleRadius : ℝ := 2

/-- Distance from a point to a line --/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : IntersectingLine) : ℝ :=
  |3 * l.slope| / Real.sqrt (l.slope^2 + 1)

/-- Theorem: If a line passing through (-1, 0) intersects the circle x^2 + y^2 - 4x = 0
    such that it forms an equilateral triangle with the center of the circle,
    then the slope of the line is ±√2/2 --/
theorem intersecting_line_slope (l : IntersectingLine) :
  (∃ A B : ℝ × ℝ, A ∈ Circle ∧ B ∈ Circle ∧
    distancePointToLine CircleCenter l = CircleRadius) →
  l.slope = Real.sqrt 2 / 2 ∨ l.slope = -Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersecting_line_slope_l316_31618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_theorem_l316_31667

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem ellipse_foci_theorem (e : Ellipse) : 
  let p := Point.mk 1 (Real.sqrt 3 / 2)
  let f1 := Point.mk (-Real.sqrt 3) 0
  let f2 := Point.mk (Real.sqrt 3) 0
  (p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1) → 
  (distance p f1 + distance p f2 = 4) →
  (e.a = 2 ∧ e.b = 1 ∧ 
   f1 = Point.mk (-Real.sqrt 3) 0 ∧ 
   f2 = Point.mk (Real.sqrt 3) 0) :=
by sorry

#check ellipse_foci_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_theorem_l316_31667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_smallest_angles_sum_l316_31642

/-- A quadrilateral with special angle properties -/
structure SpecialQuadrilateral where
  /-- The four angles of the quadrilateral -/
  angles : Fin 4 → ℝ
  /-- The angles form an arithmetic progression -/
  is_arithmetic : ∃ (a d : ℝ), ∀ i, angles i = a + i * d
  /-- The sum of all angles is 360° -/
  sum_360 : (angles 0) + (angles 1) + (angles 2) + (angles 3) = 360
  /-- One angle is double another -/
  double_angle : ∃ (i j : Fin 4), i ≠ j ∧ angles i = 2 * angles j
  /-- Two triangles in the quadrilateral are similar -/
  similar_triangles : ∃ (i j k l : Fin 4), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    angles i = angles k ∧ (angles j = 2 * angles i ∨ angles l = 2 * angles k)
  /-- The angles of one triangle form an arithmetic progression -/
  triangle_arithmetic : ∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
    ∃ (a d : ℝ), angles i = a ∧ angles j = a + d ∧ angles k = a + 2*d

/-- The sum of the two smallest angles in a SpecialQuadrilateral is 90° -/
theorem special_quadrilateral_smallest_angles_sum
  (q : SpecialQuadrilateral) : ∃ (i j : Fin 4), i ≠ j ∧
  q.angles i + q.angles j = 90 ∧
  ∀ (k : Fin 4), k ≠ i → k ≠ j → q.angles k ≥ q.angles i ∧ q.angles k ≥ q.angles j := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_quadrilateral_smallest_angles_sum_l316_31642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_l316_31684

def P (n : ℕ) : ℕ := Nat.factors n |>.foldl max 0

theorem no_solution : ¬ ∃ n : ℕ, 1 < n ∧ n < 100 ∧ P n = 5 ∧ P (n + 36) = Nat.sqrt (n + 36) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_l316_31684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mono_inc_func_properties_l316_31643

/-- A monotonically increasing function satisfying the given properties -/
structure MonoIncFunc where
  f : ℝ → ℝ
  mono : ∀ {x y : ℝ}, x < y → f x < f y
  add_log : ∀ (x y : ℝ), f (x * y) = f x + f y
  f_3 : f 3 = 1
  pos_dom : ∀ x, f x > 0 → x > 0

theorem mono_inc_func_properties (φ : MonoIncFunc) :
  (φ.f 1 = 0) ∧
  (∀ x : ℝ, (φ.f x + φ.f (x - 8) ≤ 2 ↔ 8 < x ∧ x ≤ 9)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mono_inc_func_properties_l316_31643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_framed_painting_ratio_l316_31691

/-- The ratio of the smaller dimension to the larger dimension of a framed painting -/
noncomputable def framed_painting_ratio (painting_width painting_height frame_side_width : ℝ) : ℝ :=
  let framed_width := painting_width + 2 * frame_side_width
  let framed_height := painting_height + 6 * frame_side_width
  min framed_width framed_height / max framed_width framed_height

/-- Theorem stating the ratio for the specific painting and frame conditions -/
theorem specific_framed_painting_ratio :
  ∃ (frame_side_width : ℝ),
    frame_side_width > 0 ∧
    (20 + 2 * frame_side_width) * (30 + 6 * frame_side_width) - 20 * 30 = 2 * (20 * 30) ∧
    framed_painting_ratio 20 30 frame_side_width = 1 / 2 := by
  sorry

#check specific_framed_painting_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_framed_painting_ratio_l316_31691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l316_31697

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 * Real.tan x + Real.cos (2 * x)

-- Theorem statement
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ (M = Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l316_31697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l316_31681

/-- The function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + m * x^2 - 3 * x + 1

/-- Theorem stating the properties of the function f and its extreme points -/
theorem function_properties (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), x₁ < x₂ ∧
  (∀ x, f m x ≥ f m x₁) ∧
  (∀ x, f m x ≥ f m x₂) ∧
  (x₁ + x₂) / (x₁ * x₂) = 2/3 →
  (m = 1 ∧
   ∀ x ∈ Set.Icc 0 3, f m x ≥ -2/3 ∧ f m x ≤ 10 ∧
   ∃ y ∈ Set.Icc 0 3, f m y = -2/3 ∧
   ∃ z ∈ Set.Icc 0 3, f m z = 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l316_31681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_b_2023_l316_31644

def b : ℕ → ℚ
  | 0 => 2  -- Adding this case to cover Nat.zero
  | 1 => 2
  | 2 => 1/3
  | n+3 => (b (n+1) * b (n+2)) / (3 * b (n+1) + 2 * b (n+2))

theorem b_formula (n : ℕ) (h : n ≥ 1) : b n = 2 / (3 * n - 1) := by
  sorry

theorem b_2023 : b 2023 = 2 / 6068 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_b_2023_l316_31644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_line_l316_31639

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 5 = 1

noncomputable def inside_point : ℝ × ℝ := (Real.sqrt 5, Real.sqrt 2)

noncomputable def line_equation (x y : ℝ) : Prop := 
  (Real.sqrt 5 / 9) * x + (Real.sqrt 2 / 5) * y = 1

theorem tangent_intersection_line 
  (A B C D : ℝ × ℝ) 
  (hA : ellipse A.1 A.2) 
  (hB : ellipse B.1 B.2) 
  (hC : ellipse C.1 C.2) 
  (hD : ellipse D.1 D.2) 
  (hAB : ∃ (t : ℝ), 
    (inside_point.1 + t * (A.1 - inside_point.1), 
     inside_point.2 + t * (A.2 - inside_point.2)) = B)
  (hCD : ∃ (t : ℝ), 
    (inside_point.1 + t * (C.1 - inside_point.1), 
     inside_point.2 + t * (C.2 - inside_point.2)) = D)
  (E : ℝ × ℝ) 
  (F : ℝ × ℝ) :
  line_equation E.1 E.2 ∧ line_equation F.1 F.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_line_l316_31639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x₀_l316_31606

/-- The function for which we're finding the tangent line -/
noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x + 1

/-- The point at which we're finding the tangent line -/
def x₀ : ℝ := 1

/-- The proposed equation of the tangent line -/
def tangent_line (x : ℝ) : ℝ := x + 2

theorem tangent_line_at_x₀ :
  (deriv f) x₀ • (X - x₀) + f x₀ = tangent_line X := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_x₀_l316_31606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_coloring_inequality_l316_31637

/-- A grid of cells that can be colored black or white -/
structure Grid (m n : ℕ) where
  cells : Fin m → Fin n → Bool

/-- A path from left to right in the grid -/
def GridPath (m n : ℕ) := List (Fin m × Fin n)

/-- Check if a path is valid (all cells are adjacent and black) -/
def isValidPath (g : Grid m n) (p : GridPath m n) : Prop := sorry

/-- Check if two paths are non-intersecting -/
def areNonIntersecting (p1 p2 : GridPath m n) : Prop := sorry

/-- The number of ways to color the grid with at least one black path -/
noncomputable def N (m n : ℕ) : ℕ := sorry

/-- The number of ways to color the grid with two non-intersecting black paths -/
noncomputable def M (m n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem grid_coloring_inequality (m n : ℕ) : (N m n)^2 ≥ 2^(m*n) * (M m n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_coloring_inequality_l316_31637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_eleven_pi_sixths_l316_31601

theorem tan_eleven_pi_sixths : Real.tan (11 * π / 6) = -(Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_eleven_pi_sixths_l316_31601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_l316_31641

-- Define the fractions
noncomputable def f1 (x : ℝ) := 10 / (15 * x)
noncomputable def f2 (a b : ℝ) := (2 * a * b) / (3 * a^2)
noncomputable def f3 (x : ℝ) := (x + 1) / (3 * x + 3)
noncomputable def f4 (x : ℝ) := (x + 1) / (x^2 + 1)

-- Define a predicate for simplifiable fractions
def is_simplifiable (f : ℝ → ℝ) : Prop :=
  ∃ (g : ℝ → ℝ), ∀ x, f x = g x ∧ g ≠ f

-- Theorem statement
theorem simplest_fraction :
  is_simplifiable f1 ∧
  (∀ a, is_simplifiable (f2 a)) ∧
  is_simplifiable f3 ∧
  ¬is_simplifiable f4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_fraction_l316_31641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l316_31631

/-- Given a sequence {a_n} where the sum of the first n terms is S_n,
    and the line x + y - 2n = 0 (n ∈ ℕ+) passes through the point (a_n, S_n),
    prove that the general term formula for a_n is (2^n - 1) / 2^(n-1). -/
theorem sequence_general_term (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) :
  (∀ n : ℕ+, a n + S n = 2 * (n : ℚ)) →
  (∀ n : ℕ+, a n = (2^(n : ℕ) - 1 : ℚ) / 2^((n : ℕ) - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_general_term_l316_31631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_elements_in_A_l316_31608

-- Define the set A
def A (k : ℝ) : Set ℤ :=
  {x : ℤ | (k * x - k^2 - 6) * (x - 4) > 0}

-- Define a function that returns true if the set A has finite elements
def has_finite_elements (k : ℝ) : Prop :=
  (A k).Finite

-- Theorem statement
theorem minimize_elements_in_A :
  ∀ k : ℝ, has_finite_elements k ↔ k < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_elements_in_A_l316_31608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_primitive_existence_l316_31654

-- Define the problem statement
theorem primitive_existence 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h_order : a < c ∧ c < b)
  (h_continuous : ContinuousOn f (Set.Icc a b))
  (h_primitive_left : ∃ F₁ : ℝ → ℝ, ∀ x ∈ Set.Ico a c, HasDerivAt F₁ (f x) x)
  (h_primitive_right : ∃ F₂ : ℝ → ℝ, ∀ x ∈ Set.Ioc c b, HasDerivAt F₂ (f x) x) :
  ∃ F : ℝ → ℝ, ∀ x ∈ Set.Icc a b, HasDerivAt F (f x) x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_primitive_existence_l316_31654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rightTriangleProbabilityIsThreeSevenths_l316_31646

/-- A circle with 8 equidistant points -/
structure Circle where
  points : Fin 8 → ℝ × ℝ
  equidistant : ∀ i j : Fin 8, dist (points i) (points j) = dist (points i.succ) (points j.succ)

/-- The probability of selecting 3 points that form a right triangle -/
def rightTriangleProbability (c : Circle) : ℚ :=
  3 / 7

/-- The theorem stating that the probability is indeed 3/7 -/
theorem rightTriangleProbabilityIsThreeSevenths (c : Circle) :
  rightTriangleProbability c = 3 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rightTriangleProbabilityIsThreeSevenths_l316_31646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l316_31680

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2 * x else x + 1

-- State the theorem
theorem solve_equation (a : ℝ) : f a + f 1 = 0 → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l316_31680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l316_31603

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  hypotenuse : ℝ
  hypotenuse_positive : hypotenuse > 0

/-- Calculates the area of an isosceles right triangle given its hypotenuse -/
noncomputable def area (t : IsoscelesRightTriangle) : ℝ :=
  (t.hypotenuse ^ 2) / 4

/-- Theorem: The area of an isosceles right triangle with hypotenuse 6√2 is 18 -/
theorem area_of_specific_triangle :
  let t : IsoscelesRightTriangle := ⟨6 * Real.sqrt 2, by norm_num⟩
  area t = 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_specific_triangle_l316_31603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_products_l316_31678

def is_valid_product (n : ℕ) : Prop :=
  ∃ (primes : List ℕ),
    (primes.length ≥ 2) ∧
    (primes.all Nat.Prime) ∧
    (List.Pairwise (·≠·) primes) ∧
    (n = primes.prod) ∧
    (∀ p ∈ primes, n % (p - 1) = 0)

theorem valid_products : 
  ∀ n : ℕ, is_valid_product n ↔ n ∈ [6, 42, 1806] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_products_l316_31678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_food_price_l316_31671

/-- Represents the manufacturer's suggested retail price in dollars -/
def msrp : ℝ → Prop := sorry

/-- Represents the regular discount as a percentage (between 10% and 30%) -/
def regular_discount : ℝ → Prop := sorry

/-- Represents the additional sale discount as a percentage -/
def sale_discount : ℝ → Prop := sorry

/-- Represents the final sale price in dollars -/
def final_price : ℝ → Prop := sorry

theorem pet_food_price :
  ∀ m : ℝ,
  msrp m →
  regular_discount 30 →
  sale_discount 20 →
  final_price 22.40 →
  m = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pet_food_price_l316_31671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_C_not_quadratic_l316_31666

-- Define the expressions
def expr_A (x : ℝ) : ℝ := -2 * x^2
def expr_B (x : ℝ) : ℝ := 2 * (x - 1)^2 + 1
def expr_C (x : ℝ) : ℝ := (x - 3)^2 - x^2
def expr_D (a : ℝ) : ℝ := a * (8 - a)

-- Define what it means to be a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Theorem stating that expr_C is not quadratic while others are
theorem expr_C_not_quadratic :
  ¬(is_quadratic expr_C) ∧ 
  (is_quadratic expr_A) ∧ 
  (is_quadratic expr_B) ∧ 
  (is_quadratic (λ x ↦ expr_D x)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expr_C_not_quadratic_l316_31666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_broken_line_passes_all_vertices_l316_31683

/-- Represents a segment of a broken line on a grid -/
structure Segment where
  length : Nat
  direction : Bool

/-- Represents a broken line on a grid -/
def BrokenLine := List Segment

/-- The size of the square grid -/
def gridSize : Nat := 100

/-- Checks if a broken line is valid according to the problem conditions -/
def isValidBrokenLine (line : BrokenLine) : Prop :=
  line.all (fun seg => Odd seg.length) ∧
  (line.zip (line.tail)).all (fun (s1, s2) => s1.direction ≠ s2.direction) ∧
  line.foldl (fun acc seg => acc + seg.length) 0 = gridSize * 4

/-- Checks if a broken line passes through all vertices of the grid -/
def passesAllVertices (line : BrokenLine) : Prop :=
  line.foldl (fun acc seg => acc + seg.length + 1) 0 = (gridSize + 1) * (gridSize + 1)

/-- The main theorem stating that no valid broken line can pass through all vertices -/
theorem no_valid_broken_line_passes_all_vertices :
  ¬ ∃ (line : BrokenLine), isValidBrokenLine line ∧ passesAllVertices line := by
  sorry

#check no_valid_broken_line_passes_all_vertices

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_broken_line_passes_all_vertices_l316_31683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_debt_payment_difference_l316_31669

theorem debt_payment_difference (total_installments first_payment_count : ℕ) 
  (first_payment_amount average_payment : ℚ) 
  (h1 : total_installments = 65)
  (h2 : first_payment_count = 20)
  (h3 : first_payment_amount = 410)
  (h4 : average_payment = 455)
  : (total_installments * average_payment - 
     first_payment_count * first_payment_amount) / 
     (total_installments - first_payment_count) - 
     first_payment_amount = 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_debt_payment_difference_l316_31669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_l316_31676

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Definition of an ellipse with foci F₁ and F₂ -/
def is_ellipse (F₁ F₂ : ℝ × ℝ) (P : ℝ × ℝ) (c : ℝ) : Prop :=
  distance F₁.1 F₁.2 P.1 P.2 + distance F₂.1 F₂.2 P.1 P.2 = c

theorem ellipse_trajectory (m : ℝ) (h : m > 0) :
  let F₁ : ℝ × ℝ := (0, -3)
  let F₂ : ℝ × ℝ := (0, 3)
  let c : ℝ := m + 16 / m
  ∀ P : ℝ × ℝ, is_ellipse F₁ F₂ P c ↔ 
    distance F₁.1 F₁.2 P.1 P.2 + distance F₂.1 F₂.2 P.1 P.2 = m + 16 / m :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_trajectory_l316_31676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_roots_l316_31692

theorem max_sum_of_roots (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 10) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 10 → Real.sqrt (x + 2) + Real.sqrt (y + 3) ≤ Real.sqrt 30) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 10 ∧ Real.sqrt (x + 2) + Real.sqrt (y + 3) = Real.sqrt 30) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_roots_l316_31692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_to_fly_B_to_C_l316_31629

/-- Represents the cost structure for air travel -/
structure AirplaneCost where
  bookingFee : ℝ
  perKmCost : ℝ

/-- Calculates the distance between two points using the Pythagorean theorem -/
noncomputable def calculateDistance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

/-- Calculates the cost of air travel given a distance and cost structure -/
def calculateAirplaneCost (distance : ℝ) (cost : AirplaneCost) : ℝ :=
  cost.bookingFee + cost.perKmCost * distance

/-- Main theorem stating the cost to fly from B to C -/
theorem cost_to_fly_B_to_C : 
  let distAC : ℝ := 4000
  let distAB : ℝ := 4500
  let airplaneCost : AirplaneCost := { bookingFee := 120, perKmCost := 0.12 }
  let distBC := calculateDistance distAB distAC
  ∃ ε > 0, |calculateAirplaneCost distBC airplaneCost - 367.39| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_to_fly_B_to_C_l316_31629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_theorem_l316_31645

/-- Represents a book with a given number of chapters and pages per chapter -/
structure Book where
  num_chapters : Nat
  pages_per_chapter : Fin num_chapters → Nat
  total_pages : Nat

/-- Theorem about the number of pages in the last chapter and average page length -/
theorem book_pages_theorem (b : Book) 
  (h1 : b.num_chapters = 5)
  (h2 : b.pages_per_chapter ⟨0, by simp [h1]⟩ = 60)
  (h3 : b.pages_per_chapter ⟨1, by simp [h1]⟩ = 75)
  (h4 : b.pages_per_chapter ⟨2, by simp [h1]⟩ = 56)
  (h5 : b.pages_per_chapter ⟨3, by simp [h1]⟩ = 42)
  (h6 : b.total_pages = 325) :
  (b.pages_per_chapter ⟨4, by simp [h1]⟩ = 92) ∧ 
  (b.total_pages / b.num_chapters = 65) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_theorem_l316_31645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_theorem_reflected_ray_theorem_l316_31600

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - 2*y + 3 = 0
def l₂ (x y : ℝ) : Prop := 2*x + 3*y - 8 = 0

-- Define the intersection point of l₁ and l₂
def intersection_point : ℝ × ℝ := (1, 2)

-- Define the distance function from a point to the origin
noncomputable def distance_to_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

-- Define points M and N
def M : ℝ × ℝ := (2, 5)
def N : ℝ × ℝ := (-2, 4)

-- Theorem for part (1)
theorem intersection_line_theorem :
  ∃ (a b c : ℝ), 
    (∀ (x y : ℝ), a*x + b*y + c = 0 → 
      (x = (intersection_point.1) ∧ y = (intersection_point.2)) ∧
      distance_to_origin x y = 1) ∧
    ((a = 1 ∧ b = 0 ∧ c = -1) ∨ (a = 3 ∧ b = -4 ∧ c = 5)) :=
by sorry

-- Theorem for part (2)
theorem reflected_ray_theorem :
  ∀ (x y : ℝ), x + 2*y - 6 = 0 ↔ 
    (∃ (t : ℝ), x = M.1 + t*(N.1 - M.1) ∧ y = M.2 + t*(N.2 - M.2)) ∧
    (∃ (p q : ℝ), l₁ p q ∧ 
      (p - M.1)*(N.1 - p) + (q - M.2)*(N.2 - q) = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_theorem_reflected_ray_theorem_l316_31600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_biking_distance_l316_31620

/-- Calculates the total distance biked by two friends in a week given their biking schedules. -/
theorem total_biking_distance (onur_speed onur_duration hanil_speed extra_distance : ℝ) 
  (onur_days hanil_days : ℕ) : 
  onur_speed > 0 → 
  onur_duration > 0 → 
  hanil_speed > 0 → 
  extra_distance > 0 → 
  onur_days = 5 → 
  hanil_days = 3 → 
  (onur_speed * onur_duration * (onur_days : ℝ) + 
   (onur_speed * onur_duration + extra_distance) * (hanil_days : ℝ)) = 1800 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_biking_distance_l316_31620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_weight_l316_31613

theorem cake_weight (total_parts : ℕ) (nathalie_portion : ℚ) (pierre_portion : ℚ) (pierre_weight : ℚ) :
  total_parts = 8 ∧
  nathalie_portion = 1 / 8 ∧
  pierre_portion = 2 * nathalie_portion ∧
  pierre_weight = 100 →
  total_parts * (pierre_weight / pierre_portion) = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cake_weight_l316_31613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_in_second_quadrant_l316_31679

def z (m : ℝ) : ℂ := Complex.mk (m^2 - m - 6) (m^2 - 2*m - 3)

theorem z_purely_imaginary (m : ℝ) : z m = Complex.I * (z m).im ↔ m = -2 := by sorry

theorem z_in_second_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im > 0 ↔ -2 < m ∧ m < -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_purely_imaginary_z_in_second_quadrant_l316_31679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l316_31604

open Real

-- Define the equation
def equation (t : ℝ) : Prop :=
  (tan t) / (cos (5 * t))^2 - (tan (5 * t)) / (cos t)^2 = 0

-- Define the solution set
def solution_set (t : ℝ) : Prop :=
  (∃ k : ℤ, t = (Real.pi / 12) * (2 * ↑k + 1)) ∨ (∃ n : ℤ, t = Real.pi * ↑n)

-- Theorem statement
theorem equation_solution :
  ∀ t : ℝ, (cos t ≠ 0 ∧ cos (5 * t) ≠ 0) →
  (equation t ↔ solution_set t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l316_31604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_point_cube_root_cube_root_relationship_cube_root_comparison_l316_31624

-- 1. Relationship between decimal point movement and cube root
theorem decimal_point_cube_root (a : ℝ) :
  ∃ (a' : ℝ), a' = 1000 * a ∧ (a' ^ (1/3 : ℝ)) = 10 * (a ^ (1/3 : ℝ)) := by sorry

-- 2. Relationship between x and y given their cube roots
theorem cube_root_relationship (x y : ℝ) (hx : x ^ (1/3 : ℝ) = 1.587) (hy : y ^ (1/3 : ℝ) = -0.1587) :
  y = -x / 1000 := by sorry

-- 3. Relationships between a number and its cube root for different ranges
theorem cube_root_comparison (a : ℝ) :
  ((-1 < a ∧ a < 0) ∨ a > 1 → a ^ (1/3 : ℝ) < a) ∧
  (a = -1 ∨ a = 1 → a ^ (1/3 : ℝ) = a) ∧
  (a < -1 ∨ (0 < a ∧ a < 1) → a ^ (1/3 : ℝ) > a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_point_cube_root_cube_root_relationship_cube_root_comparison_l316_31624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l316_31640

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if three distances form an arithmetic sequence -/
def isArithmeticSequence (d1 d2 d3 : ℝ) : Prop :=
  d1 + d3 = 2 * d2

theorem ellipse_equation (e : Ellipse) (p f1 f2 : Point) :
  pointOnEllipse p e →
  p.x = 2 ∧ p.y = Real.sqrt 3 →
  f1.y = 0 ∧ f2.y = 0 →
  isArithmeticSequence (distance p f1) (distance f1 f2) (distance p f2) →
  e.a^2 = 8 ∧ e.b^2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l316_31640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_trig_bound_l316_31635

-- Define a triangle with sides forming a geometric sequence
structure GeometricTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_geometric : ∃ (q : ℝ), b = a * q ∧ c = b * q

-- Define the expression we want to bound
noncomputable def trigExpression (t : GeometricTriangle) (A B : ℝ) : ℝ :=
  Real.sin A * (1 / Real.tan A + 1 / Real.tan B)

-- State the theorem
theorem geometric_triangle_trig_bound (t : GeometricTriangle) (A B C : ℝ) :
  A + B + C = π →
  Real.sin A * t.a = Real.sin B * t.b →
  Real.sin B * t.b = Real.sin C * t.c →
  (Real.sqrt 5 - 1) / 2 < trigExpression t A B ∧ trigExpression t A B < (Real.sqrt 5 + 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_trig_bound_l316_31635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l316_31615

/-- An equilateral triangle with perimeter 18 cm has side length 6 cm. -/
theorem equilateral_triangle_side_length :
  let perimeter : ℝ := 18
  let side_length : ℝ := perimeter / 3
  side_length = 6 := by
  -- Define perimeter and side_length
  let perimeter : ℝ := 18
  let side_length : ℝ := perimeter / 3
  
  -- Prove that side_length = 6
  calc
    side_length = perimeter / 3 := rfl
    _ = 18 / 3 := rfl
    _ = 6 := by norm_num

  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_side_length_l316_31615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_painting_cost_is_2020_l316_31632

/-- Represents the cost of painting a house given contributions from three people in different currencies. -/
noncomputable def house_painting_cost (judson_usd kenny_euro_percent camilo_gbp_extra : ℝ) 
  (euro_to_usd gbp_to_usd : ℝ) : ℝ :=
  let kenny_usd := judson_usd * (1 + kenny_euro_percent / 100) * euro_to_usd
  let camilo_usd := (kenny_usd / gbp_to_usd + camilo_gbp_extra) * gbp_to_usd
  judson_usd + kenny_usd + camilo_usd

/-- Theorem stating that the total cost of painting the house is $2020 USD. -/
theorem house_painting_cost_is_2020 : 
  house_painting_cost 500 20 200 1.1 1.3 = 2020 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_painting_cost_is_2020_l316_31632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_stationary_points_relationship_l316_31674

-- Define the concept of a "new stationary point"
def new_stationary_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = (deriv f) x

-- Define the functions
def g (x : ℝ) : ℝ := x
noncomputable def h (x : ℝ) : ℝ := Real.log (x + 1)
def φ (x : ℝ) : ℝ := x^3 - 1

-- Assume the existence of new stationary points
axiom exists_new_stationary_point_g : ∃ x, new_stationary_point g x
axiom exists_new_stationary_point_h : ∃ x, new_stationary_point h x
axiom exists_new_stationary_point_φ : ∃ x, new_stationary_point φ x

-- Define α, β, and γ as the new stationary points of g, h, and φ respectively
noncomputable def α : ℝ := Classical.choose exists_new_stationary_point_g
noncomputable def β : ℝ := Classical.choose exists_new_stationary_point_h
noncomputable def γ : ℝ := Classical.choose exists_new_stationary_point_φ

-- The theorem to prove
theorem new_stationary_points_relationship : γ > α ∧ α > β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_stationary_points_relationship_l316_31674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_properties_l316_31621

-- Define the geometric mean operation for positive real numbers
noncomputable def geometric_mean (x y : ℝ) : ℝ := Real.sqrt (x * y)

-- State the theorem
theorem geometric_mean_properties :
  (∀ (a b : ℝ), a > 0 → b > 0 → geometric_mean a b = geometric_mean b a) ∧
  (¬ ∃ (e : ℝ), e > 0 ∧ ∀ (x : ℝ), x > 0 → geometric_mean x e = x) ∧
  (¬ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 
    geometric_mean (geometric_mean a b) c = geometric_mean a (geometric_mean b c)) ∧
  (¬ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 
    geometric_mean a (b * c) = geometric_mean a b * geometric_mean a c) ∧
  (¬ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 
    a * geometric_mean b c = geometric_mean (a * b) (a * c)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_mean_properties_l316_31621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divisibility_l316_31653

theorem square_divisibility (a b : ℕ+) (h : (a.val * b.val + 1) ∣ (a.val ^ 2 + b.val ^ 2)) :
  ∃ k : ℕ+, (a.val ^ 2 + b.val ^ 2) / (a.val * b.val + 1) = k.val ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_divisibility_l316_31653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l316_31658

theorem repeating_decimal_sum (c d : ℕ) : 
  (7 : ℚ) / 19 = (0.1 * c + 0.01 * d : ℚ) / (1 - 0.01) → c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l316_31658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_radius_relation_l316_31627

/-- Given four circles arranged as described, the radius of the large circle
    is related to the radius of the small circles by R = r(1 + √3). -/
theorem circle_arrangement_radius_relation (r R : ℝ) : R > 0 → r > 0 → 
  (∃ (C₁ C₂ C₃ C₄ O : ℝ × ℝ),
    -- Three small circles mutually externally tangent
    ‖C₁ - C₂‖ = 2*r ∧ ‖C₂ - C₃‖ = 2*r ∧ ‖C₃ - C₁‖ = 2*r ∧
    -- Small circles internally tangent to large circle
    ‖C₁ - O‖ = R - r ∧ ‖C₂ - O‖ = R - r ∧ ‖C₃ - O‖ = R - r ∧ ‖C₄ - O‖ = R - r ∧
    -- Fourth small circle externally tangent to two others
    ‖C₄ - C₂‖ = 2*r ∧ ‖C₄ - C₃‖ = 2*r) →
  R = r * (1 + Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_arrangement_radius_relation_l316_31627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_constant_implies_n_12_l316_31661

/-- 
Given a natural number n, and considering the binomial expansion of (√x - 2/x)^n,
if the fifth term of this expansion is a constant (i.e., independent of x),
then n must equal 12.
-/
theorem fifth_term_constant_implies_n_12 (n : ℕ) : 
  (∃ c : ℝ, ∀ x : ℝ, x > 0 → 
    (Nat.choose n 4) * (Real.sqrt x)^(n-4) * (-2/x)^4 = c) → 
  n = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_constant_implies_n_12_l316_31661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_system_accelerations_l316_31664

/-- A pulley system with a body and a massive pulley -/
structure PulleySystem where
  m : ℝ  -- Mass of the body
  M : ℝ  -- Mass of the pulley
  g : ℝ  -- Gravitational acceleration

/-- The acceleration of the body in the pulley system -/
noncomputable def body_acceleration (ps : PulleySystem) : ℝ :=
  (2 * (2 * ps.m + ps.M) * ps.g) / (4 * ps.m + ps.M)

/-- The acceleration of the pulley in the pulley system -/
noncomputable def pulley_acceleration (ps : PulleySystem) : ℝ :=
  ((2 * ps.m + ps.M) * ps.g) / (4 * ps.m + ps.M)

/-- Theorem stating the accelerations in the pulley system -/
theorem pulley_system_accelerations (ps : PulleySystem) 
  (hm : ps.m > 0) (hM : ps.M > 0) (hg : ps.g > 0) :
  body_acceleration ps = (2 * (2 * ps.m + ps.M) * ps.g) / (4 * ps.m + ps.M) ∧
  pulley_acceleration ps = ((2 * ps.m + ps.M) * ps.g) / (4 * ps.m + ps.M) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_system_accelerations_l316_31664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_three_valid_polynomials_l316_31675

/-- A structure representing a 2nd degree polynomial with integer coefficients -/
structure MyPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluate the polynomial at a given value -/
def MyPolynomial.eval (p : MyPolynomial) (x : ℤ) : ℤ :=
  p.a * x^2 + p.b * x + p.c

/-- Check if a polynomial satisfies the given conditions -/
def satisfiesConditions (p : MyPolynomial) : Prop :=
  p.eval p.a = p.b ∧ p.eval p.b = p.a ∧ p.a ≠ p.b

/-- The set of all polynomials satisfying the conditions -/
def validPolynomials : Set MyPolynomial :=
  {p | satisfiesConditions p}

/-- The theorem stating that only three specific polynomials satisfy the conditions -/
theorem only_three_valid_polynomials :
  validPolynomials = {
    MyPolynomial.mk 1 (-1) (-1),
    MyPolynomial.mk (-2) 5 23,
    MyPolynomial.mk (-3) 5 47
  } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_three_valid_polynomials_l316_31675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_number_solution_l316_31634

def alice_number (n : ℕ) : Prop :=
  n % 180 = 0 ∧ n % 45 = 0 ∧ 1000 ≤ n ∧ n ≤ 3000

theorem alice_number_solution :
  ∀ n : ℕ, alice_number n ↔ n ∈ ({1260, 1440, 1620, 1800, 1980, 2160, 2340, 2520, 2700, 2880} : Finset ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_number_solution_l316_31634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_in_cube_l316_31605

/-- Represents a path in a cubical box --/
structure BoxPath where
  length : ℝ
  spaceDigonalsUsed : ℕ
  cornersVisited : Finset (Fin 8)
  startsAndEndsAtSameCorner : Bool

/-- The maximum possible length of a path in a cubical box with given constraints --/
noncomputable def maxPathLength (boxSideLength : ℝ) : ℝ :=
  4 * Real.sqrt 3 + 8 * Real.sqrt 2 + 4

/-- Theorem stating the maximum path length in a cubical box --/
theorem max_path_length_in_cube (p : BoxPath) :
    p.length ≤ maxPathLength 2 ∧
    p.spaceDigonalsUsed ≤ 2 ∧
    p.cornersVisited.card = 8 ∧
    p.startsAndEndsAtSameCorner = true := by
  sorry

#check max_path_length_in_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_path_length_in_cube_l316_31605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l316_31688

noncomputable section

-- Define the curve C₁
def C₁ (α : Real) : Real × Real :=
  (2 + 4 * Real.cos α, 2 * Real.sqrt 3 + 4 * Real.sin α)

-- Define the polar equation
noncomputable def polar_equation (θ : Real) : Real :=
  8 * Real.cos (θ - Real.pi / 3)

-- Define the distance between intersection points
noncomputable def intersection_distance (β : Real) : Real :=
  Real.sqrt (12 * Real.sin β ^ 2 + 52)

theorem curve_properties :
  -- 1. Polar equation of C₁
  (∀ θ : Real, polar_equation θ = 
    Real.sqrt ((C₁ (Real.arccos ((polar_equation θ * Real.cos θ - 2) / 4))).1 ^ 2 + 
               (C₁ (Real.arccos ((polar_equation θ * Real.cos θ - 2) / 4))).2 ^ 2)) ∧
  -- 2. Minimum distance between intersection points
  (∀ β : Real, intersection_distance β ≥ 2 * Real.sqrt 13) ∧
  -- 3. Maximum distance between intersection points
  (∀ β : Real, intersection_distance β ≤ 8) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_properties_l316_31688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_k_range_l316_31655

-- Define the circle
def my_circle (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 2*y + 2 = 0

-- Define the condition that k is positive
def k_positive (k : ℝ) : Prop := k > 0

-- Define the condition that the circle doesn't intersect with coordinate axes
def no_intersection_with_axes (k : ℝ) : Prop :=
  ∀ x y : ℝ, my_circle k x y → (x ≠ 0 ∧ y ≠ 0)

-- The theorem to prove
theorem circle_k_range (k : ℝ) :
  k_positive k → no_intersection_with_axes k → 1 < k ∧ k < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_k_range_l316_31655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l316_31686

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (Real.pi * x / 4)

theorem f_properties :
  (∀ x, f x = f (-x)) ∧
  (∀ x, f (x - 2) = f (-x - 2)) ∧
  (∀ x, f x ≤ 3) ∧
  (∃ x, f x = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l316_31686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l316_31638

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 25

-- Define the line l
def line_l (x y : ℝ) : Prop := (5*x + 12*y + 20 = 0) ∨ (x + 4 = 0)

-- Define the point (-4,0)
def point_on_l : ℝ × ℝ := (-4, 0)

-- Define the intersection points A and B
def intersect_points (A B : ℝ × ℝ) : Prop :=
  my_circle A.1 A.2 ∧ my_circle B.1 B.2 ∧ line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define the distance between A and B
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem line_equation :
  ∀ A B : ℝ × ℝ,
  line_l point_on_l.1 point_on_l.2 →
  intersect_points A B →
  distance A B = 8 →
  line_l A.1 A.2 ∧ line_l B.1 B.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l316_31638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_period_l316_31609

/-- The minimum positive period of the cosine function y = 3cos(2/5x - π/6) -/
noncomputable def minimum_period (f : ℝ → ℝ) : ℝ :=
  5 * Real.pi

/-- The given cosine function -/
noncomputable def f (x : ℝ) : ℝ :=
  3 * Real.cos ((2 / 5 : ℝ) * x - Real.pi / 6)

theorem cosine_function_period :
  minimum_period f = 5 * Real.pi := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_period_l316_31609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l316_31610

/-- The focus of a parabola with equation y = 4ax² (a ≠ 0) has coordinates (0, 1/(16a)) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  ∃ (f : ℝ × ℝ), f = (0, 1 / (16 * a)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l316_31610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_l316_31698

/-- Given a parabola y² = 2px and a point P(2, 2√2) on it, 
    the distance from P to the directrix is 3. -/
theorem distance_to_directrix (p : ℝ) : 
  (2 * Real.sqrt 2)^2 = 2 * p * 2 →
  let x_directrix := -p / 2
  abs (2 - x_directrix) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_l316_31698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_double_mean_value_range_l316_31649

/-- A function is a double mean value function on an interval [a, b] if there exist
    x₁, x₂ ∈ (a, b) such that f''(x₁) = f''(x₂) = (f(b) - f(a)) / (b - a) -/
def is_double_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
    (deriv^[2] f) x₁ = (f b - f a) / (b - a) ∧
    (deriv^[2] f) x₂ = (f b - f a) / (b - a)

/-- The main theorem stating the range of m for which f(x) = x³ - (6/5)x² 
    is a double mean value function on [0, m] -/
theorem double_mean_value_range (m : ℝ) :
  is_double_mean_value_function (fun x => x^3 - 6/5*x^2) 0 m ↔ 3/5 < m ∧ m < 6/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_double_mean_value_range_l316_31649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percent_l316_31647

theorem shopkeeper_loss_percent (initial_value : ℝ) (profit_rate : ℝ) (theft_rate : ℝ) : 
  profit_rate = 0.1 →
  theft_rate = 0.4 →
  initial_value > 0 →
  let expected_sale_value := initial_value * (1 + profit_rate)
  let remaining_value := initial_value * (1 - theft_rate)
  let actual_sale_value := remaining_value * (1 + profit_rate)
  let loss_value := expected_sale_value - actual_sale_value
  let loss_percent := loss_value / expected_sale_value
  loss_percent = 0.4 := by
  intros h_profit h_theft h_initial
  -- The proof steps would go here
  sorry

#check shopkeeper_loss_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_loss_percent_l316_31647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l316_31612

/-- A function f is even if f(-x) = f(x) for all x in the domain of f -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x * e^x) / (e^(ax) - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

/-- If f(x) = (x * e^x) / (e^(ax) - 1) is an even function, then a = 2 -/
theorem f_even_implies_a_eq_two (a : ℝ) : IsEven (f a) → a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l316_31612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_women_reseating_l316_31633

/-- Number of ways n-1 women can be reseated in their original or adjacent seats -/
def S : ℕ → ℕ 
  | 0 => 1
  | 1 => 1
  | (n + 2) => S (n + 1) + S n

/-- Number of ways n women can be reseated with one initially standing -/
def T : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 2
  | (n + 3) => 2 * (S (n + 2) + T (n + 2))

/-- The problem statement -/
theorem eleven_women_reseating : T 11 = 610 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eleven_women_reseating_l316_31633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_8000_l316_31622

theorem cube_root_8000 : ∃ (c d : ℕ), (c : ℝ) * ((d : ℝ) ^ (1/3 : ℝ)) = (8000 : ℝ) ^ (1/3 : ℝ) ∧ 
  (∀ (c' d' : ℕ), (c' : ℝ) * ((d' : ℝ) ^ (1/3 : ℝ)) = (8000 : ℝ) ^ (1/3 : ℝ) → d ≤ d') ∧ 
  c = 20 ∧ d = 1 ∧ c + d = 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_8000_l316_31622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l316_31682

/-- Represents an ellipse with semi-major axis a, semi-minor axis b, and focal distance c. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_c_sq : c^2 = a^2 - b^2

/-- Represents a point in 2D space. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of an ellipse. -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- States that the perpendicular bisector of F₁P passes through F₂. -/
def perp_bisector_passes_through_F2 (e : Ellipse) (p : Point) : Prop :=
  p.x = e.a^2 / e.c ∧ 
  ∃ m : ℝ, p.y = m ∧ 
    (m / (e.a^2 / e.c + e.c)) * ((m/2) / ((e.a^2 - e.c^2)/(2*e.c) - e.c)) = -1

/-- The main theorem stating the range of eccentricity. -/
theorem eccentricity_range (e : Ellipse) (p : Point) 
    (h : perp_bisector_passes_through_F2 e p) : 
    Real.sqrt 3 / 3 ≤ eccentricity e ∧ eccentricity e < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l316_31682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l316_31607

-- Define the domain for both functions
noncomputable def Domain : Set ℝ := {x : ℝ | x > 0}

-- Define the two functions
noncomputable def f : Domain → ℝ := fun x ↦ (Real.sqrt x.val)^2 / x.val
noncomputable def g : Domain → ℝ := fun x ↦ x.val / (Real.sqrt x.val)^2

-- Theorem statement
theorem f_equals_g : ∀ x : Domain, f x = g x := by
  intro x
  simp [f, g]
  -- The actual proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l316_31607
