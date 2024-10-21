import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l804_80419

/-- The force acting on a sail --/
noncomputable def F (B ρ S v₀ v : ℝ) : ℝ := (B * S * ρ * (v₀ - v)^2) / 2

/-- The instantaneous wind power --/
noncomputable def N (B ρ S v₀ v : ℝ) : ℝ := F B ρ S v₀ v * v

/-- The wind speed --/
def v₀ : ℝ := 6.3

/-- The sail area --/
def S : ℝ := 7

/-- Theorem: The speed that maximizes instantaneous wind power is one-third of the wind speed --/
theorem max_power_speed (B ρ : ℝ) (hB : B > 0) (hρ : ρ > 0) :
  ∃ v : ℝ, v > 0 ∧ v = v₀ / 3 ∧ ∀ u : ℝ, u ≠ v → N B ρ S v₀ v ≥ N B ρ S v₀ u := by
  sorry

#check max_power_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_power_speed_l804_80419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_representation_l804_80403

/-- Given a triangle ABC with points E on AC and F on AB, prove that the intersection
    point P of BE and CF can be expressed as a specific linear combination of A, B, and C. -/
theorem intersection_point_representation (A B C E F P : ℝ × ℝ × ℝ) : 
  (∃ (t : ℝ), E = (1 - t) • A + t • C ∧ t = 2/5) →
  (∃ (s : ℝ), F = (1 - s) • A + s • B ∧ s = 3/5) →
  (∃ (α β : ℝ), P = α • B + (1 - α) • E ∧ P = β • C + (1 - β) • F) →
  P = 15/22 • A + 3/22 • B + 4/22 • C :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_representation_l804_80403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_integers_between_fractions_l804_80480

theorem average_of_integers_between_fractions : 
  let N := Finset.filter (fun n => 15 < n ∧ n < 25) (Finset.range 25)
  (N.sum id) / N.card = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_integers_between_fractions_l804_80480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segment_length_l804_80457

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a line segment with a given length -/
structure LineSegment where
  length : ℝ

/-- Represents a point on a line segment -/
structure PointOnSegment where
  segment : LineSegment
  position : ℝ  -- Relative position on the segment (0 ≤ position ≤ 1)

/-- The centroid of a triangle divides each median in a 2:1 ratio -/
noncomputable def centroid_ratio : ℝ := 2 / 3

/-- Given a triangle and a line parallel to one of its sides passing through the centroid,
    calculates the length of the parallel line segment -/
noncomputable def parallel_through_centroid (t : Triangle) (base : LineSegment) : ℝ :=
  centroid_ratio * base.length

theorem parallel_segment_length (t : Triangle) (qr : LineSegment) (f : PointOnSegment) (g : PointOnSegment) :
  t.a = 20 →
  t.b = 21 →
  t.c = 19 →
  qr.length = 19 →
  -- FG is parallel to QR and passes through the centroid
  parallel_through_centroid t qr = 38 / 3 := by
  sorry

#eval (38 : Nat) + (3 : Nat)  -- Should output 41

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_segment_length_l804_80457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_element_of_S_l804_80421

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the set of positive real numbers
def PositiveReals : Set ℝ := {x : ℝ | x > 0}

-- State the properties of f
axiom f_double (x : ℝ) : f (2 * x) = 2 * f x

axiom f_interval (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) : f x = 1 - |x - 3|

-- Define the set S
def S : Set ℝ := {x : ℝ | x ∈ PositiveReals ∧ f x = f 36}

-- State the theorem
theorem smallest_element_of_S : 
  ∃ (y : ℝ), y ∈ S ∧ ∀ (x : ℝ), x ∈ S → y ≤ x ∧ y = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_element_of_S_l804_80421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_at_13th_and_120th_positions_l804_80406

def natural_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => natural_sequence n + 1

def digit_at_position (pos : ℕ) : ℕ :=
  let num_string := (natural_sequence pos).repr
  if h : pos - 1 < num_string.length then
    (num_string.toList.get ⟨pos - 1, h⟩).toNat - '0'.toNat
  else
    0 -- Default value if position is out of range

theorem digits_at_13th_and_120th_positions :
  (digit_at_position 13 = 2) ∧ (digit_at_position 120 = 6) := by
  sorry

#eval digit_at_position 13
#eval digit_at_position 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_at_13th_and_120th_positions_l804_80406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_implies_a_range_l804_80484

/-- The function for which we're finding the minimum -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 1) / Real.log a

/-- The condition that f has a minimum value -/
def has_minimum (a : ℝ) : Prop := ∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m

/-- The theorem stating that if f has a minimum, then 1 < a < 2 -/
theorem minimum_implies_a_range :
  ∀ (a : ℝ), has_minimum a → 1 < a ∧ a < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_implies_a_range_l804_80484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l804_80407

/-- The distance between the foci of a hyperbola -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the foci of the hyperbola x²/25 - y²/4 = 1 is 2√29 -/
theorem hyperbola_foci_distance :
  distance_between_foci 5 2 = 2 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l804_80407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homogeneous_equation_solution_l804_80436

/-- The homogeneous differential equation -/
def homogeneous_equation (x y : ℝ) : Prop :=
  ∃ (dx dy : ℝ), (y^4 - 2*x^3*y) * dx + (x^4 - 2*x*y^3) * dy = 0

/-- The general solution of the homogeneous differential equation -/
def general_solution (x y C : ℝ) : Prop :=
  x^3 + y^3 = C*x*y

/-- Theorem stating that the general solution solves the homogeneous equation -/
theorem homogeneous_equation_solution :
  ∀ (x y C : ℝ), general_solution x y C → homogeneous_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_homogeneous_equation_solution_l804_80436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_three_quadratics_l804_80494

-- Auxiliary definition
noncomputable def number_of_real_roots (f : ℝ → ℝ) : ℕ := sorry

theorem max_real_roots_three_quadratics (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (r : ℕ), r ≤ 4 ∧
  ∀ (s : ℕ), (∃ (x y z : ℕ), 
    x = number_of_real_roots (λ t : ℝ ↦ a*t^2 + b*t + c) ∧
    y = number_of_real_roots (λ t : ℝ ↦ b*t^2 + c*t + a) ∧
    z = number_of_real_roots (λ t : ℝ ↦ c*t^2 + a*t + b) ∧
    s = x + y + z) → s ≤ r :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_roots_three_quadratics_l804_80494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_calculation_l804_80487

/-- The distance Lucy was from the lightning flash, in miles -/
noncomputable def lightning_distance (time_delay : ℝ) (sound_speed : ℝ) (feet_per_mile : ℝ) : ℝ :=
  (time_delay * sound_speed) / feet_per_mile

theorem lightning_distance_calculation :
  let time_delay : ℝ := 15
  let sound_speed : ℝ := 1100
  let feet_per_mile : ℝ := 5280
  lightning_distance time_delay sound_speed feet_per_mile = 3.125 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval lightning_distance 15 1100 5280

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lightning_distance_calculation_l804_80487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_separate_from_circle_l804_80472

-- Define the circle
def my_circle (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = r^2}

-- Define a point inside the circle
structure PointInsideCircle (r : ℝ) where
  a : ℝ
  b : ℝ
  inside : a^2 + b^2 < r^2

-- Define the line
def my_line (r : ℝ) (P : PointInsideCircle r) : Set (ℝ × ℝ) := {p | P.a * p.1 + P.b * p.2 = r^2}

-- Theorem statement
theorem line_separate_from_circle (r : ℝ) (P : PointInsideCircle r) :
  ∀ x ∈ my_line r P, x ∉ my_circle r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_separate_from_circle_l804_80472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l804_80432

theorem cos_minus_sin_value (α : ℝ) (h1 : Real.sin α * Real.cos α = -1/6) (h2 : α ∈ Set.Ioo 0 Real.pi) :
  Real.cos α - Real.sin α = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l804_80432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_eq_one_third_l804_80428

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1/2 then x + 1/2
  else if x < 1 then 2*x - 1
  else x - 1

-- Define the sequence recursively
noncomputable def a : ℕ → ℝ
  | 0 => 7/3
  | n + 1 => f (a n)

-- Theorem statement
theorem a_2019_eq_one_third : a 2018 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2019_eq_one_third_l804_80428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l804_80482

theorem problem_solution : 
  (Real.sqrt 2)^2 + |Real.sqrt 2 - 2| - (Real.pi - 1)^0 = 3 - Real.sqrt 2 ∧
  (27 : Real)^(1/3) + Real.sqrt ((-2)^2) - Real.sqrt (3^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l804_80482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_of_H_l804_80442

/-- The mass percentage of an element in a compound -/
noncomputable def mass_percentage (mass_element : ℝ) (mass_compound : ℝ) : ℝ :=
  (mass_element / mass_compound) * 100

/-- The given mass percentage of H in the compound -/
def given_percentage : ℝ := 2.76

theorem mass_percentage_of_H (mass_H : ℝ) (mass_compound : ℝ) 
  (h : mass_percentage mass_H mass_compound = given_percentage) :
  mass_percentage mass_H mass_compound = 2.76 := by
  rw [h]
  exact rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_of_H_l804_80442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l804_80499

theorem equation_roots : 
  let f : ℝ → ℝ := λ x => 3 * Real.sqrt x + 3 * x^(-(1/2 : ℝ)) - 8
  let root1 := ((8 + Real.sqrt 28) / 6)^2
  let root2 := ((8 - Real.sqrt 28) / 6)^2
  f root1 = 0 ∧ f root2 = 0 ∧ ∀ x, f x = 0 → x = root1 ∨ x = root2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_roots_l804_80499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_equals_two_sqrt_three_over_three_l804_80412

open Real

noncomputable def trigExpression : ℝ :=
  (sin (38 * π / 180) * sin (38 * π / 180) +
   cos (38 * π / 180) * sin (52 * π / 180) -
   tan (15 * π / 180) ^ 2) /
  (3 * tan (15 * π / 180))

theorem trigExpression_equals_two_sqrt_three_over_three :
  trigExpression = 2 * sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_equals_two_sqrt_three_over_three_l804_80412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_graph_and_sum_l804_80441

-- Define a function f
def f : ℝ → ℝ := sorry

-- Define the inverse function of f
def f_inv : ℝ → ℝ := sorry

-- State the theorem
theorem point_on_inverse_graph_and_sum (h : f 2 = 2) :
  (f_inv 2 = 2) ∧ 
  (2, (f_inv 2) / 3) = (2, 2 / 3) ∧
  2 + (f_inv 2) / 3 = 8 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_graph_and_sum_l804_80441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l804_80464

def T : ℕ → ℕ
  | 0 => 3  -- Add this case to handle Nat.zero
  | 1 => 3
  | n + 2 => 3^(T (n + 1))

theorem t_100_mod_7 : T 100 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l804_80464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l804_80459

def A : Fin 3 → ℝ := ![2, -5, 3]
def B : Fin 3 → ℝ := ![4, -9, 6]
def C : Fin 3 → ℝ := ![1, -4, 1]
def D : Fin 3 → ℝ := ![3, -8, 4]

def vector_subtract (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun i => v i - w i

def cross_product (v w : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![v 1 * w 2 - v 2 * w 1, v 2 * w 0 - v 0 * w 2, v 0 * w 1 - v 1 * w 0]

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt (v 0 * v 0 + v 1 * v 1 + v 2 * v 2)

theorem parallelogram_area : 
  vector_subtract B A = vector_subtract D C ∧ 
  magnitude (cross_product (vector_subtract B A) (vector_subtract C A)) = Real.sqrt 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l804_80459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slate_rock_probability_l804_80431

/-- The probability of selecting two slate rocks without replacement from a collection of 44 rocks (14 slate, 20 pumice, 10 granite) is 182/1892. -/
theorem slate_rock_probability : 
  (14 : ℚ) / 44 * (13 : ℚ) / 43 = 182 / 1892 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slate_rock_probability_l804_80431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_150_151_l804_80493

def g (x : ℤ) : ℤ := x^2 - 2*x + 3020

theorem gcd_g_150_151 : Int.gcd (g 150) (g 151) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_150_151_l804_80493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_cos_product_l804_80463

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem min_positive_period_sin_cos_product :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = π := by
  sorry

#check min_positive_period_sin_cos_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_sin_cos_product_l804_80463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_GAC_measure_l804_80418

-- Define the point type
def Point := ℝ × ℝ

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the pentagon BDEFG
structure Pentagon :=
  (B D E F G : Point)

-- Define the properties of the configuration
def Configuration (tri : Triangle) (pent : Pentagon) : Prop :=
  -- Triangle ABC is right-angled and isosceles
  (tri.A ≠ tri.B) ∧ (tri.B ≠ tri.C) ∧ (tri.C ≠ tri.A) ∧
  (∃ (angle : Point → Point → Point → ℝ),
    angle tri.A tri.B tri.C = Real.pi / 2) ∧
  (∃ (distance : Point → Point → ℝ),
    distance tri.A tri.B = distance tri.B tri.C) ∧
  -- ABC shares vertex B with pentagon BDEFG
  (tri.B = pent.B) ∧
  -- BDEFG is a regular pentagon
  (∃ (is_regular_pentagon : Point → Point → Point → Point → Point → Prop),
    is_regular_pentagon pent.B pent.D pent.E pent.F pent.G)

-- The theorem to be proved
theorem angle_GAC_measure (tri : Triangle) (pent : Pentagon) 
  (h : Configuration tri pent) : 
  ∃ (angle : Point → Point → Point → ℝ),
    angle pent.G tri.A tri.C = Real.pi / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_GAC_measure_l804_80418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l804_80400

/-- The line on which point M lies -/
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 34 = 0

/-- The circle on which point N lies -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 2 * y - 8 = 0

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem min_distance_line_circle :
  ∃ (d : ℝ), d = 5 ∧
  ∀ (x1 y1 x2 y2 : ℝ),
    line x1 y1 → circle_eq x2 y2 →
    distance x1 y1 x2 y2 ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_circle_l804_80400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l804_80408

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + (2 * Real.cos x - 4 * Real.sin x)^2)

theorem sum_of_max_and_min_f :
  (⨆ x, f x) + (⨅ x, f x) = 22 / 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_and_min_f_l804_80408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_P₁P₂_l804_80473

open Real

-- Define the functions
noncomputable def f (x : ℝ) := 4 * tan x
noncomputable def g (x : ℝ) := 6 * sin x
noncomputable def h (x : ℝ) := cos x

-- Define the domain
def I : Set ℝ := Set.Ioo 0 (π/2)

-- Define the intersection point P
noncomputable def P : ℝ × ℝ := 
  let x := arccos (2/3)
  (x, f x)

-- Define P₁
noncomputable def P₁ : ℝ × ℝ := (P.1, 0)

-- Define P₂
noncomputable def P₂ : ℝ × ℝ := (P.1, h P.1)

theorem length_P₁P₂ : abs (P₂.2 - P₁.2) = 2/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_P₁P₂_l804_80473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_mean_estimate_differs_from_incorrect_l804_80471

/-- Represents a rectangle in a frequency distribution histogram -/
structure HistogramRectangle where
  height : ℝ
  width : ℝ
  baseMiddle : ℝ

/-- Calculates the correct contribution of a rectangle to the mean estimate -/
def correctContribution (rect : HistogramRectangle) : ℝ :=
  rect.height * rect.width * rect.baseMiddle

/-- Calculates the incorrect contribution of a rectangle to the mean estimate -/
def incorrectContribution (rect : HistogramRectangle) : ℝ :=
  rect.height * rect.baseMiddle

/-- A list of rectangles in a frequency distribution histogram -/
def histogram : List HistogramRectangle := sorry

/-- The correct method to estimate the mean from a frequency distribution histogram -/
noncomputable def correctMeanEstimate : ℝ :=
  (histogram.map correctContribution).sum / (histogram.map (fun r => r.height * r.width)).sum

/-- The incorrect method to estimate the mean from a frequency distribution histogram -/
noncomputable def incorrectMeanEstimate : ℝ :=
  (histogram.map incorrectContribution).sum / (histogram.map (fun r => r.width)).sum

/-- Theorem stating that the correct mean estimate differs from the incorrect one -/
theorem correct_mean_estimate_differs_from_incorrect :
  correctMeanEstimate ≠ incorrectMeanEstimate := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_mean_estimate_differs_from_incorrect_l804_80471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_point_to_plane_l804_80461

noncomputable def M₀ : ℝ × ℝ × ℝ := (3, -2, -9)
noncomputable def M₁ : ℝ × ℝ × ℝ := (1, 2, -3)
noncomputable def M₂ : ℝ × ℝ × ℝ := (1, 0, 1)
noncomputable def M₃ : ℝ × ℝ × ℝ := (-2, -1, 6)

noncomputable def plane_equation (x y z : ℝ) : ℝ := x + 2*y + z - 2

noncomputable def distance_to_plane (p : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := p
  |plane_equation x y z| / Real.sqrt 6

theorem distance_from_point_to_plane :
  distance_to_plane M₀ = 2 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_point_to_plane_l804_80461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_120_l804_80416

/-- Represents a parallelogram ABCD with a shaded region BEDC -/
structure Parallelogram where
  -- Total area of parallelogram ABCD
  total_area : ℝ
  -- Length of side BC
  bc_length : ℝ
  -- Length of segment ED
  ed_length : ℝ
  -- Height of parallelogram (BE)
  height : ℝ

/-- The area of the shaded region BEDC in the parallelogram -/
noncomputable def shaded_area (p : Parallelogram) : ℝ :=
  p.total_area - (p.bc_length - p.ed_length) * p.height / 2

/-- Theorem stating that the area of the shaded region BEDC is 120 -/
theorem shaded_area_is_120 (p : Parallelogram) 
  (h1 : p.total_area = 150)
  (h2 : p.bc_length = 15)
  (h3 : p.ed_length = 9)
  : shaded_area p = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_120_l804_80416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_bijective_l804_80495

def y : ℕ → ℕ
  | 0 => 0  -- Added case for 0
  | 1 => 1
  | n + 2 =>
    let k := (n + 1) / 2
    if (n + 2) % 2 = 0 then
      if k % 2 = 0 then 2 * y k else 2 * y k + 1
    else
      if k % 2 = 0 then 2 * y k + 1 else 2 * y k

theorem y_bijective : Function.Bijective y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_bijective_l804_80495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_plus_pi_3_l804_80444

theorem sin_2theta_plus_pi_3 (θ : Real) (h : Real.tan θ = 3) :
  Real.sin (2 * θ + Real.pi / 3) = (3 - 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_plus_pi_3_l804_80444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_points_tangent_to_line_or_circle_l804_80479

-- Define the necessary structures
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

-- Define necessary concepts
def Circle.boundary (C : Circle) : Set Point :=
  {P : Point | (P.x - C.center.x)^2 + (P.y - C.center.y)^2 = C.radius^2}

def Line.points (l : Line) : Set Point :=
  {P : Point | l.a * P.x + l.b * P.y + l.c = 0}

def Circle.tangent_to_line (C : Circle) (l : Line) : Prop :=
  ∃ P : Point, P ∈ C.boundary ∧ P ∈ l.points ∧ 
  ∀ Q : Point, Q ≠ P → Q ∈ C.boundary → Q ∉ l.points

def Circle.tangent_to_circle (C D : Circle) : Prop :=
  ∃ P : Point, P ∈ C.boundary ∧ P ∈ D.boundary ∧
  ∀ Q : Point, Q ≠ P → (Q ∈ C.boundary → Q ∉ D.boundary) ∧ (Q ∈ D.boundary → Q ∉ C.boundary)

-- Define the theorem
theorem circle_through_points_tangent_to_line_or_circle 
  (A B : Point) (l : Line) (S : Circle) : 
  (∃ C : Circle, (A ∈ C.boundary ∧ B ∈ C.boundary ∧ C.tangent_to_line l)) ∧
  (∃ D : Circle, (A ∈ D.boundary ∧ B ∈ D.boundary ∧ D.tangent_to_circle S)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_through_points_tangent_to_line_or_circle_l804_80479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_correct_l804_80429

/-- Represents a right triangle with sides 6, 8, and 10 inches -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- The length of the crease when folding the right angle vertex onto the opposite vertex -/
noncomputable def crease_length (t : RightTriangle) : ℝ := 20/3

/-- Theorem stating that the crease length is 20/3 inches -/
theorem crease_length_is_correct (t : RightTriangle) : 
  crease_length t = 20/3 := by
  -- The proof goes here
  sorry

#eval toString (20/3 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crease_length_is_correct_l804_80429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l804_80462

/-- The cosine function with angular frequency ω and phase φ -/
noncomputable def f (ω φ x : ℝ) : ℝ := Real.cos (ω * x + φ)

/-- The smallest positive period of a periodic function -/
noncomputable def smallestPositivePeriod (g : ℝ → ℝ) : ℝ := sorry

theorem min_omega_value (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < Real.pi) :
  let T := smallestPositivePeriod (f ω φ)
  (f ω φ T = Real.sqrt 3 / 2) →
  (f ω φ (Real.pi / 9) = 0) →
  (∀ ω' : ℝ, ω' > 0 ∧ 
    (∃ φ' : ℝ, 0 < φ' ∧ φ' < Real.pi ∧
      let T' := smallestPositivePeriod (f ω' φ')
      (f ω' φ' T' = Real.sqrt 3 / 2) ∧
      (f ω' φ' (Real.pi / 9) = 0)) →
    ω' ≥ ω) →
  ω = 3 := by
  sorry

#check min_omega_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l804_80462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l804_80490

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the inequality
def inequality (x : ℝ) : Prop :=
  x^2 ≤ 2 * (floor (cubeRoot x + 0.5) + floor (cubeRoot x))

-- Theorem statement
theorem solution_difference :
  ∃ (min max : ℝ), 
    (∀ x, inequality x → min ≤ x ∧ x ≤ max) ∧
    (inequality min ∧ inequality max) ∧
    (max - min = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l804_80490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_min_sum_l804_80466

/-- Represents a monic quadratic polynomial ax² + bx + c -/
structure MonicQuadratic where
  b : ℝ
  c : ℝ

/-- Evaluates a monic quadratic polynomial at x -/
noncomputable def MonicQuadratic.eval (p : MonicQuadratic) (x : ℝ) : ℝ :=
  x^2 + p.b * x + p.c

/-- Composes two monic quadratic polynomials -/
def MonicQuadratic.compose (p q : MonicQuadratic) : MonicQuadratic :=
  { b := q.b^2 + 2*q.c + p.b,
    c := q.c^2 + p.b*q.c + p.c }

/-- The minimum value of a monic quadratic polynomial -/
noncomputable def MonicQuadratic.minValue (p : MonicQuadratic) : ℝ :=
  p.eval (-p.b/2)

theorem monic_quadratic_min_sum (p q : MonicQuadratic) :
  (MonicQuadratic.compose p q).eval (-23) = 0 ∧
  (MonicQuadratic.compose p q).eval (-21) = 0 ∧
  (MonicQuadratic.compose p q).eval (-17) = 0 ∧
  (MonicQuadratic.compose p q).eval (-15) = 0 ∧
  (MonicQuadratic.compose q p).eval (-59) = 0 ∧
  (MonicQuadratic.compose q p).eval (-57) = 0 ∧
  (MonicQuadratic.compose q p).eval (-51) = 0 ∧
  (MonicQuadratic.compose q p).eval (-49) = 0 →
  p.minValue + q.minValue = -100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_min_sum_l804_80466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_radius_l804_80488

theorem greatest_integer_radius (r : ℕ) (A : ℝ) : 
  A < 140 * Real.pi → A = Real.pi * (r : ℝ)^2 → r ≤ 11 ∧ ∃ (s : ℕ), s = 11 ∧ A < Real.pi * (s : ℝ)^2 :=
by
  sorry

#check greatest_integer_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_radius_l804_80488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_min_side_l804_80467

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- angle A
  B : ℝ  -- angle B
  C : ℝ  -- angle C
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C

-- Define the conditions and theorems
theorem triangle_area_and_min_side (t : Triangle) :
  (t.c / Real.sin t.C = t.a / (Real.sqrt 3 * Real.cos t.A)) →
  (
    -- Part 1
    (4 * Real.sin t.C = t.c^2 * Real.sin t.B) →
    (1/2 * t.b * t.c * Real.sin t.A = Real.sqrt 3)
  ) ∧
  (
    -- Part 2
    (t.c^2 - t.a * t.c * Real.cos t.B = 4) →
    (∀ a' : ℝ, t.a ≤ a' → 2 * Real.sqrt 2 ≤ a')
  ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_min_side_l804_80467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PACB_l804_80423

/-- The line on which point P moves -/
def line_equation (x y : ℝ) : Prop := 3 * x + 4 * y + 8 = 0

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

/-- P is a point on the line -/
def P : ℝ × ℝ → Prop := λ p => line_equation p.1 p.2

/-- A and B are points on the circle -/
def A : ℝ × ℝ → Prop := λ a => circle_equation a.1 a.2
def B : ℝ × ℝ → Prop := λ b => circle_equation b.1 b.2

/-- C is the center of the circle -/
def C : ℝ × ℝ := (1, 1)

/-- PA and PB are tangent to the circle -/
def is_tangent (p a : ℝ × ℝ) : Prop := 
  P p ∧ A a ∧ ((p.1 - a.1) * (a.1 - C.1) + (p.2 - a.2) * (a.2 - C.2) = 0)

/-- The area of quadrilateral PACB -/
noncomputable def area_PACB (p a b : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - a.1)^2 + (p.2 - a.2)^2) +
  Real.sqrt ((p.1 - b.1)^2 + (p.2 - b.2)^2)

/-- The main theorem: minimum area of PACB is 2√2 -/
theorem min_area_PACB : 
  ∃ (p a b : ℝ × ℝ), 
    P p ∧ A a ∧ B b ∧ 
    is_tangent p a ∧ is_tangent p b ∧
    (∀ (p' a' b' : ℝ × ℝ), 
      P p' ∧ A a' ∧ B b' ∧ 
      is_tangent p' a' ∧ is_tangent p' b' →
      area_PACB p a b ≤ area_PACB p' a' b') ∧
    area_PACB p a b = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_PACB_l804_80423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_theorem_l804_80445

/-- Represents the speed of a boat in still water given downstream and upstream travel times and stream speed. -/
noncomputable def boat_speed_in_still_water (downstream_time : ℝ) (upstream_time : ℝ) (stream_speed : ℝ) : ℝ :=
  (downstream_time + upstream_time) / (downstream_time * upstream_time) * stream_speed

/-- Theorem stating that the boat's speed in still water is 16.8 kmph given the problem conditions. -/
theorem boat_speed_theorem (downstream_time upstream_time stream_speed : ℝ) 
    (h1 : downstream_time = 2)
    (h2 : upstream_time = 3.25)
    (h3 : stream_speed = 4) :
  boat_speed_in_still_water downstream_time upstream_time stream_speed = 16.8 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_theorem_l804_80445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncoverable_board_l804_80424

/-- Represents a game board -/
structure GameBoard where
  rows : ℕ
  cols : ℕ
  removed_squares : ℕ

/-- Checks if a game board can be covered by dominoes -/
def can_be_covered (board : GameBoard) : Prop :=
  (board.rows * board.cols - board.removed_squares) % 2 = 0

/-- Theorem stating which board cannot be covered -/
theorem uncoverable_board :
  let boards : List GameBoard := [
    ⟨4, 6, 0⟩,
    ⟨5, 5, 1⟩,
    ⟨5, 3, 0⟩,
    ⟨7, 4, 0⟩,
    ⟨4, 7, 2⟩
  ]
  ∃! board, board ∈ boards ∧ ¬can_be_covered board ∧ 
    board.rows = 5 ∧ board.cols = 3 ∧ board.removed_squares = 0 := by
  sorry

#check uncoverable_board

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncoverable_board_l804_80424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_babysitting_earnings_l804_80448

noncomputable def weekday_hours : ℝ := 4
noncomputable def weekend_hours : ℝ := 2.5
noncomputable def weekday_rate : ℝ := 12
noncomputable def weekend_rate : ℝ := 15
def weekdays : ℕ := 5
def weekend_days : ℕ := 2

noncomputable def makeup_fraction : ℝ := 3/8
noncomputable def skincare_fraction : ℝ := 2/5
noncomputable def cellphone_fraction : ℝ := 1/6

noncomputable def total_earnings : ℝ := weekday_hours * weekday_rate * (weekdays : ℝ) + weekend_hours * weekend_rate * (weekend_days : ℝ)

noncomputable def remaining_after_makeup : ℝ := total_earnings * (1 - makeup_fraction)
noncomputable def remaining_after_skincare : ℝ := remaining_after_makeup * (1 - skincare_fraction)
noncomputable def final_remaining : ℝ := remaining_after_skincare * (1 - cellphone_fraction)

theorem babysitting_earnings : final_remaining = 98.4375 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_babysitting_earnings_l804_80448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_four_divisors_l804_80420

theorem remainder_four_divisors : 
  ∀ n : ℕ, (∃ k : ℕ, 49 = n * k) ∧ (n > 4) ∧ (53 % n = 4) ↔ (n = 7 ∨ n = 49) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_four_divisors_l804_80420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l804_80498

/-- Given two points A and B in a 2D plane, calculate the magnitude of vector AB -/
theorem vector_magnitude (A B : ℝ × ℝ) : 
  A = (1, 3) → B = (4, -1) → ‖(B.1 - A.1, B.2 - A.2)‖ = 5 := by
  sorry

#check vector_magnitude

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l804_80498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_point_l804_80417

/-- The function f(x) = a * ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

/-- The slope of the tangent line at x = e -/
noncomputable def tangentSlope (a : ℝ) : ℝ := a / Real.exp 1

/-- The y-intercept of the tangent line -/
noncomputable def yIntercept (a : ℝ) : ℝ := a - tangentSlope a * Real.exp 1

/-- The tangent line equation -/
noncomputable def tangentLine (a : ℝ) (x : ℝ) : ℝ := tangentSlope a * x + yIntercept a

theorem tangent_line_passes_through_point (a : ℝ) :
  tangentLine a (-1) = -1 → a = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_point_l804_80417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_games_for_valid_tournament_l804_80449

/-- Represents a football team in the tournament. -/
structure Team where
  points : Nat

/-- Represents a football tournament. -/
structure Tournament where
  teams : Finset Team
  games : Nat

/-- The number of teams in the tournament. -/
def numTeams : Nat := 5

/-- Predicate to check if all teams have different non-zero point totals. -/
def validPointTotals (t : Tournament) : Prop :=
  t.teams.card = numTeams ∧
  (∀ team, team ∈ t.teams → team.points > 0) ∧
  (∀ team1 team2, team1 ∈ t.teams → team2 ∈ t.teams → team1 ≠ team2 → team1.points ≠ team2.points)

/-- Theorem stating the minimum number of games required. -/
theorem min_games_for_valid_tournament :
  ∀ t : Tournament, validPointTotals t → t.games ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_games_for_valid_tournament_l804_80449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l804_80405

-- Define structures and definitions
structure Quadrilateral where
  -- Define properties of a quadrilateral
  mk :: -- This is a placeholder, you might want to add actual fields here

def DiagonalsEqual (q : Quadrilateral) : Prop :=
  sorry

def IsRectangle (q : Quadrilateral) : Prop :=
  sorry

theorem three_true_propositions :
  let prop1 := ∀ k : ℝ, k > 0 → ∃ x : ℝ, x^2 + 2*x - k = 0
  let prop2 := ∀ a b c : ℝ, a ≤ b → a + c ≤ b + c
  let prop3 := ¬(∀ q : Quadrilateral, DiagonalsEqual q → IsRectangle q)
  let prop4 := ∀ x y : ℝ, x*y ≠ 0 → x ≠ 0 ∧ y ≠ 0
  (prop1 ∧ prop2 ∧ prop3 ∧ prop4) ∧
  (¬prop1 ∧ prop2 ∧ prop3 ∧ prop4 ∨
   prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4 ∨
   prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 ∨
   prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_propositions_l804_80405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_languages_have_basic_statements_l804_80458

structure ProgrammingLanguage where
  name : String
  hasInputStatement : Bool
  hasOutputStatement : Bool
  hasAssignmentStatement : Bool
  hasConditionalStatement : Bool
  hasLoopStatement : Bool

def hasAllBasicStatements (lang : ProgrammingLanguage) : Bool :=
  lang.hasInputStatement &&
  lang.hasOutputStatement &&
  lang.hasAssignmentStatement &&
  lang.hasConditionalStatement &&
  lang.hasLoopStatement

theorem all_languages_have_basic_statements (lang : ProgrammingLanguage) :
  hasAllBasicStatements lang = true := by
  sorry

#eval hasAllBasicStatements (ProgrammingLanguage.mk "Example Lang" true true true true true)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_languages_have_basic_statements_l804_80458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_divided_triangle_l804_80470

/-- Perimeter function for a piece of the divided triangle -/
noncomputable def P (k : ℕ) : ℝ :=
  1.25 + Real.sqrt (12^2 + k^2 * (10/8)^2) + Real.sqrt (12^2 + (k+1)^2 * (10/8)^2)

/-- Theorem stating the maximum perimeter of the triangle pieces -/
theorem max_perimeter_of_divided_triangle :
  ∃ (max_perimeter : ℝ),
    (∀ k, k ∈ Finset.range 8 → P k ≤ max_perimeter) ∧
    (∃ k, k ∈ Finset.range 8 ∧ P k = max_perimeter) ∧
    (abs (max_perimeter - 30.53) < 0.01) := by
  sorry

#check max_perimeter_of_divided_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_divided_triangle_l804_80470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_sin_product_l804_80485

theorem max_cos_sin_product :
  (∀ θ : ℝ, 5 * (Real.cos θ * Real.sin θ) ≤ 5/2) ∧
  (∃ θ : ℝ, 5 * (Real.cos θ * Real.sin θ) = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cos_sin_product_l804_80485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l804_80481

theorem trigonometric_identity (α : ℝ) 
  (h : Real.sin (2 * α) - 2 = 2 * Real.cos (2 * α)) : 
  Real.sin α ^ 2 + Real.sin (2 * α) = 1 ∨ Real.sin α ^ 2 + Real.sin (2 * α) = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l804_80481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_negative_six_l804_80409

-- Define the quadratic polynomials
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem sum_of_roots_is_negative_six 
  (a b c d : ℝ) 
  (h1 : f a b 1 = g c d 2) 
  (h2 : g c d 1 = f a b 2) :
  ∃ r1 r2 r3 r4 : ℝ, 
    (r1 + r2 = -a) ∧ 
    (r3 + r4 = -c) ∧ 
    (r1 + r2 + r3 + r4 = -6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_negative_six_l804_80409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l804_80434

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem problem_statement : (-1 : ℤ) ^ 53 + 3 ^ (2 ^ 3 + 5 ^ 2 - factorial 4) = 19682 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l804_80434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l804_80456

def vector_a (lambda : ℝ) : Fin 3 → ℝ := ![1, lambda, 2]
def vector_b : Fin 3 → ℝ := ![2, -1, 2]

def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

noncomputable def magnitude (v : Fin 3 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2 + (v 2)^2)

theorem lambda_value :
  ∃ lambda : ℝ, 
    (dot_product (vector_a lambda) vector_b) / (magnitude (vector_a lambda) * magnitude vector_b) = 8/9 ∧
    lambda = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l804_80456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_properties_l804_80402

/-- Represents a parabolic arch -/
structure ParabolicArch where
  height : ℝ
  span : ℝ

/-- Calculate the height of the arch at a given distance from the center -/
noncomputable def archHeight (arch : ParabolicArch) (x : ℝ) : ℝ :=
  -((4 * arch.height) / (arch.span ^ 2)) * x^2 + arch.height

theorem parabolic_arch_properties (arch : ParabolicArch) 
  (h1 : arch.height = 20)
  (h2 : arch.span = 50) :
  (archHeight arch 10 = 16.8) ∧ 
  (∃ x : ℝ, x = 10 ∧ archHeight arch x = |x|) ∧
  (∃ x : ℝ, x = -10 ∧ archHeight arch x = |x|) := by
  sorry

#check parabolic_arch_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolic_arch_properties_l804_80402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_chairs_removal_l804_80443

def chairs_to_remove (chairs_per_row total_chairs expected_attendees : ℕ) : ℕ :=
  let remaining_chairs := (((expected_attendees + chairs_per_row - 1) / chairs_per_row) * chairs_per_row)
  total_chairs - remaining_chairs

theorem conference_chairs_removal (chairs_per_row total_chairs expected_attendees : ℕ) 
  (h1 : chairs_per_row = 15)
  (h2 : total_chairs = 225)
  (h3 : expected_attendees = 160) :
  chairs_to_remove chairs_per_row total_chairs expected_attendees = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conference_chairs_removal_l804_80443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l804_80415

def a : ℕ → ℤ
  | 0 => 0  -- Added case for 0
  | 1 => 0
  | (n + 2) => a (n + 1) + 4 * Int.sqrt (a (n + 1) + 1) + 4

theorem a_closed_form (n : ℕ) : a n = 4 * n ^ 2 - 4 * n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_l804_80415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l804_80437

noncomputable def equation (x : ℝ) : Prop :=
  8.4682 * Real.tan (Real.pi * x^2) - Real.tan (Real.pi * x) + Real.tan (Real.pi * x) * Real.tan (Real.pi * x^2)^2 = 0

noncomputable def solution_set : Set ℝ :=
  {0} ∪ 
  {x | ∃ k : ℤ, k > 0 ∧ (∀ l : ℤ, l > 0 → k ≠ l * (2 * l + 1)) ∧ x = (1 + Real.sqrt (1 + 8 * ↑k)) / 4} ∪
  {x | ∃ k : ℤ, k > 0 ∧ (∀ t : ℤ, t > 0 → k ≠ t * (2 * t - 1)) ∧ x = (1 - Real.sqrt (1 + 8 * ↑k)) / 4}

theorem equation_solution : ∀ x : ℝ, equation x ↔ x ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l804_80437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l804_80435

/-- The greatest positive even integer less than or equal to a real number -/
noncomputable def greatest_even_le (y : ℝ) : ℤ :=
  2 * ⌊y / 2⌋

/-- The smallest odd integer greater than a real number -/
noncomputable def smallest_odd_gt (x : ℝ) : ℤ :=
  2 * ⌈x / 2⌉ + 1

theorem problem_solution (x y : ℝ) (hx : x = 3.25) (hy : y = 12.5) :
  (6.32 - (greatest_even_le y : ℝ)) * ((smallest_odd_gt x : ℝ) - x) = -9.94 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l804_80435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sine_function_param_range_l804_80414

/-- A function f(x) = x + a*sin(x) is increasing on the real line if and only if a is in the closed interval [-1, 1] -/
theorem increasing_sine_function_param_range (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (λ x ↦ x + a * Real.sin x) (1 + a * Real.cos x) x) →
  (∀ x : ℝ, (1 + a * Real.cos x) ≥ 0) ↔
  a ∈ Set.Icc (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sine_function_param_range_l804_80414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l804_80401

theorem expression_evaluation (b : ℝ) (hb : b ≠ 0) : 
  (1 / 9 : ℝ) * b^0 + (1 / (9 * b))^0 - 27^(-(1/3 : ℝ)) - (-27 : ℝ)^(-(3/4 : ℝ)) = 
  1 + 1/9 - 1/3 + 1/(3^(9/4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l804_80401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_combinations_count_l804_80476

/-- Represents the types of socks --/
inductive SockType
  | Striped
  | Solid
  | Checkered

/-- Represents a pair of socks --/
structure SockPair :=
  (first : SockType)
  (second : SockType)

/-- Defines a valid sock combination --/
def isValidCombination (pair : SockPair) : Prop :=
  pair.first = SockType.Striped ∧ (pair.second = SockType.Solid ∨ pair.second = SockType.Checkered)

/-- Counts the number of valid sock combinations --/
def countValidCombinations (stripedPairs solidPairs checkeredPairs : Nat) : Nat :=
  (stripedPairs * solidPairs) + (stripedPairs * checkeredPairs)

theorem sock_combinations_count :
  countValidCombinations 4 4 4 = 32 :=
by
  -- Unfold the definition of countValidCombinations
  unfold countValidCombinations
  -- Evaluate the arithmetic expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_combinations_count_l804_80476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_race_distance_l804_80460

/-- The distance of the second race in meters -/
noncomputable def D : ℝ := 800

/-- The speed ratio of A to B -/
noncomputable def speed_ratio_AB : ℝ := 10 / 9

/-- The speed ratio of A to C -/
noncomputable def speed_ratio_AC : ℝ := 80 / 63

/-- The speed ratio of B to C -/
noncomputable def speed_ratio_BC : ℝ := D / (D - 100)

theorem second_race_distance :
  (speed_ratio_AC / speed_ratio_AB = speed_ratio_BC) → D = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_race_distance_l804_80460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l804_80438

-- Define the set M
noncomputable def M : Set (ℝ → ℝ) :=
  {f | (∀ x y, x < y → f x ≤ f y ∨ f x ≥ f y) ∧
       (∃ a b, a < b ∧ (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (a/2) (b/2)) ∧
                       (∀ y ∈ Set.Icc (a/2) (b/2), ∃ x ∈ Set.Icc a b, f x = y))}

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x + 2/x

noncomputable def g (x : ℝ) : ℝ := -x^3

-- Theorem statement
theorem problem_solution :
  (f ∉ M) ∧
  (g ∈ M) ∧
  (∃ a b, a = -Real.sqrt 2 / 2 ∧ b = Real.sqrt 2 / 2 ∧
    (∀ x ∈ Set.Icc a b, g x ∈ Set.Icc (a/2) (b/2)) ∧
    (∀ y ∈ Set.Icc (a/2) (b/2), ∃ x ∈ Set.Icc a b, g x = y)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l804_80438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l804_80455

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def geometric_mean (x y : ℝ) : ℝ :=
  Real.sqrt (x * y)

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_sequence_properties (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 3 = 8 →
  a 4 = geometric_mean (a 2) (a 9) →
  (∀ n : ℕ, a n = 4 ∧ sum_of_arithmetic_sequence a n = 4 * n) ∨
  (∀ n : ℕ, a n = 3 * n - 2 ∧ sum_of_arithmetic_sequence a n = (3 * n^2 - n) / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l804_80455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_is_one_l804_80413

def seq_property (a : ℕ → ℚ) : Prop :=
  (∀ n, a n * a (n + 1) * a (n + 2) = -1/2) ∧
  (a 1 = -2) ∧
  (a 2 = 1/4)

def prod_seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  Finset.prod (Finset.range n) (λ i => a (i + 1))

theorem max_product_is_one (a : ℕ → ℚ) (h : seq_property a) :
  ∃ n : ℕ, prod_seq a n = 1 ∧ ∀ m : ℕ, prod_seq a m ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_is_one_l804_80413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_problem_l804_80486

/-- Proportionality constant for α and β -/
noncomputable def k : ℝ := 5 * 10

/-- Proportionality constant for γ and β -/
noncomputable def c : ℝ := 4 / 2

/-- α as a function of β -/
noncomputable def α (β : ℝ) : ℝ := k / β

/-- γ as a function of β -/
noncomputable def γ (β : ℝ) : ℝ := c * β

theorem proportionality_problem (β : ℝ) (hβ : β = 40) :
  α β = 5/4 ∧ γ β = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_problem_l804_80486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_closer_vertex_l804_80433

-- Define a convex polygon
def ConvexPolygon (vertices : Set (ℝ × ℝ)) : Prop := sorry

-- Define a point being inside a polygon
def InsidePolygon (p : ℝ × ℝ) (polygon : Set (ℝ × ℝ)) : Prop := sorry

-- Define Euclidean distance
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem exists_closer_vertex 
  (vertices : Set (ℝ × ℝ)) 
  (P Q : ℝ × ℝ) 
  (h_convex : ConvexPolygon vertices)
  (h_P_inside : InsidePolygon P vertices)
  (h_Q_inside : InsidePolygon Q vertices) :
  ∃ V ∈ vertices, distance V Q < distance V P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_closer_vertex_l804_80433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_range_l804_80426

/-- Curve C defined by x^2 + y^2/4 = 1 -/
def CurveC (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- Point P at (0, 1) -/
def P : ℝ × ℝ := (0, 1)

/-- Line ℓ passing through P(0, 1) with inclination angle α -/
noncomputable def Line_ℓ (α t : ℝ) : ℝ × ℝ := (t * Real.cos α, 1 + t * Real.sin α)

/-- Points A and B are intersections of Line_ℓ and CurveC -/
def Intersections (α : ℝ) : Prop :=
  ∃ t₁ t₂, CurveC (Line_ℓ α t₁).1 (Line_ℓ α t₁).2 ∧ CurveC (Line_ℓ α t₂).1 (Line_ℓ α t₂).2

/-- Distance between two points -/
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

/-- Product of distances PA and PB -/
noncomputable def distanceProduct (α t₁ t₂ : ℝ) : ℝ :=
  distance P (Line_ℓ α t₁) * distance P (Line_ℓ α t₂)

theorem distance_product_range :
  ∀ α, Intersections α →
  ∃ t₁ t₂, 3/4 ≤ distanceProduct α t₁ t₂ ∧ distanceProduct α t₁ t₂ ≤ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_product_range_l804_80426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_speed_to_arrive_before_bob_l804_80440

/-- The minimum speed Alice needs to arrive before Bob -/
noncomputable def aliceMinSpeed (distance : ℝ) (bobSpeed : ℝ) (aliceDelay : ℝ) : ℝ :=
  distance / (distance / bobSpeed - aliceDelay)

/-- Theorem stating the conditions and the result to be proved -/
theorem alice_speed_to_arrive_before_bob :
  let distance := (220 : ℝ)
  let bobSpeed := (40 : ℝ)
  let aliceDelay := (0.5 : ℝ)
  aliceMinSpeed distance bobSpeed aliceDelay > 44 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_speed_to_arrive_before_bob_l804_80440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_open_box_volume_l804_80496

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume 
  (sheet_length sheet_width cut_length : ℝ) 
  (h1 : sheet_length = 48) 
  (h2 : sheet_width = 36) 
  (h3 : cut_length = 3) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 3780 := by
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Simplify the expression
  simp [mul_sub, sub_mul]
  -- Evaluate the arithmetic
  norm_num

#check open_box_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_open_box_volume_l804_80496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_bound_l804_80497

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -2*x - 2 else x^2 - 2*x - 1

theorem sum_bound (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ ≤ x₂) (h₂ : x₂ ≤ x₃) 
  (h₃ : f x₁ = f x₂) (h₄ : f x₂ = f x₃) : 
  3/2 ≤ x₁ + x₂ + x₃ ∧ x₁ + x₂ + x₃ < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_bound_l804_80497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_equilateral_triangle_l804_80491

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1

/-- The ellipse equation -/
def ellipse_eq (x y : ℝ) : Prop := 9 * x^2 + (y + 1)^2 = 9

/-- A point satisfying both circle and ellipse equations -/
def intersection_point (p : ℝ × ℝ) : Prop :=
  circle_eq p.1 p.2 ∧ ellipse_eq p.1 p.2

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The intersection points form an equilateral triangle -/
theorem intersection_points_form_equilateral_triangle :
  ∃ (A B C : ℝ × ℝ),
    intersection_point A ∧
    intersection_point B ∧
    intersection_point C ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    distance A B = distance B C ∧
    distance B C = distance C A :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_form_equilateral_triangle_l804_80491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_symmetric_about_negative_pi_over_4_0_l804_80410

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + Real.sqrt 3 * Real.sin (2 * x + Real.pi / 3) + 1

theorem f_not_symmetric_about_negative_pi_over_4_0 :
  ¬ (∀ (x : ℝ), f ((-Real.pi/4) + x) + f ((-Real.pi/4) - x) = 2 * f (-Real.pi/4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_symmetric_about_negative_pi_over_4_0_l804_80410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_weight_theorem_l804_80452

/-- Calculates the weight of a hollow iron pipe. -/
noncomputable def pipe_weight (length : ℝ) (external_diameter : ℝ) (thickness : ℝ) (density : ℝ) : ℝ :=
  let external_radius := external_diameter / 2
  let internal_radius := external_radius - thickness
  let external_volume := Real.pi * external_radius^2 * length
  let internal_volume := Real.pi * internal_radius^2 * length
  let iron_volume := external_volume - internal_volume
  iron_volume * density

/-- The weight of the hollow iron pipe is 1176π grams. -/
theorem pipe_weight_theorem :
  pipe_weight 21 8 1 8 = 1176 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_weight_theorem_l804_80452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_value_of_a_l804_80450

noncomputable section

open Real

theorem trigonometric_value_of_a (b x a : ℝ) 
  (hb : b > 1)
  (hsinpos : sin x > 0)
  (hcospos : cos x > 0)
  (hsina : sin x = b^(-a))
  (hapos : a > 0)
  (hcosa : cos x = b^(-2*a)) :
  a = -1 / (2 * log b) * log ((sqrt 5 - 1) / 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_value_of_a_l804_80450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_p_l804_80465

/-- Sequence a_n where S_n = (-1)^n * n is the sum of the first n terms -/
def a : ℕ → ℝ
| 0 => 0  -- Add a case for 0
| 1 => -1
| (n + 2) => (-1)^(n + 2) * (n + 2) - (-1)^(n + 1) * (n + 1)

/-- The property that (a_{n+1} - p)(a_n - p) < 0 for all positive integers n -/
def property (p : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (a (n + 1) - p) * (a n - p) < 0

theorem range_of_p :
  ∀ p : ℝ, property p ↔ -1 < p ∧ p < 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_p_l804_80465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_place_is_C_l804_80427

-- Define the students
inductive Student : Type
| A : Student
| B : Student
| C : Student

-- Define the places
inductive Place : Type
| First : Place
| Second : Place
| Third : Place

-- Define a function to assign places to students
variable (assignment : Student → Place)

-- Define the correctness of predictions
def correct_prediction (s : Student) : Prop :=
  match s with
  | Student.A => assignment Student.A ≠ Place.Second
  | Student.B => assignment Student.B = Place.Second
  | Student.C => assignment Student.C ≠ Place.First

-- State the theorem
theorem third_place_is_C :
  -- Each student has a unique place
  (∀ s1 s2 : Student, s1 ≠ s2 → assignment s1 ≠ assignment s2) →
  -- All places are assigned
  (∀ p : Place, ∃ s : Student, assignment s = p) →
  -- Only one prediction is correct
  (∃! s : Student, correct_prediction assignment s) →
  -- C got third place
  assignment Student.C = Place.Third :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_place_is_C_l804_80427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_three_expression_sqrt_five_minus_abs_diff_decimal_cube_root_expression_l804_80430

-- Problem 1
theorem sqrt_three_expression : Real.sqrt 3 * (Real.sqrt 3 + 1 / Real.sqrt 3) = 4 := by sorry

-- Problem 2
theorem sqrt_five_minus_abs_diff : 2 * Real.sqrt 5 - |Real.sqrt 3 - Real.sqrt 5| = Real.sqrt 5 + Real.sqrt 3 := by sorry

-- Problem 3
theorem decimal_cube_root_expression : Real.sqrt 0.16 - (8 : Real)^(1/3) + Real.sqrt (1/4) = -1.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_three_expression_sqrt_five_minus_abs_diff_decimal_cube_root_expression_l804_80430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sides_l804_80446

theorem triangle_arithmetic_sides (a b c : ℝ) (A B C : ℝ) :
  a > b ∧ b > c ∧ c > 0 →
  a - b = 2 ∧ b - c = 2 →
  A > B ∧ B > C →
  Real.sin A = Real.sqrt 3 / 2 →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  Real.sin C = 3*Real.sqrt 3 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_arithmetic_sides_l804_80446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l804_80469

/-- Given mutually orthogonal vectors a, b, and c, prove that scalars p, q, and r
    satisfy the equation (5, -1, 4) = p*a + q*b + r*c -/
theorem vector_decomposition (a b c : ℝ × ℝ × ℝ)
  (h_orthogonal : a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2 = 0 ∧
                  a.1 * c.1 + a.2.1 * c.2.1 + a.2.2 * c.2.2 = 0 ∧
                  b.1 * c.1 + b.2.1 * c.2.1 + b.2.2 * c.2.2 = 0)
  (h_a : a = (1, 2, 2))
  (h_b : b = (2, -1, 0))
  (h_c : c = (0, 2, -1)) :
  ∃ (p q r : ℝ), p = 11/9 ∧ q = 11/5 ∧ r = -6/5 ∧
    (5, -1, 4) = (p * a.1 + q * b.1 + r * c.1,
                  p * a.2.1 + q * b.2.1 + r * c.2.1,
                  p * a.2.2 + q * b.2.2 + r * c.2.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l804_80469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_63871_to_hundredth_l804_80489

noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem round_24_63871_to_hundredth :
  round_to_hundredth 24.63871 = 24.64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_24_63871_to_hundredth_l804_80489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_tangent_line_l804_80454

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem extremum_and_tangent_line (a b : ℝ) :
  (∃ c, c = 0 ∨ c = 2) ∧
  (∃ max min : ℝ, max = 8 ∧ min = -4) := by
  let f := f a b
  have h1 : ∃ c, c = a ∧ (∀ x, x = 1 → (deriv f) x = 0) := by sorry
  have h2 : ∃ d, d = a ∧ f 1 = 2 ∧ (deriv f) 1 = -1 := by sorry
  have h3 : ∀ x, x ∈ Set.Icc (-2 : ℝ) 4 → f x ≤ 8 := by sorry
  have h4 : ∀ x, x ∈ Set.Icc (-2 : ℝ) 4 → f x ≥ -4 := by sorry
  have h5 : f (-2) = -4 ∧ f 4 = 8 := by sorry
  sorry

#check extremum_and_tangent_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_tangent_line_l804_80454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_with_exponential_constraint_l804_80483

theorem max_sum_with_exponential_constraint (a b : ℝ) :
  (2 : ℝ)^a + (2 : ℝ)^b = 1 → (∀ x y : ℝ, (2 : ℝ)^x + (2 : ℝ)^y = 1 → a + b ≥ x + y) → a + b = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_with_exponential_constraint_l804_80483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_imply_a_range_l804_80447

open Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
def g (x : ℝ) : ℝ := x^2

-- Define the property of having exactly three distinct intersection points
def has_three_intersections (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
  f a x = g x ∧ f a y = g y ∧ f a z = g z ∧
  ∀ w : ℝ, f a w = g w → w = x ∨ w = y ∨ w = z

-- State the theorem
theorem intersection_points_imply_a_range :
  ∀ a : ℝ, a > 1 → (has_three_intersections a ↔ a > 1 ∧ a < exp (2/exp 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_imply_a_range_l804_80447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_expression_l804_80422

theorem compute_expression (a b : ℚ) (ha : a = 4/7) (hb : b = 5/3) :
  2 * a^(-3 : ℤ) * b^2 = 17150/576 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compute_expression_l804_80422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_k_power_difference_not_sum_l804_80439

theorem infinitely_many_k_power_difference_not_sum (k : ℕ) (hk : k > 1) :
  ∃ (f : ℕ → ℕ), StrictMono f ∧
    ∀ n, ∃ (a b : ℕ), (a > 0 ∧ b > 0) ∧ f n = a^k - b^k ∧
      ∀ (c d : ℕ), (c > 0 ∧ d > 0) → f n ≠ c^k + d^k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_k_power_difference_not_sum_l804_80439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thin_mints_sold_l804_80477

/-- Represents the number of boxes of each type of cookie sold --/
structure CookieSales where
  samoas : ℕ
  thinMints : ℕ
  fudgeDelights : ℕ
  sugarCookies : ℕ

/-- Represents the price of each type of cookie --/
structure CookiePrices where
  samoas : ℚ
  thinMints : ℚ
  fudgeDelights : ℚ
  sugarCookies : ℚ

/-- Calculates the total revenue from cookie sales --/
def totalRevenue (sales : CookieSales) (prices : CookiePrices) : ℚ :=
  sales.samoas * prices.samoas +
  sales.thinMints * prices.thinMints +
  sales.fudgeDelights * prices.fudgeDelights +
  sales.sugarCookies * prices.sugarCookies

theorem thin_mints_sold (sales : CookieSales) (prices : CookiePrices) :
  sales.samoas = 3 ∧
  sales.fudgeDelights = 1 ∧
  sales.sugarCookies = 9 ∧
  prices.samoas = 4 ∧
  prices.thinMints = 7/2 ∧
  prices.fudgeDelights = 5 ∧
  prices.sugarCookies = 2 ∧
  totalRevenue sales prices = 42 →
  sales.thinMints = 2 := by
  sorry

-- Remove the #eval line as it's not necessary and can cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thin_mints_sold_l804_80477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_4_seconds_l804_80474

noncomputable def displacement (t : ℝ) : ℝ := 3 * t^2 + t + 4

noncomputable def instantaneous_velocity (t : ℝ) : ℝ := 
  deriv displacement t

theorem velocity_at_4_seconds :
  instantaneous_velocity 4 = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_velocity_at_4_seconds_l804_80474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l804_80492

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0) : 
  let e := Real.sqrt (a^2 - b^2) / a
  ∃ (F A B P : ℝ × ℝ),
    (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ↔ (x, y) ∈ Set.range (λ t ↦ (a * Real.cos t, b * Real.sin t))) ∧
    F.1 = -Real.sqrt (a^2 - b^2) ∧
    A = (a, 0) ∧
    B ∈ Set.range (λ t ↦ (a * Real.cos t, b * Real.sin t)) ∧
    B.1 = F.1 ∧
    P.1 = 0 ∧
    (A.2 - P.2) / (P.2 - B.2) = 2 →
  e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l804_80492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_little_john_money_left_l804_80411

theorem little_john_money_left (initial_amount spent_on_sweets amount_per_friend number_of_friends : ℚ) 
  (h1 : initial_amount = 8.5)
  (h2 : spent_on_sweets = 1.25)
  (h3 : amount_per_friend = 1.2)
  (h4 : number_of_friends = 2) :
  initial_amount - spent_on_sweets - (amount_per_friend * number_of_friends) = 4.85 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and might cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_little_john_money_left_l804_80411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_counts_l804_80404

def number_of_men : ℕ := 5
def number_of_women : ℕ := 3
def total_people : ℕ := number_of_men + number_of_women

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements_women_together : ℕ := 
  factorial (total_people - number_of_women + 1) * factorial number_of_women

def arrangements_men_on_sides : ℕ := 
  factorial number_of_men * (number_of_men + 1 - number_of_women)

theorem arrangement_counts :
  arrangements_women_together = 4320 ∧
  arrangements_men_on_sides = 2880 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_counts_l804_80404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l804_80468

/-- Given vectors in R^2 -/
def OA : Fin 2 → ℝ := ![1, 7]
def OB : Fin 2 → ℝ := ![5, 1]
def OP : Fin 2 → ℝ := ![2, 1]

/-- Q is a point on the line OP -/
def Q (t : ℝ) : Fin 2 → ℝ := fun i => t * OP i

/-- The dot product of two vectors -/
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

/-- The magnitude of a vector -/
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

/-- Vector addition -/
def vec_add (v w : Fin 2 → ℝ) : Fin 2 → ℝ :=
  fun i => v i + w i

/-- Vector subtraction -/
def vec_sub (v w : Fin 2 → ℝ) : Fin 2 → ℝ :=
  fun i => v i - w i

theorem vector_problem :
  (magnitude (vec_add OA OB) = 10) ∧
  (∃ t : ℝ, Q t = ![4, 2] ∧
    ∀ s : ℝ, dot_product (vec_sub OA (Q s)) (vec_sub OB (Q s)) ≥
             dot_product (vec_sub OA (Q t)) (vec_sub OB (Q t))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l804_80468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l804_80451

def f : ℕ → ℕ
  | 0 => 0
  | n + 1 => if n % 2 = 0 then 2 * f (n / 2) else n / 2 + 2 * f (n / 2)

def L : Set ℕ := {n | ∃ k > 0, n = 2 * k}
def E : Set ℕ := {0} ∪ {n | ∃ k ≥ 0, n = 4 * k + 1}
def G : Set ℕ := {n | ∃ k ≥ 0, n = 4 * k + 3}

def a (k : ℕ) : ℕ := k * 2^(k-1) - 2^k + 1

theorem f_properties :
  (∀ n, f (2*n) = 2 * f n) ∧
  (∀ n, f (2*n + 1) = n + 2 * f n) ∧
  (L = {n | f n < f (n+1)}) ∧
  (E = {n | f n = f (n+1)}) ∧
  (G = {n | f n > f (n+1)}) ∧
  (∀ k, a k = Finset.sup (Finset.range (2^k + 1)) f) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l804_80451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_k_with_no_primes_l804_80425

def sequenceX (k : ℕ) : ℕ → ℕ
| 0 => 1
| 1 => k + 2
| (n + 2) => (k + 1) * sequenceX k (n + 1) - sequenceX k n

theorem infinitely_many_k_with_no_primes :
  ∃ S : Set ℕ, (Set.Infinite S) ∧ 
  (∀ k ∈ S, k > 1 ∧ 
    (∀ n : ℕ, ¬(Nat.Prime (sequenceX k n)))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_k_with_no_primes_l804_80425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l804_80475

theorem hyperbola_eccentricity (a b : ℝ) (h1 : b > 0) 
  (h2 : ∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) 
  (h3 : ∃ (θ : ℝ), θ = π/3 ∧ ∃ (m : ℝ), m = Real.tan θ ∧ m = b/a) :
  Real.sqrt (1 + b^2/a^2) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l804_80475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_of_special_pyramid_l804_80453

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cuboid in 3D space -/
structure Cuboid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Represents a triangular pyramid in 3D space -/
structure TriangularPyramid where
  D₁ : Point3D
  D : Point3D
  M : Point3D
  N : Point3D

noncomputable def dihedral_angle (p : TriangularPyramid) : ℝ := sorry

noncomputable def volume (p : TriangularPyramid) : ℝ := sorry

noncomputable def circumscribed_sphere_surface_area (p : TriangularPyramid) : ℝ := sorry

theorem circumscribed_sphere_surface_area_of_special_pyramid 
  (c : Cuboid) (p : TriangularPyramid) :
  c.A.x - c.B.x = 6 →
  c.B.y - c.C.y = 6 →
  c.A.z - c.A₁.z = 2 →
  p.D = c.D →
  p.D₁ = c.D₁ →
  p.M.x = c.D.x ∧ p.M.y = c.D.y →
  p.N.x = c.D.x ∧ p.N.z = c.D.z →
  dihedral_angle p = π / 4 →
  volume p = 16 * Real.sqrt 3 / 9 →
  circumscribed_sphere_surface_area p = 76 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_of_special_pyramid_l804_80453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_strong_boys_possible_l804_80478

-- Define the type for a boy with height and weight
structure Boy :=
  (height : ℕ)
  (weight : ℕ)

-- Define what it means for a boy to be not inferior to another
def notInferiorTo (a b : Boy) : Prop :=
  a.height > b.height ∨ a.weight > b.weight

-- Define what it means for a boy to be a strong boy in a group
def isStrongBoy (boy : Boy) (group : List Boy) : Prop :=
  ∀ other, other ∈ group → other ≠ boy → notInferiorTo boy other

-- Theorem: It's possible to have 100 strong boys in a group of 100 boys
theorem max_strong_boys_possible :
  ∃ (group : List Boy),
    group.length = 100 ∧
    (∀ b₁ b₂, b₁ ∈ group → b₂ ∈ group → b₁ ≠ b₂ → b₁.height ≠ b₂.height ∧ b₁.weight ≠ b₂.weight) ∧
    (∀ boy, boy ∈ group → isStrongBoy boy group) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_strong_boys_possible_l804_80478
