import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_num_common_tangents_l684_68489

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y + 4 = 0

-- Define the centers of the circles
def center₁ : ℝ × ℝ := (-1, -1)
def center₂ : ℝ × ℝ := (2, 1)

-- Define the radii of the circles
def radius₁ : ℝ := 2
def radius₂ : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 13

-- Theorem: The number of common tangent lines is 4
def num_common_tangents : ℕ := 4

-- The proof
theorem prove_num_common_tangents : num_common_tangents = 4 := by
  -- Unfold the definition of num_common_tangents
  unfold num_common_tangents
  -- Assert that the circles are separate
  have h_separate : distance_between_centers > radius₁ + radius₂ := by
    -- Prove that the circles are separate
    sorry
  -- Conclude that there are 4 common tangent lines
  exact rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_num_common_tangents_l684_68489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_segments_less_than_perimeter_l684_68454

/-- Triangle with side lengths a, b, c where c ≤ b ≤ a -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h₁ : 0 < c
  h₂ : c ≤ b
  h₃ : b ≤ a
  h₄ : c + b > a -- triangle inequality

/-- Interior point of a triangle -/
structure InteriorPoint (t : Triangle) where
  x : ℝ
  y : ℝ
  h : x > 0 ∧ y > 0 ∧ x + y < 1 -- Simple representation of an interior point

/-- Line segment from a vertex through an interior point to the opposite side -/
noncomputable def VertexToOpposite (t : Triangle) (p : InteriorPoint t) : ℝ × ℝ × ℝ :=
  sorry

/-- Sum of lengths of line segments from each vertex through P to the opposite side -/
noncomputable def SumOfSegments (t : Triangle) (p : InteriorPoint t) : ℝ :=
  let (aa', bb', cc') := VertexToOpposite t p
  aa' + bb' + cc'

/-- Theorem: The sum of segments is less than the sum of triangle sides -/
theorem sum_of_segments_less_than_perimeter (t : Triangle) (p : InteriorPoint t) :
    SumOfSegments t p < t.a + t.b + t.c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_segments_less_than_perimeter_l684_68454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_perpendicular_medians_l684_68418

/-- A right triangle with perpendicular medians -/
structure RightTriangleWithPerpendicularMedians where
  -- The points of the triangle
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  -- DP is a median
  P : ℝ × ℝ
  -- EQ is a median
  Q : ℝ × ℝ
  -- D is the right angle
  right_angle_at_D : (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0
  -- DP and EQ are perpendicular
  medians_perpendicular : (P.1 - D.1) * (Q.1 - E.1) + (P.2 - D.2) * (Q.2 - E.2) = 0
  -- P is on EF
  P_on_EF : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ P = (t * E.1 + (1 - t) * F.1, t * E.2 + (1 - t) * F.2)
  -- Q is on DF
  Q_on_DF : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ Q = (s * D.1 + (1 - s) * F.1, s * D.2 + (1 - s) * F.2)
  -- Length of DP is 27
  DP_length : Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) = 27
  -- Length of EQ is 30
  EQ_length : Real.sqrt ((Q.1 - E.1)^2 + (Q.2 - E.2)^2) = 30

/-- The main theorem -/
theorem right_triangle_perpendicular_medians
  (triangle : RightTriangleWithPerpendicularMedians) :
  Real.sqrt ((triangle.E.1 - triangle.D.1)^2 + (triangle.E.2 - triangle.D.2)^2) = 2 * Real.sqrt 181 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_perpendicular_medians_l684_68418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_2_minus_6_tangent_through_origin_l684_68404

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem for the tangent line at (2, -6)
theorem tangent_at_2_minus_6 :
  let tangent_slope := f' 2
  let tangent_eq (x : ℝ) := tangent_slope * x - 32
  (∀ x, tangent_eq x = f 2 + f' 2 * (x - 2)) ∧
  f 2 = -6 ∧
  tangent_eq 2 = -6 := by sorry

-- Theorem for the tangent line through origin
theorem tangent_through_origin :
  ∃ x y : ℝ,
    f x = y ∧
    f' x * x = y ∧
    x = -2 ∧
    y = -26 ∧
    (∀ t, f' x * t = f t + f' x * (t - x)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_at_2_minus_6_tangent_through_origin_l684_68404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l684_68451

/-- Calculates the length of a train given its speed, the platform length, and the time to cross the platform. -/
noncomputable def train_length (speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : ℝ :=
  speed * crossing_time / 3.6 - platform_length

/-- Theorem stating that a train with speed 72 km/hr crossing a 250 m platform in 15 seconds has a length of 50 meters. -/
theorem train_length_calculation :
  train_length 72 250 15 = 50 := by
  unfold train_length
  -- The actual proof steps would go here
  sorry

-- Remove the #eval statement as it's not computable
-- #eval train_length 72 250 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l684_68451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisects_chord_l684_68400

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 8 = 1

/-- The line that bisects the chord -/
def bisecting_line (x y : ℝ) : Prop := x + y - 3 = 0

/-- The midpoint of the chord -/
def chord_midpoint : ℝ × ℝ := (2, 1)

/-- Theorem stating that the given line bisects a chord of the ellipse at the given point -/
theorem bisects_chord :
  ∃ (a b : ℝ × ℝ),
    ellipse a.1 a.2 ∧
    ellipse b.1 b.2 ∧
    bisecting_line ((a.1 + b.1) / 2) ((a.2 + b.2) / 2) ∧
    (a.1 + b.1) / 2 = chord_midpoint.1 ∧
    (a.2 + b.2) / 2 = chord_midpoint.2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisects_chord_l684_68400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_and_sqrt_problem_l684_68415

-- Define the opposite of a real number
def opposite (x : ℝ) : ℝ := -x

-- Define the arithmetic square root
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Theorem statement
theorem opposite_and_sqrt_problem :
  (opposite (Real.sqrt 5) = -Real.sqrt 5) ∧
  (arithmetic_sqrt (Real.sqrt 81) = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_and_sqrt_problem_l684_68415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_formula_l684_68461

/-- Represents an annulus with outer radius b, inner radius c, and tangent length a -/
structure Annulus where
  b : ℝ
  c : ℝ
  a : ℝ
  h1 : b > c
  h2 : a^2 = b^2 - c^2

/-- The length of the diagonal in the annulus configuration -/
noncomputable def diagonalLength (ann : Annulus) : ℝ :=
  Real.sqrt (ann.b^2 + ann.c^2)

/-- Theorem stating that the diagonal length is equal to √(b² + c²) -/
theorem diagonal_length_formula (ann : Annulus) :
  diagonalLength ann = Real.sqrt (ann.b^2 + ann.c^2) := by
  -- Unfold the definition of diagonalLength
  unfold diagonalLength
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_formula_l684_68461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_transformation_l684_68407

noncomputable def f : ℝ → ℝ := fun x ↦ (1/2) * Real.sin (2*x - Real.pi/2) + 1

noncomputable def transform (g : ℝ → ℝ) : ℝ → ℝ := 
  fun x ↦ g ((x + Real.pi/2) / 2) - 1

theorem f_transformation :
  transform f = fun x ↦ (1/2) * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_transformation_l684_68407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l684_68421

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 3) / (x^2 - 1)

theorem f_values :
  f 0 = -3 ∧ f (-3) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l684_68421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l684_68464

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  (Real.sin (B / 2) - Real.cos (B / 2) = 1 / 5) →
  (b^2 - a^2 = a * c) →
  (Real.cos B = -7 / 25) ∧
  (Real.sin C / Real.sin A = 11 / 25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l684_68464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_P_equation_l684_68424

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ ∧ 0 ≤ θ ∧ θ < Real.pi / 2
def C₂ (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 3

-- Define the intersection point
def intersection_point : ℝ × ℝ := (2 * Real.sqrt 3, Real.pi / 6)

-- Define the relation between Q and P
def Q_P_relation (ρ_Q θ_Q ρ_P θ_P : ℝ) : Prop :=
  C₁ ρ_Q θ_Q ∧ ρ_Q = (2/3) * ρ_P ∧ θ_Q = θ_P

-- Theorem for the intersection point
theorem intersection_point_correct :
  C₁ (intersection_point.1) (intersection_point.2) ∧
  C₂ (intersection_point.1) (intersection_point.2) := by sorry

-- Theorem for the equation of P
theorem P_equation (ρ θ : ℝ) :
  (∃ ρ_Q θ_Q, Q_P_relation ρ_Q θ_Q ρ θ) →
  ρ = 10 * Real.cos θ ∧ 0 ≤ θ ∧ θ < Real.pi / 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_P_equation_l684_68424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_yoz_plane_l684_68460

/-- Given a point A (-3, 5, 2), its symmetric point with respect to the yOz plane has coordinates (3, 5, 2). -/
theorem symmetric_point_yoz_plane :
  let A : ℝ × ℝ × ℝ := (-3, 5, 2)
  let symmetric_point : ℝ × ℝ × ℝ := (3, 5, 2)
  (Prod.fst symmetric_point = -(Prod.fst A)) ∧ 
  (Prod.snd symmetric_point = Prod.snd A) ∧ 
  (Prod.snd (Prod.snd symmetric_point) = Prod.snd (Prod.snd A)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_point_yoz_plane_l684_68460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_l684_68410

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 8x -/
def Parabola := {p : Point | p.y^2 = 8 * p.x}

/-- Represents a line y = k(x-2) -/
def Line (k : ℝ) := {p : Point | p.y = k * (p.x - 2)}

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The focus of the parabola -/
def F : Point := ⟨2, 0⟩

theorem parabola_line_intersection_sum (k : ℝ) 
  (P Q : Point) 
  (hP : P ∈ Parabola ∩ Line k) 
  (hQ : Q ∈ Parabola ∩ Line k) 
  (hPQ : P ≠ Q) :
  1 / distance F P + 1 / distance F Q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_l684_68410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l684_68426

theorem cos_minus_sin_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = -12/25) 
  (h2 : α ∈ Set.Ioo (-Real.pi/2) 0) : 
  Real.cos α - Real.sin α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l684_68426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l684_68422

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem omega_range (ω : ℝ) :
  ω > 0 →
  (∀ x y : ℝ, -π/3 ≤ x ∧ x < y ∧ y ≤ π/4 → f ω x < f ω y) →
  0 < ω ∧ ω ≤ 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l684_68422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_not_72_l684_68452

theorem product_not_72 : ∃! (pair : ℚ × ℚ), 
  pair ∈ [(-6, -12), (3, 24), (2, -36), (2, 36), (4/3, 54)] ∧ 
  pair.1 * pair.2 ≠ 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_not_72_l684_68452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l684_68455

noncomputable def f (x : ℝ) : ℝ := (2*x - 1) / (x + 1)

theorem f_properties :
  -- 1. Domain is all real numbers except -1
  (∀ x : ℝ, x ≠ -1 → f x ∈ Set.univ) ∧
  -- 2. f(x) is strictly increasing on (-1, +∞)
  (∀ x y : ℝ, -1 < x ∧ x < y → f x < f y) ∧
  -- 3. Minimum and maximum values on [3,5]
  (∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → 5/4 ≤ f x ∧ f x ≤ 3/2) ∧
  (f 3 = 5/4) ∧ (f 5 = 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l684_68455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_balls_even_bins_l684_68459

/-- The probability of a ball landing in bin k -/
def prob_bin (k : ℕ+) : ℚ := 1 / (2 : ℚ)^(k : ℕ)

/-- The probability of a ball landing in any even-numbered bin -/
noncomputable def prob_even_bin : ℚ := ∑' k : ℕ+, prob_bin (2 * k)

/-- The probability of both balls landing in even-numbered bins -/
noncomputable def prob_both_even : ℚ := prob_even_bin * prob_even_bin

theorem both_balls_even_bins :
  prob_both_even = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_balls_even_bins_l684_68459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l684_68428

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 4*x + 2)

-- State the theorem about the range of the function
theorem f_range : Set.range f = { y : ℝ | 0 ≤ y ∧ y ≤ Real.sqrt 6 } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l684_68428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_calculation_l684_68483

/-- Calculates the length of a tunnel given train parameters --/
noncomputable def tunnel_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_min : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance := train_speed_ms * (crossing_time_min * 60)
  distance - train_length

theorem tunnel_length_calculation :
  let train_length := (800 : ℝ)
  let train_speed_kmh := (78 : ℝ)
  let crossing_time_min := (1 : ℝ)
  abs (tunnel_length train_length train_speed_kmh crossing_time_min - 500.2) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_calculation_l684_68483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freddy_travel_time_l684_68443

/-- Represents the travel details of a person -/
structure TravelDetails where
  distance : ℚ
  time : ℚ

/-- Calculates the average speed given travel details -/
def averageSpeed (t : TravelDetails) : ℚ :=
  t.distance / t.time

theorem freddy_travel_time
  (eddy_travel : TravelDetails)
  (freddy_distance : ℚ)
  (speed_ratio : ℚ)
  (freddy_time : ℚ)
  (h1 : eddy_travel.distance = 510)
  (h2 : eddy_travel.time = 3)
  (h3 : freddy_distance = 300)
  (h4 : speed_ratio = 2266666666666667 / 1000000000000000)
  (h5 : averageSpeed eddy_travel / (freddy_distance / freddy_time) = speed_ratio)
  : freddy_time = 4 := by
  sorry

#check freddy_travel_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freddy_travel_time_l684_68443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_solution_l684_68406

/-- Represents the water usage and cost for a month -/
structure MonthData where
  usage : ℚ
  cost : ℚ

/-- Water pricing system for a city -/
structure WaterPricing where
  standard_usage : ℚ
  normal_rate : ℚ
  excess_rate : ℚ

/-- Calculate the total cost for a given month -/
def calculate_cost (pricing : WaterPricing) (month : MonthData) : ℚ :=
  let standard_cost := pricing.standard_usage * pricing.normal_rate
  let excess_usage := max (month.usage - pricing.standard_usage) 0
  standard_cost + excess_usage * pricing.excess_rate

/-- The main theorem to prove -/
theorem water_pricing_solution :
  ∃ (pricing : WaterPricing),
    pricing.standard_usage = 10 ∧
    calculate_cost pricing ⟨15, 35⟩ = 35 ∧
    calculate_cost pricing ⟨18, 44⟩ = 44 ∧
    pricing.normal_rate = 2 ∧
    pricing.excess_rate = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_solution_l684_68406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_l684_68414

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

theorem f_monotone_intervals (a : ℝ) (h_a_pos : a > 0) :
  (∀ x ∈ Set.Icc 0 (a / 3), StrictMonoOn f (Set.Icc 0 (a / 3))) →
  (∀ x ∈ Set.Icc (2 * a) (4 * Real.pi / 3), StrictMonoOn f (Set.Icc (2 * a) (4 * Real.pi / 3))) →
  a ∈ Set.Icc (5 * Real.pi / 12) Real.pi :=
by
  sorry

#check f_monotone_intervals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_l684_68414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_equilateral_triangle_with_inscribed_circle_l684_68403

theorem shaded_area_in_equilateral_triangle_with_inscribed_circle :
  let side_length : ℝ := 10
  let radius : ℝ := side_length / 2
  let sector_area : ℝ := (π * radius^2) / 8
  let triangle_area : ℝ := (Real.sqrt 3 * radius^2) / 4
  let shaded_area : ℝ := 2 * (sector_area - triangle_area)
  shaded_area = 25 * π / 4 - 50 * Real.sqrt 3 / 4 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_equilateral_triangle_with_inscribed_circle_l684_68403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lima_beans_cost_and_weight_l684_68411

/-- Represents the mixture of lima beans and corn -/
structure Mixture where
  total_weight : ℚ
  corn_weight : ℚ
  corn_cost : ℚ
  mixture_cost : ℚ

/-- Calculates the cost and weight of lima beans in the mixture -/
def calculate_lima_beans (m : Mixture) : ℚ × ℚ :=
  let lima_weight := m.total_weight - m.corn_weight
  let total_cost := m.total_weight * m.mixture_cost
  let corn_cost := m.corn_weight * m.corn_cost
  let lima_cost := (total_cost - corn_cost) / lima_weight
  (lima_cost, lima_weight)

/-- Theorem stating the cost and weight of lima beans in the given mixture -/
theorem lima_beans_cost_and_weight (m : Mixture) 
    (h1 : m.total_weight = 256/10)
    (h2 : m.corn_weight = 16)
    (h3 : m.corn_cost = 1/2)
    (h4 : m.mixture_cost = 13/20) :
    calculate_lima_beans m = (9/10, 96/10) := by
  sorry

#eval calculate_lima_beans { total_weight := 256/10, corn_weight := 16, corn_cost := 1/2, mixture_cost := 13/20 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lima_beans_cost_and_weight_l684_68411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_128_over_3_l684_68463

-- Define the rectangle EFGH
structure Rectangle where
  EF : ℝ
  FG : ℝ
  isRectangle : EF > 0 ∧ FG > 0

-- Define the pyramid PEFGH
structure Pyramid where
  base : Rectangle
  EP : ℕ
  θ : ℝ
  hEq : Real.cos θ = 4/5
  peLengths : ℕ → ℕ → ℕ → Prop

-- Define the volume function for the pyramid
noncomputable def pyramidVolume (p : Pyramid) : ℝ :=
  (1/3) * p.base.EF * p.base.FG * (4/5 * p.EP)

-- State the theorem
theorem pyramid_volume_is_128_over_3 (p : Pyramid) 
  (h1 : p.peLengths x (x+2) (x+4))
  (h2 : p.EP = 5)
  (h3 : p.base.EF = 4)
  (h4 : p.base.FG = 8) :
  pyramidVolume p = 128/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_128_over_3_l684_68463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_area_l684_68441

-- Define the curve
noncomputable def f (x : ℝ) := Real.cos x

-- Define the domain
def domain : Set ℝ := Set.Icc 0 (3 * Real.pi / 2)

-- Define the area function
noncomputable def area : ℝ := ∫ x in domain, f x

-- Theorem statement
theorem cosine_area : area = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_area_l684_68441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l684_68429

theorem trigonometric_identities 
  (x : ℝ) 
  (h1 : Real.cos (x - Real.pi/4) = Real.sqrt 2/10) 
  (h2 : x ∈ Set.Ioo (Real.pi/2) (3*Real.pi/4)) : 
  (Real.sin x = 4/5) ∧ 
  (Real.sin (2*x - Real.pi/6) = (7 - 24*Real.sqrt 3)/50) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l684_68429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_iff_coprime_l684_68478

theorem distinct_remainders_iff_coprime (n k : ℕ) (hn : n > 0) (hk : k > 0) :
  (∃ (a : Fin n → ℤ) (b : Fin k → ℤ), ∀ (i j i' j' : ℕ) 
    (hi : i < n) (hj : j < k) (hi' : i' < n) (hj' : j' < k),
    (i ≠ i' ∨ j ≠ j') → 
    (a ⟨i, hi⟩ * b ⟨j, hj⟩) % (n * k) ≠ (a ⟨i', hi'⟩ * b ⟨j', hj'⟩) % (n * k)) ↔
  Nat.Coprime n k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_remainders_iff_coprime_l684_68478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l684_68439

def sequenceA (n : ℕ) : ℚ := (2 * n - 5) / (3 * n + 1)

theorem sequence_limit : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequenceA n - 2/3| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_l684_68439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_geometric_sequence_l684_68432

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def is_arithmetic_seq (s : List ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin (s.length - 1), s[i.val + 1] - s[i.val] = d

theorem sum_of_special_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_first : a 1 = 1)
  (h_arith : is_arithmetic_seq [4 * a 2, 2 * a 3, a 4]) :
  a 2 + a 3 + a 4 = 14 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_geometric_sequence_l684_68432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_unit_interval_l684_68420

theorem root_in_unit_interval (f : ℝ → ℝ) (a b : ℕ+) (x₀ : ℝ) :
  (∀ x, f x = 3^x + x - 5) →
  (b - a : ℝ) = 1 →
  x₀ ∈ Set.Icc (a : ℝ) (b : ℝ) →
  f x₀ = 0 →
  a + b = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_unit_interval_l684_68420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_heads_two_or_four_l684_68449

/-- A fair coin -/
def FairCoin : Type := Bool

/-- A biased eight-sided die -/
def BiasedDie : Type := Fin 8

/-- The probability of getting heads on a fair coin -/
noncomputable def probHeads : ℝ := 1 / 2

/-- The probability of rolling a 2 or 4 on the biased die -/
noncomputable def probTwoOrFour : ℝ := 1 / 2

/-- The probability of getting heads on the coin and either 2 or 4 on the die -/
noncomputable def probHeadsTwoOrFour : ℝ := probHeads * probTwoOrFour

theorem probability_heads_two_or_four :
  probHeadsTwoOrFour = 1 / 4 := by
  unfold probHeadsTwoOrFour probHeads probTwoOrFour
  field_simp
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_heads_two_or_four_l684_68449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_parallel_nonintersecting_l684_68438

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def Circle (r : ℝ) : Set Point :=
  {p : Point | p.x^2 + p.y^2 = r^2}

def isInside (p : Point) (c : Set Point) (r : ℝ) : Prop :=
  p.x^2 + p.y^2 < r^2

def isParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

def doesNotIntersect (l : Line) (c : Set Point) : Prop :=
  ∀ p : Point, p ∈ c → l.a * p.x + l.b * p.y ≠ l.c

theorem circle_chord_parallel_nonintersecting
  (r a b : ℝ)
  (P : Point)
  (m l : Line)
  (h1 : P.x = a ∧ P.y = b)
  (h2 : a ≠ 0 ∧ b ≠ 0)
  (h3 : isInside P (Circle r) r)
  (h4 : l.a = a ∧ l.b = b ∧ l.c = r^2)
  (h5 : ∃ p1 p2 : Point, p1 ≠ p2 ∧ p1 ∈ Circle r ∧ p2 ∈ Circle r ∧ m.a * p1.x + m.b * p1.y = m.c ∧ m.a * p2.x + m.b * p2.y = m.c ∧ m.a * P.x + m.b * P.y = m.c) :
  isParallel m l ∧ doesNotIntersect l (Circle r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_chord_parallel_nonintersecting_l684_68438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_not_sufficient_condition_necessary_but_not_sufficient_l684_68419

open Real

-- Define what it means for an angle to be in the fourth quadrant
def is_in_fourth_quadrant (θ : ℝ) : Prop :=
  cos θ > 0 ∧ sin θ < 0

-- Statement 1: If angle θ is in the fourth quadrant, then cosθ•tanθ < 0
theorem necessary_condition (θ : ℝ) :
  is_in_fourth_quadrant θ → cos θ * tan θ < 0 := by
  sorry

-- Statement 2: There exists an angle θ such that cosθ•tanθ < 0, but θ is not in the fourth quadrant
theorem not_sufficient_condition :
  ∃ θ : ℝ, cos θ * tan θ < 0 ∧ ¬is_in_fourth_quadrant θ := by
  sorry

-- Combine both statements to show it's necessary but not sufficient
theorem necessary_but_not_sufficient :
  (∀ θ : ℝ, is_in_fourth_quadrant θ → cos θ * tan θ < 0) ∧
  (∃ θ : ℝ, cos θ * tan θ < 0 ∧ ¬is_in_fourth_quadrant θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_necessary_condition_not_sufficient_condition_necessary_but_not_sufficient_l684_68419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_new_plan_cost_l684_68408

/-- The cost of a new phone plan given the old plan cost and percentage increase -/
noncomputable def new_plan_cost (old_cost : ℝ) (percent_increase : ℝ) : ℝ :=
  old_cost * (1 + percent_increase / 100)

/-- Theorem stating that Mark's new plan costs $195 given the conditions -/
theorem marks_new_plan_cost :
  new_plan_cost 150 30 = 195 := by
  -- Unfold the definition of new_plan_cost
  unfold new_plan_cost
  -- Simplify the arithmetic
  simp [mul_add, mul_div_right_comm]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marks_new_plan_cost_l684_68408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_to_total_ratio_solution_satisfies_conditions_l684_68491

/-- Represents the number of cats Matt has initially -/
def initial_cats : ℕ := 6

/-- Represents the number of kittens each female cat has -/
def kittens_per_female : ℕ := 7

/-- Represents the number of kittens Matt sells -/
def sold_kittens : ℕ := 9

/-- Represents the percentage of remaining cats that are kittens -/
def kitten_percentage : ℚ := 67/100

/-- Theorem stating that the ratio of female cats to total cats is 1:2 -/
theorem female_to_total_ratio (female_cats : ℕ) :
  female_cats * 2 = initial_cats :=
sorry

/-- Checks if the solution satisfies the problem conditions -/
def check_solution (female_cats : ℕ) : Prop :=
  let total_kittens := female_cats * kittens_per_female
  let remaining_kittens := total_kittens - sold_kittens
  let total_cats_after_sale := initial_cats + total_kittens - sold_kittens
  (remaining_kittens : ℚ) / total_cats_after_sale = kitten_percentage

/-- Theorem stating that the solution satisfies the problem conditions -/
theorem solution_satisfies_conditions :
  check_solution 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_to_total_ratio_solution_satisfies_conditions_l684_68491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_proposition_l684_68445

theorem negation_of_existence_proposition :
  (¬ ∃ (x : ℝ), Real.sin x > 1) ↔ (∀ (x : ℝ), Real.sin x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_proposition_l684_68445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_area_ratio_l684_68401

noncomputable section

-- Define an equilateral triangle with side length 6
def equilateral_triangle_side : ℝ := 6

-- Define the perimeter of the equilateral triangle
noncomputable def perimeter : ℝ := 3 * equilateral_triangle_side

-- Define the area of the equilateral triangle
noncomputable def area : ℝ := (equilateral_triangle_side ^ 2 * Real.sqrt 3) / 4

-- Theorem: The ratio of perimeter to area is 2√3 / 3
theorem perimeter_area_ratio :
  perimeter / area = 2 * Real.sqrt 3 / 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_area_ratio_l684_68401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_alone_time_l684_68447

/-- The time (in days) it takes David and Andrew to finish the work together -/
noncomputable def joint_time : ℚ := 12

/-- The time (in days) David and Andrew worked together before Andrew left -/
noncomputable def initial_work_time : ℚ := 8

/-- The time (in days) it took David to finish the remaining work after Andrew left -/
noncomputable def remaining_work_time : ℚ := 8

/-- David's work rate (fraction of work completed per day) -/
noncomputable def david_rate : ℚ := (1 - initial_work_time / joint_time) / remaining_work_time

theorem david_alone_time : (1 / david_rate : ℚ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_alone_time_l684_68447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_decreasing_function_l684_68485

theorem positive_decreasing_function 
  (f : ℝ → ℝ) 
  (h_decreasing : ∀ x y, x < y → f x > f y) 
  (h_inequality : ∀ x, f x / (deriv (deriv f) x) + x < 1) : 
  ∀ x, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_decreasing_function_l684_68485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_in_first_set_prove_third_number_l684_68484

theorem third_number_in_first_set (third_number : ℝ) : Prop :=
  let x : ℝ := (75.6 * 5 - (50 + 62 + 97 + 124))
  let first_set : List ℝ := [28, x, third_number, 88, 104]
  let second_set : List ℝ := [50, 62, 97, 124, x]
  (first_set.sum / first_set.length = 67) ∧
  (second_set.sum / second_set.length = 75.6) →
  third_number = 94

theorem prove_third_number : ∃ (third_number : ℝ), third_number_in_first_set third_number := by
  -- The proof goes here
  sorry

#check prove_third_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_number_in_first_set_prove_third_number_l684_68484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_roots_l684_68431

-- Define the cubic root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the equation
noncomputable def f (x : ℝ) : ℝ := (x - cubeRoot 23) * (x - cubeRoot 73) * (x - cubeRoot 123) - (1/5)

-- State the theorem
theorem sum_of_cubes_of_roots (r s t : ℝ) :
  f r = 0 → f s = 0 → f t = 0 →
  r^3 + s^3 + t^3 = 219.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_roots_l684_68431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_reflection_tangent_circle_l684_68416

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in 2D space, represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

/-- The reflection of a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: For a ray passing through P(-3, 1) and Q(a, 0), reflecting off the x-axis,
    and being tangent to the circle x^2 + y^2 = 1, the value of a is -5/3 -/
theorem ray_reflection_tangent_circle (a : ℝ) :
  let P : Point := { x := -3, y := 1 }
  let Q : Point := { x := a, y := 0 }
  let P' : Point := reflectAcrossXAxis P
  let unitCircle : Circle := { center := { x := 0, y := 0 }, radius := 1 }
  let reflectedLine : Line := { a := 1, b := -(3 + a), c := -a }
  distancePointToLine unitCircle.center reflectedLine = unitCircle.radius →
  a = -5/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ray_reflection_tangent_circle_l684_68416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_theorem_l684_68477

def arrangement_count (n a b c : ℕ) : ℕ :=
  Finset.sum (Finset.range 8) (λ y =>
    Finset.sum (Finset.range (min 8 (8 - y))) (λ x =>
      Nat.choose 7 x * Nat.choose 7 y * Nat.choose 6 (6 - y) * Nat.choose (7 - x) (7 - x - y)))

theorem arrangement_theorem (n a b c : ℕ) :
  arrangement_count n a b c =
  Finset.sum (Finset.range 8) (λ y =>
    Finset.sum (Finset.range (min 8 (8 - y))) (λ x =>
      Nat.choose 7 x * Nat.choose 7 y * Nat.choose 6 (6 - y) * Nat.choose (7 - x) (7 - x - y))) :=
by sorry

#eval arrangement_count 20 6 7 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_theorem_l684_68477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correctOptionChosen_l684_68482

def chooseOption (n : Nat) : Fin 4 :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | _ => 3

def correctAnswer : Fin 4 := 1  -- Representing option B

theorem correctOptionChosen : chooseOption 1 = correctAnswer := by
  rfl

#eval chooseOption 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correctOptionChosen_l684_68482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_for_10_l684_68480

open BigOperators

def Q (n : ℕ) : ℚ :=
  ∏ i in Finset.range (n - 1), (1 - 1 / (i + 2)^2)

theorem Q_value_for_10 : Q 10 = 1 / 50 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_value_for_10_l684_68480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_count_solution_l684_68472

/-- Represents the number of machines of each type -/
structure MachineCount where
  R : ℕ
  S : ℕ
  T : ℕ

/-- Represents the time taken by each machine type to complete the job -/
structure MachineTime where
  R : ℕ
  S : ℕ
  T : ℕ

/-- The theorem to prove -/
theorem machine_count_solution (job_time : ℕ) (machine_time : MachineTime) (ratio : MachineCount) :
  let total_rate := (ratio.R : ℚ) / machine_time.R + (ratio.S : ℚ) / machine_time.S + (ratio.T : ℚ) / machine_time.T
  let k : ℚ := 1 / (job_time * total_rate)
  machine_time.R = 36 ∧ 
  machine_time.S = 24 ∧ 
  machine_time.T = 18 ∧
  job_time = 8 ∧
  ratio.R = 2 ∧
  ratio.S = 3 ∧
  ratio.T = 4 →
  (k * ratio.R).floor = 58 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_machine_count_solution_l684_68472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_l684_68450

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x / (x - 2)

-- State the theorem
theorem max_value_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1 : ℝ) 1 ∧ 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ f c) ∧
  f c = 1/3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_l684_68450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_completion_time_l684_68456

theorem game_completion_time (initial_hours_per_day : ℝ) (initial_days : ℕ) 
  (initial_completion_percentage : ℝ) (additional_days : ℕ) 
  (increased_hours : ℝ) : 
  initial_hours_per_day = 4 →
  initial_days = 14 →
  initial_completion_percentage = 0.4 →
  additional_days = 12 →
  increased_hours = 3 →
  (initial_hours_per_day * initial_days) / initial_completion_percentage * 
    (1 - initial_completion_percentage) = 
    (initial_hours_per_day + increased_hours) * additional_days := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_completion_time_l684_68456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_equation_C_alpha_range_l684_68476

noncomputable section

-- Define the curve C in polar coordinates
def curve_C (θ : ℝ) : ℝ := 4 * Real.sin θ / (Real.cos θ)^2

-- Define the line l in parametric form
def line_l (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, 1 - t * Real.sin α)

-- Theorem for the rectangular equation of curve C
theorem rectangular_equation_C :
  ∀ x y : ℝ, (∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ↔ x^2 = 4*y := by
  sorry

-- Theorem for the range of α
theorem alpha_range (α : ℝ) :
  (0 < α ∧ α < Real.pi) →
  (∃ t₁ t₂ : ℝ, (line_l t₁ α).1^2 = 4*(line_l t₁ α).2 ∧
                   (line_l t₂ α).1^2 = 4*(line_l t₂ α).2 ∧
                   (t₁ - t₂)^2 * (Real.cos α)^2 ≥ 16) →
  (Real.pi/3 < α ∧ α ≤ Real.pi/2) ∨ (Real.pi/2 < α ∧ α ≤ 2*Real.pi/3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_equation_C_alpha_range_l684_68476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_units_digit_l684_68444

theorem greatest_difference_units_digit (n : ℕ) : 
  (n ≥ 720 ∧ n < 730 ∧ n % 5 = 0) → 
  (∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ n % 10 = x ∧ n % 10 = y ∧ (x ≥ y → x - y ≤ 5) ∧ (y > x → y - x ≤ 5)) ∧
  (∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n % 10 = a ∧ n % 10 = b ∧ (a ≥ b → a - b = 5) ∧ (b > a → b - a = 5)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_difference_units_digit_l684_68444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_distribution_difference_l684_68496

/-- Given a total profit and distribution proportions, calculate the difference between two shares -/
def share_difference (total_profit : ℚ) (proportions : List ℚ) (index1 index2 : ℕ) : ℚ :=
  let total_parts := proportions.sum
  let part_value := total_profit / total_parts
  match proportions.get? index2, proportions.get? index1 with
  | some v2, some v1 => part_value * (v2 - v1)
  | _, _ => 0

/-- Theorem stating the difference between C's and B's shares given the problem conditions -/
theorem profit_distribution_difference :
  let total_profit : ℚ := 20000
  let proportions : List ℚ := [2, 3, 5, 4]
  share_difference total_profit proportions 1 2 = 2857.14 := by
  sorry

#eval share_difference 20000 [2, 3, 5, 4] 1 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_distribution_difference_l684_68496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_sqrt3_l684_68430

noncomputable def f (n : ℕ+) : ℝ := Real.tan (n.val * Real.pi / 3)

theorem sum_of_f_equals_sqrt3 :
  (Finset.range 2017).sum (fun i => f ⟨i + 1, Nat.succ_pos i⟩) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_sqrt3_l684_68430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_on_line_l684_68423

theorem sin_2alpha_on_line (α : ℝ) : 
  (Real.sin α = -2 * Real.cos α) → Real.sin (2 * α) = -4/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_on_line_l684_68423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_1998_l684_68479

noncomputable def mySequence (a b : ℝ) : ℕ → ℝ
  | 0 => a
  | 1 => b
  | (n + 2) => (1 + mySequence a b (n + 1)) / mySequence a b n

theorem mySequence_1998 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  mySequence a b 1998 = (a + b + 1) / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mySequence_1998_l684_68479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_g_8000_l684_68487

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 
  if 2 ≤ x ∧ x ≤ 4 then
    2 - |x - 3|
  else
    0 -- Placeholder for values outside [2, 4]

-- State the properties of g
axiom g_scale (x : ℝ) (h : x > 0) : g (4 * x) = 4 * g x

-- State the theorem
theorem smallest_x_equals_g_8000 : 
  ∃ (x : ℝ), x > 0 ∧ g x = g 8000 ∧ ∀ (y : ℝ), y > 0 ∧ g y = g 8000 → x ≤ y ∧ x = 3016 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_g_8000_l684_68487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_distance_set_l684_68405

/-- Defines what it means for a set to be a circle with a given center -/
def IsCircle (C : Set (ℝ × ℝ)) (P : ℝ × ℝ) :=
  ∃ r > 0, C = {X : ℝ × ℝ | dist X P = r}

theorem circle_point_distance_set 
  (C : Set (ℝ × ℝ)) 
  (P B : ℝ × ℝ) 
  (h1 : IsCircle C P)
  (h2 : B ∈ C) :
  {A : ℝ × ℝ | ∀ X ∈ C, dist A B ≤ dist A X} = 
  {A : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ A = P + t • (B - P)} :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_distance_set_l684_68405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concurrency_of_tangent_lines_l684_68413

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a plane --/
@[ext] structure Point where
  x : ℝ
  y : ℝ

/-- Three circles externally tangent to each other pairwise --/
structure ExternallyTangentCircles where
  Γ₁ : Circle
  Γ₂ : Circle
  Γ₃ : Circle
  are_externally_tangent : Bool

/-- A circle internally tangent to three other circles --/
structure InternallyTangentCircle where
  Γ : Circle
  Γ₁ : Circle
  Γ₂ : Circle
  Γ₃ : Circle
  is_internally_tangent : Bool

/-- Tangency points between circles --/
structure TangencyPoints where
  P : Point  -- tangency point of Γ₂ and Γ₃
  Q : Point  -- tangency point of Γ₃ and Γ₁
  R : Point  -- tangency point of Γ₁ and Γ₂

/-- Internal tangency points with the larger circle --/
structure InternalTangencyPoints where
  A : Point  -- internal tangency point of Γ and Γ₁
  B : Point  -- internal tangency point of Γ and Γ₂
  C : Point  -- internal tangency point of Γ and Γ₃

/-- Three lines are concurrent if they intersect at a single point --/
def are_concurrent (l₁ l₂ l₃ : Point → Point → Prop) : Prop :=
  ∃ (X : Point), l₁ X X ∧ l₂ X X ∧ l₃ X X

/-- The main theorem --/
theorem concurrency_of_tangent_lines 
  (etc : ExternallyTangentCircles) 
  (itc : InternallyTangentCircle) 
  (tp : TangencyPoints) 
  (itp : InternalTangencyPoints) :
  are_concurrent 
    (λ X Y ↦ X = itp.A ∨ X = tp.P ∨ Y = itp.A ∨ Y = tp.P)
    (λ X Y ↦ X = itp.B ∨ X = tp.Q ∨ Y = itp.B ∨ Y = tp.Q)
    (λ X Y ↦ X = itp.C ∨ X = tp.R ∨ Y = itp.C ∨ Y = tp.R) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concurrency_of_tangent_lines_l684_68413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rate_is_17_l684_68488

/-- Represents the speed of a boat in different conditions -/
structure BoatSpeed where
  withStream : ℚ
  againstStream : ℚ

/-- Calculates the man's rowing rate in still water given his speeds with and against the stream -/
def manRate (speed : BoatSpeed) : ℚ :=
  (speed.withStream + speed.againstStream) / 2

/-- Theorem stating that given the specific speeds, the man's rate is 17 km/h -/
theorem man_rate_is_17 (speed : BoatSpeed) 
  (h1 : speed.withStream = 24) 
  (h2 : speed.againstStream = 10) : 
  manRate speed = 17 := by
  -- Unfold the definition of manRate
  unfold manRate
  -- Rewrite using the hypotheses
  rw [h1, h2]
  -- Simplify the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rate_is_17_l684_68488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_unrepresentable_number_l684_68494

theorem existence_of_unrepresentable_number 
  (a b c d : ℤ) 
  (h1 : 0 ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ a) (h5 : a > 14) :
  ∃ n : ℕ, n ≥ 1 ∧ n ≤ 29 ∧ 
    ∀ x y z : ℤ, (n : ℤ) ≠ x * (a * x + b) + y * (a * y + c) + z * (a * z + d) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_unrepresentable_number_l684_68494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_condition_g_inequality_condition_l684_68412

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - a*x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2 * Real.log ((a*x + 2) / (6 * Real.sqrt x))

-- Theorem for part (I)
theorem f_increasing_condition (a : ℝ) :
  (∀ x > 0, Monotone (f a)) ↔ a ≤ 2 * Real.sqrt 2 :=
by sorry

-- Theorem for part (II)
theorem g_inequality_condition (k : ℝ) :
  (∀ a ∈ Set.Ioo 2 4, ∃ x ∈ Set.Icc (3/2) 2, g a x > k * (4 - a^2)) ↔ 
  k ∈ Set.Ici (1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_condition_g_inequality_condition_l684_68412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l684_68497

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/4) : Real.cos (2*θ) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l684_68497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_solution_set_l684_68448

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^x - 2^(-x)) / (2^x + 2^(-x))

-- Theorem for the parity of f
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

-- Theorem for the solution set of the inequality
theorem solution_set (x : ℝ) : 
  (3/5 ≤ f x ∧ f x ≤ 15/17) ↔ (1 ≤ x ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_solution_set_l684_68448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_rotation_eq_neg_pi_third_l684_68427

/-- The number of degrees in a full rotation of a clock -/
noncomputable def full_rotation : ℝ := 360

/-- The number of minutes in a full rotation of a clock -/
noncomputable def minutes_per_rotation : ℝ := 60

/-- The number of minutes the clock is moved forward -/
noncomputable def minutes_moved : ℝ := 10

/-- The conversion factor from degrees to radians -/
noncomputable def deg_to_rad : ℝ := Real.pi / 180

/-- The angle in radians that the minute hand rotates when moved forward by 10 minutes -/
noncomputable def minute_hand_rotation : ℝ := -(minutes_moved / minutes_per_rotation * full_rotation * deg_to_rad)

theorem minute_hand_rotation_eq_neg_pi_third :
  minute_hand_rotation = -Real.pi / 3 := by
  unfold minute_hand_rotation
  unfold minutes_moved
  unfold minutes_per_rotation
  unfold full_rotation
  unfold deg_to_rad
  simp [Real.pi]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_rotation_eq_neg_pi_third_l684_68427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_portion_area_l684_68446

/-- The area of the portion of the circle x^2 + 6x + y^2 = 50 that lies below the x-axis
    and to the left of the line y = x - 3 -/
theorem circle_portion_area : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + 6*x + y^2 = 50}
  let below_x_axis := {(x, y) : ℝ × ℝ | y ≤ 0}
  let left_of_line := {(x, y) : ℝ × ℝ | y < x - 3}
  let portion := circle ∩ below_x_axis ∩ left_of_line
  ∃ A : Set (ℝ × ℝ), MeasurableSet A ∧ A = portion ∧ MeasureTheory.volume A = 59*π/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_portion_area_l684_68446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_C_l684_68433

theorem triangle_tangent_C (A B C : ℝ) :
  Real.cos A = 4/5 →
  Real.tan (A - B) = -1/2 →
  Real.tan C = 11/2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_C_l684_68433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_side_length_l684_68425

/-- A regular hexagon with the distance between parallel sides equal to 24 inches has side length 16√3 inches. -/
theorem regular_hexagon_side_length (h : Real) (parallel_sides_distance : Real) (side_length : Real) : 
  parallel_sides_distance = 24 → side_length = h → side_length = 16 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_hexagon_side_length_l684_68425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l684_68493

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the squared distance between two points -/
noncomputable def squaredDistance (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- Calculate the centroid of a triangle -/
noncomputable def centroid (a b c : Point) : Point where
  x := (a.x + b.x + c.x) / 3
  y := (a.y + b.y + c.y) / 3

theorem triangle_centroid_property (a b c : Point) :
  let g := centroid a b c
  (squaredDistance g a + squaredDistance g b + squaredDistance g c = 72) →
  (squaredDistance a b + squaredDistance a c + squaredDistance b c = 216) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_property_l684_68493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l684_68402

def line (x y : ℝ) : ℝ := x + y - 4

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 8

noncomputable def distance_point_to_line (x₀ y₀ : ℝ) : ℝ := 
  |line x₀ y₀| / Real.sqrt 2

theorem circle_tangent_to_line : 
  (∀ x y : ℝ, circle_eq x y → (x = 0 ∧ y = 0 ∨ line x y ≠ 0)) ∧
  distance_point_to_line 0 0 = Real.sqrt 8 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l684_68402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_9_6654_l684_68462

noncomputable def round_to_two_decimal_places (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

noncomputable def round_to_nearest_integer (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem rounding_9_6654 :
  round_to_two_decimal_places 9.6654 = 9.67 ∧
  round_to_nearest_integer 9.6654 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rounding_9_6654_l684_68462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_souvenir_shop_pricing_theorem_l684_68409

/-- Represents the pricing and sales model of a souvenir shop -/
structure SouvenirShop where
  wholesalePrice : ℚ
  baseRetailPrice : ℚ
  baseSalesVolume : ℚ
  salesDecreaseRate : ℚ
  maxPriceMultiplier : ℚ

/-- Calculates the sales volume for a given price -/
def salesVolume (shop : SouvenirShop) (price : ℚ) : ℚ :=
  shop.baseSalesVolume - (price - shop.baseRetailPrice) * (shop.salesDecreaseRate * 100)

/-- Calculates the daily profit for a given price -/
def dailyProfit (shop : SouvenirShop) (price : ℚ) : ℚ :=
  (price - shop.wholesalePrice) * salesVolume shop price

/-- The souvenir shop model based on the given conditions -/
def ourShop : SouvenirShop := {
  wholesalePrice := 2
  baseRetailPrice := 3
  baseSalesVolume := 500
  salesDecreaseRate := 10 / (1/10)
  maxPriceMultiplier := 5/2
}

theorem souvenir_shop_pricing_theorem :
  (salesVolume ourShop (7/2) = 450) ∧
  (∃ price, price ≤ ourShop.wholesalePrice * ourShop.maxPriceMultiplier ∧
            dailyProfit ourShop price = 800 ∧
            price = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_souvenir_shop_pricing_theorem_l684_68409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l684_68474

noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 1)
def b : ℝ × ℝ := (1, 1)

theorem projection_vector : 
  let proj := ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b
  proj = ((Real.sqrt 3 + 1) / 2, (Real.sqrt 3 + 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l684_68474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_of_exponents_1023_l684_68457

theorem least_sum_of_exponents_1023 : 
  let binary_representation : ℕ → List (ℕ × ℕ) := fun n => 
    (Nat.digits 2 n).enum.filter (fun p => p.2 = 1)
  let sum_of_exponents : ℕ → ℕ := fun n => 
    (binary_representation n).map Prod.fst |>.sum
  (sum_of_exponents 1023 = 45) ∧ 
  (∀ exponents : List ℕ, 
    exponents.length ≥ 3 → 
    (exponents.map (fun i => 2^i)).sum = 1023 → 
    exponents.sum ≥ 45) :=
by sorry

#check least_sum_of_exponents_1023

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_of_exponents_1023_l684_68457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_quadpyramid_l684_68498

/-- A quadrilateral pyramid with a rhombus base -/
structure QuadPyramid where
  -- Base rhombus side length
  a : ℝ
  -- Apex to vertex M distance
  sm : ℝ
  -- Apex to vertex N distance
  sn : ℝ
  -- Apex to vertex K distance
  sk : ℝ
  -- Apex to vertex L distance
  sl : ℝ
  -- Acute angle of the rhombus
  θ : ℝ
  -- Constraints
  pos_a : 0 < a
  pos_sm : 0 < sm
  pos_sn : 0 < sn
  pos_sk : 0 < sk
  pos_sl : 0 < sl
  angle_constraint : 0 < θ ∧ θ < π / 2

/-- Volume of the quadrilateral pyramid -/
noncomputable def volume (p : QuadPyramid) : ℝ :=
  (1 / 3) * p.a^2 * Real.sin p.θ * Real.sqrt (p.sm^2 - (p.a^2 / 4))

/-- Theorem stating the maximum volume and corresponding edge lengths -/
theorem max_volume_quadpyramid :
  ∃ (p : QuadPyramid),
    p.a = 4 ∧
    p.sm = 2 ∧
    p.sn = 4 ∧
    p.θ = π / 3 ∧
    p.sk = 3 * Real.sqrt 2 ∧
    p.sl = Real.sqrt 46 ∧
    ∀ (q : QuadPyramid),
      q.a = 4 ∧ q.sm = 2 ∧ q.sn = 4 ∧ q.θ = π / 3 →
      volume p ≥ volume q ∧
      volume p = 4 * Real.sqrt 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_quadpyramid_l684_68498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_zeros_l684_68442

noncomputable def f (x : ℝ) : ℝ :=
  if x = 1 then 1 else 1 / |x - 1|

noncomputable def h (b : ℝ) (x : ℝ) : ℝ :=
  (f x)^2 + b * (f x) + 1/2

theorem sum_of_squares_of_zeros (b : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
  x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
  x₄ ≠ x₅ ∧
  h b x₁ = 0 ∧ h b x₂ = 0 ∧ h b x₃ = 0 ∧ h b x₄ = 0 ∧ h b x₅ = 0 →
  x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 = 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_zeros_l684_68442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_cylindrical_tank_l684_68475

/-- The volume of a quarter-sphere with radius r -/
noncomputable def quarter_sphere_volume (r : ℝ) : ℝ := (1/3) * Real.pi * r^3

/-- The volume of a cylinder with radius r and height h -/
noncomputable def cylinder_volume (r h : ℝ) : ℝ := Real.pi * r^2 * h

/-- The number of trips needed to fill a cylinder with a quarter-sphere bucket -/
noncomputable def trips_needed (cylinder_radius cylinder_height bucket_radius : ℝ) : ℕ :=
  Int.toNat ⌈(cylinder_volume cylinder_radius cylinder_height) / (quarter_sphere_volume bucket_radius)⌉

theorem fill_cylindrical_tank : trips_needed 8 20 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_cylindrical_tank_l684_68475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_integer_not_fractional_l684_68469

theorem root_of_integer_not_fractional (A n : ℕ+) (x : ℚ) :
  x ^ (n : ℕ) = (A : ℚ) → ∃ k : ℤ, x = k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_integer_not_fractional_l684_68469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_and_a_range_l684_68440

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 9/2 * x^2 + 6*x - a

noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 9*x + 6

theorem max_m_value_and_a_range (a : ℝ) :
  (∃ m : ℝ, (∀ x : ℝ, f' x ≥ m) ∧ 
   ¬∃ m' : ℝ, m' > m ∧ ∀ x : ℝ, f' x ≥ m') ∧
  (∃! x : ℝ, f a x = 0) ↔ 
  (a < 2 ∨ a > 5/2) := by
  sorry

#check max_m_value_and_a_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_value_and_a_range_l684_68440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_neg_half_implies_fraction_four_thirds_l684_68470

theorem tan_alpha_neg_half_implies_fraction_four_thirds (α : ℝ) :
  Real.tan α = -1/2 →
  (2 * Real.sin α * Real.cos α) / (Real.sin α^2 - Real.cos α^2) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_neg_half_implies_fraction_four_thirds_l684_68470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_excursion_l684_68453

theorem school_excursion :
  ∃ (students_per_class : ℕ),
    let num_classes : ℕ := 8
    let theater_percentage : ℚ := 115 / 100
    let max_students : ℕ := 520
    let min_museum_students : ℕ := 230
    let museum_percentage : ℚ := 100 / 100
    students_per_class > min_museum_students / num_classes ∧
    students_per_class * num_classes * (museum_percentage + theater_percentage) ≤ max_students ∧
    students_per_class * num_classes * (museum_percentage + theater_percentage) = 516 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_excursion_l684_68453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_two_l684_68436

theorem tan_alpha_two (α : ℝ) 
  (h1 : Real.tan α = 2) 
  (h2 : α ∈ Set.Ioo π (3*π/2)) : 
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 8 ∧ 
  Real.cos α = -Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_two_l684_68436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_theorem_l684_68417

/-- Given three points A, B, and C in 3D space, we define them as follows -/
def A : Fin 3 → ℝ := ![1, 5, -2]
def B : Fin 3 → ℝ := ![2, 4, 1]
def C (p q : ℝ) : Fin 3 → ℝ := ![p, 3, q + 2]

/-- Define a function to check if three points are collinear -/
def collinear (A B C : Fin 3 → ℝ) : Prop :=
  ∃ t : ℝ, ∀ i : Fin 3, C i - A i = t * (B i - A i)

/-- Theorem: If A, B, and C are collinear, then p = 3 and q = 2 -/
theorem collinear_points_theorem (p q : ℝ) :
  collinear A B (C p q) → p = 3 ∧ q = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_theorem_l684_68417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_to_circle_l684_68481

-- Define the line C1
def C1 (x y : ℝ) : Prop := x + y + 4 = 0

-- Define the circle C2
def C2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 8

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_line_to_circle :
  ∀ (x1 y1 x2 y2 : ℝ),
    C1 x1 y1 →
    C2 x2 y2 →
    ∃ (x3 y3 : ℝ),
      C2 x3 y3 ∧
      distance x1 y1 x3 y3 ≤ distance x1 y1 x2 y2 ∧
      distance x1 y1 x3 y3 = 2 * Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_line_to_circle_l684_68481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_6_value_l684_68465

noncomputable def S (m : ℕ) (x : ℝ) : ℝ := x^m + 1/x^m

theorem S_6_value (x : ℝ) (h : x + 1/x = 4) : S 6 x = 2700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_6_value_l684_68465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l684_68492

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line segment
structure LineSegment where
  start : Point2D
  end' : Point2D

-- Function to calculate the length of a line segment
noncomputable def length (segment : LineSegment) : ℝ :=
  Real.sqrt ((segment.end'.x - segment.start.x)^2 + (segment.end'.y - segment.start.y)^2)

-- Function to check if a line segment is parallel to the y-axis
def parallelToYAxis (segment : LineSegment) : Prop :=
  segment.start.x = segment.end'.x

-- Theorem statement
theorem point_b_coordinates (A : Point2D) (B : Point2D) (AB : LineSegment) 
  (h1 : A.x = 2 ∧ A.y = -1)
  (h2 : AB.start = A ∧ AB.end' = B)
  (h3 : parallelToYAxis AB)
  (h4 : length AB = 3) :
  (B.x = 2 ∧ B.y = 2) ∨ (B.x = 2 ∧ B.y = -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l684_68492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l684_68471

def M : Set ℚ := {-2, -1, 0, 1, 2}

def N : Set ℚ := {x : ℚ | x^2 + 2*x - 3 ≤ 0}

theorem intersection_M_N : M ∩ N = {-2, -1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l684_68471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l684_68468

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 12 * x - 5 * y = 3

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 16 = 0

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- State that A and B satisfy both line and circle equations
axiom A_on_line : line_eq A.1 A.2
axiom A_on_circle : circle_eq A.1 A.2
axiom B_on_line : line_eq B.1 B.2
axiom B_on_circle : circle_eq B.1 B.2

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem: The length of AB is 4√2
theorem chord_length : distance A B = 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l684_68468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_max_power_speed_l684_68495

/-- The speed of a sailboat that maximizes wind power -/
theorem sailboat_max_power_speed
  (c s ρ v₀ : ℝ)
  (hc : c > 0)
  (hs : s > 0)
  (hρ : ρ > 0)
  (hv₀ : v₀ > 0) :
  ∃ v_max, v_max = v₀ / 3 ∧
    (∀ v, (c * s * ρ * (v₀ - v)^2 / 2) * v ≤ (c * s * ρ * (v₀ - v_max)^2 / 2) * v_max) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sailboat_max_power_speed_l684_68495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_probability_distribution_expected_value_l684_68490

-- Define the number of students
def total_students : ℕ := 8
def male_students : ℕ := 5
def female_students : ℕ := 3
def selected_students : ℕ := 3

-- Define events A and B as propositions
def event_A : Prop := True  -- Placeholder for "female student A is selected"
def event_B : Prop := True  -- Placeholder for "male student B is selected"

-- Define the random variable X
def X : ℕ → ℝ := sorry

-- Define the probability function
def P : Prop → ℝ := sorry

-- State the theorems to be proved
theorem conditional_probability : P (event_B ∧ event_A) / P event_A = 2 / 7 := by sorry

theorem probability_distribution :
  P (X 0 = 0) = 1 / 56 ∧
  P (X 1 = 1) = 15 / 56 ∧
  P (X 2 = 2) = 15 / 28 ∧
  P (X 3 = 3) = 5 / 28 := by sorry

theorem expected_value :
  (0 : ℝ) * P (X 0 = 0) + 1 * P (X 1 = 1) + 2 * P (X 2 = 2) + 3 * P (X 3 = 3) = 15 / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conditional_probability_probability_distribution_expected_value_l684_68490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_solution_unique_and_valid_l684_68437

/-- Definition of the sequence P -/
def P : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 3
  | n+3 => P (n+2) * P (n+1) - P (n+1)

/-- The type of answer choices -/
inductive Choice
  | A | B | C | D

/-- The type representing a set of answers for the three questions -/
structure AnswerSet where
  q1 : Choice
  q2 : Choice
  q3 : Choice

/-- Predicate to check if an AnswerSet is valid (all choices are different) -/
def isValidAnswerSet (as : AnswerSet) : Prop :=
  as.q1 ≠ as.q2 ∧ as.q1 ≠ as.q3 ∧ as.q2 ≠ as.q3

/-- Predicate to check if an AnswerSet is the correct solution -/
def isCorrectSolution (as : AnswerSet) : Prop :=
  as.q1 = Choice.A ∧ as.q2 = Choice.B ∧ as.q3 = Choice.D

/-- Main theorem: The correct solution is unique and valid -/
theorem correct_solution_unique_and_valid :
  ∃! (as : AnswerSet), isValidAnswerSet as ∧ isCorrectSolution as ∧
    (P 2002 % 3 = 0) ∧ (P 2002 % 7 = 0) ∧ (P 2002 % 9 = 0) := by
  sorry

#eval P 10  -- Added to check if P is correctly defined

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_solution_unique_and_valid_l684_68437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l684_68467

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition a/cos(A) = c/(2 - cos(C)) -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a / Real.cos t.A = t.c / (2 - Real.cos t.C)

/-- The area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  1/2 * t.a * t.b * Real.sin t.C

theorem triangle_properties (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : t.b = 4) 
  (h3 : t.c = 3) 
  (h4 : area t = 3) : 
  (t.a = 2) ∧ (3 * Real.sin t.C + 4 * Real.cos t.C = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l684_68467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l684_68499

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (3*θ) = -117/125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l684_68499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l684_68466

noncomputable section

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

-- Define points A, B, and P
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (3/2, -1)
def P : ℝ × ℝ := (1, -2)

-- Define line segment AB
def AB (x y : ℝ) : Prop := y = 2/3 * x - 2

-- Define the theorem
theorem ellipse_fixed_point :
  ∀ (M N T H : ℝ × ℝ),
    E M.1 M.2 →
    E N.1 N.2 →
    AB T.1 T.2 →
    (∃ k : ℝ, k * (P.1 - M.1) = P.2 - M.2 ∧ k * (P.1 - N.1) = P.2 - N.2) →
    M.2 = T.2 →
    H.1 - T.1 = T.1 - M.1 ∧ H.2 = T.2 →
    ∃ k : ℝ, k * (H.1 - N.1) = H.2 - N.2 ∧ k * H.1 = H.2 + 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l684_68466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_side_length_l684_68473

noncomputable section

open Real

-- Define the function f
def f (x φ : ℝ) : ℝ := sin (2 * x + φ) + 2 * (sin x)^2

-- Define the conditions
def conditions (φ : ℝ) : Prop :=
  (abs φ < Real.pi / 2) ∧
  (f (Real.pi / 6) φ = 3 / 2) ∧
  ∃ (C : ℝ), 0 < C ∧ C < Real.pi / 2 ∧
    (∀ x, f x φ = f (2 * C - x) φ) ∧
    (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
      (1 / 2) * a * b * sin C = 2 * sqrt 3 ∧
      a + b = 6)

-- State the theorem
theorem min_value_and_side_length (φ : ℝ) 
  (h : conditions φ) : 
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x φ ≥ 1 / 2) ∧
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (1 / 2) * a * b * sin (Real.pi / 3) = 2 * sqrt 3 ∧
    a + b = 6 ∧
    c = 2 * sqrt 3) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_and_side_length_l684_68473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_M_l684_68458

def M : ℕ := 2^2 * 3^1 * 7^2

theorem sum_of_divisors_M : (Finset.sum (Finset.filter (· ∣ M) (Finset.range (M + 1))) id) = 1596 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_divisors_M_l684_68458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_business_investment_theorem_l684_68486

/-- Represents the initial investment of person A in rupees -/
def initial_investment (x : ℕ) : Prop := True

/-- Represents the investment duration in months -/
def investment_duration (person : ℕ) (duration : ℕ) : Prop := True

/-- Represents the profit ratio between two investors -/
def profit_ratio (a : ℕ) (b : ℕ) : Prop := True

theorem business_investment_theorem (x : ℕ) :
  initial_investment x →
  investment_duration 1 12 →
  investment_duration 2 4 →
  profit_ratio 2 1 →
  x * 12 = 54000 * 4 * 2 →
  x = 36000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_business_investment_theorem_l684_68486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_and_inequality_l684_68434

/-- The distance from a point (x, y) to the line ax + by + c = 0 -/
noncomputable def distance_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / Real.sqrt (a^2 + b^2)

theorem point_distance_and_inequality (m : ℝ) :
  distance_point_to_line m 3 4 (-3) 1 = 4 ∧
  2 * m + 3 < 3 →
  m = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_distance_and_inequality_l684_68434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_intersects_skew_lines_l684_68435

-- Define the necessary structures
structure Line where

structure Plane where

-- Define the relations
def skew (l1 l2 : Line) : Prop := sorry

def in_plane (l : Line) (p : Plane) : Prop := sorry

def intersects (l1 l2 : Line) : Prop := sorry

def plane_intersection (p1 p2 : Plane) : Line := sorry

-- Theorem statement
theorem intersection_line_intersects_skew_lines 
  (a b c : Line) (α β : Plane) 
  (h1 : skew a b)
  (h2 : in_plane a α)
  (h3 : in_plane b β)
  (h4 : c = plane_intersection α β) :
  intersects c a ∨ intersects c b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_intersects_skew_lines_l684_68435
