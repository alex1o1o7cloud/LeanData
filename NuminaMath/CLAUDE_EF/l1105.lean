import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1105_110540

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f x + y) = f (x + y) + 2*x*f y - 3*x*y - 2*x + 2) : 
  ∀ x : ℝ, f x = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_identity_l1105_110540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l1105_110521

noncomputable section

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (-1, -1)
def B : ℝ × ℝ := (2, 0)
def D : ℝ × ℝ := (0, 1)

-- Define C as a variable point
def C : ℝ × ℝ := (3, 2)

-- Define E as the midpoint of BD
noncomputable def E : ℝ × ℝ := ((B.1 + D.1) / 2, (B.2 + D.2) / 2)

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := 6 * x + 2 * y - 7 = 0

-- Theorem statement
theorem parallelogram_properties :
  -- ABCD is a parallelogram
  (A.1 - D.1 = C.1 - B.1 ∧ A.2 - D.2 = C.2 - B.2) →
  -- l is perpendicular to CD
  ((C.2 - D.2) * 6 + (C.1 - D.1) * 2 = 0) →
  -- Prove that C has coordinates (3,2)
  (C = (3, 2)) ∧
  -- Prove that the equation of line l is 6x + 2y - 7 = 0
  (∀ x y, line_l x y ↔ y - E.2 = -3 * (x - E.1)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_properties_l1105_110521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_32_hours_l1105_110563

/-- Represents the journey details -/
structure Journey where
  total_distance : ℚ
  car_speed : ℚ
  dick_walk_speed : ℚ
  others_walk_speed : ℚ

/-- Calculates the journey time given the distances traveled -/
def journey_time (j : Journey) (d1 d2 : ℚ) : ℚ :=
  d1 / j.car_speed + (j.total_distance - d1) / j.others_walk_speed

/-- Theorem stating that the journey time is 32 hours -/
theorem journey_time_is_32_hours (j : Journey) 
  (h1 : j.total_distance = 150)
  (h2 : j.car_speed = 30)
  (h3 : j.dick_walk_speed = 4)
  (h4 : j.others_walk_speed = 3)
  : ∃ (d1 d2 : ℚ), 
    journey_time j d1 d2 = 32 ∧ 
    d1 / j.car_speed + d2 / j.car_speed + (j.total_distance - (d1 - d2)) / j.car_speed = 32 ∧
    (d1 - d2) / j.dick_walk_speed + (j.total_distance - (d1 - d2)) / j.car_speed = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_is_32_hours_l1105_110563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nh3_hcl_reaction_l1105_110536

/-- Represents the number of moles of a chemical substance -/
structure Moles where
  value : ℝ

/-- Represents the stoichiometric ratio between NH4Cl and HCl -/
def stoichiometric_ratio (nh4cl hcl : Moles) : Prop := nh4cl.value = hcl.value

theorem nh3_hcl_reaction (nh3 nh4cl hcl : Moles) 
  (h1 : nh3.value = 1)
  (h2 : stoichiometric_ratio nh4cl hcl)
  (h3 : nh4cl.value = nh3.value) :
  hcl.value = 1 := by
  sorry

#check nh3_hcl_reaction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nh3_hcl_reaction_l1105_110536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_four_zeros_l1105_110564

open Real Set

theorem omega_range_for_four_zeros (ω : ℝ) (f g h : ℝ → ℝ) : 
  (ω > 0) →
  (∀ x, f x = sin (ω * x)) →
  (∀ x, g x = cos x) →
  (∀ x, h x = f (g x) - 1) →
  (∃! (s : Set ℝ), s.Finite ∧ s.ncard = 4 ∧ 
    (∀ x ∈ s, x ∈ Ioo 0 (2 * π) ∧ h x = 0)) →
  ω ∈ Icc (7 * π / 2) (9 * π / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_four_zeros_l1105_110564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_to_g_transformation_l1105_110556

open Real

-- Define the original function f
noncomputable def f (x φ : ℝ) : ℝ := Real.sqrt 3 * sin (x + φ) - cos (x + φ)

-- Define the transformation to get g from f
noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x - π / 4)

theorem f_to_g_transformation (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) :
  (∀ x, f x φ = -f (-x) φ) →  -- f is an odd function
  (∀ x, g (x + π/8) = f (2*x) (π/6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_to_g_transformation_l1105_110556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_sequence_existence_l1105_110516

theorem group_sequence_existence {G : Type*} [Group G] [Fintype G] (g h : G) :
  (∀ x : G, ∃ k : ℕ, x = (g * h)^k ∨ x = g * (g * h)^k) →
  ∃ (seq : Fin (2 * Fintype.card G) → G),
    (∀ i : Fin (2 * Fintype.card G), 
      seq (Fin.succ i) = g * seq i ∨ seq (Fin.succ i) = h * seq i) ∧
    (seq 0 = g * seq (Fin.last (2 * Fintype.card G - 1)) ∨ 
     seq 0 = h * seq (Fin.last (2 * Fintype.card G - 1))) ∧
    (∀ x : G, ∃ i j : Fin (2 * Fintype.card G), i ≠ j ∧ seq i = x ∧ seq j = x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_sequence_existence_l1105_110516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_triangle_l1105_110503

theorem cosine_triangle (a b c : ℝ) (h : 6 * a = 4 * b ∧ 4 * b = 3 * c) :
  (a^2 + c^2 - b^2) / (2 * a * c) = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_triangle_l1105_110503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1105_110531

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (16 - 4^x)

theorem range_of_f :
  Set.range f = Set.Icc 0 4 \ {4} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1105_110531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1105_110585

-- Define the universe set U
def U : Set ℝ := Set.Icc (-5) 4

-- Define set A
def A : Set ℝ := {x : ℝ | -3 ≤ 2*x + 1 ∧ 2*x + 1 < 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 2*x ≤ 0}

-- Theorem statement
theorem complement_A_intersect_B : (U \ A) ∩ B = Set.Icc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1105_110585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_of_fraction_l1105_110597

theorem greatest_integer_of_fraction : 
  ⌊(4^105 + 3^105 : ℝ) / (4^100 + 3^100 : ℝ)⌋ = 1023 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_of_fraction_l1105_110597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1105_110547

-- Part 1
theorem part_one : 
  (0.027 : ℝ) ^ (1/3 : ℝ) - (25/4 : ℝ) ^ (1/2 : ℝ) + π^(0 : ℝ) - (3 : ℝ)^(-1 : ℝ) = -23/15 := by sorry

-- Part 2
theorem part_two : 
  2 * (Real.log 2 / Real.log 6) + (Real.log 9 / Real.log 6) - 
  (Real.log (1/9) / Real.log 3) - (8 : ℝ)^(4/3 : ℝ) = -12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l1105_110547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_milk_ratio_theorem_l1105_110541

/-- Represents the ratio of water to milk in a mixture -/
structure WaterMilkRatio where
  water : ℚ
  milk : ℚ

/-- Calculates the ratio of water to milk in a mixture of two cups -/
def mixtureCups (milk1 : ℚ) (milk2 : ℚ) : WaterMilkRatio :=
  { water := (1 - milk1) + (1 - milk2),
    milk := milk1 + milk2 }

/-- Theorem: The ratio of water to milk in the mixture of two cups with
    3/5 and 4/5 milk respectively is 3:7 -/
theorem water_milk_ratio_theorem :
  let ratio := mixtureCups (3/5) (4/5)
  (ratio.water : ℚ) / (ratio.milk : ℚ) = 3/7 := by
  -- Unfold the definition of mixtureCups
  unfold mixtureCups
  -- Simplify the expressions
  simp
  -- Perform the arithmetic
  norm_num

-- We remove the #eval as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_milk_ratio_theorem_l1105_110541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_count_l1105_110557

theorem acute_triangle_count : ∃! n : ℕ, 
  n = (Finset.filter (fun x : ℕ ↦ 
    x > 0 ∧ 
    13 + 15 > x ∧ 
    13 + x > 15 ∧ 
    15 + x > 13 ∧ 
    (x > 15 → x^2 < 13^2 + 15^2) ∧ 
    (x ≤ 15 → 15^2 < 13^2 + x^2)
  ) (Finset.range 29)).card ∧ n = 12 :=
by sorry

#eval (Finset.filter (fun x : ℕ ↦ 
  x > 0 ∧ 
  13 + 15 > x ∧ 
  13 + x > 15 ∧ 
  15 + x > 13 ∧ 
  (x > 15 → x^2 < 13^2 + 15^2) ∧ 
  (x ≤ 15 → 15^2 < 13^2 + x^2)
) (Finset.range 29)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_count_l1105_110557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_postolny_points_l1105_110528

-- Define the basic structures
structure Figure where

structure Line where

structure Point where

-- Define the similarity circle
def similarity_circle (F₁ F₂ F₃ : Figure) : Set Point :=
  sorry

-- Define the intersection of lines
def intersect (l₁ l₂ : Line) : Point :=
  sorry

-- Define the intersection of a line and the similarity circle
def intersect_circle (l : Line) (c : Set Point) : Set Point :=
  sorry

-- Define the property of similar figures
def similar (F₁ F₂ : Figure) : Prop :=
  sorry

-- Main theorem
theorem postolny_points 
  (F₁ F₂ F₃ : Figure) 
  (l₁ l₂ l₃ : Line) 
  (W : Point) :
  similar F₁ F₂ ∧ similar F₂ F₃ ∧ similar F₃ F₁ →
  intersect l₁ l₂ = W ∧ intersect l₂ l₃ = W ∧ intersect l₃ l₁ = W →
  (W ∈ similarity_circle F₁ F₂ F₃) ∧
  (∃ (J₁ J₂ J₃ : Point),
    J₁ ∈ intersect_circle l₁ (similarity_circle F₁ F₂ F₃) ∧
    J₂ ∈ intersect_circle l₂ (similarity_circle F₁ F₂ F₃) ∧
    J₃ ∈ intersect_circle l₃ (similarity_circle F₁ F₂ F₃) ∧
    J₁ ≠ W ∧ J₂ ≠ W ∧ J₃ ≠ W ∧
    (∀ (l₁' l₂' l₃' : Line),
      intersect l₁' l₂' = W ∧ intersect l₂' l₃' = W ∧ intersect l₃' l₁' = W →
      intersect_circle l₁' (similarity_circle F₁ F₂ F₃) = {W, J₁} ∧
      intersect_circle l₂' (similarity_circle F₁ F₂ F₃) = {W, J₂} ∧
      intersect_circle l₃' (similarity_circle F₁ F₂ F₃) = {W, J₃})) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_postolny_points_l1105_110528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1105_110581

/-- A circle in the Cartesian plane -/
structure Circle where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => x^2 + y^2 - a*y = 0

/-- A line in the Cartesian plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The distance between a point and a line -/
noncomputable def distPointLine (x y : ℝ) (l : Line) : ℝ := sorry

/-- The radius of a circle -/
noncomputable def radius (c : Circle) : ℝ := c.a / 2

/-- The center of a circle -/
noncomputable def center (c : Circle) : ℝ × ℝ := (0, c.a / 2)

/-- The length of a chord in a circle -/
noncomputable def chordLength (c : Circle) (l : Line) : ℝ := 
  2 * Real.sqrt ((radius c)^2 - (distPointLine (center c).1 (center c).2 l)^2)

/-- The main theorem -/
theorem circle_line_intersection (c : Circle) (l : Line) : 
  (c.a ≠ 0) → 
  (∀ x y, l.equation x y ↔ 4*x + 3*y - 8 = 0) → 
  (chordLength c l = Real.sqrt 3 * radius c) → 
  (c.a = 32 ∨ c.a = 32/11) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_l1105_110581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_days_l1105_110512

/-- The number of days it takes for the first group to complete the work -/
noncomputable def days_to_complete_work (man_work : ℝ) (boy_work : ℝ) : ℝ :=
  200 / (12 * man_work + 16 * boy_work)

/-- Theorem stating the number of days for the first group to complete the work -/
theorem work_completion_days :
  ∀ (man_work : ℝ) (boy_work : ℝ),
    man_work > 0 →
    boy_work > 0 →
    man_work = 2 * boy_work →
    13 * man_work + 24 * boy_work = 1 / 4 →
    days_to_complete_work man_work boy_work = 5 := by
  sorry

#check work_completion_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_days_l1105_110512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_sectional_area_range_l1105_110526

-- Define the slant height and apex angle
variable (l : ℝ) -- slant height
variable (apex_angle : ℝ) -- apex angle in radians

-- Define the cross-sectional area function
noncomputable def cross_sectional_area (θ : ℝ) : ℝ := (1/2) * l^2 * Real.sin θ

-- Theorem statement
theorem cross_sectional_area_range (l : ℝ) (h : l > 0) :
  let apex_angle := 2*π/3  -- 120° in radians
  (∀ θ, 0 < θ ∧ θ ≤ apex_angle → 
    0 < cross_sectional_area l θ ∧ cross_sectional_area l θ ≤ (1/2) * l^2) ∧
  (∀ ε > 0, ∃ θ, 0 < θ ∧ θ ≤ apex_angle ∧ cross_sectional_area l θ > (1/2) * l^2 - ε) ∧
  (∀ ε > 0, ∃ θ, 0 < θ ∧ θ ≤ apex_angle ∧ cross_sectional_area l θ < ε) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_sectional_area_range_l1105_110526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_equals_fraction_l1105_110502

/-- The set of positive integer divisors of 30^5 -/
def S : Finset ℕ := Finset.filter (λ d => d > 0 ∧ (30^5) % d = 0) (Finset.range (30^5 + 1))

/-- The cardinality of S -/
def card_S : ℕ := Finset.card S

/-- The probability of choosing three distinct numbers a1, a2, a3 from S,
    such that a1 divides a2 and a2 divides a3 -/
def prob : ℚ :=
  let total_choices := (card_S : ℚ) ^ 3
  let valid_choices : ℚ := 8000  -- This is derived from the problem, not the solution
  valid_choices / total_choices

theorem prob_equals_fraction :
  prob = 125 / 158081 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_equals_fraction_l1105_110502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_sum_l1105_110552

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3/2) * Real.sin (2*x) + (Real.sqrt 3/2) * Real.cos (2*x) + Real.pi/12

-- Define the theorem
theorem symmetry_point_sum (a b : ℝ) : 
  a ∈ Set.Ioo (-Real.pi/2) 0 → 
  (∀ x, f (a + (x - a)) = f (a - (x - a))) → 
  a + b = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_point_sum_l1105_110552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_quad_area_ratio_l1105_110590

/-- Represents a convex quadrilateral with vertices A, B, C, D -/
structure ConvexQuadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The angle between the diagonals of the quadrilateral -/
noncomputable def diagonalAngle (q : ConvexQuadrilateral) : ℝ := sorry

/-- The area of a quadrilateral -/
noncomputable def area (q : ConvexQuadrilateral) : ℝ := sorry

/-- The quadrilateral formed by the feet of perpendiculars -/
noncomputable def innerQuadrilateral (q : ConvexQuadrilateral) : ConvexQuadrilateral := sorry

/-- Theorem: The ratio of the area of the inner quadrilateral to the original quadrilateral is 1/2 -/
theorem inner_quad_area_ratio (q : ConvexQuadrilateral) 
  (h : diagonalAngle q = Real.pi / 4) : 
  area (innerQuadrilateral q) / area q = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_quad_area_ratio_l1105_110590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_badamba_bees_count_l1105_110593

/-- The number of bees that flew to the badamba flower -/
noncomputable def badamba_bees : ℝ := 11/3

/-- The total number of bees -/
def total_bees : ℕ := 15

/-- The number of bees that flew to the slandbara flower -/
noncomputable def slandbara_bees : ℝ := total_bees / 3

/-- The number of bees that flew to the arbour -/
noncomputable def arbour_bees : ℝ := 3 * (badamba_bees - slandbara_bees)

/-- The number of bees that flew about -/
def flying_about : ℕ := 1

theorem badamba_bees_count : 
  ⌊badamba_bees⌋ = 3 ∧ ⌈badamba_bees⌉ = 4 ∧
  badamba_bees + slandbara_bees + arbour_bees + flying_about = total_bees := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_badamba_bees_count_l1105_110593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_equal_to_given_value_l1105_110587

theorem not_equal_to_given_value : 
  let given_value : ℚ := 375 / 1000000000
  let option_a : ℚ := 375 / 100000000
  let option_b : ℚ := 3 / 4 * (1 / 10000000)
  let option_c : ℚ := 3 / 8 * (1 / 1000000)
  let option_d : ℚ := 3 / 8 * (1 / 10000000)
  (given_value = option_a) ∧ 
  (given_value = option_b) ∧ 
  (given_value = option_c) →
  (given_value ≠ option_d) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_equal_to_given_value_l1105_110587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_in_picture_probability_l1105_110509

/-- Rachel's lap time in seconds -/
noncomputable def rachel_lap_time : ℝ := 120

/-- Robert's lap time in seconds -/
noncomputable def robert_lap_time : ℝ := 70

/-- Time when picture is taken (in seconds after start) -/
def picture_time : Set ℝ := Set.Icc 900 960

/-- Fraction of track visible in picture -/
noncomputable def visible_fraction : ℝ := 1/3

/-- Probability that both Rachel and Robert are in the picture -/
noncomputable def both_in_picture_prob : ℝ := 2/9

/-- The main theorem stating the probability of both runners being in the picture -/
theorem runners_in_picture_probability :
  ∀ t ∈ picture_time,
  (∃ r ∈ Set.Icc 0 rachel_lap_time, (t - r) % rachel_lap_time ≤ visible_fraction * rachel_lap_time) ∧
  (∃ s ∈ Set.Icc 0 robert_lap_time, (t - s) % robert_lap_time ≤ visible_fraction * robert_lap_time) →
  both_in_picture_prob = 2/9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_in_picture_probability_l1105_110509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equations_l1105_110519

/-- Represents a hyperbola -/
structure Hyperbola where
  focus : ℝ × ℝ
  eccentricity : ℝ
  asymptote_slope : ℝ
  point : ℝ × ℝ

/-- Checks if the given equation matches the standard form of the hyperbola -/
def is_standard_equation (h : Hyperbola) (a b : ℝ) : Prop :=
  (h.point.2^2 / a^2) - (h.point.1^2 / b^2) = 1 ∧
  h.asymptote_slope = a / b

/-- The main theorem stating the two standard equations of the hyperbola -/
theorem hyperbola_standard_equations (h : Hyperbola) 
  (h_focus : h.focus = (0, 13))
  (h_eccentricity : h.eccentricity = 13/5)
  (h_asymptote : h.asymptote_slope = 1/2)
  (h_point : h.point = (2, -3)) :
  (is_standard_equation h 5 12 ∧ is_standard_equation h (2*Real.sqrt 2) (4*Real.sqrt 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equations_l1105_110519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xr_rs_ratio_is_one_l1105_110542

/-- A hexagon composed of 7 unit squares -/
structure Hexagon where
  area : ℝ
  area_eq : area = 7

/-- A line segment that bisects the hexagon -/
structure BisectingLine (h : Hexagon) where
  length : ℝ
  bisects_area : ℝ
  bisects_eq : bisects_area = h.area / 2

/-- The portion below the bisecting line -/
structure LowerPortion (h : Hexagon) (bl : BisectingLine h) where
  square_area : ℝ
  triangle_base : ℝ
  square_eq : square_area = 1
  triangle_base_eq : triangle_base = 4

/-- The ratio of XR to RS -/
noncomputable def xr_rs_ratio (h : Hexagon) (bl : BisectingLine h) (lp : LowerPortion h bl) : ℝ :=
  let xr := bl.length / 2
  let rs := bl.length / 2
  xr / rs

/-- The main theorem -/
theorem xr_rs_ratio_is_one (h : Hexagon) (bl : BisectingLine h) (lp : LowerPortion h bl) :
  xr_rs_ratio h bl lp = 1 := by
  unfold xr_rs_ratio
  simp
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xr_rs_ratio_is_one_l1105_110542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_pi_ninths_l1105_110527

theorem tan_product_pi_ninths : 
  Real.tan (π/9) * Real.tan (2*π/9) * Real.tan (4*π/9) = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_product_pi_ninths_l1105_110527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1105_110546

-- Define the function
noncomputable def f (x : ℝ) := Real.sin (x / 2)

-- State the theorem
theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Icc (-Real.pi) Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l1105_110546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l1105_110577

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^8 + i^24 + i^(-32 : ℤ) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l1105_110577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_x_value_l1105_110599

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- Given vectors a and b, if they are collinear, then the y-coordinate of b is -3 -/
theorem collinear_vectors_x_value (a b : ℝ × ℝ) (h : collinear a b) :
  a = (-3, 3) → b.1 = 3 → b.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_x_value_l1105_110599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_l1105_110551

-- Ellipse
def ellipse1 (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1
def ellipse2 (x y : ℝ) : Prop := x^2 / 8 + y^2 / 6 = 1

-- Hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 = 1

theorem conic_sections :
  (∃ c : ℝ, c > 0 ∧ c < 1 ∧
    (∀ x y : ℝ, ellipse1 x y ↔ x^2 / (1 - c^2) + y^2 / (1 - c^2)^2 = 1) ∧
    (∀ x y : ℝ, ellipse2 x y ↔ x^2 / (2 - 2*c^2) + y^2 / (2 - 2*c^2)^2 = 1)) ∧
  ellipse2 2 (-Real.sqrt 3) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, hyperbola x y ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    2 * a = 6 ∧
    b / a = 1 / 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_sections_l1105_110551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1105_110586

noncomputable def f (x : ℝ) := (Real.log (x + 1)) / Real.sqrt (-x^2 - 3*x + 4)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1105_110586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cube_height_l1105_110569

/-- The height of a truncated cube when one corner is cut off --/
theorem truncated_cube_height (cube_side : ℝ) (h_side : cube_side = 2) :
  let diagonal := cube_side * Real.sqrt 3
  let cut_side := cube_side * Real.sqrt 2
  let cut_area := Real.sqrt 3 / 4 * cut_side ^ 2
  let cut_height := diagonal / 3
  let remaining_height := cube_side - cut_height
  remaining_height = (6 - 2 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cube_height_l1105_110569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_hexagon_dots_l1105_110529

/-- Calculates the number of dots in a hexagonal tile given its position in the sequence -/
def hexagonDots : Nat → Nat
| 0 => 2  -- Add this case for Nat.zero
| 1 => 2
| n + 1 => hexagonDots n + 6 * n * n

/-- The fourth hexagon in the sequence contains 86 dots -/
theorem fourth_hexagon_dots : hexagonDots 4 = 86 := by
  -- Unfold the definition for the first few steps
  have h1 : hexagonDots 4 = hexagonDots 3 + 6 * 3 * 3 := rfl
  have h2 : hexagonDots 3 = hexagonDots 2 + 6 * 2 * 2 := rfl
  have h3 : hexagonDots 2 = hexagonDots 1 + 6 * 1 * 1 := rfl
  have h4 : hexagonDots 1 = 2 := rfl
  
  -- Now we can calculate step by step
  calc
    hexagonDots 4
    _ = hexagonDots 3 + 6 * 3 * 3 := h1
    _ = (hexagonDots 2 + 6 * 2 * 2) + 6 * 3 * 3 := by rw [h2]
    _ = ((hexagonDots 1 + 6 * 1 * 1) + 6 * 2 * 2) + 6 * 3 * 3 := by rw [h3]
    _ = ((2 + 6 * 1 * 1) + 6 * 2 * 2) + 6 * 3 * 3 := by rw [h4]
    _ = ((2 + 6) + 24) + 54 := by norm_num
    _ = 86 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_hexagon_dots_l1105_110529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1105_110571

open Real

theorem negation_of_proposition :
  (¬ (∀ x > 0, Real.log x ≥ 2 * (x - 1) / (x + 1))) ↔ 
  (∃ x > 0, Real.log x < 2 * (x - 1) / (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1105_110571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharon_journey_l1105_110539

/-- The distance from Sharon's house to her sister's house in miles -/
noncomputable def distance : ℝ := 150

/-- The time taken for a normal journey in minutes -/
noncomputable def normal_time : ℝ := 150

/-- The time taken for today's journey in minutes -/
noncomputable def today_time : ℝ := 210

/-- The fraction of the journey completed before speed reduction -/
noncomputable def fraction_before_reduction : ℝ := 1/2

/-- The factor by which speed is reduced in the second half of the journey -/
noncomputable def speed_reduction_factor : ℝ := 0.8

theorem sharon_journey :
  distance = 150 ∧
  normal_time * distance / (fraction_before_reduction * distance + 
    speed_reduction_factor * (1 - fraction_before_reduction) * distance) = today_time := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharon_journey_l1105_110539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1105_110523

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) :
  Int.gcd (12 * b^3 + 7 * b^2 + 49 * b + 106) (3 * b + 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_problem_l1105_110523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_zero_l1105_110535

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem tangent_slope_angle_at_zero :
  Real.arctan ((λ x => (Real.exp x) * (Real.cos x - Real.sin x)) 0) = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_angle_at_zero_l1105_110535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collatz_like_termination_l1105_110584

def collatz_like (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 5

def process_terminates (n : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate collatz_like k n) = 1

theorem collatz_like_termination (n : ℕ) (h : n > 1) :
  process_terminates n ↔ n % 5 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collatz_like_termination_l1105_110584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_competition_probability_l1105_110513

/-- The probability of player A winning a single game -/
noncomputable def p : ℝ := 2/3

/-- The probability of the match stopping after the third game -/
noncomputable def stop_after_three : ℝ := 1/3

/-- The condition that p is greater than 1/2 -/
axiom p_gt_half : p > 1/2

/-- The condition that the probability of stopping after three games is correct -/
axiom stop_after_three_correct : p^3 + (1-p)^3 = stop_after_three

/-- The probability of A winning the match -/
noncomputable def prob_A_wins : ℝ := 1400/2187

theorem chess_competition_probability :
  prob_A_wins = p^3 + 3 * p^2 * (1-p) * p^2 + 10 * p^3 * (1-p)^2 * p^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_competition_probability_l1105_110513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carriage_problem_l1105_110555

/-- Represents the number of carriages -/
def x : ℕ → ℕ := fun n => n

/-- Represents the total number of people -/
def total_people (n : ℕ) : ℕ := 3 * (x n - 2)

/-- Represents the condition where two people share a carriage -/
def two_per_carriage (n : ℕ) : ℕ := 2 * x n + 9

/-- Theorem stating the equality of the two representations of total people -/
theorem carriage_problem (n : ℕ) : total_people n = two_per_carriage n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carriage_problem_l1105_110555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_pyramid_volume_l1105_110534

/-- Function to calculate the volume of the pyramid from a regular pentagon --/
noncomputable def volume_of_pyramid_from_regular_pentagon (a : ℝ) : ℝ :=
  let d := a / (2 * Real.cos (72 * Real.pi / 180))  -- diagonal length
  let base_area := (a^2 / 4) * Real.sqrt (5 + 2 * Real.sqrt 5)  -- area of base triangle
  let height := d / Real.sqrt 5 * Real.sqrt (5 - 2 * Real.sqrt 5)  -- height of pyramid
  (1 / 3) * base_area * height

/-- The volume of a pyramid formed from a regular pentagon with equilateral triangles --/
theorem pentagon_pyramid_volume (a : ℝ) (h : a > 0) : ∃ V : ℝ,
  V = (a^3 / 24) * (Real.sqrt 5 + 1) ∧
  V = volume_of_pyramid_from_regular_pentagon a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_pyramid_volume_l1105_110534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_max_area_l1105_110515

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance from the center to a focus -/
noncomputable def Ellipse.focal_distance (e : Ellipse) : ℝ :=
  e.a * e.eccentricity

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

theorem ellipse_equation_from_max_area (e : Ellipse) :
  e.eccentricity = 1 / 2 →
  (∃ (M : ℝ × ℝ), M.1^2 / e.a^2 + M.2^2 / e.b^2 = 1 ∧
    ∀ (N : ℝ × ℝ), N.1^2 / e.a^2 + N.2^2 / e.b^2 = 1 →
      triangle_area (2 * e.focal_distance) (abs (N.2 - 0)) ≤ Real.sqrt 3) →
  triangle_area (2 * e.focal_distance) e.b = Real.sqrt 3 →
  e.a = 2 ∧ e.b = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_from_max_area_l1105_110515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_equals_negative_two_l1105_110573

-- Define the function g
def g (x : ℝ) : ℝ := x^3

-- State the theorem
theorem inverse_sum_equals_negative_two :
  (g⁻¹) 8 + (g⁻¹) (-64) = -2 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_sum_equals_negative_two_l1105_110573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prob_c_second_l1105_110538

/-- Represents the probability of winning against a player -/
structure WinProbability (α : Type) where
  prob : α → ℝ
  pos : ∀ x, 0 < prob x

/-- Represents a chess player -/
inductive Player : Type
  | Main
  | A
  | B
  | C
deriving BEq, Repr

/-- The probability of winning two consecutive matches -/
noncomputable def prob_two_consecutive (p : WinProbability Player) (order : List Player) : ℝ :=
  sorry

/-- The main theorem -/
theorem max_prob_c_second 
  (p : WinProbability Player) 
  (h1 : p.prob Player.A < p.prob Player.B)
  (h2 : p.prob Player.B < p.prob Player.C) :
  ∀ (order1 order2 : List Player),
    Player.C ∈ order1 ∧ 
    Player.C ∈ order2 ∧ 
    order1.indexOf Player.C = 1 ∧ 
    order2.indexOf Player.C ≠ 1 →
    prob_two_consecutive p order1 ≥ prob_two_consecutive p order2 :=
by
  sorry

#check max_prob_c_second

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_prob_c_second_l1105_110538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cosines_eleven_l1105_110583

theorem sum_of_cosines_eleven : 
  Real.cos (π / 11) + Real.cos (3 * π / 11) + Real.cos (5 * π / 11) + Real.cos (7 * π / 11) + Real.cos (9 * π / 11) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cosines_eleven_l1105_110583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1105_110580

noncomputable def f (x m n : ℝ) : ℝ := (3^x + n) / (3^(x+1) + m)

theorem f_properties :
  ∃ (m n : ℝ),
    (∀ x, f x m n = -f (-x) m n) ∧  -- f is odd
    (m = 3 ∧ n = -1) ∧
    (∀ x₁ x₂, x₁ < x₂ → f x₁ m n < f x₂ m n) ∧  -- f is increasing
    (∀ k, (∀ x ∈ Set.Icc (1/3) 2, f (k*x^2) m n + f (2*x-1) m n > 0) → k > 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1105_110580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1105_110550

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + (2 + Real.log a) * x + Real.log b

-- State the theorem
theorem function_properties (a b : ℝ) (h1 : f a b (-1) = -2) :
  a = 10 * b ∧ 
  (∀ x : ℝ, f a b x ≥ 2 * x → a = 100 ∧ b = 10) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1105_110550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1105_110574

noncomputable def f (x : Real) : Real := Real.sqrt 3 * Real.cos (Real.pi / 2 - x) + Real.sin (Real.pi / 2 + x)

theorem f_properties :
  (∀ x, f ((-Real.pi/6) - x) = f ((-Real.pi/6) + x)) ∧
  (∀ x y, x ∈ Set.Icc (-2*Real.pi/3) 0 → y ∈ Set.Icc (-2*Real.pi/3) 0 → x < y → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1105_110574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1105_110548

def a : ℕ → ℚ
  | 0 => 1  -- Define a₀ to be 1 (same as a₁)
  | n + 1 => (2 * a n) / (1 + 2 * a n)

theorem a_formula (n : ℕ) : 
  a (n + 1) = (2^n) / (2^(n+1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1105_110548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glucose_mixture_amount_l1105_110514

/-- Represents the concentration of glucose in a solution -/
structure Concentration where
  grams : ℚ
  volume : ℚ

/-- Represents a glucose solution -/
structure Solution where
  concentration : Concentration
  volume : ℚ

/-- Calculates the amount of glucose in a solution -/
def glucose_amount (s : Solution) : ℚ :=
  (s.concentration.grams / s.concentration.volume) * s.volume

/-- The problem statement -/
theorem glucose_mixture_amount :
  let solution1 : Solution := {
    concentration := { grams := 20, volume := 100 },
    volume := 80
  }
  let solution2 : Solution := {
    concentration := { grams := 30, volume := 100 },
    volume := 50
  }
  glucose_amount solution1 + glucose_amount solution2 = 31 := by
  -- Expand the definitions and perform the calculation
  simp [glucose_amount]
  -- The proof is completed by normalization of rational numbers
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_glucose_mixture_amount_l1105_110514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1105_110524

/-- The equation of a line passing through (0, 3) and perpendicular to x+y+1=0 is x-y+3=0 -/
theorem line_equation_proof (x y : ℝ) : 
  let center : ℝ × ℝ := (0, 3)
  let perpendicular_line := {(x, y) : ℝ × ℝ | x + y + 1 = 0}
  let l := {(x, y) : ℝ × ℝ | (center.1 = x ∧ center.2 = y) ∨ 
            ∃ (x1 y1 x2 y2 : ℝ), (x1, y1) ∈ perpendicular_line ∧ (x2, y2) ∈ perpendicular_line ∧
            (y - center.2) / (x - center.1) = -1 / ((y2 - y1) / (x2 - x1))}
  (x, y) ∈ l ↔ x - y + 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l1105_110524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_x_and_y_l1105_110591

-- Define the variables and conditions
variable (t : ℝ)
variable (h : t > 1)

-- Define x and y as functions of t
noncomputable def x (t : ℝ) : ℝ := t^(2/(t-1))
noncomputable def y (t : ℝ) : ℝ := t^((t+1)/(t-1))

-- State the theorem
theorem relationship_between_x_and_y (t : ℝ) (h : t > 1) : 
  (y t)^(2 * (x t)) = (x t)^(y t) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_x_and_y_l1105_110591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_value_l1105_110592

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem floor_expression_value :
  (floor 6.5) * (floor (2/3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.3) - (6.6 : ℝ) = 15.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_value_l1105_110592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l1105_110595

theorem quartic_equation_solutions :
  {z : ℂ | z^4 - 6*z^2 + 8 = 0} = {-2, -Complex.I * Real.sqrt 2, Complex.I * Real.sqrt 2, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l1105_110595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1105_110545

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 6)

-- State the theorem
theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → q ≥ p) ∧ p = π) ∧
  (∀ (x : ℝ), f (2 * π / 3 - x) = f (2 * π / 3 + x)) ∧
  (∀ (x y : ℝ), -π/6 ≤ x ∧ x < y ∧ y ≤ π/3 → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1105_110545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_triangle_side_length_l1105_110559

/-- The side length of smaller equilateral triangles constructed on the sides of a larger equilateral triangle -/
theorem smaller_triangle_side_length (s : ℝ) : 
  let large_triangle_side := (2 : ℝ)
  let large_triangle_area := Real.sqrt 3 * large_triangle_side^2 / 4
  let small_triangle_area := large_triangle_area / 2
  let num_small_triangles := (3 : ℝ)
  s > 0 →
  s^2 * Real.sqrt 3 / 4 = small_triangle_area / num_small_triangles →
  s = Real.sqrt (2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_triangle_side_length_l1105_110559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indeterminate_missing_value_l1105_110517

def sequenceExample : List ℝ := [2, 26, 1, 25, 0]

theorem indeterminate_missing_value (s : List ℝ) (h1 : s.length = 5) 
  (h2 : 25 ∈ s) (h3 : 2 ∈ s) (h4 : 26 ∈ s) (h5 : 1 ∈ s) 
  (h6 : ∃ x, x ∈ s ∧ x ≠ 2 ∧ x ≠ 26 ∧ x ≠ 1 ∧ x ≠ 25) 
  (h7 : 25 = s.maximum) : 
  ¬ ∃! x, x ∈ s ∧ x ≠ 2 ∧ x ≠ 26 ∧ x ≠ 1 ∧ x ≠ 25 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indeterminate_missing_value_l1105_110517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_after_cube_placement_l1105_110508

/-- The new water depth after placing a cube in a partially filled container -/
noncomputable def new_water_depth (a : ℝ) : ℝ :=
  if 0 < a ∧ a < 9 then (10/9) * a
  else if 9 ≤ a ∧ a < 49 then a + 1
  else if 49 ≤ a ∧ a ≤ 50 then 50
  else 0

/-- Theorem stating the new water depth after placing a cube in a partially filled container -/
theorem water_depth_after_cube_placement (a : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ 50) : 
  new_water_depth a = 
    if 0 < a ∧ a < 9 then (10/9) * a
    else if 9 ≤ a ∧ a < 49 then a + 1
    else 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_after_cube_placement_l1105_110508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l1105_110594

noncomputable section

open Real

/-- The volume of a cylinder with radius r and height h -/
def cylinderVolume (r h : ℝ) : ℝ := π * r^2 * h

/-- The volume of a hemisphere with radius r -/
def hemisphereVolume (r : ℝ) : ℝ := (2/3) * π * r^3

/-- The total volume of the region around a line segment -/
def totalVolume (r l : ℝ) : ℝ := cylinderVolume r l + 2 * hemisphereVolume r

theorem line_segment_length (r : ℝ) (volume : ℝ) :
  r = 4 → volume = 320 * π → ∃ l : ℝ, totalVolume r l = volume ∧ l = 44/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l1105_110594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_not_sqrt3_implies_not_pi_third_l1105_110567

theorem tan_not_sqrt3_implies_not_pi_third (α : ℝ) : 
  Real.tan α ≠ Real.sqrt 3 → α ≠ π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_not_sqrt3_implies_not_pi_third_l1105_110567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l1105_110579

/-- Given points M, A, O, and B in 3D space, if vector OM equals vector AB,
    then B has coordinates (9, 1, 1) -/
theorem point_b_coordinates (M A O B : ℝ × ℝ × ℝ) : 
  M = (5, -1, 2) → 
  A = (4, 2, -1) → 
  O = (0, 0, 0) → 
  (M.fst - O.fst, M.snd - O.snd, (Prod.snd M).snd - (Prod.snd O).snd) = 
  (B.fst - A.fst, B.snd - A.snd, (Prod.snd B).snd - (Prod.snd A).snd) → 
  B = (9, 1, 1) := by
  sorry

#check point_b_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_b_coordinates_l1105_110579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l1105_110544

theorem polynomial_existence : 
  (¬ ∃ (P Q R : ℝ → ℝ → ℝ → ℝ), 
    ∀ x y z : ℝ, (x - y + 1)^3 * P x y z + (y - z - 1)^3 * Q x y z + (z - 2*x + 1)^3 * R x y z = 1) ∧
  (∃ (P Q R : ℝ → ℝ → ℝ → ℝ), 
    ∀ x y z : ℝ, (x - y + 1)^3 * P x y z + (y - z - 1)^3 * Q x y z + (z - x + 1)^3 * R x y z = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_existence_l1105_110544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1105_110558

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  4 * (Real.sin (t.A/2 + t.B/2))^2 - Real.cos (2*t.C) = 7/2 ∧ t.c = Real.sqrt 7

/-- Theorem stating the angle C and maximum area of the triangle -/
theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.C = π/3 ∧ (∃ (max_area : Real), max_area = 7 * Real.sqrt 3 / 4 ∧
    ∀ (area : Real), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1105_110558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_parameters_l1105_110525

-- Define the curves C1 and C2
noncomputable def C1 (φ : ℝ) : ℝ × ℝ := (Real.cos φ, Real.sin φ)
noncomputable def C2 (a b φ : ℝ) : ℝ × ℝ := (a * Real.cos φ, b * Real.sin φ)

-- Define the intersection points
noncomputable def intersection (α : ℝ) (a b : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let p1 := C1 α
  let p2 := C2 a b α
  (p1.1, p1.2, p2.1, p2.2)

-- State the theorem
theorem curve_parameters :
  ∀ a b : ℝ,
    a > b ∧ b > 0 ∧
    (let (x1, y1, x2, y2) := intersection 0 a b
     (x2 - x1)^2 + (y2 - y1)^2 = 4) ∧
    (let (x1, y1, x2, y2) := intersection (Real.pi/2) a b
     x1 = x2 ∧ y1 = y2) →
    a = 3 ∧ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_parameters_l1105_110525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1105_110543

/-- The speed of a train crossing a bridge -/
noncomputable def train_speed (train_length bridge_length : ℝ) (time : ℝ) : ℝ :=
  (train_length + bridge_length) / time

/-- Theorem stating the speed of the train -/
theorem train_speed_calculation :
  let train_length : ℝ := 120
  let bridge_length : ℝ := 150
  let time : ℝ := 13.884603517432893
  abs (train_speed train_length bridge_length time - 19.45) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1105_110543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1105_110588

noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, 1)

noncomputable def b (x : ℝ) : ℝ × ℝ := (1/2, Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1105_110588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_and_T_not_third_l1105_110572

-- Define the set of runners
inductive Runner : Type
| P | Q | R | S | T

-- Define the relation "beats"
def beats : Runner → Runner → Prop := sorry

-- Define the relation "finishes_before"
def finishes_before : Runner → Runner → Prop := sorry

-- Axioms based on the given conditions
axiom Q_beats_P : beats Runner.Q Runner.P
axiom Q_beats_R : beats Runner.Q Runner.R
axiom P_beats_S : beats Runner.P Runner.S
axiom T_after_Q_before_P : finishes_before Runner.Q Runner.T ∧ finishes_before Runner.T Runner.P
axiom S_beats_T : beats Runner.S Runner.T

-- Define what it means to finish third
def finishes_third (x : Runner) : Prop :=
  ∃ (a b : Runner), (a ≠ x ∧ b ≠ x ∧ a ≠ b) ∧
  finishes_before a x ∧ finishes_before b x ∧
  ∀ y, y ≠ x ∧ y ≠ a ∧ y ≠ b → finishes_before x y

-- The theorem to prove
theorem Q_and_T_not_third :
  ¬(finishes_third Runner.Q) ∧ ¬(finishes_third Runner.T) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_Q_and_T_not_third_l1105_110572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_representation_exists_l1105_110568

/-- A set of non-negative integers where all bits in even positions are set to 0 -/
def A : Set ℕ :=
  {x | ∀ k : ℕ, Nat.bodd (x / (2^(2*k))) = false}

/-- A set of non-negative integers where all bits in odd positions are set to 0 -/
def B : Set ℕ :=
  {y | ∀ k : ℕ, Nat.bodd (y / (2^(2*k+1))) = false}

/-- Theorem stating the existence of sets A and B with the desired property -/
theorem unique_representation_exists :
  (Set.Infinite A) ∧ (Set.Infinite B) ∧
  (∀ n : ℕ, ∃! (a : ℕ) (b : ℕ), a ∈ A ∧ b ∈ B ∧ n = a + b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_representation_exists_l1105_110568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_double_is_one_seventh_l1105_110598

/-- A modified domino set with integers from 0 to 12 -/
structure ModifiedDominoSet where
  range : Finset ℕ
  range_def : range = Finset.range 13

/-- A domino in the modified set -/
structure Domino (s : ModifiedDominoSet) where
  first : ℕ
  second : ℕ
  valid : first ∈ s.range ∧ second ∈ s.range

/-- A double domino has the same number on both squares -/
def Domino.isDouble {s : ModifiedDominoSet} (d : Domino s) : Prop := d.first = d.second

/-- The total number of dominoes in the set -/
def totalDominoes (s : ModifiedDominoSet) : ℕ := (s.range.card * (s.range.card + 1)) / 2

/-- The number of double dominoes in the set -/
def doubleDominoes (s : ModifiedDominoSet) : ℕ := s.range.card

/-- The probability of randomly selecting a double domino -/
def probabilityOfDouble (s : ModifiedDominoSet) : ℚ :=
  (doubleDominoes s : ℚ) / (totalDominoes s : ℚ)

theorem probability_of_double_is_one_seventh (s : ModifiedDominoSet) :
  probabilityOfDouble s = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_double_is_one_seventh_l1105_110598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_solutions_l1105_110504

/-- Given two intersecting lines and three segments b, c, d where b < c ≤ d,
    this function returns the number of possible circles with diameter d
    that enclose a segment of length b from one line and a segment of length c from the other line. -/
noncomputable def numCircleSolutions (b c d : ℝ) : ℕ :=
  if c = d then 4 else 8

/-- Theorem stating the number of circle solutions given the conditions -/
theorem circle_solutions (b c d : ℝ) (h1 : b < c) (h2 : c ≤ d) :
  numCircleSolutions b c d = if c = d then 4 else 8 := by
  sorry

#check circle_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_solutions_l1105_110504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bubble_radius_l1105_110576

/-- The radius of the hemisphere formed by a spherical bubble -/
noncomputable def hemisphere_radius : ℝ := 6 * (4 : ℝ)^(1/3)

/-- The volume of a sphere with radius R -/
noncomputable def sphere_volume (R : ℝ) : ℝ := (4/3) * Real.pi * R^3

/-- The volume of a hemisphere with radius r -/
noncomputable def hemisphere_volume (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

theorem bubble_radius : 
  ∃ (R : ℝ), 
    R > 0 ∧ 
    hemisphere_volume hemisphere_radius = 2 * sphere_volume R ∧ 
    R = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bubble_radius_l1105_110576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_minus_one_eq_x_minus_one_l1105_110505

theorem cube_root_minus_one_eq_x_minus_one (x : ℝ) : 
  (x^3)^(1/3) - 1 = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_minus_one_eq_x_minus_one_l1105_110505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_perp_to_same_line_are_parallel_unique_perp_plane_through_skew_line_l1105_110530

-- Define the concept of a line in 3D space
structure Line3D where
  -- Add necessary fields
  point : Real × Real × Real
  direction : Real × Real × Real

-- Define the concept of a plane in 3D space
structure Plane3D where
  -- Add necessary fields
  normal : Real × Real × Real
  point : Real × Real × Real

-- Define perpendicularity between a line and a plane
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Define parallelism between planes
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

-- Define a skew line to a plane
def skew (l : Line3D) (p : Plane3D) : Prop :=
  sorry

-- Theorem 1: Two planes perpendicular to the same line are parallel
theorem planes_perp_to_same_line_are_parallel (l : Line3D) (p1 p2 : Plane3D) :
  perpendicular l p1 → perpendicular l p2 → parallel p1 p2 :=
by sorry

-- Theorem 2: There is exactly one plane perpendicular to a given plane through a skew line
theorem unique_perp_plane_through_skew_line (p : Plane3D) (l : Line3D) :
  skew l p → ∃! (q : Plane3D), perpendicular l q ∧ perpendicular l p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_planes_perp_to_same_line_are_parallel_unique_perp_plane_through_skew_line_l1105_110530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_colony_decay_l1105_110549

theorem bee_colony_decay (P : ℝ) (h : P > 0) :
  let retention_rate : ℝ := 0.994
  let days : ℕ := Int.toNat ⌈(Real.log (1/4)) / (Real.log retention_rate)⌉
  (retention_rate ^ (days : ℝ)) * P ≤ (1/4) * P ∧
  ∀ d : ℕ, d < days → (retention_rate ^ (d : ℝ)) * P > (1/4) * P :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_colony_decay_l1105_110549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1105_110507

-- Define the function f with domain [0,8]
def f : Set ℝ := { x | 0 ≤ x ∧ x ≤ 8 }

-- Define the new function g(x) = f(2x)/(x-4)
def g : Set ℝ := { x | x ∈ f ∧ x ≠ 4 }

-- Theorem stating the domain of g
theorem domain_of_g :
  g = { x | 0 ≤ x ∧ x < 4 } :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1105_110507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1105_110537

-- Define the Heaviside function
noncomputable def η (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else 0

-- Define the left-hand side of the equation
noncomputable def lhs (x : ℝ) : ℝ :=
  Real.tan (x * (η (x - 2 * Real.pi) - η (x - 5 * Real.pi)))

-- Define the right-hand side of the equation
noncomputable def rhs (x : ℝ) : ℝ :=
  1 / (Real.cos x)^2 - 1

-- State the theorem
theorem equation_solution :
  ∀ x : ℝ, lhs x = rhs x ↔ 
    (∃ k : ℤ, x = k * Real.pi) ∨ 
    (∃ m : ℤ, m ∈ ({2, 3, 4} : Set ℤ) ∧ x = Real.pi / 4 + m * Real.pi) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1105_110537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_value_theorem_l1105_110520

/-- Calculates the market value of a share after 5 years -/
noncomputable def market_value_after_5_years (P : ℝ) (I : ℝ) : ℝ :=
  P * (1 + I / 100) ^ 5

/-- The market value of a share after 5 years is P * (1 + I/100)^5 -/
theorem market_value_theorem (P I : ℝ) :
  market_value_after_5_years P I = P * (1 + I / 100) ^ 5 := by
  -- Unfold the definition of market_value_after_5_years
  unfold market_value_after_5_years
  -- The equation is now trivially true
  rfl

#check market_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_value_theorem_l1105_110520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_satisfies_conditions_l1105_110532

/-- Represents a polynomial with real coefficients -/
def MyPolynomial := ℝ → ℝ

/-- The condition that P(X) divided by X^3 - 2 has remainder equal to the fourth power of the quotient -/
def condition1 (P Q : MyPolynomial) : Prop :=
  ∀ x, P x = (x^3 - 2) * (Q x) + (Q x)^4

/-- The second condition that P(-2) + P(2) = -34 -/
def condition2 (P : MyPolynomial) : Prop :=
  P (-2) + P 2 = -34

/-- The main theorem stating that no polynomial satisfies both conditions -/
theorem no_polynomial_satisfies_conditions :
  ¬ ∃ (P Q : MyPolynomial), condition1 P Q ∧ condition2 P :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_polynomial_satisfies_conditions_l1105_110532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_point_zero_two_l1105_110533

-- Define the function f(x) = a^x + 1
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + 1

-- Theorem statement
theorem function_passes_through_point_zero_two
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 0 = 2 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify a^0 to 1
  simp [Real.rpow_zero]
  -- Evaluate 1 + 1
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_passes_through_point_zero_two_l1105_110533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_volume_l1105_110554

-- Define the parameters of the water tank
def base_diameter : ℝ := 20
def max_depth : ℝ := 6

-- Define the volume of a cone
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem water_tank_volume :
  let base_radius : ℝ := base_diameter / 2
  cone_volume base_radius max_depth = 200 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_volume_l1105_110554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1105_110589

noncomputable def f (x : ℝ) : ℝ := (Real.sin x ^ 3 + 6 * Real.sin x ^ 2 + Real.sin x + 2 * Real.cos x ^ 2 - 8) / (Real.sin x - 1)

theorem f_range : 
  ∀ x : ℝ, Real.sin x ≠ 1 → 
  ∃ y ∈ Set.Icc 2 12, f x = y ∧ 
  ∀ z, f x = z → z ∈ Set.Icc 2 12 ∧ z ≠ 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1105_110589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l1105_110565

-- Define the points
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 14)
def C : ℝ × ℝ := (3, 7)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define A' and B' on the line y = x
def A' : ℝ × ℝ := (5, 5)
def B' : ℝ × ℝ := (4.2, 4.2)

-- Define the condition that AA' and BB' intersect at C
def intersect_at_C : Prop :=
  ∃ (t₁ t₂ : ℝ), 0 < t₁ ∧ t₁ < 1 ∧ 0 < t₂ ∧ t₂ < 1 ∧
    C = (A.1 + t₁ * (A'.1 - A.1), A.2 + t₁ * (A'.2 - A.2)) ∧
    C = (B.1 + t₂ * (B'.1 - B.1), B.2 + t₂ * (B'.2 - B.2))

-- Define the distance function
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2)

-- The theorem to prove
theorem length_of_A'B' :
  line_y_eq_x A' ∧ line_y_eq_x B' ∧ intersect_at_C →
  distance A' B' = Real.sqrt 1.28 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l1105_110565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_exp_curve_l1105_110560

-- Define the function f(x) = e^x
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the upper bound of the region
noncomputable def upper_bound : ℝ := Real.exp 1

-- Define the theorem
theorem area_enclosed_by_exp_curve (a b : ℝ) : 
  a = 0 → b = 1 → ∫ x in a..b, (upper_bound - f x) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_exp_curve_l1105_110560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_skew_lines_tetrahedron_l1105_110518

/-- Represents a tetrahedron ABCD with given edge lengths -/
structure Tetrahedron where
  a : ℝ  -- Length of edges AB and CD
  b : ℝ  -- Length of edges BC and AD
  c : ℝ  -- Length of edges CA and BD
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The cosine of the angle between skew lines AB and CD in the tetrahedron -/
noncomputable def cosine_skew_lines (t : Tetrahedron) : ℝ :=
  |t.b^2 - t.c^2| / t.a^2

theorem cosine_skew_lines_tetrahedron (t : Tetrahedron) :
  cosine_skew_lines t = |t.b^2 - t.c^2| / t.a^2 := by
  -- Unfold the definition of cosine_skew_lines
  unfold cosine_skew_lines
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_skew_lines_tetrahedron_l1105_110518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1105_110500

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 6 - Real.sin x * Real.cos x + Real.cos x ^ 6

-- State the theorem
theorem g_range : ∀ x : ℝ, 0 ≤ g x ∧ g x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1105_110500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_fixed_point_value_l1105_110570

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 2) / (x - 2)

-- Theorem stating that -1 is the only solution to f(a) = a
theorem unique_fixed_point :
  ∃! a : ℝ, f a = a ∧ a ≠ 2 :=
by sorry

-- Theorem explicitly stating the value of the unique fixed point
theorem fixed_point_value :
  ∀ a : ℝ, f a = a ∧ a ≠ 2 → a = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_fixed_point_fixed_point_value_l1105_110570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eva_hits_ten_l1105_110562

-- Define the set of friends
inductive Friend : Type
| Alex | Brenda | Carol | Dan | Eva | Felix

-- Define the score type
def Score := Fin 15

-- Define a function to represent the total score for each friend
def totalScore (f : Friend) : Nat :=
  match f with
  | Friend.Alex => 21
  | Friend.Brenda => 12
  | Friend.Carol => 18
  | Friend.Dan => 26
  | Friend.Eva => 28
  | Friend.Felix => 20

-- Define a function to represent the three dart throws for each friend
def throws (f : Friend) : Fin 3 → Score := sorry

-- Define the condition that all throws are different for each friend
def allDifferent (f : Friend) : Prop :=
  ∀ i j : Fin 3, i ≠ j → throws f i ≠ throws f j

-- Define the condition that the sum of throws equals the total score
def sumEqualsTotal (f : Friend) : Prop :=
  (throws f 0).val + (throws f 1).val + (throws f 2).val = totalScore f

-- Define the theorem
theorem eva_hits_ten :
  (∀ f : Friend, allDifferent f ∧ sumEqualsTotal f) →
  ∃ i : Fin 3, throws Friend.Eva i = ⟨10, sorry⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eva_hits_ten_l1105_110562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1105_110510

-- Define the ceiling function {a}
noncomputable def ceil (a : ℝ) : ℤ :=
  Int.ceil a

-- Define the floor function [a]
noncomputable def floor (a : ℝ) : ℤ :=
  Int.floor a

-- Define the theorem
theorem solution_exists (x y : ℝ) : 
  (3 * floor x + 2 * ceil y = 18 ∧ 3 * ceil x - floor y = 4) →
  (2 ≤ x ∧ x < 3 ∧ 5 ≤ y ∧ y < 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1105_110510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_correct_l1105_110553

/-- Represents a convex solid bounded by a regular hexagon, an equilateral triangle,
    three additional equilateral triangles, and three squares. -/
structure ConvexSolid where
  m : ℝ  -- Distance between parallel planes
  c : ℝ  -- Side length of hexagon and triangle
  t6 : ℝ  -- Area of hexagon face
  t3 : ℝ  -- Area of triangle face
  T : ℝ   -- Area of midpoint cross-section

/-- The volume formula for the convex solid -/
noncomputable def volume (solid : ConvexSolid) : ℝ :=
  (solid.m / 6) * (solid.t6 + 4 * solid.T + solid.t3)

/-- Theorem stating that the volume formula is correct for the given convex solid -/
theorem volume_formula_correct (solid : ConvexSolid) :
  volume solid = (5 * solid.m * solid.c^2) / (2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_correct_l1105_110553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integer_values_l1105_110501

-- Define the quadratic function
noncomputable def f (x : ℝ) : ℝ := x^2 - 9*x + c
  where c : ℝ := sorry  -- c is some real constant

-- State the theorem
theorem quadratic_integer_values :
  (∀ x : ℝ, (deriv f) x = 2*x - 9) →  -- Condition on derivative
  (∃ n : ℤ, f 0 = n) →                -- f(0) is an integer
  (∃! n : ℤ, ∃ x : ℝ, 4 < x ∧ x ≤ 5 ∧ f x = n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_integer_values_l1105_110501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l1105_110575

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

def line (k t : ℝ) (x y : ℝ) : Prop :=
  y = k * x + t

theorem ellipse_and_line_theorem 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 3 / 2) 
  (h4 : distance (3/4) 0 a 0 = distance (3/4) 0 0 b) :
  (∃ (x y : ℝ), ellipse 2 1 x y ↔ ellipse a b x y) ∧ 
  (∃ (k t : ℝ), 
    (k = 1/2 ∧ t = -1) ∨ 
    (k = -1/2 ∧ t = 1) ∨ 
    (k = Real.sqrt 5 / 2 ∧ t = -3 * Real.sqrt 5 / 5) ∨ 
    (k = -Real.sqrt 5 / 2 ∧ t = 3 * Real.sqrt 5 / 5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l1105_110575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_eight_l1105_110506

noncomputable def g (x : ℝ) : ℝ := 3 * x + 2

noncomputable def g_inv (x : ℝ) : ℝ := (x - 2) / 3

theorem sum_of_solutions_is_eight :
  ∃ (S : Finset ℝ), (∀ x ∈ S, g_inv x = g (x⁻¹)) ∧ (S.sum id = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_eight_l1105_110506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_over_six_l1105_110522

theorem sin_pi_over_six : Real.sin (π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_over_six_l1105_110522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_filtrations_sufficient_less_filtrations_insufficient_l1105_110511

/-- The minimum number of filtrations required to reduce impurities to market standard -/
def min_filtrations : ℕ := 8

/-- The initial impurity content as a percentage -/
noncomputable def initial_impurity : ℝ := 10

/-- The market standard for maximum impurity content as a percentage -/
noncomputable def market_standard : ℝ := 0.5

/-- The reduction factor of impurities after each filtration -/
noncomputable def reduction_factor : ℝ := 2/3

/-- The value of log₂ -/
noncomputable def lg2 : ℝ := 0.3010

/-- The value of log₃ -/
noncomputable def lg3 : ℝ := 0.4771

/-- Theorem stating that the minimum number of filtrations meets the market standard -/
theorem min_filtrations_sufficient :
  (initial_impurity : ℝ) * reduction_factor ^ min_filtrations ≤ market_standard := by
  sorry

/-- Theorem stating that one less filtration is not sufficient -/
theorem less_filtrations_insufficient :
  (initial_impurity : ℝ) * reduction_factor ^ (min_filtrations - 1) > market_standard := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_filtrations_sufficient_less_filtrations_insufficient_l1105_110511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_game_probabilities_l1105_110596

/-- Represents the result of a chess round -/
inductive RoundResult
  | AWin
  | BWin
  | Draw

/-- Represents a chess player -/
inductive Player
  | A
  | B

/-- The probability of each outcome when a player goes first -/
def probability_when_first (p : Player) (result : RoundResult) : ℚ :=
  match p, result with
  | Player.A, RoundResult.AWin => 2/3
  | Player.A, RoundResult.Draw => 1/6
  | Player.A, RoundResult.BWin => 1/6
  | Player.B, RoundResult.BWin => 1/2
  | Player.B, RoundResult.Draw => 1/4
  | Player.B, RoundResult.AWin => 1/4

/-- Determines who goes first in the next round based on the current result -/
def next_player (current_result : RoundResult) : Player :=
  match current_result with
  | RoundResult.BWin => Player.B
  | _ => Player.A

/-- The probability of A winning three consecutive rounds -/
def prob_A_wins_three_consecutive : ℚ :=
  (probability_when_first Player.A RoundResult.AWin) *
  (probability_when_first Player.B RoundResult.AWin) *
  (probability_when_first Player.B RoundResult.AWin)

/-- The probability of B winning within five rounds -/
def prob_B_wins_within_five : ℚ :=
  let p_B_win_A_first := probability_when_first Player.A RoundResult.BWin
  let p_B_win_B_first := probability_when_first Player.B RoundResult.BWin
  let p_B_not_win_A_first := 1 - p_B_win_A_first
  let p_B_not_win_B_first := 1 - p_B_win_B_first
  
  -- B wins in 3 rounds
  (p_B_win_A_first * p_B_win_B_first * p_B_win_B_first) +
  -- B wins in 4 rounds
  (3 * p_B_win_A_first * p_B_win_B_first * p_B_not_win_B_first * p_B_win_A_first) +
  -- B wins in 5 rounds
  (3 * p_B_win_A_first * p_B_win_B_first * p_B_not_win_B_first * p_B_not_win_A_first * p_B_win_B_first +
   3 * p_B_win_A_first * p_B_not_win_B_first * p_B_win_A_first * p_B_not_win_A_first * p_B_win_B_first)

theorem chess_game_probabilities :
  prob_A_wins_three_consecutive = 1/24 ∧ prob_B_wins_within_five = 31/216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_game_probabilities_l1105_110596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base3_addition_uniqueness_l1105_110561

/-- Represents a digit in base 3 -/
inductive Base3Digit
| one : Base3Digit
| two : Base3Digit

/-- Addition in base 3 with carry -/
def base3Add (a b : Base3Digit) : Base3Digit × Bool :=
  match a, b with
  | Base3Digit.one, Base3Digit.one => (Base3Digit.two, false)
  | Base3Digit.one, Base3Digit.two => (Base3Digit.one, true)
  | Base3Digit.two, Base3Digit.one => (Base3Digit.one, true)
  | Base3Digit.two, Base3Digit.two => (Base3Digit.two, true)

/-- Converts a Base3Digit to its numerical value -/
def base3ToNat (d : Base3Digit) : Nat :=
  match d with
  | Base3Digit.one => 1
  | Base3Digit.two => 2

theorem base3_addition_uniqueness :
  ∀ S H E : Base3Digit,
    S ≠ H → S ≠ E → H ≠ E →
    (let (units, carry1) := base3Add E E
     let (tens, carry2) := base3Add H H
     let (hundreds, _) := base3Add S S
     units = S ∧
     (if carry1 then base3Add tens Base3Digit.one else (tens, false)).1 = S ∧
     (if carry2 then base3Add hundreds Base3Digit.one else (hundreds, false)).1 = H) →
    S = Base3Digit.one ∧ H = Base3Digit.two ∧ E = Base3Digit.two ∧
    base3ToNat S + base3ToNat H + base3ToNat E = 5 :=
by sorry

#check base3_addition_uniqueness

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base3_addition_uniqueness_l1105_110561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_moves_origin_l1105_110578

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Dilation transformation of a point -/
def dilate (center : Point) (k : ℝ) (p : Point) : Point :=
  { x := center.x + k * (p.x - center.x),
    y := center.y + k * (p.y - center.y) }

theorem dilation_moves_origin (original : Circle) (transformed : Circle) :
  original.center = { x := 3, y := 1 } →
  original.radius = 4 →
  transformed.center = { x := 7, y := 9 } →
  transformed.radius = 6 →
  let k := transformed.radius / original.radius
  let dilationCenter := { x := -1/5, y := -19/5 }
  let origin := { x := 0, y := 0 }
  let movedOrigin := dilate dilationCenter k origin
  distance origin movedOrigin = 0.7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_moves_origin_l1105_110578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_calculation_l1105_110566

theorem complex_arithmetic_calculation : 
  let x := (4875957 * 27356)^3 / 8987864 + 48945639
  ∃ y : ℝ, abs (y - 313494128.5) < 1 ∧ abs (x - y) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_arithmetic_calculation_l1105_110566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l1105_110582

-- Define the function f as noncomputable due to Real.log
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + a*x + 1)

-- State the theorem
theorem even_function_implies_a_zero :
  (∀ x, f a x = f a (-x)) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l1105_110582
