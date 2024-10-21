import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_collinearity_l1000_100014

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the triangle ABC
variable (A B C : Point)

-- Define the circles α, β, γ, δ
variable (α β γ δ : Circle)

-- Define the centers of circles and triangle centers
variable (center : Circle → Point)
variable (incenter circumcenter : Point → Point → Point → Point)

-- Define the properties of the circles
variable (equal_radii : Circle → Circle → Circle → Prop)
variable (tangent_to_side : Circle → Point → Point → Prop)
variable (externally_tangent : Circle → Circle → Prop)

-- Define collinearity
variable (collinear : Point → Point → Point → Prop)

-- Theorem statement
theorem center_collinearity 
  (h1 : equal_radii α β γ)
  (h2 : tangent_to_side α B C)
  (h3 : tangent_to_side β A C)
  (h4 : tangent_to_side γ A B)
  (h5 : externally_tangent δ α)
  (h6 : externally_tangent δ β)
  (h7 : externally_tangent δ γ) :
  collinear (center δ) (incenter A B C) (circumcenter A B C) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_collinearity_l1000_100014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_correct_l1000_100040

/-- The area of the region outside three circles, where two circles have radius 1 and are tangent 
    at one point, and the third circle has radius 1 - √2 and is tangent to the other two circles 
    externally. -/
noncomputable def areaOutsideCircles : ℝ := 1 - (Real.pi * (5 - 2 * Real.sqrt 2)) / 4

/-- Theorem stating that the area outside the three circles is equal to the calculated value. -/
theorem area_outside_circles_correct : 
  areaOutsideCircles = 1 - (Real.pi * (5 - 2 * Real.sqrt 2)) / 4 := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_circles_correct_l1000_100040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_coprime_counts_l1000_100043

theorem prime_coprime_counts (p : ℕ) (hp : Nat.Prime p) :
  (∃ n : ℕ, n = (Finset.filter (λ x ↦ x < p ∧ Nat.Coprime x p) (Finset.range p)).card ∧ n = p - 1) ∧
  (∃ m : ℕ, m = (Finset.filter (λ x ↦ x < p^2 ∧ Nat.Coprime x (p^2)) (Finset.range (p^2))).card ∧ m = p * (p - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_coprime_counts_l1000_100043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_sin_cos_alpha_value_l1000_100002

-- Define f(x) as given in the problem
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi - x) + Real.cos (-x) - Real.sin ((5*Real.pi/2) - x) + Real.cos (Real.pi/2 + x)

-- Theorem 1
theorem tan_alpha_value (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) (h3 : f α = 2/3) :
  Real.tan α = 2 * Real.sqrt 5 / 5 := by sorry

-- Theorem 2
theorem sin_cos_alpha_value (α : ℝ) (h : f α = 2 * Real.sin α - Real.cos α + 3/4) :
  Real.sin α * Real.cos α = 7/32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_sin_cos_alpha_value_l1000_100002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_prices_l1000_100020

/-- The price of an adult concert ticket -/
noncomputable def adult_price : ℝ := 47.25 / (6 + 5 * (2/3))

/-- The price of a student concert ticket -/
noncomputable def student_price : ℝ := (2/3) * adult_price

/-- The total cost of 6 adult tickets and 5 student tickets -/
noncomputable def given_total : ℝ := 6 * adult_price + 5 * student_price

/-- The total cost of 10 adult tickets and 8 student tickets -/
noncomputable def target_total : ℝ := 10 * adult_price + 8 * student_price

theorem concert_ticket_prices : 
  given_total = 47.25 ∧ target_total = 77.625 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_prices_l1000_100020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_percentage_is_3_5_percent_l1000_100042

/-- Percentage of products produced by each machine -/
def machine_production : Fin 5 → ℝ
| 0 => 0.30  -- M1
| 1 => 0.20  -- M2
| 2 => 0.15  -- M3
| 3 => 0.25  -- M4
| 4 => 0.10  -- M5

/-- Percentage of defective products for each machine -/
def defective_rate : Fin 5 → ℝ
| 0 => 0.04  -- M1
| 1 => 0.02  -- M2
| 2 => 0.03  -- M3 (100% - 97% = 3%)
| 3 => 0.05  -- M4
| 4 => 0.02  -- M5 (100% - 98% = 2%)

/-- Total percentage of defective products in the stockpile -/
def total_defective_percentage : ℝ :=
  Finset.sum (Finset.range 5) (λ i => machine_production i * defective_rate i)

theorem defective_percentage_is_3_5_percent :
  total_defective_percentage = 0.035 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_defective_percentage_is_3_5_percent_l1000_100042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roam_area_difference_l1000_100022

/-- Represents the dimensions of the rectangular shed -/
structure Shed :=
  (length : ℝ)
  (width : ℝ)

/-- Represents the setup for tying the dog -/
inductive Setup
  | Middle
  | Corner

/-- Calculates the area the dog can roam based on the setup -/
noncomputable def roamArea (s : Setup) (rope_length : ℝ) : ℝ :=
  match s with
  | Setup.Middle => (1/2) * Real.pi * rope_length^2
  | Setup.Corner => (3/4) * Real.pi * rope_length^2

/-- Theorem stating the difference in roaming area between setups -/
theorem roam_area_difference (shed : Shed) (rope_length : ℝ) :
  shed.length = 20 ∧ shed.width = 10 ∧ rope_length = 10 →
  roamArea Setup.Corner rope_length - roamArea Setup.Middle rope_length = 25 * Real.pi := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roam_area_difference_l1000_100022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_to_rhombus_area_ratio_is_pi_sqrt15_over_16_l1000_100059

/-- A rhombus with a height drawn from the vertex of its obtuse angle 
    that divides the opposite side in the ratio 1:3. -/
structure SpecialRhombus where
  side : ℝ
  height : ℝ
  height_divides_side : height^2 + (side / 4)^2 = (3 * side / 4)^2

/-- The ratio of the area of the inscribed circle to the area of the rhombus -/
noncomputable def circle_to_rhombus_area_ratio (r : SpecialRhombus) : ℝ :=
  (Real.pi * (r.height / 2)^2) / (r.side * r.height)

/-- The main theorem stating that the area ratio is π√15/16 -/
theorem circle_to_rhombus_area_ratio_is_pi_sqrt15_over_16 (r : SpecialRhombus) :
  circle_to_rhombus_area_ratio r = Real.pi * Real.sqrt 15 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_to_rhombus_area_ratio_is_pi_sqrt15_over_16_l1000_100059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ABD_l1000_100092

-- Define the circle M
def circleM (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define point N
def N : ℝ × ℝ := (1, 0)

-- Define point G
def G : ℝ × ℝ := (0, 1)

-- Define the ellipse E (trajectory of Q)
def ellipseE (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define a point on the ellipse E
def pointOnE (P : ℝ × ℝ) : Prop := ellipseE P.1 P.2

-- Define a line passing through G
def linePassingThroughG (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the area of triangle ABD
noncomputable def areaABD (A B D : ℝ × ℝ) (k : ℝ) : ℝ := 
  let AB := ((A.1 - B.1)^2 + (A.2 - B.2)^2).sqrt
  let d := 1 / (1 + k^2).sqrt
  AB * d

-- Theorem statement
theorem max_area_ABD :
  ∃ (A B D : ℝ × ℝ) (k : ℝ),
    pointOnE A ∧ pointOnE B ∧
    linePassingThroughG k A.1 A.2 ∧ linePassingThroughG k B.1 B.2 ∧
    D = (-A.1, -A.2) ∧
    (∀ (A' B' D' : ℝ × ℝ) (k' : ℝ),
      pointOnE A' ∧ pointOnE B' ∧
      linePassingThroughG k' A'.1 A'.2 ∧ linePassingThroughG k' B'.1 B'.2 ∧
      D' = (-A'.1, -A'.2) →
      areaABD A' B' D' k' ≤ areaABD A B D k) ∧
    areaABD A B D k = 4 * Real.sqrt 6 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_ABD_l1000_100092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_AB_l1000_100012

-- Define the lines and curve
def l₁ (a x : ℝ) : ℝ := x + a
def l₂ (x : ℝ) : ℝ := 2 * (x + 1)
noncomputable def C (x : ℝ) : ℝ := x + Real.log x

-- Define points A and B
def A (a : ℝ) : ℝ × ℝ := (a - 2, 2 * a - 2)
noncomputable def B (a : ℝ) : ℝ × ℝ := (Real.exp a, Real.exp a + a)

-- Define the distance function between A and B
noncomputable def distance (a : ℝ) : ℝ := Real.sqrt 2 * (Real.exp a - a + 2)

-- Theorem statement
theorem min_distance_AB :
  ∃ (a : ℝ), ∀ (x : ℝ), distance x ≥ distance a ∧ distance a = 3 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_AB_l1000_100012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_rose_fraction_l1000_100026

/-- Proves that Jenna receives 63/125 of the roses given the problem conditions -/
theorem jenna_rose_fraction :
  let total_money : ℕ := 300
  let rose_price : ℕ := 2
  let total_roses_given : ℕ := 125
  let imma_fraction : ℚ := 1/2
  63/125 = 63/125 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_rose_fraction_l1000_100026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l1000_100066

def CircleLocus (O P : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {M | ∃ Q : ℝ × ℝ, (Q.1 - O.1)^2 + (Q.2 - O.2)^2 = r^2 ∧
           M = ((2 * P.1 + O.1) / 3, (2 * P.2 + O.2) / 3) + 
               (1/3) * (Q.1 - ((2 * P.1 + O.1) / 3), Q.2 - ((2 * P.2 + O.2) / 3))}

theorem locus_is_circle (O P : ℝ × ℝ) (r : ℝ) (h : (P.1 - O.1)^2 + (P.2 - O.2)^2 > r^2) :
  CircleLocus O P r = {M : ℝ × ℝ | (M.1 - (2 * P.1 + O.1) / 3)^2 + (M.2 - (2 * P.2 + O.2) / 3)^2 = (r/3)^2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_l1000_100066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_new_average_l1000_100011

/-- Represents a batsman's scoring record -/
structure BatsmanRecord where
  inningsBefore : ℕ  -- number of innings before the current one
  averageBefore : ℚ  -- average score before the current inning
  currentScore : ℕ   -- score in the current inning
  averageIncrease : ℚ -- increase in average after the current inning

/-- Calculates the new average after the current inning -/
def newAverage (record : BatsmanRecord) : ℚ :=
  (record.inningsBefore * record.averageBefore + record.currentScore) / (record.inningsBefore + 1)

/-- Theorem: Given the conditions, the new average is 42 -/
theorem batsman_new_average (record : BatsmanRecord) 
  (h1 : record.inningsBefore = 16)
  (h2 : record.currentScore = 90)
  (h3 : record.averageIncrease = 3)
  (h4 : newAverage record = record.averageBefore + record.averageIncrease) :
  newAverage record = 42 := by
  sorry

#eval newAverage { inningsBefore := 16, averageBefore := 39, currentScore := 90, averageIncrease := 3 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_new_average_l1000_100011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_l1000_100052

/-- An oblique coordinate system with angle between axes of 60 degrees -/
structure ObliqueCoordSystem where
  angle : ℝ
  angle_eq : angle = Real.pi / 3

/-- A point in the oblique coordinate system -/
structure Point (sys : ObliqueCoordSystem) where
  x : ℝ
  y : ℝ

/-- The distance between two points in the oblique coordinate system -/
noncomputable def distance (sys : ObliqueCoordSystem) (p q : Point sys) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + 2 * (p.x - q.x) * (p.y - q.y) * Real.cos sys.angle)

theorem distance_to_origin (sys : ObliqueCoordSystem) :
  let M : Point sys := ⟨1, 2⟩
  let O : Point sys := ⟨0, 0⟩
  distance sys M O = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_origin_l1000_100052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l1000_100083

theorem sin_minus_cos_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = 1/8)
  (h2 : π/4 < α) 
  (h3 : α < π/2) : 
  Real.sin α - Real.cos α = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l1000_100083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_digits_different_l1000_100015

/-- Represents a time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Checks if a given time is within the range 05:00:00 to 22:59:59 -/
def Time.inRange (t : Time) : Prop :=
  (t.hours ≥ 5 ∧ t.hours ≤ 22) ∨ 
  (t.hours = 22 ∧ t.minutes ≤ 59 ∧ t.seconds ≤ 59)

/-- Checks if all digits in a given time are different -/
def Time.allDigitsDifferent (t : Time) : Prop :=
  let digits := [t.hours / 10, t.hours % 10, t.minutes / 10, t.minutes % 10, t.seconds / 10, t.seconds % 10]
  List.Nodup digits

/-- The probability of all digits being different within the specified time range -/
def probabilityAllDigitsDifferent : ℚ :=
  16 / 135

theorem probability_all_digits_different :
  probabilityAllDigitsDifferent = (Nat.card {t : Time | t.inRange ∧ t.allDigitsDifferent}) / (Nat.card {t : Time | t.inRange}) :=
by sorry

#check probability_all_digits_different

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_digits_different_l1000_100015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_in_quadrant_II_or_IV_l1000_100049

open Real

theorem alpha_in_quadrant_II_or_IV (α : ℝ) 
  (h : sin α * (1 / cos α) * (Real.sqrt ((1 / sin α)^2 - 1)) = -1) : 
  (π/2 < α ∧ α < π) ∨ (3*π/2 < α ∧ α < 2*π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_in_quadrant_II_or_IV_l1000_100049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l1000_100046

noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

noncomputable def triangle_area (a b : ℝ) : ℝ :=
  (1/2) * a * b

theorem hypotenuse_length :
  ∀ x y z : ℝ,
  x > 0 →
  y = 3*x + 3 →
  right_triangle x y z →
  triangle_area x y = 150 →
  z = (Real.sqrt (4047 + 18 * Real.sqrt 401)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hypotenuse_length_l1000_100046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l1000_100096

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then -x^2 + x else -x^2 - x

theorem f_is_even_and_correct : 
  (∀ x, f x = f (-x)) ∧ 
  (∀ x > 0, f x = -x^2 + x) ∧ 
  (∀ x < 0, f x = -x^2 - x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_correct_l1000_100096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nearsighted_light_user_probability_l1000_100025

/-- Represents the proportion of students in a school -/
structure SchoolPopulation where
  total : ℝ
  nearsighted : ℝ
  heavyPhoneUsers : ℝ
  nearsightedHeavyUsers : ℝ

/-- Conditions of the problem -/
def schoolConditions (p : SchoolPopulation) : Prop :=
  p.nearsighted = 0.4 * p.total ∧
  p.heavyPhoneUsers = 0.3 * p.total ∧
  p.nearsightedHeavyUsers = 0.5 * p.heavyPhoneUsers

/-- The probability of a student who uses their phone for no more than 2 hours per day being nearsighted -/
noncomputable def probabilityNearsightedLightUser (p : SchoolPopulation) : ℝ :=
  (p.nearsighted - p.nearsightedHeavyUsers) / (p.total - p.heavyPhoneUsers)

/-- The main theorem to be proved -/
theorem nearsighted_light_user_probability 
  (p : SchoolPopulation) (h : schoolConditions p) : 
  probabilityNearsightedLightUser p = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nearsighted_light_user_probability_l1000_100025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_is_two_l1000_100097

/-- Represents a sample category with its properties -/
structure SampleCategory where
  size : ℕ
  mean : ℝ
  variance : ℝ

/-- Calculates the variance of the total sample given two sample categories -/
noncomputable def totalSampleVariance (a b : SampleCategory) : ℝ :=
  let totalSize := (a.size + b.size : ℝ)
  let weightA := a.size / totalSize
  let weightB := b.size / totalSize
  let overallMean := (a.size * a.mean + b.size * b.mean) / totalSize
  weightA * (a.variance + (overallMean - a.mean)^2) + 
  weightB * (b.variance + (overallMean - b.mean)^2)

/-- Theorem stating that the variance of the total sample is 2 -/
theorem variance_is_two : 
  let a : SampleCategory := { size := 10, mean := 3.5, variance := 2 }
  let b : SampleCategory := { size := 30, mean := 5.5, variance := 1 }
  totalSampleVariance a b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_is_two_l1000_100097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_central_angle_approx_l1000_100039

/-- The central angle of a circular sector in degrees, given its perimeter and radius -/
noncomputable def central_angle_deg (perimeter radius : ℝ) : ℝ :=
  let arc_length := perimeter - 2 * radius
  let angle_rad := arc_length / radius
  angle_rad * (180 / Real.pi)

/-- Theorem stating that a circular sector with perimeter 83 cm and radius 14 cm has a central angle of approximately 225 degrees -/
theorem sector_central_angle_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |central_angle_deg 83 14 - 225| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_central_angle_approx_l1000_100039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_on_hyperbola_l1000_100061

/-- Given a hyperbola and a line intersecting its asymptotes, the product of
    the distances from the intersection point on the hyperbola to the
    intersection points with the asymptotes is constant. -/
theorem constant_product_on_hyperbola (a b : ℝ) (h_pos : a > 0 ∧ b > 0) (α : ℝ) :
  ∀ x₀ y₀ : ℝ, (x₀^2 / a^2) - (y₀^2 / b^2) = 1 →
  ∃ PQ PR : ℝ,
    PQ * PR = (a^2 * b^2) / |a^2 * (Real.sin α)^2 - b^2 * (Real.cos α)^2| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_product_on_hyperbola_l1000_100061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1000_100024

def is_valid_permutation (p q r s : ℕ) : Prop :=
  Multiset.ofList [p, q, r, s] = Multiset.ofList [2, 4, 6, 8]

def expression_value (p q r s : ℕ) : ℕ :=
  p * q + q * r + r * s + s * p

theorem max_expression_value :
  ∃ (p q r s : ℕ), is_valid_permutation p q r s ∧
    (∀ (a b c d : ℕ), is_valid_permutation a b c d →
      expression_value p q r s ≥ expression_value a b c d) ∧
    expression_value p q r s = 100 := by
  sorry

#check max_expression_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l1000_100024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_unit_vectors_l1000_100037

theorem max_value_unit_vectors (u v w : ℝ × ℝ × ℝ) 
  (hu : ‖u‖ = 1) (hv : ‖v‖ = 1) (hw : ‖w‖ = 1) : 
  ‖u - v‖^2 + ‖u - w‖^2 + ‖v - w‖^2 + ‖u + v + w‖^2 ≤ 12 ∧ 
  ∃ (u' v' w' : ℝ × ℝ × ℝ), ‖u'‖ = 1 ∧ ‖v'‖ = 1 ∧ ‖w'‖ = 1 ∧
    ‖u' - v'‖^2 + ‖u' - w'‖^2 + ‖v' - w'‖^2 + ‖u' + v' + w'‖^2 = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_unit_vectors_l1000_100037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rewind_time_correct_l1000_100032

/-- The time required to rewind a film from one reel to another. -/
noncomputable def rewindTime (a L S ω : ℝ) : ℝ :=
  (Real.pi / (S * ω)) * (Real.sqrt (a^2 + (4 * S * L / Real.pi)) - a)

/-- Theorem stating that the rewindTime function correctly calculates the time
    required to rewind a film under the given conditions. -/
theorem rewind_time_correct (a L S ω : ℝ) (ha : a > 0) (hL : L > 0) (hS : S > 0) (hω : ω > 0) :
  rewindTime a L S ω = (Real.pi / (S * ω)) * (Real.sqrt (a^2 + (4 * S * L / Real.pi)) - a) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rewind_time_correct_l1000_100032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_circles_divide_plane_l1000_100062

/-- 
Given n circles in a plane where:
- Each pair of circles intersects at two points
- Any three circles do not have a common point
This function represents the number of regions into which these circles divide the plane.
-/
def f (n : ℕ) : ℕ := n^2 - n + 2

/-- Two circles intersect at exactly two points -/
def CirclesIntersectTwice (i j : ℕ) : Prop := sorry

/-- Three circles do not have a common point -/
def NoCommonPoint (i j k : ℕ) : Prop := sorry

/-- The number of regions into which n circles divide the plane -/
def NumberOfRegions (n : ℕ) : ℕ := sorry

/-- 
Theorem stating that f(n) correctly represents the number of regions for n circles 
under the given conditions.
-/
theorem circles_divide_plane (n : ℕ) : 
  (∀ (i j : ℕ), i < n → j < n → i ≠ j → CirclesIntersectTwice i j) →
  (∀ (i j k : ℕ), i < n → j < n → k < n → i ≠ j → j ≠ k → i ≠ k → NoCommonPoint i j k) →
  NumberOfRegions n = f n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_circles_divide_plane_l1000_100062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1000_100074

/-- Definition of the focus of a parabola -/
noncomputable def focus_of_parabola (lhs rhs : ℝ → ℝ) : ℝ × ℝ := sorry

/-- The focus of a parabola y^2 = 4x has coordinates (1, 0) -/
theorem parabola_focus_coordinates :
  ∀ (x y : ℝ), y^2 = 4*x → (1, 0) = focus_of_parabola (λ y => y^2) (λ x => 4*x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l1000_100074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_birth_rate_l1000_100035

/-- Represents the average birth rate in a city --/
def average_birth_rate : ℕ → ℕ := sorry

/-- Theorem stating the conditions and the result to be proved --/
theorem city_birth_rate :
  ∀ (B : ℕ),
  (average_birth_rate B = B) →
  (∃ (death_rate : ℕ),
    death_rate = 3 ∧
    (B - death_rate) * 43200 = 172800) →
  B = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_birth_rate_l1000_100035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sin_to_cos_l1000_100048

-- Define the original function
noncomputable def f (x : ℝ) := Real.sin (-2 * x)

-- Define the shifted function
noncomputable def g (x : ℝ) := f (x + Real.pi / 4)

-- Define the expected result function
noncomputable def h (x : ℝ) := -Real.cos (2 * x)

-- Theorem statement
theorem shift_sin_to_cos :
  ∀ x : ℝ, g x = h x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_sin_to_cos_l1000_100048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_coprime_pairs_l1000_100067

theorem infinitely_many_coprime_pairs (m : ℤ) : 
  ∀ n : ℕ, ∃ x y : ℤ, 
    x ≠ y ∧ 
    (Nat.gcd x.natAbs y.natAbs = 1) ∧ 
    (∃ k : ℤ, y * k = x^2 + m) ∧
    (∃ l : ℤ, x * l = y^2 + m) ∧
    (x > n) ∧ (y > n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_coprime_pairs_l1000_100067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l1000_100086

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the triangle and its properties
structure Triangle where
  A : Point
  B : Point
  C : Point
  is_acute : Bool
  O : Point -- circumcenter
  H : Point -- orthocenter
  Ω : Circle -- circumcircle

-- Define the midpoints and the additional circle
def M (t : Triangle) : Point := sorry
def N (t : Triangle) : Point := sorry
def ω (t : Triangle) : Circle := sorry

-- Define membership for Point in Circle
def Point.mem (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

instance : Membership Point Circle where
  mem := Point.mem

-- Define the internally tangent property
def are_internally_tangent (c1 c2 : Circle) : Prop := sorry

-- State the theorem
theorem circles_internally_tangent (t : Triangle) 
  (h1 : t.is_acute = true)
  (h2 : M t ≠ N t ∧ M t ≠ t.O ∧ M t ≠ t.H)
  (h3 : N t ≠ t.O ∧ N t ≠ t.H)
  (h4 : t.O ≠ t.H)
  (h5 : (M t) ∈ (ω t) ∧ (N t) ∈ (ω t) ∧ t.O ∈ (ω t) ∧ t.H ∈ (ω t)) :
  are_internally_tangent (ω t) t.Ω := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l1000_100086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1000_100007

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x - Real.pi / 3) + 2 * Real.sin (3 * Real.pi / 2 - x) + 1

theorem f_properties :
  (∃ (k : ℤ), ∀ (x : ℝ), f (x + π/3 + k*π) = f (-x + π/3 + k*π)) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), f x ≠ a) ↔ (a < -1 ∨ a > 3)) ∧
  (∀ (x : ℝ), f x = 6/5 → Real.cos (2*x - Real.pi/3) = 49/50) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1000_100007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_completion_time_l1000_100071

/-- Represents the work rate of a person or group in terms of work units per hour -/
abbrev WorkRate := ℝ

/-- Represents the time taken to complete the work in hours -/
abbrev Time := ℝ

/-- The total amount of work to be done -/
def TotalWork : ℝ := 1

/-- A's work rate -/
noncomputable def rate_A : WorkRate := TotalWork / 4

/-- B's work rate -/
noncomputable def rate_B : WorkRate := TotalWork / 4

/-- B and C's combined work rate -/
noncomputable def rate_BC : WorkRate := TotalWork / 2

/-- C's work rate -/
noncomputable def rate_C : WorkRate := rate_BC - rate_B

/-- Time taken by A and C together to complete the work -/
noncomputable def time_AC : Time := TotalWork / (rate_A + rate_C)

theorem AC_completion_time :
  time_AC = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AC_completion_time_l1000_100071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c2_description_l1000_100028

/-- A function that represents the original curve C -/
def f : ℝ → ℝ := sorry

/-- The operation of reflecting a function about the line x = 1 -/
def reflect_about_x_eq_1 (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (2 - x)

/-- The operation of translating a function 2 units to the left -/
def translate_left_2 (g : ℝ → ℝ) : ℝ → ℝ := λ x => g (x + 2)

/-- The theorem stating that C₂ is described by y = f(1-x) -/
theorem c2_description : 
  translate_left_2 (reflect_about_x_eq_1 f) = λ x => f (1 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c2_description_l1000_100028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l1000_100088

/-- An arithmetic sequence with a common difference d ≠ 0, where a₁, a₃, a₄ form a geometric sequence -/
structure ArithmeticGeometricSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  h_d_neq_0 : d ≠ 0
  h_geometric : (a 3)^2 = a 1 * a 4

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticGeometricSequence) (n : ℕ) : ℚ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

/-- The main theorem -/
theorem arithmetic_geometric_ratio (seq : ArithmeticGeometricSequence) :
  (S seq 4 - S seq 2) / (S seq 5 - S seq 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_ratio_l1000_100088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_eight_l1000_100077

noncomputable def f (x : ℝ) : ℝ := 3 * x + 2

noncomputable def f_inv (x : ℝ) : ℝ := (x - 2) / 3

theorem sum_of_solutions_is_eight :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f_inv x = f (x⁻¹)) ∧ (S.sum id = 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_is_eight_l1000_100077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_ratio_l1000_100003

/-- Proves that for a garden with given dimensions and constraints, 
    the ratio of trellises area to non-tilled land area is 1:3 -/
theorem garden_area_ratio 
  (length : ℝ) 
  (width : ℝ) 
  (raised_bed_area : ℝ) 
  (h1 : length = 220)
  (h2 : width = 120)
  (h3 : raised_bed_area = 8800) : 
  (length * width / 2 - raised_bed_area) / (length * width / 2) = 1 / 3 := by
  -- Define total area
  let total_area := length * width
  -- Define non-tilled area
  let non_tilled_area := total_area / 2
  -- Define trellises area
  let trellises_area := non_tilled_area - raised_bed_area
  
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_area_ratio_l1000_100003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1000_100095

theorem problem_solution : ((2 : ℝ)^0 - 1 + 5^2 - 0)⁻¹ * 5 = 1/5 := by
  -- Evaluate the expression inside the parentheses
  have h1 : (2 : ℝ)^0 - 1 + 5^2 - 0 = 25 := by
    simp [Real.rpow_zero]
    norm_num

  -- Apply the reciprocal and multiply by 5
  calc
    ((2 : ℝ)^0 - 1 + 5^2 - 0)⁻¹ * 5 = 25⁻¹ * 5 := by rw [h1]
    _ = 1/25 * 5 := by simp
    _ = 1/5 := by norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1000_100095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_at_pi_div_6_l1000_100056

open Real Set

-- Define the function
noncomputable def f (x : ℝ) := (sin (3 * x)) ^ 2

-- Define the interval
def I : Set ℝ := Ioo 0 0.6

-- State the theorem
theorem f_has_max_at_pi_div_6 :
  ∃ (c : ℝ), c ∈ I ∧ c = π / 6 ∧ ∀ x ∈ I, f x ≤ f c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_max_at_pi_div_6_l1000_100056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_even_function_l1000_100005

open Real

-- Define the determinant operation
def det (a₁ a₂ a₃ a₄ : ℝ) : ℝ := a₁ * a₄ - a₂ * a₃

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := det (sqrt 3) (sin x) 1 (cos x)

-- Define the translated function g
noncomputable def g (n : ℝ) (x : ℝ) : ℝ := f (x + n)

-- Theorem statement
theorem min_translation_for_even_function :
  ∃ (n : ℝ), n > 0 ∧ 
  (∀ (x : ℝ), g n x = g n (-x)) ∧
  (∀ (m : ℝ), m > 0 ∧ (∀ (x : ℝ), g m x = g m (-x)) → m ≥ n) ∧
  n = 5 * π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_translation_for_even_function_l1000_100005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_squares_35th_row_l1000_100093

/-- Represents the color of a square -/
inductive Color
| Black
| White

/-- Represents a row in the stair-step figure -/
structure Row where
  n : ℕ                    -- The row number
  start_color : Color      -- The color of the starting square
  total_squares : ℕ        -- Total number of squares in the row
  black_squares : ℕ        -- Number of black squares in the row

/-- The stair-step figure -/
structure StairStep where
  rows : ℕ → Row           -- A function that returns the Row for a given row number

/-- Properties of the stair-step figure -/
class StairStepProperties (S : StairStep) where
  alternating_start : ∀ n : ℕ, (S.rows (n + 1)).start_color ≠ (S.rows n).start_color
  first_row_black : (S.rows 1).start_color = Color.Black
  end_white : ∀ n : ℕ, (S.rows n).total_squares % 2 = 0
  total_squares : ∀ n : ℕ, (S.rows n).total_squares = 2 * n

/-- The main theorem to prove -/
theorem black_squares_35th_row (S : StairStep) [StairStepProperties S] :
  (S.rows 35).black_squares = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_squares_35th_row_l1000_100093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_standing_time_l1000_100038

/-- Represents the time it takes Clea to ride down the escalator in different scenarios -/
structure EscalatorTime where
  nonOperating : ℚ  -- Time to walk down non-operating escalator
  operatingWalking : ℚ  -- Time to walk down operating escalator
  operatingStanding : ℚ  -- Time to stand on operating escalator

/-- Calculates the time it takes to stand on the operating escalator given the other times -/
def calculateStandingTime (times : EscalatorTime) : ℚ :=
  (times.nonOperating * times.operatingWalking) / (times.nonOperating - times.operatingWalking)

/-- Theorem stating that given the conditions, the time to stand on the operating escalator is 48 seconds -/
theorem escalator_standing_time (times : EscalatorTime) 
  (h1 : times.nonOperating = 80)
  (h2 : times.operatingWalking = 30) :
  calculateStandingTime times = 48 := by
  sorry

def main : IO Unit := do
  let result := calculateStandingTime { nonOperating := 80, operatingWalking := 30, operatingStanding := 0 }
  IO.println s!"The time to stand on the operating escalator is {result} seconds"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_standing_time_l1000_100038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_l1000_100029

/-- A lattice point in the xy-coordinate system is any point (x, y) where both x and y are integers. -/
def is_lattice_point (x y : ℤ) : Prop := true

/-- The line y = mx + 3 passes through no lattice point with 0 < x ≤ 50 -/
def no_lattice_point_on_line (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 50 → is_lattice_point x y → y ≠ (m * ↑x + 3).floor

/-- The maximum value of b such that y = mx + 3 passes through no lattice point
    with 0 < x ≤ 50 for all m where 1/3 < m < b, is 17/51 -/
theorem max_b_value : 
  (∃ b : ℚ, b = 17/51 ∧ 
    (∀ m : ℚ, 1/3 < m → m < b → no_lattice_point_on_line m) ∧
    (∀ b' : ℚ, b < b' → 
      ∃ m : ℚ, 1/3 < m ∧ m < b' ∧ ¬(no_lattice_point_on_line m))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_value_l1000_100029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_felix_l1000_100036

/-- The vertical distance between the midpoint of two points and a third point -/
noncomputable def vertical_distance_to_midpoint (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  let midpoint_y := (y1 + y2) / 2
  |y3 - midpoint_y|

theorem distance_to_felix : 
  vertical_distance_to_midpoint 8 (-15) 3 20 5 5 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_felix_l1000_100036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_perimeter_l1000_100016

/-- A rectangle inscribed in a circle with radius R -/
structure InscribedRectangle (R : ℝ) where
  -- We don't need to define the specific properties of the rectangle,
  -- as the problem only requires us to work with its midpoints

/-- The quadrilateral formed by joining the midpoints of the inscribed rectangle's sides -/
def MidpointQuadrilateral (R : ℝ) (rect : InscribedRectangle R) : Set (ℝ × ℝ) :=
  sorry

/-- The perimeter of a set of points in ℝ² -/
noncomputable def perimeter (p : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The perimeter of the quadrilateral formed by joining the midpoints
    of the sides of a rectangle inscribed in a circle of radius R is 4R -/
theorem midpoint_quadrilateral_perimeter (R : ℝ) (rect : InscribedRectangle R) :
  perimeter (MidpointQuadrilateral R rect) = 4 * R := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_perimeter_l1000_100016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_solution_l1000_100076

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x - a)

-- Define the inverse function f⁻¹
noncomputable def f_inv (a : ℝ) (x : ℝ) : ℝ := x^2 + a

-- Theorem statement
theorem inverse_function_solution 
  (a : ℝ) 
  (h1 : ∀ x ≥ 0, f a (f_inv a x) = x) 
  (h2 : ∀ x ≥ a, f_inv a (f a x) = x) 
  (h3 : f_inv a 0 = 1) :
  f_inv a 2 = 1 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_solution_l1000_100076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_change_theorem_l1000_100033

/-- Represents the change in revenue when tax and consumption change -/
noncomputable def revenueChange (taxRate : ℝ) (consumption : ℝ) : ℝ :=
  let newTaxRate := taxRate * (1 - 0.18)
  let newConsumption := consumption * (1 + 0.15)
  (newTaxRate * newConsumption - taxRate * consumption) / (taxRate * consumption)

/-- Theorem stating that the revenue change is -5.7% when tax decreases by 18% and consumption increases by 15% -/
theorem revenue_change_theorem (taxRate : ℝ) (consumption : ℝ) 
    (h1 : taxRate > 0) (h2 : consumption > 0) : 
  revenueChange taxRate consumption = -0.057 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_revenue_change_theorem_l1000_100033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l1000_100080

/-- The length of a platform given train speed and passing times -/
theorem platform_length
  (train_speed : ℝ)
  (platform_pass_time : ℝ)
  (man_pass_time : ℝ)
  (h1 : train_speed = 54)  -- train speed in km/hr
  (h2 : platform_pass_time = 25)  -- time to pass platform in seconds
  (h3 : man_pass_time = 20)  -- time to pass man in seconds
  : ℝ :=
by
  -- Convert train speed to m/s
  let train_speed_ms := train_speed * 1000 / 3600
  
  -- Calculate length of train
  let train_length := train_speed_ms * man_pass_time
  
  -- Calculate total distance covered when passing platform
  let total_distance := train_speed_ms * platform_pass_time
  
  -- Calculate platform length
  let platform_length := total_distance - train_length
  
  -- State the result
  have : platform_length = 75
  
  -- Skip the actual proof
  sorry

  -- Return the result
  exact platform_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l1000_100080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1000_100075

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (a * Real.sqrt (-((x+2)*(x-6)))) / (Real.sqrt (x+2) + Real.sqrt (6-x))

-- State the theorem
theorem f_properties (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) 6, f a x = f a (4-x)) ∧
  (a = 1 → ∀ x ∈ Set.Icc (-2) 6, f a x ≤ 1) ∧
  (a < 0 → ∀ x ∈ Set.Icc (-2) 6, a ≤ f a x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1000_100075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_l1000_100098

/-- The integral of sqrt(1 - x^2) from -1 to 0 equals π/4 -/
theorem integral_sqrt_one_minus_x_squared :
  ∫ x in (-1)..0, Real.sqrt (1 - x^2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_l1000_100098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1000_100006

/-- Given a triangle PQR where cos(3P - Q) + sin(P + Q) = 2 and PQ = 3, prove QR = 3√2 -/
theorem triangle_side_length (P Q R : ℝ) : 
  Real.cos (3 * P - Q) + Real.sin (P + Q) = 2 →
  3 = Real.sqrt ((P - Q)^2 + (P - R)^2) →
  Real.sqrt ((Q - R)^2) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1000_100006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l1000_100068

noncomputable def polar_distance (r1 r2 : ℝ) (θ1 θ2 : ℝ) : ℝ :=
  Real.sqrt (r1^2 + r2^2 - 2*r1*r2*(Real.cos (θ2 - θ1)))

theorem distance_between_polar_points :
  let r1 : ℝ := 1
  let r2 : ℝ := 3
  let θ1 : ℝ := π/6
  let θ2 : ℝ := 5*π/6
  polar_distance r1 r2 θ1 θ2 = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_polar_points_l1000_100068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_root_of_cubic_l1000_100034

-- Define the cubic polynomial
def cubic_poly (a b x : ℝ) : ℝ := a * x^3 + 4 * x^2 + b * x - 70

-- Define the complex root
def complex_root : ℂ := -2 - 3*Complex.I

theorem real_root_of_cubic (a b : ℝ) : 
  (∃ (z : ℂ), cubic_poly a b z.re = 0 ∧ z = complex_root) →
  ∃ (x : ℝ), cubic_poly a b x = 0 ∧ x = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_root_of_cubic_l1000_100034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solution_l1000_100089

theorem factorial_equation_solution : 
  ∃ k : ℚ, 6 * 9 * 2 * k = (8 : ℕ).factorial ∧ k = 1120 / 3 := by
  use 1120 / 3
  constructor
  · simp [Nat.factorial]
    norm_num
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_solution_l1000_100089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_period_is_five_years_l1000_100027

/-- Calculates the additional lending period given initial sum, amount after 2 years, and final amount -/
def calculate_additional_period (initial_sum : ℚ) (amount_after_2_years : ℚ) (final_amount : ℚ) : ℚ :=
  let interest_rate := (amount_after_2_years - initial_sum) / (2 * initial_sum)
  let total_interest := final_amount - initial_sum
  let additional_interest := total_interest - (amount_after_2_years - initial_sum)
  additional_interest / (interest_rate * initial_sum)

/-- Theorem stating that the additional lending period is approximately 5 years -/
theorem additional_period_is_five_years :
  let initial_sum : ℚ := 684
  let amount_after_2_years : ℚ := 780
  let final_amount : ℚ := 1020
  let additional_period := calculate_additional_period initial_sum amount_after_2_years final_amount
  ⌊additional_period⌋ = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_period_is_five_years_l1000_100027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l1000_100057

theorem max_a_value (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → 
    ((x₁^3 + a*x₁^2 + 2*x₁ - a^2) - (x₂^3 + a*x₂^2 + 2*x₂ - a^2)) / (x₁ - x₂) < 2) →
  a ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l1000_100057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_in_range_20_l1000_100004

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range n.succ)).card

/-- The set of numbers with the maximum number of divisors in the range 1 to 20 -/
def max_divisor_numbers : Finset ℕ :=
  Finset.filter (fun i => ∀ j ∈ Finset.range 20, num_divisors (i + 1) ≥ num_divisors (j + 1)) (Finset.range 20)

theorem max_divisors_in_range_20 :
  max_divisor_numbers = {11, 17, 19} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_divisors_in_range_20_l1000_100004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameter_l1000_100064

/-- A line parametrized by (a, -2) passes through the points (3, 4) and (-4, 1). -/
theorem line_parameter (a : ℚ) : 
  (∃ t₁ t₂ : ℚ, 
    (Vector.cons 3 (Vector.cons 4 Vector.nil)) = Vector.cons (t₁ * a) (Vector.cons (t₁ * (-2)) Vector.nil) ∧ 
    (Vector.cons (-4) (Vector.cons 1 Vector.nil)) = Vector.cons (t₂ * a) (Vector.cons (t₂ * (-2)) Vector.nil)) → 
  a = -14/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parameter_l1000_100064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1000_100099

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 
                     Real.sin (x + Real.pi / 4) * Real.sin (x - Real.pi / 4)

-- State the theorem
theorem f_properties :
  -- 1. The smallest positive period of f is π
  (∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ 
   (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧ T = Real.pi) ∧
  -- 2. f is monotonically increasing on [-π/6, π/3]
  (∀ x y, -Real.pi/6 ≤ x ∧ x < y ∧ y ≤ Real.pi/3 → f x < f y) ∧
  -- 3. For any x₀ where 0 ≤ x₀ ≤ π/2 and f(x₀) = 0, cos(2x₀) = (3√5 + 1) / 8
  (∀ x₀, 0 ≤ x₀ ∧ x₀ ≤ Real.pi/2 ∧ f x₀ = 0 → 
    Real.cos (2 * x₀) = (3 * Real.sqrt 5 + 1) / 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1000_100099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l1000_100001

/-- Triangle properties -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ

/-- Point inside triangle -/
structure PointInTriangle where
  d_a : ℝ
  d_b : ℝ
  d_c : ℝ

/-- Theorem for triangle inequalities -/
theorem triangle_inequalities (t : Triangle) (p : PointInTriangle) :
  min t.h_a (min t.h_b t.h_c) ≤ p.d_a + p.d_b + p.d_c ∧
  p.d_a + p.d_b + p.d_c ≤ max t.h_a (max t.h_b t.h_c) ∧
  p.d_a * p.d_b * p.d_c ≤ (8 * t.S^3) / (27 * t.a * t.b * t.c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l1000_100001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tricycle_count_l1000_100017

/-- Represents the number of children riding bicycles -/
def bicycles : ℕ := sorry

/-- Represents the number of children riding tricycles -/
def tricycles : ℕ := sorry

/-- The total number of children -/
def total_children : ℕ := 10

/-- The total number of wheels -/
def total_wheels : ℕ := 26

/-- Theorem stating that given the conditions, there are 6 tricycles -/
theorem tricycle_count :
  bicycles + tricycles = total_children ∧
  2 * bicycles + 3 * tricycles = total_wheels →
  tricycles = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tricycle_count_l1000_100017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_multiple_solutions_l1000_100054

theorem smallest_n_with_multiple_solutions :
  ∃ (n : ℕ), n ≠ 2004 ∧
  (∀ m : ℕ, m < n → m = 2004 ∨
    ¬∃ (f : MvPolynomial (Fin 1) ℤ) (a : ℤ),
      (∃ x : ℤ, MvPolynomial.eval (fun _ => x) f = 2004) ∧
      (∃ (s : Finset ℤ), s.card ≥ 2004 ∧ ∀ x ∈ s, MvPolynomial.eval (fun _ => x) f = m)) ∧
  (∃ (f : MvPolynomial (Fin 1) ℤ) (a : ℤ),
    (∃ x : ℤ, MvPolynomial.eval (fun _ => x) f = 2004) ∧
    (∃ (s : Finset ℤ), s.card ≥ 2004 ∧ ∀ x ∈ s, MvPolynomial.eval (fun _ => x) f = n)) ∧
  n = (1002 : ℕ).factorial ^ 2 + 2004 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_multiple_solutions_l1000_100054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l1000_100060

/-- The radius of the circumscribed circle of a triangle with side lengths 7, 5, and 3 -/
noncomputable def circumradius : ℝ := (7 * Real.sqrt 3) / 3

/-- Theorem: The radius of the circumscribed circle of a triangle with side lengths 7, 5, and 3 is (7 * √3) / 3 -/
theorem triangle_circumradius :
  let a : ℝ := 7
  let b : ℝ := 5
  let c : ℝ := 3
  circumradius = (a * Real.sqrt 3) / 3 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumradius_l1000_100060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterizeSolutions_l1000_100055

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Constructs a 7-digit number from given digits -/
def constructNumber (a b c : Digit) : ℕ :=
  5 * 1000000 + a.val * 100000 + b.val * 10000 + c.val * 1000 + 37 * 100 + c.val * 10 + 2

/-- Checks if a triple (a,b,c) forms a valid solution -/
def isValidSolution (a b c : Digit) : Prop :=
  let n := constructNumber a b c
  n > 4999999 ∧ n < 6000000 ∧ n % 792 = 0

/-- Main theorem: characterizes all valid solutions -/
theorem characterizeSolutions :
  ∀ (a b c : Digit),
    isValidSolution a b c ↔ 
      ((a = ⟨0, by norm_num⟩ ∧ b = ⟨5, by norm_num⟩ ∧ c = ⟨5, by norm_num⟩) ∨
       (a = ⟨4, by norm_num⟩ ∧ b = ⟨5, by norm_num⟩ ∧ c = ⟨1, by norm_num⟩) ∨
       (a = ⟨6, by norm_num⟩ ∧ b = ⟨4, by norm_num⟩ ∧ c = ⟨9, by norm_num⟩)) := by
  sorry

#eval constructNumber ⟨0, by norm_num⟩ ⟨5, by norm_num⟩ ⟨5, by norm_num⟩
#eval constructNumber ⟨4, by norm_num⟩ ⟨5, by norm_num⟩ ⟨1, by norm_num⟩
#eval constructNumber ⟨6, by norm_num⟩ ⟨4, by norm_num⟩ ⟨9, by norm_num⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterizeSolutions_l1000_100055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_chord_length_l1000_100082

open Real

-- Define the parametric equations of line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (t * cos α, 1 + t * sin α)

-- Define the polar equation of circle C
noncomputable def circle_C (θ : ℝ) : ℝ := 2 * cos θ

-- Define the constraint on α
def α_constraint (α : ℝ) : Prop := π / 2 ≤ α ∧ α < π

-- Define the number of intersection points
noncomputable def num_intersections (α : ℝ) : ℕ :=
  if α = π / 2 then 1 else 2

-- Define the trajectory of point P
noncomputable def trajectory_P (θ : ℝ) : ℝ := sin θ

-- Define the length of the chord
noncomputable def chord_length : ℝ := 2 * sqrt 5 / 5

-- Theorem statement
theorem intersection_and_chord_length (α : ℝ) (h : α_constraint α) :
  (num_intersections α = 1 ∨ num_intersections α = 2) ∧
  ∃ θ, 0 ≤ θ ∧ θ < π / 2 ∧ trajectory_P θ = circle_C θ ∧ trajectory_P θ = chord_length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_chord_length_l1000_100082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1000_100050

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi/2 - x) * Real.sin x - Real.sqrt 3 * (Real.cos x)^2 + Real.sqrt 3 / 2

theorem f_properties :
  -- The smallest positive period of f is π
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
    (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi) ∧
  -- The minimum value of f on [π/6, 5π/6] is -√3/2
  (∀ (x : ℝ), Real.pi/6 ≤ x ∧ x ≤ 5*Real.pi/6 → f x ≥ -Real.sqrt 3 / 2) ∧
  (∃ (x : ℝ), Real.pi/6 ≤ x ∧ x ≤ 5*Real.pi/6 ∧ f x = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1000_100050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_rounded_to_nearest_is_three_l1000_100094

-- Define the rounding function for rationals
def round_to_nearest (x : ℚ) : ℤ :=
  if x - ↑⌊x⌋ < 1/2 then ⌊x⌋ else ⌈x⌉

-- Notation for rounding
notation "⟨" x "⟩" => round_to_nearest x

-- Theorem statement
theorem pi_rounded_to_nearest_is_three :
  ∃ (q : ℚ), (q : ℝ) = Real.pi ∧ ⟨q⟩ = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_rounded_to_nearest_is_three_l1000_100094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_addition_difference_l1000_100091

/-- Represents a single-digit integer in base 4 -/
def SingleDigitBase4 : Type := { n : ℕ // n < 4 }

/-- Addition in base 4 -/
def addBase4 (a b : SingleDigitBase4) : SingleDigitBase4 :=
  ⟨(a.val + b.val) % 4, by sorry⟩

/-- Conversion from base 10 to base 4 -/
def toBase4 (n : ℕ) : SingleDigitBase4 :=
  ⟨n % 4, by sorry⟩

/-- The main theorem -/
theorem base4_addition_difference (C D : SingleDigitBase4) :
  (addBase4 (addBase4 (toBase4 (100 * D.val + 10 * D.val + C.val))
                      (toBase4 (10 * 3 + 2 * 1 + D.val)))
            (toBase4 (100 * C.val + 2 * 10 + 3)))
  = (toBase4 (100 * C.val + 2 * 10 + 3 * 1))
  →
  toBase4 (Int.natAbs (D.val - C.val)) = toBase4 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base4_addition_difference_l1000_100091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1000_100051

noncomputable def point1 : ℚ × ℚ := (-3/2, -7/2)
noncomputable def point2 : ℚ × ℚ := (5/2, -11/2)

noncomputable def distance (p1 p2 : ℚ × ℚ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points :
  distance point1 point2 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1000_100051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_of_sequence_l1000_100085

noncomputable def a : ℕ → ℝ
  | 0 => 1/2
  | (n+1) => 1 + (a n - 1)^2

theorem infinite_product_of_sequence : 
  ∏' i, a i = 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_of_sequence_l1000_100085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_calculation_l1000_100008

/-- Represents the scale of a map as a ratio of 1 to some number -/
structure MapScale where
  ratio : ℕ
  scale_positive : ratio > 0

/-- Calculates the actual distance given a map distance and scale -/
noncomputable def actual_distance (map_distance : ℝ) (scale : MapScale) : ℝ :=
  map_distance * scale.ratio

/-- Converts a distance in centimeters to kilometers -/
noncomputable def cm_to_km (distance_cm : ℝ) : ℝ :=
  distance_cm / 100000

theorem map_distance_calculation (map_distance : ℝ) (scale : MapScale) 
  (h1 : map_distance = 2)
  (h2 : scale.ratio = 600000) :
  cm_to_km (actual_distance map_distance scale) = 12 := by
  sorry

#check map_distance_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_distance_calculation_l1000_100008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_percentage_l1000_100065

/-- Calculates the gain percentage for a dishonest shopkeeper --/
theorem shopkeeper_gain_percentage 
  (false_weight : ℝ) 
  (true_weight : ℝ) 
  (h1 : false_weight = 930) 
  (h2 : true_weight = 1000) :
  ∃ gain_percentage : ℝ, 
    let gain := true_weight - false_weight
    gain_percentage = (gain / false_weight) * 100 ∧
    abs (gain_percentage - 7.53) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_percentage_l1000_100065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cone_l1000_100084

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.baseRadius^2 * c.height

/-- Calculates the height of water in a cone given a fill percentage -/
noncomputable def waterHeight (c : Cone) (fillPercentage : ℝ) : ℝ :=
  c.height * (fillPercentage^(1/3))

theorem water_height_in_cone (c : Cone) (h1 : c.baseRadius = 12) (h2 : c.height = 72) :
  waterHeight c 0.4 = 36 * (16/5)^(1/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_height_in_cone_l1000_100084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_max_f_l1000_100070

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) * Real.cos (ω * x) - Real.sqrt 3 * (Real.cos (ω * x))^2

theorem sum_of_min_max_f (ω : ℝ) (x₁ x₂ : ℝ) (h_ω : ω > 0) 
  (h_zeros : f ω x₁ + (2 + Real.sqrt 3) / 2 = 0 ∧ f ω x₂ + (2 + Real.sqrt 3) / 2 = 0)
  (h_min_dist : |x₁ - x₂| = π) :
  ∃ (min_f max_f : ℝ), 
    (∀ x ∈ Set.Icc 0 (7 * π / 12), min_f ≤ f ω x ∧ f ω x ≤ max_f) ∧
    min_f + max_f = (2 - 3 * Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_max_f_l1000_100070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_l1000_100063

open Real

-- Define the solution set
noncomputable def solution_set : Set ℝ :=
  {x | ∃ k : ℤ, -π/6 + k*π ≤ x ∧ x < π/2 + k*π}

-- State the theorem
theorem tan_inequality_solution :
  {x : ℝ | tan x ≥ -Real.sqrt 3 / 3} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_solution_l1000_100063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_d_value_l1000_100031

theorem min_d_value (a b c d : ℕ+) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (h4 : ∃! (x y : ℝ), 3 * x + y = 3000 ∧ y = |x - (↑a : ℝ)| + |x - (↑b : ℝ)| + |x - (↑c : ℝ)| + |x - (↑d : ℝ)|) :
  d = 2991 ∧ ∀ d' : ℕ+, (∃ a' b' c' : ℕ+, a' < b' ∧ b' < c' ∧ c' < d' ∧
    ∃! (x y : ℝ), 3 * x + y = 3000 ∧ y = |x - (↑a' : ℝ)| + |x - (↑b' : ℝ)| + |x - (↑c' : ℝ)| + |x - (↑d' : ℝ)|) →
    d' ≥ 2991 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_d_value_l1000_100031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_equations_l1000_100045

-- Define a line with y-intercept and inclination angle
def line_equation (y_intercept : ℝ) (inclination_angle : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ y = (Real.tan inclination_angle) * x + y_intercept

-- Define a circle with center and tangent to y-axis
def circle_equation (center_x center_y : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (x - center_x)^2 + (y - center_y)^2 = center_x^2

theorem line_and_circle_equations :
  (∀ x y, line_equation 2 (π/4) x y ↔ y = x + 2) ∧
  (∀ x y, circle_equation (-2) 3 x y ↔ (x + 2)^2 + (y - 3)^2 = 4) := by
  sorry

#check line_and_circle_equations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_equations_l1000_100045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_value_l1000_100087

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0) -/
def Hyperbola (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

/-- The right focus of the hyperbola -/
noncomputable def RightFocus (a b : ℝ) : ℝ × ℝ :=
  (Real.sqrt (a^2 + b^2), 0)

/-- The center of the hyperbola -/
def Center : ℝ × ℝ := (0, 0)

/-- An asymptote of the hyperbola -/
noncomputable def Asymptote (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sqrt m * p.1}

/-- Condition for an equilateral triangle -/
def IsEquilateralTriangle (A O F : ℝ × ℝ) : Prop :=
  dist A O = dist O F ∧ dist A F = dist O F

theorem hyperbola_asymptote_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ (A : ℝ × ℝ) (m : ℝ),
    A ∈ Hyperbola a b ∧
    Asymptote m ∩ Hyperbola a b ≠ ∅ ∧
    IsEquilateralTriangle A Center (RightFocus a b) →
    m = 3 + 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_value_l1000_100087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1000_100072

def s (n : ℕ) (l : ℝ) : ℝ := 3^n * (l - n) - 6

def a (n : ℕ) (l : ℝ) : ℝ := s n l - s (n-1) l

def monotonic_decreasing (f : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → f m < f n

theorem lambda_range (l : ℝ) :
  (∀ n : ℕ, s n l = 3^n * (l - n) - 6) →
  (monotonic_decreasing (a · l)) →
  l < 2 ∧ ∀ x < 2, x < l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l1000_100072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1000_100079

/-- Definition of the function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 1/a| + |x - a|

/-- Theorem stating the properties of function f -/
theorem f_properties (a : ℝ) (h : a > 0) :
  (∀ x, f a x ≥ 2) ∧
  (f a 3 < 5 ↔ (1 + Real.sqrt 5) / 2 < a ∧ a < (5 + Real.sqrt 21) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1000_100079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_positive_multiply_positive_l1000_100013

-- Define the calculator operations
noncomputable def add (x y : ℝ) : ℝ := x + y
noncomputable def subtract (x y : ℝ) : ℝ := x - y
noncomputable def reciprocal (x : ℝ) : ℝ := 1 / x

-- Define the number of operations for squaring and multiplying
def square_ops : ℕ := 6
def multiply_ops : ℕ := 20

-- Theorem for squaring a positive number
theorem square_positive (x : ℝ) (h : x > 0) :
  ∃ (f : ℕ → ℝ), f square_ops = x^2 ∧
  ∀ n < square_ops, f (n + 1) = add (f n) (f n) ∨
                    f (n + 1) = subtract (f n) (f n) ∨
                    f (n + 1) = reciprocal (f n) := by
  sorry

-- Theorem for multiplying two positive numbers
theorem multiply_positive (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ (f : ℕ → ℝ), f multiply_ops = x * y ∧
  ∀ n < multiply_ops, f (n + 1) = add (f n) (f n) ∨
                      f (n + 1) = subtract (f n) (f n) ∨
                      f (n + 1) = reciprocal (f n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_positive_multiply_positive_l1000_100013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_sequence_sum_l1000_100010

/-- Represents a sequence of 2004 positive integers. -/
def Sequence := Fin 2004 → ℕ

/-- Checks if a sequence satisfies the required conditions. -/
def ValidSequence (seq : Sequence) : Prop :=
  (∀ i j : Fin 2004, i < j → seq i < seq j) ∧
  (∀ i : Fin 2003, seq i ∣ seq (i + 1))

/-- The main theorem to be proved. -/
theorem exists_valid_sequence_sum :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N →
    ∃ seq : Sequence, n = (Finset.univ.sum seq) ∧ ValidSequence seq :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_valid_sequence_sum_l1000_100010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_theorem_l1000_100023

/-- Represents the number of students in each category -/
structure StudentCounts where
  total : ℕ
  vocational : ℕ
  undergraduate : ℕ
  graduate : ℕ

/-- Represents the number of sampled students in each category -/
structure SampledCounts where
  vocational : ℕ
  undergraduate : ℕ
  graduate : ℕ
deriving Repr

/-- Calculates the correct sample sizes for a stratified sampling -/
def calculateSampleSizes (counts : StudentCounts) (sampledVocational : ℕ) : SampledCounts :=
  { vocational := sampledVocational,
    undergraduate := (sampledVocational * counts.undergraduate) / counts.vocational,
    graduate := (sampledVocational * counts.graduate) / counts.vocational }

theorem stratified_sampling_theorem (counts : StudentCounts) (sampledVocational : ℕ) :
  counts.total = 16050 →
  counts.vocational = 4500 →
  counts.undergraduate = 9750 →
  counts.graduate = 1800 →
  sampledVocational = 60 →
  let sampledCounts := calculateSampleSizes counts sampledVocational
  sampledCounts.undergraduate = 130 ∧ sampledCounts.graduate = 24 := by
  sorry

#eval calculateSampleSizes 
  { total := 16050, vocational := 4500, undergraduate := 9750, graduate := 1800 } 
  60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_theorem_l1000_100023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiple_with_odd_digit_sum_l1000_100078

/-- Sum of digits of a natural number in base 10 --/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem exists_multiple_with_odd_digit_sum (M : ℕ) : 
  ∃ k : ℕ, (k % M = 0) ∧ (Odd (sum_of_digits k)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_multiple_with_odd_digit_sum_l1000_100078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_squares_area_l1000_100041

/-- Represents a square flag with a symmetric red cross and blue squares -/
structure MyFlag where
  side : ℝ
  cross_area_ratio : ℝ
  blue_squares_area_ratio : ℝ

/-- The flag satisfies the given conditions -/
def flag_conditions (f : MyFlag) : Prop :=
  f.side > 0 ∧
  f.cross_area_ratio = 0.4 ∧
  f.blue_squares_area_ratio > 0 ∧
  f.blue_squares_area_ratio < f.cross_area_ratio

theorem blue_squares_area (f : MyFlag) 
  (h : flag_conditions f) : 
  f.blue_squares_area_ratio = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_squares_area_l1000_100041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_circle_EQ_length_l1000_100000

/-- Represents a trapezoid EFGH with a circle centered at Q on EF and tangent to FG and HE -/
structure TrapezoidWithCircle where
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of side FG -/
  fg : ℝ
  /-- Length of side GH -/
  gh : ℝ
  /-- Length of side HE -/
  he : ℝ
  /-- EF is parallel to GH -/
  ef_parallel_gh : True
  /-- Circle with center Q on EF is tangent to FG and HE -/
  circle_tangent : True

/-- The length of EQ in the trapezoid with inscribed circle -/
noncomputable def length_EQ (t : TrapezoidWithCircle) : ℝ := 336 / 5

theorem trapezoid_circle_EQ_length (t : TrapezoidWithCircle) 
  (h1 : t.ef = 105) 
  (h2 : t.fg = 45) 
  (h3 : t.gh = 21) 
  (h4 : t.he = 80) : 
  length_EQ t = 336 / 5 := by
  sorry

#check trapezoid_circle_EQ_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_circle_EQ_length_l1000_100000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_double_angle_l1000_100019

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c = 5, B = 2π/3, and the area of triangle ABC is 15√3/4,
    then cos(2A) = 71/98 -/
theorem triangle_cosine_double_angle (a b c A B C : ℝ) : 
  c = 5 → 
  B = 2 * Real.pi / 3 →
  (1/2) * a * c * Real.sin B = 15 * Real.sqrt 3 / 4 →
  Real.cos (2 * A) = 71 / 98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_double_angle_l1000_100019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l1000_100069

/-- The angle of the minute hand on a clock at a given time -/
noncomputable def minute_hand_angle (minutes : ℕ) : ℝ :=
  (minutes % 60) * 6

/-- The angle of the hour hand on a clock at a given time -/
noncomputable def hour_hand_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  (hours % 12) * 30 + (minutes % 60) * 0.5

/-- The smaller angle between two angles on a circle -/
noncomputable def smaller_angle (a b : ℝ) : ℝ :=
  min (abs (a - b)) (360 - abs (a - b))

theorem clock_angle_at_3_40 :
  smaller_angle (hour_hand_angle 15 40) (minute_hand_angle 40) = 130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l1000_100069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1000_100021

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

/-- The derivative of f(x) -/
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem monotonic_decreasing_interval (x : ℝ) :
  x > 0 → (∀ y ∈ Set.Ioo 0 (Real.exp (-1)), f' y < 0) ∧
          (∀ z ∉ Set.Ioo 0 (Real.exp (-1)), z > 0 → f' z ≥ 0) :=
by sorry

#check monotonic_decreasing_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l1000_100021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_median_soda_distribution_l1000_100044

/-- Represents the distribution of soda cans to customers -/
structure SodaDistribution where
  total_cans : ℕ
  total_customers : ℕ
  cans_per_customer : Fin total_customers → ℕ
  h_total : (Finset.sum Finset.univ (λ i => cans_per_customer i)) = total_cans
  h_min_one : ∀ i, cans_per_customer i ≥ 1

/-- The median of a finite sequence of natural numbers -/
noncomputable def median (n : ℕ) (s : Fin n → ℕ) : ℚ :=
  sorry

/-- The maximum possible median for a given SodaDistribution -/
noncomputable def max_median (d : SodaDistribution) : ℚ :=
  sorry

theorem max_median_soda_distribution :
  ∀ d : SodaDistribution,
  d.total_cans = 300 ∧ d.total_customers = 120 →
  max_median d = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_median_soda_distribution_l1000_100044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monthly_donation_is_70_l1000_100018

/-- Represents Maxim Andreevich's daily schedule and finances --/
structure DailySchedule where
  work_hours : ℕ
  rest_hours : ℕ
  hobby_hours : ℕ
  sleep_hours : ℕ := 8
  lesson_cost : ℕ := 3
  charity_donation : ℚ := rest_hours / 3

/-- Represents Maxim Andreevich's monthly finances --/
structure MonthlyFinances where
  working_days : ℕ := 21
  investment_income : ℕ := 14
  household_expenses : ℕ := 70
  daily_schedules : Fin 21 → DailySchedule

/-- The maximum amount Maxim Andreevich can donate monthly to charity --/
noncomputable def max_monthly_donation (finances : MonthlyFinances) : ℚ :=
  (finances.daily_schedules 0).charity_donation * finances.working_days

/-- Theorem stating the maximum monthly donation --/
theorem max_monthly_donation_is_70 (finances : MonthlyFinances) :
  max_monthly_donation finances = 70 := by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monthly_donation_is_70_l1000_100018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_freight_train_passing_time_l1000_100081

/-- Calculates the time for a freight train to pass a passenger train -/
noncomputable def time_to_pass (freight_length : ℝ) (freight_speed : ℝ) (passenger_length : ℝ) (passenger_speed : ℝ) : ℝ :=
  (freight_length + passenger_length) / (freight_speed - passenger_speed)

/-- Converts speed from km/h to m/s -/
noncomputable def kmph_to_mps (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

theorem freight_train_passing_time :
  let freight_length : ℝ := 550
  let freight_speed : ℝ := kmph_to_mps 90
  let passenger_length : ℝ := 350
  let passenger_speed : ℝ := kmph_to_mps 75
  let passing_time := time_to_pass freight_length freight_speed passenger_length passenger_speed
  ∃ ε > 0, |passing_time - 215.82| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_freight_train_passing_time_l1000_100081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_proof_l1000_100047

noncomputable def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def geometric_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_proof :
  ∀ (a : ℕ → ℝ),
  a 1 = 2 →
  a 4 = -2 →
  (∀ n : ℕ, a n = geometric_sequence 2 (-1) n) ∧
  geometric_sum 2 (-1) 9 = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_proof_l1000_100047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1000_100058

-- Define the function f(x) = 2^x + 3x
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + 3*x

-- State the theorem
theorem zero_in_interval :
  (∀ x y : ℝ, x < y → f x < f y) →  -- f is monotonically increasing
  ContinuousOn f (Set.univ : Set ℝ) →  -- f is continuous over ℝ
  ∃ c ∈ Set.Ioo (-1) 0, f c = 0 -- there exists a zero in the open interval (-1, 0)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1000_100058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_theorem_l1000_100090

/-- The curve function -/
noncomputable def curve (y : ℝ) : ℝ := 0.25 * y^2 - 0.5 * Real.log y

/-- The arc length of the curve between two points -/
noncomputable def arcLength (a b : ℝ) : ℝ :=
  ∫ y in a..b, Real.sqrt (1 + ((0.5 * y - 0.5 / y)^2))

/-- Theorem stating the arc length of the curve between y = 1 and y = 1.5 -/
theorem arc_length_theorem :
  arcLength 1 1.5 = 0.3125 + 0.5 * Real.log 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_theorem_l1000_100090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_area_l1000_100030

noncomputable def A : ℝ × ℝ × ℝ := (-1, 1, 2)
noncomputable def B : ℝ × ℝ × ℝ := (1, 2, 3)
noncomputable def C (t : ℝ) : ℝ × ℝ × ℝ := (t, 2, 2)

noncomputable def triangle_area (p q r : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := p
  let (x₂, y₂, z₂) := q
  let (x₃, y₃, z₃) := r
  (1/2) * Real.sqrt (
    ((y₂ - y₁) * (z₃ - z₁) - (z₂ - z₁) * (y₃ - y₁))^2 +
    ((z₂ - z₁) * (x₃ - x₁) - (x₂ - x₁) * (z₃ - z₁))^2 +
    ((x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁))^2
  )

theorem smallest_triangle_area :
  ∃ (min_area : ℝ), min_area = (3/2) ∧
  ∀ (t : ℝ), triangle_area A B (C t) ≥ min_area := by
  sorry

#check smallest_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_area_l1000_100030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_rotation_l1000_100009

-- Define the original function
noncomputable def f (x : ℝ) := Real.log x / Real.log 2

-- Define the rotation transformation
def rotate_180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- State the theorem
theorem log_rotation (x : ℝ) (h : x > 0) :
  let (x', y') := rotate_180 (x, f x)
  y' = -f (-x') := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_rotation_l1000_100009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_roll_probability_l1000_100073

/-- The number of sides on each die -/
def sides : ℕ := 24

/-- The probability that the maximum of two dice rolls is at least N -/
noncomputable def prob_max_at_least (N : ℕ) : ℝ := 1 - ((N - 1) / sides) ^ 2

/-- The largest integer N such that the probability of the maximum roll being at least N is greater than 0.5 -/
def largest_N : ℕ := 17

/-- Theorem stating the properties of largest_N -/
theorem dice_roll_probability :
  largest_N ≤ sides ∧
  prob_max_at_least largest_N > 0.5 ∧
  ∀ m : ℕ, m > largest_N → m ≤ sides → prob_max_at_least m ≤ 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_roll_probability_l1000_100073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_cosines_l1000_100053

-- Define the function f
noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x)

-- State the theorem
theorem max_sum_of_cosines (a b c : ℝ) 
  (h : ∀ x : ℝ, f a b c x ≥ -1) : 
  a + b + c ≤ 3 ∧ ∃ a' b' c' : ℝ, a' + b' + c' = 3 ∧ ∀ x : ℝ, f a' b' c' x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_cosines_l1000_100053
