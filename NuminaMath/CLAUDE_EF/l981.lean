import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_implies_a_plus_b_l981_98129

/-- The function f(x) -/
noncomputable def f (a b x : ℝ) : ℝ := (a + Real.sin x) / (2 + Real.cos x) + b * Real.tan x

/-- The theorem statement -/
theorem sum_of_max_min_implies_a_plus_b (a b : ℝ) :
  (∃ (max min : ℝ), (∀ x, f a b x ≤ max) ∧ 
                    (∀ x, min ≤ f a b x) ∧ 
                    (max + min = 4)) →
  a + b = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_implies_a_plus_b_l981_98129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_midpoint_locus_l981_98198

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Define the line l
def line_l (m x y : ℝ) : Prop := m * x - y + 1 + 2 * m = 0

-- Define the locus of midpoint M
def locus_M (x y : ℝ) : Prop := (x + 2)^2 + (y - 1/2)^2 = 1/4 ∧ y ≠ 0

-- Theorem statement
theorem circle_line_intersection_and_midpoint_locus :
  ∀ m : ℝ,
  (∃ x y : ℝ, circle_C x y ∧ line_l m x y) ∧
  (∀ x y : ℝ, (∃ x1 y1 x2 y2 : ℝ, 
    circle_C x1 y1 ∧ circle_C x2 y2 ∧ 
    line_l m x1 y1 ∧ line_l m x2 y2 ∧
    x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2) → 
  locus_M x y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_midpoint_locus_l981_98198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l981_98191

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ m : ℝ, (∀ x ∈ Set.Icc 1 5, f x > m) → m < 1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l981_98191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_position_l981_98137

def complex_OA (m : ℝ) : ℂ := (m^2 - 8*m + 15) + (m^2 + m - 12)*Complex.I

theorem point_A_position (m : ℝ) :
  (((complex_OA m).re = 0 ∧ (complex_OA m).im ≠ 0) ↔ m = 5) ∧
  (((complex_OA m).re > 0 ∧ (complex_OA m).im < 0) ↔ -4 < m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_A_position_l981_98137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l981_98156

/-- The distance from a point (x, y) to a point (a, b) -/
noncomputable def distance (x y a b : ℝ) : ℝ := 
  Real.sqrt ((x - a)^2 + (y - b)^2)

/-- The distance from a point (x, y) to a vertical line x = a -/
def distanceToVerticalLine (x y a : ℝ) : ℝ := 
  abs (x - a)

/-- Theorem: Points satisfying the given condition form a parabola y^2 = -12x -/
theorem parabola_equation (x y : ℝ) :
  distance x y (-3) 0 = distanceToVerticalLine x y 2 + 1 →
  y^2 = -12 * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l981_98156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l981_98133

theorem tv_price_change (P : ℝ) (h : P > 0) : 
  (P * (1 - 0.2)) * (1 + 0.45) = P * 1.16 := by
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_price_change_l981_98133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parallelogram_area_l981_98108

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  -- One diagonal is equal to a side and is 4 units long
  diagonal1 : ℝ
  diagonal1_eq_side : diagonal1 = 4
  -- The other diagonal bisects two angles of the parallelogram
  bisects_angles : Bool

/-- The area of a special parallelogram is 8√3 -/
theorem special_parallelogram_area (p : SpecialParallelogram) : 
  ∃ (area : ℝ), area = 8 * Real.sqrt 3 := by
  sorry

#check special_parallelogram_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_parallelogram_area_l981_98108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_2_8_plus_5_5_l981_98115

theorem greatest_prime_factor_of_2_8_plus_5_5 :
  (Nat.factors (2^8 + 5^5)).maximum? = some 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_2_8_plus_5_5_l981_98115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l981_98107

/-- The ellipse C defined by x²/3 + y² = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}

/-- The left focus of the ellipse C -/
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)

/-- The right focus of the ellipse C -/
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 2, 0)

/-- The line y = x + m -/
def line (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + m}

/-- Intersection points of the line and the ellipse -/
def intersection (m : ℝ) : Set (ℝ × ℝ) := C ∩ line m

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_intersection_theorem (m : ℝ) 
  (hA : A ∈ intersection m) (hB : B ∈ intersection m) (hAB : A ≠ B) :
  (∀ (A B : ℝ × ℝ), triangleArea F₁ A B = 2 * triangleArea F₂ A B) → 
  m = -Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l981_98107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_ball_on_torus_l981_98155

/-- Represents a torus with given inner and outer radii -/
structure Torus where
  inner_radius : ℝ
  outer_radius : ℝ

/-- Represents a spherical ball with a given radius -/
structure SphericalBall where
  radius : ℝ

/-- Calculates the radius of the largest spherical ball that can be placed on top of a torus -/
noncomputable def largest_ball_radius (t : Torus) : ℝ :=
  (t.outer_radius - t.inner_radius) / 2 + t.inner_radius

theorem largest_ball_on_torus (t : Torus) (b : SphericalBall) :
  t.inner_radius = 3 →
  t.outer_radius = 5 →
  b.radius = largest_ball_radius t →
  b.radius = 4 := by
  sorry

-- Use #eval only for computable functions
/- 
#eval largest_ball_radius { inner_radius := 3, outer_radius := 5 }
-/

-- Instead, we can use the following to check the result:
#check largest_ball_radius { inner_radius := 3, outer_radius := 5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_ball_on_torus_l981_98155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equation_area_range_l981_98135

-- Define points A, B, and D
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (0, 1)
def D : ℝ × ℝ := (0, 2)

-- Define the locus C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≠ 0 ∧ p.1^2 / 2 + p.2^2 = 1}

-- Define the slope product condition
def slope_product (M : ℝ × ℝ) : Prop :=
  M.1 ≠ 0 ∧ (M.2 + 1) / M.1 * (M.2 - 1) / M.1 = -1/2

-- Theorem for the locus equation
theorem locus_equation (M : ℝ × ℝ) :
  slope_product M → M ∈ C :=
sorry

-- Define a line passing through D
def line_through_D (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 2}

-- Define the area of triangle OEF
noncomputable def area_OEF (E F : ℝ × ℝ) : ℝ :=
  abs (E.1 * F.2 - F.1 * E.2) / 2

-- Theorem for the area range
theorem area_range (k : ℝ) (E F : ℝ × ℝ) :
  E ∈ C → F ∈ C → E ≠ F →
  E ∈ line_through_D k → F ∈ line_through_D k →
  0 < area_OEF E F ∧ area_OEF E F ≤ Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equation_area_range_l981_98135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_PAB_l981_98166

/-- Region D defined by the system of inequalities -/
def D : Set (ℝ × ℝ) :=
  {p | 3 * p.1 + 4 * p.2 - 10 ≥ 0 ∧ p.1 ≤ 4 ∧ p.2 ≤ 3}

/-- Circle with equation x² + y² = 1 -/
def Circle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 1}

/-- Angle between three points -/
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the minimum cosine value of angle PAB -/
theorem min_cos_PAB (P : ℝ × ℝ) (hP : P ∈ D) :
  ∃ (A B : ℝ × ℝ), A ∈ Circle ∧ B ∈ Circle ∧
    (∀ (A' B' : ℝ × ℝ), A' ∈ Circle → B' ∈ Circle →
      Real.cos (angle P A B) ≤ Real.cos (angle P A' B')) →
        Real.cos (angle P A B) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cos_PAB_l981_98166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_count_l981_98152

theorem point_count (n a₁ a₂ b₁ b₂ : ℕ) 
  (h1 : a₁ * a₂ = 40)
  (h2 : b₁ * b₂ = 42)
  (h3 : a₁ + a₂ = b₁ + b₂)
  (h4 : n = a₁ + a₂ + 1) : n = 14 := by
  sorry

#check point_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_count_l981_98152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_achievable_is_correct_l981_98157

/-- The smallest achievable integer in the number replacement game -/
def smallest_achievable (n : ℕ) : ℕ := 2

/-- The number replacement game process -/
def number_replacement_game (n : ℕ) : Prop :=
  n ≥ 3 ∧
  ∃ (final : ℚ),
    (∀ sequence : List ℚ,
      (sequence.length ≥ 1 ∧
       ∀ x ∈ sequence, x ≥ 1 ∧
       ∃ y ∈ sequence, y > 1) →
      final ≤ sequence.head!) ∧
    final ≥ smallest_achievable n

theorem smallest_achievable_is_correct (n : ℕ) :
  number_replacement_game n →
  smallest_achievable n = 2 :=
by
  intro h
  rfl

#check smallest_achievable_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_achievable_is_correct_l981_98157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_18_6_l981_98165

theorem binomial_18_6 : Nat.choose 18 6 = 18564 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_18_6_l981_98165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l981_98142

-- Define the triangle sides
noncomputable def a : ℝ := 26
noncomputable def b : ℝ := 24
noncomputable def c : ℝ := 20

-- Define the semi-perimeter
noncomputable def s : ℝ := (a + b + c) / 2

-- Define Heron's formula for the area
noncomputable def heronArea : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem triangle_area_approx : 
  ∀ ε > 0, |heronArea - 228.07| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_approx_l981_98142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_household_chores_theorem_l981_98125

/-- Represents the frequency distribution of weekly household labor time --/
structure FrequencyDistribution :=
  (less_than_1 : ℕ)
  (between_1_and_2 : ℕ)
  (between_2_and_3 : ℕ)
  (between_3_and_4 : ℕ)

/-- The main theorem --/
theorem household_chores_theorem 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (freq_dist : FrequencyDistribution) 
  (h1 : total_students = 400)
  (h2 : sample_size = 40)
  (h3 : freq_dist.less_than_1 = 8)
  (h4 : freq_dist.between_1_and_2 = 20)
  (h5 : freq_dist.between_2_and_3 = 7)
  (h6 : freq_dist.between_3_and_4 = 5)
  (h7 : sample_size = freq_dist.less_than_1 + freq_dist.between_1_and_2 + 
        freq_dist.between_2_and_3 + freq_dist.between_3_and_4) :
  (∃ (estimated_less_than_2 : ℕ), 
    estimated_less_than_2 = (freq_dist.less_than_1 + freq_dist.between_1_and_2) * total_students / sample_size ∧ 
    estimated_less_than_2 = 280) ∧
  (∃ (prob : ℚ), 
    prob = 3 / 5 ∧
    prob = (Nat.choose 2 1 * Nat.choose 3 1) / Nat.choose 5 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_household_chores_theorem_l981_98125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_zero_monotone_increasing_implies_a_range_max_b_value_when_a_negative_one_l981_98117

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 / 3 - x^2 - a*x + Real.log (a*x + 1)

-- Theorem 1
theorem extreme_point_implies_a_zero (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f a x ≤ f a 2) →
  a = 0 :=
by sorry

-- Theorem 2
theorem monotone_increasing_implies_a_range (a : ℝ) :
  (∀ x y, x ∈ Set.Ici 3 → y ∈ Set.Ici 3 → x ≤ y → f a x ≤ f a y) →
  a ∈ Set.Icc 0 ((3 + Real.sqrt 13) / 2) :=
by sorry

-- Theorem 3
theorem max_b_value_when_a_negative_one (b : ℝ) :
  (∃ x, f (-1) x = x^3 / 3 + b / (1 - x)) →
  b ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_zero_monotone_increasing_implies_a_range_max_b_value_when_a_negative_one_l981_98117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_l981_98176

/-- A point (a, b) lies outside the unit circle if a^2 + b^2 > 1 -/
def outside_circle (a b : ℝ) : Prop := a^2 + b^2 > 1

/-- The distance between a line ax + by = 1 and the origin (0, 0) -/
noncomputable def line_origin_distance (a b : ℝ) : ℝ := 1 / Real.sqrt (a^2 + b^2)

/-- A line ax + by = 1 is disjoint from the unit circle if its distance from the origin is greater than 1 -/
def line_circle_disjoint (a b : ℝ) : Prop := line_origin_distance a b < 1

theorem line_circle_relationship (a b : ℝ) 
  (h : outside_circle a b) : line_circle_disjoint a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_relationship_l981_98176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_halloween_candy_total_l981_98154

/-- The total number of candy pieces for Taquon, Mack, and Jafari combined is 418. -/
theorem halloween_candy_total
  (taquon_candy : ℕ) (mack_candy : ℕ) (jafari_candy : ℕ)
  (h1 : taquon_candy = 171)
  (h2 : mack_candy = 171)
  (h3 : jafari_candy = 76) :
  taquon_candy + mack_candy + jafari_candy = 418 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_halloween_candy_total_l981_98154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_annual_cost_l981_98171

/-- Calculates the total annual cost for Tom's medication and doctor visits -/
def annual_cost (pills_per_day : ℕ) (visits_per_year : ℕ) (visit_cost : ℕ) 
  (pill_cost : ℕ) (insurance_coverage : ℚ) : ℕ :=
  let annual_visit_cost := visits_per_year * visit_cost
  let daily_med_cost := pills_per_day * pill_cost
  let daily_out_of_pocket := (daily_med_cost : ℚ) * (1 - insurance_coverage)
  let annual_med_cost := (daily_out_of_pocket * 365).floor.toNat
  annual_visit_cost + annual_med_cost

/-- Theorem stating that Tom's annual cost is $1530 -/
theorem toms_annual_cost : 
  annual_cost 2 2 400 5 (4/5) = 1530 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_annual_cost_l981_98171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_reciprocal_l981_98160

def a : ℕ := 24
def b : ℕ := 156

theorem lcm_reciprocal (h : (Nat.gcd a b : ℚ)⁻¹ = 1 / 12) :
  (Nat.lcm a b : ℚ)⁻¹ = 1 / 312 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_reciprocal_l981_98160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematicians_meeting_time_l981_98172

/-- Represents the duration of stay for mathematicians in minutes -/
noncomputable def m : ℝ := 120 - 60 * Real.sqrt 2

/-- Represents the probability of the mathematicians meeting -/
def meeting_probability : ℚ := 1/2

/-- The total time interval in minutes -/
def total_time : ℝ := 120

theorem mathematicians_meeting_time :
  (1 - (total_time - m)^2 / total_time^2) = meeting_probability :=
by sorry

-- Remove the #eval statement as it's not necessary for the theorem and causes issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematicians_meeting_time_l981_98172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l981_98164

theorem election_votes (winning_percentage : ℚ) (vote_majority : ℕ) : 
  winning_percentage = 70 / 100 → 
  vote_majority = 176 → 
  ∃ (total_votes : ℕ), 
    (winning_percentage * total_votes - (1 - winning_percentage) * total_votes = vote_majority) ∧
    total_votes = 440 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l981_98164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_womens_average_age_l981_98114

/-- The average age of two women who replace two men in a group, given specific conditions --/
theorem womens_average_age (n : ℕ) (initial_avg : ℝ) (age1 age2 : ℕ) (increase : ℝ) 
  (h1 : n = 8)  -- Number of people in the group
  (h2 : age1 = 20)  -- Age of first replaced man
  (h3 : age2 = 22)  -- Age of second replaced man
  (h4 : increase = 2)  -- Increase in average age after replacement
  : (n * (initial_avg + increase) - (n * initial_avg - age1 - age2)) / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_womens_average_age_l981_98114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l981_98128

/-- The circle with equation x^2 + y^2 = 9 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 9}

/-- The line with parametric equations x = 1 + 2t, y = 2 + t -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 1 + 2*t ∧ p.2 = 2 + t}

/-- The length of the chord intercepted by the circle on the line -/
noncomputable def ChordLength : ℝ := 3 * Real.sqrt 3

theorem chord_length_is_correct :
  ∃ p q : ℝ × ℝ,
    p ∈ Circle ∧ q ∈ Circle ∧
    p ∈ Line ∧ q ∈ Line ∧
    p ≠ q ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = ChordLength := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_correct_l981_98128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flattest_ellipse_l981_98178

-- Define the function to calculate the ratio of semi-minor to semi-major axis
noncomputable def axis_ratio (a b : ℝ) : ℝ := Real.sqrt b / Real.sqrt a

-- Define the ellipses
def ellipse_A : ℝ × ℝ := (16, 12)
def ellipse_B : ℝ × ℝ := (4, 1)
def ellipse_C : ℝ × ℝ := (6, 3)
def ellipse_D : ℝ × ℝ := (9, 8)

-- Theorem statement
theorem flattest_ellipse :
  let ratios := [
    axis_ratio ellipse_A.1 ellipse_A.2,
    axis_ratio ellipse_B.1 ellipse_B.2,
    axis_ratio ellipse_C.1 ellipse_C.2,
    axis_ratio ellipse_D.1 ellipse_D.2
  ]
  axis_ratio ellipse_B.1 ellipse_B.2 = ratios.minimum :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flattest_ellipse_l981_98178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_equation_line_passes_through_point_l981_98126

-- Define a line in a 2D Cartesian coordinate system
structure Line where
  slope : ℝ
  point : ℝ × ℝ

-- Define a vertical line (slope of 90 degrees)
def isVertical (l : Line) : Prop := l.slope = Real.pi / 2

-- Theorem 1: A vertical line passing through (x₀, y₀) has the equation x = x₀
theorem vertical_line_equation (l : Line) (h : isVertical l) :
  ∀ (x y : ℝ), (x = l.point.fst) ↔ (x, y) ∈ Set.range (λ t => (l.point.fst, t)) :=
sorry

-- Theorem 2: The line y - 3 = k(x + 1) always passes through (-1, 3)
theorem line_passes_through_point (k : ℝ) :
  (-1, 3) ∈ Set.range (λ x => (x, k * (x + 1) + 3)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_line_equation_line_passes_through_point_l981_98126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_eq_neg_cos_l981_98140

open Real

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => sin
  | n + 1 => deriv (f n)

-- State the theorem
theorem f_2011_eq_neg_cos : f 2011 = λ x => -cos x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_eq_neg_cos_l981_98140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_circle_and_line_l981_98190

/-- Given a circle with center O and radius R, a line ax + by + c = 0,
    and a radius r, there exists a point P that is both at distance R ± r
    from O and at distance r from the line. -/
theorem circle_tangent_to_circle_and_line
  (O : ℝ × ℝ) (R r : ℝ) (a b c : ℝ) :
  ∃ (P : ℝ × ℝ),
    (((P.1 - O.1)^2 + (P.2 - O.2)^2 = (R + r)^2) ∨
     ((P.1 - O.1)^2 + (P.2 - O.2)^2 = (R - r)^2)) ∧
    (|a * P.1 + b * P.2 + c| / Real.sqrt (a^2 + b^2) = r) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_circle_and_line_l981_98190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_converges_to_zero_l981_98182

open Real

-- Define the sequence
noncomputable def a : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => sin (a n x)

-- State the theorem
theorem sequence_converges_to_zero (x : ℝ) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n x| < ε := by
  sorry

#check sequence_converges_to_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_converges_to_zero_l981_98182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_correct_l981_98189

open Real

-- Define the spherical coordinates
noncomputable def ρ₁ : ℝ := 10
noncomputable def θ₁ : ℝ := π / 4
noncomputable def φ₁ : ℝ := π / 6

noncomputable def ρ₂ : ℝ := 15
noncomputable def θ₂ : ℝ := 5 * π / 4
noncomputable def φ₂ : ℝ := π / 3

-- Define the conversion function
noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

-- State the theorem
theorem spherical_to_rectangular_correct :
  spherical_to_rectangular ρ₁ θ₁ φ₁ = (5 * sqrt 2 / 2, 5 * sqrt 2 / 2, 5 * sqrt 3) ∧
  spherical_to_rectangular ρ₂ θ₂ φ₂ = (-15 * sqrt 6 / 4, -15 * sqrt 6 / 4, 15 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_correct_l981_98189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l981_98196

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The focal distance of a hyperbola -/
noncomputable def focal_distance (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

/-- Theorem about the eccentricity range of a hyperbola under specific conditions -/
theorem hyperbola_eccentricity_range (h : Hyperbola) 
  (hB : ∃ (B : ℝ × ℝ), B.1 ∈ Set.Icc h.a (focal_distance h) ∧ 
    |B.2| < 2 * (h.a + focal_distance h)) : 
  1 < eccentricity h ∧ eccentricity h < Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l981_98196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_transfers_for_card_equalization_l981_98183

/-- The number of people sitting at the round table -/
def n : ℕ := 101

/-- The initial sum of the product of position and number of cards -/
def initial_sum : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The final sum when all people have the same number of cards -/
def final_sum : ℕ := n^3

/-- The minimum number of transfers required -/
def min_transfers : ℕ := initial_sum - final_sum

theorem min_transfers_for_card_equalization :
  min_transfers = 42925 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_transfers_for_card_equalization_l981_98183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l981_98123

/-- Represents a mapping from Chinese characters to digits -/
def ChineseToDigit := Char → Fin 10

/-- The four Chinese characters in the problem -/
def characters : List Char := ['希', '望', '数', '学']

/-- Predicate to check if a mapping is valid according to the problem conditions -/
def is_valid_mapping (m : ChineseToDigit) : Prop :=
  (∀ c₁ c₂, c₁ ∈ characters → c₂ ∈ characters → c₁ ≠ c₂ → m c₁ ≠ m c₂) ∧
  (m '希' ≠ 0)  -- Ensures the number is four-digit

/-- The number represented by the Chinese characters under a given mapping -/
def number_from_mapping (m : ChineseToDigit) : ℕ :=
  1000 * (m '希').val + 100 * (m '望').val + 10 * (m '数').val + (m '学').val

/-- Main theorem stating the uniqueness and value of the solution -/
theorem unique_solution :
  ∃! m : ChineseToDigit, is_valid_mapping m ∧ number_from_mapping m = 1820 := by
  sorry

#eval "The code compiles successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l981_98123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stack_height_for_12cm_rods_l981_98170

/-- The height of 3 stacked cylindrical rods -/
noncomputable def stack_height (rod_diameter : ℝ) : ℝ :=
  rod_diameter + rod_diameter * Real.sqrt 3 / 2 + rod_diameter / 2

/-- Theorem: The height of 3 cylindrical rods with diameter 12 cm, 
    stacked in a triangular formation, is 12 + 6√3 cm -/
theorem stack_height_for_12cm_rods : 
  stack_height 12 = 12 + 6 * Real.sqrt 3 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stack_height_for_12cm_rods_l981_98170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l981_98180

/-- The function f(x) = |x-2| + |x-4| -/
def f (x : ℝ) : ℝ := |x - 2| + |x - 4|

theorem f_properties :
  (∃ (S : Set ℝ), S = {x | f x ≥ 4} ∧ S = Set.Iic 1 ∪ Set.Ici 5) ∧
  (∃ (M : ℝ), M = 2 ∧ ∀ x, f x ≥ M) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 2 →
    1/a + (b+4)/b ≥ 11/2 ∧
    ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1/a₀ + (b₀+4)/b₀ = 11/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l981_98180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_g_iff_a_in_range_l981_98113

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 + 2 * x

noncomputable def g (x : ℝ) : ℝ := (2 * x^2 + 1) / x

def interval : Set ℝ := Set.Icc 1 4

theorem f_leq_g_iff_a_in_range (a : ℝ) : 
  (∀ x₁ x₂, x₁ ∈ interval → x₂ ∈ interval → f a x₁ ≤ g x₂) ↔ a ∈ Set.Iic (-1/6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_leq_g_iff_a_in_range_l981_98113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_rig_topple_depth_l981_98153

/-- The depth of the sea where an oil rig topples over. -/
noncomputable def sea_depth (rig_height : ℝ) (horizontal_distance : ℝ) : ℝ :=
  (horizontal_distance^2 - rig_height^2) / (2 * rig_height)

/-- Theorem stating that the depth of the sea where a 40-meter tall oil rig submerges
    is approximately 68.2 meters, given that the top of the rig disappears 84 meters horizontally
    from its original position. -/
theorem oil_rig_topple_depth :
  ‖sea_depth 40 84 - 68.2‖ < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_rig_topple_depth_l981_98153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_line_l981_98103

/-- A parabola is tangent to a line if they intersect at exactly one point. -/
def is_tangent (a b : ℝ) : Prop :=
  ∃! x : ℝ, a * x^2 + b * x + 2 = 2 * x + 4

/-- The main theorem stating the values of a and b for which the parabola is tangent to the line. -/
theorem parabola_tangent_to_line :
  ∀ a b : ℝ, is_tangent a b → a = 0 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_to_line_l981_98103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_circle_plus_equals_one_l981_98151

-- Define the ⊕ operation
noncomputable def circle_plus (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- Define a function to represent the nested operation
noncomputable def nested_circle_plus : ℕ → ℝ
  | 0 => 1000  -- Base case
  | n + 1 => circle_plus (n + 1) (nested_circle_plus n)

-- Theorem statement
theorem nested_circle_plus_equals_one : 
  nested_circle_plus 999 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_circle_plus_equals_one_l981_98151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sequence_bound_l981_98139

theorem divisor_sequence_bound (n : ℕ) (h_n : n % 8 = 4) :
  ∀ (m : ℕ) (k : Fin (m + 1) → ℕ),
    (∀ i : Fin (m + 1), k i ∣ n) →
    (∀ i j : Fin (m + 1), i < j → k i < k j) →
    k 0 = 1 →
    k m = n →
    ∀ i : Fin m, i.val % 3 ≠ 0 →
      k (i + 1) ≤ 2 * k i :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_sequence_bound_l981_98139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gears_rotate_iff_even_l981_98175

/-- Represents a system of gears arranged in a circle. -/
structure GearSystem where
  n : ℕ  -- number of gears
  meshed : Fin n → Fin n → Bool
  all_meshed : ∀ i j : Fin n, (i.val + 1) % n = j.val → meshed i j = true

/-- Represents the ability of gears to rotate. -/
def can_rotate (g : GearSystem) : Prop :=
  g.n % 2 = 0

/-- Theorem stating that gears can rotate if and only if their number is even. -/
theorem gears_rotate_iff_even (g : GearSystem) : 
  can_rotate g ↔ Even g.n := by
  sorry

#check gears_rotate_iff_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gears_rotate_iff_even_l981_98175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_sixes_theorem_l981_98100

-- Define a type for arithmetic expressions
inductive ArithExpr
  | Num : ℕ → ArithExpr
  | Add : ArithExpr → ArithExpr → ArithExpr
  | Sub : ArithExpr → ArithExpr → ArithExpr
  | Mul : ArithExpr → ArithExpr → ArithExpr
  | Div : ArithExpr → ArithExpr → ArithExpr
  | Exp : ArithExpr → ArithExpr → ArithExpr

-- Define a function to count the number of sixes in an expression
def countSixes : ArithExpr → ℕ
  | ArithExpr.Num 6 => 1
  | ArithExpr.Num _ => 0
  | ArithExpr.Add e1 e2 => countSixes e1 + countSixes e2
  | ArithExpr.Sub e1 e2 => countSixes e1 + countSixes e2
  | ArithExpr.Mul e1 e2 => countSixes e1 + countSixes e2
  | ArithExpr.Div e1 e2 => countSixes e1 + countSixes e2
  | ArithExpr.Exp e1 e2 => countSixes e1 + countSixes e2

-- Define a function to evaluate an arithmetic expression
noncomputable def evaluate : ArithExpr → ℚ
  | ArithExpr.Num n => n
  | ArithExpr.Add e1 e2 => evaluate e1 + evaluate e2
  | ArithExpr.Sub e1 e2 => evaluate e1 - evaluate e2
  | ArithExpr.Mul e1 e2 => evaluate e1 * evaluate e2
  | ArithExpr.Div e1 e2 => evaluate e1 / evaluate e2
  | ArithExpr.Exp e1 e2 => (evaluate e1) ^ (Int.floor (evaluate e2))

-- Theorem statement
theorem five_sixes_theorem :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 14 →
  ∃ e : ArithExpr, countSixes e = 5 ∧ evaluate e = n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_sixes_theorem_l981_98100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawrence_even_distribution_l981_98146

/-- Lawrence's work schedule --/
def lawrence_schedule : List (String × Float) := [
  ("Monday", 8),
  ("Tuesday", 8),
  ("Wednesday", 5.5),
  ("Thursday", 5.5),
  ("Friday", 8)
]

/-- Total work hours for the week --/
def total_hours : Float := (lawrence_schedule.map Prod.snd).sum

/-- Number of days in a week --/
def days_in_week : Nat := 7

/-- Theorem: If Lawrence's total work hours were distributed evenly across 7 days, he would work 5 hours each day --/
theorem lawrence_even_distribution :
  total_hours / days_in_week.toFloat = 5 := by
  sorry

#eval total_hours -- Should output 35
#check lawrence_even_distribution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawrence_even_distribution_l981_98146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l981_98122

def our_sequence (a : ℕ+ → ℚ) : Prop :=
  a 1 = 1 ∧
  a 2 = 1/2 ∧
  ∀ n : ℕ+, 2 / (a (n + 1)) = 1 / (a n) + 1 / (a (n + 2))

theorem a_2015_value (a : ℕ+ → ℚ) (h : our_sequence a) : a 2015 = 1 / 2015 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2015_value_l981_98122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_properties_l981_98121

/-- Represents an isosceles triangle with given side lengths and perimeter -/
structure IsoscelesTriangle where
  congruentSide : ℝ
  perimeter : ℝ

/-- Calculates the base length of the isosceles triangle -/
noncomputable def baseLength (t : IsoscelesTriangle) : ℝ :=
  t.perimeter - 2 * t.congruentSide

/-- Calculates the area of the isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  let base := baseLength t
  let height := Real.sqrt (t.congruentSide^2 - (base/2)^2)
  (1/2) * base * height

theorem isosceles_triangle_properties (t : IsoscelesTriangle) 
    (h1 : t.congruentSide = 8)
    (h2 : t.perimeter = 26) : 
    baseLength t = 10 ∧ area t = 5 * Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_properties_l981_98121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l981_98168

/-- The function f(x) = 2cos²(x) + √3sin(2x) -/
noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + Real.sqrt 3 * Real.sin (2 * x)

theorem f_properties :
  (∀ x : ℝ, f x ≤ 3) ∧ 
  (∀ k : ℤ, f (π / 6 + k * π) = 3) ∧
  (∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6) → 
    ∀ y : ℝ, y ∈ Set.Icc (k * π - π / 3) (k * π + π / 6) → x ≤ y → f x ≤ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l981_98168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_multiples_of_3_and_7_l981_98185

theorem two_digit_multiples_of_3_and_7 : 
  Finset.card (Finset.filter (fun n => 10 ≤ n ∧ n < 100 ∧ 3 ∣ n ∧ 7 ∣ n) (Finset.range 100)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_multiples_of_3_and_7_l981_98185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l981_98102

/-- Conversion from spherical coordinates to rectangular coordinates -/
noncomputable def spherical_to_rectangular (ρ θ φ : Real) : Real × Real × Real :=
  (ρ * Real.sin φ * Real.cos θ,
   ρ * Real.sin φ * Real.sin θ,
   ρ * Real.cos φ)

/-- Theorem stating the equivalence of the given spherical coordinates to rectangular coordinates -/
theorem spherical_to_rectangular_conversion :
  let ρ : Real := 3
  let θ : Real := 5 * Real.pi / 12
  let φ : Real := Real.pi / 6
  spherical_to_rectangular ρ θ φ = (3 * (Real.sqrt 6 + Real.sqrt 2) / 8,
                                    3 * (Real.sqrt 6 - Real.sqrt 2) / 8,
                                    3 * Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_to_rectangular_conversion_l981_98102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OPAB_l981_98162

/-- An ellipse with equation mx^2 + 3my^2 = 1 where m > 0 -/
structure Ellipse where
  m : ℝ
  h_m_pos : m > 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin point (0,0) -/
def origin : Point := ⟨0, 0⟩

/-- Point A with coordinates (3,0) -/
def point_A : Point := ⟨3, 0⟩

/-- Checks if a point is on the given ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  e.m * p.x^2 + 3 * e.m * p.y^2 = 1

/-- Checks if a point is on the y-axis -/
def on_y_axis (p : Point) : Prop :=
  p.x = 0

/-- Checks if a point is to the right of the y-axis -/
def right_of_y_axis (p : Point) : Prop :=
  p.x > 0

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the area of a quadrilateral given its four vertices -/
noncomputable def quadrilateral_area (p1 p2 p3 p4 : Point) : ℝ :=
  sorry  -- Actual calculation would go here

/-- The main theorem stating the minimum area of quadrilateral OPAB -/
theorem min_area_OPAB (e : Ellipse) (B P : Point) 
    (h_major_axis : 2 * Real.sqrt (1 / e.m) = 2 * Real.sqrt 6)
    (h_B_on_y : on_y_axis B)
    (h_P_on_ellipse : on_ellipse e P)
    (h_P_right : right_of_y_axis P)
    (h_BA_BP : distance B point_A = distance B P) :
    ∃ (min_area : ℝ), min_area = 3 * Real.sqrt 3 ∧
    ∀ (B' P' : Point), on_y_axis B' → on_ellipse e P' → right_of_y_axis P' →
    distance B' point_A = distance B' P' →
    quadrilateral_area origin point_A B' P' ≥ min_area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_OPAB_l981_98162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_correct_l981_98143

-- Define the four propositions
def proposition1 : Prop := ∀ x : ℝ, (x^2 = 1 → x = 1) ↔ ¬(x^2 = 1 → x ≠ 1)
def proposition2 : Prop := ∀ x : ℝ, (x = -1 → x^2 - 5*x - 6 = 0) ∧ ¬(x^2 - 5*x - 6 = 0 → x = -1)
def proposition3 : Prop := (∃ x : ℝ, x^2 + x - 1 < 0) ↔ ¬(∀ x : ℝ, x^2 + x - 1 > 0)
def proposition4 : Prop := (∀ x y : ℝ, x = y → Real.sin x = Real.sin y) ↔ (∀ x y : ℝ, Real.sin x ≠ Real.sin y → x ≠ y)

-- Theorem stating that only the fourth proposition is correct
theorem only_fourth_proposition_correct :
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 :=
by
  sorry -- Proof is omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_proposition_correct_l981_98143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_direction_assignment_l981_98104

-- Define the set of directions
inductive Direction
| A
| B
| C
| D

-- Define a function that assigns a direction to each team
def team_direction : Direction → Direction := sorry

-- State the conditions
axiom condition1 : team_direction Direction.A ≠ Direction.A ∧ team_direction Direction.A ≠ Direction.D
axiom condition2 : team_direction Direction.B ≠ Direction.A ∧ team_direction Direction.B ≠ Direction.B
axiom condition3 : team_direction Direction.C ≠ Direction.A ∧ team_direction Direction.C ≠ Direction.B
axiom condition4 : team_direction Direction.D ≠ Direction.C ∧ team_direction Direction.D ≠ Direction.D
axiom additional_condition : team_direction Direction.C ≠ Direction.D → team_direction Direction.D ≠ Direction.A

-- State the theorem
theorem unique_direction_assignment :
  team_direction Direction.A = Direction.B ∧
  team_direction Direction.B = Direction.C ∧
  team_direction Direction.C = Direction.D ∧
  team_direction Direction.D = Direction.A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_direction_assignment_l981_98104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l981_98199

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  16 * x^2 + 32 * x - 4 * y^2 - 8 * y - 3 = 0

/-- The distance between the vertices of the hyperbola -/
noncomputable def vertex_distance : ℝ := Real.sqrt 15 / 2

/-- Theorem stating that the distance between the vertices of the given hyperbola is √15/2 -/
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_eq x y → 
  (∃ a b : ℝ, (x + 1)^2 / a^2 - (y + 1)^2 / b^2 = 1 ∧ 
              vertex_distance = 2 * a) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_distance_l981_98199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l981_98167

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) + 1 / 2

theorem g_monotone_increasing : 
  MonotoneOn g (Set.Icc 0 (Real.pi / 3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l981_98167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_grade_students_l981_98186

/-- The number of students in eighth grade -/
theorem eighth_grade_students : ∃ (total : ℕ), total = 68 := by
  let girls : ℕ := 28
  let boys : ℕ := 2 * girls - 16
  let total : ℕ := girls + boys
  have h : total = 68 := by
    -- Proof steps would go here
    sorry
  exact ⟨total, h⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_grade_students_l981_98186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_minor_axis_length_l981_98173

noncomputable def ellipse_center : ℝ × ℝ := (2, 3)
noncomputable def ellipse_focus : ℝ × ℝ := (2, 1)
noncomputable def semi_major_endpoint : ℝ × ℝ := (0, 3)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def semi_major_axis : ℝ := distance ellipse_center semi_major_endpoint
noncomputable def focal_distance : ℝ := distance ellipse_center ellipse_focus

theorem semi_minor_axis_length :
  Real.sqrt (semi_major_axis^2 - focal_distance^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_minor_axis_length_l981_98173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_big_product_l981_98174

def product_sequence (n : Nat) : Nat := 2^(2^n) + 1

def big_product : Nat := (2 + 1) * (Finset.range 7).prod (fun i => product_sequence i)

theorem unit_digit_of_big_product :
  big_product % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_big_product_l981_98174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l981_98192

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 - 3*x ≤ 0}
def N : Set ℝ := {x : ℝ | x < 1}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Iic 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l981_98192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l981_98118

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the foci
def left_focus : ℝ × ℝ := (-2, 0)
def right_focus : ℝ × ℝ := (2, 0)

-- Define the maximum dot product condition
def max_dot_product (P : ℝ × ℝ) : Prop :=
  ∀ Q : ℝ × ℝ, ellipse 6 2 Q.1 Q.2 → 
    (P.1 + 2) * (P.1 - 2) + P.2 * P.2 ≥ (Q.1 + 2) * (Q.1 - 2) + Q.2 * Q.2

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + 2) ∨ x = -2

-- Define the condition for line l
def line_l_condition (M N : ℝ × ℝ) (θ : ℝ) : Prop :=
  M.1 * N.1 + M.2 * N.2 * Real.sin θ = (4 * Real.sqrt 6 / 3) * Real.cos θ ∧ θ ≠ Real.pi / 2

-- State the theorem
theorem ellipse_and_line_theorem :
  ∃ (P : ℝ × ℝ), ellipse 6 2 P.1 P.2 ∧ max_dot_product P ∧
  ∃ (M N : ℝ × ℝ) (θ : ℝ), 
    ellipse 6 2 M.1 M.2 ∧ 
    ellipse 6 2 N.1 N.2 ∧
    line_l_condition M N θ ∧
    (line_l (-Real.sqrt 3 / 3) M.1 M.2 ∨
     line_l (Real.sqrt 3 / 3) M.1 M.2 ∨
     M.1 = -2) ∧
    (line_l (-Real.sqrt 3 / 3) N.1 N.2 ∨
     line_l (Real.sqrt 3 / 3) N.1 N.2 ∨
     N.1 = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l981_98118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_leq_one_l981_98130

theorem negation_of_exists_sin_leq_one :
  (¬ ∃ x : ℝ, Real.sin x ≤ 1) ↔ (∀ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_leq_one_l981_98130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_distance_equality_l981_98131

/-- The distance from the focus of a hyperbola to its asymptote -/
noncomputable def hyperbola_focus_asymptote_distance (b : ℝ) : ℝ :=
  b / Real.sqrt (1 + 1 / b^2)

/-- The distance from a point on a parabola to its focus -/
noncomputable def parabola_point_focus_distance (p : ℝ) : ℝ :=
  p / 2

/-- Theorem stating the equality of b and 2 under given conditions -/
theorem hyperbola_parabola_distance_equality (b : ℝ) (h1 : b > 0) :
  hyperbola_focus_asymptote_distance b = parabola_point_focus_distance 2 → b = 2 := by
  sorry

#check hyperbola_parabola_distance_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_distance_equality_l981_98131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_solutions_l981_98145

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the non-overlapping condition
def non_overlapping (k_A k_B : Circle) : Prop :=
  let (x_A, y_A) := k_A.center
  let (x_B, y_B) := k_B.center
  (x_A - x_B) ^ 2 + (y_A - y_B) ^ 2 > (k_A.radius + k_B.radius) ^ 2

-- Define an equilateral triangle
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point being on a circle
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx) ^ 2 + (y - cy) ^ 2 = c.radius ^ 2

-- Define a line being tangent to a circle
def is_tangent (p1 p2 : ℝ × ℝ) (c : Circle) : Prop :=
  sorry -- This would require a more complex definition

-- Define the conditions for the triangle
def valid_triangle (k_A k_B : Circle) (t : EquilateralTriangle) : Prop :=
  (on_circle t.A k_A) ∧ 
  (on_circle t.B k_B) ∧ 
  (is_tangent t.A t.C k_A) ∧ 
  (is_tangent t.B t.C k_B)

-- Main theorem
theorem eight_solutions (k_A k_B : Circle) (h : non_overlapping k_A k_B) : 
  ∃! (solutions : Finset EquilateralTriangle), 
    solutions.card = 8 ∧ 
    ∀ t ∈ solutions, valid_triangle k_A k_B t :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_solutions_l981_98145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l981_98106

theorem power_equation (x y : ℝ) (h1 : (10 : ℝ)^x = 3) (h2 : (10 : ℝ)^y = 4) :
  (10 : ℝ)^(3*x - 2*y) = 27/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_l981_98106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l981_98188

open Real

theorem triangle_problem (a b c A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  b = 2 * (a * cos B - c) →
  (A = 2 * π / 3) ∧
  (a * cos C = Real.sqrt 3 ∧ b = 1 → c = 2 * Real.sqrt 3 - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l981_98188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_players_count_l981_98138

/-- Represents the number of players taking a specific subject or combination of subjects -/
structure PlayerCount where
  count : Nat

/-- The total number of players on the soccer team -/
def total_players : PlayerCount := ⟨20⟩

/-- The number of players taking biology -/
def biology_players : PlayerCount := ⟨10⟩

/-- The number of players taking both biology and chemistry -/
def biology_and_chemistry_players : PlayerCount := ⟨4⟩

/-- The number of players taking all three subjects -/
def all_subjects_players : PlayerCount := ⟨3⟩

/-- Addition for PlayerCount -/
instance : Add PlayerCount where
  add a b := ⟨a.count + b.count⟩

/-- Subtraction for PlayerCount -/
instance : Sub PlayerCount where
  sub a b := ⟨a.count - b.count⟩

/-- LessEq for PlayerCount -/
instance : LE PlayerCount where
  le a b := a.count ≤ b.count

/-- Theorem stating that the number of players taking chemistry is 6 -/
theorem chemistry_players_count : 
  ∃ (chemistry_players : PlayerCount), 
    chemistry_players = ⟨6⟩ ∧ 
    chemistry_players ≤ total_players ∧
    chemistry_players + biology_players - biology_and_chemistry_players ≤ total_players :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chemistry_players_count_l981_98138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_from_sine_cosine_l981_98179

theorem triangle_cosine_from_sine_cosine (A B C : ℝ) (h_triangle : A + B + C = Real.pi) 
  (h_sin_A : Real.sin A = 4/5) (h_cos_B : Real.cos B = 12/13) : Real.cos C = -16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_from_sine_cosine_l981_98179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l981_98101

/-- Given a triangle ABC with specific properties, prove its area is 2√3 -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let a : ℝ × ℝ := (Real.sin (2 * A.1), Real.cos (2 * A.1))
  let b : ℝ × ℝ := (Real.sqrt 3 / 2, -1 / 2)
  (a.1 * b.1 + a.2 * b.2 = 1 / 2) →  -- dot product equals 1/2
  (2 * Real.sqrt 3 = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) →  -- AB = 2√3
  (2 = Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)) →  -- BC = 2
  (1/2 * 2 * 2 * Real.sqrt 3 * Real.sin (Real.pi / 2) = 2 * Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l981_98101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_attendance_l981_98116

theorem concert_attendance (total_students : ℕ) (boys girls : ℕ) : 
  boys = girls →
  boys + girls = total_students →
  (5 : ℚ) / 6 * girls / ((5 : ℚ) / 6 * girls + (3 : ℚ) / 4 * boys) = (10 : ℚ) / 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_attendance_l981_98116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l981_98148

noncomputable def f (x : ℝ) := x / (1 + x^2)

theorem f_properties :
  let I : Set ℝ := Set.Ioo (-1) 1
  (∀ x ∈ I, f x = x / (1 + x^2)) ∧ 
  (∀ x ∈ I, f (-x) = -f x) ∧
  (∀ x ∈ I, deriv f x = (1 - x^2) / (1 + x^2)^2) →
  (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) ∧
  (∀ t, t ∈ I → (f (t - 1) + f t < 0 ↔ 0 < t ∧ t < 1/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l981_98148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l981_98177

theorem divisibility_condition (n : ℕ) : 
  (∃ d : ℕ, d < 10 ∧ n = 62684 * 10 + d * 10) →
  (n % 8 = 0 ∧ n % 5 = 0) ↔ 
  (∃ d : ℕ, d ∈ ({0, 2, 4, 6, 8} : Finset ℕ) ∧ n = 62684 * 10 + d * 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l981_98177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_best_l981_98120

/-- Represents a troop with security capabilities -/
structure Troop where
  security_capability : ℝ

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | SystematicSampling
  | StratifiedSampling

/-- Definition: A set of troops has varying security capabilities -/
def has_varying_capabilities (troops : List Troop) : Prop :=
  ∃ t1 t2, t1 ∈ troops ∧ t2 ∈ troops ∧ t1.security_capability ≠ t2.security_capability

/-- Assumes an appropriateness score function -/
axiom appropriateness_score : SamplingMethod → List Troop → ℝ

/-- Theorem: Given troops with varying security capabilities, 
    stratified sampling is the most appropriate method -/
theorem stratified_sampling_best 
  (troops : List Troop) 
  (h : has_varying_capabilities troops) : 
  SamplingMethod.StratifiedSampling = 
    (SamplingMethod.Lottery :: 
     SamplingMethod.RandomNumberTable :: 
     SamplingMethod.SystematicSampling :: 
     SamplingMethod.StratifiedSampling :: []).argmax 
      (λ method => appropriateness_score method troops) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_best_l981_98120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_area_from_triangle_l981_98181

/-- The area of the largest circle formed from a string with length equal to the perimeter of an equilateral triangle with area 100 square units -/
noncomputable def largest_circle_area : ℝ := 300 / Real.pi

/-- The area of the largest circle, rounded to the nearest whole number -/
noncomputable def rounded_circle_area : ℕ := Int.toNat (round largest_circle_area)

theorem largest_circle_area_from_triangle : rounded_circle_area = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_circle_area_from_triangle_l981_98181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l981_98147

noncomputable def f (x : ℝ) : ℝ := x - 1/x

theorem f_properties :
  (f (-2) = -3/2) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x : ℝ, 1/2 ≤ x ∧ x ≤ 2 → f x ≤ 3/2) ∧
  (∀ x : ℝ, 1/2 ≤ x ∧ x ≤ 2 → -3/2 ≤ f x) ∧
  (f 2 = 3/2) ∧
  (f (1/2) = -3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l981_98147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_laws_do_not_hold_l981_98150

-- Define the @ operation
noncomputable def at_op (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem distributive_laws_do_not_hold :
  ∃ x y z : ℝ,
    (at_op x (y + z) ≠ at_op x y + at_op x z) ∧
    (x + at_op y z ≠ at_op (x + y) (x + z)) ∧
    (at_op x (at_op y z) ≠ at_op (at_op x y) (at_op x z)) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distributive_laws_do_not_hold_l981_98150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l981_98124

noncomputable section

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.sqrt 3) + (1/3) * (Real.sin (3*x))^2 / Real.cos (6*x)

-- State the theorem
theorem f_derivative (x : ℝ) :
  deriv f x = 2 * (Real.tan (6*x) / Real.cos (6*x)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l981_98124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l981_98149

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + Real.log x

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧
  (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f x ≤ f c) ∧
  f c = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l981_98149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equals_four_l981_98134

/-- The sum of the series 8^k / ((2^k - 1)(2^(k+1) - 1)) from k = 1 to infinity equals 4 -/
theorem infinite_sum_equals_four :
  (∑' k : ℕ, (8:ℝ)^(k+1) / ((2:ℝ)^(k+1) - 1) / ((2:ℝ)^(k+2) - 1)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equals_four_l981_98134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_P_terms_of_P_P_ascending_powers_l981_98169

/-- The polynomial P(a,b) = 2a^4 + a^3b^2 - 5a^2b^3 + a - 1 -/
def P (a b : ℝ) : ℝ := 2*a^4 + a^3*b^2 - 5*a^2*b^3 + a - 1

/-- The degree of P(a,b) is 4 -/
theorem degree_of_P : Nat := 4

/-- P(a,b) has 5 terms -/
theorem terms_of_P : Nat := 5

/-- P(a,b) arranged in ascending powers of a -/
theorem P_ascending_powers (a b : ℝ) : 
  P a b = -1 + a - 5*a^2*b^3 + a^3*b^2 + 2*a^4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_P_terms_of_P_P_ascending_powers_l981_98169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_fraction_problem_l981_98163

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_fraction_problem (some_fraction : ℝ) :
  (floor 6.5) * (floor some_fraction) + (floor 2) * (7.2 : ℝ) + (floor 8.4) - (9.8 : ℝ) = 12.599999999999998 →
  floor some_fraction = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_fraction_problem_l981_98163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_om_length_ma_length_cfm_area_l981_98112

-- Define the given geometric objects
structure Point := (x y : ℝ)

def Triangle (A B C : Point) : Type := Unit

def Circle (center : Point) (radius : ℝ) : Type := Unit

-- Define distance function
def dist (p q : Point) : ℝ := sorry

-- Define area function
noncomputable def Area (t : Type) : ℝ := sorry

-- Define the given conditions
variable (C F M O T P K A : Point)
variable (ω : Circle O 6)
variable (Ω : Circle T (5 * Real.sqrt 13 / 2))

axiom inscribed_ω : True  -- Replace with actual condition when available
axiom touches_ω_CM : True -- Replace with actual condition when available
axiom touches_ω_FM : True -- Replace with actual condition when available

axiom circumscribed_Ω : True -- Replace with actual condition when available

axiom area_ratio : Area (Triangle C F T) / Area (Triangle C F M) = 5 / 8

-- Define the statements to be proved
theorem om_length : dist O M = 5 * Real.sqrt 13 := by sorry

theorem ma_length : dist M A = 20 * Real.sqrt 13 / 3 := by sorry

theorem cfm_area : Area (Triangle C F M) = 204 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_om_length_ma_length_cfm_area_l981_98112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_sum_value_l981_98136

/-- The sum of 1/(3^a * 5^b * 7^c) for all triples (a,b,c) of positive integers such that 1 ≤ a < b < c -/
noncomputable def triple_sum : ℝ :=
  ∑' (a : ℕ), ∑' (b : ℕ), ∑' (c : ℕ), 
    if 1 ≤ a ∧ a < b ∧ b < c then 1 / (3^a * 5^b * 7^c) else 0

theorem triple_sum_value : triple_sum = 1 / 21216 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triple_sum_value_l981_98136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_real_roots_is_one_l981_98105

/-- A polynomial of degree 300 with real coefficients -/
def Polynomial300 : Type := Polynomial ℝ

/-- The property that a polynomial has degree 300 -/
def hasDegree300 (p : Polynomial300) : Prop := Polynomial.degree p = 300

/-- The property that a polynomial has real coefficients -/
def hasRealCoeffs (p : Polynomial300) : Prop := True  -- Always true for Polynomial ℝ

/-- The roots of a polynomial -/
noncomputable def roots (p : Polynomial300) : Finset ℂ := sorry

/-- The property that there are exactly 150 distinct absolute values among the roots -/
def has150DistinctAbsValues (p : Polynomial300) : Prop :=
  Finset.card ((roots p).image Complex.abs) = 150

/-- The number of real roots of a polynomial -/
noncomputable def numRealRoots (p : Polynomial300) : ℕ := sorry

/-- The main theorem -/
theorem min_real_roots_is_one (p : Polynomial300)
  (h1 : hasDegree300 p)
  (h2 : hasRealCoeffs p)
  (h3 : has150DistinctAbsValues p) :
  ∃ q : Polynomial300, 
    hasDegree300 q ∧ 
    hasRealCoeffs q ∧ 
    has150DistinctAbsValues q ∧
    numRealRoots q = 1 ∧
    ∀ r : Polynomial300, 
      hasDegree300 r → 
      hasRealCoeffs r → 
      has150DistinctAbsValues r → 
      numRealRoots r ≥ 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_real_roots_is_one_l981_98105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_array_convergence_l981_98109

/-- Represents a circular array of 2^n numbers, each being either 1 or -1 -/
def CircularArray (n : ℕ) := Fin (2^n) → Int

/-- Represents one operation on the circular array -/
def operation (n : ℕ) (arr : CircularArray n) : CircularArray n :=
  fun i => (arr i) * (arr ((i + 1) % (2^n)))

/-- Predicate to check if all elements in the array are 1 -/
def allOnes (n : ℕ) (arr : CircularArray n) : Prop :=
  ∀ i, arr i = 1

/-- Main theorem: There exists a finite number of operations that transform any initial configuration to all ones -/
theorem circular_array_convergence (n : ℕ) :
  ∀ (initial : CircularArray n), ∃ (k : ℕ), allOnes n (Nat.iterate (operation n) k initial) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_array_convergence_l981_98109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_equals_two_l981_98132

/-- The function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The function g(x) = a*ln(x) where a ≠ 0 -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * x

/-- The derivative of g(x) -/
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := a / x

theorem common_tangent_implies_a_equals_two (a : ℝ) (h : a ≠ 0) :
  f' 1 = g' a 1 → a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_a_equals_two_l981_98132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l981_98110

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (4, 7)
def C : ℝ × ℝ := (8, 3)

-- Define the area function for a triangle
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

-- Define the circumcenter function for a triangle
noncomputable def circumcenter (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  let d := 2 * ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
  let ux := ((x1^2 + y1^2) * (y2 - y3) + (x2^2 + y2^2) * (y3 - y1) + (x3^2 + y3^2) * (y1 - y2)) / d
  let uy := ((x1^2 + y1^2) * (x3 - x2) + (x2^2 + y2^2) * (x1 - x3) + (x3^2 + y3^2) * (x2 - x1)) / d
  (ux, uy)

theorem triangle_abc_properties :
  triangleArea A B C = 16 ∧ circumcenter A B C = (9/2, 7/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l981_98110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similarity_statements_l981_98197

-- Define the concept of similarity for shapes
def similar {α : Type} (shape1 shape2 : α) : Prop :=
  sorry

-- Define geometric shapes
structure RightTriangle : Type :=
  (side1 side2 hypotenuse : ℝ)

structure Square : Type :=
  (side : ℝ)

structure IsoscelesTriangle : Type :=
  (base equalSides : ℝ)

structure Rhombus : Type :=
  (side diag1 diag2 : ℝ)

-- State the theorem
theorem similarity_statements :
  (∃! n : ℕ, n = 1) ∧
  ((∀ (t1 t2 : RightTriangle), similar t1 t2) ∨
   (∀ (s1 s2 : Square), similar s1 s2) ∨
   (∀ (i1 i2 : IsoscelesTriangle), similar i1 i2) ∨
   (∀ (r1 r2 : Rhombus), similar r1 r2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_similarity_statements_l981_98197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l981_98194

def is_permutation (p q r s : ℕ) : Prop :=
  Multiset.ofList [p, q, r, s] = Multiset.ofList [1, 3, 5, 7]

theorem max_value_of_expression (p q r s : ℕ) 
  (h : is_permutation p q r s) : 
  p * q + q * r + r * s + s * p ≤ 64 ∧ 
  ∃ (a b c d : ℕ), is_permutation a b c d ∧ 
    a * b + b * c + c * d + d * a = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_expression_l981_98194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_isothermal_compression_l981_98159

/-- Work done in isothermal compression of an ideal gas in a cylinder -/
theorem work_isothermal_compression (p₀ H h R : ℝ) (p₀_pos : p₀ > 0) (H_pos : H > 0) (h_pos : h > 0) (R_pos : R > 0) (h_lt_H : h < H) :
  let S := π * R^2
  let V₀ := S * H
  let p (x : ℝ) := p₀ * H / (H - x)
  let F (x : ℝ) := p x * S
  let work := ∫ x in Set.Icc 0 h, F x
  ∃ ε > 0, |work - 97200| < ε :=
by
  sorry

#check work_isothermal_compression

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_isothermal_compression_l981_98159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_president_vp_committee_selection_l981_98141

theorem president_vp_committee_selection (n : ℕ) (h : n = 10) :
  (n * (n - 1)) * (Nat.choose (n - 2) 3) = 5040 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_president_vp_committee_selection_l981_98141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_add_one_frac_periodic_l981_98195

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - floor x

-- Theorem 1: f(x + 1) = f(x) + 1
theorem floor_add_one (x : ℝ) : floor (x + 1) = floor x + 1 := by sorry

-- Theorem 2: g(x) = x - f(x) is periodic with period 1
theorem frac_periodic (x : ℝ) : frac (x + 1) = frac x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_add_one_frac_periodic_l981_98195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_range_l981_98161

noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then Real.cos (2 * Real.pi * x - 2 * Real.pi * a)
  else x^2 - 2*(a+1)*x + a^2 + 5

def has_exactly_six_zeros (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
    0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧ x₅ < x₆ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0 ∧ f a x₅ = 0 ∧ f a x₆ = 0 ∧
    ∀ x, x > 0 ∧ f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ ∨ x = x₅ ∨ x = x₆

theorem f_zeros_range :
  ∀ a : ℝ, has_exactly_six_zeros a ↔ (5/2 < a ∧ a ≤ 11/4) ∨ (2 < a ∧ a ≤ 9/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_range_l981_98161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_is_six_l981_98184

/- The set S containing elements from 1 to 10 -/
def S : Finset ℕ := Finset.range 10

/- A family of subsets of S -/
def A : ℕ → Finset ℕ := sorry

/- The number of subsets in the family -/
def k : ℕ := sorry

/- All subsets in the family are subsets of S -/
axiom subset_of_S : ∀ i, A i ⊆ S

/- Each subset in the family has exactly 5 elements -/
axiom card_eq_five : ∀ i, (A i).card = 5

/- The intersection of any two distinct subsets in the family has at most 2 elements -/
axiom intersection_le_two : ∀ i j, i < j → (A i ∩ A j).card ≤ 2

/- The maximum value of k is 6 -/
theorem max_k_is_six : k ≤ 6 ∧ ∃ A : ℕ → Finset ℕ, 
  (∀ i, A i ⊆ S) ∧ 
  (∀ i, (A i).card = 5) ∧ 
  (∀ i j, i < j → (A i ∩ A j).card ≤ 2) ∧ 
  (∃ k : ℕ, k = 6) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_is_six_l981_98184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l981_98127

noncomputable def η : ℝ → ℝ := fun x => if x ≥ 0 then 1 else 0

def solution_set : Set ℝ :=
  {x | ∃ k : ℤ, x = 2 * Real.pi * k} ∪
  {x | ∃ k ∈ ({-1,0,1,2,3,4} : Set ℤ), x = -Real.pi/2 + 2 * Real.pi * k} ∪
  {x | ∃ k : ℤ, k ∉ ({-1,0,1,2,3} : Set ℤ) ∧ x = Real.pi/2 + 2 * Real.pi * k} ∪
  {x | ∃ m : ℤ, m ≥ -2 ∧ m ≤ 8 ∧ x = -Real.pi/4 + Real.pi * m}

theorem equation_solution :
  ∀ x : ℝ, x ∈ solution_set ↔ Real.cos (2 * x * (η (x + 3 * Real.pi) - η (x - 8 * Real.pi))) = Real.sin x + Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l981_98127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l981_98144

-- Define the function f(x) = ln x + x - 4
noncomputable def f (x : ℝ) := Real.log x + x - 4

-- State the theorem
theorem solution_in_interval :
  ∃ x₀ : ℝ, f x₀ = 0 ∧ 2 < x₀ ∧ x₀ < 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l981_98144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_guilty_l981_98193

-- Define the suspects as variables
variable (A B C D : Prop)

-- Define the predicates for guilt
def guilty (x : Prop) : Prop := x

-- Define the conditions
axiom condition1 : guilty A → guilty B
axiom condition2 : guilty B → (guilty C ∨ ¬guilty A)
axiom condition3 : ¬guilty D → (guilty A ∧ ¬guilty C)
axiom condition4 : guilty D → guilty A

-- Theorem to prove
theorem all_guilty : guilty A ∧ guilty B ∧ guilty C ∧ guilty D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_guilty_l981_98193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_same_score_l981_98111

theorem min_students_same_score (total_students : ℕ) (lowest_score highest_score : ℕ) 
  (h1 : total_students = 8000)
  (h2 : lowest_score = 30)
  (h3 : highest_score = 83)
  (h4 : lowest_score ≤ highest_score) :
  ∃ (score : ℕ), 
    (score ≥ lowest_score ∧ score ≤ highest_score) ∧ 
    (Finset.filter (fun s ↦ lowest_score ≤ s ∧ s ≤ highest_score) (Finset.range (total_students + 1))).card ≥ 149 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_students_same_score_l981_98111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisection_l981_98119

-- Define a triangle
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

-- Define a line
structure Line where
  start : EuclideanSpace ℝ (Fin 2)
  direction : EuclideanSpace ℝ (Fin 2)

-- Function to check if a point is on a line segment
def isOnSegment (P A B : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Function to check if two lines are parallel
def isParallel (l1 l2 : Line) : Prop := sorry

-- Function to calculate the area of a triangle
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

-- Function to calculate the area of a region formed by a line intersecting a triangle
noncomputable def regionArea (t : Triangle) (l : Line) : ℝ := sorry

theorem triangle_bisection (t : Triangle) (O : EuclideanSpace ℝ (Fin 2)) 
  (h : isOnSegment O t.B t.C) : 
  ∃ (l : Line), isParallel l (Line.mk t.A (O - t.A)) ∧ 
  regionArea t l = (1 / 2) * triangleArea t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisection_l981_98119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l981_98187

-- Define the sum of the geometric series
noncomputable def S (r : ℝ) : ℝ := 18 / (1 - r)

-- State the theorem
theorem geometric_series_sum (a : ℝ) : 
  -1 < a → a < 1 → S a * S (-a) = 3024 → S a + S (-a) = 337.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_sum_l981_98187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_5pi_6_l981_98158

theorem sin_2alpha_plus_5pi_6 (α : ℝ) 
  (h : Real.cos (α + π/6) = 1/4) : 
  Real.sin (2*α + 5*π/6) = -7/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_5pi_6_l981_98158
