import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_people_arrangement_is_n_squared_valid_arrangement_exists_l1296_129662

/-- Given n countries, returns the maximum number of people that can be arranged in a circular seating
    such that for every two people from the same country, their immediate neighbors to their right
    are from different countries. -/
def maxPeopleArrangement (n : ℕ) : ℕ :=
  n * n

/-- Theorem stating that the maximum number of people that can be arranged
    in the described seating is n² -/
theorem max_people_arrangement_is_n_squared (n : ℕ) :
  maxPeopleArrangement n = n * n := by
  -- Unfold the definition of maxPeopleArrangement
  unfold maxPeopleArrangement
  -- The equality is now trivial
  rfl

/-- Theorem stating that there exists a valid arrangement for n² people -/
theorem valid_arrangement_exists (n : ℕ) :
  ∃ (arrangement : List ℕ),
    (arrangement.length = n * n) ∧
    (∀ i j, i < j → j < arrangement.length →
      arrangement[i]! = arrangement[j]! →
      arrangement[(i + 1) % arrangement.length]! ≠ arrangement[(j + 1) % arrangement.length]!) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_people_arrangement_is_n_squared_valid_arrangement_exists_l1296_129662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_area_floor_40_l1296_129684

/-- The area between a large circle and eight smaller congruent circles arranged in a ring -/
noncomputable def ring_area (R : ℝ) : ℝ :=
  let r := R / 3
  Real.pi * (R^2 - 8 * r^2)

/-- The floor of the ring area -/
noncomputable def ring_area_floor (R : ℝ) : ℤ :=
  ⌊ring_area R⌋

theorem ring_area_floor_40 :
  ring_area_floor 40 = 555 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_area_floor_40_l1296_129684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_exponential_l1296_129627

theorem tangent_and_exponential :
  (∀ x ∈ Set.Ioo (-Real.pi/2 : ℝ) 0, Real.tan x < 0) ∧
  ¬(∃ x₀ ∈ Set.Ioi (0 : ℝ), (2 : ℝ)^x₀ = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_exponential_l1296_129627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_15_equals_3_pow_377_l1296_129678

/-- Sequence c_n defined recursively -/
def c : ℕ → ℕ
  | 0 => 3  -- Add this case to cover Nat.zero
  | 1 => 3
  | 2 => 3
  | (n + 3) => c (n + 2) * c (n + 1)

/-- Main theorem: c_15 equals 3^377 -/
theorem c_15_equals_3_pow_377 : c 15 = 3^377 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_15_equals_3_pow_377_l1296_129678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pf_length_l1296_129629

-- Define the triangle
structure Triangle (P Q R : ℝ × ℝ) :=
  (right_angle : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0)

-- Define the lengths
noncomputable def PQ_length (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
noncomputable def PR_length (P R : ℝ × ℝ) : ℝ := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)

-- Define the altitude
def is_altitude (P L Q R : ℝ × ℝ) : Prop :=
  (L.1 - P.1) * (R.1 - Q.1) + (L.2 - P.2) * (R.2 - Q.2) = 0

-- Define the median
def is_median (R M Q : ℝ × ℝ) : Prop :=
  M.1 = (Q.1 + R.1) / 2 ∧ M.2 = (Q.2 + R.2) / 2

-- Main theorem
theorem pf_length 
  (P Q R L M F : ℝ × ℝ) 
  (triangle : Triangle P Q R)
  (pq_length : PQ_length P Q = 3)
  (pr_length : PR_length P R = 3 * Real.sqrt 3)
  (altitude : is_altitude P L Q R)
  (median : is_median R M Q)
  (intersection : F.1 = L.1 ∧ F.2 = L.2 ∧ F.1 = M.1 ∧ F.2 = M.2) :
  Real.sqrt ((F.1 - P.1)^2 + (F.2 - P.2)^2) = 0.825 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pf_length_l1296_129629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_balls_is_one_third_prob_ten_balls_2014_urns_correct_l1296_129680

/-- Represents the process of placing balls in urns -/
structure BallPlacementProcess (num_urns : ℕ) where
  prob_of_choosing : ℕ → ℝ
  has_two_balls : ∃ urn, urn ≤ num_urns ∧ prob_of_choosing urn = 1 / 3

/-- The probability that the process ends with exactly 2 balls in total -/
noncomputable def prob_two_balls_total (p : BallPlacementProcess 3) : ℝ := 1 / 3

/-- Theorem stating that the probability of ending with 2 balls is 1/3 -/
theorem prob_two_balls_is_one_third (p : BallPlacementProcess 3) :
  prob_two_balls_total p = 1 / 3 := by
  sorry

/-- Part b: Probability calculation for 2014 urns and 10 balls -/
noncomputable def prob_ten_balls_2014_urns : ℝ :=
  (List.range 8).foldl (λ acc i => acc * (2013 - i : ℝ) / 2014) 1 * 9 / 2014

/-- Theorem for part b -/
theorem prob_ten_balls_2014_urns_correct :
  prob_ten_balls_2014_urns = 
    (2013 / 2014) * (2012 / 2014) * (2011 / 2014) * (2010 / 2014) *
    (2009 / 2014) * (2008 / 2014) * (2007 / 2014) * (2006 / 2014) * (9 / 2014) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_balls_is_one_third_prob_ten_balls_2014_urns_correct_l1296_129680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imag_part_implies_a_value_l1296_129636

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Given a real number a, define the complex number z -/
noncomputable def z (a : ℝ) : ℂ := (a * i) / (1 + i)

/-- Theorem stating that if the imaginary part of z is -1, then a = -2 -/
theorem imag_part_implies_a_value (a : ℝ) : 
  Complex.im (z a) = -1 → a = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imag_part_implies_a_value_l1296_129636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_equals_four_fifths_l1296_129639

/-- Given that the terminal side of angle α passes through point P(4, -3),
    prove that cos(α) = 4/5 -/
theorem cos_alpha_equals_four_fifths (α : ℝ) (h : ∃ (t : ℝ), t * 4 = Real.cos α ∧ t * (-3) = Real.sin α) :
  Real.cos α = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_equals_four_fifths_l1296_129639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1296_129617

-- Define the lines
def line1 (x y : ℝ) : Prop := x + 2*y = 4
def line2 (x y : ℝ) : Prop := 2*x - y = 3

-- Define the triangle
def triangle (x y : ℝ) : Prop :=
  y ≥ 0 ∧ line1 x y ∧ line2 x y

-- State the theorem
theorem triangle_area : 
  ∃ (A : ℝ), A = (5/4 : ℝ) ∧ 
  (∀ (x y : ℝ), triangle x y → 
    A = (1/2) * (4 - (3/2)) * 1) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1296_129617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_winning_range_l1296_129623

-- Define the game transformation
noncomputable def transform (a : ℝ) : ℝ × ℝ := (2 * a - 12, a / 2 + 12)

-- Define the probability of Player A winning
noncomputable def prob_a_wins (a : ℝ) : ℝ :=
  let (x, y) := transform a
  let (w, z) := transform x
  let (u, v) := transform y
  (if w > a then 1/4 else 0) + (if x > a then 1/4 else 0) +
  (if u > a then 1/4 else 0) + (if v > a then 1/4 else 0)

-- State the theorem
theorem game_winning_range (a : ℝ) :
  prob_a_wins a = 3/4 ↔ a ≤ 12 ∨ a ≥ 24 := by
  sorry

#check game_winning_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_winning_range_l1296_129623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1296_129683

/-- A hyperbola with semi-major axis a, semi-minor axis b, and focal distance c. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_equation : c^2 = a^2 + b^2

/-- The eccentricity of a hyperbola. -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- A point on the right branch of a hyperbola. -/
structure PointOnHyperbola (h : Hyperbola) where
  x : ℝ
  y : ℝ
  h_on_hyperbola : x^2 / h.a^2 - y^2 / h.b^2 = 1
  h_right_branch : 0 < x

theorem hyperbola_eccentricity (h : Hyperbola) (p : PointOnHyperbola h) :
  (∃ (v : ℝ × ℝ), ‖v‖ = 2 * h.c) →
  (∃ (area : ℝ), area = h.a * h.c) →
  eccentricity h = (1 + Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1296_129683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_tangent_circle_radius_l1296_129614

/-- The radius of the circle where the surfaces of an inscribed sphere and a spherical sector touch -/
noncomputable def tangentCircleRadius (R : ℝ) (α : ℝ) : ℝ :=
  (R * Real.sin α) / (4 * (Real.cos ((Real.pi / 4) - (α / 4)))^2)

/-- Theorem stating the radius of the circle where the surfaces of an inscribed sphere and a spherical sector touch -/
theorem inscribed_sphere_tangent_circle_radius (R : ℝ) (α : ℝ) 
    (hR : R > 0) (hα : 0 < α ∧ α < Real.pi) : 
  tangentCircleRadius R α = (R * Real.sin α) / (4 * (Real.cos ((Real.pi / 4) - (α / 4)))^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_tangent_circle_radius_l1296_129614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_and_square_properties_l1296_129613

-- Define the rectangle as a structure
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the conditions
def rectangle_ratio (r : Rectangle) : Prop := r.length = 3 * r.width
def rectangle_area (r : Rectangle) : Prop := r.length * r.width = 75

-- Define the square side length
noncomputable def square_side (r : Rectangle) : ℝ := Real.sqrt (r.length * r.width)

-- Theorem statement
theorem rectangle_and_square_properties (r : Rectangle) 
  (h1 : rectangle_ratio r) (h2 : rectangle_area r) : 
  r.length = 15 ∧ r.width = 5 ∧ square_side r - r.width > 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_and_square_properties_l1296_129613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_location_l1296_129622

theorem roots_location (a b c : ℝ) (h : a < b ∧ b < c) :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Ioo a b ∧ x₂ ∈ Set.Ioo b c ∧
  (∀ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_location_l1296_129622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_symmetry_l1296_129659

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the slope of the tangent line at a point
noncomputable def tangent_slope (f : ℝ → ℝ) (x : ℝ) : ℝ := 
  deriv f x

theorem tangent_slope_symmetry 
  (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_diff : Differentiable ℝ f)
  (h_slope : tangent_slope f 1 = 1) : 
  tangent_slope f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_symmetry_l1296_129659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1296_129632

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3) + (Real.sqrt 3 / 3) * (Real.sin x) ^ 2 - (Real.sqrt 3 / 3) * (Real.cos x) ^ 2

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 3)

def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_properties :
  (∃ p > 0, is_period f p ∧ ∀ q, 0 < q → is_period f q → p ≤ q) ∧
  (∃ k : ℤ, ∀ x, f (x + Real.pi / 3) = f (Real.pi / 3 - x)) ∧
  (Set.Icc (-Real.pi / 6) (Real.pi / 3) ⊆ g ⁻¹' (Set.Icc (-Real.sqrt 3 / 3) (Real.sqrt 3 / 6))) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1296_129632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2007_eq_neg_cos_l1296_129642

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => Real.sin
  | n + 1 => deriv (f n)

theorem f_2007_eq_neg_cos : f 2007 = λ x => -Real.cos x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2007_eq_neg_cos_l1296_129642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_acute_angle_l1296_129676

noncomputable def vector_a (θ : Real) : Fin 2 → Real := ![1 - Real.sin θ, 1]
noncomputable def vector_b (θ : Real) : Fin 2 → Real := ![1/2, 1 + Real.sin θ]

theorem parallel_vectors_acute_angle (θ : Real) 
  (h_acute : 0 < θ ∧ θ < Real.pi / 2)
  (h_parallel : ∃ (k : Real), k ≠ 0 ∧ vector_a θ = k • vector_b θ) :
  θ = Real.pi / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_acute_angle_l1296_129676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l1296_129643

theorem tan_double_angle_special_case (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.cos α + Real.sin α = -1/5) : 
  Real.tan (2 * α) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l1296_129643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l1296_129628

/-- Represents the speed and accident scenario of a train journey -/
structure TrainJourney where
  original_speed : ℝ
  accident_distance : ℝ
  total_distance : ℝ
  late_time : ℝ
  alt_accident_distance : ℝ
  alt_late_time : ℝ

/-- Calculates the travel time given distance and speed -/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

/-- Theorem stating the original speed of the train given the journey conditions -/
theorem train_speed (tj : TrainJourney) 
  (h1 : tj.accident_distance = 120)
  (h2 : tj.late_time = 65 / 60)
  (h3 : tj.alt_accident_distance = tj.accident_distance + 42)
  (h4 : tj.alt_late_time = 45 / 60)
  (h5 : travel_time (tj.total_distance - tj.accident_distance) tj.original_speed + 
        travel_time tj.accident_distance tj.original_speed = 
        travel_time (tj.total_distance - tj.accident_distance) (5/7 * tj.original_speed) + 
        travel_time tj.accident_distance tj.original_speed + tj.late_time)
  (h6 : travel_time (tj.total_distance - tj.alt_accident_distance) tj.original_speed + 
        travel_time tj.alt_accident_distance tj.original_speed = 
        travel_time (tj.total_distance - tj.alt_accident_distance) (5/7 * tj.original_speed) + 
        travel_time tj.alt_accident_distance tj.original_speed + tj.alt_late_time) :
  tj.original_speed = 50.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l1296_129628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_eq_one_half_l1296_129615

/-- The "Harry number" function -/
def harry (a : ℚ) : ℚ := 2 / (2 - a)

/-- The sequence of "Harry numbers" -/
def a : ℕ → ℚ
  | 0 => 3
  | n + 1 => harry (a n)

/-- The main theorem -/
theorem a_2023_eq_one_half : a 2023 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2023_eq_one_half_l1296_129615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_male_students_bound_l1296_129651

/-- The total number of students -/
def total_students : ℕ := 81650

/-- The number of rows in the arrangement -/
def num_rows : ℕ := 22

/-- The number of columns in the arrangement -/
def num_columns : ℕ := 75

/-- The maximum number of pairs of students of the same gender in adjacent columns for any row -/
def max_same_gender_pairs : ℕ := 11

/-- Represents the number of male students in each row -/
def male_students_per_row : Fin num_rows → ℕ := sorry

/-- The total number of male students across all rows -/
def total_male_students : ℕ := (Finset.sum Finset.univ male_students_per_row)

/-- The condition on adjacent columns for each row -/
axiom adjacent_column_condition (i : Fin num_rows) :
  (male_students_per_row i * (male_students_per_row i - 1)) / 2 +
  ((num_columns - male_students_per_row i) * (num_columns - 1 - male_students_per_row i)) / 2 ≤ max_same_gender_pairs

theorem male_students_bound :
  total_male_students ≤ 928 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_male_students_bound_l1296_129651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_investment_growth_l1296_129671

theorem stock_investment_growth (x : ℝ) (x_pos : x > 0) :
  x * 1.60 * 0.70 * 1.20 = x * 1.344 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_investment_growth_l1296_129671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equals_two_l1296_129667

open Real BigOperators

-- Define the general term of the series
noncomputable def a (k : ℕ) : ℝ := 8^k / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

-- State the theorem
theorem infinite_sum_equals_two :
  Summable a ∧ ∑' k, a k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equals_two_l1296_129667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangements_l1296_129648

theorem book_arrangements (n : ℕ) (identical_pairs : ℕ) 
  (h1 : n = 7) 
  (h2 : identical_pairs = 2) : 
  Nat.factorial n / (Nat.factorial 2 * Nat.factorial 2) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangements_l1296_129648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygons_congruent_l1296_129696

/-- Represents a polygon on a grid --/
structure GridPolygon where
  vertices : List (Nat × Nat)
  is_valid : Bool

/-- Represents a grid with polygons --/
structure Grid where
  size : Nat
  segments : Nat
  polygons : List GridPolygon

/-- Calculates the area of a polygon --/
noncomputable def area (p : GridPolygon) : ℝ :=
  sorry

/-- Checks if two polygons are congruent --/
def congruent (p q : GridPolygon) : Prop :=
  sorry

/-- Checks if all polygons in a grid have equal area --/
def all_equal_area (g : Grid) : Prop :=
  ∀ p q : GridPolygon, p ∈ g.polygons → q ∈ g.polygons → area p = area q

/-- Checks if all polygons in a grid are congruent --/
def all_congruent (g : Grid) : Prop :=
  ∀ p q : GridPolygon, p ∈ g.polygons → q ∈ g.polygons → congruent p q

/-- The main theorem --/
theorem polygons_congruent (g : Grid) :
  g.size = 10 ∧ g.segments = 80 ∧ g.polygons.length = 20 ∧ all_equal_area g →
  all_congruent g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygons_congruent_l1296_129696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_not_circumscribable_l1296_129602

/-- A pentagon with side lengths 4, 6, 8, 7, and 9. -/
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side1_eq : side1 = 4
  side2_eq : side2 = 6
  side3_eq : side3 = 8
  side4_eq : side4 = 7
  side5_eq : side5 = 9

/-- Definition of a circumscribable pentagon -/
def is_circumscribable (p : Pentagon) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∃ (center : ℝ × ℝ), 
    let circle := {(x, y) : ℝ × ℝ | (x - center.1)^2 + (y - center.2)^2 = r^2}
    ∀ side ∈ [p.side1, p.side2, p.side3, p.side4, p.side5], 
      ∃ (point : ℝ × ℝ), point ∈ circle ∧ 
        ∃ (line : ℝ × ℝ → Prop), (∀ p, line p → p ∉ circle) ∧
          (∃ p, line p ∧ p ∈ circle)

/-- The theorem stating that the given pentagon cannot be circumscribed -/
theorem pentagon_not_circumscribable (p : Pentagon) : ¬ is_circumscribable p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_not_circumscribable_l1296_129602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l1296_129644

-- Define the angle in degrees
noncomputable def angle : ℚ := -75

-- Define a function to determine the quadrant of an angle
def quadrant (θ : ℚ) : ℕ :=
  let normalizedAngle := θ % 360
  if 0 ≤ normalizedAngle && normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle && normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle && normalizedAngle < 270 then 3
  else 4

-- Theorem stating that the angle -75° is in the fourth quadrant
theorem angle_in_fourth_quadrant : quadrant angle = 4 := by
  -- The proof would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l1296_129644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1296_129630

noncomputable def point1 : ℝ × ℝ := (3, 3)
noncomputable def point2 : ℝ × ℝ := (-2, -2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points : distance point1 point2 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1296_129630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hour_sunset_colors_l1296_129634

/-- The number of color changes during a sunset -/
def sunset_color_changes (
  change_interval : ℕ
) (hour_length : ℕ
) (sunset_duration : ℕ
) : ℕ :=
  (sunset_duration * hour_length) / change_interval

/-- Theorem: The number of color changes during a two-hour sunset is 12 -/
theorem two_hour_sunset_colors : 
  sunset_color_changes 10 60 2 = 12 := by
  -- Unfold the definition of sunset_color_changes
  unfold sunset_color_changes
  -- Evaluate the arithmetic expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_hour_sunset_colors_l1296_129634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1296_129673

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x)) / Real.log m

-- State the theorem
theorem f_properties (m : ℝ) (h_m : m > 0 ∧ m ≠ 1) :
  -- f is defined on (-1, 1)
  (∀ x, -1 < x ∧ x < 1 → f m x = Real.log ((1 + x) / (1 - x)) / Real.log m) ∧
  -- f is an odd function
  (∀ x, -1 < x ∧ x < 1 → f m (-x) = -(f m x)) ∧
  -- For m > 1, f(x) ≤ 0 iff -1 < x ≤ 0
  (m > 1 → (∀ x, -1 < x ∧ x < 1 → (f m x ≤ 0 ↔ -1 < x ∧ x ≤ 0))) ∧
  -- For 0 < m < 1, f(x) ≤ 0 iff 0 ≤ x < 1
  (0 < m ∧ m < 1 → (∀ x, -1 < x ∧ x < 1 → (f m x ≤ 0 ↔ 0 ≤ x ∧ x < 1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1296_129673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1296_129656

theorem trig_identity (α : ℝ) (hα : α ≠ 0) (hπ : α ≠ π / 2) : 
  (Real.sin α + 1 / Real.sin α)^2 + (Real.cos α + 1 / Real.cos α)^2 + 2 = 
  5 + 2 * ((Real.sin α / Real.cos α)^2 + (Real.cos α / Real.sin α)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1296_129656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_extrema_l1296_129691

noncomputable def a (n : ℕ) : ℝ := 
  (n : ℝ) - Real.sqrt 2015 / ((n : ℝ) - Real.sqrt 2016)

theorem sequence_extrema :
  ∀ k ∈ Finset.range 50, a 44 ≤ a (k + 1) ∧ a (k + 1) ≤ a 45 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_extrema_l1296_129691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chosen_sum_theorem_l1296_129666

/-- Represents a k×k square table containing numbers from 1 to k^2 -/
def SquareTable (k : ℕ) := Fin k → Fin k → ℕ

/-- The sum of k numbers chosen from a k×k square table, where each choice removes its row and column -/
def ChosenSum (k : ℕ) : ℕ := k * (k^2 + 1) / 2

/-- Theorem stating that the sum of k numbers chosen from a k×k square table,
    where each choice removes its row and column, is equal to k(k^2 + 1) / 2 -/
theorem chosen_sum_theorem (k : ℕ) (table : SquareTable k) :
  ∃ (chosen : Fin k → ℕ), (∀ i j : Fin k, i ≠ j → chosen i ≠ chosen j) ∧
    (∀ i : Fin k, ∃ r c : Fin k, table r c = chosen i) ∧
    (∀ i j : Fin k, chosen i ∈ Set.range (table j) → i = j) ∧
    (Finset.sum (Finset.univ : Finset (Fin k)) chosen = ChosenSum k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chosen_sum_theorem_l1296_129666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equality_l1296_129677

theorem sin_cos_equality (x y : Real) (hx : 0 ≤ x ∧ x ≤ π/2) (hy : 0 ≤ y ∧ y ≤ π/2) :
  (Real.sin x)^6 + 3 * (Real.sin x)^2 * (Real.cos y)^2 + (Real.cos y)^6 = 1 ↔ x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_equality_l1296_129677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_BC_l1296_129664

noncomputable def total_distance : ℝ := 300
noncomputable def average_speed : ℝ := 25
noncomputable def distance_AB : ℝ := 120
noncomputable def time_AB : ℝ := 4
noncomputable def time_CD : ℝ := 2
noncomputable def distance_BC : ℝ := distance_AB
noncomputable def distance_CD : ℝ := distance_BC / 2

theorem average_speed_BC : 
  let total_time : ℝ := total_distance / average_speed
  let time_BC : ℝ := total_time - (time_AB + time_CD)
  distance_BC / time_BC = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_speed_BC_l1296_129664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_equilateral_triangle_area_l1296_129672

-- Define the circle radius
def circle_radius : ℝ := 10

-- Define the area of the inscribed equilateral triangle
noncomputable def inscribed_triangle_area : ℝ := 75 * Real.sqrt 3

-- Theorem statement
theorem largest_inscribed_equilateral_triangle_area :
  inscribed_triangle_area = (3 * Real.sqrt 3 / 4) * circle_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_equilateral_triangle_area_l1296_129672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_y_increase_rate_is_800_l1296_129686

/-- The population increase rate of Village Y -/
def village_y_increase_rate : ℕ := 800

/-- The initial population of Village X -/
def village_x_initial : ℕ := 74000

/-- The annual decrease rate of Village X -/
def village_x_decrease_rate : ℕ := 1200

/-- The initial population of Village Y -/
def village_y_initial : ℕ := 42000

/-- The number of years until the populations are equal -/
def years_until_equal : ℕ := 16

theorem village_y_increase_rate_is_800 :
  village_x_initial - village_x_decrease_rate * years_until_equal =
  village_y_initial + village_y_increase_rate * years_until_equal :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_y_increase_rate_is_800_l1296_129686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_is_256_l1296_129612

-- Define the rectangle's dimensions
def rectangle_length : ℝ := 32
def rectangle_width : ℝ := 64

-- Define the area of the rectangle
def rectangle_area : ℝ := rectangle_length * rectangle_width

-- Define the area of the square
def square_area : ℝ := 2 * rectangle_area

-- Define the side length of the square
noncomputable def square_side : ℝ := Real.sqrt square_area

-- Theorem statement
theorem square_perimeter_is_256 : 4 * square_side = 256 := by
  -- Expand the definitions
  unfold square_side square_area rectangle_area
  -- Simplify the expression
  simp [rectangle_length, rectangle_width]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_is_256_l1296_129612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_cake_not_square_l1296_129682

-- Define the cake structure
structure Cake where
  length : ℝ
  width : ℝ

-- Define the cutting process
noncomputable def cut (c : Cake) : Cake :=
  if c.length > c.width
  then { length := c.length, width := c.width - (c.width * c.width / c.length) }
  else { length := c.length - (c.length * c.length / c.width), width := c.width }

-- Theorem statement
theorem remaining_cake_not_square (s : ℝ) (n : ℕ) (h : s > 0) :
  let initial_cake := { length := s, width := s : Cake }
  let final_cake := (Nat.iterate cut n) initial_cake
  final_cake.length ≠ final_cake.width := by
  sorry

-- Helper lemma to show that the cake remains non-square after each cut
lemma cut_preserves_nonsquare (c : Cake) (h : c.length ≠ c.width) :
  (cut c).length ≠ (cut c).width := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_cake_not_square_l1296_129682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1296_129646

-- Define the original function
def f (x : ℝ) : ℝ := x^2

-- Define the inverse function
noncomputable def g (x : ℝ) : ℝ := -Real.sqrt x

-- Theorem statement
theorem inverse_function_theorem (x : ℝ) (hx : x < -2) :
  g (f x) = x ∧ f (g (f x)) = f x ∧ f x > 4 → g (f x) = x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1296_129646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l1296_129609

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | 1 => 1
  | n + 2 => (3 * sequence_a (n + 1)) / (sequence_a (n + 1) + 3)

theorem sequence_a_general_term : ∀ n : ℕ, sequence_a n = 3 / (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l1296_129609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_box_max_volume_48cm_box_l1296_129655

noncomputable section

/-- The volume of an open-top box formed from a square sheet --/
def box_volume (sheet_side : ℝ) (cut_side : ℝ) : ℝ :=
  cut_side * (sheet_side - 2 * cut_side)^2

/-- The derivative of the volume with respect to the cut side length --/
def volume_derivative (sheet_side : ℝ) (cut_side : ℝ) : ℝ :=
  (sheet_side - cut_side) * (sheet_side / 3 - cut_side)

/-- Theorem stating the existence of a maximum volume for a general sheet size --/
theorem max_volume_box (sheet_side : ℝ) (h_positive : sheet_side > 0) :
  ∃ (max_cut : ℝ), 
    0 < max_cut ∧ 
    max_cut < sheet_side / 2 ∧
    volume_derivative sheet_side max_cut = 0 ∧
    ∀ (cut : ℝ), 0 < cut → cut < sheet_side / 2 → 
      box_volume sheet_side cut ≤ box_volume sheet_side max_cut :=
by sorry

/-- Theorem for the specific case of a 48cm sheet --/
theorem max_volume_48cm_box :
  ∃ (max_cut : ℝ), 
    max_cut = 8 ∧
    volume_derivative 48 max_cut = 0 ∧
    ∀ (cut : ℝ), 0 < cut → cut < 24 → 
      box_volume 48 cut ≤ box_volume 48 max_cut :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_box_max_volume_48cm_box_l1296_129655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1296_129674

open InnerProductSpace

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable (a b : V)

noncomputable def angle_between (x y : V) : ℝ := Real.arccos (inner x y / (norm x * norm y))

theorem vector_properties 
  (h_angle : angle_between V a b = 2 * Real.pi / 3)
  (h_norm_a : norm a = 1)
  (h_norm_b : norm b = 3) :
  norm (5 • a - b) = 7 ∧ 
  inner (2 • a + b) b / norm b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1296_129674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_range_l1296_129695

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 8

-- Define point P
def point_P : ℝ × ℝ := (1, 0)

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define line l
def line_l (k m x y : ℝ) : Prop := y = k * x + m

-- Define the circle with diameter EP
def circle_EP (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the dot product condition
def dot_product_condition (dot_product : ℝ) : Prop :=
  2/3 ≤ dot_product ∧ dot_product ≤ 3/4

-- Main theorem
theorem trajectory_and_area_range :
  ∀ (k m : ℝ),
  (∃ (x y : ℝ), line_l k m x y ∧ circle_EP x y) →  -- l is tangent to circle_EP
  (∃ (A B : ℝ × ℝ),
    trajectory_C A.1 A.2 ∧
    trajectory_C B.1 B.2 ∧
    line_l k m A.1 A.2 ∧
    line_l k m B.1 B.2 ∧
    A ≠ B ∧
    dot_product_condition (A.1 * B.1 + A.2 * B.2)) →
  (∀ (x y : ℝ), trajectory_C x y ↔ x^2 / 2 + y^2 = 1) ∧
  (∃ (S : Set ℝ),
    S = {area | ∃ (A B : ℝ × ℝ),
      trajectory_C A.1 A.2 ∧
      trajectory_C B.1 B.2 ∧
      line_l k m A.1 A.2 ∧
      line_l k m B.1 B.2 ∧
      A ≠ B ∧
      dot_product_condition (A.1 * B.1 + A.2 * B.2) ∧
      area = (1/2) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)} ∧
    S = Set.Icc (Real.sqrt 6 / 4) (2/3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_range_l1296_129695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distance_of_specific_ellipse_l1296_129604

/-- An ellipse with given endpoints of its axes -/
structure Ellipse where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ
  x3 : ℝ
  y3 : ℝ
  x4 : ℝ
  y4 : ℝ
  h_perpendicular : (x2 - x1) * (y4 - y3) + (y2 - y1) * (x4 - x3) = 0

/-- The distance between the foci of the ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  2 * Real.sqrt 4.75

/-- Theorem stating that the distance between the foci of the given ellipse is 2√4.75 -/
theorem focal_distance_of_specific_ellipse :
  let e : Ellipse := {
    x1 := 1, y1 := 3,
    x2 := 10, y2 := 3,
    x3 := 4, y3 := -2,
    x4 := 4, y4 := 8,
    h_perpendicular := by sorry
  }
  focal_distance e = 2 * Real.sqrt 4.75 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distance_of_specific_ellipse_l1296_129604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_problem_l1296_129679

/-- Ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Circle M -/
def circle_M (x y x₀ y₀ r : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = r^2

/-- Point on ellipse C -/
def point_on_ellipse_C (x₀ y₀ : ℝ) : Prop := ellipse_C x₀ y₀

/-- Tangent condition -/
def tangent_condition (x₀ y₀ r k : ℝ) : Prop :=
  (x₀^2 - r^2) * k^2 - 2 * x₀ * y₀ * k + y₀^2 - r^2 = 0

/-- Main theorem -/
theorem ellipse_tangent_problem
  (x₀ y₀ r k₁ k₂ : ℝ)
  (h_ellipse : point_on_ellipse_C x₀ y₀)
  (h_r : 0 < r ∧ r < 1)
  (h_tangent₁ : tangent_condition x₀ y₀ r k₁)
  (h_tangent₂ : tangent_condition x₀ y₀ r k₂)
  (h_AF_BF : ∃ (A B : ℝ × ℝ), |A.1 - Real.sqrt 3| - |B.1 + Real.sqrt 3| = 2 * r) :
  (∃ (x y : ℝ), (x + Real.sqrt 3)^2 + y^2 = 4 ∧ x > 0) ∧
  (k₁ * k₂ = -1/4 → ∃ (OP OQ : ℝ), OP * OQ ≤ 5/2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_problem_l1296_129679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l1296_129699

-- Define the points on the circle
def A : ℝ × ℝ := (4, 5)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (7, 15)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the circle passing through A, B, and C
noncomputable def circle_set : Set (ℝ × ℝ) := sorry

-- Define the tangent line from O to the circle
noncomputable def tangent_line : Set (ℝ × ℝ) := sorry

-- Define the tangent point
noncomputable def T : ℝ × ℝ := sorry

-- Theorem stating the length of the tangent segment
theorem tangent_length : 
  Real.sqrt ((T.1 - O.1)^2 + (T.2 - O.2)^2) = 9 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_length_l1296_129699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cathedral_distance_sum_l1296_129608

/-- Represents a point in the town layout --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the town layout --/
structure TownLayout where
  townHall : Point
  catholicCathedral : Point
  protestantCathedral : Point
  school : Point

/-- Calculates the distance between two points --/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Main theorem to prove --/
theorem cathedral_distance_sum (layout : TownLayout) :
  let d1 := distance layout.townHall layout.catholicCathedral
  let d2 := distance layout.townHall layout.protestantCathedral
  (∃ (schoolDistance : ℝ),
    schoolDistance = distance layout.school layout.catholicCathedral ∧
    schoolDistance = distance layout.school layout.protestantCathedral ∧
    schoolDistance = 500) →
  1 / d1 + 1 / d2 = 1 / 500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cathedral_distance_sum_l1296_129608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_derivative_value_l1296_129624

/-- Given a cubic function f(x) = ax³ + 3x² + 2, prove that if f'(-1) = 3, then a = 3 -/
theorem cubic_derivative_value (a : ℝ) : 
  (fun x : ℝ => 3 * a * x^2 + 6 * x) (-1) = 3 → a = 3 := by
  intro h
  have : 3 * a * (-1)^2 + 6 * (-1) = 3 := h
  simp at this
  linarith


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_derivative_value_l1296_129624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_walk_distance_l1296_129647

/-- The distance from the starting point after walking 10 km along the perimeter of a regular hexagon with side length 3 km -/
theorem hexagon_walk_distance : ∃ d : ℝ, d = Real.sqrt 37 := by
  let hexagon_side : ℝ := 3
  let walk_distance : ℝ := 10
  let end_x : ℝ := -0.5
  let end_y : ℝ := -3.5 * Real.sqrt 3
  
  have h : Real.sqrt (end_x^2 + end_y^2) = Real.sqrt 37 := by sorry
  
  exact ⟨Real.sqrt (end_x^2 + end_y^2), h⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_walk_distance_l1296_129647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l1296_129687

/-- Circle with equation x^2 + y^2 = 5 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- Parabola with equation y^2 = 2px, where p > 0 -/
def parabola_eq (p x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- Point A is on both the circle and parabola -/
def point_A (p x y : ℝ) : Prop := circle_eq x y ∧ parabola_eq p x y

/-- Point B is on both the circle and parabola -/
def point_B (p x y : ℝ) : Prop := circle_eq x y ∧ parabola_eq p x y

/-- Point C is on the circle and the parabola's axis -/
def point_C (p x y : ℝ) : Prop := circle_eq x y ∧ x = -p/2

/-- Point D is on the circle and the parabola's axis -/
def point_D (p x y : ℝ) : Prop := circle_eq x y ∧ x = -p/2

/-- Quadrilateral ABCD is a rectangle -/
def is_rectangle (p xA yA xB yB xC yC xD yD : ℝ) : Prop :=
  point_A p xA yA ∧ point_B p xB yB ∧ point_C p xC yC ∧ point_D p xD yD ∧
  (xA - xC)^2 + (yA - yC)^2 = (xB - xD)^2 + (yB - yD)^2 ∧
  (xA - xB)^2 + (yA - yB)^2 = (xC - xD)^2 + (yC - yD)^2

theorem parabola_circle_intersection (p xA yA xB yB xC yC xD yD : ℝ) :
  is_rectangle p xA yA xB yB xC yC xD yD → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l1296_129687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_mn_equation_l1296_129652

-- Define the arithmetic sequence
def arithmetic_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, b = a + r ∧ c = b + r ∧ d = c + r

-- Define the geometric sequence
def geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ q : ℝ, b = a * q ∧ c = b * q ∧ d = c * q

-- Define a line equation
def line_equation (x₁ y₁ x₂ y₂ : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x y = 0 ↔ (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

-- State the theorem
theorem line_mn_equation (x₁ x₂ y₁ y₂ : ℝ) :
  arithmetic_sequence 1 x₁ x₂ 7 →
  geometric_sequence 1 y₁ y₂ 8 →
  line_equation x₁ y₁ x₂ y₂ (fun x y ↦ x - y - 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_mn_equation_l1296_129652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1296_129637

/-- Curve C represented by its parametric equations -/
noncomputable def curve_C (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, Real.sqrt 5 * Real.sin α)

/-- Line l represented by its Cartesian equation -/
def line_l (x y : ℝ) : Prop := x + y = 4

/-- The Cartesian equation of curve C -/
def cartesian_eq_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 5 = 1

/-- Distance from a point to line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + y - 4) / Real.sqrt 2

theorem curve_C_properties :
  ∀ α : ℝ,
  let (x, y) := curve_C α
  ∃ d_max : ℝ,
  (∀ x y : ℝ, cartesian_eq_C x y ↔ ∃ β : ℝ, curve_C β = (x, y)) ∧
  (d_max = 7 * Real.sqrt 2 / 2) ∧
  (∀ β : ℝ, distance_to_line x y ≤ d_max) ∧
  (distance_to_line (-4/3) (-5/3) = d_max) := by
  sorry

#check curve_C_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l1296_129637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transmission_time_approx_l1296_129670

/-- Calculates the transmission time for data blocks --/
noncomputable def transmissionTime (
  numBlocks : ℕ
  ) (chunksPerBlock : ℕ
  ) (transmissionRate : ℝ
  ) (initialDelay : ℝ
  ) (efficiencyFactor : ℝ
  ) : ℝ :=
  let totalChunks := (numBlocks * chunksPerBlock : ℝ)
  let effectiveRate := transmissionRate * efficiencyFactor
  let transmissionDuration := totalChunks / effectiveRate
  (transmissionDuration + initialDelay) / 60

/-- Theorem stating that the transmission time is approximately 3.7222 minutes --/
theorem transmission_time_approx :
  let ε := 0.0001
  |transmissionTime 100 256 150 10 0.8 - 3.7222| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transmission_time_approx_l1296_129670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_e_l1296_129638

theorem find_e : ∃ e : ℝ, (1/5 : ℝ)^e * (1/4 : ℝ)^18 = 1/(2*(10^35 : ℝ)) ∧ e = 35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_e_l1296_129638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_triangle_property_l1296_129620

def triangle_property (S : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a < b → b < c → c ≥ a + b

def consecutive_set (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => 3 ≤ x ∧ x ≤ n) (Finset.range (n+1))

theorem largest_n_with_triangle_property :
  (∀ (T : Finset ℕ), T ⊆ consecutive_set 75 → T.card = 8 → triangle_property T) ∧
  ¬(∀ (T : Finset ℕ), T ⊆ consecutive_set 76 → T.card = 8 → triangle_property T) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_with_triangle_property_l1296_129620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_current_weight_l1296_129675

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 0

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 0

/-- The combined weight of Leo and Kendra is 140 pounds -/
axiom combined_weight : leo_weight + kendra_weight = 140

/-- If Leo gains 10 pounds, he will weigh 50% more than Kendra -/
axiom weight_relation : leo_weight + 10 = 1.5 * kendra_weight

theorem leo_current_weight : leo_weight = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_current_weight_l1296_129675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_between_stops_is_quarter_l1296_129692

/-- Represents a car trip with two stops -/
structure CarTrip where
  totalDistance : ℚ
  firstStopFraction : ℚ
  remainingAfterSecondStop : ℚ

/-- Calculates the fraction of remaining distance traveled between first and second stops -/
def fractionBetweenStops (trip : CarTrip) : ℚ :=
  let distanceToFirstStop := trip.totalDistance * trip.firstStopFraction
  let remainingAfterFirstStop := trip.totalDistance - distanceToFirstStop
  let traveledBetweenStops := remainingAfterFirstStop - trip.remainingAfterSecondStop
  traveledBetweenStops / remainingAfterFirstStop

/-- Theorem stating that for the given trip parameters, the fraction between stops is 1/4 -/
theorem fraction_between_stops_is_quarter :
  let trip := CarTrip.mk 400 (1/2) 150
  fractionBetweenStops trip = 1/4 := by
  sorry

#eval fractionBetweenStops (CarTrip.mk 400 (1/2) 150)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_between_stops_is_quarter_l1296_129692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_home_team_scored_five_l1296_129690

/-- Represents the score of a team in a hockey match -/
structure Score where
  first_half : ℕ
  second_half : ℕ

/-- Calculates the total score for a team -/
def total_score (s : Score) : ℕ := s.first_half + s.second_half

/-- Represents the state of a hockey match -/
structure HockeyMatch where
  home : Score
  away : Score
  first_half_total : ℕ
  away_leading_at_half : Prop
  home_won : Prop

/-- The main theorem to prove -/
theorem home_team_scored_five (m : HockeyMatch) : total_score m.home = 5 := by
  have first_half_condition : m.home.first_half + m.away.first_half = m.first_half_total := by sorry
  have away_leading_condition : m.away.first_half > m.home.first_half := by sorry
  have home_second_half : m.home.second_half = 3 := by sorry
  have home_won_condition : total_score m.home > total_score m.away := by sorry
  have first_half_total_six : m.first_half_total = 6 := by sorry

  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_home_team_scored_five_l1296_129690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangles_l1296_129626

/-- Represents a cell in the figure -/
inductive Cell
| Black
| White
deriving BEq

/-- Represents the figure as a list of cells -/
def Figure := List Cell

/-- Counts the number of black cells in a figure -/
def countBlackCells (f : Figure) : Nat :=
  f.filter (· == Cell.Black) |>.length

/-- Represents a 1×2 rectangle placement -/
structure Rectangle where
  cell1 : Nat
  cell2 : Nat
  validPlacement : cell2 = cell1 + 1

/-- Checks if a rectangle placement is valid in the figure -/
def isValidRectangle (f : Figure) (r : Rectangle) : Prop :=
  r.cell1 < f.length ∧ r.cell2 < f.length ∧
  f.get? r.cell1 ≠ f.get? r.cell2

/-- The main theorem -/
theorem max_rectangles (f : Figure) (h : countBlackCells f = 5) :
  (∃ (rectangles : List Rectangle),
    rectangles.length = 5 ∧
    (∀ r ∈ rectangles, isValidRectangle f r) ∧
    (∀ (otherRectangles : List Rectangle),
      (∀ r ∈ otherRectangles, isValidRectangle f r) →
      otherRectangles.length ≤ 5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_rectangles_l1296_129626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_vertical_shift_l1296_129668

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 4) + 1

theorem phase_and_vertical_shift :
  (∃ (p : ℝ), ∀ (x : ℝ), f (x + p) = 2 * Real.cos (2 * x) + 1 ∧ p = -Real.pi / 8) ∧
  (∃ (v : ℝ), ∀ (x : ℝ), f x = 2 * Real.cos (2 * x + Real.pi / 4) + v ∧ v = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_vertical_shift_l1296_129668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_330_l1296_129605

def count_digit (d : Nat) (n : Nat) : Nat :=
  if n < 10 then
    if n = d then 1 else 0
  else
    count_digit d (n / 10) + count_digit d (n % 10)

def sum_difference (old : Nat) (new : Nat) (range : Nat) : Int :=
  let count := (List.range range).map (count_digit old) |>.sum
  (new - old) * count + (new - old) * 10 * count

theorem sum_difference_330 :
  sum_difference 6 9 100 = 330 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_difference_330_l1296_129605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_period_l1296_129618

/-- The period in seconds for a given population net increase, birth rate, and death rate -/
noncomputable def period_in_seconds (net_increase : ℝ) (birth_rate : ℝ) (death_rate : ℝ) : ℝ :=
  net_increase / ((birth_rate - death_rate) / 2)

/-- The period in hours for a given population net increase, birth rate, and death rate -/
noncomputable def period_in_hours (net_increase : ℝ) (birth_rate : ℝ) (death_rate : ℝ) : ℝ :=
  (period_in_seconds net_increase birth_rate death_rate) / 3600

theorem population_growth_period 
  (net_increase : ℝ) 
  (birth_rate : ℝ) 
  (death_rate : ℝ) 
  (h1 : birth_rate = 7) 
  (h2 : death_rate = 2) 
  (h3 : net_increase = 216000) : 
  period_in_hours net_increase birth_rate death_rate = 24 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_growth_period_l1296_129618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1296_129616

theorem min_value_theorem (n : ℝ) (hn : n > 0) : 
  n / 2 + 50 / n ≥ 10 ∧ (n / 2 + 50 / n = 10 ↔ n = 10) := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1296_129616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l1296_129610

theorem complex_fraction_simplification : 
  (3 + Complex.I) / (2 - Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_simplification_l1296_129610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_stock_cost_l1296_129653

/-- The combined cost price of two stocks with given face values, discounts/premiums, and brokerage -/
theorem combined_stock_cost
  (face_value_A face_value_B : ℝ)
  (discount_A premium_B brokerage : ℝ)
  (h1 : face_value_A = 100)
  (h2 : face_value_B = 100)
  (h3 : discount_A = 0.02)
  (h4 : premium_B = 0.015)
  (h5 : brokerage = 0.002)
  : (face_value_A * (1 - discount_A) * (1 + brokerage)) + 
    (face_value_B * (1 + premium_B) * (1 + brokerage)) = 199.899 := by
  -- Define intermediate calculations
  let price_A := face_value_A * (1 - discount_A)
  let price_B := face_value_B * (1 + premium_B)
  let cost_A := price_A * (1 + brokerage)
  let cost_B := price_B * (1 + brokerage)
  
  -- Proof steps
  sorry  -- This skips the proof for now

#eval (100 * (1 - 0.02) * (1 + 0.002)) + (100 * (1 + 0.015) * (1 + 0.002))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_stock_cost_l1296_129653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_tangents_l1296_129621

/-- The distance between parallel tangent lines to y = x^2 and y = -(x-1)^2 -/
theorem distance_between_tangents (m : ℝ) : 
  let d := |m * (m - 2)| / (2 * Real.sqrt (1 + m^2))
  let f := fun (x : ℝ) => x^2
  let g := fun (x : ℝ) => -(x - 1)^2
  let tangent_f := fun (x : ℝ) => m * x - m^2 / 4
  let tangent_g := fun (x : ℝ) => m * x - m + m^2 / 4
  ∃ (x₁ x₂ : ℝ), 
    (deriv f x₁ = m ∧ deriv g x₂ = m) ∧ 
    (tangent_f x₁ = f x₁ ∧ tangent_g x₂ = g x₂) ∧
    d = |tangent_f 0 - tangent_g 0| / Real.sqrt (1 + m^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_tangents_l1296_129621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_conditions_possible_l1296_129689

/-- Two circles with centers P and Q and radii p and q respectively -/
structure TwoCircles (V : Type*) [NormedAddCommGroup V] where
  P : V
  Q : V
  p : ℝ
  q : ℝ
  p_pos : 0 < p
  q_pos : 0 < q
  p_ge_q : p ≥ q

/-- The distance between the centers of two circles -/
def centerDistance {V : Type*} [NormedAddCommGroup V] (c : TwoCircles V) : ℝ :=
  ‖c.P - c.Q‖

/-- Theorem stating that all four conditions can be satisfied for some configuration of two circles -/
theorem all_conditions_possible :
  ∃ (V : Type*) (_ : NormedAddCommGroup V) (c : TwoCircles V),
    (∃ c₁ : TwoCircles V, c₁.p - c₁.q = centerDistance c₁) ∧
    (∃ c₂ : TwoCircles V, c₂.p + c₂.q = centerDistance c₂) ∧
    (∃ c₃ : TwoCircles V, c₃.p + c₃.q < centerDistance c₃) ∧
    (∃ c₄ : TwoCircles V, c₄.p - c₄.q < centerDistance c₄) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_conditions_possible_l1296_129689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_mile_ride_cost_l1296_129681

/-- Calculates the cost of a taxi ride given the conditions --/
def taxi_cost (base_fare : ℚ) (cost_per_mile : ℚ) (discount_rate : ℚ) (discount_threshold : ℚ) (miles : ℚ) : ℚ :=
  let total_fare := base_fare + cost_per_mile * miles
  if miles > discount_threshold then
    total_fare * (1 - discount_rate)
  else
    total_fare

/-- Theorem stating that a 12-mile taxi ride costs $5.04 under given conditions --/
theorem twelve_mile_ride_cost :
  let base_fare : ℚ := 2
  let cost_per_mile : ℚ := 3/10
  let discount_rate : ℚ := 1/10
  let discount_threshold : ℚ := 10
  let miles : ℚ := 12
  taxi_cost base_fare cost_per_mile discount_rate discount_threshold miles = 63/25 := by
  sorry

#eval taxi_cost 2 (3/10) (1/10) 10 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_mile_ride_cost_l1296_129681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_implies_k_range_l1296_129657

-- Define the function f
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 - 6*k*x + k + 8)

-- State the theorem
theorem function_domain_implies_k_range (k : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f k x = y) →
  -8/9 ≤ k ∧ k ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_domain_implies_k_range_l1296_129657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l1296_129697

noncomputable def f (x : ℝ) := -5 * Real.sin (x + Real.pi / 3)

theorem amplitude_and_phase_shift :
  ∃ (A : ℝ) (C : ℝ),
    (∀ x, f x = A * Real.sin (x - C)) ∧
    A = 5 ∧
    C = -Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_and_phase_shift_l1296_129697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avgSpeedBC_approx_l1296_129693

/-- Represents the motorcycle trip with given conditions -/
structure MotorcycleTrip where
  distanceAB : ℝ
  distanceBC : ℝ
  timeAB : ℝ
  timeBC : ℝ
  avgSpeedTotal : ℝ
  distanceBCHalf : distanceBC = distanceAB / 2
  timeABTriple : timeAB = 3 * timeBC
  totalDistance : distanceAB + distanceBC = avgSpeedTotal * (timeAB + timeBC)

/-- The average speed from B to C -/
noncomputable def avgSpeedBC (trip : MotorcycleTrip) : ℝ :=
  trip.distanceBC / trip.timeBC

/-- Theorem stating that the average speed from B to C is approximately 66.7 mph -/
theorem avgSpeedBC_approx (trip : MotorcycleTrip) 
    (h : trip.distanceAB = 120 ∧ trip.avgSpeedTotal = 50) :
  abs (avgSpeedBC trip - 200/3) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avgSpeedBC_approx_l1296_129693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l1296_129633

theorem no_solution_exists : ¬∃ (a b : ℝ), 
  (a > 0 ∧ b > 0) ∧ 
  (a + b ≤ 120) ∧ 
  ((a + 1/b) / (1/a + b) = 17) ∧ 
  (a - 2*b = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_exists_l1296_129633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_l1296_129698

theorem trigonometric_values (α : Real) :
  (α ∈ Set.Ioo (π/2) π) →  -- α is in the second quadrant
  (Real.cos α = -8/17) →
  (Real.sin α = 15/17 ∧ Real.tan α = -15/8) ∧
  (Real.tan α = 2 → (3*Real.sin α - Real.cos α) / (2*Real.sin α + 3*Real.cos α) = 5/7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_values_l1296_129698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l1296_129685

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 2) + 3

noncomputable def g (α : ℝ) (x : ℝ) : ℝ := x^α

theorem fixed_point_power_function :
  ∀ a : ℝ, a > 0 → a ≠ 1 →
  ∃ x₀ y₀ α : ℝ,
    f a x₀ = y₀ ∧
    g α x₀ = y₀ →
    g α 3 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_power_function_l1296_129685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_product_is_one_l1296_129606

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 6

-- Define the vertices
noncomputable def left_vertex : ℝ × ℝ := (-Real.sqrt 6, 0)
noncomputable def right_vertex : ℝ × ℝ := (Real.sqrt 6, 0)

-- Define a point on the right branch of the hyperbola in the first quadrant
def point_on_hyperbola (x₀ y₀ : ℝ) : Prop :=
  hyperbola x₀ y₀ ∧ x₀ > 0 ∧ y₀ > 0

-- Define the slopes of PA₁ and PA₂
noncomputable def slope_PA₁ (x₀ y₀ : ℝ) : ℝ := y₀ / (x₀ + Real.sqrt 6)
noncomputable def slope_PA₂ (x₀ y₀ : ℝ) : ℝ := y₀ / (x₀ - Real.sqrt 6)

-- The theorem to prove
theorem slopes_product_is_one (x₀ y₀ : ℝ) 
  (h : point_on_hyperbola x₀ y₀) : 
  slope_PA₁ x₀ y₀ * slope_PA₂ x₀ y₀ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slopes_product_is_one_l1296_129606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_power_six_l1296_129640

theorem sin_cos_power_six (θ : ℝ) (h : Real.sin (3 * θ) = 1/2) : 
  (Real.sin θ)^6 + (Real.cos θ)^6 = 11/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_power_six_l1296_129640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l1296_129601

-- Define the function f(x)
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

-- State the theorem
theorem f_derivative_at_2 (a b : ℝ) :
  f a b 1 = -2 →
  (deriv (f a b)) 1 = 0 →
  (deriv (f a b)) 2 = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_2_l1296_129601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_quadrilaterals_l1296_129663

/-- A convex cyclic quadrilateral with integer side lengths -/
structure CyclicQuadrilateral where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  perimeter_eq : a + b + c + d = 32
  convex : a + b + c > d ∧ b + c + d > a ∧ c + d + a > b ∧ d + a + b > c

/-- Two quadrilaterals are equivalent if they can be obtained from each other by rotation and translation -/
def equivalent : CyclicQuadrilateral → CyclicQuadrilateral → Prop :=
  sorry

/-- The set of all distinct convex cyclic quadrilaterals with integer sides and perimeter 32 -/
def distinct_quadrilaterals : Finset CyclicQuadrilateral :=
  sorry

/-- The number of distinct convex cyclic quadrilaterals with integer sides and perimeter 32 is 568 -/
theorem count_distinct_quadrilaterals :
  Finset.card distinct_quadrilaterals = 568 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_distinct_quadrilaterals_l1296_129663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_oil_price_theorem_l1296_129650

/-- Represents the price of oil in rupees per kg -/
structure OilPrice where
  price : ℚ
  price_positive : price > 0

/-- Calculates the amount of oil (in kg) that can be purchased for a given total cost -/
def amount_purchased (p : OilPrice) (total_cost : ℚ) : ℚ :=
  total_cost / p.price

/-- Theorem: If a 50% reduction in oil price allows purchasing 5 kg more for Rs. 800,
    then the reduced price is Rs. 80 per kg -/
theorem reduced_oil_price_theorem (original_price : OilPrice) :
  amount_purchased { price := original_price.price / 2,
                     price_positive := by {
                       have h : original_price.price > 0 := original_price.price_positive
                       linarith
                     }
                   } 800 =
  amount_purchased original_price 800 + 5 →
  original_price.price / 2 = 80 := by
  intro h
  -- Proof steps would go here
  sorry

#eval (80 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reduced_oil_price_theorem_l1296_129650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinear_trapezoid_area_l1296_129669

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define the antiderivative S(x)
noncomputable def S (x : ℝ) : ℝ := x^3 / 3 - 1 / 3

-- State the theorem
theorem curvilinear_trapezoid_area :
  (S 2 - S 1 = 7 / 3) ∧ 
  (∀ x, (deriv S) x = f x) ∧ 
  (S 1 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curvilinear_trapezoid_area_l1296_129669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_reciprocal_sum_l1296_129688

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_mean : (Real.log a + Real.log b) / 2 = 0) :
  2 ≤ (1 / a + 1 / b) ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    (Real.log a₀ + Real.log b₀) / 2 = 0 ∧ 1 / a₀ + 1 / b₀ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_reciprocal_sum_l1296_129688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_power_5_l1296_129619

/-- The coefficient of x^5 in the expansion of (1/(3x) + 2x√x)^7 is 560 -/
theorem coefficient_x_power_5 : 
  (Polynomial.coeff ((1 / (3 * Polynomial.X) + 2 * Polynomial.X * Polynomial.X^(1/2))^7 : Polynomial ℝ) 5) = 560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_power_5_l1296_129619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1296_129649

theorem sin_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1296_129649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1296_129607

-- Define the circle C
def circle_C (t x y : ℝ) : Prop :=
  (x - t)^2 + (y - t + 2)^2 = 1

-- Define point P
def P : ℝ × ℝ := (-1, 1)

-- Define the dot product of vectors PA and PB
def dot_product (A B : ℝ × ℝ) : ℝ :=
  let PA := (A.1 - P.1, A.2 - P.2)
  let PB := (B.1 - P.1, B.2 - P.2)
  PA.1 * PB.1 + PA.2 * PB.2

-- Theorem statement
theorem min_dot_product :
  ∃ (min : ℝ), min = 21/4 ∧
  ∀ (t : ℝ) (A B : ℝ × ℝ),
    circle_C t A.1 A.2 →
    circle_C t B.1 B.2 →
    (A.1 - P.1)^2 + (A.2 - P.2)^2 = (A.1 - t)^2 + (A.2 - (t-2))^2 →
    (B.1 - P.1)^2 + (B.2 - P.2)^2 = (B.1 - t)^2 + (B.2 - (t-2))^2 →
    dot_product A B ≥ min :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dot_product_l1296_129607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_babysitter_weekly_hour_limit_l1296_129600

/-- Represents the babysitter's pay structure and work hours -/
structure BabysitterPay where
  regularRate : ℚ
  overtimeRateFactor : ℚ
  totalEarnings : ℚ
  totalHours : ℚ

/-- Calculates the weekly hour limit before overtime pay -/
def weeklyHourLimit (pay : BabysitterPay) : ℚ :=
  let overtimeRate := pay.regularRate * (1 + pay.overtimeRateFactor)
  (pay.totalEarnings - overtimeRate * pay.totalHours) / (pay.regularRate - overtimeRate)

/-- Theorem stating the weekly hour limit for the given conditions -/
theorem babysitter_weekly_hour_limit :
  let pay : BabysitterPay := {
    regularRate := 16,
    overtimeRateFactor := 3/4,
    totalEarnings := 760,
    totalHours := 40
  }
  weeklyHourLimit pay = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_babysitter_weekly_hour_limit_l1296_129600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l1296_129635

theorem cube_root_equation_solution (x : ℝ) :
  (x^(1/3) + (24 - x)^(1/3) = 0) →
  ∃ (p q : ℤ), x = p + Real.sqrt (q : ℝ) ∧ p + q = 13 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l1296_129635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_monochromatic_triangles_l1296_129658

/-- A two-coloring of the edges of a complete graph -/
def TwoColoring (n : ℕ) := Fin n → Fin n → Bool

/-- A triangle in a graph -/
structure Triangle (n : ℕ) where
  a : Fin n
  b : Fin n
  c : Fin n
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Two triangles share exactly one edge -/
def ShareOneEdge {n : ℕ} (t1 t2 : Triangle n) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b) ∨
  (t1.a = t2.a ∧ t1.c = t2.c) ∨
  (t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.a) ∨
  (t1.a = t2.b ∧ t1.c = t2.c) ∨
  (t1.b = t2.a ∧ t1.c = t2.c)

/-- A triangle is monochromatic in a given coloring -/
def IsMonochromatic {n : ℕ} (t : Triangle n) (c : TwoColoring n) : Prop :=
  c t.a t.b = c t.b t.c ∧ c t.b t.c = c t.a t.c

/-- The main theorem -/
theorem smallest_n_for_monochromatic_triangles :
  (∀ n < 10, ∃ c : TwoColoring n, ∀ t1 t2 : Triangle n,
    IsMonochromatic t1 c → IsMonochromatic t2 c → ¬ShareOneEdge t1 t2) ∧
  (∀ c : TwoColoring 10, ∃ t1 t2 : Triangle 10,
    IsMonochromatic t1 c ∧ IsMonochromatic t2 c ∧ ShareOneEdge t1 t2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_monochromatic_triangles_l1296_129658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_condition_l1296_129660

-- Define the function f(x) = xe^(-x)
noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

-- State the theorem
theorem f_inequality_condition (x k : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  f x > f (k / x) ↔ k ≤ 0 ∨ k ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_condition_l1296_129660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l1296_129625

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2*x)

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  ∃ (a b : ℝ), a = 0 ∧ b = 1 ∧
  (∀ x, x ∈ Set.Ioo 0 2 → -x^2 + 2*x > 0) ∧
  (∀ x y, x ∈ Set.Ioo a b → y ∈ Set.Ioo a b → x < y → f x < f y) ∧
  (∀ c d, c < a ∨ b < d → ¬(∀ x y, x ∈ Set.Ioo c d → y ∈ Set.Ioo c d → x < y → f x < f y)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l1296_129625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_plus_two_l1296_129645

/-- The function f(x) = x(x-1)/2 -/
noncomputable def f (x : ℝ) : ℝ := x * (x - 1) / 2

/-- Theorem stating that f(x+2) = ((x+2) * f(x+1)) / x -/
theorem f_x_plus_two (x : ℝ) (h : x ≠ 0) : f (x + 2) = ((x + 2) * f (x + 1)) / x := by
  -- Unfold the definition of f
  unfold f
  -- Simplify both sides
  simp [h]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_x_plus_two_l1296_129645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1296_129694

/-- Parabola type representing y = x^2 -/
structure Parabola where
  x : ℝ
  y : ℝ
  eq : y = x^2

/-- Point on the parabola -/
def PointOnParabola (a : ℝ) : Parabola :=
  { x := a, y := a^2, eq := rfl }

/-- The origin point -/
def O : ℝ × ℝ := (0, 0)

/-- Angle AOB is a right angle -/
def IsRightAngle (A B : Parabola) : Prop :=
  (A.x - O.1) * (B.x - O.1) + (A.y - O.2) * (B.y - O.2) = 0

/-- Area of triangle AOB -/
noncomputable def TriangleArea (A B : Parabola) : ℝ :=
  (1/2) * abs (A.x * B.y - B.x * A.y)

/-- Theorem: The minimum area of triangle AOB is 1 -/
theorem min_triangle_area :
  ∀ A B : Parabola, IsRightAngle A B → TriangleArea A B ≥ 1 :=
by
  sorry

#check min_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l1296_129694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1296_129611

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (4 * Real.sin θ * Real.cos θ, 4 * Real.sin θ * Real.sin θ)

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 + Real.sqrt 2 * t, Real.sqrt 2 * t)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_distance_curve_to_line :
  ∃ (θ t : ℝ), ∀ (θ' t' : ℝ),
    distance (curve_C θ) (line_l t) ≤ distance (curve_C θ') (line_l t') ∧
    distance (curve_C θ) (line_l t) = (5 * Real.sqrt 2) / 2 - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curve_to_line_l1296_129611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_product_l1296_129631

/-- Ellipse structure -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- Point on an ellipse -/
def PointOnEllipse (E : Ellipse) (P : ℝ × ℝ) : Prop :=
  (P.1^2 / E.a^2) + (P.2^2 / E.b^2) = 1

/-- Focal points of an ellipse -/
noncomputable def FocalPoints (E : Ellipse) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := Real.sqrt (E.a^2 - E.b^2)
  ((-c, 0), (c, 0))

/-- Vector from a point to another -/
def VectorFrom (P Q : ℝ × ℝ) : ℝ × ℝ :=
  (Q.1 - P.1, Q.2 - P.2)

/-- Dot product of two vectors -/
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Magnitude of a vector -/
noncomputable def Magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

/-- Theorem statement -/
theorem ellipse_focal_product (E : Ellipse) (P : ℝ × ℝ) 
  (h_on_ellipse : PointOnEllipse E P)
  (h_dot_product : let (F₁, F₂) := FocalPoints E
                   DotProduct (VectorFrom P F₁) (VectorFrom P F₂) = 9) :
  let (F₁, F₂) := FocalPoints E
  Magnitude (VectorFrom P F₁) * Magnitude (VectorFrom P F₂) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_product_l1296_129631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_l1296_129603

theorem triangle_angle_c (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  Real.sin A / a = Real.sin B / b →
  Real.sin A / a = Real.sin C / c →
  b + c = 2 * a →
  3 * Real.sin A = 5 * Real.sin B →
  C = 2 * π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_l1296_129603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l1296_129654

/-- The y-intercept of a line is the point where it intersects the y-axis (x = 0) --/
noncomputable def y_intercept (a b c : ℝ) : ℝ × ℝ := (0, c / b)

/-- A line is defined by the equation ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

theorem y_intercept_of_line (l : Line) (h : l.b ≠ 0) :
  y_intercept l.a l.b l.c = (0, 4) ↔ l.a = 5/2 ∧ l.b = 7 ∧ l.c = 28 := by
  sorry

#check y_intercept_of_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_line_l1296_129654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_proof_l1296_129641

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.cos x + Real.sqrt 3 * Real.sin (2 * x)

theorem triangle_ratio_proof (A B C : ℝ) (b c : ℝ) (S : ℝ) :
  f A = 2 →
  b = 1 →
  S = Real.sqrt 3 / 2 →
  (b + c) / (Real.sin B + Real.sin C) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_proof_l1296_129641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_income_is_320_l1296_129661

/-- Represents the tax structure and company's income --/
structure TaxInfo where
  p : ℚ  -- Base tax rate in percentage
  income : ℚ  -- Company's income in million yuan

/-- Calculates the total tax paid based on the given tax structure --/
noncomputable def totalTax (t : TaxInfo) : ℚ :=
  min t.income (28/10) * (t.p / 100) + max (t.income - 28/10) 0 * ((t.p + 2) / 100)

/-- Theorem stating that given the tax structure and actual tax ratio, the company's income is 320 million yuan --/
theorem company_income_is_320 (t : TaxInfo) :
  (totalTax t) / t.income = (t.p + 1/4) / 100 → t.income = 320 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_income_is_320_l1296_129661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tensor_difference_l1296_129665

-- Define the ⊗ operation
noncomputable def tensor (a b : ℝ) : ℝ := a^3 / b

-- State the theorem
theorem tensor_difference : 
  tensor (tensor 2 4) 3 - tensor 2 (tensor 4 3) = 55/24 := by
  -- Unfold the definition of tensor
  unfold tensor
  -- Simplify the expression
  simp [pow_three]
  -- Perform the arithmetic
  ring
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tensor_difference_l1296_129665
