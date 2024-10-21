import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_arg_theorem_l1242_124232

/-- The complex number with the smallest positive argument satisfying |z - 5i| ≤ 4 -/
noncomputable def smallest_positive_arg_z : ℂ := 2.4 + 1.8 * Complex.I

/-- The condition that z satisfies |z - 5i| ≤ 4 -/
def satisfies_condition (z : ℂ) : Prop := Complex.abs (z - 5 * Complex.I) ≤ 4

theorem smallest_positive_arg_theorem :
  satisfies_condition smallest_positive_arg_z ∧
  ∀ z : ℂ, satisfies_condition z → Complex.arg z ≥ 0 →
    Complex.arg smallest_positive_arg_z ≤ Complex.arg z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_arg_theorem_l1242_124232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ST_l1242_124208

/-- A circle passing through (1,0) and tangent to x=-1 -/
structure CircleM where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : center.1 + radius = 1
  tangent_to : center.1 - radius = -1

/-- The trajectory of the center M -/
def trajectory (c : CircleM) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The line x+y+4=0 -/
def line_T : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 + 4 = 0}

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The minimum distance between S and T is 3√2/2 -/
theorem min_distance_ST (c : CircleM) :
  ∃ (S : ℝ × ℝ) (T : ℝ × ℝ), S ∈ trajectory c ∧ T ∈ line_T ∧
  ∀ (S' : ℝ × ℝ) (T' : ℝ × ℝ), S' ∈ trajectory c → T' ∈ line_T →
  distance S T ≤ distance S' T' ∧ distance S T = 3 * Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ST_l1242_124208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_outstanding_boys_l1242_124239

/-- Represents a person with height and weight -/
structure Person where
  height : ℝ
  weight : ℝ

/-- A group of 100 people -/
def PeopleGroup := Fin 100 → Person

/-- Defines when one person is not inferior to another -/
def notInferior (a b : Person) : Prop :=
  a.height > b.height ∨ a.weight > b.weight

/-- Defines an outstanding person in the group -/
def isOutstanding (g : PeopleGroup) (i : Fin 100) : Prop :=
  ∀ j : Fin 100, j ≠ i → notInferior (g i) (g j)

/-- The theorem to be proved -/
theorem max_outstanding_boys (g : PeopleGroup) 
  (h₁ : ∀ i j : Fin 100, i ≠ j → g i ≠ g j)
  (h₂ : ∀ i j : Fin 100, i ≠ j → (g i).height ≠ (g j).height)
  (h₃ : ∀ i j : Fin 100, i ≠ j → (g i).weight ≠ (g j).weight) :
  ∃ f : Fin 100 ↪ Fin 100, ∀ i : Fin 100, isOutstanding g (f i) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_outstanding_boys_l1242_124239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l1242_124283

-- Define the bicycle parameters
noncomputable def front_wheel_radius : ℝ := 3
noncomputable def back_wheel_radius : ℝ := 5 / 12  -- Convert 5 inches to feet
noncomputable def front_wheel_revolutions : ℝ := 150

-- Define the theorem
theorem bicycle_wheel_revolutions :
  let front_circumference := 2 * Real.pi * front_wheel_radius
  let back_circumference := 2 * Real.pi * back_wheel_radius
  let total_distance := front_circumference * front_wheel_revolutions
  let back_wheel_revolutions := total_distance / back_circumference
  back_wheel_revolutions = 1080 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l1242_124283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aspect_translation_l1242_124233

theorem aspect_translation : True := by
  trivial

#check aspect_translation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_aspect_translation_l1242_124233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_football_game_cost_l1242_124284

/-- Represents the cost of football games attended by Sam over two years -/
def football_game_cost (this_year_games : ℕ) (last_year_games : ℕ) 
  (this_year_price : ℚ) (last_year_low_price : ℚ) (last_year_high_price : ℚ) : ℚ :=
  let this_year_cost := this_year_games * this_year_price
  let last_year_low_games := (last_year_games / 3 : ℚ).floor
  let last_year_high_games := (last_year_games / 4 : ℚ).floor
  let last_year_cost := last_year_low_games * last_year_low_price + 
                        last_year_high_games * last_year_high_price
  this_year_cost + last_year_cost

/-- Theorem stating the total cost of football games for Sam over two years -/
theorem total_football_game_cost : 
  football_game_cost 14 29 45 40 65 = 1445 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_football_game_cost_l1242_124284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_stamp_price_l1242_124218

theorem yellow_stamp_price
  (red_stamps : ℕ)
  (blue_stamps : ℕ)
  (yellow_stamps : ℕ)
  (red_price : ℚ)
  (blue_price : ℚ)
  (total_goal : ℚ)
  (h1 : red_stamps = 20)
  (h2 : blue_stamps = 80)
  (h3 : yellow_stamps = 7)
  (h4 : red_price = 11/10)
  (h5 : blue_price = 8/10)
  (h6 : total_goal = 100)
  : (total_goal - (red_stamps * red_price + blue_stamps * blue_price)) / yellow_stamps = 2 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yellow_stamp_price_l1242_124218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_playground_width_l1242_124238

/-- Proves that a playground with length 16 meters has width 12 meters if its area equals that of a garden with width 24 meters and perimeter 64 meters. -/
theorem playground_width (garden_width garden_perimeter playground_length playground_width : ℝ) 
  (hw : garden_width = 24)
  (hp : garden_perimeter = 64)
  (hl : playground_length = 16)
  (area_eq : garden_width * (garden_perimeter / 2 - garden_width) = playground_length * playground_width) :
  playground_width = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_playground_width_l1242_124238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_equals_one_l1242_124281

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the line x + y - m = 0 -/
def slope1 : ℝ := -1

/-- The slope of the line x + (3-2m)y = 0 -/
noncomputable def slope2 (m : ℝ) : ℝ := -1 / (3 - 2*m)

theorem perpendicular_lines_m_equals_one (m : ℝ) : 
  perpendicular slope1 (slope2 m) → m = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_m_equals_one_l1242_124281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_exterior_probability_is_one_l1242_124243

/-- A cube with side length 2 and blue faces -/
structure BlueCube where
  side_length : ℝ
  is_blue : Bool

/-- A unit cube derived from subdividing a larger cube -/
structure UnitCube where
  parent : BlueCube
  position : Fin 8

/-- The probability of the new cube being completely blue after rearrangement -/
noncomputable def blue_exterior_probability (c : BlueCube) : ℝ :=
  if c.side_length = 2 ∧ c.is_blue then 1 else 0

/-- Theorem stating that the probability of a blue exterior after rearrangement is 1 -/
theorem blue_exterior_probability_is_one (c : BlueCube) 
  (h1 : c.side_length = 2) 
  (h2 : c.is_blue = true) :
  blue_exterior_probability c = 1 := by
  unfold blue_exterior_probability
  simp [h1, h2]

#check blue_exterior_probability_is_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_exterior_probability_is_one_l1242_124243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1242_124227

/-- The hyperbola equation: x^2 - y^2/3 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- The right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (2, 0)

/-- The line passing through the right focus and perpendicular to x-axis -/
def perpendicular_line (x : ℝ) : Prop := x = right_focus.1

/-- The asymptotes of the hyperbola -/
noncomputable def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

/-- The intersection points A and B -/
noncomputable def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ asymptotes x y ∧ perpendicular_line x}

/-- The length of line segment AB -/
noncomputable def length_AB : ℝ := 
  let A := (2, 2 * Real.sqrt 3)
  let B := (2, -2 * Real.sqrt 3)
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem length_of_AB : length_AB = 4 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1242_124227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_most_noteworthy_l1242_124231

-- Define a type for outing spots
structure OutingSpot where
  name : String
deriving BEq, Repr

-- Define a type for survey data
def SurveyData : Type := List OutingSpot

-- Define a function to calculate the mode of a list
def mode (data : SurveyData) : Option OutingSpot :=
  sorry

-- Define a function to calculate the mean of a list (for comparison)
noncomputable def mean (data : SurveyData) : ℝ :=
  sorry

-- Define a function to calculate the median of a list (for comparison)
def median (data : SurveyData) : Option OutingSpot :=
  sorry

-- Define a function to calculate the weighted mean of a list (for comparison)
noncomputable def weightedMean (data : SurveyData) (weights : List ℝ) : ℝ :=
  sorry

-- Theorem stating that the mode is the most noteworthy measure for outing spot preferences
theorem mode_most_noteworthy (data : SurveyData) :
  ∃ (spot : OutingSpot), (mode data = some spot) ∧ 
  (∀ (other : OutingSpot), other ≠ spot → (data.count other) ≤ (data.count spot)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mode_most_noteworthy_l1242_124231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l1242_124272

open Real

-- Define the propositions
def proposition1 : Prop := ∀ x : ℝ, (sin x)^4 - (cos x)^4 = -cos (2*x)

def proposition2 : Prop := {α : ℝ | ∃ k : ℤ, α = k * π / 2} = {α : ℝ | ∃ k : ℤ, α = k * π + π / 2}

def proposition3 : Prop := ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ sin x = x ∧ sin y = y ∧ sin z = z

def proposition4 : Prop := ∀ x y : ℝ, x < y → tan x < tan y

def proposition5 : Prop := ∀ x : ℝ, sin (x - π/2) = sin (π/2 - x)

-- Theorem statement
theorem correct_propositions :
  proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 ∧ proposition5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l1242_124272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_underlined_sum_positive_l1242_124266

def is_underlined (seq : List ℤ) (i : ℕ) : Bool :=
  seq[i]! > 0 ||
  (i + 1 < seq.length && seq[i]! + seq[i+1]! > 0) ||
  (i + 2 < seq.length && seq[i]! + seq[i+1]! + seq[i+2]! > 0)

def sum_underlined (seq : List ℤ) : ℤ :=
  (List.range seq.length).foldl (fun sum i => 
    if is_underlined seq i then sum + seq[i]! else sum) 0

theorem underlined_sum_positive (seq : List ℤ) (h : seq.length = 100) :
  sum_underlined seq > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_underlined_sum_positive_l1242_124266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_GKL_l1242_124297

-- Define the points
noncomputable def J : ℝ × ℝ := (0, 8)
noncomputable def K : ℝ × ℝ := (0, 0)
noncomputable def L : ℝ × ℝ := (10, 0)

-- Define midpoints
noncomputable def G : ℝ × ℝ := ((J.1 + K.1) / 2, (J.2 + K.2) / 2)
noncomputable def H : ℝ × ℝ := ((K.1 + L.1) / 2, (K.2 + L.2) / 2)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Theorem statement
theorem area_of_triangle_GKL : triangleArea G K L = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_GKL_l1242_124297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_eight_l1242_124256

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

-- Define the theorem
theorem a_less_than_eight (a : ℝ) : 
  (∃ x : ℕ+, floor ((x : ℝ) + a) / 3 = 2) → a < 8 := by
  intro h
  -- The proof steps would go here
  sorry

#check a_less_than_eight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_eight_l1242_124256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_l1242_124224

theorem cos_half_angle (α : ℝ) (h1 : α ∈ Set.Ioo π (2 * π)) (h2 : Real.cos α - 3 * Real.sin α = 1) :
  Real.cos (α / 2) = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_half_angle_l1242_124224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_trophy_increase_l1242_124255

/-- The increase in Michael's trophies after three years -/
def trophy_increase : ℕ → ℕ := sorry

theorem michael_trophy_increase :
  let current_trophies : ℕ := 30
  let total_after_three_years : ℕ := 430
  let jack_trophies : ℕ → ℕ := λ current ↦ 10 * current
  trophy_increase current_trophies = 100 ∧
  current_trophies + trophy_increase current_trophies + jack_trophies current_trophies = total_after_three_years :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_trophy_increase_l1242_124255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l1242_124214

noncomputable def h (x : ℝ) : ℝ := 3 / (1 + 9 * x^2)

theorem range_of_h :
  Set.range h = Set.Ioo 0 3 ∪ {3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_h_l1242_124214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_330_degrees_l1242_124212

theorem sin_330_degrees : 
  Real.sin (330 * Real.pi / 180) = -(1 / 2) := by
  let angle : Real := 330 * Real.pi / 180
  have h1 : angle > 3 * Real.pi / 2 ∧ angle < 2 * Real.pi := by sorry
  have h2 : ∀ θ : Real, θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi → Real.sin θ < 0 := by sorry
  have h3 : angle = 2 * Real.pi - (30 * Real.pi / 180) := by sorry
  have h4 : Real.sin (30 * Real.pi / 180) = 1 / 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_330_degrees_l1242_124212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_property_l1242_124273

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line l
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the foci of the ellipse
noncomputable def F1 : ℝ × ℝ := (0, -Real.sqrt 3)
noncomputable def F2 : ℝ × ℝ := (0, Real.sqrt 3)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem ellipse_intersection_property :
  ∀ (k : ℝ) (A B : ℝ × ℝ),
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
  line k A.1 A.2 ∧ line k B.1 B.2 ∧
  (distance A F1 + distance A F2 = 4) ∧
  (distance B F1 + distance B F2 = 4) →
  (distance O A + distance O B = distance A B ↔ k = 1/2 ∨ k = -1/2) ∧
  (k = 1/2 ∨ k = -1/2 → distance A B = 4 * Real.sqrt 65 / 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_property_l1242_124273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_decomposition_l1242_124234

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

def is_valid_decomposition (n : ℕ) (indices : List ℕ) : Prop :=
  (indices.map fib).sum = n ∧
  indices.Sorted (· < ·) ∧
  ∀ i j, indices.indexOf i + 1 = indices.indexOf j → j ≥ i + 2

theorem fibonacci_decomposition (n : ℕ) (h : n ≥ 1) :
  ∃ indices : List ℕ, is_valid_decomposition n indices := by
  sorry

#check fibonacci_decomposition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_decomposition_l1242_124234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_prime_factors_theorem_l1242_124274

def max_distinct_prime_factors (c d : ℕ) : Prop :=
  (∃ (p : ℕ), p = 11 ∧ (Nat.gcd c d).factors.length = p) ∧
  (∃ (q : ℕ), q = 31 ∧ (Nat.lcm c d).factors.length = q) ∧
  (c.factors.length > d.factors.length) →
  d.factors.length ≤ 21

theorem max_distinct_prime_factors_theorem (c d : ℕ) :
  max_distinct_prime_factors c d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distinct_prime_factors_theorem_l1242_124274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_covering_theorem_l1242_124264

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  center : Point
  sideLength : ℝ

/-- Checks if a point is inside an equilateral triangle -/
def isPointInTriangle (p : Point) (t : EquilateralTriangle) : Prop := sorry

/-- Checks if a triangle covers a set of points -/
def triangleCoversPoints (t : EquilateralTriangle) (points : Finset Point) : Prop := sorry

/-- Checks if two triangles have parallel sides -/
def haveParallelSides (t1 t2 : EquilateralTriangle) : Prop := sorry

/-- Calculates the area of an equilateral triangle -/
noncomputable def triangleArea (t : EquilateralTriangle) : ℝ := sorry

theorem equilateral_triangle_covering_theorem 
  (originalTriangle : EquilateralTriangle)
  (points : Finset Point)
  (h1 : triangleArea originalTriangle = 1)
  (h2 : points.card = 5)
  (h3 : ∀ p ∈ points, isPointInTriangle p originalTriangle) :
  ∃ (t1 t2 t3 : EquilateralTriangle),
    (∀ p ∈ points, triangleCoversPoints t1 {p} ∨ triangleCoversPoints t2 {p} ∨ triangleCoversPoints t3 {p}) ∧
    (haveParallelSides t1 originalTriangle ∧ haveParallelSides t2 originalTriangle ∧ haveParallelSides t3 originalTriangle) ∧
    (triangleArea t1 + triangleArea t2 + triangleArea t3 ≤ 0.64) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_covering_theorem_l1242_124264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_33_l1242_124261

theorem x_plus_y_equals_33 (x y : ℝ) (h1 : Real.sqrt (y - 5) = 5) (h2 : (2 : ℝ)^x = 8) : x + y = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_equals_33_l1242_124261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_cast_l1242_124211

theorem total_votes_cast (candidate_percentage : ℚ) (vote_difference : ℕ) : 
  candidate_percentage = 34/100 →
  vote_difference = 640 →
  ∃ (total_votes : ℕ), 
    (candidate_percentage * total_votes) + vote_difference = 
    ((1 - candidate_percentage) * total_votes) ∧
    total_votes = 2000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_votes_cast_l1242_124211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_integer_digit_sum_l1242_124219

/-- Given six consecutive integers where the sum of five of them is 2012,
    the sum of the digits of the remaining integer is 7. -/
theorem erased_integer_digit_sum :
  ∀ (n : ℤ),
  (∃ (a : ℕ), a ≤ 5 ∧
    (6 * n + 15) - (n + a) = 2012) →
  (∃ (x : ℤ), x ∈ Set.range (fun i => n + i) ∧ i ∈ Finset.range 6 ∧
    (x.natAbs.digits 10).sum = 7) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_erased_integer_digit_sum_l1242_124219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_saved_theorem_l1242_124242

/-- Given a bus rental cost and the number of students, calculates the amount saved per student when additional students join. -/
noncomputable def amountSavedPerStudent (rentalCost : ℝ) (originalStudents : ℝ) : ℝ :=
  rentalCost / originalStudents - rentalCost / (originalStudents + 3)

/-- Theorem: For a bus rental of $800 and x original students, 
    the amount saved per student when 3 more join is 2400 / (x * (x + 3)) -/
theorem amount_saved_theorem (x : ℝ) (h : x > 0) : 
  amountSavedPerStudent 800 x = 2400 / (x * (x + 3)) := by
  sorry

#check amount_saved_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_saved_theorem_l1242_124242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reflections_l1242_124237

/-- The angle between lines AD and CD in degrees -/
def angle : ℚ := 7

/-- The maximum angle allowed for reflection in degrees -/
def max_angle : ℚ := 90

/-- The number of reflections must be a natural number -/
def n : ℕ := (max_angle / angle).floor.toNat

theorem max_reflections :
  n = (max_angle / angle).floor.toNat ∧ 
  (n : ℚ) * angle ≤ max_angle ∧ 
  ((n + 1) : ℚ) * angle > max_angle :=
sorry

#eval n  -- This will evaluate to 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_reflections_l1242_124237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_satisfies_conditions_l1242_124269

-- Define the coloring function
def color (x y : ℤ) : Fin 5 :=
  (x + 2*y) % 5

-- Define a figure of type 1
def figureType1 (x y : ℤ) : Set (ℤ × ℤ) :=
  {(x, y), (x+1, y), (x+2, y), (x+1, y+1), (x+1, y-1)}

-- Define a figure of type 2 (any set of cells different from type 1)
def figureType2 : Set (Set (ℤ × ℤ)) :=
  {s | s ⊆ Set.univ ∧ ∃ x y, s ≠ figureType1 x y}

-- Theorem statement
theorem coloring_satisfies_conditions :
  (∀ x y : ℤ, ∃ c0 c1 c2 c3 c4 : (ℤ × ℤ),
    c0 ∈ figureType1 x y ∧ color c0.1 c0.2 = 0 ∧
    c1 ∈ figureType1 x y ∧ color c1.1 c1.2 = 1 ∧
    c2 ∈ figureType1 x y ∧ color c2.1 c2.2 = 2 ∧
    c3 ∈ figureType1 x y ∧ color c3.1 c3.2 = 3 ∧
    c4 ∈ figureType1 x y ∧ color c4.1 c4.2 = 4) ∧
  (∃ s ∈ figureType2, ¬(∀ c : Fin 5, ∃ xy ∈ s, color xy.1 xy.2 = c)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coloring_satisfies_conditions_l1242_124269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_seven_l1242_124282

/-- The area of a triangle given its vertices using the shoelace formula -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x1*y2 + x2*y3 + x3*y1) - (y1*x2 + y2*x3 + y3*x1))

/-- Theorem: The area of the triangle with vertices (0,2), (3,0), and (1,6) is 7 square units -/
theorem triangle_area_is_seven :
  triangleArea 0 2 3 0 1 6 = 7 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_seven_l1242_124282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coefficient_value_l1242_124253

theorem min_coefficient_value (a b box : ℤ) : 
  (a ≠ b ∧ a ≠ box ∧ b ≠ box) → 
  ((∀ x : ℤ, (a*x + b) * (b*x + a) = 35*x^2 + box*x + 35) → 
   (∀ a' b' box' : ℤ, 
     (a' ≠ b' ∧ a' ≠ box' ∧ b' ≠ box') → 
     (∀ x : ℤ, (a'*x + b') * (b'*x + a') = 35*x^2 + box'*x + 35) → 
     box ≤ box')) →
  box = 74 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coefficient_value_l1242_124253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l1242_124278

theorem cube_root_simplification : (2^9 * 3^6 * 5^3)^(1/3) = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l1242_124278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_horizontal_length_l1242_124249

/-- Represents a television screen -/
structure TVScreen where
  aspect_ratio : ℚ
  diagonal : ℝ

/-- Calculates the horizontal length of a TV screen -/
noncomputable def horizontal_length (tv : TVScreen) : ℝ :=
  (4 / 5) * tv.diagonal

/-- Theorem: The horizontal length of a 27-inch TV with 4:3 aspect ratio is 21.6 inches -/
theorem tv_horizontal_length :
  let tv : TVScreen := { aspect_ratio := 4 / 3, diagonal := 27 }
  horizontal_length tv = 21.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_horizontal_length_l1242_124249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_parabola_vertex_count_l1242_124260

/-- The number of values of a for which the line y = x + 2a passes through
    the vertex of the parabola y = x^2 + 2a^2 -/
theorem line_through_parabola_vertex_count : 
  ∃! (S : Set ℝ), (∀ a ∈ S, ∃ x y : ℝ, 
    y = x + 2*a ∧ 
    y = x^2 + 2*a^2 ∧ 
    ∀ x' y' : ℝ, y' = x'^2 + 2*a^2 → y ≤ y') ∧ 
  Finite S ∧ Nat.card S = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_parabola_vertex_count_l1242_124260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_relation_l1242_124215

/-- Represents the property of being a convex pentagon with given areas -/
structure ConvexPentagon (S a b c d e : ℝ) : Prop where
  positive : S > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0
  sum : S = a + b + c + d + e

/-- Given a convex pentagon ABCDE with area S, and triangles ABC, BCD, CDE, DEA, EAB
    with areas a, b, c, d, e respectively, prove that:
    S^2 - S(a + b + c + d + e) + ab + bc + cd + de + ea = 0 -/
theorem pentagon_area_relation (S a b c d e : ℝ) 
    (h : ConvexPentagon S a b c d e) : 
    S^2 - S*(a + b + c + d + e) + a*b + b*c + c*d + d*e + e*a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_relation_l1242_124215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_product_division_rounded_l1242_124294

theorem cube_product_division_rounded (c : ℤ) : c = 550 ↔ c = round ((8^3 * 9^3 : ℚ) / 679) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_product_division_rounded_l1242_124294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l1242_124267

/-- The function f(x) = tan(2x) + cot(2x) -/
noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x) + (1 / Real.tan (2 * x))

/-- The period of f(x) is π/2 -/
theorem f_period : ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧ p = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_l1242_124267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_undefined_for_given_condition_l1242_124210

theorem tan_undefined_for_given_condition (α : ℝ) 
  (h1 : Real.sin α + 2 * (Real.sin (α / 2))^2 = 2) 
  (h2 : 0 < α) 
  (h3 : α < Real.pi) : 
  ¬∃(x : ℝ), Real.tan α = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_undefined_for_given_condition_l1242_124210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_plane_l1242_124276

/-- The distance from a point to a plane -/
noncomputable def distance_point_to_plane (x y z a b c d : ℝ) : ℝ :=
  |a * x + b * y + c * z + d| / Real.sqrt (a^2 + b^2 + c^2)

/-- Theorem: The distance from point A(2,3,-4) to the plane 2x + 6y - 3z + 16 = 0 is 50/7 -/
theorem distance_point_to_specific_plane :
  distance_point_to_plane 2 3 (-4) 2 6 (-3) 16 = 50 / 7 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_specific_plane_l1242_124276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_30_degrees_area_of_triangle_l1242_124289

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi

-- Define the properties of the triangle
def TriangleProperties (t : Triangle) : Prop :=
  t.a = 2 * t.b * Real.sin t.A ∧
  t.a = 3 * Real.sqrt 3 ∧
  t.c = 5

-- Theorem 1: Prove that angle B is 30°
theorem angle_B_is_30_degrees (t : Triangle) (h : TriangleProperties t) :
  t.B = Real.pi / 6 := by
  sorry

-- Theorem 2: Prove the area of the triangle
theorem area_of_triangle (t : Triangle) (h : TriangleProperties t) :
  (1/2) * t.a * t.c * Real.sin t.B = 15 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_30_degrees_area_of_triangle_l1242_124289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_hexagon_equality_l1242_124228

-- Define the structure for a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the structure for a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define the hexagon vertices
variable (A B C D E F : Point)

-- Define the circles
variable (O₁ O₂ O₃ : Circle)

-- Define the function to calculate distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Helper definitions (these would need to be properly defined in a real implementation)
def convex_hexagon : (Point → Point → Point → Point → Point → Point → Prop) := sorry
def triangle : (Point → Point → Point → Prop) := sorry
def non_intersecting_circles : (Circle → Circle → Circle → Prop) := sorry
def tangent_to_circles : (Point → Point → Point → Point → Point → Point → Circle → Circle → Circle → Prop) := sorry

-- State the theorem
theorem tangent_hexagon_equality 
  (h_convex : convex_hexagon A B C D E F)
  (h_equal_radius : O₁.radius = O₂.radius ∧ O₂.radius = O₃.radius)
  (h_centers_triangle : triangle O₁.center O₂.center O₃.center)
  (h_non_intersecting : non_intersecting_circles O₁ O₂ O₃)
  (h_tangent : tangent_to_circles A B C D E F O₁ O₂ O₃) :
  distance A B + distance C D + distance E F = 
  distance B C + distance D E + distance F A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_hexagon_equality_l1242_124228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactlyOnePointOnCircleAtDistanceSqrt2FromLine_l1242_124287

-- Define the circle
def circleEq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y = 0

-- Define the line
def lineEq (x y : ℝ) : Prop := x + y + 1 = 0

-- Define the distance function from a point to the line
noncomputable def distanceToLine (x y : ℝ) : ℝ :=
  |x + y + 1| / Real.sqrt 2

-- Theorem statement
theorem exactlyOnePointOnCircleAtDistanceSqrt2FromLine :
  ∃! p : ℝ × ℝ, circleEq p.1 p.2 ∧ distanceToLine p.1 p.2 = Real.sqrt 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactlyOnePointOnCircleAtDistanceSqrt2FromLine_l1242_124287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1242_124225

/-- Given that a = 4^0.8, b = 8^0.46, and c = (1/2)^(-1.2), prove that c < b < a -/
theorem relationship_abc (a b c : ℝ) 
  (ha : a = (4 : ℝ)^(0.8 : ℝ)) 
  (hb : b = (8 : ℝ)^(0.46 : ℝ)) 
  (hc : c = ((1/2) : ℝ)^(-(1.2 : ℝ))) :
  c < b ∧ b < a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1242_124225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_is_70_sum_70_exists_l1242_124248

/-- A sequence of five consecutive even numbers -/
def ConsecutiveEvenSequence : Type := Fin 5 → ℕ

/-- The property that a sequence contains consecutive even numbers -/
def IsConsecutiveEven (seq : ConsecutiveEvenSequence) : Prop :=
  ∀ i : Fin 4, seq (i.succ) = seq i + 2

/-- The property that a sequence contains 10 and 12 -/
def Contains10And12 (seq : ConsecutiveEvenSequence) : Prop :=
  (∃ i : Fin 5, seq i = 10) ∧ (∃ i : Fin 5, seq i = 12)

/-- The sum of a sequence -/
def SequenceSum (seq : ConsecutiveEvenSequence) : ℕ :=
  (Finset.univ : Finset (Fin 5)).sum seq

/-- The theorem stating that the largest possible sum is 70 -/
theorem largest_sum_is_70 :
  ∀ seq : ConsecutiveEvenSequence,
    IsConsecutiveEven seq →
    Contains10And12 seq →
    SequenceSum seq ≤ 70 :=
by
  sorry

/-- The theorem stating that there exists a sequence with sum 70 -/
theorem sum_70_exists :
  ∃ seq : ConsecutiveEvenSequence,
    IsConsecutiveEven seq ∧
    Contains10And12 seq ∧
    SequenceSum seq = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_sum_is_70_sum_70_exists_l1242_124248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_income_A_l1242_124271

/-- Given the monthly incomes of A, B, and C, calculate A's annual income -/
theorem annual_income_A (income_C : ℕ) (h1 : income_C = 13000) : ℕ := by
  let income_B := income_C + income_C * 12 / 100
  let income_A := income_B * 5 / 2
  have h2 : income_A * 12 = 436800 := by sorry
  exact 436800


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annual_income_A_l1242_124271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_number_product_l1242_124206

def S : Finset Nat := {1, 2, 3, 4, 5, 6, 7}

theorem five_number_product (A : Finset Nat) : 
  A ⊆ S → A.card = 5 →
  (∃ B C : Finset Nat, B ⊆ S ∧ C ⊆ S ∧ B.card = 5 ∧ C.card = 5 ∧
   (A.prod id = B.prod id ∧ A.prod id = C.prod id) ∧
   (A.sum id % 2 ≠ B.sum id % 2 ∨ A.sum id % 2 ≠ C.sum id % 2)) →
  A.prod id = 420 := by
  sorry

#check five_number_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_number_product_l1242_124206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1242_124290

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x else x^3 - 1/x + 1

theorem solution_set (x : ℝ) : f (6 - x^2) > f x ↔ -3 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1242_124290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_l1242_124200

/-- Hyperbola C with foci F₁(-2,0) and F₂(2,0) passing through P(7,12) -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/3 = 1

/-- Line l with slope 1 -/
def line_l (x y t : ℝ) : Prop := y = x + t

/-- A and B are intersection points of C and l -/
def intersection_points (xA yA xB yB t : ℝ) : Prop :=
  hyperbola_C xA yA ∧ hyperbola_C xB yB ∧
  line_l xA yA t ∧ line_l xB yB t

/-- OA ⊥ OB (O is the origin) -/
def perpendicular (xA yA xB yB : ℝ) : Prop :=
  xA * xB + yA * yB = 0

theorem hyperbola_line_intersection :
  ∀ t : ℝ, (∃ xA yA xB yB : ℝ,
    intersection_points xA yA xB yB t ∧
    perpendicular xA yA xB yB) →
    t = Real.sqrt 3 ∨ t = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_line_intersection_l1242_124200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymmetric_second_derivative_implies_squared_inequality_l1242_124293

/-- A function with the property that its second derivative at x is less than its second derivative at -x for positive x -/
def StrictlyAsymmetricSecondDerivative (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, (deriv^[2] f) x < (deriv^[2] f) (-x)

theorem asymmetric_second_derivative_implies_squared_inequality
  (f : ℝ → ℝ)
  (h_asymmetric : StrictlyAsymmetricSecondDerivative f)
  (a b : ℝ)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0)
  (h_inequality : f a - f b > f (-b) - f (-a)) :
  a^2 < b^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymmetric_second_derivative_implies_squared_inequality_l1242_124293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1242_124226

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (sequence_a n + 1) / (12 * sequence_a n)

theorem sequence_a_properties :
  (∀ n : ℕ, sequence_a n > 0) ∧
  (∀ n : ℕ, n > 0 → sequence_a (2 * n + 1) < sequence_a (2 * n - 1)) ∧
  (∀ n : ℕ, 1 / 6 ≤ sequence_a n ∧ sequence_a n ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1242_124226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_equivalence_l1242_124265

/-- A curve in 2D space parameterized by θ -/
structure ParametricCurve where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The specific curve given in the problem -/
noncomputable def givenCurve : ParametricCurve where
  x := λ θ => 1 + 2 * Real.cos θ
  y := λ θ => 2 + 3 * Real.sin θ

/-- The Cartesian equation of the curve -/
def cartesianEquation (x y : ℝ) : Prop :=
  (x - 1)^2 / 4 + (y - 2)^2 / 9 = 1

/-- Theorem stating that the Cartesian equation holds for all points on the curve -/
theorem curve_equation_equivalence :
  ∀ θ, cartesianEquation (givenCurve.x θ) (givenCurve.y θ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_equivalence_l1242_124265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l1242_124259

open Real Set

-- Define the function
noncomputable def f (x : ℝ) : ℝ := sin (π / 6 - x)

-- State the theorem
theorem monotonic_decreasing_interval_of_f :
  ∃ (a b : ℝ), a = 0 ∧ b = 2 * π / 3 ∧
  (∀ x ∈ Icc a b, x ∈ Icc 0 (3 * π / 2) →
    StrictMonoOn (fun x => -f x) (Icc a b)) ∧
  (∀ c d, c < a ∨ b < d →
    ¬StrictMonoOn (fun x => -f x) (Icc c d ∩ Icc 0 (3 * π / 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_of_f_l1242_124259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_vote_count_l1242_124209

theorem election_vote_count (candidate1_percent : ℚ) (candidate2_votes : ℕ) : 
  candidate1_percent = 4/5 → 
  candidate2_votes = 240 → 
  ∃ (total_votes : ℕ), 
    (candidate1_percent * total_votes = (total_votes - candidate2_votes)) ∧ 
    ((1 - candidate1_percent) * total_votes = candidate2_votes) ∧
    total_votes = 1200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_vote_count_l1242_124209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l1242_124240

noncomputable def a : ℝ × ℝ := (2, 4)
noncomputable def b : ℝ × ℝ := (3, -4)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

noncomputable def projection (v w : ℝ × ℝ) : ℝ := (dot_product v w) / (magnitude w)

theorem projection_a_onto_b :
  projection a b = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_onto_b_l1242_124240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_uniqueness_l1242_124251

/-- The circle C with equation x² + y² = 5 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 5}

/-- The point P(-1, 2) -/
def P : ℝ × ℝ := (-1, 2)

/-- The line L with equation x - 2y + 5 = 0 -/
def L : Set (ℝ × ℝ) := {p | p.1 - 2*p.2 + 5 = 0}

/-- L is tangent to C if it intersects C at exactly one point -/
def is_tangent (L : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ L ∩ C

theorem tangent_line_uniqueness :
  P ∈ L ∧ is_tangent L C ∧ 
  ∀ L' : Set (ℝ × ℝ), P ∈ L' ∧ is_tangent L' C → L' = L :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_uniqueness_l1242_124251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_16_minus_4pi_l1242_124245

/-- A square with side length 4 -/
structure Square where
  side_length : ℝ
  is_four : side_length = 4

/-- A line segment with length 4 and endpoints on adjacent sides of the square -/
structure Segment (s : Square) where
  length : ℝ
  is_four : length = 4
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ
  on_adjacent_sides : Bool

/-- The set of all valid segments -/
def T (s : Square) : Set (Segment s) := {seg : Segment s | seg.on_adjacent_sides = true}

/-- The region enclosed by the midpoints of segments in T -/
def enclosed_region (s : Square) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (r : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem enclosed_area_is_16_minus_4pi (s : Square) :
  area (enclosed_region s) = 16 - 4 * Real.pi := by sorry

#check enclosed_area_is_16_minus_4pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_16_minus_4pi_l1242_124245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_driver_net_pay_rate_l1242_124291

theorem driver_net_pay_rate (travel_time speed fuel_efficiency earnings_per_mile gasoline_cost : ℝ) 
  (h1 : travel_time > 0) 
  (h2 : speed > 0) 
  (h3 : fuel_efficiency > 0) 
  (h4 : earnings_per_mile > 0) 
  (h5 : gasoline_cost > 0) : 
  let distance := travel_time * speed
  let gasoline_used := distance / fuel_efficiency
  let total_earnings := distance * earnings_per_mile
  let total_gasoline_cost := gasoline_used * gasoline_cost
  let net_earnings := total_earnings - total_gasoline_cost
  let net_pay_rate := net_earnings / travel_time
  net_pay_rate = 28.8 := by
  sorry

#check driver_net_pay_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_driver_net_pay_rate_l1242_124291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_tangent_equation_l1242_124285

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := (1/3) * x^3

-- Define the point P
noncomputable def P : ℝ × ℝ := (2, 8/3)

-- Theorem for the slope of the tangent line
theorem tangent_slope :
  (deriv curve) P.1 = 4 := by sorry

-- Theorem for the equation of the tangent line
theorem tangent_equation (x y : ℝ) :
  (12 * x - 3 * y - 16 = 0) ↔ 
  (y - P.2 = (deriv curve P.1) * (x - P.1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_tangent_equation_l1242_124285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_statements_l1242_124204

theorem three_true_statements :
  (∀ x : ℝ, x^2 - x + (1/4 : ℝ) ≥ 0) ∧
  (∃ x : ℝ, x > 0 ∧ Real.log x + 1 / Real.log x ≤ 2) ∧
  (¬ (∀ a b c : ℝ, a > b ↔ a * c^2 > b * c^2)) ∧
  (∀ x : ℝ, (3 : ℝ)^x - (3 : ℝ)^(-x) = -((3 : ℝ)^(-x) - (3 : ℝ)^x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_true_statements_l1242_124204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_height_sarah_height_l1242_124230

-- Define the lamppost height and shadow length
noncomputable def lamppost_height : ℝ := 60
noncomputable def lamppost_shadow : ℝ := 15

-- Define Mike's shadow length
noncomputable def mike_shadow : ℝ := 18 / 12  -- Convert inches to feet

-- Define Sarah's shadow length
noncomputable def sarah_shadow : ℝ := 24 / 12  -- Convert inches to feet

-- Define the height-to-shadow ratio
noncomputable def height_shadow_ratio : ℝ := lamppost_height / lamppost_shadow

-- Theorem for Mike's height
theorem mike_height : 
  height_shadow_ratio * mike_shadow * 12 = 72 := by sorry

-- Theorem for Sarah's height
theorem sarah_height : 
  height_shadow_ratio * sarah_shadow * 12 = 96 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mike_height_sarah_height_l1242_124230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_in_101st_group_l1242_124220

/-- The sequence of numbers in the nth group -/
def group_sequence (n : ℕ+) : List ℕ :=
  List.range n.val |>.map (fun i => 2^(n.val - 1 + i))

/-- The sum of the first n positive integers -/
def triangle_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The first number in the nth group -/
def first_in_group (n : ℕ+) : ℕ :=
  2^(triangle_number (n.val - 1))

theorem first_in_101st_group :
  first_in_group 101 = 2^5050 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_in_101st_group_l1242_124220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_solution_set_implies_a_range_l1242_124229

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) → 
  a ∈ Set.Icc (-2) (6/5) ∧ a ≠ 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_solution_set_implies_a_range_l1242_124229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_squares_minimizes_sum_squared_residuals_l1242_124279

/-- Represents a data point in a 2D plane -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Represents the parameters of a linear regression line -/
structure RegressionParams where
  a : ℝ
  b : ℝ

/-- Calculates the residual for a single data point given regression parameters -/
def calculateResidual (point : DataPoint) (params : RegressionParams) : ℝ :=
  point.y - (params.a + params.b * point.x)

/-- Calculates the sum of squared residuals for a set of data points -/
def sumSquaredResiduals (data : List DataPoint) (params : RegressionParams) : ℝ :=
  (data.map fun point => (calculateResidual point params) ^ 2).sum

/-- States that the method of least squares minimizes the sum of squared residuals -/
theorem least_squares_minimizes_sum_squared_residuals
  (data : List DataPoint) (params : RegressionParams) :
  ∀ otherParams : RegressionParams,
    sumSquaredResiduals data params ≤ sumSquaredResiduals data otherParams := by
  sorry

#check least_squares_minimizes_sum_squared_residuals

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_squares_minimizes_sum_squared_residuals_l1242_124279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_calculation_max_marks_calculation_proof_l1242_124213

theorem max_marks_calculation (passing_percentage : ℚ) (scored_marks : ℕ) (shortfall : ℕ) (max_marks : ℕ) : Prop :=
  passing_percentage = 45 / 100 ∧
  scored_marks = 267 ∧
  shortfall = 43 ∧
  max_marks = 689 ∧
  max_marks = (scored_marks + shortfall) / passing_percentage

theorem max_marks_calculation_proof : ∃ (passing_percentage : ℚ) (scored_marks shortfall max_marks : ℕ),
  max_marks_calculation passing_percentage scored_marks shortfall max_marks :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_calculation_max_marks_calculation_proof_l1242_124213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangles_is_160_l1242_124275

/-- Represents a line in the coordinate plane --/
inductive Line
  | Horizontal (k : ℤ)
  | PositiveSlope (k : ℤ)
  | NegativeSlope (k : ℤ)

/-- The set of all lines in the problem --/
def all_lines : Set Line :=
  { l | ∃ k : ℤ, -5 ≤ k ∧ k ≤ 5 ∧
    (l = Line.Horizontal k ∨ l = Line.PositiveSlope k ∨ l = Line.NegativeSlope k) }

/-- The side length of the equilateral triangles formed --/
noncomputable def triangle_side_length : ℝ := 1 / Real.sqrt 3

/-- The number of equilateral triangles formed by the intersection of the lines --/
def num_triangles (lines : Set Line) : ℕ := sorry

/-- The main theorem stating that the number of triangles formed is 160 --/
theorem num_triangles_is_160 : num_triangles all_lines = 160 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_triangles_is_160_l1242_124275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_diverse_collection_size_l1242_124246

/-- A collection of coins is n-diverse if no value occurs more than n times -/
def nDiverse (n : ℕ) (coins : Multiset ℤ) : Prop :=
  ∀ x : ℤ, (coins.count x) ≤ n

/-- A number is n-reachable if there exist n coins in the collection that sum to it -/
def nReachable (n : ℕ) (coins : Multiset ℤ) (s : ℤ) : Prop :=
  ∃ subcoins : Multiset ℤ, subcoins ⊆ coins ∧ Multiset.card subcoins = n ∧ Multiset.sum subcoins = s

/-- The main theorem -/
theorem least_diverse_collection_size (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) :
  (∀ D : ℕ, D ≥ n + k - 1 →
    ∀ coins : Multiset ℤ, Multiset.card coins = D → nDiverse n coins →
      ∃ reachable : Finset ℤ, Finset.card reachable ≥ k ∧
        ∀ s ∈ reachable, nReachable n coins s) ∧
  (∀ D : ℕ, D < n + k - 1 →
    ∃ coins : Multiset ℤ, Multiset.card coins = D ∧ nDiverse n coins ∧
      ∀ reachable : Finset ℤ, (∀ s ∈ reachable, nReachable n coins s) →
        Finset.card reachable < k) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_diverse_collection_size_l1242_124246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_is_24pi_l1242_124201

/-- Represents a right-angled triangle inside a square -/
structure TriangleInSquare where
  /-- Length of side AB of the triangle -/
  ab : ℝ
  /-- Length of side BP of the triangle -/
  bp : ℝ
  /-- Side length of the square -/
  square_side : ℝ
  /-- Condition: AB is positive -/
  ab_pos : 0 < ab
  /-- Condition: BP is positive -/
  bp_pos : 0 < bp
  /-- Condition: Square side is positive -/
  square_pos : 0 < square_side
  /-- Condition: Triangle fits inside the square -/
  fits_in_square : ab ≤ square_side ∧ bp ≤ square_side

/-- The path length of point P when rotating the triangle -/
noncomputable def path_length (t : TriangleInSquare) : ℝ := 24 * Real.pi

/-- Theorem stating the path length for the given triangle and square -/
theorem path_length_is_24pi (t : TriangleInSquare) 
  (h1 : t.ab = 2) 
  (h2 : t.bp = 4) 
  (h3 : t.square_side = 6) : 
  path_length t = 24 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_is_24pi_l1242_124201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1242_124250

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.rpow (6/10) (6/10)) 
  (hb : b = Real.rpow (6/10) (3/2)) 
  (hc : c = Real.rpow (3/2) (6/10)) :
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1242_124250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_octagon_with_semicircles_l1242_124292

/-- The side length of the regular octagon -/
def octagon_side_length : ℝ := 3

/-- The number of semicircles inside the octagon -/
def num_semicircles : ℕ := 8

/-- The area of a regular octagon with side length s -/
noncomputable def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

/-- The area of a semicircle with radius r -/
noncomputable def semicircle_area (r : ℝ) : ℝ := Real.pi * r^2 / 2

/-- The theorem stating the area of the shaded region -/
theorem shaded_area_in_octagon_with_semicircles :
  let octagon_area := octagon_area octagon_side_length
  let total_semicircle_area := num_semicircles * semicircle_area (octagon_side_length / 2)
  octagon_area - total_semicircle_area = 54 + 54 * Real.sqrt 2 - 18 * Real.pi := by
  sorry

#check shaded_area_in_octagon_with_semicircles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_in_octagon_with_semicircles_l1242_124292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_properties_l1242_124202

/-- The ellipse C with given properties -/
structure EllipseC where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  eq : Set (ℝ × ℝ)
  ecc : (a^2 - b^2) / a^2 = 3/4

/-- The line l₁ with given properties -/
def l₁ (C : EllipseC) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 / C.a + p.2 / C.b = 1}

/-- The chord length of l₁ on C is √5 -/
axiom chord_length (C : EllipseC) : 
  ∃ p q : ℝ × ℝ, p ∈ C.eq ∧ q ∈ C.eq ∧ p ∈ l₁ C ∧ q ∈ l₁ C ∧ 
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 5

/-- The circle D with given properties -/
structure CircleD where
  m : ℝ
  eq : Set (ℝ × ℝ)

/-- l₁ is tangent to D -/
axiom l₁_tangent_D (C : EllipseC) (D : CircleD) :
  ∃! p : ℝ × ℝ, p ∈ l₁ C ∧ p ∈ D.eq

/-- The line l₂ passing through (3, 0) -/
def l₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ k : ℝ, p.2 = k * (p.1 - 3)}

/-- Theorem: Properties of ellipse C, circle D, and lines l₁ and l₂ -/
theorem ellipse_circle_properties (C : EllipseC) (D : CircleD) :
  -- 1. Standard equation of ellipse C
  (C.eq = {p : ℝ × ℝ | p.1^2/4 + p.2^2 = 1}) ∧
  -- 2. Standard equation of circle D
  (D.eq = {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 2)^2 = 5}) ∧
  -- 3. Range of |EF|•|MN|
  (∀ E F M N : ℝ × ℝ,
    E ∈ C.eq ∧ F ∈ C.eq ∧ E ∈ l₂ ∧ F ∈ l₂ ∧
    M ∈ D.eq ∧ N ∈ D.eq ∧ M ∈ l₂ ∧ N ∈ l₂ →
    0 < ((E.1 - F.1)^2 + (E.2 - F.2)^2) * ((M.1 - N.1)^2 + (M.2 - N.2)^2) ∧
    ((E.1 - F.1)^2 + (E.2 - F.2)^2) * ((M.1 - N.1)^2 + (M.2 - N.2)^2) ≤ 64) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_circle_properties_l1242_124202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_x_axis_max_at_negative_one_three_zero_points_l1242_124223

noncomputable def f (a x : ℝ) : ℝ := (x^2 + (a+1)*x + 1) * Real.exp x

theorem tangent_parallel_x_axis (a : ℝ) : 
  (deriv (f a)) 0 = 0 ↔ a = -2 := by sorry

theorem max_at_negative_one (a : ℝ) : 
  (∀ x : ℝ, f a (-1) ≥ f a x) ↔ a < -1 := by sorry

noncomputable def g (m a x : ℝ) : ℝ := m * f a x - 1

theorem three_zero_points (m : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g m 2 x = 0 ∧ g m 2 y = 0 ∧ g m 2 z = 0) 
  ↔ m > Real.exp 4 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_parallel_x_axis_max_at_negative_one_three_zero_points_l1242_124223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_length_l1242_124270

/-- Represents a point in the complex plane -/
structure ComplexPoint where
  re : ℝ
  im : ℝ

/-- The length of a vector in the complex plane -/
noncomputable def vectorLength (v : ComplexPoint) : ℝ :=
  Real.sqrt (v.re ^ 2 + v.im ^ 2)

/-- The difference between two complex points -/
def vectorDiff (p q : ComplexPoint) : ComplexPoint :=
  { re := q.re - p.re, im := q.im - p.im }

theorem parallelogram_diagonal_length :
  let a : ComplexPoint := { re := 0, im := 1 }  -- i
  let b : ComplexPoint := { re := 1, im := 0 }  -- 1
  let c : ComplexPoint := { re := 4, im := 2 }  -- 4+2i
  let d : ComplexPoint := { re := 3, im := 3 }  -- Calculated point D
  vectorLength (vectorDiff b d) = Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_length_l1242_124270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_train_crossing_time_l1242_124286

/-- The time it takes for two bullet trains to cross each other -/
noncomputable def crossingTime (trainLength : ℝ) (time1 : ℝ) (time2 : ℝ) : ℝ :=
  (2 * trainLength) / (trainLength / time1 + trainLength / time2)

/-- Theorem: The crossing time for the given conditions is approximately 13.33 seconds -/
theorem bullet_train_crossing_time :
  let trainLength : ℝ := 120
  let time1 : ℝ := 10
  let time2 : ℝ := 20
  ∃ ε > 0, |crossingTime trainLength time1 time2 - 40/3| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bullet_train_crossing_time_l1242_124286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_101_l1242_124241

def is_valid_sequence (seq : List Nat) : Prop :=
  seq.length > 0 ∧
  ∀ n ∈ seq, 1000 ≤ n ∧ n < 10000 ∧
  ∀ i, 0 ≤ i ∧ i < seq.length - 1 →
    (seq[i]! / 100) % 10 = (seq[i+1]! / 1000) % 10 ∧
    (seq[i]! / 10) % 10 = (seq[i+1]! / 100) % 10 ∧
    seq[i]! % 10 = (seq[i+1]! / 10) % 10 ∧
  (seq.getLast! / 100) % 10 = (seq.head! / 1000) % 10 ∧
  (seq.getLast! / 10) % 10 = (seq.head! / 100) % 10 ∧
  seq.getLast! % 10 = (seq.head! / 10) % 10

theorem sum_divisible_by_101 (seq : List Nat) (h : is_valid_sequence seq) :
  ∃ k : Nat, seq.sum = 101 * k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_101_l1242_124241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_bound_iff_a_in_range_l1242_124235

/-- The polynomial P(x) = x^2 - 2ax - a^2 - 3/4 -/
noncomputable def P (a x : ℝ) : ℝ := x^2 - 2*a*x - a^2 - 3/4

theorem polynomial_bound_iff_a_in_range :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 1, |P a x| ≤ 1) ↔ a ∈ Set.Icc (-1/2) (Real.sqrt 2 / 4) := by
  sorry

#check polynomial_bound_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_bound_iff_a_in_range_l1242_124235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_threshold_l1242_124247

noncomputable def F (m : ℝ) : ℝ :=
  m^4 * Real.sin (Real.pi / m) * Real.sqrt (1 / m^6 + Real.sin (Real.pi / m)^4)

theorem F_threshold (m : ℝ) :
  (m > 5 → F m > 100) ∧ (m > 44200 → F m > 1000000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_threshold_l1242_124247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_power_function_l1242_124203

-- Define the power function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + 2*m - 3)

-- Theorem statement
theorem increasing_power_function :
  ∃! m : ℝ, (∀ x > 0, Monotone (f m)) ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_power_function_l1242_124203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1242_124257

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x - 1)

def domain : Set ℝ := {1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {1, Real.sqrt 3, Real.sqrt 5} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1242_124257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_shift_l1242_124280

theorem sin_cos_sum_shift (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * x + π / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_shift_l1242_124280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_product_derivative_l1242_124262

noncomputable def f (x : ℝ) := Real.sqrt (1 + x^2)

theorem f_product_derivative (x : ℝ) (hx : x > 0) : 
  f x * (deriv f x) = x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_product_derivative_l1242_124262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_from_lines_l1242_124295

-- Define the basic geometric elements
variable (Circle₁ Circle₂ : Type) [MetricSpace Circle₁] [MetricSpace Circle₂]
variable (A B C D : EuclideanSpace ℝ (Fin 2))
variable (line : Set (EuclideanSpace ℝ (Fin 2)))

-- Define the conditions
def TouchExternally (C₁ C₂ : Type) [MetricSpace C₁] [MetricSpace C₂] (p : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def Tangent (C : Type) [MetricSpace C] (l : Set (EuclideanSpace ℝ (Fin 2))) (p : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

def Intersects (C : Type) [MetricSpace C] (l : Set (EuclideanSpace ℝ (Fin 2))) (p : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

variable (h₁ : TouchExternally Circle₁ Circle₂ D)
variable (h₂ : Tangent Circle₁ line A)
variable (h₃ : Intersects Circle₂ line B)
variable (h₄ : Intersects Circle₂ line C)

-- Define the distance function
noncomputable def distanceToLine (p : EuclideanSpace ℝ (Fin 2)) (l : Set (EuclideanSpace ℝ (Fin 2))) : ℝ := sorry

-- Define the line through two points
def Line (p q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- State the theorem
theorem point_equidistant_from_lines :
  distanceToLine A (Line B D) = distanceToLine A (Line C D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_equidistant_from_lines_l1242_124295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1242_124254

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - 2^x) + 1 / Real.sqrt (x + 3)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x > -3 ∧ x ≤ 0}

-- Theorem statement
theorem f_domain : 
  {x : ℝ | ∃ y, f x = y} = domain_f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1242_124254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_seating_probability_l1242_124299

/-- Represents a group of people with their majors -/
structure GroupInfo where
  total : ℕ
  math : ℕ
  physics : ℕ
  biology : ℕ

/-- Calculates the probability of a specific seating arrangement -/
def seating_probability (g : GroupInfo) : ℚ :=
  let favorable_outcomes := g.total * 7 * 5 * (g.biology.factorial)
  let total_outcomes := (g.total - 1).factorial
  ↑favorable_outcomes / ↑total_outcomes

/-- The main theorem stating the probability of the specific seating arrangement -/
theorem specific_seating_probability :
  let g : GroupInfo := { total := 13, math := 6, physics := 3, biology := 4 }
  seating_probability g = 1 / 21952 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_seating_probability_l1242_124299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triple_angle_integer_l1242_124217

theorem tangent_triple_angle_integer (α : Real) :
  (Int.floor (Real.tan α) = Real.tan α) ∧ (Int.floor (Real.tan (3 * α)) = Real.tan (3 * α)) →
  Real.tan α = -1 ∨ Real.tan α = 0 ∨ Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triple_angle_integer_l1242_124217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1242_124252

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x * Real.cos x - Real.cos x ^ 2 - 1 / 2

theorem problem_solution :
  ∃ (A B C a b : ℝ),
    (∀ x, f x ≥ -2) ∧
    (∀ x, f (x + π) = f x) ∧
    (∀ T, (∀ x, f (x + T) = f x) → T ≥ π) ∧
    3 = 2 * a ∧
    f C = 0 ∧
    1 * Real.sin B = 2 * Real.sin A ∧
    a = Real.sqrt 3 ∧
    b = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1242_124252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l1242_124205

/-- Represents the area of a lawn -/
structure LawnArea where
  size : ℚ
  size_pos : size > 0

/-- Represents the mowing rate of a lawn mower -/
structure MowingRate where
  rate : ℚ
  rate_pos : rate > 0

/-- Represents a person with their lawn area and mowing rate -/
structure Person where
  name : String
  lawn : LawnArea
  mower : MowingRate

/-- Calculates the time taken to mow a lawn -/
def mowingTime (p : Person) : ℚ :=
  p.lawn.size / p.mower.rate

theorem beth_finishes_first (andy beth carlos : Person)
  (h1 : andy.lawn.size = 2 * beth.lawn.size)
  (h2 : andy.lawn.size = 3 * carlos.lawn.size)
  (h3 : carlos.mower.rate = (1/2) * beth.mower.rate)
  (h4 : carlos.mower.rate = (1/3) * andy.mower.rate) :
  mowingTime beth < min (mowingTime andy) (mowingTime carlos) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l1242_124205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_integer_points_l1242_124263

theorem hyperbola_integer_points : 
  ∃ (points : Finset (ℤ × ℤ)), 
    points = {p : ℤ × ℤ | p.2 = 2013 / p.1 ∧ p.1 ≠ 0} ∧ 
    Finset.card points = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_integer_points_l1242_124263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_perpendicular_lines_l1242_124244

/-- Parabola with focus F and intersecting line -/
structure ParabolaWithLine where
  p : ℝ
  F : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_p_pos : p > 0
  h_focus : F = (p/2, 0)
  h_parabola : ∀ (x y : ℝ), y^2 = 2*p*x ↔ (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)
  h_line : ∀ (x y : ℝ), y = x - 8 ↔ (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ (x = C.1 ∧ y = C.2)
  h_C : C = (8, 0)
  h_vector : F.1 - C.1 = 3 * (F.1 - 0)

/-- Main theorem -/
theorem parabola_and_perpendicular_lines (pwl : ParabolaWithLine) :
  (∀ (x y : ℝ), y^2 = 8*x ↔ (x = pwl.A.1 ∧ y = pwl.A.2) ∨ (x = pwl.B.1 ∧ y = pwl.B.2)) ∧
  (pwl.A.1 * pwl.B.1 + pwl.A.2 * pwl.B.2 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_perpendicular_lines_l1242_124244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_race_odds_l1242_124298

/-- Represents the odds against a horse winning --/
structure Odds where
  against : ℚ
  forWinning : ℚ

/-- Calculates the probability of winning given the odds --/
def probability (odds : Odds) : ℚ :=
  odds.forWinning / (odds.against + odds.forWinning)

theorem horse_race_odds (oddsA oddsB oddsC : Odds) : 
  oddsA = { against := 4, forWinning := 1 } →
  oddsB = { against := 1, forWinning := 2 } →
  probability oddsA + probability oddsB + probability oddsC = 1 →
  oddsC = { against := 13, forWinning := 2 } := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_race_odds_l1242_124298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_count_l1242_124277

def is_lattice_point (p : ℤ × ℤ) : Prop := true

def in_region (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  (y = x.natAbs ∧ y ≤ -x^3 + 6) ∨ (x.natAbs ≤ y ∧ y ≤ -x^3 + 6)

theorem lattice_points_count :
  ∃! (s : Finset (ℤ × ℤ)), (∀ p ∈ s, is_lattice_point p ∧ in_region p) ∧ s.card = 9 :=
sorry

#check lattice_points_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_count_l1242_124277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1242_124216

def A : Set ℝ := {x | x > -2}
def B : Set ℝ := {x | 1 - x > 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (-2) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1242_124216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_hyperbola_tangency_l1242_124258

-- Define the concepts
def Line : Type := sorry
def Hyperbola : Type := sorry
def Point : Type := sorry

-- Define the relationships
def has_one_common_point (l : Line) (h : Hyperbola) : Prop := sorry
def is_tangent (l : Line) (h : Hyperbola) : Prop := sorry

-- Define necessary but not sufficient
def necessary_but_not_sufficient (P Q : Line → Hyperbola → Prop) : Prop :=
  (∀ l h, Q l h → P l h) ∧ ∃ l h, P l h ∧ ¬Q l h

-- State the theorem
theorem line_hyperbola_tangency :
  necessary_but_not_sufficient
    (λ l h => has_one_common_point l h)
    (λ l h => is_tangent l h) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_hyperbola_tangency_l1242_124258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_trisection_point_l1242_124222

/-- Given points A, B, and C in the plane, and a line passing through A and 
    one of the trisection points of BC, prove that the line has a specific equation. -/
theorem line_through_trisection_point (A B C : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (1, -2) → 
  C = (7, 4) → 
  ∃ (T : ℝ × ℝ), (T.1 - B.1 = (2/3) * (C.1 - B.1) ∧ T.2 - B.2 = (2/3) * (C.2 - B.2)) ∨
                 (T.1 - B.1 = (1/3) * (C.1 - B.1) ∧ T.2 - B.2 = (1/3) * (C.2 - B.2)) →
  A.1 + 2*A.2 - 9 = 0 ∧ T.1 + 2*T.2 - 9 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_trisection_point_l1242_124222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_area_consistency_l1242_124221

/-- A rhombus with given diagonals and area -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  area : ℝ

/-- Calculate the perimeter of a rhombus -/
noncomputable def perimeter (r : Rhombus) : ℝ :=
  4 * Real.sqrt ((r.diagonal1 / 2) ^ 2 + (r.diagonal2 / 2) ^ 2)

/-- Theorem stating the perimeter of a specific rhombus -/
theorem rhombus_perimeter : 
  let r : Rhombus := { diagonal1 := 24, diagonal2 := 10, area := 120 }
  perimeter r = 52 := by
  sorry

/-- Verify that the area is consistent with the diagonals -/
theorem area_consistency (r : Rhombus) : 
  r.area = (1 / 2) * r.diagonal1 * r.diagonal2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_area_consistency_l1242_124221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_collinearity_l1242_124296

/-- Parabola E with equation y^2 = 4x -/
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Line with equation y = kx + b, where k ≠ 0 -/
def Line (k b x y : ℝ) : Prop := y = k*x + b

/-- Point on x-axis -/
def OnXAxis (x y : ℝ) : Prop := y = 0

/-- Point symmetric to another point with respect to x-axis -/
def SymmetricXAxis (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ = x₂ ∧ y₁ = -y₂

/-- Point symmetric to another point with respect to y-axis -/
def SymmetricYAxis (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ = -x₂ ∧ y₁ = y₂

/-- Three points are collinear -/
def AreCollinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

theorem parabola_line_intersection_collinearity 
  (k b : ℝ) (h : k ≠ 0)
  (xA yA xB yB xC yC xP yP xQ yQ : ℝ) :
  Parabola xA yA →
  Parabola xB yB →
  Line k b xA yA →
  Line k b xB yB →
  Line k b xC yC →
  OnXAxis xC yC →
  SymmetricXAxis xB yB xP yP →
  SymmetricYAxis xC yC xQ yQ →
  AreCollinear xA yA xP yP xQ yQ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_collinearity_l1242_124296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_and_range_of_a_l1242_124236

-- Define set A
def A : Set ℝ := {x | (4 : ℝ) / (x + 1) > 1}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*x - a^2 + 2*a < 0}

-- Theorem statement
theorem set_A_and_range_of_a (h : ∀ a, a < 1) :
  (A = Set.Ioo (-1 : ℝ) 3) ∧
  (∀ a, (∀ x, x ∈ A → x ∈ B a) ↔ a ≤ -3) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_and_range_of_a_l1242_124236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_rectangle_l1242_124288

/-- Two lines in a plane -/
structure TwoLines where
  line1 : Set (ℝ × ℝ)
  line2 : Set (ℝ × ℝ)

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : Point) (l : Set Point) : ℝ := sorry

/-- Angle bisector of two lines -/
def angleBisector (l1 l2 : Set Point) : Set Point := sorry

/-- Rectangle in the plane -/
structure Rectangle where
  vertices : Set Point
  is_rectangle : ∃ (v1 v2 v3 v4 : Point), vertices = {v1, v2, v3, v4} ∧ sorry

/-- The locus of points with constant sum of distances to two lines -/
def locusOfConstantSumDistance (lines : TwoLines) (c : ℝ) : Set Point :=
  {p : Point | distanceToLine p lines.line1 + distanceToLine p lines.line2 = c}

/-- Perpendicularity of a line to another line -/
def isPerpendicular (l1 l2 : Set Point) : Prop := sorry

theorem locus_is_rectangle (lines : TwoLines) (c : ℝ) :
  ∃ (r : Rectangle), locusOfConstantSumDistance lines c = r.vertices ∧
    (∀ (side : Set Point), side ⊆ r.vertices → 
      ∃ (bisector : Set Point), bisector = angleBisector lines.line1 lines.line2 ∧ 
        isPerpendicular side bisector) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_rectangle_l1242_124288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_P_4_l1242_124268

noncomputable def y₀ : ℝ := 25
noncomputable def y₁ : ℝ := 60
noncomputable def y₂ : ℝ := 50
noncomputable def y₃ : ℝ := 48

noncomputable def P : ℝ → ℝ := λ x => y₀ + (y₁ - y₀) * x + 
  (y₂ - 2*y₁ + y₀) * x * (x-1) / 2 + 
  (y₃ - 3*y₂ + 3*y₁ - y₀) * x * (x-1) * (x-2) / 6

theorem expected_value_P_4 : 
  P 4 = 107 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_P_4_l1242_124268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_l1242_124207

/-- Represents the price of oil per kg in Rupees -/
structure OilPrice where
  value : ℝ
  is_positive : value > 0

/-- Calculates the reduced price after a percentage reduction -/
noncomputable def reduced_price (original : OilPrice) (reduction_percent : ℝ) : OilPrice :=
  { value := original.value * (1 - reduction_percent / 100),
    is_positive := by
      have h : 1 - reduction_percent / 100 > 0 := sorry
      exact (mul_pos original.is_positive h) }

/-- Calculates the quantity of oil that can be bought for a given price and amount -/
noncomputable def quantity (price : OilPrice) (amount : ℝ) : ℝ :=
  amount / price.value

theorem oil_price_reduction (original : OilPrice) :
  let reduced := reduced_price original 25
  quantity reduced 1300 = quantity original 1300 + 5 →
  ∃ ε > 0, abs (reduced.value - 65) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_reduction_l1242_124207
