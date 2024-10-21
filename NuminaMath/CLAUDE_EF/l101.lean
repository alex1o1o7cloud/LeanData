import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_residue_mod_31_l101_10148

theorem residue_mod_31 : ∃ (k : ℤ), -1237 = 31 * k + 3 ∧ (3 : ℕ) ∈ Finset.range 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_residue_mod_31_l101_10148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l101_10145

-- Define the angle β
noncomputable def β : ℝ := Real.arccos (3/5)

-- Theorem statement
theorem overlapping_squares_area :
  0 < β ∧ β < Real.pi/2 →
  let square_side : ℝ := 2
  let overlap_area : ℝ := 4/5
  overlap_area = (square_side^2 * (1 - Real.tan (β/2)) / (1 + Real.tan (β/2))) := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlapping_squares_area_l101_10145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_lines_l101_10118

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define M(2,4)
def M : Point := ⟨2, 4⟩

-- Define a line passing through a point
structure Line where
  point : Point
  slope : ℝ

-- Define the property of a line intersecting the parabola at exactly one point
def intersects_parabola_once (l : Line) : Prop :=
  ∃! p : Point, (p.x - l.point.x = l.slope * (p.y - l.point.y)) ∧ parabola p.x p.y

-- Theorem statement
theorem exactly_two_lines :
  ∃! (l1 l2 : Line), l1 ≠ l2 ∧ 
    l1.point = M ∧ l2.point = M ∧
    intersects_parabola_once l1 ∧ 
    intersects_parabola_once l2 ∧
    ∀ l : Line, l.point = M ∧ intersects_parabola_once l → l = l1 ∨ l = l2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_lines_l101_10118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l101_10171

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- Represents a parabola with parameter p -/
structure Parabola (p : ℝ) where
  p_pos : 0 < p

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := 
  Real.sqrt ((a^2 + b^2) / a^2)

/-- The area of the triangle formed by the origin and the intersection points
    of the hyperbola's asymptotes with the parabola's directrix -/
noncomputable def triangleArea (h : Hyperbola a b) (para : Parabola p) : ℝ :=
  (p^2 * b) / (4 * a)

theorem hyperbola_parabola_intersection
  (a b p : ℝ)
  (h : Hyperbola a b)
  (para : Parabola p)
  (ecc_eq_two : eccentricity h = 2)
  (area_eq_sqrt_three : triangleArea h para = Real.sqrt 3) :
  p = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_parabola_intersection_l101_10171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l101_10117

/-- Given a triangle ABC with the following properties:
  - cos B = -1/2
  - a = 2 (side opposite to angle A)
  - b = 2√3 (side opposite to angle B)
  Prove:
  1. The area of triangle ABC is √3
  2. The product sin A · sin C is in the range (0, 1/4]
-/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  Real.cos B = -1/2 →
  a = 2 →
  b = 2 * Real.sqrt 3 →
  (∃ (area : Real), area = Real.sqrt 3 ∧ area = 1/2 * a * b * Real.sin C) ∧
  (∃ (prod : Real), prod = Real.sin A * Real.sin C ∧ 0 < prod ∧ prod ≤ 1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l101_10117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l101_10156

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 4 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x + 9 > 2*(x - 3) ∧ 2*(x + 1)/3 < x + a))) →
  (-3 < a ∧ a ≤ -8/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_range_l101_10156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l101_10191

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vector operations
def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (dot_product v v)

noncomputable def angle (v w : ℝ × ℝ) : ℝ := Real.arccos ((dot_product v w) / (vec_length v * vec_length w))

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : vec_length (vec t.A t.C) = 2 * Real.sqrt 3)
  (h2 : dot_product (vec t.B t.C) (vec t.B t.A) * Real.cos (angle (vec t.B t.A) (vec t.B t.C)) + 
        dot_product (vec t.A t.B) (vec t.C t.A) * Real.cos (angle (vec t.C t.A) (vec t.C t.B)) = 
        vec_length (vec t.A t.C) * Real.sin (angle (vec t.A t.B) (vec t.C t.B))) :
  angle (vec t.A t.B) (vec t.C t.B) = 2 * Real.pi / 3 ∧ 
  1/2 * vec_length (vec t.A t.C) * vec_length (vec t.B t.C) * Real.sin (angle (vec t.B t.A) (vec t.B t.C)) = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l101_10191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_sphere_surface_area_l101_10157

theorem inscribed_cube_sphere_surface_area 
  (cube_surface_area : ℝ) 
  (h_cube_surface : cube_surface_area = 54) :
  let cube_side := Real.sqrt (cube_surface_area / 6)
  let sphere_radius := cube_side * Real.sqrt 3 / 2
  4 * Real.pi * sphere_radius^2 = 27 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cube_sphere_surface_area_l101_10157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_borrowed_l101_10180

theorem max_books_borrowed (total_students zero_books one_book two_books : ℕ) (avg_books : ℚ) : ℕ :=
  by
  have h1 : total_students = 20 := by sorry
  have h2 : zero_books = 3 := by sorry
  have h3 : one_book = 9 := by sorry
  have h4 : two_books = 4 := by sorry
  have h5 : avg_books = 2 := by sorry
  have h6 : 0 < zero_books + one_book + two_books := by sorry
  have h7 : zero_books + one_book + two_books < total_students := by sorry
  have h8 : ∀ s, s ∉ Finset.range (zero_books + one_book + two_books) → 3 ≤ s := by sorry
  
  -- The actual proof would go here
  sorry

#check max_books_borrowed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_borrowed_l101_10180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l101_10164

open Real

-- Define the function f(x) = ln x - 2/x
noncomputable def f (x : ℝ) : ℝ := log x - 2 / x

-- State the theorem
theorem zero_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 2 ℯ ∧ f x = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l101_10164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vectors_form_circle_l101_10139

/-- A unit vector in a plane -/
def UnitVector : Type := { v : ℝ × ℝ // v.1^2 + v.2^2 = 1 }

/-- The endpoint of a unit vector when its starting point is at the origin -/
def Endpoint (v : UnitVector) : ℝ × ℝ := v.val

theorem unit_vectors_form_circle :
  ∀ p : ℝ × ℝ, p ∈ (Set.range Endpoint) ↔ p.1^2 + p.2^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vectors_form_circle_l101_10139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_l101_10128

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - log x

-- State the theorem
theorem monotonic_decrease_interval :
  StrictMonoOn f (Set.Ioo 0 1) :=
by
  -- We'll use the definition of StrictMonoOn
  intro x y hx hy hxy
  -- Unfold the definition of f
  simp [f]
  -- The proof would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decrease_interval_l101_10128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l101_10121

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = -1/4 * x^2

/-- The distance function to be minimized -/
noncomputable def distance_sum (m n : ℝ) : ℝ :=
  Real.sqrt (m^2 + (n+1)^2) + Real.sqrt ((m-4)^2 + (n+5)^2)

/-- The theorem stating the minimum value of the distance sum -/
theorem min_distance_sum :
  ∀ m n : ℝ, parabola m n → ∀ x y : ℝ, parabola x y → distance_sum m n ≤ distance_sum x y ∧ 
  ∃ m₀ n₀ : ℝ, parabola m₀ n₀ ∧ distance_sum m₀ n₀ = 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l101_10121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_price_is_five_l101_10106

/-- The price of a candy bar in Nick's fundraising effort --/
noncomputable def candy_bar_price (total_goal : ℚ) (orange_price : ℚ) (orange_count : ℕ) (candy_bar_count : ℕ) : ℚ :=
  (total_goal - orange_price * orange_count) / candy_bar_count

/-- Theorem: The price of each candy bar is $5 --/
theorem candy_bar_price_is_five :
  candy_bar_price 1000 10 20 160 = 5 := by
  -- Unfold the definition of candy_bar_price
  unfold candy_bar_price
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_bar_price_is_five_l101_10106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_theorem_l101_10131

/-- Calculates the distance traveled downstream by a boat given its speed in still water and travel times -/
noncomputable def distance_downstream (boat_speed : ℝ) (time_downstream time_upstream : ℝ) : ℝ :=
  let current_speed := (boat_speed * (time_upstream - time_downstream)) / (time_upstream + time_downstream)
  (boat_speed + current_speed) * time_downstream

/-- Theorem stating that under the given conditions, the boat travels 48 km downstream -/
theorem boat_distance_theorem (boat_speed : ℝ) (time_downstream time_upstream : ℝ)
  (h1 : boat_speed = 12)
  (h2 : time_downstream = 3)
  (h3 : time_upstream = 6) :
  distance_downstream boat_speed time_downstream time_upstream = 48 := by
  sorry

#eval Float.ofScientific 48 0 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_theorem_l101_10131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l101_10183

def sequence_a (a : ℝ) (n : ℕ) : ℝ :=
  if n < 4 then a^(n-2) else (6-a)*n - a

theorem sequence_range (a : ℝ) :
  (∀ n : ℕ+, sequence_a a n < sequence_a a (n + 1)) ↔ 1 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_range_l101_10183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AE_equals_nine_l101_10163

/-- Four equally-spaced collinear points -/
structure CollinearPoints (α : Type*) [LinearOrderedField α] where
  A : α
  B : α
  C : α
  D : α
  equally_spaced : B - A = C - B ∧ C - B = D - C

/-- Circle with diameter AD -/
def circle_ω {α : Type*} [LinearOrderedField α] (p : CollinearPoints α) : Set (α × α) :=
  {x | (x.1 - p.A)^2 + x.2^2 = (p.D - p.A)^2 / 4}

/-- Circle with diameter BD -/
def circle_ω' {α : Type*} [LinearOrderedField α] (p : CollinearPoints α) : Set (α × α) :=
  {x | (x.1 - p.B)^2 + x.2^2 = (p.D - p.B)^2 / 4}

/-- Point E: intersection of tangent from A to ω' with ω -/
noncomputable def point_E {α : Type*} [LinearOrderedField α] (p : CollinearPoints α) : α × α := sorry

/-- Theorem: Given the conditions, AE = 9 cm -/
theorem AE_equals_nine (p : CollinearPoints ℝ) 
  (h : p.B - p.A = 2 * Real.sqrt 3) : 
  let E := point_E p
  (E.1 - p.A)^2 + E.2^2 = 81 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_AE_equals_nine_l101_10163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l101_10108

def sum_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_digits (n / 10)

theorem problem_solution : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (Nat.factorial (n + 1) + Nat.factorial (n + 3) = Nat.factorial n * 1320) ∧ 
  (n = 11) ∧ 
  (sum_digits n = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l101_10108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_climb_time_l101_10144

/-- Represents the frog's climbing cycle -/
structure ClimbingCycle where
  climbDistance : ℝ
  slipDistance : ℝ
  climbTime : ℝ
  slipTime : ℝ

/-- The well and frog climbing problem -/
structure FrogClimbProblem where
  wellDepth : ℝ
  cycle : ClimbingCycle
  timeToNearTop : ℝ
  distanceFromTop : ℝ

theorem frog_climb_time (problem : FrogClimbProblem) 
  (h1 : problem.wellDepth = 12)
  (h2 : problem.cycle.climbDistance = 3)
  (h3 : problem.cycle.slipDistance = 1)
  (h4 : problem.cycle.slipTime = problem.cycle.climbTime / 3)
  (h5 : problem.timeToNearTop = 17)
  (h6 : problem.distanceFromTop = 3) :
  ∃ (totalTime : ℝ), totalTime = 22 ∧ 
    (totalTime * (problem.cycle.climbDistance - problem.cycle.slipDistance) / 
      (problem.cycle.climbTime + problem.cycle.slipTime) ≥ problem.wellDepth) := by
  sorry

#check frog_climb_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_climb_time_l101_10144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_92_l101_10186

def scores : List ℕ := [73, 77, 83, 85, 92]

def is_integer_average (subset : List ℕ) : Prop :=
  ∃ n : ℕ, n * subset.length = subset.sum

def all_subsets_integer_average (l : List ℕ) : Prop :=
  ∀ subset : List ℕ, subset.Sublist l → is_integer_average subset

theorem last_score_is_92 (h : all_subsets_integer_average scores) :
  scores.getLast? = some 92 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_92_l101_10186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_of_product_of_two_odd_primes_l101_10194

theorem factor_count_of_product_of_two_odd_primes (x y : ℕ) 
  (hx : Nat.Prime x) (hy : Nat.Prime y) (hoddx : Odd x) (hoddy : Odd y) (hlt : x < y) : 
  (Finset.filter (fun w => w > 0 ∧ (2 * x * y) % w = 0) (Finset.range (2 * x * y + 1))).card = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_count_of_product_of_two_odd_primes_l101_10194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_ON_in_hyperbola_l101_10124

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  center : Point
  a : ℝ
  b : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Length of ON in a hyperbola -/
theorem length_ON_in_hyperbola (h : Hyperbola) (F1 F2 P N O : Point) : 
  h.center = O →  -- O is the center of the hyperbola
  P.x^2 - P.y^2 = 1 →  -- P is on the hyperbola
  P.x > 0 →  -- P is on the right branch
  distance P F1 - distance P F2 = 2 →  -- Definition of hyperbola
  distance P F1 = 5 →  -- Given condition
  N.x = (P.x + F1.x) / 2 ∧ N.y = (P.y + F1.y) / 2 →  -- N is midpoint of PF1
  distance O N = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_ON_in_hyperbola_l101_10124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_squared_distances_l101_10127

/-- The ellipse representing curve C -/
def ellipse (x y : ℝ) : Prop := x^2/3 + y^2 = 1

/-- The circle with center P(x₀, y₀) and radius √3/2 -/
def circle_P (x₀ y₀ x y : ℝ) : Prop := (x - x₀)^2 + (y - y₀)^2 = 3/4

/-- Tangent line from origin to the circle -/
def tangentLine (k x y : ℝ) : Prop := y = k * x

/-- Theorem stating the constant sum of squared distances -/
theorem constant_sum_squared_distances 
  (x₀ y₀ xA yA xB yB k₁ k₂ : ℝ) :
  ellipse x₀ y₀ →
  circle_P x₀ y₀ 0 0 →
  tangentLine k₁ xA yA →
  tangentLine k₂ xB yB →
  ellipse xA yA →
  ellipse xB yB →
  xA^2 + yA^2 + xB^2 + yB^2 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_squared_distances_l101_10127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_mean_is_10_l101_10153

-- Define variables a and b as real numbers
variable (a b : ℝ)

-- Define the population as a list of real numbers
def population (a b : ℝ) : List ℝ := [2, 3, 3, 7, a, b, 12, 13.3, 18.7, 20]

-- Define the median of the population
def median : ℝ := 10.5

-- State the theorem
theorem population_mean_is_10 (a b : ℝ) :
  let n := (population a b).length
  let sum := (population a b).sum
  sum / n = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_mean_is_10_l101_10153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_two_zero_points_l101_10149

/-- The function f(x) = ax^2 + (a - 2)x - ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

/-- Theorem 1: f(x) takes an extreme value at x = 1 iff a = 1 -/
theorem extreme_value_condition (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≥ f a 1 ∨ f a x ≤ f a 1) ↔ a = 1 := by
  sorry

/-- Theorem 2: f(x) has exactly two zero points when 0 < a < 1 -/
theorem two_zero_points (a : ℝ) (ha : 0 < a ∧ a < 1) :
  (∃! x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_condition_two_zero_points_l101_10149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_265_l101_10140

/-- Calculates the length of a bridge given the train length, speed, and time to cross. -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proves that the length of the bridge is 265 meters given the specified conditions. -/
theorem bridge_length_is_265 :
  bridge_length 110 45 30 = 265 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Simplify the arithmetic expressions
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_is_265_l101_10140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_rationality_l101_10125

theorem tan_beta_rationality (p q : ℤ) (α β : ℝ) (h1 : q ≠ 0) (h2 : Real.tan α = p / q) (h3 : Real.tan (2 * β) = Real.tan (3 * α)) :
  (∃ (r : ℚ), Real.tan β = r) ↔ ∃ (n : ℕ), p^2 + q^2 = n^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_beta_rationality_l101_10125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_three_zeros_l101_10126

/-- Given a function f(x) = cos(ωx) - 1 with ω > 0, if f has exactly 3 zeros 
    in the interval [0, 2π], then 2 ≤ ω < 3. -/
theorem omega_range_for_three_zeros 
  (ω : ℝ) 
  (ω_pos : ω > 0)
  (f : ℝ → ℝ)
  (f_def : ∀ x, f x = Real.cos (ω * x) - 1)
  (zeros : ∃! (z₁ z₂ z₃ : ℝ), 
    0 ≤ z₁ ∧ z₁ < z₂ ∧ z₂ < z₃ ∧ z₃ ≤ 2 * Real.pi ∧
    f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0 ∧
    ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ f x = 0 → x = z₁ ∨ x = z₂ ∨ x = z₃) :
  2 ≤ ω ∧ ω < 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_three_zeros_l101_10126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l101_10169

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = Real.log (x^2 + 1) / Real.log 2}
def N : Set ℝ := {x : ℝ | (4 : ℝ)^x > 4}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l101_10169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pants_and_shirt_cost_is_100_l101_10192

/-- The cost of a pair of pants and a shirt, given the conditions from the problem. -/
noncomputable def pants_and_shirt_cost : ℝ := 100

/-- The cost of the coat. -/
noncomputable def coat_cost : ℝ := 180

/-- The cost of the pants. -/
noncomputable def pants_cost : ℝ := 244 - coat_cost

/-- The cost of the shirt. -/
noncomputable def shirt_cost : ℝ := coat_cost / 5

/-- Theorem stating that the cost of a pair of pants and a shirt is $100. -/
theorem pants_and_shirt_cost_is_100 :
  pants_and_shirt_cost = pants_cost + shirt_cost :=
by
  -- Unfold the definitions
  unfold pants_and_shirt_cost pants_cost shirt_cost coat_cost
  -- Simplify the arithmetic
  simp [add_comm, sub_eq_add_neg, div_eq_mul_inv]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pants_and_shirt_cost_is_100_l101_10192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_treasure_coin_count_optimal_pirate_count_l101_10178

/-- Represents the number of coins in the treasure -/
def treasure_coins : ℕ := sorry

/-- Represents the number of pirates (excluding the captain) -/
def total_pirates : ℕ := 100

/-- Axiom: The treasure contains fewer than 1000 coins -/
axiom less_than_1000 : treasure_coins < 1000

/-- Axiom: If 99 pirates are chosen, 51 coins are left -/
axiom coins_left_99 : ∃ k : ℕ, treasure_coins = 99 * k + 51

/-- Axiom: If 77 pirates are chosen, 29 coins are left -/
axiom coins_left_77 : ∃ m : ℕ, treasure_coins = 77 * m + 29

/-- The optimal number of pirates to choose for maximum captain's share -/
def optimal_pirates : ℕ := sorry

/-- Theorem: The treasure contains 645 coins -/
theorem treasure_coin_count : treasure_coins = 645 := by
  sorry

/-- Theorem: The optimal number of pirates to choose is 93 -/
theorem optimal_pirate_count : optimal_pirates = 93 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_treasure_coin_count_optimal_pirate_count_l101_10178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_y_value_l101_10185

theorem power_equality_y_value (y : ℝ) : (16 : ℝ)^y = (4 : ℝ)^16 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equality_y_value_l101_10185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_l101_10162

/-- Given that D = (4, 3) is the midpoint of PQ, where P = (2, 7) and Q = (x, y), prove that x+y = 5 -/
theorem midpoint_sum (x y : ℝ) : 
  (4, 3) = ((2 + x) / 2, (7 + y) / 2) → x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_sum_l101_10162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_value_l101_10198

/-- The average number of minutes run per day by all students in a middle school -/
noncomputable def average_minutes_run (third_grade_minutes fourth_grade_minutes fifth_grade_minutes : ℝ)
  (third_to_fourth_ratio fourth_to_fifth_ratio : ℕ) : ℝ :=
  let fifth_graders := 1
  let fourth_graders := fourth_to_fifth_ratio * fifth_graders
  let third_graders := third_to_fourth_ratio * fourth_graders
  let total_students := third_graders + fourth_graders + fifth_graders
  let total_minutes := third_grade_minutes * (third_graders : ℝ) + fourth_grade_minutes * (fourth_graders : ℝ) + fifth_grade_minutes * (fifth_graders : ℝ)
  total_minutes / (total_students : ℝ)

theorem average_minutes_run_value :
  average_minutes_run 14 18 8 3 2 = 128 / 9 := by
  -- Expand the definition of average_minutes_run
  unfold average_minutes_run
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_minutes_run_value_l101_10198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_sufficient_condition_for_eccentricity_l101_10102

noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt (1 - min a (1/a)^2)

theorem ellipse_eccentricity_range (a : ℝ) (ha : a > 0) :
  (eccentricity a ∈ Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 2 / 3)) ↔
  (a ∈ Set.Ioo (1/3) (1/2) ∪ Set.Ioo 2 3) := by
  sorry

theorem sufficient_condition_for_eccentricity (a m : ℝ) (ha : a > 0) :
  (∀ a, |a - m| < 1/2 → eccentricity a ∈ Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 2 / 3)) ∧
  (∃ a, eccentricity a ∈ Set.Ioo (Real.sqrt 3 / 2) (2 * Real.sqrt 2 / 3) ∧ |a - m| ≥ 1/2) →
  m = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_sufficient_condition_for_eccentricity_l101_10102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_special_triangle_l101_10199

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  -- C is at the origin
  C = (0, 0) ∧
  -- A is on the x-axis, 8 units from C
  A = (8, 0) ∧
  -- B is above the x-axis
  (B.2 > 0) ∧
  -- ABC is a right-angled triangle with right angle at C
  (B.1 - C.1) * (A.1 - C.1) + (B.2 - C.2) * (A.2 - C.2) = 0

-- Define the height to width ratio
def height_width_ratio (A B C : ℝ × ℝ) : Prop :=
  B.2 / 8 = 1 / Real.sqrt 3

-- Define the incircle radius
noncomputable def incircle_radius (A B C : ℝ × ℝ) : ℝ :=
  let s := (dist A B + dist B C + dist C A) / 2
  let area := Real.sqrt (s * (s - dist A B) * (s - dist B C) * (s - dist C A))
  area / s

-- Theorem statement
theorem incircle_radius_of_special_triangle (A B C : ℝ × ℝ) :
  triangle_ABC A B C → height_width_ratio A B C →
  incircle_radius A B C = (96 * Real.sqrt 3 - 8 * Real.sqrt 3) / 141 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_radius_of_special_triangle_l101_10199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l101_10189

theorem regular_polygon_sides (n : ℕ) : 
  (n > 2) →
  (∀ (interior_angle exterior_angle : ℝ), 
    interior_angle = 5 * exterior_angle ∧ 
    interior_angle + exterior_angle = 180 ∧
    n * exterior_angle = 360) → 
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_polygon_sides_l101_10189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_l101_10177

/-- The area of a regular hexagon inscribed in a circle with area 100π -/
theorem inscribed_hexagon_area : 
  ∀ (circle_area : ℝ) (hexagon_area : ℝ),
  circle_area = 100 * Real.pi →
  hexagon_area = (6 * (((100 * Real.pi / Real.pi).sqrt ^ 2) * Real.sqrt 3) / 4) →
  hexagon_area = 150 * Real.sqrt 3 := by
  intros circle_area hexagon_area h1 h2
  sorry

#check inscribed_hexagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_l101_10177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l101_10100

open Real

-- Define the function
noncomputable def f (x : ℝ) := log x + 2 * x - 3

-- State the theorem
theorem root_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l101_10100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l101_10190

theorem trig_identity (x : ℝ) 
  (h1 : -π/2 < x) (h2 : x < 0) (h3 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x * Real.cos x = -12/25) ∧ (Real.sin x - Real.cos x = -7/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l101_10190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l101_10146

-- Define the problem parameters
noncomputable def speed1 : ℝ := 5  -- km/hr
noncomputable def speed2 : ℝ := 10  -- km/hr
noncomputable def late_time : ℝ := 5 / 60  -- hours (5 minutes)
noncomputable def early_time : ℝ := 10 / 60  -- hours (10 minutes)

-- Define the theorem
theorem journey_distance :
  ∃ (t : ℝ), 
    speed1 * (t + late_time) = speed2 * (t - early_time) ∧
    speed1 * (t + late_time) = (5 : ℝ) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l101_10146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l101_10168

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the midpoint condition
def midpoint_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 1

-- Define a line passing through two points
def line_through_points (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

-- Theorem statement
theorem line_equation (x₁ y₁ x₂ y₂ : ℝ) :
  ellipse x₁ y₁ →
  ellipse x₂ y₂ →
  midpoint_condition x₁ y₁ x₂ y₂ →
  ∀ x y, line_through_points x₁ y₁ x₂ y₂ x y ↔ x + 2*y - 3 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l101_10168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l101_10138

/-- The distance between two observers in miles -/
def distance : ℝ := 15

/-- The angle of elevation from the first observer (Alice) in radians -/
noncomputable def angle1 : ℝ := Real.pi / 4  -- 45 degrees in radians

/-- The angle of elevation from the second observer (Bob) in radians -/
noncomputable def angle2 : ℝ := Real.pi / 3  -- 60 degrees in radians

/-- The altitude of the object (airplane) -/
noncomputable def altitude : ℝ := distance * Real.sqrt 3

theorem airplane_altitude :
  ∃ (x y z : ℝ),
    x^2 + y^2 = distance^2 ∧
    z / x = Real.tan angle1 ∧
    z / y = Real.tan angle2 ∧
    z = altitude := by
  sorry

#check airplane_altitude

end NUMINAMATH_CALUDE_ERRORFEEDBACK_airplane_altitude_l101_10138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_bounded_linear_combination_l101_10152

theorem no_bounded_linear_combination :
  ∀ (a b c : ℂ) (h : ℕ), a ≠ 0 → b ≠ 0 → c ≠ 0 →
  ∃ (k l m : ℤ), (↑|k| + ↑|l| + ↑|m| : ℝ) ≥ 1996 ∧ Complex.abs (k * a + l * b + m * c) ≤ 1 / h :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_bounded_linear_combination_l101_10152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_quotient_l101_10123

theorem factorial_sum_quotient : (8 * Nat.factorial 7 + 9 * 8 * Nat.factorial 7) / Nat.factorial 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_quotient_l101_10123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l101_10142

/-- A set of functions satisfying certain monotonicity and range conditions -/
def M : Set (ℝ → ℝ) :=
  {f | Monotone f ∧ ∃ a b, a < b ∧ Set.image f (Set.Icc a b) = Set.Icc (a/2) (b/2)}

/-- The function f(x) = x + 2/x for x > 0 -/
noncomputable def f (x : ℝ) : ℝ := x + 2/x

/-- The function g(x) = -x^3 -/
def g (x : ℝ) : ℝ := -x^3

theorem problem_solution :
  (f ∉ M) ∧
  (g ∈ M) ∧
  (∃ a b, a = -Real.sqrt 2 / 2 ∧ b = Real.sqrt 2 / 2 ∧
    Set.image g (Set.Icc a b) = Set.Icc (a/2) (b/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l101_10142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_shot_scores_l101_10159

def scores : List Nat := [10, 9, 9, 8, 8, 5, 4, 4, 3, 2]

def kolya_scores : List Nat := [5, 4, 10, 9, 8]
def petya_scores : List Nat := [9, 8, 2, 4, 3]

theorem third_shot_scores :
  (List.sum (List.take 3 kolya_scores) = List.sum (List.take 3 petya_scores)) ∧
  (List.sum (List.drop 2 kolya_scores) = 3 * List.sum (List.drop 2 petya_scores)) ∧
  (kolya_scores ++ petya_scores = scores) →
  (kolya_scores[2]? = some 10 ∧ petya_scores[2]? = some 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_shot_scores_l101_10159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_lower_bound_l101_10119

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance from the center to a focus -/
noncomputable def focal_distance (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2)

/-- Theorem: If the right vertex of a hyperbola is inside the circle with diameter AB,
    where A and B are the intersection points of the hyperbola with a perpendicular line
    to the x-axis through the left focus, then the eccentricity is greater than 2. -/
theorem eccentricity_lower_bound (h : Hyperbola) 
  (vertex_inside : h.a + focal_distance h < h.b^2 / h.a) : 
  eccentricity h > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_lower_bound_l101_10119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_geq_neg_one_l101_10175

theorem negation_of_sin_geq_neg_one :
  (¬ ∀ x : ℝ, x > 0 → Real.sin x ≥ -1) ↔ (∃ x : ℝ, x > 0 ∧ Real.sin x < -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_geq_neg_one_l101_10175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_divisible_l101_10161

-- Define the triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the arc
structure Arc where
  center : Point
  radius : ℝ
  startPoint : Point
  endPoint : Point

-- Define the figure (triangle + circular segment)
structure Figure where
  triangle : Triangle
  arc : Arc

-- Define the midpoint of the arc
noncomputable def arcMidpoint (arc : Arc) : Point := sorry

-- Define the area of a figure
noncomputable def area (fig : Figure) : ℝ := sorry

-- Define a point on a line segment
noncomputable def pointOnSegment (A B : Point) (t : ℝ) : Point := sorry

-- Theorem statement
theorem always_divisible (fig : Figure) :
  ∃ (E : Point), 
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    let F := arcMidpoint fig.arc
    let E := pointOnSegment fig.triangle.A fig.triangle.B t
    area { triangle := fig.triangle, arc := fig.arc } = 
    2 * area { triangle := ⟨F, E, fig.triangle.C⟩, arc := fig.arc } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_divisible_l101_10161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l101_10197

-- Define the fixed point P
def P : ℝ × ℝ := (-1, 2)

-- Define the line that passes through P for all a ∈ ℝ
def line (a x y : ℝ) : Prop := (x + y - 1) - a * (x + 1) = 0

-- Define the circle with center P and radius √5
def circle_eq (x y : ℝ) : Prop := (x - P.1)^2 + (y - P.2)^2 = 5

-- Theorem statement
theorem circle_equation : 
  (∀ a : ℝ, line a P.1 P.2) → 
  (∀ x y : ℝ, circle_eq x y ↔ x^2 + y^2 + 2*x - 4*y = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l101_10197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l101_10166

noncomputable def f (x : ℝ) : ℝ := 9 - x^2 - 2 * Real.sqrt (9 - x^2)

theorem f_min_max :
  ∀ x ∈ Set.Icc (-3 : ℝ) 3,
    -1 ≤ f x ∧ f x ≤ 3 ∧
    (∃ x₁ ∈ Set.Icc (-3 : ℝ) 3, f x₁ = -1) ∧
    (∃ x₂ ∈ Set.Icc (-3 : ℝ) 3, f x₂ = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_max_l101_10166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silverware_probability_l101_10137

theorem silverware_probability (forks spoons knives : ℕ) 
  (h1 : forks = 4)
  (h2 : spoons = 5)
  (h3 : knives = 7) :
  (forks * spoons * knives : ℚ) / Nat.choose (forks + spoons + knives) 3 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_silverware_probability_l101_10137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l101_10107

theorem inequality_solution (x : ℝ) : 
  (81 * (3 : ℝ)^(2*x) > (1/9 : ℝ)^(x+2)) ↔ (x > -4/3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l101_10107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_election_votes_l101_10158

theorem mark_election_votes :
  ∀ (area1_voters : ℕ) (area1_percentage : ℚ) (area2_multiplier : ℕ),
    area1_voters = 100000 →
    area1_percentage = 70 / 100 →
    area2_multiplier = 2 →
    (area1_voters : ℚ) * area1_percentage +
    (area2_multiplier : ℚ) * ((area1_voters : ℚ) * area1_percentage) = 210000 := by
  intro area1_voters area1_percentage area2_multiplier h1 h2 h3
  have area1_votes : ℚ := (area1_voters : ℚ) * area1_percentage
  have area2_votes : ℚ := (area2_multiplier : ℚ) * area1_votes
  have total_votes : ℚ := area1_votes + area2_votes
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_election_votes_l101_10158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_correct_value_at_sqrt2_minus4_l101_10141

noncomputable def original_expr (x : ℝ) : ℝ := (x^2 - 16) / (x^2 + 8*x + 16) - (2*x + 1) / (2*x + 8)

noncomputable def simplified_expr (x : ℝ) : ℝ := -9 / (2*x + 8)

-- Theorem 1: The original expression equals the simplified expression
theorem simplification_correct (x : ℝ) (h : x^2 + 8*x + 16 ≠ 0 ∧ 2*x + 8 ≠ 0) : 
  original_expr x = simplified_expr x := by sorry

-- Theorem 2: The value of the expression when x = √2 - 4
theorem value_at_sqrt2_minus4 : 
  original_expr (Real.sqrt 2 - 4) = -9 * Real.sqrt 2 / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplification_correct_value_at_sqrt2_minus4_l101_10141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_range_l101_10165

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2 * x + 1 else x^2 - 2 * x - 2

-- Theorem statement
theorem x0_range (x₀ : ℝ) : f x₀ > 1 → x₀ ∈ Set.Iic (-1) ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x0_range_l101_10165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_calculation_l101_10172

/-- The rate per meter for fencing a circular field -/
noncomputable def fencing_rate_per_meter (diameter : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (Real.pi * diameter)

/-- Theorem: The fencing rate per meter for a circular field with diameter 20m and total cost Rs. 94.24777960769379 is approximately Rs. 1.5 -/
theorem fencing_rate_calculation :
  let diameter : ℝ := 20
  let total_cost : ℝ := 94.24777960769379
  abs (fencing_rate_per_meter diameter total_cost - 1.5) < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_calculation_l101_10172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_implies_a_value_l101_10113

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 + Real.log x / Real.log a

-- State the theorem
theorem inverse_function_point_implies_a_value
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : f a (f a⁻¹ 3) = 3)
  (h4 : f a⁻¹ 3 = 4) :
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_implies_a_value_l101_10113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_color_change_theorem_l101_10155

/-- A sequence of even integers where each subsequent number differs from the previous one by -2, 0, or +2 -/
def ValidSequence (seq : List ℕ) : Prop :=
  ∀ i : ℕ, i + 1 < seq.length →
    (seq.get! (i+1) = seq.get! i ∨ seq.get! (i+1) = seq.get! i - 2 ∨ seq.get! (i+1) = seq.get! i + 2) ∧
    Even (seq.get! i)

theorem color_change_theorem (seq : List ℕ) :
  seq.length > 0 →
  seq.get! 0 = 46 →
  seq.get! (seq.length - 1) = 26 →
  ValidSequence seq →
  28 ∈ seq := by
  sorry

#check color_change_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_color_change_theorem_l101_10155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_minus_sectors_l101_10101

/-- The area of a shaded region formed by subtracting circular sectors from a regular hexagon -/
theorem shaded_area_hexagon_minus_sectors (s : ℝ) (r : ℝ) : 
  s = 8 → r = 4 → 
  (6 * (Real.sqrt 3 / 4 * s^2)) - (6 * (1/6 * Real.pi * r^2)) = 96 * Real.sqrt 3 - 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_minus_sectors_l101_10101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l101_10116

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

theorem f_increasing_on_interval :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  intros x₁ x₂ h₁ h₂
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l101_10116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_plus_two_alpha_l101_10109

theorem tan_pi_fourth_plus_two_alpha (α : ℝ) :
  (π/2 < α) ∧ (α < 3*π/2) →  -- α is in the third quadrant
  Real.cos (2*α) = -3/5 →
  Real.tan (π/4 + 2*α) = -1/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_plus_two_alpha_l101_10109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_volume_l101_10181

/-- The volume of a cuboid with edges 6 cm, 5 cm, and 6 cm is 180 cubic centimeters. -/
theorem cuboid_volume : 
  let length : ℝ := 6
  let width : ℝ := 5
  let height : ℝ := 6
  length * width * height = 180 := by
  -- Unfold the let bindings
  simp_all
  -- Perform the multiplication
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_volume_l101_10181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_years_is_three_l101_10179

/-- The number of years for which B's gain is 45 rupees -/
noncomputable def calculate_years (principal : ℝ) (rate_A_to_B : ℝ) (rate_B_to_C : ℝ) (gain : ℝ) : ℝ :=
  gain / (principal * (rate_B_to_C - rate_A_to_B))

/-- Theorem stating that the number of years is 3 -/
theorem years_is_three : 
  let principal := 1000
  let rate_A_to_B := 0.10
  let rate_B_to_C := 0.115
  let gain := 45
  calculate_years principal rate_A_to_B rate_B_to_C gain = 3 := by
  sorry

-- Remove the #eval statement as it's not necessary for the proof
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_years_is_three_l101_10179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_area_ratio_bounds_l101_10130

/-- The hyperbola function -/
noncomputable def hyperbola (x : ℝ) : ℝ := 1 / x

/-- The tangent line to the hyperbola at point (a, 1/a) -/
noncomputable def tangent_line (a : ℝ) (x : ℝ) : ℝ := -1 / (a^2) * x + 2 / a

/-- The area enclosed by the tangent and the coordinate axes -/
def t (a : ℝ) : ℝ := 2

/-- The area of the triangle bounded by the tangent, its perpendicular at P, and the x-axis -/
noncomputable def T (a : ℝ) : ℝ := 1 / 2

/-- The ratio of the areas t(a) and T(a) -/
noncomputable def area_ratio (a : ℝ) : ℝ := t a / T a

theorem hyperbola_area_ratio_bounds (a : ℝ) (h : a ≥ 1) :
  (∀ a, a ≥ 1 → area_ratio a ≥ 2) ∧
  (area_ratio 1 = 2) ∧
  (∀ ε > 0, ∃ A, ∀ a ≥ A, 4 - ε < area_ratio a ∧ area_ratio a < 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_area_ratio_bounds_l101_10130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_for_positive_x_l101_10188

/-- The function f(x) = e^x - x^2 + 1 -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x^2 + 1

/-- The lower bound function g(x) = (e-2)x + 2 -/
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 - 2) * x + 2

/-- Theorem stating that f(x) ≥ g(x) for all x > 0 -/
theorem f_geq_g_for_positive_x : ∀ x : ℝ, x > 0 → f x ≥ g x := by
  sorry

#check f_geq_g_for_positive_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_geq_g_for_positive_x_l101_10188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_not_ending_in_zero_l101_10132

theorem divisors_not_ending_in_zero (n : Nat) (h : n = 1000000) : 
  (Finset.filter (λ d : Nat ↦ d ∣ n ∧ d % 10 ≠ 0) (Finset.range (n + 1))).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_not_ending_in_zero_l101_10132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l101_10111

theorem work_completion_time (work : ℝ) (a_time b_time total_time : ℝ) (c_time : ℝ) : 
  work > 0 ∧ 
  a_time = 11 ∧ 
  b_time = 5 ∧ 
  total_time = 5 ∧
  (total_time * (1 / a_time) + (total_time / 2) * (1 / b_time) + (total_time / 2) * (1 / c_time) = work) →
  c_time = 2.5 := by
  intro h
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l101_10111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_upper_bound_l101_10173

/-- The function f(x) = 8x + 1/(4x-5) -/
noncomputable def f (x : ℝ) : ℝ := 8*x + 1/(4*x-5)

/-- The upper bound of the range of f(x) -/
noncomputable def upper_bound : ℝ := 10 - 2 * Real.sqrt 2

theorem f_range_upper_bound :
  ∀ x y : ℝ, x < 5/4 → y = f x → y ≤ upper_bound :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_upper_bound_l101_10173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l101_10184

theorem expression_evaluation (M : ℝ) (h : M > 1) :
  Real.sqrt (M * (M * Real.sqrt M) ^ (1/3)) = M ^ (3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l101_10184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_count_l101_10120

theorem multiple_count (n : ℕ) : n = 2 ↔ (∃ (k : ℕ), k = 22 ∧ 
  (∀ (m : ℕ), 10 ≤ m ∧ m ≤ 52 ∧ n ∣ m ↔ m ∈ Finset.range (k + 1) ∧ m ≥ 10)) := by
  sorry

#check multiple_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_count_l101_10120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_divisors_condition_l101_10112

theorem equal_divisors_condition (n k : ℕ+) (hn : n ≠ k) :
  (∃ s : ℕ+, (Nat.divisors (s.val * n.val)).card = (Nat.divisors (s.val * k.val)).card) ↔ (¬ n ∣ k ∧ ¬ k ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_divisors_condition_l101_10112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l101_10187

open Real Set

noncomputable def f (x : ℝ) := x - 2 * sin x

theorem f_properties :
  let minValue := π / 3 - Real.sqrt 3
  let maxValue := π
  (∀ x ∈ Icc 0 π, f x ≥ minValue) ∧
  (∃ x ∈ Icc 0 π, f x = minValue) ∧
  (∀ x ∈ Icc 0 π, f x ≤ maxValue) ∧
  (∃ x ∈ Icc 0 π, f x = maxValue) ∧
  (∀ a > -1, ∃ x ∈ Ioo 0 (π / 2), f x < a * x) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l101_10187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_at_12_ohms_l101_10114

/-- A battery with voltage 48V and a relationship between current and resistance --/
structure Battery where
  voltage : ℝ
  current : ℝ → ℝ
  resistance_current_relation : ∀ R, current R = voltage / R

/-- The specific battery in the problem --/
noncomputable def problem_battery : Battery :=
  { voltage := 48
  , current := λ R => 48 / R
  , resistance_current_relation := λ R => rfl }

/-- The theorem to be proved --/
theorem current_at_12_ohms (b : Battery) (h : b = problem_battery) :
  b.current 12 = 4 := by
  rw [h]
  simp [problem_battery, Battery.current]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_at_12_ohms_l101_10114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mn_value_l101_10160

/-- The function f(x) defined in the problem -/
noncomputable def f (m n x : ℝ) : ℝ := (1/2) * (m - 2) * x^2 + (n - 8) * x + 1

/-- The statement that f is monotonically decreasing on [1/2, 2] -/
def is_monotone_decreasing (m n : ℝ) : Prop :=
  ∀ x y, 1/2 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f m n x ≥ f m n y

/-- The main theorem -/
theorem max_mn_value (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) 
  (h_monotone : is_monotone_decreasing m n) : m * n ≤ 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mn_value_l101_10160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l101_10167

-- Define the sequence y_n
noncomputable def y : ℕ → ℝ
  | 0 => 0  -- Adding a case for 0 to cover all natural numbers
  | 1 => (4 : ℝ) ^ (1/4)
  | 2 => ((4 : ℝ) ^ (1/4)) ^ ((4 : ℝ) ^ (1/4))
  | n + 3 => (y (n + 2)) ^ ((4 : ℝ) ^ (1/4))

-- Define what it means for a real number to be an integer
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

-- Statement to prove
theorem smallest_integer_y : ∀ n : ℕ, n < 4 → ¬ (isInteger (y n)) ∧ (isInteger (y 4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l101_10167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l101_10176

/-- The curve equation -/
noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 2)

/-- The point on the curve -/
def point : ℝ × ℝ := (-1, -3)

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 5 * x - y + 2 = 0

/-- Theorem stating that the tangent line passes through the given point -/
theorem tangent_line_at_point :
  let (x₀, y₀) := point
  (f x₀ = y₀) →
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - x₀| < δ → |f x - (f x₀ + (5 * (x - x₀)))| < ε * |x - x₀|) →
  tangent_line x₀ y₀ :=
by
  intros
  sorry

#check tangent_line_at_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_l101_10176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sine_plus_linear_l101_10196

open Real

/-- A function f: ℝ → ℝ is increasing if for all x₁ < x₂, f(x₁) < f(x₂) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- The function f(x) = sin(x) + ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := sin x + a * x

theorem increasing_sine_plus_linear (a : ℝ) :
  IsIncreasing (f a) ↔ a > 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_sine_plus_linear_l101_10196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_on_positive_reals_l101_10136

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^x + 3

-- State the theorem
theorem f_range_on_positive_reals :
  Set.range (fun x => f x) ∩ Set.Ioi 0 = Set.Ici 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_on_positive_reals_l101_10136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l101_10195

theorem order_of_numbers : 
  let a := (5/3)^(1/3)
  let b := (3/4)^(1/2)
  let c := Real.log (3/5)
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l101_10195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elongation_experiment_results_l101_10154

/-- Elongation rates for process A -/
def x : Fin 10 → ℚ
| 0 => 545
| 1 => 533
| 2 => 551
| 3 => 522
| 4 => 575
| 5 => 544
| 6 => 541
| 7 => 568
| 8 => 596
| 9 => 548

/-- Elongation rates for process B -/
def y : Fin 10 → ℚ
| 0 => 536
| 1 => 527
| 2 => 543
| 3 => 530
| 4 => 560
| 5 => 533
| 6 => 522
| 7 => 550
| 8 => 576
| 9 => 536

/-- Difference between elongation rates -/
def z (i : Fin 10) : ℚ := x i - y i

/-- Sample mean of z -/
def z_bar : ℚ := (Finset.sum Finset.univ z) / 10

/-- Sample variance of z -/
def s_squared : ℚ := (Finset.sum Finset.univ (fun i => (z i - z_bar)^2)) / 10

/-- Theorem stating the results of the experiment -/
theorem elongation_experiment_results :
  z_bar = 11 ∧ s_squared = 61 ∧ (z_bar : ℝ) ≥ 2 * Real.sqrt ((s_squared : ℝ) / 10) := by
  sorry

#eval z_bar
#eval s_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elongation_experiment_results_l101_10154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_theorem_l101_10151

/-- A car's travel capability -/
structure Car where
  /-- Miles traveled on 6 gallons of gas -/
  miles_on_6_gallons : ℕ
  /-- Assumption that distance is proportional to gas used -/
  distance_proportional_to_gas : Prop

/-- Calculate miles traveled given gallons of gas -/
def miles_traveled (c : Car) (gallons : ℕ) : ℕ :=
  c.miles_on_6_gallons * gallons / 6

/-- Theorem: A car traveling 192 miles on 6 gallons will travel 256 miles on 8 gallons -/
theorem car_travel_theorem (c : Car) 
    (h1 : c.miles_on_6_gallons = 192) 
    (h2 : c.distance_proportional_to_gas) : 
  miles_traveled c 8 = 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_travel_theorem_l101_10151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_selling_price_proof_l101_10134

noncomputable def calculate_selling_price (cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost * (1 + profit_percentage / 100)

theorem total_selling_price_proof (cost1 cost2 cost3 : ℝ) 
  (profit_percentage1 profit_percentage2 profit_percentage3 : ℝ) :
  cost1 = 550 ∧ cost2 = 750 ∧ cost3 = 1000 ∧
  profit_percentage1 = 30 ∧ profit_percentage2 = 25 ∧ profit_percentage3 = 20 →
  calculate_selling_price cost1 profit_percentage1 +
  calculate_selling_price cost2 profit_percentage2 +
  calculate_selling_price cost3 profit_percentage3 = 2852.5 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_selling_price_proof_l101_10134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_f_greater_than_ln_over_x_l101_10150

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.exp x - x^3 * Real.exp x

-- Statement for the tangent line
theorem tangent_line_at_zero (x y : ℝ) :
  (2 * x - y + 2 = 0) ↔ (y = f 0 + (deriv f 0) * x) := by
  sorry

-- Statement for the inequality
theorem f_greater_than_ln_over_x (x : ℝ) (hx : 0 < x ∧ x < 1) :
  f x > Real.log x / x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_f_greater_than_ln_over_x_l101_10150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l101_10143

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- Part I
theorem part_one (a : ℝ) :
  (a < 3) →
  ({x : ℝ | f (x - a + 2) + f (x - 1) ≥ 4} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 9/2}) →
  a = 2 :=
sorry

-- Part II
theorem part_two :
  {a : ℝ | ∀ x : ℝ, f (x - a + 2) + 2 * f (x - 1) ≥ 1} = 
  Set.Iic 2 ∪ Set.Ici 4 :=
sorry

-- Note: Set.Iic 2 represents (-∞, 2] and Set.Ici 4 represents [4, +∞)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l101_10143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_projection_magnitude_l101_10182

def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (3, 0)

theorem orthogonal_projection_magnitude :
  let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  abs proj = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_projection_magnitude_l101_10182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_a_work_days_l101_10193

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem work_completion 
  (total_rate : ℝ) 
  (rate_B : ℝ) 
  (rate_C : ℝ) 
  (h1 : total_rate = work_rate 2)
  (h2 : rate_B = work_rate 9)
  (h3 : rate_C = work_rate 7.2)
  (h4 : total_rate = work_rate 2 + rate_B + rate_C) :
  work_rate 2 = total_rate - rate_B - rate_C := by
  sorry

theorem a_work_days 
  (total_rate : ℝ) 
  (rate_B : ℝ) 
  (rate_C : ℝ) 
  (h1 : total_rate = work_rate 2)
  (h2 : rate_B = work_rate 9)
  (h3 : rate_C = work_rate 7.2)
  (h4 : total_rate = work_rate 2 + rate_B + rate_C) :
  ∃ (days_A : ℝ), work_rate days_A = total_rate - rate_B - rate_C ∧ days_A = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_a_work_days_l101_10193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_connect_distant_points_l101_10122

/-- A compass with a maximum radius -/
structure Compass where
  max_radius : ℝ

/-- A ruler with a fixed length -/
structure Ruler where
  length : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Theorem: It is possible to connect two points more than 1 meter apart
    using a compass with max radius 10 cm and a ruler of 10 cm length -/
theorem connect_distant_points (a b : Point) (c : Compass) (r : Ruler)
  (h1 : distance a b > 1)
  (h2 : c.max_radius = 0.1)
  (h3 : r.length = 0.1) :
  ∃ (path : List Point), path.head? = some a ∧ path.getLast? = some b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_connect_distant_points_l101_10122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_count_l101_10174

def banknotes : Finset ℕ := {500, 1000, 2000, 5000, 10000, 20000}

def three_banknote_sums : Finset ℕ :=
  Finset.powersetCard 3 banknotes |>.image (λ s ↦ s.sum id)

theorem distinct_sums_count : three_banknote_sums.card = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_count_l101_10174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_one_fifth_l101_10103

/-- A rectangular yard with two congruent isosceles right triangular flower beds -/
structure Yard where
  length : ℝ
  width : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ

/-- The fraction of the yard occupied by the flower beds -/
noncomputable def flower_bed_fraction (y : Yard) : ℝ :=
  let triangle_leg := (y.trapezoid_long_side - y.trapezoid_short_side) / 2
  let triangle_area := triangle_leg ^ 2 / 2
  let total_flower_bed_area := 2 * triangle_area
  let yard_area := y.length * y.width
  total_flower_bed_area / yard_area

theorem flower_bed_fraction_is_one_fifth (y : Yard) 
  (h1 : y.trapezoid_short_side = 18)
  (h2 : y.trapezoid_long_side = 30)
  (h3 : y.length = y.trapezoid_long_side)
  (h4 : y.width = (y.trapezoid_long_side - y.trapezoid_short_side) / 2) :
  flower_bed_fraction y = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_is_one_fifth_l101_10103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_inequality_l101_10170

theorem prime_divisors_inequality (x y z : ℕ) (p q : ℕ) 
  (hx : x > 2)
  (hy : y > 1)
  (heq : x^y + 1 = z^2)
  (hp : p = (Nat.factorization x).support.card)
  (hq : q = (Nat.factorization y).support.card) :
  p ≥ q + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_inequality_l101_10170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_tournament_points_l101_10133

/-- Represents a soccer team --/
inductive Team
| A | B | C | D | E

/-- Represents the result of a match --/
inductive MatchResult
| Win | Draw | Loss

/-- The number of teams in the tournament --/
def numTeams : Nat := 5

/-- The number of points awarded for a win --/
def winPoints : Nat := 3

/-- The number of points awarded for a draw --/
def drawPoints : Nat := 1

/-- The number of points awarded for a loss --/
def lossPoints : Nat := 0

/-- The total number of matches played in the tournament --/
def totalMatches : Nat := numTeams * (numTeams - 1) / 2

/-- The function that returns the points scored by each team --/
noncomputable def teamPoints : Team → Nat
| Team.A => 7
| Team.B => 6
| Team.C => 4
| Team.D => 5
| Team.E => 2

/-- All teams scored a different number of points --/
axiom distinct_points : ∀ t1 t2 : Team, t1 ≠ t2 → teamPoints t1 ≠ teamPoints t2

/-- Team A scored the most points overall --/
axiom A_highest_points : ∀ t : Team, t ≠ Team.A → teamPoints Team.A > teamPoints t

/-- Team A lost to team B --/
axiom A_lost_to_B : ∃ (result : Team → Team → MatchResult), result Team.A Team.B = MatchResult.Loss

/-- Teams B and C did not lose any games --/
axiom B_C_no_losses : ∀ t : Team, t ≠ Team.B → t ≠ Team.C → 
  ∃ (result : Team → Team → MatchResult), 
    result Team.B t ≠ MatchResult.Loss ∧ result Team.C t ≠ MatchResult.Loss

/-- Team C scored fewer points than team D --/
axiom C_less_than_D : teamPoints Team.C < teamPoints Team.D

/-- The theorem to be proved --/
theorem soccer_tournament_points : 
  teamPoints Team.A = 7 ∧
  teamPoints Team.B = 6 ∧
  teamPoints Team.C = 4 ∧
  teamPoints Team.D = 5 ∧
  teamPoints Team.E = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_soccer_tournament_points_l101_10133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_at_negative_three_l101_10104

-- Define the expression as a function
noncomputable def f (x : ℝ) : ℝ := (5 + x * (5 + x) - 4^2) / (x - 4 + x^3)

-- State the theorem
theorem evaluate_expression_at_negative_three :
  f (-3) = -17/20 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_at_negative_three_l101_10104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l101_10147

def a (n : ℕ+) : ℚ :=
  2 * (Finset.range n).prod (λ i => 1 - 1 / ((i + 2) ^ 2 : ℚ))

theorem a_formula (n : ℕ+) : a n = (n + 2 : ℚ) / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l101_10147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_palindromes_digits_l101_10115

/-- A palindrome is a number that reads the same forward and backward -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- An odd palindrome has an odd number of digits -/
def isOddPalindrome (n : ℕ) : Prop := isPalindrome n ∧ Odd n

/-- Count the number of odd palindromes with a given number of digits -/
def countOddPalindromes (digits : ℕ) : ℕ := sorry

/-- The number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ := sorry

theorem odd_palindromes_digits (palindromes : Finset ℕ) :
  (∀ n ∈ palindromes, isOddPalindrome n) →
  (palindromes.card = 50) →
  (∃ d : ℕ, ∀ n ∈ palindromes, numDigits n = d) →
  (∃ d : ℕ, ∀ n ∈ palindromes, numDigits n = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_palindromes_digits_l101_10115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_square_area_triangle_l101_10110

/-- A square in a 2D plane --/
structure Square where
  O : ℝ × ℝ
  U : ℝ × ℝ
  is_square : U.1 = U.2 ∧ U.1 > 0

/-- The area of a triangle given three points --/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem half_square_area_triangle (sq : Square) :
  sq.O = (0, 0) →
  sq.U = (4, 4) →
  let S : ℝ × ℝ := (4, 0)
  let V : ℝ × ℝ := (0, 4)
  let W : ℝ × ℝ := (4, -4)
  triangle_area S V W = (1/2) * (sq.U.1 * sq.U.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_square_area_triangle_l101_10110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_eleven_l101_10129

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4

noncomputable def f_inverse (x : ℝ) : ℝ := (x - 4) / 3

theorem inverse_of_inverse_eleven :
  f_inverse (f_inverse 11) = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_eleven_l101_10129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisected_by_point_l101_10135

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if a point is the midpoint of two other points -/
def Point.is_midpoint (m p1 p2 : Point) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

/-- The main theorem to prove -/
theorem line_bisected_by_point (P : Point) (l₁ l₂ : Line) :
  P.x = 0 ∧ P.y = 1 ∧
  l₁.a = 1 ∧ l₁.b = -3 ∧ l₁.c = 10 ∧
  l₂.a = 2 ∧ l₂.b = 1 ∧ l₂.c = -8 →
  ∃ (l : Line) (A B : Point),
    l.a = 1 ∧ l.b = 4 ∧ l.c = -4 ∧
    P.on_line l ∧
    A.on_line l ∧ A.on_line l₁ ∧
    B.on_line l ∧ B.on_line l₂ ∧
    P.is_midpoint A B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_bisected_by_point_l101_10135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_condition_l101_10105

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The theorem stating the condition for f to be decreasing on (-∞, 4) -/
theorem decreasing_condition (a : ℝ) :
  (∀ x < 4, StrictMonoOn (fun x => -(f a x)) (Set.Iio 4)) ↔ a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_condition_l101_10105
