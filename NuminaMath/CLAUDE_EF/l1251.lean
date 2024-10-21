import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_margin_proof_l1251_125100

theorem election_margin_proof (total_votes : ℕ) (changed_votes : ℕ) (new_margin_percent : ℚ) :
  total_votes = 20000 →
  changed_votes = 2000 →
  new_margin_percent = 10 / 100 →
  ∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    (loser_votes + changed_votes) - (winner_votes - changed_votes) = (new_margin_percent * total_votes).floor →
    (winner_votes - loser_votes : ℚ) / total_votes = 10 / 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_margin_proof_l1251_125100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l1251_125178

theorem smallest_positive_z (x z : ℝ) : 
  Real.cos x = 0 → 
  Real.cos (x + z) = -1/2 → 
  z > 0 → 
  (∀ w, w > 0 → Real.cos x = 0 → Real.cos (x + w) = -1/2 → z ≤ w) → 
  z = π/6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l1251_125178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_darrys_ladder_climbs_l1251_125138

theorem darrys_ladder_climbs : ∀ (full_ladder_climbs : ℕ), 
  (11 : ℕ) * full_ladder_climbs + (6 : ℕ) * (7 : ℕ) = (152 : ℕ) →
  full_ladder_climbs = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_darrys_ladder_climbs_l1251_125138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_on_ellipse_l1251_125145

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse (x^2/a^2) + (y^2/b^2) = 1 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Checks if a point lies on the ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Represents a triangle defined by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.p1.x + t.p2.x + t.p3.x) / 3,
    y := (t.p1.y + t.p2.y + t.p3.y) / 3 }

/-- Calculates the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  (1/2) * abs (t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y))

/-- Main theorem: Area of triangle with vertices on ellipse and centroid at origin -/
theorem triangle_area_on_ellipse (e : Ellipse) (t : Triangle) 
  (h1 : onEllipse t.p1 e) (h2 : onEllipse t.p2 e) (h3 : onEllipse t.p3 e)
  (h_centroid : centroid t = { x := 0, y := 0 }) :
  area t = (3 * Real.sqrt 3 / 4) * e.a * e.b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_on_ellipse_l1251_125145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problems_solved_l1251_125103

theorem problems_solved (start finish : ℕ) (h1 : start = 80) (h2 : finish = 125) :
  finish - start + 1 = 46 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problems_solved_l1251_125103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangements_l1251_125143

inductive Marble
  | Aggie
  | Bumblebee
  | Steelie
  | Tiger
  | Crystal

def is_valid_arrangement (arr : Fin 5 → Marble) : Prop :=
  ∀ i : Fin 4, 
    (arr i = Marble.Steelie ∧ arr (i + 1) = Marble.Tiger) → False ∧
    (arr i = Marble.Tiger ∧ arr (i + 1) = Marble.Steelie) → False ∧
    (arr i = Marble.Aggie ∧ arr (i + 1) = Marble.Bumblebee) → False ∧
    (arr i = Marble.Bumblebee ∧ arr (i + 1) = Marble.Aggie) → False

theorem no_valid_arrangements : 
  ¬∃ (arr : Fin 5 → Marble), Function.Injective arr ∧ is_valid_arrangement arr :=
by sorry

#check no_valid_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_arrangements_l1251_125143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_honor_students_count_l1251_125120

/-- Represents a class of students -/
structure StudentClass where
  total_students : ℕ
  girls : ℕ
  boys : ℕ
  honor_girls : ℕ
  honor_boys : ℕ
  h_total : total_students < 30
  h_sum : total_students = girls + boys
  h_girl_prob : (honor_girls : ℚ) / girls = 3 / 13
  h_boy_prob : (honor_boys : ℚ) / boys = 4 / 11

/-- The total number of honor students in the class is 7 -/
theorem honor_students_count (c : StudentClass) : c.honor_girls + c.honor_boys = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_honor_students_count_l1251_125120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_system_solution_l1251_125152

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the system of equations
def equation_system (x y a : ℝ) : Prop :=
  (log y (log y x) = log x (log x y)) ∧
  ((log a x)^2 + (log a y)^2 = 8)

-- State the theorem
theorem log_system_solution (a : ℝ) (h_a : a > 0) :
  ∃ x y : ℝ, equation_system x y a ∧
  ((x = a^2 ∧ y = a^2) ∨ (x = a^((-2):ℤ) ∧ y = a^((-2):ℤ))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_system_solution_l1251_125152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_distance_between_circles_l1251_125102

noncomputable def circle1_center : ℝ × ℝ := (3, 3)
noncomputable def circle2_center : ℝ × ℝ := (20, 12)

noncomputable def distance_between_centers (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def radius (c : ℝ × ℝ) : ℝ := c.2

theorem closest_distance_between_circles :
  distance_between_centers circle1_center circle2_center - (radius circle1_center + radius circle2_center) =
  Real.sqrt 370 - 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_distance_between_circles_l1251_125102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l1251_125168

/-- The function f(x) = a(x+a)(x-a+3) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x + a) * (x - a + 3)

/-- The function g(x) = 2^(x+2) - 1 -/
noncomputable def g (x : ℝ) : ℝ := (2 : ℝ)^(x + 2) - 1

/-- Theorem: If for any x ∈ ℝ, at least one of f(x) > 0 and g(x) > 0 holds, 
    then a ∈ (1, 2) -/
theorem function_range (a : ℝ) : 
  (∀ x : ℝ, f a x > 0 ∨ g x > 0) → 1 < a ∧ a < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_l1251_125168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_representation_l1251_125124

/-- Represents the ship's path from A to C via B --/
noncomputable def ship_path (r : ℝ) (t : ℝ) : ℝ :=
  if t ≤ 1 then r else r + (t - 1)

/-- Theorem representing the ship's distance from Island X --/
theorem ship_distance_representation (r : ℝ) (h : r > 0) :
  (∀ t ∈ Set.Icc 0 1, ship_path r t = r) ∧
  (∀ t₁ t₂, 1 < t₁ → t₁ < t₂ → ship_path r t₁ < ship_path r t₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_representation_l1251_125124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_odd_l1251_125154

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x * (x + 1)
  else -Real.exp (-x) * (-x + 1)

-- State the theorem
theorem f_properties :
  (∀ x₁ x₂ : ℝ, |f x₁ - f x₂| < 2) ∧
  (∀ x : ℝ, f x > 0 ↔ (-1 < x ∧ x < 0) ∨ x > 1) := by
  sorry

-- Prove that f is an odd function
theorem f_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_odd_l1251_125154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l1251_125106

/-- The length of a platform crossed by a train -/
noncomputable def platform_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Theorem stating the length of the platform -/
theorem platform_length_calculation :
  let train_length : ℝ := 360
  let train_speed_kmh : ℝ := 55
  let crossing_time : ℝ := 57.59539236861051
  ∃ ε > 0, |platform_length train_length train_speed_kmh crossing_time - 520| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l1251_125106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bulb_power_consumption_l1251_125176

/-- Represents the daily power consumption of a single bulb in watts -/
def daily_bulb_consumption : ℝ → Prop := λ _ ↦ True

/-- The number of bulbs in Allyn's house -/
def num_bulbs : ℕ := 40

/-- The cost of electricity in dollars per watt -/
def cost_per_watt : ℝ := 0.20

/-- The number of days in June -/
def days_in_june : ℕ := 30

/-- Allyn's total monthly expenses on electricity in June in dollars -/
def monthly_expense : ℝ := 14400

theorem bulb_power_consumption :
  ∃ (w : ℝ), daily_bulb_consumption w ∧ 
  (w * (num_bulbs : ℝ) * cost_per_watt * (days_in_june : ℝ) = monthly_expense) ∧
  w = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bulb_power_consumption_l1251_125176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ln_less_than_zero_l1251_125121

/-- Given a function f(x) = x^3 - ax where a ∈ ℝ and 2 is a root of f(x),
    this theorem states that for any x₀ randomly chosen from the interval (0, a),
    the probability that ln(x₀) < 0 is 1/4. -/
theorem probability_ln_less_than_zero (a : ℝ) (f : ℝ → ℝ) 
    (h1 : f = λ x ↦ x^3 - a*x) (h2 : f 2 = 0) : 
    ∃ (P : Set ℝ → ℝ), P { x | x ∈ Set.Ioo 0 a ∧ Real.log x < 0 } = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_ln_less_than_zero_l1251_125121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_empty_proper_subsets_count_l1251_125179

def S : Finset ℕ := {2, 3, 4}

theorem non_empty_proper_subsets_count : 
  (S.powerset.filter (λ A => A.Nonempty ∧ A ⊂ S)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_empty_proper_subsets_count_l1251_125179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_l1251_125151

/-- The y-coordinate of the third vertex of an equilateral triangle -/
noncomputable def third_vertex_y_coord : ℝ := 5 + 4 * Real.sqrt 3

/-- Proves that the y-coordinate of the third vertex of an equilateral triangle
    with two vertices at (0,5) and (8,5), and the third vertex in the first quadrant,
    is 5 + 4√3 -/
theorem equilateral_triangle_third_vertex :
  let v1 : ℝ × ℝ := (0, 5)
  let v2 : ℝ × ℝ := (8, 5)
  let side_length : ℝ := v2.1 - v1.1
  ∀ v3 : ℝ × ℝ,
    v3.1 > 0 →  -- third vertex is in the first quadrant
    v3.2 > 5 →  -- third vertex is above the base
    (v3.1 - v1.1)^2 + (v3.2 - v1.2)^2 = side_length^2 →  -- distance from v3 to v1 is side_length
    (v3.1 - v2.1)^2 + (v3.2 - v2.2)^2 = side_length^2 →  -- distance from v3 to v2 is side_length
    v3.2 = third_vertex_y_coord := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_l1251_125151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_amplitude_l1251_125112

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.cos (2 * x)

-- Theorem for the smallest positive period and amplitude
theorem f_period_and_amplitude :
  (∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧
    (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q)) ∧
  (∃ (A : ℝ), A > 0 ∧ (∀ x, f x ≤ A) ∧ (∃ x₀, f x₀ = A) ∧
    (∀ B, B > 0 → (∀ x, f x ≤ B) → A ≤ B)) ∧
  (let p := Real.pi; let A := 1;
    p > 0 ∧ (∀ x, f (x + p) = f x) ∧
    (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
    A > 0 ∧ (∀ x, f x ≤ A) ∧ (∃ x₀, f x₀ = A) ∧
    (∀ B, B > 0 → (∀ x, f x ≤ B) → A ≤ B)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_amplitude_l1251_125112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l1251_125183

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | (n + 1) => Real.sqrt (49 * (sequence_a n)^2)

theorem a_100_value : sequence_a 99 = 7^99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l1251_125183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1251_125125

open Set

noncomputable def f (x : ℝ) : ℝ := (9*x^2 + 18*x - 60) / ((3*x - 4)*(x + 5))

theorem inequality_solution :
  {x : ℝ | f x < 2} = Ioo (-5/3) (4/3) ∪ Ioi 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1251_125125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_line_equation_l1251_125171

-- Define the circle C
def circle_C (a b r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = r^2}

-- Define the line that the center lies on
def center_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 + p.2 = 0}

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 - 1 = 0}

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - 2)}

-- Theorem for part 1
theorem circle_equation :
  ∀ a b r : ℝ,
  (a, b) ∈ center_line →
  (2, -1) ∈ circle_C a b r →
  (∃ p ∈ circle_C a b r ∩ tangent_line, True) →
  circle_C a b r = circle_C 1 (-2) (Real.sqrt 2) :=
by sorry

-- Helper function to represent arc length (not actually defined)
noncomputable def arc_length (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem for part 2
theorem line_equation :
  ∀ k : ℝ,
  (∃ p q : ℝ × ℝ, p ∈ circle_C 1 (-2) (Real.sqrt 2) ∧
                  q ∈ circle_C 1 (-2) (Real.sqrt 2) ∧
                  p ∈ line_l k ∧ q ∈ line_l k ∧
                  p ≠ q ∧
                  (arc_length p q) / (2 * π * Real.sqrt 2) = 1/3) →
  k = 1 ∨ k = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_line_equation_l1251_125171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1251_125180

theorem constant_term_expansion (x : ℝ) : 
  ∃ c : ℝ, ∃ p : ℝ → ℝ, (8*x + 1/(4*x))^8 = c + x * p x ∧ c = 1120 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1251_125180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_l1251_125159

theorem integral_sqrt_one_minus_x_squared : ∫ x in (0:ℝ)..(1:ℝ), Real.sqrt (1 - x^2) = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_l1251_125159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertices_distance_l1251_125150

/-- The equation of the parabolas -/
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + |y + 1| = 5

/-- A vertex of the parabolas -/
structure Vertex where
  x : ℝ
  y : ℝ
  is_vertex : parabola_equation x y

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The main theorem statement -/
theorem parabola_vertices_distance :
  ∃ (v1 v2 : Vertex), distance (v1.x, v1.y) (v2.x, v2.y) = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertices_distance_l1251_125150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1251_125118

/-- The rational function f(x) = (3x² + 8x + 12) / (2x + 3) -/
noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 8 * x + 12) / (2 * x + 3)

/-- The proposed oblique asymptote function -/
noncomputable def g (x : ℝ) : ℝ := (3/2) * x + 5/2

/-- Theorem stating that g is the oblique asymptote of f -/
theorem oblique_asymptote_of_f : 
  ∀ ε > 0, ∃ M, ∀ x > M, |f x - g x| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oblique_asymptote_of_f_l1251_125118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_geq_neg_one_l1251_125191

theorem negation_of_sin_geq_neg_one :
  (¬ ∀ x : ℝ, Real.sin x ≥ -1) ↔ (∃ x₀ : ℝ, Real.sin x₀ < -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_geq_neg_one_l1251_125191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_amount_l1251_125116

/-- Simple interest calculation function -/
noncomputable def simple_interest (P r t : ℝ) : ℝ := P + (P * r * t) / 100

/-- Proof of initial investment amount -/
theorem initial_investment_amount :
  ∃ (P r : ℝ),
    simple_interest P r 2 = 600 ∧
    simple_interest P r 7 = 850 ∧
    P = 500 := by
  -- Proof steps would go here
  sorry

#check initial_investment_amount

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_amount_l1251_125116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_associates_count_l1251_125107

theorem company_associates_count :
  let num_managers : ℕ := 15
  let manager_avg_salary : ℚ := 90000
  let associate_avg_salary : ℚ := 30000
  let company_avg_salary : ℚ := 40000
  let num_associates : ℚ := (num_managers * manager_avg_salary - num_managers * company_avg_salary) / (company_avg_salary - associate_avg_salary)
  ⌊num_associates⌋ = 75 := by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_associates_count_l1251_125107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_w_value_l1251_125158

theorem parabola_point_w_value (w : ℝ) : 
  (3, w^3) ∈ {p : ℝ × ℝ | p.2 = p.1^2 - 1} → w = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_w_value_l1251_125158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sylvia_time_is_45_l1251_125149

/-- The time it takes Sylvia to complete the job alone -/
def sylvia_time : ℝ := sorry

/-- The time it takes Carla to complete the job alone -/
def carla_time : ℝ := 30

/-- The time it takes Sylvia and Carla to complete the job together -/
def combined_time : ℝ := 18

/-- Theorem stating that Sylvia's time to complete the job alone is 45 minutes -/
theorem sylvia_time_is_45 :
  (1 / sylvia_time + 1 / carla_time = 1 / combined_time) →
  sylvia_time = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sylvia_time_is_45_l1251_125149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_third_term_is_three_l1251_125167

def Permutation := Fin 6 → Fin 6

def isValidPermutation (p : Permutation) : Prop :=
  Function.Bijective p ∧ p 0 ≠ 0 ∧ p 0 ≠ 5

def T : Set Permutation :=
  { p | isValidPermutation p }

theorem probability_third_term_is_three :
  ∃ (s : Finset Permutation),
    (s.filter (fun p => p 2 = 2)).card /
    s.card = 1 / 5 ∧
    ∀ p ∈ s, isValidPermutation p := by
  sorry

#check probability_third_term_is_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_third_term_is_three_l1251_125167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_problem_l1251_125129

def book_arrangement_count (arabic_count german_count spanish_count : ℕ) : ℕ :=
  let total_count := arabic_count + german_count + spanish_count
  let group_count := 2 + spanish_count  -- Arabic group, German group, and individual Spanish books
  (group_count.factorial) * (arabic_count.factorial) * (german_count.factorial)

theorem book_arrangement_problem (arabic_count german_count spanish_count : ℕ) 
  (h1 : arabic_count = 3)
  (h2 : german_count = 3)
  (h3 : spanish_count = 4) :
  book_arrangement_count arabic_count german_count spanish_count = 25920 := by
  sorry

#eval book_arrangement_count 3 3 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_arrangement_problem_l1251_125129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_circle_area_l1251_125101

-- Define the polar equation
noncomputable def polar_equation (r θ : ℝ) : Prop := r = 3 * Real.cos θ - 4 * Real.sin θ

-- Define the area of the circle
noncomputable def circle_area : ℝ := 25 * Real.pi / 4

-- Theorem statement
theorem polar_equation_circle_area :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ (r θ : ℝ), polar_equation r θ ↔ 
      (r * Real.cos θ - center.1)^2 + (r * Real.sin θ - center.2)^2 = radius^2) ∧
    circle_area = Real.pi * radius^2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_equation_circle_area_l1251_125101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_tangent_points_l1251_125130

noncomputable def f (x : ℝ) : ℝ := max (max (-7*x - 21) (2*x - 2)) (5*x + 10)

noncomputable def q : ℝ → ℝ := sorry

noncomputable def x₁ : ℝ := sorry

noncomputable def x₂ : ℝ := sorry

noncomputable def x₃ : ℝ := sorry

theorem sum_of_tangent_points :
  q 0 = 1 →
  (∃ a b c : ℝ, ∀ x, q x = a*x^2 + b*x + c) →
  (∀ x, q x ≥ f x) →
  q x₁ = f x₁ →
  q x₂ = f x₂ →
  q x₃ = f x₃ →
  x₁ ≠ x₂ →
  x₂ ≠ x₃ →
  x₁ ≠ x₃ →
  x₁ + x₂ + x₃ = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_tangent_points_l1251_125130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_complex_number_l1251_125108

theorem rotate_complex_number : 
  let z₁ : ℂ := 1 + Complex.I
  let θ : ℝ := -2 * Real.pi / 3  -- Negative for clockwise rotation
  let z₂ : ℂ := z₁ * Complex.exp (θ * Complex.I)
  z₂ = Complex.ofReal ((Real.sqrt 3 - 1) / 2) + Complex.I * Complex.ofReal (-(Real.sqrt 3 + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_complex_number_l1251_125108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dacid_weighted_average_l1251_125197

/-- Represents the marks and credits for a subject -/
structure Subject where
  name : String
  marks : ℚ
  credits : ℚ

/-- Calculates the weighted average given a list of subjects -/
def weightedAverage (subjects : List Subject) : ℚ :=
  let weightedSum := subjects.foldl (fun sum s => sum + s.marks * s.credits) 0
  let totalCredits := subjects.foldl (fun sum s => sum + s.credits) 0
  weightedSum / totalCredits

/-- Dacid's subjects with their marks and credits -/
def dacidSubjects : List Subject := [
  ⟨"English", 90, 3⟩,
  ⟨"Mathematics", 92, 4⟩,
  ⟨"Physics", 85, 4⟩,
  ⟨"Chemistry", 87, 3⟩,
  ⟨"Biology", 85, 2⟩
]

theorem dacid_weighted_average :
  weightedAverage dacidSubjects = 88.0625 := by
  sorry

#eval weightedAverage dacidSubjects

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dacid_weighted_average_l1251_125197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_given_sum_of_squares_and_product_l1251_125114

theorem max_sum_given_sum_of_squares_and_product (a b : ℝ) 
  (h1 : a^2 + b^2 = 130) (h2 : a * b = 45) : 
  (a + b) ≤ Real.sqrt 220 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_given_sum_of_squares_and_product_l1251_125114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_composition_function_equivalence_l1251_125136

-- Part 1
noncomputable def linear_function_1 (x : ℝ) : ℝ := 3 * x + 1 / 4
noncomputable def linear_function_2 (x : ℝ) : ℝ := -3 * x - 1 / 2

theorem linear_function_composition (x : ℝ) :
  (linear_function_1 (linear_function_1 x) = 9 * x + 1) ∧
  (linear_function_2 (linear_function_2 x) = 9 * x + 1) :=
by sorry

-- Part 2
noncomputable def f (x : ℝ) : ℝ := x^2 + x - 1

theorem function_equivalence (x : ℝ) :
  f (x - 2) = x^2 - 3 * x + 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_function_composition_function_equivalence_l1251_125136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_average_speed_l1251_125190

/-- Represents a segment of Tom's trip -/
structure Segment where
  distance : ℚ
  speed : ℚ

/-- Calculates the time taken for a segment -/
def time_for_segment (s : Segment) : ℚ := s.distance / s.speed

/-- Tom's trip details -/
def tom_trip : List Segment := [
  { distance := 10, speed := 20 },
  { distance := 15, speed := 30 },
  { distance := 25, speed := 45 },
  { distance := 40, speed := 60 }
]

/-- Total distance of Tom's trip -/
def total_distance : ℚ := 90

/-- Theorem stating Tom's average speed for the entire trip -/
theorem tom_average_speed :
  let total_time := (tom_trip.map time_for_segment).sum
  total_distance / total_time = 81/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_average_speed_l1251_125190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_theorem_l1251_125172

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

theorem cos_shift_theorem (t m : ℝ) (h_m : m > 0) :
  f (Real.pi / 4) = t ∧ g (Real.pi / 4 + m) = t →
  t = -1 / 2 ∧ ∃ (k : ℤ), m = Real.pi / 12 + k * Real.pi ∧ 
  ∀ (m' : ℝ), (∃ (k' : ℤ), m' = Real.pi / 12 + k' * Real.pi) → m ≤ m' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_shift_theorem_l1251_125172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_correct_l1251_125157

/-- The area of the triangle bounded by the y-axis and two lines -/
noncomputable def triangle_area (line1 line2 : ℝ → ℝ → Prop) : ℝ :=
  64 / 5

/-- The first line equation: y - 2x = -3 -/
def line1 (x y : ℝ) : Prop :=
  y - 2 * x = -3

/-- The second line equation: 2y + x = 10 -/
def line2 (x y : ℝ) : Prop :=
  2 * y + x = 10

/-- The theorem stating that the calculated area is correct -/
theorem triangle_area_is_correct :
  triangle_area line1 line2 = 64 / 5 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_correct_l1251_125157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l1251_125161

/-- Represents the time taken to complete a task -/
structure TaskTime where
  days : ℚ
  inv_days : ℚ
  inv_days_eq : inv_days = 1 / days

/-- Represents a worker's rate of work -/
def WorkRate := ℚ

theorem b_alone_time (a_time : TaskTime) (joint_time : ℚ) : TaskTime :=
  let result : TaskTime := {
    days := 6,
    inv_days := 1 / 6,
    inv_days_eq := rfl
  }
  have h1 : a_time.days = 12 := by sorry
  have h2 : a_time.inv_days = 1 / 12 := by sorry
  have h3 : joint_time = 3 := by sorry
  have h4 : (3 : ℚ) * a_time.inv_days + joint_time * (a_time.inv_days + result.inv_days) = 1 := by sorry
  result

#check b_alone_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l1251_125161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_interpolation_l1251_125119

-- Define the type of cubic polynomials
def CubicPolynomial := ℝ → ℝ

-- State the theorem
theorem cubic_polynomial_interpolation 
  (P Q R : CubicPolynomial) 
  (h1 : ∀ x, P x ≤ Q x ∧ Q x ≤ R x) 
  (h2 : ∃ u, P u = R u) :
  ∃ lambda : ℝ, 0 ≤ lambda ∧ lambda ≤ 1 ∧ ∀ x, Q x = lambda * P x + (1 - lambda) * R x :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_interpolation_l1251_125119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_exists_l1251_125153

/-- Triangle represented by three points in 2D space -/
structure Triangle where
  A : ℤ × ℤ
  B : ℤ × ℤ
  C : ℤ × ℤ

/-- Calculate the area of a triangle given its vertices -/
def area (t : Triangle) : ℚ :=
  let (x1, y1) := t.A
  let (x2, y2) := t.B
  let (x3, y3) := t.C
  (1 / 2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℤ × ℤ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem stating the existence of a triangle with the required properties -/
theorem triangle_exists : ∃ (t : Triangle),
  (area t < 1) ∧
  (distance t.A t.B > 2) ∧
  (distance t.B t.C > 2) ∧
  (distance t.C t.A > 2) := by
  -- Construct the triangle
  let t : Triangle := ⟨(0, 0), (2, 3), (3, 5)⟩
  
  -- Prove that this triangle satisfies all conditions
  have h1 : area t < 1 := by sorry
  have h2 : distance t.A t.B > 2 := by sorry
  have h3 : distance t.B t.C > 2 := by sorry
  have h4 : distance t.C t.A > 2 := by sorry

  -- Conclude the proof
  exact ⟨t, ⟨h1, h2, h3, h4⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_exists_l1251_125153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_upper_bound_sine_cosine_inequality_l1251_125189

theorem least_upper_bound_sine_cosine_inequality :
  ∀ C : ℝ, (∀ x : ℝ, Real.sin x * Real.cos x ≤ C * (Real.sin x ^ 6 + Real.cos x ^ 6)) → C ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_upper_bound_sine_cosine_inequality_l1251_125189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_specific_triangle_l1251_125184

/-- Given a triangle with sides a, b, and c, and a similar triangle with perimeter p,
    this function returns the length of the longest side in the similar triangle. -/
noncomputable def longest_side_similar_triangle (a b c p : ℝ) : ℝ :=
  let k := p / (a + b + c)
  max a (max b c) * k

/-- The theorem stating that for a triangle similar to one with sides 5, 12, and 13,
    if the perimeter is 150, then the longest side is 65. -/
theorem longest_side_specific_triangle :
  longest_side_similar_triangle 5 12 13 150 = 65 := by
  -- Unfold the definition of longest_side_similar_triangle
  unfold longest_side_similar_triangle
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry

-- Use #eval only for computable functions
def computable_longest_side (a b c p : Nat) : Nat :=
  let k : Nat := p / (a + b + c)
  max a (max b c) * k

#eval computable_longest_side 5 12 13 150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_specific_triangle_l1251_125184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_60_degrees_l1251_125135

/-- The length of an arc in a circle, given the radius and central angle in degrees -/
noncomputable def arcLength (radius : ℝ) (centralAngleDegrees : ℝ) : ℝ :=
  (centralAngleDegrees * Real.pi / 180) * radius

/-- Theorem: The length of an arc in a circle with radius 10 cm and a central angle of 60° is 10π/3 cm -/
theorem arc_length_60_degrees :
  arcLength 10 60 = 10 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_60_degrees_l1251_125135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l1251_125196

open Real

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * log x

-- State the theorem
theorem f_monotonicity :
  (∀ x ∈ Set.Ioo (0 : ℝ) (1 / Real.exp 1), StrictMonoOn (fun y ↦ -f y) {x}) ∧
  (∀ x ∈ Set.Ioi (1 / Real.exp 1), StrictMonoOn f {x}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_l1251_125196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_mopping_time_l1251_125173

/-- The time Jack spends mopping given the areas of the bathroom and kitchen floors and his mopping rate. -/
noncomputable def mopping_time (bathroom_area kitchen_area mopping_rate : ℝ) : ℝ :=
  (bathroom_area + kitchen_area) / mopping_rate

/-- Theorem stating that Jack spends 13 minutes mopping given the specific areas and mopping rate. -/
theorem jack_mopping_time :
  mopping_time 24 80 8 = 13 := by
  -- Unfold the definition of mopping_time
  unfold mopping_time
  -- Simplify the arithmetic
  simp [add_div]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_mopping_time_l1251_125173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_capital_after_m_years_l1251_125155

/-- Sequence representing the remaining capital after each year -/
noncomputable def a (d : ℝ) : ℕ → ℝ
  | 0 => 20 -- Initial capital
  | 1 => 30 - d
  | n + 2 => (3/2) * a d (n + 1) - d

/-- Theorem stating the relationship between d and m -/
theorem remaining_capital_after_m_years (m : ℕ) (d : ℝ) (h_m : m ≥ 3) :
  a d m = 40 ↔ d = (1000 * (3^m - 2^(m+1))) / (3^m - 2^m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_capital_after_m_years_l1251_125155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_time_is_8_01_l1251_125156

/-- Represents the number of minutes past 8:00 AM -/
def current_time : ℝ := sorry

/-- The minute hand moves at 6 degrees per minute -/
def minute_hand_speed : ℝ := 6

/-- The hour hand moves at 0.5 degrees per minute -/
def hour_hand_speed : ℝ := 0.5

/-- The starting position of the hour hand at 8:00 AM in degrees -/
def hour_hand_start : ℝ := 240

/-- Theorem stating that the current time is 8:01 AM -/
theorem current_time_is_8_01 : 
  (0 ≤ current_time) ∧ 
  (current_time < 60) ∧ 
  (|minute_hand_speed * (current_time + 8) - 
   (hour_hand_start + hour_hand_speed * (current_time - 4))| = 180) → 
  current_time = 1 := by
  sorry

#check current_time_is_8_01

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_time_is_8_01_l1251_125156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_eq_1260_l1251_125165

def count_pairs : ℕ :=
  let a_max := 70
  let b_max := 50
  let is_valid (a b : ℕ) : Prop := 1 ≤ a ∧ a ≤ a_max ∧ 1 ≤ b ∧ b ≤ b_max ∧ (a * b) % 5 = 0
  Finset.card (Finset.filter (fun p => is_valid p.1 p.2) (Finset.product (Finset.range (a_max + 1)) (Finset.range (b_max + 1))))

theorem count_pairs_eq_1260 : count_pairs = 1260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_eq_1260_l1251_125165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1251_125164

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ p = Real.pi ∧ ∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (x : ℝ), f (2 * Real.pi / 3 - x) = f (2 * Real.pi / 3 + x)) ∧
  (∀ (x y : ℝ), -Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ Real.pi / 3 → f x < f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1251_125164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_w_and_y_is_seven_l1251_125126

theorem sum_of_w_and_y_is_seven (W X Y Z : ℕ) : 
  W ∈ ({1, 2, 3, 4} : Set ℕ) → 
  X ∈ ({1, 2, 3, 4} : Set ℕ) → 
  Y ∈ ({1, 2, 3, 4} : Set ℕ) → 
  Z ∈ ({1, 2, 3, 4} : Set ℕ) → 
  W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
  (W : ℚ) / (X : ℚ) - (Y : ℚ) / (Z : ℚ) = 1 →
  W + Y = 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_w_and_y_is_seven_l1251_125126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1251_125188

/-- A quadratic radical is considered simple if it cannot be simplified further without changing its form. --/
def is_simple_quadratic_radical (x : ℝ) : Prop :=
  ∀ y z : ℝ, x = y * Real.sqrt z → y = 1 ∧ z = x^2

/-- The cube root of a real number --/
noncomputable def cuberoot (x : ℝ) : ℝ :=
  Real.rpow x (1/3)

theorem simplest_quadratic_radical : 
  is_simple_quadratic_radical (Real.sqrt 15) ∧
  ¬is_simple_quadratic_radical (Real.sqrt 12) ∧
  ¬is_simple_quadratic_radical (Real.sqrt (1/3)) ∧
  ¬is_simple_quadratic_radical (cuberoot 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_quadratic_radical_l1251_125188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fundraising_group_composition_l1251_125162

theorem fundraising_group_composition (initial_total : ℕ) 
  (h1 : initial_total > 0)
  (h2 : (60 : ℚ) / 100 * initial_total = (initial_total - 3 : ℚ) * (40 : ℚ) / 100) :
  (60 : ℚ) / 100 * initial_total = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fundraising_group_composition_l1251_125162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_players_theorem_l1251_125193

/-- Represents a classroom with rows and columns -/
structure Classroom where
  rows : ℕ
  columns : ℕ

/-- Represents the setup of the problem -/
def classroom_setup : Classroom :=
  { rows := 5, columns := 6 }

/-- The probability that a student can play games based on their height rank in a column -/
def play_probability (rank : ℕ) : ℚ :=
  if rank ≤ classroom_setup.rows then
    (classroom_setup.rows - rank) / classroom_setup.rows
  else
    0

/-- The expected number of students who can play games in one column -/
def expected_players_per_column : ℚ :=
  (Finset.range classroom_setup.rows).sum (λ i => play_probability (i + 1))

/-- The total expected number of students who can play games -/
def total_expected_players : ℚ :=
  classroom_setup.columns * expected_players_per_column

/-- The main theorem stating the expected number of students who can play games -/
theorem expected_players_theorem :
  total_expected_players = 163 / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_players_theorem_l1251_125193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1251_125185

/-- A line is tangent to a circle if it intersects the circle at exactly one point. -/
def IsTangentTo (line : Set (ℝ × ℝ)) (circle : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ line ∧ p ∈ circle

theorem tangent_lines_to_circle (x y : ℝ) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let slope := 1
  let line1 := {(x, y) : ℝ × ℝ | x - y + Real.sqrt 2 = 0}
  let line2 := {(x, y) : ℝ × ℝ | x - y - Real.sqrt 2 = 0}
  (∀ l : Set (ℝ × ℝ), (∃ c : ℝ, l = {(x, y) : ℝ × ℝ | y = slope * x + c}) →
    (IsTangentTo l circle ↔ l = line1 ∨ l = line2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l1251_125185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_l1251_125195

theorem existence_of_special_number : ∃ n : ℕ,
  (n > 0) ∧
  (∀ k ∈ Finset.range 100, k ≠ 0 → (n % k = 0 ↔ k % 2 = 1)) ∧
  (n = Finset.prod (Finset.filter (λ x => x % 2 = 1) (Finset.range 100)) id) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_l1251_125195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_distribution_theorem_l1251_125139

/-- Represents the number of students who submitted designs -/
def x : ℕ := sorry

/-- Represents the number of boxes of chocolates -/
def y : ℕ := sorry

/-- The condition that if every 2 students are rewarded with one box of chocolate, there will be 2 boxes left -/
axiom condition1 : x = 2 * y + 4

/-- The condition that if every 3 students are rewarded with one box of chocolate, there will be 3 extra boxes -/
axiom condition2 : x = 3 * y - 9

/-- Theorem stating that both conditions hold simultaneously -/
theorem chocolate_distribution_theorem : x = 2 * y + 4 ∧ x = 3 * y - 9 := by
  constructor
  . exact condition1
  . exact condition2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_distribution_theorem_l1251_125139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1251_125194

/-- Curve C -/
def curve_C (x y : ℝ) : Prop := x^2 + (y-1)^2 = 4

/-- Line l -/
def line_l (x y : ℝ) : Prop := y = 3*x

/-- Point P -/
def point_P : ℝ × ℝ := (1, 3)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem statement -/
theorem intersection_distance_sum :
  ∃ (M N : ℝ × ℝ),
    curve_C M.1 M.2 ∧
    curve_C N.1 N.2 ∧
    line_l M.1 M.2 ∧
    line_l N.1 N.2 ∧
    distance point_P M + distance point_P N = 7 * Real.sqrt 10 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_sum_l1251_125194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_16_l1251_125110

/-- The area of a triangle given by three points in 2D space -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * |x₁*y₂ + x₂*y₃ + x₃*y₁ - y₁*x₂ - y₂*x₃ - y₃*x₁|

/-- The area of triangle ABC with given coordinates is 16 -/
theorem triangle_area_is_16 : 
  triangleArea (3, -1) (1, -3) (-6, 6) = 16 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_16_l1251_125110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_with_five_black_l1251_125146

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  corner_x : Nat
  corner_y : Nat

/-- The size of the checkerboard -/
def board_size : Nat := 10

/-- Checks if a square contains at least 5 black squares -/
def has_five_black_squares (s : Square) : Bool :=
  if s.size ≥ 4 then true
  else if s.size = 3 then 
    (s.corner_x + s.corner_y) % 2 = 0  -- Upper left corner is black
  else false

/-- Counts the number of valid squares of a given size -/
def count_squares (size : Nat) : Nat :=
  if size = 3 then (board_size - size + 1)^2 / 2
  else (board_size - size + 1)^2

/-- Theorem: The total number of squares containing at least 5 black squares is 172 -/
theorem count_squares_with_five_black : (Finset.sum (Finset.range (board_size - 2)) (fun i => count_squares (i + 3))) = 172 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_squares_with_five_black_l1251_125146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monocolor_isosceles_triangles_l1251_125181

/-- The number of monocolor isosceles triangles in a regular polygon with n sides and m colored vertices -/
def number_of_monocolor_isosceles_triangles (n m : ℕ) : ℚ := (2 * n : ℚ) / 3

/-- Theorem stating that the number of monocolor isosceles triangles is independent of coloring -/
theorem monocolor_isosceles_triangles
  (n : ℕ) (h1 : n > 3) (h2 : Odd n) (h3 : ¬ 3 ∣ n) :
  ∀ (m : ℕ) (h4 : m ≤ n),
  number_of_monocolor_isosceles_triangles n m = (2 * n : ℚ) / 3 :=
by
  intro m h4
  rfl  -- reflexivity, since the definition matches the conclusion


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monocolor_isosceles_triangles_l1251_125181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_five_l1251_125198

/-- An arithmetic sequence with positive terms and common ratio greater than 1 -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  q : ℝ
  h1 : ∀ n, a n > 0
  h2 : q > 1
  h3 : ∀ n, a (n + 1) = q * a n

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sum (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (seq.a 1) * (1 - seq.q^n) / (1 - seq.q)

theorem arithmetic_sequence_sum_five
  (seq : ArithmeticSequence)
  (h4 : seq.a 3 + seq.a 5 = 20)
  (h5 : seq.a 2 * seq.a 6 = 64) :
  sum seq 5 = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_five_l1251_125198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l1251_125115

theorem system_solution :
  ∃! (x y : ℝ), x * (Real.log 3 / Real.log 2) + y = Real.log 18 / Real.log 2 ∧ 
                 (5 : ℝ)^x = (25 : ℝ)^y ∧ 
                 x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l1251_125115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_length_difference_l1251_125144

/-- Represents the length of the spring in cm for a given weight in kg. -/
def spring_length : ℝ → ℝ := sorry

/-- The set of weights for which we have data. -/
def weight_data : Set ℝ := {0, 1, 2, 3, 4, 5}

/-- Assumption that the spring_length function matches the given data. -/
axiom spring_length_data : 
  ∀ x ∈ weight_data, spring_length x = 20 + 0.5 * x

/-- The theorem stating that for any two consecutive weights in the data set,
    the difference in spring length is 0.5 cm. -/
theorem spring_length_difference (x₁ x₂ : ℝ) 
  (h₁ : x₁ ∈ weight_data) (h₂ : x₂ ∈ weight_data) (h_diff : x₂ - x₁ = 1) :
  spring_length x₂ - spring_length x₁ = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_length_difference_l1251_125144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_stride_difference_approx_six_l1251_125141

/-- The number of telephone poles -/
def num_poles : ℕ := 51

/-- The total distance between the first and last pole in feet -/
def total_distance : ℝ := 6600

/-- The number of Elmer's strides between consecutive poles -/
def elmer_strides : ℕ := 50

/-- The number of Oscar's leaps between consecutive poles -/
def oscar_leaps : ℕ := 15

/-- Elmer's stride length in feet -/
noncomputable def elmer_stride_length : ℝ := total_distance / (elmer_strides * (num_poles - 1))

/-- Oscar's leap length in feet -/
noncomputable def oscar_leap_length : ℝ := total_distance / (oscar_leaps * (num_poles - 1))

/-- The difference between Oscar's leap length and Elmer's stride length -/
noncomputable def length_difference : ℝ := oscar_leap_length - elmer_stride_length

theorem leap_stride_difference_approx_six :
  ∃ ε > 0, abs (length_difference - 6) < ε := by
  sorry

#eval num_poles
#eval total_distance
#eval elmer_strides
#eval oscar_leaps

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_stride_difference_approx_six_l1251_125141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_f_984_l1251_125131

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 5 then 1 - |x - 1| else 0  -- Default value for x outside [1,5]

-- State the properties of f
axiom f_property (x : ℝ) : x > 0 → f (3 * x) = 3 * f x

-- State the theorem
theorem smallest_x_equals_f_984 :
  ∃ x : ℝ, x > 0 ∧ f x = f 984 ∧ ∀ y : ℝ, y > 0 ∧ f y = f 984 → x ≤ y ∧ x = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_equals_f_984_l1251_125131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_greater_than_ten_l1251_125128

open BigOperators

def our_sequence (k : ℕ) : ℚ := (2 * k) / (2 * k - 1)

def our_product_sequence : ℚ := ∏ k in Finset.range 50, our_sequence (k + 1)

theorem product_greater_than_ten : our_product_sequence > 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_greater_than_ten_l1251_125128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sum_is_72_root_2_l1251_125169

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  a : ℝ
  r : ℝ
  volume_condition : a^3 = 432
  surface_area_condition : 2 * (a^2 / r + a^2 * r + a^2) = 384
  ratio_condition : r ≠ 1

/-- The sum of the lengths of all edges of the rectangular solid -/
noncomputable def edge_sum (solid : RectangularSolid) : ℝ :=
  4 * (solid.a / solid.r + solid.a + solid.a * solid.r)

/-- Theorem stating that the sum of the lengths of all edges is 72√2 -/
theorem edge_sum_is_72_root_2 (solid : RectangularSolid) :
  edge_sum solid = 72 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sum_is_72_root_2_l1251_125169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_prob_one_success_in_three_trials_l1251_125142

/-- The probability of exactly one success in three independent trials,
    where the probability of success in each trial is 1/3. -/
theorem binomial_prob_one_success_in_three_trials :
  let n : ℕ := 3
  let p : ℝ := 1/3
  let ξ : ℕ → ℝ := λ k => Nat.choose n k * p^k * (1-p)^(n-k)
  ξ 1 = 4/9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_prob_one_success_in_three_trials_l1251_125142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_is_rectangle_l1251_125111

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Represents two edges of a tetrahedron -/
structure TetrahedronEdges where
  edge1 : Fin 2 → ℝ × ℝ × ℝ
  edge2 : Fin 2 → ℝ × ℝ × ℝ

/-- Checks if two edges are opposite in a tetrahedron -/
def are_opposite_edges (t : RegularTetrahedron) (e : TetrahedronEdges) : Prop :=
  sorry

/-- Checks if a plane is parallel to two edges of a tetrahedron -/
def is_parallel_to_edges (p : Plane) (t : RegularTetrahedron) (e : TetrahedronEdges) : Prop :=
  sorry

/-- Represents the cross-section formed by a plane intersecting a tetrahedron -/
def cross_section (p : Plane) (t : RegularTetrahedron) : Set (ℝ × ℝ) :=
  sorry

/-- Predicate to check if a set of points forms a rectangle -/
def IsRectangle (s : Set (ℝ × ℝ)) : Prop :=
  sorry

/-- Theorem: The cross-section formed by a plane intersecting a regular tetrahedron, 
    where the plane is parallel to two opposite edges of the tetrahedron, is a rectangle -/
theorem cross_section_is_rectangle 
  (t : RegularTetrahedron) 
  (p : Plane) 
  (e : TetrahedronEdges) 
  (h1 : are_opposite_edges t e) 
  (h2 : is_parallel_to_edges p t e) : 
  IsRectangle (cross_section p t) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_is_rectangle_l1251_125111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_visible_area_approximate_area_is_close_to_82_l1251_125105

/-- The area visible during a walk around a square boundary -/
noncomputable def visible_area (side_length : ℝ) (visibility_range : ℝ) : ℝ :=
  let inner_square_side := side_length - 2 * visibility_range
  let inner_area := side_length^2 - inner_square_side^2
  let rectangle_area := 4 * side_length * visibility_range
  let circle_area := 2.25 * Real.pi
  inner_area + rectangle_area + circle_area

/-- Theorem stating the area visible during the walk -/
theorem walk_visible_area :
  visible_area 7 1.5 = 33 + 42 + 2.25 * Real.pi :=
by sorry

/-- The approximate numeric value of the visible area -/
noncomputable def approximate_visible_area : ℝ :=
  33 + 42 + 2.25 * Real.pi

/-- Theorem stating the approximate visible area is close to 82 -/
theorem approximate_area_is_close_to_82 :
  ∃ ε > 0, |approximate_visible_area - 82| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_visible_area_approximate_area_is_close_to_82_l1251_125105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_power_equation_solutions_l1251_125170

theorem factorial_power_equation_solutions :
  ∀ n k : ℕ, n > 0 → k > 0 → ((n + 1)^k - 1 = Nat.factorial n ↔ 
    (n = 1 ∧ k = 1) ∨ (n = 2 ∧ k = 1) ∨ (n = 4 ∧ k = 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_power_equation_solutions_l1251_125170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l1251_125148

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x - 2*a*y + 2*a^2 - 4*a = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  y = x + 4

-- Define the chord length function
noncomputable def chord_length (a : ℝ) : ℝ :=
  2 * Real.sqrt (-2 * (a - 3)^2 + 10)

-- Theorem statement
theorem max_chord_length :
  ∀ a : ℝ, 0 < a → a ≤ 4 →
  ∀ x y : ℝ, circle_equation x y a →
  ∀ x' y' : ℝ, line_equation x' y' →
  ∃ a_max : ℝ, 0 < a_max ∧ a_max ≤ 4 ∧
  chord_length a_max = 2 * Real.sqrt 10 ∧
  ∀ a' : ℝ, 0 < a' → a' ≤ 4 → chord_length a' ≤ chord_length a_max :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l1251_125148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_function_min_value_optimal_point_valid_min_value_achieved_at_optimal_point_l1251_125199

/-- The minimum value of 1/m + 2/n given the conditions of the exponential function problem -/
theorem exp_function_min_value (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) : 
  (1 / m + 2 / n : ℝ) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

/-- The point where the minimum value is achieved -/
noncomputable def optimal_point : ℝ × ℝ := (Real.sqrt 2 - 1, 2 - Real.sqrt 2)

/-- Proof that the optimal point satisfies the conditions -/
theorem optimal_point_valid : 
  let (m, n) := optimal_point
  m > 0 ∧ n > 0 ∧ m + n = 1 := by
  sorry

/-- Proof that the minimum value is achieved at the optimal point -/
theorem min_value_achieved_at_optimal_point :
  let (m, n) := optimal_point
  1 / m + 2 / n = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_function_min_value_optimal_point_valid_min_value_achieved_at_optimal_point_l1251_125199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l1251_125137

noncomputable def f (x : ℝ) : ℝ := 4 * x + 7

noncomputable def g (x : ℝ) : ℝ := 3 * x - 2

noncomputable def h (x : ℝ) : ℝ := f (g x)

noncomputable def h_inv (x : ℝ) : ℝ := (x + 1) / 12

theorem h_inverse_correct : Function.LeftInverse h_inv h ∧ Function.RightInverse h_inv h := by
  constructor
  · -- Left inverse proof
    intro x
    simp [h_inv, h, f, g]
    field_simp
    ring
  · -- Right inverse proof
    intro x
    simp [h_inv, h, f, g]
    field_simp
    ring

#check h_inverse_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_l1251_125137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_condition_l1251_125113

theorem cosine_sum_condition (α β γ : ℝ) : 
  0 < α → α < β → β < γ → γ < 2 * Real.pi →
  (∀ x : ℝ, Real.cos (x + α) + Real.cos (x + β) + Real.cos (x + γ) = 0) →
  α - β = -2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_condition_l1251_125113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coprime_product_l1251_125140

theorem max_coprime_product (k : ℕ) :
  ∃ (x y : ℕ), 
    x + y = 2 * k ∧ 
    Nat.Coprime x y ∧
    (∀ (a b : ℕ), a + b = 2 * k → Nat.Coprime a b → x * y ≥ a * b) ∧
    ((k = 1 ∧ x = 1 ∧ y = 1) ∨
     (k % 2 = 0 ∧ k ≠ 1 ∧ x = k + 1 ∧ y = k - 1) ∨
     (k % 2 = 1 ∧ k ≠ 1 ∧ x = k + 2 ∧ y = k - 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coprime_product_l1251_125140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_external_triangle_l1251_125177

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle (α : Type*) where
  A : α
  B : α
  C : α

/-- Represents the length of a side in a triangle -/
noncomputable def side_length {α : Type*} [NormedAddCommGroup α] (p q : α) : ℝ := ‖p - q‖

/-- Represents the perimeter of a triangle -/
noncomputable def perimeter {α : Type*} [NormedAddCommGroup α] (t : Triangle α) : ℝ :=
  side_length t.A t.B + side_length t.B t.C + side_length t.C t.A

/-- Represents the angle between three points -/
noncomputable def angle {α : Type*} [NormedAddCommGroup α] (p q r : α) : ℝ := sorry

theorem isosceles_right_triangle_external_triangle
  {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α]
  (ABC : Triangle α) (D : α) :
  side_length ABC.A ABC.B = 2 →
  side_length ABC.B ABC.C = 2 →
  angle ABC.B ABC.A ABC.C = π / 2 →
  angle ABC.A ABC.C D = π / 2 →
  perimeter (Triangle.mk ABC.A ABC.C D) = 2 * perimeter ABC →
  angle ABC.A ABC.C D = π / 3 →
  Real.cos (2 * angle ABC.B ABC.A D) = -1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_external_triangle_l1251_125177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l1251_125166

theorem factorial_divisibility (m n : ℕ) : 
  (n.factorial * (m.factorial)^n) ∣ (m*n).factorial := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_divisibility_l1251_125166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1251_125160

/-- The area of a triangle with sides 7, 24, and 25 is 84 square units. -/
theorem triangle_area : ∃ (side1 side2 side3 area : ℝ),
  side1 = 7 ∧ side2 = 24 ∧ side3 = 25 ∧
  area = (1/2) * side1 * side2 ∧
  side1^2 + side2^2 = side3^2 ∧
  area = 84 := by
  -- Introduce the variables
  let side1 : ℝ := 7
  let side2 : ℝ := 24
  let side3 : ℝ := 25
  let area : ℝ := (1/2) * side1 * side2
  
  -- Prove the existence
  use side1, side2, side3, area
  
  -- Prove the conditions
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · norm_num
  · norm_num

#check triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1251_125160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_pyramid_l1251_125187

theorem multiplication_pyramid (x : ℝ) (hx : x ≠ 0) :
  3 * x * (1 / x^2) = 3 / x ∧
  (3 / x) * ((x + 2) / (3 * x^3)) = x + 2 ∧
  3 * x * (x + 2) = 3 * x^2 + 6 * x ∧
  (9 * x^4 - 36 * x^2) / (3 * x^2 + 6 * x) = 3 * x^2 - 6 * x ∧
  (3 * x^2 - 6 * x) / (3 * x) = x - 2 ∧
  (x - 2) * x^2 = x^3 - 2 * x^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiplication_pyramid_l1251_125187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rakesh_gross_salary_calculation_l1251_125109

/-- Represents Rakesh's financial situation for a month --/
structure RakeshFinances where
  gross_salary : ℝ
  fixed_deposit_rate : ℝ
  additional_deposit : ℝ
  grocery_rate : ℝ
  utility_rate : ℝ
  vacation_fund_rate : ℝ
  medical_expense : ℝ
  income_tax_rate : ℝ
  net_cash : ℝ

/-- Calculates the net cash in hand based on Rakesh's financial parameters --/
def calculate_net_cash (rf : RakeshFinances) : ℝ :=
  let net_salary := rf.gross_salary * (1 - rf.income_tax_rate)
  let fixed_deposit := rf.gross_salary * 0.15
  let remaining := net_salary - fixed_deposit - rf.additional_deposit
  let expenses := remaining * (rf.grocery_rate + rf.utility_rate + rf.vacation_fund_rate)
  remaining - expenses - rf.medical_expense

/-- Theorem stating that given the conditions, Rakesh's gross salary is approximately 11305.41 --/
theorem rakesh_gross_salary_calculation (rf : RakeshFinances)
  (h1 : rf.fixed_deposit_rate = 0.03 / 12)
  (h2 : rf.additional_deposit = 200)
  (h3 : rf.grocery_rate = 0.3)
  (h4 : rf.utility_rate = 0.2)
  (h5 : rf.vacation_fund_rate = 0.05)
  (h6 : rf.medical_expense = 1500)
  (h7 : rf.income_tax_rate = 0.07)
  (h8 : rf.net_cash = 2380)
  (h9 : calculate_net_cash rf = rf.net_cash) :
  ∃ ε > 0, |rf.gross_salary - 11305.41| < ε := by
  sorry

#eval Float.round (11305.41 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rakesh_gross_salary_calculation_l1251_125109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l1251_125134

/-- Given vectors a, b, and c in ℝ², prove properties about their relationships -/
theorem vector_relationships (a b c : ℝ × ℝ) (m : ℝ) :
  a = (-1, 2) →
  c = (m - 1, 3 * m) →
  (∃ k : ℝ, c = k • a) →  -- c is parallel to a
  ‖b‖ = Real.sqrt 5 / 2 →
  (a + 2 • b) • (2 • a - b) = 0 →  -- (a + 2b) ⊥ (2a - b)
  (m = 2 / 5 ∧ a • b = -5 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relationships_l1251_125134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_line_segment_l1251_125127

/-- Represents a finite plane (e.g., a sheet of paper) --/
structure FinitePlane where
  bounds : Set (Real × Real)

/-- Represents a line in 2D space --/
structure Line where
  a : Real
  b : Real
  c : Real  -- ax + by + c = 0

/-- Checks if a point is within the finite plane --/
def FinitePlane.contains (plane : FinitePlane) (point : Real × Real) : Prop :=
  point ∈ plane.bounds

/-- Finds the intersection point of two lines --/
noncomputable def Line.intersect (l1 l2 : Line) : Real × Real :=
  sorry

/-- Checks if a point is on a line --/
def Line.contains (l : Line) (point : Real × Real) : Prop :=
  sorry

/-- Checks if three points are collinear --/
def collinear (p1 p2 p3 : Real × Real) : Prop :=
  sorry

theorem construct_line_segment 
  (plane : FinitePlane) 
  (L1 L2 : Line) 
  (P : Real × Real) 
  (h_P_in_plane : plane.contains P) 
  (h_lines_intersect : ∃ M, ¬plane.contains M ∧ L1.contains M ∧ L2.contains M) :
  ∃ M' : Real × Real, 
    plane.contains M' ∧ 
    collinear P M' (Line.intersect L1 L2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_line_segment_l1251_125127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balls_in_cube_l1251_125133

noncomputable def cube_side_length : ℝ := 5
noncomputable def ball_radius : ℝ := 1

noncomputable def cube_volume : ℝ := cube_side_length ^ 3
noncomputable def ball_volume : ℝ := (4 / 3) * Real.pi * ball_radius ^ 3

noncomputable def max_balls : ℕ := Int.toNat ⌊cube_volume / ball_volume⌋

theorem max_balls_in_cube :
  max_balls = 29 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_balls_in_cube_l1251_125133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_B_forms_triangle_l1251_125104

-- Define the sets of line segments
def set_A : List ℝ := [1, 2, 3]
def set_B : List ℝ := [2, 2, 2]
def set_C : List ℝ := [2, 2, 4]
def set_D : List ℝ := [1, 3, 5]

-- Define a function to check if a set of three line segments can form a triangle
def can_form_triangle (sides : List ℝ) : Prop :=
  sides.length = 3 ∧
  ∀ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    sides[i.val]! + sides[j.val]! > sides[k.val]!

-- Theorem statement
theorem only_set_B_forms_triangle :
  ¬(can_form_triangle set_A) ∧
  can_form_triangle set_B ∧
  ¬(can_form_triangle set_C) ∧
  ¬(can_form_triangle set_D) := by
  sorry

#check only_set_B_forms_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_set_B_forms_triangle_l1251_125104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_interval_l1251_125163

-- Define the function f(x) = 2x - ln x
noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.log x

-- Theorem statement
theorem f_monotone_decreasing_on_interval :
  StrictMonoOn f (Set.Ioo (0 : ℝ) (1/2 : ℝ)) := by
  sorry

#check f_monotone_decreasing_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_on_interval_l1251_125163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l1251_125175

theorem absolute_value_inequality (k : ℤ) : k = 13 ↔ 
  (∃ (S : Finset ℤ), S.card = 8 ∧ 
    (∀ x : ℤ, x ∈ S ↔ (x > 0 ∧ |x + 4| < k)) ∧
    (∀ m : ℤ, m < k → ∃ (T : Finset ℤ), T.card < 8 ∧ 
      (∀ x : ℤ, x ∈ T ↔ (x > 0 ∧ |x + 4| < m)))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_l1251_125175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_framed_painting_border_area_l1251_125174

/-- The area of the border of a framed rectangular painting -/
theorem framed_painting_border_area 
  (painting_height : ℝ) 
  (painting_width : ℝ) 
  (frame_width : ℝ) : 
  painting_height = 12 → 
  painting_width = 16 → 
  frame_width = 3 → 
  (painting_height + 2 * frame_width) * (painting_width + 2 * frame_width) - 
  painting_height * painting_width = 204 := by
  sorry

#check framed_painting_border_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_framed_painting_border_area_l1251_125174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_base_length_l1251_125192

/-- The area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ := (base * height) / 2

/-- Theorem: For a triangle with area 30 square meters and height 5 meters, the base is 12 meters -/
theorem triangle_base_length :
  ∃ (base : ℝ), triangleArea base 5 = 30 ∧ base = 12 := by
  use 12
  constructor
  · simp [triangleArea]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_base_length_l1251_125192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_carved_region_is_correct_volume_carved_region_proof_l1251_125117

/-- The volume of the region carved out by rotating y = -x^2 + 4 around the Y-axis
    from the solid formed by rotating 16x^2 + 25y^2 = 400 around the Y-axis -/
noncomputable def volume_carved_region : ℝ := 29.96 * Real.pi

/-- The equation of the parabola -/
def parabola (x : ℝ) : ℝ := -x^2 + 4

/-- The equation of the ellipse -/
def ellipse (x y : ℝ) : Prop := 16 * x^2 + 25 * y^2 = 400

/-- Theorem stating that the volume of the carved region is 29.96π -/
theorem volume_carved_region_is_correct :
  volume_carved_region = 29.96 * Real.pi :=
by rfl

/-- Lemma for the intersection points of the parabola and ellipse -/
lemma intersection_points :
  ∃ (x y : ℝ), parabola x = y ∧ ellipse x y ∧
  ((x = 0 ∧ y = 4) ∨ (x = 2 * Real.sqrt 6 / 5 ∧ y = 76 / 25) ∨ (x = -2 * Real.sqrt 6 / 5 ∧ y = 76 / 25)) :=
sorry

/-- Lemma for the volume of the paraboloid segment -/
lemma paraboloid_segment_volume :
  ∃ (K₁ : ℝ), K₁ = 0.448 * Real.pi :=
sorry

/-- Lemma for the volume of the ellipsoid segment -/
lemma ellipsoid_segment_volume :
  ∃ (K₂ : ℝ), K₂ = 29.51 * Real.pi :=
sorry

/-- Theorem proving that the volume of the carved region is indeed 29.96π -/
theorem volume_carved_region_proof :
  ∃ (K₁ K₂ : ℝ), K₁ = 0.448 * Real.pi ∧ K₂ = 29.51 * Real.pi ∧ volume_carved_region = K₁ + K₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_carved_region_is_correct_volume_carved_region_proof_l1251_125117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_angle_l1251_125132

/-- The acute angle between diagonals of a parallelogram -/
noncomputable def acute_angle_between_diagonals (a b h : ℝ) : ℝ :=
  Real.arctan ((2 * a * h) / (a^2 - b^2))

/-- Theorem: In a parallelogram with sides a > b and height h to the longer side,
    the acute angle between diagonals is arctan(2ah / (a² - b²)) -/
theorem parallelogram_diagonal_angle (a b h : ℝ) 
    (h1 : a > b) (h2 : a > 0) (h3 : b > 0) (h4 : h > 0) :
  ∃ α : ℝ, α = acute_angle_between_diagonals a b h ∧ 
             0 < α ∧ α < π/2 ∧
             α = Real.arctan ((2 * a * h) / (a^2 - b^2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_diagonal_angle_l1251_125132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cost_for_flowers_l1251_125122

/-- Represents a flower type with its survival rate and seed pack options -/
structure FlowerType where
  name : String
  survivalRate : Rat
  packSizes : List Nat
  packCosts : List Rat

/-- Calculates the minimum number of seeds needed for a given number of surviving flowers -/
def minSeedsNeeded (survivingFlowers : Nat) (survivalRate : Rat) : Nat :=
  Nat.ceil (survivingFlowers / survivalRate)

/-- Calculates the cost of buying seeds for a flower type -/
def costForFlowerType (ft : FlowerType) (survivingFlowers : Nat) : Rat :=
  let seedsNeeded := minSeedsNeeded survivingFlowers ft.survivalRate
  match ft.packSizes.zip ft.packCosts |>.find? (fun (size, _) => size ≥ seedsNeeded) with
  | some (_, cost) => cost
  | none => 0 -- This case should not occur in our problem

/-- Calculates the total cost with discount -/
def totalCostWithDiscount (costs : List Rat) (discountRate : Rat) : Rat :=
  let totalCost := costs.sum
  totalCost * (1 - discountRate)

/-- Main theorem statement -/
theorem minimum_cost_for_flowers : 
  let roses := FlowerType.mk "Roses" (2/5) [15, 40] [5, 10]
  let daisies := FlowerType.mk "Daisies" (3/5) [20, 50] [4, 9]
  let sunflowers := FlowerType.mk "Sunflowers" (1/2) [10, 30] [3, 7]
  let flowerTypes := [roses, daisies, sunflowers]
  let totalFlowers := 20
  let flowersPerType := totalFlowers / flowerTypes.length
  let costs := flowerTypes.map (costForFlowerType · flowersPerType)
  let discountRate := 1/5
  totalCostWithDiscount costs discountRate = 84/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_cost_for_flowers_l1251_125122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_min_on_interval_f_max_on_interval_f_achieves_min_max_l1251_125147

noncomputable def f (x : ℝ) : ℝ := 3 * x / (x + 1)

-- Statement 1: f is increasing on (-1, +∞)
theorem f_increasing : ∀ x y : ℝ, -1 < x → x < y → f x < f y := by sorry

-- Statement 2: The minimum value of f on [2, 5] is 2
theorem f_min_on_interval : ∀ x : ℝ, 2 ≤ x → x ≤ 5 → f x ≥ 2 := by sorry

-- Statement 3: The maximum value of f on [2, 5] is 5/2
theorem f_max_on_interval : ∀ x : ℝ, 2 ≤ x → x ≤ 5 → f x ≤ 5/2 := by sorry

-- Additional theorem to show that the min and max are achieved
theorem f_achieves_min_max : ∃ x y : ℝ, 2 ≤ x ∧ x ≤ 5 ∧ 2 ≤ y ∧ y ≤ 5 ∧ f x = 2 ∧ f y = 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_min_on_interval_f_max_on_interval_f_achieves_min_max_l1251_125147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_when_a_zero_condition_for_subset_l1251_125182

-- Define the function f (marked as noncomputable due to dependency on Real)
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - x) + 1 / Real.sqrt x

-- Define the domain A
def A : Set ℝ := {x | 0 < x ∧ x ≤ 3}

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | 1 - a < x ∧ x ≤ 2 * a + 4}

-- Theorem for part (1)
theorem intersection_and_union_when_a_zero :
  (A ∩ B 0 = {x : ℝ | 1 < x ∧ x ≤ 3}) ∧
  (A ∪ B 0 = {x : ℝ | 0 < x ∧ x ≤ 4}) := by
  sorry

-- Theorem for part (2)
theorem condition_for_subset :
  ∀ a > -1, (∀ x ∈ A, x ∈ B a) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_union_when_a_zero_condition_for_subset_l1251_125182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_l1251_125123

theorem trigonometric_sum (x y z : ℝ) :
  Real.sin x = Real.cos y →
  Real.sin y = Real.cos z →
  Real.sin z = Real.cos x →
  0 ≤ x ∧ x ≤ π/2 →
  0 ≤ y ∧ y ≤ π/2 →
  0 ≤ z ∧ z ≤ π/2 →
  x + y + z = 3*π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_sum_l1251_125123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_matrix_is_one_l1251_125186

open Real

noncomputable def matrix (α β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![sin α * sin β, sin α * cos β, cos α],
    ![cos β, -sin β, 0],
    ![cos α * sin β, cos α * cos β, -sin α]]

theorem det_matrix_is_one (α β : ℝ) :
  Matrix.det (matrix α β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_matrix_is_one_l1251_125186
