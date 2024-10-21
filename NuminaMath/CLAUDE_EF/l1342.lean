import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l1342_134279

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)

theorem derivative_f_at_one :
  deriv f 1 = 2 * Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_one_l1342_134279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nicolai_fruit_pounds_l1342_134250

-- Define the total amount of fruit eaten
noncomputable def total_fruit : ℚ := 8

-- Define the amount of fruit eaten by Mario in ounces
noncomputable def mario_fruit_oz : ℚ := 8

-- Define the amount of fruit eaten by Lydia in ounces
noncomputable def lydia_fruit_oz : ℚ := 24

-- Define the conversion factor from ounces to pounds
noncomputable def oz_to_lb : ℚ := 1 / 16

-- Theorem to prove
theorem nicolai_fruit_pounds :
  let mario_fruit_lb := mario_fruit_oz * oz_to_lb
  let lydia_fruit_lb := lydia_fruit_oz * oz_to_lb
  let nicolai_fruit_lb := total_fruit - (mario_fruit_lb + lydia_fruit_lb)
  nicolai_fruit_lb = 6 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nicolai_fruit_pounds_l1342_134250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1342_134235

theorem solve_exponential_equation :
  ∃ x : ℝ, 64 = 4 * (16 : ℝ) ^ (x - 1) → x = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1342_134235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_l1342_134222

noncomputable section

-- Define the fixed point F
def F : ℝ × ℝ := (-Real.sqrt 3, 0)

-- Define the line m
def m (x : ℝ) : Prop := x = -4 * Real.sqrt 3 / 3

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the distance ratio condition
def distance_ratio (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  Real.sqrt ((x + Real.sqrt 3)^2 + y^2) / abs (x + 4 * Real.sqrt 3 / 3) = Real.sqrt 3 / 2

-- Define lines l₁ and l₂
def l₁ (k t₁ x y : ℝ) : Prop := y = k * x + t₁
def l₂ (k t₂ x y : ℝ) : Prop := y = k * x + t₂

-- Define the theorem
theorem max_area_quadrilateral 
  (k t₁ t₂ : ℝ) 
  (h₁ : t₁ ≠ t₂) 
  (A B D E : ℝ × ℝ) 
  (hA : C A.1 A.2 ∧ l₁ k t₁ A.1 A.2) 
  (hB : C B.1 B.2 ∧ l₁ k t₁ B.1 B.2) 
  (hD : C D.1 D.2 ∧ l₂ k t₂ D.1 D.2) 
  (hE : C E.1 E.2 ∧ l₂ k t₂ E.1 E.2) 
  (h₂ : A ≠ B ∧ D ≠ E) 
  (h₃ : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2)) :
  (∀ M, distance_ratio M → C M.1 M.2) → 
  ∃ S : ℝ, S ≤ 4 ∧ 
    (∀ S' : ℝ, S' = abs ((A.1 - D.1) * (B.2 - D.2) - (B.1 - D.1) * (A.2 - D.2)) → S' ≤ S) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_quadrilateral_l1342_134222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_football_game_attendance_l1342_134276

theorem football_game_attendance :
  let adult_price : ℚ := 60 / 100
  let child_price : ℚ := 25 / 100
  let total_money : ℚ := 140
  let num_children : ℚ := 80
  let num_adults : ℚ := (total_money - child_price * num_children) / adult_price
  ⌊num_adults + num_children⌋ = 280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_football_game_attendance_l1342_134276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_from_plane_relations_l1342_134278

-- Define the basic types
variable (Point : Type) -- Type for points
variable (Line : Type) -- Type for lines
variable (Plane : Type) -- Type for planes

-- Define the relations
variable (contained_in : Line → Plane → Prop) -- Line contained in plane
variable (perpendicular : Line → Plane → Prop) -- Line perpendicular to plane
variable (parallel : Plane → Plane → Prop) -- Planes parallel
variable (line_perpendicular : Line → Line → Prop) -- Lines perpendicular

-- Theorem statement
theorem line_perpendicular_from_plane_relations
  (a b : Line) (α β : Plane)
  (h1 : contained_in a α)
  (h2 : perpendicular b β)
  (h3 : parallel α β) :
  line_perpendicular a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_from_plane_relations_l1342_134278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_parabola_l1342_134266

theorem points_form_parabola :
  ∀ (u : ℝ), 
  ∃ (a b c : ℝ),
  (9^u - 6 * 3^u - 2) = a * (3^u - 4)^2 + b * (3^u - 4) + c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_form_parabola_l1342_134266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_increase_is_100_percent_l1342_134275

/-- Represents the percentage increase in savings from the first year to the second year -/
noncomputable def savings_increase_percentage (
  first_year_income : ℝ)
  (first_year_savings_rate : ℝ)
  (second_year_income_increase_rate : ℝ)
  (total_expenditure_ratio : ℝ) : ℝ :=
  let first_year_savings := first_year_income * first_year_savings_rate
  let first_year_expenditure := first_year_income - first_year_savings
  let second_year_income := first_year_income * (1 + second_year_income_increase_rate)
  let second_year_expenditure := first_year_expenditure
  let second_year_savings := second_year_income - second_year_expenditure
  let savings_increase := second_year_savings - first_year_savings
  (savings_increase / first_year_savings) * 100

/-- Theorem stating that under the given conditions, the savings increase percentage is 100% -/
theorem savings_increase_is_100_percent :
  savings_increase_percentage 1 0.3 0.3 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_increase_is_100_percent_l1342_134275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_chord_theorem_l1342_134240

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determine if a point lies on a circle -/
def Point.onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Determine if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Determine if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The distance between two points -/
noncomputable def Point.distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem cyclic_quadrilateral_chord_theorem 
  (A B C D O X Y Z E F : Point) 
  (circle : Circle) 
  (AC BD EF : Line) : 
  (A.onCircle circle) →
  (B.onCircle circle) →
  (C.onCircle circle) →
  (D.onCircle circle) →
  (circle.center = O) →
  (X.onLine AC) →
  (X.onLine BD) →
  (Y.onLine EF) →
  (Y.onLine (Line.mk 1 0 (-A.x))) → -- AD is x = A.x
  (Z.onLine EF) →
  (Z.onLine (Line.mk 1 0 (-C.x))) → -- BC is x = C.x
  (X.onLine EF) →
  (Line.perpendicular EF (Line.mk (O.x - X.x) (O.y - X.y) 0)) → -- EF ⟂ OX
  (Point.distance X Y = Point.distance X Z) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_chord_theorem_l1342_134240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_win_four_games_prob_distribution_X_expected_value_X_player_B_advantage_l1342_134212

-- Define the probability of player A winning a single game
noncomputable def p_win : ℝ := 2/3

-- Define the number of games needed to win the match
def games_to_win : ℕ := 3

-- Define the maximum number of games in the match
def max_games : ℕ := 2 * games_to_win - 1

-- Define the random variable X as the total number of games played
noncomputable def X : ℕ → ℝ
| 3 => 1/3
| 4 => 10/27
| 5 => 8/27
| _ => 0

-- Theorem for the probability of player A winning exactly four games
theorem prob_win_four_games :
  Nat.choose 3 2 * p_win^2 * (1 - p_win) * p_win = 8/27 := by
  sorry

-- Theorem for the probability distribution of X
theorem prob_distribution_X :
  (X 3 = 1/3) ∧ (X 4 = 10/27) ∧ (X 5 = 8/27) := by
  sorry

-- Theorem for the expected value of X
theorem expected_value_X :
  3 * X 3 + 4 * X 4 + 5 * X 5 = 107/27 := by
  sorry

-- Define the probability of player B winning in best-of-three format
noncomputable def p_B_win_three : ℝ := (1 - p_win)^2 + 2 * p_win * (1 - p_win)^2

-- Define the probability of player B winning in best-of-five format
noncomputable def p_B_win_five : ℝ := (1 - p_win)^3 + 3 * p_win * (1 - p_win)^3 + 6 * p_win^2 * (1 - p_win)^3

-- Theorem that player B has a higher probability of winning in best-of-three format
theorem player_B_advantage :
  p_B_win_three > p_B_win_five := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_win_four_games_prob_distribution_X_expected_value_X_player_B_advantage_l1342_134212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1342_134286

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ 2 * x^2 + a * x - a^2 = 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∃! x₀ : ℝ, x₀^2 + 2 * a * x₀ + 2 * a ≤ 0

-- Define the range of a
def a_range : Set ℝ :=
  Set.Ioi 2 ∪ Set.Iio (-2)

-- State the theorem
theorem range_of_a : ∀ a : ℝ, ¬(p a ∨ q a) ↔ a ∈ a_range := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1342_134286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_60_divisors_prime_factors_l1342_134229

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

def product_of_divisors (n : ℕ) : ℕ :=
  (divisors n).prod id

def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.card

theorem product_of_60_divisors_prime_factors :
  num_distinct_prime_factors (product_of_divisors 60) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_60_divisors_prime_factors_l1342_134229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1342_134217

/-- Given an ellipse with semi-major axis a and semi-minor axis b -/
def ellipse (a b : ℝ) := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}

/-- The ellipse equation -/
def ellipse_equation (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- Theorem stating the eccentricity and the specific equation of the ellipse -/
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : Real.sqrt 3 * a = 2 * b) :
  eccentricity a b = 1/2 ∧ (∀ x y, ellipse_equation 4 (2 * Real.sqrt 3) x y ↔ x^2 / 16 + y^2 / 12 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1342_134217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_range_l1342_134205

/-- Represents a frequency distribution group -/
structure FrequencyGroup where
  lowerBound : ℚ
  upperBound : ℚ
  frequency : ℕ

/-- Represents a frequency distribution -/
def FrequencyDistribution := List FrequencyGroup

def sampleCapacity : ℕ := 66

def frequencyDistribution : FrequencyDistribution := [
  ⟨11.5, 15.5, 2⟩,
  ⟨15.5, 19.5, 4⟩,
  ⟨19.5, 23.5, 9⟩,
  ⟨23.5, 27.5, 18⟩,
  ⟨27.5, 31.5, 11⟩,
  ⟨31.5, 35.5, 12⟩,
  ⟨35.5, 39.5, 7⟩,
  ⟨39.5, 43.5, 3⟩
]

def countInRange (dist : FrequencyDistribution) (lower upper : ℚ) : ℕ :=
  (dist.filter (λ g => lower ≤ g.lowerBound ∧ g.upperBound ≤ upper)).foldr (λ g acc => g.frequency + acc) 0

theorem probability_in_range :
  (countInRange frequencyDistribution (31.5 : ℚ) (43.5 : ℚ) : ℚ) / sampleCapacity = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_range_l1342_134205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_not_sufficient_nor_necessary_l1342_134296

/-- A hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_relation : c^2 = a^2 + b^2

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := h.c / h.a

/-- The slope of the asymptotic lines of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

theorem hyperbola_asymptotes_not_sufficient_nor_necessary :
  (∃ h : Hyperbola, asymptote_slope h = Real.sqrt 2 ∧ eccentricity h ≠ Real.sqrt 3) ∧
  (∃ h : Hyperbola, eccentricity h = Real.sqrt 3 ∧ asymptote_slope h ≠ Real.sqrt 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_not_sufficient_nor_necessary_l1342_134296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_mean_difference_l1342_134257

/-- Represents the income data for a group of families -/
structure IncomeData where
  num_families : ℕ
  min_income : ℚ
  max_income : ℚ
  incorrect_max_income : ℚ
  sum_without_max : ℚ

/-- Calculates the mean of the actual data -/
noncomputable def actual_mean (data : IncomeData) : ℚ :=
  (data.sum_without_max + data.max_income) / data.num_families

/-- Calculates the mean of the incorrect data -/
noncomputable def incorrect_mean (data : IncomeData) : ℚ :=
  (data.sum_without_max + data.incorrect_max_income) / data.num_families

/-- The main theorem to be proved -/
theorem income_mean_difference (data : IncomeData)
  (h1 : data.num_families = 500)
  (h2 : data.min_income = 12000)
  (h3 : data.max_income = 120000)
  (h4 : data.incorrect_max_income = 1200000) :
  incorrect_mean data - actual_mean data = 2160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_mean_difference_l1342_134257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_vertices_distance_sum_l1342_134223

/-- Given a square with side length s, prove that a point P(x,y) satisfies the condition that the sum of
    the squares of its distances to the square's vertices is 4s^2 if and only if P lies on a circle
    centered at (s/2, s/2) with radius s/√2. -/
theorem square_vertices_distance_sum (s : ℝ) (h : s > 0) (x y : ℝ) :
  (x^2 + y^2) + (x^2 + (y-s)^2) + ((x-s)^2 + y^2) + ((x-s)^2 + (y-s)^2) = 4 * s^2 ↔
  (x - s/2)^2 + (y - s/2)^2 = (s/Real.sqrt 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_vertices_distance_sum_l1342_134223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_condition_solutions_l1342_134274

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 0 then Real.sin (Real.pi * x^2)
  else if x ≥ 0 then Real.exp (x - 1)
  else 0  -- This case is not specified in the original problem, so we set it to 0

-- State the theorem
theorem f_condition_solutions (a : ℝ) :
  (f 1 + f a = 2) ↔ (a = 1 ∨ a = -Real.sqrt 2 / 2) := by
  sorry

#check f_condition_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_condition_solutions_l1342_134274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_from_midpoints_l1342_134247

/-- Given a triangle PQR in 3D space, this theorem proves that if the midpoints of its sides
    have specific coordinates, then the coordinates of vertex P can be determined. -/
theorem triangle_vertex_from_midpoints (P Q R : ℝ × ℝ × ℝ) :
  let midpoint_QR : ℝ × ℝ × ℝ := (2, 7, 2)
  let midpoint_PR : ℝ × ℝ × ℝ := (3, 5, -3)
  let midpoint_PQ : ℝ × ℝ × ℝ := (1, 8, 5)
  (midpoint_QR = ((Q.1 + R.1)/2, (Q.2.1 + R.2.1)/2, (Q.2.2 + R.2.2)/2)) →
  (midpoint_PR = ((P.1 + R.1)/2, (P.2.1 + R.2.1)/2, (P.2.2 + R.2.2)/2)) →
  (midpoint_PQ = ((P.1 + Q.1)/2, (P.2.1 + Q.2.1)/2, (P.2.2 + Q.2.2)/2)) →
  P = (2, 6, 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vertex_from_midpoints_l1342_134247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l1342_134260

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The parabola y^2 = 4x -/
def IsOnParabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- The focus of the parabola y^2 = 4x -/
def FocusOfParabola : Point :=
  { x := 1, y := 0 }

/-- The distance between two points -/
noncomputable def Distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem -/
theorem parabola_point_x_coordinate
  (A : Point)
  (h1 : IsOnParabola A)
  (h2 : Distance A FocusOfParabola = 6) :
  A.x = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_x_coordinate_l1342_134260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1342_134233

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.sin (2 * x + Real.pi / 2)

theorem f_properties :
  -- Smallest positive period is π
  (∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧ ∀ (x : ℝ), f (x + T) = f x) ∧
  -- Intervals of monotonic increase
  (∀ (k : ℤ), ∀ (x : ℝ), 
    x ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) →
    ∀ (y : ℝ), x < y → y ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8) →
    f x < f y) ∧
  -- Maximum value when x ∈ [0, π/3]
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 3) ∧ f x = Real.sqrt 2 ∧
    ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 3) → f y ≤ Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1342_134233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1342_134228

-- Define the equation
noncomputable def equation (x a : ℝ) : Prop := Real.sqrt x + Real.sqrt (6 - 2*x) = a

-- Define the number of solutions function
noncomputable def num_solutions (a : ℝ) : ℕ :=
  if a < Real.sqrt 3 then 0
  else if a = Real.sqrt 3 then 1
  else if a > Real.sqrt 3 ∧ a < 3 then 2
  else if a = 3 then 1
  else 0

-- Theorem statement
theorem solution_count (a : ℝ) :
  (∃ x, equation x a) ↔ num_solutions a ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_l1342_134228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1342_134245

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.A < Real.pi ∧
  t.B > 0 ∧ t.B < Real.pi ∧
  t.C > 0 ∧ t.C < Real.pi ∧
  2 * t.a * Real.cos t.B = 2 * t.c - t.b ∧
  t.a = 2 ∧
  t.b + t.c = 4

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.A = Real.pi / 3 ∧ (1/2 * t.a * t.b * Real.sin t.C) = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1342_134245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1342_134234

/-- Calculates the speed of a train given its length and time to pass a fixed point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem stating that a train with given length and crossing time has a specific speed -/
theorem train_speed_calculation (length time : ℝ) 
  (h1 : length = 200) 
  (h2 : time = 8) : 
  train_speed length time = 25 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Rewrite using the hypotheses
  rw [h1, h2]
  -- Evaluate the division
  norm_num

#check train_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1342_134234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perspective_drawing_area_l1342_134248

theorem perspective_drawing_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (1/2 : ℝ) * a * b = (Real.sqrt 6 : ℝ) / 2 →
  (1/2 : ℝ) * a * ((Real.sqrt 2 : ℝ) / 4 * b) = (Real.sqrt 3 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perspective_drawing_area_l1342_134248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_and_orders_of_f_l1342_134254

open Complex

-- Define the hyperbolic sine function
noncomputable def sh (z : ℂ) : ℂ := (exp z - exp (-z)) / 2

-- Define the function f(z)
noncomputable def f (z : ℂ) : ℂ := (z^2 + 1)^3 * sh z

-- Theorem statement
theorem zeros_and_orders_of_f :
  (∀ k : ℤ, f (k * π * I) = 0 ∧ 
    ∃ g : ℂ → ℂ, ∃ U : Set ℂ, IsOpen U ∧ (k * π * I) ∈ U ∧
    (∀ z ∈ U, f z = (z - k * π * I) * g z) ∧ g (k * π * I) ≠ 0) ∧
  (f I = 0 ∧ f (-I) = 0 ∧
    ∃ g₁ g₂ : ℂ → ℂ, ∃ U₁ U₂ : Set ℂ, 
    IsOpen U₁ ∧ I ∈ U₁ ∧ IsOpen U₂ ∧ (-I) ∈ U₂ ∧
    (∀ z ∈ U₁, f z = (z - I)^3 * g₁ z) ∧ g₁ I ≠ 0 ∧
    (∀ z ∈ U₂, f z = (z + I)^3 * g₂ z) ∧ g₂ (-I) ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_and_orders_of_f_l1342_134254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_value_range_l1342_134256

/-- Given a function f(x) = x^2 - 2x + a ln(x) with two extreme points, 
    prove that the value of f at the larger extreme point is within a specific range. -/
theorem extreme_point_value_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (x₁ x₂ : ℝ) 
  (h_domain : ∀ x, x > 0 → f x = x^2 - 2*x + a * Real.log x)
  (h_extreme : x₁ < x₂ ∧ DifferentiableAt ℝ f x₁ ∧ DifferentiableAt ℝ f x₂ ∧ 
               deriv f x₁ = 0 ∧ deriv f x₂ = 0)
  (h_only_extremes : ∀ x, x > 0 → deriv f x = 0 → x = x₁ ∨ x = x₂) :
  (-3 - 2 * Real.log 2) / 4 < f x₂ ∧ f x₂ < -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_value_range_l1342_134256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_16_miles_l1342_134284

/-- Represents the walking scenario between Yolanda and Bob -/
structure WalkingScenario where
  total_distance : ℝ
  yolanda_rate : ℝ
  bob_rate : ℝ
  bob_start_delay : ℝ

/-- Calculates the distance Bob walked when he met Yolanda -/
noncomputable def bob_distance (scenario : WalkingScenario) : ℝ :=
  let remaining_distance := scenario.total_distance - scenario.yolanda_rate * scenario.bob_start_delay
  let total_rate := scenario.yolanda_rate + scenario.bob_rate
  let meeting_time := remaining_distance / total_rate
  scenario.bob_rate * meeting_time

/-- Theorem stating that Bob walked 16 miles when he met Yolanda -/
theorem bob_walked_16_miles (scenario : WalkingScenario) 
  (h1 : scenario.total_distance = 31)
  (h2 : scenario.yolanda_rate = 3)
  (h3 : scenario.bob_rate = 4)
  (h4 : scenario.bob_start_delay = 1) :
  bob_distance scenario = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_walked_16_miles_l1342_134284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_accuracy_percentage_l1342_134242

theorem test_accuracy_percentage (correct : ℕ) (total : ℕ) 
  (h1 : correct = 58) (h2 : total = 84) :
  ∃ (accuracy : ℝ), abs (accuracy - 69.05) < 0.01 ∧ accuracy = (correct : ℝ) / total * 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_accuracy_percentage_l1342_134242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_6_l1342_134202

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  q : ℝ      -- Common ratio
  h_geom : ∀ n, a (n + 1) = a n * q

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def sumGeometric (seq : GeometricSequence) (n : ℕ) : ℝ :=
  if seq.q = 1 then seq.a 1 * n
  else seq.a 1 * (1 - seq.q^n) / (1 - seq.q)

theorem geometric_sum_6 (seq : GeometricSequence) 
    (h1 : seq.a 1 + seq.a 3 = 5)
    (h2 : sumGeometric seq 4 = 15) :
  sumGeometric seq 6 = 63 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_6_l1342_134202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_rational_difference_l1342_134287

/-- A polynomial of degree at most 2 -/
def PolynomialDegree2 (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The difference between two real numbers is rational -/
def RationalDifference (x y : ℝ) : Prop :=
  ∃ q : ℚ, x - y = q

/-- The main theorem -/
theorem polynomial_with_rational_difference
  (f : ℝ → ℝ)
  (h_poly : PolynomialDegree2 f)
  (h_rat_diff : ∀ x y : ℝ, RationalDifference x y → ∃ q : ℚ, f x - f y = q) :
  ∃ b : ℚ, ∃ c : ℝ, ∀ x, f x = b * x + c :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_rational_difference_l1342_134287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1342_134246

-- Define the function f(x) = 6/x
noncomputable def f (x : ℝ) : ℝ := 6 / x

-- Define the property of being in Quadrant I or III
def in_quadrant_I_or_III (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)

-- Theorem statement
theorem inverse_proportion_quadrants :
  ∀ x : ℝ, x ≠ 0 → in_quadrant_I_or_III x (f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l1342_134246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_league_teams_l1342_134203

theorem k_league_teams (n : ℕ) : 
  (∀ i j : ℕ, i < n ∧ j < n ∧ i ≠ j → ∃! k : ℕ, k < n * (n - 1) / 2 ∧ (i, j) = (k / (n - 1), k % (n - 1))) →
  n * (n - 1) / 2 = 91 →
  n = 14 := by
    intro h1 h2
    sorry

#check k_league_teams

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_league_teams_l1342_134203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_section_perimeter_approx_l1342_134225

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ

/-- Calculates the perimeter of a section of the divided isosceles triangle -/
noncomputable def sectionPerimeter (triangle : IsoscelesTriangle) (k : ℕ) : ℝ :=
  let sectionBase := triangle.base / 6
  sectionBase + Real.sqrt (triangle.height^2 + (k * sectionBase)^2) + 
    Real.sqrt (triangle.height^2 + ((k + 1) * sectionBase)^2)

/-- Finds the maximum perimeter among the six sections -/
noncomputable def maxSectionPerimeter (triangle : IsoscelesTriangle) : ℝ :=
  List.foldl max 0 (List.map (sectionPerimeter triangle) (List.range 6))

/-- The theorem to be proved -/
theorem max_section_perimeter_approx (triangle : IsoscelesTriangle) 
  (h1 : triangle.base = 12) 
  (h2 : triangle.height = 15) : 
  ∃ ε > 0, abs (maxSectionPerimeter triangle - 33.97) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_section_perimeter_approx_l1342_134225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_expansion_l1342_134253

theorem middle_term_expansion (n a : ℕ) (x : ℝ) (h1 : n > a) (h2 : n > 0) (h3 : a > 0)
  (h4 : 1 + a^n = 65) : 
  (Nat.choose n (n / 2)) * x^(n - n / 2) * (a / x)^(n / 2) = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_term_expansion_l1342_134253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rectangles_count_l1342_134210

/-- Represents a cell in the triangle grid -/
structure Cell where
  row : Nat
  col : Nat

/-- The step-like right triangle with legs of 6 cells -/
def Triangle : List Cell :=
  (List.range 7).bind (λ i =>
    (List.range (7 - i)).map (λ j =>
      { row := i, col := j }))

/-- Number of rectangles with a given cell as the top-right corner -/
def rectanglesForCell (cell : Cell) : Nat :=
  (7 - cell.row) * (7 - cell.col)

/-- Total number of rectangles in the triangle -/
def totalRectangles : Nat :=
  (Triangle.map rectanglesForCell).sum

theorem triangle_rectangles_count :
  totalRectangles = 126 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_rectangles_count_l1342_134210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_with_l_pieces_l1342_134291

/-- Represents an L-shaped piece made of three 1x1x1 cubes -/
structure LPiece :=
  (x : ℕ) (y : ℕ) (z : ℕ)

/-- Represents a 1x1x3 bar -/
structure Bar :=
  (x : ℕ) (y : ℕ) (z : ℕ)

/-- Represents a box with dimensions m × n × k -/
structure Box (m n k : ℕ) where
  dim_m : m > 1
  dim_n : n > 1
  dim_k : k > 1

/-- Calculate the volume of a list of L-pieces -/
def volumeL (l : List LPiece) : ℕ :=
  3 * l.length

/-- Calculate the volume of a list of bars -/
def volumeB (r : List Bar) : ℕ :=
  3 * r.length

/-- Predicate to check if a box can be filled with L-pieces and bars -/
def canFillWithBoth (b : Box m n k) : Prop :=
  ∃ (l : List LPiece) (r : List Bar), volumeL l + volumeB r = m * n * k

/-- Predicate to check if a box can be filled with only L-pieces -/
def canFillWithLPieces (b : Box m n k) : Prop :=
  ∃ (l : List LPiece), volumeL l = m * n * k

/-- The main theorem to be proved -/
theorem fill_with_l_pieces 
  (m n k : ℕ) (b : Box m n k) :
  canFillWithBoth b → canFillWithLPieces b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fill_with_l_pieces_l1342_134291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_intersection_l1342_134258

/-- The length of a set {x | a ≤ x ≤ b} is defined as b - a -/
def setLength (a b : ℝ) : ℝ := b - a

/-- Set M -/
def M (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 3/4}

/-- Set N -/
def N (n : ℝ) : Set ℝ := {x | n - 1/3 ≤ x ∧ x ≤ n}

/-- M and N are subsets of [0, 1] -/
axiom M_subset (m : ℝ) : M m ⊆ Set.Icc 0 1

/-- M and N are subsets of [0, 1] -/
axiom N_subset (n : ℝ) : N n ⊆ Set.Icc 0 1

/-- The minimum length of M ∩ N is 1/12 -/
theorem min_length_intersection (m n : ℝ) :
  ∃ l, l = 1/12 ∧ ∀ m' n', setLength (max m (n' - 1/3)) (min (m' + 3/4) n') ≥ l := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_length_intersection_l1342_134258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_calculation_l1342_134215

open Real

/-- The height of the shorter tower in meters -/
noncomputable def short_tower_height : ℝ := 42

/-- The height of the taller tower in meters -/
noncomputable def tall_tower_height : ℝ := 56

/-- The angle of sight from the top of the taller tower to the top of the shorter tower in radians -/
noncomputable def sight_angle : ℝ := 15 * Real.pi / 180 + 58 * Real.pi / (180 * 60)

/-- The vertical distance between the bases of the towers in meters -/
noncomputable def base_height_diff : ℝ := 10

/-- The inclination angle of the slope -/
noncomputable def slope_angle : ℝ := 4 * Real.pi / 180 + 14 * Real.pi / (180 * 60)

/-- Theorem stating that given the conditions, the slope angle is approximately 4°14' -/
theorem slope_angle_calculation :
  ∃ (u : ℝ),
    Real.tan (sight_angle + Real.arctan (u / tall_tower_height)) = u / (tall_tower_height - short_tower_height) ∧
    u / base_height_diff = 1 / Real.tan slope_angle :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_calculation_l1342_134215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_bound_l1342_134262

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the left focus
def left_focus : ℝ × ℝ := (-1, 0)

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Define a line passing through the left focus
def line_through_focus (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Define the dot product of two 2D vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Main theorem
theorem ellipse_dot_product_bound :
  ∀ k : ℝ, ∀ A B : ℝ × ℝ,
  ellipse A.1 A.2 →
  ellipse B.1 B.2 →
  A.2 = line_through_focus k A.1 →
  B.2 = line_through_focus k B.1 →
  dot_product (A.1 - P.1, A.2 - P.2) (B.1 - P.1, B.2 - P.2) ≤ 17/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_dot_product_bound_l1342_134262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_occurs_at_4_or_5_l1342_134269

def a (n : ℕ) : ℤ := 20 - 4 * n

def S (n : ℕ+) : ℤ := 18 * (n : ℤ) - 2 * (n : ℤ)^2

theorem max_S_occurs_at_4_or_5 :
  ∃ k ∈ ({4, 5} : Set ℕ+), ∀ n : ℕ+, S n ≤ S k :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_occurs_at_4_or_5_l1342_134269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sodium_hydride_moles_required_l1342_134264

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- Represents a chemical reaction -/
structure Reaction where
  reactant1 : Moles
  reactant2 : Moles
  product1 : Moles
  product2 : Moles

/-- The balanced equation for the reaction NaH + H2O → NaOH + H2 -/
def sodium_hydride_water_reaction (r : Reaction) : Prop :=
  r.reactant1 = r.reactant2 ∧ r.reactant1 = r.product1 ∧ r.reactant1 = r.product2

theorem sodium_hydride_moles_required 
  (water_moles hydrogen_moles : Moles)
  (h_water : water_moles = (2 : ℝ))
  (h_hydrogen : hydrogen_moles = (2 : ℝ))
  (r : Reaction)
  (h_reaction : sodium_hydride_water_reaction r)
  (h_water_used : r.reactant2 = water_moles)
  (h_hydrogen_produced : r.product2 = hydrogen_moles) :
  r.reactant1 = (2 : ℝ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sodium_hydride_moles_required_l1342_134264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_geometry_problem_l1342_134241

/-- Square ABCD with side length 4 -/
structure Square (A B C D : ℝ × ℝ) :=
  (side_length : ℝ := 4)
  (is_square : Prop) -- Changed from IsSquare A B C D to Prop

/-- Point on a line segment -/
def PointOnSegment (P X Y : ℝ × ℝ) : Prop := sorry

/-- Perpendicular lines -/
def Perpendicular (L1 L2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

/-- Area of a polygon -/
noncomputable def Area (vertices : List (ℝ × ℝ)) : ℝ := sorry

/-- Main theorem -/
theorem square_geometry_problem 
  (A B C D E F G H I J : ℝ × ℝ) 
  (square : Square A B C D) :
  PointOnSegment E A B →
  PointOnSegment H D A →
  PointOnSegment F B C →
  PointOnSegment G C D →
  PointOnSegment I E H →
  PointOnSegment J E H →
  Perpendicular (F, I) (E, H) →
  Perpendicular (G, J) (E, H) →
  A.1 - E.1 = A.2 - H.2 →
  Area [A, E, H] = 2 →
  Area [B, F, I, E] = 2 →
  Area [D, H, J, G] = 2 →
  Area [F, C, G, J, I] = 2 →
  (F.1 - I.1)^2 + (F.2 - I.2)^2 = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_geometry_problem_l1342_134241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_condition_l1342_134213

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 8 * x / (1 + x^2)
noncomputable def g (a x : ℝ) : ℝ := x^2 - a * x + 1

-- Define the theorem
theorem function_equivalence_condition (a : ℝ) : 
  (∀ x₁ : ℝ, x₁ ≥ 0 → ∃! x₀ : ℝ, -1 ≤ x₀ ∧ x₀ ≤ 2 ∧ f x₁ = g a x₀) ↔ 
  (a ≤ -2 ∨ a > 5/2) := by
  sorry

#check function_equivalence_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_condition_l1342_134213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_except_a_l1342_134259

noncomputable def f (n : ℕ+) : ℕ := 
  ⌊(n : ℝ) + Real.sqrt ((n : ℝ) / 3) + 1/2⌋.toNat

def a (n : ℕ+) : ℕ := 3 * n^2 - 2 * n

theorem f_range_except_a :
  ∀ m : ℕ+, (∃ n : ℕ+, f n = m) ↔ ¬∃ k : ℕ+, a k = m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_except_a_l1342_134259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1342_134249

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (a b c : V)

-- State the theorem
theorem angle_between_vectors
  (sum_zero : a + b + c = 0)
  (norm_a : ‖a‖ = 1)
  (norm_b : ‖b‖ = 2)
  (norm_c : ‖c‖ = Real.sqrt 7) :
  Real.arccos (inner a b / (‖a‖ * ‖b‖)) = π / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l1342_134249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_meeting_probability_early_zhenya_meeting_probability_late_zhenya_meeting_probability_l1342_134281

-- Define the time range (in minutes) for arrivals
noncomputable def timeRange : ℝ := 60

-- Define the waiting time (in minutes)
noncomputable def waitTime : ℝ := 10

-- Define the probability of meeting in the original scenario
noncomputable def probMeetOriginal : ℝ := 7 / 36

-- Define the probability of meeting when Zhenya arrives before 12:30 PM
noncomputable def probMeetZhenyaEarly : ℝ := 2 / 9

-- Define the probability of meeting when Zhenya arrives between 12:00 PM and 12:50 PM
noncomputable def probMeetZhenyaLate : ℝ := 11 / 60

-- Define a function to calculate the area of intersection
noncomputable def intersectionArea (x y : ℝ) : ℝ := 
  (min x y * waitTime * 2) - (waitTime * waitTime)

-- Theorem for the original scenario
theorem original_meeting_probability : 
  (∫ x in (0 : ℝ)..timeRange, ∫ y in (0 : ℝ)..timeRange, 
    (if |x - y| ≤ waitTime then 1 else 0) / (timeRange * timeRange)) = probMeetOriginal := by
  sorry

-- Theorem for Zhenya arriving before 12:30 PM
theorem early_zhenya_meeting_probability :
  (∫ x in (0 : ℝ)..timeRange, ∫ y in (0 : ℝ)..(timeRange/2), 
    (if |x - y| ≤ waitTime then 1 else 0) / (timeRange * (timeRange/2))) = probMeetZhenyaEarly := by
  sorry

-- Theorem for Zhenya arriving between 12:00 PM and 12:50 PM
theorem late_zhenya_meeting_probability :
  (∫ x in (0 : ℝ)..timeRange, ∫ y in (0 : ℝ)..(timeRange - waitTime), 
    (if |x - y| ≤ waitTime then 1 else 0) / (timeRange * (timeRange - waitTime))) = probMeetZhenyaLate := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_meeting_probability_early_zhenya_meeting_probability_late_zhenya_meeting_probability_l1342_134281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1342_134267

open Set
open Function

def f (x : ℝ) : ℝ := |x| + 1

theorem f_monotone_increasing :
  MonotoneOn f (Set.Ioi 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l1342_134267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_bitcoin_donation_l1342_134236

theorem jake_bitcoin_donation :
  ∀ (initial_donation : ℕ),
  (3 * ((80 - initial_donation) / 2) - 10 = 80) →
  initial_donation = 20 :=
by
  intro initial_donation h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jake_bitcoin_donation_l1342_134236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_n_l1342_134294

/-- Definition of a triangle with integral sides -/
structure IntegralTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Semiperimeter of a triangle -/
def semiperimeter (t : IntegralTriangle) : ℚ :=
  (t.a.val + t.b.val + t.c.val : ℚ) / 2

/-- Inradius of a triangle using Heron's formula -/
noncomputable def inradius (t : IntegralTriangle) : ℝ :=
  let s := semiperimeter t
  let area := Real.sqrt (s * (s - t.a.val) * (s - t.b.val) * (s - t.c.val))
  area / s

/-- The set of positive integers n for which there exists a triangle with integral sides
    such that its semiperimeter divided by its inradius is n -/
def validN : Set ℕ+ :=
  {n | ∃ t : IntegralTriangle, (↑(semiperimeter t) / inradius t) = n}

/-- The main theorem stating that the set of valid n is infinite -/
theorem infinitely_many_valid_n : Set.Infinite validN := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_valid_n_l1342_134294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_approximation_l1342_134280

theorem product_approximation (n : ℕ) (h : n = 1730) (h_div : n % 24 = 0) :
  ∃ x : ℕ, x > 0 ∧ 173 * x = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_approximation_l1342_134280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_g_zeros_properties_l1342_134218

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.exp x - f x - a

-- Statement 1
theorem inequality_proof (x : ℝ) (h : x > -1 ∧ x < 0) : f x < x ∧ x < -f (-x) := by
  sorry

-- Statement 2
theorem g_zeros_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : g a x₁ = 0) (h2 : g a x₂ = 0) (h3 : x₁ ≠ x₂) (h4 : x₁ < x₂) :
  a > 1 ∧ x₁ + x₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_g_zeros_properties_l1342_134218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1342_134226

-- Define the function
noncomputable def f (x y : ℝ) : ℝ := (x * y) / (x^2 + y^2)

-- Define the domain constraints
def domain (x y : ℝ) : Prop :=
  3/7 ≤ x ∧ x ≤ 2/3 ∧ 1/4 ≤ y ∧ y ≤ 1/2

-- State the theorem
theorem max_value_of_f :
  ∀ x y : ℝ, domain x y → f x y ≤ 24/73 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1342_134226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_properties_l1342_134232

/-- A convex polyhedron whose vertices divide each edge of a regular tetrahedron into 3 equal parts -/
structure ConvexPolyhedron where
  edge_length : ℝ
  h : edge_length > 0

namespace ConvexPolyhedron

/-- The length of a body diagonal -/
noncomputable def body_diagonal_length (p : ConvexPolyhedron) : ℝ :=
  p.edge_length * Real.sqrt 5

/-- The distance from a body diagonal to the centroid -/
noncomputable def distance_to_centroid (p : ConvexPolyhedron) : ℝ :=
  p.edge_length * Real.sqrt 2 / 4

/-- The number of intersection points of body diagonals -/
def intersection_points : ℕ := 30

/-- Main theorem about the properties of the convex polyhedron -/
theorem polyhedron_properties (p : ConvexPolyhedron) :
  (body_diagonal_length p = p.edge_length * Real.sqrt 5) ∧
  (distance_to_centroid p = p.edge_length * Real.sqrt 2 / 4) ∧
  (intersection_points = 30) := by
  sorry

end ConvexPolyhedron

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polyhedron_properties_l1342_134232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_stair_climbing_time_l1342_134261

/-- The sum of an arithmetic sequence with given parameters -/
noncomputable def arithmetic_sequence_sum (n : ℕ) (a : ℝ) (d : ℝ) : ℝ :=
  n / 2 * (2 * a + (n - 1) * d)

/-- Theorem: The sum of the specific arithmetic sequence is 378 -/
theorem jimmy_stair_climbing_time : arithmetic_sequence_sum 7 30 8 = 378 := by
  -- Unfold the definition of arithmetic_sequence_sum
  unfold arithmetic_sequence_sum
  -- Simplify the expression
  simp [Nat.cast_add, Nat.cast_mul, Nat.cast_sub, Nat.cast_one]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_stair_climbing_time_l1342_134261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1342_134298

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 6)

theorem min_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l1342_134298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_supplementary_angle_l1342_134270

/-- Given a circle with radius 15 cm and a central angle of 45°, 
    the length of the arc subtended by the supplementary angle is 22.5π cm. -/
theorem arc_length_supplementary_angle (O : Point) (r : ℝ) (θ : Real) :
  r = 15 →
  θ = 45 * (π / 180) →
  2 * π * r * ((360 * (π / 180) - 2 * θ) / (2 * π)) = 22.5 * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_supplementary_angle_l1342_134270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l1342_134273

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance from a point to a vertical line -/
def distToVerticalLine (p : Point) (lineX : ℝ) : ℝ :=
  |p.x - lineX|

/-- The distance between two points -/
noncomputable def distBetweenPoints (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The set of points satisfying the given condition -/
noncomputable def trajectorySet : Set Point :=
  {p : Point | distToVerticalLine p (-1) = distBetweenPoints p ⟨2, 0⟩ - 1}

theorem trajectory_is_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ p : Point, p ∈ trajectorySet ↔ a * p.x^2 + b * p.x + c * p.y = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_parabola_l1342_134273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beads_removed_l1342_134289

theorem beads_removed (white_beads black_beads : ℕ) 
  (h1 : white_beads = 51)
  (h2 : black_beads = 90)
  (white_fraction : ℚ)
  (black_fraction : ℚ)
  (h3 : white_fraction = 1 / 3)
  (h4 : black_fraction = 1 / 6) :
  (white_fraction * white_beads + black_fraction * black_beads).floor = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beads_removed_l1342_134289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_sum_condition_l1342_134277

open Set Finset

theorem max_subset_sum_condition (n : ℕ) (hn : n > 0) :
  let S : Finset ℕ := Finset.range (3*n) \ {0}
  ∃ (T : Finset ℕ), T ⊆ S ∧ 
    (∀ x y z, x ∈ T → y ∈ T → z ∈ T → x + y + z ∉ T) ∧
    (∀ U : Finset ℕ, U ⊆ S → (∀ x y z, x ∈ U → y ∈ U → z ∈ U → x + y + z ∉ U) → U.card ≤ T.card) ∧
    T.card = 2*n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_sum_condition_l1342_134277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l1342_134243

/-- The constant term in the expansion of (x + 2/x^2)^6 -/
def constant_term : ℕ := 60

/-- The binomial expression (x + 2/x^2)^6 -/
noncomputable def binomial_expression (x : ℝ) : ℝ := (x + 2 / x^2) ^ 6

theorem constant_term_proof :
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = binomial_expression x) ∧
  (∃ c : ℝ, f 0 = c) ∧
  (constant_term : ℝ) = f 0 :=
by
  sorry

#check constant_term_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_proof_l1342_134243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exists_l1342_134299

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 4
  else -x^2 + 3 * x - 5

-- State the theorem
theorem unique_solution_exists :
  ∃! x : ℝ, x ≥ 1 ∧ f x = 0 ∧ |x - 1.192| < 0.001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exists_l1342_134299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1342_134268

/-- A quadratic function f(x) with a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/4) * x^2 - a * x + 4

/-- The statement that f has two distinct positive zeros -/
def has_two_distinct_positive_zeros (a : ℝ) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0

/-- Theorem stating that if f has two distinct positive zeros, then a > 2 -/
theorem a_range (a : ℝ) : has_two_distinct_positive_zeros a → a > 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_l1342_134268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_real_solutions_l1342_134272

theorem equation_real_solutions : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, (6*x)/(x^2 + 2*x + 5) + (7*x)/(x^2 - 7*x + 5) = 1) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_real_solutions_l1342_134272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1342_134231

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.log (1 - 2 * x) / Real.log 10)

-- State the theorem about the domain of f(x)
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Set.Iic 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1342_134231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_implies_m_eq_8_l1342_134200

-- Define the ellipse equation
def is_ellipse (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (m - 2) + y^2 / (10 - m) = 1

-- Define the condition that the major axis lies on the x-axis
def major_axis_on_x (m : ℝ) : Prop :=
  m - 2 > 10 - m ∧ m - 2 > 0 ∧ 10 - m > 0

-- Define the focal length
noncomputable def focal_length (m : ℝ) : ℝ :=
  Real.sqrt ((m - 2) - (10 - m))

-- Theorem statement
theorem ellipse_focal_length_implies_m_eq_8 (m : ℝ) :
  is_ellipse m → major_axis_on_x m → focal_length m = 4 → m = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_implies_m_eq_8_l1342_134200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_altitude_ratio_l1342_134290

/-- Triangle with vertices X, Y, Z -/
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Orthocenter of a triangle -/
noncomputable def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Altitude from vertex Z to side XY -/
noncomputable def altitude_Z (t : Triangle) : ℝ × ℝ := sorry

/-- Length of a line segment -/
noncomputable def segment_length (a b : ℝ × ℝ) : ℝ := sorry

/-- Tangent of an angle in a triangle -/
noncomputable def tan_angle (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

theorem orthocenter_altitude_ratio (t : Triangle) :
  let H := orthocenter t
  let K := altitude_Z t
  segment_length H K = 8 ∧ segment_length H t.Z = 20 →
  tan_angle t t.X * tan_angle t t.Y = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_altitude_ratio_l1342_134290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l1342_134252

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3/2) * x - 7/2

-- State the theorem
theorem find_a (h1 : ∀ x, f (2*x + 1) = 3*x - 2) (h2 : f a = 7) : a = 7 := by
  -- The proof goes here
  sorry

#check find_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l1342_134252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bills_original_selling_price_l1342_134239

/-- Represents the original purchase price of the product -/
def P : ℝ := sorry

/-- Represents Bill's original selling price -/
def S : ℝ := 1.1 * P

/-- Represents the new selling price if Bill had purchased the product for 10% less and sold it at a 30% profit -/
def S_new : ℝ := 1.17 * P

theorem bills_original_selling_price :
  S_new - S = 42 →
  S = 660 := by
  intro h
  sorry

#check bills_original_selling_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bills_original_selling_price_l1342_134239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_fifth_root_l1342_134285

theorem cos_pi_fifth_root (a b : ℕ+) (h : (a : ℝ) * (Real.cos (π / 5))^3 - (b : ℝ) * Real.cos (π / 5) - 1 = 0) :
  a + b = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_fifth_root_l1342_134285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_p_value_l1342_134255

-- Define the points
noncomputable def A : ℝ × ℝ := (4, 12)
noncomputable def B : ℝ × ℝ := (12, 0)
noncomputable def C (p : ℝ) : ℝ × ℝ := (0, p)
noncomputable def Q : ℝ × ℝ := (0, 12)

-- Define the area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem triangle_area_implies_p_value (p : ℝ) :
  triangleArea A B (C p) = 20 → p = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_implies_p_value_l1342_134255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1342_134293

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_domain : ∀ x, x ∈ Set.Icc (-1) 1 → f x ∈ Set.range f
axiom f_increasing : ∀ x y, x < y → x ∈ Set.Icc (-1) 1 → y ∈ Set.Icc (-1) 1 → f x < f y
axiom f_odd : ∀ x, x ∈ Set.Icc (-1) 1 → f (-x) = -f x

-- Define the condition
def condition (a : ℝ) : Prop := f (-a + 1) + f (4*a - 5) > 0

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, condition a ↔ a ∈ Set.Ioo (4/3) (3/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1342_134293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_stock_is_400_l1342_134238

/-- Represents the coffee stock and calculations for a grocer --/
structure CoffeeStock where
  initial : ℝ
  -- Initial stock of coffee in pounds
  decaf_ratio_initial : ℝ
  -- Ratio of decaffeinated coffee in initial stock
  additional : ℝ
  -- Additional coffee bought in pounds
  decaf_ratio_additional : ℝ
  -- Ratio of decaffeinated coffee in additional stock
  decaf_ratio_final : ℝ
  -- Final ratio of decaffeinated coffee after purchase
  decaf_ratio_initial_value : decaf_ratio_initial = 0.20
  additional_value : additional = 100
  decaf_ratio_additional_value : decaf_ratio_additional = 0.70
  decaf_ratio_final_value : decaf_ratio_final = 0.30
  balance_equation : (decaf_ratio_initial * initial + decaf_ratio_additional * additional) / (initial + additional) = decaf_ratio_final

/-- Theorem stating that the initial stock of coffee is 400 pounds --/
theorem initial_stock_is_400 (stock : CoffeeStock) : stock.initial = 400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_stock_is_400_l1342_134238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_implies_composite_middle_l1342_134208

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, 1 < d ∧ d < n ∧ n % d = 0

theorem prime_sum_implies_composite_middle 
  (n : ℕ) 
  (h : n ∈ ({5, 11, 17, 29, 41} : Set ℕ)) : 
  Nat.Prime n → Nat.Prime (n + 4) → is_composite (n + 2) :=
by
  sorry

#check prime_sum_implies_composite_middle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_sum_implies_composite_middle_l1342_134208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_100_l1342_134206

/-- Represents the boat's journey with given conditions -/
structure BoatJourney where
  downstream_time : ℚ
  upstream_distance : ℚ
  upstream_time : ℚ
  stream_speed : ℚ

/-- Calculates the downstream distance given a BoatJourney -/
def downstream_distance (j : BoatJourney) : ℚ :=
  let boat_speed := (j.upstream_distance / j.upstream_time) + j.stream_speed
  (boat_speed + j.stream_speed) * j.downstream_time

/-- Theorem stating that the downstream distance is 100 km given the specific conditions -/
theorem downstream_distance_is_100 (j : BoatJourney) 
  (h1 : j.downstream_time = 4)
  (h2 : j.upstream_distance = 75)
  (h3 : j.upstream_time = 15)
  (h4 : j.stream_speed = 10) :
  downstream_distance j = 100 := by
  sorry

def example_journey : BoatJourney :=
{ downstream_time := 4
  upstream_distance := 75
  upstream_time := 15
  stream_speed := 10 }

#eval downstream_distance example_journey

end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_100_l1342_134206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unit_cubes_correct_l1342_134201

/-- The minimum number of unit cubes in a dissection of a cube -/
def min_unit_cubes (n : ℕ) : ℕ := 2 * n + 1

/-- The function representing the true minimum number of unit cubes in a dissection -/
def minimum_unit_cubes_in_dissection (edge_length : ℕ) : ℕ :=
  sorry

/-- The theorem stating the minimum number of unit cubes required -/
theorem min_unit_cubes_correct (n : ℕ) :
  min_unit_cubes n = minimum_unit_cubes_in_dissection (2 * n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unit_cubes_correct_l1342_134201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_l1342_134295

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones -/
structure IntersectingCones where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Predicate to check if a sphere fits in the intersecting cones -/
def sphere_fits_in_cones (ic : IntersectingCones) (s : Sphere) : Prop :=
  sorry -- The actual implementation would go here

/-- Theorem stating the maximum squared radius of a sphere fitting in the intersecting cones -/
theorem max_sphere_radius_squared (ic : IntersectingCones) 
  (h1 : ic.cone1 = ic.cone2)
  (h2 : ic.cone1.baseRadius = 4)
  (h3 : ic.cone1.height = 10)
  (h4 : ic.intersectionDistance = 4) :
  (∃ (r : ℝ), r^2 = 144/29 ∧ 
    ∀ (s : ℝ), (∃ (sphere : Sphere), sphere.radius = s ∧ 
      sphere_fits_in_cones ic sphere) → s^2 ≤ 144/29) :=
by
  sorry

#check max_sphere_radius_squared

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_squared_l1342_134295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_to_decimal_l1342_134221

theorem binary_to_decimal (b : List Bool) :
  (List.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0 (List.enum b.reverse)) =
  (1 * 2^6 + 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) := by
  sorry

#check binary_to_decimal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_to_decimal_l1342_134221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_eq_two_implies_expression_eq_neg_one_third_l1342_134263

theorem tan_alpha_eq_two_implies_expression_eq_neg_one_third (α : ℝ) 
  (h : Real.tan α = 2) : 
  (Real.cos (π + α) + Real.cos (π / 2 - α)) / (Real.sin (-α) + Real.cos (α - π)) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_eq_two_implies_expression_eq_neg_one_third_l1342_134263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_areas_l1342_134271

theorem rectangle_areas (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ a ≤ 20 ∧ b ≤ 20 ∧ 2 * (a + b) = 24 →
  a * b ∈ ({11, 20, 27, 32, 35, 36} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_areas_l1342_134271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_for_a_8_range_of_a_for_inequality_l1342_134224

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - sin x / (cos x)^3

-- Part 1: Monotonicity for a = 8
theorem monotonicity_for_a_8 :
  ∀ x ∈ Set.Ioo 0 (π/2),
    (x < π/4 → (deriv (f 8)) x > 0) ∧
    (x > π/4 → (deriv (f 8)) x < 0) := by
  sorry

-- Part 2: Range of a for f(x) < sin(2x)
theorem range_of_a_for_inequality :
  ∀ a : ℝ,
    (∀ x ∈ Set.Ioo 0 (π/2), f a x < sin (2*x)) ↔ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_for_a_8_range_of_a_for_inequality_l1342_134224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_is_rectangle_is_minor_premise_l1342_134288

-- Define the basic shapes
structure Rectangle : Type
structure Square : Type
structure Parallelogram : Type

-- Define the relationships between shapes
axiom rectangle_is_parallelogram : Rectangle → Parallelogram
axiom square_is_rectangle : Square → Rectangle
axiom square_is_parallelogram : Square → Parallelogram

-- Define the structure of a syllogism
structure Syllogism where
  major_premise : Prop
  minor_premise : Prop
  conclusion : Prop

-- Define our specific syllogism
def our_syllogism : Syllogism :=
  { major_premise := ∀ r : Rectangle, ∃ _p : Parallelogram, rectangle_is_parallelogram r = _p
  , minor_premise := ∀ s : Square, ∃ _r : Rectangle, square_is_rectangle s = _r
  , conclusion := ∀ s : Square, ∃ _p : Parallelogram, square_is_parallelogram s = _p
  }

-- Theorem stating that "A square is a rectangle" is the minor premise
theorem square_is_rectangle_is_minor_premise :
  our_syllogism.minor_premise = (∀ s : Square, ∃ _r : Rectangle, square_is_rectangle s = _r) := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_is_rectangle_is_minor_premise_l1342_134288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_tank_radius_l1342_134220

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

theorem truck_tank_radius 
  (stationaryTank : Cylinder)
  (truckTank : Cylinder)
  (oilLevelDrop : ℝ)
  (h1 : stationaryTank.radius = 100)
  (h2 : stationaryTank.height = 25)
  (h3 : truckTank.height = 10)
  (h4 : oilLevelDrop = 0.049)
  (h5 : cylinderVolume { radius := stationaryTank.radius, height := oilLevelDrop } = 
        cylinderVolume truckTank) : 
  truckTank.radius = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_tank_radius_l1342_134220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_puzzle_l1342_134297

/-- Represents the types of people on the island -/
inductive PersonType
| Good
| Bad

/-- Represents the gender of people on the island -/
inductive Gender
| Boy
| Girl

/-- Represents a person on the island -/
structure Person where
  name : String
  type : PersonType
  gender : Gender

/-- Function to determine if a statement is true based on the person's type -/
def isTruthful (p : Person) (statement : Prop) : Prop :=
  match p.type with
  | PersonType.Good => statement
  | PersonType.Bad => ¬statement

/-- Ali's statement: "We are bad" -/
def aliStatement (ali : Person) (bali : Person) : Prop :=
  ali.type = PersonType.Bad ∧ bali.type = PersonType.Bad

/-- Bali's statement: "We are boys" -/
def baliStatement (ali : Person) (bali : Person) : Prop :=
  ali.gender = Gender.Boy ∧ bali.gender = Gender.Boy

theorem island_puzzle :
  ∃ (ali bali : Person),
    ali.name = "Ali" ∧
    bali.name = "Bali" ∧
    isTruthful ali (aliStatement ali bali) ∧
    isTruthful bali (baliStatement ali bali) ∧
    ali.type = PersonType.Bad ∧
    ali.gender = Gender.Boy ∧
    bali.type = PersonType.Good ∧
    bali.gender = Gender.Boy :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_puzzle_l1342_134297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_parallel_l1342_134265

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : Fin 3 → ℝ) : Prop := ∃ k : ℝ, v = k • w

/-- Two non-zero vectors are parallel if they have the same or opposite direction -/
def parallel (v w : Fin 3 → ℝ) : Prop := v ≠ 0 ∧ w ≠ 0 ∧ ∃ k : ℝ, k ≠ 0 ∧ v = k • w

/-- Theorem: Collinear vectors are equivalent to parallel vectors -/
theorem collinear_iff_parallel (v w : Fin 3 → ℝ) : collinear v w ↔ parallel v w ∨ (v = 0 ∨ w = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_iff_parallel_l1342_134265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1342_134244

/-- Given a function f: ℝ → ℝ and a constant T > 0 satisfying certain periodicity and symmetry conditions, 
    prove that f(x) = f(T-x) for all x ∈ ℝ. -/
theorem function_symmetry 
  (f : ℝ → ℝ) 
  (T : ℝ) 
  (h_pos : T > 0)
  (h_periodic : ∀ x, f (x + 2*T) = f x)
  (h_sym1 : ∀ x, T/2 ≤ x → x ≤ T → f x = f (T-x))
  (h_sym2 : ∀ x, T ≤ x → x ≤ 3*T/2 → f x = -f (x-T))
  (h_sym3 : ∀ x, 3*T/2 ≤ x → x ≤ 2*T → f x = -f (2*T-x)) :
  ∀ x, f x = f (T-x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_symmetry_l1342_134244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olgas_grandchildren_l1342_134237

/-- Represents the number of sons each of Grandma Olga's daughters has -/
def sons_per_daughter : ℕ → Prop := λ _ => True

/-- Grandma Olga has 3 daughters -/
def num_daughters : ℕ := 3

/-- Grandma Olga has 3 sons -/
def num_sons : ℕ := 3

/-- Each of Grandma Olga's sons has 5 daughters -/
def daughters_per_son : ℕ := 5

/-- Grandma Olga has a total of 33 grandchildren -/
def total_grandchildren : ℕ := 33

theorem olgas_grandchildren (x : ℕ) : 
  sons_per_daughter x → 
  x * num_daughters + num_sons * daughters_per_son = total_grandchildren → 
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olgas_grandchildren_l1342_134237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1342_134216

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Represents that four points form a rectangle -/
def IsRectangle (A B C D : ℝ × ℝ) : Prop := sorry

/-- Calculates the area of a triangle given three points -/
def TriangleArea (A B C : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The eccentricity of a hyperbola with specific properties is √2 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (M N : ℝ × ℝ) -- Points on the asymptote
  (F₁ F₂ : ℝ × ℝ) -- Foci of the hyperbola
  (A : ℝ × ℝ) -- Vertex of the hyperbola
  (hM : M.1^2 / h.a^2 - M.2^2 / h.b^2 = 1) -- M is on the hyperbola
  (hN : N.1^2 / h.a^2 - N.2^2 / h.b^2 = 1) -- N is on the hyperbola
  (hRect : IsRectangle M F₁ N F₂) -- MF₁NF₂ is a rectangle
  (hArea : TriangleArea A M N = (1/2) * (F₁.1 - h.a)^2) -- Area of △AMN is 1/2c²
  : eccentricity h = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1342_134216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_pile_new_height_l1342_134207

/-- Represents a conical sand pile -/
structure SandPile where
  diameter : ℝ
  height : ℝ

/-- Calculates the volume of a conical sand pile -/
noncomputable def volume (pile : SandPile) : ℝ :=
  (1 / 3) * Real.pi * (pile.diameter / 2) ^ 2 * pile.height

/-- Theorem: New height of sand pile after adding sand -/
theorem sand_pile_new_height (initial_pile : SandPile)
  (h_diameter : initial_pile.diameter = 10)
  (h_height : initial_pile.height = 0.6 * initial_pile.diameter)
  (added_sand : ℝ)
  (h_added_sand : added_sand = 2) :
  ∃ (new_pile : SandPile),
    new_pile.diameter = initial_pile.diameter ∧
    volume new_pile = volume initial_pile + added_sand ∧
    new_pile.height = 6 + 6 / (25 * Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_pile_new_height_l1342_134207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1342_134282

/-- The range of a given the specified conditions -/
theorem range_of_a (p q : Prop) (a : ℝ) : 
  (∀ x : ℝ, x^2 + x + a > 0) →  -- Domain of log₀.₅(x² + x + a) is ℝ
  (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) →  -- x² - 2ax + 1 ≤ 0 has solutions in ℝ
  (p ↔ ∀ x : ℝ, x^2 + x + a > 0) →  -- p is true iff domain condition holds
  (q ↔ ∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) →  -- q is true iff inequality condition holds
  (p ≠ q) →  -- Either p or q is true, but not both
  a ∈ Set.union (Set.Ioo (1/4 : ℝ) 1) (Set.Iic (-1 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1342_134282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dennis_teaching_years_l1342_134230

/-- Represents the number of years a person has taught history. -/
def TeachingYears : Type := ℕ

/-- The total number of years Virginia, Adrienne, and Dennis have taught combined. -/
def TotalYears : ℕ := 75

theorem dennis_teaching_years 
  (virginia adrienne dennis : ℕ)
  (h1 : virginia + adrienne + dennis = TotalYears)
  (h2 : virginia = adrienne + 9)
  (h3 : dennis = virginia + 9) :
  dennis = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dennis_teaching_years_l1342_134230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_in_cube_equation_l1342_134214

theorem smallest_z_in_cube_equation : 
  ∀ w x y z : ℕ,
  w^3 + x^3 + y^3 = z^3 →
  w < x ∧ x < y ∧ y < z →
  (∃ k : ℕ, w = k^3) ∧ (∃ k : ℕ, x = k^3) ∧ (∃ k : ℕ, y = k^3) ∧ (∃ k : ℕ, z = k^3) →
  w^3 + 1 = x^3 ∧ x^3 + 1 = y^3 ∧ y^3 + 1 = z^3 →
  z ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_in_cube_equation_l1342_134214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l1342_134227

theorem power_of_three (y : ℝ) (h : (3 : ℝ)^y = 81) : (3 : ℝ)^(y+3) = 2187 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_l1342_134227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1342_134204

/-- The circle with center (2,2) and radius √2 -/
def myCircle : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 2)^2 = 2}

/-- The line x - y - 4 = 0 -/
def myLine : Set (ℝ × ℝ) := {p | p.1 - p.2 - 4 = 0}

/-- The distance from a point to the line -/
noncomputable def distToLine (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 - 4| / Real.sqrt 2

theorem max_distance_circle_to_line :
  (⨆ p ∈ myCircle, distToLine p) = 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_circle_to_line_l1342_134204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_proof_l1342_134219

theorem book_pages_proof (x : ℚ) : 
  x > 0 →
  let remaining_after_day1 := x - (x / 6 + 10)
  let remaining_after_day2 := remaining_after_day1 - (remaining_after_day1 / 5 + 20)
  let remaining_after_day3 := remaining_after_day2 - (remaining_after_day2 / 4 + 25)
  remaining_after_day3 = 50 →
  x = 192 := by
  intro h_positive
  intro h_final_pages
  -- The proof steps would go here
  sorry

#check book_pages_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_pages_proof_l1342_134219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_condition_l1342_134211

-- Define the function f(x) = x - a*ln(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 - a / x

-- Theorem statement
theorem perpendicular_tangents_condition (a : ℝ) :
  (∃ x y, 1 < x ∧ x < 6 ∧ 1 < y ∧ y < 6 ∧ x ≠ y ∧
    (f_derivative a x) * (f_derivative a y) = -1) ↔
  (3 < a ∧ a < 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_tangents_condition_l1342_134211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_hyperbola_equation_with_asymptotes_l1342_134283

/-- Given a hyperbola passing through the point (6, √3) with asymptotes y = ± (1/3)x,
    prove that its equation is x²/9 - y² = 1 -/
theorem hyperbola_equation (h : Set (ℝ × ℝ)) 
  (passes_through : (6, Real.sqrt 3) ∈ h)
  (asymptotes : Set (ℝ × ℝ) → Prop) :
  asymptotes h →
  (∀ (x y : ℝ), (x, y) ∈ h ↔ x^2 / 9 - y^2 = 1) := by
  sorry

/-- Definition of asymptotes for this specific hyperbola -/
def has_asymptotes (h : Set (ℝ × ℝ)) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x y, (x, y) ∈ h → 
    (|x| > M → |y - (1/3) * x| < ε ∨ |y + (1/3) * x| < ε)

/-- The main theorem restated with the specific asymptote definition -/
theorem hyperbola_equation_with_asymptotes (h : Set (ℝ × ℝ)) :
  ((6, Real.sqrt 3) ∈ h) →
  (has_asymptotes h) →
  (∀ (x y : ℝ), (x, y) ∈ h ↔ x^2 / 9 - y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_hyperbola_equation_with_asymptotes_l1342_134283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1342_134251

noncomputable def f (x : ℝ) := Real.sqrt x - 2 + Real.log x / Real.log 2

theorem zero_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l1342_134251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_time_calculation_l1342_134209

/-- The time taken for the second train to pass a fixed point -/
noncomputable def second_train_time (train_length : ℝ) (first_train_time : ℝ) (crossing_time : ℝ) : ℝ :=
  2 * train_length / (2 * train_length / crossing_time - train_length / first_train_time)

/-- Theorem stating the time taken for the second train to pass a fixed point -/
theorem second_train_time_calculation : 
  second_train_time 120 10 (40/3) = 20 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_time_calculation_l1342_134209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_crates_pigeonhole_l1342_134292

theorem apple_crates_pigeonhole (total_crates min_apples max_apples : ℕ) 
  (h_total : total_crates = 128)
  (h_min : min_apples = 120)
  (h_max : max_apples = 144)
  (h_range : ∀ crate, crate ∈ Finset.Icc min_apples max_apples) :
  ∃ (n : ℕ), n ≥ 6 ∧ ∃ (apple_count : ℕ), 
    (apple_count ∈ Finset.Icc min_apples max_apples) ∧
    (Finset.filter (λ crate => crate = apple_count) (Finset.range total_crates)).card ≥ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_crates_pigeonhole_l1342_134292
