import Mathlib

namespace NUMINAMATH_CALUDE_racing_game_cost_l1858_185822

/-- The cost of the racing game given the total spent and the cost of the basketball game -/
theorem racing_game_cost (total_spent basketball_cost : ℚ) 
  (h1 : total_spent = 9.43)
  (h2 : basketball_cost = 5.2) : 
  total_spent - basketball_cost = 4.23 := by
  sorry

end NUMINAMATH_CALUDE_racing_game_cost_l1858_185822


namespace NUMINAMATH_CALUDE_bug_distance_is_28_l1858_185865

def bug_crawl (start end1 end2 end3 : Int) : Int :=
  |end1 - start| + |end2 - end1| + |end3 - end2|

theorem bug_distance_is_28 :
  bug_crawl 3 (-4) 8 (-1) = 28 := by
  sorry

end NUMINAMATH_CALUDE_bug_distance_is_28_l1858_185865


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l1858_185860

theorem cos_alpha_plus_pi_fourth (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is an obtuse angle
  (h2 : Real.sin (α - 3*π/4) = 3/5) :
  Real.cos (α + π/4) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_fourth_l1858_185860


namespace NUMINAMATH_CALUDE_hyperbola_center_l1858_185834

/-- The center of a hyperbola is the midpoint of its foci -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  f1 = (3, 2) ∧ f2 = (11, 6) →
  center = (7, 4) := by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l1858_185834


namespace NUMINAMATH_CALUDE_sphere_cylinder_ratio_l1858_185800

theorem sphere_cylinder_ratio (R : ℝ) (h : R > 0) : 
  let sphere_volume := (4 / 3) * Real.pi * R^3
  let cylinder_volume := 2 * Real.pi * R^3
  let empty_space := cylinder_volume - sphere_volume
  let total_empty_space := 5 * empty_space
  let total_occupied_space := 5 * sphere_volume
  (total_empty_space / total_occupied_space) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_cylinder_ratio_l1858_185800


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1858_185856

theorem inequality_solution_set (x : ℝ) :
  (-x^2 + 3*x - 2 > 0) ↔ (1 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1858_185856


namespace NUMINAMATH_CALUDE_function_upper_bound_l1858_185857

/-- Given a function f(x) = ax - x ln x - a, prove that if f(x) ≤ 0 for all x ≥ 2, 
    then a ≤ 2ln 2 -/
theorem function_upper_bound (a : ℝ) : 
  (∀ x : ℝ, x ≥ 2 → a * x - x * Real.log x - a ≤ 0) → 
  a ≤ 2 * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_function_upper_bound_l1858_185857


namespace NUMINAMATH_CALUDE_problem_solution_l1858_185859

def f (a : ℝ) (x : ℝ) : ℝ := |3*x - a| - 2*|x - 1|

theorem problem_solution :
  (∀ x : ℝ, f (-3) x > 1 ↔ (x < -6 ∨ x > 1)) ∧
  (∃ x : ℝ, f a x ≥ 6 + |x - 1| → (a ≥ 9 ∨ a < -3)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1858_185859


namespace NUMINAMATH_CALUDE_train_platform_time_l1858_185873

theorem train_platform_time (l t T : ℝ) (v : ℝ) (h1 : v > 0) (h2 : l > 0) (h3 : t > 0) :
  v = l / t →
  v = (l + 2.5 * l) / T →
  T = 3.5 * t := by
sorry

end NUMINAMATH_CALUDE_train_platform_time_l1858_185873


namespace NUMINAMATH_CALUDE_smallest_m_is_24_l1858_185883

/-- The set of complex numbers with real part between 1/2 and 2/3 -/
def S : Set ℂ :=
  {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ 2/3}

/-- Definition of the property we want to prove for m -/
def has_nth_root_of_unity (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z ∈ S, z^n = 1

/-- The theorem stating that 24 is the smallest positive integer satisfying the property -/
theorem smallest_m_is_24 :
  has_nth_root_of_unity 24 ∧ ∀ m : ℕ, 0 < m → m < 24 → ¬has_nth_root_of_unity m :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_24_l1858_185883


namespace NUMINAMATH_CALUDE_decimal_division_equals_forty_l1858_185826

theorem decimal_division_equals_forty : (0.24 : ℚ) / (0.006 : ℚ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_equals_forty_l1858_185826


namespace NUMINAMATH_CALUDE_grape_candy_count_l1858_185878

/-- Represents the number of cherry candies -/
def cherry : ℕ := sorry

/-- Represents the number of grape candies -/
def grape : ℕ := 3 * cherry

/-- Represents the number of apple candies -/
def apple : ℕ := 2 * grape

/-- The cost of each candy in cents -/
def cost_per_candy : ℕ := 250

/-- The total cost of all candies in cents -/
def total_cost : ℕ := 20000

theorem grape_candy_count :
  grape = 24 ∧
  cherry + grape + apple = total_cost / cost_per_candy :=
sorry

end NUMINAMATH_CALUDE_grape_candy_count_l1858_185878


namespace NUMINAMATH_CALUDE_last_three_digits_of_8_to_104_l1858_185866

theorem last_three_digits_of_8_to_104 : 8^104 ≡ 984 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_8_to_104_l1858_185866


namespace NUMINAMATH_CALUDE_min_f_1998_l1858_185881

/-- A function satisfying the given property -/
def SpecialFunction (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, f (n^2 * f m) = m * (f n)^2

/-- The theorem stating the minimum value of f(1998) -/
theorem min_f_1998 (f : ℕ → ℕ) (hf : SpecialFunction f) : 
  (∀ g : ℕ → ℕ, SpecialFunction g → f 1998 ≤ g 1998) → f 1998 = 120 :=
sorry

end NUMINAMATH_CALUDE_min_f_1998_l1858_185881


namespace NUMINAMATH_CALUDE_second_train_speed_l1858_185817

/-- The speed of the first train in km/h -/
def speed_first : ℝ := 40

/-- The time difference between the departure of the two trains in hours -/
def time_difference : ℝ := 1

/-- The distance at which the two trains meet, in km -/
def meeting_distance : ℝ := 200

/-- The speed of the second train in km/h -/
def speed_second : ℝ := 50

theorem second_train_speed :
  speed_second = meeting_distance / (meeting_distance / speed_first - time_difference) :=
by sorry

end NUMINAMATH_CALUDE_second_train_speed_l1858_185817


namespace NUMINAMATH_CALUDE_right_triangular_prism_volume_l1858_185855

/-- Given a right triangular prism where:
    - The lateral edge is equal to the height of its base
    - The area of the cross-section passing through this lateral edge and the height of the base is Q
    Prove that the volume of the prism is Q √(3Q) -/
theorem right_triangular_prism_volume (Q : ℝ) (Q_pos : Q > 0) :
  ∃ (V : ℝ), V = Q * Real.sqrt (3 * Q) ∧
  (∃ (a h : ℝ) (a_pos : a > 0) (h_pos : h > 0),
    h = a * Real.sqrt 5 / 2 ∧
    Q = a * Real.sqrt 5 / 2 * h ∧
    V = Real.sqrt 3 / 4 * a^2 * h) :=
by sorry

end NUMINAMATH_CALUDE_right_triangular_prism_volume_l1858_185855


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l1858_185813

theorem pie_crust_flour_calculation (initial_crusts : ℕ) (initial_flour_per_crust : ℚ) 
  (new_crusts : ℕ) (h1 : initial_crusts = 40) (h2 : initial_flour_per_crust = 1/8) 
  (h3 : new_crusts = 25) :
  let total_flour := initial_crusts * initial_flour_per_crust
  let new_flour_per_crust := total_flour / new_crusts
  new_flour_per_crust = 1/5 := by
sorry

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l1858_185813


namespace NUMINAMATH_CALUDE_parabola_directrix_p_l1858_185833

/-- A parabola with equation y^2 = 2px and directrix x = -1 has p = 2 -/
theorem parabola_directrix_p (y x p : ℝ) : 
  (∀ y, y^2 = 2*p*x) →  -- Parabola equation
  (x = -1)             -- Directrix equation
  → p = 2 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_p_l1858_185833


namespace NUMINAMATH_CALUDE_floor_equation_solution_l1858_185896

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3*x⌋ + 1/3⌋ = ⌊x + 3⌋) ↔ (4/3 ≤ x ∧ x < 5/3) := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l1858_185896


namespace NUMINAMATH_CALUDE_certain_number_is_30_l1858_185886

theorem certain_number_is_30 (x : ℝ) : 0.5 * x = 0.1667 * x + 10 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_30_l1858_185886


namespace NUMINAMATH_CALUDE_solution_set_2x_plus_y_eq_9_l1858_185828

theorem solution_set_2x_plus_y_eq_9 :
  {(x, y) : ℕ × ℕ | 2 * x + y = 9} = {(0, 9), (1, 7), (2, 5), (3, 3), (4, 1)} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_2x_plus_y_eq_9_l1858_185828


namespace NUMINAMATH_CALUDE_tan_series_equality_l1858_185831

theorem tan_series_equality (x : ℝ) (h : |Real.tan x| < 1) :
  8.407 * ((1 - Real.tan x)⁻¹) / ((1 + Real.tan x)⁻¹) = 1 + Real.sin (2 * x) ↔
  ∃ k : ℤ, x = k * Real.pi := by sorry

end NUMINAMATH_CALUDE_tan_series_equality_l1858_185831


namespace NUMINAMATH_CALUDE_min_perimeter_ABQP_l1858_185821

/-- The minimum perimeter of quadrilateral ABQP -/
theorem min_perimeter_ABQP :
  let A : ℝ × ℝ := (6, 5)
  let B : ℝ × ℝ := (10, 2)
  let M : Set (ℝ × ℝ) := {(x, y) | y = x ∧ x ≥ 0}
  let N : Set (ℝ × ℝ) := {(x, y) | y = 0 ∧ x ≥ 0}
  let P : Set (ℝ × ℝ) := M
  let Q : Set (ℝ × ℝ) := N
  ∀ p ∈ P, ∀ q ∈ Q,
    Real.sqrt ((p.1 - A.1)^2 + (p.2 - A.2)^2) +
    Real.sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2) +
    Real.sqrt ((B.1 - q.1)^2 + (B.2 - q.2)^2) +
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥
    6.5 + Real.sqrt 89 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_ABQP_l1858_185821


namespace NUMINAMATH_CALUDE_decade_cost_l1858_185890

/-- Vivian's annual car insurance cost in dollars -/
def annual_cost : ℕ := 2000

/-- Number of years in a decade -/
def decade : ℕ := 10

/-- Theorem: Vivian's total car insurance cost over a decade -/
theorem decade_cost : annual_cost * decade = 20000 := by
  sorry

end NUMINAMATH_CALUDE_decade_cost_l1858_185890


namespace NUMINAMATH_CALUDE_shopping_expenditure_l1858_185867

theorem shopping_expenditure (x : ℝ) 
  (emma_spent : x > 0)
  (elsa_spent : ℝ → ℝ)
  (elizabeth_spent : ℝ → ℝ)
  (elsa_condition : elsa_spent x = 2 * x)
  (elizabeth_condition : elizabeth_spent x = 4 * elsa_spent x)
  (total_spent : x + elsa_spent x + elizabeth_spent x = 638) :
  x = 58 := by
sorry

end NUMINAMATH_CALUDE_shopping_expenditure_l1858_185867


namespace NUMINAMATH_CALUDE_triangle_count_l1858_185835

/-- A triangle with integral side lengths. -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Check if the given integers form a valid triangle. -/
def is_valid_triangle (t : IntTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- Check if the triangle has a perimeter of 9. -/
def has_perimeter_9 (t : IntTriangle) : Prop :=
  t.a + t.b + t.c = 9

/-- Two triangles are considered different if they are not congruent. -/
def are_different (t1 t2 : IntTriangle) : Prop :=
  ¬(t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∧
  ¬(t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∧
  ¬(t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

theorem triangle_count : 
  ∃ (t1 t2 : IntTriangle), 
    is_valid_triangle t1 ∧ 
    is_valid_triangle t2 ∧ 
    has_perimeter_9 t1 ∧ 
    has_perimeter_9 t2 ∧ 
    are_different t1 t2 ∧
    (∀ (t3 : IntTriangle), 
      is_valid_triangle t3 → 
      has_perimeter_9 t3 → 
      (t3 = t1 ∨ t3 = t2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l1858_185835


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l1858_185840

theorem parabola_vertex_on_x_axis (a : ℝ) : 
  (∃ x : ℝ, x^2 - (a + 2)*x + 9 = 0 ∧ 
   ∀ y : ℝ, y^2 - (a + 2)*y + 9 ≥ x^2 - (a + 2)*x + 9) →
  a = 4 ∨ a = -8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l1858_185840


namespace NUMINAMATH_CALUDE_product_of_roots_l1858_185842

theorem product_of_roots (x : ℝ) : 
  let a : ℝ := 24
  let b : ℝ := 36
  let c : ℝ := -648
  let equation := a * x^2 + b * x + c
  let root_product := c / a
  equation = 0 → root_product = -27 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l1858_185842


namespace NUMINAMATH_CALUDE_max_square_pen_area_l1858_185808

def fencing_length : ℝ := 64

def square_pen_area (side_length : ℝ) : ℝ := side_length ^ 2

def perimeter_constraint (side_length : ℝ) : Prop := 4 * side_length = fencing_length

theorem max_square_pen_area :
  ∃ (side_length : ℝ), perimeter_constraint side_length ∧
    ∀ (x : ℝ), perimeter_constraint x → square_pen_area x ≤ square_pen_area side_length ∧
    square_pen_area side_length = 256 :=
  sorry

end NUMINAMATH_CALUDE_max_square_pen_area_l1858_185808


namespace NUMINAMATH_CALUDE_medication_frequency_l1858_185825

/-- The number of times Kara takes her medication per day -/
def medication_times_per_day : ℕ := sorry

/-- The amount of water Kara drinks with each medication dose in ounces -/
def water_per_dose : ℕ := 4

/-- The number of days Kara followed her medication schedule -/
def days_followed : ℕ := 14

/-- The number of doses Kara missed in the two-week period -/
def doses_missed : ℕ := 2

/-- The total amount of water Kara drank with her medication over two weeks in ounces -/
def total_water_consumed : ℕ := 160

theorem medication_frequency :
  medication_times_per_day = 3 :=
by
  have h1 : water_per_dose * (days_followed * medication_times_per_day - doses_missed) = total_water_consumed := sorry
  sorry

end NUMINAMATH_CALUDE_medication_frequency_l1858_185825


namespace NUMINAMATH_CALUDE_strawberry_percentage_l1858_185820

def total_weight : ℝ := 20
def apple_weight : ℝ := 4
def orange_weight : ℝ := 2
def grape_weight : ℝ := 4
def banana_weight : ℝ := 1
def pineapple_weight : ℝ := 3

def strawberry_weight : ℝ := total_weight - (apple_weight + orange_weight + grape_weight + banana_weight + pineapple_weight)

theorem strawberry_percentage : (strawberry_weight / total_weight) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_percentage_l1858_185820


namespace NUMINAMATH_CALUDE_olympic_tournament_winners_l1858_185837

/-- Represents an Olympic system tournament -/
structure OlympicTournament where
  rounds : ℕ
  initialParticipants : ℕ
  winnersEachRound : List ℕ

/-- Checks if the tournament is valid -/
def isValidTournament (t : OlympicTournament) : Prop :=
  t.rounds > 0 ∧
  t.initialParticipants = 2^t.rounds ∧
  t.winnersEachRound.length = t.rounds ∧
  ∀ i, i ∈ t.winnersEachRound → i = t.initialParticipants / (2^(t.winnersEachRound.indexOf i + 1))

/-- Calculates the number of participants who won more games than they lost -/
def participantsWithMoreWins (t : OlympicTournament) : ℕ :=
  t.initialParticipants / 4

theorem olympic_tournament_winners (t : OlympicTournament) 
  (h1 : isValidTournament t) 
  (h2 : t.rounds = 6) : 
  participantsWithMoreWins t = 16 := by
  sorry

#check olympic_tournament_winners

end NUMINAMATH_CALUDE_olympic_tournament_winners_l1858_185837


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l1858_185830

theorem rhombus_diagonal (side_length square_area rhombus_area diagonal1 diagonal2 : ℝ) :
  square_area = side_length * side_length →
  rhombus_area = square_area →
  rhombus_area = (diagonal1 * diagonal2) / 2 →
  side_length = 8 →
  diagonal1 = 16 →
  diagonal2 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l1858_185830


namespace NUMINAMATH_CALUDE_fixed_point_exponential_l1858_185880

theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∀ x : ℝ, (a^(x - 1) + 1 = 2) ↔ (x = 1) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_l1858_185880


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1858_185832

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 1375 →
  divisor = 66 →
  quotient = 20 →
  dividend = divisor * quotient + remainder →
  remainder = 55 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1858_185832


namespace NUMINAMATH_CALUDE_water_fountain_trips_l1858_185870

/-- The number of trips to the water fountain -/
def number_of_trips (total_distance : ℕ) (distance_to_fountain : ℕ) : ℕ :=
  total_distance / distance_to_fountain

theorem water_fountain_trips : 
  number_of_trips 120 30 = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_fountain_trips_l1858_185870


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1858_185812

theorem yellow_balls_count (total : ℕ) (red yellow green : ℕ) : 
  total = 68 →
  2 * red = yellow →
  3 * green = 4 * yellow →
  red + yellow + green = total →
  yellow = 24 := by
sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1858_185812


namespace NUMINAMATH_CALUDE_minimum_value_of_f_minimum_value_case1_minimum_value_case2_l1858_185869

-- Define the function f(x) = x^2 - 2x
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the theorem
theorem minimum_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) a, f x ≥ f a ∧ -2 < a ∧ a ≤ 1) ∨
  (∀ x ∈ Set.Icc (-2) a, f x ≥ -1 ∧ a > 1) := by
  sorry

-- Define helper theorems for each case
theorem minimum_value_case1 (a : ℝ) (h1 : -2 < a) (h2 : a ≤ 1) :
  ∀ x ∈ Set.Icc (-2) a, f x ≥ f a := by
  sorry

theorem minimum_value_case2 (a : ℝ) (h : a > 1) :
  ∀ x ∈ Set.Icc (-2) a, f x ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_minimum_value_case1_minimum_value_case2_l1858_185869


namespace NUMINAMATH_CALUDE_ellipse_properties_l1858_185811

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  sum_focal_distances : ℝ → ℝ → ℝ
  eccentricity : ℝ
  focal_sum_eq : ∀ x y, x^2/a^2 + y^2/b^2 = 1 → sum_focal_distances x y = 2 * Real.sqrt 3
  ecc_eq : eccentricity = Real.sqrt 3 / 3

/-- Point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h : x^2/E.a^2 + y^2/E.b^2 = 1

/-- Theorem about the standard form of the ellipse and slope product -/
theorem ellipse_properties (E : Ellipse) :
  (E.a^2 = 3 ∧ E.b^2 = 2) ∧
  ∀ (P : PointOnEllipse E) (Q : PointOnEllipse E),
    P.x = 3 →
    (Q.x - 1) * (P.y - 0) + (Q.y - 0) * (P.x - 1) = 0 →
    (Q.y / Q.x) * ((Q.y - P.y) / (Q.x - P.x)) = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1858_185811


namespace NUMINAMATH_CALUDE_line_equation_proof_l1858_185802

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (p : Point) :
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = 3 →
  p.x = -1 ∧ p.y = 3 →
  ∃ (result_line : Line),
    result_line.a = 1 ∧ result_line.b = -2 ∧ result_line.c = 7 ∧
    pointOnLine p result_line ∧
    parallelLines given_line result_line := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1858_185802


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1858_185899

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a1 : a 1 = 2)
  (h_a3 : a 3 = 4) :
  a 7 = 16 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l1858_185899


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1858_185888

/-- Represents the number of players selected from a class using stratified sampling -/
def stratified_sample (total_players : ℕ) (class_players : ℕ) (sample_size : ℕ) : ℕ :=
  (class_players * sample_size) / total_players

theorem basketball_team_selection (class5_players class16_players class33_players : ℕ) 
  (h1 : class5_players = 6)
  (h2 : class16_players = 8)
  (h3 : class33_players = 10)
  (h4 : class5_players + class16_players + class33_players = 24)
  (sample_size : ℕ)
  (h5 : sample_size = 12) :
  stratified_sample (class5_players + class16_players + class33_players) class5_players sample_size = 3 ∧
  stratified_sample (class5_players + class16_players + class33_players) class16_players sample_size = 4 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1858_185888


namespace NUMINAMATH_CALUDE_determinant_zero_l1858_185839

theorem determinant_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![1, Real.sin (a + b), Real.sin a],
    ![Real.sin (a + b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_determinant_zero_l1858_185839


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1858_185841

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  ∃ (min_val : ℝ), min_val = 4 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 1/x' + 1/y' = 1 →
    1/(x' - 1) + 4/(y' - 1) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1858_185841


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l1858_185884

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 14 = 0

-- Define the line
def line (x y : ℝ) : Prop := x + y - 2 = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l1858_185884


namespace NUMINAMATH_CALUDE_store_products_l1858_185871

theorem store_products (big_box_capacity small_box_capacity total_products : ℕ) 
  (h1 : big_box_capacity = 50)
  (h2 : small_box_capacity = 40)
  (h3 : total_products = 212) :
  ∃ (big_boxes small_boxes : ℕ), 
    big_boxes * big_box_capacity + small_boxes * small_box_capacity = total_products :=
by sorry

end NUMINAMATH_CALUDE_store_products_l1858_185871


namespace NUMINAMATH_CALUDE_simplify_expression_l1858_185853

theorem simplify_expression (a b : ℝ) :
  3 * a * (3 * a^3 + 2 * a^2) - 2 * a^2 * (b^2 + 1) = 9 * a^4 + 6 * a^3 - 2 * a^2 * b^2 - 2 * a^2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1858_185853


namespace NUMINAMATH_CALUDE_sand_pile_removal_l1858_185805

theorem sand_pile_removal (initial_weight : ℚ) (first_removal : ℚ) (second_removal : ℚ)
  (h1 : initial_weight = 8 / 3)
  (h2 : first_removal = 1 / 4)
  (h3 : second_removal = 5 / 6) :
  first_removal + second_removal = 13 / 12 := by
sorry

end NUMINAMATH_CALUDE_sand_pile_removal_l1858_185805


namespace NUMINAMATH_CALUDE_moving_sidewalk_speed_l1858_185838

/-- The speed of a moving sidewalk given a child's running parameters -/
theorem moving_sidewalk_speed
  (child_speed : ℝ)
  (with_distance : ℝ)
  (with_time : ℝ)
  (against_distance : ℝ)
  (against_time : ℝ)
  (h1 : child_speed = 74)
  (h2 : with_distance = 372)
  (h3 : with_time = 4)
  (h4 : against_distance = 165)
  (h5 : against_time = 3)
  : ∃ (sidewalk_speed : ℝ),
    sidewalk_speed = 19 ∧
    with_distance = (child_speed + sidewalk_speed) * with_time ∧
    against_distance = (child_speed - sidewalk_speed) * against_time :=
by sorry

end NUMINAMATH_CALUDE_moving_sidewalk_speed_l1858_185838


namespace NUMINAMATH_CALUDE_pascal_third_element_51st_row_l1858_185894

/-- The number of elements in the nth row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

/-- The kth element in the nth row of Pascal's triangle -/
def pascal_element (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_third_element_51st_row : 
  pascal_element 51 2 = 1275 :=
sorry

end NUMINAMATH_CALUDE_pascal_third_element_51st_row_l1858_185894


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1858_185823

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (lies_in : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (intersection_line : Plane → Plane → Line)

-- State the theorem
theorem line_perpendicular_to_plane 
  (α β : Plane) (a c : Line) :
  perpendicular_planes α β →
  lies_in a α →
  c = intersection_line α β →
  perpendicular_lines a c →
  perpendicular_line_plane a β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l1858_185823


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l1858_185824

/-- Represents a die in the cube --/
structure Die where
  sides : Fin 6 → ℕ
  sum_opposite : ∀ i : Fin 3, sides i + sides (i + 3) = 7

/-- Represents the 4x4x4 cube made of dice --/
def Cube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Calculates the sum of visible faces on the large cube --/
def visible_sum (c : Cube) : ℕ := sorry

/-- Theorem stating the smallest possible sum of visible faces --/
theorem smallest_visible_sum (c : Cube) : 
  visible_sum c ≥ 136 ∧ ∃ c', visible_sum c' = 136 := by sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l1858_185824


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l1858_185810

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 4) 
  (h_a4 : a 4 = 2) : 
  a 8 = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l1858_185810


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1858_185858

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4}

-- Define set A
def A : Finset Nat := {1, 3}

-- Theorem stating that the complement of A in U is {2, 4}
theorem complement_of_A_in_U :
  (U \ A) = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1858_185858


namespace NUMINAMATH_CALUDE_alcohol_water_mixture_ratio_l1858_185845

theorem alcohol_water_mixture_ratio 
  (p q r : ℝ) 
  (hp : p > 0) 
  (hq : q > 0) 
  (hr : r > 0) :
  let jar1_ratio := p / (p + 1)
  let jar2_ratio := q / (q + 1)
  let jar3_ratio := r / (r + 1)
  let total_alcohol := jar1_ratio + jar2_ratio + jar3_ratio
  let total_water := 1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1)
  total_alcohol / total_water = (p*q*r + p*q + p*r + q*r + p + q + r) / (p*q + p*r + q*r + p + q + r + 1) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_water_mixture_ratio_l1858_185845


namespace NUMINAMATH_CALUDE_village_language_problem_l1858_185875

theorem village_language_problem (total_population : ℕ) 
  (tamil_speakers : ℕ) (english_speakers : ℕ) (hindi_probability : ℚ) :
  total_population = 1024 →
  tamil_speakers = 720 →
  english_speakers = 562 →
  hindi_probability = 0.0859375 →
  ∃ (both_speakers : ℕ),
    both_speakers = 434 ∧
    total_population = tamil_speakers + english_speakers - both_speakers + 
      (↑total_population * hindi_probability).floor := by
  sorry

end NUMINAMATH_CALUDE_village_language_problem_l1858_185875


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1858_185849

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem smallest_dual_base_palindrome :
  ∀ n : ℕ,
    n > 5 →
    isPalindrome n 2 →
    isPalindrome n 4 →
    (∀ m : ℕ, m > 5 ∧ m < n → ¬(isPalindrome m 2 ∧ isPalindrome m 4)) →
    n = 15 :=
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l1858_185849


namespace NUMINAMATH_CALUDE_expression_equalities_l1858_185864

theorem expression_equalities :
  (-2^3 = (-2)^3) ∧ 
  (2^3 ≠ 2 * 3) ∧ 
  (-(-2)^2 ≠ (-2)^2) ∧ 
  (-3^2 ≠ 3^2) := by
sorry

end NUMINAMATH_CALUDE_expression_equalities_l1858_185864


namespace NUMINAMATH_CALUDE_smallest_constant_degenerate_triangle_l1858_185854

/-- A degenerate triangle is represented by three non-negative real numbers a, b, and c,
    where a + b = c --/
structure DegenerateTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  non_neg_a : 0 ≤ a
  non_neg_b : 0 ≤ b
  non_neg_c : 0 ≤ c
  sum_eq_c : a + b = c

/-- The smallest constant N such that (a^2 + b^2) / c^2 < N for all degenerate triangles
    is 1/2 --/
theorem smallest_constant_degenerate_triangle :
  ∃ N : ℝ, (∀ t : DegenerateTriangle, (t.a^2 + t.b^2) / t.c^2 < N) ∧
  (∀ ε > 0, ∃ t : DegenerateTriangle, (t.a^2 + t.b^2) / t.c^2 > N - ε) ∧
  N = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_constant_degenerate_triangle_l1858_185854


namespace NUMINAMATH_CALUDE_starting_lineup_count_l1858_185816

def total_players : ℕ := 20
def num_goalies : ℕ := 1
def num_forwards : ℕ := 6
def num_defenders : ℕ := 4

def starting_lineup_combinations : ℕ := 
  (total_players.choose num_goalies) * 
  ((total_players - num_goalies).choose num_forwards) * 
  ((total_players - num_goalies - num_forwards).choose num_defenders)

theorem starting_lineup_count : starting_lineup_combinations = 387889200 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l1858_185816


namespace NUMINAMATH_CALUDE_vera_doll_count_l1858_185847

theorem vera_doll_count (aida sophie vera : ℕ) : 
  aida = 2 * sophie →
  sophie = 2 * vera →
  aida + sophie + vera = 140 →
  vera = 20 :=
by sorry

end NUMINAMATH_CALUDE_vera_doll_count_l1858_185847


namespace NUMINAMATH_CALUDE_fib_1960_1988_gcd_l1858_185846

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The greatest common divisor of the 1960th and 1988th Fibonacci numbers is 317811 -/
theorem fib_1960_1988_gcd : Nat.gcd (fib 1988) (fib 1960) = 317811 := by
  sorry

end NUMINAMATH_CALUDE_fib_1960_1988_gcd_l1858_185846


namespace NUMINAMATH_CALUDE_expression_evaluation_l1858_185852

theorem expression_evaluation (x z : ℝ) (h : x = Real.sqrt z) :
  (x - 1 / x) * (Real.sqrt z + 1 / Real.sqrt z) = z - 1 / z := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1858_185852


namespace NUMINAMATH_CALUDE_boat_current_speed_ratio_l1858_185818

/-- Proves that the ratio of boat speed to current speed is 4:1 given upstream and downstream travel times -/
theorem boat_current_speed_ratio 
  (distance : ℝ) 
  (upstream_time : ℝ) 
  (downstream_time : ℝ) 
  (h_upstream : upstream_time = 6) 
  (h_downstream : downstream_time = 10) 
  (h_positive_distance : distance > 0) :
  ∃ (boat_speed current_speed : ℝ),
    boat_speed > 0 ∧ 
    current_speed > 0 ∧
    distance = upstream_time * (boat_speed - current_speed) ∧
    distance = downstream_time * (boat_speed + current_speed) ∧
    boat_speed = 4 * current_speed :=
sorry

end NUMINAMATH_CALUDE_boat_current_speed_ratio_l1858_185818


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1858_185815

theorem complex_fraction_simplification :
  ((-4 : ℂ) - 6*I) / (5 - 2*I) = -32/21 - 38/21*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1858_185815


namespace NUMINAMATH_CALUDE_fish_value_calculation_l1858_185897

/-- Calculates the total value of non-spoiled fish after sales, spoilage, and new stock arrival --/
def fish_value (initial_trout initial_bass : ℕ) 
               (sold_trout sold_bass : ℕ) 
               (trout_price bass_price : ℚ) 
               (spoil_trout_ratio spoil_bass_ratio : ℚ)
               (new_trout new_bass : ℕ) : ℚ :=
  let remaining_trout := initial_trout - sold_trout
  let remaining_bass := initial_bass - sold_bass
  let spoiled_trout := ⌊remaining_trout * spoil_trout_ratio⌋
  let spoiled_bass := ⌊remaining_bass * spoil_bass_ratio⌋
  let final_trout := remaining_trout - spoiled_trout + new_trout
  let final_bass := remaining_bass - spoiled_bass + new_bass
  final_trout * trout_price + final_bass * bass_price

/-- The theorem statement --/
theorem fish_value_calculation :
  fish_value 120 80 30 20 5 10 (1/4) (1/3) 150 50 = 1990 := by
  sorry

end NUMINAMATH_CALUDE_fish_value_calculation_l1858_185897


namespace NUMINAMATH_CALUDE_equation_solution_l1858_185861

theorem equation_solution : ∃ x : ℝ, 
  ((3^2 - 5) / (0.08 * 7 + 2)) + Real.sqrt x = 10 ∧ x = 71.2715625 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1858_185861


namespace NUMINAMATH_CALUDE_total_suit_cost_l1858_185801

/-- The cost of a suit given the following conditions:
  1. A jacket costs as much as trousers and a vest.
  2. A jacket and two pairs of trousers cost 175 dollars.
  3. Trousers and two vests cost 100 dollars. -/
def suit_cost (jacket trousers vest : ℝ) : Prop :=
  jacket = trousers + vest ∧
  jacket + 2 * trousers = 175 ∧
  trousers + 2 * vest = 100

/-- Theorem stating that the total cost of the suit is 150 dollars. -/
theorem total_suit_cost :
  ∀ (jacket trousers vest : ℝ),
    suit_cost jacket trousers vest →
    jacket + trousers + vest = 150 :=
by
  sorry

#check total_suit_cost

end NUMINAMATH_CALUDE_total_suit_cost_l1858_185801


namespace NUMINAMATH_CALUDE_angle_measure_problem_l1858_185807

theorem angle_measure_problem (angle_B angle_small_triangle : ℝ) :
  angle_B = 120 →
  angle_small_triangle = 50 →
  ∃ angle_A : ℝ,
    angle_A = 70 ∧
    angle_A + angle_small_triangle + (180 - angle_B) = 180 :=
by sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l1858_185807


namespace NUMINAMATH_CALUDE_burger_sharing_l1858_185814

theorem burger_sharing (burger_length : ℝ) (brother_fraction : ℝ) (friend1_fraction : ℝ) (friend2_fraction : ℝ) :
  burger_length = 12 →
  brother_fraction = 1/3 →
  friend1_fraction = 1/4 →
  friend2_fraction = 1/2 →
  ∃ (brother_share friend1_share friend2_share valentina_share : ℝ),
    brother_share = burger_length * brother_fraction ∧
    friend1_share = (burger_length - brother_share) * friend1_fraction ∧
    friend2_share = (burger_length - brother_share - friend1_share) * friend2_fraction ∧
    valentina_share = burger_length - brother_share - friend1_share - friend2_share ∧
    brother_share = 4 ∧
    friend1_share = 2 ∧
    friend2_share = 3 ∧
    valentina_share = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_burger_sharing_l1858_185814


namespace NUMINAMATH_CALUDE_operation_result_l1858_185806

def at_op (a b : ℝ) : ℝ := a * b - b^2 + b^3

def hash_op (a b : ℝ) : ℝ := a + b - a * b^2 + a * b^3

theorem operation_result : (at_op 7 3) / (hash_op 7 3) = 39 / 136 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l1858_185806


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l1858_185876

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / (Real.exp x)

theorem tangent_line_at_zero (x y : ℝ) :
  (∃ (m : ℝ), HasDerivAt f m 0 ∧ m = -1) →
  f 0 = 1 →
  (x + y - 1 = 0 ↔ y - f 0 = m * (x - 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l1858_185876


namespace NUMINAMATH_CALUDE_event_probability_comparison_l1858_185891

theorem event_probability_comparison (v : ℝ) (n : ℕ) (h₁ : v = 0.1) (h₂ : n = 998) :
  (n.choose 99 : ℝ) * v^99 * (1 - v)^(n - 99) > (n.choose 100 : ℝ) * v^100 * (1 - v)^(n - 100) :=
sorry

end NUMINAMATH_CALUDE_event_probability_comparison_l1858_185891


namespace NUMINAMATH_CALUDE_range_of_a_l1858_185893

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1858_185893


namespace NUMINAMATH_CALUDE_jacket_price_theorem_l1858_185809

theorem jacket_price_theorem (SRP : ℝ) (marked_discount : ℝ) (additional_discount : ℝ) :
  SRP = 120 →
  marked_discount = 0.4 →
  additional_discount = 0.2 →
  let marked_price := SRP * (1 - marked_discount)
  let final_price := marked_price * (1 - additional_discount)
  (final_price / SRP) * 100 = 48 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_theorem_l1858_185809


namespace NUMINAMATH_CALUDE_all_zeros_assignment_l1858_185843

/-- Represents a vertex in the triangular grid -/
structure Vertex (n : ℕ) where
  x : Fin (n + 1)
  y : Fin (n + 1)
  h : x.val + y.val ≤ n

/-- Represents an assignment of real numbers to vertices -/
def Assignment (n : ℕ) := Vertex n → ℝ

/-- Checks if three vertices form a triangle parallel to the sides of the main triangle -/
def is_parallel_triangle (n : ℕ) (v1 v2 v3 : Vertex n) : Prop :=
  ∃ (dx dy : Fin (n + 1)), 
    (v2.x = v1.x + dx ∧ v2.y = v1.y) ∧
    (v3.x = v1.x ∧ v3.y = v1.y + dy)

/-- The main theorem -/
theorem all_zeros_assignment {n : ℕ} (h : n ≥ 3) 
  (f : Assignment n) 
  (sum_zero : ∀ (v1 v2 v3 : Vertex n), 
    is_parallel_triangle n v1 v2 v3 → f v1 + f v2 + f v3 = 0) :
  ∀ v : Vertex n, f v = 0 := by sorry

end NUMINAMATH_CALUDE_all_zeros_assignment_l1858_185843


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1858_185804

theorem polynomial_divisibility (m n : ℕ) :
  ∃ q : Polynomial ℤ, (X^2 + X + 1) * q = X^(3*m+2) + (-X^2 - 1)^(3*n+1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1858_185804


namespace NUMINAMATH_CALUDE_enriques_commission_l1858_185868

/-- Represents the commission rate as a real number between 0 and 1 -/
def commission_rate : ℝ := 0.15

/-- Represents the number of suits sold -/
def suits_sold : ℕ := 2

/-- Represents the price of each suit in dollars -/
def suit_price : ℝ := 700.00

/-- Represents the number of shirts sold -/
def shirts_sold : ℕ := 6

/-- Represents the price of each shirt in dollars -/
def shirt_price : ℝ := 50.00

/-- Represents the number of loafers sold -/
def loafers_sold : ℕ := 2

/-- Represents the price of each pair of loafers in dollars -/
def loafer_price : ℝ := 150.00

/-- Calculates the total sales amount -/
def total_sales : ℝ := 
  suits_sold * suit_price + shirts_sold * shirt_price + loafers_sold * loafer_price

/-- Theorem: Enrique's commission is $300.00 -/
theorem enriques_commission : commission_rate * total_sales = 300.00 := by
  sorry

end NUMINAMATH_CALUDE_enriques_commission_l1858_185868


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l1858_185803

theorem angle_measure_in_special_triangle (a b c : ℝ) (h : b^2 + c^2 = a^2 + b*c) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2*b*c))
  A = π/3 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l1858_185803


namespace NUMINAMATH_CALUDE_sum_of_binary_digits_157_l1858_185819

/-- The sum of the digits in the binary representation of 157 is 5. -/
theorem sum_of_binary_digits_157 : 
  (Nat.digits 2 157).sum = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_binary_digits_157_l1858_185819


namespace NUMINAMATH_CALUDE_stationary_encounter_rate_l1858_185863

/-- The encounter rate of meteors when a ship is stationary, given two parallel streams of meteors with specific encounter rates when the ship is moving. -/
theorem stationary_encounter_rate 
  (rate1 : ℚ) -- Rate of encountering meteors coming towards the ship
  (rate2 : ℚ) -- Rate of encountering meteors traveling in the same direction as the ship
  (h1 : rate1 = 1 / 7)
  (h2 : rate2 = 1 / 13) :
  rate1 + rate2 = 20 / 91 :=
sorry

end NUMINAMATH_CALUDE_stationary_encounter_rate_l1858_185863


namespace NUMINAMATH_CALUDE_prob_heads_win_value_l1858_185829

/-- The probability of getting heads in a fair coin flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting tails in a fair coin flip -/
def p_tails : ℚ := 1/2

/-- The number of consecutive heads needed to win -/
def heads_to_win : ℕ := 6

/-- The number of consecutive tails needed to lose -/
def tails_to_lose : ℕ := 3

/-- The probability of encountering a run of 6 heads before a run of 3 tails 
    when repeatedly flipping a fair coin -/
def prob_heads_win : ℚ := 32/63

/-- Theorem stating that the probability of encountering a run of 6 heads 
    before a run of 3 tails when repeatedly flipping a fair coin is 32/63 -/
theorem prob_heads_win_value : 
  prob_heads_win = 32/63 :=
sorry

end NUMINAMATH_CALUDE_prob_heads_win_value_l1858_185829


namespace NUMINAMATH_CALUDE_circle_elimination_count_l1858_185844

/-- Calculates the total number of counts in a circle elimination game. -/
def totalCounts (initialPeople : ℕ) : ℕ :=
  let rec countRounds (remaining : ℕ) (acc : ℕ) : ℕ :=
    if remaining ≤ 2 then acc
    else
      let eliminated := remaining / 3
      let newRemaining := remaining - eliminated
      countRounds newRemaining (acc + remaining)
  countRounds initialPeople 0

/-- Theorem stating that for 21 initial people, the total count is 64. -/
theorem circle_elimination_count :
  totalCounts 21 = 64 := by
  sorry

end NUMINAMATH_CALUDE_circle_elimination_count_l1858_185844


namespace NUMINAMATH_CALUDE_sum_of_arguments_l1858_185879

def complex_equation (z : ℂ) : Prop := z^6 = 64 * Complex.I

theorem sum_of_arguments (z₁ z₂ z₃ z₄ z₅ z₆ : ℂ) 
  (h₁ : complex_equation z₁)
  (h₂ : complex_equation z₂)
  (h₃ : complex_equation z₃)
  (h₄ : complex_equation z₄)
  (h₅ : complex_equation z₅)
  (h₆ : complex_equation z₆)
  (distinct : z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₁ ≠ z₅ ∧ z₁ ≠ z₆ ∧
              z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₂ ≠ z₅ ∧ z₂ ≠ z₆ ∧
              z₃ ≠ z₄ ∧ z₃ ≠ z₅ ∧ z₃ ≠ z₆ ∧
              z₄ ≠ z₅ ∧ z₄ ≠ z₆ ∧
              z₅ ≠ z₆) :
  (Complex.arg z₁ + Complex.arg z₂ + Complex.arg z₃ + 
   Complex.arg z₄ + Complex.arg z₅ + Complex.arg z₆) * (180 / Real.pi) = 990 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_arguments_l1858_185879


namespace NUMINAMATH_CALUDE_max_candy_leftover_l1858_185851

theorem max_candy_leftover (x : ℕ+) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l1858_185851


namespace NUMINAMATH_CALUDE_unique_root_condition_l1858_185887

/-- The equation x + 1 = √(px) has exactly one real root if and only if p = 4 or p ≤ 0. -/
theorem unique_root_condition (p : ℝ) : 
  (∃! x : ℝ, x + 1 = Real.sqrt (p * x)) ↔ p = 4 ∨ p ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_condition_l1858_185887


namespace NUMINAMATH_CALUDE_divisible_number_is_six_l1858_185848

/-- The number of three-digit numbers divisible by the specific number -/
def divisible_count : ℕ := 150

/-- The lower bound of three-digit numbers -/
def lower_bound : ℕ := 100

/-- The upper bound of three-digit numbers -/
def upper_bound : ℕ := 999

/-- The total count of three-digit numbers -/
def total_count : ℕ := upper_bound - lower_bound + 1

theorem divisible_number_is_six :
  ∃ (n : ℕ), n = 6 ∧
  (∀ k : ℕ, lower_bound ≤ k ∧ k ≤ upper_bound →
    (divisible_count * n = total_count)) :=
sorry

end NUMINAMATH_CALUDE_divisible_number_is_six_l1858_185848


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l1858_185889

theorem sum_of_fourth_powers (a : ℝ) (h : (a + 1/a)^4 = 16) : a^4 + 1/a^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l1858_185889


namespace NUMINAMATH_CALUDE_percentage_proof_l1858_185892

/-- The percentage of students who scored in the 70%-79% range -/
def percentage_in_range (total_students : ℕ) (students_in_range : ℕ) : ℚ :=
  students_in_range / total_students

/-- Proof that the percentage of students who scored in the 70%-79% range is 8/33 -/
theorem percentage_proof : 
  let total_students : ℕ := 33
  let students_in_range : ℕ := 8
  percentage_in_range total_students students_in_range = 8 / 33 := by
  sorry

end NUMINAMATH_CALUDE_percentage_proof_l1858_185892


namespace NUMINAMATH_CALUDE_store_price_calculation_l1858_185862

/-- If an item's online price is 300 yuan and it's 20% less than the store price,
    then the store price is 375 yuan. -/
theorem store_price_calculation (online_price store_price : ℝ) : 
  online_price = 300 →
  online_price = store_price - 0.2 * store_price →
  store_price = 375 := by
  sorry

end NUMINAMATH_CALUDE_store_price_calculation_l1858_185862


namespace NUMINAMATH_CALUDE_velocity_at_t_1_is_zero_l1858_185895

-- Define the motion equation
def s (t : ℝ) : ℝ := -t^2 + 2*t

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := -2*t + 2

-- Theorem statement
theorem velocity_at_t_1_is_zero : v 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_velocity_at_t_1_is_zero_l1858_185895


namespace NUMINAMATH_CALUDE_davids_biology_marks_l1858_185872

def english_marks : ℕ := 96
def math_marks : ℕ := 95
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 97
def average_marks : ℕ := 93
def total_subjects : ℕ := 5

theorem davids_biology_marks :
  let known_subjects_total := english_marks + math_marks + physics_marks + chemistry_marks
  let all_subjects_total := average_marks * total_subjects
  all_subjects_total - known_subjects_total = 95 := by
  sorry

end NUMINAMATH_CALUDE_davids_biology_marks_l1858_185872


namespace NUMINAMATH_CALUDE_toy_cost_correct_l1858_185885

/-- The cost of the assortment box of toys for Julia's new puppy -/
def toy_cost : ℝ := 40

/-- The adoption fee for the puppy -/
def adoption_fee : ℝ := 20

/-- The cost of dog food -/
def dog_food_cost : ℝ := 20

/-- The cost of one bag of treats -/
def treat_cost : ℝ := 2.5

/-- The number of treat bags purchased -/
def treat_bags : ℕ := 2

/-- The cost of the crate -/
def crate_cost : ℝ := 20

/-- The cost of the bed -/
def bed_cost : ℝ := 20

/-- The cost of the collar/leash combo -/
def collar_leash_cost : ℝ := 15

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.2

/-- The total amount Julia spent on the puppy -/
def total_spent : ℝ := 96

theorem toy_cost_correct : 
  (1 - discount_rate) * (adoption_fee + dog_food_cost + treat_cost * treat_bags + 
  crate_cost + bed_cost + collar_leash_cost + toy_cost) = total_spent := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_correct_l1858_185885


namespace NUMINAMATH_CALUDE_joe_haircut_time_l1858_185874

/-- The time it takes to cut different types of hair and the number of haircuts Joe performed --/
structure HaircutData where
  womenTime : ℕ  -- Time to cut a woman's hair
  menTime : ℕ    -- Time to cut a man's hair
  kidsTime : ℕ   -- Time to cut a kid's hair
  womenCount : ℕ -- Number of women's haircuts
  menCount : ℕ   -- Number of men's haircuts
  kidsCount : ℕ  -- Number of kids' haircuts

/-- Calculate the total time Joe spent cutting hair --/
def totalHaircutTime (data : HaircutData) : ℕ :=
  data.womenTime * data.womenCount +
  data.menTime * data.menCount +
  data.kidsTime * data.kidsCount

/-- Theorem stating that Joe spent 255 minutes cutting hair --/
theorem joe_haircut_time :
  let data : HaircutData := {
    womenTime := 50,
    menTime := 15,
    kidsTime := 25,
    womenCount := 3,
    menCount := 2,
    kidsCount := 3
  }
  totalHaircutTime data = 255 := by
  sorry


end NUMINAMATH_CALUDE_joe_haircut_time_l1858_185874


namespace NUMINAMATH_CALUDE_max_chain_length_is_optimal_l1858_185836

/-- Represents a triangular grid formed by dividing an equilateral triangle --/
structure TriangularGrid where
  n : ℕ
  total_triangles : ℕ := n^2

/-- Represents a chain of triangles in the grid --/
structure TriangleChain (grid : TriangularGrid) where
  length : ℕ
  is_valid : length ≤ grid.total_triangles

/-- The maximum length of a valid triangle chain in a given grid --/
def max_chain_length (grid : TriangularGrid) : ℕ :=
  grid.n^2 - grid.n + 1

/-- Theorem stating that the maximum chain length is n^2 - n + 1 --/
theorem max_chain_length_is_optimal (grid : TriangularGrid) :
  ∀ (chain : TriangleChain grid), chain.length ≤ max_chain_length grid :=
by sorry

end NUMINAMATH_CALUDE_max_chain_length_is_optimal_l1858_185836


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_specific_coefficients_l1858_185850

theorem infinite_solutions_imply_specific_coefficients :
  ∀ (a b : ℝ),
  (∀ x : ℝ, a * (x - 3) + b * (3 * x + 1) = 5 * (x + 1)) →
  (a = -1 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_specific_coefficients_l1858_185850


namespace NUMINAMATH_CALUDE_fraction_nonnegative_iff_l1858_185827

theorem fraction_nonnegative_iff (x : ℝ) :
  (x - 3) / (x^2 + 5*x + 11) ≥ 0 ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_fraction_nonnegative_iff_l1858_185827


namespace NUMINAMATH_CALUDE_rose_price_is_three_l1858_185898

-- Define the sales data
def tulips_day1 : ℕ := 30
def roses_day1 : ℕ := 20
def tulips_day2 : ℕ := 2 * tulips_day1
def roses_day2 : ℕ := 2 * roses_day1
def tulips_day3 : ℕ := (tulips_day2 * 10) / 100
def roses_day3 : ℕ := 16

-- Define the total sales
def total_tulips : ℕ := tulips_day1 + tulips_day2 + tulips_day3
def total_roses : ℕ := roses_day1 + roses_day2 + roses_day3

-- Define the price of a tulip
def tulip_price : ℚ := 2

-- Define the total earnings
def total_earnings : ℚ := 420

-- Theorem to prove
theorem rose_price_is_three :
  ∃ (rose_price : ℚ), 
    rose_price * total_roses + tulip_price * total_tulips = total_earnings ∧
    rose_price = 3 := by
  sorry


end NUMINAMATH_CALUDE_rose_price_is_three_l1858_185898


namespace NUMINAMATH_CALUDE_quadratic_roots_integer_l1858_185877

theorem quadratic_roots_integer (b c : ℤ) (k : ℤ) (h : b^2 - 4*c = k^2) :
  ∃ x1 x2 : ℤ, x1^2 + b*x1 + c = 0 ∧ x2^2 + b*x2 + c = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_integer_l1858_185877


namespace NUMINAMATH_CALUDE_min_value_a_plus_4b_l1858_185882

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : 1 / (a - 1) + 1 / (b - 1) = 1) : 
  ∀ x y, x > 1 → y > 1 → 1 / (x - 1) + 1 / (y - 1) = 1 → a + 4 * b ≤ x + 4 * y ∧ 
  ∃ a₀ b₀, a₀ > 1 ∧ b₀ > 1 ∧ 1 / (a₀ - 1) + 1 / (b₀ - 1) = 1 ∧ a₀ + 4 * b₀ = 14 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_4b_l1858_185882
