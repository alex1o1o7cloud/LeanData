import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2613_261384

theorem sum_of_coefficients (a b c d e : ℝ) : 
  (∀ x, 216 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 36 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2613_261384


namespace NUMINAMATH_CALUDE_possible_ad_values_l2613_261393

/-- Represents a point on a line -/
structure Point :=
  (x : ℝ)

/-- The distance between two points -/
def distance (p q : Point) : ℝ := |p.x - q.x|

/-- Theorem: Possible values of AD given AB = 1, BC = 2, CD = 4 -/
theorem possible_ad_values (A B C D : Point) 
  (h1 : distance A B = 1)
  (h2 : distance B C = 2)
  (h3 : distance C D = 4) :
  (distance A D = 1) ∨ (distance A D = 3) ∨ (distance A D = 5) ∨ (distance A D = 7) :=
sorry

end NUMINAMATH_CALUDE_possible_ad_values_l2613_261393


namespace NUMINAMATH_CALUDE_additional_plates_count_l2613_261342

/-- Represents the number of choices for each position in a license plate. -/
structure LicensePlateChoices where
  first : Nat
  second : Nat
  third : Nat
  fourth : Nat

/-- Calculates the total number of possible license plates. -/
def totalPlates (choices : LicensePlateChoices) : Nat :=
  choices.first * choices.second * choices.third * choices.fourth

/-- The original choices for each position in TriCity license plates. -/
def originalChoices : LicensePlateChoices :=
  { first := 3, second := 4, third := 2, fourth := 5 }

/-- The new choices after adding two new letters. -/
def newChoices : LicensePlateChoices :=
  { first := originalChoices.first + 1,
    second := originalChoices.second,
    third := originalChoices.third + 1,
    fourth := originalChoices.fourth }

/-- Theorem stating the number of additional license plates after the change. -/
theorem additional_plates_count :
  totalPlates newChoices - totalPlates originalChoices = 120 := by
  sorry

end NUMINAMATH_CALUDE_additional_plates_count_l2613_261342


namespace NUMINAMATH_CALUDE_mina_pi_digits_l2613_261364

/-- The number of digits of pi memorized by each person -/
structure PiDigits where
  sam : ℕ
  carlos : ℕ
  mina : ℕ

/-- The conditions of the problem -/
def problem_conditions (d : PiDigits) : Prop :=
  d.sam = d.carlos + 6 ∧
  d.mina = 6 * d.carlos ∧
  d.sam = 10

/-- The theorem to prove -/
theorem mina_pi_digits (d : PiDigits) : 
  problem_conditions d → d.mina = 24 := by
  sorry

end NUMINAMATH_CALUDE_mina_pi_digits_l2613_261364


namespace NUMINAMATH_CALUDE_a_positive_necessary_not_sufficient_l2613_261325

theorem a_positive_necessary_not_sufficient :
  (∀ a : ℝ, a^2 < a → a > 0) ∧
  (∃ a : ℝ, a > 0 ∧ a^2 ≥ a) :=
by sorry

end NUMINAMATH_CALUDE_a_positive_necessary_not_sufficient_l2613_261325


namespace NUMINAMATH_CALUDE_john_bench_press_sets_l2613_261329

/-- The number of sets John does in his workout -/
def number_of_sets (weight_per_rep : ℕ) (reps_per_set : ℕ) (total_weight : ℕ) : ℕ :=
  total_weight / (weight_per_rep * reps_per_set)

/-- Theorem: John does 3 sets of bench presses -/
theorem john_bench_press_sets :
  number_of_sets 15 10 450 = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_bench_press_sets_l2613_261329


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2613_261339

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The first line from the problem -/
def line1 (a : ℝ) : Line :=
  { a := a, b := -1, c := 3 }

/-- The second line from the problem -/
def line2 (a : ℝ) : Line :=
  { a := 2, b := -(a+1), c := 4 }

/-- The condition a=-2 is sufficient for the lines to be parallel -/
theorem sufficient_condition :
  parallel (line1 (-2)) (line2 (-2)) := by sorry

/-- The condition a=-2 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ a : ℝ, a ≠ -2 ∧ parallel (line1 a) (line2 a) := by sorry

/-- The main theorem stating that a=-2 is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary :
  (parallel (line1 (-2)) (line2 (-2))) ∧
  (∃ a : ℝ, a ≠ -2 ∧ parallel (line1 a) (line2 a)) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l2613_261339


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2613_261355

theorem fixed_point_on_line (a b : ℝ) (h : a + b = 1) :
  2 * a * (1/2) - b * (-1) = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2613_261355


namespace NUMINAMATH_CALUDE_min_value_of_c_l2613_261302

theorem min_value_of_c (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  b = a + 1 →
  c = b + 1 →
  d = c + 1 →
  e = d + 1 →
  ∃ n : ℕ, a + b + c + d + e = n^3 →
  ∃ m : ℕ, b + c + d = m^2 →
  ∀ c' : ℕ, (∃ a' b' d' e' : ℕ, 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0 ∧ e' > 0 ∧
    b' = a' + 1 ∧ c' = b' + 1 ∧ d' = c' + 1 ∧ e' = d' + 1 ∧
    ∃ n' : ℕ, a' + b' + c' + d' + e' = n'^3 ∧
    ∃ m' : ℕ, b' + c' + d' = m'^2) →
  c' ≥ c →
  c = 675 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_c_l2613_261302


namespace NUMINAMATH_CALUDE_range_of_sum_l2613_261331

theorem range_of_sum (a b : ℝ) :
  (∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) ≥ -1) →
  -1 ≤ a + b ∧ a + b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l2613_261331


namespace NUMINAMATH_CALUDE_coronavirus_case_ratio_l2613_261392

/-- Given the number of coronavirus cases in a country during two waves, 
    prove the ratio of average daily cases between the waves. -/
theorem coronavirus_case_ratio 
  (first_wave_daily : ℕ) 
  (second_wave_total : ℕ) 
  (second_wave_days : ℕ) 
  (h1 : first_wave_daily = 300)
  (h2 : second_wave_total = 21000)
  (h3 : second_wave_days = 14) :
  (second_wave_total / second_wave_days) / first_wave_daily = 5 := by
  sorry


end NUMINAMATH_CALUDE_coronavirus_case_ratio_l2613_261392


namespace NUMINAMATH_CALUDE_abs_neg_five_plus_three_l2613_261372

theorem abs_neg_five_plus_three : |(-5 : ℤ) + 3| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_plus_three_l2613_261372


namespace NUMINAMATH_CALUDE_cos_sin_75_product_l2613_261395

theorem cos_sin_75_product (θ : Real) (h : θ = 75 * π / 180) : 
  (Real.cos θ + Real.sin θ) * (Real.cos θ - Real.sin θ) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_75_product_l2613_261395


namespace NUMINAMATH_CALUDE_monotonic_function_upper_bound_l2613_261332

open Real

/-- A monotonic function on (0, +∞) satisfying certain conditions -/
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  Monotone f ∧ 
  (∀ x > 0, f (f x - exp x + x) = exp 1) ∧
  (∀ x > 0, DifferentiableAt ℝ f x)

/-- The theorem stating the upper bound of a -/
theorem monotonic_function_upper_bound 
  (f : ℝ → ℝ) 
  (hf : MonotonicFunction f) 
  (h : ∀ x > 0, f x + deriv f x ≥ (a : ℝ) * x) :
  a ≤ 2 * exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_upper_bound_l2613_261332


namespace NUMINAMATH_CALUDE_parabola_directrix_l2613_261387

/-- Given a parabola x² = ay with directrix y = -1/4, prove that a = 1 -/
theorem parabola_directrix (x y a : ℝ) : 
  (x^2 = a * y) →  -- Parabola equation
  (y = -1/4 → a = 1) :=  -- Directrix equation implies a = 1
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2613_261387


namespace NUMINAMATH_CALUDE_friends_contribution_l2613_261385

/-- Represents the expenses of a group of friends -/
structure Expenses where
  num_friends : Nat
  total_amount : Rat

/-- Calculates the amount each friend should contribute -/
def calculate_contribution (e : Expenses) : Rat :=
  e.total_amount / e.num_friends

/-- Theorem: For 5 friends with total expenses of $61, each should contribute $12.20 -/
theorem friends_contribution :
  let e : Expenses := { num_friends := 5, total_amount := 61 }
  calculate_contribution e = 61 / 5 := by sorry

end NUMINAMATH_CALUDE_friends_contribution_l2613_261385


namespace NUMINAMATH_CALUDE_only_negative_three_l2613_261356

theorem only_negative_three (a b c d : ℝ) : 
  a = |-3| ∧ b = -3 ∧ c = -(-3) ∧ d = 1/3 → 
  (b < 0 ∧ a ≥ 0 ∧ c ≥ 0 ∧ d > 0) := by sorry

end NUMINAMATH_CALUDE_only_negative_three_l2613_261356


namespace NUMINAMATH_CALUDE_initial_average_production_l2613_261379

theorem initial_average_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) 
  (h1 : n = 1)
  (h2 : today_production = 60)
  (h3 : new_average = 55) :
  ∃ initial_average : ℕ, initial_average = 50 ∧ 
    (initial_average * n + today_production) / (n + 1) = new_average := by
  sorry

end NUMINAMATH_CALUDE_initial_average_production_l2613_261379


namespace NUMINAMATH_CALUDE_limit_of_a_is_three_fourths_l2613_261399

def a (n : ℕ) : ℚ := (3 * n^2 + 2) / (4 * n^2 - 1)

theorem limit_of_a_is_three_fourths :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/4| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_of_a_is_three_fourths_l2613_261399


namespace NUMINAMATH_CALUDE_farm_animals_after_transaction_l2613_261365

/-- Represents the number of animals on the farm -/
structure FarmAnimals where
  horses : ℕ
  cows : ℕ

/-- Represents the ratio of horses to cows -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

def initial_ratio : Ratio := { numerator := 3, denominator := 1 }
def final_ratio : Ratio := { numerator := 5, denominator := 3 }

def transaction (farm : FarmAnimals) : FarmAnimals :=
  { horses := farm.horses - 15, cows := farm.cows + 15 }

theorem farm_animals_after_transaction (farm : FarmAnimals) :
  farm.horses / farm.cows = initial_ratio.numerator / initial_ratio.denominator →
  (transaction farm).horses / (transaction farm).cows = final_ratio.numerator / final_ratio.denominator →
  (transaction farm).horses - (transaction farm).cows = 30 :=
by sorry

end NUMINAMATH_CALUDE_farm_animals_after_transaction_l2613_261365


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_age_l2613_261377

/-- Proves that the number of years until a man's age is twice his son's age is 2,
    given the man is 22 years older than his son and the son's present age is 20 years. -/
theorem mans_age_twice_sons_age (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 20 → age_difference = 22 → 
  ∃ (years : ℕ), years = 2 ∧ 
    (son_age + years) * 2 = (son_age + age_difference + years) := by
  sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_age_l2613_261377


namespace NUMINAMATH_CALUDE_concert_revenue_l2613_261363

def adult_price : ℕ := 26
def child_price : ℕ := adult_price / 2
def adult_attendees : ℕ := 183
def child_attendees : ℕ := 28

theorem concert_revenue :
  adult_price * adult_attendees + child_price * child_attendees = 5122 :=
by sorry

end NUMINAMATH_CALUDE_concert_revenue_l2613_261363


namespace NUMINAMATH_CALUDE_tips_fraction_l2613_261305

/-- Represents the income structure of a waiter -/
structure WaiterIncome where
  salary : ℚ
  tips : ℚ

/-- Calculates the fraction of income from tips -/
def fractionFromTips (income : WaiterIncome) : ℚ :=
  income.tips / (income.salary + income.tips)

/-- Theorem: Given the conditions, the fraction of income from tips is 9/13 -/
theorem tips_fraction (income : WaiterIncome) 
  (h : income.tips = (9 / 4) * income.salary) : 
  fractionFromTips income = 9 / 13 := by
  sorry

#check tips_fraction

end NUMINAMATH_CALUDE_tips_fraction_l2613_261305


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2613_261344

def U : Set ℕ := {x | x < 8}

def A : Set ℕ := {x | (x - 1) * (x - 3) * (x - 4) * (x - 7) = 0}

theorem complement_of_A_in_U : U \ A = {0, 2, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2613_261344


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2613_261373

theorem quadratic_roots_relation (r s : ℝ) (p q : ℝ) : 
  (3 * r^2 + 4 * r + 2 = 0) →
  (3 * s^2 + 4 * s + 2 = 0) →
  ((1/r^2) + (1/s^2) = -p) →
  ((1/r^2) * (1/s^2) = q) →
  p = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2613_261373


namespace NUMINAMATH_CALUDE_intersected_cubes_count_l2613_261357

/-- Represents a 3x3x3 cube composed of unit cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  diagonal : Real

/-- Represents a plane that bisects the diagonal of the large cube -/
structure BisectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

/-- Represents the number of unit cubes intersected by the bisecting plane -/
def intersected_cubes (cube : LargeCube) (plane : BisectingPlane) : Nat :=
  sorry

/-- Main theorem: A plane perpendicular to and bisecting a space diagonal of a 3x3x3 cube
    intersects exactly 19 of the unit cubes -/
theorem intersected_cubes_count 
  (cube : LargeCube) 
  (plane : BisectingPlane) 
  (h1 : cube.size = 3)
  (h2 : cube.total_cubes = 27)
  (h3 : plane.perpendicular_to_diagonal)
  (h4 : plane.bisects_diagonal) :
  intersected_cubes cube plane = 19 := by
  sorry

end NUMINAMATH_CALUDE_intersected_cubes_count_l2613_261357


namespace NUMINAMATH_CALUDE_part_one_part_two_l2613_261380

/-- The function f(x) = mx^2 - mx - 1 --/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

/-- Theorem for part 1 of the problem --/
theorem part_one :
  ∀ x : ℝ, f (1/2) x < 0 ↔ -1 < x ∧ x < 2 := by sorry

/-- Theorem for part 2 of the problem --/
theorem part_two (m : ℝ) (x : ℝ) :
  f m x < (m - 1) * x^2 + 2 * x - 2 * m - 1 ↔
    (m < 2 ∧ m < x ∧ x < 2) ∨ (m > 2 ∧ 2 < x ∧ x < m) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2613_261380


namespace NUMINAMATH_CALUDE_min_value_sum_l2613_261368

theorem min_value_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 19 / x + 98 / y = 1) :
  x + y ≥ 117 + 14 * Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l2613_261368


namespace NUMINAMATH_CALUDE_circle_position_l2613_261315

def circle_center : ℝ × ℝ := (1, 2)
def circle_radius : ℝ := 1

def distance_to_y_axis (center : ℝ × ℝ) : ℝ := |center.1|
def distance_to_x_axis (center : ℝ × ℝ) : ℝ := |center.2|

def is_tangent_to_y_axis (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  distance_to_y_axis center = radius

def is_disjoint_from_x_axis (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  distance_to_x_axis center > radius

theorem circle_position :
  is_tangent_to_y_axis circle_center circle_radius ∧
  is_disjoint_from_x_axis circle_center circle_radius :=
by sorry

end NUMINAMATH_CALUDE_circle_position_l2613_261315


namespace NUMINAMATH_CALUDE_line_l_equation_l2613_261391

-- Define points A and B
def A : ℝ × ℝ := (3, 3)
def B : ℝ × ℝ := (5, 2)

-- Define lines l1 and l2
def l1 (x y : ℝ) : Prop := 3 * x - y - 1 = 0
def l2 (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the intersection point of l1 and l2
def intersection : ℝ × ℝ := (1, 2)

-- Define the property that l passes through the intersection
def passes_through_intersection (l : ℝ → ℝ → Prop) : Prop :=
  l (intersection.1) (intersection.2)

-- Define the property of equal distance from A and B to l
def equal_distance (l : ℝ → ℝ → Prop) : Prop :=
  ∃ d : ℝ, d > 0 ∧
    (∃ x y : ℝ, l x y ∧ (x - A.1)^2 + (y - A.2)^2 = d^2) ∧
    (∃ x y : ℝ, l x y ∧ (x - B.1)^2 + (y - B.2)^2 = d^2)

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := x - 6 * y + 11 = 0 ∨ x + 2 * y - 5 = 0

-- Theorem statement
theorem line_l_equation :
  ∀ l : ℝ → ℝ → Prop,
    passes_through_intersection l →
    equal_distance l →
    (∀ x y : ℝ, l x y ↔ line_l x y) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_l2613_261391


namespace NUMINAMATH_CALUDE_smallest_eulerian_polyhedron_sum_l2613_261383

/-- A polyhedron is Eulerian if it has an Eulerian path -/
def IsEulerianPolyhedron (V E F : ℕ) : Prop :=
  ∃ (oddDegreeVertices : ℕ), oddDegreeVertices = 2 ∧ 
  V ≥ 4 ∧ E ≥ 6 ∧ F ≥ 4 ∧ V - E + F = 2

/-- The sum of vertices, edges, and faces for a polyhedron -/
def PolyhedronSum (V E F : ℕ) : ℕ := V + E + F

theorem smallest_eulerian_polyhedron_sum :
  ∀ V E F : ℕ, IsEulerianPolyhedron V E F →
  PolyhedronSum V E F ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_eulerian_polyhedron_sum_l2613_261383


namespace NUMINAMATH_CALUDE_tetrahedron_bisector_ratio_l2613_261354

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents the area of a triangle -/
def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Represents a point on an edge of the tetrahedron -/
def intersectionPoint (t : Tetrahedron) : Point3D := sorry

/-- Theorem: In a tetrahedron ABCD, where the bisector plane of the dihedral angle around edge CD
    intersects AB at point E, the ratio of AE to BE is equal to the ratio of the areas of
    triangles ACD and BCD -/
theorem tetrahedron_bisector_ratio (t : Tetrahedron) :
  let E := intersectionPoint t
  let AE := Real.sqrt ((t.A.x - E.x)^2 + (t.A.y - E.y)^2 + (t.A.z - E.z)^2)
  let BE := Real.sqrt ((t.B.x - E.x)^2 + (t.B.y - E.y)^2 + (t.B.z - E.z)^2)
  let t_ACD := triangleArea t.A t.C t.D
  let t_BCD := triangleArea t.B t.C t.D
  AE / BE = t_ACD / t_BCD := by sorry

end NUMINAMATH_CALUDE_tetrahedron_bisector_ratio_l2613_261354


namespace NUMINAMATH_CALUDE_prime_square_plus_one_triples_l2613_261378

theorem prime_square_plus_one_triples :
  ∀ a b c : ℕ,
    Prime (a^2 + 1) →
    Prime (b^2 + 1) →
    (a^2 + 1) * (b^2 + 1) = c^2 + 1 →
    ((a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3)) :=
by sorry

end NUMINAMATH_CALUDE_prime_square_plus_one_triples_l2613_261378


namespace NUMINAMATH_CALUDE_fathers_age_l2613_261376

theorem fathers_age (man_age father_age : ℕ) : 
  man_age = (2 * father_age) / 5 →
  man_age + 5 = (father_age + 5) / 2 →
  father_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_l2613_261376


namespace NUMINAMATH_CALUDE_tshirts_sold_count_l2613_261321

/-- The revenue generated from selling t-shirts -/
def tshirt_revenue : ℕ := 4300

/-- The revenue generated from each t-shirt -/
def revenue_per_tshirt : ℕ := 215

/-- The number of t-shirts sold -/
def num_tshirts : ℕ := tshirt_revenue / revenue_per_tshirt

theorem tshirts_sold_count : num_tshirts = 20 := by
  sorry

end NUMINAMATH_CALUDE_tshirts_sold_count_l2613_261321


namespace NUMINAMATH_CALUDE_problem_statement_l2613_261348

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x * (x - 4) * (x + 1) < 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

theorem problem_statement :
  (∀ x : ℝ, x ∈ (A ∪ B 4) ↔ -1 < x ∧ x < 5) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ (U \ A) ↔ x ∈ (U \ B a)) ↔ 0 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l2613_261348


namespace NUMINAMATH_CALUDE_john_cookies_left_l2613_261390

/-- The number of cookies John has left after sharing with his friend -/
def cookies_left : ℕ :=
  let initial_cookies : ℕ := 2 * 12
  let after_first_day : ℕ := initial_cookies - (initial_cookies / 4)
  let after_second_day : ℕ := after_first_day - 5
  let shared_cookies : ℕ := after_second_day / 3
  after_second_day - shared_cookies

theorem john_cookies_left : cookies_left = 9 := by
  sorry

end NUMINAMATH_CALUDE_john_cookies_left_l2613_261390


namespace NUMINAMATH_CALUDE_gcd_g_x_equals_120_l2613_261317

def g (x : ℤ) : ℤ := (5*x + 7)*(11*x + 3)*(17*x + 8)*(4*x + 5)

theorem gcd_g_x_equals_120 (x : ℤ) (h : ∃ k : ℤ, x = 17280 * k) :
  Nat.gcd (Int.natAbs (g x)) (Int.natAbs x) = 120 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_x_equals_120_l2613_261317


namespace NUMINAMATH_CALUDE_distribute_seven_among_three_l2613_261306

/-- The number of ways to distribute n indistinguishable items among k distinct groups,
    with each group receiving at least one item. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 15 ways to distribute 7 recommended places among 3 schools,
    with each school receiving at least one place. -/
theorem distribute_seven_among_three :
  distribute 7 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_distribute_seven_among_three_l2613_261306


namespace NUMINAMATH_CALUDE_b_share_calculation_l2613_261326

theorem b_share_calculation (total : ℝ) : 
  let a := (2 : ℝ) / 15 * total
  let b := (3 : ℝ) / 15 * total
  let c := (4 : ℝ) / 15 * total
  let d := (6 : ℝ) / 15 * total
  d - c = 700 → b = 1050 := by
  sorry

end NUMINAMATH_CALUDE_b_share_calculation_l2613_261326


namespace NUMINAMATH_CALUDE_remaining_money_l2613_261320

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def john_savings : Nat := base_to_decimal [5, 3, 2, 5] 9
def ticket_cost : Nat := base_to_decimal [0, 5, 2, 1] 8

theorem remaining_money :
  john_savings - ticket_cost = 3159 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l2613_261320


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l2613_261333

theorem smallest_positive_solution (x : ℕ) : x = 21 ↔ 
  (x > 0 ∧ 
   (45 * x + 7) % 25 = 3 ∧ 
   ∀ y : ℕ, y > 0 → y < x → (45 * y + 7) % 25 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l2613_261333


namespace NUMINAMATH_CALUDE_icosagon_diagonals_from_vertex_l2613_261382

/-- The number of sides in an icosagon -/
def icosagon_sides : ℕ := 20

/-- The number of diagonals from a single vertex in an icosagon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

theorem icosagon_diagonals_from_vertex :
  diagonals_from_vertex icosagon_sides = 17 := by sorry

end NUMINAMATH_CALUDE_icosagon_diagonals_from_vertex_l2613_261382


namespace NUMINAMATH_CALUDE_lottery_expected_profit_l2613_261362

-- Define the lottery ticket parameters
def ticket_cost : ℝ := 2
def winning_probability : ℝ := 0.01
def prize : ℝ := 50

-- Define the expected profit function
def expected_profit (cost winning_prob prize : ℝ) : ℝ :=
  winning_prob * prize - cost

-- Theorem statement
theorem lottery_expected_profit :
  expected_profit ticket_cost winning_probability prize = -1.5 := by
  sorry

end NUMINAMATH_CALUDE_lottery_expected_profit_l2613_261362


namespace NUMINAMATH_CALUDE_num_al_sandwiches_l2613_261371

/-- Represents the number of different types of bread available. -/
def num_breads : Nat := 5

/-- Represents the number of different types of meat available. -/
def num_meats : Nat := 6

/-- Represents the number of different types of cheese available. -/
def num_cheeses : Nat := 6

/-- Represents the number of forbidden combinations. -/
def num_forbidden : Nat := 3

/-- Represents the number of overcounted combinations. -/
def num_overcounted : Nat := 1

/-- Calculates the total number of possible sandwich combinations. -/
def total_combinations : Nat := num_breads * num_meats * num_cheeses

/-- Calculates the number of forbidden sandwich combinations. -/
def forbidden_combinations : Nat :=
  num_breads + num_cheeses + num_cheeses - num_overcounted

/-- Theorem stating the number of different sandwiches Al can order. -/
theorem num_al_sandwiches : 
  total_combinations - forbidden_combinations = 164 := by
  sorry

end NUMINAMATH_CALUDE_num_al_sandwiches_l2613_261371


namespace NUMINAMATH_CALUDE_product_of_squares_and_products_l2613_261347

theorem product_of_squares_and_products (a b c : ℝ) 
  (sum_eq : a + b + c = 5)
  (sum_squares_eq : a^2 + b^2 + c^2 = 15)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 47) :
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) = 625 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_and_products_l2613_261347


namespace NUMINAMATH_CALUDE_max_cookie_price_l2613_261389

theorem max_cookie_price (k p : ℕ) 
  (h1 : 8 * k + 3 * p < 200)
  (h2 : 4 * k + 5 * p > 150) :
  k ≤ 19 ∧ ∃ (k' p' : ℕ), k' = 19 ∧ 8 * k' + 3 * p' < 200 ∧ 4 * k' + 5 * p' > 150 :=
sorry

end NUMINAMATH_CALUDE_max_cookie_price_l2613_261389


namespace NUMINAMATH_CALUDE_power_of_ten_negative_y_l2613_261343

theorem power_of_ten_negative_y (y : ℝ) (h : (10 : ℝ) ^ (2 * y) = 25) : (10 : ℝ) ^ (-y) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_ten_negative_y_l2613_261343


namespace NUMINAMATH_CALUDE_unique_g_2_value_l2613_261398

theorem unique_g_2_value (g : ℤ → ℤ) 
  (h : ∀ m n : ℤ, g (m + n) + g (m * n + 1) = g m * g n + 1) : 
  ∃! x : ℤ, g 2 = x ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_unique_g_2_value_l2613_261398


namespace NUMINAMATH_CALUDE_squared_lengths_sum_l2613_261358

/-- Two circles O and O₁, where O has equation x² + y² = 25 and O₁ has center (m, 0) -/
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 25}
def circle_O₁ (m : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - m)^2 + p.2^2 = (m - 3)^2 + 4^2}

/-- Point P where the circles intersect -/
def P : ℝ × ℝ := (3, 4)

/-- Line l with slope k passing through P -/
def line_l (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 - 4 = k * (p.1 - 3)}

/-- Line l₁ perpendicular to l passing through P -/
def line_l₁ (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 - 4 = (-1/k) * (p.1 - 3)}

/-- Points A and B where line l intersects circles O and O₁ -/
def A (k m : ℝ) : ℝ × ℝ := sorry
def B (k m : ℝ) : ℝ × ℝ := sorry

/-- Points C and D where line l₁ intersects circles O and O₁ -/
def C (k m : ℝ) : ℝ × ℝ := sorry
def D (k m : ℝ) : ℝ × ℝ := sorry

/-- The main theorem -/
theorem squared_lengths_sum (m : ℝ) (k : ℝ) (h : k ≠ 0) :
  ∀ (A B : ℝ × ℝ) (C D : ℝ × ℝ),
  A ∈ circle_O ∧ A ∈ line_l k ∧
  B ∈ circle_O₁ m ∧ B ∈ line_l k ∧
  C ∈ circle_O ∧ C ∈ line_l₁ k ∧
  D ∈ circle_O₁ m ∧ D ∈ line_l₁ k →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (C.1 - D.1)^2 + (C.2 - D.2)^2 = 4 * m^2 := by
  sorry

end NUMINAMATH_CALUDE_squared_lengths_sum_l2613_261358


namespace NUMINAMATH_CALUDE_exchange_rate_problem_l2613_261322

theorem exchange_rate_problem (d : ℕ) : 
  (3 / 2 : ℚ) * d - 72 = d → d = 144 := by sorry

end NUMINAMATH_CALUDE_exchange_rate_problem_l2613_261322


namespace NUMINAMATH_CALUDE_aria_cookie_spending_l2613_261374

/-- The number of days in March -/
def days_in_march : ℕ := 31

/-- The number of cookies Aria purchased each day -/
def cookies_per_day : ℕ := 4

/-- The cost of each cookie in dollars -/
def cost_per_cookie : ℕ := 19

/-- The total amount Aria spent on cookies in March -/
def total_spent : ℕ := days_in_march * cookies_per_day * cost_per_cookie

theorem aria_cookie_spending :
  total_spent = 2356 := by sorry

end NUMINAMATH_CALUDE_aria_cookie_spending_l2613_261374


namespace NUMINAMATH_CALUDE_lcm_gcd_product_24_60_l2613_261314

theorem lcm_gcd_product_24_60 : Nat.lcm 24 60 * Nat.gcd 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_24_60_l2613_261314


namespace NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l2613_261300

def T : Finset ℕ := Finset.range 11

def difference_sum (S : Finset ℕ) : ℕ :=
  S.sum (fun i => S.sum (fun j => if i > j then 3^i - 3^j else 0))

theorem difference_sum_of_powers_of_three :
  difference_sum T = 783492 := by
  sorry

end NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l2613_261300


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2613_261394

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 - 4*x + 2 = 0) ↔ ((x - 2)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2613_261394


namespace NUMINAMATH_CALUDE_simplify_fraction_solve_inequality_system_l2613_261388

-- Problem 1
theorem simplify_fraction (m n : ℝ) (hm : m ≠ 0) (hmn : 9*m^2 ≠ 4*n^2) :
  (1/(3*m-2*n) - 1/(3*m+2*n)) / (m*n/((9*m^2)-(4*n^2))) = 4/m := by sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) :
  (3*x + 10 > 5*x - 2*(5-x) ∧ (x+3)/5 > 1-x) ↔ (1/3 < x ∧ x < 5) := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_solve_inequality_system_l2613_261388


namespace NUMINAMATH_CALUDE_union_and_intersection_when_m_is_3_intersection_empty_iff_l2613_261340

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m}

-- Theorem for part 1
theorem union_and_intersection_when_m_is_3 :
  (A ∪ B 3 = {x | -2 ≤ x ∧ x < 6}) ∧ (A ∩ B 3 = ∅) := by sorry

-- Theorem for part 2
theorem intersection_empty_iff :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ≤ 1 ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_when_m_is_3_intersection_empty_iff_l2613_261340


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2613_261341

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 - 20 * x - 8 * y + 72 = 0

/-- The circle is inscribed in a rectangle -/
def is_inscribed (circle : (ℝ → ℝ → Prop)) (rectangle : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), circle x y → (x, y) ∈ rectangle

/-- One pair of sides of the rectangle is parallel to the y-axis -/
def sides_parallel_to_y_axis (rectangle : Set (ℝ × ℝ)) : Prop :=
  ∃ (x₁ x₂ : ℝ), ∀ (y : ℝ), (x₁, y) ∈ rectangle ∨ (x₂, y) ∈ rectangle

/-- The area of the rectangle -/
def rectangle_area (rectangle : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem -/
theorem inscribed_circle_rectangle_area :
  ∀ (rectangle : Set (ℝ × ℝ)),
  is_inscribed circle_equation rectangle →
  sides_parallel_to_y_axis rectangle →
  rectangle_area rectangle = 28 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2613_261341


namespace NUMINAMATH_CALUDE_circle_radii_sum_l2613_261367

theorem circle_radii_sum : 
  ∀ r : ℝ, 
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 14) :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_sum_l2613_261367


namespace NUMINAMATH_CALUDE_matrix_power_four_l2613_261397

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 2, -1]

theorem matrix_power_four :
  A^4 = !![(-4), 6; (-6), 5] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l2613_261397


namespace NUMINAMATH_CALUDE_simplify_expressions_l2613_261330

theorem simplify_expressions :
  (∀ x y : ℝ, x^2 - 5*y - 4*x^2 + y - 1 = -3*x^2 - 4*y - 1) ∧
  (∀ a b : ℝ, 7*a + 3*(a - 3*b) - 2*(b - 3*a) = 16*a - 11*b) :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2613_261330


namespace NUMINAMATH_CALUDE_sine_identity_and_not_monotonicity_l2613_261301

theorem sine_identity_and_not_monotonicity : 
  (∀ x : ℝ, Real.sin (π - x) = Real.sin x) ∧ 
  ¬(∀ α β : ℝ, 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 → α > β → Real.sin α > Real.sin β) := by
  sorry

end NUMINAMATH_CALUDE_sine_identity_and_not_monotonicity_l2613_261301


namespace NUMINAMATH_CALUDE_k_range_for_negative_sum_l2613_261312

/-- A power function that passes through the point (3, 27) -/
def f (x : ℝ) : ℝ := x^3

/-- The theorem stating the range of k for which f(k^2 + 3) + f(9 - 8k) < 0 holds -/
theorem k_range_for_negative_sum (k : ℝ) :
  f (k^2 + 3) + f (9 - 8*k) < 0 ↔ 2 < k ∧ k < 6 := by
  sorry


end NUMINAMATH_CALUDE_k_range_for_negative_sum_l2613_261312


namespace NUMINAMATH_CALUDE_triangle_max_area_l2613_261328

theorem triangle_max_area (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : Real.tan A * Real.tan B = 3/4) : 
  let a : ℝ := 4
  let b : ℝ := a * Real.sin B / Real.sin A
  let c : ℝ := a * Real.sin C / Real.sin A
  ∀ (S : ℝ), S = 1/2 * a * b * Real.sin C → S ≤ 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2613_261328


namespace NUMINAMATH_CALUDE_triangle_inequality_l2613_261304

/-- For any triangle ABC with side lengths a, b, c, circumradius R, and inradius r,
    the inequality (b² + c²) / (2bc) ≤ R / (2r) holds. -/
theorem triangle_inequality (a b c R r : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (hR : 0 < R) (hr : 0 < r) (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) :
    (b^2 + c^2) / (2 * b * c) ≤ R / (2 * r) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2613_261304


namespace NUMINAMATH_CALUDE_sara_team_wins_l2613_261338

/-- Represents a basketball team's game statistics -/
structure TeamStats where
  total_games : ℕ
  lost_games : ℕ

/-- Calculates the number of games won by a team -/
def games_won (stats : TeamStats) : ℕ :=
  stats.total_games - stats.lost_games

/-- Theorem: For Sara's team, the number of games won is 12 -/
theorem sara_team_wins (sara_team : TeamStats) 
  (h1 : sara_team.total_games = 16) 
  (h2 : sara_team.lost_games = 4) : 
  games_won sara_team = 12 := by
  sorry

end NUMINAMATH_CALUDE_sara_team_wins_l2613_261338


namespace NUMINAMATH_CALUDE_train_length_l2613_261369

/-- The length of a train given its speed, time to cross a bridge, and the bridge length. -/
theorem train_length (v : ℝ) (t : ℝ) (bridge_length : ℝ) (h1 : v = 70) (h2 : t = 13.884603517432893) (h3 : bridge_length = 150) :
  ∃ (train_length : ℝ), abs (train_length - 120) < 1 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2613_261369


namespace NUMINAMATH_CALUDE_quadratic_roots_negative_l2613_261308

theorem quadratic_roots_negative (p : ℝ) : 
  (∀ x : ℝ, x^2 + 2*(p+1)*x + 9*p - 5 = 0 → x < 0) ↔ 
  (p > 5/9 ∧ p ≤ 1) ∨ p ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_negative_l2613_261308


namespace NUMINAMATH_CALUDE_rotation_implies_equilateral_l2613_261351

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Rotation of a point around another point by a given angle -/
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Check if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Theorem: If rotating a triangle 60° around point A moves B to C, then the triangle is equilateral -/
theorem rotation_implies_equilateral (t : Triangle) :
  rotate t.A (π / 3) t.B = t.C → is_equilateral t := by sorry

end NUMINAMATH_CALUDE_rotation_implies_equilateral_l2613_261351


namespace NUMINAMATH_CALUDE_problem_solution_l2613_261396

theorem problem_solution (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 125 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2613_261396


namespace NUMINAMATH_CALUDE_chess_draw_probability_l2613_261366

theorem chess_draw_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.6)
  (h2 : prob_A_not_lose = 0.8) :
  prob_A_not_lose - prob_A_win = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_chess_draw_probability_l2613_261366


namespace NUMINAMATH_CALUDE_binomial_product_l2613_261318

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l2613_261318


namespace NUMINAMATH_CALUDE_second_number_proof_l2613_261361

theorem second_number_proof (a b c : ℝ) 
  (sum_eq : a + b + c = 98)
  (ratio_ab : a / b = 2 / 3)
  (ratio_bc : b / c = 5 / 8)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c) :
  b = 30 := by
sorry

end NUMINAMATH_CALUDE_second_number_proof_l2613_261361


namespace NUMINAMATH_CALUDE_tim_stacked_bales_l2613_261309

theorem tim_stacked_bales (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 28)
  (h2 : final_bales = 82) :
  final_bales - initial_bales = 54 := by
  sorry

end NUMINAMATH_CALUDE_tim_stacked_bales_l2613_261309


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l2613_261334

theorem no_positive_integer_solutions : 
  ¬∃ (x y : ℕ+), x^2 + y^2 + x = 2 * x^3 := by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l2613_261334


namespace NUMINAMATH_CALUDE_sandra_savings_proof_l2613_261323

-- Define the given conditions
def mother_contribution : ℝ := 4
def father_contribution : ℝ := 2 * mother_contribution
def candy_cost : ℝ := 0.5
def jelly_bean_cost : ℝ := 0.2
def candy_quantity : ℕ := 14
def jelly_bean_quantity : ℕ := 20
def money_left : ℝ := 11

-- Define Sandra's initial savings
def sandra_initial_savings : ℝ := 10

-- Theorem to prove
theorem sandra_savings_proof :
  sandra_initial_savings = 
    (candy_cost * candy_quantity + jelly_bean_cost * jelly_bean_quantity + money_left) - 
    (mother_contribution + father_contribution) := by
  sorry


end NUMINAMATH_CALUDE_sandra_savings_proof_l2613_261323


namespace NUMINAMATH_CALUDE_certain_number_proof_l2613_261360

theorem certain_number_proof (x : ℝ) : (60 / 100 * 500 = 50 / 100 * x) → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2613_261360


namespace NUMINAMATH_CALUDE_marias_flower_bed_area_l2613_261352

/-- Represents a rectangular flower bed with fence posts --/
structure FlowerBed where
  total_posts : ℕ
  post_spacing : ℕ
  longer_side_posts : ℕ
  shorter_side_posts : ℕ

/-- Calculates the area of the flower bed --/
def flower_bed_area (fb : FlowerBed) : ℕ :=
  (fb.shorter_side_posts - 1) * fb.post_spacing * ((fb.longer_side_posts - 1) * fb.post_spacing)

/-- Theorem stating that Maria's flower bed has an area of 350 square yards --/
theorem marias_flower_bed_area :
  ∃ fb : FlowerBed,
    fb.total_posts = 24 ∧
    fb.post_spacing = 5 ∧
    fb.longer_side_posts = 3 * fb.shorter_side_posts - 1 ∧
    fb.total_posts = fb.longer_side_posts + fb.shorter_side_posts + 2 ∧
    flower_bed_area fb = 350 :=
by sorry

end NUMINAMATH_CALUDE_marias_flower_bed_area_l2613_261352


namespace NUMINAMATH_CALUDE_distance_center_M_to_line_L_is_zero_l2613_261311

/-- The circle M -/
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 1 = 0

/-- The line L -/
def line_L (t x y : ℝ) : Prop :=
  x = 4*t + 3 ∧ y = 3*t + 1

/-- The center of circle M -/
def center_M : ℝ × ℝ :=
  (1, 2)

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a^2 + b^2)

theorem distance_center_M_to_line_L_is_zero :
  distance_point_to_line center_M 3 (-4) 5 = 0 := by sorry

end NUMINAMATH_CALUDE_distance_center_M_to_line_L_is_zero_l2613_261311


namespace NUMINAMATH_CALUDE_age_problem_l2613_261307

theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 52) : 
  b = 20 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2613_261307


namespace NUMINAMATH_CALUDE_wrong_calculation_correction_l2613_261319

theorem wrong_calculation_correction (x : ℝ) : 
  x / 5 + 16 = 58 → x / 15 + 74 = 88 := by
  sorry

end NUMINAMATH_CALUDE_wrong_calculation_correction_l2613_261319


namespace NUMINAMATH_CALUDE_complex_in_first_quadrant_l2613_261375

theorem complex_in_first_quadrant (m : ℝ) (h : m < 1) :
  let z : ℂ := (1 - m) + I
  z.re > 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_in_first_quadrant_l2613_261375


namespace NUMINAMATH_CALUDE_sum_of_two_equals_third_l2613_261324

theorem sum_of_two_equals_third (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = a + c ∨ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_equals_third_l2613_261324


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l2613_261370

/-- The surface area of a sphere that circumscribes a rectangular solid -/
theorem sphere_surface_area_from_rectangular_solid 
  (length width height : ℝ) 
  (h_length : length = 3) 
  (h_width : width = 2) 
  (h_height : height = 1) : 
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 14 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l2613_261370


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2613_261381

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : 3 * a + 4 * b + 2 * c = 3) :
  (1 / (2 * a + b) + 1 / (a + 3 * c) + 1 / (4 * b + c)) ≥ (3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2613_261381


namespace NUMINAMATH_CALUDE_remaining_investment_rate_l2613_261336

def total_investment : ℝ := 12000
def investment_at_7_percent : ℝ := 5500
def total_interest : ℝ := 970

def remaining_investment : ℝ := total_investment - investment_at_7_percent
def interest_from_7_percent : ℝ := investment_at_7_percent * 0.07
def interest_from_remaining : ℝ := total_interest - interest_from_7_percent

theorem remaining_investment_rate : 
  (interest_from_remaining / remaining_investment) * 100 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remaining_investment_rate_l2613_261336


namespace NUMINAMATH_CALUDE_tangent_condition_l2613_261316

-- Define the curve and line
def curve (x y : ℝ) : Prop := x^2 + 3*y^2 = 12
def line (m x y : ℝ) : Prop := m*x + y = 16

-- Define the tangency condition
def is_tangent (m : ℝ) : Prop := ∃! p : ℝ × ℝ, curve p.1 p.2 ∧ line m p.1 p.2

-- State the theorem
theorem tangent_condition (m : ℝ) : is_tangent m → m^2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_tangent_condition_l2613_261316


namespace NUMINAMATH_CALUDE_financial_equation_proof_l2613_261386

-- Define variables
variable (q v j p : ℝ)

-- Define the theorem
theorem financial_equation_proof :
  (3 * q - v = 8000) →
  (q = 4) →
  (v = 4 + 50 * j) →
  (p = 2669 + (50/3) * j) := by
sorry

end NUMINAMATH_CALUDE_financial_equation_proof_l2613_261386


namespace NUMINAMATH_CALUDE_extreme_values_imply_b_zero_l2613_261337

/-- A cubic function with extreme values at 1 and -1 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x * (a * x^2 + b * x + c)

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem extreme_values_imply_b_zero (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : f' a b c 1 = 0) (h3 : f' a b c (-1) = 0) : b = 0 := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_imply_b_zero_l2613_261337


namespace NUMINAMATH_CALUDE_equation_D_is_quadratic_l2613_261349

/-- A quadratic equation is an equation of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (a b c : ℝ) : Prop := a ≠ 0

/-- The equation 3x² + 1 = 0 -/
def equation_D : ℝ → Prop := fun x ↦ 3 * x^2 + 1 = 0

theorem equation_D_is_quadratic :
  is_quadratic_equation 3 0 1 := by sorry

end NUMINAMATH_CALUDE_equation_D_is_quadratic_l2613_261349


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2613_261310

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0) → -8 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2613_261310


namespace NUMINAMATH_CALUDE_perpendicular_line_through_M_l2613_261345

-- Define the line l: 2x - y - 4 = 0
def line_l (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Define point M as the intersection of line l with the x-axis
def point_M : ℝ × ℝ := (2, 0)

-- Define the perpendicular line: x + 2y - 2 = 0
def perp_line (x y : ℝ) : Prop := x + 2 * y - 2 = 0

-- Theorem statement
theorem perpendicular_line_through_M :
  (perp_line (point_M.1) (point_M.2)) ∧
  (∀ x y : ℝ, line_l x y → perp_line x y → 
    (y - point_M.2) * (x - point_M.1) = -(2 * (x - point_M.1) * (y - point_M.2))) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_M_l2613_261345


namespace NUMINAMATH_CALUDE_cubic_difference_l2613_261313

theorem cubic_difference (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_l2613_261313


namespace NUMINAMATH_CALUDE_laundry_cleaning_rate_l2613_261353

/-- Given a total number of laundry pieces and available hours, 
    calculate the number of pieces to be cleaned per hour -/
def pieces_per_hour (total_pieces : ℕ) (available_hours : ℕ) : ℕ :=
  total_pieces / available_hours

/-- Theorem stating that cleaning 80 pieces of laundry in 4 hours 
    requires cleaning 20 pieces per hour -/
theorem laundry_cleaning_rate : pieces_per_hour 80 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_laundry_cleaning_rate_l2613_261353


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2613_261327

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2613_261327


namespace NUMINAMATH_CALUDE_middle_bead_value_is_92_l2613_261303

/-- Represents a string of beads with specific properties -/
structure BeadString where
  total_beads : Nat
  middle_bead_index : Nat
  price_diff_left : Nat
  price_diff_right : Nat
  total_value : Nat

/-- Calculates the value of the middle bead in a BeadString -/
def middle_bead_value (bs : BeadString) : Nat :=
  sorry

/-- Theorem stating the value of the middle bead in the specific BeadString -/
theorem middle_bead_value_is_92 :
  let bs : BeadString := {
    total_beads := 31,
    middle_bead_index := 15,
    price_diff_left := 3,
    price_diff_right := 4,
    total_value := 2012
  }
  middle_bead_value bs = 92 := by sorry

end NUMINAMATH_CALUDE_middle_bead_value_is_92_l2613_261303


namespace NUMINAMATH_CALUDE_weight_of_A_l2613_261350

theorem weight_of_A (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 7 →
  (b + c + d + e) / 4 = 79 →
  a = 79 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_A_l2613_261350


namespace NUMINAMATH_CALUDE_orange_juice_fraction_l2613_261335

theorem orange_juice_fraction (pitcher_capacity : ℚ) 
  (orange_juice_fraction : ℚ) (apple_juice_fraction : ℚ) : 
  pitcher_capacity > 0 →
  orange_juice_fraction = 1/4 →
  apple_juice_fraction = 3/8 →
  (orange_juice_fraction * pitcher_capacity) / (2 * pitcher_capacity) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_fraction_l2613_261335


namespace NUMINAMATH_CALUDE_angle_A_measure_l2613_261346

-- Define the measure of angles A and B
def measure_A : ℝ := sorry
def measure_B : ℝ := sorry

-- Define the conditions
axiom supplementary : measure_A + measure_B = 180
axiom relation : measure_A = 3 * measure_B

-- Theorem to prove
theorem angle_A_measure : measure_A = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_measure_l2613_261346


namespace NUMINAMATH_CALUDE_floor_frac_equation_solutions_l2613_261359

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ :=
  x - (floor x : ℝ)

-- State the theorem
theorem floor_frac_equation_solutions :
  ∀ x : ℝ, (floor x : ℝ) * frac x = 2019 * x ↔ x = 0 ∨ x = -1/2020 := by
  sorry

end NUMINAMATH_CALUDE_floor_frac_equation_solutions_l2613_261359
