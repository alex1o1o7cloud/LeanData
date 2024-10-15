import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3235_323518

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 3 * x^2 - 6 * x
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3235_323518


namespace NUMINAMATH_CALUDE_incorrect_option_l3235_323568

/-- Represents the area of a rectangle with length 8 and width a -/
def option_a (a : ℝ) : ℝ := 8 * a

/-- Represents the selling price after a discount of 8% on an item priced a -/
def option_b (a : ℝ) : ℝ := 0.92 * a

/-- Represents the cost of 8 notebooks priced a each -/
def option_c (a : ℝ) : ℝ := 8 * a

/-- Represents the distance traveled at speed a for 8 hours -/
def option_d (a : ℝ) : ℝ := 8 * a

theorem incorrect_option (a : ℝ) : 
  option_a a = 8 * a ∧ 
  option_b a ≠ 8 * a ∧ 
  option_c a = 8 * a ∧ 
  option_d a = 8 * a :=
sorry

end NUMINAMATH_CALUDE_incorrect_option_l3235_323568


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3235_323567

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) : 
  2*x^2*y - 8*x*y + 8*y = 2*y*(x-2)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (x : ℝ) :
  x^4 - 81 = (x^2 + 9)*(x - 3)*(x + 3) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l3235_323567


namespace NUMINAMATH_CALUDE_only_2012_is_ternary_l3235_323542

def is_ternary (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 3 → d < 3

theorem only_2012_is_ternary :
  is_ternary 2012 ∧
  ¬is_ternary 2013 ∧
  ¬is_ternary 2014 ∧
  ¬is_ternary 2015 :=
by sorry

end NUMINAMATH_CALUDE_only_2012_is_ternary_l3235_323542


namespace NUMINAMATH_CALUDE_part1_part2_l3235_323506

-- Define the conditions p and q as functions of x and m
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0

def q (x m : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part 1
theorem part1 (m : ℝ) (h : m > 0) :
  (∀ x, p x → q x m) → m ≥ 4 := by sorry

-- Part 2
theorem part2 (x : ℝ) :
  (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) →
  (x ∈ Set.Icc (-3) (-2) ∪ Set.Ioc 6 7) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3235_323506


namespace NUMINAMATH_CALUDE_factorial_division_l3235_323528

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 5 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3235_323528


namespace NUMINAMATH_CALUDE_martian_calendar_months_l3235_323508

/-- Represents the number of days in a Martian month -/
inductive MartianMonth
  | long : MartianMonth  -- 100 days
  | short : MartianMonth -- 77 days

/-- Calculates the number of days in a Martian month -/
def daysInMonth (m : MartianMonth) : Nat :=
  match m with
  | MartianMonth.long => 100
  | MartianMonth.short => 77

/-- Represents a Martian calendar year -/
structure MartianYear where
  months : List MartianMonth
  total_days : Nat
  total_days_eq : total_days = List.sum (months.map daysInMonth)

/-- The theorem to be proved -/
theorem martian_calendar_months (year : MartianYear) 
    (h : year.total_days = 5882) : year.months.length = 74 := by
  sorry

#check martian_calendar_months

end NUMINAMATH_CALUDE_martian_calendar_months_l3235_323508


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3235_323572

/-- A line passing through (1,1) and perpendicular to x+2y-3=0 has the equation y=2x-1 -/
theorem perpendicular_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  (∀ p : ℝ × ℝ, p ∈ l ↔ p.1 + 2 * p.2 - 3 = 0) →  -- Definition of line l'
  ((1, 1) ∈ l) →  -- l passes through (1,1)
  (∀ p q : ℝ × ℝ, p ∈ l → q ∈ l → p ≠ q → (p.1 - q.1) * (p.1 + 2 * p.2 - 3 - (q.1 + 2 * q.2 - 3)) = 0) →  -- l is perpendicular to l'
  (∀ p : ℝ × ℝ, p ∈ l ↔ p.2 = 2 * p.1 - 1) :=  -- l has equation y=2x-1
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3235_323572


namespace NUMINAMATH_CALUDE_storks_on_fence_l3235_323539

theorem storks_on_fence (initial_birds : ℕ) (new_birds : ℕ) (total_birds : ℕ) :
  initial_birds = 4 →
  new_birds = 6 →
  total_birds = 10 →
  initial_birds + new_birds = total_birds →
  ∃ storks : ℕ, storks = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_on_fence_l3235_323539


namespace NUMINAMATH_CALUDE_area_ratio_octagon_quadrilateral_l3235_323500

/-- Regular octagon with vertices ABCDEFGH -/
structure RegularOctagon where
  area : ℝ

/-- Quadrilateral ACEG within the regular octagon -/
structure Quadrilateral where
  area : ℝ

/-- Theorem stating that the ratio of the quadrilateral area to the octagon area is √2/2 -/
theorem area_ratio_octagon_quadrilateral (octagon : RegularOctagon) (quad : Quadrilateral) :
  quad.area / octagon.area = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_octagon_quadrilateral_l3235_323500


namespace NUMINAMATH_CALUDE_ice_cream_cone_types_l3235_323543

theorem ice_cream_cone_types (num_flavors : ℕ) (num_combinations : ℕ) (h1 : num_flavors = 4) (h2 : num_combinations = 8) :
  num_combinations / num_flavors = 2 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_cone_types_l3235_323543


namespace NUMINAMATH_CALUDE_avg_calculation_l3235_323532

/-- Calculates the average of two numbers -/
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

/-- Calculates the average of three numbers -/
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- The main theorem to prove -/
theorem avg_calculation : avg3 (avg3 2 2 0) (avg2 1 2) 1 = 23 / 18 := by
  sorry

end NUMINAMATH_CALUDE_avg_calculation_l3235_323532


namespace NUMINAMATH_CALUDE_triangle_properties_l3235_323524

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.sin t.A = t.a * Real.sin (2 * t.B))
  (h2 : t.b = Real.sqrt 10)
  (h3 : t.a + t.c = t.a * t.c) :
  t.B = π / 3 ∧ 
  (1/2) * t.a * t.c * Real.sin t.B = (5 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3235_323524


namespace NUMINAMATH_CALUDE_smallest_factor_of_4896_l3235_323591

theorem smallest_factor_of_4896 : 
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧ 
    10 ≤ b ∧ b < 100 ∧ 
    a * b = 4896 ∧ 
    (∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x * y = 4896 → min x y ≥ 32) :=
by sorry

end NUMINAMATH_CALUDE_smallest_factor_of_4896_l3235_323591


namespace NUMINAMATH_CALUDE_initial_distance_correct_l3235_323504

/-- The initial distance between Seonghyeon and Jisoo -/
def initial_distance : ℝ := 1200

/-- The distance Seonghyeon ran towards Jisoo -/
def distance_towards : ℝ := 200

/-- The distance Seonghyeon ran in the opposite direction -/
def distance_away : ℝ := 1000

/-- The final distance between Seonghyeon and Jisoo -/
def final_distance : ℝ := 2000

/-- Theorem stating that the initial distance is correct given the conditions -/
theorem initial_distance_correct : 
  initial_distance - distance_towards + distance_away = final_distance :=
by sorry

end NUMINAMATH_CALUDE_initial_distance_correct_l3235_323504


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l3235_323533

theorem value_of_a_minus_b (a b : ℤ) 
  (eq1 : 3015 * a + 3019 * b = 3023) 
  (eq2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l3235_323533


namespace NUMINAMATH_CALUDE_cookies_remaining_l3235_323513

/-- Given the initial number of cookies and the number of cookies taken by each person,
    prove that the remaining number of cookies is 6. -/
theorem cookies_remaining (initial : ℕ) (eaten : ℕ) (brother : ℕ) (friend1 : ℕ) (friend2 : ℕ) (friend3 : ℕ)
    (h1 : initial = 22)
    (h2 : eaten = 2)
    (h3 : brother = 1)
    (h4 : friend1 = 3)
    (h5 : friend2 = 5)
    (h6 : friend3 = 5) :
    initial - eaten - brother - friend1 - friend2 - friend3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cookies_remaining_l3235_323513


namespace NUMINAMATH_CALUDE_parabola_properties_l3235_323511

/-- Parabola properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  h : a ≠ 0

/-- Points on the parabola -/
structure ParabolaPoints (p : Parabola) where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  h₁ : y₁ = p.a * x₁^2 + p.b * x₁
  h₂ : y₂ = p.a * x₂^2 + p.b * x₂
  h₃ : x₁ < x₂
  h₄ : x₁ + x₂ = 2

/-- Theorem about parabola properties -/
theorem parabola_properties (p : Parabola) (pts : ParabolaPoints p)
  (h₁ : p.a * 3^2 + p.b * 3 = 3) :
  (p.b = 1 - 3 * p.a) ∧
  (pts.y₁ = pts.y₂ → p.a = 1) ∧
  (pts.y₁ < pts.y₂ → 0 < p.a ∧ p.a < 1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3235_323511


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_l3235_323525

theorem greatest_integer_fraction (x : ℤ) :
  x ≠ 3 →
  (∃ y : ℤ, (x^2 + 2*x + 5) = (x - 3) * y) →
  x ≤ 23 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_l3235_323525


namespace NUMINAMATH_CALUDE_cost_increase_operation_l3235_323576

/-- Represents the cost function -/
def cost (t : ℝ) (b : ℝ) : ℝ := t * b^4

/-- Theorem: If the new cost after an operation on b is 1600% of the original cost,
    then the operation performed on b is multiplication by 2 -/
theorem cost_increase_operation (t : ℝ) (b₀ b₁ : ℝ) (h : t > 0) :
  cost t b₁ = 16 * cost t b₀ → b₁ = 2 * b₀ :=
by sorry

end NUMINAMATH_CALUDE_cost_increase_operation_l3235_323576


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_two_distinct_roots_l3235_323545

theorem at_least_one_equation_has_two_distinct_roots
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  ¬(4*b^2 - 4*a*c ≤ 0 ∧ 4*c^2 - 4*a*b ≤ 0 ∧ 4*a^2 - 4*b*c ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_two_distinct_roots_l3235_323545


namespace NUMINAMATH_CALUDE_tank_solution_volume_l3235_323527

theorem tank_solution_volume 
  (V : ℝ) 
  (h1 : V > 0) 
  (h2 : 0.05 * V / (V - 5500) = 1 / 9) : 
  V = 10000 := by
sorry

end NUMINAMATH_CALUDE_tank_solution_volume_l3235_323527


namespace NUMINAMATH_CALUDE_distance_to_incenter_l3235_323526

/-- An isosceles right triangle with side length 6√2 -/
structure IsoscelesRightTriangle where
  /-- The length of the equal sides -/
  side_length : ℝ
  /-- The side length is 6√2 -/
  side_length_eq : side_length = 6 * Real.sqrt 2

/-- The incenter of a triangle -/
def incenter (t : IsoscelesRightTriangle) : ℝ × ℝ := sorry

/-- The distance from a vertex to a point -/
def distance_to_point (v : ℝ × ℝ) (p : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the distance from the right angle vertex to the incenter -/
theorem distance_to_incenter (t : IsoscelesRightTriangle) : 
  distance_to_point (0, t.side_length) (incenter t) = 6 - 3 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_incenter_l3235_323526


namespace NUMINAMATH_CALUDE_tree_growth_theorem_l3235_323554

/-- Calculates the height of a tree after a given number of months -/
def treeHeight (initialHeight : ℝ) (growthRate : ℝ) (weeksPerMonth : ℕ) (months : ℕ) : ℝ :=
  initialHeight + growthRate * (months * weeksPerMonth : ℝ)

/-- Theorem: A tree with initial height 10 feet, growing 2 feet per week, 
    will be 42 feet tall after 4 months (with 4 weeks per month) -/
theorem tree_growth_theorem : 
  treeHeight 10 2 4 4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_theorem_l3235_323554


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l3235_323514

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  let side := d / Real.sqrt 2
  side * side = 72 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l3235_323514


namespace NUMINAMATH_CALUDE_inverse_f_of_3_l3235_323575

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem inverse_f_of_3 :
  ∃ (y : ℝ), y < 0 ∧ f y = 3 ∧ ∀ (z : ℝ), z < 0 ∧ f z = 3 → z = y :=
by sorry

end NUMINAMATH_CALUDE_inverse_f_of_3_l3235_323575


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3235_323578

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Existence of the hyperbola
  (∃ x : ℝ, x = 2 ∧ x^2 / a^2 - 0^2 / b^2 = 1) →  -- Right vertex at (2, 0)
  (∃ c : ℝ, c / a = 3/2) →  -- Eccentricity is 3/2
  (∃ x y : ℝ, y^2 = 8*x) →  -- Existence of the parabola
  (∀ x y : ℝ, x^2 / 4 - y^2 / 5 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3235_323578


namespace NUMINAMATH_CALUDE_isosceles_triangle_midpoint_property_l3235_323534

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the properties of the isosceles triangle
def IsIsosceles (t : Triangle) (a : ℝ) : Prop :=
  dist t.X t.Y = a ∧ dist t.Y t.Z = a

-- Define point M on XZ
def PointOnXZ (t : Triangle) (M : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ M = (1 - k) • t.X + k • t.Z

-- Define the midpoint property of M
def IsMidpoint (t : Triangle) (M : ℝ × ℝ) : Prop :=
  dist t.Y M = dist M t.Z

-- Define the sum of distances property
def SumOfDistances (t : Triangle) (M : ℝ × ℝ) (a : ℝ) : Prop :=
  dist t.X M + dist M t.Z = 2 * a

-- Main theorem
theorem isosceles_triangle_midpoint_property
  (t : Triangle) (M : ℝ × ℝ) (a : ℝ) :
  IsIsosceles t a →
  PointOnXZ t M →
  IsMidpoint t M →
  SumOfDistances t M a →
  dist t.Y M = a / 2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_midpoint_property_l3235_323534


namespace NUMINAMATH_CALUDE_tablet_diagonal_l3235_323596

/-- Given two square tablets, if one has a diagonal of 5 inches and the other has a screen area 5.5 square inches larger, then the diagonal of the larger tablet is 6 inches. -/
theorem tablet_diagonal (d : ℝ) : 
  (d ^ 2 / 2 = 25 / 2 + 5.5) → d = 6 := by
  sorry

end NUMINAMATH_CALUDE_tablet_diagonal_l3235_323596


namespace NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l3235_323502

theorem unique_solution_3x_4y_5z :
  ∀ (x y z : ℕ), 3^x + 4^y = 5^z → (x = 2 ∧ y = 2 ∧ z = 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_3x_4y_5z_l3235_323502


namespace NUMINAMATH_CALUDE_base5_413_equals_108_l3235_323510

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (a b c : ℕ) : ℕ := a * 5^2 + b * 5^1 + c * 5^0

/-- The base 5 number 413₅ is equal to 108 in base 10 --/
theorem base5_413_equals_108 : base5ToBase10 4 1 3 = 108 := by sorry

end NUMINAMATH_CALUDE_base5_413_equals_108_l3235_323510


namespace NUMINAMATH_CALUDE_survey_respondents_l3235_323564

/-- The number of people who preferred brand X -/
def X : ℕ := 60

/-- The number of people who preferred brand Y -/
def Y : ℕ := X / 3

/-- The number of people who preferred brand Z -/
def Z : ℕ := X * 3 / 2

/-- The total number of respondents to the survey -/
def total_respondents : ℕ := X + Y + Z

/-- Theorem stating that the total number of respondents is 170 -/
theorem survey_respondents : total_respondents = 170 := by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l3235_323564


namespace NUMINAMATH_CALUDE_difference_of_squares_153_147_l3235_323521

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_153_147_l3235_323521


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3235_323523

/-- A line is tangent to a circle if and only if the distance from the center of the circle to the line equals the radius of the circle. -/
axiom line_tangent_to_circle_iff_distance_eq_radius {a b c : ℝ} {x₀ y₀ r : ℝ} :
  (∀ x y, (x - x₀)^2 + (y - y₀)^2 = r^2 → a*x + b*y + c ≠ 0) ↔
  |a*x₀ + b*y₀ + c| / Real.sqrt (a^2 + b^2) = r

/-- The theorem to be proved -/
theorem tangent_line_to_circle (m : ℝ) :
  m > 0 →
  (∀ x y, (x - 3)^2 + (y - 4)^2 = 4 → 3*x - 4*y - m ≠ 0) →
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3235_323523


namespace NUMINAMATH_CALUDE_june1st_is_tuesday_l3235_323566

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific year -/
structure Year where
  febHasFiveSundays : Bool
  febHas29Days : Bool
  feb1stIsSunday : Bool

/-- Function to calculate the day of the week for June 1st -/
def june1stDayOfWeek (y : Year) : DayOfWeek :=
  sorry

/-- Theorem stating that June 1st is a Tuesday in the specified year -/
theorem june1st_is_tuesday (y : Year) 
  (h1 : y.febHasFiveSundays = true) 
  (h2 : y.febHas29Days = true) 
  (h3 : y.feb1stIsSunday = true) : 
  june1stDayOfWeek y = DayOfWeek.Tuesday :=
  sorry

end NUMINAMATH_CALUDE_june1st_is_tuesday_l3235_323566


namespace NUMINAMATH_CALUDE_count_non_divisors_is_33_l3235_323563

/-- g(n) is the product of the proper positive integer divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- The number of integers n between 2 and 100 (inclusive) that do not divide g(n) -/
def count_non_divisors : ℕ := sorry

/-- Theorem stating that the count of non-divisors is 33 -/
theorem count_non_divisors_is_33 : count_non_divisors = 33 := by sorry

end NUMINAMATH_CALUDE_count_non_divisors_is_33_l3235_323563


namespace NUMINAMATH_CALUDE_speed_ratio_is_two_sevenths_l3235_323536

/-- Two objects A and B moving uniformly along perpendicular paths -/
structure MovingObjects where
  vA : ℝ  -- Speed of object A
  vB : ℝ  -- Speed of object B

/-- The conditions of the problem -/
def satisfies_conditions (obj : MovingObjects) : Prop :=
  ∃ (t₁ t₂ : ℝ),
    t₁ > 0 ∧ t₂ > t₁ ∧
    (obj.vA * t₁)^2 = (750 - obj.vB * t₁)^2 ∧
    (obj.vA * t₂)^2 = (750 - obj.vB * t₂)^2 ∧
    t₂ - t₁ = 6 ∧
    t₁ = 3

/-- The theorem statement -/
theorem speed_ratio_is_two_sevenths (obj : MovingObjects) 
  (h : satisfies_conditions obj) : 
  obj.vA / obj.vB = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_speed_ratio_is_two_sevenths_l3235_323536


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3235_323503

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^15 + 11^21) ∧ ∀ q, Nat.Prime q → q ∣ (3^15 + 11^21) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3235_323503


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l3235_323517

/-- A sequence of three real numbers is geometric if the ratio between consecutive terms is constant. -/
def IsGeometricSequence (x y z : ℝ) : Prop :=
  y / x = z / y

/-- The statement that a = ±6 is equivalent to the sequence 4, a, 9 being geometric. -/
theorem geometric_sequence_condition :
  ∀ a : ℝ, IsGeometricSequence 4 a 9 ↔ (a = 6 ∨ a = -6) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l3235_323517


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l3235_323584

/-- Calculates the overall profit percentage for a retailer selling three items -/
theorem retailer_profit_percentage
  (radio_purchase : ℝ) (tv_purchase : ℝ) (speaker_purchase : ℝ)
  (radio_overhead : ℝ) (tv_overhead : ℝ) (speaker_overhead : ℝ)
  (radio_selling : ℝ) (tv_selling : ℝ) (speaker_selling : ℝ)
  (h1 : radio_purchase = 225)
  (h2 : tv_purchase = 4500)
  (h3 : speaker_purchase = 1500)
  (h4 : radio_overhead = 30)
  (h5 : tv_overhead = 200)
  (h6 : speaker_overhead = 100)
  (h7 : radio_selling = 300)
  (h8 : tv_selling = 5400)
  (h9 : speaker_selling = 1800) :
  let total_cp := radio_purchase + tv_purchase + speaker_purchase +
                  radio_overhead + tv_overhead + speaker_overhead
  let total_sp := radio_selling + tv_selling + speaker_selling
  let profit := total_sp - total_cp
  let profit_percentage := (profit / total_cp) * 100
  abs (profit_percentage - 14.42) < 0.01 := by
sorry


end NUMINAMATH_CALUDE_retailer_profit_percentage_l3235_323584


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3235_323559

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - x < 0.1) → (-0.1 < x ∧ x < 1.1) ∧
  ¬(∀ x : ℝ, (-0.1 < x ∧ x < 1.1) → (x^2 - x < 0.1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3235_323559


namespace NUMINAMATH_CALUDE_prob_one_one_ten_dice_l3235_323574

/-- The probability of rolling exactly one 1 out of 10 standard 6-sided dice -/
def prob_one_one (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  ↑(n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of rolling exactly one 1 out of 10 standard 6-sided dice
    is equal to (10 * 5^9) / 6^10 -/
theorem prob_one_one_ten_dice :
  prob_one_one 10 1 (1/6) = (10 * 5^9) / 6^10 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_one_ten_dice_l3235_323574


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3235_323535

/-- The length of the major axis of an ellipse with given foci and tangent to y-axis -/
theorem ellipse_major_axis_length :
  let f1 : ℝ × ℝ := (15, 10)
  let f2 : ℝ × ℝ := (35, 40)
  let ellipse := {p : ℝ × ℝ | ∃ k, dist p f1 + dist p f2 = k}
  let tangent_to_y_axis := ∃ y, (0, y) ∈ ellipse
  let major_axis_length := Real.sqrt 3400
  tangent_to_y_axis →
  ∃ a b : ℝ × ℝ, a ∈ ellipse ∧ b ∈ ellipse ∧ dist a b = major_axis_length :=
by sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3235_323535


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a8_l3235_323516

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 = 2 → a 4 = 6 → a 8 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a8_l3235_323516


namespace NUMINAMATH_CALUDE_sweetsies_remainder_l3235_323562

theorem sweetsies_remainder (m : ℕ) (h : m % 7 = 5) : (2 * m) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sweetsies_remainder_l3235_323562


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3235_323581

theorem solve_linear_equation :
  ∃ x : ℚ, -3 * x - 10 = 4 * x + 5 ∧ x = -15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3235_323581


namespace NUMINAMATH_CALUDE_max_nine_letter_palindromes_l3235_323544

/-- The number of letters in the English alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the palindromes we're considering -/
def palindrome_length : ℕ := 9

/-- A palindrome is a word that reads the same forward and backward -/
def is_palindrome (word : List Char) : Prop :=
  word = word.reverse

/-- The maximum number of 9-letter palindromes using the English alphabet -/
theorem max_nine_letter_palindromes :
  (alphabet_size ^ ((palindrome_length - 1) / 2 + 1) : ℕ) = 11881376 :=
sorry

end NUMINAMATH_CALUDE_max_nine_letter_palindromes_l3235_323544


namespace NUMINAMATH_CALUDE_parabola_line_intersection_slopes_l3235_323587

/-- Given a parabola y^2 = 2px and a line intersecting it at points A and B, 
    if the slope of OA is 2 and the slope of AB is 6, then the slope of OB is -3. -/
theorem parabola_line_intersection_slopes (p : ℝ) (y₁ y₂ : ℝ) : 
  let A := (y₁^2 / (2*p), y₁)
  let B := (y₂^2 / (2*p), y₂)
  let k_OA := y₁ / (y₁^2 / (2*p))
  let k_AB := (y₂ - y₁) / (y₂^2 / (2*p) - y₁^2 / (2*p))
  let k_OB := y₂ / (y₂^2 / (2*p))
  k_OA = 2 ∧ k_AB = 6 → k_OB = -3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_slopes_l3235_323587


namespace NUMINAMATH_CALUDE_jerry_hawk_feathers_l3235_323530

/-- The number of hawk feathers Jerry found -/
def hawk_feathers : ℕ := 6

/-- The number of eagle feathers Jerry found -/
def eagle_feathers : ℕ := 17 * hawk_feathers

/-- The total number of feathers Jerry initially had -/
def total_feathers : ℕ := hawk_feathers + eagle_feathers

/-- The number of feathers Jerry had after giving 10 to his sister -/
def feathers_after_giving : ℕ := total_feathers - 10

/-- The number of feathers Jerry had after selling half of the remaining feathers -/
def feathers_after_selling : ℕ := feathers_after_giving / 2

theorem jerry_hawk_feathers :
  hawk_feathers = 6 ∧
  eagle_feathers = 17 * hawk_feathers ∧
  total_feathers = hawk_feathers + eagle_feathers ∧
  feathers_after_giving = total_feathers - 10 ∧
  feathers_after_selling = feathers_after_giving / 2 ∧
  feathers_after_selling = 49 :=
by sorry

end NUMINAMATH_CALUDE_jerry_hawk_feathers_l3235_323530


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3235_323582

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 2 → x ≠ 4 →
  (6 * x^2 + 3 * x) / ((x - 4) * (x - 2)^3) =
  13.5 / (x - 4) + (-27) / (x - 2) + (-15) / (x - 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3235_323582


namespace NUMINAMATH_CALUDE_overhead_percentage_example_l3235_323531

/-- Given the purchase price, markup, and net profit of an article, 
    calculate the percentage of cost for overhead. -/
def overhead_percentage (purchase_price markup net_profit : ℚ) : ℚ :=
  let overhead := markup - net_profit
  (overhead / purchase_price) * 100

/-- Theorem stating that for the given values, the overhead percentage is 58.33% -/
theorem overhead_percentage_example : 
  overhead_percentage 48 40 12 = 58.33 := by
  sorry

end NUMINAMATH_CALUDE_overhead_percentage_example_l3235_323531


namespace NUMINAMATH_CALUDE_min_c_value_l3235_323573

/-- Given positive integers a, b, c satisfying a < b < c, and a system of equations
    with exactly one solution, prove that the minimum value of c is 2002. -/
theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hab : a < b) (hbc : b < c)
    (h_unique : ∃! (x y : ℝ), 2 * x + y = 2004 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 2002 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 2002 ∧
    ∃! (x y : ℝ), 2 * x + y = 2004 ∧ y = |x - a'| + |x - b'| + |x - 2002| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l3235_323573


namespace NUMINAMATH_CALUDE_total_games_played_l3235_323579

-- Define the parameters
def win_percentage : ℚ := 50 / 100
def games_won : ℕ := 70

-- State the theorem
theorem total_games_played : ℕ := by
  -- The proof goes here
  sorry

-- The goal to prove
#check total_games_played = 140

end NUMINAMATH_CALUDE_total_games_played_l3235_323579


namespace NUMINAMATH_CALUDE_sum_of_integers_l3235_323565

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 130)
  (h2 : x * y = 45) : 
  ∃ (ε : ℝ), abs ((x : ℝ) + y - 15) < ε ∧ ε > 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3235_323565


namespace NUMINAMATH_CALUDE_apple_tree_difference_l3235_323550

theorem apple_tree_difference (ava_trees lily_trees total_trees : ℕ) : 
  ava_trees = 9 → 
  total_trees = 15 → 
  lily_trees = total_trees - ava_trees → 
  ava_trees - lily_trees = 3 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_difference_l3235_323550


namespace NUMINAMATH_CALUDE_robert_nickel_chocolate_difference_l3235_323561

theorem robert_nickel_chocolate_difference :
  let robert_chocolates : ℕ := 9
  let nickel_chocolates : ℕ := 2
  robert_chocolates - nickel_chocolates = 7 := by
sorry

end NUMINAMATH_CALUDE_robert_nickel_chocolate_difference_l3235_323561


namespace NUMINAMATH_CALUDE_nonagon_perimeter_l3235_323580

/-- A regular nonagon is a polygon with 9 sides of equal length and equal angles -/
structure RegularNonagon where
  sideLength : ℝ
  numSides : ℕ
  numSides_eq : numSides = 9

/-- The perimeter of a regular nonagon is the product of its number of sides and side length -/
def perimeter (n : RegularNonagon) : ℝ := n.numSides * n.sideLength

/-- Theorem: The perimeter of a regular nonagon with side length 2 cm is 18 cm -/
theorem nonagon_perimeter :
  ∀ (n : RegularNonagon), n.sideLength = 2 → perimeter n = 18 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_perimeter_l3235_323580


namespace NUMINAMATH_CALUDE_orthogonality_iff_k_eq_4_l3235_323512

/-- Two unit vectors with an angle of 60° between them -/
structure UnitVectorPair :=
  (e₁ e₂ : ℝ × ℝ)
  (unit_e₁ : e₁.1 ^ 2 + e₁.2 ^ 2 = 1)
  (unit_e₂ : e₂.1 ^ 2 + e₂.2 ^ 2 = 1)
  (angle_60 : e₁.1 * e₂.1 + e₁.2 * e₂.2 = 1/2)

/-- The orthogonality condition -/
def orthogonality (v : UnitVectorPair) (k : ℝ) : Prop :=
  (2 * v.e₁.1 - k * v.e₂.1) * v.e₁.1 + (2 * v.e₁.2 - k * v.e₂.2) * v.e₁.2 = 0

/-- The main theorem -/
theorem orthogonality_iff_k_eq_4 (v : UnitVectorPair) :
  orthogonality v 4 ∧ (∀ k : ℝ, orthogonality v k → k = 4) :=
sorry

end NUMINAMATH_CALUDE_orthogonality_iff_k_eq_4_l3235_323512


namespace NUMINAMATH_CALUDE_roots_equation_result_l3235_323557

theorem roots_equation_result (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 + x₁ - 4 = 0) 
  (h₂ : x₂^2 + x₂ - 4 = 0) 
  (h₃ : x₁ ≠ x₂) : 
  x₁^3 - 5*x₂^2 + 10 = -19 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_result_l3235_323557


namespace NUMINAMATH_CALUDE_tom_bob_sticker_ratio_l3235_323586

def bob_stickers : ℕ := 12

theorem tom_bob_sticker_ratio :
  ∃ (tom_stickers : ℕ),
    tom_stickers = bob_stickers ∧
    tom_stickers / bob_stickers = 1 := by
  sorry

end NUMINAMATH_CALUDE_tom_bob_sticker_ratio_l3235_323586


namespace NUMINAMATH_CALUDE_base_prime_441_l3235_323501

/-- Base prime representation of a natural number --/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- 441 in base prime representation --/
theorem base_prime_441 : base_prime_repr 441 = [0, 2, 2, 0] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_441_l3235_323501


namespace NUMINAMATH_CALUDE_number_from_percentages_l3235_323519

theorem number_from_percentages (x : ℝ) : 
  0.15 * 0.30 * 0.50 * x = 126 → x = 5600 := by
  sorry

end NUMINAMATH_CALUDE_number_from_percentages_l3235_323519


namespace NUMINAMATH_CALUDE_system_solution_l3235_323546

theorem system_solution (x y k : ℝ) : 
  x - y = k - 3 →
  3 * x + 5 * y = 2 * k + 8 →
  x + y = 2 →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3235_323546


namespace NUMINAMATH_CALUDE_range_of_a_l3235_323556

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x + 1| ≤ 3 * a - a^2) → 1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3235_323556


namespace NUMINAMATH_CALUDE_german_team_goals_l3235_323560

def journalist1 (x : ℕ) : Prop := 10 < x ∧ x < 17

def journalist2 (x : ℕ) : Prop := 11 < x ∧ x < 18

def journalist3 (x : ℕ) : Prop := x % 2 = 1

def twoCorrect (x : ℕ) : Prop :=
  (journalist1 x ∧ journalist2 x ∧ ¬journalist3 x) ∨
  (journalist1 x ∧ ¬journalist2 x ∧ journalist3 x) ∨
  (¬journalist1 x ∧ journalist2 x ∧ journalist3 x)

theorem german_team_goals :
  ∀ x : ℕ, twoCorrect x ↔ x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
by sorry

end NUMINAMATH_CALUDE_german_team_goals_l3235_323560


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3235_323507

theorem binomial_coefficient_ratio (n : ℕ) : 4^n / 2^n = 64 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l3235_323507


namespace NUMINAMATH_CALUDE_f_of_f_one_eq_two_l3235_323593

def f (x : ℝ) : ℝ := 4 * x^2 - 6 * x + 2

theorem f_of_f_one_eq_two : f (f 1) = 2 := by sorry

end NUMINAMATH_CALUDE_f_of_f_one_eq_two_l3235_323593


namespace NUMINAMATH_CALUDE_pig_purchase_equation_l3235_323592

/-- Represents a group purchase of pigs -/
structure PigPurchase where
  numPeople : ℕ
  excessAmount : ℕ
  exactAmount : ℕ

/-- The equation for the pig purchase problem is correct -/
theorem pig_purchase_equation (p : PigPurchase) 
  (h1 : p.numPeople * p.excessAmount - p.numPeople * p.exactAmount = p.excessAmount) 
  (h2 : p.excessAmount = 100) 
  (h3 : p.exactAmount = 90) : 
  100 * p.numPeople - 90 * p.numPeople = 100 := by
  sorry

end NUMINAMATH_CALUDE_pig_purchase_equation_l3235_323592


namespace NUMINAMATH_CALUDE_inequality_proof_l3235_323547

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a * b^2 * c^3 * d^4 ≤ ((a + 2*b + 3*c + 4*d) / 10)^10 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3235_323547


namespace NUMINAMATH_CALUDE_aluminum_carbonate_weight_l3235_323589

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The number of moles of Aluminum carbonate -/
def moles : ℝ := 5

/-- The molecular weight of Aluminum carbonate (Al2(CO3)3) in g/mol -/
def Al2CO3_3_weight : ℝ := 2 * Al_weight + 3 * C_weight + 9 * O_weight

/-- The total weight of the given moles of Aluminum carbonate in grams -/
def total_weight : ℝ := moles * Al2CO3_3_weight

theorem aluminum_carbonate_weight : total_weight = 1169.95 := by sorry

end NUMINAMATH_CALUDE_aluminum_carbonate_weight_l3235_323589


namespace NUMINAMATH_CALUDE_apples_picked_l3235_323541

theorem apples_picked (initial_apples new_apples final_apples : ℕ) 
  (h1 : initial_apples = 11)
  (h2 : new_apples = 2)
  (h3 : final_apples = 6) :
  initial_apples - (initial_apples - new_apples - final_apples) = 7 := by
  sorry

end NUMINAMATH_CALUDE_apples_picked_l3235_323541


namespace NUMINAMATH_CALUDE_no_odd_three_digit_div_five_without_five_l3235_323537

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def divisible_by_five (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k

def does_not_contain_five (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≠ 5

theorem no_odd_three_digit_div_five_without_five :
  {n : ℕ | is_odd n ∧ is_three_digit n ∧ divisible_by_five n ∧ does_not_contain_five n} = ∅ :=
sorry

end NUMINAMATH_CALUDE_no_odd_three_digit_div_five_without_five_l3235_323537


namespace NUMINAMATH_CALUDE_complex_expression_equality_l3235_323571

/-- Given complex numbers x and y, prove that 3x + 4y = 17 + 2i -/
theorem complex_expression_equality (x y : ℂ) (hx : x = 3 + 2*I) (hy : y = 2 - I) :
  3*x + 4*y = 17 + 2*I := by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l3235_323571


namespace NUMINAMATH_CALUDE_total_dolls_count_l3235_323540

/-- The number of dolls owned by the grandmother -/
def grandmother_dolls : ℕ := 50

/-- The number of dolls owned by the sister -/
def sister_dolls : ℕ := grandmother_dolls + 2

/-- The number of dolls owned by Rene -/
def rene_dolls : ℕ := 3 * sister_dolls

/-- The total number of dolls owned by Rene, her sister, and their grandmother -/
def total_dolls : ℕ := grandmother_dolls + sister_dolls + rene_dolls

theorem total_dolls_count : total_dolls = 258 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_count_l3235_323540


namespace NUMINAMATH_CALUDE_median_name_length_l3235_323585

/-- Represents the distribution of name lengths -/
structure NameLengthDistribution where
  three_letter : Nat
  four_letter : Nat
  five_letter : Nat
  six_letter : Nat
  seven_letter : Nat

/-- The median of a list of natural numbers -/
def median (l : List Nat) : Nat :=
  sorry

/-- Generates a list of name lengths based on the distribution -/
def generateNameLengths (d : NameLengthDistribution) : List Nat :=
  sorry

theorem median_name_length (d : NameLengthDistribution) :
  d.three_letter = 6 →
  d.four_letter = 5 →
  d.five_letter = 2 →
  d.six_letter = 4 →
  d.seven_letter = 4 →
  d.three_letter + d.four_letter + d.five_letter + d.six_letter + d.seven_letter = 21 →
  median (generateNameLengths d) = 4 := by
  sorry

end NUMINAMATH_CALUDE_median_name_length_l3235_323585


namespace NUMINAMATH_CALUDE_simplify_expression_l3235_323583

theorem simplify_expression (b : ℚ) (h : b = 2) :
  (15 * b^4 - 45 * b^3) / (75 * b^2) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3235_323583


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3235_323520

theorem quadratic_inequality_solution (a : ℝ) :
  let solution_set := {x : ℝ | x^2 - a*x - 2*a^2 < 0}
  if a > 0 then
    solution_set = {x : ℝ | -a < x ∧ x < 2*a}
  else if a < 0 then
    solution_set = {x : ℝ | 2*a < x ∧ x < -a}
  else
    solution_set = ∅ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3235_323520


namespace NUMINAMATH_CALUDE_digit_symmetrical_equation_l3235_323522

theorem digit_symmetrical_equation (a b : ℤ) (h : 2 ≤ a + b ∧ a + b ≤ 9) :
  (10*a + b) * (100*b + 10*(a + b) + a) = (100*a + 10*(a + b) + b) * (10*b + a) := by
  sorry

end NUMINAMATH_CALUDE_digit_symmetrical_equation_l3235_323522


namespace NUMINAMATH_CALUDE_min_sum_of_chord_lengths_l3235_323588

/-- Parabola defined by y² = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Line passing through the focus with slope k -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * (p.1 - Focus.1)}

/-- Length of the chord formed by a line with slope k on the parabola -/
noncomputable def ChordLength (k : ℝ) : ℝ :=
  4 + 4 / k^2

/-- Sum of chord lengths for two lines with slopes k₁ and k₂ -/
noncomputable def SumOfChordLengths (k₁ k₂ : ℝ) : ℝ :=
  ChordLength k₁ + ChordLength k₂

/-- Theorem stating the minimum value of the sum of chord lengths -/
theorem min_sum_of_chord_lengths :
  ∀ k₁ k₂ : ℝ, k₁^2 + k₂^2 = 1 →
  24 ≤ SumOfChordLengths k₁ k₂ ∧
  (∃ k₁' k₂' : ℝ, k₁'^2 + k₂'^2 = 1 ∧ SumOfChordLengths k₁' k₂' = 24) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_chord_lengths_l3235_323588


namespace NUMINAMATH_CALUDE_power_of_negative_square_l3235_323553

theorem power_of_negative_square (x : ℝ) : (-2 * x^2)^3 = -8 * x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_square_l3235_323553


namespace NUMINAMATH_CALUDE_farm_animals_l3235_323594

/-- The number of animals in a farm with ducks and dogs -/
def total_animals (num_ducks : ℕ) (total_legs : ℕ) : ℕ :=
  num_ducks + (total_legs - 2 * num_ducks) / 4

/-- Theorem: Given the conditions, there are 11 animals in total -/
theorem farm_animals : total_animals 6 32 = 11 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l3235_323594


namespace NUMINAMATH_CALUDE_infinitely_many_n_with_bounded_prime_divisors_l3235_323505

theorem infinitely_many_n_with_bounded_prime_divisors :
  ∃ (S : Set ℕ), (Set.Infinite S) ∧ 
  (∀ n ∈ S, ∀ p : ℕ, Prime p → p ∣ (n^2 + n + 1) → p ≤ Real.sqrt n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_n_with_bounded_prime_divisors_l3235_323505


namespace NUMINAMATH_CALUDE_haley_current_height_l3235_323598

/-- Haley's growth rate in inches per year -/
def growth_rate : ℝ := 3

/-- Number of years in the future -/
def years : ℝ := 10

/-- Haley's height after 10 years in inches -/
def future_height : ℝ := 50

/-- Haley's current height in inches -/
def current_height : ℝ := future_height - growth_rate * years

theorem haley_current_height : current_height = 20 := by
  sorry

end NUMINAMATH_CALUDE_haley_current_height_l3235_323598


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3235_323549

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 10 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3235_323549


namespace NUMINAMATH_CALUDE_exists_permutation_9_not_exists_permutation_11_exists_permutation_1996_l3235_323590

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Theorem for n = 9 -/
theorem exists_permutation_9 :
  ∃ f : Fin 9 → Fin 9, Function.Bijective f ∧
    ∀ k : Fin 9, isPerfectSquare ((k : ℕ) + 1 + (f k : ℕ)) :=
sorry

/-- Theorem for n = 11 -/
theorem not_exists_permutation_11 :
  ¬ ∃ f : Fin 11 → Fin 11, Function.Bijective f ∧
    ∀ k : Fin 11, isPerfectSquare ((k : ℕ) + 1 + (f k : ℕ)) :=
sorry

/-- Theorem for n = 1996 -/
theorem exists_permutation_1996 :
  ∃ f : Fin 1996 → Fin 1996, Function.Bijective f ∧
    ∀ k : Fin 1996, isPerfectSquare ((k : ℕ) + 1 + (f k : ℕ)) :=
sorry

end NUMINAMATH_CALUDE_exists_permutation_9_not_exists_permutation_11_exists_permutation_1996_l3235_323590


namespace NUMINAMATH_CALUDE_solve_equation_l3235_323515

theorem solve_equation : ∃ x : ℝ, (12 : ℝ) ^ x * 6^2 / 432 = 144 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3235_323515


namespace NUMINAMATH_CALUDE_mixed_fraction_division_subtraction_l3235_323577

theorem mixed_fraction_division_subtraction :
  (1 + 5/6) / (2 + 3/4) - 1/2 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_mixed_fraction_division_subtraction_l3235_323577


namespace NUMINAMATH_CALUDE_wood_rope_measurement_l3235_323538

/-- Represents the relationship between the length of a piece of wood and a rope used to measure it. -/
theorem wood_rope_measurement (x y : ℝ) :
  y = x + 4.5 ∧ 0.5 * y = x - 1 →
  (y - x = 4.5 ∧ y / 2 - x = -1) :=
by sorry

end NUMINAMATH_CALUDE_wood_rope_measurement_l3235_323538


namespace NUMINAMATH_CALUDE_problem_statement_l3235_323599

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : 
  (x - 1)^2 + 16/((x - 1)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3235_323599


namespace NUMINAMATH_CALUDE_orange_pricing_theorem_l3235_323595

/-- Pricing tiers for oranges -/
def price_4 : ℕ := 15
def price_7 : ℕ := 25
def price_10 : ℕ := 32

/-- Number of groups purchased -/
def num_groups : ℕ := 3

/-- Total number of oranges purchased -/
def total_oranges : ℕ := 4 * num_groups + 7 * num_groups + 10 * num_groups

/-- Calculate the total cost in cents -/
def total_cost : ℕ := price_4 * num_groups + price_7 * num_groups + price_10 * num_groups

/-- Calculate the average cost per orange in cents (as a rational number) -/
def avg_cost_per_orange : ℚ := total_cost / total_oranges

theorem orange_pricing_theorem :
  total_oranges = 21 ∧ 
  total_cost = 216 ∧ 
  avg_cost_per_orange = 1029 / 100 := by
  sorry

end NUMINAMATH_CALUDE_orange_pricing_theorem_l3235_323595


namespace NUMINAMATH_CALUDE_bottle_caps_count_l3235_323551

/-- The number of bottle caps in the box after removing some and adding others. -/
def final_bottle_caps (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that given the initial conditions, the final number of bottle caps is 137. -/
theorem bottle_caps_count : final_bottle_caps 144 63 56 = 137 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_count_l3235_323551


namespace NUMINAMATH_CALUDE_april_cookie_spending_l3235_323548

/-- Calculates the total amount spent on cookies in April given the following conditions:
  * April has 30 days
  * On even days, 3 chocolate chip cookies and 2 sugar cookies are bought
  * On odd days, 4 oatmeal cookies and 1 snickerdoodle cookie are bought
  * Prices: chocolate chip $18, sugar $22, oatmeal $15, snickerdoodle $25
-/
theorem april_cookie_spending : 
  let days_in_april : ℕ := 30
  let even_days : ℕ := days_in_april / 2
  let odd_days : ℕ := days_in_april / 2
  let choc_chip_price : ℕ := 18
  let sugar_price : ℕ := 22
  let oatmeal_price : ℕ := 15
  let snickerdoodle_price : ℕ := 25
  let even_day_cost : ℕ := 3 * choc_chip_price + 2 * sugar_price
  let odd_day_cost : ℕ := 4 * oatmeal_price + 1 * snickerdoodle_price
  let total_cost : ℕ := even_days * even_day_cost + odd_days * odd_day_cost
  total_cost = 2745 := by
sorry


end NUMINAMATH_CALUDE_april_cookie_spending_l3235_323548


namespace NUMINAMATH_CALUDE_ellipse_max_dot_product_l3235_323558

/-- Definition of the ellipse M -/
def ellipse_M (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the circle N -/
def circle_N (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 1

/-- The ellipse passes through the point (2, √6) -/
def ellipse_point (a b : ℝ) : Prop :=
  ellipse_M a b 2 (Real.sqrt 6)

/-- The eccentricity of the ellipse is √2/2 -/
def ellipse_eccentricity (a b : ℝ) : Prop :=
  (Real.sqrt (a^2 - b^2)) / a = Real.sqrt 2 / 2

/-- Definition of the dot product PA · PB -/
def dot_product (x y : ℝ) : ℝ :=
  x^2 + y^2 - 4*y + 3

/-- Main theorem -/
theorem ellipse_max_dot_product (a b : ℝ) :
  ellipse_M a b 2 (Real.sqrt 6) →
  ellipse_eccentricity a b →
  (∀ x y : ℝ, ellipse_M a b x y → dot_product x y ≤ 23) ∧
  (∃ x y : ℝ, ellipse_M a b x y ∧ dot_product x y = 23) :=
sorry

end NUMINAMATH_CALUDE_ellipse_max_dot_product_l3235_323558


namespace NUMINAMATH_CALUDE_angle_terminal_side_value_l3235_323552

theorem angle_terminal_side_value (k : ℝ) (θ : ℝ) (h : k < 0) :
  (∃ (r : ℝ), r > 0 ∧ -4 * k = r * Real.cos θ ∧ 3 * k = r * Real.sin θ) →
  2 * Real.sin θ + Real.cos θ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_value_l3235_323552


namespace NUMINAMATH_CALUDE_youngest_son_cotton_correct_l3235_323509

/-- The amount of cotton for the youngest son in the "Dividing Cotton among Eight Sons" problem -/
def youngest_son_cotton : ℕ := 184

/-- The total amount of cotton to be divided -/
def total_cotton : ℕ := 996

/-- The number of sons -/
def num_sons : ℕ := 8

/-- The difference in cotton amount between each son -/
def cotton_difference : ℕ := 17

/-- Theorem stating that the youngest son's cotton amount is correct given the problem conditions -/
theorem youngest_son_cotton_correct :
  youngest_son_cotton * num_sons + (num_sons * (num_sons - 1) / 2) * cotton_difference = total_cotton :=
by sorry

end NUMINAMATH_CALUDE_youngest_son_cotton_correct_l3235_323509


namespace NUMINAMATH_CALUDE_smallest_class_size_class_with_25_students_exists_l3235_323570

/-- Represents a class of students who took a history test -/
structure HistoryClass where
  /-- The total number of students in the class -/
  num_students : ℕ
  /-- The total score of all students -/
  total_score : ℕ
  /-- The number of students who scored 120 points -/
  perfect_scores : ℕ
  /-- The number of students who scored 115 points -/
  near_perfect_scores : ℕ

/-- The properties of the history class based on the given problem -/
def valid_history_class (c : HistoryClass) : Prop :=
  c.perfect_scores = 8 ∧
  c.near_perfect_scores = 3 ∧
  c.total_score = c.num_students * 92 ∧
  c.total_score ≥ c.perfect_scores * 120 + c.near_perfect_scores * 115 + (c.num_students - c.perfect_scores - c.near_perfect_scores) * 70

/-- The theorem stating that the smallest possible number of students in the class is 25 -/
theorem smallest_class_size (c : HistoryClass) (h : valid_history_class c) : c.num_students ≥ 25 := by
  sorry

/-- The theorem stating that a class with 25 students satisfying all conditions exists -/
theorem class_with_25_students_exists : ∃ c : HistoryClass, valid_history_class c ∧ c.num_students = 25 := by
  sorry

end NUMINAMATH_CALUDE_smallest_class_size_class_with_25_students_exists_l3235_323570


namespace NUMINAMATH_CALUDE_oranges_per_box_l3235_323569

def total_oranges : ℕ := 45
def num_boxes : ℕ := 9

theorem oranges_per_box : total_oranges / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l3235_323569


namespace NUMINAMATH_CALUDE_merchant_profit_l3235_323555

theorem merchant_profit (cost : ℝ) (cost_positive : cost > 0) : 
  let marked_price := cost * 1.2
  let discounted_price := marked_price * 0.9
  let profit := discounted_price - cost
  let profit_percentage := (profit / cost) * 100
  profit_percentage = 8 := by sorry

end NUMINAMATH_CALUDE_merchant_profit_l3235_323555


namespace NUMINAMATH_CALUDE_simplify_expression_expand_expression_l3235_323597

-- Problem 1
theorem simplify_expression (a : ℝ) : (2 * a^2)^3 + (-3 * a^3)^2 = 17 * a^6 := by
  sorry

-- Problem 2
theorem expand_expression (x y : ℝ) : (x + 3*y) * (x - y) = x^2 + 2*x*y - 3*y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_expand_expression_l3235_323597


namespace NUMINAMATH_CALUDE_method2_more_profitable_above_15000_l3235_323529

/-- Profit calculation for Method 1 (end of month) -/
def profit_method1 (x : ℝ) : ℝ := 0.3 * x - 900

/-- Profit calculation for Method 2 (beginning of month with reinvestment) -/
def profit_method2 (x : ℝ) : ℝ := 0.26 * x

/-- Theorem stating that Method 2 is more profitable when x > 15000 -/
theorem method2_more_profitable_above_15000 (x : ℝ) (h : x > 15000) :
  profit_method2 x > profit_method1 x :=
sorry

end NUMINAMATH_CALUDE_method2_more_profitable_above_15000_l3235_323529
