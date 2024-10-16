import Mathlib

namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l2283_228343

theorem negative_fractions_comparison : -1/3 < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l2283_228343


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l2283_228351

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationship operators
variable (contained_in : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_implies_perpendicular_to_contained_line
  (m n : Line) (α : Plane)
  (h1 : contained_in m α)
  (h2 : perpendicular n α) :
  perpendicular_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l2283_228351


namespace NUMINAMATH_CALUDE_family_suitcases_l2283_228370

theorem family_suitcases (num_siblings : ℕ) (suitcases_per_sibling : ℕ) (parent_suitcases : ℕ) : 
  num_siblings = 4 →
  suitcases_per_sibling = 2 →
  parent_suitcases = 3 →
  num_siblings * suitcases_per_sibling + parent_suitcases * 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_family_suitcases_l2283_228370


namespace NUMINAMATH_CALUDE_sqrt_nested_equals_power_l2283_228366

theorem sqrt_nested_equals_power (N : ℝ) (h : N > 1) :
  Real.sqrt (N * Real.sqrt (N * Real.sqrt N)) = N^(7/8) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nested_equals_power_l2283_228366


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l2283_228318

/-- The curve y = x^3 + 2x has a tangent line at (1, 3) perpendicular to ax - y + 2019 = 0 -/
theorem tangent_perpendicular_to_line (a : ℝ) : 
  let f (x : ℝ) := x^3 + 2*x
  let f' (x : ℝ) := 3*x^2 + 2
  let tangent_slope := f' 1
  let perpendicular_slope := a
  (f 1 = 3) ∧ (tangent_slope * perpendicular_slope = -1) → a = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_line_l2283_228318


namespace NUMINAMATH_CALUDE_divisibility_condition_l2283_228332

def is_divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

theorem divisibility_condition (a b : ℕ) : 
  (a ≤ 9 ∧ b ≤ 9) →
  (is_divisible_by (62684 * 10 + a * 10 + b) 8 ∧ 
   is_divisible_by (62684 * 10 + a * 10 + b) 5) →
  (b = 0 ∧ (a = 0 ∨ a = 8)) := by
  sorry

#check divisibility_condition

end NUMINAMATH_CALUDE_divisibility_condition_l2283_228332


namespace NUMINAMATH_CALUDE_intersection_point_l2283_228381

def P : ℝ × ℝ × ℝ := (10, -1, 3)
def Q : ℝ × ℝ × ℝ := (20, -11, 8)
def R : ℝ × ℝ × ℝ := (3, 8, -9)
def S : ℝ × ℝ × ℝ := (5, 0, 6)

def line_PQ (t : ℝ) : ℝ × ℝ × ℝ :=
  (P.1 + t * (Q.1 - P.1), P.2.1 + t * (Q.2.1 - P.2.1), P.2.2 + t * (Q.2.2 - P.2.2))

def line_RS (s : ℝ) : ℝ × ℝ × ℝ :=
  (R.1 + s * (S.1 - R.1), R.2.1 + s * (S.2.1 - R.2.1), R.2.2 + s * (S.2.2 - R.2.2))

theorem intersection_point :
  ∃ t s : ℝ, line_PQ t = line_RS s ∧ line_PQ t = (11, -2, 3.5) := by sorry

end NUMINAMATH_CALUDE_intersection_point_l2283_228381


namespace NUMINAMATH_CALUDE_max_choir_members_correct_l2283_228330

/-- The maximum number of choir members that satisfies the given conditions. -/
def max_choir_members : ℕ := 266

/-- Predicate to check if a number satisfies the square formation condition. -/
def is_square_formation (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * k + 11

/-- Predicate to check if a number satisfies the rectangular formation condition. -/
def is_rectangular_formation (m : ℕ) : Prop :=
  ∃ n : ℕ, m = n * (n + 5)

/-- Theorem stating that max_choir_members satisfies both formation conditions
    and is the maximum number that does so. -/
theorem max_choir_members_correct :
  is_square_formation max_choir_members ∧
  is_rectangular_formation max_choir_members ∧
  ∀ m : ℕ, m > max_choir_members →
    ¬(is_square_formation m ∧ is_rectangular_formation m) :=
by sorry

end NUMINAMATH_CALUDE_max_choir_members_correct_l2283_228330


namespace NUMINAMATH_CALUDE_right_triangle_area_l2283_228369

/-- 
Given a right-angled triangle with perpendicular sides a and b,
prove that its area is 1/2 when a + b = 4 and a² + b² = 14
-/
theorem right_triangle_area (a b : ℝ) 
  (sum_sides : a + b = 4) 
  (sum_squares : a^2 + b^2 = 14) : 
  (1/2) * a * b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2283_228369


namespace NUMINAMATH_CALUDE_point_coordinates_l2283_228334

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the conditions for the point
def on_x_axis (p : Point) : Prop := p.2 = 0
def right_of_origin (p : Point) : Prop := p.1 > 0
def distance_from_origin (p : Point) (d : ℝ) : Prop := p.1^2 + p.2^2 = d^2

-- Theorem statement
theorem point_coordinates :
  ∀ (p : Point),
    on_x_axis p →
    right_of_origin p →
    distance_from_origin p 3 →
    p = (3, 0) :=
by
  sorry


end NUMINAMATH_CALUDE_point_coordinates_l2283_228334


namespace NUMINAMATH_CALUDE_mike_total_cards_l2283_228364

def initial_cards : ℕ := 87
def received_cards : ℕ := 13

theorem mike_total_cards : initial_cards + received_cards = 100 := by
  sorry

end NUMINAMATH_CALUDE_mike_total_cards_l2283_228364


namespace NUMINAMATH_CALUDE_jellybean_count_is_84_l2283_228376

/-- Calculates the final number of jellybeans in a jar after a series of actions. -/
def final_jellybean_count (initial : ℕ) (samantha_took : ℕ) (shelby_ate : ℕ) : ℕ :=
  let remaining_after_samantha := initial - samantha_took
  let remaining_after_shelby := remaining_after_samantha - shelby_ate
  let scarlett_returned := shelby_ate
  let shannon_added := (samantha_took + shelby_ate) / 2
  remaining_after_shelby + scarlett_returned + shannon_added

/-- Theorem stating that given the initial conditions, the final number of jellybeans is 84. -/
theorem jellybean_count_is_84 :
  final_jellybean_count 90 24 12 = 84 := by
  sorry

#eval final_jellybean_count 90 24 12

end NUMINAMATH_CALUDE_jellybean_count_is_84_l2283_228376


namespace NUMINAMATH_CALUDE_divisible_by_two_l2283_228340

theorem divisible_by_two (a : ℤ) (h : 2 ∣ a^2) : 2 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_two_l2283_228340


namespace NUMINAMATH_CALUDE_sum_of_bases_is_nineteen_l2283_228342

/-- Represents a repeating decimal in a given base -/
def RepeatingDecimal (numerator : ℕ) (denominator : ℕ) (base : ℕ) : Prop :=
  ∃ (k : ℕ), (base ^ k * numerator) % denominator = numerator

/-- The main theorem -/
theorem sum_of_bases_is_nineteen (R₁ R₂ : ℕ) :
  R₁ > 1 ∧ R₂ > 1 ∧
  RepeatingDecimal 5 11 R₁ ∧
  RepeatingDecimal 6 11 R₁ ∧
  RepeatingDecimal 5 13 R₂ ∧
  RepeatingDecimal 8 13 R₂ →
  R₁ + R₂ = 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_bases_is_nineteen_l2283_228342


namespace NUMINAMATH_CALUDE_student_correct_problems_l2283_228350

/-- Represents the number of problems solved correctly by a student. -/
def correct_problems (total : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (final_score : ℤ) : ℕ :=
  sorry

/-- Theorem stating that given the problem conditions, the number of correctly solved problems is 31. -/
theorem student_correct_problems :
  correct_problems 80 5 3 8 = 31 := by sorry

end NUMINAMATH_CALUDE_student_correct_problems_l2283_228350


namespace NUMINAMATH_CALUDE_christinas_speed_l2283_228328

/-- Prove Christina's speed given the problem conditions -/
theorem christinas_speed (initial_distance : ℝ) (jacks_speed : ℝ) (lindys_speed : ℝ) 
  (lindys_total_distance : ℝ) (h1 : initial_distance = 360) 
  (h2 : jacks_speed = 5) (h3 : lindys_speed = 12) (h4 : lindys_total_distance = 360) :
  ∃ (christinas_speed : ℝ), christinas_speed = 7 := by
  sorry


end NUMINAMATH_CALUDE_christinas_speed_l2283_228328


namespace NUMINAMATH_CALUDE_subtraction_two_minus_three_l2283_228393

theorem subtraction_two_minus_three : 2 - 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_two_minus_three_l2283_228393


namespace NUMINAMATH_CALUDE_intermediate_value_theorem_l2283_228386

theorem intermediate_value_theorem 
  {f : ℝ → ℝ} {a b : ℝ} (h₁ : a < b) (h₂ : Continuous f) (h₃ : f a * f b < 0) :
  ∃ c ∈ Set.Ioo a b, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_intermediate_value_theorem_l2283_228386


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l2283_228378

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  2 * X^2 - 21 * X + 55 = (X + 3) * q + 136 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l2283_228378


namespace NUMINAMATH_CALUDE_maximum_spent_l2283_228303

/-- Represents the denominations of money in fen -/
inductive Denomination
  | Yuan100
  | Yuan50
  | Yuan20
  | Yuan10
  | Yuan5
  | Yuan1
  | Jiao5
  | Jiao1
  | Fen5
  | Fen2
  | Fen1

/-- Converts a denomination to its value in fen -/
def denominationToFen (d : Denomination) : ℕ :=
  match d with
  | .Yuan100 => 10000
  | .Yuan50  => 5000
  | .Yuan20  => 2000
  | .Yuan10  => 1000
  | .Yuan5   => 500
  | .Yuan1   => 100
  | .Jiao5   => 50
  | .Jiao1   => 10
  | .Fen5    => 5
  | .Fen2    => 2
  | .Fen1    => 1

/-- Represents a set of banknotes or coins -/
structure Change where
  denominations : List Denomination
  distinct : denominations.Nodup

/-- The problem statement -/
theorem maximum_spent (initialAmount : ℕ) 
  (banknotes : Change) 
  (coins : Change) :
  (initialAmount = 10000) →
  (banknotes.denominations.length = 4) →
  (coins.denominations.length = 4) →
  (∀ d ∈ banknotes.denominations, denominationToFen d > 100) →
  (∀ d ∈ coins.denominations, denominationToFen d < 100) →
  ((banknotes.denominations.map denominationToFen).sum % 300 = 0) →
  ((coins.denominations.map denominationToFen).sum % 7 = 0) →
  (initialAmount - (banknotes.denominations.map denominationToFen).sum - 
   (coins.denominations.map denominationToFen).sum = 6337) :=
by sorry

end NUMINAMATH_CALUDE_maximum_spent_l2283_228303


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2283_228398

/-- Represents a truncated cone with given radii of horizontal bases -/
structure TruncatedCone where
  bottomRadius : ℝ
  topRadius : ℝ

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Predicate to check if a sphere is tangent to a truncated cone -/
def isTangent (cone : TruncatedCone) (sphere : Sphere) : Prop :=
  -- The actual implementation of this predicate is complex and would depend on geometric calculations
  sorry

/-- The main theorem stating the radius of the sphere tangent to the truncated cone -/
theorem sphere_radius_in_truncated_cone (cone : TruncatedCone) 
  (h1 : cone.bottomRadius = 24)
  (h2 : cone.topRadius = 8) :
  ∃ (sphere : Sphere), isTangent cone sphere ∧ sphere.radius = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l2283_228398


namespace NUMINAMATH_CALUDE_minutes_to_seconds_l2283_228310

theorem minutes_to_seconds (minutes : Real) (seconds_per_minute : Nat) :
  minutes * seconds_per_minute = 468 → minutes = 7.8 ∧ seconds_per_minute = 60 := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_seconds_l2283_228310


namespace NUMINAMATH_CALUDE_distance_between_cars_l2283_228375

theorem distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) :
  initial_distance = 105 →
  car1_distance = 50 →
  car2_distance = 35 →
  initial_distance - (car1_distance + car2_distance) = 20 := by
sorry

end NUMINAMATH_CALUDE_distance_between_cars_l2283_228375


namespace NUMINAMATH_CALUDE_sum_fourth_power_ge_two_min_sum_cube_and_reciprocal_cube_min_sum_cube_and_reciprocal_cube_equality_l2283_228397

-- Part I
theorem sum_fourth_power_ge_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^4 + b^4 ≥ 2 := by sorry

-- Part II
theorem min_sum_cube_and_reciprocal_cube (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 + (1/a + 1/b + 1/c)^3 ≥ 18 := by sorry

theorem min_sum_cube_and_reciprocal_cube_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 + (1/a + 1/b + 1/c)^3 = 18 ↔ a = (3 : ℝ)^(1/3) ∧ b = (3 : ℝ)^(1/3) ∧ c = (3 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_sum_fourth_power_ge_two_min_sum_cube_and_reciprocal_cube_min_sum_cube_and_reciprocal_cube_equality_l2283_228397


namespace NUMINAMATH_CALUDE_probability_is_one_twelfth_l2283_228395

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with at least two sides that are also sides of the decagon -/
def favorable_triangles : ℕ := decagon_vertices

/-- The probability of forming a triangle with at least two sides that are also sides of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

/-- Theorem stating the probability is 1/12 -/
theorem probability_is_one_twelfth : probability = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_twelfth_l2283_228395


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2283_228346

theorem square_plus_reciprocal_square (x : ℝ) (h : x + (1/x) = 4) : x^2 + (1/x^2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2283_228346


namespace NUMINAMATH_CALUDE_triangle_satisfies_equation_l2283_228390

/-- Converts a number from base 5 to base 10 -/
def base5To10 (d1 d2 : ℕ) : ℕ := 5 * d1 + d2

/-- Converts a number from base 12 to base 10 -/
def base12To10 (d1 d2 : ℕ) : ℕ := 12 * d1 + d2

/-- The digit satisfying the equation in base 5 and base 12 -/
def triangle : ℕ := 2

theorem triangle_satisfies_equation :
  base5To10 5 triangle = base12To10 triangle 3 ∧ triangle < 10 := by sorry

end NUMINAMATH_CALUDE_triangle_satisfies_equation_l2283_228390


namespace NUMINAMATH_CALUDE_smallest_among_given_numbers_l2283_228309

theorem smallest_among_given_numbers :
  let numbers : List ℚ := [-6/7, 2, 0, -1]
  ∀ x ∈ numbers, -1 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_among_given_numbers_l2283_228309


namespace NUMINAMATH_CALUDE_cook_is_innocent_l2283_228380

-- Define the type for individuals
def Individual : Type := String

-- Define the property of stealing pepper
def stole_pepper (x : Individual) : Prop := sorry

-- Define the property of lying
def always_lies (x : Individual) : Prop := sorry

-- Define the property of knowing who stole the pepper
def knows_thief (x : Individual) : Prop := sorry

-- The cook
def cook : Individual := "Cook"

-- Axiom: Individuals who steal pepper always lie
axiom pepper_thieves_lie : ∀ x : Individual, stole_pepper x → always_lies x

-- Axiom: The cook stated they know who stole the pepper
axiom cook_statement : knows_thief cook

-- Theorem: The cook is innocent (did not steal the pepper)
theorem cook_is_innocent : ¬(stole_pepper cook) := by sorry

end NUMINAMATH_CALUDE_cook_is_innocent_l2283_228380


namespace NUMINAMATH_CALUDE_smallest_y_in_triangle_l2283_228348

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_y_in_triangle (A B C x y : ℕ) : 
  A + B + C = 180 →
  isPrime A ∧ isPrime C →
  B ≤ A ∧ B ≤ C →
  2 * x + y = 180 →
  isPrime x →
  (∀ z : ℕ, z < y → ¬(isPrime z ∧ ∃ w : ℕ, isPrime w ∧ 2 * w + z = 180)) →
  y = 101 := by
sorry

end NUMINAMATH_CALUDE_smallest_y_in_triangle_l2283_228348


namespace NUMINAMATH_CALUDE_min_manager_ratio_l2283_228399

/-- The minimum ratio of managers to non-managers in a company -/
def min_ratio : ℚ := 2 / 9

/-- The number of managers in the example department -/
def managers : ℕ := 8

/-- The maximum number of non-managers in the example department -/
def max_non_managers : ℕ := 36

/-- Theorem stating that the minimum ratio of managers to non-managers is 2:9 -/
theorem min_manager_ratio :
  (managers : ℚ) / max_non_managers = min_ratio :=
by sorry

end NUMINAMATH_CALUDE_min_manager_ratio_l2283_228399


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l2283_228335

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x

-- Theorem stating that the axis of symmetry is x = 1
theorem axis_of_symmetry :
  ∀ y : ℝ, ∃ x : ℝ, parabola (x + 1) = parabola (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l2283_228335


namespace NUMINAMATH_CALUDE_polynomial_division_l2283_228326

theorem polynomial_division (x : ℝ) :
  x^5 - 17*x^3 + 8*x^2 - 9*x + 12 = (x - 3) * (x^4 + 3*x^3 - 8*x^2 - 16*x - 57) + (-159) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l2283_228326


namespace NUMINAMATH_CALUDE_inequality_solution_l2283_228373

theorem inequality_solution (x : ℝ) : 
  (2 * x + 2) / (3 * x + 1) < (x - 3) / (x + 4) ↔ 
  (x > -Real.sqrt 11 ∧ x < -1/3) ∨ (x > Real.sqrt 11) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2283_228373


namespace NUMINAMATH_CALUDE_triangle_side_range_l2283_228357

theorem triangle_side_range (a b c : ℝ) : 
  -- Triangle ABC is acute
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  -- Side lengths form an arithmetic sequence
  (b - a = c - b) ∧
  -- Sum of squares of sides equals 21
  a^2 + b^2 + c^2 = 21 →
  -- Range of b
  2 * Real.sqrt 42 / 5 < b ∧ b ≤ Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_range_l2283_228357


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2283_228324

theorem min_value_quadratic (x y : ℝ) :
  x^2 + y^2 - 8*x - 6*y + 20 ≥ -5 ∧
  ∃ (a b : ℝ), a^2 + b^2 - 8*a - 6*b + 20 = -5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2283_228324


namespace NUMINAMATH_CALUDE_log_expression_simplification_l2283_228367

theorem log_expression_simplification 
  (p q r s x y : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (hx : x > 0) (hy : y > 0) : 
  Real.log (p^2 / q) + Real.log (q^3 / r) + Real.log (r^2 / s) - Real.log (p^2 * y / (s^3 * x)) 
  = Real.log (q^2 * r * x * s^2 / y) := by
  sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l2283_228367


namespace NUMINAMATH_CALUDE_range_of_x_when_not_p_range_of_m_for_not_q_sufficient_not_necessary_l2283_228372

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - x - 2 ≤ 0
def q (x m : ℝ) : Prop := x^2 - x - m^2 - m ≤ 0

-- Theorem for the range of x when ¬p is true
theorem range_of_x_when_not_p (x : ℝ) :
  ¬(p x) ↔ (x > 2 ∨ x < -1) :=
sorry

-- Theorem for the range of m when ¬q is a sufficient but not necessary condition for ¬p
theorem range_of_m_for_not_q_sufficient_not_necessary (m : ℝ) :
  (∀ x, ¬(q x m) → ¬(p x)) ∧ ¬(∀ x, q x m → p x) ↔ (m > 1 ∨ m < -2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_not_p_range_of_m_for_not_q_sufficient_not_necessary_l2283_228372


namespace NUMINAMATH_CALUDE_compare_values_l2283_228308

theorem compare_values : 0.5^(1/10) > 0.4^(1/10) ∧ 0.4^(1/10) > Real.log 0.1 / Real.log 4 := by
  sorry

end NUMINAMATH_CALUDE_compare_values_l2283_228308


namespace NUMINAMATH_CALUDE_charity_ticket_revenue_l2283_228392

theorem charity_ticket_revenue :
  ∀ (f d : ℕ) (p : ℚ),
    f + d = 160 →
    f * p + d * (2/3 * p) = 2800 →
    ∃ (full_revenue : ℚ),
      full_revenue = f * p ∧
      full_revenue = 1680 :=
by sorry

end NUMINAMATH_CALUDE_charity_ticket_revenue_l2283_228392


namespace NUMINAMATH_CALUDE_expression_simplification_l2283_228391

theorem expression_simplification :
  0.7264 * 0.4329 + 0.1235 * 0.3412 + 0.1289 * 0.5634 - 0.3785 * 0.4979 = 0.2407 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2283_228391


namespace NUMINAMATH_CALUDE_original_number_proof_l2283_228374

theorem original_number_proof (x : ℝ) : 2 - (1 / x) = 5/2 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2283_228374


namespace NUMINAMATH_CALUDE_line_through_points_l2283_228347

/-- Given a line y = mx + c passing through points (3,2) and (7,14), prove that m - c = 10 -/
theorem line_through_points (m c : ℝ) : 
  (2 = m * 3 + c) → (14 = m * 7 + c) → m - c = 10 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l2283_228347


namespace NUMINAMATH_CALUDE_intersection_locus_is_hyperbola_l2283_228387

/-- The locus of points (x, y) satisfying the given system of equations is a hyperbola -/
theorem intersection_locus_is_hyperbola :
  ∀ (x y s : ℝ), 
    (s * x - 3 * y - 4 * s = 0) → 
    (x - 3 * s * y + 4 = 0) → 
    ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ x^2 / a^2 - y^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_is_hyperbola_l2283_228387


namespace NUMINAMATH_CALUDE_cinema_ticket_pricing_l2283_228382

theorem cinema_ticket_pricing (adult_price : ℚ) : 
  (10 * adult_price + 6 * (adult_price / 2) = 35) →
  ((12 * adult_price + 8 * (adult_price / 2)) * (9 / 10) = 504 / 13) := by
  sorry

end NUMINAMATH_CALUDE_cinema_ticket_pricing_l2283_228382


namespace NUMINAMATH_CALUDE_same_color_probability_l2283_228305

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 5

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 6

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 7

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 2

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

/-- The number of marbles drawn -/
def drawn_marbles : ℕ := 4

/-- The probability of drawing four marbles of the same color without replacement -/
theorem same_color_probability : 
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)) =
  55 / 4855 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2283_228305


namespace NUMINAMATH_CALUDE_prism_volume_l2283_228389

/-- The volume of a right rectangular prism with specific face areas and dimension ratio -/
theorem prism_volume (a b c : ℝ) : 
  a * b = 64 → 
  b * c = 81 → 
  a * c = 72 → 
  b = 2 * a → 
  |a * b * c - 1629| < 1 := by
sorry

end NUMINAMATH_CALUDE_prism_volume_l2283_228389


namespace NUMINAMATH_CALUDE_circle_equation_radius_l2283_228362

theorem circle_equation_radius (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 8*x + y^2 + 4*y - k = 0 ↔ (x + 4)^2 + (y + 2)^2 = 49) ↔ 
  k = 29 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l2283_228362


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l2283_228394

/-- The x-coordinate of the point on the x-axis equidistant from C(-3, 0) and D(2, 5) is 2 -/
theorem equidistant_point_x_coordinate :
  let C : ℝ × ℝ := (-3, 0)
  let D : ℝ × ℝ := (2, 5)
  ∃ x : ℝ, x = 2 ∧
    (x - C.1)^2 + C.2^2 = (x - D.1)^2 + D.2^2 :=
by sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l2283_228394


namespace NUMINAMATH_CALUDE_sum_of_squares_problem_l2283_228356

theorem sum_of_squares_problem (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_sum_squares : a^2 + b^2 + c^2 = 48) 
  (h_sum_products : a*b + b*c + c*a = 26) : 
  a + b + c = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_problem_l2283_228356


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_quadratic_inequality_l2283_228301

theorem negation_of_universal_quantifier (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - 3*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 3 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_negation_of_quadratic_inequality_l2283_228301


namespace NUMINAMATH_CALUDE_cubic_roots_divisibility_l2283_228315

theorem cubic_roots_divisibility (p a b c : ℤ) (hp : Prime p) 
  (ha : p ∣ a) (hb : p ∣ b) (hc : p ∣ c)
  (hroots : ∃ (r s : ℤ), r ≠ s ∧ r^3 + a*r^2 + b*r + c = 0 ∧ s^3 + a*s^2 + b*s + c = 0) :
  p^3 ∣ c := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_divisibility_l2283_228315


namespace NUMINAMATH_CALUDE_tallest_person_position_l2283_228306

/-- Represents a person with a height -/
structure Person where
  height : ℝ

/-- A line of people sorted by height -/
def SortedLine (n : ℕ) := Fin n → Person

theorem tallest_person_position
  (n : ℕ)
  (line : SortedLine n)
  (h_sorted : ∀ i j : Fin n, i < j → (line i).height ≤ (line j).height)
  (tallest : Fin n)
  (h_tallest : ∀ i : Fin n, (line i).height ≤ (line tallest).height) :
  tallest.val + 1 = n :=
sorry

end NUMINAMATH_CALUDE_tallest_person_position_l2283_228306


namespace NUMINAMATH_CALUDE_carrie_work_duration_l2283_228302

def hourly_wage : ℝ := 8
def weekly_hours : ℝ := 35
def bike_cost : ℝ := 400
def money_left : ℝ := 720

theorem carrie_work_duration :
  (money_left + bike_cost) / (hourly_wage * weekly_hours) = 4 := by
  sorry

end NUMINAMATH_CALUDE_carrie_work_duration_l2283_228302


namespace NUMINAMATH_CALUDE_beams_per_panel_is_two_l2283_228319

/-- Represents the number of fence panels in the fence -/
def num_panels : ℕ := 10

/-- Represents the number of metal sheets in each fence panel -/
def sheets_per_panel : ℕ := 3

/-- Represents the number of metal rods in each sheet -/
def rods_per_sheet : ℕ := 10

/-- Represents the number of metal rods in each beam -/
def rods_per_beam : ℕ := 4

/-- Represents the total number of metal rods needed for the fence -/
def total_rods : ℕ := 380

/-- Calculates the number of metal beams in each fence panel -/
def beams_per_panel : ℕ := 
  let total_sheets := num_panels * sheets_per_panel
  let rods_for_sheets := total_sheets * rods_per_sheet
  let remaining_rods := total_rods - rods_for_sheets
  let total_beams := remaining_rods / rods_per_beam
  total_beams / num_panels

/-- Theorem stating that the number of metal beams in each fence panel is 2 -/
theorem beams_per_panel_is_two : beams_per_panel = 2 := by sorry

end NUMINAMATH_CALUDE_beams_per_panel_is_two_l2283_228319


namespace NUMINAMATH_CALUDE_tens_digit_of_3_to_2017_l2283_228341

theorem tens_digit_of_3_to_2017 : ∃ n : ℕ, 3^2017 ≡ 87 + 100*n [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_to_2017_l2283_228341


namespace NUMINAMATH_CALUDE_b_arrives_first_l2283_228320

theorem b_arrives_first (x y S : ℝ) (hx : x > 0) (hy : y > 0) (hS : S > 0) (hxy : x < y) :
  (S * (x + y)) / (2 * x * y) > (2 * S) / (x + y) := by
  sorry

end NUMINAMATH_CALUDE_b_arrives_first_l2283_228320


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l2283_228360

/-- The volume of a slice of pizza -/
theorem pizza_slice_volume :
  let thickness : ℝ := 1/2
  let diameter : ℝ := 12
  let num_slices : ℕ := 8
  let radius : ℝ := diameter / 2
  let pizza_volume : ℝ := π * radius^2 * thickness
  let slice_volume : ℝ := pizza_volume / num_slices
  slice_volume = 9*π/4 := by sorry

end NUMINAMATH_CALUDE_pizza_slice_volume_l2283_228360


namespace NUMINAMATH_CALUDE_stating_valid_arrangements_count_l2283_228358

/-- 
Given n players with distinct heights, this function returns the number of ways to 
arrange them such that for each player, the total number of players either to their 
left and taller or to their right and shorter is even.
-/
def validArrangements (n : ℕ) : ℕ :=
  (n / 2).factorial * ((n + 1) / 2).factorial

/-- 
Theorem stating that the number of valid arrangements for n players
is equal to ⌊n/2⌋! * ⌈n/2⌉!
-/
theorem valid_arrangements_count (n : ℕ) :
  validArrangements n = (n / 2).factorial * ((n + 1) / 2).factorial := by
  sorry

end NUMINAMATH_CALUDE_stating_valid_arrangements_count_l2283_228358


namespace NUMINAMATH_CALUDE_alannah_extra_books_l2283_228388

/-- The number of books each person has -/
structure BookCount where
  alannah : ℕ
  beatrix : ℕ
  queen : ℕ

/-- The conditions of the book distribution problem -/
def BookProblem (bc : BookCount) : Prop :=
  bc.alannah > bc.beatrix ∧
  bc.queen = bc.alannah + bc.alannah / 5 ∧
  bc.beatrix = 30 ∧
  bc.alannah + bc.beatrix + bc.queen = 140

/-- The theorem stating that Alannah has 20 more books than Beatrix -/
theorem alannah_extra_books (bc : BookCount) (h : BookProblem bc) : 
  bc.alannah = bc.beatrix + 20 := by
  sorry


end NUMINAMATH_CALUDE_alannah_extra_books_l2283_228388


namespace NUMINAMATH_CALUDE_pizza_order_l2283_228363

theorem pizza_order (adults children adult_slices child_slices slices_per_pizza : ℕ) 
  (h1 : adults = 2)
  (h2 : children = 6)
  (h3 : adult_slices = 3)
  (h4 : child_slices = 1)
  (h5 : slices_per_pizza = 4) :
  (adults * adult_slices + children * child_slices) / slices_per_pizza = 3 := by
  sorry


end NUMINAMATH_CALUDE_pizza_order_l2283_228363


namespace NUMINAMATH_CALUDE_counterexample_exists_l2283_228333

theorem counterexample_exists : ∃ (a b c d : ℝ), 
  ((a + b) / (3*a - b) = (b + c) / (3*b - c)) ∧
  ((b + c) / (3*b - c) = (c + d) / (3*c - d)) ∧
  ((c + d) / (3*c - d) = (d + a) / (3*d - a)) ∧
  (3*a - b ≠ 0) ∧ (3*b - c ≠ 0) ∧ (3*c - d ≠ 0) ∧ (3*d - a ≠ 0) ∧
  (a^2 + b^2 + c^2 + d^2 ≠ a*b + b*c + c*d + d*a) :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2283_228333


namespace NUMINAMATH_CALUDE_max_annual_profit_l2283_228379

/-- Represents the annual profit function in million yuan -/
noncomputable def annual_profit (x : ℝ) : ℝ :=
  if x < 80 then
    50 * x - (1/3 * x^2 + 10 * x) / 100 - 250
  else
    50 * x - (51 * x + 10000 / x - 1450) / 100 - 250

/-- The maximum annual profit is 1000 million yuan -/
theorem max_annual_profit :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → annual_profit x ≥ annual_profit y ∧ annual_profit x = 1000 :=
sorry

end NUMINAMATH_CALUDE_max_annual_profit_l2283_228379


namespace NUMINAMATH_CALUDE_positive_integer_triplets_l2283_228359

theorem positive_integer_triplets :
  ∀ x y z : ℕ+,
    x ≤ y ∧ y ≤ z ∧ (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 1 ↔
    (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_positive_integer_triplets_l2283_228359


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2283_228321

theorem solve_linear_equation :
  ∃ x : ℚ, -3 * x - 12 = 6 * x + 9 → x = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2283_228321


namespace NUMINAMATH_CALUDE_brick_length_calculation_l2283_228349

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The problem statement -/
theorem brick_length_calculation (wall : Dimensions) (brick_width : ℝ) (brick_height : ℝ) 
    (num_bricks : ℕ) (h_wall : wall = ⟨800, 600, 22.5⟩) (h_brick_width : brick_width = 11.25) 
    (h_brick_height : brick_height = 6) (h_num_bricks : num_bricks = 2000) :
    ∃ (brick_length : ℝ), 
      volume wall = num_bricks * volume ⟨brick_length, brick_width, brick_height⟩ ∧ 
      brick_length = 80 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l2283_228349


namespace NUMINAMATH_CALUDE_survey_properties_l2283_228314

/-- Represents a student in the survey -/
structure Student where
  physicalCondition : String

/-- Represents the survey conducted by the school -/
structure Survey where
  students : List Student
  classes : Nat

/-- Defines the sample of the survey -/
def sample (s : Survey) : String :=
  s.students.map (λ student => student.physicalCondition) |> toString

/-- Defines the sample size of the survey -/
def sampleSize (s : Survey) : Nat :=
  s.students.length

/-- Theorem stating the properties of the survey -/
theorem survey_properties (s : Survey) 
  (h1 : s.students.length = 190)
  (h2 : s.classes = 19) :
  sample s = "physical condition of 190 students" ∧ 
  sampleSize s = 190 := by
  sorry

#check survey_properties

end NUMINAMATH_CALUDE_survey_properties_l2283_228314


namespace NUMINAMATH_CALUDE_riverdale_rangers_loss_percentage_l2283_228300

/-- Represents the statistics of a sports team --/
structure TeamStats where
  totalGames : ℕ
  winLossRatio : ℚ

/-- Calculates the percentage of games lost --/
def percentLost (stats : TeamStats) : ℚ :=
  let lostGames := stats.totalGames / (1 + stats.winLossRatio)
  (lostGames / stats.totalGames) * 100

/-- Theorem stating that for a team with given statistics, the percentage of games lost is 38% --/
theorem riverdale_rangers_loss_percentage :
  let stats : TeamStats := { totalGames := 65, winLossRatio := 8/5 }
  percentLost stats = 38 := by sorry


end NUMINAMATH_CALUDE_riverdale_rangers_loss_percentage_l2283_228300


namespace NUMINAMATH_CALUDE_triangle_angle_determinant_l2283_228313

theorem triangle_angle_determinant (θ φ ψ : Real) 
  (h : θ + φ + ψ = Real.pi) : 
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![Real.cos θ, Real.sin θ, 1],
    ![Real.cos φ, Real.sin φ, 1],
    ![Real.cos ψ, Real.sin ψ, 1]
  ]
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_determinant_l2283_228313


namespace NUMINAMATH_CALUDE_inverse_proportionality_l2283_228365

/-- Given that x is inversely proportional to y, prove that x = -2 when y = 10,
    given that x = 5 when y = -4. -/
theorem inverse_proportionality (x y : ℝ) (k : ℝ) :
  (∀ x y, x * y = k) →  -- x is inversely proportional to y
  (5 * (-4) = k) →      -- x = 5 when y = -4
  (10 * x = k) →        -- condition for y = 10
  x = -2 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l2283_228365


namespace NUMINAMATH_CALUDE_f_min_value_l2283_228396

/-- The function f(x) = 9x - 4x^2 -/
def f (x : ℝ) := 9 * x - 4 * x^2

/-- The minimum value of f(x) is -81/16 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ -81/16 := by sorry

end NUMINAMATH_CALUDE_f_min_value_l2283_228396


namespace NUMINAMATH_CALUDE_two_a_minus_b_equals_two_l2283_228368

theorem two_a_minus_b_equals_two (a b : ℝ) 
  (ha : a^3 - 12*a^2 + 47*a - 60 = 0)
  (hb : -b^3 + 12*b^2 - 47*b + 180 = 0) : 
  2*a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_two_a_minus_b_equals_two_l2283_228368


namespace NUMINAMATH_CALUDE_probability_all_white_or_all_black_l2283_228307

def white_balls : ℕ := 7
def black_balls : ℕ := 8
def total_balls : ℕ := white_balls + black_balls
def drawn_balls : ℕ := 5

theorem probability_all_white_or_all_black :
  (Nat.choose white_balls drawn_balls + Nat.choose black_balls drawn_balls) / Nat.choose total_balls drawn_balls = 77 / 3003 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_white_or_all_black_l2283_228307


namespace NUMINAMATH_CALUDE_spinner_final_direction_l2283_228352

-- Define the possible directions
inductive Direction
| North
| East
| South
| West

-- Define a function to calculate the final direction
def finalDirection (initialDir : Direction) (clockwiseRev : ℚ) (counterClockwiseRev : ℚ) : Direction :=
  sorry

-- Theorem statement
theorem spinner_final_direction :
  finalDirection Direction.North (7/2) (21/4) = Direction.East :=
sorry

end NUMINAMATH_CALUDE_spinner_final_direction_l2283_228352


namespace NUMINAMATH_CALUDE_problem_solution_l2283_228361

theorem problem_solution (x y : ℝ) (h1 : 0.2 * x = 200) (h2 : 0.3 * y = 150) :
  (0.8 * x - 0.5 * y) + 0.4 * (x + y) = 1150 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2283_228361


namespace NUMINAMATH_CALUDE_rachel_leah_age_difference_l2283_228327

/-- Given that Rachel is 19 years old and the sum of Rachel and Leah's ages is 34,
    prove that Rachel is 4 years older than Leah. -/
theorem rachel_leah_age_difference :
  ∀ (rachel_age leah_age : ℕ),
  rachel_age = 19 →
  rachel_age + leah_age = 34 →
  rachel_age - leah_age = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_rachel_leah_age_difference_l2283_228327


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l2283_228337

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Distance from a point to the right directrix -/
def distToRightDirectrix (p : Point) (e : Ellipse) : ℝ :=
  sorry

/-- Distance from a point to the left focus -/
def distToLeftFocus (p : Point) (e : Ellipse) : ℝ :=
  sorry

theorem ellipse_focus_distance 
  (p : Point) 
  (e : Ellipse) 
  (h1 : e.a = 10 ∧ e.b = 6) 
  (h2 : isOnEllipse p e) 
  (h3 : distToRightDirectrix p e = 17/2) : 
  distToLeftFocus p e = 66/5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l2283_228337


namespace NUMINAMATH_CALUDE_angle_CED_measure_l2283_228323

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the circles
def circle1 : Set (ℝ × ℝ) := sorry
def circle2 : Set (ℝ × ℝ) := sorry

-- State the conditions
axiom circles_congruent : circle1 = circle2
axiom A_center_circle1 : A ∈ circle1
axiom B_center_circle2 : B ∈ circle2
axiom B_on_circle1 : B ∈ circle1
axiom A_on_circle2 : A ∈ circle2
axiom C_on_line_AB : sorry
axiom D_on_line_AB : sorry
axiom E_intersection : E ∈ circle1 ∩ circle2

-- Define the angle CED
def angle_CED : ℝ := sorry

-- Theorem to prove
theorem angle_CED_measure : angle_CED = 120 := by sorry

end NUMINAMATH_CALUDE_angle_CED_measure_l2283_228323


namespace NUMINAMATH_CALUDE_iphone_price_calculation_l2283_228353

theorem iphone_price_calculation (P : ℝ) : 
  (P * (1 - 0.1) * (1 - 0.2) = 720) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_iphone_price_calculation_l2283_228353


namespace NUMINAMATH_CALUDE_max_value_theorem_l2283_228312

theorem max_value_theorem (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 2) :
  (x^2 * y) / (x + y) + (y^2 * z) / (y + z) + (z^2 * x) / (z + x) ≤ 1 ∧
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2 ∧
    (x^2 * y) / (x + y) + (y^2 * z) / (y + z) + (z^2 * x) / (z + x) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2283_228312


namespace NUMINAMATH_CALUDE_joan_balloons_l2283_228345

/-- The number of blue balloons Joan has now, given her initial count and the number lost. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

theorem joan_balloons : remaining_balloons 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l2283_228345


namespace NUMINAMATH_CALUDE_employee_remaining_hours_l2283_228325

/-- Calculates the remaining hours for an employee who uses half of their allotted sick and vacation days --/
def remaining_hours (sick_days : ℕ) (vacation_days : ℕ) (hours_per_day : ℕ) : ℕ :=
  let remaining_sick_days := sick_days / 2
  let remaining_vacation_days := vacation_days / 2
  (remaining_sick_days + remaining_vacation_days) * hours_per_day

/-- Proves that an employee with 10 sick days and 10 vacation days, using half of each, has 80 hours left --/
theorem employee_remaining_hours :
  remaining_hours 10 10 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_employee_remaining_hours_l2283_228325


namespace NUMINAMATH_CALUDE_carnival_prize_percentage_carnival_prize_percentage_proof_l2283_228338

theorem carnival_prize_percentage (total_minnows : ℕ) (minnows_per_prize : ℕ) 
  (total_players : ℕ) (leftover_minnows : ℕ) : ℕ → Prop :=
  λ percentage_winners =>
    total_minnows = 600 ∧
    minnows_per_prize = 3 ∧
    total_players = 800 ∧
    leftover_minnows = 240 →
    percentage_winners = 15 ∧
    (total_minnows - leftover_minnows) / minnows_per_prize * 100 / total_players = percentage_winners

-- Proof
theorem carnival_prize_percentage_proof : 
  ∃ (percentage_winners : ℕ), carnival_prize_percentage 600 3 800 240 percentage_winners :=
by
  sorry

end NUMINAMATH_CALUDE_carnival_prize_percentage_carnival_prize_percentage_proof_l2283_228338


namespace NUMINAMATH_CALUDE_function_transformation_l2283_228336

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 3) : f (-(-1)) + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l2283_228336


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2283_228339

theorem gcd_factorial_eight_and_factorial_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 2880 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l2283_228339


namespace NUMINAMATH_CALUDE_hens_count_l2283_228331

/-- Given a total number of heads and feet, and the number of feet for hens and cows,
    calculate the number of hens. -/
def count_hens (total_heads : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) : ℕ :=
  sorry

theorem hens_count :
  let total_heads := 44
  let total_feet := 140
  let hen_feet := 2
  let cow_feet := 4
  count_hens total_heads total_feet hen_feet cow_feet = 18 := by
  sorry

end NUMINAMATH_CALUDE_hens_count_l2283_228331


namespace NUMINAMATH_CALUDE_parabola_focus_lines_range_l2283_228354

/-- A parabola with equation y^2 = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- A line passing through the focus of the parabola -/
structure FocusLine (para : Parabola) where
  k : ℝ  -- slope of the line

/-- Intersection points of a focus line with the parabola -/
def intersection_points (para : Parabola) (line : FocusLine para) : ℝ × ℝ := sorry

/-- Distance between intersection points -/
def distance (para : Parabola) (line : FocusLine para) : ℝ := sorry

/-- Number of focus lines with a specific intersection distance -/
def num_lines_with_distance (para : Parabola) (d : ℝ) : ℕ := sorry

theorem parabola_focus_lines_range (para : Parabola) :
  (num_lines_with_distance para 4 = 2) → (0 < para.p ∧ para.p < 2) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_lines_range_l2283_228354


namespace NUMINAMATH_CALUDE_parallelogram_max_area_l2283_228317

/-- Given a parallelogram with perimeter 60 units and one side three times the length of the other,
    the maximum possible area is 168.75 square units. -/
theorem parallelogram_max_area :
  ∀ (a b : ℝ),
  a > 0 → b > 0 →
  a = 3 * b →
  2 * a + 2 * b = 60 →
  ∀ (θ : ℝ),
  0 < θ → θ < π →
  a * b * Real.sin θ ≤ 168.75 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_max_area_l2283_228317


namespace NUMINAMATH_CALUDE_abs_less_implies_sum_positive_l2283_228311

theorem abs_less_implies_sum_positive (a b : ℝ) : |a| < b → a + b > 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_less_implies_sum_positive_l2283_228311


namespace NUMINAMATH_CALUDE_base2_10101010_equals_base4_2212_l2283_228322

def base2_to_decimal (b : List Bool) : ℕ :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def decimal_to_base4 (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List (Fin 4) :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

theorem base2_10101010_equals_base4_2212 :
  decimal_to_base4 (base2_to_decimal [true, false, true, false, true, false, true, false]) =
  [2, 2, 1, 2] := by sorry

end NUMINAMATH_CALUDE_base2_10101010_equals_base4_2212_l2283_228322


namespace NUMINAMATH_CALUDE_yellow_peaches_count_l2283_228355

/-- The number of yellow peaches in a basket, given the number of green peaches
    and the difference between green and yellow peaches. -/
def yellow_peaches (green : ℕ) (difference : ℕ) : ℕ :=
  green - difference

/-- Theorem stating that the number of yellow peaches is 6, given the conditions. -/
theorem yellow_peaches_count : yellow_peaches 14 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_yellow_peaches_count_l2283_228355


namespace NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l2283_228344

theorem cos_pi_plus_2alpha (α : Real) (h : Real.sin (π / 2 + α) = 1 / 3) : 
  Real.cos (π + 2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l2283_228344


namespace NUMINAMATH_CALUDE_segment_area_equilateral_triangle_l2283_228383

/-- The area of a circular segment cut off by one side of an equilateral triangle inscribed in a circle -/
theorem segment_area_equilateral_triangle (a : ℝ) (ha : a > 0) :
  let R := a / Real.sqrt 3
  let sector_area := π * R^2 / 3
  let triangle_area := a * R / 4
  sector_area - triangle_area = (a^2 * (4 * π - 3 * Real.sqrt 3)) / 36 := by
  sorry

end NUMINAMATH_CALUDE_segment_area_equilateral_triangle_l2283_228383


namespace NUMINAMATH_CALUDE_sqrt_seven_irrational_negative_one_third_rational_two_rational_decimal_rational_irrational_among_options_l2283_228316

theorem sqrt_seven_irrational :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 7 = p / q) :=
by
  sorry

theorem negative_one_third_rational :
  ∃ (p q : ℤ), q ≠ 0 ∧ (-1 : ℚ) / 3 = p / q :=
by
  sorry

theorem two_rational :
  ∃ (p q : ℤ), q ≠ 0 ∧ (2 : ℚ) = p / q :=
by
  sorry

theorem decimal_rational :
  ∃ (p q : ℤ), q ≠ 0 ∧ (0.0101 : ℚ) = p / q :=
by
  sorry

theorem irrational_among_options :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 7 = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (-1 : ℚ) / 3 = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (2 : ℚ) = p / q) ∧
  (∃ (p q : ℤ), q ≠ 0 ∧ (0.0101 : ℚ) = p / q) :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_irrational_negative_one_third_rational_two_rational_decimal_rational_irrational_among_options_l2283_228316


namespace NUMINAMATH_CALUDE_mcdonald_farm_production_l2283_228304

/-- Calculates the total number of eggs needed in a month for Mcdonald's farm --/
def monthly_egg_production (saly_weekly : ℕ) (ben_weekly : ℕ) (weeks_in_month : ℕ) : ℕ :=
  let ked_weekly := ben_weekly / 2
  let total_weekly := saly_weekly + ben_weekly + ked_weekly
  total_weekly * weeks_in_month

/-- Proves that Mcdonald's farm should produce 124 eggs in a month --/
theorem mcdonald_farm_production : monthly_egg_production 10 14 4 = 124 := by
  sorry

#eval monthly_egg_production 10 14 4

end NUMINAMATH_CALUDE_mcdonald_farm_production_l2283_228304


namespace NUMINAMATH_CALUDE_chess_game_players_l2283_228377

def number_of_players : ℕ := 15
def total_games : ℕ := 105

theorem chess_game_players :
  ∃ k : ℕ,
    k > 0 ∧
    k < number_of_players ∧
    (number_of_players.choose k) = total_games ∧
    k = 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_players_l2283_228377


namespace NUMINAMATH_CALUDE_chocolate_problem_l2283_228384

theorem chocolate_problem (total : ℕ) (eaten_with_nuts : ℚ) (left : ℕ) : 
  total = 80 →
  eaten_with_nuts = 4/5 →
  left = 28 →
  (total / 2 - (total / 2 * eaten_with_nuts) - (total - left)) / (total / 2) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_problem_l2283_228384


namespace NUMINAMATH_CALUDE_set_equality_l2283_228371

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def M : Set ℕ := {3, 4, 5}
def N : Set ℕ := {1, 3, 6}

theorem set_equality : (Uᶜ ∪ M) ∩ (Uᶜ ∪ N) = {2, 7} := by sorry

end NUMINAMATH_CALUDE_set_equality_l2283_228371


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2283_228329

theorem diophantine_equation_solutions :
  let S : Set (ℤ × ℤ) := {(3995, 3993), (1, -1), (1999, 3996005), (3996005, 1997), (1997, -3996005), (-3996005, 1995)}
  ∀ x y : ℤ, (1996 * x + 1998 * y + 1 = x * y) ↔ (x, y) ∈ S :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2283_228329


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_over_product_l2283_228385

theorem cubic_root_sum_cubes_over_product (p q a b c : ℝ) : 
  q ≠ 0 → 
  (∀ x : ℝ, x^3 + p*x + q = (x-a)*(x-b)*(x-c)) → 
  (a^3 + b^3 + c^3) / (a*b*c) = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_over_product_l2283_228385
