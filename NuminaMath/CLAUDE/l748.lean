import Mathlib

namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l748_74836

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 9 / y) ≥ 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l748_74836


namespace NUMINAMATH_CALUDE_continuity_at_one_l748_74871

def f (x : ℝ) := -3 * x^2 - 6

theorem continuity_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_one_l748_74871


namespace NUMINAMATH_CALUDE_min_tangent_length_l748_74820

/-- The minimum length of a tangent from a point on y = x + 1 to (x-3)^2 + y^2 = 1 is √7 -/
theorem min_tangent_length :
  let line := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 1}
  ∃ (min_length : ℝ),
    min_length = Real.sqrt 7 ∧
    ∀ (p : ℝ × ℝ) (t : ℝ × ℝ),
      p ∈ line → t ∈ circle →
      dist p t ≥ min_length :=
by sorry


end NUMINAMATH_CALUDE_min_tangent_length_l748_74820


namespace NUMINAMATH_CALUDE_rectangle_to_square_l748_74893

/-- A rectangle can be divided into two parts that form a square -/
theorem rectangle_to_square (length width : ℝ) (h1 : length = 9) (h2 : width = 4) :
  ∃ (side : ℝ), side^2 = length * width ∧ side = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l748_74893


namespace NUMINAMATH_CALUDE_trig_identity_l748_74895

theorem trig_identity (α : ℝ) : 
  Real.sin α ^ 2 + Real.cos (π / 6 - α) ^ 2 - Real.sin α * Real.cos (π / 6 - α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l748_74895


namespace NUMINAMATH_CALUDE_relay_team_members_l748_74801

/-- Represents a cross-country relay team -/
structure RelayTeam where
  totalDistance : ℝ
  standardMemberDistance : ℝ
  ralphDistance : ℝ
  otherMembersCount : ℕ

/-- Conditions for the relay team -/
def validRelayTeam (team : RelayTeam) : Prop :=
  team.totalDistance = 18 ∧
  team.standardMemberDistance = 3 ∧
  team.ralphDistance = 2 * team.standardMemberDistance ∧
  team.totalDistance = team.ralphDistance + team.otherMembersCount * team.standardMemberDistance

/-- Theorem: The number of other team members is 4 -/
theorem relay_team_members (team : RelayTeam) (h : validRelayTeam team) : 
  team.otherMembersCount = 4 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_members_l748_74801


namespace NUMINAMATH_CALUDE_triangle_inequality_l748_74833

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^3 + b^3 + 3*a*b*c > c^3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l748_74833


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l748_74894

/-- Represents the duration of each color in the traffic light cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the total cycle duration -/
def cycleDuration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green + cycle.yellow + cycle.red

/-- Calculates the number of seconds where a color change can be observed -/
def changeObservationWindow (cycle : TrafficLightCycle) : ℕ :=
  3 * 5  -- 5 seconds before each color change, and there are 3 changes

/-- Theorem: The probability of observing a color change in a 5-second interval
    for the given traffic light cycle is 3/20 -/
theorem traffic_light_change_probability 
  (cycle : TrafficLightCycle) 
  (h1 : cycle.green = 50) 
  (h2 : cycle.yellow = 5) 
  (h3 : cycle.red = 45) :
  (changeObservationWindow cycle : ℚ) / (cycleDuration cycle) = 3 / 20 := by
  sorry


end NUMINAMATH_CALUDE_traffic_light_change_probability_l748_74894


namespace NUMINAMATH_CALUDE_min_value_abc_min_value_equals_one_over_nine_to_nine_l748_74886

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) : 
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 1/x + 1/y + 1/z = 9 → 
  a^4 * b^3 * c^2 ≤ x^4 * y^3 * z^2 :=
by sorry

theorem min_value_equals_one_over_nine_to_nine (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1/a + 1/b + 1/c = 9) : 
  a^4 * b^3 * c^2 = 1 / 9^9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_min_value_equals_one_over_nine_to_nine_l748_74886


namespace NUMINAMATH_CALUDE_simplify_P_P_value_on_inverse_proportion_l748_74891

/-- Simplification of the expression P -/
theorem simplify_P (a b : ℝ) :
  (2*a + 3*b)^2 - (2*a + b)*(2*a - b) - 2*b*(3*a + 5*b) = 6*a*b := by sorry

/-- Value of P when (a,b) lies on y = -2/x -/
theorem P_value_on_inverse_proportion (a b : ℝ) (h : a*b = -2) :
  6*a*b = -12 := by sorry

end NUMINAMATH_CALUDE_simplify_P_P_value_on_inverse_proportion_l748_74891


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l748_74865

theorem arithmetic_sequence_middle_term (a₁ a₃ y : ℤ) :
  a₁ = 3^2 →
  a₃ = 3^4 →
  y = (a₁ + a₃) / 2 →
  y = 45 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l748_74865


namespace NUMINAMATH_CALUDE_max_intersections_circle_two_lines_triangle_l748_74805

/-- Represents a circle in a plane -/
structure Circle where
  -- Definition of a circle (not needed for this proof)

/-- Represents a line in a plane -/
structure Line where
  -- Definition of a line (not needed for this proof)

/-- Represents a triangle in a plane -/
structure Triangle where
  -- Definition of a triangle (not needed for this proof)

/-- The maximum number of intersection points between a circle and a line -/
def maxCircleLineIntersections : ℕ := 2

/-- The maximum number of intersection points between two distinct lines -/
def maxTwoLinesIntersections : ℕ := 1

/-- The maximum number of intersection points between a circle and a triangle -/
def maxCircleTriangleIntersections : ℕ := 6

/-- The maximum number of intersection points between two lines and a triangle -/
def maxTwoLinesTriangleIntersections : ℕ := 6

/-- Theorem: The maximum number of intersection points between a circle, two distinct lines, and a triangle is 17 -/
theorem max_intersections_circle_two_lines_triangle :
  ∀ (c : Circle) (l1 l2 : Line) (t : Triangle),
    l1 ≠ l2 →
    (maxCircleLineIntersections * 2 + maxTwoLinesIntersections +
     maxCircleTriangleIntersections + maxTwoLinesTriangleIntersections) = 17 :=
by
  sorry


end NUMINAMATH_CALUDE_max_intersections_circle_two_lines_triangle_l748_74805


namespace NUMINAMATH_CALUDE_constant_term_expansion_l748_74877

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = (1/x - x^(1/2))^6) ∧ 
  (∃ c : ℝ, ∀ x ≠ 0, f x = c + x * (f x - c) ∧ c = 15) :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l748_74877


namespace NUMINAMATH_CALUDE_non_square_difference_characterization_l748_74830

/-- A natural number that cannot be represented as the difference of squares of any two natural numbers. -/
def NonSquareDifference (n : ℕ) : Prop :=
  ∀ x y : ℕ, n ≠ x^2 - y^2

/-- Characterization of numbers that cannot be represented as the difference of squares. -/
theorem non_square_difference_characterization (n : ℕ) :
  NonSquareDifference n ↔ n = 1 ∨ n = 4 ∨ ∃ k : ℕ, n = 4*k + 2 :=
sorry

end NUMINAMATH_CALUDE_non_square_difference_characterization_l748_74830


namespace NUMINAMATH_CALUDE_art_fair_customers_one_painting_l748_74847

/-- The number of customers who bought one painting each at Tracy's art fair booth -/
def customers_one_painting (total_customers : ℕ) (two_painting_customers : ℕ) (four_painting_customers : ℕ) (total_paintings_sold : ℕ) : ℕ :=
  total_paintings_sold - (2 * two_painting_customers + 4 * four_painting_customers)

/-- Theorem stating that the number of customers who bought one painting each is 12 -/
theorem art_fair_customers_one_painting :
  customers_one_painting 20 4 4 36 = 12 := by
  sorry

#eval customers_one_painting 20 4 4 36

end NUMINAMATH_CALUDE_art_fair_customers_one_painting_l748_74847


namespace NUMINAMATH_CALUDE_octagon_diagonals_l748_74885

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 vertices -/
def octagon_vertices : ℕ := 8

theorem octagon_diagonals : num_diagonals octagon_vertices = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l748_74885


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l748_74888

theorem complex_fraction_equality : 
  1 / ( 3 + 1 / ( 3 + 1 / ( 3 - 1 / 3 ) ) ) = 27/89 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l748_74888


namespace NUMINAMATH_CALUDE_expression_value_l748_74879

theorem expression_value (x y : ℝ) (h : x^2 - 4*x + 4 + |y - 1| = 0) :
  (2*x - y)^2 - 2*(2*x - y)*(x + 2*y) + (x + 2*y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l748_74879


namespace NUMINAMATH_CALUDE_cheese_division_theorem_l748_74829

/-- Represents the weight of cheese pieces -/
structure CheesePair :=
  (larger : ℕ)
  (smaller : ℕ)

/-- Simulates taking a bite from the larger piece -/
def takeBite (pair : CheesePair) : CheesePair :=
  ⟨pair.larger - pair.smaller, pair.smaller⟩

/-- Theorem: If after three bites, the cheese pieces are equal and weigh 20 grams each,
    then the original cheese weight was 680 grams -/
theorem cheese_division_theorem (initial : CheesePair) :
  (takeBite (takeBite (takeBite initial))) = ⟨20, 20⟩ →
  initial.larger + initial.smaller = 680 :=
by
  sorry

#check cheese_division_theorem

end NUMINAMATH_CALUDE_cheese_division_theorem_l748_74829


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l748_74859

theorem equal_roots_quadratic (h : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - 4 * x + h / 3 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - 4 * y + h / 3 = 0 → y = x) ↔ h = 4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l748_74859


namespace NUMINAMATH_CALUDE_S_intersect_T_eq_T_l748_74866

def S : Set ℝ := { y | ∃ x, y = 3^x }
def T : Set ℝ := { y | ∃ x, y = x^2 + 1 }

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end NUMINAMATH_CALUDE_S_intersect_T_eq_T_l748_74866


namespace NUMINAMATH_CALUDE_cube_vertices_l748_74803

/-- A cube is a polyhedron with 6 faces and 12 edges -/
structure Cube where
  faces : ℕ
  edges : ℕ
  faces_eq : faces = 6
  edges_eq : edges = 12

/-- The number of vertices in a cube -/
def num_vertices (c : Cube) : ℕ := sorry

theorem cube_vertices (c : Cube) : num_vertices c = 8 := by sorry

end NUMINAMATH_CALUDE_cube_vertices_l748_74803


namespace NUMINAMATH_CALUDE_smallest_reunion_time_l748_74860

def horse_lap_times : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def is_valid_time (t : ℕ) : Prop :=
  ∃ (subset : List ℕ), subset.length ≥ 4 ∧ 
    subset.all (λ x => x ∈ horse_lap_times) ∧
    subset.all (λ x => t % x = 0)

theorem smallest_reunion_time :
  ∃ (T : ℕ), T > 0 ∧ is_valid_time T ∧
    ∀ (t : ℕ), 0 < t ∧ t < T → ¬is_valid_time t :=
  sorry

end NUMINAMATH_CALUDE_smallest_reunion_time_l748_74860


namespace NUMINAMATH_CALUDE_residue_negative_811_mod_24_l748_74811

theorem residue_negative_811_mod_24 : Int.mod (-811) 24 = 5 := by
  sorry

end NUMINAMATH_CALUDE_residue_negative_811_mod_24_l748_74811


namespace NUMINAMATH_CALUDE_starting_lineup_with_twins_l748_74878

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem starting_lineup_with_twins (total_players : ℕ) (lineup_size : ℕ) (twin_count : ℕ) :
  total_players = 12 →
  lineup_size = 5 →
  twin_count = 2 →
  choose (total_players - twin_count) (lineup_size - twin_count) = 120 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_with_twins_l748_74878


namespace NUMINAMATH_CALUDE_sum_ratio_equals_3_l748_74818

def sum_multiples_of_3 (n : ℕ) : ℕ := 
  3 * (n * (n + 1) / 2)

def sum_integers (m : ℕ) : ℕ := 
  m * (m + 1) / 2

theorem sum_ratio_equals_3 : 
  (sum_multiples_of_3 200) / (sum_integers 200) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_ratio_equals_3_l748_74818


namespace NUMINAMATH_CALUDE_polynomial_with_arithmetic_progression_roots_l748_74819

theorem polynomial_with_arithmetic_progression_roots (j : ℝ) : 
  (∃ a b c d : ℝ, a < b ∧ b < c ∧ c < d ∧ 
    (∀ x : ℝ, x^4 + j*x^2 + 16*x + 64 = (x - a) * (x - b) * (x - c) * (x - d)) ∧
    b - a = c - b ∧ d - c = c - b) →
  j = -160/9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_with_arithmetic_progression_roots_l748_74819


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l748_74804

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 2*x + y = 4) 
  (eq2 : x + 2*y = 5) : 
  5*x^2 + 8*x*y + 5*y^2 = 41 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l748_74804


namespace NUMINAMATH_CALUDE_inequality_condition_sum_l748_74817

theorem inequality_condition_sum (a₁ a₂ : ℝ) : 
  (∀ x : ℝ, (x^2 - a₁*x + 2) / (x^2 - x + 1) < 3) ∧
  (∀ x : ℝ, (x^2 - a₂*x + 2) / (x^2 - x + 1) < 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, (x^2 - a*x + 2) / (x^2 - x + 1) < 3) → a > a₁ ∧ a < a₂) →
  a₁ = 3 - 2*Real.sqrt 2 ∧ a₂ = 3 + 2*Real.sqrt 2 ∧ a₁ + a₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_sum_l748_74817


namespace NUMINAMATH_CALUDE_two_tails_probability_l748_74814

theorem two_tails_probability (n : ℕ) (h : n = 5) : 
  (Nat.choose n 2 : ℚ) / (2^n : ℚ) = 10/32 := by
  sorry

end NUMINAMATH_CALUDE_two_tails_probability_l748_74814


namespace NUMINAMATH_CALUDE_pikes_caught_l748_74835

theorem pikes_caught (total_fishes sturgeons herrings : ℕ) 
  (h1 : total_fishes = 145)
  (h2 : sturgeons = 40)
  (h3 : herrings = 75) :
  total_fishes - (sturgeons + herrings) = 30 := by
  sorry

end NUMINAMATH_CALUDE_pikes_caught_l748_74835


namespace NUMINAMATH_CALUDE_inverse_proposition_correct_l748_74875

/-- The statement of a geometric proposition -/
structure GeometricProposition :=
  (hypothesis : String)
  (conclusion : String)

/-- The inverse of a geometric proposition -/
def inverse_proposition (p : GeometricProposition) : GeometricProposition :=
  { hypothesis := p.conclusion,
    conclusion := p.hypothesis }

/-- The original proposition -/
def original_prop : GeometricProposition :=
  { hypothesis := "Triangles are congruent",
    conclusion := "Corresponding angles are equal" }

/-- Theorem stating that the inverse proposition is correct -/
theorem inverse_proposition_correct : 
  inverse_proposition original_prop = 
  { hypothesis := "Corresponding angles are equal",
    conclusion := "Triangles are congruent" } := by
  sorry


end NUMINAMATH_CALUDE_inverse_proposition_correct_l748_74875


namespace NUMINAMATH_CALUDE_roots_of_unity_sum_one_l748_74823

theorem roots_of_unity_sum_one (n : ℕ) (h : Even n) (h_pos : n > 0) :
  ∃ (z₁ z₂ z₃ : ℂ), (z₁^n = 1) ∧ (z₂^n = 1) ∧ (z₃^n = 1) ∧ (z₁ + z₂ + z₃ = 1) :=
sorry

end NUMINAMATH_CALUDE_roots_of_unity_sum_one_l748_74823


namespace NUMINAMATH_CALUDE_blood_cell_count_l748_74826

theorem blood_cell_count (total : ℕ) (second : ℕ) (first : ℕ) : 
  total = 7341 → second = 3120 → first = total - second → first = 4221 := by
  sorry

end NUMINAMATH_CALUDE_blood_cell_count_l748_74826


namespace NUMINAMATH_CALUDE_juvys_garden_rows_l748_74806

/-- Represents Juvy's garden -/
structure Garden where
  rows : ℕ
  plants_per_row : ℕ
  parsley_rows : ℕ
  rosemary_rows : ℕ
  chive_plants : ℕ

/-- Theorem: The number of rows in Juvy's garden is 20 -/
theorem juvys_garden_rows (g : Garden) 
  (h1 : g.plants_per_row = 10)
  (h2 : g.parsley_rows = 3)
  (h3 : g.rosemary_rows = 2)
  (h4 : g.chive_plants = 150)
  (h5 : g.chive_plants = g.plants_per_row * (g.rows - g.parsley_rows - g.rosemary_rows)) :
  g.rows = 20 := by
  sorry

end NUMINAMATH_CALUDE_juvys_garden_rows_l748_74806


namespace NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l748_74852

theorem equidistant_point_on_y_axis : 
  ∃ y : ℝ, 
    ((-3 : ℝ) - 0)^2 + (0 - y)^2 = ((-2 : ℝ) - 0)^2 + (5 - y)^2 ∧ 
    y = 2 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_on_y_axis_l748_74852


namespace NUMINAMATH_CALUDE_calculation_problem_l748_74837

theorem calculation_problem (x : ℝ) : 10 * 1.8 - (2 * x / 0.3) = 50 ↔ x = -4.8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_problem_l748_74837


namespace NUMINAMATH_CALUDE_quadratic_comparison_l748_74849

/-- Proves that for a quadratic function y = a(x-1)^2 + 3 where a < 0,
    if (-1, y₁) and (2, y₂) are points on the graph, then y₁ < y₂ -/
theorem quadratic_comparison (a : ℝ) (y₁ y₂ : ℝ)
    (h₁ : a < 0)
    (h₂ : y₁ = a * (-1 - 1)^2 + 3)
    (h₃ : y₂ = a * (2 - 1)^2 + 3) :
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_comparison_l748_74849


namespace NUMINAMATH_CALUDE_nicks_sister_age_difference_l748_74858

theorem nicks_sister_age_difference (nick_age : ℕ) (sister_age_diff : ℕ) : 
  nick_age = 13 →
  (nick_age + sister_age_diff) / 2 + 5 = 21 →
  sister_age_diff = 19 := by
  sorry

end NUMINAMATH_CALUDE_nicks_sister_age_difference_l748_74858


namespace NUMINAMATH_CALUDE_baker_pastries_sold_l748_74807

/-- Given information about a baker's production and sales of cakes and pastries, 
    prove that the number of pastries sold equals the number of cakes made. -/
theorem baker_pastries_sold (cakes_made pastries_made : ℕ) 
    (h1 : cakes_made = 19)
    (h2 : pastries_made = 131)
    (h3 : pastries_made - cakes_made = 112) :
    pastries_made - (pastries_made - cakes_made) = cakes_made := by
  sorry

end NUMINAMATH_CALUDE_baker_pastries_sold_l748_74807


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l748_74870

def initial_people : ℕ := 8
def initial_average_age : ℚ := 35
def leaving_person_age : ℕ := 22
def remaining_people : ℕ := initial_people - 1

theorem average_age_after_leaving :
  (initial_people * initial_average_age - leaving_person_age) / remaining_people = 258 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l748_74870


namespace NUMINAMATH_CALUDE_price_decrease_sales_increase_ratio_l748_74874

theorem price_decrease_sales_increase_ratio (P U : ℝ) (h_positive : P > 0 ∧ U > 0) :
  let new_price := 0.8 * P
  let new_units := U / 0.8
  let revenue_unchanged := P * U = new_price * new_units
  let percent_decrease_price := 20
  let percent_increase_units := (new_units - U) / U * 100
  revenue_unchanged →
  percent_increase_units / percent_decrease_price = 1.25 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_sales_increase_ratio_l748_74874


namespace NUMINAMATH_CALUDE_pipe_filling_speed_l748_74896

/-- Proves that if Pipe A fills a tank in 24 minutes, and both Pipe A and Pipe B together fill the tank in 3 minutes, then Pipe B fills the tank 7 times faster than Pipe A. -/
theorem pipe_filling_speed (fill_time_A : ℝ) (fill_time_both : ℝ) (speed_ratio : ℝ) : 
  fill_time_A = 24 → 
  fill_time_both = 3 → 
  (1 / fill_time_A + speed_ratio / fill_time_A) * fill_time_both = 1 →
  speed_ratio = 7 := by
sorry

end NUMINAMATH_CALUDE_pipe_filling_speed_l748_74896


namespace NUMINAMATH_CALUDE_election_votes_l748_74832

/-- In an election with 3 candidates, where two candidates received 5000 and 15000 votes
    respectively, and the winning candidate got 66.66666666666666% of the total votes,
    the winning candidate (third candidate) received 40000 votes. -/
theorem election_votes :
  let total_votes : ℕ := 60000
  let first_candidate_votes : ℕ := 5000
  let second_candidate_votes : ℕ := 15000
  let winning_percentage : ℚ := 200 / 3
  ∀ third_candidate_votes : ℕ,
    first_candidate_votes + second_candidate_votes + third_candidate_votes = total_votes →
    (third_candidate_votes : ℚ) / total_votes * 100 = winning_percentage →
    third_candidate_votes = 40000 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l748_74832


namespace NUMINAMATH_CALUDE_no_valid_propositions_l748_74843

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the given conditions
variable (m n : Line) (α β : Plane)
variable (h1 : perpendicular m α)
variable (h2 : contains β n)

-- State the theorem
theorem no_valid_propositions :
  ¬(∀ (m n : Line) (α β : Plane), 
    perpendicular m α → contains β n → 
    ((parallel_planes α β → perpendicular_lines m n) ∧
     (perpendicular_lines m n → parallel_planes α β) ∧
     (parallel_lines m n → perpendicular_planes α β) ∧
     (perpendicular_planes α β → parallel_lines m n))) :=
sorry

end NUMINAMATH_CALUDE_no_valid_propositions_l748_74843


namespace NUMINAMATH_CALUDE_right_triangle_set_l748_74857

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that only one of the given sets forms a right triangle -/
theorem right_triangle_set :
  ¬(is_right_triangle 2 4 3) ∧
  ¬(is_right_triangle 6 8 9) ∧
  ¬(is_right_triangle 3 4 6) ∧
  is_right_triangle 1 1 (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_set_l748_74857


namespace NUMINAMATH_CALUDE_original_proposition_is_true_negation_is_false_l748_74800

theorem original_proposition_is_true : ∀ (a b : ℝ), a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1) := by
  sorry

theorem negation_is_false : ¬(∀ (a b : ℝ), a + b ≥ 2 → (a < 1 ∧ b < 1)) := by
  sorry

end NUMINAMATH_CALUDE_original_proposition_is_true_negation_is_false_l748_74800


namespace NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l748_74881

theorem common_root_of_quadratic_equations (p q : ℝ) (x : ℝ) :
  (2017 * x^2 + p * x + q = 0) ∧ 
  (p * x^2 + q * x + 2017 = 0) →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l748_74881


namespace NUMINAMATH_CALUDE_gcd_71_19_l748_74887

theorem gcd_71_19 : Nat.gcd 71 19 = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_71_19_l748_74887


namespace NUMINAMATH_CALUDE_intersection_conditions_l748_74855

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_conditions (a : ℝ) :
  (9 ∈ A a ∩ B a ↔ a = 5 ∨ a = -3) ∧
  ({9} = A a ∩ B a ↔ a = -3) :=
sorry

end NUMINAMATH_CALUDE_intersection_conditions_l748_74855


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l748_74863

theorem reciprocal_of_negative_2023 :
  ((-2023)⁻¹ : ℚ) = -1/2023 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l748_74863


namespace NUMINAMATH_CALUDE_inequality_range_l748_74844

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0) ↔ (-1 < a ∧ a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l748_74844


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l748_74802

theorem smallest_cube_root_with_small_remainder (m n : ℕ) (r : ℝ) : 
  (∀ k < m, ¬∃ (j : ℕ) (s : ℝ), k^(1/3 : ℝ) = j + s ∧ 0 < s ∧ s < 1/2000) →
  (m : ℝ)^(1/3 : ℝ) = n + r →
  0 < r →
  r < 1/2000 →
  n = 26 := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l748_74802


namespace NUMINAMATH_CALUDE_cylindrical_cans_radius_l748_74854

/-- Proves that for two cylindrical cans with equal volumes, where one can is four times taller than the other,
    if the taller can has a radius of 5 units, then the shorter can has a radius of 10 units. -/
theorem cylindrical_cans_radius (volume : ℝ) (h : ℝ) (r : ℝ) :
  volume = 500 ∧
  volume = π * 5^2 * (4 * h) ∧
  volume = π * r^2 * h →
  r = 10 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_cans_radius_l748_74854


namespace NUMINAMATH_CALUDE_overlap_area_of_intersecting_rectangles_l748_74851

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of overlap between two rectangles intersecting at 45 degrees -/
noncomputable def overlapArea (r1 r2 : Rectangle) : ℝ :=
  min r1.width r2.width * min r1.height r2.height

/-- The theorem stating the area of the overlapping region -/
theorem overlap_area_of_intersecting_rectangles :
  let r1 : Rectangle := { width := 3, height := 12 }
  let r2 : Rectangle := { width := 4, height := 10 }
  rectangleArea r1 + rectangleArea r2 - overlapArea r1 r2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_overlap_area_of_intersecting_rectangles_l748_74851


namespace NUMINAMATH_CALUDE_max_xy_value_l748_74812

theorem max_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) : x * y ≤ 348 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l748_74812


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l748_74846

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) :
  0 < a 1 → a 1 < a 2 → a 2 > Real.sqrt (a 1 * a 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l748_74846


namespace NUMINAMATH_CALUDE_gcd_1978_2017_l748_74828

theorem gcd_1978_2017 : Nat.gcd 1978 2017 = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_1978_2017_l748_74828


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l748_74873

theorem simplify_sqrt_expression :
  Real.sqrt (12 + 8 * Real.sqrt 3) + Real.sqrt (12 - 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l748_74873


namespace NUMINAMATH_CALUDE_water_width_after_drop_l748_74892

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -2*y

-- Define the point that the parabola passes through
def parabola_point : ℝ × ℝ := (2, -2)

-- Theorem to prove
theorem water_width_after_drop :
  parabola parabola_point.1 parabola_point.2 →
  ∀ x : ℝ, parabola x (-3) → 2 * |x| = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_water_width_after_drop_l748_74892


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l748_74868

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 27⌉ + ⌈Real.sqrt 243⌉ = 24 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l748_74868


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l748_74897

-- Define the type for a point in 2D space
def Point := ℝ × ℝ

-- Define the type for a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a point is on a line
def isPointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

-- Define a function to check if two lines are perpendicular
def areLinesPerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Theorem statement
theorem perpendicular_line_through_point 
  (l : Line) 
  (h1 : isPointOnLine (-1, 2) l) 
  (h2 : areLinesPerpendicular l ⟨1, -3, 5⟩) : 
  l = ⟨3, 1, 1⟩ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l748_74897


namespace NUMINAMATH_CALUDE_polynomial_factorization_l748_74848

theorem polynomial_factorization (x : ℝ) : 
  75 * x^7 - 175 * x^13 = 25 * x^7 * (3 - 7 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l748_74848


namespace NUMINAMATH_CALUDE_volume_of_circumscribed_polyhedron_l748_74822

-- Define a polyhedron circumscribed around a sphere
structure CircumscribedPolyhedron where
  -- R is the radius of the inscribed sphere
  R : ℝ
  -- S is the surface area of the polyhedron
  S : ℝ
  -- V is the volume of the polyhedron
  V : ℝ
  -- Ensure R and S are positive
  R_pos : 0 < R
  S_pos : 0 < S

-- Theorem statement
theorem volume_of_circumscribed_polyhedron (p : CircumscribedPolyhedron) :
  p.V = (p.S * p.R) / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_circumscribed_polyhedron_l748_74822


namespace NUMINAMATH_CALUDE_distribute_8_balls_3_boxes_l748_74862

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes --/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes,
    with each box containing at least one ball --/
def distribute_balls_nonempty (n : ℕ) (k : ℕ) : ℕ := distribute_balls (n - k) k

theorem distribute_8_balls_3_boxes : distribute_balls_nonempty 8 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_distribute_8_balls_3_boxes_l748_74862


namespace NUMINAMATH_CALUDE_sin_neg_135_degrees_l748_74853

theorem sin_neg_135_degrees : Real.sin (-(135 * π / 180)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_135_degrees_l748_74853


namespace NUMINAMATH_CALUDE_round_trip_time_l748_74850

/-- The total time for a round trip between two points, given the distance and speeds in each direction -/
theorem round_trip_time (distance : ℝ) (speed_to : ℝ) (speed_from : ℝ) :
  distance = 19.999999999999996 →
  speed_to = 25 →
  speed_from = 4 →
  (distance / speed_to) + (distance / speed_from) = 5.8 := by
  sorry

#check round_trip_time

end NUMINAMATH_CALUDE_round_trip_time_l748_74850


namespace NUMINAMATH_CALUDE_joe_trip_expenses_l748_74809

/-- Calculates the remaining money after expenses -/
def remaining_money (initial_savings flight_cost hotel_cost food_cost : ℕ) : ℕ :=
  initial_savings - (flight_cost + hotel_cost + food_cost)

/-- Proves that Joe has $1,000 left after his trip expenses -/
theorem joe_trip_expenses :
  remaining_money 6000 1200 800 3000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_joe_trip_expenses_l748_74809


namespace NUMINAMATH_CALUDE_sum_of_r_p_x_is_negative_eleven_l748_74834

def p (x : ℝ) : ℝ := |x| - 2

def r (x : ℝ) : ℝ := -|p x - 1|

def x_values : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_r_p_x_is_negative_eleven :
  (x_values.map (λ x => r (p x))).sum = -11 := by sorry

end NUMINAMATH_CALUDE_sum_of_r_p_x_is_negative_eleven_l748_74834


namespace NUMINAMATH_CALUDE_polynomial_factorization_l748_74856

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 9*x + 20) + (x^2 + 6*x - 9) = 
  (x^2 + 6*x + 6) * (x^2 + 6*x + 3) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l748_74856


namespace NUMINAMATH_CALUDE_jellybean_problem_l748_74890

theorem jellybean_problem (initial : ℕ) (first_removal : ℕ) (second_removal : ℕ) (final : ℕ) 
  (h1 : initial = 37)
  (h2 : first_removal = 15)
  (h3 : second_removal = 4)
  (h4 : final = 23) :
  ∃ (added_back : ℕ), initial - first_removal + added_back - second_removal = final ∧ added_back = 5 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l748_74890


namespace NUMINAMATH_CALUDE_modified_pyramid_volume_l748_74898

/-- Given a pyramid with a square base and volume of 60 cubic inches, 
    if the base side length is tripled and the height is decreased by 25%, 
    the new volume will be 405 cubic inches. -/
theorem modified_pyramid_volume 
  (s : ℝ) (h : ℝ) 
  (original_volume : (1/3 : ℝ) * s^2 * h = 60) 
  (s_positive : s > 0) 
  (h_positive : h > 0) : 
  (1/3 : ℝ) * (3*s)^2 * (0.75*h) = 405 :=
by sorry

end NUMINAMATH_CALUDE_modified_pyramid_volume_l748_74898


namespace NUMINAMATH_CALUDE_unique_positive_solution_l748_74815

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) := by sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l748_74815


namespace NUMINAMATH_CALUDE_oddDigitSequence_157th_l748_74899

/-- A function that generates the nth number in the sequence of positive integers formed only by odd digits -/
def oddDigitSequence (n : ℕ) : ℕ :=
  sorry

/-- The set of odd digits -/
def oddDigits : Set ℕ := {1, 3, 5, 7, 9}

/-- A predicate to check if a number consists only of odd digits -/
def hasOnlyOddDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ∈ oddDigits

/-- The main theorem stating that the 157th number in the sequence is 1113 -/
theorem oddDigitSequence_157th :
  oddDigitSequence 157 = 1113 ∧ hasOnlyOddDigits (oddDigitSequence 157) :=
sorry

end NUMINAMATH_CALUDE_oddDigitSequence_157th_l748_74899


namespace NUMINAMATH_CALUDE_barbaras_allowance_l748_74889

theorem barbaras_allowance 
  (watch_cost : ℕ) 
  (current_savings : ℕ) 
  (weeks_left : ℕ) 
  (h1 : watch_cost = 100)
  (h2 : current_savings = 20)
  (h3 : weeks_left = 16) :
  (watch_cost - current_savings) / weeks_left = 5 :=
by sorry

end NUMINAMATH_CALUDE_barbaras_allowance_l748_74889


namespace NUMINAMATH_CALUDE_aspirations_necessary_for_reaching_l748_74813

-- Define the universe of discourse
variable (Person : Type)

-- Define predicates
variable (has_aspirations : Person → Prop)
variable (can_reach_extraordinary : Person → Prop)
variable (is_remote_dangerous : Person → Prop)
variable (few_venture : Person → Prop)

-- State the theorem
theorem aspirations_necessary_for_reaching :
  (∀ p : Person, is_remote_dangerous p → few_venture p) →
  (∀ p : Person, can_reach_extraordinary p → has_aspirations p) →
  (∀ p : Person, can_reach_extraordinary p → has_aspirations p) :=
by
  sorry


end NUMINAMATH_CALUDE_aspirations_necessary_for_reaching_l748_74813


namespace NUMINAMATH_CALUDE_division_problem_l748_74841

theorem division_problem : ∃ (n : ℕ), n ≠ 0 ∧ 45 = 11 * n + 1 ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l748_74841


namespace NUMINAMATH_CALUDE_kim_cookie_boxes_l748_74816

theorem kim_cookie_boxes (jennifer_boxes : ℕ) (difference : ℕ) (h1 : jennifer_boxes = 71) (h2 : difference = 17) :
  jennifer_boxes - difference = 54 :=
by sorry

end NUMINAMATH_CALUDE_kim_cookie_boxes_l748_74816


namespace NUMINAMATH_CALUDE_g_zero_l748_74880

/-- The function g(x) = 5x - 6 -/
def g (x : ℝ) : ℝ := 5 * x - 6

/-- Theorem: g(6/5) = 0 -/
theorem g_zero : g (6 / 5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_g_zero_l748_74880


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_component_l748_74821

/-- Given two 2D vectors m and n, if they are perpendicular and have specific components,
    then the x-component of m must be 2. -/
theorem perpendicular_vectors_x_component
  (m n : ℝ × ℝ)  -- m and n are 2D real vectors
  (h1 : m.1 = x ∧ m.2 = 2)  -- m = (x, 2)
  (h2 : n = (1, -1))  -- n = (1, -1)
  (h3 : m • n = 0)  -- m is perpendicular to n (dot product is zero)
  : x = 2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_component_l748_74821


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l748_74831

/-- For an arithmetic sequence {a_n} with a_2 = 3 and a_5 = 12, the common difference d is 3. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Definition of arithmetic sequence
  (h_a2 : a 2 = 3)  -- Given: a_2 = 3
  (h_a5 : a 5 = 12)  -- Given: a_5 = 12
  : ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l748_74831


namespace NUMINAMATH_CALUDE_probability_both_boys_given_one_boy_l748_74883

/-- Represents the gender of a child -/
inductive Gender
  | Boy
  | Girl

/-- Represents a family with two children -/
structure Family :=
  (child1 : Gender)
  (child2 : Gender)

/-- The set of all possible families with two children -/
def allFamilies : Finset Family :=
  sorry

/-- The set of families with at least one boy -/
def familiesWithBoy : Finset Family :=
  sorry

/-- The set of families with two boys -/
def familiesWithTwoBoys : Finset Family :=
  sorry

theorem probability_both_boys_given_one_boy :
    (familiesWithTwoBoys.card : ℚ) / familiesWithBoy.card = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_boys_given_one_boy_l748_74883


namespace NUMINAMATH_CALUDE_no_heptagon_intersection_l748_74842

/-- A cube in 3D space -/
structure Cube where
  -- Add necessary fields for a cube

/-- A plane in 3D space -/
structure Plane where
  -- Add necessary fields for a plane

/-- Represents the intersection of a plane and a cube -/
def Intersection (c : Cube) (p : Plane) : Set Point := sorry

/-- The number of edges of the cube that the plane intersects -/
def numIntersectedEdges (c : Cube) (p : Plane) : ℕ := sorry

/-- Predicate to check if a plane passes through a vertex more than once -/
def passesVertexTwice (c : Cube) (p : Plane) : Prop := sorry

/-- Theorem: A plane intersecting a cube cannot form a heptagon -/
theorem no_heptagon_intersection (c : Cube) (p : Plane) : 
  ¬(numIntersectedEdges c p = 7 ∧ ¬passesVertexTwice c p) := by
  sorry

end NUMINAMATH_CALUDE_no_heptagon_intersection_l748_74842


namespace NUMINAMATH_CALUDE_not_isosceles_if_distinct_sides_l748_74845

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Define what it means for a triangle to be isosceles
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Theorem statement
theorem not_isosceles_if_distinct_sides (t : Triangle) 
  (distinct_sides : t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c) : 
  ¬(is_isosceles t) := by
  sorry

end NUMINAMATH_CALUDE_not_isosceles_if_distinct_sides_l748_74845


namespace NUMINAMATH_CALUDE_inequality_proof_l748_74810

-- Define the set M
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- Theorem statement
theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2 * |a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l748_74810


namespace NUMINAMATH_CALUDE_trigonometric_identity_l748_74869

theorem trigonometric_identity : 
  - Real.sin (133 * π / 180) * Real.cos (197 * π / 180) - 
  Real.cos (47 * π / 180) * Real.cos (73 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l748_74869


namespace NUMINAMATH_CALUDE_factor_polynomial_l748_74872

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = -27 * x^5 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l748_74872


namespace NUMINAMATH_CALUDE_marco_strawberry_weight_l748_74876

theorem marco_strawberry_weight (total_weight : ℕ) (weight_difference : ℕ) :
  total_weight = 47 →
  weight_difference = 13 →
  ∃ (marco_weight dad_weight : ℕ),
    marco_weight + dad_weight = total_weight ∧
    marco_weight = dad_weight + weight_difference ∧
    marco_weight = 30 :=
by sorry

end NUMINAMATH_CALUDE_marco_strawberry_weight_l748_74876


namespace NUMINAMATH_CALUDE_females_advanced_count_l748_74839

/-- A company's employee distribution by gender and education level -/
structure Company where
  total_employees : ℕ
  females : ℕ
  males : ℕ
  advanced_degrees : ℕ
  college_degrees : ℕ
  vocational_training : ℕ
  males_college : ℕ
  females_vocational : ℕ

/-- The number of females with advanced degrees in the company -/
def females_advanced (c : Company) : ℕ :=
  c.advanced_degrees - (c.males - c.males_college - (c.vocational_training - c.females_vocational))

/-- Theorem stating the number of females with advanced degrees -/
theorem females_advanced_count (c : Company) 
  (h1 : c.total_employees = 360)
  (h2 : c.females = 220)
  (h3 : c.males = 140)
  (h4 : c.advanced_degrees = 140)
  (h5 : c.college_degrees = 160)
  (h6 : c.vocational_training = 60)
  (h7 : c.males_college = 55)
  (h8 : c.females_vocational = 25)
  (h9 : c.total_employees = c.females + c.males)
  (h10 : c.total_employees = c.advanced_degrees + c.college_degrees + c.vocational_training) :
  females_advanced c = 90 := by
  sorry

#eval females_advanced {
  total_employees := 360,
  females := 220,
  males := 140,
  advanced_degrees := 140,
  college_degrees := 160,
  vocational_training := 60,
  males_college := 55,
  females_vocational := 25
}

end NUMINAMATH_CALUDE_females_advanced_count_l748_74839


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l748_74827

theorem simplify_sqrt_expression :
  (Real.sqrt 450 / Real.sqrt 200) - (Real.sqrt 175 / Real.sqrt 75) = (9 - 2 * Real.sqrt 21) / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l748_74827


namespace NUMINAMATH_CALUDE_inequality_range_of_p_l748_74867

-- Define the inequality function
def inequality (x p : ℝ) : Prop := x^2 + p*x + 1 > 2*x + p

-- Define the theorem
theorem inequality_range_of_p :
  ∀ p : ℝ, (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → inequality x p) → p > -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_of_p_l748_74867


namespace NUMINAMATH_CALUDE_escalator_length_l748_74884

/-- The length of an escalator given specific conditions -/
theorem escalator_length : 
  ∀ (escalator_speed person_speed time : ℝ) (length : ℝ),
  escalator_speed = 12 →
  person_speed = 2 →
  time = 14 →
  length = (escalator_speed + person_speed) * time →
  length = 196 := by
sorry

end NUMINAMATH_CALUDE_escalator_length_l748_74884


namespace NUMINAMATH_CALUDE_largest_gold_coin_distribution_l748_74861

theorem largest_gold_coin_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 13 * k + 3) → 
  n < 110 → 
  (∀ m : ℕ, (∃ j : ℕ, m = 13 * j + 3) → m < 110 → m ≤ n) →
  n = 107 := by
  sorry

end NUMINAMATH_CALUDE_largest_gold_coin_distribution_l748_74861


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l748_74882

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the stack configuration -/
structure StackConfiguration where
  num_3ft : ℕ
  num_4ft : ℕ
  num_5ft : ℕ

def crate_dimensions : CrateDimensions :=
  { length := 3, width := 4, height := 5 }

def num_crates : ℕ := 12

def target_height : ℕ := 50

def valid_configuration (config : StackConfiguration) : Prop :=
  config.num_3ft + config.num_4ft + config.num_5ft = num_crates ∧
  3 * config.num_3ft + 4 * config.num_4ft + 5 * config.num_5ft = target_height

def num_valid_configurations : ℕ := 33616

def total_possible_configurations : ℕ := 3^num_crates

theorem crate_stacking_probability :
  (num_valid_configurations : ℚ) / (total_possible_configurations : ℚ) = 80 / 1593 := by
  sorry

#check crate_stacking_probability

end NUMINAMATH_CALUDE_crate_stacking_probability_l748_74882


namespace NUMINAMATH_CALUDE_malcolm_green_lights_l748_74824

/-- The number of green lights Malcolm bought -/
def green_lights (red blue green total_needed : ℕ) : Prop :=
  green = total_needed - (red + blue)

/-- Theorem stating the number of green lights Malcolm bought -/
theorem malcolm_green_lights :
  ∃ (green : ℕ), 
    let red := 12
    let blue := 3 * red
    let total_needed := 59 - 5
    green_lights red blue green total_needed ∧ green = 6 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_green_lights_l748_74824


namespace NUMINAMATH_CALUDE_point_P_satisfies_conditions_l748_74825

def P₁ : ℝ × ℝ := (1, 3)
def P₂ : ℝ × ℝ := (4, -6)
def P : ℝ × ℝ := (3, -3)

def vector (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, vector A C = (t • (vector A B).1, t • (vector A B).2)

theorem point_P_satisfies_conditions :
  collinear P₁ P₂ P ∧ vector P₁ P = (2 • (vector P P₂).1, 2 • (vector P P₂).2) := by
  sorry

end NUMINAMATH_CALUDE_point_P_satisfies_conditions_l748_74825


namespace NUMINAMATH_CALUDE_f_recursive_relation_l748_74838

/-- The smallest integer such that any permutation on n elements, repeated f(n) times, gives the identity. -/
def f (n : ℕ) : ℕ := sorry

/-- Checks if a number is a prime power -/
def isPrimePower (n : ℕ) : Prop := sorry

/-- The prime base of a prime power -/
def primeBase (n : ℕ) : ℕ := sorry

theorem f_recursive_relation (n : ℕ) :
  (isPrimePower n → f n = primeBase n * f (n - 1)) ∧
  (¬isPrimePower n → f n = f (n - 1)) := by sorry

end NUMINAMATH_CALUDE_f_recursive_relation_l748_74838


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l748_74808

theorem fractional_equation_solution :
  ∃ x : ℝ, (1 / x = 2 / (x + 1)) ∧ (x = 1) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l748_74808


namespace NUMINAMATH_CALUDE_function_value_at_negative_five_l748_74840

/-- Given a function f(x) = ax + b * sin(x) + 1 where f(5) = 7, prove that f(-5) = -5 -/
theorem function_value_at_negative_five 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x + b * Real.sin x + 1)
  (h2 : f 5 = 7) : 
  f (-5) = -5 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_five_l748_74840


namespace NUMINAMATH_CALUDE_expression_simplification_l748_74864

theorem expression_simplification :
  (((3 + 4 + 6 + 7) / 3) + ((4 * 3 + 5 - 2) / 4)) = 125 / 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l748_74864
