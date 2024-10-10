import Mathlib

namespace coin_array_problem_l2698_269828

/-- The number of coins in a triangular array with n rows -/
def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem coin_array_problem :
  ∃ N : ℕ, triangle_sum N = 2080 ∧ sum_of_digits N = 10 := by
  sorry

end coin_array_problem_l2698_269828


namespace solution_set_a_eq_one_range_of_a_l2698_269843

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x + a| + |x|

-- Theorem 1: Solution set when a = 1
theorem solution_set_a_eq_one :
  {x : ℝ | f x 1 < 3} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

-- Theorem 2: Range of a when f(x) < 3 has a solution
theorem range_of_a (a : ℝ) :
  (∃ x, f x a < 3) ↔ -3 < a ∧ a < 3 := by sorry

end solution_set_a_eq_one_range_of_a_l2698_269843


namespace inequality_solution_set_l2698_269817

theorem inequality_solution_set (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  let S := {x : ℝ | (x - a) * (x + a - 1) < 0}
  (0 ≤ a ∧ a < (1/2) → S = Set.Ioo a (1 - a)) ∧
  (a = (1/2) → S = ∅) ∧
  ((1/2) < a ∧ a ≤ 1 → S = Set.Ioo (1 - a) a) :=
by sorry

end inequality_solution_set_l2698_269817


namespace cardinality_of_B_l2698_269877

def A : Finset ℚ := {1, 2, 3, 4, 6}

def B : Finset ℚ := Finset.image (λ (p : ℚ × ℚ) => p.1 / p.2) (A.product A)

theorem cardinality_of_B : Finset.card B = 13 := by sorry

end cardinality_of_B_l2698_269877


namespace max_distance_ellipse_line_l2698_269864

/-- The maximum distance between a point on the ellipse x²/12 + y²/4 = 1 and the line x + √3y - 6 = 0 is √6 + 3, occurring at the point (-√6, -√2) -/
theorem max_distance_ellipse_line :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 12 + p.2^2 / 4 = 1}
  let line := {p : ℝ × ℝ | p.1 + Real.sqrt 3 * p.2 - 6 = 0}
  let distance (p : ℝ × ℝ) := |p.1 + Real.sqrt 3 * p.2 - 6| / 2
  ∃ (p : ℝ × ℝ), p ∈ ellipse ∧
    (∀ q ∈ ellipse, distance q ≤ distance p) ∧
    distance p = Real.sqrt 6 + 3 ∧
    p = (-Real.sqrt 6, -Real.sqrt 2) := by
  sorry

end max_distance_ellipse_line_l2698_269864


namespace james_older_brother_age_l2698_269863

/-- Given information about John and James' ages, prove James' older brother's age -/
theorem james_older_brother_age :
  ∀ (john_age james_age : ℕ),
  john_age = 39 →
  john_age - 3 = 2 * (james_age + 6) →
  ∃ (james_brother_age : ℕ),
  james_brother_age = james_age + 4 ∧
  james_brother_age = 16 := by
sorry

end james_older_brother_age_l2698_269863


namespace inequality_proof_l2698_269891

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c = d) : a + c > b + d := by
  sorry

end inequality_proof_l2698_269891


namespace linear_function_value_l2698_269806

/-- A linear function f(x) = px + q -/
def f (p q : ℝ) (x : ℝ) : ℝ := p * x + q

theorem linear_function_value (p q : ℝ) :
  f p q 3 = 5 → f p q 5 = 9 → f p q 1 = 1 := by
  sorry

end linear_function_value_l2698_269806


namespace equation_solution_l2698_269832

theorem equation_solution : 
  ∃ (x : ℚ), x = -3/4 ∧ x/(x+1) = 3/x + 1 := by
  sorry

end equation_solution_l2698_269832


namespace nested_sqrt_unique_solution_l2698_269840

/-- Recursive function representing the nested square root expression -/
def nestedSqrt (x : ℤ) : ℕ → ℤ
  | 0 => x
  | n + 1 => (nestedSqrt x n).sqrt

/-- Theorem stating that the only integer solution to the nested square root equation is (0, 0) -/
theorem nested_sqrt_unique_solution :
  ∀ x y : ℤ, (nestedSqrt x 1998 : ℤ) = y → x = 0 ∧ y = 0 := by
  sorry

#check nested_sqrt_unique_solution

end nested_sqrt_unique_solution_l2698_269840


namespace complete_residue_system_product_l2698_269893

theorem complete_residue_system_product (m n : ℕ) (a : Fin m → ℤ) (b : Fin n → ℤ) :
  (∀ k : Fin (m * n), ∃ i : Fin m, ∃ j : Fin n, (a i * b j) % (m * n) = k) →
  ((∀ k : Fin m, ∃ i : Fin m, a i % m = k) ∧
   (∀ k : Fin n, ∃ j : Fin n, b j % n = k)) :=
by sorry

end complete_residue_system_product_l2698_269893


namespace video_game_lives_l2698_269889

/-- Calculates the total lives after completing all levels in a video game -/
def total_lives (initial : ℝ) (hard_part : ℝ) (next_level : ℝ) (extra_challenge1 : ℝ) (extra_challenge2 : ℝ) : ℝ :=
  initial + hard_part + next_level + extra_challenge1 + extra_challenge2

/-- Theorem stating that the total lives after completing all levels is 261.0 -/
theorem video_game_lives :
  let initial : ℝ := 143.0
  let hard_part : ℝ := 14.0
  let next_level : ℝ := 27.0
  let extra_challenge1 : ℝ := 35.0
  let extra_challenge2 : ℝ := 42.0
  total_lives initial hard_part next_level extra_challenge1 extra_challenge2 = 261.0 := by
  sorry


end video_game_lives_l2698_269889


namespace daniels_purchase_l2698_269885

/-- Given the costs of a magazine and a pencil, and a coupon discount,
    calculate the total amount spent. -/
def total_spent (magazine_cost pencil_cost coupon_discount : ℚ) : ℚ :=
  magazine_cost + pencil_cost - coupon_discount

/-- Theorem stating that given the specific costs and discount,
    the total amount spent is $1.00. -/
theorem daniels_purchase :
  total_spent 0.85 0.50 0.35 = 1.00 := by
  sorry

end daniels_purchase_l2698_269885


namespace planes_parallel_l2698_269897

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (lineParallel : Line → Line → Prop)

-- State the theorem
theorem planes_parallel (α β γ : Plane) (a b : Line) :
  (parallel α γ ∧ parallel β γ) ∧
  (perpendicular a α ∧ perpendicular b β ∧ lineParallel a b) →
  parallel α β :=
sorry

end planes_parallel_l2698_269897


namespace zero_function_satisfies_equation_zero_function_is_solution_l2698_269878

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + 2*y) * f (x - 2*y) = (f x + f y)^2 - 16 * y^2 * f x

/-- The zero function -/
def ZeroFunction : ℝ → ℝ := λ x => 0

/-- Theorem: The zero function satisfies the functional equation -/
theorem zero_function_satisfies_equation : SatisfiesFunctionalEquation ZeroFunction := by
  sorry

/-- Theorem: The zero function is a solution to the functional equation -/
theorem zero_function_is_solution :
  ∃ f : ℝ → ℝ, SatisfiesFunctionalEquation f ∧ (∀ x : ℝ, f x = 0) := by
  sorry

end zero_function_satisfies_equation_zero_function_is_solution_l2698_269878


namespace first_group_size_l2698_269866

/-- Given a work that takes some men 80 days to complete and 20 men 32 days to complete,
    prove that the number of men in the first group is 8. -/
theorem first_group_size (work : ℕ) : ∃ (x : ℕ), x * 80 = 20 * 32 ∧ x = 8 := by sorry

end first_group_size_l2698_269866


namespace overlapping_segments_length_l2698_269819

/-- Given a set of overlapping segments with known total length and span, 
    this theorem proves the length of each overlapping part. -/
theorem overlapping_segments_length 
  (total_length : ℝ) 
  (edge_to_edge : ℝ) 
  (num_overlaps : ℕ) 
  (h1 : total_length = 98) 
  (h2 : edge_to_edge = 83) 
  (h3 : num_overlaps = 6) :
  (total_length - edge_to_edge) / num_overlaps = 2.5 := by
  sorry

#check overlapping_segments_length

end overlapping_segments_length_l2698_269819


namespace cans_difference_l2698_269838

/-- The number of cans Sarah collected yesterday -/
def sarah_yesterday : ℕ := 50

/-- The number of additional cans Lara collected compared to Sarah yesterday -/
def lara_extra_yesterday : ℕ := 30

/-- The number of cans Sarah collected today -/
def sarah_today : ℕ := 40

/-- The number of cans Lara collected today -/
def lara_today : ℕ := 70

/-- Theorem stating the difference in total cans collected between yesterday and today -/
theorem cans_difference : 
  (sarah_yesterday + (sarah_yesterday + lara_extra_yesterday)) - (sarah_today + lara_today) = 20 := by
  sorry

end cans_difference_l2698_269838


namespace symmetric_point_x_axis_l2698_269884

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricPointXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

theorem symmetric_point_x_axis :
  let P : Point3D := { x := 1, y := 3, z := 6 }
  symmetricPointXAxis P = { x := 1, y := -3, z := -6 } := by
  sorry

end symmetric_point_x_axis_l2698_269884


namespace ellipse_m_value_l2698_269823

/-- The value of m for an ellipse with given properties -/
theorem ellipse_m_value (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2 / 25 + y^2 / m^2 = 1) →
  (∃ c : ℝ, c = 4 ∧ c^2 = 25 - m^2) →
  m = 3 := by
  sorry

end ellipse_m_value_l2698_269823


namespace rationalize_denominator_l2698_269857

theorem rationalize_denominator :
  ∃ (A B C D E : ℚ),
    (3 : ℚ) / (4 * Real.sqrt 7 + 5 * Real.sqrt 3) = (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = 12 ∧
    B = 7 ∧
    C = -15 ∧
    D = 3 ∧
    E = 37 ∧
    (∀ k : ℚ, k ≠ 0 → (k * A * Real.sqrt B + k * C * Real.sqrt D) / (k * E) = (A * Real.sqrt B + C * Real.sqrt D) / E) ∧
    (∀ n : ℕ, n > 1 → ¬(∃ m : ℕ, B = m^2 * n)) ∧
    (∀ n : ℕ, n > 1 → ¬(∃ m : ℕ, D = m^2 * n)) :=
by sorry

end rationalize_denominator_l2698_269857


namespace simplify_expression_l2698_269844

theorem simplify_expression (x : ℝ) : 7*x + 8 - 3*x - 4 + 5 = 4*x + 9 := by
  sorry

end simplify_expression_l2698_269844


namespace clothes_fraction_l2698_269861

def incentive : ℚ := 240
def food_fraction : ℚ := 1/3
def savings_fraction : ℚ := 3/4
def savings_amount : ℚ := 84

theorem clothes_fraction (clothes_amount : ℚ) 
  (h1 : clothes_amount = incentive - food_fraction * incentive - savings_amount / savings_fraction) 
  (h2 : clothes_amount / incentive = 1/5) : 
  clothes_amount / incentive = 1/5 := by
sorry

end clothes_fraction_l2698_269861


namespace salary_percent_increase_l2698_269892

def salary_increase : ℝ := 5000
def new_salary : ℝ := 25000

theorem salary_percent_increase :
  let original_salary := new_salary - salary_increase
  let percent_increase := (salary_increase / original_salary) * 100
  percent_increase = 25 := by
sorry

end salary_percent_increase_l2698_269892


namespace no_inscribed_sphere_l2698_269821

structure Polyhedron where
  faces : ℕ
  paintedFaces : ℕ
  convex : Bool
  noAdjacentPainted : Bool

def canInscribeSphere (p : Polyhedron) : Prop :=
  sorry

theorem no_inscribed_sphere (p : Polyhedron) 
  (h_convex : p.convex = true)
  (h_painted : p.paintedFaces > p.faces / 2)
  (h_noAdjacent : p.noAdjacentPainted = true) :
  ¬(canInscribeSphere p) := by
  sorry

end no_inscribed_sphere_l2698_269821


namespace even_number_of_even_scores_l2698_269886

/-- Represents a team's score in the basketball competition -/
structure TeamScore where
  wins : ℕ
  losses : ℕ
  draws : ℕ

/-- The total number of teams in the competition -/
def numTeams : ℕ := 10

/-- The number of games each team plays -/
def gamesPerTeam : ℕ := numTeams - 1

/-- Calculate the total score for a team -/
def totalScore (ts : TeamScore) : ℕ :=
  2 * ts.wins + ts.draws

/-- The scores of all teams in the competition -/
def allTeamScores : Finset TeamScore :=
  sorry

/-- The sum of all team scores is equal to the total number of games multiplied by 2 -/
axiom total_score_sum : 
  (allTeamScores.sum totalScore) = (numTeams * gamesPerTeam)

/-- Theorem: There must be an even number of teams with an even total score -/
theorem even_number_of_even_scores : 
  Even (Finset.filter (fun ts => Even (totalScore ts)) allTeamScores).card :=
sorry

end even_number_of_even_scores_l2698_269886


namespace parabola_vertex_y_coordinate_l2698_269810

/-- The y-coordinate of the vertex of the parabola y = -3x^2 - 30x - 81 is -6 -/
theorem parabola_vertex_y_coordinate :
  let f : ℝ → ℝ := λ x ↦ -3 * x^2 - 30 * x - 81
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≤ f x₀ ∧ f x₀ = -6 :=
sorry

end parabola_vertex_y_coordinate_l2698_269810


namespace rental_cost_equality_l2698_269814

/-- The daily rate charged by Safety Rent-a-Car in dollars -/
def safety_daily_rate : ℝ := 21.95

/-- The per-mile rate charged by Safety Rent-a-Car in dollars -/
def safety_mile_rate : ℝ := 0.19

/-- The daily rate charged by City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The per-mile rate charged by City Rentals in dollars -/
def city_mile_rate : ℝ := 0.21

/-- The mileage at which the cost is the same for both rental companies -/
def equal_cost_mileage : ℝ := 150

theorem rental_cost_equality :
  safety_daily_rate + safety_mile_rate * equal_cost_mileage =
  city_daily_rate + city_mile_rate * equal_cost_mileage :=
by sorry

end rental_cost_equality_l2698_269814


namespace equation_solution_l2698_269852

theorem equation_solution :
  ∃! y : ℝ, (7 : ℝ) ^ (y + 6) = 343 ^ y :=
by
  -- The proof goes here
  sorry

end equation_solution_l2698_269852


namespace expand_product_l2698_269880

theorem expand_product (y : ℝ) : 4 * (y - 3) * (y^2 + 2*y + 4) = 4*y^3 - 4*y^2 - 8*y - 48 := by
  sorry

end expand_product_l2698_269880


namespace other_factor_of_60n_l2698_269837

theorem other_factor_of_60n (n : ℕ) (other : ℕ) : 
  (∃ k : ℕ, 60 * n = 4 * other * k) → 
  (∀ m : ℕ, m < n → ¬∃ j : ℕ, 60 * m = 4 * other * j) → 
  n = 8 → 
  other = 120 :=
by sorry

end other_factor_of_60n_l2698_269837


namespace quadratic_equation_solution_l2698_269875

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - 2*x + 3 - 4*x
  ∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 6 ∧ x₂ = 3 - Real.sqrt 6 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end quadratic_equation_solution_l2698_269875


namespace ace_of_hearts_probability_l2698_269887

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of cards drawn simultaneously -/
def CardsDrawn : ℕ := 2

/-- Number of Ace of Hearts in a standard deck -/
def AceOfHearts : ℕ := 1

/-- Probability of drawing the Ace of Hearts when 2 cards are drawn simultaneously from a standard 52-card deck -/
theorem ace_of_hearts_probability :
  (AceOfHearts * (StandardDeck - CardsDrawn)) / (StandardDeck.choose CardsDrawn) = 1 / 26 := by
  sorry

end ace_of_hearts_probability_l2698_269887


namespace family_savings_l2698_269890

def initial_savings : ℕ := 1147240
def income : ℕ := 509600
def expenses : ℕ := 276000

theorem family_savings : initial_savings + income - expenses = 1340840 := by
  sorry

end family_savings_l2698_269890


namespace rational_as_cube_sum_ratio_l2698_269859

theorem rational_as_cube_sum_ratio (q : ℚ) (hq : 0 < q) : 
  ∃ (a b c d : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
    q = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end rational_as_cube_sum_ratio_l2698_269859


namespace parking_spaces_on_first_level_l2698_269808

/-- Represents a 4-level parking garage -/
structure ParkingGarage where
  level1 : ℕ
  level2 : ℕ
  level3 : ℕ
  level4 : ℕ

/-- The conditions of the parking garage problem -/
def validParkingGarage (g : ParkingGarage) : Prop :=
  g.level2 = g.level1 + 8 ∧
  g.level3 = g.level2 + 12 ∧
  g.level4 = g.level3 - 9 ∧
  g.level1 + g.level2 + g.level3 + g.level4 = 299 - 100

theorem parking_spaces_on_first_level (g : ParkingGarage) 
  (h : validParkingGarage g) : g.level1 = 40 := by
  sorry

end parking_spaces_on_first_level_l2698_269808


namespace correct_factorization_l2698_269822

theorem correct_factorization (a : ℝ) : a^2 - 3*a - 4 = (a - 4) * (a + 1) := by
  sorry

end correct_factorization_l2698_269822


namespace perpendicular_bisector_b_value_l2698_269896

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The perpendicular bisector of a line segment -/
def isPerpBisector (p q : Point) (a b c : ℝ) : Prop :=
  let midpoint : Point := ⟨(p.x + q.x) / 2, (p.y + q.y) / 2⟩
  a * midpoint.x + b * midpoint.y = c ∧
  (q.y - p.y) * a = (q.x - p.x) * b

/-- The theorem stating that b = 6 given the conditions -/
theorem perpendicular_bisector_b_value :
  let p : Point := ⟨0, 0⟩
  let q : Point := ⟨4, 8⟩
  ∀ b : ℝ, isPerpBisector p q 1 1 b → b = 6 := by
  sorry

end perpendicular_bisector_b_value_l2698_269896


namespace quadratic_solutions_l2698_269835

/-- Given a quadratic function f(x) = x^2 + mx, if its axis of symmetry is x = 3,
    then the solutions to x^2 + mx = 0 are 0 and 6. -/
theorem quadratic_solutions (m : ℝ) :
  (∀ x, x^2 + m*x = (x - 3)^2 + k) →
  (∃ k, ∀ x, x^2 + m*x = (x - 3)^2 + k) →
  (∀ x, x^2 + m*x = 0 ↔ x = 0 ∨ x = 6) :=
by sorry

end quadratic_solutions_l2698_269835


namespace exists_valid_assignment_l2698_269874

/-- Represents a 7x7 square table with four corner squares deleted -/
def Table := Fin 7 → Fin 7 → Option ℤ

/-- Checks if a position is a valid square on the table -/
def isValidSquare (row col : Fin 7) : Prop :=
  ¬((row = 0 ∧ col = 0) ∨ (row = 0 ∧ col = 6) ∨ (row = 6 ∧ col = 0) ∨ (row = 6 ∧ col = 6))

/-- Represents a Greek cross on the table -/
structure GreekCross (t : Table) where
  center_row : Fin 7
  center_col : Fin 7
  valid : isValidSquare center_row center_col ∧
          isValidSquare center_row (center_col - 1) ∧
          isValidSquare center_row (center_col + 1) ∧
          isValidSquare (center_row - 1) center_col ∧
          isValidSquare (center_row + 1) center_col

/-- Calculates the sum of integers in a Greek cross -/
def sumGreekCross (t : Table) (cross : GreekCross t) : ℤ :=
  sorry

/-- Calculates the sum of all integers in the table -/
def sumTable (t : Table) : ℤ :=
  sorry

/-- Main theorem to prove -/
theorem exists_valid_assignment :
  ∃ (t : Table), (∀ (cross : GreekCross t), sumGreekCross t cross < 0) ∧ sumTable t > 0 :=
sorry

end exists_valid_assignment_l2698_269874


namespace smallest_b_value_l2698_269830

theorem smallest_b_value : ∃ (b : ℝ), b > 0 ∧
  (∀ (x : ℝ), x > 0 →
    (9 * Real.sqrt ((3 * x)^2 + 2^2) - 6 * x^2 - 4) / (Real.sqrt (4 + 6 * x^2) + 5) = 3 →
    b ≤ x) ∧
  (9 * Real.sqrt ((3 * b)^2 + 2^2) - 6 * b^2 - 4) / (Real.sqrt (4 + 6 * b^2) + 5) = 3 ∧
  b = Real.sqrt (11 / 30) :=
by sorry

end smallest_b_value_l2698_269830


namespace curve_is_hyperbola_l2698_269824

/-- The equation of the curve in polar coordinates -/
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.sin θ)

/-- The equation of the curve in Cartesian coordinates -/
def cartesian_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) - y = 1

/-- The definition of a hyperbola in Cartesian coordinates -/
def is_hyperbola (f : ℝ × ℝ → ℝ) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
    ∀ x y, f (x, y) = a * x^2 + b * y^2 + c * x * y + d * x + e * y

theorem curve_is_hyperbola :
  ∃ f : ℝ × ℝ → ℝ, (∀ x y, f (x, y) = 0 ↔ cartesian_equation x y) ∧ is_hyperbola f :=
sorry

end curve_is_hyperbola_l2698_269824


namespace linear_function_properties_l2698_269853

/-- A linear function passing through two given points -/
def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

theorem linear_function_properties :
  ∃ (k b : ℝ),
    (linear_function k b (-4) = -9) ∧
    (linear_function k b 3 = 5) ∧
    (k = 2 ∧ b = -1) ∧
    (∃ x, linear_function k b x = 0 ∧ x = 1/2) ∧
    (linear_function k b 0 = -1) ∧
    (1/2 * 1/2 * 1 = 1/4) := by
  sorry

end linear_function_properties_l2698_269853


namespace parabola_sequence_property_l2698_269839

/-- Sequence of parabolas Lₙ with general form y² = (2/Sₙ)(x - Tₙ/Sₙ) -/
def T (n : ℕ) : ℚ := (3^n - 1) / 2

def S (n : ℕ) : ℚ := 3^(n-1)

/-- The expression 2Tₙ - 3Sₙ always equals -1 for any positive integer n -/
theorem parabola_sequence_property (n : ℕ) (h : n > 0) : 2 * T n - 3 * S n = -1 := by
  sorry

end parabola_sequence_property_l2698_269839


namespace coefficient_of_x_term_l2698_269803

theorem coefficient_of_x_term (x : ℝ) : 
  let expansion := (Real.sqrt x - 2 / x) ^ 5
  ∃ c : ℝ, c = -10 ∧ 
    ∃ t : ℝ → ℝ, (expansion = c * x + t x) ∧ 
    (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → |t h / h| < ε) :=
by sorry

end coefficient_of_x_term_l2698_269803


namespace green_hat_cost_l2698_269869

theorem green_hat_cost (total_hats : ℕ) (green_hats : ℕ) (blue_hat_cost : ℕ) (total_price : ℕ) :
  total_hats = 85 →
  green_hats = 20 →
  blue_hat_cost = 6 →
  total_price = 530 →
  (total_hats - green_hats) * blue_hat_cost + green_hats * 7 = total_price :=
by sorry

end green_hat_cost_l2698_269869


namespace paco_cookies_problem_l2698_269860

/-- Calculates the initial number of salty cookies Paco had --/
def initial_salty_cookies (initial_sweet : ℕ) (sweet_eaten : ℕ) (salty_eaten : ℕ) (difference : ℕ) : ℕ :=
  initial_sweet - difference

theorem paco_cookies_problem (initial_sweet : ℕ) (sweet_eaten : ℕ) (salty_eaten : ℕ) (difference : ℕ)
  (h1 : initial_sweet = 39)
  (h2 : sweet_eaten = 32)
  (h3 : salty_eaten = 23)
  (h4 : difference = sweet_eaten - salty_eaten)
  (h5 : difference = 9) :
  initial_salty_cookies initial_sweet sweet_eaten salty_eaten difference = 30 := by
  sorry

#eval initial_salty_cookies 39 32 23 9

end paco_cookies_problem_l2698_269860


namespace trig_functions_and_expression_l2698_269841

theorem trig_functions_and_expression (α : Real) (h : Real.tan α = -Real.sqrt 3) :
  (((Real.sin α = Real.sqrt 3 / 2) ∧ (Real.cos α = -1 / 2)) ∨
   ((Real.sin α = -Real.sqrt 3 / 2) ∧ (Real.cos α = 1 / 2))) ∧
  ((Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 2 + Real.sqrt 3) := by
  sorry

end trig_functions_and_expression_l2698_269841


namespace integer_division_property_l2698_269812

theorem integer_division_property (n : ℕ) : 
  100 ≤ n ∧ n ≤ 1997 →
  (∃ k : ℕ, (2^n + 2 : ℕ) = k * n) ↔ n ∈ ({66, 198, 398, 798} : Set ℕ) :=
by sorry

end integer_division_property_l2698_269812


namespace minimum_value_curve_exponent_l2698_269826

theorem minimum_value_curve_exponent (m n : ℝ) (a : ℝ) : 
  m > 0 → n > 0 → m + n = 1 → 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → (1/x) + (16/y) ≥ (1/m) + (16/n)) →
  ((m/5)^a = n/4) →
  a = 1/2 := by sorry

end minimum_value_curve_exponent_l2698_269826


namespace locus_of_Q_is_ellipse_l2698_269855

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define a point P on circle E
def point_P (x y : ℝ) : Prop := circle_E x y

-- Define point Q as the intersection of perpendicular bisector of PF and radius PE
def point_Q (x y : ℝ) : Prop :=
  ∃ (px py : ℝ), point_P px py ∧
  (x - px)^2 + (y - py)^2 = (x - 1)^2 + y^2 ∧
  (x + px + 2) * (x - 1) + y * py = 0

-- Theorem stating the locus of Q is an ellipse
theorem locus_of_Q_is_ellipse :
  ∀ (x y : ℝ), point_Q x y ↔ x^2/4 + y^2/3 = 1 :=
sorry

end locus_of_Q_is_ellipse_l2698_269855


namespace multiples_of_3_or_5_not_6_l2698_269813

def count_multiples (n : ℕ) (max : ℕ) : ℕ :=
  (max / n)

theorem multiples_of_3_or_5_not_6 (max : ℕ) (h : max = 150) : 
  count_multiples 3 max + count_multiples 5 max - count_multiples 15 max - count_multiples 6 max = 45 := by
  sorry

#check multiples_of_3_or_5_not_6

end multiples_of_3_or_5_not_6_l2698_269813


namespace race_head_start_l2698_269898

theorem race_head_start (course_length : ℝ) (speed_ratio : ℝ) (head_start : ℝ) : 
  course_length = 84 →
  speed_ratio = 2 →
  course_length / speed_ratio = (course_length - head_start) / 1 →
  head_start = 42 := by
sorry

end race_head_start_l2698_269898


namespace percentage_problem_l2698_269825

theorem percentage_problem : 
  let percentage : ℝ := 12
  let total : ℝ := 160
  let given_percentage : ℝ := 38
  let given_total : ℝ := 80
  let difference : ℝ := 11.2
  (given_percentage / 100) * given_total - (percentage / 100) * total = difference
  := by sorry

end percentage_problem_l2698_269825


namespace complement_union_theorem_l2698_269888

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_theorem : (U \ A) ∪ B = {0, 2, 4} := by sorry

end complement_union_theorem_l2698_269888


namespace factorial_expression_l2698_269846

theorem factorial_expression : (Nat.factorial (Nat.factorial 4)) / (Nat.factorial 4) = Nat.factorial 23 := by
  sorry

end factorial_expression_l2698_269846


namespace remainder_not_always_power_of_four_l2698_269856

theorem remainder_not_always_power_of_four :
  ∃ n : ℕ, n ≥ 2 ∧ ¬∃ k : ℕ, (2^(2^n) : ℕ) % (2^n - 1) = 4^k := by
  sorry

end remainder_not_always_power_of_four_l2698_269856


namespace gasoline_tank_capacity_l2698_269858

theorem gasoline_tank_capacity : ∃ (x : ℝ),
  x > 0 ∧
  (7/8 * x - 15 = 2/3 * x) ∧
  x = 72 := by
sorry

end gasoline_tank_capacity_l2698_269858


namespace howard_earnings_l2698_269894

/-- Calculates the money earned from washing windows --/
def money_earned (initial_amount current_amount : ℕ) : ℕ :=
  current_amount - initial_amount

theorem howard_earnings :
  let initial_amount : ℕ := 26
  let current_amount : ℕ := 52
  money_earned initial_amount current_amount = 26 := by
  sorry

end howard_earnings_l2698_269894


namespace complex_sum_problem_l2698_269862

theorem complex_sum_problem (p r t u : ℝ) :
  let q : ℝ := 5
  let s : ℝ := 2 * q
  t = -p - r →
  Complex.I * (q + s + u) = Complex.I * 7 →
  Complex.I * u + Complex.I = Complex.I * (-8) + Complex.I :=
by
  sorry

end complex_sum_problem_l2698_269862


namespace fred_grew_four_carrots_l2698_269871

/-- The number of carrots Sally grew -/
def sally_carrots : ℕ := 6

/-- The total number of carrots grown by Sally and Fred -/
def total_carrots : ℕ := 10

/-- The number of carrots Fred grew -/
def fred_carrots : ℕ := total_carrots - sally_carrots

theorem fred_grew_four_carrots : fred_carrots = 4 := by
  sorry

end fred_grew_four_carrots_l2698_269871


namespace complex_number_problem_l2698_269811

theorem complex_number_problem (z : ℂ) : 
  z + Complex.abs z = 5 + Complex.I * Real.sqrt 3 → z = 11/5 + Complex.I * Real.sqrt 3 := by
  sorry

end complex_number_problem_l2698_269811


namespace total_necklaces_is_1942_l2698_269834

/-- Represents the production of necklaces for a single machine on a given day -/
structure DailyProduction where
  machine : Nat
  day : Nat
  amount : Nat

/-- Calculates the total number of necklaces produced over three days -/
def totalNecklaces (productions : List DailyProduction) : Nat :=
  productions.map (·.amount) |>.sum

/-- The production data for all machines over three days -/
def necklaceProduction : List DailyProduction := [
  -- Sunday
  { machine := 1, day := 1, amount := 45 },
  { machine := 2, day := 1, amount := 108 },
  { machine := 3, day := 1, amount := 230 },
  { machine := 4, day := 1, amount := 184 },
  -- Monday
  { machine := 1, day := 2, amount := 59 },
  { machine := 2, day := 2, amount := 54 },
  { machine := 3, day := 2, amount := 230 },
  { machine := 4, day := 2, amount := 368 },
  -- Tuesday
  { machine := 1, day := 3, amount := 59 },
  { machine := 2, day := 3, amount := 108 },
  { machine := 3, day := 3, amount := 276 },
  { machine := 4, day := 3, amount := 221 }
]

/-- Theorem: The total number of necklaces produced over three days is 1942 -/
theorem total_necklaces_is_1942 : totalNecklaces necklaceProduction = 1942 := by
  sorry

end total_necklaces_is_1942_l2698_269834


namespace units_digit_of_product_first_four_composites_l2698_269804

def first_four_composite_numbers : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_first_four_composites :
  units_digit (product_of_list first_four_composite_numbers) = 8 := by
  sorry

end units_digit_of_product_first_four_composites_l2698_269804


namespace rachel_homework_pages_l2698_269829

/-- The total number of pages for math and biology homework -/
def total_math_biology_pages (math_pages biology_pages : ℕ) : ℕ :=
  math_pages + biology_pages

/-- Theorem: Given Rachel has 8 pages of math homework and 3 pages of biology homework,
    the total number of pages for math and biology homework is 11. -/
theorem rachel_homework_pages : total_math_biology_pages 8 3 = 11 := by
  sorry

end rachel_homework_pages_l2698_269829


namespace set_subset_relations_l2698_269805

theorem set_subset_relations : 
  ({1,2,3} : Set ℕ) ⊆ {1,2,3} ∧ (∅ : Set ℕ) ⊆ {1} := by sorry

end set_subset_relations_l2698_269805


namespace best_calorie_deal_l2698_269816

-- Define the food options
structure FoodOption where
  name : String
  quantity : Nat
  price : Nat
  caloriesPerItem : Nat

-- Define the function to calculate calories per dollar
def caloriesPerDollar (option : FoodOption) : Rat :=
  (option.quantity * option.caloriesPerItem : Rat) / option.price

-- Define the food options
def burritos : FoodOption := ⟨"Burritos", 10, 6, 120⟩
def burgers : FoodOption := ⟨"Burgers", 5, 8, 400⟩
def pizza : FoodOption := ⟨"Pizza", 8, 10, 300⟩
def donuts : FoodOption := ⟨"Donuts", 15, 12, 250⟩

-- Define the list of food options
def foodOptions : List FoodOption := [burritos, burgers, pizza, donuts]

-- Theorem statement
theorem best_calorie_deal :
  (caloriesPerDollar donuts = 312.5) ∧
  (∀ option ∈ foodOptions, caloriesPerDollar option ≤ caloriesPerDollar donuts) ∧
  (caloriesPerDollar donuts - caloriesPerDollar burgers = 62.5) :=
sorry

end best_calorie_deal_l2698_269816


namespace unique_solution_equation_l2698_269818

theorem unique_solution_equation : ∃! x : ℝ, 3 * x + 3 * 15 + 3 * 18 + 11 = 152 := by
  sorry

end unique_solution_equation_l2698_269818


namespace mean_equality_implies_y_value_l2698_269895

theorem mean_equality_implies_y_value : 
  let nums : List ℝ := [4, 6, 10, 14]
  let mean_nums := (nums.sum) / (nums.length : ℝ)
  mean_nums = (y + 18) / 2 → y = -1 :=
by
  sorry

end mean_equality_implies_y_value_l2698_269895


namespace initial_speed_proof_l2698_269827

theorem initial_speed_proof (total_distance : ℝ) (first_duration : ℝ) (second_speed : ℝ) (second_duration : ℝ) (remaining_distance : ℝ) :
  total_distance = 600 →
  first_duration = 3 →
  second_speed = 80 →
  second_duration = 4 →
  remaining_distance = 130 →
  ∃ initial_speed : ℝ,
    initial_speed * first_duration + second_speed * second_duration = total_distance - remaining_distance ∧
    initial_speed = 50 := by
  sorry

end initial_speed_proof_l2698_269827


namespace geometric_sequence_product_l2698_269815

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The roots of the quadratic equation -/
def are_roots (a : ℕ → ℝ) : Prop :=
  3 * (a 1)^2 + 7 * (a 1) - 9 = 0 ∧ 3 * (a 10)^2 + 7 * (a 10) - 9 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → are_roots a → a 4 * a 7 = -3 := by
  sorry

end geometric_sequence_product_l2698_269815


namespace stone_piles_impossible_l2698_269842

/-- Represents a configuration of stone piles -/
def StonePiles := List Nat

/-- The initial configuration of stone piles -/
def initial_piles : StonePiles := [51, 49, 5]

/-- Merges two piles in the configuration -/
def merge_piles (piles : StonePiles) (i j : Nat) : StonePiles :=
  sorry

/-- Splits an even-numbered pile into two equal piles -/
def split_pile (piles : StonePiles) (i : Nat) : StonePiles :=
  sorry

/-- Checks if a configuration consists of 105 piles of 1 stone each -/
def is_final_state (piles : StonePiles) : Prop :=
  piles.length = 105 ∧ piles.all (· = 1)

/-- Represents a sequence of operations on the stone piles -/
inductive Operation
  | Merge (i j : Nat)
  | Split (i : Nat)

/-- Applies a sequence of operations to the initial configuration -/
def apply_operations (ops : List Operation) : StonePiles :=
  sorry

theorem stone_piles_impossible :
  ∀ (ops : List Operation), ¬(is_final_state (apply_operations ops)) :=
sorry

end stone_piles_impossible_l2698_269842


namespace modulus_of_complex_reciprocal_l2698_269868

theorem modulus_of_complex_reciprocal (z : ℂ) : 
  Complex.abs (1 / (1 + Complex.I * Real.sqrt 3)) = 1/4 := by
  sorry

end modulus_of_complex_reciprocal_l2698_269868


namespace complex_cube_root_l2698_269802

theorem complex_cube_root (a b : ℕ+) :
  (↑a + ↑b * Complex.I) ^ 3 = 2 + 11 * Complex.I →
  ↑a + ↑b * Complex.I = 2 + Complex.I := by
  sorry

end complex_cube_root_l2698_269802


namespace cube_surface_area_l2698_269845

/-- Given a cube with volume 1728 cubic centimeters, its surface area is 864 square centimeters. -/
theorem cube_surface_area (v : ℝ) (h : v = 1728) : 
  (6 * ((v ^ (1/3)) ^ 2)) = 864 :=
sorry

end cube_surface_area_l2698_269845


namespace problem_statement_l2698_269849

def A (n r : ℕ) : ℕ := n.factorial / (n - r).factorial

def C (n r : ℕ) : ℕ := n.factorial / (r.factorial * (n - r).factorial)

theorem problem_statement : A 6 2 + C 6 4 = 45 := by
  sorry

end problem_statement_l2698_269849


namespace boy_walking_time_l2698_269809

/-- Given a boy who walks at 6/7 of his usual rate and reaches school 4 minutes early, 
    his usual time to reach the school is 24 minutes. -/
theorem boy_walking_time (usual_rate : ℝ) (usual_time : ℝ) 
  (h1 : usual_rate > 0) (h2 : usual_time > 0) : 
  (6 / 7 * usual_rate) * (usual_time - 4) = usual_rate * usual_time → 
  usual_time = 24 := by
  sorry

end boy_walking_time_l2698_269809


namespace parabola_symmetry_l2698_269870

theorem parabola_symmetry (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = 2 * x₁^2 →
  y₂ = 2 * x₂^2 →
  y₁ + y₂ = x₁ + x₂ + 2*m →
  x₁ * x₂ = -1/2 →
  m = 3/2 := by sorry

end parabola_symmetry_l2698_269870


namespace factorization_proof_l2698_269899

theorem factorization_proof (x y : ℝ) : 9*x^2*y - y = y*(3*x + 1)*(3*x - 1) := by
  sorry

end factorization_proof_l2698_269899


namespace intersection_point_theorem_l2698_269833

/-- A parabola that intersects the coordinate axes at three distinct points -/
structure Parabola where
  p : ℝ
  q : ℝ
  distinct_intersections : ∃ (a b : ℝ), a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ q ≠ 0 ∧ a^2 + p*a + q = 0 ∧ b^2 + p*b + q = 0

/-- The circle passing through the three intersection points of the parabola with the coordinate axes -/
def intersection_circle (par : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ (a b : ℝ), a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧ par.q ≠ 0 ∧
            a^2 + par.p*a + par.q = 0 ∧ b^2 + par.p*b + par.q = 0 ∧
            (x^2 + y^2) * (a*b) + x * (par.q*(a+b)) + y * (par.q*par.p) + par.q^2 = 0}

/-- Theorem: All intersection circles pass through the point (0, 1) -/
theorem intersection_point_theorem (par : Parabola) :
  (0, 1) ∈ intersection_circle par :=
sorry

end intersection_point_theorem_l2698_269833


namespace shoe_cost_l2698_269848

/-- Given a suit purchase, a discount, and a total paid amount, prove the cost of shoes. -/
theorem shoe_cost (suit_price discount total_paid : ℤ) (h1 : suit_price = 430) (h2 : discount = 100) (h3 : total_paid = 520) :
  suit_price + (total_paid + discount - suit_price) = total_paid + discount := by
  sorry

#eval 520 + 100 - 430  -- Expected output: 190

end shoe_cost_l2698_269848


namespace retirement_sum_is_70_l2698_269831

/-- Represents the retirement policy of a company -/
structure RetirementPolicy where
  hireYear : Nat
  hireAge : Nat
  retirementYear : Nat
  retirementSum : Nat

/-- Theorem: The required total of age and years of employment for retirement is 70 -/
theorem retirement_sum_is_70 (policy : RetirementPolicy) 
  (h1 : policy.hireYear = 1987)
  (h2 : policy.hireAge = 32)
  (h3 : policy.retirementYear = 2006) :
  policy.retirementSum = 70 := by
  sorry

#check retirement_sum_is_70

end retirement_sum_is_70_l2698_269831


namespace triangular_pyramid_least_faces_triangular_pyramid_faces_l2698_269882

-- Define the shapes
inductive Shape
  | TriangularPrism
  | QuadrangularPrism
  | TriangularPyramid
  | QuadrangularPyramid
  | TruncatedQuadrangularPyramid

-- Function to count the number of faces for each shape
def numFaces (s : Shape) : ℕ :=
  match s with
  | Shape.TriangularPrism => 5
  | Shape.QuadrangularPrism => 6
  | Shape.TriangularPyramid => 4
  | Shape.QuadrangularPyramid => 5
  | Shape.TruncatedQuadrangularPyramid => 6  -- Assuming the truncated pyramid has a top face

-- Theorem stating that the triangular pyramid has the least number of faces
theorem triangular_pyramid_least_faces :
  ∀ s : Shape, numFaces Shape.TriangularPyramid ≤ numFaces s :=
by
  sorry

-- Theorem stating that the number of faces of a triangular pyramid is 4
theorem triangular_pyramid_faces :
  numFaces Shape.TriangularPyramid = 4 :=
by
  sorry

end triangular_pyramid_least_faces_triangular_pyramid_faces_l2698_269882


namespace wednesday_water_intake_l2698_269881

/-- Represents the water intake for a week -/
structure WeeklyWaterIntake where
  total : ℕ
  mon_thu_sat : ℕ
  tue_fri_sun : ℕ
  wed : ℕ

/-- The water intake on Wednesday can be determined from the other data -/
theorem wednesday_water_intake (w : WeeklyWaterIntake)
  (h_total : w.total = 60)
  (h_mon_thu_sat : w.mon_thu_sat = 9)
  (h_tue_fri_sun : w.tue_fri_sun = 8)
  (h_balance : w.total = 3 * w.mon_thu_sat + 3 * w.tue_fri_sun + w.wed) :
  w.wed = 9 := by
  sorry

#check wednesday_water_intake

end wednesday_water_intake_l2698_269881


namespace cube_constructions_l2698_269865

/-- The number of rotational symmetries of a cube -/
def cubeRotations : ℕ := 24

/-- The total number of ways to place 3 blue cubes in 8 positions -/
def totalPlacements : ℕ := Nat.choose 8 3

/-- The number of invariant configurations under 180° rotation around edge axes -/
def edgeRotationInvariants : ℕ := 4

/-- The number of invariant configurations under 180° rotation around face axes -/
def faceRotationInvariants : ℕ := 4

/-- The sum of all fixed points under different rotations -/
def sumFixedPoints : ℕ := totalPlacements + 6 * edgeRotationInvariants + 3 * faceRotationInvariants

/-- The number of unique constructions of a 2x2x2 cube with 5 white and 3 blue unit cubes -/
def uniqueConstructions : ℕ := sumFixedPoints / cubeRotations

theorem cube_constructions : uniqueConstructions = 4 := by
  sorry

end cube_constructions_l2698_269865


namespace arithmetic_sequence_and_parabola_vertex_l2698_269800

/-- Given that a, b, c, and d form an arithmetic sequence, and (a, d) is the vertex of y = x^2 - 2x + 5, prove that b + c = 5 -/
theorem arithmetic_sequence_and_parabola_vertex (a b c d : ℝ) :
  (∃ k : ℝ, b = a + k ∧ c = a + 2*k ∧ d = a + 3*k) →  -- arithmetic sequence condition
  (a = 1 ∧ d = 4) →  -- vertex condition (derived from y = x^2 - 2x + 5)
  b + c = 5 := by
sorry

end arithmetic_sequence_and_parabola_vertex_l2698_269800


namespace necklace_cost_proof_l2698_269851

/-- The cost of a single necklace -/
def necklace_cost : ℝ := 40000

/-- The total cost of the purchase -/
def total_cost : ℝ := 240000

/-- The number of necklaces purchased -/
def num_necklaces : ℕ := 3

theorem necklace_cost_proof :
  (num_necklaces : ℝ) * necklace_cost + 3 * necklace_cost = total_cost :=
by sorry

end necklace_cost_proof_l2698_269851


namespace dining_bill_calculation_l2698_269850

theorem dining_bill_calculation (total_spent : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) 
  (h_total : total_spent = 132)
  (h_tip : tip_rate = 0.20)
  (h_tax : tax_rate = 0.10) :
  ∃ (original_price : ℝ),
    original_price = 100 ∧
    total_spent = original_price * (1 + tax_rate) * (1 + tip_rate) := by
  sorry

end dining_bill_calculation_l2698_269850


namespace inequality_and_minimum_value_proof_l2698_269883

theorem inequality_and_minimum_value_proof 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  ∃ (min_val : ℝ) (min_x : ℝ),
    (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
      a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y)) ∧ 
    (∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
      a^2 / x + b^2 / y = (a + b)^2 / (x + y) ↔ x / y = a / b) ∧
    (∀ (x : ℝ) (hx : 0 < x ∧ x < 1/2), 
      2 / x + 9 / (1 - 2*x) ≥ min_val) ∧
    (0 < min_x ∧ min_x < 1/2) ∧
    (2 / min_x + 9 / (1 - 2*min_x) = min_val) ∧
    min_val = 25 ∧
    min_x = 1/5 := by
  sorry

end inequality_and_minimum_value_proof_l2698_269883


namespace camila_weeks_to_match_steven_l2698_269801

-- Define the initial number of hikes for Camila
def camila_initial_hikes : ℕ := 7

-- Define Amanda's hikes in terms of Camila's
def amanda_hikes : ℕ := 8 * camila_initial_hikes

-- Define Steven's hikes in terms of Amanda's
def steven_hikes : ℕ := amanda_hikes + 15

-- Define Camila's planned hikes per week
def camila_weekly_hikes : ℕ := 4

-- Theorem to prove
theorem camila_weeks_to_match_steven :
  (steven_hikes - camila_initial_hikes) / camila_weekly_hikes = 16 := by
  sorry

end camila_weeks_to_match_steven_l2698_269801


namespace watermelon_seeds_l2698_269807

/-- Given 4 watermelons with a total of 400 seeds, prove that each watermelon has 100 seeds. -/
theorem watermelon_seeds (num_watermelons : ℕ) (total_seeds : ℕ) 
  (h1 : num_watermelons = 4) 
  (h2 : total_seeds = 400) : 
  total_seeds / num_watermelons = 100 := by
  sorry

end watermelon_seeds_l2698_269807


namespace range_of_f_on_large_interval_l2698_269820

/-- A function with period 1 --/
def periodic_function (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 1) = g x

/-- The function f defined as f(x) = x + g(x) --/
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := x + g x

/-- The range of a function on an interval --/
def range_on (f : ℝ → ℝ) (a b : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc a b, f x = y}

theorem range_of_f_on_large_interval
    (g : ℝ → ℝ)
    (h_periodic : periodic_function g)
    (h_range : range_on (f g) 3 4 = Set.Icc (-2) 5) :
    range_on (f g) (-10) 10 = Set.Icc (-15) 11 := by
  sorry

end range_of_f_on_large_interval_l2698_269820


namespace inequality_proof_l2698_269847

theorem inequality_proof (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ 16 * (x / z + z / x + 2) :=
by sorry

end inequality_proof_l2698_269847


namespace circle_equation_l2698_269854

/-- The equation of a circle with center (1, 1) and radius 1 -/
theorem circle_equation : 
  ∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 1 ↔ 
  ((x, y) : ℝ × ℝ) ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 1} :=
by sorry

end circle_equation_l2698_269854


namespace even_odd_sum_difference_l2698_269873

/-- Sum of first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of first n positive odd integers -/
def sumFirstOddIntegers (n : ℕ) : ℕ := n * n

/-- The positive difference between the sum of the first 25 positive even integers
    and the sum of the first 20 positive odd integers is 250 -/
theorem even_odd_sum_difference : 
  (sumFirstEvenIntegers 25) - (sumFirstOddIntegers 20) = 250 := by
  sorry

end even_odd_sum_difference_l2698_269873


namespace sum_of_reciprocals_of_roots_l2698_269872

theorem sum_of_reciprocals_of_roots (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^3 - 2022*a + 1011 = 0 →
  b^3 - 2022*b + 1011 = 0 →
  c^3 - 2022*c + 1011 = 0 →
  1/a + 1/b + 1/c = 2 := by
sorry

end sum_of_reciprocals_of_roots_l2698_269872


namespace similar_right_triangles_leg_length_l2698_269836

/-- Two similar right triangles with legs 12 and 9 in one, and x and 6 in the other, have x = 8 -/
theorem similar_right_triangles_leg_length : ∀ x : ℝ,
  (12 : ℝ) / x = 9 / 6 → x = 8 := by
  sorry

end similar_right_triangles_leg_length_l2698_269836


namespace total_pets_l2698_269876

theorem total_pets (taylor_pets : ℕ) (friends_with_double : ℕ) (friends_with_two : ℕ) : 
  taylor_pets = 4 → 
  friends_with_double = 3 → 
  friends_with_two = 2 → 
  taylor_pets + friends_with_double * (2 * taylor_pets) + friends_with_two * 2 = 32 := by
  sorry

end total_pets_l2698_269876


namespace range_of_a_l2698_269867

noncomputable section

-- Define the piecewise function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

-- Define the property of f being increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem range_of_a (a : ℝ) 
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : is_increasing (f a)) :
  a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end range_of_a_l2698_269867


namespace absolute_value_inequality_solution_l2698_269879

theorem absolute_value_inequality_solution (x : ℝ) :
  |2*x - 7| < 3 ↔ 2 < x ∧ x < 5 := by sorry

end absolute_value_inequality_solution_l2698_269879
