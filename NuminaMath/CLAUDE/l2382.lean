import Mathlib

namespace inequality_equivalence_l2382_238206

theorem inequality_equivalence (p : ℝ) (hp : p > 0) : 
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → (1 / Real.sin x ^ 2) + (p / Real.cos x ^ 2) ≥ 9) ↔ 
  p ≥ 4 := by
sorry

end inequality_equivalence_l2382_238206


namespace joe_egg_hunt_l2382_238278

theorem joe_egg_hunt (park_eggs : ℕ) (town_hall_eggs : ℕ) (total_eggs : ℕ) 
  (h1 : park_eggs = 5)
  (h2 : town_hall_eggs = 3)
  (h3 : total_eggs = 20) :
  total_eggs - park_eggs - town_hall_eggs = 12 := by
  sorry

end joe_egg_hunt_l2382_238278


namespace abc_sum_and_squares_l2382_238287

theorem abc_sum_and_squares (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  (a*b + b*c + c*a = -1/2) ∧ (a^4 + b^4 + c^4 = 1/2) := by
  sorry

end abc_sum_and_squares_l2382_238287


namespace tetrahedron_smallest_faces_l2382_238251

/-- Represents the number of faces in a geometric shape. -/
def faces (shape : String) : ℕ :=
  match shape with
  | "Tetrahedron" => 4
  | "Quadrangular pyramid" => 5
  | "Triangular prism" => 5
  | "Triangular pyramid" => 4
  | _ => 0

/-- The list of shapes we're considering. -/
def shapes : List String :=
  ["Tetrahedron", "Quadrangular pyramid", "Triangular prism", "Triangular pyramid"]

/-- Theorem stating that the tetrahedron has the smallest number of faces among the given shapes. -/
theorem tetrahedron_smallest_faces :
    ∀ shape ∈ shapes, faces "Tetrahedron" ≤ faces shape := by
  sorry

#check tetrahedron_smallest_faces

end tetrahedron_smallest_faces_l2382_238251


namespace condition_relationship_l2382_238243

/-- Given propositions p, q, and r, we define what it means for one proposition
    to be a sufficient but not necessary condition for another. -/
def sufficient_not_necessary (a b : Prop) : Prop :=
  (a → b) ∧ ¬(b → a)

/-- Given propositions p, q, and r, we define what it means for one proposition
    to be a necessary but not sufficient condition for another. -/
def necessary_not_sufficient (a b : Prop) : Prop :=
  (b → a) ∧ ¬(a → b)

/-- Theorem stating the relationship between p, q, and r based on their conditional properties. -/
theorem condition_relationship (p q r : Prop) 
  (h1 : sufficient_not_necessary p q) 
  (h2 : sufficient_not_necessary q r) : 
  necessary_not_sufficient r p :=
sorry

end condition_relationship_l2382_238243


namespace smallest_cube_multiplier_l2382_238205

theorem smallest_cube_multiplier (n : ℕ) (h : n = 1512) :
  (∃ (y : ℕ), 49 * n = y^3) ∧
  (∀ (x : ℕ), x > 0 → x < 49 → ¬∃ (y : ℕ), x * n = y^3) :=
sorry

end smallest_cube_multiplier_l2382_238205


namespace average_yield_is_100_l2382_238250

/-- Calculates the average yield per tree given the number of trees and their yields. -/
def averageYield (x : ℕ) : ℚ :=
  let trees1 := x + 2
  let trees2 := x
  let trees3 := x - 2
  let yield1 := 30
  let yield2 := 120
  let yield3 := 180
  let totalTrees := trees1 + trees2 + trees3
  let totalNuts := trees1 * yield1 + trees2 * yield2 + trees3 * yield3
  totalNuts / totalTrees

/-- Theorem stating that the average yield per tree is 100 when x = 10. -/
theorem average_yield_is_100 : averageYield 10 = 100 := by
  sorry

end average_yield_is_100_l2382_238250


namespace manufacturing_earnings_l2382_238215

/-- Calculates total earnings given hourly wage, bonus per widget, number of widgets, and work hours -/
def totalEarnings (hourlyWage : ℚ) (bonusPerWidget : ℚ) (numWidgets : ℕ) (workHours : ℕ) : ℚ :=
  hourlyWage * workHours + bonusPerWidget * numWidgets

/-- Proves that the total earnings for the given conditions is $700 -/
theorem manufacturing_earnings :
  totalEarnings (12.5) (0.16) 1250 40 = 700 := by
  sorry

end manufacturing_earnings_l2382_238215


namespace no_common_points_necessary_not_sufficient_for_parallel_l2382_238262

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields to define a line in 3D space
  -- This is a placeholder and should be properly defined

/-- Two lines have no common points -/
def no_common_points (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines having no common points
  sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  -- Define the condition for two lines being parallel
  sorry

/-- Skew lines: lines that are not parallel and do not intersect -/
def skew (l1 l2 : Line3D) : Prop :=
  no_common_points l1 l2 ∧ ¬parallel l1 l2

theorem no_common_points_necessary_not_sufficient_for_parallel :
  (∀ l1 l2 : Line3D, parallel l1 l2 → no_common_points l1 l2) ∧
  (∃ l1 l2 : Line3D, no_common_points l1 l2 ∧ ¬parallel l1 l2) :=
by
  sorry

#check no_common_points_necessary_not_sufficient_for_parallel

end no_common_points_necessary_not_sufficient_for_parallel_l2382_238262


namespace average_of_four_numbers_l2382_238292

theorem average_of_four_numbers (x y z w : ℝ) 
  (h : (5 / 2) * (x + y + z + w) = 25) : 
  (x + y + z + w) / 4 = 2.5 := by
  sorry

end average_of_four_numbers_l2382_238292


namespace cookies_in_class_l2382_238298

/-- The number of cookies brought by Mona, Jasmine, and Rachel -/
def total_cookies (mona jasmine rachel : ℕ) : ℕ := mona + jasmine + rachel

/-- Theorem stating the total number of cookies brought to class -/
theorem cookies_in_class : ∃ (jasmine rachel : ℕ),
  jasmine = 20 - 5 ∧ 
  rachel = jasmine + 10 ∧
  total_cookies 20 jasmine rachel = 60 := by
sorry

end cookies_in_class_l2382_238298


namespace sum_of_coefficients_equals_14_9_l2382_238277

/-- A quadratic function f(x) = ax^2 + bx + c with vertex at (6, -2) and passing through (3, 0) -/
def QuadraticFunction (a b c : ℚ) : ℚ → ℚ :=
  fun x ↦ a * x^2 + b * x + c

theorem sum_of_coefficients_equals_14_9 (a b c : ℚ) :
  (QuadraticFunction a b c 6 = -2) →  -- vertex at (6, -2)
  (QuadraticFunction a b c 3 = 0) →   -- passes through (3, 0)
  (∀ x, QuadraticFunction a b c (12 - x) = QuadraticFunction a b c x) →  -- vertical symmetry
  a + b + c = 14 / 9 :=
by sorry

end sum_of_coefficients_equals_14_9_l2382_238277


namespace min_value_of_u_l2382_238239

theorem min_value_of_u (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (x + 1/x) * (y + 1/(4*y)) ≥ 25/8 := by
  sorry

end min_value_of_u_l2382_238239


namespace right_triangle_identification_l2382_238202

theorem right_triangle_identification (a b c : ℝ) : 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 1 ∧ b = 2 ∧ c = Real.sqrt 3) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 4) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 9) →
  (a^2 + b^2 = c^2 ↔ a = 1 ∧ b = 2 ∧ c = Real.sqrt 3) :=
by sorry

end right_triangle_identification_l2382_238202


namespace sequence_sum_l2382_238296

def is_six_digit_number (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def sequence_property (x : ℕ → ℕ) : Prop :=
  is_six_digit_number (x 1) ∧
  ∀ n : ℕ, n ≥ 1 → Nat.Prime (x (n + 1)) ∧ (x (n + 1) ∣ x n + 1)

theorem sequence_sum (x : ℕ → ℕ) (h : sequence_property x) : x 19 + x 20 = 5 := by
  sorry

end sequence_sum_l2382_238296


namespace lisa_max_below_a_l2382_238279

/-- Represents Lisa's quiz performance and goal --/
structure QuizPerformance where
  total_quizzes : ℕ
  goal_percentage : ℚ
  completed_quizzes : ℕ
  completed_as : ℕ

/-- Calculates the maximum number of remaining quizzes where Lisa can score below 'A' --/
def max_below_a (qp : QuizPerformance) : ℕ :=
  let total_as_needed : ℕ := (qp.goal_percentage * qp.total_quizzes).ceil.toNat
  let remaining_quizzes : ℕ := qp.total_quizzes - qp.completed_quizzes
  remaining_quizzes - (total_as_needed - qp.completed_as)

/-- Theorem stating that given Lisa's quiz performance, the maximum number of remaining quizzes where she can score below 'A' is 7 --/
theorem lisa_max_below_a :
  let qp : QuizPerformance := {
    total_quizzes := 60,
    goal_percentage := 3/4,
    completed_quizzes := 30,
    completed_as := 22
  }
  max_below_a qp = 7 := by sorry

end lisa_max_below_a_l2382_238279


namespace meaningful_expression_l2382_238270

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x + 2)) ↔ x > -2 := by sorry

end meaningful_expression_l2382_238270


namespace max_value_on_ellipse_l2382_238294

theorem max_value_on_ellipse :
  ∃ (M : ℝ), M = Real.sqrt 13 ∧
  ∀ (x y : ℝ), x^2 / 9 + y^2 / 4 = 1 →
  x + y ≤ M :=
by sorry

end max_value_on_ellipse_l2382_238294


namespace tangent_line_parallel_increasing_intervals_decreasing_interval_extreme_values_l2382_238201

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - (2*a + 3)*x + a^2

-- Define the derivative of f(x)
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - (2*a + 3)

-- Theorem for part 1
theorem tangent_line_parallel (a : ℝ) :
  f_derivative a (-1) = 2 → a = -1/2 := by sorry

-- Theorems for part 2
theorem increasing_intervals :
  let a := -2
  ∀ x, (x < 1/3 ∨ x > 1) → (f_derivative a x > 0) := by sorry

theorem decreasing_interval :
  let a := -2
  ∀ x, (1/3 < x ∧ x < 1) → (f_derivative a x < 0) := by sorry

theorem extreme_values :
  let a := -2
  (f a (1/3) = 112/27) ∧ (f a 1 = 4) := by sorry

end tangent_line_parallel_increasing_intervals_decreasing_interval_extreme_values_l2382_238201


namespace tensor_product_result_l2382_238253

-- Define the sets P and Q
def P : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def Q : Set ℝ := {x | x > 1}

-- Define the ⊗ operation
def tensorProduct (A B : Set ℝ) : Set ℝ :=
  {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

-- Theorem statement
theorem tensor_product_result :
  tensorProduct P Q = {x | (0 ≤ x ∧ x ≤ 1) ∨ (x > 2)} := by
  sorry

end tensor_product_result_l2382_238253


namespace equation_solutions_l2382_238241

theorem equation_solutions : 
  ∀ m : ℝ, 9 * m^2 - (2*m + 1)^2 = 0 ↔ m = 1 ∨ m = -1/5 := by
sorry

end equation_solutions_l2382_238241


namespace parent_teacher_night_duration_l2382_238285

def time_to_school : ℕ := 20
def time_from_school : ℕ := 20
def total_time : ℕ := 110

theorem parent_teacher_night_duration :
  total_time - (time_to_school + time_from_school) = 70 :=
by sorry

end parent_teacher_night_duration_l2382_238285


namespace smallest_fitting_polygon_l2382_238273

/-- A regular polygon with n sides that can fit perfectly when rotated by 40° or 60° -/
def FittingPolygon (n : ℕ) : Prop :=
  n > 0 ∧ (40 * n) % 360 = 0 ∧ (60 * n) % 360 = 0

/-- The smallest number of sides for a fitting polygon is 18 -/
theorem smallest_fitting_polygon : ∃ (n : ℕ), FittingPolygon n ∧ ∀ m, FittingPolygon m → n ≤ m :=
  sorry

end smallest_fitting_polygon_l2382_238273


namespace find_x_l2382_238203

theorem find_x (x : ℕ+) 
  (n : ℤ) (h_n : n = x.val^2 + 2*x.val + 17)
  (d : ℤ) (h_d : d = 2*x.val + 5)
  (h_div : n = d * x.val + 7) : 
  x.val = 2 := by
sorry

end find_x_l2382_238203


namespace sufficient_not_necessary_condition_l2382_238236

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a > b ∧ b > 0 → a^2 > b^2) ∧
  ∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0) :=
by sorry

end sufficient_not_necessary_condition_l2382_238236


namespace fifteen_factorial_largest_square_exponent_sum_l2382_238297

def largest_perfect_square_exponent_sum (n : ℕ) : ℕ :=
  let prime_factors := Nat.factors n
  let max_square_exponents := prime_factors.map (fun p => (Nat.factorization n p) / 2)
  max_square_exponents.sum

theorem fifteen_factorial_largest_square_exponent_sum :
  largest_perfect_square_exponent_sum (Nat.factorial 15) = 10 := by
  sorry

end fifteen_factorial_largest_square_exponent_sum_l2382_238297


namespace arithmetic_sequence_sum_l2382_238233

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  d > 0 →
  a 1 + a 2 + a 3 = 15 →
  a 1 * a 2 * a 3 = 80 →
  a 11 + a 12 + a 13 = 105 := by
  sorry

end arithmetic_sequence_sum_l2382_238233


namespace perp_condition_l2382_238293

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The first line x + y = 0 -/
def line1 : Line := { slope := -1, intercept := 0 }

/-- The second line x - ay = 0 -/
def line2 (a : ℝ) : Line := { slope := a, intercept := 0 }

/-- Theorem: a = 1 is necessary and sufficient for perpendicularity -/
theorem perp_condition (a : ℝ) :
  perpendicular line1 (line2 a) ↔ a = 1 := by
  sorry

end perp_condition_l2382_238293


namespace regular_polygon_sides_l2382_238219

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  interior_angle = 135 → n * (180 - interior_angle) = 360 → n = 8 := by
  sorry

end regular_polygon_sides_l2382_238219


namespace arrangements_count_l2382_238290

/-- The number of different arrangements of 5 students (2 girls and 3 boys) 
    where the two girls are not next to each other -/
def num_arrangements : ℕ := 72

/-- The number of ways to arrange 3 boys -/
def boy_arrangements : ℕ := 6

/-- The number of ways to insert 2 girls into 4 possible spaces -/
def girl_insertions : ℕ := 12

theorem arrangements_count : 
  num_arrangements = boy_arrangements * girl_insertions :=
by sorry

end arrangements_count_l2382_238290


namespace new_computer_cost_new_computer_cost_is_600_l2382_238210

theorem new_computer_cost (used_computers_cost : ℕ) (savings : ℕ) : ℕ :=
  let new_computer_cost := used_computers_cost + savings
  new_computer_cost

#check new_computer_cost 400 200

theorem new_computer_cost_is_600 :
  new_computer_cost 400 200 = 600 := by sorry

end new_computer_cost_new_computer_cost_is_600_l2382_238210


namespace hilton_marbles_l2382_238212

/-- Calculates the final number of marbles Hilton has -/
def final_marbles (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost + 2 * lost

theorem hilton_marbles : final_marbles 26 6 10 = 42 := by
  sorry

end hilton_marbles_l2382_238212


namespace smallest_four_digit_divisible_by_53_l2382_238281

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end smallest_four_digit_divisible_by_53_l2382_238281


namespace roots_of_polynomial_l2382_238291

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

-- Theorem statement
theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 4) :=
by sorry

end roots_of_polynomial_l2382_238291


namespace cost_verification_max_purchase_l2382_238275

/-- Represents the cost of a single bat -/
def bat_cost : ℝ := 70

/-- Represents the cost of a single ball -/
def ball_cost : ℝ := 20

/-- Represents the discount rate when purchasing at least 3 bats and 3 balls -/
def discount_rate : ℝ := 0.10

/-- Represents the sales tax rate -/
def sales_tax_rate : ℝ := 0.08

/-- Represents the budget -/
def budget : ℝ := 270

/-- Verifies that the given costs satisfy the conditions -/
theorem cost_verification : 
  2 * bat_cost + 4 * ball_cost = 220 ∧ 
  bat_cost + 6 * ball_cost = 190 := by sorry

/-- Proves that the maximum number of bats and balls that can be purchased is 3 -/
theorem max_purchase : 
  ∀ n : ℕ, 
    n ≥ 3 → 
    n * (bat_cost + ball_cost) * (1 - discount_rate) * (1 + sales_tax_rate) ≤ budget → 
    n ≤ 3 := by sorry

end cost_verification_max_purchase_l2382_238275


namespace man_work_time_l2382_238237

/-- The time taken by a man to complete a work given the following conditions:
    - A man, a woman, and a boy together complete the work in 3 days
    - A woman alone can do the work in 6 days
    - A boy alone can do the work in 18 days -/
theorem man_work_time (work : ℝ) (man_rate woman_rate boy_rate : ℝ) :
  work > 0 ∧
  man_rate > 0 ∧ woman_rate > 0 ∧ boy_rate > 0 ∧
  man_rate + woman_rate + boy_rate = work / 3 ∧
  woman_rate = work / 6 ∧
  boy_rate = work / 18 →
  work / man_rate = 9 := by
  sorry

#check man_work_time

end man_work_time_l2382_238237


namespace infinite_symmetry_centers_l2382_238224

/-- A point in a 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A figure in a 2D space -/
structure Figure :=
  (points : Set Point)

/-- A symmetry transformation with respect to a center point -/
def symmetryTransform (center : Point) (p : Point) : Point :=
  { x := 2 * center.x - p.x,
    y := 2 * center.y - p.y }

/-- A center of symmetry for a figure -/
def isSymmetryCenter (f : Figure) (c : Point) : Prop :=
  ∀ p ∈ f.points, symmetryTransform c p ∈ f.points

/-- The set of all symmetry centers for a figure -/
def symmetryCenters (f : Figure) : Set Point :=
  { c | isSymmetryCenter f c }

/-- Main theorem: If a figure has more than one center of symmetry, 
    it must have infinitely many centers of symmetry -/
theorem infinite_symmetry_centers (f : Figure) :
  (∃ c₁ c₂ : Point, c₁ ≠ c₂ ∧ c₁ ∈ symmetryCenters f ∧ c₂ ∈ symmetryCenters f) →
  ¬ Finite (symmetryCenters f) :=
sorry

end infinite_symmetry_centers_l2382_238224


namespace parallel_vectors_k_value_l2382_238240

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, b.1 = t * a.1 ∧ b.2 = t * a.2

theorem parallel_vectors_k_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (k, 4)
  are_parallel a b → k = -2 := by
sorry

end parallel_vectors_k_value_l2382_238240


namespace smallest_angle_of_quadrilateral_with_ratio_l2382_238259

/-- 
Given a quadrilateral with interior angles in a 4:5:6:7 ratio,
prove that the smallest interior angle measures 720/11 degrees.
-/
theorem smallest_angle_of_quadrilateral_with_ratio (a b c d : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- All angles are positive
  b = (5/4) * a ∧ c = (6/4) * a ∧ d = (7/4) * a →  -- Angles are in 4:5:6:7 ratio
  a + b + c + d = 360 →  -- Sum of angles in a quadrilateral is 360°
  a = 720 / 11 := by
  sorry

end smallest_angle_of_quadrilateral_with_ratio_l2382_238259


namespace circus_tent_capacity_l2382_238214

/-- The number of sections in the circus tent -/
def num_sections : ℕ := 4

/-- The capacity of each section in the circus tent -/
def section_capacity : ℕ := 246

/-- The total capacity of the circus tent -/
def total_capacity : ℕ := num_sections * section_capacity

theorem circus_tent_capacity : total_capacity = 984 := by
  sorry

end circus_tent_capacity_l2382_238214


namespace chess_tournament_theorem_l2382_238260

/-- Represents a player's score sequence in the chess tournament -/
structure PlayerScore where
  round1 : ℕ
  round2 : ℕ
  round3 : ℕ
  round4 : ℕ

/-- Checks if a sequence is quadratic -/
def isQuadraticSequence (s : PlayerScore) : Prop :=
  ∃ a t r : ℕ, 
    s.round1 = a ∧
    s.round2 = a + t + r ∧
    s.round3 = a + 2*t + 4*r ∧
    s.round4 = a + 3*t + 9*r

/-- Checks if a sequence is arithmetic -/
def isArithmeticSequence (s : PlayerScore) : Prop :=
  ∃ b d : ℕ, 
    s.round1 = b ∧
    s.round2 = b + d ∧
    s.round3 = b + 2*d ∧
    s.round4 = b + 3*d

/-- Calculates the total score of a player -/
def totalScore (s : PlayerScore) : ℕ :=
  s.round1 + s.round2 + s.round3 + s.round4

/-- The main theorem -/
theorem chess_tournament_theorem 
  (playerA playerB : PlayerScore)
  (h1 : isQuadraticSequence playerA)
  (h2 : isArithmeticSequence playerB)
  (h3 : totalScore playerA = totalScore playerB)
  (h4 : totalScore playerA ≤ 25)
  (h5 : totalScore playerB ≤ 25) :
  playerA.round1 + playerA.round2 + playerB.round1 + playerB.round2 = 12 :=
by sorry

end chess_tournament_theorem_l2382_238260


namespace even_operations_l2382_238227

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem even_operations (n : ℤ) (h : is_even n) :
  (is_even (n + 4)) ∧ (is_even (n - 6)) ∧ (is_even (n * 8)) := by
  sorry

end even_operations_l2382_238227


namespace myrtle_eggs_count_l2382_238220

/-- The number of eggs Myrtle has after her trip -/
def myrtle_eggs : ℕ :=
  let num_hens : ℕ := 3
  let eggs_per_hen_per_day : ℕ := 3
  let days_gone : ℕ := 7
  let neighbor_taken : ℕ := 12
  let dropped : ℕ := 5
  
  let total_laid : ℕ := num_hens * eggs_per_hen_per_day * days_gone
  let remaining_after_neighbor : ℕ := total_laid - neighbor_taken
  remaining_after_neighbor - dropped

theorem myrtle_eggs_count : myrtle_eggs = 46 := by
  sorry

end myrtle_eggs_count_l2382_238220


namespace prove_a_equals_3x_l2382_238286

theorem prove_a_equals_3x (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 27*x^3) 
  (h3 : a - b = 3*x) : 
  a = 3*x := by sorry

end prove_a_equals_3x_l2382_238286


namespace min_ratio_T2_T1_l2382_238232

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a triangle is acute -/
def is_acute (t : Triangle) : Prop := sorry

/-- Calculates the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Represents the altitude of a triangle -/
structure Altitude :=
  (base : Point) (foot : Point)

/-- Calculates the projection of a point onto a line -/
def project (p : Point) (l : Point × Point) : Point := sorry

/-- Calculates the area of T_1 as defined in the problem -/
def area_T1 (t : Triangle) (AD BE CF : Altitude) : ℝ := sorry

/-- Calculates the area of T_2 as defined in the problem -/
def area_T2 (t : Triangle) (AD BE CF : Altitude) : ℝ := sorry

/-- The main theorem: The ratio T_2/T_1 is always greater than or equal to 25 for any acute triangle -/
theorem min_ratio_T2_T1 (t : Triangle) (AD BE CF : Altitude) :
  is_acute t →
  (area_T2 t AD BE CF) / (area_T1 t AD BE CF) ≥ 25 := by
  sorry

end min_ratio_T2_T1_l2382_238232


namespace mass_ratio_simplification_l2382_238244

-- Define the units
def kg : ℚ → ℚ := id
def ton : ℚ → ℚ := (· * 1000)

-- Define the ratio
def ratio (a b : ℚ) : ℚ × ℚ := (a, b)

-- Define the problem
theorem mass_ratio_simplification :
  let mass1 := kg 250
  let mass2 := ton 0.5
  let simplified_ratio := ratio 1 2
  let decimal_value := 0.5
  (mass1 / mass2 = decimal_value) ∧
  (ratio (mass1 / gcd mass1 mass2) (mass2 / gcd mass1 mass2) = simplified_ratio) := by
  sorry


end mass_ratio_simplification_l2382_238244


namespace descent_time_is_two_hours_l2382_238280

/-- Proves that the time taken to descend a hill is 2 hours, given specific conditions. -/
theorem descent_time_is_two_hours 
  (time_to_top : ℝ) 
  (avg_speed_total : ℝ) 
  (avg_speed_up : ℝ) 
  (time_to_top_is_four : time_to_top = 4)
  (avg_speed_total_is_three : avg_speed_total = 3)
  (avg_speed_up_is_two_point_two_five : avg_speed_up = 2.25) :
  let distance_to_top : ℝ := avg_speed_up * time_to_top
  let total_distance : ℝ := 2 * distance_to_top
  let total_time : ℝ := total_distance / avg_speed_total
  time_to_top - (total_time - time_to_top) = 2 := by
  sorry

#check descent_time_is_two_hours

end descent_time_is_two_hours_l2382_238280


namespace tournament_rankings_count_l2382_238247

/-- Represents a player in the tournament -/
inductive Player : Type
| P : Player
| Q : Player
| R : Player
| S : Player

/-- Represents a match between two players -/
structure Match :=
  (player1 : Player)
  (player2 : Player)

/-- Represents the tournament structure -/
structure Tournament :=
  (saturday_match1 : Match)
  (saturday_match2 : Match)
  (sunday_championship : Match)
  (sunday_consolation : Match)

/-- Represents a ranking of players -/
def Ranking := List Player

/-- Returns all possible rankings for a given tournament -/
def possibleRankings (t : Tournament) : List Ranking :=
  sorry

theorem tournament_rankings_count :
  ∀ t : Tournament,
  (t.saturday_match1.player1 = Player.P ∧ t.saturday_match1.player2 = Player.Q) →
  (t.saturday_match2.player1 = Player.R ∧ t.saturday_match2.player2 = Player.S) →
  (List.length (possibleRankings t) = 16) :=
by sorry

end tournament_rankings_count_l2382_238247


namespace replaced_person_weight_l2382_238242

/-- Proves that the weight of the replaced person is 55 kg given the conditions of the problem. -/
theorem replaced_person_weight
  (initial_count : Nat)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : initial_count = 8)
  (h2 : weight_increase = 2.5)
  (h3 : new_person_weight = 75) :
  (new_person_weight - initial_count * weight_increase) = 55 := by
  sorry

end replaced_person_weight_l2382_238242


namespace min_value_of_expression_min_value_attained_l2382_238213

theorem min_value_of_expression (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -961 :=
by sorry

theorem min_value_attained (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) = -961 ↔ x = Real.sqrt 197 ∨ x = -Real.sqrt 197 :=
by sorry

end min_value_of_expression_min_value_attained_l2382_238213


namespace pigeon_count_theorem_l2382_238276

theorem pigeon_count_theorem :
  ∃! n : ℕ,
    300 < n ∧ n < 900 ∧
    n % 2 = 1 ∧
    n % 3 = 2 ∧
    n % 4 = 3 ∧
    n % 5 = 4 ∧
    n % 6 = 5 ∧
    n % 7 = 0 ∧
    n = 539 := by
  sorry

end pigeon_count_theorem_l2382_238276


namespace faster_speed_calculation_l2382_238255

/-- Proves that given a distance traveled at a certain speed, if the person were to travel an additional distance in the same time at a faster speed, we can calculate that faster speed. -/
theorem faster_speed_calculation (D : ℝ) (v_original : ℝ) (additional_distance : ℝ) (v_faster : ℝ) :
  D = 33.333333333333336 →
  v_original = 10 →
  additional_distance = 20 →
  D / v_original = (D + additional_distance) / v_faster →
  v_faster = 16 := by
  sorry

end faster_speed_calculation_l2382_238255


namespace complex_fraction_simplification_l2382_238289

theorem complex_fraction_simplification :
  ((-1 : ℂ) + 3*Complex.I) / (1 + Complex.I) = 1 + 2*Complex.I :=
by sorry

end complex_fraction_simplification_l2382_238289


namespace quadrilateral_areas_product_is_square_l2382_238217

/-- Represents a convex quadrilateral divided by its diagonals -/
structure ConvexQuadrilateral where
  /-- Areas of the four triangles formed by the diagonals -/
  areas : Fin 4 → ℕ

/-- Theorem: The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals is a perfect square -/
theorem quadrilateral_areas_product_is_square (q : ConvexQuadrilateral) :
  ∃ k : ℕ, (q.areas 0) * (q.areas 1) * (q.areas 2) * (q.areas 3) = k^2 := by
  sorry

end quadrilateral_areas_product_is_square_l2382_238217


namespace rectangular_field_area_l2382_238274

/-- Given a rectangular field with one side uncovered and three sides fenced, 
    calculate its area. -/
theorem rectangular_field_area 
  (L : ℝ) -- Length of the uncovered side
  (total_fencing : ℝ) -- Total length of fencing used
  (h1 : L = 20) -- The uncovered side is 20 feet long
  (h2 : total_fencing = 64) -- The total fencing required is 64 feet
  : L * ((total_fencing - L) / 2) = 440 := by
sorry

end rectangular_field_area_l2382_238274


namespace train_length_train_length_proof_l2382_238264

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : Real) (time : Real) : Real :=
  let length_km := speed * (time / 3600)
  let length_m := length_km * 1000
  length_m

/-- Proves that a train with speed 60 km/hr crossing a pole in 15 seconds has a length of 250 meters -/
theorem train_length_proof :
  train_length 60 15 = 250 := by
  sorry

end train_length_train_length_proof_l2382_238264


namespace prob_third_batch_value_l2382_238207

/-- Represents a batch of parts -/
structure Batch :=
  (total : ℕ)
  (standard : ℕ)
  (h : standard ≤ total)

/-- Represents the experiment of selecting two standard parts from a batch -/
def select_two_standard (b : Batch) : ℚ :=
  (b.standard : ℚ) / b.total * ((b.standard - 1) : ℚ) / (b.total - 1)

/-- The probability of selecting the third batch given that two standard parts were selected -/
def prob_third_batch (b1 b2 b3 : Batch) : ℚ :=
  let p1 := select_two_standard b1
  let p2 := select_two_standard b2
  let p3 := select_two_standard b3
  p3 / (p1 + p2 + p3)

theorem prob_third_batch_value :
  let b1 : Batch := ⟨30, 20, by norm_num⟩
  let b2 : Batch := ⟨30, 15, by norm_num⟩
  let b3 : Batch := ⟨30, 10, by norm_num⟩
  prob_third_batch b1 b2 b3 = 3 / 68 := by
  sorry

end prob_third_batch_value_l2382_238207


namespace triangle_x_coordinate_sum_l2382_238282

/-- Given two triangles ABC and ADF with specific areas and coordinates,
    prove that the sum of all possible x-coordinates of A is -635.6 -/
theorem triangle_x_coordinate_sum :
  let triangle_ABC_area : ℝ := 2010
  let triangle_ADF_area : ℝ := 8020
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (226, 0)
  let D : ℝ × ℝ := (680, 380)
  let F : ℝ × ℝ := (700, 400)
  ∃ (x₁ x₂ : ℝ), 
    (∃ (y₁ : ℝ), triangle_ABC_area = (1/2) * 226 * |y₁|) ∧
    (∃ (y₂ : ℝ), triangle_ADF_area = (1/2) * 20 * |x₁ - y₂ + 300| / Real.sqrt 2) ∧
    (∃ (y₃ : ℝ), triangle_ADF_area = (1/2) * 20 * |x₂ - y₃ + 300| / Real.sqrt 2) ∧
    x₁ + x₂ = -635.6 := by
  sorry

#check triangle_x_coordinate_sum

end triangle_x_coordinate_sum_l2382_238282


namespace perpendicular_line_to_plane_perpendicular_to_all_lines_l2382_238254

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_to_plane_perpendicular_to_all_lines
  (m n : Line) (α : Plane)
  (h1 : perpendicular m α)
  (h2 : contained_in n α) :
  perpendicular_lines m n :=
sorry

end perpendicular_line_to_plane_perpendicular_to_all_lines_l2382_238254


namespace intersection_of_specific_lines_l2382_238258

/-- Two lines in a plane -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- The point of intersection of two lines -/
def intersection (l1 l2 : Line) : ℚ × ℚ :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  (x, y)

/-- Theorem: The intersection of y = -3x + 1 and y + 1 = 15x is (1/9, 2/3) -/
theorem intersection_of_specific_lines :
  let line1 : Line := { slope := -3, intercept := 1 }
  let line2 : Line := { slope := 15, intercept := -1 }
  intersection line1 line2 = (1/9, 2/3) := by
sorry

end intersection_of_specific_lines_l2382_238258


namespace product_remainder_l2382_238261

theorem product_remainder (a b c : ℕ) (ha : a = 2457) (hb : b = 7623) (hc : c = 91309) : 
  (a * b * c) % 10 = 9 := by
  sorry

end product_remainder_l2382_238261


namespace package_weight_is_five_l2382_238272

/-- Calculates the weight of a package given the total shipping cost, flat fee, and cost per pound. -/
def package_weight (total_cost flat_fee cost_per_pound : ℚ) : ℚ :=
  (total_cost - flat_fee) / cost_per_pound

/-- Theorem stating that given the specific shipping costs, the package weighs 5 pounds. -/
theorem package_weight_is_five :
  package_weight 9 5 (4/5) = 5 := by
  sorry

end package_weight_is_five_l2382_238272


namespace bookstore_inventory_theorem_l2382_238230

theorem bookstore_inventory_theorem (historical_fiction : ℝ) (mystery : ℝ) (science_fiction : ℝ) (romance : ℝ)
  (historical_fiction_new : ℝ) (mystery_new : ℝ) (science_fiction_new : ℝ) (romance_new : ℝ)
  (h1 : historical_fiction = 0.4)
  (h2 : mystery = 0.3)
  (h3 : science_fiction = 0.2)
  (h4 : romance = 0.1)
  (h5 : historical_fiction_new = 0.35 * historical_fiction)
  (h6 : mystery_new = 0.6 * mystery)
  (h7 : science_fiction_new = 0.45 * science_fiction)
  (h8 : romance_new = 0.8 * romance) :
  historical_fiction_new / (historical_fiction_new + mystery_new + science_fiction_new + romance_new) = 2 / 7 := by
  sorry

end bookstore_inventory_theorem_l2382_238230


namespace smallest_factor_for_cube_l2382_238271

theorem smallest_factor_for_cube (n : ℕ) : n > 0 ∧ n * 49 = (7 : ℕ)^3 ∧ ∀ m : ℕ, m > 0 ∧ m < n → ¬∃ k : ℕ, m * 49 = k^3 → n = 7 := by
  sorry

end smallest_factor_for_cube_l2382_238271


namespace fish_pond_population_l2382_238295

theorem fish_pond_population (initial_tagged : Nat) (second_catch : Nat) (tagged_in_second : Nat) :
  initial_tagged = 60 →
  second_catch = 60 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = initial_tagged / (1800 : ℚ) :=
by sorry

end fish_pond_population_l2382_238295


namespace complex_magnitude_problem_l2382_238266

theorem complex_magnitude_problem (z : ℂ) (h : Complex.I * z = 2 + 4 * Complex.I) :
  Complex.abs z = 2 * Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l2382_238266


namespace square_sum_mod_three_solution_l2382_238218

theorem square_sum_mod_three_solution (x y z : ℕ) :
  (x^2 + y^2 + z^2) % 3 = 1 →
  ((x = 3 ∧ y = 3 ∧ z = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 3) ∨
   (x = 2 ∧ y = 3 ∧ z = 3)) :=
by sorry

end square_sum_mod_three_solution_l2382_238218


namespace ice_cream_sales_l2382_238234

def tuesday_sales : ℕ := 12000

def wednesday_sales : ℕ := 2 * tuesday_sales

def total_sales : ℕ := tuesday_sales + wednesday_sales

theorem ice_cream_sales : total_sales = 36000 := by
  sorry

end ice_cream_sales_l2382_238234


namespace mr_green_potato_yield_l2382_238231

/-- Calculates the expected potato yield for a rectangular garden -/
def expected_potato_yield (length_steps : ℕ) (width_steps : ℕ) (step_length : ℚ) (yield_per_sqft : ℚ) : ℚ :=
  (length_steps : ℚ) * step_length * (width_steps : ℚ) * step_length * yield_per_sqft

/-- Theorem: The expected potato yield for Mr. Green's garden is 2109.375 pounds -/
theorem mr_green_potato_yield :
  expected_potato_yield 18 25 (5/2) (3/4) = 2109375/1000 := by
  sorry

end mr_green_potato_yield_l2382_238231


namespace parabola_equation_l2382_238229

/-- A parabola with vertex at the origin, directrix perpendicular to the x-axis, 
    and passing through the point (1, -√2) has the equation y² = 2x -/
theorem parabola_equation : ∃ (f : ℝ → ℝ),
  (∀ x y : ℝ, f x = y ↔ y^2 = 2*x) ∧ 
  (f 0 = 0) ∧ 
  (∃ a : ℝ, ∀ x : ℝ, (x < a ↔ f x < 0) ∧ (x > a ↔ f x > 0)) ∧
  (f 1 = -Real.sqrt 2) :=
sorry

end parabola_equation_l2382_238229


namespace solve_quadratic_equation_l2382_238238

theorem solve_quadratic_equation (x : ℝ) (h1 : 3 * x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 / 3 := by
  sorry

end solve_quadratic_equation_l2382_238238


namespace mean_proportional_proof_l2382_238223

theorem mean_proportional_proof :
  let a : ℝ := 7921
  let b : ℝ := 9481
  let m : ℝ := 8665
  m = (a * b).sqrt := by sorry

end mean_proportional_proof_l2382_238223


namespace sin_2alpha_minus_cos_2alpha_l2382_238246

theorem sin_2alpha_minus_cos_2alpha (α : Real) (h : Real.tan α = 3) :
  Real.sin (2 * α) - Real.cos (2 * α) = 7/5 := by
  sorry

end sin_2alpha_minus_cos_2alpha_l2382_238246


namespace triangle_ABC_properties_l2382_238216

open Real

theorem triangle_ABC_properties (A B C a b c : Real) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  2 * sin A * sin C * (1 / (tan A * tan C) - 1) = -1 ∧
  a + c = 3 * sqrt 3 / 2 ∧
  b = sqrt 3 →
  B = π / 3 ∧
  (1 / 2) * a * c * sin B = 5 * sqrt 3 / 16 := by
sorry


end triangle_ABC_properties_l2382_238216


namespace inscribed_sphere_polyhedron_volume_l2382_238225

/-- A polyhedron with an inscribed sphere -/
structure InscribedSpherePolyhedron where
  /-- The volume of the polyhedron -/
  volume : ℝ
  /-- The total surface area of the polyhedron -/
  surface_area : ℝ
  /-- The radius of the inscribed sphere -/
  inscribed_sphere_radius : ℝ
  /-- The radius is positive -/
  radius_pos : 0 < inscribed_sphere_radius

/-- 
Theorem: For a polyhedron with an inscribed sphere, 
the volume of the polyhedron is equal to one-third of the product 
of its total surface area and the radius of the inscribed sphere.
-/
theorem inscribed_sphere_polyhedron_volume 
  (p : InscribedSpherePolyhedron) : 
  p.volume = (1 / 3) * p.surface_area * p.inscribed_sphere_radius := by
  sorry

end inscribed_sphere_polyhedron_volume_l2382_238225


namespace reflection_result_l2382_238256

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 2)
  let reflected := (-p'.2, -p'.1)
  (reflected.1, reflected.2 + 2)

def D : ℝ × ℝ := (5, 0)

theorem reflection_result :
  let D' := reflect_x D
  let D'' := reflect_line D'
  D'' = (2, -3) := by sorry

end reflection_result_l2382_238256


namespace simplify_expression_l2382_238283

theorem simplify_expression (a : ℝ) (h : a ≠ 1) :
  1 - 1 / (1 + (a + 2) / (1 - a)) = (2 + a) / 3 := by
  sorry

end simplify_expression_l2382_238283


namespace proportion_problem_l2382_238299

-- Define the proportion relation
def in_proportion (a b c d : ℝ) : Prop := a * d = b * c

-- State the theorem
theorem proportion_problem :
  ∀ (a b c d : ℝ),
  in_proportion a b c d →
  a = 2 →
  b = 3 →
  c = 6 →
  d = 9 := by
sorry

end proportion_problem_l2382_238299


namespace gcd_of_36_and_60_l2382_238257

theorem gcd_of_36_and_60 : Nat.gcd 36 60 = 12 := by
  sorry

end gcd_of_36_and_60_l2382_238257


namespace square_circle_union_area_l2382_238208

/-- The area of the union of a square and a circle with specific dimensions -/
theorem square_circle_union_area :
  let square_side : ℝ := 12
  let circle_radius : ℝ := 12
  let square_area : ℝ := square_side ^ 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let overlap_area : ℝ := (1 / 4) * circle_area
  square_area + circle_area - overlap_area = 144 + 108 * π :=
by sorry

end square_circle_union_area_l2382_238208


namespace sum_of_primes_equals_210_l2382_238248

theorem sum_of_primes_equals_210 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h : 100^2 + 1^2 = 65^2 + 76^2 ∧ 100^2 + 1^2 = p * q) : 
  p + q = 210 := by
sorry

end sum_of_primes_equals_210_l2382_238248


namespace cone_height_l2382_238228

/-- The height of a cone given its lateral surface properties -/
theorem cone_height (r l : ℝ) (h : r > 0) (h' : l > 0) : 
  (l = 3) → (2 * Real.pi * r = 2 * Real.pi / 3 * 3) → 
  Real.sqrt (l^2 - r^2) = 2 * Real.sqrt 2 :=
by sorry

end cone_height_l2382_238228


namespace total_rope_inches_is_264_l2382_238269

/-- Represents the length of rope in feet for each week -/
def rope_length : Fin 4 → ℕ
  | 0 => 6  -- Week 1
  | 1 => 2 * rope_length 0  -- Week 2
  | 2 => rope_length 1 - 4  -- Week 3
  | 3 => rope_length 2 / 2  -- Week 4

/-- Converts feet to inches -/
def feet_to_inches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the total length of rope in feet at the end of the month -/
def total_rope_length : ℕ :=
  rope_length 0 + rope_length 1 + rope_length 2 - rope_length 3

/-- Theorem stating the total length of rope in inches at the end of the month -/
theorem total_rope_inches_is_264 : feet_to_inches total_rope_length = 264 := by
  sorry

end total_rope_inches_is_264_l2382_238269


namespace gcd_problem_l2382_238204

theorem gcd_problem (h : Nat.Prime 101) :
  Nat.gcd (101^6 + 1) (3 * 101^6 + 101^3 + 1) = 1 := by
  sorry

end gcd_problem_l2382_238204


namespace pen_profit_percentage_l2382_238284

/-- Calculates the profit percentage for a retailer selling pens --/
theorem pen_profit_percentage
  (num_pens : ℕ)
  (price_pens : ℕ)
  (discount_percent : ℚ)
  (h1 : num_pens = 120)
  (h2 : price_pens = 36)
  (h3 : discount_percent = 1/100)
  : ∃ (profit_percent : ℚ), profit_percent = 230/100 :=
by sorry

end pen_profit_percentage_l2382_238284


namespace complex_number_properties_l2382_238209

def z : ℂ := 4 + 3 * Complex.I

theorem complex_number_properties :
  Complex.abs z = 5 ∧ (1 + Complex.I) / z = (7 + Complex.I) / 25 := by
  sorry

end complex_number_properties_l2382_238209


namespace pet_store_total_l2382_238222

/-- The number of dogs for sale in the pet store -/
def num_dogs : ℕ := 12

/-- The number of cats for sale in the pet store -/
def num_cats : ℕ := num_dogs / 3

/-- The number of birds for sale in the pet store -/
def num_birds : ℕ := 4 * num_dogs

/-- The number of fish for sale in the pet store -/
def num_fish : ℕ := 5 * num_dogs

/-- The number of reptiles for sale in the pet store -/
def num_reptiles : ℕ := 2 * num_dogs

/-- The number of rodents for sale in the pet store -/
def num_rodents : ℕ := num_dogs

/-- The total number of animals for sale in the pet store -/
def total_animals : ℕ := num_dogs + num_cats + num_birds + num_fish + num_reptiles + num_rodents

theorem pet_store_total : total_animals = 160 := by
  sorry

end pet_store_total_l2382_238222


namespace grassland_area_l2382_238221

theorem grassland_area (width1 : ℝ) (length : ℝ) : 
  width1 > 0 → length > 0 →
  (width1 + 10) * length = 1000 →
  (width1 - 4) * length = 650 →
  width1 * length = 750 := by
sorry

end grassland_area_l2382_238221


namespace complement_of_M_l2382_238267

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 3, 5}

theorem complement_of_M : (U \ M) = {2, 4, 6} := by sorry

end complement_of_M_l2382_238267


namespace monotone_decreasing_range_a_l2382_238268

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4*a*x + 3 else (2 - 3*a)*x + 1

theorem monotone_decreasing_range_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → a ∈ Set.Icc (1/2) (2/3) := by
  sorry

end monotone_decreasing_range_a_l2382_238268


namespace oranges_per_box_l2382_238245

/-- Given a fruit farm that packs oranges, prove that each box contains 10 oranges. -/
theorem oranges_per_box (total_oranges : ℕ) (total_boxes : ℝ) 
  (h1 : total_oranges = 26500) 
  (h2 : total_boxes = 2650.0) : 
  (total_oranges : ℝ) / total_boxes = 10 := by
  sorry

end oranges_per_box_l2382_238245


namespace factorial_expression_equals_1584_l2382_238252

theorem factorial_expression_equals_1584 :
  (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end factorial_expression_equals_1584_l2382_238252


namespace line_symmetry_l2382_238263

/-- The equation of a line in the Cartesian plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry of a point about another point -/
def symmetric_point (p q : Point) : Point :=
  ⟨2 * q.x - p.x, 2 * q.y - p.y⟩

/-- Two lines are symmetric about a point if for any point on one line,
    its symmetric point about the given point lies on the other line -/
def symmetric_lines (l₁ l₂ : Line) (p : Point) : Prop :=
  ∀ x y : ℝ, l₁.a * x + l₁.b * y + l₁.c = 0 →
    let sym := symmetric_point ⟨x, y⟩ p
    l₂.a * sym.x + l₂.b * sym.y + l₂.c = 0

theorem line_symmetry :
  let l₁ : Line := ⟨3, -1, 2, sorry⟩
  let l₂ : Line := ⟨3, -1, -6, sorry⟩
  let p : Point := ⟨1, 1⟩
  symmetric_lines l₁ l₂ p := by sorry

end line_symmetry_l2382_238263


namespace intersection_A_complement_B_l2382_238226

-- Define set A
def A : Set ℝ := {a : ℝ | ∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0}

-- Define set B
def B : Set ℝ := {a : ℝ | ∀ x : ℝ, ¬(|x - 4| + |x - 3| < a)}

-- State the theorem
theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Ioo 1 2 := by sorry

end intersection_A_complement_B_l2382_238226


namespace symmetry_of_P_l2382_238265

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis. -/
def symmetry_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The original point P. -/
def P : Point :=
  ⟨-2, -1⟩

theorem symmetry_of_P :
  symmetry_x_axis P = Point.mk (-2) 1 := by
  sorry

end symmetry_of_P_l2382_238265


namespace intersection_point_coordinates_l2382_238200

/-- Given a triangle ABC with points F on BC and G on AC, prove that the intersection Q of BG and AF
    can be expressed as a linear combination of A, B, and C. -/
theorem intersection_point_coordinates (A B C F G Q : ℝ × ℝ) : 
  (∃ t : ℝ, F = (1 - t) • B + t • C ∧ t = 1/3) →  -- F lies on BC with BF:FC = 2:1
  (∃ s : ℝ, G = (1 - s) • A + s • C ∧ s = 3/5) →  -- G lies on AC with AG:GC = 3:2
  (∃ u v : ℝ, Q = (1 - u) • B + u • G ∧ Q = (1 - v) • A + v • F) →  -- Q is intersection of BG and AF
  Q = (2/5) • A + (1/3) • B + (4/9) • C := by sorry

end intersection_point_coordinates_l2382_238200


namespace factorize_ax_minus_ay_l2382_238249

theorem factorize_ax_minus_ay (a x y : ℝ) : a * x - a * y = a * (x - y) := by
  sorry

end factorize_ax_minus_ay_l2382_238249


namespace max_diff_six_digit_even_numbers_l2382_238235

/-- A function that checks if a natural number has only even digits -/
def has_only_even_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d % 2 = 0

/-- A function that checks if a natural number has at least one odd digit -/
def has_odd_digit (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ n.digits 10 ∧ d % 2 ≠ 0

/-- The theorem stating the maximum difference between two 6-digit numbers with the given conditions -/
theorem max_diff_six_digit_even_numbers :
  ∃ (a b : ℕ),
    (100000 ≤ a ∧ a < 1000000) ∧
    (100000 ≤ b ∧ b < 1000000) ∧
    has_only_even_digits a ∧
    has_only_even_digits b ∧
    (∀ n : ℕ, a < n ∧ n < b → has_odd_digit n) ∧
    b - a = 111112 ∧
    (∀ a' b' : ℕ,
      (100000 ≤ a' ∧ a' < 1000000) →
      (100000 ≤ b' ∧ b' < 1000000) →
      has_only_even_digits a' →
      has_only_even_digits b' →
      (∀ n : ℕ, a' < n ∧ n < b' → has_odd_digit n) →
      b' - a' ≤ 111112) :=
by sorry

end max_diff_six_digit_even_numbers_l2382_238235


namespace completing_square_result_l2382_238288

theorem completing_square_result (x : ℝ) : x^2 + 4*x - 1 = 0 → (x + 2)^2 = 5 := by
  sorry

end completing_square_result_l2382_238288


namespace family_age_calculation_l2382_238211

theorem family_age_calculation (initial_members : ℕ) (initial_avg_age : ℝ) 
  (current_members : ℕ) (current_avg_age : ℝ) (baby_age : ℝ) : ℝ :=
by
  -- Define the conditions
  have h1 : initial_members = 5 := by sorry
  have h2 : initial_avg_age = 17 := by sorry
  have h3 : current_members = 6 := by sorry
  have h4 : current_avg_age = 17 := by sorry
  have h5 : baby_age = 2 := by sorry

  -- Define the function to calculate the time elapsed
  let time_elapsed := 
    (current_members * current_avg_age - initial_members * initial_avg_age - baby_age) / 
    (initial_members : ℝ)

  -- Prove that the time elapsed is 3 years
  have : time_elapsed = 3 := by sorry

  -- Return the result
  exact time_elapsed

end family_age_calculation_l2382_238211
