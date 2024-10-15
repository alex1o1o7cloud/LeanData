import Mathlib

namespace NUMINAMATH_CALUDE_tetrahedron_face_area_relation_l2414_241405

/-- Theorem about the relationship between face areas, edges, and angles in a tetrahedron -/
theorem tetrahedron_face_area_relation 
  (S₁ S₂ a b : ℝ) (α φ : ℝ) 
  (h_S₁ : S₁ > 0) (h_S₂ : S₂ > 0) 
  (h_a : a > 0) (h_b : b > 0)
  (h_α : 0 < α ∧ α < π) (h_φ : 0 < φ ∧ φ < π) :
  S₁^2 + S₂^2 - 2*S₁*S₂*(Real.cos α) = (a*b*(Real.sin φ) / 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_face_area_relation_l2414_241405


namespace NUMINAMATH_CALUDE_equation_solution_l2414_241435

theorem equation_solution : ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2414_241435


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2414_241476

/-- A function satisfying the given functional equation. -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y + f (x + y) = x * y

/-- The theorem stating that functions satisfying the equation are of the form x - 1 or x + 1. -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    (∀ x, f x = x - 1) ∨ (∀ x, f x = x + 1) := by
  sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l2414_241476


namespace NUMINAMATH_CALUDE_max_m_condition_l2414_241411

theorem max_m_condition (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ m : ℝ, 4 / a + 1 / b ≥ m / (a + 4 * b)) →
  (∃ m_max : ℝ, ∀ m : ℝ, m ≤ m_max ∧ (m = m_max ↔ b / a = 1 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_condition_l2414_241411


namespace NUMINAMATH_CALUDE_bug_eating_ratio_l2414_241442

theorem bug_eating_ratio (gecko lizard frog toad : ℕ) : 
  gecko = 12 →
  lizard = gecko / 2 →
  toad = frog + frog / 2 →
  gecko + lizard + frog + toad = 63 →
  frog / lizard = 3 := by
  sorry

end NUMINAMATH_CALUDE_bug_eating_ratio_l2414_241442


namespace NUMINAMATH_CALUDE_graph_translation_symmetry_l2414_241451

theorem graph_translation_symmetry (m : Real) : m > 0 →
  (∀ x, 2 * Real.sin (x + m - π / 3) = 2 * Real.sin (-x + m - π / 3)) →
  m = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_graph_translation_symmetry_l2414_241451


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2414_241413

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 4}
def N : Set Nat := {1, 3, 5}

theorem intersection_complement_equality : N ∩ (U \ M) = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2414_241413


namespace NUMINAMATH_CALUDE_parkway_soccer_boys_percentage_l2414_241471

/-- Given the student population data for the fifth grade at Parkway Elementary School,
    prove that 86% of the students playing soccer are boys. -/
theorem parkway_soccer_boys_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (soccer_players : ℕ)
  (girls_not_playing : ℕ)
  (h1 : total_students = 470)
  (h2 : boys = 300)
  (h3 : soccer_players = 250)
  (h4 : girls_not_playing = 135)
  : (boys_playing_soccer : ℚ) / soccer_players * 100 = 86 :=
by sorry

end NUMINAMATH_CALUDE_parkway_soccer_boys_percentage_l2414_241471


namespace NUMINAMATH_CALUDE_conic_is_parabola_l2414_241484

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  |x - 4| = Real.sqrt ((y + 3)^2 + x^2)

-- Theorem stating that the equation describes a parabola
theorem conic_is_parabola :
  ∃ (a b c d : ℝ), a ≠ 0 ∧
  ∀ (x y : ℝ), conic_equation x y ↔ y = a * x^2 + b * x + c * y + d :=
sorry

end NUMINAMATH_CALUDE_conic_is_parabola_l2414_241484


namespace NUMINAMATH_CALUDE_gmat_exam_correct_answers_l2414_241497

theorem gmat_exam_correct_answers 
  (total : ℕ) 
  (first_correct : ℕ) 
  (second_correct : ℕ) 
  (neither_correct : ℕ) 
  (h1 : first_correct = (85 * total) / 100)
  (h2 : second_correct = (80 * total) / 100)
  (h3 : neither_correct = (5 * total) / 100)
  : ((first_correct + second_correct - (total - neither_correct)) * 100) / total = 70 :=
by sorry

end NUMINAMATH_CALUDE_gmat_exam_correct_answers_l2414_241497


namespace NUMINAMATH_CALUDE_average_of_three_l2414_241433

theorem average_of_three (total : ℝ) (avg_all : ℝ) (avg_two : ℝ) :
  total = 5 →
  avg_all = 10 →
  avg_two = 19 →
  (total * avg_all - 2 * avg_two) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_of_three_l2414_241433


namespace NUMINAMATH_CALUDE_ellipse_equation_with_given_parameters_l2414_241416

/-- Standard equation of an ellipse with foci on coordinate axes -/
def standard_ellipse_equation (a b : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 / a^2 + y^2 / b^2 = 1}

/-- Theorem: Standard equation of ellipse with given parameters -/
theorem ellipse_equation_with_given_parameters :
  ∀ (a c : ℝ),
  a^2 = 13 →
  c^2 = 12 →
  ∃ (b : ℝ),
  b^2 = 1 ∧
  (standard_ellipse_equation 13 1 = standard_ellipse_equation 13 1 ∨
   standard_ellipse_equation 1 13 = standard_ellipse_equation 1 13) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_with_given_parameters_l2414_241416


namespace NUMINAMATH_CALUDE_sqrt_10_parts_product_l2414_241474

theorem sqrt_10_parts_product (x y : ℝ) : 
  (x = ⌊Real.sqrt 10⌋) → 
  (y = Real.sqrt 10 - ⌊Real.sqrt 10⌋) → 
  y * (x + Real.sqrt 10) = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_10_parts_product_l2414_241474


namespace NUMINAMATH_CALUDE_probability_red_ball_is_two_fifths_l2414_241493

/-- The probability of drawing a red ball from a bag -/
def probability_red_ball (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  red_balls / total_balls

/-- The total number of balls in the bag -/
def total_balls : ℕ := 5

/-- The number of red balls in the bag -/
def red_balls : ℕ := 2

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

theorem probability_red_ball_is_two_fifths :
  probability_red_ball total_balls red_balls = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_is_two_fifths_l2414_241493


namespace NUMINAMATH_CALUDE_condition_type_l2414_241437

theorem condition_type (A B : Prop) 
  (h1 : ¬B → ¬A) 
  (h2 : ¬(¬A → ¬B)) : 
  (A → B) ∧ ¬(B → A) := by sorry

end NUMINAMATH_CALUDE_condition_type_l2414_241437


namespace NUMINAMATH_CALUDE_short_trees_after_planting_l2414_241404

/-- The number of short trees in the park after planting -/
def total_short_trees (initial_short_trees new_short_trees : ℕ) : ℕ :=
  initial_short_trees + new_short_trees

/-- Theorem stating that the total number of short trees after planting is 98 -/
theorem short_trees_after_planting :
  total_short_trees 41 57 = 98 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_after_planting_l2414_241404


namespace NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l2414_241447

theorem square_difference_of_quadratic_solutions : ∃ α β : ℝ,
  (α ≠ β) ∧ (α^2 = 2*α + 1) ∧ (β^2 = 2*β + 1) ∧ ((α - β)^2 = 8) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_of_quadratic_solutions_l2414_241447


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l2414_241483

/-- The area of a shape composed of a right triangle and 12 congruent squares -/
theorem shaded_area_theorem (hypotenuse : ℝ) (num_squares : ℕ) :
  hypotenuse = 10 →
  num_squares = 12 →
  let leg := hypotenuse / Real.sqrt 2
  let triangle_area := leg * leg / 2
  let square_side := leg / 3
  let square_area := square_side * square_side
  let total_squares_area := num_squares * square_area
  triangle_area + total_squares_area = 275 / 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l2414_241483


namespace NUMINAMATH_CALUDE_second_serving_is_ten_l2414_241430

/-- Represents the number of maggots in various scenarios --/
structure MaggotCounts where
  total : ℕ
  firstServing : ℕ
  firstEaten : ℕ
  secondEaten : ℕ

/-- Calculates the number of maggots in the second serving --/
def secondServing (counts : MaggotCounts) : ℕ :=
  counts.total - counts.firstServing

/-- Theorem stating that the second serving contains 10 maggots --/
theorem second_serving_is_ten (counts : MaggotCounts)
  (h1 : counts.total = 20)
  (h2 : counts.firstServing = 10)
  (h3 : counts.firstEaten = 1)
  (h4 : counts.secondEaten = 3) :
  secondServing counts = 10 := by
  sorry

#eval secondServing { total := 20, firstServing := 10, firstEaten := 1, secondEaten := 3 }

end NUMINAMATH_CALUDE_second_serving_is_ten_l2414_241430


namespace NUMINAMATH_CALUDE_tax_difference_is_twenty_cents_l2414_241410

/-- The price of the item before tax -/
def price : ℝ := 40

/-- The first tax rate as a percentage -/
def tax_rate1 : ℝ := 7.25

/-- The second tax rate as a percentage -/
def tax_rate2 : ℝ := 6.75

/-- Theorem stating the difference between the two tax amounts -/
theorem tax_difference_is_twenty_cents :
  (price * (tax_rate1 / 100)) - (price * (tax_rate2 / 100)) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_tax_difference_is_twenty_cents_l2414_241410


namespace NUMINAMATH_CALUDE_solution_product_l2414_241485

theorem solution_product (a b : ℝ) : 
  (3 * a^2 + 4 * a - 7 = 0) → 
  (3 * b^2 + 4 * b - 7 = 0) → 
  (a - 2) * (b - 2) = 0 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l2414_241485


namespace NUMINAMATH_CALUDE_bread_calories_eq_100_l2414_241440

/-- Represents the number of calories in a serving of peanut butter -/
def peanut_butter_calories : ℕ := 200

/-- Represents the total desired calories for breakfast -/
def total_calories : ℕ := 500

/-- Represents the number of servings of peanut butter used -/
def peanut_butter_servings : ℕ := 2

/-- Calculates the calories in a piece of bread -/
def bread_calories : ℕ := total_calories - (peanut_butter_calories * peanut_butter_servings)

/-- Proves that the calories in a piece of bread equal 100 -/
theorem bread_calories_eq_100 : bread_calories = 100 := by
  sorry

end NUMINAMATH_CALUDE_bread_calories_eq_100_l2414_241440


namespace NUMINAMATH_CALUDE_dogs_can_prevent_wolf_escape_l2414_241409

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square plot -/
structure Square where
  side : ℝ
  center : Point

/-- Represents an animal (wolf or dog) -/
structure Animal where
  position : Point
  speed : ℝ

/-- Represents the game state -/
structure GameState where
  square : Square
  wolf : Animal
  dogs : List Animal

/-- Checks if a point is inside or on the boundary of a square -/
def isInsideSquare (s : Square) (p : Point) : Prop :=
  abs (p.x - s.center.x) ≤ s.side / 2 ∧ abs (p.y - s.center.y) ≤ s.side / 2

/-- Checks if a point is on the boundary of a square -/
def isOnSquareBoundary (s : Square) (p : Point) : Prop :=
  (abs (p.x - s.center.x) = s.side / 2 ∧ abs (p.y - s.center.y) ≤ s.side / 2) ∨
  (abs (p.x - s.center.x) ≤ s.side / 2 ∧ abs (p.y - s.center.y) = s.side / 2)

/-- Theorem: Dogs can prevent the wolf from escaping -/
theorem dogs_can_prevent_wolf_escape (g : GameState) 
  (h1 : g.wolf.position = g.square.center) 
  (h2 : ∀ d ∈ g.dogs, isOnSquareBoundary g.square d.position)
  (h3 : ∀ d ∈ g.dogs, d.speed = 1.5 * g.wolf.speed)
  (h4 : g.dogs.length = 4) :
  ∀ t : ℝ, ∃ strategy : ℝ → List Point, 
    (∀ p ∈ strategy t, isOnSquareBoundary g.square p) ∧ 
    isInsideSquare g.square (g.wolf.position) :=
sorry

end NUMINAMATH_CALUDE_dogs_can_prevent_wolf_escape_l2414_241409


namespace NUMINAMATH_CALUDE_nomogram_relationships_l2414_241443

/-- A structure representing the scales in the nomogram -/
structure Scales where
  X : ℝ
  Y : ℝ
  Z : ℝ
  W : ℝ
  V : ℝ
  U : ℝ
  T : ℝ
  S : ℝ

/-- The theorem stating the relationships between the scales -/
theorem nomogram_relationships (scales : Scales) :
  scales.Z = (scales.X + scales.Y) / 2 ∧
  scales.W = scales.X + scales.Y ∧
  scales.Y = scales.W - scales.X ∧
  scales.V = 2 * (scales.X + scales.Z) ∧
  scales.X + scales.Z + 5 * scales.U = 0 ∧
  scales.T = (6 + scales.Y + scales.Z) / 2 ∧
  scales.Y + scales.Z + 4 * scales.S - 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nomogram_relationships_l2414_241443


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l2414_241425

theorem concentric_circles_radii_difference
  (r R : ℝ) -- r is radius of smaller circle, R is radius of larger circle
  (h_positive : r > 0) -- r is positive
  (h_area_ratio : R^2 / r^2 = 16 / 3) -- area ratio condition
  : R - r = r * (4 * Real.sqrt 3 - 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l2414_241425


namespace NUMINAMATH_CALUDE_largest_m_for_factorization_l2414_241420

theorem largest_m_for_factorization : 
  ∀ m : ℤ, (∃ a b c d : ℤ, 5 * x^2 + m * x + 120 = (a * x + b) * (c * x + d)) → m ≤ 601 :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_factorization_l2414_241420


namespace NUMINAMATH_CALUDE_cindy_envelopes_l2414_241481

theorem cindy_envelopes (friends : ℕ) (envelopes_per_friend : ℕ) (envelopes_left : ℕ) :
  friends = 5 →
  envelopes_per_friend = 3 →
  envelopes_left = 22 →
  friends * envelopes_per_friend + envelopes_left = 37 :=
by sorry

end NUMINAMATH_CALUDE_cindy_envelopes_l2414_241481


namespace NUMINAMATH_CALUDE_problem_statement_l2414_241452

theorem problem_statement (a : ℤ) (h1 : 0 < a) (h2 : a < 13) 
  (h3 : ∃ k : ℤ, 53^2016 + a = 13 * k) : a = 12 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2414_241452


namespace NUMINAMATH_CALUDE_tyler_meal_choices_l2414_241450

def meat_options : ℕ := 4
def vegetable_options : ℕ := 5
def dessert_options : ℕ := 3

def meat_choices : ℕ := 2
def vegetable_choices : ℕ := 3
def dessert_choices : ℕ := 1

theorem tyler_meal_choices :
  (Nat.choose meat_options meat_choices) *
  (Nat.choose vegetable_options vegetable_choices) *
  (Nat.choose dessert_options dessert_choices) = 180 := by
  sorry

end NUMINAMATH_CALUDE_tyler_meal_choices_l2414_241450


namespace NUMINAMATH_CALUDE_polygon_diagonals_with_disconnected_vertex_l2414_241441

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals connected to a single vertex in a polygon with n sides -/
def diagonals_per_vertex (n : ℕ) : ℕ := n - 3

theorem polygon_diagonals_with_disconnected_vertex :
  diagonals 17 - diagonals_per_vertex 17 = 105 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_with_disconnected_vertex_l2414_241441


namespace NUMINAMATH_CALUDE_chess_tournament_l2414_241468

/-- Represents the number of participants from each city --/
structure Participants where
  moscow : ℕ
  saintPetersburg : ℕ
  kazan : ℕ

/-- Represents the number of games played between cities --/
structure Games where
  moscowSaintPetersburg : ℕ
  moscowKazan : ℕ
  saintPetersburgKazan : ℕ

/-- The theorem stating the conditions and the result to be proved --/
theorem chess_tournament (p : Participants) (g : Games) : 
  p.moscow * 9 = p.saintPetersburg * 6 ∧ 
  p.moscow * g.moscowKazan = p.kazan * 8 ∧ 
  p.saintPetersburg * 2 = p.kazan * 6 →
  g.moscowKazan = 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_l2414_241468


namespace NUMINAMATH_CALUDE_gold_checkpoint_problem_l2414_241456

theorem gold_checkpoint_problem (x : ℝ) : 
  x > 0 →
  x - x * (1/2 + 1/3 * 1/2 + 1/4 * 2/3 + 1/5 * 3/4 + 1/6 * 4/5) = 1 →
  x = 1.2 := by
sorry

end NUMINAMATH_CALUDE_gold_checkpoint_problem_l2414_241456


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2414_241454

theorem simplify_square_roots : Real.sqrt 81 - Real.sqrt 49 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2414_241454


namespace NUMINAMATH_CALUDE_shepherd_sheep_equations_correct_l2414_241421

/-- Represents the number of sheep each shepherd has -/
structure ShepherdSheep where
  a : ℤ  -- number of sheep A has
  b : ℤ  -- number of sheep B has

/-- Checks if the given system of equations satisfies the conditions of the problem -/
def satisfies_conditions (s : ShepherdSheep) : Prop :=
  (s.a + 9 = 2 * (s.b - 9)) ∧ (s.b + 9 = s.a - 9)

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem shepherd_sheep_equations_correct :
  ∃ (s : ShepherdSheep), satisfies_conditions s :=
sorry

end NUMINAMATH_CALUDE_shepherd_sheep_equations_correct_l2414_241421


namespace NUMINAMATH_CALUDE_correct_linear_system_l2414_241463

-- Define a structure for a system of two equations
structure EquationSystem where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ

-- Define the four systems of equations
def systemA : EquationSystem := {
  eq1 := fun x y => x + 5*y - 2,
  eq2 := fun x y => x*y - 7
}

def systemB : EquationSystem := {
  eq1 := fun x y => 2*x + 1 - 1,
  eq2 := fun x y => 3*x + 4*y
}

def systemC : EquationSystem := {
  eq1 := fun x y => 3*x^2 - 5*y,
  eq2 := fun x y => x + y - 4
}

def systemD : EquationSystem := {
  eq1 := fun x y => x - 2*y - 8,
  eq2 := fun x y => x + 3*y - 12
}

-- Define a predicate for linear equations with two variables
def isLinearSystem (s : EquationSystem) : Prop :=
  ∃ a b c d e f : ℝ, 
    (∀ x y, s.eq1 x y = a*x + b*y + c) ∧
    (∀ x y, s.eq2 x y = d*x + e*y + f)

-- Theorem statement
theorem correct_linear_system : 
  ¬(isLinearSystem systemA) ∧ 
  ¬(isLinearSystem systemB) ∧ 
  ¬(isLinearSystem systemC) ∧ 
  isLinearSystem systemD := by
  sorry

end NUMINAMATH_CALUDE_correct_linear_system_l2414_241463


namespace NUMINAMATH_CALUDE_book_arrangements_eq_120960_l2414_241417

/-- The number of ways to arrange 4 different math books and 5 different history books on a bookshelf,
    with a math book at both ends and exactly one math book in the middle -/
def book_arrangements : ℕ :=
  let math_books := 4
  let history_books := 5
  let end_arrangements := math_books * (math_books - 1)
  let middle_math_book := math_books - 2
  let remaining_books := (math_books - 3) + history_books
  end_arrangements * middle_math_book * Nat.factorial remaining_books

theorem book_arrangements_eq_120960 : book_arrangements = 120960 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangements_eq_120960_l2414_241417


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2414_241482

theorem trigonometric_equation_solution (x : ℝ) : 
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 ↔ 
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2414_241482


namespace NUMINAMATH_CALUDE_midpoint_triangle_area_ratio_l2414_241402

/-- Given a triangle with area T, the triangle formed by joining the midpoints of its sides has area M = T/4 -/
theorem midpoint_triangle_area_ratio (T : ℝ) (h : T > 0) : 
  ∃ M : ℝ, M = T / 4 ∧ M > 0 := by
sorry

end NUMINAMATH_CALUDE_midpoint_triangle_area_ratio_l2414_241402


namespace NUMINAMATH_CALUDE_special_sequence_2003_l2414_241459

/-- The sequence formed by removing multiples of 3 and 4 (except multiples of 5) from positive integers -/
def special_sequence : ℕ → ℕ := sorry

/-- The 2003rd term of the special sequence -/
def a_2003 : ℕ := special_sequence 2003

/-- Theorem stating that the 2003rd term of the special sequence is 3338 -/
theorem special_sequence_2003 : a_2003 = 3338 := by sorry

end NUMINAMATH_CALUDE_special_sequence_2003_l2414_241459


namespace NUMINAMATH_CALUDE_integral_problem_1_l2414_241426

theorem integral_problem_1 (x : ℝ) (h : x > 0) :
  (deriv (fun x => 4 * (x^(1/2)/2 - x^(1/4) + Real.log (1 + x^(1/4)))) x) = 1 / (x^(1/2) + x^(1/4)) :=
sorry

end NUMINAMATH_CALUDE_integral_problem_1_l2414_241426


namespace NUMINAMATH_CALUDE_congruence_addition_l2414_241487

theorem congruence_addition (a b c d m : ℤ) : 
  a ≡ b [ZMOD m] → c ≡ d [ZMOD m] → (a + c) ≡ (b + d) [ZMOD m] := by
  sorry

end NUMINAMATH_CALUDE_congruence_addition_l2414_241487


namespace NUMINAMATH_CALUDE_overtaking_distance_l2414_241418

/-- Represents a vehicle with a given length -/
structure Vehicle where
  length : ℝ

/-- Represents the overtaking scenario on a highway -/
structure OvertakingScenario where
  sedan : Vehicle
  truck : Vehicle

/-- The additional distance traveled by the sedan during overtaking -/
def additionalDistance (scenario : OvertakingScenario) : ℝ :=
  scenario.sedan.length + scenario.truck.length

theorem overtaking_distance (scenario : OvertakingScenario) :
  additionalDistance scenario = scenario.sedan.length + scenario.truck.length := by
  sorry

end NUMINAMATH_CALUDE_overtaking_distance_l2414_241418


namespace NUMINAMATH_CALUDE_at_least_four_same_prob_l2414_241424

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a specific value on a single die -/
def single_prob : ℚ := 1 / sides

/-- The probability that all five dice show the same number -/
def all_same_prob : ℚ := single_prob ^ (num_dice - 1)

/-- The probability that exactly four dice show the same number and one die shows a different number -/
def four_same_prob : ℚ := num_dice * (single_prob ^ (num_dice - 2)) * ((sides - 1) / sides)

/-- The theorem stating the probability of at least four out of five fair six-sided dice showing the same value -/
theorem at_least_four_same_prob : all_same_prob + four_same_prob = 13 / 648 := by
  sorry

end NUMINAMATH_CALUDE_at_least_four_same_prob_l2414_241424


namespace NUMINAMATH_CALUDE_square_of_97_l2414_241439

theorem square_of_97 : 97^2 = 9409 := by
  sorry

end NUMINAMATH_CALUDE_square_of_97_l2414_241439


namespace NUMINAMATH_CALUDE_student_uniform_cost_l2414_241400

/-- Calculates the total cost for a student's uniforms including discounts, fees, and taxes -/
def uniform_cost (num_uniforms : ℕ) (pants_cost : ℚ) (socks_cost : ℚ) (shoes_cost : ℚ) 
  (uniform_fee : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let shirt_cost := 2 * pants_cost
  let tie_cost := shirt_cost / 5
  let jacket_cost := 3 * shirt_cost
  let uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost + jacket_cost + shoes_cost
  let subtotal := num_uniforms * uniform_cost * (1 - discount_rate) + uniform_fee
  subtotal * (1 + tax_rate)

/-- The total cost for a student buying 5 uniforms is $1117.77 -/
theorem student_uniform_cost : 
  uniform_cost 5 20 3 40 15 (10/100) (6/100) = 1117.77 := by
  sorry

end NUMINAMATH_CALUDE_student_uniform_cost_l2414_241400


namespace NUMINAMATH_CALUDE_fly_ceiling_distance_l2414_241473

def fly_distance (x y z : ℝ) : Prop :=
  x = 2 ∧ y = 7 ∧ x^2 + y^2 + z^2 = 10^2

theorem fly_ceiling_distance :
  ∀ x y z : ℝ, fly_distance x y z → z = Real.sqrt 47 :=
by
  sorry

end NUMINAMATH_CALUDE_fly_ceiling_distance_l2414_241473


namespace NUMINAMATH_CALUDE_roses_distribution_l2414_241412

def distribute_roses (initial : ℕ) (stolen : ℕ) (recipients : ℕ) : ℕ :=
  (initial - stolen) / recipients

theorem roses_distribution (initial : ℕ) (stolen : ℕ) (recipients : ℕ)
  (h1 : initial = 40)
  (h2 : stolen = 4)
  (h3 : recipients = 9)
  : distribute_roses initial stolen recipients = 4 := by
  sorry

end NUMINAMATH_CALUDE_roses_distribution_l2414_241412


namespace NUMINAMATH_CALUDE_percentage_of_sikh_boys_l2414_241490

theorem percentage_of_sikh_boys (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (other : ℕ) :
  total = 400 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  other = 72 →
  (total - (muslim_percent * total).num - (hindu_percent * total).num - other) / total * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_sikh_boys_l2414_241490


namespace NUMINAMATH_CALUDE_unique_valid_cube_configuration_l2414_241408

-- Define a cube face
inductive Face
| White
| Gray
| Mixed

-- Define a cube
structure Cube :=
(front back left right top bottom : Face)

-- Define the conditions
def oppositeFacesValid (c : Cube) : Prop :=
  (c.front = Face.White → c.back = Face.Gray) ∧
  (c.left = Face.White → c.right = Face.Gray) ∧
  (c.top = Face.White → c.bottom = Face.Gray)

def adjacentFacesValid (c : Cube) : Prop :=
  (c.front = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed ∧ 
                          c.left ≠ Face.Mixed ∧ c.right ≠ Face.Mixed) ∧
  (c.back = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed ∧ 
                         c.left ≠ Face.Mixed ∧ c.right ≠ Face.Mixed) ∧
  (c.left = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed) ∧
  (c.right = Face.Mixed → c.top ≠ Face.Mixed ∧ c.bottom ≠ Face.Mixed)

-- Theorem stating the uniqueness of the valid cube configuration
theorem unique_valid_cube_configuration :
  ∃! c : Cube, oppositeFacesValid c ∧ adjacentFacesValid c :=
sorry

end NUMINAMATH_CALUDE_unique_valid_cube_configuration_l2414_241408


namespace NUMINAMATH_CALUDE_investment_sum_l2414_241428

theorem investment_sum (raghu_investment : ℕ) : 
  raghu_investment = 2100 →
  let trishul_investment := raghu_investment - raghu_investment / 10
  let vishal_investment := trishul_investment + trishul_investment / 10
  raghu_investment + trishul_investment + vishal_investment = 6069 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l2414_241428


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l2414_241458

variables (μ σ : ℝ) (ξ : ℝ → ℝ)

-- Define the normal distribution
def normal_dist (μ σ : ℝ) (ξ : ℝ → ℝ) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x, f x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-((x - μ)^2) / (2 * σ^2))

-- Define the probability function
noncomputable def P (A : Set ℝ) : ℝ := sorry

theorem normal_distribution_probability 
  (h1 : normal_dist μ σ ξ)
  (h2 : P {x | ξ x < -1} = 0.3)
  (h3 : P {x | ξ x > 2} = 0.3) :
  P {x | ξ x < 2*μ + 1} = 0.7 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l2414_241458


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_abs_sum_l2414_241461

theorem sqrt_sum_equals_abs_sum (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 6*x + 9) = |x - 2| + |x + 3| := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_abs_sum_l2414_241461


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l2414_241489

/-- Given a line L1 with equation x - 2y = 0 and a point P (-3, -1),
    prove that the line L2 with equation x - 2y + 1 = 0 passes through P
    and is parallel to L1. -/
theorem parallel_line_through_point (L1 L2 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  L1 = {(x, y) | x - 2*y = 0} →
  L2 = {(x, y) | x - 2*y + 1 = 0} →
  P = (-3, -1) →
  (P ∈ L2) ∧ (∀ (x y : ℝ), (x, y) ∈ L1 ↔ ∃ (k : ℝ), (x + k, y + k/2) ∈ L2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l2414_241489


namespace NUMINAMATH_CALUDE_equality_condition_l2414_241427

theorem equality_condition (x y : ℝ) : 
  (x - 9)^2 + (y - 10)^2 + (x - y)^2 = 1/3 → x = 28/3 ∧ y = 29/3 := by
  sorry

end NUMINAMATH_CALUDE_equality_condition_l2414_241427


namespace NUMINAMATH_CALUDE_spelling_bee_contestants_l2414_241488

theorem spelling_bee_contestants (total : ℕ) 
  (h1 : (total : ℝ) * (1 - 0.6) * 0.25 = 30) : total = 300 := by
  sorry

end NUMINAMATH_CALUDE_spelling_bee_contestants_l2414_241488


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2414_241472

theorem triangle_area_proof (a b : ℝ × ℝ) : 
  a = (2, -3) → b = (4, -1) → 
  abs (a.1 * b.2 - a.2 * b.1) / 2 = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l2414_241472


namespace NUMINAMATH_CALUDE_finite_solutions_factorial_difference_l2414_241475

theorem finite_solutions_factorial_difference (u : ℕ+) :
  ∃ (S : Finset (ℕ × ℕ × ℕ)), ∀ (n a b : ℕ),
    n! = u^a - u^b → (n, a, b) ∈ S :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_factorial_difference_l2414_241475


namespace NUMINAMATH_CALUDE_cat_difference_l2414_241486

theorem cat_difference (sheridan_cats garrett_cats : ℕ) 
  (h1 : sheridan_cats = 11) 
  (h2 : garrett_cats = 24) : 
  garrett_cats - sheridan_cats = 13 := by
  sorry

end NUMINAMATH_CALUDE_cat_difference_l2414_241486


namespace NUMINAMATH_CALUDE_problem_solution_l2414_241423

theorem problem_solution (a b : ℚ) (h1 : 7 * a + 3 * b = 0) (h2 : a = 2 * b - 3) :
  5 * b - 4 * a = 141 / 17 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2414_241423


namespace NUMINAMATH_CALUDE_three_card_draw_probability_l2414_241491

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Finset (Fin 52) := Finset.univ

/-- The number of diamonds in a standard deck -/
def numDiamonds : Nat := 13

/-- The number of kings in a standard deck -/
def numKings : Nat := 4

/-- The number of aces in a standard deck -/
def numAces : Nat := 4

/-- The probability of drawing a diamond as the first card, 
    a king as the second card, and an ace as the third card 
    from a standard 52-card deck -/
theorem three_card_draw_probability : 
  (numDiamonds * numKings * numAces : ℚ) / (52 * 51 * 50) = 142 / 66300 := by
  sorry

end NUMINAMATH_CALUDE_three_card_draw_probability_l2414_241491


namespace NUMINAMATH_CALUDE_amit_left_after_three_days_l2414_241467

/-- The number of days Amit can complete the work alone -/
def amit_days : ℝ := 15

/-- The number of days Ananthu can complete the work alone -/
def ananthu_days : ℝ := 30

/-- The total number of days taken to complete the work -/
def total_days : ℝ := 27

/-- The number of days Amit worked before leaving -/
def amit_worked_days : ℝ := 3

theorem amit_left_after_three_days :
  amit_worked_days * (1 / amit_days) + (total_days - amit_worked_days) * (1 / ananthu_days) = 1 :=
sorry

end NUMINAMATH_CALUDE_amit_left_after_three_days_l2414_241467


namespace NUMINAMATH_CALUDE_relative_complement_of_T_in_S_l2414_241446

open Set

def A₁ : Set ℕ := {0, 1}
def A₂ : Set ℕ := {1, 2}
def S : Set ℕ := A₁ ∪ A₂
def T : Set ℕ := A₁ ∩ A₂

theorem relative_complement_of_T_in_S :
  S \ T = {0, 2} := by sorry

end NUMINAMATH_CALUDE_relative_complement_of_T_in_S_l2414_241446


namespace NUMINAMATH_CALUDE_minimum_value_and_range_l2414_241415

def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem minimum_value_and_range (a : ℝ) :
  (∀ x, f x a ≥ 3) ∧ (∃ x, f x a = 3) →
  ((a = 1 ∨ a = 7) ∧
   (a = 1 → ∀ x, f x a ≤ 5 ↔ 0 ≤ x ∧ x ≤ 5) ∧
   (a = 7 → ∀ x, f x a ≤ 5 ↔ 3 ≤ x ∧ x ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_and_range_l2414_241415


namespace NUMINAMATH_CALUDE_geometric_series_product_l2414_241470

theorem geometric_series_product (y : ℝ) : y = 9 ↔ 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/y)^n :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_product_l2414_241470


namespace NUMINAMATH_CALUDE_smallest_with_18_divisors_l2414_241453

/-- Count the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number has exactly 18 positive divisors -/
def has18Divisors (n : ℕ) : Prop := countDivisors n = 18

/-- The smallest positive integer with exactly 18 positive divisors -/
def smallestWith18Divisors : ℕ := 288

theorem smallest_with_18_divisors :
  (has18Divisors smallestWith18Divisors) ∧
  (∀ m : ℕ, m < smallestWith18Divisors → ¬(has18Divisors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_18_divisors_l2414_241453


namespace NUMINAMATH_CALUDE_percent_decrease_proof_l2414_241401

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 75) : 
  (original_price - sale_price) / original_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_proof_l2414_241401


namespace NUMINAMATH_CALUDE_opposite_numbers_l2414_241478

theorem opposite_numbers (x y z : ℝ) (h : 1/x + 1/y + 1/z = 1/(x+y+z)) :
  x + y = 0 ∨ y + z = 0 ∨ x + z = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_l2414_241478


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l2414_241448

theorem arithmetic_square_root_of_16 : ∃ x : ℝ, x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l2414_241448


namespace NUMINAMATH_CALUDE_intersection_A_B_l2414_241419

-- Define set A
def A : Set ℝ := {x : ℝ | x * (x - 4) < 0}

-- Define set B
def B : Set ℝ := {0, 1, 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2414_241419


namespace NUMINAMATH_CALUDE_arrangements_with_restriction_l2414_241496

theorem arrangements_with_restriction (n : ℕ) (h : n = 6) :
  (n - 1) * Nat.factorial (n - 1) = 600 :=
by sorry

end NUMINAMATH_CALUDE_arrangements_with_restriction_l2414_241496


namespace NUMINAMATH_CALUDE_intersection_reciprocals_sum_l2414_241462

/-- Circle C with equation x^2 + y^2 + 2x - 3 = 0 -/
def CircleC (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 3 = 0

/-- Line l passing through the origin with slope k -/
def LineL (k x y : ℝ) : Prop := y = k * x

/-- Theorem: For any line passing through the origin and intersecting CircleC, 
    the sum of reciprocals of x-coordinates of intersection points is 2/3 -/
theorem intersection_reciprocals_sum (k : ℝ) (hk : k ≠ 0) : 
  ∃ x₁ x₂ y₁ y₂ : ℝ, 
    CircleC x₁ y₁ ∧ CircleC x₂ y₂ ∧ 
    LineL k x₁ y₁ ∧ LineL k x₂ y₂ ∧
    x₁ ≠ x₂ ∧ 
    1 / x₁ + 1 / x₂ = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_reciprocals_sum_l2414_241462


namespace NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l2414_241464

/-- Three numbers forming an arithmetic sequence -/
structure ArithmeticSequence :=
  (a : ℝ)
  (d : ℝ)

/-- The sum of three numbers in an arithmetic sequence -/
def sum (seq : ArithmeticSequence) : ℝ :=
  (seq.a - seq.d) + seq.a + (seq.a + seq.d)

/-- The sum of squares of three numbers in an arithmetic sequence -/
def sumOfSquares (seq : ArithmeticSequence) : ℝ :=
  (seq.a - seq.d)^2 + seq.a^2 + (seq.a + seq.d)^2

/-- Theorem: If three numbers form an arithmetic sequence with a sum of 15 and a sum of squares of 83,
    then these numbers are either 3, 5, 7 or 7, 5, 3 -/
theorem arithmetic_sequence_theorem (seq : ArithmeticSequence) :
  sum seq = 15 ∧ sumOfSquares seq = 83 →
  (seq.a = 5 ∧ (seq.d = 2 ∨ seq.d = -2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_theorem_l2414_241464


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2414_241499

theorem arithmetic_mean_of_fractions : 
  let a := 8 / 11
  let b := 5 / 6
  let c := 19 / 22
  b = (a + c) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2414_241499


namespace NUMINAMATH_CALUDE_range_of_m_l2414_241406

def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem range_of_m : 
  (∃ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) → 
  (∀ m : ℝ, (1 < m ∧ m ≤ 2) ∨ m ≥ 3 ↔ (p m ∨ q m) ∧ ¬(p m ∧ q m)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2414_241406


namespace NUMINAMATH_CALUDE_exists_fraction_with_99th_digit_4_l2414_241403

/-- Represents a decimal expansion as a sequence of digits -/
def DecimalExpansion := ℕ → Fin 10

/-- Returns the nth digit after the decimal point in the decimal expansion of a rational number -/
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : Fin 10 := sorry

/-- The decimal expansion of 3/11 -/
def threeElevenths : DecimalExpansion := 
  fun n => if n % 2 = 0 then 2 else 7

theorem exists_fraction_with_99th_digit_4 : 
  ∃ q : ℚ, nthDigitAfterDecimal (q + 3/11) 99 = 4 := by sorry

end NUMINAMATH_CALUDE_exists_fraction_with_99th_digit_4_l2414_241403


namespace NUMINAMATH_CALUDE_qin_jiushao_algorithm_l2414_241455

theorem qin_jiushao_algorithm (n : ℕ) (x : ℝ) (h1 : n = 5) (h2 : x = 2) :
  (Finset.range (n + 1)).sum (fun i => x ^ i) = 63 := by
  sorry

end NUMINAMATH_CALUDE_qin_jiushao_algorithm_l2414_241455


namespace NUMINAMATH_CALUDE_age_difference_proof_l2414_241429

theorem age_difference_proof (ann_age susan_age : ℕ) : 
  ann_age > susan_age →
  ann_age + susan_age = 27 →
  susan_age = 11 →
  ann_age - susan_age = 5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2414_241429


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_l2414_241465

/-- The time taken for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 375) 
  (h2 : train_speed_kmph = 90) 
  (h3 : bridge_length = 1250) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 65 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_l2414_241465


namespace NUMINAMATH_CALUDE_intersection_M_N_l2414_241449

def M : Set ℝ := {x | ∃ t : ℝ, x = 2^(-t)}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2414_241449


namespace NUMINAMATH_CALUDE_sean_played_14_days_l2414_241445

/-- The number of days Sean played cricket -/
def sean_days (sean_minutes_per_day : ℕ) (total_minutes : ℕ) (indira_minutes : ℕ) : ℕ :=
  (total_minutes - indira_minutes) / sean_minutes_per_day

/-- Proof that Sean played cricket for 14 days -/
theorem sean_played_14_days :
  sean_days 50 1512 812 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sean_played_14_days_l2414_241445


namespace NUMINAMATH_CALUDE_count_triples_eq_12_l2414_241457

/-- Least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given conditions -/
def count_triples : ℕ := sorry

theorem count_triples_eq_12 :
  count_triples = 12 := by sorry

end NUMINAMATH_CALUDE_count_triples_eq_12_l2414_241457


namespace NUMINAMATH_CALUDE_relative_errors_equal_l2414_241422

theorem relative_errors_equal (length1 length2 error1 error2 : ℝ) 
  (h1 : length1 = 20)
  (h2 : length2 = 150)
  (h3 : error1 = 0.04)
  (h4 : error2 = 0.3) :
  error1 / length1 = error2 / length2 := by
  sorry

end NUMINAMATH_CALUDE_relative_errors_equal_l2414_241422


namespace NUMINAMATH_CALUDE_complex_square_plus_self_l2414_241414

theorem complex_square_plus_self (z : ℂ) :
  z = -1/2 + (Complex.I * Real.sqrt 3) / 2 →
  z^2 + z = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_square_plus_self_l2414_241414


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2414_241466

theorem quadratic_roots_relation (a b : ℝ) (p : ℝ) : 
  (3 * a^2 + 7 * a + 6 = 0) →
  (3 * b^2 + 7 * b + 6 = 0) →
  (a^3 + b^3 = -p) →
  (p = -35/27) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2414_241466


namespace NUMINAMATH_CALUDE_square_sum_identity_l2414_241498

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l2414_241498


namespace NUMINAMATH_CALUDE_inequality_proof_l2414_241438

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 
  3 / ((a * b * c) ^ (1/3) * (1 + (a * b * c) ^ (1/3))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2414_241438


namespace NUMINAMATH_CALUDE_sets_problem_l2414_241495

-- Define the sets M and N
def M : Set ℝ := {x | (x + 3)^2 ≤ 0}
def N : Set ℝ := {x | x^2 + x - 6 = 0}

-- Define set A as (complement_I M) ∩ N
def A : Set ℝ := (Set.univ \ M) ∩ N

-- Define set B
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 5 - a}

-- Theorem statement
theorem sets_problem :
  (A = {2}) ∧
  ({a : ℝ | B a ∪ A = A} = {a : ℝ | a ≥ 3}) := by
  sorry

end NUMINAMATH_CALUDE_sets_problem_l2414_241495


namespace NUMINAMATH_CALUDE_circus_illumination_theorem_l2414_241477

/-- A convex figure in a plane -/
structure ConvexFigure where
  -- Define properties of a convex figure

/-- The plane -/
structure Plane where
  -- Define properties of a plane

/-- Represents the illumination of the arena -/
def Illumination (n : ℕ) := Fin n → ConvexFigure

/-- The union of a subset of convex figures -/
def UnionOfFigures (i : Illumination n) (s : Finset (Fin n)) : Set Plane :=
  sorry

/-- The entire plane is covered -/
def CoversPlaane (s : Set Plane) : Prop :=
  sorry

/-- Main theorem: For any n ≥ 2, there exists an illumination arrangement satisfying the conditions -/
theorem circus_illumination_theorem (n : ℕ) (h : n ≥ 2) :
  ∃ (i : Illumination n),
    (∀ (k : Fin n), CoversPlaane (UnionOfFigures i (Finset.erase (Finset.univ : Finset (Fin n)) k))) ∧
    (∀ (j k : Fin n), j ≠ k → ¬CoversPlaane (UnionOfFigures i (Finset.erase (Finset.erase (Finset.univ : Finset (Fin n)) j) k))) :=
  sorry

end NUMINAMATH_CALUDE_circus_illumination_theorem_l2414_241477


namespace NUMINAMATH_CALUDE_lucy_fish_count_l2414_241469

/-- The number of fish Lucy wants to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy would have after buying more -/
def total_fish_after : ℕ := 280

/-- The current number of fish in Lucy's aquarium -/
def current_fish : ℕ := total_fish_after - fish_to_buy

theorem lucy_fish_count : current_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l2414_241469


namespace NUMINAMATH_CALUDE_julies_savings_l2414_241407

/-- Represents the initial savings amount in each account -/
def P : ℝ := sorry

/-- Represents the annual interest rate (as a decimal) -/
def r : ℝ := sorry

/-- Theorem stating that given the conditions, Julie's initial total savings was $1000 -/
theorem julies_savings : 
  (P * r * 2 = 100) →  -- Simple interest earned after 2 years
  (P * ((1 + r)^2 - 1) = 105) →  -- Compound interest earned after 2 years
  (2 * P = 1000) :=  -- Total initial savings
by sorry

end NUMINAMATH_CALUDE_julies_savings_l2414_241407


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2414_241432

theorem probability_of_white_ball (p_red_or_white p_yellow_or_white : ℝ) 
  (h1 : p_red_or_white = 0.65)
  (h2 : p_yellow_or_white = 0.6) :
  1 - (1 - p_yellow_or_white) - (1 - p_red_or_white) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l2414_241432


namespace NUMINAMATH_CALUDE_all_students_accounted_for_no_unsatisfactory_grades_l2414_241436

theorem all_students_accounted_for (top_marks : ℚ) (average_marks : ℚ) (good_marks : ℚ)
  (h1 : top_marks = 1 / 6)
  (h2 : average_marks = 1 / 3)
  (h3 : good_marks = 1 / 2) :
  top_marks + average_marks + good_marks = 1 :=
by
  sorry

theorem no_unsatisfactory_grades (total_fraction : ℚ)
  (h : total_fraction = 1) :
  1 - total_fraction = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_all_students_accounted_for_no_unsatisfactory_grades_l2414_241436


namespace NUMINAMATH_CALUDE_always_defined_division_by_two_l2414_241434

theorem always_defined_division_by_two (a : ℝ) : ∃ (x : ℝ), x = a / 2 := by
  sorry

end NUMINAMATH_CALUDE_always_defined_division_by_two_l2414_241434


namespace NUMINAMATH_CALUDE_number_of_bowls_l2414_241460

/-- Given a table with bowls of grapes, prove that there are 16 bowls when:
  - 8 grapes are added to each of 12 bowls
  - The average number of grapes in all bowls increases by 6
-/
theorem number_of_bowls : ℕ → Prop := λ n =>
  -- n is the number of bowls
  -- Define the increase in total grapes
  let total_increase : ℕ := 12 * 8
  -- Define the increase in average
  let avg_increase : ℕ := 6
  -- The theorem: if the total increase divided by the average increase equals n, 
  -- then n is the number of bowls
  total_increase / avg_increase = n

-- The proof (skipped with sorry)
example : number_of_bowls 16 := by sorry

end NUMINAMATH_CALUDE_number_of_bowls_l2414_241460


namespace NUMINAMATH_CALUDE_fraction_simplification_l2414_241492

theorem fraction_simplification (x : ℝ) (h : x = 5) : 
  (x^4 + 12*x^2 + 36) / (x^2 + 6) = 31 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2414_241492


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l2414_241479

theorem min_value_expression (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (h : x * y * z * w = 16) :
  x + 2 * y + 4 * z + 8 * w ≥ 16 :=
sorry

theorem min_value_achieved (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (h : x * y * z * w = 16) :
  ∃ (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a * b * c * d = 16 ∧ a + 2 * b + 4 * c + 8 * d = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l2414_241479


namespace NUMINAMATH_CALUDE_purely_imaginary_modulus_l2414_241494

theorem purely_imaginary_modulus (a : ℝ) :
  (a - 2 : ℂ) + a * I = (0 : ℂ) + (a * I) → Complex.abs (a + I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_modulus_l2414_241494


namespace NUMINAMATH_CALUDE_coefficient_of_term_l2414_241431

theorem coefficient_of_term (x y : ℝ) : 
  ∃ (c : ℝ), -π * x * y^3 / 5 = c * x * y^3 ∧ c = -π / 5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_term_l2414_241431


namespace NUMINAMATH_CALUDE_balloons_kept_winnie_keeps_balloons_l2414_241480

def total_balloons : ℕ := 22 + 44 + 78 + 90
def num_friends : ℕ := 10

theorem balloons_kept (total : ℕ) (friends : ℕ) (h : friends > 0) :
  total % friends = total - friends * (total / friends) :=
by sorry

theorem winnie_keeps_balloons :
  total_balloons % num_friends = 4 :=
by sorry

end NUMINAMATH_CALUDE_balloons_kept_winnie_keeps_balloons_l2414_241480


namespace NUMINAMATH_CALUDE_ship_cats_count_l2414_241444

/-- Represents the passengers on the ship --/
structure ShipPassengers where
  cats : ℕ
  sailors : ℕ
  cook : ℕ
  captain : ℕ

/-- Calculates the total number of heads on the ship --/
def totalHeads (p : ShipPassengers) : ℕ :=
  p.cats + p.sailors + p.cook + p.captain

/-- Calculates the total number of legs on the ship --/
def totalLegs (p : ShipPassengers) : ℕ :=
  4 * p.cats + 2 * p.sailors + 2 * p.cook + p.captain

/-- Theorem stating that given the conditions, the number of cats is 7 --/
theorem ship_cats_count (p : ShipPassengers) 
  (h1 : p.cook = 1) 
  (h2 : p.captain = 1) 
  (h3 : totalHeads p = 16) 
  (h4 : totalLegs p = 45) : 
  p.cats = 7 := by
  sorry


end NUMINAMATH_CALUDE_ship_cats_count_l2414_241444
