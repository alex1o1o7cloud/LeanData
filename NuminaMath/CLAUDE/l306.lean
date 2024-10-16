import Mathlib

namespace NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l306_30678

-- Part 1: System of equations
theorem system_of_equations_solution :
  let x : ℚ := 10
  let y : ℚ := 8/3
  (x / 3 + y / 4 = 4) ∧ (2 * x - 3 * y = 12) := by sorry

-- Part 2: System of inequalities
theorem system_of_inequalities_solution :
  ∀ x : ℚ, -1 ≤ x ∧ x < 3 →
    (x / 3 > (x - 1) / 2) ∧ (3 * (x + 2) ≥ 2 * x + 5) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_system_of_inequalities_solution_l306_30678


namespace NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l306_30679

/-- The shortest distance from a point on the curve y = ln x to the line y = x + 1 is √2 -/
theorem shortest_distance_ln_to_line : 
  ∃ (x y : ℝ), y = Real.log x ∧ 
  (∀ (x' y' : ℝ), y' = Real.log x' → 
    Real.sqrt 2 ≤ Real.sqrt ((x' - x)^2 + (y' - (x + 1))^2)) := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l306_30679


namespace NUMINAMATH_CALUDE_rounded_number_accuracy_l306_30695

/-- Represents a number with its value and accuracy -/
structure ApproximateNumber where
  value : ℝ
  accuracy : ℕ

/-- Defines the concept of "accurate to the hundreds place" -/
def accurate_to_hundreds (n : ApproximateNumber) : Prop :=
  ∃ (k : ℤ), n.value = (k * 100 : ℝ) ∧ 
  ∀ (m : ℤ), |n.value - (m * 100 : ℝ)| ≥ 50

/-- The main theorem to prove -/
theorem rounded_number_accuracy :
  let n := ApproximateNumber.mk (8.80 * 10^4) 2
  accurate_to_hundreds n :=
by sorry

end NUMINAMATH_CALUDE_rounded_number_accuracy_l306_30695


namespace NUMINAMATH_CALUDE_binomial_fraction_is_nat_smallest_k_binomial_fraction_is_nat_smallest_k_property_l306_30649

/-- For any natural number m, 1/(m+1) * binomial(2m, m) is a natural number -/
theorem binomial_fraction_is_nat (m : ℕ) : ∃ (k : ℕ), k = (1 : ℚ) / (m + 1 : ℚ) * (Nat.choose (2 * m) m) := by sorry

/-- For any natural numbers m and n where n ≥ m, 
    (2m+1)/(n+m+1) * binomial(2n, n+m) is a natural number -/
theorem smallest_k_binomial_fraction_is_nat (m n : ℕ) (h : n ≥ m) : 
  ∃ (k : ℕ), k = ((2 * m + 1 : ℚ) / (n + m + 1 : ℚ)) * (Nat.choose (2 * n) (n + m)) := by sorry

/-- 2m+1 is the smallest natural number k such that 
    k/(n+m+1) * binomial(2n, n+m) is a natural number for all n ≥ m -/
theorem smallest_k_property (m : ℕ) : 
  ∀ (k : ℕ), (∀ (n : ℕ), n ≥ m → ∃ (j : ℕ), j = (k : ℚ) / (n + m + 1 : ℚ) * (Nat.choose (2 * n) (n + m))) 
  → k ≥ 2 * m + 1 := by sorry

end NUMINAMATH_CALUDE_binomial_fraction_is_nat_smallest_k_binomial_fraction_is_nat_smallest_k_property_l306_30649


namespace NUMINAMATH_CALUDE_line_equation_through_points_l306_30661

/-- The equation x - y + 1 = 0 represents the line passing through the points (-1, 0) and (0, 1) -/
theorem line_equation_through_points : 
  ∀ (x y : ℝ), (x = -1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) → x - y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_points_l306_30661


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l306_30669

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l306_30669


namespace NUMINAMATH_CALUDE_equation_solution_l306_30639

theorem equation_solution :
  ∃ x : ℝ, (5 * x - 3 * x = 360 + 6 * (x + 4)) ∧ (x = -96) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l306_30639


namespace NUMINAMATH_CALUDE_quadratic_function_range_l306_30638

theorem quadratic_function_range (m : ℝ) : 
  (∀ x : ℝ, -1 < x → x < 0 → x^2 - 4*m*x + 3 > 1) → m > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l306_30638


namespace NUMINAMATH_CALUDE_negative_two_squared_times_negative_one_to_2015_l306_30683

theorem negative_two_squared_times_negative_one_to_2015 : -2^2 * (-1)^2015 = 4 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_squared_times_negative_one_to_2015_l306_30683


namespace NUMINAMATH_CALUDE_boys_without_calculators_l306_30646

/-- Given a class with boys and girls, and information about calculator possession,
    prove the number of boys without calculators. -/
theorem boys_without_calculators
  (total_boys : ℕ)
  (total_with_calculators : ℕ)
  (girls_with_calculators : ℕ)
  (h1 : total_boys = 20)
  (h2 : total_with_calculators = 26)
  (h3 : girls_with_calculators = 13) :
  total_boys - (total_with_calculators - girls_with_calculators) = 7 :=
by
  sorry

#check boys_without_calculators

end NUMINAMATH_CALUDE_boys_without_calculators_l306_30646


namespace NUMINAMATH_CALUDE_reflection_of_point_2_5_l306_30690

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectAcrossXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: The reflection of point (2, 5) across the x-axis is (2, -5) -/
theorem reflection_of_point_2_5 :
  let p := Point.mk 2 5
  reflectAcrossXAxis p = Point.mk 2 (-5) := by
  sorry

end NUMINAMATH_CALUDE_reflection_of_point_2_5_l306_30690


namespace NUMINAMATH_CALUDE_danny_thrice_jane_age_l306_30607

theorem danny_thrice_jane_age (danny_age jane_age : ℕ) (h1 : danny_age = 40) (h2 : jane_age = 26) :
  ∃ x : ℕ, x ≤ jane_age ∧ danny_age - x = 3 * (jane_age - x) ∧ x = 19 :=
by sorry

end NUMINAMATH_CALUDE_danny_thrice_jane_age_l306_30607


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l306_30658

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -5
  let y : ℝ := 2
  second_quadrant x y :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l306_30658


namespace NUMINAMATH_CALUDE_square_binomial_plus_cube_problem_solution_l306_30610

theorem square_binomial_plus_cube (a b : ℕ) : 
  a^2 + 2*a*b + b^2 + b^3 = (a + b)^2 + b^3 := by sorry

theorem problem_solution : 15^2 + 2*(15*5) + 5^2 + 5^3 = 525 := by
  have h1 : 15^2 + 2*(15*5) + 5^2 + 5^3 = (15 + 5)^2 + 5^3 := by
    exact square_binomial_plus_cube 15 5
  have h2 : (15 + 5)^2 = 400 := by norm_num
  have h3 : 5^3 = 125 := by norm_num
  calc
    15^2 + 2*(15*5) + 5^2 + 5^3 = (15 + 5)^2 + 5^3 := h1
    _ = 400 + 125 := by rw [h2, h3]
    _ = 525 := by norm_num

end NUMINAMATH_CALUDE_square_binomial_plus_cube_problem_solution_l306_30610


namespace NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l306_30606

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: The equation x^2 = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_x_squared_eq_one_is_quadratic_l306_30606


namespace NUMINAMATH_CALUDE_total_score_is_26_l306_30689

-- Define the scores for Keith, Larry, and Danny
def keith_score : ℕ := 3
def larry_score : ℕ := 3 * keith_score
def danny_score : ℕ := larry_score + 5

-- Define the total score
def total_score : ℕ := keith_score + larry_score + danny_score

-- Theorem to prove
theorem total_score_is_26 : total_score = 26 := by
  sorry

end NUMINAMATH_CALUDE_total_score_is_26_l306_30689


namespace NUMINAMATH_CALUDE_cube_pyramid_sum_is_34_l306_30660

/-- Represents a three-dimensional shape --/
structure Shape3D where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- A cube --/
def cube : Shape3D :=
  { faces := 6, edges := 12, vertices := 8 }

/-- Adds a pyramid to one face of a given shape --/
def addPyramid (shape : Shape3D) : Shape3D :=
  { faces := shape.faces + 3,  -- One face is covered, 4 new faces added
    edges := shape.edges + 4,  -- 4 new edges from apex to base
    vertices := shape.vertices + 1 }  -- 1 new vertex (apex)

/-- Calculates the sum of faces, edges, and vertices --/
def sumComponents (shape : Shape3D) : Nat :=
  shape.faces + shape.edges + shape.vertices

/-- Theorem: The maximum sum of exterior faces, vertices, and edges
    of a shape formed by adding a pyramid to one face of a cube is 34 --/
theorem cube_pyramid_sum_is_34 :
  sumComponents (addPyramid cube) = 34 := by
  sorry

end NUMINAMATH_CALUDE_cube_pyramid_sum_is_34_l306_30660


namespace NUMINAMATH_CALUDE_cheetah_catches_deer_l306_30691

/-- Proves that a cheetah catches up with a deer in 10 minutes given specific conditions -/
theorem cheetah_catches_deer (deer_speed cheetah_speed : ℝ) 
  (time_difference : ℝ) (catch_up_time : ℝ) : 
  deer_speed = 50 → 
  cheetah_speed = 60 → 
  time_difference = 2 / 60 → 
  (deer_speed * time_difference) / (cheetah_speed - deer_speed) = catch_up_time →
  catch_up_time = 1 / 6 := by
  sorry

#check cheetah_catches_deer

end NUMINAMATH_CALUDE_cheetah_catches_deer_l306_30691


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l306_30601

theorem regular_polygon_sides (n : ℕ) (h1 : n > 2) : 
  (180 * (n - 2) : ℝ) = 156 * n ↔ n = 15 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l306_30601


namespace NUMINAMATH_CALUDE_quadratic_function_range_l306_30619

/-- A quadratic function f(x) = ax^2 + bx satisfying given conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x^2 + b * x

theorem quadratic_function_range (f : ℝ → ℝ) 
  (hf : QuadraticFunction f)
  (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2)
  (h2 : 2 ≤ f 1 ∧ f 1 ≤ 4) :
  5 ≤ f (-2) ∧ f (-2) ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l306_30619


namespace NUMINAMATH_CALUDE_number_difference_l306_30633

theorem number_difference (L S : ℕ) (h1 : L > S) (h2 : L = 1650) (h3 : L = 5 * S + 5) : L - S = 1321 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l306_30633


namespace NUMINAMATH_CALUDE_tuesday_attendance_l306_30629

/-- Proves that given the attendance conditions, 15 people attended class on Tuesday -/
theorem tuesday_attendance : 
  ∀ (T : ℕ), 
  (10 + T + 10 + 10 + 10) / 5 = 11 → 
  T = 15 := by
sorry

end NUMINAMATH_CALUDE_tuesday_attendance_l306_30629


namespace NUMINAMATH_CALUDE_sum_square_plus_sqrt_sum_squares_l306_30620

theorem sum_square_plus_sqrt_sum_squares :
  (5 + 9)^2 + Real.sqrt (5^2 + 9^2) = 196 + Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_sum_square_plus_sqrt_sum_squares_l306_30620


namespace NUMINAMATH_CALUDE_complement_of_A_range_of_a_l306_30647

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 4*x > 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem for the complement of A
theorem complement_of_A : (Aᶜ : Set ℝ) = {x : ℝ | -4 ≤ x ∧ x ≤ 0} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : B a ⊆ Aᶜ ↔ -3 ≤ a ∧ a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_range_of_a_l306_30647


namespace NUMINAMATH_CALUDE_car_distance_proof_l306_30622

/-- Proves that the distance covered by a car is 450 km given the specified conditions -/
theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) : 
  initial_time = 6 →
  speed = 50 →
  (3/2 : ℝ) * initial_time * speed = 450 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l306_30622


namespace NUMINAMATH_CALUDE_games_difference_l306_30696

/-- Represents the number of games Tara's dad attended in a year -/
def games_attended (year : ℕ) : ℕ :=
  if year = 1 then
    (90 * 20) / 100  -- 90% of 20 games in the first year
  else if year = 2 then
    14  -- Given number of games attended in the second year
  else
    0   -- For any other year

/-- The total number of games Tara played each year -/
def total_games : ℕ := 20

/-- Theorem stating the difference in games attended between first and second year -/
theorem games_difference : games_attended 1 - games_attended 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_games_difference_l306_30696


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l306_30687

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n < 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 36 →
  a 3 + a 5 = -6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l306_30687


namespace NUMINAMATH_CALUDE_rhombus_inscribed_circle_area_ratio_l306_30618

theorem rhombus_inscribed_circle_area_ratio (d₁ d₂ : ℝ) (h : d₁ / d₂ = 3 / 4) :
  let r := d₁ * d₂ / (2 * Real.sqrt ((d₁/2)^2 + (d₂/2)^2))
  (d₁ * d₂ / 2) / (π * r^2) = 25 / (6 * π) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_inscribed_circle_area_ratio_l306_30618


namespace NUMINAMATH_CALUDE_symmetric_points_on_number_line_l306_30666

/-- Given points A, B, and C on a number line corresponding to real numbers a, b, and c respectively,
    with A and C symmetric with respect to B, a = √5, and b = 3, prove that c = 6 - √5. -/
theorem symmetric_points_on_number_line (a b c : ℝ) 
  (h_symmetric : b = (a + c) / 2) 
  (h_a : a = Real.sqrt 5) 
  (h_b : b = 3) : 
  c = 6 - Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_on_number_line_l306_30666


namespace NUMINAMATH_CALUDE_larger_ball_radius_l306_30636

theorem larger_ball_radius (r : ℝ) (n : ℕ) (V : ℝ → ℝ) (R : ℝ) :
  r = 2 →
  n = 5 →
  (∀ x, V x = (4 / 3) * Real.pi * x^3) →
  n * V r = V R →
  R = (40 : ℝ)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_larger_ball_radius_l306_30636


namespace NUMINAMATH_CALUDE_bottle_capacity_l306_30676

theorem bottle_capacity (V : ℝ) 
  (h1 : V > 0) 
  (h2 : (0.12 * V - 0.24 + 0.12 / V) / V = 0.03) : 
  V = 2 := by
sorry

end NUMINAMATH_CALUDE_bottle_capacity_l306_30676


namespace NUMINAMATH_CALUDE_min_value_expression_l306_30651

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) ≥ 3 ∧
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) = 3 ↔ y = x * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l306_30651


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l306_30623

def num_chairs : ℕ := 12
def num_students : ℕ := 5
def num_professors : ℕ := 4
def available_positions : ℕ := 6

theorem seating_arrangements_count :
  (Nat.choose available_positions num_professors) * (Nat.factorial num_professors) = 360 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l306_30623


namespace NUMINAMATH_CALUDE_opposite_of_pi_l306_30657

theorem opposite_of_pi : 
  ∃ (x : ℝ), x = -π ∧ x + π = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_pi_l306_30657


namespace NUMINAMATH_CALUDE_triangle_tangent_slopes_sum_l306_30632

theorem triangle_tangent_slopes_sum (A B C : ℝ × ℝ) : 
  let triangle_slopes : List ℝ := [63, 73, 97]
  let curve (x : ℝ) := (x + 3) * (x^2 + 3)
  let tangent_slope (x : ℝ) := 3 * x^2 + 6 * x + 3
  (∀ p ∈ [A, B, C], p.1 ≥ 0 ∧ p.2 ≥ 0) →
  (∀ p ∈ [A, B, C], curve p.1 = p.2) →
  (List.zip [A, B, C] (A :: B :: C :: A :: nil)).all 
    (λ (p, q) => (q.2 - p.2) / (q.1 - p.1) ∈ triangle_slopes) →
  (tangent_slope A.1 + tangent_slope B.1 + tangent_slope C.1 = 237) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_slopes_sum_l306_30632


namespace NUMINAMATH_CALUDE_football_team_progress_l306_30680

def team_progress (loss : Int) (gain : Int) : Int :=
  gain - loss

theorem football_team_progress :
  team_progress 5 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l306_30680


namespace NUMINAMATH_CALUDE_bob_has_winning_strategy_l306_30699

/-- Represents the state of the game board -/
structure GameState where
  value : Nat

/-- Represents a player's move -/
inductive Move
  | Bob (a : Nat)
  | Alice (k : Nat)

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.Bob a => ⟨state.value - a^2⟩
  | Move.Alice k => ⟨state.value^k⟩

/-- Defines a winning sequence of moves for Bob -/
def WinningSequence (initialState : GameState) (moves : List Move) : Prop :=
  moves.foldl applyMove initialState = ⟨0⟩

/-- The main theorem stating Bob's winning strategy exists -/
theorem bob_has_winning_strategy :
  ∀ (initialState : GameState), initialState.value > 0 →
  ∃ (moves : List Move), WinningSequence initialState moves :=
sorry


end NUMINAMATH_CALUDE_bob_has_winning_strategy_l306_30699


namespace NUMINAMATH_CALUDE_rectangle_width_is_six_l306_30697

/-- A rectangle with given properties -/
structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ
  diagonal_squares : ℕ

/-- The properties of our specific rectangle -/
def my_rectangle : Rectangle where
  length := 8
  width := 6
  area := 48
  diagonal_squares := 12

/-- Theorem stating that the width of the rectangle is 6 inches -/
theorem rectangle_width_is_six (r : Rectangle) 
  (h1 : r.length = 8)
  (h2 : r.area = 48)
  (h3 : r.diagonal_squares = 12) : 
  r.width = 6 := by
  sorry

#check rectangle_width_is_six

end NUMINAMATH_CALUDE_rectangle_width_is_six_l306_30697


namespace NUMINAMATH_CALUDE_circle_line_intersection_l306_30674

theorem circle_line_intersection :
  ∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 ∧ x + y = 1 ∧
  ¬(∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 ∧ x + y = 1 ∧ x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l306_30674


namespace NUMINAMATH_CALUDE_cos_2a_given_tan_a_l306_30654

theorem cos_2a_given_tan_a (a : ℝ) (h : Real.tan a = 2) : Real.cos (2 * a) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2a_given_tan_a_l306_30654


namespace NUMINAMATH_CALUDE_not_prime_n4_2n2_3_l306_30694

theorem not_prime_n4_2n2_3 (n : ℤ) : ∃ k : ℤ, n^4 + 2*n^2 + 3 = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_not_prime_n4_2n2_3_l306_30694


namespace NUMINAMATH_CALUDE_m_positive_sufficient_not_necessary_for_hyperbola_l306_30641

-- Define a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  m ≠ 0 ∧ ∃ (x y : ℝ), x^2 / m - y^2 / m = 1

-- State the theorem
theorem m_positive_sufficient_not_necessary_for_hyperbola :
  ∃ (m : ℝ), m ≠ 0 ∧
  (∀ (m : ℝ), m > 0 → is_hyperbola m) ∧
  (∃ (m : ℝ), m < 0 ∧ is_hyperbola m) :=
sorry

end NUMINAMATH_CALUDE_m_positive_sufficient_not_necessary_for_hyperbola_l306_30641


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l306_30684

/-- A sequence a_n is geometric if there exists a constant r such that a_{n+1} = r * a_n for all n. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence a_n is increasing if a_n < a_{n+1} for all n. -/
def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The condition a₁ < a₂ < a₄ for a sequence a_n. -/
def Condition (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 4

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) 
  (h : IsGeometric a) :
  (IsIncreasing a → Condition a) ∧ 
  ¬(Condition a → IsIncreasing a) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l306_30684


namespace NUMINAMATH_CALUDE_henry_total_games_l306_30662

def wins : ℕ := 2
def losses : ℕ := 2
def draws : ℕ := 10

theorem henry_total_games : wins + losses + draws = 14 := by
  sorry

end NUMINAMATH_CALUDE_henry_total_games_l306_30662


namespace NUMINAMATH_CALUDE_factorize_x4_plus_81_l306_30642

theorem factorize_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6*x + 9) * (x^2 - 6*x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factorize_x4_plus_81_l306_30642


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l306_30617

/-- Given a triangle XYZ with specific point ratios, prove that the intersection of certain lines has coordinates (1/3, 2/3, 0) -/
theorem intersection_point_coordinates (X Y Z D E F P : ℝ × ℝ × ℝ) : 
  -- Triangle XYZ exists
  X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X →
  -- D is on YZ extended with ratio 4:1
  ∃ t : ℝ, t > 1 ∧ D = t • Z + (1 - t) • Y ∧ (t - 1) / (5 - t) = 4 →
  -- E is on XZ with ratio 3:2
  ∃ s : ℝ, 0 < s ∧ s < 1 ∧ E = s • X + (1 - s) • Z ∧ s / (1 - s) = 3 / 2 →
  -- F is on XY with ratio 2:1
  ∃ r : ℝ, 0 < r ∧ r < 1 ∧ F = r • X + (1 - r) • Y ∧ r / (1 - r) = 2 →
  -- P is the intersection of BF and YD
  ∃ u v : ℝ, P = u • F + (1 - u) • E ∧ P = v • D + (1 - v) • Y →
  -- Conclusion: P has coordinates (1/3, 2/3, 0) in terms of X, Y, Z
  P = (1/3) • X + (2/3) • Y + 0 • Z :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l306_30617


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l306_30653

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the y-axis -/
def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The original point P -/
def P : Point :=
  { x := -3, y := 5 }

theorem reflection_across_y_axis :
  reflect_y P = { x := 3, y := 5 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l306_30653


namespace NUMINAMATH_CALUDE_abc_problem_l306_30616

def base_6_value (a b : ℕ) : ℕ := a * 6 + b

theorem abc_problem (A B C : ℕ) : 
  (0 < A) → (A ≤ 5) →
  (0 < B) → (B ≤ 5) →
  (0 < C) → (C ≤ 5) →
  (A ≠ B) → (B ≠ C) → (A ≠ C) →
  (base_6_value A B + A = base_6_value B A) →
  (base_6_value A B + B = base_6_value C 1) →
  (A = 5 ∧ B = 5 ∧ C = 1) := by
sorry

end NUMINAMATH_CALUDE_abc_problem_l306_30616


namespace NUMINAMATH_CALUDE_screening_methods_count_l306_30673

/-- The number of units showing the documentary -/
def num_units : ℕ := 4

/-- The number of different screening methods -/
def screening_methods : ℕ := num_units ^ num_units

/-- Theorem stating that the number of different screening methods
    is equal to 4^4 when there are 4 units each showing the film once -/
theorem screening_methods_count :
  screening_methods = 4^4 :=
by sorry

end NUMINAMATH_CALUDE_screening_methods_count_l306_30673


namespace NUMINAMATH_CALUDE_sin_180_degrees_equals_zero_l306_30650

theorem sin_180_degrees_equals_zero : Real.sin (π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_180_degrees_equals_zero_l306_30650


namespace NUMINAMATH_CALUDE_integral_f_minus_one_to_pi_l306_30626

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x - 1 else Real.cos x

theorem integral_f_minus_one_to_pi :
  ∫ x in (-1)..(Real.pi), f x = 1 := by sorry

end NUMINAMATH_CALUDE_integral_f_minus_one_to_pi_l306_30626


namespace NUMINAMATH_CALUDE_scarf_parity_l306_30656

theorem scarf_parity (initial_count : ℕ) (actions : ℕ) (final_count : ℕ) : 
  initial_count % 2 = 0 → 
  actions % 2 = 1 → 
  (∃ (changes : List ℤ), 
    changes.length = actions ∧ 
    (∀ c ∈ changes, c = 1 ∨ c = -1) ∧
    final_count = initial_count + changes.sum) →
  final_count % 2 = 1 :=
by sorry

#check scarf_parity 20 17 10

end NUMINAMATH_CALUDE_scarf_parity_l306_30656


namespace NUMINAMATH_CALUDE_orchid_bushes_after_planting_l306_30603

/-- The number of orchid bushes in a park after planting new ones -/
def total_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The total number of orchid bushes after planting is the sum of initial and planted bushes -/
theorem orchid_bushes_after_planting (initial : ℕ) (planted : ℕ) :
  total_bushes initial planted = initial + planted := by
  sorry

/-- Example with given values -/
example : total_bushes 2 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_orchid_bushes_after_planting_l306_30603


namespace NUMINAMATH_CALUDE_distinct_collections_proof_l306_30663

/-- The number of letters in "COMPUTATION" -/
def word_length : ℕ := 11

/-- The number of vowels in "COMPUTATION" -/
def num_vowels : ℕ := 5

/-- The number of consonants in "COMPUTATION" -/
def num_consonants : ℕ := 6

/-- The number of indistinguishable T's in "COMPUTATION" -/
def num_ts : ℕ := 2

/-- The number of vowels removed -/
def vowels_removed : ℕ := 3

/-- The number of consonants removed -/
def consonants_removed : ℕ := 4

/-- The function to calculate the number of distinct possible collections -/
def distinct_collections : ℕ := 110

theorem distinct_collections_proof :
  distinct_collections = 110 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_proof_l306_30663


namespace NUMINAMATH_CALUDE_largest_b_value_l306_30648

theorem largest_b_value : ∃ b_max : ℝ,
  (∀ b : ℝ, (3 * b + 7) * (b - 2) = 4 * b → b ≤ b_max) ∧
  ((3 * b_max + 7) * (b_max - 2) = 4 * b_max) ∧
  b_max = 81.5205 / 30 :=
by sorry

end NUMINAMATH_CALUDE_largest_b_value_l306_30648


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l306_30671

theorem quadratic_equation_properties (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c = 0 → 
    ((a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
    ((-1 ∈ {x | a * x^2 + b * x + c = 0} ∧ 2 ∈ {x | a * x^2 + b * x + c = 0}) → 2*a + c = 0) ∧
    ((∃ x y, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) → 
      ∃ u v, u ≠ v ∧ a * u^2 + b * u + c = 0 ∧ a * v^2 + b * v + c = 0))) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l306_30671


namespace NUMINAMATH_CALUDE_total_population_l306_30615

/-- The population of New England -/
def new_england_pop : ℕ := 2100000

/-- The population of New York -/
def new_york_pop : ℕ := (2 * new_england_pop) / 3

/-- The population of Pennsylvania -/
def pennsylvania_pop : ℕ := (3 * new_england_pop) / 2

/-- The combined population of Maryland and New Jersey -/
def md_nj_pop : ℕ := new_england_pop + new_england_pop / 5

/-- Theorem stating the total population of all five states -/
theorem total_population : 
  new_york_pop + new_england_pop + pennsylvania_pop + md_nj_pop = 9170000 := by
  sorry

end NUMINAMATH_CALUDE_total_population_l306_30615


namespace NUMINAMATH_CALUDE_lower_bound_x_l306_30640

theorem lower_bound_x (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8)
  (h_diff : ∃ (n : ℤ), n = ⌊y - x⌋ ∧ n = 4) : 3 < x :=
sorry

end NUMINAMATH_CALUDE_lower_bound_x_l306_30640


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_multiples_l306_30600

theorem smallest_number_divisible_by_multiples (n : ℕ) : n = 200 ↔ 
  (∀ m : ℕ, m < n → ¬(15 ∣ (m - 20) ∧ 30 ∣ (m - 20) ∧ 45 ∣ (m - 20) ∧ 60 ∣ (m - 20))) ∧
  (15 ∣ (n - 20) ∧ 30 ∣ (n - 20) ∧ 45 ∣ (n - 20) ∧ 60 ∣ (n - 20)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_multiples_l306_30600


namespace NUMINAMATH_CALUDE_M_inter_N_eq_M_l306_30631

open Set

def M : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 3}
def N : Set ℝ := {x : ℝ | x^2 - 3*x - 4 < 0}

theorem M_inter_N_eq_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_M_inter_N_eq_M_l306_30631


namespace NUMINAMATH_CALUDE_problem_1_l306_30693

theorem problem_1 : (-8) + (-7) - (-6) + 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l306_30693


namespace NUMINAMATH_CALUDE_curve_and_tangent_l306_30655

noncomputable section

-- Define the curve C
def C (k : ℝ) (x y : ℝ) : Prop :=
  x^(2/3) + y^(2/3) = k^(2/3)

-- Define the line segment AB
def AB (k : ℝ) (α β : ℝ) : Prop :=
  α^2 + β^2 = k^2

-- Define the midpoint M of AB
def M (α β : ℝ) : ℝ × ℝ :=
  (α^3 / (α^2 + β^2), β^3 / (α^2 + β^2))

-- State the theorem
theorem curve_and_tangent (k : ℝ) (h : k > 0) :
  ∀ α β : ℝ, AB k α β →
  let (x, y) := M α β
  (C k x y) ∧
  (∃ t : ℝ, t * α + (1 - t) * 0 = x ∧ t * 0 + (1 - t) * β = y) :=
sorry

end

end NUMINAMATH_CALUDE_curve_and_tangent_l306_30655


namespace NUMINAMATH_CALUDE_f_properties_l306_30670

-- Define the function f
def f (a b x : ℝ) : ℝ := |x - a| + |x - b|

-- State the theorem
theorem f_properties (a b : ℝ) (h : -1 < a ∧ a < b) :
  -- Part 1
  (∀ x : ℝ, f 1 2 x ≥ Real.sin x) ∧
  -- Part 2
  {x : ℝ | f a b x < a + b + 2} = {x : ℝ | |2*x - a - b| < a + b + 2} :=
by sorry


end NUMINAMATH_CALUDE_f_properties_l306_30670


namespace NUMINAMATH_CALUDE_salary_percentage_difference_l306_30692

theorem salary_percentage_difference (a b : ℝ) (h : b = 1.25 * a) :
  (b - a) / b = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_salary_percentage_difference_l306_30692


namespace NUMINAMATH_CALUDE_no_solution_equation_l306_30621

theorem no_solution_equation (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  (x + y) / (2 * x) ≠ y / (x + y) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_equation_l306_30621


namespace NUMINAMATH_CALUDE_gcf_of_7_factorial_and_8_factorial_l306_30698

theorem gcf_of_7_factorial_and_8_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_7_factorial_and_8_factorial_l306_30698


namespace NUMINAMATH_CALUDE_average_problem_l306_30612

theorem average_problem (x : ℝ) : (2 + 76 + x) / 3 = 5 → x = -63 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l306_30612


namespace NUMINAMATH_CALUDE_linear_mapping_specification_l306_30624

open Complex

/-- A linear mapping in the complex plane -/
def linearMapping (a w₁ z₁ : ℂ) (z : ℂ) : ℂ :=
  w₁ + a * (z - z₁)

theorem linear_mapping_specification (a w₁ z₁ : ℂ) :
  (linearMapping a w₁ z₁ z₁ = w₁) ∧
  (deriv (linearMapping a w₁ z₁) z₁ = a) :=
by sorry

end NUMINAMATH_CALUDE_linear_mapping_specification_l306_30624


namespace NUMINAMATH_CALUDE_cubic_quintic_inequality_l306_30675

theorem cubic_quintic_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_quintic_inequality_l306_30675


namespace NUMINAMATH_CALUDE_max_value_of_m_l306_30667

theorem max_value_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, a > 0 → b > 0 → m / (3 * a + b) - 3 / a - 1 / b ≤ 0) : 
  m ≤ 16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_m_l306_30667


namespace NUMINAMATH_CALUDE_distinct_x_intercepts_l306_30611

/-- The number of distinct real solutions to the equation (x-5)(x^2 - x - 6) = 0 -/
def num_solutions : ℕ := 3

/-- The equation representing the x-intercepts of the graph -/
def equation (x : ℝ) : ℝ := (x - 5) * (x^2 - x - 6)

theorem distinct_x_intercepts :
  ∃ (s : Finset ℝ), (∀ x ∈ s, equation x = 0) ∧ s.card = num_solutions :=
sorry

end NUMINAMATH_CALUDE_distinct_x_intercepts_l306_30611


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l306_30644

theorem a_gt_one_sufficient_not_necessary (a : ℝ) (h : a ≠ 0) :
  (∀ a, a > 1 → a > 1/a) ∧
  (∃ a, a > 1/a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l306_30644


namespace NUMINAMATH_CALUDE_marbles_selection_theorem_l306_30613

def total_marbles : ℕ := 15
def special_marbles : ℕ := 6
def ordinary_marbles : ℕ := total_marbles - special_marbles

def choose_marbles (n k : ℕ) : ℕ := Nat.choose n k

theorem marbles_selection_theorem :
  (choose_marbles special_marbles 2 * choose_marbles ordinary_marbles 3) +
  (choose_marbles special_marbles 3 * choose_marbles ordinary_marbles 2) +
  (choose_marbles special_marbles 4 * choose_marbles ordinary_marbles 1) +
  (choose_marbles special_marbles 5 * choose_marbles ordinary_marbles 0) = 2121 :=
by sorry

end NUMINAMATH_CALUDE_marbles_selection_theorem_l306_30613


namespace NUMINAMATH_CALUDE_probability_black_not_white_is_three_fifths_l306_30681

structure Bag where
  total : ℕ
  white : ℕ
  black : ℕ
  red : ℕ

def probability_black_given_not_white (b : Bag) : ℚ :=
  b.black / (b.total - b.white)

theorem probability_black_not_white_is_three_fifths (b : Bag) 
  (h1 : b.total = 10)
  (h2 : b.white = 5)
  (h3 : b.black = 3)
  (h4 : b.red = 2) :
  probability_black_given_not_white b = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_black_not_white_is_three_fifths_l306_30681


namespace NUMINAMATH_CALUDE_finite_solutions_cube_sum_l306_30686

theorem finite_solutions_cube_sum (n : ℕ) : 
  Finite {p : ℤ × ℤ | (p.1 ^ 3 + p.2 ^ 3 : ℤ) = n} := by
  sorry

end NUMINAMATH_CALUDE_finite_solutions_cube_sum_l306_30686


namespace NUMINAMATH_CALUDE_equation_solution_l306_30668

theorem equation_solution (a : ℝ) : 
  ((4 - 2) / 2 + a = 4) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l306_30668


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_a_equals_two_sufficient_a_equals_two_not_necessary_a_equals_two_sufficient_not_necessary_l306_30643

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + a = 0) ↔ a ≤ 9/4 :=
by sorry

theorem a_equals_two_sufficient (x : ℝ) :
  x^2 - 3*x + 2 = 0 → ∃ y : ℝ, y^2 - 3*y + 2 = 0 :=
by sorry

theorem a_equals_two_not_necessary :
  ∃ a : ℝ, a ≠ 2 ∧ (∃ x : ℝ, x^2 - 3*x + a = 0) :=
by sorry

theorem a_equals_two_sufficient_not_necessary :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → ∃ y : ℝ, y^2 - 3*y + 2 = 0) ∧
  (∃ a : ℝ, a ≠ 2 ∧ (∃ x : ℝ, x^2 - 3*x + a = 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_a_equals_two_sufficient_a_equals_two_not_necessary_a_equals_two_sufficient_not_necessary_l306_30643


namespace NUMINAMATH_CALUDE_karen_late_start_l306_30682

/-- Proves that Karen starts the race 4 minutes late given the conditions of the car race. -/
theorem karen_late_start (karen_speed tom_speed : ℝ) (tom_distance : ℝ) (karen_win_margin : ℝ) :
  karen_speed = 60 →
  tom_speed = 45 →
  tom_distance = 24 →
  karen_win_margin = 4 →
  (tom_distance / tom_speed * 60 - (tom_distance + karen_win_margin) / karen_speed * 60 : ℝ) = 4 := by
  sorry

#check karen_late_start

end NUMINAMATH_CALUDE_karen_late_start_l306_30682


namespace NUMINAMATH_CALUDE_second_term_value_l306_30685

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem second_term_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 9) →
  a 1 = 1 →
  a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_second_term_value_l306_30685


namespace NUMINAMATH_CALUDE_max_value_theorem_l306_30652

theorem max_value_theorem (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -2)
  (z_ge : z ≥ -4) :
  (∀ a b c : ℝ, a + b + c = 3 → a ≥ -1 → b ≥ -2 → c ≥ -4 →
    Real.sqrt (4*a + 4) + Real.sqrt (4*b + 8) + Real.sqrt (4*c + 16) ≤
    Real.sqrt (4*x + 4) + Real.sqrt (4*y + 8) + Real.sqrt (4*z + 16)) ∧
  Real.sqrt (4*x + 4) + Real.sqrt (4*y + 8) + Real.sqrt (4*z + 16) = 2 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l306_30652


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l306_30659

/-- Given a geometric sequence {a_n} with sum S_n, prove that if S_3 = 39 and a_2 = 9,
    then the common ratio q satisfies q^2 - (10/3)q + 1 = 0 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : S 3 = 39) 
  (h2 : a 2 = 9) 
  (h3 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * q) 
  (h4 : ∀ n : ℕ, n ≥ 1 → S n = a 1 * (1 - q^n) / (1 - q)) 
  : q^2 - (10/3) * q + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l306_30659


namespace NUMINAMATH_CALUDE_function_symmetry_l306_30625

/-- The function f(x) = 3cos(2x + π/6) is symmetric about the point (π/6, 0) -/
theorem function_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = 3 * Real.cos (2 * x + π / 6)) :
  ∀ x, f (π / 3 - x) = f (π / 3 + x) :=
sorry

end NUMINAMATH_CALUDE_function_symmetry_l306_30625


namespace NUMINAMATH_CALUDE_math_team_combinations_l306_30604

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- The number of girls in the math club --/
def num_girls : ℕ := 4

/-- The number of boys in the math club --/
def num_boys : ℕ := 6

/-- The number of girls to be chosen for the team --/
def girls_in_team : ℕ := 2

/-- The number of boys to be chosen for the team --/
def boys_in_team : ℕ := 2

theorem math_team_combinations :
  (choose num_girls girls_in_team) * (choose num_boys boys_in_team) = 90 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l306_30604


namespace NUMINAMATH_CALUDE_quadratic_polynomial_problem_l306_30635

theorem quadratic_polynomial_problem (p : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) →
  (∀ x, (x - 2) * (x + 2) * (x - 9) ∣ (p x)^3 - x) →
  p 14 = -36 / 79 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_problem_l306_30635


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l306_30688

/-- A rhombus with side length 65 and shorter diagonal 72 has a longer diagonal of 108 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) : 
  side = 65 → shorter_diag = 72 → longer_diag = 108 → 
  side^2 = (shorter_diag/2)^2 + (longer_diag/2)^2 := by
  sorry

#check rhombus_longer_diagonal

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l306_30688


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l306_30677

-- Define an increasing function on ℝ
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem increasing_function_inequality (f : ℝ → ℝ) (m : ℝ) 
  (h_increasing : IncreasingFunction f) 
  (h_inequality : f (m + 1) > f (2 * m - 1)) : 
  m < 2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l306_30677


namespace NUMINAMATH_CALUDE_tower_surface_area_l306_30634

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def Cube.volume (c : Cube) : ℝ := c.sideLength ^ 3

/-- Calculates the surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℝ := 6 * c.sideLength ^ 2

/-- Represents the tower of cubes -/
structure CubeTower where
  cubes : List Cube
  isDecreasing : ∀ i j, i < j → (cubes.get i).volume > (cubes.get j).volume
  thirdCubeShifted : True

/-- Calculates the total surface area of the tower -/
def CubeTower.totalSurfaceArea (t : CubeTower) : ℝ :=
  let visibleFaces := [5, 5, 4.5] ++ List.replicate 5 4 ++ [5]
  List.sum (List.zipWith (λ c f => f * c.sideLength ^ 2) t.cubes visibleFaces)

/-- The theorem to be proved -/
theorem tower_surface_area (t : CubeTower) 
  (h1 : t.cubes.length = 9)
  (h2 : List.map Cube.volume t.cubes = [512, 343, 216, 125, 64, 27, 8, 1, 0.125]) :
  t.totalSurfaceArea = 948.25 := by
  sorry

end NUMINAMATH_CALUDE_tower_surface_area_l306_30634


namespace NUMINAMATH_CALUDE_inequality_solution_set_l306_30627

theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_deriv : ∀ x, deriv f x > f x) : 
  {x : ℝ | f x / Real.exp x > f 1 / Real.exp 1} = Set.Ioi 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l306_30627


namespace NUMINAMATH_CALUDE_tan_identity_l306_30672

theorem tan_identity (α : ℝ) (h : Real.tan (α + π / 6) = 2) :
  Real.tan (2 * α + 7 * π / 12) = -1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_identity_l306_30672


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_l306_30608

/-- The number of kids from Lawrence county who go to camp -/
def kids_at_camp : ℕ := 610769

/-- The number of kids from Lawrence county who stay home -/
def kids_at_home : ℕ := 590796

/-- The number of kids from outside the county who attended the camp -/
def outside_kids_at_camp : ℕ := 22

/-- The total number of kids in Lawrence county -/
def total_kids : ℕ := kids_at_camp + kids_at_home

theorem lawrence_county_kids_count :
  total_kids = 1201565 :=
sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_count_l306_30608


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l306_30602

theorem simplify_and_rationalize :
  (Real.sqrt 5 / Real.sqrt 2) * (Real.sqrt 9 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 14) =
  3 * Real.sqrt 420 / 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l306_30602


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l306_30630

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 0 ∧ 
  (∀ k : ℕ, k > m → ¬(∀ i : ℕ, i > 0 → k ∣ (i * (i + 1) * (i + 2) * (i + 3)))) ∧
  (∀ i : ℕ, i > 0 → m ∣ (i * (i + 1) * (i + 2) * (i + 3))) ∧
  m = 12 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l306_30630


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l306_30665

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  q : ℝ      -- The common ratio
  h1 : ∀ n, a (n + 1) = a n * q  -- Definition of geometric sequence
  h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)  -- Sum formula for geometric sequence

/-- The theorem statement -/
theorem geometric_sequence_ratio (seq : GeometricSequence) 
  (h3 : seq.a 3 = 4)
  (h4 : seq.S 3 = 12) :
  seq.q = 1 ∨ seq.q = -1/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l306_30665


namespace NUMINAMATH_CALUDE_prob_king_ace_value_l306_30645

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the event of drawing a King as the first card -/
def first_card_king (d : Deck) : Finset (Fin 52) :=
  d.cards.filter (λ c => c ≤ 4)

/-- Represents the event of drawing an Ace as the second card -/
def second_card_ace (d : Deck) : Finset (Fin 52) :=
  d.cards.filter (λ c => c ≤ 4 ∧ c ≠ c)

/-- The probability of drawing a King first and an Ace second -/
def prob_king_ace (d : Deck) : ℚ :=
  (first_card_king d).card * (second_card_ace d).card / (d.cards.card * (d.cards.card - 1))

theorem prob_king_ace_value (d : Deck) : prob_king_ace d = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_ace_value_l306_30645


namespace NUMINAMATH_CALUDE_house_spacing_l306_30628

/-- Given a city of length 11.5 km and 6 houses to be built at regular intervals
    including both ends, the distance between each house is 2.3 km. -/
theorem house_spacing (city_length : ℝ) (num_houses : ℕ) :
  city_length = 11.5 ∧ num_houses = 6 →
  (city_length / (num_houses - 1 : ℝ)) = 2.3 := by
  sorry

end NUMINAMATH_CALUDE_house_spacing_l306_30628


namespace NUMINAMATH_CALUDE_time_to_fill_cistern_l306_30614

/-- Given a cistern that can be partially filled by a pipe, this theorem proves
    the time required to fill the entire cistern. -/
theorem time_to_fill_cistern (partial_fill_time : ℝ) (partial_fill_fraction : ℝ) 
    (h1 : partial_fill_time = 4)
    (h2 : partial_fill_fraction = 1 / 11) : 
  partial_fill_time / partial_fill_fraction = 44 := by
  sorry

#check time_to_fill_cistern

end NUMINAMATH_CALUDE_time_to_fill_cistern_l306_30614


namespace NUMINAMATH_CALUDE_adjacent_integers_product_l306_30605

theorem adjacent_integers_product (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 740 → (x - 1) * x * (x + 1) = 17550 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_integers_product_l306_30605


namespace NUMINAMATH_CALUDE_choose_20_6_l306_30664

theorem choose_20_6 : Nat.choose 20 6 = 2584 := by
  sorry

end NUMINAMATH_CALUDE_choose_20_6_l306_30664


namespace NUMINAMATH_CALUDE_st_length_is_135_14_l306_30609

/-- Triangle PQR with given side lengths and a parallel line ST containing the incenter -/
structure SpecialTriangle where
  -- Define the triangle PQR
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  -- Define points S and T
  S : ℝ × ℝ
  T : ℝ × ℝ
  -- Conditions
  pq_length : PQ = 13
  pr_length : PR = 14
  qr_length : QR = 15
  s_on_pq : S.1 ≥ 0 ∧ S.1 ≤ PQ
  t_on_pr : T.2 ≥ 0 ∧ T.2 ≤ PR
  st_parallel_qr : sorry -- ST is parallel to QR
  st_contains_incenter : sorry -- ST contains the incenter of PQR

/-- The length of ST in the special triangle -/
def ST_length (triangle : SpecialTriangle) : ℝ := sorry

/-- Theorem stating that ST length is 135/14 -/
theorem st_length_is_135_14 (triangle : SpecialTriangle) : 
  ST_length triangle = 135 / 14 := by sorry

end NUMINAMATH_CALUDE_st_length_is_135_14_l306_30609


namespace NUMINAMATH_CALUDE_defective_product_selection_l306_30637

def total_products : ℕ := 10
def qualified_products : ℕ := 8
def defective_products : ℕ := 2
def products_to_select : ℕ := 3

theorem defective_product_selection :
  (Nat.choose total_products products_to_select - 
   Nat.choose qualified_products products_to_select) = 64 := by
  sorry

end NUMINAMATH_CALUDE_defective_product_selection_l306_30637
