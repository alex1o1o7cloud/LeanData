import Mathlib

namespace min_value_of_expression_l2231_223179

theorem min_value_of_expression (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 1 → 
  (∀ x y : ℝ, a * x + b * y = 1 → x^2 + y^2 - 2*x - 2*y = 0 → x = 1 ∧ y = 1) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' + b' = 1 → 1/a' + 2/b' ≥ 3 + 2 * Real.sqrt 2) ∧
  (1/a + 2/b = 3 + 2 * Real.sqrt 2) :=
by sorry

end min_value_of_expression_l2231_223179


namespace negation_of_universal_proposition_l2231_223101

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) := by
  sorry

end negation_of_universal_proposition_l2231_223101


namespace dartboard_probability_l2231_223107

structure Dartboard :=
  (outer_radius : ℝ)
  (inner_radius : ℝ)
  (sections : ℕ)
  (inner_values : Fin 2 → ℕ)
  (outer_values : Fin 2 → ℕ)

def probability_of_score (db : Dartboard) (score : ℕ) (darts : ℕ) : ℚ :=
  sorry

theorem dartboard_probability (db : Dartboard) :
  db.outer_radius = 8 ∧
  db.inner_radius = 4 ∧
  db.sections = 4 ∧
  db.inner_values 0 = 3 ∧
  db.inner_values 1 = 4 ∧
  db.outer_values 0 = 2 ∧
  db.outer_values 1 = 5 →
  probability_of_score db 12 3 = 9 / 1024 :=
sorry

end dartboard_probability_l2231_223107


namespace fraction_equality_l2231_223100

theorem fraction_equality (x y : ℝ) (h : x / y = 4 / 3) : (x - y) / y = 1 / 3 := by
  sorry

end fraction_equality_l2231_223100


namespace x_equals_two_l2231_223174

theorem x_equals_two : ∀ x : ℝ, 3*x - 2*x + x = 3 - 2 + 1 → x = 2 := by
  sorry

end x_equals_two_l2231_223174


namespace negation_equivalence_l2231_223185

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x - 2) / x ≥ 0) ↔ (∃ x : ℝ, x > 0 ∧ 0 ≤ x ∧ x < 2) :=
by sorry

end negation_equivalence_l2231_223185


namespace distance_to_nearest_city_l2231_223188

-- Define the distance to the nearest city
variable (d : ℝ)

-- Define the conditions based on the false statements
def alice_condition : Prop := d < 8
def bob_condition : Prop := d > 7
def charlie_condition : Prop := d > 5
def david_condition : Prop := d ≠ 3

-- Theorem statement
theorem distance_to_nearest_city :
  alice_condition d ∧ bob_condition d ∧ charlie_condition d ∧ david_condition d ↔ d ∈ Set.Ioo 7 8 := by
  sorry

end distance_to_nearest_city_l2231_223188


namespace hyperbola_ellipse_foci_coincide_l2231_223125

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := y^2 / 2 - x^2 / m = 1

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

-- Define the major axis endpoints of the ellipse
def ellipse_major_axis_endpoints : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

-- Define the foci of the hyperbola
def hyperbola_foci (m : ℝ) : Set (ℝ × ℝ) := {(0, 2), (0, -2)}

-- Theorem statement
theorem hyperbola_ellipse_foci_coincide (m : ℝ) :
  (∀ x y, hyperbola x y m → ellipse x y) ∧
  (hyperbola_foci m = ellipse_major_axis_endpoints) →
  m = 2 := by sorry

end hyperbola_ellipse_foci_coincide_l2231_223125


namespace hyperbola_orthogonal_asymptotes_l2231_223103

/-- A hyperbola is defined by its coefficients a, b, c, d, e, f in the equation ax^2 + 2bxy + cy^2 + dx + ey + f = 0 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Asymptotes of a hyperbola are orthogonal -/
def has_orthogonal_asymptotes (h : Hyperbola) : Prop :=
  h.a + h.c = 0

/-- The theorem stating that a hyperbola has orthogonal asymptotes if and only if a + c = 0 -/
theorem hyperbola_orthogonal_asymptotes (h : Hyperbola) :
  has_orthogonal_asymptotes h ↔ h.a + h.c = 0 := by
  sorry

end hyperbola_orthogonal_asymptotes_l2231_223103


namespace platform_length_l2231_223128

/-- Calculates the length of a platform given train specifications -/
theorem platform_length 
  (train_length : ℝ) 
  (time_tree : ℝ) 
  (time_platform : ℝ) 
  (h1 : train_length = 1200)
  (h2 : time_tree = 120)
  (h3 : time_platform = 210) : 
  ∃ platform_length : ℝ, platform_length = 900 ∧ 
  time_platform = (train_length + platform_length) / (train_length / time_tree) :=
by
  sorry

end platform_length_l2231_223128


namespace min_value_at_three_l2231_223197

/-- The quadratic function we want to minimize -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 27

/-- The theorem stating that f(x) is minimized when x = 3 -/
theorem min_value_at_three :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 3 := by sorry

end min_value_at_three_l2231_223197


namespace rational_function_property_l2231_223104

/-- Represents a rational function -/
structure RationalFunction where
  numerator : Polynomial ℝ
  denominator : Polynomial ℝ

/-- Counts the number of holes in the graph of a rational function -/
def count_holes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of vertical asymptotes in the graph of a rational function -/
def count_vertical_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of horizontal asymptotes in the graph of a rational function -/
def count_horizontal_asymptotes (f : RationalFunction) : ℕ := sorry

/-- Counts the number of oblique asymptotes in the graph of a rational function -/
def count_oblique_asymptotes (f : RationalFunction) : ℕ := sorry

/-- The main theorem about the specific rational function -/
theorem rational_function_property : 
  let f : RationalFunction := {
    numerator := Polynomial.monomial 2 1 - Polynomial.monomial 1 5 + Polynomial.monomial 0 6,
    denominator := Polynomial.monomial 3 1 - Polynomial.monomial 2 3 + Polynomial.monomial 1 2
  }
  let p := count_holes f
  let q := count_vertical_asymptotes f
  let r := count_horizontal_asymptotes f
  let s := count_oblique_asymptotes f
  p + 2*q + 3*r + 4*s = 8 := by sorry

end rational_function_property_l2231_223104


namespace player_one_wins_l2231_223178

/-- Represents a player in the stone game -/
inductive Player
| One
| Two

/-- Represents the state of the game -/
structure GameState where
  piles : List Nat
  currentPlayer : Player

/-- Represents a move in the game -/
structure Move where
  pileIndices : List Nat
  stonesRemoved : List Nat

/-- Defines a valid move for Player One -/
def isValidMovePlayerOne (m : Move) : Prop :=
  m.pileIndices.length = 1 ∧ 
  m.stonesRemoved.length = 1 ∧
  (m.stonesRemoved.head! = 1 ∨ m.stonesRemoved.head! = 2 ∨ m.stonesRemoved.head! = 3)

/-- Defines a valid move for Player Two -/
def isValidMovePlayerTwo (m : Move) : Prop :=
  m.pileIndices.length = m.stonesRemoved.length ∧
  m.pileIndices.length ≤ 3 ∧
  m.stonesRemoved.all (· = 1)

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  state.piles.all (· = 0)

/-- Determines if a player has a winning strategy from a given state -/
def hasWinningStrategy (state : GameState) : Prop :=
  sorry

/-- The main theorem: Player One has a winning strategy in the initial game state -/
theorem player_one_wins :
  hasWinningStrategy (GameState.mk (List.replicate 11 10) Player.One) :=
  sorry

end player_one_wins_l2231_223178


namespace max_socks_is_eighteen_l2231_223131

/-- Represents the amount of yarn needed for different items -/
structure YarnAmount where
  sock : ℕ
  hat : ℕ
  sweater : ℕ

/-- Represents the two balls of yarn -/
structure YarnBalls where
  large : YarnAmount
  small : YarnAmount

/-- The given conditions for the yarn balls -/
def yarn_conditions : YarnBalls where
  large := { sock := 3, hat := 5, sweater := 1 }
  small := { sock := 0, hat := 2, sweater := 1/2 }

/-- The maximum number of socks that can be knitted -/
def max_socks : ℕ := 18

/-- Theorem stating that the maximum number of socks that can be knitted is 18 -/
theorem max_socks_is_eighteen (y : YarnBalls) (h : y = yarn_conditions) : 
  (∃ (n : ℕ), n ≤ max_socks ∧ 
    n * y.large.sock ≤ y.large.hat * y.large.sock + y.small.hat * y.large.sock ∧
    ∀ (m : ℕ), m * y.large.sock ≤ y.large.hat * y.large.sock + y.small.hat * y.large.sock → m ≤ n) :=
by sorry

end max_socks_is_eighteen_l2231_223131


namespace parallel_perpendicular_lines_l2231_223123

/-- Given a point A and a line l, find the equations of lines passing through A
    that are parallel and perpendicular to l. -/
theorem parallel_perpendicular_lines
  (A : ℝ × ℝ)
  (l : ℝ → ℝ → Prop)
  (h_A : A = (2, 2))
  (h_l : l = fun x y ↦ 3 * x + 4 * y - 20 = 0) :
  ∃ (l_parallel l_perpendicular : ℝ → ℝ → Prop),
    (∀ x y, l_parallel x y ↔ 3 * x + 4 * y - 14 = 0) ∧
    (∀ x y, l_perpendicular x y ↔ 4 * x - 3 * y - 2 = 0) ∧
    (∀ x y, l_parallel x y → l_parallel A.1 A.2) ∧
    (∀ x y, l_perpendicular x y → l_perpendicular A.1 A.2) ∧
    (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l x₂ y₂ → (y₂ - y₁) * 4 = (x₂ - x₁) * 3) ∧
    (∀ x₁ y₁ x₂ y₂, l_parallel x₁ y₁ → l_parallel x₂ y₂ → (y₂ - y₁) * 4 = (x₂ - x₁) * 3) ∧
    (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l_perpendicular x₂ y₂ → (y₂ - y₁) * 3 = -(x₂ - x₁) * 4) :=
by sorry

end parallel_perpendicular_lines_l2231_223123


namespace square_of_binomial_l2231_223183

theorem square_of_binomial (b : ℝ) : 
  (∃ (a c : ℝ), ∀ x, 16*x^2 + 40*x + b = (a*x + c)^2) → b = 25 := by
  sorry

end square_of_binomial_l2231_223183


namespace largest_expression_l2231_223167

theorem largest_expression : 
  let a := 3 + 2 + 1 + 9
  let b := 3 * 2 + 1 + 9
  let c := 3 + 2 * 1 + 9
  let d := 3 + 2 + 1 / 9
  let e := 3 * 2 / 1 + 9
  b ≥ a ∧ b > c ∧ b > d ∧ b ≥ e := by
sorry

end largest_expression_l2231_223167


namespace remainder_of_product_product_remainder_l2231_223138

theorem remainder_of_product (a b m : ℕ) : (a * b) % m = ((a % m) * (b % m)) % m := by sorry

theorem product_remainder : (2002 * 1493) % 300 = 86 := by
  -- The proof would go here, but we're omitting it as per instructions
  sorry

end remainder_of_product_product_remainder_l2231_223138


namespace parallel_lines_slope_l2231_223159

/-- Two lines in slope-intercept form -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of parallel lines -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem parallel_lines_slope (l1 l2 : Line) : 
  l1 = Line.mk 2 (-1) → 
  l2 = Line.mk a 1 → 
  parallel l1 l2 → 
  a = 2 := by
  sorry

end parallel_lines_slope_l2231_223159


namespace length_of_PQ_l2231_223194

/-- The circle C with center (3, 2) and radius 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 2)^2 = 1}

/-- The line L defined by y = (3/4)x -/
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = (3/4) * p.1}

/-- The intersection points of C and L -/
def intersection := C ∩ L

/-- Assuming the intersection contains exactly two points -/
axiom two_intersection_points : ∃ P Q : ℝ × ℝ, P ≠ Q ∧ intersection = {P, Q}

/-- The length of the line segment PQ -/
noncomputable def PQ_length : ℝ := sorry

/-- The main theorem: The length of PQ is 4√6/5 -/
theorem length_of_PQ : PQ_length = 4 * Real.sqrt 6 / 5 := by sorry

end length_of_PQ_l2231_223194


namespace quadratic_intersects_negative_x_axis_l2231_223189

/-- A quadratic function f(x) = (m-2)x^2 - 4mx + 2m - 6 intersects with the negative x-axis at least once
    if and only if m is in the range 1 ≤ m < 2 or 2 < m < 3. -/
theorem quadratic_intersects_negative_x_axis (m : ℝ) :
  (∃ x : ℝ, x < 0 ∧ (m - 2) * x^2 - 4 * m * x + 2 * m - 6 = 0) ↔
  (1 ≤ m ∧ m < 2) ∨ (2 < m ∧ m < 3) :=
sorry

end quadratic_intersects_negative_x_axis_l2231_223189


namespace car_journey_equation_l2231_223144

theorem car_journey_equation (x : ℝ) (h : x > 0) :
  let distance : ℝ := 120
  let slow_car_speed : ℝ := x
  let fast_car_speed : ℝ := 1.5 * x
  let slow_car_delay : ℝ := 1
  let slow_car_travel_time : ℝ := distance / slow_car_speed - slow_car_delay
  let fast_car_travel_time : ℝ := distance / fast_car_speed
  slow_car_travel_time = fast_car_travel_time :=
by sorry

end car_journey_equation_l2231_223144


namespace prob_sum_seven_l2231_223124

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The set of all possible outcomes when throwing two dice -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_sides) (Finset.range num_sides)

/-- The set of outcomes where the sum is 7 -/
def sum_seven : Finset (ℕ × ℕ) :=
  all_outcomes.filter (λ p => p.1 + p.2 + 2 = 7)

/-- The probability of getting a sum of 7 when throwing two fair dice -/
theorem prob_sum_seven :
  (sum_seven.card : ℚ) / all_outcomes.card = 1 / 6 := by
  sorry


end prob_sum_seven_l2231_223124


namespace complex_arithmetic_equality_l2231_223173

theorem complex_arithmetic_equality : (-1 : ℚ)^2023 + (6 - 5/4) * 4/3 + 4 / (-2/3) = -2/3 := by
  sorry

end complex_arithmetic_equality_l2231_223173


namespace solve_for_m_l2231_223147

theorem solve_for_m : ∃ m : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = -1 → 2*x + m + y = 0) → m = -1 :=
by sorry

end solve_for_m_l2231_223147


namespace bank_teller_coins_l2231_223186

theorem bank_teller_coins (num_5c num_10c : ℕ) (total_value : ℚ) : 
  num_5c = 16 →
  num_10c = 16 →
  total_value = (5 * num_5c + 10 * num_10c) / 100 →
  total_value = 21/5 →
  num_5c + num_10c = 32 := by
sorry

end bank_teller_coins_l2231_223186


namespace smallest_percent_increase_l2231_223116

def question_value : Fin 15 → ℕ
  | 0 => 100
  | 1 => 200
  | 2 => 300
  | 3 => 500
  | 4 => 1000
  | 5 => 2000
  | 6 => 4000
  | 7 => 8000
  | 8 => 16000
  | 9 => 32000
  | 10 => 64000
  | 11 => 125000
  | 12 => 250000
  | 13 => 500000
  | 14 => 1000000

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def options : List (Fin 15 × Fin 15) :=
  [(0, 1), (1, 2), (2, 3), (10, 11), (13, 14)]

theorem smallest_percent_increase :
  ∀ (pair : Fin 15 × Fin 15), pair ∈ options →
    percent_increase (question_value pair.1) (question_value pair.2) ≥
    percent_increase (question_value 1) (question_value 2) :=
by sorry

end smallest_percent_increase_l2231_223116


namespace exists_shape_with_five_faces_l2231_223109

/-- A geometric shape. -/
structure Shape where
  faces : ℕ

/-- A square pyramid is a shape with 5 faces. -/
def SquarePyramid : Shape :=
  { faces := 5 }

/-- There exists a shape with exactly 5 faces. -/
theorem exists_shape_with_five_faces : ∃ (s : Shape), s.faces = 5 := by
  sorry

end exists_shape_with_five_faces_l2231_223109


namespace exists_nonconvergent_sequence_l2231_223170

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Property: The sequence is increasing -/
def IsIncreasing (a : Sequence) : Prop :=
  ∀ n, a n < a (n + 1)

/-- Property: Each term is either the arithmetic mean or the geometric mean of its neighbors -/
def IsMeanOfNeighbors (a : Sequence) : Prop :=
  ∀ n, (2 * a (n + 1) = a n + a (n + 2)) ∨ (a (n + 1) * a (n + 1) = a n * a (n + 2))

/-- Property: The sequence is an arithmetic progression from a certain point -/
def EventuallyArithmetic (a : Sequence) : Prop :=
  ∃ N, ∀ n ≥ N, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Property: The sequence is a geometric progression from a certain point -/
def EventuallyGeometric (a : Sequence) : Prop :=
  ∃ N, ∀ n ≥ N, a (n + 2) * a n = a (n + 1) * a (n + 1)

/-- The main theorem -/
theorem exists_nonconvergent_sequence :
  ∃ (a : Sequence), IsIncreasing a ∧ IsMeanOfNeighbors a ∧
    ¬(EventuallyArithmetic a ∨ EventuallyGeometric a) :=
sorry

end exists_nonconvergent_sequence_l2231_223170


namespace sale_price_calculation_l2231_223117

theorem sale_price_calculation (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := 0.8 * original_price
  let final_price := 0.9 * first_sale_price
  final_price / original_price = 0.72 :=
by sorry

end sale_price_calculation_l2231_223117


namespace complex_equation_sum_l2231_223195

theorem complex_equation_sum (a b : ℝ) (h : (3 * b : ℂ) + (2 * a - 2) * Complex.I = 1 - Complex.I) : 
  a + b = 5/6 := by
  sorry

end complex_equation_sum_l2231_223195


namespace inverse_sum_reciprocals_l2231_223190

theorem inverse_sum_reciprocals (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (y * z + x * z + x * y) := by
  sorry

end inverse_sum_reciprocals_l2231_223190


namespace symmetric_circle_l2231_223191

/-- Given a circle C and a line l, find the equation of the circle symmetric to C with respect to l -/
theorem symmetric_circle (x y : ℝ) : 
  (∀ x y, x^2 + y^2 = 4 → x - y - 3 = 0 → 
    ∃ a b r, (x - a)^2 + (y - b)^2 = r^2 ∧ 
    (x - a)^2 + (y - b)^2 = x^2 + y^2 - 6*x + 6*y + 14) := by
  sorry

end symmetric_circle_l2231_223191


namespace cos_double_angle_special_case_l2231_223135

-- Define the logarithm base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem cos_double_angle_special_case (x : ℝ) (h : lg (Real.cos x) = -1/2) : 
  Real.cos (2 * x) = -4/5 := by
  sorry

end cos_double_angle_special_case_l2231_223135


namespace mod_eq_two_l2231_223193

theorem mod_eq_two (n : ℤ) : 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -5678 [ZMOD 10] → n = 2 := by
  sorry

end mod_eq_two_l2231_223193


namespace parabola_equation_parabola_final_equation_l2231_223154

/-- A parabola with axis of symmetry parallel to the y-axis -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ
  eq_def : ∀ x, eq x = a * (x - 1) * (x - 4)

/-- The parabola passes through points (1,0) and (4,0) -/
def passes_through_points (p : Parabola) : Prop :=
  p.eq 1 = 0 ∧ p.eq 4 = 0

/-- The line y = 2x -/
def line (x : ℝ) : ℝ := 2 * x

/-- The parabola is tangent to the line y = 2x -/
def is_tangent (p : Parabola) : Prop :=
  ∃ x : ℝ, p.eq x = line x ∧ 
  ∀ y : ℝ, y ≠ x → p.eq y ≠ line y

/-- The main theorem -/
theorem parabola_equation (p : Parabola) 
  (h1 : passes_through_points p) 
  (h2 : is_tangent p) : 
  p.a = -2/9 ∨ p.a = -2 := by
  sorry

/-- The final result -/
theorem parabola_final_equation (p : Parabola) 
  (h1 : passes_through_points p) 
  (h2 : is_tangent p) : 
  (∀ x, p.eq x = -2/9 * (x - 1) * (x - 4)) ∨ 
  (∀ x, p.eq x = -2 * (x - 1) * (x - 4)) := by
  sorry

end parabola_equation_parabola_final_equation_l2231_223154


namespace rex_to_total_ratio_l2231_223130

/-- Represents the number of Pokemon cards collected by each person -/
structure PokemonCards where
  nicole : ℕ
  cindy : ℕ
  rex : ℕ

/-- Represents the problem statement and conditions -/
def pokemon_card_problem (cards : PokemonCards) : Prop :=
  cards.nicole = 400 ∧
  cards.cindy = 2 * cards.nicole ∧
  cards.rex = 150 * 4 ∧
  cards.rex < cards.nicole + cards.cindy

/-- Theorem stating the ratio of Rex's cards to Nicole and Cindy's combined total -/
theorem rex_to_total_ratio (cards : PokemonCards) 
  (h : pokemon_card_problem cards) : 
  (cards.rex : ℚ) / (cards.nicole + cards.cindy : ℚ) = 1 / 2 := by
  sorry

end rex_to_total_ratio_l2231_223130


namespace soccer_ball_seams_soccer_ball_seams_eq_90_l2231_223171

/-- The number of seams needed to make a soccer ball with pentagons and hexagons -/
theorem soccer_ball_seams (num_pentagons num_hexagons : ℕ) 
  (h_pentagons : num_pentagons = 12)
  (h_hexagons : num_hexagons = 20) : ℕ :=
  let total_sides := num_pentagons * 5 + num_hexagons * 6
  total_sides / 2

/-- Proof that a soccer ball with 12 pentagons and 20 hexagons requires 90 seams -/
theorem soccer_ball_seams_eq_90 :
  soccer_ball_seams 12 20 rfl rfl = 90 := by
  sorry

end soccer_ball_seams_soccer_ball_seams_eq_90_l2231_223171


namespace correct_student_distribution_l2231_223119

/-- Ticket pricing structure -/
def ticket_price (n : ℕ) : ℕ :=
  if n ≤ 50 then 15
  else if n ≤ 100 then 12
  else 10

/-- Total number of students -/
def total_students : ℕ := 105

/-- Total amount paid -/
def total_paid : ℕ := 1401

/-- Number of students in Class (1) -/
def class_1_students : ℕ := 47

/-- Number of students in Class (2) -/
def class_2_students : ℕ := total_students - class_1_students

/-- Theorem: Given the ticket pricing structure and total amount paid, 
    the number of students in Class (1) is 47 and in Class (2) is 58 -/
theorem correct_student_distribution :
  class_1_students > 40 ∧ 
  class_1_students < 50 ∧
  class_2_students = 58 ∧
  class_1_students + class_2_students = total_students ∧
  ticket_price class_1_students * class_1_students + 
  ticket_price class_2_students * class_2_students = total_paid :=
by sorry

end correct_student_distribution_l2231_223119


namespace negation_equivalence_l2231_223139

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) := by sorry

end negation_equivalence_l2231_223139


namespace integral_sqrt_4_minus_x_squared_plus_x_cubed_l2231_223182

theorem integral_sqrt_4_minus_x_squared_plus_x_cubed : 
  ∫ x in (-1)..1, (Real.sqrt (4 - x^2) + x^3) = Real.sqrt 3 + (2 * Real.pi / 3) := by
  sorry

end integral_sqrt_4_minus_x_squared_plus_x_cubed_l2231_223182


namespace product_of_roots_abs_equation_l2231_223175

theorem product_of_roots_abs_equation (x : ℝ) :
  (∃ a b : ℝ, a ≠ b ∧ 
   (abs a)^2 - 3 * abs a - 10 = 0 ∧
   (abs b)^2 - 3 * abs b - 10 = 0 ∧
   a * b = -25) := by
sorry

end product_of_roots_abs_equation_l2231_223175


namespace square_fence_perimeter_36_posts_l2231_223118

/-- Calculates the perimeter of a square fence given the number of posts, post width, and gap between posts. -/
def square_fence_perimeter (total_posts : ℕ) (post_width_inches : ℕ) (gap_feet : ℕ) : ℕ :=
  let posts_per_side : ℕ := (total_posts - 4) / 4 + 1
  let side_length : ℕ := (posts_per_side - 1) * gap_feet
  4 * side_length

/-- Theorem stating that a square fence with 36 posts, 6-inch wide posts, and 6-foot gaps has a perimeter of 192 feet. -/
theorem square_fence_perimeter_36_posts :
  square_fence_perimeter 36 6 6 = 192 := by
  sorry

end square_fence_perimeter_36_posts_l2231_223118


namespace largest_T_for_inequality_l2231_223163

theorem largest_T_for_inequality (a b c d e : ℝ) 
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e) 
  (h_sum : a + b = c + d + e) : 
  ∃ T : ℝ, T = (5 * Real.sqrt 30 - 2 * Real.sqrt 5) / 6 ∧
  (∀ S : ℝ, (Real.sqrt (a^2 + b^2 + c^2 + d^2 + e^2) ≥ 
    S * (Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d + Real.sqrt e)^2) → 
    S ≤ T) :=
by sorry

end largest_T_for_inequality_l2231_223163


namespace count_figures_l2231_223146

/-- The number of large triangles in Figure 1 -/
def large_triangles : ℕ := 8

/-- The number of medium triangles in Figure 1 -/
def medium_triangles : ℕ := 4

/-- The number of small triangles in Figure 1 -/
def small_triangles : ℕ := 4

/-- The number of small squares (1x1) in Figure 2 -/
def small_squares : ℕ := 20

/-- The number of medium squares (2x2) in Figure 2 -/
def medium_squares : ℕ := 10

/-- The number of large squares (3x3) in Figure 2 -/
def large_squares : ℕ := 4

/-- The number of largest squares (4x4) in Figure 2 -/
def largest_square : ℕ := 1

/-- Theorem stating the total number of triangles in Figure 1 and squares in Figure 2 -/
theorem count_figures :
  (large_triangles + medium_triangles + small_triangles = 16) ∧
  (small_squares + medium_squares + large_squares + largest_square = 35) := by
  sorry

end count_figures_l2231_223146


namespace prob_even_product_two_dice_l2231_223157

/-- A fair six-sided die -/
def SixSidedDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The probability space for rolling two dice -/
def TwoDiceRoll : Finset (ℕ × ℕ) := SixSidedDie.product SixSidedDie

/-- The event of rolling an even product -/
def EvenProduct : Set (ℕ × ℕ) := {p | p.1 * p.2 % 2 = 0}

theorem prob_even_product_two_dice :
  Finset.card (TwoDiceRoll.filter (λ p => p.1 * p.2 % 2 = 0)) / Finset.card TwoDiceRoll = 3 / 4 := by
  sorry

end prob_even_product_two_dice_l2231_223157


namespace sum_of_7th_and_11th_terms_l2231_223143

/-- An arithmetic sequence {a_n} with the sum of its first 17 terms equal to 51 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (a₁ d : ℚ), (∀ n, a n = a₁ + (n - 1) * d) ∧
  (a 1 + a 17) * 17 / 2 = 51

/-- Theorem: In an arithmetic sequence {a_n} where the sum of the first 17 terms is 51,
    the sum of the 7th and 11th terms is 6 -/
theorem sum_of_7th_and_11th_terms
  (a : ℕ → ℚ) (h : arithmetic_sequence a) : a 7 + a 11 = 6 := by
  sorry

end sum_of_7th_and_11th_terms_l2231_223143


namespace min_sum_squares_cube_edges_l2231_223156

/-- Represents a cube with 8 vertices -/
structure Cube :=
  (v1 v2 v3 v4 v5 v6 v7 v8 : ℝ)

/-- Calculates the sum of squares of differences on the edges of a cube -/
def sumOfSquaresOfDifferences (c : Cube) : ℝ :=
  (c.v1 - c.v2)^2 + (c.v1 - c.v3)^2 + (c.v1 - c.v5)^2 +
  (c.v2 - c.v4)^2 + (c.v2 - c.v6)^2 +
  (c.v3 - c.v4)^2 + (c.v3 - c.v7)^2 +
  (c.v4 - c.v8)^2 +
  (c.v5 - c.v6)^2 + (c.v5 - c.v7)^2 +
  (c.v6 - c.v8)^2 +
  (c.v7 - c.v8)^2

/-- Theorem stating the minimum sum of squares of differences on cube edges -/
theorem min_sum_squares_cube_edges :
  ∃ (c : Cube),
    c.v1 = 0 ∧
    c.v8 = 2013 ∧
    c.v2 = 2013/2 ∧
    c.v3 = 2013/2 ∧
    c.v4 = 2013/2 ∧
    c.v5 = 2013/2 ∧
    c.v6 = 2013/2 ∧
    c.v7 = 2013/2 ∧
    sumOfSquaresOfDifferences c = (3 * 2013^2) / 2 ∧
    ∀ (c' : Cube), c'.v1 = 0 ∧ c'.v8 = 2013 →
      sumOfSquaresOfDifferences c' ≥ sumOfSquaresOfDifferences c :=
by
  sorry

end min_sum_squares_cube_edges_l2231_223156


namespace hyperbola_real_axis_length_l2231_223111

/-- Hyperbola struct -/
structure Hyperbola where
  F₁ : ℝ × ℝ  -- First focus
  F₂ : ℝ × ℝ  -- Second focus
  e : ℝ        -- Eccentricity

/-- Point on the hyperbola -/
def Point : Type := ℝ × ℝ

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Area of a triangle given three points -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- Length of the real axis of a hyperbola -/
def realAxisLength (h : Hyperbola) : ℝ := sorry

/-- Theorem statement -/
theorem hyperbola_real_axis_length 
  (C : Hyperbola) 
  (P : Point) 
  (h_eccentricity : C.e = Real.sqrt 5)
  (h_point_on_hyperbola : P ∈ {p : Point | distance p C.F₁ - distance p C.F₂ = realAxisLength C})
  (h_distance_ratio : 2 * distance P C.F₁ = 3 * distance P C.F₂)
  (h_triangle_area : triangleArea P C.F₁ C.F₂ = 2 * Real.sqrt 5) :
  realAxisLength C = Real.sqrt 2 := by sorry

end hyperbola_real_axis_length_l2231_223111


namespace arithmetic_sequence_sum_l2231_223114

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum1 : a 1 + a 4 + a 7 = 39) 
  (h_sum2 : a 2 + a 5 + a 8 = 33) : 
  a 5 + a 8 + a 11 = 15 := by
sorry

end arithmetic_sequence_sum_l2231_223114


namespace bob_question_creation_l2231_223169

theorem bob_question_creation (x : ℕ) : 
  x + 2*x + 4*x = 91 → x = 13 := by
  sorry

end bob_question_creation_l2231_223169


namespace solution_system_l2231_223192

theorem solution_system (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 110) : 
  x^2 + y^2 = 8044 / 169 := by
  sorry

end solution_system_l2231_223192


namespace smallest_n_value_smallest_n_is_99000_l2231_223166

/-- The number of ordered quadruplets satisfying the conditions -/
def num_quadruplets : ℕ := 91000

/-- The given GCD value for all quadruplets -/
def given_gcd : ℕ := 55

/-- 
Proposition: The smallest positive integer n satisfying the following conditions is 99000:
1. There exist exactly 91000 ordered quadruplets of positive integers (a, b, c, d)
2. For each quadruplet, gcd(a, b, c, d) = 55
3. For each quadruplet, lcm(a, b, c, d) = n
-/
theorem smallest_n_value (n : ℕ) : 
  (∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)), 
    S.card = num_quadruplets ∧ 
    ∀ (a b c d : ℕ), (a, b, c, d) ∈ S → 
      Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = given_gcd ∧
      Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = n) →
  n ≥ 99000 :=
by sorry

/-- The smallest value of n satisfying the conditions is indeed 99000 -/
theorem smallest_n_is_99000 : 
  ∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)), 
    S.card = num_quadruplets ∧ 
    ∀ (a b c d : ℕ), (a, b, c, d) ∈ S → 
      Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = given_gcd ∧
      Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = 99000 :=
by sorry

end smallest_n_value_smallest_n_is_99000_l2231_223166


namespace range_of_m_l2231_223150

/-- Given an increasing function f on ℝ and the condition f(m^2) > f(-m),
    the range of m is (-∞, -1) ∪ (0, +∞) -/
theorem range_of_m (f : ℝ → ℝ) (h_incr : Monotone f) (m : ℝ) (h_cond : f (m^2) > f (-m)) :
  m ∈ Set.Iio (-1) ∪ Set.Ioi 0 :=
sorry

end range_of_m_l2231_223150


namespace pet_store_puppies_l2231_223153

theorem pet_store_puppies 
  (bought : ℝ) 
  (puppies_per_cage : ℝ) 
  (cages_used : ℝ) 
  (h1 : bought = 3.0)
  (h2 : puppies_per_cage = 5.0)
  (h3 : cages_used = 4.2) :
  cages_used * puppies_per_cage - bought = 18.0 := by
sorry

end pet_store_puppies_l2231_223153


namespace opposite_of_negative_six_l2231_223196

theorem opposite_of_negative_six (m : ℤ) : (m + (-6) = 0) → m = 6 := by
  sorry

end opposite_of_negative_six_l2231_223196


namespace grass_seed_min_cost_l2231_223133

/-- Represents a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  price : Rat

/-- Finds the minimum cost to purchase grass seed given constraints -/
def minCostGrassSeed (bags : List GrassSeedBag) (minWeight maxWeight : Nat) : Rat :=
  sorry

/-- The problem statement -/
theorem grass_seed_min_cost :
  let bags : List GrassSeedBag := [
    { weight := 5, price := 1382/100 },
    { weight := 10, price := 2043/100 },
    { weight := 25, price := 3225/100 }
  ]
  let minWeight : Nat := 65
  let maxWeight : Nat := 80
  minCostGrassSeed bags minWeight maxWeight = 9875/100 := by
  sorry

end grass_seed_min_cost_l2231_223133


namespace constant_product_l2231_223181

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 4}

-- Define the symmetry axis
def symmetry_axis : Set (ℝ × ℝ) :=
  {p | 2 * p.1 - 3 * p.2 + 6 = 0}

-- Define point P
def P : ℝ × ℝ := (1, 0)

-- Define line m
def line_m : Set (ℝ × ℝ) :=
  {p | p.1 + 2 * p.2 + 2 = 0}

-- Define the theorem
theorem constant_product :
  ∀ l : Set (ℝ × ℝ),
  (P ∈ l) →
  (∃ A B : ℝ × ℝ,
    A ∈ circle_C ∧
    B ∈ line_m ∧
    A ∈ l ∧
    B ∈ l ∧
    (∃ C : ℝ × ℝ, C ∈ circle_C ∧ C ∈ l ∧ A ≠ C ∧ 
      A = ((C.1 + A.1) / 2, (C.2 + A.2) / 2)) →
    (Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) * 
     Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) = 6)) :=
sorry


end constant_product_l2231_223181


namespace percentage_problem_l2231_223176

theorem percentage_problem (p : ℝ) (x : ℝ) : 
  (p / 100) * x = 100 → 
  (120 / 100) * x = 600 → 
  p = 20 := by
sorry

end percentage_problem_l2231_223176


namespace trigonometric_signs_l2231_223141

theorem trigonometric_signs :
  let expr1 := Real.sin (1125 * π / 180)
  let expr2 := Real.tan (37 * π / 12) * Real.sin (37 * π / 12)
  let expr3 := Real.sin 4 / Real.tan 4
  let expr4 := Real.sin (|(-1)|)
  (expr1 > 0) ∧ (expr2 < 0) ∧ (expr3 < 0) ∧ (expr4 > 0) := by sorry

end trigonometric_signs_l2231_223141


namespace rotten_oranges_count_l2231_223177

/-- The number of rotten oranges on a truck --/
def rotten_oranges : ℕ :=
  let total_oranges : ℕ := 10 * 30
  let oranges_for_juice : ℕ := 30
  let oranges_sold : ℕ := 220
  total_oranges - oranges_for_juice - oranges_sold

/-- Theorem stating that the number of rotten oranges is 50 --/
theorem rotten_oranges_count : rotten_oranges = 50 := by
  sorry

end rotten_oranges_count_l2231_223177


namespace range_of_a_for_monotonic_f_l2231_223161

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else a^(x - 1)

-- State the theorem
theorem range_of_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Ioo 2 4 := by sorry

end range_of_a_for_monotonic_f_l2231_223161


namespace smallest_y_theorem_l2231_223155

def x : ℕ := 6 * 18 * 42

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def smallest_y_for_perfect_cube : ℕ := 441

theorem smallest_y_theorem :
  (∀ y : ℕ, y < smallest_y_for_perfect_cube → ¬(is_perfect_cube (x * y))) ∧
  (is_perfect_cube (x * smallest_y_for_perfect_cube)) := by sorry

end smallest_y_theorem_l2231_223155


namespace triangle_area_formula_l2231_223105

variable (m₁ m₂ m₃ : ℝ)
variable (u u₁ u₂ u₃ t : ℝ)

def is_altitude (m : ℝ) : Prop := m > 0

theorem triangle_area_formula 
  (h₁ : is_altitude m₁)
  (h₂ : is_altitude m₂)
  (h₃ : is_altitude m₃)
  (hu : u = 1/2 * (1/m₁ + 1/m₂ + 1/m₃))
  (hu₁ : u₁ = u - 1/m₁)
  (hu₂ : u₂ = u - 1/m₂)
  (hu₃ : u₃ = u - 1/m₃)
  : t = 4 * Real.sqrt (u * u₁ * u₂ * u₃) :=
sorry

end triangle_area_formula_l2231_223105


namespace fraction_inequality_solution_set_l2231_223199

theorem fraction_inequality_solution_set (x : ℝ) (h : x ≠ 1) :
  x / (x - 1) < 0 ↔ 0 < x ∧ x < 1 := by
  sorry

end fraction_inequality_solution_set_l2231_223199


namespace transformed_curve_equation_l2231_223134

/-- Given a curve y = (1/3)cos(2x) and a scaling transformation x' = 2x, y' = 3y,
    the transformed curve is y' = cos(x'). -/
theorem transformed_curve_equation (x y x' y' : ℝ) :
  y = (1/3) * Real.cos (2 * x) →
  x' = 2 * x →
  y' = 3 * y →
  y' = Real.cos x' := by
  sorry

end transformed_curve_equation_l2231_223134


namespace louisa_travel_problem_l2231_223142

/-- Louisa's vacation travel problem -/
theorem louisa_travel_problem (first_day_distance : ℝ) (speed : ℝ) (time_difference : ℝ) 
  (h1 : first_day_distance = 100)
  (h2 : speed = 25)
  (h3 : time_difference = 3)
  : ∃ (second_day_distance : ℝ), second_day_distance = 175 := by
  sorry

end louisa_travel_problem_l2231_223142


namespace polynomial_expansion_equality_l2231_223145

theorem polynomial_expansion_equality (x : ℝ) : 
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 2)*(x + 3) = 
  6*x^3 - 16*x^2 + 43*x - 70 := by
  sorry

end polynomial_expansion_equality_l2231_223145


namespace junior_high_ten_total_games_l2231_223198

/-- Represents a basketball conference -/
structure BasketballConference where
  num_teams : ℕ
  intra_conference_games : ℕ
  non_conference_games : ℕ

/-- Calculates the total number of games in a season for a given basketball conference -/
def total_games (conf : BasketballConference) : ℕ :=
  (conf.num_teams.choose 2 * conf.intra_conference_games) + (conf.num_teams * conf.non_conference_games)

/-- The Junior High Ten conference -/
def junior_high_ten : BasketballConference :=
  { num_teams := 10
  , intra_conference_games := 3
  , non_conference_games := 5 }

theorem junior_high_ten_total_games :
  total_games junior_high_ten = 185 := by
  sorry


end junior_high_ten_total_games_l2231_223198


namespace diophantine_approximation_2005_l2231_223102

theorem diophantine_approximation_2005 (m n : ℕ+) : 
  |n * Real.sqrt 2005 - m| > (1 : ℝ) / (90 * n) := by sorry

end diophantine_approximation_2005_l2231_223102


namespace max_subgrid_sum_l2231_223148

/-- Represents a 5x5 grid filled with integers -/
def Grid := Fin 5 → Fin 5 → ℕ

/-- Checks if all numbers in the grid are unique and between 1 and 25 -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j, 1 ≤ g i j ∧ g i j ≤ 25 ∧
  ∀ i' j', (i ≠ i' ∨ j ≠ j') → g i j ≠ g i' j'

/-- Calculates the sum of a 2x2 subgrid starting at (i, j) -/
def subgrid_sum (g : Grid) (i j : Fin 4) : ℕ :=
  g i j + g i (j+1) + g (i+1) j + g (i+1) (j+1)

/-- The main theorem -/
theorem max_subgrid_sum (g : Grid) (h : valid_grid g) :
  (∀ i j : Fin 4, 45 ≤ subgrid_sum g i j) ∧
  ¬∃ N > 45, ∀ i j : Fin 4, N ≤ subgrid_sum g i j :=
sorry

end max_subgrid_sum_l2231_223148


namespace min_sum_squares_l2231_223164

theorem min_sum_squares (x y z : ℝ) (h : x - 2*y - 3*z = 4) :
  ∃ (m : ℝ), m = 8/7 ∧ ∀ (a b c : ℝ), a - 2*b - 3*c = 4 → a^2 + b^2 + c^2 ≥ m := by
  sorry

end min_sum_squares_l2231_223164


namespace base_to_lateral_area_ratio_l2231_223168

/-- Represents a cone where the height is equal to the diameter of its circular base -/
structure SpecialCone where
  r : ℝ  -- radius of the base
  h : ℝ  -- height of the cone
  h_eq_diam : h = 2 * r  -- condition that height equals diameter

/-- The ratio of base area to lateral area for a SpecialCone is 1:√5 -/
theorem base_to_lateral_area_ratio (cone : SpecialCone) :
  (π * cone.r^2) / (π * cone.r * Real.sqrt (cone.h^2 + cone.r^2)) = 1 / Real.sqrt 5 := by
  sorry

end base_to_lateral_area_ratio_l2231_223168


namespace floor_abs_negative_real_l2231_223108

theorem floor_abs_negative_real : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end floor_abs_negative_real_l2231_223108


namespace triangle_angle_proof_l2231_223132

theorem triangle_angle_proof (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →        -- sum of angles in a triangle
  c = 45 →                 -- one angle is 45°
  b = 2 * a →              -- ratio of other two angles is 2:1
  a = 45 :=                -- prove that the smaller angle is also 45°
by sorry

end triangle_angle_proof_l2231_223132


namespace equal_numbers_exist_l2231_223106

/-- Triangle inequality for three sides --/
def is_triangle (x y z : ℝ) : Prop :=
  x ≤ y + z ∧ y ≤ x + z ∧ z ≤ x + y

/-- Main theorem --/
theorem equal_numbers_exist (a b c : ℝ) :
  (∀ n : ℕ, is_triangle (a^n) (b^n) (c^n)) →
  (a = b ∨ b = c ∨ a = c) :=
by sorry

end equal_numbers_exist_l2231_223106


namespace f_has_root_iff_f_ln_b_gt_inv_b_l2231_223140

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Theorem 1: f has a root iff 0 < a ≤ 1/e
theorem f_has_root_iff (a : ℝ) (h : a > 0) :
  (∃ x > 0, f a x = 0) ↔ a ≤ (Real.exp 1)⁻¹ :=
sorry

-- Theorem 2: When a ≥ 2/e and b > 1, f(ln b) > 1/b
theorem f_ln_b_gt_inv_b (a b : ℝ) (ha : a ≥ 2 / Real.exp 1) (hb : b > 1) :
  f a (Real.log b) > b⁻¹ :=
sorry

end f_has_root_iff_f_ln_b_gt_inv_b_l2231_223140


namespace deployment_plans_count_l2231_223137

def number_of_volunteers : ℕ := 6
def number_of_positions : ℕ := 4
def number_of_restricted_volunteers : ℕ := 2

theorem deployment_plans_count :
  (number_of_volunteers.choose number_of_positions * number_of_positions.factorial) -
  (number_of_restricted_volunteers * ((number_of_volunteers - 1).choose (number_of_positions - 1) * (number_of_positions - 1).factorial)) = 240 :=
sorry

end deployment_plans_count_l2231_223137


namespace sin_2x_derivative_l2231_223149

open Real

theorem sin_2x_derivative (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x)
  (deriv f) x = 2 * Real.cos (2 * x) := by
  sorry

end sin_2x_derivative_l2231_223149


namespace meaningful_range_l2231_223160

def is_meaningful (x : ℝ) : Prop :=
  x + 3 ≥ 0 ∧ x ≠ 1

theorem meaningful_range :
  ∀ x : ℝ, is_meaningful x ↔ x ≥ -3 ∧ x ≠ 1 := by sorry

end meaningful_range_l2231_223160


namespace line_equation_through_point_with_slope_l2231_223151

/-- The equation of a line passing through (0, 2) with slope 2 is y = 2x + 2 -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  y - 2 = 2 * (x - 0) → y = 2 * x + 2 := by
  sorry

end line_equation_through_point_with_slope_l2231_223151


namespace sequence_a_property_l2231_223165

def sequence_a (n : ℕ) : ℚ := 2 * n^2 - n

theorem sequence_a_property :
  (sequence_a 1 = 1) ∧
  (∀ n m : ℕ, n ≠ 0 → m ≠ 0 → sequence_a m / m - sequence_a n / n = 2 * (m - n)) :=
by sorry

end sequence_a_property_l2231_223165


namespace solution_relationship_l2231_223180

theorem solution_relationship (x y : ℝ) : 
  (2 * x + y = 7) → (x - y = 5) → (x + 2 * y = 2) := by
  sorry

end solution_relationship_l2231_223180


namespace orthogonal_vectors_imply_x_equals_two_l2231_223122

/-- Given two vectors a and b in ℝ², prove that if they are orthogonal
    and have the form a = (x-5, 3) and b = (2, x), then x = 2. -/
theorem orthogonal_vectors_imply_x_equals_two :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x - 5, 3)
  let b : ℝ × ℝ := (2, x)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 2 := by
sorry

end orthogonal_vectors_imply_x_equals_two_l2231_223122


namespace difference_of_squares_divisible_by_nine_l2231_223184

theorem difference_of_squares_divisible_by_nine (a b : ℤ) : 
  ∃ k : ℤ, (3*a + 2)^2 - (3*b + 2)^2 = 9*k := by sorry

end difference_of_squares_divisible_by_nine_l2231_223184


namespace union_of_M_and_N_l2231_223112

def M : Set ℕ := {1, 2, 4, 5}
def N : Set ℕ := {2, 3, 4}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3, 4, 5} := by
  sorry

end union_of_M_and_N_l2231_223112


namespace cube_root_equation_solution_l2231_223162

theorem cube_root_equation_solution :
  ∃ (x y z : ℕ+),
    (4 * ((7 : ℝ)^(1/3) - (6 : ℝ)^(1/3))^(1/2) = x^(1/3) + y^(1/3) - z^(1/3)) ∧
    (x + y + z = 51) :=
by sorry

end cube_root_equation_solution_l2231_223162


namespace semi_truck_journey_l2231_223110

/-- A problem about a semi truck's journey on paved and dirt roads. -/
theorem semi_truck_journey (total_distance : ℝ) (paved_time : ℝ) (dirt_speed : ℝ) 
  (speed_difference : ℝ) (h1 : total_distance = 200) 
  (h2 : paved_time = 2) (h3 : dirt_speed = 32) (h4 : speed_difference = 20) : 
  (total_distance - paved_time * (dirt_speed + speed_difference)) / dirt_speed = 3 := by
  sorry

#check semi_truck_journey

end semi_truck_journey_l2231_223110


namespace investment_rate_problem_l2231_223152

/-- Given a sum of money invested for a certain period, this function calculates the simple interest. -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_problem (sum : ℝ) (time : ℝ) (base_rate : ℝ) (interest_difference : ℝ) 
  (higher_rate : ℝ) :
  sum = 14000 →
  time = 2 →
  base_rate = 0.12 →
  interest_difference = 840 →
  simpleInterest sum higher_rate time = simpleInterest sum base_rate time + interest_difference →
  higher_rate = 0.15 := by
sorry

end investment_rate_problem_l2231_223152


namespace root_product_theorem_l2231_223187

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^4 - x^3 + 2*x - 3

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 - 3

-- State the theorem
theorem root_product_theorem (x₁ x₂ x₃ x₄ : ℝ) : 
  f x₁ = 0 → f x₂ = 0 → f x₃ = 0 → f x₄ = 0 → 
  g x₁ * g x₂ * g x₃ * g x₄ = 33 := by
sorry

end root_product_theorem_l2231_223187


namespace reflection_property_l2231_223126

/-- A reflection in R^2 -/
def Reflection (v : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ := sorry

theorem reflection_property (r : ℝ × ℝ → ℝ × ℝ) :
  r (2, 4) = (10, -2) →
  r (1, 6) = (107/37, -198/37) :=
by sorry

end reflection_property_l2231_223126


namespace power_division_l2231_223120

theorem power_division (x : ℕ) : 8^15 / 64^3 = 8^9 := by
  sorry

end power_division_l2231_223120


namespace percent_equality_problem_l2231_223136

theorem percent_equality_problem (x : ℝ) : (80 / 100 * 600 = 50 / 100 * x) → x = 960 := by
  sorry

end percent_equality_problem_l2231_223136


namespace abc_sum_range_l2231_223127

theorem abc_sum_range (a b c : ℝ) (h : a + b + 2*c = 0) :
  (∃ y : ℝ, y < 0 ∧ ab + ac + bc = y) ∧ ab + ac + bc ≤ 0 :=
by sorry

end abc_sum_range_l2231_223127


namespace masks_duration_for_andrew_family_l2231_223158

/-- The number of days a pack of masks lasts for a family -/
def masksDuration (familySize : ℕ) (packSize : ℕ) (daysPerMask : ℕ) : ℕ :=
  let masksUsedPer2Days := familySize
  let fullSets := packSize / masksUsedPer2Days
  let remainingMasks := packSize % masksUsedPer2Days
  let fullDays := fullSets * daysPerMask
  if remainingMasks ≥ familySize then
    fullDays + daysPerMask
  else
    fullDays + 1

/-- Theorem: A pack of 75 masks lasts 21 days for a family of 7, changing masks every 2 days -/
theorem masks_duration_for_andrew_family :
  masksDuration 7 75 2 = 21 := by
  sorry

end masks_duration_for_andrew_family_l2231_223158


namespace x_greater_than_one_sufficient_not_necessary_l2231_223113

theorem x_greater_than_one_sufficient_not_necessary :
  (∀ x : ℝ, x > 1 → x^2 - 2*x + 1 > 0) ∧
  (∃ x : ℝ, x ≤ 1 ∧ x^2 - 2*x + 1 > 0) := by
  sorry

end x_greater_than_one_sufficient_not_necessary_l2231_223113


namespace geometric_series_problem_l2231_223121

theorem geometric_series_problem (n : ℝ) : 
  let a₁ : ℝ := 15
  let b₁ : ℝ := 3
  let a₂ : ℝ := 15
  let b₂ : ℝ := 3 + n
  let r₁ : ℝ := b₁ / a₁
  let r₂ : ℝ := b₂ / a₂
  let S₁ : ℝ := a₁ / (1 - r₁)
  let S₂ : ℝ := a₂ / (1 - r₂)
  S₂ = 5 * S₁ → n = 9.6 := by
sorry

end geometric_series_problem_l2231_223121


namespace rational_sum_problem_l2231_223129

theorem rational_sum_problem (a b c d : ℚ) 
  (h1 : b + c + d = -1)
  (h2 : a + c + d = -3)
  (h3 : a + b + d = 2)
  (h4 : a + b + c = 17) :
  a = 6 ∧ b = 8 ∧ c = 3 ∧ d = -12 := by
  sorry

end rational_sum_problem_l2231_223129


namespace sufficient_but_not_necessary_l2231_223172

theorem sufficient_but_not_necessary (x : ℝ) :
  (x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ∧
  ¬(x^2 - 3*x + 2 ≠ 0 → x ≠ 1) := by
  sorry

end sufficient_but_not_necessary_l2231_223172


namespace sum_mod_thirteen_equals_zero_l2231_223115

theorem sum_mod_thirteen_equals_zero :
  (7650 + 7651 + 7652 + 7653 + 7654) % 13 = 0 := by
sorry

end sum_mod_thirteen_equals_zero_l2231_223115
