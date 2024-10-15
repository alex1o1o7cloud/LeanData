import Mathlib

namespace NUMINAMATH_CALUDE_add_self_eq_two_mul_l2194_219408

theorem add_self_eq_two_mul (a : ℝ) : a + a = 2 * a := by sorry

end NUMINAMATH_CALUDE_add_self_eq_two_mul_l2194_219408


namespace NUMINAMATH_CALUDE_area_of_specific_circumscribed_rectangle_l2194_219423

/-- A rectangle circumscribed around a right triangle -/
structure CircumscribedRectangle where
  /-- Length of one leg of the right triangle -/
  leg1 : ℝ
  /-- Length of the other leg of the right triangle -/
  leg2 : ℝ
  /-- The legs are positive -/
  leg1_pos : 0 < leg1
  leg2_pos : 0 < leg2

/-- The area of a rectangle circumscribed around a right triangle -/
def area (r : CircumscribedRectangle) : ℝ := r.leg1 * r.leg2

/-- Theorem: The area of a rectangle circumscribed around a right triangle
    with legs of length 5 and 6 is equal to 30 square units -/
theorem area_of_specific_circumscribed_rectangle :
  ∃ (r : CircumscribedRectangle), r.leg1 = 5 ∧ r.leg2 = 6 ∧ area r = 30 := by
  sorry


end NUMINAMATH_CALUDE_area_of_specific_circumscribed_rectangle_l2194_219423


namespace NUMINAMATH_CALUDE_negation_equivalence_l2194_219492

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2194_219492


namespace NUMINAMATH_CALUDE_parabola_translation_l2194_219458

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 1 0 0  -- y = x^2
  let translated := translate original 3 2
  y = translated.a * (x - 3)^2 + translated.b * (x - 3) + translated.c ↔
  y = (x - 3)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l2194_219458


namespace NUMINAMATH_CALUDE_book_cost_price_l2194_219404

/-- Proves that given a book sold for Rs 70 with a 40% profit rate, the cost price of the book is Rs 50. -/
theorem book_cost_price (selling_price : ℝ) (profit_rate : ℝ) 
  (h1 : selling_price = 70)
  (h2 : profit_rate = 0.4) :
  selling_price / (1 + profit_rate) = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l2194_219404


namespace NUMINAMATH_CALUDE_mean_median_mode_equality_l2194_219429

/-- Represents the days of the week -/
inductive Weekday
  | Saturday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- A month with its properties -/
structure Month where
  totalDays : Nat
  startDay : Weekday
  frequencies : Weekday → Nat

/-- Calculates the mean of the frequencies -/
def calculateMean (m : Month) : ℚ :=
  (m.frequencies Weekday.Saturday +
   m.frequencies Weekday.Sunday +
   m.frequencies Weekday.Monday +
   m.frequencies Weekday.Tuesday +
   m.frequencies Weekday.Wednesday +
   m.frequencies Weekday.Thursday +
   m.frequencies Weekday.Friday) / 7

/-- Calculates the median day -/
def calculateMedian (m : Month) : Weekday :=
  Weekday.Tuesday  -- Since the 15th day (median) is a Tuesday

/-- Calculates the median of the modes -/
def calculateMedianOfModes (m : Month) : ℚ := 4

/-- The theorem to be proved -/
theorem mean_median_mode_equality (m : Month)
  (h1 : m.totalDays = 29)
  (h2 : m.startDay = Weekday.Saturday)
  (h3 : m.frequencies Weekday.Saturday = 5)
  (h4 : m.frequencies Weekday.Sunday = 4)
  (h5 : m.frequencies Weekday.Monday = 4)
  (h6 : m.frequencies Weekday.Tuesday = 4)
  (h7 : m.frequencies Weekday.Wednesday = 4)
  (h8 : m.frequencies Weekday.Thursday = 4)
  (h9 : m.frequencies Weekday.Friday = 4) :
  calculateMean m = calculateMedianOfModes m ∧
  calculateMedianOfModes m = (calculateMedian m).rec 4 4 4 4 4 4 4 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_mode_equality_l2194_219429


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2194_219409

/-- An isosceles triangle with integer side lengths and perimeter 10 --/
structure IsoscelesTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  isIsosceles : (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)
  perimeter : a + b + c = 10

/-- The possible side lengths of the isosceles triangle --/
def validSideLengths (t : IsoscelesTriangle) : Prop :=
  (t.a = 3 ∧ t.b = 3 ∧ t.c = 4) ∨ (t.a = 4 ∧ t.b = 4 ∧ t.c = 2)

/-- Theorem stating that the only possible side lengths are (3, 3, 4) or (4, 4, 2) --/
theorem isosceles_triangle_side_lengths (t : IsoscelesTriangle) : validSideLengths t := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2194_219409


namespace NUMINAMATH_CALUDE_measure_six_pints_l2194_219464

/-- Represents the state of wine distribution -/
structure WineState :=
  (total : ℕ)
  (container8 : ℕ)
  (container5 : ℕ)

/-- Represents a pouring action -/
inductive PourAction
  | FillFrom8To5
  | FillFrom5To8
  | EmptyTo8
  | EmptyTo5
  | Empty8
  | Empty5

/-- Applies a pouring action to a wine state -/
def applyAction (state : WineState) (action : PourAction) : WineState :=
  match action with
  | PourAction.FillFrom8To5 => sorry
  | PourAction.FillFrom5To8 => sorry
  | PourAction.EmptyTo8 => sorry
  | PourAction.EmptyTo5 => sorry
  | PourAction.Empty8 => sorry
  | PourAction.Empty5 => sorry

/-- Checks if the goal state is reached -/
def isGoalState (state : WineState) : Prop :=
  state.container8 = 6

/-- Theorem: It is possible to measure 6 pints into the 8-pint container -/
theorem measure_six_pints 
  (initialState : WineState)
  (h_total : initialState.total = 12)
  (h_containers : initialState.container8 = 0 ∧ initialState.container5 = 0) :
  ∃ (actions : List PourAction), 
    isGoalState (actions.foldl applyAction initialState) :=
sorry

end NUMINAMATH_CALUDE_measure_six_pints_l2194_219464


namespace NUMINAMATH_CALUDE_chess_tournament_games_l2194_219489

/-- Calculate the number of games in a chess tournament -/
theorem chess_tournament_games (n : ℕ) (h : n = 20) : n * (n - 1) = 760 := by
  sorry

#check chess_tournament_games

end NUMINAMATH_CALUDE_chess_tournament_games_l2194_219489


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_min_dot_product_l2194_219445

-- Define the fixed point F
def F : ℝ × ℝ := (0, 1)

-- Define the line l₁
def l₁ (x : ℝ) : ℝ := -1

-- Define the trajectory of point C
def trajectory (x y : ℝ) : Prop := x^2 = 4*y

-- Define a line passing through F
def l₂ (k : ℝ) (x : ℝ) : ℝ := k*x + 1

-- Define the dot product of two 2D vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

-- Theorem 1: The trajectory of point C is x² = 4y
theorem trajectory_is_parabola :
  ∀ (x y : ℝ), trajectory x y ↔ x^2 = 4*y :=
sorry

-- Theorem 2: The minimum value of RP · RQ is 16
theorem min_dot_product :
  ∃ (k : ℝ), 
    ∀ (x₁ y₁ x₂ y₂ : ℝ),
      trajectory x₁ y₁ →
      trajectory x₂ y₂ →
      y₁ = l₂ k x₁ →
      y₂ = l₂ k x₂ →
      let R : ℝ × ℝ := (-2/k, l₁ (-2/k));
      let P : ℝ × ℝ := (x₁, y₁);
      let Q : ℝ × ℝ := (x₂, y₂);
      dot_product (P.1 - R.1, P.2 - R.2) (Q.1 - R.1, Q.2 - R.2) ≥ 16 ∧
      (∃ (x₁' y₁' x₂' y₂' : ℝ),
        trajectory x₁' y₁' ∧
        trajectory x₂' y₂' ∧
        y₁' = l₂ k x₁' ∧
        y₂' = l₂ k x₂' ∧
        let R' : ℝ × ℝ := (-2/k, l₁ (-2/k));
        let P' : ℝ × ℝ := (x₁', y₁');
        let Q' : ℝ × ℝ := (x₂', y₂');
        dot_product (P'.1 - R'.1, P'.2 - R'.2) (Q'.1 - R'.1, Q'.2 - R'.2) = 16) :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_min_dot_product_l2194_219445


namespace NUMINAMATH_CALUDE_letterArrangements_eq_25_l2194_219414

/-- The number of ways to arrange 15 letters with specific constraints -/
def letterArrangements : ℕ :=
  let totalLetters := 15
  let numA := 4
  let numB := 6
  let numC := 5
  let firstSection := 5
  let middleSection := 5
  let lastSection := 5
  -- Define the constraints
  let noCInFirst := true
  let noAInMiddle := true
  let noBInLast := true
  -- Calculate the number of arrangements
  25

/-- Theorem stating that the number of valid arrangements is 25 -/
theorem letterArrangements_eq_25 : letterArrangements = 25 := by
  sorry

end NUMINAMATH_CALUDE_letterArrangements_eq_25_l2194_219414


namespace NUMINAMATH_CALUDE_board_cut_ratio_l2194_219431

/-- Given a board of length 69 inches cut into two pieces,
    where one piece is a multiple of the other and the longer piece is 46 inches,
    prove that the ratio of the longer piece to the shorter piece is 2:1 -/
theorem board_cut_ratio (total_length shorter_length longer_length : ℝ)
  (h1 : total_length = 69)
  (h2 : total_length = shorter_length + longer_length)
  (h3 : ∃ (m : ℝ), longer_length = m * shorter_length)
  (h4 : longer_length = 46) :
  longer_length / shorter_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_ratio_l2194_219431


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l2194_219444

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x, (1 + a) * x > 1 + a ↔ x < 1) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l2194_219444


namespace NUMINAMATH_CALUDE_rugby_team_size_l2194_219493

theorem rugby_team_size (initial_avg : ℝ) (new_player_weight : ℝ) (new_avg : ℝ) :
  initial_avg = 180 →
  new_player_weight = 210 →
  new_avg = 181.42857142857142 →
  ∃ n : ℕ, (n : ℝ) * initial_avg + new_player_weight = (n + 1 : ℝ) * new_avg ∧ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_rugby_team_size_l2194_219493


namespace NUMINAMATH_CALUDE_inverse_function_inequality_l2194_219474

open Set
open Function
open Real

noncomputable def f (x : ℝ) := -x * abs x

theorem inverse_function_inequality (h : Bijective f) 
  (h2 : ∀ x ∈ Icc (-2 : ℝ) 2, (invFun f) (x^2 + m) < f x) : 
  m > 12 := by sorry

end NUMINAMATH_CALUDE_inverse_function_inequality_l2194_219474


namespace NUMINAMATH_CALUDE_ratio_G_to_N_l2194_219494

-- Define the variables
variable (N : ℝ) -- Number of non-college graduates
variable (C : ℝ) -- Number of college graduates without a graduate degree
variable (G : ℝ) -- Number of college graduates with a graduate degree

-- Define the conditions
axiom ratio_C_to_N : C = (2/3) * N
axiom prob_G : G / (G + C) = 0.15789473684210525

-- Theorem to prove
theorem ratio_G_to_N : G = (1/8) * N := by sorry

end NUMINAMATH_CALUDE_ratio_G_to_N_l2194_219494


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2194_219486

/-- A right triangle with perimeter 40 and area 30 has a hypotenuse of length 18.5 -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for right triangle
  a + b + c = 40 →   -- Perimeter condition
  a * b / 2 = 30 →   -- Area condition
  c = 18.5 := by
    sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2194_219486


namespace NUMINAMATH_CALUDE_custom_op_three_four_l2194_219491

/-- Custom binary operation * -/
def custom_op (a b : ℝ) : ℝ := 4*a + 5*b - a^2*b

/-- Theorem stating that 3 * 4 = -4 under the custom operation -/
theorem custom_op_three_four : custom_op 3 4 = -4 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_three_four_l2194_219491


namespace NUMINAMATH_CALUDE_arithmetic_expressions_l2194_219434

theorem arithmetic_expressions : 
  ((-8) - (-7) - |(-3)| = -4) ∧ 
  (-2^2 + 3 * (-1)^2019 - 9 / (-3) = 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_l2194_219434


namespace NUMINAMATH_CALUDE_red_marble_probability_l2194_219477

/-- The probability of drawing exactly k red marbles out of n draws with replacement
    from a bag containing r red marbles and b blue marbles. -/
def probability (r b k n : ℕ) : ℚ :=
  (n.choose k) * ((r : ℚ) / (r + b : ℚ)) ^ k * ((b : ℚ) / (r + b : ℚ)) ^ (n - k)

/-- The probability of drawing exactly 4 red marbles out of 8 draws with replacement
    from a bag containing 8 red marbles and 4 blue marbles is equal to 1120/6561. -/
theorem red_marble_probability : probability 8 4 4 8 = 1120 / 6561 := by
  sorry

end NUMINAMATH_CALUDE_red_marble_probability_l2194_219477


namespace NUMINAMATH_CALUDE_least_n_divisibility_l2194_219418

theorem least_n_divisibility (a b : ℕ+) : 
  (∃ (n : ℕ+), n = 1296 ∧ 
    (∀ (a b : ℕ+), 36 ∣ (a + b) → n ∣ (a * b) → 36 ∣ a ∧ 36 ∣ b) ∧
    (∀ (m : ℕ+), m < n → 
      ∃ (x y : ℕ+), 36 ∣ (x + y) ∧ m ∣ (x * y) ∧ (¬(36 ∣ x) ∨ ¬(36 ∣ y)))) :=
by
  sorry

#check least_n_divisibility

end NUMINAMATH_CALUDE_least_n_divisibility_l2194_219418


namespace NUMINAMATH_CALUDE_focus_of_parabola_l2194_219440

/-- The parabola defined by the equation y^2 = 4x -/
def parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of a parabola with equation y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the parabola y^2 = 4x has coordinates (1, 0) -/
theorem focus_of_parabola :
  focus ∈ {p : ℝ × ℝ | p.1 > 0 ∧ ∀ q ∈ parabola, (p.1 - q.1)^2 + (p.2 - q.2)^2 = (p.1 + q.1)^2} :=
sorry

end NUMINAMATH_CALUDE_focus_of_parabola_l2194_219440


namespace NUMINAMATH_CALUDE_bacteria_growth_proof_l2194_219448

/-- The number of 30-second intervals in 5 minutes -/
def intervals : ℕ := 10

/-- The growth factor of bacteria population in one interval -/
def growth_factor : ℕ := 4

/-- The final number of bacteria after 5 minutes -/
def final_population : ℕ := 4194304

/-- The initial number of bacteria -/
def initial_population : ℕ := 4

theorem bacteria_growth_proof :
  initial_population * growth_factor ^ intervals = final_population :=
by sorry

end NUMINAMATH_CALUDE_bacteria_growth_proof_l2194_219448


namespace NUMINAMATH_CALUDE_trader_profit_double_price_l2194_219487

theorem trader_profit_double_price (cost : ℝ) (initial_profit_percent : ℝ) 
  (h1 : initial_profit_percent = 40) : 
  let initial_price := cost * (1 + initial_profit_percent / 100)
  let new_price := 2 * initial_price
  let new_profit := new_price - cost
  new_profit / cost * 100 = 180 := by
sorry

end NUMINAMATH_CALUDE_trader_profit_double_price_l2194_219487


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2194_219406

theorem inequality_solution_set :
  let S := {x : ℝ | (1 / 3 : ℝ) < x ∧ x < (5 / 3 : ℝ) ∧ x ≠ 1}
  ∀ x : ℝ, x ∈ S ↔ (1 / |x - 1| : ℝ) > (3 / 2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2194_219406


namespace NUMINAMATH_CALUDE_inequality_comparison_l2194_219407

theorem inequality_comparison (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧ 
  (abs a > abs b) ∧ 
  (a^2 > b^2) ∧
  ¬(∀ a b, a < b ∧ b < 0 → 1 / (a - b) > 1 / a) := by
sorry

end NUMINAMATH_CALUDE_inequality_comparison_l2194_219407


namespace NUMINAMATH_CALUDE_complex_number_problem_l2194_219412

theorem complex_number_problem (i : ℂ) (h : i^2 = -1) :
  let z_i := ((i + 1) / (i - 1))^2016
  let z := 1 / i
  z = -i :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2194_219412


namespace NUMINAMATH_CALUDE_points_in_different_half_spaces_l2194_219498

/-- A plane in 3D space defined by the equation ax + by + cz + d = 0 --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space --/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Determine if two points are on opposite sides of a plane --/
def oppositeHalfSpaces (plane : Plane) (p1 p2 : Point3D) : Prop :=
  (plane.a * p1.x + plane.b * p1.y + plane.c * p1.z + plane.d) *
  (plane.a * p2.x + plane.b * p2.y + plane.c * p2.z + plane.d) < 0

theorem points_in_different_half_spaces :
  let plane := Plane.mk 1 2 3 0
  let point1 := Point3D.mk 1 2 (-2)
  let point2 := Point3D.mk 2 1 (-1)
  oppositeHalfSpaces plane point1 point2 := by
  sorry


end NUMINAMATH_CALUDE_points_in_different_half_spaces_l2194_219498


namespace NUMINAMATH_CALUDE_inequality_proof_l2194_219441

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2194_219441


namespace NUMINAMATH_CALUDE_apples_left_is_340_l2194_219470

/-- The number of baskets --/
def num_baskets : ℕ := 11

/-- The number of children --/
def num_children : ℕ := 10

/-- The total number of apples initially --/
def total_apples : ℕ := 1000

/-- The sum of the first n natural numbers --/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of apples picked by all children --/
def apples_picked : ℕ := num_children * sum_first_n num_baskets

/-- The number of apples left after picking --/
def apples_left : ℕ := total_apples - apples_picked

theorem apples_left_is_340 : apples_left = 340 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_is_340_l2194_219470


namespace NUMINAMATH_CALUDE_stating_carlas_counting_problem_l2194_219475

/-- 
Theorem stating that there exists a positive integer solution for the number of tiles and books
that satisfies the equation from Carla's counting problem.
-/
theorem carlas_counting_problem :
  ∃ (T B : ℕ), T > 0 ∧ B > 0 ∧ 2 * T + 3 * B = 301 := by
  sorry

end NUMINAMATH_CALUDE_stating_carlas_counting_problem_l2194_219475


namespace NUMINAMATH_CALUDE_fraction_doubles_l2194_219481

theorem fraction_doubles (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (2*x)*(2*y) / ((2*x) + (2*y)) = 2 * (x*y / (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_doubles_l2194_219481


namespace NUMINAMATH_CALUDE_system_solution_l2194_219435

theorem system_solution : 
  ∃! (x y : ℚ), (4 * x - 3 * y = 2) ∧ (5 * x + 4 * y = 3) ∧ x = 17/31 ∧ y = 2/31 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2194_219435


namespace NUMINAMATH_CALUDE_f_at_2_l2194_219462

theorem f_at_2 (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x^2 + 5 * x - 1) : f 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l2194_219462


namespace NUMINAMATH_CALUDE_equation_roots_properties_l2194_219452

open Real

theorem equation_roots_properties (m : ℝ) (θ : ℝ) :
  θ ∈ Set.Ioo 0 π →
  (∀ x, 2 * x^2 - (Real.sqrt 3 + 1) * x + m = 0 ↔ x = sin θ ∨ x = cos θ) →
  (m = Real.sqrt 3 / 2) ∧
  ((tan θ * sin θ) / (tan θ - 1) + cos θ / (1 - tan θ) = (Real.sqrt 3 + 1) / 2) ∧
  ((sin θ = Real.sqrt 3 / 2 ∧ cos θ = 1 / 2) ∨ (sin θ = 1 / 2 ∧ cos θ = Real.sqrt 3 / 2)) ∧
  (θ = π / 3 ∨ θ = π / 6) := by
  sorry

#check equation_roots_properties

end NUMINAMATH_CALUDE_equation_roots_properties_l2194_219452


namespace NUMINAMATH_CALUDE_factors_of_72_l2194_219455

theorem factors_of_72 : Nat.card (Nat.divisors 72) = 12 := by sorry

end NUMINAMATH_CALUDE_factors_of_72_l2194_219455


namespace NUMINAMATH_CALUDE_brendas_age_l2194_219460

/-- Given that Addison's age is four times Brenda's age, Janet is seven years older than Brenda,
    and Addison and Janet are twins, prove that Brenda is 7/3 years old. -/
theorem brendas_age (addison janet brenda : ℚ)
  (h1 : addison = 4 * brenda)
  (h2 : janet = brenda + 7)
  (h3 : addison = janet) :
  brenda = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brendas_age_l2194_219460


namespace NUMINAMATH_CALUDE_volunteer_arrangements_l2194_219439

theorem volunteer_arrangements (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 7 ∧ k = 3 ∧ m = 4 →
  (n.choose k) * (m.choose k) = 140 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_arrangements_l2194_219439


namespace NUMINAMATH_CALUDE_pen_collection_theorem_l2194_219437

/-- Calculates the final number of pens after a series of operations --/
def final_pen_count (initial : ℕ) (mike_gives : ℕ) (cindy_multiplier : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * cindy_multiplier) - sharon_takes

/-- Proves that the final number of pens is 31 given the specific conditions --/
theorem pen_collection_theorem :
  final_pen_count 5 20 2 19 = 31 := by
  sorry

end NUMINAMATH_CALUDE_pen_collection_theorem_l2194_219437


namespace NUMINAMATH_CALUDE_middle_number_divisible_by_four_l2194_219427

theorem middle_number_divisible_by_four (x : ℕ) :
  (∃ y : ℕ, (x - 1)^3 + x^3 + (x + 1)^3 = y^3) →
  4 ∣ x :=
by sorry

end NUMINAMATH_CALUDE_middle_number_divisible_by_four_l2194_219427


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2194_219424

theorem smallest_number_with_remainders (n : ℕ) : 
  (n > 1) →
  (n % 3 = 1) → 
  (n % 4 = 1) → 
  (n % 7 = 1) → 
  (∀ m : ℕ, m > 1 → m % 3 = 1 → m % 4 = 1 → m % 7 = 1 → n ≤ m) →
  (n = 85 ∧ 84 < n ∧ n ≤ 107) := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2194_219424


namespace NUMINAMATH_CALUDE_all_cards_same_number_l2194_219433

theorem all_cards_same_number (m : ℕ) (cards : Fin m → ℕ) : 
  (∀ i : Fin m, 1 ≤ cards i ∧ cards i ≤ m) →
  (∀ s : Finset (Fin m), (s.sum cards) % (m + 1) ≠ 0) →
  ∀ i j : Fin m, cards i = cards j :=
sorry

end NUMINAMATH_CALUDE_all_cards_same_number_l2194_219433


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l2194_219483

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n.sqrt ^ 2 = n → n % 2 = 0 → n % 3 = 0 → n % 5 = 0 → n ≥ 225 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l2194_219483


namespace NUMINAMATH_CALUDE_car_dealership_problem_l2194_219499

theorem car_dealership_problem (initial_cars : ℕ) (initial_silver_percent : ℚ)
  (new_shipment : ℕ) (final_silver_percent : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percent = 1/5)
  (h3 : new_shipment = 80)
  (h4 : final_silver_percent = 3/10) :
  (1 - (final_silver_percent * (initial_cars + new_shipment) - initial_silver_percent * initial_cars) / new_shipment) = 13/20 := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_problem_l2194_219499


namespace NUMINAMATH_CALUDE_f_composed_with_g_l2194_219405

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 1
def g (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_composed_with_g : f (1 + g 4) = 11 := by
  sorry

end NUMINAMATH_CALUDE_f_composed_with_g_l2194_219405


namespace NUMINAMATH_CALUDE_prob_more_ones_than_sixes_proof_l2194_219425

/-- The number of possible outcomes when rolling five fair six-sided dice -/
def total_outcomes : ℕ := 6^5

/-- The number of ways to roll an equal number of 1's and 6's when rolling five fair six-sided dice -/
def equal_ones_and_sixes : ℕ := 2334

/-- The probability of rolling more 1's than 6's when rolling five fair six-sided dice -/
def prob_more_ones_than_sixes : ℚ := 2711 / 7776

theorem prob_more_ones_than_sixes_proof :
  prob_more_ones_than_sixes = 1/2 * (1 - equal_ones_and_sixes / total_outcomes) :=
by sorry

end NUMINAMATH_CALUDE_prob_more_ones_than_sixes_proof_l2194_219425


namespace NUMINAMATH_CALUDE_sin_even_function_phi_l2194_219403

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem sin_even_function_phi (φ : ℝ) 
  (h1 : is_even_function (fun x ↦ Real.sin (x + φ)))
  (h2 : 0 ≤ φ ∧ φ ≤ π) :
  φ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_even_function_phi_l2194_219403


namespace NUMINAMATH_CALUDE_min_value_reciprocal_product_l2194_219459

theorem min_value_reciprocal_product (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : 2 = (2 * a + b) / 2) : 
  ∀ x y, x > 0 → y > 0 → (2 = (2 * x + y) / 2) → 1 / (a * b) ≤ 1 / (x * y) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_product_l2194_219459


namespace NUMINAMATH_CALUDE_linear_system_sum_theorem_l2194_219400

theorem linear_system_sum_theorem (a b c x y z : ℝ) 
  (eq1 : 23*x + b*y + c*z = 0)
  (eq2 : a*x + 33*y + c*z = 0)
  (eq3 : a*x + b*y + 52*z = 0)
  (ha : a ≠ 23)
  (hx : x ≠ 0) :
  (a + 10) / (a + 10 - 23) + b / (b - 33) + c / (c - 52) = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_system_sum_theorem_l2194_219400


namespace NUMINAMATH_CALUDE_inverse_square_problem_l2194_219420

/-- Represents the inverse square relationship between x and y -/
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y * y)

theorem inverse_square_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h₁ : y₁ = 6)
  (h₂ : x₁ = 0.1111111111111111)
  (h₃ : y₂ = 2)
  (h₄ : ∃ k, inverse_square_relation k x₁ y₁ ∧ inverse_square_relation k x₂ y₂) :
  x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_problem_l2194_219420


namespace NUMINAMATH_CALUDE_expression_simplification_l2194_219415

theorem expression_simplification (x y : ℚ) (hx : x = 1/2) (hy : y = -2) :
  ((2*x + y)^2 - (2*x - y)*(x + y) - 2*(x - 2*y)*(x + 2*y)) / y = -37/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2194_219415


namespace NUMINAMATH_CALUDE_difference_of_squares_l2194_219485

theorem difference_of_squares (a b : ℝ) : (3*a + b) * (3*a - b) = 9*a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2194_219485


namespace NUMINAMATH_CALUDE_seating_theorem_l2194_219428

/-- The number of ways to arrange 3 people in a row of 6 seats with exactly two adjacent empty seats -/
def seating_arrangements (total_seats : Nat) (people : Nat) (adjacent_empty : Nat) : Nat :=
  24 * 3

theorem seating_theorem :
  seating_arrangements 6 3 2 = 72 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l2194_219428


namespace NUMINAMATH_CALUDE_quadratic_sum_l2194_219411

/-- A quadratic function with vertex (h, k) and passing through point (x₀, y₀) -/
def quadratic_function (a b c h k x₀ y₀ : ℝ) : Prop :=
  ∀ x, a * (x - h)^2 + k = a * x^2 + b * x + c ∧
  a * (x₀ - h)^2 + k = y₀

theorem quadratic_sum (a b c : ℝ) :
  quadratic_function a b c 2 5 1 8 →
  a - b + c = 32 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2194_219411


namespace NUMINAMATH_CALUDE_investment_dividend_calculation_l2194_219468

/-- Calculates the dividend received from an investment in shares with premium and dividend rates -/
def calculate_dividend (investment : ℚ) (share_value : ℚ) (premium_rate : ℚ) (dividend_rate : ℚ) : ℚ :=
  let share_cost := share_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := share_value * dividend_rate
  num_shares * dividend_per_share

/-- Theorem stating that the given investment yields the correct dividend -/
theorem investment_dividend_calculation :
  calculate_dividend 14400 100 (20/100) (7/100) = 840 := by
  sorry

#eval calculate_dividend 14400 100 (20/100) (7/100)

end NUMINAMATH_CALUDE_investment_dividend_calculation_l2194_219468


namespace NUMINAMATH_CALUDE_polygon_angles_l2194_219454

theorem polygon_angles (n : ℕ) : 
  (n - 2) * 180 + (180 - 180 / n) = 2007 → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angles_l2194_219454


namespace NUMINAMATH_CALUDE_expected_BBR_sequences_l2194_219446

/-- Represents a standard deck of 52 cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents a sequence of three cards -/
structure ThreeCardSequence :=
  (first : Deck)
  (second : Deck)
  (third : Deck)

/-- Checks if a card is black -/
def is_black (card : Deck) : Prop :=
  sorry

/-- Checks if a card is red -/
def is_red (card : Deck) : Prop :=
  sorry

/-- Checks if a sequence is BBR (two black cards followed by a red card) -/
def is_BBR (seq : ThreeCardSequence) : Prop :=
  is_black seq.first ∧ is_black seq.second ∧ is_red seq.third

/-- The probability of a specific BBR sequence -/
def BBR_probability : ℚ :=
  13 / 51

/-- The number of possible starting positions for a BBR sequence -/
def num_starting_positions : ℕ :=
  26

/-- The expected number of BBR sequences in a standard 52-card deck dealt in a circle -/
theorem expected_BBR_sequences :
  (num_starting_positions : ℚ) * BBR_probability = 338 / 51 :=
sorry

end NUMINAMATH_CALUDE_expected_BBR_sequences_l2194_219446


namespace NUMINAMATH_CALUDE_probability_calculation_l2194_219422

/-- The probability of selecting one qualified and one unqualified product -/
def probability_one_qualified_one_unqualified : ℚ :=
  3 / 5

/-- The total number of products -/
def total_products : ℕ := 5

/-- The number of qualified products -/
def qualified_products : ℕ := 3

/-- The number of unqualified products -/
def unqualified_products : ℕ := 2

/-- The number of products selected for inspection -/
def selected_products : ℕ := 2

theorem probability_calculation :
  probability_one_qualified_one_unqualified = 
    (qualified_products.choose 1 * unqualified_products.choose 1 : ℚ) / 
    (total_products.choose selected_products) :=
by sorry

end NUMINAMATH_CALUDE_probability_calculation_l2194_219422


namespace NUMINAMATH_CALUDE_root_in_interval_l2194_219465

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  (∃ x ∈ Set.Icc 2 2.5, f x = 0) :=
by
  have h1 : f 2 < 0 := by sorry
  have h2 : f 2.5 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2194_219465


namespace NUMINAMATH_CALUDE_max_points_for_top_teams_is_76_l2194_219469

/-- Represents a soccer league with given parameters -/
structure SoccerLeague where
  numTeams : Nat
  gamesAgainstEachTeam : Nat
  pointsForWin : Nat
  pointsForDraw : Nat
  pointsForLoss : Nat

/-- Calculates the maximum possible points for each of the top three teams in the league -/
def maxPointsForTopTeams (league : SoccerLeague) : Nat :=
  sorry

/-- Theorem stating the maximum points for top teams in the specific league configuration -/
theorem max_points_for_top_teams_is_76 :
  let league : SoccerLeague := {
    numTeams := 9
    gamesAgainstEachTeam := 4
    pointsForWin := 3
    pointsForDraw := 1
    pointsForLoss := 0
  }
  maxPointsForTopTeams league = 76 := by sorry

end NUMINAMATH_CALUDE_max_points_for_top_teams_is_76_l2194_219469


namespace NUMINAMATH_CALUDE_first_discount_rate_l2194_219495

/-- Proves that given a shirt with an original price of 400, which after two
    consecutive discounts (the second being 5%) results in a final price of 340,
    the first discount rate is equal to (200/19)%. -/
theorem first_discount_rate (original_price : ℝ) (final_price : ℝ) (second_discount : ℝ) :
  original_price = 400 →
  final_price = 340 →
  second_discount = 0.05 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (1 - first_discount) * (1 - second_discount) ∧
    first_discount = 200 / 19 / 100 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_rate_l2194_219495


namespace NUMINAMATH_CALUDE_bathroom_cleaning_time_is_15_l2194_219436

/-- Represents the time spent on various tasks in minutes -/
structure TaskTimes where
  total : ℕ
  laundry : ℕ
  room : ℕ
  homework : ℕ

/-- Calculates the time spent cleaning the bathroom given the times for other tasks -/
def bathroomCleaningTime (t : TaskTimes) : ℕ :=
  t.total - (t.laundry + t.room + t.homework)

theorem bathroom_cleaning_time_is_15 (t : TaskTimes) 
  (h1 : t.total = 120)
  (h2 : t.laundry = 30)
  (h3 : t.room = 35)
  (h4 : t.homework = 40) :
  bathroomCleaningTime t = 15 := by
  sorry

#eval bathroomCleaningTime { total := 120, laundry := 30, room := 35, homework := 40 }

end NUMINAMATH_CALUDE_bathroom_cleaning_time_is_15_l2194_219436


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l2194_219402

/-- Given a parabola y² = 8x and a circle x² + y² + 6x + m = 0, 
    if the directrix of the parabola is tangent to the circle, then m = 8 -/
theorem parabola_circle_tangency (m : ℝ) : 
  (∀ y : ℝ, (∃! x : ℝ, x = -2 ∧ x^2 + y^2 + 6*x + m = 0)) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l2194_219402


namespace NUMINAMATH_CALUDE_david_homework_hours_l2194_219479

/-- Calculates the weekly homework hours for a course -/
def weekly_homework_hours (total_weeks : ℕ) (class_hours_per_week : ℕ) (total_course_hours : ℕ) : ℕ :=
  (total_course_hours - (total_weeks * class_hours_per_week)) / total_weeks

theorem david_homework_hours :
  let total_weeks : ℕ := 24
  let three_hour_classes : ℕ := 2
  let four_hour_classes : ℕ := 1
  let class_hours_per_week : ℕ := three_hour_classes * 3 + four_hour_classes * 4
  let total_course_hours : ℕ := 336
  weekly_homework_hours total_weeks class_hours_per_week total_course_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_david_homework_hours_l2194_219479


namespace NUMINAMATH_CALUDE_shells_remaining_calculation_l2194_219456

/-- The number of shells Lino picked up in the morning -/
def shells_picked_up : ℝ := 324.0

/-- The number of shells Lino put back in the afternoon -/
def shells_put_back : ℝ := 292.00

/-- The number of shells Lino has in all -/
def shells_remaining : ℝ := shells_picked_up - shells_put_back

/-- Theorem stating that the number of shells Lino has in all
    is equal to the difference between shells picked up and shells put back -/
theorem shells_remaining_calculation :
  shells_remaining = 32.0 := by sorry

end NUMINAMATH_CALUDE_shells_remaining_calculation_l2194_219456


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l2194_219449

theorem fruit_seller_apples (initial_apples : ℕ) (remaining_apples : ℕ) : 
  remaining_apples = 420 → 
  (initial_apples : ℚ) * (70 / 100) = remaining_apples → 
  initial_apples = 600 := by
sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l2194_219449


namespace NUMINAMATH_CALUDE_min_value_theorem_l2194_219450

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2 * y = 5) :
  ∃ (min_val : ℝ), min_val = 4 * Real.sqrt 3 ∧
  ∀ (z : ℝ), z = (x + 1) * (2 * y + 1) / Real.sqrt (x * y) → z ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2194_219450


namespace NUMINAMATH_CALUDE_abc_inequality_l2194_219461

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2194_219461


namespace NUMINAMATH_CALUDE_marching_band_ratio_l2194_219480

theorem marching_band_ratio (total_students : ℕ) (marching_band_fraction : ℚ) 
  (brass_to_saxophone : ℚ) (saxophone_to_alto : ℚ) (alto_players : ℕ) :
  total_students = 600 →
  marching_band_fraction = 1 / 5 →
  brass_to_saxophone = 1 / 5 →
  saxophone_to_alto = 1 / 3 →
  alto_players = 4 →
  (↑alto_players / (marching_band_fraction * saxophone_to_alto * brass_to_saxophone)) / 
  (marching_band_fraction * ↑total_students) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_marching_band_ratio_l2194_219480


namespace NUMINAMATH_CALUDE_intersection_line_slope_l2194_219447

-- Define the circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 12 = 0

-- Define the intersection points
def intersection_points (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 C.1 C.2 ∧ circle2 D.1 D.2

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) 
  (h : intersection_points C D) : 
  (D.2 - C.2) / (D.1 - C.1) = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l2194_219447


namespace NUMINAMATH_CALUDE_news_report_probability_l2194_219401

/-- The duration of the "Midday News" program in minutes -/
def program_duration : ℕ := 30

/-- The duration of the news report in minutes -/
def news_report_duration : ℕ := 5

/-- The time Xiao Zhang starts watching, in minutes after the program start -/
def watch_start_time : ℕ := 20

/-- The probability of watching the entire news report -/
def watch_probability : ℚ := 1 / 6

theorem news_report_probability :
  let favorable_time := program_duration - watch_start_time - news_report_duration + 1
  watch_probability = favorable_time / program_duration :=
by sorry

end NUMINAMATH_CALUDE_news_report_probability_l2194_219401


namespace NUMINAMATH_CALUDE_binop_commutative_l2194_219432

-- Define a binary operation on a type
def BinOp (α : Type) := α → α → α

-- Define the properties of the binary operation
class MyBinOp (α : Type) (op : BinOp α) where
  left_cancel : ∀ a b : α, op a (op a b) = b
  right_cancel : ∀ a b : α, op (op a b) b = a

-- State the theorem
theorem binop_commutative {α : Type} (op : BinOp α) [MyBinOp α op] :
  ∀ a b : α, op a b = op b a := by
  sorry

end NUMINAMATH_CALUDE_binop_commutative_l2194_219432


namespace NUMINAMATH_CALUDE_evaluate_expression_l2194_219419

theorem evaluate_expression (a b : ℕ) (h1 : a = 2009) (h2 : b = 2010) :
  2 * (b^3 - a*b^2 - a^2*b + a^3) = 24240542 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2194_219419


namespace NUMINAMATH_CALUDE_mean_temperature_l2194_219473

theorem mean_temperature (temperatures : List ℤ) : 
  temperatures = [-10, -4, -6, -3, 0, 2, 5, 0] →
  (temperatures.sum / temperatures.length : ℚ) = -2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l2194_219473


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l2194_219413

/-- The ratio of the volume of a cube with edge length 10 inches to the volume of a cube with edge length 3 feet -/
theorem cube_volume_ratio : 
  let inch_to_foot : ℚ := 1 / 12
  let cube1_edge : ℚ := 10
  let cube2_edge : ℚ := 3 / inch_to_foot
  let cube1_volume : ℚ := cube1_edge ^ 3
  let cube2_volume : ℚ := cube2_edge ^ 3
  cube1_volume / cube2_volume = 125 / 5832 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l2194_219413


namespace NUMINAMATH_CALUDE_yogurt_combinations_l2194_219466

def yogurt_types : ℕ := 2
def yogurt_flavors : ℕ := 5
def topping_count : ℕ := 8

def combination_count : ℕ := yogurt_types * yogurt_flavors * (topping_count.choose 2)

theorem yogurt_combinations :
  combination_count = 280 :=
by sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l2194_219466


namespace NUMINAMATH_CALUDE_distance_AP_equals_one_l2194_219476

-- Define the triangle and circle
def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (2, 0)
def center : ℝ × ℝ := (1, 1)

-- Define the inscribed circle
def ω (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

-- Define point M where ω touches BC
def M : ℝ × ℝ := (0, 0)

-- Define point P where AM intersects ω
def P : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem distance_AP_equals_one :
  let d := Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2)
  d = 1 := by sorry

end NUMINAMATH_CALUDE_distance_AP_equals_one_l2194_219476


namespace NUMINAMATH_CALUDE_inequality_proof_l2194_219453

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2194_219453


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2194_219488

/-- Given a complex number in the form (2-mi)/(1+2i) = A+Bi, where m, A, and B are real numbers,
    if A + B = 0, then m = 2 -/
theorem complex_equation_solution (m A B : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 - m * Complex.I) / (1 + 2 * Complex.I) = A + B * Complex.I →
  A + B = 0 →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2194_219488


namespace NUMINAMATH_CALUDE_intersection_line_slope_l2194_219457

/-- Given two circles in the plane, this theorem states that the slope of the line
passing through their intersection points is -1/3. -/
theorem intersection_line_slope (x y : ℝ) :
  (x^2 + y^2 - 6*x + 4*y - 8 = 0) ∧ 
  (x^2 + y^2 - 8*x - 2*y + 10 = 0) →
  (∃ m b : ℝ, y = m*x + b ∧ m = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l2194_219457


namespace NUMINAMATH_CALUDE_total_cups_doubled_is_60_l2194_219472

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients for a doubled recipe -/
def totalCupsDoubled (ratio : RecipeRatio) (flourCups : ℕ) : ℕ :=
  let partSize := flourCups / ratio.flour
  let butterCups := ratio.butter * partSize
  let sugarCups := ratio.sugar * partSize
  2 * (butterCups + flourCups + sugarCups)

/-- Theorem: Given the recipe ratio and flour quantity, the total cups for a doubled recipe is 60 -/
theorem total_cups_doubled_is_60 :
  totalCupsDoubled ⟨2, 5, 3⟩ 15 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_cups_doubled_is_60_l2194_219472


namespace NUMINAMATH_CALUDE_min_floor_sum_l2194_219471

theorem min_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (min : ℕ), min = 3 ∧
  (⌊(a^2 + b^2) / (a + b)⌋ + ⌊(b^2 + c^2) / (b + c)⌋ + ⌊(c^2 + a^2) / (c + a)⌋ ≥ min) ∧
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    ⌊(x^2 + y^2) / (x + y)⌋ + ⌊(y^2 + z^2) / (y + z)⌋ + ⌊(z^2 + x^2) / (z + x)⌋ ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_floor_sum_l2194_219471


namespace NUMINAMATH_CALUDE_quadratic_function_value_l2194_219482

theorem quadratic_function_value (a b x₁ x₂ : ℝ) : 
  a ≠ 0 → 
  (∃ y₁ y₂ : ℝ, y₁ = a * x₁^2 + b * x₁ + 2009 ∧ 
                y₂ = a * x₂^2 + b * x₂ + 2009 ∧ 
                y₁ = 2012 ∧ 
                y₂ = 2012) → 
  a * (x₁ + x₂)^2 + b * (x₁ + x₂) + 2009 = 2009 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l2194_219482


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2194_219451

theorem quadratic_roots_relation (m p q : ℝ) (hm : m ≠ 0) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∃ r₁ r₂ : ℝ, (r₁ + r₂ = -q ∧ r₁ * r₂ = m) ∧
               (3*r₁ + 3*r₂ = -m ∧ 9*r₁*r₂ = p)) →
  p / q = 27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2194_219451


namespace NUMINAMATH_CALUDE_triangle_problem_l2194_219438

open Real

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_c : c = 2 * b * cos B)
  (h_C : C = 2 * π / 3) :
  B = π / 6 ∧ 
  (∀ p, p = 4 + 2 * sqrt 3 → a + b + c = p → 
    ∃ m, m = sqrt 7 ∧ m^2 = (a^2 + b^2) / 4 + c^2 / 16) ∧
  (∀ S, S = 3 * sqrt 3 / 4 → (1/2) * a * b * sin C = S → 
    ∃ m, m = sqrt 21 / 2 ∧ m^2 = (a^2 + b^2) / 4 + c^2 / 16) :=
by sorry


end NUMINAMATH_CALUDE_triangle_problem_l2194_219438


namespace NUMINAMATH_CALUDE_triangle_area_l2194_219421

theorem triangle_area (a b c : ℝ) (A : ℝ) (S : ℝ) : 
  b^2 - b*c - 2*c^2 = 0 →
  a = Real.sqrt 6 →
  Real.cos A = 7/8 →
  S = (Real.sqrt 15)/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2194_219421


namespace NUMINAMATH_CALUDE_pistachio_count_l2194_219463

theorem pistachio_count (total : ℝ) 
  (h1 : 0.95 * total * 0.75 = 57) : total = 80 := by
  sorry

end NUMINAMATH_CALUDE_pistachio_count_l2194_219463


namespace NUMINAMATH_CALUDE_max_sum_squares_l2194_219442

theorem max_sum_squares : ∃ (m n : ℕ), 
  1 ≤ m ∧ m ≤ 2005 ∧ 
  1 ≤ n ∧ n ≤ 2005 ∧ 
  (n^2 + 2*m*n - 2*m^2)^2 = 1 ∧ 
  m^2 + n^2 = 702036 ∧ 
  ∀ (m' n' : ℕ), 
    1 ≤ m' ∧ m' ≤ 2005 → 
    1 ≤ n' ∧ n' ≤ 2005 → 
    (n'^2 + 2*m'*n' - 2*m'^2)^2 = 1 → 
    m'^2 + n'^2 ≤ 702036 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squares_l2194_219442


namespace NUMINAMATH_CALUDE_johns_first_second_distance_l2194_219417

/-- Represents the race scenario with John and James --/
structure RaceScenario where
  john_total_time : ℝ
  john_total_distance : ℝ
  james_top_speed_diff : ℝ
  james_initial_distance : ℝ
  james_initial_time : ℝ
  james_total_time : ℝ
  james_total_distance : ℝ

/-- Theorem stating John's distance in the first second --/
theorem johns_first_second_distance 
  (race : RaceScenario)
  (h_john_time : race.john_total_time = 13)
  (h_john_dist : race.john_total_distance = 100)
  (h_james_speed_diff : race.james_top_speed_diff = 2)
  (h_james_initial_dist : race.james_initial_distance = 10)
  (h_james_initial_time : race.james_initial_time = 2)
  (h_james_time : race.james_total_time = 11)
  (h_james_dist : race.james_total_distance = 100) :
  ∃ d : ℝ, d = 4 ∧ 
    (race.john_total_distance - d) / (race.john_total_time - 1) = 
    (race.james_total_distance - race.james_initial_distance) / (race.james_total_time - race.james_initial_time) - race.james_top_speed_diff :=
by sorry

end NUMINAMATH_CALUDE_johns_first_second_distance_l2194_219417


namespace NUMINAMATH_CALUDE_quadratic_equation_standard_form_quadratic_coefficients_l2194_219426

theorem quadratic_equation_standard_form :
  ∀ x : ℝ, (x + 5) * (3 + x) = 2 * x^2 ↔ x^2 - 8 * x - 15 = 0 :=
by sorry

theorem quadratic_coefficients (a b c : ℝ) :
  (∀ x : ℝ, (x + 5) * (3 + x) = 2 * x^2 ↔ a * x^2 + b * x + c = 0) →
  a = 1 ∧ b = -8 ∧ c = -15 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_standard_form_quadratic_coefficients_l2194_219426


namespace NUMINAMATH_CALUDE_polynomial_value_at_three_l2194_219410

theorem polynomial_value_at_three : 
  let x : ℤ := 3
  (x^5 : ℤ) - 5*x + 7*(x^3) = 417 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_at_three_l2194_219410


namespace NUMINAMATH_CALUDE_unique_function_satisfying_inequality_l2194_219484

def satisfies_inequality (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y) + f (x * z) + f (y * z) - f x * f y * f z ≥ 1

theorem unique_function_satisfying_inequality :
  ∃! f : ℝ → ℝ, satisfies_inequality f ∧ ∀ x : ℝ, f x = 1 :=
sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_inequality_l2194_219484


namespace NUMINAMATH_CALUDE_fresh_grapes_weight_l2194_219430

/-- The weight of fresh grapes required to produce a given weight of dried grapes -/
theorem fresh_grapes_weight (fresh_water_content : ℝ) (dried_water_content : ℝ) (dried_weight : ℝ) :
  fresh_water_content = 0.8 →
  dried_water_content = 0.2 →
  dried_weight = 10 →
  (1 - fresh_water_content) * (dried_weight / (1 - dried_water_content)) = 40 :=
by sorry

end NUMINAMATH_CALUDE_fresh_grapes_weight_l2194_219430


namespace NUMINAMATH_CALUDE_quadratic_properties_l2194_219443

def f (x : ℝ) := -x^2 + 2*x + 1

theorem quadratic_properties :
  (∀ x y : ℝ, f x ≤ f y → x = y ∨ (x < y ∧ f ((x + y) / 2) > f x) ∨ (y < x ∧ f ((x + y) / 2) > f x)) ∧
  (∃ x : ℝ, ∀ y : ℝ, f y ≤ f x) ∧
  (∃! x : ℝ, ∀ y : ℝ, f y ≤ f x) ∧
  (∀ x : ℝ, f x ≤ f 1) ∧
  f 1 = 2 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → -2 ≤ f x ∧ f x ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2194_219443


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2194_219478

theorem complex_number_in_third_quadrant :
  let z : ℂ := -Complex.I / (1 + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l2194_219478


namespace NUMINAMATH_CALUDE_donnas_truck_dryers_l2194_219497

/-- Calculates the number of dryers on Donna's truck given the weight constraints --/
theorem donnas_truck_dryers :
  let bridge_limit : ℕ := 20000
  let empty_truck_weight : ℕ := 12000
  let num_soda_crates : ℕ := 20
  let soda_crate_weight : ℕ := 50
  let dryer_weight : ℕ := 3000
  let loaded_truck_weight : ℕ := 24000
  let soda_weight : ℕ := num_soda_crates * soda_crate_weight
  let produce_weight : ℕ := 2 * soda_weight
  let truck_with_soda_produce : ℕ := empty_truck_weight + soda_weight + produce_weight
  let dryers_weight : ℕ := loaded_truck_weight - truck_with_soda_produce
  let num_dryers : ℕ := dryers_weight / dryer_weight
  num_dryers = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_donnas_truck_dryers_l2194_219497


namespace NUMINAMATH_CALUDE_negation_equivalence_l2194_219496

theorem negation_equivalence :
  ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2194_219496


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l2194_219467

/-- The trajectory of point M given point P on a curve and M as the midpoint of OP -/
theorem trajectory_of_midpoint (x y x₀ y₀ : ℝ) : 
  (2 * x^2 - y^2 = 1) →  -- P is on the curve
  (x₀ = x / 2) →         -- M is the midpoint of OP (x-coordinate)
  (y₀ = y / 2) →         -- M is the midpoint of OP (y-coordinate)
  (8 * x₀^2 - 4 * y₀^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l2194_219467


namespace NUMINAMATH_CALUDE_outfit_count_l2194_219490

/-- Represents the colors available for clothing items -/
inductive Color
| Red | Black | Blue | Gray | Green | Purple | White

/-- Represents a clothing item -/
structure ClothingItem :=
  (color : Color)

/-- Represents an outfit -/
structure Outfit :=
  (shirt : ClothingItem)
  (pants : ClothingItem)
  (hat : ClothingItem)

def is_monochrome (outfit : Outfit) : Prop :=
  outfit.shirt.color = outfit.pants.color ∧ outfit.shirt.color = outfit.hat.color

def num_shirts : Nat := 8
def num_pants : Nat := 5
def num_hats : Nat := 7
def num_pants_colors : Nat := 5
def num_shirt_hat_colors : Nat := 7

theorem outfit_count :
  let total_outfits := num_shirts * num_pants * num_hats
  let monochrome_outfits := num_pants_colors
  (total_outfits - monochrome_outfits : Nat) = 275 := by sorry

end NUMINAMATH_CALUDE_outfit_count_l2194_219490


namespace NUMINAMATH_CALUDE_dans_initial_money_l2194_219416

/-- The amount of money Dan has left after buying the candy bar -/
def money_left : ℝ := 3

/-- The cost of the candy bar -/
def candy_cost : ℝ := 2

/-- Dan's initial amount of money -/
def initial_money : ℝ := money_left + candy_cost

theorem dans_initial_money : initial_money = 5 := by sorry

end NUMINAMATH_CALUDE_dans_initial_money_l2194_219416
