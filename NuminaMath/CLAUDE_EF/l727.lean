import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_B_to_A_is_five_l727_72787

/-- Represents a compound with two elements -/
structure Compound where
  totalWeight : ℝ
  weightB : ℝ

/-- Calculates the ratio of element B to element A in a compound -/
noncomputable def ratioBtoA (c : Compound) : ℝ :=
  c.weightB / (c.totalWeight - c.weightB)

/-- Theorem stating that for the given compound X, the ratio of B to A is 5 -/
theorem ratio_B_to_A_is_five (X : Compound)
    (h1 : X.totalWeight = 330)
    (h2 : X.weightB = 275) :
    ratioBtoA X = 5 := by
  sorry

#check ratio_B_to_A_is_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_B_to_A_is_five_l727_72787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l727_72734

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through a point
def line_through_point (m : ℝ) (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem parabola_line_intersection
  (A B : ℝ × ℝ)  -- Points of intersection
  (h_parabola_A : parabola A.1 A.2)
  (h_parabola_B : parabola B.1 B.2)
  (h_line : ∃ m : ℝ, line_through_point m focus A.1 A.2 ∧ line_through_point m focus B.1 B.2)
  (h_distance : distance A B = 5) :
  (∃ m : ℝ, m = 2 ∨ m = -2) ∧
  (∀ x y : ℝ, line_through_point 2 focus x y ∨ line_through_point (-2) focus x y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l727_72734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_coordinates_sum_intersection_coordinates_sum_is_four_l727_72788

theorem intersection_coordinates_sum : ℝ → Prop := fun s =>
  ∃ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ),
    (∀ i : Fin 4, (y₁ = (x₁ - 2)^2 ∧ x₁ + 3 = (y₁ + 1)^2)) →
    x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = s →
    s = 4

-- The proof goes here
theorem intersection_coordinates_sum_is_four : intersection_coordinates_sum 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_coordinates_sum_intersection_coordinates_sum_is_four_l727_72788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supremum_of_f_l727_72712

noncomputable def f (a b : ℝ) : ℝ := -1 / (2 * a) - 2 / b

theorem supremum_of_f :
  ∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 →
  f a b ≤ -9/2 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ f a b = -9/2 :=
by
  sorry

#check supremum_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supremum_of_f_l727_72712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_equals_11110_l727_72797

/-- A function that checks if a list of four digits is divisible by 11 -/
def is_divisible_by_11 (digits : List Nat) : Prop :=
  digits.length = 4 ∧ (digits.get! 0 + digits.get! 2 - digits.get! 1 - digits.get! 3) % 11 = 0

/-- A function that generates all possible four-digit subsequences from a list -/
def four_digit_subsequences (digits : List Nat) : List (List Nat) :=
  sorry

/-- The property that any four consecutive digits in the number are divisible by 11 -/
def all_four_digits_divisible_by_11 (digits : List Nat) : Prop :=
  ∀ subseq ∈ four_digit_subsequences digits, is_divisible_by_11 subseq

/-- The main theorem -/
theorem sum_of_digits_equals_11110 (digits : List Nat) :
  digits.length = 2020 ∧
  digits.take 4 = [5, 3, 6, 8] ∧
  all_four_digits_divisible_by_11 digits →
  digits.sum = 11110 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_equals_11110_l727_72797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_time_double_speed_theorem_l727_72733

/-- Represents the walking scenario of a brother and sister -/
structure WalkingScenario where
  sister_speed : ℝ
  brother_speed : ℝ
  head_start : ℝ
  catch_up_time : ℝ

/-- Calculates the time it takes for the brother to catch up if he walks twice as fast -/
noncomputable def catch_up_time_double_speed (scenario : WalkingScenario) : ℝ :=
  scenario.head_start / (2 * scenario.brother_speed - scenario.sister_speed)

/-- Theorem stating the catch-up time when the brother walks twice as fast -/
theorem catch_up_time_double_speed_theorem (scenario : WalkingScenario) 
  (h1 : scenario.head_start = 6)
  (h2 : scenario.catch_up_time = 12)
  (h3 : scenario.sister_speed > 0)
  (h4 : scenario.brother_speed > scenario.sister_speed) :
  catch_up_time_double_speed scenario = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_catch_up_time_double_speed_theorem_l727_72733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_triangle_l727_72789

/-- An ellipse with semi-major axis 4 and semi-minor axis 2√3 -/
noncomputable def Ellipse : Type :=
  { a : ℝ // a = 4 } × { b : ℝ // b = 2 * Real.sqrt 3 }

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.1.val^2 + y^2 / e.2.val^2 = 1

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  on_ellipse : e.equation x y

/-- The two foci of the ellipse -/
noncomputable def Ellipse.foci (e : Ellipse) : ℝ × ℝ × ℝ × ℝ :=
  let c := Real.sqrt (e.1.val^2 - e.2.val^2)
  (-c, 0, c, 0)

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

/-- Theorem: If the difference in distances from a point on the ellipse to the foci is 2,
    then the triangle formed by the point and the foci is a right triangle -/
theorem ellipse_right_triangle (e : Ellipse) (p : PointOnEllipse e) :
  let (x₁, y₁, x₂, y₂) := e.foci
  let d₁ := distance p.x p.y x₁ y₁
  let d₂ := distance p.x p.y x₂ y₂
  d₁ - d₂ = 2 →
  d₁^2 = d₂^2 + (distance x₁ y₁ x₂ y₂)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_right_triangle_l727_72789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_187_is_blue_l727_72768

/-- Represents the color of a marble -/
inductive MarbleColor
  | Red
  | Blue
  | Green

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 15 with
  | m => if m < 6 then MarbleColor.Red
         else if m < 11 then MarbleColor.Blue
         else MarbleColor.Green

theorem marble_187_is_blue :
  marbleColor 187 = MarbleColor.Blue := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_187_is_blue_l727_72768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dislike_all_three_l727_72741

/-- Given a population where some dislike TV, books, and games, calculate those who dislike all three. -/
theorem dislike_all_three (total : ℕ) (tv_dislike_ratio : ℚ) (book_dislike_ratio : ℚ) (game_dislike_ratio : ℚ)
  (htotal : total = 1500)
  (htv : tv_dislike_ratio = 2/5)
  (hbook : book_dislike_ratio = 3/20)
  (hgame : game_dislike_ratio = 1/2) :
  (total : ℚ) * tv_dislike_ratio * book_dislike_ratio * game_dislike_ratio = 45 := by
  sorry

#eval (1500 : ℕ) * (2/5 : ℚ) * (3/20 : ℚ) * (1/2 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dislike_all_three_l727_72741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l727_72760

/-- The function g(x) defined by (x^2 - 3x + c) / (x^2 - x - 12) -/
noncomputable def g (c : ℝ) (x : ℝ) : ℝ := (x^2 - 3*x + c) / (x^2 - x - 12)

/-- A function has exactly one vertical asymptote -/
def has_exactly_one_vertical_asymptote (f : ℝ → ℝ) : Prop := sorry

/-- The main theorem: g(x) has exactly one vertical asymptote iff c = -4 or c = -18 -/
theorem g_one_vertical_asymptote (c : ℝ) : 
  has_exactly_one_vertical_asymptote (g c) ↔ (c = -4 ∨ c = -18) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_one_vertical_asymptote_l727_72760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_product_exceeding_million_l727_72748

theorem least_n_for_product_exceeding_million :
  ∀ n : ℕ, n < 12 → (10 : ℝ)^((n * (n + 1) : ℝ) / 18) ≤ 1000000 ∧
  (10 : ℝ)^((12 * 13 : ℝ) / 18) > 1000000 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_product_exceeding_million_l727_72748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_of_f_l727_72711

/-- The function we're differentiating -/
noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x^5)

/-- The third derivative of f -/
noncomputable def f''' (x : ℝ) : ℝ := (107 - 210 * Real.log x) / (x^8)

/-- Theorem stating that f''' is the third derivative of f -/
theorem third_derivative_of_f (x : ℝ) (h : x > 0) : 
  (deriv^[3] f) x = f''' x := by
  sorry

#check third_derivative_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_derivative_of_f_l727_72711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_expression_equality_l727_72782

/-- Given an angle α in the second quadrant, with a point P (x, √5) on its terminal side,
    and cosα = (√2/4)x, prove that 4cos(α + π/2) - 3tanα = √15 - √10 -/
theorem angle_expression_equality (α : Real) (x : Real) : 
  α ∈ Set.Ioo (π/2) π → -- α is in the second quadrant
  (x, Real.sqrt 5) ∈ {p : ℝ × ℝ | p.1 * Real.cos α - p.2 * Real.sin α = 0} → -- P (x, √5) is on the terminal side of α
  Real.cos α = (Real.sqrt 2 / 4) * x →
  4 * Real.cos (α + π/2) - 3 * Real.tan α = Real.sqrt 15 - Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_expression_equality_l727_72782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l727_72765

def num_children : ℕ := 6

noncomputable def prob_at_least_three_boys_or_half_girls : ℚ :=
  let total_outcomes := 2^num_children
  let at_least_three_boys := (Finset.range 4).sum (λ i => 
    (num_children.choose (num_children - i)) * (1 / 2)^num_children)
  at_least_three_boys

theorem probability_theorem :
  prob_at_least_three_boys_or_half_girls = 21/32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l727_72765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_l727_72731

/-- Represents a right circular cone --/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a spherical marble --/
structure Marble where
  radius : ℝ

/-- Calculates the volume of a cone --/
noncomputable def coneVolume (c : Cone) : ℝ := (1/3) * Real.pi * c.radius^2 * c.height

/-- Calculates the volume of a sphere --/
noncomputable def sphereVolume (m : Marble) : ℝ := (4/3) * Real.pi * m.radius^3

/-- Theorem: The ratio of liquid level rise in narrow cone to wide cone is 2:1 --/
theorem liquid_level_rise_ratio (k : ℝ) :
  let narrowCone : Cone := { radius := 4, height := 2 * k }
  let wideCone : Cone := { radius := 8, height := k }
  let marble : Marble := { radius := 1 }
  let narrowVolumeIncrease := 2 * sphereVolume marble
  let wideVolumeIncrease := 2 * sphereVolume marble
  let narrowNewHeight := (coneVolume narrowCone + narrowVolumeIncrease) / ((1/3) * Real.pi * narrowCone.radius^2)
  let wideNewHeight := (coneVolume wideCone + wideVolumeIncrease) / ((1/3) * Real.pi * wideCone.radius^2)
  let narrowRise := narrowNewHeight - narrowCone.height
  let wideRise := wideNewHeight - wideCone.height
  narrowRise / wideRise = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_level_rise_ratio_l727_72731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l727_72746

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6) + 2 * (Real.cos x) ^ 2 - 1

theorem f_properties :
  (∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M ∧ M = 1) ∧
  (∀ (x : ℝ), f x = 1 ↔ ∃ (k : ℤ), x = k * Real.pi + Real.pi / 6) ∧
  (∀ (α : ℝ), Real.pi / 4 < α → α < Real.pi / 2 → f α = 4 / 5 →
    Real.cos (2 * α) = (-3 * Real.sqrt 3 + 4) / 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l727_72746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_at_negative_1860_degrees_f_special_case_l727_72722

open Real

/-- The function f(α) as defined in the problem -/
noncomputable def f (α : ℝ) : ℝ :=
  (sin (π/2 - α) * cos (10*π - α) * tan (-α + 3*π)) /
  (tan (π + α) * sin (5*π/2 + α))

/-- Theorem stating the simplification of f(α) -/
theorem f_simplification (α : ℝ) : f α = -cos α := by sorry

/-- Theorem for the value of f(-1860°) -/
theorem f_at_negative_1860_degrees : f (-1860 * π / 180) = -1/2 := by sorry

/-- Theorem for the value of f(α) under specific conditions -/
theorem f_special_case (α : ℝ) 
  (h1 : α > 0) (h2 : α < π/2) (h3 : sin (α - π/6) = 1/3) : 
  f α = (1 - 2 * sqrt 6) / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_at_negative_1860_degrees_f_special_case_l727_72722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_calculations_l727_72703

/-- Tree data structure -/
structure TreeData where
  root_area : ℝ
  volume : ℝ

/-- Forest data -/
structure ForestData where
  samples : List TreeData
  total_root_area : ℝ
  sum_root_area : ℝ
  sum_volume : ℝ
  sum_root_area_squared : ℝ
  sum_volume_squared : ℝ
  sum_root_area_volume : ℝ

/-- Theorem stating the correctness of calculations based on forest data -/
theorem forest_calculations (forest : ForestData) 
  (h1 : forest.samples.length = 10)
  (h2 : forest.sum_root_area = 0.6)
  (h3 : forest.sum_volume = 3.9)
  (h4 : forest.sum_root_area_squared = 0.038)
  (h5 : forest.sum_volume_squared = 1.6158)
  (h6 : forest.sum_root_area_volume = 0.2474)
  (h7 : forest.total_root_area = 186) :
  let avg_root_area := forest.sum_root_area / 10
  let avg_volume := forest.sum_volume / 10
  let correlation := (forest.sum_root_area_volume - 10 * avg_root_area * avg_volume) / 
    (Real.sqrt ((forest.sum_root_area_squared - 10 * avg_root_area ^ 2) * 
    (forest.sum_volume_squared - 10 * avg_volume ^ 2)))
  let estimated_total_volume := (avg_volume / avg_root_area) * forest.total_root_area
  (avg_root_area = 0.06 ∧ avg_volume = 0.39) ∧
  (abs (correlation - 0.97) < 0.01) ∧
  (abs (estimated_total_volume - 1209) < 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_forest_calculations_l727_72703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_proof_l727_72759

/-- The minimum sum of distances from any point on the x-axis to the fixed points (0,2) and (1,1) -/
noncomputable def min_sum_distances : ℝ := Real.sqrt 10

/-- The first fixed point -/
def point1 : ℝ × ℝ := (0, 2)

/-- The second fixed point -/
def point2 : ℝ × ℝ := (1, 1)

theorem min_sum_distances_proof :
  ∀ x : ℝ,
  Real.sqrt ((x - point1.1)^2 + point1.2^2) +
  Real.sqrt ((x - point2.1)^2 + point2.2^2) ≥
  min_sum_distances :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_proof_l727_72759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l727_72708

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0,
    and eccentricity e = √5/2, prove that its asymptotes are y = ±(1/2)x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := (Real.sqrt 5) / 2
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let eccentricity := fun (c : ℝ) ↦ c / a = e
  let asymptotes := fun (x y : ℝ) ↦ y = (1/2) * x ∨ y = -(1/2) * x
  ∃ c, eccentricity c → (∀ x y, asymptotes x y ↔ hyperbola x y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l727_72708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_120_miles_l727_72779

/-- The distance Daniel drives back from work every day -/
noncomputable def distance : ℝ := sorry

/-- The speed at which Daniel drives on Sunday -/
noncomputable def sunday_speed : ℝ := sorry

/-- The time it takes Daniel to drive back from work on Sunday -/
noncomputable def sunday_time : ℝ := distance / sunday_speed

/-- The time it takes Daniel to drive the first 32 miles on Monday -/
noncomputable def monday_first_part_time : ℝ := 32 / (2 * sunday_speed)

/-- The time it takes Daniel to drive the rest of the way on Monday -/
noncomputable def monday_second_part_time : ℝ := (distance - 32) / (sunday_speed / 2)

/-- The total time it takes Daniel to drive back from work on Monday -/
noncomputable def monday_time : ℝ := monday_first_part_time + monday_second_part_time

/-- Theorem stating that the distance Daniel drives back from work every day is 120 miles -/
theorem distance_is_120_miles :
  (monday_time = 1.6 * sunday_time) → distance = 120 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_120_miles_l727_72779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camel_cost_l727_72743

/-- The cost of animals in rupees -/
structure AnimalCosts where
  camel : ℚ
  horse : ℚ
  ox : ℚ
  elephant : ℚ

/-- The conditions of the problem -/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  26 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 170000

/-- The theorem stating the cost of a camel -/
theorem camel_cost (costs : AnimalCosts) 
  (h : problem_conditions costs) : 
  ∃ n : ℕ, costs.camel = 4184 + (62 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_camel_cost_l727_72743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_negative_sum_l727_72739

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

noncomputable def sequence_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem smallest_negative_sum :
  let a₁ := (7 : ℝ)
  let d := (-2 : ℝ)
  ∀ k : ℕ, k < 9 → sequence_sum a₁ d k ≥ 0 ∧
  sequence_sum a₁ d 9 < 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_negative_sum_l727_72739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l727_72793

def is_finite (S : Set ℝ) : Prop :=
  ∃ (n : ℕ) (l : Finset ℝ), S = l.toSet ∧ l.card = n

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x - 1 - f x) = f x - 1 - x) ∧
  is_finite {y | ∃ x : ℝ, x ≠ 0 ∧ y = f x / x}

theorem unique_function_satisfying_conditions :
  ∃! f : ℝ → ℝ, satisfies_conditions f ∧ f = id := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l727_72793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l727_72728

/-- A chessboard is represented by its dimensions -/
structure Chessboard where
  m : ℕ+
  n : ℕ+

/-- A player in the game -/
inductive Player
  | First
  | Second

/-- The result of a game -/
inductive GameResult
  | FirstWins
  | SecondWins

/-- The game state -/
structure GameState where
  board : Chessboard
  currentPlayer : Player
  visitedSquares : Set (ℕ × ℕ)

/-- A strategy is a function that takes a game state and returns a move -/
def Strategy := GameState → Option (ℕ × ℕ)

/-- Initialize a new game state -/
def GameState.init (board : Chessboard) : GameState :=
  { board := board,
    currentPlayer := Player.First,
    visitedSquares := {(1, 1)} }

/-- Play the game given two strategies -/
def playGame (first_strategy : Strategy) (second_strategy : Strategy) (initial_state : GameState) : GameResult :=
  sorry

/-- The theorem stating the winning condition for the first player -/
theorem first_player_winning_strategy (board : Chessboard) :
  (∃ (s : Strategy), ∀ (opponent_strategy : Strategy),
    playGame s opponent_strategy (GameState.init board) = GameResult.FirstWins) ↔
  (board.m > 1 ∨ board.n > 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_winning_strategy_l727_72728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_symmetry_l727_72753

theorem sin_graph_symmetry :
  ∀ h : ℝ,
  let f (x : ℝ) := Real.sin (2 * x + π / 3)
  let g (x : ℝ) := f (x - π / 12)
  g (-(π / 12) + h) = -g (-(π / 12) - h) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_symmetry_l727_72753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_coordinates_l727_72755

theorem rotation_coordinates (θ : ℝ) (h : ∀ k : ℤ, θ ≠ k * π + π / 2) :
  let A : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let B : ℝ × ℝ := (-(A.2), A.1)  -- Clockwise rotation by 3π/2
  B = (-Real.sin θ, Real.cos θ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotation_coordinates_l727_72755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_eleven_l727_72785

/-- The sum of all integers from -29 to 79 (inclusive) that are divisible by 11 is equal to 275. -/
theorem sum_divisible_by_eleven : 
  (Finset.filter (fun x => x % 11 = 0) (Finset.range (80 + 29))).sum 
    (fun x => x - 29) = 275 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_divisible_by_eleven_l727_72785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l727_72784

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then Real.log x / Real.log (1/3)
  else 3^x

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Iio 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l727_72784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pool_volume_l727_72767

/-- Represents a circular swimming pool with linearly varying depth -/
structure CircularPool where
  diameter : ℝ
  shallow_depth : ℝ
  deep_depth : ℝ

/-- Calculates the volume of a circular pool with linearly varying depth -/
noncomputable def pool_volume (pool : CircularPool) : ℝ :=
  Real.pi * (pool.diameter / 2)^2 * (pool.shallow_depth + pool.deep_depth) / 2

/-- Theorem stating the volume of the specific pool described in the problem -/
theorem specific_pool_volume :
  let pool : CircularPool := ⟨60, 3, 15⟩
  pool_volume pool = 8100 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pool_volume_l727_72767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l727_72710

/-- For all real α, the given trigonometric expression simplifies to -1. -/
theorem trig_simplification (α : ℝ) : 
  (Real.sin (2 * Real.pi - α) * Real.cos (3 * Real.pi + α) * Real.cos ((3 * Real.pi) / 2 - α)) / 
  (Real.sin (-Real.pi + α) * Real.sin (3 * Real.pi - α) * Real.cos (-α - Real.pi)) = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_simplification_l727_72710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_green_marbles_l727_72762

def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def total_marbles : ℕ := green_marbles + purple_marbles
def num_trials : ℕ := 7
def num_green_chosen : ℕ := 3

theorem probability_three_green_marbles :
  (Nat.choose num_trials num_green_chosen : ℚ) *
  ((green_marbles : ℚ) / total_marbles) ^ num_green_chosen *
  ((purple_marbles : ℚ) / total_marbles) ^ (num_trials - num_green_chosen) =
  17210408 / 68343750 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_green_marbles_l727_72762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equals_pi_over_48_probability_is_volume_ratio_l727_72766

/-- The probability of a randomly selected point (x, y, z) satisfying x^2 + y^2 + z^2 ≤ 1,
    given that -2 ≤ x ≤ 2, -2 ≤ y ≤ 2, and -2 ≤ z ≤ 2 -/
noncomputable def probability_in_unit_sphere_given_cube : ℝ := 
  Real.pi / 48

/-- The volume of the cube with side length 4 -/
def cube_volume : ℝ := 64

/-- The volume of the unit sphere -/
noncomputable def sphere_volume : ℝ := 4 * Real.pi / 3

/-- The probability is equal to π/48 -/
theorem probability_equals_pi_over_48 :
  probability_in_unit_sphere_given_cube = Real.pi / 48 := by
  rfl

/-- The probability is the ratio of the sphere volume to the cube volume -/
theorem probability_is_volume_ratio :
  probability_in_unit_sphere_given_cube = sphere_volume / cube_volume := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equals_pi_over_48_probability_is_volume_ratio_l727_72766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l727_72769

noncomputable def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (3 * x + φ)

theorem function_properties (A : ℝ) (φ : ℝ) 
  (h1 : A > 0) (h2 : 0 < φ) (h3 : φ < π) (h4 : f A φ (π/6) = 4) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f A φ (x + T) = f A φ x ∧ 
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f A φ (x + T') = f A φ x) → T ≤ T') ∧
  (∀ (x : ℝ), f A φ x = 4 * Real.sin (3 * x + π/2)) ∧
  (∀ (α : ℝ), f A φ (α + π/3) = 2 * Real.sqrt 3 → 
    Real.sin α = (Real.sqrt (4 - 2 * Real.sqrt 3)) / 2 ∨ 
    Real.sin α = -(Real.sqrt (4 - 2 * Real.sqrt 3)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l727_72769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_item_price_l727_72783

theorem magical_item_price (discounted_price_after_tax : ℝ) 
  (discount_factor : ℝ) (tax_rate : ℝ) (regular_price : ℝ) :
  discounted_price_after_tax = 8 →
  discount_factor = (1 / 5 : ℝ) →
  tax_rate = (15 / 100 : ℝ) →
  discounted_price_after_tax = discount_factor * regular_price * (1 + tax_rate) →
  ‖regular_price - 34.78‖ < 0.01 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_item_price_l727_72783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flowchart_output_is_13_l727_72796

def flowchart_result (initial_S : ℕ) (initial_n : ℕ) (loop_condition : ℕ → Bool) 
  (update_S : ℕ → ℕ → ℕ) (update_n : ℕ → ℕ) : ℕ :=
  let rec loop (S : ℕ) (n : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then S
    else if loop_condition n then
      loop (update_S S n) (update_n n) (fuel - 1)
    else
      S
  loop initial_S initial_n 100 -- Use a sufficiently large fuel value

theorem flowchart_output_is_13 : 
  flowchart_result 1 1 (λ n => n ≤ 3) (λ S n => S + 2*n) (λ n => n + 1) = 13 := by
  sorry

#eval flowchart_result 1 1 (λ n => n ≤ 3) (λ S n => S + 2*n) (λ n => n + 1)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flowchart_output_is_13_l727_72796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l727_72790

/-- The area of a trapezoid with given parameters -/
noncomputable def trapezoidArea (lateralSide : ℝ) (distance1 : ℝ) (distance2 : ℝ) : ℝ :=
  (1 / 2) * (distance1 + distance2) * lateralSide

/-- Theorem: The area of the specific trapezoid is 18 -/
theorem specific_trapezoid_area :
  trapezoidArea 3 5 7 = 18 := by
  -- Unfold the definition of trapezoidArea
  unfold trapezoidArea
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_trapezoid_area_l727_72790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_to_side_ratio_formula_l727_72775

/-- A cylinder with a square rectangular side unfolding -/
structure SquareUnfoldCylinder where
  radius : ℝ
  height : ℝ
  square_unfolding : height = 2 * Real.pi * radius

/-- The ratio of surface area to side area for a SquareUnfoldCylinder -/
noncomputable def surface_to_side_ratio (c : SquareUnfoldCylinder) : ℝ :=
  (2 * Real.pi * c.radius^2 + 2 * Real.pi * c.radius * c.height) / (2 * Real.pi * c.radius * c.height)

/-- Theorem: The ratio of surface area to side area for a SquareUnfoldCylinder is (1 + 2π) / (2π) -/
theorem surface_to_side_ratio_formula (c : SquareUnfoldCylinder) :
  surface_to_side_ratio c = (1 + 2 * Real.pi) / (2 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_to_side_ratio_formula_l727_72775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_coloring_probability_l727_72776

/-- Represents a 4-by-4 grid where each cell can be colored red or blue -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 3-by-3 subgrid starting at (i, j) is all red -/
def is_red_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ (x y : Fin 3), g (i + x) (j + y) = true

/-- Checks if a grid contains a 3-by-3 red square -/
def has_red_3x3 (g : Grid) : Prop :=
  ∃ (i j : Fin 2), is_red_3x3 g i j

/-- The probability of a single cell being red -/
noncomputable def p_red : ℝ := 1/2

/-- The total number of possible grid colorings -/
def total_colorings : ℕ := 2^16

/-- The number of colorings without a 3-by-3 red square -/
def colorings_without_red_3x3 : ℕ := 65056

theorem grid_coloring_probability :
  (colorings_without_red_3x3 : ℝ) / total_colorings = 271 / 273 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_coloring_probability_l727_72776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_calculation_l727_72772

/-- Represents the walking scenario with given parameters -/
structure WalkingScenario where
  initial_distance : ℚ  -- in feet
  initial_time : ℚ      -- in minutes
  remaining_distance : ℚ -- in yards

/-- Calculates the time needed to walk a given distance at a given rate -/
def time_to_walk (rate : ℚ) (distance : ℚ) : ℚ :=
  distance / rate

/-- Converts yards to feet -/
def yards_to_feet (yards : ℚ) : ℚ :=
  yards * 3

theorem walking_time_calculation (scenario : WalkingScenario) :
  let rate := scenario.initial_distance / scenario.initial_time
  let remaining_feet := yards_to_feet scenario.remaining_distance
  time_to_walk rate remaining_feet = 150 :=
by
  -- Unfold definitions
  simp [time_to_walk, yards_to_feet]
  -- The rest of the proof would go here
  sorry

#eval time_to_walk (80 / 40) (yards_to_feet 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_walking_time_calculation_l727_72772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_weight_is_one_pound_l727_72773

/-- Represents the weight of food in pounds -/
noncomputable def total_weight : ℝ := 7

/-- Represents the weight of brie cheese in ounces -/
noncomputable def brie_weight : ℝ := 8

/-- Represents the weight of tomatoes in pounds -/
noncomputable def tomato_weight : ℝ := 1

/-- Represents the weight of zucchini in pounds -/
noncomputable def zucchini_weight : ℝ := 2

/-- Represents the weight of chicken breasts in pounds -/
noncomputable def chicken_weight : ℝ := 1.5

/-- Represents the weight of raspberries in ounces -/
noncomputable def raspberry_weight : ℝ := 8

/-- Represents the weight of blueberries in ounces -/
noncomputable def blueberry_weight : ℝ := 8

/-- Conversion factor from ounces to pounds -/
noncomputable def ounces_per_pound : ℝ := 16

/-- Calculates the weight of bread in pounds -/
noncomputable def bread_weight : ℝ :=
  total_weight - (brie_weight / ounces_per_pound + tomato_weight + zucchini_weight + chicken_weight +
    raspberry_weight / ounces_per_pound + blueberry_weight / ounces_per_pound)

/-- Theorem stating that the weight of bread Melanie buys is 1 pound -/
theorem bread_weight_is_one_pound : bread_weight = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bread_weight_is_one_pound_l727_72773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l727_72771

theorem solve_exponential_equation :
  ∃ x : ℝ, 2 * (5 : ℝ) ^ x = 1250 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l727_72771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coords_l727_72713

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  h_A : A = (13, 11)
  h_B : B = (5, -1)
  h_D : D = (2, 7)
  h_isosceles : dist A B = dist A C
  h_altitude : (D.1 - A.1) * (C.1 - B.1) + (D.2 - A.2) * (C.2 - B.2) = 0

/-- The coordinates of point C in the given triangle -/
def point_C (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem stating that the coordinates of point C are (-1, 15) -/
theorem point_C_coords (t : Triangle) : point_C t = (-1, 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_C_coords_l727_72713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_l727_72719

/-- Represents a quadratic polynomial ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates a quadratic polynomial at a given x -/
noncomputable def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Represents the rational function r(x)/s(x) -/
structure RationalFunction where
  r : QuadraticPolynomial
  s : QuadraticPolynomial

/-- Evaluates the rational function at a given x -/
noncomputable def RationalFunction.eval (f : RationalFunction) (x : ℝ) : ℝ :=
  f.r.eval x / f.s.eval x

/-- Main theorem statement -/
theorem rational_function_value (f : RationalFunction) :
  (∀ x, x ≠ 3 → x ≠ -4 → f.eval x = (-3 * (x - 1)) / (x - 3)) →  -- Simplified form
  (∀ x, x ≠ 3 → x ≠ -4 → f.eval x ≠ 0 → f.s.eval x ≠ 0) →        -- Well-defined except at x = 3 and x = -4
  (∃ C, ∀ x, x ≠ 3 → x ≠ -4 → |x| > C → |f.eval x + 3| < 1/1000) →  -- Horizontal asymptote at y = -3
  f.s.eval 3 = 0 →                                               -- Vertical asymptote at x = 3
  f.r.eval (-4) = 0 ∧ f.s.eval (-4) = 0 →                        -- Hole at x = -4
  f.eval (-1) = 3/2 :=                                           -- Conclusion
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_l727_72719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_condition_l727_72792

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^3 + 1 else x^2 - a*x

-- State the theorem
theorem function_composition_condition (a : ℝ) :
  (f a (f a 0) = -2) → a = 3 := by
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_condition_l727_72792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_move_inside_25_reflections_cannot_move_inside_24_reflections_l727_72704

-- Define the circle and point A
def circle_radius : ℝ := 1
def point_A_distance : ℝ := 50

-- Define a reflection operation
def reflect (point : ℝ × ℝ) (line : ℝ × ℝ → Bool) : ℝ × ℝ := sorry

-- Define a function to check if a point is inside the circle
def is_inside_circle (point : ℝ × ℝ) : Bool := sorry

-- Theorem for part (a)
theorem can_move_inside_25_reflections :
  ∃ (reflections : List (ℝ × ℝ → Bool)),
    reflections.length = 25 ∧
    is_inside_circle (reflections.foldl reflect (point_A_distance, 0)) := by sorry

-- Theorem for part (b)
theorem cannot_move_inside_24_reflections :
  ∀ (reflections : List (ℝ × ℝ → Bool)),
    reflections.length = 24 →
    ¬is_inside_circle (reflections.foldl reflect (point_A_distance, 0)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_move_inside_25_reflections_cannot_move_inside_24_reflections_l727_72704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l727_72794

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x - 16 * y^2 - 32 * y = 144

/-- The distance between the foci of the hyperbola -/
noncomputable def foci_distance : ℝ := Real.sqrt 3425 / 72

/-- Theorem stating that the distance between the foci of the given hyperbola is equal to foci_distance -/
theorem hyperbola_foci_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    hyperbola_equation x₁ y₁ ∧ 
    hyperbola_equation x₂ y₂ ∧ 
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = foci_distance^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l727_72794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l727_72799

theorem cos_alpha_minus_pi_fourth (α : ℝ) 
  (h1 : Real.cos α = 5/13) 
  (h2 : α ∈ Set.Ioo (3*Real.pi/2) (2*Real.pi)) : 
  Real.cos (α - Real.pi/4) = -7*Real.sqrt 2/26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l727_72799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l727_72725

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3) + 2

theorem min_omega_value (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x, f ω x = f ω (x + 4 * Real.pi / 3)) : 
  ω = 3 / 2 ∧ ∀ ω' > 0, (∀ x, f ω' x = f ω' (x + 4 * Real.pi / 3)) → ω' ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l727_72725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l727_72774

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain (x : ℝ) : Prop := x < 0 ∨ x > 0

-- f is monotonically increasing on (-∞, 0)
axiom f_increasing_neg : ∀ x y, x < y → y < 0 → f x < f y

-- Functional equation for f
axiom f_equation : ∀ a b, domain a → domain b → f (a * b) = f a + f b - 1

-- Existence of x satisfying the inequality
axiom exists_x : ∃ x : ℝ, x > 1 ∧ ∀ m : ℝ, f (m * x) - f (Real.log x) > f (-1) - 1

-- Theorem to prove
theorem range_of_m :
  ∀ m : ℝ, (∃ x : ℝ, x > 1 ∧ f (m * x) - f (Real.log x) > f (-1) - 1) ↔ 
  (m > 0 ∧ m < 1 / Real.exp 1) ∨ (m < 0 ∧ m > -1 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l727_72774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l727_72732

/-- The star operation defined as (a ★ b) = √(a+b) / √(a-b) -/
noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b) / Real.sqrt (a - b)

/-- Theorem: If x ★ 16 = 2, then x = 80/3 -/
theorem star_equation_solution (x : ℝ) (h : star x 16 = 2) : x = 80 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_equation_solution_l727_72732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_of_one_equals_eleven_point_twenty_five_l727_72735

-- Define the functions u and v
def u (x : ℝ) : ℝ := 4 * x - 9

noncomputable def v (x : ℝ) : ℝ := 
  let y := (x + 9) / 4  -- This is u⁻¹(x)
  y^2 + 4 * y - 5

-- State the theorem
theorem v_of_one_equals_eleven_point_twenty_five : v 1 = 11.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_of_one_equals_eleven_point_twenty_five_l727_72735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_tree_probability_l727_72720

/-- Right-angled trapezoid with given dimensions -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- Right-angled triangle with given dimensions -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Calculate the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  (t.base1 + t.base2) * t.height / 2

/-- Calculate the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  t.base * t.height / 2

/-- The probability of a random point in the trapezoid being inside the triangle -/
noncomputable def probability (trap : Trapezoid) (tri : Triangle) : ℝ :=
  triangleArea tri / trapezoidArea trap

/-- Theorem stating the probability for the given dimensions -/
theorem tea_tree_probability (trap : Trapezoid) (tri : Triangle) 
    (h1 : trap.base1 = 10) (h2 : trap.base2 = 20) (h3 : trap.height = 10)
    (h4 : tri.base = 8) (h5 : tri.height = 5) :
    probability trap tri = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tea_tree_probability_l727_72720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l727_72770

theorem quadratic_equation_roots (x : ℝ) :
  let equation := 2 * x^2 - 5 * x - 3
  let root1 := 3
  let root2 := -1/2
  (equation = 0 ↔ x = root1 ∨ x = root2) ∧
  (2 = 2) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_roots_l727_72770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hill_run_time_l727_72709

/-- The time taken to run up and down a hill -/
noncomputable def hill_run_time (ground_length : ℝ) (incline_angle : ℝ) (uphill_speed : ℝ) (downhill_speed : ℝ) : ℝ :=
  let actual_length := ground_length / (Real.cos incline_angle)
  let time_up := actual_length / uphill_speed
  let time_down := actual_length / downhill_speed
  time_up + time_down

/-- Theorem stating the approximate time to run up and down the specific hill -/
theorem specific_hill_run_time :
  let ground_length := (900 : ℝ)
  let incline_angle := 30 * Real.pi / 180  -- Convert degrees to radians
  let uphill_speed := (9 : ℝ)
  let downhill_speed := (12 : ℝ)
  abs (hill_run_time ground_length incline_angle uphill_speed downhill_speed - 202.07) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hill_run_time_l727_72709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_new_average_l727_72756

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  initialAverage : ℚ
  inningNumber : ℕ
  runsScored : ℕ
  averageIncrease : ℚ

/-- Calculates the new average after an inning -/
noncomputable def newAverage (stats : BatsmanStats) : ℚ :=
  (stats.initialAverage * (stats.inningNumber - 1 : ℚ) + stats.runsScored) / stats.inningNumber

/-- Theorem: The batsman's new average is 19 runs -/
theorem batsman_new_average (stats : BatsmanStats) 
  (h1 : stats.inningNumber = 16)
  (h2 : stats.runsScored = 64)
  (h3 : stats.averageIncrease = 3)
  (h4 : newAverage stats = stats.initialAverage + stats.averageIncrease) :
  newAverage stats = 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_new_average_l727_72756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separate_l727_72736

def line (θ : ℝ) (x y : ℝ) : Prop := x * Real.cos θ + y - 1 = 0

def my_circle (x y : ℝ) : Prop := 2 * x^2 + 2 * y^2 = 1

def is_separate (θ : ℝ) : Prop :=
  ∀ x y : ℝ, line θ x y → ¬ my_circle x y

theorem line_circle_separate (θ : ℝ) (h : ∀ k : ℤ, θ ≠ k * Real.pi) :
  is_separate θ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_separate_l727_72736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_l727_72730

theorem square_division (k m n : ℕ) : 
  (∃ x : ℕ, m^2 = x * k + n^2) ↔ (m - n) * (m + n) % k = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_l727_72730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_rectangle_equals_pi_minus_sqrt5_minus_2_l727_72727

noncomputable def areaOutsideRectangle : ℝ :=
  Real.pi - (Real.sqrt 5 - 2)

def rectangleLength : ℝ := 2

def rectangleWidth : ℝ := 1

def circleRadius : ℝ := 1

theorem area_outside_rectangle_equals_pi_minus_sqrt5_minus_2 :
  areaOutsideRectangle = Real.pi - (Real.sqrt 5 - 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_outside_rectangle_equals_pi_minus_sqrt5_minus_2_l727_72727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_local_minimum_f_minus_g_monotonic_f_minus_g_greater_h_l727_72721

noncomputable section

-- Define the functions
def g (x : ℝ) : ℝ := 2/x + Real.log x
def f (m : ℝ) (x : ℝ) : ℝ := m*x - (m-2)/x
def h (x : ℝ) : ℝ := 2*Real.exp 1/x

-- Theorem statements
theorem g_local_minimum : 
  ∃ ε > 0, ∀ x, x ∈ Set.Ioo (2 - ε) (2 + ε) → g x ≥ g 2 :=
sorry

theorem f_minus_g_monotonic (m : ℝ) : 
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f m x - g x < f m y - g y) ∨
  (∀ x y, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f m x - g x > f m y - g y) ↔ 
  m ≤ 0 ∨ m ≥ 1 :=
sorry

theorem f_minus_g_greater_h (m : ℝ) :
  (∃ x₀, x₀ ∈ Set.Icc 1 (Real.exp 1) ∧ f m x₀ - g x₀ > h x₀) ↔ 
  m > 4 * Real.exp 1 / (Real.exp 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_local_minimum_f_minus_g_monotonic_f_minus_g_greater_h_l727_72721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_power_four_thirds_nonnegative_sum_of_sqrt_inequality_l727_72724

-- Define the function f(x) = ln x - 2/x
noncomputable def f (x : ℝ) := Real.log x - 2 / x

-- Theorem 1: f(x) has a root in the interval (2, 3)
theorem f_has_root_in_interval : ∃ x ∈ Set.Ioo 2 3, f x = 0 := by sorry

-- Theorem 2: For all real x, x^(4/3) ≥ 0
theorem power_four_thirds_nonnegative (x : ℝ) : x^(4/3) ≥ 0 := by sorry

-- Theorem 3: For a > 0, b > 0, and a + b = 1, √a + √b ≤ √2
theorem sum_of_sqrt_inequality {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_root_in_interval_power_four_thirds_nonnegative_sum_of_sqrt_inequality_l727_72724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l727_72751

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h_pos : ∀ i, 0 < a i) 
  (h_ineq : ∀ i, i ∈ Finset.range (n-1) → a i * a (i+2) ≤ (a (i+1))^2) 
  (h_n : 1 < n) : 
  (Finset.sum (Finset.range (n+1)) a / (n+1 : ℝ)) * 
  (Finset.sum (Finset.range (n-1)) (λ i => a (i+1)) / (n-1 : ℝ)) ≥ 
  (Finset.sum (Finset.range n) a / n) * 
  (Finset.sum (Finset.range n) (λ i => a (i+1)) / n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l727_72751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_calculation_l727_72791

def voltage : ℂ := 3 + 4 * Complex.I
def impedance : ℂ := 2 + 5 * Complex.I

theorem current_calculation :
  voltage / impedance = 26/29 - 7/29 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_calculation_l727_72791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_encyclopedia_payment_ways_l727_72738

theorem encyclopedia_payment_ways :
  let total_cost : ℕ := 270
  let small_note : ℕ := 20
  let large_note : ℕ := 50
  (Finset.filter (fun (p : ℕ × ℕ) => small_note * p.1 + large_note * p.2 = total_cost) (Finset.product (Finset.range 14) (Finset.range 6))).card = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_encyclopedia_payment_ways_l727_72738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_greater_than_power_of_three_l727_72714

def x : ℕ → ℝ
  | 0 => 5  -- Add this case to handle Nat.zero
  | 1 => 5
  | (k+1) => (x k)^2 - 3*(x k) + 3

theorem x_greater_than_power_of_three (k : ℕ) (h : k > 0) :
  x k > 3^(2^(k-1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_greater_than_power_of_three_l727_72714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_maximum_l727_72706

noncomputable def f (x : ℝ) := Real.log x - x^2 + x + 2

theorem f_monotonicity_and_maximum (a : ℝ) (ha : a > 0) :
  (MonotoneOn f (Set.Ioo 0 1)) ∧
  (StrictAntiOn f (Set.Ioi 1)) ∧
  (∀ x ∈ Set.Ioc 0 a, f x ≤ if a ≤ 1 then f a else f 1) ∧
  (if a ≤ 1 then f a else f 1) = 
    if a ≤ 1 then Real.log a - a^2 + a + 2 else 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_maximum_l727_72706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petr_rectangle_area_l727_72749

/-- Represents the length of Petr's rectangle in cm -/
def d : ℝ := sorry

/-- The width of both Petr's and Radka's rectangles in cm -/
def width : ℝ := 2

/-- The perimeter of the combined rectangle in cm -/
def combined_perimeter : ℝ := 63

/-- The area of Petr's rectangle in cm² -/
def petr_area : ℝ := d * width

theorem petr_rectangle_area :
  (6 * d + 16 = combined_perimeter) →
  petr_area = 17 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_petr_rectangle_area_l727_72749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_equals_sqrt35_l727_72750

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, (1 / 2) * t)

-- Define the circle C
noncomputable def circle_C (θ : ℝ) : ℝ × ℝ := (2 + 3 * Real.cos θ, 3 * Real.sin θ)

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- State the theorem
theorem distance_sum_equals_sqrt35 :
  ∃ (A B : ℝ × ℝ) (t₁ t₂ θ₁ θ₂ : ℝ),
    A = line_l t₁ ∧ A = circle_C θ₁ ∧
    B = line_l t₂ ∧ B = circle_C θ₂ ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - point_F.1)^2 + (A.2 - point_F.2)^2) +
    Real.sqrt ((B.1 - point_F.1)^2 + (B.2 - point_F.2)^2) =
    Real.sqrt 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_equals_sqrt35_l727_72750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_two_three_equals_seven_thirtieths_l727_72745

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ
  repeatingLength : ℕ+

/-- Converts a repeating decimal to a rational number -/
def toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + (d.repeatingPart : ℚ) / ((10 ^ (d.repeatingLength : ℕ) - 1) : ℚ)

/-- The repeating decimal 0.2̇3̇ -/
def d : RepeatingDecimal :=
  { integerPart := 0
  , repeatingPart := 23
  , repeatingLength := 2 }

/-- Theorem stating that 0.2̇3̇ is equal to 7/30 -/
theorem repeating_decimal_two_three_equals_seven_thirtieths :
  toRational d = 7 / 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_two_three_equals_seven_thirtieths_l727_72745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_log_l727_72778

/-- Given a function f(x) = 4a^(x-9) - 1 where a > 0 and a ≠ 1,
    if f(m) = n, then log_m(n) = 1/2 -/
theorem fixed_point_log (a m n : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  4 * a^(m - 9) - 1 = n → Real.log n / Real.log m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_log_l727_72778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l727_72786

open BigOperators

def f (n : ℕ+) : ℚ := ∑ i in Finset.range (3*n), 1 / (i + 1 : ℚ)

theorem f_difference (n : ℕ+) : 
  f (n + 1) - f n = 1 / (3*n : ℚ) + 1 / ((3*n + 1 : ℕ) : ℚ) + 1 / ((3*n + 2 : ℕ) : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_difference_l727_72786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l727_72737

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log x + a * x - x^2

-- State the theorem
theorem f_properties (a : ℝ) (h : 0 < a ∧ a ≤ 1) :
  -- Part I: Tangent line equation when a = 1/2
  (let tangent_line (x : ℝ) := -1/2 * x
   ∀ x, (a = 1/2) → (f (1/2) 1 + (deriv (f (1/2))) 1 * (x - 1) = tangent_line x)) ∧
  -- Part II: Maximum value of t-s
  (∃ s t : ℝ, s < t ∧
    (∀ x ∈ Set.Ioo s t, (deriv (f a)) x > 0) ∧
    (∀ x, x ≤ s ∨ t ≤ x → (deriv (f a)) x ≤ 0) ∧
    t - s ≤ 1 ∧
    (∃ a₀ : ℝ, 0 < a₀ ∧ a₀ ≤ 1 ∧ 
      ∃ s₀ t₀ : ℝ, s₀ < t₀ ∧
        (∀ x ∈ Set.Ioo s₀ t₀, (deriv (f a₀)) x > 0) ∧
        (∀ x, x ≤ s₀ ∨ t₀ ≤ x → (deriv (f a₀)) x ≤ 0) ∧
        t₀ - s₀ = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l727_72737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l727_72758

noncomputable def fill_time (p q r : ℝ) : ℝ := 1 / (p + q + r)

noncomputable def valve_rate (time : ℝ) : ℝ := 1 / time

theorem pool_fill_time 
  (p q r : ℝ) 
  (h_pqr : fill_time p q r = 2)
  (h_pr : fill_time p 0 r = 3)
  (h_qr : fill_time 0 q r = 4) :
  fill_time p q 0 = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_fill_time_l727_72758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_odd_and_periodic_l727_72777

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * x)

theorem sin_3x_odd_and_periodic :
  (∀ x, f (-x) = -f x) ∧ 
  (∃ T > 0, ∀ x, f (x + T) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_3x_odd_and_periodic_l727_72777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_equality_l727_72795

/-- The sum of the geometric series with first term 1 and common ratio 1/3 -/
noncomputable def series1 : ℝ := 3/2

/-- The sum of the geometric series with first term 1 and common ratio -1/3 -/
noncomputable def series2 : ℝ := 3/4

/-- The product of series1 and series2 -/
noncomputable def product : ℝ := series1 * series2

/-- The sum of the geometric series with first term 1 and common ratio 1/y -/
noncomputable def series3 (y : ℝ) : ℝ := 1 / (1 - 1/y)

theorem geometric_series_equality :
  product = series3 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_equality_l727_72795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_real_l727_72729

/-- The function f(x) with parameter k -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (2*k*x - 8) / (k*x^2 + 2*k*x + 1)

/-- The theorem stating the condition for f to have a domain of all real numbers -/
theorem f_domain_real (k : ℝ) : 
  (∀ x, ∃ y, f k x = y) ↔ (0 ≤ k ∧ k < 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_real_l727_72729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l727_72723

theorem trigonometric_identities (α : ℝ) 
  (h : Real.sin α - 2 * Real.cos α = 0) : 
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -5 ∧ 
  2 * Real.sin α * Real.cos α = 4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l727_72723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l727_72757

theorem angle_in_fourth_quadrant (θ : ℝ) 
  (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  0 < θ ∧ θ < π / 2 ∧ Real.sin θ < 0 ∧ Real.cos θ > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_fourth_quadrant_l727_72757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l727_72705

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b / a)^2)

/-- The eccentricity of a hyperbola with real semi-axis a and imaginary semi-axis b -/
noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b / a)^2)

theorem hyperbola_equation :
  let ellipse_eq := fun (x y : ℝ) => x^2 + y^2 / 2 = 1
  let ellipse_semi_major := Real.sqrt 2
  let ellipse_semi_minor := 1
  let hyperbola_vertices := fun (x y : ℝ) => x = 0 ∧ y = ellipse_semi_major
  let eccentricity_product := ellipse_eccentricity ellipse_semi_major ellipse_semi_minor *
                              hyperbola_eccentricity (Real.sqrt 2) (Real.sqrt 2) = 1
  ellipse_eq (Real.sqrt 2) 1 ∧ 
  hyperbola_vertices 0 (Real.sqrt 2) ∧ 
  eccentricity_product →
  ∀ x y : ℝ, y^2 - x^2 = 2 := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l727_72705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_S_formula_l727_72798

/-- An arithmetic sequence with special properties -/
structure SpecialArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  S : ℕ → ℝ
  S_def : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)
  S_5_condition : S 5 = 4 * a 3 + 6
  geometric_subsequence : (a 3)^2 = a 2 * a 9
  a_1_ne_a_5 : a 1 ≠ a 5

/-- The sum of the first n terms of the sequence {1/S_n} -/
noncomputable def sum_inverse_S (seq : SpecialArithmeticSequence) (n : ℕ) : ℝ :=
  n / (n + 1)

theorem sum_inverse_S_formula (seq : SpecialArithmeticSequence) (n : ℕ) :
  sum_inverse_S seq n = n / (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_S_formula_l727_72798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos6_l727_72752

theorem min_sin6_plus_2cos6 :
  ∀ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ 1/3 ∧
  ∃ y : ℝ, Real.sin y ^ 6 + 2 * Real.cos y ^ 6 = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_plus_2cos6_l727_72752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cement_mixture_water_fraction_l727_72763

-- Define the total weight of the mixture
noncomputable def total_weight : ℝ := 23.999999999999996

-- Define the weight of gravel
noncomputable def gravel_weight : ℝ := 10

-- Define the fraction of sand in the mixture
noncomputable def sand_fraction : ℝ := 1/3

-- Define the fraction of water in the mixture
noncomputable def water_fraction : ℝ := 1/4

-- Theorem statement
theorem cement_mixture_water_fraction :
  let sand_weight := sand_fraction * total_weight
  let water_weight := total_weight - sand_weight - gravel_weight
  water_weight / total_weight = water_fraction := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cement_mixture_water_fraction_l727_72763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l727_72781

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 5 = 0

-- Define the midpoint of chord AB
def chord_midpoint : ℝ × ℝ := (-2, 3)

-- Theorem statement
theorem line_equation_proof :
  ∃ (A B : ℝ × ℝ),
    circle_eq A.1 A.2 ∧
    circle_eq B.1 B.2 ∧
    A ≠ B ∧
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = chord_midpoint →
    ∀ (x y : ℝ), line_l x y ↔ (∃ t : ℝ, x = A.1 + t * (B.1 - A.1) ∧ y = A.2 + t * (B.2 - A.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l727_72781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golf_ball_max_height_l727_72717

/-- The flight height function of a golf ball -/
noncomputable def flight_height (x : ℝ) : ℝ := -1/50 * (x - 25)^2 + 12

/-- The maximum height of the golf ball's flight -/
def max_height : ℝ := 12

/-- Theorem: The maximum height of the golf ball's flight is 12 meters -/
theorem golf_ball_max_height :
  ∀ x : ℝ, flight_height x ≤ max_height := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golf_ball_max_height_l727_72717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l727_72702

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 5^x

-- Define the transformed function
noncomputable def g (x : ℝ) : ℝ := 5^(x - 3) - 2

-- Theorem statement
theorem function_transformation (x : ℝ) : 
  g x = f (x - 3) - 2 := by
  -- Unfold the definitions of g and f
  unfold g f
  -- The proof is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_transformation_l727_72702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_count_l727_72740

theorem polynomial_roots_count (n : ℕ) (P : Polynomial ℝ) :
  P.degree = n → (P.roots.toFinset.card : ℕ) ≤ n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_count_l727_72740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l727_72715

/-- The number of integer solutions to x + y + z = 2016 with x > 1000, y > 600, and z > 400 -/
def num_solutions : ℕ := 105

/-- The set of all integer solutions to x + y + z = 2016 with x > 1000, y > 600, and z > 400 -/
def solution_set : Set (ℤ × ℤ × ℤ) :=
  {(x, y, z) | x + y + z = 2016 ∧ x > 1000 ∧ y > 600 ∧ z > 400}

/-- The count of solutions is equal to num_solutions -/
theorem count_solutions :
  Finset.card (Finset.filter (λ (x, y, z) => x + y + z = 2016 ∧ x > 1000 ∧ y > 600 ∧ z > 400)
    (Finset.range 2017 ×ˢ Finset.range 2017 ×ˢ Finset.range 2017)) = num_solutions := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_solutions_l727_72715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l727_72707

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem distance_between_specific_points :
  distance (1, -1) (7, 7) = 10 := by
  -- Unfold the definition of distance
  unfold distance
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l727_72707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l727_72742

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 1 / (Real.exp x + 1) - 1/2

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l727_72742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_distance_l727_72754

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 9 = 1

/-- The line equation -/
def line_equation (x y : ℝ) : ℝ := 3 * x - 4 * y - 24

/-- Distance from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := 
  |line_equation x y| / Real.sqrt (3^2 + (-4)^2)

/-- Maximum distance from any point on the ellipse to the line -/
noncomputable def max_distance : ℝ := 12 * (2 + Real.sqrt 2) / 5

/-- Minimum distance from any point on the ellipse to the line -/
noncomputable def min_distance : ℝ := 12 * (2 - Real.sqrt 2) / 5

/-- Theorem stating the maximum and minimum distances -/
theorem ellipse_line_distance : 
  ∀ x y : ℝ, is_on_ellipse x y → 
    (∀ x' y' : ℝ, is_on_ellipse x' y' → 
      distance_to_line x y ≤ max_distance ∧ 
      min_distance ≤ distance_to_line x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_distance_l727_72754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_coeff_values_l727_72761

/-- Monic quadratic polynomial -/
def MonicQuadratic (α β : ℝ) : ℝ → ℝ := fun x ↦ x^2 + α*x + β

theorem monic_quadratic_coeff_values 
  (P Q : ℝ → ℝ) 
  (B C E F : ℝ)
  (hP : P = MonicQuadratic B C)
  (hQ : Q = MonicQuadratic E F)
  (hPQ : ∀ x, x ∈ ({-19, -13, -7, -1} : Set ℝ) → P (Q x) = 0)
  (hQP : ∀ x, x ∈ ({-53, -47, -41, -35} : Set ℝ) → Q (P x) = 0) :
  E = 20 ∧ B = 88 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_coeff_values_l727_72761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_square_roots_of_m_l727_72716

theorem square_root_problem (a : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ Real.sqrt x = 2*a - 1 ∧ Real.sqrt x = a - 5) →
  (∃ x : ℝ, x = 9 ∧ x ≥ 0 ∧ Real.sqrt x = 2*a - 1 ∧ Real.sqrt x = a - 5) :=
by sorry

theorem square_roots_of_m (a m : ℝ) :
  ((a - 1)^2 = m ∧ (5 - 2*a)^2 = m) →
  ((a = 2 ∧ m = 1) ∨ (a = 4 ∧ m = 9)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_square_roots_of_m_l727_72716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_millet_in_brand_b_l727_72780

/-- Represents the percentage of a component in a mixture -/
def Percentage := ℝ

/-- Brand A birdseed -/
structure BrandA where
  millet : Percentage
  sunflower : Percentage

/-- Brand B birdseed -/
structure BrandB where
  millet : Percentage
  safflower : Percentage

/-- Mixed birdseed -/
structure MixedBirdseed where
  brandA_proportion : Percentage
  brandB_proportion : Percentage
  millet : Percentage

/-- The theorem stating the percentage of millet in Brand B -/
theorem millet_in_brand_b 
  (a : BrandA)
  (mix : MixedBirdseed)
  (h1 : a.millet = (4 : ℝ) / 10)
  (h2 : mix.brandA_proportion = (6 : ℝ) / 10)
  (h3 : mix.millet = (1 : ℝ) / 2) :
  ∃ (b : BrandB), b.millet = (13 : ℝ) / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_millet_in_brand_b_l727_72780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_divides_circles_equally_l727_72700

-- Define the circles
def circles : List (ℝ × ℝ) :=
  [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5), (3.5, 0.5), (4.5, 0.5),
   (0.5, 1.5), (1.5, 1.5), (2.5, 1.5), (3.5, 1.5), (4.5, 1.5)]

-- Define the region S (union of circular regions)
def S : Set (ℝ × ℝ) :=
  {point | ∃ (center : ℝ × ℝ), center ∈ circles ∧ 
           (point.1 - center.1)^2 + (point.2 - center.2)^2 ≤ 0.25}

-- Define the line m
def m (x y : ℝ) : Prop := y = -2 * x + 5

-- Define the property of m dividing S into equal areas
noncomputable def divides_equally (line : ℝ → ℝ → Prop) (region : Set (ℝ × ℝ)) : Prop :=
  ∃ (A₁ A₂ : Set (ℝ × ℝ)), 
    A₁ ∪ A₂ = region ∧ 
    A₁ ∩ A₂ = ∅ ∧
    (∀ (x y : ℝ), (x, y) ∈ A₁ → ¬line x y) ∧
    (∀ (x y : ℝ), (x, y) ∈ A₂ → line x y) ∧
    MeasureTheory.volume A₁ = MeasureTheory.volume A₂

-- The theorem to prove
theorem line_divides_circles_equally :
  divides_equally m S →
  ∃ (p q r : ℕ), 
    p > 0 ∧ q > 0 ∧ r > 0 ∧
    Nat.gcd p (Nat.gcd q r) = 1 ∧
    (∀ (x y : ℝ), m x y ↔ p * x = q * y + r) ∧
    p^2 + q^2 + r^2 = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_divides_circles_equally_l727_72700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_pricing_strategy_l727_72701

theorem store_pricing_strategy (list_price : ℝ) (h : list_price > 0) :
  let cost_price := 0.7 * list_price
  let selling_price := (4/3) * cost_price
  let marked_price := (4/3) * selling_price
  ∃ (ε : ℝ), abs (marked_price / list_price - 1.24) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_pricing_strategy_l727_72701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_ratio_bound_max_perimeter_ratio_achievable_l727_72718

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the excircle points
structure ExcirclePoints where
  A1 : ℝ × ℝ
  B1 : ℝ × ℝ
  C1 : ℝ × ℝ
  A2 : ℝ × ℝ
  B2 : ℝ × ℝ
  C2 : ℝ × ℝ
  A3 : ℝ × ℝ
  B3 : ℝ × ℝ
  C3 : ℝ × ℝ

-- Function to calculate the perimeter of a triangle given its vertices
noncomputable def perimeter (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Function to calculate the circumradius of a triangle
noncomputable def circumradius (t : Triangle) : ℝ := sorry

-- Function to calculate the ratio of sum of perimeters to circumradius
noncomputable def perimeterRatio (t : Triangle) (e : ExcirclePoints) : ℝ :=
  (perimeter e.A1 e.B1 e.C1 + perimeter e.A2 e.B2 e.C2 + perimeter e.A3 e.B3 e.C3) / circumradius t

-- Theorem statement
theorem max_perimeter_ratio_bound {t : Triangle} {e : ExcirclePoints} :
  perimeterRatio t e ≤ 9 + 9 * Real.sqrt 3 / 2 := by
  sorry

-- Theorem stating that the bound is achievable
theorem max_perimeter_ratio_achievable :
  ∃ (t : Triangle) (e : ExcirclePoints), perimeterRatio t e = 9 + 9 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_ratio_bound_max_perimeter_ratio_achievable_l727_72718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_donovan_mitchell_average_l727_72747

/-- Calculates the required average points per game for remaining games to achieve a target season average -/
noncomputable def required_average (current_average : ℝ) (games_played : ℕ) (total_games : ℕ) (target_average : ℝ) : ℝ :=
  let remaining_games := total_games - games_played
  let total_points_needed := target_average * (total_games : ℝ)
  let points_scored := current_average * (games_played : ℝ)
  let points_needed := total_points_needed - points_scored
  points_needed / (remaining_games : ℝ)

/-- Theorem stating the required average for the given scenario -/
theorem donovan_mitchell_average : 
  required_average 26 15 20 30 = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_donovan_mitchell_average_l727_72747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_abc_l727_72764

theorem min_value_abc (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 10 * a ^ 2 - 3 * a * b + 7 * c ^ 2 = 0) :
  (Nat.gcd a b) * (Nat.gcd b c) * (Nat.gcd c a) ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_abc_l727_72764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_two_thirds_l727_72726

/-- The rectangle in which points are chosen -/
def Rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

/-- The region where x + 2y < 6 -/
def Region : Set (ℝ × ℝ) :=
  {p | p.1 + 2 * p.2 < 6}

/-- The measure of the rectangle -/
noncomputable def rectangleArea : ℝ := 12

/-- The measure of the intersection of the rectangle and the region -/
noncomputable def intersectionArea : ℝ := 8

/-- The probability of a point in the rectangle satisfying x + 2y < 6 -/
noncomputable def probability : ℝ := intersectionArea / rectangleArea

theorem probability_is_two_thirds :
  probability = 2/3 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_two_thirds_l727_72726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_8_l727_72744

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  a * c = b^2

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) :=
  (n : ℝ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem arithmetic_sequence_sum_8 (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  a 1 = 1 →
  d ≠ 0 →
  geometric_sequence (a 1) (a 2) (a 5) →
  sum_of_arithmetic_sequence a 8 = 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_8_l727_72744
