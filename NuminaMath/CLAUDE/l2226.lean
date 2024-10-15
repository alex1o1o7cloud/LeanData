import Mathlib

namespace NUMINAMATH_CALUDE_students_per_bus_l2226_222636

/-- Given a field trip scenario with buses and students, calculate the number of students per bus. -/
theorem students_per_bus (total_seats : ℕ) (num_buses : ℚ) : 
  total_seats = 28 → num_buses = 2 → (total_seats : ℚ) / num_buses = 14 := by
  sorry

end NUMINAMATH_CALUDE_students_per_bus_l2226_222636


namespace NUMINAMATH_CALUDE_first_interest_rate_is_eight_percent_l2226_222603

/-- Proves that the first interest rate is 8% given the problem conditions -/
theorem first_interest_rate_is_eight_percent 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (second_rate : ℝ) 
  (h1 : total_investment = 5400)
  (h2 : first_investment = 3000)
  (h3 : second_investment = total_investment - first_investment)
  (h4 : second_rate = 0.10)
  (h5 : first_investment * (first_rate : ℝ) = second_investment * second_rate) :
  first_rate = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_first_interest_rate_is_eight_percent_l2226_222603


namespace NUMINAMATH_CALUDE_simplify_expression_l2226_222612

theorem simplify_expression (x : ℝ) : x + 3 - 4*x - 5 + 6*x + 7 - 8*x - 9 = -5*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2226_222612


namespace NUMINAMATH_CALUDE_box_volume_l2226_222680

theorem box_volume (l w h : ℝ) (shortest_path : ℝ) : 
  l = 6 → w = 6 → shortest_path = 20 → 
  shortest_path^2 = (l + w + h)^2 + w^2 →
  l * w * h = 576 := by
sorry

end NUMINAMATH_CALUDE_box_volume_l2226_222680


namespace NUMINAMATH_CALUDE_fundamental_inequality_variant_l2226_222639

theorem fundamental_inequality_variant (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  1 / (2 * a) + 1 / b ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_fundamental_inequality_variant_l2226_222639


namespace NUMINAMATH_CALUDE_javiers_cats_l2226_222668

/-- Calculates the number of cats in Javier's household -/
def number_of_cats (adults children dogs total_legs : ℕ) : ℕ :=
  let human_legs := 2 * (adults + children)
  let dog_legs := 4 * dogs
  let remaining_legs := total_legs - human_legs - dog_legs
  remaining_legs / 4

/-- Theorem stating that the number of cats in Javier's household is 1 -/
theorem javiers_cats :
  number_of_cats 2 3 2 22 = 1 :=
by sorry

end NUMINAMATH_CALUDE_javiers_cats_l2226_222668


namespace NUMINAMATH_CALUDE_largest_difference_is_209_l2226_222660

/-- A type representing a 20 × 20 square table filled with distinct natural numbers from 1 to 400. -/
def Table := Fin 20 → Fin 20 → Fin 400

/-- The property that all numbers in the table are distinct. -/
def all_distinct (t : Table) : Prop :=
  ∀ i j k l, t i j = t k l → (i = k ∧ j = l)

/-- The property that there exist two numbers in the same row or column with a difference of at least N. -/
def has_difference_at_least (t : Table) (N : ℕ) : Prop :=
  ∃ i j k, (j = k ∧ |t i j - t i k| ≥ N) ∨ (i = k ∧ |t i j - t k j| ≥ N)

/-- The main theorem stating that 209 is the largest value satisfying the condition. -/
theorem largest_difference_is_209 :
  (∀ t : Table, all_distinct t → has_difference_at_least t 209) ∧
  ¬(∀ t : Table, all_distinct t → has_difference_at_least t 210) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_is_209_l2226_222660


namespace NUMINAMATH_CALUDE_beth_crayons_count_l2226_222683

/-- The number of crayon packs Beth has -/
def num_packs : ℕ := 4

/-- The number of crayons in each pack -/
def crayons_per_pack : ℕ := 10

/-- The number of extra crayons Beth has -/
def extra_crayons : ℕ := 6

/-- The total number of crayons Beth has -/
def total_crayons : ℕ := num_packs * crayons_per_pack + extra_crayons

theorem beth_crayons_count : total_crayons = 46 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_count_l2226_222683


namespace NUMINAMATH_CALUDE_f_properties_l2226_222629

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a * b

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a + b

-- Define the function f
def f (x : ℝ) : ℝ := otimes x 2 - oplus 2 x

-- Theorem statement
theorem f_properties :
  (¬ ∀ x, f (-x) = f x) ∧  -- not even
  (¬ ∀ x, f (-x) = -f x) ∧ -- not odd
  (∀ x y, x < y → f x > f y) -- decreasing
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l2226_222629


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2226_222652

theorem trigonometric_equation_solution (x : ℝ) :
  2 * Real.cos (13 * x) + 3 * Real.cos (3 * x) + 3 * Real.cos (5 * x) - 8 * Real.cos x * (Real.cos (4 * x))^3 = 0 →
  ∃ k : ℤ, x = π * k / 12 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2226_222652


namespace NUMINAMATH_CALUDE_product_equality_l2226_222640

theorem product_equality (p r j : ℝ) : 
  (6 * p^2 - 4 * p + r) * (2 * p^2 + j * p - 7) = 12 * p^4 - 34 * p^3 - 19 * p^2 + 28 * p - 21 →
  r + j = 3 := by
sorry

end NUMINAMATH_CALUDE_product_equality_l2226_222640


namespace NUMINAMATH_CALUDE_motorcyclists_travel_time_l2226_222650

/-- 
Two motorcyclists start simultaneously from opposite points A and B.
They meet at some point between A and B.
The first motorcyclist (from A to B) arrives at B 2.5 hours after meeting.
The second motorcyclist (from B to A) arrives at A 1.6 hours after meeting.
This theorem proves that their total travel times are 4.5 hours and 3.6 hours respectively.
-/
theorem motorcyclists_travel_time (s : ℝ) (h : s > 0) : 
  ∃ (t : ℝ), t > 0 ∧ 
    (s / (t + 2.5) * 2.5 = s / (t + 1.6) * t) ∧ 
    (t + 2.5 = 4.5) ∧ 
    (t + 1.6 = 3.6) := by
  sorry

#check motorcyclists_travel_time

end NUMINAMATH_CALUDE_motorcyclists_travel_time_l2226_222650


namespace NUMINAMATH_CALUDE_max_achievable_grade_l2226_222662

theorem max_achievable_grade (test1_score test2_score test3_score : ℝ)
  (test1_weight test2_weight test3_weight test4_weight : ℝ)
  (max_extra_credit : ℝ) (target_grade : ℝ) :
  test1_score = 95 ∧ test2_score = 80 ∧ test3_score = 90 ∧
  test1_weight = 0.25 ∧ test2_weight = 0.3 ∧ test3_weight = 0.25 ∧ test4_weight = 0.2 ∧
  max_extra_credit = 5 ∧ target_grade = 93 →
  let current_weighted_grade := test1_score * test1_weight + test2_score * test2_weight + test3_score * test3_weight
  let max_fourth_test_score := 100 + max_extra_credit
  let max_achievable_grade := current_weighted_grade + max_fourth_test_score * test4_weight
  max_achievable_grade < target_grade ∧ max_achievable_grade = 91.25 :=
by sorry

end NUMINAMATH_CALUDE_max_achievable_grade_l2226_222662


namespace NUMINAMATH_CALUDE_expand_expression_l2226_222688

theorem expand_expression (x : ℝ) : 4 * (5 * x^3 - 3 * x^2 + 7 * x - 2) = 20 * x^3 - 12 * x^2 + 28 * x - 8 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2226_222688


namespace NUMINAMATH_CALUDE_quotient_rational_l2226_222646

-- Define the set A as a subset of positive reals
def A : Set ℝ := {x : ℝ | x > 0}

-- Define the property that A is non-empty
axiom A_nonempty : Set.Nonempty A

-- Define the condition that for all a, b, c in A, ab + bc + ca is rational
axiom sum_rational (a b c : ℝ) (ha : a ∈ A) (hb : b ∈ A) (hc : c ∈ A) :
  ∃ (q : ℚ), (a * b + b * c + c * a : ℝ) = q

-- State the theorem to be proved
theorem quotient_rational (a b : ℝ) (ha : a ∈ A) (hb : b ∈ A) :
  ∃ (q : ℚ), (a / b : ℝ) = q :=
sorry

end NUMINAMATH_CALUDE_quotient_rational_l2226_222646


namespace NUMINAMATH_CALUDE_max_y_coordinate_polar_graph_l2226_222617

theorem max_y_coordinate_polar_graph :
  let r : ℝ → ℝ := λ θ ↦ 2 * Real.sin (2 * θ)
  let y : ℝ → ℝ := λ θ ↦ r θ * Real.sin θ
  (∀ θ, y θ ≤ (8 * Real.sqrt 3) / 9) ∧ 
  (∃ θ, y θ = (8 * Real.sqrt 3) / 9) := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_polar_graph_l2226_222617


namespace NUMINAMATH_CALUDE_tan_x_equals_zero_l2226_222638

theorem tan_x_equals_zero (x : Real) 
  (h1 : 0 ≤ x ∧ x ≤ π) 
  (h2 : 3 * Real.sin (x/2) = Real.sqrt (1 + Real.sin x) - Real.sqrt (1 - Real.sin x)) : 
  Real.tan x = 0 := by
sorry

end NUMINAMATH_CALUDE_tan_x_equals_zero_l2226_222638


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l2226_222689

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- Calculates the number of ways to arrange n crates with given counts of each orientation -/
def arrangementCount (n a b c : ℕ) : ℕ := sorry

/-- The probability of stacking crates to achieve a specific height -/
def stackProbability (dimensions : CrateDimensions) (numCrates targetHeight : ℕ) : ℚ :=
  let totalArrangements := 3^numCrates
  let validArrangements := 
    arrangementCount numCrates 8 0 4 + 
    arrangementCount numCrates 6 3 3 + 
    arrangementCount numCrates 4 6 2 + 
    arrangementCount numCrates 2 9 1 + 
    arrangementCount numCrates 0 12 0
  validArrangements / totalArrangements

theorem crate_stacking_probability : 
  let dimensions := CrateDimensions.mk 3 4 6
  stackProbability dimensions 12 48 = 37522 / 531441 := by sorry

end NUMINAMATH_CALUDE_crate_stacking_probability_l2226_222689


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2226_222644

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_condition : a 3 + a 9 = a 10 - a 8) :
  a 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2226_222644


namespace NUMINAMATH_CALUDE_non_integer_mean_arrangement_l2226_222670

theorem non_integer_mean_arrangement (N : ℕ) (h : Even N) :
  ∃ (arr : List ℕ),
    (arr.length = N) ∧
    (∀ x, x ∈ arr ↔ 1 ≤ x ∧ x ≤ N) ∧
    (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ N →
      ¬(∃ (k : ℕ), (arr.take j).sum - (arr.take (i-1)).sum = k * (j - i + 1))) :=
by sorry

end NUMINAMATH_CALUDE_non_integer_mean_arrangement_l2226_222670


namespace NUMINAMATH_CALUDE_essay_time_calculation_l2226_222610

/-- The time Rachel spent on her essay -/
def essay_time (
  page_writing_time : ℕ)  -- Time to write one page in seconds
  (research_time : ℕ)     -- Time spent researching in seconds
  (outline_time : ℕ)      -- Time spent on outline in minutes
  (brainstorm_time : ℕ)   -- Time spent brainstorming in seconds
  (total_pages : ℕ)       -- Total number of pages written
  (break_time : ℕ)        -- Break time after each page in seconds
  (editing_time : ℕ)      -- Time spent editing in seconds
  (proofreading_time : ℕ) -- Time spent proofreading in seconds
  : ℚ :=
  let total_seconds : ℕ := 
    research_time + 
    (outline_time * 60) + 
    brainstorm_time + 
    (total_pages * page_writing_time) + 
    (total_pages * break_time) + 
    editing_time + 
    proofreading_time
  (total_seconds : ℚ) / 3600

theorem essay_time_calculation : 
  essay_time 1800 2700 15 1200 6 600 4500 1800 = 25500 / 3600 := by
  sorry

#eval essay_time 1800 2700 15 1200 6 600 4500 1800

end NUMINAMATH_CALUDE_essay_time_calculation_l2226_222610


namespace NUMINAMATH_CALUDE_line_contains_point_l2226_222623

theorem line_contains_point (k : ℝ) : 
  (3 / 4 - 3 * k * (1 / 3) = 7 * (-4)) ↔ k = 28.75 := by sorry

end NUMINAMATH_CALUDE_line_contains_point_l2226_222623


namespace NUMINAMATH_CALUDE_three_line_hexagon_angle_sum_l2226_222635

/-- A hexagon formed by the intersection of three lines -/
structure ThreeLineHexagon where
  -- Define the six angles of the hexagon
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ
  angle6 : ℝ

/-- The sum of angles in a hexagon formed by three intersecting lines is 360° -/
theorem three_line_hexagon_angle_sum (h : ThreeLineHexagon) : 
  h.angle1 + h.angle2 + h.angle3 + h.angle4 + h.angle5 + h.angle6 = 360 := by
  sorry

end NUMINAMATH_CALUDE_three_line_hexagon_angle_sum_l2226_222635


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2226_222621

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (((x > 1 ∧ y > 2) → x + y > 3) ∧
   ∃ x y, x + y > 3 ∧ ¬(x > 1 ∧ y > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2226_222621


namespace NUMINAMATH_CALUDE_joan_football_games_l2226_222600

theorem joan_football_games (games_this_year games_total : ℕ) 
  (h1 : games_this_year = 4)
  (h2 : games_total = 13) :
  games_total - games_this_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_football_games_l2226_222600


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2226_222684

theorem interest_rate_calculation (total_sum second_part : ℝ)
  (h1 : total_sum = 2691)
  (h2 : second_part = 1656)
  (h3 : total_sum > second_part) :
  let first_part := total_sum - second_part
  let interest_rate_first := 0.03
  let time_first := 8
  let time_second := 3
  let interest_rate_second := (first_part * interest_rate_first * time_first) / (second_part * time_second)
  ∃ ε > 0, |interest_rate_second - 0.05| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2226_222684


namespace NUMINAMATH_CALUDE_product_is_zero_matrix_l2226_222643

def skew_symmetric_matrix (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, d, -e;
     -d, 0, f;
     e, -f, 0]

def symmetric_matrix (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![d^2, d*e, d*f;
     d*e, e^2, e*f;
     d*f, e*f, f^2]

theorem product_is_zero_matrix (d e f : ℝ) : 
  skew_symmetric_matrix d e f * symmetric_matrix d e f = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_is_zero_matrix_l2226_222643


namespace NUMINAMATH_CALUDE_prop_2_prop_4_l2226_222664

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Theorem for proposition ②
theorem prop_2 (l : Line) (α β : Plane) :
  perpendicular l α → parallel l β → plane_perpendicular α β := by sorry

-- Theorem for proposition ④
theorem prop_4 (α β γ : Plane) :
  plane_perpendicular α β → plane_parallel α γ → plane_perpendicular γ β := by sorry

end NUMINAMATH_CALUDE_prop_2_prop_4_l2226_222664


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2226_222620

/-- A geometric sequence with common ratio q > 1 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (4 * a 2005 ^ 2 - 8 * a 2005 + 3 = 0) →
  (4 * a 2006 ^ 2 - 8 * a 2006 + 3 = 0) →
  a 2007 + a 2008 = 18 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2226_222620


namespace NUMINAMATH_CALUDE_original_height_is_100_l2226_222699

/-- The rebound factor of the ball -/
def rebound_factor : ℝ := 0.5

/-- The total travel distance when the ball touches the floor for the third time -/
def total_distance : ℝ := 250

/-- Calculates the total travel distance for a ball dropped from height h -/
def calculate_total_distance (h : ℝ) : ℝ :=
  h + 2 * h * rebound_factor + 2 * h * rebound_factor^2

/-- Theorem stating that the original height is 100 cm -/
theorem original_height_is_100 :
  ∃ h : ℝ, h > 0 ∧ calculate_total_distance h = total_distance ∧ h = 100 :=
sorry

end NUMINAMATH_CALUDE_original_height_is_100_l2226_222699


namespace NUMINAMATH_CALUDE_sequence_sum_1993_l2226_222658

def sequence_sum (n : ℕ) : ℤ :=
  let group_sum := 5
  let num_groups := n / 5
  ↑num_groups * group_sum

theorem sequence_sum_1993 :
  sequence_sum 1993 = 1990 :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_1993_l2226_222658


namespace NUMINAMATH_CALUDE_determinant_calculation_l2226_222615

variable (a₁ b₁ b₂ c₁ c₂ c₃ d₁ d₂ d₃ d₄ : ℝ)

def matrix : Matrix (Fin 4) (Fin 4) ℝ := λ i j =>
  match i, j with
  | 0, 0 => a₁
  | 0, 1 => b₁
  | 0, 2 => c₁
  | 0, 3 => d₁
  | 1, 0 => a₁
  | 1, 1 => b₂
  | 1, 2 => c₂
  | 1, 3 => d₂
  | 2, 0 => a₁
  | 2, 1 => b₂
  | 2, 2 => c₃
  | 2, 3 => d₃
  | 3, 0 => a₁
  | 3, 1 => b₂
  | 3, 2 => c₃
  | 3, 3 => d₄
  | _, _ => 0

theorem determinant_calculation :
  Matrix.det (matrix a₁ b₁ b₂ c₁ c₂ c₃ d₁ d₂ d₃ d₄) = a₁ * (b₂ - b₁) * (c₃ - c₂) * (d₄ - d₃) := by
  sorry

end NUMINAMATH_CALUDE_determinant_calculation_l2226_222615


namespace NUMINAMATH_CALUDE_balloon_theorem_l2226_222607

/-- Represents a person's balloon collection -/
structure BalloonCollection where
  count : ℕ
  cost : ℕ

/-- Calculates the total number of balloons from a list of balloon collections -/
def totalBalloons (collections : List BalloonCollection) : ℕ :=
  collections.map (·.count) |>.sum

/-- Calculates the total cost of balloons from a list of balloon collections -/
def totalCost (collections : List BalloonCollection) : ℕ :=
  collections.map (fun c => c.count * c.cost) |>.sum

theorem balloon_theorem (fred sam mary susan tom : BalloonCollection)
    (h1 : fred = ⟨5, 3⟩)
    (h2 : sam = ⟨6, 4⟩)
    (h3 : mary = ⟨7, 5⟩)
    (h4 : susan = ⟨4, 6⟩)
    (h5 : tom = ⟨10, 2⟩) :
    let collections := [fred, sam, mary, susan, tom]
    totalBalloons collections = 32 ∧ totalCost collections = 118 := by
  sorry

end NUMINAMATH_CALUDE_balloon_theorem_l2226_222607


namespace NUMINAMATH_CALUDE_inequalities_not_necessarily_true_l2226_222604

theorem inequalities_not_necessarily_true
  (x y z a b c : ℝ)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hxa : x < a) (hyb : y > b) (hzc : z < c) :
  ∃ (x' y' z' a' b' c' : ℝ),
    x' < a' ∧ y' > b' ∧ z' < c' ∧
    ¬(x'*z' + y' < a'*z' + b') ∧
    ¬(x'*y' < a'*b') ∧
    ¬((x' + y') / z' < (a' + b') / c') ∧
    ¬(x'*y'*z' < a'*b'*c') :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_necessarily_true_l2226_222604


namespace NUMINAMATH_CALUDE_creature_probability_l2226_222678

/-- Represents the type of creature on the island -/
inductive Creature
| Hare
| Rabbit

/-- The probability of a creature being mistaken -/
def mistakeProbability (c : Creature) : ℚ :=
  match c with
  | Creature.Hare => 1/4
  | Creature.Rabbit => 1/3

/-- The probability of a creature being correct -/
def correctProbability (c : Creature) : ℚ :=
  1 - mistakeProbability c

/-- The probability of a creature being of a certain type -/
def populationProbability (c : Creature) : ℚ := 1/2

theorem creature_probability (A B C : Prop) :
  let pA := populationProbability Creature.Hare
  let pNotA := populationProbability Creature.Rabbit
  let pBA := mistakeProbability Creature.Hare
  let pCA := correctProbability Creature.Hare
  let pBNotA := correctProbability Creature.Rabbit
  let pCNotA := mistakeProbability Creature.Rabbit
  let pABC := pA * pBA * pCA
  let pNotABC := pNotA * pBNotA * pCNotA
  let pBC := pABC + pNotABC
  pABC / pBC = 27/59 := by sorry

end NUMINAMATH_CALUDE_creature_probability_l2226_222678


namespace NUMINAMATH_CALUDE_remainder_three_divisor_l2226_222695

theorem remainder_three_divisor (n : ℕ) (h : n = 1680) (h9 : n % 9 = 0) :
  ∃ m : ℕ, m = 1677 ∧ n % m = 3 :=
by sorry

end NUMINAMATH_CALUDE_remainder_three_divisor_l2226_222695


namespace NUMINAMATH_CALUDE_tile_ratio_l2226_222657

theorem tile_ratio (total : Nat) (yellow purple white : Nat)
  (h_total : total = 20)
  (h_yellow : yellow = 3)
  (h_purple : purple = 6)
  (h_white : white = 7) :
  (total - (yellow + purple + white)) / yellow = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tile_ratio_l2226_222657


namespace NUMINAMATH_CALUDE_probability_two_defective_tubes_l2226_222663

/-- The probability of selecting two defective tubes without replacement from a consignment of picture tubes -/
theorem probability_two_defective_tubes (total : ℕ) (defective : ℕ) 
  (h1 : total = 20) (h2 : defective = 5) (h3 : defective < total) :
  (defective : ℚ) / total * ((defective - 1) : ℚ) / (total - 1) = 1 / 19 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_defective_tubes_l2226_222663


namespace NUMINAMATH_CALUDE_oliver_money_l2226_222611

/-- 
Given that Oliver:
- Had x dollars in January
- Spent 4 dollars by March
- Received 32 dollars from his mom
- Then had 61 dollars

Prove that x must equal 33.
-/
theorem oliver_money (x : ℤ) 
  (spent : ℤ) 
  (received : ℤ) 
  (final_amount : ℤ) 
  (h1 : spent = 4)
  (h2 : received = 32)
  (h3 : final_amount = 61)
  (h4 : x - spent + received = final_amount) : 
  x = 33 := by
  sorry

end NUMINAMATH_CALUDE_oliver_money_l2226_222611


namespace NUMINAMATH_CALUDE_geometric_sum_eight_thirds_l2226_222606

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 8 terms of a geometric sequence with first term 1/3 and common ratio 1/3 -/
theorem geometric_sum_eight_thirds : geometric_sum (1/3) (1/3) 8 = 3280/6561 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_eight_thirds_l2226_222606


namespace NUMINAMATH_CALUDE_pencil_problem_l2226_222676

/-- Given the initial number of pencils, the number of containers, and the number of pencils that can be evenly distributed after receiving more, calculate the number of additional pencils received. -/
def additional_pencils (initial : ℕ) (containers : ℕ) (even_distribution : ℕ) : ℕ :=
  containers * even_distribution - initial

/-- Prove that given the specific conditions in the problem, the number of additional pencils is 30. -/
theorem pencil_problem : additional_pencils 150 5 36 = 30 := by
  sorry

end NUMINAMATH_CALUDE_pencil_problem_l2226_222676


namespace NUMINAMATH_CALUDE_new_students_average_age_l2226_222682

/-- Proves that the average age of new students is 32 years given the problem conditions. -/
theorem new_students_average_age
  (original_average : ℕ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℕ)
  (h1 : original_average = 40)
  (h2 : original_strength = 12)
  (h3 : new_students = 12)
  (h4 : average_decrease = 4) :
  (original_average * original_strength + new_students * 32) / (original_strength + new_students) =
  original_average - average_decrease :=
by sorry


end NUMINAMATH_CALUDE_new_students_average_age_l2226_222682


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2226_222642

theorem sufficient_but_not_necessary (x : ℝ) :
  ((-1 < x ∧ x < 3) → (x^2 - 5*x - 6 < 0)) ∧
  ¬((x^2 - 5*x - 6 < 0) → (-1 < x ∧ x < 3)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2226_222642


namespace NUMINAMATH_CALUDE_complex_fraction_equation_l2226_222681

theorem complex_fraction_equation (y : ℚ) : 
  3 + 1 / (1 + 1 / (3 + 3 / (4 + y))) = 169 / 53 → y = -605 / 119 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equation_l2226_222681


namespace NUMINAMATH_CALUDE_negation_of_implication_l2226_222696

theorem negation_of_implication :
  (¬(∀ x : ℝ, x > 1 → x > 0)) ↔ (∀ x : ℝ, x ≤ 1 → x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2226_222696


namespace NUMINAMATH_CALUDE_green_to_yellow_area_ratio_l2226_222634

-- Define the diameters of the circles
def small_diameter : ℝ := 2
def large_diameter : ℝ := 6

-- Define the theorem
theorem green_to_yellow_area_ratio :
  let small_radius := small_diameter / 2
  let large_radius := large_diameter / 2
  let yellow_area := π * small_radius^2
  let total_area := π * large_radius^2
  let green_area := total_area - yellow_area
  green_area / yellow_area = 8 := by
  sorry

end NUMINAMATH_CALUDE_green_to_yellow_area_ratio_l2226_222634


namespace NUMINAMATH_CALUDE_clothing_percentage_is_half_l2226_222624

/-- The percentage of total amount spent on clothing -/
def clothing_percentage : ℝ := sorry

/-- The percentage of total amount spent on food -/
def food_percentage : ℝ := 0.20

/-- The percentage of total amount spent on other items -/
def other_percentage : ℝ := 0.30

/-- The tax rate on clothing -/
def clothing_tax_rate : ℝ := 0.05

/-- The tax rate on food -/
def food_tax_rate : ℝ := 0

/-- The tax rate on other items -/
def other_tax_rate : ℝ := 0.10

/-- The total tax rate as a percentage of the total amount spent excluding taxes -/
def total_tax_rate : ℝ := 0.055

theorem clothing_percentage_is_half :
  clothing_percentage +
  food_percentage +
  other_percentage = 1 ∧
  clothing_percentage * clothing_tax_rate +
  food_percentage * food_tax_rate +
  other_percentage * other_tax_rate = total_tax_rate →
  clothing_percentage = 0.5 := by sorry

end NUMINAMATH_CALUDE_clothing_percentage_is_half_l2226_222624


namespace NUMINAMATH_CALUDE_floor_sum_equals_four_l2226_222673

theorem floor_sum_equals_four (x y : ℝ) : 
  (⌊x⌋^2 + ⌊y⌋^2 = 4) ↔ 
  ((2 ≤ x ∧ x < 3 ∧ 0 ≤ y ∧ y < 1) ∨
   (-2 ≤ x ∧ x < -1 ∧ 0 ≤ y ∧ y < 1) ∨
   (0 ≤ x ∧ x < 1 ∧ 2 ≤ y ∧ y < 3) ∨
   (0 ≤ x ∧ x < 1 ∧ -2 ≤ y ∧ y < -1)) :=
by sorry

end NUMINAMATH_CALUDE_floor_sum_equals_four_l2226_222673


namespace NUMINAMATH_CALUDE_perfect_square_property_l2226_222669

theorem perfect_square_property (x y z : ℤ) (h : x * y + y * z + z * x = 1) :
  (1 + x^2) * (1 + y^2) * (1 + z^2) = ((x + y) * (y + z) * (x + z))^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_property_l2226_222669


namespace NUMINAMATH_CALUDE_sixth_term_equals_23_l2226_222697

/-- Given a sequence with general term a(n) = 4n - 1, prove that a(6) = 23 -/
theorem sixth_term_equals_23 (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 1) : a 6 = 23 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_equals_23_l2226_222697


namespace NUMINAMATH_CALUDE_two_diamonds_balance_l2226_222605

/-- Represents the balance between shapes -/
structure Balance where
  triangle : ℝ
  diamond : ℝ
  circle : ℝ

/-- The given balance conditions -/
def balance_conditions (b : Balance) : Prop :=
  3 * b.triangle + b.diamond = 9 * b.circle ∧
  b.triangle = b.diamond + 2 * b.circle

/-- The theorem to prove -/
theorem two_diamonds_balance (b : Balance) :
  balance_conditions b → 2 * b.diamond = 1.5 * b.circle := by
  sorry

end NUMINAMATH_CALUDE_two_diamonds_balance_l2226_222605


namespace NUMINAMATH_CALUDE_no_solution_l2226_222613

/-- Sequence definition -/
def u : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 6 * u (n + 1) + 7 * u n

/-- Main theorem -/
theorem no_solution :
  ¬ ∃ (a b c n : ℕ), a * b * (a + b) * (a^2 + a*b + b^2) = c^2022 + 42 ∧ c^2022 + 42 = u n :=
by sorry

end NUMINAMATH_CALUDE_no_solution_l2226_222613


namespace NUMINAMATH_CALUDE_current_speed_l2226_222649

/-- The speed of the current given a woman's swimming times and distances -/
theorem current_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 125) (h2 : upstream_distance = 60) 
  (h3 : time = 10) : ∃ (v_w v_c : ℝ), 
  downstream_distance = (v_w + v_c) * time ∧ 
  upstream_distance = (v_w - v_c) * time ∧ 
  v_c = 3.25 :=
by sorry

end NUMINAMATH_CALUDE_current_speed_l2226_222649


namespace NUMINAMATH_CALUDE_rescue_team_distribution_l2226_222626

/-- The number of ways to distribute rescue teams to disaster sites. -/
def distribute_teams (total_teams : ℕ) (num_sites : ℕ) : ℕ :=
  sorry

/-- Constraint that each site gets at least one team -/
def at_least_one_each (distribution : List ℕ) : Prop :=
  sorry

/-- Constraint that site A gets at least two teams -/
def site_A_at_least_two (distribution : List ℕ) : Prop :=
  sorry

theorem rescue_team_distribution :
  ∃ (distributions : List (List ℕ)),
    (∀ d ∈ distributions,
      d.length = 3 ∧
      d.sum = 6 ∧
      at_least_one_each d ∧
      site_A_at_least_two d) ∧
    distributions.length = 360 :=
  sorry

end NUMINAMATH_CALUDE_rescue_team_distribution_l2226_222626


namespace NUMINAMATH_CALUDE_system_solution_is_one_two_l2226_222618

theorem system_solution_is_one_two :
  ∃! (s : Set ℝ), s = {1, 2} ∧
  (∀ x y : ℝ, (x^4 + y^4 = 17 ∧ x + y = 3) ↔ (x ∈ s ∧ y ∈ s ∧ x ≠ y)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_is_one_two_l2226_222618


namespace NUMINAMATH_CALUDE_det_A_squared_minus_2A_l2226_222698

/-- Given a 2x2 matrix A, prove that det(A^2 - 2A) = 25 -/
theorem det_A_squared_minus_2A (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A = ![![1, 3], ![2, 1]]) : 
  Matrix.det (A ^ 2 - 2 • A) = 25 := by
sorry

end NUMINAMATH_CALUDE_det_A_squared_minus_2A_l2226_222698


namespace NUMINAMATH_CALUDE_coefficient_of_quadratic_term_l2226_222690

/-- The coefficient of the quadratic term in a quadratic equation ax^2 + bx + c = 0 -/
def quadratic_coefficient (a b c : ℝ) : ℝ := a

theorem coefficient_of_quadratic_term :
  quadratic_coefficient (-5) 5 6 = -5 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_quadratic_term_l2226_222690


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l2226_222654

theorem base_10_to_base_7 (n : ℕ) (h : n = 947) :
  ∃ (a b c d : ℕ),
    n = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧
    a = 2 ∧ b = 5 ∧ c = 2 ∧ d = 2 :=
by sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l2226_222654


namespace NUMINAMATH_CALUDE_three_number_sum_l2226_222645

theorem three_number_sum (a b c : ℝ) : 
  a + b = 35 ∧ b + c = 54 ∧ c + a = 58 → a + b + c = 73.5 := by
  sorry

end NUMINAMATH_CALUDE_three_number_sum_l2226_222645


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2226_222633

theorem largest_prime_factor_of_expression : 
  ∃ p : ℕ, Nat.Prime p ∧ 
    p ∣ (20^3 + 15^4 - 10^5) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (20^3 + 15^4 - 10^5) → q ≤ p ∧ 
    p = 113 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2226_222633


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2226_222631

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 6) 
  (h_sum : a 3 + a 5 = 0) :
  ∀ n : ℕ, a n = 8 - 2 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2226_222631


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2226_222608

theorem average_marks_chemistry_mathematics 
  (P C M : ℕ) 
  (h : P + C + M = P + 110) : 
  (C + M) / 2 = 55 := by
sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2226_222608


namespace NUMINAMATH_CALUDE_cricket_average_increase_l2226_222666

/-- Represents the problem of calculating the increase in average runs -/
def calculateAverageIncrease (initialMatches : ℕ) (initialAverage : ℚ) (nextMatchRuns : ℕ) : ℚ :=
  let totalInitialRuns := initialMatches * initialAverage
  let totalMatches := initialMatches + 1
  let totalRuns := totalInitialRuns + nextMatchRuns
  (totalRuns / totalMatches) - initialAverage

/-- The theorem stating the solution to the cricket player's average problem -/
theorem cricket_average_increase :
  calculateAverageIncrease 10 32 76 = 4 := by
  sorry


end NUMINAMATH_CALUDE_cricket_average_increase_l2226_222666


namespace NUMINAMATH_CALUDE_min_value_of_function_l2226_222677

theorem min_value_of_function (x : ℝ) (h : x > 1) : x + 2 / (x - 1) ≥ 2 * Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2226_222677


namespace NUMINAMATH_CALUDE_mean_of_smallest_elements_l2226_222653

/-- The arithmetic mean of the smallest elements of all r-element subsets of {1, 2, ..., n} -/
def F (n r : ℕ) : ℚ :=
  sorry

/-- Theorem stating that F(n,r) = (n+1)/(r+1) for 1 ≤ r ≤ n -/
theorem mean_of_smallest_elements (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) :
  F n r = (n + 1 : ℚ) / (r + 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_mean_of_smallest_elements_l2226_222653


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l2226_222616

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1)
  (h2 : Complex.abs b = 1)
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b^2 * c) + b^3 / (a^2 * c) + c^3 / (a^2 * b) = 1) :
  Complex.abs (a + b + c) = Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l2226_222616


namespace NUMINAMATH_CALUDE_jake_fewer_than_steven_peach_difference_is_twelve_l2226_222675

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 19

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := 7

/-- Jake has fewer peaches than Steven -/
theorem jake_fewer_than_steven : jake_peaches < steven_peaches := by sorry

/-- The difference between Steven's and Jake's peaches -/
def peach_difference : ℕ := steven_peaches - jake_peaches

/-- Prove that the difference between Steven's and Jake's peaches is 12 -/
theorem peach_difference_is_twelve : peach_difference = 12 := by sorry

end NUMINAMATH_CALUDE_jake_fewer_than_steven_peach_difference_is_twelve_l2226_222675


namespace NUMINAMATH_CALUDE_smallest_number_l2226_222674

theorem smallest_number (S : Set ℤ) (h : S = {-3, 2, -2, 0}) : 
  ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2226_222674


namespace NUMINAMATH_CALUDE_joan_has_six_balloons_l2226_222659

/-- The number of orange balloons Joan has now, given she initially had 8 and lost 2. -/
def joans_balloons : ℕ := 8 - 2

/-- Theorem stating that Joan has 6 orange balloons now. -/
theorem joan_has_six_balloons : joans_balloons = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_six_balloons_l2226_222659


namespace NUMINAMATH_CALUDE_closest_integer_to_double_sum_l2226_222661

/-- The number of distinct prime divisors of n that are at least k -/
def mho (n k : ℕ+) : ℕ := sorry

/-- The double sum in the problem -/
noncomputable def doubleSum : ℝ := sorry

theorem closest_integer_to_double_sum : 
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 1/2 ∧ doubleSum = 167 + ε := by sorry

end NUMINAMATH_CALUDE_closest_integer_to_double_sum_l2226_222661


namespace NUMINAMATH_CALUDE_possible_m_values_l2226_222628

theorem possible_m_values (A B : Set ℝ) (m : ℝ) : 
  A = {-1, 1} →
  B = {x | m * x = 1} →
  A ∪ B = A →
  m = 0 ∨ m = 1 ∨ m = -1 := by
sorry

end NUMINAMATH_CALUDE_possible_m_values_l2226_222628


namespace NUMINAMATH_CALUDE_min_value_ab_l2226_222627

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2 / a + 3 / b = Real.sqrt (a * b)) : 
  ∀ x y : ℝ, x > 0 → y > 0 → 2 / x + 3 / y = Real.sqrt (x * y) → a * b ≤ x * y :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l2226_222627


namespace NUMINAMATH_CALUDE_sum_s_r_equals_negative_62_l2226_222685

def r (x : ℝ) : ℝ := abs x + 1

def s (x : ℝ) : ℝ := -2 * abs x

def xValues : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_s_r_equals_negative_62 :
  (xValues.map (fun x => s (r x))).sum = -62 := by
  sorry

end NUMINAMATH_CALUDE_sum_s_r_equals_negative_62_l2226_222685


namespace NUMINAMATH_CALUDE_remove_one_gives_average_seven_point_five_l2226_222691

def original_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13]

def remove_element (l : List ℕ) (n : ℕ) : List ℕ :=
  l.filter (λ x => x ≠ n)

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem remove_one_gives_average_seven_point_five :
  average (remove_element original_list 1) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_remove_one_gives_average_seven_point_five_l2226_222691


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2226_222656

/-- The number of distinct diagonals in a convex nonagon -/
def diagonals_in_nonagon : ℕ := 27

/-- A convex polygon with 9 sides -/
structure Nonagon where
  sides : ℕ
  convex : Bool
  is_nonagon : sides = 9 ∧ convex = true

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonals (n : Nonagon) : diagonals_in_nonagon = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2226_222656


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2226_222614

variable (a b : ℝ × ℝ)

theorem vector_magnitude_proof 
  (h1 : ‖a - 2 • b‖ = 1) 
  (h2 : a • b = 1) : 
  ‖a + 2 • b‖ = 3 := by sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2226_222614


namespace NUMINAMATH_CALUDE_a_3_value_l2226_222601

def sequence_a (n : ℕ+) : ℚ := 1 / (n.val * (n.val + 1))

theorem a_3_value : sequence_a 3 = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_a_3_value_l2226_222601


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2226_222641

theorem arithmetic_sequence_sum : 
  let a₁ : ℤ := -5  -- First term
  let d : ℤ := 3    -- Common difference
  let n : ℕ := 20   -- Number of terms
  let S := n * (2 * a₁ + (n - 1) * d) / 2  -- Sum formula for arithmetic sequence
  S = 470
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2226_222641


namespace NUMINAMATH_CALUDE_absolute_value_and_exponent_zero_sum_l2226_222692

theorem absolute_value_and_exponent_zero_sum : |-5| + (2 - Real.sqrt 3)^0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponent_zero_sum_l2226_222692


namespace NUMINAMATH_CALUDE_shooting_game_probability_l2226_222637

theorem shooting_game_probability (A B : Type) 
  (hit_score : ℕ) (miss_score : ℕ) 
  (A_hit_rate : ℚ) (B_hit_rate : ℚ) 
  (sum_two_prob : ℚ) :
  hit_score = 2 →
  miss_score = 0 →
  A_hit_rate = 3/5 →
  sum_two_prob = 9/20 →
  (A_hit_rate * (1 - B_hit_rate) + (1 - A_hit_rate) * B_hit_rate = sum_two_prob) →
  B_hit_rate = 3/4 := by
sorry

end NUMINAMATH_CALUDE_shooting_game_probability_l2226_222637


namespace NUMINAMATH_CALUDE_min_value_sin_cos_l2226_222619

theorem min_value_sin_cos (α β : ℝ) (h1 : α ≥ 0) (h2 : β ≥ 0) (h3 : α + β ≤ 2 * Real.pi) :
  Real.sin α + 2 * Real.cos β ≥ -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sin_cos_l2226_222619


namespace NUMINAMATH_CALUDE_min_value_fraction_l2226_222694

theorem min_value_fraction (x : ℝ) (h : x > 9) : 
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2226_222694


namespace NUMINAMATH_CALUDE_vertex_in_second_quadrant_l2226_222648

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 1)^2 + 2

-- Define the vertex of the quadratic function
def vertex : ℝ × ℝ := (-1, 2)

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem vertex_in_second_quadrant :
  in_second_quadrant vertex := by sorry

end NUMINAMATH_CALUDE_vertex_in_second_quadrant_l2226_222648


namespace NUMINAMATH_CALUDE_base_10_515_equals_base_6_2215_l2226_222672

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (a b c d : ℕ) : ℕ :=
  a * 6^3 + b * 6^2 + c * 6^1 + d * 6^0

/-- Theorem stating that 515 in base 10 is equal to 2215 in base 6 --/
theorem base_10_515_equals_base_6_2215 :
  515 = base6ToBase10 2 2 1 5 := by
  sorry

end NUMINAMATH_CALUDE_base_10_515_equals_base_6_2215_l2226_222672


namespace NUMINAMATH_CALUDE_no_double_composition_f_l2226_222651

def q : ℕ+ → ℕ+ :=
  fun n => match n with
  | 1 => 3
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | _ => n

theorem no_double_composition_f (f : ℕ+ → ℕ+) :
  ¬(∀ n : ℕ+, f (f n) = q n + 2) :=
sorry

end NUMINAMATH_CALUDE_no_double_composition_f_l2226_222651


namespace NUMINAMATH_CALUDE_sum_of_ages_l2226_222665

theorem sum_of_ages (henry_age jill_age : ℕ) : 
  henry_age = 20 →
  jill_age = 13 →
  henry_age - 6 = 2 * (jill_age - 6) →
  henry_age + jill_age = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2226_222665


namespace NUMINAMATH_CALUDE_twenty_fifth_term_is_173_l2226_222686

/-- The nth term of an arithmetic progression -/
def arithmetic_progression (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 25th term of the arithmetic progression with first term 5 and common difference 7 is 173 -/
theorem twenty_fifth_term_is_173 :
  arithmetic_progression 5 7 25 = 173 := by
  sorry

end NUMINAMATH_CALUDE_twenty_fifth_term_is_173_l2226_222686


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2226_222625

theorem perpendicular_lines_a_values (a : ℝ) :
  (∀ x y : ℝ, 3 * a * x - y - 1 = 0 ∧ (a - 1) * x + y + 1 = 0 →
    (3 * a * ((a - 1) * x + y + 1) + (-1) * (3 * a * x - y - 1) = 0)) →
  a = -1 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2226_222625


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l2226_222655

def Q (n : ℕ) : ℚ := 4 / ((n + 2) * (n + 3))

theorem smallest_n_for_Q_less_than_threshold : 
  (∃ n : ℕ, Q n < 1/4022) ∧ 
  (∀ m : ℕ, m < 62 → Q m ≥ 1/4022) ∧ 
  Q 62 < 1/4022 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l2226_222655


namespace NUMINAMATH_CALUDE_money_sharing_l2226_222630

theorem money_sharing (total : ℝ) (maggie_share : ℝ) : 
  maggie_share = 0.75 * total ∧ maggie_share = 4500 → total = 6000 :=
by sorry

end NUMINAMATH_CALUDE_money_sharing_l2226_222630


namespace NUMINAMATH_CALUDE_lowest_two_digit_product_12_l2226_222602

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem lowest_two_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_lowest_two_digit_product_12_l2226_222602


namespace NUMINAMATH_CALUDE_sphere_radius_ratio_l2226_222632

theorem sphere_radius_ratio (V₁ V₂ : ℝ) (h₁ : V₁ = 512 * Real.pi) (h₂ : V₂ = 32 * Real.pi) :
  (V₂ / V₁) ^ (1/3 : ℝ) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_ratio_l2226_222632


namespace NUMINAMATH_CALUDE_boat_stream_speed_l2226_222622

/-- Proves that the speed of the stream is 6 kmph given the conditions of the boat problem -/
theorem boat_stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ)
  (h_boat_speed : boat_speed = 8)
  (h_distance : distance = 210)
  (h_total_time : total_time = 120)
  (h_equation : (distance / (boat_speed - stream_speed)) + (distance / (boat_speed + stream_speed)) = total_time)
  : stream_speed = 6 := by
  sorry

#check boat_stream_speed

end NUMINAMATH_CALUDE_boat_stream_speed_l2226_222622


namespace NUMINAMATH_CALUDE_f_shifted_f_identity_l2226_222687

-- Define the function f
def f : ℝ → ℝ := fun x => x^2 + 1

-- State the theorem
theorem f_shifted (x : ℝ) : f (x - 1) = x^2 - 2*x + 2 := by
  sorry

-- Prove that f(x) = x^2 + 1
theorem f_identity (x : ℝ) : f x = x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_shifted_f_identity_l2226_222687


namespace NUMINAMATH_CALUDE_portfolio_growth_l2226_222609

/-- Calculates the final portfolio value after two years of investment -/
theorem portfolio_growth (initial_investment : ℝ) (growth_rate_1 : ℝ) (additional_investment : ℝ) (growth_rate_2 : ℝ) 
  (h1 : initial_investment = 80)
  (h2 : growth_rate_1 = 0.15)
  (h3 : additional_investment = 28)
  (h4 : growth_rate_2 = 0.10) :
  let value_after_year_1 := initial_investment * (1 + growth_rate_1)
  let value_before_year_2 := value_after_year_1 + additional_investment
  let final_value := value_before_year_2 * (1 + growth_rate_2)
  final_value = 132 := by
  sorry

end NUMINAMATH_CALUDE_portfolio_growth_l2226_222609


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2226_222679

theorem negation_of_proposition (p : ℕ → Prop) : 
  (¬∀ n : ℕ, 3^n ≥ n + 1) ↔ (∃ n : ℕ, 3^n < n + 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2226_222679


namespace NUMINAMATH_CALUDE_r_amount_unchanged_l2226_222667

/-- Represents the financial situation of three friends P, Q, and R. -/
structure FriendFinances where
  p : ℝ  -- Amount with P
  q : ℝ  -- Amount with Q
  r : ℝ  -- Amount with R

/-- The total amount among the three friends is 4000. -/
def total_amount (f : FriendFinances) : Prop :=
  f.p + f.q + f.r = 4000

/-- R has two-thirds of the total amount with P and Q. -/
def r_two_thirds_pq (f : FriendFinances) : Prop :=
  f.r = (2/3) * (f.p + f.q)

/-- The ratio of amount with P to amount with Q is 3:2. -/
def p_q_ratio (f : FriendFinances) : Prop :=
  f.p / f.q = 3/2

/-- 10% of P's amount will be donated to charity. -/
def charity_donation (f : FriendFinances) : ℝ :=
  0.1 * f.p

/-- Theorem stating that R's amount remains unchanged after P's charity donation. -/
theorem r_amount_unchanged (f : FriendFinances) 
  (h1 : total_amount f) 
  (h2 : r_two_thirds_pq f) 
  (h3 : p_q_ratio f) : 
  f.r = 1600 :=
sorry

end NUMINAMATH_CALUDE_r_amount_unchanged_l2226_222667


namespace NUMINAMATH_CALUDE_mark_tree_count_l2226_222693

/-- Calculates the final number of trees after planting and removing sessions -/
def final_tree_count (x y : ℕ) (plant_rate remove_rate : ℕ) : ℤ :=
  let days : ℕ := y / plant_rate
  let removed : ℕ := days * remove_rate
  (x : ℤ) + (y : ℤ) - (removed : ℤ)

/-- Theorem stating the final number of trees after Mark's planting session -/
theorem mark_tree_count (x : ℕ) : final_tree_count x 12 2 3 = (x : ℤ) - 6 := by
  sorry

end NUMINAMATH_CALUDE_mark_tree_count_l2226_222693


namespace NUMINAMATH_CALUDE_library_books_l2226_222647

theorem library_books (borrowed : ℕ) (left : ℕ) (initial : ℕ) : 
  borrowed = 18 → left = 57 → initial = borrowed + left → initial = 75 := by
  sorry

end NUMINAMATH_CALUDE_library_books_l2226_222647


namespace NUMINAMATH_CALUDE_petes_number_l2226_222671

theorem petes_number (x : ℝ) : 5 * (3 * x - 6) = 195 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l2226_222671
