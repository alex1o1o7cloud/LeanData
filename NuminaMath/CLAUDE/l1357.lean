import Mathlib

namespace NUMINAMATH_CALUDE_bright_numbers_l1357_135739

def isBright (x : ℕ) : Prop :=
  ∃ a b : ℕ, x = a^2 + b^3

theorem bright_numbers (r s : ℕ+) :
  (∃ f : ℕ → ℕ, StrictMono f ∧ ∀ i, isBright (r + f i) ∧ isBright (s + f i)) ∧
  (∃ g : ℕ → ℕ, StrictMono g ∧ ∀ i, isBright (r * g i) ∧ isBright (s * g i)) := by
  sorry

end NUMINAMATH_CALUDE_bright_numbers_l1357_135739


namespace NUMINAMATH_CALUDE_log_216_simplification_l1357_135797

theorem log_216_simplification :
  (216 : ℝ) = 6^3 →
  Real.log 216 = 3 * (Real.log 2 + Real.log 3) := by
sorry

end NUMINAMATH_CALUDE_log_216_simplification_l1357_135797


namespace NUMINAMATH_CALUDE_min_value_expression_l1357_135759

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 3) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (3/c - 1)^2 ≥ 4 * (9^(1/4) - 5/4)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1357_135759


namespace NUMINAMATH_CALUDE_correct_sunset_time_l1357_135736

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

def addTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes + (t.hours + d.hours) * 60
  { hours := totalMinutes / 60 % 24,
    minutes := totalMinutes % 60 }

def sunsetTime (sunrise : Time) (daylight : Duration) : Time :=
  addTime sunrise daylight

theorem correct_sunset_time :
  let sunrise : Time := { hours := 16, minutes := 35 }
  let daylight : Duration := { hours := 9, minutes := 48 }
  sunsetTime sunrise daylight = { hours := 2, minutes := 23 } := by
  sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l1357_135736


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1357_135790

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1357_135790


namespace NUMINAMATH_CALUDE_optimal_tic_tac_toe_draw_l1357_135760

/-- Represents a player in Tic-Tac-Toe -/
inductive Player : Type
| X : Player
| O : Player

/-- Represents a position on the Tic-Tac-Toe board -/
inductive Position : Type
| one | two | three | four | five | six | seven | eight | nine

/-- Represents the state of a Tic-Tac-Toe game -/
structure GameState :=
  (board : Position → Option Player)
  (currentPlayer : Player)

/-- Represents an optimal move in Tic-Tac-Toe -/
def OptimalMove : GameState → Position → Prop := sorry

/-- Represents the outcome of a Tic-Tac-Toe game -/
inductive GameOutcome : Type
| Draw : GameOutcome
| Win : Player → GameOutcome

/-- Plays a full game of Tic-Tac-Toe with optimal moves -/
def playOptimalGame : GameState → GameOutcome := sorry

/-- Theorem: Every game of Tic-Tac-Toe between optimal players ends in a draw -/
theorem optimal_tic_tac_toe_draw :
  ∀ (initialState : GameState),
  (∀ (state : GameState) (move : Position), OptimalMove state move → 
    playOptimalGame (sorry : GameState) = playOptimalGame state) →
  playOptimalGame initialState = GameOutcome.Draw :=
sorry

end NUMINAMATH_CALUDE_optimal_tic_tac_toe_draw_l1357_135760


namespace NUMINAMATH_CALUDE_negation_equivalence_l1357_135735

-- Define the universe of discourse
variable (Teacher : Type)

-- Define the predicates
variable (loves_math : Teacher → Prop)
variable (dislikes_math : Teacher → Prop)

-- State the theorem
theorem negation_equivalence :
  (∃ t : Teacher, dislikes_math t) ↔ ¬(∀ t : Teacher, loves_math t) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1357_135735


namespace NUMINAMATH_CALUDE_banana_change_l1357_135716

/-- Calculates the change received when buying bananas -/
theorem banana_change (num_bananas : ℕ) (cost_per_banana : ℚ) (amount_paid : ℚ) :
  num_bananas = 5 →
  cost_per_banana = 30 / 100 →
  amount_paid = 10 →
  amount_paid - (num_bananas : ℚ) * cost_per_banana = 17 / 2 :=
by sorry

end NUMINAMATH_CALUDE_banana_change_l1357_135716


namespace NUMINAMATH_CALUDE_problem_statement_l1357_135726

theorem problem_statement (x y : ℤ) (hx : x = 12) (hy : y = 7) :
  (x - y)^2 * (x + y)^2 = 9025 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1357_135726


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1357_135712

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt ((6 / (x + 1)) - 1)}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (-x^2 + 2*x + 3) / Real.log 10}

-- Define the complement of B in ℝ
def complement_B : Set ℝ := {x | x ∉ B}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ complement_B = {x | 3 ≤ x ∧ x ≤ 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1357_135712


namespace NUMINAMATH_CALUDE_interval_equivalence_l1357_135751

theorem interval_equivalence (x : ℝ) : 
  (1/4 < x ∧ x < 1/2) ↔ (1 < 5*x ∧ 5*x < 3) ∧ (2 < 8*x ∧ 8*x < 4) := by
  sorry

end NUMINAMATH_CALUDE_interval_equivalence_l1357_135751


namespace NUMINAMATH_CALUDE_bills_speed_day2_l1357_135705

/-- Represents Bill's two-day drive --/
structure TwoDayDrive where
  speed_day2 : ℝ
  time_day2 : ℝ
  total_distance : ℝ
  total_time : ℝ

/-- The conditions of Bill's drive satisfy the problem statement --/
def satisfies_conditions (d : TwoDayDrive) : Prop :=
  d.total_distance = (d.speed_day2 + 5) * (d.time_day2 + 2) + d.speed_day2 * d.time_day2 ∧
  d.total_time = d.time_day2 + 2 + d.time_day2 ∧
  d.total_distance = 680 ∧
  d.total_time = 18

/-- Theorem stating that Bill's average speed on the second day was 35 mph --/
theorem bills_speed_day2 (d : TwoDayDrive) (h : satisfies_conditions d) :
  d.speed_day2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_bills_speed_day2_l1357_135705


namespace NUMINAMATH_CALUDE_jordan_wins_l1357_135773

theorem jordan_wins (peter_wins peter_losses emma_wins emma_losses jordan_losses : ℕ)
  (h1 : peter_wins = 5)
  (h2 : peter_losses = 4)
  (h3 : emma_wins = 4)
  (h4 : emma_losses = 5)
  (h5 : jordan_losses = 2) :
  ∃ jordan_wins : ℕ,
    jordan_wins = 2 ∧
    2 * (peter_wins + peter_losses + emma_wins + emma_losses + jordan_wins + jordan_losses) =
    peter_wins + emma_wins + jordan_wins + peter_losses + emma_losses + jordan_losses :=
by sorry

end NUMINAMATH_CALUDE_jordan_wins_l1357_135773


namespace NUMINAMATH_CALUDE_odd_divisors_implies_perfect_square_l1357_135793

theorem odd_divisors_implies_perfect_square (n : ℕ) : 
  (Odd (Nat.card {d : ℕ | d ∣ n})) → ∃ m : ℕ, n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_odd_divisors_implies_perfect_square_l1357_135793


namespace NUMINAMATH_CALUDE_intersection_condition_l1357_135711

/-- Set A defined by the given conditions -/
def set_A (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | m/2 ≤ (p.1 - 2)^2 + p.2^2 ∧ (p.1 - 2)^2 + p.2^2 ≤ m^2}

/-- Set B defined by the given conditions -/
def set_B (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2*m ≤ p.1 + p.2 ∧ p.1 + p.2 ≤ 2*m + 1}

/-- The main theorem stating the condition for non-empty intersection of A and B -/
theorem intersection_condition (m : ℝ) :
  (set_A m ∩ set_B m).Nonempty ↔ 1/2 ≤ m ∧ m ≤ 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l1357_135711


namespace NUMINAMATH_CALUDE_quadratic_function_value_l1357_135701

/-- Given a quadratic function f(x) = -(x+h)^2 with axis of symmetry at x=-3,
    prove that f(0) = -9 -/
theorem quadratic_function_value (h : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = -(x + h)^2) →
  (∀ x < -3, ∀ y > x, f y > f x) →
  (∀ x > -3, ∀ y > x, f y < f x) →
  f 0 = -9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l1357_135701


namespace NUMINAMATH_CALUDE_both_runners_in_picture_probability_l1357_135744

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Calculates the probability of both runners being in a picture -/
def probability_both_in_picture (rachel : Runner) (robert : Runner) : ℚ :=
  sorry

/-- Main theorem: The probability of both runners being in the picture is 3/16 -/
theorem both_runners_in_picture_probability :
  let rachel : Runner := { lapTime := 90, direction := true }
  let robert : Runner := { lapTime := 80, direction := false }
  probability_both_in_picture rachel robert = 3 / 16 := by sorry

end NUMINAMATH_CALUDE_both_runners_in_picture_probability_l1357_135744


namespace NUMINAMATH_CALUDE_square_field_area_l1357_135713

/-- Given a square field with two 1-meter wide gates, where the cost of drawing barbed wire
    is 1.10 per meter and the total cost is 732.6, prove that the area of the field is 27889 sq m. -/
theorem square_field_area (side : ℝ) (gate_width : ℝ) (wire_cost_per_meter : ℝ) (total_cost : ℝ)
  (h1 : gate_width = 1)
  (h2 : wire_cost_per_meter = 1.1)
  (h3 : total_cost = 732.6)
  (h4 : wire_cost_per_meter * (4 * side - 2 * gate_width) = total_cost) :
  side^2 = 27889 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l1357_135713


namespace NUMINAMATH_CALUDE_lunchroom_tables_l1357_135772

theorem lunchroom_tables (total_students : ℕ) (students_per_table : ℕ) (h1 : total_students = 204) (h2 : students_per_table = 6) :
  total_students / students_per_table = 34 := by
  sorry

end NUMINAMATH_CALUDE_lunchroom_tables_l1357_135772


namespace NUMINAMATH_CALUDE_no_three_primes_arithmetic_progression_no_k_primes_arithmetic_progression_l1357_135718

theorem no_three_primes_arithmetic_progression (p₁ p₂ p₃ : ℕ) (d : ℕ) : 
  p₁ > 3 → p₂ > 3 → p₃ > 3 → 
  Nat.Prime p₁ → Nat.Prime p₂ → Nat.Prime p₃ → 
  d < 5 → 
  ¬(p₂ = p₁ + d ∧ p₃ = p₁ + 2*d) :=
sorry

theorem no_k_primes_arithmetic_progression (k : ℕ) (p : ℕ → ℕ) (d : ℕ) :
  k > 3 → 
  (∀ i, i ≤ k → p i > k) →
  (∀ i, i ≤ k → Nat.Prime (p i)) →
  d ≤ k + 1 →
  ¬(∀ i, i ≤ k → p i = p 1 + (i - 1) * d) :=
sorry

end NUMINAMATH_CALUDE_no_three_primes_arithmetic_progression_no_k_primes_arithmetic_progression_l1357_135718


namespace NUMINAMATH_CALUDE_max_b_value_l1357_135706

/-- Given two functions f and g with a common point and equal tangents, 
    prove the maximum value of b -/
theorem max_b_value (a : ℝ) (h_a : a > 0) :
  let f := fun x : ℝ => (1/2) * x^2 + 2*a*x
  let g := fun x b : ℝ => 3*a^2 * Real.log x + b
  ∃ (x₀ b : ℝ), 
    (f x₀ = g x₀ b) ∧ 
    (deriv f x₀ = deriv (fun x => g x b) x₀) →
    (∀ b' : ℝ, ∃ x : ℝ, f x = g x b' → b' ≤ (3/2) * Real.exp ((2:ℝ)/3)) :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l1357_135706


namespace NUMINAMATH_CALUDE_max_cubes_for_given_prism_l1357_135715

/-- Represents the dimensions and properties of a wooden rectangular prism --/
structure WoodenPrism where
  totalSurfaceArea : ℝ
  cubeSurfaceArea : ℝ
  wastePerCut : ℝ

/-- Calculates the maximum number of cubes that can be sawed from the prism --/
def maxCubes (prism : WoodenPrism) : ℕ :=
  sorry

/-- Theorem stating the maximum number of cubes for the given problem --/
theorem max_cubes_for_given_prism :
  let prism : WoodenPrism := {
    totalSurfaceArea := 2448,
    cubeSurfaceArea := 216,
    wastePerCut := 0.2
  }
  maxCubes prism = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_for_given_prism_l1357_135715


namespace NUMINAMATH_CALUDE_calculation_proof_l1357_135731

theorem calculation_proof : (((20^10 / 20^9)^3 * 10^6) / 2^12) = 1953125 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1357_135731


namespace NUMINAMATH_CALUDE_quadratic_max_value_l1357_135741

def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem quadratic_max_value (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 3, f x m ≤ 1) ∧ 
  (∃ x ∈ Set.Icc 0 3, f x m = 1) → 
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l1357_135741


namespace NUMINAMATH_CALUDE_three_cuts_make_5x5_mat_l1357_135746

/-- Represents a rectangular piece of cloth -/
structure Cloth where
  rows : ℕ
  cols : ℕ
  checkered : Bool

/-- Represents a cut on the cloth -/
inductive Cut
  | Vertical (col : ℕ)
  | Horizontal (row : ℕ)

/-- Represents the result of cutting a cloth -/
def cut_result (c : Cloth) (cut : Cut) : Cloth × Cloth :=
  match cut with
  | Cut.Vertical col => ⟨⟨c.rows, col, c.checkered⟩, ⟨c.rows, c.cols - col, c.checkered⟩⟩
  | Cut.Horizontal row => ⟨⟨row, c.cols, c.checkered⟩, ⟨c.rows - row, c.cols, c.checkered⟩⟩

/-- Checks if a cloth can form a 5x5 mat -/
def is_5x5_mat (c : Cloth) : Bool :=
  c.rows = 5 && c.cols = 5 && c.checkered

/-- The main theorem -/
theorem three_cuts_make_5x5_mat :
  ∃ (cut1 cut2 cut3 : Cut),
    let initial_cloth := Cloth.mk 6 7 true
    let (c1, c2) := cut_result initial_cloth cut1
    let (c3, c4) := cut_result c1 cut2
    let (c5, c6) := cut_result c2 cut3
    ∃ (final_cloth : Cloth),
      is_5x5_mat final_cloth ∧
      (final_cloth.rows * final_cloth.cols =
       c3.rows * c3.cols + c4.rows * c4.cols + c5.rows * c5.cols + c6.rows * c6.cols) :=
by
  sorry


end NUMINAMATH_CALUDE_three_cuts_make_5x5_mat_l1357_135746


namespace NUMINAMATH_CALUDE_sphere_volume_l1357_135784

theorem sphere_volume (surface_area : Real) (volume : Real) : 
  surface_area = 100 * Real.pi → volume = (500 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l1357_135784


namespace NUMINAMATH_CALUDE_min_fraction_sum_l1357_135780

def Digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

theorem min_fraction_sum :
  ∃ (W X Y Z : ℕ), W ∈ Digits ∧ X ∈ Digits ∧ Y ∈ Digits ∧ Z ∈ Digits ∧
  W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
  ∀ (W' X' Y' Z' : ℕ), W' ∈ Digits → X' ∈ Digits → Y' ∈ Digits → Z' ∈ Digits →
  W' ≠ X' → W' ≠ Y' → W' ≠ Z' → X' ≠ Y' → X' ≠ Z' → Y' ≠ Z' →
  (W : ℚ) / X + (Y : ℚ) / Z ≤ (W' : ℚ) / X' + (Y' : ℚ) / Z' ∧
  (W : ℚ) / X + (Y : ℚ) / Z = 15 / 56 :=
by sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l1357_135780


namespace NUMINAMATH_CALUDE_successive_numbers_product_l1357_135729

theorem successive_numbers_product (n : ℕ) : 
  n * (n + 1) = 2652 → n = 51 := by
  sorry

end NUMINAMATH_CALUDE_successive_numbers_product_l1357_135729


namespace NUMINAMATH_CALUDE_rectangle_overlap_theorem_l1357_135747

/-- A rectangle with a given area -/
structure Rectangle where
  area : ℝ

/-- A configuration of rectangles placed within a larger rectangle -/
structure Configuration where
  outer : Rectangle
  inner : List Rectangle

/-- Predicate to check if two rectangles overlap by at least a given area -/
def overlaps (r1 r2 : Rectangle) (min_overlap : ℝ) : Prop :=
  ∃ (overlap_area : ℝ), overlap_area ≥ min_overlap

theorem rectangle_overlap_theorem (config : Configuration) :
  config.outer.area = 5 →
  config.inner.length = 9 →
  ∀ r ∈ config.inner, r.area = 1 →
  ∃ (r1 r2 : Rectangle), r1 ∈ config.inner ∧ r2 ∈ config.inner ∧ r1 ≠ r2 ∧ overlaps r1 r2 (1/9) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_overlap_theorem_l1357_135747


namespace NUMINAMATH_CALUDE_equation_solution_l1357_135702

theorem equation_solution (y : ℝ) (h : y ≠ 2) :
  (7 * y / (y - 2) - 4 / (y - 2) = 3 / (y - 2)) ↔ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1357_135702


namespace NUMINAMATH_CALUDE_odd_function_iff_condition_l1357_135737

def f (x a b : ℝ) : ℝ := x * |x + a| + b

theorem odd_function_iff_condition (a b : ℝ) :
  (∀ x, f x a b = -f (-x) a b) ↔ a^2 + b^2 = 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_iff_condition_l1357_135737


namespace NUMINAMATH_CALUDE_boxes_per_hand_for_seven_people_l1357_135721

/-- Given a group of people and the total number of boxes they can hold, 
    calculate the number of boxes one person can hold in each hand. -/
def boxes_per_hand (num_people : ℕ) (total_boxes : ℕ) : ℕ :=
  (total_boxes / num_people) / 2

/-- Theorem stating that given 7 people holding 14 boxes in total, 
    each person can hold 1 box in each hand. -/
theorem boxes_per_hand_for_seven_people : 
  boxes_per_hand 7 14 = 1 := by sorry

end NUMINAMATH_CALUDE_boxes_per_hand_for_seven_people_l1357_135721


namespace NUMINAMATH_CALUDE_shifted_graph_symmetry_l1357_135700

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (x / 2) + Real.sin (x / 2)

theorem shifted_graph_symmetry (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, f (x + m) = f (-x + m)) ↔ m ≥ 4 * Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_shifted_graph_symmetry_l1357_135700


namespace NUMINAMATH_CALUDE_inequality_solution_l1357_135723

theorem inequality_solution (x : ℝ) : 
  3/20 + |x - 13/60| < 7/30 ↔ 2/15 < x ∧ x < 3/10 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1357_135723


namespace NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l1357_135757

theorem ceiling_fraction_evaluation : 
  (⌈(23 : ℝ) / 9 - ⌈(35 : ℝ) / 23⌉⌉) / (⌈(35 : ℝ) / 9 + ⌈(9 : ℝ) * 23 / 35⌉⌉) = (1 : ℝ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_fraction_evaluation_l1357_135757


namespace NUMINAMATH_CALUDE_probability_red_or_blue_l1357_135765

theorem probability_red_or_blue 
  (prob_red : ℝ) 
  (prob_red_or_yellow : ℝ) 
  (h1 : prob_red = 0.45) 
  (h2 : prob_red_or_yellow = 0.65) 
  : prob_red + (1 - prob_red_or_yellow) = 0.80 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_blue_l1357_135765


namespace NUMINAMATH_CALUDE_difference_of_squares_2006_l1357_135794

def is_difference_of_squares (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 - b^2

theorem difference_of_squares_2006 :
  ¬(is_difference_of_squares 2006) ∧
  (is_difference_of_squares 2004) ∧
  (is_difference_of_squares 2005) ∧
  (is_difference_of_squares 2007) :=
sorry

end NUMINAMATH_CALUDE_difference_of_squares_2006_l1357_135794


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1357_135745

theorem quadratic_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 11 = 25) → 
  (d^2 - 6*d + 11 = 25) → 
  c ≥ d → 
  c + 2*d = 9 - Real.sqrt 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1357_135745


namespace NUMINAMATH_CALUDE_sum_product_inequality_l1357_135756

theorem sum_product_inequality (a b c x y z k : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hk1 : a + x = k) (hk2 : b + y = k) (hk3 : c + z = k) : 
  a * y + b * z + c * x < k^2 := by
sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l1357_135756


namespace NUMINAMATH_CALUDE_john_gets_55_messages_l1357_135709

/-- The number of text messages John used to get per day -/
def old_messages_per_day : ℕ := 20

/-- The number of unintended text messages John gets per week -/
def unintended_messages_per_week : ℕ := 245

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Calculates the total number of text messages John gets per day now -/
def total_messages_per_day : ℕ :=
  old_messages_per_day + unintended_messages_per_week / days_per_week

/-- Theorem stating that John now gets 55 text messages per day -/
theorem john_gets_55_messages : total_messages_per_day = 55 := by
  sorry

end NUMINAMATH_CALUDE_john_gets_55_messages_l1357_135709


namespace NUMINAMATH_CALUDE_equation_solution_l1357_135799

theorem equation_solution :
  ∃ y : ℝ, y > 0 ∧ 6 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + 2 * y^(1/3) ∧ y = 1000 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1357_135799


namespace NUMINAMATH_CALUDE_problem_solution_l1357_135733

theorem problem_solution : ∃ m : ℚ, 15 + m * (25/3) = 6 * (25/3) - 10 ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1357_135733


namespace NUMINAMATH_CALUDE_randy_piggy_bank_l1357_135717

/-- Calculates the initial amount in Randy's piggy bank -/
def initial_amount (spend_per_trip : ℕ) (trips_per_month : ℕ) (months_per_year : ℕ) (amount_left : ℕ) : ℕ :=
  spend_per_trip * trips_per_month * months_per_year + amount_left

/-- Proves that Randy initially had $200 in his piggy bank -/
theorem randy_piggy_bank : initial_amount 2 4 12 104 = 200 := by
  sorry

end NUMINAMATH_CALUDE_randy_piggy_bank_l1357_135717


namespace NUMINAMATH_CALUDE_intersection_point_a_value_l1357_135785

-- Define the three lines
def line1 (a x y : ℝ) : Prop := a * x + 2 * y + 8 = 0
def line2 (x y : ℝ) : Prop := 4 * x + 3 * y = 10
def line3 (x y : ℝ) : Prop := 2 * x - y = 10

-- Theorem statement
theorem intersection_point_a_value :
  ∃! (a : ℝ), ∃! (x y : ℝ), line1 a x y ∧ line2 x y ∧ line3 x y → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_a_value_l1357_135785


namespace NUMINAMATH_CALUDE_job_completion_time_l1357_135778

theorem job_completion_time 
  (m d r : ℕ) 
  (h1 : m > 0) 
  (h2 : d > 0) 
  (h3 : m + r > 0) : 
  (m * d : ℚ) / (m + r) = (m * d : ℕ) / (m + r) := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1357_135778


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l1357_135791

theorem geometric_arithmetic_sequence :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive numbers
    a = 2 ∧  -- First number is 2
    b / a = c / b ∧  -- Geometric sequence
    (b + 4 - a = c - (b + 4)) ∧  -- Arithmetic sequence when 4 is added to b
    a = 2 ∧ b = 6 ∧ c = 18 :=  -- The solution
by
  sorry


end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l1357_135791


namespace NUMINAMATH_CALUDE_acrobats_count_correct_unique_solution_l1357_135725

/-- Represents the number of acrobats in the circus show -/
def num_acrobats : ℕ := 10

/-- Represents the number of elephants in the circus show -/
def num_elephants : ℕ := 5

/-- The total number of legs observed in the circus show -/
def total_legs : ℕ := 40

/-- The total number of heads observed in the circus show -/
def total_heads : ℕ := 15

/-- Theorem stating that the number of acrobats is correct given the conditions -/
theorem acrobats_count_correct :
  (2 * num_acrobats + 4 * num_elephants = total_legs) ∧
  (num_acrobats + num_elephants = total_heads) :=
by sorry

/-- Theorem proving the uniqueness of the solution -/
theorem unique_solution (a e : ℕ) :
  (2 * a + 4 * e = total_legs) →
  (a + e = total_heads) →
  a = num_acrobats ∧ e = num_elephants :=
by sorry

end NUMINAMATH_CALUDE_acrobats_count_correct_unique_solution_l1357_135725


namespace NUMINAMATH_CALUDE_power_function_value_l1357_135783

-- Define a power function type
def PowerFunction := ℝ → ℝ

-- Define the property of passing through a point for a power function
def PassesThroughPoint (f : PowerFunction) (x y : ℝ) : Prop :=
  f x = y

-- State the theorem
theorem power_function_value (f : PowerFunction) :
  PassesThroughPoint f 9 (1/3) → f 25 = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l1357_135783


namespace NUMINAMATH_CALUDE_curve_properties_l1357_135777

-- Define the curve
def on_curve (x y : ℝ) : Prop := Real.sqrt x + Real.sqrt y = 1

-- Theorem statement
theorem curve_properties :
  (∀ a b : ℝ, on_curve a b → on_curve b a) ∧
  on_curve 0 1 ∧
  on_curve 1 0 ∧
  on_curve (1/4) (1/4) :=
by sorry

end NUMINAMATH_CALUDE_curve_properties_l1357_135777


namespace NUMINAMATH_CALUDE_geometric_sequence_a1_l1357_135796

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_a1 (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 3 = 1 →
  (a 5 + (3/2) * a 4) / 2 = 1/2 →
  a 1 = 4 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_a1_l1357_135796


namespace NUMINAMATH_CALUDE_x_range_l1357_135743

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing : ∀ x y, 0 ≤ x ∧ x < y → f y < f x

-- State the theorem
theorem x_range (x : ℝ) :
  (f (x - 2) > f 3) → -1 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1357_135743


namespace NUMINAMATH_CALUDE_marys_maximum_earnings_l1357_135719

/-- Mary's maximum weekly earnings problem -/
theorem marys_maximum_earnings :
  let max_hours : ℕ := 60
  let regular_rate : ℚ := 12
  let regular_hours : ℕ := 30
  let overtime_rate : ℚ := regular_rate * (3/2)
  let overtime_hours : ℕ := max_hours - regular_hours
  let regular_earnings : ℚ := regular_rate * regular_hours
  let overtime_earnings : ℚ := overtime_rate * overtime_hours
  regular_earnings + overtime_earnings = 900 := by
  sorry

end NUMINAMATH_CALUDE_marys_maximum_earnings_l1357_135719


namespace NUMINAMATH_CALUDE_sculpture_surface_area_l1357_135789

/-- Represents a step in the staircase sculpture -/
structure Step where
  cubes : ℕ
  exposed_front : ℕ

/-- Represents the staircase sculpture -/
def Sculpture : List Step := [
  { cubes := 6, exposed_front := 6 },
  { cubes := 5, exposed_front := 5 },
  { cubes := 4, exposed_front := 4 },
  { cubes := 2, exposed_front := 2 },
  { cubes := 1, exposed_front := 5 }
]

/-- Calculates the total exposed surface area of the sculpture -/
def total_exposed_area (sculpture : List Step) : ℕ :=
  let top_area := sculpture.map (·.cubes) |>.sum
  let side_area := sculpture.map (·.exposed_front) |>.sum
  top_area + side_area

/-- Theorem: The total exposed surface area of the sculpture is 40 square meters -/
theorem sculpture_surface_area :
  total_exposed_area Sculpture = 40 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_surface_area_l1357_135789


namespace NUMINAMATH_CALUDE_treasure_in_blown_out_dunes_l1357_135762

/-- The probability that a sand dune remains after being formed -/
def prob_remain : ℚ := 1 / 3

/-- The probability that a sand dune has a lucky coupon -/
def prob_lucky_coupon : ℚ := 2 / 3

/-- The probability that a blown-out sand dune contains both treasure and lucky coupon -/
def prob_both : ℚ := 8888888888888889 / 100000000000000000

/-- The number of blown-out sand dunes considered to find the one with treasure -/
def num_blown_out_dunes : ℕ := 8

theorem treasure_in_blown_out_dunes :
  ∃ (n : ℕ), n = num_blown_out_dunes ∧ 
  (1 : ℚ) / n * prob_lucky_coupon = prob_both ∧
  n = ⌈(1 : ℚ) / (prob_both / prob_lucky_coupon)⌉ :=
sorry

end NUMINAMATH_CALUDE_treasure_in_blown_out_dunes_l1357_135762


namespace NUMINAMATH_CALUDE_bedroom_size_problem_l1357_135748

theorem bedroom_size_problem (total_area : ℝ) (difference : ℝ) :
  total_area = 300 →
  difference = 60 →
  ∃ (smaller larger : ℝ),
    smaller + larger = total_area ∧
    larger = smaller + difference ∧
    smaller = 120 := by
  sorry

end NUMINAMATH_CALUDE_bedroom_size_problem_l1357_135748


namespace NUMINAMATH_CALUDE_bicycle_wheels_count_prove_bicycle_wheels_count_l1357_135724

theorem bicycle_wheels_count : ℕ → Prop :=
  fun b : ℕ =>
    let bicycles : ℕ := 3
    let tricycles : ℕ := 4
    let unicycles : ℕ := 7
    let tricycle_wheels : ℕ := 3
    let unicycle_wheels : ℕ := 1
    let total_wheels : ℕ := 25
    bicycles * b + tricycles * tricycle_wheels + unicycles * unicycle_wheels = total_wheels →
    b = 2

-- The proof is omitted
theorem prove_bicycle_wheels_count : bicycle_wheels_count 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheels_count_prove_bicycle_wheels_count_l1357_135724


namespace NUMINAMATH_CALUDE_modular_inverse_seven_mod_thirtysix_l1357_135774

theorem modular_inverse_seven_mod_thirtysix : 
  ∃ x : ℤ, 0 ≤ x ∧ x < 36 ∧ (7 * x) % 36 = 1 :=
by
  use 31
  sorry

end NUMINAMATH_CALUDE_modular_inverse_seven_mod_thirtysix_l1357_135774


namespace NUMINAMATH_CALUDE_multiply_125_3_2_25_solve_equation_l1357_135786

-- Part 1: Prove that 125 × 3.2 × 25 = 10000
theorem multiply_125_3_2_25 : 125 * 3.2 * 25 = 10000 := by sorry

-- Part 2: Prove that the solution to 24(x-12) = 16(x-4) is x = 28
theorem solve_equation : ∃ x : ℝ, 24 * (x - 12) = 16 * (x - 4) ∧ x = 28 := by sorry

end NUMINAMATH_CALUDE_multiply_125_3_2_25_solve_equation_l1357_135786


namespace NUMINAMATH_CALUDE_car_tank_capacity_l1357_135754

/-- Calculates the capacity of a car's gas tank given initial and final mileage, efficiency, and number of fill-ups -/
def tank_capacity (initial_mileage final_mileage : ℕ) (efficiency : ℚ) (fill_ups : ℕ) : ℚ :=
  (final_mileage - initial_mileage : ℚ) / (efficiency * fill_ups)

/-- Proves that the car's tank capacity is 20 gallons given the problem conditions -/
theorem car_tank_capacity :
  tank_capacity 1728 2928 30 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_car_tank_capacity_l1357_135754


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l1357_135753

/-- Given two vectors a and b in ℝ³, where a = (-2, 3, 1) and b = (4, m, n),
    if a is parallel to b, then m + n = -8 -/
theorem parallel_vectors_sum (m n : ℝ) : 
  let a : ℝ × ℝ × ℝ := (-2, 3, 1)
  let b : ℝ × ℝ × ℝ := (4, m, n)
  (∃ (k : ℝ), b = k • a) → m + n = -8 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l1357_135753


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1357_135769

theorem quadratic_inequality (z : ℝ) : z^2 - 40*z + 360 ≤ 16*z ↔ 8 ≤ z ∧ z ≤ 45 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1357_135769


namespace NUMINAMATH_CALUDE_periodic_function_value_l1357_135768

theorem periodic_function_value (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 2008 = -1 → f 2009 = 1 := by sorry

end NUMINAMATH_CALUDE_periodic_function_value_l1357_135768


namespace NUMINAMATH_CALUDE_candy_cookie_packs_l1357_135763

-- Define the problem parameters
def num_trays : ℕ := 4
def cookies_per_tray : ℕ := 24
def cookies_per_pack : ℕ := 12

-- Define the theorem
theorem candy_cookie_packs : 
  (num_trays * cookies_per_tray) / cookies_per_pack = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_cookie_packs_l1357_135763


namespace NUMINAMATH_CALUDE_collinear_implies_coplanar_not_coplanar_implies_not_collinear_l1357_135766

-- Define a point in space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define collinearity for three points
def collinear (p q r : Point3D) : Prop := sorry

-- Define coplanarity for four points
def coplanar (p q r s : Point3D) : Prop := sorry

-- Theorem 1: If three points are collinear, then four points are coplanar
theorem collinear_implies_coplanar (p q r s : Point3D) :
  (collinear p q r ∨ collinear p q s ∨ collinear p r s ∨ collinear q r s) →
  coplanar p q r s := by sorry

-- Theorem 2: If four points are not coplanar, then no three points are collinear
theorem not_coplanar_implies_not_collinear (p q r s : Point3D) :
  ¬(coplanar p q r s) →
  ¬(collinear p q r) ∧ ¬(collinear p q s) ∧ ¬(collinear p r s) ∧ ¬(collinear q r s) := by sorry

end NUMINAMATH_CALUDE_collinear_implies_coplanar_not_coplanar_implies_not_collinear_l1357_135766


namespace NUMINAMATH_CALUDE_min_squares_for_symmetric_x_l1357_135798

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the state of the 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Bool

/-- The initial grid with squares at (1,3) and (2,4) shaded -/
def initialGrid : Grid :=
  fun r c => (r = 0 ∧ c = 2) ∨ (r = 1 ∧ c = 3)

/-- Checks if a grid has both vertical and horizontal symmetry -/
def isSymmetric (g : Grid) : Prop :=
  (∀ r c, g r c = g r (3 - c)) ∧  -- Vertical symmetry
  (∀ r c, g r c = g (3 - r) c)    -- Horizontal symmetry

/-- Checks if a grid forms an 'X' shape -/
def formsX (g : Grid) : Prop :=
  (∀ r, g r r = true) ∧ 
  (∀ r, g r (3 - r) = true) ∧
  (∀ r c, r ≠ c ∧ r ≠ (3 - c) → g r c = false)

/-- The main theorem stating that 4 additional squares are needed -/
theorem min_squares_for_symmetric_x : 
  ∃ (finalGrid : Grid),
    (∀ r c, initialGrid r c → finalGrid r c) ∧
    isSymmetric finalGrid ∧
    formsX finalGrid ∧
    (∀ (g : Grid), 
      (∀ r c, initialGrid r c → g r c) → 
      isSymmetric g → 
      formsX g → 
      (∃ (newSquares : List Position),
        newSquares.length = 4 ∧
        (∀ p ∈ newSquares, g p.row p.col ∧ ¬initialGrid p.row p.col))) :=
sorry

end NUMINAMATH_CALUDE_min_squares_for_symmetric_x_l1357_135798


namespace NUMINAMATH_CALUDE_acute_triangle_special_angles_l1357_135720

theorem acute_triangle_special_angles :
  ∃ (α β γ : ℕ),
    α + β + γ = 180 ∧
    0 < γ ∧ γ < β ∧ β < α ∧ α < 90 ∧
    α = 5 * γ ∧
    (α = 85 ∧ β = 78 ∧ γ = 17) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_special_angles_l1357_135720


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1357_135758

theorem trigonometric_identities (α : Real) (h : Real.tan (π / 4 + α) = 3) :
  (Real.tan α = 1 / 2) ∧ 
  (Real.tan (2 * α) = 4 / 3) ∧ 
  ((2 * Real.sin α * Real.cos α + 3 * Real.cos (2 * α)) / 
   (5 * Real.cos (2 * α) - 3 * Real.sin (2 * α)) = 13 / 3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1357_135758


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l1357_135795

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ 1 / Real.rpow 2 (1/3) :=
sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) = 1 / Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l1357_135795


namespace NUMINAMATH_CALUDE_vertical_shift_of_linear_function_l1357_135708

theorem vertical_shift_of_linear_function (x : ℝ) :
  (-3/4 * x) - (-3/4 * x - 4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_vertical_shift_of_linear_function_l1357_135708


namespace NUMINAMATH_CALUDE_percentage_of_women_in_study_group_l1357_135749

theorem percentage_of_women_in_study_group 
  (percentage_women_lawyers : Real) 
  (probability_selecting_woman_lawyer : Real) :
  let percentage_women := probability_selecting_woman_lawyer / percentage_women_lawyers
  percentage_women_lawyers = 0.4 →
  probability_selecting_woman_lawyer = 0.28 →
  percentage_women = 0.7 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_study_group_l1357_135749


namespace NUMINAMATH_CALUDE_all_pollywogs_gone_pollywogs_present_before_44_l1357_135787

/-- Represents the number of pollywogs in the pond after a given number of days -/
def pollywogs_remaining (days : ℕ) : ℕ :=
  if days ≤ 20 then
    2400 - 60 * days
  else
    2400 - 60 * 20 - 50 * (days - 20)

/-- The theorem states that after 44 days, no pollywogs remain in the pond -/
theorem all_pollywogs_gone : pollywogs_remaining 44 = 0 := by
  sorry

/-- The theorem states that before 44 days, there are still pollywogs in the pond -/
theorem pollywogs_present_before_44 (d : ℕ) (h : d < 44) : pollywogs_remaining d > 0 := by
  sorry

end NUMINAMATH_CALUDE_all_pollywogs_gone_pollywogs_present_before_44_l1357_135787


namespace NUMINAMATH_CALUDE_sum_of_a_values_l1357_135730

/-- The equation for which we need to find the values of 'a' -/
def equation (a x : ℝ) : Prop := 4 * x^2 + a * x + 8 * x + 9 = 0

/-- The condition for the equation to have only one solution -/
def has_one_solution (a : ℝ) : Prop :=
  ∃! x, equation a x

/-- The theorem stating that the sum of 'a' values is -16 -/
theorem sum_of_a_values :
  ∃ a₁ a₂ : ℝ, a₁ ≠ a₂ ∧ has_one_solution a₁ ∧ has_one_solution a₂ ∧ a₁ + a₂ = -16 :=
sorry

end NUMINAMATH_CALUDE_sum_of_a_values_l1357_135730


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l1357_135703

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  a = (2, -1) →
  b = (0, 1) →
  ‖a + 2 • b‖ = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l1357_135703


namespace NUMINAMATH_CALUDE_no_integer_solution_l1357_135728

theorem no_integer_solution : ∀ x y : ℤ, 2 * x^2 - 5 * y^2 ≠ 7 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1357_135728


namespace NUMINAMATH_CALUDE_ellipse_equation_form_l1357_135788

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_axes : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) : (ℝ → ℝ → Prop) :=
  sorry

theorem ellipse_equation_form (e : Ellipse) 
  (h_center : e.center = (0, 0))
  (h_foci : e.foci_on_axes = true)
  (h_eccentricity : e.eccentricity = Real.sqrt 3 / 2)
  (h_point : e.passes_through = (2, 0)) :
  (ellipse_equation e = fun x y => x^2 + 4*y^2 = 4) ∨
  (ellipse_equation e = fun x y => 4*x^2 + y^2 = 16) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_form_l1357_135788


namespace NUMINAMATH_CALUDE_equilateral_not_unique_from_angle_and_median_l1357_135755

/-- Represents a triangle -/
structure Triangle where
  a : ℝ  -- side length a
  b : ℝ  -- side length b
  c : ℝ  -- side length c
  α : ℝ  -- angle opposite to side a
  β : ℝ  -- angle opposite to side b
  γ : ℝ  -- angle opposite to side c

/-- Represents a median of a triangle -/
def Median (t : Triangle) (side : ℕ) : ℝ := sorry

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- Theorem stating that one angle and the median to the opposite side
    do not uniquely determine an equilateral triangle -/
theorem equilateral_not_unique_from_angle_and_median :
  ∃ (t1 t2 : Triangle) (side : ℕ),
    t1.α = t2.α ∧
    Median t1 side = Median t2 side ∧
    IsEquilateral t1 ∧
    IsEquilateral t2 ∧
    t1 ≠ t2 :=
  sorry

end NUMINAMATH_CALUDE_equilateral_not_unique_from_angle_and_median_l1357_135755


namespace NUMINAMATH_CALUDE_coin_problem_l1357_135710

theorem coin_problem (x : ℕ) :
  (x : ℚ) + x / 2 + x / 4 = 105 →
  x = 60 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l1357_135710


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1357_135727

-- Define the polynomial
def p (x : ℝ) : ℝ := -3*(x^8 - 2*x^5 + 4*x^3 - 6) + 5*(x^4 + 3*x^2) - 4*(x^6 - 5)

-- Theorem: The sum of coefficients of p is 45
theorem sum_of_coefficients : p 1 = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1357_135727


namespace NUMINAMATH_CALUDE_parabola_inscribed_triangle_l1357_135770

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
def Parabola (p : ℝ) :=
  {point : Point | point.y^2 = 2 * p * point.x}

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Calculates the area of a quadrilateral -/
def quadrilateralArea (q : Quadrilateral) : ℝ :=
  sorry

/-- Main theorem -/
theorem parabola_inscribed_triangle 
  (p : ℝ) 
  (parabola : Parabola p)
  (ABC : Triangle)
  (AFBC : Quadrilateral)
  (h1 : ABC.B.y = 0) -- B is on x-axis
  (h2 : ABC.C.y = 0) -- C is on x-axis
  (h3 : ABC.A.y^2 = 2 * p * ABC.A.x) -- A is on parabola
  (h4 : (ABC.B.x - ABC.A.x) * (ABC.C.x - ABC.A.x) + (ABC.B.y - ABC.A.y) * (ABC.C.y - ABC.A.y) = 0) -- ABC is right-angled
  (h5 : quadrilateralArea AFBC = 8 * p^2) -- Area of AFBC is 8p^2
  : ∃ (D : Point), triangleArea ⟨ABC.A, ABC.C, D⟩ = 15/2 * p^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_inscribed_triangle_l1357_135770


namespace NUMINAMATH_CALUDE_polygon_diagonals_with_disconnected_vertex_l1357_135782

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals connected to a single vertex in a polygon with n sides -/
def diagonals_per_vertex (n : ℕ) : ℕ := n - 3

theorem polygon_diagonals_with_disconnected_vertex :
  diagonals 17 - diagonals_per_vertex 17 = 105 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_with_disconnected_vertex_l1357_135782


namespace NUMINAMATH_CALUDE_john_overall_loss_l1357_135714

def grinder_cost : ℝ := 15000
def mobile_cost : ℝ := 8000
def bicycle_cost : ℝ := 12000
def laptop_cost : ℝ := 25000

def grinder_loss_percent : ℝ := 0.02
def mobile_profit_percent : ℝ := 0.10
def bicycle_profit_percent : ℝ := 0.15
def laptop_loss_percent : ℝ := 0.08

def total_cost : ℝ := grinder_cost + mobile_cost + bicycle_cost + laptop_cost

def grinder_sale : ℝ := grinder_cost * (1 - grinder_loss_percent)
def mobile_sale : ℝ := mobile_cost * (1 + mobile_profit_percent)
def bicycle_sale : ℝ := bicycle_cost * (1 + bicycle_profit_percent)
def laptop_sale : ℝ := laptop_cost * (1 - laptop_loss_percent)

def total_sale : ℝ := grinder_sale + mobile_sale + bicycle_sale + laptop_sale

theorem john_overall_loss : total_sale - total_cost = -700 := by sorry

end NUMINAMATH_CALUDE_john_overall_loss_l1357_135714


namespace NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_60_l1357_135722

/-- The cost of paint per kilogram, given the coverage and the cost to paint a cube. -/
theorem paint_cost_per_kg (coverage : Real) (cube_side : Real) (total_cost : Real) : Real :=
  let surface_area := 6 * cube_side * cube_side
  let paint_needed := surface_area / coverage
  total_cost / paint_needed

/-- Proof that the cost of paint per kilogram is $60. -/
theorem paint_cost_is_60 :
  paint_cost_per_kg 20 10 1800 = 60 := by
  sorry

end NUMINAMATH_CALUDE_paint_cost_per_kg_paint_cost_is_60_l1357_135722


namespace NUMINAMATH_CALUDE_birds_in_tree_l1357_135707

theorem birds_in_tree (initial_birds : ℕ) (new_birds : ℕ) (total_birds : ℕ) : 
  initial_birds = 14 → new_birds = 21 → total_birds = initial_birds + new_birds → total_birds = 35 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l1357_135707


namespace NUMINAMATH_CALUDE_marathon_distance_theorem_l1357_135740

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- Converts a MarathonDistance to total yards -/
def marathonToYards (d : MarathonDistance) : ℕ :=
  d.miles * 1760 + d.yards

/-- The standard marathon distance -/
def standardMarathon : MarathonDistance :=
  { miles := 26, yards := 395 }

/-- Converts total yards to miles and remaining yards -/
def yardsToMilesAndYards (totalYards : ℕ) : MarathonDistance :=
  { miles := totalYards / 1760,
    yards := totalYards % 1760 }

theorem marathon_distance_theorem :
  let totalYards := 15 * marathonToYards standardMarathon
  let result := yardsToMilesAndYards totalYards
  result.yards = 645 := by sorry

end NUMINAMATH_CALUDE_marathon_distance_theorem_l1357_135740


namespace NUMINAMATH_CALUDE_decimal_to_binary_89_l1357_135732

theorem decimal_to_binary_89 : 
  (89 : ℕ).digits 2 = [1, 0, 0, 1, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_decimal_to_binary_89_l1357_135732


namespace NUMINAMATH_CALUDE_equation_solution_l1357_135764

theorem equation_solution :
  ∀ x y : ℝ, 
    y ≠ 0 →
    (4 * y^2 + 1) * (x^4 + 2 * x^2 + 2) = 8 * |y| * (x^2 + 1) →
    ((x = 0 ∧ y = 1/2) ∨ (x = 0 ∧ y = -1/2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1357_135764


namespace NUMINAMATH_CALUDE_original_average_rent_l1357_135776

theorem original_average_rent (num_friends : ℕ) (original_rent : ℝ) (increase_percent : ℝ) (new_mean : ℝ) : 
  num_friends = 4 →
  original_rent = 1600 →
  increase_percent = 0.2 →
  new_mean = 880 →
  (num_friends * new_mean - original_rent * (1 + increase_percent) + original_rent) / num_friends = 800 :=
by sorry

end NUMINAMATH_CALUDE_original_average_rent_l1357_135776


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_l1357_135704

theorem quadratic_always_real_roots (m : ℝ) (hm : m ≠ 0) :
  let a := m
  let b := 1 - 5 * m
  let c := -5
  (b^2 - 4*a*c) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_l1357_135704


namespace NUMINAMATH_CALUDE_cubic_inequality_l1357_135775

theorem cubic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 ≥ a^2*b + b^2*c + c^2*a ∧
  (a^3 + b^3 + c^3 = a^2*b + b^2*c + c^2*a ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1357_135775


namespace NUMINAMATH_CALUDE_markus_to_son_age_ratio_l1357_135771

/-- Represents the ages of Markus, his son, and his grandson. -/
structure FamilyAges where
  markus : ℕ
  son : ℕ
  grandson : ℕ

/-- The conditions given in the problem. -/
def problemConditions (ages : FamilyAges) : Prop :=
  ages.grandson = 20 ∧
  ages.son = 2 * ages.grandson ∧
  ages.markus + ages.son + ages.grandson = 140

/-- The theorem stating that under the given conditions, 
    the ratio of Markus's age to his son's age is 2:1. -/
theorem markus_to_son_age_ratio 
  (ages : FamilyAges) 
  (h : problemConditions ages) : 
  ages.markus * 1 = ages.son * 2 := by
  sorry

#check markus_to_son_age_ratio

end NUMINAMATH_CALUDE_markus_to_son_age_ratio_l1357_135771


namespace NUMINAMATH_CALUDE_phone_reps_calculation_l1357_135767

/-- The number of hours each phone rep works per day -/
def hours_per_day : ℕ := 8

/-- The hourly wage of each phone rep in dollars -/
def hourly_wage : ℚ := 14

/-- The number of days worked -/
def days_worked : ℕ := 5

/-- The total payment for all new employees after 5 days in dollars -/
def total_payment : ℚ := 28000

/-- The number of new phone reps the company wants to hire -/
def num_phone_reps : ℕ := 50

theorem phone_reps_calculation :
  (hours_per_day * hourly_wage * days_worked : ℚ) * num_phone_reps = total_payment :=
by sorry

end NUMINAMATH_CALUDE_phone_reps_calculation_l1357_135767


namespace NUMINAMATH_CALUDE_line_parameterization_l1357_135734

/-- Given a line y = 2x - 40 parameterized by (x, y) = (g(t), 20t - 14), 
    prove that g(t) = 10t + 13 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ x y, y = 2*x - 40 ↔ ∃ t, x = g t ∧ y = 20*t - 14) →
  ∀ t, g t = 10*t + 13 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l1357_135734


namespace NUMINAMATH_CALUDE_snakes_in_pond_l1357_135738

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := 56

/-- The number of eyes each alligator has -/
def eyes_per_alligator : ℕ := 2

/-- The number of eyes each snake has -/
def eyes_per_snake : ℕ := 2

/-- The number of snakes in the pond -/
def num_snakes : ℕ := (total_eyes - num_alligators * eyes_per_alligator) / eyes_per_snake

theorem snakes_in_pond : num_snakes = 18 := by
  sorry

end NUMINAMATH_CALUDE_snakes_in_pond_l1357_135738


namespace NUMINAMATH_CALUDE_art_collection_cost_l1357_135750

theorem art_collection_cost (price_first_three : ℝ) (price_fourth : ℝ) : 
  price_first_three = 45000 →
  price_fourth = (price_first_three / 3) * 1.5 →
  price_first_three + price_fourth = 67500 := by
sorry

end NUMINAMATH_CALUDE_art_collection_cost_l1357_135750


namespace NUMINAMATH_CALUDE_cube_root_of_two_solves_equation_l1357_135761

theorem cube_root_of_two_solves_equation :
  ∃ x : ℝ, x^3 = 2 ∧ x = Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_cube_root_of_two_solves_equation_l1357_135761


namespace NUMINAMATH_CALUDE_coefficient_x3y3_in_x_plus_y_to_6_l1357_135752

theorem coefficient_x3y3_in_x_plus_y_to_6 :
  (Finset.range 7).sum (fun k => (Nat.choose 6 k : ℕ) * 
    (if k = 3 then 1 else 0)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y3_in_x_plus_y_to_6_l1357_135752


namespace NUMINAMATH_CALUDE_bread_calories_eq_100_l1357_135781

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

end NUMINAMATH_CALUDE_bread_calories_eq_100_l1357_135781


namespace NUMINAMATH_CALUDE_circle_symmetry_l1357_135792

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the symmetry line
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetry transformation
def symmetry (x y x' y' : ℝ) : Prop :=
  symmetry_line ((x + x') / 2) ((y + y') / 2) ∧ 
  (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2

-- State the theorem
theorem circle_symmetry :
  ∀ (x y : ℝ),
    (∃ (x' y' : ℝ), symmetry x y x' y' ∧ given_circle x' y') ↔
    x^2 + (y + 1)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1357_135792


namespace NUMINAMATH_CALUDE_divisor_of_p_l1357_135742

theorem divisor_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 40)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 100 < Nat.gcd s p)
  (h5 : Nat.gcd s p < 150) :
  7 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_p_l1357_135742


namespace NUMINAMATH_CALUDE_two_statements_correct_l1357_135779

-- Define a structure for a line in 2D plane
structure Line where
  slope : Option ℝ
  angle_of_inclination : ℝ

-- Define parallel and perpendicular relations
def parallel (l₁ l₂ : Line) : Prop := sorry

def perpendicular (l₁ l₂ : Line) : Prop := sorry

-- Define the four statements
def statement1 (l₁ l₂ : Line) : Prop :=
  (l₁.slope.isSome ∧ l₂.slope.isSome ∧ l₁.slope = l₂.slope) → parallel l₁ l₂

def statement2 (l₁ l₂ : Line) : Prop :=
  perpendicular l₁ l₂ →
    (l₁.slope.isSome ∧ l₂.slope.isSome ∧
     ∃ (s₁ s₂ : ℝ), l₁.slope = some s₁ ∧ l₂.slope = some s₂ ∧ s₁ * s₂ = -1)

def statement3 (l₁ l₂ : Line) : Prop :=
  l₁.angle_of_inclination = l₂.angle_of_inclination → parallel l₁ l₂

def statement4 : Prop :=
  ∀ (l₁ l₂ : Line), parallel l₁ l₂ → (l₁.slope.isSome ∧ l₂.slope.isSome ∧ l₁.slope = l₂.slope)

theorem two_statements_correct (l₁ l₂ : Line) (h : l₁ ≠ l₂) :
  (statement1 l₁ l₂ ∧ statement3 l₁ l₂ ∧ ¬statement2 l₁ l₂ ∧ ¬statement4) := by
  sorry

end NUMINAMATH_CALUDE_two_statements_correct_l1357_135779
