import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_major_c_l297_29747

theorem stratified_sampling_major_c (total_students : ℕ) (sample_size : ℕ) 
  (major_a_students : ℕ) (major_b_students : ℕ) : 
  total_students = 1200 →
  sample_size = 120 →
  major_a_students = 380 →
  major_b_students = 420 →
  (total_students - major_a_students - major_b_students) * sample_size / total_students = 40 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_major_c_l297_29747


namespace NUMINAMATH_CALUDE_ghee_mixture_proof_l297_29719

/-- Proves that adding 20 kg of pure ghee to a 30 kg mixture of 50% pure ghee and 50% vanaspati
    results in a mixture where vanaspati constitutes 30% of the total. -/
theorem ghee_mixture_proof (original_quantity : ℝ) (pure_ghee_added : ℝ) 
  (h1 : original_quantity = 30)
  (h2 : pure_ghee_added = 20) : 
  let initial_vanaspati := 0.5 * original_quantity
  let total_after_addition := original_quantity + pure_ghee_added
  initial_vanaspati / total_after_addition = 0.3 := by
  sorry

#check ghee_mixture_proof

end NUMINAMATH_CALUDE_ghee_mixture_proof_l297_29719


namespace NUMINAMATH_CALUDE_arithmetic_geometric_means_l297_29725

theorem arithmetic_geometric_means (a b c x y : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧  -- a, b, c are positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- a, b, c are distinct
  2 * b = a + c ∧          -- a, b, c form an arithmetic sequence
  x^2 = a * b ∧            -- x is the geometric mean of a and b
  y^2 = b * c →            -- y is the geometric mean of b and c
  (2 * b^2 = x^2 + y^2) ∧  -- x^2, b^2, y^2 form an arithmetic sequence
  (b^4 ≠ x^2 * y^2)        -- x^2, b^2, y^2 do not form a geometric sequence
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_means_l297_29725


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_ratio_l297_29774

/-- Given a decreasing geometric progression a, b, c with common ratio q,
    if 19a, 124b/13, c/13 form an arithmetic progression, then q = 247. -/
theorem geometric_arithmetic_progression_ratio 
  (a b c : ℝ) (q : ℝ) (h_pos : a > 0) (h_decr : q > 1) :
  b = a * q ∧ c = a * q^2 ∧ 
  2 * (124 * b / 13) = 19 * a + c / 13 →
  q = 247 := by
sorry


end NUMINAMATH_CALUDE_geometric_arithmetic_progression_ratio_l297_29774


namespace NUMINAMATH_CALUDE_inverse_existence_l297_29705

-- Define the set of graph labels
inductive GraphLabel
  | A | B | C | D | E

-- Define a predicate for passing the horizontal line test
def passes_horizontal_line_test (g : GraphLabel) : Prop :=
  match g with
  | GraphLabel.A => False
  | GraphLabel.B => True
  | GraphLabel.C => True
  | GraphLabel.D => True
  | GraphLabel.E => False

-- Define a predicate for having an inverse
def has_inverse (g : GraphLabel) : Prop :=
  passes_horizontal_line_test g

-- Theorem statement
theorem inverse_existence (g : GraphLabel) :
  has_inverse g ↔ (g = GraphLabel.B ∨ g = GraphLabel.C ∨ g = GraphLabel.D) :=
by sorry

end NUMINAMATH_CALUDE_inverse_existence_l297_29705


namespace NUMINAMATH_CALUDE_digit_placement_ways_l297_29793

/-- The number of corner boxes in a 3x3 grid -/
def num_corners : ℕ := 4

/-- The total number of boxes in a 3x3 grid -/
def total_boxes : ℕ := 9

/-- The number of digits to be placed -/
def num_digits : ℕ := 4

/-- The number of ways to place digits 1, 2, 3, and 4 in a 3x3 grid -/
def num_ways : ℕ := num_corners * (total_boxes - 1) * (total_boxes - 2) * (total_boxes - 3)

theorem digit_placement_ways :
  num_ways = 1344 :=
sorry

end NUMINAMATH_CALUDE_digit_placement_ways_l297_29793


namespace NUMINAMATH_CALUDE_distance_after_rest_l297_29789

/-- The length of a football field in meters -/
def football_field_length : ℝ := 168

/-- The distance Nate ran before resting, in meters -/
def distance_before_rest : ℝ := 4 * football_field_length

/-- The total distance Nate ran, in meters -/
def total_distance : ℝ := 1172

/-- Theorem: The distance Nate ran after resting is 500 meters -/
theorem distance_after_rest :
  total_distance - distance_before_rest = 500 := by sorry

end NUMINAMATH_CALUDE_distance_after_rest_l297_29789


namespace NUMINAMATH_CALUDE_money_division_l297_29716

theorem money_division (p q r : ℕ) (total : ℚ) : 
  p + q + r = 22 →  -- ratio sum: 3 + 7 + 12 = 22
  (12 / 22) * total - (7 / 22) * total = 4000 →
  (7 / 22) * total - (3 / 22) * total = 3200 :=
by
  sorry

end NUMINAMATH_CALUDE_money_division_l297_29716


namespace NUMINAMATH_CALUDE_expression_evaluation_l297_29756

theorem expression_evaluation (a b : ℝ) (h1 : a = 3) (h2 : b = 2) :
  (a^2 + b)^2 - (a^2 - b)^2 + 2*a*b = 78 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l297_29756


namespace NUMINAMATH_CALUDE_angle_relationship_l297_29770

theorem angle_relationship (α β : Real) 
  (h1 : 0 < α) 
  (h2 : α < 2 * β) 
  (h3 : 2 * β ≤ π / 2)
  (h4 : 2 * Real.cos (α + β) * Real.cos β = -1 + 2 * Real.sin (α + β) * Real.sin β) : 
  α + 2 * β = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_relationship_l297_29770


namespace NUMINAMATH_CALUDE_odell_kershaw_passing_l297_29791

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ  -- speed in m/min
  radius : ℝ  -- track radius in meters
  direction : ℤ  -- 1 for clockwise, -1 for counterclockwise

/-- Calculates the number of times two runners pass each other on a circular track -/
def passingCount (runner1 runner2 : Runner) (duration : ℝ) : ℕ :=
  sorry

theorem odell_kershaw_passing :
  let odell : Runner := { speed := 260, radius := 55, direction := 1 }
  let kershaw : Runner := { speed := 310, radius := 65, direction := -1 }
  passingCount odell kershaw 35 = 52 :=
sorry

end NUMINAMATH_CALUDE_odell_kershaw_passing_l297_29791


namespace NUMINAMATH_CALUDE_union_of_sets_l297_29758

theorem union_of_sets (a : ℝ) : 
  let A : Set ℝ := {x | x ≥ 0}
  let B : Set ℝ := {x | x ≤ a}
  (A ∪ B = Set.univ) → a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l297_29758


namespace NUMINAMATH_CALUDE_divisors_of_2700_l297_29713

def number_of_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_2700 : number_of_divisors 2700 = 36 := by sorry

end NUMINAMATH_CALUDE_divisors_of_2700_l297_29713


namespace NUMINAMATH_CALUDE_correct_large_slices_per_pepper_l297_29759

/-- The number of bell peppers Tamia uses. -/
def num_peppers : ℕ := 5

/-- The total number of slices and pieces Tamia wants to add to her meal. -/
def total_slices : ℕ := 200

/-- Calculates the total number of slices and pieces based on the number of large slices per pepper. -/
def total_slices_func (x : ℕ) : ℕ :=
  num_peppers * x + num_peppers * (x / 2) * 3

/-- The number of large slices Tamia cuts each bell pepper into. -/
def large_slices_per_pepper : ℕ := 16

/-- Theorem stating that the number of large slices per pepper is correct. -/
theorem correct_large_slices_per_pepper : 
  total_slices_func large_slices_per_pepper = total_slices :=
by sorry

end NUMINAMATH_CALUDE_correct_large_slices_per_pepper_l297_29759


namespace NUMINAMATH_CALUDE_a_alone_finish_time_l297_29703

/-- Represents the time taken by A alone to finish the job -/
def time_a : ℝ := 16

/-- Represents the time taken by A and B together to finish the job -/
def time_ab : ℝ := 40

/-- Represents the number of days A and B worked together -/
def days_together : ℝ := 10

/-- Represents the number of days A worked alone after B left -/
def days_a_alone : ℝ := 12

/-- Theorem stating that given the conditions, A alone can finish the job in 16 days -/
theorem a_alone_finish_time :
  (1 / time_a + 1 / time_ab) * days_together + (1 / time_a) * days_a_alone = 1 :=
sorry

end NUMINAMATH_CALUDE_a_alone_finish_time_l297_29703


namespace NUMINAMATH_CALUDE_triangle_area_arithmetic_angles_l297_29796

/-- Given a triangle ABC with sides a and c, and angles A, B, C forming an arithmetic sequence,
    prove that its area is 3√3 when a = 4 and c = 3. -/
theorem triangle_area_arithmetic_angles (A B C : ℝ) (a c : ℝ) :
  -- Angles form an arithmetic sequence
  ∃ d : ℝ, A = B - d ∧ C = B + d
  -- Sum of angles in a triangle is π (180°)
  → A + B + C = π
  -- Given side lengths
  → a = 4
  → c = 3
  -- Area of the triangle
  → (1/2) * a * c * Real.sin B = 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_arithmetic_angles_l297_29796


namespace NUMINAMATH_CALUDE_min_value_parallel_lines_l297_29776

theorem min_value_parallel_lines (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_parallel : a * (b - 3) - 2 * b = 0) : 
  (∀ x y : ℝ, 2 * a + 3 * b ≥ 25) ∧ (∃ x y : ℝ, 2 * a + 3 * b = 25) := by
  sorry

end NUMINAMATH_CALUDE_min_value_parallel_lines_l297_29776


namespace NUMINAMATH_CALUDE_equal_cell_squares_l297_29702

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents the 5x5 grid -/
def Grid := Fin 5 → Fin 5 → Cell

/-- Checks if a square in the grid has an equal number of black and white cells -/
def has_equal_cells (g : Grid) (top_left : Fin 5 × Fin 5) (size : Nat) : Bool :=
  sorry

/-- Counts the number of squares with equal black and white cells -/
def count_equal_squares (g : Grid) : Nat :=
  sorry

/-- The main theorem stating that there are exactly 16 squares with equal black and white cells -/
theorem equal_cell_squares (g : Grid) : count_equal_squares g = 16 := by
  sorry

end NUMINAMATH_CALUDE_equal_cell_squares_l297_29702


namespace NUMINAMATH_CALUDE_towel_shrinkage_l297_29764

theorem towel_shrinkage (original_length original_breadth : ℝ) 
  (original_length_pos : 0 < original_length) 
  (original_breadth_pos : 0 < original_breadth) : 
  let new_length := 0.7 * original_length
  let new_area := 0.595 * (original_length * original_breadth)
  ∃ new_breadth : ℝ, 
    new_breadth = 0.85 * original_breadth ∧ 
    new_area = new_length * new_breadth :=
by sorry

end NUMINAMATH_CALUDE_towel_shrinkage_l297_29764


namespace NUMINAMATH_CALUDE_line_parameterization_l297_29745

/-- Given a line y = 2x - 40 parameterized by (x,y) = (g(t), 10t - 12), 
    prove that g(t) = 5t + 14 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ x y t : ℝ, y = 2*x - 40 ∧ x = g t ∧ y = 10*t - 12) → 
  (∀ t : ℝ, g t = 5*t + 14) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l297_29745


namespace NUMINAMATH_CALUDE_problem_solution_l297_29762

theorem problem_solution (a b m n : ℝ) : 
  (a + b - 1)^2 = -|a + 2| → mn = 1 → a^b + mn = -7 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l297_29762


namespace NUMINAMATH_CALUDE_ship_length_observation_l297_29708

/-- The length of a ship observed from shore --/
theorem ship_length_observation (same_direction : ℝ) (opposite_direction : ℝ) :
  same_direction = 200 →
  opposite_direction = 40 →
  (∃ ship_length : ℝ, (ship_length = 100 ∨ ship_length = 200 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ship_length_observation_l297_29708


namespace NUMINAMATH_CALUDE_fuel_purchase_l297_29761

/-- Fuel purchase problem -/
theorem fuel_purchase (total_spent : ℝ) (initial_cost final_cost : ℝ) :
  total_spent = 90 ∧
  initial_cost = 3 ∧
  final_cost = 4 ∧
  ∃ (mid_cost : ℝ), initial_cost < mid_cost ∧ mid_cost < final_cost →
  ∃ (quantity : ℝ),
    quantity > 0 ∧
    total_spent = initial_cost * quantity + ((initial_cost + final_cost) / 2) * quantity + final_cost * quantity ∧
    initial_cost * quantity + final_cost * quantity = 60 :=
by sorry

end NUMINAMATH_CALUDE_fuel_purchase_l297_29761


namespace NUMINAMATH_CALUDE_circle_through_points_equation_l297_29724

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The circle passing through three given points -/
def CircleThroughThreePoints (A B C : Point) : Prop :=
  ∃ (center : Point) (radius : ℝ),
    (center.x - A.x)^2 + (center.y - A.y)^2 = radius^2 ∧
    (center.x - B.x)^2 + (center.y - B.y)^2 = radius^2 ∧
    (center.x - C.x)^2 + (center.y - C.y)^2 = radius^2

/-- The standard equation of a circle -/
def StandardCircleEquation (center : Point) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.x)^2 + (y - center.y)^2 = radius^2

theorem circle_through_points_equation :
  let A : Point := ⟨1, 12⟩
  let B : Point := ⟨7, 10⟩
  let C : Point := ⟨-9, 2⟩
  CircleThroughThreePoints A B C →
  ∃ (x y : ℝ), StandardCircleEquation ⟨1, 2⟩ 10 x y :=
by sorry

end NUMINAMATH_CALUDE_circle_through_points_equation_l297_29724


namespace NUMINAMATH_CALUDE_min_m_plus_n_range_of_a_l297_29711

-- Part I
theorem min_m_plus_n (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x, f x = |x + 1| + (1/2) * |2*x - 1|) →
  (m > 0 ∧ n > 0) →
  (∀ x, f x ≥ 1/m + 1/n) →
  m + n ≥ 8/3 :=
sorry

-- Part II
theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = |x + 1| + a * |2*x - 1|) →
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ |x - 2|) →
  a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_m_plus_n_range_of_a_l297_29711


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l297_29799

theorem x_positive_sufficient_not_necessary_for_x_nonzero :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_nonzero_l297_29799


namespace NUMINAMATH_CALUDE_food_duration_l297_29795

/-- The number of days the food was initially meant to last -/
def initial_days : ℝ := 22

/-- The initial number of men -/
def initial_men : ℝ := 760

/-- The number of men that join after two days -/
def additional_men : ℝ := 134.11764705882354

/-- The number of days the food lasts after the additional men join -/
def remaining_days : ℝ := 17

/-- The total number of men after the additional men join -/
def total_men : ℝ := initial_men + additional_men

theorem food_duration :
  initial_men * (initial_days - 2) = total_men * remaining_days :=
sorry

end NUMINAMATH_CALUDE_food_duration_l297_29795


namespace NUMINAMATH_CALUDE_identity_value_l297_29734

theorem identity_value (a b c : ℝ) (m n : ℤ) :
  (∀ x : ℝ, (x^n + c)^m = (a*x^m + 1)*(b*x^m + 1)) →
  |a + b + c| = 3 :=
by sorry

end NUMINAMATH_CALUDE_identity_value_l297_29734


namespace NUMINAMATH_CALUDE_prob_multiple_13_eq_l297_29710

/-- Represents a standard deck of 54 cards with 4 suits (1-13) and 2 jokers -/
def Deck : Type := Fin 54

/-- Represents the rank of a card (1-13 for regular cards, 0 for jokers) -/
def rank (card : Deck) : ℕ :=
  if card.val < 52 then
    (card.val % 13) + 1
  else
    0

/-- Shuffles the deck uniformly randomly -/
def shuffle (deck : Deck → α) : Deck → α :=
  sorry

/-- Calculates the score based on the shuffled deck -/
def score (shuffled_deck : Deck → Deck) : ℕ :=
  sorry

/-- Probability that the score is a multiple of 13 -/
def prob_multiple_13 : ℚ :=
  sorry

/-- Main theorem: The probability of the score being a multiple of 13 is 77/689 -/
theorem prob_multiple_13_eq : prob_multiple_13 = 77 / 689 :=
  sorry

end NUMINAMATH_CALUDE_prob_multiple_13_eq_l297_29710


namespace NUMINAMATH_CALUDE_rectangle_inscribed_circle_perpendicular_projections_l297_29701

structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

def UnitCircle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 1}

def projection (M : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ × ℝ :=
  sorry

theorem rectangle_inscribed_circle_perpendicular_projections
  (ABCD : Rectangle)
  (h_inscribed : {ABCD.A, ABCD.B, ABCD.C, ABCD.D} ⊆ UnitCircle)
  (M : ℝ × ℝ)
  (h_M_on_arc : M ∈ UnitCircle ∧ M ≠ ABCD.A ∧ M ≠ ABCD.B)
  (P : ℝ × ℝ) (Q : ℝ × ℝ) (R : ℝ × ℝ) (S : ℝ × ℝ)
  (h_P : P = projection M {x | x.1 = ABCD.A.1})
  (h_Q : Q = projection M {x | x.2 = ABCD.A.2})
  (h_R : R = projection M {x | x.1 = ABCD.C.1})
  (h_S : S = projection M {x | x.2 = ABCD.C.2}) :
  (P.1 - Q.1) * (R.1 - S.1) + (P.2 - Q.2) * (R.2 - S.2) = 0 :=
sorry

end NUMINAMATH_CALUDE_rectangle_inscribed_circle_perpendicular_projections_l297_29701


namespace NUMINAMATH_CALUDE_michael_born_in_1979_l297_29733

/-- The year of the first AMC 8 -/
def first_amc8_year : ℕ := 1985

/-- The number of AMC 8 competitions Michael has taken -/
def michaels_amc8_number : ℕ := 10

/-- Michael's age when he took his AMC 8 -/
def michaels_age : ℕ := 15

/-- Function to calculate the year of a given AMC 8 competition -/
def amc8_year (n : ℕ) : ℕ := first_amc8_year + n - 1

/-- Michael's birth year -/
def michaels_birth_year : ℕ := amc8_year michaels_amc8_number - michaels_age

theorem michael_born_in_1979 : michaels_birth_year = 1979 := by
  sorry

end NUMINAMATH_CALUDE_michael_born_in_1979_l297_29733


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l297_29773

-- Problem 1
theorem problem_1 : (Real.sqrt 48 - (1/4) * Real.sqrt 6) / (-(1/9) * Real.sqrt 27) = -12 + (3/4) * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = (1/2) * (Real.sqrt 3 + 1)) (hy : y = (1/2) * (1 - Real.sqrt 3)) :
  x^2 + y^2 - 2*x*y = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l297_29773


namespace NUMINAMATH_CALUDE_fox_can_equalize_cheese_l297_29738

/-- Represents the state of the cheese pieces -/
structure CheeseState where
  piece1 : ℕ
  piece2 : ℕ
  piece3 : ℕ

/-- Represents a single cut operation by the fox -/
inductive CutOperation
  | cut12 : CutOperation  -- Cut 1g from piece1 and piece2
  | cut13 : CutOperation  -- Cut 1g from piece1 and piece3
  | cut23 : CutOperation  -- Cut 1g from piece2 and piece3

/-- Applies a single cut operation to a cheese state -/
def applyCut (state : CheeseState) (cut : CutOperation) : CheeseState :=
  match cut with
  | CutOperation.cut12 => ⟨state.piece1 - 1, state.piece2 - 1, state.piece3⟩
  | CutOperation.cut13 => ⟨state.piece1 - 1, state.piece2, state.piece3 - 1⟩
  | CutOperation.cut23 => ⟨state.piece1, state.piece2 - 1, state.piece3 - 1⟩

/-- Applies a sequence of cut operations to a cheese state -/
def applyCuts (state : CheeseState) (cuts : List CutOperation) : CheeseState :=
  cuts.foldl applyCut state

/-- The theorem to be proved -/
theorem fox_can_equalize_cheese :
  ∃ (cuts : List CutOperation),
    let finalState := applyCuts ⟨5, 8, 11⟩ cuts
    finalState.piece1 = finalState.piece2 ∧
    finalState.piece2 = finalState.piece3 ∧
    finalState.piece1 > 0 :=
  sorry


end NUMINAMATH_CALUDE_fox_can_equalize_cheese_l297_29738


namespace NUMINAMATH_CALUDE_cheese_distribution_l297_29755

/-- Represents the amount of cheese bought by the first n customers -/
def S (n : ℕ) : ℚ := 20 * n / (n + 10)

/-- The total amount of cheese available -/
def total_cheese : ℚ := 20

theorem cheese_distribution (n : ℕ) (h : n ≤ 10) :
  (total_cheese - S n = 10 * (S n / n)) ∧
  (∀ k : ℕ, k ≤ n → S k - S (k-1) > 0) ∧
  (S 10 = 10) := by sorry

#check cheese_distribution

end NUMINAMATH_CALUDE_cheese_distribution_l297_29755


namespace NUMINAMATH_CALUDE_sum_geq_two_l297_29748

noncomputable def f (x : ℝ) : ℝ := Real.log x + (1/2) * x^2 - 2*x + 3/2

theorem sum_geq_two (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : f x₁ + f x₂ = 0) : 
  x₁ + x₂ ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_two_l297_29748


namespace NUMINAMATH_CALUDE_set_operations_l297_29785

def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 7}
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

theorem set_operations :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 5}) ∧
  ((U \ A) ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l297_29785


namespace NUMINAMATH_CALUDE_stadium_length_in_feet_l297_29717

/-- Proves that the length of a 61-yard stadium is 183 feet. -/
theorem stadium_length_in_feet :
  let stadium_length_yards : ℕ := 61
  let yards_to_feet_conversion : ℕ := 3
  stadium_length_yards * yards_to_feet_conversion = 183 :=
by sorry

end NUMINAMATH_CALUDE_stadium_length_in_feet_l297_29717


namespace NUMINAMATH_CALUDE_mike_limes_l297_29766

-- Define the given conditions
def total_limes : ℕ := 57
def alyssa_limes : ℕ := 25

-- State the theorem
theorem mike_limes : total_limes - alyssa_limes = 32 := by
  sorry

end NUMINAMATH_CALUDE_mike_limes_l297_29766


namespace NUMINAMATH_CALUDE_pencils_bought_is_three_l297_29746

/-- Calculates the number of pencils bought given the total paid, cost per pencil, cost of glue, and change received. -/
def number_of_pencils (total_paid change cost_per_pencil cost_of_glue : ℕ) : ℕ :=
  ((total_paid - change - cost_of_glue) / cost_per_pencil)

/-- Proves that the number of pencils bought is 3 under the given conditions. -/
theorem pencils_bought_is_three :
  number_of_pencils 1000 100 210 270 = 3 := by
  sorry

#eval number_of_pencils 1000 100 210 270

end NUMINAMATH_CALUDE_pencils_bought_is_three_l297_29746


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l297_29772

theorem partial_fraction_decomposition :
  ∀ x : ℚ, x ≠ 10 → x ≠ -5 →
  (8 * x - 3) / (x^2 - 5*x - 50) = (77/15) / (x - 10) + (43/15) / (x + 5) :=
by
  sorry

#check partial_fraction_decomposition

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l297_29772


namespace NUMINAMATH_CALUDE_baker_cakes_theorem_l297_29715

/-- The number of cakes Baker made initially -/
def total_cakes : ℕ := 48

/-- The number of cakes Baker sold -/
def sold_cakes : ℕ := 44

/-- The number of cakes Baker has left -/
def remaining_cakes : ℕ := 4

/-- Theorem stating that the total number of cakes is equal to the sum of sold and remaining cakes -/
theorem baker_cakes_theorem : total_cakes = sold_cakes + remaining_cakes := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_theorem_l297_29715


namespace NUMINAMATH_CALUDE_min_value_problem_l297_29794

theorem min_value_problem (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 1) (hab : a + b = 1) :
  (((a^2 + 1) / (a * b) - 2) * c + Real.sqrt 2 / (c - 1)) ≥ 4 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l297_29794


namespace NUMINAMATH_CALUDE_coloring_books_total_l297_29779

theorem coloring_books_total (initial : ℕ) (given_away : ℕ) (bought : ℕ) : 
  initial = 45 → given_away = 6 → bought = 20 → 
  initial - given_away + bought = 59 := by
sorry

end NUMINAMATH_CALUDE_coloring_books_total_l297_29779


namespace NUMINAMATH_CALUDE_magnitude_of_sum_l297_29765

/-- Given two vectors a and b in ℝ², prove that under certain conditions, 
    the magnitude of a + 2b is √29. -/
theorem magnitude_of_sum (a b : ℝ × ℝ) : 
  (a.1 = 4 ∧ a.2 = 3) → -- a = (4, 3)
  (a.1 * b.1 + a.2 * b.2 = 0) → -- a ⟂ b (dot product is 0)
  (b.1^2 + b.2^2 = 1) → -- |b| = 1
  ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2 = 29) := by
sorry

end NUMINAMATH_CALUDE_magnitude_of_sum_l297_29765


namespace NUMINAMATH_CALUDE_bank_balance_deduction_l297_29727

theorem bank_balance_deduction (X : ℝ) (current_balance : ℝ) : 
  current_balance = X * 0.9 ∧ current_balance = 90000 → X = 100000 := by
sorry

end NUMINAMATH_CALUDE_bank_balance_deduction_l297_29727


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l297_29752

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l297_29752


namespace NUMINAMATH_CALUDE_specific_modular_home_cost_l297_29750

/-- Calculates the cost of a modular home given specific parameters. -/
def modularHomeCost (totalSize kitchenSize bathroomSize : ℕ) 
  (kitchenCost bathroomCost otherCost : ℕ) (numBathrooms : ℕ) : ℕ :=
  let kitchenArea := kitchenSize
  let bathroomArea := bathroomSize * numBathrooms
  let otherArea := totalSize - (kitchenArea + bathroomArea)
  kitchenCost + (bathroomCost * numBathrooms) + (otherArea * otherCost)

/-- Theorem stating the cost of the specific modular home described in the problem. -/
theorem specific_modular_home_cost :
  modularHomeCost 2000 400 150 20000 12000 100 2 = 174000 := by
  sorry

end NUMINAMATH_CALUDE_specific_modular_home_cost_l297_29750


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l297_29797

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(p ∣ n)

theorem smallest_non_prime_non_square_with_large_factors : 
  (∀ m : ℕ, m < 4087 → 
    is_prime m ∨ 
    is_square m ∨ 
    ¬(has_no_prime_factor_less_than m 60)) ∧ 
  ¬(is_prime 4087) ∧ 
  ¬(is_square 4087) ∧ 
  has_no_prime_factor_less_than 4087 60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_with_large_factors_l297_29797


namespace NUMINAMATH_CALUDE_fraction_relation_l297_29740

theorem fraction_relation (m n p s : ℝ) 
  (h1 : m / n = 18)
  (h2 : p / n = 2)
  (h3 : p / s = 1 / 9) :
  m / s = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l297_29740


namespace NUMINAMATH_CALUDE_store_price_difference_l297_29736

/-- Given the total price and quantity of shirts and sweaters, prove that the difference
    between the average price of a sweater and the average price of a shirt is $2. -/
theorem store_price_difference (shirt_price shirt_quantity sweater_price sweater_quantity : ℕ) 
  (h1 : shirt_price = 360)
  (h2 : shirt_quantity = 20)
  (h3 : sweater_price = 900)
  (h4 : sweater_quantity = 45) :
  (sweater_price / sweater_quantity : ℚ) - (shirt_price / shirt_quantity : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_store_price_difference_l297_29736


namespace NUMINAMATH_CALUDE_inequality_proof_l297_29700

theorem inequality_proof (a : ℝ) (h : a ≠ 2) :
  1 / (a^2 - 4*a + 4) > 2 / (a^3 - 8) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l297_29700


namespace NUMINAMATH_CALUDE_triangle_excircle_radii_relation_l297_29742

/-- For a triangle ABC with side lengths a, b, c and excircle radii r_a, r_b, r_c opposite to vertices A, B, C respectively -/
theorem triangle_excircle_radii_relation 
  (a b c r_a r_b r_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_excircle : r_a = (a + b + c) * (b + c - a) / (4 * (b + c)) ∧
                r_b = (a + b + c) * (c + a - b) / (4 * (c + a)) ∧
                r_c = (a + b + c) * (a + b - c) / (4 * (a + b))) :
  a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b)) = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_excircle_radii_relation_l297_29742


namespace NUMINAMATH_CALUDE_jan_claims_proof_l297_29767

-- Define the number of claims each agent can handle
def missy_claims : ℕ := 41
def john_claims : ℕ := missy_claims - 15
def jan_claims : ℕ := 20

-- Theorem to prove
theorem jan_claims_proof : 
  missy_claims = 41 ∧ 
  john_claims = missy_claims - 15 ∧ 
  john_claims = (13 * jan_claims) / 10 → 
  jan_claims = 20 := by
  sorry


end NUMINAMATH_CALUDE_jan_claims_proof_l297_29767


namespace NUMINAMATH_CALUDE_ball_problem_proof_l297_29729

/-- Represents the arrangement of 8 balls with specific conditions -/
def arrangement_count : ℕ := 576

/-- Represents the number of ways to take out 4 balls ensuring each color is taken -/
def takeout_count : ℕ := 40

/-- Represents the number of ways to divide 8 balls into three groups, each with at least 2 balls -/
def division_count : ℕ := 490

/-- Total number of balls -/
def total_balls : ℕ := 8

/-- Number of black balls -/
def black_balls : ℕ := 4

/-- Number of red balls -/
def red_balls : ℕ := 2

/-- Number of yellow balls -/
def yellow_balls : ℕ := 2

theorem ball_problem_proof :
  (total_balls = black_balls + red_balls + yellow_balls) ∧
  (arrangement_count = 576) ∧
  (takeout_count = 40) ∧
  (division_count = 490) := by
  sorry

end NUMINAMATH_CALUDE_ball_problem_proof_l297_29729


namespace NUMINAMATH_CALUDE_a_33_mod_77_l297_29757

/-- Defines a_n as the large integer formed by concatenating integers from 1 to n -/
def a (n : ℕ) : ℕ :=
  -- Definition of a_n goes here
  sorry

/-- The remainder when a_33 is divided by 77 is 22 -/
theorem a_33_mod_77 : a 33 % 77 = 22 := by
  sorry

end NUMINAMATH_CALUDE_a_33_mod_77_l297_29757


namespace NUMINAMATH_CALUDE_correct_multiplication_factor_l297_29749

theorem correct_multiplication_factor (incorrect_factor : ℕ) (difference : ℕ) (number : ℕ) (correct_factor : ℕ) : 
  incorrect_factor = 34 →
  difference = 1206 →
  number = 134 →
  correct_factor = 43 →
  number * correct_factor - number * incorrect_factor = difference :=
by sorry

end NUMINAMATH_CALUDE_correct_multiplication_factor_l297_29749


namespace NUMINAMATH_CALUDE_fibonacci_5k_divisible_by_5_l297_29744

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_5k_divisible_by_5 (k : ℕ) : ∃ m : ℕ, fibonacci (5 * k) = 5 * m := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_5k_divisible_by_5_l297_29744


namespace NUMINAMATH_CALUDE_upstream_travel_time_l297_29778

theorem upstream_travel_time
  (distance : ℝ)
  (downstream_time : ℝ)
  (current_speed : ℝ)
  (h1 : distance = 126)
  (h2 : downstream_time = 7)
  (h3 : current_speed = 2)
  : (distance / (distance / downstream_time - 2 * current_speed)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_upstream_travel_time_l297_29778


namespace NUMINAMATH_CALUDE_system_solution_ratio_l297_29782

theorem system_solution_ratio (x y a b : ℝ) 
  (eq1 : 8 * x - 6 * y = a)
  (eq2 : 9 * y - 12 * x = b)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (b_nonzero : b ≠ 0) :
  a / b = -2 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l297_29782


namespace NUMINAMATH_CALUDE_sum_and_product_calculation_l297_29760

theorem sum_and_product_calculation :
  (199 + 298 + 397 + 496 + 595 + 20 = 2005) ∧
  (39 * 25 = 975) := by
sorry

end NUMINAMATH_CALUDE_sum_and_product_calculation_l297_29760


namespace NUMINAMATH_CALUDE_f_properties_l297_29741

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) - (1 / 2) * Real.cos (2 * x)

theorem f_properties :
  ∃ (T : ℝ), 
    (∀ x, f (x + T) = f x) ∧ 
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
    T = Real.pi ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≤ 1) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = 1) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ≥ -1/2) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 2), f x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l297_29741


namespace NUMINAMATH_CALUDE_octahedron_construction_count_l297_29786

/-- The number of faces in a regular octahedron -/
def octahedron_faces : ℕ := 8

/-- The number of distinct colored triangles available -/
def available_colors : ℕ := 9

/-- The number of rotational symmetries around a fixed face of an octahedron -/
def rotational_symmetries : ℕ := 3

/-- The number of distinguishable ways to construct a regular octahedron -/
def distinguishable_constructions : ℕ := 13440

theorem octahedron_construction_count :
  (Nat.choose available_colors (octahedron_faces - 1)) * 
  (Nat.factorial (octahedron_faces - 1)) / 
  rotational_symmetries = distinguishable_constructions := by
  sorry

end NUMINAMATH_CALUDE_octahedron_construction_count_l297_29786


namespace NUMINAMATH_CALUDE_art_gallery_sculptures_l297_29771

theorem art_gallery_sculptures (total_pieces : ℕ) 
  (h1 : total_pieces = 2700)
  (h2 : ∃ (displayed : ℕ), displayed = total_pieces / 3)
  (h3 : ∃ (displayed_sculptures : ℕ), 
    displayed_sculptures = (total_pieces / 3) / 6)
  (h4 : ∃ (not_displayed_paintings : ℕ), 
    not_displayed_paintings = (total_pieces * 2 / 3) / 3)
  (h5 : ∃ (not_displayed_sculptures : ℕ), 
    not_displayed_sculptures > 0) : 
  ∃ (sculptures_not_displayed : ℕ), 
    sculptures_not_displayed = 1200 := by
sorry

end NUMINAMATH_CALUDE_art_gallery_sculptures_l297_29771


namespace NUMINAMATH_CALUDE_cos_power_five_identity_l297_29739

/-- For all real angles θ, cos^5 θ = (1/64) cos 5θ + (65/64) cos θ -/
theorem cos_power_five_identity (θ : ℝ) : 
  Real.cos θ ^ 5 = (1/64) * Real.cos (5 * θ) + (65/64) * Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_cos_power_five_identity_l297_29739


namespace NUMINAMATH_CALUDE_inequality_solution_set_l297_29723

theorem inequality_solution_set (x : ℝ) : 
  1 / (x^2 + 1) < 5 / x + 21 / 10 ↔ x ∈ Set.Ioi (-1/2) ∪ Set.Ioi 0 \ {-1/2} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l297_29723


namespace NUMINAMATH_CALUDE_erica_pie_fraction_l297_29712

theorem erica_pie_fraction (apple_fraction : ℚ) : 
  (apple_fraction + 3/4 = 95/100) → apple_fraction = 1/5 := by
sorry

end NUMINAMATH_CALUDE_erica_pie_fraction_l297_29712


namespace NUMINAMATH_CALUDE_arc_length_for_36_degree_angle_l297_29753

theorem arc_length_for_36_degree_angle (d : ℝ) (θ : ℝ) (L : ℝ) : 
  d = 4 → θ = 36 * π / 180 → L = d * π * θ / 360 → L = 2 * π / 5 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_36_degree_angle_l297_29753


namespace NUMINAMATH_CALUDE_hotel_room_charge_difference_l297_29787

theorem hotel_room_charge_difference (G R P : ℝ) : 
  P = G * 0.9 →
  R = G * 1.19999999999999986 →
  (R - P) / R * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_hotel_room_charge_difference_l297_29787


namespace NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l297_29784

open Real

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 3 * log x = log (3 * x) := by sorry

end NUMINAMATH_CALUDE_intersection_of_logarithmic_functions_l297_29784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l297_29722

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmeticSequence a) 
  (h_a1 : a 1 = 3) 
  (h_a3 : a 3 = 7) : 
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l297_29722


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l297_29720

/-- The distance from a point P(2-a, -5) to the y-axis is |2-a| -/
theorem distance_to_y_axis (a : ℝ) : 
  let P : ℝ × ℝ := (2 - a, -5)
  abs (P.1) = abs (2 - a) := by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l297_29720


namespace NUMINAMATH_CALUDE_morning_campers_count_l297_29783

/-- Given a total number of campers and a ratio for morning:afternoon:evening,
    calculate the number of campers who went rowing in the morning. -/
def campers_in_morning (total : ℕ) (morning_ratio afternoon_ratio evening_ratio : ℕ) : ℕ :=
  let total_ratio := morning_ratio + afternoon_ratio + evening_ratio
  let part_size := total / total_ratio
  morning_ratio * part_size

/-- Theorem stating that given 60 total campers and a ratio of 3:2:4,
    the number of campers who went rowing in the morning is 18. -/
theorem morning_campers_count :
  campers_in_morning 60 3 2 4 = 18 := by
  sorry

#eval campers_in_morning 60 3 2 4

end NUMINAMATH_CALUDE_morning_campers_count_l297_29783


namespace NUMINAMATH_CALUDE_five_solutions_l297_29714

/-- The system of equations has exactly 5 real solutions -/
theorem five_solutions (x y z w : ℝ) : 
  (x = z + w + z*w*x ∧
   y = w + x + w*x*y ∧
   z = x + y + x*y*z ∧
   w = y + z + y*z*w) →
  ∃! (sol : Finset (ℝ × ℝ × ℝ × ℝ)), 
    sol.card = 5 ∧ 
    ∀ (a b c d : ℝ), (a, b, c, d) ∈ sol ↔ 
      (a = c + d + c*d*a ∧
       b = d + a + d*a*b ∧
       c = a + b + a*b*c ∧
       d = b + c + b*c*d) :=
by sorry

end NUMINAMATH_CALUDE_five_solutions_l297_29714


namespace NUMINAMATH_CALUDE_board_9x16_fills_12x12_square_l297_29731

/-- Represents a rectangular board with integer dimensions -/
structure Board where
  width : ℕ
  length : ℕ

/-- Represents a square hole with integer side length -/
structure Square where
  side : ℕ

/-- Checks if a board can be cut to fill a square hole using the staircase method -/
def canFillSquare (b : Board) (s : Square) : Prop :=
  ∃ (steps : ℕ), 
    steps > 0 ∧
    b.width * (steps + 1) = s.side ∧
    b.length * steps = s.side

theorem board_9x16_fills_12x12_square :
  canFillSquare (Board.mk 9 16) (Square.mk 12) :=
sorry

end NUMINAMATH_CALUDE_board_9x16_fills_12x12_square_l297_29731


namespace NUMINAMATH_CALUDE_subdivided_square_properties_l297_29768

/-- Represents the state of the square after n subdivisions --/
structure SquareState (n : ℕ) where
  remaining_squares : ℕ
  remaining_side_length : ℚ
  removed_area : ℚ

/-- The process of subdividing the square n times --/
def subdivide_square (n : ℕ) : SquareState n :=
  sorry

/-- Theorem stating the properties of the subdivided square --/
theorem subdivided_square_properties (n : ℕ) :
  let state := subdivide_square n
  state.remaining_squares = 8^n ∧
  state.remaining_side_length = 1 / 3^n ∧
  state.removed_area = 1 - (8/9)^n :=
by sorry

end NUMINAMATH_CALUDE_subdivided_square_properties_l297_29768


namespace NUMINAMATH_CALUDE_hannah_fair_money_l297_29737

theorem hannah_fair_money (initial_money : ℝ) : 
  (initial_money / 2 + 5 + 10 = initial_money) → initial_money = 30 := by
  sorry

end NUMINAMATH_CALUDE_hannah_fair_money_l297_29737


namespace NUMINAMATH_CALUDE_first_1500_even_integers_digit_count_l297_29777

/-- Count the number of digits in a positive integer -/
def countDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for all even numbers from 2 to n -/
def sumDigitsEven (n : ℕ) : ℕ := sorry

/-- The 1500th positive even integer -/
def n1500 : ℕ := 3000

theorem first_1500_even_integers_digit_count :
  sumDigitsEven n1500 = 5448 := by sorry

end NUMINAMATH_CALUDE_first_1500_even_integers_digit_count_l297_29777


namespace NUMINAMATH_CALUDE_determinant_scaling_l297_29704

theorem determinant_scaling (x y z w : ℝ) :
  Matrix.det ![![x, y], ![z, w]] = -3 →
  Matrix.det ![![3*x, 3*y], ![5*z, 5*w]] = -45 := by
  sorry

end NUMINAMATH_CALUDE_determinant_scaling_l297_29704


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l297_29763

theorem least_three_digit_multiple_of_11 : ∃ n : ℕ, 
  n = 110 ∧ 
  n % 11 = 0 ∧ 
  100 ≤ n ∧ n ≤ 999 ∧ 
  ∀ m : ℕ, (m % 11 = 0 ∧ 100 ≤ m ∧ m ≤ 999) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l297_29763


namespace NUMINAMATH_CALUDE_rug_coverage_theorem_l297_29726

/-- The total floor area covered by three overlapping rugs -/
def totalFloorArea (combinedArea twoLayerArea threeLayerArea : ℝ) : ℝ :=
  combinedArea - (twoLayerArea + 2 * threeLayerArea)

/-- Theorem stating that the total floor area covered by the rugs is 140 square meters -/
theorem rug_coverage_theorem :
  totalFloorArea 200 22 19 = 140 := by
  sorry

end NUMINAMATH_CALUDE_rug_coverage_theorem_l297_29726


namespace NUMINAMATH_CALUDE_oliver_new_socks_l297_29788

theorem oliver_new_socks (initial_socks : ℕ) (thrown_away : ℕ) (final_socks : ℕ)
  (h1 : initial_socks = 11)
  (h2 : thrown_away = 4)
  (h3 : final_socks = 33) :
  final_socks - (initial_socks - thrown_away) = 26 := by
  sorry

end NUMINAMATH_CALUDE_oliver_new_socks_l297_29788


namespace NUMINAMATH_CALUDE_nested_bracket_value_l297_29769

def bracket (a b c : ℚ) : ℚ :=
  if c ≠ 0 then (a + b) / c else 0

theorem nested_bracket_value :
  bracket (bracket 30 45 75) (bracket 4 2 6) (bracket 12 18 30) = 2 :=
by sorry

end NUMINAMATH_CALUDE_nested_bracket_value_l297_29769


namespace NUMINAMATH_CALUDE_rectangle_toothpicks_l297_29798

/-- The number of toothpicks needed to form a rectangle --/
def toothpicks_in_rectangle (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Theorem: A rectangle with length 20 and width 10 requires 430 toothpicks --/
theorem rectangle_toothpicks : toothpicks_in_rectangle 20 10 = 430 := by
  sorry

#eval toothpicks_in_rectangle 20 10

end NUMINAMATH_CALUDE_rectangle_toothpicks_l297_29798


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l297_29754

theorem cyclic_sum_inequality (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (sum_one : x + y + z = 1) :
  x^2 * y + y^2 * z + z^2 * x ≤ 4/27 ∧ 
  (x^2 * y + y^2 * z + z^2 * x = 4/27 ↔ 
    ((x = 2/3 ∧ y = 1/3 ∧ z = 0) ∨ 
     (x = 0 ∧ y = 2/3 ∧ z = 1/3) ∨ 
     (x = 1/3 ∧ y = 0 ∧ z = 2/3))) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l297_29754


namespace NUMINAMATH_CALUDE_movie_tickets_bought_l297_29780

def computer_game_cost : ℕ := 66
def movie_ticket_cost : ℕ := 12
def total_spent : ℕ := 102

theorem movie_tickets_bought : 
  ∃ (x : ℕ), x * movie_ticket_cost + computer_game_cost = total_spent ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_movie_tickets_bought_l297_29780


namespace NUMINAMATH_CALUDE_polynomial_properties_l297_29732

variable (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)

-- Define the polynomial equality
def poly_eq (x : ℝ) : Prop :=
  (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5

-- Theorem statement
theorem polynomial_properties :
  (∀ x, poly_eq a₀ a₁ a₂ a₃ a₄ a₅ x) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) ∧
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l297_29732


namespace NUMINAMATH_CALUDE_max_garden_area_l297_29792

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.width * d.length

/-- Calculates the perimeter of a rectangular garden (excluding the side adjacent to the house) -/
def gardenPerimeter (d : GardenDimensions) : ℝ := d.length + 2 * d.width

/-- Theorem stating the maximum area of the garden under given constraints -/
theorem max_garden_area :
  ∃ (d : GardenDimensions),
    d.length = 2 * d.width ∧
    gardenPerimeter d = 480 ∧
    ∀ (d' : GardenDimensions),
      d'.length = 2 * d'.width →
      gardenPerimeter d' = 480 →
      gardenArea d' ≤ 28800 :=
by
  sorry

end NUMINAMATH_CALUDE_max_garden_area_l297_29792


namespace NUMINAMATH_CALUDE_rationalize_denominator_l297_29751

theorem rationalize_denominator : Real.sqrt (5 / 18) = Real.sqrt 10 / 6 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l297_29751


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l297_29781

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x^3 - x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l297_29781


namespace NUMINAMATH_CALUDE_addition_problem_base_5_l297_29718

def base_5_to_10 (n : ℕ) : ℕ := sorry

def base_10_to_5 (n : ℕ) : ℕ := sorry

theorem addition_problem_base_5 (X Y : ℕ) : 
  base_10_to_5 (3 * 25 + X * 5 + Y) + base_10_to_5 (3 * 5 + 2) = 
  base_10_to_5 (4 * 25 + 2 * 5 + X) →
  X + Y = 6 := by sorry

end NUMINAMATH_CALUDE_addition_problem_base_5_l297_29718


namespace NUMINAMATH_CALUDE_remainder_369963_div_6_l297_29743

theorem remainder_369963_div_6 : 369963 % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_369963_div_6_l297_29743


namespace NUMINAMATH_CALUDE_skating_rink_visitors_l297_29707

/-- The number of people at a skating rink at noon, given the initial number of visitors,
    the number of people who left, and the number of new arrivals. -/
def people_at_noon (initial : ℕ) (left : ℕ) (arrived : ℕ) : ℕ :=
  initial - left + arrived

/-- Theorem stating that the number of people at the skating rink at noon is 280,
    given the specific values from the problem. -/
theorem skating_rink_visitors : people_at_noon 264 134 150 = 280 := by
  sorry

end NUMINAMATH_CALUDE_skating_rink_visitors_l297_29707


namespace NUMINAMATH_CALUDE_danny_watermelons_l297_29706

theorem danny_watermelons (danny_slices_per_melon : ℕ) (sister_slices : ℕ) (total_slices : ℕ)
  (h1 : danny_slices_per_melon = 10)
  (h2 : sister_slices = 15)
  (h3 : total_slices = 45) :
  ∃ danny_melons : ℕ, danny_melons * danny_slices_per_melon + sister_slices = total_slices ∧ danny_melons = 3 := by
  sorry

end NUMINAMATH_CALUDE_danny_watermelons_l297_29706


namespace NUMINAMATH_CALUDE_premium_probability_option2_higher_price_probability_relationship_l297_29709

-- Define the grades of oranges
inductive Grade : Type
| Premium : Grade
| Special : Grade
| Superior : Grade
| FirstGrade : Grade

-- Define the distribution of boxes
def total_boxes : ℕ := 100
def premium_boxes : ℕ := 40
def special_boxes : ℕ := 30
def superior_boxes : ℕ := 10
def first_grade_boxes : ℕ := 20

-- Define the pricing options
def option1_price : ℚ := 27
def premium_price : ℚ := 36
def special_price : ℚ := 30
def superior_price : ℚ := 24
def first_grade_price : ℚ := 18

-- Theorem 1: Probability of selecting a premium grade box
theorem premium_probability : 
  (premium_boxes : ℚ) / total_boxes = 2 / 5 := by sorry

-- Theorem 2: Average price of Option 2 is higher than Option 1
theorem option2_higher_price :
  (premium_price * premium_boxes + special_price * special_boxes + 
   superior_price * superior_boxes + first_grade_price * first_grade_boxes) / 
  total_boxes > option1_price := by sorry

-- Define probabilities for selecting 3 boxes with different grades
def p₁ : ℚ := 1465 / 1617  -- from 100 boxes
def p₂ : ℚ := 53 / 57      -- from 20 boxes in stratified sampling

-- Theorem 3: Relationship between p₁ and p₂
theorem probability_relationship : p₁ < p₂ := by sorry

end NUMINAMATH_CALUDE_premium_probability_option2_higher_price_probability_relationship_l297_29709


namespace NUMINAMATH_CALUDE_percentage_women_non_union_l297_29730

-- Define the total number of employees
variable (E : ℝ)
-- Assume E is positive
variable (hE : E > 0)

-- Define the percentage of unionized employees
def unionized_percent : ℝ := 0.60

-- Define the percentage of men among unionized employees
def men_in_union_percent : ℝ := 0.70

-- Define the percentage of women among non-union employees
def women_non_union_percent : ℝ := 0.65

-- Theorem to prove
theorem percentage_women_non_union :
  women_non_union_percent = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_women_non_union_l297_29730


namespace NUMINAMATH_CALUDE_area_of_triangle_formed_by_tangents_l297_29735

/-- Given two circles with radii R and r, where their common internal tangents
    are perpendicular to each other, the area of the triangle formed by these
    tangents and the common external tangent is equal to R * r. -/
theorem area_of_triangle_formed_by_tangents (R r : ℝ) (R_pos : R > 0) (r_pos : r > 0) :
  ∃ (S : ℝ), S = R * r ∧ S > 0 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_formed_by_tangents_l297_29735


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l297_29775

theorem arithmetic_calculation : (-1) * (-3) + 3^2 / (8 - 5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l297_29775


namespace NUMINAMATH_CALUDE_carolyn_practice_time_l297_29721

/-- Calculates the total practice time in minutes for a month given daily practice times and schedule -/
def monthly_practice_time (piano_time : ℕ) (violin_multiplier : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let violin_time := piano_time * violin_multiplier
  let daily_total := piano_time + violin_time
  let weekly_total := daily_total * days_per_week
  weekly_total * weeks_per_month

/-- Proves that Carolyn's monthly practice time is 1920 minutes -/
theorem carolyn_practice_time :
  monthly_practice_time 20 3 6 4 = 1920 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_practice_time_l297_29721


namespace NUMINAMATH_CALUDE_area_lower_bound_l297_29790

/-- A plane convex polygon with given projections -/
structure ConvexPolygon where
  /-- Projection onto OX axis -/
  proj_ox : ℝ
  /-- Projection onto bisector of 1st and 3rd coordinate angles -/
  proj_bisector13 : ℝ
  /-- Projection onto OY axis -/
  proj_oy : ℝ
  /-- Projection onto bisector of 2nd and 4th coordinate angles -/
  proj_bisector24 : ℝ
  /-- Area of the polygon -/
  area : ℝ
  /-- Convexity property (simplified) -/
  convex : True

/-- Theorem: The area of a convex polygon with given projections is at least 10 -/
theorem area_lower_bound (p : ConvexPolygon)
  (h1 : p.proj_ox = 4)
  (h2 : p.proj_bisector13 = 3 * Real.sqrt 2)
  (h3 : p.proj_oy = 5)
  (h4 : p.proj_bisector24 = 4 * Real.sqrt 2) :
  p.area ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_area_lower_bound_l297_29790


namespace NUMINAMATH_CALUDE_complex_number_location_l297_29728

theorem complex_number_location (z : ℂ) (h : z * Complex.I = 2 + 3 * Complex.I) :
  0 < z.re ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l297_29728
