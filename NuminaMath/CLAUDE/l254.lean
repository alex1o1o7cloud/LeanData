import Mathlib

namespace proposition_evaluation_l254_25451

open Real

-- Define proposition p
def p : Prop := ∃ x₀ : ℝ, x₀ - 2 > log x₀

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- Theorem statement
theorem proposition_evaluation :
  (p ∧ q) ∧ ¬(p ∧ ¬q) ∧ (¬p ∨ q) ∧ (p ∨ ¬q) := by sorry

end proposition_evaluation_l254_25451


namespace min_value_sqrt_expression_l254_25416

theorem min_value_sqrt_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 16) :
  Real.sqrt (x + 36) + Real.sqrt (16 - x) + 2 * Real.sqrt x ≥ 13.46 := by
  sorry

end min_value_sqrt_expression_l254_25416


namespace panda_weight_l254_25495

theorem panda_weight (monkey_weight : ℕ) (panda_weight : ℕ) : 
  monkey_weight = 25 →
  panda_weight = 6 * monkey_weight + 12 →
  panda_weight = 162 := by
sorry

end panda_weight_l254_25495


namespace blister_slowdown_proof_l254_25406

/-- Represents the speed reduction caused by each blister -/
def blister_slowdown : ℝ := 10

theorem blister_slowdown_proof :
  let old_speed : ℝ := 6
  let new_speed : ℝ := 11
  let hike_duration : ℝ := 4
  let blister_interval : ℝ := 2
  let num_blisters : ℝ := hike_duration / blister_interval
  old_speed * hike_duration = 
    new_speed * blister_interval + 
    (new_speed - num_blisters * blister_slowdown) * blister_interval →
  blister_slowdown = 10 := by
sorry

end blister_slowdown_proof_l254_25406


namespace triangle_perimeter_increase_l254_25459

/-- Given three equilateral triangles where each subsequent triangle has sides 200% of the previous,
    prove that the percent increase in perimeter from the first to the third triangle is 300%. -/
theorem triangle_perimeter_increase (side_length : ℝ) (side_length_positive : side_length > 0) :
  let first_perimeter := 3 * side_length
  let third_perimeter := 3 * (4 * side_length)
  let percent_increase := (third_perimeter - first_perimeter) / first_perimeter * 100
  percent_increase = 300 := by
sorry

end triangle_perimeter_increase_l254_25459


namespace scientific_notation_proof_l254_25425

/-- Scientific notation representation of 185000 -/
def scientific_notation : ℝ := 1.85 * (10 : ℝ) ^ 5

/-- The original number -/
def original_number : ℕ := 185000

theorem scientific_notation_proof : 
  (original_number : ℝ) = scientific_notation := by
  sorry

end scientific_notation_proof_l254_25425


namespace painting_cost_after_modification_l254_25473

/-- Represents the dimensions of a rectangular surface -/
structure Dimensions where
  length : Float
  width : Float

/-- Calculates the area of a rectangular surface -/
def area (d : Dimensions) : Float :=
  d.length * d.width

/-- Represents a room with walls, windows, and doors -/
structure Room where
  walls : List Dimensions
  windows : List Dimensions
  doors : List Dimensions

/-- Calculates the total wall area of a room -/
def totalWallArea (r : Room) : Float :=
  r.walls.map area |> List.sum

/-- Calculates the total area of openings (windows and doors) in a room -/
def totalOpeningArea (r : Room) : Float :=
  (r.windows.map area |> List.sum) + (r.doors.map area |> List.sum)

/-- Calculates the net paintable area of a room -/
def netPaintableArea (r : Room) : Float :=
  totalWallArea r - totalOpeningArea r

/-- Calculates the cost to paint a room given the cost per square foot -/
def paintingCost (r : Room) (costPerSqFt : Float) : Float :=
  netPaintableArea r * costPerSqFt

/-- Increases the dimensions of a room by a given factor -/
def increaseRoomSize (r : Room) (factor : Float) : Room :=
  { walls := r.walls.map fun d => { length := d.length * factor, width := d.width * factor },
    windows := r.windows,
    doors := r.doors }

/-- Adds additional windows and doors to a room -/
def addOpenings (r : Room) (additionalWindows : List Dimensions) (additionalDoors : List Dimensions) : Room :=
  { walls := r.walls,
    windows := r.windows ++ additionalWindows,
    doors := r.doors ++ additionalDoors }

theorem painting_cost_after_modification (originalRoom : Room) (costPerSqFt : Float) : 
  let modifiedRoom := addOpenings (increaseRoomSize originalRoom 1.5) 
                        [⟨3, 4⟩, ⟨3, 4⟩] [⟨3, 7⟩]
  paintingCost modifiedRoom costPerSqFt = 1732.50 :=
by
  sorry

#check painting_cost_after_modification

end painting_cost_after_modification_l254_25473


namespace greta_letter_difference_greta_letter_difference_proof_l254_25469

theorem greta_letter_difference : ℕ → ℕ → ℕ → Prop :=
fun greta_letters brother_letters mother_letters =>
  greta_letters > brother_letters ∧
  mother_letters = 2 * (greta_letters + brother_letters) ∧
  greta_letters + brother_letters + mother_letters = 270 ∧
  brother_letters = 40 →
  greta_letters - brother_letters = 10

-- Proof
theorem greta_letter_difference_proof :
  ∃ (greta_letters brother_letters mother_letters : ℕ),
    greta_letter_difference greta_letters brother_letters mother_letters :=
by
  sorry

end greta_letter_difference_greta_letter_difference_proof_l254_25469


namespace towel_price_problem_l254_25499

theorem towel_price_problem (x : ℚ) : 
  (3 * 100 + 5 * 150 + 2 * x) / 10 = 150 → x = 225 := by
  sorry

end towel_price_problem_l254_25499


namespace max_product_l254_25462

def digits : List Nat := [3, 5, 7, 8, 9]

def is_valid_pair (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def three_digit (a b c : Nat) : Nat := 100 * a + 10 * b + c

def two_digit (d e : Nat) : Nat := 10 * d + e

def product (a b c d e : Nat) : Nat := (three_digit a b c) * (two_digit d e)

theorem max_product :
  ∀ a b c d e,
    is_valid_pair a b c d e →
    product a b c d e ≤ product 9 7 5 8 3 :=
by sorry

end max_product_l254_25462


namespace factor_difference_of_squares_l254_25448

theorem factor_difference_of_squares (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := by
  sorry

end factor_difference_of_squares_l254_25448


namespace dealer_purchase_fraction_l254_25488

/-- Represents the pricing details of an article sold by a dealer -/
structure ArticlePricing where
  listPrice : ℝ
  sellingPrice : ℝ
  purchasePrice : ℝ

/-- Conditions for the dealer's pricing strategy -/
def validPricing (a : ArticlePricing) : Prop :=
  a.sellingPrice = 1.5 * a.listPrice ∧ 
  a.sellingPrice = 2 * a.purchasePrice ∧
  a.listPrice > 0

/-- The theorem to be proved -/
theorem dealer_purchase_fraction (a : ArticlePricing) 
  (h : validPricing a) : 
  a.purchasePrice = (3/8 : ℝ) * a.listPrice :=
sorry

end dealer_purchase_fraction_l254_25488


namespace arithmetic_geometric_sequence_product_l254_25429

/-- Given that -9, a, -1 form an arithmetic sequence and 
    -9, m, b, n, -1 form a geometric sequence, prove that ab = 15 -/
theorem arithmetic_geometric_sequence_product (a m b n : ℝ) : 
  ((-9 + (-1)) / 2 = a) →  -- arithmetic sequence condition
  ((-1 / -9) ^ (1/4) = (-1 / -9) ^ (1/4)) →  -- geometric sequence condition
  a * b = 15 := by
  sorry

end arithmetic_geometric_sequence_product_l254_25429


namespace probability_sum_le_four_l254_25439

-- Define the type for a die
def Die := Fin 6

-- Define the sum of two dice
def diceSum (d1 d2 : Die) : Nat := d1.val + d2.val + 2

-- Define the condition for the sum being less than or equal to 4
def sumLEFour (d1 d2 : Die) : Prop := diceSum d1 d2 ≤ 4

-- Define the probability space
def totalOutcomes : Nat := 36

-- Define the number of favorable outcomes
def favorableOutcomes : Nat := 6

-- Theorem statement
theorem probability_sum_le_four :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 6 := by sorry

end probability_sum_le_four_l254_25439


namespace words_with_consonants_l254_25426

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 6

/-- The number of vowels in the alphabet --/
def vowel_count : ℕ := 2

/-- The length of words we're considering --/
def word_length : ℕ := 5

/-- The total number of possible words --/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words containing only vowels --/
def vowel_only_words : ℕ := vowel_count ^ word_length

theorem words_with_consonants :
  total_words - vowel_only_words = 7744 := by sorry

end words_with_consonants_l254_25426


namespace equation_solution_l254_25453

theorem equation_solution (y : ℚ) : (1 / 3 : ℚ) + 1 / y = 7 / 9 → y = 9 / 4 := by
  sorry

end equation_solution_l254_25453


namespace opposite_of_two_l254_25461

theorem opposite_of_two : (- 2 : ℤ) = -2 := by sorry

end opposite_of_two_l254_25461


namespace solution_set_of_inequality_l254_25441

-- Define the function f
def f (x : ℝ) : ℝ := x^(1/4)

-- State the theorem
theorem solution_set_of_inequality :
  {x : ℝ | f x > f (8*x - 16)} = {x : ℝ | 2 ≤ x ∧ x < 16/7} := by sorry

end solution_set_of_inequality_l254_25441


namespace painting_rate_calculation_l254_25464

theorem painting_rate_calculation (room_length room_width room_height : ℝ)
  (door_width door_height : ℝ) (num_doors : ℕ)
  (window1_width window1_height : ℝ) (num_window1 : ℕ)
  (window2_width window2_height : ℝ) (num_window2 : ℕ)
  (total_cost : ℝ) :
  room_length = 10 ∧ room_width = 7 ∧ room_height = 5 ∧
  door_width = 1 ∧ door_height = 3 ∧ num_doors = 2 ∧
  window1_width = 2 ∧ window1_height = 1.5 ∧ num_window1 = 1 ∧
  window2_width = 1 ∧ window2_height = 1.5 ∧ num_window2 = 2 ∧
  total_cost = 474 →
  (total_cost / (2 * (room_length * room_height + room_width * room_height) -
    (num_doors * door_width * door_height +
     num_window1 * window1_width * window1_height +
     num_window2 * window2_width * window2_height))) = 3 := by
  sorry

end painting_rate_calculation_l254_25464


namespace abs_sum_inequality_solution_existence_l254_25493

theorem abs_sum_inequality_solution_existence (a : ℝ) :
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := by
  sorry

end abs_sum_inequality_solution_existence_l254_25493


namespace constant_proof_l254_25421

theorem constant_proof (n : ℤ) (c : ℝ) : 
  (∀ k : ℤ, c * k^2 ≤ 12100 → k ≤ 10) →
  (c * 10^2 ≤ 12100) →
  c = 121 := by
  sorry

end constant_proof_l254_25421


namespace calculation_proof_equation_no_solution_l254_25485

-- Part 1
theorem calculation_proof : (Real.sqrt 12 - 3 * Real.sqrt (1/3)) / Real.sqrt 3 = 1 := by
  sorry

-- Part 2
theorem equation_no_solution :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 → (x - 1) / (x + 1) + 4 / (x^2 - 1) ≠ (x + 1) / (x - 1) := by
  sorry

end calculation_proof_equation_no_solution_l254_25485


namespace perpendicular_vector_scalar_l254_25449

/-- Given vectors a and b in R², and c defined as a linear combination of a and b,
    prove that if a is perpendicular to c, then the scalar k in the linear combination
    has a specific value. -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (k : ℝ) :
  a = (3, 1) →
  b = (1, 0) →
  let c := a + k • b
  (a.1 * c.1 + a.2 * c.2 = 0) →
  k = -10/3 := by sorry

end perpendicular_vector_scalar_l254_25449


namespace proposition_truth_l254_25491

theorem proposition_truth : ∃ (a b : ℝ), (a * b = 0 ∧ a ≠ 0) ∧ (3 ≥ 3) := by
  sorry

end proposition_truth_l254_25491


namespace initial_birds_count_l254_25438

/-- The number of birds initially on the fence -/
def initial_birds : ℕ := sorry

/-- The number of birds that joined the fence -/
def joined_birds : ℕ := 4

/-- The total number of birds on the fence after joining -/
def total_birds : ℕ := 5

/-- Theorem stating that the initial number of birds is 1 -/
theorem initial_birds_count : initial_birds = 1 :=
  by sorry

end initial_birds_count_l254_25438


namespace superman_game_cost_l254_25471

/-- The cost of Tom's video game purchases -/
def total_spent : ℝ := 18.66

/-- The cost of the Batman game -/
def batman_cost : ℝ := 13.6

/-- The number of games Tom already owns -/
def existing_games : ℕ := 2

/-- The cost of the Superman game -/
def superman_cost : ℝ := total_spent - batman_cost

theorem superman_game_cost : superman_cost = 5.06 := by
  sorry

end superman_game_cost_l254_25471


namespace average_monthly_production_theorem_l254_25444

def initial_production : ℝ := 1000

def monthly_increases : List ℝ := [0.05, 0.07, 0.10, 0.04, 0.08, 0.05, 0.07, 0.06, 0.12, 0.10, 0.08]

def calculate_monthly_production (prev : ℝ) (increase : ℝ) : ℝ :=
  prev * (1 + increase)

def calculate_yearly_production (initial : ℝ) (increases : List ℝ) : ℝ :=
  initial + (increases.scanl calculate_monthly_production initial).sum

theorem average_monthly_production_theorem :
  let yearly_production := calculate_yearly_production initial_production monthly_increases
  let average_production := yearly_production / 12
  ∃ ε > 0, |average_production - 1445.084204| < ε :=
sorry

end average_monthly_production_theorem_l254_25444


namespace not_sum_of_two_rational_squares_168_l254_25460

theorem not_sum_of_two_rational_squares_168 : ¬ ∃ (a b : ℚ), a^2 + b^2 = 168 := by
  sorry

end not_sum_of_two_rational_squares_168_l254_25460


namespace complement_of_A_in_U_l254_25430

-- Define the universal set U
def U : Set ℝ := {x : ℝ | -Real.sqrt 3 < x}

-- Define set A
def A : Set ℝ := {x : ℝ | 1 < 4 - x^2 ∧ 4 - x^2 ≤ 2}

-- State the theorem
theorem complement_of_A_in_U :
  (U \ A) = {x : ℝ | (-Real.sqrt 3 < x ∧ x < -Real.sqrt 2) ∨ (Real.sqrt 2 < x)} :=
by sorry

end complement_of_A_in_U_l254_25430


namespace randy_tower_blocks_l254_25498

/-- Given information about Randy's blocks and constructions -/
structure RandysBlocks where
  total : ℕ
  house : ℕ
  tower_and_house : ℕ

/-- The number of blocks Randy used for the tower -/
def blocks_for_tower (r : RandysBlocks) : ℕ :=
  r.tower_and_house - r.house

/-- Theorem stating that Randy used 27 blocks for the tower -/
theorem randy_tower_blocks (r : RandysBlocks)
  (h1 : r.total = 58)
  (h2 : r.house = 53)
  (h3 : r.tower_and_house = 80) :
  blocks_for_tower r = 27 := by
  sorry

end randy_tower_blocks_l254_25498


namespace parabola_intersection_theorem_l254_25447

/-- A parabola with equation y^2 = 6x -/
structure Parabola where
  equation : ∀ x y, y^2 = 6*x

/-- A point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 6*x

/-- Two lines intersecting the parabola -/
structure IntersectingLines (C : Parabola) (P : PointOnParabola C) where
  A : PointOnParabola C
  B : PointOnParabola C
  slope_AB : (B.y - A.y) / (B.x - A.x) = 2
  sum_reciprocal_slopes : 
    ((P.y - A.y) / (P.x - A.x))⁻¹ + ((P.y - B.y) / (P.x - B.x))⁻¹ = 3

/-- The theorem to be proved -/
theorem parabola_intersection_theorem 
  (C : Parabola) 
  (P : PointOnParabola C) 
  (L : IntersectingLines C P) : 
  P.y = 15/2 := by sorry

end parabola_intersection_theorem_l254_25447


namespace quadratic_inequality_solution_l254_25418

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 + 9 * x + 6 < 0) ↔ (-2 < x ∧ x < -1) := by
  sorry

end quadratic_inequality_solution_l254_25418


namespace license_plate_count_l254_25414

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 5

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of possible positions for the letter block -/
def letter_block_positions : ℕ := digits_count + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  letter_block_positions * (num_digits ^ digits_count) * (num_letters ^ letters_count)

theorem license_plate_count : total_license_plates = 105456000 := by
  sorry

end license_plate_count_l254_25414


namespace tommy_nickels_count_l254_25412

/-- Proves that Tommy has 100 nickels given the relationships between his coins -/
theorem tommy_nickels_count : 
  ∀ (quarters pennies dimes nickels : ℕ),
    quarters = 4 →
    pennies = 10 * quarters →
    dimes = pennies + 10 →
    nickels = 2 * dimes →
    nickels = 100 := by sorry

end tommy_nickels_count_l254_25412


namespace cupcake_problem_l254_25452

theorem cupcake_problem (total_girls : ℕ) (avg_cupcakes : ℚ) (max_cupcakes : ℕ) (no_cupcake_girls : ℕ) :
  total_girls = 12 →
  avg_cupcakes = 3/2 →
  max_cupcakes = 2 →
  no_cupcake_girls = 2 →
  ∃ (two_cupcake_girls : ℕ),
    two_cupcake_girls = 8 ∧
    two_cupcake_girls + no_cupcake_girls + (total_girls - two_cupcake_girls - no_cupcake_girls) = total_girls ∧
    2 * two_cupcake_girls + (total_girls - two_cupcake_girls - no_cupcake_girls) = (avg_cupcakes * total_girls).num :=
by
  sorry

end cupcake_problem_l254_25452


namespace cube_base_diagonal_l254_25468

/-- Given a cube with space diagonal length of 5 units, 
    the diagonal of its base has length 5 * sqrt(2/3) units. -/
theorem cube_base_diagonal (c : Real) (h : c > 0) 
  (space_diagonal : c * Real.sqrt 3 = 5) : 
  c * Real.sqrt 2 = 5 * Real.sqrt (2/3) := by
  sorry

end cube_base_diagonal_l254_25468


namespace a_101_mod_49_l254_25476

/-- Definition of the sequence a_n -/
def a (n : ℕ) : ℕ := 5^n + 9^n

/-- Theorem stating that a_101 is congruent to 0 modulo 49 -/
theorem a_101_mod_49 : a 101 ≡ 0 [ZMOD 49] := by
  sorry

end a_101_mod_49_l254_25476


namespace cubic_root_l254_25456

/-- Given a cubic expression ax³ - 2x + c, prove that if it equals -5 when x = 1
and equals 52 when x = 4, then it equals 0 when x = 2. -/
theorem cubic_root (a c : ℝ) 
  (h1 : a * 1^3 - 2 * 1 + c = -5)
  (h2 : a * 4^3 - 2 * 4 + c = 52) :
  a * 2^3 - 2 * 2 + c = 0 := by
sorry

end cubic_root_l254_25456


namespace correct_matching_probability_l254_25409

-- Define the number of celebrities and child photos
def num_celebrities : ℕ := 4

-- Define the function to calculate the number of possible arrangements
def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Define the probability of correct matching
def probability_correct_matching : ℚ := 1 / num_arrangements num_celebrities

-- Theorem statement
theorem correct_matching_probability :
  probability_correct_matching = 1 / 24 := by
  sorry

end correct_matching_probability_l254_25409


namespace min_distance_circle_ellipse_l254_25463

/-- The minimum distance between a point on a unit circle centered at the origin
    and a point on an ellipse centered at (-1, 0) with semi-major axis 3 and semi-minor axis 5 -/
theorem min_distance_circle_ellipse :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let ellipse := {(x, y) : ℝ × ℝ | ((x + 1)^2 / 9) + (y^2 / 25) = 1}
  ∃ d : ℝ, d = Real.sqrt 14 - 1 ∧
    ∀ (a : ℝ × ℝ) (b : ℝ × ℝ), a ∈ circle → b ∈ ellipse →
      Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≥ d :=
by sorry

end min_distance_circle_ellipse_l254_25463


namespace square_difference_equality_l254_25407

theorem square_difference_equality : (36 + 12)^2 - (12^2 + 36^2 + 24) = 840 := by
  sorry

end square_difference_equality_l254_25407


namespace quadratic_roots_theorem_l254_25446

def quadratic_equation (m n x : ℝ) : ℝ := 9 * x^2 - 2 * m * x + n

def has_two_real_roots (m n : ℤ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m n x = 0 ∧ quadratic_equation m n y = 0

def roots_in_interval (m n : ℤ) : Prop :=
  ∀ x : ℝ, quadratic_equation m n x = 0 → 0 < x ∧ x < 1

theorem quadratic_roots_theorem :
  ∀ m n : ℤ, has_two_real_roots m n ∧ roots_in_interval m n ↔ (m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2) :=
sorry

end quadratic_roots_theorem_l254_25446


namespace solution_set_of_inequality_l254_25490

theorem solution_set_of_inequality (x : ℝ) :
  (1/2 - x) * (x - 1/3) > 0 ↔ 1/3 < x ∧ x < 1/2 :=
by sorry

end solution_set_of_inequality_l254_25490


namespace corrected_mean_calculation_l254_25433

/-- Given a set of observations with incorrect recordings, calculate the corrected mean. -/
theorem corrected_mean_calculation 
  (n : ℕ) 
  (original_mean : ℚ) 
  (incorrect_value1 incorrect_value2 correct_value1 correct_value2 : ℚ) :
  n = 50 →
  original_mean = 36 →
  incorrect_value1 = 23 →
  incorrect_value2 = 55 →
  correct_value1 = 34 →
  correct_value2 = 45 →
  let original_sum := n * original_mean
  let adjusted_sum := original_sum - incorrect_value1 - incorrect_value2 + correct_value1 + correct_value2
  let new_mean := adjusted_sum / n
  new_mean = 36.02 := by
  sorry

end corrected_mean_calculation_l254_25433


namespace triangle_segment_sum_squares_l254_25440

-- Define the triangle ABC and points D and E
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = b^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = c^2

def RightTriangle (A B C : ℝ × ℝ) : Prop :=
  Triangle A B C ∧ 
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def DivideHypotenuse (A B C D E : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t > 0 ∧
  D = ((2*B.1 + C.1)/3, (2*B.2 + C.2)/3) ∧
  E = ((B.1 + 2*C.1)/3, (B.2 + 2*C.2)/3)

-- State the theorem
theorem triangle_segment_sum_squares 
  (A B C D E : ℝ × ℝ) 
  (h1 : RightTriangle A B C) 
  (h2 : DivideHypotenuse A B C D E) : 
  ((D.1 - A.1)^2 + (D.2 - A.2)^2) + 
  ((E.1 - D.1)^2 + (E.2 - D.2)^2) + 
  ((E.1 - A.1)^2 + (E.2 - A.2)^2) = 
  2/3 * ((C.1 - B.1)^2 + (C.2 - B.2)^2) := by
  sorry

end triangle_segment_sum_squares_l254_25440


namespace divides_trans_divides_mul_l254_25420

/-- Divisibility relation for positive integers -/
def divides (a b : ℕ+) : Prop := ∃ k : ℕ+, b = a * k

/-- Transitivity of divisibility -/
theorem divides_trans {a b c : ℕ+} (h1 : divides a b) (h2 : divides b c) : 
  divides a c := by sorry

/-- Product of divisibilities -/
theorem divides_mul {a b c d : ℕ+} (h1 : divides a b) (h2 : divides c d) :
  divides (a * c) (b * d) := by sorry

end divides_trans_divides_mul_l254_25420


namespace group_size_is_ten_l254_25482

/-- The number of people in a group that can hold a certain number of boxes. -/
def group_size (total_boxes : ℕ) (boxes_per_person : ℕ) : ℕ :=
  total_boxes / boxes_per_person

/-- Theorem: The group size is 10 when the total boxes is 20 and each person can hold 2 boxes. -/
theorem group_size_is_ten : group_size 20 2 = 10 := by
  sorry

end group_size_is_ten_l254_25482


namespace no_complete_turn_l254_25466

/-- Represents the position of a bead on a ring as an angle in radians -/
def BeadPosition := ℝ

/-- Represents the state of all beads on the ring -/
def RingState := List BeadPosition

/-- A move that places a bead between its two neighbors -/
def move (state : RingState) (index : Nat) : RingState :=
  sorry

/-- Predicate to check if a bead has made a complete turn -/
def hasMadeCompleteTurn (initialState finalState : RingState) (beadIndex : Nat) : Prop :=
  sorry

/-- The main theorem stating that no bead can make a complete turn -/
theorem no_complete_turn (initialState : RingState) :
    initialState.length = 2009 →
    ∀ (moves : List Nat) (beadIndex : Nat),
      let finalState := moves.foldl move initialState
      ¬ hasMadeCompleteTurn initialState finalState beadIndex :=
  sorry

end no_complete_turn_l254_25466


namespace both_make_basket_l254_25408

-- Define the probabilities
def prob_A : ℚ := 2/5
def prob_B : ℚ := 1/2

-- Define the theorem
theorem both_make_basket : 
  prob_A * prob_B = 1/5 := by sorry

end both_make_basket_l254_25408


namespace units_digit_of_7_power_75_plus_6_l254_25432

theorem units_digit_of_7_power_75_plus_6 : 
  (7^75 + 6) % 10 = 9 :=
by sorry

end units_digit_of_7_power_75_plus_6_l254_25432


namespace johns_annual_profit_l254_25431

/-- Calculates the annual profit for John's subletting arrangement -/
def annual_profit (rent_a rent_b rent_c apartment_rent utilities maintenance : ℕ) : ℕ := 
  let total_income := rent_a + rent_b + rent_c
  let total_expenses := apartment_rent + utilities + maintenance
  let monthly_profit := total_income - total_expenses
  12 * monthly_profit

/-- Theorem stating John's annual profit given the specified conditions -/
theorem johns_annual_profit : 
  annual_profit 350 400 450 900 100 50 = 1800 := by
  sorry

end johns_annual_profit_l254_25431


namespace no_three_integer_solutions_l254_25404

theorem no_three_integer_solutions (b : ℤ) : 
  ¬(∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    (x^2 + b*x + 5 ≤ 0) ∧ (y^2 + b*y + 5 ≤ 0) ∧ (z^2 + b*z + 5 ≤ 0) ∧
    (∀ (w : ℤ), w ≠ x ∧ w ≠ y ∧ w ≠ z → w^2 + b*w + 5 > 0)) :=
by sorry

#check no_three_integer_solutions

end no_three_integer_solutions_l254_25404


namespace right_triangle_point_condition_l254_25470

theorem right_triangle_point_condition (a b c x : ℝ) :
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  0 ≤ x → x ≤ b →
  let s := x^2 + (b - x)^2 + (a * x / b)^2
  s = 2 * (b - x)^2 ↔ x = b^2 / Real.sqrt (a^2 + 2 * b^2) :=
by sorry

end right_triangle_point_condition_l254_25470


namespace max_perimeter_rectangle_from_triangles_l254_25467

theorem max_perimeter_rectangle_from_triangles :
  let num_triangles : ℕ := 60
  let leg1 : ℝ := 2
  let leg2 : ℝ := 3
  let triangle_area : ℝ := (1 / 2) * leg1 * leg2
  let total_area : ℝ := num_triangles * triangle_area
  ∀ a b : ℝ,
    a > 0 → b > 0 →
    a * b = total_area →
    2 * (a + b) ≤ 184 :=
by sorry

end max_perimeter_rectangle_from_triangles_l254_25467


namespace females_with_advanced_degrees_l254_25489

theorem females_with_advanced_degrees 
  (total_employees : ℕ)
  (total_females : ℕ)
  (employees_with_advanced_degrees : ℕ)
  (males_with_college_only : ℕ)
  (h1 : total_employees = 148)
  (h2 : total_females = 92)
  (h3 : employees_with_advanced_degrees = 78)
  (h4 : males_with_college_only = 31)
  : total_females - (total_employees - employees_with_advanced_degrees - males_with_college_only) = 53 := by
  sorry

end females_with_advanced_degrees_l254_25489


namespace equation_solution_l254_25483

theorem equation_solution : ∀ x : ℝ, (3 / (x - 3) = 3 / (x^2 - 9)) ↔ x = -2 := by sorry

end equation_solution_l254_25483


namespace square_plus_one_positive_l254_25403

theorem square_plus_one_positive (a : ℝ) : a^2 + 1 > 0 := by
  sorry

end square_plus_one_positive_l254_25403


namespace symmetric_line_equation_l254_25487

/-- Given a line symmetric to 4x - 3y + 5 = 0 with respect to the y-axis, prove its equation is 4x + 3y - 5 = 0 -/
theorem symmetric_line_equation : 
  ∃ (l : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ (-x, y) ∈ {(x, y) | 4*x - 3*y + 5 = 0}) → 
    (∀ (x y : ℝ), (x, y) ∈ l ↔ 4*x + 3*y - 5 = 0) :=
by sorry

end symmetric_line_equation_l254_25487


namespace arithmetic_computation_l254_25422

theorem arithmetic_computation : 5 + 4 * (4 - 9)^2 = 105 := by
  sorry

end arithmetic_computation_l254_25422


namespace number_count_from_average_correction_l254_25496

/-- Given an initial average and a corrected average after fixing a misread number,
    calculate the number of numbers in the original set. -/
theorem number_count_from_average_correction (initial_avg : ℚ) (corrected_avg : ℚ) 
    (misread : ℚ) (correct : ℚ) (h1 : initial_avg = 16) (h2 : corrected_avg = 19) 
    (h3 : misread = 25) (h4 : correct = 55) : 
    ∃ n : ℕ, (n : ℚ) * initial_avg + misread = (n : ℚ) * corrected_avg + correct ∧ n = 10 := by
  sorry

end number_count_from_average_correction_l254_25496


namespace intersection_points_properties_l254_25417

open Real

theorem intersection_points_properties (k : ℝ) (h_k : k > 0) :
  let f := fun x => Real.exp x
  let g := fun x => Real.exp (-x)
  let n := f k
  let m := g k
  n < 2 * m →
  (n + m < 3 * Real.sqrt 2 / 2) ∧
  (n - m < Real.sqrt 2 / 2) ∧
  (n^(m + 1) < (m + 1)^n) :=
by sorry

end intersection_points_properties_l254_25417


namespace complement_intersection_empty_and_range_l254_25497

-- Define the sets A and B as functions of a
def A (a : ℝ) : Set ℝ :=
  if 3 * a + 1 > 2 then {x : ℝ | 2 < x ∧ x < 3 * a + 1}
  else {x : ℝ | 3 * a + 1 < x ∧ x < 2}

def B (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a^2 + 2}

-- Define propositions p and q
def p (x : ℝ) (a : ℝ) : Prop := x ∈ A a
def q (x : ℝ) (a : ℝ) : Prop := x ∈ B a

theorem complement_intersection_empty_and_range (a : ℝ) :
  (A a ≠ ∅ ∧ B a ≠ ∅) →
  ((a = 1/3 → (Set.univ \ B a) ∩ A a = ∅) ∧
   (∀ x, p x a → q x a) ↔ (1/3 ≤ a ∧ a ≤ (Real.sqrt 5 - 1) / 2)) :=
sorry

end complement_intersection_empty_and_range_l254_25497


namespace dalmatian_spots_l254_25434

theorem dalmatian_spots (bill_spots phil_spots : ℕ) : 
  bill_spots = 39 → 
  bill_spots = 2 * phil_spots - 1 → 
  bill_spots + phil_spots = 59 := by
sorry

end dalmatian_spots_l254_25434


namespace min_dot_product_on_ellipse_l254_25427

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- The fixed point K -/
def K : ℝ × ℝ := (2, 0)

/-- Dot product of two 2D vectors -/
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  (v₁.1 * v₂.1) + (v₁.2 * v₂.2)

/-- Vector from K to a point -/
def vector_from_K (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - K.1, p.2 - K.2)

theorem min_dot_product_on_ellipse :
  ∀ (M N : ℝ × ℝ),
  is_on_ellipse M.1 M.2 →
  is_on_ellipse N.1 N.2 →
  dot_product (vector_from_K M) (vector_from_K N) = 0 →
  ∃ (min_value : ℝ),
    min_value = 23/3 ∧
    ∀ (P Q : ℝ × ℝ),
    is_on_ellipse P.1 P.2 →
    is_on_ellipse Q.1 Q.2 →
    dot_product (vector_from_K P) (vector_from_K Q) = 0 →
    dot_product (vector_from_K P) (vector_from_K Q - vector_from_K P) ≥ min_value :=
by sorry

end min_dot_product_on_ellipse_l254_25427


namespace min_length_shared_side_l254_25436

/-- Given two triangles PQR and SQR that share side QR, with PQ = 7, PR = 15, SR = 10, and QS = 25,
    prove that the length of QR is at least 15. -/
theorem min_length_shared_side (PQ PR SR QS : ℝ) (hPQ : PQ = 7) (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
  ∃ (QR : ℝ), QR ≥ 15 ∧ QR > PR - PQ ∧ QR > QS - SR :=
by sorry

end min_length_shared_side_l254_25436


namespace alice_met_tweedledee_l254_25472

-- Define the type for brothers
inductive Brother
| Tweedledee
| Tweedledum

-- Define the type for truthfulness
inductive Truthfulness
| AlwaysTruth
| AlwaysLie

-- Define the statement made by the brother
structure Statement where
  lying : Prop
  name : Brother

-- Define the meeting scenario
structure Meeting where
  brother : Brother
  truthfulness : Truthfulness
  statement : Statement

-- Theorem to prove
theorem alice_met_tweedledee (m : Meeting) :
  m.statement = { lying := true, name := Brother.Tweedledee } →
  (m.truthfulness = Truthfulness.AlwaysTruth ∨ m.truthfulness = Truthfulness.AlwaysLie) →
  m.brother = Brother.Tweedledee :=
by sorry

end alice_met_tweedledee_l254_25472


namespace feb_first_is_monday_l254_25437

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a date in February of a leap year -/
structure FebruaryDate :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- Given that February 29th is a Monday in a leap year, 
    prove that February 1st is also a Monday -/
theorem feb_first_is_monday 
  (feb29 : FebruaryDate)
  (h1 : feb29.day = 29)
  (h2 : feb29.dayOfWeek = DayOfWeek.Monday) :
  ∃ (feb1 : FebruaryDate), 
    feb1.day = 1 ∧ feb1.dayOfWeek = DayOfWeek.Monday :=
by sorry

end feb_first_is_monday_l254_25437


namespace intersection_M_N_l254_25478

-- Define set M
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define set N
def N : Set ℝ := {x | 2*x - x^2 ≥ 0}

-- Statement to prove
theorem intersection_M_N : M ∩ N = Set.Ioo 1 2 ∪ {2} := by sorry

end intersection_M_N_l254_25478


namespace partition_contains_perfect_square_sum_l254_25458

/-- A partition of a set of natural numbers -/
def Partition (n : ℕ) := Fin n → Bool

/-- Checks if a pair of numbers sum to a perfect square -/
def IsPerfectSquareSum (a b : ℕ) : Prop :=
  ∃ k : ℕ, a + b = k * k

/-- The main theorem statement -/
theorem partition_contains_perfect_square_sum (n : ℕ) (h : n ≥ 15) :
  ∀ (p : Partition n), ∃ (i j : Fin n), i ≠ j ∧ p i = p j ∧ IsPerfectSquareSum (i.val + 1) (j.val + 1) :=
sorry

end partition_contains_perfect_square_sum_l254_25458


namespace system_of_equations_solution_l254_25486

theorem system_of_equations_solution (x y m : ℝ) : 
  2 * x + y = 4 → 
  x + 2 * y = m → 
  x + y = 1 → 
  m = -1 := by
sorry

end system_of_equations_solution_l254_25486


namespace max_sum_of_factors_l254_25442

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 3003 → 
  A + B + C ≤ 45 := by
sorry

end max_sum_of_factors_l254_25442


namespace triangle_perimeter_l254_25477

theorem triangle_perimeter (a b : ℝ) (perimeters : List ℝ) : 
  a = 25 → b = 20 → perimeters = [58, 64, 70, 76, 82] →
  ∃ (p : ℝ), p ∈ perimeters ∧ 
  (∀ (x : ℝ), x > 0 ∧ a + b > x ∧ a + x > b ∧ b + x > a → 
    p ≠ a + b + x) ∧
  (∀ (q : ℝ), q ∈ perimeters ∧ q ≠ p → 
    ∃ (y : ℝ), y > 0 ∧ a + b > y ∧ a + y > b ∧ b + y > a ∧ 
    q = a + b + y) := by
  sorry

end triangle_perimeter_l254_25477


namespace first_year_after_2020_with_digit_sum_15_l254_25484

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2020WithDigitSum15 (year : ℕ) : Prop :=
  year > 2020 ∧ 
  sumOfDigits year = 15 ∧ 
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 15

theorem first_year_after_2020_with_digit_sum_15 :
  isFirstYearAfter2020WithDigitSum15 2058 := by
  sorry

end first_year_after_2020_with_digit_sum_15_l254_25484


namespace inequality_solution_l254_25479

theorem inequality_solution (x : ℝ) :
  (2 ≤ |3*x - 6| ∧ |3*x - 6| ≤ 12) ↔ (x ∈ Set.Icc (-2) (4/3) ∪ Set.Icc (8/3) 6) := by
  sorry

end inequality_solution_l254_25479


namespace max_absolute_value_constrained_complex_l254_25445

theorem max_absolute_value_constrained_complex (z : ℂ) (h : Complex.abs (z - 2 * Complex.I) ≤ 1) :
  Complex.abs z ≤ 3 ∧ ∃ w : ℂ, Complex.abs (w - 2 * Complex.I) ≤ 1 ∧ Complex.abs w = 3 :=
by sorry

end max_absolute_value_constrained_complex_l254_25445


namespace total_students_at_concert_l254_25405

/-- The number of buses used for the concert. -/
def num_buses : ℕ := 8

/-- The number of students each bus can carry. -/
def students_per_bus : ℕ := 45

/-- Theorem: The total number of students who went to the concert is 360. -/
theorem total_students_at_concert : num_buses * students_per_bus = 360 := by
  sorry

end total_students_at_concert_l254_25405


namespace toms_apple_purchase_l254_25443

/-- The problem of determining how many kg of apples Tom purchased -/
theorem toms_apple_purchase (apple_price mango_price total_paid : ℕ) 
  (mango_quantity : ℕ) (h1 : apple_price = 70) (h2 : mango_price = 75) 
  (h3 : mango_quantity = 9) (h4 : total_paid = 1235) :
  ∃ (apple_quantity : ℕ), 
    apple_quantity * apple_price + mango_quantity * mango_price = total_paid ∧ 
    apple_quantity = 8 := by
  sorry

end toms_apple_purchase_l254_25443


namespace inequality_system_solution_l254_25413

theorem inequality_system_solution (x : ℝ) :
  (2 * (x - 1) ≤ x + 1 ∧ (x + 2) / 2 ≥ (x + 3) / 3) ↔ (0 ≤ x ∧ x ≤ 3) := by
  sorry

end inequality_system_solution_l254_25413


namespace oliver_new_cards_l254_25481

/-- Calculates the number of new baseball cards Oliver had -/
def new_cards (cards_per_page : ℕ) (total_pages : ℕ) (old_cards : ℕ) : ℕ :=
  cards_per_page * total_pages - old_cards

/-- Proves that Oliver had 2 new baseball cards -/
theorem oliver_new_cards : new_cards 3 4 10 = 2 := by
  sorry

end oliver_new_cards_l254_25481


namespace octal_367_equals_decimal_247_l254_25492

-- Define the octal number as a list of digits
def octal_number : List Nat := [3, 6, 7]

-- Define the conversion function from octal to decimal
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem octal_367_equals_decimal_247 :
  octal_to_decimal octal_number = 247 := by
  sorry

end octal_367_equals_decimal_247_l254_25492


namespace sum_of_three_circles_l254_25400

theorem sum_of_three_circles (square circle : ℝ) 
  (eq1 : 3 * square + 2 * circle = 27)
  (eq2 : 2 * square + 3 * circle = 25) : 
  3 * circle = 12.6 := by
  sorry

end sum_of_three_circles_l254_25400


namespace expression_simplification_l254_25465

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  ((a + b)^2 + 2*b^2) / (a^3 - b^3) - 1 / (a - b) + (a + b) / (a^2 + a*b + b^2) *
  (1 / b - 1 / a) = 1 / (a * b) := by
  sorry

end expression_simplification_l254_25465


namespace vector_from_origin_to_line_l254_25494

/-- A line parameterized by t -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The given line -/
def givenLine : ParametricLine where
  x := λ t => 3 * t + 1
  y := λ t => 2 * t + 3

/-- Check if a vector is parallel to another vector -/
def isParallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Check if a point lies on the given line -/
def liesOnLine (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p.1 = givenLine.x t ∧ p.2 = givenLine.y t

theorem vector_from_origin_to_line : 
  liesOnLine (-3, -2) ∧ 
  isParallel (-3, -2) (3, 2) := by
  sorry

#check vector_from_origin_to_line

end vector_from_origin_to_line_l254_25494


namespace sheila_hourly_rate_l254_25415

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  monday_hours : ℕ
  tuesday_hours : ℕ
  wednesday_hours : ℕ
  thursday_hours : ℕ
  friday_hours : ℕ
  weekly_earnings : ℕ

/-- Calculate total weekly hours -/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  schedule.monday_hours + schedule.tuesday_hours + schedule.wednesday_hours +
  schedule.thursday_hours + schedule.friday_hours

/-- Calculate hourly rate -/
def hourly_rate (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_weekly_hours schedule)

/-- Sheila's work schedule -/
def sheila_schedule : WorkSchedule :=
  { monday_hours := 8
    tuesday_hours := 6
    wednesday_hours := 8
    thursday_hours := 6
    friday_hours := 8
    weekly_earnings := 360 }

/-- Theorem: Sheila's hourly rate is $10 -/
theorem sheila_hourly_rate :
  hourly_rate sheila_schedule = 10 := by
  sorry


end sheila_hourly_rate_l254_25415


namespace sock_pair_combinations_l254_25419

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_sock_pairs (white brown blue : ℕ) : ℕ :=
  white * brown + white * blue + brown * blue

/-- Theorem: Given 5 white socks, 3 brown socks, and 2 blue socks,
    there are 31 ways to choose a pair of socks with different colors -/
theorem sock_pair_combinations :
  different_color_sock_pairs 5 3 2 = 31 := by
  sorry

#eval different_color_sock_pairs 5 3 2

end sock_pair_combinations_l254_25419


namespace not_perfect_square_l254_25410

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℤ, (3 : ℤ)^n + 2 * (17 : ℤ)^n = k^2 := by
  sorry

end not_perfect_square_l254_25410


namespace negation_of_proposition_l254_25411

theorem negation_of_proposition (a : ℝ) (h : 0 < a ∧ a < 1) :
  (¬ (∀ x : ℝ, x < 0 → a^x > 1)) ↔ (∃ x₀ : ℝ, x₀ < 0 ∧ a^x₀ ≤ 1) :=
by sorry

end negation_of_proposition_l254_25411


namespace renovation_project_material_l254_25435

theorem renovation_project_material (sand dirt cement : ℝ) 
  (h_sand : sand = 0.17)
  (h_dirt : dirt = 0.33)
  (h_cement : cement = 0.17) :
  sand + dirt + cement = 0.67 := by
  sorry

end renovation_project_material_l254_25435


namespace gcd_seven_eight_factorial_l254_25401

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_seven_eight_factorial :
  Nat.gcd (factorial 7) (factorial 8) = factorial 7 := by
  sorry

end gcd_seven_eight_factorial_l254_25401


namespace vertical_angles_congruence_equivalence_l254_25450

-- Define what it means for angles to be vertical
def are_vertical_angles (a b : Angle) : Prop := sorry

-- Define what it means for angles to be congruent
def are_congruent (a b : Angle) : Prop := sorry

-- The theorem to prove
theorem vertical_angles_congruence_equivalence :
  (∀ a b : Angle, are_vertical_angles a b → are_congruent a b) ↔
  (∀ a b : Angle, are_vertical_angles a b → are_congruent a b) :=
sorry

end vertical_angles_congruence_equivalence_l254_25450


namespace three_isosceles_triangles_l254_25424

-- Define a point in 2D space
structure Point :=
  (x : Int) (y : Int)

-- Define a triangle by its three vertices
structure Triangle :=
  (v1 : Point) (v2 : Point) (v3 : Point)

-- Function to calculate the squared distance between two points
def squaredDistance (p1 p2 : Point) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Function to check if a triangle is isosceles
def isIsosceles (t : Triangle) : Prop :=
  let d1 := squaredDistance t.v1 t.v2
  let d2 := squaredDistance t.v2 t.v3
  let d3 := squaredDistance t.v3 t.v1
  d1 = d2 ∨ d2 = d3 ∨ d3 = d1

-- Define the five triangles
def triangle1 : Triangle := ⟨⟨1, 5⟩, ⟨3, 5⟩, ⟨2, 2⟩⟩
def triangle2 : Triangle := ⟨⟨4, 3⟩, ⟨4, 6⟩, ⟨7, 3⟩⟩
def triangle3 : Triangle := ⟨⟨1, 1⟩, ⟨5, 2⟩, ⟨9, 1⟩⟩
def triangle4 : Triangle := ⟨⟨7, 5⟩, ⟨6, 6⟩, ⟨9, 3⟩⟩
def triangle5 : Triangle := ⟨⟨8, 2⟩, ⟨10, 5⟩, ⟨10, 0⟩⟩

-- Theorem: Exactly 3 out of the 5 given triangles are isosceles
theorem three_isosceles_triangles :
  (isIsosceles triangle1 ∧ isIsosceles triangle2 ∧ isIsosceles triangle3 ∧
   ¬isIsosceles triangle4 ∧ ¬isIsosceles triangle5) :=
by sorry

end three_isosceles_triangles_l254_25424


namespace number_ordering_l254_25454

def A : ℕ := (Nat.factorial 8) ^ (Nat.factorial 8)
def B : ℕ := 8 ^ (8 ^ 8)
def C : ℕ := 8 ^ 88
def D : ℕ := (8 ^ 8) ^ 8

theorem number_ordering : D < C ∧ C < B ∧ B < A := by sorry

end number_ordering_l254_25454


namespace proportional_calculation_l254_25402

/-- Given that 2994 ã · 14.5 = 171, prove that 29.94 ã · 1.45 = 1.71 -/
theorem proportional_calculation (h : 2994 * 14.5 = 171) : 29.94 * 1.45 = 1.71 := by
  sorry

end proportional_calculation_l254_25402


namespace min_value_a_k_l254_25480

/-- A positive arithmetic sequence satisfying the given condition -/
def PositiveArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  (∀ k : ℕ, k ≥ 2 → 1 / a 1 + 4 / a (2 * k - 1) ≤ 1)

/-- The theorem stating the minimum value of a_k -/
theorem min_value_a_k (a : ℕ → ℝ) (h : PositiveArithmeticSequence a) :
    ∀ k : ℕ, k ≥ 2 → a k ≥ 9/2 :=
  sorry

end min_value_a_k_l254_25480


namespace consecutive_even_numbers_divisible_by_eight_l254_25474

theorem consecutive_even_numbers_divisible_by_eight (n : ℤ) : 
  ∃ k : ℤ, 4 * n * (n + 1) = 8 * k := by
  sorry

end consecutive_even_numbers_divisible_by_eight_l254_25474


namespace sum_of_tenth_set_l254_25475

/-- Calculates the sum of the first n triangular numbers -/
def sumOfTriangularNumbers (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- Calculates the first element of the nth set -/
def firstElementOfSet (n : ℕ) : ℕ := sumOfTriangularNumbers (n - 1) + 1

/-- Calculates the number of elements in the nth set -/
def numberOfElementsInSet (n : ℕ) : ℕ := n + 2 * (n - 1)

/-- Calculates the last element of the nth set -/
def lastElementOfSet (n : ℕ) : ℕ := firstElementOfSet n + numberOfElementsInSet n - 1

/-- Calculates the sum of elements in the nth set -/
def sumOfSet (n : ℕ) : ℕ := 
  (numberOfElementsInSet n * (firstElementOfSet n + lastElementOfSet n)) / 2

theorem sum_of_tenth_set : sumOfSet 10 = 5026 := by sorry

end sum_of_tenth_set_l254_25475


namespace simeon_water_consumption_l254_25428

/-- Simeon's daily water consumption in fluid ounces -/
def daily_water : ℕ := 64

/-- Size of old serving in fluid ounces -/
def old_serving : ℕ := 8

/-- Size of new serving in fluid ounces -/
def new_serving : ℕ := 16

/-- Difference in number of servings -/
def serving_difference : ℕ := 4

theorem simeon_water_consumption :
  ∃ (old_servings new_servings : ℕ),
    old_servings * old_serving = daily_water ∧
    new_servings * new_serving = daily_water ∧
    old_servings = new_servings + serving_difference :=
by sorry

end simeon_water_consumption_l254_25428


namespace well_climbing_l254_25457

/-- Proves that a man climbing out of a well slips down 3 meters each day -/
theorem well_climbing (well_depth : ℝ) (days : ℕ) (climb_up : ℝ) (slip_down : ℝ) 
  (h1 : well_depth = 30)
  (h2 : days = 27)
  (h3 : climb_up = 4)
  (h4 : (days - 1) * (climb_up - slip_down) + climb_up = well_depth) :
  slip_down = 3 := by
  sorry


end well_climbing_l254_25457


namespace horner_method_v3_l254_25455

def horner_polynomial (x : ℝ) : ℝ := 2*x^6 + 5*x^4 + x^3 + 7*x^2 + 3*x + 1

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x + 0
  let v2 := v1 * x + 5
  v2 * x + 1

theorem horner_method_v3 :
  horner_v3 3 = 70 :=
by sorry

end horner_method_v3_l254_25455


namespace points_four_units_from_negative_two_l254_25423

theorem points_four_units_from_negative_two (x : ℝ) : 
  (|x - (-2)| = 4) ↔ (x = 2 ∨ x = -6) := by sorry

end points_four_units_from_negative_two_l254_25423
