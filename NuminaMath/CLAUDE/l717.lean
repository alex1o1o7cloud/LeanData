import Mathlib

namespace NUMINAMATH_CALUDE_orange_count_l717_71714

/-- Given the ratio of mangoes : oranges : apples and the number of mangoes and apples,
    calculate the number of oranges -/
theorem orange_count (mango_ratio orange_ratio apple_ratio mango_count apple_count : ℕ) :
  mango_ratio ≠ 0 →
  orange_ratio ≠ 0 →
  apple_ratio ≠ 0 →
  mango_ratio = 10 →
  orange_ratio = 2 →
  apple_ratio = 3 →
  mango_count = 120 →
  apple_count = 36 →
  mango_count / mango_ratio = apple_count / apple_ratio →
  (mango_count / mango_ratio) * orange_ratio = 24 := by
sorry

end NUMINAMATH_CALUDE_orange_count_l717_71714


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l717_71735

theorem solution_set_of_inequality (x : ℝ) :
  (x + 3) / (2 * x - 1) < 0 ↔ -3 < x ∧ x < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l717_71735


namespace NUMINAMATH_CALUDE_largest_multiple_six_negation_greater_than_neg_150_l717_71703

theorem largest_multiple_six_negation_greater_than_neg_150 :
  ∀ n : ℤ, (∃ k : ℤ, n = 6 * k) → -n > -150 → n ≤ 144 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_six_negation_greater_than_neg_150_l717_71703


namespace NUMINAMATH_CALUDE_sum_of_cubes_l717_71715

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l717_71715


namespace NUMINAMATH_CALUDE_find_d_l717_71730

theorem find_d (a b c d : ℕ+) 
  (h1 : a.val ^ 2 = c.val * (d.val + 20)) 
  (h2 : b.val ^ 2 = c.val * (d.val - 18)) : 
  d.val = 180 := by
  sorry

end NUMINAMATH_CALUDE_find_d_l717_71730


namespace NUMINAMATH_CALUDE_fair_coin_probability_difference_l717_71709

def probability_n_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1/2)^n

theorem fair_coin_probability_difference : 
  (probability_n_heads 4 3) - (probability_n_heads 4 4) = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_probability_difference_l717_71709


namespace NUMINAMATH_CALUDE_number_problem_l717_71734

theorem number_problem (N : ℝ) :
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 10 →
  (40/100 : ℝ) * N = 120 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l717_71734


namespace NUMINAMATH_CALUDE_move_up_coordinates_l717_71766

/-- Moving a point up in a 2D coordinate system -/
def move_up (p : ℝ × ℝ) (n : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + n)

/-- The theorem states that moving a point up by n units results in the expected coordinates -/
theorem move_up_coordinates (x y n : ℝ) :
  move_up (x, y) n = (x, y + n) := by
  sorry

end NUMINAMATH_CALUDE_move_up_coordinates_l717_71766


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l717_71718

def U : Set ℕ := {1, 2, 3, 4, 5}

def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | ∃ α ∈ A, x = 2*α}

theorem complement_of_A_union_B (h : Set ℕ) : 
  h = U \ (A ∪ B) → h = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l717_71718


namespace NUMINAMATH_CALUDE_eggs_left_l717_71780

/-- Given a box with 47 eggs, if Harry takes 5 eggs and Susan takes x eggs,
    then the number of eggs left in the box is equal to 42 - x. -/
theorem eggs_left (x : ℕ) : 47 - 5 - x = 42 - x := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_l717_71780


namespace NUMINAMATH_CALUDE_number_transformation_l717_71747

theorem number_transformation (initial_number : ℕ) : 
  initial_number = 6 → 3 * ((2 * initial_number) + 9) = 63 := by
  sorry

end NUMINAMATH_CALUDE_number_transformation_l717_71747


namespace NUMINAMATH_CALUDE_expected_games_at_negative_one_l717_71725

/-- The expected number of games in a best-of-five series -/
def f (x : ℝ) : ℝ :=
  3 * (x^3 + (1-x)^3) + 
  4 * (3*x^3*(1-x) + 3*(1-x)^3*x) + 
  5 * (6*x^2*(1-x)^2)

/-- Theorem: The expected number of games when x = -1 is 21 -/
theorem expected_games_at_negative_one : f (-1) = 21 := by
  sorry

end NUMINAMATH_CALUDE_expected_games_at_negative_one_l717_71725


namespace NUMINAMATH_CALUDE_odd_and_div_by_5_probability_l717_71733

/-- A set of digits to form a four-digit number -/
def digits : Finset Nat := {8, 5, 9, 7}

/-- Predicate for a number being odd and divisible by 5 -/
def is_odd_and_div_by_5 (n : Nat) : Prop :=
  n % 2 = 1 ∧ n % 5 = 0

/-- The total number of possible four-digit numbers -/
def total_permutations : Nat := Nat.factorial 4

/-- The number of valid permutations (odd and divisible by 5) -/
def valid_permutations : Nat := Nat.factorial 3

/-- The probability of forming a number that is odd and divisible by 5 -/
def probability : Rat := valid_permutations / total_permutations

theorem odd_and_div_by_5_probability :
  probability = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_odd_and_div_by_5_probability_l717_71733


namespace NUMINAMATH_CALUDE_sequences_sum_and_diff_total_l717_71728

def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def sequence1_sum : ℤ := arithmetic_sum 4 10 6
def sequence2_sum : ℤ := arithmetic_sum 12 10 6

theorem sequences_sum_and_diff_total : 
  (sequence1_sum + sequence2_sum) + (sequence2_sum - sequence1_sum) = 444 := by
  sorry

end NUMINAMATH_CALUDE_sequences_sum_and_diff_total_l717_71728


namespace NUMINAMATH_CALUDE_seths_ice_cream_purchase_l717_71788

/-- Seth's ice cream purchase problem -/
theorem seths_ice_cream_purchase
  (ice_cream_cost : ℕ → ℕ)
  (yogurt_cost : ℕ)
  (yogurt_quantity : ℕ)
  (cost_difference : ℕ)
  (h1 : yogurt_quantity = 2)
  (h2 : ∀ n, ice_cream_cost n = 6 * n)
  (h3 : yogurt_cost = 1)
  (h4 : ∃ x : ℕ, ice_cream_cost x = yogurt_quantity * yogurt_cost + cost_difference)
  (h5 : cost_difference = 118) :
  ∃ x : ℕ, ice_cream_cost x = yogurt_quantity * yogurt_cost + cost_difference ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_seths_ice_cream_purchase_l717_71788


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l717_71772

/-- Given a triangle ABC with angles B = 60°, C = 75°, and side a = 4,
    prove that side b = 2√6 -/
theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) : 
  B = π / 3 →  -- 60° in radians
  C = 5 * π / 12 →  -- 75° in radians
  a = 4 →
  A + B + C = π →  -- Sum of angles in a triangle
  a / Real.sin A = b / Real.sin B →  -- Law of Sines
  b = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l717_71772


namespace NUMINAMATH_CALUDE_sum_of_coordinates_B_l717_71750

/-- Given point A at (0, 0), point B on the line y = 6, and the slope of segment AB is 3/4,
    the sum of the x- and y-coordinates of point B is 14. -/
theorem sum_of_coordinates_B (B : ℝ × ℝ) : 
  B.2 = 6 ∧ (B.2 - 0) / (B.1 - 0) = 3/4 → B.1 + B.2 = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_B_l717_71750


namespace NUMINAMATH_CALUDE_exists_m_in_range_l717_71787

def sequence_x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (sequence_x n ^ 2 + 7 * sequence_x n + 12) / (sequence_x n + 8)

theorem exists_m_in_range :
  ∃ m : ℕ, 81 ≤ m ∧ m ≤ 242 ∧
  sequence_x m ≤ 5 + 1 / 2^15 ∧
  ∀ k : ℕ, 0 < k ∧ k < m → sequence_x k > 5 + 1 / 2^15 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_in_range_l717_71787


namespace NUMINAMATH_CALUDE_coefficient_x4_expansion_l717_71765

def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem coefficient_x4_expansion :
  let n : ℕ := 8
  let k : ℕ := 4
  let a : ℝ := 1
  let b : ℝ := 3 * Real.sqrt 3
  binomial_coefficient n k * a^(n-k) * b^k = 51030 := by sorry

end NUMINAMATH_CALUDE_coefficient_x4_expansion_l717_71765


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l717_71740

-- Define the triangle
def triangle (x : ℝ) : Fin 3 → ℝ
| 0 => 8
| 1 => 2*x + 5
| 2 => 3*x - 1
| _ => 0  -- This case is never reached due to Fin 3

-- State the theorem
theorem longest_side_of_triangle :
  ∃ x : ℝ, 
    (triangle x 0 + triangle x 1 + triangle x 2 = 45) ∧ 
    (∀ i : Fin 3, triangle x i ≤ 18.8) ∧
    (∃ i : Fin 3, triangle x i = 18.8) :=
by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l717_71740


namespace NUMINAMATH_CALUDE_tetrahedron_volume_prove_tetrahedron_volume_l717_71754

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  /-- Length of edge AB in cm -/
  ab_length : ℝ
  /-- Area of face ABC in cm² -/
  abc_area : ℝ
  /-- Area of face ABD in cm² -/
  abd_area : ℝ
  /-- Angle between faces ABC and ABD in radians -/
  face_angle : ℝ
  /-- Conditions on the tetrahedron -/
  ab_length_eq : ab_length = 3
  abc_area_eq : abc_area = 15
  abd_area_eq : abd_area = 12
  face_angle_eq : face_angle = Real.pi / 6

/-- The volume of the tetrahedron is 20 cm³ -/
theorem tetrahedron_volume (t : Tetrahedron) : ℝ :=
  20

#check tetrahedron_volume

/-- Proof of the tetrahedron volume -/
theorem prove_tetrahedron_volume (t : Tetrahedron) :
  tetrahedron_volume t = 20 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_prove_tetrahedron_volume_l717_71754


namespace NUMINAMATH_CALUDE_dima_guarantee_win_or_draw_l717_71790

/-- Represents a player in the game -/
inductive Player : Type
| Gosha : Player
| Dima : Player

/-- Represents a cell on the board -/
structure Cell :=
(row : Nat)
(col : Nat)

/-- Represents the game board -/
def Board := List Cell

/-- Represents a game state -/
structure GameState :=
(board : Board)
(currentPlayer : Player)

/-- Checks if a sequence of 7 consecutive cells is occupied -/
def isWinningSequence (sequence : List Cell) (board : Board) : Bool :=
  sorry

/-- Checks if the game is in a winning state for the current player -/
def isWinningState (state : GameState) : Bool :=
  sorry

/-- Represents a game strategy -/
def Strategy := GameState → Cell

/-- Theorem: Dima can guarantee a win or draw -/
theorem dima_guarantee_win_or_draw :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      game.currentPlayer = Player.Dima →
      (∃ (future_game : GameState), 
        isWinningState future_game ∧ future_game.currentPlayer = Player.Dima) ∨
      (∀ (future_game : GameState), ¬isWinningState future_game) :=
sorry

end NUMINAMATH_CALUDE_dima_guarantee_win_or_draw_l717_71790


namespace NUMINAMATH_CALUDE_reciprocal_problem_l717_71778

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 4) : 150 * (1 / x) = 300 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l717_71778


namespace NUMINAMATH_CALUDE_frank_lamp_purchase_l717_71756

theorem frank_lamp_purchase (cheapest_lamp : ℕ) (frank_money : ℕ) :
  cheapest_lamp = 20 →
  frank_money = 90 →
  frank_money - (3 * cheapest_lamp) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_frank_lamp_purchase_l717_71756


namespace NUMINAMATH_CALUDE_greatest_four_digit_sum_15_l717_71722

/-- A function that returns true if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that returns the product of digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The theorem stating that the sum of digits of the greatest four-digit number
    with digit product 36 is 15 -/
theorem greatest_four_digit_sum_15 :
  ∃ M : ℕ, is_four_digit M ∧ 
           digit_product M = 36 ∧ 
           (∀ n : ℕ, is_four_digit n → digit_product n = 36 → n ≤ M) ∧
           digit_sum M = 15 := by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_sum_15_l717_71722


namespace NUMINAMATH_CALUDE_evaluate_expression_l717_71717

theorem evaluate_expression : 5 - (-3)^(2 - (1 - 3)) = -76 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l717_71717


namespace NUMINAMATH_CALUDE_min_xy_equals_nine_l717_71753

theorem min_xy_equals_nine (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 1 / (x + 1) + 1 / (y + 1) = 1 / 2) :
  ∀ z, z = x * y → z ≥ 9 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ 1 / (a + 1) + 1 / (b + 1) = 1 / 2 ∧ a * b = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_equals_nine_l717_71753


namespace NUMINAMATH_CALUDE_marshmallow_challenge_l717_71738

/-- The marshmallow challenge problem -/
theorem marshmallow_challenge (haley michael brandon sofia : ℕ) : 
  haley = 8 →
  michael = 3 * haley →
  brandon = michael / 2 →
  sofia = 2 * (haley + brandon) →
  haley + michael + brandon + sofia = 84 := by
  sorry

end NUMINAMATH_CALUDE_marshmallow_challenge_l717_71738


namespace NUMINAMATH_CALUDE_solve_for_a_l717_71719

theorem solve_for_a (x y a : ℝ) : 
  x + y = 1 → 
  2 * x + y = 0 → 
  a * x - 3 * y = 0 → 
  a = -6 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l717_71719


namespace NUMINAMATH_CALUDE_polynomial_identity_l717_71704

theorem polynomial_identity (x : ℝ) :
  (5 * x^3 - 32 * x^2 + 75 * x - 71 = 
   5 * (x - 2)^3 + (-2) * (x - 2)^2 + 7 * (x - 2) + (-9)) ∧
  (∀ (a b c d : ℝ), 
    (∀ x : ℝ, 5 * x^3 - 32 * x^2 + 75 * x - 71 = 
      a * (x - 2)^3 + b * (x - 2)^2 + c * (x - 2) + d) →
    a = 5 ∧ b = -2 ∧ c = 7 ∧ d = -9) := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l717_71704


namespace NUMINAMATH_CALUDE_rectangular_field_area_l717_71716

theorem rectangular_field_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  width = (1/3) * length →
  2 * (width + length) = 80 →
  width * length = 300 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l717_71716


namespace NUMINAMATH_CALUDE_sixPeopleArrangements_l717_71742

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange six people in a row with Person A and Person B adjacent. -/
def adjacentArrangements : ℕ :=
  arrangements 2 * arrangements 5

/-- The number of ways to arrange six people in a row with Person A and Person B not adjacent. -/
def nonAdjacentArrangements : ℕ :=
  arrangements 4 * arrangements 2

/-- The number of ways to arrange six people in a row with exactly two people between Person A and Person B. -/
def twoPersonsBetweenArrangements : ℕ :=
  arrangements 2 * arrangements 2 * arrangements 3

/-- The number of ways to arrange six people in a row with Person A not at the left end and Person B not at the right end. -/
def notAtEndsArrangements : ℕ :=
  arrangements 6 - 2 * arrangements 5 + arrangements 4

theorem sixPeopleArrangements :
  adjacentArrangements = 240 ∧
  nonAdjacentArrangements = 480 ∧
  twoPersonsBetweenArrangements = 144 ∧
  notAtEndsArrangements = 504 := by
  sorry

end NUMINAMATH_CALUDE_sixPeopleArrangements_l717_71742


namespace NUMINAMATH_CALUDE_correct_diagnosis_l717_71762

structure Doctor where
  name : String
  statements : List String

structure Patient where
  diagnosis : List String

def homeopath : Doctor :=
  { name := "Homeopath"
  , statements := 
    [ "The patient has a strong astigmatism"
    , "The patient smokes too much"
    , "The patient has a tropical fever"
    ]
  }

def therapist : Doctor :=
  { name := "Therapist"
  , statements := 
    [ "The patient has a strong astigmatism"
    , "The patient doesn't eat well"
    , "The patient suffers from high blood pressure"
    ]
  }

def ophthalmologist : Doctor :=
  { name := "Ophthalmologist"
  , statements := 
    [ "The patient has a strong astigmatism"
    , "The patient is near-sighted"
    , "The patient has no signs of retinal detachment"
    ]
  }

def correct_statements : List (Doctor × Nat) :=
  [ (homeopath, 1)
  , (therapist, 0)
  , (ophthalmologist, 0)
  ]

theorem correct_diagnosis (doctors : List Doctor) 
  (correct : List (Doctor × Nat)) : 
  ∃ (p : Patient), 
    p.diagnosis = 
      [ "I have a strong astigmatism"
      , "I smoke too much"
      , "I am not eating well enough!"
      , "I do not have tropical fever"
      ] :=
  sorry

end NUMINAMATH_CALUDE_correct_diagnosis_l717_71762


namespace NUMINAMATH_CALUDE_square_division_reversible_l717_71737

/-- A square of cells can be divided into equal figures -/
structure CellSquare where
  side : ℕ
  total_cells : ℕ
  total_cells_eq : total_cells = side * side

/-- A division of a cell square into equal figures -/
structure SquareDivision (square : CellSquare) where
  num_figures : ℕ
  cells_per_figure : ℕ
  division_valid : square.total_cells = num_figures * cells_per_figure

theorem square_division_reversible (square : CellSquare) 
  (div1 : SquareDivision square) :
  ∃ (div2 : SquareDivision square), 
    div2.num_figures = div1.cells_per_figure ∧ 
    div2.cells_per_figure = div1.num_figures :=
sorry

end NUMINAMATH_CALUDE_square_division_reversible_l717_71737


namespace NUMINAMATH_CALUDE_smallest_multiple_105_with_105_divisors_l717_71797

/-- The number of positive integral divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is the smallest positive integer that is a multiple of 105 and has exactly 105 positive integral divisors -/
def n : ℕ := sorry

theorem smallest_multiple_105_with_105_divisors :
  n > 0 ∧ 
  105 ∣ n ∧ 
  num_divisors n = 105 ∧ 
  ∀ m : ℕ, m > 0 → 105 ∣ m → num_divisors m = 105 → m ≥ n ∧
  n / 105 = 6289125 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_105_with_105_divisors_l717_71797


namespace NUMINAMATH_CALUDE_printing_presses_l717_71770

theorem printing_presses (time1 time2 : ℝ) (newspapers1 newspapers2 : ℕ) (presses2 : ℕ) :
  time1 = 6 →
  time2 = 9 →
  newspapers1 = 8000 →
  newspapers2 = 6000 →
  presses2 = 2 →
  ∃ (presses1 : ℕ), 
    (presses1 : ℝ) * (newspapers2 : ℝ) / (time2 * presses2) = newspapers1 / time1 ∧
    presses1 = 4 :=
by sorry


end NUMINAMATH_CALUDE_printing_presses_l717_71770


namespace NUMINAMATH_CALUDE_propositions_true_l717_71732

-- Define the propositions
def proposition1 (x y : ℝ) : Prop := x + y = 0 → (x = -y ∨ y = -x)
def proposition3 (q : ℝ) : Prop := q ≤ 1 → ∃ x : ℝ, x^2 + 2*x + q = 0

-- Theorem statement
theorem propositions_true :
  (∀ x y : ℝ, ¬(x + y = 0) → ¬(x = -y ∨ y = -x)) ∧
  (∀ q : ℝ, (¬∃ x : ℝ, x^2 + 2*x + q = 0) → ¬(q ≤ 1)) := by sorry

end NUMINAMATH_CALUDE_propositions_true_l717_71732


namespace NUMINAMATH_CALUDE_red_beads_count_l717_71793

theorem red_beads_count (green : ℕ) (brown : ℕ) (taken : ℕ) (left : ℕ) : 
  green = 1 → brown = 2 → taken = 2 → left = 4 → 
  ∃ (red : ℕ), red = (green + brown + taken + left) - (green + brown) :=
by sorry

end NUMINAMATH_CALUDE_red_beads_count_l717_71793


namespace NUMINAMATH_CALUDE_opposite_faces_l717_71752

/-- Represents the six faces of a cube -/
inductive Face : Type
  | xiao : Face  -- 小
  | xue  : Face  -- 学
  | xi   : Face  -- 希
  | wang : Face  -- 望
  | bei  : Face  -- 杯
  | sai  : Face  -- 赛

/-- Defines the adjacency relationship between faces -/
def adjacent : Face → Face → Prop :=
  sorry

/-- Defines the opposite relationship between faces -/
def opposite : Face → Face → Prop :=
  sorry

/-- The cube configuration satisfies the given conditions -/
axiom cube_config :
  adjacent Face.xue Face.xiao ∧
  adjacent Face.xue Face.xi ∧
  adjacent Face.xue Face.wang ∧
  adjacent Face.xue Face.sai

/-- Theorem stating the opposite face relationships -/
theorem opposite_faces :
  opposite Face.xi Face.sai ∧
  opposite Face.wang Face.xiao ∧
  opposite Face.bei Face.xue :=
by sorry

end NUMINAMATH_CALUDE_opposite_faces_l717_71752


namespace NUMINAMATH_CALUDE_stone_arrangement_exists_l717_71784

theorem stone_arrangement_exists (P : ℕ) (h : P = 23) : ∃ (F : ℕ → ℤ), 
  F 0 = 0 ∧ 
  F 1 = 1 ∧ 
  (∀ i : ℕ, i ≥ 2 → F i = 3 * F (i - 1) - F (i - 2)) ∧
  F 12 % P = 0 :=
by sorry

end NUMINAMATH_CALUDE_stone_arrangement_exists_l717_71784


namespace NUMINAMATH_CALUDE_squats_on_fourth_day_l717_71792

/-- Calculates the number of squats on a given day, given the initial number of squats and daily increase. -/
def squats_on_day (initial_squats : ℕ) (daily_increase : ℕ) (day : ℕ) : ℕ :=
  initial_squats + (day - 1) * daily_increase

/-- Theorem stating that on the fourth day, the number of squats will be 45, given the initial conditions. -/
theorem squats_on_fourth_day :
  squats_on_day 30 5 4 = 45 := by
  sorry

end NUMINAMATH_CALUDE_squats_on_fourth_day_l717_71792


namespace NUMINAMATH_CALUDE_acid_concentration_increase_l717_71712

theorem acid_concentration_increase (initial_volume initial_concentration water_removed : ℝ) :
  initial_volume = 18 →
  initial_concentration = 0.4 →
  water_removed = 6 →
  let acid_amount := initial_volume * initial_concentration
  let final_volume := initial_volume - water_removed
  let final_concentration := acid_amount / final_volume
  final_concentration = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_acid_concentration_increase_l717_71712


namespace NUMINAMATH_CALUDE_intersection_M_N_l717_71746

open Set

def M : Set ℝ := {x : ℝ | 3 * x - x^2 > 0}
def N : Set ℝ := {x : ℝ | x^2 - 4 * x + 3 > 0}

theorem intersection_M_N : M ∩ N = Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l717_71746


namespace NUMINAMATH_CALUDE_test_probability_l717_71782

/-- The probability of answering exactly k questions correctly out of n questions,
    where the probability of answering each question correctly is p. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of answering exactly 2 questions correctly out of 6 questions,
    where the probability of answering each question correctly is 1/3, is 240/729. -/
theorem test_probability : binomial_probability 6 2 (1/3) = 240/729 := by
  sorry

end NUMINAMATH_CALUDE_test_probability_l717_71782


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_university_sample_sizes_correct_l717_71781

/-- Represents a stratum in a population -/
structure Stratum where
  size : ℕ

/-- Represents a population with stratified sampling -/
structure StratifiedPopulation where
  total : ℕ
  strata : List Stratum
  sample_size : ℕ

/-- Calculates the number of samples for a given stratum -/
def sample_size_for_stratum (pop : StratifiedPopulation) (stratum : Stratum) : ℕ :=
  (pop.sample_size * stratum.size) / pop.total

/-- Theorem: The sum of samples from all strata equals the total sample size -/
theorem stratified_sampling_theorem (pop : StratifiedPopulation) 
  (h : pop.total = (pop.strata.map Stratum.size).sum) :
  (pop.strata.map (sample_size_for_stratum pop)).sum = pop.sample_size := by
  sorry

/-- The university population -/
def university_pop : StratifiedPopulation :=
  { total := 5600
  , strata := [⟨1300⟩, ⟨3000⟩, ⟨1300⟩]
  , sample_size := 280 }

/-- Theorem: The calculated sample sizes for the university population are correct -/
theorem university_sample_sizes_correct :
  (university_pop.strata.map (sample_size_for_stratum university_pop)) = [65, 150, 65] := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_university_sample_sizes_correct_l717_71781


namespace NUMINAMATH_CALUDE_valid_outfits_count_l717_71731

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 8

/-- The number of colors shared by shirts, pants, and hats -/
def num_shared_colors : ℕ := 5

/-- The number of additional colors for shirts and hats -/
def num_additional_colors : ℕ := 2

/-- The total number of outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of combinations where shirt and hat have the same color -/
def same_color_combinations : ℕ := num_shared_colors * num_pants

/-- The number of valid outfit combinations -/
def valid_combinations : ℕ := total_combinations - same_color_combinations

theorem valid_outfits_count :
  valid_combinations = 295 := by sorry

end NUMINAMATH_CALUDE_valid_outfits_count_l717_71731


namespace NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_4_l717_71702

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isFirstYearAfter2010WithDigitSum4 (year : Nat) : Prop :=
  year > 2010 ∧ 
  sumOfDigits year = 4 ∧
  ∀ y, 2010 < y ∧ y < year → sumOfDigits y ≠ 4

theorem first_year_after_2010_with_digit_sum_4 :
  isFirstYearAfter2010WithDigitSum4 2011 := by
  sorry

end NUMINAMATH_CALUDE_first_year_after_2010_with_digit_sum_4_l717_71702


namespace NUMINAMATH_CALUDE_expected_socks_taken_l717_71739

/-- Represents a collection of socks -/
structure SockCollection where
  pairs : ℕ  -- number of pairs of socks
  nonIdentical : Bool  -- whether all pairs are non-identical

/-- Represents the process of selecting socks -/
def selectSocks (sc : SockCollection) : ℕ → ℕ
  | 0 => 0
  | n + 1 => n + 1  -- simplified representation of sock selection

/-- Expected number of socks taken until a pair is found -/
def expectedSocksTaken (sc : SockCollection) : ℝ :=
  2 * sc.pairs

/-- Theorem stating the expected number of socks taken is 2p -/
theorem expected_socks_taken (sc : SockCollection) (h1 : sc.nonIdentical = true) :
  expectedSocksTaken sc = 2 * sc.pairs := by
  sorry

#check expected_socks_taken

end NUMINAMATH_CALUDE_expected_socks_taken_l717_71739


namespace NUMINAMATH_CALUDE_gcd_linear_combination_l717_71723

theorem gcd_linear_combination (a b : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1) :
  Nat.gcd (11 * a + 2 * b).natAbs (18 * a + 5 * b).natAbs = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_linear_combination_l717_71723


namespace NUMINAMATH_CALUDE_two_digit_sum_l717_71711

/-- Given two single-digit natural numbers A and B, if 6A + B2 = 77, then B = 1 -/
theorem two_digit_sum (A B : ℕ) : 
  A < 10 → B < 10 → (60 + A) + (10 * B + 2) = 77 → B = 1 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_l717_71711


namespace NUMINAMATH_CALUDE_shoe_selection_theorem_l717_71706

/-- The number of ways to select 4 shoes from 10 pairs such that 2 form a pair and 2 do not -/
def shoeSelectionWays (totalPairs : Nat) : Nat :=
  if totalPairs = 10 then
    Nat.choose totalPairs 1 * Nat.choose (totalPairs - 1) 2 * 4
  else
    0

theorem shoe_selection_theorem :
  shoeSelectionWays 10 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_shoe_selection_theorem_l717_71706


namespace NUMINAMATH_CALUDE_circle_properties_l717_71771

/-- Given a circle with circumference 31.4 decimeters, prove its diameter, radius, and area -/
theorem circle_properties (C : Real) (h : C = 31.4) :
  ∃ (d r A : Real),
    d = 10 ∧ 
    r = 5 ∧ 
    A = 78.5 ∧
    C = 2 * Real.pi * r ∧
    d = 2 * r ∧
    A = Real.pi * r^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l717_71771


namespace NUMINAMATH_CALUDE_lucy_snowballs_l717_71700

theorem lucy_snowballs (charlie_snowballs : ℕ) (difference : ℕ) (lucy_snowballs : ℕ) : 
  charlie_snowballs = 50 → 
  difference = 31 → 
  charlie_snowballs = lucy_snowballs + difference → 
  lucy_snowballs = 19 := by
sorry

end NUMINAMATH_CALUDE_lucy_snowballs_l717_71700


namespace NUMINAMATH_CALUDE_g_composition_of_three_l717_71767

def g (x : ℝ) : ℝ := 3 * x - 5

theorem g_composition_of_three : g (g (g 3)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_of_three_l717_71767


namespace NUMINAMATH_CALUDE_jenga_players_l717_71760

/-- The number of players in a Jenga game -/
def num_players : ℕ := 5

/-- The initial number of blocks in the Jenga tower -/
def initial_blocks : ℕ := 54

/-- The number of full rounds played -/
def full_rounds : ℕ := 5

/-- The number of blocks remaining after 5 full rounds and one additional move -/
def remaining_blocks : ℕ := 28

/-- The number of blocks removed in the 6th round before the tower falls -/
def extra_blocks_removed : ℕ := 1

theorem jenga_players :
  initial_blocks - remaining_blocks = full_rounds * num_players + extra_blocks_removed :=
sorry

end NUMINAMATH_CALUDE_jenga_players_l717_71760


namespace NUMINAMATH_CALUDE_tax_rate_on_remaining_income_l717_71741

def total_earnings : ℝ := 100000
def deductions : ℝ := 30000
def first_bracket_limit : ℝ := 20000
def first_bracket_rate : ℝ := 0.1
def total_tax : ℝ := 12000

def taxable_income : ℝ := total_earnings - deductions

def tax_on_first_bracket : ℝ := first_bracket_limit * first_bracket_rate

def remaining_taxable_income : ℝ := taxable_income - first_bracket_limit

theorem tax_rate_on_remaining_income : 
  (total_tax - tax_on_first_bracket) / remaining_taxable_income = 0.2 := by sorry

end NUMINAMATH_CALUDE_tax_rate_on_remaining_income_l717_71741


namespace NUMINAMATH_CALUDE_angle_sum_tangent_l717_71708

theorem angle_sum_tangent (a β : Real) (ha : 0 < a ∧ a < π/2) (hβ : 0 < β ∧ β < π/2)
  (tan_a : Real.tan a = 2) (tan_β : Real.tan β = 3) :
  a + β = 3 * π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_tangent_l717_71708


namespace NUMINAMATH_CALUDE_existence_of_tangent_circle_l717_71794

/-- Given three circles with radii 1, 3, and 4 touching each other and the sides of a rectangle,
    there exists a circle touching all three circles and one side of the rectangle. -/
theorem existence_of_tangent_circle (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ = 1) (h₂ : r₂ = 3) (h₃ : r₃ = 4) : 
  ∃ x : ℝ, 
    (x + r₁)^2 - (x - r₁)^2 = (r₂ + x)^2 - (r₂ + r₁ - x)^2 ∧
    ∃ y : ℝ, 
      (y + r₂)^2 - (r₂ + r₁ - y)^2 = (r₃ + y)^2 - (r₃ - y)^2 ∧
      x = y := by
  sorry

end NUMINAMATH_CALUDE_existence_of_tangent_circle_l717_71794


namespace NUMINAMATH_CALUDE_remainder_of_expression_l717_71758

theorem remainder_of_expression (p t : ℕ) (hp : p > t) (ht : t > 1) :
  (92^p * 5^(p + t) + 11^t * 6^(p*t)) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_expression_l717_71758


namespace NUMINAMATH_CALUDE_fraction_comparison_l717_71777

theorem fraction_comparison : (1 : ℚ) / 4 = 24999999 / (10^8 : ℚ) + 1 / (4 * 10^8 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l717_71777


namespace NUMINAMATH_CALUDE_triangle_area_l717_71729

/-- Given a triangle ABC with side length a = 6, angle B = 30°, and angle C = 120°,
    prove that its area is 9√3. -/
theorem triangle_area (a b c : ℝ) (A B C : Real) : 
  a = 6 → B = 30 * Real.pi / 180 → C = 120 * Real.pi / 180 →
  (1/2) * a * b * Real.sin C = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l717_71729


namespace NUMINAMATH_CALUDE_train_length_l717_71710

/-- The length of a train given its speed, the speed of a person it passes, and the time it takes to pass them. -/
theorem train_length (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 63 →
  person_speed = 3 →
  passing_time = 53.99568034557235 →
  ∃ (length : ℝ), abs (length - 899.93) < 0.01 ∧
  length = (train_speed - person_speed) * (5 / 18) * passing_time :=
sorry

end NUMINAMATH_CALUDE_train_length_l717_71710


namespace NUMINAMATH_CALUDE_domain_of_f_l717_71774

noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arccos (Real.sin x))

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∀ k : ℤ, x ≠ k * Real.pi} :=
sorry

end NUMINAMATH_CALUDE_domain_of_f_l717_71774


namespace NUMINAMATH_CALUDE_expand_and_simplify_fraction_l717_71759

theorem expand_and_simplify_fraction (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_fraction_l717_71759


namespace NUMINAMATH_CALUDE_divisors_of_180_l717_71791

/-- The number of positive divisors of 180 is 18. -/
theorem divisors_of_180 : Finset.card (Nat.divisors 180) = 18 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_180_l717_71791


namespace NUMINAMATH_CALUDE_log_inequality_implies_a_range_l717_71744

theorem log_inequality_implies_a_range (a : ℝ) : 
  (∃ (loga : ℝ → ℝ → ℝ), loga a 3 < 1) → (a > 3 ∨ (0 < a ∧ a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_implies_a_range_l717_71744


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l717_71720

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2 * Real.sqrt (a*b*c)) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l717_71720


namespace NUMINAMATH_CALUDE_bob_has_62_pennies_l717_71743

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bob_pennies : ℕ := sorry

/-- If Alex gives Bob two pennies, Bob will have four times as many pennies as Alex has left -/
axiom condition1 : bob_pennies + 2 = 4 * (alex_pennies - 2)

/-- If Bob gives Alex two pennies, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bob_pennies - 2 = 3 * (alex_pennies + 2)

/-- Bob currently has 62 pennies -/
theorem bob_has_62_pennies : bob_pennies = 62 := by sorry

end NUMINAMATH_CALUDE_bob_has_62_pennies_l717_71743


namespace NUMINAMATH_CALUDE_custom_operation_result_l717_71789

/-- Custom operation ⊕ -/
def oplus (x y : ℝ) : ℝ := x + 2*y + 3

theorem custom_operation_result :
  ∀ (a b : ℝ), 
  (oplus (oplus (a^3) (a^2)) a = oplus (a^3) (oplus (a^2) a)) ∧
  (oplus (oplus (a^3) (a^2)) a = b) →
  a + b = 21/8 := by
sorry

end NUMINAMATH_CALUDE_custom_operation_result_l717_71789


namespace NUMINAMATH_CALUDE_employee_payment_percentage_l717_71748

theorem employee_payment_percentage (total payment_B : ℝ) 
  (h1 : total = 570)
  (h2 : payment_B = 228) : 
  (total - payment_B) / payment_B * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_employee_payment_percentage_l717_71748


namespace NUMINAMATH_CALUDE_correct_calculation_l717_71721

theorem correct_calculation (a b : ℝ) : 9 * a^2 * b - 9 * a^2 * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l717_71721


namespace NUMINAMATH_CALUDE_sum_coordinates_of_D_l717_71745

/-- Given a point M that is the midpoint of line segment CD, 
    prove that the sum of coordinates of D is 12 -/
theorem sum_coordinates_of_D (M C D : ℝ × ℝ) : 
  M = (2, 5) → 
  C = (1/2, 3/2) → 
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_D_l717_71745


namespace NUMINAMATH_CALUDE_min_value_theorem_l717_71726

theorem min_value_theorem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 2/b = 4) :
  9/4 ≤ a + 2*b ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 1/a₀ + 2/b₀ = 4 ∧ a₀ + 2*b₀ = 9/4 :=
by sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_min_value_theorem_l717_71726


namespace NUMINAMATH_CALUDE_negation_of_absolute_sine_bound_l717_71751

theorem negation_of_absolute_sine_bound :
  (¬ ∀ x : ℝ, |Real.sin x| ≤ 1) ↔ (∃ x : ℝ, |Real.sin x| > 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_sine_bound_l717_71751


namespace NUMINAMATH_CALUDE_intersection_M_and_naturals_l717_71768

def M : Set ℝ := {x | (x + 2) / (x - 1) ≤ 0}

theorem intersection_M_and_naturals :
  M ∩ Set.range (Nat.cast : ℕ → ℝ) = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_and_naturals_l717_71768


namespace NUMINAMATH_CALUDE_sophias_book_length_l717_71705

theorem sophias_book_length (total_pages : ℕ) : 
  (2 : ℚ) / 3 * total_pages = (1 : ℚ) / 3 * total_pages + 90 → 
  total_pages = 270 := by
sorry

end NUMINAMATH_CALUDE_sophias_book_length_l717_71705


namespace NUMINAMATH_CALUDE_susans_books_l717_71727

/-- Proves that Susan has 600 books given the conditions of the problem -/
theorem susans_books (susan_books : ℕ) (lidia_books : ℕ) : 
  lidia_books = 4 * susan_books → -- Lidia's collection is four times bigger than Susan's
  susan_books + lidia_books = 3000 → -- Total books is 3000
  susan_books = 600 := by
sorry

end NUMINAMATH_CALUDE_susans_books_l717_71727


namespace NUMINAMATH_CALUDE_arithmetic_sequence_l717_71755

def a (n : ℕ) : ℤ := 3 * n + 1

theorem arithmetic_sequence :
  ∀ n : ℕ, a (n + 1) - a n = (3 : ℤ) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_l717_71755


namespace NUMINAMATH_CALUDE_correct_pairing_l717_71736

structure Couple where
  wife : String
  husband : String
  wife_bottles : Nat
  husband_bottles : Nat

def total_bottles : Nat := 44

def couples : List Couple := [
  ⟨"Anna", "Smith", 2, 8⟩,
  ⟨"Betty", "White", 3, 9⟩,
  ⟨"Carol", "Green", 4, 8⟩,
  ⟨"Dorothy", "Brown", 5, 5⟩
]

theorem correct_pairing : 
  (couples.map (λ c => c.wife_bottles + c.husband_bottles)).sum = total_bottles ∧
  (∃ c ∈ couples, c.husband = "Brown" ∧ c.wife_bottles = c.husband_bottles) ∧
  (∃ c ∈ couples, c.husband = "Green" ∧ c.husband_bottles = 2 * c.wife_bottles) ∧
  (∃ c ∈ couples, c.husband = "White" ∧ c.husband_bottles = 3 * c.wife_bottles) ∧
  (∃ c ∈ couples, c.husband = "Smith" ∧ c.husband_bottles = 4 * c.wife_bottles) :=
by sorry

end NUMINAMATH_CALUDE_correct_pairing_l717_71736


namespace NUMINAMATH_CALUDE_sin_240_degrees_l717_71724

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l717_71724


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l717_71763

/-- The volume of a sphere inscribed in a cube with edge length 4 is 32π/3 -/
theorem inscribed_sphere_volume (cube_edge : ℝ) (sphere_volume : ℝ) :
  cube_edge = 4 →
  sphere_volume = (4 / 3) * π * (cube_edge / 2)^3 →
  sphere_volume = (32 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l717_71763


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l717_71761

open Set

-- Define the universal set U
def U : Set Int := {-1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 0}

-- Define set B
def B : Set Int := {0, 1, 2}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l717_71761


namespace NUMINAMATH_CALUDE_strictly_increasing_derivative_properties_l717_71798

/-- A function with a strictly increasing derivative -/
structure StrictlyIncreasingDerivative (f : ℝ → ℝ) : Prop where
  deriv_increasing : ∀ x y, x < y → (deriv f x) < (deriv f y)

/-- The main theorem -/
theorem strictly_increasing_derivative_properties
  (f : ℝ → ℝ) (hf : StrictlyIncreasingDerivative f) :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ ↔ f (x₁ + 1) + f x₂ > f x₁ + f (x₂ + 1)) ∧
  (StrictMono f ↔ ∀ x : ℝ, x < 0 → f x < f 0) :=
sorry

end NUMINAMATH_CALUDE_strictly_increasing_derivative_properties_l717_71798


namespace NUMINAMATH_CALUDE_two_hundred_twenty_fifth_number_with_digit_sum_2018_l717_71713

def digit_sum (n : ℕ) : ℕ := sorry

def nth_number_with_digit_sum (n : ℕ) (sum : ℕ) : ℕ := sorry

theorem two_hundred_twenty_fifth_number_with_digit_sum_2018 :
  nth_number_with_digit_sum 225 2018 = 39 * 10^224 + (10^224 - 10) * 9 + 8 :=
sorry

end NUMINAMATH_CALUDE_two_hundred_twenty_fifth_number_with_digit_sum_2018_l717_71713


namespace NUMINAMATH_CALUDE_room_width_calculation_l717_71785

/-- Given a rectangular room with specified length, total paving cost, and paving rate per square meter, 
    prove that the width of the room is as calculated. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) (width : ℝ) : 
  length = 7 →
  total_cost = 29925 →
  rate_per_sqm = 900 →
  width = total_cost / rate_per_sqm / length →
  width = 4.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l717_71785


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l717_71796

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l717_71796


namespace NUMINAMATH_CALUDE_keaton_yearly_earnings_l717_71783

/-- Represents Keaton's farm earnings --/
def farm_earnings (orange_harvest_interval : ℕ) (orange_harvest_value : ℕ) 
                  (apple_harvest_interval : ℕ) (apple_harvest_value : ℕ) : ℕ :=
  let orange_harvests_per_year := 12 / orange_harvest_interval
  let apple_harvests_per_year := 12 / apple_harvest_interval
  orange_harvests_per_year * orange_harvest_value + apple_harvests_per_year * apple_harvest_value

/-- Theorem stating Keaton's yearly earnings --/
theorem keaton_yearly_earnings : farm_earnings 2 50 3 30 = 420 := by
  sorry

end NUMINAMATH_CALUDE_keaton_yearly_earnings_l717_71783


namespace NUMINAMATH_CALUDE_sum_remainder_l717_71779

theorem sum_remainder (S : ℤ) : S = (2 * 3^500) / 3 → S % 1000 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l717_71779


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l717_71795

theorem quadratic_roots_problem (a b m p q : ℝ) : 
  (a^2 - m*a + 5 = 0) →
  (b^2 - m*b + 5 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 36/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l717_71795


namespace NUMINAMATH_CALUDE_problem_hexagon_area_l717_71769

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- A hexagon defined by six points -/
structure Hexagon where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point
  p5 : Point
  p6 : Point

/-- The area of a hexagon -/
def hexagonArea (h : Hexagon) : ℝ := sorry

/-- The specific hexagon in the problem -/
def problemHexagon : Hexagon := {
  p1 := { x := 0, y := 0 },
  p2 := { x := 2, y := 4 },
  p3 := { x := 6, y := 4 },
  p4 := { x := 8, y := 0 },
  p5 := { x := 6, y := -4 },
  p6 := { x := 2, y := -4 }
}

/-- Theorem stating that the area of the problem hexagon is 16 square units -/
theorem problem_hexagon_area : hexagonArea problemHexagon = 16 := by sorry

end NUMINAMATH_CALUDE_problem_hexagon_area_l717_71769


namespace NUMINAMATH_CALUDE_T_equiv_horizontal_lines_l717_71757

/-- The set of points R forming a right triangle PQR with area 4, where P(2,0) and Q(-2,0) -/
def T : Set (ℝ × ℝ) :=
  {R | ∃ (x y : ℝ), R = (x, y) ∧ 
       ((x - 2)^2 + y^2) * ((x + 2)^2 + y^2) = 16 * (x^2 + y^2) ∧
       (abs ((x - 2) * y - (x + 2) * y)) = 8}

/-- The set of points with y-coordinate equal to 2 or -2 -/
def horizontal_lines : Set (ℝ × ℝ) :=
  {R | ∃ (x y : ℝ), R = (x, y) ∧ (y = 2 ∨ y = -2)}

theorem T_equiv_horizontal_lines : T = horizontal_lines := by
  sorry

end NUMINAMATH_CALUDE_T_equiv_horizontal_lines_l717_71757


namespace NUMINAMATH_CALUDE_specific_card_draw_probability_l717_71776

theorem specific_card_draw_probability : 
  let deck_size : ℕ := 52
  let prob_specific_card : ℚ := 1 / deck_size
  let prob_both_specific_cards : ℚ := prob_specific_card * prob_specific_card
  prob_both_specific_cards = 1 / 2704 := by
  sorry

end NUMINAMATH_CALUDE_specific_card_draw_probability_l717_71776


namespace NUMINAMATH_CALUDE_figure_100_squares_l717_71764

/-- The number of nonoverlapping unit squares in figure n -/
def f (n : ℕ) : ℕ := 2 * n^3 + n^2 + 2 * n + 1

theorem figure_100_squares :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 25 ∧ f 3 = 63 → f 100 = 2010201 := by
  sorry

end NUMINAMATH_CALUDE_figure_100_squares_l717_71764


namespace NUMINAMATH_CALUDE_park_cycling_time_l717_71707

/-- Proves that for a rectangular park with given specifications, 
    a cyclist completes one round in 8 minutes -/
theorem park_cycling_time (length width : ℝ) (area perimeter : ℝ) (speed : ℝ) :
  width = 4 * length →
  area = length * width →
  area = 102400 →
  perimeter = 2 * (length + width) →
  speed = 12 * 1000 / 60 →
  (perimeter / speed) = 8 :=
by sorry

end NUMINAMATH_CALUDE_park_cycling_time_l717_71707


namespace NUMINAMATH_CALUDE_third_bottle_volume_is_250ml_l717_71701

/-- Represents the volume of milk in a bottle -/
structure MilkBottle where
  volume : ℝ
  unit : String

/-- Converts liters to milliliters -/
def litersToMilliliters (liters : ℝ) : ℝ := liters * 1000

/-- Calculates the volume of the third milk bottle -/
def thirdBottleVolume (bottle1 : MilkBottle) (bottle2 : MilkBottle) (totalVolume : ℝ) : ℝ :=
  litersToMilliliters totalVolume - (litersToMilliliters bottle1.volume + bottle2.volume)

/-- Theorem: The third milk bottle contains 250 milliliters -/
theorem third_bottle_volume_is_250ml 
  (bottle1 : MilkBottle) 
  (bottle2 : MilkBottle) 
  (totalVolume : ℝ) :
  bottle1.volume = 2 ∧ 
  bottle1.unit = "liters" ∧
  bottle2.volume = 750 ∧ 
  bottle2.unit = "milliliters" ∧
  totalVolume = 3 →
  thirdBottleVolume bottle1 bottle2 totalVolume = 250 := by
  sorry

end NUMINAMATH_CALUDE_third_bottle_volume_is_250ml_l717_71701


namespace NUMINAMATH_CALUDE_sum_differences_theorem_l717_71799

def numeral1 : ℕ := 987348621829
def numeral2 : ℕ := 74693251

def local_value (digit : ℕ) (position : ℕ) : ℕ := digit * (10 ^ position)

def face_value (digit : ℕ) : ℕ := digit

def difference_local_face (digit : ℕ) (position : ℕ) : ℕ :=
  local_value digit position - face_value digit

theorem sum_differences_theorem : 
  let first_8_pos := 8
  let second_8_pos := 1
  let seven_pos := 7
  (difference_local_face 8 first_8_pos + difference_local_face 8 second_8_pos) * 
  difference_local_face 7 seven_pos = 55999994048000192 := by
sorry

end NUMINAMATH_CALUDE_sum_differences_theorem_l717_71799


namespace NUMINAMATH_CALUDE_matthew_egg_rolls_l717_71749

/-- The number of egg rolls eaten by each person -/
structure EggRolls where
  kimberly : ℕ
  alvin : ℕ
  patrick : ℕ
  matthew : ℕ

/-- The conditions of the egg roll problem -/
def EggRollConditions (e : EggRolls) : Prop :=
  e.kimberly = 5 ∧
  e.alvin = 2 * e.kimberly - 1 ∧
  e.patrick = e.alvin / 2 ∧
  e.matthew = 2 * e.patrick

theorem matthew_egg_rolls (e : EggRolls) (h : EggRollConditions e) : e.matthew = 8 := by
  sorry

#check matthew_egg_rolls

end NUMINAMATH_CALUDE_matthew_egg_rolls_l717_71749


namespace NUMINAMATH_CALUDE_carbon_atoms_in_compound_l717_71775

/-- Represents the number of atoms of each element in a compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight oxygenWeight hydrogenWeight : ℕ) : ℕ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

/-- Theorem: A compound with 4 Hydrogen and 2 Oxygen atoms, and molecular weight 60,
    must have 2 Carbon atoms -/
theorem carbon_atoms_in_compound (c : Compound) 
    (h1 : c.hydrogen = 4)
    (h2 : c.oxygen = 2)
    (h3 : molecularWeight c 12 16 1 = 60) :
    c.carbon = 2 := by
  sorry

end NUMINAMATH_CALUDE_carbon_atoms_in_compound_l717_71775


namespace NUMINAMATH_CALUDE_snooker_ticket_difference_l717_71786

/-- Represents the ticket sales for a snooker tournament --/
structure TicketSales where
  vipPrice : ℕ
  regularPrice : ℕ
  totalTickets : ℕ
  totalRevenue : ℕ

/-- Calculates the difference between regular and VIP tickets sold --/
def ticketDifference (sales : TicketSales) : ℕ :=
  let vipTickets := (sales.totalRevenue - sales.regularPrice * sales.totalTickets) / 
                    (sales.vipPrice - sales.regularPrice)
  let regularTickets := sales.totalTickets - vipTickets
  regularTickets - vipTickets

/-- Theorem stating the difference in ticket sales --/
theorem snooker_ticket_difference :
  let sales : TicketSales := {
    vipPrice := 45,
    regularPrice := 20,
    totalTickets := 320,
    totalRevenue := 7500
  }
  ticketDifference sales = 232 := by
  sorry


end NUMINAMATH_CALUDE_snooker_ticket_difference_l717_71786


namespace NUMINAMATH_CALUDE_concert_attendance_theorem_l717_71773

/-- Represents the relationship between number of attendees and ticket price -/
structure ConcertAttendance where
  n : ℕ  -- number of attendees
  t : ℕ  -- ticket price in dollars
  k : ℕ  -- constant of proportionality
  h : n * t = k  -- inverse proportionality relationship

/-- Given initial conditions and final ticket price, calculates the final number of attendees -/
def calculate_attendance (initial : ConcertAttendance) (final_price : ℕ) : ℕ :=
  initial.k / final_price

theorem concert_attendance_theorem (initial : ConcertAttendance) 
    (h1 : initial.n = 300) 
    (h2 : initial.t = 50) 
    (h3 : calculate_attendance initial 75 = 200) : 
  calculate_attendance initial 75 = 200 := by
  sorry

#check concert_attendance_theorem

end NUMINAMATH_CALUDE_concert_attendance_theorem_l717_71773
