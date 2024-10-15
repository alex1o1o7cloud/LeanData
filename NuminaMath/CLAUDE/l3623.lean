import Mathlib

namespace NUMINAMATH_CALUDE_probability_selecting_two_types_l3623_362368

theorem probability_selecting_two_types (total : ℕ) (type_c : ℕ) (type_r : ℕ) (type_a : ℕ) :
  total = type_c + type_r + type_a →
  type_c = type_r →
  type_a = 1 →
  (type_c : ℚ) * type_r / (total * (total - 1)) = 5 / 11 :=
by sorry

end NUMINAMATH_CALUDE_probability_selecting_two_types_l3623_362368


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_9_l3623_362333

theorem arithmetic_square_root_of_9 : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_9_l3623_362333


namespace NUMINAMATH_CALUDE_no_common_root_l3623_362345

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ x₀ : ℝ, (x₀^2 + b*x₀ + c = 0) ∧ (x₀^2 + a*x₀ + d = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_root_l3623_362345


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l3623_362312

/-- Given distinct digits a, b, and c different from 1,
    prove that abb × c = bcb1 implies a = 5, b = 3, and c = 7 -/
theorem multiplication_puzzle :
  ∀ a b c : ℕ,
    a ≠ b → b ≠ c → a ≠ c →
    a ≠ 1 → b ≠ 1 → c ≠ 1 →
    a < 10 → b < 10 → c < 10 →
    (100 * a + 10 * b + b) * c = 1000 * b + 100 * c + 10 * b + 1 →
    a = 5 ∧ b = 3 ∧ c = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l3623_362312


namespace NUMINAMATH_CALUDE_soda_price_proof_l3623_362371

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := regular_price * 0.85

/-- The total price for 100 cans purchased in 24-can cases -/
def total_price : ℝ := 34

theorem soda_price_proof :
  discounted_price * 100 = total_price ∧ regular_price = 0.40 :=
sorry

end NUMINAMATH_CALUDE_soda_price_proof_l3623_362371


namespace NUMINAMATH_CALUDE_length_width_ratio_l3623_362319

/-- Represents a rectangular roof --/
structure RectangularRoof where
  length : ℝ
  width : ℝ

/-- Properties of the specific roof in the problem --/
def problem_roof : RectangularRoof → Prop
  | roof => roof.length * roof.width = 900 ∧ 
            roof.length - roof.width = 45

/-- The theorem stating the ratio of length to width --/
theorem length_width_ratio (roof : RectangularRoof) 
  (h : problem_roof roof) : 
  roof.length / roof.width = 4 := by
  sorry

#check length_width_ratio

end NUMINAMATH_CALUDE_length_width_ratio_l3623_362319


namespace NUMINAMATH_CALUDE_max_triangle_perimeter_l3623_362367

/-- Given a triangle with two sides of length 8 and 15 units, and the third side
    length x being an integer, the maximum perimeter of the triangle is 45 units. -/
theorem max_triangle_perimeter :
  ∀ x : ℤ,
  (8 : ℝ) + 15 > (x : ℝ) →
  (8 : ℝ) + (x : ℝ) > 15 →
  (15 : ℝ) + (x : ℝ) > 8 →
  (∀ y : ℤ, (8 : ℝ) + 15 > (y : ℝ) →
             (8 : ℝ) + (y : ℝ) > 15 →
             (15 : ℝ) + (y : ℝ) > 8 →
             8 + 15 + (x : ℝ) ≥ 8 + 15 + (y : ℝ)) →
  8 + 15 + (x : ℝ) = 45 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_perimeter_l3623_362367


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3623_362362

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem condition_sufficient_not_necessary (a : ℕ → ℝ) :
  (∀ n, a (n + 1) > |a n|) → is_increasing a ∧
  ¬(is_increasing a → ∀ n, a (n + 1) > |a n|) :=
by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3623_362362


namespace NUMINAMATH_CALUDE_inequality_proof_l3623_362339

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2) / (2 * a^5 * b^5) + 81 * a^2 * b^2 / 4 + 9 * a * b > 18 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3623_362339


namespace NUMINAMATH_CALUDE_mack_journal_pages_l3623_362305

/-- The number of pages Mack writes on Monday -/
def monday_pages : ℕ := 60 / 30

/-- The number of pages Mack writes on Tuesday -/
def tuesday_pages : ℕ := 45 / 15

/-- The number of pages Mack writes on Wednesday -/
def wednesday_pages : ℕ := 5

/-- The total number of pages Mack writes from Monday to Wednesday -/
def total_pages : ℕ := monday_pages + tuesday_pages + wednesday_pages

theorem mack_journal_pages : total_pages = 10 := by sorry

end NUMINAMATH_CALUDE_mack_journal_pages_l3623_362305


namespace NUMINAMATH_CALUDE_typists_productivity_l3623_362335

/-- Given that 10 typists can type 20 letters in 20 minutes, 
    prove that 40 typists working at the same rate for 1 hour will complete 240 letters. -/
theorem typists_productivity 
  (base_typists : ℕ) 
  (base_letters : ℕ) 
  (base_minutes : ℕ) 
  (new_typists : ℕ) 
  (new_minutes : ℕ)
  (h1 : base_typists = 10)
  (h2 : base_letters = 20)
  (h3 : base_minutes = 20)
  (h4 : new_typists = 40)
  (h5 : new_minutes = 60) :
  (new_typists * new_minutes * base_letters) / (base_typists * base_minutes) = 240 :=
sorry

end NUMINAMATH_CALUDE_typists_productivity_l3623_362335


namespace NUMINAMATH_CALUDE_remainder_fifty_pow_2019_plus_one_mod_seven_l3623_362372

theorem remainder_fifty_pow_2019_plus_one_mod_seven (n : ℕ) : (50^2019 + 1) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_fifty_pow_2019_plus_one_mod_seven_l3623_362372


namespace NUMINAMATH_CALUDE_aquarium_count_l3623_362357

theorem aquarium_count (total_animals : ℕ) (animals_per_aquarium : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : animals_per_aquarium = 2) :
  total_animals / animals_per_aquarium = 20 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_count_l3623_362357


namespace NUMINAMATH_CALUDE_equation_solutions_l3623_362373

theorem equation_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 6*x + 3 = 0
  let eq2 : ℝ → Prop := λ x ↦ 2*x*(x-1) = 3-3*x
  let sol1 : Set ℝ := {3 + Real.sqrt 6, 3 - Real.sqrt 6}
  let sol2 : Set ℝ := {1, -3/2}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y, eq1 y → y ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y, eq2 y → y ∈ sol2) :=
by sorry


end NUMINAMATH_CALUDE_equation_solutions_l3623_362373


namespace NUMINAMATH_CALUDE_days_missed_difference_l3623_362370

/-- Represents the frequency histogram of days missed --/
structure FrequencyHistogram :=
  (days : List Nat)
  (frequencies : List Nat)
  (total_students : Nat)

/-- Calculate the median of the dataset --/
def median (h : FrequencyHistogram) : Rat :=
  sorry

/-- Calculate the mean of the dataset --/
def mean (h : FrequencyHistogram) : Rat :=
  sorry

/-- The main theorem --/
theorem days_missed_difference (h : FrequencyHistogram) 
  (h_days : h.days = [0, 1, 2, 3, 4, 5])
  (h_frequencies : h.frequencies = [4, 3, 6, 2, 3, 2])
  (h_total : h.total_students = 20) :
  mean h - median h = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_days_missed_difference_l3623_362370


namespace NUMINAMATH_CALUDE_pie_shop_revenue_calculation_l3623_362358

/-- Represents the revenue calculation for a pie shop --/
def pie_shop_revenue (apple_price blueberry_price cherry_price : ℕ) 
                     (slices_per_pie : ℕ) 
                     (apple_pies blueberry_pies cherry_pies : ℕ) : ℕ :=
  (apple_price * slices_per_pie * apple_pies) + 
  (blueberry_price * slices_per_pie * blueberry_pies) + 
  (cherry_price * slices_per_pie * cherry_pies)

/-- Theorem stating the revenue of the pie shop --/
theorem pie_shop_revenue_calculation : 
  pie_shop_revenue 5 6 7 6 12 8 10 = 1068 := by
  sorry

end NUMINAMATH_CALUDE_pie_shop_revenue_calculation_l3623_362358


namespace NUMINAMATH_CALUDE_star_polygon_points_l3623_362375

/-- A regular star polygon with ℓ points -/
structure StarPolygon where
  ℓ : ℕ
  x_angle : Real
  y_angle : Real
  h_x_less_y : x_angle = y_angle - 15
  h_external_sum : ℓ * (x_angle + y_angle) = 360
  h_internal_sum : ℓ * (180 - x_angle - y_angle) = 2 * 360

/-- Theorem: The number of points in the star polygon is 24 -/
theorem star_polygon_points (s : StarPolygon) : s.ℓ = 24 := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_points_l3623_362375


namespace NUMINAMATH_CALUDE_petya_win_probability_l3623_362359

/-- The "Heap of Stones" game -/
structure HeapOfStones where
  initialStones : Nat
  maxTake : Nat
  minTake : Nat

/-- A player in the game -/
inductive Player
  | Petya
  | Computer

/-- The strategy used by a player -/
inductive Strategy
  | Random
  | Optimal

/-- The result of the game -/
inductive GameResult
  | PetyaWins
  | ComputerWins

/-- The probability of Petya winning the game -/
def winProbability (game : HeapOfStones) (firstPlayer : Player) 
    (petyaStrategy : Strategy) (computerStrategy : Strategy) : ℚ :=
  sorry

/-- The theorem to prove -/
theorem petya_win_probability :
  let game : HeapOfStones := ⟨16, 4, 1⟩
  winProbability game Player.Petya Strategy.Random Strategy.Optimal = 1 / 256 :=
sorry

end NUMINAMATH_CALUDE_petya_win_probability_l3623_362359


namespace NUMINAMATH_CALUDE_root_sum_ratio_l3623_362322

theorem root_sum_ratio (m₁ m₂ : ℝ) (a b : ℝ → ℝ) : 
  (∀ m, m * (a m)^2 - (3 * m - 2) * (a m) + 7 = 0) →
  (∀ m, m * (b m)^2 - (3 * m - 2) * (b m) + 7 = 0) →
  (a m₁ / b m₁ + b m₁ / a m₁ = 2) →
  (a m₂ / b m₂ + b m₂ / a m₂ = 2) →
  m₁ / m₂ + m₂ / m₁ = 194 / 9 := by
  sorry


end NUMINAMATH_CALUDE_root_sum_ratio_l3623_362322


namespace NUMINAMATH_CALUDE_smallest_representable_number_l3623_362307

/-- Sum of decimal digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number can be represented as the sum of k positive integers
    with the same sum of decimal digits -/
def representable (n k : ℕ) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ n = k * d ∧ sum_of_digits d = sum_of_digits n / k

theorem smallest_representable_number :
  (∀ m : ℕ, m < 10010 → ¬(representable m 2002 ∧ representable m 2003)) ∧
  (representable 10010 2002 ∧ representable 10010 2003) := by sorry

end NUMINAMATH_CALUDE_smallest_representable_number_l3623_362307


namespace NUMINAMATH_CALUDE_fourth_to_third_grade_ratio_l3623_362366

/-- Given the number of students in each grade, prove the ratio of 4th to 3rd grade students -/
theorem fourth_to_third_grade_ratio 
  (third_grade : ℕ) 
  (second_grade : ℕ) 
  (total_students : ℕ) 
  (h1 : third_grade = 19) 
  (h2 : second_grade = 29) 
  (h3 : total_students = 86) :
  (total_students - second_grade - third_grade) / third_grade = 2 := by
sorry

end NUMINAMATH_CALUDE_fourth_to_third_grade_ratio_l3623_362366


namespace NUMINAMATH_CALUDE_sin_cos_sum_ratio_equals_tan_60_l3623_362337

theorem sin_cos_sum_ratio_equals_tan_60 :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_ratio_equals_tan_60_l3623_362337


namespace NUMINAMATH_CALUDE_monotonically_decreasing_cubic_l3623_362382

/-- A function f is monotonically decreasing on an interval (a, b) if for all x, y in (a, b),
    x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem monotonically_decreasing_cubic (a d : ℝ) :
  MonotonicallyDecreasing (fun x => x^3 - a*x^2 + 4*d) 0 2 → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_cubic_l3623_362382


namespace NUMINAMATH_CALUDE_simplify_expression_l3623_362316

theorem simplify_expression (y : ℝ) : 3*y + 9*y^2 + 15 - (6 - 3*y - 9*y^2) = 18*y^2 + 6*y + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3623_362316


namespace NUMINAMATH_CALUDE_percent_relation_l3623_362310

theorem percent_relation (x y z w v : ℝ) 
  (hx : x = 1.3 * y) 
  (hy : y = 0.6 * z) 
  (hw : w = 1.25 * x) 
  (hv : v = 0.85 * w) : 
  v = 0.82875 * z := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l3623_362310


namespace NUMINAMATH_CALUDE_zoo_treats_problem_l3623_362329

/-- The percentage of pieces of bread Jane brings compared to treats -/
def jane_bread_percentage (jane_treats : ℕ) (jane_bread : ℕ) (wanda_treats : ℕ) (wanda_bread : ℕ) : ℚ :=
  (jane_bread : ℚ) / (jane_treats : ℚ) * 100

/-- The problem statement -/
theorem zoo_treats_problem (jane_treats : ℕ) (jane_bread : ℕ) (wanda_treats : ℕ) (wanda_bread : ℕ) :
  wanda_treats = jane_treats / 2 →
  wanda_bread = 3 * wanda_treats →
  wanda_bread = 90 →
  jane_treats + jane_bread + wanda_treats + wanda_bread = 225 →
  jane_bread_percentage jane_treats jane_bread wanda_treats wanda_bread = 75 := by
  sorry

end NUMINAMATH_CALUDE_zoo_treats_problem_l3623_362329


namespace NUMINAMATH_CALUDE_cistern_filling_problem_l3623_362351

/-- The time taken for pipe A to fill the cistern -/
def time_A : ℝ := 16

/-- The time taken for pipe B to empty the cistern -/
def time_B : ℝ := 20

/-- The time taken to fill the cistern when both pipes are open -/
def time_both : ℝ := 80

/-- Theorem stating that the given times satisfy the cistern filling problem -/
theorem cistern_filling_problem :
  1 / time_A - 1 / time_B = 1 / time_both := by sorry

end NUMINAMATH_CALUDE_cistern_filling_problem_l3623_362351


namespace NUMINAMATH_CALUDE_mass_equivalence_l3623_362348

-- Define symbols as real numbers representing their masses
variable (circle square triangle zero : ℝ)

-- Define the balanced scales conditions
axiom scale1 : 3 * circle = 2 * triangle
axiom scale2 : square + circle + triangle = 2 * square

-- Define the mass of the left side of the equation to prove
def left_side : ℝ := circle + 3 * triangle

-- Define the mass of the right side of the equation to prove
def right_side : ℝ := 3 * zero + square

-- Theorem to prove
theorem mass_equivalence : left_side = right_side :=
sorry

end NUMINAMATH_CALUDE_mass_equivalence_l3623_362348


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3623_362327

/-- The first term of the geometric series -/
def a₁ : ℚ := 4/3

/-- The second term of the geometric series -/
def a₂ : ℚ := 16/9

/-- The third term of the geometric series -/
def a₃ : ℚ := 64/27

/-- The common ratio of the geometric series -/
def r : ℚ := 4/3

theorem geometric_series_common_ratio :
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3623_362327


namespace NUMINAMATH_CALUDE_soccer_team_starters_l3623_362313

theorem soccer_team_starters (total_players : ℕ) (first_half_subs : ℕ) (players_not_played : ℕ) :
  total_players = 24 →
  first_half_subs = 2 →
  players_not_played = 7 →
  ∃ (starters : ℕ), starters = 11 ∧ 
    starters + first_half_subs + 2 * first_half_subs + players_not_played = total_players :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l3623_362313


namespace NUMINAMATH_CALUDE_b_value_proof_l3623_362306

theorem b_value_proof (a b c m : ℝ) (h : m = (c * a * b) / (a - b)) : 
  b = (m * a) / (m + c * a) := by
  sorry

end NUMINAMATH_CALUDE_b_value_proof_l3623_362306


namespace NUMINAMATH_CALUDE_distribute_5_4_l3623_362346

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 1024 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_4 : distribute 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_4_l3623_362346


namespace NUMINAMATH_CALUDE_expression_evaluation_l3623_362320

/-- Given real numbers x, y, and z, prove that the expression
    ((P+Q)/(P-Q) - (P-Q)/(P+Q)) equals (x^2 - y^2 - 2yz - z^2) / (xy + xz),
    where P = x + y + z and Q = x - y - z. -/
theorem expression_evaluation (x y z : ℝ) :
  let P := x + y + z
  let Q := x - y - z
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (x^2 - y^2 - 2*y*z - z^2) / (x*y + x*z) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3623_362320


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l3623_362343

/-- The weight of one kayak in pounds -/
def kayak_weight : ℝ := 35

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 28

theorem bowling_ball_weight_proof :
  (5 * bowling_ball_weight = 4 * kayak_weight) →
  bowling_ball_weight = 28 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l3623_362343


namespace NUMINAMATH_CALUDE_jack_classic_authors_l3623_362393

/-- The number of books each classic author has in Jack's collection -/
def books_per_author : ℕ := 33

/-- The total number of books in Jack's classics section -/
def total_books : ℕ := 198

/-- The number of classic authors in Jack's collection -/
def num_authors : ℕ := total_books / books_per_author

theorem jack_classic_authors :
  num_authors = 6 :=
by sorry

end NUMINAMATH_CALUDE_jack_classic_authors_l3623_362393


namespace NUMINAMATH_CALUDE_unit_digit_of_7_to_500_l3623_362363

theorem unit_digit_of_7_to_500 : 7^500 % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_7_to_500_l3623_362363


namespace NUMINAMATH_CALUDE_max_product_of_functions_l3623_362356

/-- Given two real-valued functions f and g with specified ranges,
    prove that the maximum value of their product is 14 -/
theorem max_product_of_functions (f g : ℝ → ℝ)
  (hf : Set.range f = Set.Icc (-7) 4)
  (hg : Set.range g = Set.Icc 0 2) :
  ∃ x : ℝ, f x * g x = 14 ∧ ∀ y : ℝ, f y * g y ≤ 14 := by
  sorry


end NUMINAMATH_CALUDE_max_product_of_functions_l3623_362356


namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_l3623_362396

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x ≤ y → f a x ≤ f a y) →
  a ≥ -1 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_l3623_362396


namespace NUMINAMATH_CALUDE_second_player_wins_l3623_362384

/-- Represents a position on the chessboard --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a move in the game --/
inductive Move
  | Up : Nat → Move
  | Left : Nat → Move

/-- Applies a move to a position --/
def applyMove (pos : Position) (move : Move) : Position :=
  match move with
  | Move.Up n => ⟨pos.x, pos.y + n⟩
  | Move.Left n => ⟨pos.x - n, pos.y⟩

/-- Checks if a position is valid on the 8x8 board --/
def isValidPosition (pos : Position) : Prop :=
  1 ≤ pos.x ∧ pos.x ≤ 8 ∧ 1 ≤ pos.y ∧ pos.y ≤ 8

/-- Checks if a move is valid from a given position --/
def isValidMove (pos : Position) (move : Move) : Prop :=
  isValidPosition (applyMove pos move)

/-- Represents the game state --/
structure GameState :=
  (position : Position)
  (currentPlayer : Bool)  -- True for first player, False for second player

/-- The winning strategy for the second player --/
def secondPlayerWinningStrategy : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (initialState : GameState),
      initialState.position = ⟨1, 1⟩ →
      initialState.currentPlayer = true →
      ∀ (game : ℕ → GameState),
        game 0 = initialState →
        (∀ n : ℕ, 
          (game (n+1)).position = 
            if (game n).currentPlayer
            then applyMove (game n).position (strategy (game n))
            else applyMove (game n).position (strategy (game n))) →
        ∃ (n : ℕ), ¬isValidMove (game n).position (strategy (game n))

theorem second_player_wins : secondPlayerWinningStrategy :=
  sorry

end NUMINAMATH_CALUDE_second_player_wins_l3623_362384


namespace NUMINAMATH_CALUDE_plane_perpendicularity_l3623_362394

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularP : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (m n : Line) (α β : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) 
  (h3 : parallel m n) 
  (h4 : parallelLP m α) 
  (h5 : perpendicular n β) : 
  perpendicularP α β := by sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_l3623_362394


namespace NUMINAMATH_CALUDE_x_investment_value_l3623_362350

/-- Represents the investment and profit scenario of a business partnership --/
structure BusinessPartnership where
  x_investment : ℕ  -- X's investment
  y_investment : ℕ  -- Y's investment
  z_investment : ℕ  -- Z's investment
  total_profit : ℕ  -- Total profit
  z_profit : ℕ      -- Z's share of the profit
  x_months : ℕ      -- Months X and Y were in business before Z joined
  z_months : ℕ      -- Months Z was in business

/-- The main theorem stating that X's investment was 35700 given the conditions --/
theorem x_investment_value (bp : BusinessPartnership) : 
  bp.y_investment = 42000 ∧ 
  bp.z_investment = 48000 ∧ 
  bp.total_profit = 14300 ∧ 
  bp.z_profit = 4160 ∧
  bp.x_months = 12 ∧
  bp.z_months = 8 →
  bp.x_investment = 35700 := by
  sorry

#check x_investment_value

end NUMINAMATH_CALUDE_x_investment_value_l3623_362350


namespace NUMINAMATH_CALUDE_group_four_frequency_and_relative_frequency_l3623_362330

/-- Given a sample with capacity 50 and frequencies for groups 1, 2, 3, and 5,
    prove the frequency and relative frequency of group 4 -/
theorem group_four_frequency_and_relative_frequency 
  (total_capacity : ℕ) 
  (freq_1 freq_2 freq_3 freq_5 : ℕ) 
  (h1 : total_capacity = 50)
  (h2 : freq_1 = 8)
  (h3 : freq_2 = 11)
  (h4 : freq_3 = 10)
  (h5 : freq_5 = 9) :
  ∃ (freq_4 : ℕ) (rel_freq_4 : ℚ),
    freq_4 = total_capacity - (freq_1 + freq_2 + freq_3 + freq_5) ∧
    rel_freq_4 = freq_4 / total_capacity ∧
    freq_4 = 12 ∧
    rel_freq_4 = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_group_four_frequency_and_relative_frequency_l3623_362330


namespace NUMINAMATH_CALUDE_fraction_equality_l3623_362369

theorem fraction_equality (a b c d : ℝ) (h : a / b = c / d) :
  (a * b) / (c * d) = ((a + b) / (c + d))^2 ∧ (a * b) / (c * d) = ((a - b) / (c - d))^2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3623_362369


namespace NUMINAMATH_CALUDE_jenny_research_time_l3623_362315

/-- Represents the time allocation for Jenny's school project -/
structure ProjectTime where
  total : ℕ
  proposal : ℕ
  report : ℕ

/-- Calculates the time spent on research given the project time allocation -/
def researchTime (pt : ProjectTime) : ℕ :=
  pt.total - pt.proposal - pt.report

/-- Theorem stating that Jenny spent 10 hours on research -/
theorem jenny_research_time :
  ∀ (pt : ProjectTime),
  pt.total = 20 ∧ pt.proposal = 2 ∧ pt.report = 8 →
  researchTime pt = 10 := by
  sorry

end NUMINAMATH_CALUDE_jenny_research_time_l3623_362315


namespace NUMINAMATH_CALUDE_stratified_sampling_l3623_362374

/-- Stratified sampling problem -/
theorem stratified_sampling 
  (total_items : ℕ) 
  (sample_size : ℕ) 
  (stratum_A_size : ℕ) 
  (h1 : total_items = 600) 
  (h2 : sample_size = 100) 
  (h3 : stratum_A_size = 150) :
  let items_from_A := (sample_size * stratum_A_size) / total_items
  let prob_item_A := sample_size / total_items
  (items_from_A = 25) ∧ (prob_item_A = 1 / 6) := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_l3623_362374


namespace NUMINAMATH_CALUDE_hexagon_rearrangement_l3623_362326

/-- Represents a rectangle with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a square with given side length -/
structure Square where
  side : ℝ

/-- Represents the problem setup -/
structure HexagonProblem where
  original_rectangle : Rectangle
  resulting_square : Square
  is_valid : Prop

/-- The theorem stating the relationship between the original rectangle and the resulting square -/
theorem hexagon_rearrangement (p : HexagonProblem) 
  (h1 : p.original_rectangle.length = 9)
  (h2 : p.original_rectangle.width = 16)
  (h3 : p.is_valid)
  (h4 : p.original_rectangle.length * p.original_rectangle.width = p.resulting_square.side ^ 2) :
  p.resulting_square.side / 2 = 6 := by sorry

end NUMINAMATH_CALUDE_hexagon_rearrangement_l3623_362326


namespace NUMINAMATH_CALUDE_jacob_winning_strategy_l3623_362398

/-- Represents the game board --/
structure Board :=
  (m : ℕ) -- number of rows
  (n : ℕ) -- number of columns

/-- Represents a position on the board --/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Defines a valid move on the board --/
def ValidMove (b : Board) (start finish : Position) : Prop :=
  (finish.row ≥ start.row ∧ finish.col = start.col) ∨
  (finish.col ≥ start.col ∧ finish.row = start.row)

/-- Defines the winning position --/
def IsWinningPosition (b : Board) (p : Position) : Prop :=
  p.row = b.m ∧ p.col = b.n

/-- Jacob's winning strategy exists --/
def JacobHasWinningStrategy (b : Board) : Prop :=
  ∃ (strategy : Position → Position),
    ∀ (p : Position),
      ValidMove b p (strategy p) ∧
      (∀ (q : Position), ValidMove b (strategy p) q →
        (IsWinningPosition b q ∨ ∃ (r : Position), ValidMove b q r ∧ IsWinningPosition b (strategy r)))

/-- The main theorem: Jacob has a winning strategy iff m ≠ n --/
theorem jacob_winning_strategy (b : Board) :
  JacobHasWinningStrategy b ↔ b.m ≠ b.n :=
sorry

end NUMINAMATH_CALUDE_jacob_winning_strategy_l3623_362398


namespace NUMINAMATH_CALUDE_number_difference_proof_l3623_362383

theorem number_difference_proof (x : ℚ) : x - (3/5) * x = 50 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l3623_362383


namespace NUMINAMATH_CALUDE_max_b_over_a_l3623_362352

theorem max_b_over_a (a b : ℝ) (h_a : a > 0) : 
  (∀ x : ℝ, a * Real.exp x ≥ 2 * x + b) → b / a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_b_over_a_l3623_362352


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3623_362304

theorem sum_of_numbers : (6 / 5 : ℚ) + (1 / 10 : ℚ) + (156 / 100 : ℚ) = 286 / 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3623_362304


namespace NUMINAMATH_CALUDE_find_A_l3623_362323

theorem find_A : ∃ A : ℕ, A = 38 ∧ A / 7 = 5 ∧ A % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l3623_362323


namespace NUMINAMATH_CALUDE_chord_length_problem_l3623_362385

/-- The chord length cut by a line on a circle -/
def chord_length (circle_center : ℝ × ℝ) (circle_radius : ℝ) (line : ℝ → ℝ → ℝ) : ℝ :=
  2 * circle_radius

/-- The problem statement -/
theorem chord_length_problem :
  let circle_center := (3, 0)
  let circle_radius := 3
  let line := fun x y => 3 * x - 4 * y - 9
  chord_length circle_center circle_radius line = 6 := by
  sorry


end NUMINAMATH_CALUDE_chord_length_problem_l3623_362385


namespace NUMINAMATH_CALUDE_full_house_prob_modified_deck_l3623_362303

/-- Represents a modified deck of cards -/
structure ModifiedDeck :=
  (ranks : Nat)
  (cards_per_rank : Nat)
  (hand_size : Nat)

/-- Calculate the number of ways to choose k items from n items -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculate the probability of drawing a full house -/
def full_house_probability (deck : ModifiedDeck) : Rat :=
  let total_cards := deck.ranks * deck.cards_per_rank
  let total_combinations := choose total_cards deck.hand_size
  let full_house_combinations := 
    deck.ranks * choose deck.cards_per_rank 3 * (deck.ranks - 1) * choose deck.cards_per_rank 2
  full_house_combinations / total_combinations

/-- Theorem: The probability of drawing a full house in the given modified deck is 40/1292 -/
theorem full_house_prob_modified_deck :
  full_house_probability ⟨5, 4, 5⟩ = 40 / 1292 := by
  sorry


end NUMINAMATH_CALUDE_full_house_prob_modified_deck_l3623_362303


namespace NUMINAMATH_CALUDE_average_after_removing_numbers_l3623_362318

theorem average_after_removing_numbers (n : ℕ) (initial_avg : ℚ) (removed1 removed2 : ℚ) :
  n = 50 →
  initial_avg = 38 →
  removed1 = 45 →
  removed2 = 55 →
  (n : ℚ) * initial_avg - (removed1 + removed2) = ((n - 2) : ℚ) * 37.5 :=
by sorry

end NUMINAMATH_CALUDE_average_after_removing_numbers_l3623_362318


namespace NUMINAMATH_CALUDE_sandwich_problem_solution_l3623_362347

/-- Represents the sandwich shop problem -/
def sandwich_problem (sandwich_price : ℝ) (delivery_fee : ℝ) (tip_percentage : ℝ) (total_received : ℝ) : Prop :=
  ∃ (num_sandwiches : ℝ),
    sandwich_price * num_sandwiches + delivery_fee + 
    (sandwich_price * num_sandwiches + delivery_fee) * tip_percentage = total_received ∧
    num_sandwiches = 18

/-- Theorem stating the solution to the sandwich problem -/
theorem sandwich_problem_solution :
  sandwich_problem 5 20 0.1 121 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_problem_solution_l3623_362347


namespace NUMINAMATH_CALUDE_candy_distribution_l3623_362361

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 15 →
  num_bags = 5 →
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 3 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l3623_362361


namespace NUMINAMATH_CALUDE_adult_elephant_weekly_bananas_eq_630_l3623_362389

/-- The number of bananas an adult elephant eats per day -/
def adult_elephant_daily_bananas : ℕ := 90

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of bananas an adult elephant eats in a week -/
def adult_elephant_weekly_bananas : ℕ := adult_elephant_daily_bananas * days_in_week

theorem adult_elephant_weekly_bananas_eq_630 :
  adult_elephant_weekly_bananas = 630 := by
  sorry

end NUMINAMATH_CALUDE_adult_elephant_weekly_bananas_eq_630_l3623_362389


namespace NUMINAMATH_CALUDE_largest_fraction_l3623_362399

theorem largest_fraction : (5 : ℚ) / 6 > 3 / 4 ∧ (5 : ℚ) / 6 > 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l3623_362399


namespace NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_90_l3623_362314

theorem smallest_of_three_consecutive_sum_90 (x y z : ℤ) :
  y = x + 1 ∧ z = y + 1 ∧ x + y + z = 90 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_90_l3623_362314


namespace NUMINAMATH_CALUDE_triangle_equal_area_division_l3623_362387

theorem triangle_equal_area_division :
  let triangle := [(0, 0), (1, 1), (9, 1)]
  let total_area := 4
  let dividing_line := 3
  let left_area := (1/2) * dividing_line * (dividing_line/9)
  let right_area := (1/2) * (1 - dividing_line/9) * (9 - dividing_line)
  left_area = right_area ∧ left_area = total_area/2 := by sorry

end NUMINAMATH_CALUDE_triangle_equal_area_division_l3623_362387


namespace NUMINAMATH_CALUDE_oranges_remaining_l3623_362390

def initial_oranges : ℕ := 60
def percentage_taken : ℚ := 45 / 100

theorem oranges_remaining : 
  initial_oranges - (percentage_taken * initial_oranges).floor = 33 := by
  sorry

end NUMINAMATH_CALUDE_oranges_remaining_l3623_362390


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3623_362317

theorem solution_set_inequality (x : ℝ) :
  (1/2 - x) * (x - 1/3) > 0 ↔ 1/3 < x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3623_362317


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l3623_362300

theorem parallel_lines_b_value (b : ℝ) : 
  (∀ x y : ℝ, 4 * y - 3 * x - 2 = 0 ↔ y = (3/4) * x + 1/2) →
  (∀ x y : ℝ, 6 * y + b * x + 1 = 0 ↔ y = (-b/6) * x - 1/6) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (4 * y₁ - 3 * x₁ - 2 = 0 ∧ 6 * y₂ + b * x₂ + 1 = 0) → 
    (y₂ - y₁) / (x₂ - x₁) = (3/4)) →
  b = -4.5 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l3623_362300


namespace NUMINAMATH_CALUDE_binomial_divisibility_implies_prime_l3623_362338

theorem binomial_divisibility_implies_prime (n : ℕ) (h : ∀ k : ℕ, 1 ≤ k → k < n → (n.choose k) % n = 0) : Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_implies_prime_l3623_362338


namespace NUMINAMATH_CALUDE_subset_implies_max_a_max_a_is_negative_three_l3623_362377

theorem subset_implies_max_a (a : ℝ) : 
  let A : Set ℝ := {x | |x| ≥ 3}
  let B : Set ℝ := {x | x ≥ a}
  A ⊆ B → a ≤ -3 :=
by
  sorry

theorem max_a_is_negative_three :
  ∃ c, c = -3 ∧ 
  (∀ a : ℝ, (let A : Set ℝ := {x | |x| ≥ 3}; let B : Set ℝ := {x | x ≥ a}; A ⊆ B) → a ≤ c) ∧
  (let A : Set ℝ := {x | |x| ≥ 3}; let B : Set ℝ := {x | x ≥ c}; A ⊆ B) :=
by
  sorry

end NUMINAMATH_CALUDE_subset_implies_max_a_max_a_is_negative_three_l3623_362377


namespace NUMINAMATH_CALUDE_pamphlet_cost_is_correct_l3623_362380

/-- The cost of one pamphlet in dollars -/
def pamphlet_cost : ℝ := 1.11

/-- Condition 1: Nine copies cost less than $10.00 -/
axiom condition1 : 9 * pamphlet_cost < 10

/-- Condition 2: Ten copies cost more than $11.00 -/
axiom condition2 : 10 * pamphlet_cost > 11

/-- Theorem: The cost of one pamphlet is $1.11 -/
theorem pamphlet_cost_is_correct : pamphlet_cost = 1.11 := by
  sorry


end NUMINAMATH_CALUDE_pamphlet_cost_is_correct_l3623_362380


namespace NUMINAMATH_CALUDE_direct_proportion_problem_l3623_362355

/-- A direct proportion function -/
def DirectProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

theorem direct_proportion_problem (f : ℝ → ℝ) 
  (h1 : DirectProportion f) 
  (h2 : f (-2) = 4) : 
  f 3 = -6 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_problem_l3623_362355


namespace NUMINAMATH_CALUDE_pine_cone_weight_l3623_362376

/-- The weight of each pine cone given the conditions in Alan's backyard scenario -/
theorem pine_cone_weight (
  trees : ℕ)
  (cones_per_tree : ℕ)
  (roof_percentage : ℚ)
  (total_roof_weight : ℕ)
  (h1 : trees = 8)
  (h2 : cones_per_tree = 200)
  (h3 : roof_percentage = 3/10)
  (h4 : total_roof_weight = 1920)
  : (total_roof_weight : ℚ) / ((trees * cones_per_tree : ℕ) * roof_percentage) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pine_cone_weight_l3623_362376


namespace NUMINAMATH_CALUDE_trig_sum_problem_l3623_362332

theorem trig_sum_problem (α : Real) (h1 : 0 < α) (h2 : α < Real.pi) 
  (h3 : Real.sin α * Real.cos α = -1/2) : 
  1/(1 + Real.sin α) + 1/(1 + Real.cos α) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trig_sum_problem_l3623_362332


namespace NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l3623_362336

/-- Calculate the amount John paid out of pocket for his new computer setup --/
theorem johns_out_of_pocket_expense :
  let computer_cost : ℚ := 1200
  let computer_discount : ℚ := 0.15
  let chair_cost : ℚ := 300
  let chair_discount : ℚ := 0.10
  let accessories_cost : ℚ := 350
  let sales_tax_rate : ℚ := 0.08
  let playstation_value : ℚ := 500
  let playstation_discount : ℚ := 0.30
  let bicycle_sale : ℚ := 100

  let discounted_computer := computer_cost * (1 - computer_discount)
  let discounted_chair := chair_cost * (1 - chair_discount)
  let total_before_tax := discounted_computer + discounted_chair + accessories_cost
  let total_with_tax := total_before_tax * (1 + sales_tax_rate)
  let sold_items := playstation_value * (1 - playstation_discount) + bicycle_sale
  let out_of_pocket := total_with_tax - sold_items

  out_of_pocket = 1321.20
  := by sorry

end NUMINAMATH_CALUDE_johns_out_of_pocket_expense_l3623_362336


namespace NUMINAMATH_CALUDE_equation_solution_l3623_362331

def solution_set : Set (ℤ × ℤ) :=
  {(1, 12), (1, -12), (-9, 12), (-9, -12), (-4, 12), (-4, -12), (0, 0), (-8, 0), (-1, 0), (-7, 0)}

def satisfies_equation (p : ℤ × ℤ) : Prop :=
  let x := p.1
  let y := p.2
  x * (x + 1) * (x + 7) * (x + 8) = y^2

theorem equation_solution :
  ∀ p : ℤ × ℤ, satisfies_equation p ↔ p ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3623_362331


namespace NUMINAMATH_CALUDE_birch_planting_l3623_362386

theorem birch_planting (total_students : ℕ) (roses_per_girl : ℕ) (total_plants : ℕ) (total_birches : ℕ)
  (h1 : total_students = 24)
  (h2 : roses_per_girl = 3)
  (h3 : total_plants = 24)
  (h4 : total_birches = 6) :
  (total_students - (total_plants - total_birches) / roses_per_girl) / 3 = total_birches :=
by sorry

end NUMINAMATH_CALUDE_birch_planting_l3623_362386


namespace NUMINAMATH_CALUDE_diameter_endpoint_coordinates_l3623_362397

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a point in 2D space --/
def Point := ℝ × ℝ

/-- Checks if two points are endpoints of a diameter in a circle --/
def are_diameter_endpoints (c : Circle) (p1 p2 : Point) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint = c.center

theorem diameter_endpoint_coordinates (c : Circle) (p1 p2 : Point) :
  c.center = (3, 4) →
  p1 = (1, -2) →
  are_diameter_endpoints c p1 p2 →
  p2 = (5, 10) := by
  sorry

end NUMINAMATH_CALUDE_diameter_endpoint_coordinates_l3623_362397


namespace NUMINAMATH_CALUDE_adam_ate_three_more_than_bill_l3623_362301

-- Define the number of pies eaten by each person
def sierra_pies : ℕ := 12
def total_pies : ℕ := 27

-- Define the relationships between the number of pies eaten
def bill_pies : ℕ := sierra_pies / 2
def adam_pies : ℕ := total_pies - sierra_pies - bill_pies

-- Theorem to prove
theorem adam_ate_three_more_than_bill :
  adam_pies = bill_pies + 3 := by
  sorry

end NUMINAMATH_CALUDE_adam_ate_three_more_than_bill_l3623_362301


namespace NUMINAMATH_CALUDE_ab_length_in_two_isosceles_triangles_l3623_362354

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.base + 2 * t.leg

theorem ab_length_in_two_isosceles_triangles 
  (abc cde : IsoscelesTriangle)
  (h1 : perimeter cde = 22)
  (h2 : perimeter abc = 24)
  (h3 : cde.base = 8)
  (h4 : abc.leg = cde.leg) : 
  abc.base = 10 := by sorry

end NUMINAMATH_CALUDE_ab_length_in_two_isosceles_triangles_l3623_362354


namespace NUMINAMATH_CALUDE_initial_chips_count_l3623_362325

/-- The number of tortilla chips Nancy initially had in her bag -/
def initial_chips : ℕ := sorry

/-- The number of tortilla chips Nancy gave to her brother -/
def chips_to_brother : ℕ := 7

/-- The number of tortilla chips Nancy gave to her sister -/
def chips_to_sister : ℕ := 5

/-- The number of tortilla chips Nancy kept for herself -/
def chips_kept : ℕ := 10

/-- Theorem stating that the initial number of chips is 22 -/
theorem initial_chips_count : initial_chips = 22 := by sorry

end NUMINAMATH_CALUDE_initial_chips_count_l3623_362325


namespace NUMINAMATH_CALUDE_magnitude_of_parallel_vector_difference_l3623_362392

/-- Given two vectors a and b in ℝ², where a is parallel to b, 
    prove that the magnitude of their difference is 2√5. -/
theorem magnitude_of_parallel_vector_difference :
  ∀ (x : ℝ), 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 6]
  (∃ (k : ℝ), ∀ (i : Fin 2), a i = k * b i) →  -- Parallel condition
  ‖a - b‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_parallel_vector_difference_l3623_362392


namespace NUMINAMATH_CALUDE_trenton_fixed_earnings_l3623_362309

/-- Trenton's weekly earnings structure -/
structure WeeklyEarnings where
  fixed : ℝ
  commissionRate : ℝ
  salesGoal : ℝ
  totalEarningsGoal : ℝ

/-- Trenton's actual weekly earnings -/
def actualEarnings (w : WeeklyEarnings) : ℝ :=
  w.fixed + w.commissionRate * w.salesGoal

/-- Theorem: Trenton's fixed weekly earnings are $190 -/
theorem trenton_fixed_earnings :
  ∀ w : WeeklyEarnings,
  w.commissionRate = 0.04 →
  w.salesGoal = 7750 →
  w.totalEarningsGoal = 500 →
  actualEarnings w ≥ w.totalEarningsGoal →
  w.fixed = 190 := by
sorry

end NUMINAMATH_CALUDE_trenton_fixed_earnings_l3623_362309


namespace NUMINAMATH_CALUDE_remainder_product_mod_75_l3623_362341

theorem remainder_product_mod_75 : (3203 * 4507 * 9929) % 75 = 34 := by
  sorry

end NUMINAMATH_CALUDE_remainder_product_mod_75_l3623_362341


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l3623_362321

/-- Represents days of the week --/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with specific properties --/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  numberOfFridays : Nat
  numberOfDays : Nat
  firstDayNotFriday : firstDay ≠ DayOfWeek.Friday
  lastDayNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : numberOfFridays = 5

/-- Function to determine the day of the week for a given day number --/
def dayOfWeekForDay (m : Month) (day : Nat) : DayOfWeek :=
  sorry

theorem twelfth_day_is_monday (m : Month) : 
  dayOfWeekForDay m 12 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l3623_362321


namespace NUMINAMATH_CALUDE_primitive_points_polynomial_theorem_l3623_362391

/-- A primitive point is an ordered pair of integers with greatest common divisor 1. -/
def PrimitivePoint : Type := { p : ℤ × ℤ // Int.gcd p.1 p.2 = 1 }

/-- The theorem statement -/
theorem primitive_points_polynomial_theorem (S : Finset PrimitivePoint) :
  ∃ (n : ℕ+) (a : Fin (n + 1) → ℤ),
    ∀ (p : PrimitivePoint), p ∈ S →
      (Finset.range (n + 1)).sum (fun i => a i * p.val.1^(n - i) * p.val.2^i) = 1 := by
  sorry

end NUMINAMATH_CALUDE_primitive_points_polynomial_theorem_l3623_362391


namespace NUMINAMATH_CALUDE_expression_simplification_l3623_362365

theorem expression_simplification (w v : ℝ) :
  3 * w + 5 * w + 7 * v + 9 * w + 11 * v + 15 = 17 * w + 18 * v + 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3623_362365


namespace NUMINAMATH_CALUDE_min_stamps_proof_l3623_362302

/-- The minimum number of stamps needed to make 60 cents using only 5 cent and 6 cent stamps -/
def min_stamps : ℕ := 10

/-- The value of stamps in cents -/
def total_value : ℕ := 60

/-- Proves that the minimum number of stamps needed to make 60 cents using only 5 cent and 6 cent stamps is 10 -/
theorem min_stamps_proof :
  ∀ c f : ℕ, 5 * c + 6 * f = total_value → c + f ≥ min_stamps :=
sorry

end NUMINAMATH_CALUDE_min_stamps_proof_l3623_362302


namespace NUMINAMATH_CALUDE_students_in_two_classes_l3623_362364

theorem students_in_two_classes
  (total_students : ℕ)
  (history_students : ℕ)
  (math_students : ℕ)
  (english_students : ℕ)
  (all_three_classes : ℕ)
  (h_total : total_students = 68)
  (h_history : history_students = 19)
  (h_math : math_students = 14)
  (h_english : english_students = 26)
  (h_all_three : all_three_classes = 3)
  (h_at_least_one : total_students = history_students + math_students + english_students
    - (history_students + math_students - all_three_classes
    + history_students + english_students - all_three_classes
    + math_students + english_students - all_three_classes)
    + all_three_classes) :
  history_students + math_students - all_three_classes
  + history_students + english_students - all_three_classes
  + math_students + english_students - all_three_classes
  - 3 * all_three_classes = 6 :=
sorry

end NUMINAMATH_CALUDE_students_in_two_classes_l3623_362364


namespace NUMINAMATH_CALUDE_equation_one_solutions_l3623_362395

theorem equation_one_solutions (x : ℝ) :
  (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l3623_362395


namespace NUMINAMATH_CALUDE_salary_distribution_l3623_362311

/-- Represents the salary distribution problem for three teams of workers --/
theorem salary_distribution
  (total_value : ℝ)
  (team1_people team1_days : ℕ)
  (team2_people team2_days : ℕ)
  (team3_days : ℕ)
  (team3_people_ratio : ℝ)
  (h1 : total_value = 325500)
  (h2 : team1_people = 15)
  (h3 : team1_days = 21)
  (h4 : team2_people = 14)
  (h5 : team2_days = 25)
  (h6 : team3_days = 20)
  (h7 : team3_people_ratio = 1.4) :
  ∃ (salary_per_day : ℝ),
    let team1_salary := salary_per_day * team1_people * team1_days
    let team2_salary := salary_per_day * team2_people * team2_days
    let team3_salary := salary_per_day * (team3_people_ratio * team1_people) * team3_days
    team1_salary + team2_salary + team3_salary = total_value ∧
    team1_salary = 94500 ∧
    team2_salary = 105000 ∧
    team3_salary = 126000 :=
by sorry

end NUMINAMATH_CALUDE_salary_distribution_l3623_362311


namespace NUMINAMATH_CALUDE_sum_divisible_by_17_l3623_362324

theorem sum_divisible_by_17 : 
  ∃ k : ℤ, 82 + 83 + 84 + 85 + 86 + 87 + 88 + 89 = 17 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_17_l3623_362324


namespace NUMINAMATH_CALUDE_missing_figure_proof_l3623_362344

theorem missing_figure_proof (x : ℝ) (h : (0.50 / 100) * x = 0.12) : x = 24 := by
  sorry

end NUMINAMATH_CALUDE_missing_figure_proof_l3623_362344


namespace NUMINAMATH_CALUDE_bicycle_distance_l3623_362388

/-- Given a bicycle traveling b/2 feet in t seconds, prove it travels 50b/t yards in 5 minutes -/
theorem bicycle_distance (b t : ℝ) (h : b > 0) (h' : t > 0) : 
  (b / 2) / t * (5 * 60) / 3 = 50 * b / t := by
  sorry

end NUMINAMATH_CALUDE_bicycle_distance_l3623_362388


namespace NUMINAMATH_CALUDE_landscape_breadth_l3623_362378

/-- Represents a rectangular landscape with specific features -/
structure Landscape where
  length : ℝ
  breadth : ℝ
  playground_area : ℝ
  walking_path_ratio : ℝ
  water_body_ratio : ℝ

/-- Theorem stating the breadth of the landscape given specific conditions -/
theorem landscape_breadth (l : Landscape) 
  (h1 : l.breadth = 8 * l.length)
  (h2 : l.playground_area = 3200)
  (h3 : l.playground_area = (l.length * l.breadth) / 9)
  (h4 : l.walking_path_ratio = 1 / 18)
  (h5 : l.water_body_ratio = 1 / 6)
  : l.breadth = 480 := by
  sorry

end NUMINAMATH_CALUDE_landscape_breadth_l3623_362378


namespace NUMINAMATH_CALUDE_infinitely_many_a_without_solution_l3623_362342

-- Define τ(n) as the number of positive divisors of n
def tau (n : ℕ+) : ℕ := sorry

-- Statement of the theorem
theorem infinitely_many_a_without_solution :
  ∃ (S : Set ℕ+), (Set.Infinite S) ∧ 
  (∀ (a : ℕ+), a ∈ S → ∀ (n : ℕ+), tau (a * n) ≠ n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_a_without_solution_l3623_362342


namespace NUMINAMATH_CALUDE_sqrt_sum_simplification_l3623_362353

theorem sqrt_sum_simplification : Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_simplification_l3623_362353


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l3623_362328

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) 
  (h1 : total_length = 28)
  (h2 : ratio = 2.00001 / 5) : 
  ∃ (shorter_piece : ℝ), 
    shorter_piece + ratio * shorter_piece = total_length ∧ 
    shorter_piece = 20 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l3623_362328


namespace NUMINAMATH_CALUDE_train_length_l3623_362381

/-- The length of a train given its speed, a man's speed in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) :
  train_speed = 50 →
  man_speed = 5 →
  passing_time = 7.2 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.1 :=
by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3623_362381


namespace NUMINAMATH_CALUDE_hyperbola_center_l3623_362308

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * x - 8)^2 / 8^2 - (5 * y + 10)^2 / 3^2 = 1

-- Theorem stating that the center of the hyperbola is at (2, -2)
theorem hyperbola_center :
  ∃ (h k : ℝ), h = 2 ∧ k = -2 ∧
  (∀ (x y : ℝ), hyperbola_equation x y ↔ hyperbola_equation (x - h) (y - k)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l3623_362308


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3623_362379

/-- The quadratic equation (a-5)x^2 - 4x - 1 = 0 has real roots if and only if a ≥ 1 and a ≠ 5. -/
theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, (a - 5) * x^2 - 4*x - 1 = 0) ↔ (a ≥ 1 ∧ a ≠ 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3623_362379


namespace NUMINAMATH_CALUDE_prime_power_sum_l3623_362334

theorem prime_power_sum (p q r : ℕ) : 
  p.Prime → q.Prime → r.Prime → p^q + q^p = r → 
  ((p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17)) :=
sorry

end NUMINAMATH_CALUDE_prime_power_sum_l3623_362334


namespace NUMINAMATH_CALUDE_revenue_is_90_dollars_l3623_362340

def total_bags : ℕ := 10
def oranges_per_bag : ℕ := 30
def bags_with_10_percent_rotten : ℕ := 4
def bags_with_20_percent_rotten : ℕ := 3
def bags_with_5_percent_rotten : ℕ := 3
def oranges_for_juice : ℕ := 70
def oranges_for_jams : ℕ := 15
def selling_price_per_orange : ℚ := 0.50

def total_oranges : ℕ := total_bags * oranges_per_bag

def rotten_oranges : ℕ := 
  bags_with_10_percent_rotten * oranges_per_bag / 10 +
  bags_with_20_percent_rotten * oranges_per_bag / 5 +
  bags_with_5_percent_rotten * oranges_per_bag / 20

def good_oranges : ℕ := total_oranges - rotten_oranges

def oranges_for_sale : ℕ := good_oranges - oranges_for_juice - oranges_for_jams

def total_revenue : ℚ := oranges_for_sale * selling_price_per_orange

theorem revenue_is_90_dollars : total_revenue = 90 := by
  sorry

end NUMINAMATH_CALUDE_revenue_is_90_dollars_l3623_362340


namespace NUMINAMATH_CALUDE_factorial_squared_greater_than_power_l3623_362349

theorem factorial_squared_greater_than_power (n : ℕ) (h : n > 2) :
  (Nat.factorial n)^2 > n^n := by
  sorry

end NUMINAMATH_CALUDE_factorial_squared_greater_than_power_l3623_362349


namespace NUMINAMATH_CALUDE_twenty_three_in_base_two_l3623_362360

theorem twenty_three_in_base_two : 23 = 1*2^4 + 0*2^3 + 1*2^2 + 1*2^1 + 1*2^0 := by
  sorry

end NUMINAMATH_CALUDE_twenty_three_in_base_two_l3623_362360
