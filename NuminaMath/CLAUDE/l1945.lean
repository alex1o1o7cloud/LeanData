import Mathlib

namespace NUMINAMATH_CALUDE_closest_cube_root_to_50_l1945_194564

theorem closest_cube_root_to_50 :
  ∀ n : ℤ, |((2:ℝ)^n)^(1/3) - 50| ≥ |((2:ℝ)^17)^(1/3) - 50| :=
by sorry

end NUMINAMATH_CALUDE_closest_cube_root_to_50_l1945_194564


namespace NUMINAMATH_CALUDE_correct_price_reduction_equation_l1945_194515

/-- Represents the price reduction scenario for a shirt -/
def price_reduction_scenario (original_price final_price : ℝ) (x : ℝ) : Prop :=
  original_price * (1 - x)^2 = final_price

/-- The theorem stating that the given equation correctly represents the price reduction scenario -/
theorem correct_price_reduction_equation :
  price_reduction_scenario 400 200 x ↔ 400 * (1 - x)^2 = 200 :=
by sorry

end NUMINAMATH_CALUDE_correct_price_reduction_equation_l1945_194515


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l1945_194559

theorem pencil_pen_cost (p q : ℝ) 
  (h1 : 4 * p + 3 * q = 4.20)
  (h2 : 3 * p + 4 * q = 4.55) :
  p + q = 1.25 := by
sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l1945_194559


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l1945_194590

theorem consecutive_integers_sum_of_squares (b : ℤ) : 
  (b - 1) * b * (b + 1) = 12 * (3 * b) + b^2 → 
  (b - 1)^2 + b^2 + (b + 1)^2 = 149 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l1945_194590


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1945_194592

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

def has_equal_intercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line2D),
    (point_on_line ⟨1, 2⟩ l1 ∧ has_equal_intercepts l1) ∧
    (point_on_line ⟨1, 2⟩ l2 ∧ has_equal_intercepts l2) ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -3) ∨ (l2.a = 2 ∧ l2.b = -1 ∧ l2.c = 0)) :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1945_194592


namespace NUMINAMATH_CALUDE_eggs_per_cake_l1945_194526

def total_eggs : ℕ := 60
def fridge_eggs : ℕ := 10
def num_cakes : ℕ := 10

theorem eggs_per_cake :
  (total_eggs - fridge_eggs) / num_cakes = 5 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_cake_l1945_194526


namespace NUMINAMATH_CALUDE_final_cell_count_l1945_194574

def initial_cells : ℕ := 5
def split_ratio : ℕ := 3
def split_interval : ℕ := 3
def total_days : ℕ := 12

def geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * r^(n - 1)

theorem final_cell_count :
  geometric_sequence initial_cells split_ratio (total_days / split_interval) = 135 := by
  sorry

end NUMINAMATH_CALUDE_final_cell_count_l1945_194574


namespace NUMINAMATH_CALUDE_unique_solution_iff_k_zero_l1945_194578

/-- 
Theorem: The pair of equations y = x^2 and y = 2x^2 + k have exactly one solution 
if and only if k = 0.
-/
theorem unique_solution_iff_k_zero (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = p.1^2 ∧ p.2 = 2*p.1^2 + k) ↔ k = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_k_zero_l1945_194578


namespace NUMINAMATH_CALUDE_segment_AE_length_l1945_194565

-- Define the quadrilateral ABCD and point E
structure Quadrilateral :=
  (A B C D E : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let d_AB := Real.sqrt ((q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2)
  let d_CD := Real.sqrt ((q.D.1 - q.C.1)^2 + (q.D.2 - q.C.2)^2)
  let d_AC := Real.sqrt ((q.C.1 - q.A.1)^2 + (q.C.2 - q.A.2)^2)
  let d_AE := Real.sqrt ((q.E.1 - q.A.1)^2 + (q.E.2 - q.A.2)^2)
  let d_EC := Real.sqrt ((q.C.1 - q.E.1)^2 + (q.C.2 - q.E.2)^2)
  d_AB = 10 ∧ d_CD = 15 ∧ d_AC = 18 ∧
  (q.E.1 - q.A.1) * (q.C.1 - q.A.1) + (q.E.2 - q.A.2) * (q.C.2 - q.A.2) = d_AE * d_AC ∧
  (q.E.1 - q.B.1) * (q.D.1 - q.B.1) + (q.E.2 - q.B.2) * (q.D.2 - q.B.2) = 
    Real.sqrt ((q.E.1 - q.B.1)^2 + (q.E.2 - q.B.2)^2) * Real.sqrt ((q.D.1 - q.B.1)^2 + (q.D.2 - q.B.2)^2) ∧
  d_AE / d_EC = 10 / 15

theorem segment_AE_length (q : Quadrilateral) (h : is_valid_quadrilateral q) :
  Real.sqrt ((q.E.1 - q.A.1)^2 + (q.E.2 - q.A.2)^2) = 36 / 5 := by
  sorry

end NUMINAMATH_CALUDE_segment_AE_length_l1945_194565


namespace NUMINAMATH_CALUDE_fourth_power_subset_exists_l1945_194558

/-- The set of prime numbers less than or equal to 26 -/
def primes_le_26 : Finset ℕ := sorry

/-- A function that represents a number as a tuple of exponents of primes <= 26 -/
def exponent_tuple (n : ℕ) : Fin 9 → ℕ := sorry

/-- The set M of 1985 different positive integers with prime factors <= 26 -/
def M : Finset ℕ := sorry

/-- The cardinality of M is 1985 -/
axiom M_card : Finset.card M = 1985

/-- All elements in M have prime factors <= 26 -/
axiom M_primes (n : ℕ) : n ∈ M → ∀ p : ℕ, p.Prime → p ∣ n → p ≤ 26

/-- All elements in M are different -/
axiom M_distinct : ∀ a b : ℕ, a ∈ M → b ∈ M → a ≠ b

/-- Main theorem: There exists a subset of 4 elements from M whose product is a fourth power -/
theorem fourth_power_subset_exists : 
  ∃ (a b c d : ℕ) (k : ℕ), a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = k^4 := by sorry

end NUMINAMATH_CALUDE_fourth_power_subset_exists_l1945_194558


namespace NUMINAMATH_CALUDE_solution_inequality_l1945_194503

theorem solution_inequality (x : ℝ) (h : x = 1.8) : x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_inequality_l1945_194503


namespace NUMINAMATH_CALUDE_player_A_wins_l1945_194533

/-- Represents a pile of matches -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Represents a player's move -/
structure Move :=
  (take : Nat)
  (split : Nat)
  (into : Nat × Nat)

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  move.take ∈ state.piles.map Pile.count ∧
  move.split ∈ state.piles.map Pile.count ∧
  move.split ≠ move.take ∧
  move.into.1 > 0 ∧ move.into.2 > 0 ∧
  move.into.1 + move.into.2 = move.split

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { piles := (state.piles.filter (λ p => p.count ≠ move.take ∧ p.count ≠ move.split)) ++
              [Pile.mk move.into.1, Pile.mk move.into.2] }

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Prop :=
  ∀ move, ¬isValidMove state move

/-- Represents the optimal strategy for a player -/
def OptimalStrategy := GameState → Option Move

/-- Theorem: Player A has a winning strategy -/
theorem player_A_wins (initialState : GameState)
  (h : initialState.piles = [Pile.mk 100, Pile.mk 200, Pile.mk 300]) :
  ∃ (strategyA : OptimalStrategy),
    ∀ (strategyB : OptimalStrategy),
      ∃ (finalState : GameState),
        isGameOver finalState ∧
        -- The last move was made by Player B (meaning A wins)
        (∃ (moves : List Move),
          moves.length % 2 = 1 ∧
          finalState = moves.foldl applyMove initialState) :=
sorry

end NUMINAMATH_CALUDE_player_A_wins_l1945_194533


namespace NUMINAMATH_CALUDE_sprinkles_remaining_l1945_194541

theorem sprinkles_remaining (initial_cans : ℕ) (remaining_cans : ℕ) : 
  initial_cans = 12 → 
  remaining_cans = initial_cans / 2 - 3 → 
  remaining_cans = 3 := by
sorry

end NUMINAMATH_CALUDE_sprinkles_remaining_l1945_194541


namespace NUMINAMATH_CALUDE_sum_of_absolute_roots_l1945_194544

/-- Given a cubic polynomial x^3 - 2023x + m with three integer roots,
    prove that the sum of the absolute values of the roots is 80. -/
theorem sum_of_absolute_roots (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  |a| + |b| + |c| = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_roots_l1945_194544


namespace NUMINAMATH_CALUDE_number_satisfies_equation_l1945_194509

theorem number_satisfies_equation : ∃ (n : ℕ), n = 14 ∧ 2^n - 2^(n-2) = 3 * 2^12 :=
by
  sorry

end NUMINAMATH_CALUDE_number_satisfies_equation_l1945_194509


namespace NUMINAMATH_CALUDE_john_cannot_achieve_goal_l1945_194502

/-- Represents John's quiz scores throughout the year -/
structure QuizScores where
  total : Nat
  goal_percentage : Rat
  taken : Nat
  high_scores : Nat

/-- Checks if it's possible to achieve the goal given the current scores -/
def can_achieve_goal (scores : QuizScores) : Prop :=
  ∃ (remaining_high_scores : Nat),
    remaining_high_scores ≤ scores.total - scores.taken ∧
    (scores.high_scores + remaining_high_scores : Rat) / scores.total ≥ scores.goal_percentage

/-- John's actual quiz scores -/
def john_scores : QuizScores :=
  { total := 60
  , goal_percentage := 9/10
  , taken := 40
  , high_scores := 32 }

/-- Theorem stating that John cannot achieve his goal -/
theorem john_cannot_achieve_goal :
  ¬(can_achieve_goal john_scores) := by
  sorry

end NUMINAMATH_CALUDE_john_cannot_achieve_goal_l1945_194502


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l1945_194529

theorem stewart_farm_ratio : 
  ∀ (horse_food_per_day : ℕ) (total_horse_food : ℕ) (num_sheep : ℕ),
    horse_food_per_day = 230 →
    total_horse_food = 12880 →
    num_sheep = 32 →
    let num_horses := total_horse_food / horse_food_per_day
    (num_sheep : ℚ) / num_horses = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l1945_194529


namespace NUMINAMATH_CALUDE_range_of_a_part1_range_of_a_part2_l1945_194596

def proposition_p (a : ℝ) : Prop := a^2 - 5*a - 6 > 0

def proposition_q (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^2 + a*x₁ + 1 = 0 ∧ x₂^2 + a*x₂ + 1 = 0

theorem range_of_a_part1 :
  {a : ℝ | proposition_p a} = {a | a < -1 ∨ a > 6} :=
sorry

theorem range_of_a_part2 :
  {a : ℝ | (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a)} =
  {a | a < -1 ∨ (2 < a ∧ a ≤ 6)} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_part1_range_of_a_part2_l1945_194596


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1945_194531

/-- The greatest common factor of 18, 30, and 45 -/
def C : ℕ := Nat.gcd 18 (Nat.gcd 30 45)

/-- The least common multiple of 18, 30, and 45 -/
def D : ℕ := Nat.lcm 18 (Nat.lcm 30 45)

/-- The sum of the greatest common factor and the least common multiple of 18, 30, and 45 is 93 -/
theorem gcd_lcm_sum : C + D = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1945_194531


namespace NUMINAMATH_CALUDE_catfish_weight_l1945_194550

theorem catfish_weight (trout_count : ℕ) (catfish_count : ℕ) (bluegill_count : ℕ)
  (trout_weight : ℝ) (bluegill_weight : ℝ) (total_weight : ℝ)
  (h1 : trout_count = 4)
  (h2 : catfish_count = 3)
  (h3 : bluegill_count = 5)
  (h4 : trout_weight = 2)
  (h5 : bluegill_weight = 2.5)
  (h6 : total_weight = 25)
  (h7 : total_weight = trout_count * trout_weight + catfish_count * (total_weight - trout_count * trout_weight - bluegill_count * bluegill_weight) / catfish_count + bluegill_count * bluegill_weight) :
  (total_weight - trout_count * trout_weight - bluegill_count * bluegill_weight) / catfish_count = 1.5 := by
sorry

end NUMINAMATH_CALUDE_catfish_weight_l1945_194550


namespace NUMINAMATH_CALUDE_biker_problem_l1945_194520

/-- Two bikers on a circular path problem -/
theorem biker_problem (t1 t2 meet_time : ℕ) : 
  t1 = 12 →  -- First rider completes a round in 12 minutes
  meet_time = 36 →  -- They meet again at the starting point after 36 minutes
  meet_time % t1 = 0 →  -- First rider completes whole number of rounds
  meet_time % t2 = 0 →  -- Second rider completes whole number of rounds
  t2 > t1 →  -- Second rider is slower than the first
  t2 = 36  -- Second rider takes 36 minutes to complete a round
  := by sorry

end NUMINAMATH_CALUDE_biker_problem_l1945_194520


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l1945_194535

/-- A point P with coordinates (m, 4+2m) is in the third quadrant if and only if m < -2 -/
theorem point_in_third_quadrant (m : ℝ) :
  (m < 0 ∧ 4 + 2 * m < 0) ↔ m < -2 := by
sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l1945_194535


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1945_194540

def A : Set ℝ := {x | x^2 < 4}
def B : Set ℝ := {y | ∃ x ∈ A, y = x^2 - 2*x - 1}

theorem union_of_A_and_B : A ∪ B = {x | -2 ≤ x ∧ x < 7} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1945_194540


namespace NUMINAMATH_CALUDE_special_function_sqrt_5753_l1945_194522

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = x * f y + y * f x) ∧
  (∀ x y : ℝ, f (x + y) = f (x * 1993) + f (y * 1993))

/-- The main theorem -/
theorem special_function_sqrt_5753 (f : ℝ → ℝ) (h : special_function f) :
  f (Real.sqrt 5753) = 0 := by sorry

end NUMINAMATH_CALUDE_special_function_sqrt_5753_l1945_194522


namespace NUMINAMATH_CALUDE_function_property_l1945_194582

theorem function_property (f : ℝ → ℝ) (m : ℝ)
  (h1 : ∀ x, f (2 + x) = f (-x))
  (h2 : ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f y < f x)
  (h3 : f (1 - m) < f m) :
  m > 1/2 := by sorry

end NUMINAMATH_CALUDE_function_property_l1945_194582


namespace NUMINAMATH_CALUDE_brownie_theorem_l1945_194584

/-- The number of brownie pieces obtained from a rectangular tray -/
def brownie_pieces (tray_length tray_width piece_length piece_width : ℕ) : ℕ :=
  (tray_length * tray_width) / (piece_length * piece_width)

/-- Theorem stating that a 24-inch by 30-inch tray yields 60 brownie pieces of size 3 inches by 4 inches -/
theorem brownie_theorem :
  brownie_pieces 24 30 3 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_brownie_theorem_l1945_194584


namespace NUMINAMATH_CALUDE_stevens_height_l1945_194545

/-- Given a pole's height and shadow length, and a person's shadow length,
    calculate the person's height in centimeters. -/
def calculate_height (pole_height pole_shadow person_shadow : ℚ) : ℚ :=
  let ratio := pole_height / pole_shadow
  let person_height_feet := ratio * (person_shadow / 12)
  person_height_feet * 30.48

/-- Theorem stating that under the given conditions, Steven's height is 190.5 cm. -/
theorem stevens_height :
  let pole_height : ℚ := 60
  let pole_shadow : ℚ := 20
  let steven_shadow_inches : ℚ := 25
  calculate_height pole_height pole_shadow (steven_shadow_inches / 12) = 190.5 := by
  sorry

end NUMINAMATH_CALUDE_stevens_height_l1945_194545


namespace NUMINAMATH_CALUDE_four_digit_sum_1989_l1945_194593

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- Apply the digit sum transformation n times -/
def iterateDigitSum (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterateDigitSum (digitSum n) k

/-- The main theorem stating that applying digit sum 4 times to 1989 results in 9 -/
theorem four_digit_sum_1989 : iterateDigitSum 1989 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_sum_1989_l1945_194593


namespace NUMINAMATH_CALUDE_fraction_equality_l1945_194542

theorem fraction_equality : (7 + 21) / (14 + 42) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1945_194542


namespace NUMINAMATH_CALUDE_count_common_divisors_36_90_l1945_194571

def divisors_of_both (a b : ℕ) : Finset ℕ :=
  (Finset.range a).filter (fun x => x > 0 ∧ a % x = 0 ∧ b % x = 0)

theorem count_common_divisors_36_90 : (divisors_of_both 36 90).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_common_divisors_36_90_l1945_194571


namespace NUMINAMATH_CALUDE_permutation_problem_l1945_194514

theorem permutation_problem (n : ℕ) : (n * (n - 1) = 132) ↔ (n = 12) := by sorry

end NUMINAMATH_CALUDE_permutation_problem_l1945_194514


namespace NUMINAMATH_CALUDE_correct_distribution_probability_l1945_194553

/-- The number of rolls of each type -/
def rolls_per_type : ℕ := 3

/-- The number of types of rolls -/
def num_types : ℕ := 4

/-- The total number of rolls -/
def total_rolls : ℕ := rolls_per_type * num_types

/-- The number of guests -/
def num_guests : ℕ := 3

/-- The number of rolls each guest receives -/
def rolls_per_guest : ℕ := num_types

/-- The probability of each guest getting one roll of each type -/
def probability_correct_distribution : ℚ := 27 / 1925

theorem correct_distribution_probability :
  (rolls_per_type : ℚ) * (rolls_per_type - 1) * (rolls_per_type - 2) /
  (total_rolls * (total_rolls - 1) * (total_rolls - 2) * (total_rolls - 3)) *
  ((rolls_per_type - 1) * (rolls_per_type - 1) * (rolls_per_type - 1) /
  ((total_rolls - 4) * (total_rolls - 5) * (total_rolls - 6) * (total_rolls - 7))) *
  1 = probability_correct_distribution := by
  sorry

end NUMINAMATH_CALUDE_correct_distribution_probability_l1945_194553


namespace NUMINAMATH_CALUDE_double_root_condition_l1945_194513

theorem double_root_condition (m : ℝ) :
  (∃! x : ℝ, (x - 3) / (x - 1) = m / (x - 1) ∧ x ≠ 1) ↔ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_double_root_condition_l1945_194513


namespace NUMINAMATH_CALUDE_rainfall_difference_l1945_194543

theorem rainfall_difference (monday_rain tuesday_rain : Real) 
  (h1 : monday_rain = 0.9)
  (h2 : tuesday_rain = 0.2) :
  monday_rain - tuesday_rain = 0.7 := by
sorry

end NUMINAMATH_CALUDE_rainfall_difference_l1945_194543


namespace NUMINAMATH_CALUDE_no_roots_of_composite_l1945_194552

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem no_roots_of_composite (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b c x ≠ 2 * x) →
  (∀ x, f a b c (f a b c x) ≠ 4 * x) :=
sorry

end NUMINAMATH_CALUDE_no_roots_of_composite_l1945_194552


namespace NUMINAMATH_CALUDE_correct_calculation_l1945_194546

theorem correct_calculation : (-2)^3 + 6 / ((1/2) - (1/3)) = 28 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1945_194546


namespace NUMINAMATH_CALUDE_p_minus_q_value_l1945_194563

theorem p_minus_q_value (p q : ℚ) (hp : 3 / p = 4) (hq : 3 / q = 18) : p - q = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_p_minus_q_value_l1945_194563


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1945_194549

theorem quadratic_equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + a*a + b = 0) ∧ (b^2 + a*b + b = 0) → a = 1 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1945_194549


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l1945_194539

/-- The number of tickets Amanda sells on the first day -/
def day1_tickets : ℕ := 5 * 4

/-- The number of tickets Amanda sells on the second day -/
def day2_tickets : ℕ := 32

/-- The number of tickets Amanda needs to sell on the third day -/
def day3_tickets : ℕ := 28

/-- The total number of tickets Amanda needs to sell -/
def total_tickets : ℕ := day1_tickets + day2_tickets + day3_tickets

/-- Theorem stating that the total number of tickets Amanda needs to sell is 80 -/
theorem amanda_ticket_sales : total_tickets = 80 := by
  sorry

end NUMINAMATH_CALUDE_amanda_ticket_sales_l1945_194539


namespace NUMINAMATH_CALUDE_black_squares_in_29th_row_l1945_194525

/-- Represents the number of squares in a row of the pattern -/
def squaresInRow (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- Represents the number of black squares in a row of the pattern -/
def blackSquaresInRow (n : ℕ) : ℕ := (squaresInRow n - 1) / 2

/-- Theorem stating that the 29th row contains 28 black squares -/
theorem black_squares_in_29th_row : blackSquaresInRow 29 = 28 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_in_29th_row_l1945_194525


namespace NUMINAMATH_CALUDE_cube_labeling_impossibility_cube_labeling_with_13_l1945_194551

/-- The number of edges in a cube -/
def num_edges : ℕ := 12

/-- The number of vertices in a cube -/
def num_vertices : ℕ := 8

/-- The number of edges connected to each vertex in a cube -/
def edges_per_vertex : ℕ := 3

/-- A labeling of a cube's edges -/
def Labeling := Fin num_edges → ℕ

/-- The sum of labels at a vertex for a given labeling -/
def vertex_sum (l : Labeling) : ℕ := sorry

/-- Predicate for a valid labeling with values 1 to 12 -/
def valid_labeling (l : Labeling) : Prop :=
  ∀ e : Fin num_edges, l e ∈ Finset.range num_edges

/-- Predicate for a constant sum labeling -/
def constant_sum_labeling (l : Labeling) : Prop :=
  ∃ s : ℕ, ∀ v : Fin num_vertices, vertex_sum l = s

/-- Predicate for a valid labeling with one value replaced by 13 -/
def valid_labeling_with_13 (l : Labeling) : Prop :=
  ∃ e : Fin num_edges, l e = 13 ∧
    ∀ e' : Fin num_edges, e' ≠ e → l e' ∈ Finset.range num_edges

theorem cube_labeling_impossibility :
  ¬∃ l : Labeling, valid_labeling l ∧ constant_sum_labeling l :=
sorry

theorem cube_labeling_with_13 :
  ∃ l : Labeling, valid_labeling_with_13 l ∧ constant_sum_labeling l ↔
    ∃ i ∈ ({3, 7, 11} : Finset ℕ), ∃ l : Labeling,
      valid_labeling_with_13 l ∧ constant_sum_labeling l ∧
      ∃ e : Fin num_edges, l e = 13 ∧ (∀ e' : Fin num_edges, e' ≠ e → l e' ≠ i) :=
sorry

end NUMINAMATH_CALUDE_cube_labeling_impossibility_cube_labeling_with_13_l1945_194551


namespace NUMINAMATH_CALUDE_unique_salaries_l1945_194568

/-- Represents the weekly salaries of three employees -/
structure Salaries where
  n : ℝ  -- Salary of employee N
  m : ℝ  -- Salary of employee M
  p : ℝ  -- Salary of employee P

/-- Checks if the given salaries satisfy the problem conditions -/
def satisfiesConditions (s : Salaries) : Prop :=
  s.m = 1.2 * s.n ∧
  s.p = 1.5 * s.m ∧
  s.n + s.m + s.p = 1500

/-- Theorem stating that the given salaries are the unique solution -/
theorem unique_salaries : 
  ∃! s : Salaries, satisfiesConditions s ∧ 
    s.n = 375 ∧ s.m = 450 ∧ s.p = 675 := by
  sorry

end NUMINAMATH_CALUDE_unique_salaries_l1945_194568


namespace NUMINAMATH_CALUDE_butternut_figurines_eq_four_l1945_194511

/-- The number of figurines that can be created from a block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be created from a block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 245

/-- The number of blocks of basswood Adam has -/
def basswood_blocks : ℕ := 15

/-- The number of blocks of butternut wood Adam has -/
def butternut_blocks : ℕ := 20

/-- The number of blocks of Aspen wood Adam has -/
def aspen_blocks : ℕ := 20

/-- The number of figurines that can be created from a block of butternut wood -/
def butternut_figurines : ℕ := (total_figurines - basswood_blocks * basswood_figurines - aspen_blocks * aspen_figurines) / butternut_blocks

theorem butternut_figurines_eq_four : butternut_figurines = 4 := by
  sorry

end NUMINAMATH_CALUDE_butternut_figurines_eq_four_l1945_194511


namespace NUMINAMATH_CALUDE_unique_linear_function_l1945_194528

/-- A linear function passing through two points -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem unique_linear_function :
  ∃! (k b : ℝ), linear_function k b 3 = 4 ∧ linear_function k b 4 = 5 ∧
  ∀ x, linear_function k b x = x + 1 :=
sorry

end NUMINAMATH_CALUDE_unique_linear_function_l1945_194528


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_range_l1945_194562

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (1 - m) + y^2 / (m + 2) = 1
def q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (2 * m) + y^2 / (2 - m) = 1

-- Define the condition that q represents an ellipse with foci on the x-axis
def q_ellipse (m : ℝ) : Prop := 2 * m > 0 ∧ 2 - m > 0

-- Define the theorem
theorem hyperbola_ellipse_range (m : ℝ) : 
  (¬(p m ∧ q m) ∧ q_ellipse m) ↔ (m ≤ 1 ∨ m ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_range_l1945_194562


namespace NUMINAMATH_CALUDE_total_distance_not_unique_l1945_194501

/-- Represents a part of a journey with a specific speed -/
structure JourneyPart where
  speed : ℝ
  time : ℝ

/-- Represents a complete journey -/
structure Journey where
  parts : List JourneyPart
  totalTime : ℝ

/-- Calculates the distance of a journey part -/
def distanceOfPart (part : JourneyPart) : ℝ :=
  part.speed * part.time

/-- Calculates the total distance of a journey -/
def totalDistance (journey : Journey) : ℝ :=
  (journey.parts.map distanceOfPart).sum

/-- Theorem stating that the total distance cannot be uniquely determined -/
theorem total_distance_not_unique (totalTime : ℝ) (speeds : List ℝ) :
  ∃ (j1 j2 : Journey), 
    j1.totalTime = totalTime ∧ 
    j2.totalTime = totalTime ∧ 
    (j1.parts.map (·.speed)) = speeds ∧ 
    (j2.parts.map (·.speed)) = speeds ∧ 
    totalDistance j1 ≠ totalDistance j2 := by
  sorry

#check total_distance_not_unique

end NUMINAMATH_CALUDE_total_distance_not_unique_l1945_194501


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l1945_194530

/-- The line x = k intersects the parabola x = -3y^2 - 2y + 7 at exactly one point if and only if k = 22/3 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! y : ℝ, k = -3 * y^2 - 2 * y + 7) ↔ k = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l1945_194530


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1945_194588

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) 
  (h_arith : 2 * a 1 - a 2 = a 2 - a 3 / 2) :
  (a 2017 + a 2016) / (a 2015 + a 2014) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1945_194588


namespace NUMINAMATH_CALUDE_average_of_numbers_l1945_194561

def numbers : List ℝ := [10, 4, 8, 7, 6]

theorem average_of_numbers : (numbers.sum / numbers.length : ℝ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l1945_194561


namespace NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l1945_194517

/-- Represents a two-digit positive integer -/
structure TwoDigitNumber where
  value : Nat
  is_two_digit : 10 ≤ value ∧ value ≤ 99

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- The main theorem -/
theorem smallest_fourth_lucky_number :
  ∀ (n : TwoDigitNumber),
    (sumOfDigits 46 + sumOfDigits 24 + sumOfDigits 85 + sumOfDigits n.value = 
     (46 + 24 + 85 + n.value) / 4) →
    n.value ≥ 59 := by
  sorry

#eval sumOfDigits 59  -- Expected output: 14
#eval (46 + 24 + 85 + 59) / 4  -- Expected output: 53
#eval sumOfDigits 46 + sumOfDigits 24 + sumOfDigits 85 + sumOfDigits 59  -- Expected output: 53

end NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l1945_194517


namespace NUMINAMATH_CALUDE_add_fractions_l1945_194581

theorem add_fractions : (3 : ℚ) / 4 + (5 : ℚ) / 6 = (19 : ℚ) / 12 := by sorry

end NUMINAMATH_CALUDE_add_fractions_l1945_194581


namespace NUMINAMATH_CALUDE_triangle_1234_l1945_194569

/-- Define the operation △ -/
def triangle (n m : ℕ) : ℕ := sorry

/-- Axiom for the first condition -/
axiom triangle_1 {a b c d : ℕ} (h1 : 0 < a ∧ a < 10) (h2 : 0 < b ∧ b < 10) 
  (h3 : 0 < c ∧ c < 10) (h4 : 0 < d ∧ d < 10) : 
  triangle (a * 1000 + b * 100 + c * 10 + d) 1 = b * 1000 + c * 100 + a * 10 + d

/-- Axiom for the second condition -/
axiom triangle_2 {a b c d : ℕ} (h1 : 0 < a ∧ a < 10) (h2 : 0 < b ∧ b < 10) 
  (h3 : 0 < c ∧ c < 10) (h4 : 0 < d ∧ d < 10) : 
  triangle (a * 1000 + b * 100 + c * 10 + d) 2 = c * 1000 + d * 100 + a * 10 + b

/-- The main theorem to prove -/
theorem triangle_1234 : triangle (triangle 1234 1) 2 = 3412 := by sorry

end NUMINAMATH_CALUDE_triangle_1234_l1945_194569


namespace NUMINAMATH_CALUDE_opposite_of_negative_six_l1945_194512

theorem opposite_of_negative_six : ∃ x : ℤ, ((-6 : ℤ) + x = 0) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_six_l1945_194512


namespace NUMINAMATH_CALUDE_prime_equation_solutions_l1945_194586

theorem prime_equation_solutions :
  ∃ (S : Finset ℕ), (∀ p ∈ S, Nat.Prime p ∧ p^3 + p^2 - 18*p + 26 = 0) ∧ S.card = 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solutions_l1945_194586


namespace NUMINAMATH_CALUDE_repair_cost_percentage_l1945_194585

def apple_price : ℚ := 5/4
def bike_cost : ℚ := 80
def apples_sold : ℕ := 20
def remaining_fraction : ℚ := 1/5

theorem repair_cost_percentage :
  let total_earned : ℚ := apple_price * apples_sold
  let repair_cost : ℚ := total_earned * (1 - remaining_fraction)
  repair_cost / bike_cost = 1/4
:= by sorry

end NUMINAMATH_CALUDE_repair_cost_percentage_l1945_194585


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1945_194599

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1945_194599


namespace NUMINAMATH_CALUDE_football_league_analysis_l1945_194524

structure Team :=
  (avg_goals_conceded : ℝ)
  (std_dev_goals : ℝ)

def team1 : Team := ⟨1.5, 1.1⟩
def team2 : Team := ⟨2.1, 0.4⟩

def better_defense (t1 t2 : Team) : Prop :=
  t1.avg_goals_conceded < t2.avg_goals_conceded

def more_stable_defense (t1 t2 : Team) : Prop :=
  t1.std_dev_goals < t2.std_dev_goals

def inconsistent_defense (t : Team) : Prop :=
  t.std_dev_goals > 1.0

def rarely_concedes_no_goals (t : Team) : Prop :=
  t.avg_goals_conceded > 2.0 ∧ t.std_dev_goals < 0.5

theorem football_league_analysis :
  (better_defense team1 team2) ∧
  (more_stable_defense team2 team1) ∧
  (inconsistent_defense team1) ∧
  ¬(rarely_concedes_no_goals team2) :=
by sorry

end NUMINAMATH_CALUDE_football_league_analysis_l1945_194524


namespace NUMINAMATH_CALUDE_sum_A_B_equals_negative_five_halves_l1945_194594

theorem sum_A_B_equals_negative_five_halves (A B : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 →
    (B * x - 15) / (x^2 - 9*x + 20) = A / (x - 4) + 4 / (x - 5)) →
  A + B = -5/2 := by
sorry

end NUMINAMATH_CALUDE_sum_A_B_equals_negative_five_halves_l1945_194594


namespace NUMINAMATH_CALUDE_equation_satisfied_l1945_194583

theorem equation_satisfied (x y z : ℤ) : 
  x = z ∧ y = x + 1 → x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_satisfied_l1945_194583


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1945_194566

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

theorem plywood_cut_perimeter_difference :
  let original := Rectangle.mk 9 6
  let pieces : List Rectangle := [
    Rectangle.mk 9 2,  -- Configuration 1
    Rectangle.mk 6 3   -- Configuration 2
  ]
  let perimeters := pieces.map perimeter
  let max_perimeter := perimeters.maximum?
  let min_perimeter := perimeters.minimum?
  ∀ (max min : ℝ), max_perimeter = some max → min_perimeter = some min →
    max - min = 6 :=
by sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l1945_194566


namespace NUMINAMATH_CALUDE_decimal_2009_to_octal_l1945_194570

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem decimal_2009_to_octal :
  decimal_to_octal 2009 = [3, 7, 3, 1] :=
by sorry

end NUMINAMATH_CALUDE_decimal_2009_to_octal_l1945_194570


namespace NUMINAMATH_CALUDE_probability_at_least_one_boy_and_girl_l1945_194554

theorem probability_at_least_one_boy_and_girl (p : ℝ) : 
  p = 1/2 → (1 - 2 * p^4) = 7/8 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_boy_and_girl_l1945_194554


namespace NUMINAMATH_CALUDE_max_m_value_l1945_194557

theorem max_m_value (x y m : ℝ) : 
  (4 * x + 3 * y = 4 * m + 5) →
  (3 * x - y = m - 1) →
  (x + 4 * y ≤ 3) →
  (∀ m' : ℝ, m' > m → ¬(∃ x' y' : ℝ, 
    (4 * x' + 3 * y' = 4 * m' + 5) ∧
    (3 * x' - y' = m' - 1) ∧
    (x' + 4 * y' ≤ 3))) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_max_m_value_l1945_194557


namespace NUMINAMATH_CALUDE_eight_n_even_when_n_seven_l1945_194523

theorem eight_n_even_when_n_seven :
  ∃ k : ℤ, 8 * 7 = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_eight_n_even_when_n_seven_l1945_194523


namespace NUMINAMATH_CALUDE_passing_percentage_is_45_percent_l1945_194572

/-- Represents an examination with a passing percentage. -/
structure Examination where
  max_marks : ℕ
  passing_percentage : ℚ

/-- Calculates the passing marks for an examination. -/
def passing_marks (exam : Examination) : ℚ :=
  (exam.passing_percentage / 100) * exam.max_marks

/-- Theorem: The passing percentage is 45% given the conditions. -/
theorem passing_percentage_is_45_percent 
  (max_marks : ℕ) 
  (failing_score : ℕ) 
  (deficit : ℕ) 
  (h1 : max_marks = 500)
  (h2 : failing_score = 180)
  (h3 : deficit = 45)
  : ∃ (exam : Examination), 
    exam.max_marks = max_marks ∧ 
    exam.passing_percentage = 45 ∧
    passing_marks exam = failing_score + deficit :=
  sorry


end NUMINAMATH_CALUDE_passing_percentage_is_45_percent_l1945_194572


namespace NUMINAMATH_CALUDE_rooms_with_two_windows_l1945_194500

/-- Represents a building with rooms and windows. -/
structure Building where
  total_windows : ℕ
  rooms_with_four : ℕ
  rooms_with_three : ℕ
  rooms_with_two : ℕ

/-- Conditions for the building. -/
def building_conditions (b : Building) : Prop :=
  b.total_windows = 122 ∧
  b.rooms_with_four = 5 ∧
  b.rooms_with_three = 8 ∧
  b.total_windows = 4 * b.rooms_with_four + 3 * b.rooms_with_three + 2 * b.rooms_with_two

/-- Theorem stating the number of rooms with two windows. -/
theorem rooms_with_two_windows (b : Building) :
  building_conditions b → b.rooms_with_two = 39 := by
  sorry

end NUMINAMATH_CALUDE_rooms_with_two_windows_l1945_194500


namespace NUMINAMATH_CALUDE_fifth_term_is_32_l1945_194597

/-- A geometric sequence where all terms are positive and satisfy a_n * a_(n+1) = 2^(2n+1) -/
def special_geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ n, a n * a (n + 1) = 2^(2*n + 1)) ∧
  (∀ n, ∃ q > 0, a (n + 1) = q * a n)

/-- The 5th term of the special geometric sequence is 32 -/
theorem fifth_term_is_32 (a : ℕ → ℝ) (h : special_geometric_sequence a) : 
  a 5 = 32 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_is_32_l1945_194597


namespace NUMINAMATH_CALUDE_square_sum_value_l1945_194532

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) :
  x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l1945_194532


namespace NUMINAMATH_CALUDE_harmonic_mean_inequality_l1945_194534

theorem harmonic_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b > 1 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_inequality_l1945_194534


namespace NUMINAMATH_CALUDE_prob_A_wins_sixth_game_l1945_194547

/-- Represents a player in the coin tossing game -/
inductive Player : Type
| A : Player
| B : Player

/-- Represents the outcome of a single game -/
inductive GameOutcome : Type
| Win : Player → GameOutcome
| Lose : Player → GameOutcome

/-- Represents the state of the game after a certain number of rounds -/
structure GameState :=
  (round : ℕ)
  (last_loser : Player)

/-- The probability of winning a single coin toss -/
def coin_toss_prob : ℚ := 1/2

/-- The probability of a player winning a game given they start first -/
def win_prob_starting (p : Player) : ℚ := coin_toss_prob

/-- The probability of a player winning a game given they start second -/
def win_prob_second (p : Player) : ℚ := 1 - coin_toss_prob

/-- The probability of player A winning the nth game given the initial state -/
def prob_A_wins_nth_game (n : ℕ) (initial_state : GameState) : ℚ :=
  sorry

theorem prob_A_wins_sixth_game :
  prob_A_wins_nth_game 6 ⟨0, Player.B⟩ = 7/30 :=
sorry

end NUMINAMATH_CALUDE_prob_A_wins_sixth_game_l1945_194547


namespace NUMINAMATH_CALUDE_charlie_crayon_count_l1945_194595

/-- The number of crayons each person has -/
structure CrayonCounts where
  billie : ℕ
  bobbie : ℕ
  lizzie : ℕ
  charlie : ℕ

/-- The conditions of the crayon problem -/
def crayon_problem (c : CrayonCounts) : Prop :=
  c.billie = 18 ∧
  c.bobbie = 3 * c.billie ∧
  c.lizzie = c.bobbie / 2 ∧
  c.charlie = 2 * c.lizzie

theorem charlie_crayon_count (c : CrayonCounts) (h : crayon_problem c) : c.charlie = 54 := by
  sorry

end NUMINAMATH_CALUDE_charlie_crayon_count_l1945_194595


namespace NUMINAMATH_CALUDE_rectangle_area_l1945_194518

/-- Proves that the area of a rectangle is 432 square meters, given that its length is thrice its breadth and its perimeter is 96 meters. -/
theorem rectangle_area (b : ℝ) (l : ℝ) : 
  l = 3 * b →                  -- Length is thrice the breadth
  2 * (l + b) = 96 →           -- Perimeter is 96 meters
  l * b = 432 := by            -- Area is 432 square meters
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1945_194518


namespace NUMINAMATH_CALUDE_remaining_flowers_l1945_194587

/-- Represents the flower arrangement along the path --/
structure FlowerPath :=
  (peonies : Nat)
  (tulips : Nat)
  (watered : Nat)
  (unwatered : Nat)
  (picked_tulips : Nat)

/-- Theorem stating the number of remaining flowers after Neznayka's picking --/
theorem remaining_flowers (path : FlowerPath) 
  (h1 : path.peonies = 15)
  (h2 : path.tulips = 15)
  (h3 : path.unwatered = 10)
  (h4 : path.watered + path.unwatered = path.peonies + path.tulips)
  (h5 : path.picked_tulips = 6) :
  path.watered - path.picked_tulips = 19 := by
  sorry

#check remaining_flowers

end NUMINAMATH_CALUDE_remaining_flowers_l1945_194587


namespace NUMINAMATH_CALUDE_largest_triangle_area_l1945_194598

/-- The largest area of a triangle ABC, where A = (2,1), B = (5,3), and C = (p,q) 
    lie on the parabola y = -x^2 + 7x - 10, with 2 ≤ p ≤ 5 -/
theorem largest_triangle_area : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (5, 3)
  let C : ℝ → ℝ × ℝ := λ p => (p, -p^2 + 7*p - 10)
  let triangle_area : ℝ → ℝ := λ p => 
    (1/2) * abs (A.1 * B.2 + B.1 * (C p).2 + (C p).1 * A.2 - 
                 A.2 * B.1 - B.2 * (C p).1 - (C p).2 * A.1)
  ∃ (max_area : ℝ), max_area = 13/8 ∧ 
    ∀ p : ℝ, 2 ≤ p ∧ p ≤ 5 → triangle_area p ≤ max_area :=
by sorry

end NUMINAMATH_CALUDE_largest_triangle_area_l1945_194598


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1945_194537

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 4 -/
theorem equilateral_triangle_area_perimeter_ratio : 
  let side_length : ℝ := 4
  let area : ℝ := side_length^2 * Real.sqrt 3 / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1945_194537


namespace NUMINAMATH_CALUDE_binomial_60_3_l1945_194555

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l1945_194555


namespace NUMINAMATH_CALUDE_stratified_sampling_calculation_l1945_194505

/-- Stratified sampling calculation -/
theorem stratified_sampling_calculation 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (stratum_size : ℕ) 
  (h1 : total_population = 2000) 
  (h2 : sample_size = 200) 
  (h3 : stratum_size = 250) : 
  (stratum_size : ℚ) / total_population * sample_size = 25 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_calculation_l1945_194505


namespace NUMINAMATH_CALUDE_negative_integer_sum_and_square_equals_neg_twelve_l1945_194508

theorem negative_integer_sum_and_square_equals_neg_twelve (N : ℤ) :
  N < 0 → N^2 + N = -12 → N = -3 ∨ N = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_integer_sum_and_square_equals_neg_twelve_l1945_194508


namespace NUMINAMATH_CALUDE_mom_gets_eighteen_strawberries_l1945_194519

def strawberries_for_mom (dozen_picked : ℕ) (eaten : ℕ) : ℕ :=
  dozen_picked * 12 - eaten

theorem mom_gets_eighteen_strawberries :
  strawberries_for_mom 2 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_mom_gets_eighteen_strawberries_l1945_194519


namespace NUMINAMATH_CALUDE_company_problem_solution_l1945_194589

def company_problem (total_employees : ℕ) 
                    (clerical_fraction technical_fraction managerial_fraction : ℚ)
                    (clerical_reduction technical_reduction managerial_reduction : ℚ) : ℚ :=
  let initial_clerical := (clerical_fraction * total_employees : ℚ)
  let initial_technical := (technical_fraction * total_employees : ℚ)
  let initial_managerial := (managerial_fraction * total_employees : ℚ)
  
  let remaining_clerical := initial_clerical * (1 - clerical_reduction)
  let remaining_technical := initial_technical * (1 - technical_reduction)
  let remaining_managerial := initial_managerial * (1 - managerial_reduction)
  
  let total_remaining := remaining_clerical + remaining_technical + remaining_managerial
  
  remaining_clerical / total_remaining

theorem company_problem_solution :
  let result := company_problem 5000 (1/5) (2/5) (2/5) (1/3) (1/4) (1/5)
  ∃ (ε : ℚ), abs (result - 177/1000) < ε ∧ ε < 1/1000 :=
by
  sorry

end NUMINAMATH_CALUDE_company_problem_solution_l1945_194589


namespace NUMINAMATH_CALUDE_absolute_value_of_S_eq_121380_l1945_194510

/-- The sum of all integers b for which x^2 + bx + 2023b can be factored over the integers -/
def S : ℤ := sorry

/-- The polynomial x^2 + bx + 2023b -/
def polynomial (x b : ℤ) : ℤ := x^2 + b*x + 2023*b

/-- Predicate to check if a polynomial can be factored over the integers -/
def is_factorable (b : ℤ) : Prop := ∃ (p q : ℤ → ℤ), ∀ x, polynomial x b = p x * q x

theorem absolute_value_of_S_eq_121380 : |S| = 121380 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_S_eq_121380_l1945_194510


namespace NUMINAMATH_CALUDE_role_assignment_count_l1945_194536

def num_men : ℕ := 6
def num_women : ℕ := 7
def num_male_roles : ℕ := 3
def num_female_roles : ℕ := 3
def num_neutral_roles : ℕ := 2

def total_roles : ℕ := num_male_roles + num_female_roles + num_neutral_roles

theorem role_assignment_count : 
  (num_men.factorial / (num_men - num_male_roles).factorial) *
  (num_women.factorial / (num_women - num_female_roles).factorial) *
  ((num_men + num_women - num_male_roles - num_female_roles).factorial / 
   (num_men + num_women - total_roles).factorial) = 1058400 := by
  sorry

end NUMINAMATH_CALUDE_role_assignment_count_l1945_194536


namespace NUMINAMATH_CALUDE_paving_stone_width_l1945_194577

/-- Given a rectangular courtyard and paving stones with specific dimensions,
    prove that the width of each paving stone is 2 meters. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (total_stones : ℕ)
  (h1 : courtyard_length = 60)
  (h2 : courtyard_width = 16.5)
  (h3 : stone_length = 2.5)
  (h4 : total_stones = 198) :
  ∃ (stone_width : ℝ), stone_width = 2 ∧
    courtyard_length * courtyard_width = stone_length * stone_width * total_stones :=
by sorry

end NUMINAMATH_CALUDE_paving_stone_width_l1945_194577


namespace NUMINAMATH_CALUDE_nates_run_ratio_l1945_194579

theorem nates_run_ratio (total_distance field_length rest_distance : ℕ) 
  (h1 : total_distance = 1172)
  (h2 : field_length = 168)
  (h3 : rest_distance = 500)
  (h4 : ∃ k : ℕ, total_distance - rest_distance = k * field_length) :
  (total_distance - rest_distance) / field_length = 4 := by
sorry

end NUMINAMATH_CALUDE_nates_run_ratio_l1945_194579


namespace NUMINAMATH_CALUDE_circle_area_difference_l1945_194556

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 15
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 843.75 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1945_194556


namespace NUMINAMATH_CALUDE_factor_congruence_l1945_194567

theorem factor_congruence (n : ℕ+) (k : ℕ) :
  k ∣ (2 * n.val)^(2^n.val) + 1 → k ≡ 1 [MOD 2^(n.val + 1)] := by
  sorry

end NUMINAMATH_CALUDE_factor_congruence_l1945_194567


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1945_194538

theorem arithmetic_computation : 8 + 6 * (3 - 8)^2 = 158 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1945_194538


namespace NUMINAMATH_CALUDE_students_count_l1945_194516

/-- The number of seats on each school bus -/
def seats_per_bus : ℕ := 2

/-- The number of buses needed for the trip -/
def buses_needed : ℕ := 7

/-- The number of students going on the field trip -/
def students_on_trip : ℕ := seats_per_bus * buses_needed

theorem students_count : students_on_trip = 14 := by
  sorry

end NUMINAMATH_CALUDE_students_count_l1945_194516


namespace NUMINAMATH_CALUDE_road_trip_distances_l1945_194575

theorem road_trip_distances (total_distance : ℕ) 
  (tracy_distance michelle_distance katie_distance : ℕ) : 
  total_distance = 1000 →
  tracy_distance = 2 * michelle_distance + 20 →
  michelle_distance = 3 * katie_distance →
  tracy_distance + michelle_distance + katie_distance = total_distance →
  michelle_distance = 294 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_distances_l1945_194575


namespace NUMINAMATH_CALUDE_training_completion_time_l1945_194504

/-- Calculates the number of days required to complete a training regimen. -/
def trainingDays (totalHours : ℕ) (multiplicationMinutes : ℕ) (divisionMinutes : ℕ) : ℕ :=
  let totalMinutes := totalHours * 60
  let dailyMinutes := multiplicationMinutes + divisionMinutes
  totalMinutes / dailyMinutes

/-- Proves that given the specified training schedule, it takes 10 days to complete the training. -/
theorem training_completion_time :
  trainingDays 5 10 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_training_completion_time_l1945_194504


namespace NUMINAMATH_CALUDE_arithmetic_mean_square_difference_l1945_194506

theorem arithmetic_mean_square_difference (p u v : ℕ) : 
  Nat.Prime p → 
  u ≠ v → 
  u > 0 → 
  v > 0 → 
  p * p = (u * u + v * v) / 2 → 
  ∃ (x : ℕ), (2 * p - u - v = x * x) ∨ (2 * p - u - v = 2 * x * x) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_square_difference_l1945_194506


namespace NUMINAMATH_CALUDE_cos_arcsin_plus_arccos_l1945_194521

theorem cos_arcsin_plus_arccos : 
  Real.cos (Real.arcsin (3/5) + Real.arccos (-5/13)) = -56/65 := by sorry

end NUMINAMATH_CALUDE_cos_arcsin_plus_arccos_l1945_194521


namespace NUMINAMATH_CALUDE_lunch_combo_options_count_l1945_194548

/-- The number of lunch combo options for Terry at the salad bar -/
def lunch_combo_options : ℕ :=
  let lettuce_types : ℕ := 2
  let tomato_types : ℕ := 3
  let olive_types : ℕ := 4
  let soup_types : ℕ := 2
  lettuce_types * tomato_types * olive_types * soup_types

theorem lunch_combo_options_count : lunch_combo_options = 48 := by
  sorry

end NUMINAMATH_CALUDE_lunch_combo_options_count_l1945_194548


namespace NUMINAMATH_CALUDE_initial_amount_of_liquid_A_solution_is_correct_l1945_194573

/-- Given a mixture of liquids A and B, this theorem proves the initial amount of liquid A. -/
theorem initial_amount_of_liquid_A
  (initial_ratio : ℚ) -- Initial ratio of A to B
  (replacement_volume : ℚ) -- Volume of mixture replaced with B
  (final_ratio : ℚ) -- Final ratio of A to B
  (h1 : initial_ratio = 4 / 1)
  (h2 : replacement_volume = 40)
  (h3 : final_ratio = 2 / 3)
  : ℚ :=
by
  sorry

#check initial_amount_of_liquid_A

/-- The solution to the problem -/
def solution : ℚ := 32

/-- Proof that the solution is correct -/
theorem solution_is_correct :
  initial_amount_of_liquid_A (4 / 1) 40 (2 / 3) rfl rfl rfl = solution :=
by
  sorry

end NUMINAMATH_CALUDE_initial_amount_of_liquid_A_solution_is_correct_l1945_194573


namespace NUMINAMATH_CALUDE_paper_supply_duration_l1945_194576

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of stories John writes per week -/
def stories_per_week : ℕ := 3

/-- The number of pages in each short story -/
def pages_per_story : ℕ := 50

/-- The number of pages in John's yearly novel -/
def novel_pages_per_year : ℕ := 1200

/-- The number of pages that can fit on one sheet of paper -/
def pages_per_sheet : ℕ := 2

/-- The number of reams of paper John buys -/
def reams_bought : ℕ := 3

/-- The number of sheets in each ream of paper -/
def sheets_per_ream : ℕ := 500

/-- The number of weeks John is buying paper for -/
def weeks_of_paper_supply : ℕ := 18

theorem paper_supply_duration :
  let total_pages_per_week := stories_per_week * pages_per_story + novel_pages_per_year / weeks_per_year
  let sheets_per_week := (total_pages_per_week + pages_per_sheet - 1) / pages_per_sheet
  let total_sheets := reams_bought * sheets_per_ream
  (total_sheets + sheets_per_week - 1) / sheets_per_week = weeks_of_paper_supply :=
by sorry

end NUMINAMATH_CALUDE_paper_supply_duration_l1945_194576


namespace NUMINAMATH_CALUDE_final_value_of_A_l1945_194527

theorem final_value_of_A (A : Int) : A = 20 → -A + 10 = -10 := by
  sorry

end NUMINAMATH_CALUDE_final_value_of_A_l1945_194527


namespace NUMINAMATH_CALUDE_common_root_equations_l1945_194507

theorem common_root_equations (k : ℝ) :
  (∃ x : ℝ, x^2 - k*x - 7 = 0 ∧ x^2 - 6*x - (k + 1) = 0) →
  (k = -6 ∧
   (∃ x : ℝ, x^2 + 6*x - 7 = 0 ∧ x^2 - 6*x + 5 = 0 ∧ x = 1) ∧
   (∃ y z : ℝ, y^2 + 6*y - 7 = 0 ∧ z^2 - 6*z + 5 = 0 ∧ y = -7 ∧ z = 5)) :=
by sorry

end NUMINAMATH_CALUDE_common_root_equations_l1945_194507


namespace NUMINAMATH_CALUDE_bird_cage_problem_l1945_194560

theorem bird_cage_problem (B : ℚ) : 
  (B > 0) →                         -- Ensure positive number of birds
  (B * (2/3) * (3/5) * (1/3) = 60)  -- Remaining birds after three stages equal 60
  ↔ 
  (B = 450) :=                      -- Total initial number of birds is 450
by sorry

end NUMINAMATH_CALUDE_bird_cage_problem_l1945_194560


namespace NUMINAMATH_CALUDE_f_is_fraction_l1945_194580

-- Define what a fraction is
def is_fraction (f : ℚ → ℚ) : Prop :=
  ∃ (a b : ℚ), ∀ x, b ≠ 0 → f x = a / b

-- Define the specific function we're proving is a fraction
def f (x : ℚ) : ℚ := x / (x + 2)

-- Theorem statement
theorem f_is_fraction : is_fraction f := by sorry

end NUMINAMATH_CALUDE_f_is_fraction_l1945_194580


namespace NUMINAMATH_CALUDE_remainder_sum_l1945_194591

theorem remainder_sum (a b : ℤ) (h1 : a % 60 = 49) (h2 : b % 40 = 29) : (a + b) % 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1945_194591
