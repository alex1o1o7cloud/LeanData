import Mathlib

namespace units_digit_of_p_squared_plus_3_to_p_l551_55131

theorem units_digit_of_p_squared_plus_3_to_p (p : ℕ) : 
  p = 2017^3 + 3^2017 → (p^2 + 3^p) % 10 = 5 := by
sorry

end units_digit_of_p_squared_plus_3_to_p_l551_55131


namespace solve_snake_problem_l551_55106

def snake_problem (total_length : ℝ) (head_ratio : ℝ) : Prop :=
  let head_length := head_ratio * total_length
  let body_length := total_length - head_length
  (head_ratio = 1 / 10) ∧ (total_length = 10) → body_length = 9

theorem solve_snake_problem :
  snake_problem 10 (1 / 10) :=
by sorry

end solve_snake_problem_l551_55106


namespace log_meaningful_iff_in_range_l551_55119

def meaningful_log (a : ℝ) : Prop :=
  a - 2 > 0 ∧ a - 2 ≠ 1 ∧ 5 - a > 0

theorem log_meaningful_iff_in_range (a : ℝ) :
  meaningful_log a ↔ (a > 2 ∧ a < 3) ∨ (a > 3 ∧ a < 5) :=
sorry

end log_meaningful_iff_in_range_l551_55119


namespace log_equation_solution_l551_55136

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 3 + Real.log 3 / Real.log x = 2 → x = 3 := by
  sorry

end log_equation_solution_l551_55136


namespace factor_tree_value_l551_55158

/-- Given a factor tree with the following relationships:
  X = Y * Z
  Y = 7 * F
  Z = 11 * G
  F = 7 * 2
  G = 3 * 2
  Prove that X = 12936 -/
theorem factor_tree_value (X Y Z F G : ℕ) 
  (h1 : X = Y * Z)
  (h2 : Y = 7 * F)
  (h3 : Z = 11 * G)
  (h4 : F = 7 * 2)
  (h5 : G = 3 * 2) : 
  X = 12936 := by
  sorry

#check factor_tree_value

end factor_tree_value_l551_55158


namespace mixture_ratio_change_l551_55128

/-- Given an initial mixture of milk and water, prove the new ratio after adding water -/
theorem mixture_ratio_change (initial_volume : ℚ) (initial_milk_ratio : ℚ) (initial_water_ratio : ℚ) 
  (added_water : ℚ) (new_milk_ratio : ℚ) (new_water_ratio : ℚ) : 
  initial_volume = 60 ∧ 
  initial_milk_ratio = 2 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 60 ∧ 
  new_milk_ratio = 1 ∧ 
  new_water_ratio = 2 →
  let initial_milk := (initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let initial_water := (initial_water_ratio / (initial_milk_ratio + initial_water_ratio)) * initial_volume
  let new_water := initial_water + added_water
  new_milk_ratio / new_water_ratio = initial_milk / new_water :=
by
  sorry


end mixture_ratio_change_l551_55128


namespace james_chores_time_l551_55115

/-- Proves that James spends 12 hours on his chores given the conditions -/
theorem james_chores_time (vacuum_time : ℝ) (other_chores_factor : ℝ) : 
  vacuum_time = 3 →
  other_chores_factor = 3 →
  vacuum_time + (other_chores_factor * vacuum_time) = 12 := by
  sorry

end james_chores_time_l551_55115


namespace parkway_elementary_girls_not_playing_soccer_l551_55168

theorem parkway_elementary_girls_not_playing_soccer 
  (total_students : ℕ) 
  (total_boys : ℕ) 
  (total_soccer_players : ℕ) 
  (boys_soccer_percentage : ℚ) :
  total_students = 450 →
  total_boys = 320 →
  total_soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  ∃ (girls_not_playing_soccer : ℕ), 
    girls_not_playing_soccer = 
      total_students - total_boys - 
      (total_soccer_players - (boys_soccer_percentage * total_soccer_players).floor) :=
by
  sorry

end parkway_elementary_girls_not_playing_soccer_l551_55168


namespace remaining_squares_after_removal_l551_55155

/-- Represents the initial arrangement of matchsticks -/
structure Arrangement where
  matchsticks : ℕ
  squares : ℕ

/-- Represents a claim about the arrangement after removing matchsticks -/
inductive Claim
  | A : Claim  -- 5 squares of size 1x1 remain
  | B : Claim  -- 3 squares of size 2x2 remain
  | C : Claim  -- All 3x3 squares remain
  | D : Claim  -- Removed matchsticks are all on different lines
  | E : Claim  -- Four of the removed matchsticks are on the same line

/-- The main theorem to be proved -/
theorem remaining_squares_after_removal 
  (initial : Arrangement)
  (removed : ℕ)
  (incorrect_claims : Finset Claim)
  (h1 : initial.matchsticks = 40)
  (h2 : initial.squares = 30)
  (h3 : removed = 5)
  (h4 : incorrect_claims.card = 2)
  (h5 : Claim.A ∈ incorrect_claims)
  (h6 : Claim.D ∈ incorrect_claims)
  (h7 : Claim.E ∉ incorrect_claims)
  (h8 : Claim.B ∉ incorrect_claims)
  (h9 : Claim.C ∉ incorrect_claims) :
  ∃ (final : Arrangement), final.squares = 28 :=
sorry

end remaining_squares_after_removal_l551_55155


namespace lost_ship_depth_l551_55125

/-- The depth of a lost ship given the diver's descent rate and time taken --/
theorem lost_ship_depth (descent_rate : ℝ) (time_taken : ℝ) (h1 : descent_rate = 35) (h2 : time_taken = 100) :
  descent_rate * time_taken = 3500 := by
  sorry

end lost_ship_depth_l551_55125


namespace prime_pythagorean_inequality_l551_55183

theorem prime_pythagorean_inequality (p m n : ℕ) 
  (prime_p : Nat.Prime p) 
  (pos_m : m > 0) 
  (pos_n : n > 0) 
  (pyth_eq : p^2 + m^2 = n^2) : 
  m > p := by
sorry

end prime_pythagorean_inequality_l551_55183


namespace pictures_on_sixth_day_l551_55167

def artists_group1 : ℕ := 6
def artists_group2 : ℕ := 8
def days_interval1 : ℕ := 2
def days_interval2 : ℕ := 3
def days_observed : ℕ := 5
def pictures_in_5_days : ℕ := 30

theorem pictures_on_sixth_day :
  let total_6_days := artists_group1 * (6 / days_interval1) + artists_group2 * (6 / days_interval2)
  (total_6_days - pictures_in_5_days : ℕ) = 4 := by
  sorry

end pictures_on_sixth_day_l551_55167


namespace aladdin_gold_bars_l551_55152

theorem aladdin_gold_bars (x : ℕ) : 
  (x + 1023000) / 1024 ≤ x := by sorry

end aladdin_gold_bars_l551_55152


namespace geometric_mean_minimum_l551_55176

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : 2 = Real.sqrt (4^a * 2^b)) :
  (2/a + 1/b) ≥ 9/2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    2 = Real.sqrt (4^a₀ * 2^b₀) ∧ (2/a₀ + 1/b₀) = 9/2 := by
  sorry

end geometric_mean_minimum_l551_55176


namespace apartments_greater_than_scales_l551_55110

theorem apartments_greater_than_scales (houses : ℕ) (K A P C : ℕ) :
  houses > 0 ∧ K > 0 ∧ A > 0 ∧ P > 0 ∧ C > 0 →  -- All quantities are positive
  K * A * P > A * P * C →                      -- Fish in house > scales in apartment
  K > C                                        -- Apartments in house > scales on fish
  := by sorry

end apartments_greater_than_scales_l551_55110


namespace no_equal_conversions_l551_55104

def fahrenheit_to_celsius (f : ℤ) : ℤ :=
  ⌊(5 : ℚ) / 9 * (f - 32)⌋

def celsius_to_fahrenheit (c : ℤ) : ℤ :=
  ⌊(9 : ℚ) / 5 * c + 33⌋

theorem no_equal_conversions :
  ∀ f : ℤ, 34 ≤ f ∧ f ≤ 1024 →
    f ≠ celsius_to_fahrenheit (fahrenheit_to_celsius f) :=
by sorry

end no_equal_conversions_l551_55104


namespace sufficient_not_necessary_l551_55189

theorem sufficient_not_necessary (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ a ≤ 1) := by
  sorry

end sufficient_not_necessary_l551_55189


namespace percentage_calculation_l551_55157

theorem percentage_calculation (total : ℝ) (part : ℝ) (h1 : total = 600) (h2 : part = 150) :
  (part / total) * 100 = 25 := by
sorry

end percentage_calculation_l551_55157


namespace largest_four_digit_binary_is_15_l551_55116

/-- A binary digit is either 0 or 1 -/
def BinaryDigit : Type := {n : Nat // n = 0 ∨ n = 1}

/-- A four-digit binary number -/
def FourDigitBinary : Type := BinaryDigit × BinaryDigit × BinaryDigit × BinaryDigit

/-- Convert a four-digit binary number to its decimal representation -/
def binaryToDecimal (b : FourDigitBinary) : Nat :=
  b.1.val * 8 + b.2.1.val * 4 + b.2.2.1.val * 2 + b.2.2.2.val

/-- The largest four-digit binary number -/
def largestFourDigitBinary : FourDigitBinary :=
  (⟨1, Or.inr rfl⟩, ⟨1, Or.inr rfl⟩, ⟨1, Or.inr rfl⟩, ⟨1, Or.inr rfl⟩)

theorem largest_four_digit_binary_is_15 :
  binaryToDecimal largestFourDigitBinary = 15 := by
  sorry

#eval binaryToDecimal largestFourDigitBinary

end largest_four_digit_binary_is_15_l551_55116


namespace max_right_angles_in_triangle_l551_55161

theorem max_right_angles_in_triangle : ℕ :=
  -- Define the sum of angles in a triangle
  let sum_of_angles : ℝ := 180

  -- Define a right angle in degrees
  let right_angle : ℝ := 90

  -- Define the maximum number of right angles
  let max_right_angles : ℕ := 1

  -- Theorem statement
  max_right_angles

end max_right_angles_in_triangle_l551_55161


namespace operation_equations_l551_55151

theorem operation_equations :
  (37.3 / (1/2) = 74 + 3/5) ∧
  (33/40 * 10/11 = 0.75) ∧
  (0.45 - 1/20 = 2/5) ∧
  (0.375 + 1/40 = 0.4) := by
sorry

end operation_equations_l551_55151


namespace megan_books_count_l551_55120

theorem megan_books_count :
  ∀ (m k g : ℕ),
  k = m / 4 →
  g = 2 * k + 9 →
  m + k + g = 65 →
  m = 32 :=
by sorry

end megan_books_count_l551_55120


namespace three_digit_ends_in_five_divisible_by_five_l551_55135

/-- A three-digit positive integer -/
def ThreeDigitInt (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem three_digit_ends_in_five_divisible_by_five :
  ∀ N : ℕ, ThreeDigitInt N → onesDigit N = 5 → 
    (∃ k : ℕ, N = 5 * k) := by sorry

end three_digit_ends_in_five_divisible_by_five_l551_55135


namespace magical_stack_size_l551_55174

/-- A magical stack is a stack of cards with the following properties:
  1. There are 3n cards numbered consecutively from 1 to 3n.
  2. Cards are divided into three piles A, B, and C, each with n cards.
  3. Cards are restacked alternately from C, B, and A.
  4. At least one card from each pile occupies its original position after restacking.
  5. Card number 101 retains its original position. -/
structure MagicalStack (n : ℕ) :=
  (total_cards : ℕ := 3 * n)
  (pile_size : ℕ := n)
  (card_101_position : ℕ)
  (is_magical : Bool)
  (card_101_retained : Bool)

/-- The theorem states that for a magical stack where card 101 retains its position,
    the total number of cards is 303. -/
theorem magical_stack_size (stack : MagicalStack n) 
  (h1 : stack.is_magical = true) 
  (h2 : stack.card_101_retained = true) 
  (h3 : stack.card_101_position = 101) :
  stack.total_cards = 303 :=
sorry

end magical_stack_size_l551_55174


namespace reverse_sum_divisibility_l551_55165

def reverse_number (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem reverse_sum_divisibility (n : ℕ) (m : ℕ) (h1 : n ≥ 10^(m-1)) (h2 : n < 10^m) :
  (81 ∣ (n + reverse_number n)) ↔ (81 ∣ sum_of_digits n) := by sorry

end reverse_sum_divisibility_l551_55165


namespace sqrt_50_between_consecutive_integers_product_l551_55195

theorem sqrt_50_between_consecutive_integers_product : ∃ n : ℕ, 
  (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 := by
  sorry

end sqrt_50_between_consecutive_integers_product_l551_55195


namespace q_must_be_true_l551_55100

theorem q_must_be_true (h1 : ¬p) (h2 : p ∨ q) : q :=
sorry

end q_must_be_true_l551_55100


namespace not_q_is_true_l551_55107

theorem not_q_is_true (p q : Prop) (hp : p) (hq : ¬q) : ¬q := by
  sorry

end not_q_is_true_l551_55107


namespace green_sweets_count_l551_55108

theorem green_sweets_count (total : ℕ) (red : ℕ) (neither : ℕ) (h1 : total = 285) (h2 : red = 49) (h3 : neither = 177) :
  total - red - neither = 59 := by
  sorry

end green_sweets_count_l551_55108


namespace domain_of_f_squared_l551_55129

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x - 2)
def domain_f_shifted : Set ℝ := Set.Icc 0 3

-- State the theorem
theorem domain_of_f_squared (h : ∀ x ∈ domain_f_shifted, f (x - 2) = f (x - 2)) :
  {x : ℝ | ∃ y, f (y^2) = x} = Set.Icc (-1) 1 := by sorry

end domain_of_f_squared_l551_55129


namespace one_absent_one_present_probability_l551_55101

/-- The probability of a student being absent on any given day -/
def absent_prob : ℚ := 1 / 30

/-- The probability of a student being present on any given day -/
def present_prob : ℚ := 1 - absent_prob

/-- The probability that out of two randomly chosen students, exactly one is absent while the other is present -/
def one_absent_one_present_prob : ℚ := 2 * (absent_prob * present_prob)

theorem one_absent_one_present_probability :
  one_absent_one_present_prob = 58 / 900 := by sorry

end one_absent_one_present_probability_l551_55101


namespace line_circle_intersection_l551_55123

/-- The line equation x*cos(θ) + y*sin(θ) + a = 0 intersects the circle x^2 + y^2 = a^2 at exactly one point -/
theorem line_circle_intersection (θ a : ℝ) :
  ∃! p : ℝ × ℝ, 
    (p.1 * Real.cos θ + p.2 * Real.sin θ + a = 0) ∧ 
    (p.1^2 + p.2^2 = a^2) := by
  sorry

end line_circle_intersection_l551_55123


namespace venus_hall_rental_cost_l551_55146

theorem venus_hall_rental_cost (caesars_rental : ℕ) (caesars_meal : ℕ) (venus_meal : ℕ) (guests : ℕ) :
  caesars_rental = 800 →
  caesars_meal = 30 →
  venus_meal = 35 →
  guests = 60 →
  ∃ venus_rental : ℕ, venus_rental = 500 ∧ 
    caesars_rental + guests * caesars_meal = venus_rental + guests * venus_meal :=
by sorry

end venus_hall_rental_cost_l551_55146


namespace max_cylinder_surface_area_in_sphere_l551_55154

/-- The maximum surface area of a cylinder inscribed in a sphere -/
theorem max_cylinder_surface_area_in_sphere (R : ℝ) (h_pos : R > 0) :
  ∃ (r h : ℝ),
    r > 0 ∧ h > 0 ∧
    R^2 = r^2 + (h/2)^2 ∧
    ∀ (r' h' : ℝ),
      r' > 0 → h' > 0 → R^2 = r'^2 + (h'/2)^2 →
      2 * π * r * (h + r) ≤ 2 * π * r' * (h' + r') →
      2 * π * r * (h + r) = R^2 * π * (1 + Real.sqrt 5) :=
sorry

end max_cylinder_surface_area_in_sphere_l551_55154


namespace functional_equation_solutions_l551_55182

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)

/-- The theorem stating that functions satisfying the equation are either constant zero or square. -/
theorem functional_equation_solutions (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x^2) := by
  sorry

end functional_equation_solutions_l551_55182


namespace grid_solution_l551_55181

/-- A 3x3 grid represented as a function from (Fin 3 × Fin 3) to ℕ -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if two cells are adjacent in the grid -/
def adjacent (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ b.2.val + 1 = a.2.val)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ b.1.val + 1 = a.1.val))

/-- The given grid with known values -/
def given_grid : Grid :=
  fun i j =>
    if i = 0 ∧ j = 1 then 1
    else if i = 0 ∧ j = 2 then 9
    else if i = 1 ∧ j = 0 then 3
    else if i = 1 ∧ j = 1 then 5
    else if i = 2 ∧ j = 2 then 7
    else 0  -- placeholder for unknown values

theorem grid_solution :
  ∀ g : Grid,
  (∀ i j, g i j ∈ Finset.range 10) →  -- all numbers are from 1 to 9
  (∀ a b, adjacent a b → g a.1 a.2 + g b.1 b.2 < 12) →  -- sum of adjacent cells < 12
  (∀ i j, given_grid i j ≠ 0 → g i j = given_grid i j) →  -- matches given values
  g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 :=
by sorry

end grid_solution_l551_55181


namespace largest_interesting_number_l551_55179

def is_interesting (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length ≥ 3 ∧
  ∀ i, 1 < i ∧ i < digits.length - 1 →
    digits[i]! < (digits[i-1]! + digits[i+1]!) / 2

theorem largest_interesting_number :
  (∀ m : ℕ, is_interesting m → m ≤ 96433469) ∧ is_interesting 96433469 := by
  sorry

end largest_interesting_number_l551_55179


namespace max_removed_squares_elegantly_destroyed_l551_55150

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)
  (removed : Finset (Nat × Nat))

/-- Represents a domino --/
inductive Domino
  | Horizontal : Nat → Nat → Domino
  | Vertical : Nat → Nat → Domino

/-- Checks if a domino can be placed on the board --/
def canPlaceDomino (board : Chessboard) (d : Domino) : Prop :=
  match d with
  | Domino.Horizontal x y => 
      x < board.size ∧ y < board.size - 1 ∧ 
      (x, y) ∉ board.removed ∧ (x, y + 1) ∉ board.removed
  | Domino.Vertical x y => 
      x < board.size - 1 ∧ y < board.size ∧ 
      (x, y) ∉ board.removed ∧ (x + 1, y) ∉ board.removed

/-- Defines an "elegantly destroyed" board --/
def isElegantlyDestroyed (board : Chessboard) : Prop :=
  (∀ d : Domino, ¬canPlaceDomino board d) ∧
  (∀ s : Nat × Nat, s ∈ board.removed →
    ∃ d : Domino, canPlaceDomino { size := board.size, removed := board.removed.erase s } d)

/-- The main theorem --/
theorem max_removed_squares_elegantly_destroyed :
  ∃ (board : Chessboard),
    board.size = 8 ∧
    isElegantlyDestroyed board ∧
    board.removed.card = 48 ∧
    (∀ (board' : Chessboard), board'.size = 8 →
      isElegantlyDestroyed board' →
      board'.removed.card ≤ 48) :=
  sorry

end max_removed_squares_elegantly_destroyed_l551_55150


namespace total_students_in_halls_l551_55162

theorem total_students_in_halls (general : ℕ) (biology : ℕ) (math : ℕ) : 
  general = 30 ∧ 
  biology = 2 * general ∧ 
  math = (3 * (general + biology)) / 5 → 
  general + biology + math = 144 :=
by sorry

end total_students_in_halls_l551_55162


namespace dodecahedral_die_expected_value_l551_55138

/-- A fair dodecahedral die with faces numbered from 1 to 12 -/
def dodecahedral_die := Finset.range 12

/-- The probability of each outcome for a fair die -/
def prob (n : ℕ) : ℚ := 1 / 12

/-- The expected value of rolling the dodecahedral die -/
def expected_value : ℚ := (dodecahedral_die.sum fun i => (i + 1 : ℚ) * prob i) / 1

/-- Theorem: The expected value of rolling a fair dodecahedral die is 6.5 -/
theorem dodecahedral_die_expected_value : expected_value = 13/2 := by sorry

end dodecahedral_die_expected_value_l551_55138


namespace correct_non_attacking_placements_non_attacking_placements_positive_l551_55127

/-- Represents a chess piece type -/
inductive ChessPiece
  | Rook
  | King
  | Bishop
  | Knight
  | Queen

/-- Represents the dimensions of a chessboard -/
def BoardSize : Nat := 8

/-- Calculates the number of ways to place two pieces of the same type on a chessboard without attacking each other -/
def nonAttackingPlacements (piece : ChessPiece) : Nat :=
  match piece with
  | ChessPiece.Rook => 1568
  | ChessPiece.King => 1806
  | ChessPiece.Bishop => 1736
  | ChessPiece.Knight => 1848
  | ChessPiece.Queen => 1288

/-- Theorem stating the correct number of non-attacking placements for each piece type -/
theorem correct_non_attacking_placements :
  (nonAttackingPlacements ChessPiece.Rook = 1568) ∧
  (nonAttackingPlacements ChessPiece.King = 1806) ∧
  (nonAttackingPlacements ChessPiece.Bishop = 1736) ∧
  (nonAttackingPlacements ChessPiece.Knight = 1848) ∧
  (nonAttackingPlacements ChessPiece.Queen = 1288) := by
  sorry

/-- Theorem stating that the number of non-attacking placements is always positive -/
theorem non_attacking_placements_positive (piece : ChessPiece) :
  nonAttackingPlacements piece > 0 := by
  sorry

end correct_non_attacking_placements_non_attacking_placements_positive_l551_55127


namespace triangle_side_length_l551_55105

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  (a + b + 10 * c = 2 * (Real.sin A + Real.sin B + 10 * Real.sin C)) →
  (A = π / 3) →
  (a = Real.sqrt 3) :=
by sorry

end triangle_side_length_l551_55105


namespace median_in_70_74_interval_l551_55163

/-- Represents the frequency of scores in each interval -/
structure ScoreFrequency where
  interval : ℕ × ℕ
  count : ℕ

/-- The list of score frequencies for the test -/
def scoreDistribution : List ScoreFrequency := [
  ⟨(80, 84), 16⟩,
  ⟨(75, 79), 12⟩,
  ⟨(70, 74), 6⟩,
  ⟨(65, 69), 3⟩,
  ⟨(60, 64), 2⟩,
  ⟨(55, 59), 20⟩,
  ⟨(50, 54), 22⟩
]

/-- The total number of students -/
def totalStudents : ℕ := 81

/-- The position of the median in the ordered list of scores -/
def medianPosition : ℕ := (totalStudents + 1) / 2

/-- Function to find the interval containing the median score -/
def findMedianInterval (distribution : List ScoreFrequency) (medianPos : ℕ) : ℕ × ℕ :=
  sorry

/-- Theorem stating that the median score is in the interval 70-74 -/
theorem median_in_70_74_interval :
  findMedianInterval scoreDistribution medianPosition = (70, 74) := by
  sorry

end median_in_70_74_interval_l551_55163


namespace square_sum_equals_one_l551_55164

theorem square_sum_equals_one (a b : ℝ) (h : a = 1 - b) : a^2 + 2*a*b + b^2 = 1 := by
  sorry

end square_sum_equals_one_l551_55164


namespace correct_ages_are_valid_correct_ages_are_unique_l551_55117

/-- Represents the ages of a family in 1978 -/
structure FamilyAges where
  son : Nat
  daughter : Nat
  mother : Nat
  father : Nat

/-- Checks if the given ages satisfy the problem conditions -/
def validAges (ages : FamilyAges) : Prop :=
  ages.son < 21 ∧
  ages.daughter < 21 ∧
  ages.son ≠ ages.daughter ∧
  ages.father = ages.mother + 8 ∧
  ages.son^3 + ages.daughter^2 > 1900 ∧
  ages.son^3 + ages.daughter^2 < 1978 ∧
  ages.son^3 + ages.daughter^2 + ages.father = 1978

/-- The correct ages of the family members -/
def correctAges : FamilyAges :=
  { son := 12
  , daughter := 14
  , mother := 46
  , father := 54 }

/-- Theorem stating that the correct ages satisfy the problem conditions -/
theorem correct_ages_are_valid : validAges correctAges := by
  sorry

/-- Theorem stating that the correct ages are the only solution -/
theorem correct_ages_are_unique : ∀ ages : FamilyAges, validAges ages → ages = correctAges := by
  sorry

end correct_ages_are_valid_correct_ages_are_unique_l551_55117


namespace jake_snakes_l551_55178

/-- The number of eggs each snake lays -/
def eggs_per_snake : ℕ := 2

/-- The price of a regular baby snake in dollars -/
def regular_price : ℕ := 250

/-- The price of the rare baby snake in dollars -/
def rare_price : ℕ := 4 * regular_price

/-- The total amount Jake received from selling the snakes in dollars -/
def total_revenue : ℕ := 2250

/-- The number of snakes Jake has -/
def num_snakes : ℕ := 3

theorem jake_snakes :
  num_snakes * eggs_per_snake * regular_price + (rare_price - regular_price) = total_revenue :=
sorry

end jake_snakes_l551_55178


namespace playground_dimensions_l551_55142

theorem playground_dimensions :
  ∃! n : ℕ, n = (Finset.filter (fun pair : ℕ × ℕ =>
    pair.2 > pair.1 ∧
    (pair.1 - 4) * (pair.2 - 4) = 2 * pair.1 * pair.2 / 3
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ n = 3 := by
  sorry

end playground_dimensions_l551_55142


namespace negation_of_existence_power_of_two_l551_55124

theorem negation_of_existence_power_of_two (p : Prop) : 
  (p ↔ ∃ n : ℕ, 2^n > 1000) → 
  (¬p ↔ ∀ n : ℕ, 2^n ≤ 1000) :=
by sorry

end negation_of_existence_power_of_two_l551_55124


namespace roden_gold_fish_l551_55133

/-- The number of fish Roden bought in total -/
def total_fish : ℕ := 22

/-- The number of blue fish Roden bought -/
def blue_fish : ℕ := 7

/-- The number of gold fish Roden bought -/
def gold_fish : ℕ := total_fish - blue_fish

theorem roden_gold_fish : gold_fish = 15 := by
  sorry

end roden_gold_fish_l551_55133


namespace derivative_of_periodic_is_periodic_l551_55173

/-- A function f: ℝ → ℝ is periodic with period T if f(x + T) = f(x) for all x ∈ ℝ -/
def IsPeriodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem derivative_of_periodic_is_periodic
  (f : ℝ → ℝ) (f' : ℝ → ℝ) (T : ℝ) (hT : T ≠ 0)
  (hf : Differentiable ℝ f)
  (hf' : ∀ x, HasDerivAt f (f' x) x)
  (hperiodic : IsPeriodic f T) :
  IsPeriodic f' T :=
sorry

end derivative_of_periodic_is_periodic_l551_55173


namespace gasoline_tank_capacity_l551_55199

theorem gasoline_tank_capacity :
  ∀ x : ℚ,
  (5/6 : ℚ) * x - (2/3 : ℚ) * x = 15 →
  x = 90 :=
by
  sorry

end gasoline_tank_capacity_l551_55199


namespace inequality_proof_l551_55191

theorem inequality_proof (x y z : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) (hz_pos : z > 0)
  (hx_bound : x < 2) (hy_bound : y < 2) (hz_bound : z < 2)
  (h_sum : x^2 + y^2 + z^2 = 3) : 
  (3/2 : ℝ) < (1+y^2)/(x+2) + (1+z^2)/(y+2) + (1+x^2)/(z+2) ∧ 
  (1+y^2)/(x+2) + (1+z^2)/(y+2) + (1+x^2)/(z+2) < 3 := by
  sorry

end inequality_proof_l551_55191


namespace age_difference_l551_55139

theorem age_difference (albert_age mary_age betty_age : ℕ) : 
  albert_age = 2 * mary_age →
  albert_age = 4 * betty_age →
  betty_age = 7 →
  albert_age - mary_age = 14 := by
  sorry

end age_difference_l551_55139


namespace consecutive_composites_exist_l551_55118

/-- A natural number is composite if it has more than two distinct positive divisors -/
def IsComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

/-- A sequence of n consecutive composite numbers starting from k -/
def ConsecutiveComposites (k n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → IsComposite (k + i)

theorem consecutive_composites_exist :
  (∃ k : ℕ, k ≤ 500 ∧ ConsecutiveComposites k 9) ∧
  (∃ k : ℕ, k ≤ 500 ∧ ConsecutiveComposites k 11) := by
  sorry

end consecutive_composites_exist_l551_55118


namespace problem_statement_l551_55137

theorem problem_statement (a b x y : ℝ) 
  (h1 : a + b = 2) 
  (h2 : x + y = 3) 
  (h3 : a * x + b * y = 4) : 
  (a^2 + b^2) * x * y + a * b * (x^2 + y^2) = 8 := by
  sorry

end problem_statement_l551_55137


namespace jovanas_shells_l551_55144

theorem jovanas_shells (x : ℝ) : x + 15 + 17 = 37 → x = 5 := by
  sorry

end jovanas_shells_l551_55144


namespace socks_cost_l551_55169

theorem socks_cost (num_players : ℕ) (jersey_cost shorts_cost total_cost : ℚ) : 
  num_players = 16 →
  jersey_cost = 25 →
  shorts_cost = 15.20 →
  total_cost = 752 →
  ∃ (socks_cost : ℚ), 
    num_players * (jersey_cost + shorts_cost + socks_cost) = total_cost ∧ 
    socks_cost = 6.80 := by
  sorry

end socks_cost_l551_55169


namespace kristin_reads_half_l551_55143

/-- The number of books Peter and Kristin need to read -/
def total_books : ℕ := 20

/-- Peter's reading speed in hours per book -/
def peter_speed : ℚ := 18

/-- The ratio of Kristin's reading speed to Peter's -/
def speed_ratio : ℚ := 3

/-- The time Kristin has to read in hours -/
def kristin_time : ℚ := 540

/-- The portion of books Kristin reads in the given time -/
def kristin_portion : ℚ := kristin_time / (peter_speed * speed_ratio * total_books)

theorem kristin_reads_half :
  kristin_portion = 1 / 2 := by sorry

end kristin_reads_half_l551_55143


namespace sum_of_products_l551_55175

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 75)
  (eq2 : y^2 + y*z + z^2 = 64)
  (eq3 : z^2 + x*z + x^2 = 139) :
  x*y + y*z + x*z = 80 := by
sorry

end sum_of_products_l551_55175


namespace tan_45_degrees_l551_55148

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l551_55148


namespace stratified_sample_factory_a_l551_55132

theorem stratified_sample_factory_a (total : ℕ) (factory_a : ℕ) (sample_size : ℕ)
  (h_total : total = 98)
  (h_factory_a : factory_a = 56)
  (h_sample_size : sample_size = 14) :
  (factory_a : ℚ) / total * sample_size = 8 := by
  sorry

end stratified_sample_factory_a_l551_55132


namespace abs_sum_min_value_abs_sum_min_value_achieved_l551_55147

theorem abs_sum_min_value (x : ℝ) : 
  |x + 1| + |x + 2| + |x + 3| + |x + 4| + |x + 5| ≥ 6 :=
sorry

theorem abs_sum_min_value_achieved : 
  ∃ x : ℝ, |x + 1| + |x + 2| + |x + 3| + |x + 4| + |x + 5| = 6 :=
sorry

end abs_sum_min_value_abs_sum_min_value_achieved_l551_55147


namespace he_more_apples_l551_55193

/-- The number of apples Adam and Jackie have together -/
def total_adam_jackie : ℕ := 12

/-- The number of apples Adam has more than Jackie -/
def adam_more_than_jackie : ℕ := 8

/-- The number of apples He has -/
def he_apples : ℕ := 21

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 2

/-- The number of apples Adam has -/
def adam_apples : ℕ := jackie_apples + adam_more_than_jackie

theorem he_more_apples : he_apples - total_adam_jackie = 9 := by
  sorry

end he_more_apples_l551_55193


namespace equation_is_linear_l551_55171

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x - 1 = 20 -/
def f (x : ℝ) : ℝ := 2 * x - 1

/-- Theorem: The equation 2x - 1 = 20 is a linear equation -/
theorem equation_is_linear : is_linear_equation f := by
  sorry

end equation_is_linear_l551_55171


namespace no_natural_solutions_l551_55188

theorem no_natural_solutions : ¬∃ (x y : ℕ), (2 * x + y) * (2 * y + x) = 2017^2017 := by
  sorry

end no_natural_solutions_l551_55188


namespace container_capacity_l551_55197

/-- Given a container where 8 liters represents 20% of its capacity,
    this theorem proves that the total capacity of 40 such containers is 1600 liters. -/
theorem container_capacity (container_capacity : ℝ) 
  (h1 : 8 = 0.2 * container_capacity) 
  (num_containers : ℕ := 40) : 
  (num_containers : ℝ) * container_capacity = 1600 := by
  sorry

end container_capacity_l551_55197


namespace arithmetic_sequence_n_value_l551_55170

/-- An arithmetic sequence with first term a₁, common difference d, and nth term aₙ -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  n : ℕ
  aₙ : ℝ
  seq_def : aₙ = a₁ + (n - 1) * d

/-- The theorem stating that for the given arithmetic sequence, n = 100 -/
theorem arithmetic_sequence_n_value
  (seq : ArithmeticSequence)
  (h1 : seq.a₁ = 1)
  (h2 : seq.d = 3)
  (h3 : seq.aₙ = 298) :
  seq.n = 100 := by
  sorry

#check arithmetic_sequence_n_value

end arithmetic_sequence_n_value_l551_55170


namespace jackson_charity_collection_l551_55187

/-- Proves the number of houses Jackson needs to visit per day to meet his goal -/
theorem jackson_charity_collection (total_goal : ℕ) (days_per_week : ℕ) (monday_earnings : ℕ) (tuesday_earnings : ℕ) (houses_per_collection : ℕ) (earnings_per_collection : ℕ) : 
  total_goal = 1000 →
  days_per_week = 5 →
  monday_earnings = 300 →
  tuesday_earnings = 40 →
  houses_per_collection = 4 →
  earnings_per_collection = 10 →
  ∃ (houses_per_day : ℕ), 
    houses_per_day = 88 ∧ 
    (total_goal - monday_earnings - tuesday_earnings) = 
      (days_per_week - 2) * houses_per_day * (earnings_per_collection / houses_per_collection) :=
by
  sorry

end jackson_charity_collection_l551_55187


namespace f_extrema_on_interval_l551_55130

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 5*x^3 + 1

-- Define the interval
def interval : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

-- Theorem statement
theorem f_extrema_on_interval :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∃ x ∈ interval, f x = 153) ∧
  (∃ x ∈ interval, f x = -4) := by
sorry

end f_extrema_on_interval_l551_55130


namespace sally_fries_proof_l551_55121

/-- Calculates the final number of fries Sally has after receiving one-third of Mark's fries -/
def sallys_final_fries (sally_initial : ℕ) (mark_initial : ℕ) : ℕ :=
  sally_initial + mark_initial / 3

/-- Proves that Sally's final fry count is 26 given the initial conditions -/
theorem sally_fries_proof :
  sallys_final_fries 14 36 = 26 := by
  sorry

end sally_fries_proof_l551_55121


namespace f_simplification_f_value_in_third_quadrant_l551_55140

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by
  sorry

theorem f_value_in_third_quadrant (α : Real) 
  (h1 : α > Real.pi ∧ α < 3 * Real.pi / 2) 
  (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) : 
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry

end f_simplification_f_value_in_third_quadrant_l551_55140


namespace sum_of_coefficients_equals_negative_two_l551_55153

theorem sum_of_coefficients_equals_negative_two :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ),
  (∀ x : ℝ, (x^2 + 1) * (2*x + 1)^9 = 
    a₀ + a₁*(x+2) + a₂*(x+2)^2 + a₃*(x+2)^3 + a₄*(x+2)^4 + 
    a₅*(x+2)^5 + a₆*(x+2)^6 + a₇*(x+2)^7 + a₈*(x+2)^8 + 
    a₉*(x+2)^9 + a₁₀*(x+2)^10 + a₁₁*(x+2)^11) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = -2 := by
sorry

end sum_of_coefficients_equals_negative_two_l551_55153


namespace gcd_153_119_l551_55122

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_l551_55122


namespace symmetric_point_correct_l551_55196

/-- The symmetric point of (a, b) with respect to the y-axis is (-a, b) -/
def symmetric_point_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- The given point A -/
def point_A : ℝ × ℝ := (2, -3)

/-- The expected symmetric point -/
def expected_symmetric_point : ℝ × ℝ := (-2, -3)

theorem symmetric_point_correct :
  symmetric_point_y_axis point_A = expected_symmetric_point := by
  sorry

end symmetric_point_correct_l551_55196


namespace dave_cleaning_time_l551_55172

/-- Proves that Dave's cleaning time is 15 minutes, given Carla's cleaning time and the ratio of Dave's to Carla's time -/
theorem dave_cleaning_time (carla_time : ℕ) (dave_ratio : ℚ) : 
  carla_time = 40 → dave_ratio = 3/8 → dave_ratio * carla_time = 15 := by
  sorry

end dave_cleaning_time_l551_55172


namespace short_bingo_first_column_possibilities_l551_55192

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def primesBetween1And15 : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 15 ∧ isPrime n}

theorem short_bingo_first_column_possibilities :
  Nat.factorial 6 / Nat.factorial 1 = 720 :=
sorry

end short_bingo_first_column_possibilities_l551_55192


namespace cost_price_is_100_l551_55198

/-- Given a toy's cost price, calculate the final selling price after markup and discount --/
def final_price (cost : ℝ) : ℝ := cost * 1.5 * 0.8

/-- The profit made on the toy --/
def profit (cost : ℝ) : ℝ := final_price cost - cost

/-- Theorem stating that if the profit is 20 yuan, the cost price must be 100 yuan --/
theorem cost_price_is_100 : 
  ∀ x : ℝ, profit x = 20 → x = 100 := by
  sorry

end cost_price_is_100_l551_55198


namespace maria_apple_sales_l551_55112

/-- Calculate the average revenue per hour for Maria's apple sales -/
theorem maria_apple_sales (a1 a2 b1 b2 pa1 pa2 pb : ℕ) 
  (h1 : a1 = 10) -- kg of type A apples sold in first hour
  (h2 : a2 = 2)  -- kg of type A apples sold in second hour
  (h3 : b1 = 5)  -- kg of type B apples sold in first hour
  (h4 : b2 = 3)  -- kg of type B apples sold in second hour
  (h5 : pa1 = 3) -- price of type A apples in first hour
  (h6 : pa2 = 4) -- price of type A apples in second hour
  (h7 : pb = 2)  -- price of type B apples (constant)
  : (a1 * pa1 + b1 * pb + a2 * pa2 + b2 * pb) / 2 = 27 := by
  sorry

#check maria_apple_sales

end maria_apple_sales_l551_55112


namespace largest_b_value_l551_55177

theorem largest_b_value (b : ℝ) (h : (3*b + 4)*(b - 2) = 7*b) :
  b ≤ (9 + Real.sqrt 177) / 6 ∧
  ∃ (b : ℝ), (3*b + 4)*(b - 2) = 7*b ∧ b = (9 + Real.sqrt 177) / 6 :=
by sorry

end largest_b_value_l551_55177


namespace can_form_triangle_l551_55126

/-- Triangle Inequality Theorem: A set of three line segments can form a triangle
    if and only if the sum of the lengths of any two sides is greater than
    the length of the third side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that the set of line segments (3, 4, 5) can form a triangle. -/
theorem can_form_triangle : triangle_inequality 3 4 5 := by
  sorry

end can_form_triangle_l551_55126


namespace apple_to_mango_ratio_l551_55141

/-- Represents the total produce of fruits in kilograms -/
structure FruitProduce where
  apples : ℝ
  mangoes : ℝ
  oranges : ℝ

/-- Represents the fruit selling details -/
structure FruitSale where
  price_per_kg : ℝ
  total_amount : ℝ

/-- Theorem stating the ratio of apple to mango production -/
theorem apple_to_mango_ratio (fp : FruitProduce) (fs : FruitSale) :
  fp.mangoes = 400 ∧
  fp.oranges = fp.mangoes + 200 ∧
  fs.price_per_kg = 50 ∧
  fs.total_amount = 90000 ∧
  fs.total_amount = fs.price_per_kg * (fp.apples + fp.mangoes + fp.oranges) →
  fp.apples / fp.mangoes = 2.5 := by
  sorry

end apple_to_mango_ratio_l551_55141


namespace smallest_prime_dividing_sum_l551_55190

theorem smallest_prime_dividing_sum : 
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ (7^15 + 9^17) ∧ ∀ (q : Nat), Nat.Prime q → q ∣ (7^15 + 9^17) → p ≤ q :=
by sorry

end smallest_prime_dividing_sum_l551_55190


namespace amusement_park_visits_l551_55159

theorem amusement_park_visits 
  (season_pass_cost : ℕ) 
  (cost_per_trip : ℕ) 
  (youngest_son_visits : ℕ) 
  (oldest_son_visits : ℕ) : 
  season_pass_cost = 100 → 
  cost_per_trip = 4 → 
  youngest_son_visits = 15 → 
  oldest_son_visits * cost_per_trip = season_pass_cost - (youngest_son_visits * cost_per_trip) → 
  oldest_son_visits = 10 := by
sorry

end amusement_park_visits_l551_55159


namespace events_mutually_exclusive_l551_55102

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define the set of ball colors
inductive BallColor : Type
| Red : BallColor
| Black : BallColor
| White : BallColor

-- Define a distribution of balls to people
def Distribution := Person → BallColor

-- Define the event "A receives the white ball"
def event_A_white (d : Distribution) : Prop :=
  d Person.A = BallColor.White

-- Define the event "B receives the white ball"
def event_B_white (d : Distribution) : Prop :=
  d Person.B = BallColor.White

-- Theorem stating that the events are mutually exclusive
theorem events_mutually_exclusive :
  ∀ (d : Distribution), ¬(event_A_white d ∧ event_B_white d) :=
by sorry

end events_mutually_exclusive_l551_55102


namespace average_study_time_difference_l551_55184

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of weekdays -/
def weekdays : ℕ := 5

/-- The number of weekend days -/
def weekend_days : ℕ := 2

/-- The differences in study time on weekdays -/
def weekday_differences : List ℤ := [5, -5, 15, 25, -15]

/-- The additional time Sasha studied on weekends compared to usual -/
def weekend_additional_time : ℤ := 15

/-- The average difference in study time per day -/
def average_difference : ℚ := 12

theorem average_study_time_difference :
  (weekday_differences.sum + 2 * (weekend_additional_time + 15)) / days_in_week = average_difference := by
  sorry

end average_study_time_difference_l551_55184


namespace faye_pencils_count_l551_55180

theorem faye_pencils_count (rows : ℕ) (pencils_per_row : ℕ) 
  (h1 : rows = 14) (h2 : pencils_per_row = 11) : 
  rows * pencils_per_row = 154 := by
  sorry

end faye_pencils_count_l551_55180


namespace rubble_initial_money_l551_55134

/-- The amount of money Rubble had initially -/
def initial_money : ℝ := 15

/-- The cost of a notebook -/
def notebook_cost : ℝ := 4

/-- The cost of a pen -/
def pen_cost : ℝ := 1.5

/-- The number of notebooks Rubble bought -/
def num_notebooks : ℕ := 2

/-- The number of pens Rubble bought -/
def num_pens : ℕ := 2

/-- The amount of money Rubble had left after the purchase -/
def money_left : ℝ := 4

theorem rubble_initial_money :
  initial_money = 
    (num_notebooks : ℝ) * notebook_cost + 
    (num_pens : ℝ) * pen_cost + 
    money_left :=
by
  sorry

end rubble_initial_money_l551_55134


namespace largest_integer_satisfying_inequality_l551_55111

theorem largest_integer_satisfying_inequality :
  ∀ n : ℕ, (1 / 5 : ℚ) + (n : ℚ) / 8 < 9 / 5 ↔ n ≤ 12 :=
by sorry

end largest_integer_satisfying_inequality_l551_55111


namespace bacteria_count_theorem_l551_55166

/-- The number of bacteria after growth, given the original count and increase. -/
def bacteria_after_growth (original : ℕ) (increase : ℕ) : ℕ :=
  original + increase

/-- Theorem stating that the number of bacteria after growth is 8917,
    given the original count of 600 and an increase of 8317. -/
theorem bacteria_count_theorem :
  bacteria_after_growth 600 8317 = 8917 := by
  sorry

end bacteria_count_theorem_l551_55166


namespace max_value_of_z_l551_55186

theorem max_value_of_z (x y : ℝ) (h1 : x + y ≤ 10) (h2 : 3 * x + y ≤ 18) 
  (h3 : x ≥ 0) (h4 : y ≥ 0) : 
  ∃ (z : ℝ), z = x + y / 2 ∧ z ≤ 7 ∧ ∀ (w : ℝ), w = x + y / 2 → w ≤ z :=
sorry

end max_value_of_z_l551_55186


namespace corn_cobs_picked_l551_55194

theorem corn_cobs_picked (bushel_weight : ℝ) (ear_weight : ℝ) (bushels_picked : ℝ) : 
  bushel_weight = 56 → 
  ear_weight = 0.5 → 
  bushels_picked = 2 → 
  (bushels_picked * bushel_weight / ear_weight : ℝ) = 224 := by
sorry

end corn_cobs_picked_l551_55194


namespace kaleb_chocolate_pieces_l551_55109

theorem kaleb_chocolate_pieces (initial_boxes : ℕ) (boxes_given : ℕ) (pieces_per_box : ℕ) : 
  initial_boxes = 14 → boxes_given = 5 → pieces_per_box = 6 →
  (initial_boxes - boxes_given) * pieces_per_box = 54 := by
  sorry

end kaleb_chocolate_pieces_l551_55109


namespace power_product_equals_one_third_l551_55114

theorem power_product_equals_one_third :
  (-3 : ℚ)^2022 * (1/3 : ℚ)^2023 = 1/3 :=
by sorry

end power_product_equals_one_third_l551_55114


namespace divisibility_by_81_invariant_under_reversal_l551_55103

/-- A sequence of digits represented as a list of natural numbers. -/
def DigitSequence := List Nat

/-- Check if a number represented by a digit sequence is divisible by 81. -/
def isDivisibleBy81 (digits : DigitSequence) : Prop :=
  digits.foldl (fun acc d => (10 * acc + d) % 81) 0 = 0

/-- Reverse a digit sequence. -/
def reverseDigits (digits : DigitSequence) : DigitSequence :=
  digits.reverse

theorem divisibility_by_81_invariant_under_reversal
  (digits : DigitSequence)
  (h : digits.length = 2016)
  (h_divisible : isDivisibleBy81 digits) :
  isDivisibleBy81 (reverseDigits digits) := by
  sorry

#check divisibility_by_81_invariant_under_reversal

end divisibility_by_81_invariant_under_reversal_l551_55103


namespace parabola_focus_l551_55149

/-- The parabola defined by the equation y = (1/4)x^2 -/
def parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (1/4) * p.1^2}

/-- The focus of a parabola is a point from which the distance to any point on the parabola
    is equal to the distance from that point to a fixed line called the directrix -/
def is_focus (f : ℝ × ℝ) (p : Set (ℝ × ℝ)) : Prop :=
  ∃ (d : ℝ), ∀ x y : ℝ, (x, y) ∈ p → 
    (x - f.1)^2 + (y - f.2)^2 = (y + d)^2

/-- The theorem stating that the focus of the parabola y = (1/4)x^2 is at (0, 1) -/
theorem parabola_focus :
  is_focus (0, 1) parabola := by sorry

end parabola_focus_l551_55149


namespace log_base_value_log_inequality_greater_than_one_log_inequality_less_than_one_l551_55113

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem 1
theorem log_base_value (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  log a 8 = 3 → a = 2 := by sorry

-- Theorem 2
theorem log_inequality_greater_than_one (a : ℝ) (ha : a > 1) (x : ℝ) :
  log a x ≤ log a (2 - 3*x) ↔ (0 < x ∧ x ≤ 1/2) := by sorry

-- Theorem 3
theorem log_inequality_less_than_one (a : ℝ) (ha : 0 < a ∧ a < 1) (x : ℝ) :
  log a x ≤ log a (2 - 3*x) ↔ (1/2 ≤ x ∧ x < 2/3) := by sorry

end log_base_value_log_inequality_greater_than_one_log_inequality_less_than_one_l551_55113


namespace solution_set_and_roots_negative_at_two_implies_bound_l551_55145

def f (a b x : ℝ) : ℝ := -3 * x^2 + a * (5 - a) * x + b

theorem solution_set_and_roots (a b : ℝ) :
  (∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  ((a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9)) := by sorry

theorem negative_at_two_implies_bound (b : ℝ) :
  (∀ a, f a b 2 < 0) →
  b < -1/2 := by sorry

end solution_set_and_roots_negative_at_two_implies_bound_l551_55145


namespace bob_jacket_purchase_percentage_l551_55156

/-- Calculates the percentage of the suggested retail price that Bob paid for a jacket -/
theorem bob_jacket_purchase_percentage (P : ℝ) (P_pos : P > 0) : 
  let marked_price := P * (1 - 0.4)
  let bob_price := marked_price * (1 - 0.4)
  bob_price / P = 0.36 := by
  sorry

end bob_jacket_purchase_percentage_l551_55156


namespace inscribed_circle_areas_l551_55185

-- Define the square with its diagonal
def square_diagonal : ℝ := 40

-- Define the theorem
theorem inscribed_circle_areas :
  let square_side := square_diagonal / Real.sqrt 2
  let square_area := square_side ^ 2
  let circle_radius := square_side / 2
  let circle_area := π * circle_radius ^ 2
  square_area = 800 ∧ circle_area = 200 * π := by
  sorry


end inscribed_circle_areas_l551_55185


namespace x_divisibility_l551_55160

def x : ℕ := 36^2 + 48^2 + 64^3 + 81^2

theorem x_divisibility :
  (∃ k : ℕ, x = 3 * k) ∧
  (∃ k : ℕ, x = 4 * k) ∧
  (∃ k : ℕ, x = 9 * k) ∧
  ¬(∃ k : ℕ, x = 16 * k) := by
  sorry

end x_divisibility_l551_55160
