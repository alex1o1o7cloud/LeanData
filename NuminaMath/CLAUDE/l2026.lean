import Mathlib

namespace ladder_in_alley_l2026_202630

/-- In a narrow alley, a ladder of length b is placed between two walls.
    When resting against one wall, it makes a 60° angle with the ground and reaches height s.
    When resting against the other wall, it makes a 70° angle with the ground and reaches height m.
    This theorem states that the width of the alley w is equal to m. -/
theorem ladder_in_alley (w b s m : ℝ) (h1 : 0 < w) (h2 : 0 < b) (h3 : 0 < s) (h4 : 0 < m)
  (h5 : w = b * Real.sin (60 * π / 180))
  (h6 : s = b * Real.sin (60 * π / 180))
  (h7 : w = b * Real.sin (70 * π / 180))
  (h8 : m = b * Real.sin (70 * π / 180)) :
  w = m :=
sorry

end ladder_in_alley_l2026_202630


namespace bob_has_31_pennies_l2026_202675

/-- The number of pennies Alex has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob has -/
def bob_pennies : ℕ := sorry

/-- If Alex gives Bob a penny, Bob will have four times as many pennies as Alex has -/
axiom condition1 : bob_pennies + 1 = 4 * (alex_pennies - 1)

/-- If Bob gives Alex a penny, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bob_pennies - 1 = 3 * (alex_pennies + 1)

/-- Bob has 31 pennies -/
theorem bob_has_31_pennies : bob_pennies = 31 := by sorry

end bob_has_31_pennies_l2026_202675


namespace robot_path_lengths_l2026_202693

/-- Represents the direction the robot is facing -/
inductive Direction
| North
| East
| South
| West

/-- Represents a point in the plane -/
structure Point where
  x : Int
  y : Int

/-- Represents the state of the robot -/
structure RobotState where
  position : Point
  direction : Direction

/-- The robot's path -/
def RobotPath := List RobotState

/-- Function to check if a path is valid according to the problem conditions -/
def is_valid_path (path : RobotPath) : Bool :=
  sorry

/-- Function to check if a path returns to the starting point -/
def returns_to_start (path : RobotPath) : Bool :=
  sorry

/-- Function to check if a path visits any point more than once -/
def no_revisits (path : RobotPath) : Bool :=
  sorry

/-- Theorem stating the possible path lengths for the robot -/
theorem robot_path_lengths :
  ∀ (n : Nat), 
    (∃ (path : RobotPath), 
      path.length = n ∧ 
      is_valid_path path ∧ 
      returns_to_start path ∧ 
      no_revisits path) ↔ 
    (∃ (k : Nat), n = 4 * k ∧ k ≥ 3) :=
  sorry

end robot_path_lengths_l2026_202693


namespace function_positive_implies_m_bound_l2026_202634

open Real

theorem function_positive_implies_m_bound (m : ℝ) :
  (∀ x : ℝ, x > 0 → (Real.exp x / x - m * x) > 0) →
  m < Real.exp 2 / 4 := by
  sorry

end function_positive_implies_m_bound_l2026_202634


namespace polynomial_division_remainder_l2026_202638

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (X^4 - 1) * (X^2 - 1) = (X^2 + X + 1) * q + 3 := by
  sorry

end polynomial_division_remainder_l2026_202638


namespace existence_of_h_l2026_202627

theorem existence_of_h : ∃ h : ℝ, ∀ n : ℕ, 
  ¬(⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋) :=
sorry

end existence_of_h_l2026_202627


namespace tournament_result_l2026_202668

-- Define the tournament structure
structure Tournament :=
  (teams : Fin 9 → ℕ)  -- Each team's points
  (t1_wins : ℕ)
  (t1_draws : ℕ)
  (t1_losses : ℕ)
  (t9_wins : ℕ)
  (t9_draws : ℕ)
  (t9_losses : ℕ)

-- Define the conditions of the tournament
def valid_tournament (t : Tournament) : Prop :=
  t.teams 0 = 3 * t.t1_wins + t.t1_draws ∧  -- T1's score
  t.teams 8 = t.t9_draws ∧  -- T9's score
  t.t1_wins = 3 ∧ t.t1_draws = 4 ∧ t.t1_losses = 1 ∧
  t.t9_wins = 0 ∧ t.t9_draws = 5 ∧ t.t9_losses = 3 ∧
  (∀ i j, i < j → t.teams i > t.teams j) ∧  -- Strict ordering
  (∀ i, t.teams i ≤ 24)  -- Maximum possible points

-- Define the theorem
theorem tournament_result (t : Tournament) (h : valid_tournament t) :
  (¬ ∃ (t3_defeats_t4 : Bool), t.teams 2 > t.teams 3) ∧
  (∃ (t4_defeats_t3 : Bool), t.teams 3 > t.teams 2) :=
sorry

end tournament_result_l2026_202668


namespace constant_term_binomial_expansion_l2026_202654

/-- The constant term in the expansion of (√5/5 * x^2 + 1/x)^6 is 3 -/
theorem constant_term_binomial_expansion :
  let a := Real.sqrt 5 / 5
  let b := 1
  let n := 6
  let k := 4  -- The value of k where x^(2n-3k) = x^0
  (Nat.choose n k) * a^(n-k) * b^k = 3 := by
  sorry

end constant_term_binomial_expansion_l2026_202654


namespace cube_volume_surface_area_l2026_202650

theorem cube_volume_surface_area (y : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 6*y ∧ 6*s^2 = 2*y) → y = 5832 :=
by
  sorry

end cube_volume_surface_area_l2026_202650


namespace number_of_boys_in_class_number_of_boys_is_correct_l2026_202683

/-- The number of boys in a class given certain weight conditions -/
theorem number_of_boys_in_class : ℕ :=
  let initial_average : ℚ := 58.4
  let misread_weight : ℕ := 56
  let correct_weight : ℕ := 68
  let correct_average : ℚ := 59
  20

theorem number_of_boys_is_correct (n : ℕ) :
  (n : ℚ) * 58.4 + (68 - 56) = n * 59 → n = number_of_boys_in_class :=
by sorry

end number_of_boys_in_class_number_of_boys_is_correct_l2026_202683


namespace chest_value_is_35000_l2026_202699

/-- Represents the pirate treasure distribution problem -/
structure PirateTreasure where
  total_pirates : ℕ
  total_chests : ℕ
  pirates_with_chests : ℕ
  contribution_per_chest : ℕ

/-- The specific instance of the pirate treasure problem -/
def pirate_problem : PirateTreasure := {
  total_pirates := 7
  total_chests := 5
  pirates_with_chests := 5
  contribution_per_chest := 10000
}

/-- Calculates the value of one chest based on the given problem parameters -/
def chest_value (p : PirateTreasure) : ℕ :=
  let total_contribution := p.pirates_with_chests * p.contribution_per_chest
  let pirates_without_chests := p.total_pirates - p.pirates_with_chests
  let compensation_per_pirate := total_contribution / pirates_without_chests
  p.total_pirates * compensation_per_pirate / p.total_chests

/-- Theorem stating that the chest value for the given problem is 35000 -/
theorem chest_value_is_35000 : chest_value pirate_problem = 35000 := by
  sorry

end chest_value_is_35000_l2026_202699


namespace friends_score_l2026_202674

/-- Given that Edward and his friend scored a total of 13 points in basketball,
    and Edward scored 7 points, prove that Edward's friend scored 6 points. -/
theorem friends_score (total : ℕ) (edward : ℕ) (friend : ℕ)
    (h1 : total = 13)
    (h2 : edward = 7)
    (h3 : total = edward + friend) :
  friend = 6 := by
sorry

end friends_score_l2026_202674


namespace greatest_difference_l2026_202653

theorem greatest_difference (x y : ℤ) 
  (hx : 5 < x ∧ x < 8) 
  (hy : 8 < y ∧ y < 13) 
  (hxdiv : x % 3 = 0) 
  (hydiv : y % 3 = 0) : 
  (∀ a b : ℤ, 5 < a ∧ a < 8 ∧ 8 < b ∧ b < 13 ∧ a % 3 = 0 ∧ b % 3 = 0 → b - a ≤ y - x) ∧ y - x = 6 := by
  sorry

end greatest_difference_l2026_202653


namespace sea_turtle_shell_age_l2026_202697

/-- Converts an octal digit to decimal --/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- Converts an octal number to decimal --/
def octal_to_decimal_full (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + octal_to_decimal digit * 8^i) 0

theorem sea_turtle_shell_age :
  octal_to_decimal_full [4, 5, 7, 3] = 2028 := by
  sorry

end sea_turtle_shell_age_l2026_202697


namespace rhombus_side_length_l2026_202651

/-- A rhombus with area K and one diagonal three times the length of the other has side length √(5K/3) -/
theorem rhombus_side_length (K : ℝ) (d₁ d₂ s : ℝ) (h₁ : K > 0) (h₂ : d₁ > 0) (h₃ : d₂ > 0) (h₄ : s > 0) :
  d₂ = 3 * d₁ →
  K = (1/2) * d₁ * d₂ →
  s^2 = (d₁/2)^2 + (d₂/2)^2 →
  s = Real.sqrt ((5 * K) / 3) :=
by sorry

end rhombus_side_length_l2026_202651


namespace max_numbers_summing_to_1000_with_distinct_digit_sums_l2026_202663

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a list of natural numbers has pairwise distinct sums of digits -/
def hasPairwiseDistinctDigitSums (list : List ℕ) : Prop := sorry

/-- The maximum number of natural numbers summing to 1000 with pairwise distinct digit sums -/
theorem max_numbers_summing_to_1000_with_distinct_digit_sums :
  ∃ (list : List ℕ),
    list.sum = 1000 ∧
    hasPairwiseDistinctDigitSums list ∧
    list.length = 19 ∧
    ∀ (other_list : List ℕ),
      other_list.sum = 1000 →
      hasPairwiseDistinctDigitSums other_list →
      other_list.length ≤ 19 := by
  sorry

end max_numbers_summing_to_1000_with_distinct_digit_sums_l2026_202663


namespace lice_check_time_l2026_202625

/-- Calculates the time required for lice checks in an elementary school -/
theorem lice_check_time (kindergarteners : ℕ) (first_graders : ℕ) (second_graders : ℕ) (third_graders : ℕ) 
  (time_per_check : ℕ) (h1 : kindergarteners = 26) (h2 : first_graders = 19) (h3 : second_graders = 20) 
  (h4 : third_graders = 25) (h5 : time_per_check = 2) : 
  (kindergarteners + first_graders + second_graders + third_graders) * time_per_check / 60 = 3 := by
  sorry

end lice_check_time_l2026_202625


namespace geometric_sequence_general_term_l2026_202672

/-- Given a geometric sequence where the first three terms are a-1, a+1, and a+4,
    prove that the general term formula is a_n = 4 × (3/2)^(n-1) -/
theorem geometric_sequence_general_term (a : ℝ) (n : ℕ) :
  (a - 1 : ℝ) * (a + 4 : ℝ) = (a + 1 : ℝ)^2 →
  ∃ (seq : ℕ → ℝ), seq 1 = a - 1 ∧ seq 2 = a + 1 ∧ seq 3 = a + 4 ∧
    (∀ k : ℕ, seq (k + 1) / seq k = seq 2 / seq 1) →
    ∀ m : ℕ, seq m = 4 * (3/2 : ℝ)^(m - 1) :=
by sorry

end geometric_sequence_general_term_l2026_202672


namespace smallest_number_with_given_remainders_l2026_202603

theorem smallest_number_with_given_remainders : ∃! n : ℕ, 
  n > 0 ∧
  n % 2 = 1 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  ∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 → n ≤ m :=
by
  sorry

end smallest_number_with_given_remainders_l2026_202603


namespace consecutive_integers_product_l2026_202608

theorem consecutive_integers_product (a b : ℕ) (h1 : 0 < a) (h2 : a < b) :
  ∀ (start : ℕ), ∃ (x y : ℕ), 
    start < x ∧ x ≤ start + b - 1 ∧
    start < y ∧ y ≤ start + b - 1 ∧
    x ≠ y ∧
    (a * b) ∣ (x * y) :=
by sorry

end consecutive_integers_product_l2026_202608


namespace roots_of_polynomial_l2026_202602

/-- The polynomial f(x) = x^3 - 5x^2 + 8x - 4 -/
def f (x : ℝ) : ℝ := x^3 - 5*x^2 + 8*x - 4

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 10*x + 8

theorem roots_of_polynomial :
  (f 1 = 0) ∧ 
  (f 2 = 0) ∧ 
  (f' 2 = 0) ∧
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = 2) :=
sorry

end roots_of_polynomial_l2026_202602


namespace P_necessary_not_sufficient_for_Q_l2026_202637

-- Define propositions P and Q
def P (a b : ℝ) : Prop := a^2 + b^2 > 2*a*b
def Q (a b : ℝ) : Prop := |a + b| < |a| + |b|

-- Theorem statement
theorem P_necessary_not_sufficient_for_Q :
  (∀ a b : ℝ, Q a b → P a b) ∧
  (∃ a b : ℝ, P a b ∧ ¬(Q a b)) :=
sorry

end P_necessary_not_sufficient_for_Q_l2026_202637


namespace prob_sum_five_is_one_third_l2026_202610

/-- Representation of the cube faces -/
def cube_faces : Finset ℕ := {1, 2, 2, 3, 3, 3}

/-- The number of faces on the cube -/
def num_faces : ℕ := Finset.card cube_faces

/-- The set of all possible outcomes when throwing the cube twice -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product cube_faces cube_faces

/-- The set of outcomes that sum to 5 -/
def sum_five_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun p => p.1 + p.2 = 5)

/-- The probability of getting a sum of 5 when throwing the cube twice -/
def prob_sum_five : ℚ :=
  (Finset.card sum_five_outcomes : ℚ) / (Finset.card all_outcomes : ℚ)

theorem prob_sum_five_is_one_third :
  prob_sum_five = 1 / 3 := by
  sorry

end prob_sum_five_is_one_third_l2026_202610


namespace line_through_points_l2026_202632

/-- A line passing through given points -/
structure Line where
  a : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  y = l.a * x + l.b

/-- The main theorem -/
theorem line_through_points : ∃ (l : Line),
  l.contains 2 8 ∧
  l.contains 5 17 ∧
  l.contains 8 26 ∧
  l.contains 34 104 := by
  sorry

end line_through_points_l2026_202632


namespace swimmer_speed_in_still_water_l2026_202640

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed of the swimmer. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 5.17 km/h. -/
theorem swimmer_speed_in_still_water :
  ∃ (s : SwimmerSpeeds),
    (effectiveSpeed s true * 5 = 36) ∧
    (effectiveSpeed s false * 7 = 22) ∧
    (s.swimmer = 5.17) := by
  sorry

end swimmer_speed_in_still_water_l2026_202640


namespace triangles_in_4x6_grid_l2026_202662

/-- Represents a grid with vertical and horizontal sections -/
structure Grid :=
  (vertical_sections : ℕ)
  (horizontal_sections : ℕ)

/-- Calculates the number of triangles in a grid with diagonal lines -/
def count_triangles (g : Grid) : ℕ :=
  let small_right_triangles := g.vertical_sections * g.horizontal_sections
  let medium_right_triangles := 2 * (g.vertical_sections - 1) * (g.horizontal_sections - 1)
  let large_isosceles_triangles := g.horizontal_sections - 1
  small_right_triangles + medium_right_triangles + large_isosceles_triangles

/-- Theorem: The number of triangles in a 4x6 grid is 90 -/
theorem triangles_in_4x6_grid :
  count_triangles { vertical_sections := 4, horizontal_sections := 6 } = 90 := by
  sorry

end triangles_in_4x6_grid_l2026_202662


namespace marla_errand_time_l2026_202621

/-- The time Marla spends driving one way to her son's school -/
def drive_time : ℕ := 20

/-- The time Marla spends attending parent-teacher night -/
def meeting_time : ℕ := 70

/-- The total time Marla spends on the errand -/
def total_time : ℕ := 2 * drive_time + meeting_time

/-- Theorem stating that the total time Marla spends on the errand is 110 minutes -/
theorem marla_errand_time : total_time = 110 := by
  sorry

end marla_errand_time_l2026_202621


namespace five_star_three_eq_nineteen_l2026_202615

/-- Definition of the star operation -/
def star (a b : ℝ) : ℝ := a^2 - a*b + b^2

/-- Theorem: The value of 5 star 3 is 19 -/
theorem five_star_three_eq_nineteen : star 5 3 = 19 := by
  sorry

end five_star_three_eq_nineteen_l2026_202615


namespace expression_simplification_l2026_202607

theorem expression_simplification (x : ℝ) : 24 * (3 * x - 4) - 6 * x = 66 * x - 96 := by
  sorry

end expression_simplification_l2026_202607


namespace anne_sweettarts_distribution_l2026_202613

theorem anne_sweettarts_distribution (total_sweettarts : ℕ) (sweettarts_per_friend : ℕ) (num_friends : ℕ) : 
  total_sweettarts = 15 → 
  sweettarts_per_friend = 5 → 
  total_sweettarts = sweettarts_per_friend * num_friends → 
  num_friends = 3 := by
  sorry

end anne_sweettarts_distribution_l2026_202613


namespace trajectory_equation_l2026_202667

/-- The trajectory of a point M that satisfies |MF₁| + |MF₂| = 10, where F₁ = (-3, 0) and F₂ = (3, 0) -/
theorem trajectory_equation (M : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  F₁ = (-3, 0) →
  F₂ = (3, 0) →
  ‖M - F₁‖ + ‖M - F₂‖ = 10 →
  (M.1^2 / 25 + M.2^2 / 16 = 1) :=
by sorry

end trajectory_equation_l2026_202667


namespace sum_ends_in_zero_squares_same_last_digit_l2026_202690

theorem sum_ends_in_zero_squares_same_last_digit (a b : ℤ) :
  (a + b) % 10 = 0 → (a ^ 2) % 10 = (b ^ 2) % 10 := by
  sorry

end sum_ends_in_zero_squares_same_last_digit_l2026_202690


namespace ali_baba_cave_theorem_l2026_202647

/-- Represents the state of a barrel (herring head up or down) -/
inductive BarrelState
| Up
| Down

/-- Represents a configuration of n barrels -/
def Configuration (n : ℕ) := Fin n → BarrelState

/-- Represents a move by Ali Baba -/
def Move (n : ℕ) := Fin n → Bool

/-- Apply a move to a configuration -/
def applyMove (n : ℕ) (config : Configuration n) (move : Move n) : Configuration n :=
  fun i => if move i then match config i with
    | BarrelState.Up => BarrelState.Down
    | BarrelState.Down => BarrelState.Up
  else config i

/-- Check if all barrels are in the same state -/
def allSameState (n : ℕ) (config : Configuration n) : Prop :=
  (∀ i : Fin n, config i = BarrelState.Up) ∨ (∀ i : Fin n, config i = BarrelState.Down)

/-- Ali Baba can win in a finite number of moves -/
def canWin (n : ℕ) : Prop :=
  ∃ (strategy : ℕ → Move n), ∀ (initialConfig : Configuration n),
    ∃ (k : ℕ), allSameState n (Nat.rec initialConfig (fun i config => applyMove n config (strategy i)) k)

/-- n is a power of 2 -/
def isPowerOfTwo (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem ali_baba_cave_theorem (n : ℕ) :
  canWin n ↔ isPowerOfTwo n :=
sorry

end ali_baba_cave_theorem_l2026_202647


namespace power_function_through_point_l2026_202601

/-- A power function is a function of the form f(x) = x^a for some real number a. -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

theorem power_function_through_point (f : ℝ → ℝ) :
  is_power_function f → f 2 = 16 → f = fun x ↦ x^4 := by
  sorry

end power_function_through_point_l2026_202601


namespace quadratic_radical_always_defined_l2026_202619

theorem quadratic_radical_always_defined (x : ℝ) : 0 ≤ x^2 + 2 := by sorry

#check quadratic_radical_always_defined

end quadratic_radical_always_defined_l2026_202619


namespace root_sum_reciprocal_l2026_202633

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - 2*p + 3 = 0) → 
  (q^3 - 2*q + 3 = 0) → 
  (r^3 - 2*r + 3 = 0) → 
  (1/(p+2) + 1/(q+2) + 1/(r+2) = -10) := by
sorry

end root_sum_reciprocal_l2026_202633


namespace alcohol_mixture_proof_l2026_202692

/-- Proves that adding 1.5 liters of 90% alcohol solution to 6 liters of 40% alcohol solution 
    results in a final mixture that is 50% alcohol. -/
theorem alcohol_mixture_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.4
  let added_concentration : ℝ := 0.9
  let added_volume : ℝ := 1.5
  let final_concentration : ℝ := 0.5
  let final_volume : ℝ := initial_volume + added_volume
  let initial_alcohol : ℝ := initial_volume * initial_concentration
  let added_alcohol : ℝ := added_volume * added_concentration
  let total_alcohol : ℝ := initial_alcohol + added_alcohol
  total_alcohol = final_volume * final_concentration :=
by
  sorry


end alcohol_mixture_proof_l2026_202692


namespace rectangle_measurement_error_l2026_202694

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) 
  (h1 : L > 0) (h2 : W > 0) (h3 : x > 0) :
  L * (1 + x / 100) * W * 0.95 = L * W * 1.045 → x = 10 := by
  sorry

end rectangle_measurement_error_l2026_202694


namespace maximal_value_S_l2026_202657

theorem maximal_value_S (a b c d : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0)
  (hsum : a + b + c + d = 100) :
  let S := (a / (b + 7))^(1/3) + (b / (c + 7))^(1/3) + (c / (d + 7))^(1/3) + (d / (a + 7))^(1/3)
  S ≤ 8 / 7^(1/3) :=
by sorry

end maximal_value_S_l2026_202657


namespace perfect_squares_equivalence_l2026_202658

theorem perfect_squares_equivalence (n : ℕ+) :
  (∃ k : ℕ, 2 * n + 1 = k^2) ∧ (∃ t : ℕ, 3 * n + 1 = t^2) ↔
  (∃ k : ℕ, n + 1 = k^2 + (k + 1)^2) ∧ 
  (∃ t : ℕ, n + 1 = (t - 1)^2 + 2 * t^2 ∨ n + 1 = (t + 1)^2 + 2 * t^2) :=
by sorry

end perfect_squares_equivalence_l2026_202658


namespace intersection_of_A_and_B_l2026_202687

def A : Set ℚ := {x | ∃ k : ℕ, x = 3 * k + 1}
def B : Set ℚ := {x | x ≤ 7}

theorem intersection_of_A_and_B : A ∩ B = {1, 4, 7} := by sorry

end intersection_of_A_and_B_l2026_202687


namespace revenue_in_scientific_notation_l2026_202643

/-- Represents the value of 1 billion in scientific notation -/
def billion : ℝ := 10^9

/-- The box office revenue in billions -/
def revenue : ℝ := 53.96

theorem revenue_in_scientific_notation :
  revenue * billion = 5.396 * 10^10 := by sorry

end revenue_in_scientific_notation_l2026_202643


namespace paper_clip_distribution_l2026_202661

theorem paper_clip_distribution (total_clips : ℕ) (num_boxes : ℕ) (clips_per_box : ℕ) :
  total_clips = 81 →
  num_boxes = 9 →
  total_clips = num_boxes * clips_per_box →
  clips_per_box = 9 :=
by
  sorry

end paper_clip_distribution_l2026_202661


namespace cafeteria_bill_calculation_l2026_202673

/-- Calculates the total cost for Mell and her friends at the cafeteria --/
def cafeteria_bill (coffee_price ice_cream_price cake_price : ℚ) 
  (discount_rate tax_rate : ℚ) : ℚ :=
  let mell_order := 2 * coffee_price + cake_price
  let friend_order := 2 * coffee_price + cake_price + ice_cream_price
  let total_before_discount := mell_order + 2 * friend_order
  let discounted_total := total_before_discount * (1 - discount_rate)
  let final_total := discounted_total * (1 + tax_rate)
  final_total

/-- Theorem stating that the total bill for Mell and her friends is $47.69 --/
theorem cafeteria_bill_calculation : 
  cafeteria_bill 4 3 7 (15/100) (10/100) = 47.69 := by
  sorry

end cafeteria_bill_calculation_l2026_202673


namespace quadratic_real_roots_l2026_202655

theorem quadratic_real_roots (n : ℕ+) :
  (∃ x : ℝ, x^2 - 4*x + n.val = 0) ↔ n.val ∈ ({1, 2, 3, 4} : Set ℕ) := by
  sorry

end quadratic_real_roots_l2026_202655


namespace square_area_from_rectangle_l2026_202609

theorem square_area_from_rectangle (circle_radius : ℝ) (rectangle_length rectangle_breadth rectangle_area square_side : ℝ) : 
  rectangle_length = (2 / 3) * circle_radius →
  circle_radius = square_side →
  rectangle_area = 598 →
  rectangle_breadth = 13 →
  rectangle_area = rectangle_length * rectangle_breadth →
  square_side ^ 2 = 4761 :=
by sorry

end square_area_from_rectangle_l2026_202609


namespace triangle_properties_l2026_202620

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  c = Real.sqrt 7 →
  4 * Real.sin (((A + B) / 2) : ℝ)^2 - Real.cos (2 * C) = 7/2 →
  C = π/3 ∧
  (∀ (a' b' : ℝ), (1/2) * a' * b' * Real.sin C ≤ (7 * Real.sqrt 3) / 4) :=
by sorry

end triangle_properties_l2026_202620


namespace correct_plate_set_is_valid_l2026_202645

/-- Represents a plate with a count of bacteria -/
structure Plate where
  count : ℕ
  deriving Repr

/-- Represents a set of plates used in the dilution spread plate method -/
structure PlateSet where
  plates : List Plate
  dilutionFactor : ℕ
  deriving Repr

/-- Checks if a plate count is valid (between 30 and 300) -/
def isValidCount (count : ℕ) : Bool :=
  30 ≤ count ∧ count ≤ 300

/-- Checks if a plate set is valid for the dilution spread plate method -/
def isValidPlateSet (ps : PlateSet) : Bool :=
  ps.plates.length ≥ 3 ∧ 
  ps.plates.all (fun p => isValidCount p.count) ∧
  ps.dilutionFactor = 10^6

/-- Calculates the average count of a plate set -/
def averageCount (ps : PlateSet) : ℚ :=
  let total : ℚ := ps.plates.foldl (fun acc p => acc + p.count) 0
  total / ps.plates.length

/-- The correct plate set for the problem -/
def correctPlateSet : PlateSet :=
  { plates := [⟨210⟩, ⟨240⟩, ⟨250⟩],
    dilutionFactor := 10^6 }

theorem correct_plate_set_is_valid :
  isValidPlateSet correctPlateSet ∧ 
  averageCount correctPlateSet = 233 :=
sorry

end correct_plate_set_is_valid_l2026_202645


namespace expand_product_l2026_202649

theorem expand_product (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * (4 / x - 5 * x^2 + 20 / x^3) = 3 / x - 15 * x^2 / 4 + 15 / x^3 := by
  sorry

end expand_product_l2026_202649


namespace x_intercept_distance_l2026_202698

/-- Two lines with slopes 4 and 6 intersecting at (8,12) have x-intercepts with distance 1 -/
theorem x_intercept_distance (line1 line2 : ℝ → ℝ) : 
  (∀ x, line1 x = 4*x - 20) →  -- Equation of line with slope 4
  (∀ x, line2 x = 6*x - 36) →  -- Equation of line with slope 6
  line1 8 = 12 →              -- Lines intersect at (8,12)
  line2 8 = 12 →              -- Lines intersect at (8,12)
  ∃ x1 x2, line1 x1 = 0 ∧ line2 x2 = 0 ∧ |x2 - x1| = 1 := by
sorry

end x_intercept_distance_l2026_202698


namespace necessary_not_implies_sufficient_l2026_202679

theorem necessary_not_implies_sufficient (A B : Prop) : 
  (A → B) → ¬(∀ A B, (A → B) → (B → A)) :=
by sorry

end necessary_not_implies_sufficient_l2026_202679


namespace expression_meets_requirements_l2026_202669

/-- Represents an algebraic expression -/
inductive AlgebraicExpression
  | constant (n : ℚ)
  | variable (name : String)
  | product (e1 e2 : AlgebraicExpression)
  | power (base : AlgebraicExpression) (exponent : ℕ)
  | fraction (numerator denominator : AlgebraicExpression)
  | negation (e : AlgebraicExpression)

/-- Checks if an algebraic expression meets the standard writing requirements -/
def meetsWritingRequirements (e : AlgebraicExpression) : Prop :=
  match e with
  | AlgebraicExpression.constant _ => true
  | AlgebraicExpression.variable _ => true
  | AlgebraicExpression.product e1 e2 => meetsWritingRequirements e1 ∧ meetsWritingRequirements e2
  | AlgebraicExpression.power base exponent => meetsWritingRequirements base ∧ exponent > 0
  | AlgebraicExpression.fraction num den => meetsWritingRequirements num ∧ meetsWritingRequirements den
  | AlgebraicExpression.negation e => meetsWritingRequirements e

/-- The expression -1/3 * x^2 * y -/
def expression : AlgebraicExpression :=
  AlgebraicExpression.negation
    (AlgebraicExpression.fraction
      (AlgebraicExpression.constant 1)
      (AlgebraicExpression.constant 3))

theorem expression_meets_requirements :
  meetsWritingRequirements expression :=
sorry


end expression_meets_requirements_l2026_202669


namespace matrix_inverse_scalar_multiple_l2026_202644

/-- Given a 2x2 matrix B with elements [[4, 5], [3, m]], prove that if B^(-1) = j * B, 
    then m = -4 and j = 1/31 -/
theorem matrix_inverse_scalar_multiple 
  (B : Matrix (Fin 2) (Fin 2) ℝ)
  (h_B : B = !![4, 5; 3, m])
  (h_inv : B⁻¹ = j • B) :
  m = -4 ∧ j = 1 / 31 := by
  sorry

end matrix_inverse_scalar_multiple_l2026_202644


namespace seokmin_school_cookies_l2026_202631

/-- The number of boxes of cookies needed for a given number of students -/
def cookies_boxes_needed (num_students : ℕ) (cookies_per_student : ℕ) (cookies_per_box : ℕ) : ℕ :=
  ((num_students * cookies_per_student + cookies_per_box - 1) / cookies_per_box : ℕ)

/-- Theorem stating the number of boxes needed for Seokmin's school -/
theorem seokmin_school_cookies :
  cookies_boxes_needed 134 7 28 = 34 := by
  sorry

end seokmin_school_cookies_l2026_202631


namespace min_value_at_one_min_value_is_constant_l2026_202666

/-- A quadratic function f(x) = x^2 + (a+2)x + b symmetric about x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + (a+2)*x + b

/-- The property of f being symmetric about x = 1 -/
def symmetric_about_one (a b : ℝ) : Prop :=
  ∀ x, f a b (1 + x) = f a b (1 - x)

/-- The minimum value of f -/
def min_value (a b : ℝ) : ℝ := f a b 1

theorem min_value_at_one (a b : ℝ) 
  (h : symmetric_about_one a b) : 
  ∀ x, f a b x ≥ min_value a b :=
sorry

theorem min_value_is_constant (a b : ℝ) 
  (h : symmetric_about_one a b) : 
  ∃ c, min_value a b = c :=
sorry

end min_value_at_one_min_value_is_constant_l2026_202666


namespace correct_calculation_l2026_202671

theorem correct_calculation (x : ℤ) (h : x - 32 = 33) : x + 32 = 97 := by
  sorry

end correct_calculation_l2026_202671


namespace dark_lord_sword_distribution_l2026_202691

/-- Calculates the weight of swords each orc must carry given the total weight,
    number of squads, and orcs per squad. -/
def weight_per_orc (total_weight : ℕ) (num_squads : ℕ) (orcs_per_squad : ℕ) : ℚ :=
  total_weight / (num_squads * orcs_per_squad)

/-- Proves that given 1200 pounds of swords, 10 squads, and 8 orcs per squad,
    each orc must carry 15 pounds of swords. -/
theorem dark_lord_sword_distribution :
  weight_per_orc 1200 10 8 = 15 := by
  sorry

end dark_lord_sword_distribution_l2026_202691


namespace erased_number_is_202_l2026_202626

-- Define the sequence of consecutive positive integers
def consecutive_sequence (n : ℕ) : List ℕ := List.range n

-- Define the function to calculate the sum of the first n positive integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the function to calculate the average of the remaining numbers after erasing x
def average_after_erasing (n : ℕ) (x : ℕ) : ℚ :=
  (sum_first_n n - x) / (n - 1 : ℚ)

-- The theorem to prove
theorem erased_number_is_202 (n : ℕ) (x : ℕ) :
  x ∈ consecutive_sequence n →
  average_after_erasing n x = 151 / 3 →
  x = 202 := by
  sorry

end erased_number_is_202_l2026_202626


namespace sum_of_squares_equality_l2026_202678

variables {a b c x y z : ℝ}

theorem sum_of_squares_equality 
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 0)
  : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := by
  sorry

end sum_of_squares_equality_l2026_202678


namespace candy_mixture_proof_l2026_202636

/-- Proves that mixing 1 pound of candy A with 4 pounds of candy B
    results in 5 pounds of mixture costing $2 per pound -/
theorem candy_mixture_proof :
  let candy_a_price : ℝ := 3.20
  let candy_b_price : ℝ := 1.70
  let candy_a_amount : ℝ := 1
  let candy_b_amount : ℝ := 4
  let total_amount : ℝ := candy_a_amount + candy_b_amount
  let total_cost : ℝ := candy_a_price * candy_a_amount + candy_b_price * candy_b_amount
  let mixture_price_per_pound : ℝ := total_cost / total_amount
  total_amount = 5 ∧ mixture_price_per_pound = 2 := by
sorry

end candy_mixture_proof_l2026_202636


namespace arithmetic_operations_l2026_202688

theorem arithmetic_operations :
  (-10 + 2 = -8) ∧
  (-6 - 3 = -9) ∧
  ((-4) * 6 = -24) ∧
  ((-15) / 5 = -3) ∧
  ((-4)^2 / 2 = 8) ∧
  (|(-2)| - 2 = 0) := by
  sorry

end arithmetic_operations_l2026_202688


namespace min_value_polynomial_l2026_202635

theorem min_value_polynomial (x : ℝ) :
  ∃ (min : ℝ), min = 2022 - (5 + Real.sqrt 5) / 2 ∧
  ∀ y : ℝ, (y + 1) * (y + 2) * (y + 3) * (y + 4) + y + 2023 ≥ min :=
by sorry

end min_value_polynomial_l2026_202635


namespace floor_of_e_equals_two_l2026_202629

theorem floor_of_e_equals_two : ⌊Real.exp 1⌋ = 2 := by
  sorry

end floor_of_e_equals_two_l2026_202629


namespace abc_product_l2026_202696

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 24 * Real.sqrt 3)
  (hac : a * c = 30 * Real.sqrt 3)
  (hbc : b * c = 40 * Real.sqrt 3) :
  a * b * c = 120 * Real.sqrt 6 := by
sorry

end abc_product_l2026_202696


namespace triangle_property_l2026_202664

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h1 : t.A = 2 * t.B) 
  (h2 : t.A = 3 * t.B) : 
  t.a^2 = t.b * (t.b + t.c) ∧ 
  t.c^2 = (1 / t.b) * (t.a - t.b) * (t.a^2 - t.b^2) :=
by sorry


end triangle_property_l2026_202664


namespace simplify_expression_l2026_202684

theorem simplify_expression (c : ℝ) : ((3 * c + 5) - 3 * c) / 2 = 5 / 2 := by
  sorry

end simplify_expression_l2026_202684


namespace quadratic_roots_sum_l2026_202656

theorem quadratic_roots_sum (α β : ℝ) : 
  (α^2 + 3*α - 7 = 0) → 
  (β^2 + 3*β - 7 = 0) → 
  α^2 + 4*α + β = 4 := by
sorry

end quadratic_roots_sum_l2026_202656


namespace refrigerator_profit_percentage_l2026_202614

/-- Calculates the profit percentage for a refrigerator sale --/
theorem refrigerator_profit_percentage 
  (labelled_price : ℝ)
  (discount_rate : ℝ)
  (purchase_price : ℝ)
  (transport_cost : ℝ)
  (installation_cost : ℝ)
  (selling_price : ℝ)
  (h1 : labelled_price = purchase_price / (1 - discount_rate))
  (h2 : discount_rate = 0.20)
  (h3 : purchase_price = 12500)
  (h4 : transport_cost = 125)
  (h5 : installation_cost = 250)
  (h6 : selling_price = 19200)
  : ∃ (profit_percentage : ℝ), 
    abs (profit_percentage - 49.13) < 0.01 ∧
    profit_percentage = (selling_price - (purchase_price + transport_cost + installation_cost)) / 
                        (purchase_price + transport_cost + installation_cost) * 100 :=
sorry

end refrigerator_profit_percentage_l2026_202614


namespace range_of_m_l2026_202659

-- Define the sets A and B
def A (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}
def B := {x : ℝ | x^2 - 2*x - 15 ≤ 0}

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (∃ x, x ∈ A m) ∧ -- A is non-empty
  (∃ x, x ∈ B) ∧ -- B is non-empty
  (∀ x, x ∈ A m → x ∈ B) → -- A ⊆ B
  2 ≤ m ∧ m ≤ 3 :=
by sorry

end range_of_m_l2026_202659


namespace triangle_height_l2026_202677

/-- Given a triangle with base 6 and area 24, prove its height is 8 -/
theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 6 → 
  area = 24 → 
  area = 1/2 * base * height → 
  height = 8 := by
sorry

end triangle_height_l2026_202677


namespace cloth_cost_price_l2026_202681

/-- Given a trader sells cloth with the following conditions:
    - Total length of cloth sold is 75 meters
    - Total selling price is Rs. 4950
    - Profit per meter is Rs. 15
    This theorem proves that the cost price per meter is Rs. 51 -/
theorem cloth_cost_price (total_length : ℕ) (total_selling_price : ℕ) (profit_per_meter : ℕ) :
  total_length = 75 →
  total_selling_price = 4950 →
  profit_per_meter = 15 →
  (total_selling_price - total_length * profit_per_meter) / total_length = 51 :=
by sorry

end cloth_cost_price_l2026_202681


namespace binomial_coefficient_16_10_l2026_202680

theorem binomial_coefficient_16_10 :
  (Nat.choose 15 8 = 6435) →
  (Nat.choose 15 9 = 5005) →
  (Nat.choose 17 10 = 19448) →
  Nat.choose 16 10 = 8008 := by
  sorry

end binomial_coefficient_16_10_l2026_202680


namespace equation_solution_l2026_202606

theorem equation_solution (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 - b*d) / (b + 2*c + d) + (b^2 - c*a) / (c + 2*d + a) + 
  (c^2 - d*b) / (d + 2*a + b) + (d^2 - a*c) / (a + 2*b + c) = 0 ↔ 
  a = c ∧ b = d := by sorry

end equation_solution_l2026_202606


namespace multiply_and_add_l2026_202623

theorem multiply_and_add : 19 * 42 + 81 * 19 = 2337 := by
  sorry

end multiply_and_add_l2026_202623


namespace fifteen_points_max_planes_l2026_202642

/-- The maximum number of planes determined by n points in space,
    where no four points are coplanar. -/
def maxPlanes (n : ℕ) : ℕ := Nat.choose n 3

/-- The condition that no four points are coplanar is implicitly assumed
    in the definition of maxPlanes. -/
theorem fifteen_points_max_planes :
  maxPlanes 15 = 455 := by sorry

end fifteen_points_max_planes_l2026_202642


namespace problem_1_l2026_202652

theorem problem_1 (m : ℝ) : m * m^3 + (-m^2)^3 / m^2 = 0 := by
  sorry

end problem_1_l2026_202652


namespace geometric_sequence_product_l2026_202676

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- Define the property that a_1 and a_19 are roots of x^2 - 10x + 16 = 0
def roots_property (a : ℕ → ℝ) : Prop :=
  a 1 ^ 2 - 10 * a 1 + 16 = 0 ∧ a 19 ^ 2 - 10 * a 19 + 16 = 0

theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  roots_property a →
  a 8 * a 10 * a 12 = 64 :=
by
  sorry

end geometric_sequence_product_l2026_202676


namespace harolds_class_size_l2026_202628

/-- Represents the number of apples Harold split among classmates -/
def total_apples : ℕ := 15

/-- Represents the number of apples each classmate received -/
def apples_per_classmate : ℕ := 5

/-- Theorem stating the number of people in Harold's class who received apples -/
theorem harolds_class_size : 
  total_apples / apples_per_classmate = 3 := by sorry

end harolds_class_size_l2026_202628


namespace parallel_vectors_proportion_l2026_202618

-- Define the Cartesian coordinate system
def Point := ℝ × ℝ

-- Define points A and B
def A : Point := (-1, -2)
def B : Point := (2, 3)

-- Define vector AB
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector a
def a (k : ℝ) : ℝ × ℝ := (1, k)

-- Theorem statement
theorem parallel_vectors_proportion :
  ∃ k : ℝ, a k = (1, k) ∧ ∃ c : ℝ, c • (AB.1, AB.2) = a k :=
by sorry

end parallel_vectors_proportion_l2026_202618


namespace product_and_divisibility_l2026_202648

theorem product_and_divisibility (n : ℕ) : 
  n = 3 → 
  ((n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720) ∧ 
  ¬(720 % 11 = 0) := by
  sorry

end product_and_divisibility_l2026_202648


namespace dollar_neg_three_four_l2026_202665

-- Define the $ operation
def dollar (x y : ℤ) : ℤ := x * (y + 2) + x * y

-- Theorem statement
theorem dollar_neg_three_four : dollar (-3) 4 = -30 := by
  sorry

end dollar_neg_three_four_l2026_202665


namespace sphere_volume_circumscribing_prism_l2026_202695

/-- The volume of a sphere that circumscribes a rectangular prism with dimensions 2 × 1 × 1 is √6π -/
theorem sphere_volume_circumscribing_prism :
  let l : ℝ := 2
  let w : ℝ := 1
  let h : ℝ := 1
  let diagonal := Real.sqrt (l^2 + w^2 + h^2)
  let radius := diagonal / 2
  let volume := (4/3) * Real.pi * radius^3
  volume = Real.sqrt 6 * Real.pi := by
sorry

end sphere_volume_circumscribing_prism_l2026_202695


namespace N_subset_M_l2026_202682

-- Define the sets M and N
def M : Set ℝ := {x | |x| ≤ 1}
def N : Set ℝ := {y | ∃ x, y = 2^x ∧ x ≤ 0}

-- State the theorem
theorem N_subset_M : N ⊆ M := by
  sorry

end N_subset_M_l2026_202682


namespace solve_system_for_q_l2026_202605

theorem solve_system_for_q (p q : ℚ) 
  (eq1 : 2 * p + 5 * q = 7) 
  (eq2 : 5 * p + 2 * q = 16) : 
  q = 1 / 7 := by
sorry

end solve_system_for_q_l2026_202605


namespace divisors_of_100n4_l2026_202611

-- Define a function to count divisors
def count_divisors (n : ℕ) : ℕ := sorry

-- Define the condition from the problem
def condition (n : ℕ) : Prop :=
  n > 0 ∧ count_divisors (90 * n^2) = 90

-- Theorem statement
theorem divisors_of_100n4 (n : ℕ) (h : condition n) :
  count_divisors (100 * n^4) = 245 :=
sorry

end divisors_of_100n4_l2026_202611


namespace pairball_playing_time_l2026_202641

theorem pairball_playing_time (n : ℕ) (total_time : ℕ) (h1 : n = 7) (h2 : total_time = 105) :
  let players_per_game : ℕ := 2
  let total_child_minutes : ℕ := players_per_game * total_time
  let time_per_child : ℕ := total_child_minutes / n
  time_per_child = 30 := by sorry

end pairball_playing_time_l2026_202641


namespace helmet_discount_percentage_l2026_202616

def original_price : ℝ := 40
def amount_saved : ℝ := 8
def amount_spent : ℝ := 32

theorem helmet_discount_percentage :
  (amount_saved / original_price) * 100 = 20 :=
by
  sorry

end helmet_discount_percentage_l2026_202616


namespace fish_ratio_calculation_l2026_202660

/-- The ratio of tagged fish to total fish in a second catch -/
def fish_ratio (tagged_first : ℕ) (total_second : ℕ) (tagged_second : ℕ) (total_pond : ℕ) : ℚ :=
  tagged_second / total_second

/-- Theorem stating the ratio of tagged fish to total fish in the second catch -/
theorem fish_ratio_calculation :
  let tagged_first : ℕ := 40
  let total_second : ℕ := 50
  let tagged_second : ℕ := 2
  let total_pond : ℕ := 1000
  fish_ratio tagged_first total_second tagged_second total_pond = 1 / 25 := by
  sorry


end fish_ratio_calculation_l2026_202660


namespace robot_max_score_l2026_202686

def initial_iq : ℕ := 25

def point_range : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 10}

def solve_problem (iq : ℕ) (points : ℕ) : Option ℕ :=
  if iq ≥ points then some (iq - points + 1) else none

def max_score (problems : List ℕ) : ℕ :=
  problems.foldl (λ acc p => acc + p) 0

theorem robot_max_score :
  ∃ (problems : List ℕ),
    (∀ p ∈ problems, p ∈ point_range) ∧
    (problems.foldl (λ iq p => (solve_problem iq p).getD 0) initial_iq ≠ 0) ∧
    (max_score problems = 31) ∧
    (∀ (other_problems : List ℕ),
      (∀ p ∈ other_problems, p ∈ point_range) →
      (other_problems.foldl (λ iq p => (solve_problem iq p).getD 0) initial_iq ≠ 0) →
      max_score other_problems ≤ 31) :=
by sorry

end robot_max_score_l2026_202686


namespace grid_toothpick_count_l2026_202617

/-- Calculates the number of toothpicks in a grid with missing pieces -/
def toothpick_count (length width missing_vertical missing_horizontal : ℕ) : ℕ :=
  let vertical_lines := length + 1 - missing_vertical
  let horizontal_lines := width + 1 - missing_horizontal
  (vertical_lines * width) + (horizontal_lines * length)

/-- Theorem stating the correct number of toothpicks in the specific grid -/
theorem grid_toothpick_count :
  toothpick_count 45 25 8 5 = 1895 := by
  sorry

end grid_toothpick_count_l2026_202617


namespace king_will_be_checked_l2026_202685

/-- Represents a chess piece -/
inductive Piece
| King
| Rook

/-- Represents a position on the chessboard -/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the state of the chessboard -/
structure ChessboardState :=
  (kingPos : Position)
  (rookPositions : List Position)

/-- Represents a move in the game -/
inductive Move
| KingMove (newPos : Position)
| RookMove (oldPos : Position) (newPos : Position)

/-- The game ends when the king is in check or reaches the top-right corner -/
def gameEnded (state : ChessboardState) : Prop :=
  (state.kingPos.x = 20 ∧ state.kingPos.y = 20) ∨
  state.rookPositions.any (λ rookPos => rookPos.x = state.kingPos.x ∨ rookPos.y = state.kingPos.y)

/-- A valid game sequence -/
def ValidGameSequence : List Move → Prop :=
  sorry

/-- The theorem to be proved -/
theorem king_will_be_checked
  (initialState : ChessboardState)
  (h1 : initialState.kingPos = ⟨1, 1⟩)
  (h2 : initialState.rookPositions.length = 10)
  (h3 : ∀ pos ∈ initialState.rookPositions, pos.x ≤ 20 ∧ pos.y ≤ 20) :
  ∀ (moves : List Move), ValidGameSequence moves →
    ∃ (n : Nat), let finalState := (moves.take n).foldl (λ s m => sorry) initialState
                 gameEnded finalState :=
sorry

end king_will_be_checked_l2026_202685


namespace rectangular_field_area_l2026_202622

/-- Proves that a rectangular field with sides in ratio 3:4 and fencing cost of 98 rupees at 25 paise per metre has an area of 9408 square meters -/
theorem rectangular_field_area (length width : ℝ) (cost_per_metre : ℚ) (total_cost : ℚ) : 
  length / width = 4 / 3 →
  cost_per_metre = 25 / 100 →
  total_cost = 98 →
  2 * (length + width) * cost_per_metre = total_cost →
  length * width = 9408 := by
  sorry

#check rectangular_field_area

end rectangular_field_area_l2026_202622


namespace tangent_line_theorem_l2026_202646

theorem tangent_line_theorem (a b : ℝ) : 
  (∀ x y : ℝ, y = x^2 + a*x + b) →
  (∀ x y : ℝ, x - y + 1 = 0 ↔ y = b ∧ x = 0) →
  a + b = 2 := by
sorry

end tangent_line_theorem_l2026_202646


namespace fraction_equality_l2026_202639

theorem fraction_equality : (10^9 : ℚ) / (2 * 5^2 * 10^3) = 20000 := by
  sorry

end fraction_equality_l2026_202639


namespace inequality_solution_l2026_202689

theorem inequality_solution (x : ℝ) :
  (∃ a : ℝ, a ∈ Set.Icc (-1) 2 ∧ (2 - a) * x^3 + (1 - 2*a) * x^2 - 6*x + 5 + 4*a - a^2 < 0) ↔
  (x < -2 ∨ (0 < x ∧ x < 1) ∨ x > 1) := by
sorry

end inequality_solution_l2026_202689


namespace fourth_power_roots_l2026_202670

theorem fourth_power_roots (p q : ℝ) (r₁ r₂ : ℂ) : 
  (r₁^2 + p*r₁ + q = 0) → 
  (r₂^2 + p*r₂ + q = 0) → 
  (r₁^4)^2 + ((p^2 - 2*q)^2 - 2*q^2)*(r₁^4) + q^4 = 0 ∧
  (r₂^4)^2 + ((p^2 - 2*q)^2 - 2*q^2)*(r₂^4) + q^4 = 0 := by
  sorry

end fourth_power_roots_l2026_202670


namespace point_on_line_l2026_202600

/-- A point (x, y) lies on the line y = 2x + 1 if y equals 2x + 1 -/
def lies_on_line (x y : ℚ) : Prop := y = 2 * x + 1

/-- The point (-2, -3) lies on the line y = 2x + 1 -/
theorem point_on_line : lies_on_line (-2) (-3) := by sorry

end point_on_line_l2026_202600


namespace fourth_friend_age_l2026_202604

theorem fourth_friend_age (age1 age2 age3 avg : ℕ) (h1 : age1 = 7) (h2 : age2 = 9) (h3 : age3 = 12) 
  (h_avg : (age1 + age2 + age3 + age4) / 4 = avg) (h_avg_val : avg = 9) : age4 = 8 :=
by
  sorry

end fourth_friend_age_l2026_202604


namespace mrs_brier_bakes_18_muffins_l2026_202612

/-- The number of muffins baked by Mrs. Brier's class -/
def mrs_brier_muffins : ℕ := 55 - (20 + 17)

/-- Theorem stating that Mrs. Brier's class bakes 18 muffins -/
theorem mrs_brier_bakes_18_muffins :
  mrs_brier_muffins = 18 := by sorry

end mrs_brier_bakes_18_muffins_l2026_202612


namespace least_k_cube_divisible_by_120_l2026_202624

theorem least_k_cube_divisible_by_120 : 
  ∃ k : ℕ+, k.val = 30 ∧ 
  (∀ m : ℕ+, m.val < k.val → ¬(120 ∣ m.val^3)) ∧ 
  (120 ∣ k.val^3) := by
  sorry

end least_k_cube_divisible_by_120_l2026_202624
