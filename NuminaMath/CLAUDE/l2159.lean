import Mathlib

namespace NUMINAMATH_CALUDE_tetrahedron_properties_l2159_215992

-- Define the vertices of the tetrahedron
def A₁ : ℝ × ℝ × ℝ := (4, -1, 3)
def A₂ : ℝ × ℝ × ℝ := (-2, 1, 0)
def A₃ : ℝ × ℝ × ℝ := (0, -5, 1)
def A₄ : ℝ × ℝ × ℝ := (3, 2, -6)

-- Function to calculate the volume of a tetrahedron
def tetrahedron_volume (a b c d : ℝ × ℝ × ℝ) : ℝ := sorry

-- Function to calculate the height from a point to a plane defined by three points
def height_to_plane (point plane1 plane2 plane3 : ℝ × ℝ × ℝ) : ℝ := sorry

theorem tetrahedron_properties :
  let volume := tetrahedron_volume A₁ A₂ A₃ A₄
  let height := height_to_plane A₄ A₁ A₂ A₃
  volume = 136 / 3 ∧ height = 17 / Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l2159_215992


namespace NUMINAMATH_CALUDE_digit_410_of_7_29_l2159_215960

/-- The decimal expansion of 7/29 has a repeating cycle of 28 digits -/
def cycle_length : ℕ := 28

/-- The repeating cycle of digits in the decimal expansion of 7/29 -/
def repeating_cycle : List ℕ := [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7]

/-- The position we're interested in -/
def target_position : ℕ := 410

theorem digit_410_of_7_29 : 
  (repeating_cycle.get! ((target_position - 1) % cycle_length)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_digit_410_of_7_29_l2159_215960


namespace NUMINAMATH_CALUDE_expand_expression_l2159_215934

theorem expand_expression (y : ℝ) : 5 * (4 * y^3 - 3 * y^2 + 2 * y - 6) = 20 * y^3 - 15 * y^2 + 10 * y - 30 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2159_215934


namespace NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_455_l2159_215951

theorem multiplicative_inverse_123_mod_455 : ∃ x : ℕ, x < 455 ∧ (123 * x) % 455 = 1 :=
by
  use 223
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_123_mod_455_l2159_215951


namespace NUMINAMATH_CALUDE_equal_integers_from_divisor_properties_l2159_215939

/-- The product of all divisors of a natural number -/
noncomputable def productOfDivisors (n : ℕ) : ℕ := sorry

/-- The number of divisors of a natural number -/
def numberOfDivisors (n : ℕ) : ℕ := sorry

theorem equal_integers_from_divisor_properties (m n s : ℕ) 
  (h_m_ge_n : m ≥ n) 
  (h_s_pos : s > 0) 
  (h_product : productOfDivisors (s * m) = productOfDivisors (s * n))
  (h_number : numberOfDivisors (s * m) = numberOfDivisors (s * n)) : 
  m = n :=
sorry

end NUMINAMATH_CALUDE_equal_integers_from_divisor_properties_l2159_215939


namespace NUMINAMATH_CALUDE_election_winner_percentage_l2159_215948

theorem election_winner_percentage (winner_votes loser_votes : ℕ) 
  (h1 : winner_votes = 1344)
  (h2 : winner_votes - loser_votes = 288) :
  (winner_votes : ℚ) / ((winner_votes : ℚ) + (loser_votes : ℚ)) * 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l2159_215948


namespace NUMINAMATH_CALUDE_intersection_when_m_neg_one_subset_iff_m_leq_neg_two_l2159_215958

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m < x ∧ x < 1 - m}

-- Theorem 1: When m = -1, A ∩ B = {x | 1 < x < 2}
theorem intersection_when_m_neg_one :
  A ∩ B (-1) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 2: A ⊆ B if and only if m ≤ -2
theorem subset_iff_m_leq_neg_two :
  ∀ m : ℝ, A ⊆ B m ↔ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_neg_one_subset_iff_m_leq_neg_two_l2159_215958


namespace NUMINAMATH_CALUDE_greatest_common_length_l2159_215959

theorem greatest_common_length (a b c : ℕ) (ha : a = 48) (hb : b = 64) (hc : c = 80) :
  Nat.gcd a (Nat.gcd b c) = 16 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_length_l2159_215959


namespace NUMINAMATH_CALUDE_third_square_perimeter_l2159_215931

/-- Given two squares with perimeters 60 cm and 48 cm, prove that a third square
    whose area is equal to the difference of the areas of the first two squares
    has a perimeter of 36 cm. -/
theorem third_square_perimeter (square1 square2 square3 : ℝ → ℝ) :
  (∀ s, square1 s = s^2) →
  (∀ s, square2 s = s^2) →
  (∀ s, square3 s = s^2) →
  (4 * Real.sqrt (square1 (60 / 4))) = 60 →
  (4 * Real.sqrt (square2 (48 / 4))) = 48 →
  square3 (Real.sqrt (square1 (60 / 4) - square2 (48 / 4))) =
    square1 (60 / 4) - square2 (48 / 4) →
  (4 * Real.sqrt (square3 (Real.sqrt (square1 (60 / 4) - square2 (48 / 4))))) = 36 :=
by sorry

end NUMINAMATH_CALUDE_third_square_perimeter_l2159_215931


namespace NUMINAMATH_CALUDE_carol_trivia_game_points_l2159_215989

theorem carol_trivia_game_points (first_round : ℕ) (last_round : ℤ) (final_score : ℕ) 
  (h1 : first_round = 17)
  (h2 : last_round = -16)
  (h3 : final_score = 7) :
  ∃ second_round : ℕ, (first_round : ℤ) + second_round + last_round = final_score ∧ second_round = 6 := by
  sorry

end NUMINAMATH_CALUDE_carol_trivia_game_points_l2159_215989


namespace NUMINAMATH_CALUDE_find_divisor_l2159_215938

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 52 →
  quotient = 16 →
  remainder = 4 →
  dividend = divisor * quotient + remainder →
  divisor = 3 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2159_215938


namespace NUMINAMATH_CALUDE_complement_M_correct_l2159_215917

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 4 ≤ 0}

-- Define the complement of M in U
def complement_M : Set ℝ := {x : ℝ | x < -2 ∨ x > 2}

-- Theorem statement
theorem complement_M_correct : 
  U \ M = complement_M := by sorry

end NUMINAMATH_CALUDE_complement_M_correct_l2159_215917


namespace NUMINAMATH_CALUDE_quartic_polynomial_satisfies_conditions_l2159_215981

theorem quartic_polynomial_satisfies_conditions :
  let p : ℝ → ℝ := λ x => -x^4 + 2*x^2 - 5*x + 1
  (p 1 = -3) ∧ (p 2 = -5) ∧ (p 3 = -11) ∧ (p 4 = -27) ∧ (p 5 = -59) := by
  sorry

end NUMINAMATH_CALUDE_quartic_polynomial_satisfies_conditions_l2159_215981


namespace NUMINAMATH_CALUDE_lisa_marbles_distribution_l2159_215971

/-- The minimum number of additional marbles needed -/
def additional_marbles_needed (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's distribution -/
theorem lisa_marbles_distribution (num_friends : ℕ) (initial_marbles : ℕ) 
  (h1 : num_friends = 12) (h2 : initial_marbles = 50) : 
  additional_marbles_needed num_friends initial_marbles = 28 := by
  sorry

#eval additional_marbles_needed 12 50  -- Should output 28

end NUMINAMATH_CALUDE_lisa_marbles_distribution_l2159_215971


namespace NUMINAMATH_CALUDE_greatest_int_with_gcd_18_6_l2159_215957

theorem greatest_int_with_gcd_18_6 : 
  (∀ n : ℕ, n < 200 ∧ n > 174 → Nat.gcd n 18 ≠ 6) ∧ 
  Nat.gcd 174 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_int_with_gcd_18_6_l2159_215957


namespace NUMINAMATH_CALUDE_game_C_more_likely_than_D_l2159_215923

/-- Probability of getting heads when tossing the biased coin -/
def p_heads : ℚ := 3/4

/-- Probability of getting tails when tossing the biased coin -/
def p_tails : ℚ := 1/4

/-- Probability of winning Game C -/
def p_win_C : ℚ := p_heads^4

/-- Probability of winning Game D -/
def p_win_D : ℚ := p_heads^5 + p_heads^3 * p_tails^2 + p_tails^3 * p_heads^2 + p_tails^5

theorem game_C_more_likely_than_D : p_win_C - p_win_D = 11/256 := by
  sorry

end NUMINAMATH_CALUDE_game_C_more_likely_than_D_l2159_215923


namespace NUMINAMATH_CALUDE_percentage_problem_l2159_215945

theorem percentage_problem (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = 0.3 * (x + y) →
  y = 0.4 * x →
  P = 70 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l2159_215945


namespace NUMINAMATH_CALUDE_inequality_proof_l2159_215928

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2159_215928


namespace NUMINAMATH_CALUDE_carpenter_needs_eight_more_logs_l2159_215902

/-- Represents the carpenter's log and woodblock problem -/
def CarpenterProblem (total_woodblocks : ℕ) (initial_logs : ℕ) (woodblocks_per_log : ℕ) : Prop :=
  let initial_woodblocks := initial_logs * woodblocks_per_log
  let remaining_woodblocks := total_woodblocks - initial_woodblocks
  remaining_woodblocks % woodblocks_per_log = 0 ∧
  remaining_woodblocks / woodblocks_per_log = 8

/-- The carpenter needs 8 more logs to reach the required 80 woodblocks -/
theorem carpenter_needs_eight_more_logs :
  CarpenterProblem 80 8 5 := by
  sorry

#check carpenter_needs_eight_more_logs

end NUMINAMATH_CALUDE_carpenter_needs_eight_more_logs_l2159_215902


namespace NUMINAMATH_CALUDE_inequality_proof_l2159_215949

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c ≥ 1) :
  1 / (2 + a) + 1 / (2 + b) + 1 / (2 + c) ≤ 1 ∧
  (1 / (2 + a) + 1 / (2 + b) + 1 / (2 + c) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2159_215949


namespace NUMINAMATH_CALUDE_remainder_2519_div_7_l2159_215905

theorem remainder_2519_div_7 : 2519 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_7_l2159_215905


namespace NUMINAMATH_CALUDE_trig_identity_l2159_215991

theorem trig_identity (α : Real) (h : Real.sin (α + 7 * Real.pi / 6) = 1) :
  Real.cos (2 * α - 2 * Real.pi / 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2159_215991


namespace NUMINAMATH_CALUDE_winning_strategy_iff_not_div_four_l2159_215962

/-- A game where two players take turns removing stones from a pile. -/
structure StoneGame where
  n : ℕ  -- Initial number of stones

/-- Represents a valid move in the game -/
inductive ValidMove : ℕ → ℕ → Prop where
  | prime_divisor {n m : ℕ} (h : m.Prime) (d : m ∣ n) : ValidMove n m
  | one {n : ℕ} : ValidMove n 1

/-- Defines a winning strategy for the first player -/
def has_winning_strategy (game : StoneGame) : Prop :=
  ∃ (strategy : ℕ → ℕ),
    ∀ (opponent_move : ℕ → ℕ),
      ValidMove game.n (strategy game.n) ∧
      (∀ k, k < game.n →
        ValidMove k (opponent_move k) →
          ValidMove (k - opponent_move k) (strategy (k - opponent_move k)))

/-- The main theorem: The first player has a winning strategy iff n is not divisible by 4 -/
theorem winning_strategy_iff_not_div_four (game : StoneGame) :
  has_winning_strategy game ↔ ¬(4 ∣ game.n) :=
sorry

end NUMINAMATH_CALUDE_winning_strategy_iff_not_div_four_l2159_215962


namespace NUMINAMATH_CALUDE_factors_of_1320_l2159_215913

/-- The number of distinct, positive factors of 1320 -/
def num_factors_1320 : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating that the number of distinct, positive factors of 1320 is 32 -/
theorem factors_of_1320 : num_factors_1320 = 32 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1320_l2159_215913


namespace NUMINAMATH_CALUDE_three_lines_intersection_l2159_215904

/-- The curve (x + 2y + a)(x^2 - y^2) = 0 represents three lines intersecting at a single point if and only if a = 0 -/
theorem three_lines_intersection (a : ℝ) : 
  (∃! p : ℝ × ℝ, ∀ x y : ℝ, (x + 2*y + a)*(x^2 - y^2) = 0 ↔ 
    (x = p.1 ∧ y = p.2) ∨ (x = -y ∧ x = p.1) ∨ (x = y ∧ x = p.1)) ↔ 
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l2159_215904


namespace NUMINAMATH_CALUDE_circle_through_origin_equation_l2159_215977

/-- Defines a circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def onCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem circle_through_origin_equation 
  (c : Circle) 
  (h1 : c.center = (3, 4)) 
  (h2 : onCircle c (0, 0)) : 
  ∀ (x y : ℝ), onCircle c (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_circle_through_origin_equation_l2159_215977


namespace NUMINAMATH_CALUDE_num_divisors_2_pow_7_num_divisors_5_pow_4_num_divisors_2_pow_7_mul_5_pow_4_num_divisors_2_pow_m_mul_5_pow_n_mul_3_pow_k_num_divisors_3600_num_divisors_42_pow_5_l2159_215964

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

-- Theorem for 2^7
theorem num_divisors_2_pow_7 : num_divisors (2^7) = 8 := by sorry

-- Theorem for 5^4
theorem num_divisors_5_pow_4 : num_divisors (5^4) = 5 := by sorry

-- Theorem for 2^7 * 5^4
theorem num_divisors_2_pow_7_mul_5_pow_4 : num_divisors (2^7 * 5^4) = 40 := by sorry

-- Theorem for 2^m * 5^n * 3^k
theorem num_divisors_2_pow_m_mul_5_pow_n_mul_3_pow_k (m n k : ℕ) :
  num_divisors (2^m * 5^n * 3^k) = (m + 1) * (n + 1) * (k + 1) := by sorry

-- Theorem for 3600
theorem num_divisors_3600 : num_divisors 3600 = 45 := by sorry

-- Theorem for 42^5
theorem num_divisors_42_pow_5 : num_divisors (42^5) = 216 := by sorry

end NUMINAMATH_CALUDE_num_divisors_2_pow_7_num_divisors_5_pow_4_num_divisors_2_pow_7_mul_5_pow_4_num_divisors_2_pow_m_mul_5_pow_n_mul_3_pow_k_num_divisors_3600_num_divisors_42_pow_5_l2159_215964


namespace NUMINAMATH_CALUDE_sector_area_l2159_215965

/-- The area of a sector with central angle 150° and radius 3 is 15π/4 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = 150 * π / 180) (h2 : r = 3) :
  (1/2) * r^2 * θ = 15 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2159_215965


namespace NUMINAMATH_CALUDE_raspberry_pies_l2159_215941

theorem raspberry_pies (total_pies : ℝ) (peach_ratio strawberry_ratio raspberry_ratio : ℝ) :
  total_pies = 36 ∧
  peach_ratio = 2 ∧
  strawberry_ratio = 5 ∧
  raspberry_ratio = 3 →
  (raspberry_ratio / (peach_ratio + strawberry_ratio + raspberry_ratio)) * total_pies = 10.8 :=
by sorry

end NUMINAMATH_CALUDE_raspberry_pies_l2159_215941


namespace NUMINAMATH_CALUDE_vector_collinearity_l2159_215966

/-- Given vectors a, b, and c in ℝ², prove that if k*a + b is collinear with c, then k = -26/15 -/
theorem vector_collinearity (a b c : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (2, 3)) (h3 : c = (4, -7)) :
  (∃ (k : ℝ), ∃ (t : ℝ), t • c = k • a + b) → 
  (∃ (k : ℝ), k • a + b = (-26/15) • a + b) :=
by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2159_215966


namespace NUMINAMATH_CALUDE_dans_music_store_spending_l2159_215929

/-- The amount Dan spent at the music store -/
def amount_spent (clarinet_cost song_book_cost amount_left : ℚ) : ℚ :=
  clarinet_cost + song_book_cost - amount_left

/-- Proof that Dan spent $129.22 at the music store -/
theorem dans_music_store_spending :
  amount_spent 130.30 11.24 12.32 = 129.22 := by
  sorry

end NUMINAMATH_CALUDE_dans_music_store_spending_l2159_215929


namespace NUMINAMATH_CALUDE_power_inequality_l2159_215926

theorem power_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^2 > x + y) (h2 : x^4 > x^3 + y) : x^3 > x^2 + y := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2159_215926


namespace NUMINAMATH_CALUDE_power_of_two_mod_nine_periodic_l2159_215900

/-- The sequence of remainders when powers of 2 are divided by 9 is periodic with period 6 -/
theorem power_of_two_mod_nine_periodic :
  ∃ (p : ℕ), p > 0 ∧ ∀ (n : ℕ), (2^(n + p) : ℕ) % 9 = (2^n : ℕ) % 9 ∧ p = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_mod_nine_periodic_l2159_215900


namespace NUMINAMATH_CALUDE_sum_220_is_5500_div_3_l2159_215967

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  /-- The first term of the progression -/
  a : ℚ
  /-- The common difference of the progression -/
  d : ℚ
  /-- The sum of the first 20 terms is 500 -/
  sum_20 : (20 : ℚ) / 2 * (2 * a + (19 : ℚ) * d) = 500
  /-- The sum of the first 200 terms is 2000 -/
  sum_200 : (200 : ℚ) / 2 * (2 * a + (199 : ℚ) * d) = 2000

/-- The sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℚ) : ℚ :=
  n / 2 * (2 * ap.a + (n - 1) * ap.d)

/-- Theorem: The sum of the first 220 terms is 5500/3 -/
theorem sum_220_is_5500_div_3 (ap : ArithmeticProgression) :
  sum_n ap 220 = 5500 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_220_is_5500_div_3_l2159_215967


namespace NUMINAMATH_CALUDE_max_product_combination_l2159_215954

def digits : List Nat := [1, 3, 5, 8, 9]

def is_valid_combination (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product_combination :
  ∀ a b c d e : Nat,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 8 3 1 9 5 :=
by sorry

end NUMINAMATH_CALUDE_max_product_combination_l2159_215954


namespace NUMINAMATH_CALUDE_sum_coordinates_of_X_l2159_215963

/-- Given three points X, Y, and Z in the plane satisfying certain conditions,
    prove that the sum of the coordinates of X is -28. -/
theorem sum_coordinates_of_X (X Y Z : ℝ × ℝ) : 
  (∃ (k : ℝ), k = 1/2 ∧ Z - X = k • (Y - X) ∧ Y - Z = k • (Y - X)) → 
  Y = (3, 9) →
  Z = (1, -9) →
  X.1 + X.2 = -28 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_X_l2159_215963


namespace NUMINAMATH_CALUDE_tom_battery_usage_l2159_215986

theorem tom_battery_usage (flashlights : Nat) (toys : Nat) (controllers : Nat)
  (batteries_per_flashlight : Nat) (batteries_per_toy : Nat) (batteries_per_controller : Nat)
  (h1 : flashlights = 3)
  (h2 : toys = 5)
  (h3 : controllers = 6)
  (h4 : batteries_per_flashlight = 2)
  (h5 : batteries_per_toy = 4)
  (h6 : batteries_per_controller = 2) :
  flashlights * batteries_per_flashlight +
  toys * batteries_per_toy +
  controllers * batteries_per_controller = 38 := by
  sorry

#check tom_battery_usage

end NUMINAMATH_CALUDE_tom_battery_usage_l2159_215986


namespace NUMINAMATH_CALUDE_problem_solution_l2159_215919

theorem problem_solution (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) :
  (b > 1) ∧ 
  (∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → a + b ≤ x + y) ∧
  (a * b = 16 ∧ ∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → 16 ≤ x * y) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2159_215919


namespace NUMINAMATH_CALUDE_solve_for_m_l2159_215942

-- Define the equation
def is_quadratic (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, (m + 1) * x^(|m| + 1) + 6 * m * x - 2 = a * x^2 + b * x + c

-- Theorem statement
theorem solve_for_m :
  ∀ m : ℝ, is_quadratic m ∧ m + 1 ≠ 0 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l2159_215942


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_parabola_l2159_215983

def f (x : ℝ) : ℝ := (x - 3)^2 - 8

theorem minimum_point_of_translated_parabola :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x₀ ≤ f x ∧ x₀ = 3 ∧ f x₀ = -8 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_parabola_l2159_215983


namespace NUMINAMATH_CALUDE_convex_pentagon_inner_lattice_point_l2159_215984

/-- A point in the 2D Cartesian plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A pentagon defined by five points -/
structure Pentagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point

/-- Checks if a pentagon is convex -/
def isConvex (p : Pentagon) : Prop :=
  sorry

/-- Checks if a point is a lattice point -/
def isLatticePoint (p : Point) : Prop :=
  sorry

/-- Constructs the inner pentagon formed by the intersection of diagonals -/
def innerPentagon (p : Pentagon) : Pentagon :=
  sorry

/-- Checks if a point is inside or on the boundary of a pentagon -/
def isInOrOnPentagon (point : Point) (p : Pentagon) : Prop :=
  sorry

/-- The main theorem -/
theorem convex_pentagon_inner_lattice_point (p : Pentagon) :
  isConvex p →
  isLatticePoint p.A ∧ isLatticePoint p.B ∧ isLatticePoint p.C ∧ isLatticePoint p.D ∧ isLatticePoint p.E →
  ∃ (point : Point), isLatticePoint point ∧ isInOrOnPentagon point (innerPentagon p) :=
sorry

end NUMINAMATH_CALUDE_convex_pentagon_inner_lattice_point_l2159_215984


namespace NUMINAMATH_CALUDE_greatest_angle_in_triangle_l2159_215922

theorem greatest_angle_in_triangle (a b c : ℝ) (h : (b / (c - a)) - (a / (b + c)) = 1) :
  ∃ (A B C : ℝ), 
    A + B + C = 180 ∧ 
    0 < A ∧ 0 < B ∧ 0 < C ∧
    a^2 = b^2 + c^2 - 2*b*c*Real.cos A ∧
    b^2 = a^2 + c^2 - 2*a*c*Real.cos B ∧
    c^2 = a^2 + b^2 - 2*a*b*Real.cos C ∧
    max A (max B C) = 120 := by
  sorry

end NUMINAMATH_CALUDE_greatest_angle_in_triangle_l2159_215922


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2159_215911

theorem quadratic_inequality_solution (a : ℝ) (x : ℝ) :
  a * x^2 - 2 ≥ 2 * x - a * x ↔
    (a = 0 ∧ x ≤ -1) ∨
    (a > 0 ∧ (x ≥ 2 / a ∨ x ≤ -1)) ∨
    (-2 < a ∧ a < 0 ∧ 2 / a ≤ x ∧ x ≤ -1) ∨
    (a = -2 ∧ x = -1) ∨
    (a < -2 ∧ -1 ≤ x ∧ x ≤ 2 / a) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2159_215911


namespace NUMINAMATH_CALUDE_jakes_comic_books_l2159_215988

/-- Jake's comic book problem -/
theorem jakes_comic_books (jake_books : ℕ) (total_books : ℕ) (brother_books : ℕ) : 
  jake_books = 36 →
  total_books = 87 →
  brother_books > jake_books →
  total_books = jake_books + brother_books →
  brother_books - jake_books = 15 := by
sorry

end NUMINAMATH_CALUDE_jakes_comic_books_l2159_215988


namespace NUMINAMATH_CALUDE_no_positive_integer_divisible_by_its_square_plus_one_l2159_215946

theorem no_positive_integer_divisible_by_its_square_plus_one :
  ∀ n : ℕ, n > 0 → ¬(n^2 + 1 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_divisible_by_its_square_plus_one_l2159_215946


namespace NUMINAMATH_CALUDE_cable_length_l2159_215955

/-- Given a curve in 3D space defined by the system of equations:
    x + y + z = 10
    xy + yz + xz = 18
    This theorem states that the length of the curve is 4π√(23/3) -/
theorem cable_length (x y z : ℝ) 
  (eq1 : x + y + z = 10)
  (eq2 : x * y + y * z + x * z = 18) : 
  ∃ (curve_length : ℝ), curve_length = 4 * Real.pi * Real.sqrt (23 / 3) :=
by sorry

end NUMINAMATH_CALUDE_cable_length_l2159_215955


namespace NUMINAMATH_CALUDE_jim_age_l2159_215918

theorem jim_age (jim fred sam : ℕ) 
  (h1 : jim = 2 * fred)
  (h2 : fred = sam + 9)
  (h3 : jim - 6 = 5 * (sam - 6)) :
  jim = 46 := by sorry

end NUMINAMATH_CALUDE_jim_age_l2159_215918


namespace NUMINAMATH_CALUDE_map_distance_l2159_215956

/-- Given a map scale where 0.6 cm represents 6.6 km, and an actual distance of 885.5 km
    between two points, the distance between these points on the map is 80.5 cm. -/
theorem map_distance (scale_map : Real) (scale_actual : Real) (actual_distance : Real) :
  scale_map = 0.6 ∧ scale_actual = 6.6 ∧ actual_distance = 885.5 →
  (actual_distance / (scale_actual / scale_map)) = 80.5 := by
  sorry

#check map_distance

end NUMINAMATH_CALUDE_map_distance_l2159_215956


namespace NUMINAMATH_CALUDE_billys_restaurant_bill_l2159_215996

/-- Calculates the total bill for three families at Billy's Restaurant -/
theorem billys_restaurant_bill (adult_meal_cost child_meal_cost drink_cost : ℕ) 
  (family1_adults family1_children : ℕ)
  (family2_adults family2_children : ℕ)
  (family3_adults family3_children : ℕ) :
  adult_meal_cost = 8 →
  child_meal_cost = 5 →
  drink_cost = 2 →
  family1_adults = 2 →
  family1_children = 3 →
  family2_adults = 4 →
  family2_children = 2 →
  family3_adults = 3 →
  family3_children = 4 →
  (family1_adults * adult_meal_cost + family1_children * child_meal_cost + 
   (family1_adults + family1_children) * drink_cost) +
  (family2_adults * adult_meal_cost + family2_children * child_meal_cost + 
   (family2_adults + family2_children) * drink_cost) +
  (family3_adults * adult_meal_cost + family3_children * child_meal_cost + 
   (family3_adults + family3_children) * drink_cost) = 153 :=
by
  sorry

end NUMINAMATH_CALUDE_billys_restaurant_bill_l2159_215996


namespace NUMINAMATH_CALUDE_time_in_terms_of_angle_and_angular_velocity_l2159_215924

theorem time_in_terms_of_angle_and_angular_velocity 
  (α ω ω₀ θ t : ℝ) 
  (h1 : ω = α * t + ω₀) 
  (h2 : θ = (1/2) * α * t^2 + ω₀ * t) : 
  t = 2 * θ / (ω + ω₀) := by
sorry

end NUMINAMATH_CALUDE_time_in_terms_of_angle_and_angular_velocity_l2159_215924


namespace NUMINAMATH_CALUDE_distribute_six_among_three_l2159_215997

/-- The number of ways to distribute n positions among k schools -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n positions among k schools,
    where each school receives at least one position -/
def distributeAtLeastOne (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute n positions among k schools,
    where each school receives at least one position and
    the number of positions for each school is distinct -/
def distributeDistinct (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 6 ways to distribute 6 positions among 3 schools,
    where each school receives at least one position and
    the number of positions for each school is distinct -/
theorem distribute_six_among_three : distributeDistinct 6 3 = 6 := by sorry

end NUMINAMATH_CALUDE_distribute_six_among_three_l2159_215997


namespace NUMINAMATH_CALUDE_saras_result_unique_l2159_215950

/-- Represents a student's exam results -/
structure ExamResult where
  correct : ℕ
  wrong : ℕ
  unanswered : ℕ

/-- Calculates the score based on the exam result -/
def calculateScore (result : ExamResult) : ℕ :=
  30 + 5 * result.correct - 2 * result.wrong - result.unanswered

/-- Theorem: Sara's exam result is uniquely determined -/
theorem saras_result_unique :
  ∃! result : ExamResult,
    result.correct + result.wrong + result.unanswered = 30 ∧
    calculateScore result = 90 ∧
    (∀ s : ℕ, 85 < s ∧ s < 90 → 
      ∃ r1 r2 : ExamResult, r1 ≠ r2 ∧ 
        calculateScore r1 = s ∧ 
        calculateScore r2 = s ∧
        r1.correct + r1.wrong + r1.unanswered = 30 ∧
        r2.correct + r2.wrong + r2.unanswered = 30) ∧
    result.correct = 12 := by
  sorry

#check saras_result_unique

end NUMINAMATH_CALUDE_saras_result_unique_l2159_215950


namespace NUMINAMATH_CALUDE_gold_coins_puzzle_l2159_215903

theorem gold_coins_puzzle (n c : ℕ) 
  (h1 : n = 9 * (c - 2))  -- Condition 1: 9 coins per chest, 2 empty chests
  (h2 : n = 6 * c + 3)    -- Condition 2: 6 coins per chest, 3 coins leftover
  : n = 45 := by
  sorry

end NUMINAMATH_CALUDE_gold_coins_puzzle_l2159_215903


namespace NUMINAMATH_CALUDE_modulus_of_z_l2159_215985

open Complex

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2159_215985


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2159_215995

theorem rationalize_denominator : 7 / Real.sqrt 63 = Real.sqrt 7 / 3 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2159_215995


namespace NUMINAMATH_CALUDE_first_group_count_l2159_215925

theorem first_group_count (avg_first : ℝ) (avg_second : ℝ) (count_second : ℕ) (avg_all : ℝ)
  (h1 : avg_first = 20)
  (h2 : avg_second = 30)
  (h3 : count_second = 20)
  (h4 : avg_all = 24) :
  ∃ (count_first : ℕ), 
    (count_first : ℝ) * avg_first + (count_second : ℝ) * avg_second = 
    (count_first + count_second : ℝ) * avg_all ∧ count_first = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_group_count_l2159_215925


namespace NUMINAMATH_CALUDE_parabola_intersection_condition_l2159_215973

-- Define the parabola function
def parabola (m : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(m-1)*x + m + 1

-- Theorem statement
theorem parabola_intersection_condition (m : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b < 0 ∧ parabola m a = 0 ∧ parabola m b = 0) ↔ m > -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_condition_l2159_215973


namespace NUMINAMATH_CALUDE_tracy_initial_balloons_l2159_215998

theorem tracy_initial_balloons : 
  ∀ T : ℕ,
  (T + 24) / 2 + 20 = 35 →
  T = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_tracy_initial_balloons_l2159_215998


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_2800_l2159_215980

def largest_perfect_square_factor (n : ℕ) : ℕ := sorry

theorem largest_perfect_square_factor_of_2800 :
  largest_perfect_square_factor 2800 = 400 := by sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_2800_l2159_215980


namespace NUMINAMATH_CALUDE_expression_simplification_l2159_215970

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 - 3) :
  (a - 3) / (a^2 + 6*a + 9) / (1 - 6 / (a + 3)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2159_215970


namespace NUMINAMATH_CALUDE_modulus_of_z_l2159_215994

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2159_215994


namespace NUMINAMATH_CALUDE_subtraction_problem_l2159_215990

theorem subtraction_problem (x : ℤ) : 821 - x = 267 → x - 267 = 287 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_problem_l2159_215990


namespace NUMINAMATH_CALUDE_total_pools_l2159_215901

def arkAndAthleticPools : ℕ := 200
def poolSupplyMultiplier : ℕ := 3

theorem total_pools : arkAndAthleticPools + poolSupplyMultiplier * arkAndAthleticPools = 800 := by
  sorry

end NUMINAMATH_CALUDE_total_pools_l2159_215901


namespace NUMINAMATH_CALUDE_pencil_count_l2159_215976

theorem pencil_count (people notebooks_per_person pencil_multiplier : ℕ) 
  (h1 : people = 6)
  (h2 : notebooks_per_person = 9)
  (h3 : pencil_multiplier = 6) :
  people * notebooks_per_person * pencil_multiplier = 324 :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l2159_215976


namespace NUMINAMATH_CALUDE_pencil_case_problem_l2159_215975

theorem pencil_case_problem (total : ℕ) (difference : ℕ) (erasers : ℕ) : 
  total = 240 →
  difference = 2 →
  erasers + (erasers - difference) = total →
  erasers = 121 := by
  sorry

end NUMINAMATH_CALUDE_pencil_case_problem_l2159_215975


namespace NUMINAMATH_CALUDE_only_fourth_statement_correct_l2159_215999

/-- Represents a programming statement --/
inductive Statement
| Input (s : String)
| Output (s : String)

/-- Checks if an input statement is syntactically correct --/
def isValidInputStatement (s : Statement) : Prop :=
  match s with
  | Statement.Input str => ∃ (vars : List String), str = s!"INPUT {String.intercalate ", " vars}"
  | _ => False

/-- Checks if an output statement is syntactically correct --/
def isValidOutputStatement (s : Statement) : Prop :=
  match s with
  | Statement.Output str => ∃ (expr : String), str = s!"PRINT {expr}"
  | _ => False

/-- The given statements --/
def statements : List Statement :=
  [Statement.Input "a; b; c",
   Statement.Input "x=3",
   Statement.Output "\"A=4\"",
   Statement.Output "3*2"]

/-- Theorem stating that only the fourth statement is correct --/
theorem only_fourth_statement_correct :
  ∃! (i : Fin 4), isValidInputStatement (statements[i.val]) ∨ 
                  isValidOutputStatement (statements[i.val]) :=
by sorry

end NUMINAMATH_CALUDE_only_fourth_statement_correct_l2159_215999


namespace NUMINAMATH_CALUDE_gcd_204_85_l2159_215952

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_204_85_l2159_215952


namespace NUMINAMATH_CALUDE_ceiling_minus_value_l2159_215909

theorem ceiling_minus_value (x : ℝ) (h : ⌈(2 * x)⌉ - ⌊(2 * x)⌋ = 0) : 
  ⌈(2 * x)⌉ - (2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_value_l2159_215909


namespace NUMINAMATH_CALUDE_rectangle_area_l2159_215947

theorem rectangle_area (p1 p2 p3 p4 : ℝ × ℝ) : 
  p1 = (-8, 1) → p2 = (1, 1) → p3 = (1, -7) → p4 = (-8, -7) →
  let length := |p2.1 - p1.1|
  let width := |p1.2 - p4.2|
  length * width = 72 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2159_215947


namespace NUMINAMATH_CALUDE_equation_solution_l2159_215907

theorem equation_solution : ∃ x : ℚ, x ≠ 1 ∧ (x^2 - 2*x + 3) / (x - 1) = x + 4 ∧ x = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2159_215907


namespace NUMINAMATH_CALUDE_rectangle_tiling_l2159_215987

/-- 
Given a rectangle with dimensions m × n that can be tiled with a combination of 
vertical b × 1 tiles and horizontal 1 × a tiles, prove that either b divides m or a divides n.
-/
theorem rectangle_tiling (m n a b : ℕ) 
  (h_tiling : ∃ (v h : ℕ), m * n = v * b + h * a) : 
  b ∣ m ∨ a ∣ n :=
sorry

end NUMINAMATH_CALUDE_rectangle_tiling_l2159_215987


namespace NUMINAMATH_CALUDE_four_balls_three_boxes_l2159_215915

/-- The number of ways to distribute n distinct balls into k distinct boxes,
    with each box containing at least one ball. -/
def distributeWays (n k : ℕ) : ℕ :=
  sorry

theorem four_balls_three_boxes :
  distributeWays 4 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_four_balls_three_boxes_l2159_215915


namespace NUMINAMATH_CALUDE_valid_array_iff_even_l2159_215908

/-- Represents an n × n array with entries -1, 0, or 1 -/
def ValidArray (n : ℕ) := Matrix (Fin n) (Fin n) (Fin 3)

/-- Checks if all 2n sums of rows and columns are different -/
def HasAllDifferentSums (A : ValidArray n) : Prop := sorry

/-- The main theorem: such an array exists if and only if n is even -/
theorem valid_array_iff_even (n : ℕ) :
  (∃ A : ValidArray n, HasAllDifferentSums A) ↔ Even n := by sorry

end NUMINAMATH_CALUDE_valid_array_iff_even_l2159_215908


namespace NUMINAMATH_CALUDE_beka_jackson_miles_difference_l2159_215953

/-- The difference in miles flown between two people -/
def miles_difference (beka_miles jackson_miles : ℕ) : ℕ :=
  beka_miles - jackson_miles

/-- Theorem stating the difference in miles flown between Beka and Jackson -/
theorem beka_jackson_miles_difference :
  miles_difference 873 563 = 310 := by
  sorry

end NUMINAMATH_CALUDE_beka_jackson_miles_difference_l2159_215953


namespace NUMINAMATH_CALUDE_pond_to_field_ratio_l2159_215920

theorem pond_to_field_ratio : 
  let field_length : ℝ := 48
  let field_width : ℝ := 24
  let pond_side : ℝ := 8
  let field_area : ℝ := field_length * field_width
  let pond_area : ℝ := pond_side * pond_side
  pond_area / field_area = 1 / 18 := by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_ratio_l2159_215920


namespace NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l2159_215906

theorem sin_sum_inverse_sin_tan : 
  Real.sin (Real.arcsin (4/5) + Real.arctan 3) = 13 * Real.sqrt 10 / 50 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_inverse_sin_tan_l2159_215906


namespace NUMINAMATH_CALUDE_factorization_equality_l2159_215982

theorem factorization_equality (a : ℝ) : (a + 2) * (a - 2) - 3 * a = (a - 4) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2159_215982


namespace NUMINAMATH_CALUDE_function_identity_l2159_215933

open Real

theorem function_identity (f : ℝ → ℝ) (h : ∀ x, f (1 - cos x) = sin x ^ 2) :
  ∀ x, f x = 2 * x - x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l2159_215933


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l2159_215935

/-- Given a hyperbola and a parabola with specific properties, prove that p = 1 -/
theorem hyperbola_parabola_intersection (a b p : ℝ) : 
  a > 0 → b > 0 → p > 0 →
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y, y^2 = 2 * p * x) →
  (a^2 + b^2) / a^2 = 4 →
  1/2 * (p/2) * (b*p/a) = Real.sqrt 3 / 4 →
  p = 1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l2159_215935


namespace NUMINAMATH_CALUDE_letter_value_proof_l2159_215937

/-- Given random integer values for letters of the alphabet, prove that A = 16 -/
theorem letter_value_proof (M A T E : ℤ) : 
  M + A + T + 8 = 28 →
  T + E + A + M = 34 →
  M + E + E + T = 30 →
  A = 16 := by
  sorry

end NUMINAMATH_CALUDE_letter_value_proof_l2159_215937


namespace NUMINAMATH_CALUDE_max_log_product_l2159_215912

theorem max_log_product (x y : ℝ) (hx : x > 1) (hy : y > 1) (h_sum : Real.log x / Real.log 10 + Real.log y / Real.log 10 = 4) :
  (Real.log x / Real.log 10) * (Real.log y / Real.log 10) ≤ 4 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 1 ∧ y₀ > 1 ∧
    Real.log x₀ / Real.log 10 + Real.log y₀ / Real.log 10 = 4 ∧
    (Real.log x₀ / Real.log 10) * (Real.log y₀ / Real.log 10) = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_log_product_l2159_215912


namespace NUMINAMATH_CALUDE_solution_y_l2159_215972

theorem solution_y (y : ℝ) : 
  (y / 5) / 3 = 15 / (y / 3) → y = 15 * Real.sqrt 3 ∨ y = -15 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_y_l2159_215972


namespace NUMINAMATH_CALUDE_decimal_parts_sum_l2159_215914

theorem decimal_parts_sum (a b : ℝ) : 
  (∃ n : ℕ, n^2 < 5 ∧ (n+1)^2 > 5 ∧ a = Real.sqrt 5 - n) →
  (∃ m : ℕ, m^2 < 13 ∧ (m+1)^2 > 13 ∧ b = Real.sqrt 13 - m) →
  a + b - Real.sqrt 5 = Real.sqrt 13 - 5 := by
sorry

end NUMINAMATH_CALUDE_decimal_parts_sum_l2159_215914


namespace NUMINAMATH_CALUDE_function_extrema_condition_l2159_215968

def f (a x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

theorem function_extrema_condition (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ f a max ∧ f a min ≤ f a x) →
  (a < 1/3 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_condition_l2159_215968


namespace NUMINAMATH_CALUDE_T_always_one_smallest_n_correct_l2159_215916

/-- Definition of T_n -/
def T (n : ℕ+) : ℚ :=
  (Finset.filter (fun i => i ≠ 0) (Finset.range 2)).sum (fun i => 1 / i)

/-- Theorem: T_n is always 1 for any positive integer n -/
theorem T_always_one (n : ℕ+) : T n = 1 := by
  sorry

/-- The smallest positive integer n for which T_n is an integer -/
def smallest_n : ℕ+ := 1

/-- Theorem: smallest_n is indeed the smallest positive integer for which T_n is an integer -/
theorem smallest_n_correct : 
  ∀ k : ℕ+, (∃ m : ℤ, T k = m) → k ≥ smallest_n := by
  sorry

end NUMINAMATH_CALUDE_T_always_one_smallest_n_correct_l2159_215916


namespace NUMINAMATH_CALUDE_third_term_coefficient_a_plus_b_10_l2159_215921

def binomial_coefficient (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem third_term_coefficient_a_plus_b_10 :
  binomial_coefficient 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_third_term_coefficient_a_plus_b_10_l2159_215921


namespace NUMINAMATH_CALUDE_connie_total_markers_l2159_215944

/-- The number of red markers Connie has -/
def red_markers : ℕ := 5230

/-- The number of blue markers Connie has -/
def blue_markers : ℕ := 4052

/-- The number of green markers Connie has -/
def green_markers : ℕ := 3180

/-- The number of purple markers Connie has -/
def purple_markers : ℕ := 2763

/-- The total number of markers Connie has -/
def total_markers : ℕ := red_markers + blue_markers + green_markers + purple_markers

theorem connie_total_markers : total_markers = 15225 := by
  sorry

end NUMINAMATH_CALUDE_connie_total_markers_l2159_215944


namespace NUMINAMATH_CALUDE_dropped_student_score_l2159_215943

theorem dropped_student_score
  (initial_students : ℕ)
  (initial_average : ℚ)
  (remaining_students : ℕ)
  (new_average : ℚ)
  (h1 : initial_students = 16)
  (h2 : initial_average = 62.5)
  (h3 : remaining_students = 15)
  (h4 : new_average = 63)
  (h5 : remaining_students = initial_students - 1) :
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * new_average = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_dropped_student_score_l2159_215943


namespace NUMINAMATH_CALUDE_cage_cost_proof_l2159_215974

def cat_toy_cost : ℝ := 10.22
def total_cost : ℝ := 21.95

theorem cage_cost_proof : total_cost - cat_toy_cost = 11.73 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_proof_l2159_215974


namespace NUMINAMATH_CALUDE_greatest_common_divisor_and_digit_sum_l2159_215993

def a : ℕ := 1305
def b : ℕ := 4665
def c : ℕ := 6905

def diff1 : ℕ := b - a
def diff2 : ℕ := c - b
def diff3 : ℕ := c - a

def n : ℕ := Nat.gcd diff1 (Nat.gcd diff2 diff3)

def sum_of_digits (k : ℕ) : ℕ :=
  if k < 10 then k else (k % 10) + sum_of_digits (k / 10)

theorem greatest_common_divisor_and_digit_sum :
  n = 1120 ∧ sum_of_digits n = 4 := by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_and_digit_sum_l2159_215993


namespace NUMINAMATH_CALUDE_brown_mice_count_l2159_215930

theorem brown_mice_count (total white brown : ℕ) : 
  total = white + brown →
  (2 : ℚ) / 3 * total = white →
  white = 14 →
  brown = 7 := by
sorry

end NUMINAMATH_CALUDE_brown_mice_count_l2159_215930


namespace NUMINAMATH_CALUDE_temperature_conversion_l2159_215940

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 75 → k = 167 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2159_215940


namespace NUMINAMATH_CALUDE_floor_of_4_7_l2159_215979

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l2159_215979


namespace NUMINAMATH_CALUDE_prob_red_base_is_half_l2159_215978

-- Define the total number of bases
def total_bases : ℕ := 4

-- Define the number of red educational bases
def red_bases : ℕ := 2

-- Define the probability of choosing a red educational base
def prob_red_base : ℚ := red_bases / total_bases

-- Theorem statement
theorem prob_red_base_is_half : prob_red_base = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_base_is_half_l2159_215978


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l2159_215927

theorem divisibility_of_expression (n : ℕ) (h : Odd n) (h' : n > 0) :
  ∃ k : ℤ, n^4 - n^2 - n = n * k :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l2159_215927


namespace NUMINAMATH_CALUDE_inequality_proof_l2159_215969

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) / 3 ≤ Real.sqrt ((a^2 + b^2 + c^2) / 3) ∧
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≤ (a*b/c + b*c/a + c*a/b) / 3 ∧
  ((a + b + c) / 3 = Real.sqrt ((a^2 + b^2 + c^2) / 3) ∧
   Real.sqrt ((a^2 + b^2 + c^2) / 3) = (a*b/c + b*c/a + c*a/b) / 3) ↔ (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2159_215969


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l2159_215932

theorem no_solution_fractional_equation :
  ¬∃ (x : ℝ), (3 / x) + (6 / (x - 1)) - ((x + 5) / (x^2 - x)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l2159_215932


namespace NUMINAMATH_CALUDE_fraction_simplification_l2159_215910

theorem fraction_simplification (x y z : ℝ) :
  (16 * x^4 * z^4 - x^4 * y^16 - 64 * x^4 * y^2 * z^4 + 4 * x^4 * y^18 + 32 * x^2 * y * z^4 - 2 * x^2 * y^17 + 16 * y^2 * z^4 - y^18) /
  ((2 * x^2 * y - x^2 - y) * (8 * z^3 + 2 * y^8 * z + 4 * y^4 * z^2 + y^12) * (2 * z - y^4)) =
  -(2 * x^2 * y + x^2 + y) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2159_215910


namespace NUMINAMATH_CALUDE_sin_product_equals_two_fifths_l2159_215936

theorem sin_product_equals_two_fifths (α : Real) (h : Real.tan α = 2) :
  Real.sin α * Real.sin (π / 2 - α) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equals_two_fifths_l2159_215936


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2159_215961

theorem unique_solution_trigonometric_equation :
  ∃! x : Real, 0 < x ∧ x < 180 ∧
  Real.tan ((150 : Real) * degree - x * degree) = 
    (Real.sin ((150 : Real) * degree) - Real.sin (x * degree)) / 
    (Real.cos ((150 : Real) * degree) - Real.cos (x * degree)) ∧
  x = 105 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2159_215961
