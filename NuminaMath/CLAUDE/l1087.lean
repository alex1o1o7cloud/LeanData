import Mathlib

namespace NUMINAMATH_CALUDE_red_joker_probability_l1087_108778

/-- A modified deck of cards -/
structure ModifiedDeck :=
  (total_cards : ℕ)
  (standard_cards : ℕ)
  (red_jokers : ℕ)
  (black_jokers : ℕ)

/-- Definition of our specific modified deck -/
def our_deck : ModifiedDeck :=
  { total_cards := 54,
    standard_cards := 52,
    red_jokers := 1,
    black_jokers := 1 }

/-- The probability of drawing a specific card from a deck -/
def probability_of_draw (deck : ModifiedDeck) (specific_cards : ℕ) : ℚ :=
  specific_cards / deck.total_cards

theorem red_joker_probability :
  probability_of_draw our_deck our_deck.red_jokers = 1 / 54 := by
  sorry


end NUMINAMATH_CALUDE_red_joker_probability_l1087_108778


namespace NUMINAMATH_CALUDE_problem_solution_l1087_108715

theorem problem_solution (x : ℝ) (h : 9 - 16/x + 9/x^2 = 0) : 3/x = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1087_108715


namespace NUMINAMATH_CALUDE_erdos_binomial_prime_factors_l1087_108727

-- Define the number of distinct prime factors function
noncomputable def num_distinct_prime_factors (m : ℕ) : ℕ := sorry

-- State the theorem
theorem erdos_binomial_prime_factors :
  ∃ (c : ℝ), c > 1 ∧
  ∀ (n k : ℕ), n > 0 ∧ k > 0 →
  (n : ℝ) > c^k →
  num_distinct_prime_factors (Nat.choose n k) ≥ k :=
sorry

end NUMINAMATH_CALUDE_erdos_binomial_prime_factors_l1087_108727


namespace NUMINAMATH_CALUDE_sports_league_games_l1087_108786

theorem sports_league_games (total_teams : Nat) (divisions : Nat) (teams_per_division : Nat)
  (intra_division_games : Nat) (inter_division_games : Nat) :
  total_teams = divisions * teams_per_division →
  divisions = 3 →
  teams_per_division = 4 →
  intra_division_games = 3 →
  inter_division_games = 1 →
  (total_teams * (((teams_per_division - 1) * intra_division_games) +
    ((total_teams - teams_per_division) * inter_division_games))) / 2 = 102 := by
  sorry

end NUMINAMATH_CALUDE_sports_league_games_l1087_108786


namespace NUMINAMATH_CALUDE_fraction_difference_l1087_108796

theorem fraction_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x / y) :
  1 / x - 1 / y = -(1 / y^2) := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_l1087_108796


namespace NUMINAMATH_CALUDE_inequality_solution_l1087_108732

theorem inequality_solution (a x : ℝ) :
  (a * x) / (x - 1) < (a - 1) / (x - 1) ↔
  (a > 0 ∧ (a - 1) / a < x ∧ x < 1) ∨
  (a = 0 ∧ x < 1) ∨
  (a < 0 ∧ (x > (a - 1) / a ∨ x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1087_108732


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1087_108765

theorem division_remainder_problem (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 161)
  (h2 : divisor = 16)
  (h3 : quotient = 10)
  (h4 : dividend = divisor * quotient + (dividend % divisor)) :
  dividend % divisor = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1087_108765


namespace NUMINAMATH_CALUDE_cube_surface_area_l1087_108711

/-- The surface area of a cube with edge length 7 cm is 294 square centimeters. -/
theorem cube_surface_area : 
  ∀ (edge_length : ℝ), 
  edge_length = 7 → 
  6 * edge_length^2 = 294 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1087_108711


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1087_108709

theorem necessary_but_not_sufficient : 
  (∀ x y : ℝ, x > 3 ∧ y ≥ 3 → x^2 + y^2 ≥ 9) ∧ 
  (∃ x y : ℝ, x^2 + y^2 ≥ 9 ∧ ¬(x > 3 ∧ y ≥ 3)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1087_108709


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1087_108742

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (l m : Line) (α : Plane) : 
  perpendicular l α → perpendicular m α → parallel l m :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1087_108742


namespace NUMINAMATH_CALUDE_geometric_sequence_a10_l1087_108751

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a10 (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 6 = 2/3 → q = Real.sqrt 3 → a 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a10_l1087_108751


namespace NUMINAMATH_CALUDE_sara_movie_tickets_l1087_108788

-- Define the constants
def ticket_cost : ℚ := 10.62
def rental_cost : ℚ := 1.59
def purchase_cost : ℚ := 13.95
def total_spent : ℚ := 36.78

-- Define the theorem
theorem sara_movie_tickets :
  ∃ (n : ℕ), n * ticket_cost + rental_cost + purchase_cost = total_spent ∧ n = 2 :=
sorry

end NUMINAMATH_CALUDE_sara_movie_tickets_l1087_108788


namespace NUMINAMATH_CALUDE_odds_against_C_l1087_108768

-- Define the type for horses
inductive Horse : Type
  | A
  | B
  | C

-- Define the race with no ties
def Race := Horse → ℕ

-- Define the odds against winning for each horse
def oddsAgainst (h : Horse) : ℚ :=
  match h with
  | Horse.A => 5/2
  | Horse.B => 3/1
  | Horse.C => 15/13  -- This is what we want to prove

-- Define the probability of winning for a horse given its odds against
def probWinning (odds : ℚ) : ℚ := 1 / (1 + odds)

-- State the theorem
theorem odds_against_C (race : Race) :
  (oddsAgainst Horse.A = 5/2) →
  (oddsAgainst Horse.B = 3/1) →
  (probWinning (oddsAgainst Horse.A) + probWinning (oddsAgainst Horse.B) + probWinning (oddsAgainst Horse.C) = 1) →
  oddsAgainst Horse.C = 15/13 := by
  sorry

end NUMINAMATH_CALUDE_odds_against_C_l1087_108768


namespace NUMINAMATH_CALUDE_optionB_is_suitable_only_optionB_is_suitable_l1087_108764

/-- Represents a sampling experiment --/
structure SamplingExperiment where
  sampleSize : Nat
  populationSize : Nat
  numFactories : Nat
  numBoxes : Nat

/-- Criteria for lottery method suitability --/
def isLotteryMethodSuitable (exp : SamplingExperiment) : Prop :=
  exp.sampleSize < 20 ∧ 
  exp.populationSize < 100 ∧ 
  exp.numFactories = 1 ∧
  exp.numBoxes > 1

/-- The four options given in the problem --/
def optionA : SamplingExperiment := ⟨600, 3000, 1, 1⟩
def optionB : SamplingExperiment := ⟨6, 30, 1, 2⟩
def optionC : SamplingExperiment := ⟨6, 30, 2, 2⟩
def optionD : SamplingExperiment := ⟨10, 3000, 1, 1⟩

/-- Theorem stating that option B is suitable for the lottery method --/
theorem optionB_is_suitable : isLotteryMethodSuitable optionB := by
  sorry

/-- Theorem stating that option B is the only suitable option --/
theorem only_optionB_is_suitable : 
  isLotteryMethodSuitable optionB ∧ 
  ¬isLotteryMethodSuitable optionA ∧ 
  ¬isLotteryMethodSuitable optionC ∧ 
  ¬isLotteryMethodSuitable optionD := by
  sorry

end NUMINAMATH_CALUDE_optionB_is_suitable_only_optionB_is_suitable_l1087_108764


namespace NUMINAMATH_CALUDE_remainder_13_pow_2031_mod_100_l1087_108705

theorem remainder_13_pow_2031_mod_100 : 13^2031 % 100 = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_13_pow_2031_mod_100_l1087_108705


namespace NUMINAMATH_CALUDE_log_sum_equality_l1087_108730

theorem log_sum_equality : 
  Real.log 8 / Real.log 2 + 3 * (Real.log 4 / Real.log 2) + 
  4 * (Real.log 16 / Real.log 4) + 2 * (Real.log 32 / Real.log 8) = 61 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1087_108730


namespace NUMINAMATH_CALUDE_animal_count_l1087_108721

theorem animal_count (num_cats : ℕ) : 
  (1 : ℕ) +                   -- 1 dog
  num_cats +                  -- cats
  2 * num_cats +              -- rabbits (2 per cat)
  3 * (2 * num_cats) = 37 →   -- hares (3 per rabbit)
  num_cats = 4 := by
sorry

end NUMINAMATH_CALUDE_animal_count_l1087_108721


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l1087_108752

theorem point_in_second_quadrant (a : ℝ) : 
  (a - 3 < 0 ∧ a + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

#check point_in_second_quadrant

end NUMINAMATH_CALUDE_point_in_second_quadrant_l1087_108752


namespace NUMINAMATH_CALUDE_integer_between_sqrt_2n_and_sqrt_5n_l1087_108703

theorem integer_between_sqrt_2n_and_sqrt_5n (n : ℕ) (h : n ≥ 1) :
  ∃ k : ℤ, Real.sqrt (2 * n) < k ∧ k < Real.sqrt (5 * n) := by
  sorry

end NUMINAMATH_CALUDE_integer_between_sqrt_2n_and_sqrt_5n_l1087_108703


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1087_108748

theorem fraction_equals_zero (x : ℝ) : 
  (x - 2) / (1 - x) = 0 → x = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1087_108748


namespace NUMINAMATH_CALUDE_B_minus_A_equality_l1087_108779

def A : Set ℝ := {y | ∃ x, 1/3 ≤ x ∧ x ≤ 1 ∧ y = 1/x}
def B : Set ℝ := {y | ∃ x, -1 ≤ x ∧ x ≤ 2 ∧ y = x^2 - 1}

theorem B_minus_A_equality : 
  B \ A = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_B_minus_A_equality_l1087_108779


namespace NUMINAMATH_CALUDE_mothers_age_l1087_108710

theorem mothers_age (daughter_age mother_age : ℕ) 
  (h1 : 2 * daughter_age + mother_age = 70)
  (h2 : daughter_age + 2 * mother_age = 95) :
  mother_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_mothers_age_l1087_108710


namespace NUMINAMATH_CALUDE_transformed_roots_equation_l1087_108754

/-- Given that a, b, c, and d are the solutions of x^4 + 2x^3 - 5 = 0,
    prove that abc/d, abd/c, acd/b, and bcd/a are the solutions of the same equation. -/
theorem transformed_roots_equation (a b c d : ℂ) : 
  (a^4 + 2*a^3 - 5 = 0) ∧ 
  (b^4 + 2*b^3 - 5 = 0) ∧ 
  (c^4 + 2*c^3 - 5 = 0) ∧ 
  (d^4 + 2*d^3 - 5 = 0) →
  ((a*b*c/d)^4 + 2*(a*b*c/d)^3 - 5 = 0) ∧
  ((a*b*d/c)^4 + 2*(a*b*d/c)^3 - 5 = 0) ∧
  ((a*c*d/b)^4 + 2*(a*c*d/b)^3 - 5 = 0) ∧
  ((b*c*d/a)^4 + 2*(b*c*d/a)^3 - 5 = 0) := by
  sorry


end NUMINAMATH_CALUDE_transformed_roots_equation_l1087_108754


namespace NUMINAMATH_CALUDE_friends_team_assignment_l1087_108785

theorem friends_team_assignment (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 4) :
  k ^ n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_friends_team_assignment_l1087_108785


namespace NUMINAMATH_CALUDE_coin_value_difference_l1087_108719

/-- Represents the number of coins of each type -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value in cents for a given coin count -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes

/-- Represents the constraint that there are 2500 coins in total -/
def totalCoins (coins : CoinCount) : Prop :=
  coins.pennies + coins.nickels + coins.dimes = 2500

/-- Represents the constraint that there is at least one of each type of coin -/
def atLeastOne (coins : CoinCount) : Prop :=
  coins.pennies ≥ 1 ∧ coins.nickels ≥ 1 ∧ coins.dimes ≥ 1

theorem coin_value_difference :
  ∃ (maxCoins minCoins : CoinCount),
    totalCoins maxCoins ∧
    totalCoins minCoins ∧
    atLeastOne maxCoins ∧
    atLeastOne minCoins ∧
    (∀ (coins : CoinCount), totalCoins coins → atLeastOne coins →
      totalValue coins ≤ totalValue maxCoins) ∧
    (∀ (coins : CoinCount), totalCoins coins → atLeastOne coins →
      totalValue coins ≥ totalValue minCoins) ∧
    totalValue maxCoins - totalValue minCoins = 22473 :=
by sorry

end NUMINAMATH_CALUDE_coin_value_difference_l1087_108719


namespace NUMINAMATH_CALUDE_quilt_cost_theorem_l1087_108755

def quilt_width : ℕ := 16
def quilt_length : ℕ := 20
def patch_area : ℕ := 4
def initial_patch_cost : ℕ := 10
def initial_patch_count : ℕ := 10

def total_quilt_area : ℕ := quilt_width * quilt_length
def total_patches : ℕ := total_quilt_area / patch_area
def discounted_patch_cost : ℕ := initial_patch_cost / 2
def discounted_patches : ℕ := total_patches - initial_patch_count

def total_cost : ℕ := initial_patch_count * initial_patch_cost + discounted_patches * discounted_patch_cost

theorem quilt_cost_theorem : total_cost = 450 := by
  sorry

end NUMINAMATH_CALUDE_quilt_cost_theorem_l1087_108755


namespace NUMINAMATH_CALUDE_nested_expression_value_l1087_108736

theorem nested_expression_value : (2*(2*(2*(2*(2*(2+1)+1)+1)+1)+1)+1) = 127 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l1087_108736


namespace NUMINAMATH_CALUDE_constant_fraction_iff_proportional_coefficients_l1087_108769

/-- A fraction of quadratic polynomials is constant if and only if the coefficients are proportional -/
theorem constant_fraction_iff_proportional_coefficients 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (h : a₂ ≠ 0) :
  (∃ k : ℝ, ∀ x : ℝ, (a₁ * x^2 + b₁ * x + c₁) / (a₂ * x^2 + b₂ * x + c₂) = k) ↔ 
  (∃ k : ℝ, a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ = k * c₂) :=
sorry

end NUMINAMATH_CALUDE_constant_fraction_iff_proportional_coefficients_l1087_108769


namespace NUMINAMATH_CALUDE_jason_blue_marbles_count_l1087_108761

/-- The number of blue marbles Jason and Tom have in total -/
def total_blue_marbles : ℕ := 68

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := 24

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := total_blue_marbles - tom_blue_marbles

theorem jason_blue_marbles_count : jason_blue_marbles = 44 := by
  sorry

end NUMINAMATH_CALUDE_jason_blue_marbles_count_l1087_108761


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1087_108740

/-- Calculates the number of groups in systematic sampling -/
def systematicSamplingGroups (populationSize sampleSize : ℕ) : ℕ :=
  if sampleSize > 0 then sampleSize else 0

/-- Theorem: For a population of 56 and sample size of 8, 
    systematic sampling produces 8 groups -/
theorem systematic_sampling_theorem :
  systematicSamplingGroups 56 8 = 8 := by
  sorry

#eval systematicSamplingGroups 56 8

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1087_108740


namespace NUMINAMATH_CALUDE_unique_permutation_with_difference_one_l1087_108720

theorem unique_permutation_with_difference_one (n : ℕ+) :
  ∃! (x : Fin (2 * n) → Fin (2 * n)), 
    Function.Bijective x ∧ 
    (∀ i : Fin (2 * n), |x i - i.val| = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_permutation_with_difference_one_l1087_108720


namespace NUMINAMATH_CALUDE_xiaohuas_apples_l1087_108756

theorem xiaohuas_apples :
  ∃ (x : ℕ), 
    x > 0 ∧ 
    (0 < 4 * x + 20 - 8 * (x - 1)) ∧ 
    (4 * x + 20 - 8 * (x - 1) < 8) ∧
    (4 * x + 20 = 44) := by
  sorry

end NUMINAMATH_CALUDE_xiaohuas_apples_l1087_108756


namespace NUMINAMATH_CALUDE_fraction_denominator_l1087_108784

theorem fraction_denominator (n : ℕ) (d : ℕ) :
  n = 35 →
  (n : ℚ) / d = 2 / 10^20 →
  d = 175 * 10^20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_l1087_108784


namespace NUMINAMATH_CALUDE_P_plus_8_divisible_P_minus_8_divisible_P_unique_l1087_108780

/-- A fifth-degree polynomial P(x) that satisfies specific divisibility conditions -/
def P (x : ℝ) : ℝ := 3*x^5 - 10*x^3 + 15*x

/-- P(x) + 8 is divisible by (x+1)^3 -/
theorem P_plus_8_divisible (x : ℝ) : ∃ (q : ℝ → ℝ), P x + 8 = (x + 1)^3 * q x := by sorry

/-- P(x) - 8 is divisible by (x-1)^3 -/
theorem P_minus_8_divisible (x : ℝ) : ∃ (r : ℝ → ℝ), P x - 8 = (x - 1)^3 * r x := by sorry

/-- P(x) is the unique fifth-degree polynomial satisfying both divisibility conditions -/
theorem P_unique : ∀ (Q : ℝ → ℝ), 
  (∃ (q r : ℝ → ℝ), (∀ x, Q x + 8 = (x + 1)^3 * q x) ∧ (∀ x, Q x - 8 = (x - 1)^3 * r x)) →
  (∃ (a b c d e f : ℝ), ∀ x, Q x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (∀ x, Q x = P x) := by sorry

end NUMINAMATH_CALUDE_P_plus_8_divisible_P_minus_8_divisible_P_unique_l1087_108780


namespace NUMINAMATH_CALUDE_ratio_problem_l1087_108749

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 4) : 
  (a + 2*b) / (b + 2*c) = 7/27 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1087_108749


namespace NUMINAMATH_CALUDE_range_of_a_l1087_108737

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + 2*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), ¬ Monotone (g a))
  ∧ (∀ x ∈ Set.Icc 1 (Real.exp 1), g a x ≤ g a (Real.exp 1))
  ∧ (∀ x ∈ Set.Icc 1 (Real.exp 1), x ≠ Real.exp 1 → g a x < g a (Real.exp 1))
  → 3 < a ∧ a < (Real.exp 1)^2 / 2 + 2 * Real.exp 1 - 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1087_108737


namespace NUMINAMATH_CALUDE_naclo4_formation_l1087_108741

-- Define the chemical reaction
structure Reaction where
  naoh : ℝ
  hclo4 : ℝ
  naclo4 : ℝ
  h2o : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.naoh = r.hclo4 ∧ r.naoh = r.naclo4 ∧ r.naoh = r.h2o

-- Define the initial conditions
def initial_conditions (initial_naoh initial_hclo4 : ℝ) (r : Reaction) : Prop :=
  initial_naoh = 3 ∧ initial_hclo4 = 3 ∧ r.naoh ≤ initial_naoh ∧ r.hclo4 ≤ initial_hclo4

-- Theorem statement
theorem naclo4_formation 
  (initial_naoh initial_hclo4 : ℝ) 
  (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : initial_conditions initial_naoh initial_hclo4 r) :
  r.naclo4 = min initial_naoh initial_hclo4 :=
sorry

end NUMINAMATH_CALUDE_naclo4_formation_l1087_108741


namespace NUMINAMATH_CALUDE_samantha_birth_year_l1087_108799

def mathLeagueYear (n : ℕ) : ℕ := 1995 + 2 * (n - 1)

theorem samantha_birth_year :
  (∀ n : ℕ, mathLeagueYear n = 1995 + 2 * (n - 1)) →
  mathLeagueYear 5 - 13 = 1990 :=
by sorry

end NUMINAMATH_CALUDE_samantha_birth_year_l1087_108799


namespace NUMINAMATH_CALUDE_savings_calculation_l1087_108773

theorem savings_calculation (income expenditure savings : ℕ) : 
  (income * 3 = expenditure * 5) →  -- Income and expenditure ratio is 5:3
  (income = 10000) →                -- Income is Rs. 10000
  (savings = income - expenditure) →  -- Definition of savings
  (savings = 4000) :=                -- Prove that savings are Rs. 4000
by
  sorry

#check savings_calculation

end NUMINAMATH_CALUDE_savings_calculation_l1087_108773


namespace NUMINAMATH_CALUDE_square_and_sqrt_identities_l1087_108745

theorem square_and_sqrt_identities :
  (1001 : ℕ)^2 = 1002001 ∧
  (1001001 : ℕ)^2 = 1002003002001 ∧
  (1002003004005004003002001 : ℕ).sqrt = 1001001001001 := by
  sorry

end NUMINAMATH_CALUDE_square_and_sqrt_identities_l1087_108745


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1087_108758

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 6) 
  (eq2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 98.08 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1087_108758


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1087_108774

/-- The function f(x) = x³ - 3x² + x -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 1

theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m*x + b ↔ 2*x + y - 1 = 0) ∧
    (m = f' 1) ∧
    (f 1 = m*1 + b) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1087_108774


namespace NUMINAMATH_CALUDE_white_ball_count_l1087_108789

/-- Given a bag of 100 glass balls with red, black, and white colors,
    prove that if the frequency of drawing red balls is 15% and black balls is 40%,
    then the number of white balls is 45. -/
theorem white_ball_count (total : ℕ) (red_freq black_freq : ℚ) :
  total = 100 →
  red_freq = 15 / 100 →
  black_freq = 40 / 100 →
  ∃ (white_count : ℕ), white_count = 45 ∧ white_count = total * (1 - red_freq - black_freq) :=
sorry

end NUMINAMATH_CALUDE_white_ball_count_l1087_108789


namespace NUMINAMATH_CALUDE_first_hour_distance_car_distance_problem_l1087_108735

/-- Given a car with increasing speed, calculate the distance traveled in the first hour -/
theorem first_hour_distance (speed_increase : ℕ → ℕ) (total_distance : ℕ) : ℕ :=
  let first_hour_dist : ℕ := 55
  have speed_increase_def : ∀ n : ℕ, speed_increase n = 2 * n := by sorry
  have total_distance_def : total_distance = 792 := by sorry
  have sum_formula : total_distance = (12 : ℕ) * first_hour_dist + 11 * 12 := by sorry
  first_hour_dist

/-- The main theorem stating the distance traveled in the first hour -/
theorem car_distance_problem : first_hour_distance (λ n => 2 * n) 792 = 55 := by sorry

end NUMINAMATH_CALUDE_first_hour_distance_car_distance_problem_l1087_108735


namespace NUMINAMATH_CALUDE_prob_AC_less_than_8_l1087_108700

/-- The probability that AC < 8 cm given the conditions of the problem -/
def probability_AC_less_than_8 : ℝ := 0.46

/-- The length of AB in cm -/
def AB : ℝ := 10

/-- The length of BC in cm -/
def BC : ℝ := 6

/-- The angle ABC in radians -/
def angle_ABC : Set ℝ := Set.Ioo 0 (Real.pi / 2)

/-- The theorem stating the probability of AC < 8 cm -/
theorem prob_AC_less_than_8 :
  ∃ (p : ℝ → Bool), p = λ β => ‖(0, -AB) - (BC * Real.cos β, BC * Real.sin β)‖ < 8 ∧
  ∫ β in angle_ABC, (if p β then 1 else 0) / Real.pi * 2 = probability_AC_less_than_8 :=
sorry

end NUMINAMATH_CALUDE_prob_AC_less_than_8_l1087_108700


namespace NUMINAMATH_CALUDE_t_range_max_radius_equation_l1087_108733

-- Define the circle equation
def circle_equation (x y t : ℝ) : Prop := x^2 + y^2 - 2*x + t^2 = 0

-- Theorem for the range of t
theorem t_range : ∀ x y t : ℝ, circle_equation x y t → -1 < t ∧ t < 1 := by sorry

-- Define the maximum radius
def max_radius (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Theorem for the circle equation when radius is maximum
theorem max_radius_equation : 
  (∃ t : ℝ, ∀ x y : ℝ, circle_equation x y t ∧ 
    (∀ t' : ℝ, circle_equation x y t' → 
      (x - 1)^2 + y^2 ≥ (x - 1)^2 + y^2)) → 
  ∀ x y : ℝ, max_radius x y := by sorry

end NUMINAMATH_CALUDE_t_range_max_radius_equation_l1087_108733


namespace NUMINAMATH_CALUDE_product_of_five_integers_l1087_108712

theorem product_of_five_integers (E F G H I : ℕ) 
  (sum_condition : E + F + G + H + I = 110)
  (equality_condition : (E : ℚ) / 2 = (F : ℚ) / 3 ∧ 
                        (F : ℚ) / 3 = G * 4 ∧ 
                        G * 4 = H * 2 ∧ 
                        H * 2 = I - 5) : 
  (E : ℚ) * F * G * H * I = 623400000 / 371293 := by
sorry

end NUMINAMATH_CALUDE_product_of_five_integers_l1087_108712


namespace NUMINAMATH_CALUDE_tape_overlap_division_l1087_108762

/-- Given 5 pieces of tape, each 2.7 meters long, with an overlap of 0.3 meters between pieces,
    when divided into 6 equal parts, each part is 2.05 meters long. -/
theorem tape_overlap_division (n : ℕ) (piece_length overlap_length : ℝ) (h1 : n = 5) 
    (h2 : piece_length = 2.7) (h3 : overlap_length = 0.3) : 
  (n * piece_length - (n - 1) * overlap_length) / 6 = 2.05 := by
  sorry

#check tape_overlap_division

end NUMINAMATH_CALUDE_tape_overlap_division_l1087_108762


namespace NUMINAMATH_CALUDE_fraction_simplification_l1087_108781

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) (h3 : x ≠ 0) :
  (3 * x / (x - 2) - x / (x + 2)) * ((x^2 - 4) / x) = 2 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1087_108781


namespace NUMINAMATH_CALUDE_ball_color_probability_l1087_108723

theorem ball_color_probability : 
  let n : ℕ := 8
  let p : ℝ := 1/2
  let num_arrangements : ℕ := n.choose (n/2)
  Fintype.card {s : Finset (Fin n) | s.card = n/2} / 2^n = 35/128 :=
by sorry

end NUMINAMATH_CALUDE_ball_color_probability_l1087_108723


namespace NUMINAMATH_CALUDE_woodworker_tables_l1087_108771

theorem woodworker_tables (total_legs : ℕ) (chairs : ℕ) (chair_legs : ℕ) (table_legs : ℕ) 
  (h1 : total_legs = 40)
  (h2 : chairs = 6)
  (h3 : chair_legs = 4)
  (h4 : table_legs = 4) :
  (total_legs - chairs * chair_legs) / table_legs = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_woodworker_tables_l1087_108771


namespace NUMINAMATH_CALUDE_three_sequence_non_decreasing_indices_l1087_108792

theorem three_sequence_non_decreasing_indices
  (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, p ≠ q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q := by
sorry

end NUMINAMATH_CALUDE_three_sequence_non_decreasing_indices_l1087_108792


namespace NUMINAMATH_CALUDE_invalid_vote_percentage_l1087_108739

/-- Proves that the percentage of invalid votes is 15% given the specified conditions --/
theorem invalid_vote_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_valid_votes : ℕ)
  (h_total : total_votes = 560000)
  (h_percentage : candidate_a_percentage = 75 / 100)
  (h_valid_votes : candidate_a_valid_votes = 357000) :
  (total_votes - (candidate_a_valid_votes / candidate_a_percentage : ℚ)) / total_votes = 15 / 100 :=
sorry

end NUMINAMATH_CALUDE_invalid_vote_percentage_l1087_108739


namespace NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l1087_108725

/-- Represents the state of the game at any given round -/
structure GameState where
  tokensA : ℕ
  tokensB : ℕ
  tokensC : ℕ

/-- Represents the rules of the game -/
def nextRound (state : GameState) : GameState :=
  match state with
  | ⟨a, b, c⟩ =>
    if a ≥ b ∧ a ≥ c then ⟨a - 3, b + 1, c + 1⟩
    else if b ≥ a ∧ b ≥ c then ⟨a + 1, b - 3, c + 1⟩
    else ⟨a + 1, b + 1, c - 3⟩

/-- Checks if the game has ended (i.e., if any player has run out of tokens) -/
def gameEnded (state : GameState) : Bool :=
  state.tokensA = 0 ∨ state.tokensB = 0 ∨ state.tokensC = 0

/-- Plays the game for a given number of rounds -/
def playGame (initialState : GameState) (rounds : ℕ) : GameState :=
  match rounds with
  | 0 => initialState
  | n + 1 => nextRound (playGame initialState n)

/-- The main theorem statement -/
theorem token_game_ends_in_37_rounds :
  let initialState := GameState.mk 15 14 13
  gameEnded (playGame initialState 37) ∧ ¬gameEnded (playGame initialState 36) := by
  sorry


end NUMINAMATH_CALUDE_token_game_ends_in_37_rounds_l1087_108725


namespace NUMINAMATH_CALUDE_num_triangles_in_dodecagon_l1087_108783

/-- A regular dodecagon has 12 vertices -/
def regular_dodecagon_vertices : ℕ := 12

/-- The number of triangles formed by choosing 3 vertices from a regular dodecagon -/
def num_triangles : ℕ := Nat.choose regular_dodecagon_vertices 3

/-- Theorem: The number of triangles formed by choosing 3 vertices from a regular dodecagon is 220 -/
theorem num_triangles_in_dodecagon : num_triangles = 220 := by sorry

end NUMINAMATH_CALUDE_num_triangles_in_dodecagon_l1087_108783


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1087_108772

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1087_108772


namespace NUMINAMATH_CALUDE_max_value_expression_l1087_108787

theorem max_value_expression (a b c d : ℕ) : 
  a ∈ ({1, 3, 5, 7} : Set ℕ) → 
  b ∈ ({1, 3, 5, 7} : Set ℕ) → 
  c ∈ ({1, 3, 5, 7} : Set ℕ) → 
  d ∈ ({1, 3, 5, 7} : Set ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (a + b) * (c + d) + (a + 1) * (d + 1) ≤ 112 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1087_108787


namespace NUMINAMATH_CALUDE_trajectory_of_right_angle_vertex_l1087_108791

/-- Given points M(-2,0) and N(2,0), prove that any point P(x,y) forming a right-angled triangle
    with MN as the hypotenuse satisfies the equation x^2 + y^2 = 4, where x ≠ ±2. -/
theorem trajectory_of_right_angle_vertex (x y : ℝ) :
  x ≠ -2 → x ≠ 2 →
  (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16 →
  x^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_right_angle_vertex_l1087_108791


namespace NUMINAMATH_CALUDE_missing_digits_sum_l1087_108777

/-- Given an addition problem 7□8 + 2182 = 863□91 where □ represents a single digit (0-9),
    the sum of the two missing digits is 7. -/
theorem missing_digits_sum (d1 d2 : Nat) : 
  d1 ≤ 9 → d2 ≤ 9 → 
  708 + d1 * 10 + 2182 = 86300 + d2 * 10 + 91 →
  d1 + d2 = 7 := by
sorry

end NUMINAMATH_CALUDE_missing_digits_sum_l1087_108777


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l1087_108793

/-- Calculates the gain percent when an item is bought and sold at given prices. -/
def gainPercent (costPrice sellingPrice : ℚ) : ℚ :=
  ((sellingPrice - costPrice) / costPrice) * 100

/-- Theorem: The gain percent is 50% when a cycle is bought for Rs. 900 and sold for Rs. 1350. -/
theorem cycle_gain_percent :
  let costPrice : ℚ := 900
  let sellingPrice : ℚ := 1350
  gainPercent costPrice sellingPrice = 50 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l1087_108793


namespace NUMINAMATH_CALUDE_unread_fraction_of_book_l1087_108724

theorem unread_fraction_of_book (total : ℝ) (read : ℝ) : 
  total > 0 → read > total / 2 → read < total → (total - read) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_unread_fraction_of_book_l1087_108724


namespace NUMINAMATH_CALUDE_divisible_by_nine_l1087_108713

theorem divisible_by_nine (A : Nat) : A < 10 → (7000 + 100 * A + 46) % 9 = 0 ↔ A = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l1087_108713


namespace NUMINAMATH_CALUDE_scale_heights_theorem_l1087_108728

theorem scale_heights_theorem (n : ℕ) (adults children : Fin n → ℝ) 
  (h : ∀ i : Fin n, adults i > children i) :
  ∃ (scales : Fin n → ℕ+), 
    (∀ i j : Fin n, (scales i : ℝ) * adults i > (scales j : ℝ) * children j) := by
  sorry

end NUMINAMATH_CALUDE_scale_heights_theorem_l1087_108728


namespace NUMINAMATH_CALUDE_max_value_of_f_l1087_108704

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 12*x + 16

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-2) 3 ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-2) 3 → f y ≤ f x) ∧
  f x = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1087_108704


namespace NUMINAMATH_CALUDE_area_of_union_S_l1087_108738

/-- A disc D in the 2D plane -/
structure Disc where
  center : ℝ × ℝ
  radius : ℝ

/-- The set S of discs D -/
def S : Set Disc :=
  {D : Disc | D.center.2 = D.center.1^2 - 3/4 ∧ 
              ∀ (x y : ℝ), (x - D.center.1)^2 + (y - D.center.2)^2 < D.radius^2 → y < 0}

/-- The area of the union of all discs in S -/
def unionArea (S : Set Disc) : ℝ := sorry

/-- Theorem stating the area of the union of discs in S -/
theorem area_of_union_S : unionArea S = (2 * Real.pi / 3) + (Real.sqrt 3 / 4) := by sorry

end NUMINAMATH_CALUDE_area_of_union_S_l1087_108738


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_plus_self_l1087_108750

theorem complex_magnitude_squared_plus_self (z : ℂ) (h : z = 1 + I) :
  Complex.abs (z^2 + z) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_plus_self_l1087_108750


namespace NUMINAMATH_CALUDE_john_arcade_spending_l1087_108797

/-- The fraction of John's allowance spent at the arcade -/
def arcade_fraction (allowance arcade_spent : ℚ) : ℚ :=
  arcade_spent / allowance

/-- The amount remaining after spending at the arcade and toy store -/
def remaining_after_toy_store (allowance arcade_spent : ℚ) : ℚ :=
  allowance - arcade_spent - (1/3) * (allowance - arcade_spent)

theorem john_arcade_spending :
  ∃ (arcade_spent : ℚ),
    arcade_fraction 3.30 arcade_spent = 3/5 ∧
    remaining_after_toy_store 3.30 arcade_spent = 0.88 := by
  sorry

end NUMINAMATH_CALUDE_john_arcade_spending_l1087_108797


namespace NUMINAMATH_CALUDE_roxanne_change_l1087_108731

/-- Represents the purchase and payment scenario for Roxanne --/
structure Purchase where
  lemonade_count : ℕ
  lemonade_price : ℚ
  sandwich_count : ℕ
  sandwich_price : ℚ
  paid_amount : ℚ

/-- Calculates the change Roxanne should receive --/
def calculate_change (p : Purchase) : ℚ :=
  p.paid_amount - (p.lemonade_count * p.lemonade_price + p.sandwich_count * p.sandwich_price)

/-- Theorem stating that Roxanne's change should be $11 --/
theorem roxanne_change :
  let p : Purchase := {
    lemonade_count := 2,
    lemonade_price := 2,
    sandwich_count := 2,
    sandwich_price := 2.5,
    paid_amount := 20
  }
  calculate_change p = 11 := by sorry

end NUMINAMATH_CALUDE_roxanne_change_l1087_108731


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1087_108782

theorem smallest_integer_with_remainders : ∃ n : ℕ,
  n > 0 ∧
  n % 10 = 9 ∧
  n % 9 = 8 ∧
  n % 8 = 7 ∧
  n % 7 = 6 ∧
  n % 6 = 5 ∧
  n % 5 = 4 ∧
  n % 4 = 3 ∧
  n % 3 = 2 ∧
  n % 2 = 1 ∧
  (∀ m : ℕ, m > 0 →
    m % 10 = 9 →
    m % 9 = 8 →
    m % 8 = 7 →
    m % 7 = 6 →
    m % 6 = 5 →
    m % 5 = 4 →
    m % 4 = 3 →
    m % 3 = 2 →
    m % 2 = 1 →
    n ≤ m) ∧
  n = 2519 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1087_108782


namespace NUMINAMATH_CALUDE_place_value_ratio_l1087_108716

def number : ℚ := 86572.4908

theorem place_value_ratio : 
  ∃ (tens hundredths : ℚ), 
    (tens = 10) ∧ 
    (hundredths = 0.01) ∧ 
    (tens / hundredths = 1000) :=
by sorry

end NUMINAMATH_CALUDE_place_value_ratio_l1087_108716


namespace NUMINAMATH_CALUDE_k_range_l1087_108794

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := (3 : ℝ) / (x + 1) < 1

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (p q : ℝ → Prop) : Prop :=
  (∀ x, q x → p x) ∧ ∃ x, p x ∧ ¬q x

-- Theorem statement
theorem k_range (k : ℝ) : 
  necessary_but_not_sufficient (p k) q ↔ k > 2 :=
sorry

end NUMINAMATH_CALUDE_k_range_l1087_108794


namespace NUMINAMATH_CALUDE_abc_inequality_l1087_108795

noncomputable def a : ℝ := 3^(1/5)
noncomputable def b : ℝ := (1/5)^3
noncomputable def c : ℝ := Real.log 3 / Real.log (1/5)

theorem abc_inequality : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l1087_108795


namespace NUMINAMATH_CALUDE_rectangle_length_l1087_108790

/-- Given a rectangle with width 4 inches and area 8 square inches, prove its length is 2 inches. -/
theorem rectangle_length (width : ℝ) (area : ℝ) (h1 : width = 4) (h2 : area = 8) :
  area / width = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l1087_108790


namespace NUMINAMATH_CALUDE_syrup_problem_l1087_108706

/-- Represents a container with a certain volume of liquid --/
structure Container where
  syrup : ℝ
  water : ℝ

/-- The state of the three containers --/
structure ContainerState where
  a : Container
  b : Container
  c : Container

/-- Represents a pouring action --/
inductive PourAction
  | PourAll : Fin 3 → Fin 3 → PourAction
  | Equalize : Fin 3 → Fin 3 → PourAction
  | PourToSink : Fin 3 → PourAction

/-- Defines if a given sequence of actions is valid --/
def isValidActionSequence (initialState : ContainerState) (actions : List PourAction) : Prop :=
  sorry

/-- Defines if a final state has 10L of 30% syrup in one container --/
def hasTenLitersThirtyPercentSyrup (state : ContainerState) : Prop :=
  sorry

/-- The main theorem to prove --/
theorem syrup_problem (n : ℕ) :
  (∃ (actions : List PourAction),
    isValidActionSequence
      ⟨⟨3, 0⟩, ⟨0, n⟩, ⟨0, 0⟩⟩
      actions ∧
    hasTenLitersThirtyPercentSyrup
      (actions.foldl (λ state action => sorry) ⟨⟨3, 0⟩, ⟨0, n⟩, ⟨0, 0⟩⟩)) ↔
  ∃ (k : ℕ), n = 3 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_syrup_problem_l1087_108706


namespace NUMINAMATH_CALUDE_johns_income_l1087_108701

theorem johns_income (john_tax_rate ingrid_tax_rate combined_tax_rate : ℚ)
  (ingrid_income : ℕ) :
  john_tax_rate = 30 / 100 →
  ingrid_tax_rate = 40 / 100 →
  combined_tax_rate = 35625 / 100000 →
  ingrid_income = 72000 →
  ∃ john_income : ℕ,
    john_income = 56000 ∧
    (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) /
      (john_income + ingrid_income) = combined_tax_rate :=
by sorry

end NUMINAMATH_CALUDE_johns_income_l1087_108701


namespace NUMINAMATH_CALUDE_sally_boxes_proof_l1087_108718

/-- The number of boxes Sally sold on Saturday -/
def saturday_boxes : ℕ := 65

/-- The number of boxes Sally sold on Sunday -/
def sunday_boxes : ℕ := (3 * saturday_boxes) / 2

/-- The number of boxes Sally sold on Monday -/
def monday_boxes : ℕ := (13 * sunday_boxes) / 10

theorem sally_boxes_proof :
  saturday_boxes + sunday_boxes + monday_boxes = 290 :=
sorry

end NUMINAMATH_CALUDE_sally_boxes_proof_l1087_108718


namespace NUMINAMATH_CALUDE_num_triangles_eq_choose_l1087_108775

/-- The number of triangles formed by n lines in general position on a plane -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose n 3

/-- 
Theorem: The number of triangles formed by n lines in general position on a plane
is equal to (n choose 3).
-/
theorem num_triangles_eq_choose (n : ℕ) : 
  num_triangles n = Nat.choose n 3 := by
  sorry

end NUMINAMATH_CALUDE_num_triangles_eq_choose_l1087_108775


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l1087_108708

theorem square_difference_divided_by_nine : (108^2 - 99^2) / 9 = 207 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l1087_108708


namespace NUMINAMATH_CALUDE_locus_of_A_is_ellipse_l1087_108747

/-- Given ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/9 + y^2/5 = 1

/-- Right focus of the ellipse -/
def F : ℝ × ℝ := (2, 0)

/-- Point on the ellipse -/
def point_on_ellipse (B : ℝ × ℝ) : Prop := ellipse B.1 B.2

/-- Equilateral triangle property -/
def is_equilateral (A B : ℝ × ℝ) : Prop :=
  let FA := (A.1 - F.1, A.2 - F.2)
  let FB := (B.1 - F.1, B.2 - F.2)
  let AB := (B.1 - A.1, B.2 - A.2)
  FA.1^2 + FA.2^2 = FB.1^2 + FB.2^2 ∧ FA.1^2 + FA.2^2 = AB.1^2 + AB.2^2

/-- Counterclockwise arrangement -/
def is_counterclockwise (A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.2 - F.2) - (A.2 - F.2) * (B.1 - F.1) > 0

/-- Locus of point A -/
def locus_A (A : ℝ × ℝ) : Prop :=
  ∃ (B : ℝ × ℝ), point_on_ellipse B ∧ is_equilateral A B ∧ is_counterclockwise A B

/-- Theorem statement -/
theorem locus_of_A_is_ellipse :
  ∀ (A : ℝ × ℝ), locus_A A ↔ 
    (A.1 - 2)^2 + A.2^2 + (A.1)^2 + (A.2 - 2*Real.sqrt 3)^2 = 36 :=
sorry

end NUMINAMATH_CALUDE_locus_of_A_is_ellipse_l1087_108747


namespace NUMINAMATH_CALUDE_range_of_x_l1087_108767

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the property of f being even
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of f being decreasing on [0,+∞)
def IsDecreasingOnNonnegativeReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

-- State the theorem
theorem range_of_x (h1 : IsEven f) (h2 : IsDecreasingOnNonnegativeReals f) 
  (h3 : ∀ x > 0, f (Real.log x / Real.log 10) > f 1) :
  ∀ x > 0, f (Real.log x / Real.log 10) > f 1 → 1/10 < x ∧ x < 10 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l1087_108767


namespace NUMINAMATH_CALUDE_largest_decimal_l1087_108776

theorem largest_decimal : 
  let a := 0.97
  let b := 0.979
  let c := 0.9709
  let d := 0.907
  let e := 0.9089
  b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_decimal_l1087_108776


namespace NUMINAMATH_CALUDE_monotonic_function_property_l1087_108729

/-- A monotonic function f: ℝ → ℝ satisfying f[f(x) - 3^x] = 4 for all x ∈ ℝ has f(2) = 10 -/
theorem monotonic_function_property (f : ℝ → ℝ) 
  (h_monotonic : Monotone f)
  (h_property : ∀ x : ℝ, f (f x - 3^x) = 4) :
  f 2 = 10 := by sorry

end NUMINAMATH_CALUDE_monotonic_function_property_l1087_108729


namespace NUMINAMATH_CALUDE_smallest_c_plus_d_l1087_108766

theorem smallest_c_plus_d : ∃ (c d : ℕ+), 
  (3^6 * 7^2 : ℕ) = c^(d:ℕ) ∧ 
  (∀ (c' d' : ℕ+), (3^6 * 7^2 : ℕ) = c'^(d':ℕ) → c + d ≤ c' + d') ∧
  c + d = 1325 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_plus_d_l1087_108766


namespace NUMINAMATH_CALUDE_payroll_after_layoffs_l1087_108744

/-- Represents the company's employee structure and payroll --/
structure Company where
  total_employees : Nat
  employees_2000 : Nat
  employees_2500 : Nat
  employees_3000 : Nat
  bonus_2000 : Nat
  health_benefit_2500 : Nat
  retirement_benefit_3000 : Nat

/-- Calculates the remaining employees after a layoff --/
def layoff (employees : Nat) (percentage : Nat) : Nat :=
  employees - (employees * percentage / 100)

/-- Applies the first round of layoffs and benefit changes --/
def first_round (c : Company) : Company :=
  { c with
    employees_2000 := layoff c.employees_2000 20,
    employees_2500 := layoff c.employees_2500 25,
    employees_3000 := layoff c.employees_3000 15,
    bonus_2000 := 400,
    health_benefit_2500 := 300 }

/-- Applies the second round of layoffs and benefit changes --/
def second_round (c : Company) : Company :=
  { c with
    employees_2000 := layoff c.employees_2000 10,
    employees_2500 := layoff c.employees_2500 15,
    employees_3000 := layoff c.employees_3000 5,
    retirement_benefit_3000 := 480 }

/-- Calculates the total payroll after both rounds of layoffs --/
def total_payroll (c : Company) : Nat :=
  c.employees_2000 * (2000 + c.bonus_2000) +
  c.employees_2500 * (2500 + c.health_benefit_2500) +
  c.employees_3000 * (3000 + c.retirement_benefit_3000)

/-- The initial company state --/
def initial_company : Company :=
  { total_employees := 450,
    employees_2000 := 150,
    employees_2500 := 200,
    employees_3000 := 100,
    bonus_2000 := 500,
    health_benefit_2500 := 400,
    retirement_benefit_3000 := 600 }

theorem payroll_after_layoffs :
  total_payroll (second_round (first_round initial_company)) = 893200 := by
  sorry

end NUMINAMATH_CALUDE_payroll_after_layoffs_l1087_108744


namespace NUMINAMATH_CALUDE_smallest_cut_length_for_non_triangle_l1087_108798

theorem smallest_cut_length_for_non_triangle : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (∀ (y : ℕ), y < x → (9 - y) + (16 - y) > (18 - y)) ∧
  ((9 - x) + (16 - x) ≤ (18 - x)) ∧ 
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cut_length_for_non_triangle_l1087_108798


namespace NUMINAMATH_CALUDE_symmetry_sum_for_17gon_l1087_108726

/-- The number of sides in our regular polygon -/
def n : ℕ := 17

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle of rotational symmetry (in degrees) for a regular n-gon -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 17-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is 649/17 -/
theorem symmetry_sum_for_17gon : L n + R n = 649 / 17 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_for_17gon_l1087_108726


namespace NUMINAMATH_CALUDE_growth_rate_correct_l1087_108770

/-- The average annual growth rate of vegetable production value from 2013 to 2015 -/
def average_growth_rate : ℝ := 0.25

/-- The initial production value in 2013 (in millions of yuan) -/
def initial_value : ℝ := 6.4

/-- The final production value in 2015 (in millions of yuan) -/
def final_value : ℝ := 10

/-- Theorem stating that the average annual growth rate correctly relates the initial and final values -/
theorem growth_rate_correct : initial_value * (1 + average_growth_rate)^2 = final_value := by
  sorry

end NUMINAMATH_CALUDE_growth_rate_correct_l1087_108770


namespace NUMINAMATH_CALUDE_frosting_theorem_l1087_108714

-- Define the frosting amounts for each type of baked good
def frosting_layer_cake : ℚ := 1
def frosting_single_cake : ℚ := 1/2
def frosting_pan_brownies : ℚ := 1/2
def frosting_dozen_cupcakes : ℚ := 1/2

-- Define the quantities of each baked good
def num_layer_cakes : ℕ := 3
def num_dozen_cupcakes : ℕ := 6
def num_single_cakes : ℕ := 12
def num_pans_brownies : ℕ := 18

-- Calculate total frosting needed
def total_frosting_needed : ℚ :=
  frosting_layer_cake * num_layer_cakes +
  frosting_dozen_cupcakes * num_dozen_cupcakes +
  frosting_single_cake * num_single_cakes +
  frosting_pan_brownies * num_pans_brownies

-- Theorem to prove
theorem frosting_theorem : total_frosting_needed = 21 := by
  sorry

end NUMINAMATH_CALUDE_frosting_theorem_l1087_108714


namespace NUMINAMATH_CALUDE_total_cost_for_nuggets_l1087_108707

-- Define the number of chicken nuggets ordered
def total_nuggets : ℕ := 100

-- Define the number of nuggets in a box
def nuggets_per_box : ℕ := 20

-- Define the cost of one box
def cost_per_box : ℕ := 4

-- Theorem to prove
theorem total_cost_for_nuggets : 
  (total_nuggets / nuggets_per_box) * cost_per_box = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_for_nuggets_l1087_108707


namespace NUMINAMATH_CALUDE_infinite_non_representable_numbers_l1087_108717

theorem infinite_non_representable_numbers : 
  ∃ S : Set ℕ, Set.Infinite S ∧ 
  ∀ k ∈ S, ∀ n : ℕ, ∀ p : ℕ, 
    Prime p → k ≠ n^2 + p := by
  sorry

end NUMINAMATH_CALUDE_infinite_non_representable_numbers_l1087_108717


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_less_than_one_l1087_108763

theorem inequality_solution_implies_a_less_than_one :
  ∀ a : ℝ, (∀ x : ℝ, (a - 1) * x > 2 ↔ x < 2 / (a - 1)) → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_less_than_one_l1087_108763


namespace NUMINAMATH_CALUDE_income_comparison_l1087_108734

theorem income_comparison (juan tim mart : ℝ) 
  (h1 : tim = juan * (1 - 0.4))
  (h2 : mart = tim * (1 + 0.4)) :
  mart = juan * 0.84 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l1087_108734


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l1087_108702

-- Define the polynomials
def f (x : ℝ) : ℝ := -2 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 4 * x^2 + 6 * x + 3
def j (x : ℝ) : ℝ := 3 * x^2 - x + 2

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x + j x = -x^2 + 11 * x - 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l1087_108702


namespace NUMINAMATH_CALUDE_no_solution_for_system_l1087_108743

theorem no_solution_for_system :
  ¬ ∃ (x y : ℝ), (2 * x - 3 * y = 6) ∧ (4 * x - 6 * y = 8) ∧ (5 * x - 5 * y = 15) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_system_l1087_108743


namespace NUMINAMATH_CALUDE_binomial_12_3_l1087_108746

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by sorry

end NUMINAMATH_CALUDE_binomial_12_3_l1087_108746


namespace NUMINAMATH_CALUDE_no_primes_of_form_l1087_108757

theorem no_primes_of_form (m : ℕ) (hm : m > 0) : 
  ¬ Prime (2^(5*m) + 2^m + 1) := by
sorry

end NUMINAMATH_CALUDE_no_primes_of_form_l1087_108757


namespace NUMINAMATH_CALUDE_station_length_l1087_108722

/-- The length of a station given a train passing through it -/
theorem station_length (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 250 →
  train_speed_kmh = 36 →
  passing_time = 45 →
  (train_speed_kmh * 1000 / 3600) * passing_time - train_length = 200 :=
by sorry

end NUMINAMATH_CALUDE_station_length_l1087_108722


namespace NUMINAMATH_CALUDE_virus_length_scientific_notation_l1087_108753

/-- Represents the scientific notation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem virus_length_scientific_notation :
  toScientificNotation 0.00000032 = ScientificNotation.mk 3.2 (-7) :=
sorry

end NUMINAMATH_CALUDE_virus_length_scientific_notation_l1087_108753


namespace NUMINAMATH_CALUDE_vector_equality_exists_l1087_108760

theorem vector_equality_exists (a b : ℝ × ℝ) :
  let a : ℝ × ℝ := (1, Real.sqrt 3)
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ π ∧
    let b : ℝ × ℝ := (Real.cos θ, Real.sin θ)
    ‖a + b‖ = ‖a - b‖ :=
by sorry

end NUMINAMATH_CALUDE_vector_equality_exists_l1087_108760


namespace NUMINAMATH_CALUDE_expansion_properties_l1087_108759

-- Define the binomial expansion function
def binomial_expansion (x : ℝ) (n : ℕ) : ℝ → ℝ := sorry

-- Define the coefficient function for the expansion
def coefficient (x : ℝ) (n r : ℕ) : ℝ := sorry

-- Define the general term of the expansion
def general_term (x : ℝ) (n r : ℕ) : ℝ := sorry

theorem expansion_properties :
  let f := binomial_expansion x 8
  -- The first three coefficients are in arithmetic sequence
  ∃ (a d : ℝ), coefficient x 8 0 = a ∧ 
               coefficient x 8 1 = a + d ∧ 
               coefficient x 8 2 = a + 2*d →
  -- 1. The term containing x to the first power
  (∃ (r : ℕ), general_term x 8 r = (35/8) * x) ∧
  -- 2. The rational terms involving x
  (∀ (r : ℕ), r ≤ 8 → 
    (∃ (k : ℤ), general_term x 8 r = x^k) ↔ 
    (general_term x 8 r = x^4 ∨ 
     general_term x 8 r = (35/8) * x ∨ 
     general_term x 8 r = 1/(256 * x^2))) ∧
  -- 3. The terms with the largest coefficient
  (∀ (r : ℕ), r ≤ 8 → 
    coefficient x 8 r ≤ 7 ∧
    (coefficient x 8 r = 7 ↔ (r = 2 ∨ r = 3))) :=
sorry

end NUMINAMATH_CALUDE_expansion_properties_l1087_108759
