import Mathlib

namespace NUMINAMATH_GPT_true_propositions_l520_52083

open Set

theorem true_propositions (M N : Set ℕ) (a b m : ℕ) (h1 : M ⊆ N) 
  (h2 : a > b) (h3 : b > 0) (h4 : m > 0) (p : ∀ x : ℝ, x > 0) :
  (M ⊆ M ∪ N) ∧ ((b + m) / (a + m) > b / a) ∧ 
  ¬(∀ (a b c : ℝ), a = b ↔ a * c ^ 2 = b * c ^ 2) ∧ 
  ¬(∃ x₀ : ℝ, x₀ ≤ 0) := sorry

end NUMINAMATH_GPT_true_propositions_l520_52083


namespace NUMINAMATH_GPT_negation_of_existential_l520_52036

theorem negation_of_existential :
  (¬ ∃ (x : ℝ), x^2 + x + 1 < 0) ↔ (∀ (x : ℝ), x^2 + x + 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_l520_52036


namespace NUMINAMATH_GPT_total_pears_picked_l520_52039

theorem total_pears_picked :
  let mike_pears := 8
  let jason_pears := 7
  let fred_apples := 6
  -- The total number of pears picked is the sum of Mike's and Jason's pears.
  mike_pears + jason_pears = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_pears_picked_l520_52039


namespace NUMINAMATH_GPT_simplify_expression_l520_52002

variable (a b : ℝ)

theorem simplify_expression : (a + b) * (3 * a - b) - b * (a - b) = 3 * a ^ 2 + a * b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l520_52002


namespace NUMINAMATH_GPT_candy_per_packet_l520_52067

-- Define the conditions as hypotheses
def bobby_weekly_candies (mon_to_fri_candies : ℕ) (sat_sun_candies : ℕ) : ℕ :=
  mon_to_fri_candies + sat_sun_candies

def total_candies_in_n_weeks (weekly_candies : ℕ) (n : ℕ) : ℕ :=
  weekly_candies * n

theorem candy_per_packet
  (mon_to_fri_candies_per_day : ℕ)
  (sat_sun_candies_per_day : ℕ)
  (days_mon_to_fri : ℕ)
  (days_weekend : ℕ)
  (num_weeks : ℕ)
  (total_packets : ℕ)
  (candies_per_packet : ℕ)
  (h1 : mon_to_fri_candies_per_day = 2)
  (h2 : sat_sun_candies_per_day = 1)
  (h3 : days_mon_to_fri = 5)
  (h4 : days_weekend = 2)
  (h5 : num_weeks = 3)
  (h6 : total_packets = 2)
  (h7 : candies_per_packet = (total_candies_in_n_weeks (bobby_weekly_candies (mon_to_fri_candies_per_day * days_mon_to_fri) (sat_sun_candies_per_day * days_weekend)) num_weeks) / total_packets) :
  candies_per_packet = 18 :=
sorry

end NUMINAMATH_GPT_candy_per_packet_l520_52067


namespace NUMINAMATH_GPT_apples_left_l520_52050

-- Define the initial number of apples and the conditions
def initial_apples := 150
def percent_sold_to_jill := 20 / 100
def percent_sold_to_june := 30 / 100
def apples_given_to_teacher := 2

-- Formulate the problem statement in Lean
theorem apples_left (initial_apples percent_sold_to_jill percent_sold_to_june apples_given_to_teacher : ℕ) :
  let sold_to_jill := percent_sold_to_jill * initial_apples
  let remaining_after_jill := initial_apples - sold_to_jill
  let sold_to_june := percent_sold_to_june * remaining_after_jill
  let remaining_after_june := remaining_after_jill - sold_to_june
  let final_apples := remaining_after_june - apples_given_to_teacher
  final_apples = 82 := 
by 
  sorry

end NUMINAMATH_GPT_apples_left_l520_52050


namespace NUMINAMATH_GPT_irreducible_fraction_l520_52098

theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end NUMINAMATH_GPT_irreducible_fraction_l520_52098


namespace NUMINAMATH_GPT_mean_of_two_remaining_numbers_l520_52074

theorem mean_of_two_remaining_numbers (a b c: ℝ) (h1: (a + b + c + 100) / 4 = 90) (h2: a = 70) : (b + c) / 2 = 95 := by
  sorry

end NUMINAMATH_GPT_mean_of_two_remaining_numbers_l520_52074


namespace NUMINAMATH_GPT_total_votes_l520_52060

theorem total_votes (T F A : ℝ)
  (h1 : F = A + 68)
  (h2 : A = 0.40 * T)
  (h3 : T = F + A) :
  T = 340 :=
by sorry

end NUMINAMATH_GPT_total_votes_l520_52060


namespace NUMINAMATH_GPT_total_points_scored_l520_52018

-- Define the points scored by Sam and his friend
def points_scored_by_sam : ℕ := 75
def points_scored_by_friend : ℕ := 12

-- The main theorem stating the total points
theorem total_points_scored : points_scored_by_sam + points_scored_by_friend = 87 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_points_scored_l520_52018


namespace NUMINAMATH_GPT_infinite_primes_congruent_3_mod_4_infinite_primes_congruent_5_mod_6_l520_52001

-- Problem 1: Infinitely many primes congruent to 3 modulo 4
theorem infinite_primes_congruent_3_mod_4 :
  ∀ (ps : Finset ℕ), (∀ p ∈ ps, Nat.Prime p ∧ p % 4 = 3) → ∃ q, Nat.Prime q ∧ q % 4 = 3 ∧ q ∉ ps :=
by
  sorry

-- Problem 2: Infinitely many primes congruent to 5 modulo 6
theorem infinite_primes_congruent_5_mod_6 :
  ∀ (ps : Finset ℕ), (∀ p ∈ ps, Nat.Prime p ∧ p % 6 = 5) → ∃ q, Nat.Prime q ∧ q % 6 = 5 ∧ q ∉ ps :=
by
  sorry

end NUMINAMATH_GPT_infinite_primes_congruent_3_mod_4_infinite_primes_congruent_5_mod_6_l520_52001


namespace NUMINAMATH_GPT_angle_sum_proof_l520_52066

theorem angle_sum_proof (x y : ℝ) (h : 3 * x + 6 * x + (x + y) + 4 * y = 360) : x = 0 ∧ y = 72 :=
by {
  sorry
}

end NUMINAMATH_GPT_angle_sum_proof_l520_52066


namespace NUMINAMATH_GPT_students_count_l520_52016

theorem students_count (S : ℕ) (num_adults : ℕ) (cost_student cost_adult total_cost : ℕ)
  (h1 : num_adults = 4)
  (h2 : cost_student = 5)
  (h3 : cost_adult = 6)
  (h4 : total_cost = 199) :
  5 * S + 4 * 6 = 199 → S = 35 := by
  sorry

end NUMINAMATH_GPT_students_count_l520_52016


namespace NUMINAMATH_GPT_brendan_match_ratio_l520_52091

noncomputable def brendanMatches (totalMatches firstRound secondRound matchesWonFirstTwoRounds matchesWonTotal matchesInLastRound : ℕ) :=
  matchesWonFirstTwoRounds = firstRound + secondRound ∧
  matchesWonFirstTwoRounds = 12 ∧
  totalMatches = matchesWonTotal ∧
  matchesWonTotal = 14 ∧
  firstRound = 6 ∧
  secondRound = 6 ∧
  matchesInLastRound = 4

theorem brendan_match_ratio :
  ∃ ratio: ℕ × ℕ,
    let firstRound := 6
    let secondRound := 6
    let matchesInLastRound := 4
    let matchesWonFirstTwoRounds := firstRound + secondRound
    let matchesWonTotal := 14
    let matchesWonLastRound := matchesWonTotal - matchesWonFirstTwoRounds
    let ratio := (matchesWonLastRound, matchesInLastRound)
    brendanMatches matchesWonTotal firstRound secondRound matchesWonFirstTwoRounds matchesWonTotal matchesInLastRound ∧
    ratio = (1, 2) :=
by
  sorry

end NUMINAMATH_GPT_brendan_match_ratio_l520_52091


namespace NUMINAMATH_GPT_find_cos_minus_sin_l520_52007

-- Definitions from the conditions
variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π)  -- Second quadrant
variable (h2 : Real.sin (2 * α) = -24 / 25)  -- Given sin 2α

-- Lean statement of the problem
theorem find_cos_minus_sin (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : Real.sin (2 * α) = -24 / 25) :
  Real.cos α - Real.sin α = -7 / 5 := 
sorry

end NUMINAMATH_GPT_find_cos_minus_sin_l520_52007


namespace NUMINAMATH_GPT_total_eyes_insects_l520_52080

-- Defining the conditions given in the problem
def numSpiders : Nat := 3
def numAnts : Nat := 50
def eyesPerSpider : Nat := 8
def eyesPerAnt : Nat := 2

-- Statement to prove: the total number of eyes among Nina's pet insects is 124
theorem total_eyes_insects : (numSpiders * eyesPerSpider + numAnts * eyesPerAnt) = 124 := by
  sorry

end NUMINAMATH_GPT_total_eyes_insects_l520_52080


namespace NUMINAMATH_GPT_intersection_A_B_l520_52061

open Set

def A : Set ℕ := {x | -2 < (x : ℤ) ∧ (x : ℤ) < 2}
def B : Set ℤ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ {x : ℕ | (x : ℤ) ∈ B} = {0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l520_52061


namespace NUMINAMATH_GPT_staplers_left_is_correct_l520_52006

-- Define the initial conditions as constants
def initial_staplers : ℕ := 450
def stacie_reports : ℕ := 8 * 12 -- Stacie's reports in dozens converted to actual number
def jack_reports : ℕ := 9 * 12   -- Jack's reports in dozens converted to actual number
def laura_reports : ℕ := 50      -- Laura's individual reports

-- Define the stapler usage rates
def stacie_usage_rate : ℕ := 1                  -- Stacie's stapler usage rate (1 stapler per report)
def jack_usage_rate : ℕ := stacie_usage_rate / 2  -- Jack's stapler usage rate (half of Stacie's)
def laura_usage_rate : ℕ := stacie_usage_rate * 2 -- Laura's stapler usage rate (twice of Stacie's)

-- Define the usage calculations
def stacie_usage : ℕ := stacie_reports * stacie_usage_rate
def jack_usage : ℕ := jack_reports * jack_usage_rate
def laura_usage : ℕ := laura_reports * laura_usage_rate

-- Define total staplers used
def total_usage : ℕ := stacie_usage + jack_usage + laura_usage

-- Define the number of staplers left
def staplers_left : ℕ := initial_staplers - total_usage

-- Prove that the staplers left is 200
theorem staplers_left_is_correct : staplers_left = 200 := by
  unfold staplers_left initial_staplers total_usage stacie_usage jack_usage laura_usage
  unfold stacie_reports jack_reports laura_reports
  unfold stacie_usage_rate jack_usage_rate laura_usage_rate
  sorry   -- Place proof here

end NUMINAMATH_GPT_staplers_left_is_correct_l520_52006


namespace NUMINAMATH_GPT_max_profit_l520_52087

noncomputable def profit_A (x : ℕ) : ℝ := -↑x^2 + 21 * ↑x
noncomputable def profit_B (x : ℕ) : ℝ := 2 * ↑x
noncomputable def total_profit (x : ℕ) : ℝ := profit_A x + profit_B (15 - x)

theorem max_profit : 
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 15 ∧ total_profit x = 120 := sorry

end NUMINAMATH_GPT_max_profit_l520_52087


namespace NUMINAMATH_GPT_Jonas_initial_socks_l520_52025

noncomputable def pairsOfSocks(Jonas_pairsOfShoes : ℕ) (Jonas_pairsOfPants : ℕ) 
                              (Jonas_tShirts : ℕ) (Jonas_pairsOfNewSocks : ℕ) : ℕ :=
    let individualShoes := Jonas_pairsOfShoes * 2
    let individualPants := Jonas_pairsOfPants * 2
    let individualTShirts := Jonas_tShirts
    let totalWithoutSocks := individualShoes + individualPants + individualTShirts
    let totalToDouble := (totalWithoutSocks + Jonas_pairsOfNewSocks * 2) / 2
    (totalToDouble * 2 - totalWithoutSocks) / 2

theorem Jonas_initial_socks (Jonas_pairsOfShoes : ℕ) (Jonas_pairsOfPants : ℕ) 
                             (Jonas_tShirts : ℕ) (Jonas_pairsOfNewSocks : ℕ) 
                             (h1 : Jonas_pairsOfShoes = 5)
                             (h2 : Jonas_pairsOfPants = 10)
                             (h3 : Jonas_tShirts = 10)
                             (h4 : Jonas_pairsOfNewSocks = 35) :
    pairsOfSocks Jonas_pairsOfShoes Jonas_pairsOfPants Jonas_tShirts Jonas_pairsOfNewSocks = 15 :=
by
    subst h1
    subst h2
    subst h3
    subst h4
    sorry

end NUMINAMATH_GPT_Jonas_initial_socks_l520_52025


namespace NUMINAMATH_GPT_poly_expansion_sum_l520_52097

theorem poly_expansion_sum (A B C D E : ℤ) (x : ℤ):
  (x + 3) * (4 * x^3 - 2 * x^2 + 3 * x - 1) = A * x^4 + B * x^3 + C * x^2 + D * x + E → 
  A + B + C + D + E = 16 :=
by
  sorry

end NUMINAMATH_GPT_poly_expansion_sum_l520_52097


namespace NUMINAMATH_GPT_complement_of_intersection_l520_52055

-- Declare the universal set U
def U : Set ℤ := {-1, 1, 2, 3}

-- Declare the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the given quadratic equation
def is_solution (x : ℤ) : Prop := x^2 - 2 * x - 3 = 0
def B : Set ℤ := {x : ℤ | is_solution x}

-- The main theorem to prove
theorem complement_of_intersection (A_inter_B_complement : Set ℤ) :
  A_inter_B_complement = {1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_intersection_l520_52055


namespace NUMINAMATH_GPT_part1_part2_l520_52099

-- Definitions for Part (1)
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- Part (1) Statement
theorem part1 (m : ℝ) (hm : m = 2) : A ∩ ((compl B m)) = {x | (-2 ≤ x ∧ x < -1) ∨ (3 < x ∧ x ≤ 4)} := 
by
  sorry

-- Definitions for Part (2)
def B_interval (m : ℝ) : Set ℝ := { x | (1 - m) ≤ x ∧ x ≤ (1 + m) }

-- Part (2) Statement
theorem part2 (m : ℝ) (h : ∀ x, (x ∈ A → x ∈ B_interval m)) : 0 < m ∧ m < 3 := 
by
  sorry

end NUMINAMATH_GPT_part1_part2_l520_52099


namespace NUMINAMATH_GPT_find_y_l520_52024

theorem find_y 
  (x y z : ℕ) 
  (h₁ : x + y + z = 25)
  (h₂ : x + y = 19) 
  (h₃ : y + z = 18) :
  y = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l520_52024


namespace NUMINAMATH_GPT_length_of_each_piece_cm_l520_52004

theorem length_of_each_piece_cm 
  (total_length : ℝ) 
  (number_of_pieces : ℕ) 
  (htotal : total_length = 17) 
  (hpieces : number_of_pieces = 20) : 
  (total_length / number_of_pieces) * 100 = 85 := 
by
  sorry

end NUMINAMATH_GPT_length_of_each_piece_cm_l520_52004


namespace NUMINAMATH_GPT_A_days_to_complete_work_alone_l520_52081

theorem A_days_to_complete_work_alone (x : ℝ) (h1 : 0 < x) (h2 : 0 < 18) (h3 : 1/x + 1/18 = 1/6) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_A_days_to_complete_work_alone_l520_52081


namespace NUMINAMATH_GPT_cos_alpha_minus_beta_l520_52038

theorem cos_alpha_minus_beta : 
  ∀ (α β : ℝ), 
  2 * Real.cos α - Real.cos β = 3 / 2 →
  2 * Real.sin α - Real.sin β = 2 →
  Real.cos (α - β) = -5 / 16 :=
by
  intros α β h1 h2
  sorry

end NUMINAMATH_GPT_cos_alpha_minus_beta_l520_52038


namespace NUMINAMATH_GPT_find_quantities_of_raib_ornaments_and_pendants_l520_52070

theorem find_quantities_of_raib_ornaments_and_pendants (x y : ℕ)
  (h1 : x + y = 90)
  (h2 : 40 * x + 25 * y = 2850) :
  x = 40 ∧ y = 50 :=
sorry

end NUMINAMATH_GPT_find_quantities_of_raib_ornaments_and_pendants_l520_52070


namespace NUMINAMATH_GPT_trig_identity_l520_52027

theorem trig_identity (θ : ℝ) (h₁ : Real.tan θ = 2) :
  2 * Real.cos θ / (Real.sin (Real.pi / 2 + θ) + Real.sin (Real.pi + θ)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l520_52027


namespace NUMINAMATH_GPT_complex_eq_l520_52082

theorem complex_eq : ∀ (z : ℂ), (i * z = i + z) → (z = (1 - i) / 2) :=
by
  intros z h
  sorry

end NUMINAMATH_GPT_complex_eq_l520_52082


namespace NUMINAMATH_GPT_negation_of_proposition_true_l520_52086

theorem negation_of_proposition_true :
  (¬ (∀ x: ℝ, x^2 < 1 → -1 < x ∧ x < 1)) ↔ (∃ x: ℝ, x^2 ≥ 1 ∧ (x ≤ -1 ∨ x ≥ 1)) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_true_l520_52086


namespace NUMINAMATH_GPT_parabola_constant_term_l520_52088

theorem parabola_constant_term
  (a b c : ℝ)
  (h1 : ∀ x, (-2 * (x - 1)^2 + 3) = a * x^2 + b * x + c ) :
  c = 2 :=
sorry

end NUMINAMATH_GPT_parabola_constant_term_l520_52088


namespace NUMINAMATH_GPT_least_number_to_subtract_l520_52041

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (h1: n = 509) (h2 : d = 9): ∃ k : ℕ, k = 5 ∧ ∃ m : ℕ, n - k = d * m :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l520_52041


namespace NUMINAMATH_GPT_num_valid_pairs_equals_four_l520_52071

theorem num_valid_pairs_equals_four 
  (a b : ℕ) (ha : a > 0) (hb : b > 0) (hba : b > a)
  (hcond : a * b = 3 * (a - 4) * (b - 4)) :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ 
    ∀ (p : ℕ × ℕ), p ∈ s → p.1 > 0 ∧ p.2 > 0 ∧ p.2 > p.1 ∧
      p.1 * p.2 = 3 * (p.1 - 4) * (p.2 - 4) := sorry

end NUMINAMATH_GPT_num_valid_pairs_equals_four_l520_52071


namespace NUMINAMATH_GPT_roots_of_quadratic_eq_l520_52020

theorem roots_of_quadratic_eq : ∃ (x : ℝ), (x^2 - 4 = 0) ↔ (x = 2 ∨ x = -2) :=
sorry

end NUMINAMATH_GPT_roots_of_quadratic_eq_l520_52020


namespace NUMINAMATH_GPT_correct_multiplicand_l520_52030

theorem correct_multiplicand (x : ℕ) (h1 : x * 467 = 1925817) : 
  ∃ n : ℕ, n * 467 = 1325813 :=
by
  sorry

end NUMINAMATH_GPT_correct_multiplicand_l520_52030


namespace NUMINAMATH_GPT_total_pastries_l520_52084

-- Defining the initial conditions
def Grace_pastries : ℕ := 30
def Calvin_pastries : ℕ := Grace_pastries - 5
def Phoebe_pastries : ℕ := Grace_pastries - 5
def Frank_pastries : ℕ := Calvin_pastries - 8

-- The theorem we want to prove
theorem total_pastries : 
  Calvin_pastries + Phoebe_pastries + Frank_pastries + Grace_pastries = 97 := by
  sorry

end NUMINAMATH_GPT_total_pastries_l520_52084


namespace NUMINAMATH_GPT_f_at_6_5_l520_52043

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  ∀ x : ℝ, f (x + p) = f x

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = x - 2

theorem f_at_6_5:
  (∀ x : ℝ, f (x + 2) = -1 / f x) →
  even_function f →
  specific_values f →
  f 6.5 = -0.5 :=
by
  sorry

end NUMINAMATH_GPT_f_at_6_5_l520_52043


namespace NUMINAMATH_GPT_B_pow_101_eq_B_l520_52013

-- Define the matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![-1, 0, 0], ![0, 0, 0]]

-- State the theorem
theorem B_pow_101_eq_B : B^101 = B :=
  sorry

end NUMINAMATH_GPT_B_pow_101_eq_B_l520_52013


namespace NUMINAMATH_GPT_seeds_per_watermelon_l520_52094

theorem seeds_per_watermelon (total_seeds : ℕ) (num_watermelons : ℕ) (h : total_seeds = 400 ∧ num_watermelons = 4) : total_seeds / num_watermelons = 100 :=
by
  sorry

end NUMINAMATH_GPT_seeds_per_watermelon_l520_52094


namespace NUMINAMATH_GPT_dragon_cake_votes_l520_52021

theorem dragon_cake_votes (W U D : ℕ) (x : ℕ) 
  (hW : W = 7) 
  (hU : U = 3 * W) 
  (hD : D = W + x) 
  (hTotal : W + U + D = 60) 
  (hx : x = D - W) : 
  x = 25 := 
by
  sorry

end NUMINAMATH_GPT_dragon_cake_votes_l520_52021


namespace NUMINAMATH_GPT_teenas_speed_l520_52040

theorem teenas_speed (T : ℝ) :
  (7.5 + 15 + 40 * 1.5 = T * 1.5) → T = 55 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_teenas_speed_l520_52040


namespace NUMINAMATH_GPT_jane_played_rounds_l520_52011

-- Define the conditions
def points_per_round := 10
def points_ended_with := 60
def points_lost := 20

-- Define the proof problem
theorem jane_played_rounds : (points_ended_with + points_lost) / points_per_round = 8 :=
by
  sorry

end NUMINAMATH_GPT_jane_played_rounds_l520_52011


namespace NUMINAMATH_GPT_age_difference_l520_52051

-- Definitions based on the problem statement
def son_present_age : ℕ := 33

-- Represent the problem in terms of Lean
theorem age_difference (M : ℕ) (h : M + 2 = 2 * (son_present_age + 2)) : M - son_present_age = 35 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l520_52051


namespace NUMINAMATH_GPT_middle_number_consecutive_sum_l520_52068

theorem middle_number_consecutive_sum (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : a + b + c = 30) : b = 10 :=
by
  sorry

end NUMINAMATH_GPT_middle_number_consecutive_sum_l520_52068


namespace NUMINAMATH_GPT_rocky_running_ratio_l520_52062

theorem rocky_running_ratio (x y : ℕ) (h1 : x = 4) (h2 : 2 * x + y = 36) : y / (2 * x) = 3 :=
by
  sorry

end NUMINAMATH_GPT_rocky_running_ratio_l520_52062


namespace NUMINAMATH_GPT_prism_faces_l520_52042

theorem prism_faces (E L F : ℕ) (h1 : E = 18) (h2 : 3 * L = E) (h3 : F = L + 2) : F = 8 :=
sorry

end NUMINAMATH_GPT_prism_faces_l520_52042


namespace NUMINAMATH_GPT_find_x_l520_52048

theorem find_x (x : ℝ) : 0.20 * x - (1 / 3) * (0.20 * x) = 24 → x = 180 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l520_52048


namespace NUMINAMATH_GPT_min_value_sq_distance_l520_52053

theorem min_value_sq_distance {x y : ℝ} (h : x^2 + y^2 - 4 * x + 2 = 0) : 
  ∃ (m : ℝ), m = 2 ∧ (∀ x y, x^2 + y^2 - 4 * x + 2 = 0 → x^2 + (y - 2)^2 ≥ m) :=
sorry

end NUMINAMATH_GPT_min_value_sq_distance_l520_52053


namespace NUMINAMATH_GPT_Z_equals_i_l520_52063

noncomputable def Z : ℂ := (Real.sqrt 2 - (Complex.I ^ 3)) / (1 - Real.sqrt 2 * Complex.I)

theorem Z_equals_i : Z = Complex.I := 
by 
  sorry

end NUMINAMATH_GPT_Z_equals_i_l520_52063


namespace NUMINAMATH_GPT_fraction_of_sum_l520_52075

theorem fraction_of_sum (S n : ℝ) (h1 : n = S / 6) : n / (S + n) = 1 / 7 :=
by sorry

end NUMINAMATH_GPT_fraction_of_sum_l520_52075


namespace NUMINAMATH_GPT_g_eq_g_inv_l520_52033

-- Define the function g
def g (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

-- Define the inverse function of g
noncomputable def g_inv (y : ℝ) : ℝ := (5 + Real.sqrt (1 + 8 * y)) / 4 -- simplified to handle the principal value

theorem g_eq_g_inv (x : ℝ) : g x = g_inv x → x = 1 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_g_eq_g_inv_l520_52033


namespace NUMINAMATH_GPT_pencils_total_l520_52059

-- Defining the conditions
def packs_to_pencils (packs : ℕ) : ℕ := packs * 12

def jimin_packs : ℕ := 2
def jimin_individual_pencils : ℕ := 7

def yuna_packs : ℕ := 1
def yuna_individual_pencils : ℕ := 9

-- Translating to Lean 4 statement
theorem pencils_total : 
  packs_to_pencils jimin_packs + jimin_individual_pencils + packs_to_pencils yuna_packs + yuna_individual_pencils = 52 := 
by
  sorry

end NUMINAMATH_GPT_pencils_total_l520_52059


namespace NUMINAMATH_GPT_solve_system_eq_l520_52029

theorem solve_system_eq (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x + y ≠ 0) 
  (hyz : y + z ≠ 0) (hzx : z + x ≠ 0) :
  (xy / (x + y) = 1 / 3) ∧ (yz / (y + z) = 1 / 4) ∧ (zx / (z + x) = 1 / 5) →
  (x = 1 / 2) ∧ (y = 1) ∧ (z = 1 / 3) :=
  sorry

end NUMINAMATH_GPT_solve_system_eq_l520_52029


namespace NUMINAMATH_GPT_thirteenth_result_is_878_l520_52028

-- Definitions based on the conditions
def avg_25_results : ℕ := 50
def num_25_results : ℕ := 25

def avg_first_12_results : ℕ := 14
def num_first_12_results : ℕ := 12

def avg_last_12_results : ℕ := 17
def num_last_12_results : ℕ := 12

-- Prove the 13th result is 878 given the above conditions.
theorem thirteenth_result_is_878 : 
  ((avg_25_results * num_25_results) - ((avg_first_12_results * num_first_12_results) + (avg_last_12_results * num_last_12_results))) = 878 :=
by
  sorry

end NUMINAMATH_GPT_thirteenth_result_is_878_l520_52028


namespace NUMINAMATH_GPT_count_multiples_5_or_10_l520_52093

theorem count_multiples_5_or_10 (n : ℕ) (hn : n = 999) : 
  ∃ k : ℕ, k = 199 ∧ (∀ i : ℕ, i < 1000 → (i % 5 = 0 ∨ i % 10 = 0) → i = k) := 
by {
  sorry
}

end NUMINAMATH_GPT_count_multiples_5_or_10_l520_52093


namespace NUMINAMATH_GPT_jen_shooting_game_times_l520_52026

theorem jen_shooting_game_times (x : ℕ) (h1 : 5 * x + 9 = 19) : x = 2 := by
  sorry

end NUMINAMATH_GPT_jen_shooting_game_times_l520_52026


namespace NUMINAMATH_GPT_Petya_bonus_points_l520_52064

def bonus_points (p : ℕ) : ℕ :=
  if p < 1000 then
    (20 * p) / 100
  else if p ≤ 2000 then
    200 + (30 * (p - 1000)) / 100
  else
    200 + 300 + (50 * (p - 2000)) / 100

theorem Petya_bonus_points : bonus_points 2370 = 685 :=
by sorry

end NUMINAMATH_GPT_Petya_bonus_points_l520_52064


namespace NUMINAMATH_GPT_h_value_at_3_l520_52078

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4
noncomputable def g (x : ℝ) : ℝ := (Real.sqrt (f x) - 3) ^ 2
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_value_at_3 : h 3 = 70 - 18 * Real.sqrt 13 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_h_value_at_3_l520_52078


namespace NUMINAMATH_GPT_john_moves_3594_pounds_l520_52017

def bench_press_weight := 15
def bench_press_reps := 10
def bench_press_sets := 3

def bicep_curls_weight := 12
def bicep_curls_reps := 8
def bicep_curls_sets := 4

def squats_weight := 50
def squats_reps := 12
def squats_sets := 3

def deadlift_weight := 80
def deadlift_reps := 6
def deadlift_sets := 2

def total_weight_moved : Nat :=
  (bench_press_weight * bench_press_reps * bench_press_sets) +
  (bicep_curls_weight * bicep_curls_reps * bicep_curls_sets) +
  (squats_weight * squats_reps * squats_sets) +
  (deadlift_weight * deadlift_reps * deadlift_sets)

theorem john_moves_3594_pounds :
  total_weight_moved = 3594 := by {
    sorry
}

end NUMINAMATH_GPT_john_moves_3594_pounds_l520_52017


namespace NUMINAMATH_GPT_total_spent_on_toys_and_clothes_l520_52073

def cost_toy_cars : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def cost_toy_trucks : ℝ := 5.86
def cost_pants : ℝ := 14.55
def cost_shirt : ℝ := 7.43
def cost_hat : ℝ := 12.50

theorem total_spent_on_toys_and_clothes :
  (cost_toy_cars + cost_skateboard + cost_toy_trucks) + (cost_pants + cost_shirt + cost_hat) = 60.10 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_on_toys_and_clothes_l520_52073


namespace NUMINAMATH_GPT_solve_inequality_l520_52019

theorem solve_inequality (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  ( if 0 ≤ a ∧ a < 1 / 2 then (x > a ∧ x < 1 - a) else 
    if a = 1 / 2 then false else 
    if 1 / 2 < a ∧ a ≤ 1 then (x > 1 - a ∧ x < a) else false ) ↔ ((x - a) * (x + a - 1) < 0) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l520_52019


namespace NUMINAMATH_GPT_right_triangle_sides_l520_52046

theorem right_triangle_sides (p m : ℝ)
  (hp : 0 < p)
  (hm : 0 < m) :
  ∃ a b c : ℝ, 
    a + b + c = 2 * p ∧
    a^2 + b^2 = c^2 ∧
    (1 / 2) * a * b = m^2 ∧
    c = (p^2 - m^2) / p ∧
    a = (p^2 + m^2 + Real.sqrt ((p^2 + m^2)^2 - 8 * p^2 * m^2)) / (2 * p) ∧
    b = (p^2 + m^2 - Real.sqrt ((p^2 + m^2)^2 - 8 * p^2 * m^2)) / (2 * p) := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_sides_l520_52046


namespace NUMINAMATH_GPT_max_number_of_small_boxes_l520_52052

def volume_of_large_box (length width height : ℕ) : ℕ :=
  length * width * height

def volume_of_small_box (length width height : ℕ) : ℕ :=
  length * width * height

def number_of_small_boxes (large_volume small_volume : ℕ) : ℕ :=
  large_volume / small_volume

theorem max_number_of_small_boxes :
  let large_box_length := 4 * 100  -- in cm
  let large_box_width := 2 * 100  -- in cm
  let large_box_height := 4 * 100  -- in cm
  let small_box_length := 4  -- in cm
  let small_box_width := 2  -- in cm
  let small_box_height := 2  -- in cm
  let large_volume := volume_of_large_box large_box_length large_box_width large_box_height
  let small_volume := volume_of_small_box small_box_length small_box_width small_box_height
  number_of_small_boxes large_volume small_volume = 2000000 := by
  -- Prove the statement
  sorry

end NUMINAMATH_GPT_max_number_of_small_boxes_l520_52052


namespace NUMINAMATH_GPT_tangent_line_l520_52003

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 16

theorem tangent_line (x y : ℝ) (h : f 2 = 6) : 13 * x - y - 20 = 0 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_tangent_line_l520_52003


namespace NUMINAMATH_GPT_find_b_l520_52000

-- Define the conditions as given in the problem
def poly1 (x : ℝ) : ℝ := x^2 - 2 * x - 1
def poly2 (x a b : ℝ) : ℝ := a * x^3 + b * x^2 + 1

-- Define the problem statement using these conditions
theorem find_b (a b : ℤ) (h : ∀ x, poly1 x = 0 → poly2 x a b = 0) : b = -3 :=
sorry

end NUMINAMATH_GPT_find_b_l520_52000


namespace NUMINAMATH_GPT_find_other_number_l520_52057

theorem find_other_number (B : ℕ)
  (HCF : Nat.gcd 24 B = 12)
  (LCM : Nat.lcm 24 B = 312) :
  B = 156 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l520_52057


namespace NUMINAMATH_GPT_hexagon_tiling_colors_l520_52072

-- Problem Definition
theorem hexagon_tiling_colors (k l : ℕ) (hk : 0 < k ∨ 0 < l) : 
  ∃ n: ℕ, n = k^2 + k * l + l^2 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_tiling_colors_l520_52072


namespace NUMINAMATH_GPT_cost_price_of_article_l520_52096

theorem cost_price_of_article (C : ℝ) (h1 : 86 - C = C - 42) : C = 64 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l520_52096


namespace NUMINAMATH_GPT_determine_n_l520_52090

theorem determine_n (n : ℕ) : (2 : ℕ)^n = 2 * 4^2 * 16^3 ↔ n = 17 := 
by
  sorry

end NUMINAMATH_GPT_determine_n_l520_52090


namespace NUMINAMATH_GPT_divisibility_by_5_l520_52065

theorem divisibility_by_5 (n : ℕ) (h : 0 < n) : (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end NUMINAMATH_GPT_divisibility_by_5_l520_52065


namespace NUMINAMATH_GPT_total_pies_baked_in_7_days_l520_52008

-- Define the baking rates (pies per day)
def Eddie_rate : Nat := 3
def Sister_rate : Nat := 6
def Mother_rate : Nat := 8

-- Define the duration in days
def duration : Nat := 7

-- Define the total number of pies baked in 7 days
def total_pies : Nat := Eddie_rate * duration + Sister_rate * duration + Mother_rate * duration

-- Prove the total number of pies is 119
theorem total_pies_baked_in_7_days : total_pies = 119 := by
  -- The proof will be filled here, adding sorry to skip it for now
  sorry

end NUMINAMATH_GPT_total_pies_baked_in_7_days_l520_52008


namespace NUMINAMATH_GPT_cost_of_watch_l520_52085

variable (saved amount_needed total_cost : ℕ)

-- Conditions
def connie_saved : Prop := saved = 39
def connie_needs : Prop := amount_needed = 16

-- Theorem to prove
theorem cost_of_watch : connie_saved saved → connie_needs amount_needed → total_cost = 55 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_watch_l520_52085


namespace NUMINAMATH_GPT_negation_exists_l520_52012

theorem negation_exists (x : ℝ) (h : x ≥ 0) : (¬ (∀ x : ℝ, (x ≥ 0) → (2^x > x^2))) ↔ (∃ x₀ : ℝ, (x₀ ≥ 0) ∧ (2 ^ x₀ ≤ x₀^2)) := by
  sorry

end NUMINAMATH_GPT_negation_exists_l520_52012


namespace NUMINAMATH_GPT_archer_score_below_8_probability_l520_52058

theorem archer_score_below_8_probability :
  ∀ (p10 p9 p8 : ℝ), p10 = 0.2 → p9 = 0.3 → p8 = 0.3 → 
  (1 - (p10 + p9 + p8) = 0.2) :=
by
  intros p10 p9 p8 hp10 hp9 hp8
  rw [hp10, hp9, hp8]
  sorry

end NUMINAMATH_GPT_archer_score_below_8_probability_l520_52058


namespace NUMINAMATH_GPT_sum_of_x_and_y_l520_52014

theorem sum_of_x_and_y (x y : ℝ) (h : (x + y + 2)^2 + |2 * x - 3 * y - 1| = 0) : x + y = -2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l520_52014


namespace NUMINAMATH_GPT_solve_for_x_l520_52069

theorem solve_for_x (x : ℤ) (h : x + 1 = 4) : x = 3 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l520_52069


namespace NUMINAMATH_GPT_willam_tax_payment_correct_l520_52032

noncomputable def willamFarmTax : ℝ :=
  let totalTax := 3840
  let willamPercentage := 0.2777777777777778
  totalTax * willamPercentage

-- Lean theorem statement for the problem
theorem willam_tax_payment_correct : 
  willamFarmTax = 1066.67 :=
by
  sorry

end NUMINAMATH_GPT_willam_tax_payment_correct_l520_52032


namespace NUMINAMATH_GPT_crayons_lost_or_given_away_total_l520_52095

def initial_crayons_box1 := 479
def initial_crayons_box2 := 352
def initial_crayons_box3 := 621

def remaining_crayons_box1 := 134
def remaining_crayons_box2 := 221
def remaining_crayons_box3 := 487

def crayons_lost_or_given_away_box1 := initial_crayons_box1 - remaining_crayons_box1
def crayons_lost_or_given_away_box2 := initial_crayons_box2 - remaining_crayons_box2
def crayons_lost_or_given_away_box3 := initial_crayons_box3 - remaining_crayons_box3

def total_crayons_lost_or_given_away := crayons_lost_or_given_away_box1 + crayons_lost_or_given_away_box2 + crayons_lost_or_given_away_box3

theorem crayons_lost_or_given_away_total : total_crayons_lost_or_given_away = 610 :=
by
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_crayons_lost_or_given_away_total_l520_52095


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l520_52044

-- Define the conditions for Payneful pairs
def isPaynefulPair (f g : ℝ → ℝ) : Prop :=
  (∀ x, f x ∈ Set.univ) ∧
  (∀ x, g x ∈ Set.univ) ∧
  (∀ x y, f (x + y) = f x * g y + g x * f y) ∧
  (∀ x y, g (x + y) = g x * g y - f x * f y) ∧
  (∃ a, f a ≠ 0)

-- Questions and corresponding proofs as Lean theorems
theorem part_a (f g : ℝ → ℝ) (hf : isPaynefulPair f g) : f 0 = 0 ∧ g 0 = 1 := sorry

def h (f g : ℝ → ℝ) (x : ℝ) : ℝ := (f x) ^ 2 + (g x) ^ 2

theorem part_b (f g : ℝ → ℝ) (hf : isPaynefulPair f g) : h f g 5 * h f g (-5) = 1 := sorry

theorem part_c (f g : ℝ → ℝ) (hf : isPaynefulPair f g)
  (h_bound_f : ∀ x, -10 ≤ f x ∧ f x ≤ 10) (h_bound_g : ∀ x, -10 ≤ g x ∧ g x ≤ 10):
  h f g 2021 = 1 := sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l520_52044


namespace NUMINAMATH_GPT_solution_set_of_inequality_l520_52035

noncomputable def f (x : ℝ) : ℝ := (1 / x) * (1 / 2 * (Real.log x) ^ 2 + 1 / 2)

theorem solution_set_of_inequality :
  (∀ x : ℝ, x > 0 → x < e → f x - x > f e - e) ↔ (∀ x : ℝ, 0 < x ∧ x < e) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l520_52035


namespace NUMINAMATH_GPT_base6_addition_correct_l520_52023

-- We define the numbers in base 6
def a_base6 : ℕ := 2 * 6^3 + 4 * 6^2 + 5 * 6^1 + 3 * 6^0
def b_base6 : ℕ := 1 * 6^4 + 6 * 6^3 + 4 * 6^2 + 3 * 6^1 + 2 * 6^0

-- Define the expected result in base 6 and its base 10 equivalent
def result_base6 : ℕ := 2 * 6^4 + 5 * 6^3 + 5 * 6^2 + 4 * 6^1 + 5 * 6^0
def result_base10 : ℕ := 3881

-- The proof statement
theorem base6_addition_correct : (a_base6 + b_base6 = result_base6) ∧ (result_base6 = result_base10) := by
  sorry

end NUMINAMATH_GPT_base6_addition_correct_l520_52023


namespace NUMINAMATH_GPT_find_number_l520_52076

theorem find_number 
    (x : ℝ)
    (h1 : 3 < x) 
    (h2 : x < 8) 
    (h3 : 6 < x) 
    (h4 : x < 10) : 
    x = 7 :=
sorry

end NUMINAMATH_GPT_find_number_l520_52076


namespace NUMINAMATH_GPT_sequence_general_term_l520_52022

/-- The general term formula for the sequence 0.3, 0.33, 0.333, 0.3333, … is (1 / 3) * (1 - 1 / 10 ^ n). -/
theorem sequence_general_term (n : ℕ) : 
  (∃ a : ℕ → ℚ, (∀ n, a n = 0.3 + 0.03 * (10 ^ (n + 1) - 1) / 10 ^ (n + 1))) ↔
  ∀ n, (0.3 + 0.03 * (10 ^ (n + 1) - 1) / 10 ^ (n + 1)) = (1 / 3) * (1 - 1 / 10 ^ n) :=
sorry

end NUMINAMATH_GPT_sequence_general_term_l520_52022


namespace NUMINAMATH_GPT_two_mathematicians_contemporaries_l520_52031

def contemporaries_probability :=
  let total_area := 600 * 600
  let triangle_area := 1/2 * 480 * 480
  let non_contemporaneous_area := 2 * triangle_area
  let contemporaneous_area := total_area - non_contemporaneous_area
  let probability := contemporaneous_area / total_area
  probability

theorem two_mathematicians_contemporaries :
  contemporaries_probability = 9 / 25 :=
by
  -- Skipping the intermediate proof steps
  sorry

end NUMINAMATH_GPT_two_mathematicians_contemporaries_l520_52031


namespace NUMINAMATH_GPT_least_positive_multiple_of_13_gt_418_l520_52015

theorem least_positive_multiple_of_13_gt_418 : ∃ (n : ℕ), n > 418 ∧ (13 ∣ n) ∧ n = 429 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_multiple_of_13_gt_418_l520_52015


namespace NUMINAMATH_GPT_rainfall_ratio_l520_52079

noncomputable def total_rainfall := 35
noncomputable def rainfall_second_week := 21

theorem rainfall_ratio 
  (R1 R2 : ℝ)
  (hR2 : R2 = rainfall_second_week)
  (hTotal : R1 + R2 = total_rainfall) :
  R2 / R1 = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_rainfall_ratio_l520_52079


namespace NUMINAMATH_GPT_coconut_grove_produce_trees_l520_52047

theorem coconut_grove_produce_trees (x : ℕ)
  (h1 : 60 * (x + 3) + 120 * x + 180 * (x - 3) = 100 * 3 * x)
  : x = 6 := sorry

end NUMINAMATH_GPT_coconut_grove_produce_trees_l520_52047


namespace NUMINAMATH_GPT_weekly_milk_production_l520_52009

theorem weekly_milk_production 
  (bess_milk_per_day : ℕ) 
  (brownie_milk_per_day : ℕ) 
  (daisy_milk_per_day : ℕ) 
  (total_milk_per_day : ℕ) 
  (total_milk_per_week : ℕ) 
  (h1 : bess_milk_per_day = 2) 
  (h2 : brownie_milk_per_day = 3 * bess_milk_per_day) 
  (h3 : daisy_milk_per_day = bess_milk_per_day + 1) 
  (h4 : total_milk_per_day = bess_milk_per_day + brownie_milk_per_day + daisy_milk_per_day)
  (h5 : total_milk_per_week = total_milk_per_day * 7) : 
  total_milk_per_week = 77 := 
by sorry

end NUMINAMATH_GPT_weekly_milk_production_l520_52009


namespace NUMINAMATH_GPT_opposite_of_x_abs_of_x_recip_of_x_l520_52077

noncomputable def x : ℝ := 1 - Real.sqrt 2

theorem opposite_of_x : -x = Real.sqrt 2 - 1 := 
by sorry

theorem abs_of_x : |x| = Real.sqrt 2 - 1 :=
by sorry

theorem recip_of_x : 1/x = -1 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_opposite_of_x_abs_of_x_recip_of_x_l520_52077


namespace NUMINAMATH_GPT_a_alone_can_finish_in_60_days_l520_52056

variables (A B C : ℚ)

noncomputable def a_b_work_rate := A + B = 1/40
noncomputable def a_c_work_rate := A + 1/30 = 1/20

theorem a_alone_can_finish_in_60_days (A B C : ℚ) 
  (h₁ : a_b_work_rate A B) 
  (h₂ : a_c_work_rate A) : 
  A = 1/60 := 
sorry

end NUMINAMATH_GPT_a_alone_can_finish_in_60_days_l520_52056


namespace NUMINAMATH_GPT_ab_zero_l520_52010

theorem ab_zero
  (a b : ℤ)
  (h : ∀ (m n : ℕ), ∃ (k : ℤ), a * (m : ℤ) ^ 2 + b * (n : ℤ) ^ 2 = k ^ 2) :
  a * b = 0 :=
sorry

end NUMINAMATH_GPT_ab_zero_l520_52010


namespace NUMINAMATH_GPT_ninth_term_is_83_l520_52045

-- Definitions based on conditions
def a : ℕ := 3
def d : ℕ := 10
def arith_sequence (n : ℕ) : ℕ := a + n * d

-- Theorem to prove the 9th term is 83
theorem ninth_term_is_83 : arith_sequence 8 = 83 :=
by
  sorry

end NUMINAMATH_GPT_ninth_term_is_83_l520_52045


namespace NUMINAMATH_GPT_multiply_expression_l520_52034

theorem multiply_expression (x : ℝ) : 
  (x^4 + 49 * x^2 + 2401) * (x^2 - 49) = x^6 - 117649 :=
by
  sorry

end NUMINAMATH_GPT_multiply_expression_l520_52034


namespace NUMINAMATH_GPT_fraction_simplifies_l520_52005

def current_age_grant := 25
def current_age_hospital := 40

def age_in_five_years (current_age : Nat) : Nat := current_age + 5

def grant_age_in_5_years := age_in_five_years current_age_grant
def hospital_age_in_5_years := age_in_five_years current_age_hospital

def fraction_of_ages := grant_age_in_5_years / hospital_age_in_5_years

theorem fraction_simplifies : fraction_of_ages = (2 / 3) := by
  sorry

end NUMINAMATH_GPT_fraction_simplifies_l520_52005


namespace NUMINAMATH_GPT_children_selection_l520_52092

-- Conditions and definitions
def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Proof problem statement
theorem children_selection : ∃ r : ℕ, comb 10 r = 210 ∧ r = 4 :=
by
  sorry

end NUMINAMATH_GPT_children_selection_l520_52092


namespace NUMINAMATH_GPT_mul_powers_same_base_l520_52054

theorem mul_powers_same_base : 2^2 * 2^3 = 2^5 :=
by sorry

end NUMINAMATH_GPT_mul_powers_same_base_l520_52054


namespace NUMINAMATH_GPT_num_red_balls_l520_52049

theorem num_red_balls (x : ℕ) (h1 : 60 = 60) (h2 : (x : ℝ) / (x + 60) = 0.25) : x = 20 :=
sorry

end NUMINAMATH_GPT_num_red_balls_l520_52049


namespace NUMINAMATH_GPT_teachers_photos_l520_52089

theorem teachers_photos (n : ℕ) (ht : n = 5) : 6 * 7 = 42 :=
by
  sorry

end NUMINAMATH_GPT_teachers_photos_l520_52089


namespace NUMINAMATH_GPT_part_one_part_two_part_three_l520_52037

open Nat

def number_boys := 5
def number_girls := 4
def total_people := 9
def A_included := 1
def B_included := 1

theorem part_one : (number_boys.choose 2 * number_girls.choose 2) = 60 := sorry

theorem part_two : (total_people.choose 4 - (total_people - A_included - B_included).choose 4) = 91 := sorry

theorem part_three : (total_people.choose 4 - number_boys.choose 4 - number_girls.choose 4) = 120 := sorry

end NUMINAMATH_GPT_part_one_part_two_part_three_l520_52037
