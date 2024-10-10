import Mathlib

namespace total_apples_eaten_l456_45696

def simone_daily_consumption : ℚ := 1/2
def simone_days : ℕ := 16
def lauri_daily_consumption : ℚ := 1/3
def lauri_days : ℕ := 15

theorem total_apples_eaten :
  simone_daily_consumption * simone_days + lauri_daily_consumption * lauri_days = 13 := by
  sorry

end total_apples_eaten_l456_45696


namespace fruit_theorem_l456_45608

def fruit_problem (apples pears plums cherries : ℕ) : Prop :=
  apples = 180 ∧
  apples = 3 * plums ∧
  pears = 2 * plums ∧
  cherries = 4 * apples ∧
  251 = apples - (13 * apples / 15) +
        plums - (5 * plums / 6) +
        pears - (3 * pears / 4) +
        cherries - (37 * cherries / 50)

theorem fruit_theorem :
  ∃ (apples pears plums cherries : ℕ),
    fruit_problem apples pears plums cherries := by
  sorry

end fruit_theorem_l456_45608


namespace work_time_ratio_l456_45691

/-- Given two workers A and B, this theorem proves the ratio of their individual work times
    based on their combined work time and B's individual work time. -/
theorem work_time_ratio (time_together time_B : ℝ) (h1 : time_together = 4) (h2 : time_B = 24) :
  ∃ time_A : ℝ, time_A / time_B = 1 / 5 := by
  sorry

end work_time_ratio_l456_45691


namespace sandy_marks_lost_l456_45610

theorem sandy_marks_lost (marks_per_correct : ℕ) (total_attempts : ℕ) (total_marks : ℕ) (correct_sums : ℕ) :
  marks_per_correct = 3 →
  total_attempts = 30 →
  total_marks = 45 →
  correct_sums = 21 →
  ∃ (marks_lost_per_incorrect : ℕ), 
    marks_lost_per_incorrect = 2 ∧
    total_marks = correct_sums * marks_per_correct - (total_attempts - correct_sums) * marks_lost_per_incorrect :=
by sorry

end sandy_marks_lost_l456_45610


namespace archie_record_l456_45677

/-- The number of games in a season -/
def season_length : ℕ := 16

/-- Richard's average touchdowns per game in the first 14 games -/
def richard_average : ℕ := 6

/-- The number of games Richard has played so far -/
def games_played : ℕ := 14

/-- The number of remaining games -/
def remaining_games : ℕ := season_length - games_played

/-- The average number of touchdowns Richard needs in the remaining games to beat Archie's record -/
def needed_average : ℕ := 3

theorem archie_record :
  let richard_total := richard_average * games_played + needed_average * remaining_games
  richard_total - 1 = 89 := by sorry

end archie_record_l456_45677


namespace f_properties_l456_45644

noncomputable def f (x : ℝ) := (2 * x - x^2) * Real.exp x

theorem f_properties :
  (∀ x ∈ Set.Ioo 0 2, f x > 0) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-Real.sqrt 2 - ε) (-Real.sqrt 2 + ε), f (-Real.sqrt 2) ≤ f x) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (Real.sqrt 2 - ε) (Real.sqrt 2 + ε), f x ≤ f (Real.sqrt 2)) :=
by sorry

end f_properties_l456_45644


namespace gambler_win_rate_is_40_percent_l456_45600

/-- Represents the gambler's statistics -/
structure GamblerStats where
  games_played : ℕ
  future_games : ℕ
  future_win_rate : ℚ
  target_win_rate : ℚ

/-- Calculates the current win rate of the gambler -/
def current_win_rate (stats : GamblerStats) : ℚ :=
  let total_games := stats.games_played + stats.future_games
  let future_wins := stats.future_win_rate * stats.future_games
  let total_wins := stats.target_win_rate * total_games
  (total_wins - future_wins) / stats.games_played

/-- Theorem stating the gambler's current win rate is 40% under given conditions -/
theorem gambler_win_rate_is_40_percent (stats : GamblerStats) 
  (h1 : stats.games_played = 40)
  (h2 : stats.future_games = 80)
  (h3 : stats.future_win_rate = 7/10)
  (h4 : stats.target_win_rate = 6/10) :
  current_win_rate stats = 4/10 := by
  sorry

#eval current_win_rate { games_played := 40, future_games := 80, future_win_rate := 7/10, target_win_rate := 6/10 }

end gambler_win_rate_is_40_percent_l456_45600


namespace parabola_shift_theorem_l456_45655

/-- Represents a vertical shift of a parabola -/
def vertical_shift (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f x + shift

/-- The original parabola function -/
def original_parabola : ℝ → ℝ := λ x => -x^2

theorem parabola_shift_theorem :
  vertical_shift original_parabola 2 = λ x => -x^2 + 2 := by
  sorry

end parabola_shift_theorem_l456_45655


namespace donkey_elephant_weight_difference_l456_45689

-- Define the weights and conversion factor
def elephant_weight_tons : ℝ := 3
def pounds_per_ton : ℝ := 2000
def combined_weight_pounds : ℝ := 6600

-- Define the theorem
theorem donkey_elephant_weight_difference : 
  let elephant_weight_pounds := elephant_weight_tons * pounds_per_ton
  let donkey_weight_pounds := combined_weight_pounds - elephant_weight_pounds
  let weight_difference_percentage := (elephant_weight_pounds - donkey_weight_pounds) / elephant_weight_pounds * 100
  weight_difference_percentage = 90 := by
sorry

end donkey_elephant_weight_difference_l456_45689


namespace tan_alpha_plus_beta_l456_45648

theorem tan_alpha_plus_beta (α β : ℝ) 
  (h1 : 3 * Real.tan (α / 2) + Real.tan (α / 2) ^ 2 = 1)
  (h2 : Real.sin β = 3 * Real.sin (2 * α + β)) :
  Real.tan (α + β) = -4/3 := by
  sorry

end tan_alpha_plus_beta_l456_45648


namespace rock_paper_scissors_games_l456_45601

/-- The number of students in the group -/
def num_students : ℕ := 9

/-- The number of neighbors each student doesn't play with -/
def neighbors : ℕ := 2

/-- The number of games each student plays -/
def games_per_student : ℕ := num_students - 1 - neighbors

/-- The total number of games played, counting each game twice -/
def total_games : ℕ := num_students * games_per_student

/-- The number of unique games played -/
def unique_games : ℕ := total_games / 2

theorem rock_paper_scissors_games :
  unique_games = 27 :=
sorry

end rock_paper_scissors_games_l456_45601


namespace proportion_reciprocal_outer_terms_l456_45630

theorem proportion_reciprocal_outer_terms (a b c d : ℚ) : 
  (a / b = c / d) →  -- proportion
  (b * c = 1) →      -- middle terms are reciprocals
  (a = 7 / 9) →      -- one outer term is 7/9
  (d = 9 / 7) :=     -- other outer term is 9/7
by
  sorry


end proportion_reciprocal_outer_terms_l456_45630


namespace square_sum_reciprocal_l456_45645

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end square_sum_reciprocal_l456_45645


namespace gcd_of_2_powers_l456_45637

theorem gcd_of_2_powers : Nat.gcd (2^2021 - 1) (2^2000 - 1) = 2^21 - 1 := by sorry

end gcd_of_2_powers_l456_45637


namespace triangle_side_length_l456_45657

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  b = 7 →
  c = 3 →
  Real.cos (B - C) = 17/18 →
  a = 40/3 := by
  sorry

end triangle_side_length_l456_45657


namespace min_value_sin_cos_squared_l456_45693

theorem min_value_sin_cos_squared (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  ∃ (m : ℝ), m = -1/9 ∧ ∀ z, Real.sin z + Real.sin (y + z - x) = 1/3 →
    m ≤ Real.sin z + (Real.cos z)^2 :=
sorry

end min_value_sin_cos_squared_l456_45693


namespace vector_at_t_5_l456_45671

/-- A line in 3D space parameterized by t -/
def Line := ℝ → ℝ × ℝ × ℝ

/-- The given line satisfying the conditions -/
def givenLine : Line := sorry

theorem vector_at_t_5 (h1 : givenLine 1 = (2, -1, 3))
                      (h2 : givenLine 4 = (8, -5, 11)) :
  givenLine 5 = (10, -19/3, 41/3) := by sorry

end vector_at_t_5_l456_45671


namespace minimum_value_reciprocal_sum_l456_45682

theorem minimum_value_reciprocal_sum (a b : ℝ) : 
  a > 0 → b > 0 → (2 : ℝ) = Real.sqrt (2^a * 2^b) → 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 : ℝ) = Real.sqrt (2^x * 2^y) → 1/a + 1/b ≤ 1/x + 1/y) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (2 : ℝ) = Real.sqrt (2^x * 2^y) ∧ 1/a + 1/b = 1/x + 1/y) :=
by sorry

end minimum_value_reciprocal_sum_l456_45682


namespace range_of_a_p_or_q_range_of_a_p_or_q_not_p_and_q_l456_45609

def p (a : ℝ) : Prop := 
  (a > 3 ∨ (1 < a ∧ a < 2))

def q (a : ℝ) : Prop := 
  (2 < a ∧ a < 4)

theorem range_of_a_p_or_q (a : ℝ) : 
  p a ∨ q a → a ∈ Set.union (Set.Ioo 1 2) (Set.Ioi 2) := by
  sorry

theorem range_of_a_p_or_q_not_p_and_q (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → 
  a ∈ Set.union (Set.union (Set.Ioo 1 2) (Set.Ico 2 3)) (Set.Ici 4) := by
  sorry

end range_of_a_p_or_q_range_of_a_p_or_q_not_p_and_q_l456_45609


namespace perfect_square_polynomial_l456_45653

/-- A polynomial is a perfect square if it can be expressed as (ax + b)^2 for some real numbers a and b -/
def is_perfect_square (p : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, p x = (a * x + b)^2

/-- The given polynomial -/
def polynomial (m : ℝ) (x : ℝ) : ℝ := m - 10*x + x^2

theorem perfect_square_polynomial (m : ℝ) :
  is_perfect_square (polynomial m) → m = 25 := by
  sorry

end perfect_square_polynomial_l456_45653


namespace range_of_a_l456_45688

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, Real.exp x - a ≥ 0) → a ≤ Real.exp 1 := by
  sorry

end range_of_a_l456_45688


namespace interest_rate_proof_l456_45627

/-- Proves that for given conditions, the annual interest rate is 10% -/
theorem interest_rate_proof (principal : ℝ) (time : ℝ) (diff : ℝ) : 
  principal = 1700 → 
  time = 1 → 
  diff = 4.25 → 
  ∃ (rate : ℝ), 
    rate = 10 ∧ 
    principal * ((1 + rate / 200)^2 - 1) - principal * rate * time / 100 = diff :=
by sorry

end interest_rate_proof_l456_45627


namespace nehas_mother_twice_age_l456_45631

/-- Represents the age difference between Neha's mother and Neha when the mother will be twice Neha's age -/
def AgeDifference (n : ℕ) : Prop :=
  ∃ (neha_age : ℕ),
    -- Neha's mother's current age is 60
    60 = neha_age + n ∧
    -- 12 years ago, Neha's mother was 4 times Neha's age
    (60 - 12) = 4 * (neha_age - 12) ∧
    -- In n years, Neha's mother will be twice as old as Neha
    (60 + n) = 2 * (neha_age + n)

/-- The number of years until Neha's mother is twice as old as Neha is 12 -/
theorem nehas_mother_twice_age : AgeDifference 12 := by
  sorry

end nehas_mother_twice_age_l456_45631


namespace order_of_abc_l456_45649

theorem order_of_abc (a b c : ℝ) : 
  a = 5^(1/5) → b = Real.log 3 / Real.log π → c = Real.log 0.2 / Real.log 5 → a > b ∧ b > c := by
  sorry

end order_of_abc_l456_45649


namespace function_properties_l456_45647

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def IsSymmetricAboutPoint (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

def IsSymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h1 : ∀ x, f (x + 3/2) + f x = 0)
    (h2 : IsOddFunction (fun x ↦ f (x - 3/4))) :
  (IsPeriodic f 3 ∧ 
   ¬ IsPeriodic f (3/2)) ∧ 
  IsSymmetricAboutPoint f (-3/4) ∧ 
  ¬ IsSymmetricAboutYAxis f :=
sorry

end function_properties_l456_45647


namespace complex_number_parts_opposite_l456_45686

theorem complex_number_parts_opposite (b : ℝ) : 
  let z : ℂ := (2 - b * I) / (1 + 2 * I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end complex_number_parts_opposite_l456_45686


namespace quadratic_equation_roots_l456_45668

theorem quadratic_equation_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (4 * x₁^2 - 2 * x₁ = (1/4 : ℝ)) ∧ (4 * x₂^2 - 2 * x₂ = (1/4 : ℝ)) := by
  sorry

end quadratic_equation_roots_l456_45668


namespace zhuhai_visitors_scientific_notation_l456_45607

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem zhuhai_visitors_scientific_notation :
  toScientificNotation 3001000 = ScientificNotation.mk 3.001 6 (by norm_num) :=
sorry

end zhuhai_visitors_scientific_notation_l456_45607


namespace total_wood_needed_l456_45665

def bench1_wood (length1 length2 : ℝ) (count1 count2 : ℕ) : ℝ :=
  length1 * count1 + length2 * count2

def bench2_wood (length1 length2 : ℝ) (count1 count2 : ℕ) : ℝ :=
  length1 * count1 + length2 * count2

def bench3_wood (length1 length2 : ℝ) (count1 count2 : ℕ) : ℝ :=
  length1 * count1 + length2 * count2

theorem total_wood_needed :
  let bench1 := bench1_wood 4 2 6 2
  let bench2 := bench2_wood 3 1.5 8 5
  let bench3 := bench3_wood 5 2.5 4 3
  bench1 + bench2 + bench3 = 87 := by sorry

end total_wood_needed_l456_45665


namespace unique_third_shot_combination_l456_45669

/-- Represents the scores of a player's shots -/
structure PlayerScores :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)
  (fourth : ℕ)
  (fifth : ℕ)

/-- The set of all possible scores -/
def possibleScores : Finset ℕ := {10, 9, 9, 8, 8, 5, 4, 4, 3, 2}

/-- Conditions for the shooting problem -/
structure ShootingProblem :=
  (petya : PlayerScores)
  (vasya : PlayerScores)
  (first_three_equal : petya.first + petya.second + petya.third = vasya.first + vasya.second + vasya.third)
  (last_three_relation : petya.third + petya.fourth + petya.fifth = 3 * (vasya.third + vasya.fourth + vasya.fifth))
  (all_scores_valid : ∀ s, s ∈ [petya.first, petya.second, petya.third, petya.fourth, petya.fifth,
                               vasya.first, vasya.second, vasya.third, vasya.fourth, vasya.fifth] → s ∈ possibleScores)

/-- The theorem to be proved -/
theorem unique_third_shot_combination (problem : ShootingProblem) : 
  problem.petya.third = 10 ∧ problem.vasya.third = 2 :=
sorry

end unique_third_shot_combination_l456_45669


namespace max_quarters_kevin_l456_45605

/-- Represents the value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- Represents the total amount of money Kevin has in dollars -/
def total_money : ℚ := 4.85

/-- 
Given that Kevin has $4.85 in U.S. coins and twice as many nickels as quarters,
prove that the maximum number of quarters he could have is 13.
-/
theorem max_quarters_kevin : 
  ∃ (q : ℕ), 
    q ≤ 13 ∧ 
    q * quarter_value + 2 * q * nickel_value ≤ total_money ∧
    ∀ (n : ℕ), n * quarter_value + 2 * n * nickel_value ≤ total_money → n ≤ q :=
sorry

end max_quarters_kevin_l456_45605


namespace geometric_arithmetic_progression_sum_l456_45660

theorem geometric_arithmetic_progression_sum : 
  ∃ (a b : ℝ), 
    3 < a ∧ a < b ∧ b < 9 ∧ 
    (∃ (r : ℝ), r > 0 ∧ a = 3 * r ∧ b = 3 * r^2) ∧ 
    (∃ (d : ℝ), b = a + d ∧ 9 = b + d) ∧ 
    a + b = 11.25 := by
  sorry

end geometric_arithmetic_progression_sum_l456_45660


namespace ring_arrangements_count_l456_45640

def choose (n k : ℕ) : ℕ := Nat.choose n k

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem ring_arrangements_count : 
  let total_rings : ℕ := 10
  let arranged_rings : ℕ := 6
  let fingers : ℕ := 4
  choose total_rings arranged_rings * factorial arranged_rings * choose (arranged_rings + fingers - 1) (fingers - 1) = 9130560 := by
  sorry

end ring_arrangements_count_l456_45640


namespace intersection_and_parallel_line_l456_45639

/-- Given two lines in R², prove their intersection point and a parallel line through that point. -/
theorem intersection_and_parallel_line 
  (l₁ : Set (ℝ × ℝ)) 
  (l₂ : Set (ℝ × ℝ))
  (h₁ : l₁ = {p : ℝ × ℝ | p.1 + 8 * p.2 + 7 = 0})
  (h₂ : l₂ = {p : ℝ × ℝ | 2 * p.1 + p.2 - 1 = 0})
  (l₃ : Set (ℝ × ℝ))
  (h₃ : l₃ = {p : ℝ × ℝ | p.1 + p.2 + 1 = 0}) :
  (∃! p : ℝ × ℝ, p ∈ l₁ ∧ p ∈ l₂ ∧ p = (1, -1)) ∧
  (∃ l : Set (ℝ × ℝ), l = {p : ℝ × ℝ | p.1 + p.2 = 0} ∧ 
    (1, -1) ∈ l ∧ 
    ∀ (p q : ℝ × ℝ), p ∈ l ∧ q ∈ l → p.1 - q.1 = q.2 - p.2) :=
by sorry

end intersection_and_parallel_line_l456_45639


namespace solution_set_equality_l456_45602

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the set of x that satisfy the inequality
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | f (|1/x|) < f 1}

-- Theorem statement
theorem solution_set_equality (f : ℝ → ℝ) (h : DecreasingFunction f) :
  SolutionSet f = {x | -1 < x ∧ x < 0} ∪ {x | 0 < x ∧ x < 1} :=
by sorry

end solution_set_equality_l456_45602


namespace math_contest_problem_l456_45625

theorem math_contest_problem (a b c d e f g : ℕ) : 
  a + b + c + d + e + f + g = 25 →
  b + d = 2 * (c + d) →
  a = 1 + (e + f + g) →
  a = b + c →
  b = 6 :=
by sorry

end math_contest_problem_l456_45625


namespace jack_queen_king_prob_in_standard_deck_l456_45626

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (jacks : ℕ)
  (queens : ℕ)
  (kings : ℕ)

/-- Calculates the probability of drawing a specific card from a deck -/
def draw_probability (n : ℕ) (total : ℕ) : ℚ :=
  n / total

/-- Calculates the probability of drawing a Jack, then a Queen, then a King -/
def jack_queen_king_probability (d : Deck) : ℚ :=
  (draw_probability d.jacks d.total_cards) *
  (draw_probability d.queens (d.total_cards - 1)) *
  (draw_probability d.kings (d.total_cards - 2))

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52
  , jacks := 4
  , queens := 4
  , kings := 4 }

theorem jack_queen_king_prob_in_standard_deck :
  jack_queen_king_probability standard_deck = 8 / 16575 :=
by sorry

end jack_queen_king_prob_in_standard_deck_l456_45626


namespace total_stamps_l456_45674

def stamps_problem (snowflake truck rose : ℕ) : Prop :=
  (snowflake = 11) ∧
  (truck = snowflake + 9) ∧
  (rose = truck - 13)

theorem total_stamps :
  ∀ snowflake truck rose : ℕ,
    stamps_problem snowflake truck rose →
    snowflake + truck + rose = 38 :=
by
  sorry

end total_stamps_l456_45674


namespace cone_section_height_ratio_l456_45632

/-- Given a cone with height h and base radius r, if a cross-section parallel to the base
    has an area that is half of the base area, then the ratio of the height of this section
    to the remaining height of the cone is 1:(√2 - 1). -/
theorem cone_section_height_ratio (h r x : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let base_area := π * r^2
  let section_area := π * (r * x / h)^2
  section_area = base_area / 2 →
  x / (h - x) = 1 / (Real.sqrt 2 - 1) :=
by sorry

end cone_section_height_ratio_l456_45632


namespace max_surface_area_inscribed_cylinder_l456_45697

/-- Given a cone with height h and base radius r, where h > 2r, 
    the maximum total surface area of an inscribed cylinder is πh²r / (2(h - r)). -/
theorem max_surface_area_inscribed_cylinder (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) (h_gt_2r : h > 2 * r) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    x ≤ r ∧ y ≤ h ∧
    (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x' ≤ r → y' ≤ h →
      2 * π * x' * (x' + y') ≤ 2 * π * x * (x + y)) ∧
    2 * π * x * (x + y) = π * h^2 * r / (2 * (h - r)) :=
sorry

end max_surface_area_inscribed_cylinder_l456_45697


namespace rectangle_rotation_l456_45638

theorem rectangle_rotation (w : ℝ) (a : ℝ) (l : ℝ) : 
  w = 6 →
  (1/4) * Real.pi * (l^2 + w^2) = a →
  a = 45 * Real.pi →
  l = 12 := by
sorry

end rectangle_rotation_l456_45638


namespace problem_statement_l456_45621

theorem problem_statement (x y z : ℝ) (h : (5 : ℝ) ^ x = (9 : ℝ) ^ y ∧ (9 : ℝ) ^ y = (225 : ℝ) ^ z) : 
  1 / z = 2 / x + 1 / y := by
sorry

end problem_statement_l456_45621


namespace select_blocks_count_l456_45658

def grid_size : ℕ := 6
def blocks_to_select : ℕ := 4

/-- The number of ways to select 4 blocks from a 6x6 grid, 
    such that no two blocks are in the same row or column -/
def select_blocks : ℕ := Nat.choose grid_size blocks_to_select * 
                         Nat.choose grid_size blocks_to_select * 
                         Nat.factorial blocks_to_select

theorem select_blocks_count : select_blocks = 5400 := by
  sorry

end select_blocks_count_l456_45658


namespace multiply_difference_of_cubes_l456_45656

theorem multiply_difference_of_cubes (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end multiply_difference_of_cubes_l456_45656


namespace sheila_hourly_wage_l456_45606

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_long_day : ℕ  -- Hours worked on long days
  hours_short_day : ℕ -- Hours worked on short days
  long_days : ℕ       -- Number of long workdays per week
  short_days : ℕ      -- Number of short workdays per week
  weekly_earnings : ℕ -- Weekly earnings in dollars

/-- Calculates the hourly wage given a work schedule --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.hours_long_day * schedule.long_days + 
                     schedule.hours_short_day * schedule.short_days
  schedule.weekly_earnings / total_hours

/-- Sheila's actual work schedule --/
def sheila_schedule : WorkSchedule := {
  hours_long_day := 8,
  hours_short_day := 6,
  long_days := 3,
  short_days := 2,
  weekly_earnings := 468
}

/-- Theorem stating that Sheila's hourly wage is $13 --/
theorem sheila_hourly_wage : hourly_wage sheila_schedule = 13 := by
  sorry

end sheila_hourly_wage_l456_45606


namespace min_ones_in_valid_grid_l456_45611

/-- A grid of zeros and ones -/
def Grid := Matrix (Fin 11) (Fin 11) Bool

/-- The sum of elements in a 2x2 subgrid is odd -/
def valid_subgrid (g : Grid) (i j : Fin 10) : Prop :=
  (g i j).toNat + (g i (j+1)).toNat + (g (i+1) j).toNat + (g (i+1) (j+1)).toNat % 2 = 1

/-- A grid is valid if all its 2x2 subgrids have odd sum -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j : Fin 10, valid_subgrid g i j

/-- Count the number of ones in a grid -/
def count_ones (g : Grid) : Nat :=
  (Finset.univ.sum fun i => (Finset.univ.sum fun j => (g i j).toNat))

/-- The main theorem: the minimum number of ones in a valid 11x11 grid is 25 -/
theorem min_ones_in_valid_grid :
  ∃ (g : Grid), valid_grid g ∧ count_ones g = 25 ∧
  ∀ (h : Grid), valid_grid h → count_ones h ≥ 25 :=
sorry

end min_ones_in_valid_grid_l456_45611


namespace scaled_standard_deviation_l456_45654

def data := List ℝ

def variance (d : data) : ℝ := sorry

def standardDeviation (d : data) : ℝ := sorry

def scaleData (d : data) (k : ℝ) : data := sorry

theorem scaled_standard_deviation 
  (d : data) 
  (h : variance d = 2) : 
  standardDeviation (scaleData d 2) = 2 * Real.sqrt 2 := by sorry

end scaled_standard_deviation_l456_45654


namespace derivative_even_implies_b_zero_l456_45661

/-- A cubic function with a constant term of 2 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- A function is even if f(x) = f(-x) for all x -/
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

theorem derivative_even_implies_b_zero (a b c : ℝ) :
  is_even (f' a b c) → b = 0 := by sorry

end derivative_even_implies_b_zero_l456_45661


namespace water_left_in_cooler_l456_45667

/-- Proves that the amount of water left in the cooler after filling all cups is 84 ounces -/
theorem water_left_in_cooler : 
  let initial_gallons : ℕ := 3
  let ounces_per_cup : ℕ := 6
  let rows : ℕ := 5
  let chairs_per_row : ℕ := 10
  let ounces_per_gallon : ℕ := 128
  
  let total_chairs : ℕ := rows * chairs_per_row
  let initial_ounces : ℕ := initial_gallons * ounces_per_gallon
  let ounces_used : ℕ := total_chairs * ounces_per_cup
  let ounces_left : ℕ := initial_ounces - ounces_used

  ounces_left = 84 := by sorry

end water_left_in_cooler_l456_45667


namespace dylans_mother_hotdogs_l456_45641

theorem dylans_mother_hotdogs (helens_hotdogs : ℕ) (total_hotdogs : ℕ) 
  (h1 : helens_hotdogs = 101)
  (h2 : total_hotdogs = 480) :
  total_hotdogs - helens_hotdogs = 379 := by
  sorry

end dylans_mother_hotdogs_l456_45641


namespace square_circle_octagon_l456_45681

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Square type
structure Square :=
  (a b c d : Point)

-- Define a Circle type
structure Circle :=
  (center : Point) (radius : ℝ)

-- Define an Octagon type
structure Octagon :=
  (vertices : List Point)

-- Function to check if a square can be circumscribed by a circle
def can_circumscribe (s : Square) (c : Circle) : Prop :=
  sorry

-- Function to check if an octagon is regular and inscribed in a circle
def is_regular_inscribed_octagon (o : Octagon) (c : Circle) : Prop :=
  sorry

-- Main theorem
theorem square_circle_octagon (s : Square) :
  ∃ (c : Circle) (o : Octagon),
    can_circumscribe s c ∧ is_regular_inscribed_octagon o c :=
  sorry

end square_circle_octagon_l456_45681


namespace last_locker_opened_l456_45676

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents the set of lockers -/
def Lockers := Fin 2048 → LockerState

/-- Defines the opening pattern for the lockers -/
def openingPattern (n : Nat) (lockers : Lockers) : Lockers :=
  sorry

/-- Defines the process of opening lockers until all are open -/
def openAllLockers (lockers : Lockers) : Nat :=
  sorry

/-- Theorem stating that the last locker to be opened is 1999 -/
theorem last_locker_opened (initialLockers : Lockers) 
    (h : ∀ i, initialLockers i = LockerState.Closed) : 
    openAllLockers initialLockers = 1999 :=
  sorry

end last_locker_opened_l456_45676


namespace problem_1_problem_2_problem_3_problem_4_l456_45614

-- Problem 1
theorem problem_1 : (-8) + 10 - 2 + (-1) = -1 := by sorry

-- Problem 2
theorem problem_2 : 12 - 7 * (-4) + 8 / (-2) = 36 := by sorry

-- Problem 3
theorem problem_3 : (1/2 + 1/3 - 1/6) / (-1/18) = -12 := by sorry

-- Problem 4
theorem problem_4 : -1^4 - (1 + 0.5) * (1/3) / (-4)^2 = -33/32 := by sorry

end problem_1_problem_2_problem_3_problem_4_l456_45614


namespace water_needed_for_growth_medium_l456_45659

/-- Given a growth medium mixture with initial volumes of nutrient concentrate and water,
    calculate the amount of water needed for a specified total volume. -/
theorem water_needed_for_growth_medium 
  (nutrient_vol : ℝ) 
  (initial_water_vol : ℝ) 
  (total_vol : ℝ) 
  (h1 : nutrient_vol = 0.08)
  (h2 : initial_water_vol = 0.04)
  (h3 : total_vol = 1) :
  (total_vol * initial_water_vol) / (nutrient_vol + initial_water_vol) = 1/3 := by
  sorry

#check water_needed_for_growth_medium

end water_needed_for_growth_medium_l456_45659


namespace wednesday_bags_theorem_l456_45692

/-- Represents the leaf raking business of Bob and Johnny --/
structure LeafRakingBusiness where
  price_per_bag : ℕ
  monday_bags : ℕ
  tuesday_bags : ℕ
  total_money : ℕ

/-- Calculates the number of bags raked on Wednesday --/
def bags_raked_wednesday (business : LeafRakingBusiness) : ℕ :=
  (business.total_money - business.price_per_bag * (business.monday_bags + business.tuesday_bags)) / business.price_per_bag

/-- Theorem stating that given the conditions, the number of bags raked on Wednesday is 9 --/
theorem wednesday_bags_theorem (business : LeafRakingBusiness) 
  (h1 : business.price_per_bag = 4)
  (h2 : business.monday_bags = 5)
  (h3 : business.tuesday_bags = 3)
  (h4 : business.total_money = 68) :
  bags_raked_wednesday business = 9 := by
  sorry

#eval bags_raked_wednesday { price_per_bag := 4, monday_bags := 5, tuesday_bags := 3, total_money := 68 }

end wednesday_bags_theorem_l456_45692


namespace factorial_last_nonzero_digit_not_periodic_l456_45651

/-- The last nonzero digit of n! -/
def lastNonzeroDigit (n : ℕ) : ℕ :=
  sorry

/-- The sequence of last nonzero digits of factorials is not eventually periodic -/
theorem factorial_last_nonzero_digit_not_periodic :
  ¬ ∃ (p d : ℕ), p > 0 ∧ d > 0 ∧ 
  ∀ n ≥ d, lastNonzeroDigit n = lastNonzeroDigit (n + p) :=
sorry

end factorial_last_nonzero_digit_not_periodic_l456_45651


namespace club_membership_l456_45623

theorem club_membership (total : ℕ) (lit : ℕ) (hist : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : lit = 50)
  (h3 : hist = 40)
  (h4 : both = 25) :
  total - (lit + hist - both) = 15 := by
  sorry

end club_membership_l456_45623


namespace zero_count_in_circular_sequence_l456_45622

/-- Represents a circular sequence without repetitions -/
structure CircularSequence (α : Type) where
  elements : List α
  no_repetitions : elements.Nodup
  circular : elements ≠ []

/-- Counts the number of occurrences of an element in a list -/
def count (α : Type) [DecidableEq α] (l : List α) (x : α) : Nat :=
  l.filter (· = x) |>.length

/-- Theorem: The number of zeroes in a circular sequence without repetitions is 0, 1, 2, or 4 -/
theorem zero_count_in_circular_sequence (m : ℕ) (seq : CircularSequence ℕ) :
  let zero_count := count ℕ seq.elements 0
  zero_count = 0 ∨ zero_count = 1 ∨ zero_count = 2 ∨ zero_count = 4 :=
sorry

end zero_count_in_circular_sequence_l456_45622


namespace carbon_neutrality_time_l456_45694

/-- Carbon neutrality time calculation -/
theorem carbon_neutrality_time (a : ℝ) (b : ℝ) :
  a > 0 →
  a * b^7 = (4/5) * a →
  ∃ t : ℝ, t ≥ 42 ∧ a * b^t = (1/4) * a :=
by
  sorry

end carbon_neutrality_time_l456_45694


namespace range_of_m_existence_of_a_b_l456_45685

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + m - 1

-- Part 1: Range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x ∈ Set.Icc 2 4, f m x ≥ -1) ↔ m ≤ 4 :=
sorry

-- Part 2: Existence of integers a and b
theorem existence_of_a_b :
  ∃ (a b : ℤ), a < b ∧
  (∀ x : ℝ, a ≤ f (↑a + ↑b - 1) x ∧ f (↑a + ↑b - 1) x ≤ b ↔ a ≤ x ∧ x ≤ b) ∧
  a = 0 ∧ b = 2 :=
sorry

end range_of_m_existence_of_a_b_l456_45685


namespace polygon_perimeter_sum_l456_45695

theorem polygon_perimeter_sum (n : ℕ) (x y c : ℝ) : 
  n ≥ 3 →
  x = (2 * n : ℝ) * (Real.tan (π / (n : ℝ))) * (c / (2 * π)) →
  y = (2 * n : ℝ) * (Real.sin (π / (n : ℝ))) * (c / (2 * π)) →
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < π / 2 → Real.tan θ ≥ θ) →
  x + y ≥ 2 * c :=
by sorry

end polygon_perimeter_sum_l456_45695


namespace product_equals_one_l456_45650

theorem product_equals_one (x y : ℝ) 
  (h : x * y - x / (y^2) - y / (x^2) + x^2 / (y^3) = 4) : 
  (x - 2) * (y - 2) = 1 := by
  sorry

end product_equals_one_l456_45650


namespace problem_statement_l456_45633

theorem problem_statement (a b x y : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
  (h1 : a = 2 * b)
  (h2 : x = 3 * y)
  (h3 : a + b = x * y)
  (h4 : b = 4)
  (h5 : y = 2) :
  x * a = 48 := by
sorry

end problem_statement_l456_45633


namespace necessary_but_not_sufficient_condition_l456_45620

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a > b → a > b - 1) ∧ 
  (∃ a b, a > b - 1 ∧ ¬(a > b)) :=
by sorry

end necessary_but_not_sufficient_condition_l456_45620


namespace triangle_formation_l456_45619

/-- A function that checks if three stick lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem stating which set of stick lengths can form a triangle -/
theorem triangle_formation :
  can_form_triangle 2 3 4 ∧
  ¬can_form_triangle 3 7 2 ∧
  ¬can_form_triangle 3 3 7 ∧
  ¬can_form_triangle 1 2 3 :=
by sorry

end triangle_formation_l456_45619


namespace constant_b_value_l456_45634

theorem constant_b_value (x y : ℝ) (b : ℝ) 
  (h1 : (7 * x + b * y) / (x - 2 * y) = 13)
  (h2 : x / (2 * y) = 5 / 2) :
  b = 4 := by
sorry

end constant_b_value_l456_45634


namespace weight_loss_percentage_l456_45698

def weight_before : ℝ := 840
def weight_after : ℝ := 546

theorem weight_loss_percentage : 
  (weight_before - weight_after) / weight_before * 100 = 35 := by
  sorry

end weight_loss_percentage_l456_45698


namespace unique_base_solution_l456_45652

/-- Converts a number from base b to decimal --/
def to_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Converts a number from decimal to base b --/
def from_decimal (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Checks if a number is valid in base b --/
def is_valid_in_base (n : ℕ) (b : ℕ) : Prop := sorry

theorem unique_base_solution :
  ∃! b : ℕ, b > 6 ∧ 
    is_valid_in_base 243 b ∧
    is_valid_in_base 156 b ∧
    is_valid_in_base 411 b ∧
    to_decimal 243 b + to_decimal 156 b = to_decimal 411 b ∧
    b = 10 := by sorry

end unique_base_solution_l456_45652


namespace work_completion_time_l456_45664

theorem work_completion_time (total_work : ℝ) (raja_rate : ℝ) (ram_rate : ℝ) :
  raja_rate + ram_rate = total_work / 4 →
  raja_rate = total_work / 12 →
  ram_rate = total_work / 6 :=
by sorry

end work_completion_time_l456_45664


namespace kangaroo_six_hops_l456_45699

def hop_distance (n : ℕ) : ℚ :=
  1 - (3/4)^n

theorem kangaroo_six_hops :
  hop_distance 6 = 3367 / 4096 := by
  sorry

end kangaroo_six_hops_l456_45699


namespace point_on_unit_circle_l456_45687

theorem point_on_unit_circle (s : ℝ) :
  let x := (s^2 - 1) / (s^2 + 1)
  let y := 2*s / (s^2 + 1)
  x^2 + y^2 = 1 := by
sorry

end point_on_unit_circle_l456_45687


namespace sum_of_possible_DE_values_l456_45678

/-- A function that checks if a number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

/-- A function that constructs the number D465E32 from digits D and E -/
def constructNumber (D E : ℕ) : ℕ := D * 100000 + 465000 + E * 100 + 32

/-- The theorem stating the sum of all possible values of D+E is 24 -/
theorem sum_of_possible_DE_values : 
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, ∃ D E : ℕ, isSingleDigit D ∧ isSingleDigit E ∧ 
      7 ∣ constructNumber D E ∧ n = D + E) ∧
    (∀ D E : ℕ, isSingleDigit D → isSingleDigit E → 
      7 ∣ constructNumber D E → D + E ∈ S) ∧
    S.sum id = 24 := by
  sorry

end sum_of_possible_DE_values_l456_45678


namespace solution_count_33_l456_45629

/-- The number of solutions to 3x + 2y + z = n in positive integers x, y, z -/
def solution_count (n : ℕ+) : ℕ := sorry

/-- The set of possible values for n -/
def possible_values : Set ℕ+ := {22, 24, 25}

/-- Theorem: If the equation 3x + 2y + z = n has exactly 33 solutions in positive integers x, y, and z,
    then n is in the set {22, 24, 25} -/
theorem solution_count_33 (n : ℕ+) : solution_count n = 33 → n ∈ possible_values := by sorry

end solution_count_33_l456_45629


namespace only_one_correct_statement_l456_45675

theorem only_one_correct_statement :
  (∃! n : Nat, n = 1 ∧
    (¬ (∀ a b : ℝ, a < b ∧ a ≠ 0 ∧ b ≠ 0 → 1/b < 1/a)) ∧
    (¬ (∀ a b c : ℝ, a < b → a*c < b*c)) ∧
    ((∀ a b c : ℝ, a < b → a + c < b + c)) ∧
    (¬ (∀ a b : ℝ, a^2 < b^2 → a < b))) :=
by sorry

end only_one_correct_statement_l456_45675


namespace product_equality_l456_45673

/-- The product of 12, -0.5, 3/4, and 0.20 is equal to -9/10 -/
theorem product_equality : 12 * (-0.5) * (3/4 : ℚ) * 0.20 = -9/10 := by
  sorry

end product_equality_l456_45673


namespace abs_inequality_and_fraction_inequality_l456_45616

theorem abs_inequality_and_fraction_inequality :
  (∀ x : ℝ, |x + 3| - |x - 2| ≥ 3 ↔ x ≥ 1) ∧
  (∀ a b : ℝ, a > b ∧ b > 0 → (a^2 - b^2) / (a^2 + b^2) > (a - b) / (a + b)) := by
  sorry

end abs_inequality_and_fraction_inequality_l456_45616


namespace equation_represents_hyperbola_and_ellipse_l456_45604

-- Define the equation
def equation (y z : ℝ) : Prop :=
  z^4 - 6*y^4 = 3*z^2 - 8

-- Define what it means for the equation to represent a hyperbola
def represents_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
    ∀ (y z : ℝ), eq y z ↔ a*y^2 + b*z^2 + c*y*z + d*y + e*z + f = 0

-- Define what it means for the equation to represent an ellipse
def represents_ellipse (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (a b c d e f : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (y z : ℝ), eq y z ↔ a*y^2 + b*z^2 + c*y*z + d*y + e*z + f = 0

-- Theorem statement
theorem equation_represents_hyperbola_and_ellipse :
  represents_hyperbola equation ∧ represents_ellipse equation :=
sorry

end equation_represents_hyperbola_and_ellipse_l456_45604


namespace andrew_jeffrey_walk_l456_45690

/-- Calculates the number of steps Andrew walks given Jeffrey's steps and their step ratio -/
def andrews_steps (jeffreys_steps : ℕ) (andrew_ratio jeffrey_ratio : ℕ) : ℕ :=
  (andrew_ratio * jeffreys_steps) / jeffrey_ratio

/-- Theorem stating that if Jeffrey walks 200 steps and the ratio of Andrew's to Jeffrey's steps is 3:4, then Andrew walks 150 steps -/
theorem andrew_jeffrey_walk :
  andrews_steps 200 3 4 = 150 := by
  sorry

end andrew_jeffrey_walk_l456_45690


namespace smallest_integer_y_minus_three_is_smallest_l456_45646

theorem smallest_integer_y (y : ℤ) : 3 - 5 * y < 23 ↔ y ≥ -3 :=
  sorry

theorem minus_three_is_smallest : ∃ (y : ℤ), 3 - 5 * y < 23 ∧ ∀ (z : ℤ), 3 - 5 * z < 23 → z ≥ y :=
  sorry

end smallest_integer_y_minus_three_is_smallest_l456_45646


namespace bennys_card_collection_l456_45679

theorem bennys_card_collection (original_cards : ℕ) (remaining_cards : ℕ) : 
  (remaining_cards = original_cards / 2) → (remaining_cards = 34) → (original_cards = 68) := by
  sorry

end bennys_card_collection_l456_45679


namespace candidate_vote_percentage_l456_45628

/-- Theorem: Given a total of 6000 votes and a candidate losing by 1800 votes,
    the percentage of votes the candidate received is 35%. -/
theorem candidate_vote_percentage
  (total_votes : ℕ)
  (vote_difference : ℕ)
  (h_total : total_votes = 6000)
  (h_diff : vote_difference = 1800) :
  (total_votes - vote_difference) * 100 / (2 * total_votes) = 35 := by
  sorry

end candidate_vote_percentage_l456_45628


namespace leftover_coin_value_l456_45617

/-- Represents the number of coins in a complete roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents the number of coins a person has --/
structure CoinCount where
  quarters : Nat
  dimes : Nat

/-- Calculates the value of coins in dollars --/
def coinValue (quarters dimes : Nat) : Rat :=
  (quarters * 25 + dimes * 10) / 100

theorem leftover_coin_value
  (charles marta : CoinCount)
  (roll_size : RollSize)
  (h1 : charles.quarters = 57)
  (h2 : charles.dimes = 216)
  (h3 : marta.quarters = 88)
  (h4 : marta.dimes = 193)
  (h5 : roll_size.quarters = 50)
  (h6 : roll_size.dimes = 40) :
  let total_quarters := charles.quarters + marta.quarters
  let total_dimes := charles.dimes + marta.dimes
  let leftover_quarters := total_quarters % roll_size.quarters
  let leftover_dimes := total_dimes % roll_size.dimes
  coinValue leftover_quarters leftover_dimes = 1215 / 100 := by
  sorry

end leftover_coin_value_l456_45617


namespace f_min_at_4_l456_45643

/-- The quadratic function f(x) = x^2 - 8x + 15 -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- Theorem: The function f(x) = x^2 - 8x + 15 has a minimum value when x = 4 -/
theorem f_min_at_4 : ∀ y : ℝ, f 4 ≤ f y := by
  sorry

end f_min_at_4_l456_45643


namespace derivative_f_at_zero_l456_45684

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then
    Real.sqrt (1 + Real.log (1 + x^2 * Real.sin (1/x))) - 1
  else
    0

theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end derivative_f_at_zero_l456_45684


namespace segment_ratio_in_quadrilateral_l456_45680

/-- Given four distinct points on a plane with segment lengths a, a, a, b, b, and c,
    prove that the ratio of c to a is √3/2 -/
theorem segment_ratio_in_quadrilateral (a b c : ℝ) :
  (∃ A B C D : ℝ × ℝ,
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    dist A B = a ∧ dist A C = a ∧ dist B C = a ∧
    dist A D = b ∧ dist B D = b ∧ dist C D = c) →
  c / a = Real.sqrt 3 / 2 :=
sorry

end segment_ratio_in_quadrilateral_l456_45680


namespace correct_commission_calculation_l456_45615

/-- Calculates the total commission for a salesperson selling appliances -/
def calculate_commission (num_appliances : ℕ) (total_selling_price : ℚ) : ℚ :=
  let fixed_commission := 50 * num_appliances
  let percentage_commission := 0.1 * total_selling_price
  fixed_commission + percentage_commission

/-- Theorem stating the correct commission calculation for the given scenario -/
theorem correct_commission_calculation :
  calculate_commission 6 3620 = 662 := by
  sorry

end correct_commission_calculation_l456_45615


namespace smallest_composite_no_small_factors_l456_45636

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 289 ∧ has_no_small_prime_factors 289) ∧ 
  (∀ m : ℕ, m < 289 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end smallest_composite_no_small_factors_l456_45636


namespace polynomial_factor_coefficients_l456_45663

/-- Given a polynomial ax^4 + bx^3 + 45x^2 - 18x + 10 with a factor of 5x^2 - 3x + 2,
    prove that a = 151.25 and b = -98.25 -/
theorem polynomial_factor_coefficients :
  ∀ (a b : ℝ),
  (∃ (c d : ℝ), ∀ (x : ℝ),
    a * x^4 + b * x^3 + 45 * x^2 - 18 * x + 10 =
    (5 * x^2 - 3 * x + 2) * (c * x^2 + d * x + 5)) →
  a = 151.25 ∧ b = -98.25 := by
sorry

end polynomial_factor_coefficients_l456_45663


namespace remaining_lives_l456_45612

def initial_lives : ℕ := 98
def lives_lost : ℕ := 25

theorem remaining_lives : initial_lives - lives_lost = 73 := by
  sorry

end remaining_lives_l456_45612


namespace polynomial_common_factor_l456_45666

-- Define variables
variable (x y m n : ℝ)

-- Define the polynomial
def polynomial (x y m n : ℝ) : ℝ := 4*x*(m-n) + 2*y*(m-n)^2

-- Define the common factor
def common_factor (m n : ℝ) : ℝ := 2*(m-n)

-- Theorem statement
theorem polynomial_common_factor :
  ∃ (a b : ℝ), polynomial x y m n = common_factor m n * (a*x + b*y*(m-n)) :=
sorry

end polynomial_common_factor_l456_45666


namespace small_circle_radius_l456_45662

theorem small_circle_radius (R : ℝ) (r : ℝ) : 
  R = 10 →  -- radius of large circle is 10 meters
  (3 * (2 * r) = 2 * R) →  -- three diameters of small circles equal diameter of large circle
  r = 10 / 3 :=  -- radius of small circle is 10/3 meters
by sorry

end small_circle_radius_l456_45662


namespace sqrt_of_square_of_negative_l456_45642

theorem sqrt_of_square_of_negative : ∀ (x : ℝ), x < 0 → Real.sqrt (x^2) = -x := by sorry

end sqrt_of_square_of_negative_l456_45642


namespace bailey_towel_cost_l456_45683

def guest_sets : ℕ := 2
def master_sets : ℕ := 4
def guest_price : ℚ := 40
def master_price : ℚ := 50
def discount_rate : ℚ := 0.20

theorem bailey_towel_cost : 
  let total_before_discount := guest_sets * guest_price + master_sets * master_price
  let discount_amount := discount_rate * total_before_discount
  let final_cost := total_before_discount - discount_amount
  final_cost = 224 := by sorry

end bailey_towel_cost_l456_45683


namespace factors_of_expression_factorization_of_expression_factorization_of_cube_difference_l456_45624

variable (a b c x y z : ℝ)

-- Statement 1
theorem factors_of_expression :
  (a - b) ∣ (a^2*(b - c) + b^2*(c - a) + c^2*(a - b)) ∧
  (b - c) ∣ (a^2*(b - c) + b^2*(c - a) + c^2*(a - b)) ∧
  (c - a) ∣ (a^2*(b - c) + b^2*(c - a) + c^2*(a - b)) :=
by sorry

-- Statement 2
theorem factorization_of_expression :
  a^2*(b - c) + b^2*(c - a) + c^2*(a - b) = -(a - b)*(b - c)*(c - a) :=
by sorry

-- Statement 3
theorem factorization_of_cube_difference :
  (x + y + z)^3 - x^3 - y^3 - z^3 = 3*(x + y)*(y + z)*(z + x) :=
by sorry

end factors_of_expression_factorization_of_expression_factorization_of_cube_difference_l456_45624


namespace triangle_base_calculation_l456_45670

theorem triangle_base_calculation (base1 height1 height2 : ℝ) :
  base1 = 15 ∧ height1 = 12 ∧ height2 = 18 →
  ∃ base2 : ℝ, base2 = 20 ∧ base2 * height2 = 2 * (base1 * height1) :=
by sorry

end triangle_base_calculation_l456_45670


namespace vote_increase_l456_45613

/-- Represents the voting scenario for a bill --/
structure VotingScenario where
  total_members : ℕ
  initial_for : ℕ
  initial_against : ℕ
  revote_for : ℕ
  revote_against : ℕ

/-- Conditions for the voting scenario --/
def voting_conditions (v : VotingScenario) : Prop :=
  v.total_members = 500 ∧
  v.initial_for + v.initial_against = v.total_members ∧
  v.initial_against > v.initial_for ∧
  v.revote_for + v.revote_against = v.total_members ∧
  v.revote_for = (10 * v.initial_against) / 9 ∧
  (v.revote_for - v.revote_against) = 3 * (v.initial_against - v.initial_for)

/-- Theorem stating the increase in votes for the bill --/
theorem vote_increase (v : VotingScenario) (h : voting_conditions v) :
  v.revote_for - v.initial_for = 59 :=
sorry

end vote_increase_l456_45613


namespace ellipse_equation_l456_45618

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci are on the x-axis
  foci_on_x_axis : Bool
  -- Passes through (0,1) and (3,0)
  passes_through_points : (ℝ × ℝ) → (ℝ × ℝ) → Prop
  -- Eccentricity is 3/5
  eccentricity : ℚ
  -- Length of minor axis is 8
  minor_axis_length : ℝ

/-- The standard equation of an ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 16 = 1

/-- Theorem: The standard equation of the ellipse with given properties -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.foci_on_x_axis = true)
  (h2 : e.passes_through_points (0, 1) (3, 0))
  (h3 : e.eccentricity = 3/5)
  (h4 : e.minor_axis_length = 8) :
  ∀ x y : ℝ, standard_equation e x y :=
sorry

end ellipse_equation_l456_45618


namespace triangle_side_length_l456_45672

/-- Given a triangle ABC with side lengths a, b, and c opposite to angles A, B, and C respectively,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →
  B = π/3 →
  a^2 + c^2 = 3*a*c →
  b = 2 * Real.sqrt 2 := by
sorry

end triangle_side_length_l456_45672


namespace smallest_perfect_cube_l456_45603

/-- Given distinct prime numbers p, q, and r, prove that (p * q * r^2)^3 is the smallest positive
    perfect cube that includes the factor n = p * q^2 * r^4 -/
theorem smallest_perfect_cube (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ (k : ℕ), k > 0 ∧ (p * q * r^2)^3 = k^3 ∧
  ∀ (m : ℕ), m > 0 → m^3 ≥ (p * q * r^2)^3 → (p * q^2 * r^4) ∣ m^3 := by
  sorry

end smallest_perfect_cube_l456_45603


namespace translated_line_through_origin_l456_45635

/-- A line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically by a given amount -/
def translate_line (l : Line) (dy : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - dy }

/-- Check if a line passes through a point -/
def passes_through (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem translated_line_through_origin (b : ℝ) :
  let original_line : Line := { slope := 2, intercept := b }
  let translated_line := translate_line original_line 2
  passes_through translated_line 0 0 → b = 2 :=
by
  sorry

end translated_line_through_origin_l456_45635
