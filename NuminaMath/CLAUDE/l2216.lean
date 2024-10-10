import Mathlib

namespace nina_savings_time_l2216_221621

theorem nina_savings_time (video_game_cost : ℝ) (headset_cost : ℝ) (sales_tax_rate : ℝ) 
  (weekly_allowance : ℝ) (savings_rate : ℝ) :
  video_game_cost = 50 →
  headset_cost = 70 →
  sales_tax_rate = 0.12 →
  weekly_allowance = 10 →
  savings_rate = 0.40 →
  ⌈(((video_game_cost + headset_cost) * (1 + sales_tax_rate)) / 
    (weekly_allowance * savings_rate))⌉ = 34 := by
  sorry

end nina_savings_time_l2216_221621


namespace sqrt_three_between_l2216_221669

theorem sqrt_three_between (n : ℕ+) : 
  (1 + 3 / (n + 1 : ℝ) < Real.sqrt 3 ∧ Real.sqrt 3 < 1 + 3 / (n : ℝ)) → n = 4 := by
  sorry

end sqrt_three_between_l2216_221669


namespace inequality_solution_set_l2216_221617

theorem inequality_solution_set :
  {x : ℝ | (4 : ℝ) / (x + 1) ≤ 1} = Set.Iic (-1) ∪ Set.Ici 3 := by
  sorry

end inequality_solution_set_l2216_221617


namespace function_properties_l2216_221614

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * sin (2 * x) - cos x ^ 2 - 1/2

theorem function_properties (α : ℝ) (h1 : tan α = 2 * Real.sqrt 3) (h2 : f (Real.sqrt 3 / 2) α = -3/26) :
  ∃ (m : ℝ),
    m = Real.sqrt 3 / 2 ∧
    (∀ x, f m (x + π) = f m x) ∧
    (∀ x ∈ Set.Icc 0 π, 
      (x ∈ Set.Icc 0 (π/3) ∨ x ∈ Set.Icc (5*π/6) π) → 
      ∀ y ∈ Set.Icc 0 π, x < y → f m x < f m y) :=
by sorry

end function_properties_l2216_221614


namespace union_of_A_and_B_l2216_221607

def A : Set ℕ := {0, 1}
def B : Set ℕ := {0, 2}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2} := by sorry

end union_of_A_and_B_l2216_221607


namespace det_A_equals_one_l2216_221654

open Matrix

theorem det_A_equals_one (a d : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 1; -2, d]
  (A + A⁻¹ = 0) → det A = 1 := by
  sorry

end det_A_equals_one_l2216_221654


namespace fraction_zero_iff_x_zero_l2216_221652

theorem fraction_zero_iff_x_zero (x : ℝ) (h : x ≠ -2) :
  2 * x / (x + 2) = 0 ↔ x = 0 := by sorry

end fraction_zero_iff_x_zero_l2216_221652


namespace pages_per_chapter_book_chapters_calculation_l2216_221692

theorem pages_per_chapter 
  (total_chapters : Nat) 
  (days_to_finish : Nat) 
  (chapters_per_day : Nat) : Nat :=
  let total_chapters_read := days_to_finish * chapters_per_day
  total_chapters_read / total_chapters

theorem book_chapters_calculation 
  (total_chapters : Nat) 
  (days_to_finish : Nat) 
  (chapters_per_day : Nat) :
  total_chapters = 2 →
  days_to_finish = 664 →
  chapters_per_day = 332 →
  pages_per_chapter total_chapters days_to_finish chapters_per_day = 110224 := by
  sorry

end pages_per_chapter_book_chapters_calculation_l2216_221692


namespace complex_sum_theorem_l2216_221633

theorem complex_sum_theorem (a b c d e f g h : ℝ) : 
  b = 2 → 
  g = -(a + c + e) → 
  (a + b * Complex.I) + (c + d * Complex.I) + (e + f * Complex.I) + (g + h * Complex.I) = 3 * Complex.I → 
  d + f + h = 1 := by
  sorry

end complex_sum_theorem_l2216_221633


namespace population_doubling_time_l2216_221605

/-- The number of years required for a population to double given birth and death rates -/
theorem population_doubling_time (birth_rate death_rate : ℚ) : 
  birth_rate = 39.4 ∧ death_rate = 19.4 → 
  (70 : ℚ) / ((birth_rate - death_rate) / 10) = 35 := by
  sorry

end population_doubling_time_l2216_221605


namespace registration_theorem_l2216_221668

/-- The number of possible ways for students to register for events. -/
def registration_combinations (num_students : ℕ) (num_events : ℕ) : ℕ :=
  num_events ^ num_students

/-- Theorem stating that with 4 students and 3 events, there are 81 possible registration combinations. -/
theorem registration_theorem :
  registration_combinations 4 3 = 81 := by
  sorry

end registration_theorem_l2216_221668


namespace point_on_line_l2216_221616

/-- Given three points A, B, and C on a straight line, prove that the y-coordinate of C is 7. -/
theorem point_on_line (A B C : ℝ × ℝ) : 
  A = (1, -1) → B = (3, 3) → C.1 = 5 → 
  (C.2 - A.2) / (C.1 - A.1) = (B.2 - A.2) / (B.1 - A.1) → 
  C.2 = 7 := by sorry

end point_on_line_l2216_221616


namespace number_of_friends_l2216_221673

/-- Given that Mary, Sam, Keith, and Alyssa each have 6 baseball cards,
    prove that the number of friends is 4. -/
theorem number_of_friends : ℕ :=
  let mary_cards := 6
  let sam_cards := 6
  let keith_cards := 6
  let alyssa_cards := 6
  4

#check number_of_friends

end number_of_friends_l2216_221673


namespace minimum_additional_weeks_to_win_l2216_221667

def puppy_cost : ℕ := 1000
def weekly_prize : ℕ := 100
def additional_wins_needed : ℕ := 8

theorem minimum_additional_weeks_to_win (current_savings : ℕ) : 
  (current_savings + additional_wins_needed * weekly_prize = puppy_cost) → 
  additional_wins_needed = 8 := by
  sorry

end minimum_additional_weeks_to_win_l2216_221667


namespace candidate_vote_difference_l2216_221689

theorem candidate_vote_difference :
  let total_votes : ℝ := 25000.000000000007
  let candidate_percentage : ℝ := 0.4
  let rival_percentage : ℝ := 1 - candidate_percentage
  let candidate_votes : ℝ := total_votes * candidate_percentage
  let rival_votes : ℝ := total_votes * rival_percentage
  let vote_difference : ℝ := rival_votes - candidate_votes
  ∃ (ε : ℝ), ε > 0 ∧ |vote_difference - 5000| < ε :=
by
  sorry

end candidate_vote_difference_l2216_221689


namespace quadratic_equation_root_l2216_221675

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = 1) → m = -4 := by
  sorry

end quadratic_equation_root_l2216_221675


namespace high_school_twelve_games_l2216_221672

/-- The number of teams in the conference -/
def num_teams : ℕ := 12

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- The total number of games in a season -/
def total_games : ℕ := num_teams * (num_teams - 1) + num_teams * non_conference_games

/-- Theorem stating the total number of games in a season -/
theorem high_school_twelve_games :
  total_games = 204 :=
sorry

end high_school_twelve_games_l2216_221672


namespace triangle_equilateral_l2216_221634

theorem triangle_equilateral (a b c : ℝ) 
  (triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h : a^2 + b^2 + c^2 = a*b + b*c + c*a) : 
  a = b ∧ b = c := by
sorry

end triangle_equilateral_l2216_221634


namespace regular_polygon_with_150_degree_angles_has_12_sides_l2216_221659

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ θ : ℝ, θ = 150 → (n : ℝ) * θ = 180 * ((n : ℝ) - 2)) →
  n = 12 := by
  sorry

end regular_polygon_with_150_degree_angles_has_12_sides_l2216_221659


namespace function_properties_l2216_221682

/-- Given functions f and g with parameter a, proves properties about their extrema and monotonicity -/
theorem function_properties (a : ℝ) (h : a ≤ 0) :
  let f := fun x : ℝ ↦ Real.exp x + a * x
  let g := fun x : ℝ ↦ a * x - Real.log x
  -- The minimum of f occurs at ln(-a)
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = Real.log (-a)) ∧
  -- The minimum value of f is -a + a * ln(-a)
  (∃ (y_min : ℝ), ∀ (x : ℝ), f x ≥ y_min ∧ y_min = -a + a * Real.log (-a)) ∧
  -- f has no maximum value
  (¬∃ (y_max : ℝ), ∀ (x : ℝ), f x ≤ y_max) ∧
  -- f and g have the same monotonicity on some interval iff a ∈ (-∞, -1)
  (∃ (M : Set ℝ), (∀ (x y : ℝ), x ∈ M → y ∈ M → x < y → (f x < f y ↔ g x < g y)) ↔ a < -1) := by
  sorry

end function_properties_l2216_221682


namespace first_group_size_correct_l2216_221696

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 24

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_correct :
  first_group_size * first_group_days * first_group_hours_per_day =
  second_group_size * second_group_days * second_group_hours_per_day :=
by sorry

end first_group_size_correct_l2216_221696


namespace circle_area_increase_l2216_221680

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
  sorry

end circle_area_increase_l2216_221680


namespace correct_transformation_l2216_221620

theorem correct_transformation (x : ℝ) : 3 * x + 5 = 4 * x → 3 * x - 4 * x = -5 := by
  sorry

end correct_transformation_l2216_221620


namespace bridge_length_calculation_l2216_221695

/-- Calculates the length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 170 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 205 := by sorry

#check bridge_length_calculation

end bridge_length_calculation_l2216_221695


namespace original_price_calculation_l2216_221687

/-- 
Theorem: If an item's price is increased by 15%, then decreased by 20%, 
resulting in a final price of 46 yuan, the original price was 50 yuan.
-/
theorem original_price_calculation (original_price : ℝ) : 
  (original_price * 1.15 * 0.8 = 46) → original_price = 50 := by
  sorry

end original_price_calculation_l2216_221687


namespace total_crayons_l2216_221650

def packs : ℕ := 4
def crayons_per_pack : ℕ := 10
def extra_crayons : ℕ := 6

theorem total_crayons : packs * crayons_per_pack + extra_crayons = 46 := by
  sorry

end total_crayons_l2216_221650


namespace same_terminal_side_l2216_221691

theorem same_terminal_side : ∀ θ : Real,
  θ ≥ 0 ∧ θ < 2 * Real.pi →
  (θ = 2 * Real.pi / 3) ↔ ∃ k : Int, θ = -4 * Real.pi / 3 + 2 * Real.pi * k := by
  sorry

end same_terminal_side_l2216_221691


namespace physical_fitness_test_participation_l2216_221624

/-- The number of students who met the standards in the physical fitness test -/
def students_met_standards : ℕ := 900

/-- The percentage of students who took the test but did not meet the standards -/
def percentage_not_met_standards : ℚ := 25 / 100

/-- The percentage of students who did not participate in the test -/
def percentage_not_participated : ℚ := 4 / 100

/-- The total number of students in the sixth grade -/
def total_students : ℕ := 1200

/-- The number of students who did not participate in the physical fitness test -/
def students_not_participated : ℕ := 48

theorem physical_fitness_test_participation :
  (students_not_participated : ℚ) = percentage_not_participated * (total_students : ℚ) :=
by sorry

end physical_fitness_test_participation_l2216_221624


namespace average_home_runs_l2216_221653

theorem average_home_runs : 
  let players_5 := 7
  let players_6 := 5
  let players_7 := 4
  let players_9 := 3
  let players_11 := 1
  let total_players := players_5 + players_6 + players_7 + players_9 + players_11
  let total_home_runs := 5 * players_5 + 6 * players_6 + 7 * players_7 + 9 * players_9 + 11 * players_11
  (total_home_runs : ℚ) / total_players = 131 / 20 := by
  sorry

end average_home_runs_l2216_221653


namespace jellybean_problem_minimum_jellybean_count_l2216_221638

theorem jellybean_problem (n : ℕ) : 
  (n ≥ 150) ∧ (n % 17 = 15) → n ≥ 151 :=
by sorry

theorem minimum_jellybean_count : 
  ∃ (n : ℕ), n ≥ 150 ∧ n % 17 = 15 ∧ ∀ (m : ℕ), m ≥ 150 ∧ m % 17 = 15 → m ≥ n :=
by sorry

end jellybean_problem_minimum_jellybean_count_l2216_221638


namespace intersection_point_unique_l2216_221699

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 3) / 2 = (y + 1) / 3 ∧ (x - 3) / 2 = (z + 3) / 2

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  3 * x + 4 * y + 7 * z - 16 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (5, 2, -1)

theorem intersection_point_unique :
  ∃! p : ℝ × ℝ × ℝ, 
    line_equation p.1 p.2.1 p.2.2 ∧ 
    plane_equation p.1 p.2.1 p.2.2 ∧
    p = intersection_point :=
by
  sorry

end intersection_point_unique_l2216_221699


namespace composition_of_odd_functions_l2216_221660

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem composition_of_odd_functions (f : ℝ → ℝ) (h : IsOdd f) :
  IsOdd (fun x ↦ f (f (f (f x)))) := by sorry

end composition_of_odd_functions_l2216_221660


namespace initial_sales_tax_percentage_l2216_221609

/-- Proves that the initial sales tax percentage is 3.5% given the conditions -/
theorem initial_sales_tax_percentage 
  (market_price : ℝ) 
  (new_tax_rate : ℝ) 
  (tax_difference : ℝ) 
  (h1 : market_price = 7800)
  (h2 : new_tax_rate = 10 / 3)
  (h3 : tax_difference = 13) :
  ∃ (x : ℝ), x = 3.5 ∧ market_price * (x / 100 - new_tax_rate / 100) = tax_difference :=
sorry

end initial_sales_tax_percentage_l2216_221609


namespace equation_solution_l2216_221645

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  3 - 5 / x + 2 / (x^2) = 0 → (3 / x = 9 / 2 ∨ 3 / x = 3) :=
by sorry

end equation_solution_l2216_221645


namespace right_triangle_sides_exist_l2216_221666

/-- A right triangle with perimeter k and incircle radius ρ --/
structure RightTriangle (k ρ : ℝ) where
  a : ℝ
  b : ℝ
  c : ℝ
  perimeter_eq : a + b + c = k
  incircle_eq : a * b = 2 * ρ * (k / 2)
  pythagorean : a^2 + b^2 = c^2

/-- The side lengths of a right triangle satisfy the given conditions --/
theorem right_triangle_sides_exist (k ρ : ℝ) (hk : k > 0) (hρ : ρ > 0) :
  ∃ (t : RightTriangle k ρ), t.a > 0 ∧ t.b > 0 ∧ t.c > 0 :=
sorry

end right_triangle_sides_exist_l2216_221666


namespace upload_time_calculation_l2216_221631

def file_size : ℝ := 160
def upload_speed : ℝ := 8

theorem upload_time_calculation : 
  file_size / upload_speed = 20 := by sorry

end upload_time_calculation_l2216_221631


namespace p_sufficient_not_necessary_for_q_l2216_221643

-- Define the propositions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := ¬(∀ x : ℝ, x^2 + a*x + 1 ≥ 0)

-- Theorem statement
theorem p_sufficient_not_necessary_for_q :
  (∃ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end p_sufficient_not_necessary_for_q_l2216_221643


namespace three_dice_probability_l2216_221670

/-- A fair 6-sided die -/
def Die : Type := Fin 6

/-- A roll of three dice -/
def ThreeDiceRoll : Type := Die × Die × Die

/-- The total number of possible outcomes when rolling three 6-sided dice -/
def totalOutcomes : ℕ := 216

/-- The number of permutations of three distinct numbers -/
def permutations : ℕ := 6

/-- The probability of rolling a 2, 3, and 4 in any order with three fair 6-sided dice -/
def winProbability : ℚ := 1 / 36

/-- Theorem: The probability of rolling a 2, 3, and 4 in any order with three fair 6-sided dice is 1/36 -/
theorem three_dice_probability :
  (permutations : ℚ) / totalOutcomes = winProbability :=
sorry

end three_dice_probability_l2216_221670


namespace rectangular_prism_sum_of_dimensions_l2216_221661

theorem rectangular_prism_sum_of_dimensions 
  (α β γ : ℝ) 
  (h1 : α * β = 18) 
  (h2 : α * γ = 36) 
  (h3 : β * γ = 72) : 
  α + β + γ = 21 := by
sorry

end rectangular_prism_sum_of_dimensions_l2216_221661


namespace solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2216_221622

-- Define the function f
def f (x a : ℝ) : ℝ := 5 - |x + a| - |x - 2|

-- Theorem for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for part (2)
theorem range_of_a_when_f_leq_one :
  {a : ℝ | ∀ x, f x a ≤ 1} = {a : ℝ | a ≤ -6 ∨ a ≥ 2} := by sorry

end solution_set_when_a_is_one_range_of_a_when_f_leq_one_l2216_221622


namespace count_satisfying_integers_is_five_l2216_221619

/-- The count of positive integers n satisfying (n + 1050) / 90 = ⌊√n⌋ -/
def count_satisfying_integers : ℕ := 5

/-- Predicate defining when a positive integer satisfies the equation -/
def satisfies_equation (n : ℕ+) : Prop :=
  (n + 1050) / 90 = ⌊Real.sqrt n⌋

/-- Theorem stating that exactly 5 positive integers satisfy the equation -/
theorem count_satisfying_integers_is_five :
  (∃! (S : Finset ℕ+), S.card = count_satisfying_integers ∧ 
    ∀ n, n ∈ S ↔ satisfies_equation n) :=
by sorry

end count_satisfying_integers_is_five_l2216_221619


namespace target_hit_probability_l2216_221610

theorem target_hit_probability (total_groups : ℕ) (hit_groups : ℕ) : 
  total_groups = 20 → hit_groups = 5 → 
  (hit_groups : ℚ) / total_groups = 1/4 := by
  sorry

end target_hit_probability_l2216_221610


namespace final_week_study_hours_l2216_221678

def study_hours : List ℕ := [8, 10, 9, 11, 10, 7]
def total_weeks : ℕ := 7
def required_average : ℕ := 9

theorem final_week_study_hours :
  ∃ (x : ℕ), 
    (List.sum study_hours + x) / total_weeks = required_average ∧
    x = 8 := by
  sorry

end final_week_study_hours_l2216_221678


namespace projection_vector_l2216_221658

def a : Fin 3 → ℝ := ![0, 1, 1]
def b : Fin 3 → ℝ := ![1, 1, 0]

theorem projection_vector :
  let proj := (a • b) / (a • a) • a
  proj = ![0, 1/2, 1/2] := by sorry

end projection_vector_l2216_221658


namespace investment_calculation_l2216_221630

/-- Given two investors P and Q, where the profit is divided in the ratio 2:3
    and P invested Rs 40000, prove that Q invested Rs 60000 -/
theorem investment_calculation (P Q : ℕ) (profit_ratio : ℚ) :
  P = 40000 →
  profit_ratio = 2 / 3 →
  Q = 60000 :=
by sorry

end investment_calculation_l2216_221630


namespace principal_amount_l2216_221649

/-- Calculates the total interest paid over 11 years given the principal amount -/
def totalInterest (principal : ℝ) : ℝ :=
  principal * 0.06 * 3 + principal * 0.09 * 5 + principal * 0.13 * 3

/-- Theorem stating that the principal amount borrowed is 8000 given the total interest paid -/
theorem principal_amount (totalInterestPaid : ℝ) 
  (h : totalInterestPaid = 8160) : 
  ∃ (principal : ℝ), totalInterest principal = totalInterestPaid ∧ principal = 8000 := by
  sorry

#check principal_amount

end principal_amount_l2216_221649


namespace part1_part2_l2216_221698

-- Define the operation F
def F (a b x y : ℝ) : ℝ := a * x + b * y

-- Theorem for part 1
theorem part1 (a b : ℝ) : 
  F a b (-1) 3 = 2 ∧ F a b 1 (-2) = 8 → a = 28 ∧ b = 10 := by sorry

-- Theorem for part 2
theorem part2 (a b : ℝ) :
  b ≥ 0 ∧ F a b 2 1 = 5 → a ≤ 5/2 := by sorry

end part1_part2_l2216_221698


namespace cans_recycled_from_64_l2216_221606

def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 4 then 0
  else (initial_cans / 4) + recycle_cans (initial_cans / 4)

theorem cans_recycled_from_64 :
  recycle_cans 64 = 21 :=
sorry

end cans_recycled_from_64_l2216_221606


namespace bathroom_break_duration_l2216_221656

theorem bathroom_break_duration
  (total_distance : ℝ)
  (driving_speed : ℝ)
  (lunch_break : ℝ)
  (num_bathroom_breaks : ℕ)
  (total_trip_time : ℝ)
  (h1 : total_distance = 480)
  (h2 : driving_speed = 60)
  (h3 : lunch_break = 0.5)
  (h4 : num_bathroom_breaks = 2)
  (h5 : total_trip_time = 9) :
  (total_trip_time - total_distance / driving_speed - lunch_break) / num_bathroom_breaks = 0.25 := by
  sorry

#check bathroom_break_duration

end bathroom_break_duration_l2216_221656


namespace beyonce_album_songs_l2216_221632

/-- The number of songs in Beyonce's first two albums -/
def songs_in_first_two_albums (total_songs num_singles num_albums songs_in_third_album : ℕ) : ℕ :=
  total_songs - num_singles - songs_in_third_album

theorem beyonce_album_songs :
  songs_in_first_two_albums 55 5 3 20 = 30 := by
  sorry

end beyonce_album_songs_l2216_221632


namespace tetrahedron_volume_not_determined_by_face_areas_l2216_221665

/-- A tetrahedron with four faces --/
structure Tetrahedron where
  faces : Fin 4 → Real
  volume : Real

/-- Theorem stating that the volume of a tetrahedron is not uniquely determined by its face areas --/
theorem tetrahedron_volume_not_determined_by_face_areas :
  ∃ (t1 t2 : Tetrahedron), (∀ i : Fin 4, t1.faces i = t2.faces i) ∧ t1.volume ≠ t2.volume :=
sorry

end tetrahedron_volume_not_determined_by_face_areas_l2216_221665


namespace complex_root_magnitude_l2216_221604

theorem complex_root_magnitude (z : ℂ) : z^2 + z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_root_magnitude_l2216_221604


namespace airplane_speed_l2216_221628

/-- Given a distance and flight times with and against wind, calculate the average speed without wind -/
theorem airplane_speed (distance : ℝ) (time_with_wind : ℝ) (time_against_wind : ℝ)
  (h1 : distance = 9360)
  (h2 : time_with_wind = 12)
  (h3 : time_against_wind = 13) :
  ∃ (speed_no_wind : ℝ) (wind_speed : ℝ),
    speed_no_wind = 750 ∧
    time_with_wind * (speed_no_wind + wind_speed) = distance ∧
    time_against_wind * (speed_no_wind - wind_speed) = distance :=
by sorry

end airplane_speed_l2216_221628


namespace inequality_solution_l2216_221611

theorem inequality_solution (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 8) ≥ 1/2 ↔ x ∈ Set.Ioc (-8) (-2) ∪ Set.Icc 6 8 := by
  sorry

end inequality_solution_l2216_221611


namespace equation_two_distinct_roots_l2216_221662

theorem equation_two_distinct_roots (k : ℂ) : 
  (∃ (x y : ℂ), x ≠ y ∧ 
    (∀ z : ℂ, z / (z + 3) + z / (z - 1) = k * z ↔ z = x ∨ z = y)) ↔ 
  k ≠ 0 :=
sorry

end equation_two_distinct_roots_l2216_221662


namespace when_you_rescind_price_is_85_l2216_221651

/-- The price of a CD of "The Life Journey" -/
def life_journey_price : ℕ := 100

/-- The price of a CD of "A Day a Life" -/
def day_life_price : ℕ := 50

/-- The number of each CD type bought -/
def quantity : ℕ := 3

/-- The total amount spent -/
def total_spent : ℕ := 705

/-- The price of a CD of "When You Rescind" -/
def when_you_rescind_price : ℕ := 85

/-- Theorem stating that the price of "When You Rescind" CD is 85 -/
theorem when_you_rescind_price_is_85 :
  quantity * life_journey_price + quantity * day_life_price + quantity * when_you_rescind_price = total_spent :=
by sorry

end when_you_rescind_price_is_85_l2216_221651


namespace badminton_players_count_l2216_221603

theorem badminton_players_count (total : ℕ) (tennis : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 40)
  (h_tennis : tennis = 18)
  (h_neither : neither = 5)
  (h_both : both = 3) :
  total = tennis + (total - tennis - neither) - both + neither :=
by sorry

end badminton_players_count_l2216_221603


namespace solve_for_b_l2216_221647

theorem solve_for_b (b : ℝ) : 
  4 * ((3.6 * b * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005 → b = 0.48 := by
  sorry

end solve_for_b_l2216_221647


namespace cyclists_speed_problem_l2216_221676

/-- Two cyclists problem -/
theorem cyclists_speed_problem (north_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  north_speed = 10 →
  time = 1.4285714285714286 →
  distance = 50 →
  ∃ (south_speed : ℝ), south_speed = 25 ∧ (north_speed + south_speed) * time = distance :=
by sorry

end cyclists_speed_problem_l2216_221676


namespace smallest_lcm_with_gcd_5_l2216_221671

theorem smallest_lcm_with_gcd_5 (p q : ℕ) : 
  1000 ≤ p ∧ p < 10000 ∧ 
  1000 ≤ q ∧ q < 10000 ∧ 
  Nat.gcd p q = 5 →
  201000 ≤ Nat.lcm p q ∧ 
  ∃ (p' q' : ℕ), 1000 ≤ p' ∧ p' < 10000 ∧ 
                 1000 ≤ q' ∧ q' < 10000 ∧ 
                 Nat.gcd p' q' = 5 ∧
                 Nat.lcm p' q' = 201000 := by
  sorry

end smallest_lcm_with_gcd_5_l2216_221671


namespace smallest_integer_square_75_more_than_double_l2216_221642

theorem smallest_integer_square_75_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 75 ∧ ∀ y : ℤ, y^2 = 2*y + 75 → x ≤ y :=
by sorry

end smallest_integer_square_75_more_than_double_l2216_221642


namespace correct_calculation_l2216_221688

theorem correct_calculation (x y : ℝ) : -4 * x * y + 3 * x * y = -x * y := by
  sorry

end correct_calculation_l2216_221688


namespace exists_distribution_prob_white_gt_two_thirds_l2216_221657

/-- Represents a distribution of balls in two boxes -/
structure BallDistribution :=
  (white_box1 : ℕ)
  (black_box1 : ℕ)
  (white_box2 : ℕ)
  (black_box2 : ℕ)

/-- The total number of white balls -/
def total_white : ℕ := 8

/-- The total number of black balls -/
def total_black : ℕ := 8

/-- Calculates the probability of drawing a white ball given a distribution -/
def prob_white (d : BallDistribution) : ℚ :=
  let p_box1 := (d.white_box1 : ℚ) / (d.white_box1 + d.black_box1 : ℚ)
  let p_box2 := (d.white_box2 : ℚ) / (d.white_box2 + d.black_box2 : ℚ)
  (1/2 : ℚ) * p_box1 + (1/2 : ℚ) * p_box2

/-- Theorem stating that there exists a distribution where the probability of drawing a white ball is greater than 2/3 -/
theorem exists_distribution_prob_white_gt_two_thirds :
  ∃ (d : BallDistribution),
    d.white_box1 + d.white_box2 = total_white ∧
    d.black_box1 + d.black_box2 = total_black ∧
    prob_white d > 2/3 := by
  sorry

end exists_distribution_prob_white_gt_two_thirds_l2216_221657


namespace decimal_multiplication_l2216_221655

theorem decimal_multiplication :
  (10 * 0.1 = 1) ∧ (10 * 0.01 = 0.1) ∧ (10 * 0.001 = 0.01) := by
  sorry

end decimal_multiplication_l2216_221655


namespace ice_cream_shop_problem_l2216_221623

/-- Ice cream shop problem -/
theorem ice_cream_shop_problem (total_revenue : ℕ) (cone_price : ℕ) (free_cones : ℕ) (n : ℕ) :
  total_revenue = 100 ∧ 
  cone_price = 2 ∧ 
  free_cones = 10 ∧
  (total_revenue / cone_price + free_cones) % n = 0 →
  n = 6 := by
sorry

end ice_cream_shop_problem_l2216_221623


namespace gcd_10_factorial_12_factorial_l2216_221697

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_10_factorial_12_factorial : Nat.gcd (factorial 10) (factorial 12) = factorial 10 := by
  sorry

end gcd_10_factorial_12_factorial_l2216_221697


namespace vovochka_candy_theorem_l2216_221684

/-- Represents the candy distribution problem --/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- Calculates the maximum number of candies that can be kept --/
def max_kept_candies (cd : CandyDistribution) : ℕ := 
  cd.total_candies - (cd.num_classmates * (cd.min_group_candies / cd.min_group_size))

/-- The theorem stating the maximum number of candies that can be kept --/
theorem vovochka_candy_theorem (cd : CandyDistribution) 
  (h1 : cd.total_candies = 200)
  (h2 : cd.num_classmates = 25)
  (h3 : cd.min_group_size = 16)
  (h4 : cd.min_group_candies = 100) :
  max_kept_candies cd = 37 := by
  sorry

#eval max_kept_candies { total_candies := 200, num_classmates := 25, min_group_size := 16, min_group_candies := 100 }

end vovochka_candy_theorem_l2216_221684


namespace function_property_l2216_221677

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x + f (1 - x) = 10)
  (h2 : ∃ a : ℝ, ∀ x : ℝ, f (1 + x) = a + f x)
  (h3 : ∀ x : ℝ, f x + f (-x) = 7) :
  ∃ a : ℝ, (∀ x : ℝ, f (1 + x) = a + f x) ∧ a = 3 := by
  sorry

end function_property_l2216_221677


namespace unique_residue_mod_16_l2216_221636

theorem unique_residue_mod_16 : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ -3125 [ZMOD 16] := by
  sorry

end unique_residue_mod_16_l2216_221636


namespace sum_of_digits_of_greatest_prime_divisor_l2216_221613

def n : ℕ := 4095

-- Define the greatest prime divisor of n
def greatest_prime_divisor (n : ℕ) : ℕ :=
  (Nat.factors n).foldl max 0

-- Define a function to sum the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.repr.foldl (λ sum c => sum + c.toNat - 48) 0

theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor n) = 4 := by
  sorry

end sum_of_digits_of_greatest_prime_divisor_l2216_221613


namespace marble_223_is_white_l2216_221679

def marble_color (n : ℕ) : String :=
  let cycle := n % 15
  if cycle < 6 then "gray"
  else if cycle < 11 then "white"
  else "black"

theorem marble_223_is_white :
  marble_color 223 = "white" := by
  sorry

end marble_223_is_white_l2216_221679


namespace age_problem_l2216_221608

theorem age_problem (x y : ℕ) (h1 : y = 3 * x) (h2 : x + y = 40) : x = 10 ∧ y = 30 := by
  sorry

end age_problem_l2216_221608


namespace car_fuel_efficiency_l2216_221639

/-- Given a car that uses 6.5 gallons of gasoline to travel 130 kilometers,
    prove that its fuel efficiency is 20 kilometers per gallon. -/
theorem car_fuel_efficiency :
  ∀ (distance : ℝ) (fuel : ℝ),
    distance = 130 →
    fuel = 6.5 →
    distance / fuel = 20 := by
  sorry

end car_fuel_efficiency_l2216_221639


namespace f_minimum_value_l2216_221648

noncomputable def f (x : ℝ) : ℝ := x + 1/x + 1/(x + 1/x) + 1/x^2

theorem f_minimum_value (x : ℝ) (hx : x > 0) : f x ≥ 3.5 ∧ f 1 = 3.5 := by
  sorry

end f_minimum_value_l2216_221648


namespace apple_problem_l2216_221644

theorem apple_problem (initial_apples : ℕ) (sold_to_jill : ℚ) (sold_to_june : ℚ) (sold_to_jeff : ℚ) (donated_to_school : ℚ) :
  initial_apples = 150 →
  sold_to_jill = 20 / 100 →
  sold_to_june = 30 / 100 →
  sold_to_jeff = 10 / 100 →
  donated_to_school = 5 / 100 →
  let remaining_after_jill := initial_apples - ⌊initial_apples * sold_to_jill⌋
  let remaining_after_june := remaining_after_jill - ⌊remaining_after_jill * sold_to_june⌋
  let remaining_after_jeff := remaining_after_june - ⌊remaining_after_june * sold_to_jeff⌋
  let final_remaining := remaining_after_jeff - ⌈remaining_after_jeff * donated_to_school⌉
  final_remaining = 72 := by
    sorry

end apple_problem_l2216_221644


namespace fraction_sum_theorem_l2216_221686

theorem fraction_sum_theorem (a b c d : ℝ) 
  (h1 : a/b + b/c + c/d + d/a = 6) 
  (h2 : a/c + b/d + c/a + d/b = 8) : 
  a/b + c/d = 2 ∨ a/b + c/d = 4 := by
sorry

end fraction_sum_theorem_l2216_221686


namespace simplify_and_ratio_l2216_221625

theorem simplify_and_ratio (m n : ℤ) : 
  let simplified := (5*m + 15*n + 20) / 5
  ∃ (a b c : ℤ), 
    simplified = a*m + b*n + c ∧ 
    (a + b) / c = 1 :=
by sorry

end simplify_and_ratio_l2216_221625


namespace total_players_l2216_221683

theorem total_players (outdoor : ℕ) (indoor : ℕ) (both : ℕ)
  (h1 : outdoor = 350)
  (h2 : indoor = 110)
  (h3 : both = 60) :
  outdoor + indoor - both = 400 := by
  sorry

end total_players_l2216_221683


namespace solution_is_83_l2216_221674

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the equation
def equation (x : ℝ) : Prop :=
  log 3 (x^2 - 3) + log 9 (x - 2) + log (1/3) (x^2 - 3) = 2

-- Theorem statement
theorem solution_is_83 :
  ∃ (x : ℝ), x > 0 ∧ equation x ∧ x = 83 :=
by sorry

end solution_is_83_l2216_221674


namespace final_result_proof_l2216_221601

theorem final_result_proof (chosen_number : ℕ) (h : chosen_number = 2976) :
  (chosen_number / 12) - 240 = 8 := by
  sorry

end final_result_proof_l2216_221601


namespace smallest_n_with_four_pairs_l2216_221664

/-- The function g(n) returns the number of distinct ordered pairs of positive integers (a, b) 
    such that a^2 + b^2 + ab = n -/
def g (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => 
  p.1^2 + p.2^2 + p.1 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 21 is the smallest positive integer n for which g(n) = 4 -/
theorem smallest_n_with_four_pairs : (∀ m < 21, g m ≠ 4) ∧ g 21 = 4 := by sorry

end smallest_n_with_four_pairs_l2216_221664


namespace range_of_m_l2216_221663

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) (h_ineq : ∃ m : ℝ, x + y/4 < m^2 - 3*m) :
  ∃ m : ℝ, m < -1 ∨ m > 4 :=
sorry

end range_of_m_l2216_221663


namespace number_multiplication_l2216_221681

theorem number_multiplication (x : ℤ) : 50 = x + 26 → 9 * x = 216 := by
  sorry

end number_multiplication_l2216_221681


namespace rotated_angle_measure_l2216_221637

/-- Given an initial angle of 60 degrees that is rotated 520 degrees clockwise,
    the resulting acute angle is 100 degrees. -/
theorem rotated_angle_measure (initial_angle rotation : ℝ) : 
  initial_angle = 60 →
  rotation = 520 →
  (initial_angle + rotation) % 360 = 100 := by
  sorry

end rotated_angle_measure_l2216_221637


namespace min_value_of_expression_l2216_221600

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > -1) (hab : a + b = 1) :
  1 / a + 1 / (b + 1) ≥ 2 := by
  sorry

end min_value_of_expression_l2216_221600


namespace game_win_fraction_l2216_221629

theorem game_win_fraction (total_matches : ℕ) (points_per_win : ℕ) (player1_points : ℕ) :
  total_matches = 8 →
  points_per_win = 10 →
  player1_points = 20 →
  (total_matches - player1_points / points_per_win) / total_matches = 3/4 := by
  sorry

end game_win_fraction_l2216_221629


namespace rex_cards_left_is_150_l2216_221690

/-- The number of Pokemon cards Rex has left after dividing his cards among himself and his siblings -/
def rexCardsLeft (nicolesCards : ℕ) : ℕ :=
  let cindysCards := 2 * nicolesCards
  let combinedTotal := nicolesCards + cindysCards
  let rexCards := combinedTotal / 2
  rexCards / 4

/-- Theorem stating that Rex has 150 cards left given the initial conditions -/
theorem rex_cards_left_is_150 : rexCardsLeft 400 = 150 := by
  sorry

end rex_cards_left_is_150_l2216_221690


namespace special_function_at_65_l2216_221626

/-- A function satisfying f(xy) = xf(y) for all real x and y, with f(1) = 40 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = x * f y) ∧ (f 1 = 40)

/-- Theorem: If f is a special function, then f(65) = 2600 -/
theorem special_function_at_65 (f : ℝ → ℝ) (h : special_function f) : f 65 = 2600 := by
  sorry

end special_function_at_65_l2216_221626


namespace liar_paradox_l2216_221635

/-- Represents the types of people in the land -/
inductive Person
  | Knight
  | Liar
  | Outsider

/-- Represents the statement "I am a liar" -/
def liarStatement : Prop := True

/-- A function that determines if a person tells the truth -/
def tellsTruth (p : Person) : Prop :=
  match p with
  | Person.Knight => True
  | Person.Liar => False
  | Person.Outsider => True

/-- A function that determines if a person's statement matches their nature -/
def statementMatches (p : Person) : Prop :=
  (p = Person.Liar) = tellsTruth p

theorem liar_paradox :
  ∀ p : Person, (p = Person.Knight ∨ p = Person.Liar) →
    (tellsTruth p = (p = Person.Liar)) → p = Person.Outsider := by
  sorry

end liar_paradox_l2216_221635


namespace absolute_value_equation_solution_product_l2216_221627

theorem absolute_value_equation_solution_product : 
  ∃ (x₁ x₂ : ℝ), 
    (|18 / x₁ + 4| = 3) ∧ 
    (|18 / x₂ + 4| = 3) ∧ 
    (x₁ ≠ x₂) ∧ 
    (x₁ * x₂ = 324 / 7) :=
by sorry

end absolute_value_equation_solution_product_l2216_221627


namespace inequality_relation_l2216_221612

theorem inequality_relation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) :=
by sorry

end inequality_relation_l2216_221612


namespace min_value_abs_sum_l2216_221646

theorem min_value_abs_sum (x : ℝ) : |x - 2| + |5 - x| ≥ 3 := by
  sorry

end min_value_abs_sum_l2216_221646


namespace reduced_price_is_34_2_l2216_221615

/-- Represents the price reduction percentage -/
def price_reduction : ℚ := 20 / 100

/-- Represents the additional amount of oil obtained after price reduction (in kg) -/
def additional_oil : ℚ := 4

/-- Represents the total cost in Rupees -/
def total_cost : ℚ := 684

/-- Calculates the reduced price per kg of oil -/
def reduced_price_per_kg (price_reduction : ℚ) (additional_oil : ℚ) (total_cost : ℚ) : ℚ :=
  total_cost / (total_cost / (total_cost / ((1 - price_reduction) * total_cost / total_cost)) + additional_oil)

/-- Theorem stating that the reduced price per kg of oil is 34.2 Rupees -/
theorem reduced_price_is_34_2 :
  reduced_price_per_kg price_reduction additional_oil total_cost = 34.2 := by
  sorry

end reduced_price_is_34_2_l2216_221615


namespace complex_fraction_simplification_l2216_221602

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 - 3 * Complex.I) = Complex.mk (-11/13) (29/13) := by
  sorry

end complex_fraction_simplification_l2216_221602


namespace number_count_in_average_calculation_l2216_221685

theorem number_count_in_average_calculation 
  (initial_average : ℚ)
  (correct_average : ℚ)
  (incorrect_number : ℚ)
  (correct_number : ℚ)
  (h1 : initial_average = 46)
  (h2 : correct_average = 50)
  (h3 : incorrect_number = 25)
  (h4 : correct_number = 65) :
  ∃ (n : ℕ), n > 0 ∧ 
    (n : ℚ) * correct_average = (n : ℚ) * initial_average + (correct_number - incorrect_number) ∧
    n = 10 := by
  sorry

end number_count_in_average_calculation_l2216_221685


namespace school_athletes_equation_l2216_221640

/-- 
Given a school with x athletes divided into y groups, prove that the following system of equations holds:
7y = x - 3
8y = x + 5
-/
theorem school_athletes_equation (x y : ℕ) 
  (h1 : 7 * y = x - 3)  -- If there are 7 people in each group, there will be 3 people left over
  (h2 : 8 * y = x + 5)  -- If there are 8 people in each group, there will be a shortage of 5 people
  : 7 * y = x - 3 ∧ 8 * y = x + 5 := by
  sorry

end school_athletes_equation_l2216_221640


namespace total_orchestra_members_l2216_221618

/-- Represents the number of boys in the orchestra -/
def boys : ℕ := sorry

/-- Represents the number of girls in the orchestra -/
def girls : ℕ := sorry

/-- The number of girls is twice the number of boys -/
axiom girls_twice_boys : girls = 2 * boys

/-- If 24 girls are transferred, the number of boys will be twice the number of girls -/
axiom boys_twice_remaining_girls : boys = 2 * (girls - 24)

/-- The total number of boys and girls in the orchestra is 48 -/
theorem total_orchestra_members : boys + girls = 48 := by sorry

end total_orchestra_members_l2216_221618


namespace line_properties_l2216_221693

-- Define the line
def line_equation (x : ℝ) : ℝ := -4 * x - 12

-- Theorem statement
theorem line_properties :
  (∀ x, line_equation x = -4 * x - 12) →
  (line_equation (-3) = 0) →
  (line_equation 0 = -12) ∧
  (line_equation 2 = -20) := by
  sorry

end line_properties_l2216_221693


namespace secret_spread_day_secret_spread_saturday_unique_day_for_3280_l2216_221694

def secret_spread (n : ℕ) : ℕ := (3^(n+1) - 1) / 2 + 1

theorem secret_spread_day : ∃ (d : ℕ), secret_spread d = 3280 :=
  sorry

theorem secret_spread_saturday : secret_spread 7 = 3280 :=
  sorry

theorem unique_day_for_3280 : ∀ (d : ℕ), secret_spread d = 3280 → d = 7 :=
  sorry

end secret_spread_day_secret_spread_saturday_unique_day_for_3280_l2216_221694


namespace line_equation_proof_l2216_221641

/-- 
Given a line mx + (n/2)y - 1 = 0 with a y-intercept of -1 and an angle of inclination 
twice that of the line √3x - y - 3√3 = 0, prove that m = -√3 and n = -2.
-/
theorem line_equation_proof (m n : ℝ) : 
  (∀ x y, m * x + (n / 2) * y - 1 = 0) →  -- Line equation
  (0 + (n / 2) * (-1) - 1 = 0) →  -- y-intercept is -1
  (Real.arctan m = 2 * Real.arctan (Real.sqrt 3)) →  -- Angle of inclination relation
  (m = -Real.sqrt 3 ∧ n = -2) := by
sorry

end line_equation_proof_l2216_221641
