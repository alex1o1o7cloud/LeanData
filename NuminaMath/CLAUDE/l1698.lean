import Mathlib

namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1698_169850

-- Define the equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x + 2 = 0
def equation2 (x : ℝ) : Prop := (x-3)^2 = 2*x - 6

-- Theorem for the first equation
theorem solutions_equation1 : 
  {x : ℝ | equation1 x} = {2 + Real.sqrt 2, 2 - Real.sqrt 2} :=
sorry

-- Theorem for the second equation
theorem solutions_equation2 :
  {x : ℝ | equation2 x} = {3, 5} :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1698_169850


namespace NUMINAMATH_CALUDE_b_investment_is_8000_l1698_169888

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Calculates b's investment in the partnership. -/
def calculate_b_investment (p : Partnership) : ℕ :=
  (22 * p.c_investment - 9 * (p.a_investment + p.c_investment)) / 9

/-- Theorem stating that given the conditions, b's investment is 8000. -/
theorem b_investment_is_8000 (p : Partnership)
  (h1 : p.a_investment = 5000)
  (h2 : p.c_investment = 9000)
  (h3 : p.total_profit = 88000)
  (h4 : p.c_profit = 36000)
  : calculate_b_investment p = 8000 := by
  sorry

#eval calculate_b_investment ⟨5000, 0, 9000, 88000, 36000⟩

end NUMINAMATH_CALUDE_b_investment_is_8000_l1698_169888


namespace NUMINAMATH_CALUDE_eighteen_times_two_minus_four_l1698_169827

theorem eighteen_times_two_minus_four (x : ℝ) : x * 2 = 18 → x - 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_times_two_minus_four_l1698_169827


namespace NUMINAMATH_CALUDE_team_b_four_wins_prob_l1698_169826

/-- Represents a team in the tournament -/
inductive Team
  | A
  | B
  | C

/-- The probability of one team beating another -/
def beat_prob (winner loser : Team) : ℝ :=
  match winner, loser with
  | Team.A, Team.B => 0.4
  | Team.B, Team.C => 0.5
  | Team.C, Team.A => 0.6
  | _, _ => 0 -- For other combinations, we set probability to 0

/-- The probability of Team B winning four consecutive matches -/
def team_b_four_wins : ℝ :=
  (1 - beat_prob Team.A Team.B) * (beat_prob Team.B Team.C) * 
  (1 - beat_prob Team.A Team.B) * (beat_prob Team.B Team.C)

theorem team_b_four_wins_prob : team_b_four_wins = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_team_b_four_wins_prob_l1698_169826


namespace NUMINAMATH_CALUDE_seventh_degree_equation_reduction_l1698_169890

theorem seventh_degree_equation_reduction (a b : ℝ) :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = x^7 - 7*a*x^5 + 14*a^2*x^3 - 7*a^3*x - b) →
    (∃ α β : ℝ, α * β = a ∧ α^7 + β^7 = b ∧ f α = 0 ∧ f β = 0) :=
by sorry

end NUMINAMATH_CALUDE_seventh_degree_equation_reduction_l1698_169890


namespace NUMINAMATH_CALUDE_solve_equation_l1698_169859

theorem solve_equation (x : ℝ) : 
  (1 : ℝ) / 7 + 7 / x = 15 / x + (1 : ℝ) / 15 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1698_169859


namespace NUMINAMATH_CALUDE_range_of_b_l1698_169829

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then x^2 - 2*a*x + 1 else -(x^2 - 2*a*x + 1)

-- State the theorem
theorem range_of_b (a : ℝ) (b : ℝ) :
  (a > 0) →
  (∀ x : ℝ, f a (x^3 + a) = -f a (-(x^3 + a))) →
  (∀ x : ℝ, x ∈ Set.Icc (b - 1) (b + 2) → f a (b * x) ≥ 4 * f a (x + 1)) →
  b ∈ Set.Iic (-Real.sqrt 5) ∪ Set.Ici ((3 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l1698_169829


namespace NUMINAMATH_CALUDE_factor_expression_l1698_169842

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) + 2 * (y - 4) = (3 * y + 7) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1698_169842


namespace NUMINAMATH_CALUDE_A_intersect_B_l1698_169812

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {y | ∃ x ∈ A, y = 2 * x - 1}

theorem A_intersect_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1698_169812


namespace NUMINAMATH_CALUDE_swim_club_members_swim_club_members_proof_l1698_169816

theorem swim_club_members : ℕ → Prop :=
  fun total_members =>
    let passed_test := (30 : ℚ) / 100 * total_members
    let not_passed := total_members - passed_test
    let prep_course := 12
    let no_prep_course := 30
    passed_test + not_passed = total_members ∧
    prep_course + no_prep_course = not_passed ∧
    total_members = 60

-- Proof
theorem swim_club_members_proof : ∃ n : ℕ, swim_club_members n :=
  sorry

end NUMINAMATH_CALUDE_swim_club_members_swim_club_members_proof_l1698_169816


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l1698_169879

/-- The coefficient of x^2 in the expansion of (2x^2 + 3x + 4)(5x^2 + 6x + 7) is 52 -/
theorem coefficient_x_squared (x : ℝ) : 
  (2*x^2 + 3*x + 4) * (5*x^2 + 6*x + 7) = 10*x^4 + 27*x^3 + 52*x^2 + 45*x + 28 := by
  sorry

#check coefficient_x_squared

end NUMINAMATH_CALUDE_coefficient_x_squared_l1698_169879


namespace NUMINAMATH_CALUDE_m_nonpositive_l1698_169801

theorem m_nonpositive (m : ℝ) (h : Real.sqrt (m^2) = -m) : m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_m_nonpositive_l1698_169801


namespace NUMINAMATH_CALUDE_updated_mean_example_l1698_169831

/-- The updated mean of a dataset after corrections -/
def updated_mean (original_mean original_count : ℕ) 
                 (decrement : ℕ) 
                 (missing_obs : List ℕ) 
                 (extra_obs : ℕ) : ℚ :=
  let original_sum := original_mean * original_count
  let corrected_sum := original_sum - decrement * original_count + missing_obs.sum - extra_obs
  let corrected_count := original_count - 1 + missing_obs.length
  (corrected_sum : ℚ) / corrected_count

/-- Theorem stating the updated mean after corrections -/
theorem updated_mean_example : 
  updated_mean 200 50 34 [150, 190, 210] 250 = 8600 / 52 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_example_l1698_169831


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_first_five_primes_l1698_169832

theorem smallest_four_digit_divisible_by_first_five_primes :
  ∃ n : ℕ,
    n ≥ 1000 ∧
    n < 10000 ∧
    2 ∣ n ∧
    3 ∣ n ∧
    5 ∣ n ∧
    7 ∣ n ∧
    11 ∣ n ∧
    (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ 2 ∣ m ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m ∧ 11 ∣ m → n ≤ m) ∧
    n = 2310 :=
by
  sorry

#eval 2310

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_first_five_primes_l1698_169832


namespace NUMINAMATH_CALUDE_atLeastOneMale_and_allFemales_mutuallyExclusive_l1698_169877

/-- Represents the outcome of selecting 2 students from the group -/
inductive Selection
| TwoMales
| OneMaleOneFemale
| TwoFemales

/-- The sample space of all possible selections -/
def sampleSpace : Set Selection :=
  {Selection.TwoMales, Selection.OneMaleOneFemale, Selection.TwoFemales}

/-- The event "At least 1 male student" -/
def atLeastOneMale : Set Selection :=
  {Selection.TwoMales, Selection.OneMaleOneFemale}

/-- The event "All female students" -/
def allFemales : Set Selection :=
  {Selection.TwoFemales}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (A B : Set Selection) : Prop :=
  A ∩ B = ∅

theorem atLeastOneMale_and_allFemales_mutuallyExclusive :
  mutuallyExclusive atLeastOneMale allFemales :=
sorry

end NUMINAMATH_CALUDE_atLeastOneMale_and_allFemales_mutuallyExclusive_l1698_169877


namespace NUMINAMATH_CALUDE_kids_difference_l1698_169884

/-- The number of kids Julia played with on Monday and Tuesday, and the difference between them. -/
def tag_game (monday tuesday : ℕ) : Prop :=
  monday = 16 ∧ tuesday = 4 ∧ monday - tuesday = 12

/-- Theorem stating the difference in the number of kids Julia played with. -/
theorem kids_difference : ∃ (monday tuesday : ℕ), tag_game monday tuesday :=
  sorry

end NUMINAMATH_CALUDE_kids_difference_l1698_169884


namespace NUMINAMATH_CALUDE_apple_sales_proof_l1698_169889

/-- The number of kilograms of apples sold in the first hour -/
def first_hour_sales : ℝ := 10

/-- The number of kilograms of apples sold in the second hour -/
def second_hour_sales : ℝ := 2

/-- The average number of kilograms of apples sold per hour over two hours -/
def average_sales : ℝ := 6

theorem apple_sales_proof :
  first_hour_sales = 10 :=
by
  have h1 : average_sales = (first_hour_sales + second_hour_sales) / 2 :=
    sorry
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_apple_sales_proof_l1698_169889


namespace NUMINAMATH_CALUDE_sector_area_l1698_169808

theorem sector_area (r : ℝ) (θ : ℝ) (h : r = 2) (h' : θ = π / 4) :
  (1 / 2) * r^2 * θ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1698_169808


namespace NUMINAMATH_CALUDE_max_value_theorem_l1698_169806

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  2 * x * y * Real.sqrt 6 + 8 * y * z^2 ≤ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1698_169806


namespace NUMINAMATH_CALUDE_jennys_change_l1698_169865

/-- The problem of calculating Jenny's change --/
theorem jennys_change 
  (cost_per_page : ℚ)
  (num_copies : ℕ)
  (pages_per_essay : ℕ)
  (num_pens : ℕ)
  (cost_per_pen : ℚ)
  (payment : ℚ)
  (h1 : cost_per_page = 1/10)
  (h2 : num_copies = 7)
  (h3 : pages_per_essay = 25)
  (h4 : num_pens = 7)
  (h5 : cost_per_pen = 3/2)
  (h6 : payment = 40) :
  payment - (cost_per_page * num_copies * pages_per_essay + cost_per_pen * num_pens) = 12 := by
  sorry


end NUMINAMATH_CALUDE_jennys_change_l1698_169865


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a11_l1698_169818

theorem arithmetic_sequence_a11 (a : ℕ → ℚ) 
  (h_arith : ∀ n, (a (n+4) + 1)⁻¹ = ((a n + 1)⁻¹ + (a (n+8) + 1)⁻¹) / 2)
  (h_a3 : a 3 = 2)
  (h_a7 : a 7 = 1) :
  a 11 = 1/2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a11_l1698_169818


namespace NUMINAMATH_CALUDE_smallest_possible_students_l1698_169866

theorem smallest_possible_students : ∃ (n : ℕ), 
  (5 * n + 2 > 50) ∧ 
  (∀ m : ℕ, m < n → 5 * m + 2 ≤ 50) ∧
  (5 * n + 2 = 52) := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_students_l1698_169866


namespace NUMINAMATH_CALUDE_converse_of_proposition_l1698_169867

theorem converse_of_proposition :
  (∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) →
  (∀ x : ℝ, -1 < x ∧ x < 1 → x^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_converse_of_proposition_l1698_169867


namespace NUMINAMATH_CALUDE_three_objects_five_containers_l1698_169869

/-- The number of ways to place n distinct objects into m distinct containers -/
def placement_count (n m : ℕ) : ℕ := m^n

/-- Theorem: Placing 3 distinct objects into 5 distinct containers results in 125 different arrangements -/
theorem three_objects_five_containers : placement_count 3 5 = 125 := by
  sorry

end NUMINAMATH_CALUDE_three_objects_five_containers_l1698_169869


namespace NUMINAMATH_CALUDE_combination_problem_classification_l1698_169894

-- Define a type for the scenarios
inductive Scenario
| sets_two_elements
| round_robin_tournament
| two_digit_number_formation
| two_digit_number_no_repeat

-- Define what it means for a scenario to be a combination problem
def is_combination_problem (s : Scenario) : Prop :=
  match s with
  | Scenario.sets_two_elements => True
  | Scenario.round_robin_tournament => True
  | Scenario.two_digit_number_formation => False
  | Scenario.two_digit_number_no_repeat => False

-- Theorem statement
theorem combination_problem_classification :
  (is_combination_problem Scenario.sets_two_elements) ∧
  (is_combination_problem Scenario.round_robin_tournament) ∧
  (¬ is_combination_problem Scenario.two_digit_number_formation) ∧
  (¬ is_combination_problem Scenario.two_digit_number_no_repeat) := by
  sorry


end NUMINAMATH_CALUDE_combination_problem_classification_l1698_169894


namespace NUMINAMATH_CALUDE_expected_balls_in_position_l1698_169855

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 5

/-- The probability that a specific ball is chosen for a swap -/
def prob_chosen : ℚ := 2 / 5

/-- The probability that a specific pair is chosen again -/
def prob_same_pair : ℚ := 1 / 5

/-- The probability that a specific ball is not involved in a swap -/
def prob_not_involved : ℚ := 3 / 5

/-- The number of independent transpositions -/
def num_transpositions : ℕ := 2

/-- 
Theorem: Given 5 balls arranged in a circle, with two independent random transpositions 
of adjacent balls, the expected number of balls in their original positions is 2.2.
-/
theorem expected_balls_in_position : 
  let prob_in_position := prob_chosen * prob_same_pair + prob_not_involved ^ num_transpositions
  num_balls * prob_in_position = 11/5 := by
  sorry

end NUMINAMATH_CALUDE_expected_balls_in_position_l1698_169855


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l1698_169876

/-- Calculates the total cost of plastering a rectangular tank. -/
def plastering_cost (length width depth : ℝ) (cost_per_sqm : ℝ) : ℝ :=
  let bottom_area := length * width
  let long_walls_area := 2 * (length * depth)
  let short_walls_area := 2 * (width * depth)
  let total_area := bottom_area + long_walls_area + short_walls_area
  total_area * cost_per_sqm

/-- Theorem stating the cost of plastering a specific tank. -/
theorem tank_plastering_cost :
  plastering_cost 60 25 10 0.9 = 2880 := by
  sorry

#eval plastering_cost 60 25 10 0.9

end NUMINAMATH_CALUDE_tank_plastering_cost_l1698_169876


namespace NUMINAMATH_CALUDE_quadratic_transform_sum_l1698_169813

/-- Given a quadratic equation 9x^2 - 54x - 81 = 0, when transformed into (x+q)^2 = p,
    the sum of q and p is 15 -/
theorem quadratic_transform_sum (q p : ℝ) : 
  (∀ x, 9*x^2 - 54*x - 81 = 0 ↔ (x + q)^2 = p) → q + p = 15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transform_sum_l1698_169813


namespace NUMINAMATH_CALUDE_candidate_count_l1698_169848

theorem candidate_count (total_selections : ℕ) (h : total_selections = 90) : 
  ∃ n : ℕ, n * (n - 1) = total_selections ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_candidate_count_l1698_169848


namespace NUMINAMATH_CALUDE_lcm_five_equals_lcm_three_l1698_169847

def is_subset_prime_factorization (a b : Nat) : Prop :=
  ∀ p : Nat, Prime p → (p^(a.factorization p) ∣ b)

theorem lcm_five_equals_lcm_three
  (a b c d e : Nat)
  (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)
  (h_lcm : Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d e))) = Nat.lcm a (Nat.lcm b c)) :
  (is_subset_prime_factorization d a ∨ is_subset_prime_factorization d b ∨ is_subset_prime_factorization d c) ∧
  (is_subset_prime_factorization e a ∨ is_subset_prime_factorization e b ∨ is_subset_prime_factorization e c) :=
sorry

end NUMINAMATH_CALUDE_lcm_five_equals_lcm_three_l1698_169847


namespace NUMINAMATH_CALUDE_jerry_butterflies_l1698_169893

/-- The number of butterflies Jerry originally had -/
def original_butterflies : ℕ := 93

/-- The number of butterflies Jerry let go -/
def butterflies_let_go : ℕ := 11

/-- The number of butterflies Jerry has left -/
def butterflies_left : ℕ := 82

/-- Theorem: Jerry originally had 93 butterflies -/
theorem jerry_butterflies : original_butterflies = butterflies_let_go + butterflies_left := by
  sorry

end NUMINAMATH_CALUDE_jerry_butterflies_l1698_169893


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1698_169823

/-- A rectangular field with width half its length and perimeter 54 meters has an area of 162 square meters. -/
theorem rectangular_field_area : ∀ w l : ℝ,
  w > 0 →
  l > 0 →
  w = l / 2 →
  2 * (w + l) = 54 →
  w * l = 162 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1698_169823


namespace NUMINAMATH_CALUDE_sum_xyz_l1698_169874

theorem sum_xyz (x y z : ℝ) 
  (eq1 : y + z = 20 - 4*x)
  (eq2 : x + z = 10 - 5*y)
  (eq3 : x + y = 15 - 2*z) :
  3*x + 3*y + 3*z = 22.5 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_l1698_169874


namespace NUMINAMATH_CALUDE_initial_boys_count_l1698_169815

theorem initial_boys_count (total : ℕ) : 
  let initial_boys := (60 * total) / 100
  let final_total := total + 2
  let final_boys := initial_boys - 3
  (2 * final_boys = final_total) → initial_boys = 24 :=
by sorry

end NUMINAMATH_CALUDE_initial_boys_count_l1698_169815


namespace NUMINAMATH_CALUDE_complex_product_real_l1698_169857

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 - 2 * I
  let z₂ : ℂ := 1 + a * I
  (z₁ * z₂).im = 0 → a = 2/3 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_l1698_169857


namespace NUMINAMATH_CALUDE_hike_length_l1698_169897

/-- Represents a four-day hike with given conditions -/
structure FourDayHike where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  first_two_days : day1 + day2 = 24
  second_third_avg : (day2 + day3) / 2 = 15
  last_two_days : day3 + day4 = 32
  first_third_days : day1 + day3 = 28

/-- The total length of the hike is 56 miles -/
theorem hike_length (h : FourDayHike) : h.day1 + h.day2 + h.day3 + h.day4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_hike_length_l1698_169897


namespace NUMINAMATH_CALUDE_min_odd_integers_l1698_169836

theorem min_odd_integers (a b c d e f : ℤ) : 
  a + b = 28 → 
  a + b + c + d = 45 → 
  a + b + c + d + e + f = 60 → 
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    (∀ x ∈ odds, Odd x) ∧ 
    odds.card = 2 ∧
    (∀ (other_odds : Finset ℤ), other_odds ⊆ {a, b, c, d, e, f} ∧ 
      (∀ x ∈ other_odds, Odd x) → 
      other_odds.card ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_odd_integers_l1698_169836


namespace NUMINAMATH_CALUDE_inequality_condition_l1698_169822

theorem inequality_condition (x : ℝ) : 
  (|x - 1| < 1 → x^2 - 5*x < 0) ∧ 
  ¬(∀ x : ℝ, x^2 - 5*x < 0 → |x - 1| < 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l1698_169822


namespace NUMINAMATH_CALUDE_n_minus_m_not_odd_l1698_169899

theorem n_minus_m_not_odd (n m : ℤ) (h : Even (n^2 - m^2)) : ¬Odd (n - m) := by
  sorry

end NUMINAMATH_CALUDE_n_minus_m_not_odd_l1698_169899


namespace NUMINAMATH_CALUDE_base_subtraction_equals_160_l1698_169802

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem base_subtraction_equals_160 :
  let base9_to_decimal := base_to_decimal [3, 2, 5] 9
  let base6_to_decimal := base_to_decimal [2, 5, 4] 6
  base9_to_decimal - base6_to_decimal = 160 := by
sorry

end NUMINAMATH_CALUDE_base_subtraction_equals_160_l1698_169802


namespace NUMINAMATH_CALUDE_fish_caught_fisherman_catch_l1698_169811

theorem fish_caught (fish_per_line : ℕ) (initial_lines : ℕ) (broken_lines : ℕ) : ℕ :=
  let usable_lines : ℕ := initial_lines - broken_lines
  usable_lines * fish_per_line

theorem fisherman_catch : fish_caught 3 226 3 = 669 := by
  sorry

end NUMINAMATH_CALUDE_fish_caught_fisherman_catch_l1698_169811


namespace NUMINAMATH_CALUDE_composite_number_impossibility_l1698_169895

theorem composite_number_impossibility (n a q : ℕ) (h_n : n > 1) (h_q_prime : Nat.Prime q) 
  (h_q_div : q ∣ (n - 1)) (h_q_sqrt : q > Nat.sqrt n - 1) (h_n_div : n ∣ (a^(n-1) - 1)) 
  (h_gcd : Nat.gcd (a^((n-1)/q) - 1) n = 1) : 
  Nat.Prime n := by
sorry

end NUMINAMATH_CALUDE_composite_number_impossibility_l1698_169895


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l1698_169800

/-- A rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- A hexagon inscribed in a rectangle, with vertices touching midpoints of the rectangle's edges -/
structure InscribedHexagon (r : Rectangle) where

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The area of an inscribed hexagon -/
def InscribedHexagon.area (h : InscribedHexagon r) : ℝ := sorry

theorem inscribed_hexagon_area (r : Rectangle) (h : InscribedHexagon r) 
    (h_width : r.width = 5) (h_height : r.height = 4) : 
    InscribedHexagon.area h = 10 := by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l1698_169800


namespace NUMINAMATH_CALUDE_parabola_focus_construction_l1698_169872

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola -/
structure Parabola where
  focus : Point
  directrix : Line

def reflect_line (l : Line) (t : Line) : Line :=
  sorry

def intersection_point (l1 : Line) (l2 : Line) : Point :=
  sorry

def is_tangent (p : Parabola) (t : Line) : Prop :=
  sorry

theorem parabola_focus_construction 
  (p : Parabola) (t1 t2 : Line) 
  (h1 : is_tangent p t1) 
  (h2 : is_tangent p t2) :
  p.focus = intersection_point 
    (reflect_line p.directrix t1) 
    (reflect_line p.directrix t2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_construction_l1698_169872


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1698_169846

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (ax^2 - 1/x)^6
def coefficient (a : ℝ) : ℝ := -a^3 * binomial 6 3

-- Theorem statement
theorem expansion_coefficient (a : ℝ) : coefficient a = 160 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1698_169846


namespace NUMINAMATH_CALUDE_lindas_savings_l1698_169843

theorem lindas_savings (savings : ℝ) 
  (h1 : savings * (1/4) = 200)
  (h2 : ∃ (furniture_cost : ℝ), furniture_cost = savings * (3/4) ∧ 
        furniture_cost * 0.8 = savings * (3/4))
  : savings = 800 := by
sorry

end NUMINAMATH_CALUDE_lindas_savings_l1698_169843


namespace NUMINAMATH_CALUDE_tan_x_plus_pi_fourth_l1698_169853

theorem tan_x_plus_pi_fourth (x : ℝ) (h : Real.tan x = 2) : 
  Real.tan (x + π / 4) = -3 := by sorry

end NUMINAMATH_CALUDE_tan_x_plus_pi_fourth_l1698_169853


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l1698_169804

theorem polynomial_root_sum (a b : ℝ) : 
  (Complex.I * Real.sqrt 3 + 1 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 3 + 1) + b = 0 ∧ 
  (-3 : ℂ) ^ 3 + a * (-3) + b = 0 → 
  a + b = 11 := by sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l1698_169804


namespace NUMINAMATH_CALUDE_division_problem_l1698_169803

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 122 →
  quotient = 6 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  divisor = 20 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1698_169803


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1698_169892

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  d_nonzero : d ≠ 0
  a_1_eq_2 : a 1 = 2
  geometric : a 1 * a 4 = a 2 * a 2  -- a_1, a_2, a_4 form a geometric sequence

/-- The theorem stating the general term of the sequence -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l1698_169892


namespace NUMINAMATH_CALUDE_sheet_area_difference_l1698_169819

/-- The difference in combined area of front and back between two rectangular sheets of paper -/
theorem sheet_area_difference (l1 w1 l2 w2 : ℕ) : 
  l1 = 14 ∧ w1 = 12 ∧ l2 = 9 ∧ w2 = 14 → 2 * (l1 * w1) - 2 * (l2 * w2) = 84 := by
  sorry

#check sheet_area_difference

end NUMINAMATH_CALUDE_sheet_area_difference_l1698_169819


namespace NUMINAMATH_CALUDE_whole_number_between_l1698_169858

theorem whole_number_between : 
  ∀ N : ℤ, (9 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 10) → (N = 37 ∨ N = 38 ∨ N = 39) :=
by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_l1698_169858


namespace NUMINAMATH_CALUDE_power_of_power_l1698_169851

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l1698_169851


namespace NUMINAMATH_CALUDE_area_of_inscribed_circle_rectangle_l1698_169860

/-- A rectangle with an inscribed circle -/
structure InscribedCircleRectangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The width of the rectangle -/
  w : ℝ
  /-- The height of the rectangle -/
  h : ℝ
  /-- The circle is tangent to all sides -/
  tangent_to_sides : w = h
  /-- The circle passes through the midpoint of a diagonal -/
  passes_through_midpoint : w^2 / 4 + h^2 / 4 = r^2

/-- The area of a rectangle with an inscribed circle passing through the midpoint of a diagonal is 2r^2 -/
theorem area_of_inscribed_circle_rectangle (rect : InscribedCircleRectangle) : 
  rect.w * rect.h = 2 * rect.r^2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribed_circle_rectangle_l1698_169860


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_denominator_factorization_l1698_169862

theorem partial_fraction_decomposition (x : ℝ) : 
  let A : ℝ := 1/2
  let B : ℝ := 9/2
  (6*x - 7) / (3*x^2 + 2*x - 8) = A / (x - 2) + B / (3*x + 4) :=
by
  sorry

-- Auxiliary theorem to establish the factorization of the denominator
theorem denominator_factorization (x : ℝ) :
  3*x^2 + 2*x - 8 = (3*x + 4)*(x - 2) :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_denominator_factorization_l1698_169862


namespace NUMINAMATH_CALUDE_roadwork_pitch_calculation_l1698_169838

/-- Calculates the number of barrels of pitch needed to pave the remaining road -/
def barrels_of_pitch_needed (total_road_length : ℕ) (truckloads_per_mile : ℕ) (gravel_bags_per_truckload : ℕ) (gravel_to_pitch_ratio : ℕ) (paved_miles : ℕ) : ℕ :=
  let remaining_miles := total_road_length - paved_miles
  let total_truckloads := remaining_miles * truckloads_per_mile
  let total_gravel_bags := total_truckloads * gravel_bags_per_truckload
  total_gravel_bags / gravel_to_pitch_ratio

theorem roadwork_pitch_calculation :
  barrels_of_pitch_needed 16 3 2 5 11 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roadwork_pitch_calculation_l1698_169838


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l1698_169898

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, Real.tan x = 1) ↔ (∀ x : ℝ, Real.tan x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l1698_169898


namespace NUMINAMATH_CALUDE_solve_slurpee_problem_l1698_169839

def slurpee_problem (money_given : ℝ) (change_received : ℝ) (num_slurpees : ℕ) : Prop :=
  let total_spent := money_given - change_received
  let cost_per_slurpee := total_spent / num_slurpees
  cost_per_slurpee = 2

theorem solve_slurpee_problem :
  slurpee_problem 20 8 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_slurpee_problem_l1698_169839


namespace NUMINAMATH_CALUDE_john_earnings_l1698_169883

/-- Calculates the money earned by John for repairing cars -/
def money_earned (total_cars : ℕ) (standard_cars : ℕ) (standard_time : ℕ) (hourly_rate : ℕ) : ℕ :=
  let remaining_cars := total_cars - standard_cars
  let standard_total_time := standard_cars * standard_time
  let remaining_time := remaining_cars * (standard_time + standard_time / 2)
  let total_time := standard_total_time + remaining_time
  let total_hours := (total_time + 59) / 60  -- Ceiling division
  hourly_rate * total_hours

/-- Theorem stating that John earns $80 for repairing the cars -/
theorem john_earnings : money_earned 5 3 40 20 = 80 := by
  sorry

end NUMINAMATH_CALUDE_john_earnings_l1698_169883


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l1698_169814

theorem exponential_equation_solution :
  ∃ x : ℝ, 3^(3*x + 2) = (1:ℝ)/81 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l1698_169814


namespace NUMINAMATH_CALUDE_tom_initial_investment_l1698_169875

/-- Represents the initial investment of Tom in rupees -/
def tom_investment : ℝ := 3000

/-- Represents Jose's investment in rupees -/
def jose_investment : ℝ := 4500

/-- Represents the total duration of the business in months -/
def total_duration : ℝ := 12

/-- Represents the time after which Jose joined in months -/
def jose_join_time : ℝ := 2

/-- Represents the total profit in rupees -/
def total_profit : ℝ := 6300

/-- Represents Jose's share of the profit in rupees -/
def jose_profit : ℝ := 3500

theorem tom_initial_investment :
  tom_investment * total_duration / (jose_investment * (total_duration - jose_join_time)) =
  (total_profit - jose_profit) / jose_profit :=
sorry

end NUMINAMATH_CALUDE_tom_initial_investment_l1698_169875


namespace NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l1698_169845

/-- Given a point (1, 2, 3) in a three-dimensional Cartesian coordinate system,
    its symmetric point with respect to the xoy plane is (1, 2, -3). -/
theorem symmetric_point_xoy_plane :
  let original_point : ℝ × ℝ × ℝ := (1, 2, 3)
  let xoy_plane : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}
  let symmetric_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, p.2.1, -p.2.2)
  symmetric_point original_point = (1, 2, -3) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_point_xoy_plane_l1698_169845


namespace NUMINAMATH_CALUDE_candy_distribution_count_l1698_169809

/-- The number of ways to partition n identical objects into at most k parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 30 ways to partition 10 identical objects into at most 5 parts -/
theorem candy_distribution_count : partition_count 10 5 = 30 := by sorry

end NUMINAMATH_CALUDE_candy_distribution_count_l1698_169809


namespace NUMINAMATH_CALUDE_knowledge_competition_probability_l1698_169844

/-- The probability of correctly answering a single question -/
def p_correct : ℝ := 0.8

/-- The number of preset questions in the competition -/
def total_questions : ℕ := 5

/-- The probability of answering exactly 4 questions before advancing -/
def prob_four_questions : ℝ := p_correct * p_correct * (1 - p_correct) * p_correct

theorem knowledge_competition_probability :
  prob_four_questions = 0.128 :=
sorry

end NUMINAMATH_CALUDE_knowledge_competition_probability_l1698_169844


namespace NUMINAMATH_CALUDE_cube_edge_length_l1698_169861

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 24) :
  ∃ edge_length : ℝ, edge_length > 0 ∧ 6 * edge_length^2 = surface_area ∧ edge_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1698_169861


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1698_169830

theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (selected_pens : ℕ) 
  (h1 : total_pens = 12) 
  (h2 : defective_pens = 4) 
  (h3 : selected_pens = 2) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 14/33 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1698_169830


namespace NUMINAMATH_CALUDE_right_triangles_on_circle_l1698_169896

/-- The number of right-angled triangles formed by 2n equally spaced points on a circle -/
theorem right_triangles_on_circle (n : ℕ) (h : n > 1) :
  (number_of_right_triangles : ℕ) = 2 * n * (n - 1) :=
by sorry

end NUMINAMATH_CALUDE_right_triangles_on_circle_l1698_169896


namespace NUMINAMATH_CALUDE_borrowed_sheets_theorem_l1698_169824

/-- Represents a collection of lecture notes --/
structure LectureNotes where
  total_pages : Nat
  total_sheets : Nat
  borrowed_sheets : Nat

/-- Calculates the average of remaining page numbers after some sheets are borrowed --/
def averageRemainingPages (notes : LectureNotes) : Rat :=
  let remaining_sheets := notes.total_sheets - notes.borrowed_sheets
  let first_remaining_page := 2 * notes.borrowed_sheets + 1
  let last_remaining_page := notes.total_pages
  ((first_remaining_page + last_remaining_page) * remaining_sheets) / (2 * remaining_sheets)

/-- The theorem to be proved --/
theorem borrowed_sheets_theorem (notes : LectureNotes) :
  notes.total_pages = 72 ∧ 
  notes.total_sheets = 36 ∧ 
  notes.borrowed_sheets = 17 →
  averageRemainingPages notes = 40 := by
  sorry

#eval averageRemainingPages { total_pages := 72, total_sheets := 36, borrowed_sheets := 17 }

end NUMINAMATH_CALUDE_borrowed_sheets_theorem_l1698_169824


namespace NUMINAMATH_CALUDE_sequence_properties_l1698_169805

theorem sequence_properties :
  (∃ a : ℕ → ℕ, a 1 = 2 ∧ (∀ n, a (n + 1) = a n + n + 1) ∧ a 20 = 211) ∧
  (∃ b : ℕ → ℕ, b 1 = 1 ∧ (∀ n, b (n + 1) = 3 * b n + 2) ∧ b 4 = 53) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1698_169805


namespace NUMINAMATH_CALUDE_average_side_length_of_squares_l1698_169863

theorem average_side_length_of_squares (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 225) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 28 / 3 := by
sorry

end NUMINAMATH_CALUDE_average_side_length_of_squares_l1698_169863


namespace NUMINAMATH_CALUDE_airplane_distance_difference_l1698_169885

/-- Theorem: Distance difference for an airplane flying with and against wind -/
theorem airplane_distance_difference (a : ℝ) : 
  let windless_speed : ℝ := a
  let wind_speed : ℝ := 20
  let time_without_wind : ℝ := 4
  let time_against_wind : ℝ := 3
  windless_speed * time_without_wind - (windless_speed - wind_speed) * time_against_wind = a + 60 := by
  sorry

end NUMINAMATH_CALUDE_airplane_distance_difference_l1698_169885


namespace NUMINAMATH_CALUDE_walking_probability_is_four_sevenths_l1698_169868

/-- The number of bus stops -/
def num_stops : ℕ := 15

/-- The distance between adjacent stops in feet -/
def distance_between_stops : ℕ := 100

/-- The maximum walking distance in feet -/
def max_walking_distance : ℕ := 500

/-- The probability of walking 500 feet or less between two randomly chosen stops -/
def walking_probability : ℚ :=
  let total_possibilities := num_stops * (num_stops - 1)
  let favorable_outcomes := 120  -- This is derived from the problem, not the solution
  favorable_outcomes / total_possibilities

theorem walking_probability_is_four_sevenths :
  walking_probability = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_walking_probability_is_four_sevenths_l1698_169868


namespace NUMINAMATH_CALUDE_eight_custom_op_eight_eq_four_l1698_169856

/-- Custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a * b) / (a + b)

/-- Theorem stating that 8 @ 8 = 4 -/
theorem eight_custom_op_eight_eq_four :
  custom_op 8 8 = 4 := by sorry

end NUMINAMATH_CALUDE_eight_custom_op_eight_eq_four_l1698_169856


namespace NUMINAMATH_CALUDE_unique_n_existence_and_value_l1698_169840

theorem unique_n_existence_and_value : ∃! n : ℤ,
  50 ≤ n ∧ n ≤ 150 ∧
  n % 7 = 0 ∧
  n % 9 = 3 ∧
  n % 6 = 3 ∧
  n = 75 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_existence_and_value_l1698_169840


namespace NUMINAMATH_CALUDE_intersection_empty_union_real_l1698_169886

-- Define sets A and B
def A (a : ℝ) := {x : ℝ | 2*a ≤ x ∧ x ≤ a + 3}
def B := {x : ℝ | x < -1 ∨ x > 1}

-- Theorem for part I
theorem intersection_empty (a : ℝ) : A a ∩ B = ∅ ↔ a > 3 := by sorry

-- Theorem for part II
theorem union_real (a : ℝ) : A a ∪ B = Set.univ ↔ -2 ≤ a ∧ a ≤ -1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_union_real_l1698_169886


namespace NUMINAMATH_CALUDE_prob_certain_event_prob_union_l1698_169810

-- Define a probability space
variable (Ω : Type*) [MeasurableSpace Ω] (P : Measure Ω)

-- Define a certain event
def certain_event : Set Ω := Set.univ

-- Define the probability of an event
def prob (A : Set Ω) : ℝ := P A

-- Theorem 1: The probability of a certain event is 1
theorem prob_certain_event :
  prob P certain_event = 1 := by sorry

-- Theorem 2: Probability of union of two events
theorem prob_union (A B : Set Ω) :
  prob P (A ∪ B) = prob P A + prob P B - prob P (A ∩ B) := by sorry

end NUMINAMATH_CALUDE_prob_certain_event_prob_union_l1698_169810


namespace NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1698_169807

theorem arithmetic_progression_first_term
  (a : ℕ → ℝ)
  (h_increasing : ∀ n, a n < a (n + 1))
  (h_arithmetic : ∃ d, ∀ n, a (n + 1) - a n = d)
  (h_sum : a 0 + a 1 + a 2 = 12)
  (h_product : a 0 * a 1 * a 2 = 48) :
  a 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_first_term_l1698_169807


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1698_169880

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_first : a 1 = 1024)
  (h_fifth : a 5 = 128) :
  a 3 = 256 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1698_169880


namespace NUMINAMATH_CALUDE_birth_year_problem_l1698_169849

theorem birth_year_problem (x : ℕ) : 
  (1800 ≤ x^2 - x) ∧ (x^2 - x < 1850) ∧ (x^2 = x + 1806) → x^2 - x = 1806 :=
by sorry

end NUMINAMATH_CALUDE_birth_year_problem_l1698_169849


namespace NUMINAMATH_CALUDE_student_tape_cost_problem_l1698_169820

theorem student_tape_cost_problem :
  ∃ (n : ℕ) (x : ℕ) (price : ℕ),
    Even n ∧
    10 < n ∧ n < 20 ∧
    100 ≤ price ∧ price ≤ 120 ∧
    n * x = price ∧
    (n - 2) * (x + 1) = price ∧
    n = 14 := by
  sorry

end NUMINAMATH_CALUDE_student_tape_cost_problem_l1698_169820


namespace NUMINAMATH_CALUDE_base_10_to_base_5_512_l1698_169878

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

theorem base_10_to_base_5_512 :
  toBase5 512 = [4, 0, 2, 2] :=
sorry

end NUMINAMATH_CALUDE_base_10_to_base_5_512_l1698_169878


namespace NUMINAMATH_CALUDE_triangle_segment_length_l1698_169841

structure Triangle :=
  (A B C : ℝ × ℝ)

def angleBisector (t : Triangle) (D : ℝ × ℝ) : Prop :=
  sorry  -- Definition of angle bisector

theorem triangle_segment_length 
  (ABC : Triangle) 
  (D : ℝ × ℝ) 
  (h_bisector : angleBisector ABC D)
  (h_AD : dist D ABC.A = 15)
  (h_DC : dist D ABC.C = 45)
  (h_DB : dist D ABC.B = 24) :
  dist ABC.A ABC.B = 39 :=
sorry

#check triangle_segment_length

end NUMINAMATH_CALUDE_triangle_segment_length_l1698_169841


namespace NUMINAMATH_CALUDE_bucket_leak_problem_l1698_169854

/-- Converts gallons to quarts -/
def gallons_to_quarts (g : ℝ) : ℝ := 4 * g

/-- Calculates the amount of water leaked given initial and remaining amounts -/
def water_leaked (initial : ℝ) (remaining : ℝ) : ℝ := initial - remaining

theorem bucket_leak_problem (initial : ℝ) (remaining_gallons : ℝ) 
  (h1 : initial = 4) 
  (h2 : remaining_gallons = 0.33) : 
  water_leaked initial (gallons_to_quarts remaining_gallons) = 2.68 := by
  sorry

#eval water_leaked 4 (gallons_to_quarts 0.33)

end NUMINAMATH_CALUDE_bucket_leak_problem_l1698_169854


namespace NUMINAMATH_CALUDE_trapezium_side_length_l1698_169870

/-- Given a trapezium with area 342 cm², one parallel side of 14 cm, and height 18 cm,
    prove that the length of the other parallel side is 24 cm. -/
theorem trapezium_side_length (area : ℝ) (side1 : ℝ) (height : ℝ) (side2 : ℝ) :
  area = 342 →
  side1 = 14 →
  height = 18 →
  area = (1 / 2) * (side1 + side2) * height →
  side2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l1698_169870


namespace NUMINAMATH_CALUDE_unique_solution_l1698_169825

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 12 ∧ y = 10 ∧ z = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1698_169825


namespace NUMINAMATH_CALUDE_points_symmetric_about_x_axis_l1698_169817

/-- Two points are symmetric about the x-axis if they have the same x-coordinate
    and their y-coordinates are negatives of each other. -/
def symmetric_about_x_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 = p2.1 ∧ p1.2 = -p2.2

/-- Given points P₁(-4, 3) and P₂(-4, -3), prove they are symmetric about the x-axis. -/
theorem points_symmetric_about_x_axis :
  let p1 : ℝ × ℝ := (-4, 3)
  let p2 : ℝ × ℝ := (-4, -3)
  symmetric_about_x_axis p1 p2 := by
  sorry


end NUMINAMATH_CALUDE_points_symmetric_about_x_axis_l1698_169817


namespace NUMINAMATH_CALUDE_power_sum_difference_l1698_169871

theorem power_sum_difference : 2^6 + 2^6 + 2^6 + 2^6 - 4^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l1698_169871


namespace NUMINAMATH_CALUDE_antenna_tower_height_l1698_169852

/-- Given an antenna tower on flat terrain, if the sum of the angles of elevation
    measured at distances of 100 m, 200 m, and 300 m from its base is 90°,
    then the height of the tower is 100 m. -/
theorem antenna_tower_height (α β γ : Real) (h : Real) :
  (α + β + γ = Real.pi / 2) →
  (h / 100 = Real.tan α) →
  (h / 200 = Real.tan β) →
  (h / 300 = Real.tan γ) →
  h = 100 := by
  sorry

#check antenna_tower_height

end NUMINAMATH_CALUDE_antenna_tower_height_l1698_169852


namespace NUMINAMATH_CALUDE_intersection_point_is_minus_one_minus_one_l1698_169887

-- Define the two line equations
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 7 = 0
def line2 (x y : ℝ) : Prop := x - 2 * y - 1 = 0

-- Theorem stating that (-1, -1) is the unique intersection point
theorem intersection_point_is_minus_one_minus_one :
  ∃! (x y : ℝ), line1 x y ∧ line2 x y ∧ x = -1 ∧ y = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_is_minus_one_minus_one_l1698_169887


namespace NUMINAMATH_CALUDE_pictures_deleted_l1698_169834

theorem pictures_deleted (zoo_pics : ℕ) (museum_pics : ℕ) (pics_left : ℕ) : 
  zoo_pics = 50 → museum_pics = 8 → pics_left = 20 → 
  zoo_pics + museum_pics - pics_left = 38 := by
sorry

end NUMINAMATH_CALUDE_pictures_deleted_l1698_169834


namespace NUMINAMATH_CALUDE_sum_product_theorem_l1698_169881

theorem sum_product_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 1)
  (eq3 : a + c + d = 12)
  (eq4 : b + c + d = 7) :
  a * b + c * d = 176 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_product_theorem_l1698_169881


namespace NUMINAMATH_CALUDE_product_of_consecutive_integers_l1698_169882

theorem product_of_consecutive_integers (n : ℕ) : 
  n = 5 → (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) = 30240 := by
  sorry

end NUMINAMATH_CALUDE_product_of_consecutive_integers_l1698_169882


namespace NUMINAMATH_CALUDE_polynomial_inequality_l1698_169835

theorem polynomial_inequality (x : ℝ) : x^4 + x^3 - 10*x^2 > -25*x ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l1698_169835


namespace NUMINAMATH_CALUDE_star_18_6_l1698_169891

/-- The star operation defined for integers -/
def star (a b : ℤ) : ℚ := a - a / b

/-- Theorem stating that 18 ★ 6 = 15 -/
theorem star_18_6 : star 18 6 = 15 := by sorry

end NUMINAMATH_CALUDE_star_18_6_l1698_169891


namespace NUMINAMATH_CALUDE_division_base4_correct_l1698_169821

/-- Converts a number from base 4 to base 10 --/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a number from base 10 to base 4 --/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Performs division in base 4 --/
def divBase4 (a b : List Nat) : (List Nat × List Nat) :=
  let a10 := base4ToBase10 a
  let b10 := base4ToBase10 b
  let q := a10 / b10
  let r := a10 % b10
  (base10ToBase4 q, base10ToBase4 r)

theorem division_base4_correct (a b : List Nat) :
  a = [2, 3, 0, 2] ∧ b = [2, 1] →
  divBase4 a b = ([3, 1, 1], [0, 1]) := by
  sorry

end NUMINAMATH_CALUDE_division_base4_correct_l1698_169821


namespace NUMINAMATH_CALUDE_typists_letters_time_relation_typists_letters_theorem_l1698_169873

/-- The number of letters a single typist can type in one minute -/
def typing_rate (typists : ℕ) (letters : ℕ) (minutes : ℕ) : ℚ :=
  (letters : ℚ) / (typists * minutes)

/-- The theorem stating the relationship between typists, letters, and time -/
theorem typists_letters_time_relation 
  (initial_typists : ℕ) (initial_letters : ℕ) (initial_minutes : ℕ)
  (final_typists : ℕ) (final_minutes : ℕ) :
  initial_typists > 0 → initial_minutes > 0 → final_typists > 0 → final_minutes > 0 →
  (typing_rate initial_typists initial_letters initial_minutes) * 
    (final_typists * final_minutes) = 
  (final_typists * final_minutes * initial_letters : ℚ) / (initial_typists * initial_minutes) :=
by sorry

/-- The main theorem to prove -/
theorem typists_letters_theorem :
  typing_rate 20 42 20 * (30 * 60) = 189 :=
by sorry

end NUMINAMATH_CALUDE_typists_letters_time_relation_typists_letters_theorem_l1698_169873


namespace NUMINAMATH_CALUDE_train_length_calculation_l1698_169828

/-- Represents the properties of a train and its movement --/
structure Train where
  length : ℝ
  speed : ℝ
  platform_crossing_time : ℝ
  pole_crossing_time : ℝ
  platform_length : ℝ

/-- Theorem stating the length of the train given specific conditions --/
theorem train_length_calculation (t : Train)
  (h1 : t.platform_crossing_time = 39)
  (h2 : t.pole_crossing_time = 16)
  (h3 : t.platform_length = 431.25)
  (h4 : t.length = t.speed * t.pole_crossing_time)
  (h5 : t.length + t.platform_length = t.speed * t.platform_crossing_time) :
  t.length = 6890 / 23 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l1698_169828


namespace NUMINAMATH_CALUDE_dog_age_difference_l1698_169837

theorem dog_age_difference (
  avg_age_1_5 : ℝ)
  (age_1 : ℝ)
  (age_2 : ℝ)
  (age_3 : ℝ)
  (age_4 : ℝ)
  (age_5 : ℝ)
  (h1 : avg_age_1_5 = 18)
  (h2 : age_1 = 10)
  (h3 : age_2 = age_1 - 2)
  (h4 : age_3 = age_2 + 4)
  (h5 : age_4 = age_3 / 2)
  (h6 : age_5 = age_4 + 20)
  (h7 : avg_age_1_5 = (age_1 + age_5) / 2) :
  age_3 - age_2 = 4 := by
sorry

end NUMINAMATH_CALUDE_dog_age_difference_l1698_169837


namespace NUMINAMATH_CALUDE_unique_solution_l1698_169833

def complex_number (a : ℝ) : ℂ := Complex.mk (a^2 - 2) (3*a - 4)

theorem unique_solution :
  ∃! a : ℝ,
    (complex_number a).re = (complex_number a).im ∧
    (complex_number a).re < 0 ∧
    (complex_number a).im < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1698_169833


namespace NUMINAMATH_CALUDE_sum_product_bounds_l1698_169864

theorem sum_product_bounds (a b c : ℝ) (h : a + b + c = 3) :
  ∃ (lower_bound upper_bound : ℝ),
    lower_bound = -9/2 ∧
    upper_bound = 3 ∧
    (∀ ε > 0, ∃ (x y z : ℝ), x + y + z = 3 ∧ x*y + x*z + y*z < lower_bound + ε) ∧
    (∀ (x y z : ℝ), x + y + z = 3 → x*y + x*z + y*z ≤ upper_bound) :=
by sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l1698_169864
