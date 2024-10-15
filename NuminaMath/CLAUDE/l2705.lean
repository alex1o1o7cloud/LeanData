import Mathlib

namespace NUMINAMATH_CALUDE_orange_juice_distribution_l2705_270599

theorem orange_juice_distribution (pitcher_capacity : ℝ) (h : pitcher_capacity > 0) :
  let juice_volume := (2/3) * pitcher_capacity
  let num_cups := 8
  let juice_per_cup := juice_volume / num_cups
  juice_per_cup / pitcher_capacity = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_distribution_l2705_270599


namespace NUMINAMATH_CALUDE_lyle_friends_served_l2705_270580

/-- Calculates the maximum number of friends who can have a sandwich and a juice pack. -/
def max_friends_served (sandwich_cost juice_cost total_money : ℚ) : ℕ :=
  let cost_per_person := sandwich_cost + juice_cost
  let total_servings := (total_money / cost_per_person).floor
  (total_servings - 1).natAbs

/-- Proves that Lyle can buy a sandwich and a juice pack for 4 friends. -/
theorem lyle_friends_served :
  max_friends_served 0.30 0.20 2.50 = 4 := by
  sorry

#eval max_friends_served 0.30 0.20 2.50

end NUMINAMATH_CALUDE_lyle_friends_served_l2705_270580


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2705_270542

/-- Arithmetic sequence a_n with a₁ = 8 and a₃ = 4 -/
def a (n : ℕ) : ℚ :=
  8 - 2 * (n - 1)

/-- Sum of first n terms of a_n -/
def S (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

/-- b_n sequence -/
def b (n : ℕ+) : ℚ :=
  1 / ((n : ℚ) * (12 - a n))

/-- Sum of first n terms of b_n -/
def T (n : ℕ+) : ℚ :=
  (n : ℚ) / (2 * (n + 1))

theorem arithmetic_sequence_properties :
  (∃ n : ℕ, S n = 20 ∧ ∀ m : ℕ, S m ≤ S n) ∧
  (∀ n : ℕ+, T n = (n : ℚ) / (2 * (n + 1))) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2705_270542


namespace NUMINAMATH_CALUDE_stadium_ratio_l2705_270539

theorem stadium_ratio (initial_total : ℕ) (initial_girls : ℕ) (final_total : ℕ) 
  (h1 : initial_total = 600)
  (h2 : initial_girls = 240)
  (h3 : final_total = 480)
  (h4 : (initial_total - initial_girls) / 4 + (initial_girls - (initial_total - final_total - (initial_total - initial_girls) / 4)) = initial_girls) :
  (initial_total - final_total - (initial_total - initial_girls) / 4) / initial_girls = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_stadium_ratio_l2705_270539


namespace NUMINAMATH_CALUDE_det_A_squared_minus_3A_l2705_270501

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_A_squared_minus_3A : Matrix.det ((A ^ 2) - 3 • A) = 88 := by
  sorry

end NUMINAMATH_CALUDE_det_A_squared_minus_3A_l2705_270501


namespace NUMINAMATH_CALUDE_cool_function_periodic_l2705_270555

/-- A function is cool if there exist real numbers a and b such that
    f(x + a) is even and f(x + b) is odd. -/
def IsCool (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, (∀ x, f (x + a) = f (-x + a)) ∧ (∀ x, f (x + b) = -f (-x + b))

/-- Every cool function is periodic. -/
theorem cool_function_periodic (f : ℝ → ℝ) (h : IsCool f) :
    ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x :=
  sorry

end NUMINAMATH_CALUDE_cool_function_periodic_l2705_270555


namespace NUMINAMATH_CALUDE_cassidy_poster_count_l2705_270567

/-- Represents Cassidy's poster collection over time -/
structure PosterCollection where
  initial : Nat  -- Initial number of posters 3 years ago
  lost : Nat     -- Number of posters lost
  sold : Nat     -- Number of posters sold
  future : Nat   -- Number of posters to be added this summer

/-- Calculates the current number of posters in Cassidy's collection -/
def currentPosters (c : PosterCollection) : Nat :=
  2 * c.initial - 6

theorem cassidy_poster_count (c : PosterCollection) 
  (h1 : c.initial = 18)
  (h2 : c.lost = 2)
  (h3 : c.sold = 5)
  (h4 : c.future = 6) :
  currentPosters c = 30 := by
  sorry

#eval currentPosters { initial := 18, lost := 2, sold := 5, future := 6 }

end NUMINAMATH_CALUDE_cassidy_poster_count_l2705_270567


namespace NUMINAMATH_CALUDE_max_entropy_is_n_minus_two_l2705_270559

/-- A configuration of children around a circular table -/
structure Configuration (n : ℕ) :=
  (boys : Fin n → Bool)
  (girls : Fin n → Bool)
  (valid : ∀ i, boys i ≠ girls i)

/-- The entropy of a configuration -/
def entropy (n : ℕ) (config : Configuration n) : ℕ :=
  sorry

/-- Theorem: The maximal entropy for any configuration is n-2 when n > 3 -/
theorem max_entropy_is_n_minus_two (n : ℕ) (h : n > 3) :
  ∃ (config : Configuration n), entropy n config = n - 2 ∧
  ∀ (other : Configuration n), entropy n other ≤ n - 2 :=
sorry

end NUMINAMATH_CALUDE_max_entropy_is_n_minus_two_l2705_270559


namespace NUMINAMATH_CALUDE_boat_rental_problem_l2705_270535

theorem boat_rental_problem :
  ∀ (big_boats small_boats : ℕ),
    big_boats + small_boats = 12 →
    6 * big_boats + 4 * small_boats = 58 →
    big_boats = 5 ∧ small_boats = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_rental_problem_l2705_270535


namespace NUMINAMATH_CALUDE_tigers_losses_l2705_270538

theorem tigers_losses (total_games wins : ℕ) (h1 : total_games = 56) (h2 : wins = 38) : 
  ∃ losses ties : ℕ, 
    losses + ties + wins = total_games ∧ 
    ties = losses / 2 ∧
    losses = 12 := by
sorry

end NUMINAMATH_CALUDE_tigers_losses_l2705_270538


namespace NUMINAMATH_CALUDE_card_selection_count_l2705_270521

/-- Represents a standard deck of cards -/
def StandardDeck : Nat := 52

/-- Number of suits in a standard deck -/
def NumSuits : Nat := 4

/-- Number of ranks in a standard deck -/
def NumRanks : Nat := 13

/-- Number of cards to be chosen -/
def CardsToChoose : Nat := 5

/-- Number of cards that must be of the same suit -/
def SameSuitCards : Nat := 2

/-- Number of cards that must be of different suits -/
def DiffSuitCards : Nat := 3

theorem card_selection_count : 
  (Nat.choose NumSuits 1) * 
  (Nat.choose NumRanks SameSuitCards) * 
  (Nat.choose (NumSuits - 1) DiffSuitCards) * 
  ((Nat.choose (NumRanks - SameSuitCards) 1) ^ DiffSuitCards) = 414384 := by
  sorry

end NUMINAMATH_CALUDE_card_selection_count_l2705_270521


namespace NUMINAMATH_CALUDE_carmela_difference_l2705_270590

def cecil_money : ℕ := 600
def catherine_money : ℕ := 2 * cecil_money - 250
def total_money : ℕ := 2800

theorem carmela_difference : ℕ := by
  have h1 : cecil_money + catherine_money + (2 * cecil_money + (total_money - (cecil_money + catherine_money))) = total_money := by sorry
  have h2 : total_money - (cecil_money + catherine_money) = 50 := by sorry
  exact 50

#check carmela_difference

end NUMINAMATH_CALUDE_carmela_difference_l2705_270590


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_inequality_l2705_270577

theorem sum_and_reciprocal_inequality (x : ℝ) (hx : x > 0) : 
  x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_inequality_l2705_270577


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_360_l2705_270551

/-- The sum of positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The number of distinct prime factors of a natural number n -/
def num_distinct_prime_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of distinct prime factors of the sum of positive divisors of 360 is 4 -/
theorem distinct_prime_factors_of_divisor_sum_360 : 
  num_distinct_prime_factors (sum_of_divisors 360) = 4 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_360_l2705_270551


namespace NUMINAMATH_CALUDE_airplane_fraction_is_one_third_l2705_270532

/-- Represents the travel scenario with given conditions -/
structure TravelScenario where
  driving_time : ℕ
  airport_drive_time : ℕ
  airport_wait_time : ℕ
  post_flight_time : ℕ
  time_saved : ℕ

/-- Calculates the fraction of time spent on the airplane compared to driving -/
def airplane_time_fraction (scenario : TravelScenario) : ℚ :=
  let airplane_time := scenario.driving_time - scenario.airport_drive_time - 
                       scenario.airport_wait_time - scenario.post_flight_time - 
                       scenario.time_saved
  airplane_time / scenario.driving_time

/-- The main theorem stating that the fraction of time spent on the airplane is 1/3 -/
theorem airplane_fraction_is_one_third (scenario : TravelScenario) 
    (h1 : scenario.driving_time = 195)
    (h2 : scenario.airport_drive_time = 10)
    (h3 : scenario.airport_wait_time = 20)
    (h4 : scenario.post_flight_time = 10)
    (h5 : scenario.time_saved = 90) :
    airplane_time_fraction scenario = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_airplane_fraction_is_one_third_l2705_270532


namespace NUMINAMATH_CALUDE_equation_solutions_l2705_270595

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -2 + Real.sqrt 5 ∧ x₂ = -2 - Real.sqrt 5 ∧
    x₁^2 + 4*x₁ - 1 = 0 ∧ x₂^2 + 4*x₂ - 1 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 3 ∧ y₂ = 1 ∧
    (y₁ - 3)^2 + 2*y₁*(y₁ - 3) = 0 ∧ (y₂ - 3)^2 + 2*y₂*(y₂ - 3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2705_270595


namespace NUMINAMATH_CALUDE_water_in_bucket_l2705_270514

theorem water_in_bucket (initial_amount : ℝ) (poured_out : ℝ) : 
  initial_amount = 0.8 → poured_out = 0.2 → initial_amount - poured_out = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_water_in_bucket_l2705_270514


namespace NUMINAMATH_CALUDE_odd_function_negative_x_l2705_270587

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_x
  (f : ℝ → ℝ)
  (odd : OddFunction f)
  (pos : ∀ x > 0, f x = x * (1 - x)) :
  ∀ x < 0, f x = x * (1 + x) := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_x_l2705_270587


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l2705_270526

theorem zoo_animal_ratio (initial_animals : ℕ) (final_animals : ℕ)
  (gorillas_sent : ℕ) (hippo_adopted : ℕ) (rhinos_taken : ℕ) (lion_cubs_born : ℕ)
  (h1 : initial_animals = 68)
  (h2 : final_animals = 90)
  (h3 : gorillas_sent = 6)
  (h4 : hippo_adopted = 1)
  (h5 : rhinos_taken = 3)
  (h6 : lion_cubs_born = 8) :
  (final_animals - (initial_animals - gorillas_sent + hippo_adopted + rhinos_taken + lion_cubs_born)) / lion_cubs_born = 2 :=
by sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l2705_270526


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2705_270569

def P : Set ℕ := {0, 2, 4, 6}
def Q : Set ℕ := {x | x ≤ 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2705_270569


namespace NUMINAMATH_CALUDE_first_division_percentage_l2705_270509

theorem first_division_percentage (total_students : ℕ) 
  (second_division_percent : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  second_division_percent = 54 / 100 →
  just_passed = 54 →
  (just_passed : ℚ) / total_students + second_division_percent + 28 / 100 = 1 :=
by sorry

end NUMINAMATH_CALUDE_first_division_percentage_l2705_270509


namespace NUMINAMATH_CALUDE_max_regions_four_lines_l2705_270597

/-- The maximum number of regions into which a plane can be divided using n straight lines -/
def L (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- The theorem stating that 4 straight lines can divide a plane into at most 11 regions -/
theorem max_regions_four_lines : L 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_four_lines_l2705_270597


namespace NUMINAMATH_CALUDE_subset_condition_l2705_270562

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ m + 1}

theorem subset_condition (m : ℝ) : B m ⊆ A → -1 ≤ m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l2705_270562


namespace NUMINAMATH_CALUDE_assignment_result_l2705_270550

def assignment_sequence (initial_a : ℕ) : ℕ :=
  let a₁ := initial_a
  let a₂ := a₁ + 1
  a₂

theorem assignment_result : assignment_sequence 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_assignment_result_l2705_270550


namespace NUMINAMATH_CALUDE_no_periodic_sequence_exists_l2705_270541

-- Define a_n as the first non-zero digit from the unit place in n!
def a (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_periodic_sequence_exists :
  ∀ N : ℕ, ¬∃ T : ℕ, T > 0 ∧ ∀ k : ℕ, a (N + k + T) = a (N + k) :=
sorry

end NUMINAMATH_CALUDE_no_periodic_sequence_exists_l2705_270541


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2705_270540

theorem min_value_trig_expression (α γ : ℝ) :
  (3 * Real.cos α + 4 * Real.sin γ - 7)^2 + (3 * Real.sin α + 4 * Real.cos γ - 12)^2 ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2705_270540


namespace NUMINAMATH_CALUDE_plum_count_l2705_270515

/-- The number of plums initially in the basket -/
def initial_plums : ℕ := 17

/-- The number of plums added to the basket -/
def added_plums : ℕ := 4

/-- The final number of plums in the basket -/
def final_plums : ℕ := initial_plums + added_plums

theorem plum_count : final_plums = 21 := by sorry

end NUMINAMATH_CALUDE_plum_count_l2705_270515


namespace NUMINAMATH_CALUDE_value_of_C_l2705_270548

theorem value_of_C : ∃ C : ℝ, (4 * C + 3 = 25) ∧ (C = 5.5) := by sorry

end NUMINAMATH_CALUDE_value_of_C_l2705_270548


namespace NUMINAMATH_CALUDE_lotto_winning_percentage_l2705_270536

theorem lotto_winning_percentage :
  let total_tickets : ℕ := 200
  let cost_per_ticket : ℚ := 2
  let grand_prize : ℚ := 5000
  let profit : ℚ := 4830
  let five_dollar_win_ratio : ℚ := 4/5
  let ten_dollar_win_ratio : ℚ := 1/5
  let five_dollar_prize : ℚ := 5
  let ten_dollar_prize : ℚ := 10
  ∃ (winning_tickets : ℕ),
    (winning_tickets : ℚ) / total_tickets * 100 = 19 ∧
    profit = five_dollar_win_ratio * winning_tickets * five_dollar_prize +
             ten_dollar_win_ratio * winning_tickets * ten_dollar_prize +
             grand_prize -
             (total_tickets * cost_per_ticket) :=
by sorry

end NUMINAMATH_CALUDE_lotto_winning_percentage_l2705_270536


namespace NUMINAMATH_CALUDE_nine_caps_per_box_l2705_270504

/-- Given a total number of bottle caps and a number of boxes, 
    calculate the number of bottle caps in each box. -/
def bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℕ) : ℕ :=
  total_caps / num_boxes

/-- Theorem stating that with 54 total bottle caps and 6 boxes, 
    there are 9 bottle caps in each box. -/
theorem nine_caps_per_box :
  bottle_caps_per_box 54 6 = 9 := by
  sorry

#eval bottle_caps_per_box 54 6

end NUMINAMATH_CALUDE_nine_caps_per_box_l2705_270504


namespace NUMINAMATH_CALUDE_largest_three_digit_perfect_square_diff_l2705_270572

/-- A function that returns the sum of digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is a three-digit number. -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The main theorem stating that 919 is the largest three-digit number
    such that the number minus the sum of its digits is a perfect square. -/
theorem largest_three_digit_perfect_square_diff :
  ∀ n : ℕ, is_three_digit n →
    (∃ k : ℕ, n - sum_of_digits n = k^2) →
    n ≤ 919 := by sorry

end NUMINAMATH_CALUDE_largest_three_digit_perfect_square_diff_l2705_270572


namespace NUMINAMATH_CALUDE_no_integer_solution_l2705_270519

theorem no_integer_solution (P : Int → Int) (a b c : Int) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  P a = 2 ∧ P b = 2 ∧ P c = 2 →
  ∀ k : Int, P k ≠ 3 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2705_270519


namespace NUMINAMATH_CALUDE_team_selection_count_l2705_270596

/-- The number of ways to select a team of 3 people from 3 male and 3 female teachers,
    with both genders included -/
def select_team (male_teachers female_teachers team_size : ℕ) : ℕ :=
  (male_teachers.choose 2 * female_teachers.choose 1) +
  (male_teachers.choose 1 * female_teachers.choose 2)

/-- Theorem: There are 18 ways to select a team of 3 from 3 male and 3 female teachers,
    with both genders included -/
theorem team_selection_count :
  select_team 3 3 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l2705_270596


namespace NUMINAMATH_CALUDE_soap_lasts_two_months_l2705_270593

def soap_problem (cost_per_bar : ℚ) (yearly_cost : ℚ) (months_per_year : ℕ) : ℚ :=
  (months_per_year : ℚ) / (yearly_cost / cost_per_bar)

theorem soap_lasts_two_months :
  soap_problem 8 48 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_soap_lasts_two_months_l2705_270593


namespace NUMINAMATH_CALUDE_equation_solution_l2705_270558

theorem equation_solution : ∃ n : ℝ, 0.03 * n + 0.08 * (20 + n) = 12.6 ∧ n = 100 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2705_270558


namespace NUMINAMATH_CALUDE_circle_ratio_problem_l2705_270589

theorem circle_ratio_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  π * b^2 - π * a^2 = 3 * (π * a^2) → a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_problem_l2705_270589


namespace NUMINAMATH_CALUDE_remainder_theorem_l2705_270534

theorem remainder_theorem (z : ℕ) (hz : z > 0) (hz_div : 4 ∣ z) : (z * (2 + 4 + z) + 3) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2705_270534


namespace NUMINAMATH_CALUDE_complex_square_roots_l2705_270560

theorem complex_square_roots : 
  let z₁ : ℂ := Complex.mk (3 * Real.sqrt 2) (-55 * Real.sqrt 2 / 6)
  let z₂ : ℂ := Complex.mk (-3 * Real.sqrt 2) (55 * Real.sqrt 2 / 6)
  z₁^2 = Complex.mk (-121) (-110) ∧ z₂^2 = Complex.mk (-121) (-110) :=
by sorry

end NUMINAMATH_CALUDE_complex_square_roots_l2705_270560


namespace NUMINAMATH_CALUDE_foot_of_perpendicular_l2705_270537

-- Define the point A
def A : ℝ × ℝ := (1, 2)

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the perpendicular line from A to the x-axis
def perp_line : Set (ℝ × ℝ) := {p | p.1 = A.1}

-- Define point M as the intersection of the perpendicular line and the x-axis
def M : ℝ × ℝ := (A.1, 0)

-- Theorem statement
theorem foot_of_perpendicular : M ∈ x_axis ∧ M ∈ perp_line := by sorry

end NUMINAMATH_CALUDE_foot_of_perpendicular_l2705_270537


namespace NUMINAMATH_CALUDE_weight_of_new_person_l2705_270510

/-- Given a group of 9 people where one person is replaced, this theorem calculates the weight of the new person based on the average weight increase. -/
theorem weight_of_new_person
  (n : ℕ) -- number of people
  (w : ℝ) -- weight of the person being replaced
  (d : ℝ) -- increase in average weight
  (h1 : n = 9)
  (h2 : w = 65)
  (h3 : d = 1.5) :
  w + n * d = 78.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l2705_270510


namespace NUMINAMATH_CALUDE_more_birds_than_nests_l2705_270505

/-- Given 6 birds and 3 nests, prove that there are 3 more birds than nests. -/
theorem more_birds_than_nests (birds : ℕ) (nests : ℕ) 
  (h1 : birds = 6) (h2 : nests = 3) : birds - nests = 3 := by
  sorry

end NUMINAMATH_CALUDE_more_birds_than_nests_l2705_270505


namespace NUMINAMATH_CALUDE_delta_y_over_delta_x_l2705_270570

/-- Given a function f(x) = -x² + x and two points on its graph,
    prove that Δy/Δx = 3 - Δx -/
theorem delta_y_over_delta_x (f : ℝ → ℝ) (Δx Δy : ℝ) :
  (∀ x, f x = -x^2 + x) →
  f (-1) = -2 →
  f (-1 + Δx) = -2 + Δy →
  Δx ≠ 0 →
  Δy / Δx = 3 - Δx :=
by sorry

end NUMINAMATH_CALUDE_delta_y_over_delta_x_l2705_270570


namespace NUMINAMATH_CALUDE_sum_of_roots_l2705_270591

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2705_270591


namespace NUMINAMATH_CALUDE_max_value_on_curve_l2705_270552

noncomputable def max_value (b : ℝ) : ℝ :=
  if 0 < b ∧ b ≤ 4 then b^2 / 4 + 4 else 2 * b

theorem max_value_on_curve (b : ℝ) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / 4 + y^2 / b^2 = 1 → x^2 + 2*y ≤ max_value b) ∧
  (∃ x y : ℝ, x^2 / 4 + y^2 / b^2 = 1 ∧ x^2 + 2*y = max_value b) :=
sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l2705_270552


namespace NUMINAMATH_CALUDE_sin_product_45_deg_l2705_270571

theorem sin_product_45_deg (α β : Real) 
  (h1 : Real.sin (α + β) = 0.2) 
  (h2 : Real.cos (α - β) = 0.3) : 
  Real.sin (α + Real.pi/4) * Real.sin (β + Real.pi/4) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_45_deg_l2705_270571


namespace NUMINAMATH_CALUDE_ellipse_properties_l2705_270598

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the semi-major axis, semi-minor axis, and semi-focal distance
def semi_major_axis : ℝ := 5
def semi_minor_axis : ℝ := 4
def semi_focal_distance : ℝ := 3

-- Theorem statement
theorem ellipse_properties :
  (∀ x y : ℝ, ellipse_equation x y) →
  semi_major_axis = 5 ∧ semi_minor_axis = 4 ∧ semi_focal_distance = 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2705_270598


namespace NUMINAMATH_CALUDE_difference_of_squares_l2705_270547

theorem difference_of_squares (m : ℝ) : (m + 1) * (m - 1) = m^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2705_270547


namespace NUMINAMATH_CALUDE_password_combinations_l2705_270525

/-- A digit is either odd or even -/
inductive Digit
| odd
| even

/-- The set of possible digits -/
def digit_set : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- A valid password is a list of four digits satisfying the given conditions -/
def ValidPassword : Type := List Digit

/-- The number of odd digits in the digit set -/
def num_odd_digits : Nat := (digit_set.filter (fun n => n % 2 = 1)).card

/-- The number of even digits in the digit set -/
def num_even_digits : Nat := (digit_set.filter (fun n => n % 2 = 0)).card

/-- The total number of digits in the digit set -/
def total_digits : Nat := digit_set.card

/-- The number of valid passwords -/
def num_valid_passwords : Nat := 
  (num_odd_digits * num_even_digits * total_digits * total_digits) +
  (num_even_digits * num_odd_digits * total_digits * total_digits)

theorem password_combinations : num_valid_passwords = 648 := by
  sorry

end NUMINAMATH_CALUDE_password_combinations_l2705_270525


namespace NUMINAMATH_CALUDE_complement_A_relative_to_I_l2705_270581

def I : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {x : Int | x^2 < 3}

theorem complement_A_relative_to_I :
  {x ∈ I | x ∉ A} = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_relative_to_I_l2705_270581


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_iff_a_in_range_l2705_270556

/-- Piecewise function f(x) defined by parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5*a - 4)*x + 7*a - 3 else (2*a - 1)^x

/-- The range of a for which f is monotonically decreasing -/
def a_range : Set ℝ := Set.Icc (3/5) (4/5)

/-- Theorem stating that f is monotonically decreasing iff a is in the specified range -/
theorem f_monotone_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) ↔ a ∈ a_range :=
sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_iff_a_in_range_l2705_270556


namespace NUMINAMATH_CALUDE_joint_savings_account_total_l2705_270511

def kimmie_earnings : ℚ := 1950
def zahra_earnings_ratio : ℚ := 1 - 2/3
def layla_earnings_ratio : ℚ := 9/4
def kimmie_savings_rate : ℚ := 35/100
def zahra_savings_rate : ℚ := 40/100
def layla_savings_rate : ℚ := 30/100

theorem joint_savings_account_total :
  let zahra_earnings := kimmie_earnings * zahra_earnings_ratio
  let layla_earnings := kimmie_earnings * layla_earnings_ratio
  let kimmie_savings := kimmie_earnings * kimmie_savings_rate
  let zahra_savings := zahra_earnings * zahra_savings_rate
  let layla_savings := layla_earnings * layla_savings_rate
  let total_savings := kimmie_savings + zahra_savings + layla_savings
  total_savings = 2258.75 := by
  sorry

end NUMINAMATH_CALUDE_joint_savings_account_total_l2705_270511


namespace NUMINAMATH_CALUDE_interest_calculation_l2705_270520

theorem interest_calculation (P : ℝ) : 
  P * (1 + 5/100)^2 - P - (P * 5 * 2 / 100) = 17 → P = 6800 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l2705_270520


namespace NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l2705_270523

/-- Given a rectangle with dimensions 6 and 9, prove that the ratio of the volumes of cylinders
    formed by rolling along each side is 3/4, with the larger volume in the numerator. -/
theorem rectangle_cylinder_volume_ratio :
  let rect_width : ℝ := 6
  let rect_height : ℝ := 9
  let volume1 := π * (rect_width / (2 * π))^2 * rect_height
  let volume2 := π * (rect_height / (2 * π))^2 * rect_width
  max volume1 volume2 / min volume1 volume2 = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l2705_270523


namespace NUMINAMATH_CALUDE_lost_bottle_caps_l2705_270530

/-- Represents the number of bottle caps Danny has now -/
def current_bottle_caps : ℕ := 25

/-- Represents the number of bottle caps Danny had at first -/
def initial_bottle_caps : ℕ := 91

/-- Theorem stating that the number of lost bottle caps is the difference between
    the initial number and the current number of bottle caps -/
theorem lost_bottle_caps : 
  initial_bottle_caps - current_bottle_caps = 66 := by
  sorry

end NUMINAMATH_CALUDE_lost_bottle_caps_l2705_270530


namespace NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l2705_270531

theorem p_neither_sufficient_nor_necessary_for_q :
  ¬(∀ x y : ℝ, x + y ≠ -2 → (x ≠ -1 ∧ y ≠ -1)) ∧
  ¬(∀ x y : ℝ, (x ≠ -1 ∧ y ≠ -1) → x + y ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_p_neither_sufficient_nor_necessary_for_q_l2705_270531


namespace NUMINAMATH_CALUDE_max_spheres_in_frustum_l2705_270565

/-- Represents a frustum with given height and spheres inside it -/
structure Frustum :=
  (height : ℝ)
  (O₁_radius : ℝ)
  (O₂_radius : ℝ)

/-- Calculates the maximum number of additional spheres that can fit in the frustum -/
def max_additional_spheres (f : Frustum) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- The main theorem stating the maximum number of additional spheres -/
theorem max_spheres_in_frustum (f : Frustum) 
  (h₁ : f.height = 8)
  (h₂ : f.O₁_radius = 2)
  (h₃ : f.O₂_radius = 3) :
  max_additional_spheres f = 2 :=
sorry

end NUMINAMATH_CALUDE_max_spheres_in_frustum_l2705_270565


namespace NUMINAMATH_CALUDE_mary_warm_hours_l2705_270594

/-- The number of sticks of wood produced by chopping up a chair -/
def sticks_per_chair : ℕ := 6

/-- The number of sticks of wood produced by chopping up a table -/
def sticks_per_table : ℕ := 9

/-- The number of sticks of wood produced by chopping up a stool -/
def sticks_per_stool : ℕ := 2

/-- The number of sticks of wood Mary needs to burn per hour to stay warm -/
def sticks_per_hour : ℕ := 5

/-- The number of chairs Mary chops up -/
def chairs_chopped : ℕ := 18

/-- The number of tables Mary chops up -/
def tables_chopped : ℕ := 6

/-- The number of stools Mary chops up -/
def stools_chopped : ℕ := 4

/-- Theorem stating how many hours Mary can keep warm -/
theorem mary_warm_hours : 
  (chairs_chopped * sticks_per_chair + 
   tables_chopped * sticks_per_table + 
   stools_chopped * sticks_per_stool) / sticks_per_hour = 34 := by
  sorry

end NUMINAMATH_CALUDE_mary_warm_hours_l2705_270594


namespace NUMINAMATH_CALUDE_green_beads_count_l2705_270517

/-- The number of green beads in a jewelry pattern -/
def green_beads : ℕ := 3

/-- The number of purple beads in the pattern -/
def purple_beads : ℕ := 5

/-- The number of red beads in the pattern -/
def red_beads : ℕ := 2 * green_beads

/-- The number of times the pattern repeats in a bracelet -/
def bracelet_repeats : ℕ := 3

/-- The number of times the pattern repeats in a necklace -/
def necklace_repeats : ℕ := 5

/-- The total number of beads needed for 1 bracelet and 10 necklaces -/
def total_beads : ℕ := 742

/-- The number of bracelets to be made -/
def num_bracelets : ℕ := 1

/-- The number of necklaces to be made -/
def num_necklaces : ℕ := 10

theorem green_beads_count : 
  num_bracelets * bracelet_repeats * (green_beads + purple_beads + red_beads) + 
  num_necklaces * necklace_repeats * (green_beads + purple_beads + red_beads) = total_beads :=
by sorry

end NUMINAMATH_CALUDE_green_beads_count_l2705_270517


namespace NUMINAMATH_CALUDE_vector_properties_l2705_270549

/-- Given points in a 2D Cartesian coordinate system -/
def O : Fin 2 → ℝ := ![0, 0]
def A : Fin 2 → ℝ := ![1, 2]
def B : Fin 2 → ℝ := ![-3, 4]

/-- Vector AB -/
def vecAB : Fin 2 → ℝ := ![B 0 - A 0, B 1 - A 1]

/-- Theorem stating properties of vectors and angles in the given problem -/
theorem vector_properties :
  (vecAB 0 = -4 ∧ vecAB 1 = 2) ∧
  Real.sqrt ((vecAB 0)^2 + (vecAB 1)^2) = 2 * Real.sqrt 5 ∧
  ((A 0 * B 0 + A 1 * B 1) / (Real.sqrt (A 0^2 + A 1^2) * Real.sqrt (B 0^2 + B 1^2))) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l2705_270549


namespace NUMINAMATH_CALUDE_problem_statement_l2705_270500

theorem problem_statement (a b : ℝ) 
  (h1 : a + b = -3) 
  (h2 : a^2 * b + a * b^2 = -30) : 
  a^2 - a*b + b^2 + 11 = -10 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2705_270500


namespace NUMINAMATH_CALUDE_quadratic_function_values_l2705_270568

def f (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 + b

theorem quadratic_function_values (a b : ℝ) (h : a ≠ 0) :
  (∀ x ∈ Set.Icc 2 3, f a b x ≤ 5) ∧
  (∃ x ∈ Set.Icc 2 3, f a b x = 5) ∧
  (∀ x ∈ Set.Icc 2 3, f a b x ≥ 2) ∧
  (∃ x ∈ Set.Icc 2 3, f a b x = 2) →
  ((a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_values_l2705_270568


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2705_270573

theorem absolute_value_inequality_solution (a : ℝ) : 
  (∀ x : ℝ, |a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3) → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_l2705_270573


namespace NUMINAMATH_CALUDE_money_redistribution_l2705_270584

/-- Represents the amount of money each person has -/
structure Money where
  amy : ℝ
  jan : ℝ
  toy : ℝ
  kim : ℝ

/-- Represents the redistribution rules -/
def redistribute (m : Money) : Money :=
  let step1 := Money.mk m.amy m.jan m.toy m.kim -- Kim equalizes others
  let step2 := Money.mk m.amy m.jan m.toy m.kim -- Amy doubles Jan and Toy
  let step3 := Money.mk m.amy m.jan m.toy m.kim -- Jan doubles Amy and Toy
  let step4 := Money.mk m.amy m.jan m.toy m.kim -- Toy doubles others
  step4

theorem money_redistribution (initial final : Money) :
  initial.toy = 48 →
  final.toy = 48 →
  final = redistribute initial →
  initial.amy + initial.jan + initial.toy + initial.kim = 192 :=
by
  sorry

#check money_redistribution

end NUMINAMATH_CALUDE_money_redistribution_l2705_270584


namespace NUMINAMATH_CALUDE_ted_green_mushrooms_l2705_270561

/-- The number of green mushrooms Ted gathered -/
def green_mushrooms : ℕ := sorry

/-- The number of red mushrooms Bill gathered -/
def red_mushrooms : ℕ := 12

/-- The number of brown mushrooms Bill gathered -/
def brown_mushrooms : ℕ := 6

/-- The number of blue mushrooms Ted gathered -/
def blue_mushrooms : ℕ := 6

/-- The total number of white-spotted mushrooms gathered -/
def total_white_spotted : ℕ := 17

theorem ted_green_mushrooms :
  green_mushrooms = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ted_green_mushrooms_l2705_270561


namespace NUMINAMATH_CALUDE_max_value_inequality_l2705_270533

theorem max_value_inequality (x y : ℝ) : 
  (x + 3*y + 4) / Real.sqrt (x^2 + y^2 + x + 1) ≤ Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2705_270533


namespace NUMINAMATH_CALUDE_race_distance_l2705_270583

theorem race_distance (total : ℝ) (selena : ℝ) (josh : ℝ) 
  (h1 : total = 36)
  (h2 : selena + josh = total)
  (h3 : josh = selena / 2) : 
  selena = 24 := by
sorry

end NUMINAMATH_CALUDE_race_distance_l2705_270583


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l2705_270546

theorem solution_satisfies_equations :
  let x : ℚ := -256 / 29
  let y : ℚ := -37 / 29
  (7 * x - 50 * y = 2) ∧ (3 * y - x = 5) := by
sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l2705_270546


namespace NUMINAMATH_CALUDE_basketball_game_result_l2705_270582

/-- Represents a basketball player with their score and penalties -/
structure Player where
  score : ℕ
  penalties : List ℕ

/-- Calculates the total points for a player after applying penalties -/
def playerPoints (p : Player) : ℤ :=
  p.score - (List.sum p.penalties)

/-- Calculates the total points for a team -/
def teamPoints (team : List Player) : ℤ :=
  List.sum (team.map playerPoints)

theorem basketball_game_result :
  let team_a := [
    Player.mk 12 [1, 2],
    Player.mk 18 [1, 2, 3],
    Player.mk 5 [],
    Player.mk 7 [1, 2],
    Player.mk 6 [1]
  ]
  let team_b := [
    Player.mk 10 [1, 2],
    Player.mk 9 [1],
    Player.mk 12 [],
    Player.mk 8 [1, 2, 3],
    Player.mk 5 [1, 2],
    Player.mk 4 [1]
  ]
  teamPoints team_a - teamPoints team_b = 1 := by
  sorry


end NUMINAMATH_CALUDE_basketball_game_result_l2705_270582


namespace NUMINAMATH_CALUDE_factory_underpayment_l2705_270574

/-- The hourly wage in yuan -/
def hourly_wage : ℚ := 6

/-- The nominal work day duration in hours -/
def nominal_work_day : ℚ := 8

/-- The time for clock hands to coincide in the inaccurate clock (in minutes) -/
def inaccurate_coincidence_time : ℚ := 69

/-- The time for clock hands to coincide in an accurate clock (in minutes) -/
def accurate_coincidence_time : ℚ := 720 / 11

/-- Calculate the actual work time based on the inaccurate clock -/
def actual_work_time : ℚ :=
  (inaccurate_coincidence_time * nominal_work_day) / accurate_coincidence_time

/-- Calculate the underpayment amount -/
def underpayment : ℚ := hourly_wage * (actual_work_time - nominal_work_day)

theorem factory_underpayment :
  underpayment = 13/5 :=
by sorry

end NUMINAMATH_CALUDE_factory_underpayment_l2705_270574


namespace NUMINAMATH_CALUDE_point_inside_circle_l2705_270576

theorem point_inside_circle (a b : ℝ) : 
  a ≠ b → 
  a^2 - a - Real.sqrt 2 = 0 → 
  b^2 - b - Real.sqrt 2 = 0 → 
  a^2 + b^2 < 8 := by
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l2705_270576


namespace NUMINAMATH_CALUDE_surface_area_increase_l2705_270508

/-- Represents a rectangular solid with given dimensions -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the original solid -/
def originalSolid : RectangularSolid :=
  { length := 4, width := 3, height := 2 }

/-- Represents the size of the removed cube -/
def cubeSize : ℝ := 1

/-- Theorem stating that removing a 1-foot cube from the center of the original solid
    increases its surface area by 6 square feet -/
theorem surface_area_increase :
  surfaceArea originalSolid + 6 = surfaceArea originalSolid + 6 * cubeSize^2 := by
  sorry

#check surface_area_increase

end NUMINAMATH_CALUDE_surface_area_increase_l2705_270508


namespace NUMINAMATH_CALUDE_f_difference_f_equals_x_plus_3_l2705_270524

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem 1: For any real number a, f(a) - f(a + 1) = -2a - 1
theorem f_difference (a : ℝ) : f a - f (a + 1) = -2 * a - 1 := by
  sorry

-- Theorem 2: If f(x) = x + 3, then x = -1 or x = 2
theorem f_equals_x_plus_3 (x : ℝ) : f x = x + 3 → x = -1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_f_equals_x_plus_3_l2705_270524


namespace NUMINAMATH_CALUDE_characterization_of_k_set_l2705_270566

-- Define h as 2^r where r is a non-negative integer
def h (r : ℕ) : ℕ := 2^r

-- Define the set of k that satisfy the conditions
def k_set (h : ℕ) : Set ℕ := {k : ℕ | ∃ (m n : ℕ), m > n ∧ k ∣ (m^h - 1) ∧ n^((m^h - 1) / k) ≡ -1 [ZMOD m]}

-- The theorem to prove
theorem characterization_of_k_set (r : ℕ) : 
  k_set (h r) = {k : ℕ | ∃ (s t : ℕ), k = 2^(r+s) * t ∧ Odd t} :=
sorry

end NUMINAMATH_CALUDE_characterization_of_k_set_l2705_270566


namespace NUMINAMATH_CALUDE_bicycle_price_increase_l2705_270579

theorem bicycle_price_increase (original_price new_price : ℝ) 
  (h1 : original_price = 220)
  (h2 : new_price = 253) :
  (new_price - original_price) / original_price * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_price_increase_l2705_270579


namespace NUMINAMATH_CALUDE_all_transylvanians_answer_yes_l2705_270506

-- Define the types of Transylvanians
inductive TransylvanianType
  | SaneHuman
  | InsaneHuman
  | SaneVampire
  | InsaneVampire

-- Define the possible questions
inductive Question
  | ConsiderHuman
  | Reliable

-- Define the function that represents a Transylvanian's answer
def transylvanianAnswer (t : TransylvanianType) (q : Question) : Bool :=
  match q with
  | Question.ConsiderHuman => true
  | Question.Reliable => true

-- Theorem statement
theorem all_transylvanians_answer_yes
  (t : TransylvanianType) (q : Question) :
  transylvanianAnswer t q = true := by sorry

end NUMINAMATH_CALUDE_all_transylvanians_answer_yes_l2705_270506


namespace NUMINAMATH_CALUDE_no_prime_roots_for_specific_quadratic_l2705_270578

theorem no_prime_roots_for_specific_quadratic :
  ¬∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    (p : ℤ) + q = 71 ∧
    (p : ℤ) * q = k ∧
    p ≠ q :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_specific_quadratic_l2705_270578


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_squared_equation_solutions_l2705_270588

-- Problem 1
theorem quadratic_equation_roots (x : ℝ) : 
  x^2 - 7*x + 6 = 0 ↔ x = 1 ∨ x = 6 := by sorry

-- Problem 2
theorem squared_equation_solutions (x : ℝ) :
  (2*x + 3)^2 = (x - 3)^2 ↔ x = 0 ∨ x = -6 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_squared_equation_solutions_l2705_270588


namespace NUMINAMATH_CALUDE_ant_position_2024_l2705_270557

-- Define the ant's movement pattern
def antMove (n : ℕ) : ℤ × ℤ :=
  sorry

-- Theorem statement
theorem ant_position_2024 : antMove 2024 = (13, 0) := by
  sorry

end NUMINAMATH_CALUDE_ant_position_2024_l2705_270557


namespace NUMINAMATH_CALUDE_cos_sin_fifteen_degrees_l2705_270585

theorem cos_sin_fifteen_degrees : 
  Real.cos (15 * π / 180) ^ 4 - Real.sin (15 * π / 180) ^ 4 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_fifteen_degrees_l2705_270585


namespace NUMINAMATH_CALUDE_derivative_ln_2x_squared_plus_1_l2705_270564

open Real

theorem derivative_ln_2x_squared_plus_1 (x : ℝ) :
  deriv (λ x => Real.log (2 * x^2 + 1)) x = (4 * x) / (2 * x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_derivative_ln_2x_squared_plus_1_l2705_270564


namespace NUMINAMATH_CALUDE_odd_function_symmetric_behavior_l2705_270553

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def HasMinimumOn (f : ℝ → ℝ) (a b m : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → m ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = m)

def HasMaximumOn (f : ℝ → ℝ) (a b M : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ M) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = M)

theorem odd_function_symmetric_behavior (f : ℝ → ℝ) :
  IsOdd f →
  IsIncreasingOn f 3 7 →
  HasMinimumOn f 3 7 5 →
  IsIncreasingOn f (-7) (-3) ∧ HasMaximumOn f (-7) (-3) (-5) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_symmetric_behavior_l2705_270553


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_three_l2705_270512

theorem trigonometric_product_equals_three :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_three_l2705_270512


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2705_270528

theorem solution_set_of_inequality (x : ℝ) :
  (x * (x - 1) < 2) ↔ (-1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2705_270528


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2705_270503

/-- The area of an isosceles right triangle with hypotenuse 6√2 is 18 square units. -/
theorem isosceles_right_triangle_area (h : ℝ) (a : ℝ) (A : ℝ) : 
  h = 6 * Real.sqrt 2 →  -- hypotenuse is 6√2
  h = a * Real.sqrt 2 →  -- relationship between hypotenuse and leg in isosceles right triangle
  A = (1/2) * a^2 →      -- area formula for right triangle
  A = 18 := by
    sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l2705_270503


namespace NUMINAMATH_CALUDE_maggies_share_l2705_270545

def total_sum : ℝ := 6000
def debby_percentage : ℝ := 0.25

theorem maggies_share :
  let debby_share := debby_percentage * total_sum
  let maggie_share := total_sum - debby_share
  maggie_share = 4500 := by sorry

end NUMINAMATH_CALUDE_maggies_share_l2705_270545


namespace NUMINAMATH_CALUDE_negation_existence_l2705_270518

theorem negation_existence (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ ¬(∀ x : ℝ, x^2 - a*x + 1 ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_negation_existence_l2705_270518


namespace NUMINAMATH_CALUDE_product_zero_in_special_set_l2705_270522

theorem product_zero_in_special_set (n : ℕ) (h : n = 1997) (S : Finset ℝ) 
  (hS : S.card = n) 
  (hSum : ∀ x ∈ S, (S.sum id - x) ∈ S) : 
  S.prod id = 0 := by
sorry

end NUMINAMATH_CALUDE_product_zero_in_special_set_l2705_270522


namespace NUMINAMATH_CALUDE_fathers_seedlings_count_l2705_270527

/-- The number of seedlings Remi planted on the first day -/
def first_day_seedlings : ℕ := 200

/-- The number of seedlings Remi planted on the second day -/
def second_day_seedlings : ℕ := 2 * first_day_seedlings

/-- The total number of seedlings transferred to the farm on both days -/
def total_seedlings : ℕ := 1200

/-- The number of seedlings Remi's father planted -/
def fathers_seedlings : ℕ := total_seedlings - (first_day_seedlings + second_day_seedlings)

theorem fathers_seedlings_count : fathers_seedlings = 600 := by
  sorry

end NUMINAMATH_CALUDE_fathers_seedlings_count_l2705_270527


namespace NUMINAMATH_CALUDE_fair_coin_prob_heads_l2705_270554

-- Define a fair coin
def fair_coin : Type := Unit

-- Define the probability of landing heads for a fair coin
def prob_heads (c : fair_coin) : ℚ := 1 / 2

-- Define a sequence of coin tosses
def coin_tosses : ℕ → fair_coin
  | _ => ()

-- State the theorem
theorem fair_coin_prob_heads (n : ℕ) : 
  prob_heads (coin_tosses n) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_prob_heads_l2705_270554


namespace NUMINAMATH_CALUDE_remaining_laps_after_sunday_morning_l2705_270543

def total_required_laps : ℕ := 198

def friday_morning_laps : ℕ := 23
def friday_afternoon_laps : ℕ := 12
def friday_evening_laps : ℕ := 28

def saturday_morning_laps : ℕ := 35
def saturday_afternoon_laps : ℕ := 27

def sunday_morning_laps : ℕ := 15

def friday_total : ℕ := friday_morning_laps + friday_afternoon_laps + friday_evening_laps
def saturday_total : ℕ := saturday_morning_laps + saturday_afternoon_laps
def laps_before_sunday_break : ℕ := friday_total + saturday_total + sunday_morning_laps

theorem remaining_laps_after_sunday_morning :
  total_required_laps - laps_before_sunday_break = 58 := by
  sorry

end NUMINAMATH_CALUDE_remaining_laps_after_sunday_morning_l2705_270543


namespace NUMINAMATH_CALUDE_no_special_primes_l2705_270507

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def digit_swap (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10)

theorem no_special_primes :
  ∀ n : ℕ, 13 ≤ n → n < 100 →
    is_prime n →
    is_prime (digit_swap n) →
    is_prime (digit_sum n) →
    False :=
sorry

end NUMINAMATH_CALUDE_no_special_primes_l2705_270507


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2705_270575

/-- The cubic function f(x) = x^3 - 12x + 12 -/
def f (x : ℝ) : ℝ := x^3 - 12*x + 12

theorem cubic_function_properties :
  (∃ x : ℝ, f x = 28) ∧  -- Maximum value is 28
  (f 2 = -4) ∧           -- Extreme value at x = 2 is -4
  (∀ x ∈ Set.Icc (-3) 3, f x ≥ -4) ∧  -- Minimum value on [-3, 3] is -4
  (∃ x ∈ Set.Icc (-3) 3, f x = -4) -- The minimum is attained on [-3, 3]
  := by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l2705_270575


namespace NUMINAMATH_CALUDE_prime_sum_gcd_ratio_l2705_270502

theorem prime_sum_gcd_ratio (n : ℕ) (p : ℕ) (hp : Prime p) (h_p : p = 2 * n - 1) 
  (a : Fin n → ℕ+) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ i j : Fin n, i ≠ j ∧ (a i + a j : ℕ) / Nat.gcd (a i) (a j) ≥ p := by
sorry

end NUMINAMATH_CALUDE_prime_sum_gcd_ratio_l2705_270502


namespace NUMINAMATH_CALUDE_traffic_light_problem_l2705_270592

/-- A sequence of independent events with a fixed probability -/
structure EventSequence where
  n : ℕ  -- number of events
  p : ℝ  -- probability of each event occurring
  indep : Bool  -- events are independent

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (k : ℕ) (es : EventSequence) : ℝ :=
  (es.n.choose k) * (es.p ^ k) * ((1 - es.p) ^ (es.n - k))

/-- The expected value of a binomial distribution -/
def binomial_expectation (es : EventSequence) : ℝ :=
  es.n * es.p

/-- The variance of a binomial distribution -/
def binomial_variance (es : EventSequence) : ℝ :=
  es.n * es.p * (1 - es.p)

theorem traffic_light_problem (es : EventSequence) 
  (h1 : es.n = 6) (h2 : es.p = 1/3) (h3 : es.indep = true) :
  (binomial_pmf 1 {n := 3, p := 1/3, indep := true} = 4/27) ∧
  (binomial_expectation es = 2) ∧
  (binomial_variance es = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_problem_l2705_270592


namespace NUMINAMATH_CALUDE_sodium_bisulfite_moles_l2705_270529

-- Define the molecules and their molar quantities
structure Reaction :=
  (NaHSO3 : ℝ)  -- moles of Sodium bisulfite
  (HCl : ℝ)     -- moles of Hydrochloric acid
  (H2O : ℝ)     -- moles of Water

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.NaHSO3 = r.HCl ∧ r.NaHSO3 = r.H2O

-- Theorem statement
theorem sodium_bisulfite_moles :
  ∀ r : Reaction,
  r.HCl = 1 →        -- 1 mole of Hydrochloric acid is used
  r.H2O = 1 →        -- The reaction forms 1 mole of Water
  balanced_equation r →  -- The reaction equation is balanced
  r.NaHSO3 = 1 :=    -- The number of moles of Sodium bisulfite is 1
by
  sorry

end NUMINAMATH_CALUDE_sodium_bisulfite_moles_l2705_270529


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2705_270544

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant (a : ℝ) : second_quadrant (-1) (a^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2705_270544


namespace NUMINAMATH_CALUDE_roses_recipients_l2705_270586

/-- Given Ricky's initial number of roses, the number of roses stolen, and the number of roses
    per person, calculate the number of people who will receive roses. -/
def number_of_recipients (initial_roses : ℕ) (stolen_roses : ℕ) (roses_per_person : ℕ) : ℕ :=
  (initial_roses - stolen_roses) / roses_per_person

/-- Theorem stating that given the specific values in the problem, 
    the number of people who will receive roses is 9. -/
theorem roses_recipients : 
  number_of_recipients 40 4 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_roses_recipients_l2705_270586


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2705_270516

theorem absolute_value_equality (x : ℝ) (h : x < -2) :
  |x - Real.sqrt ((x + 2)^2)| = -2*x - 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2705_270516


namespace NUMINAMATH_CALUDE_binomial_sum_one_l2705_270563

theorem binomial_sum_one (a : ℝ) (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (a*x - 1)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₃ = 80 →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_one_l2705_270563


namespace NUMINAMATH_CALUDE_q_min_at_2_l2705_270513

-- Define the function q
def q (x : ℝ) : ℝ := (x - 5)^2 + (x - 2)^2 - 6

-- State the theorem
theorem q_min_at_2 : 
  ∀ x : ℝ, q x ≥ q 2 :=
sorry

end NUMINAMATH_CALUDE_q_min_at_2_l2705_270513
