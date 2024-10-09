import Mathlib

namespace firm_partners_initial_count_l354_35481

theorem firm_partners_initial_count
  (x : ℕ)
  (h1 : 2*x/(63*x + 35) = 1/34)
  (h2 : 2*x/(20*x + 10) = 1/15) :
  2*x = 14 :=
by
  sorry

end firm_partners_initial_count_l354_35481


namespace solve_for_a_l354_35427

open Complex

theorem solve_for_a (a : ℝ) (h : ∃ x : ℝ, (2 * Complex.I - (a * Complex.I) / (1 - Complex.I) = x)) : a = 4 := 
sorry

end solve_for_a_l354_35427


namespace candy_cost_l354_35412

-- Definitions and assumptions from problem conditions
def cents_per_page := 1
def pages_per_book := 150
def books_read := 12
def leftover_cents := 300  -- $3 in cents

-- Total pages read
def total_pages_read := pages_per_book * books_read

-- Total earnings in cents
def total_cents_earned := total_pages_read * cents_per_page

-- Cost of the candy in cents
def candy_cost_cents := total_cents_earned - leftover_cents

-- Theorem statement
theorem candy_cost : candy_cost_cents = 1500 := 
  by 
    -- proof goes here
    sorry

end candy_cost_l354_35412


namespace round_robin_games_l354_35488

theorem round_robin_games (x : ℕ) (h : 45 = (1 / 2) * x * (x - 1)) : (1 / 2) * x * (x - 1) = 45 :=
sorry

end round_robin_games_l354_35488


namespace local_minimum_at_1_1_l354_35407

noncomputable def function (x y : ℝ) : ℝ :=
  x^3 + y^3 - 3 * x * y

theorem local_minimum_at_1_1 : 
  ∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ (∀ (z : ℝ), z = function x y → z = -1) :=
sorry

end local_minimum_at_1_1_l354_35407


namespace sum_2001_and_1015_l354_35498

theorem sum_2001_and_1015 :
  2001 + 1015 = 3016 :=
sorry

end sum_2001_and_1015_l354_35498


namespace range_of_a_l354_35439

def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

noncomputable def roots (a : ℝ) : (ℝ × ℝ) :=
  (1, 3)

noncomputable def f_max (a : ℝ) :=
  -a

theorem range_of_a (a b c : ℝ) 
  (h1 : ∀ x, quadratic_function a b c x < 0 ↔ (x < 1 ∨ 3 < x))
  (h2 : f_max a < 2) : 
  -2 < a ∧ a < 0 :=
sorry

end range_of_a_l354_35439


namespace infinite_perfect_squares_in_arithmetic_sequence_l354_35438

theorem infinite_perfect_squares_in_arithmetic_sequence 
  (a d : ℕ) 
  (h_exists_perfect_square : ∃ (n₀ k : ℕ), a + n₀ * d = k^2) 
  : ∃ (S : ℕ → ℕ), (∀ n, ∃ t, S n = a + t * d ∧ ∃ k, S n = k^2) ∧ (∀ m n, S m = S n → m = n) :=
sorry

end infinite_perfect_squares_in_arithmetic_sequence_l354_35438


namespace solution_of_system_l354_35446

theorem solution_of_system :
  ∃ x y : ℝ, (x^4 + y^4 = 17) ∧ (x + y = 3) ∧ ((x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1)) :=
by
  sorry

end solution_of_system_l354_35446


namespace value_of_x_squared_plus_y_squared_l354_35462

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l354_35462


namespace marked_price_of_jacket_l354_35432

variable (x : ℝ) -- Define the variable x as a real number representing the marked price.

-- Define the conditions as a Lean theorem statement
theorem marked_price_of_jacket (cost price_sold profit : ℝ) (h1 : cost = 350) (h2 : price_sold = 0.8 * x) (h3 : profit = price_sold - cost) : 
  x = 550 :=
by
  -- We would solve the proof here using provided conditions
  sorry

end marked_price_of_jacket_l354_35432


namespace find_winner_votes_l354_35415

-- Define the conditions
variables (V : ℝ) (winner_votes second_votes : ℝ)
def election_conditions :=
  winner_votes = 0.468 * V ∧
  second_votes = 0.326 * V ∧
  winner_votes - second_votes = 752

-- State the theorem
theorem find_winner_votes (h : election_conditions V winner_votes second_votes) :
  winner_votes = 2479 :=
sorry

end find_winner_votes_l354_35415


namespace total_students_at_year_end_l354_35483

def initial_students : ℝ := 10.0
def added_students : ℝ := 4.0
def new_students : ℝ := 42.0

theorem total_students_at_year_end : initial_students + added_students + new_students = 56.0 :=
by
  sorry

end total_students_at_year_end_l354_35483


namespace minimize_quadratic_expression_l354_35450

theorem minimize_quadratic_expression :
  ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, (y^2 - 6*y + 8) ≥ (x^2 - 6*x + 8) := by
sorry

end minimize_quadratic_expression_l354_35450


namespace unfolded_side_view_of_cone_is_sector_l354_35494

theorem unfolded_side_view_of_cone_is_sector 
  (shape : Type)
  (curved_side : shape)
  (straight_side1 : shape)
  (straight_side2 : shape) 
  (condition1 : ∃ (s : shape), s = curved_side) 
  (condition2 : ∃ (s1 s2 : shape), s1 = straight_side1 ∧ s2 = straight_side2)
  : shape = sector :=
sorry

end unfolded_side_view_of_cone_is_sector_l354_35494


namespace work_rate_proof_l354_35464

def combined_rate (a b c : ℚ) : ℚ := a + b + c

def inv (x : ℚ) : ℚ := 1 / x

theorem work_rate_proof (A B C : ℚ) (h₁ : A + B = 1/15) (h₂ : C = 1/10) :
  inv (combined_rate A B C) = 6 :=
by
  sorry

end work_rate_proof_l354_35464


namespace reading_schedule_correct_l354_35499

-- Defining the conditions
def total_words : ℕ := 34685
def words_day1 (x : ℕ) : ℕ := x
def words_day2 (x : ℕ) : ℕ := 2 * x
def words_day3 (x : ℕ) : ℕ := 4 * x

-- Defining the main statement of the problem
theorem reading_schedule_correct (x : ℕ) : 
  words_day1 x + words_day2 x + words_day3 x = total_words := 
sorry

end reading_schedule_correct_l354_35499


namespace fixed_point_at_5_75_l354_35491

-- Defining the function
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 + k * x - 5 * k

-- Stating the theorem that the graph passes through the fixed point (5, 75)
theorem fixed_point_at_5_75 (k : ℝ) : quadratic_function k 5 = 75 := by
  sorry

end fixed_point_at_5_75_l354_35491


namespace election_votes_l354_35436

theorem election_votes (V : ℝ) (ha : 0.45 * V = 4860)
                       (hb : 0.30 * V = 3240)
                       (hc : 0.20 * V = 2160)
                       (hd : 0.05 * V = 540)
                       (hmaj : (0.45 - 0.30) * V = 1620) :
                       V = 10800 :=
by
  sorry

end election_votes_l354_35436


namespace range_of_a_l354_35443

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a-1)*x^2 + a*x + 1 ≥ 0) : a ≥ 1 :=
by {
  sorry
}

end range_of_a_l354_35443


namespace part_i_part_ii_part_iii_l354_35468

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + f (x + 3 * Real.pi / 4)

theorem part_i : f (Real.pi / 2) = 1 :=
sorry

theorem part_ii : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi :=
sorry

theorem part_iii : ∃ x, g x = -2 :=
sorry

end part_i_part_ii_part_iii_l354_35468


namespace prime_implies_power_of_two_l354_35474

-- Conditions:
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

-- Problem:
theorem prime_implies_power_of_two (n : ℕ) (h : is_prime (2^n + 1)) : ∃ k : ℕ, n = 2^k := sorry

end prime_implies_power_of_two_l354_35474


namespace find_y_eq_54_div_23_l354_35405

open BigOperators

theorem find_y_eq_54_div_23 (y : ℚ) (h : (Real.sqrt (8 * y) / Real.sqrt (6 * (y - 2))) = 3) : y = 54 / 23 := 
by
  sorry

end find_y_eq_54_div_23_l354_35405


namespace find_k_l354_35423

   theorem find_k (m n : ℝ) (k : ℝ) (hm : m > 0) (hn : n > 0)
     (h1 : k = Real.log m / Real.log 2)
     (h2 : k = Real.log n / (Real.log 4))
     (h3 : k = Real.log (4 * m + 3 * n) / (Real.log 8)) :
     k = 2 :=
   by
     sorry
   
end find_k_l354_35423


namespace angus_tokens_count_l354_35406

def worth_of_token : ℕ := 4
def elsa_tokens : ℕ := 60
def difference_worth : ℕ := 20

def elsa_worth : ℕ := elsa_tokens * worth_of_token
def angus_worth : ℕ := elsa_worth - difference_worth

def angus_tokens : ℕ := angus_worth / worth_of_token

theorem angus_tokens_count : angus_tokens = 55 := by
  sorry

end angus_tokens_count_l354_35406


namespace part_I_part_II_l354_35447

noncomputable def f_I (x : ℝ) : ℝ := abs (3*x - 1) + abs (x + 3)

theorem part_I :
  ∀ x : ℝ, f_I x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 1/2 :=
by sorry

noncomputable def f_II (x b c : ℝ) : ℝ := abs (x - b) + abs (x + c)

theorem part_II :
  ∀ b c : ℝ, b > 0 → c > 0 → b + c = 1 → 
  (∀ x : ℝ, f_II x b c ≥ 1) → (1 / b + 1 / c = 4) :=
by sorry

end part_I_part_II_l354_35447


namespace ratio_of_cream_l354_35482

def initial_coffee := 18
def cup_capacity := 22
def Emily_drank := 3
def Emily_added_cream := 4
def Ethan_added_cream := 4
def Ethan_drank := 3

noncomputable def cream_in_Emily := Emily_added_cream

noncomputable def cream_remaining_in_Ethan :=
  Ethan_added_cream - (Ethan_added_cream * Ethan_drank / (initial_coffee + Ethan_added_cream))

noncomputable def resulting_ratio := cream_in_Emily / cream_remaining_in_Ethan

theorem ratio_of_cream :
  resulting_ratio = 200 / 173 :=
by
  sorry

end ratio_of_cream_l354_35482


namespace sum_of_odd_integers_less_than_50_l354_35484

def sumOddIntegersLessThan (n : Nat) : Nat :=
  List.sum (List.filter (λ x => x % 2 = 1) (List.range n))

theorem sum_of_odd_integers_less_than_50 : sumOddIntegersLessThan 50 = 625 :=
  by
    sorry

end sum_of_odd_integers_less_than_50_l354_35484


namespace intersection_complement_eq_l354_35417

-- Define the universal set U, and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Define the complement of B in U
def complement_B_in_U : Set ℕ := { x ∈ U | x ∉ B }

-- The main theorem statement stating the required equality
theorem intersection_complement_eq : A ∩ complement_B_in_U = {2, 3} := by
  sorry

end intersection_complement_eq_l354_35417


namespace minimize_cost_per_km_l354_35493

section ship_cost_minimization

variables (u v k : ℝ) (fuel_cost other_cost total_cost_per_km: ℝ)

-- Condition 1: The fuel cost per unit time is directly proportional to the cube of its speed.
def fuel_cost_eq : Prop := u = k * v^3

-- Condition 2: When the speed of the ship is 10 km/h, the fuel cost is 35 yuan per hour.
def fuel_cost_at_10 : Prop := u = 35 ∧ v = 10

-- Condition 3: The other costs are 560 yuan per hour.
def other_cost_eq : Prop := other_cost = 560

-- Condition 4: The maximum speed of the ship is 25 km/h.
def max_speed : Prop := v ≤ 25

-- Prove that the speed of the ship that minimizes the cost per kilometer is 20 km/h.
theorem minimize_cost_per_km : 
  fuel_cost_eq u v k ∧ fuel_cost_at_10 u v ∧ other_cost_eq other_cost ∧ max_speed v → v = 20 :=
by
  sorry

end ship_cost_minimization

end minimize_cost_per_km_l354_35493


namespace max_regions_1002_1000_l354_35457

def regions_through_point (n : ℕ) : ℕ := (n * (n + 1)) / 2 + 1

def max_regions (a b : ℕ) : ℕ := 
  let rB := regions_through_point b
  let first_line_through_A := rB + b + 1
  let remaining_lines_through_A := (a - 1) * (b + 2)
  first_line_through_A + remaining_lines_through_A

theorem max_regions_1002_1000 : max_regions 1002 1000 = 1504503 := by
  sorry

end max_regions_1002_1000_l354_35457


namespace A_salary_is_3000_l354_35495

theorem A_salary_is_3000 
    (x y : ℝ) 
    (h1 : x + y = 4000)
    (h2 : 0.05 * x = 0.15 * y) 
    : x = 3000 := by
  sorry

end A_salary_is_3000_l354_35495


namespace nina_has_9_times_more_reading_homework_l354_35402

theorem nina_has_9_times_more_reading_homework
  (ruby_math_homework : ℕ)
  (ruby_reading_homework : ℕ)
  (nina_total_homework : ℕ)
  (nina_math_homework_factor : ℕ)
  (h1 : ruby_math_homework = 6)
  (h2 : ruby_reading_homework = 2)
  (h3 : nina_total_homework = 48)
  (h4 : nina_math_homework_factor = 4) :
  nina_total_homework - (ruby_math_homework * (nina_math_homework_factor + 1)) = 9 * ruby_reading_homework := by
  sorry

end nina_has_9_times_more_reading_homework_l354_35402


namespace partitions_equiv_l354_35475

-- Definition of partitions into distinct integers
def a (n : ℕ) : ℕ := sorry  -- Placeholder for the actual definition or count function

-- Definition of partitions into odd integers
def b (n : ℕ) : ℕ := sorry  -- Placeholder for the actual definition or count function

-- Theorem stating that the number of partitions into distinct integers equals the number of partitions into odd integers
theorem partitions_equiv (n : ℕ) : a n = b n :=
sorry

end partitions_equiv_l354_35475


namespace jane_emily_total_accessories_l354_35490

def total_accessories : ℕ :=
  let jane_dresses := 4 * 10
  let emily_dresses := 3 * 8
  let jane_ribbons := 3 * jane_dresses
  let jane_buttons := 2 * jane_dresses
  let jane_lace_trims := 1 * jane_dresses
  let jane_beads := 4 * jane_dresses
  let emily_ribbons := 2 * emily_dresses
  let emily_buttons := 3 * emily_dresses
  let emily_lace_trims := 2 * emily_dresses
  let emily_beads := 5 * emily_dresses
  let emily_bows := 1 * emily_dresses
  jane_ribbons + jane_buttons + jane_lace_trims + jane_beads +
  emily_ribbons + emily_buttons + emily_lace_trims + emily_beads + emily_bows 

theorem jane_emily_total_accessories : total_accessories = 712 := 
by
  sorry

end jane_emily_total_accessories_l354_35490


namespace find_integers_satisfying_equation_l354_35400

theorem find_integers_satisfying_equation :
  ∃ (a b c : ℤ), (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = 0 ∧ b = 1 ∧ c = 0) ∨ (a = 0 ∧ b = 0 ∧ c = 1) ∨
                  (a = 2 ∧ b = -1 ∧ c = -1) ∨ (a = -1 ∧ b = 2 ∧ c = -1) ∨ (a = -1 ∧ b = -1 ∧ c = 2)
  ↔ (∃ (a b c : ℤ), 1 / 2 * (a + b) * (b + c) * (c + a) + (a + b + c) ^ 3 = 1 - a * b * c) := sorry

end find_integers_satisfying_equation_l354_35400


namespace mom_age_when_Jayson_born_l354_35425

theorem mom_age_when_Jayson_born
  (Jayson_age : ℕ)
  (Dad_age : ℕ)
  (Mom_age : ℕ)
  (H1 : Jayson_age = 10)
  (H2 : Dad_age = 4 * Jayson_age)
  (H3 : Mom_age = Dad_age - 2) :
  Mom_age - Jayson_age = 28 := by
  sorry

end mom_age_when_Jayson_born_l354_35425


namespace adults_had_meal_l354_35453

theorem adults_had_meal (A : ℕ) (h1 : 70 ≥ A) (h2 : ((70 - A) * 9) = (72 * 7)) : A = 14 := 
by
  sorry

end adults_had_meal_l354_35453


namespace percentage_problem_l354_35471

variable (y x z : ℝ)

def A := y * x^2 + 3 * z - 6

theorem percentage_problem (h : A y x z > 0) :
  (2 * A y x z / 5) + (3 * A y x z / 10) = (70 / 100) * A y x z :=
by
  sorry

end percentage_problem_l354_35471


namespace time_per_lice_check_l354_35445

-- Define the number of students in each grade
def kindergartners := 26
def first_graders := 19
def second_graders := 20
def third_graders := 25

-- Define the total number of students
def total_students := kindergartners + first_graders + second_graders + third_graders

-- Define the total time in minutes
def hours := 3
def minutes_per_hour := 60
def total_minutes := hours * minutes_per_hour

-- Define the correct answer for time per check
def time_per_check := total_minutes / total_students

-- Prove that the time for each check is 2 minutes
theorem time_per_lice_check : time_per_check = 2 := 
by
  sorry

end time_per_lice_check_l354_35445


namespace river_lengths_l354_35459

theorem river_lengths (x : ℝ) (dnieper don : ℝ)
  (h1 : dnieper = (5 / (19 / 3)) * x)
  (h2 : don = (6.5 / 9.5) * x)
  (h3 : dnieper - don = 300) :
  x = 2850 ∧ dnieper = 2250 ∧ don = 1950 :=
by
  sorry

end river_lengths_l354_35459


namespace inradius_of_triangle_l354_35477

theorem inradius_of_triangle (A p r s : ℝ) (h1 : A = 3 * p) (h2 : A = r * s) (h3 : s = p / 2) :
  r = 6 :=
by
  sorry

end inradius_of_triangle_l354_35477


namespace cos_60_eq_one_half_l354_35409

theorem cos_60_eq_one_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end cos_60_eq_one_half_l354_35409


namespace value_a7_l354_35418

variables {a : ℕ → ℝ}

-- Condition 1: Arithmetic sequence where each term is non-zero
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variable (h1 : arithmetic_sequence a)
-- Condition 2: 2a_3 - a_1^2 + 2a_11 = 0
variable (h2 : 2 * a 3 - (a 1)^2 + 2 * a 11 = 0)
-- Condition 3: a_3 + a_11 = 2a_7
variable (h3 : a 3 + a 11 = 2 * a 7)

theorem value_a7 : a 7 = 4 := by
  sorry

end value_a7_l354_35418


namespace find_k_from_given_solution_find_other_root_l354_35403

-- Given
def one_solution_of_first_eq_is_same_as_second (x k : ℝ) : Prop :=
  x^2 + k * x - 2 = 0 ∧ (x + 1) / (x - 1) = 3

-- To find k
theorem find_k_from_given_solution : ∃ k : ℝ, ∃ x : ℝ, one_solution_of_first_eq_is_same_as_second x k ∧ k = -1 := by
  sorry

-- To find the other root
theorem find_other_root : ∃ x2 : ℝ, (x2 = -1) := by
  sorry

end find_k_from_given_solution_find_other_root_l354_35403


namespace range_of_m_l354_35435

theorem range_of_m (m : ℝ) 
  (h : ∀ x : ℝ, 0 < x → m * x^2 + 2 * x + m ≤ 0) : m ≤ -1 :=
sorry

end range_of_m_l354_35435


namespace delta_maximum_success_ratio_l354_35433

theorem delta_maximum_success_ratio (x y z w : ℕ) (h1 : 0 < x ∧ x * 5 < y * 3)
    (h2 : 0 < z ∧ z * 5 < w * 3) (h3 : y + w = 600) :
    (x + z) / 600 ≤ 359 / 600 :=
by
  sorry

end delta_maximum_success_ratio_l354_35433


namespace inequality_proof_l354_35440

theorem inequality_proof 
  {a b c : ℝ}
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c)
  (h1 : a^2 ≤ b^2 + c^2)
  (h2 : b^2 ≤ c^2 + a^2)
  (h3 : c^2 ≤ a^2 + b^2) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) :=
sorry

end inequality_proof_l354_35440


namespace jonah_added_yellow_raisins_l354_35420

variable (y : ℝ)

theorem jonah_added_yellow_raisins (h : y + 0.4 = 0.7) : y = 0.3 := by
  sorry

end jonah_added_yellow_raisins_l354_35420


namespace solve_absolute_value_inequality_l354_35455

theorem solve_absolute_value_inequality (x : ℝ) :
  3 ≤ |x + 3| ∧ |x + 3| ≤ 7 ↔ (-10 ≤ x ∧ x ≤ -6) ∨ (0 ≤ x ∧ x ≤ 4) :=
by
  sorry

end solve_absolute_value_inequality_l354_35455


namespace time_to_write_numbers_in_minutes_l354_35463

theorem time_to_write_numbers_in_minutes : 
  (1 * 5 + 2 * (99 - 10 + 1) + 3 * (105 - 100 + 1)) / 60 = 4 := 
  by
  -- Calculation steps would go here
  sorry

end time_to_write_numbers_in_minutes_l354_35463


namespace divisible_by_42_l354_35431

theorem divisible_by_42 (n : ℕ) : 42 ∣ (n^3 * (n^6 - 1)) :=
sorry

end divisible_by_42_l354_35431


namespace find_b_l354_35410

noncomputable def f (b x : ℝ) : ℝ :=
if x < 1 then 2 * x - b else 2 ^ x

theorem find_b (b : ℝ) (h : f b (f b (1 / 2)) = 4) : b = -1 :=
sorry

end find_b_l354_35410


namespace symmetric_conic_transform_l354_35487

open Real

theorem symmetric_conic_transform (x y : ℝ) 
  (h1 : 2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0)
  (h2 : x - y + 1 = 0) : 
  5 * x^2 + 4 * x * y + 2 * y^2 + 6 * x - 19 = 0 := 
sorry

end symmetric_conic_transform_l354_35487


namespace symmetric_origin_coordinates_l354_35486

def symmetric_coordinates (x y : ℚ) (x_line y_line : ℚ) : Prop :=
  x_line - 2 * y_line + 2 = 0 ∧ y_line = -2 * x_line ∧ x = -4/5 ∧ y = 8/5

theorem symmetric_origin_coordinates :
  ∃ (x_0 y_0 : ℚ), symmetric_coordinates x_0 y_0 (-4/5) (8/5) :=
by
  use -4/5, 8/5
  sorry

end symmetric_origin_coordinates_l354_35486


namespace determine_n_l354_35401

-- Constants and variables
variables {a : ℕ → ℝ} {n : ℕ}

-- Definition for the condition at each vertex
def vertex_condition (a : ℕ → ℝ) (i : ℕ) : Prop :=
  a i = a (i - 1) * a (i + 1)

-- Mathematical problem statement
theorem determine_n (h : ∀ i, vertex_condition a i) (distinct_a : ∀ i j, a i ≠ a j) : n = 6 :=
sorry

end determine_n_l354_35401


namespace regular_polygon_sides_l354_35430

theorem regular_polygon_sides (n : ℕ) (h : ∀ i < n, (interior_angle_i : ℝ) = 150) :
  (n = 12) :=
by
  sorry

end regular_polygon_sides_l354_35430


namespace bmws_sold_l354_35458

-- Definitions stated by the problem:
def total_cars : ℕ := 300
def percentage_mercedes : ℝ := 0.20
def percentage_toyota : ℝ := 0.25
def percentage_nissan : ℝ := 0.10
def percentage_bmws : ℝ := 1 - (percentage_mercedes + percentage_toyota + percentage_nissan)

-- Statement to prove:
theorem bmws_sold : (total_cars : ℝ) * percentage_bmws = 135 := by
  sorry

end bmws_sold_l354_35458


namespace octopus_shoes_needed_l354_35461

-- Defining the basic context: number of legs and current shod legs
def num_legs : ℕ := 8

-- Conditions based on the number of already shod legs for each member
def father_shod_legs : ℕ := num_legs / 2       -- Father-octopus has half of his legs shod
def mother_shod_legs : ℕ := 3                  -- Mother-octopus has 3 legs shod
def son_shod_legs : ℕ := 6                     -- Each son-octopus has 6 legs shod
def num_sons : ℕ := 2                          -- There are 2 sons

-- Calculate unshod legs for each 
def father_unshod_legs : ℕ := num_legs - father_shod_legs
def mother_unshod_legs : ℕ := num_legs - mother_shod_legs
def son_unshod_legs : ℕ := num_legs - son_shod_legs

-- Aggregate the total shoes needed based on unshod legs
def total_shoes_needed : ℕ :=
  father_unshod_legs + 
  mother_unshod_legs + 
  (son_unshod_legs * num_sons)

-- The theorem to prove
theorem octopus_shoes_needed : total_shoes_needed = 13 := 
  by 
    sorry

end octopus_shoes_needed_l354_35461


namespace perimeter_of_resulting_figure_l354_35456

def side_length := 100
def original_square_perimeter := 4 * side_length
def rectangle_width := side_length
def rectangle_height := side_length / 2
def number_of_longer_sides_of_rectangles_touching := 4

theorem perimeter_of_resulting_figure :
  let new_perimeter := 3 * side_length + number_of_longer_sides_of_rectangles_touching * rectangle_height
  new_perimeter = 500 :=
by
  sorry

end perimeter_of_resulting_figure_l354_35456


namespace range_of_a_l354_35480

open Set

def p (a : ℝ) := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def q (a : ℝ) := ∀ x : ℝ, x ∈ (Icc 1 2) → x^2 ≥ a

theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ (Ioo 1 2 ∪ Iic (-2)) :=
by sorry

end range_of_a_l354_35480


namespace count_valid_subsets_l354_35492

open Set

theorem count_valid_subsets :
  ∀ (A : Set ℕ), (A ⊆ {1, 2, 3, 4, 5, 6, 7}) → 
  (∀ (a : ℕ), a ∈ A → (8 - a) ∈ A) → A ≠ ∅ → 
  ∃! (n : ℕ), n = 15 :=
  by
    sorry

end count_valid_subsets_l354_35492


namespace grey_area_of_first_grid_is_16_grey_area_of_second_grid_is_15_white_area_of_third_grid_is_5_l354_35437

theorem grey_area_of_first_grid_is_16 (side_length : ℝ := 1) :
  let area_triangle (base height : ℝ) := 0.5 * base * height
  let area_rectangle (length width : ℝ) := length * width
  let grey_area := area_triangle 3 side_length 
                    + area_triangle 4 side_length 
                    + area_rectangle 6 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_rectangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 3 side_length
  grey_area = 16 := by
  sorry

theorem grey_area_of_second_grid_is_15 (side_length : ℝ := 1) :
  let area_triangle (base height : ℝ) := 0.5 * base * height
  let area_rectangle (length width : ℝ) := length * width
  let grey_area := area_triangle 4 side_length 
                    + area_rectangle 2 side_length
                    + area_triangle 6 side_length 
                    + area_rectangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_rectangle 4 side_length
  grey_area = 15 := by
  sorry

theorem white_area_of_third_grid_is_5 (total_rectangle_area dark_grey_area : ℝ) (grey_area1 grey_area2 : ℝ) :
    total_rectangle_area = 32 ∧ dark_grey_area = 4 ∧ grey_area1 = 16 ∧ grey_area2 = 15 →
    let total_grey_area_recounted := grey_area1 + grey_area2 - dark_grey_area
    let white_area := total_rectangle_area - total_grey_area_recounted
    white_area = 5 := by
  sorry

end grey_area_of_first_grid_is_16_grey_area_of_second_grid_is_15_white_area_of_third_grid_is_5_l354_35437


namespace tear_paper_l354_35441

theorem tear_paper (n : ℕ) : 1 + 3 * n ≠ 2007 :=
by
  sorry

end tear_paper_l354_35441


namespace points_on_circle_l354_35442

theorem points_on_circle (t : ℝ) (ht : t ≠ 0) :
  let x := (t + 1) / t ^ 2
  let y := (t - 1) / t ^ 2
  (x - 2)^2 + (y - 2)^2 = 4 :=
by
  let x := (t + 1) / t ^ 2
  let y := (t - 1) / t ^ 2
  sorry

end points_on_circle_l354_35442


namespace intersection_P_compl_M_l354_35428

-- Define universal set U
def U : Set ℤ := Set.univ

-- Define set M
def M : Set ℤ := {1, 2}

-- Define set P
def P : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the complement of M in U
def M_compl : Set ℤ := { x | x ∉ M }

-- Define the intersection of P and the complement of M
def P_inter_M_compl : Set ℤ := P ∩ M_compl

-- The theorem we want to prove
theorem intersection_P_compl_M : P_inter_M_compl = {-2, -1, 0} := 
by {
  sorry
}

end intersection_P_compl_M_l354_35428


namespace interval_length_l354_35469

theorem interval_length (c d : ℝ) (h : ∃ x : ℝ, c ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ d)
  (length : (d - 4) / 3 - (c - 4) / 3 = 15) : d - c = 45 :=
by
  sorry

end interval_length_l354_35469


namespace floor_plus_r_eq_10_3_implies_r_eq_5_3_l354_35460

noncomputable def floor (x : ℝ) : ℤ := sorry -- Assuming the function exists

theorem floor_plus_r_eq_10_3_implies_r_eq_5_3 (r : ℝ) 
  (h : floor r + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_r_eq_10_3_implies_r_eq_5_3_l354_35460


namespace girls_divisible_by_nine_l354_35448

def total_students (m c d u : ℕ) : ℕ := 1000 * m + 100 * c + 10 * d + u
def number_of_boys (m c d u : ℕ) : ℕ := m + c + d + u
def number_of_girls (m c d u : ℕ) : ℕ := total_students m c d u - number_of_boys m c d u 

theorem girls_divisible_by_nine (m c d u : ℕ) : 
  number_of_girls m c d u % 9 = 0 := 
by
    sorry

end girls_divisible_by_nine_l354_35448


namespace simplify_expression_l354_35473

theorem simplify_expression : 8 * (15 / 9) * (-45 / 40) = -1 :=
  by
  sorry

end simplify_expression_l354_35473


namespace max_value_g_l354_35404

noncomputable def g (x : ℝ) := 4 * x - x ^ 4

theorem max_value_g : 
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.sqrt 4 ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ Real.sqrt 4 → g y ≤ 3 :=
sorry

end max_value_g_l354_35404


namespace hours_per_day_in_deliberation_l354_35478

noncomputable def jury_selection_days : ℕ := 2
noncomputable def trial_days : ℕ := 4 * jury_selection_days
noncomputable def total_deliberation_hours : ℕ := 6 * 24
noncomputable def total_days_on_jury_duty : ℕ := 19

theorem hours_per_day_in_deliberation :
  (total_deliberation_hours / (total_days_on_jury_duty - (jury_selection_days + trial_days))) = 16 :=
by
  sorry

end hours_per_day_in_deliberation_l354_35478


namespace whale_consumption_l354_35421

-- Define the conditions
def first_hour_consumption (x : ℕ) := x
def second_hour_consumption (x : ℕ) := x + 3
def third_hour_consumption (x : ℕ) := x + 6
def fourth_hour_consumption (x : ℕ) := x + 9
def fifth_hour_consumption (x : ℕ) := x + 12
def sixth_hour_consumption (x : ℕ) := x + 15
def seventh_hour_consumption (x : ℕ) := x + 18
def eighth_hour_consumption (x : ℕ) := x + 21
def ninth_hour_consumption (x : ℕ) := x + 24

def total_consumed (x : ℕ) := 
  first_hour_consumption x + 
  second_hour_consumption x + 
  third_hour_consumption x + 
  fourth_hour_consumption x + 
  fifth_hour_consumption x + 
  sixth_hour_consumption x + 
  seventh_hour_consumption x + 
  eighth_hour_consumption x + 
  ninth_hour_consumption x

-- Prove that the total sum consumed equals 540
theorem whale_consumption : ∃ x : ℕ, total_consumed x = 540 ∧ sixth_hour_consumption x = 63 :=
by
  sorry

end whale_consumption_l354_35421


namespace zoe_total_songs_l354_35444

def total_songs (country_albums pop_albums songs_per_country_album songs_per_pop_album : ℕ) : ℕ :=
  country_albums * songs_per_country_album + pop_albums * songs_per_pop_album

theorem zoe_total_songs :
  total_songs 4 7 5 6 = 62 :=
by
  sorry

end zoe_total_songs_l354_35444


namespace quadratic_solution_l354_35485

theorem quadratic_solution (x : ℝ) : (x^2 + 6 * x + 8 = -2 * (x + 4) * (x + 5)) ↔ (x = -8 ∨ x = -4) :=
by
  sorry

end quadratic_solution_l354_35485


namespace slices_per_pie_l354_35411

variable (S : ℕ) -- Let S be the number of slices per pie

theorem slices_per_pie (h1 : 5 * S * 9 = 180) : S = 4 := by
  sorry

end slices_per_pie_l354_35411


namespace runners_meet_again_l354_35419

-- Definitions based on the problem conditions
def track_length : ℝ := 500 
def speed_runner1 : ℝ := 4.4
def speed_runner2 : ℝ := 4.8
def speed_runner3 : ℝ := 5.0

-- The time at which runners meet again at the starting point
def time_when_runners_meet : ℝ := 2500

theorem runners_meet_again :
  ∀ t : ℝ, t = time_when_runners_meet → 
  (∀ n1 n2 n3 : ℤ, 
    ∃ k : ℤ, 
    speed_runner1 * t = n1 * track_length ∧ 
    speed_runner2 * t = n2 * track_length ∧ 
    speed_runner3 * t = n3 * track_length) :=
by 
  sorry

end runners_meet_again_l354_35419


namespace probability_at_least_one_hit_l354_35479

-- Define probabilities of each shooter hitting the target
def P_A : ℚ := 1 / 2
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Define the complementary probabilities (each shooter misses the target)
def P_A_miss : ℚ := 1 - P_A
def P_B_miss : ℚ := 1 - P_B
def P_C_miss : ℚ := 1 - P_C

-- Calculate the probability of all shooters missing the target
def P_all_miss : ℚ := P_A_miss * P_B_miss * P_C_miss

-- Calculate the probability of at least one shooter hitting the target
def P_at_least_one_hit : ℚ := 1 - P_all_miss

-- The theorem to be proved
theorem probability_at_least_one_hit : 
  P_at_least_one_hit = 3 / 4 := 
by sorry

end probability_at_least_one_hit_l354_35479


namespace homework_duration_decrease_l354_35422

variable (a b x : ℝ)

theorem homework_duration_decrease (h: a * (1 - x)^2 = b) :
  a * (1 - x)^2 = b := 
by
  sorry

end homework_duration_decrease_l354_35422


namespace jana_walk_distance_l354_35434

theorem jana_walk_distance :
  (1 / 20 * 15 : ℝ) = 0.8 :=
by sorry

end jana_walk_distance_l354_35434


namespace union_of_A_and_B_l354_35454

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem union_of_A_and_B :
  (A ∪ B) = {1, 2, 3, 4, 5, 7} := 
by
  sorry

end union_of_A_and_B_l354_35454


namespace ratio_a_to_c_l354_35496

theorem ratio_a_to_c (a b c : ℕ) (h1 : a / b = 5 / 3) (h2 : b / c = 1 / 5) : a / c = 1 / 3 :=
sorry

end ratio_a_to_c_l354_35496


namespace power_of_product_l354_35465

variable (a b : ℝ) (m : ℕ)
theorem power_of_product (h : 0 < m) : (a * b)^m = a^m * b^m :=
sorry

end power_of_product_l354_35465


namespace kirin_calculations_l354_35452

theorem kirin_calculations (calculations_per_second : ℝ) (seconds : ℝ) (h1 : calculations_per_second = 10^10) (h2 : seconds = 2022) : 
    calculations_per_second * seconds = 2.022 * 10^13 := 
by
  sorry

end kirin_calculations_l354_35452


namespace euler_totient_bound_l354_35413

theorem euler_totient_bound (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : (Nat.totient^[k]) n = 1) :
  n ≤ 3^k :=
sorry

end euler_totient_bound_l354_35413


namespace linear_function_no_second_quadrant_l354_35472

theorem linear_function_no_second_quadrant (k : ℝ) :
  (∀ x : ℝ, (y : ℝ) → y = k * x - k + 3 → ¬(x < 0 ∧ y > 0)) ↔ k ≥ 3 :=
sorry

end linear_function_no_second_quadrant_l354_35472


namespace value_20_percent_greater_l354_35476

theorem value_20_percent_greater (x : ℝ) : (x = 88 * 1.20) ↔ (x = 105.6) :=
by
  sorry

end value_20_percent_greater_l354_35476


namespace triangle_angle_equality_l354_35497

theorem triangle_angle_equality
  (α β γ α₁ β₁ γ₁ : ℝ)
  (hABC : α + β + γ = 180)
  (hA₁B₁C₁ : α₁ + β₁ + γ₁ = 180)
  (angle_relation : (α = α₁ ∨ α + α₁ = 180) ∧ (β = β₁ ∨ β + β₁ = 180) ∧ (γ = γ₁ ∨ γ + γ₁ = 180)) :
  α = α₁ ∧ β = β₁ ∧ γ = γ₁ :=
by {
  sorry
}

end triangle_angle_equality_l354_35497


namespace prob1_prob2_prob3_l354_35424

def star (a b : ℤ) : ℤ :=
  if a = 0 then b^2
  else if b = 0 then a^2
  else if a > 0 ∧ b > 0 then a^2 + b^2
  else if a < 0 ∧ b < 0 then a^2 + b^2
  else -(a^2 + b^2)

theorem prob1 :
  star (-1) (-1) = 2 :=
sorry

theorem prob2 :
  star (-1) (star 0 (-2)) = -17 :=
sorry

theorem prob3 (m n : ℤ) :
  star (m-1) (n+2) = -2 → (m - n = 1 ∨ m - n = 5) :=
sorry

end prob1_prob2_prob3_l354_35424


namespace min_value_of_sum_of_squares_l354_35408

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x * y + y * z + x * z = 4) :
  x^2 + y^2 + z^2 ≥ 4 :=
sorry

end min_value_of_sum_of_squares_l354_35408


namespace number_of_girls_l354_35449

-- Define the number of boys and girls as natural numbers
variable (B G : ℕ)

-- First condition: The number of girls is 458 more than the number of boys
axiom h1 : G = B + 458

-- Second condition: The total number of pupils is 926
axiom h2 : G + B = 926

-- The theorem to be proved: The number of girls is 692
theorem number_of_girls : G = 692 := by
  sorry

end number_of_girls_l354_35449


namespace least_n_satisfies_inequality_l354_35426

theorem least_n_satisfies_inequality : ∃ n : ℕ, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15 ∧ ∀ m : ℕ, (1 : ℚ) / m - (1 : ℚ) / (m + 1) < 1 / 15 -> ¬ (m < n) := 
sorry

end least_n_satisfies_inequality_l354_35426


namespace sum_of_digits_is_8_l354_35416

theorem sum_of_digits_is_8 (d : ℤ) (h1 : d ≥ 0)
  (h2 : 8 * d / 5 - 80 = d) : (d / 100) + ((d % 100) / 10) + (d % 10) = 8 :=
by
  sorry

end sum_of_digits_is_8_l354_35416


namespace math_equivalence_l354_35467

theorem math_equivalence (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) (hbc : b + c = 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c := 
by 
  sorry

end math_equivalence_l354_35467


namespace total_hair_cut_l354_35451

-- Definitions from conditions
def first_cut : ℝ := 0.375
def second_cut : ℝ := 0.5

-- The theorem stating the math problem
theorem total_hair_cut : first_cut + second_cut = 0.875 := by
  sorry

end total_hair_cut_l354_35451


namespace find_m_plus_t_l354_35429

-- Define the system of equations represented by the augmented matrix
def equation1 (m t : ℝ) : Prop := 3 * m - t = 22
def equation2 (t : ℝ) : Prop := t = 2

-- State the main theorem with the given conditions and the goal
theorem find_m_plus_t (m t : ℝ) (h1 : equation1 m t) (h2 : equation2 t) : m + t = 10 := 
by
  sorry

end find_m_plus_t_l354_35429


namespace factorize_expression_l354_35470

theorem factorize_expression (a b : ℝ) : 2 * a ^ 2 - 8 * b ^ 2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by
  sorry

end factorize_expression_l354_35470


namespace students_height_order_valid_after_rearrangement_l354_35414
open List

variable {n : ℕ} -- number of students in each row
variable (a b : Fin n → ℝ) -- heights of students in each row

/-- Prove Gábor's observation remains valid after rearrangement: 
    each student in the back row is taller than the student in front of them.
    Given:
    - ∀ i, b i < a i (initial condition)
    - ∀ i < j, a i ≤ a j (rearrangement condition)
    Prove:
    - ∀ i, b i < a i (remains valid after rearrangement)
-/
theorem students_height_order_valid_after_rearrangement
  (h₁ : ∀ i : Fin n, b i < a i)
  (h₂ : ∀ (i j : Fin n), i < j → a i ≤ a j) :
  ∀ i : Fin n, b i < a i :=
by sorry

end students_height_order_valid_after_rearrangement_l354_35414


namespace symmetric_points_origin_l354_35466

theorem symmetric_points_origin (a b : ℝ) (h1 : a = -(-2)) (h2 : 1 = -b) : a + b = 1 :=
by
  sorry

end symmetric_points_origin_l354_35466


namespace anthony_balloon_count_l354_35489

variable (Tom Luke Anthony : ℕ)

theorem anthony_balloon_count
  (h1 : Tom = 3 * Luke)
  (h2 : Luke = Anthony / 4)
  (hTom : Tom = 33) :
  Anthony = 44 := by
    sorry

end anthony_balloon_count_l354_35489
