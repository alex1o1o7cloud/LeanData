import Mathlib

namespace right_trapezoid_bases_l3720_372009

/-- 
Given a right trapezoid with lateral sides c and d (c < d), if a line parallel to its bases 
splits it into two smaller trapezoids each with an inscribed circle, then the bases of the 
original trapezoid are (√(d+c) + √(d-c))² / 4 and (√(d+c) - √(d-c))² / 4.
-/
theorem right_trapezoid_bases (c d : ℝ) (h : c < d) : 
  ∃ (x y z : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    (∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧ 
      y^2 = x * z ∧
      c + d = x + 2*y + z) →
    x = ((Real.sqrt (d+c) - Real.sqrt (d-c))^2) / 4 ∧
    z = ((Real.sqrt (d+c) + Real.sqrt (d-c))^2) / 4 := by
  sorry

end right_trapezoid_bases_l3720_372009


namespace functional_equation_solutions_l3720_372057

-- Define the set of positive integers
def PositiveInt := {n : ℤ | n > 0}

-- Define the functional equation
def SatisfiesEquation (f : ℚ → ℤ) : Prop :=
  ∀ (x : ℚ) (a : ℤ) (b : PositiveInt), f ((f x + a) / b) = f ((x + a) / b)

-- Define the possible solution functions
def ConstantFunction (C : ℤ) : ℚ → ℤ := λ _ => C
def FloorFunction : ℚ → ℤ := λ x => ⌊x⌋
def CeilingFunction : ℚ → ℤ := λ x => ⌈x⌉

-- State the theorem
theorem functional_equation_solutions (f : ℚ → ℤ) (h : SatisfiesEquation f) :
  (∃ C : ℤ, f = ConstantFunction C) ∨ f = FloorFunction ∨ f = CeilingFunction :=
sorry

end functional_equation_solutions_l3720_372057


namespace ivar_water_planning_l3720_372012

def water_planning (initial_horses : ℕ) (added_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) (total_water : ℕ) : ℕ :=
  let total_horses := initial_horses + added_horses
  let daily_consumption := total_horses * (drinking_water + bathing_water)
  total_water / daily_consumption

theorem ivar_water_planning :
  water_planning 3 5 5 2 1568 = 28 := by
  sorry

end ivar_water_planning_l3720_372012


namespace completing_square_equivalence_l3720_372043

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x - 22 = 0 ↔ (x - 2)^2 = 26 := by
  sorry

end completing_square_equivalence_l3720_372043


namespace sin_210_degrees_l3720_372077

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l3720_372077


namespace max_x_minus_y_l3720_372046

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l3720_372046


namespace expression_evaluation_l3720_372031

theorem expression_evaluation : 
  (0 : ℝ) - 2 - 2 * Real.sin (45 * π / 180) + (π - 3.14) * 0 + (-1)^3 = -3 - Real.sqrt 2 := by
  sorry

end expression_evaluation_l3720_372031


namespace managers_salary_l3720_372021

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 ∧ 
  avg_salary = 1200 ∧ 
  salary_increase = 100 → 
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - num_employees * avg_salary) = 3300 := by
  sorry

end managers_salary_l3720_372021


namespace least_k_for_inequality_l3720_372029

theorem least_k_for_inequality : ∃ k : ℤ, k = 5 ∧ 
  (∀ n : ℤ, 0.0010101 * (10 : ℝ)^n > 10 → n ≥ k) ∧
  (0.0010101 * (10 : ℝ)^k > 10) := by
  sorry

end least_k_for_inequality_l3720_372029


namespace average_of_numbers_between_40_and_80_divisible_by_3_l3720_372082

def numbers_between_40_and_80_divisible_by_3 : List ℕ :=
  (List.range 41).filter (λ n => 40 < n ∧ n ≤ 80 ∧ n % 3 = 0)

theorem average_of_numbers_between_40_and_80_divisible_by_3 :
  (List.sum numbers_between_40_and_80_divisible_by_3) / 
  (List.length numbers_between_40_and_80_divisible_by_3) = 63 := by
  sorry

end average_of_numbers_between_40_and_80_divisible_by_3_l3720_372082


namespace money_division_l3720_372044

/-- 
Given an amount of money divided between three people in the ratio 3:7:12,
where the difference between the first two shares is 4000,
prove that the difference between the second and third shares is 5000.
-/
theorem money_division (total : ℝ) : 
  let p := (3 / 22) * total
  let q := (7 / 22) * total
  let r := (12 / 22) * total
  q - p = 4000 → r - q = 5000 := by
sorry

end money_division_l3720_372044


namespace sandy_jessica_marble_ratio_l3720_372069

/-- The number of marbles in a dozen -/
def marbles_per_dozen : ℕ := 12

/-- The number of dozens of red marbles Jessica has -/
def jessica_dozens : ℕ := 3

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := 144

/-- The ratio of Sandy's red marbles to Jessica's red marbles -/
def marble_ratio : ℚ := sandy_marbles / (jessica_dozens * marbles_per_dozen)

theorem sandy_jessica_marble_ratio :
  marble_ratio = 4 := by sorry

end sandy_jessica_marble_ratio_l3720_372069


namespace legos_given_to_sister_l3720_372010

theorem legos_given_to_sister (initial : ℕ) (lost : ℕ) (current : ℕ) : 
  initial = 380 → lost = 57 → current = 299 → initial - lost - current = 24 :=
by sorry

end legos_given_to_sister_l3720_372010


namespace movie_shelf_distribution_l3720_372081

/-- The number of shelves in a movie store given the following conditions:
  * There are 9 movies in total
  * The owner wants to distribute the movies evenly among the shelves
  * The owner needs 1 more movie to achieve an even distribution
-/
def numShelves : ℕ := 4

theorem movie_shelf_distribution (total_movies : ℕ) (movies_needed : ℕ) : 
  total_movies = 9 → movies_needed = 1 → numShelves = 4 := by
  sorry

#check movie_shelf_distribution

end movie_shelf_distribution_l3720_372081


namespace correct_average_weight_l3720_372006

/-- Given a class of boys with an initially miscalculated average weight and a single misread weight, 
    calculate the correct average weight. -/
theorem correct_average_weight 
  (n : ℕ) 
  (initial_avg : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) 
  (h1 : n = 20) 
  (h2 : initial_avg = 58.4) 
  (h3 : misread_weight = 56) 
  (h4 : correct_weight = 61) : 
  (n * initial_avg + (correct_weight - misread_weight)) / n = 58.65 := by
  sorry

end correct_average_weight_l3720_372006


namespace dans_music_store_spending_l3720_372050

def clarinet_cost : ℚ := 130.30
def songbook_cost : ℚ := 11.24

theorem dans_music_store_spending :
  clarinet_cost + songbook_cost = 141.54 := by sorry

end dans_music_store_spending_l3720_372050


namespace smallest_x_factorization_l3720_372000

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def is_perfect_fifth_power (n : ℕ) : Prop := ∃ m : ℕ, n = m^5

theorem smallest_x_factorization :
  let x := 2^15 * 3^20 * 5^24
  ∀ y : ℕ, y > 0 →
    (is_perfect_square (2*y) ∧ 
     is_perfect_cube (3*y) ∧ 
     is_perfect_fifth_power (5*y)) →
    y ≥ x :=
by sorry

end smallest_x_factorization_l3720_372000


namespace f_composition_of_two_l3720_372079

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^2 + 2 * x - 1

-- State the theorem
theorem f_composition_of_two : f (f 2) = 1481 := by
  sorry

end f_composition_of_two_l3720_372079


namespace equation_solutions_l3720_372067

-- Define the equation
def equation (x : ℂ) : Prop :=
  (x - 2)^4 + (x - 6)^4 = 32

-- State the theorem
theorem equation_solutions :
  ∀ x : ℂ, equation x ↔ (x = 4 ∨ x = 4 + 2*Complex.I*Real.sqrt 6 ∨ x = 4 - 2*Complex.I*Real.sqrt 6) :=
by sorry

end equation_solutions_l3720_372067


namespace biased_coin_prob_l3720_372095

/-- The probability of getting heads for a biased coin -/
def h : ℚ := 2/5

/-- The number of flips -/
def n : ℕ := 4

/-- The probability of getting exactly k heads in n flips -/
def prob_k_heads (k : ℕ) : ℚ := 
  (n.choose k) * h^k * (1-h)^(n-k)

theorem biased_coin_prob : 
  prob_k_heads 1 = prob_k_heads 2 → 
  prob_k_heads 2 = 216/625 :=
by sorry

end biased_coin_prob_l3720_372095


namespace cube_volume_from_surface_area_l3720_372090

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 600 → s^3 = 1000 :=
by
  sorry

end cube_volume_from_surface_area_l3720_372090


namespace complex_square_root_l3720_372056

theorem complex_square_root (z : ℂ) : z^2 = -4 ∧ z.im > 0 → z = 2*I :=
sorry

end complex_square_root_l3720_372056


namespace class_test_probabilities_l3720_372011

theorem class_test_probabilities (P_A P_B P_neither : ℝ)
  (h_A : P_A = 0.8)
  (h_B : P_B = 0.55)
  (h_neither : P_neither = 0.55) :
  P_A + P_B - (1 - P_neither) = 0.9 :=
by sorry

end class_test_probabilities_l3720_372011


namespace inequality_solution_l3720_372083

theorem inequality_solution : ∀ x : ℕ+, 
  (2 * x.val + 9 ≥ 3 * (x.val + 2)) ↔ (x.val = 1 ∨ x.val = 2 ∨ x.val = 3) := by
  sorry

end inequality_solution_l3720_372083


namespace correct_object_clause_introducer_l3720_372076

-- Define a type for words that can introduce clauses
inductive ClauseIntroducer
  | That
  | What
  | Where
  | Which

-- Define a function to check if a word is the correct introducer for an object clause
def isCorrectObjectClauseIntroducer (word : ClauseIntroducer) : Prop :=
  word = ClauseIntroducer.What

-- Theorem stating that "what" is the correct word to introduce the object clause
theorem correct_object_clause_introducer :
  isCorrectObjectClauseIntroducer ClauseIntroducer.What :=
by sorry

end correct_object_clause_introducer_l3720_372076


namespace intersection_line_equation_l3720_372049

/-- Given two lines l₁ and l₂ that intersect to form a line segment with midpoint P(0, 0),
    prove that the line l passing through their intersection points has equation y = 7/6 * x. -/
theorem intersection_line_equation 
  (l₁ : Set (ℝ × ℝ)) 
  (l₂ : Set (ℝ × ℝ)) 
  (h₁ : l₁ = {(x, y) | 4 * x + y + 6 = 0})
  (h₂ : l₂ = {(x, y) | 3 * x - 5 * y - 6 = 0})
  (h_midpoint : ∃ (a b : ℝ × ℝ), a ∈ l₁ ∧ a ∈ l₂ ∧ b ∈ l₁ ∧ b ∈ l₂ ∧ (0, 0) = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)) :
  ∃ (l : Set (ℝ × ℝ)), l = {(x, y) | y = 7/6 * x} ∧ 
    ∀ (p : ℝ × ℝ), (p ∈ l₁ ∧ p ∈ l₂) → p ∈ l :=
sorry

end intersection_line_equation_l3720_372049


namespace sufficient_condition_implies_m_range_l3720_372035

theorem sufficient_condition_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, (x - 1) / x ≤ 0 → 4^x + 2^x - m ≤ 0) → m ≥ 6 := by
  sorry

end sufficient_condition_implies_m_range_l3720_372035


namespace quadratic_always_negative_l3720_372084

theorem quadratic_always_negative (m : ℝ) :
  (∀ x : ℝ, m * x^2 + (m - 1) * x + (m - 1) < 0) ↔ m < -1/3 := by
  sorry

end quadratic_always_negative_l3720_372084


namespace max_value_of_ab_l3720_372058

theorem max_value_of_ab (a b : ℝ) : 
  (Real.sqrt 3 = Real.sqrt (3^a * 3^b)) → (∀ x y : ℝ, (Real.sqrt 3 = Real.sqrt (3^x * 3^y)) → a * b ≥ x * y) → 
  a * b = 1/4 := by
  sorry

end max_value_of_ab_l3720_372058


namespace union_M_N_l3720_372092

def M : Set ℤ := {x | |x| < 2}
def N : Set ℤ := {-2, -1, 0}

theorem union_M_N : M ∪ N = {-2, -1, 0, 1} := by sorry

end union_M_N_l3720_372092


namespace min_vertical_distance_l3720_372045

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2 + 2*x - 1

-- Define the vertical distance between the two functions
def vertical_distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x₀ : ℝ), vertical_distance x₀ = 3/4 ∧
  ∀ (x : ℝ), vertical_distance x ≥ 3/4 := by
sorry

end min_vertical_distance_l3720_372045


namespace equation_solution_l3720_372042

theorem equation_solution :
  let x : ℚ := 1/2
  2 * x - 1 = 0 :=
by sorry

end equation_solution_l3720_372042


namespace proposition_q_false_iff_a_lt_2_l3720_372051

theorem proposition_q_false_iff_a_lt_2 (a : ℝ) :
  (¬∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) ↔ a < 2 :=
by sorry

end proposition_q_false_iff_a_lt_2_l3720_372051


namespace first_piece_cost_l3720_372038

/-- Given the total spent on clothing, the number of pieces, and the prices of some pieces,
    prove the cost of the first piece. -/
theorem first_piece_cost (total : ℕ) (num_pieces : ℕ) (price_one : ℕ) (price_others : ℕ) :
  total = 610 →
  num_pieces = 7 →
  price_one = 81 →
  price_others = 96 →
  ∃ (first_piece : ℕ), first_piece + price_one + (num_pieces - 2) * price_others = total ∧ first_piece = 49 := by
  sorry

end first_piece_cost_l3720_372038


namespace wendy_run_distance_l3720_372065

/-- The distance Wendy walked in miles -/
def walked_distance : ℝ := 9.166666666666666

/-- The additional distance Wendy ran compared to what she walked in miles -/
def additional_run_distance : ℝ := 10.666666666666666

/-- The total distance Wendy ran in miles -/
def total_run_distance : ℝ := walked_distance + additional_run_distance

theorem wendy_run_distance : total_run_distance = 19.833333333333332 := by
  sorry

end wendy_run_distance_l3720_372065


namespace fraction_to_decimal_l3720_372048

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^7) = 0.0000752 := by sorry

end fraction_to_decimal_l3720_372048


namespace limit_x_minus_sin_x_ln_x_at_zero_l3720_372066

theorem limit_x_minus_sin_x_ln_x_at_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |(x - Real.sin x) * Real.log x| < ε :=
sorry

end limit_x_minus_sin_x_ln_x_at_zero_l3720_372066


namespace abs_neg_a_eq_five_implies_a_eq_plus_minus_five_l3720_372080

theorem abs_neg_a_eq_five_implies_a_eq_plus_minus_five (a : ℝ) :
  |(-a)| = 5 → (a = 5 ∨ a = -5) := by
  sorry

end abs_neg_a_eq_five_implies_a_eq_plus_minus_five_l3720_372080


namespace problem_solution_l3720_372088

theorem problem_solution (x y z : ℝ) (hx : x = 550) (hy : y = 104) (hz : z = Real.sqrt 20.8) :
  x - (y / z^2)^3 = 425 := by
  sorry

end problem_solution_l3720_372088


namespace age_fraction_proof_l3720_372015

theorem age_fraction_proof (age : ℕ) (h : age = 64) :
  (8 * (age + 8) - 8 * (age - 8)) / age = 2 := by
  sorry

end age_fraction_proof_l3720_372015


namespace condition_relationship_l3720_372019

theorem condition_relationship (a : ℝ) : 
  (a = 1 → a^2 - 3*a + 2 = 0) ∧ 
  (∃ b : ℝ, b ≠ 1 ∧ b^2 - 3*b + 2 = 0) :=
by sorry

end condition_relationship_l3720_372019


namespace inequality_proof_l3720_372093

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) ≤ 1 / 2 ∧
  (a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) = 1 / 2 ↔ 
   a = 1 / 2 ∧ b = 1 / 3 ∧ c = 1 / 6) :=
by sorry

end inequality_proof_l3720_372093


namespace staff_distribution_theorem_l3720_372030

def distribute_staff (n : ℕ) (k : ℕ) : ℕ :=
  let arrangements := (n.choose 1 * (n-1).choose 1) / 2 +
                      (n.choose 2 * (n-2).choose 2) / 2 +
                      (n.choose 3 * (n-3).choose 3) / 2
  arrangements * (k.factorial)

theorem staff_distribution_theorem :
  distribute_staff 7 3 = 1176 := by
  sorry

end staff_distribution_theorem_l3720_372030


namespace sqrt_2023_divided_by_sum_of_digits_l3720_372089

theorem sqrt_2023_divided_by_sum_of_digits : Real.sqrt (2023 / (2 + 0 + 2 + 3)) = 17 := by
  sorry

end sqrt_2023_divided_by_sum_of_digits_l3720_372089


namespace line_is_integral_curve_no_inflection_points_l3720_372002

/-- Represents a function y(x) that satisfies the differential equation y' = 2x - y -/
def IntegralCurve (y : ℝ → ℝ) : Prop :=
  ∀ x, (deriv y) x = 2 * x - y x

/-- The line y = 2x - 2 is an integral curve of the differential equation y' = 2x - y -/
theorem line_is_integral_curve :
  IntegralCurve (λ x ↦ 2 * x - 2) := by sorry

/-- For any integral curve of y' = 2x - y, its second derivative is never zero -/
theorem no_inflection_points (y : ℝ → ℝ) (h : IntegralCurve y) :
  ∀ x, (deriv (deriv y)) x ≠ 0 := by sorry

end line_is_integral_curve_no_inflection_points_l3720_372002


namespace divisibility_property_l3720_372052

theorem divisibility_property (a b c d : ℤ) 
  (h : (a - c) ∣ (a * b + c * d)) : 
  (a - c) ∣ (a * d + b * c) := by
  sorry

end divisibility_property_l3720_372052


namespace vote_ways_l3720_372062

/-- The number of ways an open vote can occur in a society of n members -/
def openVoteWays (n : ℕ) : ℕ := n^n

/-- The number of ways a secret vote can occur in a society of n members -/
def secretVoteWays (n : ℕ) : ℕ := Nat.choose (2*n - 1) (n - 1)

/-- Theorem stating the number of ways for open and secret votes in a society of n members -/
theorem vote_ways (n : ℕ) :
  (openVoteWays n = n^n) ∧ (secretVoteWays n = Nat.choose (2*n - 1) (n - 1)) := by
  sorry

end vote_ways_l3720_372062


namespace used_car_seller_problem_l3720_372028

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : num_clients = 15)
  (h2 : cars_per_client = 2)
  (h3 : selections_per_car = 3) :
  (num_clients * cars_per_client) / selections_per_car = 10 := by
  sorry

#check used_car_seller_problem

end used_car_seller_problem_l3720_372028


namespace baseball_cards_per_pack_l3720_372025

theorem baseball_cards_per_pack : 
  ∀ (total_cards : ℕ) (num_people : ℕ) (cards_per_person : ℕ) (total_packs : ℕ),
    num_people = 4 →
    cards_per_person = 540 →
    total_packs = 108 →
    total_cards = num_people * cards_per_person →
    total_cards / total_packs = 20 :=
by
  sorry

end baseball_cards_per_pack_l3720_372025


namespace remaining_sales_l3720_372086

-- Define the weekly goal
def weekly_goal : ℕ := 90

-- Define Monday's sales
def monday_sales : ℕ := 45

-- Define Tuesday's sales
def tuesday_sales : ℕ := monday_sales - 16

-- Define the total sales so far
def total_sales : ℕ := monday_sales + tuesday_sales

-- Theorem to prove
theorem remaining_sales : weekly_goal - total_sales = 16 := by
  sorry

end remaining_sales_l3720_372086


namespace prime_relative_frequency_l3720_372059

/-- The number of natural numbers considered -/
def total_numbers : ℕ := 4000

/-- The number of prime numbers among the first 4000 natural numbers -/
def prime_count : ℕ := 551

/-- The relative frequency of prime numbers among the first 4000 natural numbers -/
def relative_frequency : ℚ := prime_count / total_numbers

theorem prime_relative_frequency :
  relative_frequency = 551 / 4000 :=
by sorry

end prime_relative_frequency_l3720_372059


namespace mod_equivalence_problem_l3720_372047

theorem mod_equivalence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 21 ∧ 47635 % 21 = n ∧ n = 19 := by
  sorry

end mod_equivalence_problem_l3720_372047


namespace jogger_speed_l3720_372091

/-- Jogger's speed calculation -/
theorem jogger_speed (train_length : ℝ) (initial_distance : ℝ) (train_speed : ℝ) (passing_time : ℝ) :
  let relative_speed : ℝ := (train_length + initial_distance) / passing_time
  let train_speed_mps : ℝ := train_speed * (5/18)
  let jogger_speed_mps : ℝ := train_speed_mps - relative_speed
  let jogger_speed_kmh : ℝ := jogger_speed_mps * (18/5)
  train_length = 120 →
  initial_distance = 280 →
  train_speed = 45 →
  passing_time = 40 →
  jogger_speed_kmh = 9 := by
  sorry

end jogger_speed_l3720_372091


namespace min_operations_to_measure_88_l3720_372061

/-- Represents the state of the puzzle -/
structure PuzzleState where
  barrel : ℕ
  vessel7 : ℕ
  vessel5 : ℕ

/-- Represents a pouring operation -/
inductive PourOperation
  | FillFrom7 : PourOperation
  | FillFrom5 : PourOperation
  | EmptyTo7 : PourOperation
  | EmptyTo5 : PourOperation
  | Pour7To5 : PourOperation
  | Pour5To7 : PourOperation

/-- Applies a single pouring operation to a puzzle state -/
def applyOperation (state : PuzzleState) (op : PourOperation) : PuzzleState :=
  sorry

/-- Checks if a sequence of operations is valid and results in the target state -/
def isValidSequence (initialState : PuzzleState) (targetBarrel : ℕ) (ops : List PourOperation) : Bool :=
  sorry

/-- Theorem: The minimum number of operations to measure 88 quarts is 17 -/
theorem min_operations_to_measure_88 :
  ∃ (ops : List PourOperation),
    ops.length = 17 ∧
    isValidSequence (PuzzleState.mk 108 0 0) 88 ops ∧
    ∀ (other_ops : List PourOperation),
      isValidSequence (PuzzleState.mk 108 0 0) 88 other_ops →
      other_ops.length ≥ 17 :=
  sorry

end min_operations_to_measure_88_l3720_372061


namespace x_range_l3720_372003

theorem x_range (x : ℝ) (h1 : (1 : ℝ) / x < 3) (h2 : (1 : ℝ) / x > -4) : 
  x > 1/3 ∨ x < -1/4 := by
  sorry

end x_range_l3720_372003


namespace no_lions_present_l3720_372068

theorem no_lions_present (total : ℕ) (tigers monkeys : ℕ) : 
  tigers = 7 * (total - tigers) →
  monkeys = (total - monkeys) / 7 →
  tigers + monkeys = total →
  ∀ other : ℕ, other ≤ total - (tigers + monkeys) → other = 0 :=
by sorry

end no_lions_present_l3720_372068


namespace sequence_inequality_l3720_372018

theorem sequence_inequality (n : ℕ) (a : ℝ) (seq : ℕ → ℝ) 
  (h1 : seq 1 = a)
  (h2 : seq n = a)
  (h3 : ∀ k ∈ Finset.range (n - 2), seq (k + 2) ≤ (seq (k + 1) + seq (k + 3)) / 2) :
  ∀ k ∈ Finset.range n, seq (k + 1) ≤ a := by
  sorry

end sequence_inequality_l3720_372018


namespace no_common_root_for_specific_quadratics_l3720_372075

theorem no_common_root_for_specific_quadratics
  (a b c d : ℝ)
  (h_order : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ (x : ℝ), (x^2 + b*x + c = 0 ∧ x^2 + a*x + d = 0) :=
by sorry

end no_common_root_for_specific_quadratics_l3720_372075


namespace arithmetic_seq_fifth_term_l3720_372005

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

theorem arithmetic_seq_fifth_term
  (a : ℕ → ℤ)
  (h_arith : arithmetic_seq a)
  (h_eq : 2 * a 9 = a 12 + 6) :
  a 5 = 4 := by
  sorry

end arithmetic_seq_fifth_term_l3720_372005


namespace fifteenth_term_of_sequence_l3720_372034

-- Define an arithmetic sequence
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Theorem statement
theorem fifteenth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 8) (h₃ : a₃ = 13) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 73 := by
  sorry

end fifteenth_term_of_sequence_l3720_372034


namespace tangent_line_equation_l3720_372064

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 1

-- Define the point of tangency
def P : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_equation :
  let m := (deriv f) P.1  -- Slope of the tangent line
  let b := P.2 - m * P.1  -- y-intercept of the tangent line
  (∀ x y, y = m * x + b ↔ 3 * x - y - 3 = 0) ∧ 
  (f P.1 = P.2) ∧  -- The point P lies on the curve
  (∀ x, (deriv f) x = 3 * x^2) -- The derivative of f is correct
  :=
by sorry

end tangent_line_equation_l3720_372064


namespace fahrenheit_to_celsius_l3720_372016

theorem fahrenheit_to_celsius (F C : ℚ) : F = (9 / 5) * C + 32 → F = 10 → C = -110 / 9 := by
  sorry

end fahrenheit_to_celsius_l3720_372016


namespace power_of_negative_product_l3720_372070

theorem power_of_negative_product (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end power_of_negative_product_l3720_372070


namespace smallest_n_for_rain_probability_l3720_372098

def rain_probability (n : ℕ) : ℝ :=
  let rec prob (k : ℕ) (p : ℝ) : ℝ :=
    match k with
    | 0 => p
    | k + 1 => prob k (0.5 * p + 0.25)
  prob n 0

theorem smallest_n_for_rain_probability :
  ∀ k : ℕ, k < 9 → rain_probability k ≤ 0.499 ∧
  rain_probability 9 > 0.499 := by
  sorry

end smallest_n_for_rain_probability_l3720_372098


namespace roots_sum_reciprocal_cubes_l3720_372013

theorem roots_sum_reciprocal_cubes (r s : ℂ) : 
  (3 * r^2 + 4 * r + 2 = 0) →
  (3 * s^2 + 4 * s + 2 = 0) →
  (r ≠ s) →
  (1 / r^3 + 1 / s^3 = 1) := by
sorry

end roots_sum_reciprocal_cubes_l3720_372013


namespace num_ways_to_sum_equals_two_pow_n_minus_one_l3720_372040

/-- The number of ways to express a positive integer as a sum of one or more positive integers. -/
def num_ways_to_sum (n : ℕ+) : ℕ :=
  2^(n.val - 1)

/-- Theorem: For any positive integer n, the number of ways to express n as a sum of one or more
    positive integers is equal to 2^(n-1). -/
theorem num_ways_to_sum_equals_two_pow_n_minus_one (n : ℕ+) :
  (num_ways_to_sum n) = 2^(n.val - 1) := by
  sorry

end num_ways_to_sum_equals_two_pow_n_minus_one_l3720_372040


namespace interpretation_correct_l3720_372054

-- Define propositions
variable (p : Prop)  -- Student A's math score is not less than 100 points
variable (q : Prop)  -- Student B's math score is less than 100 points

-- Define the interpretation of p∨(¬q)
def interpretation : Prop := p ∨ (¬q)

-- Theorem statement
theorem interpretation_correct : 
  interpretation p q ↔ (p ∨ ¬q) :=
sorry

end interpretation_correct_l3720_372054


namespace polynomial_remainder_l3720_372032

theorem polynomial_remainder (x : ℝ) : 
  (x^15 + 1) % (x + 1) = 0 := by
  sorry

end polynomial_remainder_l3720_372032


namespace consecutive_square_roots_l3720_372033

theorem consecutive_square_roots (n : ℕ) (h : Real.sqrt n = 3) :
  Real.sqrt (n + 1) = 3 + Real.sqrt 1 := by
  sorry

end consecutive_square_roots_l3720_372033


namespace range_of_a_l3720_372026

/-- Given propositions p and q, where p: x^2 + 2x - 3 > 0 and q: x > a,
    and a sufficient but not necessary condition for ¬q is ¬p,
    prove that the range of values for a is a ≥ 1 -/
theorem range_of_a (x a : ℝ) : 
  (∀ x, (x^2 + 2*x - 3 > 0 → x > a) ∧ 
       (x ≤ a → x^2 + 2*x - 3 ≤ 0)) → 
  a ≥ 1 := by
  sorry

end range_of_a_l3720_372026


namespace smallest_number_l3720_372071

theorem smallest_number (S : Set ℤ) (h1 : S = {1, 0, -2, -3}) :
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = -3 :=
by sorry

end smallest_number_l3720_372071


namespace floor_x_eq_1994_minus_n_l3720_372074

def x : ℕ → ℚ
  | 0 => 1994
  | n + 1 => (x n)^2 / (x n + 1)

theorem floor_x_eq_1994_minus_n (n : ℕ) (h : n ≤ 998) :
  ⌊x n⌋ = 1994 - n :=
by sorry

end floor_x_eq_1994_minus_n_l3720_372074


namespace bear_discount_calculation_l3720_372036

/-- The discount per bear after the first bear, given the price of the first bear,
    the total number of bears, and the total amount paid. -/
def discount_per_bear (first_bear_price : ℚ) (total_bears : ℕ) (total_paid : ℚ) : ℚ :=
  let full_price := first_bear_price * total_bears
  let discount := full_price - total_paid
  discount / (total_bears - 1)

/-- Theorem stating that under the given conditions, the discount per bear after the first bear is $0.50 -/
theorem bear_discount_calculation :
  let first_bear_price : ℚ := 4
  let total_bears : ℕ := 101
  let total_paid : ℚ := 354
  discount_per_bear first_bear_price total_bears total_paid = 1/2 := by
sorry


end bear_discount_calculation_l3720_372036


namespace russian_doll_price_l3720_372001

theorem russian_doll_price (original_quantity : ℕ) (discounted_quantity : ℕ) (discounted_price : ℚ) :
  original_quantity = 15 →
  discounted_quantity = 20 →
  discounted_price = 3 →
  (discounted_quantity * discounted_price) / original_quantity = 4 := by
  sorry

end russian_doll_price_l3720_372001


namespace tomato_suggestion_count_tomato_suggestion_count_proof_l3720_372008

theorem tomato_suggestion_count : ℕ → ℕ → ℕ → Prop :=
  fun bacon_count difference tomato_count =>
    (bacon_count = tomato_count + difference) →
    (bacon_count = 337 ∧ difference = 314) →
    tomato_count = 23

theorem tomato_suggestion_count_proof :
  ∃ (tomato_count : ℕ), tomato_suggestion_count 337 314 tomato_count :=
sorry

end tomato_suggestion_count_tomato_suggestion_count_proof_l3720_372008


namespace fraction_to_zero_power_l3720_372023

theorem fraction_to_zero_power (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a / b : ℚ) ^ (0 : ℤ) = 1 := by sorry

end fraction_to_zero_power_l3720_372023


namespace total_unique_photos_l3720_372037

/-- Represents the number of photographs taken by Octavia -/
def octavia_photos : ℕ := 36

/-- Represents the number of Octavia's photographs framed by Jack -/
def jack_framed_octavia : ℕ := 24

/-- Represents the number of photographs framed by Jack that were taken by other photographers -/
def jack_framed_others : ℕ := 12

/-- Theorem stating the total number of unique photographs either framed by Jack or taken by Octavia -/
theorem total_unique_photos : 
  (octavia_photos + (jack_framed_octavia + jack_framed_others) - jack_framed_octavia) = 48 := by
  sorry


end total_unique_photos_l3720_372037


namespace product_digit_sum_l3720_372094

/-- The first 101-digit number -/
def number1 : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

/-- The second 101-digit number -/
def number2 : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

/-- Function to get the hundreds digit of a number -/
def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

/-- Function to get the tens digit of a number -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem product_digit_sum :
  hundreds_digit (number1 * number2) + tens_digit (number1 * number2) = 8 := by
  sorry

end product_digit_sum_l3720_372094


namespace solve_equation_l3720_372087

theorem solve_equation (x : ℝ) (h : 5 - 5/x = 4 + 4/x) : x = 9 := by
  sorry

end solve_equation_l3720_372087


namespace tim_has_156_golf_balls_l3720_372053

/-- The number of units in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of golf balls Tim has -/
def tims_dozens : ℕ := 13

/-- The total number of golf balls Tim has -/
def tims_golf_balls : ℕ := tims_dozens * dozen

theorem tim_has_156_golf_balls : tims_golf_balls = 156 := by
  sorry

end tim_has_156_golf_balls_l3720_372053


namespace smallest_b_for_factorization_l3720_372055

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∃ (r s : ℤ), x^2 + b*x + 1512 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬∃ (r s : ℤ), x^2 + b'*x + 1512 = (x + r) * (x + s)) ∧
  b = 78 :=
sorry

end smallest_b_for_factorization_l3720_372055


namespace dave_initial_boxes_l3720_372096

def boxes_given : ℕ := 5
def pieces_per_box : ℕ := 3
def pieces_left : ℕ := 21

theorem dave_initial_boxes : 
  ∃ (initial_boxes : ℕ), 
    initial_boxes * pieces_per_box = 
      boxes_given * pieces_per_box + pieces_left ∧
    initial_boxes = 12 :=
by sorry

end dave_initial_boxes_l3720_372096


namespace prize_distributions_count_l3720_372024

/-- Represents the number of bowlers in the tournament -/
def num_bowlers : ℕ := 7

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := num_bowlers - 1

/-- The number of possible outcomes for each game -/
def outcomes_per_game : ℕ := 2

/-- The total number of possible prize distributions -/
def total_distributions : ℕ := outcomes_per_game ^ num_games

/-- Theorem stating that the number of possible prize distributions is 64 -/
theorem prize_distributions_count :
  total_distributions = 64 := by sorry

end prize_distributions_count_l3720_372024


namespace min_distance_MN_l3720_372017

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define a point on the parabola
def point_on_parabola (x y : ℝ) : Prop := parabola x y

-- Define a line passing through F(0,1)
def line_through_F (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the intersection of a line with the parabola
def intersection_line_parabola (k : ℝ) (x y : ℝ) : Prop :=
  point_on_parabola x y ∧ line_through_F k x y

-- Define the line AO (or BO)
def line_AO (x₁ y₁ : ℝ) (x y : ℝ) : Prop := y = (y₁/x₁) * x

-- Define the intersection of line AO (or BO) with line l
def intersection_AO_l (x₁ y₁ : ℝ) (x y : ℝ) : Prop :=
  line_AO x₁ y₁ x y ∧ line_l x y

-- The main theorem
theorem min_distance_MN :
  ∃ (min_dist : ℝ),
    min_dist = 8 * Real.sqrt 2 / 5 ∧
    ∀ (k : ℝ) (x₁ y₁ x₂ y₂ xM yM xN yN : ℝ),
      intersection_line_parabola k x₁ y₁ →
      intersection_line_parabola k x₂ y₂ →
      intersection_AO_l x₁ y₁ xM yM →
      intersection_AO_l x₂ y₂ xN yN →
      Real.sqrt ((xM - xN)^2 + (yM - yN)^2) ≥ min_dist :=
sorry

end min_distance_MN_l3720_372017


namespace company_employee_count_l3720_372039

theorem company_employee_count (december_count : ℕ) (percent_increase : ℚ) : 
  december_count = 480 → percent_increase = 15 / 100 → 
  ∃ (january_count : ℕ), 
    (↑december_count : ℚ) = (1 + percent_increase) * ↑january_count ∧ 
    january_count = 417 := by
  sorry

end company_employee_count_l3720_372039


namespace great_wall_soldiers_l3720_372022

/-- Calculates the total number of soldiers in beacon towers along a wall -/
def total_soldiers (wall_length : ℕ) (tower_interval : ℕ) (soldiers_per_tower : ℕ) : ℕ :=
  (wall_length / tower_interval) * soldiers_per_tower

/-- Theorem stating that for a wall of 7300 km with towers every 5 km and 2 soldiers per tower, 
    the total number of soldiers is 2920 -/
theorem great_wall_soldiers : 
  total_soldiers 7300 5 2 = 2920 := by
  sorry

end great_wall_soldiers_l3720_372022


namespace larger_interior_angle_measure_l3720_372085

/-- A circular pavilion constructed with congruent isosceles trapezoids -/
structure CircularPavilion where
  /-- The number of trapezoids in the pavilion -/
  num_trapezoids : ℕ
  /-- The measure of the larger interior angle of a typical trapezoid in degrees -/
  larger_interior_angle : ℝ
  /-- Assertion that the bottom sides of the two end trapezoids are horizontal -/
  horizontal_bottom_sides : Prop

/-- Theorem stating the measure of the larger interior angle in a circular pavilion with 12 trapezoids -/
theorem larger_interior_angle_measure (p : CircularPavilion) 
  (h1 : p.num_trapezoids = 12)
  (h2 : p.horizontal_bottom_sides) :
  p.larger_interior_angle = 97.5 := by
  sorry

end larger_interior_angle_measure_l3720_372085


namespace golden_ratio_between_zero_and_one_l3720_372041

theorem golden_ratio_between_zero_and_one :
  let φ := (Real.sqrt 5 - 1) / 2
  0 < φ ∧ φ < 1 := by sorry

end golden_ratio_between_zero_and_one_l3720_372041


namespace factor_expression_l3720_372004

theorem factor_expression (t : ℝ) : 4 * t^2 - 144 + 8 = 4 * (t^2 - 34) := by
  sorry

end factor_expression_l3720_372004


namespace books_per_shelf_l3720_372063

theorem books_per_shelf (total_books : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ) 
  (h1 : total_books = 72) 
  (h2 : mystery_shelves = 5) 
  (h3 : picture_shelves = 4) :
  ∃ (books_per_shelf : ℕ), 
    books_per_shelf * (mystery_shelves + picture_shelves) = total_books ∧ 
    books_per_shelf = 8 := by
  sorry

end books_per_shelf_l3720_372063


namespace correct_time_allocation_l3720_372014

/-- Represents the time allocation for different tasks -/
structure TimeAllocation where
  clientCalls : ℕ
  accounting : ℕ
  reports : ℕ
  meetings : ℕ

/-- Calculates the time allocation based on a given ratio and total time -/
def calculateTimeAllocation (ratio : List ℚ) (totalTime : ℕ) : TimeAllocation :=
  sorry

/-- Checks if the calculated time allocation is correct -/
def isCorrectAllocation (allocation : TimeAllocation) : Prop :=
  allocation.clientCalls = 383 ∧
  allocation.accounting = 575 ∧
  allocation.reports = 767 ∧
  allocation.meetings = 255

/-- Theorem stating that the calculated time allocation for the given ratio and total time is correct -/
theorem correct_time_allocation :
  let ratio := [3, 4.5, 6, 2]
  let totalTime := 1980
  let allocation := calculateTimeAllocation ratio totalTime
  isCorrectAllocation allocation ∧ 
  allocation.clientCalls + allocation.accounting + allocation.reports + allocation.meetings = totalTime :=
by sorry

end correct_time_allocation_l3720_372014


namespace aluminum_ball_radius_l3720_372007

theorem aluminum_ball_radius (small_radius : ℝ) (num_small_balls : ℕ) (large_radius : ℝ) :
  small_radius = 0.5 →
  num_small_balls = 12 →
  (4 / 3) * π * large_radius^3 = num_small_balls * ((4 / 3) * π * small_radius^3) →
  large_radius = (3 / 2)^(1 / 3) :=
by sorry

end aluminum_ball_radius_l3720_372007


namespace locus_of_equal_power_l3720_372072

/-- Given two non-concentric circles in a plane, the locus of points with equal power
    relative to both circles is a straight line. -/
theorem locus_of_equal_power (R₁ R₂ a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℝ, ∀ x y : ℝ, ((x + a)^2 + y^2 - R₁^2 = (x - a)^2 + y^2 - R₂^2) ↔ (x = k) :=
by sorry

end locus_of_equal_power_l3720_372072


namespace intense_goblet_points_difference_l3720_372078

/-- The number of teams in the tournament -/
def num_teams : ℕ := 10

/-- The number of points awarded for a win -/
def win_points : ℕ := 4

/-- The number of points awarded for a tie -/
def tie_points : ℕ := 2

/-- The number of points awarded for a loss -/
def loss_points : ℕ := 1

/-- The total number of games played in the tournament -/
def total_games : ℕ := (num_teams * (num_teams - 1)) / 2

/-- The maximum total points possible in the tournament -/
def max_total_points : ℕ := total_games * win_points

/-- The minimum total points possible in the tournament -/
def min_total_points : ℕ := num_teams * (num_teams - 1) * loss_points

theorem intense_goblet_points_difference :
  max_total_points - min_total_points = 90 := by
  sorry

end intense_goblet_points_difference_l3720_372078


namespace prob_rain_weekend_l3720_372073

-- Define the probabilities
def prob_rain_sat : ℝ := 0.30
def prob_rain_sun : ℝ := 0.60
def prob_rain_sun_given_rain_sat : ℝ := 0.40

-- Define the theorem
theorem prob_rain_weekend : 
  let prob_no_rain_sat := 1 - prob_rain_sat
  let prob_no_rain_sun := 1 - prob_rain_sun
  let prob_no_rain_sun_given_rain_sat := 1 - prob_rain_sun_given_rain_sat
  let prob_no_rain_both := prob_no_rain_sat * prob_no_rain_sun
  let prob_rain_sat_no_rain_sun := prob_rain_sat * prob_no_rain_sun_given_rain_sat
  let prob_no_rain_all_scenarios := prob_no_rain_both + prob_rain_sat_no_rain_sun
  1 - prob_no_rain_all_scenarios = 0.54 :=
by
  sorry

#check prob_rain_weekend

end prob_rain_weekend_l3720_372073


namespace flower_pollination_l3720_372027

/-- Represents the types of flowers -/
inductive FlowerType
| Rose
| Sunflower
| Tulip
| Daisy
| Orchid

/-- Represents a bee -/
structure Bee where
  roses_per_hour : ℕ
  sunflowers_per_hour : ℕ
  tulips_per_hour : ℕ
  daisies_per_hour : ℕ
  orchids_per_hour : ℕ

/-- The problem setup -/
def flower_problem : Prop :=
  let total_flowers : ℕ := 60
  let roses : ℕ := 12
  let sunflowers : ℕ := 15
  let tulips : ℕ := 9
  let daisies : ℕ := 18
  let orchids : ℕ := 6
  let hours : ℕ := 3
  let bee_A : Bee := ⟨2, 3, 1, 0, 0⟩
  let bee_B : Bee := ⟨0, 0, 0, 4, 1⟩
  let bee_C : Bee := ⟨1, 2, 2, 3, 1⟩
  let bees : List Bee := [bee_A, bee_B, bee_C]

  total_flowers = roses + sunflowers + tulips + daisies + orchids ∧
  (bees.map (λ b => b.roses_per_hour + b.sunflowers_per_hour + b.tulips_per_hour + 
                    b.daisies_per_hour + b.orchids_per_hour)).sum * hours = 60 ∧
  ∀ ft : FlowerType, 
    (bees.map (λ b => match ft with
      | FlowerType.Rose => b.roses_per_hour
      | FlowerType.Sunflower => b.sunflowers_per_hour
      | FlowerType.Tulip => b.tulips_per_hour
      | FlowerType.Daisy => b.daisies_per_hour
      | FlowerType.Orchid => b.orchids_per_hour
    )).sum * hours ≤ match ft with
      | FlowerType.Rose => roses
      | FlowerType.Sunflower => sunflowers
      | FlowerType.Tulip => tulips
      | FlowerType.Daisy => daisies
      | FlowerType.Orchid => orchids

theorem flower_pollination : flower_problem := by sorry

end flower_pollination_l3720_372027


namespace magnitude_sum_of_vectors_l3720_372060

/-- Given two plane vectors a and b, prove that |a + b| = √5 under specific conditions -/
theorem magnitude_sum_of_vectors (a b : ℝ × ℝ) : 
  a = (1, 1) → 
  ‖b‖ = 1 → 
  Real.cos (Real.pi / 4) * ‖a‖ * ‖b‖ = a.fst * b.fst + a.snd * b.snd →
  ‖a + b‖ = Real.sqrt 5 := by
  sorry

end magnitude_sum_of_vectors_l3720_372060


namespace five_variable_inequality_l3720_372097

theorem five_variable_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 > 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

end five_variable_inequality_l3720_372097


namespace at_least_one_is_diff_of_squares_l3720_372020

theorem at_least_one_is_diff_of_squares (a b : ℕ) : 
  ∃ (x y z w : ℤ), (a = x^2 - y^2) ∨ (b = z^2 - w^2) ∨ (a + b = x^2 - y^2) := by
  sorry

end at_least_one_is_diff_of_squares_l3720_372020


namespace arithmetic_sequence_product_l3720_372099

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 5 * b 6 = 21 →
  b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
by sorry

end arithmetic_sequence_product_l3720_372099
