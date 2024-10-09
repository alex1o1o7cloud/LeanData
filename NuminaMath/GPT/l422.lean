import Mathlib

namespace walking_speed_l422_42203

theorem walking_speed (d : ℝ) (w_speed r_speed : ℝ) (w_time r_time : ℝ)
    (h1 : d = r_speed * r_time)
    (h2 : r_speed = 24)
    (h3 : r_time = 1)
    (h4 : w_time = 3) :
    w_speed = 8 :=
by
  sorry

end walking_speed_l422_42203


namespace probability_Q_within_2_of_origin_eq_pi_div_9_l422_42201

noncomputable def probability_within_circle (π : ℝ) : ℝ :=
  let area_of_square := (2 * 3)^2
  let area_of_circle := π * 2^2
  area_of_circle / area_of_square

theorem probability_Q_within_2_of_origin_eq_pi_div_9 :
  probability_within_circle Real.pi = Real.pi / 9 :=
by
  sorry

end probability_Q_within_2_of_origin_eq_pi_div_9_l422_42201


namespace range_of_m_l422_42226

theorem range_of_m (m : ℤ) (x : ℤ) (h1 : (m + 3) / (x - 1) = 1) (h2 : x > 0) : m > -4 ∧ m ≠ -3 :=
sorry

end range_of_m_l422_42226


namespace sufficient_but_not_necessary_l422_42216

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 < x ∧ x < 2) : x < 2 ∧ ∀ y, (y < 2 → y ≤ 1 ∨ y ≥ 2) :=
by
  sorry

end sufficient_but_not_necessary_l422_42216


namespace jongkook_points_l422_42291

-- Define the conditions in the problem
def num_questions_solved_each : ℕ := 18
def shinhye_points : ℕ := 100
def jongkook_correct_6_points : ℕ := 8
def jongkook_correct_5_points : ℕ := 6
def points_per_question_6 : ℕ := 6
def points_per_question_5 : ℕ := 5
def jongkook_wrong_questions : ℕ := num_questions_solved_each - jongkook_correct_6_points - jongkook_correct_5_points

-- Calculate Jongkook's points from correct answers
def jongkook_points_from_6 : ℕ := jongkook_correct_6_points * points_per_question_6
def jongkook_points_from_5 : ℕ := jongkook_correct_5_points * points_per_question_5

-- Calculate total points
def jongkook_total_points : ℕ := jongkook_points_from_6 + jongkook_points_from_5

-- Prove that Jongkook's total points is 78
theorem jongkook_points : jongkook_total_points = 78 :=
by
  sorry

end jongkook_points_l422_42291


namespace constant_term_of_expansion_l422_42262

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the expansion term
def expansion_term (n k : ℕ) (a b : ℚ) : ℚ :=
  (binom n k) * (a ^ k) * (b ^ (n - k))

-- Define the specific example
def specific_expansion_term : ℚ :=
  expansion_term 8 4 3 (2 : ℚ)

theorem constant_term_of_expansion : specific_expansion_term = 90720 :=
by
  -- The proof is omitted
  sorry

end constant_term_of_expansion_l422_42262


namespace monthly_salary_l422_42292

variable {S : ℝ}

-- Conditions based on the problem description
def spends_on_food (S : ℝ) : ℝ := 0.40 * S
def spends_on_house_rent (S : ℝ) : ℝ := 0.20 * S
def spends_on_entertainment (S : ℝ) : ℝ := 0.10 * S
def spends_on_conveyance (S : ℝ) : ℝ := 0.10 * S
def savings (S : ℝ) : ℝ := 0.20 * S

-- Given savings
def savings_amount : ℝ := 2500

-- The proof statement for the monthly salary
theorem monthly_salary (h : savings S = savings_amount) : S = 12500 := by
  sorry

end monthly_salary_l422_42292


namespace top_card_is_joker_probability_l422_42272

theorem top_card_is_joker_probability :
  let totalCards := 54
  let jokerCards := 2
  let probability := (jokerCards : ℚ) / (totalCards : ℚ)
  probability = 1 / 27 :=
by
  sorry

end top_card_is_joker_probability_l422_42272


namespace number_of_workers_is_25_l422_42221

noncomputable def original_workers (W : ℕ) :=
  W * 35 = (W + 10) * 25

theorem number_of_workers_is_25 : ∃ W, original_workers W ∧ W = 25 :=
by
  use 25
  unfold original_workers
  sorry

end number_of_workers_is_25_l422_42221


namespace probJackAndJillChosen_l422_42246

-- Define the probabilities of each worker being chosen
def probJack : ℝ := 0.20
def probJill : ℝ := 0.15

-- Define the probability that Jack and Jill are both chosen
def probJackAndJill : ℝ := probJack * probJill

-- Theorem stating the probability that Jack and Jill are both chosen
theorem probJackAndJillChosen : probJackAndJill = 0.03 := 
by
  -- Replace this sorry with the complete proof
  sorry

end probJackAndJillChosen_l422_42246


namespace least_number_to_add_l422_42215

theorem least_number_to_add (n : ℕ) (d : ℕ) (r : ℕ) (k : ℕ) (l : ℕ) (h₁ : n = 1077) (h₂ : d = 23) (h₃ : n % d = r) (h₄ : d - r = k) (h₅ : r = 19) (h₆ : k = l) : l = 4 :=
by
  sorry

end least_number_to_add_l422_42215


namespace flour_per_new_base_is_one_fifth_l422_42252

def total_flour : ℚ := 40 * (1 / 8)

def flour_per_new_base (p : ℚ) (total_flour : ℚ) : ℚ := total_flour / p

theorem flour_per_new_base_is_one_fifth :
  flour_per_new_base 25 total_flour = 1 / 5 :=
by
  sorry

end flour_per_new_base_is_one_fifth_l422_42252


namespace problem_statement_l422_42281

open Real

theorem problem_statement :
  (10 * 1.8 - 2 * 1.5) / 0.3 + 3^(2/3) - Real.log 4 = 50.6938 :=
by
  sorry

end problem_statement_l422_42281


namespace solution_l422_42276

theorem solution (x : ℝ) 
  (h1 : 1/x < 3)
  (h2 : 1/x > -4) 
  (h3 : x^2 - 3*x + 2 < 0) : 
  1 < x ∧ x < 2 :=
sorry

end solution_l422_42276


namespace problem_statement_l422_42207

-- Define the arithmetic sequence and the conditions
noncomputable def a : ℕ → ℝ := sorry
axiom a_arith_seq : ∃ d : ℝ, ∀ n m : ℕ, a (n + m) = a n + m • d
axiom condition : a 4 + a 10 + a 16 = 30

-- State the theorem
theorem problem_statement : a 18 - 2 * a 14 = -10 :=
sorry

end problem_statement_l422_42207


namespace max_value_q_l422_42283

namespace proof

theorem max_value_q (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end proof

end max_value_q_l422_42283


namespace parallel_lines_slope_l422_42258

theorem parallel_lines_slope (a : ℝ) :
  (∀ x y : ℝ, 3 * x + 4 * y - 2 = 0 → ax * x - 8 * y - 3 = 0 → a = -6) :=
by
  sorry

end parallel_lines_slope_l422_42258


namespace xiaohong_total_score_l422_42285

theorem xiaohong_total_score :
  ∀ (midterm_score final_score : ℕ) (midterm_weight final_weight : ℝ),
    midterm_score = 80 →
    final_score = 90 →
    midterm_weight = 0.4 →
    final_weight = 0.6 →
    (midterm_score * midterm_weight + final_score * final_weight) = 86 :=
by
  intros midterm_score final_score midterm_weight final_weight
  intros h1 h2 h3 h4
  sorry

end xiaohong_total_score_l422_42285


namespace find_angle_A_l422_42236

theorem find_angle_A (a b : ℝ) (B A : ℝ)
  (h1 : a = 2) 
  (h2 : b = Real.sqrt 3) 
  (h3 : B = Real.pi / 3) : 
  A = Real.pi / 2 := 
sorry

end find_angle_A_l422_42236


namespace has_three_real_zeros_l422_42278

noncomputable def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x + m

theorem has_three_real_zeros (m : ℝ) : 
    (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ m = 0 ∧ f x₂ m = 0 ∧ f x₃ m = 0) ↔ (-4 < m ∧ m < 4) :=
sorry

end has_three_real_zeros_l422_42278


namespace drank_bottles_of_juice_l422_42294

theorem drank_bottles_of_juice
  (bottles_in_refrigerator : ℕ)
  (bottles_in_pantry : ℕ)
  (bottles_bought : ℕ)
  (bottles_left : ℕ)
  (initial_bottles := bottles_in_refrigerator + bottles_in_pantry)
  (total_bottles := initial_bottles + bottles_bought)
  (bottles_drank := total_bottles - bottles_left) :
  bottles_in_refrigerator = 4 ∧
  bottles_in_pantry = 4 ∧
  bottles_bought = 5 ∧
  bottles_left = 10 →
  bottles_drank = 3 :=
by sorry

end drank_bottles_of_juice_l422_42294


namespace sum_of_two_digit_odd_numbers_l422_42254

-- Define the set of all two-digit numbers with both digits odd
def two_digit_odd_numbers : List ℕ := 
  [11, 13, 15, 17, 19, 31, 33, 35, 37, 39,
   51, 53, 55, 57, 59, 71, 73, 75, 77, 79,
   91, 93, 95, 97, 99]

-- Define a function to compute the sum of elements in a list
def list_sum (l : List ℕ) : ℕ := l.foldl (.+.) 0

theorem sum_of_two_digit_odd_numbers :
  list_sum two_digit_odd_numbers = 1375 :=
by
  sorry

end sum_of_two_digit_odd_numbers_l422_42254


namespace problem1_problem2_l422_42284

-- Statement for problem 1
theorem problem1 : 
  (-2020 - 2 / 3) + (2019 + 3 / 4) + (-2018 - 5 / 6) + (2017 + 1 / 2) = -2 - 1 / 4 := 
sorry

-- Statement for problem 2
theorem problem2 : 
  (-1 - 1 / 2) + (-2000 - 5 / 6) + (4000 + 3 / 4) + (-1999 - 2 / 3) = -5 / 4 := 
sorry

end problem1_problem2_l422_42284


namespace disk_diameter_solution_l422_42273

noncomputable def disk_diameter_condition : Prop :=
∃ x : ℝ, 
  (4 * Real.sqrt 3 + 2 * Real.pi) * x^2 - 12 * x + Real.sqrt 3 = 0 ∧
  x < Real.sqrt 3 / 6 ∧ 
  2 * x = 0.36

theorem disk_diameter_solution : exists (x : ℝ), 
  disk_diameter_condition := 
sorry

end disk_diameter_solution_l422_42273


namespace geom_progression_contra_l422_42213

theorem geom_progression_contra (q : ℝ) (p n : ℕ) (hp : p > 0) (hn : n > 0) :
  (11 = 10 * q^p) → (12 = 10 * q^n) → False :=
by
  -- proof steps should follow here
  sorry

end geom_progression_contra_l422_42213


namespace find_prob_real_roots_l422_42264

-- Define the polynomial q(x)
def q (a : ℝ) (x : ℝ) : ℝ := x^4 + 3*a*x^3 + (3*a - 5)*x^2 + (-6*a + 4)*x - 3

-- Define the conditions for a to ensure all roots of the polynomial are real
noncomputable def all_roots_real_condition (a : ℝ) : Prop :=
  a ≤ -1/3 ∨ 1 ≤ a

-- Define the probability that given a in the interval [-12, 32] all q's roots are real
noncomputable def probability_real_roots : ℝ :=
  let total_length := 32 - (-12)
  let excluded_interval_length := 1 - (-1/3)
  let valid_interval_length := total_length - excluded_interval_length
  valid_interval_length / total_length

-- State the theorem
theorem find_prob_real_roots :
  probability_real_roots = 32 / 33 :=
sorry

end find_prob_real_roots_l422_42264


namespace base_seven_to_base_ten_l422_42224

theorem base_seven_to_base_ten (n : ℕ) (h : n = 54231) : 
  (1 * 7^0 + 3 * 7^1 + 2 * 7^2 + 4 * 7^3 + 5 * 7^4) = 13497 :=
by
  sorry

end base_seven_to_base_ten_l422_42224


namespace dot_product_correct_l422_42214

theorem dot_product_correct:
  let a : ℝ × ℝ := (5, -7)
  let b : ℝ × ℝ := (-6, -4)
  (a.1 * b.1) + (a.2 * b.2) = -2 := by
sorry

end dot_product_correct_l422_42214


namespace machineA_finishing_time_l422_42268

theorem machineA_finishing_time
  (A : ℝ)
  (hA : 0 < A)
  (hB : 0 < 12)
  (hC : 0 < 6)
  (h_total_time : 0 < 2)
  (h_work_done_per_hour : (1 / A) + (1 / 12) + (1 / 6) = 1 / 2) :
  A = 4 := sorry

end machineA_finishing_time_l422_42268


namespace solve_for_x_l422_42260

variable (x : ℝ)
axiom h : 3 / 4 + 1 / x = 7 / 8

theorem solve_for_x : x = 8 :=
by
  sorry

end solve_for_x_l422_42260


namespace rectangle_perimeter_l422_42286

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 4 * (a + b)) : 2 * (a + b) = 36 := by
  sorry

end rectangle_perimeter_l422_42286


namespace intersecting_lines_b_plus_m_l422_42210

theorem intersecting_lines_b_plus_m :
  ∃ (m b : ℚ), (∀ x y : ℚ, y = m * x + 5 → y = 4 * x + b → (x, y) = (8, 14)) →
               b + m = -63 / 4 :=
by
  sorry

end intersecting_lines_b_plus_m_l422_42210


namespace laptop_price_difference_l422_42299

theorem laptop_price_difference :
  let list_price := 59.99
  let tech_bargains_discount := 15
  let budget_bytes_discount_percentage := 0.30
  let tech_bargains_price := list_price - tech_bargains_discount
  let budget_bytes_price := list_price * (1 - budget_bytes_discount_percentage)
  let cheaper_price := min tech_bargains_price budget_bytes_price
  let expensive_price := max tech_bargains_price budget_bytes_price
  (expensive_price - cheaper_price) * 100 = 300 :=
by
  sorry

end laptop_price_difference_l422_42299


namespace mohit_discount_l422_42220

variable (SP : ℝ) -- Selling price
variable (CP : ℝ) -- Cost price
variable (discount_percentage : ℝ) -- Discount percentage

-- Conditions
axiom h1 : SP = 21000
axiom h2 : CP = 17500
axiom h3 : discount_percentage = ( (SP - (CP + 0.08 * CP)) / SP) * 100

-- Theorem to prove
theorem mohit_discount : discount_percentage = 10 :=
  sorry

end mohit_discount_l422_42220


namespace insufficient_data_l422_42298

variable (M P O : ℝ)

theorem insufficient_data
  (h1 : M < P)
  (h2 : O > M) :
  ¬(P < O) ∧ ¬(O < P) ∧ ¬(P = O) := 
sorry

end insufficient_data_l422_42298


namespace complete_square_solution_l422_42288

theorem complete_square_solution (x: ℝ) : (x^2 + 8 * x - 3 = 0) -> ((x + 4)^2 = 19) := 
by
  sorry

end complete_square_solution_l422_42288


namespace min_ab_sum_l422_42279

theorem min_ab_sum (a b : ℤ) (h : a * b = 72) : a + b >= -17 :=
by
  sorry

end min_ab_sum_l422_42279


namespace frac_nonneg_iff_pos_l422_42234

theorem frac_nonneg_iff_pos (x : ℝ) : (2 / x ≥ 0) ↔ (x > 0) :=
by sorry

end frac_nonneg_iff_pos_l422_42234


namespace total_seats_l422_42219

theorem total_seats (F : ℕ) 
  (h1 : 305 = 4 * F + 2) 
  (h2 : 310 = 4 * F + 2) : 
  310 + F = 387 :=
by
  sorry

end total_seats_l422_42219


namespace central_angle_of_sector_l422_42250

variable (A : ℝ) (r : ℝ) (α : ℝ)

-- Given conditions: A is the area of the sector, and r is the radius.
def is_sector (A : ℝ) (r : ℝ) (α : ℝ) : Prop :=
  A = (1 / 2) * α * r^2

-- Proof that the central angle α given the conditions is 3π/4.
theorem central_angle_of_sector (h1 : is_sector (3 * Real.pi / 8) 1 α) : 
  α = 3 * Real.pi / 4 := 
  sorry

end central_angle_of_sector_l422_42250


namespace no_integer_solutions_for_eq_l422_42271

theorem no_integer_solutions_for_eq {x y : ℤ} : ¬ (∃ x y : ℤ, (x + 7) * (x + 6) = 8 * y + 3) := by
  sorry

end no_integer_solutions_for_eq_l422_42271


namespace gcd_90_150_l422_42237

theorem gcd_90_150 : Int.gcd 90 150 = 30 := 
by sorry

end gcd_90_150_l422_42237


namespace solutionToSystemOfEquations_solutionToSystemOfInequalities_l422_42243

open Classical

noncomputable def solveSystemOfEquations (x y : ℝ) : Prop :=
  2 * x - y = 3 ∧ 3 * x + 2 * y = 22

theorem solutionToSystemOfEquations : ∃ (x y : ℝ), solveSystemOfEquations x y ∧ x = 4 ∧ y = 5 := by
  sorry

def solveSystemOfInequalities (x : ℝ) : Prop :=
  (x - 2) / 2 + 1 < (x + 1) / 3 ∧ 5 * x + 1 ≥ 2 * (2 + x)

theorem solutionToSystemOfInequalities : ∃ x : ℝ, solveSystemOfInequalities x ∧ 1 ≤ x ∧ x < 2 := by
  sorry

end solutionToSystemOfEquations_solutionToSystemOfInequalities_l422_42243


namespace line_not_in_first_quadrant_l422_42280

theorem line_not_in_first_quadrant (m x : ℝ) (h : mx + 3 = 4) (hx : x = 1) : 
  ∀ x y : ℝ, y = (m - 2) * x - 3 → ¬(0 < x ∧ 0 < y) :=
by
  -- The actual proof would go here
  sorry

end line_not_in_first_quadrant_l422_42280


namespace total_doors_needed_correct_l422_42238

-- Define the conditions
def buildings : ℕ := 2
def floors_per_building : ℕ := 12
def apartments_per_floor : ℕ := 6
def doors_per_apartment : ℕ := 7

-- Define the total number of doors needed
def total_doors_needed : ℕ := buildings * floors_per_building * apartments_per_floor * doors_per_apartment

-- State the theorem to prove the total number of doors needed is 1008
theorem total_doors_needed_correct : total_doors_needed = 1008 := by
  sorry

end total_doors_needed_correct_l422_42238


namespace problem_l422_42231

theorem problem (a : ℝ) (h : a^2 - 2 * a - 2 = 0) :
  (1 - 1 / (a + 1)) / (a^3 / (a^2 + 2 * a + 1)) = 1 / 2 :=
by
  sorry

end problem_l422_42231


namespace correct_investment_allocation_l422_42206

noncomputable def investment_division (x : ℤ) : Prop :=
  let s := 2000
  let w := 500
  let rogers_investment := 2500
  let total_initial_capital := (5 / 2 : ℚ) * x
  let new_total_capital := total_initial_capital + rogers_investment
  let equal_share := new_total_capital / 3
  s + w = rogers_investment ∧ 
  (3 / 2 : ℚ) * x + s = equal_share ∧ 
  x + w = equal_share

theorem correct_investment_allocation (x : ℤ) (hx : 3 * x % 2 = 0) :
  x > 0 ∧ investment_division x :=
by
  sorry

end correct_investment_allocation_l422_42206


namespace rays_form_straight_lines_l422_42274

theorem rays_form_straight_lines
  (α β : ℝ)
  (h1 : 2 * α + 2 * β = 360) :
  α + β = 180 :=
by
  -- proof details are skipped using sorry
  sorry

end rays_form_straight_lines_l422_42274


namespace train_problem_l422_42275

theorem train_problem (Sat M S C : ℕ) 
  (h_boarding_day : true)
  (h_arrival_day : true)
  (h_date_matches_car_on_monday : M = C)
  (h_seat_less_than_car : S < C)
  (h_sat_date_greater_than_car : Sat > C) :
  C = 2 ∧ S = 1 :=
by sorry

end train_problem_l422_42275


namespace time_for_P_to_finish_job_alone_l422_42225

variable (T : ℝ)

theorem time_for_P_to_finish_job_alone (h1 : 0 < T) (h2 : 3 * (1 / T + 1 / 20) + 0.4 * (1 / T) = 1) : T = 4 :=
by
  sorry

end time_for_P_to_finish_job_alone_l422_42225


namespace paintings_in_four_weeks_l422_42248

theorem paintings_in_four_weeks (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) :
  hours_per_week = 30 → hours_per_painting = 3 → num_weeks = 4 → 
  (hours_per_week / hours_per_painting) * num_weeks = 40 :=
by
  -- Sorry is used since we are not providing the proof
  sorry

end paintings_in_four_weeks_l422_42248


namespace circle_condition_l422_42200

theorem circle_condition (m : ℝ): (∃ x y : ℝ, (x^2 + y^2 - 2*x - 4*y + m = 0)) ↔ (m < 5) :=
by
  sorry

end circle_condition_l422_42200


namespace nancy_initial_bottle_caps_l422_42297

theorem nancy_initial_bottle_caps (found additional_bottle_caps: ℕ) (total_bottle_caps: ℕ) (h1: additional_bottle_caps = 88) (h2: total_bottle_caps = 179) : 
  (total_bottle_caps - additional_bottle_caps) = 91 :=
by
  sorry

end nancy_initial_bottle_caps_l422_42297


namespace perfect_cubes_between_100_and_900_l422_42257

theorem perfect_cubes_between_100_and_900:
  ∃ n, n = 5 ∧ (∀ k, (k ≥ 5 ∧ k ≤ 9) → (k^3 ≥ 100 ∧ k^3 ≤ 900)) :=
by
  sorry

end perfect_cubes_between_100_and_900_l422_42257


namespace sum_of_coefficients_correct_l422_42232

-- Define the polynomial
def polynomial (x y : ℤ) : ℤ := (x + 3 * y) ^ 17

-- Define the sum of coefficients by substituting x = 1 and y = 1
def sum_of_coefficients : ℤ := polynomial 1 1

-- Statement of the mathematical proof problem
theorem sum_of_coefficients_correct :
  sum_of_coefficients = 17179869184 :=
by
  -- proof will be provided here
  sorry

end sum_of_coefficients_correct_l422_42232


namespace dog_roaming_area_l422_42245

theorem dog_roaming_area :
  let shed_radius := 20
  let rope_length := 10
  let distance_from_edge := 10
  let radius_from_center := shed_radius - distance_from_edge
  radius_from_center = rope_length →
  (π * rope_length^2 = 100 * π) :=
by
  intros shed_radius rope_length distance_from_edge radius_from_center h
  sorry

end dog_roaming_area_l422_42245


namespace son_present_age_l422_42289

variable (S M : ℕ)

-- Condition 1: M = S + 20
def man_age_relation (S M : ℕ) : Prop := M = S + 20

-- Condition 2: In two years, the man's age will be twice the age of his son
def age_relation_in_two_years (S M : ℕ) : Prop := M + 2 = 2*(S + 2)

theorem son_present_age : 
  ∀ (S M : ℕ), man_age_relation S M → age_relation_in_two_years S M → S = 18 :=
by
  intros S M h1 h2
  sorry

end son_present_age_l422_42289


namespace chord_length_l422_42282

theorem chord_length (a b : ℝ) (M : ℝ) (h : M * M = a * b) : ∃ AB : ℝ, AB = 2 * Real.sqrt (a * b) :=
by
  sorry

end chord_length_l422_42282


namespace inscribed_circle_radius_in_quadrilateral_pyramid_l422_42222

theorem inscribed_circle_radius_in_quadrilateral_pyramid
  (a : ℝ) (α : ℝ)
  (h_pos : 0 < a) (h_α : 0 < α ∧ α < π / 2) :
  ∃ r : ℝ, r = a * Real.sqrt 2 / (1 + 2 * Real.cos α + Real.sqrt (4 * Real.cos α ^ 2 + 1)) :=
by
  sorry

end inscribed_circle_radius_in_quadrilateral_pyramid_l422_42222


namespace infinite_solutions_d_eq_5_l422_42269

theorem infinite_solutions_d_eq_5 :
  ∃ (d : ℝ), d = 5 ∧ ∀ (y : ℝ), 3 * (5 + d * y) = 15 * y + 15 :=
by
  sorry

end infinite_solutions_d_eq_5_l422_42269


namespace hexagon_diagonals_l422_42241

-- Define a hexagon as having 6 vertices
def hexagon_vertices : ℕ := 6

-- From one vertex of a hexagon, there are (6 - 1) vertices it can potentially connect to
def potential_connections (vertices : ℕ) : ℕ := vertices - 1

-- Remove the two adjacent vertices to count diagonals
def diagonals_from_vertex (connections : ℕ) : ℕ := connections - 2

theorem hexagon_diagonals : diagonals_from_vertex (potential_connections hexagon_vertices) = 3 := by
  -- The proof is intentionally left as a sorry placeholder.
  sorry

end hexagon_diagonals_l422_42241


namespace sum_of_powers_l422_42277

theorem sum_of_powers (x : ℝ) (h1 : x^10 - 3*x + 2 = 0) (h2 : x ≠ 1) : 
  x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 :=
by
  sorry

end sum_of_powers_l422_42277


namespace min_distance_squared_l422_42296

theorem min_distance_squared (a b c d : ℝ) (e : ℝ) (h₀ : e = Real.exp 1) 
  (h₁ : (a - 2 * Real.exp a) / b = 1) (h₂ : (2 - c) / d = 1) :
  (a - c)^2 + (b - d)^2 = 8 := by
  sorry

end min_distance_squared_l422_42296


namespace sqrt_meaningful_l422_42230

theorem sqrt_meaningful (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end sqrt_meaningful_l422_42230


namespace find_factor_l422_42202

theorem find_factor {n f : ℝ} (h1 : n = 10) (h2 : f * (2 * n + 8) = 84) : f = 3 :=
by
  sorry

end find_factor_l422_42202


namespace positive_integer_solutions_l422_42259

theorem positive_integer_solutions (n x y z : ℕ) (h1 : n > 1) (h2 : n^z < 2001) (h3 : n^x + n^y = n^z) :
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ x = k ∧ y = k ∧ z = k + 1) :=
sorry

end positive_integer_solutions_l422_42259


namespace calculate_yield_l422_42256

-- Define the conditions
def x := 6
def x_pos := 3
def x_tot := 3 * x
def nuts_x_pos := x + x_pos
def nuts_x := x
def nuts_x_neg := x - x_pos
def yield_x_pos := 60
def yield_x := 120
def avg_yield := 100

-- Calculate yields
def nuts_x_pos_yield : ℕ := nuts_x_pos * yield_x_pos
def nuts_x_yield : ℕ := nuts_x * yield_x
noncomputable def total_yield (yield_x_neg : ℕ) : ℕ :=
  nuts_x_pos_yield + nuts_x_yield + nuts_x_neg * yield_x_neg

-- Equation combining all
lemma yield_per_tree : (total_yield Y) / x_tot = avg_yield := sorry

-- Prove Y = 180
theorem calculate_yield : (x = 6 → ((nuts_x_neg * 180 = 540) ∧ rate = 180)) := sorry

end calculate_yield_l422_42256


namespace shem_earnings_l422_42287

theorem shem_earnings (kem_hourly: ℝ) (ratio: ℝ) (workday_hours: ℝ) (shem_hourly: ℝ) (shem_daily: ℝ) :
  kem_hourly = 4 →
  ratio = 2.5 →
  shem_hourly = kem_hourly * ratio →
  workday_hours = 8 →
  shem_daily = shem_hourly * workday_hours →
  shem_daily = 80 :=
by
  -- Proof omitted
  sorry

end shem_earnings_l422_42287


namespace find_number_l422_42266

theorem find_number (N : ℕ) : 
  (N % 13 = 11) ∧ (N % 17 = 9) ↔ N = 141 :=
by 
  sorry

end find_number_l422_42266


namespace power_of_four_l422_42228

-- Definition of the conditions
def prime_factors (x: ℕ): ℕ := 2 * x + 5 + 2

-- The statement we need to prove given the conditions
theorem power_of_four (x: ℕ) (h: prime_factors x = 33) : x = 13 :=
by
  -- Proof goes here
  sorry

end power_of_four_l422_42228


namespace yellow_jelly_bean_probability_l422_42217

theorem yellow_jelly_bean_probability :
  ∀ (p_red p_orange p_green p_total p_yellow : ℝ),
    p_red = 0.15 →
    p_orange = 0.35 →
    p_green = 0.25 →
    p_total = 1 →
    p_red + p_orange + p_green + p_yellow = p_total →
    p_yellow = 0.25 :=
by
  intros p_red p_orange p_green p_total p_yellow h_red h_orange h_green h_total h_sum
  sorry

end yellow_jelly_bean_probability_l422_42217


namespace rate_of_discount_l422_42240

theorem rate_of_discount (marked_price selling_price : ℝ) (h1 : marked_price = 200) (h2 : selling_price = 120) : 
  ((marked_price - selling_price) / marked_price) * 100 = 40 :=
by
  sorry

end rate_of_discount_l422_42240


namespace polynomial_degree_add_sub_l422_42244

noncomputable def degree (p : Polynomial ℂ) : ℕ := 
p.natDegree

variable (M N : Polynomial ℂ)

def is_fifth_degree (M : Polynomial ℂ) : Prop :=
degree M = 5

def is_third_degree (N : Polynomial ℂ) : Prop :=
degree N = 3

theorem polynomial_degree_add_sub (hM : is_fifth_degree M) (hN : is_third_degree N) :
  degree (M + N) = 5 ∧ degree (M - N) = 5 :=
by sorry

end polynomial_degree_add_sub_l422_42244


namespace father_l422_42227

theorem father's_age (M F : ℕ) 
  (h1 : M = (2 / 5 : ℝ) * F)
  (h2 : M + 14 = (1 / 2 : ℝ) * (F + 14)) : 
  F = 70 := 
  sorry

end father_l422_42227


namespace intersection_of_sets_l422_42229

def setA : Set ℝ := {x | x^2 ≤ 4 * x}
def setB : Set ℝ := {x | x < 1}

theorem intersection_of_sets : setA ∩ setB = {x | x < 1} := by
  sorry

end intersection_of_sets_l422_42229


namespace curve_is_line_l422_42265

theorem curve_is_line (θ : ℝ) (hθ : θ = 5 * Real.pi / 6) : 
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ (r : ℝ), r = 0 ↔
  (∃ p : ℝ × ℝ, p.1 = r * Real.cos θ ∧ p.2 = r * Real.sin θ ∧
                p.1 * a + p.2 * b = 0) :=
sorry

end curve_is_line_l422_42265


namespace find_T_l422_42255

theorem find_T (T : ℝ) : (1 / 2) * (1 / 7) * T = (1 / 3) * (1 / 5) * 90 → T = 84 :=
by sorry

end find_T_l422_42255


namespace smallest_positive_a_integer_root_l422_42204

theorem smallest_positive_a_integer_root :
  ∀ x a : ℚ, (exists x : ℚ, (x > 0) ∧ (a > 0) ∧ 
    (
      ((x - a) / 2 + (x - 2 * a) / 3) / ((x + 4 * a) / 5 - (x + 3 * a) / 4) =
      ((x - 3 * a) / 4 + (x - 4 * a) / 5) / ((x + 2 * a) / 3 - (x + a) / 2)
    )
  ) → a = 419 / 421 :=
by sorry

end smallest_positive_a_integer_root_l422_42204


namespace pedro_squares_correct_l422_42261

def squares_jesus : ℕ := 60
def squares_linden : ℕ := 75
def squares_pedro (s_jesus s_linden : ℕ) : ℕ := (s_jesus + s_linden) + 65

theorem pedro_squares_correct :
  squares_pedro squares_jesus squares_linden = 200 :=
by
  sorry

end pedro_squares_correct_l422_42261


namespace units_digit_of_sum_of_cubes_l422_42211

theorem units_digit_of_sum_of_cubes : 
  (24^3 + 42^3) % 10 = 2 := by
sorry

end units_digit_of_sum_of_cubes_l422_42211


namespace regular_polygon_sides_l422_42218

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end regular_polygon_sides_l422_42218


namespace initial_dozens_of_doughnuts_l422_42270

theorem initial_dozens_of_doughnuts (doughnuts_eaten doughnuts_left : ℕ)
  (h_eaten : doughnuts_eaten = 8)
  (h_left : doughnuts_left = 16) :
  (doughnuts_eaten + doughnuts_left) / 12 = 2 := by
  sorry

end initial_dozens_of_doughnuts_l422_42270


namespace pow_1986_mod_7_l422_42293

theorem pow_1986_mod_7 : (5 ^ 1986) % 7 = 1 := by
  sorry

end pow_1986_mod_7_l422_42293


namespace quadratic_condition_l422_42251

theorem quadratic_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ x1^2 - x2^2 = c^2 / a^2) ↔
  b^4 - c^4 = 4 * a * b^2 * c :=
sorry

end quadratic_condition_l422_42251


namespace buses_required_l422_42242

theorem buses_required (students : ℕ) (bus_capacity : ℕ) (h_students : students = 325) (h_bus_capacity : bus_capacity = 45) : 
∃ n : ℕ, n = 8 ∧ bus_capacity * n ≥ students :=
by
  sorry

end buses_required_l422_42242


namespace square_area_l422_42247

theorem square_area (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  ∃ (s : ℝ), (s * Real.sqrt 2 = d) ∧ (s^2 = 144) := by
  sorry

end square_area_l422_42247


namespace find_c_l422_42209

theorem find_c (x y c : ℝ) (h : x = 5 * y) (h2 : 7 * x + 4 * y = 13 * c) : c = 3 * y :=
by
  sorry

end find_c_l422_42209


namespace intersection_of_multiples_of_2_l422_42235

theorem intersection_of_multiples_of_2 : 
  let M := {1, 2, 4, 8}
  let N := {x : ℤ | ∃ k : ℤ, x = 2 * k}
  M ∩ N = {2, 4, 8} :=
by
  sorry

end intersection_of_multiples_of_2_l422_42235


namespace cans_purchased_l422_42239

theorem cans_purchased (S Q E : ℕ) (hQ : Q ≠ 0) :
  (∃ x : ℕ, x = (5 * S * E) / Q) := by
  sorry

end cans_purchased_l422_42239


namespace polynomial_equality_l422_42263

theorem polynomial_equality (x : ℝ) : 
  x * (x * (x * (3 - x) - 3) + 5) + 1 = -x^4 + 3*x^3 - 3*x^2 + 5*x + 1 :=
by 
  sorry

end polynomial_equality_l422_42263


namespace problem1_problem2_l422_42205

-- Proof problem 1
theorem problem1 (x : ℝ) : (x - 1)^2 + x * (3 - x) = x + 1 := sorry

-- Proof problem 2
theorem problem2 (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -2) : (a - 2) / (a - 1) / (a + 1 - 3 / (a - 1)) = 1 / (a + 2) := sorry

end problem1_problem2_l422_42205


namespace find_triangle_angles_l422_42253

theorem find_triangle_angles 
  (α β γ : ℝ)
  (a b : ℝ)
  (h1 : γ = 2 * α)
  (h2 : b = 2 * a)
  (h3 : α + β + γ = 180) :
  α = 30 ∧ β = 90 ∧ γ = 60 := 
by 
  sorry

end find_triangle_angles_l422_42253


namespace zero_and_one_positions_l422_42212

theorem zero_and_one_positions (a : ℝ) :
    (0 = (a + (-a)) / 2) ∧ (1 = ((a + (-a)) / 2 + 1)) :=
by
  sorry

end zero_and_one_positions_l422_42212


namespace unique_solution_c_value_l422_42223

-- Define the main problem: the parameter c for which a given system of equations has a unique solution.
theorem unique_solution_c_value (c : ℝ) : 
  (∀ x y : ℝ, 2 * abs (x + 7) + abs (y - 4) = c ∧ abs (x + 4) + 2 * abs (y - 7) = c → 
   (x = -7 ∧ y = 7)) ↔ c = 3 :=
by sorry

end unique_solution_c_value_l422_42223


namespace total_distance_traveled_l422_42267

noncomputable def travel_distance (speed : ℝ) (time : ℝ) (headwind : ℝ) : ℝ :=
  (speed - headwind) * time

theorem total_distance_traveled :
  let headwind := 5
  let eagle_speed := 15
  let eagle_time := 2.5
  let eagle_distance := travel_distance eagle_speed eagle_time headwind

  let falcon_speed := 46
  let falcon_time := 2.5
  let falcon_distance := travel_distance falcon_speed falcon_time headwind

  let pelican_speed := 33
  let pelican_time := 2.5
  let pelican_distance := travel_distance pelican_speed pelican_time headwind

  let hummingbird_speed := 30
  let hummingbird_time := 2.5
  let hummingbird_distance := travel_distance hummingbird_speed hummingbird_time headwind

  let hawk_speed := 45
  let hawk_time := 3
  let hawk_distance := travel_distance hawk_speed hawk_time headwind

  let swallow_speed := 25
  let swallow_time := 1.5
  let swallow_distance := travel_distance swallow_speed swallow_time headwind

  eagle_distance + falcon_distance + pelican_distance + hummingbird_distance + hawk_distance + swallow_distance = 410 :=
sorry

end total_distance_traveled_l422_42267


namespace test_score_based_on_preparation_l422_42290

theorem test_score_based_on_preparation :
  (grade_varies_directly_with_effective_hours : Prop) →
  (effective_hour_constant : ℝ) →
  (actual_hours_first_test : ℕ) →
  (actual_hours_second_test : ℕ) →
  (score_first_test : ℕ) →
  effective_hour_constant = 0.8 →
  actual_hours_first_test = 5 →
  score_first_test = 80 →
  actual_hours_second_test = 6 →
  grade_varies_directly_with_effective_hours →
  ∃ score_second_test : ℕ, score_second_test = 96 := by
  sorry

end test_score_based_on_preparation_l422_42290


namespace cricket_throwers_l422_42249

theorem cricket_throwers (T L R : ℕ) 
  (h1 : T + L + R = 55)
  (h2 : T + R = 49) 
  (h3 : L = (1/3) * (L + R))
  (h4 : R = (2/3) * (L + R)) :
  T = 37 :=
by sorry

end cricket_throwers_l422_42249


namespace equation_has_solution_iff_l422_42233

open Real

theorem equation_has_solution_iff (a : ℝ) : 
  (∃ x : ℝ, (1/3)^|x| + a - 1 = 0) ↔ (0 ≤ a ∧ a < 1) :=
by
  sorry

end equation_has_solution_iff_l422_42233


namespace smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6_l422_42295

theorem smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6 : 
  ∃ n : ℕ, (∃ k : ℕ, n = 60 * k + 1) ∧ n % 9 = 0 ∧ ∀ m : ℕ, (∃ k' : ℕ, m = 60 * k' + 1) ∧ m % 9 = 0 → n ≤ m :=
by
  sorry

end smallest_number_divisible_by_9_and_remainder_1_mod_2_thru_6_l422_42295


namespace same_function_absolute_value_l422_42208

theorem same_function_absolute_value :
  (∀ (x : ℝ), |x| = if x > 0 then x else -x) :=
by
  intro x
  split_ifs with h
  · exact abs_of_pos h
  · exact abs_of_nonpos (le_of_not_gt h)

end same_function_absolute_value_l422_42208
