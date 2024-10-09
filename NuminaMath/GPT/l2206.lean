import Mathlib

namespace michael_bought_crates_on_thursday_l2206_220639

theorem michael_bought_crates_on_thursday :
  ∀ (eggs_per_crate crates_tuesday crates_given current_eggs bought_on_thursday : ℕ),
    crates_tuesday = 6 →
    crates_given = 2 →
    eggs_per_crate = 30 →
    current_eggs = 270 →
    bought_on_thursday = (current_eggs - (crates_tuesday * eggs_per_crate - crates_given * eggs_per_crate)) / eggs_per_crate →
    bought_on_thursday = 5 :=
by
  intros _ _ _ _ _
  sorry

end michael_bought_crates_on_thursday_l2206_220639


namespace find_coefficients_l2206_220608

theorem find_coefficients (A B : ℝ) (h_roots : (x^2 + A * x + B = 0 ∧ (x = A ∨ x = B))) :
  (A = 0 ∧ B = 0) ∨ (A = 1 ∧ B = -2) :=
by sorry

end find_coefficients_l2206_220608


namespace sophomores_more_than_first_graders_l2206_220647

def total_students : ℕ := 95
def first_graders : ℕ := 32
def second_graders : ℕ := total_students - first_graders

theorem sophomores_more_than_first_graders : second_graders - first_graders = 31 := by
  sorry

end sophomores_more_than_first_graders_l2206_220647


namespace subtract_fractions_correct_l2206_220649

theorem subtract_fractions_correct :
  (3 / 8 + 5 / 12 - 1 / 6) = (5 / 8) := by
sorry

end subtract_fractions_correct_l2206_220649


namespace rate_is_five_l2206_220632

noncomputable def rate_per_sq_meter (total_cost : ℕ) (total_area : ℕ) : ℕ :=
  total_cost / total_area

theorem rate_is_five :
  let length := 80
  let breadth := 60
  let road_width := 10
  let total_cost := 6500
  let area_road1 := road_width * breadth
  let area_road2 := road_width * length
  let area_intersection := road_width * road_width
  let total_area := area_road1 + area_road2 - area_intersection
  rate_per_sq_meter total_cost total_area = 5 :=
by
  sorry

end rate_is_five_l2206_220632


namespace smallest_positive_integer_l2206_220654

theorem smallest_positive_integer 
  (x : ℤ) (h1 : x % 6 = 3) (h2 : x % 8 = 2) : x = 33 :=
sorry

end smallest_positive_integer_l2206_220654


namespace coloring_count_l2206_220623

theorem coloring_count : 
  ∀ (n : ℕ), n = 2021 → 
  ∃ (ways : ℕ), ways = 3 * 2 ^ 2020 :=
by
  intros n hn
  existsi 3 * 2 ^ 2020
  sorry

end coloring_count_l2206_220623


namespace grid_square_division_l2206_220662

theorem grid_square_division (m n k : ℕ) (h : m * m = n * k) : ℕ := sorry

end grid_square_division_l2206_220662


namespace calculation_l2206_220665

theorem calculation : (-6)^6 / 6^4 + 4^3 - 7^2 * 2 = 2 :=
by
  -- We add "sorry" here to indicate where the proof would go.
  sorry

end calculation_l2206_220665


namespace suitable_for_systematic_sampling_l2206_220664

-- Define the given conditions as a structure
structure SamplingProblem where
  option_A : String
  option_B : String
  option_C : String
  option_D : String

-- Define the equivalence theorem to prove Option C is the most suitable
theorem suitable_for_systematic_sampling (p : SamplingProblem) 
(hA: p.option_A = "Randomly selecting 8 students from a class of 48 students to participate in an activity")
(hB: p.option_B = "A city has 210 department stores, including 20 large stores, 40 medium stores, and 150 small stores. To understand the business situation of each store, a sample of 21 stores needs to be drawn")
(hC: p.option_C = "Randomly selecting 100 candidates from 1200 exam participants to analyze the answer situation of the questions")
(hD: p.option_D = "Randomly selecting 10 students from 1200 high school students participating in a mock exam to understand the situation") :
  p.option_C = "Randomly selecting 100 candidates from 1200 exam participants to analyze the answer situation of the questions" := 
sorry

end suitable_for_systematic_sampling_l2206_220664


namespace blue_tissue_length_exists_l2206_220617

theorem blue_tissue_length_exists (B R : ℝ) (h1 : R = B + 12) (h2 : 2 * R = 3 * B) : B = 24 := 
by
  sorry

end blue_tissue_length_exists_l2206_220617


namespace least_number_to_add_to_246835_l2206_220652

-- Define relevant conditions and computations
def lcm_of_169_and_289 : ℕ := Nat.lcm 169 289
def remainder_246835_mod_lcm : ℕ := 246835 % lcm_of_169_and_289
def least_number_to_add : ℕ := lcm_of_169_and_289 - remainder_246835_mod_lcm

-- The theorem statement
theorem least_number_to_add_to_246835 : least_number_to_add = 52 :=
by
  sorry

end least_number_to_add_to_246835_l2206_220652


namespace sum_of_repeating_decimals_l2206_220622

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end sum_of_repeating_decimals_l2206_220622


namespace sequences_cover_naturals_without_repetition_l2206_220644

theorem sequences_cover_naturals_without_repetition
  (x y : Real) 
  (hx : Irrational x) 
  (hy : Irrational y) 
  (hxy : 1/x + 1/y = 1) :
  (∀ n : ℕ, ∃! k : ℕ, (⌊k * x⌋ = n) ∨ (⌊k * y⌋ = n)) :=
sorry

end sequences_cover_naturals_without_repetition_l2206_220644


namespace prime_square_minus_one_divisible_by_24_l2206_220610

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p) : 
  ∃ k : ℤ, p^2 - 1 = 24 * k :=
  sorry

end prime_square_minus_one_divisible_by_24_l2206_220610


namespace yellow_highlighters_count_l2206_220625

theorem yellow_highlighters_count 
  (Y : ℕ) 
  (pink_highlighters : ℕ := Y + 7) 
  (blue_highlighters : ℕ := Y + 12) 
  (total_highlighters : ℕ := Y + pink_highlighters + blue_highlighters) : 
  total_highlighters = 40 → Y = 7 :=
by
  sorry

end yellow_highlighters_count_l2206_220625


namespace algebra_problem_l2206_220666

noncomputable def expression (a b : ℝ) : ℝ :=
  (3 * a + b / 3)⁻¹ * ((3 * a)⁻¹ + (b / 3)⁻¹)

theorem algebra_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  expression a b = (a * b)⁻¹ :=
by
  sorry

end algebra_problem_l2206_220666


namespace least_positive_integer_multiple_of_53_l2206_220679

-- Define the problem in a Lean statement.
theorem least_positive_integer_multiple_of_53 :
  ∃ x : ℕ, (3 * x) ^ 2 + 2 * 58 * 3 * x + 58 ^ 2 % 53 = 0 ∧ x = 16 :=
by
  sorry

end least_positive_integer_multiple_of_53_l2206_220679


namespace base6_problem_l2206_220630

theorem base6_problem
  (x y : ℕ)
  (h1 : 453 = 2 * x * 10 + y) -- Constraint from base-6 to base-10 conversion
  (h2 : 0 ≤ x ∧ x ≤ 9) -- x is a base-10 digit
  (h3 : 0 ≤ y ∧ y ≤ 9) -- y is a base-10 digit
  (h4 : 4 * 6^2 + 5 * 6 + 3 = 177) -- Conversion result for 453_6
  (h5 : 2 * x * 10 + y = 177) -- Conversion from condition
  (hx : x = 7) -- x value from solution
  (hy : y = 7) -- y value from solution
  : (x * y) / 10 = 49 / 10 := 
by 
  sorry

end base6_problem_l2206_220630


namespace transformed_cubic_polynomial_l2206_220638

theorem transformed_cubic_polynomial (x z : ℂ) 
    (h1 : z = x + x⁻¹) (h2 : x^3 - 3 * x^2 + x + 2 = 0) : 
    x^2 * (z^2 - z - 1) + 3 = 0 :=
sorry

end transformed_cubic_polynomial_l2206_220638


namespace product_lcm_gcd_l2206_220609

theorem product_lcm_gcd (a b : ℕ) (h_a : a = 24) (h_b : b = 36):
  Nat.lcm a b * Nat.gcd a b = 864 :=
by
  rw [h_a, h_b]
  sorry

end product_lcm_gcd_l2206_220609


namespace no_common_period_l2206_220604

theorem no_common_period (g h : ℝ → ℝ) 
  (hg : ∀ x, g (x + 2) = g x) 
  (hh : ∀ x, h (x + π/2) = h x) : 
  ¬ (∃ T > 0, ∀ x, g (x + T) + h (x + T) = g x + h x) :=
sorry

end no_common_period_l2206_220604


namespace product_of_good_numbers_is_good_l2206_220691

def is_good (n : ℕ) : Prop :=
  ∃ (a b c x y : ℤ), n = a * x * x + b * x * y + c * y * y ∧ b * b - 4 * a * c = -20

theorem product_of_good_numbers_is_good {n1 n2 : ℕ} (h1 : is_good n1) (h2 : is_good n2) : is_good (n1 * n2) :=
sorry

end product_of_good_numbers_is_good_l2206_220691


namespace star_7_3_l2206_220692

def star (a b : ℤ) : ℤ := 4 * a + 3 * b - a * b

theorem star_7_3 : star 7 3 = 16 := 
by 
  sorry

end star_7_3_l2206_220692


namespace students_per_group_l2206_220616

-- Defining the conditions
def total_students : ℕ := 256
def number_of_teachers : ℕ := 8

-- The statement to prove
theorem students_per_group :
  total_students / number_of_teachers = 32 :=
by
  sorry

end students_per_group_l2206_220616


namespace range_of_a_l2206_220668

theorem range_of_a (a : ℝ) (h : a ≥ 0) :
  ∃ a, (2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 * Real.sqrt 2) ↔
  (∀ x y : ℝ, 
    ((x - a)^2 + y^2 = 1) ∧ (x^2 + (y - 2)^2 = 25)) :=
sorry

end range_of_a_l2206_220668


namespace problem1_problem2_l2206_220651

open Set

noncomputable def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3 * a) < 0}

theorem problem1 (a : ℝ) (h1 : A ⊆ (A ∩ B a)) : (4 / 3 : ℝ) ≤ a ∧ a ≤ 2 :=
sorry

theorem problem2 (a : ℝ) (h2 : A ∩ B a = ∅) : a ≤ (2 / 3 : ℝ) ∨ a ≥ 4 :=
sorry

end problem1_problem2_l2206_220651


namespace number_of_math_books_l2206_220678

theorem number_of_math_books (M H : ℕ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 397) : M = 53 :=
by
  sorry

end number_of_math_books_l2206_220678


namespace additional_water_added_l2206_220607

variable (M W : ℕ)

theorem additional_water_added (M W : ℕ) (initial_mix : ℕ) (initial_ratio : ℕ × ℕ) (new_ratio : ℚ) :
  initial_mix = 45 →
  initial_ratio = (4, 1) →
  new_ratio = 4 / 3 →
  (4 / 5) * initial_mix = M →
  (1 / 5) * initial_mix + W = 3 / 4 * M →
  W = 18 :=
by
  sorry

end additional_water_added_l2206_220607


namespace grocery_packs_l2206_220643

theorem grocery_packs (cookie_packs cake_packs : ℕ)
  (h1 : cookie_packs = 23)
  (h2 : cake_packs = 4) :
  cookie_packs + cake_packs = 27 :=
by
  sorry

end grocery_packs_l2206_220643


namespace time_to_store_vaccine_l2206_220613

def final_temp : ℤ := -24
def current_temp : ℤ := -4
def rate_of_change : ℤ := -5

theorem time_to_store_vaccine : 
  ∃ t : ℤ, current_temp + rate_of_change * t = final_temp ∧ t = 4 :=
by
  use 4
  sorry

end time_to_store_vaccine_l2206_220613


namespace adam_students_in_10_years_l2206_220655

-- Define the conditions
def teaches_per_year : Nat := 50
def first_year_students : Nat := 40
def years_teaching : Nat := 10

-- Define the total number of students Adam will teach in 10 years
def total_students (first_year: Nat) (rest_years: Nat) (students_per_year: Nat) : Nat :=
  first_year + (rest_years * students_per_year)

-- State the theorem
theorem adam_students_in_10_years :
  total_students first_year_students (years_teaching - 1) teaches_per_year = 490 :=
by
  sorry

end adam_students_in_10_years_l2206_220655


namespace red_balls_count_l2206_220631

-- Define the conditions
def white_red_ratio : ℕ × ℕ := (5, 3)
def num_white_balls : ℕ := 15

-- Define the theorem to prove
theorem red_balls_count (r : ℕ) : r = num_white_balls / (white_red_ratio.1) * (white_red_ratio.2) :=
by sorry

end red_balls_count_l2206_220631


namespace difference_between_numbers_l2206_220635

theorem difference_between_numbers 
  (L S : ℕ) 
  (hL : L = 1584) 
  (hDiv : L = 6 * S + 15) : 
  L - S = 1323 := 
by
  sorry

end difference_between_numbers_l2206_220635


namespace max_5_cent_coins_l2206_220698

theorem max_5_cent_coins :
  ∃ (x y z : ℕ), 
  x + y + z = 25 ∧ 
  x + 2*y + 5*z = 60 ∧
  (∀ y' z' : ℕ, y' + 4*z' = 35 → z' ≤ 8) ∧
  y + 4*z = 35 ∧ z = 8 := 
sorry

end max_5_cent_coins_l2206_220698


namespace combined_score_is_75_l2206_220601

variable (score1 : ℕ) (total1 : ℕ)
variable (score2 : ℕ) (total2 : ℕ)
variable (score3 : ℕ) (total3 : ℕ)

-- Conditions: Antonette's scores and the number of problems in each test
def Antonette_scores : Prop :=
  score1 = 60 * total1 / 100 ∧ total1 = 15 ∧
  score2 = 85 * total2 / 100 ∧ total2 = 20 ∧
  score3 = 75 * total3 / 100 ∧ total3 = 25

-- Theorem to prove the combined score is 75% (45 out of 60) rounded to the nearest percent
theorem combined_score_is_75
  (h : Antonette_scores score1 total1 score2 total2 score3 total3) :
  100 * (score1 + score2 + score3) / (total1 + total2 + total3) = 75 :=
by sorry

end combined_score_is_75_l2206_220601


namespace intersection_complement_l2206_220677

open Set

def UniversalSet := ℝ
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def CU_M : Set ℝ := compl M

theorem intersection_complement :
  N ∩ CU_M = {x | 1 < x ∧ x ≤ 2} :=
by sorry

end intersection_complement_l2206_220677


namespace bus_seat_problem_l2206_220629

theorem bus_seat_problem 
  (left_seats : ℕ) 
  (right_seats := left_seats - 3) 
  (left_capacity := 3 * left_seats)
  (right_capacity := 3 * right_seats)
  (back_seat_capacity := 12)
  (total_capacity := left_capacity + right_capacity + back_seat_capacity)
  (h1 : total_capacity = 93) 
  : left_seats = 15 := 
by 
  sorry

end bus_seat_problem_l2206_220629


namespace molecular_weight_calculation_l2206_220636

-- Define the condition given in the problem
def molecular_weight_of_4_moles := 488 -- molecular weight of 4 moles in g/mol

-- Define the number of moles
def number_of_moles := 4

-- Define the expected molecular weight of 1 mole
def expected_molecular_weight_of_1_mole := 122 -- molecular weight of 1 mole in g/mol

-- Theorem statement
theorem molecular_weight_calculation : 
  molecular_weight_of_4_moles / number_of_moles = expected_molecular_weight_of_1_mole := 
by
  sorry

end molecular_weight_calculation_l2206_220636


namespace money_made_march_to_august_l2206_220695

section
variable (H : ℕ)

-- Given conditions
def hoursMarchToAugust : ℕ := 23
def hoursSeptToFeb : ℕ := 8
def additionalHours : ℕ := 16
def totalCost : ℕ := 600 + 340
def totalHours : ℕ := hoursMarchToAugust + hoursSeptToFeb + additionalHours

-- Total money equation
def totalMoney : ℕ := totalHours * H

-- Theorem to prove the money made from March to August
theorem money_made_march_to_august : totalMoney = totalCost → hoursMarchToAugust * H = 460 :=
by
  intro h
  have hH : H = 20 := by
    sorry
  rw [hH]
  sorry
end

end money_made_march_to_august_l2206_220695


namespace triangle_inequality_l2206_220693

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a^2 + 2 * b * c) / (b^2 + c^2) + (b^2 + 2 * a * c) / (c^2 + a^2) + (c^2 + 2 * a * b) / (a^2 + b^2) > 3 :=
by
  sorry

end triangle_inequality_l2206_220693


namespace candidate_p_wage_difference_l2206_220634

theorem candidate_p_wage_difference
  (P Q : ℝ)    -- Candidate p's hourly wage is P, Candidate q's hourly wage is Q
  (H : ℝ)      -- Candidate p's working hours
  (total_payment : ℝ)
  (wage_ratio : P = 1.5 * Q)  -- Candidate p is paid 50% more per hour than candidate q
  (hours_diff : Q * (H + 10) = total_payment)  -- Candidate q's total payment equation
  (candidate_q_payment : Q * (H + 10) = 480)   -- total payment for candidate q
  (candidate_p_payment : 1.5 * Q * H = 480)    -- total payment for candidate p
  : P - Q = 8 := sorry

end candidate_p_wage_difference_l2206_220634


namespace total_revenue_l2206_220683

theorem total_revenue (chips_sold : ℕ) (chips_price : ℝ) (hotdogs_sold : ℕ) (hotdogs_price : ℝ)
(drinks_sold : ℕ) (drinks_price : ℝ) (sodas_sold : ℕ) (lemonades_sold : ℕ) (sodas_ratio : ℕ)
(lemonades_ratio : ℕ) (h1 : chips_sold = 27) (h2 : chips_price = 1.50) (h3 : hotdogs_sold = chips_sold - 8)
(h4 : hotdogs_price = 3.00) (h5 : drinks_sold = hotdogs_sold + 12) (h6 : drinks_price = 2.00)
(h7 : sodas_ratio = 2) (h8 : lemonades_ratio = 3) (h9 : sodas_sold = (sodas_ratio * drinks_sold) / (sodas_ratio + lemonades_ratio))
(h10 : lemonades_sold = drinks_sold - sodas_sold) :
chips_sold * chips_price + hotdogs_sold * hotdogs_price + drinks_sold * drinks_price = 159.50 := 
by
  -- Proof is left as an exercise for the reader
  sorry

end total_revenue_l2206_220683


namespace cat_head_start_15_minutes_l2206_220689

theorem cat_head_start_15_minutes :
  ∀ (t : ℕ), (25 : ℝ) = (20 : ℝ) * (1 + (t : ℝ) / 60) → t = 15 := by
  sorry

end cat_head_start_15_minutes_l2206_220689


namespace hallie_hours_worked_on_tuesday_l2206_220648

theorem hallie_hours_worked_on_tuesday
    (hourly_wage : ℝ := 10)
    (hours_monday : ℝ := 7)
    (tips_monday : ℝ := 18)
    (hours_wednesday : ℝ := 7)
    (tips_wednesday : ℝ := 20)
    (tips_tuesday : ℝ := 12)
    (total_earnings : ℝ := 240)
    (tuesday_hours : ℝ) :
    (hourly_wage * hours_monday + tips_monday) +
    (hourly_wage * hours_wednesday + tips_wednesday) +
    (hourly_wage * tuesday_hours + tips_tuesday) = total_earnings →
    tuesday_hours = 5 :=
by
  sorry

end hallie_hours_worked_on_tuesday_l2206_220648


namespace log_evaluation_l2206_220694

theorem log_evaluation
  (x : ℝ)
  (h : x = (Real.log 3 / Real.log 5) ^ (Real.log 5 / Real.log 3)) :
  Real.log x / Real.log 7 = -(Real.log 5 / Real.log 3) * (Real.log (Real.log 5 / Real.log 3) / Real.log 7) :=
by
  sorry

end log_evaluation_l2206_220694


namespace luka_age_difference_l2206_220605

theorem luka_age_difference (a l : ℕ) (h1 : a = 8) (h2 : ∀ m : ℕ, m = 6 → l = m + 4) : l - a = 2 :=
by
  -- Assume Aubrey's age is 8
  have ha : a = 8 := h1
  -- Assume Max's age at Aubrey's 8th birthday is 6
  have hl : l = 10 := h2 6 rfl
  -- Hence, Luka is 2 years older than Aubrey
  sorry

end luka_age_difference_l2206_220605


namespace part1_part2_l2206_220618

variable (a b c : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (h_sum : a + b + c = 1)

theorem part1 : 2 * a * b + b * c + c * a + c ^ 2 / 2 ≤ 1 / 2 := sorry

theorem part2 : (a ^ 2 + c ^ 2) / b + (b ^ 2 + a ^ 2) / c + (c ^ 2 + b ^ 2) / a ≥ 2 := sorry

end part1_part2_l2206_220618


namespace problem_1_solution_problem_2_solution_l2206_220669

variables (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (green_ball : ℕ)

def probability_of_red_or_black_ball (total_balls red_balls black_balls white_balls green_ball : ℕ) : ℚ :=
  (red_balls + black_balls : ℚ) / total_balls

def probability_of_at_least_one_red_ball (total_balls red_balls black_balls white_balls green_ball : ℕ) : ℚ :=
  (((red_balls * (total_balls - red_balls)) + ((red_balls * (red_balls - 1)) / 2)) : ℚ)
  / ((total_balls * (total_balls - 1) / 2) : ℚ)

theorem problem_1_solution :
  probability_of_red_or_black_ball 12 5 4 2 1 = 3 / 4 :=
by
  sorry

theorem problem_2_solution :
  probability_of_at_least_one_red_ball 12 5 4 2 1 = 15 / 22 :=
by
  sorry

end problem_1_solution_problem_2_solution_l2206_220669


namespace calculate_hardcover_volumes_l2206_220653

theorem calculate_hardcover_volumes (h p : ℕ) 
  (h_total_volumes : h + p = 12)
  (h_cost_equation : 27 * h + 16 * p = 284)
  (h_p_relation : p = 12 - h) : h = 8 :=
by
  sorry

end calculate_hardcover_volumes_l2206_220653


namespace complement_intersection_l2206_220671

-- Define the universal set U and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {4, 7, 8}

-- Compute the complements
def complement_U (s : Set ℕ) : Set ℕ := U \ s
def comp_A : Set ℕ := complement_U A
def comp_B : Set ℕ := complement_U B

-- Define the intersection of the complements
def intersection_complements : Set ℕ := comp_A ∩ comp_B

-- The theorem to prove
theorem complement_intersection :
  intersection_complements = {1, 2, 6} :=
by
  sorry

end complement_intersection_l2206_220671


namespace find_a_l2206_220645

noncomputable def A : Set ℝ := {x | x^2 - x - 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}
def is_solution (a : ℝ) : Prop := ∀ b, b ∈ B a → b ∈ A

theorem find_a (a : ℝ) : (B a ⊆ A) → a = 0 ∨ a = -1 ∨ a = 1/2 := by
  intro h
  sorry

end find_a_l2206_220645


namespace symmetric_sum_l2206_220697

theorem symmetric_sum (a b : ℤ) (h1 : a = -4) (h2 : b = -3) : a + b = -7 := by
  sorry

end symmetric_sum_l2206_220697


namespace percentage_runs_by_running_l2206_220659

theorem percentage_runs_by_running
  (total_runs : ℕ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (runs_per_boundary : ℕ)
  (runs_per_six : ℕ)
  (eq_total_runs : total_runs = 120)
  (eq_boundaries : boundaries = 3)
  (eq_sixes : sixes = 8)
  (eq_runs_per_boundary : runs_per_boundary = 4)
  (eq_runs_per_six : runs_per_six = 6) :
  ((total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs * 100) = 50 :=
by
  sorry

end percentage_runs_by_running_l2206_220659


namespace eq_neg_one_fifth_l2206_220670

theorem eq_neg_one_fifth : 
  ((1 : ℝ) / ((-5) ^ 4) ^ 2 * (-5) ^ 7) = -1 / 5 := by
  sorry

end eq_neg_one_fifth_l2206_220670


namespace calculate_seasons_l2206_220627

theorem calculate_seasons :
  ∀ (episodes_per_season : ℕ) (episodes_per_day : ℕ) (days : ℕ),
  episodes_per_season = 20 →
  episodes_per_day = 2 →
  days = 30 →
  (episodes_per_day * days) / episodes_per_season = 3 :=
by
  intros episodes_per_season episodes_per_day days h_eps h_epd h_d
  sorry

end calculate_seasons_l2206_220627


namespace max_ab_l2206_220641

theorem max_ab (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ∃ (M : ℝ), M = 1 / 8 ∧ ∀ (a b : ℝ), (a + 2 * b = 1) → 0 < a → 0 < b → ab ≤ M :=
sorry

end max_ab_l2206_220641


namespace solution_to_equation_l2206_220686

theorem solution_to_equation : 
    (∃ x : ℤ, (x = 2 ∨ x = -2 ∨ x = 1 ∨ x = -1) ∧ (2 * x - 3 = -1)) → x = 1 :=
by
  sorry

end solution_to_equation_l2206_220686


namespace seats_needed_l2206_220621

theorem seats_needed (children seats_per_seat : ℕ) (h1 : children = 58) (h2 : seats_per_seat = 2) : children / seats_per_seat = 29 :=
by sorry

end seats_needed_l2206_220621


namespace distinct_positive_least_sum_seven_integers_prod_2016_l2206_220657

theorem distinct_positive_least_sum_seven_integers_prod_2016 :
  ∃ (n1 n2 n3 n4 n5 n6 n7 : ℕ),
    n1 < n2 ∧ n2 < n3 ∧ n3 < n4 ∧ n4 < n5 ∧ n5 < n6 ∧ n6 < n7 ∧
    (n1 * n2 * n3 * n4 * n5 * n6 * n7) % 2016 = 0 ∧
    n1 + n2 + n3 + n4 + n5 + n6 + n7 = 31 :=
sorry

end distinct_positive_least_sum_seven_integers_prod_2016_l2206_220657


namespace expression_value_l2206_220690

theorem expression_value : (8 * 6) - (4 / 2) = 46 :=
by
  sorry

end expression_value_l2206_220690


namespace smallest_abcd_value_l2206_220612

theorem smallest_abcd_value (A B C D : ℕ) (h1 : A ≠ B) (h2 : 1 ≤ A) (h3 : A ≤ 9) (h4 : 0 ≤ B) 
                            (h5 : B ≤ 9) (h6 : 1 ≤ C) (h7 : C ≤ 9) (h8 : 1 ≤ D) (h9 : D ≤ 9)
                            (h10 : 10 * A * A + A * B = 1000 * A + 100 * B + 10 * C + D)
                            (h11 : A ≠ C) (h12 : A ≠ D) (h13 : B ≠ C) (h14 : B ≠ D) (h15 : C ≠ D) :
  1000 * A + 100 * B + 10 * C + D = 2046 :=
sorry

end smallest_abcd_value_l2206_220612


namespace number_of_trees_l2206_220614

theorem number_of_trees (l d : ℕ) (h_l : l = 441) (h_d : d = 21) : (l / d) + 1 = 22 :=
by
  sorry

end number_of_trees_l2206_220614


namespace fraction_Renz_Miles_l2206_220615

-- Given definitions and conditions
def Mitch_macarons : ℕ := 20
def Joshua_diff : ℕ := 6
def kids : ℕ := 68
def macarons_per_kid : ℕ := 2
def total_macarons_given : ℕ := kids * macarons_per_kid
def Joshua_macarons : ℕ := Mitch_macarons + Joshua_diff
def Miles_macarons : ℕ := 2 * Joshua_macarons
def Mitch_Joshua_Miles_macarons : ℕ := Mitch_macarons + Joshua_macarons + Miles_macarons
def Renz_macarons : ℕ := total_macarons_given - Mitch_Joshua_Miles_macarons

-- The theorem to prove
theorem fraction_Renz_Miles : (Renz_macarons : ℚ) / (Miles_macarons : ℚ) = 19 / 26 :=
by
  sorry

end fraction_Renz_Miles_l2206_220615


namespace range_of_x_l2206_220603

theorem range_of_x (x : ℝ) : x + 2 ≥ 0 ∧ x - 3 ≠ 0 → x ≥ -2 ∧ x ≠ 3 :=
by
  sorry

end range_of_x_l2206_220603


namespace find_larger_number_l2206_220681

theorem find_larger_number (L S : ℕ) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 :=
sorry

end find_larger_number_l2206_220681


namespace loss_per_meter_calculation_l2206_220658

/-- Define the given constants and parameters. --/
def total_meters : ℕ := 600
def selling_price : ℕ := 18000
def cost_price_per_meter : ℕ := 35

/-- Now we define the total cost price, total loss and loss per meter --/
def total_cost_price : ℕ := cost_price_per_meter * total_meters
def total_loss : ℕ := total_cost_price - selling_price
def loss_per_meter : ℕ := total_loss / total_meters

/-- State the theorem we need to prove. --/
theorem loss_per_meter_calculation : loss_per_meter = 5 :=
by
  sorry

end loss_per_meter_calculation_l2206_220658


namespace terms_are_equal_l2206_220600

theorem terms_are_equal (n : ℕ) (a b : ℕ → ℕ)
  (h_n : n ≥ 2018)
  (h_a : ∀ i j : ℕ, i ≠ j → a i ≠ a j)
  (h_b : ∀ i j : ℕ, i ≠ j → b i ≠ b j)
  (h_a_pos : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i > 0)
  (h_b_pos : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → b i > 0)
  (h_a_le : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≤ 5 * n)
  (h_b_le : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → b i ≤ 5 * n)
  (h_arith : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → (a j * b i - a i * b j) * (j - i) = 0):
  ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → a i * b j = a j * b i :=
by
  sorry

end terms_are_equal_l2206_220600


namespace cocktail_cost_l2206_220633

noncomputable def costPerLitreCocktail (cost_mixed_fruit_juice : ℝ) (cost_acai_juice : ℝ) (volume_mixed_fruit : ℝ) (volume_acai : ℝ) : ℝ :=
  let total_cost := cost_mixed_fruit_juice * volume_mixed_fruit + cost_acai_juice * volume_acai
  let total_volume := volume_mixed_fruit + volume_acai
  total_cost / total_volume

theorem cocktail_cost : costPerLitreCocktail 262.85 3104.35 32 21.333333333333332 = 1399.99 :=
  by
    sorry

end cocktail_cost_l2206_220633


namespace gate_perimeter_l2206_220684

theorem gate_perimeter (r : ℝ) (theta : ℝ) (h1 : r = 2) (h2 : theta = π / 2) :
  let arc_length := (3 / 4) * (2 * π * r)
  let radii_length := 2 * r
  arc_length + radii_length = 3 * π + 4 :=
by
  simp [h1, h2]
  sorry

end gate_perimeter_l2206_220684


namespace length_more_than_breadth_by_10_l2206_220661

-- Definitions based on conditions
def length : ℕ := 55
def cost_per_meter : ℚ := 26.5
def total_fencing_cost : ℚ := 5300
def perimeter : ℚ := total_fencing_cost / cost_per_meter

-- Calculate breadth (b) and difference (x)
def breadth := 45 -- This is inferred manually from the solution for completeness
def difference (b : ℚ) := length - b

-- The statement we need to prove
theorem length_more_than_breadth_by_10 :
  difference 45 = 10 :=
by
  sorry

end length_more_than_breadth_by_10_l2206_220661


namespace find_n_l2206_220619

variable (P : ℕ → ℝ) (n : ℕ)

def polynomialDegree (P : ℕ → ℝ) (deg : ℕ) : Prop :=
  ∀ k, k > deg → P k = 0

def zeroValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n + 1)).map (λ k => 2 * k) → P i = 0

def twoValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n)).map (λ k => 2 * k + 1) → P i = 2

def specialValue (P : ℕ → ℝ) (n : ℕ) : Prop :=
  P (2 * n + 1) = -30

theorem find_n :
  (∃ n, polynomialDegree P (2 * n) ∧ zeroValues P n ∧ twoValues P n ∧ specialValue P n) →
  n = 2 :=
by
  sorry

end find_n_l2206_220619


namespace amount_kept_by_Tim_l2206_220699

-- Define the conditions
def totalAmount : ℝ := 100
def percentageGivenAway : ℝ := 0.2

-- Prove the question == answer
theorem amount_kept_by_Tim : totalAmount - totalAmount * percentageGivenAway = 80 :=
by
  -- Here the proof would take place
  sorry

end amount_kept_by_Tim_l2206_220699


namespace yen_exchange_rate_l2206_220646

theorem yen_exchange_rate (yen_per_dollar : ℕ) (dollars : ℕ) (y : ℕ) (h1 : yen_per_dollar = 120) (h2 : dollars = 10) : y = 1200 :=
by
  have h3 : y = yen_per_dollar * dollars := by sorry
  rw [h1, h2] at h3
  exact h3

end yen_exchange_rate_l2206_220646


namespace claire_balloon_count_l2206_220620

variable (start_balloons lost_balloons initial_give_away more_give_away final_balloons grabbed_from_coworker : ℕ)

theorem claire_balloon_count (h1 : start_balloons = 50)
                           (h2 : lost_balloons = 12)
                           (h3 : initial_give_away = 1)
                           (h4 : more_give_away = 9)
                           (h5 : final_balloons = 39)
                           (h6 : start_balloons - initial_give_away - lost_balloons - more_give_away + grabbed_from_coworker = final_balloons) :
                           grabbed_from_coworker = 11 :=
by
  sorry

end claire_balloon_count_l2206_220620


namespace michael_watermelon_weight_l2206_220640

theorem michael_watermelon_weight (m c j : ℝ) (h1 : c = 3 * m) (h2 : j = c / 2) (h3 : j = 12) : m = 8 :=
by
  sorry

end michael_watermelon_weight_l2206_220640


namespace fraction_difference_eq_l2206_220606

theorem fraction_difference_eq (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
sorry

end fraction_difference_eq_l2206_220606


namespace triangle_converse_inverse_false_l2206_220673

variables {T : Type} (p q : T → Prop)

-- Condition: If a triangle is equilateral, then it is isosceles
axiom h : ∀ t, p t → q t

-- Conclusion: Neither the converse nor the inverse is true
theorem triangle_converse_inverse_false : 
  (∃ t, q t ∧ ¬ p t) ∧ (∃ t, ¬ p t ∧ q t) :=
sorry

end triangle_converse_inverse_false_l2206_220673


namespace angle_measure_l2206_220656

theorem angle_measure (A B C : ℝ) (h1 : A = B) (h2 : A + B = 110 ∨ (A = 180 - 110)) :
  A = 70 ∨ A = 55 := by
  sorry

end angle_measure_l2206_220656


namespace number_of_girls_l2206_220682

variable (boys : ℕ) (total_children : ℕ)

theorem number_of_girls (h1 : boys = 40) (h2 : total_children = 117) : total_children - boys = 77 :=
by
  sorry

end number_of_girls_l2206_220682


namespace equilateral_triangle_isosceles_triangle_l2206_220611

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

noncomputable def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem equilateral_triangle (a b c : ℝ) (h : abs (a - b) + abs (b - c) = 0) : is_equilateral a b c :=
  sorry

theorem isosceles_triangle (a b c : ℝ) (h : (a - b) * (b - c) = 0) : is_isosceles a b c :=
  sorry

end equilateral_triangle_isosceles_triangle_l2206_220611


namespace storm_first_thirty_minutes_rain_l2206_220602

theorem storm_first_thirty_minutes_rain 
  (R: ℝ)
  (H1: R + (R / 2) + (1 / 2) = 8)
  : R = 5 :=
by
  sorry

end storm_first_thirty_minutes_rain_l2206_220602


namespace bus_passengers_l2206_220672

def passengers_after_first_stop := 7

def passengers_after_second_stop := passengers_after_first_stop - 3 + 5

def passengers_after_third_stop := passengers_after_second_stop - 2 + 4

theorem bus_passengers (passengers_after_first_stop passengers_after_second_stop passengers_after_third_stop : ℕ) : passengers_after_third_stop = 11 :=
by
  sorry

end bus_passengers_l2206_220672


namespace share_sheets_equally_l2206_220680

theorem share_sheets_equally (sheets friends : ℕ) (h_sheets : sheets = 15) (h_friends : friends = 3) : sheets / friends = 5 := by
  sorry

end share_sheets_equally_l2206_220680


namespace Patricia_money_l2206_220688

theorem Patricia_money 
(P L C : ℝ)
(h1 : L = 5 * P)
(h2 : L = 2 * C)
(h3 : P + L + C = 51) :
P = 6.8 := 
by 
  sorry

end Patricia_money_l2206_220688


namespace option_B_correct_l2206_220642

theorem option_B_correct : 1 ∈ ({0, 1} : Set ℕ) := 
by
  sorry

end option_B_correct_l2206_220642


namespace geometric_sequence_value_sum_l2206_220628

variable {a : ℕ → ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) :=
  ∀ n m, a (n + m) * a 0 = a n * a m

theorem geometric_sequence_value_sum {a : ℕ → ℝ}
  (hpos : ∀ n, a n > 0)
  (geom : is_geometric_sequence a)
  (given : a 0 * a 2 + 2 * a 1 * a 3 + a 2 * a 4 = 16) 
  : a 1 + a 3 = 4 :=
sorry

end geometric_sequence_value_sum_l2206_220628


namespace ratio_elyse_to_rick_l2206_220675

-- Define the conditions
def Elyse_initial_gum : ℕ := 100
def Shane_leftover_gum : ℕ := 14
def Shane_chewed_gum : ℕ := 11

-- Theorem stating the ratio of pieces Elyse gave to Rick to the total number of pieces Elyse had
theorem ratio_elyse_to_rick :
  let total_gum := Elyse_initial_gum
  let Shane_initial_gum := Shane_leftover_gum + Shane_chewed_gum
  let Rick_initial_gum := 2 * Shane_initial_gum
  let Elyse_given_to_Rick := Rick_initial_gum
  (Elyse_given_to_Rick : ℚ) / total_gum = 1 / 2 :=
by
  sorry

end ratio_elyse_to_rick_l2206_220675


namespace right_angled_triangle_exists_l2206_220637

def is_triangle (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_right_angled_triangle (a b c : ℕ) : Prop :=
  (a^2 + b^2 = c^2) ∨ (a^2 + c^2 = b^2) ∨ (b^2 + c^2 = a^2)

theorem right_angled_triangle_exists :
  is_triangle 3 4 5 ∧ is_right_angled_triangle 3 4 5 :=
by
  sorry

end right_angled_triangle_exists_l2206_220637


namespace triangle_inequality_proof_l2206_220626

theorem triangle_inequality_proof (a b c : ℝ) (PA QA PB QB PC QC : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hpa : PA ≥ 0) (hqa : QA ≥ 0) (hpb : PB ≥ 0) (hqb : QB ≥ 0) 
  (hpc : PC ≥ 0) (hqc : QC ≥ 0):
  a * PA * QA + b * PB * QB + c * PC * QC ≥ a * b * c := 
sorry

end triangle_inequality_proof_l2206_220626


namespace sum_of_solutions_of_absolute_value_l2206_220674

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end sum_of_solutions_of_absolute_value_l2206_220674


namespace cos_sub_sin_alpha_l2206_220650

theorem cos_sub_sin_alpha (alpha : ℝ) (h1 : π / 4 < alpha) (h2 : alpha < π / 2)
    (h3 : Real.sin (2 * alpha) = 24 / 25) : Real.cos alpha - Real.sin alpha = -1 / 5 :=
by
  sorry

end cos_sub_sin_alpha_l2206_220650


namespace find_multiple_l2206_220663

theorem find_multiple :
  ∀ (total_questions correct_answers score : ℕ) (m : ℕ),
  total_questions = 100 →
  correct_answers = 90 →
  score = 70 →
  score = correct_answers - m * (total_questions - correct_answers) →
  m = 2 :=
by
  intros total_questions correct_answers score m h1 h2 h3 h4
  sorry

end find_multiple_l2206_220663


namespace scrabble_letter_values_l2206_220696

-- Definitions based on conditions
def middle_letter_value : ℕ := 8
def final_score : ℕ := 30

-- The theorem we need to prove
theorem scrabble_letter_values (F T : ℕ)
  (h1 : 3 * (F + middle_letter_value + T) = final_score) :
  F = 1 ∧ T = 1 :=
sorry

end scrabble_letter_values_l2206_220696


namespace distance_from_apex_to_larger_cross_section_l2206_220660

namespace PyramidProof

variables (As Al : ℝ) (d h : ℝ)

theorem distance_from_apex_to_larger_cross_section 
  (As_eq : As = 256 * Real.sqrt 2) 
  (Al_eq : Al = 576 * Real.sqrt 2) 
  (d_eq : d = 12) :
  h = 36 := 
sorry

end PyramidProof

end distance_from_apex_to_larger_cross_section_l2206_220660


namespace geometric_sequence_sum_l2206_220624

theorem geometric_sequence_sum (a r : ℚ) (h_a : a = 1/3) (h_r : r = 1/2) (S_n : ℚ) (h_S_n : S_n = 80/243) : ∃ n : ℕ, S_n = a * ((1 - r^n) / (1 - r)) ∧ n = 4 := by
  sorry

end geometric_sequence_sum_l2206_220624


namespace ratio_of_second_to_first_show_l2206_220685

-- Definitions based on conditions
def first_show_length : ℕ := 30
def total_show_time : ℕ := 150
def second_show_length := total_show_time - first_show_length

-- Proof problem in Lean 4 statement
theorem ratio_of_second_to_first_show : 
  (second_show_length / first_show_length) = 4 := by
  sorry

end ratio_of_second_to_first_show_l2206_220685


namespace distance_apart_after_3_hours_l2206_220676

-- Definitions derived from conditions
def Ann_speed (hour : ℕ) : ℕ :=
  if hour = 1 then 6 else if hour = 2 then 8 else 4

def Glenda_speed (hour : ℕ) : ℕ :=
  if hour = 1 then 8 else if hour = 2 then 5 else 9

-- The total distance function for a given skater
def total_distance (speed : ℕ → ℕ) : ℕ :=
  speed 1 + speed 2 + speed 3

-- Ann's total distance skated
def Ann_total_distance : ℕ := total_distance Ann_speed

-- Glenda's total distance skated
def Glenda_total_distance : ℕ := total_distance Glenda_speed

-- The total distance between Ann and Glenda after 3 hours
def total_distance_apart : ℕ := Ann_total_distance + Glenda_total_distance

-- Proof statement (without the proof itself; just the goal declaration)
theorem distance_apart_after_3_hours : total_distance_apart = 40 := by
  sorry

end distance_apart_after_3_hours_l2206_220676


namespace grouping_equal_products_l2206_220687

def group1 : List Nat := [12, 42, 95, 143]
def group2 : List Nat := [30, 44, 57, 91]

def product (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem grouping_equal_products :
  product group1 = product group2 := by
  sorry

end grouping_equal_products_l2206_220687


namespace theta_in_fourth_quadrant_l2206_220667

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.tan θ < 0) : 
  (π < θ ∧ θ < 2 * π) :=
by
  sorry

end theta_in_fourth_quadrant_l2206_220667
