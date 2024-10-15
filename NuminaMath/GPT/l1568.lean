import Mathlib

namespace NUMINAMATH_GPT_ceil_sub_self_eq_half_l1568_156848

theorem ceil_sub_self_eq_half (n : ℤ) (x : ℝ) (h : x = n + 1/2) : ⌈x⌉ - x = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_ceil_sub_self_eq_half_l1568_156848


namespace NUMINAMATH_GPT_find_m_range_l1568_156825

def p (m : ℝ) : Prop := (4 - 4 * m) ≤ 0
def q (m : ℝ) : Prop := (5 - 2 * m) > 1

theorem find_m_range (m : ℝ) (hp_false : ¬ p m) (hq_true : q m) : 1 ≤ m ∧ m < 2 :=
by {
 sorry
}

end NUMINAMATH_GPT_find_m_range_l1568_156825


namespace NUMINAMATH_GPT_correct_statements_about_f_l1568_156810

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem correct_statements_about_f : 
  (∀ x, (f x) ≤ (f e)) ∧ (f e = 1 / e) ∧ 
  (∀ x, (f x = 0) → x = 1) ∧ 
  (f 2 < f π ∧ f π < f 3) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_about_f_l1568_156810


namespace NUMINAMATH_GPT_additional_life_vests_needed_l1568_156885

def num_students : ℕ := 40
def num_instructors : ℕ := 10
def life_vests_on_hand : ℕ := 20
def percent_students_with_vests : ℕ := 20

def total_people : ℕ := num_students + num_instructors
def students_with_vests : ℕ := (percent_students_with_vests * num_students) / 100
def total_vests_available : ℕ := life_vests_on_hand + students_with_vests

theorem additional_life_vests_needed : 
  total_people - total_vests_available = 22 :=
by 
  sorry

end NUMINAMATH_GPT_additional_life_vests_needed_l1568_156885


namespace NUMINAMATH_GPT_soda_difference_l1568_156892

-- Define the number of regular soda bottles
def R : ℕ := 79

-- Define the number of diet soda bottles
def D : ℕ := 53

-- The theorem that states the number of regular soda bottles minus the number of diet soda bottles is 26
theorem soda_difference : R - D = 26 := 
by
  sorry

end NUMINAMATH_GPT_soda_difference_l1568_156892


namespace NUMINAMATH_GPT_sum_base6_l1568_156876

theorem sum_base6 : 
  ∀ (a b : ℕ) (h1 : a = 4532) (h2 : b = 3412),
  (a + b = 10414) :=
by
  intros a b h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_sum_base6_l1568_156876


namespace NUMINAMATH_GPT_sum_of_squares_bounds_l1568_156882

theorem sum_of_squares_bounds (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 10) : 
  (x^2 + y^2 ≤ 100) ∧ (x^2 + y^2 ≥ 50) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_squares_bounds_l1568_156882


namespace NUMINAMATH_GPT_probability_of_draw_l1568_156891

-- Define the probabilities as given conditions
def P_A : ℝ := 0.4
def P_A_not_losing : ℝ := 0.9

-- Define the probability of drawing
def P_draw : ℝ :=
  P_A_not_losing - P_A

-- State the theorem to be proved
theorem probability_of_draw : P_draw = 0.5 := by
  sorry

end NUMINAMATH_GPT_probability_of_draw_l1568_156891


namespace NUMINAMATH_GPT_square_side_increase_l1568_156831

variable (s : ℝ)  -- original side length of the square.
variable (p : ℝ)  -- percentage increase of the side length.

theorem square_side_increase (h1 : (s * (1 + p / 100))^2 = 1.21 * s^2) : p = 10 := 
by
  sorry

end NUMINAMATH_GPT_square_side_increase_l1568_156831


namespace NUMINAMATH_GPT_functional_inequality_solution_l1568_156854

theorem functional_inequality_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, 2 + f x * f y ≤ x * y + 2 * f (x + y + 1)) ↔ (∀ x : ℝ, f x = x + 2) :=
by
  sorry

end NUMINAMATH_GPT_functional_inequality_solution_l1568_156854


namespace NUMINAMATH_GPT_calculate_selling_price_l1568_156841

noncomputable def originalPrice : ℝ := 120
noncomputable def firstDiscountRate : ℝ := 0.30
noncomputable def secondDiscountRate : ℝ := 0.15
noncomputable def taxRate : ℝ := 0.08

def discountedPrice1 (originalPrice firstDiscountRate : ℝ) : ℝ :=
  originalPrice * (1 - firstDiscountRate)

def discountedPrice2 (discountedPrice1 secondDiscountRate : ℝ) : ℝ :=
  discountedPrice1 * (1 - secondDiscountRate)

def finalPrice (discountedPrice2 taxRate : ℝ) : ℝ :=
  discountedPrice2 * (1 + taxRate)

theorem calculate_selling_price : 
  finalPrice (discountedPrice2 (discountedPrice1 originalPrice firstDiscountRate) secondDiscountRate) taxRate = 77.112 := 
sorry

end NUMINAMATH_GPT_calculate_selling_price_l1568_156841


namespace NUMINAMATH_GPT_binom_10_3_eq_120_l1568_156870

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_GPT_binom_10_3_eq_120_l1568_156870


namespace NUMINAMATH_GPT_complex_number_z_l1568_156897

theorem complex_number_z (i : ℂ) (z : ℂ) (hi : i * i = -1) (h : 2 * i / z = 1 - i) : z = -1 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_number_z_l1568_156897


namespace NUMINAMATH_GPT_fraction_scaled_l1568_156898

theorem fraction_scaled (x y : ℝ) :
  ∃ (k : ℝ), (k = 3 * y) ∧ ((5 * x + 3 * y) / (x + 3 * y) = 5 * ((x + (3 * y)) / (x + (3 * y)))) := 
  sorry

end NUMINAMATH_GPT_fraction_scaled_l1568_156898


namespace NUMINAMATH_GPT_solution_set_correct_l1568_156821

noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then 2^(-x) - 4 else 2^(x) - 4

theorem solution_set_correct : 
  (∀ x, f x = f |x|) → 
  (∀ x, f x = 2^(-x) - 4 ∨ f x = 2^(x) - 4) → 
  { x | f (x - 2) > 0 } = { x | x < 0 ∨ x > 4 } :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_solution_set_correct_l1568_156821


namespace NUMINAMATH_GPT_solve_for_x_l1568_156866

namespace proof_problem

-- Define the operation a * b = 4 * a * b
def star (a b : ℝ) : ℝ := 4 * a * b

-- Given condition rewritten in terms of the operation star
def equation (x : ℝ) : Prop := star x x + star 2 x - star 2 4 = 0

-- The statement we intend to prove
theorem solve_for_x (x : ℝ) : equation x → (x = 2 ∨ x = -4) :=
by
  -- Proof omitted
  sorry

end proof_problem

end NUMINAMATH_GPT_solve_for_x_l1568_156866


namespace NUMINAMATH_GPT_find_salary_january_l1568_156807

noncomputable section
open Real

def average_salary_jan_to_apr (J F M A : ℝ) : Prop := 
  (J + F + M + A) / 4 = 8000

def average_salary_feb_to_may (F M A May : ℝ) : Prop := 
  (F + M + A + May) / 4 = 9500

def may_salary_value (May : ℝ) : Prop := 
  May = 6500

theorem find_salary_january : 
  ∀ J F M A May, 
    average_salary_jan_to_apr J F M A → 
    average_salary_feb_to_may F M A May → 
    may_salary_value May → 
    J = 500 :=
by
  intros J F M A May h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_salary_january_l1568_156807


namespace NUMINAMATH_GPT_inverse_proportion_l1568_156879

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x^2 * y^4 = k)
  (h2 : 6^2 * 2^4 = k) (hy : y = 4) : x^2 = 2.25 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_l1568_156879


namespace NUMINAMATH_GPT_sara_spent_on_movies_l1568_156822

def cost_of_movie_tickets : ℝ := 2 * 10.62
def cost_of_rented_movie : ℝ := 1.59
def cost_of_purchased_movie : ℝ := 13.95

theorem sara_spent_on_movies :
  cost_of_movie_tickets + cost_of_rented_movie + cost_of_purchased_movie = 36.78 := by
  sorry

end NUMINAMATH_GPT_sara_spent_on_movies_l1568_156822


namespace NUMINAMATH_GPT_one_div_a_plus_one_div_b_l1568_156800

theorem one_div_a_plus_one_div_b (a b : ℝ) (h₀ : a ≠ b) (ha : a^2 - 3 * a + 2 = 0) (hb : b^2 - 3 * b + 2 = 0) :
  1 / a + 1 / b = 3 / 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_one_div_a_plus_one_div_b_l1568_156800


namespace NUMINAMATH_GPT_sampling_interval_divisor_l1568_156887

theorem sampling_interval_divisor (P : ℕ) (hP : P = 524) (k : ℕ) (hk : k ∣ P) : k = 4 :=
by
  sorry

end NUMINAMATH_GPT_sampling_interval_divisor_l1568_156887


namespace NUMINAMATH_GPT_proof_correct_word_choice_l1568_156880

def sentence_completion_correct (word : String) : Prop :=
  "Most of them are kind, but " ++ word ++ " is so good to me as Bruce" = "Most of them are kind, but none is so good to me as Bruce"

theorem proof_correct_word_choice : 
  (sentence_completion_correct "none") → 
  ("none" = "none") := 
by
  sorry

end NUMINAMATH_GPT_proof_correct_word_choice_l1568_156880


namespace NUMINAMATH_GPT_twenty_five_percent_of_five_hundred_is_one_twenty_five_l1568_156873

theorem twenty_five_percent_of_five_hundred_is_one_twenty_five :
  let percent := 0.25
  let amount := 500
  percent * amount = 125 :=
by
  sorry

end NUMINAMATH_GPT_twenty_five_percent_of_five_hundred_is_one_twenty_five_l1568_156873


namespace NUMINAMATH_GPT_flight_duration_l1568_156823

theorem flight_duration :
  ∀ (h m : ℕ),
  3 * 60 + 42 = 15 * 60 + 57 →
  0 < m ∧ m < 60 →
  h + m = 18 :=
by
  intros h m h_def hm_bound
  sorry

end NUMINAMATH_GPT_flight_duration_l1568_156823


namespace NUMINAMATH_GPT_dividend_is_correct_l1568_156801

def divisor : ℕ := 17
def quotient : ℕ := 9
def remainder : ℕ := 6

def calculate_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem dividend_is_correct : calculate_dividend divisor quotient remainder = 159 :=
  by sorry

end NUMINAMATH_GPT_dividend_is_correct_l1568_156801


namespace NUMINAMATH_GPT_union_A_B_l1568_156852

-- Define set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 0}

-- Define set B
def B : Set ℝ := {x | x^2 > 1}

-- Prove the union of A and B is the expected result
theorem union_A_B : A ∪ B = {x | x ≤ 0 ∨ x > 1} :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_l1568_156852


namespace NUMINAMATH_GPT_man_l1568_156816

-- Define the man's rowing speed in still water, the speed of the current, the downstream speed and headwind reduction.
def v : Real := 17.5
def speed_current : Real := 4.5
def speed_downstream : Real := 22
def headwind_reduction : Real := 1.5

-- Define the man's speed against the current and headwind.
def speed_against_current_headwind := v - speed_current - headwind_reduction

-- The statement to prove. 
theorem man's_speed_against_current_and_headwind :
  speed_against_current_headwind = 11.5 := by
  -- Using the conditions (which are already defined in lean expressions above), we can end the proof here.
  sorry

end NUMINAMATH_GPT_man_l1568_156816


namespace NUMINAMATH_GPT_highest_x_value_satisfies_equation_l1568_156857

theorem highest_x_value_satisfies_equation:
  ∃ x, x ≤ 4 ∧ (∀ x1, x1 ≤ 4 → x1 = 4 ↔ (15 * x1^2 - 40 * x1 + 18) / (4 * x1 - 3) + 7 * x1 = 9 * x1 - 2) :=
by
  sorry

end NUMINAMATH_GPT_highest_x_value_satisfies_equation_l1568_156857


namespace NUMINAMATH_GPT_range_of_a_l1568_156812

-- Definitions and theorems
theorem range_of_a (a : ℝ) : 
  (∀ (x y z : ℝ), x + y + z = 1 → abs (a - 2) ≤ x^2 + 2*y^2 + 3*z^2) → (16 / 11 ≤ a ∧ a ≤ 28 / 11) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1568_156812


namespace NUMINAMATH_GPT_original_population_l1568_156837

theorem original_population (P : ℕ) (h1 : 0.1 * (P : ℝ) + 0.2 * (0.9 * P) = 4500) : P = 6250 :=
sorry

end NUMINAMATH_GPT_original_population_l1568_156837


namespace NUMINAMATH_GPT_units_digit_24_pow_4_plus_42_pow_4_l1568_156817

theorem units_digit_24_pow_4_plus_42_pow_4 : 
    (24^4 + 42^4) % 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_24_pow_4_plus_42_pow_4_l1568_156817


namespace NUMINAMATH_GPT_no_quaint_two_digit_integers_l1568_156818

theorem no_quaint_two_digit_integers :
  ∀ x : ℕ, 10 ≤ x ∧ x < 100 ∧ (∃ a b : ℕ, x = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) →  ¬(10 * x.div 10 + x % 10 = (x.div 10) + (x % 10)^3) :=
by
  sorry

end NUMINAMATH_GPT_no_quaint_two_digit_integers_l1568_156818


namespace NUMINAMATH_GPT_K_time_for_distance_l1568_156884

theorem K_time_for_distance (s : ℝ) (hs : s > 0) :
  (let K_time := 45 / s
   let M_speed := s - 1 / 2
   let M_time := 45 / M_speed
   K_time = M_time - 3 / 4) -> K_time = 45 / s := 
by
  sorry

end NUMINAMATH_GPT_K_time_for_distance_l1568_156884


namespace NUMINAMATH_GPT_symmetric_circle_equation_l1568_156867

theorem symmetric_circle_equation :
  ∀ (x y : ℝ),
    (x^2 + y^2 - 6 * x + 8 * y + 24 = 0) →
    (x - 3 * y - 5 = 0) →
    (∃ x₀ y₀ : ℝ, (x₀ - 1)^2 + (y₀ - 2)^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_circle_equation_l1568_156867


namespace NUMINAMATH_GPT_cost_per_square_meter_l1568_156819

noncomputable def costPerSquareMeter 
  (length : ℝ) (breadth : ℝ) (width : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / ((length * width) + (breadth * width) - (width * width))

theorem cost_per_square_meter (H1 : length = 110)
                              (H2 : breadth = 60)
                              (H3 : width = 10)
                              (H4 : total_cost = 4800) : 
  costPerSquareMeter length breadth width total_cost = 3 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_square_meter_l1568_156819


namespace NUMINAMATH_GPT_frequency_of_group_of_samples_l1568_156830

def sample_capacity : ℝ := 32
def frequency_rate : ℝ := 0.125

theorem frequency_of_group_of_samples : frequency_rate * sample_capacity = 4 :=
by 
  sorry

end NUMINAMATH_GPT_frequency_of_group_of_samples_l1568_156830


namespace NUMINAMATH_GPT_value_of_x_l1568_156864

theorem value_of_x (x : ℝ) (h : 0.75 * 600 = 0.50 * x) : x = 900 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l1568_156864


namespace NUMINAMATH_GPT_stratified_sampling_l1568_156861

theorem stratified_sampling (total_students boys girls sample_size x y : ℕ)
  (h1 : total_students = 8)
  (h2 : boys = 6)
  (h3 : girls = 2)
  (h4 : sample_size = 4)
  (h5 : x + y = sample_size)
  (h6 : (x : ℚ) / boys = 3 / 4)
  (h7 : (y : ℚ) / girls = 1 / 4) :
  x = 3 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l1568_156861


namespace NUMINAMATH_GPT_number_of_people_l1568_156847

theorem number_of_people (clinks : ℕ) (h : clinks = 45) : ∃ x : ℕ, x * (x - 1) / 2 = clinks ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_l1568_156847


namespace NUMINAMATH_GPT_boys_girls_relationship_l1568_156805

theorem boys_girls_relationship (b g : ℕ) (h1 : b > 0) (h2 : g > 2) (h3 : ∀ n : ℕ, n < b → (n + 1) + 2 ≤ g) (h4 : b + 2 = g) : b = g - 2 := 
by
  sorry

end NUMINAMATH_GPT_boys_girls_relationship_l1568_156805


namespace NUMINAMATH_GPT_neg_prop_p_equiv_l1568_156871

variable {x : ℝ}

def prop_p : Prop := ∃ x ≥ 0, 2^x = 3

theorem neg_prop_p_equiv : ¬prop_p ↔ ∀ x ≥ 0, 2^x ≠ 3 :=
by sorry

end NUMINAMATH_GPT_neg_prop_p_equiv_l1568_156871


namespace NUMINAMATH_GPT_contractor_fine_per_absent_day_l1568_156863

noncomputable def fine_per_absent_day (total_days : ℕ) (pay_per_day : ℝ) (total_amount_received : ℝ) (days_absent : ℕ) : ℝ :=
  let days_worked := total_days - days_absent
  let earned := days_worked * pay_per_day
  let fine := (earned - total_amount_received) / days_absent
  fine

theorem contractor_fine_per_absent_day :
  fine_per_absent_day 30 25 425 10 = 7.5 := by
  sorry

end NUMINAMATH_GPT_contractor_fine_per_absent_day_l1568_156863


namespace NUMINAMATH_GPT_phone_prices_purchase_plans_l1568_156838

noncomputable def modelA_price : ℝ := 2000
noncomputable def modelB_price : ℝ := 1000

theorem phone_prices :
  (∀ x y : ℝ, (2 * x + y = 5000 ∧ 3 * x + 2 * y = 8000) → x = modelA_price ∧ y = modelB_price) :=
by
    intro x y
    intro h
    have h1 := h.1
    have h2 := h.2
    -- We would provide the detailed proof here
    sorry

theorem purchase_plans :
  (∀ a : ℕ, (4 ≤ a ∧ a ≤ 6) ↔ (24000 ≤ 2000 * a + 1000 * (20 - a) ∧ 2000 * a + 1000 * (20 - a) ≤ 26000)) :=
by
    intro a
    -- We would provide the detailed proof here
    sorry

end NUMINAMATH_GPT_phone_prices_purchase_plans_l1568_156838


namespace NUMINAMATH_GPT_seventh_term_arith_seq_l1568_156890

/-- 
The seventh term of an arithmetic sequence given that the sum of the first five terms 
is 15 and the sixth term is 7.
-/
theorem seventh_term_arith_seq (a d : ℚ) 
  (h1 : 5 * a + 10 * d = 15) 
  (h2 : a + 5 * d = 7) : 
  a + 6 * d = 25 / 3 := 
sorry

end NUMINAMATH_GPT_seventh_term_arith_seq_l1568_156890


namespace NUMINAMATH_GPT_zookeeper_feeding_problem_l1568_156844

noncomputable def feeding_ways : ℕ :=
  sorry

theorem zookeeper_feeding_problem :
  feeding_ways = 2880 := 
sorry

end NUMINAMATH_GPT_zookeeper_feeding_problem_l1568_156844


namespace NUMINAMATH_GPT_daily_wage_c_l1568_156834

-- Definitions according to the conditions
def days_worked_a : ℕ := 6
def days_worked_b : ℕ := 9
def days_worked_c : ℕ := 4

def ratio_wages : ℕ × ℕ × ℕ := (3, 4, 5)
def total_earning : ℕ := 1628

-- Goal: Prove that the daily wage of c is Rs. 110
theorem daily_wage_c : (5 * (total_earning / (18 + 36 + 20))) = 110 :=
by
  sorry

end NUMINAMATH_GPT_daily_wage_c_l1568_156834


namespace NUMINAMATH_GPT_minimum_red_points_for_square_l1568_156855

/-- Given a circle divided into 100 equal segments with points randomly colored red. 
Prove that the minimum number of red points needed to ensure at least four red points 
form the vertices of a square is 76. --/
theorem minimum_red_points_for_square (n : ℕ) (h : n = 100) (red_points : Finset ℕ)
  (hred : red_points.card ≥ 76) (hseg : ∀ i j : ℕ, i ≤ j → (j - i) % 25 ≠ 0 → ¬ (∃ a b c d : ℕ, 
  a ∈ red_points ∧ b ∈ red_points ∧ c ∈ red_points ∧ d ∈ red_points ∧ 
  (a + b + c + d) % n = 0)) : 
  ∃ a b c d : ℕ, a ∈ red_points ∧ b ∈ red_points ∧ c ∈ red_points ∧ d ∈ red_points ∧ 
  (a + b + c + d) % n = 0 :=
sorry

end NUMINAMATH_GPT_minimum_red_points_for_square_l1568_156855


namespace NUMINAMATH_GPT_maximal_regions_convex_quadrilaterals_l1568_156845

theorem maximal_regions_convex_quadrilaterals (n : ℕ) (hn : n ≥ 1) : 
  ∃ a_n : ℕ, a_n = 4*n^2 - 4*n + 2 :=
by
  sorry

end NUMINAMATH_GPT_maximal_regions_convex_quadrilaterals_l1568_156845


namespace NUMINAMATH_GPT_find_n_interval_l1568_156824

theorem find_n_interval :
  ∃ n : ℕ, n < 1000 ∧
  (∃ ghijkl : ℕ, (ghijkl < 999999) ∧ (ghijkl * n = 999999 * ghijkl)) ∧
  (∃ mnop : ℕ, (mnop < 9999) ∧ (mnop * (n + 5) = 9999 * mnop)) ∧
  151 ≤ n ∧ n ≤ 300 :=
sorry

end NUMINAMATH_GPT_find_n_interval_l1568_156824


namespace NUMINAMATH_GPT_system_of_equations_solution_system_of_inequalities_solution_l1568_156862

-- Problem (1): Solve the system of equations
theorem system_of_equations_solution :
  ∃ (x y : ℝ), (3 * (y - 2) = x - 1) ∧ (2 * (x - 1) = 5 * y - 8) ∧ (x = 7) ∧ (y = 4) :=
by
  sorry

-- Problem (2): Solve the system of linear inequalities
theorem system_of_inequalities_solution :
  ∃ (x : ℝ), (3 * x ≤ 2 * x + 3) ∧ ((x + 1) / 6 - 1 < (2 * (x + 1)) / 3) ∧ (-3 < x) ∧ (x ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_system_of_inequalities_solution_l1568_156862


namespace NUMINAMATH_GPT_convert_deg_to_rad_l1568_156814

theorem convert_deg_to_rad (deg_to_rad : ℝ → ℝ) (conversion_factor : deg_to_rad 1 = π / 180) :
  deg_to_rad (-300) = - (5 * π) / 3 :=
by
  sorry

end NUMINAMATH_GPT_convert_deg_to_rad_l1568_156814


namespace NUMINAMATH_GPT_age_ratio_3_2_l1568_156874

/-
Define variables: 
  L : ℕ -- Liam's current age
  M : ℕ -- Mia's current age
  y : ℕ -- number of years until the age ratio is 3:2
-/

theorem age_ratio_3_2 (L M : ℕ) 
  (h1 : L - 4 = 2 * (M - 4)) 
  (h2 : L - 10 = 3 * (M - 10)) 
  (h3 : ∃ y, (L + y) * 2 = (M + y) * 3) : 
  ∃ y, y = 8 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_3_2_l1568_156874


namespace NUMINAMATH_GPT_chelsea_total_time_l1568_156809

def num_batches := 4
def bake_time_per_batch := 20  -- minutes
def ice_time_per_batch := 30   -- minutes
def cupcakes_per_batch := 6
def additional_time_first_batch := 10 -- per cupcake
def additional_time_second_batch := 15 -- per cupcake
def additional_time_third_batch := 12 -- per cupcake
def additional_time_fourth_batch := 20 -- per cupcake

def total_bake_ice_time := bake_time_per_batch + ice_time_per_batch
def total_bake_ice_time_all_batches := total_bake_ice_time * num_batches

def total_additional_time_first_batch := additional_time_first_batch * cupcakes_per_batch
def total_additional_time_second_batch := additional_time_second_batch * cupcakes_per_batch
def total_additional_time_third_batch := additional_time_third_batch * cupcakes_per_batch
def total_additional_time_fourth_batch := additional_time_fourth_batch * cupcakes_per_batch

def total_additional_time := 
  total_additional_time_first_batch +
  total_additional_time_second_batch +
  total_additional_time_third_batch +
  total_additional_time_fourth_batch

def total_time := total_bake_ice_time_all_batches + total_additional_time

theorem chelsea_total_time : total_time = 542 := by
  sorry

end NUMINAMATH_GPT_chelsea_total_time_l1568_156809


namespace NUMINAMATH_GPT_g_of_1001_l1568_156878

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x * y) + x = x * g y + g x
axiom g_of_1 : g 1 = -3

theorem g_of_1001 : g 1001 = -2001 := 
by sorry

end NUMINAMATH_GPT_g_of_1001_l1568_156878


namespace NUMINAMATH_GPT_simplify_expression_l1568_156860

theorem simplify_expression (p q r s : ℝ) (hp : p ≠ 6) (hq : q ≠ 7) (hr : r ≠ 8) (hs : s ≠ 9) :
    (p - 6) / (8 - r) * (q - 7) / (6 - p) * (r - 8) / (7 - q) * (s - 9) / (9 - s) = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1568_156860


namespace NUMINAMATH_GPT_weekly_spending_l1568_156842

-- Definitions based on the conditions outlined in the original problem
def weekly_allowance : ℝ := 50
def hours_per_week : ℕ := 30
def hourly_wage : ℝ := 9
def weeks_per_year : ℕ := 52
def first_year_allowance : ℝ := weekly_allowance * weeks_per_year
def second_year_earnings : ℝ := (hourly_wage * hours_per_week) * weeks_per_year
def total_car_cost : ℝ := 15000
def additional_needed : ℝ := 2000
def total_savings : ℝ := first_year_allowance + second_year_earnings

-- The amount Thomas needs over what he has saved
def total_needed : ℝ := total_savings + additional_needed
def amount_spent_on_self : ℝ := total_needed - total_car_cost
def total_weeks : ℕ := 2 * weeks_per_year

theorem weekly_spending :
  amount_spent_on_self / total_weeks = 35 := by
  sorry

end NUMINAMATH_GPT_weekly_spending_l1568_156842


namespace NUMINAMATH_GPT_product_of_t_values_l1568_156808

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end NUMINAMATH_GPT_product_of_t_values_l1568_156808


namespace NUMINAMATH_GPT_bed_width_is_4_feet_l1568_156886

def total_bags : ℕ := 16
def soil_per_bag : ℕ := 4
def bed_length : ℝ := 8
def bed_height : ℝ := 1
def num_beds : ℕ := 2

theorem bed_width_is_4_feet :
  (total_bags * soil_per_bag / num_beds) = (bed_length * 4 * bed_height) :=
by
  sorry

end NUMINAMATH_GPT_bed_width_is_4_feet_l1568_156886


namespace NUMINAMATH_GPT_difference_between_largest_and_smallest_l1568_156804

def largest_number := 9765310
def smallest_number := 1035679
def expected_difference := 8729631
def digits := [3, 9, 6, 0, 5, 1, 7]

theorem difference_between_largest_and_smallest :
  (largest_number - smallest_number) = expected_difference :=
sorry

end NUMINAMATH_GPT_difference_between_largest_and_smallest_l1568_156804


namespace NUMINAMATH_GPT_probability_good_or_excellent_l1568_156868

noncomputable def P_H1 : ℚ := 5 / 21
noncomputable def P_H2 : ℚ := 10 / 21
noncomputable def P_H3 : ℚ := 6 / 21

noncomputable def P_A_given_H1 : ℚ := 1
noncomputable def P_A_given_H2 : ℚ := 1
noncomputable def P_A_given_H3 : ℚ := 1 / 3

noncomputable def P_A : ℚ := 
  P_H1 * P_A_given_H1 + 
  P_H2 * P_A_given_H2 + 
  P_H3 * P_A_given_H3

theorem probability_good_or_excellent : P_A = 17 / 21 :=
by
  sorry

end NUMINAMATH_GPT_probability_good_or_excellent_l1568_156868


namespace NUMINAMATH_GPT_find_a_value_l1568_156802

theorem find_a_value (a x y : ℝ) (h1 : x = 2) (h2 : y = 3) (h3 : a * x - 3 * y = 3) : a = 6 :=
by
  rw [h1, h2] at h3 -- Substitute x and y values into the equation
  sorry -- The proof is omitted as per instructions.

end NUMINAMATH_GPT_find_a_value_l1568_156802


namespace NUMINAMATH_GPT_current_age_of_son_l1568_156881

variables (S F : ℕ)

-- Define the conditions
def condition1 : Prop := F = 3 * S
def condition2 : Prop := F - 8 = 4 * (S - 8)

-- The theorem statement
theorem current_age_of_son (h1 : condition1 S F) (h2 : condition2 S F) : S = 24 :=
sorry

end NUMINAMATH_GPT_current_age_of_son_l1568_156881


namespace NUMINAMATH_GPT_new_oranges_added_l1568_156836

def initial_oranges : Nat := 31
def thrown_away_oranges : Nat := 9
def final_oranges : Nat := 60
def remaining_oranges : Nat := initial_oranges - thrown_away_oranges
def new_oranges (initial_oranges thrown_away_oranges final_oranges : Nat) : Nat := 
  final_oranges - (initial_oranges - thrown_away_oranges)

theorem new_oranges_added :
  new_oranges initial_oranges thrown_away_oranges final_oranges = 38 := by
  sorry

end NUMINAMATH_GPT_new_oranges_added_l1568_156836


namespace NUMINAMATH_GPT_no_term_un_eq_neg1_l1568_156865

theorem no_term_un_eq_neg1 (p : ℕ) [hp_prime: Fact (Nat.Prime p)] (hp_odd: p % 2 = 1) (hp_not_five: p ≠ 5) :
  ∀ n : ℕ, ∀ u : ℕ → ℤ, ((u 0 = 0) ∧ (u 1 = 1) ∧ (∀ k, k ≥ 2 → u (k-2) = 2 * u (k-1) - p * u k)) → 
    (u n ≠ -1) :=
  sorry

end NUMINAMATH_GPT_no_term_un_eq_neg1_l1568_156865


namespace NUMINAMATH_GPT_calculate_total_weight_AlBr3_l1568_156833

-- Definitions for the atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_Br : ℝ := 79.90

-- Definition for the molecular weight of AlBr3
def molecular_weight_AlBr3 : ℝ := atomic_weight_Al + 3 * atomic_weight_Br

-- Number of moles
def number_of_moles : ℝ := 5

-- Total weight of 5 moles of AlBr3
def total_weight_5_moles_AlBr3 : ℝ := molecular_weight_AlBr3 * number_of_moles

-- Desired result
def expected_total_weight : ℝ := 1333.40

-- Statement to prove that total_weight_5_moles_AlBr3 equals the expected total weight
theorem calculate_total_weight_AlBr3 :
  total_weight_5_moles_AlBr3 = expected_total_weight :=
sorry

end NUMINAMATH_GPT_calculate_total_weight_AlBr3_l1568_156833


namespace NUMINAMATH_GPT_expansion_identity_l1568_156896

theorem expansion_identity : 121 + 2 * 11 * 9 + 81 = 400 := by
  sorry

end NUMINAMATH_GPT_expansion_identity_l1568_156896


namespace NUMINAMATH_GPT_angle_B_is_pi_over_3_range_of_expression_l1568_156872

variable {A B C a b c : ℝ}

-- Conditions
def sides_opposite_angles (A B C : ℝ) (a b c : ℝ): Prop :=
  (2 * c - a) * Real.cos B - b * Real.cos A = 0

-- Part 1: Prove B = π/3
theorem angle_B_is_pi_over_3 (h : sides_opposite_angles A B C a b c) : 
    B = Real.pi / 3 := 
  sorry

-- Part 2: Prove the range of sqrt(3) * (sin A + sin(C - π/6)) is (1, 2]
theorem range_of_expression (h : 0 < A ∧ A < 2 * Real.pi / 3) : 
    (1:ℝ) < Real.sqrt 3 * (Real.sin A + Real.sin (C - Real.pi / 6)) 
    ∧ Real.sqrt 3 * (Real.sin A + Real.sin (C - Real.pi / 6)) ≤ 2 := 
  sorry

end NUMINAMATH_GPT_angle_B_is_pi_over_3_range_of_expression_l1568_156872


namespace NUMINAMATH_GPT_kitchen_upgrade_cost_l1568_156899

-- Define the number of cabinet knobs and their cost
def num_knobs : ℕ := 18
def cost_per_knob : ℝ := 2.50

-- Define the number of drawer pulls and their cost
def num_pulls : ℕ := 8
def cost_per_pull : ℝ := 4.00

-- Calculate the total cost of the knobs
def total_cost_knobs : ℝ := num_knobs * cost_per_knob

-- Calculate the total cost of the pulls
def total_cost_pulls : ℝ := num_pulls * cost_per_pull

-- Calculate the total cost of the kitchen upgrade
def total_cost : ℝ := total_cost_knobs + total_cost_pulls

-- Theorem statement
theorem kitchen_upgrade_cost : total_cost = 77 := by
  sorry

end NUMINAMATH_GPT_kitchen_upgrade_cost_l1568_156899


namespace NUMINAMATH_GPT_two_digit_number_representation_l1568_156829

theorem two_digit_number_representation (a b : ℕ) (ha : a < 10) (hb : b < 10) : 10 * b + a = d :=
  sorry

end NUMINAMATH_GPT_two_digit_number_representation_l1568_156829


namespace NUMINAMATH_GPT_work_completion_l1568_156828

theorem work_completion (a b : ℕ) (hab : a = 2 * b) (hwork_together : (1/a + 1/b) = 1/8) : b = 24 := by
  sorry

end NUMINAMATH_GPT_work_completion_l1568_156828


namespace NUMINAMATH_GPT_min_colors_rect_condition_l1568_156827

theorem min_colors_rect_condition (n : ℕ) (hn : n ≥ 2) :
  ∃ k : ℕ, (∀ (coloring : Fin n → Fin n → Fin k), 
           (∀ i j, coloring i j < k) → 
           (∀ c, ∃ i j, coloring i j = c) →
           (∃ i1 i2 j1 j2, i1 ≠ i2 ∧ j1 ≠ j2 ∧ 
                            coloring i1 j1 ≠ coloring i1 j2 ∧ 
                            coloring i1 j1 ≠ coloring i2 j1 ∧ 
                            coloring i1 j2 ≠ coloring i2 j2 ∧ 
                            coloring i2 j1 ≠ coloring i2 j2)) → 
           k = 2 * n :=
sorry

end NUMINAMATH_GPT_min_colors_rect_condition_l1568_156827


namespace NUMINAMATH_GPT_triangle_perfect_square_l1568_156877

theorem triangle_perfect_square (a b c : ℤ) (h : ∃ h₁ h₂ h₃ : ℤ, (1/2) * a * h₁ = (1/2) * b * h₂ ∧ (1/2) * b * h₂ = (1/2) * c * h₃ ∧ (h₁ = h₂ + h₃)) :
  ∃ k : ℤ, a^2 + b^2 + c^2 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perfect_square_l1568_156877


namespace NUMINAMATH_GPT_math_problem_l1568_156893

def f (x : ℝ) : ℝ := sorry

theorem math_problem (n s : ℕ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y))
  (hn : n = 1)
  (hs : s = 6) :
  n * s = 6 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1568_156893


namespace NUMINAMATH_GPT_clothing_price_decrease_l1568_156820

theorem clothing_price_decrease (P : ℝ) (h₁ : P > 0) :
  let price_first_sale := (4 / 5) * P
  let price_second_sale := (1 / 2) * P
  let price_difference := price_first_sale - price_second_sale
  let percent_decrease := (price_difference / price_first_sale) * 100
  percent_decrease = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_clothing_price_decrease_l1568_156820


namespace NUMINAMATH_GPT_algebra_expression_value_l1568_156856

theorem algebra_expression_value (m : ℝ) (hm : m^2 - m - 1 = 0) : m^2 - m + 2008 = 2009 :=
by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l1568_156856


namespace NUMINAMATH_GPT_point_A_in_fourth_quadrant_l1568_156875

def Point := ℤ × ℤ

def is_in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def point_A : Point := (3, -2)
def point_B : Point := (2, 5)
def point_C : Point := (-1, -2)
def point_D : Point := (-2, 2)

theorem point_A_in_fourth_quadrant : is_in_fourth_quadrant point_A :=
  sorry

end NUMINAMATH_GPT_point_A_in_fourth_quadrant_l1568_156875


namespace NUMINAMATH_GPT_trigonometric_identity_l1568_156888

open Real

noncomputable def acute (x : ℝ) := 0 < x ∧ x < π / 2

theorem trigonometric_identity 
  {α β : ℝ} (hα : acute α) (hβ : acute β) (h : cos α > sin β) :
  α + β < π / 2 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1568_156888


namespace NUMINAMATH_GPT_minimum_value_of_f_l1568_156849

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 + 8 * x + 13) / (6 * (1 + Real.exp (-x)))

theorem minimum_value_of_f : ∀ x : ℝ, 0 ≤ x → f x ≥ f 0 :=
by
  intro x hx
  unfold f
  admit

end NUMINAMATH_GPT_minimum_value_of_f_l1568_156849


namespace NUMINAMATH_GPT_find_m_value_l1568_156815

def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem find_m_value :
  ∃ m : ℝ, (∀ x ∈ (Set.Icc 0 3), f x m ≤ 1) ∧ (∃ x ∈ (Set.Icc 0 3), f x m = 1) ↔ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l1568_156815


namespace NUMINAMATH_GPT_gcd_1230_990_l1568_156858

theorem gcd_1230_990 : Int.gcd 1230 990 = 30 := by
  sorry

end NUMINAMATH_GPT_gcd_1230_990_l1568_156858


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l1568_156853

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1 / 3) :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l1568_156853


namespace NUMINAMATH_GPT_intersection_height_correct_l1568_156806

noncomputable def intersection_height 
  (height_pole_1 height_pole_2 distance : ℝ) : ℝ := 
  let slope_1 := -(height_pole_1 / distance)
  let slope_2 := height_pole_2 / distance
  let y_intercept_1 := height_pole_1
  let y_intercept_2 := 0
  let x_intersection := height_pole_1 / (slope_2 - slope_1)
  let y_intersection := slope_2 * x_intersection + y_intercept_2
  y_intersection

theorem intersection_height_correct 
  : intersection_height 30 90 150 = 22.5 := 
by sorry

end NUMINAMATH_GPT_intersection_height_correct_l1568_156806


namespace NUMINAMATH_GPT_neg_exists_exp_l1568_156889

theorem neg_exists_exp (p : Prop) :
  (¬ (∃ x : ℝ, Real.exp x < 0)) = (∀ x : ℝ, Real.exp x ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_neg_exists_exp_l1568_156889


namespace NUMINAMATH_GPT_value_of_x_is_4_l1568_156811

variable {A B C D E F G H P : ℕ}

theorem value_of_x_is_4 (h1 : 5 + A + B = 19)
                        (h2 : A + B + C = 19)
                        (h3 : C + D + E = 19)
                        (h4 : D + E + F = 19)
                        (h5 : F + x + G = 19)
                        (h6 : x + G + H = 19)
                        (h7 : H + P + 10 = 19) :
                        x = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_is_4_l1568_156811


namespace NUMINAMATH_GPT_percentage_of_profit_if_no_discount_l1568_156895

-- Conditions
def discount : ℝ := 0.05
def profit_w_discount : ℝ := 0.216
def cost_price : ℝ := 100
def expected_profit : ℝ := 28

-- Proof statement
theorem percentage_of_profit_if_no_discount :
  ∃ (marked_price selling_price_no_discount : ℝ),
    selling_price_no_discount = marked_price ∧
    (marked_price - cost_price) / cost_price * 100 = expected_profit :=
by
  -- Definitions and logic will go here
  sorry

end NUMINAMATH_GPT_percentage_of_profit_if_no_discount_l1568_156895


namespace NUMINAMATH_GPT_algebraic_expression_l1568_156843

def a (x : ℕ) := 2005 * x + 2009
def b (x : ℕ) := 2005 * x + 2010
def c (x : ℕ) := 2005 * x + 2011

theorem algebraic_expression (x : ℕ) : 
  a x ^ 2 + b x ^ 2 + c x ^ 2 - a x * b x - b x * c x - c x * a x = 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_l1568_156843


namespace NUMINAMATH_GPT_train_length_is_300_l1568_156846

noncomputable def speed_kmph : Float := 90
noncomputable def speed_mps : Float := (speed_kmph * 1000) / 3600
noncomputable def time_sec : Float := 12
noncomputable def length_of_train : Float := speed_mps * time_sec

theorem train_length_is_300 : length_of_train = 300 := by
  sorry

end NUMINAMATH_GPT_train_length_is_300_l1568_156846


namespace NUMINAMATH_GPT_three_digit_odd_number_is_803_l1568_156803

theorem three_digit_odd_number_is_803 :
  ∃ (a b c : ℕ), 0 < a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ c % 2 = 1 ∧
  100 * a + 10 * b + c = 803 ∧ (100 * a + 10 * b + c) / 11 = a^2 + b^2 + c^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_three_digit_odd_number_is_803_l1568_156803


namespace NUMINAMATH_GPT_robert_finite_moves_l1568_156835

noncomputable def onlyFiniteMoves (numbers : List ℕ) : Prop :=
  ∀ (a b : ℕ), a > b → ∃ (moves : ℕ), moves < numbers.length

theorem robert_finite_moves (numbers : List ℕ) :
  onlyFiniteMoves numbers := sorry

end NUMINAMATH_GPT_robert_finite_moves_l1568_156835


namespace NUMINAMATH_GPT_find_number_l1568_156850

-- Define the conditions and the theorem
theorem find_number (number : ℝ)
  (h₁ : ∃ w : ℝ, w = (69.28 * number) / 0.03 ∧ abs (w - 9.237333333333334) ≤ 1e-10) :
  abs (number - 0.004) ≤ 1e-10 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1568_156850


namespace NUMINAMATH_GPT_point_in_second_quadrant_l1568_156883

def P : ℝ × ℝ := (-5, 4)

theorem point_in_second_quadrant (p : ℝ × ℝ) (hx : p.1 = -5) (hy : p.2 = 4) : p.1 < 0 ∧ p.2 > 0 :=
by
  sorry

example : P.1 < 0 ∧ P.2 > 0 :=
  point_in_second_quadrant P rfl rfl

end NUMINAMATH_GPT_point_in_second_quadrant_l1568_156883


namespace NUMINAMATH_GPT_stationery_sales_l1568_156826

theorem stationery_sales :
  let pen_percentage : ℕ := 42
  let pencil_percentage : ℕ := 27
  let total_sales_percentage : ℕ := 100
  total_sales_percentage - (pen_percentage + pencil_percentage) = 31 :=
by
  sorry

end NUMINAMATH_GPT_stationery_sales_l1568_156826


namespace NUMINAMATH_GPT_Eve_total_running_distance_l1568_156894

def Eve_walked_distance := 0.6

def Eve_ran_distance := Eve_walked_distance + 0.1

theorem Eve_total_running_distance : Eve_ran_distance = 0.7 := 
by sorry

end NUMINAMATH_GPT_Eve_total_running_distance_l1568_156894


namespace NUMINAMATH_GPT_a_2_value_l1568_156839

theorem a_2_value (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) (x : ℝ) :
  x^3 + x^10 = a + a1 * (x+1) + a2 * (x+1)^2 + a3 * (x+1)^3 + a4 * (x+1)^4 + a5 * (x+1)^5 +
  a6 * (x+1)^6 + a7 * (x+1)^7 + a8 * (x+1)^8 + a9 * (x+1)^9 + a10 * (x+1)^10 → 
  a2 = 42 :=
by
  sorry

end NUMINAMATH_GPT_a_2_value_l1568_156839


namespace NUMINAMATH_GPT_angle_measure_l1568_156840

theorem angle_measure (x : ℝ) 
  (h1 : 5 * x + 12 = 180 - x) : x = 28 := by
  sorry

end NUMINAMATH_GPT_angle_measure_l1568_156840


namespace NUMINAMATH_GPT_time_after_1750_minutes_is_1_10_pm_l1568_156832

def add_minutes_to_time (hours : Nat) (minutes : Nat) : Nat × Nat :=
  let total_minutes := hours * 60 + minutes
  (total_minutes / 60, total_minutes % 60)

def time_after_1750_minutes (current_hour : Nat) (current_minute : Nat) : Nat × Nat :=
  let (new_hour, new_minute) := add_minutes_to_time current_hour current_minute
  let final_hour := (new_hour + 1750 / 60) % 24
  let final_minute := (new_minute + 1750 % 60) % 60
  (final_hour, final_minute)

theorem time_after_1750_minutes_is_1_10_pm : 
  time_after_1750_minutes 8 0 = (13, 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_time_after_1750_minutes_is_1_10_pm_l1568_156832


namespace NUMINAMATH_GPT_interest_rate_borrowed_l1568_156869

variables {P : Type} [LinearOrderedField P]

def borrowed_amount : P := 9000
def lent_interest_rate : P := 0.06
def gain_per_year : P := 180
def per_cent : P := 100

theorem interest_rate_borrowed (r : P) (h : borrowed_amount * lent_interest_rate - gain_per_year = borrowed_amount * r) : 
  r = 0.04 :=
by sorry

end NUMINAMATH_GPT_interest_rate_borrowed_l1568_156869


namespace NUMINAMATH_GPT_find_starting_number_l1568_156813

theorem find_starting_number (n : ℕ) (h : ((28 + n) / 2) = 18) : n = 8 :=
sorry

end NUMINAMATH_GPT_find_starting_number_l1568_156813


namespace NUMINAMATH_GPT_angles_supplementary_l1568_156851

theorem angles_supplementary (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : ∃ S : Finset ℕ, S.card = 17 ∧ (∀ a ∈ S, ∃ k : ℕ, k * (180 / (k + 1)) = a ∧ A = a) :=
by
  sorry

end NUMINAMATH_GPT_angles_supplementary_l1568_156851


namespace NUMINAMATH_GPT_john_sells_20_woodburnings_l1568_156859

variable (x : ℕ)

theorem john_sells_20_woodburnings (price_per_woodburning cost profit : ℤ) 
  (h1 : price_per_woodburning = 15) (h2 : cost = 100) (h3 : profit = 200) :
  (profit = price_per_woodburning * x - cost) → 
  x = 20 :=
by
  intros h_profit
  rw [h1, h2, h3] at h_profit
  linarith

end NUMINAMATH_GPT_john_sells_20_woodburnings_l1568_156859
