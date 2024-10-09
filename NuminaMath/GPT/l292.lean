import Mathlib

namespace exponent_multiplication_l292_29233

theorem exponent_multiplication (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (3^a)^b = 3^3) : 3^a * 3^b = 3^4 :=
by
  sorry

end exponent_multiplication_l292_29233


namespace total_chickens_l292_29231

theorem total_chickens (hens : ℕ) (roosters : ℕ) (h1 : hens = 52) (h2 : roosters = hens + 16) : hens + roosters = 120 :=
by
  rw [h1, h2]
  norm_num
  sorry

end total_chickens_l292_29231


namespace triangle_at_most_one_obtuse_l292_29289

-- Define the notion of a triangle and obtuse angle
def isTriangle (A B C: ℝ) : Prop := (A + B > C) ∧ (A + C > B) ∧ (B + C > A)
def isObtuseAngle (theta: ℝ) : Prop := 90 < theta ∧ theta < 180

-- A theorem to prove that a triangle cannot have more than one obtuse angle 
theorem triangle_at_most_one_obtuse (A B C: ℝ) (angleA angleB angleC : ℝ) 
    (h1 : isTriangle A B C)
    (h2 : isObtuseAngle angleA)
    (h3 : isObtuseAngle angleB)
    (h4 : angleA + angleB + angleC = 180):
    false :=
by
  sorry

end triangle_at_most_one_obtuse_l292_29289


namespace simplify_polynomial_l292_29206

theorem simplify_polynomial (s : ℝ) :
  (2 * s ^ 2 + 5 * s - 3) - (2 * s ^ 2 + 9 * s - 6) = -4 * s + 3 :=
by 
  sorry

end simplify_polynomial_l292_29206


namespace part_1_part_2_l292_29224

-- Define proposition p
def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) ∧ (x^2 - (a + 2) * x + 2 * a = 0)

-- Proposition q: x₁ and x₂ are two real roots of the equation x^2 - 2mx - 3 = 0
def proposition_q (m x₁ x₂ : ℝ) : Prop :=
  x₁ ^ 2 - 2 * m * x₁ - 3 = 0 ∧ x₂ ^ 2 - 2 * m * x₂ - 3 = 0

-- Inequality condition
def inequality_condition (a m x₁ x₂ : ℝ) : Prop :=
  a ^ 2 - 3 * a ≥ abs (x₁ - x₂)

-- Part 1: If proposition p is true, find the range of the real number a
theorem part_1 (a : ℝ) (h_p : proposition_p a) : -1 < a ∧ a < 1 :=
  sorry

-- Part 2: If exactly one of propositions p or q is true, find the range of the real number a
theorem part_2 (a m x₁ x₂ : ℝ) (h_p_or_q : (proposition_p a ∧ ¬(proposition_q m x₁ x₂)) ∨ (¬(proposition_p a) ∧ (proposition_q m x₁ x₂))) : (a < 1) ∨ (a ≥ 4) :=
  sorry

end part_1_part_2_l292_29224


namespace simplify_expression_1_simplify_expression_2_l292_29228

section Problem1
variables (a b c : ℝ) (h1 : c ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0)

theorem simplify_expression_1 :
  ((a^2 * b / (-c))^3 * (c^2 / (- (a * b)))^2 / (b * c / a)^4)
  = - (a^10 / (b^3 * c^7)) :=
by sorry
end Problem1

section Problem2
variables (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ a) (h3 : b ≠ 0)

theorem simplify_expression_2 :
  ((2 / (a^2 - b^2) - 1 / (a^2 - a * b)) / (a / (a + b))) = 1 / a^2 :=
by sorry
end Problem2

end simplify_expression_1_simplify_expression_2_l292_29228


namespace average_mileage_city_l292_29282

variable (total_distance : ℝ) (gallons : ℝ) (highway_mpg : ℝ) (city_mpg : ℝ)

-- The given conditions
def conditions : Prop := (total_distance = 280.6) ∧ (gallons = 23) ∧ (highway_mpg = 12.2)

-- The theorem to prove
theorem average_mileage_city (h : conditions total_distance gallons highway_mpg) :
  total_distance / gallons = 12.2 :=
sorry

end average_mileage_city_l292_29282


namespace fiona_correct_answers_l292_29285

-- 5 marks for each correct answer in Questions 1-15
def marks_questions_1_to_15 (correct1 : ℕ) : ℕ := 5 * correct1

-- 6 marks for each correct answer in Questions 16-25
def marks_questions_16_to_25 (correct2 : ℕ) : ℕ := 6 * correct2

-- 1 mark penalty for incorrect answers in Questions 16-20
def penalty_questions_16_to_20 (incorrect1 : ℕ) : ℕ := incorrect1

-- 2 mark penalty for incorrect answers in Questions 21-25
def penalty_questions_21_to_25 (incorrect2 : ℕ) : ℕ := 2 * incorrect2

-- Total marks given correct and incorrect answers
def total_marks (correct1 correct2 incorrect1 incorrect2 : ℕ) : ℕ :=
  marks_questions_1_to_15 correct1 +
  marks_questions_16_to_25 correct2 -
  penalty_questions_16_to_20 incorrect1 -
  penalty_questions_21_to_25 incorrect2

-- Fiona's total score
def fionas_total_score : ℕ := 80

-- The proof problem: Fiona answered 16 questions correctly
theorem fiona_correct_answers (correct1 correct2 incorrect1 incorrect2 : ℕ) :
  total_marks correct1 correct2 incorrect1 incorrect2 = fionas_total_score → 
  (correct1 + correct2 = 16) := sorry

end fiona_correct_answers_l292_29285


namespace total_distance_l292_29288

theorem total_distance (D : ℝ) 
  (h₁ : 60 * (D / 2 / 60) = D / 2) 
  (h₂ : 40 * ((D / 2) / 4 / 40) = D / 8) 
  (h₃ : 50 * (105 / 50) = 105)
  (h₄ : D = D / 2 + D / 8 + 105) : 
  D = 280 :=
by sorry

end total_distance_l292_29288


namespace quilt_shaded_fraction_l292_29204

theorem quilt_shaded_fraction :
  let total_squares := 16
  let shaded_squares := 8
  let fully_shaded := 4
  let half_shaded := 4
  let shaded_area := fully_shaded + half_shaded * 1 / 2
  shaded_area / total_squares = 3 / 8 :=
by
  sorry

end quilt_shaded_fraction_l292_29204


namespace old_fridge_cost_l292_29270

-- Define the daily cost of Kurt's old refrigerator
variable (x : ℝ)

-- Define the conditions given in the problem
def new_fridge_cost_per_day : ℝ := 0.45
def savings_per_month : ℝ := 12
def days_in_month : ℝ := 30

-- State the theorem to prove
theorem old_fridge_cost :
  30 * x - 30 * new_fridge_cost_per_day = savings_per_month → x = 0.85 := 
by
  intro h
  sorry

end old_fridge_cost_l292_29270


namespace number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other_l292_29237

def number_of_seatings (n : ℕ) : ℕ := Nat.factorial n

theorem number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other :
  let k := 2      -- Kolya and Olya as a unit
  let remaining := 3 -- The remaining people
  let pairs := 4 -- Pairs of seats that Kolya and Olya can take
  let arrangements_kolya_olya := pairs * 2 -- Each pair can have Kolya and Olya in 2 arrangements
  let arrangements_remaining := number_of_seatings remaining 
  arrangements_kolya_olya * arrangements_remaining = 48 := by
{
  -- This would be the location for the proof implementation
  sorry
}

end number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other_l292_29237


namespace solve_cubic_root_eq_l292_29208

theorem solve_cubic_root_eq (x : ℝ) : (∃ x, 3 - x / 3 = -8) -> x = 33 :=
by
  sorry

end solve_cubic_root_eq_l292_29208


namespace volume_of_rectangular_parallelepiped_l292_29213

theorem volume_of_rectangular_parallelepiped (x y z p q r : ℝ) 
  (h1 : p = x * y) 
  (h2 : q = x * z) 
  (h3 : r = y * z) : 
  x * y * z = Real.sqrt (p * q * r) :=
by
  sorry

end volume_of_rectangular_parallelepiped_l292_29213


namespace alloy_chromium_amount_l292_29269

theorem alloy_chromium_amount
  (x : ℝ) -- The amount of the first alloy used (in kg)
  (h1 : 0.10 * x + 0.08 * 35 = 0.086 * (x + 35)) -- Condition based on percentages of chromium
  : x = 15 := 
by
  sorry

end alloy_chromium_amount_l292_29269


namespace count_distinct_even_numbers_l292_29236

theorem count_distinct_even_numbers : 
  ∃ c, c = 37 ∧ ∀ d1 d2 d3, d1 ≠ d2 → d2 ≠ d3 → d1 ≠ d3 → (d1 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (d2 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (d3 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (∃ n : ℕ, n / 10 ^ 2 = d1 ∧ (n / 10) % 10 = d2 ∧ n % 10 = d3 ∧ n % 2 = 0) :=
sorry

end count_distinct_even_numbers_l292_29236


namespace twice_minus_three_algebraic_l292_29220

def twice_minus_three (x : ℝ) : ℝ := 2 * x - 3

theorem twice_minus_three_algebraic (x : ℝ) : 
  twice_minus_three x = 2 * x - 3 :=
by sorry

end twice_minus_three_algebraic_l292_29220


namespace john_website_days_l292_29240

theorem john_website_days
  (monthly_visits : ℕ)
  (cents_per_visit : ℝ)
  (dollars_per_day : ℝ)
  (monthly_visits_eq : monthly_visits = 30000)
  (cents_per_visit_eq : cents_per_visit = 0.01)
  (dollars_per_day_eq : dollars_per_day = 10) :
  (monthly_visits / (dollars_per_day / cents_per_visit)) = 30 :=
by
  sorry

end john_website_days_l292_29240


namespace arithmetic_expression_l292_29214

theorem arithmetic_expression : 5 + 12 / 3 - 3 ^ 2 + 1 = 1 := by
  sorry

end arithmetic_expression_l292_29214


namespace solve_inequalities_l292_29242

theorem solve_inequalities (x : ℝ) :
  (3 * x^2 - x > 4) ∧ (x < 3) ↔ (1 < x ∧ x < 3) := 
by 
  sorry

end solve_inequalities_l292_29242


namespace reflected_point_correct_l292_29217

-- Defining the original point coordinates
def original_point : ℝ × ℝ := (3, -5)

-- Defining the transformation function
def reflect_across_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

-- Proving the point after reflection is as expected
theorem reflected_point_correct : reflect_across_y_axis original_point = (-3, -5) :=
by
  sorry

end reflected_point_correct_l292_29217


namespace gcd_and_lcm_of_18_and_24_l292_29212

-- Definitions of gcd and lcm for the problem's context
def my_gcd (a b : ℕ) : ℕ := a.gcd b
def my_lcm (a b : ℕ) : ℕ := a.lcm b

-- Constants given in the problem
def a := 18
def b := 24

-- Proof problem statement
theorem gcd_and_lcm_of_18_and_24 : my_gcd a b = 6 ∧ my_lcm a b = 72 := by
  sorry

end gcd_and_lcm_of_18_and_24_l292_29212


namespace fg_minus_gf_l292_29268

noncomputable def f (x : ℝ) : ℝ := 8 * x - 12
noncomputable def g (x : ℝ) : ℝ := x / 4 - 1

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = -16 :=
by
  -- We skip the proof.
  sorry

end fg_minus_gf_l292_29268


namespace totalInterest_l292_29280

-- Definitions for the amounts and interest rates
def totalInvestment : ℝ := 22000
def investedAt18 : ℝ := 7000
def rate18 : ℝ := 0.18
def rate14 : ℝ := 0.14

-- Calculations as conditions
def interestFrom18 (p r : ℝ) : ℝ := p * r
def investedAt14 (total inv18 : ℝ) : ℝ := total - inv18
def interestFrom14 (p r : ℝ) : ℝ := p * r

-- Proof statement
theorem totalInterest : interestFrom18 investedAt18 rate18 + interestFrom14 (investedAt14 totalInvestment investedAt18) rate14 = 3360 :=
by
  sorry

end totalInterest_l292_29280


namespace rate_of_interest_l292_29221

noncomputable def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ (n : ℝ)

theorem rate_of_interest (P : ℝ) (r : ℝ) (A : ℕ → ℝ) :
  A 2 = compound_interest P r 2 →
  A 3 = compound_interest P r 3 →
  A 2 = 2420 →
  A 3 = 2662 →
  r = 10 :=
by
  sorry

end rate_of_interest_l292_29221


namespace original_people_in_room_l292_29286

theorem original_people_in_room (x : ℝ) (h1 : x / 3 * 2 / 2 = 18) : x = 54 :=
sorry

end original_people_in_room_l292_29286


namespace first_day_of_month_is_thursday_l292_29216

theorem first_day_of_month_is_thursday :
  (27 - 7 - 7 - 7 + 1) % 7 = 4 :=
by
  sorry

end first_day_of_month_is_thursday_l292_29216


namespace zero_of_f_when_m_is_neg1_monotonicity_of_f_m_gt_neg1_l292_29209

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  x - 1/x - 2 * m * Real.log x

theorem zero_of_f_when_m_is_neg1 : ∃ x > 0, f x (-1) = 0 :=
  by
    use 1
    sorry

theorem monotonicity_of_f_m_gt_neg1 (m : ℝ) (hm : m > -1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x m ≤ f y m) ∨
  (∃ a b : ℝ, 0 < a ∧ a < b ∧
    (∀ x : ℝ, 0 < x ∧ x < a → f x m ≤ f a m) ∧
    (∀ x : ℝ, a < x ∧ x < b → f a m ≥ f x m) ∧
    (∀ x : ℝ, b < x → f b m ≤ f x m)) :=
  by
    cases lt_or_le m 1 with
    | inl hlt =>
        left
        intros x y hx hy hxy
        sorry
    | inr hle =>
        right
        use m - Real.sqrt (m^2 - 1), m + Real.sqrt (m^2 - 1)
        sorry

end zero_of_f_when_m_is_neg1_monotonicity_of_f_m_gt_neg1_l292_29209


namespace marta_total_spent_l292_29249

theorem marta_total_spent :
  let sale_book_cost := 5 * 10
  let online_book_cost := 40
  let bookstore_book_cost := 3 * online_book_cost
  let total_spent := sale_book_cost + online_book_cost + bookstore_book_cost
  total_spent = 210 := sorry

end marta_total_spent_l292_29249


namespace option_C_is_quadratic_l292_29297

-- Define the conditions
def option_A (x : ℝ) : Prop := 2 * x = 3
def option_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def option_C (x : ℝ) : Prop := (4 * x - 3) * (3 * x + 1) = 0
def option_D (x : ℝ) : Prop := (x + 3) * (x - 2) = (x - 2) * (x + 1)

-- Define what it means to be a quadratic equation
def is_quadratic (f : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, (∀ x, f x = (a * x^2 + b * x + c = 0)) ∧ a ≠ 0

-- The main theorem statement
theorem option_C_is_quadratic : is_quadratic option_C :=
sorry

end option_C_is_quadratic_l292_29297


namespace sufficient_not_necessary_condition_of_sin_l292_29279

open Real

theorem sufficient_not_necessary_condition_of_sin (θ : ℝ) :
  (abs (θ - π / 12) < π / 12) → (sin θ < 1 / 2) :=
sorry

end sufficient_not_necessary_condition_of_sin_l292_29279


namespace distinct_exponentiations_are_four_l292_29234

def power (a b : ℕ) : ℕ := a^b

def expr1 := power 3 (power 3 (power 3 3))
def expr2 := power 3 (power (power 3 3) 3)
def expr3 := power (power (power 3 3) 3) 3
def expr4 := power (power 3 (power 3 3)) 3
def expr5 := power (power 3 3) (power 3 3)

theorem distinct_exponentiations_are_four : 
  (expr1 ≠ expr2 ∧ expr1 ≠ expr3 ∧ expr1 ≠ expr4 ∧ expr1 ≠ expr5 ∧
   expr2 ≠ expr3 ∧ expr2 ≠ expr4 ∧ expr2 ≠ expr5 ∧
   expr3 ≠ expr4 ∧ expr3 ≠ expr5 ∧
   expr4 ≠ expr5) :=
sorry

end distinct_exponentiations_are_four_l292_29234


namespace neither_coffee_nor_tea_l292_29253

theorem neither_coffee_nor_tea (total_businesspeople coffee_drinkers tea_drinkers both_drinkers : ℕ) 
    (h_total : total_businesspeople = 35)
    (h_coffee : coffee_drinkers = 18)
    (h_tea : tea_drinkers = 15)
    (h_both : both_drinkers = 6) :
    (total_businesspeople - (coffee_drinkers + tea_drinkers - both_drinkers)) = 8 := 
by
  sorry

end neither_coffee_nor_tea_l292_29253


namespace factor_theorem_l292_29283

-- Define the polynomial function f(x)
def f (k : ℚ) (x : ℚ) : ℚ := k * x^3 + 27 * x^2 - k * x + 55

-- State the theorem to find the value of k such that x+5 is a factor of f(x)
theorem factor_theorem (k : ℚ) : f k (-5) = 0 ↔ k = 73 / 12 :=
by sorry

end factor_theorem_l292_29283


namespace number_of_even_factors_l292_29211

theorem number_of_even_factors {n : ℕ} (h : n = 2^4 * 3^3 * 7) : 
  ∃ (count : ℕ), count = 32 ∧ ∀ k, (k ∣ n) → k % 2 = 0 → count = 32 :=
by
  sorry

end number_of_even_factors_l292_29211


namespace ratio_final_to_initial_l292_29243

def initial_amount (P : ℝ) := P
def interest_rate := 4 / 100
def time_period := 25

def simple_interest (P : ℝ) := P * interest_rate * time_period

def final_amount (P : ℝ) := P + simple_interest P

theorem ratio_final_to_initial (P : ℝ) (hP : P > 0) :
  final_amount P / initial_amount P = 2 := by
  sorry

end ratio_final_to_initial_l292_29243


namespace number_of_selected_in_interval_l292_29238

-- Definitions and conditions based on the problem statement
def total_employees : ℕ := 840
def sample_size : ℕ := 42
def systematic_sampling_interval : ℕ := total_employees / sample_size
def interval_start : ℕ := 481
def interval_end : ℕ := 720

-- Main theorem statement that we need to prove
theorem number_of_selected_in_interval :
  let selected_in_interval : ℕ := (interval_end - interval_start + 1) / systematic_sampling_interval
  selected_in_interval = 12 := by
  sorry

end number_of_selected_in_interval_l292_29238


namespace at_least_one_distinct_root_l292_29203

theorem at_least_one_distinct_root {a b : ℝ} (ha : a > 4) (hb : b > 4) :
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a * x₁ + b = 0 ∧ x₂^2 + a * x₂ + b = 0) ∨
    (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 + b * y₁ + a = 0 ∧ y₂^2 + b * y₂ + a = 0) :=
sorry

end at_least_one_distinct_root_l292_29203


namespace largest_real_solution_l292_29284

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l292_29284


namespace problem_condition_l292_29259

noncomputable def f : ℝ → ℝ := sorry

theorem problem_condition (h: ∀ x : ℝ, f x > (deriv f) x) : 3 * f (Real.log 2) > 2 * f (Real.log 3) :=
sorry

end problem_condition_l292_29259


namespace inequality_proof_l292_29266

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) : 
  a^4 + b^4 + c^4 ≥ a * b * c * (a + b + c) := 
by 
  sorry

end inequality_proof_l292_29266


namespace avg_age_of_coaches_l292_29201

theorem avg_age_of_coaches (n_girls n_boys n_coaches : ℕ)
  (avg_age_girls avg_age_boys avg_age_members : ℕ)
  (h_girls : n_girls = 30)
  (h_boys : n_boys = 15)
  (h_coaches : n_coaches = 5)
  (h_avg_age_girls : avg_age_girls = 18)
  (h_avg_age_boys : avg_age_boys = 19)
  (h_avg_age_members : avg_age_members = 20) :
  (n_girls * avg_age_girls + n_boys * avg_age_boys + n_coaches * 35) / (n_girls + n_boys + n_coaches) = avg_age_members :=
by sorry

end avg_age_of_coaches_l292_29201


namespace english_homework_correct_time_l292_29223

-- Define the given conditions as constants
def total_time : ℕ := 180 -- 3 hours in minutes
def math_homework_time : ℕ := 45
def science_homework_time : ℕ := 50
def history_homework_time : ℕ := 25
def special_project_time : ℕ := 30

-- Define the function to compute english homework time
def english_homework_time : ℕ :=
  total_time - (math_homework_time + science_homework_time + history_homework_time + special_project_time)

-- The theorem to show the English homework time is 30 minutes
theorem english_homework_correct_time :
  english_homework_time = 30 :=
  by
    sorry

end english_homework_correct_time_l292_29223


namespace refills_needed_l292_29265

theorem refills_needed 
  (cups_per_day : ℕ)
  (bottle_capacity_oz : ℕ)
  (oz_per_cup : ℕ)
  (total_oz : ℕ)
  (refills : ℕ)
  (h1 : cups_per_day = 12)
  (h2 : bottle_capacity_oz = 16)
  (h3 : oz_per_cup = 8)
  (h4 : total_oz = cups_per_day * oz_per_cup)
  (h5 : refills = total_oz / bottle_capacity_oz) :
  refills = 6 :=
by
  sorry

end refills_needed_l292_29265


namespace five_digit_number_with_integer_cube_root_l292_29267

theorem five_digit_number_with_integer_cube_root (n : ℕ) 
  (h1 : n ≥ 10000 ∧ n < 100000) 
  (h2 : n % 10 = 3) 
  (h3 : ∃ k : ℕ, k^3 = n) : 
  n = 19683 ∨ n = 50653 :=
sorry

end five_digit_number_with_integer_cube_root_l292_29267


namespace volume_after_increase_l292_29273

variable (l w h : ℕ)
variable (V S E : ℕ)

noncomputable def original_volume : ℕ := l * w * h
noncomputable def surface_sum : ℕ := (l * w) + (w * h) + (h * l)
noncomputable def edge_sum : ℕ := l + w + h

theorem volume_after_increase (h_volume : original_volume l w h = 5400)
  (h_surface : surface_sum l w h = 1176)
  (h_edge : edge_sum l w h = 60) : 
  (l + 1) * (w + 1) * (h + 1) = 6637 := sorry

end volume_after_increase_l292_29273


namespace flu_infection_equation_l292_29291

theorem flu_infection_equation
  (x : ℝ) :
  (1 + x)^2 = 25 :=
sorry

end flu_infection_equation_l292_29291


namespace f_relationship_l292_29295

noncomputable def f (x : ℝ) : ℝ := sorry -- definition of f needs to be filled in later

-- Conditions given in the problem
variable (h_diff : Differentiable ℝ f)
variable (h_gt : ∀ x: ℝ, deriv f x > f x)
variable (a : ℝ) (h_pos : a > 0)

theorem f_relationship (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_gt : ∀ x: ℝ, deriv f x > f x) (a : ℝ) (h_pos : a > 0) :
  f a > Real.exp a * f 0 :=
sorry

end f_relationship_l292_29295


namespace value_of_a_plus_b_l292_29274

variable {F : Type} [Field F]

theorem value_of_a_plus_b (a b : F) (h1 : ∀ x, x ≠ 0 → a + b / x = 2 ↔ x = -2)
                                      (h2 : ∀ x, x ≠ 0 → a + b / x = 6 ↔ x = -6) :
  a + b = 20 :=
sorry

end value_of_a_plus_b_l292_29274


namespace min_value_of_x2_add_y2_l292_29250

theorem min_value_of_x2_add_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) : x^2 + y^2 ≥ 1 :=
sorry

end min_value_of_x2_add_y2_l292_29250


namespace angle_F_measure_l292_29257

-- Given conditions
def D := 74
def sum_of_angles (x E D : ℝ) := x + E + D = 180
def E_formula (x : ℝ) := 2 * x - 10

-- Proof problem statement in Lean 4
theorem angle_F_measure :
  ∃ x : ℝ, x = (116 / 3) ∧
    sum_of_angles x (E_formula x) D :=
sorry

end angle_F_measure_l292_29257


namespace equivalence_of_statements_l292_29264

theorem equivalence_of_statements (S X Y : Prop) : 
  (S → (¬ X ∧ ¬ Y)) ↔ ((X ∨ Y) → ¬ S) :=
by sorry

end equivalence_of_statements_l292_29264


namespace mike_pumpkins_l292_29275

def pumpkins : ℕ :=
  let sandy_pumpkins := 51
  let total_pumpkins := 74
  total_pumpkins - sandy_pumpkins

theorem mike_pumpkins : pumpkins = 23 :=
by
  sorry

end mike_pumpkins_l292_29275


namespace prove_value_range_for_a_l292_29232

noncomputable def f (x a : ℝ) : ℝ :=
  (x^2 + a*x + 7 + a) / (x + 1)

noncomputable def g (x : ℝ) : ℝ := 
  - ((x + 1) + (8 / (x + 1))) + 6

theorem prove_value_range_for_a (a : ℝ) :
  (∀ x : ℕ, x > 0 → f x a ≥ 4) ↔ (a ≥ 1 / 3) :=
sorry

end prove_value_range_for_a_l292_29232


namespace arithmetic_seq_S13_l292_29246

noncomputable def arithmetic_sequence_sum (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_seq_S13 (a_1 d : ℕ) (h : a_1 + 6 * d = 10) :
  arithmetic_sequence_sum a_1 d 13 = 130 :=
by
  sorry

end arithmetic_seq_S13_l292_29246


namespace hyperbola_eccentricity_l292_29251

-- Define the context/conditions
noncomputable def hyperbola_vertex_to_asymptote_distance (a b e : ℝ) : Prop :=
  (2 = b / e)

noncomputable def hyperbola_focus_to_asymptote_distance (a b e : ℝ) : Prop :=
  (6 = b)

-- Define the main theorem to prove the eccentricity
theorem hyperbola_eccentricity (a b e : ℝ) (h1 : hyperbola_vertex_to_asymptote_distance a b e) (h2 : hyperbola_focus_to_asymptote_distance a b e) : 
  e = 3 := 
sorry 

end hyperbola_eccentricity_l292_29251


namespace students_wearing_other_colors_l292_29230

-- Definitions based on conditions
def total_students := 700
def percentage_blue := 45 / 100
def percentage_red := 23 / 100
def percentage_green := 15 / 100

-- The proof problem statement
theorem students_wearing_other_colors :
  (total_students - total_students * (percentage_blue + percentage_red + percentage_green)) = 119 :=
by
  sorry

end students_wearing_other_colors_l292_29230


namespace right_triangle_sides_l292_29287

-- Definitions based on the conditions
def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2
def perimeter (a b c : ℕ) : ℕ := a + b + c
def inscribed_circle_radius (a b c : ℕ) : ℕ := (a + b - c) / 2

-- The theorem statement
theorem right_triangle_sides (a b c : ℕ) 
  (h_perimeter : perimeter a b c = 40)
  (h_radius : inscribed_circle_radius a b c = 3)
  (h_right : is_right_triangle a b c) :
  (a = 8 ∧ b = 15 ∧ c = 17) ∨ (a = 15 ∧ b = 8 ∧ c = 17) :=
by sorry

end right_triangle_sides_l292_29287


namespace min_value_xyz_l292_29205

theorem min_value_xyz (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_prod : x * y * z = 8) : 
  x + 3 * y + 6 * z ≥ 18 :=
sorry

end min_value_xyz_l292_29205


namespace carols_total_peanuts_l292_29296

-- Define the initial number of peanuts Carol has
def initial_peanuts : ℕ := 2

-- Define the number of peanuts given by Carol's father
def peanuts_given : ℕ := 5

-- Define the total number of peanuts Carol has
def total_peanuts : ℕ := initial_peanuts + peanuts_given

-- The statement we need to prove
theorem carols_total_peanuts : total_peanuts = 7 := by
  sorry

end carols_total_peanuts_l292_29296


namespace seating_arrangements_l292_29299

def count_arrangements (n k : ℕ) : ℕ :=
  (n.factorial) / (n - k).factorial

theorem seating_arrangements : count_arrangements 6 5 * 3 = 360 :=
  sorry

end seating_arrangements_l292_29299


namespace always_positive_sum_l292_29215

def f : ℝ → ℝ := sorry  -- assuming f(x) is provided elsewhere

theorem always_positive_sum (f : ℝ → ℝ)
    (h1 : ∀ x, f x = -f (2 - x))
    (h2 : ∀ x, x < 1 → f (x) < f (x + 1))
    (x1 x2 : ℝ)
    (h3 : x1 + x2 > 2)
    (h4 : (x1 - 1) * (x2 - 1) < 0) :
  f x1 + f x2 > 0 :=
by {
  sorry
}

end always_positive_sum_l292_29215


namespace first_term_geometric_sequence_l292_29262

theorem first_term_geometric_sequence (a5 a6 : ℚ) (h1 : a5 = 48) (h2 : a6 = 64) : 
  ∃ a : ℚ, a = 243 / 16 :=
by
  sorry

end first_term_geometric_sequence_l292_29262


namespace part1_part2a_part2b_part2c_l292_29261

def f (x a : ℝ) := |2 * x - 1| + |x - a|

theorem part1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : f x 3 ≤ 4 := sorry

theorem part2a (a x : ℝ) (h0 : a < 1 / 2) (h1 : a ≤ x ∧ x ≤ 1 / 2) : f x a = |x - 1 + a| := sorry

theorem part2b (a x : ℝ) (h0 : a = 1 / 2) (h1 : x = 1 / 2) : f x a = |x - 1 + a| := sorry

theorem part2c (a x : ℝ) (h0 : a > 1 / 2) (h1 : 1 / 2 ≤ x ∧ x ≤ a) : f x a = |x - 1 + a| := sorry

end part1_part2a_part2b_part2c_l292_29261


namespace Reeya_fifth_subject_score_l292_29222

theorem Reeya_fifth_subject_score 
  (a1 a2 a3 a4 : ℕ) (avg : ℕ) (subjects : ℕ) (a1_eq : a1 = 55) (a2_eq : a2 = 67) (a3_eq : a3 = 76) 
  (a4_eq : a4 = 82) (avg_eq : avg = 73) (subjects_eq : subjects = 5) :
  ∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5) / subjects = avg ∧ a5 = 85 :=
by
  sorry

end Reeya_fifth_subject_score_l292_29222


namespace smallest_x_l292_29277

theorem smallest_x (x : ℕ) : 
  (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 9 = 8) ↔ x = 314 := 
by
  sorry

end smallest_x_l292_29277


namespace additional_amount_deductibles_next_year_l292_29254

theorem additional_amount_deductibles_next_year :
  let avg_deductible : ℝ := 3000
  let inflation_rate : ℝ := 0.03
  let plan_a_rate : ℝ := 2 / 3
  let plan_b_rate : ℝ := 1 / 2
  let plan_c_rate : ℝ := 3 / 5
  let plan_a_percent : ℝ := 0.40
  let plan_b_percent : ℝ := 0.30
  let plan_c_percent : ℝ := 0.30
  let additional_a : ℝ := avg_deductible * plan_a_rate
  let additional_b : ℝ := avg_deductible * plan_b_rate
  let additional_c : ℝ := avg_deductible * plan_c_rate
  let weighted_additional : ℝ := (additional_a * plan_a_percent) + (additional_b * plan_b_percent) + (additional_c * plan_c_percent)
  let inflation_increase : ℝ := weighted_additional * inflation_rate
  let total_additional_amount : ℝ := weighted_additional + inflation_increase
  total_additional_amount = 1843.70 :=
sorry

end additional_amount_deductibles_next_year_l292_29254


namespace candy_mixture_problem_l292_29207

theorem candy_mixture_problem:
  ∃ x y : ℝ, x + y = 5 ∧ 3.20 * x + 1.70 * y = 10 ∧ x = 1 :=
by
  sorry

end candy_mixture_problem_l292_29207


namespace four_consecutive_integers_product_2520_l292_29225

theorem four_consecutive_integers_product_2520 {a b c d : ℕ}
  (h1 : a + 1 = b) 
  (h2 : b + 1 = c) 
  (h3 : c + 1 = d) 
  (h4 : a * b * c * d = 2520) : 
  a = 6 := 
sorry

end four_consecutive_integers_product_2520_l292_29225


namespace negate_proposition_l292_29290

theorem negate_proposition :
  (¬ ∃ (x₀ : ℝ), x₀^2 + 2 * x₀ + 3 ≤ 0) ↔ (∀ (x : ℝ), x^2 + 2 * x + 3 > 0) :=
by
  sorry

end negate_proposition_l292_29290


namespace expected_value_a_squared_is_correct_l292_29256

variables (n : ℕ)
noncomputable def expected_value_a_squared := ((2 * n) + (n^2)) / 3

theorem expected_value_a_squared_is_correct : 
  expected_value_a_squared n = ((2 * n) + (n^2)) / 3 := 
by 
  sorry

end expected_value_a_squared_is_correct_l292_29256


namespace max_vehicles_div_10_l292_29226

-- Each vehicle is 5 meters long
def vehicle_length : ℕ := 5

-- The speed rule condition
def speed_rule (m : ℕ) : ℕ := 20 * m

-- Maximum number of vehicles in one hour
def max_vehicles_per_hour (m : ℕ) : ℕ := 4000 * m / (m + 1)

-- N is the maximum whole number of vehicles
def N : ℕ := 4000

-- The target statement to prove: quotient when N is divided by 10
theorem max_vehicles_div_10 : N / 10 = 400 :=
by
  -- Definitions and given conditions go here
  sorry

end max_vehicles_div_10_l292_29226


namespace correct_number_of_outfits_l292_29294

def num_shirts : ℕ := 7
def num_pants : ℕ := 7
def num_hats : ℕ := 7
def num_colors : ℕ := 7

def total_outfits : ℕ := num_shirts * num_pants * num_hats
def invalid_outfits : ℕ := num_colors
def valid_outfits : ℕ := total_outfits - invalid_outfits

theorem correct_number_of_outfits : valid_outfits = 336 :=
by {
  -- sorry can be removed when providing the proof.
  sorry
}

end correct_number_of_outfits_l292_29294


namespace simplify_expression_l292_29272

variable (x : ℝ)

theorem simplify_expression :
  (25 * x^3) * (8 * x^2) * (1 / (4 * x) ^ 3) = (25 / 8) * x^2 :=
by
  sorry

end simplify_expression_l292_29272


namespace correct_observation_value_l292_29229

theorem correct_observation_value (mean : ℕ) (n : ℕ) (incorrect_obs : ℕ) (corrected_mean : ℚ) (original_sum : ℚ) (remaining_sum : ℚ) (corrected_sum : ℚ) :
  mean = 30 →
  n = 50 →
  incorrect_obs = 23 →
  corrected_mean = 30.5 →
  original_sum = (n * mean) →
  remaining_sum = (original_sum - incorrect_obs) →
  corrected_sum = (n * corrected_mean) →
  ∃ x : ℕ, remaining_sum + x = corrected_sum → x = 48 :=
by
  intros h_mean h_n h_incorrect_obs h_corrected_mean h_original_sum h_remaining_sum h_corrected_sum
  have original_mean := h_mean
  have observations := h_n
  have incorrect_observation := h_incorrect_obs
  have new_mean := h_corrected_mean
  have original_sum_calc := h_original_sum
  have remaining_sum_calc := h_remaining_sum
  have corrected_sum_calc := h_corrected_sum
  use 48
  sorry

end correct_observation_value_l292_29229


namespace distinct_values_of_expr_l292_29255

theorem distinct_values_of_expr : 
  let a := 3^(3^(3^3));
  let b := 3^((3^3)^3);
  let c := ((3^3)^3)^3;
  let d := (3^(3^3))^3;
  let e := (3^3)^(3^3);
  (a ≠ b) ∧ (c ≠ b) ∧ (d ≠ b) ∧ (d ≠ a) ∧ (e ≠ a) ∧ (e ≠ b) ∧ (e ≠ d) := sorry

end distinct_values_of_expr_l292_29255


namespace tree_break_height_l292_29278

-- Define the problem conditions and prove the required height h
theorem tree_break_height (height_tree : ℝ) (distance_shore : ℝ) (height_break : ℝ) : 
  height_tree = 20 → distance_shore = 6 → 
  (distance_shore ^ 2 + height_break ^ 2 = (height_tree - height_break) ^ 2) →
  height_break = 9.1 :=
by
  intros h_tree_eq h_shore_eq hyp_eq
  have h_tree_20 := h_tree_eq
  have h_shore_6 := h_shore_eq
  have hyp := hyp_eq
  sorry -- Proof of the theorem is omitted

end tree_break_height_l292_29278


namespace beth_final_students_l292_29241

-- Define the initial conditions
def initial_students : ℕ := 150
def students_joined : ℕ := 30
def students_left : ℕ := 15

-- Define the number of students after the first additional year
def after_first_year : ℕ := initial_students + students_joined

-- Define the final number of students after students leaving
def final_students : ℕ := after_first_year - students_left

-- Theorem to prove the number of students in the final year
theorem beth_final_students : 
  final_students = 165 :=
by
  sorry

end beth_final_students_l292_29241


namespace alice_chicken_weight_l292_29245

theorem alice_chicken_weight (total_cost_needed : ℝ)
  (amount_to_spend_more : ℝ)
  (cost_lettuce : ℝ)
  (cost_tomatoes : ℝ)
  (sweet_potato_quantity : ℝ)
  (cost_per_sweet_potato : ℝ)
  (broccoli_quantity : ℝ)
  (cost_per_broccoli : ℝ)
  (brussel_sprouts_weight : ℝ)
  (cost_per_brussel_sprouts : ℝ)
  (cost_per_pound_chicken : ℝ)
  (total_cost_excluding_chicken : ℝ) :
  total_cost_needed = 35 ∧
  amount_to_spend_more = 11 ∧
  cost_lettuce = 3 ∧
  cost_tomatoes = 2.5 ∧
  sweet_potato_quantity = 4 ∧
  cost_per_sweet_potato = 0.75 ∧
  broccoli_quantity = 2 ∧
  cost_per_broccoli = 2 ∧
  brussel_sprouts_weight = 1 ∧
  cost_per_brussel_sprouts = 2.5 ∧
  total_cost_excluding_chicken = (cost_lettuce + cost_tomatoes + sweet_potato_quantity * cost_per_sweet_potato + broccoli_quantity * cost_per_broccoli + brussel_sprouts_weight * cost_per_brussel_sprouts) →
  (total_cost_needed - amount_to_spend_more - total_cost_excluding_chicken) / cost_per_pound_chicken = 1.5 :=
by
  intros
  sorry

end alice_chicken_weight_l292_29245


namespace other_root_of_quadratic_l292_29258

variable (p : ℝ)

theorem other_root_of_quadratic (h1: 3 * (-2) * r_2 = -6) : r_2 = 1 :=
by
  sorry

end other_root_of_quadratic_l292_29258


namespace difference_two_digit_interchanged_l292_29248

theorem difference_two_digit_interchanged
  (x y : ℕ)
  (h1 : y = 2 * x)
  (h2 : (10 * x + y) - (x + y) = 8) :
  (10 * y + x) - (10 * x + y) = 9 := by
sorry

end difference_two_digit_interchanged_l292_29248


namespace neg_or_implication_l292_29210

theorem neg_or_implication {p q : Prop} : ¬(p ∨ q) → (¬p ∧ ¬q) :=
by
  intros h
  sorry

end neg_or_implication_l292_29210


namespace pyramid_volume_l292_29235

-- Definitions based on the given conditions
def AB : ℝ := 15
def AD : ℝ := 8
def Area_Δ_ABE : ℝ := 120
def Area_Δ_CDE : ℝ := 64
def h : ℝ := 16
def Base_Area : ℝ := AB * AD

-- Statement to prove the volume of the pyramid is 640
theorem pyramid_volume : (1 / 3) * Base_Area * h = 640 :=
sorry

end pyramid_volume_l292_29235


namespace cost_price_l292_29292

variables (SP DS CP : ℝ)
variables (discount_rate profit_rate : ℝ)
variables (H1 : SP = 24000)
variables (H2 : discount_rate = 0.10)
variables (H3 : profit_rate = 0.08)
variables (H4 : DS = SP - (discount_rate * SP))
variables (H5 : DS = CP + (profit_rate * CP))

theorem cost_price (H1 : SP = 24000) (H2 : discount_rate = 0.10) 
  (H3 : profit_rate = 0.08) (H4 : DS = SP - (discount_rate * SP)) 
  (H5 : DS = CP + (profit_rate * CP)) : 
  CP = 20000 := 
sorry

end cost_price_l292_29292


namespace Jules_height_l292_29276

theorem Jules_height (Ben_initial_height Jules_initial_height Ben_current_height Jules_current_height : ℝ) 
  (h_initial : Ben_initial_height = Jules_initial_height)
  (h_Ben_growth : Ben_current_height = 1.25 * Ben_initial_height)
  (h_Jules_growth : Jules_current_height = Jules_initial_height + (Ben_current_height - Ben_initial_height) / 3)
  (h_Ben_current : Ben_current_height = 75) 
  : Jules_current_height = 65 := 
by
  -- Use the conditions to prove that Jules is now 65 inches tall
  sorry

end Jules_height_l292_29276


namespace isosceles_triangle_l292_29293

theorem isosceles_triangle
  (a b c : ℝ)
  (α β γ : ℝ)
  (h1 : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β)) :
  α = β ∨ α = γ ∨ β = γ :=
sorry

end isosceles_triangle_l292_29293


namespace solve_for_y_l292_29202

theorem solve_for_y (y : ℝ) (h_sum : (1 + 99) * 99 / 2 = 4950)
  (h_avg : (4950 + y) / 100 = 50 * y) : y = 4950 / 4999 :=
by
  sorry

end solve_for_y_l292_29202


namespace meatballs_fraction_each_son_eats_l292_29219

theorem meatballs_fraction_each_son_eats
  (f1 f2 f3 : ℝ)
  (h1 : ∃ f1 f2 f3, f1 + f2 + f3 = 2)
  (meatballs_initial : ∀ n, n = 3) :
  f1 = 2/3 ∧ f2 = 2/3 ∧ f3 = 2/3 := by
  sorry

end meatballs_fraction_each_son_eats_l292_29219


namespace initial_money_l292_29227

-- Define the conditions
def spent_toy_truck : ℕ := 3
def spent_pencil_case : ℕ := 2
def money_left : ℕ := 5

-- Define the total money spent
def total_spent := spent_toy_truck + spent_pencil_case

-- Theorem statement
theorem initial_money (I : ℕ) (h : total_spent + money_left = I) : I = 10 :=
sorry

end initial_money_l292_29227


namespace john_text_messages_l292_29247

/-- John decides to get a new phone number and it ends up being a recycled number. 
    He used to get some text messages a day. 
    Now he is getting 55 text messages a day, 
    and he is getting 245 text messages per week that are not intended for him. 
    How many text messages a day did he used to get?
-/
theorem john_text_messages (m : ℕ) (h1 : 55 = m + 35) (h2 : 245 = 7 * 35) : m = 20 := 
by 
  sorry

end john_text_messages_l292_29247


namespace derivative_f_at_2_l292_29281

noncomputable def f (x : ℝ) : ℝ := (x + 1) * (x - 1)

theorem derivative_f_at_2 : (deriv f 2) = 4 := by
  sorry

end derivative_f_at_2_l292_29281


namespace binom_11_1_l292_29200

theorem binom_11_1 : Nat.choose 11 1 = 11 :=
by
  sorry

end binom_11_1_l292_29200


namespace exists_radius_for_marked_points_l292_29252

theorem exists_radius_for_marked_points :
  ∃ R : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ < 2 * π) →
    (∃ n : ℕ, (θ ≤ (n * 2 * π * R) % (2 * π * R) + 1 / R ∧ (n * 2 * π * R) % (2 * π * R) < θ + 1))) :=
sorry

end exists_radius_for_marked_points_l292_29252


namespace find_a_plus_k_l292_29271

variable (a k : ℝ)

noncomputable def f (x : ℝ) : ℝ := (a - 1) * x^k

theorem find_a_plus_k
  (h1 : f a k (Real.sqrt 2) = 2)
  (h2 : (Real.sqrt 2)^2 = 2) : a + k = 4 := 
sorry

end find_a_plus_k_l292_29271


namespace sixth_edge_length_l292_29263

theorem sixth_edge_length (a b c d o : Type) (distance : a -> a -> ℝ) (circumradius : ℝ) 
  (edge_length : ℝ) (h : ∀ (x y : a), x ≠ y → distance x y = edge_length ∨ distance x y = circumradius)
  (eq_edge_length : edge_length = 3) (eq_circumradius : circumradius = 2) : 
  ∃ ad : ℝ, ad = 6 * Real.sqrt (3 / 7) := 
by
  sorry

end sixth_edge_length_l292_29263


namespace product_of_intersection_coords_l292_29298

open Real

-- Define the two circles
def circle1 (x y: ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 21 = 0
def circle2 (x y: ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 52 = 0

-- Prove that the product of the coordinates of intersection points equals 189
theorem product_of_intersection_coords :
  (∃ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1 ∧ circle1 x2 y2 ∧ circle2 x2 y2 ∧ x1 * y1 * x2 * y2 = 189) :=
by
  sorry

end product_of_intersection_coords_l292_29298


namespace multiply_expression_l292_29239

variable {x : ℝ}

theorem multiply_expression :
  (x^4 + 10*x^2 + 25) * (x^2 - 25) = x^4 + 10*x^2 :=
by
  sorry

end multiply_expression_l292_29239


namespace trajectory_moving_circle_l292_29260

theorem trajectory_moving_circle : 
  (∃ P : ℝ × ℝ, (∃ r : ℝ, (P.1 + 1)^2 = r^2 ∧ (P.1 - 2)^2 + P.2^2 = (r + 1)^2) ∧
  P.2^2 = 8 * P.1) :=
sorry

end trajectory_moving_circle_l292_29260


namespace simple_interest_rate_l292_29244

theorem simple_interest_rate (P A T : ℕ) (P_val : P = 750) (A_val : A = 900) (T_val : T = 8) : 
  ∃ (R : ℚ), R = 2.5 :=
by {
  sorry
}

end simple_interest_rate_l292_29244


namespace prob_B_draws_given_A_draws_black_fairness_l292_29218

noncomputable def event_A1 : Prop := true  -- A draws the red ball
noncomputable def event_A2 : Prop := true  -- B draws the red ball
noncomputable def event_A3 : Prop := true  -- C draws the red ball

noncomputable def prob_A1 : ℝ := 1 / 3
noncomputable def prob_not_A1 : ℝ := 2 / 3
noncomputable def prob_A2_given_not_A1 : ℝ := 1 / 2

theorem prob_B_draws_given_A_draws_black : (prob_not_A1 * prob_A2_given_not_A1) / prob_not_A1 = 1 / 2 := by
  sorry

theorem fairness :
  let prob_A1 := 1 / 3
  let prob_A2 := prob_not_A1 * prob_A2_given_not_A1
  let prob_A3 := prob_not_A1 * prob_A2_given_not_A1 * 1
  prob_A1 = prob_A2 ∧ prob_A2 = prob_A3 := by
  sorry

end prob_B_draws_given_A_draws_black_fairness_l292_29218
