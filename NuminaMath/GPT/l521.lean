import Mathlib

namespace snail_returns_to_starting_point_l521_52198

-- Define the variables and conditions
variables (a1 a2 b1 b2 : ℕ)

-- Prove that snail can return to starting point after whole number of hours
theorem snail_returns_to_starting_point (h1 : a1 = a2) (h2 : b1 = b2) : (a1 + b1 : ℕ) = (a1 + b1 : ℕ) :=
by sorry

end snail_returns_to_starting_point_l521_52198


namespace vector_addition_l521_52178

def v1 : ℝ × ℝ := (3, -6)
def v2 : ℝ × ℝ := (2, -9)
def v3 : ℝ × ℝ := (-1, 3)
def c1 : ℝ := 4
def c2 : ℝ := 5
def result : ℝ × ℝ := (23, -72)

theorem vector_addition :
  c1 • v1 + c2 • v2 - v3 = result :=
by
  sorry

end vector_addition_l521_52178


namespace fish_population_estimate_l521_52170

theorem fish_population_estimate
  (N : ℕ) 
  (tagged_initial : ℕ)
  (caught_again : ℕ)
  (tagged_again : ℕ)
  (h1 : tagged_initial = 60)
  (h2 : caught_again = 60)
  (h3 : tagged_again = 2)
  (h4 : (tagged_initial : ℚ) / N = (tagged_again : ℚ) / caught_again) :
  N = 1800 :=
by
  sorry

end fish_population_estimate_l521_52170


namespace susan_hourly_rate_l521_52154

-- Definitions based on conditions
def vacation_workdays : ℕ := 10 -- Susan is taking a two-week vacation equivalent to 10 workdays

def weekly_workdays : ℕ := 5 -- Susan works 5 days a week

def paid_vacation_days : ℕ := 6 -- Susan has 6 days of paid vacation

def hours_per_day : ℕ := 8 -- Susan works 8 hours a day

def missed_pay_total : ℕ := 480 -- Susan will miss $480 pay on her unpaid vacation days

-- Calculations
def unpaid_vacation_days : ℕ := vacation_workdays - paid_vacation_days

def daily_lost_pay : ℕ := missed_pay_total / unpaid_vacation_days

def hourly_rate : ℕ := daily_lost_pay / hours_per_day

theorem susan_hourly_rate :
  hourly_rate = 15 := by sorry

end susan_hourly_rate_l521_52154


namespace problem_part1_problem_part2_l521_52143

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l521_52143


namespace consecutive_int_sqrt_l521_52184

theorem consecutive_int_sqrt (m n : ℤ) (h1 : m < n) (h2 : m < Real.sqrt 13) (h3 : Real.sqrt 13 < n) (h4 : n = m + 1) : m * n = 12 :=
sorry

end consecutive_int_sqrt_l521_52184


namespace watermelon_yield_increase_l521_52188

noncomputable def yield_increase (initial_yield final_yield annual_increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_yield * (1 + annual_increase_rate) ^ years

theorem watermelon_yield_increase :
  ∀ (x : ℝ),
    (yield_increase 20 28.8 x 2 = 28.8) →
    (yield_increase 28.8 40 x 2 > 40) :=
by
  intros x hx
  have incEq : 20 * (1 + x) ^ 2 = 28.8 := hx
  sorry

end watermelon_yield_increase_l521_52188


namespace n_congruence_mod_9_l521_52145

def n : ℕ := 2 + 333 + 5555 + 77777 + 999999 + 2222222 + 44444444 + 666666666

theorem n_congruence_mod_9 : n % 9 = 4 :=
by
  sorry

end n_congruence_mod_9_l521_52145


namespace hannah_bananas_l521_52159

theorem hannah_bananas (B : ℕ) (h1 : B / 4 = 15 / 3) : B = 20 :=
by
  sorry

end hannah_bananas_l521_52159


namespace number_of_students_l521_52138

variables (T S n : ℕ)

-- 1. The teacher's age is 24 years more than the average age of the students.
def condition1 : Prop := T = S / n + 24

-- 2. The teacher's age is 20 years more than the average age of everyone present.
def condition2 : Prop := T = (T + S) / (n + 1) + 20

-- Proving that the number of students in the classroom is 5 given the conditions.
theorem number_of_students (h1 : condition1 T S n) (h2 : condition2 T S n) : n = 5 :=
by sorry

end number_of_students_l521_52138


namespace sufficient_but_not_necessary_l521_52137

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (a > b + 1) → (a > b) ∧ (¬(a > b) → ¬(a > b + 1)) :=
by
  sorry

end sufficient_but_not_necessary_l521_52137


namespace fractional_eq_solutions_1_fractional_eq_reciprocal_sum_fractional_eq_solution_diff_square_l521_52131

def fractional_eq_solution_1 (x : ℝ) : Prop :=
  x + 5 / x = -6

theorem fractional_eq_solutions_1 : fractional_eq_solution_1 (-1) ∧ fractional_eq_solution_1 (-5) := sorry

def fractional_eq_solution_2 (x : ℝ) : Prop :=
  x - 3 / x = 4

theorem fractional_eq_reciprocal_sum
  (m n : ℝ) (h₀ : fractional_eq_solution_2 m) (h₁ : fractional_eq_solution_2 n) :
  m * n = -3 → m + n = 4 → (1 / m + 1 / n = -4 / 3) := sorry

def fractional_eq_solution_3 (x : ℝ) (a : ℝ) : Prop :=
  x + (a^2 + 2 * a) / (x + 1) = 2 * a + 1

theorem fractional_eq_solution_diff_square (a : ℝ) (h₀ : a ≠ 0)
  (x1 x2 : ℝ) (hx1 : fractional_eq_solution_3 x1 a) (hx2 : fractional_eq_solution_3 x2 a) :
  x1 + 1 = a → x2 + 1 = a + 2 → (x1 - x2) ^ 2 = 4 := sorry

end fractional_eq_solutions_1_fractional_eq_reciprocal_sum_fractional_eq_solution_diff_square_l521_52131


namespace geom_sequence_product_l521_52179

theorem geom_sequence_product (q a1 : ℝ) (h1 : a1 * (a1 * q) * (a1 * q^2) = 3) (h2 : (a1 * q^9) * (a1 * q^10) * (a1 * q^11) = 24) :
  (a1 * q^12) * (a1 * q^13) * (a1 * q^14) = 48 :=
by
  sorry

end geom_sequence_product_l521_52179


namespace motel_total_rent_l521_52104

theorem motel_total_rent (R40 R60 : ℕ) (total_rent : ℕ) 
  (h1 : total_rent = 40 * R40 + 60 * R60) 
  (h2 : 40 * (R40 + 10) + 60 * (R60 - 10) = total_rent - total_rent / 10) 
  (h3 : total_rent / 10 = 200) : 
  total_rent = 2000 := 
sorry

end motel_total_rent_l521_52104


namespace angles_proof_l521_52180

-- Definitions (directly from the conditions)
variable {θ₁ θ₂ θ₃ θ₄ : ℝ}

def complementary (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 90
def supplementary (θ₃ θ₄ : ℝ) : Prop := θ₃ + θ₄ = 180

-- Theorem statement
theorem angles_proof (h1 : complementary θ₁ θ₂) (h2 : supplementary θ₃ θ₄) (h3 : θ₁ = θ₃) :
  θ₂ + 90 = θ₄ :=
by
  sorry

end angles_proof_l521_52180


namespace greatest_integer_l521_52168

theorem greatest_integer (y : ℤ) : (8 / 11 : ℝ) > (y / 17 : ℝ) → y ≤ 12 :=
by sorry

end greatest_integer_l521_52168


namespace Rihanna_money_left_l521_52112

theorem Rihanna_money_left (initial_money mango_count juice_count mango_price juice_price : ℕ)
  (h_initial : initial_money = 50)
  (h_mango_count : mango_count = 6)
  (h_juice_count : juice_count = 6)
  (h_mango_price : mango_price = 3)
  (h_juice_price : juice_price = 3) :
  initial_money - (mango_count * mango_price + juice_count * juice_price) = 14 :=
sorry

end Rihanna_money_left_l521_52112


namespace winning_candidate_votes_l521_52169

def total_votes : ℕ := 100000
def winning_percentage : ℚ := 42 / 100
def expected_votes : ℚ := 42000

theorem winning_candidate_votes : winning_percentage * total_votes = expected_votes := by
  sorry

end winning_candidate_votes_l521_52169


namespace money_sister_gave_l521_52196

theorem money_sister_gave (months_saved : ℕ) (savings_per_month : ℕ) (total_paid : ℕ) 
  (h1 : months_saved = 3) 
  (h2 : savings_per_month = 70) 
  (h3 : total_paid = 260) : 
  (total_paid - (months_saved * savings_per_month) = 50) :=
by {
  sorry
}

end money_sister_gave_l521_52196


namespace ellipse_abs_sum_max_min_l521_52139

theorem ellipse_abs_sum_max_min (x y : ℝ) (h : x^2 / 4 + y^2 / 9 = 1) :
  2 ≤ |x| + |y| ∧ |x| + |y| ≤ 3 :=
sorry

end ellipse_abs_sum_max_min_l521_52139


namespace problem_statement_l521_52118

theorem problem_statement : 
  (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) + (Real.sqrt 2 - Real.sqrt 3) ^ 2 = 4 - 2 * Real.sqrt 6 := 
by 
  sorry

end problem_statement_l521_52118


namespace probability_two_white_balls_l521_52122

-- Definitions based on the conditions provided
def total_balls := 17        -- 8 white + 9 black
def white_balls := 8
def drawn_without_replacement := true

-- Proposition: Probability of drawing two white balls successively
theorem probability_two_white_balls:
  drawn_without_replacement → 
  (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1)) = 7 / 34 :=
by
  intros
  sorry

end probability_two_white_balls_l521_52122


namespace mary_groceries_fitting_l521_52111

theorem mary_groceries_fitting :
  (∀ bags wt_green wt_milk wt_carrots wt_apples wt_bread wt_rice,
    bags = 2 →
    wt_green = 4 →
    wt_milk = 6 →
    wt_carrots = 2 * wt_green →
    wt_apples = 3 →
    wt_bread = 1 →
    wt_rice = 5 →
    (wt_green + wt_milk + wt_carrots + wt_apples + wt_bread + wt_rice = 27) →
    (∀ b, b < 20 →
      (b = 6 + 5 ∨ b = 22 - 11) →
      (20 - b = 9))) :=
by
  intros bags wt_green wt_milk wt_carrots wt_apples wt_bread wt_rice h_bags h_green h_milk h_carrots h_apples h_bread h_rice h_total h_b
  sorry

end mary_groceries_fitting_l521_52111


namespace history_book_cost_l521_52152

def total_books : ℕ := 90
def cost_math_book : ℕ := 4
def total_price : ℕ := 397
def math_books_bought : ℕ := 53

theorem history_book_cost :
  ∃ (H : ℕ), H = (total_price - (math_books_bought * cost_math_book)) / (total_books - math_books_bought) ∧ H = 5 :=
by
  sorry

end history_book_cost_l521_52152


namespace boxes_containing_neither_l521_52199

-- Define the conditions
def total_boxes : ℕ := 15
def boxes_with_pencils : ℕ := 8
def boxes_with_pens : ℕ := 5
def boxes_with_markers : ℕ := 3
def boxes_with_pencils_and_pens : ℕ := 2
def boxes_with_pencils_and_markers : ℕ := 1
def boxes_with_pens_and_markers : ℕ := 1
def boxes_with_all_three : ℕ := 0

-- The proof problem
theorem boxes_containing_neither (h: total_boxes = 15) : 
  total_boxes - ((boxes_with_pencils - boxes_with_pencils_and_pens - boxes_with_pencils_and_markers) + 
  (boxes_with_pens - boxes_with_pencils_and_pens - boxes_with_pens_and_markers) + 
  (boxes_with_markers - boxes_with_pencils_and_markers - boxes_with_pens_and_markers) + 
  boxes_with_pencils_and_pens + boxes_with_pencils_and_markers + boxes_with_pens_and_markers) = 3 := 
by
  -- Specify that we want to use the equality of the number of boxes
  sorry

end boxes_containing_neither_l521_52199


namespace bus_total_people_l521_52185

def number_of_boys : ℕ := 50
def additional_girls (b : ℕ) : ℕ := (2 * b) / 5
def number_of_girls (b : ℕ) : ℕ := b + additional_girls b
def total_people (b g : ℕ) : ℕ := b + g + 3  -- adding 3 for the driver, assistant, and teacher

theorem bus_total_people : total_people number_of_boys (number_of_girls number_of_boys) = 123 :=
by
  sorry

end bus_total_people_l521_52185


namespace solve_arithmetic_sequence_l521_52163

theorem solve_arithmetic_sequence :
  ∃ x > 0, (x * x = (4 + 25) / 2) :=
by
  sorry

end solve_arithmetic_sequence_l521_52163


namespace nonnegative_poly_sum_of_squares_l521_52127

open Polynomial

theorem nonnegative_poly_sum_of_squares (P : Polynomial ℝ) 
    (hP : ∀ x : ℝ, 0 ≤ P.eval x) 
    : ∃ Q R : Polynomial ℝ, P = Q^2 + R^2 := 
by
  sorry

end nonnegative_poly_sum_of_squares_l521_52127


namespace proof_F_2_f_3_l521_52121

def f (a : ℕ) : ℕ := a ^ 2 - 1

def F (a : ℕ) (b : ℕ) : ℕ := 3 * b ^ 2 + 2 * a

theorem proof_F_2_f_3 : F 2 (f 3) = 196 := by
  have h1 : f 3 = 3 ^ 2 - 1 := rfl
  rw [h1]
  have h2 : 3 ^ 2 - 1 = 8 := by norm_num
  rw [h2]
  exact rfl

end proof_F_2_f_3_l521_52121


namespace probability_non_defective_pencils_l521_52133

theorem probability_non_defective_pencils :
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_combinations := Nat.choose total_pencils selected_pencils
  let non_defective_combinations := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations:ℚ) / (total_combinations:ℚ) = 5 / 14 := by
  sorry

end probability_non_defective_pencils_l521_52133


namespace deepak_and_wife_meet_time_l521_52181

noncomputable def deepak_speed_kmph : ℝ := 20
noncomputable def wife_speed_kmph : ℝ := 12
noncomputable def track_circumference_m : ℝ := 1000

noncomputable def speed_to_m_per_min (speed_kmph : ℝ) : ℝ :=
  (speed_kmph * 1000) / 60

noncomputable def deepak_speed_m_per_min : ℝ := speed_to_m_per_min deepak_speed_kmph
noncomputable def wife_speed_m_per_min : ℝ := speed_to_m_per_min wife_speed_kmph

noncomputable def combined_speed_m_per_min : ℝ :=
  deepak_speed_m_per_min + wife_speed_m_per_min

noncomputable def meeting_time_minutes : ℝ :=
  track_circumference_m / combined_speed_m_per_min

theorem deepak_and_wife_meet_time :
  abs (meeting_time_minutes - 1.875) < 0.01 :=
by
  sorry

end deepak_and_wife_meet_time_l521_52181


namespace domain_of_log_function_l521_52123

theorem domain_of_log_function :
  {x : ℝ | x^2 - x > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end domain_of_log_function_l521_52123


namespace num_solutions_l521_52147

-- Define the problem and the condition
def matrix_eq (x : ℝ) : Prop :=
  3 * x^2 - 4 * x = 7

-- Define the main theorem to prove the number of solutions
theorem num_solutions : ∃! x : ℝ, matrix_eq x :=
sorry

end num_solutions_l521_52147


namespace result_after_subtraction_l521_52172

theorem result_after_subtraction (x : ℕ) (h : x = 125) : 2 * x - 138 = 112 :=
by
  sorry

end result_after_subtraction_l521_52172


namespace max_elements_in_S_l521_52194

theorem max_elements_in_S : ∀ (S : Finset ℕ), 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → 
    (∃ c ∈ S, Nat.Coprime c a ∧ Nat.Coprime c b) ∧
    (∃ d ∈ S, ∃ x y : ℕ, x ∣ a ∧ x ∣ b ∧ x ∣ d ∧ y ∣ a ∧ y ∣ b ∧ y ∣ d)) →
  S.card ≤ 72 :=
by sorry

end max_elements_in_S_l521_52194


namespace f1_min_max_f2_min_max_l521_52125

-- Define the first function and assert its max and min values
def f1 (x : ℝ) : ℝ := x^3 + 2 * x

theorem f1_min_max : ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  (∃ x_min x_max, x_min = -1 ∧ x_max = 1 ∧ f1 x_min = -3 ∧ f1 x_max = 3) := by
  sorry

-- Define the second function and assert its max and min values
def f2 (x : ℝ) : ℝ := (x - 1) * (x - 2)^2

theorem f2_min_max : ∀ x ∈ Set.Icc (0 : ℝ) 3,
  (∃ x_min x_max, x_min = 0 ∧ x_max = 3 ∧ (f2 x_min = -4) ∧ f2 x_max = 2) := by
  sorry

end f1_min_max_f2_min_max_l521_52125


namespace find_x_l521_52174

-- Define the operation "※" as given
def star (a b : ℕ) : ℚ := (a + 2 * b) / 3

-- Given that 6 ※ x = 22 / 3, prove that x = 8
theorem find_x : ∃ x : ℕ, star 6 x = 22 / 3 ↔ x = 8 :=
by
  sorry -- Proof not required

end find_x_l521_52174


namespace maximum_n_for_dart_probability_l521_52148

theorem maximum_n_for_dart_probability (n : ℕ) (h : n ≥ 1) :
  (∃ r : ℝ, r = 1 ∧
  ∃ A_square A_circles : ℝ, A_square = n^2 ∧ A_circles = n * π * r^2 ∧
  (A_circles / A_square) ≥ 1 / 2) → n ≤ 6 := by
  sorry

end maximum_n_for_dart_probability_l521_52148


namespace value_of_b_l521_52135

theorem value_of_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 3) : b = 3 := 
by
  sorry

end value_of_b_l521_52135


namespace diagonal_lt_half_perimeter_l521_52166

theorem diagonal_lt_half_perimeter (AB BC CD DA AC : ℝ) (h1 : AB > 0) (h2 : BC > 0) (h3 : CD > 0) (h4 : DA > 0) 
  (h_triangle1 : AC < AB + BC) (h_triangle2 : AC < AD + DC) :
  AC < (AB + BC + CD + DA) / 2 :=
by {
  sorry
}

end diagonal_lt_half_perimeter_l521_52166


namespace cos_diff_l521_52136

theorem cos_diff (α : ℝ) (h1 : Real.cos α = (Real.sqrt 2) / 10) (h2 : α > -π ∧ α < 0) :
  Real.cos (α - π / 4) = -3 / 5 :=
sorry

end cos_diff_l521_52136


namespace sculpture_paint_area_correct_l521_52190

def sculpture_exposed_area (edge_length : ℝ) (num_cubes_layer1 : ℕ) (num_cubes_layer2 : ℕ) (num_cubes_layer3 : ℕ) : ℝ :=
  let area_top_layer1 := num_cubes_layer1 * edge_length ^ 2
  let area_side_layer1 := 8 * 3 * edge_length ^ 2
  let area_top_layer2 := num_cubes_layer2 * edge_length ^ 2
  let area_side_layer2 := 10 * edge_length ^ 2
  let area_top_layer3 := num_cubes_layer3 * edge_length ^ 2
  let area_side_layer3 := num_cubes_layer3 * 4 * edge_length ^ 2
  area_top_layer1 + area_side_layer1 + area_top_layer2 + area_side_layer2 + area_top_layer3 + area_side_layer3

theorem sculpture_paint_area_correct :
  sculpture_exposed_area 1 12 6 2 = 62 := by
  sorry

end sculpture_paint_area_correct_l521_52190


namespace probability_two_females_one_male_l521_52132

theorem probability_two_females_one_male
  (total_contestants : ℕ)
  (female_contestants : ℕ)
  (male_contestants : ℕ)
  (choose_count : ℕ)
  (total_combinations : ℕ)
  (female_combinations : ℕ)
  (male_combinations : ℕ)
  (favorable_outcomes : ℕ)
  (probability : ℚ)
  (h1 : total_contestants = 8)
  (h2 : female_contestants = 5)
  (h3 : male_contestants = 3)
  (h4 : choose_count = 3)
  (h5 : total_combinations = Nat.choose total_contestants choose_count)
  (h6 : female_combinations = Nat.choose female_contestants 2)
  (h7 : male_combinations = Nat.choose male_contestants 1)
  (h8 : favorable_outcomes = female_combinations * male_combinations)
  (h9 : probability = favorable_outcomes / total_combinations) :
  probability = 15 / 28 :=
by
  sorry

end probability_two_females_one_male_l521_52132


namespace tank_never_fills_l521_52149

structure Pipe :=
(rate1 : ℕ) (rate2 : ℕ)

def net_flow (pA pB pC pD : Pipe) (time1 time2 : ℕ) : ℤ :=
  let fillA := pA.rate1 * time1 + pA.rate2 * time2
  let fillB := pB.rate1 * time1 + pB.rate2 * time2
  let drainC := pC.rate1 * time1 + pC.rate2 * time2
  let drainD := pD.rate1 * (time1 + time2)
  (fillA + fillB) - (drainC + drainD)

theorem tank_never_fills (pA pB pC pD : Pipe) (time1 time2 : ℕ)
  (hA : pA = Pipe.mk 40 20) (hB : pB = Pipe.mk 20 40) 
  (hC : pC = Pipe.mk 20 40) (hD : pD = Pipe.mk 30 30) 
  (hTime : time1 = 30 ∧ time2 = 30): 
  net_flow pA pB pC pD time1 time2 = 0 := by
  sorry

end tank_never_fills_l521_52149


namespace no_real_solution_l521_52134

theorem no_real_solution (x y : ℝ) (hx : x^2 = 1 + 1 / y^2) (hy : y^2 = 1 + 1 / x^2) : false :=
by
  sorry

end no_real_solution_l521_52134


namespace xyz_value_l521_52110

theorem xyz_value (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 30) 
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : x * y * z = 7 :=
by
  sorry

end xyz_value_l521_52110


namespace tan_diff_identity_l521_52162

theorem tan_diff_identity (α : ℝ) (hα : 0 < α ∧ α < π) (h : Real.sin α = 4 / 5) :
  Real.tan (π / 4 - α) = -1 / 7 ∨ Real.tan (π / 4 - α) = -7 :=
sorry

end tan_diff_identity_l521_52162


namespace find_x_l521_52109

theorem find_x (x : ℕ) (hcf lcm : ℕ):
  (hcf = Nat.gcd x 18) → 
  (lcm = Nat.lcm x 18) → 
  (lcm - hcf = 120) → 
  x = 42 := 
by
  sorry

end find_x_l521_52109


namespace coefficient_of_x9_in_polynomial_is_240_l521_52157

-- Define the polynomial (1 + 3x - 2x^2)^5
noncomputable def polynomial : ℕ → ℝ := (fun x => (1 + 3*x - 2*x^2)^5)

-- Define the term we are interested in (x^9)
def term := 9

-- The coefficient we want to prove
def coefficient := 240

-- The goal is to prove that the coefficient of x^9 in the expansion of (1 + 3x - 2x^2)^5 is 240
theorem coefficient_of_x9_in_polynomial_is_240 : polynomial 9 = coefficient := sorry

end coefficient_of_x9_in_polynomial_is_240_l521_52157


namespace percentage_of_men_not_speaking_french_or_spanish_l521_52191

theorem percentage_of_men_not_speaking_french_or_spanish 
  (total_employees : ℕ) 
  (men_percent women_percent : ℝ)
  (men_french percent men_spanish_percent men_other_percent : ℝ)
  (women_french_percent women_spanish_percent women_other_percent : ℝ)
  (h1 : men_percent = 60)
  (h2 : women_percent = 40)
  (h3 : men_french_percent = 55)
  (h4 : men_spanish_percent = 35)
  (h5 : men_other_percent = 10)
  (h6 : women_french_percent = 45)
  (h7 : women_spanish_percent = 25)
  (h8 : women_other_percent = 30) :
  men_other_percent = 10 := 
by
  sorry

end percentage_of_men_not_speaking_french_or_spanish_l521_52191


namespace gcd_91_72_l521_52107

/-- Prove that the greatest common divisor of 91 and 72 is 1. -/
theorem gcd_91_72 : Nat.gcd 91 72 = 1 :=
by
  sorry

end gcd_91_72_l521_52107


namespace inequality_sufficient_condition_l521_52141

theorem inequality_sufficient_condition (x : ℝ) (h : 1 < x ∧ x < 2) : 
  (x+1)/(x-1) > 2 :=
by
  sorry

end inequality_sufficient_condition_l521_52141


namespace difference_in_sums_l521_52193

def sum_of_digits (n : ℕ) : ℕ := (toString n).foldl (λ acc c => acc + (c.toNat - '0'.toNat)) 0

def Petrov_numbers := List.range' 1 2014 |>.filter (λ n => n % 2 = 1)
def Vasechkin_numbers := List.range' 2 2012 |>.filter (λ n => n % 2 = 0)

def sum_of_digits_Petrov := (Petrov_numbers.map sum_of_digits).sum
def sum_of_digits_Vasechkin := (Vasechkin_numbers.map sum_of_digits).sum

theorem difference_in_sums : sum_of_digits_Petrov - sum_of_digits_Vasechkin = 1007 := by
  sorry

end difference_in_sums_l521_52193


namespace find_a_l521_52102

theorem find_a (a : ℝ) (x₁ x₂ : ℝ) :
  (2 * x₁ + 1 = 3) →
  (2 - (a - x₂) / 3 = 1) →
  (x₁ = x₂) →
  a = 4 :=
by
  intros h₁ h₂ h₃
  sorry

end find_a_l521_52102


namespace jog_to_coffee_shop_l521_52192

def constant_pace_jogging (time_to_park : ℕ) (dist_to_park : ℝ) (dist_to_coffee_shop : ℝ) : Prop :=
  time_to_park / dist_to_park * dist_to_coffee_shop = 6

theorem jog_to_coffee_shop
  (time_to_park : ℕ)
  (dist_to_park : ℝ)
  (dist_to_coffee_shop : ℝ)
  (h1 : time_to_park = 12)
  (h2 : dist_to_park = 1.5)
  (h3 : dist_to_coffee_shop = 0.75)
: constant_pace_jogging time_to_park dist_to_park dist_to_coffee_shop :=
by sorry

end jog_to_coffee_shop_l521_52192


namespace average_movers_per_hour_l521_52115

-- Define the main problem parameters
def total_people : ℕ := 3200
def days : ℕ := 4
def hours_per_day : ℕ := 24
def total_hours : ℕ := hours_per_day * days
def average_people_per_hour := total_people / total_hours

-- State the theorem to prove
theorem average_movers_per_hour :
  average_people_per_hour = 33 :=
by
  -- Proof is omitted
  sorry

end average_movers_per_hour_l521_52115


namespace weekly_profit_function_maximize_weekly_profit_weekly_sales_quantity_l521_52116

noncomputable def cost_price : ℝ := 10
noncomputable def y (x : ℝ) : ℝ := -10 * x + 400
noncomputable def w (x : ℝ) : ℝ := -10 * x ^ 2 + 500 * x - 4000

-- Proof Step 1: Show the functional relationship between w and x
theorem weekly_profit_function : ∀ x : ℝ, w x = -10 * x ^ 2 + 500 * x - 4000 := by
  intro x
  -- This is the function definition provided, proof omitted
  sorry

-- Proof Step 2: Find the selling price x that maximizes weekly profit
theorem maximize_weekly_profit : ∃ x : ℝ, x = 25 ∧ (∀ y : ℝ, y ≠ x → w y ≤ w x) := by
  use 25
  -- The details of solving the optimization are omitted
  sorry

-- Proof Step 3: Given weekly profit w = 2000 and constraints on y, find the weekly sales quantity
theorem weekly_sales_quantity (x : ℝ) (H : w x = 2000 ∧ y x ≥ 180) : y x = 200 := by
  have Hy : y x = -10 * x + 400 := by rfl
  have Hconstraint : y x ≥ 180 := H.2
  have Hprofit : w x = 2000 := H.1
  -- The details of solving for x and ensuring constraints are omitted
  sorry

end weekly_profit_function_maximize_weekly_profit_weekly_sales_quantity_l521_52116


namespace ac_work_time_l521_52153

theorem ac_work_time (W : ℝ) (a_work_rate : ℝ) (b_work_rate : ℝ) (bc_work_rate : ℝ) (t : ℝ) : 
  (a_work_rate = W / 4) ∧ 
  (b_work_rate = W / 12) ∧ 
  (bc_work_rate = W / 3) → 
  t = 2 := 
by 
  sorry

end ac_work_time_l521_52153


namespace least_possible_area_of_square_l521_52158

theorem least_possible_area_of_square (s : ℝ) (h₁ : 4.5 ≤ s) (h₂ : s < 5.5) : 
  s * s ≥ 20.25 :=
sorry

end least_possible_area_of_square_l521_52158


namespace foldable_polygons_count_l521_52140

def isValidFolding (base_positions : Finset Nat) (additional_position : Nat) : Prop :=
  ∃ (valid_positions : Finset Nat), valid_positions = {4, 5, 6, 7, 8, 9} ∧ additional_position ∈ valid_positions

theorem foldable_polygons_count : 
  ∃ (valid_additional_positions : Finset Nat), valid_additional_positions = {4, 5, 6, 7, 8, 9} ∧ valid_additional_positions.card = 6 := 
by
  sorry

end foldable_polygons_count_l521_52140


namespace frac_add_eq_seven_halves_l521_52120

theorem frac_add_eq_seven_halves {x y : ℝ} (h : x / y = 5 / 2) : (x + y) / y = 7 / 2 :=
by
  sorry

end frac_add_eq_seven_halves_l521_52120


namespace quadratic_roots_integer_sum_eq_198_l521_52164

theorem quadratic_roots_integer_sum_eq_198 (x p q x1 x2 : ℤ) 
  (h_eqn : x^2 + p * x + q = 0)
  (h_roots : (x - x1) * (x - x2) = 0)
  (h_pq_sum : p + q = 198) :
  (x1 = 2 ∧ x2 = 200) ∨ (x1 = 0 ∧ x2 = -198) :=
sorry

end quadratic_roots_integer_sum_eq_198_l521_52164


namespace first_player_wins_l521_52126

-- Define the initial conditions
def initial_pile_1 : ℕ := 100
def initial_pile_2 : ℕ := 200

-- Define the game rules
def valid_move (pile_1 pile_2 n : ℕ) : Prop :=
  (n > 0) ∧ ((n <= pile_1) ∨ (n <= pile_2))

-- The game state is represented as a pair of natural numbers
def GameState := ℕ × ℕ

-- Define what it means to win the game
def winning_move (s: GameState) : Prop :=
  (s.1 = 0 ∧ s.2 = 1) ∨ (s.1 = 1 ∧ s.2 = 0)

-- Define the main theorem
theorem first_player_wins : 
  ∀ s : GameState, (s = (initial_pile_1, initial_pile_2)) → (∃ move, valid_move s.1 s.2 move ∧ winning_move (s.1 - move, s.2 - move)) :=
sorry

end first_player_wins_l521_52126


namespace max_nested_fraction_value_l521_52119

-- Define the problem conditions
def numbers := (List.range 100).map (λ n => n + 1)

-- Define the nested fraction function
noncomputable def nested_fraction (l : List ℕ) : ℚ :=
  l.foldr (λ x acc => x / acc) 1

-- Prove that the maximum value of the nested fraction from 1 to 100 is 100! / 4
theorem max_nested_fraction_value :
  nested_fraction numbers = (Nat.factorial 100) / 4 :=
sorry

end max_nested_fraction_value_l521_52119


namespace probability_no_shaded_in_2_by_2004_l521_52101

noncomputable def probability_no_shaded_rectangle (total_rectangles shaded_rectangles : Nat) : ℚ :=
  1 - (shaded_rectangles : ℚ) / (total_rectangles : ℚ)

theorem probability_no_shaded_in_2_by_2004 :
  let rows := 2
  let cols := 2004
  let total_rectangles := (cols + 1) * cols / 2 * rows
  let shaded_rectangles := 501 * 2507 
  probability_no_shaded_rectangle total_rectangles shaded_rectangles = 1501 / 4008 :=
by
  sorry

end probability_no_shaded_in_2_by_2004_l521_52101


namespace number_of_blue_faces_l521_52161

theorem number_of_blue_faces (n : ℕ) (h : (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 :=
by
  sorry

end number_of_blue_faces_l521_52161


namespace find_perpendicular_line_l521_52189

-- Define the point P
structure Point where
  x : ℤ
  y : ℤ

-- Define the line
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

-- The given problem conditions
def P : Point := { x := -1, y := 3 }

def given_line : Line := { a := 1, b := -2, c := 3 }

def perpendicular_line (line : Line) (point : Point) : Line :=
  ⟨ -line.b, line.a, -(line.a * point.y - line.b * point.x) ⟩

-- Theorem statement to prove
theorem find_perpendicular_line :
  perpendicular_line given_line P = { a := 2, b := 1, c := -1 } :=
by
  sorry

end find_perpendicular_line_l521_52189


namespace z_gets_amount_per_unit_l521_52130

-- Define the known conditions
variables (x y z : ℝ)
variables (x_share : ℝ)
variables (y_share : ℝ)
variables (z_share : ℝ)
variables (total : ℝ)

-- Assume the conditions given in the problem
axiom h1 : y_share = 54
axiom h2 : total = 234
axiom h3 : (y / x) = 0.45
axiom h4 : total = x_share + y_share + z_share

-- Prove the target statement
theorem z_gets_amount_per_unit : ((z_share / x_share) = 0.50) :=
by
  sorry

end z_gets_amount_per_unit_l521_52130


namespace evaluate_expression_l521_52186

theorem evaluate_expression (a b c : ℝ) (h1 : a = 4) (h2 : b = -4) (h3 : c = 3) : (3 / (a + b + c) = 1) :=
by
  sorry

end evaluate_expression_l521_52186


namespace trigonometric_identity_l521_52183

theorem trigonometric_identity
  (α : Real)
  (hcos : Real.cos α = -4/5)
  (hquad : π/2 < α ∧ α < π) :
  (-Real.sin (2 * α) / Real.cos α) = -6/5 := 
by
  sorry

end trigonometric_identity_l521_52183


namespace annual_raise_l521_52165

-- Definitions based on conditions
def new_hourly_rate := 20
def new_weekly_hours := 40
def old_hourly_rate := 16
def old_weekly_hours := 25
def weeks_in_year := 52

-- Statement of the theorem
theorem annual_raise (new_hourly_rate new_weekly_hours old_hourly_rate old_weekly_hours weeks_in_year : ℕ) : 
  new_hourly_rate * new_weekly_hours * weeks_in_year - old_hourly_rate * old_weekly_hours * weeks_in_year = 20800 := 
  sorry -- Proof is omitted

end annual_raise_l521_52165


namespace expression_evaluation_correct_l521_52177

theorem expression_evaluation_correct (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  ( ( ( (x - 2) ^ 2 * (x ^ 2 + x + 1) ^ 2 ) / (x ^ 3 - 1) ^ 2 ) ^ 2 *
    ( ( (x + 2) ^ 2 * (x ^ 2 - x + 1) ^ 2 ) / (x ^ 3 + 1) ^ 2 ) ^ 2 ) 
  = (x^2 - 4)^4 := 
sorry

end expression_evaluation_correct_l521_52177


namespace mike_seashells_l521_52155

theorem mike_seashells (initial total : ℕ) (h1 : initial = 79) (h2 : total = 142) :
    total - initial = 63 :=
by
  sorry

end mike_seashells_l521_52155


namespace problem_1_problem_2_l521_52167

-- Problem (1)
theorem problem_1 (x a : ℝ) (h_a : a = 1) (hP : x^2 - 4*a*x + 3*a^2 < 0) (hQ1 : x^2 - x - 6 ≤ 0) (hQ2 : x^2 + 2*x - 8 > 0) :
  2 < x ∧ x < 3 := sorry

-- Problem (2)
theorem problem_2 (a : ℝ) (h_a_pos : 0 < a) (h_suff_neccess : (¬(∀ x, x^2 - 4*a*x + 3*a^2 < 0) → ¬(∀ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0)) ∧
                   ¬(∀ x, x^2 - 4*a*x + 3*a^2 < 0) ≠ ¬(∀ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0)) :
  1 < a ∧ a ≤ 2 := sorry

end problem_1_problem_2_l521_52167


namespace boat_distance_against_stream_l521_52106

variable (v_s : ℝ)
variable (effective_speed_stream : ℝ := 15)
variable (speed_still_water : ℝ := 10)
variable (distance_along_stream : ℝ := 15)

theorem boat_distance_against_stream : 
  distance_along_stream / effective_speed_stream = 1 ∧ effective_speed_stream = speed_still_water + v_s →
  10 - v_s = 5 :=
by
  intros
  sorry

end boat_distance_against_stream_l521_52106


namespace intersect_at_two_points_l521_52114

theorem intersect_at_two_points (a : ℝ) :
  (∃ p q : ℝ × ℝ, 
    (p.1 - p.2 + 1 = 0) ∧ (2 * p.1 + p.2 - 4 = 0) ∧ (a * p.1 - p.2 + 2 = 0) ∧
    (q.1 - q.2 + 1 = 0) ∧ (2 * q.1 + q.2 - 4 = 0) ∧ (a * q.1 - q.2 + 2 = 0) ∧ p ≠ q) →
  (a = 1 ∨ a = -2) :=
by 
  sorry

end intersect_at_two_points_l521_52114


namespace car_service_month_l521_52195

-- Define the conditions
def first_service_month : ℕ := 3 -- Representing March as the 3rd month
def service_interval : ℕ := 7
def total_services : ℕ := 13

-- Define an auxiliary function to calculate months and reduce modulo 12
def nth_service_month (first_month : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  (first_month + (interval * (n - 1))) % 12

-- The theorem statement
theorem car_service_month : nth_service_month first_service_month service_interval total_services = 3 :=
by
  -- The proof steps will go here
  sorry

end car_service_month_l521_52195


namespace good_function_count_l521_52146

noncomputable def num_good_functions (n : ℕ) : ℕ :=
  if n < 2 then 0 else
    n * Nat.totient n

theorem good_function_count (n : ℕ) (h : n ≥ 2) :
  ∃ (f : ℤ → Fin (n + 1)), 
  (∀ k, 1 ≤ k ∧ k ≤ n - 1 → ∃ j, ∀ m, (f (m + j) : ℤ) ≡ (f (m + k) - f m : ℤ) [ZMOD (n + 1)]) → 
  num_good_functions n = n * Nat.totient n :=
sorry

end good_function_count_l521_52146


namespace range_of_a_l521_52150

theorem range_of_a (x a : ℝ) (h1 : -2 < x) (h2 : x ≤ 1) (h3 : |x - 2| < a) : a ≤ 0 :=
sorry

end range_of_a_l521_52150


namespace compare_y_values_l521_52151

variable (a : ℝ) (y₁ y₂ : ℝ)
variable (h : a > 0)
variable (p1 : y₁ = a * (-1 : ℝ)^2 - 4 * a * (-1 : ℝ) + 2)
variable (p2 : y₂ = a * (1 : ℝ)^2 - 4 * a * (1 : ℝ) + 2)

theorem compare_y_values : y₁ > y₂ :=
by {
  sorry
}

end compare_y_values_l521_52151


namespace expression_for_neg_x_l521_52142

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem expression_for_neg_x (f : ℝ → ℝ) (h_odd : odd_function f) (h_nonneg : ∀ (x : ℝ), 0 ≤ x → f x = x^2 - 2 * x) :
  ∀ x : ℝ, x < 0 → f x = -x^2 - 2 * x :=
by 
  intros x hx 
  have hx_pos : -x > 0 := by linarith 
  have h_fx_neg : f (-x) = -f x := h_odd x
  rw [h_nonneg (-x) (by linarith)] at h_fx_neg
  linarith

end expression_for_neg_x_l521_52142


namespace linear_combination_harmonic_l521_52176

-- Define the harmonic property for a function
def is_harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ x y, f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4

-- The main statement to be proven in Lean
theorem linear_combination_harmonic
  (f g : ℤ × ℤ → ℝ) (a b : ℝ) (hf : is_harmonic f) (hg : is_harmonic g) :
  is_harmonic (fun p => a * f p + b * g p) :=
by
  sorry

end linear_combination_harmonic_l521_52176


namespace monotonic_increase_range_of_alpha_l521_52124

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * Real.sin (ω * x) - (Real.sqrt 3 / 2) * Real.cos (ω * x)

theorem monotonic_increase_range_of_alpha
  (ω : ℝ) (hω : ω > 0)
  (zeros_form_ap : ∀ k : ℤ, ∃ x₀ : ℝ, f ω x₀ = 0 ∧ ∀ n : ℤ, f ω (x₀ + n * (π / 2)) = 0) :
  ∃ α : ℝ, 0 < α ∧ α < 5 * π / 12 ∧ ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ α → f ω x ≤ f ω y :=
sorry

end monotonic_increase_range_of_alpha_l521_52124


namespace range_of_m_l521_52103

noncomputable def M (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def N : Set ℝ := {x | x^2 - 2 * x - 8 < 0}
def U : Set ℝ := Set.univ
def CU_M (m : ℝ) : Set ℝ := {x | x < -m}
def empty_intersection (m : ℝ) : Prop := (CU_M m ∩ N = ∅)

theorem range_of_m (m : ℝ) : empty_intersection m → m ≥ 2 := by
  sorry

end range_of_m_l521_52103


namespace pool_depth_is_10_feet_l521_52171

-- Definitions based on conditions
def hoseRate := 60 -- cubic feet per minute
def poolWidth := 80 -- feet
def poolLength := 150 -- feet
def drainingTime := 2000 -- minutes

-- Proof goal: the depth of the pool is 10 feet
theorem pool_depth_is_10_feet :
  ∃ (depth : ℝ), depth = 10 ∧ (hoseRate * drainingTime) = (poolWidth * poolLength * depth) :=
by
  use 10
  sorry

end pool_depth_is_10_feet_l521_52171


namespace rectangular_to_polar_coordinates_l521_52105

noncomputable def polar_coordinates_of_point (x y : ℝ) : ℝ × ℝ := sorry

theorem rectangular_to_polar_coordinates :
  polar_coordinates_of_point 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) := sorry

end rectangular_to_polar_coordinates_l521_52105


namespace car_body_mass_l521_52160

theorem car_body_mass (m_model : ℕ) (scale : ℕ) : 
  m_model = 1 → scale = 11 → m_car = 1331 :=
by 
  intros h1 h2
  sorry

end car_body_mass_l521_52160


namespace smallest_natural_number_k_l521_52187

theorem smallest_natural_number_k :
  ∃ k : ℕ, k = 4 ∧ ∀ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ 1 ≤ n → a^(k) * (1 - a)^(n) < 1 / (n + 1)^3 :=
by
  sorry

end smallest_natural_number_k_l521_52187


namespace value_of_gg_neg1_l521_52108

def g (x : ℝ) : ℝ := 4 * x^2 + 3

theorem value_of_gg_neg1 : g (g (-1)) = 199 := by
  sorry

end value_of_gg_neg1_l521_52108


namespace buses_dispatched_theorem_l521_52182

-- Define the conditions and parameters
def buses_dispatched (buses: ℕ) (hours: ℕ) : ℕ :=
  buses * hours

-- Define the specific problem
noncomputable def buses_from_6am_to_4pm : ℕ :=
  let buses_per_hour := 5 / 2
  let hours         := 16 - 6
  buses_dispatched (buses_per_hour : ℕ) hours

-- State the theorem that needs to be proven
theorem buses_dispatched_theorem : buses_from_6am_to_4pm = 25 := 
by {
  -- This 'sorry' is a placeholder for the actual proof.
  sorry
}

end buses_dispatched_theorem_l521_52182


namespace correct_rounded_result_l521_52128

def round_to_nearest_ten (n : ℤ) : ℤ :=
  (n + 5) / 10 * 10

theorem correct_rounded_result :
  round_to_nearest_ten ((57 + 68) * 2) = 250 :=
by
  sorry

end correct_rounded_result_l521_52128


namespace recycling_program_earnings_l521_52197

-- Define conditions
def signup_earning : ℝ := 5.00
def referral_earning_tier1 : ℝ := 8.00
def referral_earning_tier2 : ℝ := 1.50
def friend_earning_signup : ℝ := 5.00
def friend_earning_tier2 : ℝ := 2.00

def initial_friend_count : ℕ := 5
def initial_friend_tier1_referrals_day1 : ℕ := 3
def initial_friend_tier1_referrals_week : ℕ := 2

def additional_friend_count : ℕ := 2
def additional_friend_tier1_referrals : ℕ := 1

-- Calculate Katrina's total earnings
def katrina_earnings : ℝ :=
  signup_earning +
  (initial_friend_count * referral_earning_tier1) +
  (initial_friend_count * initial_friend_tier1_referrals_day1 * referral_earning_tier2) +
  (initial_friend_count * initial_friend_tier1_referrals_week * referral_earning_tier2) +
  (additional_friend_count * referral_earning_tier1) +
  (additional_friend_count * additional_friend_tier1_referrals * referral_earning_tier2)

-- Calculate friends' total earnings
def friends_earnings : ℝ :=
  (initial_friend_count * friend_earning_signup) +
  (initial_friend_count * initial_friend_tier1_referrals_day1 * friend_earning_tier2) +
  (initial_friend_count * initial_friend_tier1_referrals_week * friend_earning_tier2) +
  (additional_friend_count * friend_earning_signup) +
  (additional_friend_count * additional_friend_tier1_referrals * friend_earning_tier2)

-- Calculate combined total earnings
def combined_earnings : ℝ := katrina_earnings + friends_earnings

-- The proof assertion
theorem recycling_program_earnings : combined_earnings = 190.50 :=
by sorry

end recycling_program_earnings_l521_52197


namespace factorization_of_m_squared_minus_4_l521_52175

theorem factorization_of_m_squared_minus_4 (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
by
  sorry

end factorization_of_m_squared_minus_4_l521_52175


namespace score_standard_deviation_l521_52156

theorem score_standard_deviation (mean std_dev : ℝ)
  (h1 : mean = 76)
  (h2 : mean - 2 * std_dev = 60) :
  100 = mean + 3 * std_dev :=
by
  -- Insert proof here
  sorry

end score_standard_deviation_l521_52156


namespace sum_of_coordinates_of_C_and_D_l521_52100

structure Point where
  x : ℤ
  y : ℤ

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def sum_coordinates (p1 p2 : Point) : ℤ :=
  p1.x + p1.y + p2.x + p2.y

def C : Point := { x := 3, y := -2 }
def D : Point := reflect_y C

theorem sum_of_coordinates_of_C_and_D : sum_coordinates C D = -4 := by
  sorry

end sum_of_coordinates_of_C_and_D_l521_52100


namespace common_ratio_value_l521_52144

variable (a : ℕ → ℝ) -- defining the geometric sequence as a function ℕ → ℝ
variable (q : ℝ) -- defining the common ratio

-- conditions from the problem
def geo_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

axiom h1 : geo_seq a q
axiom h2 : a 2020 = 8 * a 2017

-- main statement to be proved
theorem common_ratio_value : q = 2 :=
sorry

end common_ratio_value_l521_52144


namespace group_count_l521_52113

theorem group_count (sample_capacity : ℕ) (frequency : ℝ) (h_sample_capacity : sample_capacity = 80) (h_frequency : frequency = 0.125) : sample_capacity * frequency = 10 := 
by
  sorry

end group_count_l521_52113


namespace range_of_a_l521_52173

noncomputable def f : ℝ → ℝ := sorry
variable (f_even : ∀ x : ℝ, f x = f (-x))
variable (f_increasing : ∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f x ≤ f y)
variable (a : ℝ) (h : f a ≤ f 2)

theorem range_of_a (f_even : ∀ x : ℝ, f x = f (-x))
                   (f_increasing : ∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f x ≤ f y)
                   (h : f a ≤ f 2) :
                   a ≤ -2 ∨ a ≥ 2 :=
sorry

end range_of_a_l521_52173


namespace coin_problem_l521_52117

theorem coin_problem (n d q : ℕ) 
  (h1 : n + d + q = 30)
  (h2 : 5 * n + 10 * d + 25 * q = 410)
  (h3 : d = n + 4) : q - n = 2 :=
by
  sorry

end coin_problem_l521_52117


namespace number_of_solutions_eq_l521_52129

open Nat

theorem number_of_solutions_eq (n : ℕ) : 
  ∃ N, (∀ (x : ℝ), 1 ≤ x ∧ x ≤ n → x^2 - ⌊x^2⌋ = (x - ⌊x⌋)^2) → N = n^2 - n + 1 :=
by sorry

end number_of_solutions_eq_l521_52129
