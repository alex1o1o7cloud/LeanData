import Mathlib

namespace mindy_tax_rate_proof_l401_40139

noncomputable def mindy_tax_rate (M r : ℝ) : Prop :=
  let Mork_tax := 0.10 * M
  let Mindy_income := 3 * M
  let Mindy_tax := r * Mindy_income
  let Combined_tax_rate := 0.175
  let Combined_tax := Combined_tax_rate * (M + Mindy_income)
  Mork_tax + Mindy_tax = Combined_tax

theorem mindy_tax_rate_proof (M r : ℝ) 
  (h1 : Mork_tax_rate = 0.10) 
  (h2 : mindy_income = 3 * M) 
  (h3 : combined_tax_rate = 0.175) : 
  r = 0.20 := 
sorry

end mindy_tax_rate_proof_l401_40139


namespace revenue_and_empty_seats_l401_40190

-- Define seating and ticket prices
def seats_A : ℕ := 90
def seats_B : ℕ := 70
def seats_C : ℕ := 50
def VIP_seats : ℕ := 10

def ticket_A : ℕ := 15
def ticket_B : ℕ := 10
def ticket_C : ℕ := 5
def VIP_ticket : ℕ := 25

-- Define discounts
def discount : ℤ := 20

-- Define actual occupancy
def adults_A : ℕ := 35
def children_A : ℕ := 15
def adults_B : ℕ := 20
def seniors_B : ℕ := 5
def adults_C : ℕ := 10
def veterans_C : ℕ := 5
def VIP_occupied : ℕ := 10

-- Concession sales
def hot_dogs_sold : ℕ := 50
def hot_dog_price : ℕ := 4
def soft_drinks_sold : ℕ := 75
def soft_drink_price : ℕ := 2

-- Define the total revenue and empty seats calculation
theorem revenue_and_empty_seats :
  let revenue_from_tickets := (adults_A * ticket_A + children_A * ticket_A * (100 - discount) / 100 +
                               adults_B * ticket_B + seniors_B * ticket_B * (100 - discount) / 100 +
                               adults_C * ticket_C + veterans_C * ticket_C * (100 - discount) / 100 +
                               VIP_occupied * VIP_ticket)
  let revenue_from_concessions := (hot_dogs_sold * hot_dog_price + soft_drinks_sold * soft_drink_price)
  let total_revenue := revenue_from_tickets + revenue_from_concessions
  let empty_seats_A := seats_A - (adults_A + children_A)
  let empty_seats_B := seats_B - (adults_B + seniors_B)
  let empty_seats_C := seats_C - (adults_C + veterans_C)
  let empty_VIP_seats := VIP_seats - VIP_occupied
  total_revenue = 1615 ∧ empty_seats_A = 40 ∧ empty_seats_B = 45 ∧ empty_seats_C = 35 ∧ empty_VIP_seats = 0 := by
  sorry

end revenue_and_empty_seats_l401_40190


namespace cos_A_zero_l401_40167

theorem cos_A_zero (A : ℝ) (h : Real.tan A + (1 / Real.tan A) + 2 / (Real.cos A) = 4) : Real.cos A = 0 :=
sorry

end cos_A_zero_l401_40167


namespace tv_purchase_time_l401_40107

-- Define the constants
def monthly_income : ℕ := 30000
def food_expenses : ℕ := 15000
def utilities_expenses : ℕ := 5000
def other_expenses : ℕ := 2500
def current_savings : ℕ := 10000
def tv_cost : ℕ := 25000

-- Define the total expenses
def total_expenses : ℕ := food_expenses + utilities_expenses + other_expenses

-- Define the disposable income
def disposable_income : ℕ := monthly_income - total_expenses

-- Define the amount needed to buy the TV
def amount_needed : ℕ := tv_cost - current_savings

-- Define the number of months needed to save the amount needed
def number_of_months : ℕ := amount_needed / disposable_income

-- The theorem specifying that we need 2 months to save enough money for the TV
theorem tv_purchase_time : number_of_months = 2 := by
  sorry

end tv_purchase_time_l401_40107


namespace mutually_exclusive_B_C_l401_40175

-- Define the events A, B, C
def event_A (x y : ℕ → Bool) : Prop := ¬(x 1 = false ∨ x 2 = false)
def event_B (x y : ℕ → Bool) : Prop := x 1 = false ∧ x 2 = false
def event_C (x y : ℕ → Bool) : Prop := ¬(x 1 = false ∧ x 2 = false)

-- Prove that event B and event C are mutually exclusive
theorem mutually_exclusive_B_C (x y : ℕ → Bool) :
  (event_B x y ∧ event_C x y) ↔ false := sorry

end mutually_exclusive_B_C_l401_40175


namespace inheritance_amount_l401_40134

-- Define the conditions
variable (x : ℝ) -- Let x be the inheritance amount
variable (H1 : x * 0.25 + (x * 0.75 - 5000) * 0.15 + 5000 = 16500)

-- Define the theorem to prove the inheritance amount
theorem inheritance_amount (H1 : x * 0.25 + (0.75 * x - 5000) * 0.15 + 5000 = 16500) : x = 33794 := by
  sorry

end inheritance_amount_l401_40134


namespace second_hand_travel_distance_l401_40132

theorem second_hand_travel_distance (r : ℝ) (t : ℝ) (C : ℝ) :
    r = 8 ∧ t = 45 ∧ C = 2 * Real.pi * r → 
    r * C * t = 720 * Real.pi :=
by
  sorry

end second_hand_travel_distance_l401_40132


namespace problem1_problem2_l401_40125

theorem problem1 (x : ℝ) (t : ℝ) (hx : x ≥ 3) (ht : 0 < t ∧ t < 1) :
  x^t - (x-1)^t < (x-2)^t - (x-3)^t :=
sorry

theorem problem2 (x : ℝ) (t : ℝ) (hx : x ≥ 3) (ht : t > 1) :
  x^t - (x-1)^t > (x-2)^t - (x-3)^t :=
sorry

end problem1_problem2_l401_40125


namespace original_number_in_magician_game_l401_40100

theorem original_number_in_magician_game (a b c : ℕ) (habc : 100 * a + 10 * b + c = 332) (N : ℕ) (hN : N = 4332) :
    222 * (a + b + c) = 4332 → 100 * a + 10 * b + c = 332 :=
by 
  sorry

end original_number_in_magician_game_l401_40100


namespace no_b_gt_4_such_that_143b_is_square_l401_40109

theorem no_b_gt_4_such_that_143b_is_square :
  ∀ (b : ℕ), 4 < b → ¬ ∃ (n : ℕ), b^2 + 4 * b + 3 = n^2 :=
by sorry

end no_b_gt_4_such_that_143b_is_square_l401_40109


namespace cross_section_equilateral_triangle_l401_40178

-- Definitions and conditions
structure Cone where
  r : ℝ -- radius of the base circle
  R : ℝ -- radius of the semicircle
  h : ℝ -- slant height

axiom lateral_surface_unfolded (c : Cone) : c.R = 2 * c.r

def CrossSectionIsEquilateral (c : Cone) : Prop :=
  (c.h ^ 2 = (c.r * c.h)) ∧ (c.h = 2 * c.r)

-- Problem statement with conditions
theorem cross_section_equilateral_triangle (c : Cone) (h_equals_diameter : c.R = 2 * c.r) : CrossSectionIsEquilateral c :=
by
  sorry

end cross_section_equilateral_triangle_l401_40178


namespace S_15_eq_1695_l401_40183

open Nat

/-- Sum of the nth set described in the problem -/
def S (n : ℕ) : ℕ :=
  let first := 1 + (n * (n - 1)) / 2
  let last := first + n - 1
  (n * (first + last)) / 2

theorem S_15_eq_1695 : S 15 = 1695 :=
by
  sorry

end S_15_eq_1695_l401_40183


namespace max_ab_bc_ca_l401_40162

theorem max_ab_bc_ca (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 3) :
  ab + bc + ca ≤ 3 :=
sorry

end max_ab_bc_ca_l401_40162


namespace seats_not_occupied_l401_40116

def seats_per_row : ℕ := 8
def total_rows : ℕ := 12
def seat_utilization_ratio : ℚ := 3 / 4

theorem seats_not_occupied : 
  (seats_per_row * total_rows) - (seats_per_row * seat_utilization_ratio * total_rows) = 24 := 
by
  sorry

end seats_not_occupied_l401_40116


namespace find_principal_amount_l401_40166

theorem find_principal_amount 
  (A : ℝ) (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ)
  (hA : A = 3087) (hr : r = 0.05) (hn : n = 1) (ht : t = 2)
  (hcomp : A = P * (1 + r / n)^(n * t)) :
  P = 2800 := 
  by sorry

end find_principal_amount_l401_40166


namespace positive_difference_of_two_numbers_l401_40196

theorem positive_difference_of_two_numbers :
  ∀ (x y : ℝ), x + y = 8 → x^2 - y^2 = 24 → abs (x - y) = 3 :=
by
  intros x y h1 h2
  sorry

end positive_difference_of_two_numbers_l401_40196


namespace harriet_travel_time_l401_40122

theorem harriet_travel_time (D : ℝ) (h : (D / 90 + D / 160 = 5)) : (D / 90) * 60 = 192 := 
by sorry

end harriet_travel_time_l401_40122


namespace real_part_zero_implies_x3_l401_40199

theorem real_part_zero_implies_x3 (x : ℝ) : 
  (x^2 - 2*x - 3 = 0) ∧ (x + 1 ≠ 0) → x = 3 :=
by
  sorry

end real_part_zero_implies_x3_l401_40199


namespace minimum_x_condition_l401_40119

theorem minimum_x_condition (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (h : x - 2 * y = (x + 16 * y) / (2 * x * y)) : 
  x ≥ 4 :=
sorry

end minimum_x_condition_l401_40119


namespace geometric_sequence_seventh_term_l401_40182

-- Define the initial conditions
def geometric_sequence_first_term := 3
def geometric_sequence_fifth_term (r : ℝ) := geometric_sequence_first_term * r^4 = 243

-- Statement for the seventh term problem
theorem geometric_sequence_seventh_term (r : ℝ) 
  (h1 : geometric_sequence_first_term = 3) 
  (h2 : geometric_sequence_fifth_term r) : 
  3 * r^6 = 2187 :=
sorry

end geometric_sequence_seventh_term_l401_40182


namespace binary_subtraction_l401_40154

theorem binary_subtraction : ∀ (x y : ℕ), x = 0b11011 → y = 0b101 → x - y = 0b10110 :=
by
  sorry

end binary_subtraction_l401_40154


namespace original_price_l401_40126

theorem original_price (sale_price gain_percent : ℕ) (h_sale : sale_price = 130) (h_gain : gain_percent = 30) : 
    ∃ P : ℕ, (P * (1 + gain_percent / 100)) = sale_price := 
by
  use 100
  rw [h_sale, h_gain]
  norm_num
  sorry

end original_price_l401_40126


namespace max_s_value_l401_40169

noncomputable def max_s (m n : ℝ) : ℝ := (m-1)^2 + (n-1)^2 + (m-n)^2

theorem max_s_value (m n : ℝ) (h : m^2 - 4 * n ≥ 0) : 
    ∃ s : ℝ, s = (max_s m n) ∧ s ≤ 9/8 := sorry

end max_s_value_l401_40169


namespace part1_part2_l401_40163

noncomputable def f (x a : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem part1 (a : ℝ) (h : 0 < a) :
  (∀ x > 0, (Real.log x + 1/x + 1 - a ≥ 0)) ↔ (0 < a ∧ a ≤ 2) := sorry

theorem part2 (a : ℝ) (h : 0 < a) :
  (∀ x > 0, (x - 1) * f x a ≥ 0) ↔ (0 < a ∧ a ≤ 2) := sorry

end part1_part2_l401_40163


namespace average_salary_of_all_workers_l401_40110

-- Definitions of conditions
def num_technicians : ℕ := 7
def num_total_workers : ℕ := 12
def num_other_workers : ℕ := num_total_workers - num_technicians

def avg_salary_technicians : ℝ := 12000
def avg_salary_others : ℝ := 6000

-- Total salary calculations
def total_salary_technicians : ℝ := num_technicians * avg_salary_technicians
def total_salary_others : ℝ := num_other_workers * avg_salary_others

def total_salary : ℝ := total_salary_technicians + total_salary_others

-- Proof statement: the average salary of all workers is 9500
theorem average_salary_of_all_workers : total_salary / num_total_workers = 9500 :=
by
  sorry

end average_salary_of_all_workers_l401_40110


namespace sin_pi_over_six_l401_40124

theorem sin_pi_over_six : Real.sin (Real.pi / 6) = 1 / 2 := 
by 
  sorry

end sin_pi_over_six_l401_40124


namespace div120_l401_40140

theorem div120 (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
sorry

end div120_l401_40140


namespace combined_mpg_proof_l401_40120

noncomputable def combined_mpg (d : ℝ) : ℝ :=
  let ray_mpg := 50
  let tom_mpg := 20
  let alice_mpg := 25
  let total_fuel := (d / ray_mpg) + (d / tom_mpg) + (d / alice_mpg)
  let total_distance := 3 * d
  total_distance / total_fuel

theorem combined_mpg_proof :
  ∀ d : ℝ, d > 0 → combined_mpg d = 300 / 11 :=
by
  intros d hd
  rw [combined_mpg]
  simp only [div_eq_inv_mul, mul_inv, inv_inv]
  sorry

end combined_mpg_proof_l401_40120


namespace area_of_inscribed_rectangle_l401_40123

theorem area_of_inscribed_rectangle
  (s : ℕ) (R_area : ℕ)
  (h1 : s = 4) 
  (h2 : 2 * 4 + 1 * 1 + R_area = s * s) :
  R_area = 7 :=
by
  sorry

end area_of_inscribed_rectangle_l401_40123


namespace factorize_problem1_factorize_problem2_factorize_problem3_factorize_problem4_l401_40121

-- Problem 1: Prove equivalence for factorizing -2a^2 + 4a.
theorem factorize_problem1 (a : ℝ) : -2 * a^2 + 4 * a = -2 * a * (a - 2) := 
by sorry

-- Problem 2: Prove equivalence for factorizing 4x^3 y - 9xy^3.
theorem factorize_problem2 (x y : ℝ) : 4 * x^3 * y - 9 * x * y^3 = x * y * (2 * x + 3 * y) * (2 * x - 3 * y) := 
by sorry

-- Problem 3: Prove equivalence for factorizing 4x^2 - 12x + 9.
theorem factorize_problem3 (x : ℝ) : 4 * x^2 - 12 * x + 9 = (2 * x - 3)^2 := 
by sorry

-- Problem 4: Prove equivalence for factorizing (a+b)^2 - 6(a+b) + 9.
theorem factorize_problem4 (a b : ℝ) : (a + b)^2 - 6 * (a + b) + 9 = (a + b - 3)^2 := 
by sorry

end factorize_problem1_factorize_problem2_factorize_problem3_factorize_problem4_l401_40121


namespace f_comp_g_eq_g_comp_f_has_solution_l401_40195

variable {R : Type*} [Field R]

def f (a b x : R) : R := a * x + b
def g (c d x : R) : R := c * x ^ 2 + d

theorem f_comp_g_eq_g_comp_f_has_solution (a b c d : R) :
  (∃ x : R, f a b (g c d x) = g c d (f a b x)) ↔ (c = 0 ∨ a * b = 0) ∧ (a * d - c * b ^ 2 + b - d = 0) := by
  sorry

end f_comp_g_eq_g_comp_f_has_solution_l401_40195


namespace tomatoes_ruined_percentage_l401_40136

-- The definitions from the problem conditions
def tomato_cost_per_pound : ℝ := 0.80
def tomato_selling_price_per_pound : ℝ := 0.977777777777778
def desired_profit_percent : ℝ := 0.10
def revenue_equal_cost_plus_profit_cost_fraction : ℝ := (tomato_cost_per_pound + (tomato_cost_per_pound * desired_profit_percent))

-- The theorem stating the problem and the expected result
theorem tomatoes_ruined_percentage :
  ∀ (W : ℝ) (P : ℝ),
  (0.977777777777778 * (1 - P / 100) * W = (0.80 * W + 0.08 * W)) →
  P = 10.00000000000001 :=
by
  intros W P h
  have eq1 : 0.977777777777778 * (1 - P / 100) = 0.88 := sorry
  have eq2 : 1 - P / 100 = 0.8999999999999999 := sorry
  have eq3 : P / 100 = 0.1000000000000001 := sorry
  exact sorry

end tomatoes_ruined_percentage_l401_40136


namespace multiple_of_5_digits_B_l401_40174

theorem multiple_of_5_digits_B (B : ℕ) : B = 0 ∨ B = 5 ↔ 23 * 10 + B % 5 = 0 :=
by
  sorry

end multiple_of_5_digits_B_l401_40174


namespace number_of_points_on_line_l401_40168

theorem number_of_points_on_line (a b c d : ℕ) (h1 : a * b = 80) (h2 : c * d = 90) (h3 : a + b = c + d) :
  a + b + 1 = 22 :=
sorry

end number_of_points_on_line_l401_40168


namespace quadratic_roots_condition_l401_40142

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1*x1 + m*x1 + 4 = 0 ∧ x2*x2 + m*x2 + 4 = 0) →
  m ≤ -4 :=
by
  sorry

end quadratic_roots_condition_l401_40142


namespace trivia_team_points_l401_40128

theorem trivia_team_points : 
  let member1_points := 8
  let member2_points := 12
  let member3_points := 9
  let member4_points := 5
  let member5_points := 10
  let member6_points := 7
  let member7_points := 14
  let member8_points := 11
  (member1_points + member2_points + member3_points + member4_points + member5_points + member6_points + member7_points + member8_points) = 76 :=
by
  let member1_points := 8
  let member2_points := 12
  let member3_points := 9
  let member4_points := 5
  let member5_points := 10
  let member6_points := 7
  let member7_points := 14
  let member8_points := 11
  sorry

end trivia_team_points_l401_40128


namespace sum_gt_product_iff_l401_40130

theorem sum_gt_product_iff (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : m + n > m * n ↔ m = 1 ∨ n = 1 :=
sorry

end sum_gt_product_iff_l401_40130


namespace smallest_integer_is_nine_l401_40189

theorem smallest_integer_is_nine 
  (a b c : ℕ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : a + b + c = 90) 
  (h3 : (a:ℝ)/b = 2/3) 
  (h4 : (b:ℝ)/c = 3/5) : 
  a = 9 :=
by 
  sorry

end smallest_integer_is_nine_l401_40189


namespace value_of_a7_l401_40153

theorem value_of_a7 (a : ℕ → ℤ) (h1 : a 1 = 0) (h2 : ∀ n, a (n + 2) - a n = 2) : a 7 = 6 :=
by {
  sorry -- Proof goes here
}

end value_of_a7_l401_40153


namespace smallest_b_perfect_fourth_power_l401_40129

theorem smallest_b_perfect_fourth_power:
  ∃ b : ℕ, (∀ n : ℕ, 5 * n = (7 * b^2 + 7 * b + 7) → ∃ x : ℕ, n = x^4) 
  ∧ b = 41 :=
sorry

end smallest_b_perfect_fourth_power_l401_40129


namespace harry_worked_16_hours_l401_40150

-- Define the given conditions
def harrys_pay_first_30_hours (x : ℝ) : ℝ := 30 * x
def harrys_pay_additional_hours (x H : ℝ) : ℝ := (H - 30) * 2 * x
def james_pay_first_40_hours (x : ℝ) : ℝ := 40 * x
def james_pay_additional_hour (x : ℝ) : ℝ := 2 * x
def james_total_hours : ℝ := 41

-- Given that Harry and James are paid the same amount 
-- Prove that Harry worked 16 hours last week
theorem harry_worked_16_hours (x H : ℝ) 
  (h1 : harrys_pay_first_30_hours x + harrys_pay_additional_hours x H = james_pay_first_40_hours x + james_pay_additional_hour x) 
  : H = 16 :=
by
  sorry

end harry_worked_16_hours_l401_40150


namespace gift_cost_l401_40177

def ErikaSavings : ℕ := 155
def CakeCost : ℕ := 25
def LeftOver : ℕ := 5

noncomputable def CostOfGift (RickSavings : ℕ) : ℕ :=
  2 * RickSavings

theorem gift_cost (RickSavings : ℕ)
  (hRick : RickSavings = CostOfGift RickSavings / 2)
  (hTotal : ErikaSavings + RickSavings = CostOfGift RickSavings + CakeCost + LeftOver) :
  CostOfGift RickSavings = 250 :=
by
  sorry

end gift_cost_l401_40177


namespace correct_calculation_l401_40113

theorem correct_calculation (a b : ℕ) : a^3 * b^3 = (a * b)^3 :=
sorry

end correct_calculation_l401_40113


namespace min_points_condition_met_l401_40101

noncomputable def min_points_on_circle (L : ℕ) : ℕ := 1304

theorem min_points_condition_met (L : ℕ) (hL : L = 1956) :
  (∀ (points : ℕ → ℕ), (∀ n, points n ≠ points (n + 1) ∧ points n ≠ points (n + 2)) ∧ (∀ n, points n < L)) →
  min_points_on_circle L = 1304 :=
by
  -- Proof steps omitted
  sorry

end min_points_condition_met_l401_40101


namespace victory_circle_count_l401_40155

   -- Define the conditions
   def num_runners : ℕ := 8
   def num_medals : ℕ := 5
   def medals : List String := ["gold", "silver", "bronze", "titanium", "copper"]
   
   -- Define the scenarios
   def scenario1 : ℕ := 2 * 6 -- 2! * 3!
   def scenario2 : ℕ := 6 * 2 -- 3! * 2!
   def scenario3 : ℕ := 2 * 2 * 1 -- 2! * 2! * 1!

   -- Calculate the total number of victory circles
   def total_victory_circles : ℕ := scenario1 + scenario2 + scenario3

   theorem victory_circle_count : total_victory_circles = 28 := by
     sorry
   
end victory_circle_count_l401_40155


namespace total_pies_sold_l401_40115

theorem total_pies_sold :
  let shepherd_slices := 52
  let chicken_slices := 80
  let shepherd_pieces_per_pie := 4
  let chicken_pieces_per_pie := 5
  let shepherd_pies := shepherd_slices / shepherd_pieces_per_pie
  let chicken_pies := chicken_slices / chicken_pieces_per_pie
  shepherd_pies + chicken_pies = 29 :=
by
  sorry

end total_pies_sold_l401_40115


namespace rice_grains_difference_l401_40193

theorem rice_grains_difference : 
  3^15 - (3^1 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9 + 3^10) = 14260335 := 
by
  sorry

end rice_grains_difference_l401_40193


namespace factorization_of_x_squared_minus_4_l401_40135

theorem factorization_of_x_squared_minus_4 (x : ℝ) : x^2 - 4 = (x - 2) * (x + 2) :=
by
  sorry

end factorization_of_x_squared_minus_4_l401_40135


namespace average_marks_physics_mathematics_l401_40173

theorem average_marks_physics_mathematics {P C M : ℕ} (h1 : P + C + M = 180) (h2 : P = 140) (h3 : P + C = 140) : 
  (P + M) / 2 = 90 := by
  sorry

end average_marks_physics_mathematics_l401_40173


namespace randy_initial_money_l401_40180

/--
Initially, Randy had an unknown amount of money. He was given $2000 by Smith and $900 by Michelle.
After that, Randy gave Sally a 1/4th of his total money after which he gave Jake and Harry $800 and $500 respectively.
If Randy is left with $5500 after all the transactions, prove that Randy initially had $6166.67.
-/
theorem randy_initial_money (X : ℝ) :
  (3/4 * (X + 2000 + 900) - 1300 = 5500) -> (X = 6166.67) :=
by
  sorry

end randy_initial_money_l401_40180


namespace balloon_difference_l401_40133

def num_balloons_you := 7
def num_balloons_friend := 5

theorem balloon_difference : (num_balloons_you - num_balloons_friend) = 2 := by
  sorry

end balloon_difference_l401_40133


namespace R_depends_on_d_and_n_l401_40105

def arith_seq_sum (a d n : ℕ) (S1 S2 S3 : ℕ) : Prop := 
  (S1 = n * (a + (n - 1) * d / 2)) ∧ 
  (S2 = n * (2 * a + (2 * n - 1) * d)) ∧ 
  (S3 = 3 * n * (a + (3 * n - 1) * d / 2))

theorem R_depends_on_d_and_n (a d n S1 S2 S3 : ℕ) 
  (hS1 : S1 = n * (a + (n - 1) * d / 2))
  (hS2 : S2 = n * (2 * a + (2 * n - 1) * d))
  (hS3 : S3 = 3 * n * (a + (3 * n - 1) * d / 2)) 
  : S3 - S2 - S1 = 2 * n^2 * d  :=
by
  sorry

end R_depends_on_d_and_n_l401_40105


namespace smallest_prime_reverse_square_l401_40127

open Nat

-- Define a function to reverse the digits of a two-digit number
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

-- Define the conditions
def isTwoDigitPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

def isSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the main statement
theorem smallest_prime_reverse_square : 
  ∃ P, isTwoDigitPrime P ∧ isSquare (reverseDigits P) ∧ 
       ∀ Q, isTwoDigitPrime Q ∧ isSquare (reverseDigits Q) → P ≤ Q :=
by
  sorry

end smallest_prime_reverse_square_l401_40127


namespace perimeter_of_figure_l401_40186

def side_length : ℕ := 1
def num_vertical_stacks : ℕ := 2
def num_squares_per_stack : ℕ := 3
def gap_between_stacks : ℕ := 1
def squares_on_top : ℕ := 3
def squares_on_bottom : ℕ := 2

theorem perimeter_of_figure : 
  (2 * side_length * squares_on_top) + (2 * side_length * squares_on_bottom) + 
  (2 * num_squares_per_stack * num_vertical_stacks) + (2 * num_squares_per_stack * squares_on_top)
  = 22 :=
by
  sorry

end perimeter_of_figure_l401_40186


namespace trig_expression_l401_40179

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
  (3 * Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 5 / 7 :=
by
  sorry

end trig_expression_l401_40179


namespace abs_a1_plus_abs_a2_to_abs_a6_l401_40131

theorem abs_a1_plus_abs_a2_to_abs_a6 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ)
  (h : (2 - x) ^ 6 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5 + a₆ * x ^ 6) :
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| = 665 :=
sorry

end abs_a1_plus_abs_a2_to_abs_a6_l401_40131


namespace printer_ratio_l401_40191

-- Define the given conditions
def total_price_basic_computer_printer := 2500
def enhanced_computer_extra := 500
def basic_computer_price := 1500

-- The lean statement to prove the ratio of the price of the printer to the total price of the enhanced computer and printer is 1/3
theorem printer_ratio : ∀ (C_basic P C_enhanced Total_enhanced : ℕ), 
  C_basic + P = total_price_basic_computer_printer →
  C_enhanced = C_basic + enhanced_computer_extra →
  C_basic = basic_computer_price →
  C_enhanced + P = Total_enhanced →
  P / Total_enhanced = 1 / 3 := 
by
  intros C_basic P C_enhanced Total_enhanced h1 h2 h3 h4
  sorry

end printer_ratio_l401_40191


namespace range_of_a_l401_40172

noncomputable def p (x : ℝ) : Prop := x^2 - 8 * x - 20 < 0

noncomputable def q (x a : ℝ) : Prop := x^2 - 2 * x + 1 - a^2 ≤ 0

def sufficient_but_not_necessary_condition (a : ℝ) : Prop :=
  ∀ x, p x → q x a

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : sufficient_but_not_necessary_condition a) :
  9 ≤ a :=
sorry

end range_of_a_l401_40172


namespace Piper_gym_sessions_l401_40176

theorem Piper_gym_sessions
  (start_on_monday : Bool)
  (alternate_except_sunday : (∀ (n : ℕ), n % 2 = 1 → n % 7 ≠ 0 → Bool))
  (sessions_over_on_wednesday : Bool)
  : ∃ (n : ℕ), n = 5 :=
by 
  sorry

end Piper_gym_sessions_l401_40176


namespace company_profits_ratio_l401_40111

def companyN_2008_profits (RN : ℝ) : ℝ := 0.08 * RN
def companyN_2009_profits (RN : ℝ) : ℝ := 0.15 * (0.8 * RN)
def companyN_2010_profits (RN : ℝ) : ℝ := 0.10 * (1.3 * 0.8 * RN)

def companyM_2008_profits (RM : ℝ) : ℝ := 0.12 * RM
def companyM_2009_profits (RM : ℝ) : ℝ := 0.18 * RM
def companyM_2010_profits (RM : ℝ) : ℝ := 0.14 * RM

def total_profits_N (RN : ℝ) : ℝ :=
  companyN_2008_profits RN + companyN_2009_profits RN + companyN_2010_profits RN

def total_profits_M (RM : ℝ) : ℝ :=
  companyM_2008_profits RM + companyM_2009_profits RM + companyM_2010_profits RM

theorem company_profits_ratio (RN RM : ℝ) :
  total_profits_N RN / total_profits_M RM = (0.304 * RN) / (0.44 * RM) :=
by
  unfold total_profits_N companyN_2008_profits companyN_2009_profits companyN_2010_profits
  unfold total_profits_M companyM_2008_profits companyM_2009_profits companyM_2010_profits
  simp
  sorry

end company_profits_ratio_l401_40111


namespace evaluate_nested_function_l401_40112

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 / 2 else 2 ^ x

theorem evaluate_nested_function : f (f (1 / 2)) = 2 := 
by
  sorry

end evaluate_nested_function_l401_40112


namespace triangle_ABC_l401_40184

theorem triangle_ABC (a b c : ℝ) (A B C : ℝ)
  (h1 : a + b = 5)
  (h2 : c = Real.sqrt 7)
  (h3 : 4 * (Real.sin ((A + B) / 2))^2 - Real.cos (2 * C) = 7 / 2) :
  (C = Real.pi / 3)
  ∧ (1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) :=
by
  sorry

end triangle_ABC_l401_40184


namespace integer_classes_mod4_l401_40114

theorem integer_classes_mod4:
  (2021 % 4) = 1 ∧ (∀ a b : ℤ, (a % 4 = 2) ∧ (b % 4 = 3) → (a + b) % 4 = 1) := by
  sorry

end integer_classes_mod4_l401_40114


namespace constant_term_binomial_expansion_l401_40104

theorem constant_term_binomial_expansion : 
  let a := (1 : ℚ) / (x : ℚ) -- Note: Here 'x' is not bound, in actual Lean code x should be a declared variable in ℚ.
  let b := 2 * (x : ℚ)
  let n := 6
  let T (r : ℕ) := (Nat.choose n r : ℚ) * a^(n - r) * b^r
  (T 3) = (160 : ℚ) := by
  sorry

end constant_term_binomial_expansion_l401_40104


namespace tangent_line_to_parabola_l401_40138

theorem tangent_line_to_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c → y^2 = 12 * x) → c = 1 :=
by
  sorry

end tangent_line_to_parabola_l401_40138


namespace sqrt_eq_solutions_l401_40188

theorem sqrt_eq_solutions (x : ℝ) : 
  (Real.sqrt ((2 + Real.sqrt 5) ^ x) + Real.sqrt ((2 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
by
  sorry

end sqrt_eq_solutions_l401_40188


namespace find_n_l401_40103

theorem find_n (n : ℕ) (h1 : Nat.gcd n 180 = 12) (h2 : Nat.lcm n 180 = 720) : n = 48 := 
by
  sorry

end find_n_l401_40103


namespace dividends_CEO_2018_l401_40147

theorem dividends_CEO_2018 (Revenue Expenses Tax_rate Loan_payment_per_month : ℝ) 
  (Number_of_shares : ℕ) (CEO_share_percentage : ℝ)
  (hRevenue : Revenue = 2500000) 
  (hExpenses : Expenses = 1576250)
  (hTax_rate : Tax_rate = 0.2)
  (hLoan_payment_per_month : Loan_payment_per_month = 25000)
  (hNumber_of_shares : Number_of_shares = 1600)
  (hCEO_share_percentage : CEO_share_percentage = 0.35) :
  CEO_share_percentage * ((Revenue - Expenses) * (1 - Tax_rate) - Loan_payment_per_month * 12) / Number_of_shares * Number_of_shares = 153440 :=
sorry

end dividends_CEO_2018_l401_40147


namespace proof_candle_burn_l401_40151

noncomputable def candle_burn_proof : Prop :=
∃ (t : ℚ),
  (t = 40 / 11) ∧
  (∀ (H_1 H_2 : ℚ → ℚ),
    (∀ t, H_1 t = 1 - t / 5) ∧
    (∀ t, H_2 t = 1 - t / 4) →
    ∃ (t : ℚ), ((1 - t / 5) = 3 * (1 - t / 4)) ∧ (t = 40 / 11))

theorem proof_candle_burn : candle_burn_proof :=
sorry

end proof_candle_burn_l401_40151


namespace smallest_integer_satisfying_conditions_l401_40117

theorem smallest_integer_satisfying_conditions :
  ∃ M : ℕ, M % 7 = 6 ∧ M % 8 = 7 ∧ M % 9 = 8 ∧ M % 10 = 9 ∧ M % 11 = 10 ∧ M % 12 = 11 ∧ M = 27719 := by
  sorry

end smallest_integer_satisfying_conditions_l401_40117


namespace jordon_machine_input_l401_40144

theorem jordon_machine_input (x : ℝ) : (3 * x - 6) / 2 + 9 = 27 → x = 14 := 
by
  sorry

end jordon_machine_input_l401_40144


namespace probability_at_least_6_heads_l401_40165

-- Definitions of the binomial coefficient and probability function
def binom (n k : ℕ) : ℕ := Nat.choose n k

def probability (favorable total : ℕ) : ℚ := favorable / total

-- Proof problem statement
theorem probability_at_least_6_heads (flips : ℕ) (p : ℚ) 
  (h_flips : flips = 8) 
  (h_probability : p = probability (binom 8 6 + binom 8 7 + binom 8 8) (2 ^ flips)) : 
  p = 37 / 256 :=
by
  sorry

end probability_at_least_6_heads_l401_40165


namespace proving_four_digit_number_l401_40145

def distinct (a b c d : Nat) : Prop :=
a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def same_parity (x y : Nat) : Prop :=
(x % 2 = 0 ∧ y % 2 = 0) ∨ (x % 2 = 1 ∧ y % 2 = 1)

def different_parity (x y : Nat) : Prop :=
¬same_parity x y

theorem proving_four_digit_number :
  ∃ (A B C D : Nat),
    distinct A B C D ∧
    (different_parity A B → B ≠ 4) ∧
    (different_parity B C → C ≠ 3) ∧
    (different_parity C D → D ≠ 2) ∧
    (different_parity D A → A ≠ 1) ∧
    A + D < B + C ∧
    1000 * A + 100 * B + 10 * C + D = 2341 :=
by
  sorry

end proving_four_digit_number_l401_40145


namespace altitude_point_intersect_and_length_equalities_l401_40102

variables (A B C D E H : Type)
variables (triangle : A ≠ B ∧ B ≠ C ∧ C ≠ A)
variables (acute : ∀ (a b c : A), True) -- Placeholder for the acute triangle condition
variables (altitude_AD : True) -- Placeholder for the specific definition of altitude AD
variables (altitude_BE : True) -- Placeholder for the specific definition of altitude BE
variables (HD HE AD : ℝ)
variables (BD DC AE EC : ℝ)

theorem altitude_point_intersect_and_length_equalities
  (HD_eq : HD = 3)
  (HE_eq : HE = 4) 
  (sim1 : BD / 3 = (AD + 3) / DC)
  (sim2 : AE / 4 = (BE + 4) / EC)
  (sim3 : 4 * AD = 3 * BE) :
  (BD * DC) - (AE * EC) = 3 * AD - 7 := by
  sorry

end altitude_point_intersect_and_length_equalities_l401_40102


namespace inverse_r_l401_40156

def p (x: ℝ) : ℝ := 4 * x + 5
def q (x: ℝ) : ℝ := 3 * x - 4
def r (x: ℝ) : ℝ := p (q x)

theorem inverse_r (x : ℝ) : r⁻¹ x = (x + 11) / 12 :=
sorry

end inverse_r_l401_40156


namespace weights_balance_l401_40149

theorem weights_balance (k : ℕ) 
    (m n : ℕ → ℝ) 
    (h1 : ∀ i : ℕ, i < k → m i > n i) 
    (h2 : ∀ i : ℕ, i < k → ∃ j : ℕ, j ≠ i ∧ (m i + n j = n i + m j 
                                               ∨ m j + n i = n j + m i)) 
    : k = 1 ∨ k = 2 := 
by sorry

end weights_balance_l401_40149


namespace max_consecutive_integers_sum_le_500_l401_40118

def consecutive_sum (n : ℕ) : ℕ :=
  -- Formula for sum starting from 3
  (n * (n + 1)) / 2 - 3

theorem max_consecutive_integers_sum_le_500 : ∃ n : ℕ, consecutive_sum n ≤ 500 ∧ ∀ m : ℕ, m > n → consecutive_sum m > 500 :=
by
  sorry

end max_consecutive_integers_sum_le_500_l401_40118


namespace length_of_PR_l401_40146

theorem length_of_PR (x y : ℝ) (h₁ : x^2 + y^2 = 250) : 
  ∃ PR : ℝ, PR = 10 * Real.sqrt 5 :=
by
  use Real.sqrt (2 * (x^2 + y^2))
  sorry

end length_of_PR_l401_40146


namespace total_marbles_l401_40143

theorem total_marbles (r b y : ℕ) (h_ratio : 2 * b = 3 * r) (h_ratio_alt : 4 * b = 3 * y) (h_blue_marbles : b = 24) : r + b + y = 72 :=
by
  -- By assumption, b = 24
  have h1 : b = 24 := h_blue_marbles

  -- We have the ratios 2b = 3r and 4b = 3y
  have h2 : 2 * b = 3 * r := h_ratio
  have h3 : 4 * b = 3 * y := h_ratio_alt

  -- solved by given conditions 
  sorry

end total_marbles_l401_40143


namespace solve_for_x_l401_40170

theorem solve_for_x (x : ℝ) (h : (2 / 3 - 1 / 4) = 4 / x) : x = 48 / 5 :=
by sorry

end solve_for_x_l401_40170


namespace trig_identity_l401_40181

theorem trig_identity (x : ℝ) (h : 2 * Real.cos x - 5 * Real.sin x = 3) :
  (Real.sin x + 2 * Real.cos x = 1 / 2) ∨ (Real.sin x + 2 * Real.cos x = 83 / 29) := sorry

end trig_identity_l401_40181


namespace survey_no_preference_students_l401_40157

theorem survey_no_preference_students (total_students pref_mac pref_both pref_windows : ℕ) 
    (h1 : total_students = 210) 
    (h2 : pref_mac = 60) 
    (h3 : pref_both = pref_mac / 3)
    (h4 : pref_windows = 40) : 
    total_students - (pref_mac + pref_both + pref_windows) = 90 :=
by
  sorry

end survey_no_preference_students_l401_40157


namespace probability_of_6_consecutive_heads_l401_40194

/-- Define the probability of obtaining at least 6 consecutive heads in 10 flips of a fair coin. -/
def prob_at_least_6_consecutive_heads : ℚ :=
  129 / 1024

/-- Proof statement: The probability of getting at least 6 consecutive heads in 10 flips of a fair coin is 129/1024. -/
theorem probability_of_6_consecutive_heads : 
  prob_at_least_6_consecutive_heads = 129 / 1024 := 
by
  sorry

end probability_of_6_consecutive_heads_l401_40194


namespace at_least_one_inequality_holds_l401_40197

theorem at_least_one_inequality_holds
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end at_least_one_inequality_holds_l401_40197


namespace indoor_table_chairs_l401_40198

theorem indoor_table_chairs (x : ℕ) :
  (9 * x) + (11 * 3) = 123 → x = 10 :=
by
  intro h
  sorry

end indoor_table_chairs_l401_40198


namespace greatest_three_digit_base_nine_divisible_by_seven_l401_40148

/-- Define the problem setup -/
def greatest_three_digit_base_nine := 8 * 9^2 + 8 * 9 + 8

/-- Prove the greatest 3-digit base 9 positive integer that is divisible by 7 -/
theorem greatest_three_digit_base_nine_divisible_by_seven : 
  ∃ n : ℕ, n = greatest_three_digit_base_nine ∧ n % 7 = 0 ∧ (8 * 9^2 + 8 * 9 + 8) = 728 := by 
  sorry

end greatest_three_digit_base_nine_divisible_by_seven_l401_40148


namespace y_gt_1_l401_40161

theorem y_gt_1 (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
by sorry

end y_gt_1_l401_40161


namespace cubes_not_touching_foil_l401_40141

-- Define the variables for length, width, height, and total cubes
variables (l w h : ℕ)

-- Conditions extracted from the problem
def width_is_twice_length : Prop := w = 2 * l
def width_is_twice_height : Prop := w = 2 * h
def foil_covered_prism_width : Prop := w + 2 = 10

-- The proof statement
theorem cubes_not_touching_foil (l w h : ℕ) 
  (h1 : width_is_twice_length l w) 
  (h2 : width_is_twice_height w h) 
  (h3 : foil_covered_prism_width w) : 
  l * w * h = 128 := 
by sorry

end cubes_not_touching_foil_l401_40141


namespace sequence_proof_l401_40108

theorem sequence_proof (S : ℕ → ℝ) (a : ℕ → ℝ)
  (h : ∀ n : ℕ, n > 0 → a n = 2 - S n)
  (hS : ∀ n : ℕ, S (n + 1) = S n + a (n + 1) ) :
  (a 1 = 1 ∧ a 2 = 1/2 ∧ a 3 = 1/4 ∧ a 4 = 1/8) ∧ (∀ n : ℕ, n > 0 → a n = (1/2)^(n-1)) :=
by
  sorry

end sequence_proof_l401_40108


namespace stickers_total_l401_40171

def karl_stickers : ℕ := 25
def ryan_stickers : ℕ := karl_stickers + 20
def ben_stickers : ℕ := ryan_stickers - 10
def total_stickers : ℕ := karl_stickers + ryan_stickers + ben_stickers

theorem stickers_total : total_stickers = 105 := by
  sorry

end stickers_total_l401_40171


namespace value_of_N_l401_40192

theorem value_of_N (N : ℕ) (h : (20 / 100) * N = (60 / 100) * 2500) : N = 7500 :=
by {
  sorry
}

end value_of_N_l401_40192


namespace probability_non_first_class_product_l401_40187

theorem probability_non_first_class_product (P_A P_B P_C : ℝ) (hA : P_A = 0.65) (hB : P_B = 0.2) (hC : P_C = 0.1) : 1 - P_A = 0.35 :=
by
  sorry

end probability_non_first_class_product_l401_40187


namespace area_of_quadrilateral_EFGM_l401_40159

noncomputable def area_ABMJ := 1.8 -- Given area of quadrilateral ABMJ

-- Conditions described in a more abstract fashion:
def is_perpendicular (A B C D E F G H I J K L : Point) : Prop :=
  -- Description of each adjacent pairs being perpendicular
  sorry

def is_congruent (A B C D E F G H I J K L : Point) : Prop :=
  -- Description of all sides except AL and GF being congruent
  sorry

def are_segments_intersecting (B G E L : Point) (M : Point) : Prop :=
  -- Description of segments BG and EL intersecting at point M
  sorry

def area_ratio (tri1 tri2 : Finset Triangle) : ℝ :=
  -- Function that returns the ratio of areas covered by the triangles
  sorry

theorem area_of_quadrilateral_EFGM 
  (A B C D E F G H I J K L M : Point)
  (h1 : is_perpendicular A B C D E F G H I J K L)
  (h2 : is_congruent A B C D E F G H I J K L)
  (h3 : are_segments_intersecting B G E L M)
  : 7 / 3 * area_ABMJ = 4.2 :=
by
  -- Proof of the theorem that area EFGM == 4.2 using the conditions
  sorry

end area_of_quadrilateral_EFGM_l401_40159


namespace future_ratio_l401_40106

variable (j e : ℕ)

-- Conditions
axiom condition1 : j - 3 = 4 * (e - 3)
axiom condition2 : j - 5 = 5 * (e - 5)

-- Theorem to be proved
theorem future_ratio : ∃ x : ℕ, x = 1 ∧ ((j + x) / (e + x) = 3) := by
  sorry

end future_ratio_l401_40106


namespace JohnReceivedDiamonds_l401_40137

def InitialDiamonds (Bill Sam : ℕ) (John : ℕ) : Prop :=
  Bill = 12 ∧ Sam = 12

def TheftEvents (BillBefore SamBefore JohnBefore BillAfter SamAfter JohnAfter : ℕ) : Prop :=
  BillAfter = BillBefore - 1 ∧ SamAfter = SamBefore - 1 ∧ JohnAfter = JohnBefore + 1

def AverageMassChange (Bill Sam John : ℕ) (BillMassChange SamMassChange JohnMassChange : ℤ) : Prop :=
  BillMassChange = Bill - 1 ∧ SamMassChange = Sam - 2 ∧ JohnMassChange = John + 4

def JohnInitialDiamonds (John : ℕ) : Prop :=
  Exists (fun x => 4 * x = 36)

theorem JohnReceivedDiamonds : ∃ John : ℕ, 
  InitialDiamonds 12 12 John ∧
  (∃ BillBefore SamBefore JohnBefore BillAfter SamAfter JohnAfter,
      TheftEvents BillBefore SamBefore JohnBefore BillAfter SamAfter JohnAfter ∧
      AverageMassChange 12 12 12 (-12) (-24) 36) →
  John = 9 :=
sorry

end JohnReceivedDiamonds_l401_40137


namespace systematic_sampling_first_group_number_l401_40158

-- Given conditions
def total_students := 160
def group_size := 8
def groups := total_students / group_size
def number_in_16th_group := 126

-- Theorem Statement
theorem systematic_sampling_first_group_number :
  ∃ x : ℕ, (120 + x = number_in_16th_group) ∧ x = 6 :=
by
  -- Proof can be filled here
  sorry

end systematic_sampling_first_group_number_l401_40158


namespace moles_HBr_formed_l401_40152

theorem moles_HBr_formed 
    (moles_CH4 : ℝ) (moles_Br2 : ℝ) (reaction : ℝ) : 
    moles_CH4 = 1 ∧ moles_Br2 = 1 → reaction = 1 :=
by
  intros h
  cases h
  sorry

end moles_HBr_formed_l401_40152


namespace sin_of_tan_l401_40185

theorem sin_of_tan (A : ℝ) (hA_acute : 0 < A ∧ A < π / 2) (h_tan_A : Real.tan A = (Real.sqrt 2) / 3) :
  Real.sin A = (Real.sqrt 22) / 11 :=
sorry

end sin_of_tan_l401_40185


namespace fewer_seats_on_right_side_l401_40164

-- Definitions based on the conditions
def left_seats := 15
def seats_per_seat := 3
def back_seat_capacity := 8
def total_capacity := 89

-- Statement to prove the problem
theorem fewer_seats_on_right_side : left_seats - (total_capacity - back_seat_capacity - (left_seats * seats_per_seat)) / seats_per_seat = 3 := 
by
  -- proof steps go here
  sorry

end fewer_seats_on_right_side_l401_40164


namespace not_square_of_expression_l401_40160

theorem not_square_of_expression (n : ℕ) (h : n > 0) : ∀ k : ℕ, (4 * n^2 + 4 * n + 4 ≠ k^2) :=
by
  sorry

end not_square_of_expression_l401_40160
