import Mathlib

namespace gcd_175_100_65_l2178_217860

theorem gcd_175_100_65 : Nat.gcd (Nat.gcd 175 100) 65 = 5 :=
by
  sorry

end gcd_175_100_65_l2178_217860


namespace percentage_of_boys_l2178_217853

theorem percentage_of_boys (total_students boys girls : ℕ) (h_ratio : boys * 4 = girls * 3) (h_total : boys + girls = total_students) (h_total_students : total_students = 42) : (boys : ℚ) * 100 / total_students = 42.857 :=
by
  sorry

end percentage_of_boys_l2178_217853


namespace find_custom_operation_value_l2178_217836

noncomputable def custom_operation (a b : ℤ) : ℚ := (1 : ℚ)/a + (1 : ℚ)/b

theorem find_custom_operation_value (a b : ℤ) (h1 : a + b = 12) (h2 : a * b = 32) :
  custom_operation a b = 3 / 8 := by
  sorry

end find_custom_operation_value_l2178_217836


namespace expression_evaluation_l2178_217817

theorem expression_evaluation (a b c d : ℝ) 
  (h₁ : a + b = 0) 
  (h₂ : c * d = 1) : 
  (a + b)^2 - 3 * (c * d)^4 = -3 := 
by
  -- Proof steps are omitted, as only the statement is required.
  sorry

end expression_evaluation_l2178_217817


namespace tangent_value_range_l2178_217861

theorem tangent_value_range : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ (π / 4) → 0 ≤ (Real.tan x) ∧ (Real.tan x) ≤ 1) :=
by
  sorry

end tangent_value_range_l2178_217861


namespace heat_more_games_than_bulls_l2178_217863

theorem heat_more_games_than_bulls (H : ℕ) 
(h1 : 70 + H = 145) :
H - 70 = 5 :=
sorry

end heat_more_games_than_bulls_l2178_217863


namespace necessary_and_sufficient_condition_extremum_l2178_217808

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 6 * x^2 + (a - 1) * x - 5

theorem necessary_and_sufficient_condition_extremum (a : ℝ) :
  (∃ x, f a x = 0) ↔ -3 < a ∧ a < 4 :=
sorry

end necessary_and_sufficient_condition_extremum_l2178_217808


namespace find_p_l2178_217881

theorem find_p 
  (a : ℝ) (p : ℕ) 
  (h1 : 12345 * 6789 = a * 10^p)
  (h2 : 1 ≤ a) (h3 : a < 10) (h4 : 0 < p) 
  : p = 7 := 
sorry

end find_p_l2178_217881


namespace total_cupcakes_baked_l2178_217812

theorem total_cupcakes_baked
    (boxes : ℕ)
    (cupcakes_per_box : ℕ)
    (left_at_home : ℕ)
    (total_given_away : ℕ)
    (total_baked : ℕ)
    (h1 : boxes = 17)
    (h2 : cupcakes_per_box = 3)
    (h3 : left_at_home = 2)
    (h4 : total_given_away = boxes * cupcakes_per_box)
    (h5 : total_baked = total_given_away + left_at_home) :
    total_baked = 53 := by
  sorry

end total_cupcakes_baked_l2178_217812


namespace Catriona_goldfish_count_l2178_217876

theorem Catriona_goldfish_count (G : ℕ) (A : ℕ) (U : ℕ) 
    (h1 : A = G + 4) 
    (h2 : U = 2 * A) 
    (h3 : G + A + U = 44) : G = 8 :=
by
  -- Proof goes here
  sorry

end Catriona_goldfish_count_l2178_217876


namespace largest_number_in_box_l2178_217890

theorem largest_number_in_box
  (a : ℕ)
  (sum_eq_480 : a + (a + 1) + (a + 2) + (a + 10) + (a + 11) + (a + 12) = 480) :
  a + 12 = 86 :=
by
  sorry

end largest_number_in_box_l2178_217890


namespace greatest_triangle_perimeter_l2178_217816

theorem greatest_triangle_perimeter :
  ∃ (x : ℕ), 3 < x ∧ x < 6 ∧ max (x + 4 * x + 17) (5 + 4 * 5 + 17) = 42 :=
by
  sorry

end greatest_triangle_perimeter_l2178_217816


namespace mary_hourly_wage_l2178_217806

-- Defining the conditions as given in the problem
def hours_per_day_MWF : ℕ := 9
def hours_per_day_TTh : ℕ := 5
def days_MWF : ℕ := 3
def days_TTh : ℕ := 2
def weekly_earnings : ℕ := 407

-- Total hours worked in a week by Mary
def total_hours_worked : ℕ := (days_MWF * hours_per_day_MWF) + (days_TTh * hours_per_day_TTh)

-- The hourly wage calculation
def hourly_wage : ℕ := weekly_earnings / total_hours_worked

-- The statement to prove
theorem mary_hourly_wage : hourly_wage = 11 := by
  sorry

end mary_hourly_wage_l2178_217806


namespace inequality_proof_l2178_217856

theorem inequality_proof 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (hxyz : x + y + z = 1) : 
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by
  sorry

end inequality_proof_l2178_217856


namespace train_speed_is_correct_l2178_217810

noncomputable def train_length : ℕ := 900
noncomputable def platform_length : ℕ := train_length
noncomputable def time_in_minutes : ℕ := 1
noncomputable def distance_covered : ℕ := train_length + platform_length
noncomputable def speed_m_per_minute : ℕ := distance_covered / time_in_minutes
noncomputable def speed_km_per_hr : ℕ := (speed_m_per_minute * 60) / 1000

theorem train_speed_is_correct :
  speed_km_per_hr = 108 :=
by
  sorry

end train_speed_is_correct_l2178_217810


namespace sum_geometric_terms_l2178_217886

noncomputable def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^n

theorem sum_geometric_terms (a q : ℝ) :
  a * (1 + q) = 3 → a * (1 + q) * q^2 = 6 → 
  a * (1 + q) * q^6 = 24 :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end sum_geometric_terms_l2178_217886


namespace inequality_satisfaction_l2178_217893

theorem inequality_satisfaction (a b : ℝ) (h : a < 0) : (a < b) ∧ (a^2 + b^2 > 2) :=
by
  sorry

end inequality_satisfaction_l2178_217893


namespace proof_problem_l2178_217868

-- Definition for the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions
def probability (b : ℕ) : ℚ :=
  (binom (40 - b) 2 + binom (b - 1) 2 : ℚ) / 1225

def is_coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def minimum_b (b : ℕ) : Prop :=
  b = 11 ∧ probability 11 = 857 / 1225 ∧ is_coprime 857 1225 ∧ 857 + 1225 = 2082

-- Statement to prove
theorem proof_problem : ∃ b, minimum_b b := 
by
  -- Lean statement goes here
  sorry

end proof_problem_l2178_217868


namespace largest_among_a_b_c_d_l2178_217849

noncomputable def a : ℝ := Real.sin (Real.cos (2015 * Real.pi / 180))
noncomputable def b : ℝ := Real.sin (Real.sin (2015 * Real.pi / 180))
noncomputable def c : ℝ := Real.cos (Real.sin (2015 * Real.pi / 180))
noncomputable def d : ℝ := Real.cos (Real.cos (2015 * Real.pi / 180))

theorem largest_among_a_b_c_d : c = max a (max b (max c d)) := by
  sorry

end largest_among_a_b_c_d_l2178_217849


namespace tetrahedron_volume_from_pentagon_l2178_217851

noncomputable def volume_of_tetrahedron (side_length : ℝ) (diagonal_length : ℝ) (base_area : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

theorem tetrahedron_volume_from_pentagon :
  ∀ (s : ℝ), s = 1 →
  volume_of_tetrahedron s ((1 + Real.sqrt 5) / 2) ((Real.sqrt 3) / 4) (Real.sqrt ((5 + 2 * Real.sqrt 5) / 4)) =
  (1 + Real.sqrt 5) / 24 :=
by
  intros s hs
  rw [hs]
  sorry

end tetrahedron_volume_from_pentagon_l2178_217851


namespace equation_has_two_distinct_roots_l2178_217833

def quadratic (a x : ℝ) : ℝ :=
  a^2 * (x - 2) + a * (39 - 20 * x) + 20 

theorem equation_has_two_distinct_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic a x1 = 0 ∧ quadratic a x2 = 0) ↔ a = 20 := 
by
  sorry

end equation_has_two_distinct_roots_l2178_217833


namespace initial_oranges_l2178_217889

theorem initial_oranges (left_oranges taken_oranges : ℕ) (h1 : left_oranges = 25) (h2 : taken_oranges = 35) : 
  left_oranges + taken_oranges = 60 := 
by 
  sorry

end initial_oranges_l2178_217889


namespace sum_of_tripled_numbers_l2178_217800

theorem sum_of_tripled_numbers (a b S : ℤ) (h : a + b = S) : 3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_tripled_numbers_l2178_217800


namespace right_triangle_no_k_values_l2178_217848

theorem right_triangle_no_k_values (k : ℕ) (h : k > 0) : 
  ¬ (∃ k, k > 0 ∧ ((17 > k ∧ 17^2 = 13^2 + k^2) ∨ (k > 17 ∧ k < 30 ∧ k^2 = 13^2 + 17^2))) :=
sorry

end right_triangle_no_k_values_l2178_217848


namespace general_term_arithmetic_sequence_l2178_217883

-- Define an arithmetic sequence with first term a1 and common ratio q
def arithmetic_sequence (a1 : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  a1 * q ^ (n - 1)

-- Theorem: given the conditions, prove that the general term is a1 * q^(n-1)
theorem general_term_arithmetic_sequence (a1 q : ℤ) (n : ℕ) :
  arithmetic_sequence a1 q n = a1 * q ^ (n - 1) :=
by
  sorry

end general_term_arithmetic_sequence_l2178_217883


namespace max_value_range_l2178_217874

theorem max_value_range (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_deriv : ∀ x, f' x = a * (x - 1) * (x - a))
  (h_max : ∀ x, (x = a → (∀ y, f y ≤ f x))) : 0 < a ∧ a < 1 :=
sorry

end max_value_range_l2178_217874


namespace negation_of_exists_l2178_217835

theorem negation_of_exists : (¬ ∃ x_0 : ℝ, x_0 < 0 ∧ x_0^2 > 0) ↔ ∀ x : ℝ, x < 0 → x^2 ≤ 0 :=
sorry

end negation_of_exists_l2178_217835


namespace count_integer_radii_l2178_217873

theorem count_integer_radii (r : ℕ) (h : r < 150) :
  (∃ n : ℕ, n = 11 ∧ (∀ r, 0 < r ∧ r < 150 → (150 % r = 0)) ∧ (r ≠ 150)) := sorry

end count_integer_radii_l2178_217873


namespace bank_card_payment_technology_order_l2178_217803

-- Conditions as definitions
def action_tap := 1
def action_pay_online := 2
def action_swipe := 3
def action_insert_into_terminal := 4

-- Corresponding proof problem statement
theorem bank_card_payment_technology_order :
  [action_insert_into_terminal, action_swipe, action_tap, action_pay_online] = [4, 3, 1, 2] := by
  sorry

end bank_card_payment_technology_order_l2178_217803


namespace age_twice_in_Y_years_l2178_217862

def present_age_of_son : ℕ := 24
def age_difference := 26
def present_age_of_man : ℕ := present_age_of_son + age_difference

theorem age_twice_in_Y_years : 
  ∃ (Y : ℕ), present_age_of_man + Y = 2 * (present_age_of_son + Y) → Y = 2 :=
by
  sorry

end age_twice_in_Y_years_l2178_217862


namespace hannah_spent_65_l2178_217888

-- Definitions based on the conditions
def sweatshirts_count : ℕ := 3
def t_shirts_count : ℕ := 2
def sweatshirt_cost : ℕ := 15
def t_shirt_cost : ℕ := 10

-- The total amount spent
def total_spent : ℕ := sweatshirts_count * sweatshirt_cost + t_shirts_count * t_shirt_cost

-- The theorem stating the problem
theorem hannah_spent_65 : total_spent = 65 :=
by
  sorry

end hannah_spent_65_l2178_217888


namespace evaluate_256_pow_5_div_8_l2178_217830

theorem evaluate_256_pow_5_div_8 (h : 256 = 2^8) : 256^(5/8) = 32 :=
by
  sorry

end evaluate_256_pow_5_div_8_l2178_217830


namespace julia_game_difference_l2178_217854

theorem julia_game_difference :
  let tag_monday := 28
  let hide_seek_monday := 15
  let tag_tuesday := 33
  let hide_seek_tuesday := 21
  let total_monday := tag_monday + hide_seek_monday
  let total_tuesday := tag_tuesday + hide_seek_tuesday
  let difference := total_tuesday - total_monday
  difference = 11 := by
  sorry

end julia_game_difference_l2178_217854


namespace john_additional_tax_l2178_217852

-- Define the old and new tax rates
def old_tax (income : ℕ) : ℕ :=
  if income ≤ 500000 then income * 20 / 100
  else if income ≤ 1000000 then 100000 + (income - 500000) * 25 / 100
  else 225000 + (income - 1000000) * 30 / 100

def new_tax (income : ℕ) : ℕ :=
  if income ≤ 500000 then income * 30 / 100
  else if income ≤ 1000000 then 150000 + (income - 500000) * 35 / 100
  else 325000 + (income - 1000000) * 40 / 100

-- Calculate the tax for rental income after deduction
def rental_income_tax (rental_income : ℕ) : ℕ :=
  let taxable_rental_income := rental_income - rental_income * 10 / 100
  taxable_rental_income * 40 / 100

-- Calculate the tax for investment income
def investment_income_tax (investment_income : ℕ) : ℕ :=
  investment_income * 25 / 100

-- Calculate the tax for self-employment income
def self_employment_income_tax (self_employment_income : ℕ) : ℕ :=
  self_employment_income * 15 / 100

-- Define the total additional tax John pays
def additional_tax_paid (old_main_income new_main_income rental_income investment_income self_employment_income : ℕ) : ℕ :=
  let old_tax_main := old_tax old_main_income
  let new_tax_main := new_tax new_main_income
  let rental_tax := rental_income_tax rental_income
  let investment_tax := investment_income_tax investment_income
  let self_employment_tax := self_employment_income_tax self_employment_income
  (new_tax_main - old_tax_main) + rental_tax + investment_tax + self_employment_tax

-- Prove John pays $352,250 more in taxes under the new system
theorem john_additional_tax (main_income_old main_income_new rental_income investment_income self_employment_income : ℕ) :
  main_income_old = 1000000 →
  main_income_new = 1500000 →
  rental_income = 100000 →
  investment_income = 50000 →
  self_employment_income = 25000 →
  additional_tax_paid main_income_old main_income_new rental_income investment_income self_employment_income = 352250 :=
by
  intros h_old h_new h_rental h_invest h_self
  rw [h_old, h_new, h_rental, h_invest, h_self]
  -- calculation steps are omitted
  sorry

end john_additional_tax_l2178_217852


namespace blaine_fish_caught_l2178_217834

theorem blaine_fish_caught (B : ℕ) (cond1 : B + 2 * B = 15) : B = 5 := by 
  sorry

end blaine_fish_caught_l2178_217834


namespace average_salary_8800_l2178_217807

theorem average_salary_8800 
  (average_salary_start : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_salary : ℝ)
  (avg_specific_months : ℝ)
  (jan_salary_rate : average_salary_start * 4 = total_salary)
  (may_salary_rate : total_salary - salary_jan = total_salary - 3300)
  (final_salary_rate : total_salary - salary_jan + salary_may = 35200)
  (specific_avg_calculation : 35200 / 4 = avg_specific_months)
  : avg_specific_months = 8800 :=
sorry -- Proof steps will be filled in later

end average_salary_8800_l2178_217807


namespace evaluate_g_at_6_l2178_217837

def g (x : ℝ) := 3 * x^4 - 19 * x^3 + 31 * x^2 - 27 * x - 72

theorem evaluate_g_at_6 : g 6 = 666 := by
  sorry

end evaluate_g_at_6_l2178_217837


namespace find_middle_part_value_l2178_217871

-- Define the ratios
def ratio1 := 1 / 2
def ratio2 := 1 / 4
def ratio3 := 1 / 8

-- Total sum
def total_sum := 120

-- Parts proportional to ratios
def part1 (x : ℝ) := x
def part2 (x : ℝ) := ratio1 * x
def part3 (x : ℝ) := ratio2 * x

-- Equation representing the sum of the parts equals to the total sum
def equation (x : ℝ) : Prop :=
  part1 x + part2 x / 2 + part2 x = x * (1 + ratio1 + ratio2)

-- Defining the middle part
def middle_part (x : ℝ) := ratio1 * x

theorem find_middle_part_value :
  ∃ x : ℝ, equation x ∧ middle_part x = 34.2857 := sorry

end find_middle_part_value_l2178_217871


namespace solve_abs_inequality_l2178_217864

theorem solve_abs_inequality (x : ℝ) : 
    (2 ≤ |x - 1| ∧ |x - 1| ≤ 5) ↔ ( -4 ≤ x ∧ x ≤ -1 ∨ 3 ≤ x ∧ x ≤ 6) := 
by
    sorry

end solve_abs_inequality_l2178_217864


namespace number_of_plastic_bottles_l2178_217815

-- Define the weights of glass and plastic bottles
variables (G P : ℕ)

-- Define the number of plastic bottles in the second scenario
variable (x : ℕ)

-- Define the conditions
def condition_1 := 3 * G = 600
def condition_2 := G = P + 150
def condition_3 := 4 * G + x * P = 1050

-- Proof that x is equal to 5 given the conditions
theorem number_of_plastic_bottles (h1 : condition_1 G) (h2 : condition_2 G P) (h3 : condition_3 G P x) : x = 5 :=
sorry

end number_of_plastic_bottles_l2178_217815


namespace couch_cost_l2178_217869

theorem couch_cost
  (C : ℕ)  -- Cost of the couch
  (table_cost : ℕ := 100)
  (lamp_cost : ℕ := 50)
  (amount_paid : ℕ := 500)
  (amount_owed : ℕ := 400)
  (total_furniture_cost : ℕ := C + table_cost + lamp_cost)
  (remaining_amount_owed : total_furniture_cost - amount_paid = amount_owed) :
   C = 750 := 
sorry

end couch_cost_l2178_217869


namespace simplify_polynomial_l2178_217840

theorem simplify_polynomial (y : ℝ) :
    (4 * y^10 + 6 * y^9 + 3 * y^8) + (2 * y^12 + 5 * y^10 + y^9 + y^7 + 4 * y^4 + 7 * y + 9) =
    2 * y^12 + 9 * y^10 + 7 * y^9 + 3 * y^8 + y^7 + 4 * y^4 + 7 * y + 9 := by
  sorry

end simplify_polynomial_l2178_217840


namespace tony_water_intake_l2178_217844

-- Define the constants and conditions
def water_yesterday : ℝ := 48
def percentage_less_yesterday : ℝ := 0.04
def percentage_more_day_before_yesterday : ℝ := 0.05

-- Define the key quantity to find
noncomputable def water_two_days_ago : ℝ := water_yesterday / (1.05 * (1 - percentage_less_yesterday))

-- The proof statement
theorem tony_water_intake :
  water_two_days_ago = 47.62 :=
by
  sorry

end tony_water_intake_l2178_217844


namespace ara_final_height_is_59_l2178_217828

noncomputable def initial_shea_height : ℝ := 51.2
noncomputable def initial_ara_height : ℝ := initial_shea_height + 4
noncomputable def final_shea_height : ℝ := 64
noncomputable def shea_growth : ℝ := final_shea_height - initial_shea_height
noncomputable def ara_growth : ℝ := shea_growth / 3
noncomputable def final_ara_height : ℝ := initial_ara_height + ara_growth

theorem ara_final_height_is_59 :
  final_ara_height = 59 := by
  sorry

end ara_final_height_is_59_l2178_217828


namespace train_length_correct_l2178_217859

noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct 
  (speed_kmh : ℝ := 60) 
  (time_s : ℝ := 9) :
  train_length speed_kmh time_s = 150.03 := by 
  sorry

end train_length_correct_l2178_217859


namespace hypotenuse_of_right_angle_triangle_l2178_217892

theorem hypotenuse_of_right_angle_triangle {a b c : ℕ} (h1 : a^2 + b^2 = c^2) 
  (h2 : a > 0) (h3 : b > 0) 
  (h4 : a + b + c = (a * b) / 2): 
  c = 10 ∨ c = 13 :=
sorry

end hypotenuse_of_right_angle_triangle_l2178_217892


namespace abs_eq_condition_l2178_217896

theorem abs_eq_condition (x : ℝ) (h : |x - 1| + x = 1) : x ≤ 1 :=
by sorry

end abs_eq_condition_l2178_217896


namespace fraction_zero_solution_l2178_217819

theorem fraction_zero_solution (x : ℝ) (h : (|x| - 2) / (x - 2) = 0) : x = -2 :=
sorry

end fraction_zero_solution_l2178_217819


namespace smallest_value_n_l2178_217885

theorem smallest_value_n :
  ∃ (n : ℕ), n * 25 = Nat.lcm (Nat.lcm 10 18) 20 ∧ (∀ m, m * 25 = Nat.lcm (Nat.lcm 10 18) 20 → n ≤ m) := 
sorry

end smallest_value_n_l2178_217885


namespace equilateral_triangle_perimeter_l2178_217846

-- Define the condition of an equilateral triangle where each side is 7 cm
def side_length : ℕ := 7

def is_equilateral_triangle (a b c : ℕ) : Prop :=
  a = b ∧ b = c

-- Define the perimeter function for a triangle
def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

-- Statement to prove
theorem equilateral_triangle_perimeter : is_equilateral_triangle side_length side_length side_length → perimeter side_length side_length side_length = 21 :=
sorry

end equilateral_triangle_perimeter_l2178_217846


namespace toby_money_share_l2178_217801

theorem toby_money_share (initial_money : ℕ) (fraction : ℚ) (brothers : ℕ) (money_per_brother : ℚ)
  (total_shared : ℕ) (remaining_money : ℕ) :
  initial_money = 343 →
  fraction = 1/7 →
  brothers = 2 →
  money_per_brother = fraction * initial_money →
  total_shared = brothers * money_per_brother →
  remaining_money = initial_money - total_shared →
  remaining_money = 245 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end toby_money_share_l2178_217801


namespace divisible_by_3_l2178_217843

theorem divisible_by_3 (n : ℕ) : (n * 2^n + 1) % 3 = 0 ↔ n % 6 = 1 ∨ n % 6 = 2 := 
sorry

end divisible_by_3_l2178_217843


namespace points_opposite_sides_l2178_217827

theorem points_opposite_sides (m : ℝ) : (-2 < m ∧ m < -1) ↔ ((2 - 3 * 1 - m) * (1 - 3 * 1 - m) < 0) := by
  sorry

end points_opposite_sides_l2178_217827


namespace length_of_plot_l2178_217805

theorem length_of_plot (breadth length : ℕ) 
                       (h1 : length = breadth + 26)
                       (fencing_cost total_cost : ℝ)
                       (h2 : fencing_cost = 26.50)
                       (h3 : total_cost = 5300)
                       (perimeter : ℝ) 
                       (h4 : perimeter = 2 * (breadth + length)) 
                       (h5 : total_cost = perimeter * fencing_cost) :
                       length = 63 :=
by
  sorry

end length_of_plot_l2178_217805


namespace intersection_A_B_l2178_217838

def set_A : Set ℝ := {x | x > 0}
def set_B : Set ℝ := {x | x < 4}

theorem intersection_A_B :
  set_A ∩ set_B = {x | 0 < x ∧ x < 4} := sorry

end intersection_A_B_l2178_217838


namespace train_stoppage_time_l2178_217865

theorem train_stoppage_time
  (D : ℝ) -- Distance in kilometers
  (T_no_stop : ℝ := D / 300) -- Time without stoppages in hours
  (T_with_stop : ℝ := D / 200) -- Time with stoppages in hours
  (T_stop : ℝ := T_with_stop - T_no_stop) -- Time lost due to stoppages in hours
  (T_stop_minutes : ℝ := T_stop * 60) -- Time lost due to stoppages in minutes
  (stoppage_per_hour : ℝ := T_stop_minutes / (D / 300)) -- Time stopped per hour of travel
  : stoppage_per_hour = 30 := sorry

end train_stoppage_time_l2178_217865


namespace compute_expression_l2178_217802

theorem compute_expression :
  21 * 47 + 21 * 53 = 2100 := 
by
  sorry

end compute_expression_l2178_217802


namespace other_root_l2178_217850

/-- Given the quadratic equation x^2 - 3x + k = 0 has one root as 1, 
    prove that the other root is 2. -/
theorem other_root (k : ℝ) (h : 1^2 - 3 * 1 + k = 0) : 
  2^2 - 3 * 2 + k = 0 := 
by 
  sorry

end other_root_l2178_217850


namespace hexagon_area_correct_m_plus_n_l2178_217875

noncomputable def hexagon_area (b : ℝ) : ℝ :=
  let A := (0, 0)
  let B := (b, 3)
  let F := (-3 * (3 + b) / 2, 9)  -- derived from complex numbers and angle conversion
  let hexagon_height := 12  -- height difference between the y-coordinates
  let hexagon_base := 3 * (b + 3) / 2  -- distance between parallel lines AB and DE
  36 / 2 * (b + 3) + 6 * (6 + b * Real.sqrt 3)

theorem hexagon_area_correct (b : ℝ) :
  hexagon_area b = 72 * Real.sqrt 3 :=
sorry

theorem m_plus_n : 72 + 3 = 75 := rfl

end hexagon_area_correct_m_plus_n_l2178_217875


namespace find_y_l2178_217898

theorem find_y (n x y : ℝ)
  (h1 : (100 + 200 + n + x) / 4 = 250)
  (h2 : (n + 150 + 100 + x + y) / 5 = 200) :
  y = 50 :=
by
  sorry

end find_y_l2178_217898


namespace bob_total_questions_l2178_217880

theorem bob_total_questions (q1 q2 q3 : ℕ) : 
  q1 = 13 ∧ q2 = 2 * q1 ∧ q3 = 2 * q2 → q1 + q2 + q3 = 91 :=
by
  intros
  sorry

end bob_total_questions_l2178_217880


namespace buns_per_pack_is_eight_l2178_217813

-- Declaring the conditions
def burgers_per_guest : ℕ := 3
def total_friends : ℕ := 10
def friends_no_meat : ℕ := 1
def friends_no_bread : ℕ := 1
def packs_of_buns : ℕ := 3

-- Derived values from the conditions
def effective_friends_for_burgers : ℕ := total_friends - friends_no_meat
def effective_friends_for_buns : ℕ := total_friends - friends_no_bread

-- Final computation to prove
def buns_per_pack : ℕ := 24 / packs_of_buns

-- Theorem statement
theorem buns_per_pack_is_eight : buns_per_pack = 8 := by
  -- use sorry as we are not providing the proof steps 
  sorry

end buns_per_pack_is_eight_l2178_217813


namespace library_books_l2178_217882

/-- Last year, the school library purchased 50 new books. 
    This year, it purchased 3 times as many books. 
    If the library had 100 books before it purchased new books last year,
    prove that the library now has 300 books in total. -/
theorem library_books (initial_books : ℕ) (last_year_books : ℕ) (multiplier : ℕ)
  (h1 : initial_books = 100) (h2 : last_year_books = 50) (h3 : multiplier = 3) :
  initial_books + last_year_books + (multiplier * last_year_books) = 300 := 
sorry

end library_books_l2178_217882


namespace average_age_before_new_students_joined_l2178_217894

/-
Problem: Given that the original strength of the class was 18, 
18 new students with an average age of 32 years joined the class, 
and the average age decreased by 4 years, prove that 
the average age of the class before the new students joined was 40 years.
-/

def original_strength := 18
def new_students := 18
def average_age_new_students := 32
def decrease_in_average_age := 4
def original_average_age := 40

theorem average_age_before_new_students_joined :
  (original_strength * original_average_age + new_students * average_age_new_students) / (original_strength + new_students) = original_average_age - decrease_in_average_age :=
by
  sorry

end average_age_before_new_students_joined_l2178_217894


namespace triangle_perimeter_not_78_l2178_217820

theorem triangle_perimeter_not_78 (x : ℝ) (h1 : 11 < x) (h2 : x < 37) : 13 + 24 + x ≠ 78 :=
by
  -- Using the given conditions to show the perimeter is not 78
  intro h
  have h3 : 48 < 13 + 24 + x := by linarith
  have h4 : 13 + 24 + x < 74 := by linarith
  linarith

end triangle_perimeter_not_78_l2178_217820


namespace sum_product_poly_roots_eq_l2178_217842

theorem sum_product_poly_roots_eq (b c : ℝ) 
  (h1 : -1 + 2 = -b) 
  (h2 : (-1) * 2 = c) : c + b = -3 := 
by 
  sorry

end sum_product_poly_roots_eq_l2178_217842


namespace arithmetic_problem_l2178_217870

theorem arithmetic_problem : (56^2 + 56^2) / 28^2 = 8 := by
  sorry

end arithmetic_problem_l2178_217870


namespace coffee_ounces_per_cup_l2178_217845

theorem coffee_ounces_per_cup
  (persons : ℕ)
  (cups_per_person_per_day : ℕ)
  (cost_per_ounce : ℝ)
  (total_spent_per_week : ℝ)
  (total_cups_per_day : ℕ)
  (total_cups_per_week : ℕ)
  (total_ounces : ℝ)
  (ounces_per_cup : ℝ) :
  persons = 4 →
  cups_per_person_per_day = 2 →
  cost_per_ounce = 1.25 →
  total_spent_per_week = 35 →
  total_cups_per_day = persons * cups_per_person_per_day →
  total_cups_per_week = total_cups_per_day * 7 →
  total_ounces = total_spent_per_week / cost_per_ounce →
  ounces_per_cup = total_ounces / total_cups_per_week →
  ounces_per_cup = 0.5 :=
by
  sorry

end coffee_ounces_per_cup_l2178_217845


namespace fewer_gallons_for_plants_correct_l2178_217831

-- Define the initial conditions
def initial_water : ℕ := 65
def water_per_car : ℕ := 7
def total_cars : ℕ := 2
def water_for_cars : ℕ := water_per_car * total_cars
def water_remaining_after_cars : ℕ := initial_water - water_for_cars
def water_for_plates_clothes : ℕ := 24
def water_remaining_before_plates_clothes : ℕ := water_for_plates_clothes * 2
def water_for_plants : ℕ := water_remaining_after_cars - water_remaining_before_plates_clothes

-- Define the query statement
def fewer_gallons_for_plants : Prop := water_per_car - water_for_plants = 4

-- Proof skeleton
theorem fewer_gallons_for_plants_correct : fewer_gallons_for_plants :=
by sorry

end fewer_gallons_for_plants_correct_l2178_217831


namespace number_of_arrangements_l2178_217884

theorem number_of_arrangements (P : Fin 5 → Type) (youngest : Fin 5) 
  (h_in_not_first_last : ∀ (i : Fin 5), i ≠ 0 → i ≠ 4 → i ≠ youngest) : 
  ∃ n, n = 72 := 
by
  sorry

end number_of_arrangements_l2178_217884


namespace problem1_problem2_l2178_217887

theorem problem1 : -24 - (-15) + (-1) + (-15) = -25 := 
by 
  sorry

theorem problem2 : -27 / (3 / 2) * (2 / 3) = -12 := 
by 
  sorry

end problem1_problem2_l2178_217887


namespace not_possible_to_fill_6x6_with_1x4_l2178_217899

theorem not_possible_to_fill_6x6_with_1x4 :
  ¬ (∃ (a b : ℕ), a + 4 * b = 6 ∧ 4 * a + b = 6) :=
by
  -- Assuming a and b represent the number of 1x4 rectangles aligned horizontally and vertically respectively
  sorry

end not_possible_to_fill_6x6_with_1x4_l2178_217899


namespace probability_both_girls_l2178_217811

def club_probability (total_members girls chosen_members : ℕ) : ℚ :=
  (Nat.choose girls chosen_members : ℚ) / (Nat.choose total_members chosen_members : ℚ)

theorem probability_both_girls (H1 : total_members = 12) (H2 : girls = 7) (H3 : chosen_members = 2) :
  club_probability 12 7 2 = 7 / 22 :=
by {
  sorry
}

end probability_both_girls_l2178_217811


namespace geometric_sequence_four_seven_prod_l2178_217825

def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_four_seven_prod
    (a : ℕ → ℝ)
    (h_geom : is_geometric_sequence a)
    (h_roots : ∀ x, 3 * x^2 - 2 * x - 6 = 0 → (x = a 1 ∨ x = a 10)) :
  a 4 * a 7 = -2 := 
sorry

end geometric_sequence_four_seven_prod_l2178_217825


namespace slope_angle_of_line_l2178_217826

theorem slope_angle_of_line (θ : ℝ) : 
  (∃ m : ℝ, ∀ x y : ℝ, 4 * x + y - 1 = 0 ↔ y = m * x + 1) ∧ (m = -4) → 
  θ = Real.pi - Real.arctan 4 :=
by
  sorry

end slope_angle_of_line_l2178_217826


namespace least_positive_integer_reducible_fraction_l2178_217832

theorem least_positive_integer_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (∃ d : ℕ, d > 1 ∧ d ∣ (m - 10) ∧ d ∣ (9 * m + 11)) ↔ m ≥ n) ∧ n = 111 :=
by
  sorry

end least_positive_integer_reducible_fraction_l2178_217832


namespace evaluate_f_g_f_l2178_217821

-- Define f(x)
def f (x : ℝ) : ℝ := 4 * x + 4

-- Define g(x)
def g (x : ℝ) : ℝ := x^2 + 5 * x + 3

-- State the theorem we're proving
theorem evaluate_f_g_f : f (g (f 3)) = 1360 := by
  sorry

end evaluate_f_g_f_l2178_217821


namespace david_marks_in_english_l2178_217877

variable (E : ℕ)
variable (marks_in_math : ℕ := 98)
variable (marks_in_physics : ℕ := 99)
variable (marks_in_chemistry : ℕ := 100)
variable (marks_in_biology : ℕ := 98)
variable (average_marks : ℚ := 98.2)
variable (num_subjects : ℕ := 5)

theorem david_marks_in_english 
  (H1 : average_marks = (E + marks_in_math + marks_in_physics + marks_in_chemistry + marks_in_biology) / num_subjects) :
  E = 96 :=
sorry

end david_marks_in_english_l2178_217877


namespace investment_period_l2178_217809

theorem investment_period (P A : ℝ) (r n t : ℝ)
  (hP : P = 4000)
  (hA : A = 4840.000000000001)
  (hr : r = 0.10)
  (hn : n = 1)
  (hC : A = P * (1 + r / n) ^ (n * t)) :
  t = 2 := by
-- Adding a sorry to skip the actual proof.
sorry

end investment_period_l2178_217809


namespace sum_single_digit_base_eq_21_imp_b_eq_7_l2178_217839

theorem sum_single_digit_base_eq_21_imp_b_eq_7 (b : ℕ) (h : (b - 1) * b / 2 = 2 * b + 1) : b = 7 :=
sorry

end sum_single_digit_base_eq_21_imp_b_eq_7_l2178_217839


namespace find_length_of_second_train_l2178_217879

def length_of_second_train (L : ℚ) : Prop :=
  let length_first_train : ℚ := 300
  let speed_first_train : ℚ := 120 * 1000 / 3600
  let speed_second_train : ℚ := 80 * 1000 / 3600
  let crossing_time : ℚ := 9
  let relative_speed : ℚ := speed_first_train + speed_second_train
  let total_distance : ℚ := relative_speed * crossing_time
  total_distance = length_first_train + L

theorem find_length_of_second_train :
  ∃ (L : ℚ), length_of_second_train L ∧ L = 199.95 := 
by
  sorry

end find_length_of_second_train_l2178_217879


namespace exists_smallest_positive_period_even_function_l2178_217847

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

noncomputable def functions : List (ℝ → ℝ) :=
  [
    (λ x => Real.sin (2 * x + Real.pi / 2)),
    (λ x => Real.cos (2 * x + Real.pi / 2)),
    (λ x => Real.sin (2 * x) + Real.cos (2 * x)),
    (λ x => Real.sin x + Real.cos x)
  ]

def smallest_positive_period_even_function : ℝ → Prop :=
  λ T => ∃ f ∈ functions, is_even_function f ∧ period f T ∧ T > 0

theorem exists_smallest_positive_period_even_function :
  smallest_positive_period_even_function Real.pi :=
sorry

end exists_smallest_positive_period_even_function_l2178_217847


namespace distribute_places_l2178_217858

open Nat

theorem distribute_places (places schools : ℕ) (h_places : places = 7) (h_schools : schools = 3) : 
  ∃ n : ℕ, n = (Nat.choose (places - 1) (schools - 1)) ∧ n = 15 :=
by
  rw [h_places, h_schools]
  use 15
  , sorry

end distribute_places_l2178_217858


namespace sin_cos_sum_eq_l2178_217891

theorem sin_cos_sum_eq :
  (Real.sin (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) +
   Real.sin (70 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 1 / 2 :=
by 
  sorry

end sin_cos_sum_eq_l2178_217891


namespace f_lg_equality_l2178_217804

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x ^ 2) - 3 * x) + 1

theorem f_lg_equality : f (Real.log 2) + f (Real.log (1 / 2)) = 2 := sorry

end f_lg_equality_l2178_217804


namespace complement_of_intersection_l2178_217822

theorem complement_of_intersection (U M N : Set ℤ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2, 4}) (hN : N = {3, 4, 5}) :
   U \ (M ∩ N) = {1, 2, 3, 5} := by
   sorry

end complement_of_intersection_l2178_217822


namespace tina_coins_after_five_hours_l2178_217895

theorem tina_coins_after_five_hours :
  let coins_in_first_hour := 20
  let coins_in_second_hour := 30
  let coins_in_third_hour := 30
  let coins_in_fourth_hour := 40
  let coins_taken_out_in_fifth_hour := 20
  let total_coins_after_five_hours := coins_in_first_hour + coins_in_second_hour + coins_in_third_hour + coins_in_fourth_hour - coins_taken_out_in_fifth_hour
  total_coins_after_five_hours = 100 :=
by {
  sorry
}

end tina_coins_after_five_hours_l2178_217895


namespace angle_B_in_parallelogram_l2178_217878

theorem angle_B_in_parallelogram (ABCD : Parallelogram) (angle_A angle_C : ℝ) 
  (h : angle_A + angle_C = 100) : 
  angle_B = 130 :=
by
  -- Proof omitted
  sorry

end angle_B_in_parallelogram_l2178_217878


namespace largest_k_inequality_l2178_217866

theorem largest_k_inequality
  (a b c : ℝ)
  (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_pos : (a + b) * (b + c) * (c + a) > 0) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a ≥ 
  (1 / 2) * abs ((a^3 - b^3) / (a + b) + (b^3 - c^3) / (b + c) + (c^3 - a^3) / (c + a)) :=
by
  sorry

end largest_k_inequality_l2178_217866


namespace area_of_rectangle_l2178_217855

theorem area_of_rectangle (M N P Q R S X Y : Type) 
  (PQ : ℝ) (PX XY YQ : ℝ) (R_perpendicular_to_PQ S_perpendicular_to_PQ : Prop) 
  (R_through_M S_through_Q : Prop) 
  (segment_lengths : PQ = PX + XY + YQ) : PQ = 5 ∧ PX = 1 ∧ XY = 2 ∧ YQ = 2 
  → 2 * (1/2 * PQ * 2) = 10 :=
  sorry

end area_of_rectangle_l2178_217855


namespace sin_sum_of_roots_l2178_217872

theorem sin_sum_of_roots (x1 x2 m : ℝ) (hx1 : 0 ≤ x1 ∧ x1 ≤ π) (hx2 : 0 ≤ x2 ∧ x2 ≤ π)
    (hroot1 : 2 * Real.sin x1 + Real.cos x1 = m) (hroot2 : 2 * Real.sin x2 + Real.cos x2 = m) :
    Real.sin (x1 + x2) = 4 / 5 := 
sorry

end sin_sum_of_roots_l2178_217872


namespace plane_equation_passing_through_point_and_parallel_l2178_217841

-- Define the point and the plane parameters
def point : ℝ × ℝ × ℝ := (2, 3, 1)
def normal_vector : ℝ × ℝ × ℝ := (2, -1, 3)
def plane (A B C D : ℝ) (x y z : ℝ) : Prop := A * x + B * y + C * z + D = 0

-- Main theorem statement
theorem plane_equation_passing_through_point_and_parallel :
  ∃ D : ℝ, plane 2 (-1) 3 D 2 3 1 ∧ plane 2 (-1) 3 D 0 0 0 :=
sorry

end plane_equation_passing_through_point_and_parallel_l2178_217841


namespace boy_real_name_is_kolya_l2178_217814

variable (days_answers : Fin 6 → String)
variable (lies_on : Fin 6 → Bool)
variable (truth_days : List (Fin 6))

-- Define the conditions
def condition_truth_days : List (Fin 6) := [0, 1] -- Suppose Thursday is 0, Friday is 1.
def condition_lies_on (d : Fin 6) : Bool := d = 2 -- Suppose Tuesday is 2.

-- The sequence of answers
def condition_days_answers : Fin 6 → String := 
  fun d => match d with
    | 0 => "Kolya"
    | 1 => "Petya"
    | 2 => "Kolya"
    | 3 => "Petya"
    | 4 => "Vasya"
    | 5 => "Petya"
    | _ => "Unknown"

-- The proof problem statement
theorem boy_real_name_is_kolya : 
  ∀ (d : Fin 6), 
  (d ∈ condition_truth_days → condition_days_answers d = "Kolya") ∧
  (condition_lies_on d → condition_days_answers d ≠ "Vasya") ∧ 
  (¬(d ∈ condition_truth_days ∨ condition_lies_on d) → True) →
  "Kolya" = "Kolya" :=
by
  sorry

end boy_real_name_is_kolya_l2178_217814


namespace fourth_intersection_point_exists_l2178_217867

noncomputable def find_fourth_intersection_point : Prop :=
  let points := [(4, 1/2), (-6, -1/3), (1/4, 8), (-2/3, -3)]
  ∃ (h k r : ℝ), 
  ∀ (x y : ℝ), (x, y) ∈ points → (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2

theorem fourth_intersection_point_exists :
  find_fourth_intersection_point :=
by
  sorry

end fourth_intersection_point_exists_l2178_217867


namespace ending_number_of_SetB_l2178_217829

-- Definition of Set A
def SetA : Set ℕ := {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

-- Definition of Set B
def SetB_ends_at (n : ℕ) : Set ℕ := {i | 6 ≤ i ∧ i ≤ n}

-- The main theorem statement
theorem ending_number_of_SetB : ∃ n, SetA ∩ SetB_ends_at n = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15} ∧ 10 ∈ SetA ∩ SetB_ends_at n := 
sorry

end ending_number_of_SetB_l2178_217829


namespace fourth_power_mod_7_is_0_l2178_217823

def fourth_smallest_prime := 7
def square_of_fourth_smallest_prime := fourth_smallest_prime ^ 2
def fourth_power_of_square := square_of_fourth_smallest_prime ^ 4

theorem fourth_power_mod_7_is_0 : 
  (fourth_power_of_square % 7) = 0 :=
by sorry

end fourth_power_mod_7_is_0_l2178_217823


namespace employee_percentage_six_years_or_more_l2178_217818

theorem employee_percentage_six_years_or_more
  (x : ℕ)
  (total_employees : ℕ := 36 * x)
  (employees_6_or_more : ℕ := 8 * x) :
  (employees_6_or_more : ℚ) / (total_employees : ℚ) * 100 = 22.22 := 
sorry

end employee_percentage_six_years_or_more_l2178_217818


namespace inequality_proof_l2178_217897

theorem inequality_proof (x y : ℝ) (h : 2 * y + 5 * x = 10) : (3 * x * y - x^2 - y^2 < 7) :=
sorry

end inequality_proof_l2178_217897


namespace tiffany_math_homework_pages_l2178_217857

def math_problems (m : ℕ) : ℕ := 3 * m
def reading_problems : ℕ := 4 * 3
def total_problems (m : ℕ) : ℕ := math_problems m + reading_problems

theorem tiffany_math_homework_pages (m : ℕ) (h : total_problems m = 30) : m = 6 :=
by
  sorry

end tiffany_math_homework_pages_l2178_217857


namespace people_got_off_at_first_stop_l2178_217824

theorem people_got_off_at_first_stop 
  (X : ℕ)
  (h1 : 50 - X - 6 - 1 = 28) :
  X = 15 :=
by
  sorry

end people_got_off_at_first_stop_l2178_217824
