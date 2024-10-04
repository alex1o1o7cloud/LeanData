import Mathlib

namespace tan_sub_pi_div_four_eq_neg_seven_f_range_l39_39220

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 4)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

-- Proof for the first part
theorem tan_sub_pi_div_four_eq_neg_seven (x : ℝ) (h : 3 / 4 * Real.cos x + Real.sin x = 0) :
  Real.tan (x - Real.pi / 4) = -7 := sorry

noncomputable def f (x : ℝ) : ℝ := 
  2 * ((a x).fst + (b x).fst) * (b x).fst + 2 * ((a x).snd + (b x).snd) * (b x).snd

-- Proof for the second part
theorem f_range (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  1 / 2 < f x ∧ f x < 3 / 2 + Real.sqrt 2 := sorry

end tan_sub_pi_div_four_eq_neg_seven_f_range_l39_39220


namespace sum_of_first_15_odd_integers_l39_39304

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l39_39304


namespace arithmetic_mean_of_numbers_l39_39858

theorem arithmetic_mean_of_numbers (n : ℕ) (h : n > 1) :
  let one_special_number := (1 / n) + (2 / n ^ 2)
  let other_numbers := (n - 1) * 1
  (other_numbers + one_special_number) / n = 1 + 2 / n ^ 2 :=
by
  sorry

end arithmetic_mean_of_numbers_l39_39858


namespace parabola_ratio_l39_39925

-- Define the conditions and question as a theorem statement
theorem parabola_ratio
  (V₁ V₃ : ℝ × ℝ)
  (F₁ F₃ : ℝ × ℝ)
  (hV₁ : V₁ = (0, 0))
  (hF₁ : F₁ = (0, 1/8))
  (hV₃ : V₃ = (0, -1/2))
  (hF₃ : F₃ = (0, -1/4)) :
  dist F₁ F₃ / dist V₁ V₃ = 3 / 4 :=
  by
  sorry

end parabola_ratio_l39_39925


namespace germination_probability_l39_39800

noncomputable def binomial_dist (n : ℕ) (p : ℝ) : pmf ℕ := sorry

theorem germination_probability (n : ℕ) (p : ℝ) (a b : ℕ) (μ σ² : ℝ) (ε : ℝ) :
  n = 1000 →
  p = 0.75 →
  a = 700 →
  b = 800 →
  μ = n * p →
  σ² = n * p * (1 - p) →
  ε = 50 →
  a ≤ μ - ε →
  b ≥ μ + ε →
  Pr (X ∈ Ioc a b) ≥ 0.925 :=
sorry

end germination_probability_l39_39800


namespace socks_total_l39_39583

def socks_lisa (initial: Nat) := initial + 0

def socks_sandra := 20

def socks_cousin (sandra: Nat) := sandra / 5

def socks_mom (initial: Nat) := 8 + 3 * initial

theorem socks_total (initial: Nat) (sandra: Nat) :
  initial = 12 → sandra = 20 → 
  socks_lisa initial + socks_sandra + socks_cousin sandra + socks_mom initial = 80 :=
by
  intros h_initial h_sandra
  rw [h_initial, h_sandra]
  sorry

end socks_total_l39_39583


namespace krakozyabrs_count_l39_39416

theorem krakozyabrs_count :
  ∀ (K : Type) [fintype K] (has_horns : K → Prop) (has_wings : K → Prop), 
  (∀ k, has_horns k ∨ has_wings k) →
  (∃ n, (∀ k, has_horns k → has_wings k) → fintype.card ({k | has_horns k} : set K) = 5 * n) →
  (∃ n, (∀ k, has_wings k → has_horns k) → fintype.card ({k | has_wings k} : set K) = 4 * n) →
  (∃ T, 25 < T ∧ T < 35 ∧ T = 32) := 
by
  intros K _ has_horns has_wings _ _ _
  sorry

end krakozyabrs_count_l39_39416


namespace problem1_l39_39347

theorem problem1 (x : ℝ) : (2 * x - 1) * (2 * x - 3) - (1 - 2 * x) * (2 - x) = 2 * x^2 - 3 * x + 1 :=
by
  sorry

end problem1_l39_39347


namespace triangle_perimeter_is_correct_l39_39380

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (S : ℝ)

def triangle_perimeter (a b c : ℝ) := a + b + c

theorem triangle_perimeter_is_correct :
  c = sqrt 7 → C = π / 3 → S = 3 * sqrt 3 / 2 →
  S = (1 / 2) * a * b * sin (C) → c^2 = a^2 + b^2 - 2 * a * b * cos (C) →
  ∃ a b : ℝ, triangle_perimeter a b c = 5 + sqrt 7 :=
  by
    intros h1 h2 h3 h4 h5
    sorry

end triangle_perimeter_is_correct_l39_39380


namespace sum_of_first_15_odd_positives_l39_39300

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l39_39300


namespace determine_speeds_l39_39499

structure Particle :=
  (speed : ℝ)

def distance : ℝ := 3.01 -- meters

def initial_distance (m1_speed : ℝ) : ℝ :=
  301 - 11 * m1_speed -- converted to cm

theorem determine_speeds :
  ∃ (m1 m2 : Particle), 
  m1.speed = 11 ∧ m2.speed = 7 ∧ 
  ∀ t : ℝ, (t = 10 ∨ t = 45) →
  (initial_distance m1.speed) = t * (m1.speed + m2.speed) ∧
  20 * m2.speed = 35 * (m1.speed - m2.speed) :=
by {
  sorry 
}

end determine_speeds_l39_39499


namespace lisa_socks_total_l39_39584

def total_socks (initial : ℕ) (sandra : ℕ) (cousin_ratio : ℕ → ℕ) (mom_extra : ℕ → ℕ) : ℕ :=
  initial + sandra + cousin_ratio sandra + mom_extra initial

def cousin_ratio (sandra : ℕ) : ℕ := sandra / 5
def mom_extra (initial : ℕ) : ℕ := 3 * initial + 8

theorem lisa_socks_total :
  total_socks 12 20 cousin_ratio mom_extra = 80 := by
  sorry

end lisa_socks_total_l39_39584


namespace find_y_values_l39_39248

open Real

-- Problem statement as a Lean statement.
theorem find_y_values (x : ℝ) (hx : x^2 + 2 * (x / (x - 1)) ^ 2 = 20) :
  ∃ y : ℝ, (y = ((x - 1) ^ 3 * (x + 2)) / (2 * x - 1)) ∧ (y = 14 ∨ y = -56 / 3) := 
sorry

end find_y_values_l39_39248


namespace remainder_a83_l39_39431

def a_n (n : ℕ) : ℕ := 6^n + 8^n

theorem remainder_a83 (n : ℕ) : 
  a_n 83 % 49 = 35 := sorry

end remainder_a83_l39_39431


namespace total_number_of_fish_l39_39624

theorem total_number_of_fish
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (blue_spotted_fish : ℕ)
  (h1 : blue_fish = total_fish / 3)
  (h2 : blue_spotted_fish = blue_fish / 2)
  (h3 : blue_spotted_fish = 10) :
  total_fish = 60 :=
by
  sorry

end total_number_of_fish_l39_39624


namespace min_time_one_ball_l39_39460

noncomputable def children_circle_min_time (n : ℕ) := 98

theorem min_time_one_ball (n : ℕ) (h1 : n = 99) : 
  children_circle_min_time n = 98 := 
by 
  sorry

end min_time_one_ball_l39_39460


namespace harold_savings_l39_39741

theorem harold_savings :
  let income_primary := 2500
  let income_freelance := 500
  let rent := 700
  let car_payment := 300
  let car_insurance := 125
  let electricity := 0.25 * car_payment
  let water := 0.15 * rent
  let internet := 75
  let groceries := 200
  let miscellaneous := 150
  let total_income := income_primary + income_freelance
  let total_expenses := rent + car_payment + car_insurance + electricity + water + internet + groceries + miscellaneous
  let amount_before_savings := total_income - total_expenses
  let retirement := (1/3) * amount_before_savings
  let emergency := (1/3) * amount_before_savings
  let amount_after_savings := amount_before_savings - retirement - emergency
  amount_after_savings = 423.34 := 
sorry

end harold_savings_l39_39741


namespace unique_solution_l39_39043

noncomputable def f (x : ℝ) : ℝ := 2^x + 3^x + 6^x

theorem unique_solution : ∀ x : ℝ, f x = 7^x ↔ x = 2 :=
by
  sorry

end unique_solution_l39_39043


namespace solve_quadratic_l39_39793

theorem solve_quadratic : ∀ (x : ℝ), x * (x + 1) = 2014 * 2015 ↔ (x = 2014 ∨ x = -2015) := by
  sorry

end solve_quadratic_l39_39793


namespace solution_to_system_l39_39940

theorem solution_to_system : ∃ x y : ℤ, (2 * x + 3 * y = -11 ∧ 6 * x - 5 * y = 9) ↔ (x = -1 ∧ y = -3) :=
by
  sorry

end solution_to_system_l39_39940


namespace geometric_sum_4500_l39_39618

theorem geometric_sum_4500 (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300)
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 :=
sorry

end geometric_sum_4500_l39_39618


namespace find_a_b_l39_39385

noncomputable def parabola_props (a b : ℝ) : Prop :=
a ≠ 0 ∧ 
∀ x : ℝ, a * x^2 + b * x - 4 = (1 / 2) * x^2 + x - 4

theorem find_a_b {a b : ℝ} (h1 : parabola_props a b) : 
a = 1 / 2 ∧ b = -1 :=
sorry

end find_a_b_l39_39385


namespace max_gcd_is_one_l39_39458

-- Defining the sequence a_n
def a_n (n : ℕ) : ℕ := 101 + n^3

-- Defining the gcd function for a_n and a_(n+1)
def d_n (n : ℕ) : ℕ := Nat.gcd (a_n n) (a_n (n + 1))

-- The theorem stating the maximum value of d_n is 1
theorem max_gcd_is_one : ∀ n : ℕ, d_n n = 1 := by
  -- Proof is omitted as per instructions
  sorry

end max_gcd_is_one_l39_39458


namespace complex_number_is_3i_quadratic_equation_roots_l39_39206

open Complex

-- Given complex number z satisfies 2z + |z| = 3 + 6i
-- We need to prove that z = 3i
theorem complex_number_is_3i (z : ℂ) (h : 2 * z + abs z = 3 + 6 * I) : z = 3 * I :=
sorry

-- Given that z = 3i is a root of the quadratic equation with real coefficients
-- Prove that b - c = -9
theorem quadratic_equation_roots (b c : ℝ) (h1 : 3 * I + -3 * I = -b)
  (h2 : 3 * I * -3 * I = c) : b - c = -9 :=
sorry

end complex_number_is_3i_quadratic_equation_roots_l39_39206


namespace six_div_one_minus_three_div_ten_equals_twenty_four_l39_39936

theorem six_div_one_minus_three_div_ten_equals_twenty_four :
  (6 : ℤ) / (1 - (3 : ℤ) / (10 : ℤ)) = 24 := 
by
  sorry

end six_div_one_minus_three_div_ten_equals_twenty_four_l39_39936


namespace graveling_cost_is_969_l39_39178

-- Definitions for lawn dimensions
def lawn_length : ℝ := 75
def lawn_breadth : ℝ := 45

-- Definitions for road widths and costs
def road1_width : ℝ := 6
def road1_cost_per_sq_meter : ℝ := 0.90

def road2_width : ℝ := 5
def road2_cost_per_sq_meter : ℝ := 0.85

def road3_width : ℝ := 4
def road3_cost_per_sq_meter : ℝ := 0.80

def road4_width : ℝ := 3
def road4_cost_per_sq_meter : ℝ := 0.75

-- Calculate the area of each road
def road1_area : ℝ := road1_width * lawn_length
def road2_area : ℝ := road2_width * lawn_length
def road3_area : ℝ := road3_width * lawn_breadth
def road4_area : ℝ := road4_width * lawn_breadth

-- Calculate the cost of graveling each road
def road1_graveling_cost : ℝ := road1_area * road1_cost_per_sq_meter
def road2_graveling_cost : ℝ := road2_area * road2_cost_per_sq_meter
def road3_graveling_cost : ℝ := road3_area * road3_cost_per_sq_meter
def road4_graveling_cost : ℝ := road4_area * road4_cost_per_sq_meter

-- Calculate the total cost
def total_graveling_cost : ℝ := 
  road1_graveling_cost + road2_graveling_cost + road3_graveling_cost + road4_graveling_cost

-- Statement to be proved
theorem graveling_cost_is_969 : total_graveling_cost = 969 := by
  sorry

end graveling_cost_is_969_l39_39178


namespace hyperbola_focal_point_k_l39_39523

theorem hyperbola_focal_point_k (k : ℝ) :
  (∃ (c : ℝ), c = 2 ∧ (5 : ℝ) * 2 ^ 2 - k * 0 ^ 2 = 5) →
  k = (5 : ℝ) / 3 :=
by
  sorry

end hyperbola_focal_point_k_l39_39523


namespace sum_of_first_15_odd_positives_l39_39301

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l39_39301


namespace profit_percentage_l39_39316

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 60) (h_selling : selling_price = 78) :
  ((selling_price - cost_price) / cost_price) * 100 = 30 :=
by
  sorry

end profit_percentage_l39_39316


namespace compare_squares_l39_39535

theorem compare_squares (a b c : ℝ) : a^2 + b^2 + c^2 ≥ ab + bc + ca :=
by
  sorry

end compare_squares_l39_39535


namespace shaded_area_l39_39490

/--
Given a larger square containing a smaller square entirely within it,
where the side length of the smaller square is 5 units
and the side length of the larger square is 10 units,
prove that the area of the shaded region (the area of the larger square minus the area of the smaller square) is 75 square units.
-/
theorem shaded_area :
  let side_length_smaller := 5
  let side_length_larger := 10
  let area_larger := side_length_larger * side_length_larger
  let area_smaller := side_length_smaller * side_length_smaller
  area_larger - area_smaller = 75 := 
by
  let side_length_smaller := 5
  let side_length_larger := 10
  let area_larger := side_length_larger * side_length_larger
  let area_smaller := side_length_smaller * side_length_smaller
  sorry

end shaded_area_l39_39490


namespace probability_of_at_least_one_boy_and_one_girl_is_correct_l39_39337

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  (1 - ((1/2)^4 + (1/2)^4))

theorem probability_of_at_least_one_boy_and_one_girl_is_correct : 
  probability_at_least_one_boy_and_one_girl = 7/8 :=
by
  sorry

end probability_of_at_least_one_boy_and_one_girl_is_correct_l39_39337


namespace problem_statement_l39_39354

noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * Real.sqrt x

theorem problem_statement : 3 * g 3 - g 9 = -48 - 6 * Real.sqrt 3 := by
  sorry

end problem_statement_l39_39354


namespace john_additional_tax_l39_39404

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

end john_additional_tax_l39_39404


namespace lincoln_high_school_students_l39_39188

theorem lincoln_high_school_students (total students_in_either_or_both_clubs students_in_photography students_in_science : ℕ)
  (h1 : total = 300)
  (h2 : students_in_photography = 120)
  (h3 : students_in_science = 140)
  (h4 : students_in_either_or_both_clubs = 220):
  ∃ x, x = 40 ∧ (students_in_photography + students_in_science - students_in_either_or_both_clubs = x) := 
by
  use 40
  sorry

end lincoln_high_school_students_l39_39188


namespace FGH_supermarkets_total_l39_39958

theorem FGH_supermarkets_total (US Canada : ℕ) 
  (h1 : US = 49) 
  (h2 : US = Canada + 14) : 
  US + Canada = 84 := 
by 
  sorry

end FGH_supermarkets_total_l39_39958


namespace weight_of_smallest_box_l39_39459

variables (M S L : ℕ)

theorem weight_of_smallest_box
  (h1 : M + S = 83)
  (h2 : L + S = 85)
  (h3 : L + M = 86) :
  S = 41 :=
sorry

end weight_of_smallest_box_l39_39459


namespace cameron_list_count_l39_39024

theorem cameron_list_count : 
  (∃ (n m : ℕ), n = 900 ∧ m = 27000 ∧ (∀ k : ℕ, (30 * k) ≥ n ∧ (30 * k) ≤ m → ∃ count : ℕ, count = 871)) :=
by
  sorry

end cameron_list_count_l39_39024


namespace keychain_arrangement_l39_39233

open Function

theorem keychain_arrangement (keys : Finset ℕ) (h : keys.card = 7)
  (house_key car_key office_key : ℕ) (hmem : house_key ∈ keys)
  (cmem : car_key ∈ keys) (omem : office_key ∈ keys) : 
  ∃ n : ℕ, n = 72 :=
by
  sorry

end keychain_arrangement_l39_39233


namespace sum_x1_x2_l39_39265

open ProbabilityTheory

variable {Ω : Type*} {X : Ω → ℝ}
variable (p1 p2 : ℝ) (x1 x2 : ℝ)
variable (h1 : 2/3 * x1 + 1/3 * x2 = 4/9)
variable (h2 : 2/3 * (x1 - 4/9)^2 + 1/3 * (x2 - 4/9)^2 = 2)
variable (h3 : x1 < x2)

theorem sum_x1_x2 : x1 + x2 = 17/9 :=
by
  sorry

end sum_x1_x2_l39_39265


namespace trig_identity_l39_39880

theorem trig_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 :=
by
  sorry

end trig_identity_l39_39880


namespace b_investment_l39_39494

theorem b_investment (a_investment : ℝ) (c_investment : ℝ) (total_profit : ℝ) (a_share_profit : ℝ) (b_investment : ℝ) : a_investment = 6300 → c_investment = 10500 → total_profit = 14200 → a_share_profit = 4260 → b_investment = 4220 :=
by
  intro h_a h_c h_total h_a_share
  have h1 : 6300 / (6300 + 4220 + 10500) = 4260 / 14200 := sorry
  have h2 : 6300 * 14200 = 4260 * (6300 + 4220 + 10500) := sorry
  have h3 : b_investment = 4220 := sorry
  exact h3

end b_investment_l39_39494


namespace inequality_solution_l39_39261

theorem inequality_solution (x : ℝ) : 
    (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
    sorry

end inequality_solution_l39_39261


namespace price_difference_l39_39107

noncomputable def original_price (P : ℝ) : Prop :=
  0.80 * P + 4000 = 30000

theorem price_difference (P : ℝ) (h : original_price P) : P - 30000 = 2500 := by
  unfold original_price at h
  linarith

#check price_difference -- to ensure that the theorem is correct

end price_difference_l39_39107


namespace maintain_income_with_new_demand_l39_39334

variable (P D : ℝ) -- Original Price and Demand
def new_price := 1.20 * P -- New Price after 20% increase
def new_demand := 1.12 * D -- New Demand after 12% increase due to advertisement
def original_income := P * D -- Original income
def new_income := new_price * new_demand -- New income after changes

theorem maintain_income_with_new_demand :
  ∀ P D : ℝ, P * D = 1.20 * P * 1.12 * (D_new : ℝ) → (D_new = 14/15 * D) :=
by
  intro P D h
  sorry

end maintain_income_with_new_demand_l39_39334


namespace counting_integers_between_multiples_l39_39032

theorem counting_integers_between_multiples :
  let smallest_perfect_square_multiple := 900 in
  let smallest_perfect_cube_multiple := 27000 in
  let num_integers := (smallest_perfect_cube_multiple / 30) - (smallest_perfect_square_multiple / 30) + 1 in
  smallest_perfect_square_multiple = 30 * 30 ∧ 
  smallest_perfect_cube_multiple = 900 * 30 ∧ 
  num_integers = 871 :=
by
  sorry

end counting_integers_between_multiples_l39_39032


namespace least_positive_integer_with_12_factors_l39_39979

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l39_39979


namespace crystal_run_final_segment_length_l39_39042

theorem crystal_run_final_segment_length :
  let north_distance := 2
  let southeast_leg := 1 / Real.sqrt 2
  let southeast_movement_north := -southeast_leg
  let southeast_movement_east := southeast_leg
  let northeast_leg := 2 / Real.sqrt 2
  let northeast_movement_north := northeast_leg
  let northeast_movement_east := northeast_leg
  let total_north_movement := north_distance + northeast_movement_north + southeast_movement_north
  let total_east_movement := southeast_movement_east + northeast_movement_east
  total_north_movement = 2.5 ∧ 
  total_east_movement = 3 * Real.sqrt 2 / 2 ∧ 
  Real.sqrt (total_north_movement^2 + total_east_movement^2) = Real.sqrt 10.75 :=
by
  sorry

end crystal_run_final_segment_length_l39_39042


namespace largest_is_D_l39_39461

-- Definitions based on conditions
def A : ℕ := 27
def B : ℕ := A + 7
def C : ℕ := B - 9
def D : ℕ := 2 * C

-- Theorem stating D is the largest
theorem largest_is_D : D = max (max A B) (max C D) :=
by
  -- Inserting sorry because the proof is not required.
  sorry

end largest_is_D_l39_39461


namespace dasha_strip_dimensions_l39_39825

theorem dasha_strip_dimensions (a b c : ℕ) (h1 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) : 
  (a = 1 ∧ (b + c = 22)) ∨ (a = 22 ∧ (b + c = 1)) :=
by sorry

end dasha_strip_dimensions_l39_39825


namespace evaluate_expression_l39_39432

theorem evaluate_expression (x z : ℝ) (h1 : x ≠ 0) (h2 : z ≠ 0) (y : ℝ) (h3 : y = 1 / x + z) : 
    (x - 1 / x) * (y + 1 / y) = (x^2 - 1) * (1 + 2 * x * z + x^2 * z^2 + x^2) / (x^2 * (1 + x * z)) := by
  sorry

end evaluate_expression_l39_39432


namespace certain_number_is_310_l39_39000

theorem certain_number_is_310 (x : ℤ) (h : 3005 - x + 10 = 2705) : x = 310 :=
by
  sorry

end certain_number_is_310_l39_39000


namespace max_unique_sundaes_l39_39849

theorem max_unique_sundaes (n : ℕ) (h : n = 8) : 
  (n + (n.choose 2) = 36) :=
by
  rw h
  simp [Nat.choose]
  sorry

end max_unique_sundaes_l39_39849


namespace smallest_five_digit_number_divisible_l39_39703

def smallest_prime_divisible (n: ℕ) : Prop :=
  ∃ k: ℕ, n = 2310 * k ∧ 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_number_divisible :
  ∃ (n: ℕ), smallest_prime_divisible n ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_l39_39703


namespace second_athlete_high_jump_eq_eight_l39_39465

theorem second_athlete_high_jump_eq_eight :
  let first_athlete_long_jump := 26
  let first_athlete_triple_jump := 30
  let first_athlete_high_jump := 7
  let second_athlete_long_jump := 24
  let second_athlete_triple_jump := 34
  let winner_average_jump := 22
  (first_athlete_long_jump + first_athlete_triple_jump + first_athlete_high_jump) / 3 < winner_average_jump →
  ∃ (second_athlete_high_jump : ℝ), 
    second_athlete_high_jump = 
    (winner_average_jump * 3 - (second_athlete_long_jump + second_athlete_triple_jump)) ∧ 
    second_athlete_high_jump = 8 :=
by
  intros 
  sorry

end second_athlete_high_jump_eq_eight_l39_39465


namespace solve_inequality_l39_39259

theorem solve_inequality (x : ℝ) : (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
  sorry

end solve_inequality_l39_39259


namespace equal_angles_PAC_QAB_l39_39112

universe u
variables {α : Type u} [EuclideanGeometry α]

theorem equal_angles_PAC_QAB (A B C A1 B1 C1 : α) (X P Q : α)
(hX_on_bisector : IsOnBisector A A1 X)
(hBX_B1 : LineThrough B X ∩ LineThrough A C = {B1})
(hCX_C1 : LineThrough C X ∩ LineThrough A B = {C1})
(hA1B1_CC1_P : LineSegment A1 B1 ∩ LineSegment C C1 = {P})
(hA1C1_BB1_Q : LineSegment A1 C1 ∩ LineSegment B B1 = {Q}) :
Angle A P C = Angle Q A B :=
sorry

end equal_angles_PAC_QAB_l39_39112


namespace rectangle_perimeter_l39_39065

theorem rectangle_perimeter (t s : ℝ) (h : t ≥ s) : 2 * (t - s) + 2 * s = 2 * t := 
by 
  sorry

end rectangle_perimeter_l39_39065


namespace molecular_weight_K2Cr2O7_l39_39286

/--
K2Cr2O7 consists of:
- 2 K atoms
- 2 Cr atoms
- 7 O atoms

Atomic weights:
- K: 39.10 g/mol
- Cr: 52.00 g/mol
- O: 16.00 g/mol

We need to prove that the molecular weight of 4 moles of K2Cr2O7 is 1176.80 g/mol.
-/
theorem molecular_weight_K2Cr2O7 :
  let weight_K := 39.10
  let weight_Cr := 52.00
  let weight_O := 16.00
  let mol_weight_K2Cr2O7 := (2 * weight_K) + (2 * weight_Cr) + (7 * weight_O)
  (4 * mol_weight_K2Cr2O7) = 1176.80 :=
by
  sorry

end molecular_weight_K2Cr2O7_l39_39286


namespace solve_xyz_l39_39258

def is_solution (x y z : ℕ) : Prop :=
  x * y + y * z + z * x = 2 * (x + y + z)

theorem solve_xyz (x y z : ℕ) :
  is_solution x y z ↔ (x = 1 ∧ y = 2 ∧ z = 4) ∨
                     (x = 1 ∧ y = 4 ∧ z = 2) ∨
                     (x = 2 ∧ y = 1 ∧ z = 4) ∨
                     (x = 2 ∧ y = 4 ∧ z = 1) ∨
                     (x = 2 ∧ y = 2 ∧ z = 2) ∨
                     (x = 4 ∧ y = 1 ∧ z = 2) ∨
                     (x = 4 ∧ y = 2 ∧ z = 1) := sorry

end solve_xyz_l39_39258


namespace constant_value_l39_39651

theorem constant_value (x y z C : ℤ) (h1 : x = z + 2) (h2 : y = z + 1) (h3 : x > y) (h4 : y > z) (h5 : z = 2) (h6 : 2 * x + 3 * y + 3 * z = 5 * y + C) : C = 8 :=
by
  sorry

end constant_value_l39_39651


namespace sum_first_15_odd_integers_l39_39298

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l39_39298


namespace difference_of_numbers_l39_39278

variable (x y : ℝ)

theorem difference_of_numbers (h1 : x + y = 10) (h2 : x - y = 19) (h3 : x^2 - y^2 = 190) : x - y = 19 :=
by sorry

end difference_of_numbers_l39_39278


namespace smallest_five_digit_number_divisible_by_first_five_primes_l39_39698

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l39_39698


namespace contrapositive_statement_l39_39943

theorem contrapositive_statement (x y : ℤ) : ¬ (x + y) % 2 = 1 → ¬ (x % 2 = 1 ∧ y % 2 = 1) :=
sorry

end contrapositive_statement_l39_39943


namespace sequence_general_term_l39_39234

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = 3 * a n - 2 * n ^ 2 + 4 * n + 4) :
  ∀ n, a n = 3^n + n^2 - n - 2 :=
sorry

end sequence_general_term_l39_39234


namespace triangle_area_ratio_l39_39158

noncomputable def area_ratio (a b c d e f : ℕ) : ℚ :=
  (a * b) / (d * e)

theorem triangle_area_ratio : area_ratio 6 8 10 9 12 15 = 4 / 9 :=
by
  sorry

end triangle_area_ratio_l39_39158


namespace smallest_k_for_distinct_real_roots_l39_39903

noncomputable def discriminant (a b c : ℝ) := b^2 - 4 * a * c

theorem smallest_k_for_distinct_real_roots :
  ∃ k : ℤ, (k > 0) ∧ discriminant (k : ℝ) (-3) (-9/4) > 0 ∧ (∀ m : ℤ, discriminant (m : ℝ) (-3) (-9/4) > 0 → m ≥ k) := 
by
  sorry

end smallest_k_for_distinct_real_roots_l39_39903


namespace expected_area_convex_hull_correct_l39_39525

def point_placement (x : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 10

def convex_hull_area (points : Finset (ℕ × ℤ)) : ℚ := 
  -- Definition of the area calculation goes here. This is a placeholder.
  0  -- Placeholder for the actual calculation

noncomputable def expected_convex_hull_area : ℚ := 
  -- Calculation of the expected area, which is complex and requires integration of the probability.
  sorry  -- Placeholder for the actual expected value

theorem expected_area_convex_hull_correct : 
  expected_convex_hull_area = 1793 / 128 :=
sorry

end expected_area_convex_hull_correct_l39_39525


namespace purple_coincide_pairs_l39_39197

theorem purple_coincide_pairs
    (yellow_triangles_upper : ℕ)
    (yellow_triangles_lower : ℕ)
    (green_triangles_upper : ℕ)
    (green_triangles_lower : ℕ)
    (purple_triangles_upper : ℕ)
    (purple_triangles_lower : ℕ)
    (yellow_coincide_pairs : ℕ)
    (green_coincide_pairs : ℕ)
    (yellow_purple_pairs : ℕ) :
    yellow_triangles_upper = 4 →
    yellow_triangles_lower = 4 →
    green_triangles_upper = 6 →
    green_triangles_lower = 6 →
    purple_triangles_upper = 10 →
    purple_triangles_lower = 10 →
    yellow_coincide_pairs = 3 →
    green_coincide_pairs = 4 →
    yellow_purple_pairs = 3 →
    (∃ purple_coincide_pairs : ℕ, purple_coincide_pairs = 5) :=
by sorry

end purple_coincide_pairs_l39_39197


namespace ones_digit_of_11_pow_46_l39_39164

theorem ones_digit_of_11_pow_46 : (11 ^ 46) % 10 = 1 :=
by sorry

end ones_digit_of_11_pow_46_l39_39164


namespace trigonometric_cos_value_l39_39874

open Real

theorem trigonometric_cos_value (α : ℝ) (h : sin (α + π / 6) = 1 / 3) : 
  cos (2 * α - 2 * π / 3) = -7 / 9 := 
sorry

end trigonometric_cos_value_l39_39874


namespace intersection_complement_eq_l39_39738

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Theorem
theorem intersection_complement_eq : (A ∩ (U \ B)) = {2, 3} :=
by 
  sorry

end intersection_complement_eq_l39_39738


namespace books_remaining_in_library_l39_39152

def initial_books : ℕ := 250
def books_taken_out_Tuesday : ℕ := 120
def books_returned_Wednesday : ℕ := 35
def books_withdrawn_Thursday : ℕ := 15

theorem books_remaining_in_library :
  initial_books
  - books_taken_out_Tuesday
  + books_returned_Wednesday
  - books_withdrawn_Thursday = 150 :=
by
  sorry

end books_remaining_in_library_l39_39152


namespace case1_case2_case3_l39_39631

-- Definitions from conditions
def tens_digit_one : ℕ := sorry
def units_digit_one : ℕ := sorry
def units_digit_two : ℕ := sorry
def tens_digit_two : ℕ := sorry
def sum_units_digits_ten : Prop := units_digit_one + units_digit_two = 10
def same_digit : ℕ := sorry
def sum_tens_digits_ten : Prop := tens_digit_one + tens_digit_two = 10

-- The proof problems
theorem case1 (A B D : ℕ) (hBplusD : B + D = 10) :
  (10 * A + B) * (10 * A + D) = 100 * (A^2 + A) + B * D :=
sorry

theorem case2 (A B C : ℕ) (hAplusC : A + C = 10) :
  (10 * A + B) * (10 * C + B) = 100 * A * C + 100 * B + B^2 :=
sorry

theorem case3 (A B C : ℕ) (hAplusB : A + B = 10) :
  (10 * A + B) * (10 * C + C) = 100 * A * C + 100 * C + B * C :=
sorry

end case1_case2_case3_l39_39631


namespace smallest_five_digit_number_divisible_l39_39702

def smallest_prime_divisible (n: ℕ) : Prop :=
  ∃ k: ℕ, n = 2310 * k ∧ 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_number_divisible :
  ∃ (n: ℕ), smallest_prime_divisible n ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_l39_39702


namespace inequality_holds_for_all_m_l39_39520

theorem inequality_holds_for_all_m (m : ℝ) (h1 : ∀ (x : ℝ), x^2 - 8 * x + 20 > 0)
  (h2 : m < -1/2) : ∀ (x : ℝ), (x ^ 2 - 8 * x + 20) / (m * x ^ 2 + 2 * (m + 1) * x + 9 * m + 4) < 0 :=
by
  sorry

end inequality_holds_for_all_m_l39_39520


namespace fish_count_l39_39623

theorem fish_count (total_fish blue_fish blue_spotted_fish : ℕ)
  (h1 : 1 / 3 * total_fish = blue_fish)
  (h2 : 1 / 2 * blue_fish = blue_spotted_fish)
  (h3 : blue_spotted_fish = 10) : total_fish = 60 :=
sorry

end fish_count_l39_39623


namespace brenda_distance_when_first_met_l39_39189

theorem brenda_distance_when_first_met
  (opposite_points : ∀ (d : ℕ), d = 150) -- Starting at diametrically opposite points on a 300m track means distance is 150m
  (constant_speeds : ∀ (B S x : ℕ), B * x = S * x) -- Brenda/ Sally run at constant speed
  (meet_again : ∀ (d₁ d₂ : ℕ), d₁ + d₂ = 300 + 100) -- Together they run 400 meters when they meet again, additional 100m by Sally
  : ∃ (x : ℕ), x = 150 :=
  by
    sorry

end brenda_distance_when_first_met_l39_39189


namespace least_positive_integer_with_12_factors_l39_39981

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l39_39981


namespace sum_gt_product_iff_l39_39899

theorem sum_gt_product_iff (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : m + n > m * n ↔ m = 1 ∨ n = 1 :=
sorry

end sum_gt_product_iff_l39_39899


namespace westward_measurement_l39_39742

def east_mov (d : ℕ) : ℤ := - (d : ℤ)

def west_mov (d : ℕ) : ℤ := d

theorem westward_measurement :
  east_mov 50 = -50 →
  west_mov 60 = 60 :=
by
  intro h
  exact rfl

end westward_measurement_l39_39742


namespace sample_quantities_and_probability_l39_39845

-- Define the given quantities from each workshop
def q_A := 10
def q_B := 20
def q_C := 30

-- Total sample size
def n := 6

-- Given conditions, the total quantity and sample ratio
def total_quantity := q_A + q_B + q_C
def ratio := n / total_quantity

-- Derived quantities in the samples based on the proportion
def sample_A := q_A * ratio
def sample_B := q_B * ratio
def sample_C := q_C * ratio

-- Combinatorial calculations
def C (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def total_combinations := C 6 2
def workshop_C_combinations := C 3 2
def probability_C_samples := workshop_C_combinations / total_combinations

-- Theorem to prove the quantities and probability
theorem sample_quantities_and_probability :
  sample_A = 1 ∧ sample_B = 2 ∧ sample_C = 3 ∧ probability_C_samples = 1 / 5 :=
by
  sorry

end sample_quantities_and_probability_l39_39845


namespace smallest_five_digit_number_divisible_l39_39701

def smallest_prime_divisible (n: ℕ) : Prop :=
  ∃ k: ℕ, n = 2310 * k ∧ 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_number_divisible :
  ∃ (n: ℕ), smallest_prime_divisible n ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_l39_39701


namespace gnuff_tutoring_rate_l39_39740

theorem gnuff_tutoring_rate (flat_rate : ℕ) (total_paid : ℕ) (minutes : ℕ) :
  flat_rate = 20 → total_paid = 146 → minutes = 18 → (total_paid - flat_rate) / minutes = 7 :=
by
  intros
  sorry

end gnuff_tutoring_rate_l39_39740


namespace remainder_equivalence_l39_39005

theorem remainder_equivalence (x : ℕ) (r : ℕ) (hx_pos : 0 < x) 
  (h1 : ∃ q1, 100 = q1 * x + r) (h2 : ∃ q2, 197 = q2 * x + r) : 
  r = 3 :=
by
  sorry

end remainder_equivalence_l39_39005


namespace sum_first_15_odd_integers_l39_39291

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l39_39291


namespace ratio_of_area_l39_39156

-- Define the sides of the triangles
def sides_GHI := (6, 8, 10)
def sides_JKL := (9, 12, 15)

-- Define the function to calculate the area of a triangle given its sides as a Pythagorean triple
def area_of_right_triangle (a b : Nat) : Nat :=
  (a * b) / 2

-- Calculate the areas
def area_GHI := area_of_right_triangle 6 8
def area_JKL := area_of_right_triangle 9 12

-- Calculate the ratio of the areas
def ratio_of_areas := area_GHI * 2 / area_JKL * 3

-- Statement to prove
theorem ratio_of_area (sides_GHI sides_JKL : (Nat × Nat × Nat)) :
  (ratio_of_areas = (4 : ℚ) / 9) :=
sorry

end ratio_of_area_l39_39156


namespace quadratic_function_opens_downwards_l39_39399

theorem quadratic_function_opens_downwards (m : ℝ) (h₁ : m - 1 < 0) (h₂ : m^2 + 1 = 2) : m = -1 :=
by {
  -- Proof would go here.
  sorry
}

end quadratic_function_opens_downwards_l39_39399


namespace least_positive_integer_with_12_factors_is_72_l39_39975

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l39_39975


namespace friends_lunch_spending_l39_39165

-- Problem conditions and statement to prove
theorem friends_lunch_spending (x : ℝ) (h1 : x + (x + 15) + (x - 20) + 2 * x = 100) : 
  x = 21 :=
by sorry

end friends_lunch_spending_l39_39165


namespace list_of_21_numbers_l39_39485

theorem list_of_21_numbers (numbers : List ℝ) (n : ℝ) (h_length : numbers.length = 21) 
  (h_mem : n ∈ numbers) 
  (h_n_avg : n = 4 * (numbers.sum - n) / 20) 
  (h_n_sum : n = (numbers.sum) / 6) : numbers.length - 1 = 20 :=
by
  -- We provide the statement with the correct hypotheses
  -- the proof is yet to be filled in
  sorry

end list_of_21_numbers_l39_39485


namespace maximum_rectangle_area_l39_39836

theorem maximum_rectangle_area (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 44) : 
  ∃ (l_max w_max : ℕ), l_max * w_max = 121 :=
by
  sorry

end maximum_rectangle_area_l39_39836


namespace min_value_of_sum_range_of_x_l39_39726

noncomputable def ab_condition (a b : ℝ) : Prop := a + b = 1
noncomputable def ra_positive (a b : ℝ) : Prop := a > 0 ∧ b > 0

-- Problem 1: Minimum value of (1/a + 4/b)

theorem min_value_of_sum (a b : ℝ) (h_ab : ab_condition a b) (h_pos : ra_positive a b) : 
    ∃ m : ℝ, m = 9 ∧ ∀ a b, ab_condition a b → ra_positive a b → 
    (1 / a + 4 / b) ≥ m :=
by sorry

-- Problem 2: Range of x for which the inequality holds

theorem range_of_x (a b x : ℝ) (h_ab : ab_condition a b) (h_pos : ra_positive a b) : 
    (1 / a + 4 / b) ≥ |2 * x - 1| - |x + 1| → x ∈ Set.Icc (-7 : ℝ) 11 :=
by sorry

end min_value_of_sum_range_of_x_l39_39726


namespace sum_first_15_odd_integers_l39_39299

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l39_39299


namespace area_inside_C_outside_A_B_l39_39349

-- Define the given circles with corresponding radii and positions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the circles A, B, and C with the specific properties given
def CircleA : Circle := { center := (0, 0), radius := 1 }
def CircleB : Circle := { center := (2, 0), radius := 1 }
def CircleC : Circle := { center := (1, 2), radius := 2 }

-- Given that Circle C is tangent to the midpoint M of the line segment AB
-- Prove the area inside Circle C but outside Circle A and B
theorem area_inside_C_outside_A_B : 
  let area_inside_C := π * CircleC.radius ^ 2
  let overlap_area := (π - 2)
  area_inside_C - overlap_area = 3 * π + 2 := by
  sorry

end area_inside_C_outside_A_B_l39_39349


namespace simplify_expr_l39_39257

variable (a b : ℝ)

theorem simplify_expr (h : a + b ≠ 0) : 
  a - b + 2 * b^2 / (a + b) = (a^2 + b^2) / (a + b) :=
sorry

end simplify_expr_l39_39257


namespace circle_equation_l39_39666

-- Define the given conditions
def point_P : ℝ × ℝ := (-1, 0)
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def center_C : ℝ × ℝ := (1, 2)

-- Define the required equation of the circle and the claim
def required_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

-- The Lean theorem statement
theorem circle_equation :
  ∃ (x y : ℝ), required_circle x y :=
sorry

end circle_equation_l39_39666


namespace nature_of_roots_indeterminate_l39_39394

variable (a b c : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem nature_of_roots_indeterminate (h : b^2 - 4 * a * c = 0) : 
  ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) = 0) ∨ ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) < 0) ∨ ((b + 2) ^ 2 - 4 * (a + 1) * (c + 1) > 0) :=
sorry

end nature_of_roots_indeterminate_l39_39394


namespace pond_eye_count_l39_39561

def total_animal_eyes (snakes alligators spiders snails : ℕ) 
    (snake_eyes alligator_eyes spider_eyes snail_eyes: ℕ) : ℕ :=
  snakes * snake_eyes + alligators * alligator_eyes + spiders * spider_eyes + snails * snail_eyes

theorem pond_eye_count : total_animal_eyes 18 10 5 15 2 2 8 2 = 126 := 
by
  sorry

end pond_eye_count_l39_39561


namespace find_original_selling_price_l39_39010

variable (x : ℝ) (discount_rate : ℝ) (final_price : ℝ)

def original_selling_price_exists (x : ℝ) (discount_rate : ℝ) (final_price : ℝ) : Prop :=
  (x * (1 - discount_rate) = final_price) → (x = 700)

theorem find_original_selling_price
  (discount_rate : ℝ := 0.20)
  (final_price : ℝ := 560) :
  ∃ x : ℝ, original_selling_price_exists x discount_rate final_price :=
by
  use 700
  sorry

end find_original_selling_price_l39_39010


namespace ratio_a_to_c_l39_39086

theorem ratio_a_to_c (a b c d : ℕ) 
  (h1 : a / b = 5 / 4) 
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) : 
  a / c = 75 / 16 := 
  by 
    sorry

end ratio_a_to_c_l39_39086


namespace isabella_paint_area_l39_39761

theorem isabella_paint_area 
    (bedrooms : ℕ) 
    (length width height doorway_window_area : ℕ) 
    (h1 : bedrooms = 4) 
    (h2 : length = 14) 
    (h3 : width = 12) 
    (h4 : height = 9)
    (h5 : doorway_window_area = 80) :
    (2 * (length * height) + 2 * (width * height) - doorway_window_area) * bedrooms = 1552 := by
       -- Calculate the area of the walls in one bedroom
       -- 2 * (length * height) + 2 * (width * height) - doorway_window_area = 388
       -- The total paintable area for 4 bedrooms = 388 * 4 = 1552
       sorry

end isabella_paint_area_l39_39761


namespace f_f_is_even_l39_39927

-- Let f be a function from reals to reals
variables {f : ℝ → ℝ}

-- Given that f is an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Theorem to prove
theorem f_f_is_even (h : is_even f) : is_even (fun x => f (f x)) :=
by
  intros
  unfold is_even at *
  -- at this point, we assume the function f is even,
  -- follow from the assumption, we can prove the result
  sorry

end f_f_is_even_l39_39927


namespace inequality_proof_l39_39529

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  1 / (1 - x^2) + 1 / (1 - y^2) ≥ 2 / (1 - x * y) :=
sorry

end inequality_proof_l39_39529


namespace find_multiple_l39_39131

variable (P W : ℕ)
variable (h1 : ∀ P W, P * 16 * (W / (P * 16)) = W)
variable (h2 : ∀ P W m, (m * P) * 4 * (W / (16 * P)) = W / 2)

theorem find_multiple (P W : ℕ) (h1 : ∀ P W, P * 16 * (W / (P * 16)) = W)
                      (h2 : ∀ P W m, (m * P) * 4 * (W / (16 * P)) = W / 2) : m = 2 :=
by
  sorry

end find_multiple_l39_39131


namespace negation_of_prop_l39_39608

theorem negation_of_prop :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
sorry

end negation_of_prop_l39_39608


namespace ff_even_of_f_even_l39_39926

-- Define what it means for a function f to be even.
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

-- Theorem to prove that f(f(x)) is even if f is even.
theorem ff_even_of_f_even (f : ℝ → ℝ) (hf : is_even_function f) : is_even_function (f ∘ f) :=
by
  intros x,
  specialize hf x,
  specialize hf (-x),
  rw [←hf, hf],
  sorry

end ff_even_of_f_even_l39_39926


namespace runner_advantage_l39_39090

theorem runner_advantage (x y z : ℝ) (hx_y: y - x = 0.1) (hy_z: z - y = 0.11111111111111111) :
  z - x = 0.21111111111111111 :=
by
  sorry

end runner_advantage_l39_39090


namespace mushroom_children_count_l39_39797

variables {n : ℕ} {A V S R : ℕ}

-- Conditions:
def condition1 (n : ℕ) (A : ℕ) (V : ℕ) : Prop :=
  ∀ (k : ℕ), k < n → V + A / 2 = k

def condition2 (S : ℕ) (A : ℕ) (R : ℕ) (V : ℕ) : Prop :=
  S + A = R + V + A

-- Proof statement
theorem mushroom_children_count (n : ℕ) (A : ℕ) (V : ℕ) (S : ℕ) (R : ℕ) :
  condition1 n A V → condition2 S A R V → n = 6 :=
by
  intros hcondition1 hcondition2
  sorry

end mushroom_children_count_l39_39797


namespace small_gifts_combinations_large_gifts_combinations_l39_39669

/-
  Definitions based on the given conditions:
  - 12 varieties of wrapping paper.
  - 3 colors of ribbon.
  - 6 types of gift cards.
  - Small gifts can use only 2 out of the 3 ribbon colors.
-/

def wrapping_paper_varieties : ℕ := 12
def ribbon_colors : ℕ := 3
def gift_card_types : ℕ := 6
def small_gift_ribbon_colors : ℕ := 2

/-
  Proof problems:

  - For small gifts, there are 12 * 2 * 6 combinations.
  - For large gifts, there are 12 * 3 * 6 combinations.
-/

theorem small_gifts_combinations :
  wrapping_paper_varieties * small_gift_ribbon_colors * gift_card_types = 144 :=
by
  sorry

theorem large_gifts_combinations :
  wrapping_paper_varieties * ribbon_colors * gift_card_types = 216 :=
by
  sorry

end small_gifts_combinations_large_gifts_combinations_l39_39669


namespace sum_of_even_numbers_l39_39522

-- Define the sequence of even numbers between 1 and 1001
def even_numbers_sequence (n : ℕ) := 2 * n

-- Conditions
def first_term := 2
def last_term := 1000
def common_difference := 2
def num_terms := 500
def sum_arithmetic_series (n : ℕ) (a l : ℕ) := n * (a + l) / 2

-- Main statement to be proved
theorem sum_of_even_numbers : 
  sum_arithmetic_series num_terms first_term last_term = 250502 := 
by
  sorry

end sum_of_even_numbers_l39_39522


namespace inequality_proof_l39_39479

theorem inequality_proof (a b c : ℝ) (hp : 0 < a ∧ 0 < b ∧ 0 < c) (hd : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    (bc / a + ac / b + ab / c > a + b + c) :=
by
  sorry

end inequality_proof_l39_39479


namespace cameron_list_length_l39_39029

-- Definitions of multiples
def smallest_multiple_perfect_square := 900
def smallest_multiple_perfect_cube := 27000
def multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

-- Problem statement
theorem cameron_list_length :
  ∀ n, 900 ≤ n ∧ n ≤ 27000 ∧ multiple_of_30 n ->
  (871 = (900 - 30 + 1)) :=
sorry

end cameron_list_length_l39_39029


namespace Pete_latest_time_to_LA_l39_39444

def minutesInHour := 60
def minutesOfWalk := 10
def minutesOfTrain := 80
def departureTime := 7 * minutesInHour + 30

def latestArrivalTime : Prop :=
  9 * minutesInHour = departureTime + minutesOfWalk + minutesOfTrain 

theorem Pete_latest_time_to_LA : latestArrivalTime :=
by
  sorry

end Pete_latest_time_to_LA_l39_39444


namespace pipeA_fill_time_l39_39788

variable (t : ℕ) -- t is the time in minutes for Pipe A to fill the tank

-- Conditions
def pipeA_duration (t : ℕ) : Prop :=
  t > 0

def pipeB_duration (t : ℕ) : Prop :=
  t / 3 > 0

def combined_rate (t : ℕ) : Prop :=
  3 * (1 / (4 / t)) = t

-- Problem
theorem pipeA_fill_time (h1 : pipeA_duration t) (h2 : pipeB_duration t) (h3 : combined_rate t) : t = 12 :=
sorry

end pipeA_fill_time_l39_39788


namespace problem_statement_l39_39256

noncomputable def original_expression (x : ℕ) : ℚ :=
(1 - 1 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1))

theorem problem_statement (x : ℕ) (hx1 : 3 - x ≥ 0) (hx2 : x ≠ 2) (hx3 : x ≠ 1) :
  original_expression 3 = 1 :=
by
  sorry

end problem_statement_l39_39256


namespace find_a10_l39_39612

variable {a : ℕ → ℝ} (d a1 : ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 0 + a (n-1))) / 2

theorem find_a10 (h1 : a 7 + a 9 = 10) 
                (h2 : sum_of_arithmetic_sequence a S)
                (h3 : S 11 = 11) : a 10 = 9 :=
sorry

end find_a10_l39_39612


namespace frank_problems_per_type_l39_39678

-- Definitions based on the problem conditions
def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def types_of_problems : ℕ := 4

-- The proof statement equivalent to the math problem
theorem frank_problems_per_type : frank_problems / types_of_problems = 30 :=
by 
  -- skipping the proof steps
  sorry

end frank_problems_per_type_l39_39678


namespace cos_double_angle_l39_39078

variable {α : ℝ}

theorem cos_double_angle (h1 : (Real.tan α - (1 / Real.tan α) = 3 / 2)) (h2 : (α > π / 4) ∧ (α < π / 2)) :
  Real.cos (2 * α) = -3 / 5 := 
sorry

end cos_double_angle_l39_39078


namespace min_dot_product_l39_39750

variables {V : Type*} [inner_product_space ℝ V]

theorem min_dot_product
  {A B C : V}
  (BC : dist B C = 2)
  (h : ∀ t : ℝ, ∥t • (B - A) + (1 - t) • (C - A)∥ ≥ ∥t_0 • (B - A) + (1 - t_0) • (C - A)∥)
  (t0_condition : ∥t_0 • (B - A) + (1 - t_0) • (C - A)∥ = 3) :
  ∃ (t_0 : ℝ), t_0 = 1/2 ∧ inner (B - A) (C - A) = 8 :=
begin
  sorry
end

end min_dot_product_l39_39750


namespace minimum_equilateral_triangles_l39_39641

theorem minimum_equilateral_triangles (side_small : ℝ) (side_large : ℝ)
  (h_small : side_small = 1) (h_large : side_large = 15) :
  225 = (side_large / side_small)^2 :=
by
  -- Proof is skipped.
  sorry

end minimum_equilateral_triangles_l39_39641


namespace factorization_correct_l39_39509

theorem factorization_correct (x : ℝ) :
  (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 12 * x^4 + 3) = 12 * (x^6 + 4 * x^4 - 1) := by
  sorry

end factorization_correct_l39_39509


namespace video_down_votes_l39_39844

theorem video_down_votes 
  (up_votes : ℕ)
  (ratio_up_down : up_votes / 1394 = 45 / 17)
  (up_votes_known : up_votes = 3690) : 
  3690 / 1394 = 45 / 17 :=
by
  sorry

end video_down_votes_l39_39844


namespace min_value_of_function_l39_39455

noncomputable def f (x : ℝ) : ℝ := 3 * x + 12 / x^2

theorem min_value_of_function :
  ∀ x > 0, f x ≥ 9 :=
by
  intro x hx_pos
  sorry

end min_value_of_function_l39_39455


namespace compare_powers_l39_39218

def n1 := 22^44
def n2 := 33^33
def n3 := 44^22

theorem compare_powers : n1 > n2 ∧ n2 > n3 := by
  sorry

end compare_powers_l39_39218


namespace distinct_flags_count_l39_39174

theorem distinct_flags_count : 
  ∃ n, n = 36 ∧ (∀ c1 c2 c3 : Fin 4, c1 ≠ c2 ∧ c2 ≠ c3 → n = 4 * 3 * 3) := 
sorry

end distinct_flags_count_l39_39174


namespace inequality_solution_l39_39148

theorem inequality_solution (x : ℝ) : 5 * x > 4 * x + 2 → x > 2 :=
by
  sorry

end inequality_solution_l39_39148


namespace length_AB_l39_39790

theorem length_AB (x : ℝ) (h1 : 0 < x)
  (hG : G = (0 + 1) / 2)
  (hH : H = (0 + G) / 2)
  (hI : I = (0 + H) / 2)
  (hJ : J = (0 + I) / 2)
  (hAJ : J - 0 = 2) :
  x = 32 := by
  sorry

end length_AB_l39_39790


namespace tank_fish_count_l39_39628

theorem tank_fish_count (total_fish blue_fish : ℕ) 
  (h1 : blue_fish = total_fish / 3)
  (h2 : 10 * 2 = blue_fish) : 
  total_fish = 60 :=
sorry

end tank_fish_count_l39_39628


namespace number_of_valid_n_l39_39744

theorem number_of_valid_n : 
  ∃ (c : Nat), (∀ n : Nat, (n + 9) * (n - 4) * (n - 13) < 0 → n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11 ∨ n = 12) ∧ c = 11 :=
by
  sorry

end number_of_valid_n_l39_39744


namespace arithmetic_progression_of_squares_l39_39609

theorem arithmetic_progression_of_squares 
  (a b c : ℝ)
  (h : 1 / (a + b) - 1 / (a + c) = 1 / (b + c) - 1 / (a + c)) :
  2 * b^2 = a^2 + c^2 :=
by
  sorry

end arithmetic_progression_of_squares_l39_39609


namespace cameron_list_count_l39_39027

theorem cameron_list_count :
  let numbers := {n : ℕ | 30 ≤ n ∧ n ≤ 900}
  in set.card numbers = 871 :=
sorry -- proof is omitted

end cameron_list_count_l39_39027


namespace sum_angles_bisected_l39_39371

theorem sum_angles_bisected (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : 0 < θ₁) (h₂ : 0 < θ₂) (h₃ : 0 < θ₃) (h₄ : 0 < θ₄)
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = 360) :
  (θ₁ / 2 + θ₃ / 2 = 180 ∨ θ₂ / 2 + θ₄ / 2 = 180) ∧ (θ₂ / 2 + θ₄ / 2 = 180 ∨ θ₁ / 2 + θ₃ / 2 = 180) := 
by 
  sorry

end sum_angles_bisected_l39_39371


namespace students_making_stars_l39_39957

theorem students_making_stars (total_stars stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : 
  total_stars / stars_per_student = 124 :=
by
  sorry

end students_making_stars_l39_39957


namespace smallest_five_digit_number_divisible_by_five_primes_l39_39695

theorem smallest_five_digit_number_divisible_by_five_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  lcm = 2310 → (∃ n : ℕ, n = 5 ∧ 10000 ≤ lcm * n ∧ lcm * n = 11550) :=
by
  intros p1 p2 p3 p4 p5 
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  intro hlcm
  use (5 : ℕ)
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end smallest_five_digit_number_divisible_by_five_primes_l39_39695


namespace arithmetic_geometric_sequence_l39_39217

theorem arithmetic_geometric_sequence {a b c x y : ℝ} (h₁: a ≠ b) (h₂: b ≠ c) (h₃: a ≠ c)
  (h₄ : 2 * b = a + c) (h₅ : x^2 = a * b) (h₆ : y^2 = b * c) :
  (x^2 + y^2 = 2 * b^2) ∧ (x^2 * y^2 ≠ b^4) :=
by {
  sorry
}

end arithmetic_geometric_sequence_l39_39217


namespace abs_function_le_two_l39_39120

theorem abs_function_le_two {x : ℝ} (h : |x| ≤ 2) : |3 * x - x^3| ≤ 2 :=
sorry

end abs_function_le_two_l39_39120


namespace like_terms_exponent_equality_l39_39895

theorem like_terms_exponent_equality (m n : ℕ) (a b : ℝ) 
    (H : 3 * a^m * b^2 = 2/3 * a * b^n) : m = 1 ∧ n = 2 :=
by
  sorry

end like_terms_exponent_equality_l39_39895


namespace find_min_value_expression_l39_39713

theorem find_min_value_expression (a b c : ℕ) (hb : b ≠ 0) (ha : a > 0) (hc : c > 0) :
  (a + b ≠ 0) ∧ (b - c ≠ 0) ∧ (c - a ≠ 0) →
  (min ((↑(a + b)^3 + ↑(b - c)^3 + ↑(c - a)^3) / (↑b)^3) = 3.5) :=
by
  sorry

end find_min_value_expression_l39_39713


namespace closest_whole_number_l39_39680

theorem closest_whole_number :
  let x := (10^2001 + 10^2003) / (10^2002 + 10^2002)
  abs ((x : ℝ) - 5) < 1 :=
by 
  sorry

end closest_whole_number_l39_39680


namespace minimum_discount_percentage_l39_39451

theorem minimum_discount_percentage (cost_price marked_price : ℝ) (profit_margin : ℝ) (discount : ℝ) :
  cost_price = 400 ∧ marked_price = 600 ∧ profit_margin = 0.05 ∧ 
  (marked_price * (1 - discount / 100) - cost_price) / cost_price ≥ profit_margin → discount ≤ 30 := 
by
  intros h
  rcases h with ⟨hc, hm, hp, hineq⟩
  sorry

end minimum_discount_percentage_l39_39451


namespace cubic_eq_one_complex_solution_l39_39083

theorem cubic_eq_one_complex_solution (k : ℂ) :
  (∃ (x : ℂ), 8 * x^3 + 12 * x^2 + k * x + 1 = 0) ∧
  (∀ (x y z : ℂ), 8 * x^3 + 12 * x^2 + k * x + 1 = 0 → 8 * y^3 + 12 * y^2 + k * y + 1 = 0
    → 8 * z^3 + 12 * z^2 + k * z + 1 = 0 → x = y ∧ y = z) →
  k = 6 :=
sorry

end cubic_eq_one_complex_solution_l39_39083


namespace ryan_hours_on_english_l39_39865

-- Given the conditions
def hours_on_chinese := 2
def hours_on_spanish := 4
def extra_hours_between_english_and_spanish := 3

-- We want to find out the hours on learning English
def hours_on_english := hours_on_spanish + extra_hours_between_english_and_spanish

-- Proof statement
theorem ryan_hours_on_english : hours_on_english = 7 := by
  -- This is where the proof would normally go.
  sorry

end ryan_hours_on_english_l39_39865


namespace extreme_value_of_f_range_of_a_log_inequality_l39_39388

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * log x + (a - 1) * x

theorem extreme_value_of_f (a : ℝ) (h : 0 < a) : (∃ x : ℝ, f x a = (a * (1 - log a)) / 2) :=
sorry

theorem range_of_a (h : ∃ x y : ℝ, g x > 0 ∧ g y > 0 ∧ f 1 a < 0 ∧ e⁻¹ < x ∧ x < e ∧ e⁻¹ < y ∧ y < e) : 
  (2 * e - 1) / (2 * e^2 + 2 * e) < a ∧ a < 1/2 :=
sorry

theorem log_inequality (x : ℝ) (h : 0 < x) : log x + (3 / (4 * x^2)) - (1 / exp x) > 0 :=
sorry

end extreme_value_of_f_range_of_a_log_inequality_l39_39388


namespace square_area_in_ellipse_l39_39668

theorem square_area_in_ellipse : ∀ (s : ℝ), 
  (s > 0) → 
  (∀ x y, (x = s ∨ x = -s) ∧ (y = s ∨ y = -s) → (x^2) / 4 + (y^2) / 8 = 1) → 
  (2 * s)^2 = 32 / 3 := by
  sorry

end square_area_in_ellipse_l39_39668


namespace least_positive_integer_with_12_factors_l39_39984

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l39_39984


namespace pounds_added_l39_39098

-- Definitions based on conditions
def initial_weight : ℝ := 5
def weight_increase_percent : ℝ := 1.5  -- 150% increase
def final_weight : ℝ := 28

-- Statement to prove
theorem pounds_added (w_initial w_final w_percent_added : ℝ) (h_initial: w_initial = 5) (h_final: w_final = 28)
(h_percent: w_percent_added = 1.5) :
  w_final - w_initial = 23 := 
by
  sorry

end pounds_added_l39_39098


namespace bases_for_204_base_b_l39_39861

theorem bases_for_204_base_b (b : ℕ) : (∃ n : ℤ, 2 * b^2 + 4 = n^2) ↔ b = 4 ∨ b = 6 ∨ b = 8 ∨ b = 10 :=
by
  sorry

end bases_for_204_base_b_l39_39861


namespace range_of_a4_l39_39733

noncomputable def geometric_sequence (a1 a2 a3 : ℝ) (q : ℝ) (a4 : ℝ) : Prop :=
  ∃ (a1 q : ℝ), 0 < a1 ∧ a1 < 1 ∧ 
                1 < a1 * q ∧ a1 * q < 2 ∧ 
                2 < a1 * q^2 ∧ a1 * q^2 < 4 ∧ 
                a4 = (a1 * q^2) * q ∧ 
                2 * Real.sqrt 2 < a4 ∧ a4 < 16

theorem range_of_a4 (a1 a2 a3 a4 : ℝ) (q : ℝ) (h1 : 0 < a1) (h2 : a1 < 1) 
  (h3 : 1 < a2) (h4 : a2 < 2) (h5 : a2 = a1 * q)
  (h6 : 2 < a3) (h7 : a3 < 4) (h8 : a3 = a1 * q^2) :
  2 * Real.sqrt 2 < a4 ∧ a4 < 16 :=
by
  have hq1 : 2 * q^2 < 1 := sorry    -- Placeholder for necessary inequalities
  have hq2: 1 < q ∧ q < 4 := sorry   -- Placeholder for necessary inequalities
  sorry

end range_of_a4_l39_39733


namespace exactly_one_is_multiple_of_5_l39_39552

theorem exactly_one_is_multiple_of_5 (a b : ℤ) (h: 24 * a^2 + 1 = b^2) : 
  (∃ k : ℤ, a = 5 * k) ∧ (∀ l : ℤ, b ≠ 5 * l) ∨ (∃ m : ℤ, b = 5 * m) ∧ (∀ n : ℤ, a ≠ 5 * n) :=
sorry

end exactly_one_is_multiple_of_5_l39_39552


namespace probability_of_winning_first_draw_better_chance_with_yellow_ball_l39_39636

-- The probability of winning on the first draw in the lottery promotion.
theorem probability_of_winning_first_draw :
  (1 / 4 : ℚ) = 0.25 :=
sorry

-- The optimal choice to add to the bag for the highest probability of receiving a fine gift.
theorem better_chance_with_yellow_ball :
  (3 / 5 : ℚ) > (2 / 5 : ℚ) :=
by norm_num

end probability_of_winning_first_draw_better_chance_with_yellow_ball_l39_39636


namespace subtraction_contradiction_l39_39659

theorem subtraction_contradiction (k t : ℕ) (hk_non_zero : k ≠ 0) (ht_non_zero : t ≠ 0) : 
  ¬ ((8 * 100 + k * 10 + 8) - (k * 100 + 8 * 10 + 8) = 1 * 100 + 6 * 10 + t * 1) :=
by
  sorry

end subtraction_contradiction_l39_39659


namespace pete_and_ray_spent_200_cents_l39_39117

-- Define the basic units
def cents_in_a_dollar := 100
def value_of_a_nickel := 5
def value_of_a_dime := 10

-- Define the initial amounts and spending
def pete_initial_amount := 250  -- cents
def ray_initial_amount := 250  -- cents
def pete_spent_nickels := 4 * value_of_a_nickel
def ray_remaining_dimes := 7 * value_of_a_dime

-- Calculate total amounts spent
def total_spent_pete := pete_spent_nickels
def total_spent_ray := ray_initial_amount - ray_remaining_dimes
def total_spent := total_spent_pete + total_spent_ray

-- The proof problem statement
theorem pete_and_ray_spent_200_cents : total_spent = 200 := by {
 sorry
}

end pete_and_ray_spent_200_cents_l39_39117


namespace find_a_exactly_two_solutions_l39_39363

theorem find_a_exactly_two_solutions :
  (∀ x y : ℝ, |x - 6 - y| + |x - 6 + y| = 12 ∧ (|x| - 6)^2 + (|y| - 8)^2 = a) ↔ (a = 4 ∨ a = 100) :=
sorry

end find_a_exactly_two_solutions_l39_39363


namespace cos_alpha_value_l39_39378

theorem cos_alpha_value (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : Real.sin α = 3 / 5) :
  Real.cos α = 4 / 5 :=
by
  sorry

end cos_alpha_value_l39_39378


namespace regression_eq_change_in_y_l39_39207

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 2 - 1.5 * x

-- Define the statement to be proved
theorem regression_eq_change_in_y (x : ℝ) :
  regression_eq (x + 1) = regression_eq x - 1.5 :=
by sorry

end regression_eq_change_in_y_l39_39207


namespace circle_radius_l39_39387

-- Define the main geometric scenario in Lean 4
theorem circle_radius 
  (O P A B : Type) 
  (r OP PA PB : ℝ)
  (h1 : PA * PB = 24)
  (h2 : OP = 5)
  : r = 7 
:= sorry

end circle_radius_l39_39387


namespace OM_geq_ON_l39_39655

variables {A B C D E F G H P Q M N O : Type*}

-- Definitions for geometrical concepts
def is_intersection_of_diagonals (M : Type*) (A B C D : Type*) : Prop :=
-- M is the intersection of the diagonals AC and BD
sorry

def is_intersection_of_midlines (N : Type*) (A B C D : Type*) : Prop :=
-- N is the intersection of the midlines connecting the midpoints of opposite sides
sorry

def is_center_of_circumscribed_circle (O : Type*) (A B C D : Type*) : Prop :=
-- O is the center of the circumscribed circle around quadrilateral ABCD
sorry

-- Proof problem
theorem OM_geq_ON (A B C D M N O : Type*) 
  (hm : is_intersection_of_diagonals M A B C D)
  (hn : is_intersection_of_midlines N A B C D)
  (ho : is_center_of_circumscribed_circle O A B C D) : 
  ∃ (OM ON : ℝ), OM ≥ ON :=
sorry

end OM_geq_ON_l39_39655


namespace lcm_gcd_product_l39_39288

def a : ℕ := 20 -- Defining the first number as 20
def b : ℕ := 90 -- Defining the second number as 90

theorem lcm_gcd_product : Nat.lcm a b * Nat.gcd a b = 1800 := 
by 
  -- Computation and proof steps would go here
  sorry -- Replace with actual proof

end lcm_gcd_product_l39_39288


namespace tank_fill_time_l39_39551

theorem tank_fill_time (A_rate B_rate C_rate : ℝ) (hA : A_rate = 1/30) (hB : B_rate = 1/20) (hC : C_rate = -1/40) : 
  1 / (A_rate + B_rate + C_rate) = 120 / 7 :=
by
  -- proof goes here
  sorry

end tank_fill_time_l39_39551


namespace rain_in_first_hour_l39_39571

theorem rain_in_first_hour :
  ∃ x : ℕ, (let rain_second_hour := 2 * x + 7 in x + rain_second_hour = 22) ∧ x = 5 :=
by
  sorry

end rain_in_first_hour_l39_39571


namespace smallest_five_digit_number_divisible_by_first_five_primes_l39_39700

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l39_39700


namespace exists_n_coprime_to_6_l39_39101

theorem exists_n_coprime_to_6 (k : ℕ) (hk : k.coprime 6) : ∃ n : ℕ, (2^n + 3^n + 6^n - 1) % k = 0 :=
by
  sorry

end exists_n_coprime_to_6_l39_39101


namespace circumferences_ratio_l39_39448

theorem circumferences_ratio (r1 r2 : ℝ) (h : (π * r1 ^ 2) / (π * r2 ^ 2) = 49 / 64) : r1 / r2 = 7 / 8 :=
sorry

end circumferences_ratio_l39_39448


namespace green_apples_more_than_red_apples_l39_39808

theorem green_apples_more_than_red_apples 
    (total_apples : ℕ)
    (red_apples : ℕ)
    (total_apples_eq : total_apples = 44)
    (red_apples_eq : red_apples = 16) :
    (total_apples - red_apples) - red_apples = 12 :=
by
  sorry

end green_apples_more_than_red_apples_l39_39808


namespace prove_distance_uphill_l39_39080

noncomputable def distance_uphill := 
  let flat_speed := 20
  let uphill_speed := 12
  let extra_flat_distance := 30
  let uphill_time (D : ℝ) := D / uphill_speed
  let flat_time (D : ℝ) := (D + extra_flat_distance) / flat_speed
  ∃ D : ℝ, uphill_time D = flat_time D ∧ D = 45

theorem prove_distance_uphill : distance_uphill :=
sorry

end prove_distance_uphill_l39_39080


namespace linda_paint_cans_l39_39438

theorem linda_paint_cans (wall_area : ℝ) (coverage_per_gallon : ℝ) (coats : ℝ) 
  (h1 : wall_area = 600) 
  (h2 : coverage_per_gallon = 400) 
  (h3 : coats = 2) : 
  (ceil (wall_area * coats / coverage_per_gallon) = 3) := 
by 
  sorry

end linda_paint_cans_l39_39438


namespace rain_in_first_hour_l39_39573

theorem rain_in_first_hour (x : ℝ) (h1 : ∀ y : ℝ, y = 2 * x + 7) (h2 : x + (2 * x + 7) = 22) : x = 5 :=
sorry

end rain_in_first_hour_l39_39573


namespace initial_volume_mixture_l39_39177

theorem initial_volume_mixture (x : ℝ) :
  (4 * x) / (3 * x + 13) = 5 / 7 →
  13 * x = 65 →
  7 * x = 35 := 
by
  intro h1 h2
  sorry

end initial_volume_mixture_l39_39177


namespace point_in_fourth_quadrant_l39_39253

def point : ℝ × ℝ := (3, -2)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l39_39253


namespace sum_of_first_15_odd_integers_l39_39310

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l39_39310


namespace abs_add_lt_abs_sub_l39_39897

variable {a b : ℝ}

theorem abs_add_lt_abs_sub (h1 : a * b < 0) : |a + b| < |a - b| :=
sorry

end abs_add_lt_abs_sub_l39_39897


namespace yoongi_division_l39_39831

theorem yoongi_division (x : ℕ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end yoongi_division_l39_39831


namespace polynomial_has_root_of_multiplicity_2_l39_39606

theorem polynomial_has_root_of_multiplicity_2 (r s k : ℝ)
  (h1 : x^3 + k * x - 128 = (x - r)^2 * (x - s)) -- polynomial has a root of multiplicity 2
  (h2 : -2 * r - s = 0)                         -- relationship from coefficient of x²
  (h3 : r^2 + 2 * r * s = k)                    -- relationship from coefficient of x
  (h4 : r^2 * s = 128)                          -- relationship from constant term
  : k = -48 := 
sorry

end polynomial_has_root_of_multiplicity_2_l39_39606


namespace leftover_balls_when_placing_60_in_tetrahedral_stack_l39_39589

def tetrahedral_number (n : ℕ) : ℕ :=
  n * (n + 1) * (n + 2) / 6

/--
  When placing 60 balls in a tetrahedral stack, the number of leftover balls is 4.
-/
theorem leftover_balls_when_placing_60_in_tetrahedral_stack :
  ∃ n, tetrahedral_number n ≤ 60 ∧ 60 - tetrahedral_number n = 4 := by
  sorry

end leftover_balls_when_placing_60_in_tetrahedral_stack_l39_39589


namespace single_elimination_games_l39_39840

theorem single_elimination_games (n : Nat) (h : n = 21) : games_needed = n - 1 :=
by
  sorry

end single_elimination_games_l39_39840


namespace car_price_difference_l39_39108

variable (original_paid old_car_proceeds : ℝ)
variable (new_car_price additional_amount : ℝ)

theorem car_price_difference :
  old_car_proceeds = new_car_price - additional_amount →
  old_car_proceeds = 0.8 * original_paid →
  additional_amount = 4000 →
  new_car_price = 30000 →
  (original_paid - new_car_price) = 2500 :=
by
  intro h1 h2 h3 h4
  sorry

end car_price_difference_l39_39108


namespace equation_solution_l39_39792

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (3 * x + 6) / (x^2 + 5 * x - 14) = (3 - x) / (x - 2) ↔ x = 3 ∨ x = -5 :=
by 
  sorry

end equation_solution_l39_39792


namespace fish_count_l39_39621

theorem fish_count (total_fish blue_fish blue_spotted_fish : ℕ)
  (h1 : 1 / 3 * total_fish = blue_fish)
  (h2 : 1 / 2 * blue_fish = blue_spotted_fish)
  (h3 : blue_spotted_fish = 10) : total_fish = 60 :=
sorry

end fish_count_l39_39621


namespace smallest_positive_five_digit_number_divisible_by_first_five_primes_l39_39704

theorem smallest_positive_five_digit_number_divisible_by_first_five_primes :
  ∃ n : ℕ, (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ 10000 ≤ n ∧ n = 11550 :=
by
  use 11550
  split
  · intros p hp
    fin_cases hp <;> norm_num
  split
  · norm_num
  rfl

end smallest_positive_five_digit_number_divisible_by_first_five_primes_l39_39704


namespace least_positive_integer_with_12_factors_l39_39986

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l39_39986


namespace books_from_library_l39_39935

def initial_books : ℝ := 54.5
def additional_books_1 : ℝ := 23.7
def returned_books_1 : ℝ := 12.3
def additional_books_2 : ℝ := 15.6
def returned_books_2 : ℝ := 9.1
def additional_books_3 : ℝ := 7.2

def total_books : ℝ :=
  initial_books + additional_books_1 - returned_books_1 + additional_books_2 - returned_books_2 + additional_books_3

theorem books_from_library : total_books = 79.6 := by
  sorry

end books_from_library_l39_39935


namespace theta_digit_l39_39467

theorem theta_digit (Θ : ℕ) (h : Θ ≠ 0) (h1 : 252 / Θ = 10 * 4 + Θ + Θ) : Θ = 5 :=
  sorry

end theta_digit_l39_39467


namespace jaime_average_speed_l39_39096

theorem jaime_average_speed :
  let start_time := 10.0 -- 10:00 AM
  let end_time := 15.5 -- 3:30 PM (in 24-hour format)
  let total_distance := 21.0 -- kilometers
  let total_time := end_time - start_time -- time in hours
  total_distance / total_time = 3.82 := 
sorry

end jaime_average_speed_l39_39096


namespace garden_breadth_l39_39905

theorem garden_breadth (P L B : ℕ) (h1 : P = 700) (h2 : L = 250) (h3 : P = 2 * (L + B)) : B = 100 :=
by
  sorry

end garden_breadth_l39_39905


namespace rhombus_area_l39_39321

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 21) : 
  (d1 * d2) / 2 = 157.5 :=
by
  sorry

end rhombus_area_l39_39321


namespace points_per_game_l39_39110

theorem points_per_game (total_points games : ℕ) (h1 : total_points = 91) (h2 : games = 13) :
  total_points / games = 7 :=
by
  sorry

end points_per_game_l39_39110


namespace find_monotonic_bijections_l39_39427

variable {f : ℝ → ℝ}

-- Define the properties of the function f
def bijective (f : ℝ → ℝ) : Prop :=
  Function.Bijective f

def condition (f : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, f t + f (f t) = 2 * t

theorem find_monotonic_bijections (f : ℝ → ℝ) (hf_bij : bijective f) (hf_cond : condition f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end find_monotonic_bijections_l39_39427


namespace number_of_stickers_used_to_decorate_l39_39776

def initial_stickers : ℕ := 20
def bought_stickers : ℕ := 12
def birthday_stickers : ℕ := 20
def given_stickers : ℕ := 5
def remaining_stickers : ℕ := 39

theorem number_of_stickers_used_to_decorate :
  (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers) = 8 :=
by
  -- Proof goes here
  sorry

end number_of_stickers_used_to_decorate_l39_39776


namespace number_of_students_l39_39955

theorem number_of_students (total_stars : ℕ) (stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : total_stars / stars_per_student = 124 :=
by
  sorry

end number_of_students_l39_39955


namespace max_profit_l39_39405

noncomputable def fixed_cost := 20000
noncomputable def variable_cost (x : ℝ) : ℝ :=
  if x < 8 then (1/3) * x^2 + 2 * x else 7 * x + 100 / x - 37
noncomputable def sales_price_per_unit : ℝ := 6
noncomputable def profit (x : ℝ) : ℝ :=
  let revenue := sales_price_per_unit * x
  let cost := fixed_cost / 10000 + variable_cost x
  revenue - cost

theorem max_profit : ∃ x : ℝ, (0 < x) ∧ (15 = profit 10) :=
by {
  sorry
}

end max_profit_l39_39405


namespace absolute_inequality_solution_l39_39510

theorem absolute_inequality_solution (x : ℝ) (hx : x > 0) :
  |5 - 2 * x| ≤ 8 ↔ 0 ≤ x ∧ x ≤ 6.5 :=
by sorry

end absolute_inequality_solution_l39_39510


namespace radian_measure_of_240_degrees_l39_39143

theorem radian_measure_of_240_degrees : (240 * (π / 180) = 4 * π / 3) := by
  sorry

end radian_measure_of_240_degrees_l39_39143


namespace red_to_blue_l39_39126

def is_red (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2020

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n ∧ ∃ m : ℕ, n = m ^ 2019

theorem red_to_blue (n : ℕ) (hn : n > 10^100000000) (hnred : is_red n) 
    (hn1red : is_red (n+1)) :
    ∃ (k : ℕ), 1 ≤ k ∧ k ≤ 2019 ∧ is_blue (n + k) :=
sorry

end red_to_blue_l39_39126


namespace Natalia_total_distance_l39_39442

variable (d_m d_t d_w d_r d_total : ℕ)

-- Conditions
axiom cond1 : d_m = 40
axiom cond2 : d_t = 50
axiom cond3 : d_w = d_t / 2
axiom cond4 : d_r = d_m + d_w

-- Question and answer
theorem Natalia_total_distance : 
  d_total = d_m + d_t + d_w + d_r → 
  d_total = 180 := 
by
  intros h
  simp [cond1, cond2, cond3, cond4] at h
  rw [cond1, cond2, cond3, cond4] in h
  simp at h
  exact h

end Natalia_total_distance_l39_39442


namespace least_positive_integer_with_12_factors_is_72_l39_39976

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l39_39976


namespace emma_age_proof_l39_39264

def is_age_of_emma (age : Nat) : Prop := 
  let guesses := [26, 29, 31, 33, 35, 39, 42, 44, 47, 50]
  let at_least_60_percent_low := (guesses.filter (· < age)).length * 10 ≥ 6 * guesses.length
  let exactly_two_off_by_one := (guesses.filter (λ x => x = age - 1 ∨ x = age + 1)).length = 2
  let is_prime := Nat.Prime age
  at_least_60_percent_low ∧ exactly_two_off_by_one ∧ is_prime

theorem emma_age_proof : is_age_of_emma 43 := 
  by sorry

end emma_age_proof_l39_39264


namespace find_m_l39_39251

theorem find_m (m : ℝ) (h : |m| = |m + 2|) : m = -1 :=
sorry

end find_m_l39_39251


namespace opposite_of_neg_third_l39_39610

theorem opposite_of_neg_third : (-(-1 / 3)) = (1 / 3) :=
by
  sorry

end opposite_of_neg_third_l39_39610


namespace range_of_a_l39_39539

noncomputable def proof_problem (x : ℝ) (a : ℝ) : Prop :=
  (x^2 - 4*x + 3 < 0) ∧ (x^2 - 6*x + 8 < 0) → (2*x^2 - 9*x + a < 0)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, proof_problem x a) ↔ a ≤ 9 :=
by
  sorry

end range_of_a_l39_39539


namespace inscribed_square_area_l39_39667

-- Define the condition of the problem
def inscribed_square_condition (t : ℝ) : Prop :=
(∀ x y, (x = t ∨ x = -t) ∧ (y = t ∨ y = -t) →
( x^2 / 4 + y^2 / 8 = 1 ))

-- The theorem that proves the area of the square inscribed in the ellipse
theorem inscribed_square_area :
  ∃ t : ℝ, inscribed_square_condition t ∧ (2 * t)*(2 * t) = 32 / 3 :=
begin
  sorry
end

end inscribed_square_area_l39_39667


namespace cameron_list_count_l39_39026

theorem cameron_list_count :
  let numbers := {n : ℕ | 30 ≤ n ∧ n ≤ 900}
  in set.card numbers = 871 :=
sorry -- proof is omitted

end cameron_list_count_l39_39026


namespace circle_radius_l39_39661

theorem circle_radius (P Q : ℝ) (h1 : P = π * r^2) (h2 : Q = 2 * π * r) (h3 : P / Q = 15) : r = 30 :=
by
  sorry

end circle_radius_l39_39661


namespace stickers_decorate_l39_39778

theorem stickers_decorate (initial_stickers bought_stickers birthday_stickers given_stickers remaining_stickers stickers_used : ℕ)
    (h1 : initial_stickers = 20)
    (h2 : bought_stickers = 12)
    (h3 : birthday_stickers = 20)
    (h4 : given_stickers = 5)
    (h5 : remaining_stickers = 39) :
    (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers = stickers_used) →
    stickers_used = 8 
:= by {sorry}

end stickers_decorate_l39_39778


namespace charlie_coins_worth_44_cents_l39_39348

-- Definitions based on the given conditions
def total_coins := 17
def p_eq_n_plus_2 (p n : ℕ) := p = n + 2

-- The main theorem stating the problem and the expected answer
theorem charlie_coins_worth_44_cents (p n : ℕ) (h1 : p + n = total_coins) (h2 : p_eq_n_plus_2 p n) :
  (7 * 5 + p * 1 = 44) :=
sorry

end charlie_coins_worth_44_cents_l39_39348


namespace combined_mpg_proof_l39_39592

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

end combined_mpg_proof_l39_39592


namespace inequality_proof_l39_39254

theorem inequality_proof (a b : ℝ) : 
  (a^4 + a^2 * b^2 + b^4) / 3 ≥ (a^3 * b + b^3 * a) / 2 :=
by
  sorry

end inequality_proof_l39_39254


namespace first_meet_at_starting_point_l39_39161

-- Definitions
def track_length := 300
def speed_A := 2
def speed_B := 4

-- Theorem: A and B will meet at the starting point for the first time after 400 seconds.
theorem first_meet_at_starting_point : 
  (∃ (t : ℕ), t = 400 ∧ (
    (∃ (n : ℕ), n * (track_length * (speed_B - speed_A)) = t * (speed_A + speed_B) * track_length) ∨
    (∃ (m : ℕ), m * (track_length * (speed_B + speed_A)) = t * (speed_A - speed_B) * track_length))) := 
    sorry

end first_meet_at_starting_point_l39_39161


namespace one_third_12x_plus_5_l39_39748

-- Define x as a real number
variable (x : ℝ)

-- Define the hypothesis
def h := 12 * x + 5

-- State the theorem
theorem one_third_12x_plus_5 : (1 / 3) * (12 * x + 5) = 4 * x + 5 / 3 :=
  by 
    sorry -- Proof is omitted

end one_third_12x_plus_5_l39_39748


namespace inv_point_zero_l39_39578

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 1 / (ax + b) ^ (1 / 3)

theorem inv_point_zero (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∀ x : ℝ, (inv_fun (f a b)) 1 = 0 → x = (1 - b) / a :=
sorry

end inv_point_zero_l39_39578


namespace square_difference_l39_39003

theorem square_difference : (601^2 - 599^2 = 2400) :=
by {
  -- Placeholder for the proof
  sorry
}

end square_difference_l39_39003


namespace div_relation_l39_39223

variables (a b c : ℚ)

theorem div_relation (h1 : a / b = 3) (h2 : b / c = 1 / 2) : c / a = 2 / 3 :=
by
  -- proof to be filled in
  sorry

end div_relation_l39_39223


namespace fractional_expression_evaluation_l39_39898

theorem fractional_expression_evaluation (a : ℝ) (h : a^3 + 3 * a^2 + a = 0) :
  ∃ b : ℝ, b = 0 ∨ b = 1 ∧ b = 2022 * a^2 / (a^4 + 2015 * a^2 + 1) :=
by
  sorry

end fractional_expression_evaluation_l39_39898


namespace solution_set_f_x_leq_m_solution_set_inequality_a_2_l39_39732

-- Part (I)
theorem solution_set_f_x_leq_m (a m : ℝ) (h : ∀ x : ℝ, |x - a| ≤ m ↔ -1 ≤ x ∧ x ≤ 5) :
  a = 2 ∧ m = 3 :=
sorry

-- Part (II)
theorem solution_set_inequality_a_2 (t : ℝ) (h_t : t ≥ 0) :
  (∀ x : ℝ, |x - 2| + t ≥ |x + 2 * t - 2| ↔ t = 0 ∧ (∀ x : ℝ, True) ∨ t > 0 ∧ ∀ x : ℝ, x ≤ 2 - t / 2) :=
sorry

end solution_set_f_x_leq_m_solution_set_inequality_a_2_l39_39732


namespace least_number_divisible_by_12_leaves_remainder_4_is_40_l39_39964

theorem least_number_divisible_by_12_leaves_remainder_4_is_40 :
  ∃ n : ℕ, (∀ k : ℕ, n = 12 * k + 4) ∧ (∀ m : ℕ, (∀ k : ℕ, m = 12 * k + 4) → n ≤ m) ∧ n = 40 :=
by
  sorry

end least_number_divisible_by_12_leaves_remainder_4_is_40_l39_39964


namespace range_of_m_l39_39216

-- Definitions for the sets A and B
def A : Set ℝ := {x | x ≥ 2}
def B (m : ℝ) : Set ℝ := {x | x ≥ m}

-- Prove that m ≥ 2 given the condition A ∪ B = A 
theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : m ≥ 2 :=
by
  sorry

end range_of_m_l39_39216


namespace Zoe_given_card_6_l39_39751

-- Define the cards and friends
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
def friends : List String := ["Eliza", "Miguel", "Naomi", "Ivan", "Zoe"]

-- Define scores 
def scores (name : String) : ℕ :=
  match name with
  | "Eliza"  => 15
  | "Miguel" => 11
  | "Naomi"  => 9
  | "Ivan"   => 13
  | "Zoe"    => 10
  | _ => 0

-- Each friend is given a pair of cards
def cardAssignments (name : String) : List (ℕ × ℕ) :=
  match name with
  | "Eliza"  => [(6,9), (7,8), (5,10), (4,11), (3,12)]
  | "Miguel" => [(1,10), (2,9), (3,8), (4,7), (5,6)]
  | "Naomi"  => [(1,8), (2,7), (3,6), (4,5)]
  | "Ivan"   => [(1,12), (2,11), (3,10), (4,9), (5,8), (6,7)]
  | "Zoe"    => [(1,9), (2,8), (3,7), (4,6)]
  | _ => []

-- The proof statement
theorem Zoe_given_card_6 : ∃ c1 c2, (c1, c2) ∈ cardAssignments "Zoe" ∧ (c1 = 6 ∨ c2 = 6)
:= by
  sorry -- Proof omitted as per the instructions

end Zoe_given_card_6_l39_39751


namespace sum_remainder_l39_39106

theorem sum_remainder (a b c : ℕ) (h1 : a % 30 = 14) (h2 : b % 30 = 5) (h3 : c % 30 = 18) : 
  (a + b + c) % 30 = 7 :=
by
  sorry

end sum_remainder_l39_39106


namespace max_value_expression_l39_39519

theorem max_value_expression (a b : ℝ) (ha: 0 < a) (hb: 0 < b) :
  ∃ M, M = 2 * Real.sqrt 87 ∧
       (∀ a b: ℝ, 0 < a → 0 < b →
       (|4 * a - 10 * b| + |2 * (a - b * Real.sqrt 3) - 5 * (a * Real.sqrt 3 + b)|) / Real.sqrt (a ^ 2 + b ^ 2) ≤ M) :=
sorry

end max_value_expression_l39_39519


namespace range_a_two_zeros_l39_39730

-- Definition of the function f(x)
def f (a x : ℝ) : ℝ := a * x^3 - 3 * a * x + 3 * a - 5

-- The theorem statement about the range of a
theorem range_a_two_zeros (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) → 1 ≤ a ∧ a ≤ 5 := sorry

end range_a_two_zeros_l39_39730


namespace option_B_is_incorrect_l39_39737

-- Define the set A
def A := { x : ℤ | x ^ 2 - 4 = 0 }

-- Statement to prove that -2 is an element of A
theorem option_B_is_incorrect : -2 ∈ A :=
sorry

end option_B_is_incorrect_l39_39737


namespace leopards_points_l39_39403

variables (x y : ℕ)

theorem leopards_points (h₁ : x + y = 50) (h₂ : x - y = 28) : y = 11 := by
  sorry

end leopards_points_l39_39403


namespace least_pos_int_with_12_pos_factors_is_72_l39_39972

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l39_39972


namespace farmer_land_l39_39839

theorem farmer_land (initial_land remaining_land : ℚ) (h1 : initial_land - initial_land / 10 = remaining_land) (h2 : remaining_land = 10) : initial_land = 100 / 9 := by
  sorry

end farmer_land_l39_39839


namespace no_such_six_tuples_exist_l39_39209

theorem no_such_six_tuples_exist :
  ∀ (a b c x y z : ℕ),
    1 ≤ c → c ≤ b → b ≤ a →
    1 ≤ z → z ≤ y → y ≤ x →
    2 * a + b + 4 * c = 4 * x * y * z →
    2 * x + y + 4 * z = 4 * a * b * c →
    False :=
by
  intros a b c x y z h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end no_such_six_tuples_exist_l39_39209


namespace meet_at_35_l39_39962

def walking_distance_A (t : ℕ) := 5 * t

def walking_distance_B (t : ℕ) := (t * (7 + t)) / 2

def total_distance (t : ℕ) := walking_distance_A t + walking_distance_B t

theorem meet_at_35 : ∃ (t : ℕ), total_distance t = 100 ∧ walking_distance_A t - walking_distance_B t = 35 := by
  sorry

end meet_at_35_l39_39962


namespace angle_Z_of_triangle_l39_39084

theorem angle_Z_of_triangle (X Y Z : ℝ) (h1 : X + Y = 90) (h2 : X + Y + Z = 180) : 
  Z = 90 := 
sorry

end angle_Z_of_triangle_l39_39084


namespace div_by_240_l39_39204

theorem div_by_240 (a b c d : ℕ) : 240 ∣ (a ^ (4 * b + d) - a ^ (4 * c + d)) :=
sorry

end div_by_240_l39_39204


namespace fred_found_43_seashells_l39_39639

-- Define the conditions
def tom_seashells : ℕ := 15
def additional_seashells : ℕ := 28

-- Define Fred's total seashells based on the conditions
def fred_seashells : ℕ := tom_seashells + additional_seashells

-- The theorem to prove that Fred found 43 seashells
theorem fred_found_43_seashells : fred_seashells = 43 :=
by
  -- Proof goes here
  sorry

end fred_found_43_seashells_l39_39639


namespace xiaoma_miscalculation_l39_39817

theorem xiaoma_miscalculation (x : ℤ) (h : 40 + x = 35) : 40 / x = -8 := by
  sorry

end xiaoma_miscalculation_l39_39817


namespace total_number_of_fish_l39_39626

theorem total_number_of_fish
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (blue_spotted_fish : ℕ)
  (h1 : blue_fish = total_fish / 3)
  (h2 : blue_spotted_fish = blue_fish / 2)
  (h3 : blue_spotted_fish = 10) :
  total_fish = 60 :=
by
  sorry

end total_number_of_fish_l39_39626


namespace cameron_list_length_l39_39030

-- Definitions of multiples
def smallest_multiple_perfect_square := 900
def smallest_multiple_perfect_cube := 27000
def multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

-- Problem statement
theorem cameron_list_length :
  ∀ n, 900 ≤ n ∧ n ≤ 27000 ∧ multiple_of_30 n ->
  (871 = (900 - 30 + 1)) :=
sorry

end cameron_list_length_l39_39030


namespace find_a_l39_39105

open Set

-- Define set A
def A : Set ℝ := {-1, 1, 3}

-- Define set B in terms of a
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

-- State the theorem
theorem find_a (a : ℝ) (h : A ∩ B a = {3}) : a = 1 :=
sorry

end find_a_l39_39105


namespace find_other_root_l39_39252

theorem find_other_root (k r : ℝ) (h1 : ∀ x : ℝ, 3 * x^2 + k * x + 6 = 0) (h2 : ∃ x : ℝ, 3 * x^2 + k * x + 6 = 0 ∧ x = 3) :
  r = 2 / 3 :=
sorry

end find_other_root_l39_39252


namespace students_left_early_l39_39806

theorem students_left_early :
  let initial_groups := 3
  let students_per_group := 8
  let students_remaining := 22
  let total_students := initial_groups * students_per_group
  total_students - students_remaining = 2 :=
by
  -- Define the initial conditions
  let initial_groups := 3
  let students_per_group := 8
  let students_remaining := 22
  let total_students := initial_groups * students_per_group
  -- Proof (to be completed)
  sorry

end students_left_early_l39_39806


namespace peaches_picked_up_l39_39791

variable (initial_peaches : ℕ) (final_peaches : ℕ)

theorem peaches_picked_up :
  initial_peaches = 13 →
  final_peaches = 55 →
  final_peaches - initial_peaches = 42 :=
by
  intros
  sorry

end peaches_picked_up_l39_39791


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l39_39708

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l39_39708


namespace veranda_area_l39_39134

/-- The width of the veranda on all sides of the room. -/
def width_of_veranda : ℝ := 2

/-- The length of the room. -/
def length_of_room : ℝ := 21

/-- The width of the room. -/
def width_of_room : ℝ := 12

/-- The area of the veranda given the conditions. -/
theorem veranda_area (length_of_room width_of_room width_of_veranda : ℝ) :
  (length_of_room + 2 * width_of_veranda) * (width_of_room + 2 * width_of_veranda) - length_of_room * width_of_room = 148 :=
by
  sorry

end veranda_area_l39_39134


namespace question_1_question_2_question_3_l39_39073

theorem question_1 (m : ℝ) : 
  (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + (m - 1) < 1) ↔ 
    m < (1 - 2 * Real.sqrt 7) / 3 := sorry

theorem question_2 (m : ℝ) : 
  ∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + (m - 1) ≥ (m + 1) * x := sorry

theorem question_3 (m : ℝ) : 
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), (m + 1) * x^2 - (m - 1) * x + (m - 1) ≥ 0) ↔ 
    m ≥ 1 := sorry

end question_1_question_2_question_3_l39_39073


namespace rectangle_dimension_area_l39_39640

theorem rectangle_dimension_area (x : ℝ) 
  (h_dim : (3 * x - 5) * (x + 7) = 14 * x - 35) : 
  x = 0 :=
by
  sorry

end rectangle_dimension_area_l39_39640


namespace find_m_l39_39222

theorem find_m (m : ℤ) (x y : ℤ) (h1 : x = 1) (h2 : y = m) (h3 : 3 * x - 4 * y = 7) : m = -1 :=
by
  sorry

end find_m_l39_39222


namespace probability_at_least_one_boy_one_girl_l39_39338

theorem probability_at_least_one_boy_one_girl :
  (∀ (P : SampleSpace → Prop), (P = (fun outcomes => nat.size outcomes = 4
                            ∧ (∃ outcome : outcomes, outcome = "boy")
                            ∧ ∃ outcome : outcomes, outcome = "girl"))
  -> (probability P = 7/8)) :=
by
  sorry

end probability_at_least_one_boy_one_girl_l39_39338


namespace g_of_f_at_3_eq_1902_l39_39247

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3 * x^2 + x + 2

theorem g_of_f_at_3_eq_1902 : g (f 3) = 1902 := by
  sorry

end g_of_f_at_3_eq_1902_l39_39247


namespace total_spent_by_pete_and_raymond_l39_39115

def initial_money_in_cents : ℕ := 250
def pete_spent_in_nickels : ℕ := 4
def nickel_value_in_cents : ℕ := 5
def raymond_dimes_left : ℕ := 7
def dime_value_in_cents : ℕ := 10

theorem total_spent_by_pete_and_raymond : 
  (pete_spent_in_nickels * nickel_value_in_cents) 
  + (initial_money_in_cents - (raymond_dimes_left * dime_value_in_cents)) = 200 := sorry

end total_spent_by_pete_and_raymond_l39_39115


namespace sum_first_4_terms_of_arithmetic_sequence_eq_8_l39_39381

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + (a 1 - a 0)

def S4 (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3

theorem sum_first_4_terms_of_arithmetic_sequence_eq_8
  (a : ℕ → ℤ) 
  (h_seq : arithmetic_seq a) 
  (h_a2 : a 1 = 1) 
  (h_a3 : a 2 = 3) :
  S4 a = 8 :=
by
  sorry

end sum_first_4_terms_of_arithmetic_sequence_eq_8_l39_39381


namespace avg_growth_rate_leq_half_sum_l39_39141

theorem avg_growth_rate_leq_half_sum (m n p : ℝ) (hm : 0 ≤ m) (hn : 0 ≤ n)
    (hp : (1 + p / 100)^2 = (1 + m / 100) * (1 + n / 100)) : 
    p ≤ (m + n) / 2 :=
by
  sorry

end avg_growth_rate_leq_half_sum_l39_39141


namespace initial_number_of_earning_members_l39_39598

theorem initial_number_of_earning_members (n : ℕ) 
  (h1 : 840 * n - 650 * (n - 1) = 1410) : n = 4 :=
by {
  -- Proof omitted
  sorry
}

end initial_number_of_earning_members_l39_39598


namespace inequality_problem_l39_39533

variable {a b c d : ℝ}

theorem inequality_problem (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
    a + c > b + d ∧ ad^2 > bc^2 ∧ (1 / bc) < (1 / ad) :=
by
  sorry

end inequality_problem_l39_39533


namespace minor_axis_length_is_2sqrt3_l39_39180

-- Define the points given in the problem
def points : List (ℝ × ℝ) := [(1, 1), (0, 0), (0, 3), (4, 0), (4, 3)]

-- Define a function that checks if an ellipse with axes parallel to the coordinate axes
-- passes through given points, and returns the length of its minor axis if it does.
noncomputable def minor_axis_length (pts : List (ℝ × ℝ)) : ℝ :=
  if h : (0,0) ∈ pts ∧ (0,3) ∈ pts ∧ (4,0) ∈ pts ∧ (4,3) ∈ pts ∧ (1,1) ∈ pts then
    let a := (4 - 0) / 2 -- half the width of the rectangle
    let b_sq := 3 -- derived from solving the ellipse equation
    2 * Real.sqrt b_sq
  else 0

-- The theorem statement:
theorem minor_axis_length_is_2sqrt3 : minor_axis_length points = 2 * Real.sqrt 3 := by
  sorry

end minor_axis_length_is_2sqrt3_l39_39180


namespace determine_triangle_ratio_l39_39359

theorem determine_triangle_ratio (a d : ℝ) (h : (a + d) ^ 2 = (a - d) ^ 2 + a ^ 2) : a / d = 2 + Real.sqrt 3 :=
sorry

end determine_triangle_ratio_l39_39359


namespace even_numbers_count_l39_39077

open Finset

-- Define the set of digits
def digits := {0, 1, 2, 5, 7, 8}

-- Define the range of numbers
def is_in_range (n : ℕ) : Prop := 300 ≤ n ∧ n < 800

-- Define what it means for a number to have all different digits from the set
def all_different_digits (n : ℕ) : Prop :=
  let ds := n.digits 10 in
  ds.to_finset ⊆ digits ∧ ds.length = ds.to_finset.card

-- Define what it means for a number to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- The main statement
theorem even_numbers_count :
  (card {n ∈ Ico 300 800 | is_even n ∧ all_different_digits n} = 36) :=
by
  -- Sorry is used to skip the proof
  sorry

end even_numbers_count_l39_39077


namespace AlexSumLargerBy105_l39_39673

/-- Definition of Alex's list of numbers -/
def AlexNumbers : List ℕ := List.range' 1 50

/-- Function that replaces digit '3' with digit '2' in a number -/
def replaceDigit (n : ℕ) : ℕ :=
  Nat.digits 10 n |>.reverse |>.map (λ d => if d = 3 then 2 else d) |>.reverse |>.foldl (λ acc d => acc * 10 + d) 0

/-- Definition of Tony's list of numbers -/
def TonyNumbers : List ℕ := AlexNumbers.map replaceDigit

/-- Summing up the numbers in a list -/
def listSum (lst : List ℕ) : ℕ := lst.foldl (λ acc x => acc + x) 0

/-- Theorem stating that the sum of Alex's numbers is 105 larger than Tony's numbers -/
theorem AlexSumLargerBy105 : listSum AlexNumbers - listSum TonyNumbers = 105 := by
  sorry

end AlexSumLargerBy105_l39_39673


namespace domain_cannot_be_0_to_3_l39_39656

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

-- Define the range of the function f
def range_f : Set ℝ := Set.Icc 1 2

-- Statement that the domain [0, 3] cannot be the domain of f given the range
theorem domain_cannot_be_0_to_3 :
  ∀ (f : ℝ → ℝ) (range_f : Set ℝ),
    (∀ x, 1 ≤ f x ∧ f x ≤ 2) →
    ¬ ∃ dom : Set ℝ, dom = Set.Icc 0 3 ∧ 
      (∀ x ∈ dom, f x ∈ range_f) :=
by
  sorry

end domain_cannot_be_0_to_3_l39_39656


namespace alyssa_picked_42_l39_39184

variable (totalPears nancyPears : ℕ)
variable (total_picked : totalPears = 59)
variable (nancy_picked : nancyPears = 17)

theorem alyssa_picked_42 (h1 : totalPears = 59) (h2 : nancyPears = 17) :
  totalPears - nancyPears = 42 :=
by
  sorry

end alyssa_picked_42_l39_39184


namespace canFormTriangle_cannotFormIsoscelesTriangle_l39_39620

section TriangleSticks

noncomputable def stickLengths : List ℝ := 
  List.range 10 |>.map (λ n => 1.9 ^ n)

def satisfiesTriangleInequality (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

theorem canFormTriangle : ∃ (a b c : ℝ), a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧ satisfiesTriangleInequality a b c :=
sorry

theorem cannotFormIsoscelesTriangle : ¬∃ (a b c : ℝ), a = b ∧ a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧ satisfiesTriangleInequality a b c :=
sorry

end TriangleSticks

end canFormTriangle_cannotFormIsoscelesTriangle_l39_39620


namespace tank_fish_count_l39_39627

theorem tank_fish_count (total_fish blue_fish : ℕ) 
  (h1 : blue_fish = total_fish / 3)
  (h2 : 10 * 2 = blue_fish) : 
  total_fish = 60 :=
sorry

end tank_fish_count_l39_39627


namespace product_of_two_numbers_l39_39805

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 27) (h2 : x - y = 9) : x * y = 162 := 
by {
  sorry
}

end product_of_two_numbers_l39_39805


namespace total_money_shared_l39_39848

/-- Assume there are four people Amanda, Ben, Carlos, and David, sharing an amount of money.
    Their portions are in the ratio 1:2:7:3.
    Amanda's portion is $20.
    Prove that the total amount of money shared by them is $260. -/
theorem total_money_shared (A B C D : ℕ) (h_ratio : A = 20 ∧ B = 2 * A ∧ C = 7 * A ∧ D = 3 * A) :
  A + B + C + D = 260 := by 
  sorry

end total_money_shared_l39_39848


namespace categorize_numbers_l39_39054

namespace NumberSets

-- Define the numbers
def eight := 8 : ℤ
def neg_one := -1 : ℤ
def neg_four_tenths := -2 / 5 : ℚ
def three_fifths := 3 / 5 : ℚ
def zero := 0 : ℚ
def one_third := 1 / 3 : ℚ
def neg_one_three_sevenths := -10 / 7 : ℚ
def neg_neg_five := 5 : ℤ
def neg_abs_neg_twenty_sevenths := -20 / 7 : ℚ

-- Define predicates for the sets
def is_positive (x : ℚ) : Prop := x > 0
def is_negative (x : ℚ) : Prop := x < 0
def is_integer (x : ℚ) : Prop := ∃ (z : ℤ), x = z
def is_fraction (x : ℚ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_non_negative_rational (x : ℚ) : Prop := x ≥ 0

-- Theorem to prove
theorem categorize_numbers :
  {x | is_positive x} = {eight, three_fifths, one_third, neg_neg_five} ∧
  {x | is_negative x} = {neg_one, neg_four_tenths, neg_one_three_sevenths, neg_abs_neg_twenty_sevenths} ∧
  {x | is_integer x} = {eight, neg_one, zero, neg_neg_five} ∧
  {x | is_fraction x} = {neg_four_tenths, three_fifths, one_third, neg_one_three_sevenths, neg_abs_neg_twenty_sevenths} ∧
  {x | is_non_negative_rational x} = {eight, three_fifths, zero, one_third, neg_neg_five} :=
  by
    sorry

end NumberSets

end categorize_numbers_l39_39054


namespace beads_probability_l39_39871

/-
  Four red beads, three white beads, and two blue beads are placed in a line in random order.
  Prove that the probability that no two neighboring beads are the same color is 1/70.
-/
theorem beads_probability :
  let total_permutations := Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 2)
  let valid_permutations := 18 -- conservative estimate from the solution
  (valid_permutations : ℚ) / total_permutations = 1 / 70 :=
by
  let total_permutations := Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 2)
  let valid_permutations := 18
  show (valid_permutations : ℚ) / total_permutations = 1 / 70
  -- skipping proof details
  sorry

end beads_probability_l39_39871


namespace initial_number_of_men_l39_39594

theorem initial_number_of_men (M : ℝ) (P : ℝ) (h1 : P = M * 20) (h2 : P = (M + 200) * 16.67) : M = 1000 :=
by
  sorry

end initial_number_of_men_l39_39594


namespace count_p_values_l39_39553

theorem count_p_values (p : ℤ) (n : ℝ) :
  (n = 16 * 10^(-p)) →
  (-4 < p ∧ p < 4) →
  ∃ m, p ∈ m ∧ (m.count = 3 ∧ m = [-2, 0, 2]) :=
by 
  sorry

end count_p_values_l39_39553


namespace least_positive_integer_with_12_factors_l39_39997

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l39_39997


namespace max_expression_value_l39_39518

theorem max_expression_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let expr := (|4 * a - 10 * b| + |2 * (a - b * sqrt 3) - 5 * (a * sqrt 3 + b)|) / sqrt (a^2 + b^2) in
  expr ≤ 2 * sqrt 87 :=
sorry

end max_expression_value_l39_39518


namespace sum_of_solutions_l39_39428

theorem sum_of_solutions :
  (∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    ((x, y) = (-1, 0) ∨ (x, y) = (2, 3) ∨ (x, y) = (7, 2))) →
  ((∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    (1 + 1 = 3 ∨ true)) → 
  (∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    (x, y) = (-1, 0) ∨ (x, y) = (2, 3) ∨ (x, y) = (7, 2))) →
  (-1) + 0 + 2 + 3 + 7 + 2 = 13 :=
by
  sorry

end sum_of_solutions_l39_39428


namespace sum_a_b_c_l39_39228

theorem sum_a_b_c (a b c : ℝ) (h1: a^2 + b^2 + c^2 = 390) (h2: a * b + b * c + c * a = 5) : a + b + c = 20 ∨ a + b + c = -20 := 
by 
  sorry

end sum_a_b_c_l39_39228


namespace extremum_only_at_2_l39_39213

noncomputable def f (x k : ℝ) : ℝ :=
  (exp x / x^2) - k * (2 / x + log x)

open Set

theorem extremum_only_at_2 (k : ℝ) :
  (∀ x > 0, deriv (λ x, f x k) x = 0 ↔ x = 2) → k ≤ real.exp 1 :=
by
  sorry

end extremum_only_at_2_l39_39213


namespace gallons_needed_to_grandmas_house_l39_39250

def car_fuel_efficiency : ℝ := 20
def distance_to_grandmas_house : ℝ := 100

theorem gallons_needed_to_grandmas_house : (distance_to_grandmas_house / car_fuel_efficiency) = 5 :=
by
  sorry

end gallons_needed_to_grandmas_house_l39_39250


namespace functional_equation_solution_l39_39064

-- Define the functional equation with given conditions
def func_eq (f : ℤ → ℝ) (N : ℕ) : Prop :=
  (∀ k : ℤ, f (2 * k) = 2 * f k) ∧
  (∀ k : ℤ, f (N - k) = f k)

-- State the mathematically equivalent proof problem
theorem functional_equation_solution (N : ℕ) (f : ℤ → ℝ) 
  (h1 : ∀ k : ℤ, f (2 * k) = 2 * f k)
  (h2 : ∀ k : ℤ, f (N - k) = f k) : 
  ∀ a : ℤ, f a = 0 := 
sorry

end functional_equation_solution_l39_39064


namespace prove_m_set_l39_39886

-- Define set A
def A : Set ℝ := {x | x^2 - 6*x + 8 = 0}

-- Define set B as dependent on m
def B (m : ℝ) : Set ℝ := {x | m*x - 4 = 0}

-- The main proof statement
theorem prove_m_set : {m : ℝ | B m ∩ A = B m} = {0, 1, 2} :=
by
  -- Code here would prove the above theorem
  sorry

end prove_m_set_l39_39886


namespace gcd_pow_diff_l39_39469

theorem gcd_pow_diff (m n: ℤ) (H1: m = 2^2025 - 1) (H2: n = 2^2016 - 1) : Int.gcd m n = 511 := by
  sorry

end gcd_pow_diff_l39_39469


namespace solve_inequality_l39_39260

theorem solve_inequality (x : ℝ) : (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
  sorry

end solve_inequality_l39_39260


namespace smallest_five_digit_divisible_by_primes_l39_39712

theorem smallest_five_digit_divisible_by_primes : 
  let primes := [2, 3, 5, 7, 11] in
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11))) in
  let five_digit_threshold := 10000 in
  ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
by
  let primes := [2, 3, 5, 7, 11]
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))
  have lcm_2310 : lcm_primes = 2310 := sorry
  let five_digit_threshold := 10000
  have exists_n : ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
    sorry
  exists_intro 5
  have 5_condition : 5 > 0 := sorry
  have 2310_5_condition : 2310 * 5 >= five_digit_threshold := sorry
  have answer : 2310 * 5 = 11550 := sorry
  exact  ⟨5, 5_condition, 2310_5_condition, answer⟩
  exact ⟨5, 5 > 0, 2310 * 5 ≥ 10000, 2310 * 5 = 11550⟩
  sorry

end smallest_five_digit_divisible_by_primes_l39_39712


namespace union_eq_l39_39373

def A : Set ℤ := {-1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

theorem union_eq : A ∪ B = {-1, 0, 1, 2, 3} := 
by 
  sorry

end union_eq_l39_39373


namespace multiplication_trick_l39_39652

theorem multiplication_trick (a b c : ℕ) (h : b + c = 10) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c :=
by
  sorry

end multiplication_trick_l39_39652


namespace number_of_valid_subsets_l39_39893

def setA : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def oddSet : Finset ℕ := {1, 3, 5, 7}
def evenSet : Finset ℕ := {2, 4, 6}

theorem number_of_valid_subsets : 
  (oddSet.powerset.card * (evenSet.powerset.card - 1) - oddSet.powerset.card) = 96 :=
by sorry

end number_of_valid_subsets_l39_39893


namespace domain_of_log_sqrt_l39_39947

theorem domain_of_log_sqrt (x : ℝ) : (-1 < x ∧ x ≤ 3) ↔ (0 < x + 1 ∧ 3 - x ≥ 0) :=
by
  sorry

end domain_of_log_sqrt_l39_39947


namespace g_evaluation_l39_39577

def g (a b : ℚ) : ℚ :=
  if a + b ≤ 4 then (2 * a * b - a + 3) / (3 * a)
  else (a * b - b - 1) / (-3 * b)

theorem g_evaluation : g 2 1 + g 2 4 = 7 / 12 := 
by {
  sorry
}

end g_evaluation_l39_39577


namespace exceeding_fraction_l39_39341

def repeatingDecimal (n : ℚ) (d : ℕ) : ℚ := n / (10^d - 1) -- Function for repeating decimal form

def decimal (n : ℚ) (d : ℕ) : ℚ := n / (10^d) -- Function for non-repeating decimal form

theorem exceeding_fraction : 
  let x := repeatingDecimal 6 2 in
  let y := decimal 6 2 in
  x - y = 2 / 3300 :=
by {
  have hx : x = repeatingDecimal 6 2 := rfl,
  have hy : y = decimal 6 2 := rfl,
  conversion,
  rw [hx, hy],
  rw [repeatingDecimal, decimal],
  have h_repeating : repeatingDecimal 6 2 = (6 / 99),  -- Known value for repeating decimal
  { rw ← div_div, norm_num },
  have h_decimal : decimal 6 2 = (6 / 100),  -- Known value for decimal
  { rw ← div_div, norm_num },
  calc
    (6 / 99) - (6 / 100)
    = ((6 * 100) - (6 * 99)) / (99 * 100) : by {
        rw [sub_div, mul_comm (10^2 - 1), mul_comm 2 5, pow_add, pow_one, mul_comm 10 99],
        norm_num
    }
    ... = ((600 - 594) / 9900)         : by norm_num
    ... = (6 / 9900)                  : by norm_num
    ... = (2 / 3300)                  : by norm_num
}

end exceeding_fraction_l39_39341


namespace calculate_expression_l39_39344

theorem calculate_expression (x : ℝ) (h : x = 3) : (x^2 - 5 * x + 4) / (x - 4) = 2 :=
by
  rw [h]
  sorry

end calculate_expression_l39_39344


namespace Rachel_money_left_l39_39591

theorem Rachel_money_left 
  (money_earned : ℕ)
  (lunch_fraction : ℚ)
  (clothes_percentage : ℚ)
  (dvd_cost : ℚ)
  (supplies_percentage : ℚ)
  (money_left : ℚ) :
  money_earned = 200 →
  lunch_fraction = 1 / 4 →
  clothes_percentage = 15 / 100 →
  dvd_cost = 24.50 →
  supplies_percentage = 10.5 / 100 →
  money_left = 74.50 :=
by
  intros h_money h_lunch h_clothes h_dvd h_supplies
  sorry

end Rachel_money_left_l39_39591


namespace phone_prices_purchase_plans_l39_39830

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

end phone_prices_purchase_plans_l39_39830


namespace base8_difference_divisible_by_7_l39_39946

theorem base8_difference_divisible_by_7 (A B : ℕ) (h₁ : A < 8) (h₂ : B < 8) (h₃ : A ≠ B) : 
  ∃ k : ℕ, k * 7 = (if 8 * A + B > 8 * B + A then 8 * A + B - (8 * B + A) else 8 * B + A - (8 * A + B)) :=
by
  sorry

end base8_difference_divisible_by_7_l39_39946


namespace find_fraction_l39_39863

theorem find_fraction (x : ℚ) (h : (1 / x) * (5 / 9) = 1 / 1.4814814814814814) : x = 740 / 999 := sorry

end find_fraction_l39_39863


namespace pet_shop_dogs_l39_39477

theorem pet_shop_dogs (D C B : ℕ) (x : ℕ) (h1 : D = 3 * x) (h2 : C = 5 * x) (h3 : B = 9 * x) (h4 : D + B = 204) : D = 51 := by
  -- omitted proof
  sorry

end pet_shop_dogs_l39_39477


namespace Cameron_list_count_l39_39036

theorem Cameron_list_count : 
  let lower_bound := 900
  let upper_bound := 27000
  let step := 30
  let n_min := lower_bound / step
  let n_max := upper_bound / step
  n_max - n_min + 1 = 871 :=
by
  sorry

end Cameron_list_count_l39_39036


namespace condition_sufficient_not_necessary_l39_39798

theorem condition_sufficient_not_necessary (x : ℝ) : (0 < x ∧ x < 5) → (|x - 2| < 3) ∧ (¬ ((|x - 2| < 3) → (0 < x ∧ x < 5))) :=
by
  sorry

end condition_sufficient_not_necessary_l39_39798


namespace krakozyabrs_total_count_l39_39414

theorem krakozyabrs_total_count :
  (∃ (T : ℕ), 
  (∀ (H W n : ℕ),
    0.2 * H = n ∧ 0.25 * W = n ∧ H = 5 * n ∧ W = 4 * n ∧ T = H + W - n ∧ 25 < T ∧ T < 35) ∧ 
  T = 32) :=
by sorry

end krakozyabrs_total_count_l39_39414


namespace common_area_of_rectangle_and_circle_l39_39493

theorem common_area_of_rectangle_and_circle :
  let l := 10
  let w := 2 * Real.sqrt 5
  let r := 3
  ∃ (common_area : ℝ), common_area = 9 * Real.pi :=
by
  let l := 10
  let w := 2 * Real.sqrt 5
  let r := 3
  have common_area := 9 * Real.pi
  use common_area
  sorry

end common_area_of_rectangle_and_circle_l39_39493


namespace sum_of_first_three_terms_is_zero_l39_39272

variable (a d : ℤ) 

-- Definitions from the conditions
def a₄ := a + 3 * d
def a₅ := a + 4 * d
def a₆ := a + 5 * d

-- Theorem statement
theorem sum_of_first_three_terms_is_zero 
  (h₁ : a₄ = 8) 
  (h₂ : a₅ = 12) 
  (h₃ : a₆ = 16) : 
  a + (a + d) + (a + 2 * d) = 0 := 
by 
  sorry

end sum_of_first_three_terms_is_zero_l39_39272


namespace inequality_solution_l39_39263

theorem inequality_solution (x : ℝ) : (2 * x - 1) / 3 ≥ 1 → x ≥ 2 := by
  sorry

end inequality_solution_l39_39263


namespace breadth_of_rectangle_l39_39749

theorem breadth_of_rectangle 
  (Perimeter Length Breadth : ℝ)
  (h_perimeter_eq : Perimeter = 2 * (Length + Breadth))
  (h_given_perimeter : Perimeter = 480)
  (h_given_length : Length = 140) :
  Breadth = 100 := 
by
  sorry

end breadth_of_rectangle_l39_39749


namespace boys_neither_happy_nor_sad_l39_39650

theorem boys_neither_happy_nor_sad (total_children : ℕ)
  (happy_children sad_children neither_happy_nor_sad total_boys total_girls : ℕ)
  (happy_boys sad_girls : ℕ)
  (h_total : total_children = 60)
  (h_happy : happy_children = 30)
  (h_sad : sad_children = 10)
  (h_neither : neither_happy_nor_sad = 20)
  (h_boys : total_boys = 17)
  (h_girls : total_girls = 43)
  (h_happy_boys : happy_boys = 6)
  (h_sad_girls : sad_girls = 4) :
  ∃ (boys_neither_happy_nor_sad : ℕ), boys_neither_happy_nor_sad = 5 := by
  sorry

end boys_neither_happy_nor_sad_l39_39650


namespace recurring_decimal_to_fraction_l39_39360

theorem recurring_decimal_to_fraction : (∃ (x : ℚ), x = 3 + 56 / 99) :=
by
  have x : ℚ := 3 + 56 / 99
  exists x
  sorry

end recurring_decimal_to_fraction_l39_39360


namespace speed_second_half_l39_39488

theorem speed_second_half (total_time : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) :
    total_time = 12 → first_half_speed = 35 → total_distance = 560 → 
    (280 / (12 - (280 / 35)) = 70) :=
by
  intros ht hf hd
  sorry

end speed_second_half_l39_39488


namespace find_m_l39_39890

theorem find_m (x y m : ℤ) 
  (h1 : x + 2 * y = 5 * m) 
  (h2 : x - 2 * y = 9 * m) 
  (h3 : 3 * x + 2 * y = 19) : 
  m = 1 := 
by 
  sorry

end find_m_l39_39890


namespace all_numbers_rational_l39_39007

-- Define the mathematical operations for the problem
def fourth_root (x : ℝ) : ℝ := x ^ (1 / 4)
def square_root (x : ℝ) : ℝ := x ^ (1 / 2)
def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem all_numbers_rational :
    (∃ x1 : ℚ, fourth_root 81 = x1) ∧
    (∃ x2 : ℚ, square_root 0.64 = x2) ∧
    (∃ x3 : ℚ, cube_root 0.001 = x3) ∧
    (∃ x4 : ℚ, (cube_root 8) * (square_root ((0.25)⁻¹)) = x4) :=
  sorry

end all_numbers_rational_l39_39007


namespace inequality_solution_l39_39262

theorem inequality_solution (x : ℝ) : 
    (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
    sorry

end inequality_solution_l39_39262


namespace trigonometric_identity_l39_39526

open Real

theorem trigonometric_identity (α : ℝ) (h1 : cos α = -4 / 5) (h2 : π < α ∧ α < (3 * π / 2)) :
    (1 + tan (α / 2)) / (1 - tan (α / 2)) = -1 / 2 := by
  sorry

end trigonometric_identity_l39_39526


namespace find_width_l39_39802

variable (a b : ℝ)

def perimeter : ℝ := 6 * a + 4 * b
def length : ℝ := 2 * a + b
def width : ℝ := a + b

theorem find_width (h : perimeter a b = 6 * a + 4 * b)
                   (h₂ : length a b = 2 * a + b) : width a b = (perimeter a b) / 2 - length a b := by
  sorry

end find_width_l39_39802


namespace min_xyz_product_l39_39771

open Real

theorem min_xyz_product
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_sum : x + y + z = 1)
  (h_no_more_than_twice : x ≤ 2 * y ∧ y ≤ 2 * x ∧ y ≤ 2 * z ∧ z ≤ 2 * y) :
  ∃ p : ℝ, (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x + y + z = 1 → x ≤ 2 * y ∧ y ≤ 2 * x ∧ y ≤ 2 * z ∧ z ≤ 2 * y → x * y * z ≥ p) ∧ p = 1 / 32 :=
by
  sorry

end min_xyz_product_l39_39771


namespace toluene_production_l39_39057

def molar_mass_benzene : ℝ := 78.11 -- The molar mass of benzene in g/mol
def benzene_mass : ℝ := 156 -- The mass of benzene in grams
def methane_moles : ℝ := 2 -- The moles of methane

-- Define the balanced chemical reaction
def balanced_reaction (benzene methanol toluene hydrogen : ℝ) : Prop :=
  benzene + methanol = toluene + hydrogen

-- The main theorem statement
theorem toluene_production (h1 : balanced_reaction benzene_mass methane_moles 1 1)
  (h2 : benzene_mass / molar_mass_benzene = 2) :
  ∃ toluene_moles : ℝ, toluene_moles = 2 :=
by
  sorry

end toluene_production_l39_39057


namespace solution_set_of_inequality_l39_39276

theorem solution_set_of_inequality (x : ℝ) : 
  (x * |x - 1| > 0) ↔ ((0 < x ∧ x < 1) ∨ (x > 1)) := 
by
  sorry

end solution_set_of_inequality_l39_39276


namespace slope_transformation_l39_39729

theorem slope_transformation :
  ∀ (b : ℝ), ∃ k : ℝ, 
  (∀ x : ℝ, k * x + b = k * (x + 4) + b + 1) → k = -1/4 :=
by
  intros b
  use -1/4
  intros h
  sorry

end slope_transformation_l39_39729


namespace even_function_phi_l39_39074

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ)

noncomputable def f' (x φ : ℝ) : ℝ := -Real.sqrt 3 * Real.sin (Real.sqrt 3 * x + φ)

noncomputable def y (x φ : ℝ) : ℝ := f x φ + f' x φ

def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g x = g (-x)

theorem even_function_phi :
  (∀ x : ℝ, y x φ = y (-x) φ) → ∃ k : ℤ, φ = -Real.pi / 3 + k * Real.pi :=
by
  sorry

end even_function_phi_l39_39074


namespace triangle_identity_proof_l39_39590

variables (r r_a r_b r_c R S p : ℝ)
-- assume necessary properties for valid triangle (not explicitly given in problem but implied)
-- nonnegativity, relations between inradius, exradii and circumradius, etc.

theorem triangle_identity_proof
  (h_r_pos : 0 < r)
  (h_ra_pos : 0 < r_a)
  (h_rb_pos : 0 < r_b)
  (h_rc_pos : 0 < r_c)
  (h_R_pos : 0 < R)
  (h_S_pos : 0 < S)
  (h_p_pos : 0 < p)
  (h_area : S = r * p) :
  (1 / r^3) - (1 / r_a^3) - (1 / r_b^3) - (1 / r_c^3) = (12 * R) / (S^2) :=
sorry

end triangle_identity_proof_l39_39590


namespace trigonometric_identity_l39_39714

theorem trigonometric_identity (α : ℝ) : 
  (1 + Real.cos (2 * α - 2 * Real.pi) + Real.cos (4 * α + 2 * Real.pi) - Real.cos (6 * α - Real.pi)) /
  (Real.cos (2 * Real.pi - 2 * α) + 2 * Real.cos (2 * α + Real.pi) ^ 2 - 1) = 
  2 * Real.cos (2 * α) :=
by sorry

end trigonometric_identity_l39_39714


namespace sum_first_15_odd_integers_l39_39297

theorem sum_first_15_odd_integers : 
  let a := 1
  let n := 15
  let d := 2
  let l := a + (n-1) * d
  let S := n / 2 * (a + l)
  S = 225 :=
by
  sorry

end sum_first_15_odd_integers_l39_39297


namespace square_difference_example_l39_39002

theorem square_difference_example : 601^2 - 599^2 = 2400 := 
by sorry

end square_difference_example_l39_39002


namespace darma_peanut_consumption_l39_39859

theorem darma_peanut_consumption :
  ∀ (t : ℕ) (rate : ℕ),
  (rate = 20 / 15) →  -- Given the rate of peanut consumption
  (t = 6 * 60) →     -- Given that the total time is 6 minutes
  (rate * t = 480) :=  -- Prove that the total number of peanuts eaten in 6 minutes is 480
by
  intros t rate h_rate h_time
  sorry

end darma_peanut_consumption_l39_39859


namespace distance_Bella_Galya_l39_39093

theorem distance_Bella_Galya 
    (AB BV VG GD : ℝ)
    (DB : AB + 3 * BV + 2 * VG + GD = 700)
    (DV : AB + 2 * BV + 2 * VG + GD = 600)
    (DG : AB + 2 * BV + 3 * VG + GD = 650) :
    BV + VG = 150 := 
by
  -- Proof goes here
  sorry

end distance_Bella_Galya_l39_39093


namespace managers_salary_l39_39266

-- Definitions based on conditions
def avg_salary_50_employees : ℝ := 2000
def num_employees : ℕ := 50
def new_avg_salary : ℝ := 2150
def num_employees_with_manager : ℕ := 51

-- Condition statement: The manager's salary such that when added, average salary increases as given.
theorem managers_salary (M : ℝ) :
  (num_employees * avg_salary_50_employees + M) / num_employees_with_manager = new_avg_salary →
  M = 9650 := sorry

end managers_salary_l39_39266


namespace xy_in_A_l39_39921

def A : Set ℤ :=
  {z | ∃ (a b k n : ℤ), z = a^2 + k * a * b + n * b^2}

theorem xy_in_A (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := sorry

end xy_in_A_l39_39921


namespace rectangle_side_length_l39_39353

theorem rectangle_side_length (x : ℝ) (h1 : 0 < x) (h2 : 2 * (x + 6) = 40) : x = 14 :=
by
  sorry

end rectangle_side_length_l39_39353


namespace bouncy_ball_pack_count_l39_39426

theorem bouncy_ball_pack_count
  (x : ℤ)  -- Let x be the number of bouncy balls in each pack
  (r : ℤ := 7 * x)  -- Total number of red bouncy balls
  (y : ℤ := 6 * x)  -- Total number of yellow bouncy balls
  (h : r = y + 18)  -- Condition: 7x = 6x + 18
  : x = 18 := sorry

end bouncy_ball_pack_count_l39_39426


namespace total_slices_sold_l39_39462

theorem total_slices_sold (sold_yesterday served_today : ℕ) (h1 : sold_yesterday = 5) (h2 : served_today = 2) :
  sold_yesterday + served_today = 7 :=
by
  -- Proof skipped
  exact sorry

end total_slices_sold_l39_39462


namespace cos_75_degree_identity_l39_39192

theorem cos_75_degree_identity :
  Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 :=
by
  sorry

end cos_75_degree_identity_l39_39192


namespace smallest_number_condition_l39_39642

theorem smallest_number_condition
  (x : ℕ)
  (h1 : (x - 24) % 5 = 0)
  (h2 : (x - 24) % 10 = 0)
  (h3 : (x - 24) % 15 = 0)
  (h4 : (x - 24) / 30 = 84)
  : x = 2544 := 
sorry

end smallest_number_condition_l39_39642


namespace total_kilometers_ridden_l39_39443

theorem total_kilometers_ridden :
  ∀ (d1 d2 d3 d4 : ℕ),
    d1 = 40 →
    d2 = 50 →
    d3 = d2 - d2 / 2 →
    d4 = d1 + d3 →
    d1 + d2 + d3 + d4 = 180 :=
by 
  intros d1 d2 d3 d4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_kilometers_ridden_l39_39443


namespace percentage_decrease_l39_39611

theorem percentage_decrease (original_price new_price decrease: ℝ) (h₁: original_price = 2400) (h₂: new_price = 1200) (h₃: decrease = original_price - new_price): 
  decrease / original_price * 100 = 50 :=
by
  rw [h₁, h₂] at h₃ -- Update the decrease according to given prices
  sorry -- Left as a placeholder for the actual proof

end percentage_decrease_l39_39611


namespace determinant_value_l39_39072

-- Given definitions and conditions
def determinant (a b c d : ℤ) : ℤ := a * d - b * c
def special_determinant (m : ℤ) : ℤ := determinant (m^2) (m-3) (1-2*m) (m-2)

-- The proof problem
theorem determinant_value (m : ℤ) (h : m^2 - 2 * m - 3 = 0) : special_determinant m = 9 := sorry

end determinant_value_l39_39072


namespace cannot_invert_all_signs_l39_39562

structure RegularDecagon :=
  (vertices : Fin 10 → ℤ)
  (diagonals : Fin 45 → ℤ) -- Assume we encode the intersections as unique indices for simplicity.
  (all_positives : ∀ v, vertices v = 1 ∧ ∀ d, diagonals d = 1)

def isValidSignChange (t : List ℤ) : Prop :=
  t.length % 2 = 0

theorem cannot_invert_all_signs (D : RegularDecagon) :
  ¬ (∃ f : Fin 10 → ℤ → ℤ, ∀ (side : Fin 10) (val : ℤ), f side val = -val) :=
sorry

end cannot_invert_all_signs_l39_39562


namespace values_of_m_and_n_l39_39727

theorem values_of_m_and_n (m n : ℕ) (h_cond1 : 2 * m + 3 = 5 * n - 2) (h_cond2 : 5 * n - 2 < 15) : m = 5 ∧ n = 3 :=
by
  sorry

end values_of_m_and_n_l39_39727


namespace factorize_quadratic_l39_39052

theorem factorize_quadratic (x : ℝ) : x^2 - 2 * x = x * (x - 2) :=
sorry

end factorize_quadratic_l39_39052


namespace least_positive_integer_with_12_factors_is_972_l39_39991

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l39_39991


namespace problem_1_problem_2_l39_39383

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |x + 1|

theorem problem_1 : {x : ℝ | f x < 4} = {x : ℝ | -4 / 3 < x ∧ x < 4 / 3} :=
by 
  sorry

theorem problem_2 (x₀ : ℝ) (h : ∀ t : ℝ, f x₀ < |m + t| + |t - m|) : 
  {m : ℝ | ∃ x t, f x < |m + t| + |t - m|} = {m : ℝ | m < -3 / 4 ∨ m > 3 / 4} :=
by 
  sorry

end problem_1_problem_2_l39_39383


namespace carpet_area_l39_39919

def width : ℝ := 8
def length : ℝ := 1.5

theorem carpet_area : width * length = 12 := by
  sorry

end carpet_area_l39_39919


namespace largest_possible_s_l39_39928

noncomputable def max_value_of_s (p q r s : ℝ) (h1 : p + q + r + s = 8) (h2 : pq + pr + ps + qr + qs + rs = 12) : ℝ :=
  2 + 3 * Real.sqrt 2

theorem largest_possible_s (p q r s : ℝ) (h1 : p + q + r + s = 8) (h2 : pq + pr + ps + qr + qs + rs = 12) :
  s ≤ max_value_of_s p q r s h1 h2 := 
sorry

end largest_possible_s_l39_39928


namespace least_pos_int_with_12_pos_factors_is_72_l39_39969

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l39_39969


namespace total_cost_of_goods_l39_39602

theorem total_cost_of_goods :
  ∃ (M R F : ℝ),
    (10 * M = 24 * R) ∧
    (6 * F = 2 * R) ∧
    (F = 20.50) ∧
    (4 * M + 3 * R + 5 * F = 877.40) :=
by {
  sorry
}

end total_cost_of_goods_l39_39602


namespace max_z_value_l39_39891

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 0) (h2 : x * y + y * z + z * x = -3) : z ≤ 2 := sorry

end max_z_value_l39_39891


namespace reciprocal_of_neg_one_seventh_l39_39804

theorem reciprocal_of_neg_one_seventh :
  (∃ x : ℚ, - (1 / 7) * x = 1) → (-7) * (- (1 / 7)) = 1 :=
by
  sorry

end reciprocal_of_neg_one_seventh_l39_39804


namespace putnam_inequality_l39_39103

variable (a x : ℝ)

theorem putnam_inequality (h1 : 0 < x) (h2 : x < a) :
  (a - x)^6 - 3 * a * (a - x)^5 +
  5 / 2 * a^2 * (a - x)^4 -
  1 / 2 * a^4 * (a - x)^2 < 0 :=
by
  sorry

end putnam_inequality_l39_39103


namespace jaymee_older_than_twice_shara_l39_39238

-- Given conditions
def shara_age : ℕ := 10
def jaymee_age : ℕ := 22

-- Theorem to prove how many years older Jaymee is than twice Shara's age
theorem jaymee_older_than_twice_shara : jaymee_age - 2 * shara_age = 2 := by
  sorry

end jaymee_older_than_twice_shara_l39_39238


namespace distinct_triangles_count_l39_39390

def num_combinations (n k : ℕ) : ℕ := n.choose k

def count_collinear_sets_in_grid (grid_size : ℕ) : ℕ :=
  let rows := grid_size
  let cols := grid_size
  let diagonals := 2
  rows + cols + diagonals

noncomputable def distinct_triangles_in_grid (grid_size n k : ℕ) : ℕ :=
  num_combinations n k - count_collinear_sets_in_grid grid_size

theorem distinct_triangles_count :
  distinct_triangles_in_grid 3 9 3 = 76 := 
by 
  sorry

end distinct_triangles_count_l39_39390


namespace find_g_inverse_84_l39_39747

-- Definition of the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- Definition stating the goal
theorem find_g_inverse_84 : g⁻¹ 84 = 3 :=
sorry

end find_g_inverse_84_l39_39747


namespace inequalities_correct_l39_39531

-- Define the basic conditions
variables {a b c d : ℝ}

-- Conditions given in the problem
axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : 0 > c
axiom h4 : c > d

-- Correct answers to be proven
theorem inequalities_correct (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (a + c > b + d) ∧ (a * d^2 > b * c^2) ∧ (1 / (b * c) < 1 / (a * d)) :=
begin
  -- Proof part
  sorry
end

end inequalities_correct_l39_39531


namespace find_cat_video_length_l39_39679

variables (C : ℕ)

def cat_video_length (C : ℕ) : Prop :=
  C + 2 * C + 6 * C = 36

theorem find_cat_video_length : cat_video_length 4 :=
by
  sorry

end find_cat_video_length_l39_39679


namespace least_positive_integer_with_12_factors_l39_39982

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l39_39982


namespace total_amount_spent_l39_39113

noncomputable def value_of_nickel : ℕ := 5
noncomputable def value_of_dime : ℕ := 10
noncomputable def initial_amount : ℕ := 250

def amount_spent_by_Pete (nickels_spent : ℕ) : ℕ :=
  nickels_spent * value_of_nickel

def amount_remaining_with_Raymond (dimes_left : ℕ) : ℕ :=
  dimes_left * value_of_dime

theorem total_amount_spent (nickels_spent : ℕ) (dimes_left : ℕ) :
  (amount_spent_by_Pete nickels_spent + 
   (initial_amount - amount_remaining_with_Raymond dimes_left)) = 200 :=
by
  sorry

end total_amount_spent_l39_39113


namespace find_a_l39_39557

variable (y : ℝ) (a : ℝ)

theorem find_a (hy : y > 0) (h_expr : (a * y / 20) + (3 * y / 10) = 0.7 * y) : a = 8 :=
by
  sorry

end find_a_l39_39557


namespace geometric_sum_4500_l39_39617

theorem geometric_sum_4500 (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300)
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 :=
sorry

end geometric_sum_4500_l39_39617


namespace least_positive_integer_with_12_factors_is_972_l39_39989

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l39_39989


namespace percentage_decrease_of_b_l39_39274

variables (a b x m : ℝ) (p : ℝ)

-- Given conditions
def ratio_ab : Prop := a / b = 4 / 5
def expression_x : Prop := x = 1.25 * a
def expression_m : Prop := m = b * (1 - p / 100)
def ratio_mx : Prop := m / x = 0.6

-- The theorem to be proved
theorem percentage_decrease_of_b 
  (h1 : ratio_ab a b)
  (h2 : expression_x a x)
  (h3 : expression_m b m p)
  (h4 : ratio_mx m x) 
  : p = 40 :=
sorry

end percentage_decrease_of_b_l39_39274


namespace total_wheels_at_park_l39_39340

-- Conditions as definitions
def number_of_adults := 6
def number_of_children := 15
def wheels_per_bicycle := 2
def wheels_per_tricycle := 3

-- To prove: total number of wheels = 57
theorem total_wheels_at_park : 
  (number_of_adults * wheels_per_bicycle) + (number_of_children * wheels_per_tricycle) = 57 :=
by
  sorry

end total_wheels_at_park_l39_39340


namespace gcd_divisibility_and_scaling_l39_39772

theorem gcd_divisibility_and_scaling (a b n : ℕ) (c : ℕ) (h₁ : a ≠ 0) (h₂ : c > 0) (d : ℕ := Nat.gcd a b) :
  (n ∣ a ∧ n ∣ b ↔ n ∣ d) ∧ Nat.gcd (a * c) (b * c) = c * d :=
by 
  sorry

end gcd_divisibility_and_scaling_l39_39772


namespace inequality_problem_l39_39534

variable {a b c d : ℝ}

theorem inequality_problem (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
    a + c > b + d ∧ ad^2 > bc^2 ∧ (1 / bc) < (1 / ad) :=
by
  sorry

end inequality_problem_l39_39534


namespace quadratic_inequality_solution_set_empty_l39_39081

theorem quadratic_inequality_solution_set_empty
  (m : ℝ)
  (h : ∀ x : ℝ, mx^2 - mx - 1 < 0) :
  -4 < m ∧ m < 0 :=
sorry

end quadratic_inequality_solution_set_empty_l39_39081


namespace sum_and_product_formulas_l39_39208

/-- 
Given an arithmetic sequence {a_n} with the sum of the first n terms S_n = 2n^2, 
and in the sequence {b_n}, b_1 = 1 and b_{n+1} = 3b_n (n ∈ ℕ*),
prove that:
(Ⅰ) The general formula for sequences {a_n} is a_n = 4n - 2,
(Ⅱ) The general formula for sequences {b_n} is b_n = 3^{n-1},
(Ⅲ) Let c_n = a_n * b_n, prove that the sum of the first n terms of the sequence {c_n}, denoted as T_n, is T_n = (2n - 2) * 3^n + 2.
-/
theorem sum_and_product_formulas (S_n : ℕ → ℕ) (b : ℕ → ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (T_n : ℕ → ℕ) :
  (∀ n, S_n n = 2 * n^2) →
  (b 1 = 1) →
  (∀ n, b (n + 1) = 3 * (b n)) →
  (∀ n, a n = S_n n - S_n (n - 1)) →
  ∀ n, (T_n n = (2*n - 2) * 3^n + 2) := sorry

end sum_and_product_formulas_l39_39208


namespace krakozyabrs_count_l39_39415

theorem krakozyabrs_count :
  ∀ (K : Type) [fintype K] (has_horns : K → Prop) (has_wings : K → Prop), 
  (∀ k, has_horns k ∨ has_wings k) →
  (∃ n, (∀ k, has_horns k → has_wings k) → fintype.card ({k | has_horns k} : set K) = 5 * n) →
  (∃ n, (∀ k, has_wings k → has_horns k) → fintype.card ({k | has_wings k} : set K) = 4 * n) →
  (∃ T, 25 < T ∧ T < 35 ∧ T = 32) := 
by
  intros K _ has_horns has_wings _ _ _
  sorry

end krakozyabrs_count_l39_39415


namespace contrapositive_true_l39_39872

theorem contrapositive_true (q p : Prop) (h : q → p) : ¬p → ¬q :=
by sorry

end contrapositive_true_l39_39872


namespace least_positive_integer_with_12_factors_l39_39995

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l39_39995


namespace percentage_third_year_students_l39_39563

-- Define the conditions as given in the problem
variables (T : ℝ) (T_3 : ℝ) (S_2 : ℝ)

-- Conditions
def cond1 : Prop := S_2 = 0.10 * T
def cond2 : Prop := (0.10 * T) / (T - T_3) = 1 / 7

-- Define the proof goal
theorem percentage_third_year_students (h1 : cond1 T S_2) (h2 : cond2 T T_3) : T_3 = 0.30 * T :=
sorry

end percentage_third_year_students_l39_39563


namespace coordinates_of_point_A_l39_39789

    theorem coordinates_of_point_A (x y : ℝ) (h1 : y = 0) (h2 : abs x = 3) : (x, y) = (3, 0) ∨ (x, y) = (-3, 0) :=
    sorry
    
end coordinates_of_point_A_l39_39789


namespace smallest_five_digit_number_divisible_by_primes_l39_39692

theorem smallest_five_digit_number_divisible_by_primes : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
begin
  sorry
end

end smallest_five_digit_number_divisible_by_primes_l39_39692


namespace cupcakes_initial_count_l39_39718

theorem cupcakes_initial_count (x : ℕ) (h1 : x - 5 + 10 = 24) : x = 19 :=
by sorry

end cupcakes_initial_count_l39_39718


namespace sheep_problem_system_l39_39407

theorem sheep_problem_system :
  (∃ (x y : ℝ), 5 * x - y = -90 ∧ 50 * x - y = 0) ↔ 
  (5 * x - y = -90 ∧ 50 * x - y = 0) := 
by
  sorry

end sheep_problem_system_l39_39407


namespace bonus_trigger_sales_amount_l39_39017

theorem bonus_trigger_sales_amount (total_sales S : ℝ) (h1 : 0.09 * total_sales = 1260)
  (h2 : 0.03 * (total_sales - S) = 120) : S = 10000 :=
sorry

end bonus_trigger_sales_amount_l39_39017


namespace beef_weight_loss_percentage_l39_39333

theorem beef_weight_loss_percentage (weight_before weight_after weight_lost_percentage : ℝ) 
  (before_process : weight_before = 861.54)
  (after_process : weight_after = 560) 
  (weight_lost : (weight_before - weight_after) = 301.54)
  : weight_lost_percentage = 34.99 :=
by
  sorry

end beef_weight_loss_percentage_l39_39333


namespace quadratic_m_value_l39_39530

theorem quadratic_m_value (m : ℤ) (hm1 : |m| = 2) (hm2 : m ≠ 2) : m = -2 :=
sorry

end quadratic_m_value_l39_39530


namespace largest_of_seven_consecutive_integers_l39_39149

theorem largest_of_seven_consecutive_integers (a : ℕ) (h : a > 0) (sum_eq_77 : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6) = 77)) :
  a + 6 = 14 :=
by
  sorry

end largest_of_seven_consecutive_integers_l39_39149


namespace point_Q_in_first_quadrant_l39_39229

theorem point_Q_in_first_quadrant (a b : ℝ) (h : a < 0 ∧ b < 0) : (0 < -a) ∧ (0 < -b) :=
by
  have ha : -a > 0 := by linarith
  have hb : -b > 0 := by linarith
  exact ⟨ha, hb⟩

end point_Q_in_first_quadrant_l39_39229


namespace find_b_l39_39586

theorem find_b (b : ℤ) (h₁ : b < 0) : (∃ n : ℤ, (x : ℤ) * x + b * x - 36 = (x + n) * (x + n) - 20) → b = -8 :=
by
  intro hX
  sorry

end find_b_l39_39586


namespace smallest_positive_integer_between_101_and_200_l39_39146

theorem smallest_positive_integer_between_101_and_200 :
  ∃ n : ℕ, n > 1 ∧ n % 6 = 1 ∧ n % 7 = 1 ∧ n % 8 = 1 ∧ 101 ≤ n ∧ n ≤ 200 :=
by
  sorry

end smallest_positive_integer_between_101_and_200_l39_39146


namespace intersection_M_N_l39_39888

def M : Set ℝ := { x | x / (x - 1) ≥ 0 }
def N : Set ℝ := { y | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_M_N :
  { x | x / (x - 1) ≥ 0 } ∩ { y | ∃ x : ℝ, y = 3 * x^2 + 1 } = { x | x > 1 } :=
sorry

end intersection_M_N_l39_39888


namespace bijection_lcm_property_l39_39768

noncomputable def bijective_function {n : ℕ} : Fin n → Fin n := sorry

theorem bijection_lcm_property (n : ℕ) (f : Fin n → Fin n) (hf : bijective f) :
  ∃ M : ℕ, M > 0 ∧ ∀ i : Fin n, iterate f M i = f i :=
sorry

end bijection_lcm_property_l39_39768


namespace Kyle_older_than_Julian_l39_39242

variable (Tyson_age : ℕ)
variable (Frederick_age Julian_age Kyle_age : ℕ)

-- Conditions
def condition1 := Tyson_age = 20
def condition2 := Frederick_age = 2 * Tyson_age
def condition3 := Julian_age = Frederick_age - 20
def condition4 := Kyle_age = 25

-- The proof problem (statement only)
theorem Kyle_older_than_Julian :
  Tyson_age = 20 ∧
  Frederick_age = 2 * Tyson_age ∧
  Julian_age = Frederick_age - 20 ∧
  Kyle_age = 25 →
  Kyle_age - Julian_age = 5 := by
  intro h
  sorry

end Kyle_older_than_Julian_l39_39242


namespace price_for_two_bracelets_l39_39473

theorem price_for_two_bracelets
    (total_bracelets : ℕ)
    (price_per_bracelet : ℕ)
    (total_earned_for_single : ℕ)
    (total_earned : ℕ)
    (bracelets_sold_single : ℕ)
    (bracelets_left : ℕ)
    (remaining_earned : ℕ)
    (pairs_sold : ℕ)
    (price_per_pair : ℕ) :
    total_bracelets = 30 →
    price_per_bracelet = 5 →
    total_earned_for_single = 60 →
    total_earned = 132 →
    bracelets_sold_single = total_earned_for_single / price_per_bracelet →
    bracelets_left = total_bracelets - bracelets_sold_single →
    remaining_earned = total_earned - total_earned_for_single →
    pairs_sold = bracelets_left / 2 →
    price_per_pair = remaining_earned / pairs_sold →
    price_per_pair = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end price_for_two_bracelets_l39_39473


namespace mutually_exclusive_not_contradictory_l39_39723

namespace BallProbability
  -- Definitions of events based on the conditions
  def at_least_two_white (outcome : Multiset (String)) : Prop := 
    Multiset.count "white" outcome ≥ 2

  def all_red (outcome : Multiset (String)) : Prop := 
    Multiset.count "red" outcome = 3

  -- Problem statement
  theorem mutually_exclusive_not_contradictory :
    ∀ outcome : Multiset (String),
    Multiset.card outcome = 3 →
    (at_least_two_white outcome → ¬all_red outcome) ∧
    ¬(∀ outcome, at_least_two_white outcome ↔ ¬all_red outcome) := 
  by
    intros
    sorry
end BallProbability

end mutually_exclusive_not_contradictory_l39_39723


namespace f_prime_at_0_l39_39091

noncomputable def a (n : ℕ) : ℝ := 2 * (2/8)^(n-1)

def f (x : ℝ) : ℝ := x * (x - a 1) * (x - a 2) * (x - a 3) * (x - a 4) * (x - a 5) * (x - a 6) * (x - a 7) * (x - a 8)

theorem f_prime_at_0 : deriv f 0 = 4096 := 
  sorry

end f_prime_at_0_l39_39091


namespace computer_additions_per_hour_l39_39486

theorem computer_additions_per_hour : 
  ∀ (initial_rate : ℕ) (increase_rate: ℚ) (intervals_per_hour : ℕ),
  initial_rate = 12000 → 
  increase_rate = 0.05 → 
  intervals_per_hour = 4 → 
  (12000 * 900) + (12000 * 1.05 * 900) + (12000 * 1.05^2 * 900) + (12000 * 1.05^3 * 900) = 46549350 := 
by
  intros initial_rate increase_rate intervals_per_hour h1 h2 h3
  have h4 : initial_rate = 12000 := h1
  have h5 : increase_rate = 0.05 := h2
  have h6 : intervals_per_hour = 4 := h3
  sorry

end computer_additions_per_hour_l39_39486


namespace start_A_to_B_l39_39753

theorem start_A_to_B (x : ℝ)
  (A_to_C : x = 1000 * (1000 / 571.43) - 1000)
  (h1 : 1000 / (1000 - 600) = 1000 / (1000 - 428.57))
  (h2 : x = 1750 - 1000) :
  x = 750 :=
by
  rw [h2]
  sorry   -- Proof to be filled in.

end start_A_to_B_l39_39753


namespace circle_range_of_a_l39_39452

theorem circle_range_of_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * a * x - 4 * y + (a^2 + a) = 0 → (x - h)^2 + (y - k)^2 = r^2) ↔ (a < 4) :=
sorry

end circle_range_of_a_l39_39452


namespace find_a_b_find_A_l39_39100

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 2 * (Real.log x / Real.log 2) ^ 2 + 2 * a * (Real.log (1 / x) / Real.log 2) + b

theorem find_a_b : (∀ x : ℝ, 0 < x → f x a b = 2 * (Real.log x / Real.log 2)^2 + 2 * a * (Real.log (1 / x) / Real.log 2) + b) 
                     → f (1/2) a b = -8 
                     ∧ ∀ x : ℝ, 0 < x → x ≠ 1/2 → f x a b ≥ f (1 / 2) a b
                     → a = -2 ∧ b = -6 := 
sorry

theorem find_A (a b : ℝ) (h₁ : a = -2) (h₂ : b = -6) : 
  { x : ℝ | 0 < x ∧ f x a b > 0 } = {x | 0 < x ∧ (x < 1/8 ∨ x > 2)} :=
sorry

end find_a_b_find_A_l39_39100


namespace cost_of_paint_per_quart_l39_39282

/-- Tommy has a flag that is 5 feet wide and 4 feet tall. 
He needs to paint both sides of the flag. 
A quart of paint covers 4 square feet. 
He spends $20 on paint. 
Prove that the cost of paint per quart is $2. --/
theorem cost_of_paint_per_quart
  (width height : ℕ) (paint_area_per_quart : ℕ) (total_cost : ℕ) (total_area : ℕ) (quarts_needed : ℕ) :
  width = 5 →
  height = 4 →
  paint_area_per_quart = 4 →
  total_cost = 20 →
  total_area = 2 * (width * height) →
  quarts_needed = total_area / paint_area_per_quart →
  total_cost / quarts_needed = 2 := 
by
  intros h_w h_h h_papq h_tc h_ta h_qn
  sorry

end cost_of_paint_per_quart_l39_39282


namespace sum_of_first_15_odd_integers_l39_39309

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l39_39309


namespace total_spent_by_pete_and_raymond_l39_39116

def initial_money_in_cents : ℕ := 250
def pete_spent_in_nickels : ℕ := 4
def nickel_value_in_cents : ℕ := 5
def raymond_dimes_left : ℕ := 7
def dime_value_in_cents : ℕ := 10

theorem total_spent_by_pete_and_raymond : 
  (pete_spent_in_nickels * nickel_value_in_cents) 
  + (initial_money_in_cents - (raymond_dimes_left * dime_value_in_cents)) = 200 := sorry

end total_spent_by_pete_and_raymond_l39_39116


namespace sum_of_first_n_odd_numbers_l39_39294

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l39_39294


namespace probability_satisfies_inequality_l39_39330

/-- Define the conditions for the points (x, y) -/
def within_rectangle (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5

def satisfies_inequality (x y : ℝ) : Prop :=
  x + 2 * y ≤ 6

/-- Compute the probability that a randomly selected point within the rectangle
also satisfies the inequality -/
theorem probability_satisfies_inequality : (∃ p : ℚ, p = 3 / 10) :=
sorry

end probability_satisfies_inequality_l39_39330


namespace first_term_arithmetic_sequence_l39_39924

theorem first_term_arithmetic_sequence
    (a: ℚ)
    (S_n S_2n: ℕ → ℚ)
    (n: ℕ) 
    (h1: ∀ n > 0, S_n n = (n * (2 * a + (n - 1) * 5)) / 2)
    (h2: ∀ n > 0, S_2n (2 * n) = ((2 * n) * (2 * a + ((2 * n) - 1) * 5)) / 2)
    (h3: ∀ n > 0, (S_2n (2 * n)) / (S_n n) = 4) :
  a = 5 / 2 :=
by
  sorry

end first_term_arithmetic_sequence_l39_39924


namespace combination_of_15_3_l39_39894

open Nat

theorem combination_of_15_3 : choose 15 3 = 455 :=
by
  -- The statement describes that the number of ways to choose 3 books out of 15 is 455
  sorry

end combination_of_15_3_l39_39894


namespace P_sufficient_for_Q_P_not_necessary_for_Q_l39_39063

variable (x : ℝ)
def P : Prop := x >= 0
def Q : Prop := 2 * x + 1 / (2 * x + 1) >= 1

theorem P_sufficient_for_Q : P x -> Q x := 
by sorry

theorem P_not_necessary_for_Q : ¬ (Q x -> P x) := 
by sorry

end P_sufficient_for_Q_P_not_necessary_for_Q_l39_39063


namespace wolf_hunger_if_eats_11_kids_l39_39012

variable (p k : ℝ)  -- Define the satiety values of a piglet and a kid.
variable (H : ℝ)    -- Define the satiety threshold for "enough to remove hunger".

-- Conditions from the problem:
def condition1 : Prop := 3 * p + 7 * k < H  -- The wolf feels hungry after eating 3 piglets and 7 kids.
def condition2 : Prop := 7 * p + k > H      -- The wolf suffers from overeating after eating 7 piglets and 1 kid.

-- Statement to prove:
theorem wolf_hunger_if_eats_11_kids (p k H : ℝ) 
  (h1 : condition1 p k H) (h2 : condition2 p k H) : 11 * k < H :=
by
  sorry

end wolf_hunger_if_eats_11_kids_l39_39012


namespace repeating_decimal_product_l39_39047

theorem repeating_decimal_product (x y : ℚ) (h₁ : x = 8 / 99) (h₂ : y = 1 / 3) :
    x * y = 8 / 297 := by
  sorry

end repeating_decimal_product_l39_39047


namespace fraction_of_second_eq_fifth_of_first_l39_39658

theorem fraction_of_second_eq_fifth_of_first 
  (a b x y : ℕ)
  (h1 : y = 40)
  (h2 : x + 35 = 4 * y)
  (h3 : (1 / 5) * x = (a / b) * y) 
  (hb : b ≠ 0):
  a / b = 5 / 8 := by
  sorry

end fraction_of_second_eq_fifth_of_first_l39_39658


namespace sum_geometric_sequence_terms_l39_39614

theorem sum_geometric_sequence_terms (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 := 
by
  sorry

end sum_geometric_sequence_terms_l39_39614


namespace neg_p_iff_a_in_0_1_l39_39734

theorem neg_p_iff_a_in_0_1 (a : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) ∧ (0 < a ∧ a < 1) :=
sorry

end neg_p_iff_a_in_0_1_l39_39734


namespace medium_pizza_promotion_price_l39_39424

-- Define the conditions
def regular_price_medium_pizza : ℝ := 18
def total_savings : ℝ := 39
def number_of_medium_pizzas : ℝ := 3

-- Define the goal
theorem medium_pizza_promotion_price : 
  ∃ P : ℝ, 3 * regular_price_medium_pizza - 3 * P = total_savings ∧ P = 5 := 
by
  sorry

end medium_pizza_promotion_price_l39_39424


namespace surface_area_bound_l39_39145

theorem surface_area_bound
  (a b c d : ℝ)
  (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c) (h4: 0 ≤ d) 
  (h_quad: a + b + c > d) : 
  2 * (a * b + b * c + c * a) ≤ (a + b + c) ^ 2 - (d ^ 2) / 3 :=
sorry

end surface_area_bound_l39_39145


namespace sequence_periodicity_l39_39215

noncomputable def a : ℕ → ℚ
| 0       => 0
| (n + 1) => (a n - 2) / ((5/4) * a n - 2)

theorem sequence_periodicity : a 2017 = 0 := by
  sorry

end sequence_periodicity_l39_39215


namespace fraction_white_surface_area_l39_39834

theorem fraction_white_surface_area : 
  let total_surface_area := 96
  let black_faces_corners := 6
  let black_faces_centers := 6
  let black_faces_total := 12
  let white_faces_total := total_surface_area - black_faces_total
  white_faces_total / total_surface_area = 7 / 8 :=
by
  sorry

end fraction_white_surface_area_l39_39834


namespace geom_seq_S6_l39_39759

theorem geom_seq_S6 :
  ∃ (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ),
  (q = 2) →
  (S 3 = 7) →
  (∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) →
  S 6 = 63 :=
sorry

end geom_seq_S6_l39_39759


namespace least_positive_integer_with_12_factors_is_96_l39_39966

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l39_39966


namespace least_positive_integer_with_12_factors_l39_39978

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l39_39978


namespace rhombus_side_length_l39_39815

theorem rhombus_side_length (total_length : ℕ) (num_sides : ℕ) (h1 : total_length = 32) (h2 : num_sides = 4) :
    total_length / num_sides = 8 :=
by
  -- Proof will be provided here
  sorry

end rhombus_side_length_l39_39815


namespace relationship_a_b_c_l39_39721

noncomputable def a : ℝ := Real.sin (Real.pi / 16)
noncomputable def b : ℝ := 0.25
noncomputable def c : ℝ := 2 * Real.log 2 - Real.log 3

theorem relationship_a_b_c : a < b ∧ b < c :=
by
  sorry

end relationship_a_b_c_l39_39721


namespace determine_x_squared_plus_y_squared_l39_39745

theorem determine_x_squared_plus_y_squared (x y : ℝ) 
(h : (x^2 + y^2 + 2) * (x^2 + y^2 - 3) = 6) : x^2 + y^2 = 4 :=
sorry

end determine_x_squared_plus_y_squared_l39_39745


namespace chess_competition_l39_39752

theorem chess_competition (W M : ℕ) 
  (hW : W * (W - 1) / 2 = 45) 
  (hM : M * 10 = 200) :
  M * (M - 1) / 2 = 190 :=
by
  sorry

end chess_competition_l39_39752


namespace frank_problems_each_type_l39_39675

theorem frank_problems_each_type (bill_total : ℕ) (ryan_ratio bill_total_ratio : ℕ) (frank_ratio ryan_total : ℕ) (types : ℕ)
  (h1 : bill_total = 20)
  (h2 : ryan_ratio = 2)
  (h3 : bill_total_ratio = bill_total * ryan_ratio)
  (h4 : ryan_total = bill_total_ratio)
  (h5 : frank_ratio = 3)
  (h6 : ryan_total * frank_ratio = ryan_total) :
  (ryan_total * frank_ratio) / types = 30 :=
by
  sorry

end frank_problems_each_type_l39_39675


namespace each_girl_gets_2_dollars_after_debt_l39_39942

variable (Lulu_saved : ℕ)
variable (Nora_saved : ℕ)
variable (Tamara_saved : ℕ)
variable (debt : ℕ)
variable (remaining : ℕ)
variable (each_girl_share : ℕ)

-- Conditions
axiom Lulu_saved_cond : Lulu_saved = 6
axiom Nora_saved_cond : Nora_saved = 5 * Lulu_saved
axiom Nora_Tamara_relation : Nora_saved = 3 * Tamara_saved
axiom debt_cond : debt = 40

-- Question == Answer to prove
theorem each_girl_gets_2_dollars_after_debt (total_saved : ℕ) (remaining: ℕ) (each_girl_share: ℕ) :
  total_saved = Tamara_saved + Nora_saved + Lulu_saved →
  remaining = total_saved - debt →
  each_girl_share = remaining / 3 →
  each_girl_share = 2 := 
sorry

end each_girl_gets_2_dollars_after_debt_l39_39942


namespace exists_multiple_sum_divides_l39_39773

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_multiple_sum_divides {n : ℕ} (hn : n > 0) :
  ∃ (n_ast : ℕ), n ∣ n_ast ∧ sum_of_digits n_ast ∣ n_ast :=
by
  sorry

end exists_multiple_sum_divides_l39_39773


namespace inequality_proof_l39_39929

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy: 0 < y) (hz : 0 < z):
  ( ( (x + y + z) / 3 ) ^ (x + y + z) ) ≤ x^x * y^y * z^z ∧ x^x * y^y * z^z ≤ ( (x^2 + y^2 + z^2) / (x + y + z) ) ^ (x + y + z) :=
by
  sorry

end inequality_proof_l39_39929


namespace removed_term_is_a11_l39_39408

noncomputable def sequence_a (n : ℕ) (a1 d : ℤ) := a1 + (n - 1) * d

def sequence_sum (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem removed_term_is_a11 :
  ∃ d : ℤ, ∀ a1 d : ℤ, 
            a1 = -5 ∧ 
            sequence_sum 11 a1 d = 55 ∧ 
            (sequence_sum 11 a1 d - sequence_a 11 a1 d) / 10 = 4 
          → sequence_a 11 a1 d = removed_term :=
sorry

end removed_term_is_a11_l39_39408


namespace time_difference_l39_39780

/-
Malcolm's speed: 5 minutes per mile
Joshua's speed: 7 minutes per mile
Race length: 12 miles
Question: Prove that the time difference between Joshua crossing the finish line after Malcolm is 24 minutes
-/
noncomputable def time_taken (speed: ℕ) (distance: ℕ) : ℕ :=
  speed * distance

theorem time_difference :
  let malcolm_speed := 5
  let joshua_speed := 7
  let race_length := 12
  let malcolm_time := time_taken malcolm_speed race_length
  let joshua_time := time_taken joshua_speed race_length
  malcolm_time < joshua_time →
  joshua_time - malcolm_time = 24 :=
by
  intros malcolm_speed joshua_speed race_length malcolm_time joshua_time malcolm_time_lt_joshua_time
  sorry

end time_difference_l39_39780


namespace sally_pokemon_cards_count_l39_39255

-- Defining the initial conditions
def initial_cards : ℕ := 27
def cards_given_by_dan : ℕ := 41
def cards_bought_by_sally : ℕ := 20

-- Statement of the problem to be proved
theorem sally_pokemon_cards_count :
  initial_cards + cards_given_by_dan + cards_bought_by_sally = 88 := by
  sorry

end sally_pokemon_cards_count_l39_39255


namespace metallic_sheet_dimension_l39_39664

theorem metallic_sheet_dimension
  (length_cut : ℕ) (other_dim : ℕ) (volume : ℕ) (x : ℕ)
  (length_cut_eq : length_cut = 8)
  (other_dim_eq : other_dim = 36)
  (volume_eq : volume = 4800)
  (volume_formula : volume = (x - 2 * length_cut) * (other_dim - 2 * length_cut) * length_cut) :
  x = 46 :=
by
  sorry

end metallic_sheet_dimension_l39_39664


namespace interest_rate_is_10_percent_l39_39944

theorem interest_rate_is_10_percent (P : ℝ) (t : ℝ) (d : ℝ) (r : ℝ) 
  (hP : P = 9999.99999999988) 
  (ht : t = 1) 
  (hd : d = 25)
  : P * (1 + r / 2)^(2 * t) - P - (P * r * t) = d → r = 0.1 :=
by
  intros h
  rw [hP, ht, hd] at h
  sorry

end interest_rate_is_10_percent_l39_39944


namespace max_p_plus_q_l39_39379

theorem max_p_plus_q (p q : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → 2 * p * x^2 + q * x - p + 1 ≥ 0) : p + q ≤ 2 :=
sorry

end max_p_plus_q_l39_39379


namespace percentage_A_is_22_l39_39129

noncomputable def percentage_A_in_mixture : ℝ :=
  (0.8 * 0.20 + 0.2 * 0.30) * 100

theorem percentage_A_is_22 :
  percentage_A_in_mixture = 22 := 
by
  sorry

end percentage_A_is_22_l39_39129


namespace fixed_point_PQ_passes_l39_39410

theorem fixed_point_PQ_passes (P Q : ℝ × ℝ) (x1 x2 : ℝ)
  (hP : P = (x1, x1^2))
  (hQ : Q = (x2, x2^2))
  (hC1 : x1 ≠ 0)
  (hC2 : x2 ≠ 0)
  (hSlopes : (x2 / x2^2 * (2 * x1)) = -2) :
  ∃ D : ℝ × ℝ, D = (0, 1) ∧
    ∀ (x y : ℝ), (y = x1^2 + (x1 - (1 / x1)) * (x - x1)) → ((x, y) = P ∨ (x, y) = Q) := sorry

end fixed_point_PQ_passes_l39_39410


namespace lcm_gcd_product_eq_product_12_15_l39_39366

theorem lcm_gcd_product_eq_product_12_15 :
  lcm 12 15 * gcd 12 15 = 12 * 15 :=
sorry

end lcm_gcd_product_eq_product_12_15_l39_39366


namespace farmer_randy_total_acres_l39_39199

-- Define the conditions
def acres_per_tractor_per_day : ℕ := 68
def tractors_first_2_days : ℕ := 2
def days_first_period : ℕ := 2
def tractors_next_3_days : ℕ := 7
def days_second_period : ℕ := 3

-- Prove the total acres Farmer Randy needs to plant
theorem farmer_randy_total_acres :
  (tractors_first_2_days * acres_per_tractor_per_day * days_first_period) +
  (tractors_next_3_days * acres_per_tractor_per_day * days_second_period) = 1700 :=
by
  -- Here, we would provide the proof, but in this example, we will use sorry.
  sorry

end farmer_randy_total_acres_l39_39199


namespace find_income_l39_39136

def income_and_savings (x : ℕ) : ℕ := 10 * x
def expenditure (x : ℕ) : ℕ := 4 * x
def savings (x : ℕ) : ℕ := income_and_savings x - expenditure x

theorem find_income (savings_eq : 6 * 1900 = 11400) : income_and_savings 1900 = 19000 :=
by
  sorry

end find_income_l39_39136


namespace pete_and_ray_spent_200_cents_l39_39118

-- Define the basic units
def cents_in_a_dollar := 100
def value_of_a_nickel := 5
def value_of_a_dime := 10

-- Define the initial amounts and spending
def pete_initial_amount := 250  -- cents
def ray_initial_amount := 250  -- cents
def pete_spent_nickels := 4 * value_of_a_nickel
def ray_remaining_dimes := 7 * value_of_a_dime

-- Calculate total amounts spent
def total_spent_pete := pete_spent_nickels
def total_spent_ray := ray_initial_amount - ray_remaining_dimes
def total_spent := total_spent_pete + total_spent_ray

-- The proof problem statement
theorem pete_and_ray_spent_200_cents : total_spent = 200 := by {
 sorry
}

end pete_and_ray_spent_200_cents_l39_39118


namespace problem_solution_l39_39896

open Real

/-- If (y / 6) / 3 = 6 / (y / 3), then y is ±18. -/
theorem problem_solution (y : ℝ) (h : (y / 6) / 3 = 6 / (y / 3)) : y = 18 ∨ y = -18 :=
by
  sorry

end problem_solution_l39_39896


namespace reading_homework_is_4_l39_39121

-- Defining the conditions.
variables (R : ℕ)  -- Number of pages of reading homework
variables (M : ℕ)  -- Number of pages of math homework

-- Rachel has 7 pages of math homework.
def math_homework_equals_7 : Prop := M = 7

-- Rachel has 3 more pages of math homework than reading homework.
def math_minus_reads_is_3 : Prop := M = R + 3

-- Prove the number of pages of reading homework is 4.
theorem reading_homework_is_4 (M R : ℕ) 
  (h1 : math_homework_equals_7 M) -- M = 7
  (h2 : math_minus_reads_is_3 M R) -- M = R + 3
  : R = 4 :=
sorry

end reading_homework_is_4_l39_39121


namespace unit_digit_of_product_of_nine_consecutive_numbers_is_zero_l39_39862

theorem unit_digit_of_product_of_nine_consecutive_numbers_is_zero (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7) * (n + 8)) % 10 = 0 :=
by
  sorry

end unit_digit_of_product_of_nine_consecutive_numbers_is_zero_l39_39862


namespace sin_cos_term_side_l39_39731

theorem sin_cos_term_side (a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℝ, (k = 2 * (if a > 0 then -3/5 else 3/5) + (if a > 0 then 4/5 else -4/5)) ∧ (k = 2/5 ∨ k = -2/5) := by
  sorry

end sin_cos_term_side_l39_39731


namespace ab_root_inequality_l39_39267

theorem ab_root_inequality (a b : ℝ) (h1: ∀ x : ℝ, (x + a) * (x + b) = -9) (h2: a < 0) (h3: b < 0) :
  a + b < -6 :=
sorry

end ab_root_inequality_l39_39267


namespace segment_measure_l39_39219

theorem segment_measure (a b : ℝ) (m : ℝ) (h : a = m * b) : (1 / m) * a = b :=
by sorry

end segment_measure_l39_39219


namespace find_value_l39_39527

variable {a b : ℝ}

theorem find_value (h : 2 * a + b + 1 = 0) : 1 + 4 * a + 2 * b = -1 := 
by
  sorry

end find_value_l39_39527


namespace range_of_abs_function_l39_39857

theorem range_of_abs_function : ∀ (y : ℝ), (∃ (x : ℝ), y = |x + 5| - |x - 3|) ↔ y ∈ Set.Icc (-8) 8 :=
by
  sorry

end range_of_abs_function_l39_39857


namespace krakozyabr_count_l39_39419

variable (n H W T : ℕ)
variable (h1 : H = 5 * n) -- 20% of the 'krakozyabrs' with horns also have wings
variable (h2 : W = 4 * n) -- 25% of the 'krakozyabrs' with wings also have horns
variable (h3 : T = H + W - n) -- Total number of 'krakozyabrs' using inclusion-exclusion
variable (h4 : 25 < T)
variable (h5 : T < 35)

theorem krakozyabr_count : T = 32 := by
  sorry

end krakozyabr_count_l39_39419


namespace initial_capital_is_15000_l39_39167

noncomputable def initialCapital (profitIncrease: ℝ) (oldRate newRate: ℝ) (distributionRatio: ℝ) : ℝ :=
  (profitIncrease / ((newRate - oldRate) * distributionRatio))

theorem initial_capital_is_15000 :
  initialCapital 200 0.05 0.07 (2 / 3) = 15000 :=
by
  sorry

end initial_capital_is_15000_l39_39167


namespace sum_of_first_three_terms_l39_39269

theorem sum_of_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 8) 
  (h5 : a 5 = 12) 
  (h6 : a 6 = 16) : 
  a 1 + a 2 + a 3 = 0 :=
by
  sorry

end sum_of_first_three_terms_l39_39269


namespace age_of_oldest_child_l39_39597

def average_age_of_children (a b c d : ℕ) : ℕ := (a + b + c + d) / 4

theorem age_of_oldest_child :
  ∀ (a b c d : ℕ), a = 6 → b = 9 → c = 12 → average_age_of_children a b c d = 9 → d = 9 :=
by
  intros a b c d h_a h_b h_c h_avg
  sorry

end age_of_oldest_child_l39_39597


namespace find_b_l39_39725

theorem find_b 
  (a b : ℚ)
  (h_root : (1 + Real.sqrt 5) ^ 3 + a * (1 + Real.sqrt 5) ^ 2 + b * (1 + Real.sqrt 5) - 60 = 0) :
  b = 26 :=
sorry

end find_b_l39_39725


namespace tax_per_pound_is_one_l39_39569

-- Define the conditions
def bulk_price_per_pound : ℝ := 5          -- Condition 1
def minimum_spend : ℝ := 40               -- Condition 2
def total_paid : ℝ := 240                 -- Condition 4
def excess_pounds : ℝ := 32               -- Condition 5

-- Define the proof problem statement
theorem tax_per_pound_is_one :
  ∃ (T : ℝ), total_paid = (minimum_spend / bulk_price_per_pound + excess_pounds) * bulk_price_per_pound + 
  (minimum_spend / bulk_price_per_pound + excess_pounds) * T ∧ 
  T = 1 :=
by 
  sorry

end tax_per_pound_is_one_l39_39569


namespace min_points_to_win_l39_39231

theorem min_points_to_win : ∀ (points : ℕ), (∀ (race_results : ℕ → ℕ), 
  (points = race_results 1 * 4 + race_results 2 * 2 + race_results 3 * 1) 
  ∧ (∀ i, 1 ≤ race_results i ∧ race_results i ≤ 4) 
  ∧ (∀ i j, i ≠ j → race_results i ≠ race_results j) 
  ∧ (race_results 1 + race_results 2 + race_results 3 = 4)) → (15 ≤ points) :=
by
  sorry

end min_points_to_win_l39_39231


namespace gcd_pens_pencils_l39_39138

theorem gcd_pens_pencils (pens : ℕ) (pencils : ℕ) (h1 : pens = 1048) (h2 : pencils = 828) : Nat.gcd pens pencils = 4 := 
by
  -- Given: pens = 1048 and pencils = 828
  have h : pens = 1048 := h1
  have h' : pencils = 828 := h2
  sorry

end gcd_pens_pencils_l39_39138


namespace probability_at_least_one_boy_one_girl_l39_39339

theorem probability_at_least_one_boy_one_girl :
  (∀ (P : SampleSpace → Prop), (P = (fun outcomes => nat.size outcomes = 4
                            ∧ (∃ outcome : outcomes, outcome = "boy")
                            ∧ ∃ outcome : outcomes, outcome = "girl"))
  -> (probability P = 7/8)) :=
by
  sorry

end probability_at_least_one_boy_one_girl_l39_39339


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l39_39709

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l39_39709


namespace union_set_A_set_B_l39_39774

def set_A : Set ℝ := { x | x^2 - 5 * x - 6 < 0 }
def set_B : Set ℝ := { x | -3 < x ∧ x < 2 }
def set_union (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∨ x ∈ B }

theorem union_set_A_set_B : set_union set_A set_B = { x | -3 < x ∧ x < 6 } := 
by sorry

end union_set_A_set_B_l39_39774


namespace probability_red_then_white_l39_39819

-- Define the total number of balls and the probabilities
def total_balls : ℕ := 9
def red_balls : ℕ := 3
def white_balls : ℕ := 2

-- Define the probabilities
def prob_red : ℚ := red_balls / total_balls
def prob_white : ℚ := white_balls / total_balls

-- Define the combined probability of drawing a red and then a white ball 
theorem probability_red_then_white : (prob_red * prob_white) = 2/27 :=
by
  sorry

end probability_red_then_white_l39_39819


namespace appropriate_sampling_methods_l39_39632
-- Import the entire Mathlib library for broader functionality

-- Define the conditions
def community_high_income_families : ℕ := 125
def community_middle_income_families : ℕ := 280
def community_low_income_families : ℕ := 95
def community_total_households : ℕ := community_high_income_families + community_middle_income_families + community_low_income_families

def student_count : ℕ := 12

-- Define the theorem to be proven
theorem appropriate_sampling_methods :
  (community_total_households = 500 → stratified_sampling) ∧
  (student_count = 12 → random_sampling) :=
by sorry

end appropriate_sampling_methods_l39_39632


namespace total_students_l39_39172

-- Definition of the problem conditions
def buses : ℕ := 18
def seats_per_bus : ℕ := 15
def empty_seats_per_bus : ℕ := 3

-- Formulating the mathematically equivalent proof problem
theorem total_students :
  (buses * (seats_per_bus - empty_seats_per_bus) = 216) :=
by
  sorry

end total_students_l39_39172


namespace right_to_left_evaluation_l39_39912

variable (a b c d : ℝ)

theorem right_to_left_evaluation :
  a / b - c + d = a / (b - c - d) :=
sorry

end right_to_left_evaluation_l39_39912


namespace articles_produced_l39_39900

theorem articles_produced (x y : ℕ) :
  (x * x * x * (1 / (x^2 : ℝ))) = x → (y * y * y * (1 / (x^2 : ℝ))) = (y^3 / x^2 : ℝ) :=
by
  sorry

end articles_produced_l39_39900


namespace sin_lower_bound_lt_l39_39545

theorem sin_lower_bound_lt (a : ℝ) (h : ∃ x : ℝ, Real.sin x < a) : a > -1 :=
sorry

end sin_lower_bound_lt_l39_39545


namespace number_of_students_l39_39954

theorem number_of_students (total_stars : ℕ) (stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : total_stars / stars_per_student = 124 :=
by
  sorry

end number_of_students_l39_39954


namespace distance_missouri_to_new_york_by_car_l39_39604

-- Define the given conditions
def distance_plane : ℝ := 2000
def increase_percentage : ℝ := 0.40
def midway_factor : ℝ := 0.5

-- Define the problem to be proven
theorem distance_missouri_to_new_york_by_car :
  let total_distance : ℝ := distance_plane + (distance_plane * increase_percentage)
  let missouri_to_new_york_distance : ℝ := total_distance * midway_factor
  missouri_to_new_york_distance = 1400 :=
by
  sorry

end distance_missouri_to_new_york_by_car_l39_39604


namespace range_of_k_l39_39884

theorem range_of_k (k : ℝ) : (∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 - 2*k*x + k)) ↔ (k ∈ Set.Iic 0 ∨ k ∈ Set.Ici 1) :=
by
  sorry

end range_of_k_l39_39884


namespace size_relationship_l39_39818

variable (a1 a2 b1 b2 : ℝ)

theorem size_relationship (h1 : a1 < a2) (h2 : b1 < b2) : a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 := 
sorry

end size_relationship_l39_39818


namespace probability_complement_A_l39_39070

variables {Ω : Type*} [MeasurableSpace Ω] {P : ProbabilityMeasure Ω}
variables (A B : Set Ω)

-- Conditions
def mutually_exclusive (A B : Set Ω) : Prop := ∀ ω, ω ∈ A → ω ∉ B

theorem probability_complement_A :
  mutually_exclusive A B →
  P.probability (A ∪ B) = 0.8 →
  P.probability B = 0.3 →
  P.probability (Aᶜ) = 0.5 :=
by
  sorry

end probability_complement_A_l39_39070


namespace sin_cos_ratio_l39_39203

theorem sin_cos_ratio (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2)
  (h2 : Real.tan (α - β) = 3) : 
  Real.sin (2 * α) / Real.cos (2 * β) = (Real.sqrt 5 + 3 * Real.sqrt 2) / 20 := 
by
  sorry

end sin_cos_ratio_l39_39203


namespace asymptotes_of_hyperbola_l39_39406

theorem asymptotes_of_hyperbola (a : ℝ) :
  (∃ x y : ℝ, y^2 = 12 * x ∧ (x = 3) ∧ (y = 0)) →
  (a^2 = 9) →
  (∀ b c : ℝ, (b, c) ∈ ({(a, b) | (b = a/3 ∨ b = -a/3)})) :=
by
  intro h_focus_coincides vertex_condition
  sorry

end asymptotes_of_hyperbola_l39_39406


namespace Missouri_to_NewYork_by_car_l39_39605

def distance_plane : ℝ := 2000
def increase_percentage : ℝ := 0.40
def total_distance_car : ℝ := distance_plane * (1 + increase_percentage)
def distance_midway : ℝ := total_distance_car / 2

theorem Missouri_to_NewYork_by_car : distance_midway = 1400 := by
  sorry

end Missouri_to_NewYork_by_car_l39_39605


namespace totalSolutions_l39_39892

noncomputable def systemOfEquations (a b c d a1 b1 c1 d1 x y : ℝ) : Prop :=
  a * x^2 + b * x * y + c * y^2 = d ∧ a1 * x^2 + b1 * x * y + c1 * y^2 = d1

theorem totalSolutions 
  (a b c d a1 b1 c1 d1 : ℝ) 
  (h₀ : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)
  (h₁ : a1 ≠ 0 ∨ b1 ≠ 0 ∨ c1 ≠ 0) :
  ∃ x y : ℝ, systemOfEquations a b c d a1 b1 c1 d1 x y :=
sorry

end totalSolutions_l39_39892


namespace Brazil_wins_10_l39_39411

/-- In the year 3000, the World Hockey Championship will follow new rules: 12 points will be awarded for a win, 
5 points will be deducted for a loss, and no points will be awarded for a draw. If the Brazilian team plays 
38 matches, scores 60 points, and loses at least once, then the number of wins they can achieve is 10. 
List all possible scenarios and justify why there cannot be any others. -/
theorem Brazil_wins_10 (x y z : ℕ) 
    (h1: x + y + z = 38) 
    (h2: 12 * x - 5 * y = 60) 
    (h3: y ≥ 1)
    (h4: z ≥ 0): 
  x = 10 :=
by
  sorry

end Brazil_wins_10_l39_39411


namespace cuboid_count_l39_39221

def length_small (m : ℕ) : ℕ := 6
def width_small (m : ℕ) : ℕ := 4
def height_small (m : ℕ) : ℕ := 3

def length_large (m : ℕ): ℕ := 18
def width_large (m : ℕ) : ℕ := 15
def height_large (m : ℕ) : ℕ := 2

def volume (l : ℕ) (w : ℕ) (h : ℕ) : ℕ := l * w * h

def n_small_cuboids (v_large v_small : ℕ) : ℕ := v_large / v_small

theorem cuboid_count : 
  n_small_cuboids (volume (length_large 1) (width_large 1) (height_large 1)) (volume (length_small 1) (width_small 1) (height_small 1)) = 7 :=
by
  sorry

end cuboid_count_l39_39221


namespace problem_solution_l39_39194

def M : Set ℝ := { x | x < 2 }
def N : Set ℝ := { x | 0 < x ∧ x < 1 }
def complement_N : Set ℝ := { x | x ≤ 0 ∨ x ≥ 1 }

theorem problem_solution : M ∪ complement_N = Set.univ := 
sorry

end problem_solution_l39_39194


namespace probability_second_year_not_science_l39_39089

def total_students := 2000

def first_year := 600
def first_year_science := 300
def first_year_arts := 200
def first_year_engineering := 100

def second_year := 450
def second_year_science := 250
def second_year_arts := 150
def second_year_engineering := 50

def third_year := 550
def third_year_science := 300
def third_year_arts := 200
def third_year_engineering := 50

def postgraduate := 400
def postgraduate_science := 200
def postgraduate_arts := 100
def postgraduate_engineering := 100

def not_third_year_not_science :=
  (first_year_arts + first_year_engineering) +
  (second_year_arts + second_year_engineering) +
  (postgraduate_arts + postgraduate_engineering)

def second_year_not_science := second_year_arts + second_year_engineering

theorem probability_second_year_not_science :
  (second_year_not_science / not_third_year_not_science : ℚ) = (2 / 7 : ℚ) :=
by
  let total := (first_year_arts + first_year_engineering) + (second_year_arts + second_year_engineering) + (postgraduate_arts + postgraduate_engineering)
  have not_third_year_not_science : total = 300 + 200 + 200 := by sorry
  have second_year_not_science_eq : second_year_not_science = 200 := by sorry
  sorry

end probability_second_year_not_science_l39_39089


namespace general_equation_M_range_distance_D_to_l_l39_39566

noncomputable def parametric_to_general (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := 2 * Real.sin θ
  x^2 + y^2 / 4 = 1

noncomputable def distance_range (θ : ℝ) : Prop :=
  let x := Real.cos θ
  let y := 2 * Real.sin θ
  let l := x + y - 4
  let d := |x + 2 * y - 4| / Real.sqrt 2
  let min_dist := (4 * Real.sqrt 2 - Real.sqrt 10) / 2
  let max_dist := (4 * Real.sqrt 2 + Real.sqrt 10) / 2
  min_dist ≤ d ∧ d ≤ max_dist

theorem general_equation_M (θ : ℝ) : parametric_to_general θ := sorry

theorem range_distance_D_to_l (θ : ℝ) : distance_range θ := sorry

end general_equation_M_range_distance_D_to_l_l39_39566


namespace sum_first_15_odd_integers_l39_39293

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l39_39293


namespace extreme_points_l39_39883

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 / (2 * x)) - a * x^2 + x

theorem extreme_points (
  a : ℝ
) (h : 0 < a ∧ a < (1 : ℝ) / 8) :
  ∃ x1 x2 : ℝ, f a x1 + f a x2 > 3 - 4 * Real.log 2 :=
sorry

end extreme_points_l39_39883


namespace smallest_k_for_quadratic_l39_39904

noncomputable def quadratic_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k in
  let b := -3 in
  let c := -9 / 4 in
  b^2 - 4*a*c > 0

theorem smallest_k_for_quadratic : 
  ∃ k : ℤ, quadratic_has_two_distinct_real_roots k ∧ k > -1 ∧ k ≠ 0 ∧ ∀ m : ℤ, quadratic_has_two_distinct_real_roots m → m > -1 → m ≠ 0 → k ≤ m :=
sorry

end smallest_k_for_quadratic_l39_39904


namespace domain_range_of_g_l39_39835

variable (f : ℝ → ℝ)
variable (dom_f : Set.Icc 1 3)
variable (rng_f : Set.Icc 0 1)
variable (g : ℝ → ℝ)
variable (g_eq : ∀ x, g x = 2 - f (x - 1))

theorem domain_range_of_g :
  (Set.Icc 2 4) = { x | ∃ y, x = y ∧ g y = (g y) } ∧ Set.Icc 1 2 = { z | ∃ w, z = g w} :=
  sorry

end domain_range_of_g_l39_39835


namespace diff_PA_AQ_const_l39_39684

open Real

def point := (ℝ × ℝ)

noncomputable def distance (p1 p2 : point) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem diff_PA_AQ_const (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  let P := (0, -sqrt 2)
  let Q := (0, sqrt 2)
  let A := (a, sqrt (a^2 + 1))
  distance P A - distance A Q = 2 := 
sorry

end diff_PA_AQ_const_l39_39684


namespace quarters_percentage_value_l39_39008

theorem quarters_percentage_value (dimes quarters : Nat) (value_dime value_quarter : Nat) (total_value quarter_value : Nat)
(h_dimes : dimes = 30)
(h_quarters : quarters = 40)
(h_value_dime : value_dime = 10)
(h_value_quarter : value_quarter = 25)
(h_total_value : total_value = dimes * value_dime + quarters * value_quarter)
(h_quarter_value : quarter_value = quarters * value_quarter) :
(quarter_value : ℚ) / (total_value : ℚ) * 100 = 76.92 := 
sorry

end quarters_percentage_value_l39_39008


namespace initial_average_l39_39241

theorem initial_average (A : ℝ) (h : (15 * A + 14 * 15) / 15 = 54) : A = 40 :=
by
  sorry

end initial_average_l39_39241


namespace smallest_five_digit_number_divisible_by_primes_l39_39694

theorem smallest_five_digit_number_divisible_by_primes : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
begin
  sorry
end

end smallest_five_digit_number_divisible_by_primes_l39_39694


namespace amy_local_calls_l39_39823

theorem amy_local_calls (L I : ℕ) 
  (h1 : 2 * L = 5 * I)
  (h2 : 3 * L = 5 * (I + 3)) : 
  L = 15 :=
by
  sorry

end amy_local_calls_l39_39823


namespace sum_of_integers_between_neg20_5_and_10_5_l39_39290

noncomputable def sum_arithmetic_series (a l n : ℤ) : ℤ :=
  n * (a + l) / 2

theorem sum_of_integers_between_neg20_5_and_10_5 :
  (sum_arithmetic_series (-20) 10 31) = -155 := by
  sorry

end sum_of_integers_between_neg20_5_and_10_5_l39_39290


namespace total_birds_on_fence_l39_39170

-- Definitions based on conditions.
def initial_birds : ℕ := 12
def additional_birds : ℕ := 8

-- Theorem corresponding to the problem statement.
theorem total_birds_on_fence : initial_birds + additional_birds = 20 := by 
  sorry

end total_birds_on_fence_l39_39170


namespace square_difference_example_l39_39001

theorem square_difference_example : 601^2 - 599^2 = 2400 := 
by sorry

end square_difference_example_l39_39001


namespace matrix_inverse_l39_39364

-- Define the given matrix
def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![5, 4], ![-2, 8]]

-- Define the expected inverse matrix
def A_inv_expected : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1/6, -1/12], ![1/24, 5/48]]

-- The main statement: Prove that the inverse of A is equal to the expected inverse
theorem matrix_inverse :
  A⁻¹ = A_inv_expected := sorry

end matrix_inverse_l39_39364


namespace jenny_cases_l39_39097

theorem jenny_cases (total_boxes cases_per_box : ℕ) (h1 : total_boxes = 24) (h2 : cases_per_box = 8) :
  total_boxes / cases_per_box = 3 := by
  sorry

end jenny_cases_l39_39097


namespace units_digit_sum_l39_39367

def base8_to_base10 (n : Nat) : Nat :=
  let units := n % 10
  let tens := (n / 10) % 10
  tens * 8 + units

theorem units_digit_sum (n1 n2 : Nat) (h1 : n1 = 45) (h2 : n2 = 67) : ((base8_to_base10 n1) + (base8_to_base10 n2)) % 8 = 4 := by
  sorry

end units_digit_sum_l39_39367


namespace work_completion_days_l39_39824

theorem work_completion_days (Ry : ℝ) (R_combined : ℝ) (D : ℝ) :
  Ry = 1 / 40 ∧ R_combined = 1 / 13.333333333333332 → 1 / D + Ry = R_combined → D = 20 :=
by
  intros h_eqs h_combined
  sorry

end work_completion_days_l39_39824


namespace allen_change_l39_39847

-- Define the cost per box and the number of boxes
def cost_per_box : ℕ := 7
def num_boxes : ℕ := 5

-- Define the total cost including the tip
def total_cost := num_boxes * cost_per_box
def tip := total_cost / 7
def total_paid := total_cost + tip

-- Define the amount given to the delivery person
def amount_given : ℕ := 100

-- Define the change received
def change := amount_given - total_paid

-- The statement to prove
theorem allen_change : change = 60 :=
by
  -- sorry is used here to skip the proof, as per the instruction
  sorry

end allen_change_l39_39847


namespace not_divisible_by_10100_l39_39119

theorem not_divisible_by_10100 (n : ℕ) : (3^n + 1) % 10100 ≠ 0 := 
by 
  sorry

end not_divisible_by_10100_l39_39119


namespace quadratic_has_real_roots_find_specific_k_l39_39885

-- Part 1: Prove the range of values for k
theorem quadratic_has_real_roots (k : ℝ) : (k ≥ 2) ↔ ∃ x1 x2 : ℝ, x1 ^ 2 - 4 * x1 - 2 * k + 8 = 0 ∧ x2 ^ 2 - 4 * x2 - 2 * k + 8 = 0 := 
sorry

-- Part 2: Prove the specific value of k given the additional condition
theorem find_specific_k (k : ℝ) (x1 x2 : ℝ) : (x1 ^ 3 * x2 + x1 * x2 ^ 3 = 24) ∧ x1 ^ 2 - 4 * x1 - 2 * k + 8 = 0 ∧ x2 ^ 2 - 4 * x2 - 2 * k + 8 = 0 → k = 3 :=
sorry

end quadratic_has_real_roots_find_specific_k_l39_39885


namespace ms_brown_expects_8100_tulips_l39_39111

def steps_length := 3
def width_steps := 18
def height_steps := 25
def tulips_per_sqft := 2

def width_feet := width_steps * steps_length
def height_feet := height_steps * steps_length
def area_feet := width_feet * height_feet
def expected_tulips := area_feet * tulips_per_sqft

theorem ms_brown_expects_8100_tulips :
  expected_tulips = 8100 := by
  sorry

end ms_brown_expects_8100_tulips_l39_39111


namespace james_meditation_time_is_30_l39_39236

noncomputable def james_meditation_time_per_session 
  (sessions_per_day : ℕ) 
  (days_per_week : ℕ) 
  (hours_per_week : ℕ) 
  (minutes_per_hour : ℕ) : ℕ :=
  (hours_per_week * minutes_per_hour) / (sessions_per_day * days_per_week)

theorem james_meditation_time_is_30
  (sessions_per_day : ℕ) 
  (days_per_week : ℕ) 
  (hours_per_week : ℕ) 
  (minutes_per_hour : ℕ) 
  (h_sessions : sessions_per_day = 2) 
  (h_days : days_per_week = 7) 
  (h_hours : hours_per_week = 7) 
  (h_minutes : minutes_per_hour = 60) : 
  james_meditation_time_per_session sessions_per_day days_per_week hours_per_week minutes_per_hour = 30 := by
  sorry

end james_meditation_time_is_30_l39_39236


namespace b_n_expression_l39_39211

-- Define sequence a_n as an arithmetic sequence with given conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + d * (n - 1)

-- Define the conditions for the sequence a_n
def a_conditions (a : ℕ → ℤ) : Prop :=
  a 2 = 8 ∧ a 8 = 26

-- Define the new sequence b_n based on the terms of a_n
def b (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  a (3^n)

theorem b_n_expression (a : ℕ → ℤ) (n : ℕ)
  (h_arith : is_arithmetic_sequence a)
  (h_conditions : a_conditions a) :
  b a n = 3^(n + 1) + 2 := 
sorry

end b_n_expression_l39_39211


namespace decreasing_omega_range_l39_39398

open Real

theorem decreasing_omega_range {ω : ℝ} (h1 : 1 < ω) :
  (∀ x y : ℝ, π ≤ x ∧ x ≤ y ∧ y ≤ (5 * π) / 4 → 
    (|sin (ω * y + π / 3)| ≤ |sin (ω * x + π / 3)|)) → 
  (7 / 6 ≤ ω ∧ ω ≤ 4 / 3) :=
by
  sorry

end decreasing_omega_range_l39_39398


namespace find_slope_l39_39581

theorem find_slope (m : ℝ) : 
    (∀ x : ℝ, (2, 13) = (x, 5 * x + 3)) → 
    (∀ x : ℝ, (2, 13) = (x, m * x + 1)) → 
    m = 6 :=
by 
  intros hP hQ
  have h_inter_p := hP 2
  have h_inter_q := hQ 2
  simp at h_inter_p h_inter_q
  have : 13 = 5 * 2 + 3 := h_inter_p
  have : 13 = m * 2 + 1 := h_inter_q
  linarith

end find_slope_l39_39581


namespace number_of_cows_l39_39392

-- Define the total number of legs and number of legs per cow
def total_legs : ℕ := 460
def legs_per_cow : ℕ := 4

-- Mathematical proof problem as a Lean 4 statement
theorem number_of_cows : total_legs / legs_per_cow = 115 := by
  -- This is the proof statement place. We use 'sorry' as a placeholder for the actual proof.
  sorry

end number_of_cows_l39_39392


namespace weight_of_b_l39_39796

variable (Wa Wb Wc: ℝ)

-- Conditions
def avg_weight_abc : Prop := (Wa + Wb + Wc) / 3 = 45
def avg_weight_ab : Prop := (Wa + Wb) / 2 = 40
def avg_weight_bc : Prop := (Wb + Wc) / 2 = 43

-- Theorem to prove
theorem weight_of_b (Wa Wb Wc: ℝ) (h_avg_abc : avg_weight_abc Wa Wb Wc)
  (h_avg_ab : avg_weight_ab Wa Wb) (h_avg_bc : avg_weight_bc Wb Wc) : Wb = 31 :=
by
  sorry

end weight_of_b_l39_39796


namespace smallest_five_digit_divisible_by_primes_l39_39710

theorem smallest_five_digit_divisible_by_primes : 
  let primes := [2, 3, 5, 7, 11] in
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11))) in
  let five_digit_threshold := 10000 in
  ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
by
  let primes := [2, 3, 5, 7, 11]
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))
  have lcm_2310 : lcm_primes = 2310 := sorry
  let five_digit_threshold := 10000
  have exists_n : ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
    sorry
  exists_intro 5
  have 5_condition : 5 > 0 := sorry
  have 2310_5_condition : 2310 * 5 >= five_digit_threshold := sorry
  have answer : 2310 * 5 = 11550 := sorry
  exact  ⟨5, 5_condition, 2310_5_condition, answer⟩
  exact ⟨5, 5 > 0, 2310 * 5 ≥ 10000, 2310 * 5 = 11550⟩
  sorry

end smallest_five_digit_divisible_by_primes_l39_39710


namespace find_m_l39_39799

-- Define the conditions
variables {m x1 x2 : ℝ}

-- Given the equation x^2 + mx - 1 = 0 has roots x1 and x2:
-- The sum of the roots x1 + x2 is -m, and the product of the roots x1 * x2 is -1.
-- Furthermore, given that 1/x1 + 1/x2 = -3,
-- Prove that m = -3.

theorem find_m :
  (x1 + x2 = -m) →
  (x1 * x2 = -1) →
  (1 / x1 + 1 / x2 = -3) →
  m = -3 := by
  intros hSum hProd hRecip
  sorry

end find_m_l39_39799


namespace students_making_stars_l39_39956

theorem students_making_stars (total_stars stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : 
  total_stars / stars_per_student = 124 :=
by
  sorry

end students_making_stars_l39_39956


namespace daily_shampoo_usage_l39_39125

theorem daily_shampoo_usage
  (S : ℝ)
  (h1 : ∀ t : ℝ, t = 14 → 14 * S + 14 * (S / 2) = 21) :
  S = 1 := by
  sorry

end daily_shampoo_usage_l39_39125


namespace compare_expressions_l39_39645

theorem compare_expressions (n : ℕ) (hn : 0 < n):
  (n ≤ 48 ∧ 99^n + 100^n > 101^n) ∨ (n > 48 ∧ 99^n + 100^n < 101^n) :=
sorry  -- Proof is omitted.

end compare_expressions_l39_39645


namespace number_of_small_jars_l39_39011

theorem number_of_small_jars (S L : ℕ) (h1 : S + L = 100) (h2 : 3 * S + 5 * L = 376) : S = 62 := 
sorry

end number_of_small_jars_l39_39011


namespace total_cost_bicycle_helmet_l39_39015

-- Let h represent the cost of the helmet
def helmet_cost := 40

-- Let b represent the cost of the bicycle
def bicycle_cost := 5 * helmet_cost

-- We need to prove that the total cost (bicycle + helmet) is equal to 240
theorem total_cost_bicycle_helmet : bicycle_cost + helmet_cost = 240 := 
by
  -- This will skip the proof, we only need the statement
  sorry

end total_cost_bicycle_helmet_l39_39015


namespace color_triplet_exists_l39_39689

theorem color_triplet_exists (color : ℕ → Prop) :
  (∀ n, color n ∨ ¬ color n) → ∃ x y z : ℕ, (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ color x = color y ∧ color y = color z ∧ x * y = z ^ 2 :=
by
  sorry

end color_triplet_exists_l39_39689


namespace repeating_decimal_product_l39_39048

theorem repeating_decimal_product (x y : ℚ) (h₁ : x = 8 / 99) (h₂ : y = 1 / 3) :
    x * y = 8 / 297 := by
  sorry

end repeating_decimal_product_l39_39048


namespace books_remaining_in_library_l39_39153

def initial_books : ℕ := 250
def books_taken_out_Tuesday : ℕ := 120
def books_returned_Wednesday : ℕ := 35
def books_withdrawn_Thursday : ℕ := 15

theorem books_remaining_in_library :
  initial_books
  - books_taken_out_Tuesday
  + books_returned_Wednesday
  - books_withdrawn_Thursday = 150 :=
by
  sorry

end books_remaining_in_library_l39_39153


namespace original_price_l39_39319

-- Definitions based on the problem conditions
variables (P : ℝ)

def john_payment (P : ℝ) : ℝ :=
  0.9 * P + 0.15 * P

def jane_payment (P : ℝ) : ℝ :=
  0.9 * P + 0.15 * (0.9 * P)

def price_difference (P : ℝ) : ℝ :=
  john_payment P - jane_payment P

theorem original_price (h : price_difference P = 0.51) : P = 34 := 
by
  sorry

end original_price_l39_39319


namespace smallest_positive_five_digit_number_divisible_by_first_five_primes_l39_39705

theorem smallest_positive_five_digit_number_divisible_by_first_five_primes :
  ∃ n : ℕ, (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ 10000 ≤ n ∧ n = 11550 :=
by
  use 11550
  split
  · intros p hp
    fin_cases hp <;> norm_num
  split
  · norm_num
  rfl

end smallest_positive_five_digit_number_divisible_by_first_five_primes_l39_39705


namespace trapezoid_perimeter_l39_39852

noncomputable def isosceles_trapezoid_perimeter (R : ℝ) (α : ℝ) (hα : α < π / 2) : ℝ :=
  8 * R / (Real.sin α)

theorem trapezoid_perimeter (R : ℝ) (α : ℝ) (hα : α < π / 2) :
  ∃ (P : ℝ), P = isosceles_trapezoid_perimeter R α hα := by
    sorry

end trapezoid_perimeter_l39_39852


namespace cameron_list_length_l39_39028

-- Definitions of multiples
def smallest_multiple_perfect_square := 900
def smallest_multiple_perfect_cube := 27000
def multiple_of_30 (n : ℕ) : Prop := n % 30 = 0

-- Problem statement
theorem cameron_list_length :
  ∀ n, 900 ≤ n ∧ n ≤ 27000 ∧ multiple_of_30 n ->
  (871 = (900 - 30 + 1)) :=
sorry

end cameron_list_length_l39_39028


namespace proof_6_times_15_times_5_eq_2_l39_39822

noncomputable def given_condition (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

theorem proof_6_times_15_times_5_eq_2 : 
  given_condition 6 15 5 → 6 * 15 * 5 = 2 :=
by
  sorry

end proof_6_times_15_times_5_eq_2_l39_39822


namespace positive_integer_solutions_equation_l39_39515

theorem positive_integer_solutions_equation (x y : ℕ) (positive_x : x > 0) (positive_y : y > 0) :
  x^2 + 6 * x * y - 7 * y^2 = 2009 ↔ (x = 252 ∧ y = 251) ∨ (x = 42 ∧ y = 35) ∨ (x = 42 ∧ y = 1) :=
sorry

end positive_integer_solutions_equation_l39_39515


namespace solve_for_x_l39_39446

theorem solve_for_x (n m x : ℕ) (h1 : 5 / 7 = n / 91) (h2 : 5 / 7 = (m + n) / 105) (h3 : 5 / 7 = (x - m) / 140) :
    x = 110 :=
sorry

end solve_for_x_l39_39446


namespace find_t_l39_39746

theorem find_t (s t : ℤ) (h1 : 9 * s + 5 * t = 108) (h2 : s = t - 2) : t = 9 :=
sorry

end find_t_l39_39746


namespace carousel_ratio_l39_39450

theorem carousel_ratio (P : ℕ) (h : 3 + P + 2*P + P/3 = 33) : P / 3 = 3 := 
by 
  sorry

end carousel_ratio_l39_39450


namespace product_of_repeating_decimal_l39_39040

theorem product_of_repeating_decimal 
  (t : ℚ) 
  (h : t = 456 / 999) : 
  8 * t = 1216 / 333 :=
by
  sorry

end product_of_repeating_decimal_l39_39040


namespace rain_in_first_hour_l39_39572

theorem rain_in_first_hour (x : ℝ) (h1 : ∀ y : ℝ, y = 2 * x + 7) (h2 : x + (2 * x + 7) = 22) : x = 5 :=
sorry

end rain_in_first_hour_l39_39572


namespace prove_percent_liquid_X_in_new_solution_l39_39851

variable (initial_solution total_weight_x total_weight_y total_weight_new)

def percent_liquid_X_in_new_solution : Prop :=
  let liquid_X_in_initial := 0.45 * 12
  let water_in_initial := 0.55 * 12
  let remaining_liquid_X := liquid_X_in_initial
  let remaining_water := water_in_initial - 5
  let liquid_X_in_added := 0.45 * 7
  let water_in_added := 0.55 * 7
  let total_liquid_X := remaining_liquid_X + liquid_X_in_added
  let total_water := remaining_water + water_in_added
  let total_weight := total_liquid_X + total_water
  (total_liquid_X / total_weight) * 100 = 61.07

theorem prove_percent_liquid_X_in_new_solution :
  percent_liquid_X_in_new_solution := by
  sorry

end prove_percent_liquid_X_in_new_solution_l39_39851


namespace negation_of_proposition_l39_39139

theorem negation_of_proposition :
  (¬ (∃ x₀ : ℝ, x₀ > 2 ∧ x₀^3 - 2 * x₀^2 < 0)) ↔ (∀ x : ℝ, x > 2 → x^3 - 2 * x^2 ≥ 0) := by
  sorry

end negation_of_proposition_l39_39139


namespace magnitude_b_magnitude_c_area_l39_39402

-- Define the triangle ABC and parameters
variables {A B C : ℝ} {a b c : ℝ}
variables (A_pos : 0 < A) (A_lt_pi_div2 : A < Real.pi / 2)
variables (triangle_condition : a = Real.sqrt 15) (sin_A : Real.sin A = 1 / 4)

-- Problem 1
theorem magnitude_b (cos_B : Real.cos B = Real.sqrt 5 / 3) :
  b = (8 * Real.sqrt 15) / 3 := by
  sorry

-- Problem 2
theorem magnitude_c_area (b_eq_4a : b = 4 * a) :
  c = 15 ∧ (1 / 2 * b * c * Real.sin A = (15 / 2) * Real.sqrt 15) := by
  sorry

end magnitude_b_magnitude_c_area_l39_39402


namespace pizza_eaten_after_six_trips_l39_39649

theorem pizza_eaten_after_six_trips :
  (1 / 3) + (1 / 3) / 2 + (1 / 3) / 2 / 2 + (1 / 3) / 2 / 2 / 2 + (1 / 3) / 2 / 2 / 2 / 2 + (1 / 3) / 2 / 2 / 2 / 2 / 2 = 21 / 32 :=
by
  sorry

end pizza_eaten_after_six_trips_l39_39649


namespace find_number_l39_39314

theorem find_number (x : ℝ) (h : 4 * (x - 220) = 320) : (5 * x) / 3 = 500 :=
by
  sorry

end find_number_l39_39314


namespace smallest_four_digit_remainder_l39_39521

theorem smallest_four_digit_remainder :
  ∃ N : ℕ, (N % 6 = 5) ∧ (1000 ≤ N ∧ N ≤ 9999) ∧ (∀ M : ℕ, (M % 6 = 5) ∧ (1000 ≤ M ∧ M ≤ 9999) → N ≤ M) ∧ N = 1001 :=
by
  sorry

end smallest_four_digit_remainder_l39_39521


namespace quadratic_inequality_solution_l39_39879

theorem quadratic_inequality_solution {a b : ℝ} 
  (h1 : (∀ x : ℝ, ax^2 - bx - 1 ≥ 0 ↔ (x = 1/3 ∨ x = 1/2))) : 
  ∃ a b : ℝ, (∀ x : ℝ, x^2 - b * x - a < 0 ↔ (-3 < x ∧ x < -2)) :=
by
  sorry

end quadratic_inequality_solution_l39_39879


namespace number_of_valid_rods_l39_39764

theorem number_of_valid_rods : ∃ n, n = 22 ∧
  (∀ (d : ℕ), 1 < d ∧ d < 25 ∧ d ≠ 4 ∧ d ≠ 9 ∧ d ≠ 12 → d ∈ {d | d > 0}) :=
by
  use 22
  sorry

end number_of_valid_rods_l39_39764


namespace sum_geometric_sequence_terms_l39_39613

theorem sum_geometric_sequence_terms (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 := 
by
  sorry

end sum_geometric_sequence_terms_l39_39613


namespace fabric_nguyen_needs_l39_39784

-- Definitions for conditions
def fabric_per_pant : ℝ := 8.5
def total_pants : ℝ := 7
def yards_to_feet (yards : ℝ) : ℝ := yards * 3
def fabric_nguyen_has_yards : ℝ := 3.5

-- The proof we need to establish
theorem fabric_nguyen_needs : (total_pants * fabric_per_pant) - (yards_to_feet fabric_nguyen_has_yards) = 49 :=
by
  sorry

end fabric_nguyen_needs_l39_39784


namespace find_y_l39_39906

theorem find_y (x y : ℤ) (h1 : x + y = 250) (h2 : x - y = 200) : y = 25 :=
by
  sorry

end find_y_l39_39906


namespace intended_profit_l39_39838

variables (C P : ℝ)

theorem intended_profit (L S : ℝ) (h1 : L = C * (1 + P)) (h2 : S = 0.90 * L) (h3 : S = 1.17 * C) :
  P = 0.3 + 1 / 3 :=
by
  sorry

end intended_profit_l39_39838


namespace zero_is_a_root_of_polynomial_l39_39868

theorem zero_is_a_root_of_polynomial :
  (12 * (0 : ℝ)^4 + 38 * (0)^3 - 51 * (0)^2 + 40 * (0) = 0) :=
by simp

end zero_is_a_root_of_polynomial_l39_39868


namespace probability_two_digit_gt_30_l39_39382

open Finset
open Rat

-- Definition of digits set
def digits : Finset ℕ := {1, 2, 3}

-- Definition of valid two-digit numbers (in decimal)
def two_digit_numbers : Finset (ℕ × ℕ) := (digits.product digits).filter (λ p, p.1 ≠ p.2)

-- Function to check if a two-digit number made from a pair is greater than 30
def is_greater_than_30 (p : ℕ × ℕ) : Bool :=
  let num := p.1 * 10 + p.2
  num > 30

-- Set of valid two-digit numbers greater than 30
def two_digit_numbers_gt_30 : Finset (ℕ × ℕ) :=
  (two_digit_numbers.filter is_greater_than_30)

-- The proof goal stating the probability
theorem probability_two_digit_gt_30 : 
  (two_digit_numbers_gt_30.card : ℚ) / (two_digit_numbers.card : ℚ) = 1 / 3 :=
by 
  -- Begin proof steps here
  sorry

end probability_two_digit_gt_30_l39_39382


namespace tangent_parabola_line_l39_39384

theorem tangent_parabola_line (a x₀ y₀ : ℝ) 
  (h_line : x₀ - y₀ - 1 = 0)
  (h_parabola : y₀ = a * x₀^2)
  (h_tangent_slope : 2 * a * x₀ = 1) : 
  a = 1 / 4 :=
sorry

end tangent_parabola_line_l39_39384


namespace triangle_perimeter_l39_39489

-- Define the given sides of the triangle
def side_a := 15
def side_b := 6
def side_c := 12

-- Define the function to calculate the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

-- The theorem stating that the perimeter of the given triangle is 33
theorem triangle_perimeter : perimeter side_a side_b side_c = 33 := by
  -- We can include the proof later
  sorry

end triangle_perimeter_l39_39489


namespace fred_limes_l39_39370

theorem fred_limes (limes_total : ℕ) (alyssa_limes : ℕ) (nancy_limes : ℕ) (fred_limes : ℕ)
  (h_total : limes_total = 103)
  (h_alyssa : alyssa_limes = 32)
  (h_nancy : nancy_limes = 35)
  (h_fred : fred_limes = limes_total - (alyssa_limes + nancy_limes)) :
  fred_limes = 36 :=
by
  sorry

end fred_limes_l39_39370


namespace range_of_a_l39_39393

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x < -1 ↔ x ≤ a) ↔ a < -1 :=
by
  sorry

end range_of_a_l39_39393


namespace direct_proportion_function_decrease_no_first_quadrant_l39_39066

-- Part (1)
theorem direct_proportion_function (a b : ℝ) (h : y = (2*a-4)*x + (3-b)) : a ≠ 2 ∧ b = 3 :=
sorry

-- Part (2)
theorem decrease_no_first_quadrant (a b : ℝ) (h : y = (2*a-4)*x + (3-b)) : a < 2 ∧ b ≥ 3 :=
sorry

end direct_proportion_function_decrease_no_first_quadrant_l39_39066


namespace cos_double_angle_identity_l39_39377

theorem cos_double_angle_identity (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 - 2 * α) = -7 / 9 := 
sorry

end cos_double_angle_identity_l39_39377


namespace intersection_of_sets_l39_39889

def M : Set ℝ := { x | 3 * x - 6 ≥ 0 }
def N : Set ℝ := { x | x^2 < 16 }

theorem intersection_of_sets : M ∩ N = { x | 2 ≤ x ∧ x < 4 } :=
by {
  sorry
}

end intersection_of_sets_l39_39889


namespace ott_fractional_part_l39_39439

theorem ott_fractional_part (x : ℝ) :
  let moe_initial := 6 * x
  let loki_initial := 5 * x
  let nick_initial := 4 * x
  let ott_initial := 1
  
  let moe_given := (x : ℝ)
  let loki_given := (x : ℝ)
  let nick_given := (x : ℝ)
  
  let ott_returned_each := (1 / 10) * x
  
  let moe_effective := moe_given - ott_returned_each
  let loki_effective := loki_given - ott_returned_each
  let nick_effective := nick_given - ott_returned_each
  
  let ott_received := moe_effective + loki_effective + nick_effective
  let ott_final_money := ott_initial + ott_received
  
  let total_money_original := moe_initial + loki_initial + nick_initial + ott_initial
  let fraction_ott_final := ott_final_money / total_money_original
  
  ott_final_money / total_money_original = (10 + 27 * x) / (150 * x + 10) :=
by
  sorry

end ott_fractional_part_l39_39439


namespace probability_of_centrally_symmetric_card_l39_39315

def is_centrally_symmetric (shape : String) : Bool :=
  shape = "parallelogram" ∨ shape = "circle"

theorem probability_of_centrally_symmetric_card :
  let shapes := ["parallelogram", "isosceles_right_triangle", "regular_pentagon", "circle"]
  let total_cards := shapes.length
  let centrally_symmetric_cards := shapes.filter is_centrally_symmetric
  let num_centrally_symmetric := centrally_symmetric_cards.length
  (num_centrally_symmetric : ℚ) / (total_cards : ℚ) = 1 / 2 :=
by
  sorry

end probability_of_centrally_symmetric_card_l39_39315


namespace computation_result_l39_39505

theorem computation_result : 143 - 13 + 31 + 17 = 178 := 
by
  sorry

end computation_result_l39_39505


namespace hyperbola_parabola_focus_l39_39400

theorem hyperbola_parabola_focus (m : ℝ) :
  (m + (m - 2) = 4) → m = 3 :=
by
  intro h
  sorry

end hyperbola_parabola_focus_l39_39400


namespace evaluate_expression_l39_39046

theorem evaluate_expression : (2301 - 2222)^2 / 144 = 43 := 
by 
  sorry

end evaluate_expression_l39_39046


namespace least_n_froods_score_l39_39758

theorem least_n_froods_score (n : ℕ) : (n * (n + 1) / 2 > 12 * n) ↔ (n > 23) := 
by 
  sorry

end least_n_froods_score_l39_39758


namespace squares_sum_l39_39653

theorem squares_sum (a b c : ℝ) 
  (h1 : 36 - 4 * Real.sqrt 2 - 6 * Real.sqrt 3 + 12 * Real.sqrt 6 = (a * Real.sqrt 2 + b * Real.sqrt 3 + c) ^ 2) : 
  a^2 + b^2 + c^2 = 14 := 
by
  sorry

end squares_sum_l39_39653


namespace binomial_coeff_sum_abs_l39_39922

theorem binomial_coeff_sum_abs (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℤ)
  (h : (2 * x - 1)^6 = a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0):
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 729 :=
by
  sorry

end binomial_coeff_sum_abs_l39_39922


namespace train_crossing_time_l39_39842

def speed := 60 -- in km/hr
def length := 300 -- in meters
def speed_in_m_per_s := (60 * 1000) / 3600 -- converting speed from km/hr to m/s
def expected_time := 18 -- in seconds

theorem train_crossing_time :
  (300 / (speed_in_m_per_s)) = expected_time :=
sorry

end train_crossing_time_l39_39842


namespace probability_of_winning_first_draw_better_chance_with_yellow_ball_l39_39637

-- The probability of winning on the first draw in the lottery promotion.
theorem probability_of_winning_first_draw :
  (1 / 4 : ℚ) = 0.25 :=
sorry

-- The optimal choice to add to the bag for the highest probability of receiving a fine gift.
theorem better_chance_with_yellow_ball :
  (3 / 5 : ℚ) > (2 / 5 : ℚ) :=
by norm_num

end probability_of_winning_first_draw_better_chance_with_yellow_ball_l39_39637


namespace darma_eats_peanuts_l39_39860

/--
Darma can eat 20 peanuts in 15 seconds. Prove that she can eat 480 peanuts in 6 minutes.
-/
theorem darma_eats_peanuts (rate: ℕ) (per_seconds: ℕ) (minutes: ℕ) (conversion: ℕ) : 
  (rate = 20) → (per_seconds = 15) → (minutes = 6) → (conversion = 60) → 
  rate * (conversion * minutes / per_seconds) = 480 :=
by
  intros hrate hseconds hminutes hconversion
  rw [hrate, hseconds, hminutes, hconversion]
  -- skipping the detailed proof
  sorry

end darma_eats_peanuts_l39_39860


namespace socks_total_l39_39582

def socks_lisa (initial: Nat) := initial + 0

def socks_sandra := 20

def socks_cousin (sandra: Nat) := sandra / 5

def socks_mom (initial: Nat) := 8 + 3 * initial

theorem socks_total (initial: Nat) (sandra: Nat) :
  initial = 12 → sandra = 20 → 
  socks_lisa initial + socks_sandra + socks_cousin sandra + socks_mom initial = 80 :=
by
  intros h_initial h_sandra
  rw [h_initial, h_sandra]
  sorry

end socks_total_l39_39582


namespace number_of_stickers_used_to_decorate_l39_39777

def initial_stickers : ℕ := 20
def bought_stickers : ℕ := 12
def birthday_stickers : ℕ := 20
def given_stickers : ℕ := 5
def remaining_stickers : ℕ := 39

theorem number_of_stickers_used_to_decorate :
  (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers) = 8 :=
by
  -- Proof goes here
  sorry

end number_of_stickers_used_to_decorate_l39_39777


namespace number_of_even_factors_l39_39227

theorem number_of_even_factors {n : ℕ} (h : n = 2^4 * 3^3 * 7) : 
  ∃ (count : ℕ), count = 32 ∧ ∀ k, (k ∣ n) → k % 2 = 0 → count = 32 :=
by
  sorry

end number_of_even_factors_l39_39227


namespace probability_x_plus_2y_leq_6_l39_39329

noncomputable def probability_condition (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5 ∧ x + 2*y ≤ 6

theorem probability_x_plus_2y_leq_6 :
  let probability := (8 / 20 : ℝ)
  in probability = 2 / 5 :=
  sorry

end probability_x_plus_2y_leq_6_l39_39329


namespace length_AD_l39_39102

open Real

-- Define the properties of the quadrilateral
variable (A B C D: Point)
variable (angle_ABC angle_BCD: ℝ)
variable (AB BC CD: ℝ)

-- Given conditions
axiom angle_ABC_eq_135 : angle_ABC = 135 * π / 180
axiom angle_BCD_eq_120 : angle_BCD = 120 * π / 180
axiom AB_eq_sqrt_6 : AB = sqrt 6
axiom BC_eq_5_minus_sqrt_3 : BC = 5 - sqrt 3
axiom CD_eq_6 : CD = 6

-- The theorem to prove
theorem length_AD {AD : ℝ} (h : True) :
  AD = 2 * sqrt 19 :=
sorry

end length_AD_l39_39102


namespace pieces_eaten_first_night_l39_39524

-- Define the initial numbers of candies
def debby_candies : Nat := 32
def sister_candies : Nat := 42
def candies_left : Nat := 39

-- Calculate the initial total number of candies
def initial_total_candies : Nat := debby_candies + sister_candies

-- Define the number of candies eaten the first night
def candies_eaten : Nat := initial_total_candies - candies_left

-- The problem statement with the proof goal
theorem pieces_eaten_first_night : candies_eaten = 35 := by
  sorry

end pieces_eaten_first_night_l39_39524


namespace percentage_decrease_of_original_number_is_30_l39_39945

theorem percentage_decrease_of_original_number_is_30 :
  ∀ (original_number : ℕ) (difference : ℕ) (percent_increase : ℚ) (percent_decrease : ℚ),
  original_number = 40 →
  percent_increase = 0.25 →
  difference = 22 →
  original_number + percent_increase * original_number - (original_number - percent_decrease * original_number) = difference →
  percent_decrease = 0.30 :=
by
  intros original_number difference percent_increase percent_decrease h_original h_increase h_diff h_eq
  sorry

end percentage_decrease_of_original_number_is_30_l39_39945


namespace max_value_m_l39_39044

/-- Proof that the inequality (a^2 + 4(b^2 + c^2))(b^2 + 4(a^2 + c^2))(c^2 + 4(a^2 + b^2)) 
    is greater than or equal to 729 for all a, b, c ∈ ℝ \ {0} with 
    |1/a| + |1/b| + |1/c| ≤ 3. -/
theorem max_value_m (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h_cond : |1 / a| + |1 / b| + |1 / c| ≤ 3) :
  (a^2 + 4 * (b^2 + c^2)) * (b^2 + 4 * (a^2 + c^2)) * (c^2 + 4 * (a^2 + b^2)) ≥ 729 :=
by {
  sorry
}

end max_value_m_l39_39044


namespace repeating_decimal_multiplication_l39_39049

theorem repeating_decimal_multiplication :
  (0.0808080808 : ℝ) * (0.3333333333 : ℝ) = (8 / 297) := by
  sorry

end repeating_decimal_multiplication_l39_39049


namespace sum_of_square_areas_l39_39230

theorem sum_of_square_areas (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 :=
sorry

end sum_of_square_areas_l39_39230


namespace least_positive_integer_with_12_factors_l39_39988

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l39_39988


namespace percent_of_total_l39_39171

theorem percent_of_total (p n : ℝ) (h1 : p = 35 / 100) (h2 : n = 360) : p * n = 126 := by
  sorry

end percent_of_total_l39_39171


namespace total_canoes_by_end_of_april_l39_39019

def N_F : ℕ := 4
def N_M : ℕ := 3 * N_F
def N_A : ℕ := 3 * N_M
def total_canoes : ℕ := N_F + N_M + N_A

theorem total_canoes_by_end_of_april : total_canoes = 52 := by
  sorry

end total_canoes_by_end_of_april_l39_39019


namespace lisa_socks_total_l39_39585

def total_socks (initial : ℕ) (sandra : ℕ) (cousin_ratio : ℕ → ℕ) (mom_extra : ℕ → ℕ) : ℕ :=
  initial + sandra + cousin_ratio sandra + mom_extra initial

def cousin_ratio (sandra : ℕ) : ℕ := sandra / 5
def mom_extra (initial : ℕ) : ℕ := 3 * initial + 8

theorem lisa_socks_total :
  total_socks 12 20 cousin_ratio mom_extra = 80 := by
  sorry

end lisa_socks_total_l39_39585


namespace clea_ride_escalator_time_l39_39350

theorem clea_ride_escalator_time
  (s v d : ℝ)
  (h1 : 75 * s = d)
  (h2 : 30 * (s + v) = d) :
  t = 50 :=
by
  sorry

end clea_ride_escalator_time_l39_39350


namespace right_triangle_condition_l39_39688

theorem right_triangle_condition (a d : ℝ) (h : d > 0) : 
  (a = d * (1 + Real.sqrt 7)) ↔ (a^2 + (a + 2 * d)^2 = (a + 4 * d)^2) := 
sorry

end right_triangle_condition_l39_39688


namespace krakozyabrs_proof_l39_39417

-- Defining necessary variables and conditions
variable {n : ℕ}

-- Conditions given in the problem
def condition1 (H : ℕ) : Prop := 0.2 * H = n
def condition2 (W : ℕ) : Prop := 0.25 * W = n

-- Definition using the principle of inclusion-exclusion
def total_krakozyabrs (H W : ℕ) : ℕ := H + W - n

-- Statements reflecting the conditions and the final conclusion
theorem krakozyabrs_proof (H W : ℕ) (n : ℕ) : condition1 H → condition2 W → (25 < total_krakozyabrs H W) ∧ (total_krakozyabrs H W < 35) → total_krakozyabrs H W = 32 :=
by
  -- We skip the proof here
  sorry

end krakozyabrs_proof_l39_39417


namespace number_of_integers_satisfying_inequality_l39_39543

theorem number_of_integers_satisfying_inequality (S : set ℤ) :
  (S = {x : ℤ | |7 * x - 5| ≤ 9}) →
  S.card = 3 :=
by
  intro hS
  sorry

end number_of_integers_satisfying_inequality_l39_39543


namespace least_positive_integer_with_12_factors_is_96_l39_39965

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l39_39965


namespace counting_integers_between_multiples_l39_39033

theorem counting_integers_between_multiples :
  let smallest_perfect_square_multiple := 900 in
  let smallest_perfect_cube_multiple := 27000 in
  let num_integers := (smallest_perfect_cube_multiple / 30) - (smallest_perfect_square_multiple / 30) + 1 in
  smallest_perfect_square_multiple = 30 * 30 ∧ 
  smallest_perfect_cube_multiple = 900 * 30 ∧ 
  num_integers = 871 :=
by
  sorry

end counting_integers_between_multiples_l39_39033


namespace carlos_picks_24_integers_l39_39038

def is_divisor (n m : ℕ) : Prop := m % n = 0

theorem carlos_picks_24_integers :
  ∃ (s : Finset ℕ), s.card = 24 ∧ ∀ n ∈ s, is_divisor n 4500 ∧ 1 ≤ n ∧ n ≤ 4500 ∧ n % 3 = 0 :=
by
  sorry

end carlos_picks_24_integers_l39_39038


namespace find_number_of_girls_in_class_l39_39952

variable (G : ℕ)

def number_of_ways_to_select_two_boys (n : ℕ) : ℕ := Nat.choose n 2

theorem find_number_of_girls_in_class 
  (boys : ℕ := 13) 
  (ways_to_select_students : ℕ := 780) 
  (ways_to_select_two_boys : ℕ := number_of_ways_to_select_two_boys boys) :
  G * ways_to_select_two_boys = ways_to_select_students → G = 10 := 
by
  sorry

end find_number_of_girls_in_class_l39_39952


namespace total_lives_l39_39630

-- Definitions of given conditions
def original_friends : Nat := 2
def lives_per_player : Nat := 6
def additional_players : Nat := 2

-- Proof statement to show the total number of lives
theorem total_lives :
  (original_friends * lives_per_player) + (additional_players * lives_per_player) = 24 := by
  sorry

end total_lives_l39_39630


namespace positive_value_of_n_l39_39201

theorem positive_value_of_n (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 25 = 0 ∧ ∃! x : ℝ, 4 * x^2 + n * x + 25 = 0) →
  n = 20 :=
by
  sorry

end positive_value_of_n_l39_39201


namespace min_area_circle_equation_l39_39210

theorem min_area_circle_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 3 / (2 + x) + 3 / (2 + y) = 1) : (x - 4)^2 + (y - 4)^2 = 256 :=
sorry

end min_area_circle_equation_l39_39210


namespace unique_m_for_prime_condition_l39_39686

theorem unique_m_for_prime_condition :
  ∃ (m : ℕ), m > 0 ∧ (∀ (p : ℕ), Prime p → (∀ (n : ℕ), ¬ p ∣ (n^m - m))) ↔ m = 1 :=
sorry

end unique_m_for_prime_condition_l39_39686


namespace min_distance_PA_l39_39249

theorem min_distance_PA :
  let A : ℝ × ℝ := (0, 1)
  ∀ (P : ℝ × ℝ), (∃ x : ℝ, x > 0 ∧ P = (x, (x + 2) / x)) →
  ∃ d : ℝ, d = 2 ∧ ∀ Q : ℝ × ℝ, (∃ x : ℝ, x > 0 ∧ Q = (x, (x + 2) / x)) → dist A Q ≥ d :=
by
  sorry

end min_distance_PA_l39_39249


namespace mean_volume_of_cubes_l39_39463

theorem mean_volume_of_cubes (a b c : ℕ) (h1 : a = 4) (h2 : b = 5) (h3 : c = 6) :
  ((a^3 + b^3 + c^3) / 3) = 135 :=
by
  -- known cube volumes and given edge lengths conditions
  sorry

end mean_volume_of_cubes_l39_39463


namespace wendy_created_albums_l39_39285

theorem wendy_created_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) (total_pics : ℕ) (albums : ℕ) :
  phone_pics = 22 → camera_pics = 2 → pics_per_album = 6 → total_pics = phone_pics + camera_pics → albums = total_pics / pics_per_album → albums = 4 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end wendy_created_albums_l39_39285


namespace Cameron_list_count_l39_39034

theorem Cameron_list_count : 
  let lower_bound := 900
  let upper_bound := 27000
  let step := 30
  let n_min := lower_bound / step
  let n_max := upper_bound / step
  n_max - n_min + 1 = 871 :=
by
  sorry

end Cameron_list_count_l39_39034


namespace find_K_l39_39963

noncomputable def Z (K : ℕ) : ℕ := K^3

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

def K_values : Finset ℕ := {4, 9, 16}

theorem find_K (K : ℕ) (hK : K > 1) (hZ : 50 < Z K ∧ Z K < 5000) :
  is_perfect_square (Z K) ↔ K ∈ K_values :=
by
  sorry

end find_K_l39_39963


namespace tens_digit_of_9_pow_1010_l39_39312

theorem tens_digit_of_9_pow_1010 : (9 ^ 1010) % 100 = 1 :=
by sorry

end tens_digit_of_9_pow_1010_l39_39312


namespace part1_part2_l39_39736

open Function

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0       => 2
| (n+1) => a n + 2^n + 2

-- Define the shifted sequence {b_n = a_n - 2^n}
def b (n : ℕ) : ℕ := a n - 2^n

-- Prove that b_n is an arithmetic sequence with first term 0 and common difference 2
theorem part1 : b 0 = 0 ∧ ∀ n : ℕ, b (n+1) = b n + 2 :=
by
  sorry

-- Define the sequence {S_n} being the sum of the first n terms of {a_n}
def sum_of_an (n : ℕ) : ℕ := (Finset.range n).sum a

-- Prove that S_n = 2^(n+1) - 2 + n^2 - n
theorem part2 (n : ℕ) : sum_of_an n = 2^(n+1) - 2 + n^2 - n :=
by
  sorry

end part1_part2_l39_39736


namespace shobha_current_age_l39_39498

theorem shobha_current_age (S B : ℕ) (h1 : S / B = 4 / 3) (h2 : S + 6 = 26) : B = 15 :=
by
  -- Here we would begin the proof
  sorry

end shobha_current_age_l39_39498


namespace YZ_length_l39_39913

theorem YZ_length : 
  ∀ (X Y Z : Type) 
  (angle_Y angle_Z angle_X : ℝ)
  (XZ YZ : ℝ),
  angle_Y = 45 ∧ angle_Z = 60 ∧ XZ = 6 →
  angle_X = 180 - angle_Y - angle_Z →
  YZ = XZ * (Real.sin angle_X / Real.sin angle_Y) →
  YZ = 3 * (Real.sqrt 6 + Real.sqrt 2) :=
by
  intros X Y Z angle_Y angle_Z angle_X XZ YZ
  intro h1 h2 h3
  sorry

end YZ_length_l39_39913


namespace sum_geometric_sequence_terms_l39_39615

theorem sum_geometric_sequence_terms (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 := 
by
  sorry

end sum_geometric_sequence_terms_l39_39615


namespace range_of_a_l39_39538

theorem range_of_a (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 4) (h3 : a > b) (h4 : b > c) :
  a ∈ Set.Ioo (2 / 3) 2 :=
sorry

end range_of_a_l39_39538


namespace olaf_travels_miles_l39_39783

-- Define the given conditions
def men : ℕ := 25
def per_day_water_per_man : ℚ := 1 / 2
def boat_mileage_per_day : ℕ := 200
def total_water : ℚ := 250

-- Define the daily water consumption for the crew
def daily_water_consumption : ℚ := men * per_day_water_per_man

-- Define the number of days the water will last
def days_water_lasts : ℚ := total_water / daily_water_consumption

-- Define the total miles traveled
def total_miles_traveled : ℚ := days_water_lasts * boat_mileage_per_day

-- Theorem statement to prove the total miles traveled is 4000 miles
theorem olaf_travels_miles : total_miles_traveled = 4000 := by
  sorry

end olaf_travels_miles_l39_39783


namespace bug_final_position_after_2023_jumps_l39_39128

open Nat

def bug_jump (pos : Nat) : Nat :=
  if pos % 2 = 1 then (pos + 2) % 6 else (pos + 1) % 6

noncomputable def final_position (n : Nat) : Nat :=
  (iterate bug_jump n 6) % 6

theorem bug_final_position_after_2023_jumps : final_position 2023 = 1 := by
  sorry

end bug_final_position_after_2023_jumps_l39_39128


namespace jar_water_fraction_l39_39690

theorem jar_water_fraction
  (S L : ℝ)
  (h1 : S = (1 / 5) * S)
  (h2 : S = x * L)
  (h3 : (1 / 5) * S + x * L = (2 / 5) * L) :
  x = (1 / 10) :=
by
  sorry

end jar_water_fraction_l39_39690


namespace savings_on_discounted_milk_l39_39920

theorem savings_on_discounted_milk :
  let num_gallons := 8
  let price_per_gallon := 3.20
  let discount_rate := 0.25
  let discount_per_gallon := price_per_gallon * discount_rate
  let discounted_price_per_gallon := price_per_gallon - discount_per_gallon
  let total_cost_without_discount := num_gallons * price_per_gallon
  let total_cost_with_discount := num_gallons * discounted_price_per_gallon
  let savings := total_cost_without_discount - total_cost_with_discount
  savings = 6.40 :=
by
  sorry

end savings_on_discounted_milk_l39_39920


namespace smallest_value_proof_l39_39395

noncomputable def smallest_value (x : ℝ) (h : 0 < x ∧ x < 1) : Prop :=
  x^3 < x ∧ x^3 < 3*x ∧ x^3 < x^(1/3) ∧ x^3 < 1/x^2

theorem smallest_value_proof (x : ℝ) (h : 0 < x ∧ x < 1) : smallest_value x h :=
  sorry

end smallest_value_proof_l39_39395


namespace probability_C_l39_39175

variable (pA pB pD pC : ℚ)
variable (hA : pA = 1 / 4)
variable (hB : pB = 1 / 3)
variable (hD : pD = 1 / 6)
variable (total_prob : pA + pB + pD + pC = 1)

theorem probability_C (hA : pA = 1 / 4) (hB : pB = 1 / 3) (hD : pD = 1 / 6) (total_prob : pA + pB + pD + pC = 1) : pC = 1 / 4 :=
sorry

end probability_C_l39_39175


namespace range_of_a_l39_39546

theorem range_of_a (a : ℝ) : (∃ x : ℝ, real.sin x < a) → a > -1 :=
sorry

end range_of_a_l39_39546


namespace p_sq_plus_q_sq_l39_39246

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 := by
  sorry

end p_sq_plus_q_sq_l39_39246


namespace cos_sqr_sub_power_zero_l39_39480

theorem cos_sqr_sub_power_zero :
  (cos (30 * Real.pi / 180))^2 - (2 - Real.pi)^0 = -1/4 :=
by
  sorry

end cos_sqr_sub_power_zero_l39_39480


namespace anne_clean_house_in_12_hours_l39_39474

theorem anne_clean_house_in_12_hours (B A : ℝ) (h1 : 4 * (B + A) = 1) (h2 : 3 * (B + 2 * A) = 1) : A = 1 / 12 ∧ (1 / A) = 12 :=
by
  -- We will leave the proof as a placeholder
  sorry

end anne_clean_house_in_12_hours_l39_39474


namespace domain_of_v_l39_39468

-- Define the function v
noncomputable def v (x y : ℝ) : ℝ := 1 / (x^(2/3) - y^(2/3))

-- State the domain of v
def domain_v : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≠ p.2 }

-- State the main theorem
theorem domain_of_v :
  ∀ x y : ℝ, x ≠ y ↔ (x, y) ∈ domain_v :=
by
  intro x y
  -- We don't need to provide proof
  sorry

end domain_of_v_l39_39468


namespace trillion_in_scientific_notation_l39_39756

theorem trillion_in_scientific_notation :
  (10^4) * (10^4) * (10^4) = 10^(12) := 
by sorry

end trillion_in_scientific_notation_l39_39756


namespace cameron_list_count_l39_39022

theorem cameron_list_count : 
  (∃ (n m : ℕ), n = 900 ∧ m = 27000 ∧ (∀ k : ℕ, (30 * k) ≥ n ∧ (30 * k) ≤ m → ∃ count : ℕ, count = 871)) :=
by
  sorry

end cameron_list_count_l39_39022


namespace fraction_product_l39_39501

theorem fraction_product :
  (8 / 4) * (10 / 5) * (21 / 14) * (16 / 8) * (45 / 15) * (30 / 10) * (49 / 35) * (32 / 16) = 302.4 := by
  sorry

end fraction_product_l39_39501


namespace cos_alpha_minus_pi_over_4_l39_39878

theorem cos_alpha_minus_pi_over_4 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (h_tan : Real.tan α = 2) :
  Real.cos (α - π / 4) = (3 * Real.sqrt 10) / 10 := 
  sorry

end cos_alpha_minus_pi_over_4_l39_39878


namespace Kim_has_4_cousins_l39_39765

noncomputable def pieces_per_cousin : ℕ := 5
noncomputable def total_pieces : ℕ := 20
noncomputable def cousins : ℕ := total_pieces / pieces_per_cousin

theorem Kim_has_4_cousins : cousins = 4 := 
by
  show cousins = 4
  sorry

end Kim_has_4_cousins_l39_39765


namespace anna_interest_l39_39186

noncomputable def interest_earned (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n - P

theorem anna_interest : interest_earned 2000 0.08 5 = 938.66 := by
  sorry

end anna_interest_l39_39186


namespace infinite_sum_evaluation_l39_39511

theorem infinite_sum_evaluation :
  (∑' n : ℕ, (n : ℚ) / ((n^2 - 2 * n + 2) * (n^2 + 2 * n + 4))) = 5 / 24 :=
sorry

end infinite_sum_evaluation_l39_39511


namespace least_positive_integer_with_12_factors_is_972_l39_39992

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l39_39992


namespace molecular_weight_of_Y_l39_39185

def molecular_weight_X : ℝ := 136
def molecular_weight_C6H8O7 : ℝ := 192
def moles_C6H8O7 : ℝ := 5

def total_mass_reactants := molecular_weight_X + moles_C6H8O7 * molecular_weight_C6H8O7

theorem molecular_weight_of_Y :
  total_mass_reactants = 1096 := by
  sorry

end molecular_weight_of_Y_l39_39185


namespace average_apples_sold_per_day_l39_39137

theorem average_apples_sold_per_day (boxes_sold : ℕ) (days : ℕ) (apples_per_box : ℕ) (H1 : boxes_sold = 12) (H2 : days = 4) (H3 : apples_per_box = 25) : (boxes_sold * apples_per_box) / days = 75 :=
by {
  -- Based on given conditions, the total apples sold is 12 * 25 = 300.
  -- Dividing by the number of days, 300 / 4 gives us 75 apples/day.
  -- The proof is omitted as instructed.
  sorry
}

end average_apples_sold_per_day_l39_39137


namespace line_cannot_pass_through_third_quadrant_l39_39948

theorem line_cannot_pass_through_third_quadrant :
  ∀ (x y : ℝ), x + y - 1 = 0 → ¬(x < 0 ∧ y < 0) :=
by
  sorry

end line_cannot_pass_through_third_quadrant_l39_39948


namespace find_y_l39_39401

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : x = 8) : y = 1 :=
by
  sorry

end find_y_l39_39401


namespace cubes_sum_equiv_l39_39548

theorem cubes_sum_equiv (h : 2^3 + 4^3 + 6^3 + 8^3 + 10^3 + 12^3 + 14^3 + 16^3 + 18^3 = 16200) :
  3^3 + 6^3 + 9^3 + 12^3 + 15^3 + 18^3 + 21^3 + 24^3 + 27^3 = 54675 := 
  sorry

end cubes_sum_equiv_l39_39548


namespace percentage_increase_in_price_l39_39854

theorem percentage_increase_in_price (initial_price : ℝ) (total_cost : ℝ) (num_family_members : ℕ) 
  (pounds_per_person : ℝ) (new_price : ℝ) (percentage_increase : ℝ) :
  initial_price = 1.6 → 
  total_cost = 16 → 
  num_family_members = 4 → 
  pounds_per_person = 2 → 
  (total_cost / (num_family_members * pounds_per_person)) = new_price → 
  percentage_increase = ((new_price - initial_price) / initial_price) * 100 → 
  percentage_increase = 25 :=
by
  intros h_initial h_total h_members h_pounds h_new_price h_percentage
  sorry

end percentage_increase_in_price_l39_39854


namespace sum_of_first_15_odd_integers_l39_39303

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l39_39303


namespace inequalities_correct_l39_39532

-- Define the basic conditions
variables {a b c d : ℝ}

-- Conditions given in the problem
axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : 0 > c
axiom h4 : c > d

-- Correct answers to be proven
theorem inequalities_correct (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (a + c > b + d) ∧ (a * d^2 > b * c^2) ∧ (1 / (b * c) < 1 / (a * d)) :=
begin
  -- Proof part
  sorry
end

end inequalities_correct_l39_39532


namespace range_of_a_l39_39429

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + 2 > 0) → a > 9 / 8 :=
by
  sorry

end range_of_a_l39_39429


namespace area_bounded_by_circles_and_x_axis_l39_39503

/--
Circle C has its center at (5, 5) and radius 5 units.
Circle D has its center at (15, 5) and radius 5 units.
Prove that the area of the region bounded by these circles
and the x-axis is 50 - 25 * π square units.
-/
theorem area_bounded_by_circles_and_x_axis :
  let C_center := (5, 5)
  let D_center := (15, 5)
  let radius := 5
  (2 * (radius * radius) * π / 2) + (10 * radius) = 50 - 25 * π :=
sorry

end area_bounded_by_circles_and_x_axis_l39_39503


namespace side_length_of_square_l39_39648

theorem side_length_of_square (s : ℝ) (h : s^2 = 100) : s = 10 := 
sorry

end side_length_of_square_l39_39648


namespace find_x_l39_39457

theorem find_x (x : ℝ) (h : x - 1/10 = x / 10) : x = 1 / 9 := 
  sorry

end find_x_l39_39457


namespace sum_of_first_15_odd_integers_l39_39305

theorem sum_of_first_15_odd_integers : 
  ∑ i in finRange 15, (2 * (i + 1) - 1) = 225 :=
by
  sorry

end sum_of_first_15_odd_integers_l39_39305


namespace least_positive_integer_with_12_factors_l39_39985

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l39_39985


namespace fish_count_l39_39622

theorem fish_count (total_fish blue_fish blue_spotted_fish : ℕ)
  (h1 : 1 / 3 * total_fish = blue_fish)
  (h2 : 1 / 2 * blue_fish = blue_spotted_fish)
  (h3 : blue_spotted_fish = 10) : total_fish = 60 :=
sorry

end fish_count_l39_39622


namespace solution_l39_39433

noncomputable def problem (p q r s : ℝ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
  (∀ x : ℝ, x^2 - 14 * p * x - 15 * q = 0 → x = r ∨ x = s) ∧
  (∀ x : ℝ, x^2 - 14 * r * x - 15 * s = 0 → x = p ∨ x = q)

theorem solution (p q r s : ℝ) (h : problem p q r s) : p + q + r + s = 3150 :=
sorry

end solution_l39_39433


namespace cost_of_monogramming_each_backpack_l39_39541

def number_of_backpacks : ℕ := 5
def original_price_per_backpack : ℝ := 20.00
def discount_rate : ℝ := 0.20
def total_cost : ℝ := 140.00

theorem cost_of_monogramming_each_backpack : 
  (total_cost - (number_of_backpacks * (original_price_per_backpack * (1 - discount_rate)))) / number_of_backpacks = 12.00 :=
by
  sorry 

end cost_of_monogramming_each_backpack_l39_39541


namespace find_y_l39_39079

theorem find_y (x y : ℝ) (h1 : x^2 - 4 * x = y + 5) (h2 : x = 7) : y = 16 := by
  sorry

end find_y_l39_39079


namespace pow_mod_1110_l39_39244

theorem pow_mod_1110 (n : ℕ) (h₀ : 0 ≤ n ∧ n < 1111)
    (h₁ : 2^1110 % 11 = 1) (h₂ : 2^1110 % 101 = 14) : 
    n = 1024 := 
sorry

end pow_mod_1110_l39_39244


namespace Carlos_candy_share_l39_39440

theorem Carlos_candy_share (total_candy : ℚ) (num_piles : ℕ) (piles_for_Carlos : ℕ)
  (h_total_candy : total_candy = 75 / 7)
  (h_num_piles : num_piles = 5)
  (h_piles_for_Carlos : piles_for_Carlos = 2) :
  (piles_for_Carlos * (total_candy / num_piles) = 30 / 7) :=
by
  sorry

end Carlos_candy_share_l39_39440


namespace find_g3_l39_39135

variable {g : ℝ → ℝ}

-- Defining the condition from the problem
def g_condition (x : ℝ) (h : x ≠ 0) : g x - 3 * g (1 / x) = 3^x + x^2 := sorry

-- The main statement to prove
theorem find_g3 : g 3 = - (3 * 3^(1/3) + 1/3 + 36) / 8 := sorry

end find_g3_l39_39135


namespace probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l39_39633

theorem probability_winning_on_first_draw : 
  let red := 1 
  let yellow := 3 
  red / (red + yellow) = 1 / 4 :=
by 
  sorry

theorem optimal_ball_to_add_for_fine_gift :
  let red := 1 
  let yellow := 3
  -- After adding a red ball: 2 red, 3 yellow
  let p1 := (2 * 1 + 3 * 2) / (2 + 3) / (1 + 3) = (2/5)
  -- After adding a yellow ball: 1 red, 4 yellow
  let p2 := (1 * 0 + 4 * 3) / (1 + 4) / (1 + 3) = (3/5)
  p1 < p2 :=
by 
  sorry

end probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l39_39633


namespace range_of_f_l39_39881

def f (x : ℕ) : ℤ := 2 * x - 3

def domain := {x : ℕ | 1 ≤ x ∧ x ≤ 5}

def range (f : ℕ → ℤ) (s : Set ℕ) : Set ℤ :=
  {y : ℤ | ∃ x ∈ s, f x = y}

theorem range_of_f :
  range f domain = {-1, 1, 3, 5, 7} :=
by
  sorry

end range_of_f_l39_39881


namespace bridge_length_is_correct_l39_39181

noncomputable def train_length : ℝ := 135
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def bridge_crossing_time : ℝ := 30

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
noncomputable def total_distance_crossed : ℝ := train_speed_ms * bridge_crossing_time
noncomputable def bridge_length : ℝ := total_distance_crossed - train_length

theorem bridge_length_is_correct : bridge_length = 240 := by
  sorry

end bridge_length_is_correct_l39_39181


namespace exterior_angle_regular_octagon_l39_39911

theorem exterior_angle_regular_octagon : 
  ∀ {θ : ℝ}, 
  (8 - 2) * 180 / 8 = θ →
  180 - θ = 45 := 
by 
  intro θ hθ
  sorry

end exterior_angle_regular_octagon_l39_39911


namespace percentage_people_taking_bus_l39_39085

-- Definitions
def population := 80
def car_pollution := 10 -- pounds of carbon per car per year
def bus_pollution := 100 -- pounds of carbon per bus per year
def bus_capacity := 40 -- people per bus
def carbon_reduction := 100 -- pounds of carbon reduced per year after the bus is introduced

-- Problem statement in Lean 4
theorem percentage_people_taking_bus :
  (10 / 80 : ℝ) = 0.125 :=
by
  sorry

end percentage_people_taking_bus_l39_39085


namespace leak_time_to_empty_cistern_l39_39820

theorem leak_time_to_empty_cistern :
  (1/6 - 1/8) = 1/24 → (1 / (1/24)) = 24 := by
sorry

end leak_time_to_empty_cistern_l39_39820


namespace probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l39_39634

theorem probability_winning_on_first_draw : 
  let red := 1 
  let yellow := 3 
  red / (red + yellow) = 1 / 4 :=
by 
  sorry

theorem optimal_ball_to_add_for_fine_gift :
  let red := 1 
  let yellow := 3
  -- After adding a red ball: 2 red, 3 yellow
  let p1 := (2 * 1 + 3 * 2) / (2 + 3) / (1 + 3) = (2/5)
  -- After adding a yellow ball: 1 red, 4 yellow
  let p2 := (1 * 0 + 4 * 3) / (1 + 4) / (1 + 3) = (3/5)
  p1 < p2 :=
by 
  sorry

end probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l39_39634


namespace unit_digit_2_pow_2024_l39_39202

theorem unit_digit_2_pow_2024 : (2 ^ 2024) % 10 = 6 := by
  -- We observe the repeating pattern in the unit digits of powers of 2:
  -- 2^1 = 2 -> unit digit is 2
  -- 2^2 = 4 -> unit digit is 4
  -- 2^3 = 8 -> unit digit is 8
  -- 2^4 = 16 -> unit digit is 6
  -- The cycle repeats every 4 powers: 2, 4, 8, 6
  -- 2024 ≡ 0 (mod 4), so it corresponds to the unit digit of 2^4, which is 6
  sorry

end unit_digit_2_pow_2024_l39_39202


namespace average_age_of_population_l39_39909

theorem average_age_of_population
  (k : ℕ)
  (ratio_women_men : 7 * (k : ℕ) = 7 * (k : ℕ) + 5 * (k : ℕ) - 5 * (k : ℕ))
  (avg_age_women : ℝ := 38)
  (avg_age_men : ℝ := 36)
  : ( (7 * k * avg_age_women) + (5 * k * avg_age_men) ) / (12 * k) = 37 + (1 / 6) :=
by
  sorry

end average_age_of_population_l39_39909


namespace ratio_of_area_l39_39155

-- Define the sides of the triangles
def sides_GHI := (6, 8, 10)
def sides_JKL := (9, 12, 15)

-- Define the function to calculate the area of a triangle given its sides as a Pythagorean triple
def area_of_right_triangle (a b : Nat) : Nat :=
  (a * b) / 2

-- Calculate the areas
def area_GHI := area_of_right_triangle 6 8
def area_JKL := area_of_right_triangle 9 12

-- Calculate the ratio of the areas
def ratio_of_areas := area_GHI * 2 / area_JKL * 3

-- Statement to prove
theorem ratio_of_area (sides_GHI sides_JKL : (Nat × Nat × Nat)) :
  (ratio_of_areas = (4 : ℚ) / 9) :=
sorry

end ratio_of_area_l39_39155


namespace find_constants_l39_39243

open Matrix 

noncomputable def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 2, 1; 2, 0, 2; 1, 2, 0]

theorem find_constants :
  let s := (-10 : ℤ)
  let t := (-8 : ℤ)
  let u := (-36 : ℤ)
  B^3 + s • (B^2) + t • B + u • (1 : Matrix (Fin 3) (Fin 3) ℤ) = 0 := sorry

end find_constants_l39_39243


namespace zachary_pushups_l39_39009

theorem zachary_pushups (david_pushups : ℕ) (h1 : david_pushups = 44) (h2 : ∀ z : ℕ, z = david_pushups + 7) : z = 51 :=
by
  sorry

end zachary_pushups_l39_39009


namespace probability_of_event_3a_minus_1_gt_0_l39_39376

noncomputable def probability_event : ℝ :=
if h : 0 <= 1 then (1 - 1/3) else 0

theorem probability_of_event_3a_minus_1_gt_0 (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) : 
  probability_event = 2 / 3 :=
by
  sorry

end probability_of_event_3a_minus_1_gt_0_l39_39376


namespace tiles_difference_between_tenth_and_eleventh_square_l39_39837

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ :=
  3 + 2 * (n - 1)

-- Define the area of the nth square
def area (n : ℕ) : ℕ :=
  (side_length n) ^ 2

-- The math proof statement
theorem tiles_difference_between_tenth_and_eleventh_square : area 11 - area 10 = 88 :=
by 
  -- Proof goes here, but we use sorry to skip it for now
  sorry

end tiles_difference_between_tenth_and_eleventh_square_l39_39837


namespace distribute_tourists_l39_39464

theorem distribute_tourists (guides tourists : ℕ) (hguides : guides = 3) (htourists : tourists = 8) :
  ∃ k, k = 5796 := by
  sorry

end distribute_tourists_l39_39464


namespace second_month_interest_l39_39916

def compounded_interest (initial_loan : ℝ) (rate_per_month : ℝ) : ℝ :=
  initial_loan * rate_per_month

theorem second_month_interest :
  let initial_loan := 200
  let rate_per_month := 0.10
  compounded_interest (initial_loan + compounded_interest initial_loan rate_per_month) rate_per_month = 22 :=
by
  sorry

end second_month_interest_l39_39916


namespace min_value_of_function_l39_39205

open Real

theorem min_value_of_function (x : ℝ) (h : x > 2) : (∃ a : ℝ, (∀ y : ℝ, y = (4 / (x - 2) + x) → y ≥ a) ∧ a = 6) :=
sorry

end min_value_of_function_l39_39205


namespace find_t_l39_39739

open Real

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (t : ℝ) :
  let m := (t + 1, 1)
  let n := (t + 2, 2)
  dot_product (vector_add m n) (vector_sub m n) = 0 → 
  t = -3 :=
by
  intro h
  sorry

end find_t_l39_39739


namespace limit_leq_l39_39937

variables {α : Type*} [LinearOrder α] [TopologicalSpace α] [OrderTopology α]

theorem limit_leq {a_n b_n : ℕ → α} {a b : α}
  (ha : Filter.Tendsto a_n Filter.atTop (nhds a))
  (hb : Filter.Tendsto b_n Filter.atTop (nhds b))
  (h_leq : ∀ n, a_n n ≤ b_n n)
  : a ≤ b :=
by
  -- Proof will be constructed here
  sorry

end limit_leq_l39_39937


namespace angle_C_is_pi_div_3_side_c_is_2_sqrt_3_l39_39558

-- Definitions of the sides and conditions in triangle
variables {a b c : ℝ} {A B C : ℝ}

-- Condition: a + b = 6
axiom sum_of_sides : a + b = 6

-- Condition: Area of triangle ABC is 2 * sqrt(3)
axiom area_of_triangle : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3

-- Condition: a cos B + b cos A = 2c cos C
axiom cos_condition : (a * Real.cos B + b * Real.cos A) / c = 2 * Real.cos C

-- Proof problem 1: Prove that C = π/3
theorem angle_C_is_pi_div_3 (h_cos : Real.cos C = 1/2) : C = Real.pi / 3 :=
sorry

-- Proof problem 2: Prove that c = 2 sqrt(3)
theorem side_c_is_2_sqrt_3 (h_sin : Real.sin C = Real.sqrt 3 / 2) : c = 2 * Real.sqrt 3 :=
sorry

end angle_C_is_pi_div_3_side_c_is_2_sqrt_3_l39_39558


namespace find_first_part_l39_39828

variable (x y : ℕ)

theorem find_first_part (h₁ : x + y = 24) (h₂ : 7 * x + 5 * y = 146) : x = 13 :=
by
  -- The proof is omitted
  sorry

end find_first_part_l39_39828


namespace remainder_n_squared_plus_3n_plus_4_l39_39224

theorem remainder_n_squared_plus_3n_plus_4 (n : ℤ) (h : n % 100 = 99) : (n^2 + 3*n + 4) % 100 = 2 := 
by sorry

end remainder_n_squared_plus_3n_plus_4_l39_39224


namespace problem_1_problem_2_l39_39346

-- Problem 1
theorem problem_1 (x : ℝ) : (2*x - 1) * (2*x - 3) - (1 - 2*x) * (2 - x) = 2*x^2 - 3*x + 1 :=
by {
  -- Proof omitted
  sorry,
}

-- Problem 2
theorem problem_2 (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) : (a^2 - 1) / a * (1 - (2*a + 1) / (a^2 + 2*a + 1)) / (a - 1) = a / (a + 1) :=
by {
  -- Proof omitted
  sorry,
}

end problem_1_problem_2_l39_39346


namespace math_problem_l39_39549

theorem math_problem
  (x y : ℚ)
  (h1 : x + y = 11 / 17)
  (h2 : x - y = 1 / 143) :
  x^2 - y^2 = 11 / 2431 :=
by
  sorry

end math_problem_l39_39549


namespace factor_expression_l39_39051

theorem factor_expression (a : ℝ) : 198 * a ^ 2 + 36 * a + 54 = 18 * (11 * a ^ 2 + 2 * a + 3) :=
by
  sorry

end factor_expression_l39_39051


namespace divisibility_of_binomial_l39_39245

theorem divisibility_of_binomial (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 1) :
    (∀ x : ℕ, 1 ≤ x ∧ x ≤ n-1 → p ∣ Nat.choose n x) ↔ ∃ m : ℕ, n = p^m := sorry

end divisibility_of_binomial_l39_39245


namespace cos_square_sub_exp_zero_l39_39481

theorem cos_square_sub_exp_zero : 
  (cos (30 * Real.pi / 180))^2 - (2 - Real.pi) ^ 0 = -1 / 4 := by
  sorry

end cos_square_sub_exp_zero_l39_39481


namespace value_of_expression_l39_39225

theorem value_of_expression (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / |x| + |y| / y = 2) ∨ (x / |x| + |y| / y = 0) ∨ (x / |x| + |y| / y = -2) :=
by
  sorry

end value_of_expression_l39_39225


namespace students_in_class_l39_39841

theorem students_in_class (total_spent: ℝ) (packs_per_student: ℝ) (sausages_per_student: ℝ) (cost_pack_noodles: ℝ) (cost_sausage: ℝ) (cost_per_student: ℝ) (num_students: ℝ):
  total_spent = 290 → 
  packs_per_student = 2 → 
  sausages_per_student = 1 → 
  cost_pack_noodles = 3.5 → 
  cost_sausage = 7.5 → 
  cost_per_student = packs_per_student * cost_pack_noodles + sausages_per_student * cost_sausage →
  total_spent = cost_per_student * num_students →
  num_students = 20 := 
by
  sorry

end students_in_class_l39_39841


namespace smallest_five_digit_number_divisible_by_primes_l39_39693

theorem smallest_five_digit_number_divisible_by_primes : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
begin
  sorry
end

end smallest_five_digit_number_divisible_by_primes_l39_39693


namespace symmetry_line_intersection_l39_39169

theorem symmetry_line_intersection 
  (k : ℝ) (k_pos : k > 0) (k_ne_one : k ≠ 1)
  (k1 : ℝ) (h_sym : ∀ (P : ℝ × ℝ), (P.2 = k1 * P.1 + 1) ↔ P.2 - 1 = k * (P.1 + 1) + 1)
  (H : ∀ M : ℝ × ℝ, (M.2 = k * M.1 + 1) → (M.1^2 / 4 + M.2^2 = 1)) :
  (k * k1 = 1) ∧ (∀ k : ℝ, ∃ P : ℝ × ℝ, (P.fst = 0) ∧ (P.snd = -5 / 3)) :=
sorry

end symmetry_line_intersection_l39_39169


namespace shaded_area_of_rotated_semicircle_l39_39867

-- Definitions and conditions from the problem
def radius (R : ℝ) : Prop := R > 0
def central_angle (α : ℝ) : Prop := α = 30 * (Real.pi / 180)

-- Lean theorem statement for the proof problem
theorem shaded_area_of_rotated_semicircle (R : ℝ) (hR : radius R) (hα : central_angle 30) : 
  ∃ (area : ℝ), area = (Real.pi * R^2) / 3 :=
by
  -- using proofs of radius and angle conditions
  sorry

end shaded_area_of_rotated_semicircle_l39_39867


namespace cos_sq_minus_exp_equals_neg_one_fourth_l39_39482

theorem cos_sq_minus_exp_equals_neg_one_fourth :
  (Real.cos (30 * Real.pi / 180))^2 - (2 - Real.pi)^0 = -1 / 4 := by
sorry

end cos_sq_minus_exp_equals_neg_one_fourth_l39_39482


namespace bill_original_selling_price_l39_39821

variable (P : ℝ) (S : ℝ) (S_new : ℝ)

theorem bill_original_selling_price :
  (S = P + 0.10 * P) ∧ (S_new = 0.90 * P + 0.27 * P) ∧ (S_new = S + 28) →
  S = 440 :=
by
  intro h
  sorry

end bill_original_selling_price_l39_39821


namespace anna_pays_total_l39_39853

-- Define the conditions
def daily_rental_cost : ℝ := 35
def cost_per_mile : ℝ := 0.25
def rental_days : ℝ := 3
def miles_driven : ℝ := 300

-- Define the total cost function
def total_cost (daily_rental_cost cost_per_mile rental_days miles_driven : ℝ) : ℝ :=
  (daily_rental_cost * rental_days) + (cost_per_mile * miles_driven)

-- The statement to be proved
theorem anna_pays_total : total_cost daily_rental_cost cost_per_mile rental_days miles_driven = 180 :=
by
  sorry

end anna_pays_total_l39_39853


namespace original_price_of_trouser_l39_39763

theorem original_price_of_trouser (sale_price : ℝ) (percent_decrease : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 75) 
  (h2 : percent_decrease = 0.25) 
  (h3 : original_price - percent_decrease * original_price = sale_price) : 
  original_price = 100 :=
by
  sorry

end original_price_of_trouser_l39_39763


namespace smallest_real_number_among_minus3_minus2_0_2_is_minus3_l39_39335

theorem smallest_real_number_among_minus3_minus2_0_2_is_minus3 :
  min (min (-3:ℝ) (-2)) (min 0 2) = -3 :=
by {
    sorry
}

end smallest_real_number_among_minus3_minus2_0_2_is_minus3_l39_39335


namespace simple_interest_is_correct_l39_39475

def Principal : ℝ := 10000
def Rate : ℝ := 0.09
def Time : ℝ := 1

theorem simple_interest_is_correct :
  Principal * Rate * Time = 900 := by
  sorry

end simple_interest_is_correct_l39_39475


namespace least_positive_integer_with_12_factors_is_96_l39_39967

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l39_39967


namespace martinez_family_combined_height_l39_39142

def chiquita_height := 5
def mr_martinez_height := chiquita_height + 2
def mrs_martinez_height := chiquita_height - 1
def son_height := chiquita_height + 3
def combined_height := chiquita_height + mr_martinez_height + mrs_martinez_height + son_height

theorem martinez_family_combined_height : combined_height = 24 :=
by
  sorry

end martinez_family_combined_height_l39_39142


namespace a_eq_zero_l39_39069

noncomputable def f (x a : ℝ) := x^2 - abs (x + a)

theorem a_eq_zero (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 0 :=
by
  sorry

end a_eq_zero_l39_39069


namespace how_much_milk_did_joey_drink_l39_39196

theorem how_much_milk_did_joey_drink (initial_milk : ℚ) (rachel_fraction : ℚ) (joey_fraction : ℚ) :
  initial_milk = 3/4 →
  rachel_fraction = 1/3 →
  joey_fraction = 1/2 →
  joey_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/4 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end how_much_milk_did_joey_drink_l39_39196


namespace nguyen_fabric_needs_l39_39786

def yards_to_feet (yards : ℝ) := yards * 3
def total_fabric_needed (pairs : ℝ) (fabric_per_pair : ℝ) := pairs * fabric_per_pair
def fabric_still_needed (total_needed : ℝ) (already_have : ℝ) := total_needed - already_have

theorem nguyen_fabric_needs :
  let pairs := 7
  let fabric_per_pair := 8.5
  let yards_have := 3.5
  let feet_have := yards_to_feet yards_have
  let total_needed := total_fabric_needed pairs fabric_per_pair
  fabric_still_needed total_needed feet_have = 49 :=
by
  sorry

end nguyen_fabric_needs_l39_39786


namespace polygon_angle_multiple_l39_39466

theorem polygon_angle_multiple (m : ℕ) (h : m ≥ 3) : 
  (∃ k : ℕ, (2 * m - 2) * 180 = k * ((m - 2) * 180)) ↔ (m = 3 ∨ m = 4) :=
by sorry

end polygon_angle_multiple_l39_39466


namespace Melanie_dimes_and_coins_l39_39109

-- Define all given conditions
def d1 : Nat := 7
def d2 : Nat := 8
def d3 : Nat := 4
def r : Float := 2.5

-- State the theorem to prove
theorem Melanie_dimes_and_coins :
  let d_t := d1 + d2 + d3
  let c_t := Float.ofNat d_t * r
  d_t = 19 ∧ c_t = 47.5 :=
by
  sorry

end Melanie_dimes_and_coins_l39_39109


namespace stopped_clock_more_accurate_l39_39644

theorem stopped_clock_more_accurate (slow_correct_time_frequency : ℕ)
  (stopped_correct_time_frequency : ℕ)
  (h1 : slow_correct_time_frequency = 720)
  (h2 : stopped_correct_time_frequency = 2) :
  stopped_correct_time_frequency > slow_correct_time_frequency / 720 :=
by
  sorry

end stopped_clock_more_accurate_l39_39644


namespace probability_all_quit_same_tribe_l39_39949

-- Define the number of participants and the number of tribes
def numParticipants : ℕ := 18
def numTribes : ℕ := 2
def tribeSize : ℕ := 9 -- Each tribe has 9 members

-- Define the problem statement
theorem probability_all_quit_same_tribe : 
  (numParticipants.choose 3) = 816 ∧
  ((tribeSize.choose 3) * numTribes) = 168 ∧
  ((tribeSize.choose 3) * numTribes) / (numParticipants.choose 3) = 7 / 34 :=
by
  sorry

end probability_all_quit_same_tribe_l39_39949


namespace scenarios_one_route_not_visited_l39_39719

open Nat

theorem scenarios_one_route_not_visited :
  let families := 4
  let routes := 4
  ∃ (n : ℕ), n = choose families 2 * permutations (routes - 1) (routes - 1) ∧ n = 144 :=
by
  sorry

end scenarios_one_route_not_visited_l39_39719


namespace susan_cars_fewer_than_carol_l39_39930

theorem susan_cars_fewer_than_carol 
  (Lindsey_cars Carol_cars Susan_cars Cathy_cars : ℕ)
  (h1 : Lindsey_cars = Cathy_cars + 4)
  (h2 : Susan_cars < Carol_cars)
  (h3 : Carol_cars = 2 * Cathy_cars)
  (h4 : Cathy_cars = 5)
  (h5 : Cathy_cars + Carol_cars + Lindsey_cars + Susan_cars = 32) :
  Carol_cars - Susan_cars = 2 :=
sorry

end susan_cars_fewer_than_carol_l39_39930


namespace simplify_expression_l39_39683

theorem simplify_expression (x : ℝ) (h : x = 9) : 
  ((x^9 - 27 * x^6 + 729) / (x^6 - 27) = 730 + 1 / 26) :=
by {
 sorry
}

end simplify_expression_l39_39683


namespace test_tube_full_with_two_amoebas_l39_39500

-- Definition: Each amoeba doubles in number every minute.
def amoeba_doubling (initial : Nat) (minutes : Nat) : Nat :=
  initial * 2 ^ minutes

-- Condition: Starting with one amoeba, the test tube is filled in 60 minutes.
def time_to_fill_one_amoeba := 60

-- Theorem: If two amoebas are placed in the test tube, it takes 59 minutes to fill.
theorem test_tube_full_with_two_amoebas : amoeba_doubling 2 59 = amoeba_doubling 1 time_to_fill_one_amoeba :=
by sorry

end test_tube_full_with_two_amoebas_l39_39500


namespace max_value_of_expression_achieve_max_value_l39_39816

theorem max_value_of_expression : 
  ∀ x : ℝ, -3 * x ^ 2 + 18 * x - 4 ≤ 77 :=
by
  -- Placeholder proof
  sorry

theorem achieve_max_value : 
  ∃ x : ℝ, -3 * x ^ 2 + 18 * x - 4 = 77 :=
by
  -- Placeholder proof
  sorry

end max_value_of_expression_achieve_max_value_l39_39816


namespace negative_sixty_represents_expenditure_l39_39682

def positive_represents_income (x : ℤ) : Prop := x > 0
def negative_represents_expenditure (x : ℤ) : Prop := x < 0

theorem negative_sixty_represents_expenditure :
  negative_represents_expenditure (-60) ∧ abs (-60) = 60 :=
by
  sorry

end negative_sixty_represents_expenditure_l39_39682


namespace n_to_power_eight_plus_n_to_power_seven_plus_one_prime_l39_39362

theorem n_to_power_eight_plus_n_to_power_seven_plus_one_prime (n : ℕ) (hn_pos : n > 0) :
  (Nat.Prime (n^8 + n^7 + 1)) → (n = 1) :=
by
  sorry

end n_to_power_eight_plus_n_to_power_seven_plus_one_prime_l39_39362


namespace candy_crush_ratio_l39_39389

theorem candy_crush_ratio :
  ∃ m : ℕ, (400 + (400 - 70) + (400 - 70) * m = 1390) ∧ (m = 2) :=
by
  sorry

end candy_crush_ratio_l39_39389


namespace charlie_delta_total_products_l39_39327

-- Define the conditions:
def oreo_flavors : ℕ := 8
def milk_types : ℕ := 4
def charlie_constraint : ℕ := oreo_flavors + milk_types
def delta_constraint : ℕ := oreo_flavors
def total_products : ℕ := 5

-- Define the proof problem:
theorem charlie_delta_total_products : 
  (∑ k in Finset.range (total_products + 1),
     (Nat.choose charlie_constraint k) * 
     match total_products - k with 
     | 0 => 1
     | 1 => delta_constraint
     | 2 => (Nat.choose delta_constraint 2) + delta_constraint
     | 3 => (Nat.choose delta_constraint 3) + 
             (delta_constraint * (delta_constraint - 1)) + 
             delta_constraint
     | 4 => (Nat.choose delta_constraint 4) + 
             ((Nat.choose delta_constraint 2) * 
             (Nat.choose (delta_constraint - 2) 2)) / 2 + 
             (delta_constraint * (delta_constraint - 1)) + 
             delta_constraint
     | 5 => (Nat.choose delta_constraint 5) + 
             ((Nat.choose delta_constraint 2) * 
             (Nat.choose (delta_constraint - 3) 3)) + 
             (delta_constraint * (delta_constraint - 1)) + 
             delta_constraint
     | _ => 0) = 25512 := by sorry

end charlie_delta_total_products_l39_39327


namespace least_positive_integer_with_12_factors_is_72_l39_39973

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l39_39973


namespace original_cost_price_of_car_l39_39318

theorem original_cost_price_of_car (x : ℝ) (y : ℝ) (h1 : y = 0.87 * x) (h2 : 1.20 * y = 54000) :
  x = 54000 / 1.044 :=
by
  sorry

end original_cost_price_of_car_l39_39318


namespace train_length_l39_39811

theorem train_length (L : ℝ) (h1 : 46 - 36 = 10) (h2 : 45 * (10 / 3600) = 1 / 8) : L = 62.5 :=
by
  sorry

end train_length_l39_39811


namespace planks_from_friends_l39_39039

theorem planks_from_friends :
  let total_planks := 200
  let planks_from_storage := total_planks / 4
  let planks_from_parents := total_planks / 2
  let planks_from_store := 30
  let planks_from_friends := total_planks - (planks_from_storage + planks_from_parents + planks_from_store)
  planks_from_friends = 20 :=
by
  let total_planks := 200
  let planks_from_storage := total_planks / 4
  let planks_from_parents := total_planks / 2
  let planks_from_store := 30
  let planks_from_friends := total_planks - (planks_from_storage + planks_from_parents + planks_from_store)
  rfl

end planks_from_friends_l39_39039


namespace ratio_of_intercepts_l39_39961

theorem ratio_of_intercepts (b s t : ℝ) (h1 : s = -2 * b / 5) (h2 : t = -3 * b / 7) :
  s / t = 14 / 15 :=
by
  sorry

end ratio_of_intercepts_l39_39961


namespace second_carpenter_days_l39_39317

theorem second_carpenter_days (x : ℚ) (h1 : 1 / 5 + 1 / x = 1 / 2) : x = 10 / 3 :=
by
  sorry

end second_carpenter_days_l39_39317


namespace lena_glued_friends_pictures_l39_39767

-- Define the conditions
def clippings_per_friend : ℕ := 3
def glue_per_clipping : ℕ := 6
def total_glue : ℕ := 126

-- Define the proof problem statement
theorem lena_glued_friends_pictures : 
    ∃ (F : ℕ), F * (clippings_per_friend * glue_per_clipping) = total_glue ∧ F = 7 := 
by
  sorry

end lena_glued_friends_pictures_l39_39767


namespace frank_problems_each_type_l39_39676

theorem frank_problems_each_type (bill_total : ℕ) (ryan_ratio bill_total_ratio : ℕ) (frank_ratio ryan_total : ℕ) (types : ℕ)
  (h1 : bill_total = 20)
  (h2 : ryan_ratio = 2)
  (h3 : bill_total_ratio = bill_total * ryan_ratio)
  (h4 : ryan_total = bill_total_ratio)
  (h5 : frank_ratio = 3)
  (h6 : ryan_total * frank_ratio = ryan_total) :
  (ryan_total * frank_ratio) / types = 30 :=
by
  sorry

end frank_problems_each_type_l39_39676


namespace don_travel_time_to_hospital_l39_39168

noncomputable def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def time_to_travel (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem don_travel_time_to_hospital :
  let speed_mary := 60
  let speed_don := 30
  let time_mary_minutes := 15
  let time_mary_hours := time_mary_minutes / 60
  let distance := distance_traveled speed_mary time_mary_hours
  let time_don_hours := time_to_travel distance speed_don
  time_don_hours * 60 = 30 :=
by
  sorry

end don_travel_time_to_hospital_l39_39168


namespace least_positive_integer_with_12_factors_l39_39996

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l39_39996


namespace deriv_prob1_deriv_prob2_l39_39058

noncomputable def prob1 (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem deriv_prob1 : ∀ x, deriv prob1 x = -x * Real.sin x :=
by 
  sorry

noncomputable def prob2 (x : ℝ) : ℝ := x / (Real.exp x - 1)

theorem deriv_prob2 : ∀ x, x ≠ 0 → deriv prob2 x = (Real.exp x * (1 - x) - 1) / (Real.exp x - 1)^2 :=
by
  sorry

end deriv_prob1_deriv_prob2_l39_39058


namespace total_toothpicks_480_l39_39283

/- Define the number of toothpicks per side -/
def toothpicks_per_side : ℕ := 15

/- Define the number of horizontal lines in the grid -/
def horizontal_lines (sides : ℕ) : ℕ := sides + 1

/- Define the number of vertical lines in the grid -/
def vertical_lines (sides : ℕ) : ℕ := sides + 1

/- Define the total number of toothpicks used -/
def total_toothpicks (sides : ℕ) : ℕ :=
  (horizontal_lines sides * toothpicks_per_side) + (vertical_lines sides * toothpicks_per_side)

/- Theorem statement: Prove that for a grid with 15 toothpicks per side, the total number of toothpicks is 480 -/
theorem total_toothpicks_480 : total_toothpicks 15 = 480 :=
  sorry

end total_toothpicks_480_l39_39283


namespace selling_price_per_pound_l39_39487

-- Definitions based on conditions
def cost_per_pound_type1 : ℝ := 2.00
def cost_per_pound_type2 : ℝ := 3.00
def weight_type1 : ℝ := 64
def weight_type2 : ℝ := 16
def total_weight : ℝ := 80

-- The selling price per pound of the mixture
theorem selling_price_per_pound :
  let total_cost := (weight_type1 * cost_per_pound_type1) + (weight_type2 * cost_per_pound_type2)
  (total_cost / total_weight) = 2.20 :=
by
  sorry

end selling_price_per_pound_l39_39487


namespace find_r_minus_p_l39_39435

-- Define the variables and conditions
variables (p q r A1 A2 : ℝ)
noncomputable def arithmetic_mean (x y : ℝ) := (x + y) / 2

-- Given conditions in the problem
axiom hA1 : arithmetic_mean p q = 10
axiom hA2 : arithmetic_mean q r = 25

-- Statement to prove
theorem find_r_minus_p : r - p = 30 :=
by {
  -- write the necessary proof steps here
  sorry
}

end find_r_minus_p_l39_39435


namespace sum_equals_1584_l39_39190

-- Let's define the function that computes the sum, according to the pattern
def sumPattern : ℕ → ℝ
  | 0 => 0
  | k + 1 => if (k + 1) % 3 = 0 then - (k + 1) + sumPattern k
             else (k + 1) + sumPattern k

-- This function defines the problem setting and the final expected result
theorem sum_equals_1584 : sumPattern 99 = 1584 := by
  sorry

end sum_equals_1584_l39_39190


namespace total_nails_needed_l39_39076

-- Define the conditions
def nails_already_have : ℕ := 247
def nails_found : ℕ := 144
def nails_to_buy : ℕ := 109

-- The statement to prove
theorem total_nails_needed : nails_already_have + nails_found + nails_to_buy = 500 := by
  -- The proof goes here
  sorry

end total_nails_needed_l39_39076


namespace computation_correct_l39_39351

theorem computation_correct : 12 * ((216 / 3) + (36 / 6) + (16 / 8) + 2) = 984 := 
by 
  sorry

end computation_correct_l39_39351


namespace least_pos_int_with_12_pos_factors_is_72_l39_39971

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l39_39971


namespace Joey_study_time_l39_39240

theorem Joey_study_time :
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96 := by
  let weekday_hours_per_night := 2
  let nights_per_week := 5
  let weekend_hours_per_day := 3
  let days_per_weekend := 2
  let weeks_until_exam := 6
  show (weekday_hours_per_night * nights_per_week + weekend_hours_per_day * days_per_weekend) * weeks_until_exam = 96
  -- define study times
  let weekday_hours_per_week := weekday_hours_per_night * nights_per_week
  let weekend_hours_per_week := weekend_hours_per_day * days_per_weekend
  -- sum times per week
  let total_hours_per_week := weekday_hours_per_week + weekend_hours_per_week
  -- multiply by weeks until exam
  let total_study_time := total_hours_per_week * weeks_until_exam
  have h : total_study_time = 96 := by sorry
  exact h

end Joey_study_time_l39_39240


namespace find_xy_l39_39876

theorem find_xy (x y : ℝ) (h : |x^3 - 1/8| + Real.sqrt (y - 4) = 0) : x * y = 2 :=
by
  sorry

end find_xy_l39_39876


namespace brendan_taxes_correct_l39_39020

-- Definitions based on conditions
def hourly_wage : ℝ := 6
def shifts : (ℕ × ℕ) := (2, 8)
def additional_shift : ℕ := 12
def tip_rate : ℝ := 12
def tax_rate : ℝ := 0.20
def tip_reporting_fraction : ℝ := 1 / 3

-- Calculation based on conditions
noncomputable def total_hours : ℕ := (shifts.1 * shifts.2) + additional_shift
noncomputable def wage_income : ℝ := hourly_wage * total_hours
noncomputable def total_tips : ℝ := tip_rate * total_hours
noncomputable def reported_tips : ℝ := total_tips * tip_reporting_fraction
noncomputable def total_reported_income : ℝ := wage_income + reported_tips
noncomputable def taxes_paid : ℝ := total_reported_income * tax_rate

-- The proof problem statement
theorem brendan_taxes_correct : taxes_paid = 56 := by {
  sorry
}

end brendan_taxes_correct_l39_39020


namespace div_result_l39_39193

theorem div_result : 2.4 / 0.06 = 40 := 
sorry

end div_result_l39_39193


namespace veridux_male_associates_l39_39674

theorem veridux_male_associates (total_employees female_employees total_managers female_managers : ℕ)
  (h1 : total_employees = 250)
  (h2 : female_employees = 90)
  (h3 : total_managers = 40)
  (h4 : female_managers = 40) :
  total_employees - female_employees = 160 :=
by
  sorry

end veridux_male_associates_l39_39674


namespace prime_power_sum_l39_39579

theorem prime_power_sum (a b p : ℕ) (hp : p = a ^ b + b ^ a) (ha_prime : Nat.Prime a) (hb_prime : Nat.Prime b) (hp_prime : Nat.Prime p) : 
  p = 17 := 
sorry

end prime_power_sum_l39_39579


namespace fraction_of_largest_jar_filled_l39_39809

theorem fraction_of_largest_jar_filled
  (C1 C2 C3 : ℝ)
  (h1 : C1 < C2)
  (h2 : C2 < C3)
  (h3 : C1 / 6 = C2 / 5)
  (h4 : C2 / 5 = C3 / 7) :
  (C1 / 6 + C2 / 5) / C3 = 2 / 7 := sorry

end fraction_of_largest_jar_filled_l39_39809


namespace distance_Bella_Galya_l39_39092

theorem distance_Bella_Galya 
    (AB BV VG GD : ℝ)
    (DB : AB + 3 * BV + 2 * VG + GD = 700)
    (DV : AB + 2 * BV + 2 * VG + GD = 600)
    (DG : AB + 2 * BV + 3 * VG + GD = 650) :
    BV + VG = 150 := 
by
  -- Proof goes here
  sorry

end distance_Bella_Galya_l39_39092


namespace tank_fish_count_l39_39629

theorem tank_fish_count (total_fish blue_fish : ℕ) 
  (h1 : blue_fish = total_fish / 3)
  (h2 : 10 * 2 = blue_fish) : 
  total_fish = 60 :=
sorry

end tank_fish_count_l39_39629


namespace cameron_list_count_l39_39023

theorem cameron_list_count : 
  (∃ (n m : ℕ), n = 900 ∧ m = 27000 ∧ (∀ k : ℕ, (30 * k) ≥ n ∧ (30 * k) ≤ m → ∃ count : ℕ, count = 871)) :=
by
  sorry

end cameron_list_count_l39_39023


namespace power_difference_expression_l39_39512

theorem power_difference_expression : 
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * (30^1001) :=
by
  sorry

end power_difference_expression_l39_39512


namespace algebraic_inequality_l39_39724

theorem algebraic_inequality (n : ℕ) (h : n ≥ 3) (x : Fin n → ℝ) (h_nonneg : ∀ i, 0 ≤ x i) :
  (n + 1) * (∑ i, x i) ^ 2 * (∑ i, (x i) ^ 2) + (n - 2) * (∑ i, (x i) ^ 2) ^ 2 ≥ 
  (∑ i, x i) ^ 4 + (2 * n - 2) * (∑ i, x i) * (∑ i, (x i) ^ 3) :=
by
  sorry

end algebraic_inequality_l39_39724


namespace find_omega_value_l39_39554

theorem find_omega_value (ω : ℝ) (h : ω > 0) (h_dist : (1/2) * (2 * π / ω) = π / 6) : ω = 6 :=
by
  sorry

end find_omega_value_l39_39554


namespace distance_Bella_to_Galya_l39_39094

theorem distance_Bella_to_Galya (D_B D_V D_G : ℕ) (BV VG : ℕ)
  (hD_B : D_B = 700)
  (hD_V : D_V = 600)
  (hD_G : D_G = 650)
  (hBV : BV = 100)
  (hVG : VG = 50)
  : BV + VG = 150 := by
  sorry

end distance_Bella_to_Galya_l39_39094


namespace probability_hit_10_or_7_ring_probability_below_7_ring_l39_39826

noncomputable def P_hit_10_ring : ℝ := 0.21
noncomputable def P_hit_9_ring : ℝ := 0.23
noncomputable def P_hit_8_ring : ℝ := 0.25
noncomputable def P_hit_7_ring : ℝ := 0.28
noncomputable def P_below_7_ring : ℝ := 0.03

theorem probability_hit_10_or_7_ring :
  P_hit_10_ring + P_hit_7_ring = 0.49 :=
  by sorry

theorem probability_below_7_ring :
  P_below_7_ring = 0.03 :=
  by sorry

end probability_hit_10_or_7_ring_probability_below_7_ring_l39_39826


namespace calculate_value_l39_39343

theorem calculate_value : 12 * ((1/3 : ℝ) + (1/4) - (1/12))⁻¹ = 24 :=
by
  sorry

end calculate_value_l39_39343


namespace brendan_taxes_l39_39021

def total_hours (num_8hr_shifts : ℕ) (num_12hr_shifts : ℕ) : ℕ :=
  (num_8hr_shifts * 8) + (num_12hr_shifts * 12)

def total_wage (hourly_wage : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_wage * hours_worked

def total_tips (hourly_tips : ℕ) (hours_worked : ℕ) : ℕ :=
  hourly_tips * hours_worked

def reported_tips (total_tips : ℕ) (report_fraction : ℕ) : ℕ :=
  total_tips / report_fraction

def reported_income (wage : ℕ) (tips : ℕ) : ℕ :=
  wage + tips

def taxes (income : ℕ) (tax_rate : ℚ) : ℚ :=
  income * tax_rate

theorem brendan_taxes (num_8hr_shifts num_12hr_shifts : ℕ)
    (hourly_wage hourly_tips report_fraction : ℕ) (tax_rate : ℚ) :
    (hourly_wage = 6) →
    (hourly_tips = 12) →
    (report_fraction = 3) →
    (tax_rate = 0.2) →
    (num_8hr_shifts = 2) →
    (num_12hr_shifts = 1) →
    taxes (reported_income (total_wage hourly_wage (total_hours num_8hr_shifts num_12hr_shifts))
            (reported_tips (total_tips hourly_tips (total_hours num_8hr_shifts num_12hr_shifts))
            report_fraction))
          tax_rate = 56 :=
by
  intros
  sorry

end brendan_taxes_l39_39021


namespace smallest_five_digit_divisible_by_primes_l39_39711

theorem smallest_five_digit_divisible_by_primes : 
  let primes := [2, 3, 5, 7, 11] in
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11))) in
  let five_digit_threshold := 10000 in
  ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
by
  let primes := [2, 3, 5, 7, 11]
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))
  have lcm_2310 : lcm_primes = 2310 := sorry
  let five_digit_threshold := 10000
  have exists_n : ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
    sorry
  exists_intro 5
  have 5_condition : 5 > 0 := sorry
  have 2310_5_condition : 2310 * 5 >= five_digit_threshold := sorry
  have answer : 2310 * 5 = 11550 := sorry
  exact  ⟨5, 5_condition, 2310_5_condition, answer⟩
  exact ⟨5, 5 > 0, 2310 * 5 ≥ 10000, 2310 * 5 = 11550⟩
  sorry

end smallest_five_digit_divisible_by_primes_l39_39711


namespace chips_per_cookie_l39_39685

theorem chips_per_cookie (total_cookies : ℕ) (uneaten_chips : ℕ) (uneaten_cookies : ℕ) (h1 : total_cookies = 4 * 12) (h2 : uneaten_cookies = total_cookies / 2) (h3 : uneaten_chips = 168) : 
  uneaten_chips / uneaten_cookies = 7 :=
by sorry

end chips_per_cookie_l39_39685


namespace krakozyabr_count_l39_39420

variable (n H W T : ℕ)
variable (h1 : H = 5 * n) -- 20% of the 'krakozyabrs' with horns also have wings
variable (h2 : W = 4 * n) -- 25% of the 'krakozyabrs' with wings also have horns
variable (h3 : T = H + W - n) -- Total number of 'krakozyabrs' using inclusion-exclusion
variable (h4 : 25 < T)
variable (h5 : T < 35)

theorem krakozyabr_count : T = 32 := by
  sorry

end krakozyabr_count_l39_39420


namespace area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3_l39_39795

noncomputable def area_enclosed_by_sine_and_line : ℝ :=
  (∫ x in (Real.pi / 6)..(5 * Real.pi / 6), (Real.sin x - 1 / 2))

theorem area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3 :
  area_enclosed_by_sine_and_line = Real.sqrt 3 - Real.pi / 3 := by
  sorry

end area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3_l39_39795


namespace sum_of_first_three_terms_l39_39270

theorem sum_of_first_three_terms (a : ℕ → ℤ) 
  (h4 : a 4 = 8) 
  (h5 : a 5 = 12) 
  (h6 : a 6 = 16) : 
  a 1 + a 2 + a 3 = 0 :=
by
  sorry

end sum_of_first_three_terms_l39_39270


namespace ronalds_egg_sharing_l39_39124

theorem ronalds_egg_sharing (total_eggs : ℕ) (eggs_per_friend : ℕ) (num_friends : ℕ) 
  (h1 : total_eggs = 16) (h2 : eggs_per_friend = 2) 
  (h3 : num_friends = total_eggs / eggs_per_friend) : 
  num_friends = 8 := 
by 
  sorry

end ronalds_egg_sharing_l39_39124


namespace check_blank_value_l39_39567

/-- Define required constants and terms. -/
def six_point_five : ℚ := 6 + 1/2
def two_thirds : ℚ := 2/3
def three_point_five : ℚ := 3 + 1/2
def one_and_eight_fifteenths : ℚ := 1 + 8/15
def blank : ℚ := 3 + 1/20
def seventy_one_point_ninety_five : ℚ := 71 + 95/100

/-- The translated assumption and statement to be proved: -/
theorem check_blank_value :
  (six_point_five - two_thirds) / three_point_five - one_and_eight_fifteenths * (blank + seventy_one_point_ninety_five) = 1 :=
sorry

end check_blank_value_l39_39567


namespace exists_perpendicular_area_bisector_l39_39355

-- Conditions
variables {A B C M N : Point}

-- Definitions: Triangle ABC, line MN perpendicular to AB, line bisecting the area
def triangle_ABC (A B C : Point) : Triangle := ⟨A, B, C⟩
def M_on_AB (A B M : Point) : Prop := M ∈ Line(A, B)
def perpendicular_to_AB (A B M N : Point) : Prop := Perpendicular(Line(A, B), Line(M, N))
def area_bisector (A B C M N : Point) : Prop := 
  let T := Triangle(⟨A, M, N⟩) in 
  (area A B C) = 2 * (area A M N)

-- Main theorem
theorem exists_perpendicular_area_bisector (A B C : Point) :
  ∃ M N : Point, M_on_AB A B M ∧ perpendicular_to_AB A B M N ∧ area_bisector A B C M N :=
  by sorry

end exists_perpendicular_area_bisector_l39_39355


namespace triangle_area_ratio_l39_39159

theorem triangle_area_ratio (a b c : ℕ) (d e f : ℕ) 
  (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) 
  (h4 : d = 9) (h5 : e = 12) (h6 : f = 15) 
  (GHI_right : a^2 + b^2 = c^2)
  (JKL_right : d^2 + e^2 = f^2):
  (0.5 * a * b) / (0.5 * d * e) = 4 / 9 := 
by 
  sorry

end triangle_area_ratio_l39_39159


namespace sum_of_intercepts_eq_16_l39_39016

noncomputable def line_eq (x y : ℝ) : Prop :=
  y + 3 = -3 * (x - 5)

def x_intercept : ℝ := 4
def y_intercept : ℝ := 12

theorem sum_of_intercepts_eq_16 : 
  (line_eq x_intercept 0) ∧ (line_eq 0 y_intercept) → x_intercept + y_intercept = 16 :=
by
  intros h
  sorry

end sum_of_intercepts_eq_16_l39_39016


namespace paint_needed_l39_39437

theorem paint_needed (wall_area : ℕ) (coverage_per_gallon : ℕ) (number_of_coats : ℕ) (h_wall_area : wall_area = 600) (h_coverage_per_gallon : coverage_per_gallon = 400) (h_number_of_coats : number_of_coats = 2) : 
    ((number_of_coats * wall_area) / coverage_per_gallon) = 3 :=
by
  sorry

end paint_needed_l39_39437


namespace sum_first_15_odd_integers_l39_39292

theorem sum_first_15_odd_integers : 
  let seq : List ℕ := List.range' 1 30 2,
      sum_odd : ℕ := seq.sum
  in 
  sum_odd = 225 :=
by 
  sorry

end sum_first_15_odd_integers_l39_39292


namespace triangle_area_ratio_l39_39157

noncomputable def area_ratio (a b c d e f : ℕ) : ℚ :=
  (a * b) / (d * e)

theorem triangle_area_ratio : area_ratio 6 8 10 9 12 15 = 4 / 9 :=
by
  sorry

end triangle_area_ratio_l39_39157


namespace sum_of_first_n_odd_numbers_l39_39296

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l39_39296


namespace cricket_team_members_l39_39959

theorem cricket_team_members (n : ℕ)
    (captain_age : ℕ) (wicket_keeper_age : ℕ) (average_age : ℕ)
    (remaining_average_age : ℕ) (total_age : ℕ) (remaining_players : ℕ) :
    captain_age = 27 →
    wicket_keeper_age = captain_age + 3 →
    average_age = 24 →
    remaining_average_age = average_age - 1 →
    total_age = average_age * n →
    remaining_players = n - 2 →
    total_age = captain_age + wicket_keeper_age + remaining_average_age * remaining_players →
    n = 11 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end cricket_team_members_l39_39959


namespace digit_sum_of_nines_l39_39273

theorem digit_sum_of_nines (k : ℕ) (n : ℕ) (h : n = 9 * (10^k - 1) / 9):
  (8 + 9 * (k - 1) + 1 = 500) → k = 55 := 
by 
  sorry

end digit_sum_of_nines_l39_39273


namespace decagon_diagonals_l39_39743

-- Define the number of sides of the polygon
def n : ℕ := 10

-- Calculate the number of diagonals in an n-sided polygon
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem that the number of diagonals in a decagon is 35
theorem decagon_diagonals : number_of_diagonals n = 35 := by
  sorry

end decagon_diagonals_l39_39743


namespace yoojung_notebooks_l39_39647

theorem yoojung_notebooks (N : ℕ) (h : (N - 5) / 2 = 4) : N = 13 :=
by
  sorry

end yoojung_notebooks_l39_39647


namespace nguyen_fabric_needs_l39_39787

def yards_to_feet (yards : ℝ) := yards * 3
def total_fabric_needed (pairs : ℝ) (fabric_per_pair : ℝ) := pairs * fabric_per_pair
def fabric_still_needed (total_needed : ℝ) (already_have : ℝ) := total_needed - already_have

theorem nguyen_fabric_needs :
  let pairs := 7
  let fabric_per_pair := 8.5
  let yards_have := 3.5
  let feet_have := yards_to_feet yards_have
  let total_needed := total_fabric_needed pairs fabric_per_pair
  fabric_still_needed total_needed feet_have = 49 :=
by
  sorry

end nguyen_fabric_needs_l39_39787


namespace least_positive_integer_with_12_factors_l39_39998

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l39_39998


namespace not_divisible_by_44_l39_39447

theorem not_divisible_by_44 (k : ℤ) (n : ℤ) (h1 : n = k * (k + 1) * (k + 2)) (h2 : 11 ∣ n) : ¬ (44 ∣ n) :=
sorry

end not_divisible_by_44_l39_39447


namespace casper_initial_candies_l39_39191

theorem casper_initial_candies : 
  ∃ x : ℕ, 
    (∃ y1 : ℕ, y1 = x / 2 - 3) ∧
    (∃ y2 : ℕ, y2 = y1 / 2 - 5) ∧
    (∃ y3 : ℕ, y3 = y2 / 2 - 2) ∧
    (y3 = 10) ∧
    x = 122 := 
sorry

end casper_initial_candies_l39_39191


namespace travel_agency_comparison_l39_39275

variable (x : ℕ)

def cost_A (x : ℕ) : ℕ := 150 * x
def cost_B (x : ℕ) : ℕ := 160 * x - 160

theorem travel_agency_comparison (x : ℕ) : 150 * x < 160 * x - 160 → x > 16 :=
by
  intro h
  linarith

end travel_agency_comparison_l39_39275


namespace solve_xyz_l39_39068

variable {x y z : ℝ}

theorem solve_xyz (h1 : (x + y + z) * (xy + xz + yz) = 35) (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) : x * y * z = 8 := 
by
  sorry

end solve_xyz_l39_39068


namespace min_abs_diff_l39_39877

theorem min_abs_diff (a b c d : ℝ) (h1 : |a - b| = 5) (h2 : |b - c| = 8) (h3 : |c - d| = 10) : 
  ∃ m, m = |a - d| ∧ m = 3 := 
by 
  sorry

end min_abs_diff_l39_39877


namespace carol_rectangle_width_l39_39681

def carol_width (lengthC : ℕ) (widthJ : ℕ) (lengthJ : ℕ) (widthC : ℕ) :=
  lengthC * widthC = lengthJ * widthJ

theorem carol_rectangle_width 
  {lengthC widthJ lengthJ : ℕ} (h1 : lengthC = 8)
  (h2 : widthJ = 30) (h3 : lengthJ = 4)
  (h4 : carol_width lengthC widthJ lengthJ 15) : 
  widthC = 15 :=
by 
  subst h1
  subst h2
  subst h3
  sorry -- proof not required

end carol_rectangle_width_l39_39681


namespace holes_remaining_unfilled_l39_39932

def total_holes : ℕ := 8
def filled_percentage : ℝ := 0.75

theorem holes_remaining_unfilled : total_holes - (filled_percentage * total_holes).to_nat = 2 :=
by
  sorry

end holes_remaining_unfilled_l39_39932


namespace least_positive_integer_with_12_factors_is_972_l39_39990

theorem least_positive_integer_with_12_factors_is_972 : ∃ k : ℕ, (∀ n : ℕ, (∃ (d : ℕ), d * k = n) ↔ n = 12) ∧ k = 972 := sorry

end least_positive_integer_with_12_factors_is_972_l39_39990


namespace inheritance_value_l39_39441

def inheritance_proof (x : ℝ) (federal_tax_ratio : ℝ) (state_tax_ratio : ℝ) (total_tax : ℝ) : Prop :=
  let federal_taxes := federal_tax_ratio * x
  let remaining_after_federal := x - federal_taxes
  let state_taxes := state_tax_ratio * remaining_after_federal
  let total_taxes := federal_taxes + state_taxes
  total_taxes = total_tax

theorem inheritance_value :
  inheritance_proof 41379 0.25 0.15 15000 :=
by
  sorry

end inheritance_value_l39_39441


namespace tangent_line_and_point_l39_39071

theorem tangent_line_and_point (x0 y0 k: ℝ) (hx0 : x0 ≠ 0) 
  (hC : y0 = x0^3 - 3 * x0^2 + 2 * x0) (hl : y0 = k * x0) 
  (hk_tangent : k = 3 * x0^2 - 6 * x0 + 2) : 
  (k = -1/4) ∧ (x0 = 3/2) ∧ (y0 = -3/8) :=
by
  sorry

end tangent_line_and_point_l39_39071


namespace find_integer_l39_39901

theorem find_integer
  (x y : ℤ)
  (h1 : 4 * x + y = 34)
  (h2 : 2 * x - y = 20)
  (h3 : y^2 = 4) :
  y = -2 :=
by
  sorry

end find_integer_l39_39901


namespace confidence_level_unrelated_l39_39082

noncomputable def chi_squared_value : ℝ := 8.654

theorem confidence_level_unrelated :
  chi_squared_value > 6.635 →
  (100 - 99) = 1 :=
by
  sorry

end confidence_level_unrelated_l39_39082


namespace sum_of_first_15_odd_integers_l39_39311

theorem sum_of_first_15_odd_integers :
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  sum = 225 :=
by
  let first_term := 1
  let n := 15
  let d := 2
  let last_term := first_term + (n - 1) * d
  let sum := (first_term + last_term) * n / 2
  exact Eq.refl 225

end sum_of_first_15_odd_integers_l39_39311


namespace smallest_positive_five_digit_number_divisible_by_first_five_primes_l39_39706

theorem smallest_positive_five_digit_number_divisible_by_first_five_primes :
  ∃ n : ℕ, (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ 10000 ≤ n ∧ n = 11550 :=
by
  use 11550
  split
  · intros p hp
    fin_cases hp <;> norm_num
  split
  · norm_num
  rfl

end smallest_positive_five_digit_number_divisible_by_first_five_primes_l39_39706


namespace perimeter_of_irregular_pentagonal_picture_frame_l39_39933

theorem perimeter_of_irregular_pentagonal_picture_frame 
  (base : ℕ) (left_side : ℕ) (right_side : ℕ) (top_left_diagonal_side : ℕ) (top_right_diagonal_side : ℕ)
  (h_base : base = 10) (h_left_side : left_side = 12) (h_right_side : right_side = 11)
  (h_top_left_diagonal_side : top_left_diagonal_side = 6) (h_top_right_diagonal_side : top_right_diagonal_side = 7) :
  base + left_side + right_side + top_left_diagonal_side + top_right_diagonal_side = 46 :=
by {
  sorry
}

end perimeter_of_irregular_pentagonal_picture_frame_l39_39933


namespace steves_initial_emails_l39_39595

theorem steves_initial_emails (E : ℝ) (ht : E / 2 = (0.6 * E) + 120) : E = 400 :=
  by sorry

end steves_initial_emails_l39_39595


namespace frank_problems_per_type_l39_39677

-- Definitions based on the problem conditions
def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def types_of_problems : ℕ := 4

-- The proof statement equivalent to the math problem
theorem frank_problems_per_type : frank_problems / types_of_problems = 30 :=
by 
  -- skipping the proof steps
  sorry

end frank_problems_per_type_l39_39677


namespace fraction_division_l39_39163

theorem fraction_division: 
  ((3 + 1 / 2) / 7) / (5 / 3) = 3 / 10 := 
by 
  sorry

end fraction_division_l39_39163


namespace books_in_library_final_l39_39151

variable (initial_books : ℕ) (books_taken_out_tuesday : ℕ) 
          (books_returned_wednesday : ℕ) (books_taken_out_thursday : ℕ)

def books_left_in_library (initial_books books_taken_out_tuesday 
                          books_returned_wednesday books_taken_out_thursday : ℕ) : ℕ :=
  initial_books - books_taken_out_tuesday + books_returned_wednesday - books_taken_out_thursday

theorem books_in_library_final 
  (initial_books := 250) 
  (books_taken_out_tuesday := 120) 
  (books_returned_wednesday := 35) 
  (books_taken_out_thursday := 15) :
  books_left_in_library initial_books books_taken_out_tuesday 
                        books_returned_wednesday books_taken_out_thursday = 150 :=
by 
  sorry

end books_in_library_final_l39_39151


namespace probability_X_3_l39_39559

noncomputable def probability_of_two_reds_in_three_draws (white_balls red_balls : ℕ) : ℚ :=
  let total_balls := white_balls + red_balls in
  (2 * ((white_balls / total_balls) * (red_balls / total_balls) * (red_balls / total_balls))).to_rat

theorem probability_X_3 (white_balls red_balls : ℕ) (h_white : white_balls = 5) (h_red : red_balls = 3) :
  probability_of_two_reds_in_three_draws white_balls red_balls = (45 / 256 : ℚ) :=
by
  rw [h_white, h_red]
  -- Skip the proof
  sorry

end probability_X_3_l39_39559


namespace number_of_license_plates_l39_39409

-- Define the alphabet size and digit size constants.
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the number of letters in the license plate.
def letters_in_plate : ℕ := 3

-- Define the number of digits in the license plate.
def digits_in_plate : ℕ := 4

-- Calculating the total number of license plates possible as (26^3) * (10^4).
theorem number_of_license_plates : 
  (num_letters ^ letters_in_plate) * (num_digits ^ digits_in_plate) = 175760000 :=
by
  sorry

end number_of_license_plates_l39_39409


namespace frustum_surface_area_l39_39144

theorem frustum_surface_area (r r' l : ℝ) (h_r : r = 1) (h_r' : r' = 4) (h_l : l = 5) :
  π * r^2 + π * r'^2 + π * (r + r') * l = 42 * π :=
by
  rw [h_r, h_r', h_l]
  norm_num
  sorry

end frustum_surface_area_l39_39144


namespace no_integer_solutions_to_system_l39_39508

theorem no_integer_solutions_to_system :
  ¬ ∃ (x y z : ℤ),
    x^2 - 2 * x * y + y^2 - z^2 = 17 ∧
    -x^2 + 3 * y * z + 3 * z^2 = 27 ∧
    x^2 - x * y + 5 * z^2 = 50 :=
by
  sorry

end no_integer_solutions_to_system_l39_39508


namespace possible_values_x2_y2_z2_l39_39375

theorem possible_values_x2_y2_z2 {x y z : ℤ}
    (h1 : x + y + z = 3)
    (h2 : x^3 + y^3 + z^3 = 3) : (x^2 + y^2 + z^2 = 3) ∨ (x^2 + y^2 + z^2 = 57) :=
by sorry

end possible_values_x2_y2_z2_l39_39375


namespace exists_231_four_digit_integers_l39_39195

theorem exists_231_four_digit_integers (n : ℕ) : 
  (∃ A B C D : ℕ, 
     A ≠ 0 ∧ 
     1 ≤ A ∧ A ≤ 9 ∧ 
     0 ≤ B ∧ B ≤ 9 ∧ 
     0 ≤ C ∧ C ≤ 9 ∧ 
     0 ≤ D ∧ D ≤ 9 ∧ 
     999 * (A - D) + 90 * (B - C) = n^3) ↔ n = 231 :=
by sorry

end exists_231_four_digit_integers_l39_39195


namespace arithmetic_mean_of_three_digit_multiples_of_8_l39_39812

-- Define the conditions given in the problem
def smallest_three_digit_multiple_of_8 := 104
def largest_three_digit_multiple_of_8 := 992
def common_difference := 8

-- Define the sequence as an arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℕ :=
  smallest_three_digit_multiple_of_8 + n * common_difference

-- Calculate the number of terms in the sequence
def number_of_terms : ℕ :=
  (largest_three_digit_multiple_of_8 - smallest_three_digit_multiple_of_8) / common_difference + 1

-- Calculate the sum of the arithmetic sequence
def sum_of_sequence : ℕ :=
  (number_of_terms * (smallest_three_digit_multiple_of_8 + largest_three_digit_multiple_of_8)) / 2

-- Calculate the arithmetic mean
def arithmetic_mean : ℕ :=
  sum_of_sequence / number_of_terms

-- The statement to be proved
theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  arithmetic_mean = 548 :=
by
  sorry

end arithmetic_mean_of_three_digit_multiples_of_8_l39_39812


namespace sum_of_a_and_b_l39_39536

noncomputable def log_function (a b x : ℝ) : ℝ := Real.log (x + b) / Real.log a

theorem sum_of_a_and_b (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : log_function a b 2 = 1)
                      (h4 : ∃ x : ℝ, log_function a b x = 8 ∧ log_function a b x = 2) :
  a + b = 4 :=
by
  sorry

end sum_of_a_and_b_l39_39536


namespace geometric_sequence_a3_q_l39_39568

theorem geometric_sequence_a3_q (a_5 a_4 a_3 a_2 a_1 : ℝ) (q : ℝ) :
  a_5 - a_1 = 15 →
  a_4 - a_2 = 6 →
  (q = 2 ∧ a_3 = 4) ∨ (q = 1/2 ∧ a_3 = -4) :=
by
  sorry

end geometric_sequence_a3_q_l39_39568


namespace C1_C2_properties_l39_39537

-- Definitions based on the conditions
def C1_parametric (a b θ : ℝ) : ℝ × ℝ := (a * Real.cos θ, b * Real.sin θ)
def C1_equation (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def C2_equation (r x y : ℝ) : Prop := x^2 + y^2 = r^2

-- Main theorem 
theorem C1_C2_properties (a b r x y θ : ℝ) (ha_gt_hb : a > b) (hb_gt_zero : b > 0) (hr_gt_zero : r > 0) :
  (C1_equation a b (a * Real.cos θ) (b * Real.sin θ)) ∧
  (C2_equation r x y) ∧
  ((r = a ∨ r = b) → ∃ p1 p2 : ℝ × ℝ, C1_equation a b p1.1 p1.2 ∧ C2_equation r p1.1 p1.2 ∧ C1_equation a b p2.1 p2.2 ∧ C2_equation r p2.1 p2.2) ∧
  ((b < r ∧ r < a) → ∃ p1 p2 p3 p4 : ℝ × ℝ, C1_equation a b p1.1 p1.2 ∧ C2_equation r p1.1 p1.2 ∧ 
                                                C1_equation a b p2.1 p2.2 ∧ C2_equation r p2.1 p2.2 ∧ 
                                                C1_equation a b p3.1 p3.2 ∧ C2_equation r p3.1 p3.2 ∧ 
                                                C1_equation a b p4.1 p4.2 ∧ C2_equation r p4.1 p4.2) 
∧ ((0 < r ∧ r < b) ∨ (r > a) → ∀ (p : ℝ × ℝ), ¬ (C1_equation a b p.1 p.2 ∧ C2_equation r p.1 p.2)) ∧
  ((b < r) ∧ (r < a) → ∃ (θ : ℝ), 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ (2 * a * b * Real.sin (2 * θ) = 2 * a * b)) :=
by
  sorry

end C1_C2_properties_l39_39537


namespace find_n_l39_39322

theorem find_n (n : ℕ) (h1 : Nat.lcm n 14 = 56) (h2 : Nat.gcd n 14 = 10) : n = 40 :=
by
  sorry

end find_n_l39_39322


namespace smallest_n_for_divisibility_condition_l39_39358

theorem smallest_n_for_divisibility_condition :
  ∃ n : ℕ, (n > 0) ∧ (∀ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) →
    (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (xyz ∣ (x + y + z)^n)) ∧
    n = 13 :=
by
  use 13
  sorry

end smallest_n_for_divisibility_condition_l39_39358


namespace tan_105_eq_neg2_sub_sqrt3_l39_39506

theorem tan_105_eq_neg2_sub_sqrt3 :
  Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_eq_neg2_sub_sqrt3_l39_39506


namespace percentage_saved_l39_39672

theorem percentage_saved (rent milk groceries education petrol misc savings : ℝ) 
  (salary : ℝ) 
  (h_rent : rent = 5000) 
  (h_milk : milk = 1500) 
  (h_groceries : groceries = 4500) 
  (h_education : education = 2500) 
  (h_petrol : petrol = 2000) 
  (h_misc : misc = 700) 
  (h_savings : savings = 1800) 
  (h_salary : salary = rent + milk + groceries + education + petrol + misc + savings) : 
  (savings / salary) * 100 = 10 :=
by
  sorry

end percentage_saved_l39_39672


namespace sum_of_first_n_odd_numbers_l39_39295

theorem sum_of_first_n_odd_numbers (n : ℕ) (h : n = 15) : 
  ∑ k in finset.range n, (2 * (k + 1) - 1) = 225 := by
  sorry

end sum_of_first_n_odd_numbers_l39_39295


namespace proof_problem_l39_39386

open Set

variable (U : Set ℕ)
variable (P : Set ℕ)
variable (Q : Set ℕ)

noncomputable def problem_statement : Set ℕ :=
  compl (P ∪ Q) ∩ U

theorem proof_problem :
  U = {1, 2, 3, 4} →
  P = {1, 2} →
  Q = {2, 3} →
  compl (P ∪ Q) ∩ U = {4} :=
by
  intros hU hP hQ
  rw [hU, hP, hQ]
  sorry

end proof_problem_l39_39386


namespace transform_to_quadratic_l39_39810

theorem transform_to_quadratic :
  (∀ x : ℝ, (x + 1) ^ 2 + (x - 2) * (x + 2) = 1 ↔ 2 * x ^ 2 + 2 * x - 4 = 0) :=
sorry

end transform_to_quadratic_l39_39810


namespace probability_of_circle_in_square_l39_39760

open Real Set

theorem probability_of_circle_in_square :
  ∃ (p : ℝ), (∀ x y : ℝ, x ∈ Icc (-1 : ℝ) 1 → y ∈ Icc (-1 : ℝ) 1 → (x^2 + y^2 < 1/4) → True)
  → p = π / 16 :=
by
  use π / 16
  sorry

end probability_of_circle_in_square_l39_39760


namespace factorize_quadratic_l39_39053

theorem factorize_quadratic (x : ℝ) : x^2 - 2 * x = x * (x - 2) :=
sorry

end factorize_quadratic_l39_39053


namespace remainder_when_divided_l39_39472

noncomputable def y : ℝ := 19.999999999999716
def quotient : ℝ := 76.4
def remainder : ℝ := 8

theorem remainder_when_divided (x : ℝ) (hx : x = y * 76 + y * 0.4) : x % y = 8 :=
by
  -- Proof is omitted
  sorry

end remainder_when_divided_l39_39472


namespace complement_union_eq_l39_39061

-- Define the sets U, M, and N
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

-- Define the complement of a set within another set
def complement (S T : Set ℕ) : Set ℕ := { x | x ∈ S ∧ x ∉ T }

-- Define the union of M and N
def union_M_N : Set ℕ := {x | x ∈ M ∨ x ∈ N}

-- State the theorem
theorem complement_union_eq :
  complement U union_M_N = {4} :=
sorry

end complement_union_eq_l39_39061


namespace lcm_eq_792_l39_39200

-- Define the integers
def a : ℕ := 8
def b : ℕ := 9
def c : ℕ := 11

-- Define their prime factorizations (included for clarity, though not directly necessary)
def a_factorization : a = 2^3 := rfl
def b_factorization : b = 3^2 := rfl
def c_factorization : c = 11 := rfl

-- Define the LCM function
def lcm_abc := Nat.lcm (Nat.lcm a b) c

-- Prove that lcm of a, b, c is 792
theorem lcm_eq_792 : lcm_abc = 792 := 
by
  -- Include the necessary properties of LCM and prime factorizations if necessary
  sorry

end lcm_eq_792_l39_39200


namespace hypotenuse_length_l39_39775

open Real

-- Definitions corresponding to the conditions
def right_triangle_vertex_length (ADC_length : ℝ) (AEC_length : ℝ) (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2 ∧ ADC_length = sqrt 3 * sin x ∧ AEC_length = sin x

def trisect_hypotenuse (BD : ℝ) (DE : ℝ) (EC : ℝ) (c : ℝ) : Prop :=
  BD = c / 3 ∧ DE = c / 3 ∧ EC = c / 3

-- Main theorem definition
theorem hypotenuse_length (x hypotenuse ADC_length AEC_length : ℝ) :
  right_triangle_vertex_length ADC_length AEC_length x →
  trisect_hypotenuse (hypotenuse / 3) (hypotenuse / 3) (hypotenuse / 3) hypotenuse →
  hypotenuse = sqrt 3 * sin x :=
by
  intros h₁ h₂
  sorry

end hypotenuse_length_l39_39775


namespace jenny_ate_more_than_thrice_mike_l39_39422

theorem jenny_ate_more_than_thrice_mike :
  let mike_ate := 20
  let jenny_ate := 65
  jenny_ate - 3 * mike_ate = 5 :=
by
  let mike_ate := 20
  let jenny_ate := 65
  have : jenny_ate - 3 * mike_ate = 5 := by
    sorry
  exact this

end jenny_ate_more_than_thrice_mike_l39_39422


namespace sqrt_a_add_4b_eq_pm3_l39_39643

theorem sqrt_a_add_4b_eq_pm3
  (a b : ℝ)
  (A_sol : a * (-1) + 5 * (-1) = 15)
  (B_sol : 4 * 5 - b * 2 = -2) :
  (a + 4 * b)^(1/2) = 3 ∨ (a + 4 * b)^(1/2) = -3 := by
  sorry

end sqrt_a_add_4b_eq_pm3_l39_39643


namespace find_pqr_l39_39361

variable (p q r : ℚ)

theorem find_pqr (h1 : ∃ a : ℚ, ∀ x : ℚ, (p = a) ∧ (q = -2 * a * 3) ∧ (r = a * 3 * 3 + 7) ∧ (r = 10 + 7)) :
  p + q + r = 8 + 1/3 := by
  sorry

end find_pqr_l39_39361


namespace inequality_ab_l39_39166

theorem inequality_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  1 / (a^2 + 1) + 1 / (b^2 + 1) ≥ 1 := 
sorry

end inequality_ab_l39_39166


namespace Allen_change_l39_39846

theorem Allen_change (boxes_cost : ℕ) (total_boxes : ℕ) (tip_fraction : ℚ) (money_given : ℕ) :
  boxes_cost = 7 → total_boxes = 5 → tip_fraction = 1/7 → money_given = 100 →
  let total_cost := boxes_cost * total_boxes in
  let tip := total_cost * tip_fraction in
  let total_spent := total_cost + tip in
  let change := money_given - total_spent in
  change = 60 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  unfold total_cost tip total_spent change
  norm_num
  unfold tip_fraction
  norm_num
  sorry -- Proof can be completed

end Allen_change_l39_39846


namespace rectangle_base_length_l39_39843

theorem rectangle_base_length
  (h : ℝ) (b : ℝ)
  (common_height_nonzero : h ≠ 0)
  (triangle_base : ℝ := 24)
  (same_area : (1/2) * triangle_base * h = b * h) :
  b = 12 :=
by
  sorry

end rectangle_base_length_l39_39843


namespace tiffany_lives_next_level_l39_39281

theorem tiffany_lives_next_level (L1 L2 L3 : ℝ)
    (h1 : L1 = 43.0)
    (h2 : L2 = 14.0)
    (h3 : L3 = 84.0) :
    L3 - (L1 + L2) = 27 :=
by
  rw [h1, h2, h3]
  -- The proof is skipped with "sorry"
  sorry

end tiffany_lives_next_level_l39_39281


namespace blue_red_difference_l39_39807

variable (B : ℕ) -- Blue crayons
variable (R : ℕ := 14) -- Red crayons
variable (Y : ℕ := 32) -- Yellow crayons
variable (H : Y = 2 * B - 6) -- Relationship between yellow and blue crayons

theorem blue_red_difference (B : ℕ) (H : (32:ℕ) = 2 * B - 6) : (B - 14 = 5) :=
by
  -- Proof steps goes here
  sorry

end blue_red_difference_l39_39807


namespace number_of_solutions_l39_39654

theorem number_of_solutions (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 + 4 * n ∧ (∃ (x y : ℤ), x ^ 2 + 2016 * y ^ 2 = 2017 ^ n) :=
by
  sorry

end number_of_solutions_l39_39654


namespace cricket_team_players_l39_39953

theorem cricket_team_players (P N : ℕ) (h1 : 37 = 37) 
  (h2 : (57 - 37) = 20) 
  (h3 : ∀ N, (2 / 3 : ℚ) * N = 20 → N = 30) 
  (h4 : P = 37 + 30) : P = 67 := 
by
  -- Proof steps will go here
  sorry

end cricket_team_players_l39_39953


namespace equal_roots_quadratic_l39_39902

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

/--
If the quadratic equation 2x^2 - ax + 2 = 0 has two equal real roots,
then the value of a is ±4.
-/
theorem equal_roots_quadratic (a : ℝ) (h : quadratic_discriminant 2 (-a) 2 = 0) :
  a = 4 ∨ a = -4 :=
sorry

end equal_roots_quadratic_l39_39902


namespace candidate_total_score_l39_39173

theorem candidate_total_score (written_score : ℝ) (interview_score : ℝ) (written_weight : ℝ) (interview_weight : ℝ) :
    written_score = 90 → interview_score = 80 → written_weight = 0.70 → interview_weight = 0.30 →
    written_score * written_weight + interview_score * interview_weight = 87 :=
by
  intros
  sorry

end candidate_total_score_l39_39173


namespace rain_in_first_hour_l39_39570

theorem rain_in_first_hour :
  ∃ x : ℕ, (let rain_second_hour := 2 * x + 7 in x + rain_second_hour = 22) ∧ x = 5 :=
by
  sorry

end rain_in_first_hour_l39_39570


namespace max_non_managers_l39_39908

theorem max_non_managers (N : ℕ) (h : (9:ℝ) / (N:ℝ) > (7:ℝ) / (32:ℝ)) : N ≤ 41 :=
by
  -- Proof skipped
  sorry

end max_non_managers_l39_39908


namespace difference_of_squares_eval_l39_39691

-- Define the conditions
def a : ℕ := 81
def b : ℕ := 49

-- State the corresponding problem and its equivalence
theorem difference_of_squares_eval : (a^2 - b^2) = 4160 := by
  sorry -- Placeholder for the proof

end difference_of_squares_eval_l39_39691


namespace probability_of_winning_first_draw_better_chance_with_yellow_ball_l39_39638

-- The probability of winning on the first draw in the lottery promotion.
theorem probability_of_winning_first_draw :
  (1 / 4 : ℚ) = 0.25 :=
sorry

-- The optimal choice to add to the bag for the highest probability of receiving a fine gift.
theorem better_chance_with_yellow_ball :
  (3 / 5 : ℚ) > (2 / 5 : ℚ) :=
by norm_num

end probability_of_winning_first_draw_better_chance_with_yellow_ball_l39_39638


namespace betty_harvest_l39_39855

/-- 
Given the following conditions for Betty's vegetable harvest, 
- Boxes for parsnips hold 20 units, with 5/8 of the boxes full and 3/8 half-full, averaging 18 boxes per harvest.
- Boxes for carrots hold 25 units, with 7/12 of the boxes full and 5/12 half-full, averaging 12 boxes per harvest.
- Boxes for potatoes hold 30 units, with 3/5 of the boxes full and 2/5 half-full, averaging 15 boxes per harvest.
Prove Betty's average harvest yields:
- 290 parsnips
- 237.5 carrots
- 360 potatoes
--/

theorem betty_harvest :
  let full_boxes (capacity : ℕ) (ratio : ℚ) (total_boxes : ℚ) : ℚ :=
    (ratio * total_boxes) * capacity

  let half_boxes (capacity : ℕ) (ratio : ℚ) (total_boxes : ℚ) : ℚ :=
    (ratio * total_boxes) * (capacity / 2)

  full_boxes 20 (5/8) 18 + half_boxes 20 (3/8) 18 = 290 ∧
  full_boxes 25 (7/12) 12 + half_boxes 25 (5/12) 12 = 237.5 ∧
  full_boxes 30 (3/5) 15 + half_boxes 30 (2/5) 15 = 360 :=
by 
  sorry

end betty_harvest_l39_39855


namespace smallest_b_for_N_fourth_power_l39_39013

theorem smallest_b_for_N_fourth_power : 
  ∃ (b : ℤ), (∀ n : ℤ, 7 * b^2 + 7 * b + 7 = n^4) ∧ b = 18 :=
by
  sorry

end smallest_b_for_N_fourth_power_l39_39013


namespace sum_first_15_odd_integers_l39_39308

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l39_39308


namespace sufficient_but_not_necessary_l39_39067

theorem sufficient_but_not_necessary (a b : ℝ) (hp : a > 1 ∧ b > 1) (hq : a + b > 2 ∧ a * b > 1) : 
  (a > 1 ∧ b > 1 → a + b > 2 ∧ a * b > 1) ∧ ¬(a + b > 2 ∧ a * b > 1 → a > 1 ∧ b > 1) :=
by
  sorry

end sufficient_but_not_necessary_l39_39067


namespace seeds_in_bucket_A_l39_39279

theorem seeds_in_bucket_A (A B C : ℕ) (h_total : A + B + C = 100) (h_B : B = 30) (h_C : C = 30) : A = 40 :=
by
  sorry

end seeds_in_bucket_A_l39_39279


namespace maggie_kept_bouncy_balls_l39_39931

def packs_bought_yellow : ℝ := 8.0
def packs_given_away_green : ℝ := 4.0
def packs_bought_green : ℝ := 4.0
def balls_per_pack : ℝ := 10.0

theorem maggie_kept_bouncy_balls :
  packs_bought_yellow * balls_per_pack + (packs_bought_green - packs_given_away_green) * balls_per_pack = 80.0 :=
by sorry

end maggie_kept_bouncy_balls_l39_39931


namespace least_positive_integer_with_12_factors_l39_39994

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l39_39994


namespace number_of_integers_l39_39391

theorem number_of_integers (n : ℤ) : (200 < n ∧ n < 300 ∧ ∃ r : ℤ, n % 7 = r ∧ n % 9 = r) ↔ 
  n = 252 ∨ n = 253 ∨ n = 254 ∨ n = 255 ∨ n = 256 ∨ n = 257 ∨ n = 258 :=
by {
  sorry
}

end number_of_integers_l39_39391


namespace find_A_from_equation_and_conditions_l39_39226

theorem find_A_from_equation_and_conditions 
  (A B C D : ℕ)
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
  (h4 : B ≠ C) (h5 : B ≠ D)
  (h6 : C ≠ D)
  (h7 : 10 * A + B ≠ 0)
  (h8 : 10 * 10 * 10 * A + 10 * 10 * B + 8 * 10 + 2 - (900 + C * 10 + 9) = 490 + 3 * 10 + D) :
  A = 5 :=
by
  sorry

end find_A_from_equation_and_conditions_l39_39226


namespace probability_at_least_two_white_balls_l39_39829

theorem probability_at_least_two_white_balls :
  let total_ways := Nat.factorial 17 / (Nat.factorial 3 * Nat.factorial (17 - 3)) in
  let exactly_two_white := (Nat.factorial 8 / (Nat.factorial 2 * Nat.factorial (8 - 2))) *
                           (Nat.factorial 9 / (Nat.factorial 1 * Nat.factorial (9 - 1))) in
  let exactly_three_white := Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial (8 - 3)) in
  let favorable_ways := exactly_two_white + exactly_three_white in
  let probability := favorable_ways / total_ways in
  probability = 154 / 340 :=
by
  sorry

end probability_at_least_two_white_balls_l39_39829


namespace expected_value_coins_heads_l39_39491

noncomputable def expected_value_cents : ℝ :=
  let values := [1, 5, 10, 25, 50, 100]
  let probability_heads := 1 / 2
  probability_heads * (values.sum : ℝ)

theorem expected_value_coins_heads : expected_value_cents = 95.5 := by
  sorry

end expected_value_coins_heads_l39_39491


namespace find_x2_y2_l39_39547

theorem find_x2_y2 (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = (10344 / 169) := by
  sorry

end find_x2_y2_l39_39547


namespace solve_eq_proof_l39_39939

noncomputable def solve_equation : List ℚ := [-4, 1, 3 / 2, 2]

theorem solve_eq_proof :
  (∀ x : ℚ, 
    ((x^2 + 3 * x - 4)^2 + (2 * x^2 - 7 * x + 6)^2 = (3 * x^2 - 4 * x + 2)^2) ↔ 
    (x ∈ solve_equation)) :=
by
  sorry

end solve_eq_proof_l39_39939


namespace average_age_of_9_students_l39_39449

theorem average_age_of_9_students
  (avg_20_students : ℝ)
  (n_20_students : ℕ)
  (avg_10_students : ℝ)
  (n_10_students : ℕ)
  (age_20th_student : ℝ)
  (total_age_20_students : ℝ := avg_20_students * n_20_students)
  (total_age_10_students : ℝ := avg_10_students * n_10_students)
  (total_age_9_students : ℝ := total_age_20_students - total_age_10_students - age_20th_student)
  (n_9_students : ℕ)
  (expected_avg_9_students : ℝ := total_age_9_students / n_9_students)
  (H1 : avg_20_students = 20)
  (H2 : n_20_students = 20)
  (H3 : avg_10_students = 24)
  (H4 : n_10_students = 10)
  (H5 : age_20th_student = 61)
  (H6 : n_9_students = 9) :
  expected_avg_9_students = 11 :=
sorry

end average_age_of_9_students_l39_39449


namespace calculate_expression_l39_39471

theorem calculate_expression : (3.242 * 16) / 100 = 0.51872 := by
  sorry

end calculate_expression_l39_39471


namespace cistern_fill_time_l39_39832

variable (C : ℝ) -- Volume of the cistern
variable (X Y Z : ℝ) -- Rates at which pipes X, Y, and Z fill the cistern

-- Pipes X and Y together, pipes X and Z together, and pipes Y and Z together conditions
def condition1 := X + Y = C / 3
def condition2 := X + Z = C / 4
def condition3 := Y + Z = C / 5

theorem cistern_fill_time (h1 : condition1 C X Y) (h2 : condition2 C X Z) (h3 : condition3 C Y Z) :
  1 / (X + Y + Z) = 120 / 47 :=
by
  sorry

end cistern_fill_time_l39_39832


namespace sin_alpha_in_second_quadrant_l39_39728

theorem sin_alpha_in_second_quadrant 
  (α : ℝ) 
  (h1 : π / 2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.tan α = -1 / 2)  -- tan α = -1/2
  : Real.sin α = Real.sqrt 5 / 5 :=
sorry

end sin_alpha_in_second_quadrant_l39_39728


namespace BurjKhalifaHeight_l39_39593

def SearsTowerHeight : ℕ := 527
def AdditionalHeight : ℕ := 303

theorem BurjKhalifaHeight : (SearsTowerHeight + AdditionalHeight) = 830 :=
by
  sorry

end BurjKhalifaHeight_l39_39593


namespace area_of_shaded_part_l39_39187

-- Define the given condition: area of the square
def area_of_square : ℝ := 100

-- Define the proof goal: area of the shaded part
theorem area_of_shaded_part : area_of_square / 2 = 50 := by
  sorry

end area_of_shaded_part_l39_39187


namespace product_of_solutions_l39_39470

theorem product_of_solutions (a b c x : ℝ) (h1 : -x^2 - 4 * x + 10 = 0) :
  x * (-4 - x) = -10 :=
by
  sorry

end product_of_solutions_l39_39470


namespace total_distance_flash_runs_l39_39183

-- Define the problem with given conditions
theorem total_distance_flash_runs (v k d a : ℝ) (hk : k > 1) : 
  let t := d / (v * (k - 1))
  let distance_to_catch_ace := k * v * t
  let total_distance := distance_to_catch_ace + a
  total_distance = (k * d) / (k - 1) + a := 
by
  sorry

end total_distance_flash_runs_l39_39183


namespace percentage_X_correct_l39_39850

def initial_solution_Y : ℝ := 12.0
def percentage_X_in_Y : ℝ := 45.0 / 100.0
def percentage_water_in_Y : ℝ := 55.0 / 100.0
def evaporated_water : ℝ := 5.0
def added_solution_Y : ℝ := 7.0

def calculate_percentage_X_in_new_solution : ℝ :=
  let initial_X := 0.45 * 12.0
  let initial_water := 0.55 * 12.0
  let remaining_water := initial_water - evaporated_water

  let added_X := 0.45 * added_solution_Y
  let added_water := 0.55 * added_solution_Y

  let total_X := initial_X + added_X
  let total_water := remaining_water + added_water

  let total_weight := total_X + total_water
  (total_X / total_weight) * 100.0

theorem percentage_X_correct :
  abs (calculate_percentage_X_in_new_solution - 61.07) < 0.01 := 
  sorry

end percentage_X_correct_l39_39850


namespace circle_center_sum_is_one_l39_39059

def circle_center_sum (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 + 4 * x - 6 * y = 3) → ((h = -2) ∧ (k = 3))

theorem circle_center_sum_is_one :
  ∀ h k : ℝ, circle_center_sum h k → h + k = 1 :=
by
  intros h k hc
  sorry

end circle_center_sum_is_one_l39_39059


namespace bottles_purchased_l39_39941

/-- Given P bottles can be bought for R dollars, determine how many bottles can be bought for M euros
    if 1 euro is worth 1.2 dollars and there is a 10% discount when buying with euros. -/
theorem bottles_purchased (P R M : ℝ) (hR : R > 0) (hP : P > 0) :
  let euro_to_dollars := 1.2
  let discount := 0.9
  let dollars := euro_to_dollars * M * discount
  (P / R) * dollars = (1.32 * P * M) / R :=
by
  sorry

end bottles_purchased_l39_39941


namespace jackson_maximum_usd_l39_39914

-- Define the rates for chores in various currencies
def usd_per_hour : ℝ := 5
def gbp_per_hour : ℝ := 3
def jpy_per_hour : ℝ := 400
def eur_per_hour : ℝ := 4

-- Define the hours Jackson worked for each task
def usd_hours_vacuuming : ℝ := 2 * 2
def gbp_hours_washing_dishes : ℝ := 0.5
def jpy_hours_cleaning_bathroom : ℝ := 1.5
def eur_hours_sweeping_yard : ℝ := 1

-- Define the exchange rates over three days
def exchange_rates_day1 := (1.35, 0.009, 1.18)  -- (GBP to USD, JPY to USD, EUR to USD)
def exchange_rates_day2 := (1.38, 0.0085, 1.20)
def exchange_rates_day3 := (1.33, 0.0095, 1.21)

-- Define a function to convert currency to USD based on best exchange rates
noncomputable def max_usd (gbp_to_usd jpy_to_usd eur_to_usd : ℝ) : ℝ :=
  (usd_hours_vacuuming * usd_per_hour) +
  (gbp_hours_washing_dishes * gbp_per_hour * gbp_to_usd) +
  (jpy_hours_cleaning_bathroom * jpy_per_hour * jpy_to_usd) +
  (eur_hours_sweeping_yard * eur_per_hour * eur_to_usd)

-- Prove the maximum USD Jackson can have by choosing optimal rates is $32.61
theorem jackson_maximum_usd : max_usd 1.38 0.0095 1.21 = 32.61 :=
by
  sorry

end jackson_maximum_usd_l39_39914


namespace book_price_l39_39235

theorem book_price (P : ℝ) : 
  (3 * 12 * P - 500 = 220) → 
  P = 20 :=
by
  intro h
  sorry

end book_price_l39_39235


namespace least_positive_integer_with_12_factors_l39_39980

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l39_39980


namespace volume_of_tetrahedron_equiv_l39_39716

noncomputable def volume_tetrahedron (D1 D2 D3 : ℝ) 
  (h1 : D1 = 24) (h2 : D3 = 20) (h3 : D2 = 16) : ℝ :=
  30 * Real.sqrt 6

theorem volume_of_tetrahedron_equiv (D1 D2 D3 : ℝ) 
  (h1 : D1 = 24) (h2 : D3 = 20) (h3 : D2 = 16) :
  volume_tetrahedron D1 D2 D3 h1 h2 h3 = 30 * Real.sqrt 6 :=
  sorry

end volume_of_tetrahedron_equiv_l39_39716


namespace inverse_proposition_of_parallel_lines_l39_39607

theorem inverse_proposition_of_parallel_lines 
  (P : Prop) (Q : Prop) 
  (h : P ↔ Q) : 
  (Q ↔ P) :=
by 
  sorry

end inverse_proposition_of_parallel_lines_l39_39607


namespace equivalent_operation_l39_39660

theorem equivalent_operation (x : ℚ) : 
  (x * (2 / 3)) / (4 / 7) = x * (7 / 6) :=
by sorry

end equivalent_operation_l39_39660


namespace minimum_distance_l39_39397

noncomputable def point_on_curve (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

noncomputable def point_on_line (x : ℝ) : ℝ := x + 2

theorem minimum_distance 
  (a b c d : ℝ) 
  (hP : b = point_on_curve a) 
  (hQ : d = point_on_line c) 
  : (a - c)^2 + (b - d)^2 = 8 :=
by
  sorry

end minimum_distance_l39_39397


namespace find_budget_l39_39421

variable (B : ℝ)

-- Conditions provided
axiom cond1 : 0.30 * B = 300

theorem find_budget : B = 1000 :=
by
  -- Notes:
  -- The proof will go here.
  sorry

end find_budget_l39_39421


namespace segment_radius_with_inscribed_equilateral_triangle_l39_39565

theorem segment_radius_with_inscribed_equilateral_triangle (α h : ℝ) : 
  ∃ x : ℝ, x = (h / (Real.sin (α / 2))^2) * (Real.cos (α / 2) + Real.sqrt (1 + (1 / 3) * (Real.sin (α / 2))^2)) :=
sorry

end segment_radius_with_inscribed_equilateral_triangle_l39_39565


namespace least_positive_integer_with_12_factors_l39_39983

def num_factors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem least_positive_integer_with_12_factors : 
  ∃ n : ℕ, num_factors n = 12 ∧ n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l39_39983


namespace quadratic_inequality_solution_set_quadratic_inequality_solution_set2_l39_39735

-- Proof Problem 1 Statement
theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x : ℝ, b < x ∧ x < 1 → ax^2 + 3 * x + 2 > 0) : 
  a = -5 ∧ b = -2/5 := sorry

-- Proof Problem 2 Statement
theorem quadratic_inequality_solution_set2 (a : ℝ) (h_pos : a > 0) : 
  ((0 < a ∧ a < 3) → (∀ x : ℝ, x < -3 / a ∨ x > -1 → ax^2 + 3 * x + 2 > -ax - 1)) ∧
  (a = 3 → (∀ x : ℝ, x ≠ -1 → ax^2 + 3 * x + 2 > -ax - 1)) ∧
  (a > 3 → (∀ x : ℝ, x < -1 ∨ x > -3 / a → ax^2 + 3 * x + 2 > -ax - 1)) := sorry

end quadratic_inequality_solution_set_quadratic_inequality_solution_set2_l39_39735


namespace number_of_people_for_cheaper_second_caterer_l39_39934

theorem number_of_people_for_cheaper_second_caterer : 
  ∃ (x : ℕ), (150 + 20 * x > 250 + 15 * x + 50) ∧ 
  ∀ (y : ℕ), (y < x → ¬ (150 + 20 * y > 250 + 15 * y + 50)) :=
by
  sorry

end number_of_people_for_cheaper_second_caterer_l39_39934


namespace arithmetic_mean_of_three_digit_multiples_of_8_l39_39813

theorem arithmetic_mean_of_three_digit_multiples_of_8 :
  let a := 104
  let l := 1000
  let d := 8
  ∃ n: ℕ, l = a + (n - 1) * d ∧ 
           let S := n * (a + l) / 2 in
           S / n = 552 :=
by
  sorry

end arithmetic_mean_of_three_digit_multiples_of_8_l39_39813


namespace least_positive_integer_with_12_factors_l39_39987

theorem least_positive_integer_with_12_factors : ∃ n : ℕ, (nat.factors_count n = 12 ∧ ∀ m : ℕ, (nat.factors_count m = 12) → n ≤ m) ∧ n = 72 :=
begin
  sorry
end

end least_positive_integer_with_12_factors_l39_39987


namespace inequality_for_any_x_l39_39454

theorem inequality_for_any_x (a : ℝ) (h : ∀ x : ℝ, |3 * x + 2 * a| + |2 - 3 * x| - |a + 1| > 2) :
  a < -1/3 ∨ a > 5 := 
sorry

end inequality_for_any_x_l39_39454


namespace factor_tree_X_value_l39_39087

-- Define the constants
def F : ℕ := 5 * 3
def G : ℕ := 7 * 3

-- Define the intermediate values
def Y : ℕ := 5 * F
def Z : ℕ := 7 * G

-- Final value of X
def X : ℕ := Y * Z

-- Prove the value of X
theorem factor_tree_X_value : X = 11025 := by
  sorry

end factor_tree_X_value_l39_39087


namespace find_m_l39_39717

-- Define the pattern of splitting cubes into odd numbers
def split_cubes (m : ℕ) : List ℕ := 
  let rec odd_numbers (n : ℕ) : List ℕ :=
    if n = 0 then []
    else (2 * n - 1) :: odd_numbers (n - 1)
  odd_numbers m

-- Define the condition that 59 is part of the split numbers of m^3
def is_split_number (m : ℕ) (n : ℕ) : Prop :=
  n ∈ (split_cubes m)

-- Prove that if 59 is part of the split numbers of m^3, then m = 8
theorem find_m (m : ℕ) (h : is_split_number m 59) : m = 8 := 
sorry

end find_m_l39_39717


namespace ducks_in_marsh_l39_39960

theorem ducks_in_marsh 
  (num_geese : ℕ) 
  (total_birds : ℕ) 
  (num_ducks : ℕ)
  (h1 : num_geese = 58) 
  (h2 : total_birds = 95) 
  (h3 : total_birds = num_geese + num_ducks) : 
  num_ducks = 37 :=
by
  sorry

end ducks_in_marsh_l39_39960


namespace geometric_sum_4500_l39_39616

theorem geometric_sum_4500 (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300)
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 :=
sorry

end geometric_sum_4500_l39_39616


namespace range_absolute_difference_l39_39856

theorem range_absolute_difference : ∀ y, y = |x + 5| - |x - 3| → y ∈ set.Icc (-8) 8 :=
by
  sorry

end range_absolute_difference_l39_39856


namespace rosa_parks_food_drive_l39_39133

theorem rosa_parks_food_drive :
  ∀ (total_students students_collected_12_cans students_collected_none students_remaining total_cans cans_collected_first_group total_cans_first_group total_cans_last_group cans_per_student_last_group : ℕ),
    total_students = 30 →
    students_collected_12_cans = 15 →
    students_collected_none = 2 →
    students_remaining = total_students - students_collected_12_cans - students_collected_none →
    total_cans = 232 →
    cans_collected_first_group = 12 →
    total_cans_first_group = students_collected_12_cans * cans_collected_first_group →
    total_cans_last_group = total_cans - total_cans_first_group →
    cans_per_student_last_group = total_cans_last_group / students_remaining →
    cans_per_student_last_group = 4 :=
by
  intros total_students students_collected_12_cans students_collected_none students_remaining total_cans cans_collected_first_group total_cans_first_group total_cans_last_group cans_per_student_last_group
  sorry

end rosa_parks_food_drive_l39_39133


namespace middle_number_is_9_point_5_l39_39619

theorem middle_number_is_9_point_5 (x y z : ℝ) 
  (h1 : x + y = 15) (h2 : x + z = 18) (h3 : y + z = 22) : y = 9.5 := 
by {
  sorry
}

end middle_number_is_9_point_5_l39_39619


namespace sum_of_first_three_terms_is_zero_l39_39271

variable (a d : ℤ) 

-- Definitions from the conditions
def a₄ := a + 3 * d
def a₅ := a + 4 * d
def a₆ := a + 5 * d

-- Theorem statement
theorem sum_of_first_three_terms_is_zero 
  (h₁ : a₄ = 8) 
  (h₂ : a₅ = 12) 
  (h₃ : a₆ = 16) : 
  a + (a + d) + (a + 2 * d) = 0 := 
by 
  sorry

end sum_of_first_three_terms_is_zero_l39_39271


namespace decagon_diagonal_intersection_probability_l39_39352

def probability_intersect_within_decagon : ℚ :=
  let total_vertices := 10
  let total_pairs_points := Nat.choose total_vertices 2
  let total_diagonals := total_pairs_points - total_vertices
  let ways_to_pick_2_diagonals := Nat.choose total_diagonals 2
  let combinations_4_vertices := Nat.choose total_vertices 4
  (combinations_4_vertices : ℚ) / (ways_to_pick_2_diagonals : ℚ)

theorem decagon_diagonal_intersection_probability :
  probability_intersect_within_decagon = 42 / 119 :=
sorry

end decagon_diagonal_intersection_probability_l39_39352


namespace kim_cousins_l39_39766

theorem kim_cousins (pieces_per_cousin : ℕ) (total_pieces : ℕ) (h_pieces_per_cousin : pieces_per_cousin = 5) (h_total_pieces : total_pieces = 20) :
  total_pieces / pieces_per_cousin = 4 :=
by
  rw [h_pieces_per_cousin, h_total_pieces]
  norm_num

end kim_cousins_l39_39766


namespace necessary_and_sufficient_condition_l39_39803

variable (a : ℝ)

theorem necessary_and_sufficient_condition :
  (-16 ≤ a ∧ a ≤ 0) ↔ ∀ x : ℝ, ¬(x^2 + a * x - 4 * a < 0) :=
by
  sorry

end necessary_and_sufficient_condition_l39_39803


namespace second_bill_late_fee_l39_39496

def first_bill_amount : ℕ := 200
def first_bill_interest_rate : ℝ := 0.10
def first_bill_months : ℕ := 2
def second_bill_amount : ℕ := 130
def second_bill_months : ℕ := 6
def third_bill_first_month_fee : ℕ := 40
def third_bill_second_month_fee : ℕ := 80
def total_amount_owed : ℕ := 1234

theorem second_bill_late_fee (x : ℕ) 
(h : first_bill_amount * (first_bill_interest_rate * first_bill_months) + first_bill_amount + third_bill_first_month_fee + third_bill_second_month_fee + second_bill_amount + second_bill_months * x = total_amount_owed) : x = 124 :=
sorry

end second_bill_late_fee_l39_39496


namespace distance_from_missouri_l39_39603

-- Open a namespace for our problem context
namespace DrivingDistance

-- Define the conditions
def distance_by_plane := 2000 -- Distance between Arizona and New York by plane in miles
def increase_rate := 0.40 -- Increase in distance by driving

def total_driving_distance : ℝ :=
  distance_by_plane * (1 + increase_rate)

def midway_distance : ℝ :=
  total_driving_distance / 2

-- Theorem to prove the distance from Missouri to New York by car
theorem distance_from_missouri :
  midway_distance = 1400 :=
by
  sorry

end DrivingDistance

end distance_from_missouri_l39_39603


namespace suff_not_nec_cond_l39_39372

theorem suff_not_nec_cond (a : ℝ) : (a > 6 → a^2 > 36) ∧ (a^2 > 36 → (a > 6 ∨ a < -6)) := by
  sorry

end suff_not_nec_cond_l39_39372


namespace div_neg_cancel_neg_div_example_l39_39504

theorem div_neg_cancel (x y : Int) (h : y ≠ 0) : (-x) / (-y) = x / y := by
  sorry

theorem neg_div_example : (-64 : Int) / (-32) = 2 := by
  apply div_neg_cancel
  norm_num

end div_neg_cancel_neg_div_example_l39_39504


namespace least_positive_integer_with_12_factors_l39_39977

theorem least_positive_integer_with_12_factors :
  ∃ k : ℕ, (1 ≤ k) ∧ (nat.factors k).length = 12 ∧
    ∀ n : ℕ, (1 ≤ n) ∧ (nat.factors n).length = 12 → k ≤ n := 
begin
  sorry
end

end least_positive_integer_with_12_factors_l39_39977


namespace least_positive_integer_with_12_factors_is_96_l39_39968

def has_exactly_12_factors (n : ℕ) : Prop :=
  (12 : ℕ) = (List.range (n + 1)).filter (λ x, n % x = 0).length

theorem least_positive_integer_with_12_factors_is_96 :
  n ∈ (ℕ) ∧ has_exactly_12_factors n → n = 96 :=
by
  sorry

end least_positive_integer_with_12_factors_is_96_l39_39968


namespace union_A_B_complement_union_l39_39873

-- Define \( U \), \( A \), and \( B \)
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

-- Define complement in the universe \( U \)
def complement_U (s : Set ℝ) : Set ℝ := {x | x ∉ s}

-- Statements to prove
theorem union_A_B : A ∪ B = {x | 2 ≤ x ∧ x ≤ 7} :=
  sorry

theorem complement_union : complement_U A ∪ complement_U B = {x | x < 3 ∨ x ≥ 5} :=
  sorry

end union_A_B_complement_union_l39_39873


namespace how_many_unanswered_l39_39938

theorem how_many_unanswered (c w u : ℕ) (h1 : 25 + 5 * c - 2 * w = 95)
                            (h2 : 6 * c + u = 110) (h3 : c + w + u = 30) : u = 10 :=
by
  sorry

end how_many_unanswered_l39_39938


namespace three_pairs_exist_l39_39332

theorem three_pairs_exist :
  ∃! S P : ℕ, 5 * S + 7 * P = 90 :=
by
  sorry

end three_pairs_exist_l39_39332


namespace train_average_speed_l39_39866

theorem train_average_speed (speed : ℕ) (stop_time : ℕ) (running_time : ℕ) (total_time : ℕ)
  (h1 : speed = 60)
  (h2 : stop_time = 24)
  (h3 : running_time = total_time - stop_time)
  (h4 : running_time = 36)
  (h5 : total_time = 60) :
  (speed * running_time / total_time = 36) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end train_average_speed_l39_39866


namespace krakozyabrs_total_count_l39_39413

theorem krakozyabrs_total_count :
  (∃ (T : ℕ), 
  (∀ (H W n : ℕ),
    0.2 * H = n ∧ 0.25 * W = n ∧ H = 5 * n ∧ W = 4 * n ∧ T = H + W - n ∧ 25 < T ∧ T < 35) ∧ 
  T = 32) :=
by sorry

end krakozyabrs_total_count_l39_39413


namespace least_positive_integer_with_12_factors_is_72_l39_39974

open Nat

def hasExactly12Factors (n : ℕ) : Prop :=
  (divisors n).length = 12

theorem least_positive_integer_with_12_factors_is_72 :
  ∃ n : ℕ, hasExactly12Factors n ∧ ∀ m : ℕ, hasExactly12Factors m → n ≤ m → n = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_is_72_l39_39974


namespace min_tablets_to_extract_l39_39484

noncomputable def min_tablets_needed : ℕ :=
  let A := 10
  let B := 14
  let C := 18
  let D := 20
  let required_A := 3
  let required_B := 4
  let required_C := 3
  let required_D := 2
  let worst_case := B + C + D
  worst_case + required_A -- 14 + 18 + 20 + 3 = 55

theorem min_tablets_to_extract : min_tablets_needed = 55 :=
by {
  let A := 10
  let B := 14
  let C := 18
  let D := 20
  let required_A := 3
  let required_B := 4
  let required_C := 3
  let required_D := 2
  let worst_case := B + C + D
  have h : worst_case + required_A = 55 := by decide
  exact h
}

end min_tablets_to_extract_l39_39484


namespace monochromatic_triangle_probability_l39_39331

noncomputable def probability_of_monochromatic_triangle_in_hexagon : ℝ := 0.968324

theorem monochromatic_triangle_probability :
  ∃ (H : Hexagon), probability_of_monochromatic_triangle_in_hexagon = 0.968324 :=
sorry

end monochromatic_triangle_probability_l39_39331


namespace sum_first_15_odd_integers_l39_39307

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l39_39307


namespace find_m_perpendicular_l39_39540

-- Define the two vectors
def a (m : ℝ) : ℝ × ℝ := (m, -1)
def b : ℝ × ℝ := (1, 2)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Theorem stating the mathematically equivalent proof problem
theorem find_m_perpendicular (m : ℝ) (h : dot_product (a m) b = 0) : m = 2 :=
by sorry

end find_m_perpendicular_l39_39540


namespace probability_heart_spade_queen_l39_39280

theorem probability_heart_spade_queen (h_cards : ℕ) (s_cards : ℕ) (q_cards : ℕ) (total_cards : ℕ) 
    (h_not_q : ℕ) (remaining_cards_after_2 : ℕ) (remaining_spades : ℕ) 
    (queen_remaining_after_2 : ℕ) (remaining_cards_after_1 : ℕ) :
    h_cards = 13 ∧ s_cards = 13 ∧ q_cards = 4 ∧ total_cards = 52 ∧ h_not_q = 12 ∧ remaining_cards_after_2 = 50 ∧
    remaining_spades = 13 ∧ queen_remaining_after_2 = 3 ∧ remaining_cards_after_1 = 51 →
    (h_cards / total_cards) * (remaining_spades / remaining_cards_after_1) * (q_cards / remaining_cards_after_2) + 
    (q_cards / total_cards) * (remaining_spades / remaining_cards_after_1) * (queen_remaining_after_2 / remaining_cards_after_2) = 
    221 / 44200 := by 
  sorry

end probability_heart_spade_queen_l39_39280


namespace problem1_problem2_l39_39345

-- Problem 1: Calculation
theorem problem1 :
  (1:Real) - 1^2 + Real.sqrt 12 + Real.sqrt (4 / 3) = -1 + (8 * Real.sqrt 3) / 3 :=
by
  sorry
  
-- Problem 2: Solve the equation 2x^2 - x - 1 = 0
theorem problem2 (x : Real) :
  (2 * x^2 - x - 1 = 0) → (x = -1/2 ∨ x = 1) :=
by
  sorry

end problem1_problem2_l39_39345


namespace books_in_library_final_l39_39150

variable (initial_books : ℕ) (books_taken_out_tuesday : ℕ) 
          (books_returned_wednesday : ℕ) (books_taken_out_thursday : ℕ)

def books_left_in_library (initial_books books_taken_out_tuesday 
                          books_returned_wednesday books_taken_out_thursday : ℕ) : ℕ :=
  initial_books - books_taken_out_tuesday + books_returned_wednesday - books_taken_out_thursday

theorem books_in_library_final 
  (initial_books := 250) 
  (books_taken_out_tuesday := 120) 
  (books_returned_wednesday := 35) 
  (books_taken_out_thursday := 15) :
  books_left_in_library initial_books books_taken_out_tuesday 
                        books_returned_wednesday books_taken_out_thursday = 150 :=
by 
  sorry

end books_in_library_final_l39_39150


namespace arithmetic_sequence_a4_l39_39755

theorem arithmetic_sequence_a4 (a1 : ℤ) (S3 : ℤ) (h1 : a1 = 3) (h2 : S3 = 15) : 
  ∃ (a4 : ℤ), a4 = 9 :=
by
  sorry

end arithmetic_sequence_a4_l39_39755


namespace r_amount_l39_39320

-- Let p, q, and r be the amounts of money p, q, and r have, respectively
variables (p q r : ℝ)

-- Given conditions: p + q + r = 5000 and r = (2 / 3) * (p + q)
theorem r_amount (h1 : p + q + r = 5000) (h2 : r = (2 / 3) * (p + q)) :
  r = 2000 :=
sorry

end r_amount_l39_39320


namespace total_study_time_l39_39239

/-- Joey's SAT study schedule conditions. -/
variables
  (weekday_hours_per_night : ℕ := 2)
  (weekday_nights_per_week : ℕ := 5)
  (weekend_hours_per_day : ℕ := 3)
  (weekend_days_per_week : ℕ := 2)
  (weeks_until_exam : ℕ := 6)

/-- Total time Joey will spend studying for his SAT exam. -/
theorem total_study_time :
  (weekday_hours_per_night * weekday_nights_per_week + weekend_hours_per_day * weekend_days_per_week) * weeks_until_exam = 96 :=
by
  sorry

end total_study_time_l39_39239


namespace parabola_y_intersection_l39_39600

theorem parabola_y_intersection : intersects (x^2 - 4) (0, -4) :=
by
  sorry

end parabola_y_intersection_l39_39600


namespace units_digit_m_squared_plus_3_pow_m_l39_39770

def m := 2023^2 + 3^2023

theorem units_digit_m_squared_plus_3_pow_m : 
  (m^2 + 3^m) % 10 = 5 := sorry

end units_digit_m_squared_plus_3_pow_m_l39_39770


namespace find_n_l39_39324

def smallest_a (n : ℕ) : ℕ := 
  Inf { k : ℕ | n ∣ k! }

theorem find_n (n : ℕ) (h : 0 < n) : 
  (smallest_a n * 3 = 2 * n) ↔ n = 9 := 
by 
  sorry

end find_n_l39_39324


namespace part1_part2_l39_39720

theorem part1 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + 4*b^2 = 1/(a*b) + 3) :
  a*b ≤ 1 := sorry

theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + 4*b^2 = 1/(a*b) + 3) (hba : b > a) :
  1/a^3 - 1/b^3 ≥ 3 * (1/a - 1/b) := sorry

end part1_part2_l39_39720


namespace range_of_a_l39_39374

noncomputable def f (x a : ℝ) : ℝ := (x^2 + (a - 1) * x + 1) * Real.exp x

theorem range_of_a :
  (∀ x, f x a + Real.exp 2 ≥ 0) ↔ (-2 ≤ a ∧ a ≤ Real.exp 3 + 3) :=
sorry

end range_of_a_l39_39374


namespace contradiction_method_assumption_l39_39006

theorem contradiction_method_assumption (a b c : ℝ) :
  (¬(a > 0 ∨ b > 0 ∨ c > 0) → false) :=
sorry

end contradiction_method_assumption_l39_39006


namespace customers_left_l39_39670

theorem customers_left (initial_customers remaining_tables people_per_table customers_left : ℕ)
    (h_initial : initial_customers = 62)
    (h_tables : remaining_tables = 5)
    (h_people : people_per_table = 9)
    (h_left : customers_left = initial_customers - remaining_tables * people_per_table) : 
    customers_left = 17 := 
    by 
        -- Provide the proof here 
        sorry

end customers_left_l39_39670


namespace janice_work_days_l39_39237

variable (dailyEarnings : Nat)
variable (overtimeEarnings : Nat)
variable (numOvertimeShifts : Nat)
variable (totalEarnings : Nat)

theorem janice_work_days
    (h1 : dailyEarnings = 30)
    (h2 : overtimeEarnings = 15)
    (h3 : numOvertimeShifts = 3)
    (h4 : totalEarnings = 195)
    : let overtimeTotal := numOvertimeShifts * overtimeEarnings
      let regularEarnings := totalEarnings - overtimeTotal
      let workDays := regularEarnings / dailyEarnings
      workDays = 5 :=
by
  sorry

end janice_work_days_l39_39237


namespace f_periodic_odd_condition_l39_39453

theorem f_periodic_odd_condition (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 4) = f x) (h_one : f 1 = 5) : f 2015 = -5 :=
by
  sorry

end f_periodic_odd_condition_l39_39453


namespace stickers_decorate_l39_39779

theorem stickers_decorate (initial_stickers bought_stickers birthday_stickers given_stickers remaining_stickers stickers_used : ℕ)
    (h1 : initial_stickers = 20)
    (h2 : bought_stickers = 12)
    (h3 : birthday_stickers = 20)
    (h4 : given_stickers = 5)
    (h5 : remaining_stickers = 39) :
    (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers = stickers_used) →
    stickers_used = 8 
:= by {sorry}

end stickers_decorate_l39_39779


namespace least_positive_integer_with_12_factors_l39_39999

theorem least_positive_integer_with_12_factors : 
  ∃ (n : ℕ), (n > 0) ∧ (nat.factors_count n = 12) ∧ (∀ m, (m > 0) → (nat.factors_count m = 12) → n ≤ m) ∧ n = 108 :=
sorry

end least_positive_integer_with_12_factors_l39_39999


namespace f_10_half_l39_39062

noncomputable def f (x : ℝ) : ℝ := x^2 / (2 * x + 1)
noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => x
  | n + 1 => f (fn n x)

theorem f_10_half :
  fn 10 (1 / 2) = 1 / (3 ^ 1024 - 1) :=
sorry

end f_10_half_l39_39062


namespace solve_quadratic_eq_l39_39588

theorem solve_quadratic_eq (x : ℝ) : (x - 1) * (x + 2) = 0 ↔ x = 1 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l39_39588


namespace hyperbola_focal_length_l39_39268

-- Define the constants a^2 and b^2 based on the given hyperbola equation.
def a_squared : ℝ := 16
def b_squared : ℝ := 25

-- Define the constants a and b as the square roots of a^2 and b^2.
noncomputable def a : ℝ := Real.sqrt a_squared
noncomputable def b : ℝ := Real.sqrt b_squared

-- Define the constant c based on the relation c^2 = a^2 + b^2.
noncomputable def c : ℝ := Real.sqrt (a_squared + b_squared)

-- The focal length of the hyperbola is 2c.
noncomputable def focal_length : ℝ := 2 * c

-- The theorem that captures the statement of the problem.
theorem hyperbola_focal_length : focal_length = 2 * Real.sqrt 41 := by
  -- Proof omitted.
  sorry

end hyperbola_focal_length_l39_39268


namespace smaller_solution_of_quadratic_l39_39870

theorem smaller_solution_of_quadratic :
  let a := 1
  let b := -15
  let c := -56
  (∀ (x : ℝ), x^2 - 15 * x - 56 = 0 → x = (15 + Real.sqrt 449) / 2 ∨ x = (15 - Real.sqrt 449) / 2) →
  ∃ x : ℝ, x = (15 - Real.sqrt 449) / 2 ∧ (∀ y : ℝ, y = (15 + Real.sqrt 449) / 2 → x < y) :=
by
  sorry

end smaller_solution_of_quadratic_l39_39870


namespace unique_solution_implies_relation_l39_39514

open Nat

noncomputable def unique_solution (a b : ℤ) :=
  ∃! (x y z : ℤ), x + y = a - 1 ∧ x * (y + 1) - z^2 = b

theorem unique_solution_implies_relation (a b : ℤ) :
  unique_solution a b → b = (a * a) / 4 := sorry

end unique_solution_implies_relation_l39_39514


namespace angle_relationship_l39_39580

variables {AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1}
variables {angleA angleA1 angleB angleB1 angleC angleC1 angleD angleD1 : ℝ}

-- Define the conditions
def conditions (AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 : ℝ)
  (angleA angleA1 : ℝ) : Prop :=
  AB = A_1B_1 ∧ BC = B_1C_1 ∧ CD = C_1D_1 ∧ DA = D_1A_1 ∧ angleA > angleA1

theorem angle_relationship (AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 : ℝ)
  (angleA angleA1 angleB angleB1 angleC angleC1 angleD angleD1 : ℝ)
  (h : conditions AB A_1B_1 BC B_1C_1 CD C_1D_1 DA D_1A_1 angleA angleA1) :
  angleB < angleB1 ∧ angleC > angleC1 ∧ angleD < angleD1 :=
by {
  sorry
}

end angle_relationship_l39_39580


namespace ordered_pair_count_l39_39212

noncomputable def count_pairs (n : ℕ) (c : ℕ) : ℕ := 
  (if n < c then 0 else n - c + 1)

theorem ordered_pair_count :
  (count_pairs 39 5 = 35) :=
sorry

end ordered_pair_count_l39_39212


namespace smallest_five_digit_number_divisible_by_first_five_primes_l39_39699

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l39_39699


namespace consumption_increase_percentage_l39_39951

theorem consumption_increase_percentage
  (T C : ℝ)
  (H1 : 0.90 * (1 + X/100) = 0.9999999999999858) :
  X = 11.11111111110953 :=
by
  sorry

end consumption_increase_percentage_l39_39951


namespace quadratic_ineq_solution_l39_39277

theorem quadratic_ineq_solution (a b : ℝ) 
  (h_solution_set : ∀ x, (ax^2 + bx - 1 > 0) ↔ (1 / 3 < x ∧ x < 1))
  (h_roots : (a / 3 + b = -1 / a) ∧ (a / 3 = -1 / a)) 
  (h_a_neg : a < 0) : a + b = 1 := 
sorry 

end quadratic_ineq_solution_l39_39277


namespace problem1_problem2_l39_39657

open Nat

def binomial (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)
def permutation (n k : ℕ) := n.factorial / (n - k).factorial

theorem problem1 : binomial 10 4 - binomial 7 3 * permutation 3 3 = 0 := sorry

theorem problem2 (x : ℕ) (h : 3 * permutation 8 x = 4 * permutation 9 (x - 1)) : x = 6 := sorry

end problem1_problem2_l39_39657


namespace complex_multiplication_result_l39_39342

-- Define the complex numbers used in the problem
def a : ℂ := 4 - 3 * Complex.I
def b : ℂ := 4 + 3 * Complex.I

-- State the theorem we want to prove
theorem complex_multiplication_result : a * b = 25 := 
by
  -- Proof is omitted
  sorry

end complex_multiplication_result_l39_39342


namespace smallest_m_l39_39289

theorem smallest_m (m : ℕ) (p q : ℤ) (h_eq : 10*(p:ℤ)^2 - m*(p:ℤ) + 360 = 0) (h_cond : q = 2 * p) :
  p * q = 36 → 3 * p + 3 * q = m → m = 90 :=
by sorry

end smallest_m_l39_39289


namespace burrito_calories_l39_39425

theorem burrito_calories :
  ∀ (C : ℕ), 
  (10 * C = 6 * (250 - 50)) →
  C = 120 :=
by
  intros C h
  sorry

end burrito_calories_l39_39425


namespace geometric_series_sum_l39_39313

theorem geometric_series_sum :
  ∀ (a r : ℚ) (n : ℕ), 
  a = 1 / 5 → 
  r = -1 / 5 → 
  n = 6 →
  (a - a * r^n) / (1 - r) = 1562 / 9375 :=
by 
  intro a r n ha hr hn
  rw [ha, hr, hn]
  sorry

end geometric_series_sum_l39_39313


namespace mark_has_24_dollars_l39_39587

theorem mark_has_24_dollars
  (small_bag_cost : ℕ := 4)
  (small_bag_balloons : ℕ := 50)
  (medium_bag_cost : ℕ := 6)
  (medium_bag_balloons : ℕ := 75)
  (large_bag_cost : ℕ := 12)
  (large_bag_balloons : ℕ := 200)
  (total_balloons : ℕ := 400) :
  total_balloons / large_bag_balloons = 2 ∧ 2 * large_bag_cost = 24 := by
  sorry

end mark_has_24_dollars_l39_39587


namespace line_in_slope_intercept_form_l39_39663

def vec1 : ℝ × ℝ := (3, -7)
def point : ℝ × ℝ := (-2, 4)
def line_eq (x y : ℝ) : Prop := vec1.1 * (x - point.1) + vec1.2 * (y - point.2) = 0

theorem line_in_slope_intercept_form (x y : ℝ) : line_eq x y → y = (3 / 7) * x - (34 / 7) :=
by
  sorry

end line_in_slope_intercept_form_l39_39663


namespace eq_C1_curves_symmetric_l39_39434

-- Definitions
def curve_C (x : ℝ) : ℝ := x^3 - x

def curve_C1 (x t s : ℝ) : ℝ := (x - t)^3 - (x - t) + s

variables (x t s : ℝ)

-- Claim 1: Equation of curve C1
theorem eq_C1 : curve_C1 x t s = (x - t)^3 - (x - t) + s := 
  by sorry

-- Symmetry definitions
def point_A (t s : ℝ) : ℝ × ℝ := (t / 2, s / 2)

def symmetric_point (x1 y1 t s : ℝ) : ℝ × ℝ := (t - x1, s - y1)

-- Claim 2: Curves are symmetric around point A(t / 2, s / 2)
theorem curves_symmetric (x1 y1 : ℝ) (h_C : curve_C x1 = y1) :
  let pA := point_A t s in
  let p2 := symmetric_point x1 y1 t s in
  curve_C1 p2.1 t s = p2.2 :=
  by sorry

end eq_C1_curves_symmetric_l39_39434


namespace complex_root_sixth_power_sum_equals_38908_l39_39430

noncomputable def omega : ℂ :=
  -- By definition, omega should satisfy the below properties.
  -- The exact value of omega is not being defined, we will use algebraic properties in the proof.
  sorry

theorem complex_root_sixth_power_sum_equals_38908 : 
  ∀ (ω : ℂ), ω^3 = 1 ∧ ¬(ω.re = 1) → (2 - ω + 2 * ω^2)^6 + (2 + ω - 2 * ω^2)^6 = 38908 :=
by
  -- Proof will utilize given conditions:
  -- 1. ω^3 = 1
  -- 2. ω is not real (or ω.re is not 1)
  sorry

end complex_root_sixth_power_sum_equals_38908_l39_39430


namespace triplet_solution_l39_39055

theorem triplet_solution (a b c : ℕ) (h1 : a^2 + b^2 + c^2 = 2005) (h2 : a ≤ b) (h3 : b ≤ c) :
  (a = 24 ∧ b = 30 ∧ c = 23) ∨ 
  (a = 12 ∧ b = 30 ∧ c = 31) ∨
  (a = 18 ∧ b = 40 ∧ c = 9) ∨
  (a = 15 ∧ b = 22 ∧ c = 36) ∨
  (a = 12 ∧ b = 30 ∧ c = 31) :=
sorry

end triplet_solution_l39_39055


namespace lines_are_skew_iff_l39_39507

def line1 (s : ℝ) (b : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3 * s, 3 + 4 * s, b + 5 * s)

def line2 (v : ℝ) : ℝ × ℝ × ℝ :=
  (5 + 6 * v, 2 + 3 * v, 1 + 2 * v)

def lines_intersect (s v b : ℝ) : Prop :=
  line1 s b = line2 v

theorem lines_are_skew_iff (b : ℝ) : ¬ (∃ s v, lines_intersect s v b) ↔ b ≠ 9 :=
by
  sorry

end lines_are_skew_iff_l39_39507


namespace solve_system_of_equations_l39_39130

theorem solve_system_of_equations :
  ∃ x : ℕ → ℝ,
  (∀ i : ℕ, i < 100 → x i > 0) ∧
  (x 0 + 1 / x 1 = 4) ∧
  (x 1 + 1 / x 2 = 1) ∧
  (x 2 + 1 / x 0 = 4) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i < 99 → x (2 * i + 1) + 1 / x (2 * i + 2) = 1) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i < 99 → x (2 * i + 2) + 1 / x (2 * i + 3) = 4) ∧
  (x 99 + 1 / x 0 = 1) ∧
  (∀ i : ℕ, i < 50 → x (2 * i) = 2) ∧
  (∀ i : ℕ, i < 50 → x (2 * i + 1) = 1 / 2) :=
sorry

end solve_system_of_equations_l39_39130


namespace smallest_five_digit_number_divisible_by_five_primes_l39_39697

theorem smallest_five_digit_number_divisible_by_five_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  lcm = 2310 → (∃ n : ℕ, n = 5 ∧ 10000 ≤ lcm * n ∧ lcm * n = 11550) :=
by
  intros p1 p2 p3 p4 p5 
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  intro hlcm
  use (5 : ℕ)
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end smallest_five_digit_number_divisible_by_five_primes_l39_39697


namespace find_varphi_l39_39801

theorem find_varphi 
  (f g : ℝ → ℝ) 
  (x1 x2 varphi : ℝ) 
  (h_f : ∀ x, f x = 2 * Real.cos (2 * x)) 
  (h_g : ∀ x, g x = 2 * Real.cos (2 * x - 2 * varphi)) 
  (h_varphi_range : 0 < varphi ∧ varphi < π / 2) 
  (h_diff_cos : |f x1 - g x2| = 4) 
  (h_min_dist : |x1 - x2| = π / 6) 
: varphi = π / 3 := 
sorry

end find_varphi_l39_39801


namespace initial_money_given_l39_39176

def bracelet_cost : ℕ := 15
def necklace_cost : ℕ := 10
def mug_cost : ℕ := 20
def num_bracelets : ℕ := 3
def num_necklaces : ℕ := 2
def num_mugs : ℕ := 1
def change_received : ℕ := 15

theorem initial_money_given : num_bracelets * bracelet_cost + num_necklaces * necklace_cost + num_mugs * mug_cost + change_received = 100 := 
sorry

end initial_money_given_l39_39176


namespace find_possible_values_of_a_l39_39140

theorem find_possible_values_of_a (a b c : ℝ) (h1 : a * b + a + b = c) (h2 : b * c + b + c = a) (h3 : c * a + c + a = b) :
  a = 0 ∨ a = -1 ∨ a = -2 :=
by
  sorry

end find_possible_values_of_a_l39_39140


namespace intersection_ST_l39_39436

def S : Set ℝ := { x : ℝ | x < -5 } ∪ { x : ℝ | x > 5 }
def T : Set ℝ := { x : ℝ | -7 < x ∧ x < 3 }

theorem intersection_ST : S ∩ T = { x : ℝ | -7 < x ∧ x < -5 } := 
by 
  sorry

end intersection_ST_l39_39436


namespace avg_annual_reduction_l39_39671

theorem avg_annual_reduction (x : ℝ) (hx : (1 - x)^2 = 0.64) : x = 0.2 :=
by
  sorry

end avg_annual_reduction_l39_39671


namespace tan_value_of_point_on_exp_graph_l39_39555

theorem tan_value_of_point_on_exp_graph (a : ℝ) (h1 : (a, 9) ∈ {p : ℝ × ℝ | ∃ x, p = (x, 3^x)}) : 
  Real.tan (a * Real.pi / 6) = Real.sqrt 3 := by
  sorry

end tan_value_of_point_on_exp_graph_l39_39555


namespace perfect_squares_ending_in_5_or_6_lt_2000_l39_39544

theorem perfect_squares_ending_in_5_or_6_lt_2000 :
  ∃ (n : ℕ), n = 9 ∧ ∀ k, 1 ≤ k ∧ k ≤ 44 → 
  (∃ m, m * m < 2000 ∧ (m % 10 = 5 ∨ m % 10 = 6)) :=
by
  sorry

end perfect_squares_ending_in_5_or_6_lt_2000_l39_39544


namespace holiday_price_correct_l39_39127

-- Define the problem parameters
def original_price : ℝ := 250
def first_discount_rate : ℝ := 0.40
def second_discount_rate : ℝ := 0.10

-- Define the calculation for the first discount
def price_after_first_discount (original: ℝ) (rate: ℝ) : ℝ :=
  original * (1 - rate)

-- Define the calculation for the second discount
def price_after_second_discount (intermediate: ℝ) (rate: ℝ) : ℝ :=
  intermediate * (1 - rate)

-- The final Lean statement to prove
theorem holiday_price_correct : 
  price_after_second_discount (price_after_first_discount original_price first_discount_rate) second_discount_rate = 135 :=
by
  sorry

end holiday_price_correct_l39_39127


namespace evaluate_expression_l39_39368

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := by 
  sorry

end evaluate_expression_l39_39368


namespace flowmaster_pump_output_l39_39794

theorem flowmaster_pump_output (hourly_rate : ℕ) (time_minutes : ℕ) (output_gallons : ℕ) 
  (h1 : hourly_rate = 600) 
  (h2 : time_minutes = 30) 
  (h3 : output_gallons = (hourly_rate * time_minutes) / 60) : 
  output_gallons = 300 :=
by sorry

end flowmaster_pump_output_l39_39794


namespace interest_second_month_l39_39917

theorem interest_second_month {P r n : ℝ} (hP : P = 200) (hr : r = 0.10) (hn : n = 12) :
  (P * (1 + r / n) ^ (n * (1/12)) - P) * r / n = 1.68 :=
by
  sorry

end interest_second_month_l39_39917


namespace total_amount_spent_l39_39114

noncomputable def value_of_nickel : ℕ := 5
noncomputable def value_of_dime : ℕ := 10
noncomputable def initial_amount : ℕ := 250

def amount_spent_by_Pete (nickels_spent : ℕ) : ℕ :=
  nickels_spent * value_of_nickel

def amount_remaining_with_Raymond (dimes_left : ℕ) : ℕ :=
  dimes_left * value_of_dime

theorem total_amount_spent (nickels_spent : ℕ) (dimes_left : ℕ) :
  (amount_spent_by_Pete nickels_spent + 
   (initial_amount - amount_remaining_with_Raymond dimes_left)) = 200 :=
by
  sorry

end total_amount_spent_l39_39114


namespace rain_first_hour_l39_39574

theorem rain_first_hour (x : ℝ) 
  (h1 : 22 = x + (2 * x + 7)) : x = 5 :=
by
  sorry

end rain_first_hour_l39_39574


namespace Cameron_list_count_l39_39035

theorem Cameron_list_count : 
  let lower_bound := 900
  let upper_bound := 27000
  let step := 30
  let n_min := lower_bound / step
  let n_max := upper_bound / step
  n_max - n_min + 1 = 871 :=
by
  sorry

end Cameron_list_count_l39_39035


namespace find_all_triplets_l39_39056

theorem find_all_triplets (a b c : ℕ)
  (h₀_a : a > 0)
  (h₀_b : b > 0)
  (h₀_c : c > 0) :
  6^a = 1 + 2^b + 3^c ↔ 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 3) ∨ 
  (a = 2 ∧ b = 5 ∧ c = 1) :=
by
  sorry

end find_all_triplets_l39_39056


namespace intersection_y_axis_parabola_l39_39601

theorem intersection_y_axis_parabola : (0, -4) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, x^2 - 4) ∧ x = 0 } :=
by
  sorry

end intersection_y_axis_parabola_l39_39601


namespace negation_of_prop_p_is_correct_l39_39075

-- Define the original proposition p
def prop_p (x y : ℝ) : Prop := x > 0 ∧ y > 0 → x * y > 0

-- Define the negation of the proposition p
def neg_prop_p (x y : ℝ) : Prop := x ≤ 0 ∨ y ≤ 0 → x * y ≤ 0

-- The theorem we need to prove
theorem negation_of_prop_p_is_correct : ∀ x y : ℝ, neg_prop_p x y := 
sorry

end negation_of_prop_p_is_correct_l39_39075


namespace trains_crossing_time_l39_39284

-- Definitions based on conditions
def train_length : ℕ := 120
def time_train1_cross_pole : ℕ := 10
def time_train2_cross_pole : ℕ := 15

-- Question reformulated as a proof goal
theorem trains_crossing_time :
  let v1 := train_length / time_train1_cross_pole  -- Speed of train 1
  let v2 := train_length / time_train2_cross_pole  -- Speed of train 2
  let relative_speed := v1 + v2                    -- Relative speed in opposite directions
  let total_distance := train_length + train_length -- Sum of both trains' lengths
  let time_to_cross := total_distance / relative_speed -- Time to cross each other
  time_to_cross = 12 := 
by
  -- The proof here is stated, but not needed in this task
  -- All necessary computation steps
  sorry

end trains_crossing_time_l39_39284


namespace mike_age_l39_39781

theorem mike_age : ∀ (m M : ℕ), m = M - 18 ∧ m + M = 54 → m = 18 :=
by
  intros m M
  intro h
  sorry

end mike_age_l39_39781


namespace workers_to_build_cars_l39_39396

theorem workers_to_build_cars (W : ℕ) (hW : W > 0) : 
  (∃ D : ℝ, D = 63 / W) :=
by
  sorry

end workers_to_build_cars_l39_39396


namespace least_pos_int_with_12_pos_factors_is_72_l39_39970

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l39_39970


namespace evaporated_water_l39_39445

theorem evaporated_water 
  (E : ℝ)
  (h₁ : 0 < 10) -- initial mass is positive
  (h₂ : 10 * 0.3 + 10 * 0.7 = 3 + 7) -- Solution Y composition check
  (h₃ : (3 + 0.3 * E) / (10 - E + 0.7 * E) = 0.36) -- New solution composition
  : E = 0.9091 := 
sorry

end evaporated_water_l39_39445


namespace visitors_on_monday_l39_39762

theorem visitors_on_monday (M : ℕ) (h : M + 2 * M + 100 = 250) : M = 50 :=
by
  sorry

end visitors_on_monday_l39_39762


namespace rocco_piles_of_quarters_proof_l39_39123

-- Define the value of a pile of different types of coins
def pile_value (coin_value : ℕ) (num_coins_in_pile : ℕ) : ℕ :=
  coin_value * num_coins_in_pile

-- Define the number of piles for different coins
def num_piles_of_dimes : ℕ := 6
def num_piles_of_nickels : ℕ := 9
def num_piles_of_pennies : ℕ := 5
def num_coins_in_pile : ℕ := 10

-- Define the total value of each type of coin
def value_of_a_dime : ℕ := 10  -- in cents
def value_of_a_nickel : ℕ := 5  -- in cents
def value_of_a_penny : ℕ := 1  -- in cents
def value_of_a_quarter : ℕ := 25  -- in cents

-- Define the total money Rocco has in cents
def total_money : ℕ := 2100  -- since $21 = 2100 cents

-- Calculate the value of all piles of each type of coin
def total_dimes_value : ℕ := num_piles_of_dimes * (pile_value value_of_a_dime num_coins_in_pile)
def total_nickels_value : ℕ := num_piles_of_nickels * (pile_value value_of_a_nickel num_coins_in_pile)
def total_pennies_value : ℕ := num_piles_of_pennies * (pile_value value_of_a_penny num_coins_in_pile)

-- Calculate the value of the quarters
def value_of_quarters : ℕ := total_money - (total_dimes_value + total_nickels_value + total_pennies_value)
def num_piles_of_quarters : ℕ := value_of_quarters / 250 -- since each pile of quarters is worth 250 cents

-- Theorem to prove
theorem rocco_piles_of_quarters_proof : num_piles_of_quarters = 4 := by
  sorry

end rocco_piles_of_quarters_proof_l39_39123


namespace perimeter_of_ABFCDE_l39_39179

-- Define the problem parameters
def square_perimeter : ℤ := 60
def side_length (p : ℤ) : ℤ := p / 4
def equilateral_triangle_side (l : ℤ) : ℤ := l
def new_shape_sides : ℕ := 6
def new_perimeter (s : ℤ) : ℤ := new_shape_sides * s

-- Define the theorem to be proved
theorem perimeter_of_ABFCDE (p : ℤ) (s : ℕ) (len : ℤ) : len = side_length p → len = equilateral_triangle_side len →
  new_perimeter len = 90 :=
by
  intros h1 h2
  sorry

end perimeter_of_ABFCDE_l39_39179


namespace total_tips_fraction_l39_39478

variables {A : ℚ} -- average monthly tips in other months

theorem total_tips_fraction (A : ℚ) :
  let august_tips := 8 * A in
  let other_months_tips := 6 * A in
  let total_tips := other_months_tips + august_tips + A in
  august_tips / total_tips = 8 / 15 :=
by
  sorry

end total_tips_fraction_l39_39478


namespace counting_integers_between_multiples_l39_39031

theorem counting_integers_between_multiples :
  let smallest_perfect_square_multiple := 900 in
  let smallest_perfect_cube_multiple := 27000 in
  let num_integers := (smallest_perfect_cube_multiple / 30) - (smallest_perfect_square_multiple / 30) + 1 in
  smallest_perfect_square_multiple = 30 * 30 ∧ 
  smallest_perfect_cube_multiple = 900 * 30 ∧ 
  num_integers = 871 :=
by
  sorry

end counting_integers_between_multiples_l39_39031


namespace gcd_seven_eight_fact_l39_39517

-- Definitions based on the problem conditions
def seven_fact : ℕ := 1 * 2 * 3 * 4 * 5 * 6 * 7
def eight_fact : ℕ := 8 * seven_fact

-- Statement of the theorem
theorem gcd_seven_eight_fact : Nat.gcd seven_fact eight_fact = seven_fact := by
  sorry

end gcd_seven_eight_fact_l39_39517


namespace max_xyz_l39_39104

theorem max_xyz (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : 5 * x + 8 * y + 3 * z = 90) : xyz ≤ 225 :=
by
  sorry

end max_xyz_l39_39104


namespace no_analytic_roots_l39_39869

theorem no_analytic_roots : ¬∃ x : ℝ, (x - 2) * (x + 5)^3 * (5 - x) = 8 := 
sorry

end no_analytic_roots_l39_39869


namespace probability_of_at_least_one_boy_and_one_girl_is_correct_l39_39336

noncomputable def probability_at_least_one_boy_and_one_girl : ℚ :=
  (1 - ((1/2)^4 + (1/2)^4))

theorem probability_of_at_least_one_boy_and_one_girl_is_correct : 
  probability_at_least_one_boy_and_one_girl = 7/8 :=
by
  sorry

end probability_of_at_least_one_boy_and_one_girl_is_correct_l39_39336


namespace initial_machines_l39_39483

theorem initial_machines (r : ℝ) (x : ℕ) (h1 : x * 42 * r = 7 * 36 * r) : x = 6 :=
by
  sorry

end initial_machines_l39_39483


namespace solve_for_x_l39_39550

variables (x y z : ℝ)

def condition : Prop :=
  1 / (x + y) + 1 / (x - y) = z / (x - y)

theorem solve_for_x (h : condition x y z) : x = z / 2 :=
by
  sorry

end solve_for_x_l39_39550


namespace no_natural_number_n_exists_l39_39864

theorem no_natural_number_n_exists :
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), 3 * n + 1 = a * b := by
  sorry

end no_natural_number_n_exists_l39_39864


namespace cameron_list_count_l39_39025

theorem cameron_list_count :
  let numbers := {n : ℕ | 30 ≤ n ∧ n ≤ 900}
  in set.card numbers = 871 :=
sorry -- proof is omitted

end cameron_list_count_l39_39025


namespace range_of_a_l39_39528

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem range_of_a (a : ℝ) :
  (∀ (p q : ℝ), 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → (f a p - f a q) / (p - q) > 1)
  ↔ 3 ≤ a :=
by
  sorry

end range_of_a_l39_39528


namespace marble_cut_percentage_l39_39018

theorem marble_cut_percentage
  (initial_weight : ℝ)
  (final_weight : ℝ)
  (x : ℝ)
  (first_week_cut : ℝ)
  (second_week_cut : ℝ)
  (third_week_cut : ℝ) :
  initial_weight = 190 →
  final_weight = 109.0125 →
  first_week_cut = (1 - x / 100) →
  second_week_cut = 0.85 →
  third_week_cut = 0.9 →
  (initial_weight * first_week_cut * second_week_cut * third_week_cut = final_weight) →
  x = 24.95 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end marble_cut_percentage_l39_39018


namespace curve_passes_through_fixed_point_l39_39122

theorem curve_passes_through_fixed_point (m n : ℝ) :
  (2:ℝ)^2 + (-2:ℝ)^2 - 2 * m * (2:ℝ) - 2 * n * (-2:ℝ) + 4 * (m - n - 2) = 0 :=
by sorry

end curve_passes_through_fixed_point_l39_39122


namespace greatest_number_of_bouquets_l39_39502

def cherry_lollipops := 4
def orange_lollipops := 6
def raspberry_lollipops := 8
def lemon_lollipops := 10
def candy_canes := 12
def chocolate_coins := 14

theorem greatest_number_of_bouquets : 
  Nat.gcd cherry_lollipops (Nat.gcd orange_lollipops (Nat.gcd raspberry_lollipops (Nat.gcd lemon_lollipops (Nat.gcd candy_canes chocolate_coins)))) = 2 := 
by 
  sorry

end greatest_number_of_bouquets_l39_39502


namespace least_positive_integer_with_12_factors_l39_39993

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, k > 0 ∧ (∀ n : ℕ, n > 0 → (factors_count n = 12 → k ≤ n)) ∧ k = 72 :=
by {
  sorry,
}

end least_positive_integer_with_12_factors_l39_39993


namespace sum_of_first_15_odd_positives_l39_39302

theorem sum_of_first_15_odd_positives : 
  let a := 1 in
  let d := 2 in
  let n := 15 in
  (n * (2 * a + (n - 1) * d)) / 2 = 225 :=
by
  sorry

end sum_of_first_15_odd_positives_l39_39302


namespace min_segment_length_l39_39412

theorem min_segment_length 
  (angle : ℝ) (P : ℝ × ℝ)
  (dist_x : ℝ) (dist_y : ℝ) 
  (hx : P.1 ≤ dist_x ∧ P.2 = dist_y)
  (hy : P.2 ≤ dist_y ∧ P.1 = dist_x)
  (right_angle : angle = 90) 
  : ∃ (d : ℝ), d = 10 :=
by
  sorry

end min_segment_length_l39_39412


namespace rational_coefficient_term_is_third_l39_39757

noncomputable def general_term (r : ℕ) : ℝ :=
  Nat.choose 4 r * (-1 : ℝ)^r * (2 : ℝ)^((4 - 2 * r) / 3 : ℝ)

theorem rational_coefficient_term_is_third :
  ∃ (r : ℕ), 0 ≤ r ∧ r ≤ 4 ∧ (general_term r) = 1 ∧ r = 2 
  :=
by
  use 2
  split
  { exact Nat.zero_le _ }
  split
  { -- We use the fact that r <= 4
    exact le_refl 4 }
  split
  { -- We compute the general term when r = 2
    simp [general_term]
    sorry } -- You would compute the term and show it is 1 here
  { -- Finally, we show r = 2
    exact rfl }

end rational_coefficient_term_is_third_l39_39757


namespace interest_second_month_l39_39918

theorem interest_second_month {P r n : ℝ} (hP : P = 200) (hr : r = 0.10) (hn : n = 12) :
  (P * (1 + r / n) ^ (n * (1/12)) - P) * r / n = 1.68 :=
by
  sorry

end interest_second_month_l39_39918


namespace range_of_m_l39_39923

noncomputable def set_A := { x : ℝ | x^2 + x - 6 = 0 }
noncomputable def set_B (m : ℝ) := { x : ℝ | m * x + 1 = 0 }

theorem range_of_m (m : ℝ) : set_A ∪ set_B m = set_A → m = 0 ∨ m = -1 / 2 ∨ m = 1 / 3 :=
by
  sorry

end range_of_m_l39_39923


namespace problem1_problem2_l39_39715

-- Define the first problem
theorem problem1 : (Real.cos (25 / 3 * Real.pi) + Real.tan (-15 / 4 * Real.pi)) = 3 / 2 :=
by
  sorry

-- Define vector operations and the problem
variables (a b : ℝ)

theorem problem2 : 2 * (a - b) - (2 * a + b) + 3 * b = 0 :=
by
  sorry

end problem1_problem2_l39_39715


namespace students_interested_both_l39_39045

/-- total students surveyed -/
def U : ℕ := 50

/-- students who liked watching table tennis matches -/
def A : ℕ := 35

/-- students who liked watching badminton matches -/
def B : ℕ := 30

/-- students not interested in either -/
def nU_not_interest : ℕ := 5

theorem students_interested_both : (A + B - (U - nU_not_interest)) = 20 :=
by sorry

end students_interested_both_l39_39045


namespace angle_alpha_not_2pi_over_9_l39_39722

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.cos (2 * x)) * (Real.cos (4 * x))

theorem angle_alpha_not_2pi_over_9 (α : ℝ) (h : f α = 1 / 8) : α ≠ 2 * π / 9 :=
sorry

end angle_alpha_not_2pi_over_9_l39_39722


namespace subsets_P_count_l39_39887

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {2, 4, 6}
def P : Set ℕ := M ∩ N

theorem subsets_P_count : (Set.powerset P).card = 4 := by
  sorry

end subsets_P_count_l39_39887


namespace no_solutions_l39_39356

theorem no_solutions (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hne : a + b ≠ 0) :
  ¬ (1 / a + 2 / b = 3 / (a + b)) :=
by { sorry }

end no_solutions_l39_39356


namespace convert_neg_900_deg_to_rad_l39_39041

theorem convert_neg_900_deg_to_rad : (-900 : ℝ) * (Real.pi / 180) = -5 * Real.pi :=
by
  sorry

end convert_neg_900_deg_to_rad_l39_39041


namespace max_value_of_f_l39_39060

noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x - 1/2

theorem max_value_of_f : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ (∀ y, (0 ≤ y ∧ y ≤ 2) → f y ≤ f x) ∧ f x = -3 :=
by
  sorry

end max_value_of_f_l39_39060


namespace arithmetic_sequence_geometric_condition_l39_39497

theorem arithmetic_sequence_geometric_condition :
  ∃ d : ℝ, d ≠ 0 ∧ (∀ (a_n : ℕ → ℝ), (a_n 1 = 1) ∧ 
    (a_n 3 = a_n 1 + 2 * d) ∧ (a_n 13 = a_n 1 + 12 * d) ∧ 
    (a_n 3 ^ 2 = a_n 1 * a_n 13) ↔ d = 2) :=
by 
  sorry

end arithmetic_sequence_geometric_condition_l39_39497


namespace probability_of_rain_at_most_3_days_in_july_l39_39907

open Nat

def probability_of_rain (k n : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

def total_probability_of_rain (n : ℕ) (p : ℚ) (max_days : ℕ) : ℚ :=
  (Finset.range (max_days + 1)).sum (λ k, probability_of_rain k n p)

theorem probability_of_rain_at_most_3_days_in_july :
  (total_probability_of_rain 31 (1 / 5) 3).toReal ≈ 0.191 :=
  sorry

end probability_of_rain_at_most_3_days_in_july_l39_39907


namespace ratio_boys_girls_l39_39564

theorem ratio_boys_girls
  (B G : ℕ)  -- Number of boys and girls
  (h_ratio : 75 * G = 80 * B)
  (h_total_no_scholarship : 100 * (3 * B + 4 * G) = 7772727272727272 * (B + G)) :
  B = 5 * G := sorry

end ratio_boys_girls_l39_39564


namespace exterior_angle_regular_octagon_l39_39910

theorem exterior_angle_regular_octagon : ∀ (n : ℕ), n = 8 → (180 - (1080 / n)) = 45 :=
by
  intros n h
  rw h
  sorry

end exterior_angle_regular_octagon_l39_39910


namespace repeating_decimal_multiplication_l39_39050

theorem repeating_decimal_multiplication :
  (0.0808080808 : ℝ) * (0.3333333333 : ℝ) = (8 / 297) := by
  sorry

end repeating_decimal_multiplication_l39_39050


namespace jimmy_fill_pool_time_l39_39423

theorem jimmy_fill_pool_time (pool_gallons : ℕ) (bucket_gallons : ℕ) (time_per_trip_sec : ℕ) (sec_per_min : ℕ) :
  pool_gallons = 84 → 
  bucket_gallons = 2 → 
  time_per_trip_sec = 20 → 
  sec_per_min = 60 → 
  (pool_gallons / bucket_gallons) * time_per_trip_sec / sec_per_min = 14 :=
by
  sorry

end jimmy_fill_pool_time_l39_39423


namespace total_number_of_fish_l39_39625

theorem total_number_of_fish
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (blue_spotted_fish : ℕ)
  (h1 : blue_fish = total_fish / 3)
  (h2 : blue_spotted_fish = blue_fish / 2)
  (h3 : blue_spotted_fish = 10) :
  total_fish = 60 :=
by
  sorry

end total_number_of_fish_l39_39625


namespace gcd_of_2475_and_7350_is_225_l39_39814

-- Definitions and conditions based on the factorization of the given numbers
def factor_2475 := (5^2 * 3^2 * 11)
def factor_7350 := (2 * 3^2 * 5^2 * 7)

-- Proof problem: showing the GCD of 2475 and 7350 is 225
theorem gcd_of_2475_and_7350_is_225 : Nat.gcd 2475 7350 = 225 :=
by
  -- Formal proof would go here
  sorry

end gcd_of_2475_and_7350_is_225_l39_39814


namespace distance_to_nearest_edge_l39_39492

theorem distance_to_nearest_edge (wall_width picture_width : ℕ) (h1 : wall_width = 19) (h2 : picture_width = 3) (h3 : 2 * x + picture_width = wall_width) :
  x = 8 :=
by
  sorry

end distance_to_nearest_edge_l39_39492


namespace solve_problem_l39_39542

noncomputable def solution_set : Set ℤ := {x | abs (7 * x - 5) ≤ 9}

theorem solve_problem : solution_set = {0, 1, 2} := by
  sorry

end solve_problem_l39_39542


namespace line_passes_through_fixed_point_l39_39662

theorem line_passes_through_fixed_point 
  (m : ℝ) : ∃ x y : ℝ, y = m * x + (2 * m + 1) ∧ (x, y) = (-2, 1) :=
by
  use (-2), (1)
  sorry

end line_passes_through_fixed_point_l39_39662


namespace find_b_l39_39516

theorem find_b 
  (a b c x : ℝ)
  (h : (3 * x^2 - 4 * x + 5 / 2) * (a * x^2 + b * x + c) 
       = 6 * x^4 - 17 * x^3 + 11 * x^2 - 7 / 2 * x + 5 / 3) 
  (ha : 3 * a = 6) : b = -3 := 
by 
  sorry

end find_b_l39_39516


namespace weight_of_second_triangle_l39_39162

theorem weight_of_second_triangle :
  let side_len1 := 4
  let density1 := 0.9
  let weight1 := 10.8
  let side_len2 := 6
  let density2 := 1.2
  let weight2 := 18.7
  let area1 := (side_len1 ^ 2 * Real.sqrt 3) / 4
  let area2 := (side_len2 ^ 2 * Real.sqrt 3) / 4
  let calc_weight1 := area1 * density1
  let calc_weight2 := area2 * density2
  calc_weight1 = weight1 → calc_weight2 = weight2 := 
by
  intros
  -- Proof logic goes here
  sorry

end weight_of_second_triangle_l39_39162


namespace probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l39_39635

theorem probability_winning_on_first_draw : 
  let red := 1 
  let yellow := 3 
  red / (red + yellow) = 1 / 4 :=
by 
  sorry

theorem optimal_ball_to_add_for_fine_gift :
  let red := 1 
  let yellow := 3
  -- After adding a red ball: 2 red, 3 yellow
  let p1 := (2 * 1 + 3 * 2) / (2 + 3) / (1 + 3) = (2/5)
  -- After adding a yellow ball: 1 red, 4 yellow
  let p2 := (1 * 0 + 4 * 3) / (1 + 4) / (1 + 3) = (3/5)
  p1 < p2 :=
by 
  sorry

end probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l39_39635


namespace molecular_weight_of_one_mole_l39_39287

theorem molecular_weight_of_one_mole (total_molecular_weight : ℝ) (number_of_moles : ℕ) (h1 : total_molecular_weight = 304) (h2 : number_of_moles = 4) : 
  total_molecular_weight / number_of_moles = 76 := 
by
  sorry

end molecular_weight_of_one_mole_l39_39287


namespace midpoint_polar_coordinates_l39_39088

theorem midpoint_polar_coordinates (r θ_A θ_B : ℝ) (h1 : θ_A = π / 3) (h2 : θ_B = 2 * π / 3) (h3 : r > 0) : 
  let θ_M := (θ_A + θ_B) / 2 in
  (r, θ_M) = (10, π / 2) :=
by
  sorry

end midpoint_polar_coordinates_l39_39088


namespace tea_bags_count_l39_39782

-- Definitions based on the given problem
def valid_bags (b : ℕ) : Prop :=
  ∃ (a c d : ℕ), a + b - a = b ∧ c + d = b ∧ 3 * c + 2 * d = 41 ∧ 3 * a + 2 * (b - a) = 58

-- Statement of the problem, confirming the proof condition
theorem tea_bags_count (b : ℕ) : valid_bags b ↔ b = 20 :=
by {
  -- The proof is left for completion
  sorry
}

end tea_bags_count_l39_39782


namespace factorization_quad_l39_39132

theorem factorization_quad (c d : ℕ) (h_factor : (x^2 - 18 * x + 77 = (x - c) * (x - d)))
  (h_nonneg : c ≥ 0 ∧ d ≥ 0) (h_lt : c > d) : 4 * d - c = 17 := by
  sorry

end factorization_quad_l39_39132


namespace expand_polynomial_l39_39198

theorem expand_polynomial (x : ℝ) : (x + 4) * (5 * x - 10) = 5 * x ^ 2 + 10 * x - 40 := by
  sorry

end expand_polynomial_l39_39198


namespace articles_correct_l39_39014

-- Define the problem conditions
def refersToSpecific (word : String) : Prop :=
  word = "keyboard"

def refersToGeneral (word : String) : Prop :=
  word = "computer"

-- Define the articles
def the_article : String := "the"
def a_article : String := "a"

-- State the theorem for the corresponding solution
theorem articles_correct :
  refersToSpecific "keyboard" → refersToGeneral "computer" →  
  (the_article, a_article) = ("the", "a") :=
by
  intro h1 h2
  sorry

end articles_correct_l39_39014


namespace krakozyabrs_proof_l39_39418

-- Defining necessary variables and conditions
variable {n : ℕ}

-- Conditions given in the problem
def condition1 (H : ℕ) : Prop := 0.2 * H = n
def condition2 (W : ℕ) : Prop := 0.25 * W = n

-- Definition using the principle of inclusion-exclusion
def total_krakozyabrs (H W : ℕ) : ℕ := H + W - n

-- Statements reflecting the conditions and the final conclusion
theorem krakozyabrs_proof (H W : ℕ) (n : ℕ) : condition1 H → condition2 W → (25 < total_krakozyabrs H W) ∧ (total_krakozyabrs H W < 35) → total_krakozyabrs H W = 32 :=
by
  -- We skip the proof here
  sorry

end krakozyabrs_proof_l39_39418


namespace average_pages_per_day_is_correct_l39_39369

-- Definitions based on the given conditions
def first_book_pages := 249
def first_book_days := 3

def second_book_pages := 379
def second_book_days := 5

def third_book_pages := 480
def third_book_days := 6

-- Definition of total pages read
def total_pages := first_book_pages + second_book_pages + third_book_pages

-- Definition of total days spent reading
def total_days := first_book_days + second_book_days + third_book_days

-- Definition of expected average pages per day
def expected_average_pages_per_day := 79.14

-- The theorem to prove
theorem average_pages_per_day_is_correct : (total_pages.toFloat / total_days.toFloat) = expected_average_pages_per_day :=
by
  sorry

end average_pages_per_day_is_correct_l39_39369


namespace total_animal_eyes_l39_39560

def frogs_in_pond := 20
def crocodiles_in_pond := 6
def eyes_per_frog := 2
def eyes_per_crocodile := 2

theorem total_animal_eyes : (frogs_in_pond * eyes_per_frog + crocodiles_in_pond * eyes_per_crocodile) = 52 := by
  sorry

end total_animal_eyes_l39_39560


namespace smallest_five_digit_number_divisible_by_five_primes_l39_39696

theorem smallest_five_digit_number_divisible_by_five_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  lcm = 2310 → (∃ n : ℕ, n = 5 ∧ 10000 ≤ lcm * n ∧ lcm * n = 11550) :=
by
  intros p1 p2 p3 p4 p5 
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  intro hlcm
  use (5 : ℕ)
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end smallest_five_digit_number_divisible_by_five_primes_l39_39696


namespace second_month_interest_l39_39915

def compounded_interest (initial_loan : ℝ) (rate_per_month : ℝ) : ℝ :=
  initial_loan * rate_per_month

theorem second_month_interest :
  let initial_loan := 200
  let rate_per_month := 0.10
  compounded_interest (initial_loan + compounded_interest initial_loan rate_per_month) rate_per_month = 22 :=
by
  sorry

end second_month_interest_l39_39915


namespace minimal_difference_big_small_sum_l39_39232

theorem minimal_difference_big_small_sum :
  ∀ (N : ℕ), N > 0 → ∃ (S : ℕ), 
  S = (N * (N - 1) * (2 * N + 5)) / 6 :=
  by 
    sorry

end minimal_difference_big_small_sum_l39_39232


namespace shortest_distance_correct_l39_39323

noncomputable def shortest_distance_a_to_c1 (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2 + 2 * b * c)

theorem shortest_distance_correct (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) :
  shortest_distance_a_to_c1 a b c h₁ h₂ = Real.sqrt (a^2 + b^2 + c^2 + 2 * b * c) :=
by
  -- This is where the proof would go.
  sorry

end shortest_distance_correct_l39_39323


namespace moles_of_HCl_needed_l39_39365

-- Define the reaction and corresponding stoichiometry
def reaction_relates (NaHSO3 HCl NaCl H2O SO2 : ℕ) : Prop :=
  NaHSO3 = HCl ∧ HCl = NaCl ∧ NaCl = H2O ∧ H2O = SO2

-- Given condition: one mole of each reactant produces one mole of each product
axiom reaction_stoichiometry : reaction_relates 1 1 1 1 1

-- Prove that 2 moles of NaHSO3 reacting with 2 moles of HCl forms 2 moles of NaCl
theorem moles_of_HCl_needed :
  ∀ (NaHSO3 HCl NaCl : ℕ), reaction_relates NaHSO3 HCl NaCl NaCl NaCl → NaCl = 2 → HCl = 2 :=
by
  intros NaHSO3 HCl NaCl h_eq h_NaCl
  sorry

end moles_of_HCl_needed_l39_39365


namespace quadratic_is_perfect_square_l39_39687

theorem quadratic_is_perfect_square (c : ℝ) :
  (∃ b : ℝ, (3 * (x : ℝ) + b)^2 = 9 * x^2 - 24 * x + c) ↔ c = 16 :=
by sorry

end quadratic_is_perfect_square_l39_39687


namespace ladybugs_total_total_ladybugs_is_5_l39_39754

def num_ladybugs (x y : ℕ) : ℕ :=
  x + y

theorem ladybugs_total (x y n : ℕ) 
    (h_spot_calc_1: 6 * x + 4 * y = 30 ∨ 6 * x + 4 * y = 26)
    (h_total_spots_30: (6 * x + 4 * y = 30) ↔ 3 * x + 2 * y = 15)
    (h_total_spots_26: (6 * x + 4 * y = 26) ↔ 3 * x + 2 * y = 13)
    (h_truth_only_one: 
       (6 * x + 4 * y = 30 ∧ ¬(6 * x + 4 * y = 26)) ∨
       (¬(6 * x + 4 * y = 30) ∧ 6 * x + 4 * y = 26))
    : n = x + y :=
by 
  sorry

theorem total_ladybugs_is_5 : ∃ x y : ℕ, num_ladybugs x y = 5 :=
  ⟨3, 2, rfl⟩

end ladybugs_total_total_ladybugs_is_5_l39_39754


namespace rain_first_hour_l39_39575

theorem rain_first_hour (x : ℝ) 
  (h1 : 22 = x + (2 * x + 7)) : x = 5 :=
by
  sorry

end rain_first_hour_l39_39575


namespace function_has_property_T_l39_39556

noncomputable def property_T (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a ≠ b ∧ (f a ≠ 0) ∧ (f b ≠ 0) ∧ (f a * f b = -1)

theorem function_has_property_T : property_T (fun x => 1 + x * Real.log x) :=
sorry

end function_has_property_T_l39_39556


namespace miae_closer_than_hyori_l39_39325

def bowl_volume : ℝ := 1000
def miae_estimate : ℝ := 1100
def hyori_estimate : ℝ := 850

def miae_difference : ℝ := abs (miae_estimate - bowl_volume)
def hyori_difference : ℝ := abs (bowl_volume - hyori_estimate)

theorem miae_closer_than_hyori : miae_difference < hyori_difference :=
by
  sorry

end miae_closer_than_hyori_l39_39325


namespace greatest_good_t_l39_39769

noncomputable def S (a t : ℕ) : Set ℕ := {x | ∃ n : ℕ, x = a + 1 + n ∧ n < t}

def is_good (S : Set ℕ) (k : ℕ) : Prop :=
∃ (coloring : ℕ → Fin k), ∀ (x y : ℕ), x ≠ y → x + y ∈ S → coloring x ≠ coloring y

theorem greatest_good_t {k : ℕ} (hk : k > 1) : ∃ t, ∀ a, is_good (S a t) k ∧ 
  ∀ t' > t, ¬ ∀ a, is_good (S a t') k := 
sorry

end greatest_good_t_l39_39769


namespace solve_for_c_l39_39599

theorem solve_for_c (a b c d e : ℝ) 
  (h1 : a + b + c = 48)
  (h2 : c + d + e = 78)
  (h3 : a + b + c + d + e = 100) :
  c = 26 :=
by
sorry

end solve_for_c_l39_39599


namespace trigonometric_identity_l39_39875

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -2) : 
  2 * Real.sin α * Real.cos α - (Real.cos α)^2 = -1 := 
by
  sorry

end trigonometric_identity_l39_39875


namespace percent_of_y_l39_39476

theorem percent_of_y (y : ℝ) (h : y > 0) : (2 * y) / 10 + (3 * y) / 10 = (50 / 100) * y :=
by
  sorry

end percent_of_y_l39_39476


namespace gemstone_necklaces_sold_correct_l39_39099

-- Define the conditions
def bead_necklaces_sold : Nat := 4
def necklace_cost : Nat := 3
def total_earnings : Nat := 21
def bead_necklaces_earnings : Nat := bead_necklaces_sold * necklace_cost
def gemstone_necklaces_earnings : Nat := total_earnings - bead_necklaces_earnings
def gemstone_necklaces_sold : Nat := gemstone_necklaces_earnings / necklace_cost

-- Theorem to prove the number of gem stone necklaces sold
theorem gemstone_necklaces_sold_correct :
  gemstone_necklaces_sold = 3 :=
by
  -- Proof omitted
  sorry

end gemstone_necklaces_sold_correct_l39_39099


namespace square_difference_l39_39004

theorem square_difference : (601^2 - 599^2 = 2400) :=
by {
  -- Placeholder for the proof
  sorry
}

end square_difference_l39_39004


namespace shaded_area_a_length_EF_b_length_EF_c_ratio_ab_d_l39_39827

-- (a) Prove that the area of the shaded region is 36 cm^2
theorem shaded_area_a (AB EF : ℕ) (h1 : AB = 10) (h2 : EF = 8) : (AB ^ 2) - (EF ^ 2) = 36 :=
by
  sorry

-- (b) Prove that the length of EF is 7 cm
theorem length_EF_b (AB : ℕ) (shaded_area : ℕ) (h1 : AB = 13) (h2 : shaded_area = 120)
  : ∃ EF, (AB ^ 2) - (EF ^ 2) = shaded_area ∧ EF = 7 :=
by
  sorry

-- (c) Prove that the length of EF is 9 cm
theorem length_EF_c (AB : ℕ) (h1 : AB = 18)
  : ∃ EF, (AB ^ 2) - ((1 / 4) * AB ^ 2) = (3 / 4) * AB ^ 2 ∧ EF = 9 :=
by
  sorry

-- (d) Prove that a / b = 5 / 3
theorem ratio_ab_d (a b : ℕ) (shaded_percent : ℚ) (h1 : shaded_percent = 0.64)
  : (a ^ 2) - ((0.36) * a ^ 2) = (a ^ 2) * shaded_percent ∧ (a / b) = (5 / 3) :=
by
  sorry

end shaded_area_a_length_EF_b_length_EF_c_ratio_ab_d_l39_39827


namespace ways_to_fill_table_l39_39513

-- Problem statement in Lean
theorem ways_to_fill_table :
  let even_positions := (4.choose 2) * (3 + (3 * 3))
  let odd_positions := (8.choose 4)^2
  even_positions * odd_positions = 441000 :=
by
  sorry

end ways_to_fill_table_l39_39513


namespace find_k_slope_eq_l39_39328

theorem find_k_slope_eq :
  ∃ k: ℝ, (∃ k: ℝ, ((k - 4) / 7 = (-2 - k) / 14) → k = 2) :=
by
  sorry

end find_k_slope_eq_l39_39328


namespace cab_driver_income_l39_39326

theorem cab_driver_income (x : ℕ) 
  (h₁ : (45 + x + 60 + 65 + 70) / 5 = 58) : x = 50 := 
by
  -- Insert the proof here
  sorry

end cab_driver_income_l39_39326


namespace find_A_l39_39456

theorem find_A (A B : ℕ) (h1 : 15 = 3 * A) (h2 : 15 = 5 * B) : A = 5 := 
by 
  sorry

end find_A_l39_39456


namespace find_d_minus_r_l39_39596

theorem find_d_minus_r :
  ∃ d r : ℕ, 1 < d ∧ 1223 % d = r ∧ 1625 % d = r ∧ 2513 % d = r ∧ d - r = 1 :=
by
  sorry

end find_d_minus_r_l39_39596


namespace solution_set_of_inequality_l39_39147

theorem solution_set_of_inequality (x : ℝ) :
  (x - 1) * (x - 2) ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2 := by
  sorry

end solution_set_of_inequality_l39_39147


namespace wood_stove_afternoon_burn_rate_l39_39182

-- Conditions extracted as definitions
def morning_burn_rate : ℝ := 2
def morning_duration : ℝ := 4
def initial_wood : ℝ := 30
def final_wood : ℝ := 3
def afternoon_duration : ℝ := 4

-- Theorem statement matching the conditions and correct answer
theorem wood_stove_afternoon_burn_rate :
  let morning_burned := morning_burn_rate * morning_duration
  let total_burned := initial_wood - final_wood
  let afternoon_burned := total_burned - morning_burned
  ∃ R : ℝ, (afternoon_burned = R * afternoon_duration) ∧ (R = 4.75) :=
by
  sorry

end wood_stove_afternoon_burn_rate_l39_39182


namespace triangle_area_ratio_l39_39160

theorem triangle_area_ratio (a b c : ℕ) (d e f : ℕ) 
  (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) 
  (h4 : d = 9) (h5 : e = 12) (h6 : f = 15) 
  (GHI_right : a^2 + b^2 = c^2)
  (JKL_right : d^2 + e^2 = f^2):
  (0.5 * a * b) / (0.5 * d * e) = 4 / 9 := 
by 
  sorry

end triangle_area_ratio_l39_39160


namespace hexagon_vertices_zero_l39_39154

theorem hexagon_vertices_zero (n : ℕ) (a0 a1 a2 a3 a4 a5 : ℕ) 
  (h_sum : a0 + a1 + a2 + a3 + a4 + a5 = n) 
  (h_pos : 0 < n) :
  (n = 2 ∨ n % 2 = 1) → 
  ∃ (b0 b1 b2 b3 b4 b5 : ℕ), b0 = 0 ∧ b1 = 0 ∧ b2 = 0 ∧ b3 = 0 ∧ b4 = 0 ∧ b5 = 0 := sorry

end hexagon_vertices_zero_l39_39154


namespace wall_building_time_l39_39576

theorem wall_building_time (m1 m2 d1 d2 k : ℕ) (h1 : m1 = 12) (h2 : d1 = 6) (h3 : m2 = 18) (h4 : k = 72) 
  (condition : m1 * d1 = k) (rate_const : m2 * d2 = k) : d2 = 4 := by
  sorry

end wall_building_time_l39_39576


namespace bills_difference_l39_39646

variable (m j : ℝ)

theorem bills_difference :
  (0.10 * m = 2) → (0.20 * j = 2) → (m - j = 10) :=
by
  intros h1 h2
  sorry

end bills_difference_l39_39646


namespace circle_center_and_radius_l39_39357

-- Define a circle in the plane according to the given equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x = 0

-- Define the center of the circle
def center (x : ℝ) (y : ℝ) : Prop := x = -2 ∧ y = 0

-- Define the radius of the circle
def radius (r : ℝ) : Prop := r = 2

-- The theorem statement
theorem circle_center_and_radius :
  (∀ x y, circle_eq x y → center x y) ∧ radius 2 :=
sorry

end circle_center_and_radius_l39_39357


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l39_39707

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l39_39707


namespace fabric_nguyen_needs_l39_39785

-- Definitions for conditions
def fabric_per_pant : ℝ := 8.5
def total_pants : ℝ := 7
def yards_to_feet (yards : ℝ) : ℝ := yards * 3
def fabric_nguyen_has_yards : ℝ := 3.5

-- The proof we need to establish
theorem fabric_nguyen_needs : (total_pants * fabric_per_pant) - (yards_to_feet fabric_nguyen_has_yards) = 49 :=
by
  sorry

end fabric_nguyen_needs_l39_39785


namespace Camp_Cedar_number_of_counselors_l39_39037

theorem Camp_Cedar_number_of_counselors (boys : ℕ) (girls : ℕ) (total_children : ℕ) (counselors : ℕ)
  (h_boys : boys = 40)
  (h_girls : girls = 3 * boys)
  (h_total_children : total_children = boys + girls)
  (h_counselors : counselors = total_children / 8) :
  counselors = 20 :=
by
  -- this is a statement, so we conclude with sorry to skip the proof.
  sorry

end Camp_Cedar_number_of_counselors_l39_39037


namespace correct_remove_parentheses_l39_39495

theorem correct_remove_parentheses (a b c d : ℝ) :
  (a - (5 * b - (2 * c - 1)) = a - 5 * b + 2 * c - 1) :=
by sorry

end correct_remove_parentheses_l39_39495


namespace sum_first_15_odd_integers_l39_39306

   -- Define the arithmetic sequence and the sum function
   def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

   def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ :=
     (n * (2 * a + (n - 1) * d)) / 2

   -- Constants for the particular problem
   noncomputable def a := 1
   noncomputable def d := 2
   noncomputable def n := 15

   -- Theorem to show the sum of the first 15 odd positive integers
   theorem sum_first_15_odd_integers : sum_arithmetic_seq a d n = 225 := by
     sorry
   
end sum_first_15_odd_integers_l39_39306


namespace distance_Bella_to_Galya_l39_39095

theorem distance_Bella_to_Galya (D_B D_V D_G : ℕ) (BV VG : ℕ)
  (hD_B : D_B = 700)
  (hD_V : D_V = 600)
  (hD_G : D_G = 650)
  (hBV : BV = 100)
  (hVG : VG = 50)
  : BV + VG = 150 := by
  sorry

end distance_Bella_to_Galya_l39_39095


namespace slope_tangent_at_pi_div_six_l39_39882

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x - 2 * Real.cos x

theorem slope_tangent_at_pi_div_six : (deriv f π / 6) = 3 / 2 := 
by 
  sorry

end slope_tangent_at_pi_div_six_l39_39882


namespace coffee_merchant_mixture_price_l39_39833

theorem coffee_merchant_mixture_price
  (c1 c2 : ℝ) (w1 w2 total_cost mixture_price : ℝ)
  (h_c1 : c1 = 9)
  (h_c2 : c2 = 12)
  (h_w1w2 : w1 = 25 ∧ w2 = 25)
  (h_total_weight : w1 + w2 = 100)
  (h_total_cost : total_cost = w1 * c1 + w2 * c2)
  (h_mixture_price : mixture_price = total_cost / (w1 + w2)) :
  mixture_price = 5.25 :=
by sorry

end coffee_merchant_mixture_price_l39_39833


namespace find_tangent_value_l39_39214

noncomputable def tangent_value (a : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), y₀ = x₀ + 1 ∧ y₀ = Real.log (x₀ + a) ∧
  (1 / (x₀ + a) = 1)

theorem find_tangent_value : tangent_value 2 :=
  sorry

end find_tangent_value_l39_39214


namespace correct_reaction_for_phosphoric_acid_l39_39665

-- Define the reactions
def reaction_A := "H₂ + 2OH⁻ - 2e⁻ = 2H₂O"
def reaction_B := "H₂ - 2e⁻ = 2H⁺"
def reaction_C := "O₂ + 4H⁺ + 4e⁻ = 2H₂O"
def reaction_D := "O₂ + 2H₂O + 4e⁻ = 4OH⁻"

-- Define the condition that the electrolyte used is phosphoric acid
def electrolyte := "phosphoric acid"

-- Define the correct reaction
def correct_negative_electrode_reaction := reaction_B

-- Theorem to state that given the conditions above, the correct reaction is B
theorem correct_reaction_for_phosphoric_acid :
  (∃ r, r = reaction_B ∧ electrolyte = "phosphoric acid") :=
by
  sorry

end correct_reaction_for_phosphoric_acid_l39_39665


namespace sum_of_n_natural_numbers_l39_39950

theorem sum_of_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 1035) : n = 46 :=
sorry

end sum_of_n_natural_numbers_l39_39950
