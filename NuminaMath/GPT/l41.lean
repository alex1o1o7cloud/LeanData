import Mathlib

namespace smallest_among_5_8_4_l41_41950

theorem smallest_among_5_8_4 : ∀ (x y z : ℕ), x = 5 → y = 8 → z = 4 → z ≤ x ∧ z ≤ y :=
by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  exact ⟨by norm_num, by norm_num⟩

end smallest_among_5_8_4_l41_41950


namespace sum_of_zeros_gt_two_l41_41342

noncomputable def f (a x : ℝ) := 2 * a * Real.log x + x ^ 2 - 2 * (a + 1) * x

theorem sum_of_zeros_gt_two (a x1 x2 : ℝ) (h_a : -0.5 < a ∧ a < 0)
  (h_fx_zeros : f a x1 = 0 ∧ f a x2 = 0) (h_x_order : x1 < x2) : x1 + x2 > 2 := 
sorry

end sum_of_zeros_gt_two_l41_41342


namespace how_many_children_got_on_l41_41538

noncomputable def initial_children : ℝ := 42.5
noncomputable def children_got_off : ℝ := 21.3
noncomputable def final_children : ℝ := 35.8

theorem how_many_children_got_on : initial_children - children_got_off + (final_children - (initial_children - children_got_off)) = final_children := by
  sorry

end how_many_children_got_on_l41_41538


namespace symmetric_point_l41_41725

-- Define the given point M
def point_M : ℝ × ℝ × ℝ := (1, 0, -1)

-- Define the line in parametric form
def line (t : ℝ) : ℝ × ℝ × ℝ :=
  (3.5 + 2 * t, 1.5 + 2 * t, 0)

-- Define the symmetric point M'
def point_M' : ℝ × ℝ × ℝ := (2, -1, 1)

-- Statement: Prove that M' is the symmetric point to M with respect to the given line
theorem symmetric_point (M M' : ℝ × ℝ × ℝ) (line : ℝ → ℝ × ℝ × ℝ) :
  M = (1, 0, -1) →
  line (t) = (3.5 + 2 * t, 1.5 + 2 * t, 0) →
  M' = (2, -1, 1) :=
sorry

end symmetric_point_l41_41725


namespace solve_correct_problems_l41_41892

theorem solve_correct_problems (x : ℕ) (h1 : 3 * x + x = 120) : x = 30 :=
by
  sorry

end solve_correct_problems_l41_41892


namespace bamboo_pole_is_10_l41_41918

noncomputable def bamboo_pole_length (x : ℕ) : Prop :=
  (x - 4)^2 + (x - 2)^2 = x^2

theorem bamboo_pole_is_10 : bamboo_pole_length 10 :=
by
  -- The proof is not provided
  sorry

end bamboo_pole_is_10_l41_41918


namespace minimum_value_of_f_l41_41608

noncomputable def f (a b x : ℝ) := (a * x + b) / (x^2 + 4)

theorem minimum_value_of_f (a b : ℝ) (h1 : f a b (-1) = 1)
  (h2 : (deriv (f a b)) (-1) = 0) : 
  ∃ (x : ℝ), f a b x = -1 / 4 := 
sorry

end minimum_value_of_f_l41_41608


namespace no_solutions_rebus_l41_41923

theorem no_solutions_rebus : ∀ (K U S Y : ℕ), 
  (K ≠ U ∧ K ≠ S ∧ K ≠ Y ∧ U ≠ S ∧ U ≠ Y ∧ S ≠ Y) →
  (∀ d, d < 10) → 
  let KUSY := 1000 * K + 100 * U + 10 * S + Y in
  let UKSY := 1000 * U + 100 * K + 10 * S + Y in
  let result := 10000 * U + 1000 * K + 100 * S + 10 * Y + S in
  KUSY + UKSY ≠ result :=
begin
  sorry
end

end no_solutions_rebus_l41_41923


namespace jean_spots_on_sides_l41_41507

variables (total_spots upper_torso_spots back_hindquarters_spots side_spots : ℕ)

def half (x : ℕ) := x / 2
def third (x : ℕ) := x / 3

-- Given conditions
axiom h1 : upper_torso_spots = 30
axiom h2 : upper_torso_spots = half total_spots
axiom h3 : back_hindquarters_spots = third total_spots
axiom h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots

-- Theorem to prove
theorem jean_spots_on_sides (h1 : upper_torso_spots = 30)
    (h2 : upper_torso_spots = half total_spots)
    (h3 : back_hindquarters_spots = third total_spots)
    (h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots) :
    side_spots = 10 := by
  sorry

end jean_spots_on_sides_l41_41507


namespace brokerage_percentage_l41_41661

theorem brokerage_percentage
  (cash_realized : ℝ)
  (cash_before_brokerage : ℝ)
  (h₁ : cash_realized = 109.25)
  (h₂ : cash_before_brokerage = 109) :
  ((cash_realized - cash_before_brokerage) / cash_before_brokerage) * 100 = 0.23 := 
by
  sorry

end brokerage_percentage_l41_41661


namespace weight_of_new_man_l41_41817

theorem weight_of_new_man (avg_increase : ℝ) (num_oarsmen : ℕ) (old_weight : ℝ) (weight_increase : ℝ) 
  (h1 : avg_increase = 1.8) (h2 : num_oarsmen = 10) (h3 : old_weight = 53) (h4 : weight_increase = num_oarsmen * avg_increase) :
  ∃ W : ℝ, W = old_weight + weight_increase :=
by
  sorry

end weight_of_new_man_l41_41817


namespace geometric_progression_common_ratio_l41_41210

theorem geometric_progression_common_ratio (a r : ℝ) (h_pos : a > 0)
  (h_eq : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)) :
  r = 1/2 :=
sorry

end geometric_progression_common_ratio_l41_41210


namespace solve_xyz_l41_41614

theorem solve_xyz (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y + x) : x + y + z = 14 * x :=
by
  sorry

end solve_xyz_l41_41614


namespace each_person_paid_l41_41158

-- Define the conditions: total bill and number of people
def totalBill : ℕ := 135
def numPeople : ℕ := 3

-- Define the question as a theorem to prove the correct answer
theorem each_person_paid : totalBill / numPeople = 45 :=
by
  -- Here, we can skip the proof since the statement is required only.
  sorry

end each_person_paid_l41_41158


namespace problem1_problem2_l41_41347

-- Definitions from the conditions
def A (x : ℝ) : Prop := -1 < x ∧ x < 3

def B (x m : ℝ) : Prop := x^2 - 2 * m * x + m^2 - 1 < 0

-- Intersection problem
theorem problem1 (h₁ : ∀ x, A x ↔ (-1 < x ∧ x < 3))
  (h₂ : ∀ x, B x 3 ↔ (2 < x ∧ x < 4)) :
  ∀ x, (A x ∧ B x 3) ↔ (2 < x ∧ x < 3) := by
  sorry

-- Union problem
theorem problem2 (h₃ : ∀ x, A x ↔ (-1 < x ∧ x < 3))
  (h₄ : ∀ x m, B x m ↔ ((x - m)^2 < 1)) :
  ∀ m, (0 ≤ m ∧ m ≤ 2) ↔ (∀ x, A x ∨ B x m → A x) := by
  sorry

end problem1_problem2_l41_41347


namespace six_digit_square_number_cases_l41_41989

theorem six_digit_square_number_cases :
  ∃ n : ℕ, 316 ≤ n ∧ n < 1000 ∧ (n^2 = 232324 ∨ n^2 = 595984 ∨ n^2 = 929296) :=
by {
  sorry
}

end six_digit_square_number_cases_l41_41989


namespace find_largest_number_among_three_l41_41421

noncomputable def A (B : ℝ) := 2 * B - 43
noncomputable def C (A : ℝ) := 0.5 * A + 5

-- The main statement to be proven
theorem find_largest_number_among_three : 
  ∃ (A B C : ℝ), 
  A + B + C = 50 ∧ 
  A = 2 * B - 43 ∧ 
  C = 0.5 * A + 5 ∧ 
  max A (max B C) = 27.375 :=
by
  sorry

end find_largest_number_among_three_l41_41421


namespace binom_8_5_eq_56_l41_41013

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l41_41013


namespace perp_condition_vector_difference_magnitude_l41_41351

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))
noncomputable def vector_c : ℝ × ℝ := (Real.sqrt 3, -1)

-- Condition for perpendicular vectors
theorem perp_condition (x : ℝ) :
  vector_a x.1 * vector_b x.1 + vector_a x.2 * vector_b x.2 = 0 ↔
  ∃ k : ℤ, x = k * (Real.pi / 2) + (Real.pi / 4) := sorry

-- Magnitude bounds for vector difference
theorem vector_difference_magnitude (x : ℝ) :
  1 ≤ Real.sqrt ((vector_a x).1 - vector_c.1)^2 + ((vector_a x).2 - vector_c.2)^2 ≤ 3 := sorry

end perp_condition_vector_difference_magnitude_l41_41351


namespace tim_tasks_per_day_l41_41427

theorem tim_tasks_per_day (earnings_per_task : ℝ) (days_per_week : ℕ) (weekly_earnings : ℝ) :
  earnings_per_task = 1.2 ∧ days_per_week = 6 ∧ weekly_earnings = 720 → (weekly_earnings / days_per_week / earnings_per_task = 100) :=
by
  sorry

end tim_tasks_per_day_l41_41427


namespace suit_price_after_discount_l41_41440

-- Definitions based on given conditions 
def original_price : ℝ := 200
def price_increase : ℝ := 0.30 * original_price
def new_price : ℝ := original_price + price_increase
def discount : ℝ := 0.30 * new_price
def final_price : ℝ := new_price - discount

-- The theorem
theorem suit_price_after_discount :
  final_price = 182 :=
by
  -- Here we would provide the proof if required
  sorry

end suit_price_after_discount_l41_41440


namespace efficiency_ratio_l41_41549

theorem efficiency_ratio (A B : ℝ) (h1 : A ≠ B)
  (h2 : A + B = 1 / 7)
  (h3 : B = 1 / 21) :
  A / B = 2 :=
by
  sorry

end efficiency_ratio_l41_41549


namespace harold_grocery_expense_l41_41194

theorem harold_grocery_expense:
  ∀ (income rent car_payment savings utilities remaining groceries : ℝ),
    income = 2500 →
    rent = 700 →
    car_payment = 300 →
    utilities = 0.5 * car_payment →
    remaining = income - rent - car_payment - utilities →
    savings = 0.5 * remaining →
    (remaining - savings) = 650 →
    groceries = (remaining - 650) →
    groceries = 50 :=
by
  intros income rent car_payment savings utilities remaining groceries
  intro h_income
  intro h_rent
  intro h_car_payment
  intro h_utilities
  intro h_remaining
  intro h_savings
  intro h_final_remaining
  intro h_groceries
  sorry

end harold_grocery_expense_l41_41194


namespace cloth_gain_percentage_l41_41816

theorem cloth_gain_percentage 
  (x : ℝ) -- x represents the cost price of 1 meter of cloth
  (CP : ℝ := 30 * x) -- CP of 30 meters of cloth
  (profit : ℝ := 10 * x) -- profit from selling 30 meters of cloth
  (SP : ℝ := CP + profit) -- selling price of 30 meters of cloth
  (gain_percentage : ℝ := (profit / CP) * 100) : 
  gain_percentage = 33.33 := 
sorry

end cloth_gain_percentage_l41_41816


namespace larger_number_solution_l41_41795

theorem larger_number_solution (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
by
  sorry

end larger_number_solution_l41_41795


namespace sum_of_positive_odd_divisors_of_90_l41_41955

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l41_41955


namespace compound_interest_rate_l41_41697

theorem compound_interest_rate (P r : ℝ) (h1 : 17640 = P * (1 + r / 100)^8)
                                (h2 : 21168 = P * (1 + r / 100)^12) :
  4 * (r / 100) = 18.6 :=
by
  sorry

end compound_interest_rate_l41_41697


namespace mike_total_payment_l41_41391

def camera_initial_cost : ℝ := 4000
def camera_increase_rate : ℝ := 0.30
def lens_initial_cost : ℝ := 400
def lens_discount : ℝ := 200
def sales_tax_rate : ℝ := 0.08

def new_camera_cost := camera_initial_cost * (1 + camera_increase_rate)
def discounted_lens_cost := lens_initial_cost - lens_discount
def total_purchase_before_tax := new_camera_cost + discounted_lens_cost
def sales_tax := total_purchase_before_tax * sales_tax_rate
def total_purchase_with_tax := total_purchase_before_tax + sales_tax

theorem mike_total_payment : total_purchase_with_tax = 5832 := by
  sorry

end mike_total_payment_l41_41391


namespace problem_a_problem_b_l41_41442

-- Problem (a): Prove that (1 + 1/x)(1 + 1/y) ≥ 9 given x > 0, y > 0, and x + y = 1
theorem problem_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  (1 + 1 / x) * (1 + 1 / y) ≥ 9 := sorry

-- Problem (b): Prove that 0 < u + v - uv < 1 given 0 < u < 1 and 0 < v < 1
theorem problem_b (u v : ℝ) (hu : 0 < u) (hu1 : u < 1) (hv : 0 < v) (hv1 : v < 1) : 
  0 < u + v - u * v ∧ u + v - u * v < 1 := sorry

end problem_a_problem_b_l41_41442


namespace correct_union_l41_41491

universe u

-- Definitions
def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2}
def C_I (A : Set ℕ) : Set ℕ := {x ∈ I | x ∉ A}

-- Theorem statement
theorem correct_union : B ∪ C_I A = {2, 4, 5} :=
by
  sorry

end correct_union_l41_41491


namespace alice_password_probability_l41_41696

-- Definitions
def two_digit_count : ℕ := 100
def even_two_digit_count : ℕ := 50
def symbol_set : finset char := {'$', '%', '@', '!', '#'}
def favorable_symbols : finset char := {'$', '%', '@'}
def favorable_symbol_count : ℕ := 3
def symbol_count : ℕ := 5

-- Theorem Statement
theorem alice_password_probability : 
  (even_two_digit_count / two_digit_count) * 
  (favorable_symbol_count / symbol_count) * 
  (even_two_digit_count / two_digit_count) = 3 / 20 := by
{
  sorry
}

end alice_password_probability_l41_41696


namespace max_sinA_sinC_l41_41747

variable (a b c A B C : ℝ)

-- Assuming the sides opposite to angles A, B, C in triangle ABC are a, b, c respectively.
axiom triangle_ABC: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

-- Given condition a ∙ cos(A) = b ∙ sin(A)
axiom a_cosA_eq_b_sinA: a * Real.cos A = b * Real.sin A

-- Given condition B > π / 2
axiom B_gt_half_pi: B > Real.pi / 2

-- Angle sum in triangle A + B + C = π
axiom angle_sum: A + B + C = Real.pi

-- Prove that the maximum value of Real.sin A + Real.sin C equals 9 / 8
theorem max_sinA_sinC : Real.sin A + Real.sin C ≤ 9 / 8 :=
by
  sorry

end max_sinA_sinC_l41_41747


namespace solve_for_x_l41_41138

theorem solve_for_x : ∃ x : ℝ, (9 - x) ^ 2 = x ^ 2 ∧ x = 4.5 :=
by
  sorry

end solve_for_x_l41_41138


namespace volume_of_second_cube_is_twosqrt2_l41_41949

noncomputable def side_length (volume : ℝ) : ℝ :=
  volume^(1/3)

noncomputable def surface_area (side : ℝ) : ℝ :=
  6 * side^2

theorem volume_of_second_cube_is_twosqrt2
  (v1 : ℝ)
  (h1 : v1 = 1)
  (A1 := surface_area (side_length v1))
  (A2 := 2 * A1)
  (s2 := (A2 / 6)^(1/2)) :
  (s2^3 = 2 * Real.sqrt 2) :=
by
  sorry

end volume_of_second_cube_is_twosqrt2_l41_41949


namespace seconds_in_part_of_day_l41_41876

theorem seconds_in_part_of_day : (1 / 4) * (1 / 6) * (1 / 8) * 24 * 60 * 60 = 450 := by
  sorry

end seconds_in_part_of_day_l41_41876


namespace find_x_l41_41181

def perpendicular_vectors_solution (x : ℝ) : Prop :=
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (3, -2)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 2 / 3

theorem find_x (x : ℝ) : perpendicular_vectors_solution x := sorry

end find_x_l41_41181


namespace baba_yagas_savings_plan_l41_41085

-- Definitions for income and expenses
def salary (gross: ℝ) (taxRate: ℝ) : ℝ := gross * (1 - taxRate)

def familyIncome (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship: ℝ)
  (mothersPension: ℝ) (taxRate: ℝ) (date: ℕ) : ℝ :=
  if date < 20180501 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + (salary mothersSalary taxRate) + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else if date < 20180901 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship + (salary sonsNonStateScholarship taxRate)

def monthlyExpenses : ℝ := 74000

def monthlySavings (income: ℝ) (expenses: ℝ) : ℝ := income - expenses

-- Theorem to prove
theorem baba_yagas_savings_plan :
  ∀ (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship mothersPension: ℝ)
  (taxRate: ℝ),
  ivansSalary = 55000 → vasilisasSalary = 45000 → mothersSalary = 18000 →
  fathersSalary = 20000 → sonsStateScholarship = 3000 → sonsNonStateScholarship = 15000 →
  mothersPension = 10000 → taxRate = 0.13 →
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180430) monthlyExpenses = 49060 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180501) monthlyExpenses = 43400 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180901) monthlyExpenses = 56450 :=
by
  sorry

end baba_yagas_savings_plan_l41_41085


namespace area_square_field_l41_41441

-- Define the side length of the square
def side_length : ℕ := 12

-- Define the area of the square with the given side length
def area_of_square (side : ℕ) : ℕ := side * side

-- The theorem to state and prove
theorem area_square_field : area_of_square side_length = 144 :=
by
  sorry

end area_square_field_l41_41441


namespace fifth_runner_twice_as_fast_reduction_l41_41322

/-- Given the individual times of the five runners and the percentage reductions, 
if the fifth runner had run twice as fast, the total time reduction is 8%. -/
theorem fifth_runner_twice_as_fast_reduction (T T1 T2 T3 T4 T5 : ℝ)
  (h1 : T = T1 + T2 + T3 + T4 + T5)
  (h2 : T1 = 0.10 * T)
  (h3 : T2 = 0.20 * T)
  (h4 : T3 = 0.24 * T)
  (h5 : T4 = 0.30 * T)
  (h6 : T5 = 0.16 * T) :
  let T' := T1 + T2 + T3 + T4 + T5 / 2 in
  T - T' = 0.08 * T :=
by
  sorry

end fifth_runner_twice_as_fast_reduction_l41_41322


namespace integers_satisfying_condition_l41_41941

-- Define the condition
def condition (x : ℤ) : Prop := x * x < 3 * x

-- Define the theorem stating the proof problem
theorem integers_satisfying_condition :
  {x : ℤ | condition x} = {1, 2} :=
by
  sorry

end integers_satisfying_condition_l41_41941


namespace number_of_commonly_used_structures_is_3_l41_41396

def commonly_used_algorithm_structures : Nat := 3
theorem number_of_commonly_used_structures_is_3 
  (structures : Nat)
  (h : structures = 1 ∨ structures = 2 ∨ structures = 3 ∨ structures = 4) :
  commonly_used_algorithm_structures = 3 :=
by
  -- Proof to be added
  sorry

end number_of_commonly_used_structures_is_3_l41_41396


namespace save_plan_l41_41091

noncomputable def net_income (gross: ℕ) : ℕ :=
  (gross * 87) / 100

def ivan_salary : ℕ := net_income 55000
def vasilisa_salary : ℕ := net_income 45000
def vasalisa_mother_salary_before : ℕ := net_income 18000
def vasalisa_father_salary : ℕ := net_income 20000
def son_scholarship_state : ℕ := 3000
def son_scholarship_non_state : ℕ := net_income 15000

def expenses : ℕ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

def savings_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state - expenses

def total_income_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state

def total_income_may_august : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state

def savings_may_august : ℕ :=
  total_income_may_august - expenses

def total_income_september : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state + son_scholarship_non_state

def savings_september : ℕ :=
  total_income_september - expenses

theorem save_plan : 
  savings_before_may = 49060 ∧ savings_may_august = 43400 ∧ savings_september = 56450 :=
by
  sorry

end save_plan_l41_41091


namespace sufficient_not_necessary_l41_41388

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2) ↔ (x + y ≥ 4) :=
by sorry

end sufficient_not_necessary_l41_41388


namespace conversion1_conversion2_conversion3_minutes_conversion3_seconds_conversion4_l41_41444

theorem conversion1 : 4 * 60 + 35 = 275 := by
  sorry

theorem conversion2 : 4 * 1000 + 35 = 4035 := by
  sorry

theorem conversion3_minutes : 678 / 60 = 11 := by
  sorry

theorem conversion3_seconds : 678 % 60 = 18 := by
  sorry

theorem conversion4 : 120000 / 10000 = 12 := by
  sorry

end conversion1_conversion2_conversion3_minutes_conversion3_seconds_conversion4_l41_41444


namespace total_apples_eaten_l41_41306

def Apples_Tuesday : ℕ := 4
def Apples_Wednesday : ℕ := 2 * Apples_Tuesday
def Apples_Thursday : ℕ := Apples_Tuesday / 2

theorem total_apples_eaten : Apples_Tuesday + Apples_Wednesday + Apples_Thursday = 14 := by
  sorry

end total_apples_eaten_l41_41306


namespace exists_y_square_divisible_by_five_btw_50_and_120_l41_41856

theorem exists_y_square_divisible_by_five_btw_50_and_120 : ∃ y : ℕ, (∃ k : ℕ, y = k^2) ∧ (y % 5 = 0) ∧ (50 ≤ y ∧ y ≤ 120) ∧ y = 100 :=
by
  sorry

end exists_y_square_divisible_by_five_btw_50_and_120_l41_41856


namespace total_bronze_needed_l41_41080

theorem total_bronze_needed (w1 w2 w3 : ℕ) (h1 : w1 = 50) (h2 : w2 = 2 * w1) (h3 : w3 = 4 * w2) : w1 + w2 + w3 = 550 :=
by
  -- We'll complete the proof later
  sorry

end total_bronze_needed_l41_41080


namespace problem1_problem2_l41_41597

-- Problem 1: Prove that (2sin(α) - cos(α)) / (sin(α) + 2cos(α)) = 3/4 given tan(α) = 2
theorem problem1 (α : ℝ) (h : Real.tan α = 2) : (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := 
sorry

-- Problem 2: Prove that 2sin^2(x) - sin(x)cos(x) + cos^2(x) = 2 - sin(2x)/2
theorem problem2 (x : ℝ) : 2 * (Real.sin x)^2 - (Real.sin x) * (Real.cos x) + (Real.cos x)^2 = 2 - Real.sin (2 * x) / 2 := 
sorry

end problem1_problem2_l41_41597


namespace problem1_problem2_l41_41160

-- Lean statement for Problem 1
theorem problem1 (x : ℝ) : x^2 * x^3 - x^5 = 0 := 
by sorry

-- Lean statement for Problem 2
theorem problem2 (a : ℝ) : (a + 1)^2 + 2 * a * (a - 1) = 3 * a^2 + 1 :=
by sorry

end problem1_problem2_l41_41160


namespace people_own_pets_at_least_l41_41262

-- Definitions based on given conditions
def people_owning_only_dogs : ℕ := 15
def people_owning_only_cats : ℕ := 10
def people_owning_only_cats_and_dogs : ℕ := 5
def people_owning_cats_dogs_snakes : ℕ := 3
def total_snakes : ℕ := 59

-- Theorem statement to prove the total number of people owning pets
theorem people_own_pets_at_least : 
  people_owning_only_dogs + people_owning_only_cats + people_owning_only_cats_and_dogs + people_owning_cats_dogs_snakes ≥ 33 :=
by {
  -- Proof steps will go here
  sorry
}

end people_own_pets_at_least_l41_41262


namespace average_age_at_marriage_l41_41307

theorem average_age_at_marriage
  (A : ℕ)
  (combined_age_at_marriage : husband_age + wife_age = 2 * A)
  (combined_age_after_5_years : (A + 5) + (A + 5) + 1 = 57) :
  A = 23 := 
sorry

end average_age_at_marriage_l41_41307


namespace probability_dmitry_before_anatoly_l41_41898

theorem probability_dmitry_before_anatoly (m : ℝ) (non_neg_m : 0 < m) :
  let volume_prism := (m^3) / 2
  let volume_tetrahedron := (m^3) / 3
  let probability := volume_tetrahedron / volume_prism
  probability = (2 : ℝ) / 3 :=
by
  sorry

end probability_dmitry_before_anatoly_l41_41898


namespace work_completion_days_l41_41692

variables (M D X : ℕ) (W : ℝ)

-- Original conditions
def original_men : ℕ := 15
def planned_days : ℕ := 40
def men_absent : ℕ := 5

-- Theorem to prove
theorem work_completion_days :
  M = original_men →
  D = planned_days →
  W > 0 →
  (M - men_absent) * X * W = M * D * W →
  X = 60 :=
by
  intros hM hD hW h_work
  sorry

end work_completion_days_l41_41692


namespace jellybeans_left_in_jar_l41_41423

def original_jellybeans : ℕ := 250
def class_size : ℕ := 24
def sick_children : ℕ := 2
def sick_jellybeans_each : ℕ := 7
def first_group_size : ℕ := 12
def first_group_jellybeans_each : ℕ := 5
def second_group_size : ℕ := 10
def second_group_jellybeans_each : ℕ := 4

theorem jellybeans_left_in_jar : 
  original_jellybeans - ((first_group_size * first_group_jellybeans_each) + 
  (second_group_size * second_group_jellybeans_each)) = 150 := by
  sorry

end jellybeans_left_in_jar_l41_41423


namespace diane_coins_in_third_piggy_bank_l41_41812

theorem diane_coins_in_third_piggy_bank :
  ∀ n1 n2 n4 n5 n6 : ℕ, n1 = 72 → n2 = 81 → n4 = 99 → n5 = 108 → n6 = 117 → (n4 - (n4 - 9)) = 90 :=
by
  -- sorry is needed to avoid an incomplete proof, as only the statement is required.
  sorry

end diane_coins_in_third_piggy_bank_l41_41812


namespace arc_length_l41_41405

/-- Given a circle with a radius of 5 cm and a sector area of 11.25 cm², 
prove that the length of the arc forming the sector is 4.5 cm. --/
theorem arc_length (r : ℝ) (A : ℝ) (θ : ℝ) (arc_length : ℝ) 
  (h_r : r = 5) 
  (h_A : A = 11.25) 
  (h_area_formula : A = (θ / (2 * π)) * π * r ^ 2) 
  (h_arc_length_formula : arc_length = r * θ) :
  arc_length = 4.5 :=
sorry

end arc_length_l41_41405


namespace hyperbola_equation_l41_41344

-- Define the conditions
def hyperbola_eq := ∀ (x y a b : ℝ), a > 0 ∧ b > 0 → x^2 / a^2 - y^2 / b^2 = 1
def parabola_eq := ∀ (x y : ℝ), y^2 = (2 / 5) * x
def intersection_point_M := ∃ (x : ℝ), ∀ (y : ℝ), y = 1 → y^2 = (2 / 5) * x
def line_intersect_N := ∀ (F₁ M N : ℝ × ℝ), 
  (N.1 = -1 / 10) ∧ (F₁.1 ≠ M.1) ∧ (N.2 = 0)

-- State the proof problem
theorem hyperbola_equation 
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (hyp_eq : hyperbola_eq)
  (par_eq : parabola_eq)
  (int_pt_M : intersection_point_M)
  (line_int_N : line_intersect_N) :
  ∀ (x y : ℝ), x^2 / 5 - y^2 / 4 = 1 :=
by sorry

end hyperbola_equation_l41_41344


namespace one_eighth_of_power_l41_41357

theorem one_eighth_of_power (x : ℕ) (h : (1 / 8) * (2 ^ 36) = 2 ^ x) : x = 33 :=
by 
  -- Proof steps are not needed, so we leave it as sorry.
  sorry

end one_eighth_of_power_l41_41357


namespace even_function_a_is_0_l41_41733

def f (a : ℝ) (x : ℝ) : ℝ := (a+1) * x^2 + 3 * a * x + 1

theorem even_function_a_is_0 (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 :=
by sorry

end even_function_a_is_0_l41_41733


namespace simplify_expression_l41_41654

theorem simplify_expression (a b : ℤ) (h1 : a = 1) (h2 : b = -4) :
  4 * (a^2 * b + a * b^2) - 3 * (a^2 * b - 1) + 2 * a * b^2 - 6 = 89 := by
  sorry

end simplify_expression_l41_41654


namespace right_triangle_hypotenuse_len_l41_41374

theorem right_triangle_hypotenuse_len (a b : ℕ) (c : ℝ) (h₁ : a = 1) (h₂ : b = 3) 
  (h₃ : a^2 + b^2 = c^2) : c = Real.sqrt 10 := by
  sorry

end right_triangle_hypotenuse_len_l41_41374


namespace total_money_is_220_l41_41104

-- Define the amounts on Table A, B, and C
def tableA := 40
def tableC := tableA + 20
def tableB := 2 * tableC

-- Define the total amount of money on all tables
def total_money := tableA + tableB + tableC

-- The main theorem to prove
theorem total_money_is_220 : total_money = 220 :=
by
  sorry

end total_money_is_220_l41_41104


namespace problem_l41_41504

variables {A B C A1 B1 C1 A0 B0 C0 : Type}

-- Define the acute triangle and constructions
axiom acute_triangle (ABC : Type) : Prop
axiom circumcircle (ABC : Type) (A1 B1 C1 : Type) : Prop
axiom extended_angle_bisectors (ABC : Type) (A0 B0 C0 : Type) : Prop

-- Define the points according to the problem statement
axiom intersections_A0 (ABC : Type) (A0 : Type) : Prop
axiom intersections_B0 (ABC : Type) (B0 : Type) : Prop
axiom intersections_C0 (ABC : Type) (C0 : Type) : Prop

-- Define the areas of triangles and hexagon
axiom area_triangle_A0B0C0 (ABC : Type) (A0 B0 C0 : Type) : ℝ
axiom area_hexagon_AC1B_A1CB1 (ABC : Type) (A1 B1 C1 : Type) : ℝ
axiom area_triangle_ABC (ABC : Type) : ℝ

-- Problem: Prove the area relationships
theorem problem
  (ABC: Type)
  (h1 : acute_triangle ABC)
  (h2 : circumcircle ABC A1 B1 C1)
  (h3 : extended_angle_bisectors ABC A0 B0 C0)
  (h4 : intersections_A0 ABC A0)
  (h5 : intersections_B0 ABC B0)
  (h6 : intersections_C0 ABC C0):
  area_triangle_A0B0C0 ABC A0 B0 C0 = 2 * area_hexagon_AC1B_A1CB1 ABC A1 B1 C1 ∧
  area_triangle_A0B0C0 ABC A0 B0 C0 ≥ 4 * area_triangle_ABC ABC :=
sorry

end problem_l41_41504


namespace eval_expr_l41_41165

theorem eval_expr : 3^2 * 4 * 6^3 * Nat.factorial 7 = 39191040 := by
  -- the proof will be filled in here
  sorry

end eval_expr_l41_41165


namespace no_rational_roots_of_prime_3_digit_l41_41642

noncomputable def is_prime (n : ℕ) := Nat.Prime n

theorem no_rational_roots_of_prime_3_digit (a b c : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) 
(h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : 0 ≤ c ∧ c ≤ 9) 
(p := 100 * a + 10 * b + c) (hp : is_prime p) (h₃ : 100 ≤ p ∧ p ≤ 999) :
¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 :=
sorry

end no_rational_roots_of_prime_3_digit_l41_41642


namespace lcm_two_numbers_l41_41685

theorem lcm_two_numbers
  (a b : ℕ)
  (hcf_ab : Nat.gcd a b = 20)
  (product_ab : a * b = 2560) :
  Nat.lcm a b = 128 :=
by
  sorry

end lcm_two_numbers_l41_41685


namespace canoe_stream_speed_l41_41553

theorem canoe_stream_speed (C S : ℝ) (h1 : C - S = 9) (h2 : C + S = 12) : S = 1.5 :=
by
  sorry

end canoe_stream_speed_l41_41553


namespace gcd_linear_combination_l41_41653

theorem gcd_linear_combination (a b : ℤ) : 
  Int.gcd (5 * a + 3 * b) (13 * a + 8 * b) = Int.gcd a b := 
sorry

end gcd_linear_combination_l41_41653


namespace max_distance_curve_line_l41_41256

noncomputable def curve_param_x (θ : ℝ) : ℝ := 1 + Real.cos θ
noncomputable def curve_param_y (θ : ℝ) : ℝ := Real.sin θ
noncomputable def line (x y : ℝ) : Prop := x + y + 2 = 0

theorem max_distance_curve_line 
  (θ : ℝ) 
  (x := curve_param_x θ) 
  (y := curve_param_y θ) :
  ∃ (d : ℝ), 
    (∀ t : ℝ, curve_param_x t = x ∧ curve_param_y t = y → d ≤ (abs (x + y + 2)) / Real.sqrt (1^2 + 1^2)) 
    ∧ d = (3 * Real.sqrt 2) / 2 + 1 :=
sorry

end max_distance_curve_line_l41_41256


namespace total_cows_l41_41578

def number_of_cows_in_herd : ℕ := 40
def number_of_herds : ℕ := 8
def total_number_of_cows (cows_per_herd herds : ℕ) : ℕ := cows_per_herd * herds

theorem total_cows : total_number_of_cows number_of_cows_in_herd number_of_herds = 320 := by
  sorry

end total_cows_l41_41578


namespace greatest_integer_less_than_neg_seventeen_thirds_l41_41135

theorem greatest_integer_less_than_neg_seventeen_thirds : floor (-17 / 3) = -6 := by
  sorry

end greatest_integer_less_than_neg_seventeen_thirds_l41_41135


namespace sum_odd_divisors_90_eq_78_l41_41957

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l41_41957


namespace greatest_third_term_of_arithmetic_sequence_l41_41124

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h₁ : 0 < a)
  (h₂ : 0 < d) (h₃ : 4 * a + 6 * d = 50) : a + 2 * d = 16 :=
by
  -- Using the given condition
  -- 1. 4a + 6d = 50
  -- 2. a and d are in the naturals and greater than 0
  -- We prove that the greatest possible value of the third term (a + 2d)
  -- given these conditions equals 16
  sorry

end greatest_third_term_of_arithmetic_sequence_l41_41124


namespace local_minimum_at_2_l41_41320

open Real

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

def f' (x : ℝ) : ℝ := 3 * x^2 - 12

theorem local_minimum_at_2 :
  (∀ x : ℝ, -2 < x ∧ x < 2 → f' x < 0) →
  (∀ x : ℝ, x > 2 → f' x > 0) →
  (∃ ε > 0, ∀ x : ℝ, abs (x - 2) < ε → f x > f 2) :=
by
  sorry

end local_minimum_at_2_l41_41320


namespace tangency_condition_l41_41350

def functions_parallel (a b c : ℝ) (f g: ℝ → ℝ)
       (parallel: ∀ x, f x = a * x + b ∧ g x = a * x + c) := 
  ∀ x, f x = a * x + b ∧ g x = a * x + c

theorem tangency_condition (a b c A : ℝ)
    (h_parallel : a ≠ 0)
    (h_tangency : (∀ x, (a * x + b)^2 = 7 * (a * x + c))) :
  A = 0 ∨ A = -7 :=
sorry

end tangency_condition_l41_41350


namespace range_of_a_l41_41368

variable {a x : ℝ}

theorem range_of_a (h : ∀ x, (a - 5) * x > a - 5 ↔ x < 1) : a < 5 := 
sorry

end range_of_a_l41_41368


namespace conclusion_l41_41489

-- Assuming U is the universal set and Predicates represent Mems, Ens, and Veens
variable (U : Type)
variable (Mem : U → Prop)
variable (En : U → Prop)
variable (Veen : U → Prop)

-- Hypotheses
variable (h1 : ∀ x, Mem x → En x)          -- Hypothesis I: All Mems are Ens
variable (h2 : ∀ x, En x → ¬Veen x)        -- Hypothesis II: No Ens are Veens

-- To be proven
theorem conclusion (x : U) : (Mem x → ¬Veen x) ∧ (Mem x → ¬Veen x) := sorry

end conclusion_l41_41489


namespace investment_in_real_estate_l41_41526

def total_investment : ℝ := 200000
def ratio_real_estate_to_mutual_funds : ℝ := 7

theorem investment_in_real_estate (mutual_funds_investment real_estate_investment: ℝ) 
  (h1 : mutual_funds_investment + real_estate_investment = total_investment)
  (h2 : real_estate_investment = ratio_real_estate_to_mutual_funds * mutual_funds_investment) :
  real_estate_investment = 175000 := sorry

end investment_in_real_estate_l41_41526


namespace dimensions_increased_three_times_l41_41930

variables (L B H k : ℝ) (n : ℝ)
 
-- Given conditions
axiom cost_initial : 350 = k * 2 * (L + B) * H
axiom cost_increased : 3150 = k * 2 * n^2 * (L + B) * H

-- Proof statement
theorem dimensions_increased_three_times : n = 3 :=
by
  sorry

end dimensions_increased_three_times_l41_41930


namespace avg_temp_in_october_l41_41420

theorem avg_temp_in_october (a A : ℝ)
  (h1 : 28 = a + A)
  (h2 : 18 = a - A)
  (x := 10)
  (temperature : ℝ := a + A * Real.cos (π / 6 * (x - 6))) :
  temperature = 20.5 :=
by
  sorry

end avg_temp_in_october_l41_41420


namespace no_real_solution_x_plus_36_div_x_minus_3_eq_neg9_l41_41169

theorem no_real_solution_x_plus_36_div_x_minus_3_eq_neg9 : ∀ x : ℝ, x + 36 / (x - 3) = -9 → False :=
by
  assume x
  assume h : x + 36 / (x - 3) = -9
  sorry

end no_real_solution_x_plus_36_div_x_minus_3_eq_neg9_l41_41169


namespace alex_height_l41_41445

theorem alex_height
  (tree_height: ℚ) (tree_shadow: ℚ) (alex_shadow_in_inches: ℚ)
  (h_tree: tree_height = 50)
  (h_shadow_tree: tree_shadow = 25)
  (h_shadow_alex: alex_shadow_in_inches = 20) :
  ∃ alex_height_in_feet: ℚ, alex_height_in_feet = 10 / 3 :=
by
  sorry

end alex_height_l41_41445


namespace magazine_cost_l41_41665

theorem magazine_cost (C M : ℝ) 
  (h1 : 4 * C = 8 * M) 
  (h2 : 12 * C = 24) : 
  M = 1 :=
by
  sorry

end magazine_cost_l41_41665


namespace baba_yagas_savings_plan_l41_41086

-- Definitions for income and expenses
def salary (gross: ℝ) (taxRate: ℝ) : ℝ := gross * (1 - taxRate)

def familyIncome (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship: ℝ)
  (mothersPension: ℝ) (taxRate: ℝ) (date: ℕ) : ℝ :=
  if date < 20180501 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + (salary mothersSalary taxRate) + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else if date < 20180901 then
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship
  else
    (salary ivansSalary taxRate) + (salary vasilisasSalary taxRate) + mothersPension + 
    (salary fathersSalary taxRate) + sonsStateScholarship + (salary sonsNonStateScholarship taxRate)

def monthlyExpenses : ℝ := 74000

def monthlySavings (income: ℝ) (expenses: ℝ) : ℝ := income - expenses

-- Theorem to prove
theorem baba_yagas_savings_plan :
  ∀ (ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship sonsNonStateScholarship mothersPension: ℝ)
  (taxRate: ℝ),
  ivansSalary = 55000 → vasilisasSalary = 45000 → mothersSalary = 18000 →
  fathersSalary = 20000 → sonsStateScholarship = 3000 → sonsNonStateScholarship = 15000 →
  mothersPension = 10000 → taxRate = 0.13 →
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180430) monthlyExpenses = 49060 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180501) monthlyExpenses = 43400 ∧
  monthlySavings (familyIncome ivansSalary vasilisasSalary mothersSalary fathersSalary sonsStateScholarship
    sonsNonStateScholarship mothersPension taxRate 20180901) monthlyExpenses = 56450 :=
by
  sorry

end baba_yagas_savings_plan_l41_41086


namespace election_votes_l41_41215

theorem election_votes (V : ℝ) (ha : 0.45 * V = 4860)
                       (hb : 0.30 * V = 3240)
                       (hc : 0.20 * V = 2160)
                       (hd : 0.05 * V = 540)
                       (hmaj : (0.45 - 0.30) * V = 1620) :
                       V = 10800 :=
by
  sorry

end election_votes_l41_41215


namespace nine_questions_insufficient_l41_41804

/--
We have 5 stones with distinct weights and we are allowed to ask nine questions of the form
"Is it true that A < B < C?". Prove that nine such questions are insufficient to always determine
the unique ordering of these stones.
-/
theorem nine_questions_insufficient (stones : Fin 5 → Nat) 
  (distinct_weights : ∀ i j : Fin 5, i ≠ j → stones i ≠ stones j) :
  ¬ (∃ f : { q : Fin 125 | q.1 ≤ 8 } → (Fin 5 → Fin 5 → Fin 5 → Bool),
    ∀ w1 w2 w3 w4 w5 : Fin 120,
      (f ⟨0, sorry⟩) = sorry  -- This line only represents the existence of 9 questions
      )
:=
sorry

end nine_questions_insufficient_l41_41804


namespace exercise_l41_41596

open Set

theorem exercise (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5, 6}) (hA : A = {1, 3, 5}) (hB : B = {2, 4, 5}) :
  A ∩ (U \ B) = {1, 3} := by
  sorry

end exercise_l41_41596


namespace mom_tshirts_count_l41_41680

def packages : ℕ := 71
def tshirts_per_package : ℕ := 6

theorem mom_tshirts_count : packages * tshirts_per_package = 426 := by
  sorry

end mom_tshirts_count_l41_41680


namespace quadratic_roots_square_diff_l41_41746

theorem quadratic_roots_square_diff (α β : ℝ) (h : α ≠ β)
    (hα : α^2 - 3 * α + 2 = 0) (hβ : β^2 - 3 * β + 2 = 0) :
    (α - β)^2 = 1 :=
sorry

end quadratic_roots_square_diff_l41_41746


namespace three_digit_numbers_with_product_30_l41_41054

theorem three_digit_numbers_with_product_30 : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
  (let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 in 
   d1 ≠ 0 ∧ d2 ≠ 0 ∧ d3 ≠ 0 ∧ d1 * d2 * d3 = 30) ↔
  12 := 
sorry

end three_digit_numbers_with_product_30_l41_41054


namespace henry_total_payment_l41_41028

-- Define the conditions
def painting_payment : ℕ := 5
def selling_extra_payment : ℕ := 8
def total_payment_per_bike : ℕ := painting_payment + selling_extra_payment  -- 13

-- Define the quantity of bikes
def bikes_count : ℕ := 8

-- Calculate the total payment for painting and selling 8 bikes
def total_payment : ℕ := bikes_count * total_payment_per_bike  -- 144

-- The statement to prove
theorem henry_total_payment : total_payment = 144 :=
by
  -- Proof goes here
  sorry

end henry_total_payment_l41_41028


namespace field_ratio_l41_41535

theorem field_ratio (w l: ℕ) (h: l = 36)
  (h_area_ratio: 81 = (1/8) * l * w)
  (h_multiple: ∃ k : ℕ, l = k * w) :
  l / w = 2 :=
by 
  sorry

end field_ratio_l41_41535


namespace triangle_area_and_coordinates_l41_41428

noncomputable def positive_diff_of_coordinates (A B C R S : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (xr, yr) := R
  let (xs, ys) := S
  if xr = xs then abs (xr - (10 - (x3 - xr)))
  else 0 -- Should never be this case if conditions are properly followed

theorem triangle_area_and_coordinates
  (A B C R S : ℝ × ℝ)
  (h_A : A = (0, 10))
  (h_B : B = (4, 0))
  (h_C : C = (10, 0))
  (h_vertical : R.fst = S.fst)
  (h_intersect_AC : R.snd = -(R.fst - 10))
  (h_intersect_BC : S.snd = 0 ∧ S.fst = 10 - (C.fst - R.fst))
  (h_area : 1/2 * ((R.fst - C.fst) * (R.snd - C.snd)) = 15) :
  positive_diff_of_coordinates A B C R S = 2 * Real.sqrt 30 - 10 := sorry

end triangle_area_and_coordinates_l41_41428


namespace sum_of_arithmetic_series_l41_41281

-- Define the conditions
def first_term := 1
def last_term := 12
def number_of_terms := 12

-- Prop statement that the sum of the arithmetic series equals 78
theorem sum_of_arithmetic_series : (number_of_terms / 2) * (first_term + last_term) = 78 := 
by
  sorry

end sum_of_arithmetic_series_l41_41281


namespace largest_prime_factor_problem_l41_41436

def largest_prime_factor (n : ℕ) : ℕ :=
  -- This function calculates the largest prime factor of n
  sorry

theorem largest_prime_factor_problem :
  largest_prime_factor 57 = 19 ∧
  largest_prime_factor 133 = 19 ∧
  ∀ n, n = 63 ∨ n = 85 ∨ n = 143 → largest_prime_factor n < 19 :=
by
  sorry

end largest_prime_factor_problem_l41_41436


namespace sphere_volume_l41_41798

theorem sphere_volume (π : ℝ) (r : ℝ):
  4 * π * r^2 = 144 * π →
  (4 / 3) * π * r^3 = 288 * π :=
by
  sorry

end sphere_volume_l41_41798


namespace max_value_is_63_l41_41927

noncomputable def max_value (x y : ℝ) : ℝ :=
  x^2 + 3*x*y + 4*y^2

theorem max_value_is_63 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (cond : x^2 - 3*x*y + 4*y^2 = 9) :
  max_value x y ≤ 63 :=
by
  sorry

end max_value_is_63_l41_41927


namespace rebus_no_solution_l41_41922

open Nat
open DigitFin

theorem rebus_no_solution (K U S Y : Fin 10) (h1 : K ≠ U) (h2 : K ≠ S) (h3 : K ≠ Y) (h4 : U ≠ S) (h5 : U ≠ Y) (h6 : S ≠ Y) :
  let KUSY := K.val * 1000 + U.val * 100 + S.val * 10 + Y.val
  let UKSY := U.val * 1000 + K.val * 100 + S.val * 10 + Y.val
  let UKSUS := U.val * 100000 + K.val * 10000 + S.val * 1000 + U.val * 100 + S.val * 10 + S.val
  KUSY + UKSY ≠ UKSUS := by
sorry

end rebus_no_solution_l41_41922


namespace solve_for_n_l41_41024

theorem solve_for_n (n : ℤ) (h : (1 : ℤ) / (n + 2) + 2 / (n + 2) + n / (n + 2) = 2) : n = 2 :=
sorry

end solve_for_n_l41_41024


namespace M_gt_N_l41_41595

variable (x : ℝ)

def M := x^2 + 4 * x - 2

def N := 6 * x - 5

theorem M_gt_N : M x > N x := sorry

end M_gt_N_l41_41595


namespace num_people_comparison_l41_41568

def num_people_1st_session (a : ℝ) : Prop := a > 0 -- Define the number for first session
def num_people_2nd_session (a : ℝ) : ℝ := 1.1 * a -- Define the number for second session
def num_people_3rd_session (a : ℝ) : ℝ := 0.99 * a -- Define the number for third session

theorem num_people_comparison (a b : ℝ) 
    (h1 : b = 0.99 * a): 
    a > b := 
by 
  -- insert the proof here
  sorry 

end num_people_comparison_l41_41568


namespace henry_total_fee_8_bikes_l41_41030

def paint_fee := 5
def sell_fee := paint_fee + 8
def total_fee_per_bike := paint_fee + sell_fee
def total_fee (bikes : ℕ) := bikes * total_fee_per_bike

theorem henry_total_fee_8_bikes : total_fee 8 = 144 :=
by
  sorry

end henry_total_fee_8_bikes_l41_41030


namespace a_formula_b_formula_T_formula_l41_41644

variable {n : ℕ}

def S (n : ℕ) := 2 * n^2

def a (n : ℕ) : ℕ := 
  if n = 1 then S 1 else S n - S (n - 1)

def b (n : ℕ) : ℕ := 
  if n = 1 then 2 else 2 * (1 / 4 ^ (n - 1))

def c (n : ℕ) : ℕ := (4 * n - 2) / (2 * 4 ^ (n - 1))

def T (n : ℕ) : ℕ := 
  (1 / 9) * ((6 * n - 5) * (4 ^ n) + 5)

theorem a_formula :
  ∀ n, a n = 4 * n - 2 := 
sorry

theorem b_formula :
  ∀ n, b n = 2 / (4 ^ (n - 1)) :=
sorry

theorem T_formula :
  ∀ n, T n = (1 / 9) * ((6 * n - 5) * (4 ^ n) + 5) :=
sorry

end a_formula_b_formula_T_formula_l41_41644


namespace prime_sum_55_l41_41558

theorem prime_sum_55 (p q r s : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (hs : Prime s)
  (hpqrs : p < q ∧ q < r ∧ r < s) 
  (h_eqn : 1 - (1 : ℚ)/p - (1 : ℚ)/q - (1 : ℚ)/r - (1 : ℚ)/s = 1 / (p * q * r * s)) :
  p + q + r + s = 55 := 
sorry

end prime_sum_55_l41_41558


namespace inequality_correct_l41_41477

theorem inequality_correct (a b : ℝ) (h : a - |b| > 0) : a + b > 0 :=
sorry

end inequality_correct_l41_41477


namespace min_value_of_reciprocal_sum_l41_41324

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_ab : a + b = 1) : 
    ∃ c : ℝ, c = 3 + 2 * real.sqrt 2 ∧ ∀ x y : ℝ, (0 < x) → (0 < y) → (x + y = 1) → (1 / x + 2 / y) ≥ c := sorry

end min_value_of_reciprocal_sum_l41_41324


namespace amount_invested_l41_41698

theorem amount_invested (P : ℝ) :
  P * (1.03)^2 - P = 0.08 * P + 6 → P = 314.136 := by
  sorry

end amount_invested_l41_41698


namespace polynomial_expression_value_l41_41594

theorem polynomial_expression_value (a : ℕ → ℤ) (x : ℤ) :
  (x + 2)^9 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 →
  ((a 1 + 3 * a 3 + 5 * a 5 + 7 * a 7 + 9 * a 9)^2 - (2 * a 2 + 4 * a 4 + 6 * a 6 + 8 * a 8)^2) = 3^12 :=
by
  sorry

end polynomial_expression_value_l41_41594


namespace sum_of_positive_odd_divisors_of_90_l41_41954

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l41_41954


namespace tangent_line_to_ellipse_l41_41203

theorem tangent_line_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 1 → x^2 + 4 * y^2 = 1 → (x^2 + 4 * (m * x + 1)^2 = 1)) →
  m^2 = 3 / 4 :=
by
  sorry

end tangent_line_to_ellipse_l41_41203


namespace area_of_right_triangle_l41_41299

theorem area_of_right_triangle (h : ℝ) 
  (a b : ℝ) 
  (h_a_triple : b = 3 * a)
  (h_hypotenuse : h ^ 2 = a ^ 2 + b ^ 2) : 
  (1 / 2) * a * b = (3 * h ^ 2) / 20 :=
by
  sorry

end area_of_right_triangle_l41_41299


namespace triangle_count_l41_41311

def count_triangles (smallest intermediate larger even_larger whole_structure : Nat) : Nat :=
  smallest + intermediate + larger + even_larger + whole_structure

theorem triangle_count :
  count_triangles 2 6 6 6 12 = 32 :=
by
  sorry

end triangle_count_l41_41311


namespace fraction_comparison_l41_41464

theorem fraction_comparison : 
  (1 / (Real.sqrt 2 - 1)) < (Real.sqrt 3 + 1) :=
sorry

end fraction_comparison_l41_41464


namespace polynomial_non_negative_for_all_real_iff_l41_41367

theorem polynomial_non_negative_for_all_real_iff (a : ℝ) :
  (∀ x : ℝ, x^4 + (a - 1) * x^2 + 1 ≥ 0) ↔ a ≥ -1 :=
by sorry

end polynomial_non_negative_for_all_real_iff_l41_41367


namespace haley_marbles_l41_41207

theorem haley_marbles (m : ℕ) (k : ℕ) (h1 : k = 2) (h2 : m = 28) : m / k = 14 :=
by sorry

end haley_marbles_l41_41207


namespace combined_weight_l41_41232

theorem combined_weight (mary_weight : ℝ) (jamison_weight : ℝ) (john_weight : ℝ) :
  mary_weight = 160 ∧ jamison_weight = mary_weight + 20 ∧ john_weight = mary_weight + (0.25 * mary_weight) →
  john_weight + mary_weight + jamison_weight = 540 :=
by
  intros h
  obtain ⟨hm, hj, hj'⟩ := h
  rw [hm, hj, hj']
  norm_num
  sorry

end combined_weight_l41_41232


namespace find_a_l41_41722

theorem find_a :
  ∀ (a : ℝ), 
  (∀ x : ℝ, 2 * x^2 - 2016 * x + 2016^2 - 2016 * a - 1 = a^2) → 
  (∃ x1 x2 : ℝ, 2 * x1^2 - 2016 * x1 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 2 * x2^2 - 2016 * x2 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 x1 < a ∧ a < x2) → 
  2015 < a ∧ a < 2017 :=
by sorry

end find_a_l41_41722


namespace max_third_term_is_16_l41_41121

-- Define the arithmetic sequence conditions
def arithmetic_seq (a d : ℕ) : list ℕ := [a, a + d, a + 2 * d, a + 3 * d]

-- Define the sum condition
def sum_of_sequence_is_50 (a d : ℕ) : Prop :=
  (a + a + d + a + 2 * d + a + 3 * d) = 50

-- Define the third term of the sequence
def third_term (a d : ℕ) : ℕ := a + 2 * d

-- Prove that the greatest possible third term is 16
theorem max_third_term_is_16 : ∃ (a d : ℕ), sum_of_sequence_is_50 a d ∧ third_term a d = 16 :=
by
  sorry

end max_third_term_is_16_l41_41121


namespace remainder_is_15_l41_41931

-- Definitions based on conditions
def S : ℕ := 476
def L : ℕ := S + 2395
def quotient : ℕ := 6

-- The proof statement
theorem remainder_is_15 : ∃ R : ℕ, L = quotient * S + R ∧ R = 15 := by
  sorry

end remainder_is_15_l41_41931


namespace smallest_four_digit_divisible_43_l41_41269

theorem smallest_four_digit_divisible_43 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 ∧ n = 1032 :=
by
  sorry

end smallest_four_digit_divisible_43_l41_41269


namespace find_y_l41_41615

theorem find_y (y : ℕ) (hy_mult_of_7 : ∃ k, y = 7 * k) (hy_pos : 0 < y) (hy_square : y^2 > 225) (hy_upper_bound : y < 30) : y = 21 :=
sorry

end find_y_l41_41615


namespace jeans_price_increase_l41_41150

theorem jeans_price_increase (manufacturing_cost customer_price : ℝ) 
  (h1 : customer_price = 1.40 * (1.40 * manufacturing_cost))
  : (customer_price - manufacturing_cost) / manufacturing_cost * 100 = 96 :=
by sorry

end jeans_price_increase_l41_41150


namespace sum_of_roots_gt_two_l41_41488

noncomputable def f : ℝ → ℝ := λ x => Real.log x - x + 1

theorem sum_of_roots_gt_two (m : ℝ) (x1 x2 : ℝ) (hx1 : f x1 = m) (hx2 : f x2 = m) (hne : x1 ≠ x2) : x1 + x2 > 2 := by
  sorry

end sum_of_roots_gt_two_l41_41488


namespace no_graph_for_equation_l41_41534

theorem no_graph_for_equation (x y : ℝ) : 
  ¬ ∃ (x y : ℝ), x^2 + y^2 + 2*x + 4*y + 6 = 0 := 
by 
  sorry

end no_graph_for_equation_l41_41534


namespace office_speed_l41_41985

variable (d v : ℝ)

theorem office_speed (h1 : v > 0) (h2 : ∀ t : ℕ, t = 30) (h3 : (2 * d) / (d / v + d / 30) = 24) : v = 20 := 
sorry

end office_speed_l41_41985


namespace binom_8_5_eq_56_l41_41014

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l41_41014


namespace quadrilateral_inscribed_circumscribed_l41_41280

theorem quadrilateral_inscribed_circumscribed 
  (r R d : ℝ) --Given variables with their types
  (K O : Type) (radius_K : K → ℝ) (radius_O : O → ℝ) (dist : (K × O) → ℝ)  -- Defining circles properties
  (K_inside_O : ∀ p : K × O, radius_K p.fst < radius_O p.snd) 
  (dist_centers : ∀ p : K × O, dist p = d) -- Distance between the centers
  : 
  (1 / (R + d)^2) + (1 / (R - d)^2) = (1 / r^2) := 
by 
  sorry

end quadrilateral_inscribed_circumscribed_l41_41280


namespace triangle_inscribed_circle_area_l41_41453

noncomputable def circle_radius (circumference : ℝ) : ℝ :=
  circumference / (2 * Real.pi)

noncomputable def triangle_area (r : ℝ) : ℝ :=
  (1 / 2) * r^2 * (Real.sin (Real.pi / 2) + Real.sin (2 * Real.pi / 3) + Real.sin (5 * Real.pi / 6))

theorem triangle_inscribed_circle_area (a b c : ℝ) (h : a + b + c = 24) :
  ∀ (r : ℝ) (h_r : r = circle_radius 24),
  triangle_area r = 72 / Real.pi^2 * (Real.sqrt 3 + 1) :=
by
  intro r h_r
  rw [h_r, circle_radius, triangle_area]
  sorry

end triangle_inscribed_circle_area_l41_41453


namespace binom_8_5_eq_56_l41_41006

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := 
by
sorry

end binom_8_5_eq_56_l41_41006


namespace probability_of_rolling_5_is_1_over_9_l41_41279

def num_sides_dice : ℕ := 6

def favorable_combinations : List (ℕ × ℕ) :=
[(1, 4), (2, 3), (3, 2), (4, 1)]

def total_combinations : ℕ :=
num_sides_dice * num_sides_dice

def favorable_count : ℕ := favorable_combinations.length

def probability_rolling_5 : ℚ :=
favorable_count / total_combinations

theorem probability_of_rolling_5_is_1_over_9 :
  probability_rolling_5 = 1 / 9 :=
sorry

end probability_of_rolling_5_is_1_over_9_l41_41279


namespace partial_fraction_decomposition_l41_41252

theorem partial_fraction_decomposition :
  ∃ (a b c : ℤ), (0 ≤ a ∧ a < 5) ∧ (0 ≤ b ∧ b < 13) ∧ (1 / 2015 = a / 5 + b / 13 + c / 31) ∧ (a + b = 14) :=
sorry

end partial_fraction_decomposition_l41_41252


namespace non_attacking_knight_count_l41_41495

def knight_moves (pos : ℕ × ℕ) : Finset (ℕ × ℕ) :=
  {(pos.1 + 2, pos.2 + 1), (pos.1 + 2, pos.2 - 1),
   (pos.1 - 2, pos.2 + 1), (pos.1 - 2, pos.2 - 1),
   (pos.1 + 1, pos.2 + 2), (pos.1 + 1, pos.2 - 2),
   (pos.1 - 1, pos.2 + 2), (pos.1 - 1, pos.2 - 2)}.filter
  (λ p, 1 ≤ p.1 ∧ p.1 ≤ 8 ∧ 1 ≤ p.2 ∧ p.2 ≤ 8)

def knight_attacks : Finset (ℕ × ℕ) → Finset (ℕ × ℕ) :=
  Finset.bUnion knight_moves

def non_attacking_knight_placements : Finset (ℕ × ℕ) × Finset (ℕ × ℕ) :=
  (Finset.product (Finset.range 8.succ).product (Finset.range 8.succ))

theorem non_attacking_knight_count :
  non_attacking_knight_placements.card = 3696 :=
by sorry

end non_attacking_knight_count_l41_41495


namespace find_m_n_l41_41484

theorem find_m_n 
  (a b c d m n : ℕ) 
  (h₁ : a^2 + b^2 + c^2 + d^2 = 1989)
  (h₂ : a + b + c + d = m^2)
  (h₃ : a = max (max a b) (max c d) ∨ b = max (max a b) (max c d) ∨ c = max (max a b) (max c d) ∨ d = max (max a b) (max c d))
  (h₄ : exists k, k^2 = max (max a b) (max c d))
  : m = 9 ∧ n = 6 :=
by
  -- Proof omitted
  sorry

end find_m_n_l41_41484


namespace number_of_lightsabers_in_order_l41_41548

-- Let's define the given conditions
def metal_arcs_per_lightsaber : ℕ := 2
def cost_per_metal_arc : ℕ := 400
def apparatus_production_rate : ℕ := 20 -- lightsabers per hour
def combined_app_expense_rate : ℕ := 300 -- units per hour
def total_order_cost : ℕ := 65200
def lightsaber_cost : ℕ := metal_arcs_per_lightsaber * cost_per_metal_arc + (combined_app_expense_rate / apparatus_production_rate)

-- Define the main theorem to prove
theorem number_of_lightsabers_in_order : 
  (total_order_cost / lightsaber_cost) = 80 :=
by
  sorry

end number_of_lightsabers_in_order_l41_41548


namespace total_number_of_glasses_l41_41681

open scoped Nat

theorem total_number_of_glasses (x y : ℕ) (h1 : y = x + 16) (h2 : (12 * x + 16 * y) / (x + y) = 15) : 12 * x + 16 * y = 480 := by
  sorry

end total_number_of_glasses_l41_41681


namespace reciprocal_inequality_l41_41354

variable (a b : ℝ)

theorem reciprocal_inequality (ha : a < 0) (hb : b > 0) : (1 / a) < (1 / b) := sorry

end reciprocal_inequality_l41_41354


namespace triangle_bc_length_l41_41222

theorem triangle_bc_length (A B C X : Type)
  (AB AC : ℕ)
  (hAB : AB = 86)
  (hAC : AC = 97)
  (circle_eq : ∀ {r : ℕ}, r = AB → circle_centered_at_A_intersects_BC_two_points B X)
  (integer_lengths : ∃ (BX CX : ℕ), ) :
  BC = 61 :=
by
  sorry

end triangle_bc_length_l41_41222


namespace max_geometric_sequence_terms_l41_41750

theorem max_geometric_sequence_terms (a r : ℝ) (n : ℕ) (h_r : r > 1) 
    (h_seq : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → 100 ≤ a * r^(k-1) ∧ a * r^(k-1) ≤ 1000) :
  n ≤ 6 :=
sorry

end max_geometric_sequence_terms_l41_41750


namespace initial_amount_l41_41167

theorem initial_amount (P : ℝ) (h1 : ∀ x : ℝ, x * (9 / 8) * (9 / 8) = 81000) : P = 64000 :=
sorry

end initial_amount_l41_41167


namespace phantom_needs_more_money_l41_41402

def amount_phantom_has : ℤ := 50
def cost_black : ℤ := 11
def count_black : ℕ := 2
def cost_red : ℤ := 15
def count_red : ℕ := 3
def cost_yellow : ℤ := 13
def count_yellow : ℕ := 2

def total_cost : ℤ := cost_black * count_black + cost_red * count_red + cost_yellow * count_yellow
def additional_amount_needed : ℤ := total_cost - amount_phantom_has

theorem phantom_needs_more_money : additional_amount_needed = 43 := by
  sorry

end phantom_needs_more_money_l41_41402


namespace phantom_needs_more_money_l41_41400

variable (given_money black_ink_price red_ink_price yellow_ink_price total_black_inks total_red_inks total_yellow_inks : ℕ)

def total_cost (total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price : ℕ) : ℕ :=
  total_black_inks * black_ink_price + total_red_inks * red_ink_price + total_yellow_inks * yellow_ink_price

theorem phantom_needs_more_money
  (h_given : given_money = 50)
  (h_black : black_ink_price = 11)
  (h_red : red_ink_price = 15)
  (h_yellow : yellow_ink_price = 13)
  (h_total_black : total_black_inks = 2)
  (h_total_red : total_red_inks = 3)
  (h_total_yellow : total_yellow_inks = 2) :
  given_money < total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price →
  total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price - given_money = 43 := by
  sorry

end phantom_needs_more_money_l41_41400


namespace problem_statement_l41_41482

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def T : Set ℝ := {x | x < 2}

def set_otimes (A B : Set ℝ) : Set ℝ := {x | x ∈ (A ∪ B) ∧ x ∉ (A ∩ B)}

theorem problem_statement : set_otimes M T = {x | x < -1 ∨ (2 ≤ x ∧ x ≤ 4)} :=
by sorry

end problem_statement_l41_41482


namespace average_age_is_correct_l41_41451

-- Define the conditions
def num_men : ℕ := 6
def num_women : ℕ := 9
def average_age_men : ℕ := 57
def average_age_women : ℕ := 52
def total_age_men : ℕ := num_men * average_age_men
def total_age_women : ℕ := num_women * average_age_women
def total_age : ℕ := total_age_men + total_age_women
def total_people : ℕ := num_men + num_women
def average_age_group : ℕ := total_age / total_people

-- The proof will require showing average_age_group is 54, left as sorry.
theorem average_age_is_correct : average_age_group = 54 := sorry

end average_age_is_correct_l41_41451


namespace cars_produced_in_europe_l41_41148

theorem cars_produced_in_europe (total_cars : ℕ) (cars_in_north_america : ℕ) (cars_in_europe : ℕ) :
  total_cars = 6755 → cars_in_north_america = 3884 → cars_in_europe = total_cars - cars_in_north_america → cars_in_europe = 2871 :=
by
  -- necessary calculations and logical steps
  sorry

end cars_produced_in_europe_l41_41148


namespace total_stoppage_time_per_hour_l41_41130

variables (speed_ex_stoppages_1 speed_in_stoppages_1 : ℕ)
variables (speed_ex_stoppages_2 speed_in_stoppages_2 : ℕ)
variables (speed_ex_stoppages_3 speed_in_stoppages_3 : ℕ)

-- Definitions of the speeds given in the problem's conditions.
def speed_bus_1_ex_stoppages := 54
def speed_bus_1_in_stoppages := 36
def speed_bus_2_ex_stoppages := 60
def speed_bus_2_in_stoppages := 40
def speed_bus_3_ex_stoppages := 72
def speed_bus_3_in_stoppages := 48

-- The main theorem to be proved.
theorem total_stoppage_time_per_hour :
  ((1 - speed_bus_1_in_stoppages / speed_bus_1_ex_stoppages : ℚ)
   + (1 - speed_bus_2_in_stoppages / speed_bus_2_ex_stoppages : ℚ)
   + (1 - speed_bus_3_in_stoppages / speed_bus_3_ex_stoppages : ℚ)) = 1 := by
  sorry

end total_stoppage_time_per_hour_l41_41130


namespace range_g_l41_41909

theorem range_g (f : ℝ → ℝ) (g : ℝ → ℝ) (b : ℝ)
  (h1 : ∀ x, f x = 3 ^ x + b)
  (h2 : b < -1) :
  set.range g = set.Ioo 0 (2 / 9) :=
sorry

end range_g_l41_41909


namespace johns_total_working_hours_l41_41514

theorem johns_total_working_hours (d h t : Nat) (h_d : d = 5) (h_h : h = 8) : t = d * h := by
  rewrite [h_d, h_h]
  sorry

end johns_total_working_hours_l41_41514


namespace freezer_temperature_is_minus_12_l41_41625

theorem freezer_temperature_is_minus_12 (refrigeration_temp freezer_temp : ℤ) (h1 : refrigeration_temp = 5) (h2 : freezer_temp = -12) : freezer_temp = -12 :=
by sorry

end freezer_temperature_is_minus_12_l41_41625


namespace motorcycle_materials_cost_l41_41446

theorem motorcycle_materials_cost 
  (car_material_cost : ℕ) (cars_per_month : ℕ) (car_sale_price : ℕ)
  (motorcycles_per_month : ℕ) (motorcycle_sale_price : ℕ)
  (additional_profit : ℕ) :
  car_material_cost = 100 →
  cars_per_month = 4 →
  car_sale_price = 50 →
  motorcycles_per_month = 8 →
  motorcycle_sale_price = 50 →
  additional_profit = 50 →
  car_material_cost + additional_profit = 250 := by
  sorry

end motorcycle_materials_cost_l41_41446


namespace log_problem_l41_41099

noncomputable def log_expression : ℝ := log 2 (1/4) + log 2 32

theorem log_problem :
  log_expression = 3 := by
  sorry

end log_problem_l41_41099


namespace andrey_stamps_count_l41_41771

theorem andrey_stamps_count (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x ∧ x ≤ 300) → x = 208 := 
by 
  sorry

end andrey_stamps_count_l41_41771


namespace candles_number_problem_l41_41309

theorem candles_number_problem :
  ∃ c : ℕ, (choose c 2) * (choose 9 8) = 54 ∧ c = 4 :=
by {
  have h_comb_candles := @choose_eq_factorial_div_factorial (λ c, choose c 2),
  have h_comb_flowers := @choose_eq_factorial_div_factorial (λ _, choose 9 8),
  have h_comb_flower := 9,
  use 4,
  split,
  { rw [h_comb_candles, ←mul_eq_mul_left, h_comb_flowers, ←mul_assoc, choose_one_mul],
    linarith },
  { refl }
}

end candles_number_problem_l41_41309


namespace union_of_sets_l41_41874

theorem union_of_sets (A B : Set ℕ) (hA : A = {1, 2, 6}) (hB : B = {2, 3, 6}) :
  A ∪ B = {1, 2, 3, 6} :=
by
  rw [hA, hB]
  ext x
  simp [Set.union]
  sorry

end union_of_sets_l41_41874


namespace balls_sum_divisibility_l41_41700

open NatRat

def probability_sum_divisible_by_3 (n : ℕ) : ℚ :=
  let xs := Finset.range n \ Finset.single 0
  let divisible_by_3 := xs.filter (λ x, x % 3 = 0)
  let mod1 := xs.filter (λ x, x % 3 = 1)
  let mod2 := xs.filter (λ x, x % 3 = 2)
  let favorable := (Finset.card divisible_by_3.choose₂.card) + (Finset.card mod1 * Finset.card mod2)
  let total := xs.choose₂.card
  favorable / total

theorem balls_sum_divisibility :
  probability_sum_divisible_by_3 20 = 32 / 95 :=
sorry

end balls_sum_divisibility_l41_41700


namespace original_proposition_converse_negation_contrapositive_l41_41044

variable {a b : ℝ}

-- Original Proposition: If \( x^2 + ax + b \leq 0 \) has a non-empty solution set, then \( a^2 - 4b \geq 0 \)
theorem original_proposition (h : ∃ x : ℝ, x^2 + a * x + b ≤ 0) : a^2 - 4 * b ≥ 0 := sorry

-- Converse: If \( a^2 - 4b \geq 0 \), then \( x^2 + ax + b \leq 0 \) has a non-empty solution set
theorem converse (h : a^2 - 4 * b ≥ 0) : ∃ x : ℝ, x^2 + a * x + b ≤ 0 := sorry

-- Negation: If \( x^2 + ax + b \leq 0 \) does not have a non-empty solution set, then \( a^2 - 4b < 0 \)
theorem negation (h : ¬ ∃ x : ℝ, x^2 + a * x + b ≤ 0) : a^2 - 4 * b < 0 := sorry

-- Contrapositive: If \( a^2 - 4b < 0 \), then \( x^2 + ax + b \leq 0 \) does not have a non-empty solution set
theorem contrapositive (h : a^2 - 4 * b < 0) : ¬ ∃ x : ℝ, x^2 + a * x + b ≤ 0 := sorry

end original_proposition_converse_negation_contrapositive_l41_41044


namespace domain_of_f_x_plus_2_l41_41885

theorem domain_of_f_x_plus_2 (f : ℝ → ℝ) (dom_f_x_minus_1 : ∀ x, 1 ≤ x ∧ x ≤ 2 → 0 ≤ x-1 ∧ x-1 ≤ 1) :
  ∀ y, 0 ≤ y ∧ y ≤ 1 ↔ -2 ≤ y-2 ∧ y-2 ≤ -1 :=
by
  sorry

end domain_of_f_x_plus_2_l41_41885


namespace find_a_l41_41734

noncomputable def curve1 : ℝ → ℝ := λ x, x + Real.log x
noncomputable def curve2 (a : ℝ) : ℝ → ℝ := λ x, a * x^2 + (2 * a + 3) * x + 1

theorem find_a :
  ∀ a : ℝ,
  (∃ x : ℝ, curve1 x = (2 * x - 1) ∧ curve1 x = curve2 a x) →
  a = 0 ∨ a = 1 / 2 :=
by
  sorry

end find_a_l41_41734


namespace frame_painting_ratio_l41_41152

theorem frame_painting_ratio :
  ∃ (x : ℝ), (20 + 2 * x) * (30 + 6 * x) = 1800 → 1 = 2 * (20 + 2 * x) / (30 + 6 * x) :=
by
  sorry

end frame_painting_ratio_l41_41152


namespace circle_condition_l41_41790

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4 * x - 2 * y + 5 * m = 0) ↔ m < 1 := by
  sorry

end circle_condition_l41_41790


namespace number_of_balls_greater_l41_41147

theorem number_of_balls_greater (n x : ℤ) (h1 : n = 25) (h2 : n - x = 30 - n) : x = 20 := by
  sorry

end number_of_balls_greater_l41_41147


namespace range_of_m_l41_41886

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ 4) → 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → 
  -3 ≤ m ∧ m ≤ 5 := 
by 
  sorry

end range_of_m_l41_41886


namespace find_bottle_price_l41_41915

theorem find_bottle_price 
  (x : ℝ) 
  (promotion_free_bottles : ℝ := 3)
  (discount_per_bottle : ℝ := 0.6)
  (box_price : ℝ := 26)
  (box_bottles : ℝ := 4) :
  ∃ x : ℝ, (box_price / (x - discount_per_bottle)) - (box_price / x) = promotion_free_bottles :=
sorry

end find_bottle_price_l41_41915


namespace sumOddDivisorsOf90_l41_41953

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l41_41953


namespace donut_holes_covered_by_lidia_l41_41999

noncomputable def surface_area (r: ℕ) : ℕ := 4 * Nat.pi * (r * r)

def radius_lidia := 5
def radius_marco := 7
def radius_priya := 9

def surface_area_lidia := surface_area radius_lidia
def surface_area_marco := surface_area radius_marco
def surface_area_priya := surface_area radius_priya

open Nat

theorem donut_holes_covered_by_lidia :
  let lcm_area := Nat.lcm (Nat.lcm surface_area_lidia surface_area_marco) surface_area_priya
  ∃ n : ℕ, lcm_area / surface_area_lidia = n ∧ n = 3528 :=
  by
    sorry

end donut_holes_covered_by_lidia_l41_41999


namespace probabilities_inequalities_l41_41070

variables (M N : Prop) (P : Prop → ℝ)

axiom P_pos_M : P M > 0
axiom P_pos_N : P N > 0
axiom P_cond_N_M : P (N ∧ M) / P M > P (N ∧ ¬M) / P (¬M)

theorem probabilities_inequalities :
    (P (N ∧ M) / P M > P (N ∧ ¬M) / P (¬M)) ∧
    (P (N ∧ M) > P N * P M) ∧
    (P (M ∧ N) / P N > P (M ∧ ¬N) / P (¬N)) :=
by
    sorry

end probabilities_inequalities_l41_41070


namespace min_cost_yogurt_l41_41466

theorem min_cost_yogurt (cost_per_box : ℕ) (boxes : ℕ) (promotion : ℕ → ℕ) (cost : ℕ) :
  cost_per_box = 4 → 
  boxes = 10 → 
  promotion 3 = 2 → 
  cost = 28 := 
by {
  -- The proof will go here
  sorry
}

end min_cost_yogurt_l41_41466


namespace emily_lives_lost_l41_41467

variable (L : ℕ)
variable (initial_lives : ℕ) (extra_lives : ℕ) (final_lives : ℕ)

-- Conditions based on the problem statement
axiom initial_lives_def : initial_lives = 42
axiom extra_lives_def : extra_lives = 24
axiom final_lives_def : final_lives = 41

-- Mathematically equivalent proof statement
theorem emily_lives_lost : initial_lives - L + extra_lives = final_lives → L = 25 := by
  sorry

end emily_lives_lost_l41_41467


namespace light_ray_total_distance_l41_41603

theorem light_ray_total_distance 
  (M : ℝ × ℝ) (N : ℝ × ℝ)
  (M_eq : M = (2, 1))
  (N_eq : N = (4, 5)) :
  dist M N = 2 * Real.sqrt 10 := 
sorry

end light_ray_total_distance_l41_41603


namespace find_ABC_l41_41891

-- Define the angles as real numbers in degrees
variables (ABC CBD DBC DBE ABE : ℝ)

-- Assert the given conditions
axiom horz_angle: CBD = 90
axiom DBC_ABC_relation : DBC = ABC + 30
axiom straight_angle: DBE = 180
axiom measure_abe: ABE = 145

-- State the proof problem
theorem find_ABC : ABC = 30 :=
by
  -- Include all steps required to derive the conclusion in the proof
  sorry

end find_ABC_l41_41891


namespace tan_alpha_minus_pi_over_4_l41_41183

theorem tan_alpha_minus_pi_over_4 (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : Real.sin α = 3 / 5) : Real.tan (α - π / 4) = -1 / 7 ∨ Real.tan (α - π / 4) = -7 := 
sorry

end tan_alpha_minus_pi_over_4_l41_41183


namespace count_two_digit_numbers_l41_41353

theorem count_two_digit_numbers : (99 - 10 + 1) = 90 := by
  sorry

end count_two_digit_numbers_l41_41353


namespace expected_digits_of_fair_icosahedral_die_l41_41776

noncomputable def expected_num_of_digits : ℚ :=
  (9 / 20) * 1 + (11 / 20) * 2

theorem expected_digits_of_fair_icosahedral_die :
  expected_num_of_digits = 1.55 := by
  sorry

end expected_digits_of_fair_icosahedral_die_l41_41776


namespace minimum_students_ans_q1_correctly_l41_41208

variable (Total Students Q1 Q2 Q1_and_Q2 : ℕ)
variable (did_not_take_test: Student → Bool)

-- Given Conditions
def total_students := 40
def students_ans_q2_correctly := 29
def students_not_taken_test := 10
def students_ans_both_correctly := 29

theorem minimum_students_ans_q1_correctly (H1: Q2 - students_not_taken_test == 30)
                                           (H2: Q1_and_Q2 + students_not_taken_test == total_students)
                                           (H3: Q1_and_Q2 == students_ans_q2_correctly):
  Q1 ≥ 29 := by
  sorry

end minimum_students_ans_q1_correctly_l41_41208


namespace total_initial_passengers_l41_41287

theorem total_initial_passengers (M W : ℕ) 
  (h1 : W = M / 3) 
  (h2 : M - 24 = W + 12) : 
  M + W = 72 :=
sorry

end total_initial_passengers_l41_41287


namespace prime_solution_exists_l41_41479

theorem prime_solution_exists (p : ℕ) (hp : Nat.Prime p) : ∃ x y z : ℤ, x^2 + y^2 + (p:ℤ) * z = 2003 := 
by 
  sorry

end prime_solution_exists_l41_41479


namespace product_mnp_l41_41847

theorem product_mnp (m n p : ℕ) (b x z c : ℂ) (h1 : b^8 * x * z - b^7 * z - b^6 * x = b^5 * (c^5 - 1)) 
  (h2 : (b^m * x - b^n) * (b^p * z - b^3) = b^5 * c^5) : m * n * p = 30 :=
sorry

end product_mnp_l41_41847


namespace find_w_value_l41_41864

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_w_value
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : sqrt x / sqrt y - sqrt y / sqrt x = 7 / 12)
  (h2 : x - y = 7) :
  x + y = 25 := 
by
  sorry

end find_w_value_l41_41864


namespace sin_gt_cos_lt_nec_suff_l41_41634

-- Define the triangle and the angles
variables {A B C : ℝ}
variables (t : triangle A B C)

-- Define conditions in the triangle: sum of angles is 180 degrees
axiom angle_sum : A + B + C = 180

-- Define sin and cos using the sides of the triangle
noncomputable def sin_A (A : ℝ) : ℝ := sorry -- placeholder for actual definition
noncomputable def sin_B (B : ℝ) : ℝ := sorry
noncomputable def cos_A (A : ℝ) : ℝ := sorry
noncomputable def cos_B (B : ℝ) : ℝ := sorry

-- The proposition to prove
theorem sin_gt_cos_lt_nec_suff {A B : ℝ} (h1 : sin_A A > sin_B B) :
  cos_A A < cos_B B ↔ sin_A A > sin_B B := sorry

end sin_gt_cos_lt_nec_suff_l41_41634


namespace binom_eight_five_l41_41019

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l41_41019


namespace binom_eight_five_l41_41020

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l41_41020


namespace find_c_interval_l41_41717

theorem find_c_interval :
  {c : ℝ | (4 * c / 3 ≤ 8 + 4 * c) ∧ (8 + 4 * c < -3 * (1 + c))} = set.Icc (-3:ℝ) (-11 / 7) :=
by {
  -- Proof would go here, but it's omitted as instructed
  sorry
}

end find_c_interval_l41_41717


namespace investment_problem_l41_41987

theorem investment_problem :
  ∃ (S G : ℝ), S + G = 10000 ∧ 0.06 * G = 0.05 * S + 160 ∧ S = 4000 :=
by
  sorry

end investment_problem_l41_41987


namespace angle_of_parallel_l41_41060

-- Define a line and a plane
variable {L : Type} (l : L)
variable {P : Type} (β : P)

-- Define the parallel condition
def is_parallel (l : L) (β : P) : Prop := sorry

-- Define the angle function between a line and a plane
def angle (l : L) (β : P) : ℝ := sorry

-- The theorem stating that if l is parallel to β, then the angle is 0
theorem angle_of_parallel (h : is_parallel l β) : angle l β = 0 := sorry

end angle_of_parallel_l41_41060


namespace anya_more_erasers_l41_41995

theorem anya_more_erasers (anya_erasers andrea_erasers : ℕ)
  (h1 : anya_erasers = 4 * andrea_erasers)
  (h2 : andrea_erasers = 4) :
  anya_erasers - andrea_erasers = 12 := by
  sorry

end anya_more_erasers_l41_41995


namespace solve_equation_l41_41662

theorem solve_equation (m n : ℝ) (h₀ : m ≠ 0) (h₁ : n ≠ 0) (h₂ : m ≠ n) :
  ∀ x : ℝ, ((x + m)^2 - 3 * (x + n)^2 = m^2 - 3 * n^2) ↔ (x = 0 ∨ x = m - 3 * n) :=
by
  sorry

end solve_equation_l41_41662


namespace exponents_product_as_cube_l41_41579

theorem exponents_product_as_cube :
  (3^12 * 3^3) = 243^3 :=
sorry

end exponents_product_as_cube_l41_41579


namespace smallest_solution_l41_41592

theorem smallest_solution (x : ℝ) : 
  (∃ x, (3 * x / (x - 3)) + ((3 * x^2 - 27) / x) = 15 ∧ ∀ y, (3 * y / (y - 3)) + ((3 * y^2 - 27) / y) = 15 → y ≥ x) → 
  x = -1 := 
by
  sorry

end smallest_solution_l41_41592


namespace trapezium_other_side_l41_41588

theorem trapezium_other_side (x : ℝ) :
  1/2 * (20 + x) * 10 = 150 → x = 10 :=
by
  sorry

end trapezium_other_side_l41_41588


namespace eval_expr_l41_41517

namespace ProofProblem

variables (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d = a + b + c)

theorem eval_expr :
  d = a + b + c →
  (a^3 + b^3 + c^3 - 3 * a * b * c) / (a * b * c) = (d * (a^2 + b^2 + c^2 - a * b - a * c - b * c)) / (a * b * c) :=
by
  intros hd
  sorry

end ProofProblem

end eval_expr_l41_41517


namespace first_group_men_count_l41_41499

/-- Given that 10 men can complete a piece of work in 90 hours,
prove that the number of men M in the first group who can complete
the same piece of work in 25 hours is 36. -/
theorem first_group_men_count (M : ℕ) (h : (10 * 90 = 25 * M)) : M = 36 :=
by
  sorry

end first_group_men_count_l41_41499


namespace hardest_vs_least_worked_hours_difference_l41_41683

-- Let x be the scaling factor for the ratio
-- The times worked are 2x, 3x, and 4x

def project_time_difference (x : ℕ) : Prop :=
  let time1 := 2 * x
  let time2 := 3 * x
  let time3 := 4 * x
  (time1 + time2 + time3 = 90) ∧ ((4 * x - 2 * x) = 20)

theorem hardest_vs_least_worked_hours_difference :
  ∃ x : ℕ, project_time_difference x :=
by
  sorry

end hardest_vs_least_worked_hours_difference_l41_41683


namespace slope_of_line_l41_41936

theorem slope_of_line : ∀ (x y : ℝ), 2 * x - 4 * y + 7 = 0 → (y = (1/2) * x - 7 / 4) :=
by
  intro x y h
  -- This would typically involve rearranging the given equation to the slope-intercept form
  -- but as we are focusing on creating the statement, we insert sorry to skip the proof
  sorry

end slope_of_line_l41_41936


namespace new_cylinder_volume_l41_41419

theorem new_cylinder_volume (r h : ℝ) (π_ne_zero : 0 < π) (original_volume : π * r^2 * h = 10) : 
  π * (3 * r)^2 * (2 * h) = 180 :=
by
  sorry

end new_cylinder_volume_l41_41419


namespace line_increase_is_110_l41_41112

noncomputable def original_lines (increased_lines : ℕ) (percentage_increase : ℚ) : ℚ :=
  increased_lines / (1 + percentage_increase)

theorem line_increase_is_110
  (L' : ℕ)
  (percentage_increase : ℚ)
  (hL' : L' = 240)
  (hp : percentage_increase = 0.8461538461538461) :
  L' - original_lines L' percentage_increase = 110 :=
by
  sorry

end line_increase_is_110_l41_41112


namespace functions_increasing_in_interval_l41_41873

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin x * Real.cos x

theorem functions_increasing_in_interval :
  ∀ x, -Real.pi / 4 < x → x < Real.pi / 4 →
  (f x < f (x + 1e-6)) ∧ (g x < g (x + 1e-6)) :=
sorry

end functions_increasing_in_interval_l41_41873


namespace sample_size_is_150_l41_41449

theorem sample_size_is_150 
  (classes : ℕ) (students_per_class : ℕ) (selected_students : ℕ)
  (h1 : classes = 40) (h2 : students_per_class = 50) (h3 : selected_students = 150)
  : selected_students = 150 :=
sorry

end sample_size_is_150_l41_41449


namespace monthly_savings_correct_l41_41090

-- Define the gross salaries before any deductions
def ivan_salary_gross : ℝ := 55000
def vasilisa_salary_gross : ℝ := 45000
def vasilisa_mother_salary_gross : ℝ := 18000
def vasilisa_father_salary_gross : ℝ := 20000
def son_scholarship_state : ℝ := 3000
def son_scholarship_non_state_gross : ℝ := 15000

-- Tax rate definition
def tax_rate : ℝ := 0.13

-- Net income calculations using the tax rate
def net_income (gross_income : ℝ) : ℝ := gross_income * (1 - tax_rate)

def ivan_salary_net : ℝ := net_income ivan_salary_gross
def vasilisa_salary_net : ℝ := net_income vasilisa_salary_gross
def vasilisa_mother_salary_net : ℝ := net_income vasilisa_mother_salary_gross
def vasilisa_father_salary_net : ℝ := net_income vasilisa_father_salary_gross
def son_scholarship_non_state_net : ℝ := net_income son_scholarship_non_state_gross

-- Monthly expenses total
def monthly_expenses : ℝ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

-- Net incomes for different periods
def total_net_income_before_01_05_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + vasilisa_mother_salary_net + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_01_05_2018_to_31_08_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_from_01_09_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + (son_scholarship_state + son_scholarship_non_state_net)

-- Savings calculations for different periods
def monthly_savings_before_01_05_2018 : ℝ :=
  total_net_income_before_01_05_2018 - monthly_expenses

def monthly_savings_01_05_2018_to_31_08_2018 : ℝ :=
  total_net_income_01_05_2018_to_31_08_2018 - monthly_expenses

def monthly_savings_from_01_09_2018 : ℝ :=
  total_net_income_from_01_09_2018 - monthly_expenses

-- Theorem to be proved
theorem monthly_savings_correct :
  monthly_savings_before_01_05_2018 = 49060 ∧
  monthly_savings_01_05_2018_to_31_08_2018 = 43400 ∧
  monthly_savings_from_01_09_2018 = 56450 :=
by
  sorry

end monthly_savings_correct_l41_41090


namespace base7_65432_to_dec_is_16340_l41_41268

def base7_to_dec (n : ℕ) : ℕ :=
  6 * 7^4 + 5 * 7^3 + 4 * 7^2 + 3 * 7^1 + 2 * 7^0

theorem base7_65432_to_dec_is_16340 : base7_to_dec 65432 = 16340 :=
by
  sorry

end base7_65432_to_dec_is_16340_l41_41268


namespace negation_of_existential_square_inequality_l41_41417

theorem negation_of_existential_square_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) :=
by
  sorry

end negation_of_existential_square_inequality_l41_41417


namespace average_remaining_five_l41_41616

theorem average_remaining_five (S S4 S5 : ℕ) 
  (h1 : S = 18 * 9) 
  (h2 : S4 = 8 * 4) 
  (h3 : S5 = S - S4) 
  (h4 : S5 / 5 = 26) : 
  average_of_remaining_5 = 26 :=
by 
  sorry


end average_remaining_five_l41_41616


namespace compare_trig_functions_l41_41478

theorem compare_trig_functions :
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c :=
by
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  sorry

end compare_trig_functions_l41_41478


namespace no_integer_solution_for_Q_square_l41_41581

def Q (x : ℤ) : ℤ := x^4 + 5*x^3 + 10*x^2 + 5*x + 56

theorem no_integer_solution_for_Q_square :
  ∀ x : ℤ, ∃ k : ℤ, Q x = k^2 → false :=
by
  sorry

end no_integer_solution_for_Q_square_l41_41581


namespace inequality_example_l41_41098

theorem inequality_example (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
sorry

end inequality_example_l41_41098


namespace prove_ineq_l41_41023

-- Define the quadratic equation
def quadratic_eqn (a b x : ℝ) : Prop :=
  3 * x^2 + 3 * (a + b) * x + 4 * a * b = 0

-- Define the root relation
def root_relation (x1 x2 : ℝ) : Prop :=
  x1 * (x1 + 1) + x2 * (x2 + 1) = (x1 + 1) * (x2 + 1)

-- State the theorem
theorem prove_ineq (a b : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_eqn a b x1 ∧ quadratic_eqn a b x2 ∧ root_relation x1 x2) →
  (a + b)^2 ≤ 4 :=
by
  sorry

end prove_ineq_l41_41023


namespace absolute_value_property_l41_41612

theorem absolute_value_property (a b c : ℤ) (h : |a - b| + |c - a| = 1) : |a - c| + |c - b| + |b - a| = 2 :=
sorry

end absolute_value_property_l41_41612


namespace soda_price_l41_41116

-- We define the conditions as given in the problem
def regular_price (P : ℝ) : Prop :=
  -- Regular price per can is P
  ∃ P, 
  -- 25 percent discount on regular price when purchased in 24-can cases
  (∀ (discounted_price_per_can : ℝ), discounted_price_per_can = 0.75 * P) ∧
  -- Price of 70 cans at the discounted price is $28.875
  (70 * 0.75 * P = 28.875)

-- We state the theorem to prove that the regular price per can is $0.55
theorem soda_price (P : ℝ) (h : regular_price P) : P = 0.55 :=
by
  sorry

end soda_price_l41_41116


namespace rectangle_perimeter_l41_41893

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def satisfies_relations (b1 b2 b3 b4 b5 b6 b7 : ℕ) : Prop :=
  b1 + b2 = b3 ∧
  b1 + b3 = b4 ∧
  b3 + b4 = b5 ∧
  b4 + b5 = b6 ∧
  b2 + b5 = b7

def non_overlapping_squares (b1 b2 b3 b4 b5 b6 b7 : ℕ) : Prop :=
  -- Placeholder for expressing that the squares are non-overlapping.
  true -- This is assumed as given in the problem.

theorem rectangle_perimeter (b1 b2 b3 b4 b5 b6 b7 : ℕ)
  (h1 : b1 = 1) (h2 : b2 = 2)
  (h_relations : satisfies_relations b1 b2 b3 b4 b5 b6 b7)
  (h_non_overlapping : non_overlapping_squares b1 b2 b3 b4 b5 b6 b7)
  (h_rel_prime : relatively_prime b6 b7) :
  2 * (b6 + b7) = 46 := by
  sorry

end rectangle_perimeter_l41_41893


namespace yunkyung_work_per_day_l41_41635

theorem yunkyung_work_per_day (T : ℝ) (h : T > 0) (H : T / 3 = 1) : T / 3 = 1/3 := 
by sorry

end yunkyung_work_per_day_l41_41635


namespace laptop_price_l41_41774

theorem laptop_price (cost upfront : ℝ) (upfront_percentage : ℝ) (upfront_eq : upfront = 240) (upfront_percentage_eq : upfront_percentage = 20) : 
  cost = 1200 :=
by
  sorry

end laptop_price_l41_41774


namespace fraction_of_sy_not_declared_major_l41_41914

-- Conditions
variables (T : ℝ) -- Total number of students
variables (first_year : ℝ) -- Fraction of first-year students
variables (second_year : ℝ) -- Fraction of second-year students
variables (decl_fy_major : ℝ) -- Fraction of first-year students who have declared a major
variables (decl_sy_major : ℝ) -- Fraction of second-year students who have declared a major

-- Definitions from conditions
def fraction_first_year_students := 1 / 2
def fraction_second_year_students := 1 / 2
def fraction_fy_declared_major := 1 / 5
def fraction_sy_declared_major := 4 * fraction_fy_declared_major

-- Hollow statement
theorem fraction_of_sy_not_declared_major :
  first_year = fraction_first_year_students →
  second_year = fraction_second_year_students →
  decl_fy_major = fraction_fy_declared_major →
  decl_sy_major = fraction_sy_declared_major →
  (1 - decl_sy_major) * second_year = 1 / 10 :=
by
  sorry

end fraction_of_sy_not_declared_major_l41_41914


namespace least_integer_condition_l41_41543

theorem least_integer_condition : ∃ x : ℤ, (x^2 = 2 * x + 72) ∧ (x = -6) :=
sorry

end least_integer_condition_l41_41543


namespace min_value_of_T_l41_41737

noncomputable def T_min_value (a b c : ℝ) : ℝ :=
  (5 + 2*a*b + 4*a*c) / (a*b + 1)

theorem min_value_of_T :
  ∀ (a b c : ℝ),
  a < 0 →
  b > 0 →
  b^2 ≤ (4 * c) / a →
  c ≤ (1/4) * a * b^2 →
  T_min_value a b c ≥ 4 ∧ (T_min_value a b c = 4 ↔ a * b = -3) :=
by
  intros
  sorry

end min_value_of_T_l41_41737


namespace fort_blocks_count_l41_41691

noncomputable def volume_of_blocks (l w h : ℕ) (wall_thickness floor_thickness top_layer_volume : ℕ) : ℕ :=
  let interior_length := l - 2 * wall_thickness
  let interior_width := w - 2 * wall_thickness
  let interior_height := h - floor_thickness
  let volume_original := l * w * h
  let volume_interior := interior_length * interior_width * interior_height
  volume_original - volume_interior + top_layer_volume

theorem fort_blocks_count : volume_of_blocks 15 12 7 2 1 180 = 912 :=
by
  sorry

end fort_blocks_count_l41_41691


namespace square_side_length_l41_41649

theorem square_side_length (a b s : ℝ) 
  (h_area : a * b = 54) 
  (h_square_condition : 3 * a = b / 2) : 
  s = 9 :=
by 
  sorry

end square_side_length_l41_41649


namespace even_not_div_by_4_not_sum_consecutive_odds_l41_41773

theorem even_not_div_by_4_not_sum_consecutive_odds
  (e : ℤ) (h_even: e % 2 = 0) (h_nondiv4: ¬ (e % 4 = 0)) :
  ∀ n : ℤ, e ≠ n + (n + 2) :=
by
  sorry

end even_not_div_by_4_not_sum_consecutive_odds_l41_41773


namespace largest_possible_markers_in_package_l41_41695

theorem largest_possible_markers_in_package (alex_markers jordan_markers : ℕ) 
  (h1 : alex_markers = 56)
  (h2 : jordan_markers = 42) :
  Nat.gcd alex_markers jordan_markers = 14 :=
by
  sorry

end largest_possible_markers_in_package_l41_41695


namespace chemistry_textbook_weight_l41_41386

theorem chemistry_textbook_weight (G C : ℝ) (h1 : G = 0.62) (h2 : C = G + 6.5) : C = 7.12 :=
by
  sorry

end chemistry_textbook_weight_l41_41386


namespace product_of_xy_l41_41861

theorem product_of_xy (x y : ℝ) : 
  (1 / 5 * (x + y + 4 + 5 + 6) = 5) ∧ 
  (1 / 5 * ((x - 5) ^ 2 + (y - 5) ^ 2 + (4 - 5) ^ 2 + (5 - 5) ^ 2 + (6 - 5) ^ 2) = 2) 
  → x * y = 21 :=
by sorry

end product_of_xy_l41_41861


namespace part1_part2_l41_41753

-- Definitions for the sides and the target equations
def triangleSides (a b c : ℝ) (A B C : ℝ) : Prop :=
  b * Real.sin (C / 2) ^ 2 + c * Real.sin (B / 2) ^ 2 = a / 2

-- The first part of the problem
theorem part1 (a b c A B C : ℝ) (hTriangleSides : triangleSides a b c A B C) :
  b + c = 2 * a :=
  sorry

-- The second part of the problem
theorem part2 (a b c A B C : ℝ) (hTriangleSides : triangleSides a b c A B C) :
  A ≤ π / 3 :=
  sorry

end part1_part2_l41_41753


namespace value_of_x_sq_plus_inv_x_sq_l41_41047

theorem value_of_x_sq_plus_inv_x_sq (x : ℝ) (h : x + 1/x = 1.5) : x^2 + (1/x)^2 = 0.25 := 
by 
  sorry

end value_of_x_sq_plus_inv_x_sq_l41_41047


namespace empty_to_occupied_ratio_of_spheres_in_cylinder_package_l41_41456

theorem empty_to_occupied_ratio_of_spheres_in_cylinder_package
  (R : ℝ) 
  (volume_sphere : ℝ)
  (volume_cylinder : ℝ)
  (sphere_occupies_fraction : ∀ R : ℝ, volume_sphere = (2 / 3) * volume_cylinder) 
  (num_spheres : ℕ) 
  (h_num_spheres : num_spheres = 5) :
  (num_spheres : ℝ) * volume_sphere = (5 * (2 / 3) * π * R^3) → 
  volume_sphere = (4 / 3) * π * R^3 → 
  volume_cylinder = 2 * π * R^3 → 
  (volume_cylinder - volume_sphere) / volume_sphere = 1 / 2 := by 
  sorry

end empty_to_occupied_ratio_of_spheres_in_cylinder_package_l41_41456


namespace rate_percent_calculation_l41_41805

theorem rate_percent_calculation 
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ) 
  (h1 : SI = 3125) 
  (h2 : P = 12500) 
  (h3 : T = 7) 
  (h4 : SI = P * R * T / 100) :
  R = 3.57 :=
by
  sorry

end rate_percent_calculation_l41_41805


namespace correct_option_D_l41_41678

theorem correct_option_D : 
  (-3)^2 = 9 ∧ 
  - (x + y) = -x - y ∧ 
  ¬ (3 * a + 5 * b = 8 * a * b) ∧ 
  5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 :=
by { sorry }

end correct_option_D_l41_41678


namespace kittens_count_l41_41699

def cats_taken_in : ℕ := 12
def cats_initial : ℕ := cats_taken_in / 2
def cats_post_adoption : ℕ := cats_taken_in + cats_initial - 3
def cats_now : ℕ := 19

theorem kittens_count :
  ∃ k : ℕ, cats_post_adoption + k - 1 = cats_now :=
by
  use 5
  sorry

end kittens_count_l41_41699


namespace distance_to_grandma_l41_41768

-- Definitions based on the conditions
def miles_per_gallon : ℕ := 20
def gallons_needed : ℕ := 5

-- The theorem statement to prove the distance is 100 miles
theorem distance_to_grandma : miles_per_gallon * gallons_needed = 100 := by
  sorry

end distance_to_grandma_l41_41768


namespace campers_rowing_afternoon_l41_41283

theorem campers_rowing_afternoon (morning_rowing morning_hiking total : ℕ) 
  (h1 : morning_rowing = 41) 
  (h2 : morning_hiking = 4) 
  (h3 : total = 71) : 
  total - (morning_rowing + morning_hiking) = 26 :=
by
  sorry

end campers_rowing_afternoon_l41_41283


namespace quadratic_inequality_range_l41_41501

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (1/2) * a * x^2 - a * x + 2 > 0) ↔ a ∈ Set.Ico 0 4 := 
by
  sorry

end quadratic_inequality_range_l41_41501


namespace no_base_450_odd_last_digit_l41_41325

theorem no_base_450_odd_last_digit :
  ¬ ∃ b : ℕ, b^3 ≤ 450 ∧ 450 < b^4 ∧ (450 % b) % 2 = 1 :=
sorry

end no_base_450_odd_last_digit_l41_41325


namespace perpendicular_line_equation_l41_41300

theorem perpendicular_line_equation (x y : ℝ) :
  (2, -1) ∈ ({ p : ℝ × ℝ | p.1 * 2 + p.2 * 1 - 3 = 0 }) ∧ 
  (∀ p : ℝ × ℝ, (p.1 * 2 + p.2 * (-4) + 5 = 0) → (p.2 * 1 + p.1 * 2 = 0)) :=
sorry

end perpendicular_line_equation_l41_41300


namespace lee_can_make_cookies_l41_41071

def cookies_per_cup_of_flour (cookies : ℕ) (flour_cups : ℕ) : ℕ :=
  cookies / flour_cups

def flour_needed (sugar_cups : ℕ) (flour_to_sugar_ratio : ℕ) : ℕ :=
  sugar_cups * flour_to_sugar_ratio

def total_cookies (cookies_per_cup : ℕ) (total_flour : ℕ) : ℕ :=
  cookies_per_cup * total_flour

theorem lee_can_make_cookies
  (cookies : ℕ)
  (flour_cups : ℕ)
  (sugar_cups : ℕ)
  (flour_to_sugar_ratio : ℕ)
  (h1 : cookies = 24)
  (h2 : flour_cups = 4)
  (h3 : sugar_cups = 3)
  (h4 : flour_to_sugar_ratio = 2) :
  total_cookies (cookies_per_cup_of_flour cookies flour_cups)
    (flour_needed sugar_cups flour_to_sugar_ratio) = 36 :=
by
  sorry

end lee_can_make_cookies_l41_41071


namespace original_height_of_tree_l41_41903

theorem original_height_of_tree
  (current_height_in_inches : ℕ)
  (percent_taller : ℕ)
  (current_height_is_V := 180)
  (percent_taller_is_50 := 50) :
  (current_height_in_inches * 100) / (percent_taller + 100) / 12 = 10 := sorry

end original_height_of_tree_l41_41903


namespace disc_rotation_exists_l41_41556

noncomputable def sector_length (C : ℝ) (n : ℕ) : ℝ := C / (2 * n)

theorem disc_rotation_exists (C : ℝ) (n : ℕ) (hC_pos : 0 < C) :
  ∃ θ : ℝ, ∃ k : ℕ, k < 2 * n ∧
  let sum_first_part := (2 * n) * sector_length C n (θ = k * (π / n)) ∧
  sum_first_part ≥ (1 / 2) * C :=
sorry

end disc_rotation_exists_l41_41556


namespace min_third_side_length_l41_41362

theorem min_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 24) : 
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ b^2 = a^2 + c^2 ∨  a^2 = b^2 + c^2) ∧ c = 7 :=
sorry

end min_third_side_length_l41_41362


namespace find_b_l41_41077

theorem find_b (α β b : ℤ)
  (h1: α > 1)
  (h2: β < -1)
  (h3: ∃ x : ℝ, α * x^2 + β * x - 2 = 0)
  (h4: ∃ x : ℝ, x^2 + bx - 2 = 0)
  (hb: ∀ root1 root2 : ℝ, root1 * root2 = -2 ∧ root1 + root2 = -b) :
  b = 0 := 
sorry

end find_b_l41_41077


namespace penelope_starbursts_l41_41919

theorem penelope_starbursts (candies_ratio_mnm : ℕ) (candies_ratio_sb : ℕ) (candies_total_mnm : ℕ) (candies_total_sb : ℕ) :
  candies_ratio_mnm = 5 ∧
  candies_ratio_sb = 3 ∧
  candies_total_mnm = 25 →
  candies_total_sb = 15 :=
begin
  sorry
end

end penelope_starbursts_l41_41919


namespace reduction_when_fifth_runner_twice_as_fast_l41_41323

theorem reduction_when_fifth_runner_twice_as_fast (T T1 T2 T3 T4 T5 : ℝ)
  (h1 : T = T1 + T2 + T3 + T4 + T5)
  (h_T1 : (T1 / 2 + T2 + T3 + T4 + T5) = 0.95 * T)
  (h_T2 : (T1 + T2 / 2 + T3 + T4 + T5) = 0.90 * T)
  (h_T3 : (T1 + T2 + T3 / 2 + T4 + T5) = 0.88 * T)
  (h_T4 : (T1 + T2 + T3 + T4 / 2 + T5) = 0.85 * T)
  : (T1 + T2 + T3 + T4 + T5 / 2) = 0.92 * T := 
sorry

end reduction_when_fifth_runner_twice_as_fast_l41_41323


namespace freezer_temp_correct_l41_41622

variable (t_refrigeration : ℝ) (t_freezer : ℝ)

-- Given conditions
def refrigeration_temperature := t_refrigeration = 5
def freezer_temperature := t_freezer = -12

-- Goal: Prove that the freezer compartment's temperature is -12 degrees Celsius
theorem freezer_temp_correct : freezer_temperature t_freezer := by
  sorry

end freezer_temp_correct_l41_41622


namespace sqrt_equality_l41_41462

theorem sqrt_equality :
  Real.sqrt ((18: ℝ) * (17: ℝ) * (16: ℝ) * (15: ℝ) + 1) = 271 :=
by
  sorry

end sqrt_equality_l41_41462


namespace trapezoid_segment_AB_length_l41_41218

/-
In the trapezoid shown, the ratio of the area of triangle ABC to the area of triangle ADC is 5:2.
If AB + CD = 240 cm, prove that the length of segment AB is 171.42857 cm.
-/

theorem trapezoid_segment_AB_length
  (AB CD : ℝ)
  (ratio_areas : ℝ := 5 / 2)
  (area_ratio_condition : AB / CD = ratio_areas)
  (length_sum_condition : AB + CD = 240) :
  AB = 171.42857 :=
sorry

end trapezoid_segment_AB_length_l41_41218


namespace remainder_3005_98_l41_41806

theorem remainder_3005_98 : 3005 % 98 = 65 :=
by sorry

end remainder_3005_98_l41_41806


namespace quadratic_function_expression_rational_function_expression_l41_41282

-- Problem 1:
theorem quadratic_function_expression (f : ℝ → ℝ) :
  (∀ x, f (x + 1) - f x = 3 * x) ∧ (f 0 = 1) → (∀ x, f x = (3 / 2) * x^2 - (3 / 2) * x + 1) :=
by
  sorry

-- Problem 2:
theorem rational_function_expression (f : ℝ → ℝ) : 
  (∀ x, x ≠ 0 → 3 * f (1 / x) + f x = x) → 
  (∀ x, x ≠ 0 → f x = 3 / (8 * x) - x / 8) :=
by
  sorry

end quadratic_function_expression_rational_function_expression_l41_41282


namespace initial_innings_l41_41778

/-- The number of innings a player played initially given the conditions described in the problem. -/
theorem initial_innings (n : ℕ)
  (average_runs : ℕ)
  (additional_runs : ℕ)
  (new_average_increase : ℕ)
  (h1 : average_runs = 42)
  (h2 : additional_runs = 86)
  (h3 : new_average_increase = 4) :
  42 * n + 86 = 46 * (n + 1) → n = 10 :=
by
  intros h
  linarith

end initial_innings_l41_41778


namespace polynomial_identity_l41_41718

theorem polynomial_identity (P : ℝ → ℝ) :
  (∀ x, (x - 1) * P (x + 1) - (x + 2) * P x = 0) ↔ ∃ a : ℝ, ∀ x, P x = a * (x^3 - x) :=
by
  sorry

end polynomial_identity_l41_41718


namespace AB_vector_eq_l41_41037

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (A B C D : V)
variables (a b : V)
variable (ABCD_parallelogram : is_parallelogram A B C D)

-- Definition of the diagonals
def AC_vector : V := C - A
def BD_vector : V := D - B

-- The given condition that diagonals AC and BD are equal to a and b respectively
axiom AC_eq_a : AC_vector A C = a
axiom BD_eq_b : BD_vector B D = b

-- Proof problem
theorem AB_vector_eq : (B - A) = (1/2) • (a - b) :=
sorry

end AB_vector_eq_l41_41037


namespace trip_time_80_minutes_l41_41916

noncomputable def v : ℝ := 1 / 2
noncomputable def speed_highway := 4 * v -- 4 times speed on the highway
noncomputable def time_mountain : ℝ := 20 / v -- Distance on mountain road divided by speed on mountain road
noncomputable def time_highway : ℝ := 80 / speed_highway -- Distance on highway divided by speed on highway
noncomputable def total_time := time_mountain + time_highway

theorem trip_time_80_minutes : total_time = 80 :=
by sorry

end trip_time_80_minutes_l41_41916


namespace find_parameter_a_exactly_two_solutions_l41_41723

noncomputable def system_has_two_solutions (a : ℝ) : Prop :=
∃ (x y : ℝ), |y - 3 - x| + |y - 3 + x| = 6 ∧ (|x| - 4)^2 + (|y| - 3)^2 = a

theorem find_parameter_a_exactly_two_solutions :
  {a : ℝ | system_has_two_solutions a} = {1, 25} :=
by
  sorry

end find_parameter_a_exactly_two_solutions_l41_41723


namespace intersect_eq_l41_41073

variable (M N : Set Int)
def M_def : Set Int := { m | -3 < m ∧ m < 2 }
def N_def : Set Int := { n | -1 ≤ n ∧ n ≤ 3 }

theorem intersect_eq : M_def ∩ N_def = { -1, 0, 1 } := by
  sorry

end intersect_eq_l41_41073


namespace calculate_expression_l41_41839

theorem calculate_expression : (18 / (5 + 2 - 3)) * 4 = 18 := by
  sorry

end calculate_expression_l41_41839


namespace vertex_of_quadratic_l41_41929

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem vertex_of_quadratic :
  ∃ (h k : ℝ), (∀ x : ℝ, f x = (x - h)^2 + k) ∧ (h = 1) ∧ (k = -2) :=
by
  sorry

end vertex_of_quadratic_l41_41929


namespace exponentiation_rule_l41_41679

theorem exponentiation_rule (x : ℝ) : (x^5)^2 = x^10 :=
by {
  sorry
}

end exponentiation_rule_l41_41679


namespace best_model_based_on_R_squared_l41_41216

theorem best_model_based_on_R_squared:
  ∀ (R2_1 R2_2 R2_3 R2_4: ℝ), 
  R2_1 = 0.98 → R2_2 = 0.80 → R2_3 = 0.54 → R2_4 = 0.35 → 
  R2_1 ≥ R2_2 ∧ R2_1 ≥ R2_3 ∧ R2_1 ≥ R2_4 :=
by
  intros R2_1 R2_2 R2_3 R2_4 h1 h2 h3 h4
  sorry

end best_model_based_on_R_squared_l41_41216


namespace range_of_a_l41_41609

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (|x - 1| - |x - 3|) > a) → a < 2 :=
by
  sorry

end range_of_a_l41_41609


namespace fixed_point_l41_41084

theorem fixed_point (m : ℝ) : (2 * m - 1) * 2 - (m + 3) * 3 - (m - 11) = 0 :=
by {
  sorry
}

end fixed_point_l41_41084


namespace hares_cuts_l41_41787

-- Definitions representing the given conditions
def intermediates_fallen := 10
def end_pieces_fixed := 2
def total_logs := intermediates_fallen + end_pieces_fixed

-- Theorem statement
theorem hares_cuts : total_logs - 1 = 11 := by 
  sorry

end hares_cuts_l41_41787


namespace nuts_per_student_l41_41559

theorem nuts_per_student (bags : ℕ) (nuts_per_bag : ℕ) (students : ℕ) (total_nuts : ℕ) (nuts_per_student : ℕ)
    (h1 : bags = 65)
    (h2 : nuts_per_bag = 15)
    (h3 : students = 13)
    (h4 : total_nuts = bags * nuts_per_bag)
    (h5 : nuts_per_student = total_nuts / students)
    : nuts_per_student = 75 :=
by
  sorry

end nuts_per_student_l41_41559


namespace find_x_l41_41317

noncomputable def x (x : ℝ) : Prop :=
  (⌈x⌉₊ * x = 210)

theorem find_x : ∃ x : ℝ, x = 14 ∧ x (14) :=
  by
    sorry

end find_x_l41_41317


namespace intersection_cardinality_l41_41050

variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem intersection_cardinality {a b : ℝ} {f : ℝ → ℝ} :
  (∃! y, (0, y) ∈ ({ (x, y) | y = f x ∧ a ≤ x ∧ x ≤ b } ∩ { (x, y) | x = 0 })) ∨
  ¬ (∃ y, (0, y) ∈ { (x, y) | y = f x ∧ a ≤ x ∧ x ≤ b }) :=
by
  sorry

end intersection_cardinality_l41_41050


namespace cos_2000_eq_neg_inv_sqrt_l41_41196

theorem cos_2000_eq_neg_inv_sqrt (a : ℝ) (h : Real.tan (20 * Real.pi / 180) = a) :
  Real.cos (2000 * Real.pi / 180) = -1 / Real.sqrt (1 + a^2) :=
sorry

end cos_2000_eq_neg_inv_sqrt_l41_41196


namespace packs_of_beef_l41_41229

noncomputable def pounds_per_pack : ℝ := 4
noncomputable def price_per_pound : ℝ := 5.50
noncomputable def total_paid : ℝ := 110
noncomputable def price_per_pack : ℝ := price_per_pound * pounds_per_pack

theorem packs_of_beef (n : ℝ) (h : n = total_paid / price_per_pack) : n = 5 := 
by
  sorry

end packs_of_beef_l41_41229


namespace octahedron_coloring_l41_41770

theorem octahedron_coloring : 
  ∃ (n : ℕ), n = 6 ∧
  ∀ (F : Fin 8 → Fin 4), 
    (∀ (i j : Fin 8), i ≠ j → F i ≠ F j) ∧
    (∃ (pairs : Fin 8 → (Fin 4 × Fin 4)), 
      (∀ (i : Fin 8), ∃ j : Fin 4, pairs i = (j, j)) ∧ 
      (∀ j, ∃ (i : Fin 8), F i = j)) :=
by
  sorry

end octahedron_coloring_l41_41770


namespace combined_weight_l41_41233

theorem combined_weight (mary_weight : ℝ) (jamison_weight : ℝ) (john_weight : ℝ) :
  mary_weight = 160 ∧ jamison_weight = mary_weight + 20 ∧ john_weight = mary_weight + (0.25 * mary_weight) →
  john_weight + mary_weight + jamison_weight = 540 :=
by
  intros h
  obtain ⟨hm, hj, hj'⟩ := h
  rw [hm, hj, hj']
  norm_num
  sorry

end combined_weight_l41_41233


namespace sum_of_numerical_coefficients_binomial_l41_41674

theorem sum_of_numerical_coefficients_binomial (a b : ℕ) (n : ℕ) (h : n = 8) :
  let sum_num_coeff := (a + b)^n
  sum_num_coeff = 256 :=
by 
  sorry

end sum_of_numerical_coefficients_binomial_l41_41674


namespace compare_M_N_l41_41907

variables (a : ℝ)

-- Definitions based on given conditions
def M : ℝ := 2 * a * (a - 2) + 3
def N : ℝ := (a - 1) * (a - 3)

theorem compare_M_N : M a ≥ N a := 
by {
  sorry
}

end compare_M_N_l41_41907


namespace smallest_n_l41_41925

noncomputable def count_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125 + n / 15625 + n / 78125 + n / 390625

theorem smallest_n (a b c : ℕ) (m n : ℕ) (h : a + b + c = 2010)
  (hc : c = 710) (hmn : a.factorial * b.factorial * c.factorial = m * 10^n) (hmdiv : ¬ 10 ∣ m) :
  n = 500 :=
by
  -- Placeholder for the proof
  sorry

end smallest_n_l41_41925


namespace tree_original_height_l41_41906

theorem tree_original_height (current_height_in: ℝ) (growth_percentage: ℝ)
  (h1: current_height_in = 180) (h2: growth_percentage = 0.50) :
  ∃ (original_height_ft: ℝ), original_height_ft = 10 :=
by
  have original_height_in := current_height_in / (1 + growth_percentage)
  have original_height_ft := original_height_in / 12
  use original_height_ft
  sorry

end tree_original_height_l41_41906


namespace average_age_of_other_9_students_l41_41108

variable (total_students : ℕ) (total_average_age : ℝ) (group1_students : ℕ) (group1_average_age : ℝ) (age_student12 : ℝ) (group2_students : ℕ)

theorem average_age_of_other_9_students 
  (h1 : total_students = 16) 
  (h2 : total_average_age = 16) 
  (h3 : group1_students = 5) 
  (h4 : group1_average_age = 14) 
  (h5 : age_student12 = 42) 
  (h6 : group2_students = 9) : 
  (group1_students * group1_average_age + group2_students * 16 + age_student12) / total_students = total_average_age := by
  sorry

end average_age_of_other_9_students_l41_41108


namespace find_sequence_l41_41118

noncomputable def seq (a : ℕ → ℝ) :=
  a 1 = 0 ∧ (∀ n, a (n + 1) = (n / (n + 1)) * (a n + 1))

theorem find_sequence {a : ℕ → ℝ} (h : seq a) :
  ∀ n, a n = (n - 1) / 2 :=
sorry

end find_sequence_l41_41118


namespace range_of_a_l41_41045

open Real

theorem range_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + 3 = x * y) :
  ∀ a : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + y + 3 = x * y → (x + y)^2 - a * (x + y) + 1 ≥ 0)
    ↔ a ≤ 37 / 6 :=
by { sorry }

end range_of_a_l41_41045


namespace calculate_three_times_neg_two_l41_41706

-- Define the multiplication of a positive and a negative number resulting in a negative number
def multiply_positive_negative (a b : Int) (ha : a > 0) (hb : b < 0) : Int :=
  a * b

-- Define the absolute value multiplication
def absolute_value_multiplication (a b : Int) : Int :=
  abs a * abs b

-- The theorem that verifies the calculation
theorem calculate_three_times_neg_two : 3 * (-2) = -6 :=
by
  -- Using the given conditions to conclude the result
  sorry

end calculate_three_times_neg_two_l41_41706


namespace no_real_satisfies_absolute_value_equation_l41_41493

theorem no_real_satisfies_absolute_value_equation :
  ∀ x : ℝ, ¬ (|x - 2| = |x - 1| + |x - 5|) :=
by
  sorry

end no_real_satisfies_absolute_value_equation_l41_41493


namespace oscar_leap_more_than_piper_hop_l41_41241

noncomputable def difference_leap_hop : ℝ :=
let number_of_poles := 51
let total_distance := 7920 -- in feet
let Elmer_strides_per_gap := 44
let Oscar_leaps_per_gap := 15
let Piper_hops_per_gap := 22
let number_of_gaps := number_of_poles - 1
let Elmer_total_strides := Elmer_strides_per_gap * number_of_gaps
let Oscar_total_leaps := Oscar_leaps_per_gap * number_of_gaps
let Piper_total_hops := Piper_hops_per_gap * number_of_gaps
let Elmer_stride_length := total_distance / Elmer_total_strides
let Oscar_leap_length := total_distance / Oscar_total_leaps
let Piper_hop_length := total_distance / Piper_total_hops
Oscar_leap_length - Piper_hop_length

theorem oscar_leap_more_than_piper_hop :
  difference_leap_hop = 3.36 := by
  sorry

end oscar_leap_more_than_piper_hop_l41_41241


namespace yard_length_is_correct_l41_41214

-- Definitions based on the conditions
def trees : ℕ := 26
def distance_between_trees : ℕ := 11

-- Theorem stating that the length of the yard is 275 meters
theorem yard_length_is_correct : (trees - 1) * distance_between_trees = 275 :=
by sorry

end yard_length_is_correct_l41_41214


namespace sum_3000_l41_41259

-- Definitions for the conditions
def geom_sum (a r : ℝ) (n : ℕ) := a * (1 - r^n) / (1 - r)

variables (a r : ℝ)
axiom sum_1000 : geom_sum a r 1000 = 300
axiom sum_2000 : geom_sum a r 2000 = 570

-- The property to prove
theorem sum_3000 : geom_sum a r 3000 = 813 :=
sorry

end sum_3000_l41_41259


namespace total_painted_surface_area_l41_41836

-- Defining the conditions
def num_cubes := 19
def top_layer := 1
def middle_layer := 5
def bottom_layer := 13
def exposed_faces_top_layer := 5
def exposed_faces_middle_corner := 3
def exposed_faces_middle_center := 1
def exposed_faces_bottom_layer := 1

-- Question: How many square meters are painted?
theorem total_painted_surface_area : 
  let top_layer_area := top_layer * exposed_faces_top_layer
  let middle_layer_area := (4 * exposed_faces_middle_corner) + exposed_faces_middle_center
  let bottom_layer_area := bottom_layer * exposed_faces_bottom_layer
  top_layer_area + middle_layer_area + bottom_layer_area = 31 :=
by
  sorry

end total_painted_surface_area_l41_41836


namespace rank_of_A_l41_41187

def A : Matrix (Fin 3) (Fin 5) ℝ :=
  ![![1, 2, 3, 5, 8],
    ![0, 1, 4, 6, 9],
    ![0, 0, 1, 7, 10]]

theorem rank_of_A : A.rank = 3 :=
by sorry

end rank_of_A_l41_41187


namespace chips_sales_l41_41982

theorem chips_sales (total_chips : ℕ) (first_week : ℕ) (second_week : ℕ) (third_week : ℕ) (fourth_week : ℕ)
  (h1 : total_chips = 100)
  (h2 : first_week = 15)
  (h3 : second_week = 3 * first_week)
  (h4 : third_week = fourth_week)
  (h5 : total_chips = first_week + second_week + third_week + fourth_week) : third_week = 20 :=
by
  sorry

end chips_sales_l41_41982


namespace probability_two_red_books_l41_41684

theorem probability_two_red_books (total_books red_books blue_books selected_books : ℕ)
  (h_total: total_books = 8)
  (h_red: red_books = 4)
  (h_blue: blue_books = 4)
  (h_selected: selected_books = 2) :
  (Nat.choose red_books selected_books : ℚ) / (Nat.choose total_books selected_books) = 3 / 14 := by
  sorry

end probability_two_red_books_l41_41684


namespace sum_of_odd_divisors_of_90_l41_41966

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l41_41966


namespace volume_of_snow_correct_l41_41641

noncomputable def volume_of_snow : ℝ :=
  let sidewalk_length := 30
  let sidewalk_width := 3
  let depth := 3 / 4
  let sidewalk_volume := sidewalk_length * sidewalk_width * depth
  
  let garden_path_leg1 := 3
  let garden_path_leg2 := 4
  let garden_path_area := (garden_path_leg1 * garden_path_leg2) / 2
  let garden_path_volume := garden_path_area * depth
  
  let total_volume := sidewalk_volume + garden_path_volume
  total_volume

theorem volume_of_snow_correct : volume_of_snow = 72 := by
  sorry

end volume_of_snow_correct_l41_41641


namespace basketball_team_win_requirement_l41_41689

noncomputable def basketball_win_percentage_goal (games_played_so_far games_won_so_far games_remaining win_percentage_goal : ℕ) : ℕ :=
  let total_games := games_played_so_far + games_remaining
  let required_wins := (win_percentage_goal * total_games) / 100
  required_wins - games_won_so_far

theorem basketball_team_win_requirement :
  basketball_win_percentage_goal 60 45 50 75 = 38 := 
by
  sorry

end basketball_team_win_requirement_l41_41689


namespace system1_solution_system2_solution_system3_solution_l41_41655

theorem system1_solution (x y : ℝ) : 
  (x = 3/2) → (y = 1/2) → (x + 3 * y = 3) ∧ (x - y = 1) :=
by intros; sorry

theorem system2_solution (x y : ℝ) : 
  (x = 0) → (y = 2/5) → ((x + 3 * y) / 2 = 3 / 5) ∧ (5 * (x - 2 * y) = -4) :=
by intros; sorry

theorem system3_solution (x y z : ℝ) : 
  (x = 1) → (y = 2) → (z = 3) → 
  (3 * x + 4 * y + z = 14) ∧ (x + 5 * y + 2 * z = 17) ∧ (2 * x + 2 * y - z = 3) :=
by intros; sorry

end system1_solution_system2_solution_system3_solution_l41_41655


namespace desired_antifreeze_pct_in_colder_climates_l41_41563

-- Definitions for initial conditions
def initial_antifreeze_pct : ℝ := 0.10
def radiator_volume : ℝ := 4
def drained_volume : ℝ := 2.2857
def replacement_antifreeze_pct : ℝ := 0.80

-- Proof goal: Desired percentage of antifreeze in the mixture is 50%
theorem desired_antifreeze_pct_in_colder_climates :
  (drained_volume * replacement_antifreeze_pct + (radiator_volume - drained_volume) * initial_antifreeze_pct) / radiator_volume = 0.50 :=
by
  sorry

end desired_antifreeze_pct_in_colder_climates_l41_41563


namespace original_number_l41_41566

theorem original_number (x : ℝ) (h : 1.10 * x = 550) : x = 500 :=
by
  sorry

end original_number_l41_41566


namespace sum_odd_divisors_of_90_l41_41970

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l41_41970


namespace largest_root_polynomial_intersection_l41_41253

/-
Given a polynomial P(x) = x^6 - 15x^5 + 74x^4 - 130x^3 + a * x^2 + b * x
and a line L(x) = c * x - 24,
such that P(x) stays above L(x) except at three distinct values of x where they intersect,
and one of the intersections is a root of triple multiplicity.
Prove that the largest value of x for which P(x) = L(x) is 6.
-/
theorem largest_root_polynomial_intersection (a b c : ℝ) (P L : ℝ → ℝ) (x : ℝ) :
  P x = x^6 - 15*x^5 + 74*x^4 - 130*x^3 + a*x^2 + b*x →
  L x = c*x - 24 →
  (∀ x, P x ≥ L x) ∨ (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ P x1 = L x1 ∧ P x2 = L x2 ∧ P x3 = L x3 ∧
  (∃ x0 : ℝ, x1 = x0 ∧ x2 = x0 ∧ x3 = x0 ∧ ∃ k : ℕ, k = 3)) →
  x = 6 :=
sorry

end largest_root_polynomial_intersection_l41_41253


namespace black_pens_count_l41_41424

variable (T B : ℕ)
variable (h1 : (3/10:ℚ) * T = 12)
variable (h2 : (1/5:ℚ) * T = B)

theorem black_pens_count (h1 : (3/10:ℚ) * T = 12) (h2 : (1/5:ℚ) * T = B) : B = 8 := by
  sorry

end black_pens_count_l41_41424


namespace fraction_of_40_l41_41945

theorem fraction_of_40 : (3 / 4) * 40 = 30 :=
by
  -- We'll add the 'sorry' here to indicate that this is the proof part which is not required.
  sorry

end fraction_of_40_l41_41945


namespace evaluate_nested_function_l41_41870

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 / 2 else 2 ^ x

theorem evaluate_nested_function : f (f (1 / 2)) = 2 := 
by
  sorry

end evaluate_nested_function_l41_41870


namespace polygon_E_largest_area_l41_41175

def unit_square_area : ℕ := 1
def right_triangle_area : ℚ := 1 / 2
def rectangle_area : ℕ := 2

def polygon_A_area : ℚ := 3 * unit_square_area + 2 * right_triangle_area
def polygon_B_area : ℚ := 2 * unit_square_area + 4 * right_triangle_area
def polygon_C_area : ℚ := 4 * unit_square_area + 1 * rectangle_area
def polygon_D_area : ℚ := 3 * rectangle_area
def polygon_E_area : ℚ := 2 * unit_square_area + 2 * right_triangle_area + 2 * rectangle_area

theorem polygon_E_largest_area :
  polygon_E_area = max polygon_A_area (max polygon_B_area (max polygon_C_area (max polygon_D_area polygon_E_area))) := by
  sorry

end polygon_E_largest_area_l41_41175


namespace greatest_int_less_than_neg_17_div_3_l41_41134

theorem greatest_int_less_than_neg_17_div_3 : 
  ∀ (x : ℚ), x = -17/3 → ⌊x⌋ = -6 :=
by
  sorry

end greatest_int_less_than_neg_17_div_3_l41_41134


namespace frac_equiv_l41_41174

-- Define the given values of x and y.
def x : ℚ := 2 / 7
def y : ℚ := 8 / 11

-- Define the statement to prove.
theorem frac_equiv : (7 * x + 11 * y) / (77 * x * y) = 5 / 8 :=
by
  -- The proof will go here (use 'sorry' for now)
  sorry

end frac_equiv_l41_41174


namespace lizette_third_quiz_score_l41_41764

theorem lizette_third_quiz_score :
  ∀ (x : ℕ),
  (2 * 95 + x) / 3 = 94 → x = 92 :=
by
  intro x h
  have h1 : 2 * 95 = 190 := by norm_num
  have h2 : 3 * 94 = 282 := by norm_num
  sorry

end lizette_third_quiz_score_l41_41764


namespace populations_equal_after_years_l41_41540

-- Defining the initial population and rates of change
def initial_population_X : ℕ := 76000
def rate_of_decrease_X : ℕ := 1200
def initial_population_Y : ℕ := 42000
def rate_of_increase_Y : ℕ := 800

-- Define the number of years for which we need to find the populations to be equal
def years (n : ℕ) : Prop :=
  (initial_population_X - rate_of_decrease_X * n) = (initial_population_Y + rate_of_increase_Y * n)

-- Theorem stating that the populations will be equal at n = 17
theorem populations_equal_after_years {n : ℕ} (h : n = 17) : years n :=
by
  sorry

end populations_equal_after_years_l41_41540


namespace largest_possible_number_of_red_socks_l41_41983

noncomputable def max_red_socks (t : ℕ) (r : ℕ) : Prop :=
  t ≤ 1991 ∧
  ((r * (r - 1) + (t - r) * (t - r - 1)) / (t * (t - 1)) = 1 / 2) ∧
  ∀ r', r' ≤ 990 → (t ≤ 1991 ∧
    ((r' * (r' - 1) + (t - r') * (t - r' - 1)) / (t * (t - 1)) = 1 / 2) → r ≤ r')

theorem largest_possible_number_of_red_socks :
  ∃ t r, max_red_socks t r ∧ r = 990 :=
by
  sorry

end largest_possible_number_of_red_socks_l41_41983


namespace tangent_line_at_point_P_l41_41033

-- Definitions from Conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5
def point_on_circle : Prop := circle_eq 1 2

-- Statement to Prove
theorem tangent_line_at_point_P : 
  point_on_circle → ∃ (m : ℝ) (b : ℝ), (m = -1/2) ∧ (b = 5/2) ∧ (∀ x y : ℝ, y = m * x + b ↔ x + 2 * y - 5 = 0) :=
by
  sorry

end tangent_line_at_point_P_l41_41033


namespace fraction_simplification_l41_41137

theorem fraction_simplification :
  (3100 - 3037)^2 / 81 = 49 := by
  sorry

end fraction_simplification_l41_41137


namespace positive_real_solution_of_equation_l41_41583

theorem positive_real_solution_of_equation (y : ℝ) (h1 : 0 < y) (h2 : (y - 6) / 12 = 6 / (y - 12)) : y = 18 :=
by
  sorry

end positive_real_solution_of_equation_l41_41583


namespace jean_spots_on_sides_l41_41506

variables (total_spots upper_torso_spots back_hindquarters_spots side_spots : ℕ)

def half (x : ℕ) := x / 2
def third (x : ℕ) := x / 3

-- Given conditions
axiom h1 : upper_torso_spots = 30
axiom h2 : upper_torso_spots = half total_spots
axiom h3 : back_hindquarters_spots = third total_spots
axiom h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots

-- Theorem to prove
theorem jean_spots_on_sides (h1 : upper_torso_spots = 30)
    (h2 : upper_torso_spots = half total_spots)
    (h3 : back_hindquarters_spots = third total_spots)
    (h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots) :
    side_spots = 10 := by
  sorry

end jean_spots_on_sides_l41_41506


namespace scientific_notation_of_neg_0_000008691_l41_41316

theorem scientific_notation_of_neg_0_000008691:
  -0.000008691 = -8.691 * 10^(-6) :=
sorry

end scientific_notation_of_neg_0_000008691_l41_41316


namespace tan_of_negative_7pi_over_4_l41_41857

theorem tan_of_negative_7pi_over_4 : Real.tan (-7 * Real.pi / 4) = 1 := 
by 
  sorry

end tan_of_negative_7pi_over_4_l41_41857


namespace tony_fish_after_ten_years_l41_41131

theorem tony_fish_after_ten_years :
  let initial_fish := 6
  let x := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  let y := [4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
  (List.foldl (fun acc ⟨add, die⟩ => acc + add - die) initial_fish (List.zip x y)) = 34 := 
by
  sorry

end tony_fish_after_ten_years_l41_41131


namespace quadratic_inequality_l41_41027

theorem quadratic_inequality (a x : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 - 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) := 
sorry

end quadratic_inequality_l41_41027


namespace find_x_l41_41855

theorem find_x (x : ℝ) : (x + 3 * x + 1000 + 3000) / 4 = 2018 → x = 1018 :=
by 
  intro h
  sorry

end find_x_l41_41855


namespace units_digit_sum_l41_41799

theorem units_digit_sum (n : ℕ) (h : n > 0) : (35^n % 10) + (93^45 % 10) = 8 :=
by
  -- Since the units digit of 35^n is always 5 
  have h1 : 35^n % 10 = 5 := sorry
  -- Since the units digit of 93^45 is 3 (since 45 mod 4 = 1 and the pattern repeats every 4),
  have h2 : 93^45 % 10 = 3 := sorry
  -- Therefore, combining the units digits
  calc
    (35^n % 10) + (93^45 % 10)
    = 5 + 3 := by rw [h1, h2]
    _ = 8 := by norm_num

end units_digit_sum_l41_41799


namespace value_of_k_l41_41327

theorem value_of_k (k : ℝ) : 
  (∃ x y : ℝ, x = 1/3 ∧ y = -8 ∧ -3/4 - 3 * k * x = 7 * y) → k = 55.25 :=
by
  intro h
  sorry

end value_of_k_l41_41327


namespace find_volume_of_pure_alcohol_l41_41078

variable (V1 Vf V2 : ℝ)
variable (P1 Pf : ℝ)

theorem find_volume_of_pure_alcohol
  (h : V2 = Vf * Pf / 100 - V1 * P1 / 100) : 
  V2 = Vf * (Pf / 100) - V1 * (P1 / 100) :=
by
  -- This is the theorem statement. The proof is omitted.
  sorry

end find_volume_of_pure_alcohol_l41_41078


namespace cost_of_fencing_l41_41439

-- Definitions of ratio and area conditions
def sides_ratio (length width : ℕ) : Prop := length / width = 3 / 2
def area (length width : ℕ) : Prop := length * width = 3750

-- Define the cost per meter in paise
def cost_per_meter : ℕ := 70

-- Convert paise to rupees
def paise_to_rupees (paise : ℕ) : ℕ := paise / 100

-- The main statement we want to prove
theorem cost_of_fencing (length width perimeter : ℕ)
  (H1 : sides_ratio length width)
  (H2 : area length width)
  (H3 : perimeter = 2 * length + 2 * width) :
  paise_to_rupees (perimeter * cost_per_meter) = 175 := by
  sorry

end cost_of_fencing_l41_41439


namespace number_of_dress_designs_is_correct_l41_41690

-- Define the number of choices for colors, patterns, and fabric types as conditions
def num_colors : Nat := 4
def num_patterns : Nat := 5
def num_fabric_types : Nat := 2

-- Define the total number of dress designs
def total_dress_designs : Nat := num_colors * num_patterns * num_fabric_types

-- Prove that the total number of different dress designs is 40
theorem number_of_dress_designs_is_correct : total_dress_designs = 40 := by
  sorry

end number_of_dress_designs_is_correct_l41_41690


namespace biology_marks_correct_l41_41094

-- Define the known marks in other subjects
def math_marks : ℕ := 76
def science_marks : ℕ := 65
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 62

-- Define the total number of subjects
def total_subjects : ℕ := 5

-- Define the average marks
def average_marks : ℕ := 74

-- Calculate the total marks of the known four subjects
def total_known_marks : ℕ := math_marks + science_marks + social_studies_marks + english_marks

-- Define a variable to represent the marks in biology
def biology_marks : ℕ := 370 - total_known_marks

-- Statement to prove
theorem biology_marks_correct : biology_marks = 85 := by
  sorry

end biology_marks_correct_l41_41094


namespace value_of_x_l41_41102

-- Define the custom operation * for the problem
def star (a b : ℝ) : ℝ := 4 * a - 2 * b

-- Define the main problem statement
theorem value_of_x (x : ℝ) (h : star 3 (star 7 x) = 5) : x = 49 / 4 :=
by
  have h7x : star 7 x = 28 - 2 * x := by sorry  -- Derived from the definitions
  have h3star7x : star 3 (28 - 2 * x) = -44 + 4 * x := by sorry  -- Derived from substituting star 7 x
  sorry

end value_of_x_l41_41102


namespace gambler_largest_amount_received_l41_41815

def largest_amount_received_back (x y a b : ℕ) (h1: 30 * x + 100 * y = 3000)
    (h2: a + b = 16) (h3: a = b + 2) : ℕ :=
  3000 - (30 * a + 100 * b)

theorem gambler_largest_amount_received (x y a b : ℕ) (h1: 30 * x + 100 * y = 3000)
    (h2: a + b = 16) (h3: a = b + 2) : 
    largest_amount_received_back x y a b h1 h2 h3 = 2030 :=
by sorry

end gambler_largest_amount_received_l41_41815


namespace polynomial_divisibility_l41_41921

-- Definitions
def f (k l m n : ℕ) (x : ℂ) : ℂ :=
  x^(4 * k) + x^(4 * l + 1) + x^(4 * m + 2) + x^(4 * n + 3)

def g (x : ℂ) : ℂ :=
  x^3 + x^2 + x + 1

-- Theorem statement
theorem polynomial_divisibility (k l m n : ℕ) : ∀ x : ℂ, g x ∣ f k l m n x :=
  sorry

end polynomial_divisibility_l41_41921


namespace change_received_l41_41392

def totalCostBeforeDiscount : ℝ :=
  5.75 + 2.50 + 3.25 + 3.75 + 4.20

def discount : ℝ :=
  (3.75 + 4.20) * 0.10

def totalCostAfterDiscount : ℝ :=
  totalCostBeforeDiscount - discount

def salesTax : ℝ :=
  totalCostAfterDiscount * 0.06

def finalTotalCost : ℝ :=
  totalCostAfterDiscount + salesTax

def amountPaid : ℝ :=
  50.00

def change : ℝ :=
  amountPaid - finalTotalCost

theorem change_received (h : change = 30.34) : change = 30.34 := by
  sorry

end change_received_l41_41392


namespace total_bronze_needed_l41_41081

theorem total_bronze_needed (w1 w2 w3 : ℕ) (h1 : w1 = 50) (h2 : w2 = 2 * w1) (h3 : w3 = 4 * w2) : w1 + w2 + w3 = 550 :=
by
  -- We'll complete the proof later
  sorry

end total_bronze_needed_l41_41081


namespace dice_minimum_rolls_l41_41676

theorem dice_minimum_rolls (d1 d2 d3 d4 : ℕ) (h1 : 1 ≤ d1 ∧ d1 ≤ 6)
                           (h2 : 1 ≤ d2 ∧ d2 ≤ 6) (h3 : 1 ≤ d3 ∧ d3 ≤ 6) 
                           (h4 : 1 ≤ d4 ∧ d4 ≤ 6) :
  ∃ n, n = 43 ∧ ∀ (S : ℕ) (x : ℕ → ℕ), 
  (∀ i, 4 ≤ S ∧ S ≤ 24 ∧ x i = 4 ∧ (x i ≤ 6)) →
  (n ≤ 43) ∧ (∃ (k : ℕ), k ≥ 3) :=
sorry

end dice_minimum_rolls_l41_41676


namespace sum_of_odd_divisors_l41_41963

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l41_41963


namespace g_at_4_l41_41657

noncomputable def f (x : ℝ) : ℝ := 5 / (3 - x)
noncomputable def f_inv (x : ℝ) : ℝ := 3 - 5 / x
noncomputable def g (x : ℝ) : ℝ := 2 / (f_inv x) + 7

theorem g_at_4 : g 4 = 8.142857 := by
  sorry

end g_at_4_l41_41657


namespace cost_of_stuffers_number_of_combinations_l41_41193

noncomputable def candy_cane_cost : ℝ := 4 * 0.5
noncomputable def beanie_baby_cost : ℝ := 2 * 3
noncomputable def book_cost : ℝ := 5
noncomputable def toy_cost : ℝ := 3 * 1
noncomputable def gift_card_cost : ℝ := 10
noncomputable def one_child_stuffers_cost : ℝ := candy_cane_cost + beanie_baby_cost + book_cost + toy_cost + gift_card_cost
noncomputable def total_cost : ℝ := one_child_stuffers_cost * 4

def num_books := 5
def num_toys := 10
def toys_combinations : ℕ := Nat.choose num_toys 3
def total_combinations : ℕ := num_books * toys_combinations

theorem cost_of_stuffers (h : total_cost = 104) : total_cost = 104 := by
  sorry

theorem number_of_combinations (h : total_combinations = 600) : total_combinations = 600 := by
  sorry

end cost_of_stuffers_number_of_combinations_l41_41193


namespace manufacturer_price_l41_41149

theorem manufacturer_price :
  ∃ M : ℝ, 
    (∃ R : ℝ, 
      R = 1.15 * M ∧
      ∃ D : ℝ, 
        D = 0.85 * R ∧
        R - D = 57.5) ∧
    M = 333.33 := 
by
  sorry

end manufacturer_price_l41_41149


namespace contrapositive_geometric_sequence_l41_41409

def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem contrapositive_geometric_sequence (a b c : ℝ) :
  (b^2 ≠ a * c) → ¬geometric_sequence a b c :=
by
  intros h
  unfold geometric_sequence
  assumption

end contrapositive_geometric_sequence_l41_41409


namespace calculate_initial_budget_l41_41646

-- Definitions based on conditions
def cost_of_chicken := 12
def cost_per_pound_beef := 3
def pounds_of_beef := 5
def amount_left := 53

-- Derived definition for total cost of beef
def cost_of_beef := cost_per_pound_beef * pounds_of_beef
-- Derived definition for total spent
def total_spent := cost_of_chicken + cost_of_beef
-- Final calculation for initial budget
def initial_budget := total_spent + amount_left

-- Statement to prove
theorem calculate_initial_budget : initial_budget = 80 :=
by
  sorry

end calculate_initial_budget_l41_41646


namespace no_solutions_sinx_eq_sin_sinx_l41_41494

open Real

theorem no_solutions_sinx_eq_sin_sinx (x : ℝ) (h : 0 ≤ x ∧ x ≤ arcsin 0.9) : ¬ (sin x = sin (sin x)) :=
by
  sorry

end no_solutions_sinx_eq_sin_sinx_l41_41494


namespace nelly_refrigerator_payment_l41_41647

theorem nelly_refrigerator_payment (T : ℝ) (p1 p2 p3 : ℝ) (p1_percent p2_percent p3_percent : ℝ)
  (h1 : p1 = 875) (h2 : p2 = 650) (h3 : p3 = 1200)
  (h4 : p1_percent = 0.25) (h5 : p2_percent = 0.15) (h6 : p3_percent = 0.35)
  (total_paid := p1 + p2 + p3)
  (percent_paid := p1_percent + p2_percent + p3_percent)
  (total_cost := total_paid / percent_paid)
  (remaining := total_cost - total_paid) :
  remaining = 908.33 := by
  sorry

end nelly_refrigerator_payment_l41_41647


namespace problem_A_problem_C_l41_41973

section
variables {a b : ℝ}

-- A: If a and b are positive real numbers, and a > b, then a^3 + b^3 > a^2 * b + a * b^2.
theorem problem_A (ha : 0 < a) (hb : 0 < b) (h : a > b) : a^3 + b^3 > a^2 * b + a * b^2 := sorry

end

section
variables {a b : ℝ}

-- C: If a and b are real numbers, then "a > b > 0" is a sufficient but not necessary condition for "1/a < 1/b".
theorem problem_C (ha : 0 < a) (hb : 0 < b) (h : a > b) : 1/a < 1/b := sorry

end

end problem_A_problem_C_l41_41973


namespace three_digit_integers_product_30_l41_41055

theorem three_digit_integers_product_30 : 
  ∃ (n : ℕ), 
    (100 ≤ n ∧ n < 1000) ∧ 
    (∀ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 → 
    (1 ≤ d1 ∧ d1 ≤ 9) ∧ 
    (1 ≤ d2 ∧ d2 ≤ 9) ∧
    (1 ≤ d3 ∧ d3 ≤ 9) ∧
    d1 * d2 * d3 = 30) ∧ 
    n = 12 :=
sorry

end three_digit_integers_product_30_l41_41055


namespace percentage_of_x_l41_41115

theorem percentage_of_x (x y : ℝ) (h1 : y = x / 4) (p : ℝ) (h2 : p / 100 * x = 20 / 100 * y) : p = 5 :=
by sorry

end percentage_of_x_l41_41115


namespace equal_roots_quadratic_l41_41204

theorem equal_roots_quadratic {k : ℝ} 
  (h : (∃ x : ℝ, x^2 - 6 * x + k = 0 ∧ x^2 - 6 * x + k = 0)) : 
  k = 9 :=
sorry

end equal_roots_quadratic_l41_41204


namespace smallest_n_divisible_by_one_billion_l41_41643

-- Define the sequence parameters and the common ratio
def first_term : ℚ := 5 / 8
def second_term : ℚ := 50
def common_ratio : ℚ := second_term / first_term -- this is 80

-- Define the n-th term of the geometric sequence
noncomputable def nth_term (n : ℕ) : ℚ :=
  first_term * (common_ratio ^ (n - 1))

-- Define the target divisor (one billion)
def target_divisor : ℤ := 10 ^ 9

-- Prove that the smallest n such that nth_term n is divisible by 10^9 is 9
theorem smallest_n_divisible_by_one_billion :
  ∃ n : ℕ, nth_term n = (first_term * (common_ratio ^ (n - 1))) ∧ 
           (target_divisor : ℚ) ∣ nth_term n ∧
           n = 9 :=
by sorry

end smallest_n_divisible_by_one_billion_l41_41643


namespace right_triangle_angle_l41_41894

open Real

theorem right_triangle_angle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h2 : c^2 = 2 * a * b) : 
  ∃ θ : ℝ, θ = 45 ∧ tan θ = a / b := 
by sorry

end right_triangle_angle_l41_41894


namespace binom_8_5_l41_41011

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the binomial coefficient function
def binom (n k : ℕ) := fact n / (fact k * fact (n - k))

-- State the theorem to prove binom 8 5 = 56
theorem binom_8_5 : binom 8 5 = 56 := by
  sorry

end binom_8_5_l41_41011


namespace incorrect_option_D_l41_41981

-- Definitions based on conditions
def cumulative_progress (days : ℕ) : ℕ :=
  30 * days

-- The Lean statement representing the mathematically equivalent proof problem
theorem incorrect_option_D : cumulative_progress 11 = 330 ∧ ¬ (cumulative_progress 10 = 330) :=
by {
  sorry
}

end incorrect_option_D_l41_41981


namespace area_PQR_l41_41711

-- Define the coordinates of the points
def P : ℝ × ℝ := (-3, 4)
def Q : ℝ × ℝ := (4, 9)
def R : ℝ × ℝ := (5, -3)

-- Function to calculate the area of a triangle given three points
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

-- Statement to prove the area of triangle PQR is 44.5
theorem area_PQR : area_of_triangle P Q R = 44.5 := sorry

end area_PQR_l41_41711


namespace max_f_l41_41416

open Real

noncomputable def f (x : ℝ) : ℝ := 3 + log x + 4 / log x

theorem max_f (h : 0 < x ∧ x < 1) : f x ≤ -1 :=
sorry

end max_f_l41_41416


namespace distance_between_points_l41_41470

theorem distance_between_points : 
  let p1 := (0, 24)
  let p2 := (10, 0)
  dist p1 p2 = 26 := 
by
  sorry

end distance_between_points_l41_41470


namespace save_plan_l41_41092

noncomputable def net_income (gross: ℕ) : ℕ :=
  (gross * 87) / 100

def ivan_salary : ℕ := net_income 55000
def vasilisa_salary : ℕ := net_income 45000
def vasalisa_mother_salary_before : ℕ := net_income 18000
def vasalisa_father_salary : ℕ := net_income 20000
def son_scholarship_state : ℕ := 3000
def son_scholarship_non_state : ℕ := net_income 15000

def expenses : ℕ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

def savings_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state - expenses

def total_income_before_may : ℕ :=
  ivan_salary + vasilisa_salary + vasalisa_mother_salary_before + vasalisa_father_salary + son_scholarship_state

def total_income_may_august : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state

def savings_may_august : ℕ :=
  total_income_may_august - expenses

def total_income_september : ℕ :=
  ivan_salary + vasilisa_salary + 10000 + vasalisa_father_salary + son_scholarship_state + son_scholarship_non_state

def savings_september : ℕ :=
  total_income_september - expenses

theorem save_plan : 
  savings_before_may = 49060 ∧ savings_may_august = 43400 ∧ savings_september = 56450 :=
by
  sorry

end save_plan_l41_41092


namespace symmetric_point_xoz_plane_l41_41380

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_xoz (M : Point3D) : Point3D :=
  ⟨M.x, -M.y, M.z⟩

theorem symmetric_point_xoz_plane :
  let M := Point3D.mk 5 1 (-2)
  symmetric_xoz M = Point3D.mk 5 (-1) (-2) :=
by
  sorry

end symmetric_point_xoz_plane_l41_41380


namespace find_b_in_cubic_function_l41_41190

noncomputable def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem find_b_in_cubic_function (a b c d : ℝ) (h1: cubic_function a b c d 2 = 0)
  (h2: cubic_function a b c d (-1) = 0) (h3: cubic_function a b c d 1 = 4) :
  b = 6 :=
by
  sorry

end find_b_in_cubic_function_l41_41190


namespace combined_weight_of_three_l41_41231

theorem combined_weight_of_three (Mary Jamison John : ℝ) 
  (h₁ : Mary = 160) 
  (h₂ : Jamison = Mary + 20) 
  (h₃ : John = Mary + (1/4) * Mary) :
  Mary + Jamison + John = 540 := by
  sorry

end combined_weight_of_three_l41_41231


namespace snowball_total_distance_l41_41450

noncomputable def total_distance (a1 d n : ℕ) : ℕ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem snowball_total_distance :
  total_distance 6 5 25 = 1650 := by
  sorry

end snowball_total_distance_l41_41450


namespace angle_is_3_pi_over_4_l41_41192

def vec (α β : ℝ) : ℝ × ℝ := (α, β)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def magnitude (u : ℝ × ℝ) : ℝ := real.sqrt (u.1^2 + u.2^2)

def angle_between_vectors (u v : ℝ × ℝ) : ℝ := 
  real.acos ((dot_product u v) / (magnitude u * magnitude v))

def a := vec (-1) 2
def b := vec (-1) (-1)
def v := vec (-6) (-6)
def w := vec 0 1

theorem angle_is_3_pi_over_4 : angle_between_vectors v w = 3 * real.pi / 4 :=
  sorry

end angle_is_3_pi_over_4_l41_41192


namespace max_ratio_BO_BM_l41_41630

theorem max_ratio_BO_BM
  (C : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hC : C = (0, -4))
  (hCir : ∃ (P : ℝ × ℝ), (P.1 - 2)^2 + (P.2 - 4)^2 = 1 ∧ A = ((P.1 + C.1) / 2, (P.2 + C.2) / 2))
  (hPar : ∃ (x y : ℝ), B = (x, y) ∧ y^2 = 4 * x) :
  ∃ t, t = (4 * Real.sqrt 7)/7 ∧ t = Real.sqrt ((B.1^2 + 4 * B.1)/((B.1 + 1/2)^2)) := by
  -- Given conditions and definitions
  obtain ⟨P, hP, hA⟩ := hCir
  obtain ⟨x, y, hB⟩ := hPar
  use (4 * Real.sqrt 7) / 7
  sorry

end max_ratio_BO_BM_l41_41630


namespace binom_8_5_l41_41010

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the binomial coefficient function
def binom (n k : ℕ) := fact n / (fact k * fact (n - k))

-- State the theorem to prove binom 8 5 = 56
theorem binom_8_5 : binom 8 5 = 56 := by
  sorry

end binom_8_5_l41_41010


namespace reef_age_in_decimal_l41_41564

def octal_to_decimal (n: Nat) : Nat :=
  match n with
  | 367 => 7 * (8^0) + 6 * (8^1) + 3 * (8^2)
  | _   => 0  -- Placeholder for other values if needed

theorem reef_age_in_decimal : octal_to_decimal 367 = 247 := by
  sorry

end reef_age_in_decimal_l41_41564


namespace phantom_needs_more_money_l41_41401

def amount_phantom_has : ℤ := 50
def cost_black : ℤ := 11
def count_black : ℕ := 2
def cost_red : ℤ := 15
def count_red : ℕ := 3
def cost_yellow : ℤ := 13
def count_yellow : ℕ := 2

def total_cost : ℤ := cost_black * count_black + cost_red * count_red + cost_yellow * count_yellow
def additional_amount_needed : ℤ := total_cost - amount_phantom_has

theorem phantom_needs_more_money : additional_amount_needed = 43 := by
  sorry

end phantom_needs_more_money_l41_41401


namespace monkey_climb_time_l41_41291

theorem monkey_climb_time : 
  ∀ (height hop slip : ℕ), 
    height = 22 ∧ hop = 3 ∧ slip = 2 → 
    ∃ (time : ℕ), time = 20 := 
by
  intros height hop slip h
  rcases h with ⟨h_height, ⟨h_hop, h_slip⟩⟩
  sorry

end monkey_climb_time_l41_41291


namespace combination_8_5_l41_41004

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l41_41004


namespace distance_to_building_materials_l41_41132

theorem distance_to_building_materials (D : ℝ) 
  (h1 : 2 * 10 * 4 * D = 8000) : 
  D = 100 := 
by
  sorry

end distance_to_building_materials_l41_41132


namespace phantom_needs_more_money_l41_41399

variable (given_money black_ink_price red_ink_price yellow_ink_price total_black_inks total_red_inks total_yellow_inks : ℕ)

def total_cost (total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price : ℕ) : ℕ :=
  total_black_inks * black_ink_price + total_red_inks * red_ink_price + total_yellow_inks * yellow_ink_price

theorem phantom_needs_more_money
  (h_given : given_money = 50)
  (h_black : black_ink_price = 11)
  (h_red : red_ink_price = 15)
  (h_yellow : yellow_ink_price = 13)
  (h_total_black : total_black_inks = 2)
  (h_total_red : total_red_inks = 3)
  (h_total_yellow : total_yellow_inks = 2) :
  given_money < total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price →
  total_cost total_black_inks total_red_inks total_yellow_inks black_ink_price red_ink_price yellow_ink_price - given_money = 43 := by
  sorry

end phantom_needs_more_money_l41_41399


namespace num_arrangements_l41_41860

-- Define the problem conditions
def athletes : Finset ℕ := {0, 1, 2, 3, 4, 5}
def A : ℕ := 0
def B : ℕ := 1

-- Define the constraint that athlete A cannot run the first leg and athlete B cannot run the fourth leg
def valid_arrangements (sequence : Fin 4 → ℕ) : Prop :=
  sequence 0 ≠ A ∧ sequence 3 ≠ B

-- Main theorem statement: There are 252 valid arrangements
theorem num_arrangements : (Fin 4 → ℕ) → ℕ :=
  sorry

end num_arrangements_l41_41860


namespace find_x_if_vectors_parallel_l41_41492

/--
Given the vectors a = (2 * x + 1, 3) and b = (2 - x, 1), if a is parallel to b, 
then x must be equal to 1.
-/
theorem find_x_if_vectors_parallel (x : ℝ) :
  let a := (2 * x + 1, 3)
  let b := (2 - x, 1)
  (∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2) → x = 1 :=
by
  sorry

end find_x_if_vectors_parallel_l41_41492


namespace min_sum_a_b_l41_41730

theorem min_sum_a_b (a b : ℝ) (h_cond: 1/a + 4/b = 1) (a_pos : 0 < a) (b_pos : 0 < b) : 
  a + b ≥ 9 :=
sorry

end min_sum_a_b_l41_41730


namespace anna_money_ratio_l41_41713

theorem anna_money_ratio (total_money spent_furniture left_money given_to_Anna : ℕ)
  (h_total : total_money = 2000)
  (h_spent : spent_furniture = 400)
  (h_left : left_money = 400)
  (h_after_furniture : total_money - spent_furniture = given_to_Anna + left_money) :
  (given_to_Anna / left_money) = 3 :=
by
  have h1 : total_money - spent_furniture = 1600 := by sorry
  have h2 : given_to_Anna = 1200 := by sorry
  have h3 : given_to_Anna / left_money = 3 := by sorry
  exact h3

end anna_money_ratio_l41_41713


namespace sum_odd_divisors_of_90_l41_41969

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l41_41969


namespace intersection_M_N_eq_M_l41_41072

-- Definition of M
def M := {y : ℝ | ∃ x : ℝ, y = 3^x}

-- Definition of N
def N := {y : ℝ | ∃ x : ℝ, y = x^2 - 1}

-- Theorem statement
theorem intersection_M_N_eq_M : (M ∩ N) = M :=
  sorry

end intersection_M_N_eq_M_l41_41072


namespace negation_of_universal_l41_41664

-- Definitions based on the provided problem
def prop (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Main proof problem statement
theorem negation_of_universal : 
  ¬ (∀ x : ℝ, x > 0 → x^2 > 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 ≤ 0 :=
by sorry

end negation_of_universal_l41_41664


namespace min_dist_value_l41_41093

open Real

-- Defining the parabola y^2 = 4x
def on_parabola (P : ℝ × ℝ) : Prop :=
  P.2 ^ 2 = 4 * P.1

-- Defining the line x - y + 5 = 0
def on_line (Q : ℝ × ℝ) : Prop :=
  Q.1 - Q.2 + 5 = 0

-- Defining the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Function to compute the Euclidean distance between two points
def dist (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

noncomputable def min_value (P Q : ℝ × ℝ) : ℝ :=
  dist P focus + dist P Q

theorem min_dist_value :
  ∃ P Q : ℝ × ℝ, on_parabola P ∧ on_line Q ∧ (min_value P Q = 3 * sqrt 2) :=
by
  sorry

end min_dist_value_l41_41093


namespace proof_smallest_integer_proof_sum_of_integers_l41_41433

def smallest_integer (n : Int) : Prop :=
  ∃ (a b c d e : Int), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8 ∧ a + e = 204 ∧ n = 98

def sum_of_integers (n : Int) : Prop :=
  ∃ (a b c d e : Int), a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧ e = n + 8 ∧ a + e = 204 ∧ a + b + c + d + e = 510

theorem proof_smallest_integer : ∃ n : Int, smallest_integer n := by
  sorry

theorem proof_sum_of_integers : ∃ n : Int, sum_of_integers n := by
  sorry

end proof_smallest_integer_proof_sum_of_integers_l41_41433


namespace correct_statement_2_l41_41179

-- Definitions of parallel and perpendicular relationships
variables (a b : line) (α β : plane)

-- Conditions
def parallel (x y : plane) : Prop := sorry -- definition not provided
def perpendicular (x y : plane) : Prop := sorry -- definition not provided
def line_parallel_plane (l : line) (p : plane) : Prop := sorry -- definition not provided
def line_perpendicular_plane (l : line) (p : plane) : Prop := sorry -- definition not provided
def line_perpendicular (l1 l2 : line) : Prop := sorry -- definition not provided

-- Proof of the correct statement among the choices
theorem correct_statement_2 :
  line_perpendicular a b → line_perpendicular_plane a α → line_perpendicular_plane b β → perpendicular α β :=
by
  intros h1 h2 h3
  sorry

end correct_statement_2_l41_41179


namespace position_of_2010_is_correct_l41_41701

-- Definition of the arithmetic sequence and row starting points
def first_term : Nat := 1
def common_difference : Nat := 2
def S (n : Nat) : Nat := (n * (2 * first_term + (n - 1) * common_difference)) / 2

-- Definition of the position where number 2010 appears
def row_of_number (x : Nat) : Nat :=
  let n := (Nat.sqrt x) + 1
  if (n - 1) * (n - 1) < x && x <= n * n then n else n - 1

def column_of_number (x : Nat) : Nat :=
  let row := row_of_number x
  x - (S (row - 1)) + 1

-- Main theorem
theorem position_of_2010_is_correct :
  row_of_number 2010 = 45 ∧ column_of_number 2010 = 74 :=
by
  sorry

end position_of_2010_is_correct_l41_41701


namespace g_88_value_l41_41313

noncomputable def g : ℕ → ℕ := sorry

axiom g_increasing (n m : ℕ) (h : n < m) : g n < g m
axiom g_multiplicative (m n : ℕ) : g (m * n) = g m * g n
axiom g_exponential_condition (m n : ℕ) (h : m ≠ n ∧ m ^ n = n ^ m) : g m = n ∨ g n = m

theorem g_88_value : g 88 = 7744 :=
sorry

end g_88_value_l41_41313


namespace block_fraction_visible_above_water_l41_41808

-- Defining constants
def weight_of_block : ℝ := 30 -- N
def buoyant_force_submerged : ℝ := 50 -- N

-- Defining the proof problem
theorem block_fraction_visible_above_water (W Fb : ℝ) (hW : W = weight_of_block) (hFb : Fb = buoyant_force_submerged) :
  (1 - W / Fb) = 2 / 5 :=
by
  -- Proof is omitted
  sorry

end block_fraction_visible_above_water_l41_41808


namespace find_largest_number_l41_41481

theorem find_largest_number
  (a b c d : ℕ)
  (h1 : a + b + c = 222)
  (h2 : a + b + d = 208)
  (h3 : a + c + d = 197)
  (h4 : b + c + d = 180) :
  max a (max b (max c d)) = 89 :=
by
  sorry

end find_largest_number_l41_41481


namespace stocking_stuffers_total_l41_41352

-- Defining the number of items per category
def candy_canes := 4
def beanie_babies := 2
def books := 1
def small_toys := 3
def gift_cards := 1

-- Total number of stocking stuffers per child
def items_per_child := candy_canes + beanie_babies + books + small_toys + gift_cards

-- Number of children
def number_of_children := 3

-- Total number of stocking stuffers for all children
def total_stocking_stuffers := items_per_child * number_of_children

-- Statement to be proved
theorem stocking_stuffers_total : total_stocking_stuffers = 33 := by
  sorry

end stocking_stuffers_total_l41_41352


namespace complement_union_of_sets_l41_41387

variable {U M N : Set ℕ}

theorem complement_union_of_sets (h₁ : M ⊆ N) (h₂ : N ⊆ U) :
  (U \ M) ∪ (U \ N) = U \ M :=
by
  sorry

end complement_union_of_sets_l41_41387


namespace retail_price_per_book_l41_41813

theorem retail_price_per_book (n r w : ℝ)
  (h1 : r * n = 48)
  (h2 : w = r - 2)
  (h3 : w * (n + 4) = 48) :
  r = 6 := by
  sorry

end retail_price_per_book_l41_41813


namespace solution_set_for_inequality_l41_41336

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f (x)

noncomputable def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

theorem solution_set_for_inequality
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_decreasing : decreasing_on f (Set.Iio 0))
  (h_f1 : f 1 = 0) :
  {x : ℝ | x^3 * f x > 0} = {x : ℝ | x > 1 ∨ x < -1} :=
by
  sorry

end solution_set_for_inequality_l41_41336


namespace correct_calculated_value_l41_41878

theorem correct_calculated_value (x : ℝ) (h : 3 * x - 5 = 103) : x / 3 - 5 = 7 := 
by 
  sorry

end correct_calculated_value_l41_41878


namespace minimize_expression_l41_41326

theorem minimize_expression : ∃ c : ℝ, (∀ x : ℝ, (1/3 * x^2 + 7*x - 4) ≥ (1/3 * c^2 + 7*c - 4)) ∧ (c = -21/2) :=
sorry

end minimize_expression_l41_41326


namespace abs_inequality_solution_set_l41_41673

theorem abs_inequality_solution_set (x : ℝ) : (|x - 1| ≥ 5) ↔ (x ≥ 6 ∨ x ≤ -4) := 
by sorry

end abs_inequality_solution_set_l41_41673


namespace total_seats_l41_41373

-- Define the conditions
variable {S : ℝ} -- Total number of seats in the hall
variable {vacantSeats : ℝ} (h_vacant : vacantSeats = 240) -- Number of vacant seats
variable {filledPercentage : ℝ} (h_filled : filledPercentage = 0.60) -- Percentage of seats filled

-- Total seats in the hall
theorem total_seats (h : 0.40 * S = 240) : S = 600 :=
sorry

end total_seats_l41_41373


namespace bronze_needed_l41_41083

/-- 
The total amount of bronze Martin needs for three bells in pounds.
-/
theorem bronze_needed (w1 w2 w3 : ℕ) 
  (h1 : w1 = 50) 
  (h2 : w2 = 2 * w1) 
  (h3 : w3 = 4 * w2) 
  : (w1 + w2 + w3 = 550) := 
by { 
  sorry 
}

end bronze_needed_l41_41083


namespace total_marbles_l41_41890

variable (w o p : ℝ)

-- Conditions as hypothesis
axiom h1 : o + p = 10
axiom h2 : w + p = 12
axiom h3 : w + o = 5

theorem total_marbles : w + o + p = 13.5 :=
by
  sorry

end total_marbles_l41_41890


namespace one_eighth_of_two_power_36_equals_two_power_x_l41_41358

theorem one_eighth_of_two_power_36_equals_two_power_x (x : ℕ) :
  (1 / 8) * (2 : ℝ) ^ 36 = (2 : ℝ) ^ x → x = 33 :=
by
  intro h
  sorry

end one_eighth_of_two_power_36_equals_two_power_x_l41_41358


namespace porch_width_l41_41111

theorem porch_width (L_house W_house total_area porch_length W : ℝ)
  (h1 : L_house = 20.5) (h2 : W_house = 10) (h3 : total_area = 232) (h4 : porch_length = 6) (h5 : total_area = (L_house * W_house) + (porch_length * W)) :
  W = 4.5 :=
by 
  sorry

end porch_width_l41_41111


namespace cubic_roots_c_div_d_l41_41940

theorem cubic_roots_c_div_d (a b c d : ℚ) :
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 1/2 ∨ x = 4) →
  (c / d = 9 / 4) :=
by
  intros h
  -- Proof would go here
  sorry

end cubic_roots_c_div_d_l41_41940


namespace value_of_abc_l41_41613

theorem value_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + 1 / b = 5) (h2 : b + 1 / c = 2) (h3 : c + 1 / a = 3) : 
  abc = 1 :=
by
  sorry

end value_of_abc_l41_41613


namespace binom_eight_five_l41_41018

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l41_41018


namespace equivalent_single_percentage_increase_l41_41062

noncomputable def calculate_final_price (p : ℝ) : ℝ :=
  let p1 := p * (1 + 0.15)
  let p2 := p1 * (1 + 0.20)
  let p_final := p2 * (1 - 0.10)
  p_final

theorem equivalent_single_percentage_increase (p : ℝ) : 
  calculate_final_price p = p * 1.242 :=
by
  sorry

end equivalent_single_percentage_increase_l41_41062


namespace max_k_value_l41_41848

def maximum_k (k : ℕ) : ℕ := 2

theorem max_k_value
  (k : ℕ)
  (h1 : 2 * k + 1 ≤ 20)  -- Condition implicitly implied by having subsets of a 20-element set
  (h2 : ∀ (s t : Finset (Fin 20)), s.card = 7 → t.card = 7 → s ≠ t → (s ∩ t).card = k) : k ≤ maximum_k k := 
by {
  sorry
}

end max_k_value_l41_41848


namespace sqrt_meaningful_iff_l41_41206

theorem sqrt_meaningful_iff (x : ℝ) : (∃ r : ℝ, r = sqrt (6 + x)) ↔ x ≥ -6 :=
by
  sorry

end sqrt_meaningful_iff_l41_41206


namespace special_set_exists_l41_41227

def exists_special_set : Prop :=
  ∃ S : Finset ℕ, S.card = 4004 ∧ 
  (∀ A : Finset ℕ, A ⊆ S ∧ A.card = 2003 → (A.sum id % 2003 ≠ 0))

-- statement with sorry to skip the proof
theorem special_set_exists : exists_special_set :=
sorry

end special_set_exists_l41_41227


namespace greatest_divisor_same_remainder_l41_41471

theorem greatest_divisor_same_remainder (a b c : ℕ) (d1 d2 d3 : ℕ) (h1 : a = 41) (h2 : b = 71) (h3 : c = 113)
(hd1 : d1 = b - a) (hd2 : d2 = c - b) (hd3 : d3 = c - a) :
  Nat.gcd (Nat.gcd d1 d2) d3 = 6 :=
by
  -- some computation here which we are skipping
  sorry

end greatest_divisor_same_remainder_l41_41471


namespace each_person_paid_l41_41157

-- Define the conditions: total bill and number of people
def totalBill : ℕ := 135
def numPeople : ℕ := 3

-- Define the question as a theorem to prove the correct answer
theorem each_person_paid : totalBill / numPeople = 45 :=
by
  -- Here, we can skip the proof since the statement is required only.
  sorry

end each_person_paid_l41_41157


namespace newborn_members_approximation_l41_41372

-- Defining the conditions
def survival_prob_first_month : ℚ := 7/8
def survival_prob_second_month : ℚ := 7/8
def survival_prob_third_month : ℚ := 7/8
def survival_prob_three_months : ℚ := (7/8) ^ 3
def expected_survivors : ℚ := 133.984375

-- Statement to prove that the number of newborn members, N, approximates to 200
theorem newborn_members_approximation (N : ℚ) : 
  N * survival_prob_three_months = expected_survivors → 
  N = 200 :=
by
  sorry

end newborn_members_approximation_l41_41372


namespace find_ab_l41_41075

variables {a b : ℝ}

theorem find_ab
  (h : ∀ x : ℝ, 0 ≤ x → 0 ≤ x^4 - x^3 + a * x + b ∧ x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2) :
  a * b = -1 :=
sorry

end find_ab_l41_41075


namespace solve_system_equation_152_l41_41600

theorem solve_system_equation_152 (x y z a b c : ℝ)
  (h1 : x * y - 2 * y - 3 * x = 0)
  (h2 : y * z - 3 * z - 5 * y = 0)
  (h3 : x * z - 5 * x - 2 * z = 0)
  (h4 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h5 : x = a)
  (h6 : y = b)
  (h7 : z = c) :
  a^2 + b^2 + c^2 = 152 := by
  sorry

end solve_system_equation_152_l41_41600


namespace larger_solution_quadratic_l41_41858

theorem larger_solution_quadratic : 
  ∀ x1 x2 : ℝ, (x^2 - 13 * x - 48 = 0) → x1 ≠ x2 → (x1 = 16 ∨ x2 = 16) → max x1 x2 = 16 :=
by
  sorry

end larger_solution_quadratic_l41_41858


namespace quadratic_coefficients_l41_41991

theorem quadratic_coefficients :
  ∀ x : ℝ, 3 * x^2 = 5 * x - 1 → (∃ a b c : ℝ, a = 3 ∧ b = -5 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x h
  use 3, -5, 1
  sorry

end quadratic_coefficients_l41_41991


namespace percent_of_games_lost_l41_41669

theorem percent_of_games_lost (w l : ℕ) (h1 : w / l = 8 / 5) (h2 : w + l = 65) :
  (l * 100 / 65 : ℕ) = 38 :=
sorry

end percent_of_games_lost_l41_41669


namespace triangle_count_l41_41235

theorem triangle_count (a b c : ℕ) (hb : b = 2008) (hab : a ≤ b) (hbc : b ≤ c) (ht : a + b > c) : 
  ∃ n, n = 2017036 :=
by
  sorry

end triangle_count_l41_41235


namespace gas_price_increase_l41_41667

theorem gas_price_increase (P C : ℝ) (x : ℝ) 
  (h1 : P * C = P * (1 + x) * 1.10 * C * (1 - 0.27272727272727)) :
  x = 0.25 :=
by
  -- The proof will be filled here
  sorry

end gas_price_increase_l41_41667


namespace time_spent_per_bone_l41_41775

theorem time_spent_per_bone
  (total_hours : ℤ) (number_of_bones : ℤ) 
  (h1 : total_hours = 206) 
  (h2 : number_of_bones = 206) :
  (total_hours / number_of_bones = 1) := 
by {
  -- proof would go here
  sorry
}

end time_spent_per_bone_l41_41775


namespace jenny_problem_l41_41755

def round_to_nearest_ten (n : ℤ) : ℤ :=
  if n % 10 < 5 then n - (n % 10) else n + (10 - n % 10)

theorem jenny_problem : round_to_nearest_ten (58 + 29) = 90 := 
by
  sorry

end jenny_problem_l41_41755


namespace value_of_a_sq_sub_b_sq_l41_41057

theorem value_of_a_sq_sub_b_sq (a b : ℝ) (h1 : a + b = 20) (h2 : a - b = 4) : a^2 - b^2 = 80 :=
by
  sorry

end value_of_a_sq_sub_b_sq_l41_41057


namespace taxi_ride_cost_l41_41570

namespace TaxiFare

def baseFare : ℝ := 2.00
def costPerMile : ℝ := 0.30
def taxRate : ℝ := 0.10
def distance : ℝ := 8.0

theorem taxi_ride_cost :
  let fare_without_tax := baseFare + (costPerMile * distance)
  let tax := taxRate * fare_without_tax
  let total_fare := fare_without_tax + tax
  total_fare = 4.84 := by
  let fare_without_tax := baseFare + (costPerMile * distance)
  let tax := taxRate * fare_without_tax
  let total_fare := fare_without_tax + tax
  sorry

end TaxiFare

end taxi_ride_cost_l41_41570


namespace plane_through_line_and_point_l41_41729

-- Definitions from the conditions
def line (x y z : ℝ) : Prop :=
  (x - 1) / 2 = (y - 3) / 4 ∧ (x - 1) / 2 = z / (-1)

def pointP1 : ℝ × ℝ × ℝ := (1, 5, 2)

-- Correct answer
def plane_eqn (x y z : ℝ) : Prop :=
  5 * x - 2 * y + 2 * z + 1 = 0

-- The theorem to prove
theorem plane_through_line_and_point (x y z : ℝ) :
  line x y z → plane_eqn x y z := by
  sorry

end plane_through_line_and_point_l41_41729


namespace price_reduction_equation_l41_41822

theorem price_reduction_equation (x : ℝ) : 200 * (1 - x) ^ 2 = 162 :=
by
  sorry

end price_reduction_equation_l41_41822


namespace third_height_of_triangle_l41_41792

theorem third_height_of_triangle 
  (a b c ha hb hc : ℝ)
  (h_abc_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_heights : ∃ (h1 h2 h3 : ℕ), h1 = 3 ∧ h2 = 10 ∧ h3 ≠ h1 ∧ h3 ≠ h2) :
  ∃ (h3 : ℕ), h3 = 4 :=
by
  sorry

end third_height_of_triangle_l41_41792


namespace roots_polynomial_equation_l41_41429

noncomputable def rootsEquation (x y : ℝ) := x + y = 10 ∧ |x - y| = 12

theorem roots_polynomial_equation : ∃ (x y : ℝ), rootsEquation x y ∧ (x^2 - 10 * x - 11 = 0) := sorry

end roots_polynomial_equation_l41_41429


namespace children_attended_l41_41675

theorem children_attended (A C : ℕ) (h1 : C = 2 * A) (h2 : A + C = 42) : C = 28 :=
by
  sorry

end children_attended_l41_41675


namespace original_acid_concentration_l41_41288

theorem original_acid_concentration (P : ℝ) (h1 : 0.5 * P + 0.5 * 20 = 35) : P = 50 :=
by
  sorry

end original_acid_concentration_l41_41288


namespace rows_of_roses_l41_41621

variable (rows total_roses_per_row roses_per_row_red roses_per_row_non_red roses_per_row_white roses_per_row_pink total_pink_roses : ℕ)
variable (half_two_fifth three_fifth : ℚ)

-- Assume the conditions
axiom h1 : total_roses_per_row = 20
axiom h2 : roses_per_row_red = total_roses_per_row / 2
axiom h3 : roses_per_row_non_red = total_roses_per_row - roses_per_row_red
axiom h4 : roses_per_row_white = (3 / 5 : ℚ) * roses_per_row_non_red
axiom h5 : roses_per_row_pink = (2 / 5 : ℚ) * roses_per_row_non_red
axiom h6 : total_pink_roses = 40

-- Prove the number of rows in the garden
theorem rows_of_roses : rows = total_pink_roses / (roses_per_row_pink) :=
by
  sorry

end rows_of_roses_l41_41621


namespace f_monotonic_intervals_g_not_below_f_inequality_holds_l41_41487

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3 * x
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem f_monotonic_intervals :
  ∀ x : ℝ, 0 < x → 
    (0 < x ∧ x < 1 / 2 → f x < f (x + 1)) ∧ 
    (1 / 2 < x ∧ x < 1 → f x > f (x + 1)) ∧ 
    (1 < x → f x < f (x + 1)) :=
sorry

theorem g_not_below_f :
  ∀ x : ℝ, 0 < x → f x < g x :=
sorry

theorem inequality_holds (n : ℕ) : (2 * n + 1)^2 > 4 * Real.log (Nat.factorial n) :=
sorry

end f_monotonic_intervals_g_not_below_f_inequality_holds_l41_41487


namespace sum_of_seven_digits_l41_41652

theorem sum_of_seven_digits : 
  ∃ (digits : Finset ℕ), 
    digits.card = 7 ∧ 
    digits ⊆ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    ∃ (a b c d e f g : ℕ), 
      a + b + c = 25 ∧ 
      d + e + f + g = 17 ∧ 
      digits = {a, b, c, d, e, f, g} ∧ 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
      b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
      c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
      d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
      e ≠ f ∧ e ≠ g ∧
      f ≠ g ∧
      (a + b + c + d + e + f + g = 33) := sorry

end sum_of_seven_digits_l41_41652


namespace rectangular_region_area_l41_41448

-- Definitions based on conditions
variable (w : ℝ) -- length of the shorter sides
variable (l : ℝ) -- length of the longer side
variable (total_fence_length : ℝ) -- total length of the fence

-- Given conditions as hypotheses
theorem rectangular_region_area
  (h1 : l = 2 * w) -- The length of the side opposite the wall is twice the length of each of the other two fenced sides
  (h2 : w + w + l = total_fence_length) -- The total length of the fence is 40 feet
  (h3 : total_fence_length = 40) -- total fence length of 40 feet
: (w * l) = 200 := -- The area of the rectangular region is 200 square feet
sorry

end rectangular_region_area_l41_41448


namespace simplify_fraction_l41_41245

variables {a b c x y z : ℝ}

theorem simplify_fraction :
  (cx * (a^2 * x^2 + 3 * a^2 * y^2 + c^2 * z^2) + bz * (a^2 * x^2 + 3 * c^2 * x^2 + c^2 * y^2)) / (cx + bz) =
  a^2 * x^2 + c^2 * y^2 + c^2 * z^2 :=
sorry

end simplify_fraction_l41_41245


namespace total_capacity_iv_bottle_l41_41821

-- Definitions of the conditions
def initial_volume : ℝ := 100 -- milliliters
def rate_of_flow : ℝ := 2.5 -- milliliters per minute
def observation_time : ℝ := 12 -- minutes
def empty_space_at_12_min : ℝ := 80 -- milliliters

-- Definition of the problem statement in Lean 4
theorem total_capacity_iv_bottle :
  initial_volume + rate_of_flow * observation_time + empty_space_at_12_min = 150 := 
by
  sorry

end total_capacity_iv_bottle_l41_41821


namespace min_value_of_reciprocals_l41_41785

theorem min_value_of_reciprocals (m n : ℝ) (h1 : m + n = 2) (h2 : m * n > 0) : 
  (1 / m) + (1 / n) = 2 :=
by
  -- the proof needs to be completed here.
  sorry

end min_value_of_reciprocals_l41_41785


namespace inequality_x2_y4_z6_l41_41095

variable (x y z : ℝ)

theorem inequality_x2_y4_z6
    (hx : 0 < x)
    (hy : 0 < y)
    (hz : 0 < z) :
    x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
by
  sorry

end inequality_x2_y4_z6_l41_41095


namespace white_ring_weight_l41_41234

def weight_of_orange_ring : ℝ := 0.08
def weight_of_purple_ring : ℝ := 0.33
def total_weight_of_rings : ℝ := 0.83

def weight_of_white_ring (total : ℝ) (orange : ℝ) (purple : ℝ) : ℝ :=
  total - (orange + purple)

theorem white_ring_weight :
  weight_of_white_ring total_weight_of_rings weight_of_orange_ring weight_of_purple_ring = 0.42 :=
by
  sorry

end white_ring_weight_l41_41234


namespace polynomial_decomposition_l41_41761

noncomputable def s (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 1
noncomputable def t (x : ℝ) : ℝ := x + 18

def g (x : ℝ) : ℝ := 3 * x^4 + 9 * x^3 - 7 * x^2 + 2 * x + 6
def e (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem polynomial_decomposition : s 1 + t (-1) = 27 :=
by
  sorry

end polynomial_decomposition_l41_41761


namespace canteen_consumption_l41_41978

theorem canteen_consumption :
  ∀ (x : ℕ),
    (x + (500 - x) + (200 - x)) = 700 → 
    (500 - x) = 7 * (200 - x) →
    x = 150 :=
by
  sorry

end canteen_consumption_l41_41978


namespace pentagon_largest_angle_l41_41250

theorem pentagon_largest_angle (x : ℝ) (h : 2 * x + 3 * x + 4 * x + 5 * x + 6 * x = 540) : 6 * x = 162 :=
sorry

end pentagon_largest_angle_l41_41250


namespace sum_of_positive_odd_divisors_of_90_l41_41956

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l41_41956


namespace find_f_2015_l41_41829

def f (x : ℝ) := 2 * x - 1 

theorem find_f_2015 (f : ℝ → ℝ)
  (H1 : ∀ a b : ℝ, f ((2 * a + b) / 3) = (2 * f a + f b) / 3)
  (H2 : f 1 = 1)
  (H3 : f 4 = 7) :
  f 2015 = 4029 := by 
  sorry

end find_f_2015_l41_41829


namespace outlet_pipe_emptying_time_l41_41430

theorem outlet_pipe_emptying_time :
  let rate1 := 1 / 18
  let rate2 := 1 / 20
  let fill_time := 0.08333333333333333
  ∃ x : ℝ, (rate1 + rate2 - 1 / x = 1 / fill_time) → x = 45 :=
by
  intro rate1 rate2 fill_time
  use 45
  intro h
  sorry

end outlet_pipe_emptying_time_l41_41430


namespace neg_exists_equiv_forall_l41_41520

theorem neg_exists_equiv_forall (p : Prop) :
  (¬ (∃ n : ℕ, n^2 > 2^n)) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := sorry

end neg_exists_equiv_forall_l41_41520


namespace quadratic_inequality_solution_minimum_value_expression_l41_41202

theorem quadratic_inequality_solution (a : ℝ) : (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) → a > 3 :=
sorry

theorem minimum_value_expression (a : ℝ) : (a > 3) → a + 9 / (a - 1) ≥ 7 ∧ (a + 9 / (a - 1) = 7 ↔ a = 4) :=
sorry

end quadratic_inequality_solution_minimum_value_expression_l41_41202


namespace range_of_f_t_eq_1_on_0_to_4_range_of_a_for_f_le_5_range_of_t_for_diff_le_8_l41_41521

variable (t : ℝ)

def f (x : ℝ) : ℝ := x^2 - 2*t*x + 2

-- Problem 1
theorem range_of_f_t_eq_1_on_0_to_4 :
  t = 1 → set.range (λ x → f t x) (set.Icc 0 4) = set.Icc 1 10 :=
by
  intro ht
  rw [ht]
  sorry

-- Problem 2
theorem range_of_a_for_f_le_5 :
  t = 1 → (∀ x ∈ set.Icc a (a + 2), f t x ≤ 5) → a ∈ set.Icc (-1) 1 :=
by
  intro ht h
  rw [ht]
  sorry

-- Problem 3
theorem range_of_t_for_diff_le_8 :
  (∀ x1 x2 ∈ set.Icc 0 4, abs (f t x1 - f t x2) ≤ 8) → t ∈ set.Icc (4 - 2 * real.sqrt 2) (2 * real.sqrt 2) :=
by
  intro h
  sorry

end range_of_f_t_eq_1_on_0_to_4_range_of_a_for_f_le_5_range_of_t_for_diff_le_8_l41_41521


namespace average_mpg_correct_l41_41704

noncomputable def average_mpg (initial_miles final_miles : ℕ) (refill1 refill2 refill3 : ℕ) : ℚ :=
  let distance := final_miles - initial_miles
  let total_gallons := refill1 + refill2 + refill3
  distance / total_gallons

theorem average_mpg_correct :
  average_mpg 32000 33100 15 10 22 = 23.4 :=
by
  sorry

end average_mpg_correct_l41_41704


namespace find_a_l41_41721

theorem find_a :
  ∀ (a : ℝ), 
  (∀ x : ℝ, 2 * x^2 - 2016 * x + 2016^2 - 2016 * a - 1 = a^2) → 
  (∃ x1 x2 : ℝ, 2 * x1^2 - 2016 * x1 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 2 * x2^2 - 2016 * x2 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 x1 < a ∧ a < x2) → 
  2015 < a ∧ a < 2017 :=
by sorry

end find_a_l41_41721


namespace decompose_five_eighths_l41_41709

theorem decompose_five_eighths : 
  ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (5 : ℚ) / 8 = 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) := 
by
  sorry

end decompose_five_eighths_l41_41709


namespace length_of_BC_l41_41223

theorem length_of_BC 
  (A B C X : Type) 
  (d_AB : ℝ) (d_AC : ℝ) 
  (circle_center_A : A) 
  (radius_AB : ℝ)
  (intersects_BC : B → C → X)
  (BX CX : ℕ) 
  (h_BX_in_circle : BX = d_AB) 
  (h_CX_in_circle : CX = d_AC) 
  (h_integer_lengths : ∃ x y : ℕ, BX = x ∧ CX = y) :
  BX + CX = 61 :=
begin
  sorry
end

end length_of_BC_l41_41223


namespace total_cost_is_correct_l41_41384

-- Define the conditions
def piano_cost : ℝ := 500
def lesson_cost_per_lesson : ℝ := 40
def number_of_lessons : ℝ := 20
def discount_rate : ℝ := 0.25

-- Define the total cost of lessons before discount
def total_lesson_cost_before_discount : ℝ := lesson_cost_per_lesson * number_of_lessons

-- Define the discount amount
def discount_amount : ℝ := discount_rate * total_lesson_cost_before_discount

-- Define the total cost of lessons after discount
def total_lesson_cost_after_discount : ℝ := total_lesson_cost_before_discount - discount_amount

-- Define the total cost of everything
def total_cost : ℝ := piano_cost + total_lesson_cost_after_discount

-- The statement to be proven
theorem total_cost_is_correct : total_cost = 1100 := by
  sorry

end total_cost_is_correct_l41_41384


namespace problem_proof_l41_41476

theorem problem_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b = ab → a + 4 * b = 9) ∧
  (a + b = 1 → ∀ a b,  2^a + 2^(b + 1) ≥ 4) ∧
  (a + b = ab → 1 / a^2 + 2 / b^2 = 2 / 3) ∧
  (a + b = 1 → ∀ a b,  2 * a / (a + b^2) + b / (a^2 + b) = (2 * Real.sqrt 3 / 3) + 1) :=
by
  sorry

end problem_proof_l41_41476


namespace scientific_notation_86400_l41_41818

theorem scientific_notation_86400 : 86400 = 8.64 * 10^4 :=
by
  sorry

end scientific_notation_86400_l41_41818


namespace max_median_value_l41_41146

theorem max_median_value (x : ℕ) (h : 198 + x ≤ 392) : x ≤ 194 :=
by {
  sorry
}

end max_median_value_l41_41146


namespace admin_in_sample_l41_41979

-- Define the total number of staff members
def total_staff : ℕ := 200

-- Define the number of administrative personnel
def admin_personnel : ℕ := 24

-- Define the sample size taken
def sample_size : ℕ := 50

-- Goal: Prove the number of administrative personnel in the sample
theorem admin_in_sample : 
  (admin_personnel : ℚ) / (total_staff : ℚ) * (sample_size : ℚ) = 6 := 
by
  sorry

end admin_in_sample_l41_41979


namespace graph_does_not_pass_first_quadrant_l41_41329

variables {a b x : ℝ}

theorem graph_does_not_pass_first_quadrant 
  (h₁ : 0 < a ∧ a < 1) 
  (h₂ : b < -1) : 
  ¬ ∃ x : ℝ, 0 < x ∧ 0 < a^x + b :=
sorry

end graph_does_not_pass_first_quadrant_l41_41329


namespace number_of_pencils_l41_41258

variable (P L : ℕ)

-- Conditions
def condition1 : Prop := P / L = 5 / 6
def condition2 : Prop := L = P + 5

-- Statement to prove
theorem number_of_pencils (h1 : condition1 P L) (h2 : condition2 P L) : L = 30 :=
  sorry

end number_of_pencils_l41_41258


namespace domain_shift_l41_41046

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the domain of f
def domain_f : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

-- State the problem in Lean
theorem domain_shift (hf : ∀ x, f x ∈ domain_f) : 
    { x | 1 ≤ x ∧ x ≤ 2 } = { x | ∃ y, y ∈ domain_f ∧ x = y + 1 } :=
by
  sorry

end domain_shift_l41_41046


namespace intersection_of_A_and_B_l41_41875

def setA : Set ℤ := {x | abs x < 4}
def setB : Set ℤ := {x | x - 1 ≥ 0}
def setIntersection : Set ℤ := {1, 2, 3}

theorem intersection_of_A_and_B : setA ∩ setB = setIntersection :=
by
  sorry

end intersection_of_A_and_B_l41_41875


namespace fraction_of_number_l41_41944

-- Given definitions based on the problem conditions
def fraction : ℚ := 3 / 4
def number : ℕ := 40

-- Theorem statement to be proved
theorem fraction_of_number : fraction * number = 30 :=
by
  sorry -- This indicates that the proof is not yet provided

end fraction_of_number_l41_41944


namespace simplify_expression_l41_41246

theorem simplify_expression (y : ℝ) : 5 * y + 7 * y + 8 * y = 20 * y :=
by
  sorry

end simplify_expression_l41_41246


namespace unattainable_y_l41_41727

theorem unattainable_y (x : ℝ) (h1 : x ≠ -3/2) : y = (1 - x) / (2 * x + 3) -> ¬(y = -1 / 2) :=
by sorry

end unattainable_y_l41_41727


namespace bear_pies_l41_41912

-- Lean definitions model:

variables (v_M v_B u_M u_B : ℝ)
variables (M_raspberries B_raspberries : ℝ)
variables (P_M P_B : ℝ)

-- Given conditions
axiom v_B_eq_6v_M : v_B = 6 * v_M
axiom u_B_eq_3u_M : u_B = 3 * u_M
axiom B_raspberries_eq_2M_raspberries : B_raspberries = 2 * M_raspberries
axiom P_sum : P_B + P_M = 60
axiom P_B_eq_9P_M : P_B = 9 * P_M

-- The theorem to prove
theorem bear_pies : P_B = 54 :=
sorry

end bear_pies_l41_41912


namespace ratio_seconds_l41_41668

theorem ratio_seconds (x : ℕ) (h : 12 / x = 6 / 240) : x = 480 :=
sorry

end ratio_seconds_l41_41668


namespace insect_population_calculations_l41_41066

theorem insect_population_calculations :
  (let ants_1 := 100
   let ants_2 := ants_1 - 20 * ants_1 / 100
   let ants_3 := ants_2 - 25 * ants_2 / 100
   let bees_1 := 150
   let bees_2 := bees_1 - 30 * bees_1 / 100
   let termites_1 := 200
   let termites_2 := termites_1 - 10 * termites_1 / 100
   ants_3 = 60 ∧ bees_2 = 105 ∧ termites_2 = 180) :=
by
  sorry

end insect_population_calculations_l41_41066


namespace set_intersection_l41_41763

def A := {x : ℝ | x^2 - 3*x ≥ 0}
def B := {x : ℝ | x < 1}
def intersection := {x : ℝ | x ≤ 0}

theorem set_intersection : A ∩ B = intersection :=
  sorry

end set_intersection_l41_41763


namespace find_D_coordinates_l41_41255

theorem find_D_coordinates:
  ∀ (A B C : (ℝ × ℝ)), 
  A = (-2, 5) ∧ C = (3, 7) ∧ B = (-3, 0) →
  ∃ D : (ℝ × ℝ), D = (2, 2) :=
by
  sorry

end find_D_coordinates_l41_41255


namespace greatest_possible_gcd_l41_41760

theorem greatest_possible_gcd (d : ℕ) (a : ℕ → ℕ) (h_sum : (a 0) + (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 595)
  (h_gcd : ∀ i, d ∣ a i) : d ≤ 35 :=
sorry

end greatest_possible_gcd_l41_41760


namespace find_x_l41_41328

theorem find_x (x : ℕ) (h1 : 8 = 2 ^ 3) (h2 : 32 = 2 ^ 5) :
  (2^(x+2) * 8^(x-1) = 32^3) ↔ (x = 4) :=
by
  sorry

end find_x_l41_41328


namespace half_angle_in_second_and_fourth_quadrants_l41_41185

theorem half_angle_in_second_and_fourth_quadrants
  (k : ℤ) (α : ℝ) (h : 2 * k * π + π < α ∧ α < 2 * k * π + 3 * π / 2) :
  (∃ m : ℤ, m * π + π / 2 < α / 2 ∧ α / 2 < m * π + 3 * π / 4) :=
by sorry

end half_angle_in_second_and_fourth_quadrants_l41_41185


namespace cos_double_angle_l41_41602

theorem cos_double_angle (α : ℝ) (h : Real.sin ((Real.pi / 6) + α) = 1 / 3) :
  Real.cos ((2 * Real.pi / 3) - 2 * α) = -7 / 9 := by
  sorry

end cos_double_angle_l41_41602


namespace min_moves_to_emit_all_colors_l41_41143

theorem min_moves_to_emit_all_colors :
  ∀ (colors : Fin 7 → Prop) (room : Fin 4 → Fin 7)
  (h : ∀ i j, i ≠ j → room i ≠ room j) (moves : ℕ),
  (∀ (n : ℕ) (i : Fin 4), n < moves → ∃ c : Fin 7, colors c ∧ room i = c ∧
    (∀ j, j ≠ i → room j ≠ c)) →
  (∃ n, n = 8) :=
by
  sorry

end min_moves_to_emit_all_colors_l41_41143


namespace sphere_surface_area_of_solid_l41_41298

theorem sphere_surface_area_of_solid (l w h : ℝ) (hl : l = 2) (hw : w = 1) (hh : h = 2) 
: 4 * Real.pi * ((Real.sqrt (l^2 + w^2 + h^2) / 2)^2) = 9 * Real.pi := 
by 
  sorry

end sphere_surface_area_of_solid_l41_41298


namespace inequality_always_holds_l41_41199

theorem inequality_always_holds (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) →
  (a > 3) ∧ (∀ x : ℝ, x = a + 9 / (a - 1) → x ≥ 7) :=
by
  sorry

end inequality_always_holds_l41_41199


namespace fraction_remain_same_l41_41355

theorem fraction_remain_same (x y : ℝ) : (2 * x + y) / (3 * x + y) = (2 * (10 * x) + (10 * y)) / (3 * (10 * x) + (10 * y)) :=
by sorry

end fraction_remain_same_l41_41355


namespace smallest_factor_of_32_not_8_l41_41726

theorem smallest_factor_of_32_not_8 : ∃ n : ℕ, n = 16 ∧ (n ∣ 32) ∧ ¬(n ∣ 8) ∧ ∀ m : ℕ, (m ∣ 32) ∧ ¬(m ∣ 8) → n ≤ m :=
by
  sorry

end smallest_factor_of_32_not_8_l41_41726


namespace intersection_lines_l41_41789

theorem intersection_lines (a b : ℝ) (h1 : ∀ x y : ℝ, (x = 3 ∧ y = 1) → x = 1/3 * y + a)
                          (h2 : ∀ x y : ℝ, (x = 3 ∧ y = 1) → y = 1/3 * x + b) :
  a + b = 8 / 3 :=
sorry

end intersection_lines_l41_41789


namespace minimum_value_l41_41211

noncomputable def min_value_condition (a b : ℝ) : Prop :=
  a + b = 1 / 2

theorem minimum_value (a b : ℝ) (h : min_value_condition a b) :
  (4 / a) + (1 / b) ≥ 18 :=
by
  sorry

end minimum_value_l41_41211


namespace eggs_left_over_l41_41022

theorem eggs_left_over (David_eggs Ella_eggs Fiona_eggs : ℕ)
  (hD : David_eggs = 45)
  (hE : Ella_eggs = 58)
  (hF : Fiona_eggs = 29) :
  (David_eggs + Ella_eggs + Fiona_eggs) % 10 = 2 :=
by
  sorry

end eggs_left_over_l41_41022


namespace distance_between_trees_l41_41551

theorem distance_between_trees (num_trees: ℕ) (total_length: ℕ) (trees_at_end: ℕ) 
(h1: num_trees = 26) (h2: total_length = 300) (h3: trees_at_end = 2) :
  total_length / (num_trees - 1) = 12 :=
by sorry

end distance_between_trees_l41_41551


namespace area_of_hexagon_is_9_sqrt_3_div_4_l41_41378

namespace HexagonArea

noncomputable def hexagon_area : ℝ :=
  let side_length := 1
  let longer_side_length := 2
  let area_large_triangle := Real.sqrt 3 / 4 * (longer_side_length ^ 2)
  let area_small_triangle := Real.sqrt 3 / 4 * (side_length ^ 2)
  2 * area_large_triangle + area_small_triangle

theorem area_of_hexagon_is_9_sqrt_3_div_4 : 
  hexagon_area = 9 * Real.sqrt 3 / 4 :=
sorry

end HexagonArea

end area_of_hexagon_is_9_sqrt_3_div_4_l41_41378


namespace evaluate_expression_at_x_eq_3_l41_41468

theorem evaluate_expression_at_x_eq_3 :
  (3 ^ 3) ^ (3 ^ 3) = 7625597484987 := by
  sorry

end evaluate_expression_at_x_eq_3_l41_41468


namespace average_marks_of_all_students_l41_41406

theorem average_marks_of_all_students (n1 n2 a1 a2 : ℕ) (n1_eq : n1 = 12) (a1_eq : a1 = 40) 
  (n2_eq : n2 = 28) (a2_eq : a2 = 60) : 
  ((n1 * a1 + n2 * a2) / (n1 + n2) : ℕ) = 54 := 
by
  sorry

end average_marks_of_all_students_l41_41406


namespace value_of_expression_l41_41871

noncomputable def f : ℝ → ℝ
| x => if x > 0 then -1 else if x < 0 then 1 else 0

theorem value_of_expression (a b : ℝ) (h : a ≠ b) :
  (a + b + (a - b) * f (a - b)) / 2 = min a b := 
sorry

end value_of_expression_l41_41871


namespace dealership_sales_l41_41457

theorem dealership_sales (sports_cars : ℕ) (sedans : ℕ) (trucks : ℕ) 
  (h1 : sports_cars = 36)
  (h2 : (3 : ℤ) * sedans = 5 * sports_cars)
  (h3 : (3 : ℤ) * trucks = 4 * sports_cars) :
  sedans = 60 ∧ trucks = 48 := 
sorry

end dealership_sales_l41_41457


namespace problem1_problem2_real_problem2_complex_problem3_l41_41910

-- Problem 1: Prove that if 2 ∈ A, then {-1, 1/2} ⊆ A
theorem problem1 (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : 2 ∈ A) : -1 ∈ A ∧ (1/2) ∈ A := sorry

-- Problem 2: Prove that A cannot be a singleton set for real numbers, but can for complex numbers.
theorem problem2_real (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : ∃ a ∈ A, a ≠ 1) : ¬(∃ a, A = {a}) := sorry

theorem problem2_complex (A : Set ℂ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : ∃ a ∈ A, a ≠ 1) : (∃ a, A = {a}) := sorry

-- Problem 3: Prove that 1 - 1/a ∈ A given a ∈ A
theorem problem3 (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (a : ℝ) (ha : a ∈ A) : (1 - 1/a) ∈ A := sorry

end problem1_problem2_real_problem2_complex_problem3_l41_41910


namespace chess_piece_max_visitable_squares_l41_41828

-- Define initial board properties and movement constraints
structure ChessBoard :=
  (rows : ℕ)
  (columns : ℕ)
  (movement : ℕ)
  (board_size : rows * columns = 225)

-- Define condition for unique visitation
def can_visit (movement : ℕ) (board_size : ℕ) : Prop :=
  ∃ (max_squares : ℕ), (max_squares ≤ board_size) ∧ (max_squares = 196)

-- Main theorem statement 
theorem chess_piece_max_visitable_squares (cb : ChessBoard) : 
  can_visit 196 225 :=
by sorry

end chess_piece_max_visitable_squares_l41_41828


namespace breadth_of_rectangular_plot_l41_41536

theorem breadth_of_rectangular_plot (b : ℝ) (h1 : ∃ l : ℝ, l = 3 * b) (h2 : b * 3 * b = 675) : b = 15 :=
by
  sorry

end breadth_of_rectangular_plot_l41_41536


namespace solution_set_eq_2m_add_2_gt_zero_l41_41369

theorem solution_set_eq_2m_add_2_gt_zero {m : ℝ} (h : ∀ x : ℝ, mx + 2 > 0 ↔ x < 2) : m = -1 :=
sorry

end solution_set_eq_2m_add_2_gt_zero_l41_41369


namespace Frank_work_hours_l41_41474

def hoursWorked (h_monday h_tuesday h_wednesday h_thursday h_friday h_saturday : Nat) : Nat :=
  h_monday + h_tuesday + h_wednesday + h_thursday + h_friday + h_saturday

theorem Frank_work_hours
  (h_monday : Nat := 8)
  (h_tuesday : Nat := 10)
  (h_wednesday : Nat := 7)
  (h_thursday : Nat := 9)
  (h_friday : Nat := 6)
  (h_saturday : Nat := 4) :
  hoursWorked h_monday h_tuesday h_wednesday h_thursday h_friday h_saturday = 44 :=
by
  unfold hoursWorked
  sorry

end Frank_work_hours_l41_41474


namespace sum_of_arithmetic_series_51_to_100_l41_41686

theorem sum_of_arithmetic_series_51_to_100 :
  let first_term := 51
  let last_term := 100
  let n := (last_term - first_term) + 1
  2 * (n / 2) * (first_term + last_term) / 2 = 3775 :=
by
  sorry

end sum_of_arithmetic_series_51_to_100_l41_41686


namespace hall_length_l41_41830

theorem hall_length (L h : ℝ) (width volume : ℝ) 
  (h_width : width = 6) 
  (h_volume : L * width * h = 108) 
  (h_area : 12 * L = 2 * L * h + 12 * h) : 
  L = 6 := 
  sorry

end hall_length_l41_41830


namespace triangle_arithmetic_angles_l41_41777

/-- The angles in a triangle are in arithmetic progression and the side lengths are 6, 7, and y.
    The sum of the possible values of y equals a + sqrt b + sqrt c,
    where a, b, and c are positive integers. Prove that a + b + c = 68. -/
theorem triangle_arithmetic_angles (y : ℝ) (a b c : ℕ) (h1 : a = 3) (h2 : b = 22) (h3 : c = 43) :
    (∃ y1 y2 : ℝ, y1 = 3 + Real.sqrt 22 ∧ y2 = Real.sqrt 43 ∧ (y = y1 ∨ y = y2))
    → a + b + c = 68 :=
by
  sorry

end triangle_arithmetic_angles_l41_41777


namespace oomyapeck_eyes_count_l41_41754

-- Define the various conditions
def number_of_people : ℕ := 3
def fish_per_person : ℕ := 4
def eyes_per_fish : ℕ := 2
def eyes_given_to_dog : ℕ := 2

-- Compute the total number of fish
def total_fish : ℕ := number_of_people * fish_per_person

-- Compute the total number of eyes from the total number of fish
def total_eyes : ℕ := total_fish * eyes_per_fish

-- Compute the number of eyes Oomyapeck eats
def eyes_eaten_by_oomyapeck : ℕ := total_eyes - eyes_given_to_dog

-- The proof statement
theorem oomyapeck_eyes_count : eyes_eaten_by_oomyapeck = 22 := by
  sorry

end oomyapeck_eyes_count_l41_41754


namespace triangle_bc_length_l41_41226

theorem triangle_bc_length :
  ∀ (A B C X : Type) (d_AB : ℝ) (d_AC : ℝ) (d_BX d_CX BC : ℕ),
  d_AB = 86 ∧ d_AC = 97 →
  let circleA := {center := A, radius := d_AB} in
  let intersect_B := B ∈ circleA in
  let intersect_X := X ∈ circleA in
  d_BX + d_CX = BC →
  d_BX ∈ ℕ ∧ d_CX ∈ ℕ →
  BC = 61 :=
by
  intros A B C X d_AB d_AC d_BX d_CX BC h_dist h_circle h_intersect h_sum h_intBC
  sorry

end triangle_bc_length_l41_41226


namespace remainder_3005_98_l41_41807

theorem remainder_3005_98 : 3005 % 98 = 65 :=
by sorry

end remainder_3005_98_l41_41807


namespace tan_product_in_triangle_l41_41505

theorem tan_product_in_triangle (A B C : ℝ) (h1 : A + B + C = Real.pi)
  (h2 : Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = Real.sin B ^ 2) :
  Real.tan A * Real.tan C = 1 :=
sorry

end tan_product_in_triangle_l41_41505


namespace bart_trees_needed_l41_41851

noncomputable def calculate_trees (firewood_per_tree : ℕ) (logs_per_day : ℕ) (days_in_period : ℕ) : ℕ :=
  (days_in_period * logs_per_day) / firewood_per_tree

theorem bart_trees_needed :
  let firewood_per_tree := 75 in
  let logs_per_day := 5 in
  let days_in_november := 30 in
  let days_in_december := 31 in
  let days_in_january := 31 in
  let days_in_february := 28 in
  let total_days := days_in_november + days_in_december + days_in_january + days_in_february in
  calculate_trees firewood_per_tree logs_per_day total_days = 8 :=
by
  sorry

end bart_trees_needed_l41_41851


namespace find_x_for_salt_solution_l41_41814

theorem find_x_for_salt_solution : ∀ (x : ℝ),
  (1 + x) * 0.10 = (x * 0.50) →
  x = 0.25 :=
by
  intros x h
  sorry

end find_x_for_salt_solution_l41_41814


namespace unique_solution_l41_41100

theorem unique_solution (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (eq1 : x * y + y * z + z * x = 12) (eq2 : x * y * z = 2 + x + y + z) :
  x = 2 ∧ y = 2 ∧ z = 2 :=
by 
  sorry

end unique_solution_l41_41100


namespace minimum_value_l41_41518

theorem minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  8 * a^3 + 27 * b^3 + 125 * c^3 + (1 / (a * b * c)) ≥ 10 * Real.sqrt 6 :=
by
  sorry

end minimum_value_l41_41518


namespace seated_ways_alice_between_bob_and_carol_l41_41069

-- Define the necessary entities and conditions for the problem.
def num_people : Nat := 7
def alice := "Alice"
def bob := "Bob"
def carol := "Carol"

-- The main theorem
theorem seated_ways_alice_between_bob_and_carol :
  ∃ (ways : Nat), ways = 48 := by
  sorry

end seated_ways_alice_between_bob_and_carol_l41_41069


namespace cost_price_of_radio_l41_41779

theorem cost_price_of_radio (SP : ℝ) (L_p : ℝ) (C : ℝ) (h₁ : SP = 3200) (h₂ : L_p = 0.28888888888888886) 
  (h₃ : SP = C - (C * L_p)) : C = 4500 :=
by
  sorry

end cost_price_of_radio_l41_41779


namespace even_decreasing_function_l41_41101

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem even_decreasing_function :
  is_even f →
  is_decreasing_on_nonneg f →
  f 1 > f (-2) ∧ f (-2) > f 3 :=
by
  sorry

end even_decreasing_function_l41_41101


namespace product_ABCD_is_9_l41_41345

noncomputable def A : ℝ := Real.sqrt 2018 + Real.sqrt 2019 + 1
noncomputable def B : ℝ := -Real.sqrt 2018 - Real.sqrt 2019 - 1
noncomputable def C : ℝ := Real.sqrt 2018 - Real.sqrt 2019 + 1
noncomputable def D : ℝ := Real.sqrt 2019 - Real.sqrt 2018 + 1

theorem product_ABCD_is_9 : A * B * C * D = 9 :=
by sorry

end product_ABCD_is_9_l41_41345


namespace average_speed_car_y_l41_41308

-- Defining the constants based on the problem conditions
def speedX : ℝ := 35
def timeDifference : ℝ := 1.2  -- This is 72 minutes converted to hours
def distanceFromStartOfY : ℝ := 294

-- Defining the main statement
theorem average_speed_car_y : 
  ( ∀ timeX timeY distanceX distanceY : ℝ, 
      timeX = timeY + timeDifference ∧
      distanceX = speedX * timeX ∧
      distanceY = distanceFromStartOfY ∧
      distanceX = distanceFromStartOfY + speedX * timeDifference
  → distanceY / timeX = 30.625) :=
sorry

end average_speed_car_y_l41_41308


namespace bart_trees_needed_l41_41850

-- Define the constants and conditions given
def firewood_per_tree : Nat := 75
def logs_burned_per_day : Nat := 5
def days_in_november : Nat := 30
def days_in_december : Nat := 31
def days_in_january : Nat := 31
def days_in_february : Nat := 28

-- Calculate the total number of days from November 1 through February 28
def total_days : Nat := days_in_november + days_in_december + days_in_january + days_in_february

-- Calculate the total number of pieces of firewood needed
def total_firewood_needed : Nat := total_days * logs_burned_per_day

-- Calculate the number of trees needed
def trees_needed : Nat := total_firewood_needed / firewood_per_tree

-- The proof statement
theorem bart_trees_needed : trees_needed = 8 := 
by
  -- Placeholder for the proof
  sorry

end bart_trees_needed_l41_41850


namespace brianne_savings_in_may_l41_41748

-- Definitions based on conditions from a)
def initial_savings_jan : ℕ := 20
def multiplier : ℕ := 3
def additional_income : ℕ := 50

-- Savings in successive months
def savings_feb : ℕ := multiplier * initial_savings_jan
def savings_mar : ℕ := multiplier * savings_feb + additional_income
def savings_apr : ℕ := multiplier * savings_mar + additional_income
def savings_may : ℕ := multiplier * savings_apr + additional_income

-- The main theorem to verify
theorem brianne_savings_in_may : savings_may = 2270 :=
sorry

end brianne_savings_in_may_l41_41748


namespace min_shirts_to_save_money_l41_41454

theorem min_shirts_to_save_money :
  let acme_cost (x : ℕ) := 75 + 12 * x
  let gamma_cost (x : ℕ) := 18 * x
  ∀ x : ℕ, acme_cost x < gamma_cost x → x ≥ 13 := 
by
  intros
  sorry

end min_shirts_to_save_money_l41_41454


namespace savings_plan_l41_41088

noncomputable def ivan_salary : ℝ := 55000
noncomputable def vasilisa_salary : ℝ := 45000
noncomputable def mother_salary_before_retirement : ℝ := 18000
noncomputable def mother_pension_after_retirement : ℝ := 10000
noncomputable def father_salary : ℝ := 20000
noncomputable def son_state_stipend : ℝ := 3000
noncomputable def son_non_state_stipend : ℝ := 15000
noncomputable def income_tax_rate : ℝ := 0.13
noncomputable def monthly_expenses : ℝ := 74000

def net_income (salary : ℝ) : ℝ := salary * (1 - income_tax_rate)

theorem savings_plan : 
  let ivan_net := net_income ivan_salary in
  let vasilisa_net := net_income vasilisa_salary in
  let mother_net_before := net_income mother_salary_before_retirement in
  let father_net := net_income father_salary in
  let son_net := son_state_stipend in
  -- Before May 1, 2018
  let total_net_before := ivan_net + vasilisa_net + mother_net_before + father_net + son_net in
  let savings_before := total_net_before - monthly_expenses in
  -- From May 1, 2018 to August 31, 2018
  let mother_net_after := mother_pension_after_retirement in
  let total_net_after := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_after := total_net_after - monthly_expenses in
  -- From September 1, 2018 for 1 year
  let son_net := son_state_stipend + net_income son_non_state_stipend in
  let total_net_future := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_future := total_net_future - monthly_expenses in
  savings_before = 49060 ∧ savings_after = 43400 ∧ savings_future = 56450 :=
by
  sorry

end savings_plan_l41_41088


namespace locus_of_M_is_ellipse_l41_41480

theorem locus_of_M_is_ellipse :
  ∀ (a b : ℝ) (M : ℝ × ℝ),
  a > b → b > 0 → (∃ x y : ℝ, 
  (M = (x, y)) ∧ 
  ∃ (P : ℝ × ℝ),
  (∃ x0 y0 : ℝ, P = (x0, y0) ∧ (x0^2 / a^2 + y0^2 / b^2 = 1)) ∧ 
  P ≠ (a, 0) ∧ P ≠ (-a, 0) ∧
  (∃ t : ℝ, t = (x^2 + y^2 - a^2) / (2 * y)) ∧ 
  (∃ x0 y0 : ℝ, 
    x0 = -x ∧ 
    y0 = 2 * t - y ∧
    x0^2 / a^2 + y0^2 / b^2 = 1)) →
  ∃ (x y : ℝ),
  M = (x, y) ∧ 
  (x^2 / a^2 + y^2 / (a^4 / b^2) = 1) := 
sorry

end locus_of_M_is_ellipse_l41_41480


namespace fg_of_neg2_l41_41882

def f (x : ℤ) : ℤ := x^2
def g (x : ℤ) : ℤ := 2 * x + 5

theorem fg_of_neg2 : f (g (-2)) = 1 := by
  sorry

end fg_of_neg2_l41_41882


namespace prob_score_at_most_7_l41_41213

-- Definitions based on the conditions
def prob_10_ring : ℝ := 0.15
def prob_9_ring : ℝ := 0.35
def prob_8_ring : ℝ := 0.2
def prob_7_ring : ℝ := 0.1

-- Define the event of scoring no more than 7
def score_at_most_7 := prob_7_ring

-- Theorem statement
theorem prob_score_at_most_7 : score_at_most_7 = 0.1 := by 
  -- proof goes here
  sorry

end prob_score_at_most_7_l41_41213


namespace valid_integer_values_n_l41_41859

def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

theorem valid_integer_values_n : ∃ (n_values : ℕ), n_values = 3 ∧
  ∀ n : ℤ, is_integer (3200 * (2 / 5) ^ (2 * n)) ↔ n = 0 ∨ n = 1 ∨ n = 2 :=
by
  sorry

end valid_integer_values_n_l41_41859


namespace total_apples_eaten_l41_41305

def Apples_Tuesday : ℕ := 4
def Apples_Wednesday : ℕ := 2 * Apples_Tuesday
def Apples_Thursday : ℕ := Apples_Tuesday / 2

theorem total_apples_eaten : Apples_Tuesday + Apples_Wednesday + Apples_Thursday = 14 := by
  sorry

end total_apples_eaten_l41_41305


namespace isosceles_triangle_perimeter_l41_41455

theorem isosceles_triangle_perimeter
  (a b c : ℝ )
  (ha : a = 20)
  (hb : b = 20)
  (hc : c = (2/5) * 20)
  (triangle_ineq1 : a ≤ b + c)
  (triangle_ineq2 : b ≤ a + c)
  (triangle_ineq3 : c ≤ a + b) :
  a + b + c = 48 := by
  sorry

end isosceles_triangle_perimeter_l41_41455


namespace largest_sampled_item_l41_41264

theorem largest_sampled_item (n : ℕ) (m : ℕ) (a : ℕ) (k : ℕ)
  (hn : n = 360)
  (hm : m = 30)
  (hk : k = n / m)
  (ha : a = 105) :
  ∃ b, b = 433 ∧ (∃ i, i < m ∧ a = 1 + i * k) → (∃ j, j < m ∧ b = 1 + j * k) :=
by
  sorry

end largest_sampled_item_l41_41264


namespace find_f_three_l41_41186

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def f_condition (f : ℝ → ℝ) := ∀ x : ℝ, x < 0 → f x = (1/2)^x

theorem find_f_three (f : ℝ → ℝ) (h₁ : odd_function f) (h₂ : f_condition f) : f 3 = -8 :=
sorry

end find_f_three_l41_41186


namespace algebraic_identity_l41_41058

theorem algebraic_identity (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) :
    a^2 - b^2 = -8 := by
  sorry

end algebraic_identity_l41_41058


namespace derivative_at_1_l41_41177

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_at_1 : (deriv f 1) = 2 * Real.log 2 - 3 := 
sorry

end derivative_at_1_l41_41177


namespace total_amount_spent_by_jim_is_50_l41_41064

-- Definitions for conditions
def cost_per_gallon_nc : ℝ := 2.00  -- Cost per gallon in North Carolina
def gallons_nc : ℕ := 10  -- Gallons bought in North Carolina
def additional_cost_per_gallon_va : ℝ := 1.00  -- Additional cost per gallon in Virginia
def gallons_va : ℕ := 10  -- Gallons bought in Virginia

-- Definition for total cost in North Carolina
def total_cost_nc : ℝ := gallons_nc * cost_per_gallon_nc

-- Definition for cost per gallon in Virginia
def cost_per_gallon_va : ℝ := cost_per_gallon_nc + additional_cost_per_gallon_va

-- Definition for total cost in Virginia
def total_cost_va : ℝ := gallons_va * cost_per_gallon_va

-- Definition for total amount spent
def total_spent : ℝ := total_cost_nc + total_cost_va

-- Theorem to prove
theorem total_amount_spent_by_jim_is_50 : total_spent = 50.00 :=
by
  -- Place proof here
  sorry

end total_amount_spent_by_jim_is_50_l41_41064


namespace combination_8_5_l41_41001

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l41_41001


namespace Jean_spots_l41_41509

/--
Jean the jaguar has a total of 60 spots.
Half of her spots are located on her upper torso.
One-third of the spots are located on her back and hindquarters.
Jean has 30 spots on her upper torso.
Prove that Jean has 10 spots located on her sides.
-/
theorem Jean_spots (TotalSpots UpperTorsoSpots BackHindquartersSpots SidesSpots : ℕ)
  (h_half : UpperTorsoSpots = TotalSpots / 2)
  (h_back : BackHindquartersSpots = TotalSpots / 3)
  (h_total_upper : UpperTorsoSpots = 30)
  (h_total : TotalSpots = 60) :
  SidesSpots = 10 :=
by
  sorry

end Jean_spots_l41_41509


namespace max_third_side_length_l41_41788

theorem max_third_side_length (x : ℕ) (h1 : 28 + x > 47) (h2 : 47 + x > 28) (h3 : 28 + 47 > x) :
  x = 74 :=
sorry

end max_third_side_length_l41_41788


namespace base_8_digits_sum_l41_41742

theorem base_8_digits_sum
    (X Y Z : ℕ)
    (h1 : 1 ≤ X ∧ X < 8)
    (h2 : 1 ≤ Y ∧ Y < 8)
    (h3 : 1 ≤ Z ∧ Z < 8)
    (h4 : X ≠ Y)
    (h5 : Y ≠ Z)
    (h6 : Z ≠ X)
    (h7 : 8^2 * X + 8 * Y + Z + 8^2 * Y + 8 * Z + X + 8^2 * Z + 8 * X + Y = 8^3 * X + 8^2 * X + 8 * X) :
  Y + Z = 7 * X :=
by
  sorry

end base_8_digits_sum_l41_41742


namespace sum_odd_divisors_l41_41962

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l41_41962


namespace compute_58_sq_pattern_l41_41371

theorem compute_58_sq_pattern : (58 * 58 = 56 * 60 + 4) :=
by
  sorry

end compute_58_sq_pattern_l41_41371


namespace domain_of_function_l41_41411

theorem domain_of_function :
  (∀ x : ℝ, 2 + x ≥ 0 ∧ 3 - x ≥ 0 ↔ -2 ≤ x ∧ x ≤ 3) :=
by sorry

end domain_of_function_l41_41411


namespace proof_problem_l41_41348

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | |x| > 1}

def B : Set ℝ := {x | (0 : ℝ) < x ∧ x ≤ 2}

def complement_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

def intersection (s1 s2 : Set ℝ) : Set ℝ := s1 ∩ s2

theorem proof_problem : (complement_A ∩ B) = {x | 0 < x ∧ x ≤ 1} :=
by {
  sorry
}

end proof_problem_l41_41348


namespace describe_graph_of_equation_l41_41435

theorem describe_graph_of_equation :
  (∀ x y : ℝ, (x + y)^3 = x^3 + y^3 → (x = 0 ∨ y = 0 ∨ y = -x)) :=
by
  intros x y h
  sorry

end describe_graph_of_equation_l41_41435


namespace greatest_root_of_g_l41_41170

noncomputable def g (x : ℝ) : ℝ := 16 * x^4 - 20 * x^2 + 5

theorem greatest_root_of_g :
  ∃ r : ℝ, r = Real.sqrt 5 / 2 ∧ (forall x, g x ≤ g r) :=
sorry

end greatest_root_of_g_l41_41170


namespace extreme_value_of_f_range_of_a_l41_41872

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 1

theorem extreme_value_of_f (a : ℝ) (ha : 0 < a) : ∃ x, f x a = a - a * Real.log a - 1 :=
sorry

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 ∧ 0 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 a = f x2 a ∧ abs (x1 - x2) ≥ 1 ) →
  (e - 1 < a ∧ a < Real.exp 2 - Real.exp 1) :=
sorry

end extreme_value_of_f_range_of_a_l41_41872


namespace minimum_cost_for_13_bottles_l41_41977

def cost_per_bottle_shop_A := 200 -- in cents
def discount_shop_B := 15 / 100 -- discount
def promotion_B_threshold := 4
def promotion_A_threshold := 4

-- Function to calculate the cost in Shop A for given number of bottles
def shop_A_cost (bottles : ℕ) : ℕ :=
  let batches := bottles / 5
  let remainder := bottles % 5
  (batches * 4 + remainder) * cost_per_bottle_shop_A

-- Function to calculate the cost in Shop B for given number of bottles
def shop_B_cost (bottles : ℕ) : ℕ :=
  if bottles >= promotion_B_threshold then
    (bottles * cost_per_bottle_shop_A) * (1 - discount_shop_B)
  else
    bottles * cost_per_bottle_shop_A

-- Function to calculate combined cost for given numbers of bottles from Shops A and B
def combined_cost (bottles_A bottles_B : ℕ) : ℕ :=
  shop_A_cost bottles_A + shop_B_cost bottles_B

theorem minimum_cost_for_13_bottles : ∃ a b, a + b = 13 ∧ combined_cost a b = 2000 := 
sorry

end minimum_cost_for_13_bottles_l41_41977


namespace tangent_line_at_P_exists_c_for_a_l41_41189

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_P :
  ∀ x y : ℝ, y = f x → x = 1 → y = 0 → x - y - 1 = 0 := 
by 
  sorry

theorem exists_c_for_a :
  ∀ a : ℝ, 1 < a → ∃ c : ℝ, 0 < c ∧ c < 1 / a ∧ ∀ x : ℝ, c < x → x < 1 → f x > a * x * (x - 1) :=
by 
  sorry

end tangent_line_at_P_exists_c_for_a_l41_41189


namespace problem_correctness_l41_41677

theorem problem_correctness :
  ∀ (x y a b : ℝ), (-3:ℝ)^2 ≠ -9 ∧
    - (x + y) = -x - y ∧
    3 * a + 5 * b ≠ 8 * a * b ∧
    5 * a^3 * b^2 - 3 * a^3 * b^2 = 2 * a^3 * b^2 := 
by
  intro x y a b
  split
  · norm_num
  split
  · ring
  split
  · linarith
  · ring

end problem_correctness_l41_41677


namespace find_a_plus_b_l41_41743

theorem find_a_plus_b (x a b : ℝ) (ha : x = a + Real.sqrt b)
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : x^2 + 5 * x + 4/x + 1/(x^2) = 34) : a + b = 5 :=
sorry

end find_a_plus_b_l41_41743


namespace triangle_area_right_angled_l41_41527

theorem triangle_area_right_angled (a : ℝ) (h₁ : 0 < a) (h₂ : a < 24) :
  let b := 24
  let c := 48 - a
  (a^2 + b^2 = c^2) → (1/2) * a * b = 216 :=
by
  sorry

end triangle_area_right_angled_l41_41527


namespace expand_product_l41_41852

theorem expand_product :
  (3 * x + 4) * (x - 2) * (x + 6) = 3 * x^3 + 16 * x^2 - 20 * x - 48 :=
by
  sorry

end expand_product_l41_41852


namespace range_of_a_for_function_is_real_l41_41618

noncomputable def quadratic_expr (a x : ℝ) : ℝ := a * x^2 - 4 * x + a - 3

theorem range_of_a_for_function_is_real :
  (∀ x : ℝ, quadratic_expr a x > 0) → 0 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_for_function_is_real_l41_41618


namespace number_of_other_numbers_l41_41531

-- Definitions of the conditions
def avg_five_numbers (S : ℕ) : Prop := S / 5 = 20
def sum_three_numbers (S2 : ℕ) : Prop := 100 = S2 + 48
def avg_other_numbers (N S2 : ℕ) : Prop := S2 / N = 26

-- Theorem statement
theorem number_of_other_numbers (S S2 N : ℕ) 
  (h1 : avg_five_numbers S) 
  (h2 : sum_three_numbers S2) 
  (h3 : avg_other_numbers N S2) : 
  N = 2 := 
  sorry

end number_of_other_numbers_l41_41531


namespace largest_common_value_less_than_1000_l41_41660

def arithmetic_sequence_1 (n : ℕ) : ℕ := 2 + 3 * n
def arithmetic_sequence_2 (m : ℕ) : ℕ := 4 + 8 * m

theorem largest_common_value_less_than_1000 :
  ∃ a n m : ℕ, a = arithmetic_sequence_1 n ∧ a = arithmetic_sequence_2 m ∧ a < 1000 ∧ a = 980 :=
by { sorry }

end largest_common_value_less_than_1000_l41_41660


namespace transmitter_finding_probability_l41_41290

/-- 
  A license plate in the country Kerrania consists of 4 digits followed by two letters.
  The letters A, B, and C are used only by government vehicles while the letters D through Z are used by non-government vehicles.
  Kerrania's intelligence agency has recently captured a message from the country Gonzalia indicating that an electronic transmitter 
  has been installed in a Kerrania government vehicle with a license plate starting with 79. 
  In addition, the message reveals that the last three digits of the license plate form a palindromic sequence (meaning that they are 
  the same forward and backward), and the second digit is either a 3 or a 5. 
  If it takes the police 10 minutes to inspect each vehicle, what is the probability that the police will find the transmitter 
  within 3 hours, considering the additional restrictions on the possible license plate combinations?
-/
theorem transmitter_finding_probability :
  0.1 = 18 / 180 :=
by
  sorry

end transmitter_finding_probability_l41_41290


namespace find_k_l41_41039

noncomputable def S (n : ℕ) : ℤ := n^2 - 8 * n
noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem find_k (k : ℕ) (h : a k = 5) : k = 7 := by
  sorry

end find_k_l41_41039


namespace determine_a_l41_41486

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if h : x = 3 then a else 2 / |x - 3|

theorem determine_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ 3 ∧ x2 ≠ 3 ∧ (f x1 a - 4 = 0) ∧ (f x2 a - 4 = 0) ∧ f 3 a - 4 = 0) →
  a = 4 :=
by
  sorry

end determine_a_l41_41486


namespace point_in_fourth_quadrant_l41_41340

theorem point_in_fourth_quadrant (m : ℝ) (h1 : m + 2 > 0) (h2 : m < 0) : -2 < m ∧ m < 0 := by
  sorry

end point_in_fourth_quadrant_l41_41340


namespace bob_first_six_probability_l41_41573

noncomputable def probability_bob_first_six (p : ℚ) : ℚ :=
  (1 - p) * p / (1 - ( (1 - p) * (1 - p)))

theorem bob_first_six_probability :
  probability_bob_first_six (1/6) = 5/11 :=
by
  sorry

end bob_first_six_probability_l41_41573


namespace find_solution_set_l41_41469

noncomputable def is_solution (x : ℝ) : Prop :=
(1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < 1 / 4

theorem find_solution_set :
  { x : ℝ | is_solution x } = { x : ℝ | x < -2 } ∪ { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end find_solution_set_l41_41469


namespace find_larger_number_l41_41797

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 := by
  sorry

end find_larger_number_l41_41797


namespace fraction_of_40_l41_41946

theorem fraction_of_40 : (3 / 4) * 40 = 30 :=
by
  -- We'll add the 'sorry' here to indicate that this is the proof part which is not required.
  sorry

end fraction_of_40_l41_41946


namespace smallest_b_of_factored_quadratic_l41_41321

theorem smallest_b_of_factored_quadratic (r s : ℕ) (h1 : r * s = 1620) : (r + s) = 84 :=
sorry

end smallest_b_of_factored_quadratic_l41_41321


namespace area_of_garden_l41_41659

variable (P : ℝ) (A : ℝ)

theorem area_of_garden (hP : P = 38) (hA : A = 2 * P + 14.25) : A = 90.25 :=
by
  sorry

end area_of_garden_l41_41659


namespace f_odd_and_solution_set_l41_41865

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2
  else if x < 0 then - log (-x) / log 2
  else 0

theorem f_odd_and_solution_set :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ 
  {x : ℝ | f (x) ≤ 1 / 2} = {x : ℝ | x ≤ -sqrt 2 / 2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ sqrt 2} :=
by 
  -- Proof goes here
  sorry

end f_odd_and_solution_set_l41_41865


namespace more_karabases_than_barabases_l41_41752

/-- In the fairy-tale land of Perra-Terra, each Karabas is acquainted with nine Barabases, 
    and each Barabas is acquainted with ten Karabases. We aim to prove that there are more Karabases than Barabases. -/
theorem more_karabases_than_barabases (K B : ℕ) (h1 : 9 * K = 10 * B) : K > B := 
by {
    -- Following the conditions and conclusion
    sorry
}

end more_karabases_than_barabases_l41_41752


namespace complex_problem_l41_41731

theorem complex_problem 
  (a : ℝ) 
  (ha : a^2 - 9 = 0) :
  (a + (Complex.I ^ 19)) / (1 + Complex.I) = 1 - 2 * Complex.I := by
  sorry

end complex_problem_l41_41731


namespace even_function_l41_41810

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ := (x + 2)^2 + (2 * x - 1)^2

theorem even_function : is_even_function f :=
by
  sorry

end even_function_l41_41810


namespace probability_closer_to_5_than_1_l41_41988

open MeasureTheory ProbabilityTheory

noncomputable def prob_closer_to_5_than_1 : ℝ :=
  let interval : Set ℝ := Icc 0 6
  let closer_to_5 := {x | 3 ≤ x ∧ x ≤ 6}
  let p := measure_theory.volume
  (p closer_to_5 / p interval).to_real

theorem probability_closer_to_5_than_1 {prob_closer_to_5_than_1 = 0.5 : ℝ} : 
  by 
    let interval : Set ℝ := Icc 0 6
    let closer_to_5 := {x | 3 ≤ x ∧ x ≤ 6}
    let p := measure_theory.volume
    exact (p closer_to_5 / p interval).to_real = 0.5
  sorry

end probability_closer_to_5_than_1_l41_41988


namespace prism_height_relation_l41_41041

theorem prism_height_relation (a b c h : ℝ) 
  (h_perp : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_height : 0 < h) 
  (h_right_angles : true) :
  1 / h^2 = 1 / a^2 + 1 / b^2 + 1 / c^2 :=
by 
  sorry 

end prism_height_relation_l41_41041


namespace smallest_four_digit_divisible_by_43_l41_41270

theorem smallest_four_digit_divisible_by_43 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 43 = 0 ∧ n = 1032 :=
by
  use 1032
  split
  { linarith }
  split
  { linarith }
  split
  { norm_num }
  norm_num

end smallest_four_digit_divisible_by_43_l41_41270


namespace smallest_N_conditions_l41_41293

theorem smallest_N_conditions:
  ∃N : ℕ, N % 9 = 8 ∧
           N % 8 = 7 ∧
           N % 7 = 6 ∧
           N % 6 = 5 ∧
           N % 5 = 4 ∧
           N % 4 = 3 ∧
           N % 3 = 2 ∧
           N % 2 = 1 ∧
           N = 2519 :=
sorry

end smallest_N_conditions_l41_41293


namespace sumOddDivisorsOf90_l41_41952

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l41_41952


namespace probability_of_consecutive_draws_l41_41285

-- Assume chips and their counts are represented as variables for clarity
def red_chips : ℕ := 4
def green_chips : ℕ := 3
def blue_chips : ℕ := 5
def total_chips : ℕ := red_chips + green_chips + blue_chips

-- factorial calculations
def fact (n : ℕ) : ℕ := Nat.factorial n
def favorable_outcomes : ℕ := fact red_chips * fact green_chips * fact blue_chips * fact 3
def total_outcomes : ℕ := fact total_chips

theorem probability_of_consecutive_draws : 
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 4620 := 
by
  sorry

end probability_of_consecutive_draws_l41_41285


namespace max_third_term_is_16_l41_41122

-- Define the arithmetic sequence conditions
def arithmetic_seq (a d : ℕ) : list ℕ := [a, a + d, a + 2 * d, a + 3 * d]

-- Define the sum condition
def sum_of_sequence_is_50 (a d : ℕ) : Prop :=
  (a + a + d + a + 2 * d + a + 3 * d) = 50

-- Define the third term of the sequence
def third_term (a d : ℕ) : ℕ := a + 2 * d

-- Prove that the greatest possible third term is 16
theorem max_third_term_is_16 : ∃ (a d : ℕ), sum_of_sequence_is_50 a d ∧ third_term a d = 16 :=
by
  sorry

end max_third_term_is_16_l41_41122


namespace delta_value_l41_41879

theorem delta_value : ∃ Δ : ℤ, 4 * (-3) = Δ + 5 ∧ Δ = -17 := 
by
  use -17
  sorry

end delta_value_l41_41879


namespace intersect_count_l41_41926

noncomputable def f (x : ℝ) : ℝ := sorry  -- Function f defined for all real x.
noncomputable def f_inv (y : ℝ) : ℝ := sorry  -- Inverse function of f.

theorem intersect_count : 
  (∃ a b : ℝ, a ≠ b ∧ f (a^2) = f (a^3) ∧ f (b^2) = f (b^3)) :=
by sorry

end intersect_count_l41_41926


namespace num_possible_lists_l41_41266

theorem num_possible_lists :
  let binA_balls := 8
  let binB_balls := 5
  let total_lists := binA_balls * binB_balls
  total_lists = 40 := by
{
  let binA_balls := 8
  let binB_balls := 5
  let total_lists := binA_balls * binB_balls
  show total_lists = 40
  exact rfl
}

end num_possible_lists_l41_41266


namespace combination_8_5_l41_41002

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l41_41002


namespace binom_8_5_l41_41009

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the binomial coefficient function
def binom (n k : ℕ) := fact n / (fact k * fact (n - k))

-- State the theorem to prove binom 8 5 = 56
theorem binom_8_5 : binom 8 5 = 56 := by
  sorry

end binom_8_5_l41_41009


namespace moles_of_NaCl_l41_41590

def moles_of_reactants (NaCl KNO3 NaNO3 KCl : ℕ) : Prop :=
  NaCl + KNO3 = NaNO3 + KCl

theorem moles_of_NaCl (NaCl KNO3 NaNO3 KCl : ℕ) 
  (h : moles_of_reactants NaCl KNO3 NaNO3 KCl) 
  (h2 : KNO3 = 1)
  (h3 : NaNO3 = 1) :
  NaCl = 1 :=
by
  sorry

end moles_of_NaCl_l41_41590


namespace minimum_perimeter_rectangle_l41_41173

theorem minimum_perimeter_rectangle (S : ℝ) (hS : S > 0) :
  ∃ x y : ℝ, (x * y = S) ∧ (∀ u v : ℝ, (u * v = S) → (2 * (u + v) ≥ 4 * Real.sqrt S)) ∧ (x = Real.sqrt S ∧ y = Real.sqrt S) :=
by
  sorry

end minimum_perimeter_rectangle_l41_41173


namespace find_a_l41_41908

variables {a b c : ℂ}

-- Given conditions
variables (h1 : a + b + c = 5) 
variables (h2 : a * b + b * c + c * a = 5) 
variables (h3 : a * b * c = 5)
variables (h4 : a.im = 0) -- a is real

theorem find_a : a = 4 :=
by
  sorry

end find_a_l41_41908


namespace Mr_A_Mrs_A_are_normal_l41_41555

def is_knight (person : Type) : Prop := sorry
def is_liar (person : Type) : Prop := sorry
def is_normal (person : Type) : Prop := sorry

variable (Mr_A Mrs_A : Type)

axiom Mr_A_statement : is_normal Mrs_A → False
axiom Mrs_A_statement : is_normal Mr_A → False

theorem Mr_A_Mrs_A_are_normal :
  is_normal Mr_A ∧ is_normal Mrs_A :=
sorry

end Mr_A_Mrs_A_are_normal_l41_41555


namespace domain_of_function_l41_41412

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + 1 / Real.sqrt (2 - x^2)

theorem domain_of_function : 
  {x : ℝ | x > -1 ∧ x < Real.sqrt 2} = {x : ℝ | x ∈ Set.Ioo (-1) (Real.sqrt 2)} :=
by
  sorry

end domain_of_function_l41_41412


namespace smallest_k_l41_41541

theorem smallest_k (k : ℕ) (h : 201 ≡ 9 [MOD 24]) : k = 1 := by
  sorry

end smallest_k_l41_41541


namespace fraction_value_l41_41744

theorem fraction_value (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (1 / (y : ℚ) / (1 / (x : ℚ))) = 3 / 4 :=
by
  rw [hx, hy]
  norm_num

end fraction_value_l41_41744


namespace range_of_z_l41_41178

theorem range_of_z (α β : ℝ) (z : ℝ) (h1 : -2 < α) (h2 : α ≤ 3) (h3 : 2 < β) (h4 : β ≤ 4) (h5 : z = 2 * α - (1 / 2) * β) :
  -6 < z ∧ z < 5 :=
by
  sorry

end range_of_z_l41_41178


namespace extra_bananas_each_child_gets_l41_41394

-- Define the total number of students and the number of absent students
def total_students : ℕ := 260
def absent_students : ℕ := 130

-- Define the total number of bananas
variable (B : ℕ)

-- The proof statement
theorem extra_bananas_each_child_gets :
  ∀ B : ℕ, (B / (total_students - absent_students)) = (B / total_students) + (B / total_students) :=
by
  intro B
  sorry

end extra_bananas_each_child_gets_l41_41394


namespace smallest_two_digit_multiple_of_3_l41_41136

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n <= 99
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

theorem smallest_two_digit_multiple_of_3 : ∃ n : ℕ, is_two_digit n ∧ is_multiple_of_3 n ∧ ∀ m : ℕ, is_two_digit m ∧ is_multiple_of_3 m → n <= m :=
sorry

end smallest_two_digit_multiple_of_3_l41_41136


namespace ratio_of_boys_in_class_l41_41397

noncomputable def boy_to_total_ratio (p_boy p_girl : ℚ) : ℚ :=
p_boy / (p_boy + p_girl)

theorem ratio_of_boys_in_class (p_boy p_girl total_students : ℚ)
    (h1 : p_boy = (3/4) * p_girl)
    (h2 : p_boy + p_girl = 1)
    (h3 : total_students = 1) :
    boy_to_total_ratio p_boy p_girl = 3/7 :=
by
  sorry

end ratio_of_boys_in_class_l41_41397


namespace polynomial_root_sum_eq_48_l41_41257

theorem polynomial_root_sum_eq_48 {r s t : ℕ} (h1 : r * s * t = 2310) 
  (h2 : r > 0) (h3 : s > 0) (h4 : t > 0) : r + s + t = 48 :=
sorry

end polynomial_root_sum_eq_48_l41_41257


namespace color_drawing_cost_l41_41637

theorem color_drawing_cost (cost_bw : ℕ) (surcharge_ratio : ℚ) (cost_color : ℕ) :
  cost_bw = 160 →
  surcharge_ratio = 0.50 →
  cost_color = cost_bw + (surcharge_ratio * cost_bw : ℚ).natAbs →
  cost_color = 240 :=
by
  intros h_bw h_surcharge h_color
  rw [h_bw, h_surcharge] at h_color
  exact h_color

end color_drawing_cost_l41_41637


namespace workout_days_l41_41465

theorem workout_days (n : ℕ) (squats : ℕ → ℕ) 
  (h1 : squats 1 = 30)
  (h2 : ∀ k, squats (k + 1) = squats k + 5)
  (h3 : squats 4 = 45) :
  n = 4 :=
sorry

end workout_days_l41_41465


namespace original_cost_price_l41_41562

theorem original_cost_price (C : ℝ) 
  (h1 : 0.87 * C > 0) 
  (h2 : 1.2 * (0.87 * C) = 54000) : 
  C = 51724.14 :=
by
  sorry

end original_cost_price_l41_41562


namespace find_alpha_l41_41377

noncomputable def parametric_eq_line (α t : Real) : Real × Real :=
  (1 + t * Real.cos α, t * Real.sin α)

def cartesian_eq_curve (x y : Real) : Prop :=
  y^2 = 4 * x

def intersection_condition (α t₁ t₂ : Real) : Prop :=
  Real.sin α ≠ 0 ∧ 
  (1 + t₁ * Real.cos α, t₁ * Real.sin α) = (1 + t₂ * Real.cos α, t₂ * Real.sin α) ∧ 
  Real.sqrt ((t₁ + t₂)^2 - 4 * (-4 / (Real.sin α)^2)) = 8

theorem find_alpha (α : Real) (t₁ t₂ : Real) 
  (h1: 0 < α) (h2: α < π) (h3: intersection_condition α t₁ t₂) : 
  α = π/4 ∨ α = 3*π/4 :=
by 
  sorry

end find_alpha_l41_41377


namespace Tim_has_7_times_more_l41_41845

-- Define the number of Dan's violet balloons
def Dan_violet_balloons : ℕ := 29

-- Define the number of Tim's violet balloons
def Tim_violet_balloons : ℕ := 203

-- Prove that the ratio of Tim's balloons to Dan's balloons is 7
theorem Tim_has_7_times_more (h : Tim_violet_balloons = 7 * Dan_violet_balloons) : 
  Tim_violet_balloons = 7 * Dan_violet_balloons := 
by {
  sorry
}

end Tim_has_7_times_more_l41_41845


namespace prop_P_subset_q_when_m_eq_1_range_m_for_necessity_and_not_sufficiency_l41_41601

theorem prop_P_subset_q_when_m_eq_1 :
  ∀ x : ℝ, ∀ m : ℝ, m = 1 → (x ∈ {x | -2 ≤ x ∧ x ≤ 10}) ↔ (x ∈ {x | 0 ≤ x ∧ x ≤ 2}) := 
by sorry

theorem range_m_for_necessity_and_not_sufficiency :
  ∀ m : ℝ, (∀ x : ℝ, (x ∈ {x | -2 ≤ x ∧ x ≤ 10}) → (x ∈ {x | 1 - m ≤ x ∧ x ≤ 1 + m})) ↔ (m ≥ 9) := 
by sorry

end prop_P_subset_q_when_m_eq_1_range_m_for_necessity_and_not_sufficiency_l41_41601


namespace andrea_reaches_lauren_in_25_minutes_l41_41576

noncomputable def initial_distance : ℝ := 30
noncomputable def decrease_rate : ℝ := 90
noncomputable def Lauren_stop_time : ℝ := 10 / 60

theorem andrea_reaches_lauren_in_25_minutes :
  ∃ v_L v_A : ℝ, v_A = 2 * v_L ∧ v_A + v_L = decrease_rate ∧ ∃ remaining_distance remaining_time final_time : ℝ, 
  remaining_distance = initial_distance - decrease_rate * Lauren_stop_time ∧ 
  remaining_time = remaining_distance / v_A ∧ 
  final_time = Lauren_stop_time + remaining_time ∧ 
  final_time * 60 = 25 :=
sorry

end andrea_reaches_lauren_in_25_minutes_l41_41576


namespace iron_ii_sulfate_moles_l41_41582

/-- Given the balanced chemical equation for the reaction between iron (Fe) and sulfuric acid (H2SO4)
    to form Iron (II) sulfate (FeSO4) and hydrogen gas (H2) and the 1:1 molar ratio between iron and
    sulfuric acid, determine the number of moles of Iron (II) sulfate formed when 3 moles of Iron and
    2 moles of Sulfuric acid are combined. This is a limiting reactant problem with the final 
    product being 2 moles of Iron (II) sulfate (FeSO4). -/
theorem iron_ii_sulfate_moles (Fe moles_H2SO4 : Nat) (reaction_ratio : Nat) (FeSO4 moles_formed : Nat) :
  Fe = 3 → moles_H2SO4 = 2 → reaction_ratio = 1 → moles_formed = 2 :=
by
  intros hFe hH2SO4 hRatio
  apply sorry

end iron_ii_sulfate_moles_l41_41582


namespace original_price_before_discounts_l41_41835

theorem original_price_before_discounts (P : ℝ) (h : 0.684 * P = 6840) : P = 10000 :=
by
  sorry

end original_price_before_discounts_l41_41835


namespace combined_age_l41_41993

-- Conditions as definitions
def AmyAge (j : ℕ) : ℕ :=
  j / 3

def ChrisAge (a : ℕ) : ℕ :=
  2 * a

-- Given condition
def JeremyAge : ℕ := 66

-- Question to prove
theorem combined_age : 
  let j := JeremyAge
  let a := AmyAge j
  let c := ChrisAge a
  a + j + c = 132 :=
by
  sorry

end combined_age_l41_41993


namespace total_books_l41_41767

def number_of_zoology_books : ℕ := 16
def number_of_botany_books : ℕ := 4 * number_of_zoology_books

theorem total_books : number_of_zoology_books + number_of_botany_books = 80 := by
  sorry

end total_books_l41_41767


namespace shop_earnings_correct_l41_41628

theorem shop_earnings_correct :
  let cola_price := 3
  let juice_price := 1.5
  let water_price := 1
  let cola_sold := 15
  let juice_sold := 12
  let water_sold := 25
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold = 88 := 
by 
  sorry

end shop_earnings_correct_l41_41628


namespace unique_two_scoop_sundaes_l41_41159

theorem unique_two_scoop_sundaes (n : ℕ) (h : n = 8) : 
  (n.choose 2 + n) = 36 :=
by
  rw h
  -- The proof goes here
  sorry

end unique_two_scoop_sundaes_l41_41159


namespace domain_of_f_l41_41782

noncomputable def domain_f : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem domain_of_f : domain_f = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end domain_of_f_l41_41782


namespace country_X_tax_l41_41889

theorem country_X_tax (I T x : ℝ) (hI : I = 51999.99) (hT : T = 8000) (h : T = 0.14 * x + 0.20 * (I - x)) : 
  x = 39999.97 := sorry

end country_X_tax_l41_41889


namespace a6_value_l41_41346

theorem a6_value
  (a : ℕ → ℤ)
  (h1 : a 2 = 3)
  (h2 : a 4 = 15)
  (geo : ∃ q : ℤ, ∀ n : ℕ, n > 0 → a (n + 1) = q^n * (a 1 + 1) - 1):
  a 6 = 63 :=
by
  sorry

end a6_value_l41_41346


namespace triangle_side_a_l41_41063

theorem triangle_side_a (a : ℝ) : 2 < a ∧ a < 8 → a = 7 :=
by
  sorry

end triangle_side_a_l41_41063


namespace bronze_needed_l41_41082

/-- 
The total amount of bronze Martin needs for three bells in pounds.
-/
theorem bronze_needed (w1 w2 w3 : ℕ) 
  (h1 : w1 = 50) 
  (h2 : w2 = 2 * w1) 
  (h3 : w3 = 4 * w2) 
  : (w1 + w2 + w3 = 550) := 
by { 
  sorry 
}

end bronze_needed_l41_41082


namespace prob_A_C_not_third_day_l41_41620

-- Definitions of the problem
def people := {A, B, C}
def days := {1, 2, 3}

-- Definition of the random assignment
def duty_assignments := {f : people → days // function.bijective f}

-- Definition of the event of interest: A and C not on duty on the 3rd day
def A_C_not_on_duty_third_day (f : people → days) := (f A ≠ 3 ∧ f C ≠ 3)

-- Statement of the problem: Prove that the probability that both A and C are not on duty on the third day is 1/3
theorem prob_A_C_not_third_day : 
  (probability (λ f : {f // function.bijective f}, A_C_not_on_duty_third_day f.val)) = 1/3 := 
sorry

end prob_A_C_not_third_day_l41_41620


namespace first_movie_series_seasons_l41_41580

theorem first_movie_series_seasons (S : ℕ) : 
  (∀ E : ℕ, E = 16) → 
  (∀ L : ℕ, L = 2) → 
  (∀ T : ℕ, T = 364) → 
  (∀ second_series_seasons : ℕ, second_series_seasons = 14) → 
  (∀ second_series_remaining : ℕ, second_series_remaining = second_series_seasons * (E - L)) → 
  (E - L = 14) → 
  (second_series_remaining = 196) → 
  (T - second_series_remaining = S * (E - L)) → 
  S = 12 :=
by 
  intros E_16 L_2 T_364 second_series_14 second_series_remaining_196 E_L second_series_total_episodes remaining_episodes
  sorry

end first_movie_series_seasons_l41_41580


namespace quintuple_sum_not_less_than_l41_41585

theorem quintuple_sum_not_less_than (a : ℝ) : 5 * (a + 3) ≥ 6 :=
by
  -- Insert proof here
  sorry

end quintuple_sum_not_less_than_l41_41585


namespace complete_the_square_b_l41_41972

theorem complete_the_square_b (x : ℝ) : (x ^ 2 - 6 * x + 7 = 0) → ∃ a b : ℝ, (x + a) ^ 2 = b ∧ b = 2 :=
by
sorry

end complete_the_square_b_l41_41972


namespace max_area_square_pen_l41_41598

theorem max_area_square_pen (P : ℝ) (h1 : P = 64) : ∃ A : ℝ, A = 256 := 
by
  sorry

end max_area_square_pen_l41_41598


namespace find_number_l41_41976

theorem find_number (x : ℤ) (h : 4 * x - 7 = 13) : x = 5 := 
sorry

end find_number_l41_41976


namespace full_batches_needed_l41_41703

def students : Nat := 150
def cookies_per_student : Nat := 3
def cookies_per_batch : Nat := 20
def attendance_rate : Rat := 0.70

theorem full_batches_needed : 
  let attendees := (students : Rat) * attendance_rate
  let total_cookies_needed := attendees * (cookies_per_student : Rat)
  let batches_needed := total_cookies_needed / (cookies_per_batch : Rat)
  batches_needed.ceil = 16 :=
by
  sorry

end full_batches_needed_l41_41703


namespace profit_percentage_l41_41503

theorem profit_percentage (C S : ℝ) (h1 : C > 0) (h2 : S > 0)
  (h3 : S - 1.25 * C = 0.7023809523809523 * S) :
  ((S - C) / C) * 100 = 320 := by
sorry

end profit_percentage_l41_41503


namespace perp_case_parallel_distance_l41_41349

open Real

-- Define the line equations
def l1 (x y : ℝ) := 2 * x + y + 4 = 0
def l2 (a x y : ℝ) := a * x + 4 * y + 1 = 0

-- Perpendicular condition between l1 and l2
def perpendicular (a : ℝ) := (∃ x y : ℝ, l1 x y ∧ l2 a x y ∧ (2 * -a) / 4 = -1)

-- Parallel condition between l1 and l2
def parallel (a : ℝ) := (∃ x y : ℝ, l1 x y ∧ l2 a x y ∧ a = 8)

noncomputable def intersection_point : (ℝ × ℝ) := (-3/2, -1)

noncomputable def distance_between_lines : ℝ := (3 * sqrt 5) / 4

-- Statement for the intersection point when perpendicular
theorem perp_case (a : ℝ) : perpendicular a → ∃ x y, l1 x y ∧ l2 (-2) x y := 
by
  sorry

-- Statement for the distance when parallel
theorem parallel_distance {a : ℝ} : parallel a → distance_between_lines = (3 * sqrt 5) / 4 :=
by
  sorry

end perp_case_parallel_distance_l41_41349


namespace shifted_function_is_correct_l41_41532

def original_function (x : ℝ) : ℝ :=
  (x - 1)^2 + 2

def shifted_up_function (x : ℝ) : ℝ :=
  original_function x + 3

def shifted_right_function (x : ℝ) : ℝ :=
  shifted_up_function (x - 4)

theorem shifted_function_is_correct : ∀ x : ℝ, shifted_right_function x = (x - 5)^2 + 5 := 
by
  sorry

end shifted_function_is_correct_l41_41532


namespace unique_real_solution_system_l41_41310

/-- There is exactly one real solution (x, y, z, w) to the given system of equations:
  x + 1 = z + w + z * w * x,
  y - 1 = w + x + w * x * y,
  z + 2 = x + y + x * y * z,
  w - 2 = y + z + y * z * w
-/
theorem unique_real_solution_system :
  let eq1 (x y z w : ℝ) := x + 1 = z + w + z * w * x
  let eq2 (x y z w : ℝ) := y - 1 = w + x + w * x * y
  let eq3 (x y z w : ℝ) := z + 2 = x + y + x * y * z
  let eq4 (x y z w : ℝ) := w - 2 = y + z + y * z * w
  ∃! (x y z w : ℝ), eq1 x y z w ∧ eq2 x y z w ∧ eq3 x y z w ∧ eq4 x y z w := by {
  sorry
}

end unique_real_solution_system_l41_41310


namespace binom_8_5_eq_56_l41_41005

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := 
by
sorry

end binom_8_5_eq_56_l41_41005


namespace general_term_arithmetic_sequence_l41_41040

theorem general_term_arithmetic_sequence (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = a n + 2) : 
  ∀ n, a n = 2 * n - 1 := 
by 
  sorry

end general_term_arithmetic_sequence_l41_41040


namespace jenny_research_time_l41_41382

noncomputable def time_spent_on_research (total_hours : ℕ) (proposal_hours : ℕ) (report_hours : ℕ) : ℕ :=
  total_hours - proposal_hours - report_hours

theorem jenny_research_time : time_spent_on_research 20 2 8 = 10 := by
  sorry

end jenny_research_time_l41_41382


namespace password_count_l41_41113

theorem password_count : ∃ s : Finset ℕ, s.card = 4 ∧ s.sum id = 27 ∧ 
  (s = {9, 8, 7, 3} ∨ s = {9, 8, 6, 4} ∨ s = {9, 7, 6, 5}) ∧ 
  (s.toList.permutations.length = 72) := sorry

end password_count_l41_41113


namespace floor_length_l41_41254

variable (b l : ℝ)

theorem floor_length :
  (l = 3 * b) →
  (3 * b ^ 2 = 128) →
  l = 19.59 :=
by
  intros h1 h2
  sorry

end floor_length_l41_41254


namespace nth_equation_l41_41648

theorem nth_equation (n : ℕ) (h : 0 < n) : (10 * n + 5) ^ 2 = n * (n + 1) * 100 + 5 ^ 2 := 
sorry

end nth_equation_l41_41648


namespace total_apples_eaten_l41_41303

-- Define the variables based on the conditions
variable (tuesday_apples : ℕ)
variable (wednesday_apples : ℕ)
variable (thursday_apples : ℕ)
variable (total_apples : ℕ)

-- Define the conditions
def cond1 : Prop := tuesday_apples = 4
def cond2 : Prop := wednesday_apples = 2 * tuesday_apples
def cond3 : Prop := thursday_apples = tuesday_apples / 2

-- Define the total apples
def total : Prop := total_apples = tuesday_apples + wednesday_apples + thursday_apples

-- Prove the equivalence
theorem total_apples_eaten : 
  cond1 → cond2 → cond3 → total_apples = 14 :=
by 
  sorry

end total_apples_eaten_l41_41303


namespace paint_coverage_l41_41899

-- Define the conditions
def cost_per_gallon : ℝ := 45
def total_area : ℝ := 1600
def number_of_coats : ℝ := 2
def total_contribution : ℝ := 180 + 180

-- Define the target statement to prove
theorem paint_coverage (H : total_contribution = 360) : 
  let cost_per_gallon := 45 
  let number_of_gallons := total_contribution / cost_per_gallon
  let total_coverage := total_area * number_of_coats
  let coverage_per_gallon := total_coverage / number_of_gallons
  coverage_per_gallon = 400 :=
by
  sorry

end paint_coverage_l41_41899


namespace bob_wins_game_l41_41572

theorem bob_wins_game : 
  ∀ n : ℕ, 0 < n → 
  (∃ k ≥ 1, ∀ m : ℕ, 0 < m → (∃ a : ℕ, a ≥ 1 ∧ m - a*a = 0) ∨ 
    (∃ k : ℕ, k ≥ 1 ∧ (m = m^k → ¬ (∃ a : ℕ, a ≥ 1 ∧ m - a*a = 0)))
  ) :=
sorry

end bob_wins_game_l41_41572


namespace one_eighth_of_two_power_36_equals_two_power_x_l41_41359

theorem one_eighth_of_two_power_36_equals_two_power_x (x : ℕ) :
  (1 / 8) * (2 : ℝ) ^ 36 = (2 : ℝ) ^ x → x = 33 :=
by
  intro h
  sorry

end one_eighth_of_two_power_36_equals_two_power_x_l41_41359


namespace balls_in_jar_l41_41129

theorem balls_in_jar (total_balls initial_blue_balls balls_after_taking_out : ℕ) (probability_blue : ℚ) :
  initial_blue_balls = 6 →
  balls_after_taking_out = initial_blue_balls - 3 →
  probability_blue = 1 / 5 →
  (balls_after_taking_out : ℚ) / (total_balls - 3 : ℚ) = probability_blue →
  total_balls = 18 :=
by
  intros h1 h2 h3 h4
  sorry

end balls_in_jar_l41_41129


namespace problem_statement_l41_41337

open Nat

theorem problem_statement (n a : ℕ) 
  (hn : n > 1) 
  (ha : a > n^2)
  (H : ∀ i, 1 ≤ i ∧ i ≤ n → ∃ k, a + i = (n^2 + i) * k) :
  a > n^4 - n^3 := 
sorry

end problem_statement_l41_41337


namespace consecutive_weights_sum_to_63_l41_41228

theorem consecutive_weights_sum_to_63 : ∃ n : ℕ, (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5)) = 63 :=
by
  sorry

end consecutive_weights_sum_to_63_l41_41228


namespace bracelet_cost_l41_41831

theorem bracelet_cost (B : ℝ)
  (H1 : 5 = 5)
  (H2 : 3 = 3)
  (H3 : 2 * B + 5 + B + 3 = 20) : B = 4 :=
by
  sorry

end bracelet_cost_l41_41831


namespace sum_odd_divisors_of_90_l41_41971

theorem sum_odd_divisors_of_90 : 
  ∑ (d ∈ (filter (λ x, x % 2 = 1) (finset.divisors 90))), d = 78 :=
by
  sorry

end sum_odd_divisors_of_90_l41_41971


namespace remainder_9_pow_2023_div_50_l41_41432

theorem remainder_9_pow_2023_div_50 : (9 ^ 2023) % 50 = 41 := by
  sorry

end remainder_9_pow_2023_div_50_l41_41432


namespace an_expression_l41_41606

-- Given conditions
def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - n

-- The statement to be proved
theorem an_expression (a : ℕ → ℕ) (n : ℕ) (h_Sn : ∀ n, Sn a n = 2 * a n - n) :
  a n = 2^n - 1 :=
sorry

end an_expression_l41_41606


namespace matrix_addition_l41_41843

variable (A B : Matrix (Fin 2) (Fin 2) ℤ) -- Define matrices with integer entries

-- Define the specific matrices used in the problem
def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![ ![2, 3], ![-1, 4] ]

def matrix_B : Matrix (Fin 2) (Fin 2) ℤ := 
  ![ ![-1, 8], ![-3, 0] ]

-- Define the result matrix
def result_matrix : Matrix (Fin 2) (Fin 2) ℤ := 
  ![ ![3, 14], ![-5, 8] ]

-- The theorem to prove
theorem matrix_addition : 2 • matrix_A + matrix_B = result_matrix := by
  sorry -- Proof omitted

end matrix_addition_l41_41843


namespace sum_first_100_odd_l41_41271

-- Define the sequence of odd numbers.
def odd (n : ℕ) : ℕ := 2 * n + 1

-- Define the sum of the first n odd natural numbers.
def sumOdd (n : ℕ) : ℕ := (n * (n + 1))

-- State the theorem.
theorem sum_first_100_odd : sumOdd 100 = 10000 :=
by
  -- Skipping the proof as per the instructions
  sorry

end sum_first_100_odd_l41_41271


namespace hannah_games_l41_41741

theorem hannah_games (total_points : ℕ) (avg_points_per_game : ℕ) (h1 : total_points = 312) (h2 : avg_points_per_game = 13) :
  total_points / avg_points_per_game = 24 :=
sorry

end hannah_games_l41_41741


namespace sqrt_six_estimation_l41_41584

theorem sqrt_six_estimation : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 :=
by 
  sorry

end sqrt_six_estimation_l41_41584


namespace difference_between_two_numbers_l41_41793

theorem difference_between_two_numbers :
  ∃ a b : ℕ, 
    a + 5 * b = 23405 ∧ 
    (∃ b' : ℕ, b = 10 * b' + 5 ∧ b' = 5 * a) ∧ 
    5 * b - a = 21600 :=
by {
  sorry
}

end difference_between_two_numbers_l41_41793


namespace range_of_k_value_of_k_l41_41191

-- Defining the quadratic equation having two real roots condition
def has_real_roots (k : ℝ) : Prop :=
  let Δ := 9 - 4 * (k - 2)
  Δ ≥ 0

-- First part: range of k
theorem range_of_k (k : ℝ) : has_real_roots k ↔ k ≤ 17 / 4 :=
  sorry

-- Second part: specific value of k given additional condition
theorem value_of_k (x1 x2 k : ℝ) (h1 : (x1 + x2) = 3) (h2 : (x1 * x2) = k - 2) (h3 : (x1 + x2 - x1 * x2) = 1) : k = 4 :=
  sorry

end range_of_k_value_of_k_l41_41191


namespace quadratic_root_inequality_l41_41720

theorem quadratic_root_inequality (a : ℝ) :
  2015 < a ∧ a < 2017 ↔ 
  ∃ x₁ x₂ : ℝ, (2 * x₁^2 - 2016 * (x₁ - 2016 + a) - 1 = a^2) ∧ 
               (2 * x₂^2 - 2016 * (x₂ - 2016 + a) - 1 = a^2) ∧
               x₁ < a ∧ a < x₂ :=
sorry

end quadratic_root_inequality_l41_41720


namespace sumOddDivisorsOf90_l41_41951

-- Define the prime factorization and introduce necessary conditions.
noncomputable def primeFactorization (n : ℕ) : List ℕ := sorry

-- Function to compute all divisors of a number.
def divisors (n : ℕ) : List ℕ := sorry

-- Function to sum a list of integers.
def listSum (lst : List ℕ) : ℕ := lst.sum

-- Define the number 45 as the odd component of 90's prime factors.
def oddComponentOfNinety := 45

-- Define the odd divisors of 45.
noncomputable def oddDivisorsOfOddComponent := divisors oddComponentOfNinety |>.filter (λ x => x % 2 = 1)

-- The goal to prove.
theorem sumOddDivisorsOf90 : listSum oddDivisorsOfOddComponent = 78 := by
  sorry

end sumOddDivisorsOf90_l41_41951


namespace least_integer_square_l41_41544

theorem least_integer_square (x : ℤ) : x^2 = 2 * x + 72 → x = -6 := 
by
  intro h
  sorry

end least_integer_square_l41_41544


namespace average_percentage_reduction_equation_l41_41825

theorem average_percentage_reduction_equation (x : ℝ) : 200 * (1 - x)^2 = 162 :=
by 
  sorry

end average_percentage_reduction_equation_l41_41825


namespace find_f_of_13_l41_41820

def f : ℤ → ℤ := sorry  -- We define f as a function from integers to integers

theorem find_f_of_13 : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x k : ℤ, f (x + 4 * k) = f x) ∧ 
  (f (-1) = 2) → 
  f 13 = -2 := 
by 
  sorry

end find_f_of_13_l41_41820


namespace smallest_N_conditions_l41_41292

theorem smallest_N_conditions:
  ∃N : ℕ, N % 9 = 8 ∧
           N % 8 = 7 ∧
           N % 7 = 6 ∧
           N % 6 = 5 ∧
           N % 5 = 4 ∧
           N % 4 = 3 ∧
           N % 3 = 2 ∧
           N % 2 = 1 ∧
           N = 2519 :=
sorry

end smallest_N_conditions_l41_41292


namespace no_positive_reals_satisfy_equations_l41_41924

theorem no_positive_reals_satisfy_equations :
  ¬ ∃ (a b c d : ℝ), (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧
  (a / b + b / c + c / d + d / a = 6) ∧ (b / a + c / b + d / c + a / d = 32) :=
by sorry

end no_positive_reals_satisfy_equations_l41_41924


namespace length_of_BC_l41_41221

theorem length_of_BC (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
    (BX CX : ℕ) (h_pow : CX * (BX + CX) = 2013) : 
    BX + CX = 61 :=
  sorry

end length_of_BC_l41_41221


namespace weight_of_B_l41_41276

theorem weight_of_B (A B C : ℕ) (h1 : A + B + C = 90) (h2 : A + B = 50) (h3 : B + C = 56) : B = 16 := 
sorry

end weight_of_B_l41_41276


namespace quadratic_coefficients_l41_41992

theorem quadratic_coefficients :
  ∀ x : ℝ, 3 * x^2 = 5 * x - 1 → (∃ a b c : ℝ, a = 3 ∧ b = -5 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x h
  use 3, -5, 1
  sorry

end quadratic_coefficients_l41_41992


namespace fish_problem_l41_41838

theorem fish_problem : 
  ∀ (B T S : ℕ), 
    B = 10 → 
    T = 3 * B → 
    S = 35 → 
    B + T + S + 2 * S = 145 → 
    S - T = 5 :=
by sorry

end fish_problem_l41_41838


namespace probability_of_winning_l41_41896

/-- 
Given the conditions:
1. The game consists of 8 rounds.
2. The probability of Alex winning any round is 1/2.
3. Mel's chance of winning any round is twice that of Chelsea.
4. The outcomes of the rounds are independent.
Prove that the probability that Alex wins 4 rounds, Mel wins 3 rounds, and Chelsea wins 1 round is 35/324.
-/
theorem probability_of_winning (p_A p_M p_C : ℚ) (n_A n_M n_C : ℕ) (total_rounds : ℕ)
  (h_rounds : total_rounds = 8)
  (h_pA : p_A = 1/2)
  (h_pM : p_M = 2 * p_C)
  (h_total_p : p_A + p_M + p_C = 1)
  (h_nA : n_A = 4)
  (h_nM : n_M = 3)
  (h_nC : n_C = 1) :
  (nat.choose 8 4 * nat.choose (8-4) (3) * nat.choose (8-4-3) (1)) 
  * (p_A^4 * p_M^3 * p_C^1) = 35 / 324 :=
by {
  sorry
}

end probability_of_winning_l41_41896


namespace tangent_line_eq_l41_41126

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 - 2 * x - 1)

theorem tangent_line_eq :
  let x := 1
  let y := f x
  ∃ (m : ℝ), m = -2 * Real.exp 1 ∧ (∀ (x y : ℝ), y = m * (x - 1) + f 1) :=
by
  sorry

end tangent_line_eq_l41_41126


namespace panda_on_stilts_height_l41_41712

theorem panda_on_stilts_height (x : ℕ) (h_A : ℕ) 
  (h1 : h_A = x / 4) -- A Bao's height accounts for 1/4 of initial total height
  (h2 : x - 40 = 3 * h_A) -- After breaking 20 dm off each stilt, the new total height is such that A Bao's height accounts for 1/3 of this new height
  : x = 160 := 
by
  sorry

end panda_on_stilts_height_l41_41712


namespace necessary_but_not_sufficient_condition_l41_41557

theorem necessary_but_not_sufficient_condition :
  (∀ x, x > 2 → x^2 - 3*x + 2 > 0) ∧ (∃ x, x^2 - 3*x + 2 > 0 ∧ ¬ (x > 2)) :=
by {
  sorry
}

end necessary_but_not_sufficient_condition_l41_41557


namespace sum_of_odd_divisors_l41_41964

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l41_41964


namespace contrapositive_eq_l41_41811

variables (P Q : Prop)

theorem contrapositive_eq : (¬P → Q) ↔ (¬Q → P) := 
by {
    sorry
}

end contrapositive_eq_l41_41811


namespace max_k_inequality_l41_41319

open Real

theorem max_k_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b) * (a * b + 1) * (b + 1) ≥ (27 / 4) * a * b^2 :=
by
  sorry

end max_k_inequality_l41_41319


namespace determine_x_l41_41413

theorem determine_x
  (total_area : ℝ)
  (side_length_square1 : ℝ)
  (side_length_square2 : ℝ)
  (h1 : total_area = 1300)
  (h2 : side_length_square1 = 3 * x)
  (h3 : side_length_square2 = 7 * x) :
    x = Real.sqrt (2600 / 137) :=
by
  sorry

end determine_x_l41_41413


namespace inequality_example_l41_41097

theorem inequality_example (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
sorry

end inequality_example_l41_41097


namespace fabric_delivered_on_monday_amount_l41_41021

noncomputable def cost_per_yard : ℝ := 2
noncomputable def earnings : ℝ := 140

def fabric_delivered_on_monday (x : ℝ) : Prop :=
  let tuesday := 2 * x
  let wednesday := (1 / 4) * tuesday
  let total_yards := x + tuesday + wednesday
  let total_earnings := total_yards * cost_per_yard
  total_earnings = earnings

theorem fabric_delivered_on_monday_amount : ∃ x : ℝ, fabric_delivered_on_monday x ∧ x = 20 :=
by sorry

end fabric_delivered_on_monday_amount_l41_41021


namespace savings_plan_l41_41087

noncomputable def ivan_salary : ℝ := 55000
noncomputable def vasilisa_salary : ℝ := 45000
noncomputable def mother_salary_before_retirement : ℝ := 18000
noncomputable def mother_pension_after_retirement : ℝ := 10000
noncomputable def father_salary : ℝ := 20000
noncomputable def son_state_stipend : ℝ := 3000
noncomputable def son_non_state_stipend : ℝ := 15000
noncomputable def income_tax_rate : ℝ := 0.13
noncomputable def monthly_expenses : ℝ := 74000

def net_income (salary : ℝ) : ℝ := salary * (1 - income_tax_rate)

theorem savings_plan : 
  let ivan_net := net_income ivan_salary in
  let vasilisa_net := net_income vasilisa_salary in
  let mother_net_before := net_income mother_salary_before_retirement in
  let father_net := net_income father_salary in
  let son_net := son_state_stipend in
  -- Before May 1, 2018
  let total_net_before := ivan_net + vasilisa_net + mother_net_before + father_net + son_net in
  let savings_before := total_net_before - monthly_expenses in
  -- From May 1, 2018 to August 31, 2018
  let mother_net_after := mother_pension_after_retirement in
  let total_net_after := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_after := total_net_after - monthly_expenses in
  -- From September 1, 2018 for 1 year
  let son_net := son_state_stipend + net_income son_non_state_stipend in
  let total_net_future := ivan_net + vasilisa_net + mother_net_after + father_net + son_net in
  let savings_future := total_net_future - monthly_expenses in
  savings_before = 49060 ∧ savings_after = 43400 ∧ savings_future = 56450 :=
by
  sorry

end savings_plan_l41_41087


namespace cricket_problem_l41_41749

theorem cricket_problem
  (x : ℕ)
  (run_rate_initial : ℝ := 3.8)
  (overs_remaining : ℕ := 40)
  (run_rate_remaining : ℝ := 6.1)
  (target_runs : ℕ := 282) :
  run_rate_initial * x + run_rate_remaining * overs_remaining = target_runs → x = 10 :=
by
  -- proof goes here
  sorry

end cricket_problem_l41_41749


namespace fencing_problem_l41_41176

theorem fencing_problem (W L : ℝ) (hW : W = 40) (hArea : W * L = 320) : 
  2 * L + W = 56 :=
by
  sorry

end fencing_problem_l41_41176


namespace filtration_minimum_l41_41827

noncomputable def lg : ℝ → ℝ := sorry

theorem filtration_minimum (x : ℕ) (lg2 : ℝ) (lg3 : ℝ) (h1 : lg2 = 0.3010) (h2 : lg3 = 0.4771) :
  (2 / 3 : ℝ) ^ x ≤ 1 / 20 → x ≥ 8 :=
sorry

end filtration_minimum_l41_41827


namespace percentage_of_315_out_of_900_is_35_l41_41687

theorem percentage_of_315_out_of_900_is_35 :
  (315 : ℝ) / 900 * 100 = 35 := 
by
  sorry

end percentage_of_315_out_of_900_is_35_l41_41687


namespace problems_per_page_l41_41240

-- Define the initial conditions
def total_problems : ℕ := 101
def finished_problems : ℕ := 47
def remaining_pages : ℕ := 6

-- State the theorem
theorem problems_per_page : 54 / remaining_pages = 9 :=
by
  -- Sorry is used to ignore the proof step
  sorry

end problems_per_page_l41_41240


namespace sum_of_odd_divisors_of_90_l41_41968

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l41_41968


namespace range_of_m_l41_41117

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * m * x + m^2 - 1 = 0) → (-2 < x)) ↔ m > -1 :=
by
  sorry

end range_of_m_l41_41117


namespace number_of_boys_l41_41052

variable {total_marbles : ℕ} (marbles_per_boy : ℕ := 10)
variable (H_total_marbles : total_marbles = 20)

theorem number_of_boys (total_marbles_marbs_eq_20 : total_marbles = 20) (marbles_per_boy_eq_10 : marbles_per_boy = 10) :
  total_marbles / marbles_per_boy = 2 :=
by {
  sorry
}

end number_of_boys_l41_41052


namespace shop_earnings_l41_41627

theorem shop_earnings :
  let cola_price := 3
  let juice_price := 1.5
  let water_price := 1
  let cola_sold := 15
  let juice_sold := 12
  let water_sold := 25
  let cola_earnings := cola_price * cola_sold
  let juice_earnings := juice_price * juice_sold
  let water_earnings := water_price * water_sold
  let total_earnings := cola_earnings + juice_earnings + water_earnings
  total_earnings = 88 := by
    simp [cola_price, juice_price, water_price, cola_sold, juice_sold, water_sold, cola_earnings, juice_earnings, water_earnings, total_earnings]; sorry

end shop_earnings_l41_41627


namespace henry_total_payment_l41_41029

-- Define the conditions
def painting_payment : ℕ := 5
def selling_extra_payment : ℕ := 8
def total_payment_per_bike : ℕ := painting_payment + selling_extra_payment  -- 13

-- Define the quantity of bikes
def bikes_count : ℕ := 8

-- Calculate the total payment for painting and selling 8 bikes
def total_payment : ℕ := bikes_count * total_payment_per_bike  -- 144

-- The statement to prove
theorem henry_total_payment : total_payment = 144 :=
by
  -- Proof goes here
  sorry

end henry_total_payment_l41_41029


namespace greatest_possible_third_term_l41_41120

theorem greatest_possible_third_term :
  ∃ (a d : ℕ), (a > 0) ∧ (d > 0) ∧ (4 * a + 6 * d = 50) ∧ (∀ (a' d' : ℕ), (a' > 0) ∧ (d' > 0) ∧ (4 * a' + 6 * d' = 50) → (a + 2 * d ≥ a' + 2 * d')) ∧ (a + 2 * d = 16) :=
sorry

end greatest_possible_third_term_l41_41120


namespace arithmetic_mean_of_sequence_beginning_at_5_l41_41841

def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

def sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def arithmetic_mean (a d n : ℕ) : ℚ :=
  sequence_sum a d n / n

theorem arithmetic_mean_of_sequence_beginning_at_5 : 
  arithmetic_mean 5 1 60 = 34.5 :=
by
  sorry

end arithmetic_mean_of_sequence_beginning_at_5_l41_41841


namespace max_value_abs_diff_l41_41414

noncomputable def f : ℝ → ℝ := λ x, x^3 - 3 * x^2 - x + 1

theorem max_value_abs_diff :
  (x0 : ℝ) (m : ℝ) 
  (hx0_max : ∀ x, f x ≤ f x0) 
  (hx0_ne : m ≠ x0) 
  (hf_eq : f x0 = f m) :
  |m - x0| = 2 * sqrt 3 := 
begin
  sorry
end

end max_value_abs_diff_l41_41414


namespace cos_squared_sin_pi_over_2_plus_alpha_l41_41593

variable (α : ℝ)

-- Given conditions
def cond1 : Prop := (Real.pi / 2) < α * Real.pi
def cond2 : Prop := Real.cos α = -3 / 5

-- Proof goal
theorem cos_squared_sin_pi_over_2_plus_alpha :
  cond1 α → cond2 α →
  (Real.cos (Real.sin (Real.pi / 2 + α)))^2 = 8 / 25 :=
by
  intro h1 h2
  sorry

end cos_squared_sin_pi_over_2_plus_alpha_l41_41593


namespace smallest_N_exists_l41_41295

theorem smallest_N_exists : ∃ N : ℕ, 
  (N % 9 = 8) ∧
  (N % 8 = 7) ∧
  (N % 7 = 6) ∧
  (N % 6 = 5) ∧
  (N % 5 = 4) ∧
  (N % 4 = 3) ∧
  (N % 3 = 2) ∧
  (N % 2 = 1) ∧
  N = 503 :=
by {
  sorry
}

end smallest_N_exists_l41_41295


namespace no_integer_solutions_for_x2_minus_4y2_eq_2011_l41_41418

theorem no_integer_solutions_for_x2_minus_4y2_eq_2011 :
  ∀ (x y : ℤ), x^2 - 4 * y^2 ≠ 2011 := by
sorry

end no_integer_solutions_for_x2_minus_4y2_eq_2011_l41_41418


namespace largest_term_at_k_31_l41_41162

noncomputable def B_k (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.15)^k

theorem largest_term_at_k_31 : 
  ∀ k : ℕ, (k ≤ 500) →
    (B_k 31 ≥ B_k k) :=
by
  intro k hk
  sorry

end largest_term_at_k_31_l41_41162


namespace solution_l41_41125

noncomputable def determine_numbers (x y : ℚ) : Prop :=
  x^2 + y^2 = 45 / 4 ∧ x - y = x * y

theorem solution (x y : ℚ) :
  determine_numbers x y → (x = -3 ∧ y = 3/2) ∨ (x = -3/2 ∧ y = 3) :=
-- We state the main theorem that relates the determine_numbers predicate to the specific pairs of numbers
sorry

end solution_l41_41125


namespace lateral_surface_area_cone_l41_41867

-- Given definitions (conditions)
def radius : ℝ := 3
def slant_height : ℝ := 5

-- Question transformed into a theorem statement
theorem lateral_surface_area_cone : 
  ∀ (r l : ℝ), r = radius → l = slant_height → (1 / 2) * (2 * real.pi * r) * l = 15 * real.pi := 
by 
  intros r l hr hl
  rw [hr, hl]
  sorry

end lateral_surface_area_cone_l41_41867


namespace percentage_increase_first_job_percentage_increase_second_job_percentage_increase_third_job_l41_41900

theorem percentage_increase_first_job :
  let old_salary := 65
  let new_salary := 70
  (new_salary - old_salary) / old_salary * 100 = 7.69 := by
  sorry

theorem percentage_increase_second_job :
  let old_salary := 120
  let new_salary := 138
  (new_salary - old_salary) / old_salary * 100 = 15 := by
  sorry

theorem percentage_increase_third_job :
  let old_salary := 200
  let new_salary := 220
  (new_salary - old_salary) / old_salary * 100 = 10 := by
  sorry

end percentage_increase_first_job_percentage_increase_second_job_percentage_increase_third_job_l41_41900


namespace trailing_zeros_in_square_l41_41840

-- Define x as given in the conditions
def x : ℕ := 10^12 - 4

-- State the theorem which asserts that the number of trailing zeros in x^2 is 11
theorem trailing_zeros_in_square : 
  ∃ n : ℕ, n = 11 ∧ x^2 % 10^12 = 0 :=
by
  -- Placeholder for the proof
  sorry

end trailing_zeros_in_square_l41_41840


namespace maria_towels_l41_41437

-- Define the initial total towels
def initial_total : ℝ := 124.5 + 67.7

-- Define the towels given to her mother
def towels_given : ℝ := 85.35

-- Define the remaining towels (this is what we need to prove)
def towels_remaining : ℝ := 106.85

-- The theorem that states Maria ended up with the correct number of towels
theorem maria_towels :
  initial_total - towels_given = towels_remaining :=
by
  -- Here we would provide the proof, but we use sorry for now
  sorry

end maria_towels_l41_41437


namespace minimum_value_g_l41_41339

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - x - 2

def g (x : ℝ) : ℝ := (x + a)^2 - (x + a) - 2 + x

theorem minimum_value_g (a : ℝ) :
  (if 1 ≤ a then g a (-1) = a^2 - 3 * a - 1 else
   if -3 < a ∧ a < 1 then g a (-a) = -a - 2 else
   if a ≤ -3 then g a 3 = a^2 + 5 * a + 7 else false) :=
by
  sorry

end minimum_value_g_l41_41339


namespace sine_cosine_inequality_l41_41195

theorem sine_cosine_inequality (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < 2 * Real.pi)
    (hineq : Real.sin θ ^ 3 - Real.cos θ ^ 3 > (Real.cos θ ^ 5 - Real.sin θ ^ 5) / 7) :
    (Real.pi / 4) < θ ∧ θ < (5 * Real.pi / 4) :=
sorry

end sine_cosine_inequality_l41_41195


namespace worker_time_proof_l41_41398

theorem worker_time_proof (x : ℝ) (h1 : x > 2) (h2 : (100 / (x - 2) - 100 / x) = 5 / 2) : 
  (x = 10) ∧ (x - 2 = 8) :=
by
  sorry

end worker_time_proof_l41_41398


namespace alyssa_initial_puppies_l41_41574

theorem alyssa_initial_puppies (gave_away has_left : ℝ) (h1 : gave_away = 8.5) (h2 : has_left = 12.5) :
    (gave_away + has_left = 21) :=
by
    sorry

end alyssa_initial_puppies_l41_41574


namespace ratio_of_pieces_l41_41248

def total_length (len: ℕ) := len = 35
def longer_piece (len: ℕ) := len = 20

theorem ratio_of_pieces (shorter len_shorter : ℕ) : 
  total_length 35 →
  longer_piece 20 →
  shorter = 35 - 20 →
  len_shorter = 15 →
  (20:ℚ) / (len_shorter:ℚ) = (4:ℚ) / (3:ℚ) :=
by
  sorry

end ratio_of_pieces_l41_41248


namespace greatest_possible_third_term_l41_41119

theorem greatest_possible_third_term :
  ∃ (a d : ℕ), (a > 0) ∧ (d > 0) ∧ (4 * a + 6 * d = 50) ∧ (∀ (a' d' : ℕ), (a' > 0) ∧ (d' > 0) ∧ (4 * a' + 6 * d' = 50) → (a + 2 * d ≥ a' + 2 * d')) ∧ (a + 2 * d = 16) :=
sorry

end greatest_possible_third_term_l41_41119


namespace isosceles_triangle_formed_by_lines_l41_41791

theorem isosceles_triangle_formed_by_lines :
  let P1 := (1/4, 4)
  let P2 := (-3/2, -3)
  let P3 := (2, -3)
  let d12 := ((1/4 + 3/2)^2 + (4 + 3)^2)
  let d13 := ((1/4 - 2)^2 + (4 + 3)^2)
  let d23 := ((-3/2 - 2)^2)
  (d12 = d13) ∧ (d12 ≠ d23) → 
  ∃ (A B C : ℝ × ℝ), 
    A = P1 ∧ B = P2 ∧ C = P3 ∧ 
    ((dist A B = dist A C) ∧ (dist B C ≠ dist A B)) :=
by
  sorry

end isosceles_triangle_formed_by_lines_l41_41791


namespace increasing_on_interval_solution_set_l41_41343

noncomputable def f (x : ℝ) : ℝ := x / (x ^ 2 + 1)

/- Problem 1 -/
theorem increasing_on_interval : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2 :=
by
  sorry

/- Problem 2 -/
theorem solution_set : ∀ x : ℝ, f (2 * x - 1) + f x < 0 ↔ 0 < x ∧ x < 1 / 3 :=
by
  sorry

end increasing_on_interval_solution_set_l41_41343


namespace sum_not_prime_if_product_equality_l41_41759

theorem sum_not_prime_if_product_equality 
  (a b c d : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a * b = c * d) : ¬Nat.Prime (a + b + c + d) := 
by
  sorry

end sum_not_prime_if_product_equality_l41_41759


namespace volume_of_cube_l41_41800

theorem volume_of_cube (SA : ℝ) (h : SA = 486) : ∃ V : ℝ, V = 729 :=
by
  sorry

end volume_of_cube_l41_41800


namespace last_three_digits_of_8_pow_105_l41_41724

theorem last_three_digits_of_8_pow_105 : (8 ^ 105) % 1000 = 992 :=
by
  sorry

end last_three_digits_of_8_pow_105_l41_41724


namespace at_least_one_no_less_than_two_l41_41180

variable (a b c : ℝ)
variable (ha : 0 < a)
variable (hb : 0 < b)
variable (hc : 0 < c)

theorem at_least_one_no_less_than_two :
  ∃ x ∈ ({a + 1/b, b + 1/c, c + 1/a} : Set ℝ), 2 ≤ x := by
  sorry

end at_least_one_no_less_than_two_l41_41180


namespace handshakes_min_l41_41145

-- Define the number of people and the number of handshakes each person performs
def numPeople : ℕ := 35
def handshakesPerPerson : ℕ := 3

-- Define the minimum possible number of unique handshakes
theorem handshakes_min : (numPeople * handshakesPerPerson) / 2 = 105 := by
  sorry

end handshakes_min_l41_41145


namespace erasers_difference_l41_41996

-- Definitions for the conditions in the problem
def andrea_erasers : ℕ := 4
def anya_erasers : ℕ := 4 * andrea_erasers

-- Theorem statement to prove the final answer
theorem erasers_difference : anya_erasers - andrea_erasers = 12 :=
by
  -- Proof placeholder
  sorry

end erasers_difference_l41_41996


namespace rectangle_area_l41_41693

variable (a b c : ℝ)

theorem rectangle_area (h : a^2 + b^2 = c^2) : a * b = area :=
by sorry

end rectangle_area_l41_41693


namespace average_ABC_is_3_l41_41869

theorem average_ABC_is_3
  (A B C : ℝ)
  (h1 : 2003 * C - 4004 * A = 8008)
  (h2 : 2003 * B + 6006 * A = 10010)
  (h3 : B = 2 * A - 6) : 
  (A + B + C) / 3 = 3 :=
sorry

end average_ABC_is_3_l41_41869


namespace one_eighth_of_power_l41_41356

theorem one_eighth_of_power (x : ℕ) (h : (1 / 8) * (2 ^ 36) = 2 ^ x) : x = 33 :=
by 
  -- Proof steps are not needed, so we leave it as sorry.
  sorry

end one_eighth_of_power_l41_41356


namespace price_reduction_equation_l41_41823

theorem price_reduction_equation (x : ℝ) : 200 * (1 - x) ^ 2 = 162 :=
by
  sorry

end price_reduction_equation_l41_41823


namespace money_distribution_l41_41302

theorem money_distribution (A B C : ℝ) (h1 : A + B + C = 1000) (h2 : B + C = 600) (h3 : C = 300) : A + C = 700 := by
  sorry

end money_distribution_l41_41302


namespace graph_quadrant_exclusion_l41_41332

theorem graph_quadrant_exclusion (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ∀ x : ℝ, ¬ ((a^x + b > 0) ∧ (x > 0)) :=
by
  sorry

end graph_quadrant_exclusion_l41_41332


namespace probability_of_red_then_blue_is_correct_l41_41447

noncomputable def probability_red_then_blue : ℚ :=
  let total_marbles := 5 + 4 + 12 + 2
  let prob_red := 5 / total_marbles
  let remaining_marbles := total_marbles - 1
  let prob_blue_given_red := 2 / remaining_marbles
  prob_red * prob_blue_given_red

theorem probability_of_red_then_blue_is_correct :
  probability_red_then_blue = 5 / 253 := 
by 
  sorry

end probability_of_red_then_blue_is_correct_l41_41447


namespace equal_sum_sequence_a18_l41_41710

theorem equal_sum_sequence_a18
    (a : ℕ → ℕ)
    (h1 : a 1 = 2)
    (h2 : ∀ n, a n + a (n + 1) = 5) :
    a 18 = 3 :=
sorry

end equal_sum_sequence_a18_l41_41710


namespace cosine_of_angle_between_diagonals_l41_41567

-- Defining the vectors a and b
def a : ℝ × ℝ × ℝ := (3, 1, 1)
def b : ℝ × ℝ × ℝ := (1, -1, -1)

-- Calculating the diagonal vectors
def d1 := (a.1 + b.1, a.2 + b.2, a.3 + b.3)  -- a + b
def d2 := (b.1 - a.1, b.2 - a.2, b.3 - a.3)  -- b - a

-- Dot product function for 3D vectors
def dot_prod (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Norm function for 3D vectors
def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

-- Cosine of the angle between two vectors
def cos_angle (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  dot_prod v1 v2 / (norm v1 * norm v2)

-- The theorem to prove
theorem cosine_of_angle_between_diagonals : cos_angle d1 d2 = - (Real.sqrt 3 / 3) :=
by
  sorry

end cosine_of_angle_between_diagonals_l41_41567


namespace parallelogram_area_l41_41784

noncomputable def polynomial : Complex → Complex := fun z =>
  z^4 + 4 * Complex.I * z^3 + (-5 + 7 * Complex.I) * z^2 + (-10 - 4 * Complex.I) * z + (1 - 8 * Complex.I)

theorem parallelogram_area :
  let roots := polynomial.roots in
  roots.length = 4 →
  let a := roots[0]
  let b := roots[1]
  let c := roots[2]
  let d := roots[3]
  is_parallelogram a b c d →
  parallelogram_area a b c d = Real.sqrt 10 :=
by sorry

end parallelogram_area_l41_41784


namespace maximum_additional_payment_expected_value_difference_l41_41714

-- Add the conditions as definitions
def a1 : ℕ := 1298
def a2 : ℕ := 1347
def a3 : ℕ := 1337
def b1 : ℕ := 1402
def b2 : ℕ := 1310
def b3 : ℕ := 1298

-- Prices in rubles per kilowatt-hour
def peak_price : ℝ := 4.03
def night_price : ℝ := 1.01
def semi_peak_price : ℝ := 3.39

-- Actual consumptions in kilowatt-hour
def ΔP : ℝ := 104
def ΔN : ℝ := 37
def ΔSP : ℝ := 39

-- Correct payment calculated by the company
def correct_payment : ℝ := 660.72

-- Statements to prove
theorem maximum_additional_payment : 397.34 = (104 * 4.03 + 39 * 3.39 + 37 * 1.01 - 660.72) :=
by
  sorry

theorem expected_value_difference : 19.3 = ((5 * 1402 + 3 * 1347 + 1337 - 1298 - 3 * 1270 - 5 * 1214) / 15 * 8.43 - 660.72) :=
by
  sorry

end maximum_additional_payment_expected_value_difference_l41_41714


namespace smallest_n_l41_41153

theorem smallest_n :
  ∃ n : ℕ, n > 0 ∧ 2000 * n % 21 = 0 ∧ ∀ m : ℕ, m > 0 ∧ 2000 * m % 21 = 0 → n ≤ m :=
sorry

end smallest_n_l41_41153


namespace combined_weight_of_three_l41_41230

theorem combined_weight_of_three (Mary Jamison John : ℝ) 
  (h₁ : Mary = 160) 
  (h₂ : Jamison = Mary + 20) 
  (h₃ : John = Mary + (1/4) * Mary) :
  Mary + Jamison + John = 540 := by
  sorry

end combined_weight_of_three_l41_41230


namespace wrapping_paper_amount_l41_41459

theorem wrapping_paper_amount (x : ℝ) (h : x + (3/4) * x + (x + (3/4) * x) = 7) : x = 2 :=
by
  sorry

end wrapping_paper_amount_l41_41459


namespace greatest_third_term_of_arithmetic_sequence_l41_41123

theorem greatest_third_term_of_arithmetic_sequence (a d : ℕ) (h₁ : 0 < a)
  (h₂ : 0 < d) (h₃ : 4 * a + 6 * d = 50) : a + 2 * d = 16 :=
by
  -- Using the given condition
  -- 1. 4a + 6d = 50
  -- 2. a and d are in the naturals and greater than 0
  -- We prove that the greatest possible value of the third term (a + 2d)
  -- given these conditions equals 16
  sorry

end greatest_third_term_of_arithmetic_sequence_l41_41123


namespace quadratic_polynomial_correct_l41_41172

noncomputable def q (x : ℝ) : ℝ := (11/10) * x^2 - (21/10) * x + 5

theorem quadratic_polynomial_correct :
  (q (-1) = 4) ∧ (q 2 = 1) ∧ (q 4 = 10) :=
by
  -- Proof goes here
  sorry

end quadratic_polynomial_correct_l41_41172


namespace jerry_time_proof_l41_41801

noncomputable def tom_walk_speed (step_length_tom : ℕ) (pace_tom : ℕ) : ℕ := 
  step_length_tom * pace_tom

noncomputable def tom_distance_to_office (walk_speed_tom : ℕ) (time_tom : ℕ) : ℕ :=
  walk_speed_tom * time_tom

noncomputable def jerry_walk_speed (step_length_jerry : ℕ) (pace_jerry : ℕ) : ℕ :=
  step_length_jerry * pace_jerry

noncomputable def jerry_time_to_office (distance_to_office : ℕ) (walk_speed_jerry : ℕ) : ℚ :=
  distance_to_office / walk_speed_jerry

theorem jerry_time_proof :
  let step_length_tom := 80
  let pace_tom := 85
  let time_tom := 20
  let step_length_jerry := 70
  let pace_jerry := 110
  let office_distance := tom_distance_to_office (tom_walk_speed step_length_tom pace_tom) time_tom
  let jerry_speed := jerry_walk_speed step_length_jerry pace_jerry
  jerry_time_to_office office_distance jerry_speed = 53/3 := 
by
  sorry

end jerry_time_proof_l41_41801


namespace jerrys_breakfast_calories_l41_41513

-- Define the constants based on the conditions
def pancakes : ℕ := 6
def calories_per_pancake : ℕ := 120
def strips_of_bacon : ℕ := 2
def calories_per_strip_of_bacon : ℕ := 100
def calories_in_cereal : ℕ := 200

-- Define the total calories for each category
def total_calories_from_pancakes : ℕ := pancakes * calories_per_pancake
def total_calories_from_bacon : ℕ := strips_of_bacon * calories_per_strip_of_bacon
def total_calories_from_cereal : ℕ := calories_in_cereal

-- Define the total calories in the breakfast
def total_breakfast_calories : ℕ := 
  total_calories_from_pancakes + total_calories_from_bacon + total_calories_from_cereal

-- The theorem we need to prove
theorem jerrys_breakfast_calories : total_breakfast_calories = 1120 := by sorry

end jerrys_breakfast_calories_l41_41513


namespace polynomial_divisibility_l41_41236

theorem polynomial_divisibility (P : Polynomial ℂ) (n : ℕ) 
  (h : ∃ Q : Polynomial ℂ, P.comp (X ^ n) = (X - 1) * Q) : 
  ∃ R : Polynomial ℂ, P.comp (X ^ n) = (X ^ n - 1) * R :=
sorry

end polynomial_divisibility_l41_41236


namespace system_non_zero_solution_condition_l41_41431

theorem system_non_zero_solution_condition (a b c : ℝ) :
  (∃ (x y z : ℝ), x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∧
    x = b * y + c * z ∧
    y = c * z + a * x ∧
    z = a * x + b * y) ↔
  (2 * a * b * c + a * b + b * c + c * a - 1 = 0) :=
sorry

end system_non_zero_solution_condition_l41_41431


namespace angle_between_plane_and_base_l41_41289

variable (α k : ℝ)
variable (hα : ∀ S A B C : ℝ, S ≠ A ∧ S ≠ B ∧ S ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
variable (h_ratio : ∀ A D S : ℝ, AD / DS = k)

theorem angle_between_plane_and_base (α k : ℝ) 
  (hα : ∀ S A B C : ℝ, S ≠ A ∧ S ≠ B ∧ S ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (h_ratio : ∀ A D S : ℝ, AD / DS = k) 
  : ∃ γ : ℝ, γ = Real.arctan (k / (k + 3) * Real.tan α) :=
by
  sorry

end angle_between_plane_and_base_l41_41289


namespace village_population_l41_41833

noncomputable def number_of_people_in_village
  (vampire_drains_per_week : ℕ)
  (werewolf_eats_per_week : ℕ)
  (weeks : ℕ) : ℕ :=
  let drained := vampire_drains_per_week * weeks
  let eaten := werewolf_eats_per_week * weeks
  drained + eaten

theorem village_population :
  number_of_people_in_village 3 5 9 = 72 := by
  sorry

end village_population_l41_41833


namespace arithmetic_sequence_twentieth_term_l41_41663

theorem arithmetic_sequence_twentieth_term
  (a1 : ℤ) (a13 : ℤ) (a20 : ℤ) (d : ℤ)
  (h1 : a1 = 3)
  (h2 : a13 = 27)
  (h3 : a13 = a1 + 12 * d)
  (h4 : a20 = a1 + 19 * d) : 
  a20 = 41 :=
by
  --  We assume a20 and prove it equals 41 instead of solving it in steps
  sorry

end arithmetic_sequence_twentieth_term_l41_41663


namespace element_in_set_l41_41762

def M : Set (ℤ × ℤ) := {(1, 2)}

theorem element_in_set : (1, 2) ∈ M :=
by
  sorry

end element_in_set_l41_41762


namespace fill_in_blank_with_warning_l41_41854

-- Definitions corresponding to conditions
def is_noun (word : String) : Prop :=
  -- definition of being a noun
  sorry

def corresponds_to_chinese_hint (word : String) (hint : String) : Prop :=
  -- definition of corresponding to a Chinese hint
  sorry

-- The theorem we want to prove
theorem fill_in_blank_with_warning : ∀ word : String, 
  (is_noun word ∧ corresponds_to_chinese_hint word "警告") → word = "warning" :=
by {
  sorry
}

end fill_in_blank_with_warning_l41_41854


namespace well_depth_and_rope_length_l41_41106

theorem well_depth_and_rope_length (h x : ℝ) : 
  (h / 3 = x + 4) ∧ (h / 4 = x + 1) :=
sorry

end well_depth_and_rope_length_l41_41106


namespace functional_expression_point_M_coordinates_l41_41182

variables (x y : ℝ) (k : ℝ)

-- Given conditions
def proportional_relation : Prop := y + 4 = k * (x - 3)
def initial_condition : Prop := (x = 1 → y = 0)
def point_M : Prop := ∃ m : ℝ, (m + 1, 2 * m) = (1, 0)

-- Proof of the functional expression
theorem functional_expression (h1 : proportional_relation x y k) (h2 : initial_condition x y) :
  ∃ k : ℝ, k = -2 ∧ y = -2 * x + 2 := 
sorry

-- Proof of the coordinates of point M
theorem point_M_coordinates (h : ∀ m : ℝ, (m + 1, 2 * m) = (1, 0)) :
  ∃ m : ℝ, m = 0 ∧ (m + 1, 2 * m) = (1, 0) := 
sorry

end functional_expression_point_M_coordinates_l41_41182


namespace set_union_example_l41_41051

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem set_union_example : M ∪ N = {1, 2, 3, 4} := by
  sorry

end set_union_example_l41_41051


namespace max_single_painted_faces_l41_41984

theorem max_single_painted_faces (n : ℕ) (hn : n = 64) :
  ∃ max_cubes : ℕ, max_cubes = 32 := 
sorry

end max_single_painted_faces_l41_41984


namespace delta_value_l41_41880

theorem delta_value : ∃ Δ : ℤ, 4 * (-3) = Δ + 5 ∧ Δ = -17 := 
by
  use -17
  sorry

end delta_value_l41_41880


namespace product_of_two_numbers_l41_41938

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 21) (h2 : x^2 + y^2 = 527) : x * y = -43 :=
sorry

end product_of_two_numbers_l41_41938


namespace problem_statement_l41_41043

theorem problem_statement
  (x y : ℝ)
  (h1 : 4 * x + 2 * y = 12)
  (h2 : 2 * x + 4 * y = 20) :
  20 * x^2 + 24 * x * y + 20 * y^2 = 544 :=
  sorry

end problem_statement_l41_41043


namespace total_interest_obtained_l41_41832

-- Define the interest rates and face values
def interest_16 := 0.16 * 100
def interest_12 := 0.12 * 100
def interest_20 := 0.20 * 100

-- State the theorem to be proved
theorem total_interest_obtained : 
  interest_16 + interest_12 + interest_20 = 48 :=
by
  sorry

end total_interest_obtained_l41_41832


namespace value_range_f_in_0_to_4_l41_41127

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem value_range_f_in_0_to_4 :
  ∀ (x : ℝ), (0 < x ∧ x ≤ 4) → (1 ≤ f x ∧ f x ≤ 10) :=
sorry

end value_range_f_in_0_to_4_l41_41127


namespace total_cost_is_correct_l41_41385

-- Define the conditions
def piano_cost : ℝ := 500
def lesson_cost_per_lesson : ℝ := 40
def number_of_lessons : ℝ := 20
def discount_rate : ℝ := 0.25

-- Define the total cost of lessons before discount
def total_lesson_cost_before_discount : ℝ := lesson_cost_per_lesson * number_of_lessons

-- Define the discount amount
def discount_amount : ℝ := discount_rate * total_lesson_cost_before_discount

-- Define the total cost of lessons after discount
def total_lesson_cost_after_discount : ℝ := total_lesson_cost_before_discount - discount_amount

-- Define the total cost of everything
def total_cost : ℝ := piano_cost + total_lesson_cost_after_discount

-- The statement to be proven
theorem total_cost_is_correct : total_cost = 1100 := by
  sorry

end total_cost_is_correct_l41_41385


namespace tens_digit_19_pow_1987_l41_41314

theorem tens_digit_19_pow_1987 : (19 ^ 1987) % 100 / 10 = 3 := 
sorry

end tens_digit_19_pow_1987_l41_41314


namespace smaller_cube_edge_length_l41_41164

-- Given conditions
variables (s : ℝ) (volume_large_cube : ℝ) (n : ℝ)
-- n = 8 (number of smaller cubes), volume_large_cube = 1000 cm³

theorem smaller_cube_edge_length (h1 : n = 8) (h2 : volume_large_cube = 1000) :
  s^3 = volume_large_cube / n → s = 5 :=
by
  sorry

end smaller_cube_edge_length_l41_41164


namespace original_height_of_tree_l41_41904

theorem original_height_of_tree
  (current_height_in_inches : ℕ)
  (percent_taller : ℕ)
  (current_height_is_V := 180)
  (percent_taller_is_50 := 50) :
  (current_height_in_inches * 100) / (percent_taller + 100) / 12 = 10 := sorry

end original_height_of_tree_l41_41904


namespace evaluate_expression_l41_41166

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 - 3 = 31 :=
by
  sorry

end evaluate_expression_l41_41166


namespace domain_of_function_l41_41251

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x * (x - 1) ≥ 0) ↔ (x = 0 ∨ x ≥ 1) :=
by sorry

end domain_of_function_l41_41251


namespace sale_saving_percentage_l41_41142

theorem sale_saving_percentage (P : ℝ) : 
  let original_price := 8 * P
  let sale_price := 6 * P
  let amount_saved := original_price - sale_price
  let percentage_saved := (amount_saved / original_price) * 100
  percentage_saved = 25 :=
by
  sorry

end sale_saving_percentage_l41_41142


namespace geostationary_orbit_distance_l41_41688

noncomputable def distance_between_stations (earth_radius : ℝ) (orbit_altitude : ℝ) (num_stations : ℕ) : ℝ :=
  let θ : ℝ := 360 / num_stations
  let R : ℝ := earth_radius + orbit_altitude
  let sin_18 := (Real.sqrt 5 - 1) / 4
  2 * R * sin_18

theorem geostationary_orbit_distance :
  distance_between_stations 3960 22236 10 = -13098 + 13098 * Real.sqrt 5 :=
by
  sorry

end geostationary_orbit_distance_l41_41688


namespace inverse_of_f_at_10_l41_41607

noncomputable def f (x : ℝ) : ℝ := 1 + 3^(-x)

theorem inverse_of_f_at_10 :
  f⁻¹ 10 = -2 :=
sorry

end inverse_of_f_at_10_l41_41607


namespace freezer_temp_correct_l41_41623

variable (t_refrigeration : ℝ) (t_freezer : ℝ)

-- Given conditions
def refrigeration_temperature := t_refrigeration = 5
def freezer_temperature := t_freezer = -12

-- Goal: Prove that the freezer compartment's temperature is -12 degrees Celsius
theorem freezer_temp_correct : freezer_temperature t_freezer := by
  sorry

end freezer_temp_correct_l41_41623


namespace equation_of_the_line_l41_41780

noncomputable def line_equation (t : ℝ) : (ℝ × ℝ) := (3 * t + 6, 5 * t - 7)

theorem equation_of_the_line : ∃ m b : ℝ, (∀ t : ℝ, ∃ (x y : ℝ), line_equation t = (x, y) ∧ y = m * x + b) ∧ m = 5 / 3 ∧ b = -17 :=
by
  sorry

end equation_of_the_line_l41_41780


namespace arithmetic_sequence_sum_l41_41483

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℤ) (S_n : ℕ → ℤ),
  (∀ n : ℕ, S_n n = (n * (2 * (a_n 1) + (n - 1) * (a_n 2 - a_n 1))) / 2) →
  S_n 17 = 170 →
  a_n 7 + a_n 8 + a_n 12 = 30 := 
by
  sorry

end arithmetic_sequence_sum_l41_41483


namespace rectangle_length_is_4_l41_41500

theorem rectangle_length_is_4 (a : ℕ) (s : a = 4) (area_square : ℕ) 
(area_square_eq : area_square = a * a) 
(area_rectangle_eq : area_square = a * 4) : 
4 = a := by
  sorry

end rectangle_length_is_4_l41_41500


namespace henry_total_fee_8_bikes_l41_41031

def paint_fee := 5
def sell_fee := paint_fee + 8
def total_fee_per_bike := paint_fee + sell_fee
def total_fee (bikes : ℕ) := bikes * total_fee_per_bike

theorem henry_total_fee_8_bikes : total_fee 8 = 144 :=
by
  sorry

end henry_total_fee_8_bikes_l41_41031


namespace matthew_total_time_on_failure_day_l41_41766

-- Define the conditions as variables
def assembly_time : ℝ := 1 -- hours
def usual_baking_time : ℝ := 1.5 -- hours
def decoration_time : ℝ := 1 -- hours
def baking_factor : ℝ := 2 -- Factor by which baking time increased on that day

-- Prove that the total time taken is 5 hours
theorem matthew_total_time_on_failure_day : 
  (assembly_time + (usual_baking_time * baking_factor) + decoration_time) = 5 :=
by {
  sorry
}

end matthew_total_time_on_failure_day_l41_41766


namespace well_depth_and_rope_length_l41_41107

variables (h x : ℝ)

theorem well_depth_and_rope_length :
  (h / 3 = x + 4) ∧ (h / 4 = x + 1) → True := 
by
  intro h_eq x_eq
  sorry

end well_depth_and_rope_length_l41_41107


namespace new_concentration_of_mixture_l41_41834

theorem new_concentration_of_mixture :
  let v1 := 2
  let c1 := 0.25
  let v2 := 6
  let c2 := 0.40
  let V := 10
  let alcohol_amount_v1 := v1 * c1
  let alcohol_amount_v2 := v2 * c2
  let total_alcohol := alcohol_amount_v1 + alcohol_amount_v2
  let new_concentration := (total_alcohol / V) * 100
  new_concentration = 29 := 
by
  sorry

end new_concentration_of_mixture_l41_41834


namespace sum_of_reversed_integers_l41_41274

-- Definitions of properties and conditions
def reverse_digits (m n : ℕ) : Prop :=
  let to_digits (x : ℕ) : List ℕ := x.digits 10
  to_digits m = (to_digits n).reverse

-- The main theorem statement
theorem sum_of_reversed_integers
  (m n : ℕ)
  (h_rev: reverse_digits m n)
  (h_prod: m * n = 1446921630) :
  m + n = 79497 :=
sorry

end sum_of_reversed_integers_l41_41274


namespace binom_8_5_eq_56_l41_41015

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l41_41015


namespace second_number_is_three_l41_41937

theorem second_number_is_three (x y : ℝ) (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) : y = 3 :=
by
  -- To be proved: sorry for now
  sorry

end second_number_is_three_l41_41937


namespace scientific_notation_of_3100000_l41_41694

theorem scientific_notation_of_3100000 :
  ∃ (a : ℝ) (n : ℤ), 3100000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.1 ∧ n = 6 :=
  sorry

end scientific_notation_of_3100000_l41_41694


namespace min_x_4y_is_minimum_l41_41333

noncomputable def min_value_x_4y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : (1 / x) + (1 / (2 * y)) = 2) : ℝ :=
  x + 4 * y

theorem min_x_4y_is_minimum : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (1 / x + 1 / (2 * y) = 2) ∧ (x + 4 * y = (3 / 2) + Real.sqrt 2) :=
sorry

end min_x_4y_is_minimum_l41_41333


namespace solve_for_x_l41_41529

theorem solve_for_x (x : ℝ) (h : (4 / 7) * (1 / 5) * x = 2) : x = 17.5 :=
by
  -- Here we acknowledge the initial condition and conclusion without proving
  sorry

end solve_for_x_l41_41529


namespace inequality_solution_set_l41_41672

theorem inequality_solution_set (x : ℝ) :
  ((1 - x) * (x - 3) < 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end inequality_solution_set_l41_41672


namespace range_of_a_l41_41188

def f (x : ℝ) : ℝ := -x^5 - 3 * x^3 - 5 * x + 3

theorem range_of_a (a : ℝ) (h : f a + f (a - 2) > 6) : a < 1 :=
by
  -- Here, we would have to show the proof, but we're skipping it
  sorry

end range_of_a_l41_41188


namespace check_correct_conditional_expression_l41_41809
-- importing the necessary library for basic algebraic constructions and predicates

-- defining a predicate to denote the symbolic representation of conditional expressions validity
def valid_conditional_expression (expr: String) : Prop :=
  expr = "x <> 1" ∨ expr = "x > 1" ∨ expr = "x >= 1" ∨ expr = "x < 1" ∨ expr = "x <= 1" ∨ expr = "x = 1"

-- theorem to check for the valid conditional expression among the given options
theorem check_correct_conditional_expression :
  (valid_conditional_expression "1 < x < 2") = false ∧ 
  (valid_conditional_expression "x > < 1") = false ∧ 
  (valid_conditional_expression "x <> 1") = true ∧ 
  (valid_conditional_expression "x ≤ 1") = true :=
by sorry

end check_correct_conditional_expression_l41_41809


namespace solution_set_of_f_l41_41198

theorem solution_set_of_f (f : ℝ → ℝ) (h1 : ∀ x, 2 < deriv f x) (h2 : f (-1) = 2) :
  ∀ x, x > -1 → f x > 2 * x + 4 := by
  sorry

end solution_set_of_f_l41_41198


namespace find_second_dimension_l41_41629

theorem find_second_dimension (x : ℕ) 
    (h1 : 12 * x * 16 / (3 * 7 * 2) = 64) : 
    x = 14 := by
    sorry

end find_second_dimension_l41_41629


namespace find_vertex_l41_41151

noncomputable def parabola_vertex (x y : ℝ) : Prop :=
  2 * y^2 + 8 * y - 3 * x + 6 = 0

theorem find_vertex :
  ∃ (x y : ℝ), parabola_vertex x y ∧ x = -14/3 ∧ y = -2 :=
by
  sorry

end find_vertex_l41_41151


namespace linear_function_solution_l41_41883

open Function

theorem linear_function_solution (f : ℝ → ℝ)
  (h_lin : ∃ k b, k ≠ 0 ∧ ∀ x, f x = k * x + b)
  (h_comp : ∀ x, f (f x) = 4 * x - 1) :
  (∀ x, f x = 2 * x - 1 / 3) ∨ (∀ x, f x = -2 * x + 1) :=
by
  sorry

end linear_function_solution_l41_41883


namespace range_of_a_l41_41617

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 1) * x ^ 2 - (a - 1) * x - 1 < 0) ↔ a ∈ Set.Ioc (-3 : ℝ) 1 :=
by
  sorry

end range_of_a_l41_41617


namespace part1_part2_l41_41735

variables {α : Type*} [linear_ordered_field α]

-- Define the function f(x) = |x - 1| - 2|x + a|
def f (a x : α) : α := abs (x - 1) - 2 * abs (x + a)

-- Define the function g(x) = 1/2 * x + b
def g (b x : α) : α := (1 / 2) * x + b

-- Part 1: Prove the solution set of the inequality f(x) ≤ 0 given a = 1/2.
theorem part1 (x : α) :
  f (1 / 2 : α) x ≤ 0 ↔ x ≤ -2 ∨ (0 ≤ x ∧ x ≤ 1) := sorry

-- Part 2: Prove that 2b - 3a > 2 given a ≥ -1 
-- and the function g(x) is always above the function f(x).
theorem part2 (a b : α) (h : a ≥ -1)
  (h_above : ∀ x, g b x ≥ f a x) :
  2 * b - 3 * a > 2 := sorry

end part1_part2_l41_41735


namespace ninth_day_skate_time_l41_41475

-- Define the conditions
def first_4_days_skate_time : ℕ := 4 * 70
def second_4_days_skate_time : ℕ := 4 * 100
def total_days : ℕ := 9
def average_minutes_per_day : ℕ := 100

-- Define the theorem stating that Gage must skate 220 minutes on the ninth day to meet the average
theorem ninth_day_skate_time : 
  let total_minutes_needed := total_days * average_minutes_per_day
  let current_skate_time := first_4_days_skate_time + second_4_days_skate_time
  total_minutes_needed - current_skate_time = 220 := 
by
  -- Placeholder for the proof
  sorry

end ninth_day_skate_time_l41_41475


namespace binom_8_5_eq_56_l41_41007

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := 
by
sorry

end binom_8_5_eq_56_l41_41007


namespace each_person_paid_45_l41_41155

theorem each_person_paid_45 (total_bill : ℝ) (number_of_people : ℝ) (per_person_share : ℝ) 
    (h1 : total_bill = 135) 
    (h2 : number_of_people = 3) :
    per_person_share = 45 :=
by
  sorry

end each_person_paid_45_l41_41155


namespace owen_initial_turtles_l41_41524

variables (O J : ℕ)

-- Conditions
def johanna_turtles := J = O - 5
def owen_final_turtles := 2 * O + J / 2 = 50

-- Theorem statement
theorem owen_initial_turtles (h1 : johanna_turtles O J) (h2 : owen_final_turtles O J) : O = 21 :=
sorry

end owen_initial_turtles_l41_41524


namespace Jean_spots_l41_41508

/--
Jean the jaguar has a total of 60 spots.
Half of her spots are located on her upper torso.
One-third of the spots are located on her back and hindquarters.
Jean has 30 spots on her upper torso.
Prove that Jean has 10 spots located on her sides.
-/
theorem Jean_spots (TotalSpots UpperTorsoSpots BackHindquartersSpots SidesSpots : ℕ)
  (h_half : UpperTorsoSpots = TotalSpots / 2)
  (h_back : BackHindquartersSpots = TotalSpots / 3)
  (h_total_upper : UpperTorsoSpots = 30)
  (h_total : TotalSpots = 60) :
  SidesSpots = 10 :=
by
  sorry

end Jean_spots_l41_41508


namespace banknotes_sum_divisible_by_101_l41_41128

theorem banknotes_sum_divisible_by_101 (a b : ℕ) (h₀ : a ≠ b % 101) : 
  ∃ (m n : ℕ), m + n = 100 ∧ ∃ k l : ℕ, k ≤ m ∧ l ≤ n ∧ (k * a + l * b) % 101 = 0 :=
sorry

end banknotes_sum_divisible_by_101_l41_41128


namespace least_possible_third_side_l41_41361

theorem least_possible_third_side (a b : ℝ) (ha : a = 7) (hb : b = 24) : ∃ c, c = 24 ∧ a^2 - c^2 = 527  :=
by
  use (√527)
  sorry

end least_possible_third_side_l41_41361


namespace triangle_area_inscribed_circle_l41_41452

noncomputable def arc_length1 : ℝ := 6
noncomputable def arc_length2 : ℝ := 8
noncomputable def arc_length3 : ℝ := 10

noncomputable def circumference : ℝ := arc_length1 + arc_length2 + arc_length3
noncomputable def radius : ℝ := circumference / (2 * Real.pi)
noncomputable def angle_sum : ℝ := 360
noncomputable def angle1 : ℝ := 90 * Real.pi / 180
noncomputable def angle2 : ℝ := 120 * Real.pi / 180
noncomputable def angle3 : ℝ := 150 * Real.pi / 180

theorem triangle_area_inscribed_circle :
  let r := radius in
  let sin_a1 := Real.sin angle1 in
  let sin_a2 := Real.sin angle2 in
  let sin_a3 := Real.sin angle3 in
  (1 / 2) * r^2 * (sin_a1 + sin_a2 + sin_a3) = (72 * (1 + Real.sqrt 3)) / (Real.pi^2) := by
  sorry

end triangle_area_inscribed_circle_l41_41452


namespace midpoint_coordinates_l41_41383

theorem midpoint_coordinates (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 2) (hy1 : y1 = 10) (hx2 : x2 = 6) (hy2 : y2 = 2) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  mx = 4 ∧ my = 6 :=
by
  sorry

end midpoint_coordinates_l41_41383


namespace geometric_sequence_sum_l41_41217

theorem geometric_sequence_sum (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 1/2)
  (h2 : a 3 + a 4 = 1)
  (h_geom : ∀ n, a n + a (n+1) = (a 1 + a 2) * 2^(n-1)) :
  a 7 + a 8 + a 9 + a 10 = 12 := 
sorry

end geometric_sequence_sum_l41_41217


namespace citizen_income_l41_41463

noncomputable def income (I : ℝ) : Prop :=
  let P := 0.11 * 40000
  let A := I - 40000
  P + 0.20 * A = 8000

theorem citizen_income (I : ℝ) (h : income I) : I = 58000 := 
by
  -- proof steps go here
  sorry

end citizen_income_l41_41463


namespace find_larger_number_l41_41796

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 := by
  sorry

end find_larger_number_l41_41796


namespace quadratic_transformation_concept_l41_41434

theorem quadratic_transformation_concept :
  ∀ x : ℝ, (x-3)^2 - 4*(x-3) = 0 ↔ (x = 3 ∨ x = 7) :=
by
  intro x
  sorry

end quadratic_transformation_concept_l41_41434


namespace correct_X_Y_Z_l41_41379

def nucleotide_types (A_types C_types T_types : ℕ) : ℕ :=
  A_types + C_types + T_types

def lowest_stability_period := "interphase"

def separation_period := "late meiosis I or late meiosis II"

theorem correct_X_Y_Z :
  nucleotide_types 2 2 1 = 3 ∧ 
  lowest_stability_period = "interphase" ∧ 
  separation_period = "late meiosis I or late meiosis II" :=
by
  sorry

end correct_X_Y_Z_l41_41379


namespace sum_odd_divisors_90_eq_78_l41_41959

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l41_41959


namespace constants_solution_l41_41587

theorem constants_solution (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 2 → 
    (5 * x^2 / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2)) ↔ 
    (A = 20 ∧ B = -15 ∧ C = -10) :=
by
  sorry

end constants_solution_l41_41587


namespace larger_number_solution_l41_41794

theorem larger_number_solution (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
by
  sorry

end larger_number_solution_l41_41794


namespace smallest_n_value_l41_41301

theorem smallest_n_value :
  ∃ n, (∀ (sheets : Fin 2000 → Fin 4 → Fin 4),
        (∀ (n : Nat) (h : n ≤ 2000) (a b c d : Fin n) (h' : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d),
          ∃ (i j k : Fin 5), sheets a i = sheets b i ∧ sheets a j = sheets b j ∧ sheets a k = sheets b k → ¬ sheets a i = sheets c i ∧ ¬ sheets b j = sheets c j ∧ ¬ sheets a k = sheets c k)) ↔ n = 25 :=
sorry

end smallest_n_value_l41_41301


namespace min_sum_p_q_r_s_l41_41519

theorem min_sum_p_q_r_s (p q r s : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
    (h1 : 2 * p = 10 * p - 15 * q)
    (h2 : 2 * q = 6 * p - 9 * q)
    (h3 : 3 * r = 10 * r - 15 * s)
    (h4 : 3 * s = 6 * r - 9 * s) : p + q + r + s = 45 := by
  sorry

end min_sum_p_q_r_s_l41_41519


namespace arithmetic_sequence_solution_l41_41632

theorem arithmetic_sequence_solution :
  ∃ (a1 d : ℤ), 
    (a1 + 3*d + (a1 + 4*d) + (a1 + 5*d) + (a1 + 6*d) = 56) ∧
    ((a1 + 3*d) * (a1 + 6*d) = 187) ∧
    (
      (a1 = 5 ∧ d = 2) ∨
      (a1 = 23 ∧ d = -2)
    ) :=
by
  sorry

end arithmetic_sequence_solution_l41_41632


namespace total_students_is_45_l41_41154

theorem total_students_is_45
  (students_burgers : ℕ) 
  (total_students : ℕ) 
  (hb : students_burgers = 30) 
  (ht : total_students = 45) : 
  total_students = 45 :=
by
  sorry

end total_students_is_45_l41_41154


namespace calculation_l41_41842

theorem calculation : (3 * 4 * 5) * ((1 / 3 : ℚ) + (1 / 4 : ℚ) - (1 / 5 : ℚ)) = 23 := by
  sorry

end calculation_l41_41842


namespace min_third_side_length_l41_41363

theorem min_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 24) : 
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ b^2 = a^2 + c^2 ∨  a^2 = b^2 + c^2) ∧ c = 7 :=
sorry

end min_third_side_length_l41_41363


namespace distance_between_Jay_and_Paul_l41_41636

theorem distance_between_Jay_and_Paul
  (initial_distance : ℕ)
  (jay_speed : ℕ)
  (paul_speed : ℕ)
  (time : ℕ)
  (jay_distance_walked : ℕ)
  (paul_distance_walked : ℕ) :
  initial_distance = 3 →
  jay_speed = 1 / 20 →
  paul_speed = 3 / 40 →
  time = 120 →
  jay_distance_walked = jay_speed * time →
  paul_distance_walked = paul_speed * time →
  initial_distance + jay_distance_walked + paul_distance_walked = 18 := by
  sorry

end distance_between_Jay_and_Paul_l41_41636


namespace children_ticket_price_difference_l41_41561

noncomputable def regular_ticket_price : ℝ := 9
noncomputable def total_amount_given : ℝ := 2 * 20
noncomputable def total_change_received : ℝ := 1
noncomputable def num_adults : ℕ := 2
noncomputable def num_children : ℕ := 3
noncomputable def total_cost_of_tickets : ℝ := total_amount_given - total_change_received
noncomputable def children_ticket_cost := (total_cost_of_tickets - num_adults * regular_ticket_price) / num_children

theorem children_ticket_price_difference :
  (regular_ticket_price - children_ticket_cost) = 2 := by
  sorry

end children_ticket_price_difference_l41_41561


namespace sequence_x_value_l41_41895

theorem sequence_x_value (p q r x : ℕ) 
  (h1 : 13 = 5 + p + q) 
  (h2 : r = p + q + 13) 
  (h3 : x = 13 + r + 40) : 
  x = 74 := 
by 
  sorry

end sequence_x_value_l41_41895


namespace find_AB_value_l41_41184

theorem find_AB_value :
  ∃ A B : ℕ, (A + B = 5 ∧ (A - B) % 11 = 5 % 11) ∧
           990 * 991 * 992 * 993 = 966428 * 100000 + A * 9100 + B * 40 :=
sorry

end find_AB_value_l41_41184


namespace find_x_l41_41318

theorem find_x (x : ℝ) (hx_pos : x > 0) (hx_ceil_eq : ⌈x⌉ = 15) : x = 14 :=
by
  -- Define the condition
  have h_eq : ⌈x⌉ * x = 210 := sorry
  -- Prove that the only solution is x = 14
  sorry

end find_x_l41_41318


namespace necessary_and_sufficient_condition_x_eq_1_l41_41819

theorem necessary_and_sufficient_condition_x_eq_1
    (x : ℝ) :
    (x = 1 ↔ x^2 - 2 * x + 1 = 0) :=
sorry

end necessary_and_sufficient_condition_x_eq_1_l41_41819


namespace range_of_a_l41_41610

variable (a : ℝ)
def p := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def q := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (hpq_or : p a ∨ q a) (hpq_and_false : ¬ (p a ∧ q a)) : 
    a ∈ Set.Iio 0 ∪ Set.Ioo (1/4) 4 :=
by
  sorry

end range_of_a_l41_41610


namespace alan_total_payment_l41_41571

-- Define the costs of CDs
def cost_AVN : ℝ := 12
def cost_TheDark : ℝ := 2 * cost_AVN
def cost_TheDark_total : ℝ := 2 * cost_TheDark
def cost_other_CDs : ℝ := cost_AVN + cost_TheDark_total
def cost_90s : ℝ := 0.4 * cost_other_CDs
def total_cost : ℝ := cost_AVN + cost_TheDark_total + cost_90s

-- Formulate the main statement
theorem alan_total_payment :
  total_cost = 84 := by
  sorry

end alan_total_payment_l41_41571


namespace football_problem_l41_41473

-- Definitions based on conditions
def total_balls (x y : Nat) : Prop := x + y = 200
def total_cost (x y : Nat) : Prop := 80 * x + 60 * y = 14400
def football_A_profit_per_ball : Nat := 96 - 80
def football_B_profit_per_ball : Nat := 81 - 60
def total_profit (x y : Nat) : Nat :=
  football_A_profit_per_ball * x + football_B_profit_per_ball * y

-- Lean statement proving the conditions lead to the solution
theorem football_problem
  (x y : Nat)
  (h1 : total_balls x y)
  (h2 : total_cost x y)
  (h3 : x = 120)
  (h4 : y = 80) :
  total_profit x y = 3600 := by
  sorry

end football_problem_l41_41473


namespace intersection_point_l41_41917

theorem intersection_point (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ d) :
  let x := (d - c) / (2 * b)
  let y := (a * (d - c)^2) / (4 * b^2) + (d + c) / 2
  (ax^2 + bx + c = y) ∧ (ax^2 - bx + d = y) :=
by
  sorry

end intersection_point_l41_41917


namespace find_leak_rate_l41_41079

-- Conditions in Lean 4
def pool_capacity : ℝ := 60
def hose_rate : ℝ := 1.6
def fill_time : ℝ := 40

-- Define the leak rate calculation
def leak_rate (L : ℝ) : Prop :=
  pool_capacity = (hose_rate - L) * fill_time

-- The main theorem we want to prove
theorem find_leak_rate : ∃ L, leak_rate L ∧ L = 0.1 := by
  sorry

end find_leak_rate_l41_41079


namespace barbed_wire_cost_l41_41928

noncomputable def total_cost_barbed_wire (area : ℕ) (cost_per_meter : ℝ) (gate_width : ℕ) : ℝ :=
  let s := Real.sqrt area
  let perimeter := 4 * s - 2 * gate_width
  perimeter * cost_per_meter

theorem barbed_wire_cost :
  total_cost_barbed_wire 3136 3.5 1 = 777 := by
  sorry

end barbed_wire_cost_l41_41928


namespace determine_prices_l41_41658

variable (num_items : ℕ) (cost_keychains cost_plush : ℕ) (x : ℚ) (unit_price_keychains unit_price_plush : ℚ)

noncomputable def price_equation (x : ℚ) : Prop :=
  (cost_keychains / x) + (cost_plush / (1.5 * x)) = num_items

theorem determine_prices 
  (h1 : num_items = 15)
  (h2 : cost_keychains = 240)
  (h3 : cost_plush = 180)
  (h4 : price_equation num_items cost_keychains cost_plush x)
  (hx : x = 24) :
  unit_price_keychains = 24 ∧ unit_price_plush = 36 :=
  by
    sorry

end determine_prices_l41_41658


namespace cone_lateral_surface_area_l41_41868

-- Define the radius and slant height as given constants
def radius : ℝ := 3
def slant_height : ℝ := 5

-- Definition of the lateral surface area
def lateral_surface_area (r l : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * r in
  (circumference * l) / 2

-- The proof problem statement in Lean 4
theorem cone_lateral_surface_area : lateral_surface_area radius slant_height = 15 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l41_41868


namespace solve_for_b_l41_41619

theorem solve_for_b (x y b : ℝ) (h1: 4 * x + y = b) (h2: 3 * x + 4 * y = 3 * b) (hx: x = 3) : b = 39 :=
sorry

end solve_for_b_l41_41619


namespace area_of_unpainted_section_l41_41539

-- Define the conditions
def board1_width : ℝ := 5
def board2_width : ℝ := 7
def cross_angle : ℝ := 45
def negligible_holes : Prop := true

-- The main statement
theorem area_of_unpainted_section (h1 : board1_width = 5) (h2 : board2_width = 7) (h3 : cross_angle = 45) (h4 : negligible_holes) : 
  ∃ (area : ℝ), area = 35 := 
sorry

end area_of_unpainted_section_l41_41539


namespace gcd_special_case_l41_41415

theorem gcd_special_case (m n : ℕ) (h : Nat.gcd m n = 1) :
  Nat.gcd (m + 2000 * n) (n + 2000 * m) = 2000^2 - 1 :=
sorry

end gcd_special_case_l41_41415


namespace find_x_value_l41_41263

def acid_solution (m : ℕ) (x : ℕ) (h : m > 25) : Prop :=
  let initial_acid := m^2 / 100
  let total_volume := m + x
  let new_acid_concentration := (m - 5) / 100 * (m + x)
  initial_acid = new_acid_concentration

theorem find_x_value (m : ℕ) (h : m > 25) (x : ℕ) :
  (acid_solution m x h) → x = 5 * m / (m - 5) :=
sorry

end find_x_value_l41_41263


namespace kelly_games_left_l41_41640

theorem kelly_games_left (initial_games : Nat) (given_away : Nat) (remaining_games : Nat) 
  (h1 : initial_games = 106) (h2 : given_away = 64) : remaining_games = 42 := by
  sorry

end kelly_games_left_l41_41640


namespace PlayStation_cost_l41_41242

def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def price_per_game : ℝ := 7.5
def games_to_sell : ℕ := 20
def total_gift_money : ℝ := birthday_money + christmas_money
def total_games_money : ℝ := games_to_sell * price_per_game
def total_money : ℝ := total_gift_money + total_games_money

theorem PlayStation_cost : total_money = 500 := by
  sorry

end PlayStation_cost_l41_41242


namespace fraction_of_40_l41_41947

theorem fraction_of_40 : (3 / 4) * 40 = 30 :=
by
  -- We'll add the 'sorry' here to indicate that this is the proof part which is not required.
  sorry

end fraction_of_40_l41_41947


namespace BC_length_l41_41225

def triangle_ABC (A B C : Type)
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)] : Prop :=
  ∃ (X : A), (has_dist B X (coe (X.dist B))) ∧ (has_dist C X (coe (X.dist C))) ∧
  ∀ (x y : ℕ), x = X.dist B ∧ y = X.dist C → x + y = 61

theorem BC_length {A B C : Type}
  [metric_space A]
  [has_dist A (coe 86)]
  [has_dist A (coe 97)]
  (h : triangle_ABC A B C) : 
  ∃ (x y : ℕ), x + y = 61 := sorry

end BC_length_l41_41225


namespace factorization_correct_l41_41853

theorem factorization_correct : ∃ a b : ℤ, (5*y + a)*(y + b) = 5*y^2 + 17*y + 6 ∧ a - b = -1 := by
  sorry

end factorization_correct_l41_41853


namespace modified_prism_surface_area_l41_41161

theorem modified_prism_surface_area :
  let original_surface_area := 2 * (2 * 4 + 2 * 5 + 4 * 5)
  let modified_surface_area := original_surface_area + 5
  modified_surface_area = original_surface_area + 5 :=
by
  -- set the original dimensions
  let l := 2
  let w := 4
  let h := 5
  -- calculate original surface area
  let SA_original := 2 * (l * w + l * h + w * h)
  -- calculate modified surface area
  let SA_new := SA_original + 5
  -- assert the relationship
  have : SA_new = SA_original + 5 := rfl
  exact this

end modified_prism_surface_area_l41_41161


namespace total_people_on_boats_l41_41975

theorem total_people_on_boats (boats : ℕ) (people_per_boat : ℕ) (h_boats : boats = 5) (h_people : people_per_boat = 3) : boats * people_per_boat = 15 :=
by
  sorry

end total_people_on_boats_l41_41975


namespace andrey_stamps_l41_41772

theorem andrey_stamps :
  ∃ (x : ℕ), 
    x % 3 = 1 ∧ 
    x % 5 = 3 ∧ 
    x % 7 = 5 ∧ 
    150 < x ∧ 
    x ≤ 300 ∧ 
    x = 208 :=
begin
  sorry
end

end andrey_stamps_l41_41772


namespace matthew_total_time_on_malfunctioning_day_l41_41765

-- Definitions for conditions
def assembling_time : ℝ := 1
def normal_baking_time : ℝ := 1.5
def malfunctioning_baking_time : ℝ := 2 * normal_baking_time
def decorating_time : ℝ := 1

-- The theorem statement
theorem matthew_total_time_on_malfunctioning_day :
  assembling_time + malfunctioning_baking_time + decorating_time = 5 :=
by
  -- This is where the proof would go
  sorry

end matthew_total_time_on_malfunctioning_day_l41_41765


namespace distance_from_pole_l41_41897

-- Define the structure for polar coordinates.
structure PolarCoordinates where
  r : ℝ
  θ : ℝ

-- Define point A with its polar coordinates.
def A : PolarCoordinates := { r := 3, θ := -4 }

-- State the problem to prove that the distance |OA| is 3.
theorem distance_from_pole (A : PolarCoordinates) : A.r = 3 :=
by {
  sorry
}

end distance_from_pole_l41_41897


namespace complex_division_evaluation_l41_41849

open Complex

theorem complex_division_evaluation :
  (2 : ℂ) / (I * (3 - I)) = (1 / 5 : ℂ) - (3 / 5) * I :=
by
  sorry

end complex_division_evaluation_l41_41849


namespace quadratic_root_inequality_l41_41719

theorem quadratic_root_inequality (a : ℝ) :
  2015 < a ∧ a < 2017 ↔ 
  ∃ x₁ x₂ : ℝ, (2 * x₁^2 - 2016 * (x₁ - 2016 + a) - 1 = a^2) ∧ 
               (2 * x₂^2 - 2016 * (x₂ - 2016 + a) - 1 = a^2) ∧
               x₁ < a ∧ a < x₂ :=
sorry

end quadratic_root_inequality_l41_41719


namespace range_of_a_decreasing_function_l41_41338

theorem range_of_a_decreasing_function (a : ℝ) :
  (∀ x < 1, ∀ y < x, (3 * a - 1) * x + 4 * a ≥ (3 * a - 1) * y + 4 * a) ∧ 
  (∀ x ≥ 1, ∀ y > x, -a * x ≤ -a * y) ∧
  (∀ x < 1, ∀ y ≥ 1, (3 * a - 1) * x + 4 * a ≥ -a * y)  →
  (1 / 8 : ℝ) ≤ a ∧ a < (1 / 3 : ℝ) :=
sorry

end range_of_a_decreasing_function_l41_41338


namespace base_7_to_10_equivalence_l41_41267

theorem base_7_to_10_equivalence : 
  let n := 6 * 7^4 + 5 * 7^3 + 4 * 7^2 + 3 * 7^1 + 2 * 7^0
  in n = 16340 :=
by
  let n := 6 * 7^4 + 5 * 7^3 + 4 * 7^2 + 3 * 7^1 + 2 * 7^0
  show n = 16340
  -- The proof would go here, but we use sorry for the placeholder
  sorry

end base_7_to_10_equivalence_l41_41267


namespace find_HCF_of_two_numbers_l41_41105

theorem find_HCF_of_two_numbers (a b H : ℕ) 
  (H_HCF : Nat.gcd a b = H) 
  (H_LCM_Factors : Nat.lcm a b = H * 13 * 14) 
  (H_largest_number : 322 = max a b) : 
  H = 14 :=
sorry

end find_HCF_of_two_numbers_l41_41105


namespace find_cartesian_equation_of_C_find_polar_equation_of_perpendicular_line_l41_41034

-- Given conditions translated to Lean
def transformed_curve (theta : ℝ) : ℝ × ℝ :=
  (2 * Real.cos theta, Real.sin theta)

-- Cartesian equation derived from the transformed coordinates
def cartesian_equation_C (x y : ℝ) : Prop :=
  x ^ 2 / 4 + y ^ 2 = 1

-- Points of intersection of the line and the curve
def line_eq (x y : ℝ) : Prop :=
  x + 2 * y = 2

def point_P1 : ℝ × ℝ := (2, 0)
def point_P2 : ℝ × ℝ := (0, 1)

def midpoint (P1 P2 : ℝ × ℝ) : ℝ × ℝ :=
  ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

def perpendicular_line_through_midpoint (midpoint : ℝ × ℝ) (slope : ℝ) : ℝ × ℝ → Prop
| (x, y) => y = slope * (x - midpoint.1) + midpoint.2

def polar_eq (rho theta : ℝ) : ℝ :=
  rho = 3 / (4 * Real.cos theta - 2 * Real.sin theta)

-- Mathematical problem statements in Lean
theorem find_cartesian_equation_of_C :
  ∀ (theta x y : ℝ),
    transformed_curve theta = (x, y) →
    cartesian_equation_C x y :=
  sorry

theorem find_polar_equation_of_perpendicular_line :
  ∀ (x y : ℝ),
    line_eq x y →
    (x = point_P1.1 ∧ y = point_P1.2) ∨ (x = point_P2.1 ∧ y = point_P2.2) →
    let mid := midpoint point_P1 point_P2 in
    let perp_line := perpendicular_line_through_midpoint mid 2 in
    polar_eq (4 * (mid.1) - 2 * (mid.2)) (Real.atan mid.1/ mid.2) :=
  sorry

end find_cartesian_equation_of_C_find_polar_equation_of_perpendicular_line_l41_41034


namespace number_of_lightsabers_ordered_l41_41547

def cost_per_metal_arc : Nat := 400
def metal_arcs_per_lightsaber : Nat := 2
def assembly_time_per_lightsaber : Nat := 1 / 20
def combined_cost_per_hour : Nat := 200 + 100
def total_cost : Nat := 65200

theorem number_of_lightsabers_ordered (x : Nat) :
  let cost_of_metal_arcs := metal_arcs_per_lightsaber * cost_per_metal_arc
  let assembly_cost_per_lightsaber := combined_cost_per_hour / 20
  let total_cost_per_lightsaber := cost_of_metal_arcs + assembly_cost_per_lightsaber
  x * total_cost_per_lightsaber = total_cost →
  x = 80 :=
by
  let cost_of_metal_arcs := metal_arcs_per_lightsaber * cost_per_metal_arc
  let assembly_cost_per_lightsaber := combined_cost_per_hour / 20
  let total_cost_per_lightsaber := cost_of_metal_arcs + assembly_cost_per_lightsaber
  assume h : x * total_cost_per_lightsaber = total_cost
  sorry

end number_of_lightsabers_ordered_l41_41547


namespace mr_bird_speed_to_work_l41_41239

theorem mr_bird_speed_to_work (
  d t : ℝ
) (h1 : d = 45 * (t + 4 / 60)) 
  (h2 : d = 55 * (t - 2 / 60))
  (h3 : t = 29 / 60)
  (d_eq : d = 24.75) :
  (24.75 / (29 / 60)) = 51.207 := 
sorry

end mr_bird_speed_to_work_l41_41239


namespace binomial_parameters_unique_l41_41605

theorem binomial_parameters_unique (n : ℕ) (p : ℝ) (ξ : ℕ → ℝ) 
  (h₁ : ξ ~ Binomial n p) 
  (h₂ : E ξ = 1.6) 
  (h₃ : Var ξ = 1.28) 
: n = 8 ∧ p = 0.2 := 
sorry

end binomial_parameters_unique_l41_41605


namespace AB_not_selected_together_l41_41496

open Finset

-- Define the five points
def points : Finset ℕ := {0, 1, 2, 3, 4}

-- Define the specific points A and B
def A := 0
def B := 1

-- Define the event of selecting 3 out of 5 points
def selections := points.powerset.filter (λ s, s.card = 3)

-- Define the event that A and B are selected together
def AB_selected := selections.filter (λ s, A ∈ s ∧ B ∈ s)

-- Total number of selections
def total_selections := selections.card

-- Number of selections where A and B are selected together
def AB_selections := AB_selected.card

-- Probability calculation
def probability_AB_not_selected_together := 1 - (AB_selections / total_selections : ℚ)

-- Theorem statement
theorem AB_not_selected_together : 
  probability_AB_not_selected_together = 7 / 10 := 
sorry

end AB_not_selected_together_l41_41496


namespace min_distance_PQ_l41_41666

theorem min_distance_PQ :
  let P := (1, 0)
  let Q := λ t : ℝ, (3 * t, sqrt 3 + sqrt 3 * t)
  let dist := λ t : ℝ, sqrt ((3 * t - 1)^2 + (sqrt 3 + sqrt 3 * t)^2) - 1
  ∃ t : ℝ, dist t = 1 := sorry

end min_distance_PQ_l41_41666


namespace graph_does_not_pass_first_quadrant_l41_41330

variables {a b x : ℝ}

theorem graph_does_not_pass_first_quadrant 
  (h₁ : 0 < a ∧ a < 1) 
  (h₂ : b < -1) : 
  ¬ ∃ x : ℝ, 0 < x ∧ 0 < a^x + b :=
sorry

end graph_does_not_pass_first_quadrant_l41_41330


namespace murtha_pebbles_after_20_days_l41_41913

/- Define the sequence function for the pebbles collected each day -/
def pebbles_collected_day (n : ℕ) : ℕ :=
  if (n = 0) then 0 else 1 + pebbles_collected_day (n - 1)

/- Define the total pebbles collected by the nth day -/
def total_pebbles_collected (n : ℕ) : ℕ :=
  (n * (pebbles_collected_day n)) / 2

/- Define the total pebbles given away by the nth day -/
def pebbles_given_away (n : ℕ) : ℕ :=
  (n / 5) * 3

/- Define the net total of pebbles Murtha has on the nth day -/
def pebbles_net (n : ℕ) : ℕ :=
  total_pebbles_collected (n + 1) - pebbles_given_away (n + 1)

/- The main theorem about the pebbles Murtha has after the 20th day -/
theorem murtha_pebbles_after_20_days : pebbles_net 19 = 218 := 
  by sorry

end murtha_pebbles_after_20_days_l41_41913


namespace cards_problem_l41_41884

theorem cards_problem : 
  ∀ (cards people : ℕ),
  cards = 60 →
  people = 8 →
  ∃ fewer_people : ℕ,
  (∀ p: ℕ, p < people → (p < fewer_people → cards/people < 8)) ∧ 
  fewer_people = 4 := 
by 
  intros cards people h_cards h_people
  use 4
  sorry

end cards_problem_l41_41884


namespace freezer_temperature_is_minus_12_l41_41624

theorem freezer_temperature_is_minus_12 (refrigeration_temp freezer_temp : ℤ) (h1 : refrigeration_temp = 5) (h2 : freezer_temp = -12) : freezer_temp = -12 :=
by sorry

end freezer_temperature_is_minus_12_l41_41624


namespace binom_8_5_eq_56_l41_41016

theorem binom_8_5_eq_56 : Nat.choose 8 5 = 56 := by
  sorry

end binom_8_5_eq_56_l41_41016


namespace inequality_problem_l41_41237

variable (a b c d : ℝ)

open Real

theorem inequality_problem 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hprod : a * b * c * d = 1) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + d)) + 1 / (d * (1 + a)) ≥ 2 := 
by 
  sorry

end inequality_problem_l41_41237


namespace quadratic_root_a_l41_41038

theorem quadratic_root_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 4 = 0 ∧ x = 1) → a = -5 :=
by
  intro h
  have h1 : (1:ℝ)^2 + a * (1:ℝ) + 4 = 0 := sorry
  linarith

end quadratic_root_a_l41_41038


namespace BC_length_l41_41224

-- Define the given values and conditions
variable (A B C X : Type)
variable (AB AC AX BX CX : ℕ)
variable [inhabited A] [inhabited B] [inhabited C] [inhabited X]

-- Assume the lengths of AB and AC
axiom h_AB : AB = 86
axiom h_AC : AC = 97

-- Assume the circle centered at A with radius AB intersects BC at B and X
axiom h_circle : AX = AB

-- Assume BX and CX are integers
axiom h_BX_integral : ∃ (x : ℕ), BX = x
axiom h_CX_integral : ∃ (y : ℕ), CX = y

-- The statement to prove that the length of BC is 61
theorem BC_length : (∃ (x y : ℕ), BX = x ∧ CX = y ∧ x + y = 61) :=
by
  sorry

end BC_length_l41_41224


namespace jerrys_breakfast_calories_l41_41511

theorem jerrys_breakfast_calories 
    (num_pancakes : ℕ) (calories_per_pancake : ℕ) 
    (num_bacon : ℕ) (calories_per_bacon : ℕ) 
    (num_cereal : ℕ) (calories_per_cereal : ℕ) 
    (calories_total : ℕ) :
    num_pancakes = 6 →
    calories_per_pancake = 120 →
    num_bacon = 2 →
    calories_per_bacon = 100 →
    num_cereal = 1 →
    calories_per_cereal = 200 →
    calories_total = num_pancakes * calories_per_pancake
                   + num_bacon * calories_per_bacon
                   + num_cereal * calories_per_cereal →
    calories_total = 1120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  assumption

end jerrys_breakfast_calories_l41_41511


namespace determine_m_from_quadratic_l41_41604

def is_prime (n : ℕ) := 2 ≤ n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem determine_m_from_quadratic (x1 x2 m : ℕ) (hx1_prime : is_prime x1) (hx2_prime : is_prime x2) 
    (h_roots : x1 + x2 = 1999) (h_product : x1 * x2 = m) : 
    m = 3994 := 
by 
    sorry

end determine_m_from_quadratic_l41_41604


namespace triangle_third_side_l41_41061

theorem triangle_third_side (a b x : ℝ) (h : (a - 3) ^ 2 + |b - 4| = 0) :
  x = 5 ∨ x = Real.sqrt 7 :=
sorry

end triangle_third_side_l41_41061


namespace BC_length_l41_41219

-- Define the given triangle and circle conditions
variables (A B C X : Type) (AB AC BX CX : ℤ)
axiom AB_value : AB = 86
axiom AC_value : AC = 97
axiom circle_center_radius : ∃ (A : Type), ∃ (radius : ℤ), radius = AB ∧ ∃ (points : Set Type), points = {B, X} ∧ ∀ (P : Type), P ∈ points → dist A P = radius
axiom BX_CX_integers : ∃ (x y : ℤ), BX = x ∧ CX = y

-- Define calculations using the Power of a Point theorem
theorem BC_length :
  ∀ (y: ℤ) (x: ℤ), y(y + x) = AC^2 - AB^2 → x + y = 61 :=
by
  intros y x h
  have h1 : 97^2 = 9409, by norm_num,
  have h2 : 86^2 = 7396, by norm_num,
  rw [AB_value, AC_value] at h,
  rw [h1, h2] at h,
  calc y(y + x) = 2013 := by {exact h}
  -- The human verification part is skipped since we only need the statement here
  sorry

end BC_length_l41_41219


namespace bonus_distributed_correctly_l41_41407

def amount_received (A B C D E F : ℝ) :=
  -- Conditions
  (A = 2 * B) ∧ 
  (B = C) ∧ 
  (D = 2 * B - 1500) ∧ 
  (E = C + 2000) ∧ 
  (F = 1/2 * (A + D)) ∧ 
  -- Total bonus amount
  (A + B + C + D + E + F = 25000)

theorem bonus_distributed_correctly :
  ∃ (A B C D E F : ℝ), 
    amount_received A B C D E F ∧ 
    A = 4950 ∧ 
    B = 2475 ∧ 
    C = 2475 ∧ 
    D = 3450 ∧ 
    E = 4475 ∧ 
    F = 4200 :=
sorry

end bonus_distributed_correctly_l41_41407


namespace simplify_cubicroot_1600_l41_41443

theorem simplify_cubicroot_1600 : ∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ (c^3 * d = 1600) ∧ (c + d = 102) := 
by 
  sorry

end simplify_cubicroot_1600_l41_41443


namespace shaded_region_perimeter_l41_41781

theorem shaded_region_perimeter :
  let side_length := 1
  let diagonal_length := Real.sqrt 2 * side_length
  let arc_TRU_length := (1 / 4) * (2 * Real.pi * diagonal_length)
  let arc_VPW_length := (1 / 4) * (2 * Real.pi * side_length)
  let arc_UV_length := (1 / 4) * (2 * Real.pi * (Real.sqrt 2 - side_length))
  let arc_WT_length := (1 / 4) * (2 * Real.pi * (Real.sqrt 2 - side_length))
  (arc_TRU_length + arc_VPW_length + arc_UV_length + arc_WT_length) = (2 * Real.sqrt 2 - 1) * Real.pi :=
by
  sorry

end shaded_region_perimeter_l41_41781


namespace height_of_trapezium_l41_41026

-- Define the lengths of the parallel sides
def length_side1 : ℝ := 10
def length_side2 : ℝ := 18

-- Define the given area of the trapezium
def area_trapezium : ℝ := 210

-- The distance between the parallel sides (height) we want to prove
def height_between_sides : ℝ := 15

-- State the problem as a theorem in Lean: prove that the height is correct
theorem height_of_trapezium :
  (1 / 2) * (length_side1 + length_side2) * height_between_sides = area_trapezium :=
by
  sorry

end height_of_trapezium_l41_41026


namespace intersection_set_eq_l41_41739

-- Define M
def M : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 9) = 1 }

-- Define N
def N : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 / 4) + (p.2 / 3) = 1 }

-- Define the intersection of M and N
def M_intersection_N := { x : ℝ | -4 ≤ x ∧ x ≤ 4 }

-- The theorem to be proved
theorem intersection_set_eq : 
  { p : ℝ × ℝ | p ∈ M ∧ p ∈ N } = { p : ℝ × ℝ | p.1 ∈ M_intersection_N } :=
sorry

end intersection_set_eq_l41_41739


namespace symmetric_circle_eq_l41_41110

-- Define the original circle equation
def originalCircle (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 4

-- Define the equation of the circle symmetric to the original with respect to the y-axis
def symmetricCircle (x y : ℝ) : Prop :=
  (x + 1) ^ 2 + (y - 2) ^ 2 = 4

-- Theorem to prove that the symmetric circle equation is correct
theorem symmetric_circle_eq :
  ∀ x y : ℝ, originalCircle x y → symmetricCircle (-x) y := 
by
  sorry

end symmetric_circle_eq_l41_41110


namespace depth_of_channel_l41_41277

theorem depth_of_channel (h : ℝ) 
  (top_width : ℝ := 12) (bottom_width : ℝ := 6) (area : ℝ := 630) :
  1 / 2 * (top_width + bottom_width) * h = area → h = 70 :=
sorry

end depth_of_channel_l41_41277


namespace fatima_donates_75_square_inches_l41_41715

theorem fatima_donates_75_square_inches :
  ∀ (cloth: ℚ), cloth = 100 →
  (∃ (c1 c2 c3: ℚ), c1 = cloth / 2 ∧ c2 = c1 / 2 ∧ c3 = 75) →
  (c1 + c2 = c3) :=
by
  assume cloth
  assume h1 : cloth = 100
  assume h2 : ∃ (c1 c2 c3: ℚ), c1 = cloth / 2 ∧ c2 = c1 / 2 ∧ c3 = 75
  sorry

end fatima_donates_75_square_inches_l41_41715


namespace male_student_number_l41_41560

theorem male_student_number (year class_num student_num : ℕ) (h_year : year = 2011) (h_class : class_num = 6) (h_student : student_num = 23) : 
  (100000 * year + 1000 * class_num + 10 * student_num + 1 = 116231) :=
by
  sorry

end male_student_number_l41_41560


namespace cos_squared_plus_twice_sin_double_alpha_l41_41611

theorem cos_squared_plus_twice_sin_double_alpha (α : ℝ) (h : Real.tan α = 3 / 4) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end cos_squared_plus_twice_sin_double_alpha_l41_41611


namespace smallest_positive_angle_l41_41591

open Real

theorem smallest_positive_angle (θ : ℝ) :
  cos θ = sin (50 * (π / 180)) + cos (32 * (π / 180)) - sin (22 * (π / 180)) - cos (16 * (π / 180)) →
  θ = 90 * (π / 180) :=
by
  sorry

end smallest_positive_angle_l41_41591


namespace average_percentage_reduction_equation_l41_41824

theorem average_percentage_reduction_equation (x : ℝ) : 200 * (1 - x)^2 = 162 :=
by 
  sorry

end average_percentage_reduction_equation_l41_41824


namespace definite_integral_value_l41_41554

theorem definite_integral_value :
  (∫ x in (0 : ℝ)..Real.arctan (1/3), (8 + Real.tan x) / (18 * Real.sin x^2 + 2 * Real.cos x^2)) 
  = (Real.pi / 3) + (Real.log 2 / 36) :=
by
  -- Proof to be provided
  sorry

end definite_integral_value_l41_41554


namespace number_of_measures_of_C_l41_41933

theorem number_of_measures_of_C (C D : ℕ) (h1 : C + D = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ C = k * D) : 
  ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_measures_of_C_l41_41933


namespace quadratic_inequality_solution_minimum_value_expression_l41_41201

theorem quadratic_inequality_solution (a : ℝ) : (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) → a > 3 :=
sorry

theorem minimum_value_expression (a : ℝ) : (a > 3) → a + 9 / (a - 1) ≥ 7 ∧ (a + 9 / (a - 1) = 7 ↔ a = 4) :=
sorry

end quadratic_inequality_solution_minimum_value_expression_l41_41201


namespace fraction_of_number_l41_41943

-- Given definitions based on the problem conditions
def fraction : ℚ := 3 / 4
def number : ℕ := 40

-- Theorem statement to be proved
theorem fraction_of_number : fraction * number = 30 :=
by
  sorry -- This indicates that the proof is not yet provided

end fraction_of_number_l41_41943


namespace arithmetic_sequence_x_value_l41_41877

theorem arithmetic_sequence_x_value (x : ℝ) (a2 a1 d : ℝ)
  (h1 : a1 = 1 / 3)
  (h2 : a2 = x - 2)
  (h3 : d = 4 * x + 1 - a2)
  (h2_eq_d_a1 : a2 - a1 = d) : x = - (8 / 3) :=
by
  -- Proof yet to be completed
  sorry

end arithmetic_sequence_x_value_l41_41877


namespace min_value_b1_b2_l41_41844

noncomputable def seq (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b n + 2017) / (1 + b (n + 1))

theorem min_value_b1_b2 (b : ℕ → ℕ)
  (h_pos : ∀ n, b n > 0)
  (h_seq : seq b) :
  b 1 + b 2 = 2018 := sorry

end min_value_b1_b2_l41_41844


namespace total_height_of_buildings_l41_41939

noncomputable def tallest_building := 100
noncomputable def second_tallest_building := tallest_building / 2
noncomputable def third_tallest_building := second_tallest_building / 2
noncomputable def fourth_tallest_building := third_tallest_building / 5

theorem total_height_of_buildings : 
  (tallest_building + second_tallest_building + third_tallest_building + fourth_tallest_building) = 180 := by
  sorry

end total_height_of_buildings_l41_41939


namespace remainder_of_sum_l41_41472

theorem remainder_of_sum (h1 : 9375 % 5 = 0) (h2 : 9376 % 5 = 1) (h3 : 9377 % 5 = 2) (h4 : 9378 % 5 = 3) :
  (9375 + 9376 + 9377 + 9378) % 5 = 1 :=
by
  sorry

end remainder_of_sum_l41_41472


namespace travis_flight_cost_l41_41802

theorem travis_flight_cost 
  (cost_leg1 : ℕ := 1500) 
  (cost_leg2 : ℕ := 1000) 
  (discount_leg1 : ℕ := 25) 
  (discount_leg2 : ℕ := 35) : 
  cost_leg1 - (discount_leg1 * cost_leg1 / 100) + cost_leg2 - (discount_leg2 * cost_leg2 / 100) = 1775 :=
by
  sorry

end travis_flight_cost_l41_41802


namespace find_valid_pairs_l41_41586

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def valid_pair (p q : ℕ) : Prop :=
  p < 2005 ∧ q < 2005 ∧ is_prime p ∧ is_prime q ∧ q ∣ p^2 + 8 ∧ p ∣ q^2 + 8

theorem find_valid_pairs :
  ∀ p q, valid_pair p q → (p, q) = (2, 2) ∨ (p, q) = (881, 89) ∨ (p, q) = (89, 881) :=
sorry

end find_valid_pairs_l41_41586


namespace sum_of_odd_divisors_of_90_l41_41967

def is_odd (n : ℕ) : Prop := n % 2 = 1

def divisors (n : ℕ) : List ℕ := List.filter (λ d => n % d = 0) (List.range (n + 1))

def odd_divisors (n : ℕ) : List ℕ := List.filter is_odd (divisors n)

def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem sum_of_odd_divisors_of_90 : sum_list (odd_divisors 90) = 78 := 
by
  sorry

end sum_of_odd_divisors_of_90_l41_41967


namespace part1_part2_l41_41736

noncomputable def f (x a : ℝ) : ℝ := |x - 1| - 2 * |x + a|
noncomputable def g (x b : ℝ) : ℝ := 0.5 * x + b

theorem part1 (a : ℝ) (h : a = 1/2) : 
  { x : ℝ | f x a ≤ 0 } = { x : ℝ | x ≤ -2 ∨ x ≥ 0 } :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ -1) (h2 : ∀ x, g x b ≥ f x a) : 
  2 * b - 3 * a > 2 :=
sorry

end part1_part2_l41_41736


namespace find_number_l41_41565

-- Define the condition
def condition : Prop := ∃ x : ℝ, x / 0.02 = 50

-- State the theorem to prove
theorem find_number (x : ℝ) (h : x / 0.02 = 50) : x = 1 :=
sorry

end find_number_l41_41565


namespace average_rainfall_correct_l41_41888

-- Definitions based on given conditions
def total_rainfall : ℚ := 420 -- inches
def days_in_august : ℕ := 31
def hours_in_a_day : ℕ := 24

-- Defining total hours in August
def total_hours_in_august : ℕ := days_in_august * hours_in_a_day

-- The average rainfall in inches per hour
def average_rainfall_per_hour : ℚ := total_rainfall / total_hours_in_august

-- The statement to prove
theorem average_rainfall_correct :
  average_rainfall_per_hour = 420 / 744 :=
by
  sorry

end average_rainfall_correct_l41_41888


namespace A_is_7056_l41_41065

-- Define the variables and conditions
def D : ℕ := 4 * 3
def E : ℕ := 7 * 3
def B : ℕ := 4 * D
def C : ℕ := 7 * E
def A : ℕ := B * C

-- Prove that A = 7056 given the conditions
theorem A_is_7056 : A = 7056 := by
  -- We will skip the proof steps with 'sorry'
  sorry

end A_is_7056_l41_41065


namespace sought_line_eq_l41_41932

-- Definitions used in the conditions
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def line_perpendicular (x y : ℝ) : Prop := x + y = 0
def center_of_circle : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem sought_line_eq (x y : ℝ) :
  (circle_eq x y ∧ line_perpendicular x y ∧ (x, y) = center_of_circle) →
  (x + y + 1 = 0) :=
by
  sorry

end sought_line_eq_l41_41932


namespace integrality_condition_l41_41728

noncomputable def binom (n k : ℕ) : ℕ := 
  n.choose k

theorem integrality_condition (n k : ℕ) (h : 1 ≤ k) (h1 : k < n) (h2 : (k + 1) ∣ (n^2 - 3*k^2 - 2)) : 
  ∃ m : ℕ, m = (n^2 - 3*k^2 - 2) / (k + 1) ∧ (m * binom n k) % 1 = 0 :=
sorry

end integrality_condition_l41_41728


namespace ratio_of_ages_l41_41974

theorem ratio_of_ages (age_saras age_kul : ℕ) (h_saras : age_saras = 33) (h_kul : age_kul = 22) : 
  age_saras / Nat.gcd age_saras age_kul = 3 ∧ age_kul / Nat.gcd age_saras age_kul = 2 :=
by
  sorry

end ratio_of_ages_l41_41974


namespace sculpture_height_is_34_inches_l41_41707

-- Define the height of the base in inches
def height_of_base_in_inches : ℕ := 2

-- Define the total height in feet
def total_height_in_feet : ℕ := 3

-- Convert feet to inches (1 foot = 12 inches)
def total_height_in_inches (feet : ℕ) : ℕ := feet * 12

-- The height of the sculpture, given the base and total height
def height_of_sculpture (total_height base_height : ℕ) : ℕ := total_height - base_height

-- State the theorem that the height of the sculpture is 34 inches
theorem sculpture_height_is_34_inches :
  height_of_sculpture (total_height_in_inches total_height_in_feet) height_of_base_in_inches = 34 := by
  sorry

end sculpture_height_is_34_inches_l41_41707


namespace find_a99_l41_41490

def seq (a : ℕ → ℕ) :=
  a 1 = 2 ∧ ∀ n ≥ 2, a n - a (n-1) = n + 1

theorem find_a99 (a : ℕ → ℕ) (h : seq a) : a 99 = 5049 :=
by
  have : seq a := h
  sorry

end find_a99_l41_41490


namespace work_rate_c_l41_41550

theorem work_rate_c (A B C : ℝ) (h1 : A + B = 1 / 4) (h2 : B + C = 1 / 6) (h3 : C + A = 1 / 3) :
    1 / C = 8 :=
by
  sorry

end work_rate_c_l41_41550


namespace eval_expression_l41_41458

theorem eval_expression : (-1)^45 + 2^(3^2 + 5^2 - 4^2) = 262143 := by
  sorry

end eval_expression_l41_41458


namespace find_m_parallel_l41_41366

noncomputable def is_parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  -(A1 / B1) = -(A2 / B2)

theorem find_m_parallel : ∃ m : ℝ, is_parallel (m-1) 3 m 1 (m+1) 2 ∧ m = -2 :=
by
  unfold is_parallel
  exists (-2 : ℝ)
  sorry

end find_m_parallel_l41_41366


namespace chosen_number_is_121_l41_41141

theorem chosen_number_is_121 (x : ℤ) (h : 2 * x - 140 = 102) : x = 121 := 
by 
  sorry

end chosen_number_is_121_l41_41141


namespace number_of_sheep_l41_41278

theorem number_of_sheep (S H : ℕ)
  (h1 : S / H = 4 / 7)
  (h2 : H * 230 = 12880) :
  S = 32 :=
by
  sorry

end number_of_sheep_l41_41278


namespace sum_of_digits_in_7_pow_1500_l41_41272

-- Define the problem and conditions
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10
def sum_of_digits (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem sum_of_digits_in_7_pow_1500 :
  sum_of_digits (7^1500) = 2 :=
by
  sorry

end sum_of_digits_in_7_pow_1500_l41_41272


namespace least_integer_condition_l41_41542

theorem least_integer_condition : ∃ x : ℤ, (x^2 = 2 * x + 72) ∧ (x = -6) :=
sorry

end least_integer_condition_l41_41542


namespace juan_speed_l41_41639

theorem juan_speed (J : ℝ) :
  (∀ (time : ℝ) (distance : ℝ) (peter_speed : ℝ),
    time = 1.5 →
    distance = 19.5 →
    peter_speed = 5 →
    distance = J * time + peter_speed * time) →
  J = 8 :=
by
  intro h
  sorry

end juan_speed_l41_41639


namespace base12_addition_example_l41_41990

theorem base12_addition_example : 
  (5 * 12^2 + 2 * 12^1 + 8 * 12^0) + (2 * 12^2 + 7 * 12^1 + 3 * 12^0) = (7 * 12^2 + 9 * 12^1 + 11 * 12^0) :=
by sorry

end base12_addition_example_l41_41990


namespace probability_not_finish_l41_41114

theorem probability_not_finish (p : ℝ) (h : p = 5 / 8) : 1 - p = 3 / 8 := 
by 
  rw [h]
  norm_num

end probability_not_finish_l41_41114


namespace required_total_money_l41_41286

def bundle_count := 100
def number_of_bundles := 10
def bill_5_value := 5
def bill_10_value := 10
def bill_20_value := 20

-- Sum up the total money required to fill the machine
theorem required_total_money : 
  (bundle_count * bill_5_value * number_of_bundles) + 
  (bundle_count * bill_10_value * number_of_bundles) + 
  (bundle_count * bill_20_value * number_of_bundles) = 35000 := 
by 
  sorry

end required_total_money_l41_41286


namespace necessary_but_not_sufficient_l41_41497

variables (A B : Prop)

theorem necessary_but_not_sufficient 
  (h1 : ¬ B → ¬ A)  -- Condition: ¬ B → ¬ A is true
  (h2 : ¬ (¬ A → ¬ B))  -- Condition: ¬ A → ¬ B is false
  : (A → B) ∧ ¬ (B → A) := -- Conclusion: A → B and not (B → A)
by
  -- Proof is not required, so we place sorry
  sorry

end necessary_but_not_sufficient_l41_41497


namespace solve_quadratic_eq_l41_41247

theorem solve_quadratic_eq (x : ℝ) : (x^2 - 2*x + 1 = 9) → (x = 4 ∨ x = -2) :=
by
  intro h
  sorry

end solve_quadratic_eq_l41_41247


namespace percentage_increase_is_correct_l41_41682

-- Define the original and new weekly earnings
def original_earnings : ℕ := 60
def new_earnings : ℕ := 90

-- Define the percentage increase calculation
def percentage_increase (original new : ℕ) : Rat := ((new - original) / original: Rat) * 100

-- State the theorem that the percentage increase is 50%
theorem percentage_increase_is_correct : percentage_increase original_earnings new_earnings = 50 := 
sorry

end percentage_increase_is_correct_l41_41682


namespace inverse_function_evaluation_l41_41656

theorem inverse_function_evaluation :
  ∀ (f : ℕ → ℕ) (f_inv : ℕ → ℕ),
    (∀ y, f_inv (f y) = y) ∧ (∀ x, f (f_inv x) = x) →
    f 4 = 7 →
    f 6 = 3 →
    f 3 = 6 →
    f_inv (f_inv 6 + f_inv 7) = 4 :=
by
  intros f f_inv hf hf1 hf2 hf3
  sorry

end inverse_function_evaluation_l41_41656


namespace find_c_l41_41042

theorem find_c (a b c : ℝ) (h_line : 4 * a - 3 * b + c = 0) 
  (h_min : (a - 1)^2 + (b - 1)^2 = 4) : c = 9 ∨ c = -11 := 
    sorry

end find_c_l41_41042


namespace combination_8_5_l41_41003

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def combination (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem combination_8_5 : combination 8 5 = 56 := by
  sorry

end combination_8_5_l41_41003


namespace distance_from_A_to_C_l41_41525

theorem distance_from_A_to_C (x y : ℕ) (d : ℚ)
  (h1 : d = x / 3) 
  (h2 : 13 + (d * 15) / (y - 13) = 2 * x)
  (h3 : y = 2 * x + 13) 
  : x + y = 26 := 
  sorry

end distance_from_A_to_C_l41_41525


namespace square_root_condition_l41_41205

theorem square_root_condition (x : ℝ) : (6 + x ≥ 0) ↔ (x ≥ -6) :=
by sorry

end square_root_condition_l41_41205


namespace initial_num_files_l41_41025

-- Define the conditions: number of files organized in the morning, files to organize in the afternoon, and missing files.
def num_files_organized_in_morning (X : ℕ) : ℕ := X / 2
def num_files_to_organize_in_afternoon : ℕ := 15
def num_files_missing : ℕ := 15

-- Theorem to prove the initial number of files is 60.
theorem initial_num_files (X : ℕ) 
  (h1 : num_files_organized_in_morning X = X / 2)
  (h2 : num_files_to_organize_in_afternoon = 15)
  (h3 : num_files_missing = 15) :
  X = 60 :=
by
  sorry

end initial_num_files_l41_41025


namespace erasers_difference_l41_41997

-- Definitions for the conditions in the problem
def andrea_erasers : ℕ := 4
def anya_erasers : ℕ := 4 * andrea_erasers

-- Theorem statement to prove the final answer
theorem erasers_difference : anya_erasers - andrea_erasers = 12 :=
by
  -- Proof placeholder
  sorry

end erasers_difference_l41_41997


namespace lcm_12_35_l41_41171

theorem lcm_12_35 : Nat.lcm 12 35 = 420 :=
by
  sorry

end lcm_12_35_l41_41171


namespace fraction_division_l41_41948

-- Definition of fractions involved
def frac1 : ℚ := 4 / 9
def frac2 : ℚ := 5 / 8

-- Statement of the proof problem
theorem fraction_division :
  (frac1 / frac2) = 32 / 45 :=
by {
  sorry
}

end fraction_division_l41_41948


namespace velma_more_than_veronica_l41_41133

-- Defining the distances each flashlight can be seen
def veronica_distance : ℕ := 1000
def freddie_distance : ℕ := 3 * veronica_distance
def velma_distance : ℕ := 5 * freddie_distance - 2000

-- The proof problem: Prove that Velma's flashlight can be seen 12000 feet farther than Veronica's flashlight.
theorem velma_more_than_veronica : velma_distance - veronica_distance = 12000 := by
  sorry

end velma_more_than_veronica_l41_41133


namespace determine_base_l41_41626

theorem determine_base (b : ℕ) (h : (3 * b + 1)^2 = b^3 + 2 * b + 1) : b = 10 :=
by
  sorry

end determine_base_l41_41626


namespace trip_time_is_correct_l41_41757

noncomputable def total_trip_time : ℝ :=
  let wrong_direction_time := 100 / 60
  let return_time := 100 / 45
  let detour_time := 30 / 45
  let normal_trip_time := 300 / 60
  let stop_time := 2 * (15 / 60)
  wrong_direction_time + return_time + detour_time + normal_trip_time + stop_time

theorem trip_time_is_correct : total_trip_time = 10.06 :=
  by
    -- Proof steps are omitted
    sorry

end trip_time_is_correct_l41_41757


namespace Penelope_Candies_l41_41920

variable (M : ℕ) (S : ℕ)
variable (h1 : 5 * S = 3 * M)
variable (h2 : M = 25)

theorem Penelope_Candies : S = 15 := by
  sorry

end Penelope_Candies_l41_41920


namespace quadratic_roots_range_l41_41599

theorem quadratic_roots_range (a : ℝ) :
  (a-1) * x^2 - 2*x + 1 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a-1) * x1^2 - 2*x1 + 1 = 0 ∧ (a-1) * x2^2 - 2*x2 + 1 = 0) → (a < 2 ∧ a ≠ 1) :=
sorry

end quadratic_roots_range_l41_41599


namespace room_width_l41_41537

theorem room_width (length : ℝ) (cost : ℝ) (rate : ℝ) (h_length : length = 5.5)
                    (h_cost : cost = 16500) (h_rate : rate = 800) : 
                    (cost / rate / length = 3.75) :=
by 
  sorry

end room_width_l41_41537


namespace solution_set_correct_l41_41312

open Real

def differentiable (f : ℝ → ℝ) : Prop := ∀ x, ∃ f' : ℝ, has_deriv_at f f' x

noncomputable def problem_statement (f : ℝ → ℝ) (hf_diff : differentiable f) (hf_ineq : ∀ x, f x > f' x)
  (hf_zero : f 0 = 1) : Set ℝ := {x : ℝ | (f x / exp x) < 1}

theorem solution_set_correct {f : ℝ → ℝ} (hf_diff : differentiable f) (hf_ineq : ∀ x, f x > f' x)
  (hf_zero : f 0 = 1) : problem_statement f hf_diff hf_ineq hf_zero = Ioi 0 :=
begin
  sorry
end

end solution_set_correct_l41_41312


namespace basketball_students_l41_41209

variable (C B_inter_C B_union_C B : ℕ)

theorem basketball_students (hC : C = 5) (hB_inter_C : B_inter_C = 3) (hB_union_C : B_union_C = 9) (hInclusionExclusion : B_union_C = B + C - B_inter_C) : B = 7 := by
  sorry

end basketball_students_l41_41209


namespace negation_of_p_l41_41738

theorem negation_of_p :
  (∃ x : ℝ, x < 0 ∧ x + (1 / x) > -2) ↔ ¬ (∀ x : ℝ, x < 0 → x + (1 / x) ≤ -2) :=
by {
  sorry
}

end negation_of_p_l41_41738


namespace cauchy_schwarz_inequality_l41_41651

theorem cauchy_schwarz_inequality (a b x y : ℝ) :
  (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 :=
by
  sorry

end cauchy_schwarz_inequality_l41_41651


namespace total_cost_of_color_drawing_l41_41638

def cost_bwch_drawing : ℕ := 160
def bwch_to_color_cost_multiplier : ℝ := 1.5

theorem total_cost_of_color_drawing 
  (cost_bwch : ℕ)
  (bwch_to_color_mult : ℝ)
  (h₁ : cost_bwch = 160)
  (h₂ : bwch_to_color_mult = 1.5) :
  cost_bwch * bwch_to_color_mult = 240 := 
  by
    sorry

end total_cost_of_color_drawing_l41_41638


namespace knights_adjacent_probability_sum_l41_41426

open Nat

theorem knights_adjacent_probability_sum : 
  let total_knights := 30
  let chosen_knights := 4
  let total_ways := Nat.choose total_knights chosen_knights
  let no_adjacent_count := 30 * 27 * 26 * 25
  let P_no_adjacent := no_adjacent_count / total_ways
  let P_adjacent := 1 - P_no_adjacent 
  let P_adjacent_numerator := 53
  let P_adjacent_denominator := 183
  P_adjacent = P_adjacent_numerator / P_adjacent_denominator →
  P_adjacent_numerator + P_adjacent_denominator = 236 := 
by
  -- Definitions and given conditions
  let total_knights := 30
  let chosen_knights := 4
  -- Total ways to choose 4 out of 30 knights
  let total_ways := Nat.choose total_knights chosen_knights
  -- Ways to place knights such that none are adjacent
  let no_adjacent_count := 30 * 27 * 26 * 25
  -- Calculate the probability of no adjacent knights
  let P_no_adjacent := (no_adjacent_count : ℚ) / total_ways
  -- Calculate the probability of at least one pair of adjacent knights
  let P_adjacent := 1 - P_no_adjacent
  -- Given the final fraction
  let P_adjacent_numerator := 53
  let P_adjacent_denominator := 183
  -- Assert the final condition
  have P_eq : P_adjacent = P_adjacent_numerator / P_adjacent_denominator := sorry
  show P_adjacent_numerator + P_adjacent_denominator = 236 from by
    rw [←P_eq]
    exact rfl

end knights_adjacent_probability_sum_l41_41426


namespace _l41_41575

noncomputable def urn_marble_theorem (r w b g y : Nat) : Prop :=
  let n := r + w + b + g + y
  ∃ k : Nat, 
  (k * r * (r-1) * (r-2) * (r-3) * (r-4) / 120 = w * r * (r-1) * (r-2) * (r-3) / 24)
  ∧ (w * r * (r-1) * (r-2) * (r-3) / 24 = w * b * r * (r-1) * (r-2) / 6)
  ∧ (w * b * r * (r-1) * (r-2) / 6 = w * b * g * r * (r-1) / 2)
  ∧ (w * b * g * r * (r-1) / 2 = w * b * g * r * y)
  ∧ n = 55

example : ∃ (r w b g y : Nat), urn_marble_theorem r w b g y := sorry

end _l41_41575


namespace chocolates_difference_l41_41552

-- Conditions
def Robert_chocolates : Nat := 13
def Nickel_chocolates : Nat := 4

-- Statement
theorem chocolates_difference : (Robert_chocolates - Nickel_chocolates) = 9 := by
  sorry

end chocolates_difference_l41_41552


namespace problem_statement_l41_41334

theorem problem_statement (a : Fin 17 → ℕ)
  (h : ∀ i : Fin 17, a i ^ a (i + 1) = a (i + 1) ^ a (i + 2)): 
  a 0 = a 1 :=
sorry

end problem_statement_l41_41334


namespace value_of_f_l41_41335

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Conditions
axiom f_has_derivative : ∀ x, deriv f x = f' x
axiom f_equation : ∀ x, f x = 3 * x^2 + 2 * x * (f' 1)

-- Proof goal
theorem value_of_f'_at_3 : f' 3 = 6 := by
  sorry

end value_of_f_l41_41335


namespace min_third_side_of_right_triangle_l41_41365

theorem min_third_side_of_right_triangle (a b : ℕ) (h : a = 7 ∧ b = 24) : 
  ∃ (c : ℝ), c = Real.sqrt (576 - 49) :=
by
  sorry

end min_third_side_of_right_triangle_l41_41365


namespace distance_AO_min_distance_BM_l41_41376

open Real

-- Definition of rectangular distance
def rectangular_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

-- Point A and O
def A : ℝ × ℝ := (-1, 3)
def O : ℝ × ℝ := (0, 0)

-- Point B
def B : ℝ × ℝ := (1, 0)

-- Line "x - y + 2 = 0"
def on_line (M : ℝ × ℝ) : Prop :=
  M.1 - M.2 + 2 = 0

-- Proof statement 1: distance from A to O is 4
theorem distance_AO : rectangular_distance A O = 4 := 
sorry

-- Proof statement 2: minimum distance from B to any point on the line is 3
theorem min_distance_BM (M : ℝ × ℝ) (h : on_line M) : rectangular_distance B M = 3 := 
sorry

end distance_AO_min_distance_BM_l41_41376


namespace average_growth_rate_l41_41502

theorem average_growth_rate (x : ℝ) :
  (7200 * (1 + x)^2 = 8712) → x = 0.10 :=
by
  sorry

end average_growth_rate_l41_41502


namespace divisibility_by_1897_l41_41403

theorem divisibility_by_1897 (n : ℕ) : 1897 ∣ (2903 ^ n - 803 ^ n - 464 ^ n + 261 ^ n) :=
sorry

end divisibility_by_1897_l41_41403


namespace tenth_permutation_is_2561_l41_41934

-- Define the digits
def digits : List ℕ := [1, 2, 5, 6]

-- Define the permutations of those digits
def perms := digits.permutations.map (λ l, l.asString.toNat)

-- Define the 10th integer in the sorted list of those permutations
noncomputable def tenth_integer : ℕ := (perms.sort (≤)).nthLe 9 (by norm_num [List.length_permutations, perms])

-- The statement we want to prove
theorem tenth_permutation_is_2561 : tenth_integer = 2561 :=
sorry

end tenth_permutation_is_2561_l41_41934


namespace fatima_donates_75_sq_inches_l41_41716

/-- Fatima starts with 100 square inches of cloth and cuts it in half twice.
    The total amount of cloth she donates should be 75 square inches. -/
theorem fatima_donates_75_sq_inches:
  ∀ (cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second: ℕ),
  cloth_initial = 100 → 
  cloth_after_first_cut = cloth_initial / 2 →
  cloth_donated_first = cloth_initial / 2 →
  cloth_after_second_cut = cloth_after_first_cut / 2 →
  cloth_donated_second = cloth_after_first_cut / 2 →
  cloth_donated_first + cloth_donated_second = 75 := 
by
  intros cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second
  intros h_initial h_after_first h_donated_first h_after_second h_donated_second
  sorry

end fatima_donates_75_sq_inches_l41_41716


namespace probability_multiple_of_105_l41_41370

open Set

def S : Set ℕ := {5, 15, 21, 35, 45, 49, 63}

theorem probability_multiple_of_105 : 
  (let pairs := {x | ∃ (a ∈ S) (b ∈ S), a ≠ b ∧ x = (a, b)}.toFinset 
   in (pairs.filter (λ x => 105 ∣ (x.1 * x.2))).card) / ({x | ∃ (a ∈ S) (b ∈ S), a ≠ b}.toFinset.card : ℚ) 
     = 4 / 7 :=
by
  sorry

end probability_multiple_of_105_l41_41370


namespace inequality_proof_l41_41911

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom abc_eq_one : a * b * c = 1

theorem inequality_proof :
  (1 + a * b) / (1 + a) + (1 + b * c) / (1 + b) + (1 + c * a) / (1 + c) ≥ 3 :=
by
  sorry

end inequality_proof_l41_41911


namespace least_integer_square_l41_41545

theorem least_integer_square (x : ℤ) : x^2 = 2 * x + 72 → x = -6 := 
by
  intro h
  sorry

end least_integer_square_l41_41545


namespace first_day_of_month_is_thursday_l41_41249

theorem first_day_of_month_is_thursday :
  (27 - 7 - 7 - 7 + 1) % 7 = 4 :=
by
  sorry

end first_day_of_month_is_thursday_l41_41249


namespace smallest_N_exists_l41_41294

theorem smallest_N_exists : ∃ N : ℕ, 
  (N % 9 = 8) ∧
  (N % 8 = 7) ∧
  (N % 7 = 6) ∧
  (N % 6 = 5) ∧
  (N % 5 = 4) ∧
  (N % 4 = 3) ∧
  (N % 3 = 2) ∧
  (N % 2 = 1) ∧
  N = 503 :=
by {
  sorry
}

end smallest_N_exists_l41_41294


namespace sum_of_values_l41_41786

def r (x : ℝ) : ℝ := abs (x + 1) - 3
def s (x : ℝ) : ℝ := -(abs (x + 2))

theorem sum_of_values :
  (s (r (-5)) + s (r (-4)) + s (r (-3)) + s (r (-2)) + s (r (-1)) + s (r (0)) + s (r (1)) + s (r (2)) + s (r (3))) = -37 :=
by {
  sorry
}

end sum_of_values_l41_41786


namespace speed_of_water_l41_41296

theorem speed_of_water (v : ℝ) :
  (∀ (distance time : ℝ), distance = 16 ∧ time = 8 → distance = (4 - v) * time) → 
  v = 2 :=
by
  intro h
  have h1 : 16 = (4 - v) * 8 := h 16 8 (by simp)
  sorry

end speed_of_water_l41_41296


namespace kelly_sony_games_solution_l41_41902

def kelly_sony_games_left (n g : Nat) : Nat :=
  n - g

theorem kelly_sony_games_solution (initial : Nat) (given_away : Nat) 
  (h_initial : initial = 132)
  (h_given_away : given_away = 101) :
  kelly_sony_games_left initial given_away = 31 :=
by
  rw [h_initial, h_given_away]
  unfold kelly_sony_games_left
  norm_num

end kelly_sony_games_solution_l41_41902


namespace ratio_alcohol_to_water_l41_41059

theorem ratio_alcohol_to_water (vol_alcohol vol_water : ℚ) 
  (h_alcohol : vol_alcohol = 2/7) 
  (h_water : vol_water = 3/7) : 
  vol_alcohol / vol_water = 2 / 3 := 
by
  sorry

end ratio_alcohol_to_water_l41_41059


namespace rectangular_field_length_l41_41297

theorem rectangular_field_length (w : ℝ) (h₁ : w * (w + 10) = 171) : w + 10 = 19 := 
by
  sorry

end rectangular_field_length_l41_41297


namespace find_X_in_rectangle_diagram_l41_41751

theorem find_X_in_rectangle_diagram :
  ∀ (X : ℝ),
  (1 + 1 + 1 + 2 + X = 1 + 2 + 1 + 6) → X = 5 :=
by
  intros X h
  sorry

end find_X_in_rectangle_diagram_l41_41751


namespace g_at_3_l41_41533

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_3 : (∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x ^ 2 + 1) → 
  g 3 = 130 / 21 := 
by 
  sorry

end g_at_3_l41_41533


namespace inequality_x2_y4_z6_l41_41096

variable (x y z : ℝ)

theorem inequality_x2_y4_z6
    (hx : 0 < x)
    (hy : 0 < y)
    (hz : 0 < z) :
    x^2 + y^4 + z^6 ≥ x * y^2 + y^2 * z^3 + x * z^3 :=
by
  sorry

end inequality_x2_y4_z6_l41_41096


namespace cone_lateral_surface_area_l41_41866

-- Definitions and conditions
def radius (r : ℝ) := r = 3
def slant_height (l : ℝ) := l = 5
def lateral_surface_area (A : ℝ) (C : ℝ) (l : ℝ) := A = 0.5 * C * l
def circumference (C : ℝ) (r : ℝ) := C = 2 * Real.pi * r

-- Proof (statement only)
theorem cone_lateral_surface_area :
  ∀ (r l C A : ℝ), 
    radius r → 
    slant_height l → 
    circumference C r → 
    lateral_surface_area A C l → 
    A = 15 * Real.pi := 
by intros; sorry

end cone_lateral_surface_area_l41_41866


namespace sum_odd_divisors_90_eq_78_l41_41958

-- Noncomputable is used because we might need arithmetic operations that are not computable in Lean
noncomputable def sum_of_odd_divisors_of_90 : Nat :=
  1 + 3 + 5 + 9 + 15 + 45

theorem sum_odd_divisors_90_eq_78 : sum_of_odd_divisors_of_90 = 78 := 
  by 
    -- The sum is directly given; we don't need to compute it here
    sorry

end sum_odd_divisors_90_eq_78_l41_41958


namespace polynomial_bound_l41_41670

theorem polynomial_bound (a b c d : ℝ) 
  (h1 : ∀ x : ℝ, |x| ≤ 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) : 
  |a| + |b| + |c| + |d| ≤ 7 := 
sorry

end polynomial_bound_l41_41670


namespace product_to_difference_l41_41645

def x := 88 * 1.25
def y := 150 * 0.60
def z := 60 * 1.15

def product := x * y * z
def difference := x - y

theorem product_to_difference :
  product ^ difference = 683100 ^ 20 := 
sorry

end product_to_difference_l41_41645


namespace total_apples_eaten_l41_41304

-- Define the variables based on the conditions
variable (tuesday_apples : ℕ)
variable (wednesday_apples : ℕ)
variable (thursday_apples : ℕ)
variable (total_apples : ℕ)

-- Define the conditions
def cond1 : Prop := tuesday_apples = 4
def cond2 : Prop := wednesday_apples = 2 * tuesday_apples
def cond3 : Prop := thursday_apples = tuesday_apples / 2

-- Define the total apples
def total : Prop := total_apples = tuesday_apples + wednesday_apples + thursday_apples

-- Prove the equivalence
theorem total_apples_eaten : 
  cond1 → cond2 → cond3 → total_apples = 14 :=
by 
  sorry

end total_apples_eaten_l41_41304


namespace least_possible_third_side_l41_41360

theorem least_possible_third_side (a b : ℝ) (ha : a = 7) (hb : b = 24) : ∃ c, c = 24 ∧ a^2 - c^2 = 527  :=
by
  use (√527)
  sorry

end least_possible_third_side_l41_41360


namespace anya_more_erasers_l41_41994

theorem anya_more_erasers (anya_erasers andrea_erasers : ℕ)
  (h1 : anya_erasers = 4 * andrea_erasers)
  (h2 : andrea_erasers = 4) :
  anya_erasers - andrea_erasers = 12 := by
  sorry

end anya_more_erasers_l41_41994


namespace A_in_second_quadrant_l41_41862

-- Define the coordinates of point A
def A_x : ℝ := -2
def A_y : ℝ := 3

-- Define the condition that point A lies in the second quadrant
def is_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- State the theorem
theorem A_in_second_quadrant : is_second_quadrant A_x A_y :=
by
  -- The proof will be provided here.
  sorry

end A_in_second_quadrant_l41_41862


namespace sufficient_not_necessary_condition_l41_41732

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ) (h_pos : x > 0) :
  (a = 4 → x + a / x ≥ 4) ∧ (∃ b : ℝ, b ≠ 4 ∧ ∃ x : ℝ, x > 0 ∧ x + b / x ≥ 4) :=
by
  sorry

end sufficient_not_necessary_condition_l41_41732


namespace tom_rope_stories_l41_41265

/-- Define the conditions given in the problem. --/
def story_length : ℝ := 10
def rope_length : ℝ := 20
def loss_percentage : ℝ := 0.25
def pieces_of_rope : ℕ := 4

/-- Theorem to prove the number of stories Tom can lower the rope down. --/
theorem tom_rope_stories (story_length rope_length loss_percentage : ℝ) (pieces_of_rope : ℕ) : 
    story_length = 10 → 
    rope_length = 20 →
    loss_percentage = 0.25 →
    pieces_of_rope = 4 →
    pieces_of_rope * rope_length * (1 - loss_percentage) / story_length = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end tom_rope_stories_l41_41265


namespace pairs_of_real_numbers_l41_41168

theorem pairs_of_real_numbers (a b : ℝ) (h : ∀ (n : ℕ), n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)) :
  a = 0 ∨ b = 0 ∨ a = b ∨ (∃ m n : ℤ, a = (m : ℝ) ∧ b = (n : ℝ)) :=
by
  sorry

end pairs_of_real_numbers_l41_41168


namespace fixed_point_exists_l41_41523

theorem fixed_point_exists : ∀ (m : ℝ), (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
by
  intro m
  have h : (m - 1) * (7 / 2) - (m + 3) * (5 / 2) - (m - 11) = 0 :=
    sorry
  exact h

end fixed_point_exists_l41_41523


namespace notebooks_difference_l41_41390

theorem notebooks_difference 
  (cost_mika : ℝ) (cost_leo : ℝ) (notebook_price : ℝ)
  (h_cost_mika : cost_mika = 2.40)
  (h_cost_leo : cost_leo = 3.20)
  (h_notebook_price : notebook_price > 0.10)
  (h_mika : ∃ m : ℕ, cost_mika = m * notebook_price)
  (h_leo : ∃ l : ℕ, cost_leo = l * notebook_price)
  : ∃ n : ℕ, (l - m = 4) :=
by
  sorry

end notebooks_difference_l41_41390


namespace debra_flips_coin_probability_l41_41163

noncomputable theory

def probability_three_heads_after_two_tails : ℚ :=
  1 / 128

theorem debra_flips_coin_probability :
  (debra_flips_coin_probability := 1 / 128)
  :
  debra_flips_coin_probability = probability_three_heads_after_two_tails :=
sorry

end debra_flips_coin_probability_l41_41163


namespace peter_runs_more_than_andrew_each_day_l41_41243

-- Define the constants based on the conditions
def miles_andrew : ℕ := 2
def total_days : ℕ := 5
def total_miles : ℕ := 35

-- Define a theorem to prove the number of miles Peter runs more than Andrew each day
theorem peter_runs_more_than_andrew_each_day : 
  ∃ x : ℕ, total_days * (miles_andrew + x) + total_days * miles_andrew = total_miles ∧ x = 3 :=
by
  sorry

end peter_runs_more_than_andrew_each_day_l41_41243


namespace tree_original_height_l41_41905

theorem tree_original_height (current_height_in: ℝ) (growth_percentage: ℝ)
  (h1: current_height_in = 180) (h2: growth_percentage = 0.50) :
  ∃ (original_height_ft: ℝ), original_height_ft = 10 :=
by
  have original_height_in := current_height_in / (1 + growth_percentage)
  have original_height_ft := original_height_in / 12
  use original_height_ft
  sorry

end tree_original_height_l41_41905


namespace unique_ellipse_through_points_with_perpendicular_axes_infinite_ellipses_when_points_coincide_l41_41425

-- Definitions of points and lines
structure Point (α : Type*) := (x : α) (y : α)
structure Line (α : Type*) := (a : α) (b : α) -- Represented as ax + by = 0

-- Given conditions
variables {α : Type*} [Field α]
variables (P Q : Point α)
variables (L1 L2 : Line α) -- L1 and L2 are perpendicular

-- Proof problem statement
theorem unique_ellipse_through_points_with_perpendicular_axes (P Q : Point α) (L1 L2 : Line α) (h_perp : L1.a * L2.b = - (L1.b * L2.a)) :
(P ≠ Q) → 
∃! (E : Set (Point α)), -- E represents the ellipse as a set of points
(∀ (p : Point α), p ∈ E → (p = P ∨ p = Q)) ∧ -- E passes through P and Q
(∀ (p : Point α), ∃ (u v : α), p.x = u ∨ p.y = v) := -- E has axes along L1 and L2
sorry

theorem infinite_ellipses_when_points_coincide (P : Point α) (L1 L2 : Line α) (h_perp : L1.a * L2.b = - (L1.b * L2.a)) :
∃ (E : Set (Point α)), -- E represents an ellipse
(∀ (p : Point α), p ∈ E → p = P) ∧ -- E passes through P
(∀ (p : Point α), ∃ (u v : α), p.x = u ∨ p.y = v) := -- E has axes along L1 and L2
sorry

end unique_ellipse_through_points_with_perpendicular_axes_infinite_ellipses_when_points_coincide_l41_41425


namespace max_robots_count_ways_to_place_robots_l41_41986

open Finset

-- Definitions and context setting
def palaceGraph (V E : ℕ) (H : V = 32 ∧ E = 40) := ∃ (G : Type*) [graph G] (V : finset G) (E : finset (G × G)),
  V.card = 32 ∧ E.card = 40 ∧ ∀ (v ∈ V), ∃ u, (u, v) ∈ E ∨ (v, u) ∈ E

-- Given conditions: 32 rooms, 40 corridors, robots move without meeting
def robotInRoom (G : Type*) [graph G] (V : finset G) (H : V.card = 32) := 

-- Maximum matching
def maxMatching (N : ℕ) : Prop :=
∀ (G : Type*) [graph G] (V : finset G) (E : finset (G × G))
  (H1 : V.card = 32) (H2 : E.card = 40),
  (∃ (M : finset (G × G)), M.card = N ∧ is_max_matching M)

-- Proof goal: maximum value of N is 16
theorem max_robots (V E : ℕ) (H : palaceGraph V E (by refl)) :
  maxMatching 16 :=
sorry

-- Number of ways to place 16 robots in 32 rooms considering they are indistinguishable
theorem count_ways_to_place_robots : 
  ∃ N, N = 16 ∧ N.choose 16 = binomial 32 16 :=
sorry

end max_robots_count_ways_to_place_robots_l41_41986


namespace sum_of_odd_divisors_l41_41965

theorem sum_of_odd_divisors (n : ℕ) (h : n = 90) : 
  (∑ d in (Finset.filter (λ x, x ∣ n ∧ x % 2 = 1) (Finset.range (n+1))), d) = 78 :=
by
  rw h
  sorry

end sum_of_odd_divisors_l41_41965


namespace find_a1_geometric_sequence_l41_41485

theorem find_a1_geometric_sequence (a₁ q : ℝ) (h1 : q ≠ 1) 
    (h2 : a₁ * (1 - q^3) / (1 - q) = 7)
    (h3 : a₁ * (1 - q^6) / (1 - q) = 63) :
    a₁ = 1 :=
by
  sorry

end find_a1_geometric_sequence_l41_41485


namespace negation_of_p_l41_41650

-- Define the proposition p
def p : Prop := ∃ x : ℝ, x + 2 ≤ 0

-- Define the negation of p
def not_p : Prop := ∀ x : ℝ, x + 2 > 0

-- State the theorem that the negation of p is not_p
theorem negation_of_p : ¬ p = not_p := by 
  sorry -- Proof not provided

end negation_of_p_l41_41650


namespace dice_digit_distribution_l41_41803

theorem dice_digit_distribution : ∃ n : ℕ, n = 10 ∧ 
  (∀ (d1 d2 : Finset ℕ), d1.card = 6 ∧ d2.card = 6 ∧
  (0 ∈ d1) ∧ (1 ∈ d1) ∧ (2 ∈ d1) ∧ 
  (0 ∈ d2) ∧ (1 ∈ d2) ∧ (2 ∈ d2) ∧
  ({3, 4, 5, 6, 7, 8} ⊆ (d1 ∪ d2)) ∧ 
  (∀ i, i ∈ d1 ∪ d2 → i ∈ (Finset.range 10))) := 
  sorry

end dice_digit_distribution_l41_41803


namespace first_statement_second_statement_difference_between_statements_l41_41315

variable (A B C : Prop)

-- First statement: (A ∨ B) → C
theorem first_statement : (A ∨ B) → C :=
sorry

-- Second statement: (A ∧ B) → C
theorem second_statement : (A ∧ B) → C :=
sorry

-- Proof that shows the difference between the two statements
theorem difference_between_statements :
  ((A ∨ B) → C) ↔ ¬((A ∧ B) → C) :=
sorry

end first_statement_second_statement_difference_between_statements_l41_41315


namespace jerrys_breakfast_calories_l41_41512

-- Define the constants based on the conditions
def pancakes : ℕ := 6
def calories_per_pancake : ℕ := 120
def strips_of_bacon : ℕ := 2
def calories_per_strip_of_bacon : ℕ := 100
def calories_in_cereal : ℕ := 200

-- Define the total calories for each category
def total_calories_from_pancakes : ℕ := pancakes * calories_per_pancake
def total_calories_from_bacon : ℕ := strips_of_bacon * calories_per_strip_of_bacon
def total_calories_from_cereal : ℕ := calories_in_cereal

-- Define the total calories in the breakfast
def total_breakfast_calories : ℕ := 
  total_calories_from_pancakes + total_calories_from_bacon + total_calories_from_cereal

-- The theorem we need to prove
theorem jerrys_breakfast_calories : total_breakfast_calories = 1120 := by sorry

end jerrys_breakfast_calories_l41_41512


namespace time_for_Q_to_finish_job_alone_l41_41139

theorem time_for_Q_to_finish_job_alone (T_Q : ℝ) 
  (h1 : 0 < T_Q)
  (rate_P : ℝ := 1 / 4) 
  (rate_Q : ℝ := 1 / T_Q)
  (combined_work_rate : ℝ := 3 * (rate_P + rate_Q))
  (remaining_work : ℝ := 0.1) -- 0.4 * rate_P
  (total_work_done : ℝ := 0.9) -- 1 - remaining_work
  (h2 : combined_work_rate = total_work_done) : T_Q = 20 :=
by sorry

end time_for_Q_to_finish_job_alone_l41_41139


namespace water_hyacinth_indicates_connection_l41_41375

-- Definitions based on the conditions
def universally_interconnected : Prop := 
  ∀ (a b : Type), a ≠ b → ∃ (c : Type), (a ≠ c) ∧ (b ≠ c)

def connections_diverse : Prop := 
  ∀ (a b : Type), a ≠ b → ∃ (f : a → b), ∀ (x y : a), x ≠ y → f x ≠ f y

def connections_created : Prop :=
  ∃ (a b : Type), a ≠ b ∧ (∀ (f : a → b), False)

def connections_humanized : Prop :=
  ∀ (a b : Type), a ≠ b → (∃ c : Type, a = c) ∧ (∃ d : Type, b = d)

-- Problem statement
theorem water_hyacinth_indicates_connection : 
  universally_interconnected ∧ connections_diverse :=
by
  sorry

end water_hyacinth_indicates_connection_l41_41375


namespace not_divisible_by_8_l41_41381

theorem not_divisible_by_8 : ¬ (456294604884 % 8 = 0) := 
by
  have h : 456294604884 % 1000 = 884 := sorry -- This step reflects the conclusion that the last three digits are 884.
  have h_div : ¬ (884 % 8 = 0) := sorry -- This reflects that 884 is not divisible by 8.
  sorry

end not_divisible_by_8_l41_41381


namespace isosceles_triangle_perimeter_l41_41546

theorem isosceles_triangle_perimeter (side1 side2 base : ℕ)
    (h1 : side1 = 12) (h2 : side2 = 12) (h3 : base = 17) : 
    side1 + side2 + base = 41 := by
  sorry

end isosceles_triangle_perimeter_l41_41546


namespace z_in_fourth_quadrant_l41_41633

def complex_quadrant (re im : ℤ) : String :=
  if re > 0 ∧ im > 0 then "First Quadrant"
  else if re < 0 ∧ im > 0 then "Second Quadrant"
  else if re < 0 ∧ im < 0 then "Third Quadrant"
  else if re > 0 ∧ im < 0 then "Fourth Quadrant"
  else "Axis"

theorem z_in_fourth_quadrant : complex_quadrant 2 (-3) = "Fourth Quadrant" :=
by
  sorry

end z_in_fourth_quadrant_l41_41633


namespace existence_of_f_and_g_l41_41238

noncomputable def Set_n (n : ℕ) : Set ℕ := { x | x ≥ 1 ∧ x ≤ n }

theorem existence_of_f_and_g (n : ℕ) (f g : ℕ → ℕ) :
  (∀ x ∈ Set_n n, (f (g x) = x ∨ g (f x) = x) ∧ ¬(f (g x) = x ∧ g (f x) = x)) ↔ Even n := sorry

end existence_of_f_and_g_l41_41238


namespace distance_from_E_to_B_is_1_5_l41_41035

open_locale classical
noncomputable theory

-- Define Points A, B, C1, D1, and E in 3D space
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def C1 : ℝ × ℝ × ℝ := (1, 1, 1)
def D1 : ℝ × ℝ × ℝ := (0, 1, 1)
def E : ℝ × ℝ × ℝ := (0.5, 1, 1)

-- Define Euclidean distance
def euclidean_dist (p q : ℝ × ℝ × ℝ) : ℝ :=
  (real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2))

-- Problem statement: Prove the distance from E to B is 1.5
theorem distance_from_E_to_B_is_1_5 : euclidean_dist E B = 1.5 :=
by {
  sorry
}

end distance_from_E_to_B_is_1_5_l41_41035


namespace not_always_possible_triangle_sides_l41_41758

theorem not_always_possible_triangle_sides (α β γ δ : ℝ) 
  (h1 : α + β + γ + δ = 360) 
  (h2 : α < 180) 
  (h3 : β < 180) 
  (h4 : γ < 180) 
  (h5 : δ < 180) : 
  ¬ (∀ (x y z : ℝ), (x = α ∨ x = β ∨ x = γ ∨ x = δ) ∧ (y = α ∨ y = β ∨ y = γ ∨ y = δ) ∧ (z = α ∨ z = β ∨ z = γ ∨ z = δ) ∧ (x ≠ y) ∧ (x ≠ z) ∧ (y ≠ z) → x + y > z ∧ x + z > y ∧ y + z > x)
:= sorry

end not_always_possible_triangle_sides_l41_41758


namespace equal_angles_point_p_l41_41708

noncomputable def ellipse := 
  { (x : ℝ) (y : ℝ) | (x^2 / 8 + y^2 / 4 = 1) }

def focus : ℝ × ℝ := (2, 0)

theorem equal_angles_point_p :
  ∃ (p : ℝ), p > 0 ∧
  (∀ A B : ℝ × ℝ, (A ∈ ellipse) → (B ∈ ellipse) → (A ≠ B) → A.1 * B.2 - A.2 * B.1 = 0 → 
    let P := (p, 0)
    in ∠A P focus = ∠B P focus) ↔ p = 2 := 
by
  sorry

end equal_angles_point_p_l41_41708


namespace gloves_selection_l41_41261

theorem gloves_selection (total_pairs : ℕ) (total_gloves : ℕ) (num_to_select : ℕ) 
    (total_ways : ℕ) (no_pair_ways : ℕ) : 
    total_pairs = 4 → 
    total_gloves = 8 → 
    num_to_select = 4 → 
    total_ways = (Nat.choose total_gloves num_to_select) → 
    no_pair_ways = 2^total_pairs → 
    (total_ways - no_pair_ways) = 54 :=
by
  intros
  sorry

end gloves_selection_l41_41261


namespace ratio_after_girls_leave_l41_41980

-- Define the initial conditions
def initial_conditions (B G : ℕ) : Prop :=
  B = G ∧ B + G = 32

-- Define the event of girls leaving
def girls_leave (G : ℕ) : ℕ :=
  G - 8

-- Define the final ratio of boys to girls
def final_ratio (B G : ℕ) : ℕ :=
  B / (girls_leave G)

-- Prove the final ratio is 2:1
theorem ratio_after_girls_leave (B G : ℕ) (h : initial_conditions B G) :
  final_ratio B G = 2 :=
by
  sorry

end ratio_after_girls_leave_l41_41980


namespace herd_compuation_l41_41569

theorem herd_compuation (a b c : ℕ) (total_animals total_payment : ℕ) 
  (H1 : total_animals = a + b + 10 * c) 
  (H2 : total_payment = 20 * a + 10 * b + 10 * c) 
  (H3 : total_animals = 100) 
  (H4 : total_payment = 200) :
  a = 1 ∧ b = 9 ∧ 10 * c = 90 :=
by
  sorry

end herd_compuation_l41_41569


namespace diagonal_length_of_regular_hexagon_l41_41589

theorem diagonal_length_of_regular_hexagon (
  side_length : ℝ
) (h_side_length : side_length = 12) : 
  ∃ DA, DA = 12 * Real.sqrt 3 :=
by 
  sorry

end diagonal_length_of_regular_hexagon_l41_41589


namespace binom_eight_five_l41_41017

theorem binom_eight_five :
  Nat.choose 8 5 = 56 :=
sorry

end binom_eight_five_l41_41017


namespace necessary_but_not_sufficient_range_m_l41_41863

namespace problem

variable (m x y : ℝ)

/-- Propositions for m -/
def P := (1 < m ∧ m < 4) 
def Q := (2 < m ∧ m < 3) ∨ (3 < m ∧  m < 4)

/-- Statements that P => Q is necessary but not sufficient -/
theorem necessary_but_not_sufficient (hP : 1 < m ∧ m < 4) : 
  ((m-1) * (m-4) < 0) ∧ (Q m) :=
by 
  sorry

theorem range_m (h1 : ¬ (P m ∧ Q m)) (h2 : P m ∨ Q m) : 
  1 < m ∧ m ≤ 2 ∨ m = 3 :=
by
  sorry

end problem

end necessary_but_not_sufficient_range_m_l41_41863


namespace expected_heads_l41_41756

def coin_flips : Nat := 64

def prob_heads (tosses : ℕ) : ℚ :=
  1 / 2^(tosses + 1)

def total_prob_heads : ℚ :=
  prob_heads 0 + prob_heads 1 + prob_heads 2 + prob_heads 3

theorem expected_heads : (coin_flips : ℚ) * total_prob_heads = 60 := by
  sorry

end expected_heads_l41_41756


namespace map_distance_representation_l41_41395

-- Define the conditions and the question as a Lean statement
theorem map_distance_representation :
  (∀ (length_cm : ℕ), (length_cm : ℕ) = 23 → (length_cm * 50 / 10 : ℕ) = 115) :=
by
  sorry

end map_distance_representation_l41_41395


namespace monthly_savings_correct_l41_41089

-- Define the gross salaries before any deductions
def ivan_salary_gross : ℝ := 55000
def vasilisa_salary_gross : ℝ := 45000
def vasilisa_mother_salary_gross : ℝ := 18000
def vasilisa_father_salary_gross : ℝ := 20000
def son_scholarship_state : ℝ := 3000
def son_scholarship_non_state_gross : ℝ := 15000

-- Tax rate definition
def tax_rate : ℝ := 0.13

-- Net income calculations using the tax rate
def net_income (gross_income : ℝ) : ℝ := gross_income * (1 - tax_rate)

def ivan_salary_net : ℝ := net_income ivan_salary_gross
def vasilisa_salary_net : ℝ := net_income vasilisa_salary_gross
def vasilisa_mother_salary_net : ℝ := net_income vasilisa_mother_salary_gross
def vasilisa_father_salary_net : ℝ := net_income vasilisa_father_salary_gross
def son_scholarship_non_state_net : ℝ := net_income son_scholarship_non_state_gross

-- Monthly expenses total
def monthly_expenses : ℝ := 40000 + 20000 + 5000 + 5000 + 2000 + 2000

-- Net incomes for different periods
def total_net_income_before_01_05_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + vasilisa_mother_salary_net + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_01_05_2018_to_31_08_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + son_scholarship_state

def total_net_income_from_01_09_2018 : ℝ :=
  ivan_salary_net + vasilisa_salary_net + 10000 + vasilisa_father_salary_net + (son_scholarship_state + son_scholarship_non_state_net)

-- Savings calculations for different periods
def monthly_savings_before_01_05_2018 : ℝ :=
  total_net_income_before_01_05_2018 - monthly_expenses

def monthly_savings_01_05_2018_to_31_08_2018 : ℝ :=
  total_net_income_01_05_2018_to_31_08_2018 - monthly_expenses

def monthly_savings_from_01_09_2018 : ℝ :=
  total_net_income_from_01_09_2018 - monthly_expenses

-- Theorem to be proved
theorem monthly_savings_correct :
  monthly_savings_before_01_05_2018 = 49060 ∧
  monthly_savings_01_05_2018_to_31_08_2018 = 43400 ∧
  monthly_savings_from_01_09_2018 = 56450 :=
by
  sorry

end monthly_savings_correct_l41_41089


namespace BC_length_l41_41220

theorem BC_length (A B C X : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] 
  (AB AC : ℕ) (hAB : AB = 86) (hAC : AC = 97)
  (BX CX : ℕ) (h_circle_intersect : ∃ X, Metric.ball A 86 ∩ {BC} = {B, X})
  (h_integer_lengths : BX + CX = BC) :
  BC = 61 := 
by
  sorry

end BC_length_l41_41220


namespace students_spring_outing_l41_41284

theorem students_spring_outing (n : ℕ) (h1 : n = 5) : 2^n = 32 :=
  by {
    sorry
  }

end students_spring_outing_l41_41284


namespace min_value_l41_41197

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 :=
sorry

end min_value_l41_41197


namespace compare_sums_l41_41144

open Classical

-- Define the necessary sequences and their properties
variable {α : Type*} [LinearOrderedField α]

-- Arithmetic Sequence {a_n}
noncomputable def arith_seq (a_1 d : α) : ℕ → α
| 0     => a_1
| (n+1) => (arith_seq a_1 d n) + d

-- Geometric Sequence {b_n}
noncomputable def geom_seq (b_1 q : α) : ℕ → α
| 0     => b_1
| (n+1) => (geom_seq b_1 q n) * q

-- Sum of the first n terms of an arithmetic sequence
noncomputable def arith_sum (a_1 d : α) (n : ℕ) : α :=
(n + 1) * (a_1 + arith_seq a_1 d n) / 2

-- Sum of the first n terms of a geometric sequence
noncomputable def geom_sum (b_1 q : α) (n : ℕ) : α :=
if q = 1 then (n + 1) * b_1
else b_1 * (1 - q^(n + 1)) / (1 - q)

theorem compare_sums
  (a_1 b_1 : α) (d q : α)
  (hd : d ≠ 0) (hq : q > 0) (hq1 : q ≠ 1)
  (h_eq1 : a_1 = b_1)
  (h_eq2 : arith_seq a_1 d 1011 = geom_seq b_1 q 1011) :
  arith_sum a_1 d 2022 < geom_sum b_1 q 2022 :=
sorry

end compare_sums_l41_41144


namespace domain_of_function_l41_41783

theorem domain_of_function : 
  { x : ℝ | (x + 1 ≥ 0) ∧ (2 - x > 0) } = set.Ico (-1:ℝ) 2 :=
by
  sorry

end domain_of_function_l41_41783


namespace henry_books_l41_41053

def books_after_donation (initial_count : ℕ) 
    (novels science cookbooks philosophy history self_help : ℕ) 
    (donation_percentages : (ℚ × ℚ × ℚ × ℚ × ℚ × ℚ))
    (new_acquisitions : (ℕ × ℕ × ℕ))
    (reject_percentage : ℚ) : ℕ :=
  let donated := (novels * donation_percentages.1 +.to_nat) +
                 (science * donation_percentages.2.to_rat.to_nat) +
                 (cookbooks * donation_percentages.3.to_rat.to_nat) +
                 (philosophy * donation_percentages.4.to_rat.to_nat) +
                 (history * donation_percentages.5.to_rat.to_nat) +
                 self_help -- assuming all self-help books are donated
  let recycled := donated * reject_percentage.to_rat.to_nat
  let remaining_books := initial_count - donated + new_acquisitions.1 + new_acquisitions.2 + new_acquisitions.3
  remaining_books

theorem henry_books (initial_count : ℕ)
    (novels science cookbooks philosophy history self_help : ℕ)
    (donation_percentages : (ℚ × ℚ × ℚ × ℚ × ℚ × ℚ))
    (new_acquisitions : (ℕ × ℕ × ℕ))
    (reject_percentage : ℚ) :
  initial_count = 250 →
  novels = 75 →
  science = 55 →
  cookbooks = 40 →
  philosophy = 35 →
  history = 25 →
  self_help = 20 →
  donation_percentages = (60/100, 75/100, 1/2, 30/100, 1 / 4, 1) →
  new_acquisitions = (6, 10, 8) →
  reject_percentage = 5/100 →
  books_after_donation initial_count novels science cookbooks philosophy history self_help donation_percentages new_acquisitions reject_percentage = 139 :=
by
  intros 
  sorry

end henry_books_l41_41053


namespace perimeter_of_rectangle_l41_41530

-- Define the conditions
def area (l w : ℝ) : Prop := l * w = 180
def length_three_times_width (l w : ℝ) : Prop := l = 3 * w

-- Define the problem
theorem perimeter_of_rectangle (l w : ℝ) (h₁ : area l w) (h₂ : length_three_times_width l w) : 
  2 * (l + w) = 16 * Real.sqrt 15 := 
sorry

end perimeter_of_rectangle_l41_41530


namespace map_distance_ratio_l41_41341

theorem map_distance_ratio (actual_distance_km : ℝ) (map_distance_cm : ℝ) (h_actual_distance : actual_distance_km = 5) (h_map_distance : map_distance_cm = 2) :
  map_distance_cm / (actual_distance_km * 100000) = 1 / 250000 :=
by
  -- Given the actual distance in kilometers and map distance in centimeters, prove the scale ratio
  -- skip the proof
  sorry

end map_distance_ratio_l41_41341


namespace geom_seq_sum_3000_l41_41260

noncomputable
def sum_geom_seq (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then a * n
  else a * (1 - r ^ n) / (1 - r)

theorem geom_seq_sum_3000 (a r : ℝ) (h1: sum_geom_seq a r 1000 = 300) (h2: sum_geom_seq a r 2000 = 570) :
  sum_geom_seq a r 3000 = 813 :=
sorry

end geom_seq_sum_3000_l41_41260


namespace vector_AB_to_vector_BA_l41_41408

theorem vector_AB_to_vector_BA (z : ℂ) (hz : z = -3 + 2 * Complex.I) : -z = 3 - 2 * Complex.I :=
by
  rw [hz]
  sorry

end vector_AB_to_vector_BA_l41_41408


namespace cassidy_posters_l41_41460

theorem cassidy_posters (p_two_years_ago : ℕ) (p_double : ℕ) (p_current : ℕ) (p_added : ℕ) 
    (h1 : p_two_years_ago = 14) 
    (h2 : p_double = 2 * p_two_years_ago)
    (h3 : p_current = 22)
    (h4 : p_added = p_double - p_current) : 
    p_added = 6 := 
by
  sorry

end cassidy_posters_l41_41460


namespace sum_odd_divisors_l41_41961

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l41_41961


namespace kramer_vote_percentage_l41_41068

def percentage_of_votes_cast (K : ℕ) (V : ℕ) : ℕ :=
  (K * 100) / V

theorem kramer_vote_percentage (K : ℕ) (V : ℕ) (h1 : K = 942568) 
  (h2 : V = 4 * K) : percentage_of_votes_cast K V = 25 := 
by 
  rw [h1, h2, percentage_of_votes_cast]
  sorry

end kramer_vote_percentage_l41_41068


namespace fraction_of_number_l41_41942

-- Given definitions based on the problem conditions
def fraction : ℚ := 3 / 4
def number : ℕ := 40

-- Theorem statement to be proved
theorem fraction_of_number : fraction * number = 30 :=
by
  sorry -- This indicates that the proof is not yet provided

end fraction_of_number_l41_41942


namespace triangle_is_right_triangle_l41_41671

theorem triangle_is_right_triangle (a b c : ℕ) (h_ratio : a = 3 * (36 / 12)) (h_perimeter : 3 * (36 / 12) + 4 * (36 / 12) + 5 * (36 / 12) = 36) :
  a^2 + b^2 = c^2 :=
by
  -- sorry for skipping the proof.
  sorry

end triangle_is_right_triangle_l41_41671


namespace solve_abs_equation_l41_41404

theorem solve_abs_equation (x : ℝ) :
  |2 * x - 1| + |x - 2| = |x + 1| ↔ 1 / 2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end solve_abs_equation_l41_41404


namespace problem_statement_part1_problem_statement_part2_problem_statement_part3_problem_statement_part4_l41_41498

variable (a b : ℝ)

theorem problem_statement_part1 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (1 / a + 2 / b) ≥ 9 := sorry

theorem problem_statement_part2 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (2 ^ a + 4 ^ b) ≥ 2 * Real.sqrt 2 := sorry

theorem problem_statement_part3 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (a * b) ≤ (1 / 8) := sorry

theorem problem_statement_part4 (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  (a^2 + b^2) ≥ (1 / 5) := sorry

end problem_statement_part1_problem_statement_part2_problem_statement_part3_problem_statement_part4_l41_41498


namespace inequality_always_holds_l41_41200

theorem inequality_always_holds (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 6 * x + 3 > 0) →
  (a > 3) ∧ (∀ x : ℝ, x = a + 9 / (a - 1) → x ≥ 7) :=
by
  sorry

end inequality_always_holds_l41_41200


namespace α_in_quadrants_l41_41048

def α (k : ℤ) : ℝ := k * 180 + 45

theorem α_in_quadrants (k : ℤ) : 
  (0 ≤ α k ∧ α k < 90) ∨ (180 < α k ∧ α k ≤ 270) :=
sorry

end α_in_quadrants_l41_41048


namespace blithe_toy_count_l41_41705

-- Define the initial number of toys, the number lost, and the number found.
def initial_toys := 40
def toys_lost := 6
def toys_found := 9

-- Define the total number of toys after the changes.
def total_toys_after_changes := initial_toys - toys_lost + toys_found

-- The proof statement.
theorem blithe_toy_count : total_toys_after_changes = 43 :=
by
  -- Placeholder for the proof
  sorry

end blithe_toy_count_l41_41705


namespace longest_segment_CD_l41_41036

variables (A B C D : Type)
variables (angle_ABD angle_ADB angle_BDC angle_CBD : ℝ)

axiom angle_ABD_eq : angle_ABD = 30
axiom angle_ADB_eq : angle_ADB = 65
axiom angle_BDC_eq : angle_BDC = 60
axiom angle_CBD_eq : angle_CBD = 80

theorem longest_segment_CD
  (h_ABD : angle_ABD = 30)
  (h_ADB : angle_ADB = 65)
  (h_BDC : angle_BDC = 60)
  (h_CBD : angle_CBD = 80) : false :=
sorry

end longest_segment_CD_l41_41036


namespace binom_8_5_l41_41012

-- Define the factorial function
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- Define the binomial coefficient function
def binom (n k : ℕ) := fact n / (fact k * fact (n - k))

-- State the theorem to prove binom 8 5 = 56
theorem binom_8_5 : binom 8 5 = 56 := by
  sorry

end binom_8_5_l41_41012


namespace prince_cd_total_spent_l41_41998

theorem prince_cd_total_spent (total_cds : ℕ)
    (pct_20 : ℕ) (pct_15 : ℕ) (pct_10 : ℕ)
    (bought_20_pct : ℕ) (bought_15_pct : ℕ)
    (bought_10_pct : ℕ) (bought_6_pct : ℕ)
    (discount_cnt_4 : ℕ) (discount_amount_4 : ℕ)
    (discount_cnt_5 : ℕ) (discount_amount_5 : ℕ)
    (total_cost_no_discount : ℕ) (total_discount : ℕ) (total_spent : ℕ) :
    total_cds = 400 ∧
    pct_20 = 25 ∧ pct_15 = 30 ∧ pct_10 = 20 ∧
    bought_20_pct = 70 ∧ bought_15_pct = 40 ∧
    bought_10_pct = 80 ∧ bought_6_pct = 100 ∧
    discount_cnt_4 = 4 ∧ discount_amount_4 = 5 ∧
    discount_cnt_5 = 5 ∧ discount_amount_5 = 3 ∧
    total_cost_no_discount - total_discount = total_spent ∧
    total_spent = 3119 := by
  sorry

end prince_cd_total_spent_l41_41998


namespace number_of_valid_pairs_l41_41103

theorem number_of_valid_pairs :
  ∃ (n : Nat), n = 8 ∧ 
  (∃ (a b : Int), 4 < a ∧ a < b ∧ b < 22 ∧ (4 + a + b + 22) / 4 = 13) :=
sorry

end number_of_valid_pairs_l41_41103


namespace each_person_paid_45_l41_41156

theorem each_person_paid_45 (total_bill : ℝ) (number_of_people : ℝ) (per_person_share : ℝ) 
    (h1 : total_bill = 135) 
    (h2 : number_of_people = 3) :
    per_person_share = 45 :=
by
  sorry

end each_person_paid_45_l41_41156


namespace problem_statement_l41_41740

-- Define the universal set U, and the sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4}

-- Define the complement of B in U
def C_U_B : Set ℕ := { x | x ∈ U ∧ x ∉ B }

-- State the theorem
theorem problem_statement : (A ∩ C_U_B) = {1, 2} :=
by {
  -- Proof is omitted
  sorry
}

end problem_statement_l41_41740


namespace avg_weight_of_a_b_c_l41_41109

theorem avg_weight_of_a_b_c (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 43) (h3 : B = 31) :
  (A + B + C) / 3 = 45 :=
by
  sorry

end avg_weight_of_a_b_c_l41_41109


namespace calculate_discount_l41_41389

def original_price := 22
def sale_price := 16

theorem calculate_discount : original_price - sale_price = 6 := 
by
  sorry

end calculate_discount_l41_41389


namespace min_third_side_of_right_triangle_l41_41364

theorem min_third_side_of_right_triangle (a b : ℕ) (h : a = 7 ∧ b = 24) : 
  ∃ (c : ℝ), c = Real.sqrt (576 - 49) :=
by
  sorry

end min_third_side_of_right_triangle_l41_41364


namespace john_ate_12_ounces_of_steak_l41_41901

-- Conditions
def original_weight : ℝ := 30
def burned_fraction : ℝ := 0.5
def eaten_fraction : ℝ := 0.8

-- Theorem statement
theorem john_ate_12_ounces_of_steak :
  (original_weight * (1 - burned_fraction) * eaten_fraction) = 12 := by
  sorry

end john_ate_12_ounces_of_steak_l41_41901


namespace initial_people_count_l41_41837

theorem initial_people_count (x : ℕ) (h : (x - 2) + 2 = 10) : x = 10 :=
by
  sorry

end initial_people_count_l41_41837


namespace per_minute_charge_after_6_minutes_l41_41826

noncomputable def cost_plan_a (x : ℝ) (t : ℝ) : ℝ :=
  if t <= 6 then 0.60 else 0.60 + (t - 6) * x

noncomputable def cost_plan_b (t : ℝ) : ℝ :=
  t * 0.08

theorem per_minute_charge_after_6_minutes :
  ∃ (x : ℝ), cost_plan_a x 12 = cost_plan_b 12 ∧ x = 0.06 :=
by
  use 0.06
  simp [cost_plan_a, cost_plan_b]
  sorry

end per_minute_charge_after_6_minutes_l41_41826


namespace binom_8_5_eq_56_l41_41008

theorem binom_8_5_eq_56 : nat.choose 8 5 = 56 := 
by
sorry

end binom_8_5_eq_56_l41_41008


namespace no_club_member_is_fraternity_member_l41_41577

variable (Student : Type) (isHonest : Student → Prop) 
                       (isFraternityMember : Student → Prop) 
                       (isClubMember : Student → Prop)

axiom some_students_not_honest : ∃ x : Student, ¬ isHonest x
axiom all_frats_honest : ∀ y : Student, isFraternityMember y → isHonest y
axiom no_clubs_honest : ∀ z : Student, isClubMember z → ¬ isHonest z

theorem no_club_member_is_fraternity_member : ∀ w : Student, isClubMember w → ¬ isFraternityMember w :=
by sorry

end no_club_member_is_fraternity_member_l41_41577


namespace maximum_value_of_expression_l41_41516

noncomputable def max_function_value (x y z : ℝ) : ℝ := 
  (x^3 - x * y^2 + y^3) * (x^3 - x * z^2 + z^3) * (y^3 - y * z^2 + z^3)

theorem maximum_value_of_expression : 
  ∃ x y z : ℝ, (x >= 0) ∧ (y >= 0) ∧ (z >= 0) ∧ (x + y + z = 3) 
  ∧ max_function_value x y z = 2916 / 2187 := 
sorry

end maximum_value_of_expression_l41_41516


namespace number_of_hens_l41_41140

theorem number_of_hens (H C : ℕ) (h1 : H + C = 46) (h2 : 2 * H + 4 * C = 136) : H = 24 :=
by
  sorry

end number_of_hens_l41_41140


namespace lost_weights_l41_41212

-- Define the weights
def weights : List ℕ := [43, 70, 57]

-- Total remaining weight after loss
def remaining_weight : ℕ := 20172

-- Number of weights lost
def weights_lost : ℕ := 4

-- Whether a given number of weights and types of weights match the remaining weight
def valid_loss (initial_count : ℕ) (lost_weight_count : ℕ) : Prop :=
  let total_initial_weight := initial_count * (weights.sum)
  let lost_weight := lost_weight_count * 57
  total_initial_weight - lost_weight = remaining_weight

-- Proposition we need to prove
theorem lost_weights (initial_count : ℕ) (h : valid_loss initial_count weights_lost) : ∀ w ∈ weights, w = 57 :=
by {
  sorry
}

end lost_weights_l41_41212


namespace martha_pins_l41_41522

theorem martha_pins (k : ℕ) :
  (2 + 9 * k > 45) ∧ (2 + 14 * k < 90) ↔ (k = 5 ∨ k = 6) :=
by
  sorry

end martha_pins_l41_41522


namespace jerrys_breakfast_calories_l41_41510

theorem jerrys_breakfast_calories 
    (num_pancakes : ℕ) (calories_per_pancake : ℕ) 
    (num_bacon : ℕ) (calories_per_bacon : ℕ) 
    (num_cereal : ℕ) (calories_per_cereal : ℕ) 
    (calories_total : ℕ) :
    num_pancakes = 6 →
    calories_per_pancake = 120 →
    num_bacon = 2 →
    calories_per_bacon = 100 →
    num_cereal = 1 →
    calories_per_cereal = 200 →
    calories_total = num_pancakes * calories_per_pancake
                   + num_bacon * calories_per_bacon
                   + num_cereal * calories_per_cereal →
    calories_total = 1120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6] at h7
  assumption

end jerrys_breakfast_calories_l41_41510


namespace odd_square_mod_eight_l41_41244

theorem odd_square_mod_eight (k : ℤ) : ((2 * k + 1) ^ 2) % 8 = 1 := 
sorry

end odd_square_mod_eight_l41_41244


namespace partI_solution_set_partII_range_of_a_l41_41049

namespace MathProof

-- Define the function f(x)
def f (x a : ℝ) : ℝ := abs (x - a) - abs (x + 3)

-- Part (Ⅰ) Proof Problem
theorem partI_solution_set (x : ℝ) : 
  f x (-1) ≤ 1 ↔ -5/2 ≤ x :=
sorry

-- Part (Ⅱ) Proof Problem
theorem partII_range_of_a (a : ℝ) : 
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f x a ≤ 4) ↔ -7 ≤ a ∧ a ≤ 7 :=
sorry

end MathProof

end partI_solution_set_partII_range_of_a_l41_41049


namespace cos_double_angle_l41_41881

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by 
  sorry

end cos_double_angle_l41_41881


namespace super12_teams_l41_41631

theorem super12_teams :
  ∃ n : ℕ, (n * (n - 1) = 132) ∧ n = 12 := by
  sorry

end super12_teams_l41_41631


namespace sum_odd_divisors_l41_41960

open BigOperators

/-- Sum of the positive odd divisors of 90 is 78. -/
theorem sum_odd_divisors (n : ℕ) (h : n = 90) : ∑ (d in (Finset.filter (fun x => x % 2 = 1) (Finset.divisors n)), d) = 78 := by
  have h := Finset.divisors_filter_odd_of_factorisation n
  sorry

end sum_odd_divisors_l41_41960


namespace proposition_relationship_l41_41887
-- Import library

-- Statement of the problem
theorem proposition_relationship (p q : Prop) (hpq : p ∨ q) (hnp : ¬p) : ¬p ∧ q :=
  by
  sorry

end proposition_relationship_l41_41887


namespace correct_option_is_D_l41_41067

noncomputable def data : List ℕ := [7, 5, 3, 5, 10]

theorem correct_option_is_D :
  let mean := (7 + 5 + 3 + 5 + 10) / 5
  let variance := (1 / 5 : ℚ) * ((7 - mean) ^ 2 + (5 - mean) ^ 2 + (5 - mean) ^ 2 + (3 - mean) ^ 2 + (10 - mean) ^ 2)
  let mode := 5
  let median := 5
  mean = 6 ∧ variance ≠ 3.6 ∧ mode ≠ 10 ∧ median ≠ 3 :=
by
  sorry

end correct_option_is_D_l41_41067


namespace ratio_youngest_sister_to_yvonne_l41_41438

def laps_yvonne := 10
def laps_joel := 15
def joel_ratio := 3

theorem ratio_youngest_sister_to_yvonne
  (laps_yvonne : ℕ)
  (laps_joel : ℕ)
  (joel_ratio : ℕ)
  (H_joel : laps_joel = 3 * (laps_yvonne / joel_ratio))
  : (laps_joel / joel_ratio) = laps_yvonne / 2 :=
by
  sorry

end ratio_youngest_sister_to_yvonne_l41_41438


namespace find_xyz_l41_41076

-- Let a, b, c, x, y, z be nonzero complex numbers
variables (a b c x y z : ℂ)
-- Given conditions
variables (h1 : a = (b + c) / (x - 2))
variables (h2 : b = (a + c) / (y - 2))
variables (h3 : c = (a + b) / (z - 2))
variables (h4 : x * y + x * z + y * z = 5)
variables (h5 : x + y + z = 3)

-- Prove that xyz = 5
theorem find_xyz : x * y * z = 5 :=
by
  sorry

end find_xyz_l41_41076


namespace geometric_seq_not_sufficient_necessary_l41_41074

theorem geometric_seq_not_sufficient_necessary (a_n : ℕ → ℝ) (q : ℝ) 
  (h1 : ∀ n, a_n (n+1) = a_n n * q) : 
  ¬ ((∃ q > 1, ∀ n, a_n (n+1) > a_n n) ∧ (∀ q > 1, ∀ n, a_n (n+1) > a_n n)) := 
sorry

end geometric_seq_not_sufficient_necessary_l41_41074


namespace graph_quadrant_exclusion_l41_41331

theorem graph_quadrant_exclusion (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ∀ x : ℝ, ¬ ((a^x + b > 0) ∧ (x > 0)) :=
by
  sorry

end graph_quadrant_exclusion_l41_41331


namespace length_of_rectangle_l41_41410

-- Define the conditions as given in the problem
variables (width : ℝ) (perimeter : ℝ) (length : ℝ)

-- The conditions provided
def conditions : Prop :=
  width = 15 ∧ perimeter = 70 ∧ perimeter = 2 * (length + width)

-- The statement to prove: the length of the rectangle is 20 feet
theorem length_of_rectangle {width perimeter length : ℝ} (h : conditions width perimeter length) : length = 20 :=
by 
  -- This is where the proof steps would go
  sorry

end length_of_rectangle_l41_41410


namespace work_ratio_l41_41275

theorem work_ratio 
  (m b : ℝ) 
  (h : 7 * m + 2 * b = 6 * (m + b)) : 
  m / b = 4 := 
sorry

end work_ratio_l41_41275


namespace min_toys_to_add_l41_41273

theorem min_toys_to_add (T x : ℕ) (h1 : T % 12 = 3) (h2 : T % 18 = 3) :
  ((T + x) % 7 = 0) → x = 4 :=
by
  sorry

end min_toys_to_add_l41_41273


namespace positive_A_satisfies_eq_l41_41515

theorem positive_A_satisfies_eq :
  ∃ (A : ℝ), A > 0 ∧ A^2 + 49 = 194 → A = Real.sqrt 145 :=
by
  sorry

end positive_A_satisfies_eq_l41_41515


namespace nancy_flooring_area_l41_41769

def area_of_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem nancy_flooring_area :
  let central_area_length := 10
  let central_area_width := 10
  let hallway_length := 6
  let hallway_width := 4
  let central_area := area_of_rectangle central_area_length central_area_width
  let hallway_area := area_of_rectangle hallway_length hallway_width
  let total_area := central_area + hallway_area
  total_area = 124 :=
by
  rfl  -- This is where the proof would go.

end nancy_flooring_area_l41_41769


namespace moles_of_CO2_required_l41_41056

theorem moles_of_CO2_required (n_H2O n_H2CO3 : ℕ) (h1 : n_H2O = n_H2CO3) (h2 : n_H2O = 2): 
  (n_H2O = 2) → (∃ n_CO2 : ℕ, n_CO2 = n_H2O) :=
by
  sorry

end moles_of_CO2_required_l41_41056


namespace modulus_of_z_l41_41745

-- Define the given condition
def condition (z : ℂ) : Prop := (z - 3) * (1 - 3 * Complex.I) = 10

-- State the main theorem
theorem modulus_of_z (z : ℂ) (h : condition z) : Complex.abs z = 5 :=
sorry

end modulus_of_z_l41_41745


namespace simplify_expression_l41_41528

theorem simplify_expression :
  (144 / 12) * (5 / 90) * (9 / 3) * 2 = 4 := by
  sorry

end simplify_expression_l41_41528


namespace extra_days_per_grade_below_b_l41_41461

theorem extra_days_per_grade_below_b :
  ∀ (total_days lying_days grades_below_B : ℕ), 
  total_days = 26 → lying_days = 14 → grades_below_B = 4 → 
  (total_days - lying_days) / grades_below_B = 3 :=
by
  -- conditions and steps of the proof will be here
  sorry

end extra_days_per_grade_below_b_l41_41461


namespace hyperbola_equation_l41_41846

theorem hyperbola_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 - 2 → (∃ k : ℝ, k ≠ 0 ∧ x * y = k) := 
by
  intros h
  sorry

end hyperbola_equation_l41_41846


namespace relationship_between_x_y_l41_41032

theorem relationship_between_x_y (x y m : ℝ) (h₁ : x + m = 4) (h₂ : y - 5 = m) : x + y = 9 := 
sorry

end relationship_between_x_y_l41_41032


namespace juniors_to_freshmen_ratio_l41_41702

variable (f s j : ℕ)

def participated_freshmen := 3 * f / 7
def participated_sophomores := 5 * s / 7
def participated_juniors := j / 2

-- The statement
theorem juniors_to_freshmen_ratio
    (h1 : participated_freshmen = participated_sophomores)
    (h2 : participated_freshmen = participated_juniors) :
    j = 6 * f / 7 ∧ f = 7 * j / 6 :=
by
  sorry

end juniors_to_freshmen_ratio_l41_41702


namespace tenth_integer_from_permutations_l41_41935

theorem tenth_integer_from_permutations : ∃ n : ℕ, nth_permutation [1, 2, 5, 6] n = 2561 :=
by
  sorry

def nth_permutation (digits : List ℕ) (n : ℕ) : ℕ :=
  sorry

end tenth_integer_from_permutations_l41_41935


namespace product_of_sequence_is_243_l41_41000

theorem product_of_sequence_is_243 : 
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049) = 243 := 
by
  sorry

end product_of_sequence_is_243_l41_41000


namespace polygon_sides_l41_41422

theorem polygon_sides (n : ℕ) : (n - 2) * 180 + 360 = 1980 → n = 11 :=
by sorry

end polygon_sides_l41_41422


namespace oranges_picked_l41_41393

theorem oranges_picked (total_oranges second_tree third_tree : ℕ) 
    (h1 : total_oranges = 260) 
    (h2 : second_tree = 60) 
    (h3 : third_tree = 120) : 
    total_oranges - (second_tree + third_tree) = 80 := by 
  sorry

end oranges_picked_l41_41393
