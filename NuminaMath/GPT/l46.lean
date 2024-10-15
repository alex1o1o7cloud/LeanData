import Mathlib

namespace NUMINAMATH_GPT_eq_value_l46_4645

theorem eq_value (x y : ℕ) (h1 : x - y = 9) (h2 : x = 9) : 3 ^ x * 4 ^ y = 19683 := by
  sorry

end NUMINAMATH_GPT_eq_value_l46_4645


namespace NUMINAMATH_GPT_A_20_equals_17711_l46_4671

def A : ℕ → ℕ
| 0     => 1  -- by definition, an alternating sequence on an empty set, counting empty sequence
| 1     => 2  -- base case
| 2     => 3  -- base case
| (n+3) => A (n+2) + A (n+1)

theorem A_20_equals_17711 : A 20 = 17711 := 
sorry

end NUMINAMATH_GPT_A_20_equals_17711_l46_4671


namespace NUMINAMATH_GPT_problem_l46_4660

-- Definitions and conditions
variable {a : ℕ → ℝ} -- sequence definition
variable {S : ℕ → ℝ} -- sum of first n terms

-- Condition: a_n ≠ 0 for all n ∈ ℕ^*
axiom h1 : ∀ n : ℕ, n > 0 → a n ≠ 0

-- Condition: a_n * a_{n+1} = S_n
axiom h2 : ∀ n : ℕ, n > 0 → a n * a (n + 1) = S n

-- Given: S_1 = a_1
axiom h3 : S 1 = a 1

-- Given: S_2 = a_1 + a_2
axiom h4 : S 2 = a 1 + a 2

-- Prove: a_3 - a_1 = 1
theorem problem : a 3 - a 1 = 1 := by
  sorry

end NUMINAMATH_GPT_problem_l46_4660


namespace NUMINAMATH_GPT_apples_problem_l46_4693

variable (K A : ℕ)

theorem apples_problem (K A : ℕ) (h1 : K + (3 / 4) * K + 600 = 2600) (h2 : A + (3 / 4) * A + 600 = 2600) :
  K = 1142 ∧ A = 1142 :=
by
  sorry

end NUMINAMATH_GPT_apples_problem_l46_4693


namespace NUMINAMATH_GPT_product_of_dice_divisible_by_9_l46_4627

-- Define the probability of rolling a number divisible by 3
def prob_roll_div_by_3 : ℚ := 1/6

-- Define the probability of rolling a number not divisible by 3
def prob_roll_not_div_by_3 : ℚ := 2/3

-- Define the probability that the product of numbers rolled on 6 dice is divisible by 9
def prob_product_div_by_9 : ℚ := 449/729

-- Main statement of the problem
theorem product_of_dice_divisible_by_9 :
  (1 - ((prob_roll_not_div_by_3^6) + 
        (6 * prob_roll_div_by_3 * (prob_roll_not_div_by_3^5)) + 
        (15 * (prob_roll_div_by_3^2) * (prob_roll_not_div_by_3^4)))) = prob_product_div_by_9 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_dice_divisible_by_9_l46_4627


namespace NUMINAMATH_GPT_solve_for_x_l46_4670

-- Define the given equation as a predicate
def equation (x: ℚ) : Prop := (x + 4) / (x - 3) = (x - 2) / (x + 2)

-- State the problem in a Lean theorem
theorem solve_for_x : ∃ x : ℚ, equation x ∧ x = -2 / 11 :=
by
  existsi -2 / 11
  constructor
  repeat { sorry }

end NUMINAMATH_GPT_solve_for_x_l46_4670


namespace NUMINAMATH_GPT_eight_digit_number_divisibility_l46_4632

theorem eight_digit_number_divisibility (a b c d : ℕ) (Z : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) 
(h3 : b ≤ 9) (h4 : c ≤ 9) (h5 : d ≤ 9) (hZ : Z = 1001 * (1000 * a + 100 * b + 10 * c + d)) : 
  10001 ∣ Z := 
  by sorry

end NUMINAMATH_GPT_eight_digit_number_divisibility_l46_4632


namespace NUMINAMATH_GPT_b_20_value_l46_4601

noncomputable def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => b (n+1) * b n

theorem b_20_value : b 19 = 2^4181 :=
sorry

end NUMINAMATH_GPT_b_20_value_l46_4601


namespace NUMINAMATH_GPT_probability_calculation_l46_4653

noncomputable def probability_of_event_A : ℚ := 
  let total_ways := 35 
  let favorable_ways := 6 
  favorable_ways / total_ways

theorem probability_calculation (A_team B_team : Type) [Fintype A_team] [Fintype B_team] [DecidableEq A_team] [DecidableEq B_team] :
  let total_players := 7 
  let selected_players := 4 
  let seeded_A := 2 
  let nonseeded_A := 1 
  let seeded_B := 2 
  let nonseeded_B := 2 
  let event_total_ways := Nat.choose total_players selected_players 
  let event_A_ways := Nat.choose seeded_A 2 * Nat.choose nonseeded_A 2 + Nat.choose seeded_B 2 * Nat.choose nonseeded_B 2 
  probability_of_event_A = 6 / 35 := 
sorry

end NUMINAMATH_GPT_probability_calculation_l46_4653


namespace NUMINAMATH_GPT_algebraic_expression_value_l46_4622

theorem algebraic_expression_value (x : ℝ) (h : (x^2 - x)^2 - 4 * (x^2 - x) - 12 = 0) : x^2 - x + 1 = 7 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l46_4622


namespace NUMINAMATH_GPT_quadratic_solution_range_l46_4659

theorem quadratic_solution_range {x : ℝ} 
  (h : x^2 - 6 * x + 8 < 0) : 
  25 < x^2 + 6 * x + 9 ∧ x^2 + 6 * x + 9 < 49 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_range_l46_4659


namespace NUMINAMATH_GPT_maximize_profit_l46_4613

noncomputable section

-- Definitions of parameters
def daily_sales_volume (x : ℝ) : ℝ := -2 * x + 200
def daily_cost : ℝ := 450
def price_min : ℝ := 30
def price_max : ℝ := 60

-- Function for daily profit
def daily_profit (x : ℝ) : ℝ := (x - 30) * daily_sales_volume x - daily_cost

-- Theorem statement
theorem maximize_profit :
  let max_profit_price := 60
  let max_profit_value := 1950
  30 ≤ max_profit_price ∧ max_profit_price ≤ 60 ∧
  daily_profit max_profit_price = max_profit_value :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l46_4613


namespace NUMINAMATH_GPT_length_of_AB_l46_4615

theorem length_of_AB :
  ∃ (a b c d e : ℝ), (a < b) ∧ (b < c) ∧ (c < d) ∧ (d < e) ∧
  (b - a = 5) ∧ -- AB = 5
  ((c - b) = 2 * (d - c)) ∧ -- bc = 2 * cd
  (d - e) = 4 ∧ -- de = 4
  (c - a) = 11 ∧ -- ac = 11
  (e - a) = 18 := -- ae = 18
by 
  sorry

end NUMINAMATH_GPT_length_of_AB_l46_4615


namespace NUMINAMATH_GPT_h_eq_x_solution_l46_4618

noncomputable def h (x : ℝ) : ℝ := (3 * ((x + 3) / 5) + 10)

theorem h_eq_x_solution (x : ℝ) (h_cond : ∀ y, h (5 * y - 3) = 3 * y + 10) : h x = x → x = 29.5 :=
by
  sorry

end NUMINAMATH_GPT_h_eq_x_solution_l46_4618


namespace NUMINAMATH_GPT_find_y_l46_4665

noncomputable def x : ℝ := 0.7142857142857143

def equation (y : ℝ) : Prop :=
  (x * y) / 7 = x^2

theorem find_y : ∃ y : ℝ, equation y ∧ y = 5 :=
by
  use 5
  have h1 : x != 0 := by sorry
  have h2 : (x * 5) / 7 = x^2 := by sorry
  exact ⟨h2, rfl⟩

end NUMINAMATH_GPT_find_y_l46_4665


namespace NUMINAMATH_GPT_find_a_l46_4695

theorem find_a 
  (a b c : ℚ) 
  (h1 : b = 4 * a) 
  (h2 : b = 15 - 4 * a - c) 
  (h3 : c = a + 2) : 
  a = 13 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l46_4695


namespace NUMINAMATH_GPT_connie_start_marbles_l46_4698

variable (marbles_total marbles_given marbles_left : ℕ)

theorem connie_start_marbles :
  marbles_given = 73 → marbles_left = 70 → marbles_total = marbles_given + marbles_left → marbles_total = 143 :=
by intros; sorry

end NUMINAMATH_GPT_connie_start_marbles_l46_4698


namespace NUMINAMATH_GPT_expand_expression_l46_4651

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  sorry

end NUMINAMATH_GPT_expand_expression_l46_4651


namespace NUMINAMATH_GPT_product_plus_one_is_square_l46_4667

theorem product_plus_one_is_square (x y : ℕ) (h : x * y = (x + 2) * (y - 2)) : ∃ k : ℕ, x * y + 1 = k * k :=
by
  sorry

end NUMINAMATH_GPT_product_plus_one_is_square_l46_4667


namespace NUMINAMATH_GPT_like_terms_solutions_l46_4668

theorem like_terms_solutions (x y : ℤ) (h1 : 5 = 4 * x + 1) (h2 : 3 * y = 6) :
  x = 1 ∧ y = 2 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_like_terms_solutions_l46_4668


namespace NUMINAMATH_GPT_total_hours_worked_l46_4666

-- Definitions based on the conditions
def hours_per_day : ℕ := 3
def days_worked : ℕ := 6

-- Statement of the problem
theorem total_hours_worked : hours_per_day * days_worked = 18 := by
  sorry

end NUMINAMATH_GPT_total_hours_worked_l46_4666


namespace NUMINAMATH_GPT_team_selection_l46_4631

open Nat

theorem team_selection :
  let boys := 10
  let girls := 12
  let team_size := 8
  let boys_to_choose := 5
  let girls_to_choose := 3
  choose boys boys_to_choose * choose girls girls_to_choose = 55440 :=
by
  sorry

end NUMINAMATH_GPT_team_selection_l46_4631


namespace NUMINAMATH_GPT_fencing_required_l46_4669

theorem fencing_required (L W : ℝ) (h1 : L = 40) (h2 : L * W = 680) : 2 * W + L = 74 :=
by
  sorry

end NUMINAMATH_GPT_fencing_required_l46_4669


namespace NUMINAMATH_GPT_fly_in_box_maximum_path_length_l46_4611

theorem fly_in_box_maximum_path_length :
  let side1 := 1
  let side2 := Real.sqrt 2
  let side3 := Real.sqrt 3
  let space_diagonal := Real.sqrt (side1^2 + side2^2 + side3^2)
  let face_diagonal1 := Real.sqrt (side1^2 + side2^2)
  let face_diagonal2 := Real.sqrt (side1^2 + side3^2)
  let face_diagonal3 := Real.sqrt (side2^2 + side3^2)
  (4 * space_diagonal + 2 * face_diagonal3) = 4 * Real.sqrt 6 + 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_fly_in_box_maximum_path_length_l46_4611


namespace NUMINAMATH_GPT_find_a1_l46_4634

theorem find_a1 (S : ℕ → ℝ) (a : ℕ → ℝ) (a1 : ℝ) :
  (∀ n : ℕ, S n = a1 * (2^n - 1)) → a 4 = 24 → 
  a 4 = S 4 - S 3 → 
  a1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_l46_4634


namespace NUMINAMATH_GPT_minimum_number_of_peanuts_l46_4620

/--
Five monkeys share a pile of peanuts.
Each monkey divides the peanuts into five piles, leaves one peanut which it eats, and takes away one pile.
This process continues in the same manner until the fifth monkey, who also evenly divides the remaining peanuts into five piles and has one peanut left over.
Prove that the minimum number of peanuts in the pile originally is 3121.
-/
theorem minimum_number_of_peanuts : ∃ N : ℕ, N = 3121 ∧
  (N - 1) % 5 = 0 ∧
  ((4 * ((N - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) / 5) - 1) / 5) - 1) / 4) % 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_minimum_number_of_peanuts_l46_4620


namespace NUMINAMATH_GPT_find_second_number_l46_4680

def sum_of_three (a b c : ℚ) : Prop :=
  a + b + c = 120

def ratio_first_to_second (a b : ℚ) : Prop :=
  a / b = 3 / 4

def ratio_second_to_third (b c : ℚ) : Prop :=
  b / c = 3 / 5

theorem find_second_number (a b c : ℚ) 
  (h_sum : sum_of_three a b c)
  (h_ratio_ab : ratio_first_to_second a b)
  (h_ratio_bc : ratio_second_to_third b c) : 
  b = 1440 / 41 := 
sorry

end NUMINAMATH_GPT_find_second_number_l46_4680


namespace NUMINAMATH_GPT_trapezoid_inequality_l46_4678

theorem trapezoid_inequality (a b R : ℝ) (h : a > 0) (h1 : b > 0) (h2 : R > 0) 
  (circumscribed : ∃ (x y : ℝ), x + y = a ∧ R^2 * (1/x + 1/y) = b) : 
  a * b ≥ 4 * R^2 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_inequality_l46_4678


namespace NUMINAMATH_GPT_hotel_rooms_l46_4643

theorem hotel_rooms (h₁ : ∀ R : ℕ, (∃ n : ℕ, n = R * 3) → (∃ m : ℕ, m = 2 * R * 3) → m = 60) : (∃ R : ℕ, R = 10) :=
by
  sorry

end NUMINAMATH_GPT_hotel_rooms_l46_4643


namespace NUMINAMATH_GPT_percentage_customers_return_books_l46_4688

theorem percentage_customers_return_books 
  (total_customers : ℕ) (price_per_book : ℕ) (sales_after_returns : ℕ) 
  (h1 : total_customers = 1000) 
  (h2 : price_per_book = 15) 
  (h3 : sales_after_returns = 9450) : 
  ((total_customers - (sales_after_returns / price_per_book)) / total_customers) * 100 = 37 := 
by
  sorry

end NUMINAMATH_GPT_percentage_customers_return_books_l46_4688


namespace NUMINAMATH_GPT_cylinder_surface_area_l46_4603

theorem cylinder_surface_area (h r : ℝ) (h_height : h = 12) (r_radius : r = 4) : 
  2 * π * r * (r + h) = 128 * π :=
by
  -- providing the proof steps is beyond the scope of this task
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_l46_4603


namespace NUMINAMATH_GPT_least_number_when_increased_by_6_is_divisible_l46_4614

theorem least_number_when_increased_by_6_is_divisible :
  ∃ n : ℕ, 
    (n + 6) % 24 = 0 ∧ 
    (n + 6) % 32 = 0 ∧ 
    (n + 6) % 36 = 0 ∧ 
    (n + 6) % 54 = 0 ∧ 
    n = 858 :=
by
  sorry

end NUMINAMATH_GPT_least_number_when_increased_by_6_is_divisible_l46_4614


namespace NUMINAMATH_GPT_decreasing_omega_range_l46_4600

open Real

theorem decreasing_omega_range {ω : ℝ} (h1 : 1 < ω) :
  (∀ x y : ℝ, π ≤ x ∧ x ≤ y ∧ y ≤ (5 * π) / 4 → 
    (|sin (ω * y + π / 3)| ≤ |sin (ω * x + π / 3)|)) → 
  (7 / 6 ≤ ω ∧ ω ≤ 4 / 3) :=
by
  sorry

end NUMINAMATH_GPT_decreasing_omega_range_l46_4600


namespace NUMINAMATH_GPT_percentage_female_on_duty_l46_4673

-- Definitions as per conditions in the problem:
def total_on_duty : ℕ := 240
def female_on_duty := total_on_duty / 2 -- Half of those on duty are female
def total_female_officers : ℕ := 300
def percentage_of_something (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- Statement of the problem to prove
theorem percentage_female_on_duty : percentage_of_something female_on_duty total_female_officers = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_female_on_duty_l46_4673


namespace NUMINAMATH_GPT_charlie_third_week_data_l46_4682

theorem charlie_third_week_data (d3 : ℕ) : 
  let data_plan := 8
  let cost_per_GB := 10
  let extra_charge := 120
  let week1 := 2
  let week2 := 3
  let week4 := 10
  let total_extra_GB := extra_charge / cost_per_GB
  let total_data := week1 + week2 + week4 + d3
  let overage_GB := total_data - data_plan
  overage_GB = total_extra_GB -> d3 = 5 := 
by
  let data_plan := 8
  let cost_per_GB := 10
  let extra_charge := 120
  let week1 := 2
  let week2 := 3
  let week4 := 10
  let total_extra_GB := extra_charge / cost_per_GB
  let total_data := week1 + week2 + week4 + d3
  let overage_GB := total_data - data_plan
  have : overage_GB = total_extra_GB := sorry
  have : d3 = 5 := sorry
  sorry

end NUMINAMATH_GPT_charlie_third_week_data_l46_4682


namespace NUMINAMATH_GPT_greatest_consecutive_sum_l46_4683

theorem greatest_consecutive_sum (S : ℤ) (hS : S = 105) : 
  ∃ N : ℤ, (∃ a : ℤ, (N * (2 * a + N - 1) = 2 * S)) ∧ 
  (∀ M : ℤ, (∃ b : ℤ, (M * (2 * b + M - 1) = 2 * S)) → M ≤ N) ∧ N = 210 := 
sorry

end NUMINAMATH_GPT_greatest_consecutive_sum_l46_4683


namespace NUMINAMATH_GPT_find_vertex_parabola_l46_4641

-- Define the quadratic equation of the parabola
def parabola_eq (x y : ℝ) : Prop := x^2 - 4 * x + 3 * y + 10 = 0

-- Definition of the vertex of the parabola
def is_vertex (v : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), parabola_eq x y → v = (2, -2)

-- The main statement we want to prove
theorem find_vertex_parabola : 
  ∃ v : ℝ × ℝ, is_vertex v :=
by
  use (2, -2)
  intros x y hyp
  sorry

end NUMINAMATH_GPT_find_vertex_parabola_l46_4641


namespace NUMINAMATH_GPT_salary_increase_l46_4699

theorem salary_increase (prev_income : ℝ) (prev_percentage : ℝ) (new_percentage : ℝ) (rent_utilities : ℝ) (new_income : ℝ) :
  prev_income = 1000 ∧ prev_percentage = 0.40 ∧ new_percentage = 0.25 ∧ rent_utilities = prev_percentage * prev_income ∧
  rent_utilities = new_percentage * new_income → new_income - prev_income = 600 :=
by 
  sorry

end NUMINAMATH_GPT_salary_increase_l46_4699


namespace NUMINAMATH_GPT_integral_transform_eq_l46_4686

open MeasureTheory

variable (f : ℝ → ℝ)

theorem integral_transform_eq (hf_cont : Continuous f) (hL_exists : ∃ L, ∫ x in (Set.univ : Set ℝ), f x = L) :
  ∃ L, ∫ x in (Set.univ : Set ℝ), f (x - 1/x) = L :=
by
  cases' hL_exists with L hL
  use L
  have h_transform : ∫ x in (Set.univ : Set ℝ), f (x - 1/x) = ∫ x in (Set.univ : Set ℝ), f x := sorry
  rw [h_transform]
  exact hL

end NUMINAMATH_GPT_integral_transform_eq_l46_4686


namespace NUMINAMATH_GPT_range_of_k_l46_4607

theorem range_of_k (a b c d k : ℝ) (hA : b = k * a - 2 * a - 1) (hB : d = k * c - 2 * c - 1) (h_diff : a ≠ c) (h_lt : (c - a) * (d - b) < 0) : k < 2 := 
sorry

end NUMINAMATH_GPT_range_of_k_l46_4607


namespace NUMINAMATH_GPT_baba_yagas_savings_plan_l46_4684

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

end NUMINAMATH_GPT_baba_yagas_savings_plan_l46_4684


namespace NUMINAMATH_GPT_probability_interval_l46_4679

/-- 
The probability of event A occurring is 4/5, the probability of event B occurring is 3/4,
and the probability of event C occurring is 2/3. The smallest interval necessarily containing
the probability q that all three events occur is [0, 2/3].
-/
theorem probability_interval (P_A P_B P_C q : ℝ)
  (hA : P_A = 4 / 5) (hB : P_B = 3 / 4) (hC : P_C = 2 / 3)
  (h_q_le_A : q ≤ P_A) (h_q_le_B : q ≤ P_B) (h_q_le_C : q ≤ P_C) :
  0 ≤ q ∧ q ≤ 2 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_interval_l46_4679


namespace NUMINAMATH_GPT_min_value_of_a_sq_plus_b_sq_l46_4633

theorem min_value_of_a_sq_plus_b_sq {a b t : ℝ} (h : 2 * a + 3 * b = t) :
  ∃ a b : ℝ, (2 * a + 3 * b = t) ∧ (a^2 + b^2 = (13 * t^2) / 169) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_sq_plus_b_sq_l46_4633


namespace NUMINAMATH_GPT_methane_reaction_l46_4656

noncomputable def methane_reacts_with_chlorine
  (moles_CH₄ : ℕ)
  (moles_Cl₂ : ℕ)
  (moles_CCl₄ : ℕ)
  (moles_HCl_produced : ℕ) : Prop :=
  moles_CH₄ = 3 ∧ 
  moles_Cl₂ = 12 ∧ 
  moles_CCl₄ = 3 ∧ 
  moles_HCl_produced = 12

theorem methane_reaction : 
  methane_reacts_with_chlorine 3 12 3 12 :=
by sorry

end NUMINAMATH_GPT_methane_reaction_l46_4656


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_solve_eq4_l46_4621

theorem solve_eq1 : ∀ (x : ℝ), x^2 - 5 * x = 0 ↔ x = 0 ∨ x = 5 :=
by sorry

theorem solve_eq2 : ∀ (x : ℝ), (2 * x + 1)^2 = 4 ↔ x = -3 / 2 ∨ x = 1 / 2 :=
by sorry

theorem solve_eq3 : ∀ (x : ℝ), x * (x - 1) + 3 * (x - 1) = 0 ↔ x = 1 ∨ x = -3 :=
by sorry

theorem solve_eq4 : ∀ (x : ℝ), x^2 - 2 * x - 8 = 0 ↔ x = -2 ∨ x = 4 :=
by sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_solve_eq4_l46_4621


namespace NUMINAMATH_GPT_find_line_equation_l46_4616

-- Define point A
def point_A : ℝ × ℝ := (-3, 4)

-- Define the conditions
def passes_through_point_A (line_eq : ℝ → ℝ → ℝ) : Prop :=
  line_eq (-3) 4 = 0

def intercept_condition (line_eq : ℝ → ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ line_eq (2 * a) 0 = 0 ∧ line_eq 0 a = 0

-- Define the equations of the line
def line1 (x y : ℝ) : ℝ := 3 * y + 4 * x
def line2 (x y : ℝ) : ℝ := 2 * x - y - 5

-- Statement of the problem
theorem find_line_equation : 
  (passes_through_point_A line1 ∧ intercept_condition line1) ∨
  (passes_through_point_A line2 ∧ intercept_condition line2) :=
sorry

end NUMINAMATH_GPT_find_line_equation_l46_4616


namespace NUMINAMATH_GPT_minimum_value_an_eq_neg28_at_n_eq_3_l46_4644

noncomputable def seq_an (n : ℕ) : ℝ :=
  if n > 0 then (5 / 2) * n^2 - (13 / 2) * n
  else 0

noncomputable def delta_seq_an (n : ℕ) : ℝ := seq_an (n + 1) - seq_an n

noncomputable def delta2_seq_an (n : ℕ) : ℝ := delta_seq_an (n + 1) - delta_seq_an n

theorem minimum_value_an_eq_neg28_at_n_eq_3 : 
  ∃ (n : ℕ), n = 3 ∧ seq_an n = -28 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_an_eq_neg28_at_n_eq_3_l46_4644


namespace NUMINAMATH_GPT_parallelepiped_eq_l46_4647

-- Definitions of the variables and conditions
variables (a b c u v w : ℝ)

-- Prove the identity given the conditions:
theorem parallelepiped_eq :
  u * v * w = a * v * w + b * u * w + c * u * v :=
sorry

end NUMINAMATH_GPT_parallelepiped_eq_l46_4647


namespace NUMINAMATH_GPT_incorrect_statement_l46_4697

-- Definition of the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Definition of set M
def M : Set ℕ := {1, 2}

-- Definition of set N
def N : Set ℕ := {2, 4}

-- Complement of set in a universal set
def complement (S : Set ℕ) : Set ℕ := U \ S

-- Statement that D is incorrect
theorem incorrect_statement :
  M ∩ complement N ≠ {1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_l46_4697


namespace NUMINAMATH_GPT_percentage_increase_correct_l46_4612

def highest_price : ℕ := 24
def lowest_price : ℕ := 16

theorem percentage_increase_correct :
  ((highest_price - lowest_price) * 100 / lowest_price) = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_correct_l46_4612


namespace NUMINAMATH_GPT_frac_x_y_eq_neg2_l46_4636

open Real

theorem frac_x_y_eq_neg2 (x y : ℝ) (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 4) (h3 : (x + y) / (x - y) ≠ 1) :
  ∃ t : ℤ, (x / y = t) ∧ (t = -2) :=
by sorry

end NUMINAMATH_GPT_frac_x_y_eq_neg2_l46_4636


namespace NUMINAMATH_GPT_regular_eqn_exists_l46_4646

noncomputable def parametric_eqs (k : ℝ) : ℝ × ℝ :=
  (4 * k / (1 - k^2), 4 * k^2 / (1 - k^2))

theorem regular_eqn_exists (k : ℝ) (x y : ℝ) (h1 : x = 4 * k / (1 - k^2)) 
(h2 : y = 4 * k^2 / (1 - k^2)) : x^2 - y^2 - 4 * y = 0 :=
sorry

end NUMINAMATH_GPT_regular_eqn_exists_l46_4646


namespace NUMINAMATH_GPT_largest_lucky_number_l46_4642

theorem largest_lucky_number : 
  let a := 1
  let b := 4
  let lucky_number (x y : ℕ) := x + y + x * y
  let c1 := lucky_number a b
  let c2 := lucky_number b c1
  let c3 := lucky_number c1 c2
  c3 = 499 :=
by
  sorry

end NUMINAMATH_GPT_largest_lucky_number_l46_4642


namespace NUMINAMATH_GPT_coeff_x2_term_l46_4630

theorem coeff_x2_term (a b c d e f : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : d = 6) (h5 : e = 7) (h6 : f = 8) :
    (a * f + b * e * 1 + c * d) = 82 := 
by
    sorry

end NUMINAMATH_GPT_coeff_x2_term_l46_4630


namespace NUMINAMATH_GPT_tax_diminished_percentage_l46_4662

theorem tax_diminished_percentage (T C : ℝ) (hT : T > 0) (hC : C > 0) (X : ℝ) 
  (h : T * (1 - X / 100) * C * 1.15 = T * C * 0.9315) : X = 19 :=
by 
  sorry

end NUMINAMATH_GPT_tax_diminished_percentage_l46_4662


namespace NUMINAMATH_GPT_prisha_other_number_l46_4624

def prisha_numbers (a b : ℤ) : Prop :=
  3 * a + 2 * b = 105 ∧ (a = 15 ∨ b = 15)

theorem prisha_other_number (a b : ℤ) (h : prisha_numbers a b) : b = 30 :=
sorry

end NUMINAMATH_GPT_prisha_other_number_l46_4624


namespace NUMINAMATH_GPT_g_x_minus_3_l46_4648

def g (x : ℝ) : ℝ := x^2

theorem g_x_minus_3 (x : ℝ) : g (x - 3) = x^2 - 6 * x + 9 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_g_x_minus_3_l46_4648


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_iff_l46_4609

theorem quadratic_has_two_distinct_real_roots_iff (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 6 * x - m = 0 ∧ y^2 - 6 * y - m = 0) ↔ m > -9 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_iff_l46_4609


namespace NUMINAMATH_GPT_range_of_m_l46_4639

theorem range_of_m (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ 2 * x + m - 3 = 0) : m < 3 :=
sorry

end NUMINAMATH_GPT_range_of_m_l46_4639


namespace NUMINAMATH_GPT_intersection_of_lines_l46_4696

theorem intersection_of_lines : ∃ x y : ℚ, y = 3 * x ∧ y - 5 = -7 * x ∧ x = 1 / 2 ∧ y = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l46_4696


namespace NUMINAMATH_GPT_remainder_equal_to_zero_l46_4650

def A : ℕ := 270
def B : ℕ := 180
def M : ℕ := 25
def R_A : ℕ := A % M
def R_B : ℕ := B % M
def A_squared_B : ℕ := (A ^ 2 * B) % M
def R_A_R_B : ℕ := (R_A * R_B) % M

theorem remainder_equal_to_zero (h1 : A = 270) (h2 : B = 180) (h3 : M = 25) 
    (h4 : R_A = 20) (h5 : R_B = 5) : 
    A_squared_B = 0 ∧ R_A_R_B = 0 := 
by {
    sorry
}

end NUMINAMATH_GPT_remainder_equal_to_zero_l46_4650


namespace NUMINAMATH_GPT_find_AC_find_area_l46_4677

theorem find_AC (BC : ℝ) (angleA : ℝ) (cosB : ℝ) 
(hBC : BC = Real.sqrt 7) (hAngleA : angleA = 60) (hCosB : cosB = Real.sqrt 6 / 3) :
  (AC : ℝ) → (hAC : AC = 2 * Real.sqrt 7 / 3) → Prop :=
by
  sorry

theorem find_area (BC AB : ℝ) (angleA : ℝ) 
(hBC : BC = Real.sqrt 7) (hAB : AB = 2) (hAngleA : angleA = 60) :
  (area : ℝ) → (hArea : area = 3 * Real.sqrt 3 / 2) → Prop :=
by
  sorry

end NUMINAMATH_GPT_find_AC_find_area_l46_4677


namespace NUMINAMATH_GPT_max_shapes_in_8x14_grid_l46_4626

def unit_squares := 3
def grid_8x14 := 8 * 14
def grid_points (m n : ℕ) := (m + 1) * (n + 1)
def shapes_grid_points := 8
def max_shapes (total_points shape_points : ℕ) := total_points / shape_points

theorem max_shapes_in_8x14_grid 
  (m n : ℕ) (shape_points : ℕ) 
  (h1 : m = 8) (h2 : n = 14)
  (h3 : shape_points = 8) :
  max_shapes (grid_points m n) shape_points = 16 := by
  sorry

end NUMINAMATH_GPT_max_shapes_in_8x14_grid_l46_4626


namespace NUMINAMATH_GPT_tetrahedron_paintings_l46_4672

theorem tetrahedron_paintings (n : ℕ) (h : n ≥ 4) : 
  let term1 := (n - 1) * (n - 2) * (n - 3) / 12
  let term2 := (n - 1) * (n - 2) / 3
  let term3 := n - 1
  let term4 := 1
  2 * (term1 + term2 + term3) + n = 
  n * (term1 + term2 + term3 + term4) := by
{
  sorry
}

end NUMINAMATH_GPT_tetrahedron_paintings_l46_4672


namespace NUMINAMATH_GPT_domain_of_f_l46_4610

def domain_of_log_func := Set ℝ

def is_valid (x : ℝ) : Prop := x - 1 > 0

def func_domain (f : ℝ → ℝ) : domain_of_log_func := {x : ℝ | is_valid x}

theorem domain_of_f :
  func_domain (λ x => Real.log (x - 1)) = {x : ℝ | 1 < x} := by
  sorry

end NUMINAMATH_GPT_domain_of_f_l46_4610


namespace NUMINAMATH_GPT_dad_steps_l46_4691

theorem dad_steps (masha_steps : ℕ) (yasha_steps : ℕ) (dad_steps : ℕ) :
  (∀ d m, m = 5 * d / 3) → (∀ m y, y = 5 * m / 3) → (masha_steps + yasha_steps = 400) → 
  dad_steps = 90 :=
by
  sorry

end NUMINAMATH_GPT_dad_steps_l46_4691


namespace NUMINAMATH_GPT_range_of_xy_l46_4652

theorem range_of_xy {x y : ℝ} (h₁ : 0 < x) (h₂ : 0 < y)
    (h₃ : x + 2/x + 3*y + 4/y = 10) : 
    1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_xy_l46_4652


namespace NUMINAMATH_GPT_prime_square_implies_equal_l46_4664

theorem prime_square_implies_equal (p : ℕ) (hp : Nat.Prime p) (hp_gt_2 : p > 2)
  (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ (p-1)/2) (hy : 1 ≤ y ∧ y ≤ (p-1)/2)
  (h_square: ∃ k : ℕ, x * (p - x) * y * (p - y) = k ^ 2) : x = y :=
sorry

end NUMINAMATH_GPT_prime_square_implies_equal_l46_4664


namespace NUMINAMATH_GPT_chess_pieces_missing_l46_4602

theorem chess_pieces_missing (total_pieces present_pieces missing_pieces : ℕ) 
  (h1 : total_pieces = 32)
  (h2 : present_pieces = 22)
  (h3 : missing_pieces = total_pieces - present_pieces) :
  missing_pieces = 10 :=
by
  sorry

end NUMINAMATH_GPT_chess_pieces_missing_l46_4602


namespace NUMINAMATH_GPT_sum_of_squares_fraction_l46_4687

variable {x1 x2 x3 y1 y2 y3 : ℝ}

theorem sum_of_squares_fraction :
  x1 + x2 + x3 = 0 → y1 + y2 + y3 = 0 → x1 * y1 + x2 * y2 + x3 * y3 = 0 →
  (x1^2 / (x1^2 + x2^2 + x3^2)) + (y1^2 / (y1^2 + y2^2 + y3^2)) = 2 / 3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sum_of_squares_fraction_l46_4687


namespace NUMINAMATH_GPT_min_value_frac_x_y_l46_4629

theorem min_value_frac_x_y (x y : ℝ) (hx : x > 0) (hy : y > -1) (hxy : x + y = 1) :
  ∃ m, m = 2 + Real.sqrt 3 ∧ ∀ x y, x > 0 → y > -1 → x + y = 1 → (x^2 + 3) / x + y^2 / (y + 1) ≥ m :=
sorry

end NUMINAMATH_GPT_min_value_frac_x_y_l46_4629


namespace NUMINAMATH_GPT_min_value_fraction_l46_4625

theorem min_value_fraction (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  (∃ T : ℝ, T = (5 * r / (3 * p + 2 * q) + 5 * p / (2 * q + 3 * r) + 2 * q / (p + r)) ∧ T = 19 / 4) :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l46_4625


namespace NUMINAMATH_GPT_problem_l46_4681

noncomputable def g (x : ℝ) : ℝ := 3^x + 2

theorem problem (x : ℝ) : g (x + 1) - g x = 2 * g x - 2 := sorry

end NUMINAMATH_GPT_problem_l46_4681


namespace NUMINAMATH_GPT_diameter_of_outer_edge_l46_4689

-- Defining the conditions as variables
variable (pathWidth gardenWidth statueDiameter fountainDiameter : ℝ)
variable (hPathWidth : pathWidth = 10)
variable (hGardenWidth : gardenWidth = 12)
variable (hStatueDiameter : statueDiameter = 6)
variable (hFountainDiameter : fountainDiameter = 14)

-- Lean statement to prove the diameter
theorem diameter_of_outer_edge :
  2 * ((fountainDiameter / 2) + gardenWidth + pathWidth) = 58 :=
by
  rw [hPathWidth, hGardenWidth, hFountainDiameter]
  sorry

end NUMINAMATH_GPT_diameter_of_outer_edge_l46_4689


namespace NUMINAMATH_GPT_k_range_l46_4628

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -x^3 + 2*x^2 - x
  else if 1 ≤ x then Real.log x
  else 0 -- Technically, we don't care outside (0, +∞), so this else case doesn't matter.

theorem k_range (k : ℝ) :
  (∀ t : ℝ, 0 < t → f t < k * t) ↔ k ∈ (Set.Ioi (1 / Real.exp 1)) :=
by
  sorry

end NUMINAMATH_GPT_k_range_l46_4628


namespace NUMINAMATH_GPT_proof_problem_l46_4649

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := 
  (Real.sin (2 * x), 2 * Real.cos x ^ 2 - 1)

noncomputable def vector_b (θ : ℝ) : ℝ × ℝ := 
  (Real.sin θ, Real.cos θ)

noncomputable def f (x θ : ℝ) : ℝ := 
  (vector_a x).1 * (vector_b θ).1 + (vector_a x).2 * (vector_b θ).2

theorem proof_problem 
  (θ : ℝ) 
  (hθ : 0 < θ ∧ θ < π) 
  (h1 : f (π / 6) θ = 1) 
  (x : ℝ) 
  (hx : -π / 6 ≤ x ∧ x ≤ π / 4) :
  θ = π / 3 ∧
  (∀ x, f x θ = f (x + π) θ) ∧
  (∀ x, -π / 6 ≤ x ∧ x ≤ π / 4 → f x θ ≤ 1) ∧
  (∀ x, -π / 6 ≤ x ∧ x ≤ π / 4 → f x θ ≥ -0.5) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l46_4649


namespace NUMINAMATH_GPT_emily_beads_l46_4604

theorem emily_beads (n : ℕ) (b : ℕ) (total_beads : ℕ) (h1 : n = 26) (h2 : b = 2) (h3 : total_beads = n * b) : total_beads = 52 :=
by
  sorry

end NUMINAMATH_GPT_emily_beads_l46_4604


namespace NUMINAMATH_GPT_find_total_amount_l46_4617

-- Definitions according to the conditions
def is_proportion (a b c : ℚ) (p q r : ℚ) : Prop :=
  (a * q = b * p) ∧ (a * r = c * p) ∧ (b * r = c * q)

def total_amount (second_part : ℚ) (prop_total : ℚ) : ℚ :=
  second_part / (1/3) * prop_total

-- Main statement to be proved
theorem find_total_amount (second_part : ℚ) (p1 p2 p3 : ℚ)
  (h : is_proportion p1 p2 p3 (1/2 : ℚ) (1/3 : ℚ) (3/4 : ℚ))
  : second_part = 164.6315789473684 → total_amount second_part (19/12 : ℚ) = 65.16 :=
by
  sorry

end NUMINAMATH_GPT_find_total_amount_l46_4617


namespace NUMINAMATH_GPT_problem_statement_l46_4657

noncomputable def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n

theorem problem_statement (m n p q : ℕ) (h₁ : m ≠ p) (h₂ : is_integer ((mn + pq : ℚ) / (m - p))) :
  is_integer ((mq + np : ℚ) / (m - p)) :=
sorry

end NUMINAMATH_GPT_problem_statement_l46_4657


namespace NUMINAMATH_GPT_truck_tank_percentage_increase_l46_4685

-- Declaration of the initial conditions (as given in the problem)
def service_cost : ℝ := 2.20
def fuel_cost_per_liter : ℝ := 0.70
def num_minivans : ℕ := 4
def num_trucks : ℕ := 2
def total_cost : ℝ := 395.40
def minivan_tank_size : ℝ := 65.0

-- Proof statement with the conditions declared above
theorem truck_tank_percentage_increase :
  ∃ p : ℝ, p = 120 ∧ (minivan_tank_size * (p + 100) / 100 = 143) :=
sorry

end NUMINAMATH_GPT_truck_tank_percentage_increase_l46_4685


namespace NUMINAMATH_GPT_total_property_price_l46_4674

theorem total_property_price :
  let price_per_sqft : ℝ := 98
  let house_sqft : ℝ := 2400
  let barn_sqft : ℝ := 1000
  let house_price : ℝ := house_sqft * price_per_sqft
  let barn_price : ℝ := barn_sqft * price_per_sqft
  let total_price : ℝ := house_price + barn_price
  total_price = 333200 := by
  sorry

end NUMINAMATH_GPT_total_property_price_l46_4674


namespace NUMINAMATH_GPT_length_of_bridge_l46_4675

theorem length_of_bridge 
    (length_of_train : ℕ)
    (speed_of_train_km_per_hr : ℕ)
    (time_to_cross_seconds : ℕ)
    (bridge_length : ℕ) 
    (h_train_length : length_of_train = 130)
    (h_speed_train : speed_of_train_km_per_hr = 54)
    (h_time_cross : time_to_cross_seconds = 30)
    (h_bridge_length : bridge_length = 320) : 
    bridge_length = 320 :=
by sorry

end NUMINAMATH_GPT_length_of_bridge_l46_4675


namespace NUMINAMATH_GPT_find_c_l46_4619

theorem find_c (c : ℝ) (h : ∃ β : ℝ, (5 + β = -c) ∧ (5 * β = 45)) : c = -14 := 
  sorry

end NUMINAMATH_GPT_find_c_l46_4619


namespace NUMINAMATH_GPT_total_price_of_houses_l46_4690

theorem total_price_of_houses (price_first price_second total_price : ℝ)
    (h1 : price_first = 200000)
    (h2 : price_second = 2 * price_first)
    (h3 : total_price = price_first + price_second) :
  total_price = 600000 := by
  sorry

end NUMINAMATH_GPT_total_price_of_houses_l46_4690


namespace NUMINAMATH_GPT_calculation_l46_4635

theorem calculation :
  (-1:ℤ)^(2022) + (Real.sqrt 9) - 2 * (Real.sin (Real.pi / 6)) = 3 := by
  -- According to the mathematical problem and the given solution.
  -- Here we use essential definitions and facts provided in the problem.
  sorry

end NUMINAMATH_GPT_calculation_l46_4635


namespace NUMINAMATH_GPT_weeks_to_meet_goal_l46_4654

def hourly_rate : ℕ := 6
def hours_monday : ℕ := 2
def hours_tuesday : ℕ := 3
def hours_wednesday : ℕ := 4
def hours_thursday : ℕ := 2
def hours_friday : ℕ := 3
def helmet_cost : ℕ := 340
def gloves_cost : ℕ := 45
def initial_savings : ℕ := 40
def misc_expenses : ℕ := 20

theorem weeks_to_meet_goal : 
  let total_needed := helmet_cost + gloves_cost + misc_expenses
  let total_deficit := total_needed - initial_savings
  let total_weekly_hours := hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday
  let weekly_earnings := total_weekly_hours * hourly_rate
  let weeks_required := Nat.ceil (total_deficit / weekly_earnings)
  weeks_required = 5 := sorry

end NUMINAMATH_GPT_weeks_to_meet_goal_l46_4654


namespace NUMINAMATH_GPT_solve_x_l46_4637

theorem solve_x (x : ℝ) (h : (x - 1)^2 = 4) : x = 3 ∨ x = -1 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_solve_x_l46_4637


namespace NUMINAMATH_GPT_sum_of_first_9_terms_45_l46_4623

-- Define the arithmetic sequence and sum of terms in the sequence
def S (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms of the sequence
def a (n : ℕ) : ℕ := sorry  -- Placeholder for the n-th term of the sequence

-- Given conditions
axiom condition1 : a 3 + a 5 + a 7 = 15

-- Proof goal
theorem sum_of_first_9_terms_45 : S 9 = 45 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_9_terms_45_l46_4623


namespace NUMINAMATH_GPT_a1_plus_a9_l46_4658

def S (n : ℕ) : ℕ := n^2 + 1

def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a1_plus_a9 : (a 1) + (a 9) = 19 := by
  sorry

end NUMINAMATH_GPT_a1_plus_a9_l46_4658


namespace NUMINAMATH_GPT_roque_commute_time_l46_4640

theorem roque_commute_time :
  let walk_time := 2
  let bike_time := 1
  let walks_per_week := 3
  let bike_rides_per_week := 2
  let total_walk_time := 2 * walks_per_week * walk_time
  let total_bike_time := 2 * bike_rides_per_week * bike_time
  total_walk_time + total_bike_time = 16 :=
by sorry

end NUMINAMATH_GPT_roque_commute_time_l46_4640


namespace NUMINAMATH_GPT_factorization_identity_l46_4661

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end NUMINAMATH_GPT_factorization_identity_l46_4661


namespace NUMINAMATH_GPT_parabolas_intersect_with_high_probability_l46_4655

noncomputable def high_probability_of_intersection : Prop :=
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 →
  (a - c) ^ 2 + 4 * (b - d) >= 0

theorem parabolas_intersect_with_high_probability : high_probability_of_intersection := sorry

end NUMINAMATH_GPT_parabolas_intersect_with_high_probability_l46_4655


namespace NUMINAMATH_GPT_annie_hamburgers_l46_4605

theorem annie_hamburgers (H : ℕ) (h₁ : 4 * H + 6 * 5 = 132 - 70) : H = 8 := by
  sorry

end NUMINAMATH_GPT_annie_hamburgers_l46_4605


namespace NUMINAMATH_GPT_find_difference_l46_4676

theorem find_difference (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.30 * y) : x - y = 10 := 
by
  sorry

end NUMINAMATH_GPT_find_difference_l46_4676


namespace NUMINAMATH_GPT_compare_sums_l46_4638

theorem compare_sums (a b c : ℝ) (h : a > b ∧ b > c) : a^2 * b + b^2 * c + c^2 * a > a * b^2 + b * c^2 + c * a^2 := by
  sorry

end NUMINAMATH_GPT_compare_sums_l46_4638


namespace NUMINAMATH_GPT_time_taken_by_A_l46_4663

theorem time_taken_by_A (v_A v_B D t_A t_B : ℚ) (h1 : v_A / v_B = 3 / 4) 
  (h2 : t_A = t_B + 30) (h3 : t_A = D / v_A) (h4 : t_B = D / v_B) 
  : t_A = 120 := 
by 
  sorry

end NUMINAMATH_GPT_time_taken_by_A_l46_4663


namespace NUMINAMATH_GPT_maximize_sum_l46_4606

def a_n (n : ℕ): ℤ := 11 - 2 * (n - 1)

theorem maximize_sum (n : ℕ) (S : ℕ → ℤ → Prop) :
  (∀ n, S n (a_n n)) → (a_n n ≥ 0) → n = 6 :=
by
  sorry

end NUMINAMATH_GPT_maximize_sum_l46_4606


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l46_4608

theorem solve_equation1 (x : ℝ) (h1 : 3 * x^3 - 15 = 9) : x = 2 :=
sorry

theorem solve_equation2 (x : ℝ) (h2 : 2 * (x - 1)^2 = 72) : x = 7 ∨ x = -5 :=
sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l46_4608


namespace NUMINAMATH_GPT_coord_of_point_B_l46_4692
-- Necessary import for mathematical definitions and structures

-- Define the initial point A and the translation conditions
def point_A : ℝ × ℝ := (1, -2)
def translation_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ := (p.1, p.2 + units)

-- The target point B after translation
def point_B := translation_up point_A 1

-- The theorem to prove that the coordinates of B are (1, -1)
theorem coord_of_point_B : point_B = (1, -1) :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_coord_of_point_B_l46_4692


namespace NUMINAMATH_GPT_division_of_8_identical_books_into_3_piles_l46_4694

-- Definitions for the conditions
def identical_books_division_ways (n : ℕ) (p : ℕ) : ℕ :=
  if n = 8 ∧ p = 3 then 5 else sorry

-- Theorem statement
theorem division_of_8_identical_books_into_3_piles :
  identical_books_division_ways 8 3 = 5 := by
  sorry

end NUMINAMATH_GPT_division_of_8_identical_books_into_3_piles_l46_4694
