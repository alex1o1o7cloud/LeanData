import Mathlib

namespace sum_of_angles_l2315_231595

-- Definitions of acute, right, and obtuse angles
def is_acute (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def is_right (θ : ℝ) : Prop := θ = 90
def is_obtuse (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

-- The main statement we want to prove
theorem sum_of_angles :
  (∀ (α β : ℝ), is_acute α ∧ is_acute β → is_acute (α + β) ∨ is_right (α + β) ∨ is_obtuse (α + β)) ∧
  (∀ (α β : ℝ), is_acute α ∧ is_right β → is_obtuse (α + β)) :=
by sorry

end sum_of_angles_l2315_231595


namespace polygon_with_120_degree_interior_angle_has_6_sides_l2315_231517

theorem polygon_with_120_degree_interior_angle_has_6_sides (n : ℕ) (h : ∀ i, 1 ≤ i ∧ i ≤ n → (sum_interior_angles : ℕ) = (n-2) * 180 / n ∧ (each_angle : ℕ) = 120) : n = 6 :=
by
  sorry

end polygon_with_120_degree_interior_angle_has_6_sides_l2315_231517


namespace employee_total_correct_l2315_231525

variable (total_employees : ℝ)
variable (percentage_female : ℝ)
variable (percentage_male_literate : ℝ)
variable (percentage_total_literate : ℝ)
variable (number_female_literate : ℝ)
variable (percentage_male : ℝ := 1 - percentage_female)

variables (E : ℝ) (CF : ℝ) (M : ℝ) (total_literate : ℝ)

theorem employee_total_correct :
  percentage_female = 0.60 ∧
  percentage_male_literate = 0.50 ∧
  percentage_total_literate = 0.62 ∧
  number_female_literate = 546 ∧
  (total_employees = 1300) :=
by
  -- Change these variables according to the context or find a way to prove this
  let total_employees := 1300
  have Cf := number_female_literate / (percentage_female * total_employees)
  have total_male := percentage_male * total_employees
  have male_literate := percentage_male_literate * total_male
  have total_literate := percentage_total_literate * total_employees

  -- We replace "proof statements" with sorry here
  sorry

end employee_total_correct_l2315_231525


namespace a_eq_b_pow_n_l2315_231598

theorem a_eq_b_pow_n (a b n : ℕ) (h : ∀ k : ℕ, k ≠ b → (a - k^n) % (b - k) = 0) : a = b^n :=
sorry

end a_eq_b_pow_n_l2315_231598


namespace perimeter_shaded_region_l2315_231558

theorem perimeter_shaded_region (r: ℝ) (circumference: ℝ) (h1: circumference = 36) (h2: {x // x = 3 * (circumference / 6)}) : x = 18 :=
by
  sorry

end perimeter_shaded_region_l2315_231558


namespace line_b_y_intercept_l2315_231556

variable (b : ℝ → ℝ)
variable (x y : ℝ)

-- Line b is parallel to y = -3x + 6
def is_parallel (b : ℝ → ℝ) : Prop :=
  ∃ m c, (∀ x, b x = m * x + c) ∧ m = -3

-- Line b passes through the point (3, -2)
def passes_through_point (b : ℝ → ℝ) : Prop :=
  b 3 = -2

-- The y-intercept of line b
def y_intercept (b : ℝ → ℝ) : ℝ :=
  b 0

theorem line_b_y_intercept (h1 : is_parallel b) (h2 : passes_through_point b) : y_intercept b = 7 :=
sorry

end line_b_y_intercept_l2315_231556


namespace second_person_percentage_of_Deshaun_l2315_231586

variable (days : ℕ) (books_read_by_Deshaun : ℕ) (pages_per_book : ℕ) (pages_per_day_by_second_person : ℕ)

theorem second_person_percentage_of_Deshaun :
  days = 80 →
  books_read_by_Deshaun = 60 →
  pages_per_book = 320 →
  pages_per_day_by_second_person = 180 →
  ((pages_per_day_by_second_person * days) / (books_read_by_Deshaun * pages_per_book) * 100) = 75 := 
by
  intros days_eq books_eq pages_eq second_pages_eq
  rw [days_eq, books_eq, pages_eq, second_pages_eq]
  simp
  sorry

end second_person_percentage_of_Deshaun_l2315_231586


namespace divisible_by_77_l2315_231597

theorem divisible_by_77 (n : ℤ) : ∃ k : ℤ, n^18 - n^12 - n^8 + n^2 = 77 * k :=
by
  sorry

end divisible_by_77_l2315_231597


namespace correct_calculation_is_c_l2315_231579

theorem correct_calculation_is_c (a b : ℕ) :
  (2 * a ^ 2 * b) ^ 3 = 8 * a ^ 6 * b ^ 3 := 
sorry

end correct_calculation_is_c_l2315_231579


namespace invisible_trees_in_square_l2315_231544

theorem invisible_trees_in_square (n : ℕ) : 
  ∃ (N M : ℕ), ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → 
  Nat.gcd (N + i) (M + j) ≠ 1 :=
by
  sorry

end invisible_trees_in_square_l2315_231544


namespace marble_arrangement_mod_l2315_231568

def num_ways_arrange_marbles (m : ℕ) : ℕ := Nat.choose (m + 3) 3

theorem marble_arrangement_mod (N : ℕ) (m : ℕ) (h1: m = 11) (h2: N = num_ways_arrange_marbles m): 
  N % 1000 = 35 := by
  sorry

end marble_arrangement_mod_l2315_231568


namespace tangent_line_eq_l2315_231537

noncomputable def equation_of_tangent_line (x y : ℝ) : Prop := 
  ∃ k : ℝ, (y = k * (x - 2) + 2) ∧ 2 * x + y - 6 = 0

theorem tangent_line_eq :
  ∀ (x y : ℝ), 
    (y = 2 / (x - 1)) ∧ (∃ (a b : ℝ), (a, b) = (1, 4)) ->
    equation_of_tangent_line x y :=
by
  sorry

end tangent_line_eq_l2315_231537


namespace expression_for_3_diamond_2_l2315_231527

variable {a b : ℝ}

def diamond (a b : ℝ) : ℝ := 2 * a - 3 * b + a * b

theorem expression_for_3_diamond_2 (a : ℝ) :
  3 * diamond a 2 = 12 * a - 18 :=
by
  sorry

end expression_for_3_diamond_2_l2315_231527


namespace isosceles_right_triangle_area_l2315_231502

theorem isosceles_right_triangle_area (hypotenuse : ℝ) (leg_length : ℝ) (area : ℝ) :
  hypotenuse = 6 * Real.sqrt 2 →
  leg_length = hypotenuse / Real.sqrt 2 →
  area = (1 / 2) * leg_length * leg_length →
  area = 18 :=
by
  -- problem states hypotenuse is 6*sqrt(2)
  intro h₁
  -- calculus leg length from hypotenuse / sqrt(2)
  intro h₂
  -- area of the triangle from legs
  intro h₃
  -- state the desired result
  sorry

end isosceles_right_triangle_area_l2315_231502


namespace total_tiles_covering_floor_l2315_231524

-- Let n be the width of the rectangle (in tiles)
-- The length would then be 2n (in tiles)
-- The total number of tiles that lie on both diagonals is given as 39

theorem total_tiles_covering_floor (n : ℕ) (H : 2 * n + 1 = 39) : 2 * n^2 = 722 :=
by sorry

end total_tiles_covering_floor_l2315_231524


namespace measure_of_angle_BCD_l2315_231526

-- Define angles and sides as given in the problem
variables (α β : ℝ)

-- Conditions: angles and side equalities
axiom angle_ABD_eq_BDC : α = β
axiom angle_DAB_eq_80 : α = 80
axiom side_AB_eq_AD : ∀ AB AD : ℝ, AB = AD
axiom side_DB_eq_DC : ∀ DB DC : ℝ, DB = DC

-- Prove that the measure of angle BCD is 65 degrees
theorem measure_of_angle_BCD : β = 65 :=
sorry

end measure_of_angle_BCD_l2315_231526


namespace simplify_expression_l2315_231539

theorem simplify_expression (x y : ℝ) : 
    3 * x - 5 * (2 - x + y) + 4 * (1 - x - 2 * y) - 6 * (2 + 3 * x - y) = -14 * x - 7 * y - 18 := 
by 
    sorry

end simplify_expression_l2315_231539


namespace smallest_integer_value_of_m_l2315_231575

def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem smallest_integer_value_of_m :
  ∀ m : ℤ, (x^2 + 4 * x - m = 0) ∧ has_two_distinct_real_roots 1 4 (-m : ℝ) → m ≥ -3 :=
by
  intro m h
  sorry

end smallest_integer_value_of_m_l2315_231575


namespace total_squares_after_removals_l2315_231562

/-- 
Prove that the total number of squares of various sizes on a 5x5 grid,
after removing two 1x1 squares, is 55.
-/
theorem total_squares_after_removals (total_squares_in_5x5_grid: ℕ) (removed_squares: ℕ) : 
  (total_squares_in_5x5_grid = 25 + 16 + 9 + 4 + 1) →
  (removed_squares = 2) →
  (total_squares_in_5x5_grid - removed_squares = 55) :=
sorry

end total_squares_after_removals_l2315_231562


namespace domain_of_function_l2315_231529

theorem domain_of_function :
  {x : ℝ | x < -1 ∨ 4 ≤ x} = {x : ℝ | (x^2 - 7*x + 12) / (x^2 - 2*x - 3) ≥ 0} \ {3} :=
by
  sorry

end domain_of_function_l2315_231529


namespace three_divides_n_of_invertible_diff_l2315_231547

theorem three_divides_n_of_invertible_diff
  (n : ℕ)
  (A B : Matrix (Fin n) (Fin n) ℝ)
  (h1 : A * A + B * B = A * B)
  (h2 : Invertible (B * A - A * B)) :
  3 ∣ n :=
sorry

end three_divides_n_of_invertible_diff_l2315_231547


namespace sin_alpha_sol_cos_2alpha_pi4_sol_l2315_231532

open Real

-- Define the main problem conditions
def cond1 (α : ℝ) := sin (α + π / 3) + sin α = 9 * sqrt 7 / 14
def range (α : ℝ) := 0 < α ∧ α < π / 3

-- Define the statement for the first problem
theorem sin_alpha_sol (α : ℝ) (h1 : cond1 α) (h2 : range α) : sin α = 2 * sqrt 7 / 7 := 
sorry

-- Define the statement for the second problem
theorem cos_2alpha_pi4_sol (α : ℝ) (h1 : cond1 α) (h2 : range α) (h3 : sin α = 2 * sqrt 7 / 7) : 
  cos (2 * α - π / 4) = (4 * sqrt 6 - sqrt 2) / 14 := 
sorry

end sin_alpha_sol_cos_2alpha_pi4_sol_l2315_231532


namespace quadratic_minimum_value_l2315_231588

theorem quadratic_minimum_value :
  ∀ (x : ℝ), (x - 1)^2 + 2 ≥ 2 :=
by
  sorry

end quadratic_minimum_value_l2315_231588


namespace after_2_pow_2009_days_is_monday_l2315_231589

-- Define the current day as Thursday
def today := "Thursday"

-- Define the modulo operation for calculating days of the week
def day_of_week_after (days : ℕ) : ℕ :=
  days % 7

-- Define the exponent in question
def exponent := 2009

-- Since today is Thursday, which we can represent as 4 (considering Sunday as 0, Monday as 1, ..., Saturday as 6)
def today_as_num := 4

-- Calculate the day after 2^2009 days
def future_day := (today_as_num + day_of_week_after (2 ^ exponent)) % 7

-- Prove that the future_day is 1 (Monday)
theorem after_2_pow_2009_days_is_monday : future_day = 1 := by
  sorry

end after_2_pow_2009_days_is_monday_l2315_231589


namespace complement_union_l2315_231528

open Set

variable (U : Set ℕ) (M N : Set ℕ)

theorem complement_union (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 2}) (hN : N = {3, 4}) :
  compl (M ∪ N) = {x | x ∉ M ∪ N} ∧ {5} = {x | x ∈ U ∧ x ∉ M ∪ N} :=
by
  sorry

end complement_union_l2315_231528


namespace max_value_of_symmetric_function_l2315_231523

def f (x a b : ℝ) := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function 
  (a b : ℝ)
  (symmetric : ∀ t : ℝ, f (-2 + t) a b = f (-2 - t) a b) :
  ∃ M : ℝ, M = 16 ∧ ∀ x : ℝ, f x a b ≤ M :=
by
  use 16
  sorry

end max_value_of_symmetric_function_l2315_231523


namespace fx_le_1_l2315_231552

-- Statement
theorem fx_le_1 (x : ℝ) (h : x > 0) : (1 + Real.log x) / x ≤ 1 := 
sorry

end fx_le_1_l2315_231552


namespace sandwich_bread_consumption_l2315_231513

theorem sandwich_bread_consumption :
  ∀ (num_bread_per_sandwich : ℕ),
  (2 * num_bread_per_sandwich) + num_bread_per_sandwich = 6 →
  num_bread_per_sandwich = 2 := by
    intros num_bread_per_sandwich h
    sorry

end sandwich_bread_consumption_l2315_231513


namespace set_of_positive_reals_l2315_231518

theorem set_of_positive_reals (S : Set ℝ) (h1 : ∀ x, x ∈ S → 0 < x)
  (h2 : ∀ a b, a ∈ S → b ∈ S → a + b ∈ S)
  (h3 : ∀ (a b : ℝ), 0 < a → a ≤ b → ∃ c d, a ≤ c ∧ c ≤ d ∧ d ≤ b ∧ ∀ x, c ≤ x ∧ x ≤ d → x ∈ S) :
  S = {x : ℝ | 0 < x} :=
sorry

end set_of_positive_reals_l2315_231518


namespace sufficient_but_not_necessary_condition_l2315_231577

theorem sufficient_but_not_necessary_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x > 0 ∧ y > 0 → (x / y + y / x ≥ 2)) ∧ ¬((x / y + y / x ≥ 2) → (x > 0 ∧ y > 0)) :=
sorry

end sufficient_but_not_necessary_condition_l2315_231577


namespace john_fan_usage_per_day_l2315_231520

theorem john_fan_usage_per_day
  (power : ℕ := 75) -- fan's power in watts
  (energy_per_month_kwh : ℕ := 18) -- energy consumption per month in kWh
  (days_in_month : ℕ := 30) -- number of days in a month
  : (energy_per_month_kwh * 1000) / power / days_in_month = 8 := 
by
  sorry

end john_fan_usage_per_day_l2315_231520


namespace find_x_of_series_eq_15_l2315_231535

noncomputable def infinite_series (x : ℝ) : ℝ :=
  5 + (5 + x) / 3 + (5 + 2 * x) / 3^2 + (5 + 3 * x) / 3^3 + ∑' n, (5 + (n + 1) * x) / 3 ^ (n + 1)

theorem find_x_of_series_eq_15 (x : ℝ) (h : infinite_series x = 15) : x = 10 :=
sorry

end find_x_of_series_eq_15_l2315_231535


namespace time_for_A_alone_l2315_231508

variable {W : ℝ}
variable {x : ℝ}

theorem time_for_A_alone (h1 : (W / x) + (W / 24) = W / 12) : x = 24 := 
sorry

end time_for_A_alone_l2315_231508


namespace find_percentage_l2315_231522

noncomputable def percentage (P : ℝ) : Prop :=
  (P / 100) * 1265 / 6 = 354.2

theorem find_percentage : ∃ (P : ℝ), percentage P ∧ P = 168 :=
by
  sorry

end find_percentage_l2315_231522


namespace determine_b_l2315_231549

theorem determine_b (b : ℚ) (x y : ℚ) (h1 : x = -3) (h2 : y = 4) (h3 : 2 * b * x + (b + 2) * y = b + 6) :
  b = 2 / 3 := 
sorry

end determine_b_l2315_231549


namespace baby_panda_daily_bamboo_intake_l2315_231573

theorem baby_panda_daily_bamboo_intake :
  ∀ (adult_bamboo_per_day baby_bamboo_per_day total_bamboo_per_week : ℕ),
    adult_bamboo_per_day = 138 →
    total_bamboo_per_week = 1316 →
    total_bamboo_per_week = 7 * adult_bamboo_per_day + 7 * baby_bamboo_per_day →
    baby_bamboo_per_day = 50 :=
by
  intros adult_bamboo_per_day baby_bamboo_per_day total_bamboo_per_week h1 h2 h3
  sorry

end baby_panda_daily_bamboo_intake_l2315_231573


namespace cost_price_computer_table_l2315_231546

variable (C : ℝ) -- Cost price of the computer table
variable (S : ℝ) -- Selling price of the computer table

-- Conditions based on the problem
axiom h1 : S = 1.10 * C
axiom h2 : S = 8800

-- The theorem to be proven
theorem cost_price_computer_table : C = 8000 :=
by
  -- Proof will go here
  sorry

end cost_price_computer_table_l2315_231546


namespace unique_function_solution_l2315_231576

theorem unique_function_solution (f : ℝ → ℝ) (h₁ : ∀ x : ℝ, x ≥ 1 → f x ≥ 1)
  (h₂ : ∀ x : ℝ, x ≥ 1 → f x ≤ 2 * (x + 1))
  (h₃ : ∀ x : ℝ, x ≥ 1 → f (x + 1) = (f x)^2/x - 1/x) :
  ∀ x : ℝ, x ≥ 1 → f x = x + 1 :=
by
  intro x hx
  sorry

end unique_function_solution_l2315_231576


namespace initial_population_l2315_231507

theorem initial_population (P : ℝ) (h : 0.78435 * P = 4500) : P = 5738 := 
by 
  sorry

end initial_population_l2315_231507


namespace angle_QPR_l2315_231551

theorem angle_QPR (PQ QR PR RS : Real) (angle_PQR angle_PRS : Real) 
  (h1 : PQ = QR) (h2 : PR = RS) (h3 : angle_PQR = 50) (h4 : angle_PRS = 100) : 
  ∃ angle_QPR : Real, angle_QPR = 25 :=
by
  -- We are proving that angle_QPR is 25 given the conditions.
  sorry

end angle_QPR_l2315_231551


namespace modulus_zero_l2315_231567

/-- Given positive integers k and α such that 10k - α is also a positive integer, 
prove that the remainder when 8^(10k + α) + 6^(10k - α) - 7^(10k - α) - 2^(10k + α) is divided by 11 is 0. -/
theorem modulus_zero {k α : ℕ} (h₁ : 0 < k) (h₂ : 0 < α) (h₃ : 0 < 10 * k - α) :
  (8 ^ (10 * k + α) + 6 ^ (10 * k - α) - 7 ^ (10 * k - α) - 2 ^ (10 * k + α)) % 11 = 0 :=
by
  sorry

end modulus_zero_l2315_231567


namespace farmer_apples_after_giving_l2315_231516

-- Define the initial number of apples and the number of apples given to the neighbor
def initial_apples : ℕ := 127
def given_apples : ℕ := 88

-- Define the expected number of apples after giving some away
def remaining_apples : ℕ := 39

-- Formulate the proof problem
theorem farmer_apples_after_giving : initial_apples - given_apples = remaining_apples := by
  sorry

end farmer_apples_after_giving_l2315_231516


namespace mixture_kerosene_l2315_231572

theorem mixture_kerosene (x : ℝ) (h₁ : 0.25 * x + 1.2 = 0.27 * (x + 4)) : x = 6 :=
sorry

end mixture_kerosene_l2315_231572


namespace expand_product_l2315_231583

noncomputable def question_expression (x : ℝ) := -3 * (2 * x + 4) * (x - 7)
noncomputable def correct_answer (x : ℝ) := -6 * x^2 + 30 * x + 84

theorem expand_product (x : ℝ) : question_expression x = correct_answer x := 
by sorry

end expand_product_l2315_231583


namespace new_area_is_726_l2315_231591

variable (l w : ℝ)
variable (h_area : l * w = 576)
variable (l' : ℝ := 1.20 * l)
variable (w' : ℝ := 1.05 * w)

theorem new_area_is_726 : l' * w' = 726 := by
  sorry

end new_area_is_726_l2315_231591


namespace foci_distance_l2315_231503

open Real

-- Defining parameters and conditions
variables (a : ℝ) (b : ℝ) (c : ℝ)
  (F1 F2 A B : ℝ × ℝ) -- Foci and points A, B
  (hyp_cavity : c ^ 2 = a ^ 2 + b ^ 2)
  (perimeters_eq : dist A B = 3 * a ∧ dist A F1 + dist B F1 = dist B F1 + dist B F2 + dist F1 F2)
  (distance_property : dist A F2 - dist A F1 = 2 * a)
  (c_value : c = 2 * a) -- Derived from hyperbolic definition
  
-- Main theorem to prove the distance between foci
theorem foci_distance : dist F1 F2 = 4 * a :=
  sorry

end foci_distance_l2315_231503


namespace d_is_multiple_of_4_c_minus_d_is_multiple_of_4_c_minus_d_is_multiple_of_2_l2315_231514

variable (c d : ℕ)

-- Conditions: c is a multiple of 4 and d is a multiple of 8
def is_multiple_of_4 (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k
def is_multiple_of_8 (n : ℕ) : Prop := ∃ k : ℕ, n = 8 * k

-- Statements to prove:

-- A. d is a multiple of 4
theorem d_is_multiple_of_4 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : is_multiple_of_4 d :=
sorry

-- B. c - d is a multiple of 4
theorem c_minus_d_is_multiple_of_4 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : is_multiple_of_4 (c - d) :=
sorry

-- D. c - d is a multiple of 2
theorem c_minus_d_is_multiple_of_2 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : ∃ k : ℕ, c - d = 2 * k :=
sorry

end d_is_multiple_of_4_c_minus_d_is_multiple_of_4_c_minus_d_is_multiple_of_2_l2315_231514


namespace rent_expense_calculation_l2315_231542

variable (S : ℝ)
variable (saved_amount : ℝ := 2160)
variable (milk_expense : ℝ := 1500)
variable (groceries_expense : ℝ := 4500)
variable (education_expense : ℝ := 2500)
variable (petrol_expense : ℝ := 2000)
variable (misc_expense : ℝ := 3940)
variable (salary_percent_saved : ℝ := 0.10)

theorem rent_expense_calculation 
  (h1 : salary_percent_saved * S = saved_amount) :
  S = 21600 → 
  0.90 * S - (milk_expense + groceries_expense + education_expense + petrol_expense + misc_expense) = 5000 :=
by
  sorry

end rent_expense_calculation_l2315_231542


namespace ninety_eight_times_ninety_eight_l2315_231563

theorem ninety_eight_times_ninety_eight : 98 * 98 = 9604 := 
by
  sorry

end ninety_eight_times_ninety_eight_l2315_231563


namespace total_apples_correct_l2315_231510

def craig_initial := 20.5
def judy_initial := 11.25
def dwayne_initial := 17.85
def eugene_to_craig := 7.15
def craig_to_dwayne := 3.5 / 2
def judy_to_sally := judy_initial / 2

def craig_final := craig_initial + eugene_to_craig - craig_to_dwayne
def dwayne_final := dwayne_initial + craig_to_dwayne
def judy_final := judy_initial - judy_to_sally
def sally_final := judy_to_sally

def total_apples := craig_final + judy_final + dwayne_final + sally_final

theorem total_apples_correct : total_apples = 56.75 := by
  -- skipping proof
  sorry

end total_apples_correct_l2315_231510


namespace anton_stationary_escalator_steps_l2315_231553

theorem anton_stationary_escalator_steps
  (N : ℕ)
  (H1 : N = 30)
  (H2 : 5 * N = 150) :
  (stationary_steps : ℕ) = 50 :=
by
  sorry

end anton_stationary_escalator_steps_l2315_231553


namespace percentage_of_products_by_m1_l2315_231543

theorem percentage_of_products_by_m1
  (x : ℝ)
  (h1 : 30 / 100 > 0)
  (h2 : 3 / 100 > 0)
  (h3 : 1 / 100 > 0)
  (h4 : 7 / 100 > 0)
  (h_total_defective : 
    0.036 = 
      (0.03 * x / 100) + 
      (0.01 * 30 / 100) + 
      (0.07 * (100 - x - 30) / 100)) :
  x = 40 :=
by
  sorry

end percentage_of_products_by_m1_l2315_231543


namespace irrational_product_rational_l2315_231550

-- Definitions of irrational and rational for clarity
def irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q
def rational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Statement of the problem in Lean 4
theorem irrational_product_rational (a b : ℕ) (ha : irrational (Real.sqrt a)) (hb : irrational (Real.sqrt b)) :
  rational ((Real.sqrt a + Real.sqrt b) * (Real.sqrt a - Real.sqrt b)) :=
by
  sorry

end irrational_product_rational_l2315_231550


namespace common_point_graphs_l2315_231596

theorem common_point_graphs 
  (a b c d : ℝ)
  (h1 : ∃ x : ℝ, 2*a + (1 / (x - b)) = 2*c + (1 / (x - d))) :
  ∃ x : ℝ, 2*b + (1 / (x - a)) = 2*d + (1 / (x - c)) :=
by
  sorry

end common_point_graphs_l2315_231596


namespace audrey_lost_pieces_l2315_231564

theorem audrey_lost_pieces {total_pieces_on_board : ℕ} {thomas_lost : ℕ} {initial_pieces_each : ℕ} (h1 : total_pieces_on_board = 21) (h2 : thomas_lost = 5) (h3 : initial_pieces_each = 16) :
  (initial_pieces_each - (total_pieces_on_board - (initial_pieces_each - thomas_lost))) = 6 :=
by
  sorry

end audrey_lost_pieces_l2315_231564


namespace canned_boxes_equation_l2315_231512

theorem canned_boxes_equation (x : ℕ) (h₁: x ≤ 300) :
  2 * 14 * x = 32 * (300 - x) :=
by
sorry

end canned_boxes_equation_l2315_231512


namespace simplify_expression_l2315_231593

theorem simplify_expression (x : ℝ) : (2 * x)^3 + (3 * x) * (x^2) = 11 * x^3 := 
  sorry

end simplify_expression_l2315_231593


namespace arctan_tan_expression_l2315_231580

noncomputable def tan (x : ℝ) : ℝ := sorry
noncomputable def arctan (x : ℝ) : ℝ := sorry

theorem arctan_tan_expression :
  arctan (tan 65 - 2 * tan 40) = 25 := sorry

end arctan_tan_expression_l2315_231580


namespace g_value_at_2002_l2315_231569

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Conditions given in the problem
axiom f_one : f 1 = 1
axiom f_inequality_5 : ∀ x : ℝ, f (x + 5) ≥ f x + 5
axiom f_inequality_1 : ∀ x : ℝ, f (x + 1) ≤ f x + 1

-- Define the function g based on f
def g (x : ℝ) : ℝ := f x + 1 - x

-- The goal is to prove that g 2002 = 1
theorem g_value_at_2002 : g 2002 = 1 :=
sorry

end g_value_at_2002_l2315_231569


namespace find_m_n_l2315_231540

theorem find_m_n (m n : ℕ) (h_pos : m > 0 ∧ n > 0) (h_gcd : m.gcd n = 1) (h_div : (m^3 + n^3) ∣ (m^2 + 20 * m * n + n^2)) :
  (m, n) ∈ [(1, 2), (2, 1), (2, 3), (3, 2), (1, 5), (5, 1)] :=
by
  sorry

end find_m_n_l2315_231540


namespace solution_of_system_l2315_231587

variable (x y : ℝ) 

def equation1 (x y : ℝ) : Prop := 3 * |x| + 5 * y + 9 = 0
def equation2 (x y : ℝ) : Prop := 2 * x - |y| - 7 = 0

theorem solution_of_system : ∃ y : ℝ, equation1 0 y ∧ equation2 0 y := by
  sorry

end solution_of_system_l2315_231587


namespace delta_gj_l2315_231581

def vj := 120
def total := 770
def gj := total - vj

theorem delta_gj : gj - 5 * vj = 50 := by
  sorry

end delta_gj_l2315_231581


namespace proof_problem_l2315_231505

open Real

theorem proof_problem :
  (∀ x : ℕ, x * x = 16 → sqrt (x * x) = 4) →
  (∀ x : ℕ, x * x = 16 → sqrt (x * x) = 16/4) →
  (∀ x : ℤ, abs x = 4 → abs (-4) = 4) →
  (∀ x : ℤ, x^2 = 16 → (-4)^2 = 16) →
  (- sqrt 16 = -4) := 
by 
  simp
  sorry

end proof_problem_l2315_231505


namespace first_place_points_is_eleven_l2315_231554

/-
Conditions:
1. Points are awarded as follows: first place = x points, second place = 7 points, third place = 5 points, fourth place = 2 points.
2. John participated 7 times in the competition.
3. John finished in each of the top four positions at least once.
4. The product of all the points John received was 38500.
Theorem: The first place winner receives 11 points.
-/

noncomputable def archery_first_place_points (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), -- number of times John finished first, second, third, fourth respectively
    a + b + c + d = 7 ∧ -- condition 2, John participated 7 times
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ -- condition 3, John finished each position at least once
    x ^ a * 7 ^ b * 5 ^ c * 2 ^ d = 38500 -- condition 4, product of all points John received

theorem first_place_points_is_eleven : archery_first_place_points 11 :=
  sorry

end first_place_points_is_eleven_l2315_231554


namespace divisible_by_6_implies_divisible_by_2_not_divisible_by_2_implies_not_divisible_by_6_equivalence_of_propositions_l2315_231566

theorem divisible_by_6_implies_divisible_by_2 :
  ∀ (n : ℤ), (6 ∣ n) → (2 ∣ n) :=
by sorry

theorem not_divisible_by_2_implies_not_divisible_by_6 :
  ∀ (n : ℤ), ¬ (2 ∣ n) → ¬ (6 ∣ n) :=
by sorry

theorem equivalence_of_propositions :
  (∀ (n : ℤ), (6 ∣ n) → (2 ∣ n)) ↔ (∀ (n : ℤ), ¬ (2 ∣ n) → ¬ (6 ∣ n)) :=
by sorry


end divisible_by_6_implies_divisible_by_2_not_divisible_by_2_implies_not_divisible_by_6_equivalence_of_propositions_l2315_231566


namespace arithmetic_sequence_l2315_231574

theorem arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 1 + a 2 + a 3 = 32) 
  (h2 : a 11 + a 12 + a 13 = 118) 
  (arith_seq : ∀ n, a (n + 1) = a n + d) : 
  a 4 + a 10 = 50 :=
by 
  sorry

end arithmetic_sequence_l2315_231574


namespace guo_can_pay_exactly_l2315_231565

theorem guo_can_pay_exactly (
  x y z : ℕ
) (h : 10 * x + 20 * y + 50 * z = 20000) : ∃ a b c : ℕ, a + 2 * b + 5 * c = 1000 :=
sorry

end guo_can_pay_exactly_l2315_231565


namespace root_range_m_l2315_231559

theorem root_range_m (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * m * x + 4 = 0 → (x > 1 ∧ ∃ y : ℝ, y < 1 ∧ y^2 - 2 * m * y + 4 = 0)
  ∨ (x < 1 ∧ ∃ y : ℝ, y > 1 ∧ y^2 - 2 * m * y + 4 = 0))
  → m > 5 / 2 := 
sorry

end root_range_m_l2315_231559


namespace even_composite_sum_consecutive_odd_numbers_l2315_231582

theorem even_composite_sum_consecutive_odd_numbers (a k : ℤ) : ∃ (n m : ℤ), n = 2 * k ∧ m = n * (2 * a + n) ∧ m % 4 = 0 :=
by
  sorry

end even_composite_sum_consecutive_odd_numbers_l2315_231582


namespace squirrel_burrow_has_44_walnuts_l2315_231531

def boy_squirrel_initial := 30
def boy_squirrel_gathered := 20
def boy_squirrel_dropped := 4
def boy_squirrel_hid := 8
-- "Forgets where he hid 3 of them" does not affect the main burrow

def girl_squirrel_brought := 15
def girl_squirrel_ate := 5
def girl_squirrel_gave := 4
def girl_squirrel_lost_playing := 3
def girl_squirrel_knocked := 2

def third_squirrel_gathered := 10
def third_squirrel_dropped := 1
def third_squirrel_hid := 3
def third_squirrel_returned := 6 -- Given directly instead of as a formula step; 9-3=6
def third_squirrel_gave := 1 -- Given directly as a friend

def final_walnuts := boy_squirrel_initial + boy_squirrel_gathered
                    - boy_squirrel_dropped - boy_squirrel_hid
                    + girl_squirrel_brought - girl_squirrel_ate
                    - girl_squirrel_gave - girl_squirrel_lost_playing
                    - girl_squirrel_knocked + third_squirrel_returned

theorem squirrel_burrow_has_44_walnuts :
  final_walnuts = 44 :=
by
  sorry

end squirrel_burrow_has_44_walnuts_l2315_231531


namespace rosa_called_last_week_l2315_231570

noncomputable def total_pages_called : ℝ := 18.8
noncomputable def pages_called_this_week : ℝ := 8.6
noncomputable def pages_called_last_week : ℝ := total_pages_called - pages_called_this_week

theorem rosa_called_last_week :
  pages_called_last_week = 10.2 :=
by
  sorry

end rosa_called_last_week_l2315_231570


namespace Ruth_school_hours_l2315_231511

theorem Ruth_school_hours (d : ℝ) :
  0.25 * 5 * d = 10 → d = 8 :=
by
  sorry

end Ruth_school_hours_l2315_231511


namespace age_sum_in_5_years_l2315_231500

variable (MikeAge MomAge : ℕ)
variable (h1 : MikeAge = MomAge - 30)
variable (h2 : MikeAge + MomAge = 70)

theorem age_sum_in_5_years (h1 : MikeAge = MomAge - 30) (h2 : MikeAge + MomAge = 70) :
  (MikeAge + 5) + (MomAge + 5) = 80 := by
  sorry

end age_sum_in_5_years_l2315_231500


namespace ellipse_line_intersection_l2315_231519

-- Definitions of the conditions in the Lean 4 language
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1

def midpoint_eq (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2 = 1) ∧ (y1 + y2 = -2)

-- The problem statement
theorem ellipse_line_intersection :
  (∃ (l : ℝ → ℝ → Prop),
  (∀ x1 y1 x2 y2 : ℝ, ellipse_eq x1 y1 → ellipse_eq x2 y2 → midpoint_eq x1 y1 x2 y2 →
     l x1 y1 ∧ l x2 y2) ∧
  (∀ x y : ℝ, l x y → (x - 4 * y - 9 / 2 = 0))) :=
sorry

end ellipse_line_intersection_l2315_231519


namespace find_k_l2315_231504

-- Define the set A using a condition on the quadratic equation
def A (k : ℝ) : Set ℝ := {x | k * x ^ 2 + 4 * x + 4 = 0}

-- Define the condition for the set A to have exactly one element
def has_exactly_one_element (k : ℝ) : Prop :=
  ∃ x : ℝ, A k = {x}

-- The problem statement is to find the value of k for which A has exactly one element
theorem find_k : ∃ k : ℝ, has_exactly_one_element k ∧ k = 1 :=
by
  simp [has_exactly_one_element, A]
  sorry

end find_k_l2315_231504


namespace identical_cubes_probability_l2315_231530

/-- Statement of the problem -/
theorem identical_cubes_probability :
  let total_ways := 3^8 * 3^8  -- Total ways to paint two cubes
  let identical_ways := 3 + 72 + 252 + 504  -- Ways for identical appearance after rotation
  (identical_ways : ℝ) / total_ways = 1 / 51814 :=
by
  sorry

end identical_cubes_probability_l2315_231530


namespace reflected_ray_eqn_l2315_231584

theorem reflected_ray_eqn (P : ℝ × ℝ)
  (incident_ray : ∀ x : ℝ, P.2 = 2 * P.1 + 1)
  (reflection_line : P.2 = P.1) :
  P.1 - 2 * P.2 - 1 = 0 :=
sorry

end reflected_ray_eqn_l2315_231584


namespace factorize_expression_l2315_231592

theorem factorize_expression (R : Type*) [CommRing R] (m n : R) : 
  m^2 * n - n = n * (m + 1) * (m - 1) := 
sorry

end factorize_expression_l2315_231592


namespace johns_final_push_time_l2315_231548

theorem johns_final_push_time :
  ∃ t : ℝ, t = 17 / 4.2 := 
by
  sorry

end johns_final_push_time_l2315_231548


namespace intersecting_lines_ratio_l2315_231585

theorem intersecting_lines_ratio (k1 k2 a : ℝ) (h1 : k1 * a + 4 = 0) (h2 : k2 * a - 2 = 0) : k1 / k2 = -2 :=
by
    sorry

end intersecting_lines_ratio_l2315_231585


namespace quadratic_has_two_distinct_real_roots_l2315_231538

theorem quadratic_has_two_distinct_real_roots : 
  ∃ α β : ℝ, (α ≠ β) ∧ (2 * α^2 - 3 * α + 1 = 0) ∧ (2 * β^2 - 3 * β + 1 = 0) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l2315_231538


namespace swim_distance_l2315_231533

theorem swim_distance 
  (v c d : ℝ)
  (h₁ : c = 2)
  (h₂ : (d / (v + c) = 5))
  (h₃ : (25 / (v - c) = 5)) :
  d = 45 :=
by
  sorry

end swim_distance_l2315_231533


namespace cos_value_l2315_231541

variable (α : ℝ)

theorem cos_value (h : Real.sin (Real.pi / 6 + α) = 1 / 3) : Real.cos (2 * Real.pi / 3 - 2 * α) = -7 / 9 :=
by
  sorry

end cos_value_l2315_231541


namespace quadrants_I_and_II_l2315_231501

-- Define the conditions
def condition1 (x y : ℝ) : Prop := y > 3 * x
def condition2 (x y : ℝ) : Prop := y > 6 - x^2

-- Prove that any point satisfying the conditions lies in Quadrant I or II
theorem quadrants_I_and_II (x y : ℝ) (h1 : y > 3 * x) (h2 : y > 6 - x^2) : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by
  -- The proof steps are omitted
  sorry

end quadrants_I_and_II_l2315_231501


namespace constant_term_binomial_expansion_l2315_231578

theorem constant_term_binomial_expansion (n : ℕ) (hn : n = 6) :
  (2 : ℤ) * (x : ℝ) - (1 : ℤ) / (2 : ℝ) / (x : ℝ) ^ n == -20 := by
  sorry

end constant_term_binomial_expansion_l2315_231578


namespace quadratic_inequality_solution_set_l2315_231506

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3 * x + 2 ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end quadratic_inequality_solution_set_l2315_231506


namespace problem_l2315_231515

noncomputable def d : ℝ := -8.63

theorem problem :
  let floor_d := ⌊d⌋
  let frac_d := d - floor_d
  (3 * floor_d^2 + 20 * floor_d - 67 = 0) ∧
  (4 * frac_d^2 - 15 * frac_d + 5 = 0) → 
  d = -8.63 :=
by {
  sorry
}

end problem_l2315_231515


namespace factorize_expression_l2315_231509

theorem factorize_expression (x y : ℝ) : 2 * x^2 * y - 8 * y = 2 * y * (x + 2) * (x - 2) :=
  sorry

end factorize_expression_l2315_231509


namespace third_vs_second_plant_relationship_l2315_231521

-- Define the constants based on the conditions
def first_plant_tomatoes := 24
def second_plant_tomatoes := 12 + 5  -- Half of 24 plus 5
def total_tomatoes := 60

-- Define the production of the third plant based on the total number of tomatoes
def third_plant_tomatoes := total_tomatoes - (first_plant_tomatoes + second_plant_tomatoes)

-- Define the relationship to be proved
theorem third_vs_second_plant_relationship : 
  third_plant_tomatoes = second_plant_tomatoes + 2 :=
by
  -- Proof not provided, adding sorry to skip
  sorry

end third_vs_second_plant_relationship_l2315_231521


namespace tan_alpha_equals_one_l2315_231557

theorem tan_alpha_equals_one (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos (α + β) = Real.sin (α - β))
  : Real.tan α = 1 := 
by
  sorry

end tan_alpha_equals_one_l2315_231557


namespace joes_bid_l2315_231536

/--
Nelly tells her daughter she outbid her rival Joe by paying $2000 more than thrice his bid.
Nelly got the painting for $482,000. Prove that Joe's bid was $160,000.
-/
theorem joes_bid (J : ℝ) (h1 : 482000 = 3 * J + 2000) : J = 160000 :=
by
  sorry

end joes_bid_l2315_231536


namespace find_C_l2315_231599

noncomputable def A : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x < a}
def isSolutionC (C : Set ℝ) : Prop := C = {2, 3}

theorem find_C : ∃ C : Set ℝ, isSolutionC C ∧ ∀ a, (A ∪ B a = A) ↔ a ∈ C :=
by
  sorry

end find_C_l2315_231599


namespace condition_holds_iff_b_eq_10_l2315_231560

-- Define xn based on given conditions in the problem
def x_n (b : ℕ) (n : ℕ) : ℕ :=
  if b > 5 then
    b^(2*n) + b^n + 3*b - 5
  else
    0

-- State the main theorem to be proven in Lean
theorem condition_holds_iff_b_eq_10 :
  ∀ (b : ℕ), (b > 5) ↔ ∃ M : ℕ, ∀ n : ℕ, n > M → ∃ k : ℕ, x_n b n = k^2 := sorry

end condition_holds_iff_b_eq_10_l2315_231560


namespace subtraction_divisible_l2315_231571

theorem subtraction_divisible (n m d : ℕ) (h1 : n = 13603) (h2 : m = 31) (h3 : d = 13572) : 
  (n - m) % d = 0 := by
  sorry

end subtraction_divisible_l2315_231571


namespace division_and_multiply_l2315_231590

theorem division_and_multiply :
  (-128) / (-16) * 5 = 40 := 
by
  sorry

end division_and_multiply_l2315_231590


namespace final_cost_correct_l2315_231555

def dozen_cost : ℝ := 18
def num_dozen : ℝ := 2.5
def discount_rate : ℝ := 0.15

def cost_before_discount : ℝ := num_dozen * dozen_cost
def discount_amount : ℝ := discount_rate * cost_before_discount

def final_cost : ℝ := cost_before_discount - discount_amount

theorem final_cost_correct : final_cost = 38.25 := by
  -- The proof would go here, but we just provide the statement.
  sorry

end final_cost_correct_l2315_231555


namespace stock_index_approximation_l2315_231561

noncomputable def stock_index_after_days (initial_index : ℝ) (daily_increase : ℝ) (days : ℕ) : ℝ :=
  initial_index * (1 + daily_increase / 100) ^ (days - 1)

theorem stock_index_approximation :
  let initial_index := 2
  let daily_increase := 0.02
  let days := 100
  abs (stock_index_after_days initial_index daily_increase days - 2.041) < 0.001 :=
by
  sorry

end stock_index_approximation_l2315_231561


namespace handrail_length_nearest_tenth_l2315_231545

noncomputable def handrail_length (rise : ℝ) (turn_degree : ℝ) (radius : ℝ) : ℝ :=
  let arc_length := (turn_degree / 360) * (2 * Real.pi * radius)
  Real.sqrt (rise^2 + arc_length^2)

theorem handrail_length_nearest_tenth
  (h_rise : rise = 12)
  (h_turn_degree : turn_degree = 180)
  (h_radius : radius = 3) : handrail_length rise turn_degree radius = 13.1 :=
  by
  sorry

end handrail_length_nearest_tenth_l2315_231545


namespace find_cost_price_l2315_231534

theorem find_cost_price 
  (C : ℝ)
  (h1 : 1.10 * C + 110 = 1.15 * C)
  : C = 2200 :=
sorry

end find_cost_price_l2315_231534


namespace adjusted_smallest_part_proof_l2315_231594

theorem adjusted_smallest_part_proof : 
  ∀ (x : ℝ), 14 * x = 100 → x + 12 = 19 + 1 / 7 := 
by
  sorry

end adjusted_smallest_part_proof_l2315_231594
