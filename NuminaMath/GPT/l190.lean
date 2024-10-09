import Mathlib

namespace intersection_of_A_and_B_l190_19021

noncomputable def A := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
noncomputable def B := {x : ℝ | x < 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} :=
sorry

end intersection_of_A_and_B_l190_19021


namespace increasing_interval_implication_l190_19059

theorem increasing_interval_implication (a : ℝ) :
  (∀ x ∈ Set.Ioo (1 / 2) 2, (1 / x + 2 * a * x > 0)) → a > -1 / 8 :=
by
  intro h
  sorry

end increasing_interval_implication_l190_19059


namespace congruent_semicircles_ratio_l190_19030

theorem congruent_semicircles_ratio (N : ℕ) (r : ℝ) (hN : N > 0) 
    (A : ℝ) (B : ℝ) (hA : A = (N * π * r^2) / 2)
    (hB : B = (π * N^2 * r^2) / 2 - (N * π * r^2) / 2)
    (h_ratio : A / B = 1 / 9) : 
    N = 10 :=
by
  -- The proof will be filled in here.
  sorry

end congruent_semicircles_ratio_l190_19030


namespace ham_and_bread_percentage_l190_19054

-- Defining the different costs as constants
def cost_of_bread : ℝ := 50
def cost_of_ham : ℝ := 150
def cost_of_cake : ℝ := 200

-- Defining the total cost of the items
def total_cost : ℝ := cost_of_bread + cost_of_ham + cost_of_cake

-- Defining the combined cost of ham and bread
def combined_cost_ham_and_bread : ℝ := cost_of_bread + cost_of_ham

-- The theorem stating that the combined cost of ham and bread is 50% of the total cost
theorem ham_and_bread_percentage : (combined_cost_ham_and_bread / total_cost) * 100 = 50 := by
  sorry  -- Proof to be provided

end ham_and_bread_percentage_l190_19054


namespace find_q_l190_19075

variable (p q : ℝ)

theorem find_q (h1 : p > 1) (h2 : q > 1) (h3 : 1/p + 1/q = 1) (h4 : p * q = 9) : 
  q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end find_q_l190_19075


namespace average_annual_percent_change_l190_19086

-- Define the initial and final population, and the time period
def initial_population : ℕ := 175000
def final_population : ℕ := 297500
def decade_years : ℕ := 10

-- Define the theorem to find the resulting average percent change per year
theorem average_annual_percent_change
    (P₀ : ℕ := initial_population)
    (P₁₀ : ℕ := final_population)
    (years : ℕ := decade_years) :
    ((P₁₀ - P₀ : ℝ) / P₀ * 100) / years = 7 := by
        sorry

end average_annual_percent_change_l190_19086


namespace solve_real_eq_l190_19043

theorem solve_real_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (x = (23 + Real.sqrt 145) / 6 ∨ x = (23 - Real.sqrt 145) / 6) ↔
  ((x ^ 3 - 3 * x ^ 2) / (x ^ 2 - 4) + 2 * x = -16) :=
by sorry

end solve_real_eq_l190_19043


namespace problem_statement_l190_19080

variable {x y z : ℝ}

theorem problem_statement (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
    (h₁ : x^2 - y^2 = y * z) (h₂ : y^2 - z^2 = x * z) : 
    x^2 - z^2 = x * y := 
by
  sorry

end problem_statement_l190_19080


namespace property1_property2_l190_19077

/-- Given sequence a_n defined as a_n = 3(n^2 + n) + 7 -/
def a (n : ℕ) : ℕ := 3 * (n^2 + n) + 7

/-- Property 1: Out of any five consecutive terms in the sequence, only one term is divisible by 5. -/
theorem property1 (n : ℕ) : (∃ k : ℕ, a (5 * k + 2) % 5 = 0) ∧ (∀ k : ℕ, ∀ r : ℕ, r ≠ 2 → a (5 * k + r) % 5 ≠ 0) :=
by
  sorry

/-- Property 2: None of the terms in this sequence is a cube of an integer. -/
theorem property2 (n : ℕ) : ¬(∃ t : ℕ, a n = t^3) :=
by
  sorry

end property1_property2_l190_19077


namespace add_words_to_meet_requirement_l190_19093

-- Definitions required by the problem
def yvonne_words : ℕ := 400
def janna_extra_words : ℕ := 150
def words_removed : ℕ := 20
def requirement : ℕ := 1000

-- Derived values based on the conditions
def janna_words : ℕ := yvonne_words + janna_extra_words
def initial_words : ℕ := yvonne_words + janna_words
def words_after_removal : ℕ := initial_words - words_removed
def words_added : ℕ := 2 * words_removed
def total_words_after_editing : ℕ := words_after_removal + words_added
def words_to_add : ℕ := requirement - total_words_after_editing

-- The theorem to prove
theorem add_words_to_meet_requirement : words_to_add = 30 := by
  sorry

end add_words_to_meet_requirement_l190_19093


namespace find_remainder_l190_19013

theorem find_remainder (a : ℕ) :
  (a ^ 100) % 73 = 2 ∧ (a ^ 101) % 73 = 69 → a % 73 = 71 :=
by
  sorry

end find_remainder_l190_19013


namespace intersection_with_y_axis_l190_19057

theorem intersection_with_y_axis :
  ∃ (x y : ℝ), x = 0 ∧ y = 5 * x - 6 ∧ (x, y) = (0, -6) := 
sorry

end intersection_with_y_axis_l190_19057


namespace standard_deviation_is_2point5_l190_19094

noncomputable def mean : ℝ := 17.5
noncomputable def given_value : ℝ := 12.5

theorem standard_deviation_is_2point5 :
  ∀ (σ : ℝ), mean - 2 * σ = given_value → σ = 2.5 := by
  sorry

end standard_deviation_is_2point5_l190_19094


namespace range_of_a_l190_19068

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 3 * x + a < 0) ∧ (∀ x : ℝ, 2 * x + 7 > 4 * x - 1) ∧ (∀ x : ℝ, x < 0) → a = 0 := 
by sorry

end range_of_a_l190_19068


namespace larger_number_of_two_with_conditions_l190_19038

theorem larger_number_of_two_with_conditions (x y : ℕ) (h1 : x * y = 30) (h2 : x + y = 13) : max x y = 10 :=
by
  sorry

end larger_number_of_two_with_conditions_l190_19038


namespace age_difference_l190_19066

variable (A : ℕ) -- Albert's age
variable (B : ℕ) -- Albert's brother's age
variable (F : ℕ) -- Father's age
variable (M : ℕ) -- Mother's age

def age_conditions : Prop :=
  (B = A - 2) ∧ (F = A + 48) ∧ (M = B + 46)

theorem age_difference (h : age_conditions A B F M) : F - M = 4 :=
by
  sorry

end age_difference_l190_19066


namespace range_of_a_fixed_point_l190_19022

open Function

def f (x a : ℝ) := x^3 - a * x

theorem range_of_a (a : ℝ) (h1 : 0 < a) : 0 < a ∧ a ≤ 3 ↔ ∀ x ≥ 1, 3 * x^2 - a > 0 :=
sorry

theorem fixed_point (a x0 : ℝ) (h_a : 0 < a) (h_b : a ≤ 3)
  (h1 : x0 ≥ 1) (h2 : f x0 a ≥ 1) (h3 : f (f x0 a) a = x0) (strict_incr : ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f x a < f y a) :
  f x0 a = x0 :=
sorry

end range_of_a_fixed_point_l190_19022


namespace measure_of_one_interior_angle_of_regular_octagon_l190_19048

theorem measure_of_one_interior_angle_of_regular_octagon :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 := 
by
  sorry

end measure_of_one_interior_angle_of_regular_octagon_l190_19048


namespace child_ticket_cost_l190_19042

theorem child_ticket_cost 
    (x : ℝ)
    (adult_ticket_cost : ℝ := 5)
    (total_sales : ℝ := 178)
    (total_tickets_sold : ℝ := 42)
    (child_tickets_sold : ℝ := 16) 
    (adult_tickets_sold : ℝ := total_tickets_sold - child_tickets_sold)
    (total_adult_sales : ℝ := adult_tickets_sold * adult_ticket_cost)
    (sales_equation : total_adult_sales + child_tickets_sold * x = total_sales) : 
    x = 3 :=
by
  sorry

end child_ticket_cost_l190_19042


namespace complement_B_intersection_A_complement_B_l190_19052

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | x < 0}
noncomputable def B : Set ℝ := {x | x > 1}

theorem complement_B :
  (U \ B) = {x | x ≤ 1} := by
  sorry

theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x | x < 0} := by
  sorry

end complement_B_intersection_A_complement_B_l190_19052


namespace milk_water_equal_l190_19062

theorem milk_water_equal (a : ℕ) :
  let glass_a_initial := a
  let glass_b_initial := a
  let mixture_in_a := glass_a_initial + 1
  let milk_portion_in_a := 1 / mixture_in_a
  let water_portion_in_a := glass_a_initial / mixture_in_a
  let water_in_milk_glass := water_portion_in_a
  let milk_in_water_glass := milk_portion_in_a
  water_in_milk_glass = milk_in_water_glass := by
  sorry

end milk_water_equal_l190_19062


namespace sum_of_x_and_y_l190_19045

theorem sum_of_x_and_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
    (hx : ∃ (a : ℕ), 720 * x = a^2)
    (hy : ∃ (b : ℕ), 720 * y = b^4) :
    x + y = 1130 :=
sorry

end sum_of_x_and_y_l190_19045


namespace exists_unique_line_prime_x_intercept_positive_y_intercept_l190_19097

/-- There is exactly one line with x-intercept that is a prime number less than 10 and y-intercept that is a positive integer not equal to 5, which passes through the point (5, 4) -/
theorem exists_unique_line_prime_x_intercept_positive_y_intercept (x_intercept : ℕ) (hx : Nat.Prime x_intercept) (hx_lt_10 : x_intercept < 10) (y_intercept : ℕ) (hy_pos : y_intercept > 0) (hy_ne_5 : y_intercept ≠ 5) :
  (∃ (a b : ℕ), a = x_intercept ∧ b = y_intercept ∧ (∀ p q : ℕ, p = 5 ∧ q = 4 → (p / a) + (q / b) = 1)) :=
sorry

end exists_unique_line_prime_x_intercept_positive_y_intercept_l190_19097


namespace remainder_when_55_times_57_divided_by_8_l190_19098

theorem remainder_when_55_times_57_divided_by_8 :
  (55 * 57) % 8 = 7 :=
by
  -- Insert the proof here
  sorry

end remainder_when_55_times_57_divided_by_8_l190_19098


namespace mortgage_loan_amount_l190_19073

/-- Given the initial payment is 1,800,000 rubles and it represents 30% of the property cost C, 
    prove that the mortgage loan amount is 4,200,000 rubles. -/
theorem mortgage_loan_amount (C : ℝ) (h : 0.3 * C = 1800000) : C - 1800000 = 4200000 :=
by
  sorry

end mortgage_loan_amount_l190_19073


namespace find_remainder_l190_19096

def mod_condition : Prop :=
  (764251 % 31 = 5) ∧
  (1095223 % 31 = 6) ∧
  (1487719 % 31 = 1) ∧
  (263311 % 31 = 0) ∧
  (12097 % 31 = 25) ∧
  (16817 % 31 = 26) ∧
  (23431 % 31 = 0) ∧
  (305643 % 31 = 20)

theorem find_remainder (h : mod_condition) : 
  ((764251 * 1095223 * 1487719 + 263311) * (12097 * 16817 * 23431 - 305643)) % 31 = 8 := 
by
  sorry

end find_remainder_l190_19096


namespace remainder_when_15_plus_y_div_31_l190_19067

theorem remainder_when_15_plus_y_div_31 (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (15 + y) % 31 = 24 :=
by
  sorry

end remainder_when_15_plus_y_div_31_l190_19067


namespace cracker_calories_l190_19040

theorem cracker_calories (cc : ℕ) (hc1 : ∀ (n : ℕ), n = 50 → cc = 50) (hc2 : ∀ (n : ℕ), n = 7 → 7 * 50 = 350) (hc3 : ∀ (n : ℕ), n = 10 * cc → 10 * cc = 10 * cc) (hc4 : 350 + 10 * cc = 500) : cc = 15 :=
by
  sorry

end cracker_calories_l190_19040


namespace solution_set_of_inequality_eq_l190_19091

noncomputable def inequality_solution_set : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem solution_set_of_inequality_eq :
  {x : ℝ | (2 * x) / (x - 1) < 1} = inequality_solution_set := by
  sorry

end solution_set_of_inequality_eq_l190_19091


namespace books_purchased_with_grant_l190_19019

-- Define the conditions
def total_books_now : ℕ := 8582
def books_before_grant : ℕ := 5935

-- State the theorem that we need to prove
theorem books_purchased_with_grant : (total_books_now - books_before_grant) = 2647 := by
  sorry

end books_purchased_with_grant_l190_19019


namespace hyperbola_center_l190_19036

theorem hyperbola_center :
  ∃ c : ℝ × ℝ, (c = (3, 4) ∧ ∀ x y : ℝ, 9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0 ↔ (x - 3)^2 / 4 - (y - 4)^2 / 1 = 1) :=
sorry

end hyperbola_center_l190_19036


namespace qualified_flour_l190_19006

-- Define the acceptable weight range
def acceptable_range (w : ℝ) : Prop :=
  24.75 ≤ w ∧ w ≤ 25.25

-- Define the weight options
def optionA : ℝ := 24.70
def optionB : ℝ := 24.80
def optionC : ℝ := 25.30
def optionD : ℝ := 25.51

-- The statement to be proved
theorem qualified_flour : acceptable_range optionB ∧ ¬acceptable_range optionA ∧ ¬acceptable_range optionC ∧ ¬acceptable_range optionD :=
by
  sorry

end qualified_flour_l190_19006


namespace circle_parametric_eq_l190_19007

theorem circle_parametric_eq 
  (a b r : ℝ) (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi):
  (∃ (x y : ℝ), (x = r * Real.cos θ + a ∧ y = r * Real.sin θ + b)) ↔ 
  (∃ (x' y' : ℝ), (x' = r * Real.cos θ ∧ y' = r * Real.sin θ)) :=
sorry

end circle_parametric_eq_l190_19007


namespace necessary_but_not_sufficient_l190_19072

theorem necessary_but_not_sufficient (x : ℝ) :
  (x - 1) * (x + 2) = 0 → (x = 1 ∨ x = -2) ∧ (x = 1 → (x - 1) * (x + 2) = 0) ∧ ¬((x - 1) * (x + 2) = 0 ↔ x = 1) :=
by
  sorry

end necessary_but_not_sufficient_l190_19072


namespace complete_square_l190_19039

theorem complete_square :
  (∀ x: ℝ, 2 * x^2 - 4 * x + 1 = 2 * (x - 1)^2 - 1) := 
by
  intro x
  sorry

end complete_square_l190_19039


namespace operation_hash_12_6_l190_19056

axiom operation_hash (r s : ℝ) : ℝ

-- Conditions
axiom condition_1 : ∀ r : ℝ, operation_hash r 0 = r
axiom condition_2 : ∀ r s : ℝ, operation_hash r s = operation_hash s r
axiom condition_3 : ∀ r s : ℝ, operation_hash (r + 2) s = (operation_hash r s) + 2 * s + 2

-- Proof statement
theorem operation_hash_12_6 : operation_hash 12 6 = 168 :=
by
  sorry

end operation_hash_12_6_l190_19056


namespace max_cables_191_l190_19004

/-- 
  There are 30 employees: 20 with brand A computers and 10 with brand B computers.
  Cables can only connect a brand A computer to a brand B computer.
  Employees can communicate with each other if their computers are directly connected by a cable 
  or by relaying messages through a series of connected computers.
  The maximum possible number of cables used to ensure every employee can communicate with each other
  is 191.
-/
theorem max_cables_191 (A B : ℕ) (hA : A = 20) (hB : B = 10) : 
  ∃ (max_cables : ℕ), max_cables = 191 ∧ 
  (∀ (i j : ℕ), (i ≤ A ∧ j ≤ B) → (i = A ∨ j = B) → i * j ≤ max_cables) := 
sorry

end max_cables_191_l190_19004


namespace Matthew_initial_cakes_l190_19023

theorem Matthew_initial_cakes (n_cakes : ℕ) (n_crackers : ℕ) (n_friends : ℕ) (crackers_per_person : ℕ) :
  n_friends = 4 →
  n_crackers = 32 →
  crackers_per_person = 8 →
  n_crackers = n_friends * crackers_per_person →
  n_cakes = n_friends * crackers_per_person →
  n_cakes = 32 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h1, h3] at h5
  exact h5

end Matthew_initial_cakes_l190_19023


namespace system1_solution_system2_solution_l190_19025

-- System 1 Definitions
def eq1 (x y : ℝ) : Prop := 3 * x - 2 * y = 9
def eq2 (x y : ℝ) : Prop := 2 * x + 3 * y = 19

-- System 2 Definitions
def eq3 (x y : ℝ) : Prop := (2 * x + 1) / 5 - 1 = (y - 1) / 3
def eq4 (x y : ℝ) : Prop := 2 * (y - x) - 3 * (1 - y) = 6

-- Theorem Statements
theorem system1_solution (x y : ℝ) : eq1 x y ∧ eq2 x y ↔ x = 5 ∧ y = 3 := by
  sorry

theorem system2_solution (x y : ℝ) : eq3 x y ∧ eq4 x y ↔ x = 4 ∧ y = 17 / 5 := by
  sorry

end system1_solution_system2_solution_l190_19025


namespace solve_quadratic_roots_l190_19041

theorem solve_quadratic_roots (x : ℝ) : (x - 3) ^ 2 = 3 - x ↔ x = 3 ∨ x = 2 :=
by
  sorry

end solve_quadratic_roots_l190_19041


namespace ratio_of_donations_l190_19053

theorem ratio_of_donations (x : ℝ) (h1 : ∀ (y : ℝ), y = 40) (h2 : ∀ (y : ℝ), y = 40 * x)
  (h3 : ∀ (y : ℝ), y = 0.30 * (40 + 40 * x)) (h4 : ∀ (y : ℝ), y = 36) : x = 2 := 
by 
  sorry

end ratio_of_donations_l190_19053


namespace smaller_circle_radius_l190_19078

theorem smaller_circle_radius (R : ℝ) (r : ℝ) (h1 : R = 10) (h2 : R = (2 * r) / Real.sqrt 3) : r = 5 * Real.sqrt 3 :=
by
  sorry

end smaller_circle_radius_l190_19078


namespace wickets_before_last_match_l190_19084

theorem wickets_before_last_match
  (W : ℝ)  -- Number of wickets before last match
  (R : ℝ)  -- Total runs before last match
  (h1 : R = 12.4 * W)
  (h2 : (R + 26) / (W + 8) = 12.0)
  : W = 175 :=
sorry

end wickets_before_last_match_l190_19084


namespace area_of_shaded_region_l190_19087

theorem area_of_shaded_region:
  let b := 10
  let h := 6
  let n := 14
  let rect_length := 2
  let rect_height := 1.5
  (n * rect_length * rect_height - (1/2 * b * h)) = 12 := 
by
  sorry

end area_of_shaded_region_l190_19087


namespace remainder_when_sum_divided_by_30_l190_19061

theorem remainder_when_sum_divided_by_30 (x y z : ℕ) (hx : x % 30 = 14) (hy : y % 30 = 5) (hz : z % 30 = 21) :
  (x + y + z) % 30 = 10 :=
by
  sorry

end remainder_when_sum_divided_by_30_l190_19061


namespace smallest_r_minus_p_l190_19000

theorem smallest_r_minus_p 
  (p q r : ℕ) (h₀ : p * q * r = 362880) (h₁ : p < q) (h₂ : q < r) : 
  r - p = 126 :=
sorry

end smallest_r_minus_p_l190_19000


namespace arithmetic_proof_l190_19017

theorem arithmetic_proof : (28 + 48 / 69) * 69 = 1980 :=
by
  sorry

end arithmetic_proof_l190_19017


namespace picture_books_count_l190_19035

theorem picture_books_count (total_books : ℕ) (fiction_books : ℕ) (non_fiction_books : ℕ) (autobiography_books : ℕ) (picture_books : ℕ) 
  (h1 : total_books = 35)
  (h2 : fiction_books = 5)
  (h3 : non_fiction_books = fiction_books + 4)
  (h4 : autobiography_books = 2 * fiction_books)
  (h5 : picture_books = total_books - (fiction_books + non_fiction_books + autobiography_books)) :
  picture_books = 11 := 
  sorry

end picture_books_count_l190_19035


namespace not_divisible_by_5_count_l190_19063

-- Define the total number of four-digit numbers using the digits 0, 1, 2, 3, 4, 5 without repetition
def total_four_digit_numbers : ℕ := 300

-- Define the number of four-digit numbers ending with 0
def numbers_ending_with_0 : ℕ := 60

-- Define the number of four-digit numbers ending with 5
def numbers_ending_with_5 : ℕ := 48

-- Theorem stating the number of four-digit numbers that cannot be divided by 5
theorem not_divisible_by_5_count : total_four_digit_numbers - numbers_ending_with_0 - numbers_ending_with_5 = 192 :=
by
  -- Proof skipped
  sorry

end not_divisible_by_5_count_l190_19063


namespace max_principals_in_10_years_l190_19081

theorem max_principals_in_10_years : ∀ term_length num_years,
  (term_length = 4) ∧ (num_years = 10) →
  ∃ max_principals, max_principals = 3
:=
  by intros term_length num_years h
     sorry

end max_principals_in_10_years_l190_19081


namespace quadratic_roots_interlace_l190_19047

variable (p1 p2 q1 q2 : ℝ)

theorem quadratic_roots_interlace
(h : (q1 - q2)^2 + (p1 - p2) * (p1 * q2 - p2 * q1) < 0) :
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1^2 + p1 * r1 + q1 = 0 ∧ r2^2 + p1 * r2 + q1 = 0)) ∧
  (∃ s1 s2 : ℝ, s1 ≠ s2 ∧ (s1^2 + p2 * s1 + q2 = 0 ∧ s2^2 + p2 * s2 + q2 = 0)) ∧
  (∃ a b c d : ℝ, a < b ∧ b < c ∧ c < d ∧ 
  (a^2 + p1*a + q1 = 0 ∧ b^2 + p2*b + q2 = 0 ∧ c^2 + p1*c + q1 = 0 ∧ d^2 + p2*d + q2 = 0)) := 
sorry

end quadratic_roots_interlace_l190_19047


namespace action_figures_added_l190_19034

-- Definitions according to conditions
def initial_action_figures : ℕ := 4
def books_on_shelf : ℕ := 22 -- This information is not necessary for proving the action figures added
def total_action_figures_after_adding : ℕ := 10

-- Theorem to prove given the conditions
theorem action_figures_added : (total_action_figures_after_adding - initial_action_figures) = 6 := by
  sorry

end action_figures_added_l190_19034


namespace sculpture_height_l190_19032

theorem sculpture_height (base_height : ℕ) (total_height_ft : ℝ) (inches_per_foot : ℕ) 
  (h1 : base_height = 8) (h2 : total_height_ft = 3.5) (h3 : inches_per_foot = 12) : 
  (total_height_ft * inches_per_foot - base_height) = 34 := 
by
  sorry

end sculpture_height_l190_19032


namespace largest_base_5_five_digits_base_10_value_l190_19071

noncomputable def largest_base_5_five_digits_to_base_10 : ℕ :=
  4 * 5^4 + 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base_5_five_digits_base_10_value : largest_base_5_five_digits_to_base_10 = 3124 := by
  sorry

end largest_base_5_five_digits_base_10_value_l190_19071


namespace symmetric_circle_eqn_l190_19051

theorem symmetric_circle_eqn (x y : ℝ) :
  (∃ (x0 y0 : ℝ), (x - 2)^2 + (y - 2)^2 = 7 ∧ x + y = 2) → x^2 + y^2 = 7 :=
by
  sorry

end symmetric_circle_eqn_l190_19051


namespace total_value_of_remaining_books_l190_19008

-- initial definitions
def total_books : ℕ := 55
def hardback_books : ℕ := 10
def hardback_price : ℕ := 20
def paperback_price : ℕ := 10
def books_sold : ℕ := 14

-- calculate remaining books
def remaining_books : ℕ := total_books - books_sold

-- calculate remaining hardback and paperback books
def remaining_hardback_books : ℕ := hardback_books
def remaining_paperback_books : ℕ := remaining_books - remaining_hardback_books

-- calculate total values
def remaining_hardback_value : ℕ := remaining_hardback_books * hardback_price
def remaining_paperback_value : ℕ := remaining_paperback_books * paperback_price

-- total value of remaining books
def total_remaining_value : ℕ := remaining_hardback_value + remaining_paperback_value

theorem total_value_of_remaining_books : total_remaining_value = 510 := by
  -- calculation steps are skipped as instructed
  sorry

end total_value_of_remaining_books_l190_19008


namespace exists_two_numbers_l190_19069

theorem exists_two_numbers (x : Fin 7 → ℝ) :
  ∃ i j, 0 ≤ (x i - x j) / (1 + x i * x j) ∧ (x i - x j) / (1 + x i * x j) ≤ 1 / Real.sqrt 3 :=
sorry

end exists_two_numbers_l190_19069


namespace original_number_of_people_l190_19060

theorem original_number_of_people (x : ℕ) 
  (h1 : (x / 2) - ((x / 2) / 3) = 12) : 
  x = 36 :=
sorry

end original_number_of_people_l190_19060


namespace D_72_l190_19083

/-- D(n) denotes the number of ways of writing the positive integer n
    as a product n = f1 * f2 * ... * fk, where k ≥ 1, the fi are integers
    strictly greater than 1, and the order in which the factors are
    listed matters. -/
def D (n : ℕ) : ℕ := sorry

theorem D_72 : D 72 = 43 := sorry

end D_72_l190_19083


namespace period_cosine_l190_19046

noncomputable def period_of_cosine_function : ℝ := 2 * Real.pi / 3

theorem period_cosine (x : ℝ) : ∃ T, ∀ x, Real.cos (3 * x - Real.pi) = Real.cos (3 * (x + T) - Real.pi) :=
  ⟨period_of_cosine_function, by sorry⟩

end period_cosine_l190_19046


namespace ages_total_l190_19076

theorem ages_total (a b c : ℕ) (h1 : b = 8) (h2 : a = b + 2) (h3 : b = 2 * c) : a + b + c = 22 := by
  sorry

end ages_total_l190_19076


namespace total_shaded_area_l190_19031

-- Problem condition definitions
def side_length_carpet := 12
def ratio_large_square : ℕ := 4
def ratio_small_square : ℕ := 4

-- Problem statement
theorem total_shaded_area : 
  ∃ S T : ℚ, 
    12 / S = ratio_large_square ∧ S / T = ratio_small_square ∧ 
    (12 * (T * T)) + (S * S) = 15.75 := 
sorry

end total_shaded_area_l190_19031


namespace loaves_on_friday_l190_19065

theorem loaves_on_friday
  (bread_wed : ℕ)
  (bread_thu : ℕ)
  (bread_sat : ℕ)
  (bread_sun : ℕ)
  (bread_mon : ℕ)
  (inc_wed_thu : bread_thu - bread_wed = 2)
  (inc_sat_sun : bread_sun - bread_sat = 5)
  (inc_sun_mon : bread_mon - bread_sun = 6)
  (pattern : ∀ n : ℕ, bread_wed + (2 + n) + n = bread_thu + n)
  : bread_thu + 3 = 10 := 
sorry

end loaves_on_friday_l190_19065


namespace total_journey_distance_l190_19009

variable (D : ℚ) (lateTime : ℚ := 1/4)

theorem total_journey_distance :
  (∃ (T : ℚ), T = D / 40 ∧ T + lateTime = D / 35) →
  D = 70 :=
by
  intros h
  obtain ⟨T, h1, h2⟩ := h
  have h3 : T = D / 40 := h1
  have h4 : T + lateTime = D / 35 := h2
  sorry

end total_journey_distance_l190_19009


namespace canoes_to_kayaks_ratio_l190_19049

theorem canoes_to_kayaks_ratio
  (canoe_cost kayak_cost total_revenue canoes_more_than_kayaks : ℕ)
  (H1 : canoe_cost = 14)
  (H2 : kayak_cost = 15)
  (H3 : total_revenue = 288)
  (H4 : ∃ C K : ℕ, C = K + canoes_more_than_kayaks ∧ 14 * C + 15 * K = 288) :
  ∃ (r : ℚ), r = 3 / 2 := by
  sorry

end canoes_to_kayaks_ratio_l190_19049


namespace original_number_l190_19050

theorem original_number 
  (x : ℝ)
  (h₁ : 0 < x)
  (h₂ : 1000 * x = 3 * (1 / x)) : 
  x = (Real.sqrt 30) / 100 :=
sorry

end original_number_l190_19050


namespace julia_cakes_remaining_l190_19088

namespace CakeProblem

def cakes_per_day : ℕ := 5 - 1
def days_baked : ℕ := 6
def total_cakes_baked : ℕ := cakes_per_day * days_baked
def days_clifford_eats : ℕ := days_baked / 2
def cakes_eaten_by_clifford : ℕ := days_clifford_eats

theorem julia_cakes_remaining : total_cakes_baked - cakes_eaten_by_clifford = 21 :=
by
  -- proof goes here
  sorry

end CakeProblem

end julia_cakes_remaining_l190_19088


namespace janice_purchase_l190_19079

theorem janice_purchase : 
  ∃ (a b c : ℕ), a + b + c = 50 ∧ 50 * a + 400 * b + 500 * c = 10000 ∧ a = 23 :=
by
  sorry

end janice_purchase_l190_19079


namespace find_ratio_of_a_b_l190_19011

noncomputable def slope_of_tangent_to_curve_at_P := 3 * 1^2 + 1

noncomputable def perpendicular_slope (a b : ℝ) : Prop :=
  slope_of_tangent_to_curve_at_P * (a / b) = -1

noncomputable def line_slope_eq_slope_of_tangent (a b : ℝ) : Prop := 
  perpendicular_slope a b

theorem find_ratio_of_a_b (a b : ℝ) 
  (h1 : a - b * 2 = 0) 
  (h2 : line_slope_eq_slope_of_tangent a b) : 
  a / b = -1 / 4 :=
by
  sorry

end find_ratio_of_a_b_l190_19011


namespace total_height_of_sandcastles_l190_19015

structure Sandcastle :=
  (feet : Nat)
  (fraction_num : Nat)
  (fraction_den : Nat)

def janet : Sandcastle := ⟨3, 5, 6⟩
def sister : Sandcastle := ⟨2, 7, 12⟩
def tom : Sandcastle := ⟨1, 11, 20⟩
def lucy : Sandcastle := ⟨2, 13, 24⟩

-- a function to convert a Sandcastle to a common denominator
def convert_to_common_denominator (s : Sandcastle) : Sandcastle :=
  let common_den := 120 -- LCM of 6, 12, 20, 24
  ⟨s.feet, (s.fraction_num * (common_den / s.fraction_den)), common_den⟩

-- Definition of heights after conversion to common denominator
def janet_converted : Sandcastle := convert_to_common_denominator janet
def sister_converted : Sandcastle := convert_to_common_denominator sister
def tom_converted : Sandcastle := convert_to_common_denominator tom
def lucy_converted : Sandcastle := convert_to_common_denominator lucy

-- Proof problem
def total_height_proof_statement : Sandcastle :=
  let total_feet := janet.feet + sister.feet + tom.feet + lucy.feet
  let total_numerator := janet_converted.fraction_num + sister_converted.fraction_num + tom_converted.fraction_num + lucy_converted.fraction_num
  let total_denominator := 120
  ⟨total_feet + (total_numerator / total_denominator), total_numerator % total_denominator, total_denominator⟩

theorem total_height_of_sandcastles :
  total_height_proof_statement = ⟨10, 61, 120⟩ :=
by
  sorry

end total_height_of_sandcastles_l190_19015


namespace trees_died_l190_19064

theorem trees_died (initial_trees dead surviving : ℕ) 
  (h_initial : initial_trees = 11) 
  (h_surviving : surviving = dead + 7) 
  (h_total : dead + surviving = initial_trees) : 
  dead = 2 :=
by
  sorry

end trees_died_l190_19064


namespace simplify_fraction_l190_19090

-- Define the fractions and the product
def fraction1 : ℚ := 18 / 11
def fraction2 : ℚ := -42 / 45
def product : ℚ := 15 * fraction1 * fraction2

-- State the theorem to prove the correctness of the simplification
theorem simplify_fraction : product = -23 + 1 / 11 :=
by
  -- Adding this as a placeholder. The proof would go here.
  sorry

end simplify_fraction_l190_19090


namespace youngest_child_age_l190_19099

variables (child_ages : Fin 5 → ℕ)

def child_ages_eq_intervals (x : ℕ) : Prop :=
  child_ages 0 = x ∧ child_ages 1 = x + 8 ∧ child_ages 2 = x + 16 ∧ child_ages 3 = x + 24 ∧ child_ages 4 = x + 32

def sum_of_ages_eq (child_ages : Fin 5 → ℕ) (sum : ℕ) : Prop :=
  (Finset.univ : Finset (Fin 5)).sum child_ages = sum

theorem youngest_child_age (child_ages : Fin 5 → ℕ) (h1 : ∃ x, child_ages_eq_intervals child_ages x) (h2 : sum_of_ages_eq child_ages 90) :
  ∃ x, x = 2 ∧ child_ages 0 = x :=
sorry

end youngest_child_age_l190_19099


namespace hydrated_aluminum_iodide_props_l190_19027

noncomputable def Al_mass : ℝ := 26.98
noncomputable def I_mass : ℝ := 126.90
noncomputable def H2O_mass : ℝ := 18.015
noncomputable def AlI3_mass (mass_AlI3: ℝ) : ℝ := 26.98 + 3 * 126.90

noncomputable def mass_percentage_iodine (mass_AlI3 mass_sample: ℝ) : ℝ :=
  (mass_AlI3 * (3 * I_mass / (Al_mass + 3 * I_mass)) / mass_sample) * 100

noncomputable def value_x (mass_H2O mass_AlI3: ℝ) : ℝ :=
  (mass_H2O / H2O_mass) / (mass_AlI3 / (Al_mass + 3 * I_mass))

theorem hydrated_aluminum_iodide_props (mass_AlI3 mass_H2O mass_sample: ℝ)
    (h_sample: mass_AlI3 + mass_H2O = mass_sample) :
    ∃ (percentage: ℝ) (x: ℝ), percentage = mass_percentage_iodine mass_AlI3 mass_sample ∧
                                      x = value_x mass_H2O mass_AlI3 :=
by
  sorry

end hydrated_aluminum_iodide_props_l190_19027


namespace root_exists_between_a_and_b_l190_19037

variable {α : Type*} [LinearOrderedField α]

theorem root_exists_between_a_and_b (a b p q : α) (h₁ : a^2 + p * a + q = 0) (h₂ : b^2 - p * b - q = 0) (h₃ : q ≠ 0) :
  ∃ c, a < c ∧ c < b ∧ (c^2 + 2 * p * c + 2 * q = 0) := by
  sorry

end root_exists_between_a_and_b_l190_19037


namespace math_problem_l190_19020

variable (f g : ℝ → ℝ)
variable (a b x : ℝ)
variable (h_has_derivative_f : ∀ x, Differentiable ℝ f)
variable (h_has_derivative_g : ∀ x, Differentiable ℝ g)
variable (h_deriv_ineq : ∀ x, deriv f x > deriv g x)
variable (h_interval : x ∈ Ioo a b)

theorem math_problem :
  (f x + g b < g x + f b) ∧ (f x + g a > g x + f a) :=
sorry

end math_problem_l190_19020


namespace average_daily_net_income_correct_l190_19092

-- Define the income, tips, and expenses for each day.
def day1_income := 300
def day1_tips := 50
def day1_expenses := 80

def day2_income := 150
def day2_tips := 20
def day2_expenses := 40

def day3_income := 750
def day3_tips := 100
def day3_expenses := 150

def day4_income := 200
def day4_tips := 30
def day4_expenses := 50

def day5_income := 600
def day5_tips := 70
def day5_expenses := 120

-- Define the net income for each day as income + tips - expenses.
def day1_net_income := day1_income + day1_tips - day1_expenses
def day2_net_income := day2_income + day2_tips - day2_expenses
def day3_net_income := day3_income + day3_tips - day3_expenses
def day4_net_income := day4_income + day4_tips - day4_expenses
def day5_net_income := day5_income + day5_tips - day5_expenses

-- Calculate the total net income over the 5 days.
def total_net_income := 
  day1_net_income + day2_net_income + day3_net_income + day4_net_income + day5_net_income

-- Define the number of days.
def number_of_days := 5

-- Calculate the average daily net income.
def average_daily_net_income := total_net_income / number_of_days

-- Statement to prove that the average daily net income is $366.
theorem average_daily_net_income_correct :
  average_daily_net_income = 366 := by
  sorry

end average_daily_net_income_correct_l190_19092


namespace continuous_stripe_probability_l190_19024

-- Define a structure representing the configuration of each face.
structure FaceConfiguration where
  is_diagonal : Bool
  edge_pair_or_vertex_pair : Bool

-- Define the cube configuration.
structure CubeConfiguration where
  face1 : FaceConfiguration
  face2 : FaceConfiguration
  face3 : FaceConfiguration
  face4 : FaceConfiguration
  face5 : FaceConfiguration
  face6 : FaceConfiguration

noncomputable def total_configurations : ℕ := 4^6

-- Define the function that checks if a configuration results in a continuous stripe.
def results_in_continuous_stripe (c : CubeConfiguration) : Bool := sorry

-- Define the number of configurations resulting in a continuous stripe.
noncomputable def configurations_with_continuous_stripe : ℕ :=
  Nat.card {c : CubeConfiguration // results_in_continuous_stripe c}

-- Define the probability calculation.
noncomputable def probability_continuous_stripe : ℚ :=
  configurations_with_continuous_stripe / total_configurations

-- The statement of the problem: Prove the probability of continuous stripe is 3/256.
theorem continuous_stripe_probability :
  probability_continuous_stripe = 3 / 256 :=
sorry

end continuous_stripe_probability_l190_19024


namespace problem_a_problem_d_l190_19074

theorem problem_a (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : (1 / (a * b)) ≥ 1 / 4 :=
by
  sorry

theorem problem_d (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) : a^2 + b^2 ≥ 8 :=
by
  sorry

end problem_a_problem_d_l190_19074


namespace fixed_points_a_one_b_five_range_of_a_two_distinct_fixed_points_l190_19085

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means to be a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Condition 1: a = 1, b = 5; the fixed points are x = -1 or x = -4
theorem fixed_points_a_one_b_five : 
  ∀ x : ℝ, is_fixed_point (f 1 5) x ↔ x = -1 ∨ x = -4 := by
  -- Proof goes here
  sorry

-- Condition 2: For any real b, f(x) always having two distinct fixed points implies 0 < a < 1
theorem range_of_a_two_distinct_fixed_points : 
  (∀ b : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point (f a b) x1 ∧ is_fixed_point (f a b) x2) ↔ 0 < a ∧ a < 1 := by
  -- Proof goes here
  sorry

end fixed_points_a_one_b_five_range_of_a_two_distinct_fixed_points_l190_19085


namespace johns_weekly_earnings_increase_l190_19095

noncomputable def percentageIncrease (original new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem johns_weekly_earnings_increase :
  percentageIncrease 30 40 = 33.33 :=
by
  sorry

end johns_weekly_earnings_increase_l190_19095


namespace point_coordinates_with_respect_to_origin_l190_19089

theorem point_coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (3, -2)) : (x, y) = (3, -2) :=
by
  sorry

end point_coordinates_with_respect_to_origin_l190_19089


namespace general_term_l190_19070

def S (n : ℕ) : ℤ := n^2 - 4*n

noncomputable def a (n : ℕ) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem general_term (n : ℕ) (hn : n ≥ 1) : a n = (2 * n - 5) := by
  sorry

end general_term_l190_19070


namespace alex_play_friends_with_l190_19002

variables (A B V G D : Prop)

-- Condition 1: If Andrew goes, then Boris will also go and Vasya will not go.
axiom cond1 : A → (B ∧ ¬V)
-- Condition 2: If Boris goes, then either Gena or Denis will also go.
axiom cond2 : B → (G ∨ D)
-- Condition 3: If Vasya does not go, then neither Boris nor Denis will go.
axiom cond3 : ¬V → (¬B ∧ ¬D)
-- Condition 4: If Andrew does not go, then Boris will go and Gena will not go.
axiom cond4 : ¬A → (B ∧ ¬G)

theorem alex_play_friends_with :
  (B ∧ V ∧ D) :=
by
  sorry

end alex_play_friends_with_l190_19002


namespace range_of_k_l190_19033

theorem range_of_k {k : ℝ} :
  (∀ x : ℝ, k * x^2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
by sorry

end range_of_k_l190_19033


namespace number_of_real_solutions_of_equation_l190_19016

theorem number_of_real_solutions_of_equation :
  (∀ x : ℝ, ((2 : ℝ)^(4 * x + 2)) * ((4 : ℝ)^(2 * x + 8)) = ((8 : ℝ)^(3 * x + 7))) ↔ x = -3 :=
by sorry

end number_of_real_solutions_of_equation_l190_19016


namespace common_ratio_of_series_l190_19026

-- Definition of the terms in the series
def term1 : ℚ := 7 / 8
def term2 : ℚ := - (5 / 12)

-- Definition of the common ratio
def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

-- The theorem we need to prove that the common ratio is -10/21
theorem common_ratio_of_series : common_ratio term1 term2 = -10 / 21 :=
by
  -- We skip the proof with 'sorry'
  sorry

end common_ratio_of_series_l190_19026


namespace shortest_distance_proof_l190_19003

noncomputable def shortest_distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem shortest_distance_proof : 
  let A : ℝ × ℝ := (0, 250)
  let B : ℝ × ℝ := (800, 1050)
  shortest_distance A B = 1131 :=
by
  sorry

end shortest_distance_proof_l190_19003


namespace arithmetic_sequence_a5_value_l190_19028

variable (a : ℕ → ℝ)
variable (a_2 a_5 a_8 : ℝ)
variable (h1 : a 2 + a 8 = 15 - a 5)

/-- In an arithmetic sequence {a_n}, given that a_2 + a_8 = 15 - a_5, prove that a_5 equals 5. -/ 
theorem arithmetic_sequence_a5_value (h1 : a 2 + a 8 = 15 - a 5) : a 5 = 5 :=
sorry

end arithmetic_sequence_a5_value_l190_19028


namespace GreatWhiteSharkTeeth_l190_19005

-- Definition of the number of teeth for a tiger shark
def tiger_shark_teeth : ℕ := 180

-- Definition of the number of teeth for a hammerhead shark
def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

-- Definition of the number of teeth for a great white shark
def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

-- Statement to prove
theorem GreatWhiteSharkTeeth : great_white_shark_teeth = 420 :=
by
  -- Proof omitted
  sorry

end GreatWhiteSharkTeeth_l190_19005


namespace sector_area_is_2pi_l190_19001

noncomputable def sectorArea (l : ℝ) (R : ℝ) : ℝ :=
  (1 / 2) * l * R

theorem sector_area_is_2pi (R : ℝ) (l : ℝ) (hR : R = 4) (hl : l = π) :
  sectorArea l R = 2 * π :=
by
  sorry

end sector_area_is_2pi_l190_19001


namespace fraction_always_irreducible_l190_19055

-- Define the problem statement
theorem fraction_always_irreducible (n : ℤ) : gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end fraction_always_irreducible_l190_19055


namespace trajectory_equation_l190_19010

variable (m x y : ℝ)
def a := (m * x, y + 1)
def b := (x, y - 1)
def is_perpendicular (u v : ℝ × ℝ) := u.1 * v.1 + u.2 * v.2 = 0

theorem trajectory_equation 
  (h1: is_perpendicular (a m x y) (b x y)) : 
  m * x^2 + y^2 = 1 :=
sorry

end trajectory_equation_l190_19010


namespace slices_with_both_l190_19082

theorem slices_with_both (n total_slices pepperoni_slices mushroom_slices other_slices : ℕ)
  (h1 : total_slices = 24) 
  (h2 : pepperoni_slices = 15)
  (h3 : mushroom_slices = 14)
  (h4 : (pepperoni_slices - n) + (mushroom_slices - n) + n = total_slices) :
  n = 5 :=
sorry

end slices_with_both_l190_19082


namespace g_zero_g_one_l190_19058

variable (g : ℤ → ℤ)

axiom condition1 (x : ℤ) : g (x + 5) - g x = 10 * x + 30
axiom condition2 (x : ℤ) : g (x^2 - 2) = (g x - x)^2 + x^2 - 4

theorem g_zero_g_one : (g 0, g 1) = (-4, 1) := 
by 
  sorry

end g_zero_g_one_l190_19058


namespace tan_alpha_plus_pi_over_3_sin_cos_ratio_l190_19018

theorem tan_alpha_plus_pi_over_3
  (α : ℝ)
  (h : Real.tan (α / 2) = 3) :
  Real.tan (α + Real.pi / 3) = (48 - 25 * Real.sqrt 3) / 11 := 
sorry

theorem sin_cos_ratio
  (α : ℝ)
  (h : Real.tan (α / 2) = 3) :
  (Real.sin α + 2 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = -5 / 17 :=
sorry

end tan_alpha_plus_pi_over_3_sin_cos_ratio_l190_19018


namespace knight_liar_grouping_l190_19029

noncomputable def can_be_partitioned_into_knight_liar_groups (n m : ℕ) (h1 : n ≥ 2) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k < n) : Prop :=
  ∃ t : ℕ, n = (m + 1) * t

-- Show that if the company has n people, where n ≥ 2, and there exists at least one knight,
-- then n can be partitioned into groups where each group contains 1 knight and m liars.
theorem knight_liar_grouping (n m : ℕ) (h1 : n ≥ 2) (h2 : ∃ k : ℕ, 1 ≤ k ∧ k < n) : can_be_partitioned_into_knight_liar_groups n m h1 h2 :=
sorry

end knight_liar_grouping_l190_19029


namespace simplify_expression_l190_19044

theorem simplify_expression (x y : ℤ) : 1 - (2 - (3 - (4 - (5 - x)))) - y = 3 - (x + y) := 
by 
  sorry 

end simplify_expression_l190_19044


namespace total_payment_for_combined_shopping_trip_l190_19014

noncomputable def discount (amount : ℝ) : ℝ :=
  if amount ≤ 200 then amount
  else if amount ≤ 500 then amount * 0.9
  else 500 * 0.9 + (amount - 500) * 0.7

theorem total_payment_for_combined_shopping_trip :
  discount (168 + 423 / 0.9) = 546.6 :=
by
  sorry

end total_payment_for_combined_shopping_trip_l190_19014


namespace current_height_of_tree_l190_19012

-- Definitions of conditions
def growth_per_year : ℝ := 0.5
def years : ℕ := 240
def final_height : ℝ := 720

-- The goal is to prove that the current height of the tree is 600 inches
theorem current_height_of_tree :
  final_height - (growth_per_year * years) = 600 := 
sorry

end current_height_of_tree_l190_19012
