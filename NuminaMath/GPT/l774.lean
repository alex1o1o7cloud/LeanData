import Mathlib

namespace money_after_purchase_l774_77483

def initial_money : ℕ := 4
def cost_of_candy_bar : ℕ := 1
def money_left : ℕ := 3

theorem money_after_purchase :
  initial_money - cost_of_candy_bar = money_left := by
  sorry

end money_after_purchase_l774_77483


namespace solution_set_inequality_system_l774_77415

theorem solution_set_inequality_system (
  x : ℝ
) : (x + 1 ≥ 0 ∧ (x - 1) / 2 < 1) ↔ (-1 ≤ x ∧ x < 3) := by
  sorry

end solution_set_inequality_system_l774_77415


namespace e_count_estimation_l774_77475

-- Define the various parameters used in the conditions
def num_problems : Nat := 76
def avg_words_per_problem : Nat := 40
def avg_letters_per_word : Nat := 5
def frequency_of_e : Float := 0.1
def actual_e_count : Nat := 1661

-- The goal is to prove that the actual number of "e"s is 1661
theorem e_count_estimation : actual_e_count = 1661 := by
  -- Sorry, no proof is required.
  sorry

end e_count_estimation_l774_77475


namespace remaining_yards_is_720_l774_77448

-- Definitions based on conditions:
def marathon_miles : Nat := 25
def marathon_yards : Nat := 500
def yards_in_mile : Nat := 1760
def num_of_marathons : Nat := 12

-- Total distance for one marathon in yards
def one_marathon_total_yards : Nat :=
  marathon_miles * yards_in_mile + marathon_yards

-- Total distance for twelve marathons in yards
def total_distance_yards : Nat :=
  num_of_marathons * one_marathon_total_yards

-- Remaining yards after converting the total distance into miles and yards
def y : Nat :=
  total_distance_yards % yards_in_mile

-- Condition ensuring y is the remaining yards and is within the bounds 0 ≤ y < 1760
theorem remaining_yards_is_720 : 
  y = 720 := sorry

end remaining_yards_is_720_l774_77448


namespace total_points_l774_77459

noncomputable def Darius_points : ℕ := 10
noncomputable def Marius_points : ℕ := Darius_points + 3
noncomputable def Matt_points : ℕ := Darius_points + 5
noncomputable def Sofia_points : ℕ := 2 * Matt_points

theorem total_points : Darius_points + Marius_points + Matt_points + Sofia_points = 68 :=
by
  -- Definitions are directly from the problem statement, proof skipped 
  sorry

end total_points_l774_77459


namespace order_of_magnitude_l774_77458

theorem order_of_magnitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let m := a / Real.sqrt b + b / Real.sqrt a
  let n := Real.sqrt a + Real.sqrt b
  let p := Real.sqrt (a + b)
  m ≥ n ∧ n > p := 
sorry

end order_of_magnitude_l774_77458


namespace set_intersection_eq_l774_77422

def A : Set ℝ := {x | |x - 1| ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x > 0}

theorem set_intersection_eq :
  A ∩ (Set.univ \ B) = {x | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end set_intersection_eq_l774_77422


namespace LeRoy_should_pay_30_l774_77460

/-- Define the empirical amounts paid by LeRoy and Bernardo, and the total discount. -/
def LeRoy_paid : ℕ := 240
def Bernardo_paid : ℕ := 360
def total_discount : ℕ := 60

/-- Define total expenses pre-discount. -/
def total_expenses : ℕ := LeRoy_paid + Bernardo_paid

/-- Define total expenses post-discount. -/
def adjusted_expenses : ℕ := total_expenses - total_discount

/-- Define each person's adjusted share. -/
def each_adjusted_share : ℕ := adjusted_expenses / 2

/-- Define the amount LeRoy should pay Bernardo. -/
def leroy_to_pay : ℕ := each_adjusted_share - LeRoy_paid

/-- Prove that LeRoy should pay Bernardo $30 to equalize their expenses post-discount. -/
theorem LeRoy_should_pay_30 : leroy_to_pay = 30 :=
by 
  -- Proof goes here...
  sorry

end LeRoy_should_pay_30_l774_77460


namespace power_function_characterization_l774_77488

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_characterization (f : ℝ → ℝ) (h : f 2 = Real.sqrt 2) : 
  ∀ x : ℝ, f x = x ^ (1 / 2) :=
sorry

end power_function_characterization_l774_77488


namespace jake_hours_of_work_l774_77431

def initialDebt : ℕ := 100
def amountPaid : ℕ := 40
def workRate : ℕ := 15
def remainingDebt : ℕ := initialDebt - amountPaid

theorem jake_hours_of_work : remainingDebt / workRate = 4 := by
  sorry

end jake_hours_of_work_l774_77431


namespace solve_eq_integers_l774_77466

theorem solve_eq_integers (x y : ℤ) : 
    x^2 - x * y - 6 * y^2 + 2 * x + 19 * y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) := by
    sorry

end solve_eq_integers_l774_77466


namespace number_of_terms_in_arithmetic_sequence_l774_77432

theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ n → 6 + (k - 1) * 2 = 202)) ∧ n = 99 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l774_77432


namespace total_cleaning_validation_l774_77405

-- Define the cleaning frequencies and their vacations
def Michael_bath_week := 2
def Michael_shower_week := 1
def Michael_vacation_weeks := 3

def Angela_shower_day := 1
def Angela_vacation_weeks := 2

def Lucy_bath_week := 3
def Lucy_shower_week := 2
def Lucy_alter_weeks := 4
def Lucy_alter_shower_day := 1
def Lucy_alter_bath_week := 1

def weeks_year := 52
def days_week := 7

-- Calculate Michael's total cleaning times in a year
def Michael_total := (Michael_bath_week * weeks_year) + (Michael_shower_week * weeks_year)
def Michael_vacation_reduction := Michael_vacation_weeks * (Michael_bath_week + Michael_shower_week)
def Michael_cleaning_times := Michael_total - Michael_vacation_reduction

-- Calculate Angela's total cleaning times in a year
def Angela_total := (Angela_shower_day * days_week * weeks_year)
def Angela_vacation_reduction := Angela_vacation_weeks * (Angela_shower_day * days_week)
def Angela_cleaning_times := Angela_total - Angela_vacation_reduction

-- Calculate Lucy's total cleaning times in a year
def Lucy_baths_total := Lucy_bath_week * weeks_year
def Lucy_showers_total := Lucy_shower_week * weeks_year
def Lucy_alter_showers := Lucy_alter_shower_day * days_week * Lucy_alter_weeks
def Lucy_alter_baths_reduction := (Lucy_bath_week - Lucy_alter_bath_week) * Lucy_alter_weeks
def Lucy_cleaning_times := Lucy_baths_total + Lucy_showers_total + Lucy_alter_showers - Lucy_alter_baths_reduction

-- Calculate the total times they clean themselves in 52 weeks
def total_cleaning_times := Michael_cleaning_times + Angela_cleaning_times + Lucy_cleaning_times

-- The proof statement
theorem total_cleaning_validation : total_cleaning_times = 777 :=
by simp [Michael_cleaning_times, Angela_cleaning_times, Lucy_cleaning_times, total_cleaning_times]; sorry

end total_cleaning_validation_l774_77405


namespace smallest_positive_angle_l774_77404

theorem smallest_positive_angle (theta : ℝ) (h_theta : theta = -2002) :
  ∃ α : ℝ, 0 < α ∧ α < 360 ∧ ∃ k : ℤ, theta = α + k * 360 ∧ α = 158 := 
by
  sorry

end smallest_positive_angle_l774_77404


namespace number_of_triangles_from_8_points_on_circle_l774_77430

-- Definitions based on the problem conditions
def points_on_circle : ℕ := 8

-- Problem statement without the proof
theorem number_of_triangles_from_8_points_on_circle :
  ∃ n : ℕ, n = (points_on_circle.choose 3) ∧ n = 56 := 
by
  sorry

end number_of_triangles_from_8_points_on_circle_l774_77430


namespace min_points_in_set_M_l774_77425
-- Import the necessary library

-- Define the problem conditions and the result to prove
theorem min_points_in_set_M :
  ∃ (M : Finset ℝ) (C₁ C₂ C₃ C₄ C₅ C₆ C₇ : Finset ℝ),
  C₇.card = 7 ∧
  C₆.card = 6 ∧
  C₅.card = 5 ∧
  C₄.card = 4 ∧
  C₃.card = 3 ∧
  C₂.card = 2 ∧
  C₁.card = 1 ∧
  C₇ ⊆ M ∧
  C₆ ⊆ M ∧
  C₅ ⊆ M ∧
  C₄ ⊆ M ∧
  C₃ ⊆ M ∧
  C₂ ⊆ M ∧
  C₁ ⊆ M ∧
  M.card = 12 :=
sorry

end min_points_in_set_M_l774_77425


namespace circle_diameter_line_eq_l774_77400

theorem circle_diameter_line_eq (x y : ℝ) :
  x^2 + y^2 - 2*x + 6*y + 8 = 0 → (2 * 1 + (-3) + 1 = 0) :=
by
  sorry

end circle_diameter_line_eq_l774_77400


namespace find_A_coords_find_AC_equation_l774_77411

theorem find_A_coords
  (B : ℝ × ℝ) (hB : B = (1, -2))
  (median_CM : ∀ x y, 2 * x - y + 1 = 0)
  (angle_bisector_BAC : ∀ x y, x + 7 * y - 12 = 0) :
  ∃ A : ℝ × ℝ, A = (-2, 2) :=
by
  sorry

theorem find_AC_equation
  (A B : ℝ × ℝ) (hA : A = (-2, 2)) (hB : B = (1, -2))
  (median_CM : ∀ x y, 2 * x - y + 1 = 0)
  (angle_bisector_BAC : ∀ x y, x + 7 * y - 12 = 0) :
  ∃ k b : ℝ, ∀ x y, y = k * x + b ↔ 3 * x - 4 * y + 14 = 0 :=
by
  sorry

end find_A_coords_find_AC_equation_l774_77411


namespace abscissa_of_point_P_l774_77487

open Real

noncomputable def hyperbola_abscissa (x y : ℝ) : Prop :=
  (x^2 - y^2 = 4) ∧
  (x > 0) ∧
  ((x + 2 * sqrt 2) * (x - 2 * sqrt 2) = -y^2)

theorem abscissa_of_point_P :
  ∃ (x y : ℝ), hyperbola_abscissa x y ∧ x = sqrt 6 := by
  sorry

end abscissa_of_point_P_l774_77487


namespace coins_amount_correct_l774_77484

-- Definitions based on the conditions
def cost_of_flour : ℕ := 5
def cost_of_cake_stand : ℕ := 28
def amount_given_in_bills : ℕ := 20 + 20
def change_received : ℕ := 10

-- Total cost of items
def total_cost : ℕ := cost_of_flour + cost_of_cake_stand

-- Total money given
def total_money_given : ℕ := total_cost + change_received

-- Amount given in loose coins
def loose_coins_given : ℕ := total_money_given - amount_given_in_bills

-- Proposition statement
theorem coins_amount_correct : loose_coins_given = 3 := by
  sorry

end coins_amount_correct_l774_77484


namespace roots_sum_condition_l774_77454

theorem roots_sum_condition (a b : ℝ) 
  (h1 : ∃ (x y z : ℝ), (x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 9) 
    ∧ (x * y + y * z + x * z = a) ∧ (x * y * z = b)) :
  a + b = 38 := 
sorry

end roots_sum_condition_l774_77454


namespace smallest_rel_prime_210_l774_77485

theorem smallest_rel_prime_210 (x : ℕ) (hx : x > 1) (hrel_prime : Nat.gcd x 210 = 1) : x = 11 :=
sorry

end smallest_rel_prime_210_l774_77485


namespace sum_of_numbers_is_37_l774_77453

theorem sum_of_numbers_is_37 :
  ∃ (A B : ℕ), 
    1 ≤ A ∧ A ≤ 50 ∧ 1 ≤ B ∧ B ≤ 50 ∧ A ≠ B ∧
    (50 * B + A = k^2) ∧ Prime B ∧ B > 10 ∧
    A + B = 37 
  := by
    sorry

end sum_of_numbers_is_37_l774_77453


namespace find_a_b_solve_inequality_l774_77491

-- Definitions for the given conditions
def inequality1 (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 6 > 4
def sol_set1 (x : ℝ) (b : ℝ) : Prop := x < 1 ∨ x > b
def root_eq (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 2 = 0

-- The final Lean statements for the proofs
theorem find_a_b (a b : ℝ) : (∀ x, (inequality1 a x) ↔ (sol_set1 x b)) → a = 1 ∧ b = 2 :=
sorry

theorem solve_inequality (c : ℝ) : 
  (∀ x, (root_eq 1 x) ↔ (x = 1 ∨ x = 2)) → 
  (c > 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ (2 < x ∧ x < c)) ∧
  (c < 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ (c < x ∧ x < 2)) ∧
  (c = 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ false) :=
sorry

end find_a_b_solve_inequality_l774_77491


namespace value_of_t_l774_77416

theorem value_of_t (k m r s t : ℕ) 
  (hk : 1 ≤ k) (hm : 2 ≤ m) (hr : r = 13) (hs : s = 14)
  (h : k < m) (h' : m < r) (h'' : r < s) (h''' : s < t)
  (average_condition : (k + m + r + s + t) / 5 = 10) :
  t = 20 := 
sorry

end value_of_t_l774_77416


namespace symmetric_circle_equation_l774_77492

theorem symmetric_circle_equation :
  ∀ (a b : ℝ), 
    (∀ (x y : ℝ), (x-2)^2 + (y+1)^2 = 4 → y = x + 1) → 
    (∃ x y : ℝ, (x + 2)^2 + (y - 3)^2 = 4) :=
  by
    sorry

end symmetric_circle_equation_l774_77492


namespace operation_three_six_l774_77472

theorem operation_three_six : (3 * 3 * 6) / (3 + 6) = 6 :=
by
  calc (3 * 3 * 6) / (3 + 6) = 6 := sorry

end operation_three_six_l774_77472


namespace fibonacci_problem_l774_77413

theorem fibonacci_problem 
  (F : ℕ → ℕ)
  (h1 : F 1 = 1)
  (h2 : F 2 = 1)
  (h3 : ∀ n ≥ 3, F n = F (n - 1) + F (n - 2))
  (a b c : ℕ)
  (h4 : F c = 2 * F b - F a)
  (h5 : F c - F a = F a)
  (h6 : a + c = 1700) :
  a = 849 := 
sorry

end fibonacci_problem_l774_77413


namespace pentagon_largest_angle_l774_77417

theorem pentagon_largest_angle (x : ℝ) (h : 2 * x + 3 * x + 4 * x + 5 * x + 6 * x = 540) : 6 * x = 162 :=
sorry

end pentagon_largest_angle_l774_77417


namespace Beth_peas_count_l774_77401

-- Definitions based on conditions
def number_of_corn : ℕ := 10
def number_of_peas (number_of_corn : ℕ) : ℕ := 2 * number_of_corn + 15

-- Theorem that represents the proof problem
theorem Beth_peas_count : number_of_peas 10 = 35 :=
by
  sorry

end Beth_peas_count_l774_77401


namespace xyz_zero_unique_solution_l774_77442

theorem xyz_zero_unique_solution {x y z : ℝ} (h1 : x^2 * y + y^2 * z + z^2 = 0)
                                 (h2 : z^3 + z^2 * y + z * y^3 + x^2 * y = 1 / 4 * (x^4 + y^4)) :
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end xyz_zero_unique_solution_l774_77442


namespace andrew_age_proof_l774_77480

def andrew_age_problem : Prop :=
  ∃ (a g : ℚ), g = 15 * a ∧ g - a = 60 ∧ a = 30 / 7

theorem andrew_age_proof : andrew_age_problem :=
by
  sorry

end andrew_age_proof_l774_77480


namespace total_pencils_in_drawer_l774_77481

-- Definitions based on conditions from the problem
def initial_pencils : ℕ := 138
def pencils_by_Nancy : ℕ := 256
def pencils_by_Steven : ℕ := 97

-- The theorem proving the total number of pencils in the drawer
theorem total_pencils_in_drawer : initial_pencils + pencils_by_Nancy + pencils_by_Steven = 491 :=
by
  -- This statement is equivalent to the mathematical problem given
  sorry

end total_pencils_in_drawer_l774_77481


namespace merchant_profit_l774_77409

theorem merchant_profit (C S : ℝ) (h: 20 * C = 15 * S) : 
  (S - C) / C * 100 = 33.33 := by
sorry

end merchant_profit_l774_77409


namespace area_of_field_l774_77437

theorem area_of_field : ∀ (L W : ℕ), L = 20 → L + 2 * W = 88 → L * W = 680 :=
by
  intros L W hL hEq
  rw [hL] at hEq
  sorry

end area_of_field_l774_77437


namespace factorize_expression_l774_77439

theorem factorize_expression (a : ℝ) : a^3 + 2*a^2 + a = a*(a+1)^2 :=
  sorry

end factorize_expression_l774_77439


namespace exists_additive_function_close_to_f_l774_77433

variable (f : ℝ → ℝ)

theorem exists_additive_function_close_to_f (h : ∀ x y : ℝ, |f (x + y) - f x - f y| ≤ 1) :
  ∃ g : ℝ → ℝ, (∀ x : ℝ, |f x - g x| ≤ 1) ∧ (∀ x y : ℝ, g (x + y) = g x + g y) := by
  sorry

end exists_additive_function_close_to_f_l774_77433


namespace min_weighings_to_identify_fake_l774_77463

def piles := 1000000
def coins_per_pile := 1996
def weight_real_coin := 10
def weight_fake_coin := 9
def expected_total_weight : Nat :=
  (piles * (piles + 1) / 2) * weight_real_coin

theorem min_weighings_to_identify_fake :
  (∃ k : ℕ, k < piles ∧ 
  ∀ (W : ℕ), W = expected_total_weight - k → k = expected_total_weight - W) →
  true := 
by
  sorry

end min_weighings_to_identify_fake_l774_77463


namespace tan_sum_eq_tan_product_l774_77449

theorem tan_sum_eq_tan_product {α β γ : ℝ} 
  (h_sum : α + β + γ = π) : 
    Real.tan α + Real.tan β + Real.tan γ = Real.tan α * Real.tan β * Real.tan γ :=
by
  sorry

end tan_sum_eq_tan_product_l774_77449


namespace find_m_in_arith_seq_l774_77440

noncomputable def arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem find_m_in_arith_seq (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0) 
  (h_seq : arith_seq a d) 
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32) 
  (h_am : ∃ m, a m = 8) : 
  ∃ m, m = 8 := 
sorry

end find_m_in_arith_seq_l774_77440


namespace motherGaveMoney_l774_77428

-- Define the given constants and fact
def initialMoney : Real := 0.85
def foundMoney : Real := 0.50
def toyCost : Real := 1.60
def remainingMoney : Real := 0.15

-- Define the unknown amount given by his mother
def motherMoney (M : Real) := initialMoney + M + foundMoney - toyCost = remainingMoney

-- Statement to prove
theorem motherGaveMoney : ∃ M : Real, motherMoney M ∧ M = 0.40 :=
by
  sorry

end motherGaveMoney_l774_77428


namespace find_m_given_sampling_conditions_l774_77486

-- Definitions for population and sampling conditions
def population_divided_into_groups : Prop :=
  ∀ n : ℕ, n < 100 → ∃ k : ℕ, k < 10 ∧ n / 10 = k

def systematic_sampling_condition (m k : ℕ) : Prop :=
  k < 10 ∧ m < 10 ∧ (m + k - 1) % 10 < 10 ∧ (m + k - 11) % 10 < 10

-- Given conditions
def given_conditions (m k : ℕ) (n : ℕ) : Prop :=
  k = 6 ∧ n = 52 ∧ systematic_sampling_condition m k

-- The statement to prove
theorem find_m_given_sampling_conditions :
  ∃ m : ℕ, given_conditions m 6 52 ∧ m = 7 :=
by
  sorry

end find_m_given_sampling_conditions_l774_77486


namespace max_friday_more_than_wednesday_l774_77490

-- Definitions and conditions
def played_hours_wednesday : ℕ := 2
def played_hours_thursday : ℕ := 2
def played_average_hours : ℕ := 3
def played_days : ℕ := 3

-- Total hours over three days
def total_hours : ℕ := played_average_hours * played_days

-- Hours played on Friday
def played_hours_wednesday_thursday : ℕ := played_hours_wednesday + played_hours_thursday

def played_hours_friday : ℕ := total_hours - played_hours_wednesday_thursday

-- Proof problem statement
theorem max_friday_more_than_wednesday : 
  played_hours_friday - played_hours_wednesday = 3 := 
sorry

end max_friday_more_than_wednesday_l774_77490


namespace value_of_fraction_zero_l774_77418

theorem value_of_fraction_zero (x : ℝ) (h1 : x^2 - 1 = 0) (h2 : 1 - x ≠ 0) : x = -1 :=
by
  sorry

end value_of_fraction_zero_l774_77418


namespace sequence_length_l774_77470

theorem sequence_length 
  (a : ℕ)
  (b : ℕ)
  (d : ℕ)
  (steps : ℕ)
  (h1 : a = 160)
  (h2 : b = 28)
  (h3 : d = 4)
  (h4 : (28:ℕ) = (160:ℕ) - steps * 4) :
  steps + 1 = 34 :=
by
  sorry

end sequence_length_l774_77470


namespace arithmetic_sequence_a5_l774_77435

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n, a (n + 1) = a n + 2

-- Statement of the theorem with conditions and conclusion
theorem arithmetic_sequence_a5 :
  ∃ a : ℕ → ℕ, is_arithmetic_sequence a ∧ a 1 = 1 ∧ a 5 = 9 :=
by
  sorry

end arithmetic_sequence_a5_l774_77435


namespace calculate_total_houses_built_l774_77419

theorem calculate_total_houses_built :
  let initial_houses := 1426
  let final_houses := 2000
  let rate_a := 25
  let time_a := 6
  let rate_b := 15
  let time_b := 9
  let rate_c := 30
  let time_c := 4
  let total_houses_built := (rate_a * time_a) + (rate_b * time_b) + (rate_c * time_c)
  total_houses_built = 405 :=
by
  sorry

end calculate_total_houses_built_l774_77419


namespace largest_s_value_l774_77434

theorem largest_s_value (r s : ℕ) (h_r : r ≥ 3) (h_s : s ≥ 3) 
  (h_angle : (r - 2) * 180 / r = (5 * (s - 2) * 180) / (4 * s)) : s ≤ 130 :=
by {
  sorry
}

end largest_s_value_l774_77434


namespace mary_mortgage_payment_l774_77403

theorem mary_mortgage_payment :
  let a1 := 400
  let r := 2
  let n := 11
  let sum_geom_series (a1 r : ℕ) (n : ℕ) : ℕ := (a1 * (1 - r^n)) / (1 - r)
  sum_geom_series a1 r n = 819400 :=
by
  let a1 := 400
  let r := 2
  let n := 11
  let sum_geom_series (a1 r : ℕ) (n : ℕ) : ℕ := (a1 * (1 - r^n)) / (1 - r)
  have h : sum_geom_series a1 r n = 819400 := sorry
  exact h

end mary_mortgage_payment_l774_77403


namespace no_n_gt_1_divisibility_l774_77461

theorem no_n_gt_1_divisibility (n : ℕ) (h : n > 1) : ¬ (3 ^ (n - 1) + 5 ^ (n - 1)) ∣ (3 ^ n + 5 ^ n) :=
by
  sorry

end no_n_gt_1_divisibility_l774_77461


namespace multiplication_to_squares_l774_77420

theorem multiplication_to_squares :
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 :=
by
  sorry

end multiplication_to_squares_l774_77420


namespace find_value_of_fraction_l774_77421

noncomputable def a : ℝ := 5 * (Real.sqrt 2) + 7

theorem find_value_of_fraction (h : (20 * a) / (a^2 + 1) = Real.sqrt 2) (h1 : 1 < a) : 
  (14 * a) / (a^2 - 1) = 1 := 
by 
  have h_sqrt : 20 * a = Real.sqrt 2 * a^2 + Real.sqrt 2 := by sorry
  have h_rearrange : Real.sqrt 2 * a^2 - 20 * a + Real.sqrt 2 = 0 := by sorry
  have h_solution : a = 5 * (Real.sqrt 2) + 7 := by sorry
  have h_asquare : a^2 = 99 + 70 * (Real.sqrt 2) := by sorry
  exact sorry

end find_value_of_fraction_l774_77421


namespace ishaan_age_eq_6_l774_77443

-- Variables for ages
variable (I : ℕ) -- Ishaan's current age

-- Constants for ages
def daniel_current_age := 69
def years := 15
def daniel_future_age := daniel_current_age + years

-- Lean theorem statement
theorem ishaan_age_eq_6 
    (h1 : daniel_current_age = 69)
    (h2 : daniel_future_age = 4 * (I + years)) : 
    I = 6 := by
  sorry

end ishaan_age_eq_6_l774_77443


namespace quadratic_sum_terms_l774_77495

theorem quadratic_sum_terms (a b c : ℝ) :
  (∀ x : ℝ, -2 * x^2 + 16 * x - 72 = a * (x + b)^2 + c) → a + b + c = -46 :=
by
  sorry

end quadratic_sum_terms_l774_77495


namespace matrix_exp_1000_l774_77447

-- Define the matrix as a constant
noncomputable def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![2, 1]]

-- The property of matrix exponentiation
theorem matrix_exp_1000 :
  A^1000 = ![![1, 0], ![2000, 1]] :=
by
  sorry

end matrix_exp_1000_l774_77447


namespace no_rearrangement_to_positive_and_negative_roots_l774_77438

theorem no_rearrangement_to_positive_and_negative_roots (a b c : ℝ) :
  (∃ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ a ≠ 0 ∧ b = -a * (x1 + x2) ∧ c = a * x1 * x2) →
  (∃ y1 y2 : ℝ, y1 > 0 ∧ y2 > 0 ∧ a ≠ 0 ∧ b != 0 ∧ c != 0 ∧ 
    (∃ b' c' : ℝ, b' ≠ b ∧ c' ≠ c ∧ 
      b' = -a * (y1 + y2) ∧ c' = a * y1 * y2)) →
  False := by
  sorry

end no_rearrangement_to_positive_and_negative_roots_l774_77438


namespace find_value_l774_77477

-- Define the variables and given conditions
variables (x y z : ℚ)
variables (h1 : 2 * x - y = 4)
variables (h2 : 3 * x + z = 7)
variables (h3 : y = 2 * z)

-- Define the goal to prove
theorem find_value : 6 * x - 3 * y + 3 * z = 51 / 4 := by 
  sorry

end find_value_l774_77477


namespace termite_ridden_fraction_l774_77465

theorem termite_ridden_fraction (T : ℝ)
  (h1 : (3 / 10) * T = 0.1) : T = 1 / 3 :=
by
  -- proof goes here
  sorry

end termite_ridden_fraction_l774_77465


namespace ratio_buses_to_cars_l774_77446

theorem ratio_buses_to_cars (B C : ℕ) (h1 : B = C - 60) (h2 : C = 65) : B / C = 1 / 13 :=
by 
  sorry

end ratio_buses_to_cars_l774_77446


namespace prism_volume_l774_77467

noncomputable def volume_of_prism (x y z : ℝ) : ℝ :=
  x * y * z

theorem prism_volume (x y z : ℝ) (h1 : x * y = 40) (h2 : x * z = 50) (h3 : y * z = 100) :
  volume_of_prism x y z = 100 * Real.sqrt 2 :=
by
  sorry

end prism_volume_l774_77467


namespace hcf_of_three_numbers_l774_77427

theorem hcf_of_three_numbers (a b c : ℕ) (h1 : a + b + c = 60)
  (h2 : Nat.lcm (Nat.lcm a b) c = 180)
  (h3 : (1:ℚ)/a + 1/b + 1/c = 11/120)
  (h4 : a * b * c = 900) :
  Nat.gcd (Nat.gcd a b) c = 5 :=
by
  sorry

end hcf_of_three_numbers_l774_77427


namespace mean_of_all_students_l774_77464

variable (M A m a : ℕ)
variable (M_val : M = 84)
variable (A_val : A = 70)
variable (ratio : m = 3 * a / 4)

theorem mean_of_all_students (M A m a : ℕ) (M_val : M = 84) (A_val : A = 70) (ratio : m = 3 * a / 4) :
    (63 * a + 70 * a) / (7 * a / 4) = 76 := by
  sorry

end mean_of_all_students_l774_77464


namespace range_of_a_l774_77494

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), 3 * a * x + (a^2 - 3 * a + 2) * y - 9 < 0 → (3 * a * x + (a^2 - 3 * a + 2) * y - 9 = 0 → y > 0)) ↔ (1 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l774_77494


namespace probability_of_draw_l774_77474

-- Define the probabilities as constants
def prob_not_lose_xiao_ming : ℚ := 3 / 4
def prob_lose_xiao_dong : ℚ := 1 / 2

-- State the theorem we want to prove
theorem probability_of_draw :
  prob_not_lose_xiao_ming - prob_lose_xiao_dong = 1 / 4 :=
by
  sorry

end probability_of_draw_l774_77474


namespace hoseok_result_l774_77407

theorem hoseok_result :
  ∃ X : ℤ, (X - 46 = 15) ∧ (X - 29 = 32) :=
by
  sorry

end hoseok_result_l774_77407


namespace N_prime_iff_k_eq_2_l774_77426

/-- Define the number N for a given k -/
def N (k : ℕ) : ℕ := (10 ^ (2 * k) - 1) / 99

/-- Statement: Prove that N is prime if and only if k = 2 -/
theorem N_prime_iff_k_eq_2 (k : ℕ) : Prime (N k) ↔ k = 2 := by
  sorry

end N_prime_iff_k_eq_2_l774_77426


namespace factor_expression_l774_77436

noncomputable def expression (x : ℝ) : ℝ := (15 * x^3 + 80 * x - 5) - (-4 * x^3 + 4 * x - 5)

theorem factor_expression (x : ℝ) : expression x = 19 * x * (x^2 + 4) := 
by 
  sorry

end factor_expression_l774_77436


namespace habitable_fraction_of_earth_l774_77444

theorem habitable_fraction_of_earth :
  (1 / 2) * (1 / 4) = 1 / 8 := by
  sorry

end habitable_fraction_of_earth_l774_77444


namespace correct_number_of_true_propositions_l774_77497

noncomputable def true_proposition_count : ℕ := 1

theorem correct_number_of_true_propositions (a b c : ℝ) :
    (∀ a b : ℝ, (a > b) ↔ (a^2 > b^2) = false) →
    (∀ a b : ℝ, (a > b) ↔ (a^3 > b^3) = true) →
    (∀ a b : ℝ, (a > b) → (|a| > |b|) = false) →
    (∀ a b c : ℝ, (a > b) → (a*c^2 ≤ b*c^2) = false) →
    (true_proposition_count = 1) :=
by
  sorry

end correct_number_of_true_propositions_l774_77497


namespace sum_of_distances_condition_l774_77498

theorem sum_of_distances_condition (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| < a) → a > 4 :=
sorry

end sum_of_distances_condition_l774_77498


namespace cube_difference_l774_77489

theorem cube_difference {a b : ℝ} (h1 : a - b = 5) (h2 : a^2 + b^2 = 35) : a^3 - b^3 = 200 :=
sorry

end cube_difference_l774_77489


namespace b_sequence_periodic_l774_77468

theorem b_sequence_periodic (b : ℕ → ℝ)
  (h_rec : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h_b1 : b 1 = 2 + Real.sqrt 3)
  (h_b2021 : b 2021 = 11 + Real.sqrt 3) :
  b 2048 = b 2 :=
sorry

end b_sequence_periodic_l774_77468


namespace time_for_B_and_C_l774_77455

variables (a b c : ℝ)

-- Conditions
axiom cond1 : a = (1 / 2) * b
axiom cond2 : b = 2 * c
axiom cond3 : a + b + c = 1 / 26
axiom cond4 : a + b = 1 / 13
axiom cond5 : a + c = 1 / 39

-- Statement to prove
theorem time_for_B_and_C (a b c : ℝ) (cond1 : a = (1 / 2) * b)
                                      (cond2 : b = 2 * c)
                                      (cond3 : a + b + c = 1 / 26)
                                      (cond4 : a + b = 1 / 13)
                                      (cond5 : a + c = 1 / 39) :
  (1 / (b + c)) = 104 / 3 :=
sorry

end time_for_B_and_C_l774_77455


namespace amount_saved_percentage_l774_77482

variable (S : ℝ) 

-- Condition: Last year, Sandy saved 7% of her annual salary
def amount_saved_last_year (S : ℝ) : ℝ := 0.07 * S

-- Condition: This year, she made 15% more money than last year
def salary_this_year (S : ℝ) : ℝ := 1.15 * S

-- Condition: This year, she saved 10% of her salary
def amount_saved_this_year (S : ℝ) : ℝ := 0.10 * salary_this_year S

-- The statement to prove
theorem amount_saved_percentage (S : ℝ) : 
  amount_saved_this_year S = 1.642857 * amount_saved_last_year S :=
by 
  sorry

end amount_saved_percentage_l774_77482


namespace arithmetic_seq_proof_l774_77471

open Nat

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a d : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (n : ℕ) : ℤ :=
n * (a 1) + n * (n - 1) / 2 * d

theorem arithmetic_seq_proof (a : ℕ → ℤ) (d : ℤ)
  (h1 : arithmetic_seq a d)
  (h2 : a 2 = 0)
  (h3 : sum_of_arithmetic_seq a d 3 + sum_of_arithmetic_seq a d 4 = 6) :
  a 5 + a 6 = 21 :=
sorry

end arithmetic_seq_proof_l774_77471


namespace work_days_B_l774_77402

theorem work_days_B (A_days B_days : ℕ) (hA : A_days = 12) (hTogether : (1/12 + 1/A_days) = (1/8)) : B_days = 24 := 
by
  revert hTogether -- reversing to tackle proof
  sorry

end work_days_B_l774_77402


namespace find_k_l774_77496

noncomputable def arithmetic_sum (n : ℕ) (a1 d : ℚ) : ℚ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem find_k 
  (a1 d : ℚ) (k : ℕ)
  (h1 : arithmetic_sum (k - 2) a1 d = -4)
  (h2 : arithmetic_sum k a1 d = 0)
  (h3 : arithmetic_sum (k + 2) a1 d = 8) :
  k = 6 :=
by
  sorry

end find_k_l774_77496


namespace stickers_per_student_l774_77429

theorem stickers_per_student 
  (gold_stickers : ℕ) 
  (silver_stickers : ℕ) 
  (bronze_stickers : ℕ) 
  (students : ℕ)
  (h1 : gold_stickers = 50)
  (h2 : silver_stickers = 2 * gold_stickers)
  (h3 : bronze_stickers = silver_stickers - 20)
  (h4 : students = 5) : 
  (gold_stickers + silver_stickers + bronze_stickers) / students = 46 :=
by
  sorry

end stickers_per_student_l774_77429


namespace freeze_time_l774_77424

theorem freeze_time :
  ∀ (minutes_per_smoothie total_minutes num_smoothies freeze_time: ℕ),
    minutes_per_smoothie = 3 →
    total_minutes = 55 →
    num_smoothies = 5 →
    freeze_time = total_minutes - (num_smoothies * minutes_per_smoothie) →
    freeze_time = 40 :=
by
  intros minutes_per_smoothie total_minutes num_smoothies freeze_time
  intros H1 H2 H3 H4
  subst H1
  subst H2
  subst H3
  subst H4
  sorry

end freeze_time_l774_77424


namespace percentage_of_red_shirts_l774_77476

variable (total_students : ℕ) (blue_percent green_percent : ℕ) (other_students : ℕ)
  (H_total : total_students = 800)
  (H_blue : blue_percent = 45)
  (H_green : green_percent = 15)
  (H_other : other_students = 136)
  (H_blue_students : 0.45 * 800 = 360)
  (H_green_students : 0.15 * 800 = 120)
  (H_sum : 360 + 120 + 136 = 616)
  
theorem percentage_of_red_shirts :
  ((total_students - (360 + 120 + other_students)) / total_students) * 100 = 23 := 
by {
  sorry
}

end percentage_of_red_shirts_l774_77476


namespace nesting_rectangles_exists_l774_77469

theorem nesting_rectangles_exists :
  ∀ (rectangles : List (ℕ × ℕ)), rectangles.length = 101
    ∧ (∀ r ∈ rectangles, r.fst ≤ 100 ∧ r.snd ≤ 100) 
    → ∃ (A B C : ℕ × ℕ), A ∈ rectangles ∧ B ∈ rectangles ∧ C ∈ rectangles 
    ∧ (A.fst < B.fst ∧ A.snd < B.snd) 
    ∧ (B.fst < C.fst ∧ B.snd < C.snd) := 
by sorry

end nesting_rectangles_exists_l774_77469


namespace min_sum_of_squares_l774_77451

theorem min_sum_of_squares (a b c d : ℤ) (h1 : a^2 ≠ b^2 ∧ a^2 ≠ c^2 ∧ a^2 ≠ d^2 ∧ b^2 ≠ c^2 ∧ b^2 ≠ d^2 ∧ c^2 ≠ d^2)
                           (h2 : (a * b + c * d)^2 + (a * d - b * c)^2 = 2004) :
  a^2 + b^2 + c^2 + d^2 = 2 * Int.sqrt 2004 :=
sorry

end min_sum_of_squares_l774_77451


namespace _l774_77457

@[simp] theorem upper_base_length (ABCD is_trapezoid: Boolean)
  (point_M: Boolean)
  (perpendicular_DM_AB: Boolean)
  (MC_eq_CD: Boolean)
  (AD_eq_d: ℝ)
  : BC = d / 2 := sorry

end _l774_77457


namespace complementary_event_l774_77462

def car_a_selling_well : Prop := sorry
def car_b_selling_poorly : Prop := sorry

def event_A : Prop := car_a_selling_well ∧ car_b_selling_poorly
def event_complement (A : Prop) : Prop := ¬A

theorem complementary_event :
  event_complement event_A = (¬car_a_selling_well ∨ ¬car_b_selling_poorly) :=
by
  sorry

end complementary_event_l774_77462


namespace age_difference_l774_77441

variable (y m e : ℕ)

theorem age_difference (h1 : m = y + 3) (h2 : e = 3 * y) (h3 : e = 15) : 
  ∃ x, e = y + m + x ∧ x = 2 := by
  sorry

end age_difference_l774_77441


namespace log_base_change_l774_77408

theorem log_base_change (a b : ℝ) (h₁ : Real.log 2 / Real.log 10 = a) (h₂ : Real.log 3 / Real.log 10 = b) :
    Real.log 18 / Real.log 5 = (a + 2 * b) / (1 - a) := by
  sorry

end log_base_change_l774_77408


namespace train_speed_l774_77479

theorem train_speed (distance : ℝ) (time_minutes : ℝ) (time_conversion_factor : ℝ) (expected_speed : ℝ) (h_time_conversion : time_conversion_factor = 1 / 60) (h_time : time_minutes / 60 = 0.5) (h_distance : distance = 51) (h_expected_speed : expected_speed = 102) : distance / (time_minutes / 60) = expected_speed :=
by 
  sorry

end train_speed_l774_77479


namespace sequence_is_increasing_l774_77423

variable (a_n : ℕ → ℝ)

def sequence_positive_numbers (a_n : ℕ → ℝ) : Prop :=
∀ n, 0 < a_n n

def sequence_condition (a_n : ℕ → ℝ) : Prop :=
∀ n, a_n (n + 1) = 2 * a_n n

theorem sequence_is_increasing 
  (h1 : sequence_positive_numbers a_n) 
  (h2 : sequence_condition a_n) : 
  ∀ n, a_n (n + 1) > a_n n :=
by
  sorry

end sequence_is_increasing_l774_77423


namespace sugar_amount_l774_77406

-- Definitions based on conditions
variables (S F B C : ℝ) -- S = amount of sugar, F = amount of flour, B = amount of baking soda, C = amount of chocolate chips

-- Conditions
def ratio_sugar_flour (S F : ℝ) : Prop := S / F = 5 / 4
def ratio_flour_baking_soda (F B : ℝ) : Prop := F / B = 10 / 1
def ratio_baking_soda_chocolate_chips (B C : ℝ) : Prop := B / C = 3 / 2
def new_ratio_flour_baking_soda_chocolate_chips (F B C : ℝ) : Prop :=
  F / (B + 120) = 16 / 3 ∧ F / (C + 50) = 16 / 2

-- Prove that the current amount of sugar is 1714 pounds
theorem sugar_amount (S F B C : ℝ) (h1 : ratio_sugar_flour S F)
  (h2 : ratio_flour_baking_soda F B) (h3 : ratio_baking_soda_chocolate_chips B C)
  (h4 : new_ratio_flour_baking_soda_chocolate_chips F B C) : 
  S = 1714 :=
sorry

end sugar_amount_l774_77406


namespace sum_of_y_neg_l774_77410

-- Define the conditions from the problem
def condition1 (x y : ℝ) : Prop := x + y = 7
def condition2 (x z : ℝ) : Prop := x * z = -180
def condition3 (x y z : ℝ) : Prop := (x + y + z)^2 = 4

-- Define the main theorem to prove
theorem sum_of_y_neg (x y z : ℝ) (S : ℝ) :
  (condition1 x y) ∧ (condition2 x z) ∧ (condition3 x y z) →
  (S = (-29) + (-13)) →
  -S = 42 :=
by
  sorry

end sum_of_y_neg_l774_77410


namespace check_blank_value_l774_77452

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

end check_blank_value_l774_77452


namespace find_CD_squared_l774_77450

noncomputable def first_circle (x y : ℝ) : Prop := (x - 5)^2 + y^2 = 25
noncomputable def second_circle (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 25

theorem find_CD_squared : ∃ C D : ℝ × ℝ, 
  (first_circle C.1 C.2 ∧ second_circle C.1 C.2) ∧ 
  (first_circle D.1 D.2 ∧ second_circle D.1 D.2) ∧ 
  (C ≠ D) ∧ 
  ((D.1 - C.1)^2 + (D.2 - C.2)^2 = 50) :=
by
  sorry

end find_CD_squared_l774_77450


namespace find_a_from_complex_condition_l774_77456

theorem find_a_from_complex_condition (a : ℝ) (x y : ℝ) 
  (h : x = -1 ∧ y = -2 * a)
  (h_line : x - y = 0) : a = 1 / 2 :=
by
  sorry

end find_a_from_complex_condition_l774_77456


namespace incorrect_statement_g2_l774_77445

def g (x : ℚ) : ℚ := (2 * x + 3) / (x - 2)

theorem incorrect_statement_g2 : g 2 ≠ 0 := by
  sorry

end incorrect_statement_g2_l774_77445


namespace average_salary_l774_77414

theorem average_salary
  (num_technicians : ℕ) (avg_salary_technicians : ℝ)
  (num_other_workers : ℕ) (avg_salary_other_workers : ℝ)
  (total_num_workers : ℕ) (avg_salary_all_workers : ℝ) :
  num_technicians = 7 →
  avg_salary_technicians = 14000 →
  num_other_workers = total_num_workers - num_technicians →
  avg_salary_other_workers = 6000 →
  total_num_workers = 28 →
  avg_salary_all_workers = (num_technicians * avg_salary_technicians + num_other_workers * avg_salary_other_workers) / total_num_workers →
  avg_salary_all_workers = 8000 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_salary_l774_77414


namespace width_of_vessel_is_5_l774_77493

open Real

noncomputable def width_of_vessel : ℝ :=
  let edge := 5
  let rise := 2.5
  let base_length := 10
  let volume_cube := edge ^ 3
  let volume_displaced := volume_cube
  let width := volume_displaced / (base_length * rise)
  width

theorem width_of_vessel_is_5 :
  width_of_vessel = 5 := by
    sorry

end width_of_vessel_is_5_l774_77493


namespace smallest_three_digit_times_largest_single_digit_l774_77473

theorem smallest_three_digit_times_largest_single_digit :
  let x := 100
  let y := 9
  ∃ z : ℕ, z = x * y ∧ 100 ≤ z ∧ z < 1000 :=
by
  let x := 100
  let y := 9
  use x * y
  sorry

end smallest_three_digit_times_largest_single_digit_l774_77473


namespace evaluate_polynomial_l774_77478

-- Define the polynomial function
def polynomial (x : ℝ) : ℝ := x^3 + 3 * x^2 - 9 * x - 5

-- Define the condition: x is the positive root of the quadratic equation
def is_positive_root_of_quadratic (x : ℝ) : Prop := x > 0 ∧ x^2 + 3 * x - 9 = 0

-- The main theorem stating the polynomial evaluates to 22 given the condition
theorem evaluate_polynomial {x : ℝ} (h : is_positive_root_of_quadratic x) : polynomial x = 22 := 
by 
  sorry

end evaluate_polynomial_l774_77478


namespace smallest_x_for_perfect_cube_l774_77499

theorem smallest_x_for_perfect_cube (x : ℕ) (M : ℤ) (hx : x > 0) (hM : ∃ M, 1680 * x = M^3) : x = 44100 :=
sorry

end smallest_x_for_perfect_cube_l774_77499


namespace initial_position_of_M_l774_77412

theorem initial_position_of_M :
  ∃ x : ℤ, (x + 7) - 4 = 0 ∧ x = -3 :=
by sorry

end initial_position_of_M_l774_77412
