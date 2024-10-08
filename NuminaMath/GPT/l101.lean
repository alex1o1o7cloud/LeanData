import Mathlib

namespace math_problem_l101_101183

theorem math_problem 
  (a b : ℂ) (n : ℕ) (h1 : a + b = 0) (h2 : a ≠ 0) : 
  a^(2*n + 1) + b^(2*n + 1) = 0 := 
by 
  sorry

end math_problem_l101_101183


namespace ellipse_eccentricity_l101_101276

def ellipse {a : ℝ} (h : a^2 - 4 = 4) : Prop :=
  ∃ c e : ℝ, (c = 2) ∧ (e = c / a) ∧ (e = (Real.sqrt 2) / 2)

theorem ellipse_eccentricity (a : ℝ) (h : a^2 - 4 = 4) : 
  ellipse h :=
by
  sorry

end ellipse_eccentricity_l101_101276


namespace rhombus_diagonal_length_l101_101314

-- Define a rhombus with one diagonal of 10 cm and a perimeter of 52 cm.
theorem rhombus_diagonal_length (d : ℝ) 
  (h1 : ∃ a b c : ℝ, a = 10 ∧ b = d ∧ c = 13) -- The diagonals and side of rhombus.
  (h2 : 52 = 4 * c) -- The perimeter condition.
  (h3 : c^2 = (d/2)^2 + (10/2)^2) -- The relationship from Pythagorean theorem.
  : d = 24 :=
by
  sorry

end rhombus_diagonal_length_l101_101314


namespace all_buses_have_same_stoppage_time_l101_101092

-- Define the constants for speeds without and with stoppages
def speed_without_stoppage_bus1 := 50
def speed_without_stoppage_bus2 := 60
def speed_without_stoppage_bus3 := 70

def speed_with_stoppage_bus1 := 40
def speed_with_stoppage_bus2 := 48
def speed_with_stoppage_bus3 := 56

-- Stating the stoppage time per hour for each bus
def stoppage_time_per_hour (speed_without : ℕ) (speed_with : ℕ) : ℚ :=
  1 - (speed_with : ℚ) / (speed_without : ℚ)

-- Theorem to prove the stoppage time correctness
theorem all_buses_have_same_stoppage_time :
  stoppage_time_per_hour speed_without_stoppage_bus1 speed_with_stoppage_bus1 = 0.2 ∧
  stoppage_time_per_hour speed_without_stoppage_bus2 speed_with_stoppage_bus2 = 0.2 ∧
  stoppage_time_per_hour speed_without_stoppage_bus3 speed_with_stoppage_bus3 = 0.2 :=
by
  sorry  -- Proof to be completed

end all_buses_have_same_stoppage_time_l101_101092


namespace gcd_is_12_l101_101229

noncomputable def gcd_problem (b : ℤ) : Prop :=
  b % 2027 = 0 → Int.gcd (b^2 + 7*b + 18) (b + 6) = 12

-- Now, let's state the theorem
theorem gcd_is_12 (b : ℤ) : gcd_problem b :=
  sorry

end gcd_is_12_l101_101229


namespace remainder_sum_div_6_l101_101454

theorem remainder_sum_div_6 (n : ℤ) : ((5 - n) + (n + 4)) % 6 = 3 :=
by
  -- Placeholder for the actual proof
  sorry

end remainder_sum_div_6_l101_101454


namespace age_of_30th_employee_l101_101074

theorem age_of_30th_employee :
  let n := 30
  let group1_avg_age := 24
  let group1_count := 10
  let group2_avg_age := 30
  let group2_count := 12
  let group3_avg_age := 35
  let group3_count := 7
  let remaining_avg_age := 29

  let group1_total_age := group1_count * group1_avg_age
  let group2_total_age := group2_count * group2_avg_age
  let group3_total_age := group3_count * group3_avg_age
  let total_age_29 := group1_total_age + group2_total_age + group3_total_age
  let total_age_30 := remaining_avg_age * n

  let age_30th_employee := total_age_30 - total_age_29

  age_30th_employee = 25 :=
by
  let n := 30
  let group1_avg_age := 24
  let group1_count := 10
  let group2_avg_age := 30
  let group2_count := 12
  let group3_avg_age := 35
  let group3_count := 7
  let remaining_avg_age := 29

  let group1_total_age := group1_count * group1_avg_age
  let group2_total_age := group2_count * group2_avg_age
  let group3_total_age := group3_count * group3_avg_age
  let total_age_29 := group1_total_age + group2_total_age + group3_total_age
  let total_age_30 := remaining_avg_age * n

  let age_30th_employee := total_age_30 - total_age_29

  have h : age_30th_employee = 25 := sorry
  exact h

end age_of_30th_employee_l101_101074


namespace hexagon_side_squares_sum_l101_101384

variables {P Q R P' Q' R' A B C D E F : Type}
variables (a1 a2 a3 b1 b2 b3 : ℝ)
variables (h_eq_triangles : congruent (triangle P Q R) (triangle P' Q' R'))
variables (h_sides : 
  AB = a1 ∧ BC = b1 ∧ CD = a2 ∧ 
  DE = b2 ∧ EF = a3 ∧ FA = b3)
  
theorem hexagon_side_squares_sum :
  a1^2 + a2^2 + a3^2 = b1^2 + b2^2 + b3^2 :=
sorry

end hexagon_side_squares_sum_l101_101384


namespace sin_cos_inequality_l101_101504

open Real

theorem sin_cos_inequality (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * π) :
  (sin (x - π / 6) > cos x) ↔ (π / 3 < x ∧ x < 4 * π / 3) :=
by sorry

end sin_cos_inequality_l101_101504


namespace supplement_complement_diff_l101_101341

theorem supplement_complement_diff (α : ℝ) : (180 - α) - (90 - α) = 90 := 
by
  sorry

end supplement_complement_diff_l101_101341


namespace avgPercentageSpentOnFoodCorrect_l101_101032

-- Definitions for given conditions
def JanuaryIncome : ℕ := 3000
def JanuaryPetrolExpenditure : ℕ := 300
def JanuaryHouseRentPercentage : ℕ := 14
def JanuaryClothingPercentage : ℕ := 10
def JanuaryUtilityBillsPercentage : ℕ := 5
def FebruaryIncome : ℕ := 4000
def FebruaryPetrolExpenditure : ℕ := 400
def FebruaryHouseRentPercentage : ℕ := 14
def FebruaryClothingPercentage : ℕ := 10
def FebruaryUtilityBillsPercentage : ℕ := 5

-- Calculate percentage spent on food over January and February
noncomputable def avgPercentageSpentOnFood : ℝ :=
  let totalIncome := (JanuaryIncome + FebruaryIncome: ℝ)
  let totalFoodExpenditure :=
    let remainingJan := (JanuaryIncome - JanuaryPetrolExpenditure: ℝ) 
                         - (JanuaryHouseRentPercentage / 100 * (JanuaryIncome - JanuaryPetrolExpenditure: ℝ))
                         - (JanuaryClothingPercentage / 100 * JanuaryIncome)
                         - (JanuaryUtilityBillsPercentage / 100 * JanuaryIncome)
    let remainingFeb := (FebruaryIncome - FebruaryPetrolExpenditure: ℝ)
                         - (FebruaryHouseRentPercentage / 100 * (FebruaryIncome - FebruaryPetrolExpenditure: ℝ))
                         - (FebruaryClothingPercentage / 100 * FebruaryIncome)
                         - (FebruaryUtilityBillsPercentage / 100 * FebruaryIncome)
    remainingJan + remainingFeb
  (totalFoodExpenditure / totalIncome) * 100

theorem avgPercentageSpentOnFoodCorrect : avgPercentageSpentOnFood = 62.4 := by
  sorry

end avgPercentageSpentOnFoodCorrect_l101_101032


namespace no_other_integer_solutions_l101_101133

theorem no_other_integer_solutions :
  (∀ (x : ℤ), (x + 1) ^ 3 + (x + 2) ^ 3 + (x + 3) ^ 3 = (x + 4) ^ 3 → x = 2) := 
by sorry

end no_other_integer_solutions_l101_101133


namespace rotated_square_vertical_distance_is_correct_l101_101900

-- Define a setup with four 1-inch squares in a straight line
-- and the second square rotated 45 degrees around its center

-- Noncomputable setup
noncomputable def rotated_square_vert_distance : ℝ :=
  let side_length := 1
  let diagonal := side_length * Real.sqrt 2
  -- Calculate the required vertical distance according to given conditions
  Real.sqrt 2 + side_length / 2

-- Theorem statement confirming the calculated vertical distance
theorem rotated_square_vertical_distance_is_correct :
  rotated_square_vert_distance = Real.sqrt 2 + 1 / 2 :=
by
  sorry

end rotated_square_vertical_distance_is_correct_l101_101900


namespace coeff_of_quadratic_term_eq_neg5_l101_101758

theorem coeff_of_quadratic_term_eq_neg5 (a b c : ℝ) (h_eq : -5 * x^2 + 5 * x + 6 = a * x^2 + b * x + c) :
  a = -5 :=
by
  sorry

end coeff_of_quadratic_term_eq_neg5_l101_101758


namespace Jungkook_fewest_erasers_l101_101644

-- Define the number of erasers each person has.
def Jungkook_erasers : ℕ := 6
def Jimin_erasers : ℕ := Jungkook_erasers + 4
def Seokjin_erasers : ℕ := Jimin_erasers - 3

-- Prove that Jungkook has the fewest erasers.
theorem Jungkook_fewest_erasers : Jungkook_erasers < Jimin_erasers ∧ Jungkook_erasers < Seokjin_erasers :=
by
  -- Proof goes here
  sorry

end Jungkook_fewest_erasers_l101_101644


namespace shirt_cost_is_43_l101_101385

def pantsCost : ℕ := 140
def tieCost : ℕ := 15
def totalPaid : ℕ := 200
def changeReceived : ℕ := 2

def totalCostWithoutShirt := totalPaid - changeReceived
def totalCostWithPantsAndTie := pantsCost + tieCost
def shirtCost := totalCostWithoutShirt - totalCostWithPantsAndTie

theorem shirt_cost_is_43 : shirtCost = 43 := by
  have h1 : totalCostWithoutShirt = 198 := by rfl
  have h2 : totalCostWithPantsAndTie = 155 := by rfl
  have h3 : shirtCost = totalCostWithoutShirt - totalCostWithPantsAndTie := by rfl
  rw [h1, h2] at h3
  exact h3

end shirt_cost_is_43_l101_101385


namespace min_value_frac_sum_l101_101167

theorem min_value_frac_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 4) : 
  (4 / a^2 + 1 / b^2) ≥ 9 / 4 :=
by
  sorry

end min_value_frac_sum_l101_101167


namespace last_number_with_35_zeros_l101_101983

def count_zeros (n : Nat) : Nat :=
  if n = 0 then 1
  else if n < 10 then 0
  else count_zeros (n / 10) + count_zeros (n % 10)

def total_zeros_written (upto : Nat) : Nat :=
  (List.range (upto + 1)).foldl (λ acc n => acc + count_zeros n) 0

theorem last_number_with_35_zeros : ∃ n, total_zeros_written n = 35 ∧ ∀ m, m > n → total_zeros_written m ≠ 35 :=
by
  let x := 204
  have h1 : total_zeros_written x = 35 := sorry
  have h2 : ∀ m, m > x → total_zeros_written m ≠ 35 := sorry
  existsi x
  exact ⟨h1, h2⟩

end last_number_with_35_zeros_l101_101983


namespace price_of_each_sundae_l101_101105

theorem price_of_each_sundae
  (num_ice_cream_bars : ℕ := 125) 
  (num_sundaes : ℕ := 125) 
  (total_price : ℝ := 225)
  (price_per_ice_cream_bar : ℝ := 0.60) :
  ∃ (price_per_sundae : ℝ), price_per_sundae = 1.20 := 
by
  -- Variables for costs of ice-cream bars and sundaes' total cost
  let cost_ice_cream_bars := num_ice_cream_bars * price_per_ice_cream_bar
  let total_cost_sundaes := total_price - cost_ice_cream_bars
  let price_per_sundae := total_cost_sundaes / num_sundaes
  use price_per_sundae
  sorry

end price_of_each_sundae_l101_101105


namespace annual_interest_rate_is_correct_l101_101609

-- Define conditions
def principal : ℝ := 900
def finalAmount : ℝ := 992.25
def compoundingPeriods : ℕ := 2
def timeYears : ℕ := 1

-- Compound interest formula
def compound_interest (P A r : ℝ) (n t : ℕ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

-- Statement to prove
theorem annual_interest_rate_is_correct :
  ∃ r : ℝ, compound_interest principal finalAmount r compoundingPeriods timeYears ∧ r = 0.10 :=
by 
  sorry

end annual_interest_rate_is_correct_l101_101609


namespace problem_f8_f2018_l101_101857

theorem problem_f8_f2018 (f : ℕ → ℝ) (h₀ : ∀ n, f (n + 3) = (f n - 1) / (f n + 1)) 
  (h₁ : f 1 ≠ 0) (h₂ : f 1 ≠ 1) (h₃ : f 1 ≠ -1) : 
  f 8 * f 2018 = -1 :=
sorry

end problem_f8_f2018_l101_101857


namespace range_of_m_l101_101168

theorem range_of_m (x1 x2 y1 y2 m : ℝ) (h1 : x1 > x2) (h2 : y1 > y2) (h3 : y1 = (m-2)*x1) (h4 : y2 = (m-2)*x2) : m > 2 :=
by sorry

end range_of_m_l101_101168


namespace segment_parametrization_pqrs_l101_101186

theorem segment_parametrization_pqrs :
  ∃ (p q r s : ℤ), 
    q = 1 ∧ 
    s = -3 ∧ 
    p + q = 6 ∧ 
    r + s = 4 ∧ 
    p^2 + q^2 + r^2 + s^2 = 84 :=
by
  use 5, 1, 7, -3
  sorry

end segment_parametrization_pqrs_l101_101186


namespace parity_of_function_parity_neither_odd_nor_even_l101_101234

def f (x p : ℝ) : ℝ := x * |x| + p * x^2

theorem parity_of_function (p : ℝ) :
  (∀ x : ℝ, f x p = - f (-x) p) ↔ p = 0 :=
by
  sorry

theorem parity_neither_odd_nor_even (p : ℝ) :
  (∀ x : ℝ, f x p ≠ f (-x) p) ∧ (∀ x : ℝ, f x p ≠ - f (-x) p) ↔ p ≠ 0 :=
by
  sorry

end parity_of_function_parity_neither_odd_nor_even_l101_101234


namespace factorization_result_l101_101055

theorem factorization_result (a b : ℤ) (h1 : 25 * x^2 - 160 * x - 336 = (5 * x + a) * (5 * x + b)) :
  a + 2 * b = 20 :=
by
  sorry

end factorization_result_l101_101055


namespace total_people_wearing_hats_l101_101545

variable (total_adults : ℕ) (total_children : ℕ)
variable (half_adults : ℕ) (women : ℕ) (men : ℕ)
variable (women_with_hats : ℕ) (men_with_hats : ℕ)
variable (children_with_hats : ℕ)
variable (total_with_hats : ℕ)

-- Given conditions
def conditions : Prop :=
  total_adults = 1800 ∧
  total_children = 200 ∧
  half_adults = total_adults / 2 ∧
  women = half_adults ∧
  men = half_adults ∧
  women_with_hats = (25 * women) / 100 ∧
  men_with_hats = (12 * men) / 100 ∧
  children_with_hats = (10 * total_children) / 100 ∧
  total_with_hats = women_with_hats + men_with_hats + children_with_hats

-- Proof goal
theorem total_people_wearing_hats : conditions total_adults total_children half_adults women men women_with_hats men_with_hats children_with_hats total_with_hats → total_with_hats = 353 :=
by
  intros h
  sorry

end total_people_wearing_hats_l101_101545


namespace rectangle_sides_l101_101643

theorem rectangle_sides (x y : ℕ) (h_diff : x ≠ y) (h_eq : x * y = 2 * x + 2 * y) : 
  (x = 3 ∧ y = 6) ∨ (x = 6 ∧ y = 3) :=
sorry

end rectangle_sides_l101_101643


namespace true_for_2_and_5_l101_101635

theorem true_for_2_and_5 (x : ℝ) : ((x - 2) * (x - 5) = 0) ↔ (x = 2 ∨ x = 5) :=
by
  sorry

end true_for_2_and_5_l101_101635


namespace average_rainfall_in_normal_year_l101_101665

def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def rainfall_difference : ℕ := 58

theorem average_rainfall_in_normal_year :
  (total_rainfall_this_year + rainfall_difference) = 140 :=
by
  sorry

end average_rainfall_in_normal_year_l101_101665


namespace tan_product_l101_101682

theorem tan_product : (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 :=
by
  sorry

end tan_product_l101_101682


namespace four_r_eq_sum_abcd_l101_101968

theorem four_r_eq_sum_abcd (a b c d r : ℤ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : (r - a) * (r - b) * (r - c) * (r - d) = 4) : 
  4 * r = a + b + c + d :=
by 
  sorry

end four_r_eq_sum_abcd_l101_101968


namespace Sara_sister_notebooks_l101_101304

theorem Sara_sister_notebooks :
  let initial_notebooks := 4 
  let ordered_notebooks := (3 / 2) * initial_notebooks -- 150% more notebooks
  let notebooks_after_order := initial_notebooks + ordered_notebooks
  let notebooks_after_loss := notebooks_after_order - 2 -- lost 2 notebooks
  let sold_notebooks := (1 / 4) * notebooks_after_loss -- sold 25% of remaining notebooks
  let notebooks_after_sales := notebooks_after_loss - sold_notebooks
  let notebooks_after_giveaway := notebooks_after_sales - 3 -- gave away 3 notebooks
  notebooks_after_giveaway = 3 := 
by {
  sorry
}

end Sara_sister_notebooks_l101_101304


namespace sum_of_three_rel_prime_pos_integers_l101_101448

theorem sum_of_three_rel_prime_pos_integers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h_rel_prime_ab : Nat.gcd a b = 1) (h_rel_prime_ac : Nat.gcd a c = 1) (h_rel_prime_bc : Nat.gcd b c = 1)
  (h_product : a * b * c = 2700) :
  a + b + c = 56 := by
  sorry

end sum_of_three_rel_prime_pos_integers_l101_101448


namespace find_pairs_l101_101289

theorem find_pairs (m n : ℕ) (h1 : 1 < m) (h2 : 1 < n) (h3 : (mn - 1) ∣ (n^3 - 1)) :
  ∃ k : ℕ, 1 < k ∧ ((m = k ∧ n = k^2) ∨ (m = k^2 ∧ n = k)) :=
sorry

end find_pairs_l101_101289


namespace compare_points_on_line_l101_101260

theorem compare_points_on_line (m n : ℝ) 
  (hA : ∃ (x : ℝ), x = -3 ∧ m = -2 * x + 1) 
  (hB : ∃ (x : ℝ), x = 2 ∧ n = -2 * x + 1) : 
  m > n :=
by sorry

end compare_points_on_line_l101_101260


namespace find_analytical_expression_of_f_l101_101996

variable (f : ℝ → ℝ)

theorem find_analytical_expression_of_f
  (h : ∀ x : ℝ, f (2 * x + 1) = 4 * x^2 + 4 * x) :
  ∀ x : ℝ, f x = x^2 - 1 :=
sorry

end find_analytical_expression_of_f_l101_101996


namespace parallel_lines_a_l101_101559

theorem parallel_lines_a (a : ℝ) :
  ((∃ k : ℝ, (a + 2) / 6 = k ∧ (a + 3) / (2 * a - 1) = k) ∧ 
   ¬ ((-5 / -5) = ((a + 2) / 6)) ∧ ((a + 3) / (2 * a - 1) = (-5 / -5))) →
  a = -5 / 2 :=
by
  sorry

end parallel_lines_a_l101_101559


namespace koala_fiber_intake_l101_101620

theorem koala_fiber_intake (r a : ℝ) (hr : r = 0.20) (ha : a = 8) : (a / r) = 40 :=
by
  sorry

end koala_fiber_intake_l101_101620


namespace stock_percentage_calculation_l101_101405

noncomputable def stock_percentage (investment_amount stock_price annual_income : ℝ) : ℝ :=
  (annual_income / (investment_amount / stock_price) / stock_price) * 100

theorem stock_percentage_calculation :
  stock_percentage 6800 136 1000 = 14.71 :=
by
  sorry

end stock_percentage_calculation_l101_101405


namespace avg_remaining_two_l101_101475

-- Defining the given conditions
variable (six_num_avg : ℝ) (group1_avg : ℝ) (group2_avg : ℝ)

-- Defining the known values
axiom avg_val : six_num_avg = 3.95
axiom avg_group1 : group1_avg = 3.6
axiom avg_group2 : group2_avg = 3.85

-- Stating the problem to prove that the average of the remaining 2 numbers is 4.4
theorem avg_remaining_two (h : six_num_avg = 3.95) 
                           (h1: group1_avg = 3.6)
                           (h2: group2_avg = 3.85) : 
  4.4 = ((six_num_avg * 6) - (group1_avg * 2 + group2_avg * 2)) / 2 := 
sorry

end avg_remaining_two_l101_101475


namespace max_students_distribution_l101_101422

-- Define the four quantities
def pens : ℕ := 4261
def pencils : ℕ := 2677
def erasers : ℕ := 1759
def notebooks : ℕ := 1423

-- Prove that the greatest common divisor (GCD) of these four quantities is 1
theorem max_students_distribution : Nat.gcd (Nat.gcd (Nat.gcd pens pencils) erasers) notebooks = 1 :=
by
  sorry

end max_students_distribution_l101_101422


namespace min_value_expr_l101_101282

/-- Given x > y > 0 and x^2 - y^2 = 1, we need to prove that the minimum value of 2x^2 + 3y^2 - 4xy is 1. -/
theorem min_value_expr {x y : ℝ} (h1 : x > y) (h2 : y > 0) (h3 : x^2 - y^2 = 1) :
  2 * x^2 + 3 * y^2 - 4 * x * y = 1 :=
sorry

end min_value_expr_l101_101282


namespace complement_of_A_in_I_is_246_l101_101381

def universal_set : Set ℕ := {1, 2, 3, 4, 5, 6}
def set_A : Set ℕ := {1, 3, 5}
def complement_A_in_I : Set ℕ := {2, 4, 6}

theorem complement_of_A_in_I_is_246 :
  (universal_set \ set_A) = complement_A_in_I :=
  by sorry

end complement_of_A_in_I_is_246_l101_101381


namespace b_divisible_by_8_l101_101259

variable (b : ℕ) (n : ℕ)
variable (hb_even : b % 2 = 0) (hb_pos : b > 0) (hn_gt1 : n > 1)
variable (h_square : ∃ k : ℕ, k^2 = (b^n - 1) / (b - 1))

theorem b_divisible_by_8 : b % 8 = 0 :=
by
  sorry

end b_divisible_by_8_l101_101259


namespace profit_equation_correct_l101_101494

theorem profit_equation_correct (x : ℝ) : 
  let original_selling_price := 36
  let purchase_price := 20
  let original_sales_volume := 200
  let price_increase_effect := 5
  let desired_profit := 1200
  let original_profit_per_unit := original_selling_price - purchase_price
  let new_selling_price := original_selling_price + x
  let new_sales_volume := original_sales_volume - price_increase_effect * x
  (original_profit_per_unit + x) * new_sales_volume = desired_profit :=
sorry

end profit_equation_correct_l101_101494


namespace milk_drinks_on_weekdays_l101_101506

-- Defining the number of boxes Lolita drinks on a weekday as a variable W
variable (W : ℕ)

-- Condition: Lolita drinks 30 boxes of milk per week.
axiom total_milk_per_week : 5 * W + 2 * W + 3 * W = 30

-- Proof (Statement) that Lolita drinks 15 boxes of milk on weekdays.
theorem milk_drinks_on_weekdays : 5 * W = 15 :=
by {
  -- Use the given axiom to derive the solution
  sorry
}

end milk_drinks_on_weekdays_l101_101506


namespace joan_original_seashells_l101_101670

-- Definitions based on the conditions
def seashells_left : ℕ := 27
def seashells_given_away : ℕ := 43

-- Theorem statement
theorem joan_original_seashells : 
  seashells_left + seashells_given_away = 70 := 
by
  sorry

end joan_original_seashells_l101_101670


namespace john_moves_correct_total_weight_l101_101442

noncomputable def johns_total_weight_moved : ℝ := 5626.398

theorem john_moves_correct_total_weight :
  let initial_back_squat : ℝ := 200
  let back_squat_increase : ℝ := 50
  let front_squat_ratio : ℝ := 0.8
  let back_squat_set_increase : ℝ := 0.05
  let front_squat_ratio_increase : ℝ := 0.04
  let front_squat_effort : ℝ := 0.9
  let deadlift_ratio : ℝ := 1.2
  let deadlift_effort : ℝ := 0.85
  let deadlift_set_increase : ℝ := 0.03
  let updated_back_squat := (initial_back_squat + back_squat_increase)
  let back_squat_set_1 := updated_back_squat
  let back_squat_set_2 := back_squat_set_1 * (1 + back_squat_set_increase)
  let back_squat_set_3 := back_squat_set_2 * (1 + back_squat_set_increase)
  let back_squat_total := 3 * (back_squat_set_1 + back_squat_set_2 + back_squat_set_3)
  let updated_front_squat := updated_back_squat * front_squat_ratio
  let front_squat_set_1 := updated_front_squat * front_squat_effort
  let front_squat_set_2 := (updated_front_squat * (1 + front_squat_ratio_increase)) * front_squat_effort
  let front_squat_set_3 := (updated_front_squat * (1 + 2 * front_squat_ratio_increase)) * front_squat_effort
  let front_squat_total := 3 * (front_squat_set_1 + front_squat_set_2 + front_squat_set_3)
  let updated_deadlift := updated_back_squat * deadlift_ratio
  let deadlift_set_1 := updated_deadlift * deadlift_effort
  let deadlift_set_2 := (updated_deadlift * (1 + deadlift_set_increase)) * deadlift_effort
  let deadlift_set_3 := (updated_deadlift * (1 + 2 * deadlift_set_increase)) * deadlift_effort
  let deadlift_total := 2 * (deadlift_set_1 + deadlift_set_2 + deadlift_set_3)
  (back_squat_total + front_squat_total + deadlift_total) = johns_total_weight_moved :=
by sorry

end john_moves_correct_total_weight_l101_101442


namespace percent_of_value_and_divide_l101_101360

theorem percent_of_value_and_divide (x : ℝ) (y : ℝ) (z : ℝ) (h : x = 1/300 * 180) (h1 : y = x / 6) : 
  y = 0.1 := 
by
  sorry

end percent_of_value_and_divide_l101_101360


namespace pick_peanut_cluster_percentage_l101_101762

def total_chocolates := 100
def typeA_caramels := 5
def typeB_caramels := 6
def typeC_caramels := 4
def typeD_nougats := 2 * typeA_caramels
def typeE_nougats := 2 * typeB_caramels
def typeF_truffles := typeA_caramels + 6
def typeG_truffles := typeB_caramels + 6
def typeH_truffles := typeC_caramels + 6

def total_non_peanut_clusters := 
  typeA_caramels + typeB_caramels + typeC_caramels + typeD_nougats + typeE_nougats + typeF_truffles + typeG_truffles + typeH_truffles

def number_peanut_clusters := total_chocolates - total_non_peanut_clusters

def percent_peanut_clusters := (number_peanut_clusters * 100) / total_chocolates

theorem pick_peanut_cluster_percentage : percent_peanut_clusters = 30 := 
by {
  sorry
}

end pick_peanut_cluster_percentage_l101_101762


namespace sum_of_consecutive_integers_l101_101487

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l101_101487


namespace calc_expression_l101_101984

theorem calc_expression :
  5 + 7 * (2 + (1 / 4 : ℝ)) = 20.75 :=
by
  sorry

end calc_expression_l101_101984


namespace complex_quadrant_l101_101992

theorem complex_quadrant (a b : ℝ) (h : (a + Complex.I) / (b - Complex.I) = 2 - Complex.I) :
  (a < 0 ∧ b < 0) :=
by
  sorry

end complex_quadrant_l101_101992


namespace find_original_price_l101_101612

-- Define the given conditions
def decreased_price : ℝ := 836
def decrease_percentage : ℝ := 0.24
def remaining_percentage : ℝ := 1 - decrease_percentage -- 76% in decimal

-- Define the original price as a variable
variable (x : ℝ)

-- State the theorem
theorem find_original_price (h : remaining_percentage * x = decreased_price) : x = 1100 :=
by
  sorry

end find_original_price_l101_101612


namespace paper_needed_l101_101852

theorem paper_needed : 26 + 26 + 10 = 62 := by
  sorry

end paper_needed_l101_101852


namespace find_g_of_3_l101_101453

noncomputable def g (x : ℝ) : ℝ := sorry  -- Placeholder for the function g

theorem find_g_of_3 (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x) :
  g 3 = 26 / 7 :=
by sorry

end find_g_of_3_l101_101453


namespace red_car_speed_is_10mph_l101_101847

noncomputable def speed_of_red_car (speed_black : ℝ) (initial_distance : ℝ) (time_to_overtake : ℝ) : ℝ :=
  (speed_black * time_to_overtake - initial_distance) / time_to_overtake

theorem red_car_speed_is_10mph :
  ∀ (speed_black initial_distance time_to_overtake : ℝ),
  speed_black = 50 →
  initial_distance = 20 →
  time_to_overtake = 0.5 →
  speed_of_red_car speed_black initial_distance time_to_overtake = 10 :=
by
  intros speed_black initial_distance time_to_overtake hb hd ht
  rw [hb, hd, ht]
  norm_num
  sorry

end red_car_speed_is_10mph_l101_101847


namespace product_of_numbers_l101_101470

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 157) : x * y = 22 := 
by 
  sorry

end product_of_numbers_l101_101470


namespace P_subset_Q_l101_101961

def P : Set ℝ := {m | -1 < m ∧ m < 0}

def Q : Set ℝ := {m | ∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0}

theorem P_subset_Q : P ⊂ Q :=
by
  sorry

end P_subset_Q_l101_101961


namespace rope_folded_three_times_parts_l101_101409

theorem rope_folded_three_times_parts (total_length : ℕ) :
  ∀ parts : ℕ, parts = (total_length / 8) →
  ∀ n : ℕ, n = 3 →
  (∀ length_each_part : ℚ, length_each_part = 1 / (2 ^ n) →
  length_each_part = 1 / 8) :=
by
  sorry

end rope_folded_three_times_parts_l101_101409


namespace goods_train_length_l101_101188

-- Conditions
def train1_speed := 60 -- kmph
def train2_speed := 52 -- kmph
def passing_time := 9 -- seconds

-- Conversion factor from kmph to meters per second
def kmph_to_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600

-- Relative speed in m/s
def relative_speed_mps := kmph_to_mps (train1_speed + train2_speed)

-- Final theorem statement
theorem goods_train_length :
  relative_speed_mps * passing_time = 280 :=
sorry

end goods_train_length_l101_101188


namespace completing_the_square_sum_l101_101468

theorem completing_the_square_sum :
  ∃ (a b c : ℤ), 64 * (x : ℝ) ^ 2 + 96 * x - 81 = 0 ∧ a > 0 ∧ (8 * x + 6) ^ 2 = c ∧ a = 8 ∧ b = 6 ∧ a + b + c = 131 :=
by
  sorry

end completing_the_square_sum_l101_101468


namespace seats_per_bus_l101_101941

theorem seats_per_bus (students buses : ℕ) (h1 : students = 14) (h2 : buses = 7) : students / buses = 2 := by
  sorry

end seats_per_bus_l101_101941


namespace evaluate_expression_l101_101934

-- Definition of variables a, b, c as given in conditions
def a : ℕ := 7
def b : ℕ := 11
def c : ℕ := 13

-- The theorem to prove the given expression equals 31
theorem evaluate_expression : 
  (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) / 
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = 31 :=
by
  sorry

end evaluate_expression_l101_101934


namespace inequality_proof_l101_101077

theorem inequality_proof 
  (a b c x y z : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (h_sum : a + b + c = 1) : 
  (x^2 + y^2 + z^2) * 
  (a^3 / (x^2 + 2 * y^2) + b^3 / (y^2 + 2 * z^2) + c^3 / (z^2 + 2 * x^2)) 
  ≥ 1 / 9 := 
by 
  sorry

end inequality_proof_l101_101077


namespace correct_table_count_l101_101844

def stools_per_table : ℕ := 8
def chairs_per_table : ℕ := 2
def legs_per_stool : ℕ := 3
def legs_per_chair : ℕ := 4
def legs_per_table : ℕ := 4
def total_legs : ℕ := 656

theorem correct_table_count (t : ℕ) :
  stools_per_table * legs_per_stool * t +
  chairs_per_table * legs_per_chair * t +
  legs_per_table * t = total_legs → t = 18 :=
by
  intros h
  sorry

end correct_table_count_l101_101844


namespace added_number_is_6_l101_101022

theorem added_number_is_6 : ∃ x : ℤ, (∃ y : ℤ, y = 9 ∧ (2 * y + x) * 3 = 72) → x = 6 := 
by
  sorry

end added_number_is_6_l101_101022


namespace express_in_standard_form_l101_101255

theorem express_in_standard_form (x : ℝ) : x^2 - 6 * x = (x - 3)^2 - 9 :=
by
  sorry

end express_in_standard_form_l101_101255


namespace gumballs_remaining_l101_101107

theorem gumballs_remaining (a b total eaten remaining : ℕ) 
  (hAlicia : a = 20) 
  (hPedro : b = a + 3 * a) 
  (hTotal : total = a + b) 
  (hEaten : eaten = 40 * total / 100) 
  (hRemaining : remaining = total - eaten) : 
  remaining = 60 := by
  sorry

end gumballs_remaining_l101_101107


namespace catering_budget_l101_101716

namespace CateringProblem

variables (s c : Nat) (cost_steak cost_chicken : Nat)

def total_guests (s c : Nat) : Prop := s + c = 80

def steak_to_chicken_ratio (s c : Nat) : Prop := s = 3 * c

def total_cost (s c cost_steak cost_chicken : Nat) : Nat := s * cost_steak + c * cost_chicken

theorem catering_budget :
  ∃ (s c : Nat), (total_guests s c) ∧ (steak_to_chicken_ratio s c) ∧ (total_cost s c 25 18) = 1860 :=
by
  sorry

end CateringProblem

end catering_budget_l101_101716


namespace quadratic_expression_representation_quadratic_expression_integer_iff_l101_101075

theorem quadratic_expression_representation (A B C : ℤ) :
  ∃ (k l m : ℤ), 
    (k = 2 * A) ∧ 
    (l = A + B) ∧ 
    (m = C) ∧ 
    (∀ x : ℤ, A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m) := 
sorry

theorem quadratic_expression_integer_iff (A B C : ℤ) :
  (∀ x : ℤ, ∃ k l m : ℤ, (k = 2 * A) ∧ (l = A + B) ∧ (m = C) ∧ (A * x^2 + B * x + C = k * (x * (x - 1)) / 2 + l * x + m)) ↔ 
  (A % 1 = 0 ∧ B % 1 = 0 ∧ C % 1 = 0) := 
sorry

end quadratic_expression_representation_quadratic_expression_integer_iff_l101_101075


namespace exterior_angle_of_octagon_is_45_degrees_l101_101785

noncomputable def exterior_angle_of_regular_octagon : ℝ :=
  let n : ℝ := 8
  let interior_angle_sum := 180 * (n - 2) -- This is the sum of interior angles of any n-gon
  let each_interior_angle := interior_angle_sum / n -- Each interior angle in a regular polygon
  let each_exterior_angle := 180 - each_interior_angle -- Exterior angle is supplement of interior angle
  each_exterior_angle

theorem exterior_angle_of_octagon_is_45_degrees :
  exterior_angle_of_regular_octagon = 45 := by
  sorry

end exterior_angle_of_octagon_is_45_degrees_l101_101785


namespace min_abc_sum_l101_101228

theorem min_abc_sum (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 2010) : 
  a + b + c ≥ 78 := 
sorry

end min_abc_sum_l101_101228


namespace plane_through_A_perpendicular_to_BC_l101_101969

-- Define points A, B, and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨-3, 6, 4⟩
def B : Point3D := ⟨8, -3, 5⟩
def C : Point3D := ⟨10, -3, 7⟩

-- Define the vector BC
def vectorBC (B C : Point3D) : Point3D :=
  ⟨C.x - B.x, C.y - B.y, C.z - B.z⟩

-- Equation of the plane
def planeEquation (p : Point3D) (n : Point3D) (x y z : ℝ) : ℝ :=
  n.x * (x - p.x) + n.y * (y - p.y) + n.z * (z - p.z)

theorem plane_through_A_perpendicular_to_BC : 
  planeEquation A (vectorBC B C) x y z = 0 ↔ x + z - 1 = 0 :=
by
  sorry

end plane_through_A_perpendicular_to_BC_l101_101969


namespace prob_shooting_A_first_l101_101395

-- Define the probabilities
def prob_A_hits : ℝ := 0.4
def prob_A_misses : ℝ := 0.6
def prob_B_hits : ℝ := 0.6
def prob_B_misses : ℝ := 0.4

-- Define the overall problem
theorem prob_shooting_A_first (k : ℕ) (ξ : ℕ) (hξ : ξ = k) :
  ((prob_A_misses * prob_B_misses)^(k-1)) * (1 - (prob_A_misses * prob_B_misses)) = 0.24^(k-1) * 0.76 :=
by
  -- Placeholder for proof
  sorry

end prob_shooting_A_first_l101_101395


namespace find_alcohol_quantity_l101_101598

theorem find_alcohol_quantity 
  (A W : ℝ) 
  (h1 : A / W = 2 / 5)
  (h2 : A / (W + 10) = 2 / 7) : 
  A = 10 :=
sorry

end find_alcohol_quantity_l101_101598


namespace count_routes_from_A_to_B_l101_101770

-- Define cities as an inductive type
inductive City
| A
| B
| C
| D
| E

-- Define roads as a list of pairs of cities
def roads : List (City × City) := [
  (City.A, City.B),
  (City.A, City.D),
  (City.B, City.D),
  (City.C, City.D),
  (City.D, City.E),
  (City.B, City.E)
]

-- Define the problem statement
noncomputable def route_count : ℕ :=
  3  -- This should be proven

theorem count_routes_from_A_to_B : route_count = 3 :=
  by
    sorry  -- Proof goes here

end count_routes_from_A_to_B_l101_101770


namespace binomial_coeff_arithmetic_seq_l101_101371

theorem binomial_coeff_arithmetic_seq (n : ℕ) (x : ℝ) (h : ∀ (a b c : ℝ), a = 1 ∧ b = n/2 ∧ c = n*(n-1)/8 → (b - a) = (c - b)) : n = 8 :=
sorry

end binomial_coeff_arithmetic_seq_l101_101371


namespace min_b_minus_a_l101_101290

noncomputable def f (x : ℝ) : ℝ := 1 + x - (x^2) / 2 + (x^3) / 3
noncomputable def g (x : ℝ) : ℝ := 1 - x + (x^2) / 2 - (x^3) / 3
noncomputable def F (x : ℝ) : ℝ := f x * g x

theorem min_b_minus_a (a b : ℤ) (h : ∀ x, F x = 0 → a ≤ x ∧ x ≤ b) (h_a_lt_b : a < b) : b - a = 3 :=
sorry

end min_b_minus_a_l101_101290


namespace simplify_expression_l101_101082

variable (x : ℝ)

theorem simplify_expression :
  2 * x - 3 * (2 - x) + 4 * (1 + 3 * x) - 5 * (1 - x^2) = -5 * x^2 + 17 * x - 7 :=
by
  sorry

end simplify_expression_l101_101082


namespace percent_between_20000_and_150000_l101_101583

-- Define the percentages for each group of counties
def less_than_20000 := 30
def between_20000_and_150000 := 45
def more_than_150000 := 25

-- State the theorem using the above definitions
theorem percent_between_20000_and_150000 :
  between_20000_and_150000 = 45 :=
sorry -- Proof placeholder

end percent_between_20000_and_150000_l101_101583


namespace ratio_constant_l101_101036

theorem ratio_constant (a b c d : ℕ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d)
    (h : ∀ k : ℕ, ∃ m : ℤ, a + c * k = m * (b + d * k)) :
    ∃ m : ℤ, ∀ k : ℕ, a + c * k = m * (b + d * k) :=
    sorry

end ratio_constant_l101_101036


namespace no_single_two_three_digit_solution_l101_101047

theorem no_single_two_three_digit_solution :
  ¬ ∃ (x y z : ℕ),
    (1 ≤ x ∧ x ≤ 9) ∧
    (10 ≤ y ∧ y ≤ 99) ∧
    (100 ≤ z ∧ z ≤ 999) ∧
    (1/x : ℝ) = 1/y + 1/z :=
by
  sorry

end no_single_two_three_digit_solution_l101_101047


namespace still_water_speed_l101_101978

-- The conditions as given in the problem
variables (V_m V_r V'_r : ℝ)
axiom upstream_speed : V_m - V_r = 20
axiom downstream_increased_speed : V_m + V_r = 30
axiom downstream_reduced_speed : V_m + V'_r = 26

-- Prove that the man's speed in still water is 25 km/h
theorem still_water_speed : V_m = 25 :=
by
  sorry

end still_water_speed_l101_101978


namespace min_expression_value_l101_101291

theorem min_expression_value (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (∃ (min_val : ℝ), min_val = 12 ∧ (∀ (x y : ℝ), (x > 1) → (y > 1) →
  ((x^2 / (y - 1)) + (y^2 / (x - 1)) + (x + y) ≥ min_val))) :=
by
  sorry

end min_expression_value_l101_101291


namespace bleach_contains_chlorine_l101_101728

noncomputable def element_in_bleach (mass_percentage : ℝ) (substance : String) : String :=
  if mass_percentage = 31.08 ∧ substance = "sodium hypochlorite" then "Chlorine"
  else "unknown"

theorem bleach_contains_chlorine : element_in_bleach 31.08 "sodium hypochlorite" = "Chlorine" :=
by
  sorry

end bleach_contains_chlorine_l101_101728


namespace proposition_true_iff_l101_101099

theorem proposition_true_iff :
  (∀ x y : ℝ, (xy = 1 → x = 1 / y ∧ y = 1 / x) → (x = 1 / y ∧ y = 1 / x → xy = 1)) ∧
  (∀ (A B : Set ℝ), (A ∩ B = B → A ⊆ B) → (A ⊆ B → A ∩ B = B)) ∧
  (∀ m : ℝ, (m > 1 → ∃ x : ℝ, x^2 - 2 * x + m = 0) → (¬(∃ x : ℝ, x^2 - 2 * x + m = 0) → m ≤ 1)) :=
by
  sorry

end proposition_true_iff_l101_101099


namespace proof_problem_l101_101271

theorem proof_problem (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a > b) (h5 : a^2 - a * c + b * c = 7) :
  a - c = 0 ∨ a - c = 1 :=
 sorry

end proof_problem_l101_101271


namespace evaluate_m_l101_101369

theorem evaluate_m :
  ∀ m : ℝ, (243:ℝ)^(1/5) = 3^m → m = 1 :=
by
  intro m
  sorry

end evaluate_m_l101_101369


namespace find_sum_of_squares_l101_101869

theorem find_sum_of_squares (x y z : ℝ)
  (h1 : x^2 + 3 * y = 8)
  (h2 : y^2 + 5 * z = -9)
  (h3 : z^2 + 7 * x = -16) : x^2 + y^2 + z^2 = 20.75 :=
sorry

end find_sum_of_squares_l101_101869


namespace remainder_of_4n_minus_6_l101_101806

theorem remainder_of_4n_minus_6 (n : ℕ) (h : n % 9 = 5) : (4 * n - 6) % 9 = 5 :=
sorry

end remainder_of_4n_minus_6_l101_101806


namespace probability_of_yellow_ball_is_correct_l101_101243

-- Defining the conditions
def red_balls : ℕ := 2
def yellow_balls : ℕ := 5
def blue_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the probability of choosing a yellow ball
def probability_yellow_ball : ℚ := yellow_balls / total_balls

-- The theorem statement we need to prove
theorem probability_of_yellow_ball_is_correct :
  probability_yellow_ball = 5 / 11 :=
sorry

end probability_of_yellow_ball_is_correct_l101_101243


namespace range_of_a_l101_101207

/--
Given the parabola \(x^2 = y\), points \(A\) and \(B\) are on the parabola and located on both sides of the y-axis,
and the line \(AB\) intersects the y-axis at point \((0, a)\). If \(\angle AOB\) is an acute angle (where \(O\) is the origin),
then the real number \(a\) is greater than 1.
-/
theorem range_of_a (a : ℝ) (x1 x2 : ℝ) : (x1^2 = x2^2) → (x1 * x2 = -a) → ((-a + a^2) > 0) → (1 < a) :=
by 
  sorry

end range_of_a_l101_101207


namespace quadratic_has_root_in_interval_l101_101110

theorem quadratic_has_root_in_interval (a b c : ℝ) (h : 2 * a + 3 * b + 6 * c = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x^2 + b * x + c = 0 :=
sorry

end quadratic_has_root_in_interval_l101_101110


namespace solution_set_of_inequality_l101_101379

theorem solution_set_of_inequality :
  {x : ℝ | x^2 < x + 6} = {x : ℝ | -2 < x ∧ x < 3} := 
sorry

end solution_set_of_inequality_l101_101379


namespace lifting_ratio_after_gain_l101_101585

def intial_lifting_total : ℕ := 2200
def initial_bodyweight : ℕ := 245
def percentage_gain_total : ℕ := 15
def weight_gain : ℕ := 8

theorem lifting_ratio_after_gain :
  (intial_lifting_total * (100 + percentage_gain_total) / 100) / (initial_bodyweight + weight_gain) = 10 := by
  sorry

end lifting_ratio_after_gain_l101_101585


namespace cos_alpha_half_l101_101197

theorem cos_alpha_half (α : ℝ) (h : Real.cos (Real.pi + α) = -1/2) : Real.cos α = 1/2 := 
by 
  sorry

end cos_alpha_half_l101_101197


namespace solve_for_x_l101_101587

theorem solve_for_x (x z : ℝ) (h : z = 3 * x) :
  (4 * z^2 + z + 5 = 3 * (8 * x^2 + z + 3)) ↔ 
  (x = (1 + Real.sqrt 19) / 4 ∨ x = (1 - Real.sqrt 19) / 4) := by
  sorry

end solve_for_x_l101_101587


namespace mike_total_investment_l101_101654

variable (T : ℝ)
variable (H1 : 0.09 * 1800 + 0.11 * (T - 1800) = 624)

theorem mike_total_investment : T = 6000 :=
by
  sorry

end mike_total_investment_l101_101654


namespace betty_age_l101_101872

-- Define the constants and conditions
variables (A M B : ℕ)
variables (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 8)

-- Define the theorem to prove Betty's age
theorem betty_age : B = 4 :=
by sorry

end betty_age_l101_101872


namespace total_income_l101_101751

-- Definitions of conditions
def charge_per_meter : ℝ := 0.2
def number_of_fences : ℝ := 50
def length_of_each_fence : ℝ := 500

-- Theorem statement
theorem total_income :
  let total_length := number_of_fences * length_of_each_fence
  let total_income := total_length * charge_per_meter
  total_income = 5000 := 
by
  sorry

end total_income_l101_101751


namespace Paul_correct_probability_l101_101469

theorem Paul_correct_probability :
  let P_Ghana := 1/2
  let P_Bolivia := 1/6
  let P_Argentina := 1/6
  let P_France := 1/6
  (P_Ghana^2 + P_Bolivia^2 + P_Argentina^2 + P_France^2) = 1/3 :=
by
  sorry

end Paul_correct_probability_l101_101469


namespace distance_from_LV_to_LA_is_273_l101_101486

-- Define the conditions
def distance_SLC_to_LV : ℝ := 420
def total_time : ℝ := 11
def avg_speed : ℝ := 63

-- Define the total distance covered given the average speed and time
def total_distance : ℝ := avg_speed * total_time

-- Define the distance from Las Vegas to Los Angeles
def distance_LV_to_LA : ℝ := total_distance - distance_SLC_to_LV

-- Now state the theorem we want to prove
theorem distance_from_LV_to_LA_is_273 :
  distance_LV_to_LA = 273 :=
sorry

end distance_from_LV_to_LA_is_273_l101_101486


namespace discriminant_eq_13_l101_101293

theorem discriminant_eq_13 (m : ℝ) (h : (3)^2 - 4*1*(-m) = 13) : m = 1 :=
sorry

end discriminant_eq_13_l101_101293


namespace range_of_y_under_conditions_l101_101889

theorem range_of_y_under_conditions :
  (∀ x : ℝ, (x - y) * (x + y) < 1) → (-1/2 : ℝ) < y ∧ y < (3/2 : ℝ) := by
  intro h
  have h' : ∀ x : ℝ, (x - y) * (1 - x - y) < 1 := by
    sorry
  have g_min : ∀ x : ℝ, y^2 - y < x^2 - x + 1 := by
    sorry
  have min_value : y^2 - y < 3/4 := by
    sorry
  have range_y : (-1/2 : ℝ) < y ∧ y < (3/2 : ℝ) := by
    sorry
  exact range_y

end range_of_y_under_conditions_l101_101889


namespace students_in_diligence_before_transfer_l101_101390

theorem students_in_diligence_before_transfer (D I : ℕ) 
  (h1 : D + 2 = I - 2) 
  (h2 : D + I = 50) : 
  D = 23 := 
by
  sorry

end students_in_diligence_before_transfer_l101_101390


namespace arccos_neg_one_eq_pi_l101_101916

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_eq_pi_l101_101916


namespace find_k_l101_101640

-- Define the vectors and the condition of perpendicularity
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, -1)
def c (k : ℝ) : ℝ × ℝ := (3 + k, 1 - k)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The primary statement we aim to prove
theorem find_k : ∃ k : ℝ, dot_product a (c k) = 0 ∧ k = -5 :=
by
  exists -5
  sorry

end find_k_l101_101640


namespace largest_number_of_gold_coins_l101_101933

theorem largest_number_of_gold_coins (n : ℕ) :
  (∃ k : ℕ, n = 13 * k + 3 ∧ n < 150) → n ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l101_101933


namespace rectangle_perimeter_ratio_l101_101823

theorem rectangle_perimeter_ratio (side_length : ℝ) (h : side_length = 4) :
  let small_rectangle_perimeter := 2 * (side_length + (side_length / 4))
  let large_rectangle_perimeter := 2 * (side_length + (side_length / 2))
  small_rectangle_perimeter / large_rectangle_perimeter = 5 / 6 :=
by
  sorry

end rectangle_perimeter_ratio_l101_101823


namespace find_a_l101_101917

-- Define the variables
variables (m d a b : ℝ)

-- State the main theorem with conditions
theorem find_a (h : m = d * a * b / (a - b)) (h_ne : m ≠ d * b) : a = m * b / (m - d * b) :=
sorry

end find_a_l101_101917


namespace brown_gumdrops_after_replacement_l101_101579

-- Definitions based on the given conditions.
def total_gumdrops (green_gumdrops : ℕ) : ℕ :=
  (green_gumdrops * 100) / 15

def blue_gumdrops (total_gumdrops : ℕ) : ℕ :=
  total_gumdrops * 25 / 100

def brown_gumdrops_initial (total_gumdrops : ℕ) : ℕ :=
  total_gumdrops * 15 / 100

def brown_gumdrops_final (brown_initial : ℕ) (blue_gumdrops : ℕ) : ℕ :=
  brown_initial + blue_gumdrops / 3

-- The main theorem statement based on the proof problem.
theorem brown_gumdrops_after_replacement
  (green_gumdrops : ℕ)
  (h_green : green_gumdrops = 36)
  : brown_gumdrops_final (brown_gumdrops_initial (total_gumdrops green_gumdrops)) 
                         (blue_gumdrops (total_gumdrops green_gumdrops))
    = 56 := 
  by sorry

end brown_gumdrops_after_replacement_l101_101579


namespace discount_difference_l101_101258

def single_discount (original: ℝ) (discount: ℝ) : ℝ :=
  original * (1 - discount)

def successive_discount (original: ℝ) (first_discount: ℝ) (second_discount: ℝ) : ℝ :=
  original * (1 - first_discount) * (1 - second_discount)

theorem discount_difference : 
  let original := 12000
  let single_disc := 0.30
  let first_disc := 0.20
  let second_disc := 0.10
  single_discount original single_disc - successive_discount original first_disc second_disc = 240 := 
by sorry

end discount_difference_l101_101258


namespace find_a_l101_101050

theorem find_a 
  (a b c : ℤ) 
  (h_vertex : ∀ x, (a * (x - 2)^2 + 5 = a * x^2 + b * x + c))
  (h_point : ∀ y, y = a * (1 - 2)^2 + 5)
  : a = -1 := by
  sorry

end find_a_l101_101050


namespace xy_zero_iff_x_zero_necessary_not_sufficient_l101_101296

theorem xy_zero_iff_x_zero_necessary_not_sufficient {x y : ℝ} : 
  (x * y = 0) → ((x = 0) ∨ (y = 0)) ∧ ¬((x = 0) → (x * y ≠ 0)) := 
sorry

end xy_zero_iff_x_zero_necessary_not_sufficient_l101_101296


namespace statement2_statement3_l101_101250

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Conditions for the statements
axiom cond1 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c p = q ∧ f a b c q = p
axiom cond2 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c p = f a b c q
axiom cond3 (a b c p q : ℝ) (hpq : p ≠ q) : f a b c (p + q) = c

-- Statement 2 correctness
theorem statement2 (a b c p q : ℝ) (hpq : p ≠ q) (h : f a b c p = f a b c q) : 
  f a b c (p + q) = c :=
sorry

-- Statement 3 correctness
theorem statement3 (a b c p q : ℝ) (hpq : p ≠ q) (h : f a b c (p + q) = c) : 
  p + q = 0 ∨ f a b c p = f a b c q :=
sorry

end statement2_statement3_l101_101250


namespace rectangle_image_l101_101045

-- A mathematically equivalent Lean 4 proof problem statement

variable (x y : ℝ)

def rectangle_OABC (x y : ℝ) : Prop :=
  (x = 0 ∧ (0 ≤ y ∧ y ≤ 3)) ∨
  (y = 0 ∧ (0 ≤ x ∧ x ≤ 2)) ∨
  (x = 2 ∧ (0 ≤ y ∧ y ≤ 3)) ∨
  (y = 3 ∧ (0 ≤ x ∧ x ≤ 2))

def transform_u (x y : ℝ) : ℝ := x^2 - y^2 + 1
def transform_v (x y : ℝ) : ℝ := x * y

theorem rectangle_image (u v : ℝ) :
  (∃ (x y : ℝ), rectangle_OABC x y ∧ u = transform_u x y ∧ v = transform_v x y) ↔
  (u, v) = (-8, 0) ∨
  (u, v) = (1, 0) ∨
  (u, v) = (5, 0) ∨
  (u, v) = (-4, 6) :=
sorry

end rectangle_image_l101_101045


namespace quadrilateral_area_is_correct_l101_101788

-- Let's define the situation
structure TriangleDivisions where
  T1_area : ℝ
  T2_area : ℝ
  T3_area : ℝ
  Q_area : ℝ

def triangleDivisionExample : TriangleDivisions :=
  { T1_area := 4,
    T2_area := 9,
    T3_area := 9,
    Q_area := 36 }

-- The statement to prove
theorem quadrilateral_area_is_correct (T : TriangleDivisions) (h1 : T.T1_area = 4) 
  (h2 : T.T2_area = 9) (h3 : T.T3_area = 9) : T.Q_area = 36 :=
by
  sorry

end quadrilateral_area_is_correct_l101_101788


namespace purple_cars_count_l101_101544

theorem purple_cars_count
    (P R G : ℕ)
    (h1 : R = P + 6)
    (h2 : G = 4 * R)
    (h3 : P + R + G = 312) :
    P = 47 :=
by 
  sorry

end purple_cars_count_l101_101544


namespace Frank_time_correct_l101_101300

def Dave_time := 10
def Chuck_time := 5 * Dave_time
def Erica_time := 13 * Chuck_time / 10
def Frank_time := 12 * Erica_time / 10

theorem Frank_time_correct : Frank_time = 78 :=
by
  sorry

end Frank_time_correct_l101_101300


namespace replace_star_l101_101923

theorem replace_star (x : ℕ) : 2 * 18 * 14 = 6 * x * 7 → x = 12 :=
sorry

end replace_star_l101_101923


namespace Zelda_probability_success_l101_101421

variable (P : ℝ → ℝ)
variable (X Y Z : ℝ)

theorem Zelda_probability_success :
  P X = 1/3 ∧ P Y = 1/2 ∧ (P X) * (P Y) * (1 - P Z) = 0.0625 → P Z = 0.625 :=
by
  sorry

end Zelda_probability_success_l101_101421


namespace Jake_weight_loss_l101_101187

variables (J K x : ℕ)

theorem Jake_weight_loss : 
  J = 198 ∧ J + K = 293 ∧ J - x = 2 * K → x = 8 := 
by {
  sorry
}

end Jake_weight_loss_l101_101187


namespace distinct_numbers_in_list_l101_101084

def count_distinct_floors (l : List ℕ) : ℕ :=
  l.eraseDups.length

def generate_list : List ℕ :=
  List.map (λ n => Nat.floor ((n * n : ℚ) / 2000)) (List.range' 1 2000)

theorem distinct_numbers_in_list : count_distinct_floors generate_list = 1501 :=
by
  sorry

end distinct_numbers_in_list_l101_101084


namespace meaningful_expression_l101_101373

-- Definition stating the meaningfulness of the expression (condition)
def is_meaningful (a : ℝ) : Prop := (a - 1) ≠ 0

-- Theorem stating that for the expression to be meaningful, a ≠ 1
theorem meaningful_expression (a : ℝ) : is_meaningful a ↔ a ≠ 1 :=
by sorry

end meaningful_expression_l101_101373


namespace find_original_price_l101_101905

-- Define the conditions provided in the problem
def original_price (P : ℝ) : Prop :=
  let first_discount := 0.90 * P
  let second_discount := 0.85 * first_discount
  let taxed_price := 1.08 * second_discount
  taxed_price = 450

-- State and prove the main theorem
theorem find_original_price (P : ℝ) (h : original_price P) : P = 544.59 :=
  sorry

end find_original_price_l101_101905


namespace green_more_than_blue_l101_101254

-- Define the conditions
variables (B Y G n : ℕ)
def ratio_condition := 3 * n = B ∧ 7 * n = Y ∧ 8 * n = G
def total_disks_condition := B + Y + G = 72

-- State the theorem
theorem green_more_than_blue (B Y G n : ℕ) 
  (h_ratio : ratio_condition B Y G n) 
  (h_total : total_disks_condition B Y G) 
  : G - B = 20 := 
sorry

end green_more_than_blue_l101_101254


namespace evaluate_g_at_2_l101_101131

def g (x : ℝ) : ℝ := x^3 + x^2 - 1

theorem evaluate_g_at_2 : g 2 = 11 := by
  sorry

end evaluate_g_at_2_l101_101131


namespace sequence_a_is_perfect_square_l101_101451

theorem sequence_a_is_perfect_square :
  ∃ (a b : ℕ → ℤ),
    a 0 = 1 ∧ 
    b 0 = 0 ∧ 
    (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) ∧
    (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) ∧
    ∀ n, ∃ m : ℕ, a n = m * m := sorry

end sequence_a_is_perfect_square_l101_101451


namespace longest_tape_length_l101_101577

/-!
  Problem: Find the length of the longest tape that can exactly measure the lengths 
  24 m, 36 m, and 54 m in cm.
  
  Solution: Convert the given lengths to the same unit (cm), then find their GCD.
  
  Given: Lengths are 2400 cm, 3600 cm, and 5400 cm.
  To Prove: gcd(2400, 3600, 5400) = 300.
-/

theorem longest_tape_length (a b c : ℕ) : a = 2400 → b = 3600 → c = 5400 → Nat.gcd (Nat.gcd a b) c = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- omitted proof steps
  sorry

end longest_tape_length_l101_101577


namespace juan_marbles_l101_101725

-- Conditions
def connie_marbles : ℕ := 39
def extra_marbles_juan : ℕ := 25

-- Theorem statement: Total marbles Juan has
theorem juan_marbles : connie_marbles + extra_marbles_juan = 64 :=
by
  sorry

end juan_marbles_l101_101725


namespace cos_135_eq_neg_sqrt2_div_2_l101_101960

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l101_101960


namespace randy_feeds_per_day_l101_101225

theorem randy_feeds_per_day
  (pigs : ℕ) (total_feed_per_week : ℕ) (days_per_week : ℕ)
  (h1 : pigs = 2) (h2 : total_feed_per_week = 140) (h3 : days_per_week = 7) :
  total_feed_per_week / pigs / days_per_week = 10 :=
by
  sorry

end randy_feeds_per_day_l101_101225


namespace arcsin_neg_one_l101_101701

theorem arcsin_neg_one : Real.arcsin (-1) = -Real.pi / 2 := by
  sorry

end arcsin_neg_one_l101_101701


namespace arith_seq_sum_first_four_terms_l101_101908

noncomputable def sum_first_four_terms_arith_seq (a1 : ℤ) (d : ℤ) : ℤ :=
  4 * a1 + 6 * d

theorem arith_seq_sum_first_four_terms (a1 a3 : ℤ) 
  (h1 : a3 = a1 + 2 * 3)
  (h2 : a1 + a3 = 8) 
  (d : ℤ := 3) :
  sum_first_four_terms_arith_seq a1 d = 22 := by
  unfold sum_first_four_terms_arith_seq
  sorry

end arith_seq_sum_first_four_terms_l101_101908


namespace kth_term_in_sequence_l101_101843

theorem kth_term_in_sequence (k : ℕ) (hk : 0 < k) : ℚ :=
  (2 * k) / (2 * k + 1)

end kth_term_in_sequence_l101_101843


namespace decreasing_interval_b_l101_101393

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := - (1 / 2) * x ^ 2 + b * Real.log x

theorem decreasing_interval_b (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici (Real.sqrt 2) → ∀ x1 x2 : ℝ, x1 ∈ Set.Ici (Real.sqrt 2) → x2 ∈ Set.Ici (Real.sqrt 2) → 
   x1 ≤ x2 → f x1 b ≥ f x2 b) ↔ b ≤ 2 :=
by
  sorry

end decreasing_interval_b_l101_101393


namespace complement_intersection_l101_101178

open Set

-- Definitions based on conditions given
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 5, 7}

-- The mathematical proof problem
theorem complement_intersection :
  (U \ A) ∩ B = {1, 3, 7} :=
by
  sorry

end complement_intersection_l101_101178


namespace range_of_x_l101_101181

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)

theorem range_of_x (a b x : ℝ) (h : a ≠ 0) (h₁ : abs (a + b) + abs (a - b) ≥ abs a * f x) :
  0 ≤ x ∧ x ≤ 4 :=
sorry

end range_of_x_l101_101181


namespace largest_expression_is_A_l101_101786

noncomputable def A : ℝ := 3009 / 3008 + 3009 / 3010
noncomputable def B : ℝ := 3011 / 3010 + 3011 / 3012
noncomputable def C : ℝ := 3010 / 3009 + 3010 / 3011

theorem largest_expression_is_A : A > B ∧ A > C := by
  sorry

end largest_expression_is_A_l101_101786


namespace focus_of_parabola_l101_101517

theorem focus_of_parabola (f : ℝ) : 
  (∀ (x: ℝ), x^2 + ((- 1 / 16) * x^2 - f)^2 = ((- 1 / 16) * x^2 - (f + 8))^2) 
  → f = -4 :=
by
  intro h
  sorry

end focus_of_parabola_l101_101517


namespace part1_solution_set_part2_range_a_l101_101566

-- Define the function f
noncomputable def f (x a : ℝ) := |x - a| + |x + 3|

-- Part 1: Proving the solution set of the inequality
theorem part1_solution_set (x : ℝ) :
  (∀ x, f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2: Finding the range of values for a
theorem part2_range_a (a : ℝ) :
  (∀ x, f x a > -a ↔ a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_a_l101_101566


namespace basement_pump_time_l101_101377

/-- A basement has a 30-foot by 36-foot rectangular floor, flooded to a depth of 24 inches.
Using three pumps, each pumping 10 gallons per minute, and knowing that a cubic foot of water
contains 7.5 gallons, this theorem asserts it will take 540 minutes to pump out all the water. -/
theorem basement_pump_time :
  let length := 30 -- in feet
  let width := 36 -- in feet
  let depth_inch := 24 -- in inches
  let depth := depth_inch / 12 -- converting depth to feet
  let volume_ft3 := length * width * depth -- volume in cubic feet
  let gallons_per_ft3 := 7.5 -- gallons per cubic foot
  let total_gallons := volume_ft3 * gallons_per_ft3 -- total volume in gallons
  let pump_capacity_gpm := 10 -- gallons per minute per pump
  let total_pumps := 3 -- number of pumps
  let total_pump_gpm := pump_capacity_gpm * total_pumps -- total gallons per minute for all pumps
  let pump_time := total_gallons / total_pump_gpm -- time in minutes to pump all the water
  pump_time = 540 := sorry

end basement_pump_time_l101_101377


namespace problem1_problem2_l101_101485

section
variable {x a : ℝ}

-- Definitions of the functions
def f (x : ℝ) : ℝ := |x + 1|
def g (x : ℝ) (a : ℝ) : ℝ := 2 * |x| + a

-- Problem 1
theorem problem1 (a : ℝ) (H : a = -1) : 
  ∀ x : ℝ, f x ≤ g x a ↔ (x ≤ -2/3 ∨ 2 ≤ x) :=
sorry

-- Problem 2
theorem problem2 (a : ℝ) : 
  (∃ x₀ : ℝ, f x₀ ≥ 1/2 * g x₀ a) → a ≤ 2 :=
sorry

end

end problem1_problem2_l101_101485


namespace probability_A_wins_l101_101745

variable (P_A_not_lose : ℝ) (P_draw : ℝ)
variable (h1 : P_A_not_lose = 0.8)
variable (h2 : P_draw = 0.5)

theorem probability_A_wins : P_A_not_lose - P_draw = 0.3 := by
  sorry

end probability_A_wins_l101_101745


namespace average_rate_l101_101232

variable (d_run : ℝ) (d_swim : ℝ) (r_run : ℝ) (r_swim : ℝ)
variable (t_run : ℝ := d_run / r_run) (t_swim : ℝ := d_swim / r_swim)

theorem average_rate (h_dist_run : d_run = 4) (h_dist_swim : d_swim = 4)
                      (h_run_rate : r_run = 10) (h_swim_rate : r_swim = 6) : 
                      ((d_run + d_swim) / (t_run + t_swim)) / 60 = 0.125 :=
by
  -- Properly using all the conditions given
  have := (4 + 4) / (4 / 10 + 4 / 6) / 60 = 0.125
  sorry

end average_rate_l101_101232


namespace fill_trough_time_l101_101618

theorem fill_trough_time 
  (old_pump_rate : ℝ := 1 / 600) 
  (new_pump_rate : ℝ := 1 / 200) : 
  1 / (old_pump_rate + new_pump_rate) = 150 := 
by 
  sorry

end fill_trough_time_l101_101618


namespace projection_problem_l101_101727

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

theorem projection_problem :
  let v : ℝ × ℝ := (1, -1/2)
  let sum_v := (v.1 + 1, v.2 + 1)
  projection (3, 5) sum_v = (104/17, 26/17) :=
by
  sorry

end projection_problem_l101_101727


namespace value_of_d_l101_101295

theorem value_of_d (d : ℝ) (h : x^2 - 60 * x + d = (x - 30)^2) : d = 900 :=
by { sorry }

end value_of_d_l101_101295


namespace goose_eggs_count_l101_101437

theorem goose_eggs_count (E : ℕ)
  (hatch_ratio : ℚ := 2 / 3)
  (survive_first_month_ratio : ℚ := 3 / 4)
  (survive_first_year_ratio : ℚ := 2 / 5)
  (survived_first_year : ℕ := 130) :
  (survive_first_year_ratio * survive_first_month_ratio * hatch_ratio * (E : ℚ) = survived_first_year) →
  E = 1300 := by
  sorry

end goose_eggs_count_l101_101437


namespace evaluate_expression_l101_101461

theorem evaluate_expression : 
  (3^2 - 3 * 2) - (4^2 - 4 * 2) + (5^2 - 5 * 2) - (6^2 - 6 * 2) = -14 :=
by
  sorry

end evaluate_expression_l101_101461


namespace complement_U_A_l101_101929

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {1, 5, 7}

theorem complement_U_A : (U \ A) = {3, 9} :=
by
  sorry

end complement_U_A_l101_101929


namespace Ofelia_savings_l101_101810

theorem Ofelia_savings (X : ℝ) (h : 16 * X = 160) : X = 10 :=
by
  sorry

end Ofelia_savings_l101_101810


namespace luncheon_cost_l101_101864

theorem luncheon_cost (s c p : ℝ) (h1 : 5 * s + 9 * c + 2 * p = 5.95)
  (h2 : 7 * s + 12 * c + 2 * p = 7.90) (h3 : 3 * s + 5 * c + p = 3.50) :
  s + c + p = 1.05 :=
sorry

end luncheon_cost_l101_101864


namespace prove_weight_of_a_l101_101132

noncomputable def weight_proof (A B C D : ℝ) : Prop :=
  (A + B + C) / 3 = 60 ∧
  50 ≤ A ∧ A ≤ 80 ∧
  50 ≤ B ∧ B ≤ 80 ∧
  50 ≤ C ∧ C ≤ 80 ∧
  60 ≤ D ∧ D ≤ 90 ∧
  (A + B + C + D) / 4 = 65 ∧
  70 ≤ D + 3 ∧ D + 3 ≤ 100 ∧
  (B + C + D + (D + 3)) / 4 = 64 → 
  A = 87

-- Adding a theorem statement to make it clear we need to prove this.
theorem prove_weight_of_a (A B C D : ℝ) : weight_proof A B C D :=
sorry

end prove_weight_of_a_l101_101132


namespace boyden_family_tickets_l101_101774

theorem boyden_family_tickets (child_ticket_cost : ℕ) (adult_ticket_cost : ℕ) (total_cost : ℕ) (num_adults : ℕ) (num_children : ℕ) :
  adult_ticket_cost = child_ticket_cost + 6 →
  total_cost = 77 →
  adult_ticket_cost = 19 →
  num_adults = 2 →
  num_children = 3 →
  num_adults * adult_ticket_cost + num_children * child_ticket_cost = total_cost →
  num_adults + num_children = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end boyden_family_tickets_l101_101774


namespace printers_ratio_l101_101460

theorem printers_ratio (Rate_X : ℝ := 1 / 16) (Rate_Y : ℝ := 1 / 10) (Rate_Z : ℝ := 1 / 20) :
  let Time_X := 1 / Rate_X
  let Time_YZ := 1 / (Rate_Y + Rate_Z)
  (Time_X / Time_YZ) = 12 / 5 := by
  sorry

end printers_ratio_l101_101460


namespace arrangements_no_adjacent_dances_arrangements_alternating_order_l101_101154

-- Part (1)
theorem arrangements_no_adjacent_dances (singing_programs dance_programs : ℕ) (h_s : singing_programs = 5) (h_d : dance_programs = 4) :
  ∃ n, n = 43200 := 
by sorry

-- Part (2)
theorem arrangements_alternating_order (singing_programs dance_programs : ℕ) (h_s : singing_programs = 5) (h_d : dance_programs = 4) :
  ∃ n, n = 2880 := 
by sorry

end arrangements_no_adjacent_dances_arrangements_alternating_order_l101_101154


namespace change_color_while_preserving_friendship_l101_101530

-- Definitions
def children := Fin 10000
def colors := Fin 7
def friends (a b : children) : Prop := sorry -- mutual and exactly 11 friends per child
def refuses_to_change (c : children) : Prop := sorry -- only 100 specified children refuse to change color

theorem change_color_while_preserving_friendship :
  ∃ c : children, ¬refuses_to_change c ∧
    ∃ new_color : colors, 
      (∀ friend : children, friends c friend → 
      (∃ current_color current_friend_color : colors, current_color ≠ current_friend_color)) :=
sorry

end change_color_while_preserving_friendship_l101_101530


namespace largest_of_seven_consecutive_odd_numbers_l101_101541

theorem largest_of_seven_consecutive_odd_numbers (a b c d e f g : ℤ) 
  (h1: a % 2 = 1) (h2: b % 2 = 1) (h3: c % 2 = 1) (h4: d % 2 = 1) 
  (h5: e % 2 = 1) (h6: f % 2 = 1) (h7: g % 2 = 1)
  (h8 : a + b + c + d + e + f + g = 105)
  (h9 : b = a + 2) (h10 : c = a + 4) (h11 : d = a + 6)
  (h12 : e = a + 8) (h13 : f = a + 10) (h14 : g = a + 12) :
  g = 21 :=
by 
  sorry

end largest_of_seven_consecutive_odd_numbers_l101_101541


namespace papaya_tree_growth_ratio_l101_101600

theorem papaya_tree_growth_ratio :
  ∃ (a1 a2 a3 a4 a5 : ℝ),
    a1 = 2 ∧
    a2 = a1 * 1.5 ∧
    a3 = a2 * 1.5 ∧
    a4 = a3 * 2 ∧
    a1 + a2 + a3 + a4 + a5 = 23 ∧
    a5 = 4.5 ∧
    (a5 / a4) = 0.5 :=
sorry

end papaya_tree_growth_ratio_l101_101600


namespace cos_210_eq_neg_sqrt3_div_2_l101_101274

theorem cos_210_eq_neg_sqrt3_div_2 : Real.cos (210 * Real.pi / 180) = - (Real.sqrt 3) / 2 :=
  sorry

end cos_210_eq_neg_sqrt3_div_2_l101_101274


namespace find_t_l101_101413

-- Defining variables and assumptions
variables (V V0 g S t : Real)
variable (h1 : V = g * t + V0)
variable (h2 : S = (1 / 2) * g * t^2 + V0 * t)

-- The goal: to prove t equals 2S / (V + V0)
theorem find_t (V V0 g S t : Real) (h1 : V = g * t + V0) (h2 : S = (1 / 2) * g * t^2 + V0 * t):
  t = 2 * S / (V + V0) := by
  sorry

end find_t_l101_101413


namespace total_valid_votes_l101_101308

theorem total_valid_votes (V : ℝ) (H_majority : 0.70 * V - 0.30 * V = 188) : V = 470 :=
by
  sorry

end total_valid_votes_l101_101308


namespace complement_intersect_eq_l101_101977

-- Define Universal Set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define Set P
def P : Set ℕ := {2, 3, 4}

-- Define Set Q
def Q : Set ℕ := {1, 2}

-- Complement of P in U
def complement_U_P : Set ℕ := U \ P

-- Goal Statement
theorem complement_intersect_eq {U P Q : Set ℕ} 
  (hU : U = {1, 2, 3, 4}) 
  (hP : P = {2, 3, 4}) 
  (hQ : Q = {1, 2}) : 
  (complement_U_P ∩ Q) = {1} := 
by
  sorry

end complement_intersect_eq_l101_101977


namespace factor_diff_of_squares_l101_101808

theorem factor_diff_of_squares (t : ℝ) : 
  t^2 - 64 = (t - 8) * (t + 8) :=
by
  -- The purpose here is to state the theorem only, without the proof.
  sorry

end factor_diff_of_squares_l101_101808


namespace distinct_x_intercepts_l101_101726

theorem distinct_x_intercepts : 
  let f (x : ℝ) := ((x - 8) * (x^2 + 4*x + 3))
  (∃ x1 x2 x3 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) :=
by
  sorry

end distinct_x_intercepts_l101_101726


namespace expression_exists_l101_101708

theorem expression_exists (a b : ℤ) (h : 5 * a = 3125) (hb : 5 * b = 25) : b = 5 := by
  sorry

end expression_exists_l101_101708


namespace Isabella_hair_length_l101_101884

-- Define the conditions using variables
variables (h_current h_cut_off h_initial : ℕ)

-- The proof problem statement
theorem Isabella_hair_length :
  h_current = 9 → h_cut_off = 9 → h_initial = h_current + h_cut_off → h_initial = 18 :=
by
  intros hc hc' hi
  rw [hc, hc'] at hi
  exact hi


end Isabella_hair_length_l101_101884


namespace option_C_correct_inequality_l101_101253

theorem option_C_correct_inequality (x : ℝ) : 
  (1 / ((x + 1) * (x - 1)) ≤ 0) ↔ (-1 < x ∧ x < 1) :=
sorry

end option_C_correct_inequality_l101_101253


namespace scientific_notation_189100_l101_101733

  theorem scientific_notation_189100 :
    (∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 189100 = a * 10^n) ∧ (∃ (a : ℝ) (n : ℤ), a = 1.891 ∧ n = 5) :=
  by {
    sorry
  }
  
end scientific_notation_189100_l101_101733


namespace sequence_value_at_20_l101_101891

open Nat

def arithmetic_sequence (a : ℕ → ℤ) : Prop := 
  a 1 = 1 ∧ ∀ n, a (n + 1) - a n = 4

theorem sequence_value_at_20 (a : ℕ → ℤ) (h : arithmetic_sequence a) : a 20 = 77 :=
sorry

end sequence_value_at_20_l101_101891


namespace bus_passing_time_l101_101965

noncomputable def time_for_bus_to_pass (bus_length : ℝ) (bus_speed_kph : ℝ) (man_speed_kph : ℝ) : ℝ :=
  let relative_speed_kph := bus_speed_kph + man_speed_kph
  let relative_speed_mps := (relative_speed_kph * (1000/3600))
  bus_length / relative_speed_mps

theorem bus_passing_time :
  time_for_bus_to_pass 15 40 8 = 1.125 :=
by
  sorry

end bus_passing_time_l101_101965


namespace smallest_n_good_sequence_2014_l101_101537

-- Define the concept of a "good sequence"
def good_sequence (a : ℕ → ℝ) : Prop :=
  a 0 > 0 ∧
  ∀ i, a (i + 1) = 2 * a i + 1 ∨ a (i + 1) = a i / (a i + 2)

-- Define the smallest n such that a good sequence reaches 2014 at a_n
theorem smallest_n_good_sequence_2014 :
  ∃ (n : ℕ), (∀ a, good_sequence a → a n = 2014) ∧
  ∀ (m : ℕ), m < n → ∀ a, good_sequence a → a m ≠ 2014 :=
sorry

end smallest_n_good_sequence_2014_l101_101537


namespace calligraphy_prices_max_brushes_l101_101524

theorem calligraphy_prices 
  (x y : ℝ)
  (h1 : 40 * x + 100 * y = 280)
  (h2 : 30 * x + 200 * y = 260) :
  x = 6 ∧ y = 0.4 := 
by sorry

theorem max_brushes 
  (m : ℝ)
  (h_budget : 6 * m + 0.4 * (200 - m) ≤ 360) :
  m ≤ 50 :=
by sorry

end calligraphy_prices_max_brushes_l101_101524


namespace find_y_l101_101765

variable (A B C : Point)

def carla_rotate (θ : ℕ) (A B C : Point) : Prop := 
  -- Definition to indicate point A rotated by θ degrees clockwise about point B lands at point C
  sorry

def devon_rotate (θ : ℕ) (A B C : Point) : Prop := 
  -- Definition to indicate point A rotated by θ degrees counterclockwise about point B lands at point C
  sorry

theorem find_y
  (h1 : carla_rotate 690 A B C)
  (h2 : ∀ y, devon_rotate y A B C)
  (h3 : y < 360) :
  ∃ y, y = 30 :=
by
  sorry

end find_y_l101_101765


namespace parabola_coordinates_and_area_l101_101147

theorem parabola_coordinates_and_area
  (A B C : ℝ × ℝ)
  (hA : A = (2, 0))
  (hB : B = (3, 0))
  (hC : C = (5 / 2, 1 / 4))
  (h_vertex : ∀ x y, y = -x^2 + 5 * x - 6 → 
                   ((x, y) = A ∨ (x, y) = B ∨ (x, y) = C)) :
  A = (2, 0) ∧ B = (3, 0) ∧ C = (5 / 2, 1 / 4)
  ∧ (1 / 2 * (3 - 2) * (1 / 4) = 1 / 8) := 
by
  sorry

end parabola_coordinates_and_area_l101_101147


namespace problem_solution_l101_101418

theorem problem_solution :
  { x : ℝ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ (1 / (x * (x - 1)) - 1 / ((x - 1) * (x - 2)) < 1 / 5) } 
  = { x : ℝ | x < 0 } ∪ { x : ℝ | 1 < x ∧ x < 2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end problem_solution_l101_101418


namespace min_age_of_youngest_person_l101_101937

theorem min_age_of_youngest_person
  {a b c d e : ℕ}
  (h_sum : a + b + c + d + e = 256)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h_diff : 2 ≤ (b - a) ∧ (b - a) ≤ 10 ∧ 
            2 ≤ (c - b) ∧ (c - b) ≤ 10 ∧ 
            2 ≤ (d - c) ∧ (d - c) ≤ 10 ∧ 
            2 ≤ (e - d) ∧ (e - d) ≤ 10) : 
  a = 32 :=
sorry

end min_age_of_youngest_person_l101_101937


namespace spending_on_hydrangeas_l101_101936

def lily_spending : ℕ :=
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  let years := end_year - start_year
  cost_per_plant * years

theorem spending_on_hydrangeas : lily_spending = 640 := 
  sorry

end spending_on_hydrangeas_l101_101936


namespace seven_not_spheric_spheric_power_spheric_l101_101268

def is_spheric (r : ℚ) : Prop := ∃ x y z : ℚ, r = x^2 + y^2 + z^2

theorem seven_not_spheric : ¬ is_spheric 7 := 
sorry

theorem spheric_power_spheric (r : ℚ) (n : ℕ) (h : is_spheric r) (hn : n > 1) : is_spheric (r ^ n) := 
sorry

end seven_not_spheric_spheric_power_spheric_l101_101268


namespace total_share_amount_l101_101432

theorem total_share_amount (x y z : ℝ) (hx : y = 0.45 * x) (hz : z = 0.30 * x) (hy_share : y = 63) : x + y + z = 245 := by
  sorry

end total_share_amount_l101_101432


namespace henry_books_donation_l101_101713

theorem henry_books_donation
  (initial_books : ℕ := 99)
  (room_books : ℕ := 21)
  (coffee_table_books : ℕ := 4)
  (cookbook_books : ℕ := 18)
  (boxes : ℕ := 3)
  (picked_up_books : ℕ := 12)
  (final_books : ℕ := 23) :
  (initial_books - final_books + picked_up_books - (room_books + coffee_table_books + cookbook_books)) / boxes = 15 :=
by
  sorry

end henry_books_donation_l101_101713


namespace find_m_l101_101011

-- Definition of vectors in terms of the condition
def vec_a (m : ℝ) : ℝ × ℝ := (2 * m + 1, m)
def vec_b (m : ℝ) : ℝ × ℝ := (1, m)

-- Condition that vectors a and b are perpendicular
def perpendicular (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2) = 0

-- Problem statement: find m such that vec_a is perpendicular to vec_b
theorem find_m (m : ℝ) (h : perpendicular (vec_a m) (vec_b m)) : m = -1 := by
  sorry

end find_m_l101_101011


namespace find_k_l101_101915

theorem find_k : ∃ k : ℝ, (3 * k - 4) / (k + 7) = 2 / 5 ∧ k = 34 / 13 :=
by
  use 34 / 13
  sorry

end find_k_l101_101915


namespace find_denominator_l101_101783

theorem find_denominator (y x : ℝ) (hy : y > 0) (h : (1 * y) / x + (3 * y) / 10 = 0.35 * y) : x = 20 := by
  sorry

end find_denominator_l101_101783


namespace negative_solutions_iff_l101_101605

theorem negative_solutions_iff (m x y : ℝ) (h1 : x - y = 2 * m + 7) (h2 : x + y = 4 * m - 3) :
  (x < 0 ∧ y < 0) ↔ m < -2 / 3 :=
by
  sorry

end negative_solutions_iff_l101_101605


namespace find_dividend_l101_101441

theorem find_dividend (dividend divisor quotient : ℕ) 
  (h_sum : dividend + divisor + quotient = 103)
  (h_quotient : quotient = 3)
  (h_divisor : divisor = dividend / quotient) : 
  dividend = 75 :=
by
  rw [h_quotient, h_divisor] at h_sum
  sorry

end find_dividend_l101_101441


namespace smallest_k_divides_ab_l101_101001

theorem smallest_k_divides_ab (S : Finset ℕ) (hS : S = Finset.range 51)
  (k : ℕ) : (∀ T : Finset ℕ, T ⊆ S → T.card = k → ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ (a + b) ∣ (a * b)) ↔ k = 39 :=
by
  sorry

end smallest_k_divides_ab_l101_101001


namespace equidistant_point_x_axis_l101_101782

theorem equidistant_point_x_axis (x : ℝ) (C D : ℝ × ℝ)
  (hC : C = (-3, 0))
  (hD : D = (0, 5))
  (heqdist : ∀ p : ℝ × ℝ, p.2 = 0 → 
    dist p C = dist p D) :
  x = 8 / 3 :=
by
  sorry

end equidistant_point_x_axis_l101_101782


namespace afternoon_sales_l101_101097

theorem afternoon_sales :
  ∀ (morning_sold afternoon_sold total_sold : ℕ),
    afternoon_sold = 2 * morning_sold ∧
    total_sold = morning_sold + afternoon_sold ∧
    total_sold = 510 →
    afternoon_sold = 340 :=
by
  intros morning_sold afternoon_sold total_sold h
  sorry

end afternoon_sales_l101_101097


namespace subtracting_seven_percent_l101_101593

theorem subtracting_seven_percent (a : ℝ) : a - 0.07 * a = 0.93 * a :=
by 
  sorry

end subtracting_seven_percent_l101_101593


namespace moles_of_KOH_used_l101_101407

variable {n_KOH : ℝ}

theorem moles_of_KOH_used :
  ∃ n_KOH, (NH4I + KOH = KI_produced) → (KI_produced = 1) → n_KOH = 1 :=
by
  sorry

end moles_of_KOH_used_l101_101407


namespace negation_of_proposition_l101_101180

theorem negation_of_proposition :
  ¬(∀ n : ℤ, (∃ k : ℤ, n = 2 * k) → (∃ m : ℤ, n = 2 * m)) ↔ ∃ n : ℤ, (∃ k : ℤ, n = 2 * k) ∧ ¬(∃ m : ℤ, n = 2 * m) := 
sorry

end negation_of_proposition_l101_101180


namespace negation_of_proposition_l101_101652

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), a > b → a^2 > b^2) ↔ ∃ (a b : ℝ), a ≤ b ∧ a^2 ≤ b^2 :=
sorry

end negation_of_proposition_l101_101652


namespace stella_annual_income_l101_101883

-- Define the conditions
def monthly_income : ℕ := 4919
def unpaid_leave_months : ℕ := 2
def total_months : ℕ := 12

-- The question: What is Stella's annual income last year?
def annual_income (monthly_income : ℕ) (worked_months : ℕ) : ℕ :=
  monthly_income * worked_months

-- Prove that Stella's annual income last year was $49190
theorem stella_annual_income : annual_income monthly_income (total_months - unpaid_leave_months) = 49190 :=
by
  sorry

end stella_annual_income_l101_101883


namespace sam_initial_watermelons_l101_101375

theorem sam_initial_watermelons (x : ℕ) (h : x + 3 = 7) : x = 4 :=
by
  -- proof steps would go here
  sorry

end sam_initial_watermelons_l101_101375


namespace total_bananas_in_collection_l101_101169

theorem total_bananas_in_collection (groups_of_bananas : ℕ) (bananas_per_group : ℕ) 
    (h1 : groups_of_bananas = 7) (h2 : bananas_per_group = 29) :
    groups_of_bananas * bananas_per_group = 203 := by
  sorry

end total_bananas_in_collection_l101_101169


namespace max_three_topping_pizzas_l101_101025

-- Define the combinations function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Assert the condition and the question with the expected answer
theorem max_three_topping_pizzas : combination 8 3 = 56 :=
by
  sorry

end max_three_topping_pizzas_l101_101025


namespace time_spent_cutting_hair_l101_101256

theorem time_spent_cutting_hair :
  let women's_time := 50
  let men's_time := 15
  let children's_time := 25
  let women's_haircuts := 3
  let men's_haircuts := 2
  let children's_haircuts := 3
  women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255 :=
by
  -- Definitions
  let women's_time       := 50
  let men's_time         := 15
  let children's_time    := 25
  let women's_haircuts   := 3
  let men's_haircuts     := 2
  let children's_haircuts := 3
  
  show women's_haircuts * women's_time + men's_haircuts * men's_time + children's_haircuts * children's_time = 255
  sorry

end time_spent_cutting_hair_l101_101256


namespace insurance_coverage_is_80_percent_l101_101215

-- Definitions and conditions
def MRI_cost : ℕ := 1200
def doctor_hourly_fee : ℕ := 300
def doctor_examination_time : ℕ := 30  -- in minutes
def seen_fee : ℕ := 150
def amount_paid_by_tim : ℕ := 300

-- The total cost calculation
def total_cost : ℕ := MRI_cost + (doctor_hourly_fee * doctor_examination_time / 60) + seen_fee

-- The amount covered by insurance
def amount_covered_by_insurance : ℕ := total_cost - amount_paid_by_tim

-- The percentage of coverage by insurance
def insurance_coverage_percentage : ℕ := (amount_covered_by_insurance * 100) / total_cost

theorem insurance_coverage_is_80_percent : insurance_coverage_percentage = 80 := by
  sorry

end insurance_coverage_is_80_percent_l101_101215


namespace unique_solution_positive_integers_l101_101279

theorem unique_solution_positive_integers :
  ∀ (a b : ℕ), (0 < a ∧ 0 < b ∧ ∃ k m : ℤ, a^3 + 6 * a * b + 1 = k^3 ∧ b^3 + 6 * a * b + 1 = m^3) → (a = 1 ∧ b = 1) :=
by
  -- Proof goes here
  sorry

end unique_solution_positive_integers_l101_101279


namespace linear_function_iff_l101_101957

variable {x : ℝ} (m : ℝ)

def f (m : ℝ) (x : ℝ) : ℝ := (m + 2) * x + 4 * x - 5

theorem linear_function_iff (m : ℝ) : 
  (∃ c d, ∀ x, f m x = c * x + d) ↔ m ≠ -6 :=
by 
  sorry

end linear_function_iff_l101_101957


namespace two_digit_numbers_count_l101_101998

theorem two_digit_numbers_count : 
  ∃ (count : ℕ), (
    (∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ b = 2 * a → 
      (10 * b + a = 7 / 4 * (10 * a + b))) 
      ∧ count = 4
  ) :=
sorry

end two_digit_numbers_count_l101_101998


namespace arithmetic_sequence_general_term_l101_101423

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (h1 : a 2 = 4) 
  (h2 : a 4 + a 7 = 15) : 
  ∃ d : ℝ, ∀ n : ℕ, a n = n + 2 := 
by
  sorry

end arithmetic_sequence_general_term_l101_101423


namespace difference_between_numbers_l101_101270

theorem difference_between_numbers (a b : ℕ) (h1 : a + b = 20000) (h2 : b = 2 * a + 6) (h3 : 9 ∣ a) : b - a = 6670 :=
by
  sorry

end difference_between_numbers_l101_101270


namespace simplify_and_evaluate_l101_101073

-- Define the expression
def expression (x : ℝ) := -(2 * x^2 + 3 * x) + 2 * (4 * x + x^2)

-- State the theorem
theorem simplify_and_evaluate : expression (-2) = -10 :=
by
  -- The proof goes here
  sorry

end simplify_and_evaluate_l101_101073


namespace complement_union_A_B_is_correct_l101_101815

-- Define the set of real numbers R
def R : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x | ∃ (y : ℝ), y = Real.log (x + 3) }

-- Simplified definition for A to reflect x > -3
def A_simplified : Set ℝ := { x | x > -3 }

-- Define set B
def B : Set ℝ := { x | x ≥ 2 }

-- Define the union of A and B
def union_A_B : Set ℝ := A_simplified ∪ B

-- Define the complement of the union in R
def complement_R_union_A_B : Set ℝ := R \ union_A_B

-- State the theorem
theorem complement_union_A_B_is_correct :
  complement_R_union_A_B = { x | x ≤ -3 } := by
  sorry

end complement_union_A_B_is_correct_l101_101815


namespace range_of_m_l101_101567

def set_A : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 5 }
def set_B (m : ℝ) : Set ℝ := { x : ℝ | (2 * m - 1) ≤ x ∧ x ≤ (2 * m + 1) }

theorem range_of_m (m : ℝ) : (set_B m ⊆ set_A) ↔ (-1 / 2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l101_101567


namespace negation_of_universal_proposition_l101_101059

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, |x - 1| - |x + 1| ≤ 3) ↔ ∃ x : ℝ, |x - 1| - |x + 1| > 3 :=
by
  sorry

end negation_of_universal_proposition_l101_101059


namespace percentage_transform_l101_101914

theorem percentage_transform (n : ℝ) (h : 0.3 * 0.4 * n = 36) : 0.4 * 0.3 * n = 36 :=
by
  sorry

end percentage_transform_l101_101914


namespace isosceles_triangle_base_length_l101_101480

theorem isosceles_triangle_base_length
  (a b c : ℕ)
  (h_iso : a = b)
  (h_perimeter : a + b + c = 62)
  (h_leg_length : a = 25) :
  c = 12 :=
by
  sorry

end isosceles_triangle_base_length_l101_101480


namespace factorial_div_l101_101357

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l101_101357


namespace count_multiples_of_12_between_25_and_200_l101_101348

theorem count_multiples_of_12_between_25_and_200 :
  ∃ n, (∀ i, 25 < i ∧ i < 200 → (∃ k, i = 12 * k)) ↔ n = 14 :=
by
  sorry

end count_multiples_of_12_between_25_and_200_l101_101348


namespace smallest_n_l101_101628

theorem smallest_n (n : ℕ) (h1 : ∃ k : ℕ, 4 * n = k^2) (h2 : ∃ l : ℕ, 5 * n = l^3) : n = 100 :=
sorry

end smallest_n_l101_101628


namespace k_is_even_set_l101_101832

open Set -- using Set from Lean library

noncomputable def kSet (s : Set ℤ) :=
  (∀ g ∈ ({5, 8, 7, 1} : Set ℤ), ∀ k ∈ s, (g * k) % 2 = 0)

theorem k_is_even_set (s : Set ℤ) :
  (∀ g ∈ ({5, 8, 7, 1} : Set ℤ), ∀ k ∈ s, (g * k) % 2 = 0) →
  ∀ k ∈ s, k % 2 = 0 :=
by
  intro h
  sorry

end k_is_even_set_l101_101832


namespace largest_quadrilateral_angle_l101_101754

theorem largest_quadrilateral_angle (x : ℝ)
  (h1 : 3 * x + 4 * x + 5 * x + 6 * x = 360) :
  6 * x = 120 :=
by
  sorry

end largest_quadrilateral_angle_l101_101754


namespace x1_x2_product_l101_101821

theorem x1_x2_product (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : x1^2 - 2006 * x1 = 1) (h3 : x2^2 - 2006 * x2 = 1) : x1 * x2 = -1 := 
by
  sorry

end x1_x2_product_l101_101821


namespace remaining_garden_space_l101_101438

theorem remaining_garden_space : 
  let Area_rectangle := 20 * 18
  let Area_square_cutout := 4 * 4
  let Area_triangle := (1 / 2) * 3 * 2
  Area_rectangle - Area_square_cutout + Area_triangle = 347 :=
by
  let Area_rectangle := 20 * 18
  let Area_square_cutout := 4 * 4
  let Area_triangle := (1 / 2) * 3 * 2
  show Area_rectangle - Area_square_cutout + Area_triangle = 347
  sorry

end remaining_garden_space_l101_101438


namespace fewest_posts_l101_101985

def grazingAreaPosts (length width post_interval rock_wall_length : ℕ) : ℕ :=
  let side1 := width / post_interval + 1
  let side2 := length / post_interval
  side1 + 2 * side2

theorem fewest_posts (length width post_interval rock_wall_length posts : ℕ) :
  length = 70 ∧ width = 50 ∧ post_interval = 10 ∧ rock_wall_length = 150 ∧ posts = 18 →
  grazingAreaPosts length width post_interval rock_wall_length = posts := 
by
  intros h
  obtain ⟨hl, hw, hp, hr, ht⟩ := h
  simp [grazingAreaPosts, hl, hw, hp, hr]
  sorry

end fewest_posts_l101_101985


namespace harmony_implication_at_least_N_plus_1_zero_l101_101940

noncomputable def is_harmony (A B : ℕ → ℕ) (i : ℕ) : Prop :=
  A i = (1 / (2 * B i + 1)) * (Finset.range (2 * B i + 1)).sum (fun s => A (i + s - B i))

theorem harmony_implication_at_least_N_plus_1_zero {N : ℕ} (A B : ℕ → ℕ)
  (hN : N ≥ 2) 
  (h_nonneg_A : ∀ i, 0 ≤ A i)
  (h_nonneg_B : ∀ i, 0 ≤ B i)
  (h_periodic_A : ∀ i, A i = A ((i % N) + 1))
  (h_periodic_B : ∀ i, B i = B ((i % N) + 1))
  (h_harmony_AB : ∀ i, is_harmony A B i)
  (h_harmony_BA : ∀ i, is_harmony B A i)
  (h_not_constant_A : ¬ ∀ i j, A i = A j)
  (h_not_constant_B : ¬ ∀ i j, B i = B j) :
  Finset.card (Finset.filter (fun i => A i = 0 ∨ B i = 0) (Finset.range (N * 2))) ≥ N + 1 := by
  sorry

end harmony_implication_at_least_N_plus_1_zero_l101_101940


namespace probability_sum_of_10_l101_101037

theorem probability_sum_of_10 (total_outcomes : ℕ) 
  (h1 : total_outcomes = 6^4) : 
  (46 / total_outcomes) = 23 / 648 := by
  sorry

end probability_sum_of_10_l101_101037


namespace base9_problem_l101_101439

def base9_add (a : ℕ) (b : ℕ) : ℕ := sorry -- Define actual addition for base 9
def base9_mul (a : ℕ) (b : ℕ) : ℕ := sorry -- Define actual multiplication for base 9

theorem base9_problem : base9_mul (base9_add 35 273) 2 = 620 := sorry

end base9_problem_l101_101439


namespace original_number_from_sum_l101_101495

variable (a b c : ℕ) (m S : ℕ)

/-- Given a three-digit number, the magician asks the participant to add all permutations -/
def three_digit_number_permutations_sum (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c + (100 * a + 10 * c + b) + (100 * b + 10 * c + a) +
  (100 * b + 10 * a + c) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a)

/-- Given the sum of all permutations of the three-digit number is 4239, determine the original number -/
theorem original_number_from_sum (S : ℕ) (hS : S = 4239) (Sum_conditions : three_digit_number_permutations_sum a b c = S) :
  (100 * a + 10 * b + c) = 429 := by
  sorry

end original_number_from_sum_l101_101495


namespace tom_has_hours_to_spare_l101_101863

-- Conditions as definitions
def numberOfWalls : Nat := 5
def wallWidth : Nat := 2 -- in meters
def wallHeight : Nat := 3 -- in meters
def paintingRate : Nat := 10 -- in minutes per square meter
def totalAvailableTime : Nat := 10 -- in hours

-- Lean 4 statement of the problem
theorem tom_has_hours_to_spare :
  let areaOfOneWall := wallWidth * wallHeight -- 2 * 3
  let totalArea := numberOfWalls * areaOfOneWall -- 5 * (2 * 3)
  let totalTimeToPaint := (totalArea * paintingRate) / 60 -- (30 * 10) / 60
  totalAvailableTime - totalTimeToPaint = 5 :=
by
  sorry

end tom_has_hours_to_spare_l101_101863


namespace min_inequality_l101_101412

theorem min_inequality (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 2) :
  ∃ L, L = 9 / 4 ∧ (1 / (x + y) + 1 / (x + z) + 1 / (y + z) ≥ L) :=
sorry

end min_inequality_l101_101412


namespace quadratic_completing_square_sum_l101_101522

theorem quadratic_completing_square_sum (q t : ℝ) :
    (∃ (x : ℝ), 9 * x^2 - 54 * x - 36 = 0 ∧ (x + q)^2 = t) →
    q + t = 10 := sorry

end quadratic_completing_square_sum_l101_101522


namespace probability_neither_nearsighted_l101_101310

-- Definitions based on problem conditions
def P_A : ℝ := 0.4
def P_not_A : ℝ := 1 - P_A
def event_B₁_not_nearsighted : Prop := true
def event_B₂_not_nearsighted : Prop := true

-- Independence assumption
variables (indep_B₁_B₂ : event_B₁_not_nearsighted) (event_B₂_not_nearsighted)

-- Theorem statement
theorem probability_neither_nearsighted (H1 : P_A = 0.4) (H2 : P_not_A = 0.6)
  (indep_B₁_B₂ : event_B₁_not_nearsighted ∧ event_B₂_not_nearsighted) :
  P_not_A * P_not_A = 0.36 :=
by
  -- Proof omitted
  sorry

end probability_neither_nearsighted_l101_101310


namespace trigonometric_identity_l101_101921

theorem trigonometric_identity
  (α β : Real)
  (h : Real.cos α * Real.cos β - Real.sin α * Real.sin β = 0) :
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = 1 ∨
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = -1 := by
  sorry

end trigonometric_identity_l101_101921


namespace negation_prop_equiv_l101_101592

variable (a : ℝ)

theorem negation_prop_equiv :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 - 2 * a * x - 1 < 0) ↔ (∀ x : ℝ, x > 1 → x^2 - 2 * a * x - 1 ≥ 0) :=
sorry

end negation_prop_equiv_l101_101592


namespace triangle_inequality_problem_l101_101555

-- Define the problem statement: Given the specified conditions, prove the interval length and sum
theorem triangle_inequality_problem :
  ∀ (A B C D : Type) (AB AC BC BD CD AD AO : ℝ),
  AB = 12 ∧ CD = 4 →
  (∃ x : ℝ, (4 < x ∧ x < 24) ∧ (AC = x ∧ m = 4 ∧ n = 24 ∧ m + n = 28)) :=
by
  intro A B C D AB AC BC BD CD AD AO h
  sorry

end triangle_inequality_problem_l101_101555


namespace carnival_earnings_l101_101160

theorem carnival_earnings (days : ℕ) (total_earnings : ℕ) (h1 : days = 22) (h2 : total_earnings = 3168) : 
  (total_earnings / days) = 144 := 
by
  -- The proof would go here
  sorry

end carnival_earnings_l101_101160


namespace renovation_cost_distribution_l101_101814

/-- A mathematical proof that if Team A works alone for 3 weeks, followed by both Team A and Team B working together, and the total renovation cost is 4000 yuan, then the payment should be distributed equally between Team A and Team B, each receiving 2000 yuan. -/
theorem renovation_cost_distribution :
  let time_A := 18
  let time_B := 12
  let initial_time_A := 3
  let total_cost := 4000
  ∃ x, (1 / time_A * (x + initial_time_A) + 1 / time_B * x = 1) ∧
       let work_A := 1 / time_A * (x + initial_time_A)
       let work_B := 1 / time_B * x
       work_A = work_B ∧
       total_cost / 2 = 2000 :=
by
  sorry

end renovation_cost_distribution_l101_101814


namespace equivalent_systems_solution_and_value_l101_101445

-- Definitions for the conditions
def system1 (x y a b : ℝ) : Prop := 
  (2 * (x + 1) - y = 7) ∧ (x + b * y = a)

def system2 (x y a b : ℝ) : Prop := 
  (a * x + y = b) ∧ (3 * x + 2 * (y - 1) = 9)

-- The proof problem as a Lean 4 statement
theorem equivalent_systems_solution_and_value (a b : ℝ) :
  (∃ x y : ℝ, system1 x y a b ∧ system2 x y a b) →
  ((∃ x y : ℝ, x = 3 ∧ y = 1) ∧ (3 * a - b) ^ 2023 = -1) :=
  by sorry

end equivalent_systems_solution_and_value_l101_101445


namespace contrapositive_proof_l101_101201

theorem contrapositive_proof (a : ℝ) (h : a ≤ 2 → a^2 ≤ 4) : a > 2 → a^2 > 4 :=
by
  intros ha
  sorry

end contrapositive_proof_l101_101201


namespace woman_lawyer_probability_l101_101094

-- Defining conditions
def total_members : ℝ := 100
def percent_women : ℝ := 0.90
def percent_women_lawyers : ℝ := 0.60

-- Calculating numbers based on the percentages
def number_women : ℝ := percent_women * total_members
def number_women_lawyers : ℝ := percent_women_lawyers * number_women

-- Statement of the problem in Lean 4
theorem woman_lawyer_probability :
  (number_women_lawyers / total_members) = 0.54 :=
by sorry

end woman_lawyer_probability_l101_101094


namespace total_intersections_l101_101873

def north_south_streets : ℕ := 10
def east_west_streets : ℕ := 10

theorem total_intersections :
  (north_south_streets * east_west_streets = 100) :=
by
  sorry

end total_intersections_l101_101873


namespace prove_B_is_guilty_l101_101625

variables (A B C : Prop)

def guilty_conditions (A B C : Prop) : Prop :=
  (A → ¬ B → C) ∧
  (C → B ∨ A) ∧
  (A → ¬ (A ∧ C)) ∧
  (A ∨ B ∨ C) ∧ 
  ¬ (¬ A ∧ ¬ B ∧ ¬ C)

theorem prove_B_is_guilty : guilty_conditions A B C → B :=
by
  intros h
  sorry

end prove_B_is_guilty_l101_101625


namespace rug_area_is_24_l101_101942

def length_floor : ℕ := 12
def width_floor : ℕ := 10
def strip_width : ℕ := 3

theorem rug_area_is_24 :
  (length_floor - 2 * strip_width) * (width_floor - 2 * strip_width) = 24 := 
by
  sorry

end rug_area_is_24_l101_101942


namespace find_minimal_sum_n_l101_101319

noncomputable def minimal_sum_n {a : ℕ → ℤ} {S : ℕ → ℤ} (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 = -9) (h3 : S 3 = S 7) : ℕ := 
     5

theorem find_minimal_sum_n (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 = -9) (h3 : S 3 = S 7) : minimal_sum_n h1 h2 h3 = 5 :=
    sorry

end find_minimal_sum_n_l101_101319


namespace aunt_may_milk_left_l101_101428

def morningMilkProduction (numCows numGoats numSheep : ℕ) (cowMilk goatMilk sheepMilk : ℝ) : ℝ :=
  numCows * cowMilk + numGoats * goatMilk + numSheep * sheepMilk

def eveningMilkProduction (numCows numGoats numSheep : ℕ) (cowMilk goatMilk sheepMilk : ℝ) : ℝ :=
  numCows * cowMilk + numGoats * goatMilk + numSheep * sheepMilk

def spoiledMilk (milkProduction : ℝ) (spoilageRate : ℝ) : ℝ :=
  milkProduction * spoilageRate

def freshMilk (totalMilk spoiledMilk : ℝ) : ℝ :=
  totalMilk - spoiledMilk

def soldMilk (freshMilk : ℝ) (saleRate : ℝ) : ℝ :=
  freshMilk * saleRate

def milkLeft (freshMilk soldMilk : ℝ) : ℝ :=
  freshMilk - soldMilk

noncomputable def totalMilkLeft (previousLeftover : ℝ) (morningLeft eveningLeft : ℝ) : ℝ :=
  previousLeftover + morningLeft + eveningLeft

theorem aunt_may_milk_left :
  let numCows := 5
  let numGoats := 4
  let numSheep := 10
  let cowMilkMorning := 13
  let goatMilkMorning := 0.5
  let sheepMilkMorning := 0.25
  let cowMilkEvening := 14
  let goatMilkEvening := 0.6
  let sheepMilkEvening := 0.2
  let morningSpoilageRate := 0.10
  let eveningSpoilageRate := 0.05
  let iceCreamSaleRate := 0.70
  let cheeseShopSaleRate := 0.80
  let previousLeftover := 15
  let morningMilk := morningMilkProduction numCows numGoats numSheep cowMilkMorning goatMilkMorning sheepMilkMorning
  let eveningMilk := eveningMilkProduction numCows numGoats numSheep cowMilkEvening goatMilkEvening sheepMilkEvening
  let morningSpoiled := spoiledMilk morningMilk morningSpoilageRate
  let eveningSpoiled := spoiledMilk eveningMilk eveningSpoilageRate
  let freshMorningMilk := freshMilk morningMilk morningSpoiled
  let freshEveningMilk := freshMilk eveningMilk eveningSpoiled
  let morningSold := soldMilk freshMorningMilk iceCreamSaleRate
  let eveningSold := soldMilk freshEveningMilk cheeseShopSaleRate
  let morningLeft := milkLeft freshMorningMilk morningSold
  let eveningLeft := milkLeft freshEveningMilk eveningSold
  totalMilkLeft previousLeftover morningLeft eveningLeft = 47.901 :=
by
  sorry

end aunt_may_milk_left_l101_101428


namespace stadium_length_in_feet_l101_101236

theorem stadium_length_in_feet (length_in_yards : ℕ) (conversion_factor : ℕ) (h1 : length_in_yards = 62) (h2 : conversion_factor = 3) : length_in_yards * conversion_factor = 186 :=
by
  sorry

end stadium_length_in_feet_l101_101236


namespace find_sum_of_smallest_multiples_l101_101800

-- Define c as the smallest positive two-digit multiple of 5
def is_smallest_two_digit_multiple_of_5 (c : ℕ) : Prop :=
  c ≥ 10 ∧ c % 5 = 0 ∧ ∀ n, (n ≥ 10 ∧ n % 5 = 0) → n ≥ c

-- Define d as the smallest positive three-digit multiple of 7
def is_smallest_three_digit_multiple_of_7 (d : ℕ) : Prop :=
  d ≥ 100 ∧ d % 7 = 0 ∧ ∀ n, (n ≥ 100 ∧ n % 7 = 0) → n ≥ d

theorem find_sum_of_smallest_multiples :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_sum_of_smallest_multiples_l101_101800


namespace beth_overall_score_l101_101659

-- Definitions for conditions
def percent_score (score_pct : ℕ) (total_problems : ℕ) : ℕ :=
  (score_pct * total_problems) / 100

def total_correct_answers : ℕ :=
  percent_score 60 15 + percent_score 85 20 + percent_score 75 25

def total_problems : ℕ := 15 + 20 + 25

def combined_percentage : ℕ :=
  (total_correct_answers * 100) / total_problems

-- The statement to be proved
theorem beth_overall_score : combined_percentage = 75 := by
  sorry

end beth_overall_score_l101_101659


namespace train_time_36kmph_200m_l101_101597

/-- How many seconds will a train 200 meters long running at the rate of 36 kmph take to pass a certain telegraph post? -/
def time_to_pass_post (length_of_train : ℕ) (speed_kmph : ℕ) : ℕ :=
  length_of_train * 3600 / (speed_kmph * 1000)

theorem train_time_36kmph_200m : time_to_pass_post 200 36 = 20 := by
  sorry

end train_time_36kmph_200m_l101_101597


namespace distinct_positive_roots_log_sum_eq_5_l101_101678

theorem distinct_positive_roots_log_sum_eq_5 (a b : ℝ)
  (h : ∀ (x : ℝ), (8 * x ^ 3 + 6 * a * x ^ 2 + 3 * b * x + a = 0) → x > 0) 
  (h_sum : ∀ u v w : ℝ, (8 * u ^ 3 + 6 * a * u ^ 2 + 3 * b * u + a = 0) ∧
                       (8 * v ^ 3 + 6 * a * v ^ 2 + 3 * b * v + a = 0) ∧
                       (8 * w ^ 3 + 6 * a * w ^ 2 + 3 * b * w + a = 0) → 
                       u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ 
                       (Real.log (u) / Real.log (3) + Real.log (v) / Real.log (3) + Real.log (w) / Real.log (3) = 5)) :
  a = -1944 :=
sorry

end distinct_positive_roots_log_sum_eq_5_l101_101678


namespace total_shaded_area_l101_101723

theorem total_shaded_area (side_len : ℝ) (segment_len : ℝ) (h : ℝ) :
  side_len = 8 ∧ segment_len = 1 ∧ 0 ≤ h ∧ h ≤ 8 →
  (segment_len * h / 2 + segment_len * (side_len - h) / 2) = 4 := 
by
  intro h_cond
  rcases h_cond with ⟨h_side_len, h_segment_len, h_nonneg, h_le⟩
  -- Directly state the simplified computation
  sorry

end total_shaded_area_l101_101723


namespace total_fence_length_l101_101472

variable (Darren Doug : ℝ)

-- Definitions based on given conditions
def Darren_paints_more := Darren = 1.20 * Doug
def Darren_paints_360 := Darren = 360

-- The statement to prove
theorem total_fence_length (h1 : Darren_paints_more Darren Doug) (h2 : Darren_paints_360 Darren) : (Darren + Doug) = 660 := 
by
  sorry

end total_fence_length_l101_101472


namespace solve_for_n_l101_101262

theorem solve_for_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 34) : n = 7 :=
by
  sorry

end solve_for_n_l101_101262


namespace longest_segment_in_cylinder_l101_101102

theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  ∃ l, l = 10 * Real.sqrt 2 ∧
  (∀ x y z, (x = r * 2) ∧ (y = h) → z = Real.sqrt (x^2 + y^2) → z ≤ l) :=
sorry

end longest_segment_in_cylinder_l101_101102


namespace number_of_sophomores_l101_101569

theorem number_of_sophomores (n x : ℕ) (freshmen seniors selected freshmen_selected : ℕ)
  (h_freshmen : freshmen = 450)
  (h_seniors : seniors = 250)
  (h_selected : selected = 60)
  (h_freshmen_selected : freshmen_selected = 27)
  (h_eq : selected / (freshmen + seniors + x) = freshmen_selected / freshmen) :
  x = 300 := by
  sorry

end number_of_sophomores_l101_101569


namespace fraction_of_orange_juice_in_large_container_l101_101223

def total_capacity := 800 -- mL for each pitcher
def orange_juice_first_pitcher := total_capacity / 2 -- 400 mL
def orange_juice_second_pitcher := total_capacity / 4 -- 200 mL
def total_orange_juice := orange_juice_first_pitcher + orange_juice_second_pitcher -- 600 mL
def total_volume := total_capacity + total_capacity -- 1600 mL

theorem fraction_of_orange_juice_in_large_container :
  (total_orange_juice / total_volume) = 3 / 8 :=
by
  sorry

end fraction_of_orange_juice_in_large_container_l101_101223


namespace min_focal_length_l101_101784

theorem min_focal_length (a b c : ℝ) (h : a > 0 ∧ b > 0) 
    (hyperbola_eq : ∀ x y, ((x^2 / a^2) - (y^2 / b^2) = 1))
    (line_intersects_asymptotes_at : x = a)
    (area_of_triangle : 1/2 * a * (2 * b) = 8) :
    2 * c = 8 :=
by
  sorry

end min_focal_length_l101_101784


namespace hypothesis_test_l101_101737

def X : List ℕ := [3, 4, 6, 10, 13, 17]
def Y : List ℕ := [1, 2, 5, 7, 16, 20, 22]

def alpha : ℝ := 0.01
def W_lower : ℕ := 24
def W_upper : ℕ := 60
def W1 : ℕ := 41

-- stating the null hypothesis test condition
theorem hypothesis_test : (24 < 41) ∧ (41 < 60) :=
by
  sorry

end hypothesis_test_l101_101737


namespace problem_part1_problem_part2_l101_101645

theorem problem_part1 :
  ∀ m : ℝ, (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 :=
by
sorry

theorem problem_part2 :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    4 * a^2 + 9 * b^2 + c^2 ≥ 36 / 49 ∧
    (4 * a^2 + 9 * b^2 + c^2 = 36 / 49 ↔ a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49) :=
by
sorry

end problem_part1_problem_part2_l101_101645


namespace num_triples_l101_101124

/-- Theorem statement:
There are exactly 2 triples of positive integers (a, b, c) satisfying the conditions:
1. ab + ac = 60
2. bc + ac = 36
3. ab + bc = 48
--/
theorem num_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (ab + ac = 60) → (bc + ac = 36) → (ab + bc = 48) → 
  (a, b, c) ∈ [(1, 4, 8), (1, 12, 3)] →
  ∃! (a b c : ℕ), (ab + ac = 60) ∧ (bc + ac = 36) ∧ (ab + bc = 48) :=
sorry

end num_triples_l101_101124


namespace set_intersection_complement_l101_101321

def U := {x : ℝ | x > -3}
def A := {x : ℝ | x < -2 ∨ x > 3}
def B := {x : ℝ | -1 ≤ x ∧ x ≤ 4}

theorem set_intersection_complement :
  A ∩ (U \ B) = {x : ℝ | -3 < x ∧ x < -2 ∨ x > 4} :=
by sorry

end set_intersection_complement_l101_101321


namespace sales_tax_difference_l101_101899

-- Definitions for the price and tax rates
def item_price : ℝ := 50
def tax_rate1 : ℝ := 0.065
def tax_rate2 : ℝ := 0.06
def tax_rate3 : ℝ := 0.07

-- Sales tax amounts derived from the given rates and item price
def tax_amount (rate : ℝ) (price : ℝ) : ℝ := rate * price

-- Calculate the individual tax amounts
def tax_amount1 : ℝ := tax_amount tax_rate1 item_price
def tax_amount2 : ℝ := tax_amount tax_rate2 item_price
def tax_amount3 : ℝ := tax_amount tax_rate3 item_price

-- Proposition stating the proof problem
theorem sales_tax_difference :
  max tax_amount1 (max tax_amount2 tax_amount3) - min tax_amount1 (min tax_amount2 tax_amount3) = 0.50 :=
by 
  sorry

end sales_tax_difference_l101_101899


namespace three_digit_number_division_l101_101885

theorem three_digit_number_division :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∃ m : ℕ, 10 ≤ m ∧ m < 100 ∧ n / m = 8 ∧ n % m = 6) → n = 342 :=
by
  sorry

end three_digit_number_division_l101_101885


namespace john_caffeine_consumption_l101_101944

noncomputable def caffeine_consumed : ℝ :=
let drink1_ounces : ℝ := 12
let drink1_caffeine : ℝ := 250
let drink2_ratio : ℝ := 3
let drink2_ounces : ℝ := 2

-- Calculate caffeine per ounce in the first drink
let caffeine1_per_ounce : ℝ := drink1_caffeine / drink1_ounces

-- Calculate caffeine per ounce in the second drink
let caffeine2_per_ounce : ℝ := caffeine1_per_ounce * drink2_ratio

-- Calculate total caffeine in the second drink
let drink2_caffeine : ℝ := caffeine2_per_ounce * drink2_ounces

-- Total caffeine from both drinks
let total_drinks_caffeine : ℝ := drink1_caffeine + drink2_caffeine

-- Caffeine in the pill is as much as the total from both drinks
let pill_caffeine : ℝ := total_drinks_caffeine

-- Total caffeine consumed
(drink1_caffeine + drink2_caffeine) + pill_caffeine

theorem john_caffeine_consumption :
  caffeine_consumed = 749.96 := by
    -- Proof is omitted
    sorry

end john_caffeine_consumption_l101_101944


namespace russian_needed_goals_equals_tunisian_scored_goals_l101_101214

-- Define the total goals required by each team
def russian_goals := 9
def tunisian_goals := 5

-- Statement: there exists a moment where the Russian remaining goals equal the Tunisian scored goals
theorem russian_needed_goals_equals_tunisian_scored_goals :
  ∃ n : ℕ, n ≤ russian_goals ∧ (russian_goals - n) = (tunisian_goals) := by
  sorry

end russian_needed_goals_equals_tunisian_scored_goals_l101_101214


namespace car_turns_proof_l101_101909

def turns_opposite_direction (angle1 angle2 : ℝ) : Prop :=
  angle1 + angle2 = 180

theorem car_turns_proof
  (angle1 angle2 : ℝ)
  (h1 : (angle1 = 50 ∧ angle2 = 130) ∨ (angle1 = -50 ∧ angle2 = 130) ∨ 
       (angle1 = 50 ∧ angle2 = -130) ∨ (angle1 = 30 ∧ angle2 = -30)) :
  turns_opposite_direction angle1 angle2 ↔ (angle1 = 50 ∧ angle2 = 130) :=
by
  sorry

end car_turns_proof_l101_101909


namespace mateo_orange_bottles_is_1_l101_101641

def number_of_orange_bottles_mateo_has (mateo_orange : ℕ) : Prop :=
  let julios_orange_bottles := 4
  let julios_grape_bottles := 7
  let mateos_grape_bottles := 3
  let liters_per_bottle := 2
  let julios_total_liters := (julios_orange_bottles + julios_grape_bottles) * liters_per_bottle
  let mateos_grape_liters := mateos_grape_bottles * liters_per_bottle
  let mateos_total_liters := (mateo_orange * liters_per_bottle) + mateos_grape_liters
  let additional_liters_to_julio := 14
  julios_total_liters = mateos_total_liters + additional_liters_to_julio

/-
Prove that Mateo has exactly 1 bottle of orange soda (assuming the problem above)
-/
theorem mateo_orange_bottles_is_1 : number_of_orange_bottles_mateo_has 1 :=
sorry

end mateo_orange_bottles_is_1_l101_101641


namespace calculate_f_at_2_l101_101904

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem calculate_f_at_2
  (a b : ℝ)
  (h_extremum : 3 + 2 * a + b = 0)
  (h_f1 : f 1 a b = 10) :
  f 2 a b = 18 :=
sorry

end calculate_f_at_2_l101_101904


namespace range_of_a_l101_101093

noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x

theorem range_of_a (a : ℝ) (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → a < g x) : a < 0 := 
by sorry

end range_of_a_l101_101093


namespace luncheon_tables_needed_l101_101574

theorem luncheon_tables_needed (invited : ℕ) (no_show : ℕ) (people_per_table : ℕ) (people_attended : ℕ) (tables_needed : ℕ) :
  invited = 47 →
  no_show = 7 →
  people_per_table = 5 →
  people_attended = invited - no_show →
  tables_needed = people_attended / people_per_table →
  tables_needed = 8 := by {
  -- Proof here
  sorry
}

end luncheon_tables_needed_l101_101574


namespace range_of_m_perimeter_of_isosceles_triangle_l101_101198

-- Define the variables for the lengths of the sides and the range of m
variables (AB BC AC : ℝ) (m : ℝ)

-- Conditions given in the problem
def triangle_conditions (AB BC : ℝ) (AC : ℝ) (m : ℝ) : Prop :=
  AB = 17 ∧ BC = 8 ∧ AC = 2 * m - 1

-- Proof that the range for m is between 5 and 13
theorem range_of_m (AB BC : ℝ) (m : ℝ) (h : triangle_conditions AB BC (2 * m - 1) m) : 
  5 < m ∧ m < 13 :=
by
  sorry

-- Proof that the perimeter is 42 when triangle is isosceles with given conditions
theorem perimeter_of_isosceles_triangle (AB BC AC : ℝ) (h : triangle_conditions AB BC AC 0) : 
  (AB = AC ∨ BC = AC) → (2 * AB + BC = 42) :=
by
  sorry

end range_of_m_perimeter_of_isosceles_triangle_l101_101198


namespace overall_gain_percent_l101_101034

theorem overall_gain_percent (cp1 cp2 cp3: ℝ) (sp1 sp2 sp3: ℝ) (h1: cp1 = 840) (h2: cp2 = 1350) (h3: cp3 = 2250) (h4: sp1 = 1220) (h5: sp2 = 1550) (h6: sp3 = 2150) : 
  (sp1 + sp2 + sp3 - (cp1 + cp2 + cp3)) / (cp1 + cp2 + cp3) * 100 = 10.81 := 
by 
  sorry

end overall_gain_percent_l101_101034


namespace max_min_diff_c_l101_101534

theorem max_min_diff_c {a b c : ℝ} 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 15) : 
  (∃ c_max c_min, 
    (∀ a b c, a + b + c = 3 ∧ a^2 + b^2 + c^2 = 15 → c_min ≤ c ∧ c ≤ c_max) ∧ 
    c_max - c_min = 16 / 3) :=
sorry

end max_min_diff_c_l101_101534


namespace rows_in_initial_patios_l101_101450

theorem rows_in_initial_patios (r c : ℕ) (h1 : r * c = 60) (h2 : (2 * c : ℚ) / r = 3 / 2) (h3 : (r + 5) * (c - 3) = 60) : r = 10 :=
sorry

end rows_in_initial_patios_l101_101450


namespace total_time_simultaneous_l101_101888

def total_time_bread1 : Nat := 30 + 120 + 20 + 120 + 10 + 30 + 30 + 15
def total_time_bread2 : Nat := 90 + 15 + 20 + 25 + 10
def total_time_bread3 : Nat := 40 + 100 + 5 + 110 + 15 + 5 + 25 + 20

theorem total_time_simultaneous :
  max (max total_time_bread1 total_time_bread2) total_time_bread3 = 375 :=
by
  sorry

end total_time_simultaneous_l101_101888


namespace part1_part2_l101_101115

-- Defining set A
def A : Set ℝ := {x | x^2 + 4 * x = 0}

-- Defining set B parameterized by a
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

-- Problem 1: Prove that if A ∩ B = A ∪ B, then a = 1
theorem part1 (a : ℝ) : (A ∩ (B a) = A ∪ (B a)) → a = 1 := by
  sorry

-- Problem 2: Prove the range of values for a if A ∩ B = B
theorem part2 (a : ℝ) : (A ∩ (B a) = B a) → a ∈ Set.Iic (-1) ∪ {1} := by
  sorry

end part1_part2_l101_101115


namespace initial_ratio_is_four_five_l101_101216

variable (M W : ℕ)

axiom initial_conditions :
  (M + 2 = 14) ∧ (2 * (W - 3) = 24)

theorem initial_ratio_is_four_five 
  (h : M + 2 = 14) 
  (k : 2 * (W - 3) = 24) : M / W = 4 / 5 :=
by
  sorry

end initial_ratio_is_four_five_l101_101216


namespace find_b_l101_101890

noncomputable def complex_b_value (i : ℂ) (b : ℝ) : Prop :=
(1 + b * i) * i = 1 + i

theorem find_b (i : ℂ) (b : ℝ) (hi : i^2 = -1) (h : complex_b_value i b) : b = -1 :=
by {
  sorry
}

end find_b_l101_101890


namespace find_d_squared_plus_e_squared_l101_101065

theorem find_d_squared_plus_e_squared {a b c d e : ℕ} 
  (h1 : (a + 1) * (3 * b * c + 1) = d + 3 * e + 1)
  (h2 : (b + 1) * (3 * c * a + 1) = 3 * d + e + 13)
  (h3 : (c + 1) * (3 * a * b + 1) = 4 * (26 - d - e) - 1)
  : d ^ 2 + e ^ 2 = 146 := 
sorry

end find_d_squared_plus_e_squared_l101_101065


namespace radius_circle_D_eq_five_l101_101444

-- Definitions for circles with given radii and tangency conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

noncomputable def circle_C : Circle := ⟨(0, 0), 5⟩
noncomputable def circle_D (rD : ℝ) : Circle := ⟨(4 * rD, 0), 4 * rD⟩
noncomputable def circle_E (rE : ℝ) : Circle := ⟨(5 - rE, rE * 5), rE⟩

-- Prove that the radius of circle D is 5
theorem radius_circle_D_eq_five (rE : ℝ) (rD : ℝ) : circle_D rE = circle_C → rD = 5 := by
  sorry

end radius_circle_D_eq_five_l101_101444


namespace polynomial_roots_power_sum_l101_101507

theorem polynomial_roots_power_sum {a b c : ℝ}
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 6)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 21 :=
by
  sorry

end polynomial_roots_power_sum_l101_101507


namespace simplify_4sqrt2_minus_sqrt2_l101_101058

/-- Prove that 4 * sqrt 2 - sqrt 2 = 3 * sqrt 2 given standard mathematical rules -/
theorem simplify_4sqrt2_minus_sqrt2 : 4 * Real.sqrt 2 - Real.sqrt 2 = 3 * Real.sqrt 2 :=
sorry

end simplify_4sqrt2_minus_sqrt2_l101_101058


namespace polynomial_solution_exists_l101_101211

open Real

theorem polynomial_solution_exists
    (P : ℝ → ℝ → ℝ)
    (hP : ∃ (f : ℝ → ℝ), ∀ x y : ℝ, P x y = f (x + y) - f x - f y) :
  ∃ (q : ℝ → ℝ), ∀ x y : ℝ, P x y = q (x + y) - q x - q y := sorry

end polynomial_solution_exists_l101_101211


namespace find_smallest_number_l101_101138

theorem find_smallest_number (x y n a : ℕ) (h1 : x + y = 2014) (h2 : 3 * n = y + 6) (h3 : x = 100 * n + a) (ha : a < 100) : min x y = 51 :=
sorry

end find_smallest_number_l101_101138


namespace sum_proper_divisors_81_l101_101249

theorem sum_proper_divisors_81 : (3^0 + 3^1 + 3^2 + 3^3) = 40 :=
by
  sorry

end sum_proper_divisors_81_l101_101249


namespace cost_of_kid_ticket_l101_101662

theorem cost_of_kid_ticket (total_people kids adults : ℕ) 
  (adult_ticket_cost kid_ticket_cost : ℕ) 
  (total_sales : ℕ) 
  (h_people : total_people = kids + adults)
  (h_adult_cost : adult_ticket_cost = 28)
  (h_kids : kids = 203)
  (h_total_sales : total_sales = 3864)
  (h_calculate_sales : adults * adult_ticket_cost + kids * kid_ticket_cost = total_sales)
  : kid_ticket_cost = 12 :=
by
  sorry -- Proof will be filled in

end cost_of_kid_ticket_l101_101662


namespace James_total_tabs_l101_101261

theorem James_total_tabs (browsers windows tabs additional_tabs : ℕ) 
  (h_browsers : browsers = 4)
  (h_windows : windows = 5)
  (h_tabs : tabs = 12)
  (h_additional_tabs : additional_tabs = 3) : 
  browsers * (windows * (tabs + additional_tabs)) = 300 := by
  -- Proof goes here
  sorry

end James_total_tabs_l101_101261


namespace solve_equation_a_solve_equation_b_l101_101595

-- Problem a
theorem solve_equation_a (a b x : ℝ) (h₀ : x ≠ a) (h₁ : x ≠ b) (h₂ : a + b ≠ 0) (h₃ : a ≠ 0) (h₄ : b ≠ 0) (h₅ : a ≠ b):
  (x + a) / (x - a) + (x + b) / (x - b) = 2 ↔ x = (2 * a * b) / (a + b) :=
by
  sorry

-- Problem b
theorem solve_equation_b (a b c d x : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : x ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) (h₅ : ab + c ≠ 0):
  c * (d / (a * b) - (a * b) / x) + d = c^2 / x ↔ x = (a * b * c) / d :=
by
  sorry

end solve_equation_a_solve_equation_b_l101_101595


namespace hyperbola_k_range_l101_101557

theorem hyperbola_k_range (k : ℝ) : 
  (∀ x y : ℝ, (x^2 / (k + 2) - y^2 / (5 - k) = 1)) → (-2 < k ∧ k < 5) :=
by
  sorry

end hyperbola_k_range_l101_101557


namespace geese_flock_size_l101_101767

theorem geese_flock_size : 
  ∃ x : ℕ, x + x + (x / 2) + (x / 4) + 1 = 100 ∧ x = 36 := 
by
  sorry

end geese_flock_size_l101_101767


namespace equal_play_time_for_students_l101_101415

theorem equal_play_time_for_students 
  (total_students : ℕ) 
  (start_time end_time : ℕ) 
  (tables : ℕ) 
  (playing_students refereeing_students : ℕ) 
  (time_played : ℕ) :
  total_students = 6 →
  start_time = 8 * 60 →
  end_time = 11 * 60 + 30 →
  tables = 2 →
  playing_students = 4 →
  refereeing_students = 2 →
  time_played = (end_time - start_time) * tables / (total_students / refereeing_students) →
  time_played = 140 :=
by
  sorry

end equal_play_time_for_students_l101_101415


namespace estevan_initial_blankets_l101_101397

theorem estevan_initial_blankets (B : ℕ) 
  (polka_dot_initial : ℕ) 
  (polka_dot_total : ℕ) 
  (h1 : (1 / 3 : ℚ) * B = polka_dot_initial) 
  (h2 : polka_dot_initial + 2 = polka_dot_total) 
  (h3 : polka_dot_total = 10) : 
  B = 24 := 
by 
  sorry

end estevan_initial_blankets_l101_101397


namespace find_number_l101_101571

theorem find_number (N : ℝ) (h : 0.60 * N = 0.50 * 720) : N = 600 :=
sorry

end find_number_l101_101571


namespace loan_percentage_correct_l101_101005

-- Define the parameters and conditions of the problem
def house_initial_value : ℕ := 100000
def house_increase_percentage : ℝ := 0.25
def new_house_cost : ℕ := 500000
def loan_percentage : ℝ := 75.0

-- Define the theorem we want to prove
theorem loan_percentage_correct :
  let increase_value := house_initial_value * house_increase_percentage
  let sale_price := house_initial_value + increase_value
  let loan_amount := new_house_cost - sale_price
  let loan_percentage_computed := (loan_amount / new_house_cost) * 100
  loan_percentage_computed = loan_percentage :=
by
  -- Proof placeholder
  sorry

end loan_percentage_correct_l101_101005


namespace decreasing_geometric_sums_implications_l101_101141

variable (X : Type)
variable (a1 q : ℝ)
variable (S : ℕ → ℝ)

def is_geometric_sequence (a : ℕ → ℝ) :=
∀ n : ℕ, a (n + 1) = a1 * q^n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
S 0 = a 0 ∧ ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

def is_decreasing_sequence (S : ℕ → ℝ) :=
∀ n : ℕ, S (n + 1) < S n

theorem decreasing_geometric_sums_implications (a1 q : ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S (n + 1) < S n) → a1 < 0 ∧ q > 0 := 
by 
  sorry

end decreasing_geometric_sums_implications_l101_101141


namespace incorrect_proposition_statement_l101_101086

theorem incorrect_proposition_statement (p q : Prop) : (p ∨ q) → ¬ (p ∧ q) := 
sorry

end incorrect_proposition_statement_l101_101086


namespace solution_set_of_inequality_l101_101695

theorem solution_set_of_inequality (a : ℝ) (h : 0 < a) :
  {x : ℝ | x ^ 2 - 4 * a * x - 5 * a ^ 2 < 0} = {x : ℝ | -a < x ∧ x < 5 * a} :=
sorry

end solution_set_of_inequality_l101_101695


namespace population_increase_duration_l101_101661

noncomputable def birth_rate := 6 / 2 -- people every 2 seconds = 3 people per second
noncomputable def death_rate := 2 / 2 -- people every 2 seconds = 1 person per second
noncomputable def net_increase_per_second := (birth_rate - death_rate) -- net increase per second

def total_net_increase := 172800

theorem population_increase_duration :
  (total_net_increase / net_increase_per_second) / 3600 = 24 :=
by
  sorry

end population_increase_duration_l101_101661


namespace total_players_l101_101213

theorem total_players 
  (cricket_players : ℕ) (hockey_players : ℕ)
  (football_players : ℕ) (softball_players : ℕ)
  (h_cricket : cricket_players = 12)
  (h_hockey : hockey_players = 17)
  (h_football : football_players = 11)
  (h_softball : softball_players = 10)
  : cricket_players + hockey_players + football_players + softball_players = 50 :=
by sorry

end total_players_l101_101213


namespace least_value_of_x_l101_101880

theorem least_value_of_x (x p : ℕ) (h1 : (x / (11 * p)) = 3) (h2 : x > 0) (h3 : Nat.Prime p) : x = 66 := by
  sorry

end least_value_of_x_l101_101880


namespace parabola_translation_l101_101302

theorem parabola_translation :
  (∀ x, y = x^2) →
  (∀ x, y = (x + 1)^2 - 2) :=
by
  sorry

end parabola_translation_l101_101302


namespace negation_of_existential_statement_l101_101337

variable (A : Set ℝ)

theorem negation_of_existential_statement :
  ¬(∃ x ∈ A, x^2 - 2 * x - 3 > 0) ↔ ∀ x ∈ A, x^2 - 2 * x - 3 ≤ 0 := by
  sorry

end negation_of_existential_statement_l101_101337


namespace digit_problem_l101_101811

theorem digit_problem (A B C D E F : ℕ) (hABC : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ D ≠ E ∧ D ≠ F ∧ E ≠ F) 
    (h1 : 100 * A + 10 * B + C = D * 100000 + A * 10000 + E * 1000 + C * 100 + F * 10 + B)
    (h2 : 100 * C + 10 * B + A = E * 100000 + D * 10000 + C * 1000 + A * 100 + B * 10 + F) : 
    A = 3 ∧ B = 6 ∧ C = 4 ∧ D = 1 ∧ E = 2 ∧ F = 9 := 
sorry

end digit_problem_l101_101811


namespace percent_equivalence_l101_101135

variable (x : ℝ)
axiom condition : 0.30 * 0.15 * x = 18

theorem percent_equivalence :
  0.15 * 0.30 * x = 18 := sorry

end percent_equivalence_l101_101135


namespace minimum_value_expression_l101_101081

open Real

theorem minimum_value_expression (α β : ℝ) :
  ∃ x y : ℝ, x = 3 * cos α + 4 * sin β ∧ y = 3 * sin α + 4 * cos β ∧
    ((x - 7) ^ 2 + (y - 12) ^ 2) = 242 - 14 * sqrt 193 :=
sorry

end minimum_value_expression_l101_101081


namespace sum_of_squares_of_roots_l101_101459

theorem sum_of_squares_of_roots :
  (∀ y : ℝ, y ^ 3 - 8 * y ^ 2 + 9 * y + 2 = 0 → y ≥ 0) →
  let s : ℝ := 8
  let p : ℝ := 9
  let q : ℝ := -2
  (s ^ 2 - 2 * p = 46) :=
by
  -- Placeholders for definitions extracted from the conditions
  -- and additional necessary let-bindings from Vieta's formulas
  intro h
  sorry

end sum_of_squares_of_roots_l101_101459


namespace evaluate_custom_operation_l101_101715

def custom_operation (A B : ℕ) : ℕ :=
  (A + 2 * B) * (A - B)

theorem evaluate_custom_operation : custom_operation 7 5 = 34 :=
by
  sorry

end evaluate_custom_operation_l101_101715


namespace remainder_of_sum_l101_101744

theorem remainder_of_sum (x y z : ℕ) (h1 : x % 15 = 6) (h2 : y % 15 = 9) (h3 : z % 15 = 3) : 
  (x + y + z) % 15 = 3 := 
  sorry

end remainder_of_sum_l101_101744


namespace largest_num_blocks_l101_101004

-- Define the volume of the box
def volume_box (l₁ w₁ h₁ : ℕ) : ℕ :=
  l₁ * w₁ * h₁

-- Define the volume of the block
def volume_block (l₂ w₂ h₂ : ℕ) : ℕ :=
  l₂ * w₂ * h₂

-- Define the function to calculate maximum blocks
def max_blocks (V_box V_block : ℕ) : ℕ :=
  V_box / V_block

theorem largest_num_blocks :
  max_blocks (volume_box 5 4 6) (volume_block 3 3 2) = 6 :=
by
  sorry

end largest_num_blocks_l101_101004


namespace pizza_eaten_after_six_trips_l101_101535

theorem pizza_eaten_after_six_trips
  (initial_fraction: ℚ)
  (next_fraction : ℚ -> ℚ)
  (S: ℚ)
  (H0: initial_fraction = 1 / 4)
  (H1: ∀ (n: ℕ), next_fraction n = 1 / 2 ^ (n + 2))
  (H2: S = initial_fraction + (next_fraction 1) + (next_fraction 2) + (next_fraction 3) + (next_fraction 4) + (next_fraction 5)):
  S = 125 / 128 :=
by
  sorry

end pizza_eaten_after_six_trips_l101_101535


namespace pens_in_each_pack_l101_101026

-- Given the conditions
def Kendra_packs : ℕ := 4
def Tony_packs : ℕ := 2
def pens_kept_each : ℕ := 2
def friends : ℕ := 14

-- Theorem statement
theorem pens_in_each_pack : ∃ (P : ℕ), Kendra_packs * P + Tony_packs * P - pens_kept_each * 2 - friends = 0 ∧ P = 3 := by
  sorry

end pens_in_each_pack_l101_101026


namespace emir_needs_more_money_l101_101673

noncomputable def dictionary_cost : ℝ := 5.50
noncomputable def dinosaur_book_cost : ℝ := 11.25
noncomputable def childrens_cookbook_cost : ℝ := 5.75
noncomputable def science_experiment_kit_cost : ℝ := 8.50
noncomputable def colored_pencils_cost : ℝ := 3.60
noncomputable def world_map_poster_cost : ℝ := 2.40
noncomputable def puzzle_book_cost : ℝ := 4.65
noncomputable def sketchpad_cost : ℝ := 6.20

noncomputable def sales_tax_rate : ℝ := 0.07
noncomputable def dinosaur_discount_rate : ℝ := 0.10
noncomputable def saved_amount : ℝ := 28.30

noncomputable def total_cost_before_tax : ℝ :=
  dictionary_cost +
  (dinosaur_book_cost - dinosaur_discount_rate * dinosaur_book_cost) +
  childrens_cookbook_cost +
  science_experiment_kit_cost +
  colored_pencils_cost +
  world_map_poster_cost +
  puzzle_book_cost +
  sketchpad_cost

noncomputable def total_sales_tax : ℝ := sales_tax_rate * total_cost_before_tax

noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + total_sales_tax

noncomputable def additional_amount_needed : ℝ := total_cost_after_tax - saved_amount

theorem emir_needs_more_money : additional_amount_needed = 21.81 := by
  sorry

end emir_needs_more_money_l101_101673


namespace parallelogram_area_l101_101345

variable (d : ℕ) (h : ℕ)

theorem parallelogram_area (h_d : d = 30) (h_h : h = 20) : 
  ∃ a : ℕ, a = 600 := 
by
  sorry

end parallelogram_area_l101_101345


namespace exists_three_cycle_l101_101306

variable {α : Type}

def tournament (P : α → α → Prop) : Prop :=
  (∃ (participants : List α), participants.length ≥ 3) ∧
  (∀ x y, x ≠ y → P x y ∨ P y x) ∧
  (∀ x, ∃ y, P x y)

theorem exists_three_cycle {α : Type} (P : α → α → Prop) :
  tournament P → ∃ A B C, P A B ∧ P B C ∧ P C A :=
by
  sorry

end exists_three_cycle_l101_101306


namespace chinese_medicine_excess_purchased_l101_101970

-- Define the conditions of the problem

def total_plan : ℕ := 1500

def first_half_percentage : ℝ := 0.55
def second_half_percentage : ℝ := 0.65

-- State the theorem to prove the amount purchased in excess
theorem chinese_medicine_excess_purchased :
    first_half_percentage * total_plan + second_half_percentage * total_plan - total_plan = 300 :=
by 
  sorry

end chinese_medicine_excess_purchased_l101_101970


namespace three_digit_numbers_square_ends_in_1001_l101_101275

theorem three_digit_numbers_square_ends_in_1001 (n : ℕ) :
  100 ≤ n ∧ n < 1000 ∧ n^2 % 10000 = 1001 → n = 501 ∨ n = 749 :=
by
  intro h
  sorry

end three_digit_numbers_square_ends_in_1001_l101_101275


namespace union_A_B_l101_101062

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def C : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_A_B : A ∪ B = C := 
by sorry

end union_A_B_l101_101062


namespace problem_1_problem_2_l101_101610

-- Define f as an odd function on ℝ 
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the main property given in the problem
def property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, (a + b) ≠ 0 → (f a + f b) / (a + b) > 0

-- Problem 1: Prove that if a > b then f(a) > f(b)
theorem problem_1 (f : ℝ → ℝ) (h_odd : odd_function f) (h_property : property f) :
  ∀ a b : ℝ, a > b → f a > f b := sorry

-- Problem 2: Prove that given f(9^x - 2 * 3^x) + f(2 * 9^x - k) > 0 for any x in [0, +∞), the range of k is k < 1
theorem problem_2 (f : ℝ → ℝ) (h_odd : odd_function f) (h_property : property f) :
  (∀ x : ℝ, 0 ≤ x → f (9 ^ x - 2 * 3 ^ x) + f (2 * 9 ^ x - k) > 0) → k < 1 := sorry

end problem_1_problem_2_l101_101610


namespace find_a_plus_b_l101_101523

open Complex

noncomputable def problem_statement (a b : ℝ) : Prop :=
  (∃ (r1 r2 r3 : ℂ),
     r1 = 1 + I * Real.sqrt 3 ∧
     r2 = 1 - I * Real.sqrt 3 ∧
     r3 = -2 ∧
     (r1 + r2 + r3 = 0) ∧
     (r1 * r2 * r3 = -b) ∧
     (r1 * r2 + r2 * r3 + r3 * r1 = -a))

theorem find_a_plus_b (a b : ℝ) (h : problem_statement a b) : a + b = 8 :=
sorry

end find_a_plus_b_l101_101523


namespace find_x_from_roots_l101_101010

variable (x m : ℕ)

theorem find_x_from_roots (h1 : (m + 3)^2 = x) (h2 : (2 * m - 15)^2 = x) : x = 49 := by
  sorry

end find_x_from_roots_l101_101010


namespace tom_final_amount_l101_101063

-- Conditions and definitions from the problem
def initial_amount : ℝ := 74
def spent_percentage : ℝ := 0.15
def earnings : ℝ := 86
def share_percentage : ℝ := 0.60

-- Lean proof statement
theorem tom_final_amount :
  (initial_amount - (spent_percentage * initial_amount)) + (share_percentage * earnings) = 114.5 :=
by
  sorry

end tom_final_amount_l101_101063


namespace number_of_years_borrowed_l101_101165

theorem number_of_years_borrowed (n : ℕ)
  (H1 : ∃ (p : ℕ), 5000 = p ∧ 4 = 4 ∧ n * 200 = 150)
  (H2 : ∃ (q : ℕ), 5000 = q ∧ 7 = 7 ∧ n * 350 = 150)
  : n = 1 :=
by
  sorry

end number_of_years_borrowed_l101_101165


namespace find_number_l101_101078

theorem find_number (x : ℝ) : 
  (x + 72 = (2 * x) / (2 / 3)) → x = 36 :=
by
  intro h
  sorry

end find_number_l101_101078


namespace school_student_count_l101_101134

-- Definition of the conditions
def students_in_school (n : ℕ) : Prop :=
  200 ≤ n ∧ n ≤ 300 ∧
  n % 6 = 1 ∧
  n % 8 = 2 ∧
  n % 9 = 3

-- The main proof statement
theorem school_student_count : ∃ n, students_in_school n ∧ n = 265 :=
by
  sorry  -- Proof would go here

end school_student_count_l101_101134


namespace number_tower_proof_l101_101017

theorem number_tower_proof : 123456 * 9 + 7 = 1111111 := 
  sorry

end number_tower_proof_l101_101017


namespace standard_eq_of_ellipse_value_of_k_l101_101712

-- Definitions and conditions
def is_ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0

def eccentricity (a b : ℝ) (e : ℝ) : Prop :=
  e = (Real.sqrt 2) / 2 ∧ a^2 = b^2 + (a * e)^2

def minor_axis_length (b : ℝ) : Prop :=
  2 * b = 2

def is_tangency (k m : ℝ) : Prop := 
  m^2 = 1 + k^2

def line_intersect_ellipse (k m : ℝ) : Prop :=
  (4 * k * m)^2 - 4 * (1 + 2 * k^2) * (2 * m^2 - 2) > 0

def dot_product_condition (k m : ℝ) : Prop :=
  let x1 := -(4 * k * m) / (1 + 2 * k^2)
  let x2 := (2 * m^2 - 2) / (1 + 2 * k^2)
  let y1 := k * x1 + m
  let y2 := k * x2 + m
  x1 * x2 + y1 * y2 = 2 / 3

-- To prove the standard equation of the ellipse
theorem standard_eq_of_ellipse {a b : ℝ} (h_ellipse : is_ellipse a b)
  (h_eccentricity : eccentricity a b ((Real.sqrt 2) / 2)) 
  (h_minor_axis : minor_axis_length b) : 
  ∃ a, a = Real.sqrt 2 ∧ b = 1 ∧ (∀ x y, (x^2 / 2 + y^2 = 1)) := 
sorry

-- To prove the value of k
theorem value_of_k {k m : ℝ} (h_tangency : is_tangency k m) 
  (h_intersect : line_intersect_ellipse k m)
  (h_dot_product : dot_product_condition k m) :
  k = 1 ∨ k = -1 :=
sorry

end standard_eq_of_ellipse_value_of_k_l101_101712


namespace sum_squares_of_six_consecutive_even_eq_1420_l101_101685

theorem sum_squares_of_six_consecutive_even_eq_1420 
  (n : ℤ) 
  (h : n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) = 90) :
  n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 + (n + 8)^2 + (n + 10)^2 = 1420 :=
by
  sorry

end sum_squares_of_six_consecutive_even_eq_1420_l101_101685


namespace jeremie_friends_l101_101324

-- Define the costs as constants.
def ticket_cost : ℕ := 18
def snack_cost : ℕ := 5
def total_cost : ℕ := 92
def per_person_cost : ℕ := ticket_cost + snack_cost

-- Define the number of friends Jeremie is going with (to be solved/proven).
def number_of_friends (total_cost : ℕ) (per_person_cost : ℕ) : ℕ :=
  let total_people := total_cost / per_person_cost
  total_people - 1

-- The statement that we want to prove.
theorem jeremie_friends : number_of_friends total_cost per_person_cost = 3 := by
  sorry

end jeremie_friends_l101_101324


namespace rider_distance_traveled_l101_101694

noncomputable def caravan_speed := 1  -- km/h
noncomputable def rider_speed := 1 + Real.sqrt 2  -- km/h

theorem rider_distance_traveled : 
  (1 / (rider_speed - 1) + 1 / (rider_speed + 1)) = 1 :=
by
  sorry

end rider_distance_traveled_l101_101694


namespace blue_balls_in_box_l101_101190

theorem blue_balls_in_box (total_balls : ℕ) (p_two_blue : ℚ) (b : ℕ) 
  (h1 : total_balls = 12) (h2 : p_two_blue = 1/22) 
  (h3 : (↑b / 12) * (↑(b-1) / 11) = p_two_blue) : b = 3 :=
by {
  sorry
}

end blue_balls_in_box_l101_101190


namespace vectors_parallel_y_eq_minus_one_l101_101401

theorem vectors_parallel_y_eq_minus_one (y : ℝ) :
  let a := (1, 2)
  let b := (1, -2 * y)
  b.1 * a.2 - a.1 * b.2 = 0 → y = -1 :=
by
  intros a b h
  simp at h
  sorry

end vectors_parallel_y_eq_minus_one_l101_101401


namespace cos_double_angle_of_parallel_vectors_l101_101853

variables {α : Type*}

/-- Given vectors a and b specified by the problem, if they are parallel, then cos 2α = 7/9. -/
theorem cos_double_angle_of_parallel_vectors (α : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (1/3, Real.tan α)) 
  (hb : b = (Real.cos α, 1)) 
  (parallel : a.1 * b.2 = a.2 * b.1) : 
  Real.cos (2 * α) = 7/9 := 
by 
  sorry

end cos_double_angle_of_parallel_vectors_l101_101853


namespace mul_same_base_exp_ten_pow_1000_sq_l101_101054

theorem mul_same_base_exp (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by
  sorry

-- Given specific constants for this problem
theorem ten_pow_1000_sq : (10:ℝ)^(1000) * (10)^(1000) = (10)^(2000) := by
  exact mul_same_base_exp 10 1000 1000

end mul_same_base_exp_ten_pow_1000_sq_l101_101054


namespace symmetrical_point_l101_101179

structure Point :=
  (x : ℝ)
  (y : ℝ)

def reflect_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem symmetrical_point (M : Point) (hM : M = {x := 3, y := -4}) : reflect_x_axis M = {x := 3, y := 4} :=
  by
  sorry

end symmetrical_point_l101_101179


namespace kimberly_initial_skittles_l101_101505

theorem kimberly_initial_skittles : 
  ∀ (x : ℕ), (x + 7 = 12) → x = 5 :=
by
  sorry

end kimberly_initial_skittles_l101_101505


namespace baseball_cards_remaining_l101_101757

-- Define the number of baseball cards Mike originally had
def original_cards : ℕ := 87

-- Define the number of baseball cards Sam bought from Mike
def cards_bought : ℕ := 13

-- Prove that the remaining number of baseball cards Mike has is 74
theorem baseball_cards_remaining : original_cards - cards_bought = 74 := by
  sorry

end baseball_cards_remaining_l101_101757


namespace diameter_of_circle_is_60_l101_101287

noncomputable def diameter_of_circle (M N : ℝ) : ℝ :=
  if h : N ≠ 0 then 2 * (M / N * (1 / (2 * Real.pi))) else 0

theorem diameter_of_circle_is_60 (M N : ℝ) (h : M / N = 15) :
  diameter_of_circle M N = 60 :=
by
  sorry

end diameter_of_circle_is_60_l101_101287


namespace relationship_abc_l101_101603

noncomputable def a (x : ℝ) : ℝ := Real.log x
noncomputable def b (x : ℝ) : ℝ := Real.exp (Real.log x)
noncomputable def c (x : ℝ) : ℝ := Real.exp (Real.log (1 / x))

theorem relationship_abc (x : ℝ) (h : (1 / Real.exp 1) < x ∧ x < 1) : a x < b x ∧ b x < c x :=
by
  have ha : a x = Real.log x := rfl
  have hb : b x = Real.exp (Real.log x) := rfl
  have hc : c x = Real.exp (Real.log (1 / x)) := rfl
  sorry

end relationship_abc_l101_101603


namespace continuous_function_identity_l101_101020

theorem continuous_function_identity (f : ℝ → ℝ)
  (h_cont : Continuous f)
  (h_func_eq : ∀ x y : ℝ, 2 * f (x + y) = f x * f y)
  (h_f1 : f 1 = 10) :
  ∀ x : ℝ, f x = 2 * 5^x :=
by
  sorry

end continuous_function_identity_l101_101020


namespace Cindy_walking_speed_l101_101693

noncomputable def walking_speed (total_time : ℕ) (running_speed : ℕ) (running_distance : ℚ) (walking_distance : ℚ) : ℚ := 
  let time_to_run := running_distance / running_speed
  let walking_time := total_time - (time_to_run * 60)
  walking_distance / (walking_time / 60)

theorem Cindy_walking_speed : walking_speed 40 3 0.5 0.5 = 1 := 
  sorry

end Cindy_walking_speed_l101_101693


namespace reduced_price_correct_l101_101647

theorem reduced_price_correct (P R Q: ℝ) (h1 : R = 0.75 * P) (h2 : 900 = Q * P) (h3 : 900 = (Q + 5) * R)  :
  R = 45 := by 
  sorry

end reduced_price_correct_l101_101647


namespace stopped_time_per_hour_A_stopped_time_per_hour_B_stopped_time_per_hour_C_l101_101104

-- Definition of the speeds
def speed_excluding_stoppages_A : ℕ := 60
def speed_including_stoppages_A : ℕ := 48
def speed_excluding_stoppages_B : ℕ := 75
def speed_including_stoppages_B : ℕ := 60
def speed_excluding_stoppages_C : ℕ := 90
def speed_including_stoppages_C : ℕ := 72

-- Theorem to prove the stopped time per hour for each bus
theorem stopped_time_per_hour_A : (speed_excluding_stoppages_A - speed_including_stoppages_A) * 60 / speed_excluding_stoppages_A = 12 := sorry

theorem stopped_time_per_hour_B : (speed_excluding_stoppages_B - speed_including_stoppages_B) * 60 / speed_excluding_stoppages_B = 12 := sorry

theorem stopped_time_per_hour_C : (speed_excluding_stoppages_C - speed_including_stoppages_C) * 60 / speed_excluding_stoppages_C = 12 := sorry

end stopped_time_per_hour_A_stopped_time_per_hour_B_stopped_time_per_hour_C_l101_101104


namespace solution_set_of_inequality_l101_101123

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 2 * Real.exp (x - 1) else Real.log (x^2 - 1) / Real.log 3

theorem solution_set_of_inequality :
  {x : ℝ | f x > 2} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | Real.sqrt 10 < x} :=
by
  sorry

end solution_set_of_inequality_l101_101123


namespace sum_of_logs_l101_101974

open Real

noncomputable def log_base (b a : ℝ) : ℝ := log a / log b

theorem sum_of_logs (x y z : ℝ)
  (h1 : log_base 2 (log_base 4 (log_base 5 x)) = 0)
  (h2 : log_base 3 (log_base 5 (log_base 2 y)) = 0)
  (h3 : log_base 4 (log_base 2 (log_base 3 z)) = 0) :
  x + y + z = 666 := sorry

end sum_of_logs_l101_101974


namespace equivalent_discount_l101_101172

variable (P d1 d2 d : ℝ)

-- Given conditions:
def original_price : ℝ := 50
def first_discount_rate : ℝ := 0.25
def second_discount_rate : ℝ := 0.10
def equivalent_single_discount_rate : ℝ := 0.325

-- Final conclusion:
theorem equivalent_discount :
  let final_price_after_first_discount := (original_price * (1 - first_discount_rate))
  let final_price_after_second_discount := (final_price_after_first_discount * (1 - second_discount_rate))
  final_price_after_second_discount = (original_price * (1 - equivalent_single_discount_rate)) :=
by
  sorry

end equivalent_discount_l101_101172


namespace cricket_team_age_difference_l101_101491

theorem cricket_team_age_difference :
  ∀ (captain_age : ℕ) (keeper_age : ℕ) (team_size : ℕ) (team_average_age : ℕ) (remaining_size : ℕ),
  captain_age = 28 →
  keeper_age = captain_age + 3 →
  team_size = 11 →
  team_average_age = 25 →
  remaining_size = team_size - 2 →
  (team_size * team_average_age - (captain_age + keeper_age)) / remaining_size = 24 →
  team_average_age - (team_size * team_average_age - (captain_age + keeper_age)) / remaining_size = 1 :=
by
  intros captain_age keeper_age team_size team_average_age remaining_size h1 h2 h3 h4 h5 h6
  sorry

end cricket_team_age_difference_l101_101491


namespace inequality_positive_reals_l101_101895

open Real

variable (x y : ℝ)

theorem inequality_positive_reals (hx : 0 < x) (hy : 0 < y) : x^2 + (8 / (x * y)) + y^2 ≥ 8 :=
by
  sorry

end inequality_positive_reals_l101_101895


namespace truckToCarRatio_l101_101860

-- Conditions
def liftsCar (C : ℕ) : Prop := C = 5
def peopleNeeded (C T : ℕ) : Prop := 6 * C + 3 * T = 60

-- Theorem statement
theorem truckToCarRatio (C T : ℕ) (hc : liftsCar C) (hp : peopleNeeded C T) : T / C = 2 :=
by
  sorry

end truckToCarRatio_l101_101860


namespace solve_ff_eq_x_l101_101028

def f (x : ℝ) : ℝ := x^2 + 2 * x - 5

theorem solve_ff_eq_x :
  ∀ x : ℝ, f (f x) = x ↔ (x = ( -1 + Real.sqrt 21 ) / 2) ∨ (x = ( -1 - Real.sqrt 21 ) / 2) ∨
                          (x = ( -3 + Real.sqrt 17 ) / 2) ∨ (x = ( -3 - Real.sqrt 17 ) / 2) := 
by
  sorry

end solve_ff_eq_x_l101_101028


namespace arithmetic_sequence_common_difference_l101_101562

theorem arithmetic_sequence_common_difference (a_n : ℕ → ℤ) (h1 : a_n 1 = 13) (h4 : a_n 4 = 1) : 
  ∃ d : ℤ, d = -4 := by
  sorry

end arithmetic_sequence_common_difference_l101_101562


namespace value_of_expression_l101_101388

-- Conditions
def isOdd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def isIncreasingOn (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def hasMaxOn (f : ℝ → ℝ) (a b : ℝ) (M : ℝ) := ∃ x, a ≤ x ∧ x ≤ b ∧ f x = M
def hasMinOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) := ∃ x, a ≤ x ∧ x ≤ b ∧ f x = m

-- Proof statement
theorem value_of_expression (f : ℝ → ℝ) 
  (hf1 : isOdd f)
  (hf2 : isIncreasingOn f 3 7)
  (hf3 : hasMaxOn f 3 6 8)
  (hf4 : hasMinOn f 3 6 (-1)) :
  2 * f (-6) + f (-3) = -15 :=
sorry

end value_of_expression_l101_101388


namespace eval_expression_at_neg_one_l101_101881

variable (x : ℤ)

theorem eval_expression_at_neg_one : x = -1 → 3 * x ^ 2 + 2 * x - 1 = 0 := by
  intro h
  rw [h]
  show 3 * (-1) ^ 2 + 2 * (-1) - 1 = 0
  sorry

end eval_expression_at_neg_one_l101_101881


namespace total_milk_bottles_l101_101697

theorem total_milk_bottles (marcus_bottles : ℕ) (john_bottles : ℕ) (h1 : marcus_bottles = 25) (h2 : john_bottles = 20) : marcus_bottles + john_bottles = 45 := by
  sorry

end total_milk_bottles_l101_101697


namespace proof_problem_l101_101521

variable (balls : Finset ℕ) (blackBalls whiteBalls : Finset ℕ)
variable (drawnBalls : Finset ℕ)

/-- There are 6 black balls numbered 1 to 6. -/
def initialBlackBalls : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- There are 4 white balls numbered 7 to 10. -/
def initialWhiteBalls : Finset ℕ := {7, 8, 9, 10}

/-- The total balls (black + white). -/
def totalBalls : Finset ℕ := initialBlackBalls ∪ initialWhiteBalls

/-- The hypergeometric distribution condition for black balls. -/
def hypergeometricBlack : Prop :=
  true  -- placeholder: black balls follow hypergeometric distribution

/-- The probability of drawing 2 white balls is not 1/14. -/
def probDraw2White : Prop :=
  (3 / 7) ≠ (1 / 14)

/-- The probability of the maximum total score (8 points) is 1/14. -/
def probMaxScore : Prop :=
  (15 / 210) = (1 / 14)

/-- Main theorem combining the above conditions for the problem. -/
theorem proof_problem : hypergeometricBlack ∧ probMaxScore :=
by
  unfold hypergeometricBlack
  unfold probMaxScore
  sorry

end proof_problem_l101_101521


namespace largest_prime_factor_sum_of_four_digit_numbers_l101_101829

theorem largest_prime_factor_sum_of_four_digit_numbers 
  (a b c d : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) 
  (h3 : 1 ≤ b) (h4 : b ≤ 9) 
  (h5 : 1 ≤ c) (h6 : c ≤ 9) 
  (h7 : 1 ≤ d) (h8 : d ≤ 9) 
  (h_diff : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  : Nat.gcd 6666 (a + b + c + d) = 101 :=
sorry

end largest_prime_factor_sum_of_four_digit_numbers_l101_101829


namespace find_a4_l101_101959

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

theorem find_a4 (h1 : arithmetic_sequence a) (h2 : a 2 + a 6 = 2) : a 4 = 1 :=
by
  sorry

end find_a4_l101_101959


namespace pasta_cost_is_one_l101_101512

-- Define the conditions
def pasta_cost (p : ℝ) : ℝ := p -- The cost of the pasta per box
def sauce_cost : ℝ := 2.00 -- The cost of the sauce
def meatballs_cost : ℝ := 5.00 -- The cost of the meatballs
def servings : ℕ := 8 -- The number of servings
def cost_per_serving : ℝ := 1.00 -- The cost per serving

-- Calculate the total meal cost
def total_meal_cost : ℝ := servings * cost_per_serving

-- Calculate the combined cost of sauce and meatballs
def combined_cost_of_sauce_and_meatballs : ℝ := sauce_cost + meatballs_cost

-- Calculate the cost of the pasta
def pasta_cost_calculation : ℝ := total_meal_cost - combined_cost_of_sauce_and_meatballs

-- The theorem stating that the pasta cost should be $1
theorem pasta_cost_is_one (p : ℝ) (h : pasta_cost_calculation = p) : p = 1 := by
  sorry

end pasta_cost_is_one_l101_101512


namespace wayne_needs_30_more_blocks_l101_101263

def initial_blocks : ℕ := 9
def additional_blocks : ℕ := 6
def total_blocks : ℕ := initial_blocks + additional_blocks
def triple_total : ℕ := 3 * total_blocks

theorem wayne_needs_30_more_blocks :
  triple_total - total_blocks = 30 := by
  sorry

end wayne_needs_30_more_blocks_l101_101263


namespace right_triangle_perimeter_l101_101720

theorem right_triangle_perimeter
  (a b : ℝ)
  (h_area : 0.5 * 30 * b = 150)
  (h_leg : a = 30) :
  a + b + Real.sqrt (a^2 + b^2) = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l101_101720


namespace negation_of_p_l101_101431
open Classical

variable (n : ℕ)

def p : Prop := ∀ n : ℕ, n^2 < 2^n

theorem negation_of_p : ¬ p ↔ ∃ n₀ : ℕ, n₀^2 ≥ 2^n₀ := 
by
  sorry

end negation_of_p_l101_101431


namespace arithmetic_sequence_third_term_l101_101732

theorem arithmetic_sequence_third_term {a d : ℝ} (h : 2 * a + 4 * d = 10) : a + 2 * d = 5 :=
sorry

end arithmetic_sequence_third_term_l101_101732


namespace parabola_area_l101_101971

theorem parabola_area (m p : ℝ) (h1 : p > 0) (h2 : (1:ℝ)^2 = 2 * p * m)
    (h3 : (1/2) * (m + p / 2) = 1/2) : p = 1 :=
  by
    sorry

end parabola_area_l101_101971


namespace required_circle_properties_l101_101861

-- Define the two given circles' equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center of the required circle lies
def line (x y : ℝ) : Prop :=
  x - y - 4 = 0

-- The equation of the required circle
def required_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - x + 7*y - 32 = 0

-- Prove that the required circle satisfies the conditions
theorem required_circle_properties (x y : ℝ) (hx : required_circle x y) :
  (∃ x y, circle1 x y ∧ circle2 x y ∧ required_circle x y) ∧
  (∃ x y, required_circle x y ∧ line x y) :=
by
  sorry

end required_circle_properties_l101_101861


namespace find_sum_of_p_q_r_s_l101_101402

theorem find_sum_of_p_q_r_s 
    (p q r s : ℝ)
    (h1 : r + s = 12 * p)
    (h2 : r * s = -13 * q)
    (h3 : p + q = 12 * r)
    (h4 : p * q = -13 * s)
    (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
    p + q + r + s = 2028 := 
sorry

end find_sum_of_p_q_r_s_l101_101402


namespace tan_alpha_in_second_quadrant_l101_101239

theorem tan_alpha_in_second_quadrant (α : ℝ) (h₁ : π / 2 < α ∧ α < π) (hsin : Real.sin α = 5 / 13) :
    Real.tan α = -5 / 12 :=
sorry

end tan_alpha_in_second_quadrant_l101_101239


namespace triangle_angle_contradiction_l101_101072

theorem triangle_angle_contradiction (A B C : ℝ) (h_sum : A + B + C = 180) (h_lt_60 : A < 60 ∧ B < 60 ∧ C < 60) : false := 
sorry

end triangle_angle_contradiction_l101_101072


namespace max_value_y_l101_101795

theorem max_value_y (x : ℝ) (h : x < 5 / 4) : 
  (4 * x - 2 + 1 / (4 * x - 5)) ≤ 1 :=
sorry

end max_value_y_l101_101795


namespace find_c_l101_101616

theorem find_c (a b c : ℤ) (h1 : c ≥ 0) (h2 : ¬∃ m : ℤ, 2 * a * b = m^2)
  (h3 : ∀ n : ℕ, n > 0 → (a^n + (2 : ℤ)^n) ∣ (b^n + c)) :
  c = 0 ∨ c = 1 :=
by
  sorry

end find_c_l101_101616


namespace contrapositive_correct_l101_101623

-- Define the main condition: If a ≥ 1/2, then ∀ x ≥ 0, f(x) ≥ 0
def main_condition (a : ℝ) (f : ℝ → ℝ) : Prop :=
  a ≥ 1/2 → ∀ x : ℝ, x ≥ 0 → f x ≥ 0

-- Define the contrapositive statement: If ∃ x ≥ 0 such that f(x) < 0, then a < 1/2
def contrapositive (a : ℝ) (f : ℝ → ℝ) : Prop :=
  (∃ x : ℝ, x ≥ 0 ∧ f x < 0) → a < 1/2

-- Theorem to prove that the contrapositive statement is correct
theorem contrapositive_correct (a : ℝ) (f : ℝ → ℝ) :
  main_condition a f ↔ contrapositive a f :=
by
  sorry

end contrapositive_correct_l101_101623


namespace polygon_sides_l101_101294

theorem polygon_sides (n : ℕ) (h_sum : 180 * (n - 2) = 1980) : n = 13 :=
by {
  sorry
}

end polygon_sides_l101_101294


namespace minimize_area_of_quadrilateral_l101_101164

noncomputable def minimize_quad_area (AB BC CD DA A1 B1 C1 D1 : ℝ) (k : ℝ) : Prop :=
  -- Conditions
  AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0 ∧ k > 0 ∧
  A1 = k * AB ∧ B1 = k * BC ∧ C1 = k * CD ∧ D1 = k * DA →
  -- Conclusion
  k = 1 / 2

-- Statement without proof
theorem minimize_area_of_quadrilateral (AB BC CD DA : ℝ) : ∃ k : ℝ, minimize_quad_area AB BC CD DA (k * AB) (k * BC) (k * CD) (k * DA) k :=
sorry

end minimize_area_of_quadrilateral_l101_101164


namespace problem1_problem2_l101_101463

-- Problem 1: Evaluating an integer arithmetic expression
theorem problem1 : (1 * (-8)) - (-6) + (-3) = -5 := 
by
  sorry

-- Problem 2: Evaluating a mixed arithmetic expression with rational numbers and decimals
theorem problem2 : (5 / 13) - 3.7 + (8 / 13) - (-1.7) = -1 := 
by
  sorry

end problem1_problem2_l101_101463


namespace conjecture_l101_101663

noncomputable def f (x : ℝ) : ℝ :=
  1 / (3^x + Real.sqrt 3)

theorem conjecture (x : ℝ) : f (-x) + f (1 + x) = Real.sqrt 3 / 3 := sorry

end conjecture_l101_101663


namespace sum_tens_units_11_pow_2010_l101_101834

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_tens_units_digits (n : ℕ) : ℕ :=
  tens_digit n + units_digit n

theorem sum_tens_units_11_pow_2010 :
  sum_tens_units_digits (11 ^ 2010) = 1 :=
sorry

end sum_tens_units_11_pow_2010_l101_101834


namespace cows_now_l101_101833

-- Defining all conditions
def initial_cows : ℕ := 39
def cows_died : ℕ := 25
def cows_sold : ℕ := 6
def cows_increase : ℕ := 24
def cows_bought : ℕ := 43
def cows_gift : ℕ := 8

-- Lean statement for the equivalent proof problem
theorem cows_now :
  let cows_left := initial_cows - cows_died
  let cows_after_selling := cows_left - cows_sold
  let cows_this_year_increased := cows_after_selling + cows_increase
  let cows_with_purchase := cows_this_year_increased + cows_bought
  let total_cows := cows_with_purchase + cows_gift
  total_cows = 83 :=
by
  sorry

end cows_now_l101_101833


namespace inequality_abc_l101_101651

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 :=
by
  sorry

end inequality_abc_l101_101651


namespace radio_show_songs_duration_l101_101943

-- Definitions of the conditions
def hours_per_day := 3
def minutes_per_hour := 60
def talking_segments := 3
def talking_segment_duration := 10
def ad_breaks := 5
def ad_break_duration := 5

-- The main statement translating the conditions and questions to Lean
theorem radio_show_songs_duration :
  (hours_per_day * minutes_per_hour) - (talking_segments * talking_segment_duration + ad_breaks * ad_break_duration) = 125 := by
  sorry

end radio_show_songs_duration_l101_101943


namespace smallest_integer_y_solution_l101_101769

theorem smallest_integer_y_solution :
  ∃ y : ℤ, (∀ z : ℤ, (z / 4 + 3 / 7 > 9 / 4) → (z ≥ y)) ∧ (y = 8) := 
by
  sorry

end smallest_integer_y_solution_l101_101769


namespace fraction_simplification_l101_101999

theorem fraction_simplification : (4 * 5) / 10 = 2 := 
by 
  sorry

end fraction_simplification_l101_101999


namespace namjoon_used_pencils_l101_101378

variable (taehyungUsed : ℕ) (namjoonUsed : ℕ)

/-- 
Statement:
Taehyung and Namjoon each initially have 10 pencils.
Taehyung gives 3 of his remaining pencils to Namjoon.
After this, Taehyung ends up with 6 pencils and Namjoon ends up with 6 pencils.
We need to prove that Namjoon used 7 pencils.
-/
theorem namjoon_used_pencils (H1 : 10 - taehyungUsed = 9 - 3)
  (H2 : 13 - namjoonUsed = 6) : namjoonUsed = 7 :=
sorry

end namjoon_used_pencils_l101_101378


namespace sin_transformation_identity_l101_101240

theorem sin_transformation_identity 
  (θ : ℝ) 
  (h : Real.cos (π / 12 - θ) = 1 / 3) : 
  Real.sin (2 * θ + π / 3) = -7 / 9 := 
by 
  sorry

end sin_transformation_identity_l101_101240


namespace mixture_correct_l101_101156

def water_amount : ℚ := (3/5) * 20
def vinegar_amount : ℚ := (5/6) * 18
def mixture_amount : ℚ := water_amount + vinegar_amount

theorem mixture_correct : mixture_amount = 27 := 
by
  -- Here goes the proof steps
  sorry

end mixture_correct_l101_101156


namespace mikes_lower_rate_l101_101352

theorem mikes_lower_rate (x : ℕ) (high_rate : ℕ) (total_paid : ℕ) (lower_payments : ℕ) (higher_payments : ℕ)
  (h1 : high_rate = 310)
  (h2 : total_paid = 3615)
  (h3 : lower_payments = 5)
  (h4 : higher_payments = 7)
  (h5 : lower_payments * x + higher_payments * high_rate = total_paid) :
  x = 289 :=
sorry

end mikes_lower_rate_l101_101352


namespace men_with_6_boys_work_l101_101851

theorem men_with_6_boys_work (m b : ℚ) (x : ℕ) :
  2 * m + 4 * b = 1 / 4 →
  x * m + 6 * b = 1 / 3 →
  2 * b = 5 * m →
  x = 1 :=
by
  intros h1 h2 h3
  sorry

end men_with_6_boys_work_l101_101851


namespace starting_player_can_ensure_integer_roots_l101_101113

theorem starting_player_can_ensure_integer_roots :
  ∃ (a b c : ℤ), ∀ (x : ℤ), (x^3 + a * x^2 + b * x + c = 0) →
  (∃ r1 r2 r3 : ℤ, x = r1 ∨ x = r2 ∨ x = r3) :=
sorry

end starting_player_can_ensure_integer_roots_l101_101113


namespace original_number_of_people_l101_101913

theorem original_number_of_people (x : ℕ) (h1 : 3 ∣ x) (h2 : 6 ∣ x) (h3 : (x / 2) = 18) : x = 36 :=
by
  sorry

end original_number_of_people_l101_101913


namespace initial_cake_pieces_l101_101012

-- Define the initial number of cake pieces
variable (X : ℝ)

-- Define the conditions as assumptions
def cake_conditions (X : ℝ) : Prop :=
  0.60 * X + 3 * 32 = X 

theorem initial_cake_pieces (X : ℝ) (h : cake_conditions X) : X = 240 := sorry

end initial_cake_pieces_l101_101012


namespace correct_product_l101_101763

theorem correct_product (a b c : ℕ) (ha : 10 * c + 1 = a) (hb : 10 * c + 7 = a) 
(hl : (10 * c + 1) * b = 255) (hw : (10 * c + 7 + 6) * b = 335) : 
  a * b = 285 := 
  sorry

end correct_product_l101_101763


namespace inverse_proposition_l101_101330

theorem inverse_proposition (a : ℝ) :
  (a > 1 → a > 0) → (a > 0 → a > 1) :=
by 
  intros h1 h2
  sorry

end inverse_proposition_l101_101330


namespace toby_friends_girls_l101_101510

theorem toby_friends_girls (total_friends : ℕ) (num_boys : ℕ) (perc_boys : ℕ) 
  (h1 : perc_boys = 55) (h2 : num_boys = 33) (h3 : total_friends = 60) : 
  (total_friends - num_boys = 27) :=
by
  sorry

end toby_friends_girls_l101_101510


namespace find_a_and_b_l101_101080

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 - 6 * a * x^2 + b

theorem find_a_and_b :
  (∃ a b : ℝ, a ≠ 0 ∧
   (∀ x, -1 ≤ x ∧ x ≤ 2 → f a b x ≤ 3) ∧
   (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f a b x = 3) ∧
   (∃ x, -1 ≤ x ∧ x ≤ 2 ∧ f a b x = -29)
  ) → ((a = 2 ∧ b = 3) ∨ (a = -2 ∧ b = -29)) :=
sorry

end find_a_and_b_l101_101080


namespace parabola_vector_sum_distance_l101_101030

noncomputable def parabola_focus (x y : ℝ) : Prop := x^2 = 8 * y

noncomputable def on_parabola (x y : ℝ) : Prop := parabola_focus x y

theorem parabola_vector_sum_distance :
  ∀ (A B C : ℝ × ℝ) (F : ℝ × ℝ),
  on_parabola A.1 A.2 ∧ on_parabola B.1 B.2 ∧ on_parabola C.1 C.2 ∧
  F = (0, 2) ∧
  ((A.1 - F.1)^2 + (A.2 - F.2)^2) + ((B.1 - F.1)^2 + (B.2 - F.2)^2) + ((C.1 - F.1)^2 + (C.2 - F.2)^2) = 0
  → (abs ((A.2 + F.2)) + abs ((B.2 + F.2)) + abs ((C.2 + F.2))) = 12 :=
by sorry

end parabola_vector_sum_distance_l101_101030


namespace mutually_exclusive_events_l101_101162

-- Define the conditions
variable (redBalls greenBalls : ℕ)
variable (n : ℕ) -- Number of balls drawn
variable (event_one_red_ball event_two_green_balls : Prop)

-- Assumptions: more than two red balls and more than two green balls
axiom H1 : 2 < redBalls
axiom H2 : 2 < greenBalls

-- Assume that exactly one red ball and exactly two green balls are events
axiom H3 : event_one_red_ball = (n = 2 ∧ 1 ≤ redBalls ∧ 1 ≤ greenBalls)
axiom H4 : event_two_green_balls = (n = 2 ∧ greenBalls ≥ 2)

-- Definition of mutually exclusive events
def mutually_exclusive (A B : Prop) : Prop :=
  A ∧ B → false

-- Statement of the theorem
theorem mutually_exclusive_events :
  mutually_exclusive event_one_red_ball event_two_green_balls :=
by {
  sorry
}

end mutually_exclusive_events_l101_101162


namespace taxi_fare_proof_l101_101159

/-- Given equations representing the taxi fare conditions:
1. x + 7y = 16.5 (Person A's fare)
2. x + 11y = 22.5 (Person B's fare)

And using the value of the initial fare and additional charge per kilometer conditions,
prove the initial fare and additional charge and calculate the fare for a 7-kilometer ride. -/
theorem taxi_fare_proof (x y : ℝ) 
  (h1 : x + 7 * y = 16.5)
  (h2 : x + 11 * y = 22.5)
  (h3 : x = 6)
  (h4 : y = 1.5) :
  x = 6 ∧ y = 1.5 ∧ (x + y * (7 - 3)) = 12 :=
by
  sorry

end taxi_fare_proof_l101_101159


namespace product_of_values_l101_101182

theorem product_of_values (x : ℚ) (hx : abs ((18 / x) + 4) = 3) :
  x = -18 ∨ x = -18 / 7 ∧ -18 * (-18 / 7) = 324 / 7 :=
by sorry

end product_of_values_l101_101182


namespace find_k_l101_101925

theorem find_k (k : ℕ) : (1 / 3)^32 * (1 / 125)^k = 1 / 27^32 → k = 0 :=
by {
  sorry
}

end find_k_l101_101925


namespace cookies_per_child_is_22_l101_101462

def total_cookies (num_packages : ℕ) (cookies_per_package : ℕ) : ℕ :=
  num_packages * cookies_per_package

def total_children (num_friends : ℕ) : ℕ :=
  num_friends + 1

def cookies_per_child (total_cookies : ℕ) (total_children : ℕ) : ℕ :=
  total_cookies / total_children

theorem cookies_per_child_is_22 :
  total_cookies 5 36 / total_children 7 = 22 := 
by
  sorry

end cookies_per_child_is_22_l101_101462


namespace min_value_of_f_l101_101332

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (Real.log x / Real.log 2) * (Real.log (2 * x) / Real.log 2) else 0

theorem min_value_of_f : ∃ x > 0, f x = -1/4 :=
sorry

end min_value_of_f_l101_101332


namespace min_distance_l101_101722

theorem min_distance (x y z : ℝ) :
  ∃ (m : ℝ), m = (Real.sqrt (x^2 + y^2 + z^2) + Real.sqrt ((x+1)^2 + (y-2)^2 + (z-1)^2)) ∧ m = Real.sqrt 6 :=
by
  sorry

end min_distance_l101_101722


namespace equal_partitions_l101_101204

def weights : List ℕ := List.range (81 + 1) |>.map (λ n => n * n)

theorem equal_partitions (h : weights.sum = 178605) :
  ∃ P1 P2 P3 : List ℕ, P1.sum = 59535 ∧ P2.sum = 59535 ∧ P3.sum = 59535 ∧ P1 ++ P2 ++ P3 = weights := sorry

end equal_partitions_l101_101204


namespace intersection_range_l101_101265

def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem intersection_range :
  {m : ℝ | ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = m ∧ f x2 = m ∧ f x3 = m} = Set.Ioo (-3 : ℝ) 1 :=
by
  sorry

end intersection_range_l101_101265


namespace bolts_per_box_l101_101116

def total_bolts_and_nuts_used : Nat := 113
def bolts_left_over : Nat := 3
def nuts_left_over : Nat := 6
def boxes_of_bolts : Nat := 7
def boxes_of_nuts : Nat := 3
def nuts_per_box : Nat := 15

theorem bolts_per_box :
  let total_bolts_and_nuts := total_bolts_and_nuts_used + bolts_left_over + nuts_left_over
  let total_nuts := boxes_of_nuts * nuts_per_box
  let total_bolts := total_bolts_and_nuts - total_nuts
  let bolts_per_box := total_bolts / boxes_of_bolts
  bolts_per_box = 11 := by
  sorry

end bolts_per_box_l101_101116


namespace area_of_triangle_ABC_l101_101683

def A : ℝ × ℝ := (4, -3)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (2, -7)

theorem area_of_triangle_ABC : 
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := |v.1 * w.2 - v.2 * w.1|
  let triangle_area := parallelogram_area / 2
  triangle_area = 15 :=
by
  sorry

end area_of_triangle_ABC_l101_101683


namespace excluded_avg_mark_l101_101424

theorem excluded_avg_mark (N A A_remaining excluded_count : ℕ)
  (hN : N = 15)
  (hA : A = 80)
  (hA_remaining : A_remaining = 90) 
  (h_excluded : excluded_count = 5) :
  (A * N - A_remaining * (N - excluded_count)) / excluded_count = 60 := sorry

end excluded_avg_mark_l101_101424


namespace triangle_problem_l101_101443

/-- In triangle ABC, the sides opposite to angles A, B, C are a, b, c respectively.
Given that b = sqrt 2, c = 3, B + C = 3A, prove:
1. The length of side a equals sqrt 5.
2. sin (B + 3π/4) equals sqrt(10) / 10.
-/
theorem triangle_problem 
  (a b c A B C : ℝ)
  (hb : b = Real.sqrt 2)
  (hc : c = 3)
  (hBC : B + C = 3 * A)
  (hA : A = π / 4)
  : (a = Real.sqrt 5)
  ∧ (Real.sin (B + 3 * π / 4) = Real.sqrt 10 / 10) :=
sorry

end triangle_problem_l101_101443


namespace interval_satisfaction_l101_101563

theorem interval_satisfaction (a : ℝ) :
  (4 ≤ a / (3 * a - 6)) ∧ (a / (3 * a - 6) > 12) → a < 72 / 35 := 
by
  sorry

end interval_satisfaction_l101_101563


namespace additional_weight_difference_l101_101452

theorem additional_weight_difference (raw_squat sleeves_add wraps_percentage : ℝ) 
  (raw_squat_val : raw_squat = 600) 
  (sleeves_add_val : sleeves_add = 30) 
  (wraps_percentage_val : wraps_percentage = 0.25) : 
  (wraps_percentage * raw_squat) - sleeves_add = 120 :=
by
  rw [ raw_squat_val, sleeves_add_val, wraps_percentage_val ]
  norm_num

end additional_weight_difference_l101_101452


namespace parallel_condition_l101_101848

def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem parallel_condition (x : ℝ) : 
  let a := (2, 1)
  let b := (3 * x ^ 2 - 1, x)
  (x = 1 → are_parallel a b) ∧ 
  ∃ x', x' ≠ 1 ∧ are_parallel a (3 * x' ^ 2 - 1, x') :=
by
  sorry

end parallel_condition_l101_101848


namespace min_pieces_pie_l101_101114

theorem min_pieces_pie (p q : ℕ) (h_coprime : Nat.gcd p q = 1) : 
  ∃ n : ℕ, n = p + q - 1 ∧ 
    (∀ m, m < n → ¬ (∀ k : ℕ, (k < p → n % p = 0) ∧ (k < q → n % q = 0))) :=
sorry

end min_pieces_pie_l101_101114


namespace f_at_3_l101_101680

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_of_2 : f 2 = 1
axiom f_rec (x : ℝ) : f (x + 2) = f x + f 2

theorem f_at_3 : f 3 = 3 / 2 := 
by 
  sorry

end f_at_3_l101_101680


namespace repeating_decimal_to_fraction_l101_101554

theorem repeating_decimal_to_fraction : 
∀ (x : ℝ), x = 4 + (0.0036 / (1 - 0.01)) → x = 144/33 :=
by
  intro x hx
  -- This is a placeholder where the conversion proof would go.
  sorry

end repeating_decimal_to_fraction_l101_101554


namespace diagonal_length_of_cuboid_l101_101069

theorem diagonal_length_of_cuboid
  (a b c : ℝ)
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : c * a = Real.sqrt 6) : 
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 6 := 
sorry

end diagonal_length_of_cuboid_l101_101069


namespace product_of_three_numbers_l101_101158

theorem product_of_three_numbers :
  ∃ (a b c : ℚ), 
    a + b + c = 30 ∧ a = 2 * (b + c) ∧ b = 6 * c ∧ a * b * c = 12000 / 49 :=
by
  sorry

end product_of_three_numbers_l101_101158


namespace train_speed_l101_101237

theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) (total_distance : ℝ) 
    (speed_mps : ℝ) (speed_kmph : ℝ) 
    (h1 : train_length = 360) 
    (h2 : bridge_length = 140) 
    (h3 : time = 34.61538461538461) 
    (h4 : total_distance = train_length + bridge_length) 
    (h5 : speed_mps = total_distance / time) 
    (h6 : speed_kmph = speed_mps * 3.6) : 
  speed_kmph = 52 := 
by 
  sorry

end train_speed_l101_101237


namespace distribution_methods_l101_101035

theorem distribution_methods (n m k : Nat) (h : n = 23) (h1 : m = 10) (h2 : k = 2) :
  (∃ d : Nat, d = Nat.choose m 1 + 2 * Nat.choose m 2 + Nat.choose m 3) →
  ∃ x : Nat, x = 220 :=
by
  sorry

end distribution_methods_l101_101035


namespace find_number_l101_101907

theorem find_number (x : ℝ) : (x + 1) / (x + 5) = (x + 5) / (x + 13) → x = 3 :=
sorry

end find_number_l101_101907


namespace imaginary_part_inv_z_l101_101995

def z : ℂ := 1 - 2 * Complex.I

theorem imaginary_part_inv_z : Complex.im (1 / z) = 2 / 5 :=
by
  -- proof to be filled in
  sorry

end imaginary_part_inv_z_l101_101995


namespace max_value_expression_l101_101949

theorem max_value_expression (x k : ℕ) (h₀ : 0 < x) (h₁ : 0 < k) (y := k * x) : 
  (∀ x k : ℕ, 0 < x → 0 < k → y = k * x → ∃ m : ℝ, m = 2 ∧ 
    ∀ x k : ℕ, 0 < x → 0 < k → y = k * x → (x + y)^2 / (x^2 + y^2) ≤ 2) :=
sorry

end max_value_expression_l101_101949


namespace find_Q_div_P_l101_101546

variable (P Q : ℚ)
variable (h_eq : ∀ x : ℝ, x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -3 → 
  P / (x - 3) + Q / (x^2 + x - 6) = (x^2 + 3*x + 1) / (x^3 - x^2 - 12*x))

theorem find_Q_div_P : Q / P = -6 / 13 := by
  sorry

end find_Q_div_P_l101_101546


namespace find_m_range_l101_101580

variable {x y m : ℝ}

theorem find_m_range (h1 : x + 2 * y = m + 4) (h2 : 2 * x + y = 2 * m - 1)
    (h3 : x + y < 2) (h4 : x - y < 4) : m < 1 := by
  sorry

end find_m_range_l101_101580


namespace problem_statement_l101_101748

theorem problem_statement (a b : ℝ) (h : a < b) : a - b < 0 :=
sorry

end problem_statement_l101_101748


namespace distinct_flavors_count_l101_101514

-- Define the number of available candies
def red_candies := 3
def green_candies := 2
def blue_candies := 4

-- Define what it means for a flavor to be valid: includes at least one candy of each color.
def is_valid_flavor (x y z : Nat) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ x ≤ red_candies ∧ y ≤ green_candies ∧ z ≤ blue_candies

-- Define what it means for two flavors to have the same ratio
def same_ratio (x1 y1 z1 x2 y2 z2 : Nat) : Prop :=
  x1 * y2 * z2 = x2 * y1 * z1

-- Define the proof problem: the number of distinct flavors
theorem distinct_flavors_count :
  ∃ n, n = 21 ∧ ∀ (x y z : Nat), is_valid_flavor x y z ↔ (∃ x' y' z', is_valid_flavor x' y' z' ∧ ¬ same_ratio x y z x' y' z') :=
sorry

end distinct_flavors_count_l101_101514


namespace negation_of_proposition_l101_101053

-- Given condition
def original_statement (a : ℝ) : Prop :=
  ∃ x : ℝ, a*x^2 - 2*a*x + 1 ≤ 0

-- Correct answer (negation statement)
def negated_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - 2*a*x + 1 > 0

-- Statement to prove
theorem negation_of_proposition (a : ℝ) :
  ¬ (original_statement a) ↔ (negated_statement a) :=
by 
  sorry

end negation_of_proposition_l101_101053


namespace find_k_intersects_parabola_at_one_point_l101_101447

theorem find_k_intersects_parabola_at_one_point :
  ∃ k : ℝ, (∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k ↔ y = (-4 / (2 * 3))) →
    k = 25 / 3 :=
by sorry

end find_k_intersects_parabola_at_one_point_l101_101447


namespace total_time_to_virgo_l101_101091

def train_ride : ℝ := 5
def first_layover : ℝ := 1.5
def bus_ride : ℝ := 4
def second_layover : ℝ := 0.5
def first_flight : ℝ := 6
def third_layover : ℝ := 2
def second_flight : ℝ := 3 * bus_ride
def fourth_layover : ℝ := 3
def car_drive : ℝ := 3.5
def first_boat_ride : ℝ := 1.5
def fifth_layover : ℝ := 0.75
def second_boat_ride : ℝ := 2 * first_boat_ride - 0.5
def final_walk : ℝ := 1.25

def total_time : ℝ := train_ride + first_layover + bus_ride + second_layover + first_flight + third_layover + second_flight + fourth_layover + car_drive + first_boat_ride + fifth_layover + second_boat_ride + final_walk

theorem total_time_to_virgo : total_time = 44 := by
  simp [train_ride, first_layover, bus_ride, second_layover, first_flight, third_layover, second_flight, fourth_layover, car_drive, first_boat_ride, fifth_layover, second_boat_ride, final_walk, total_time]
  sorry

end total_time_to_virgo_l101_101091


namespace correct_phrase_l101_101089

-- Define statements representing each option
def option_A : String := "as twice much"
def option_B : String := "much as twice"
def option_C : String := "twice as much"
def option_D : String := "as much twice"

-- The correct option
def correct_option : String := "twice as much"

-- The main theorem statement
theorem correct_phrase : option_C = correct_option :=
by
  sorry

end correct_phrase_l101_101089


namespace shape_is_cylinder_l101_101764

def positive_constant (c : ℝ) := c > 0

def is_cylinder (r θ z : ℝ) (c : ℝ) : Prop :=
  r = c

theorem shape_is_cylinder (c : ℝ) (r θ z : ℝ) 
  (h_pos : positive_constant c) (h_eq : r = c) :
  is_cylinder r θ z c := by
  sorry

end shape_is_cylinder_l101_101764


namespace Laura_bought_one_kg_of_potatoes_l101_101667

theorem Laura_bought_one_kg_of_potatoes :
  let price_salad : ℝ := 3
  let price_beef_per_kg : ℝ := 2 * price_salad
  let price_potato_per_kg : ℝ := price_salad * (1 / 3)
  let price_juice_per_liter : ℝ := 1.5
  let total_cost : ℝ := 22
  let num_salads : ℝ := 2
  let num_beef_kg : ℝ := 2
  let num_juice_liters : ℝ := 2
  let cost_salads := num_salads * price_salad
  let cost_beef := num_beef_kg * price_beef_per_kg
  let cost_juice := num_juice_liters * price_juice_per_liter
  (total_cost - (cost_salads + cost_beef + cost_juice)) / price_potato_per_kg = 1 :=
sorry

end Laura_bought_one_kg_of_potatoes_l101_101667


namespace problem_I_problem_II_problem_III_problem_IV_l101_101363

/-- Problem I: Given: (2x - y)^2 = 1, Prove: y = 2x - 1 ∨ y = 2x + 1 --/
theorem problem_I (x y : ℝ) : (2 * x - y) ^ 2 = 1 → (y = 2 * x - 1) ∨ (y = 2 * x + 1) := 
sorry

/-- Problem II: Given: 16x^4 - 8x^2y^2 + y^4 - 8x^2 - 2y^2 + 1 = 0, Prove: y = 2x - 1 ∨ y = -2x - 1 ∨ y = 2x + 1 ∨ y = -2x + 1 --/
theorem problem_II (x y : ℝ) : 16 * x^4 - 8 * x^2 * y^2 + y^4 - 8 * x^2 - 2 * y^2 + 1 = 0 ↔ 
    (y = 2 * x - 1) ∨ (y = -2 * x - 1) ∨ (y = 2 * x + 1) ∨ (y = -2 * x + 1) := 
sorry

/-- Problem III: Given: x^2 * (1 - |y| / y) + y^2 + y * |y| = 8, Prove: (y = 2 ∧ y > 0) ∨ ((x = 2 ∨ x = -2) ∧ y < 0) --/
theorem problem_III (x y : ℝ) (hy : y ≠ 0) : x^2 * (1 - abs y / y) + y^2 + y * abs y = 8 →
    (y = 2 ∧ y > 0) ∨ ((x = 2 ∨ x = -2) ∧ y < 0) := 
sorry

/-- Problem IV: Given: x^2 + x * |x| + y^2 + (|x| * y^2 / x) = 8, Prove: x^2 + y^2 = 4 ∧ x > 0 --/
theorem problem_IV (x y : ℝ) (hx : x ≠ 0) : x^2 + x * abs x + y^2 + (abs x * y^2 / x) = 8 →
    (x^2 + y^2 = 4 ∧ x > 0) := 
sorry

end problem_I_problem_II_problem_III_problem_IV_l101_101363


namespace infinite_series_sum_l101_101090

noncomputable def S : ℝ :=
∑' n, (if n % 3 == 0 then 1 / (3 ^ (n / 3)) else if n % 3 == 1 then -1 / (3 ^ (n / 3 + 1)) else -1 / (3 ^ (n / 3 + 2)))

theorem infinite_series_sum : S = 15 / 26 := by
  sorry

end infinite_series_sum_l101_101090


namespace magnification_factor_l101_101755

variable (diameter_magnified : ℝ)
variable (diameter_actual : ℝ)
variable (M : ℝ)

theorem magnification_factor
    (h_magnified : diameter_magnified = 0.3)
    (h_actual : diameter_actual = 0.0003) :
    M = diameter_magnified / diameter_actual ↔ M = 1000 := by
  sorry

end magnification_factor_l101_101755


namespace s_of_1_l101_101590

def t (x : ℚ) : ℚ := 5 * x - 10
def s (y : ℚ) : ℚ := (y^2 / (5^2)) + (5 * y / 5) + 6  -- reformulated to fit conditions

theorem s_of_1 :
  s (1 : ℚ) = 546 / 25 := by
  sorry

end s_of_1_l101_101590


namespace polygon_area_is_1008_l101_101153

variables (vertices : List (ℕ × ℕ)) (units : ℕ)

def polygon_area (vertices : List (ℕ × ℕ)) : ℕ :=
sorry -- The function would compute the area based on vertices.

theorem polygon_area_is_1008 :
  vertices = [(0, 0), (12, 0), (24, 12), (24, 0), (36, 0), (36, 24), (24, 36), (12, 36), (0, 36), (0, 24), (0, 0)] →
  units = 1 →
  polygon_area vertices = 1008 :=
sorry

end polygon_area_is_1008_l101_101153


namespace sum_of_points_probabilities_l101_101975

-- Define probabilities for the sums of 2, 3, and 4
def P_A : ℚ := 1 / 36
def P_B : ℚ := 2 / 36
def P_C : ℚ := 3 / 36

-- Theorem statement
theorem sum_of_points_probabilities :
  (P_A < P_B) ∧ (P_B < P_C) :=
  sorry

end sum_of_points_probabilities_l101_101975


namespace amount_after_two_years_l101_101650

def present_value : ℝ := 70400
def rate : ℝ := 0.125
def years : ℕ := 2
def final_amount := present_value * (1 + rate) ^ years

theorem amount_after_two_years : final_amount = 89070 :=
by sorry

end amount_after_two_years_l101_101650


namespace sin_alpha_plus_pi_over_2_l101_101242

theorem sin_alpha_plus_pi_over_2 
  (h1 : Real.pi / 2 < α) (h2 : α < Real.pi) (h3 : Real.tan α = -4 / 3) :
  Real.sin (α + Real.pi / 2) = -3 / 5 :=
by
  sorry

end sin_alpha_plus_pi_over_2_l101_101242


namespace tetrahedron_has_six_edges_l101_101482

-- Define what a tetrahedron is
inductive Tetrahedron : Type
| mk : Tetrahedron

-- Define the number of edges of a Tetrahedron
def edges_of_tetrahedron (t : Tetrahedron) : Nat := 6

theorem tetrahedron_has_six_edges (t : Tetrahedron) : edges_of_tetrahedron t = 6 := 
by
  sorry

end tetrahedron_has_six_edges_l101_101482


namespace inscribed_circle_radius_l101_101061

theorem inscribed_circle_radius (R : ℝ) (h : 0 < R) : 
  ∃ x : ℝ, (x = R / 3) :=
by
  -- Given conditions
  have h1 : R > 0 := h

  -- Mathematical proof statement derived from conditions
  sorry

end inscribed_circle_radius_l101_101061


namespace smallest_positive_number_among_options_l101_101750

theorem smallest_positive_number_among_options :
  (10 > 3 * Real.sqrt 11) →
  (51 > 10 * Real.sqrt 26) →
  min (10 - 3 * Real.sqrt 11) (51 - 10 * Real.sqrt 26) = 51 - 10 * Real.sqrt 26 :=
by
  intros h1 h2
  sorry

end smallest_positive_number_among_options_l101_101750


namespace reema_simple_interest_l101_101687

-- Definitions and conditions
def principal : ℕ := 1200
def rate_of_interest : ℕ := 6
def time_period : ℕ := rate_of_interest

-- Simple interest calculation
def calculate_simple_interest (P R T: ℕ) : ℕ :=
  (P * R * T) / 100

-- The theorem to prove that Reema paid Rs 432 as simple interest.
theorem reema_simple_interest : calculate_simple_interest principal rate_of_interest time_period = 432 := 
  sorry

end reema_simple_interest_l101_101687


namespace plane_equation_intercept_l101_101339

theorem plane_equation_intercept (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  ∀ x y z : ℝ, ∃ k : ℝ, k = 1 → (x / a + y / b + z / c) = k :=
by sorry

end plane_equation_intercept_l101_101339


namespace seats_per_table_l101_101841

-- Definitions based on conditions
def tables := 4
def total_people := 32

-- Statement to prove
theorem seats_per_table : (total_people / tables) = 8 :=
by 
  sorry

end seats_per_table_l101_101841


namespace extreme_value_h_tangent_to_both_l101_101702

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a*x + 1
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.log x - a
noncomputable def h (x : ℝ) : ℝ := f x 1 - g x 1

theorem extreme_value_h : h (1/2) = 11/4 + Real.log 2 := by
  sorry

theorem tangent_to_both : ∀ (a : ℝ), ∃ x₁ x₂ : ℝ, (2 * x₁ + a = 1 / x₂) ∧ 
  ((x₁ = (1 / (2 * x₂)) - (a / 2)) ∧ (a ≥ -1)) := by
  sorry

end extreme_value_h_tangent_to_both_l101_101702


namespace find_x_value_l101_101836

theorem find_x_value (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
sorry

end find_x_value_l101_101836


namespace solve_equation_l101_101297

theorem solve_equation : ∀ (x : ℝ), (x / 2 - 1 = 3) → x = 8 :=
by
  intro x h
  sorry

end solve_equation_l101_101297


namespace find_large_number_l101_101440

theorem find_large_number (L S : ℕ) (h1 : L - S = 1515) (h2 : L = 16 * S + 15) : L = 1615 :=
sorry

end find_large_number_l101_101440


namespace calculate_exponent_product_l101_101376

theorem calculate_exponent_product : (2^2021) * (-1/2)^2022 = (1/2) :=
by
  sorry

end calculate_exponent_product_l101_101376


namespace bucket_full_weight_l101_101118

theorem bucket_full_weight (c d : ℝ) (x y : ℝ) (h1 : x + (1 / 4) * y = c) (h2 : x + (3 / 4) * y = d) : 
  x + y = (3 * d - 3 * c) / 2 :=
by
  sorry

end bucket_full_weight_l101_101118


namespace number_of_sequences_with_at_least_two_reds_l101_101927

theorem number_of_sequences_with_at_least_two_reds (n : ℕ) (h : n ≥ 2) :
  let T_n := 3 * 2^(n - 1)
  let R_0 := 2
  let R_1n := 4 * n - 4
  T_n - R_0 - R_1n = 3 * 2^(n - 1) - 4 * n + 2 :=
by
  intros
  let T_n := 3 * 2^(n - 1)
  let R_0 := 2
  let R_1n := 4 * n - 4
  show T_n - R_0 - R_1n = 3 * 2^(n - 1) - 4 * n + 2
  sorry

end number_of_sequences_with_at_least_two_reds_l101_101927


namespace tangent_lines_through_point_l101_101632

theorem tangent_lines_through_point :
  ∃ k : ℚ, ((5  * k - 12 * (36 - k * 2) + 36 = 0) ∨ (2 = 0)) := sorry

end tangent_lines_through_point_l101_101632


namespace correct_assertions_l101_101464

variables {A B : Type} (f : A → B)

-- 1. Different elements in set A can have the same image in set B
def statement_1 : Prop := ∃ a1 a2 : A, a1 ≠ a2 ∧ f a1 = f a2

-- 2. A single element in set A can have different images in B
def statement_2 : Prop := ∃ a1 : A, ∃ b1 b2 : B, b1 ≠ b2 ∧ (f a1 = b1 ∧ f a1 = b2)

-- 3. There can be elements in set B that do not have a pre-image in A
def statement_3 : Prop := ∃ b : B, ∀ a : A, f a ≠ b

-- Correct answer is statements 1 and 3 are true, statement 2 is false
theorem correct_assertions : statement_1 f ∧ ¬statement_2 f ∧ statement_3 f := sorry

end correct_assertions_l101_101464


namespace brady_passing_yards_proof_l101_101599

def tom_brady_current_passing_yards 
  (record_yards : ℕ) (games_left : ℕ) (average_yards_needed : ℕ) 
  (total_yards_needed_to_break_record : ℕ :=
    record_yards + 1) : ℕ :=
  total_yards_needed_to_break_record - games_left * average_yards_needed

theorem brady_passing_yards_proof :
  tom_brady_current_passing_yards 5999 6 300 = 4200 :=
by 
  sorry

end brady_passing_yards_proof_l101_101599


namespace bottle_count_l101_101087

theorem bottle_count :
  ∃ N x : ℕ, 
    N = x^2 + 36 ∧ N = (x + 1)^2 + 3 :=
by 
  sorry

end bottle_count_l101_101087


namespace area_of_circle_with_radius_2_is_4pi_l101_101362

theorem area_of_circle_with_radius_2_is_4pi :
  ∀ (π : ℝ), ∀ (r : ℝ), r = 2 → π > 0 → π * r^2 = 4 * π := 
by
  intros π r hr hπ
  sorry

end area_of_circle_with_radius_2_is_4pi_l101_101362


namespace polar_coordinates_standard_representation_l101_101210

theorem polar_coordinates_standard_representation :
  ∀ (r θ : ℝ), (r, θ) = (-4, 5 * Real.pi / 6) → (∃ (r' θ' : ℝ), r' > 0 ∧ (r', θ') = (4, 11 * Real.pi / 6))
:= by
  sorry

end polar_coordinates_standard_representation_l101_101210


namespace minimize_distance_AP_BP_l101_101532

theorem minimize_distance_AP_BP :
  ∃ P : ℝ × ℝ, P.1 = 0 ∧ P.2 = -1 ∧
    ∀ P' : ℝ × ℝ, P'.1 = 0 → 
      (dist (3, 2) P + dist (1, -2) P) ≤ (dist (3, 2) P' + dist (1, -2) P') := by
sorry

end minimize_distance_AP_BP_l101_101532


namespace compare_neg_fractions_l101_101827

theorem compare_neg_fractions : (-5 / 4) < (-4 / 5) := sorry

end compare_neg_fractions_l101_101827


namespace bisection_interval_length_l101_101039

theorem bisection_interval_length (n : ℕ) : 
  (1 / (2:ℝ)^n) ≤ 0.01 → n ≥ 7 :=
by 
  sorry

end bisection_interval_length_l101_101039


namespace prop_disjunction_is_true_l101_101033

variable (p q : Prop)
axiom hp : p
axiom hq : ¬q

theorem prop_disjunction_is_true (hp : p) (hq : ¬q) : p ∨ q :=
by
  sorry

end prop_disjunction_is_true_l101_101033


namespace time_to_cover_escalator_l101_101594

theorem time_to_cover_escalator (escalator_speed person_speed length : ℕ) (h1 : escalator_speed = 11) (h2 : person_speed = 3) (h3 : length = 126) : 
  length / (escalator_speed + person_speed) = 9 := by
  sorry

end time_to_cover_escalator_l101_101594


namespace sequence_count_l101_101301

theorem sequence_count (a : ℕ → ℤ) (h₁ : a 1 = 0) (h₂ : a 11 = 4) 
  (h₃ : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → |a (k + 1) - a k| = 1) : 
  ∃ (n : ℕ), n = 120 :=
by
  sorry

end sequence_count_l101_101301


namespace car_speed_l101_101317

theorem car_speed (v : ℝ) : 
  (4 + (1 / (80 / 3600))) = (1 / (v / 3600)) → v = 3600 / 49 :=
sorry

end car_speed_l101_101317


namespace quadratic_no_third_quadrant_l101_101144

theorem quadratic_no_third_quadrant (x y : ℝ) : 
  (y = x^2 - 2 * x) → ¬(x < 0 ∧ y < 0) :=
by
  intro hy
  sorry

end quadratic_no_third_quadrant_l101_101144


namespace fred_earned_from_car_wash_l101_101761

def weekly_allowance : ℕ := 16
def spent_on_movies : ℕ := weekly_allowance / 2
def amount_after_movies : ℕ := weekly_allowance - spent_on_movies
def final_amount : ℕ := 14
def earned_from_car_wash : ℕ := final_amount - amount_after_movies

theorem fred_earned_from_car_wash : earned_from_car_wash = 6 := by
  sorry

end fred_earned_from_car_wash_l101_101761


namespace eval_f_four_times_l101_101980

noncomputable def f (z : Complex) : Complex := 
if z.im ≠ 0 then z * z else -(z * z)

theorem eval_f_four_times : 
  f (f (f (f (Complex.mk 2 1)))) = Complex.mk 164833 354192 := 
by 
  sorry

end eval_f_four_times_l101_101980


namespace lindy_total_distance_l101_101340

theorem lindy_total_distance (distance_jc : ℝ) (speed_j : ℝ) (speed_c : ℝ) (speed_l : ℝ)
  (h1 : distance_jc = 270) (h2 : speed_j = 4) (h3 : speed_c = 5) (h4 : speed_l = 8) : 
  ∃ time : ℝ, time = distance_jc / (speed_j + speed_c) ∧ speed_l * time = 240 :=
by
  sorry

end lindy_total_distance_l101_101340


namespace heartsuit_ratio_l101_101408

-- Define the operation \heartsuit
def heartsuit (n m : ℕ) : ℕ := n^3 * m^2

-- The proposition we want to prove
theorem heartsuit_ratio :
  heartsuit 2 4 / heartsuit 4 2 = 1 / 2 := by
  sorry

end heartsuit_ratio_l101_101408


namespace yellow_more_than_green_by_l101_101798

-- Define the problem using the given conditions.
def weight_yellow_block : ℝ := 0.6
def weight_green_block  : ℝ := 0.4

-- State the theorem that the yellow block weighs 0.2 pounds more than the green block.
theorem yellow_more_than_green_by : weight_yellow_block - weight_green_block = 0.2 :=
by sorry

end yellow_more_than_green_by_l101_101798


namespace Justin_run_home_time_l101_101676

variable (blocksPerMinute : ℝ) (totalBlocks : ℝ)

theorem Justin_run_home_time (h1 : blocksPerMinute = 2 / 1.5) (h2 : totalBlocks = 8) :
  totalBlocks / blocksPerMinute = 6 := by
  sorry

end Justin_run_home_time_l101_101676


namespace inequality_proof_l101_101878

theorem inequality_proof (a b : ℝ) (h : a < b) : -a - 1 > -b - 1 :=
sorry

end inequality_proof_l101_101878


namespace determine_coefficients_l101_101354

theorem determine_coefficients (p q : ℝ) :
  (∃ x : ℝ, x^2 + p * x + q = 0 ∧ x = p) ∧ (∃ y : ℝ, y^2 + p * y + q = 0 ∧ y = q)
  ↔ (p = 0 ∧ q = 0) ∨ (p = 1 ∧ q = -2) := by
sorry

end determine_coefficients_l101_101354


namespace geometric_sequence_sum_inequality_l101_101217

theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ) (q : ℝ) (n : ℕ)
  (h_geom : ∀ k, a (k + 1) = a k * q)
  (h_pos : ∀ k ≤ 7, a k > 0)
  (h_q_ne_one : q ≠ 1) :
  a 0 + a 7 > a 3 + a 4 :=
sorry

end geometric_sequence_sum_inequality_l101_101217


namespace students_in_each_group_l101_101269

theorem students_in_each_group (num_boys : ℕ) (num_girls : ℕ) (num_groups : ℕ) 
  (h_boys : num_boys = 26) (h_girls : num_girls = 46) (h_groups : num_groups = 8) : 
  (num_boys + num_girls) / num_groups = 9 := 
by 
  sorry

end students_in_each_group_l101_101269


namespace leading_digit_not_necessarily_one_l101_101706

-- Define a condition to check if the leading digit of a number is the same
def same_leading_digit (x: ℕ) (n: ℕ) : Prop :=
  (Nat.digits 10 x).head? = (Nat.digits 10 (x^n)).head?

-- Theorem stating the digit does not need to be 1 under given conditions
theorem leading_digit_not_necessarily_one :
  (∃ x: ℕ, x > 1 ∧ same_leading_digit x 2 ∧ same_leading_digit x 3) ∧ 
  (∃ x: ℕ, x > 1 ∧ ∀ n: ℕ, 1 ≤ n ∧ n ≤ 2015 → same_leading_digit x n) :=
sorry

end leading_digit_not_necessarily_one_l101_101706


namespace size_of_first_type_package_is_5_l101_101816

noncomputable def size_of_first_type_package (total_coffee : ℕ) (num_first_type : ℕ) (num_second_type : ℕ) (size_second_type : ℕ) : ℕ :=
  (total_coffee - num_second_type * size_second_type) / num_first_type

theorem size_of_first_type_package_is_5 :
  size_of_first_type_package 70 (4 + 2) 4 10 = 5 :=
by
  sorry

end size_of_first_type_package_is_5_l101_101816


namespace unique_solution_pair_l101_101394

theorem unique_solution_pair (x y : ℝ) :
  (4 * x ^ 2 + 6 * x + 4) * (4 * y ^ 2 - 12 * y + 25) = 28 →
  (x, y) = (-3 / 4, 3 / 2) := by
  intro h
  sorry

end unique_solution_pair_l101_101394


namespace arrow_hits_apple_l101_101556

noncomputable def time_to_hit (L V0 : ℝ) (α β : ℝ) : ℝ :=
  (L / V0) * (Real.sin β / Real.sin (α + β))

theorem arrow_hits_apple (g : ℝ) (L V0 : ℝ) (α β : ℝ) (h : (L / V0) * (Real.sin β / Real.sin (α + β)) = 3 / 4) 
  : time_to_hit L V0 α β = 3 / 4 := 
  by
  sorry

end arrow_hits_apple_l101_101556


namespace f_satisfies_condition_l101_101076

noncomputable def f (x : ℝ) : ℝ := 2^x

-- Prove that f(x + 1) = 2 * f(x) for the defined function f.
theorem f_satisfies_condition (x : ℝ) : f (x + 1) = 2 * f x := by
  show 2^(x + 1) = 2 * 2^x
  sorry

end f_satisfies_condition_l101_101076


namespace total_fishes_caught_l101_101932

def melanieCatches : ℕ := 8
def tomCatches : ℕ := 3 * melanieCatches
def totalFishes : ℕ := melanieCatches + tomCatches

theorem total_fishes_caught : totalFishes = 32 := by
  sorry

end total_fishes_caught_l101_101932


namespace diagonal_cells_crossed_l101_101658

theorem diagonal_cells_crossed (m n : ℕ) (h_m : m = 199) (h_n : n = 991) :
  (m + n - Nat.gcd m n) = 1189 := by
  sorry

end diagonal_cells_crossed_l101_101658


namespace blake_change_given_l101_101002

theorem blake_change_given :
  let oranges := 40
  let apples := 50
  let mangoes := 60
  let total_amount := 300
  let total_spent := oranges + apples + mangoes
  let change_given := total_amount - total_spent
  change_given = 150 :=
by
  sorry

end blake_change_given_l101_101002


namespace part1_part2_part3_l101_101071

noncomputable def seq (a : ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else (1 - a) / n

theorem part1 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (a1_eq : seq a 1 = 1 / 2) (a2_eq : seq a 2 = 1 / 4) : true :=
by trivial

theorem part2 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (n : ℕ) (hn : n > 0) : 0 < seq a n ∧ seq a n < 1 :=
sorry

theorem part3 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (n : ℕ) (hn : n > 0) : seq a n > seq a (n + 1) :=
sorry

end part1_part2_part3_l101_101071


namespace sum_of_squares_divisibility_l101_101309

theorem sum_of_squares_divisibility (n : ℤ) : 
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  let S := a^2 + b^2 + c^2
  (S % 4 = 0 ∧ S % 3 ≠ 0) :=
by
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  let S := a^2 + b^2 + c^2
  sorry

end sum_of_squares_divisibility_l101_101309


namespace part1_part2_l101_101498

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

/-- Given sequence properties -/
axiom h1 : a 1 = 5
axiom h2 : ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1) + 2^n - 1

/-- Part (I): Proving the sequence is arithmetic -/
theorem part1 (n : ℕ) : ∃ d, (∀ m ≥ 1, (a (m + 1) - 1) / 2^(m + 1) - (a m - 1) / 2^m = d)
∧ ((a 1 - 1) / 2 = 2) := sorry

/-- Part (II): Sum of the first n terms -/
theorem part2 (n : ℕ) : S n = n * 2^(n+1) := sorry

end part1_part2_l101_101498


namespace sequence_general_term_l101_101429

theorem sequence_general_term
  (a : ℕ → ℝ)
  (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n, a (n + 1) = 3 * a n + 7) :
  ∀ n, a n = 4 * 3^(n - 1) - 7 / 2 :=
by
  sorry

end sequence_general_term_l101_101429


namespace tank_capacity_l101_101142

theorem tank_capacity (T : ℝ) (h : 0.4 * T = 0.9 * T - 36) : T = 72 := by
  sorry

end tank_capacity_l101_101142


namespace saree_discount_l101_101734

theorem saree_discount (x : ℝ) : 
  let original_price := 495
  let final_price := 378.675
  let discounted_price := original_price * ((100 - x) / 100) * 0.9
  discounted_price = final_price -> x = 15 := 
by
  intro h
  sorry

end saree_discount_l101_101734


namespace problem_statement_l101_101952

theorem problem_statement (x y : ℝ) (h1 : 4 * x + y = 12) (h2 : x + 4 * y = 18) :
  17 * x ^ 2 + 24 * x * y + 17 * y ^ 2 = 532 :=
by
  sorry

end problem_statement_l101_101952


namespace arlene_average_pace_l101_101185

theorem arlene_average_pace :
  ∃ pace : ℝ, pace = 24 / (6 - 0.75) ∧ pace = 4.57 := 
by
  sorry

end arlene_average_pace_l101_101185


namespace Brian_traveled_60_miles_l101_101887

theorem Brian_traveled_60_miles (mpg gallons : ℕ) (hmpg : mpg = 20) (hgallons : gallons = 3) :
    mpg * gallons = 60 := by
  sorry

end Brian_traveled_60_miles_l101_101887


namespace john_bathroom_uses_during_movie_and_intermissions_l101_101285

-- Define the conditions
def uses_bathroom_interval := 50   -- John uses the bathroom every 50 minutes
def walking_time := 5              -- It takes him an additional 5 minutes to walk to and from the bathroom
def movie_length := 150            -- The movie length in minutes (2.5 hours)
def intermission_length := 15      -- Each intermission length in minutes
def intermission_count := 2        -- The number of intermissions

-- Derived condition
def effective_interval := uses_bathroom_interval + walking_time

-- Total movie time including intermissions
def total_movie_time := movie_length + (intermission_length * intermission_count)

-- Define the theorem to be proved
theorem john_bathroom_uses_during_movie_and_intermissions : 
  ∃ n : ℕ, n = 3 + 2 ∧ total_movie_time = 180 ∧ effective_interval = 55 :=
by
  sorry

end john_bathroom_uses_during_movie_and_intermissions_l101_101285


namespace range_of_mu_l101_101305

noncomputable def problem_statement (a b μ : ℝ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (0 < μ) ∧ (1 / a + 9 / b = 1) → (0 < μ ∧ μ ≤ 16)

theorem range_of_mu (a b μ : ℝ) : problem_statement a b μ :=
  sorry

end range_of_mu_l101_101305


namespace paint_containers_left_l101_101790

theorem paint_containers_left (initial_containers : ℕ)
  (tiled_wall_containers : ℕ)
  (ceiling_containers : ℕ)
  (gradient_walls : ℕ)
  (additional_gradient_containers_per_wall : ℕ)
  (remaining_containers : ℕ) :
  initial_containers = 16 →
  tiled_wall_containers = 1 →
  ceiling_containers = 1 →
  gradient_walls = 3 →
  additional_gradient_containers_per_wall = 1 →
  remaining_containers = initial_containers - tiled_wall_containers - (ceiling_containers + gradient_walls * additional_gradient_containers_per_wall) →
  remaining_containers = 11 :=
by
  intros h_initial h_tiled h_ceiling h_gradient_walls h_additional_gradient h_remaining_calc
  rw [h_initial, h_tiled, h_ceiling, h_gradient_walls, h_additional_gradient] at h_remaining_calc
  exact h_remaining_calc

end paint_containers_left_l101_101790


namespace selection_3m2f_selection_at_least_one_captain_selection_at_least_one_female_selection_captain_and_female_l101_101564

section ProofProblems

-- Definitions and constants
def num_males := 6
def num_females := 4
def total_athletes := 10
def num_selections := 5
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- 1. Number of selection methods for 3 males and 2 females
theorem selection_3m2f : binom 6 3 * binom 4 2 = 120 := by sorry

-- 2. Number of selection methods with at least one captain
theorem selection_at_least_one_captain :
  2 * binom 8 4 + binom 8 3 = 196 := by sorry

-- 3. Number of selection methods with at least one female athlete
theorem selection_at_least_one_female :
  binom 10 5 - binom 6 5 = 246 := by sorry

-- 4. Number of selection methods with both a captain and at least one female athlete
theorem selection_captain_and_female :
  binom 9 4 + binom 8 4 - binom 5 4 = 191 := by sorry

end ProofProblems

end selection_3m2f_selection_at_least_one_captain_selection_at_least_one_female_selection_captain_and_female_l101_101564


namespace circle_equation_through_points_l101_101003

-- Line and circle definitions
def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 15 = 0

-- Intersection point definition
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ circle1 x y

-- Revised circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 28 * x - 15 * y = 0

-- Proof statement
theorem circle_equation_through_points :
  (∀ x y, intersection_point x y → circle_equation x y) ∧ circle_equation 0 0 :=
sorry

end circle_equation_through_points_l101_101003


namespace interval_between_births_l101_101057

def youngest_child_age : ℕ := 6

def sum_of_ages (I : ℝ) : ℝ :=
  youngest_child_age + (youngest_child_age + I) + (youngest_child_age + 2 * I) + (youngest_child_age + 3 * I) + (youngest_child_age + 4 * I)

theorem interval_between_births : ∃ (I : ℝ), sum_of_ages I = 60 ∧ I = 3.6 := 
by
  sorry

end interval_between_births_l101_101057


namespace reimbursement_calculation_l101_101736

variable (total_paid : ℕ) (pieces : ℕ) (cost_per_piece : ℕ)

theorem reimbursement_calculation
  (h1 : total_paid = 20700)
  (h2 : pieces = 150)
  (h3 : cost_per_piece = 134) :
  total_paid - (pieces * cost_per_piece) = 600 := 
by
  sorry

end reimbursement_calculation_l101_101736


namespace modem_B_download_time_l101_101781

theorem modem_B_download_time
    (time_A : ℝ) (speed_ratio : ℝ) 
    (h1 : time_A = 25.5) 
    (h2 : speed_ratio = 0.17) : 
    ∃ t : ℝ, t = 110.5425 := 
by
  sorry

end modem_B_download_time_l101_101781


namespace min_students_l101_101338

theorem min_students (b g : ℕ) 
  (h1 : 3 * b = 2 * g) 
  (h2 : (b + g) % 5 = 2) : 
  b + g = 57 :=
sorry

end min_students_l101_101338


namespace quadratic_decreasing_right_of_axis_of_symmetry_l101_101902

theorem quadratic_decreasing_right_of_axis_of_symmetry :
  ∀ x : ℝ, -2 * (x - 1)^2 < -2 * (x + 1 - 1)^2 →
  (∀ x' : ℝ, x' > 1 → -2 * (x' - 1)^2 < -2 * (x + 1 - 1)^2) :=
by
  sorry

end quadratic_decreasing_right_of_axis_of_symmetry_l101_101902


namespace seventh_fifth_tiles_difference_l101_101922

def side_length (n : ℕ) : ℕ := 2 * n - 1
def number_of_tiles (n : ℕ) : ℕ := (side_length n) ^ 2
def tiles_difference (n m : ℕ) : ℕ := number_of_tiles n - number_of_tiles m

theorem seventh_fifth_tiles_difference : tiles_difference 7 5 = 88 := by
  sorry

end seventh_fifth_tiles_difference_l101_101922


namespace probability_uniform_same_color_l101_101066

noncomputable def probability_same_color (choices : List String) (athleteA: ℕ) (athleteB: ℕ) : ℚ :=
  if choices.length = 3 ∧ athleteA ∈ [0,1,2] ∧ athleteB ∈ [0,1,2] then
    1 / 3
  else
    0

theorem probability_uniform_same_color :
  probability_same_color ["red", "white", "blue"] 0 1 = 1 / 3 :=
by
  sorry

end probability_uniform_same_color_l101_101066


namespace average_marks_mathematics_chemistry_l101_101717

theorem average_marks_mathematics_chemistry (M P C B : ℕ) 
    (h1 : M + P = 80) 
    (h2 : C + B = 120) 
    (h3 : C = P + 20) 
    (h4 : B = M - 15) : 
    (M + C) / 2 = 50 :=
by
  sorry

end average_marks_mathematics_chemistry_l101_101717


namespace largest_divisor_of_m_square_minus_n_square_l101_101551

theorem largest_divisor_of_m_square_minus_n_square (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) (h : n < m) :
  ∃ k : ℤ, k = 8 ∧ ∀ a b : ℤ, a % 2 = 1 → b % 2 = 1 → a > b → 8 ∣ (a^2 - b^2) := 
by
  sorry

end largest_divisor_of_m_square_minus_n_square_l101_101551


namespace max_value_of_f_l101_101749

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^2 + 4 * (Real.cos x)

theorem max_value_of_f : ∃ x : ℝ, f x ≤ 4 :=
sorry

end max_value_of_f_l101_101749


namespace arithmetic_sequences_sum_l101_101994

theorem arithmetic_sequences_sum
  (a b : ℕ → ℕ)
  (d1 d2 : ℕ)
  (h1 : ∀ n, a (n + 1) = a n + d1)
  (h2 : ∀ n, b (n + 1) = b n + d2)
  (h3 : a 1 + b 1 = 7)
  (h4 : a 3 + b 3 = 21) :
  a 5 + b 5 = 35 :=
sorry

end arithmetic_sequences_sum_l101_101994


namespace percent_of_a_is_b_l101_101193

variable (a b c : ℝ)
variable (h1 : c = 0.20 * a) (h2 : c = 0.10 * b)

theorem percent_of_a_is_b : b = 2 * a :=
by sorry

end percent_of_a_is_b_l101_101193


namespace sequence_general_formula_l101_101608

noncomputable def sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2^(n-2)

theorem sequence_general_formula {a : ℕ → ℝ} {S : ℕ → ℝ} (hpos : ∀ n, a n > 0)
  (hSn : ∀ n, 2 * a n = S n + 0.5) : ∀ n, a n = sequence_formula a S n :=
by 
  sorry

end sequence_general_formula_l101_101608


namespace problem_equivalence_l101_101779

theorem problem_equivalence (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (⌊(a^2 : ℚ) / b⌋ + ⌊(b^2 : ℚ) / a⌋ = ⌊(a^2 + b^2 : ℚ) / (a * b)⌋ + a * b) ↔
  (∃ k : ℕ, k > 0 ∧ ((a = k ∧ b = k^2 + 1) ∨ (a = k^2 + 1 ∧ b = k))) :=
sorry

end problem_equivalence_l101_101779


namespace klinker_twice_as_old_l101_101361

theorem klinker_twice_as_old :
  ∃ x : ℕ, (∀ (m k d : ℕ), m = 35 → d = 10 → m + x = 2 * (d + x)) → x = 15 :=
by
  sorry

end klinker_twice_as_old_l101_101361


namespace distance_to_place_l101_101825

-- Define the conditions
def speed_boat_standing_water : ℝ := 16
def speed_stream : ℝ := 2
def total_time_taken : ℝ := 891.4285714285714

-- Define the calculated speeds
def downstream_speed : ℝ := speed_boat_standing_water + speed_stream
def upstream_speed : ℝ := speed_boat_standing_water - speed_stream

-- Define the variable for the distance
variable (D : ℝ)

-- State the theorem to prove
theorem distance_to_place :
  D / downstream_speed + D / upstream_speed = total_time_taken →
  D = 7020 :=
by
  intro h
  sorry

end distance_to_place_l101_101825


namespace avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7_l101_101666

variable (c d : ℤ)
variable (h₁ : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7)

theorem avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7 :
  (d + 1 + d + 2 + d + 3 + d + 4 + d + 5 + d + 6 + d + 7) / 7 = c + 7 :=
by
  sorry

end avg_of_seven_consecutive_integers_starting_d_plus_1_is_c_plus_7_l101_101666


namespace value_of_a_l101_101636

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem value_of_a (a : ℝ) : f' (-1) a = 4 → a = 10 / 3 := by
  sorry

end value_of_a_l101_101636


namespace inequality_proof_l101_101481

noncomputable def f (x m : ℝ) : ℝ := 2 * m * x - Real.log x

theorem inequality_proof (m x₁ x₂ : ℝ) (hm : m ≥ -1) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hineq : (f x₁ m + f x₂ m) / 2 ≤ x₁ ^ 2 + x₂ ^ 2 + (3 / 2) * x₁ * x₂) :
  x₁ + x₂ ≥ (Real.sqrt 3 - 1) / 2 := 
sorry

end inequality_proof_l101_101481


namespace line_does_not_pass_through_third_quadrant_l101_101669

def line (x : ℝ) : ℝ := -x + 1

-- A line passes through the point (1, 0) and has a slope of -1
def passes_through_point (P : ℝ × ℝ) : Prop :=
  ∃ m b, m = -1 ∧ P.2 = m * P.1 + b ∧ line P.1 = P.2

def third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem line_does_not_pass_through_third_quadrant :
  ¬ ∃ p : ℝ × ℝ, passes_through_point p ∧ third_quadrant p :=
sorry

end line_does_not_pass_through_third_quadrant_l101_101669


namespace parabola_point_distance_eq_l101_101981

open Real

theorem parabola_point_distance_eq (P : ℝ × ℝ) (V : ℝ × ℝ) (F : ℝ × ℝ)
    (hV: V = (0, 0)) (hF : F = (0, 2)) (P_on_parabola : P.1 ^ 2 = 8 * P.2) 
    (hPf : dist P F = 150) (P_in_first_quadrant : 0 ≤ P.1 ∧ 0 ≤ P.2) :
    P = (sqrt 1184, 148) :=
sorry

end parabola_point_distance_eq_l101_101981


namespace grandmother_age_five_times_lingling_l101_101015

theorem grandmother_age_five_times_lingling (x : ℕ) :
  let lingling_age := 8
  let grandmother_age := 60
  (grandmother_age + x = 5 * (lingling_age + x)) ↔ (x = 5) := by
  sorry

end grandmother_age_five_times_lingling_l101_101015


namespace divisors_form_l101_101516

theorem divisors_form (p n : ℕ) (h_prime : Nat.Prime p) (h_pos : 0 < n) :
  ∃ k : ℕ, (p^n - 1 = 2^k - 1 ∨ p^n - 1 ∣ 48) :=
sorry

end divisors_form_l101_101516


namespace total_balls_l101_101334

theorem total_balls (blue red green yellow purple orange black white : ℕ) 
  (h1 : blue = 8)
  (h2 : red = 5)
  (h3 : green = 3 * (2 * blue - 1))
  (h4 : yellow = Nat.floor (2 * Real.sqrt (red * blue)))
  (h5 : purple = 4 * (blue + green))
  (h6 : orange = 7)
  (h7 : black + white = blue + red + green + yellow + purple + orange)
  (h8 : blue + red + green + yellow + purple + orange + black + white = 3 * (red + green + yellow + purple) + orange / 2)
  : blue + red + green + yellow + purple + orange + black + white = 829 :=
by
  sorry

end total_balls_l101_101334


namespace intersection_AB_l101_101056

def setA : Set ℝ := { x | x^2 - 2*x - 3 < 0}
def setB : Set ℝ := { x | x > 1 }
def intersection : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem intersection_AB : setA ∩ setB = intersection :=
by
  sorry

end intersection_AB_l101_101056


namespace eight_div_pow_64_l101_101024

theorem eight_div_pow_64 (h : 64 = 8^2) : 8^15 / 64^7 = 8 := by
  sorry

end eight_div_pow_64_l101_101024


namespace find_a_range_l101_101171

noncomputable def f (x a : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * a * x^2 + 2 * x + 1
def f' (x a : ℝ) : ℝ := x^2 - a * x + 2

theorem find_a_range (a : ℝ) :
  (0 < x1) ∧ (x1 < 1) ∧ (1 < x2) ∧ (x2 < 3) ∧
  (f' 0 a > 0) ∧ (f' 1 a < 0) ∧ (f' 3 a > 0) →
  3 < a ∧ a < 11 / 3 :=
by
  sorry

end find_a_range_l101_101171


namespace car_clock_time_correct_l101_101766

noncomputable def car_clock (t : ℝ) : ℝ := t * (4 / 3)

theorem car_clock_time_correct :
  ∀ t_real t_car,
  (car_clock 0 = 0) ∧
  (car_clock 0.5 = 2 / 3) ∧
  (car_clock t_real = t_car) ∧
  (t_car = (8 : ℝ)) → (t_real = 6) → (t_real + 1 = 7) :=
by
  intro t_real t_car h
  sorry

end car_clock_time_correct_l101_101766


namespace natural_numbers_pq_equal_l101_101085

theorem natural_numbers_pq_equal (p q : ℕ) (h : p^p + q^q = p^q + q^p) : p = q :=
sorry

end natural_numbers_pq_equal_l101_101085


namespace solution_inequality_part1_solution_inequality_part2_l101_101449

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x m : ℝ) : ℝ := -abs (x + 7) + 3 * m

theorem solution_inequality_part1 (x : ℝ) :
  (f x + x^2 - 4 > 0) ↔ (x > 2 ∨ x < -1) :=
sorry

theorem solution_inequality_part2 (m : ℝ) :
  (∃ x : ℝ, f x < g x m) → (m > 3) :=
sorry

end solution_inequality_part1_solution_inequality_part2_l101_101449


namespace cone_height_correct_l101_101935

noncomputable def cone_height (radius : ℝ) (central_angle : ℝ) : ℝ := 
  let base_radius := (central_angle * radius) / (2 * Real.pi)
  let height := Real.sqrt (radius ^ 2 - base_radius ^ 2)
  height

theorem cone_height_correct:
  cone_height 3 (2 * Real.pi / 3) = 2 * Real.sqrt 2 := 
by
  sorry

end cone_height_correct_l101_101935


namespace find_x_l101_101489

theorem find_x 
  (x y : ℤ) 
  (h1 : 2 * x - y = 5) 
  (h2 : x + 2 * y = 5) : 
  x = 3 := 
sorry

end find_x_l101_101489


namespace students_in_class_l101_101842

-- Define the relevant variables and conditions
variables (P H W T A S : ℕ)

-- Given conditions
axiom poetry_club : P = 22
axiom history_club : H = 27
axiom writing_club : W = 28
axiom two_clubs : T = 6
axiom all_clubs : A = 6

-- Statement to prove
theorem students_in_class
  (poetry_club : P = 22)
  (history_club : H = 27)
  (writing_club : W = 28)
  (two_clubs : T = 6)
  (all_clubs : A = 6) :
  S = P + H + W - T - 2 * A :=
sorry

end students_in_class_l101_101842


namespace average_salary_of_managers_l101_101536

theorem average_salary_of_managers (m_avg : ℝ) (assoc_avg : ℝ) (company_avg : ℝ)
  (managers : ℕ) (associates : ℕ) (total_employees : ℕ)
  (h_assoc_avg : assoc_avg = 30000) (h_company_avg : company_avg = 40000)
  (h_managers : managers = 15) (h_associates : associates = 75) (h_total_employees : total_employees = 90)
  (h_total_employees_def : total_employees = managers + associates)
  (h_total_salary_managers : ∀ m_avg, total_employees * company_avg = managers * m_avg + associates * assoc_avg) :
  m_avg = 90000 :=
by
  sorry

end average_salary_of_managers_l101_101536


namespace find_x_l101_101380

theorem find_x (x : ℝ) (h₁ : x > 0) (h₂ : x^4 = 390625) : x = 25 := 
by sorry

end find_x_l101_101380


namespace problem_trigonometric_identity_l101_101325

-- Define the problem conditions
theorem problem_trigonometric_identity
  (α : ℝ)
  (h : 3 * Real.sin (33 * Real.pi / 14 + α) = -5 * Real.cos (5 * Real.pi / 14 + α)) :
  Real.tan (5 * Real.pi / 14 + α) = -5 / 3 :=
sorry

end problem_trigonometric_identity_l101_101325


namespace martin_distance_l101_101830

def speed : ℝ := 12.0  -- Speed in miles per hour
def time : ℝ := 6.0    -- Time in hours

theorem martin_distance : (speed * time) = 72.0 :=
by
  sorry

end martin_distance_l101_101830


namespace value_of_x_l101_101664

theorem value_of_x (x : ℤ) : (x + 1) * (x + 1) = 16 ↔ (x = 3 ∨ x = -5) := 
by sorry

end value_of_x_l101_101664


namespace not_all_terms_positive_l101_101699

variable (a b c d : ℝ)
variable (e f g h : ℝ)

theorem not_all_terms_positive
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (he : e < 0) (hf : f < 0) (hg : g < 0) (hh : h < 0) :
  ¬ ((a * e + b * c > 0) ∧ (e * f + c * g > 0) ∧ (f * d + g * h > 0) ∧ (d * a + h * b > 0)) :=
sorry

end not_all_terms_positive_l101_101699


namespace new_apps_added_l101_101718

theorem new_apps_added (x : ℕ) (h1 : 15 + x - (x + 1) = 14) : x = 0 :=
by
  sorry

end new_apps_added_l101_101718


namespace five_times_seven_divided_by_ten_l101_101149

theorem five_times_seven_divided_by_ten : (5 * 7 : ℝ) / 10 = 3.5 := 
by 
  sorry

end five_times_seven_divided_by_ten_l101_101149


namespace sqrt_25_eq_pm_5_l101_101794

theorem sqrt_25_eq_pm_5 : {x : ℝ | x^2 = 25} = {5, -5} :=
by
  sorry

end sqrt_25_eq_pm_5_l101_101794


namespace bill_score_l101_101364

theorem bill_score (B J S E : ℕ)
                   (h1 : B = J + 20)
                   (h2 : B = S / 2)
                   (h3 : E = B + J - 10)
                   (h4 : B + J + S + E = 250) :
                   B = 50 := 
by sorry

end bill_score_l101_101364


namespace quadratic_root_condition_l101_101226

theorem quadratic_root_condition (a : ℝ) :
  (4 * Real.sqrt 2) = 3 * Real.sqrt (3 - 2 * a) → a = 1 / 2 :=
by
  sorry

end quadratic_root_condition_l101_101226


namespace quadratic_root_shift_l101_101244

theorem quadratic_root_shift (r s : ℝ)
    (hr : 2 * r^2 - 8 * r + 6 = 0)
    (hs : 2 * s^2 - 8 * s + 6 = 0)
    (h_sum_roots : r + s = 4)
    (h_prod_roots : r * s = 3)
    (b : ℝ) (c : ℝ)
    (h_b : b = - (r - 3 + s - 3))
    (h_c : c = (r - 3) * (s - 3)) : c = 0 :=
  by sorry

end quadratic_root_shift_l101_101244


namespace circle_area_ratio_l101_101803

/-- If the diameter of circle R is 60% of the diameter of circle S, 
the area of circle R is 36% of the area of circle S. -/
theorem circle_area_ratio (D_S D_R A_S A_R : ℝ) (h : D_R = 0.60 * D_S) 
  (hS : A_S = Real.pi * (D_S / 2) ^ 2) (hR : A_R = Real.pi * (D_R / 2) ^ 2): 
  A_R = 0.36 * A_S := 
sorry

end circle_area_ratio_l101_101803


namespace manolo_rate_change_after_one_hour_l101_101119

variable (masks_in_first_hour : ℕ)
variable (masks_in_remaining_time : ℕ)
variable (total_masks : ℕ)

-- Define conditions as Lean definitions
def first_hour_rate := 1 / 4  -- masks per minute
def remaining_time_rate := 1 / 6  -- masks per minute
def total_time := 4  -- hours
def masks_produced_in_first_hour (t : ℕ) := t * 15  -- t hours, 60 minutes/hour, at 15 masks/hour
def masks_produced_in_remaining_time (t : ℕ) := t * 10 -- (total_time - 1) hours, 60 minutes/hour, at 10 masks/hour

-- Main proof problem statement
theorem manolo_rate_change_after_one_hour :
  masks_in_first_hour = masks_produced_in_first_hour 1 →
  masks_in_remaining_time = masks_produced_in_remaining_time (total_time - 1) →
  total_masks = masks_in_first_hour + masks_in_remaining_time →
  (∃ t : ℕ, t = 1) :=
by
  -- Placeholder, proof not required
  sorry

end manolo_rate_change_after_one_hour_l101_101119


namespace initial_yellow_hard_hats_count_l101_101760

noncomputable def initial_yellow_hard_hats := 24

theorem initial_yellow_hard_hats_count
  (initial_pink: ℕ)
  (initial_green: ℕ)
  (carl_pink: ℕ)
  (john_pink: ℕ)
  (john_green: ℕ)
  (total_remaining: ℕ)
  (remaining_pink: ℕ)
  (remaining_green: ℕ)
  (initial_yellow: ℕ) :
  initial_pink = 26 →
  initial_green = 15 →
  carl_pink = 4 →
  john_pink = 6 →
  john_green = 2 * john_pink →
  total_remaining = 43 →
  remaining_pink = initial_pink - carl_pink - john_pink →
  remaining_green = initial_green - john_green →
  initial_yellow = total_remaining - remaining_pink - remaining_green →
  initial_yellow = initial_yellow_hard_hats :=
by
  intros
  sorry

end initial_yellow_hard_hats_count_l101_101760


namespace ott_fraction_l101_101742

/-- 
Moe, Loki, Nick, and Pat each give $2 to Ott.
Moe gave Ott one-seventh of his money.
Loki gave Ott one-fifth of his money.
Nick gave Ott one-fourth of his money.
Pat gave Ott one-sixth of his money.
-/
def fraction_of_money_ott_now_has (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) : Prop :=
  A = 14 ∧ B = 10 ∧ C = 8 ∧ D = 12 ∧ (2 * (1 / 7 : ℚ)) = 2 ∧ (2 * (1 / 5 : ℚ)) = 2 ∧ (2 * (1 / 4 : ℚ)) = 2 ∧ (2 * (1 / 6 : ℚ)) = 2

theorem ott_fraction (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) (h : fraction_of_money_ott_now_has A B C D) : 
  8 = (2 / 11 : ℚ) * (A + B + C + D) :=
by sorry

end ott_fraction_l101_101742


namespace no_solutions_xyz_l101_101868

theorem no_solutions_xyz :
  ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 4 :=
by
  sorry

end no_solutions_xyz_l101_101868


namespace percentage_difference_j_p_l101_101675

theorem percentage_difference_j_p (j p t : ℝ) (h1 : j = t * 80 / 100) 
  (h2 : t = p * (100 - t) / 100) (h3 : t = 6.25) : 
  ((p - j) / p) * 100 = 25 := 
by
  sorry

end percentage_difference_j_p_l101_101675


namespace cost_of_history_book_l101_101241

theorem cost_of_history_book (total_books : ℕ) (cost_math_book : ℕ) (total_price : ℕ) (num_math_books : ℕ) (num_history_books : ℕ) (cost_history_book : ℕ) 
    (h_books_total : total_books = 90)
    (h_cost_math : cost_math_book = 4)
    (h_total_price : total_price = 396)
    (h_num_math_books : num_math_books = 54)
    (h_num_total_books : num_math_books + num_history_books = total_books)
    (h_total_cost : num_math_books * cost_math_book + num_history_books * cost_history_book = total_price) : cost_history_book = 5 := by 
  sorry

end cost_of_history_book_l101_101241


namespace electric_energy_consumption_l101_101152

def power_rating_fan : ℕ := 75
def hours_per_day : ℕ := 8
def days_per_month : ℕ := 30
def watts_to_kWh : ℕ := 1000

theorem electric_energy_consumption : power_rating_fan * hours_per_day * days_per_month / watts_to_kWh = 18 := by
  sorry

end electric_energy_consumption_l101_101152


namespace citizen_income_l101_101533

theorem citizen_income (total_tax : ℝ) (income : ℝ) :
  total_tax = 15000 →
  (income ≤ 20000 → total_tax = income * 0.10) ∧
  (20000 < income ∧ income ≤ 50000 → total_tax = (20000 * 0.10) + ((income - 20000) * 0.15)) ∧
  (50000 < income ∧ income ≤ 90000 → total_tax = (20000 * 0.10) + (30000 * 0.15) + ((income - 50000) * 0.20)) ∧
  (income > 90000 → total_tax = (20000 * 0.10) + (30000 * 0.15) + (40000 * 0.20) + ((income - 90000) * 0.25)) →
  income = 92000 :=
by
  sorry

end citizen_income_l101_101533


namespace inequality_px_qy_l101_101068

theorem inequality_px_qy 
  (p q x y : ℝ) 
  (hp : 0 < p) 
  (hq : 0 < q) 
  (hpq : p + q < 1) 
  : (p * x + q * y) ^ 2 ≤ p * x ^ 2 + q * y ^ 2 := 
sorry

end inequality_px_qy_l101_101068


namespace candidate_percentage_valid_votes_l101_101458

theorem candidate_percentage_valid_votes (total_votes invalid_percentage valid_votes_received : ℕ) 
    (h_total_votes : total_votes = 560000)
    (h_invalid_percentage : invalid_percentage = 15)
    (h_valid_votes_received : valid_votes_received = 333200) :
    (valid_votes_received : ℚ) / (total_votes * (1 - invalid_percentage / 100) : ℚ) * 100 = 70 :=
by
  sorry

end candidate_percentage_valid_votes_l101_101458


namespace secret_sharing_problem_l101_101163

theorem secret_sharing_problem : 
  ∃ n : ℕ, (3280 = (3^(n + 1) - 1) / 2) ∧ (n = 7) :=
by
  use 7
  sorry

end secret_sharing_problem_l101_101163


namespace julia_download_songs_l101_101365

-- Basic definitions based on conditions
def internet_speed_MBps : ℕ := 20
def song_size_MB : ℕ := 5
def half_hour_seconds : ℕ := 30 * 60

-- Statement of the proof problem
theorem julia_download_songs : 
  (internet_speed_MBps * half_hour_seconds) / song_size_MB = 7200 :=
by
  sorry

end julia_download_songs_l101_101365


namespace exists_linear_eq_exactly_m_solutions_l101_101471

theorem exists_linear_eq_exactly_m_solutions (m : ℕ) (hm : 0 < m) :
  ∃ (a b c : ℤ), ∀ (x y : ℕ), a * x + b * y = c ↔
    (1 ≤ x ∧ 1 ≤ y ∧ x + y = m + 1) :=
by
  sorry

end exists_linear_eq_exactly_m_solutions_l101_101471


namespace max_x2_y2_z4_l101_101986

theorem max_x2_y2_z4 (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1) :
  x^2 + y^2 + z^4 ≤ 1 :=
sorry

end max_x2_y2_z4_l101_101986


namespace product_of_terms_l101_101721

variable {α : Type*} [LinearOrderedField α]

namespace GeometricSequence

def is_geometric_sequence (a : ℕ → α) :=
  ∃ r : α, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_terms (a : ℕ → α) (r : α) (h_geo : is_geometric_sequence a) :
  (a 4) * (a 8) = 16 → (a 2) * (a 10) = 16 :=
by
  intro h1
  sorry

end GeometricSequence

end product_of_terms_l101_101721


namespace flea_reach_B_with_7_jumps_flea_reach_C_with_9_jumps_flea_cannot_reach_D_with_2028_jumps_l101_101780

-- Problem (a)
theorem flea_reach_B_with_7_jumps (A B : ℤ) (jumps : ℤ) (distance : ℤ) (ways : ℕ) :
  B = A + 5 → jumps = 7 → distance = 5 → 
  ways = Nat.choose (7) (1) := 
sorry

-- Problem (b)
theorem flea_reach_C_with_9_jumps (A C : ℤ) (jumps : ℤ) (distance : ℤ) (ways : ℕ) :
  C = A + 5 → jumps = 9 → distance = 5 → 
  ways = Nat.choose (9) (2) :=
sorry

-- Problem (c)
theorem flea_cannot_reach_D_with_2028_jumps (A D : ℤ) (jumps : ℤ) (distance : ℤ) :
  D = A + 2013 → jumps = 2028 → distance = 2013 → 
  ∃ x y : ℤ, x + y = 2028 ∧ x - y = 2013 → false :=
sorry

end flea_reach_B_with_7_jumps_flea_reach_C_with_9_jumps_flea_cannot_reach_D_with_2028_jumps_l101_101780


namespace laura_owes_amount_l101_101919

noncomputable def calculate_amount_owed (P R T : ℝ) : ℝ :=
  let I := P * R * T 
  P + I

theorem laura_owes_amount (P : ℝ) (R : ℝ) (T : ℝ) (hP : P = 35) (hR : R = 0.09) (hT : T = 1) :
  calculate_amount_owed P R T = 38.15 := by
  -- Prove that the total amount owed calculated by the formula matches the correct answer
  sorry

end laura_owes_amount_l101_101919


namespace sum_a_m_eq_2_pow_n_b_n_l101_101411

noncomputable def a_n (x : ℝ) (n : ℕ) : ℝ := (Finset.range (n + 1)).sum (λ k => x ^ k)

noncomputable def b_n (x : ℝ) (n : ℕ) : ℝ := 
  (Finset.range (n + 1)).sum (λ k => ((x + 1) / 2) ^ k)

theorem sum_a_m_eq_2_pow_n_b_n 
  (x : ℝ) (n : ℕ) : 
  (Finset.range (n + 1)).sum (λ m => a_n x m * Nat.choose (n + 1) (m + 1)) = 2 ^ n * b_n x n :=
by
  sorry

end sum_a_m_eq_2_pow_n_b_n_l101_101411


namespace harry_sandy_midpoint_l101_101139

theorem harry_sandy_midpoint :
  ∃ (x y : ℤ), x = 9 ∧ y = -2 → ∃ (a b : ℤ), a = 1 ∧ b = 6 → ((9 + 1) / 2, (-2 + 6) / 2) = (5, 2) := 
by 
  sorry

end harry_sandy_midpoint_l101_101139


namespace problem1_problem2_l101_101311

theorem problem1 (a b : ℝ) :
  5 * a * b^2 - 2 * a^2 * b + 3 * a * b^2 - a^2 * b - 4 * a * b^2 = 4 * a * b^2 - 3 * a^2 * b := 
by sorry

theorem problem2 (m n : ℝ) :
  -5 * m * n^2 - (2 * m^2 * n - 2 * (m^2 * n - 2 * m * n^2)) = -9 * m * n^2 := 
by sorry

end problem1_problem2_l101_101311


namespace quadratic_root_value_of_b_l101_101031

theorem quadratic_root_value_of_b :
  (∃ r1 r2 : ℝ, 2 * r1^2 + b * r1 - 20 = 0 ∧ r1 = -5 ∧ r1 * r2 = -10 ∧ r1 + r2 = -b / 2) → b = 6 :=
by
  intro h
  obtain ⟨r1, r2, h_eq1, h_r1, h_prod, h_sum⟩ := h
  sorry

end quadratic_root_value_of_b_l101_101031


namespace find_x_value_l101_101235

-- Define vectors a and b
def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Define the condition that a + b is parallel to 2a - b
def parallel_vectors (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (2 * a.1 - b.1) = k * (a.1 + b.1) ∧ (2 * a.2 - b.2) = k * (a.2 + b.2)

-- Problem statement: Prove that x = -4
theorem find_x_value : ∀ (x : ℝ),
  parallel_vectors vector_a (vector_b x) → x = -4 :=
by
  sorry

end find_x_value_l101_101235


namespace arithmetic_sequence_a2_value_l101_101417

theorem arithmetic_sequence_a2_value :
  ∃ (a : ℕ) (d : ℕ), (a = 3) ∧ (a + d + (a + 2 * d) = 12) ∧ (a + d = 5) :=
by
  sorry

end arithmetic_sequence_a2_value_l101_101417


namespace area_of_sine_curve_l101_101601

theorem area_of_sine_curve :
  let f := (fun x => Real.sin x)
  let a := -Real.pi
  let b := 2 * Real.pi
  (∫ x in a..b, f x) = 6 :=
by
  sorry

end area_of_sine_curve_l101_101601


namespace max_collection_l101_101140

theorem max_collection : 
  let Yoongi := 4 
  let Jungkook := 6 / 3 
  let Yuna := 5 
  max Yoongi (max Jungkook Yuna) = 5 :=
by 
  let Yoongi := 4
  let Jungkook := (6 / 3) 
  let Yuna := 5
  show max Yoongi (max Jungkook Yuna) = 5
  sorry

end max_collection_l101_101140


namespace cookies_with_new_flour_l101_101976

-- Definitions for the conditions
def cookies_per_cup (total_cookies : ℕ) (total_flour : ℕ) : ℕ :=
  total_cookies / total_flour

noncomputable def cookies_from_flour (cookies_per_cup : ℕ) (flour : ℕ) : ℕ :=
  cookies_per_cup * flour

-- Given data
def total_cookies := 24
def total_flour := 4
def new_flour := 3

-- Theorem (problem statement)
theorem cookies_with_new_flour : cookies_from_flour (cookies_per_cup total_cookies total_flour) new_flour = 18 :=
by
  sorry

end cookies_with_new_flour_l101_101976


namespace find_m_l101_101374

-- Define the vectors a and b and the condition for parallelicity
def a : ℝ × ℝ := (2, 1)
def b (m : ℝ) : ℝ × ℝ := (m, 2)
def parallel (u v : ℝ × ℝ) := u.1 * v.2 = u.2 * v.1

-- State the theorem with the given conditions and required proof goal
theorem find_m (m : ℝ) (h : parallel a (b m)) : m = 4 :=
by sorry  -- skipping proof

end find_m_l101_101374


namespace compatible_polynomial_count_l101_101818

theorem compatible_polynomial_count (n : ℕ) : 
  ∃ num_polynomials : ℕ, num_polynomials = (n / 2) + 1 :=
by
  sorry

end compatible_polynomial_count_l101_101818


namespace equation_of_line_l101_101867

theorem equation_of_line {x y : ℝ} (b : ℝ) (h1 : ∀ x y, (3 * x + 4 * y - 7 = 0) → (y = -3/4 * x))
  (h2 : (1 / 2) * |b| * |(4 / 3) * b| = 24) : 
  ∃ b : ℝ, ∀ x, y = -3/4 * x + b := 
sorry

end equation_of_line_l101_101867


namespace initial_number_of_macaroons_l101_101446

theorem initial_number_of_macaroons 
  (w : ℕ) (bag_count : ℕ) (eaten_bag_count : ℕ) (remaining_weight : ℕ) 
  (macaroon_weight : ℕ) (remaining_bags : ℕ) (initial_macaroons : ℕ) :
  w = 5 → bag_count = 4 → eaten_bag_count = 1 → remaining_weight = 45 → 
  macaroon_weight = w → remaining_bags = (bag_count - eaten_bag_count) → 
  initial_macaroons = (remaining_bags * remaining_weight / macaroon_weight) * bag_count / remaining_bags →
  initial_macaroons = 12 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end initial_number_of_macaroons_l101_101446


namespace ratio_is_l101_101911

noncomputable def volume_dodecahedron (s : ℝ) : ℝ := (15 + 7 * Real.sqrt 5) / 4 * s ^ 3

noncomputable def volume_tetrahedron (s : ℝ) : ℝ := Real.sqrt 2 / 12 * ((Real.sqrt 3 / 2) * s) ^ 3

noncomputable def ratio_volumes (s : ℝ) : ℝ := volume_dodecahedron s / volume_tetrahedron s

theorem ratio_is (s : ℝ) : ratio_volumes s = (60 + 28 * Real.sqrt 5) / Real.sqrt 6 :=
by
  sorry

end ratio_is_l101_101911


namespace merchant_printer_count_l101_101819

theorem merchant_printer_count (P : ℕ) 
  (cost_keyboards : 15 * 20 = 300)
  (total_cost : 300 + 70 * P = 2050) :
  P = 25 := 
by
  sorry

end merchant_printer_count_l101_101819


namespace scientific_notation_of_0_00003_l101_101696

theorem scientific_notation_of_0_00003 :
  0.00003 = 3 * 10^(-5) :=
sorry

end scientific_notation_of_0_00003_l101_101696


namespace find_distance_AB_l101_101208

variable (vA vB : ℝ) -- speeds of Person A and Person B
variable (x : ℝ) -- distance between points A and B
variable (t1 t2 : ℝ) -- time variables

-- Conditions
def startTime := 0
def meetDistanceBC := 240
def returnPointBDistantFromA := 120
def doublingSpeedFactor := 2

-- Main questions and conditions
theorem find_distance_AB
  (h1 : vA > vB)
  (h2 : t1 = x / vB)
  (h3 : t2 = 2 * (x - meetDistanceBC) / vA) 
  (h4 : x = meetDistanceBC + returnPointBDistantFromA + (t1 * (doublingSpeedFactor * vB) - t2 * vA) / (doublingSpeedFactor - 1)) :
  x = 420 :=
sorry

end find_distance_AB_l101_101208


namespace sum_prime_factors_1170_l101_101386

theorem sum_prime_factors_1170 : 
  let smallest_prime_factor := 2
  let largest_prime_factor := 13
  (smallest_prime_factor + largest_prime_factor) = 15 :=
by
  sorry

end sum_prime_factors_1170_l101_101386


namespace percentage_of_men_l101_101195

theorem percentage_of_men (M : ℝ) 
  (h1 : 0 < M ∧ M < 1) 
  (h2 : 0.2 * M + 0.4 * (1 - M) = 0.3) : M = 0.5 :=
by
  sorry

end percentage_of_men_l101_101195


namespace find_x_l101_101581

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end find_x_l101_101581


namespace solve_congruence_l101_101230

-- Define the condition and residue modulo 47
def residue_modulo (a b n : ℕ) : Prop := (a ≡ b [MOD n])

-- The main theorem to be proved
theorem solve_congruence (m : ℕ) (h : residue_modulo (13 * m) 9 47) : residue_modulo m 26 47 :=
sorry

end solve_congruence_l101_101230


namespace part_time_employees_l101_101356

theorem part_time_employees (total_employees : ℕ) (full_time_employees : ℕ) (h1 : total_employees = 65134) (h2 : full_time_employees = 63093) :
  total_employees - full_time_employees = 2041 :=
by
  -- Suppose that total_employees - full_time_employees = 2041
  sorry

end part_time_employees_l101_101356


namespace total_students_l101_101508

theorem total_students (N : ℕ) (h1 : ∃ g1 g2 g3 g4 g5 g6 : ℕ, 
  g1 = 13 ∧ g2 = 13 ∧ g3 = 13 ∧ g4 = 13 ∧ 
  ((g5 = 12 ∧ g6 = 12) ∨ (g5 = 14 ∧ g6 = 14)) ∧ 
  N = g1 + g2 + g3 + g4 + g5 + g6) : 
  N = 76 ∨ N = 80 :=
by
  sorry

end total_students_l101_101508


namespace minimum_omega_l101_101768

theorem minimum_omega 
  (ω : ℝ)
  (hω : ω > 0)
  (h_shift : ∃ T > 0, T = 2 * π / ω ∧ T = 2 * π / 3) : 
  ω = 3 := 
sorry

end minimum_omega_l101_101768


namespace least_integer_l101_101822

theorem least_integer (x p d n : ℕ) (hx : x = 10^p * d + n) (hn : n = 10^p * d / 18) : x = 950 :=
by sorry

end least_integer_l101_101822


namespace smallest_sum_of_squares_l101_101322

theorem smallest_sum_of_squares :
  ∃ (x y : ℤ), x^2 - y^2 = 175 ∧ x^2 ≥ 36 ∧ y^2 ≥ 36 ∧ x^2 + y^2 = 625 :=
by
  sorry

end smallest_sum_of_squares_l101_101322


namespace min_value_inequality_l101_101688

theorem min_value_inequality (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 3 * b = 1) : 
  (2 / a + 3 / b) ≥ 14 :=
sorry

end min_value_inequality_l101_101688


namespace storage_house_blocks_needed_l101_101686

noncomputable def volume_of_storage_house
  (L_o : ℕ) (W_o : ℕ) (H_o : ℕ) (T : ℕ) : ℕ :=
  let interior_length := L_o - 2 * T
  let interior_width := W_o - 2 * T
  let interior_height := H_o - T
  let outer_volume := L_o * W_o * H_o
  let interior_volume := interior_length * interior_width * interior_height
  outer_volume - interior_volume

theorem storage_house_blocks_needed :
  volume_of_storage_house 15 12 8 2 = 912 :=
  by
    sorry

end storage_house_blocks_needed_l101_101686


namespace vector_addition_dot_product_l101_101692

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (3, 1)

theorem vector_addition :
  let c := (1, 2) + (3, 1)
  c = (4, 3) := by
  sorry

theorem dot_product :
  let d := (1 * 3 + 2 * 1)
  d = 5 := by
  sorry

end vector_addition_dot_product_l101_101692


namespace min_value_inequality_l101_101484

variable (x y : ℝ)

theorem min_value_inequality (h₀ : x > 0) (h₁ : y > 0) (h₂ : x + y = 1) : 
  ∃ m : ℝ, m = 1 / 4 ∧ (∀ x y, x > 0 → y > 0 → x + y = 1 → (x ^ 2) / (x + 2) + (y ^ 2) / (y + 1) ≥ m) :=
by
  use (1 / 4)
  sorry

end min_value_inequality_l101_101484


namespace count_total_kids_in_lawrence_l101_101548

namespace LawrenceCountyKids

/-- Number of kids who went to camp from Lawrence county -/
def kids_went_to_camp : ℕ := 610769

/-- Number of kids who stayed home -/
def kids_stayed_home : ℕ := 590796

/-- Total number of kids in Lawrence county -/
def total_kids_in_county : ℕ := 1201565

/-- Proof statement -/
theorem count_total_kids_in_lawrence :
  kids_went_to_camp + kids_stayed_home = total_kids_in_county :=
sorry

end LawrenceCountyKids

end count_total_kids_in_lawrence_l101_101548


namespace lucas_pay_per_window_l101_101320

-- Conditions
def num_floors : Nat := 3
def windows_per_floor : Nat := 3
def days_to_finish : Nat := 6
def penalty_rate : Nat := 3
def penalty_amount : Nat := 1
def final_payment : Nat := 16

-- Theorem statement
theorem lucas_pay_per_window :
  let total_windows := num_floors * windows_per_floor
  let total_penalty := penalty_amount * (days_to_finish / penalty_rate)
  let original_payment := final_payment + total_penalty
  let payment_per_window := original_payment / total_windows
  payment_per_window = 2 :=
by
  sorry

end lucas_pay_per_window_l101_101320


namespace average_runs_l101_101416

/-- The average runs scored by the batsman in the first 20 matches is 40,
and in the next 10 matches is 30. We want to prove the average runs scored
by the batsman in all 30 matches is 36.67. --/
theorem average_runs (avg20 avg10 : ℕ) (num_matches_20 num_matches_10 : ℕ)
  (h1 : avg20 = 40) (h2 : avg10 = 30) (h3 : num_matches_20 = 20) (h4 : num_matches_10 = 10) :
  ((num_matches_20 * avg20 + num_matches_10 * avg10 : ℕ) : ℚ) / (num_matches_20 + num_matches_10 : ℕ) = 36.67 := by
  sorry

end average_runs_l101_101416


namespace boys_without_pencils_l101_101222

variable (total_students : ℕ) (total_boys : ℕ) (students_with_pencils : ℕ) (girls_with_pencils : ℕ)

theorem boys_without_pencils
  (h1 : total_boys = 18)
  (h2 : students_with_pencils = 25)
  (h3 : girls_with_pencils = 15)
  (h4 : total_students = 30) :
  total_boys - (students_with_pencils - girls_with_pencils) = 8 :=
by
  sorry

end boys_without_pencils_l101_101222


namespace length_gh_parallel_lines_l101_101846

theorem length_gh_parallel_lines (
    AB CD EF GH : ℝ
) (
    h1 : AB = 300
) (
    h2 : CD = 200
) (
    h3 : EF = (AB + CD) / 2 * (1 / 2)
) (
    h4 : GH = EF * (1 - 1 / 4)
) :
    GH = 93.75 :=
by
    sorry

end length_gh_parallel_lines_l101_101846


namespace Pam_current_balance_l101_101052

-- Given conditions as definitions
def initial_balance : ℕ := 400
def tripled_balance : ℕ := 3 * initial_balance
def current_balance : ℕ := tripled_balance - 250

-- The theorem to be proved
theorem Pam_current_balance : current_balance = 950 := by
  sorry

end Pam_current_balance_l101_101052


namespace bucket_capacity_l101_101327

theorem bucket_capacity :
  (∃ (x : ℝ), 30 * x = 45 * 9) → 13.5 = 13.5 :=
by
  -- proof needed
  sorry

end bucket_capacity_l101_101327


namespace initial_percentage_acidic_liquid_l101_101634

theorem initial_percentage_acidic_liquid (P : ℝ) :
  let initial_volume := 12
  let removed_volume := 4
  let final_volume := initial_volume - removed_volume
  let desired_concentration := 60
  (P/100) * initial_volume = (desired_concentration/100) * final_volume →
  P = 40 :=
by
  intros
  sorry

end initial_percentage_acidic_liquid_l101_101634


namespace surface_area_of_box_l101_101528

variable {l w h : ℝ}

def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * h + w * h + l * w)

theorem surface_area_of_box (l w h : ℝ) : surfaceArea l w h = 2 * (l * h + w * h + l * w) :=
by
  sorry

end surface_area_of_box_l101_101528


namespace sum_of_center_coordinates_l101_101367

theorem sum_of_center_coordinates 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : (x1, y1) = (4, 3)) 
  (h2 : (x2, y2) = (-6, 5)) : 
  (x1 + x2) / 2 + (y1 + y2) / 2 = 3 := by
  sorry

end sum_of_center_coordinates_l101_101367


namespace crescent_perimeter_l101_101845

def radius_outer : ℝ := 10.5
def radius_inner : ℝ := 6.7

theorem crescent_perimeter : (radius_outer + radius_inner) * Real.pi = 54.037 :=
by
  sorry

end crescent_perimeter_l101_101845


namespace letter_puzzle_solutions_l101_101499

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def to_number (B A : ℕ) : ℕ :=
  10 * B + A

theorem letter_puzzle_solutions (A B : ℕ) (h_diff : A ≠ B) (h_digits : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9)
  (h_two_digit : is_two_digit (to_number B A)) :
  A^B = to_number B A ↔ (A = 2 ∧ B = 5 ∨ A = 6 ∧ B = 2 ∨ A = 4 ∧ B = 3) :=
by
  sorry

end letter_puzzle_solutions_l101_101499


namespace range_of_m_l101_101947

theorem range_of_m (m : ℝ) (h : (2 - m) * (|m| - 3) < 0) : (-3 < m ∧ m < 2) ∨ (m > 3) :=
sorry

end range_of_m_l101_101947


namespace katie_needs_more_sugar_l101_101967

-- Let total_cups be the total cups of sugar required according to the recipe
def total_cups : ℝ := 3

-- Let already_put_in be the cups of sugar Katie has already put in
def already_put_in : ℝ := 0.5

-- Define the amount of sugar Katie still needs to put in
def remaining_cups : ℝ := total_cups - already_put_in 

-- Prove that remaining_cups is 2.5
theorem katie_needs_more_sugar : remaining_cups = 2.5 := 
by 
  -- substitute total_cups and already_put_in
  dsimp [remaining_cups, total_cups, already_put_in]
  -- calculate the difference
  norm_num

end katie_needs_more_sugar_l101_101967


namespace linear_function_increasing_l101_101435

variable (x1 x2 : ℝ)
variable (y1 y2 : ℝ)
variable (hx : x1 < x2)
variable (P1_eq : y1 = 2 * x1 + 1)
variable (P2_eq : y2 = 2 * x2 + 1)

theorem linear_function_increasing (hx : x1 < x2) (P1_eq : y1 = 2 * x1 + 1) (P2_eq : y2 = 2 * x2 + 1) 
    : y1 < y2 := sorry

end linear_function_increasing_l101_101435


namespace reeyas_first_subject_score_l101_101433

theorem reeyas_first_subject_score
  (second_subject_score third_subject_score fourth_subject_score : ℕ)
  (num_subjects : ℕ)
  (average_score : ℕ)
  (total_subjects_score : ℕ)
  (condition1 : second_subject_score = 76)
  (condition2 : third_subject_score = 82)
  (condition3 : fourth_subject_score = 85)
  (condition4 : num_subjects = 4)
  (condition5 : average_score = 75)
  (condition6 : total_subjects_score = num_subjects * average_score) :
  67 = total_subjects_score - (second_subject_score + third_subject_score + fourth_subject_score) := 
  sorry

end reeyas_first_subject_score_l101_101433


namespace students_called_back_l101_101366

theorem students_called_back (girls boys not_called_back called_back : ℕ) 
  (h1 : girls = 17)
  (h2 : boys = 32)
  (h3 : not_called_back = 39)
  (h4 : called_back = (girls + boys) - not_called_back):
  called_back = 10 := by
  sorry

end students_called_back_l101_101366


namespace green_apples_ordered_l101_101137

-- Definitions based on the conditions
variable (red_apples : Nat := 25)
variable (students : Nat := 10)
variable (extra_apples : Nat := 32)
variable (G : Nat)

-- The mathematical problem to prove
theorem green_apples_ordered :
  red_apples + G - students = extra_apples → G = 17 := by
  sorry

end green_apples_ordered_l101_101137


namespace eight_people_lineup_two_windows_l101_101252

theorem eight_people_lineup_two_windows :
  (2 ^ 8) * (Nat.factorial 8) = 10321920 := by
  sorry

end eight_people_lineup_two_windows_l101_101252


namespace solve_for_x_l101_101231

theorem solve_for_x :
  ∀ x : ℤ, 3 * x + 36 = 48 → x = 4 :=
by
  sorry

end solve_for_x_l101_101231


namespace function_is_constant_and_straight_line_l101_101277

-- Define a function f : ℝ → ℝ
variable (f : ℝ → ℝ)

-- Condition: The derivative of f is 0 everywhere
axiom derivative_zero_everywhere : ∀ x, deriv f x = 0

-- Conclusion: f is a constant function
theorem function_is_constant_and_straight_line : ∃ C : ℝ, ∀ x, f x = C := by
  sorry

end function_is_constant_and_straight_line_l101_101277


namespace quadratic_distinct_roots_l101_101558

noncomputable def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem quadratic_distinct_roots (k : ℝ) :
  (k ≠ 0) ∧ (1 > k) ↔ has_two_distinct_real_roots k (-6) 9 :=
by
  sorry

end quadratic_distinct_roots_l101_101558


namespace five_star_three_eq_ten_l101_101898

def operation (a b : ℝ) : ℝ := b^2 + 1

theorem five_star_three_eq_ten : operation 5 3 = 10 := by
  sorry

end five_star_three_eq_ten_l101_101898


namespace area_of_rectangular_garden_l101_101777

theorem area_of_rectangular_garden (length width : ℝ) (h_length : length = 2.5) (h_width : width = 0.48) :
  length * width = 1.2 :=
by
  sorry

end area_of_rectangular_garden_l101_101777


namespace total_customers_in_line_l101_101042

-- Definition of the number of people standing in front of the last person
def num_people_in_front : Nat := 8

-- Definition of the last person in the line
def last_person : Nat := 1

-- Statement to prove
theorem total_customers_in_line : num_people_in_front + last_person = 9 := by
  sorry

end total_customers_in_line_l101_101042


namespace average_salary_of_feb_mar_apr_may_l101_101743

theorem average_salary_of_feb_mar_apr_may
  (avg_salary_jan_feb_mar_apr : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_salary_feb_mar_apr : ℝ)
  (total_salary_feb_mar_apr_may: ℝ)
  (n_months: ℝ): 
  avg_salary_jan_feb_mar_apr = 8000 ∧ 
  salary_jan = 6100 ∧ 
  salary_may = 6500 ∧ 
  total_salary_feb_mar_apr = (avg_salary_jan_feb_mar_apr * 4 - salary_jan) ∧
  total_salary_feb_mar_apr_may = (total_salary_feb_mar_apr + salary_may) ∧
  n_months = (total_salary_feb_mar_apr_may / 8100) →
  n_months = 4 :=
by
  intros 
  sorry

end average_salary_of_feb_mar_apr_may_l101_101743


namespace second_batch_jelly_beans_weight_l101_101973

theorem second_batch_jelly_beans_weight (J : ℝ) (h1 : 2 * 3 + J > 0) (h2 : (6 + J) * 2 = 16) : J = 2 :=
sorry

end second_batch_jelly_beans_weight_l101_101973


namespace range_of_a_l101_101950

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 1 ∧ 2 * a * x + 4 = 0) ↔ (-2 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l101_101950


namespace shoes_count_l101_101939

def numberOfShoes (numPairs : Nat) (matchingPairProbability : ℚ) : Nat :=
  let S := numPairs * 2
  if (matchingPairProbability = 1 / (S - 1))
  then S
  else 0

theorem shoes_count 
(numPairs : Nat)
(matchingPairProbability : ℚ)
(hp : numPairs = 9)
(hq : matchingPairProbability = 0.058823529411764705) :
numberOfShoes numPairs matchingPairProbability = 18 := 
by
  -- definition only, the proof is not required
  sorry

end shoes_count_l101_101939


namespace log_40_cannot_be_directly_calculated_l101_101875

theorem log_40_cannot_be_directly_calculated (log_3 log_5 : ℝ) (h1 : log_3 = 0.4771) (h2 : log_5 = 0.6990) : 
  ¬ (exists (log_40 : ℝ), (log_40 = (log_3 + log_5) + log_40)) :=
by {
  sorry
}

end log_40_cannot_be_directly_calculated_l101_101875


namespace find_x_l101_101956

theorem find_x (y : ℝ) (x : ℝ) (h : x / (x - 1) = (y^2 + 2 * y + 3) / (y^2 + 2 * y - 2)) :
  x = (y^2 + 2 * y + 3) / 5 := by
  sorry

end find_x_l101_101956


namespace function_inverse_l101_101329

theorem function_inverse (x : ℝ) (h : ℝ → ℝ) (k : ℝ → ℝ) 
  (h_def : ∀ x, h x = 6 - 7 * x) 
  (k_def : ∀ x, k x = (6 - x) / 7) : 
  h (k x) = x ∧ k (h x) = x := 
  sorry

end function_inverse_l101_101329


namespace bean_lands_outside_inscribed_circle_l101_101617

theorem bean_lands_outside_inscribed_circle :
  let a := 8
  let b := 15
  let c := 17  -- hypotenuse computed as sqrt(a^2 + b^2)
  let area_triangle := (1 / 2) * a * b
  let s := (a + b + c) / 2  -- semiperimeter
  let r := area_triangle / s -- radius of the inscribed circle
  let area_incircle := π * r^2
  let probability_outside := 1 - area_incircle / area_triangle
  probability_outside = 1 - (3 * π) / 20 := 
by
  sorry

end bean_lands_outside_inscribed_circle_l101_101617


namespace savings_per_egg_l101_101013

def price_per_organic_egg : ℕ := 50 
def cost_of_tray : ℕ := 1200 -- in cents
def number_of_eggs_in_tray : ℕ := 30

theorem savings_per_egg : 
  price_per_organic_egg - (cost_of_tray / number_of_eggs_in_tray) = 10 := 
by
  sorry

end savings_per_egg_l101_101013


namespace part_a_smallest_number_of_lines_part_b_smallest_number_of_lines_l101_101267

theorem part_a_smallest_number_of_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≥ 2022 → ∃ k : ℕ, k = 65 :=
sorry

theorem part_b_smallest_number_of_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≥ 2023 → ∃ k : ℕ, k = 65 :=
sorry

end part_a_smallest_number_of_lines_part_b_smallest_number_of_lines_l101_101267


namespace find_x_5pi_over_4_l101_101333

open Real

theorem find_x_5pi_over_4 (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * π) (h3 : sin x + cos x = -sqrt 2) : x = 5 * π / 4 := 
sorry

end find_x_5pi_over_4_l101_101333


namespace system_solutions_l101_101910

theorem system_solutions (a b : ℝ) :
  (∃ (x y : ℝ), x^2 - 2*x + y^2 = 0 ∧ a*x + y = a*b) ↔ 0 ≤ b ∧ b ≤ 2 := 
sorry

end system_solutions_l101_101910


namespace compute_expr_l101_101456

theorem compute_expr : 6^2 - 4 * 5 + 2^2 = 20 := by
  sorry

end compute_expr_l101_101456


namespace find_exponent_M_l101_101358

theorem find_exponent_M (M : ℕ) : (32^4) * (4^6) = 2^M → M = 32 := by
  sorry

end find_exponent_M_l101_101358


namespace largest_divisor_consecutive_odd_squares_l101_101473

theorem largest_divisor_consecutive_odd_squares (m n : ℤ) 
  (hmn : m = n + 2) 
  (hodd_m : m % 2 = 1) 
  (hodd_n : n % 2 = 1) 
  (horder : n < m) : ∃ k : ℤ, m^2 - n^2 = 8 * k :=
by 
  sorry

end largest_divisor_consecutive_odd_squares_l101_101473


namespace sum_of_roots_is_zero_l101_101776

variable {R : Type*} [LinearOrderedField R]

-- Define the function f : R -> R and its properties
variable (f : R → R)
variable (even_f : ∀ x, f x = f (-x))
variable (roots_f : Finset R)
variable (roots_f_four : roots_f.card = 4)
variable (roots_f_set : ∀ x, x ∈ roots_f → f x = 0)

theorem sum_of_roots_is_zero : (roots_f.sum id) = 0 := 
sorry

end sum_of_roots_is_zero_l101_101776


namespace correct_average_weight_l101_101130

noncomputable def initial_average_weight : ℚ := 58.4
noncomputable def num_boys : ℕ := 20
noncomputable def misread_weight : ℚ := 56
noncomputable def correct_weight : ℚ := 66

theorem correct_average_weight : 
  (initial_average_weight * num_boys + (correct_weight - misread_weight)) / num_boys = 58.9 := 
by
  sorry

end correct_average_weight_l101_101130


namespace origin_inside_ellipse_l101_101286

theorem origin_inside_ellipse (k : ℝ) (h : k^2 * 0^2 + 0^2 - 4*k*0 + 2*k*0 + k^2 - 1 < 0) : 0 < |k| ∧ |k| < 1 :=
by
  sorry

end origin_inside_ellipse_l101_101286


namespace factorized_polynomial_sum_of_squares_l101_101835

theorem factorized_polynomial_sum_of_squares :
  ∃ a b c d e f : ℤ, 
    (729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
    (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210) :=
sorry

end factorized_polynomial_sum_of_squares_l101_101835


namespace find_principal_amount_l101_101112

noncomputable def principal_amount (SI : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (SI * 100) / (R * T)

theorem find_principal_amount :
  principal_amount 130 4.166666666666667 4 = 780 :=
by
  -- Sorry is used to denote that the proof is yet to be provided
  sorry

end find_principal_amount_l101_101112


namespace rectangle_parallelepiped_angles_l101_101824

theorem rectangle_parallelepiped_angles 
  (a b c d : ℝ) 
  (α β : ℝ) 
  (h_a : a = d * Real.sin β)
  (h_b : b = d * Real.sin α)
  (h_d : d^2 = (d * Real.sin β)^2 + c^2 + (d * Real.sin α)^2) :
  (α > 0 ∧ β > 0 ∧ α + β < 90) := sorry

end rectangle_parallelepiped_angles_l101_101824


namespace immigration_per_year_l101_101436

-- Definitions based on the initial conditions
def initial_population : ℕ := 100000
def birth_rate : ℕ := 60 -- this represents 60%
def duration_years : ℕ := 10
def emigration_per_year : ℕ := 2000
def final_population : ℕ := 165000

-- Theorem statement: The number of people that immigrated per year
theorem immigration_per_year (immigration_per_year : ℕ) :
  immigration_per_year = 2500 :=
  sorry

end immigration_per_year_l101_101436


namespace television_price_reduction_l101_101724

variable (P : ℝ) (F : ℝ)
variable (h : F = 0.56 * P - 50)

theorem television_price_reduction :
  F / P = 0.56 - 50 / P :=
by {
  sorry
}

end television_price_reduction_l101_101724


namespace Sandy_tokens_difference_l101_101155

theorem Sandy_tokens_difference :
  let total_tokens : ℕ := 1000000
  let siblings : ℕ := 4
  let Sandy_tokens : ℕ := total_tokens / 2
  let sibling_tokens : ℕ := Sandy_tokens / siblings
  Sandy_tokens - sibling_tokens = 375000 :=
by
  sorry

end Sandy_tokens_difference_l101_101155


namespace arccos_one_eq_zero_l101_101931

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end arccos_one_eq_zero_l101_101931


namespace surface_area_of_equal_volume_cube_l101_101578

def vol_rect_prism (l w h : ℝ) : ℝ := l * w * h
def surface_area_cube (s : ℝ) : ℝ := 6 * s^2

theorem surface_area_of_equal_volume_cube :
  (vol_rect_prism 5 5 45 = surface_area_cube 10.5) :=
by
  sorry

end surface_area_of_equal_volume_cube_l101_101578


namespace cos_eq_neg_1_over_4_of_sin_eq_1_over_4_l101_101500

theorem cos_eq_neg_1_over_4_of_sin_eq_1_over_4
  (α : ℝ)
  (h : Real.sin (α + π / 3) = 1 / 4) :
  Real.cos (α + 5 * π / 6) = -1 / 4 :=
sorry

end cos_eq_neg_1_over_4_of_sin_eq_1_over_4_l101_101500


namespace trajectory_equation_equation_of_line_l101_101526

-- Define the parabola and the trajectory
def parabola (x y : ℝ) := y^2 = 16 * x
def trajectory (x y : ℝ) := y^2 = 4 * x

-- Define the properties of the point P and the line l
def is_midpoint (P A B : ℝ × ℝ) :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def line_through_point (x y k : ℝ) := 
  k * x + y = 1

-- Proof problem (Ⅰ): trajectory of the midpoints of segments perpendicular to the x-axis from points on the parabola
theorem trajectory_equation : ∀ (M : ℝ × ℝ), 
  (∃ (P : ℝ × ℝ), parabola P.1 P.2 ∧ is_midpoint M P (P.1, 0)) → 
  trajectory M.1 M.2 :=
sorry

-- Proof problem (Ⅱ): equation of line l
theorem equation_of_line : ∀ (A B P : ℝ × ℝ), 
  trajectory A.1 A.2 → trajectory B.1 B.2 → 
  P = (3,2) → is_midpoint P A B → 
  ∃ k, line_through_point (A.1 - B.1) (A.2 - B.2) k :=
sorry

end trajectory_equation_equation_of_line_l101_101526


namespace greatest_fraction_lt_17_l101_101570

theorem greatest_fraction_lt_17 :
  ∃ (x : ℚ), x = 15 / 4 ∧ x^2 < 17 ∧ ∀ y : ℚ, y < 4 → y^2 < 17 → y ≤ 15 / 4 := 
by
  use 15 / 4
  sorry

end greatest_fraction_lt_17_l101_101570


namespace quadratic_distinct_roots_l101_101497

theorem quadratic_distinct_roots
  (a b c : ℝ)
  (h1 : 5 * a + 3 * b + 2 * c = 0)
  (h2 : a ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1 ^ 2 + b * x1 + c = 0) ∧ (a * x2 ^ 2 + b * x2 + c = 0) :=
by
  sorry

end quadratic_distinct_roots_l101_101497


namespace depression_comparative_phrase_l101_101877

def correct_comparative_phrase (phrase : String) : Prop :=
  phrase = "twice as…as"

theorem depression_comparative_phrase :
  correct_comparative_phrase "twice as…as" :=
by
  sorry

end depression_comparative_phrase_l101_101877


namespace scrap_metal_collected_l101_101121

theorem scrap_metal_collected (a b : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9)
  (h₂ : 900 + 10 * a + b - (100 * a + 10 * b + 9) = 216) :
  900 + 10 * a + b = 975 ∧ 100 * a + 10 * b + 9 = 759 :=
by
  sorry

end scrap_metal_collected_l101_101121


namespace all_children_receive_candy_l101_101006

-- Define f(x) function
def f (x n : ℕ) : ℕ := ((x * (x + 1)) / 2) % n

-- Define the problem statement: prove that all children receive at least one candy if n is a power of 2.
theorem all_children_receive_candy (n : ℕ) (h : ∃ m, n = 2^m) : 
    ∀ i : ℕ, i < n → ∃ x : ℕ, i = f x n := 
sorry

end all_children_receive_candy_l101_101006


namespace schools_participation_l101_101894

-- Definition of the problem conditions
def school_teams : ℕ := 3

-- Paula's rank p must satisfy this
def total_participants (p : ℕ) : ℕ := 2 * p - 1

-- Predicate indicating the number of participants condition:
def participants_condition (p : ℕ) : Prop := total_participants p ≥ 75

-- Translation of number of participants to number of schools
def number_of_schools (n : ℕ) : ℕ := 3 * n

-- The statement to prove:
theorem schools_participation : ∃ (n p : ℕ), participants_condition p ∧ p = 38 ∧ number_of_schools n = total_participants p ∧ n = 25 := 
by 
  sorry

end schools_participation_l101_101894


namespace person_A_arrives_before_B_l101_101347

variable {a b S : ℝ}

theorem person_A_arrives_before_B (h : a ≠ b) (a_pos : 0 < a) (b_pos : 0 < b) (S_pos : 0 < S) :
  (2 * S / (a + b)) < ((a + b) * S / (2 * a * b)) :=
by
  sorry

end person_A_arrives_before_B_l101_101347


namespace min_value_f_exists_min_value_f_l101_101016

noncomputable def f (a b c : ℝ) := 1 / (b^2 + b * c) + 1 / (c^2 + c * a) + 1 / (a^2 + a * b)

theorem min_value_f (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) : f a b c ≥ 3 / 2 :=
  sorry

theorem exists_min_value_f : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧ f a b c = 3 / 2 :=
  sorry

end min_value_f_exists_min_value_f_l101_101016


namespace one_third_of_6_3_eq_21_10_l101_101281

theorem one_third_of_6_3_eq_21_10 : (6.3 / 3) = (21 / 10) := by
  sorry

end one_third_of_6_3_eq_21_10_l101_101281


namespace number_of_ways_to_select_one_person_number_of_ways_to_select_one_person_each_type_l101_101804

-- Definitions based on the conditions
def peopleO : ℕ := 28
def peopleA : ℕ := 7
def peopleB : ℕ := 9
def peopleAB : ℕ := 3

-- Proof for Question 1
theorem number_of_ways_to_select_one_person : peopleO + peopleA + peopleB + peopleAB = 47 := by
  sorry

-- Proof for Question 2
theorem number_of_ways_to_select_one_person_each_type : peopleO * peopleA * peopleB * peopleAB = 5292 := by
  sorry

end number_of_ways_to_select_one_person_number_of_ways_to_select_one_person_each_type_l101_101804


namespace quilt_shaded_fraction_l101_101108

theorem quilt_shaded_fraction :
  let original_squares := 9
  let shaded_column_squares := 3
  let fraction_shaded := shaded_column_squares / original_squares 
  fraction_shaded = 1/3 :=
by
  sorry

end quilt_shaded_fraction_l101_101108


namespace parabola_point_coordinates_l101_101430

theorem parabola_point_coordinates (x y : ℝ) (h_parabola : y^2 = 8 * x) 
    (h_distance_focus : (x + 2)^2 + y^2 = 81) : 
    (x = 7 ∧ y = 2 * Real.sqrt 14) ∨ (x = 7 ∧ y = -2 * Real.sqrt 14) :=
by {
  -- Proof will be inserted here
  sorry
}

end parabola_point_coordinates_l101_101430


namespace defective_percentage_is_correct_l101_101224

noncomputable def percentage_defective (defective : ℕ) (total : ℝ) : ℝ := 
  (defective / total) * 100

theorem defective_percentage_is_correct : 
  percentage_defective 2 3333.3333333333335 = 0.06000600060006 :=
by
  sorry

end defective_percentage_is_correct_l101_101224


namespace find_c_minus_d_l101_101653

variable (g : ℝ → ℝ)
variable (c d : ℝ)
variable (invertible_g : Function.Injective g)
variable (g_at_c : g c = d)
variable (g_at_d : g d = 5)

theorem find_c_minus_d : c - d = -3 := by
  sorry

end find_c_minus_d_l101_101653


namespace factorize_x4_minus_16y4_factorize_minus_2a3_plus_12a2_minus_16a_l101_101955

-- Given condition and question, prove equality for the first expression
theorem factorize_x4_minus_16y4 (x y : ℝ) :
  x^4 - 16 * y^4 = (x^2 + 4 * y^2) * (x + 2 * y) * (x - 2 * y) := 
by sorry

-- Given condition and question, prove equality for the second expression
theorem factorize_minus_2a3_plus_12a2_minus_16a (a : ℝ) :
  -2 * a^3 + 12 * a^2 - 16 * a = -2 * a * (a - 2) * (a - 4) := 
by sorry

end factorize_x4_minus_16y4_factorize_minus_2a3_plus_12a2_minus_16a_l101_101955


namespace same_solution_m_l101_101951

theorem same_solution_m (m x : ℤ) : 
  (8 - m = 2 * (x + 1)) ∧ (2 * (2 * x - 3) - 1 = 1 - 2 * x) → m = 10 / 3 :=
by
  sorry

end same_solution_m_l101_101951


namespace fraction_equation_solution_l101_101561

theorem fraction_equation_solution (x : ℝ) : (4 + x) / (7 + x) = (2 + x) / (3 + x) ↔ x = -1 := 
by
  sorry

end fraction_equation_solution_l101_101561


namespace sum_of_first_15_terms_is_largest_l101_101370

theorem sum_of_first_15_terms_is_largest
  (a : ℕ → ℝ)
  (s : ℕ → ℝ)
  (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : ∀ n, s n = n * a 1 + (n * (n - 1) * d) / 2)
  (h1: 13 * a 6 = 19 * (a 6 + 3 * d))
  (h2: a 1 > 0) : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≠ 15 → s 15 > s n :=
by
  sorry

end sum_of_first_15_terms_is_largest_l101_101370


namespace max_profit_under_budget_max_profit_no_budget_l101_101964

-- Definitions from conditions
def sales_revenue (x1 x2 : ℝ) : ℝ :=
  -2 * x1^2 - x2^2 + 13 * x1 + 11 * x2 - 28

def profit (x1 x2 : ℝ) : ℝ :=
  sales_revenue x1 x2 - x1 - x2

-- Statements for the conditions
theorem max_profit_under_budget :
  (∀ x1 x2 : ℝ, x1 + x2 = 5 → profit x1 x2 ≤ 9) ∧
  (profit 2 3 = 9) :=
by sorry

theorem max_profit_no_budget :
  (∀ x1 x2 : ℝ, profit x1 x2 ≤ 15) ∧
  (profit 3 5 = 15) :=
by sorry

end max_profit_under_budget_max_profit_no_budget_l101_101964


namespace joan_carrots_grown_correct_l101_101771

variable (total_carrots : ℕ) (jessica_carrots : ℕ) (joan_carrots : ℕ)

theorem joan_carrots_grown_correct (h1 : total_carrots = 40) (h2 : jessica_carrots = 11) (h3 : total_carrots = joan_carrots + jessica_carrots) : joan_carrots = 29 :=
by
  sorry

end joan_carrots_grown_correct_l101_101771


namespace trig_identity_example_l101_101972

noncomputable def cos24 := Real.cos (24 * Real.pi / 180)
noncomputable def cos36 := Real.cos (36 * Real.pi / 180)
noncomputable def sin24 := Real.sin (24 * Real.pi / 180)
noncomputable def sin36 := Real.sin (36 * Real.pi / 180)
noncomputable def cos60 := Real.cos (60 * Real.pi / 180)

theorem trig_identity_example :
  cos24 * cos36 - sin24 * sin36 = cos60 :=
by
  sorry

end trig_identity_example_l101_101972


namespace ratio_of_potatoes_l101_101539

def total_potatoes : ℕ := 24
def number_of_people : ℕ := 3
def potatoes_per_person : ℕ := 8
def total_each_person : ℕ := potatoes_per_person * number_of_people

theorem ratio_of_potatoes :
  total_potatoes = total_each_person → (potatoes_per_person : ℚ) / (potatoes_per_person : ℚ) = 1 :=
by
  sorry

end ratio_of_potatoes_l101_101539


namespace joan_friends_kittens_l101_101111

theorem joan_friends_kittens (initial_kittens final_kittens friends_kittens : ℕ) 
  (h1 : initial_kittens = 8) 
  (h2 : final_kittens = 10) 
  (h3 : friends_kittens = 2) : 
  final_kittens - initial_kittens = friends_kittens := 
by 
  -- Sorry is used here as a placeholder to indicate where the proof would go.
  sorry

end joan_friends_kittens_l101_101111


namespace probability_inequality_up_to_99_l101_101043

theorem probability_inequality_up_to_99 :
  (∀ x : ℕ, 1 ≤ x ∧ x < 100 → (2^x / x!) > x^2) →
    (∃ n : ℕ, (1 ≤ n ∧ n < 100) ∧ (2^n / n!) > n^2) →
      ∃ p : ℚ, p = 1/99 :=
by
  sorry

end probability_inequality_up_to_99_l101_101043


namespace joe_money_fraction_l101_101336

theorem joe_money_fraction :
  ∃ f : ℝ,
    (200 : ℝ) = 160 + (200 - 160) ∧
    160 - 160 * f - 20 = 40 + 160 * f + 20 ∧
    f = 1 / 4 :=
by
  -- The proof should go here.
  sorry

end joe_money_fraction_l101_101336


namespace greatest_possible_sum_of_two_consecutive_even_integers_l101_101775

theorem greatest_possible_sum_of_two_consecutive_even_integers
  (n : ℤ) (h1 : Even n) (h2 : n * (n + 2) < 800) :
  n + (n + 2) = 54 := 
sorry

end greatest_possible_sum_of_two_consecutive_even_integers_l101_101775


namespace area_common_to_all_four_circles_l101_101477

noncomputable def common_area (R : ℝ) : ℝ :=
  (R^2 * (2 * Real.pi - 3 * Real.sqrt 3)) / 6

theorem area_common_to_all_four_circles (R : ℝ) :
  ∃ (O1 O2 A B : ℝ × ℝ),
    dist O1 O2 = R ∧
    dist O1 A = R ∧
    dist O2 A = R ∧
    dist O1 B = R ∧
    dist O2 B = R ∧
    dist A B = R ∧
    common_area R = (R^2 * (2 * Real.pi - 3 * Real.sqrt 3)) / 6 :=
by
  sorry

end area_common_to_all_four_circles_l101_101477


namespace range_of_log2_sin_squared_l101_101709

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def sin_squared_log_range (x : ℝ) : ℝ :=
  log2 ((Real.sin x) ^ 2)

theorem range_of_log2_sin_squared (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.pi) :
  ∃ y, y = sin_squared_log_range x ∧ y ≤ 0 :=
by
  sorry

end range_of_log2_sin_squared_l101_101709


namespace tank_capacity_l101_101189

variable (C : ℕ) (t : ℕ)
variable (hC_nonzero : C > 0)
variable (ht_nonzero : t > 0)
variable (h_rate_pipe_A : t = C / 5)
variable (h_rate_pipe_B : t = C / 8)
variable (h_rate_inlet : t = 4 * 60)
variable (h_combined_time : t = 5 + 3)

theorem tank_capacity (C : ℕ) (h1 : C / 5 + C / 8 - 4 * 60 = 8) : C = 1200 := 
by
  sorry

end tank_capacity_l101_101189


namespace impossible_arrangement_of_300_numbers_in_circle_l101_101926

theorem impossible_arrangement_of_300_numbers_in_circle :
  ¬ ∃ (nums : Fin 300 → ℕ), (∀ i : Fin 300, nums i > 0) ∧
    ∃ unique_exception : Fin 300,
      ∀ i : Fin 300, i ≠ unique_exception → nums i = Int.natAbs (nums (Fin.mod (i.val - 1) 300) - nums (Fin.mod (i.val + 1) 300)) := 
sorry

end impossible_arrangement_of_300_numbers_in_circle_l101_101926


namespace intersection_A_B_l101_101455

def A : Set ℝ := { x | abs x ≤ 1 }
def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_A_B : (A ∩ B) = { x | 0 ≤ x ∧ x ≤ 1 } :=
sorry

end intersection_A_B_l101_101455


namespace max_val_of_m_if_p_true_range_of_m_if_one_prop_true_one_false_range_of_m_if_one_prop_false_one_true_l101_101543

variable {m x x0 : ℝ}

def proposition_p (m : ℝ) : Prop := ∀ x > -2, x + 49 / (x + 2) ≥ 6 * Real.sqrt 2 * m
def proposition_q (m : ℝ) : Prop := ∃ x0 : ℝ, x0 ^ 2 - m * x0 + 1 = 0

theorem max_val_of_m_if_p_true (h : proposition_p m) : m ≤ Real.sqrt 2 := by
  sorry

theorem range_of_m_if_one_prop_true_one_false (hp : proposition_p m) (hq : ¬ proposition_q m) : (-2 < m ∧ m ≤ Real.sqrt 2) ∨ (2 ≤ m) := by
  sorry

theorem range_of_m_if_one_prop_false_one_true (hp : ¬ proposition_p m) (hq : proposition_q m) : (m ≥ 2) := by
  sorry

end max_val_of_m_if_p_true_range_of_m_if_one_prop_true_one_false_range_of_m_if_one_prop_false_one_true_l101_101543


namespace colors_used_l101_101738

theorem colors_used (total_blocks number_per_color : ℕ) (h1 : total_blocks = 196) (h2 : number_per_color = 14) : 
  total_blocks / number_per_color = 14 :=
by
  sorry

end colors_used_l101_101738


namespace koala_fiber_intake_l101_101892

theorem koala_fiber_intake 
  (absorption_rate : ℝ) 
  (absorbed_fiber : ℝ) 
  (eaten_fiber : ℝ) 
  (h1 : absorption_rate = 0.40) 
  (h2 : absorbed_fiber = 16)
  (h3 : absorbed_fiber = absorption_rate * eaten_fiber) :
  eaten_fiber = 40 := 
  sorry

end koala_fiber_intake_l101_101892


namespace solve_system_of_equations_l101_101849

theorem solve_system_of_equations (x y z t : ℝ) :
  xy - t^2 = 9 ∧ x^2 + y^2 + z^2 = 18 ↔ (x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ (x = -3 ∧ y = -3 ∧ z = 0 ∨ t = 0) :=
sorry

end solve_system_of_equations_l101_101849


namespace seq_b_arithmetic_diff_seq_a_general_term_l101_101474

variable {n : ℕ}

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n / (a n + 2)

def seq_b (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = 1 / a n

theorem seq_b_arithmetic_diff (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_a : seq_a a) (h_b : seq_b a b) :
  ∀ n, b (n + 1) - b n = 1 / 2 :=
by
  sorry

theorem seq_a_general_term (a : ℕ → ℝ) (h_a : seq_a a) :
  ∀ n, a n = 2 / (n + 1) :=
by
  sorry

end seq_b_arithmetic_diff_seq_a_general_term_l101_101474


namespace linda_total_distance_l101_101307

theorem linda_total_distance
  (miles_per_gallon : ℝ) (tank_capacity : ℝ) (initial_distance : ℝ) (refuel_amount : ℝ) (final_tank_fraction : ℝ)
  (fuel_used_first_segment : ℝ := initial_distance / miles_per_gallon)
  (initial_fuel_full : fuel_used_first_segment = tank_capacity)
  (total_fuel_after_refuel : ℝ := 0 + refuel_amount)
  (remaining_fuel_stopping : ℝ := final_tank_fraction * tank_capacity)
  (fuel_used_second_segment : ℝ := total_fuel_after_refuel - remaining_fuel_stopping)
  (distance_second_leg : ℝ := fuel_used_second_segment * miles_per_gallon) :
  initial_distance + distance_second_leg = 637.5 := by
  sorry

end linda_total_distance_l101_101307


namespace factor_expression_l101_101582

theorem factor_expression (y z : ℝ) : 3 * y^2 - 75 * z^2 = 3 * (y + 5 * z) * (y - 5 * z) :=
by sorry

end factor_expression_l101_101582


namespace year_population_below_five_percent_l101_101126

def population (P0 : ℕ) (years : ℕ) : ℕ :=
  P0 / 2^years

theorem year_population_below_five_percent (P0 : ℕ) :
  ∃ n, population P0 n < P0 / 20 ∧ (2005 + n) = 2010 := 
by {
  sorry
}

end year_population_below_five_percent_l101_101126


namespace factorial_expression_l101_101060

namespace FactorialProblem

-- Definition of factorial function.
def factorial : ℕ → ℕ 
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Theorem stating the problem equivalently.
theorem factorial_expression : (factorial 12 - factorial 10) / factorial 8 = 11790 := by
  sorry

end FactorialProblem

end factorial_expression_l101_101060


namespace gen_sequence_term_l101_101756

theorem gen_sequence_term (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ k, a (k + 1) = 3 * a k + 1) :
  a n = (3^n - 1) / 2 := by
  sorry

end gen_sequence_term_l101_101756


namespace housewife_oil_expense_l101_101200

theorem housewife_oil_expense:
  ∃ M P R: ℝ, (R = 30) ∧ (0.8 * P = R) ∧ ((M / R) - (M / P) = 10) ∧ (M = 1500) :=
by
  sorry

end housewife_oil_expense_l101_101200


namespace shadow_area_l101_101103

theorem shadow_area (y : ℝ) (cube_side : ℝ) (shadow_excl_area : ℝ) 
  (h₁ : cube_side = 2) 
  (h₂ : shadow_excl_area = 200)
  (h₃ : ((14.28 - 2) / 2 = y)) :
  ⌊1000 * y⌋ = 6140 :=
by
  sorry

end shadow_area_l101_101103


namespace var_power_l101_101029

theorem var_power {a b c x y z : ℝ} (h1 : x = a * y^4) (h2 : y = b * z^(1/3)) :
  ∃ n : ℝ, x = c * z^n ∧ n = 4/3 := by
  sorry

end var_power_l101_101029


namespace range_of_a_l101_101773

-- Definitions capturing the given conditions
variables (a b c : ℝ)

-- Conditions are stated as assumptions
def condition1 := a^2 - b * c - 8 * a + 7 = 0
def condition2 := b^2 + c^2 + b * c - 6 * a + 6 = 0

-- The mathematically equivalent proof problem
theorem range_of_a (h1 : condition1 a b c) (h2 : condition2 a b c) : 1 ≤ a ∧ a ≤ 9 := 
sorry

end range_of_a_l101_101773


namespace pages_needed_l101_101990

def cards_per_page : ℕ := 3
def new_cards : ℕ := 2
def old_cards : ℕ := 10

theorem pages_needed : (new_cards + old_cards) / cards_per_page = 4 := by
  sorry

end pages_needed_l101_101990


namespace difference_is_correct_l101_101199

-- Define the given constants and conditions
def purchase_price : ℕ := 1500
def down_payment : ℕ := 200
def monthly_payment : ℕ := 65
def number_of_monthly_payments : ℕ := 24

-- Define the derived quantities based on the given conditions
def total_monthly_payments : ℕ := monthly_payment * number_of_monthly_payments
def total_amount_paid : ℕ := down_payment + total_monthly_payments
def difference : ℕ := total_amount_paid - purchase_price

-- The statement to be proven
theorem difference_is_correct : difference = 260 := by
  sorry

end difference_is_correct_l101_101199


namespace geometric_series_sum_l101_101752

theorem geometric_series_sum :
  let a := 4 / 5
  let r := 4 / 5
  let n := 15
  let S := (a * (1 - r^n)) / (1 - r)
  S = 117775277204 / 30517578125 := by
  let a := 4 / 5
  let r := 4 / 5
  let n := 15
  let S := (a * (1 - r^n)) / (1 - r)
  have : S = 117775277204 / 30517578125 := sorry
  exact this

end geometric_series_sum_l101_101752


namespace sum_at_simple_interest_l101_101674

theorem sum_at_simple_interest (P R : ℝ) (h1: ((3 * P * (R + 1))/ 100) = ((3 * P * R) / 100 + 72)) : P = 2400 := 
by 
  sorry

end sum_at_simple_interest_l101_101674


namespace original_quantity_of_ghee_l101_101850

theorem original_quantity_of_ghee
  (Q : ℝ) 
  (H1 : (0.5 * Q) = (0.3 * (Q + 20))) : 
  Q = 30 := 
by
  -- proof goes here
  sorry

end original_quantity_of_ghee_l101_101850


namespace isosceles_triangle_perimeter_l101_101419

theorem isosceles_triangle_perimeter (m : ℝ) (a b : ℝ) 
  (h1 : 3 = a ∨ 3 = b)
  (h2 : a ≠ b)
  (h3 : a^2 - (m+1)*a + 2*m = 0)
  (h4 : b^2 - (m+1)*b + 2*m = 0) :
  (a + b + a = 11) ∨ (a + a + b = 10) := 
sorry

end isosceles_triangle_perimeter_l101_101419


namespace mike_picked_64_peaches_l101_101837

theorem mike_picked_64_peaches :
  ∀ (initial peaches_given total final_picked : ℕ),
    initial = 34 →
    peaches_given = 12 →
    total = 86 →
    final_picked = total - (initial - peaches_given) →
    final_picked = 64 :=
by
  intros
  sorry

end mike_picked_64_peaches_l101_101837


namespace find_angles_and_area_l101_101353

noncomputable def angles_in_arithmetic_progression (A B C : ℝ) : Prop :=
  A + C = 2 * B ∧ A + B + C = 180

noncomputable def side_ratios (a b : ℝ) : Prop :=
  a / b = Real.sqrt 2 / Real.sqrt 3

noncomputable def triangle_area (a b c A B C : ℝ) : ℝ :=
  (1/2) * a * c * Real.sin B

theorem find_angles_and_area :
  ∃ (A B C a b c : ℝ), 
    angles_in_arithmetic_progression A B C ∧ 
    side_ratios a b ∧ 
    c = 2 ∧ 
    A = 45 ∧ 
    B = 60 ∧ 
    C = 75 ∧ 
    triangle_area a b c A B C = 3 - Real.sqrt 3 :=
sorry

end find_angles_and_area_l101_101353


namespace D_cows_grazed_l101_101807

-- Defining the given conditions:
def A_cows := 24
def A_months := 3
def A_rent := 1440

def B_cows := 10
def B_months := 5

def C_cows := 35
def C_months := 4

def D_months := 3

def total_rent := 6500

-- Calculate the cost per cow per month (CPCM)
def CPCM := A_rent / (A_cows * A_months)

-- Proving the number of cows D grazed
theorem D_cows_grazed : ∃ x : ℕ, (x * D_months * CPCM + A_rent + (B_cows * B_months * CPCM) + (C_cows * C_months * CPCM) = total_rent) ∧ x = 21 := by
  sorry

end D_cows_grazed_l101_101807


namespace ratio_of_ducks_to_total_goats_and_chickens_l101_101924

theorem ratio_of_ducks_to_total_goats_and_chickens 
    (goats chickens ducks pigs : ℕ) 
    (h1 : goats = 66)
    (h2 : chickens = 2 * goats)
    (h3 : pigs = ducks / 3)
    (h4 : goats = pigs + 33) :
    (ducks : ℚ) / (goats + chickens : ℚ) = 1 / 2 := 
by
  sorry

end ratio_of_ducks_to_total_goats_and_chickens_l101_101924


namespace class_C_payment_l101_101272

-- Definitions based on conditions
variables (x y z : ℤ) (total_C : ℤ)

-- Given conditions
def condition_A : Prop := 3 * x + 7 * y + z = 14
def condition_B : Prop := 4 * x + 10 * y + z = 16
def condition_C : Prop := 3 * (x + y + z) = total_C

-- The theorem to prove
theorem class_C_payment (hA : condition_A x y z) (hB : condition_B x y z) : total_C = 30 :=
sorry

end class_C_payment_l101_101272


namespace ways_to_distribute_balls_l101_101531

def num_balls : ℕ := 5
def num_boxes : ℕ := 3

theorem ways_to_distribute_balls : 3^5 = 243 :=
by
  sorry

end ways_to_distribute_balls_l101_101531


namespace abs_inequality_solution_l101_101209

theorem abs_inequality_solution {a : ℝ} (h : ∀ x : ℝ, |2 - x| + |x + 1| ≥ a) : a ≤ 3 :=
sorry

end abs_inequality_solution_l101_101209


namespace modulo_17_residue_l101_101014

theorem modulo_17_residue : (392 + 6 * 51 + 8 * 221 + 3^2 * 23) % 17 = 11 :=
by 
  sorry

end modulo_17_residue_l101_101014


namespace trigonometric_identity_l101_101490

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 3) : 1 / (Real.sin x ^ 2 - 2 * Real.cos x ^ 2) = 10 / 7 :=
by
  sorry

end trigonometric_identity_l101_101490


namespace divisor_is_50_l101_101313

theorem divisor_is_50 (D : ℕ) (h1 : ∃ n, n = 44 * 432 ∧ n % 44 = 0)
                      (h2 : ∃ n, n = 44 * 432 ∧ n % D = 8) : D = 50 :=
by
  sorry

end divisor_is_50_l101_101313


namespace matrix_not_invertible_x_l101_101391

theorem matrix_not_invertible_x (x : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![2 + x, 9], ![4 - x, 10]]
  A.det = 0 ↔ x = 16 / 19 := sorry

end matrix_not_invertible_x_l101_101391


namespace cos_of_F_in_def_l101_101707

theorem cos_of_F_in_def (E F : ℝ) (h₁ : E + F = π / 2) (h₂ : Real.sin E = 3 / 5) : Real.cos F = 3 / 5 :=
sorry

end cos_of_F_in_def_l101_101707


namespace center_polar_coordinates_l101_101492

-- Assuming we have a circle defined in polar coordinates
def polar_circle_center (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ + 2 * Real.sin θ

-- The goal is to prove that the center of this circle has the polar coordinates (sqrt 2, π/4)
theorem center_polar_coordinates : ∃ ρ θ, polar_circle_center ρ θ ∧ ρ = Real.sqrt 2 ∧ θ = Real.pi / 4 :=
sorry

end center_polar_coordinates_l101_101492


namespace problem1_problem2a_problem2b_problem3_l101_101434

noncomputable def f (a x : ℝ) := -x^2 + a * x - 2
noncomputable def g (x : ℝ) := x * Real.log x

-- Problem 1
theorem problem1 {a : ℝ} : (∀ x : ℝ, 0 < x → g x ≥ f a x) → a ≤ 3 :=
sorry

-- Problem 2 
theorem problem2a (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1 / Real.exp 1) :
  ∃ xmin : ℝ, g (1 / Real.exp 1) = -1 / Real.exp 1 ∧ 
  ∃ xmax : ℝ, g (m + 1) = (m + 1) * Real.log (m + 1) :=
sorry

theorem problem2b (m : ℝ) (h₀ : 1 / Real.exp 1 ≤ m) :
  ∃ xmin ymax : ℝ, xmin = g m ∧ ymax = g (m + 1) :=
sorry

-- Problem 3
theorem problem3 (x : ℝ) (h : 0 < x) : 
  Real.log x + (2 / (Real.exp 1 * x)) ≥ 1 / Real.exp x :=
sorry

end problem1_problem2a_problem2b_problem3_l101_101434


namespace sin_75_deg_l101_101350

theorem sin_75_deg : Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := 
by sorry

end sin_75_deg_l101_101350


namespace skilled_new_worker_installation_avg_cost_electric_vehicle_cost_comparison_l101_101611

-- Define the variables for the number of vehicles each type of worker can install
variables {x y : ℝ}

-- Define the conditions for system of equations
def skilled_and_new_workers_system1 (x y : ℝ) : Prop :=
  2 * x + y = 10

def skilled_and_new_workers_system2 (x y : ℝ) : Prop :=
  x + 3 * y = 10

-- Prove the number of vehicles each skilled worker and new worker can install
theorem skilled_new_worker_installation (x y : ℝ) (h1 : skilled_and_new_workers_system1 x y) (h2 : skilled_and_new_workers_system2 x y) : x = 4 ∧ y = 2 :=
by {
  -- Proof skipped
  sorry
}

-- Define the average cost equation for electric and gasoline vehicles
def avg_cost (m : ℝ) : Prop :=
  1 = 4 * (m / (m + 0.6))

-- Prove the average cost per kilometer of the electric vehicle
theorem avg_cost_electric_vehicle (m : ℝ) (h : avg_cost m) : m = 0.2 :=
by {
  -- Proof skipped
  sorry
}

-- Define annual cost equations and the comparison condition
variables {a : ℝ}
def annual_cost_electric_vehicle (a : ℝ) : ℝ :=
  0.2 * a + 6400

def annual_cost_gasoline_vehicle (a : ℝ) : ℝ :=
  0.8 * a + 4000

-- Prove that when the annual mileage is greater than 6667 kilometers, the annual cost of buying an electric vehicle is lower
theorem cost_comparison (a : ℝ) (h : a > 6667) : annual_cost_electric_vehicle a < annual_cost_gasoline_vehicle a :=
by {
  -- Proof skipped
  sorry
}

end skilled_new_worker_installation_avg_cost_electric_vehicle_cost_comparison_l101_101611


namespace projection_matrix_exists_l101_101247

noncomputable def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, (20 : ℚ) / 49], ![c, (29 : ℚ) / 49]]

theorem projection_matrix_exists :
  ∃ (a c : ℚ), P a c * P a c = P a c ∧ a = (20 : ℚ) / 49 ∧ c = (29 : ℚ) / 49 := 
by
  use ((20 : ℚ) / 49), ((29 : ℚ) / 49)
  simp [P]
  sorry

end projection_matrix_exists_l101_101247


namespace road_trip_ratio_l101_101731

theorem road_trip_ratio (D R: ℝ) (h1 : 1 / 2 * D = 40) (h2 : 2 * (D + R * D + 40) = 560 - (D + R * D + 40)) :
  R = 5 / 6 := by
  sorry

end road_trip_ratio_l101_101731


namespace houses_without_features_l101_101855

-- Definitions for the given conditions
def N : ℕ := 70
def G : ℕ := 50
def P : ℕ := 40
def GP : ℕ := 35

-- The statement of the proof problem
theorem houses_without_features : N - (G + P - GP) = 15 := by
  sorry

end houses_without_features_l101_101855


namespace units_digit_six_l101_101629

theorem units_digit_six (n : ℕ) (h : n > 0) : (6 ^ n) % 10 = 6 :=
by sorry

example : (6 ^ 7) % 10 = 6 :=
units_digit_six 7 (by norm_num)

end units_digit_six_l101_101629


namespace find_fraction_l101_101248

-- Let's define the conditions
variables (F N : ℝ)
axiom condition1 : (1 / 4) * (1 / 3) * F * N = 15
axiom condition2 : 0.4 * N = 180

-- theorem to prove the fraction F
theorem find_fraction : F = 2 / 5 :=
by
  -- proof steps would go here, but we're adding sorry to skip the proof.
  sorry

end find_fraction_l101_101248


namespace cos_double_angle_l101_101292

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + 3 * Real.pi / 2) = Real.sqrt 3 / 3) : Real.cos (2 * α) = -1 / 3 := 
  sorry

end cos_double_angle_l101_101292


namespace determine_n_l101_101886

theorem determine_n (n : ℕ) (hn : 0 < n) :
  (∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 3 * x + 2 * y + z = n) → (n = 15 ∨ n = 16) :=
  by sorry

end determine_n_l101_101886


namespace ratio_rectangle_to_semicircles_area_l101_101227

theorem ratio_rectangle_to_semicircles_area (AB AD : ℝ) (h1 : AB = 40) (h2 : AD / AB = 3 / 2) : 
  (AB * AD) / (2 * (π * (AB / 2)^2)) = 6 / π :=
by
  -- here we process the proof
  sorry

end ratio_rectangle_to_semicircles_area_l101_101227


namespace correct_transformation_l101_101148

-- Definitions of the equations and their transformations
def optionA := (forall (x : ℝ), ((x / 5) + 1 = x / 2) -> (2 * x + 10 = 5 * x))
def optionB := (forall (x : ℝ), (5 - 2 * (x - 1) = x + 3) -> (5 - 2 * x + 2 = x + 3))
def optionC := (forall (x : ℝ), (5 * x + 3 = 8) -> (5 * x = 8 - 3))
def optionD := (forall (x : ℝ), (3 * x = -7) -> (x = -7 / 3))

-- Theorem stating that option D is the correct transformation
theorem correct_transformation : optionD := 
by 
  sorry

end correct_transformation_l101_101148


namespace roots_are_prime_then_a_is_five_l101_101503

theorem roots_are_prime_then_a_is_five (x1 x2 a : ℕ) (h_prime_x1 : Prime x1) (h_prime_x2 : Prime x2)
  (h_eq : x1 + x2 = a) (h_eq_mul : x1 * x2 = a + 1) : a = 5 :=
sorry

end roots_are_prime_then_a_is_five_l101_101503


namespace cistern_water_breadth_l101_101735

theorem cistern_water_breadth 
  (length width : ℝ) (wet_surface_area : ℝ) 
  (hl : length = 9) (hw : width = 6) (hwsa : wet_surface_area = 121.5) : 
  ∃ h : ℝ, 54 + 18 * h + 12 * h = 121.5 ∧ h = 2.25 := 
by 
  sorry

end cistern_water_breadth_l101_101735


namespace number_of_days_b_worked_l101_101772

variables (d_a : ℕ) (d_c : ℕ) (total_earnings : ℝ)
variables (wage_ratio : ℝ) (wage_c : ℝ) (d_b : ℕ) (wages : ℝ)
variables (total_wage_a : ℝ) (total_wage_c : ℝ) (total_wage_b : ℝ)

-- Given conditions
def given_conditions :=
  d_a = 6 ∧
  d_c = 4 ∧
  wage_c = 95 ∧
  wage_ratio = wage_c / 5 ∧
  wages = 3 * wage_ratio ∧
  total_earnings = 1406 ∧
  total_wage_a = d_a * wages ∧
  total_wage_c = d_c * wage_c ∧
  total_wage_b = d_b * (4 * wage_ratio) ∧
  total_wage_a + total_wage_b + total_wage_c = total_earnings

-- Theorem to prove
theorem number_of_days_b_worked :
  given_conditions d_a d_c total_earnings wage_ratio wage_c d_b wages total_wage_a total_wage_c total_wage_b →
  d_b = 9 :=
by
  intro h
  sorry

end number_of_days_b_worked_l101_101772


namespace solve_equation_l101_101882

def equation (x : ℝ) : Prop := (2 / x + 3 * (4 / x / (8 / x)) = 1.2)

theorem solve_equation : 
  ∃ x : ℝ, equation x ∧ x = - 20 / 3 :=
by
  sorry

end solve_equation_l101_101882


namespace tin_silver_ratio_l101_101493

theorem tin_silver_ratio (T S : ℝ) 
  (h1 : T + S = 50) 
  (h2 : 0.1375 * T + 0.075 * S = 5) : 
  T / S = 2 / 3 :=
by
  sorry

end tin_silver_ratio_l101_101493


namespace farm_horses_more_than_cows_l101_101613

variable (x : ℤ) -- number of cows initially, must be a positive integer

def initial_horses := 6 * x
def initial_cows := x
def horses_after_transaction := initial_horses - 30
def cows_after_transaction := initial_cows + 30

-- New ratio after transaction
def new_ratio := horses_after_transaction * 1 = 4 * cows_after_transaction

-- Prove that the farm owns 315 more horses than cows after transaction
theorem farm_horses_more_than_cows :
  new_ratio → horses_after_transaction - cows_after_transaction = 315 :=
by
  sorry

end farm_horses_more_than_cows_l101_101613


namespace find_number_of_piles_l101_101425

theorem find_number_of_piles 
  (Q : ℕ) 
  (h1 : Q = Q) 
  (h2 : ∀ (piles : ℕ), piles = 3) 
  (total_coins : ℕ) 
  (h3 : total_coins = 30) 
  (e : 6 * Q = total_coins) :
  Q = 5 := 
by sorry

end find_number_of_piles_l101_101425


namespace area_of_region_l101_101496

-- Define the condition: the equation of the region
def region_equation (x y : ℝ) : Prop := x^2 + y^2 + 10 * x - 4 * y + 9 = 0

-- State the theorem: the area of the region defined by the equation is 20π
theorem area_of_region : ∀ x y : ℝ, region_equation x y → ∃ A : ℝ, A = 20 * Real.pi :=
by sorry

end area_of_region_l101_101496


namespace range_of_a_l101_101858

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 := by
  sorry

end range_of_a_l101_101858


namespace speed_of_goods_train_l101_101879

theorem speed_of_goods_train 
  (t₁ t₂ v_express : ℝ)
  (h1 : v_express = 90) 
  (h2 : t₁ = 6) 
  (h3 : t₂ = 4)
  (h4 : v_express * t₂ = v * (t₁ + t₂)) : 
  v = 36 :=
by
  sorry

end speed_of_goods_train_l101_101879


namespace projectile_height_time_l101_101838

theorem projectile_height_time :
  ∃ t, t ≥ 0 ∧ -16 * t^2 + 80 * t = 72 ↔ t = 1 := 
by sorry

end projectile_height_time_l101_101838


namespace smallest_k_divides_l101_101427

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l101_101427


namespace count_random_events_l101_101288

-- Definitions based on conditions in the problem
def total_products : ℕ := 100
def genuine_products : ℕ := 95
def defective_products : ℕ := 5
def drawn_products : ℕ := 6

-- Events definitions
def event_1 := drawn_products > defective_products  -- at least 1 genuine product
def event_2 := drawn_products ≥ 3  -- at least 3 defective products
def event_3 := drawn_products = defective_products  -- all 6 are defective
def event_4 := drawn_products - 2 = 4  -- 2 defective and 4 genuine products

-- Dummy definition for random event counter state in the problem context
def random_events : ℕ := 2

-- Main theorem statement
theorem count_random_events :
  (event_1 → true) ∧ 
  (event_2 ∧ ¬ event_3 ∧ event_4) →
  random_events = 2 :=
by
  sorry

end count_random_events_l101_101288


namespace dorms_and_students_l101_101938

theorem dorms_and_students (x : ℕ) :
  (4 * x + 19) % 6 ≠ 0 → ∃ s : ℕ, (x = 10 ∧ s = 59) ∨ (x = 11 ∧ s = 63) ∨ (x = 12 ∧ s = 67) :=
by
  sorry

end dorms_and_students_l101_101938


namespace square_garden_perimeter_l101_101048

theorem square_garden_perimeter (A : ℝ) (h : A = 450) : ∃ P : ℝ, P = 60 * Real.sqrt 2 :=
  sorry

end square_garden_perimeter_l101_101048


namespace ratio_of_sphere_radii_l101_101251

noncomputable def ratio_of_radius (V_large : ℝ) (percentage : ℝ) : ℝ :=
  let V_small := (percentage / 100) * V_large
  let ratio := (V_small / V_large) ^ (1/3)
  ratio

theorem ratio_of_sphere_radii : 
  ratio_of_radius (450 * Real.pi) 27.04 = 0.646 := 
  by
  sorry

end ratio_of_sphere_radii_l101_101251


namespace extreme_value_a_range_l101_101220

theorem extreme_value_a_range (a : ℝ) :
  (∃ x : ℝ, (1 / Real.exp 1 < x ∧ x < Real.exp 1 ∧ x + a * Real.log x + 1 + a / x = 0)) →
  -Real.exp 1 < a ∧ a < -1 / Real.exp 1 :=
by sorry

end extreme_value_a_range_l101_101220


namespace move_line_left_and_up_l101_101466

/--
The equation of the line obtained by moving the line y = 2x - 3
2 units to the left and then 3 units up is y = 2x + 4.
-/
theorem move_line_left_and_up :
  ∀ (x y : ℝ), y = 2*x - 3 → ∃ x' y', x' = x + 2 ∧ y' = y + 3 ∧ y' = 2*x' + 4 :=
by
  sorry

end move_line_left_and_up_l101_101466


namespace fourth_derivative_of_function_y_l101_101479

noncomputable def log_base_3 (x : ℝ) : ℝ := (Real.log x) / (Real.log 3)

noncomputable def function_y (x : ℝ) : ℝ := (log_base_3 x) / (x ^ 2)

theorem fourth_derivative_of_function_y (x : ℝ) (h : 0 < x) : 
    (deriv^[4] (fun x => function_y x)) x = (-154 + 120 * (Real.log x)) / (x ^ 6 * Real.log 3) :=
  sorry

end fourth_derivative_of_function_y_l101_101479


namespace solution1_solution2_l101_101106

-- Definition for problem (1)
def problem1 : ℚ :=
  - (1 ^ 4 : ℚ) - (1 / 6) * (2 - (-3 : ℚ) ^ 2) / (-7 : ℚ)

theorem solution1 : problem1 = -7 / 6 :=
by
  sorry

-- Definition for problem (2)
def problem2 : ℚ :=
  ((3 / 2 : ℚ) - (5 / 8) + (7 / 12)) / (-1 / 24) - 8 * ((-1 / 2 : ℚ) ^ 3)

theorem solution2 : problem2 = -34 :=
by
  sorry

end solution1_solution2_l101_101106


namespace quadratic_equation_transformation_l101_101740

theorem quadratic_equation_transformation (x : ℝ) :
  (-5 * x ^ 2 = 2 * x + 10) →
  (x ^ 2 + (2 / 5) * x + 2 = 0) :=
by
  intro h
  sorry

end quadratic_equation_transformation_l101_101740


namespace spies_denounced_each_other_l101_101710

theorem spies_denounced_each_other :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card ≥ 10 ∧ 
  (∀ (u v : ℕ), (u, v) ∈ pairs → (v, u) ∈ pairs) :=
sorry

end spies_denounced_each_other_l101_101710


namespace fish_too_small_l101_101410

theorem fish_too_small
    (ben_fish : ℕ) (judy_fish : ℕ) (billy_fish : ℕ) (jim_fish : ℕ) (susie_fish : ℕ)
    (total_filets : ℕ) (filets_per_fish : ℕ) :
    ben_fish = 4 →
    judy_fish = 1 →
    billy_fish = 3 →
    jim_fish = 2 →
    susie_fish = 5 →
    total_filets = 24 →
    filets_per_fish = 2 →
    (ben_fish + judy_fish + billy_fish + jim_fish + susie_fish) - (total_filets / filets_per_fish) = 3 := 
by 
  intros
  sorry

end fish_too_small_l101_101410


namespace pascal_15_5th_number_l101_101589

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l101_101589


namespace students_need_to_walk_distance_l101_101711

-- Define distance variables and the relationships
def teacher_initial_distance : ℝ := 235
def xiao_ma_initial_distance : ℝ := 87
def xiao_lu_initial_distance : ℝ := 59
def xiao_zhou_initial_distance : ℝ := 26
def speed_ratio : ℝ := 1.5

-- Prove the distance x students need to walk
theorem students_need_to_walk_distance (x : ℝ) :
  teacher_initial_distance - speed_ratio * x =
  (xiao_ma_initial_distance - x) + (xiao_lu_initial_distance - x) + (xiao_zhou_initial_distance - x) →
  x = 42 :=
by
  sorry

end students_need_to_walk_distance_l101_101711


namespace beckett_younger_than_olaf_l101_101396

-- Define variables for ages
variables (O B S J : ℕ) (x : ℕ)

-- Express conditions as Lean hypotheses
def conditions :=
  B = O - x ∧  -- Beckett's age
  B = 12 ∧    -- Beckett is 12 years old
  S = O - 2 ∧ -- Shannen's age
  J = 2 * S + 5 ∧ -- Jack's age
  O + B + S + J = 71 -- Sum of ages
  
-- The theorem stating that Beckett is 8 years younger than Olaf
theorem beckett_younger_than_olaf (h : conditions O B S J x) : x = 8 :=
by
  -- The proof is omitted (using sorry)
  sorry

end beckett_younger_than_olaf_l101_101396


namespace square_side_length_on_hexagon_l101_101127

noncomputable def side_length_of_square (s : ℝ) : Prop :=
  let hexagon_side := 1
  let internal_angle := 120
  ((s * (1 + 1 / Real.sqrt 3)) = 2) → s = (3 - Real.sqrt 3)

theorem square_side_length_on_hexagon : ∃ s : ℝ, side_length_of_square s :=
by
  use 3 - Real.sqrt 3
  -- Proof to be provided
  sorry

end square_side_length_on_hexagon_l101_101127


namespace upstream_speed_proof_l101_101573

-- Definitions based on the conditions in the problem
def speed_in_still_water : ℝ := 25
def speed_downstream : ℝ := 35

-- The speed of the man rowing upstream
def speed_upstream : ℝ := speed_in_still_water - (speed_downstream - speed_in_still_water)

theorem upstream_speed_proof : speed_upstream = 15 := by
  -- Proof is omitted by using sorry
  sorry

end upstream_speed_proof_l101_101573


namespace calories_in_300g_lemonade_l101_101657

def lemonade_calories (lemon_juice_in_g : Nat) (sugar_in_g : Nat) (water_in_g : Nat) (lemon_juice_cal : Nat) (sugar_cal : Nat) : Nat :=
  (lemon_juice_in_g * lemon_juice_cal / 100) + (sugar_in_g * sugar_cal / 100)

def total_weight (lemon_juice_in_g : Nat) (sugar_in_g : Nat) (water_in_g : Nat) : Nat :=
  lemon_juice_in_g + sugar_in_g + water_in_g

theorem calories_in_300g_lemonade :
  (lemonade_calories 500 200 1000 30 400) * 300 / (total_weight 500 200 1000) = 168 := 
  by
    sorry

end calories_in_300g_lemonade_l101_101657


namespace max_area_height_l101_101064

theorem max_area_height (h : ℝ) (x : ℝ) 
  (right_trapezoid : True) 
  (angle_30_deg : True) 
  (perimeter_eq_6 : 3 * (x + h) = 6) : 
  h = 1 :=
by 
  sorry

end max_area_height_l101_101064


namespace circle_standard_equation_l101_101655

theorem circle_standard_equation:
  ∃ (x y : ℝ), ((x + 2) ^ 2 + (y - 1) ^ 2 = 4) :=
by
  sorry

end circle_standard_equation_l101_101655


namespace volume_of_pyramid_correct_l101_101988

noncomputable def volume_of_pyramid (lateral_surface_area base_area inscribed_circle_area radius : ℝ) : ℝ :=
  if lateral_surface_area = 3 * base_area ∧ inscribed_circle_area = radius then
    (2 * Real.sqrt 6) / (Real.pi ^ 3)
  else
    0

theorem volume_of_pyramid_correct
  (lateral_surface_area base_area inscribed_circle_area radius : ℝ)
  (h1 : lateral_surface_area = 3 * base_area)
  (h2 : inscribed_circle_area = radius) :
  volume_of_pyramid lateral_surface_area base_area inscribed_circle_area radius = (2 * Real.sqrt 6) / (Real.pi ^ 3) :=
by {
  sorry
}

end volume_of_pyramid_correct_l101_101988


namespace price_per_cup_l101_101342

theorem price_per_cup
  (num_trees : ℕ)
  (oranges_per_tree_g : ℕ)
  (oranges_per_tree_a : ℕ)
  (oranges_per_tree_m : ℕ)
  (oranges_per_cup : ℕ)
  (total_income : ℕ)
  (h_g : num_trees = 110)
  (h_a : oranges_per_tree_g = 600)
  (h_al : oranges_per_tree_a = 400)
  (h_m : oranges_per_tree_m = 500)
  (h_o : oranges_per_cup = 3)
  (h_income : total_income = 220000) :
  total_income / (((num_trees * oranges_per_tree_g) + (num_trees * oranges_per_tree_a) + (num_trees * oranges_per_tree_m)) / oranges_per_cup) = 4 :=
by
  repeat {sorry}

end price_per_cup_l101_101342


namespace max_black_balls_C_is_22_l101_101796

-- Define the given parameters
noncomputable def balls_A : ℕ := 100
noncomputable def black_balls_A : ℕ := 15
noncomputable def balls_B : ℕ := 50
noncomputable def balls_C : ℕ := 80
noncomputable def probability : ℚ := 101 / 600

-- Define the maximum number of black balls in box C given the conditions
theorem max_black_balls_C_is_22 (y : ℕ) (h : (1/3 * (black_balls_A / balls_A) + 1/3 * (y / balls_B) + 1/3 * (22 / balls_C)) = probability  ) :
  ∃ (x : ℕ), x ≤ 22 := sorry

end max_black_balls_C_is_22_l101_101796


namespace lcm_of_5_6_8_9_l101_101553

theorem lcm_of_5_6_8_9 : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 9 = 360 := 
by 
  sorry

end lcm_of_5_6_8_9_l101_101553


namespace test_point_third_l101_101997

def interval := (1000, 2000)
def phi := 0.618
def x1 := 1000 + phi * (2000 - 1000)
def x2 := 1000 + 2000 - x1

-- By definition and given the conditions, x3 is computed in a specific manner
def x3 := x2 + 2000 - x1

theorem test_point_third : x3 = 1764 :=
by
  -- Skipping the proof for now
  sorry

end test_point_third_l101_101997


namespace smallest_multiple_l101_101196

theorem smallest_multiple (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ m % 45 = 0 ∧ m % 60 = 0 ∧ m % 25 ≠ 0 ∧ m = n) → n = 180 :=
by
  sorry

end smallest_multiple_l101_101196


namespace probability_two_cities_less_than_8000_l101_101315

-- Define the city names
inductive City
| Bangkok | CapeTown | Honolulu | London | NewYork
deriving DecidableEq, Inhabited

-- Define the distance between cities
def distance : City → City → ℕ
| City.Bangkok, City.CapeTown  => 6300
| City.Bangkok, City.Honolulu  => 6609
| City.Bangkok, City.London    => 5944
| City.Bangkok, City.NewYork   => 8650
| City.CapeTown, City.Bangkok  => 6300
| City.CapeTown, City.Honolulu => 11535
| City.CapeTown, City.London   => 5989
| City.CapeTown, City.NewYork  => 7800
| City.Honolulu, City.Bangkok  => 6609
| City.Honolulu, City.CapeTown => 11535
| City.Honolulu, City.London   => 7240
| City.Honolulu, City.NewYork  => 4980
| City.London, City.Bangkok    => 5944
| City.London, City.CapeTown   => 5989
| City.London, City.Honolulu   => 7240
| City.London, City.NewYork    => 3470
| City.NewYork, City.Bangkok   => 8650
| City.NewYork, City.CapeTown  => 7800
| City.NewYork, City.Honolulu  => 4980
| City.NewYork, City.London    => 3470
| _, _                         => 0

-- Prove the probability
theorem probability_two_cities_less_than_8000 :
  let pairs := [(City.Bangkok, City.CapeTown), (City.Bangkok, City.Honolulu), (City.Bangkok, City.London), (City.CapeTown, City.London), (City.CapeTown, City.NewYork), (City.Honolulu, City.London), (City.Honolulu, City.NewYork), (City.London, City.NewYork)]
  (pairs.length : ℚ) / 10 = 4 / 5 :=
sorry

end probability_two_cities_less_than_8000_l101_101315


namespace problem_inequality_1_problem_inequality_2_l101_101079

theorem problem_inequality_1 (x : ℝ) (α : ℝ) (hx : x > -1) (hα : 0 < α ∧ α < 1) : 
  (1 + x) ^ α ≤ 1 + α * x :=
sorry

theorem problem_inequality_2 (x : ℝ) (α : ℝ) (hx : x > -1) (hα : α < 0 ∨ α > 1) : 
  (1 + x) ^ α ≥ 1 + α * x :=
sorry

end problem_inequality_1_problem_inequality_2_l101_101079


namespace EquivalenceStatements_l101_101729

-- Define real numbers and sets P, Q
variables {x a b c : ℝ} {P Q : Set ℝ}

-- Prove the necessary equivalences
theorem EquivalenceStatements :
  ((x > 1) → (abs x > 1)) ∧ ((∃ x, x < -1) → (abs x > 1)) ∧
  ((a ∈ P ∩ Q) ↔ (a ∈ P ∧ a ∈ Q)) ∧
  (¬ (∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0)) ∧
  (x = 1 ↔ a + b + c = 0) :=
by
  sorry

end EquivalenceStatements_l101_101729


namespace total_students_multiple_of_8_l101_101513

theorem total_students_multiple_of_8 (B G T : ℕ) (h : G = 7 * B) (ht : T = B + G) : T % 8 = 0 :=
by
  sorry

end total_students_multiple_of_8_l101_101513


namespace percent_first_shift_participating_l101_101893

variable (total_employees_in_company : ℕ)
variable (first_shift_employees : ℕ)
variable (second_shift_employees : ℕ)
variable (third_shift_employees : ℕ)
variable (second_shift_percent_participating : ℚ)
variable (third_shift_percent_participating : ℚ)
variable (overall_percent_participating : ℚ)
variable (first_shift_percent_participating : ℚ)

theorem percent_first_shift_participating :
  total_employees_in_company = 150 →
  first_shift_employees = 60 →
  second_shift_employees = 50 →
  third_shift_employees = 40 →
  second_shift_percent_participating = 0.40 →
  third_shift_percent_participating = 0.10 →
  overall_percent_participating = 0.24 →
  first_shift_percent_participating = (12 / 60) →
  first_shift_percent_participating = 0.20 := 
by 
  intros t_e f_s_e s_s_e t_s_e s_s_p_p t_s_p_p o_p_p f_s_p_p
  -- Sorry, here would be the place for the actual proof
  sorry

end percent_first_shift_participating_l101_101893


namespace average_salary_feb_mar_apr_may_l101_101192

theorem average_salary_feb_mar_apr_may 
  (average_jan_feb_mar_apr : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_months_1 : ℤ)
  (total_months_2 : ℤ)
  (total_sum_jan_apr : average_jan_feb_mar_apr * (total_months_1:ℝ) = 32000)
  (january_salary: salary_jan = 4700)
  (may_salary: salary_may = 6500)
  (total_months_1_eq: total_months_1 = 4)
  (total_months_2_eq: total_months_2 = 4):
  average_jan_feb_mar_apr * (total_months_1:ℝ) - salary_jan + salary_may/total_months_2 = 8450 :=
by
  sorry

end average_salary_feb_mar_apr_may_l101_101192


namespace Jorge_goals_total_l101_101125

theorem Jorge_goals_total : 
  let last_season_goals := 156
  let this_season_goals := 187
  last_season_goals + this_season_goals = 343 := 
by
  sorry

end Jorge_goals_total_l101_101125


namespace complement_U_P_l101_101962

def U : Set ℝ := {y | ∃ x > 1, y = Real.log x / Real.log 2}
def P : Set ℝ := {y | ∃ x > 2, y = 1 / x}

theorem complement_U_P :
  (U \ P) = Set.Ici (1 / 2) := 
by
  sorry

end complement_U_P_l101_101962


namespace smallest_four_digit_number_divisible_by_4_l101_101991

theorem smallest_four_digit_number_divisible_by_4 : 
  ∃ n : ℕ, (1000 ≤ n ∧ n < 10000) ∧ (n % 4 = 0) ∧ n = 1000 := by
  sorry

end smallest_four_digit_number_divisible_by_4_l101_101991


namespace sum_a5_a8_eq_six_l101_101194

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) := ∀ {m n : ℕ}, a (m + 1) / a m = a (n + 1) / a n

theorem sum_a5_a8_eq_six (h_seq : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 6 * a 4 + 2 * a 8 * a 5 + a 9 * a 7 = 36) :
  a 5 + a 8 = 6 := 
sorry

end sum_a5_a8_eq_six_l101_101194


namespace problem1_problem2_l101_101335

-- Definitions based on the conditions
def internal_medicine_doctors : ℕ := 12
def surgeons : ℕ := 8
def total_doctors : ℕ := internal_medicine_doctors + surgeons
def team_size : ℕ := 5

-- Problem 1: Both doctor A and B must join the team
theorem problem1 : ∃ (ways : ℕ), ways = 816 :=
  by
    let remaining_doctors := total_doctors - 2
    let choose := remaining_doctors.choose (team_size - 2)
    have h1 : choose = 816 := sorry
    exact ⟨choose, h1⟩

-- Problem 2: At least one of doctors A or B must join the team
theorem problem2 : ∃ (ways : ℕ), ways = 5661 :=
  by
    let remaining_doctors := total_doctors - 1
    let scenario1 := 2 * remaining_doctors.choose (team_size - 1)
    let scenario2 := (total_doctors - 2).choose (team_size - 2)
    let total_ways := scenario1 + scenario2
    have h2 : total_ways = 5661 := sorry
    exact ⟨total_ways, h2⟩

end problem1_problem2_l101_101335


namespace expression_value_l101_101700

theorem expression_value (x : ℝ) (h : x = 4) :
  (x^2 - 2*x - 15) / (x - 5) = 7 :=
sorry

end expression_value_l101_101700


namespace price_reduction_to_achieve_profit_l101_101368

/-- 
A certain store sells clothing that cost $45$ yuan each to purchase for $65$ yuan each.
On average, they can sell $30$ pieces per day. For each $1$ yuan price reduction, 
an additional $5$ pieces can be sold per day. Given these conditions, 
prove that to achieve a daily profit of $800$ yuan, 
the price must be reduced by $10$ yuan per piece.
-/
theorem price_reduction_to_achieve_profit :
  ∃ x : ℝ, x = 10 ∧
    let original_cost := 45
    let original_price := 65
    let original_pieces_sold := 30
    let additional_pieces_per_yuan := 5
    let target_profit := 800
    let new_profit_per_piece := (original_price - original_cost) - x
    let new_pieces_sold := original_pieces_sold + additional_pieces_per_yuan * x
    new_profit_per_piece * new_pieces_sold = target_profit :=
by {
  sorry
}

end price_reduction_to_achieve_profit_l101_101368


namespace happy_children_count_l101_101174

theorem happy_children_count (total_children sad_children neither_children total_boys total_girls happy_boys sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : sad_children = 10)
  (h3 : neither_children = 20)
  (h4 : total_boys = 18)
  (h5 : total_girls = 42)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4) :
  ∃ happy_children, happy_children = 30 :=
  sorry

end happy_children_count_l101_101174


namespace alpha_beta_inequality_l101_101746

theorem alpha_beta_inequality (α β : ℝ) (h1 : -1 < α) (h2 : α < β) (h3 : β < 1) : 
  -2 < α - β ∧ α - β < 0 := 
sorry

end alpha_beta_inequality_l101_101746


namespace max_distinct_sums_l101_101801

/-- Given 3 boys and 20 girls standing in a row, each child counts the number of girls to their 
left and the number of boys to their right and adds these two counts together. Prove that 
the maximum number of different sums that the children could have obtained is 20. -/
theorem max_distinct_sums (boys girls : ℕ) (total_children : ℕ) 
  (h_boys : boys = 3) (h_girls : girls = 20) (h_total : total_children = boys + girls) : 
  ∃ (max_sums : ℕ), max_sums = 20 := 
by 
  sorry

end max_distinct_sums_l101_101801


namespace investment_percentage_change_l101_101862

/-- 
Isabel's investment problem statement:
Given an initial investment, and percentage changes over three years,
prove that the overall percentage change in Isabel's investment is 1.2% gain.
-/
theorem investment_percentage_change (initial_investment : ℝ) (gain1 : ℝ) (loss2 : ℝ) (gain3 : ℝ) 
    (final_investment : ℝ) :
    initial_investment = 500 →
    gain1 = 0.10 →
    loss2 = 0.20 →
    gain3 = 0.15 →
    final_investment = initial_investment * (1 + gain1) * (1 - loss2) * (1 + gain3) →
    ((final_investment - initial_investment) / initial_investment) * 100 = 1.2 :=
by
  intros h_init h_gain1 h_loss2 h_gain3 h_final
  sorry

end investment_percentage_change_l101_101862


namespace compare_abc_l101_101698

noncomputable def a : ℝ := Real.sin (145 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (52 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (47 * Real.pi / 180)

theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l101_101698


namespace number_of_negative_x_l101_101266

theorem number_of_negative_x (n : ℤ) (hn : 1 ≤ n ∧ n * n < 200) : 
  ∃ m ≥ 1, m = 14 := sorry

end number_of_negative_x_l101_101266


namespace unique_solution_x_y_z_l101_101312

theorem unique_solution_x_y_z (x y z : ℕ) (h1 : Prime y) (h2 : ¬ z % 3 = 0) (h3 : ¬ z % y = 0) :
    x^3 - y^3 = z^2 ↔ (x, y, z) = (8, 7, 13) := by
  sorry

end unique_solution_x_y_z_l101_101312


namespace n_to_the_4_plus_4_to_the_n_composite_l101_101958

theorem n_to_the_4_plus_4_to_the_n_composite (n : ℕ) (h : n ≥ 2) : ¬Prime (n^4 + 4^n) := 
sorry

end n_to_the_4_plus_4_to_the_n_composite_l101_101958


namespace sum_of_xyz_l101_101257

theorem sum_of_xyz (x y z : ℝ) (h : (x - 3)^2 + (y - 4)^2 + (z - 5)^2 = 0) : x + y + z = 12 :=
sorry

end sum_of_xyz_l101_101257


namespace distinct_positive_values_count_l101_101631

theorem distinct_positive_values_count : 
  ∃ (n : ℕ), n = 33 ∧ ∀ (x : ℕ), 
    (20 ≤ x ∧ x ≤ 99 ∧ 20 ≤ 2 * x ∧ 2 * x < 200 ∧ 3 * x ≥ 200) 
    ↔ (67 ≤ x ∧ x < 100) :=
  sorry

end distinct_positive_values_count_l101_101631


namespace carnival_friends_l101_101109

theorem carnival_friends (F : ℕ) (h1 : 865 % F ≠ 0) (h2 : 873 % F = 0) : F = 3 :=
by
  -- proof is not required
  sorry

end carnival_friends_l101_101109


namespace fg_at_3_l101_101649

def f (x : ℝ) : ℝ := x - 4
def g (x : ℝ) : ℝ := x^2 + 5

theorem fg_at_3 : f (g 3) = 10 := by
  sorry

end fg_at_3_l101_101649


namespace point_on_x_axis_l101_101212

theorem point_on_x_axis : ∃ p, (p = (-2, 0) ∧ p.snd = 0) ∧
  ((p ≠ (0, 2)) ∧ (p ≠ (-2, -3)) ∧ (p ≠ (-1, -2))) :=
by
  sorry

end point_on_x_axis_l101_101212


namespace difference_brothers_l101_101359

def aaron_brothers : ℕ := 4
def bennett_brothers : ℕ := 6

theorem difference_brothers : 2 * aaron_brothers - bennett_brothers = 2 := by
  sorry

end difference_brothers_l101_101359


namespace annie_blocks_walked_l101_101044

theorem annie_blocks_walked (x : ℕ) (h1 : 7 * 2 = 14) (h2 : 2 * x + 14 = 24) : x = 5 :=
by
  sorry

end annie_blocks_walked_l101_101044


namespace dot_product_eq_one_l101_101671

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 2)

theorem dot_product_eq_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_eq_one_l101_101671


namespace faster_train_speed_l101_101128

theorem faster_train_speed (V_s : ℝ) (t : ℝ) (l : ℝ) (V_f : ℝ) : 
  V_s = 36 → t = 20 → l = 200 → V_f = V_s + (l / t) * 3.6 → V_f = 72 
  := by
    intros _ _ _ _
    sorry

end faster_train_speed_l101_101128


namespace number_of_fills_l101_101730

-- Definitions based on conditions
def needed_flour : ℚ := 4 + 3 / 4
def cup_capacity : ℚ := 1 / 3

-- The proof statement
theorem number_of_fills : (needed_flour / cup_capacity).ceil = 15 := by
  sorry

end number_of_fills_l101_101730


namespace negation_one_zero_l101_101146

theorem negation_one_zero (a b : ℝ) (h : a ≠ 0):
  ¬ (∃! x : ℝ, a * x + b = 0) ↔ (¬ ∃ x : ℝ, a * x + b = 0 ∨ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁ + b = 0 ∧ a * x₂ + b = 0) := by
sorry

end negation_one_zero_l101_101146


namespace gcd_six_digit_repeat_l101_101633

theorem gcd_six_digit_repeat (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) : 
  ∀ m : ℕ, m = 1001 * n → (gcd m 1001 = 1001) :=
by
  sorry

end gcd_six_digit_repeat_l101_101633


namespace cats_not_eating_either_l101_101871

/-- In a shelter with 80 cats, 15 cats like tuna, 60 cats like chicken, 
and 10 like both tuna and chicken, prove that 15 cats do not eat either. -/
theorem cats_not_eating_either (total_cats : ℕ) (like_tuna : ℕ) (like_chicken : ℕ) (like_both : ℕ)
    (h1 : total_cats = 80) (h2 : like_tuna = 15) (h3 : like_chicken = 60) (h4 : like_both = 10) :
    (total_cats - (like_tuna - like_both + like_chicken - like_both + like_both) = 15) := 
by
    sorry

end cats_not_eating_either_l101_101871


namespace sequence_general_term_l101_101586

noncomputable def a_n (n : ℕ) : ℝ :=
  sorry

-- The main statement
theorem sequence_general_term (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ m n : ℕ, |a (m + n) - a m - a n| ≤ 1 / (p * m + q * n)) :
  ∀ n : ℕ, a n = n :=
by
  sorry

end sequence_general_term_l101_101586


namespace find_A_l101_101083

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) (h : 100 * A + 78 - (200 + B) = 364) : A = 5 :=
by
  sorry

end find_A_l101_101083


namespace find_original_number_of_men_l101_101870

variable (M : ℕ) (W : ℕ)

-- Given conditions translated to Lean
def condition1 := M * 10 = W -- M men complete work W in 10 days
def condition2 := (M - 10) * 20 = W -- (M - 10) men complete work W in 20 days

theorem find_original_number_of_men (h1 : condition1 M W) (h2 : condition2 M W) : M = 20 :=
sorry

end find_original_number_of_men_l101_101870


namespace lindas_daughters_and_granddaughters_no_daughters_l101_101547

def number_of_people_with_no_daughters (total_daughters total_descendants daughters_with_5_daughters : ℕ) : ℕ :=
  total_descendants - (5 * daughters_with_5_daughters - total_daughters + daughters_with_5_daughters)

theorem lindas_daughters_and_granddaughters_no_daughters
  (total_daughters : ℕ)
  (total_descendants : ℕ)
  (daughters_with_5_daughters : ℕ)
  (H1 : total_daughters = 8)
  (H2 : total_descendants = 43)
  (H3 : 5 * daughters_with_5_daughters = 35)
  : number_of_people_with_no_daughters total_daughters total_descendants daughters_with_5_daughters = 36 :=
by
  -- Code to check the proof goes here.
  sorry

end lindas_daughters_and_granddaughters_no_daughters_l101_101547


namespace most_numerous_fruit_l101_101948

-- Define the number of boxes
def num_boxes_tangerines := 5
def num_boxes_apples := 3
def num_boxes_pears := 4

-- Define the number of fruits per box
def tangerines_per_box := 30
def apples_per_box := 20
def pears_per_box := 15

-- Calculate the total number of each fruit
def total_tangerines := num_boxes_tangerines * tangerines_per_box
def total_apples := num_boxes_apples * apples_per_box
def total_pears := num_boxes_pears * pears_per_box

-- State the theorem and prove it
theorem most_numerous_fruit :
  total_tangerines = 150 ∧ total_tangerines > total_apples ∧ total_tangerines > total_pears :=
by
  -- Add here the necessary calculations to verify the conditions
  sorry

end most_numerous_fruit_l101_101948


namespace student_total_marks_l101_101161

variables {M P C : ℕ}

theorem student_total_marks
  (h1 : C = P + 20)
  (h2 : (M + C) / 2 = 35) :
  M + P = 50 :=
sorry

end student_total_marks_l101_101161


namespace solve_x_l101_101856

theorem solve_x (x y : ℝ) (h1 : 3 * x - y = 7) (h2 : x + 3 * y = 16) : x = 16 := by
  sorry

end solve_x_l101_101856


namespace original_people_in_room_l101_101143

theorem original_people_in_room (x : ℕ) 
  (h1 : 3 * x / 4 - 3 * x / 20 = 16) : x = 27 :=
sorry

end original_people_in_room_l101_101143


namespace person_walking_speed_on_escalator_l101_101542

theorem person_walking_speed_on_escalator 
  (v : ℝ) 
  (escalator_speed : ℝ := 15) 
  (escalator_length : ℝ := 180) 
  (time_taken : ℝ := 10)
  (distance_eq : escalator_length = (v + escalator_speed) * time_taken) : 
  v = 3 := 
by 
  -- The proof steps will be filled in if required
  sorry

end person_walking_speed_on_escalator_l101_101542


namespace A_wins_one_prob_A_wins_at_least_2_of_3_prob_l101_101150

-- Define the probability of A and B guessing correctly
def prob_A_correct : ℚ := 5/6
def prob_B_correct : ℚ := 3/5

-- Definition of the independent events for A and B
def prob_B_incorrect : ℚ := 1 - prob_B_correct

-- The probability of A winning in one activity
def prob_A_wins_one : ℚ := prob_A_correct * prob_B_incorrect

-- Proof (statement) that A's probability of winning one activity is 1/3
theorem A_wins_one_prob :
  prob_A_wins_one = 1/3 :=
sorry

-- Binomial coefficient for choosing 2 wins out of 3 activities
def binom_coeff_n_2 (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- Probability of A winning exactly 2 out of 3 activities
def prob_A_wins_exactly_2_of_3 : ℚ :=
  binom_coeff_n_2 3 2 * prob_A_wins_one^2 * (1 - prob_A_wins_one)

-- Probability of A winning all 3 activities
def prob_A_wins_all_3 : ℚ :=
  prob_A_wins_one^3

-- The probability of A winning at least 2 out of 3 activities
def prob_A_wins_at_least_2_of_3 : ℚ :=
  prob_A_wins_exactly_2_of_3 + prob_A_wins_all_3

-- Proof (statement) that A's probability of winning at least 2 out of 3 activities is 7/27
theorem A_wins_at_least_2_of_3_prob :
  prob_A_wins_at_least_2_of_3 = 7/27 :=
sorry

end A_wins_one_prob_A_wins_at_least_2_of_3_prob_l101_101150


namespace max_min_values_of_function_l101_101705

theorem max_min_values_of_function :
  ∀ (x : ℝ), -5 ≤ 4 * Real.sin x + 3 * Real.cos x ∧ 4 * Real.sin x + 3 * Real.cos x ≤ 5 :=
by
  sorry

end max_min_values_of_function_l101_101705


namespace remainder_when_2013_divided_by_85_l101_101245

theorem remainder_when_2013_divided_by_85 : 2013 % 85 = 58 :=
by
  sorry

end remainder_when_2013_divided_by_85_l101_101245


namespace rate_of_second_batch_of_wheat_l101_101184

theorem rate_of_second_batch_of_wheat (total_cost1 cost_per_kg1 weight1 weight2 total_weight total_cost selling_price_per_kg profit_rate cost_per_kg2 : ℝ)
  (H1 : total_cost1 = cost_per_kg1 * weight1)
  (H2 : total_weight = weight1 + weight2)
  (H3 : total_cost = total_cost1 + cost_per_kg2 * weight2)
  (H4 : selling_price_per_kg = (1 + profit_rate) * total_cost / total_weight)
  (H5 : profit_rate = 0.30)
  (H6 : cost_per_kg1 = 11.50)
  (H7 : weight1 = 30)
  (H8 : weight2 = 20)
  (H9 : selling_price_per_kg = 16.38) :
  cost_per_kg2 = 14.25 :=
by
  sorry

end rate_of_second_batch_of_wheat_l101_101184


namespace rods_in_one_mile_l101_101789

-- Define the given conditions
def mile_to_chains : ℕ := 10
def chain_to_rods : ℕ := 4

-- Prove the number of rods in one mile
theorem rods_in_one_mile : (1 * mile_to_chains * chain_to_rods) = 40 := by
  sorry

end rods_in_one_mile_l101_101789


namespace infinite_very_good_pairs_l101_101552

-- Defining what it means for a pair to be "good"
def is_good (m n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → (p ∣ m ↔ p ∣ n)

-- Defining what it means for a pair to be "very good"
def is_very_good (m n : ℕ) : Prop :=
  is_good m n ∧ is_good (m + 1) (n + 1)

-- The theorem to prove: infiniteness of very good pairs
theorem infinite_very_good_pairs : Infinite {p : ℕ × ℕ | is_very_good p.1 p.2} :=
  sorry

end infinite_very_good_pairs_l101_101552


namespace jacob_total_distance_l101_101284

/- Jacob jogs at a constant rate of 4 miles per hour.
   He jogs for 2 hours, then stops to take a rest for 30 minutes.
   After the break, he continues jogging for another 1 hour.
   Prove that the total distance jogged by Jacob is 12.0 miles.
-/
theorem jacob_total_distance :
  let joggingSpeed := 4 -- in miles per hour
  let jogBeforeBreak := 2 -- in hours
  let restDuration := 0.5 -- in hours (though it does not affect the distance)
  let jogAfterBreak := 1 -- in hours
  let totalDistance := joggingSpeed * jogBeforeBreak + joggingSpeed * jogAfterBreak
  totalDistance = 12.0 := 
by
  sorry

end jacob_total_distance_l101_101284


namespace find_x_from_equation_l101_101501

/-- If (1 / 8) * 2^36 = 4^x, then x = 16.5 -/
theorem find_x_from_equation (x : ℝ) (h : (1/8) * (2:ℝ)^36 = (4:ℝ)^x) : x = 16.5 :=
by sorry

end find_x_from_equation_l101_101501


namespace ratio_of_radii_l101_101157

-- Given conditions
variables {b a c : ℝ}
variables (h1 : π * b^2 - π * c^2 = 2 * π * a^2)
variables (h2 : c = 1.5 * a)

-- Define and prove the ratio
theorem ratio_of_radii (h1: π * b^2 - π * c^2 = 2 * π * a^2) (h2: c = 1.5 * a) :
  a / b = 2 / Real.sqrt 17 :=
sorry

end ratio_of_radii_l101_101157


namespace ratio_of_speeds_l101_101176

-- Conditions
def total_distance_Eddy : ℕ := 200 + 240 + 300
def total_distance_Freddy : ℕ := 180 + 420
def total_time_Eddy : ℕ := 5
def total_time_Freddy : ℕ := 6

-- Average speeds
def avg_speed_Eddy (d t : ℕ) : ℚ := d / t
def avg_speed_Freddy (d t : ℕ) : ℚ := d / t

-- Ratio of average speeds
def ratio_speeds (s1 s2 : ℚ) : ℚ := s1 / s2

theorem ratio_of_speeds : 
  ratio_speeds (avg_speed_Eddy total_distance_Eddy total_time_Eddy) 
               (avg_speed_Freddy total_distance_Freddy total_time_Freddy) 
  = 37 / 25 := by
  -- Proof omitted
  sorry

end ratio_of_speeds_l101_101176


namespace four_cells_same_color_rectangle_l101_101549

theorem four_cells_same_color_rectangle (color : Fin 3 → Fin 7 → Bool) :
  ∃ (r₁ r₂ r₃ r₄ : Fin 3) (c₁ c₂ c₃ c₄ : Fin 7), 
    r₁ ≠ r₂ ∧ r₃ ≠ r₄ ∧ c₁ ≠ c₂ ∧ c₃ ≠ c₄ ∧ 
    r₁ = r₃ ∧ r₂ = r₄ ∧ c₁ = c₃ ∧ c₂ = c₄ ∧
    color r₁ c₁ = color r₁ c₂ ∧ color r₂ c₁ = color r₂ c₂ := sorry

end four_cells_same_color_rectangle_l101_101549


namespace yoki_cans_collected_l101_101273

theorem yoki_cans_collected (total_cans LaDonna_cans Prikya_cans Avi_cans : ℕ) (half_Avi_cans Yoki_cans : ℕ) 
    (h1 : total_cans = 85) 
    (h2 : LaDonna_cans = 25) 
    (h3 : Prikya_cans = 2 * LaDonna_cans - 3) 
    (h4 : Avi_cans = 8) 
    (h5 : half_Avi_cans = Avi_cans / 2) 
    (h6 : total_cans = LaDonna_cans + Prikya_cans + half_Avi_cans + Yoki_cans) :
    Yoki_cans = 9 := sorry

end yoki_cans_collected_l101_101273


namespace supplementary_angles_ratio_l101_101483

theorem supplementary_angles_ratio (A B : ℝ) (h1 : A + B = 180) (h2 : A / B = 5 / 4) : B = 80 :=
by
   sorry

end supplementary_angles_ratio_l101_101483


namespace borrowing_period_l101_101346

theorem borrowing_period 
  (principal : ℕ) (rate_1 : ℕ) (rate_2 : ℕ) (gain : ℕ)
  (h1 : principal = 5000)
  (h2 : rate_1 = 4)
  (h3 : rate_2 = 8)
  (h4 : gain = 200)
  : ∃ n : ℕ, n = 1 :=
by
  sorry

end borrowing_period_l101_101346


namespace total_charge_for_trip_l101_101565

noncomputable def calc_total_charge (initial_fee : ℝ) (additional_charge : ℝ) (miles : ℝ) (increment : ℝ) :=
  initial_fee + (additional_charge * (miles / increment))

theorem total_charge_for_trip :
  calc_total_charge 2.35 0.35 3.6 (2 / 5) = 8.65 :=
by
  sorry

end total_charge_for_trip_l101_101565


namespace find_number_1920_find_number_60_l101_101576

theorem find_number_1920 : 320 * 6 = 1920 :=
by sorry

theorem find_number_60 : (1920 / 7 = 60) :=
by sorry

end find_number_1920_find_number_60_l101_101576


namespace sufficient_but_not_necessary_condition_l101_101703

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x = 0 → (x^2 - 2 * x = 0)) ∧ (∃ y : ℝ, y ≠ 0 ∧ y ^ 2 - 2 * y = 0) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l101_101703


namespace full_price_tickets_revenue_l101_101098

-- Define the conditions and then prove the statement
theorem full_price_tickets_revenue (f d p : ℕ) (h1 : f + d = 200) (h2 : f * p + d * (p / 3) = 3000) : f * p = 1500 := by
  sorry

end full_price_tickets_revenue_l101_101098


namespace stratified_sampling_l101_101175

theorem stratified_sampling 
  (total_teachers : ℕ)
  (senior_teachers : ℕ)
  (intermediate_teachers : ℕ)
  (junior_teachers : ℕ)
  (sample_size : ℕ)
  (x y z : ℕ) 
  (h1 : total_teachers = 150)
  (h2 : senior_teachers = 45)
  (h3 : intermediate_teachers = 90)
  (h4 : junior_teachers = 15)
  (h5 : sample_size = 30)
  (h6 : x + y + z = sample_size)
  (h7 : x * 10 = sample_size / 5)
  (h8 : y * 10 = (2 * sample_size) / 5)
  (h9 : z * 10 = sample_size / 15) :
  (x, y, z) = (9, 18, 3) := sorry

end stratified_sampling_l101_101175


namespace find_g_l101_101173

theorem find_g (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x+1) = 3 - 2 * x) (h2 : ∀ x : ℝ, f (g x) = 6 * x - 3) : 
  ∀ x : ℝ, g x = 4 - 3 * x := 
by
  sorry

end find_g_l101_101173


namespace mode_of_data_set_l101_101572

noncomputable def data_set : List ℝ := [1, 0, -3, 5, 5, 2, -3]

theorem mode_of_data_set
  (x : ℝ)
  (h_avg : (1 + 0 - 3 + 5 + x + 2 - 3) / 7 = 1)
  (h_x : x = 5) :
  ({-3, 5} : Set ℝ) = {y : ℝ | data_set.count y = 2} :=
by
  -- Proof would go here
  sorry

end mode_of_data_set_l101_101572


namespace find_simple_interest_rate_l101_101681

variable (P : ℝ) (n : ℕ) (r_c : ℝ) (t : ℝ) (I_c : ℝ) (I_s : ℝ) (r_s : ℝ)

noncomputable def compound_interest_amount (P r_c : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r_c / n) ^ (n * t)

noncomputable def simple_interest_amount (P r_s : ℝ) (t : ℝ) : ℝ :=
  P * r_s * t

theorem find_simple_interest_rate
  (hP : P = 5000)
  (hr_c : r_c = 0.16)
  (hn : n = 2)
  (ht : t = 1)
  (hI_c : I_c = compound_interest_amount P r_c n t - P)
  (hI_s : I_s = I_c - 16)
  (hI_s_def : I_s = simple_interest_amount P r_s t) :
  r_s = 0.1632 := sorry

end find_simple_interest_rate_l101_101681


namespace batsman_average_after_17th_inning_l101_101638

-- Define the conditions and prove the required question.
theorem batsman_average_after_17th_inning (A : ℕ) (h1 : 17 * (A + 10) = 16 * A + 300) :
  (A + 10) = 140 :=
by
  sorry

end batsman_average_after_17th_inning_l101_101638


namespace tables_difference_l101_101568

theorem tables_difference (N O : ℕ) (h1 : N + O = 40) (h2 : 6 * N + 4 * O = 212) : N - O = 12 :=
sorry

end tables_difference_l101_101568


namespace solve_for_a_l101_101615

variable (a b x : ℝ)

theorem solve_for_a (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) :
  a = 3 * x := sorry

end solve_for_a_l101_101615


namespace max_value_expression_l101_101979

theorem max_value_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hsum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27 / 8 :=
sorry

end max_value_expression_l101_101979


namespace algebraic_expression_value_l101_101264

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -1) : 6 + 2 * x - 4 * y = 4 := by
  sorry

end algebraic_expression_value_l101_101264


namespace sum_two_numbers_eq_twelve_l101_101982

theorem sum_two_numbers_eq_twelve (x y : ℕ) (h1 : x^2 + y^2 = 90) (h2 : x * y = 27) : x + y = 12 :=
by
  sorry

end sum_two_numbers_eq_twelve_l101_101982


namespace solve_system_l101_101584

theorem solve_system :
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ : ℚ),
  x₁ + 12 * x₂ = 15 ∧
  x₁ - 12 * x₂ + 11 * x₃ = 2 ∧
  x₁ - 11 * x₃ + 10 * x₄ = 2 ∧
  x₁ - 10 * x₄ + 9 * x₅ = 2 ∧
  x₁ - 9 * x₅ + 8 * x₆ = 2 ∧
  x₁ - 8 * x₆ + 7 * x₇ = 2 ∧
  x₁ - 7 * x₇ + 6 * x₈ = 2 ∧
  x₁ - 6 * x₈ + 5 * x₉ = 2 ∧
  x₁ - 5 * x₉ + 4 * x₁₀ = 2 ∧
  x₁ - 4 * x₁₀ + 3 * x₁₁ = 2 ∧
  x₁ - 3 * x₁₁ + 2 * x₁₂ = 2 ∧
  x₁ - 2 * x₁₂ = 2 ∧
  x₁ = 37 / 12 ∧
  x₂ = 143 / 144 ∧
  x₃ = 65 / 66 ∧
  x₄ = 39 / 40 ∧
  x₅ = 26 / 27 ∧
  x₆ = 91 / 96 ∧
  x₇ = 13 / 14 ∧
  x₈ = 65 / 72 ∧
  x₉ = 13 / 15 ∧
  x₁₀ = 13 / 16 ∧
  x₁₁ = 13 / 18 ∧
  x₁₂ = 13 / 24 :=
by
  sorry

end solve_system_l101_101584


namespace find_b_l101_101690

theorem find_b (p : ℕ) (hp : Nat.Prime p) :
  (∃ b : ℕ, b = (p + 1) ^ 2 ∨ b = 4 * p ∧ ∀ (x1 x2 : ℤ), x1 * x2 = p * b ∧ x1 + x2 = b) → 
  (∃ b : ℕ, b = (p + 1) ^ 2 ∨ b = 4 * p) :=
by
  sorry

end find_b_l101_101690


namespace fourth_derivative_l101_101120

noncomputable def f (x : ℝ) : ℝ := (5 * x - 8) * 2^(-x)

theorem fourth_derivative (x : ℝ) : 
  deriv (deriv (deriv (deriv f))) x = 2^(-x) * (Real.log 2)^4 * (5 * x - 9) :=
sorry

end fourth_derivative_l101_101120


namespace smallest_multiple_3_4_5_l101_101457

theorem smallest_multiple_3_4_5 : ∃ (n : ℕ), (∀ (m : ℕ), (m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 0) → n ≤ m) ∧ n = 60 := 
sorry

end smallest_multiple_3_4_5_l101_101457


namespace tangent_line_ellipse_l101_101793

variables {x y x0 y0 r a b : ℝ}

/-- Given the tangent line to the circle x^2 + y^2 = r^2 at the point (x0, y0) is x0 * x + y0 * y = r^2,
we prove the tangent line to the ellipse x^2 / a^2 + y^2 / b^2 = 1 at the point (x0, y0) is x0 * x / a^2 + y0 * y / b^2 = 1. -/
theorem tangent_line_ellipse :
  (x0 * x + y0 * y = r^2) →
  (x0^2 / a^2 + y0^2 / b^2 = 1) →
  (x0 * x / a^2 + y0 * y / b^2 = 1) :=
by
  intros hc he
  sorry

end tangent_line_ellipse_l101_101793


namespace B_needs_days_l101_101639

theorem B_needs_days (A_rate B_rate Combined_rate : ℝ) (x : ℝ) (W : ℝ) (h1: A_rate = W / 140)
(h2: B_rate = W / (3 * x)) (h3 : Combined_rate = 60 * W) (h4 : Combined_rate = A_rate + B_rate) :
 x = 140 / 25197 :=
by
  sorry

end B_needs_days_l101_101639


namespace rectangle_perimeter_greater_than_16_l101_101219

theorem rectangle_perimeter_greater_than_16 (a b : ℝ) (h : a * b > 2 * a + 2 * b) (ha : a > 0) (hb : b > 0) : 
  2 * (a + b) > 16 :=
sorry

end rectangle_perimeter_greater_than_16_l101_101219


namespace function_is_zero_l101_101096

-- Define the condition that for any three points A, B, and C forming an equilateral triangle,
-- the sum of their function values is zero.
def has_equilateral_property (f : ℝ × ℝ → ℝ) : Prop :=
  ∀ (A B C : ℝ × ℝ), dist A B = 1 ∧ dist B C = 1 ∧ dist C A = 1 → 
  f A + f B + f C = 0

-- Define the theorem that states that a function with the equilateral property is identically zero.
theorem function_is_zero {f : ℝ × ℝ → ℝ} (h : has_equilateral_property f) : 
  ∀ (x : ℝ × ℝ), f x = 0 := 
by
  sorry

end function_is_zero_l101_101096


namespace team_a_faster_than_team_t_l101_101203

-- Definitions for the conditions
def course_length : ℕ := 300
def team_t_speed : ℕ := 20
def team_t_time : ℕ := course_length / team_t_speed
def team_a_time : ℕ := team_t_time - 3
def team_a_speed : ℕ := course_length / team_a_time

-- Theorem to prove
theorem team_a_faster_than_team_t :
  team_a_speed - team_t_speed = 5 :=
by
  -- Define the necessary elements based on conditions
  let course_length := 300
  let team_t_speed := 20
  let team_t_time := course_length / team_t_speed -- 15 hours
  let team_a_time := team_t_time - 3 -- 12 hours
  let team_a_speed := course_length / team_a_time -- 25 mph
  
  -- Prove the statement
  have h : team_a_speed - team_t_speed = 5 := by sorry
  exact h

end team_a_faster_than_team_t_l101_101203


namespace find_a_l101_101630

theorem find_a 
  (a : ℝ) 
  (h1 : ∀ x : ℝ, (x - 3) ^ 2 + 5 = a * x^2 + bx + c) 
  (h2 : (3, 5) = (3, a * 3 ^ 2 + b * 3 + c))
  (h3 : (-2, -20) = (-2, a * (-2)^2 + b * (-2) + c)) : a = -1 :=
by
  sorry

end find_a_l101_101630


namespace xyz_inequality_l101_101704

theorem xyz_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x / (y + z) + y / (z + x) + z / (x + y) ≥ 3 / 2) :=
sorry

end xyz_inequality_l101_101704


namespace arithmetic_sequence_condition_l101_101691

theorem arithmetic_sequence_condition (a : ℕ → ℝ) :
  (∀ n ∈ {k : ℕ | k > 0}, (a (n+1))^2 = a n * a (n+2)) ↔
  (∀ n ∈ {k : ℕ | k > 0}, a (n+1) - a n = a (n+2) - a (n+1)) ∧ ¬ (∀ n ∈ {k : ℕ | k > 0}, (a (n+1))^2 = a n * a (n+2) → a (n+1) = a n) :=
sorry

end arithmetic_sequence_condition_l101_101691


namespace transactions_Mabel_l101_101372

variable {M A C J : ℝ}

theorem transactions_Mabel (h1 : A = 1.10 * M)
                          (h2 : C = 2 / 3 * A)
                          (h3 : J = C + 18)
                          (h4 : J = 84) :
  M = 90 :=
by
  sorry

end transactions_Mabel_l101_101372


namespace turtles_on_lonely_island_l101_101467

theorem turtles_on_lonely_island (T : ℕ) (h1 : 60 = 2 * T + 10) : T = 25 := 
by sorry

end turtles_on_lonely_island_l101_101467


namespace rows_colored_red_l101_101202

theorem rows_colored_red (total_rows total_squares_per_row blue_rows green_squares red_squares_per_row red_rows : ℕ)
  (h_total_squares : total_rows * total_squares_per_row = 150)
  (h_blue_squares : blue_rows * total_squares_per_row = 60)
  (h_green_squares : green_squares = 66)
  (h_red_squares : 150 - 60 - 66 = 24)
  (h_red_rows : 24 / red_squares_per_row = 4) :
  red_rows = 4 := 
by sorry

end rows_colored_red_l101_101202


namespace find_number_l101_101218

variable (N : ℝ)

theorem find_number (h : (5 / 6) * N = (5 / 16) * N + 50) : N = 96 := 
by 
  sorry

end find_number_l101_101218


namespace two_digit_numbers_equal_three_times_product_of_digits_l101_101051

theorem two_digit_numbers_equal_three_times_product_of_digits :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ a b : ℕ, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 3 * a * b} = {15, 24} :=
by
  sorry

end two_digit_numbers_equal_three_times_product_of_digits_l101_101051


namespace pq_inequality_l101_101660

theorem pq_inequality (p : ℝ) (q : ℝ) (hp : 0 ≤ p) (hp2 : p < 2) (hq : q > 0) :
  4 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 5 * p * q) / (p + q) > 3 * p^2 * q :=
by {
  sorry
}

end pq_inequality_l101_101660


namespace quadratic_root_k_value_l101_101040

theorem quadratic_root_k_value 
  (k : ℝ) 
  (h_roots : ∀ x : ℝ, (5 * x^2 + 7 * x + k = 0) → (x = ( -7 + Real.sqrt (-191) ) / 10 ∨ x = ( -7 - Real.sqrt (-191) ) / 10)) : 
  k = 12 :=
sorry

end quadratic_root_k_value_l101_101040


namespace ratio_of_boys_to_girls_l101_101331

open Nat

theorem ratio_of_boys_to_girls
    (B G : ℕ) 
    (boys_avg : ℕ) 
    (girls_avg : ℕ) 
    (class_avg : ℕ)
    (h1 : boys_avg = 90)
    (h2 : girls_avg = 96)
    (h3 : class_avg = 94)
    (h4 : 94 * (B + G) = 90 * B + 96 * G) :
    2 * B = G :=
by
  sorry

end ratio_of_boys_to_girls_l101_101331


namespace exist_equal_success_rate_l101_101515

noncomputable def S : ℕ → ℝ := sorry -- Definition of S(N), the number of successful free throws

theorem exist_equal_success_rate (N1 N2 : ℕ) 
  (h1 : S N1 < 0.8 * N1) 
  (h2 : S N2 > 0.8 * N2) : 
  ∃ (N : ℕ), N1 ≤ N ∧ N ≤ N2 ∧ S N = 0.8 * N :=
sorry

end exist_equal_success_rate_l101_101515


namespace connie_correct_answer_l101_101679

theorem connie_correct_answer (y : ℕ) (h1 : y - 8 = 32) : y + 8 = 48 := by
  sorry

end connie_correct_answer_l101_101679


namespace no_permutation_exists_l101_101170

open Function Set

theorem no_permutation_exists (f : ℕ → ℕ) (h : ∀ n m : ℕ, f n = f m ↔ n = m) :
  ¬ ∃ n : ℕ, (Finset.range n).image f = Finset.range n :=
by
  sorry

end no_permutation_exists_l101_101170


namespace Tim_total_expenditure_l101_101280

theorem Tim_total_expenditure 
  (appetizer_price : ℝ) (main_course_price : ℝ) (dessert_price : ℝ)
  (appetizer_tip_percentage : ℝ) (main_course_tip_percentage : ℝ) (dessert_tip_percentage : ℝ) :
  appetizer_price = 12.35 →
  main_course_price = 27.50 →
  dessert_price = 9.95 →
  appetizer_tip_percentage = 0.18 →
  main_course_tip_percentage = 0.20 →
  dessert_tip_percentage = 0.15 →
  appetizer_price * (1 + appetizer_tip_percentage) + 
  main_course_price * (1 + main_course_tip_percentage) + 
  dessert_price * (1 + dessert_tip_percentage) = 12.35 * 1.18 + 27.50 * 1.20 + 9.95 * 1.15 :=
  by sorry

end Tim_total_expenditure_l101_101280


namespace height_of_table_without_book_l101_101328

-- Define the variables and assumptions
variables (l h w : ℝ) (b : ℝ := 6)

-- State the conditions from the problem
-- Condition 1: l + h - w = 40
-- Condition 2: w + h - l + b = 34

theorem height_of_table_without_book (hlw : l + h - w = 40) (whlb : w + h - l + b = 34) : h = 34 :=
by
  -- Since we are skipping the proof, we put sorry here
  sorry

end height_of_table_without_book_l101_101328


namespace lcm_hcf_product_l101_101689

theorem lcm_hcf_product (A B : ℕ) (h_prod : A * B = 18000) (h_hcf : Nat.gcd A B = 30) : Nat.lcm A B = 600 :=
sorry

end lcm_hcf_product_l101_101689


namespace number_of_handshakes_l101_101349

-- Define the context of the problem
def total_women := 8
def teams (n : Nat) := 4

-- Define the number of people each woman will shake hands with (excluding her partner)
def handshakes_per_woman := total_women - 2

-- Define the total number of handshakes
def total_handshakes := (total_women * handshakes_per_woman) / 2

-- The theorem that we're to prove
theorem number_of_handshakes : total_handshakes = 24 :=
by
  sorry

end number_of_handshakes_l101_101349


namespace distinct_positive_integers_mod_1998_l101_101865

theorem distinct_positive_integers_mod_1998
  (a : Fin 93 → ℕ)
  (h_distinct : Function.Injective a) :
  ∃ m n p q : Fin 93, (m ≠ n ∧ p ≠ q) ∧ (a m - a n) * (a p - a q) % 1998 = 0 :=
by
  sorry

end distinct_positive_integers_mod_1998_l101_101865


namespace find_first_4_hours_speed_l101_101966

noncomputable def average_speed_first_4_hours
  (total_avg_speed : ℝ)
  (first_4_hours_avg_speed : ℝ)
  (remaining_hours_avg_speed : ℝ)
  (total_time : ℕ)
  (first_4_hours : ℕ)
  (remaining_hours : ℕ) : Prop :=
  total_avg_speed * total_time = first_4_hours_avg_speed * first_4_hours + remaining_hours * remaining_hours_avg_speed

theorem find_first_4_hours_speed :
  average_speed_first_4_hours 50 35 53 24 4 20 :=
by
  sorry

end find_first_4_hours_speed_l101_101966


namespace find_younger_age_l101_101802

def younger_age (y e : ℕ) : Prop :=
  (e = y + 20) ∧ (e - 5 = 5 * (y - 5))

theorem find_younger_age (y e : ℕ) (h : younger_age y e) : y = 10 :=
by sorry

end find_younger_age_l101_101802


namespace total_vitamins_in_box_correct_vitamins_per_half_bag_correct_l101_101414

-- Define the conditions
def number_of_bags : ℕ := 9
def vitamins_per_bag : ℚ := 0.2

-- Define the total vitamins in the box
def total_vitamins_in_box : ℚ := number_of_bags * vitamins_per_bag

-- Define the vitamins intake by drinking half a bag
def vitamins_per_half_bag : ℚ := vitamins_per_bag / 2

-- Prove that the total grams of vitamins in the box is 1.8 grams
theorem total_vitamins_in_box_correct : total_vitamins_in_box = 1.8 := by
  sorry

-- Prove that the vitamins intake by drinking half a bag is 0.1 grams
theorem vitamins_per_half_bag_correct : vitamins_per_half_bag = 0.1 := by
  sorry

end total_vitamins_in_box_correct_vitamins_per_half_bag_correct_l101_101414


namespace Tyler_age_l101_101614

variable (T B S : ℕ) -- Assuming ages are non-negative integers

theorem Tyler_age (h1 : T = B - 3) (h2 : T + B + S = 25) (h3 : S = B + 2) : T = 6 := by
  sorry

end Tyler_age_l101_101614


namespace problems_completed_l101_101382

theorem problems_completed (p t : ℕ) (hp : p > 10) (eqn : p * t = (2 * p - 2) * (t - 1)) :
  p * t = 48 := 
sorry

end problems_completed_l101_101382


namespace remainder_when_x_plus_3uy_divided_by_y_eq_v_l101_101476

theorem remainder_when_x_plus_3uy_divided_by_y_eq_v
  (x y u v : ℕ) (h_pos_y : 0 < y) (h_division_algo : x = u * y + v) (h_remainder : 0 ≤ v ∧ v < y) :
  (x + 3 * u * y) % y = v :=
by
  sorry

end remainder_when_x_plus_3uy_divided_by_y_eq_v_l101_101476


namespace janek_favorite_number_l101_101575

theorem janek_favorite_number (S : Set ℕ) (n : ℕ) :
  S = {6, 8, 16, 22, 32} →
  n / 2 ∈ S →
  (n + 6) ∈ S →
  (n - 10) ∈ S →
  2 * n ∈ S →
  n = 16 := by
  sorry

end janek_favorite_number_l101_101575


namespace right_triangle_area_inscribed_circle_l101_101896

theorem right_triangle_area_inscribed_circle (r a b c : ℝ)
  (h_c : c = 6 + 7)
  (h_a : a = 6 + r)
  (h_b : b = 7 + r)
  (h_pyth : (6 + r)^2 + (7 + r)^2 = 13^2):
  (1 / 2) * (a * b) = 42 :=
by
  -- The necessary calculations have already been derived and verified
  sorry

end right_triangle_area_inscribed_circle_l101_101896


namespace initial_amount_liquid_A_l101_101642

-- Definitions and conditions
def initial_ratio (a : ℕ) (b : ℕ) := a = 4 * b
def replaced_mixture_ratio (a : ℕ) (b : ℕ) (r₀ r₁ : ℕ) := 4 * r₀ = 2 * (r₁ + 20)

-- Theorem to prove the initial amount of liquid A
theorem initial_amount_liquid_A (a b r₀ r₁ : ℕ) :
  initial_ratio a b → replaced_mixture_ratio a b r₀ r₁ → a = 16 := 
by
  sorry

end initial_amount_liquid_A_l101_101642


namespace largest_integer_x_l101_101840

theorem largest_integer_x (x : ℕ) : (1 / 4 : ℚ) + (x / 8 : ℚ) < 1 ↔ x <= 5 := sorry

end largest_integer_x_l101_101840


namespace fraction_irreducible_l101_101637

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_irreducible_l101_101637


namespace sum_of_squared_projections_l101_101191

theorem sum_of_squared_projections (a l m n : ℝ) (l_proj m_proj n_proj : ℝ)
  (h : l_proj = a * Real.cos θ)
  (h1 : m_proj = a * Real.cos (Real.pi / 3 - θ))
  (h2 : n_proj = a * Real.cos (Real.pi / 3 + θ)) :
  l_proj ^ 2 + m_proj ^ 2 + n_proj ^ 2 = 3 / 2 * a ^ 2 :=
by sorry

end sum_of_squared_projections_l101_101191


namespace geom_seq_sum_problem_l101_101953

noncomputable def geom_sum_first_n_terms (a₁ q : ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q^n) / (1 - q)

noncomputable def geom_sum_specific_terms (a₃ q : ℕ) (n m : ℕ) : ℕ :=
  a₃ * ((1 - (q^m) ^ n) / (1 - q^m))

theorem geom_seq_sum_problem :
  ∀ (a₁ q S₈₇ : ℕ),
  q = 2 →
  S₈₇ = 140 →
  geom_sum_first_n_terms a₁ q 87 = S₈₇ →
  ∃ a₃, a₃ = ((q * q) * a₁) →
  geom_sum_specific_terms a₃ q 29 3 = 80 := 
by
  intros a₁ q S₈₇ hq₁ hS₈₇ hsum
  -- Further proof would go here
  sorry

end geom_seq_sum_problem_l101_101953


namespace shortest_distance_to_left_focus_l101_101400

def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

def left_focus : ℝ × ℝ := (-5, 0)

theorem shortest_distance_to_left_focus : 
  ∃ P : ℝ × ℝ, 
  hyperbola P.1 P.2 ∧ 
  (∀ Q : ℝ × ℝ, hyperbola Q.1 Q.2 → dist Q left_focus ≥ dist P left_focus) ∧ 
  dist P left_focus = 2 :=
sorry

end shortest_distance_to_left_focus_l101_101400


namespace words_on_each_page_l101_101607

theorem words_on_each_page (p : ℕ) (h1 : p ≤ 120) (h2 : 150 * p % 221 = 210) : p = 48 :=
sorry

end words_on_each_page_l101_101607


namespace Tim_transactions_l101_101465

theorem Tim_transactions
  (Mabel_Monday : ℕ)
  (Mabel_Tuesday : ℕ := Mabel_Monday + Mabel_Monday / 10)
  (Anthony_Tuesday : ℕ := 2 * Mabel_Tuesday)
  (Cal_Tuesday : ℕ := (2 * Anthony_Tuesday) / 3)
  (Jade_Tuesday : ℕ := Cal_Tuesday + 17)
  (Isla_Wednesday : ℕ := Mabel_Tuesday + Cal_Tuesday - 12)
  (Tim_Thursday : ℕ := (Jade_Tuesday + Isla_Wednesday) * 3 / 2)
  : Tim_Thursday = 614 := by sorry

end Tim_transactions_l101_101465


namespace percentage_fullness_before_storms_l101_101719

def capacity : ℕ := 200 -- capacity in billion gallons
def water_added_by_storms : ℕ := 15 + 30 + 75 -- total water added by storms in billion gallons
def percentage_after : ℕ := 80 -- percentage of fullness after storms
def amount_of_water_after_storms : ℕ := capacity * percentage_after / 100

theorem percentage_fullness_before_storms :
  (amount_of_water_after_storms - water_added_by_storms) * 100 / capacity = 20 := by
  sorry

end percentage_fullness_before_storms_l101_101719


namespace dice_probability_l101_101387

theorem dice_probability :
  let prob_one_digit := (9:ℚ) / 20
  let prob_two_digit := (11:ℚ) / 20
  let prob := 10 * (prob_two_digit^2) * (prob_one_digit^3)
  prob = 1062889 / 128000000 := 
by 
  sorry

end dice_probability_l101_101387


namespace distinct_units_digits_of_perfect_cube_l101_101177

theorem distinct_units_digits_of_perfect_cube :
  {d : ℕ // d < 10} → 
  ∃ (s : Finset ℕ), s.card = 10 ∧ (∀ (d : ℕ), d ∈ s ↔ (∃ (m : ℕ), m < 10 ∧ d = ((m^3 % 10) : ℕ))) :=
by
  sorry

end distinct_units_digits_of_perfect_cube_l101_101177


namespace find_T_l101_101318

theorem find_T : 
  ∃ T : ℝ, (3 / 4) * (1 / 6) * T = (1 / 5) * (1 / 4) * 120 ∧ T = 48 :=
by
  sorry

end find_T_l101_101318


namespace equation_of_line_l_l101_101100

theorem equation_of_line_l
  (a : ℝ)
  (l_intersects_circle : ∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + a = 0)
  (midpoint_chord : ∃ C : ℝ × ℝ, C = (-2, 3) ∧ ∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + B.1) / 2 = C.1 ∧ (A.2 + B.2) / 2 = C.2) :
  a < 3 →
  ∃ l : ℝ × ℝ → Prop, (∀ x y : ℝ, l (x, y) ↔ x - y + 5 = 0) :=
by {
  sorry
}

end equation_of_line_l_l101_101100


namespace seventh_grade_male_students_l101_101739

theorem seventh_grade_male_students:
  ∃ x : ℤ, (48 = x + (4*x)/5 + 3) ∧ x = 25 :=
by
  sorry

end seventh_grade_male_students_l101_101739


namespace theodore_pays_10_percent_in_taxes_l101_101049

-- Defining the quantities
def num_stone_statues : ℕ := 10
def num_wooden_statues : ℕ := 20
def price_per_stone_statue : ℕ := 20
def price_per_wooden_statue : ℕ := 5
def total_earnings_after_taxes : ℕ := 270

-- Assertion: Theodore pays 10% of his earnings in taxes
theorem theodore_pays_10_percent_in_taxes :
  (num_stone_statues * price_per_stone_statue + num_wooden_statues * price_per_wooden_statue) - total_earnings_after_taxes
  = (10 * (num_stone_statues * price_per_stone_statue + num_wooden_statues * price_per_wooden_statue)) / 100 := 
by
  sorry

end theodore_pays_10_percent_in_taxes_l101_101049


namespace intersection_complement_B_and_A_l101_101741

open Set Real

def A : Set ℝ := { x | x^2 - 4 * x + 3 < 0 }
def B : Set ℝ := { x | x > 2 }
def CR_B : Set ℝ := { x | x ≤ 2 }

theorem intersection_complement_B_and_A : CR_B ∩ A = { x | 1 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_complement_B_and_A_l101_101741


namespace part1_part2_l101_101809

-- Part 1: Prove values of m and n.
theorem part1 (m n : ℝ) :
  (∀ x : ℝ, |x - m| ≤ n ↔ 0 ≤ x ∧ x ≤ 4) → m = 2 ∧ n = 2 :=
by
  intro h
  -- Proof omitted
  sorry

-- Part 2: Prove the minimum value of a + b.
theorem part2 (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b = 2) :
  a + b = (2 / a) + (2 / b) → a + b ≥ 2 * Real.sqrt 2 :=
by
  intro h
  -- Proof omitted
  sorry

end part1_part2_l101_101809


namespace purchases_per_customer_l101_101511

noncomputable def number_of_customers_in_cars (num_cars : ℕ) (customers_per_car : ℕ) : ℕ :=
  num_cars * customers_per_car

def total_sales (sports_store_sales : ℕ) (music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

theorem purchases_per_customer {num_cars : ℕ} {customers_per_car : ℕ} {sports_store_sales : ℕ} {music_store_sales : ℕ}
    (h1 : num_cars = 10)
    (h2 : customers_per_car = 5)
    (h3 : sports_store_sales = 20)
    (h4: music_store_sales = 30) :
    (total_sales sports_store_sales music_store_sales / number_of_customers_in_cars num_cars customers_per_car) = 1 :=
by
  sorry

end purchases_per_customer_l101_101511


namespace solution_set_of_linear_inequalities_l101_101344

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l101_101344


namespace cost_buses_minimize_cost_buses_l101_101799

theorem cost_buses
  (x y : ℕ) 
  (h₁ : x + y = 500)
  (h₂ : 2 * x + 3 * y = 1300) :
  x = 200 ∧ y = 300 :=
by 
  sorry

theorem minimize_cost_buses
  (m : ℕ) 
  (h₃: 15 * m + 25 * (8 - m) ≥ 180) :
  m = 2 ∧ (200 * m + 300 * (8 - m) = 2200) :=
by 
  sorry

end cost_buses_minimize_cost_buses_l101_101799


namespace fraction_of_foreign_males_l101_101901

theorem fraction_of_foreign_males
  (total_students : ℕ)
  (female_ratio : ℚ)
  (non_foreign_males : ℕ)
  (foreign_male_fraction : ℚ)
  (h1 : total_students = 300)
  (h2 : female_ratio = 2/3)
  (h3 : non_foreign_males = 90) :
  foreign_male_fraction = 1/10 :=
by
  sorry

end fraction_of_foreign_males_l101_101901


namespace tangent_from_point_to_circle_l101_101627

theorem tangent_from_point_to_circle :
  ∀ (x y : ℝ),
  (x - 6)^2 + (y - 3)^2 = 4 →
  (x = 10 → y = 0 →
    4 * x - 3 * y = 19) :=
by
  sorry

end tangent_from_point_to_circle_l101_101627


namespace race_distance_l101_101007

theorem race_distance (D : ℝ) (h1 : (D / 36) * 45 = D + 20) : D = 80 :=
by
  sorry

end race_distance_l101_101007


namespace traveling_distance_l101_101550

/-- Let D be the total distance from the dormitory to the city in kilometers.
Given the following conditions:
1. The student traveled 1/3 of the way by foot.
2. The student traveled 3/5 of the way by bus.
3. The remaining portion of the journey was covered by car, which equals 2 kilometers.
We need to prove that the total distance D is 30 kilometers. -/ 
theorem traveling_distance (D : ℕ) 
  (h1 : (1 / 3 : ℚ) * D + (3 / 5 : ℚ) * D + 2 = D) : D = 30 := 
sorry

end traveling_distance_l101_101550


namespace computer_price_in_2016_l101_101646

def price (p₀ : ℕ) (r : ℚ) (n : ℕ) : ℚ := p₀ * (r ^ (n / 4))

theorem computer_price_in_2016 :
  price 8100 (2/3 : ℚ) 16 = 1600 :=
by
  sorry

end computer_price_in_2016_l101_101646


namespace product_of_fraction_l101_101525

-- Define the repeating decimal as given in the problem
def repeating_decimal : Rat := 0.018 -- represents 0.\overline{018}

-- Define the given fraction obtained by simplifying
def simplified_fraction : Rat := 2 / 111

-- The goal is to prove that the product of the numerator and denominator of 
-- the simplified fraction of the repeating decimal is 222
theorem product_of_fraction (y : Rat) (hy : y = 0.018) (fraction_eq : y = 18 / 999) : 
  (2:ℕ) * (111:ℕ) = 222 :=
by
  sorry

end product_of_fraction_l101_101525


namespace steve_nickels_dimes_l101_101355

theorem steve_nickels_dimes (n d : ℕ) (h1 : d = n + 4) (h2 : 5 * n + 10 * d = 70) : n = 2 :=
by
  -- The proof goes here
  sorry

end steve_nickels_dimes_l101_101355


namespace total_money_made_l101_101303

def dvd_price : ℕ := 240
def dvd_quantity : ℕ := 8
def washing_machine_price : ℕ := 898

theorem total_money_made : dvd_price * dvd_quantity + washing_machine_price = 240 * 8 + 898 :=
by
  sorry

end total_money_made_l101_101303


namespace ratio_of_square_areas_l101_101101

theorem ratio_of_square_areas (y : ℝ) (hy : y > 0) : 
  (y^2 / (3 * y)^2) = 1 / 9 :=
sorry

end ratio_of_square_areas_l101_101101


namespace median_of_consecutive_integers_l101_101876

theorem median_of_consecutive_integers (sum_n : ℤ) (n : ℤ) 
  (h1 : sum_n = 6^4) (h2 : n = 36) : 
  (sum_n / n) = 36 :=
by
  sorry

end median_of_consecutive_integers_l101_101876


namespace total_journey_distance_l101_101246

theorem total_journey_distance : 
  ∃ D : ℝ, 
    (∀ (T : ℝ), T = 10) →
    ((D/2) / 21 + (D/2) / 24 = 10) →
    D = 224 := 
by
  sorry

end total_journey_distance_l101_101246


namespace find_all_f_l101_101897

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_all_f :
  (∀ x : ℝ, f x ≥ 0) ∧
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x + 2 * y^2) →
  ∃ a c : ℝ, (∀ x : ℝ, f x = x^2 + a * x + c) ∧ (a^2 - 4 * c ≤ 0) := sorry

end find_all_f_l101_101897


namespace ones_digit_of_six_power_l101_101323

theorem ones_digit_of_six_power (n : ℕ) (hn : n ≥ 1) : (6 ^ n) % 10 = 6 :=
by
  sorry

example : (6 ^ 34) % 10 = 6 :=
by
  have h : 34 ≥ 1 := by norm_num
  exact ones_digit_of_six_power 34 h

end ones_digit_of_six_power_l101_101323


namespace ten_thousands_written_correctly_ten_thousands_truncated_correctly_l101_101019

-- Definitions to be used in the proof
def ten_thousands_description := "Three thousand nine hundred seventy-six ten thousands"
def num_written : ℕ := 39760000
def truncated_num : ℕ := 3976

-- Theorems to be proven
theorem ten_thousands_written_correctly :
  (num_written = 39760000) :=
sorry

theorem ten_thousands_truncated_correctly :
  (truncated_num = 3976) :=
sorry

end ten_thousands_written_correctly_ten_thousands_truncated_correctly_l101_101019


namespace arithmetic_progression_terms_l101_101987

theorem arithmetic_progression_terms
  (n : ℕ) (a d : ℝ)
  (hn_odd : n % 2 = 1)
  (sum_odd_terms : n / 2 * (2 * a + (n / 2 - 1) * d) = 30)
  (sum_even_terms : (n / 2 - 1) * (2 * (a + d) + (n / 2 - 2) * d) = 36)
  (sum_all_terms : n / 2 * (2 * a + (n - 1) * d) = 66)
  (last_first_diff : (n - 1) * d = 12) :
  n = 9 := sorry

end arithmetic_progression_terms_l101_101987


namespace find_constant_l101_101008

theorem find_constant (t : ℝ) (constant : ℝ) :
  (x = constant - 3 * t) → (y = 2 * t - 3) → (t = 0.8) → (x = y) → constant = 1 :=
by
  intros h1 h2 h3 h4
  sorry

end find_constant_l101_101008


namespace integer_solution_count_l101_101509

theorem integer_solution_count {a b c d : ℤ} (h : a ≠ b) :
  (∀ x y : ℤ, (x + a * y + c) * (x + b * y + d) = 2 →
    ∃ a b : ℤ, (|a - b| = 1 ∨ (|a - b| = 2 ∧ (d - c) % 2 = 1))) :=
sorry

end integer_solution_count_l101_101509


namespace convert_255_to_base8_l101_101518

-- Define the conversion function from base 10 to base 8
def base10_to_base8 (n : ℕ) : ℕ :=
  let d2 := n / 64
  let r2 := n % 64
  let d1 := r2 / 8
  let r1 := r2 % 8
  d2 * 100 + d1 * 10 + r1

-- Define the specific number and base in the conditions
def num10 : ℕ := 255
def base8_result : ℕ := 377

-- The theorem stating the proof problem
theorem convert_255_to_base8 : base10_to_base8 num10 = base8_result :=
by
  -- You would provide the proof steps here
  sorry

end convert_255_to_base8_l101_101518


namespace certain_number_l101_101903

theorem certain_number (x y z : ℕ) 
  (h1 : x + y = 15) 
  (h2 : y = 7) 
  (h3 : 3 * x = z * y - 11) : 
  z = 5 :=
by sorry

end certain_number_l101_101903


namespace parallel_lines_l101_101478

theorem parallel_lines (k1 k2 l1 l2 : ℝ) :
  (∀ x, (k1 ≠ k2) -> (k1 * x + l1 ≠ k2 * x + l2)) ↔ 
  (k1 = k2 ∧ l1 ≠ l2) := 
by sorry

end parallel_lines_l101_101478


namespace tile_floor_covering_l101_101519

theorem tile_floor_covering (n : ℕ) (h1 : 10 < n) (h2 : n < 20) (h3 : ∃ x, 9 * x = n^2) : n = 12 ∨ n = 15 ∨ n = 18 := by
  sorry

end tile_floor_covering_l101_101519


namespace smallest_number_l101_101088

theorem smallest_number (b : ℕ) :
  (b % 3 = 2) ∧ (b % 4 = 2) ∧ (b % 5 = 3) → b = 38 :=
by
  sorry

end smallest_number_l101_101088


namespace intersection_of_A_and_B_l101_101038

-- Definitions representing the conditions
def setA : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def setB : Set ℝ := {x | x < 2}

-- Proof problem statement
theorem intersection_of_A_and_B : setA ∩ setB = {x | -1 < x ∧ x < 2} :=
sorry

end intersection_of_A_and_B_l101_101038


namespace mean_of_four_numbers_l101_101817

theorem mean_of_four_numbers (a b c d : ℝ) (h : (a + b + c + d + 130) / 5 = 90) : (a + b + c + d) / 4 = 80 := by
  sorry

end mean_of_four_numbers_l101_101817


namespace total_marks_math_physics_l101_101283

variable (M P C : ℕ)

theorem total_marks_math_physics (h1 : C = P + 10) (h2 : (M + C) / 2 = 35) : M + P = 60 :=
by
  sorry

end total_marks_math_physics_l101_101283


namespace at_least_one_nonnegative_l101_101041

theorem at_least_one_nonnegative
  (a1 a2 a3 a4 a5 a6 a7 a8 : ℝ)
  (h1 : a1 ≠ 0) (h2 : a2 ≠ 0) (h3 : a3 ≠ 0) (h4 : a4 ≠ 0)
  (h5 : a5 ≠ 0) (h6 : a6 ≠ 0) (h7 : a7 ≠ 0) (h8 : a8 ≠ 0)
  : (a1 * a3 + a2 * a4 ≥ 0) ∨ (a1 * a5 + a2 * a6 ≥ 0) ∨ (a1 * a7 + a2 * a8 ≥ 0) ∨
    (a3 * a5 + a4 * a6 ≥ 0) ∨ (a3 * a7 + a4 * a8 ≥ 0) ∨ (a5 * a7 + a6 * a8 ≥ 0) := 
sorry

end at_least_one_nonnegative_l101_101041


namespace third_team_pieces_l101_101326

theorem third_team_pieces (total_pieces : ℕ) (first_team : ℕ) (second_team : ℕ) (third_team : ℕ) : 
  total_pieces = 500 → first_team = 189 → second_team = 131 → third_team = total_pieces - first_team - second_team → third_team = 180 :=
by
  intros h_total h_first h_second h_third
  rw [h_total, h_first, h_second] at h_third
  exact h_third

end third_team_pieces_l101_101326


namespace eq_to_general_quadratic_l101_101946

theorem eq_to_general_quadratic (x : ℝ) : (x - 1) * (x + 1) = 1 → x^2 - 2 = 0 :=
by
  sorry

end eq_to_general_quadratic_l101_101946


namespace greatest_integer_gcd_3_l101_101406

theorem greatest_integer_gcd_3 : ∃ n, n < 100 ∧ gcd n 18 = 3 ∧ ∀ m, m < 100 ∧ gcd m 18 = 3 → m ≤ n := by
  sorry

end greatest_integer_gcd_3_l101_101406


namespace age_of_youngest_boy_l101_101238

theorem age_of_youngest_boy (average_age : ℕ) (age_proportion : ℕ → ℕ) 
  (h1 : average_age = 120) 
  (h2 : ∀ x, age_proportion x = 2 * x ∨ age_proportion x = 6 * x ∨ age_proportion x = 8 * x)
  (total_age : ℕ) 
  (h3 : total_age = 3 * average_age) :
  ∃ x, age_proportion x = 2 * x ∧ 2 * x * (3 * average_age / total_age) = 45 :=
by {
  sorry
}

end age_of_youngest_boy_l101_101238


namespace total_volume_of_five_cubes_l101_101122

theorem total_volume_of_five_cubes (edge_length : ℕ) (n : ℕ) (volume_per_cube : ℕ) (total_volume : ℕ) 
  (h1 : edge_length = 5)
  (h2 : n = 5)
  (h3 : volume_per_cube = edge_length ^ 3)
  (h4 : total_volume = n * volume_per_cube) :
  total_volume = 625 :=
sorry

end total_volume_of_five_cubes_l101_101122


namespace greatest_possible_x_l101_101117

-- Define the conditions and the target proof in Lean 4
theorem greatest_possible_x 
  (x : ℤ)  -- x is an integer
  (h : 2.134 * (10:ℝ)^x < 21000) : 
  x ≤ 3 :=
sorry

end greatest_possible_x_l101_101117


namespace smallest_integer_x_divisibility_l101_101668

theorem smallest_integer_x_divisibility :
  ∃ x : ℤ, (2 * x + 2) % 33 = 0 ∧ (2 * x + 2) % 44 = 0 ∧ (2 * x + 2) % 55 = 0 ∧ (2 * x + 2) % 666 = 0 ∧ x = 36629 := 
sorry

end smallest_integer_x_divisibility_l101_101668


namespace count_valid_n_l101_101166

theorem count_valid_n :
  ∃ (count : ℕ), count = 9 ∧ 
  (∀ (n : ℕ), 0 < n ∧ n ≤ 2000 ∧ ∃ (k : ℕ), 21 * n = k * k ↔ count = 9) :=
by
  sorry

end count_valid_n_l101_101166


namespace number_of_foxes_l101_101070

-- Define the conditions as given in the problem
def num_cows : ℕ := 20
def num_sheep : ℕ := 20
def total_animals : ℕ := 100
def num_zebras (F : ℕ) := 3 * F

-- The theorem we want to prove based on the conditions
theorem number_of_foxes (F : ℕ) :
  num_cows + num_sheep + F + num_zebras F = total_animals → F = 15 :=
by
  sorry

end number_of_foxes_l101_101070


namespace olivia_spent_38_l101_101791

def initial_amount : ℕ := 128
def amount_left : ℕ := 90
def money_spent (initial amount_left : ℕ) : ℕ := initial - amount_left

theorem olivia_spent_38 :
  money_spent initial_amount amount_left = 38 :=
by 
  sorry

end olivia_spent_38_l101_101791


namespace paco_ate_sweet_cookies_l101_101404

noncomputable def PacoCookies (sweet: Nat) (salty: Nat) (salty_eaten: Nat) (extra_sweet: Nat) : Nat :=
  let corrected_salty_eaten := if salty_eaten > salty then salty else salty_eaten
  corrected_salty_eaten + extra_sweet

theorem paco_ate_sweet_cookies : PacoCookies 39 6 23 9 = 15 := by
  sorry

end paco_ate_sweet_cookies_l101_101404


namespace at_least_six_on_circle_l101_101136

-- Defining the types for point and circle
variable (Point : Type)
variable (Circle : Type)

-- Assuming the existence of a well-defined predicate that checks whether points lie on the same circle
variable (lies_on_circle : Circle → Point → Prop)
variable (exists_circle : Point → Point → Point → Point → Circle)
variable (five_points_condition : ∀ (p1 p2 p3 p4 p5 : Point), 
  ∃ (c : Circle), lies_on_circle c p1 ∧ lies_on_circle c p2 ∧ 
                   lies_on_circle c p3 ∧ lies_on_circle c p4)

-- Given 13 points on a plane
variables (P : List Point)
variable (length_P : P.length = 13)

-- The main theorem statement
theorem at_least_six_on_circle : 
  (∀ (P : List Point) (h : P.length = 13),
    (∀ p1 p2 p3 p4 p5 : Point, ∃ (c : Circle), lies_on_circle c p1 ∧ lies_on_circle c p2 ∧ 
                               lies_on_circle c p3 ∧ lies_on_circle c p4)) →
    (∃ (c : Circle), ∃ (l : List Point), l.length ≥ 6 ∧ ∀ p ∈ l, lies_on_circle c p) :=
sorry

end at_least_six_on_circle_l101_101136


namespace sum_series_eq_four_l101_101398

noncomputable def series_sum : ℝ :=
  ∑' n : ℕ, if n = 0 then 0 else (3 * (n + 1) + 2) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 3))

theorem sum_series_eq_four :
  series_sum = 4 :=
by
  sorry

end sum_series_eq_four_l101_101398


namespace conference_end_time_correct_l101_101591

-- Define the conference conditions
def conference_start_time : ℕ := 15 * 60 -- 3:00 p.m. in minutes
def conference_duration : ℕ := 450 -- 450 minutes duration
def daylight_saving_adjustment : ℕ := 60 -- clocks set forward by one hour

-- Define the end time computation
def end_time_without_daylight_saving : ℕ := conference_start_time + conference_duration
def end_time_with_daylight_saving : ℕ := end_time_without_daylight_saving + daylight_saving_adjustment

-- Prove that the conference ended at 11:30 p.m. (11:30 p.m. in minutes is 23 * 60 + 30)
theorem conference_end_time_correct : end_time_with_daylight_saving = 23 * 60 + 30 := by
  sorry

end conference_end_time_correct_l101_101591


namespace initial_percentage_of_salt_l101_101624

theorem initial_percentage_of_salt (P : ℝ) :
  (P / 100) * 80 = 8 → P = 10 :=
by
  intro h
  sorry

end initial_percentage_of_salt_l101_101624


namespace pencils_per_student_l101_101596

theorem pencils_per_student (num_students total_pencils : ℕ)
  (h1 : num_students = 4) (h2 : total_pencils = 8) : total_pencils / num_students = 2 :=
by
  -- Proof omitted
  sorry

end pencils_per_student_l101_101596


namespace fraction_increases_l101_101813

theorem fraction_increases (a : ℝ) (h : ℝ) (ha : a > -1) (hh : h > 0) : 
  (a + h) / (a + h + 1) > a / (a + 1) := 
by 
  sorry

end fraction_increases_l101_101813


namespace alice_number_l101_101383

theorem alice_number (n : ℕ) 
  (h1 : 180 ∣ n) 
  (h2 : 75 ∣ n) 
  (h3 : 900 ≤ n) 
  (h4 : n ≤ 3000) : 
  n = 900 ∨ n = 1800 ∨ n = 2700 := 
by
  sorry

end alice_number_l101_101383


namespace max_min_PA_l101_101831

open Classical

variables (A B P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace P]
          (dist_AB : ℝ) (dist_PA_PB : ℝ)

noncomputable def max_PA (A B : Type) [MetricSpace A] [MetricSpace B] (dist_AB : ℝ) : ℝ := sorry
noncomputable def min_PA (A B : Type) [MetricSpace A] [MetricSpace B] (dist_AB : ℝ) : ℝ := sorry

theorem max_min_PA (A B : Type) [MetricSpace A] [MetricSpace B] [Inhabited P]
                   (dist_AB : ℝ) (dist_PA_PB : ℝ) :
  dist_AB = 4 → dist_PA_PB = 6 → max_PA A B 4 = 5 ∧ min_PA A B 4 = 1 :=
by
  intros h_AB h_PA_PB
  sorry

end max_min_PA_l101_101831


namespace balls_into_boxes_l101_101602

theorem balls_into_boxes :
  ∃ n : ℕ, n = 56 ∧ (∀ a b c d : ℕ, a + b + c + d = 5 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d →
    n = 4 * (b + c + d + 1)) :=
by sorry

end balls_into_boxes_l101_101602


namespace xyz_sum_56_l101_101622

theorem xyz_sum_56 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + z = 55) (h2 : y * z + x = 55) (h3 : z * x + y = 55)
  (even_cond : x % 2 = 0 ∨ y % 2 = 0 ∨ z % 2 = 0) :
  x + y + z = 56 :=
sorry

end xyz_sum_56_l101_101622


namespace johns_out_of_pocket_expense_l101_101866

theorem johns_out_of_pocket_expense :
  let computer_cost := 700
  let accessories_cost := 200
  let playstation_value := 400
  let playstation_loss_percent := 0.2
  (computer_cost + accessories_cost - playstation_value * (1 - playstation_loss_percent) = 580) :=
by {
  sorry
}

end johns_out_of_pocket_expense_l101_101866


namespace piglet_weight_l101_101046

variable (C K P L : ℝ)

theorem piglet_weight (h1 : C = K + P) (h2 : P + C = L + K) (h3 : L = 30) : P = 15 := by
  sorry

end piglet_weight_l101_101046


namespace conic_sections_ab_value_l101_101993

theorem conic_sections_ab_value
  (a b : ℝ)
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 64) :
  |a * b| = Real.sqrt 868.5 :=
by
  -- Proof will be filled in later
  sorry

end conic_sections_ab_value_l101_101993


namespace smallest_nonprime_with_large_primes_l101_101426

theorem smallest_nonprime_with_large_primes
  (n : ℕ)
  (h1 : n > 1)
  (h2 : ¬ Prime n)
  (h3 : ∀ p : ℕ, Prime p → p ∣ n → p ≥ 20) :
  660 < n ∧ n ≤ 670 :=
sorry

end smallest_nonprime_with_large_primes_l101_101426


namespace initial_oranges_is_sum_l101_101399

-- Define the number of oranges taken by Jonathan
def oranges_taken : ℕ := 45

-- Define the number of oranges left in the box
def oranges_left : ℕ := 51

-- The theorem states that the initial number of oranges is the sum of the oranges taken and those left
theorem initial_oranges_is_sum : oranges_taken + oranges_left = 96 := 
by 
  -- This is where the proof would go
  sorry

end initial_oranges_is_sum_l101_101399


namespace Michael_made_97_dollars_l101_101874

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def quantity_large : ℕ := 2
def quantity_medium : ℕ := 2
def quantity_small : ℕ := 3

def calculate_total_money (price_large price_medium price_small : ℕ) 
                           (quantity_large quantity_medium quantity_small : ℕ) : ℕ :=
  (price_large * quantity_large) + (price_medium * quantity_medium) + (price_small * quantity_small)

theorem Michael_made_97_dollars :
  calculate_total_money price_large price_medium price_small quantity_large quantity_medium quantity_small = 97 := 
by
  sorry

end Michael_made_97_dollars_l101_101874


namespace group_c_right_angled_triangle_l101_101684

theorem group_c_right_angled_triangle :
  (3^2 + 4^2 = 5^2) := by
  sorry

end group_c_right_angled_triangle_l101_101684


namespace percentage_problem_l101_101298

variable (x : ℝ)
variable (y : ℝ)

theorem percentage_problem : 
  (x / 100 * 1442 - 36 / 100 * 1412) + 63 = 252 → x = 33.52 := by
  sorry

end percentage_problem_l101_101298


namespace train_cross_time_proof_l101_101392

noncomputable def train_cross_time_opposite (L : ℝ) (v1 v2 : ℝ) (t_same : ℝ) : ℝ :=
  let speed_same := (v1 - v2) * (5/18)
  let dist_same := speed_same * t_same
  let speed_opposite := (v1 + v2) * (5/18)
  dist_same / speed_opposite

theorem train_cross_time_proof : 
  train_cross_time_opposite 69.444 50 40 50 = 5.56 :=
by
  sorry

end train_cross_time_proof_l101_101392


namespace divisor_of_1025_l101_101488

theorem divisor_of_1025 (d : ℕ) (h1: 1015 + 10 = 1025) (h2 : d ∣ 1025) : d = 5 := 
sorry

end divisor_of_1025_l101_101488


namespace min_value_of_expression_l101_101027

-- positive real numbers a and b
variables (a b : ℝ)
variables (ha : 0 < a) (hb : 0 < b)
-- given condition: 1/a + 9/b = 6
variable (h : 1 / a + 9 / b = 6)

theorem min_value_of_expression : (a + 1) * (b + 9) ≥ 16 := by
  sorry

end min_value_of_expression_l101_101027


namespace min_length_BC_l101_101233

theorem min_length_BC (A B C D : Type) (AB AC DC BD BC : ℝ) :
  AB = 8 → AC = 15 → DC = 10 → BD = 25 → (BC > AC - AB) ∧ (BC > BD - DC) → BC ≥ 15 :=
by
  intros hAB hAC hDC hBD hIneq
  sorry

end min_length_BC_l101_101233


namespace number_is_4_less_than_opposite_l101_101812

-- Define the number and its opposite relationship
def opposite_relation (x : ℤ) : Prop := x = -x + (-4)

-- Theorem stating that the given number is 4 less than its opposite
theorem number_is_4_less_than_opposite (x : ℤ) : opposite_relation x :=
sorry

end number_is_4_less_than_opposite_l101_101812


namespace chickens_rabbits_l101_101009

theorem chickens_rabbits (c r : ℕ) 
  (h1 : c = r - 20)
  (h2 : 4 * r = 6 * c + 10) :
  c = 35 := by
  sorry

end chickens_rabbits_l101_101009


namespace fraction_of_earth_surface_habitable_for_humans_l101_101389

theorem fraction_of_earth_surface_habitable_for_humans
  (total_land_fraction : ℚ) (habitable_land_fraction : ℚ)
  (h1 : total_land_fraction = 1/3)
  (h2 : habitable_land_fraction = 3/4) :
  (total_land_fraction * habitable_land_fraction) = 1/4 :=
by
  sorry

end fraction_of_earth_surface_habitable_for_humans_l101_101389


namespace even_function_implies_a_is_2_l101_101792

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l101_101792


namespace number_of_math_students_l101_101656

-- Definitions for the problem conditions
variables (total_students : ℕ) (math_class : ℕ) (physics_class : ℕ) (both_classes : ℕ)
variable (total_students_eq : total_students = 100)
variable (both_classes_eq : both_classes = 10)
variable (math_class_relation : math_class = 4 * (physics_class - both_classes + 10))

-- Theorem statement
theorem number_of_math_students (total_students : ℕ) (math_class : ℕ) (physics_class : ℕ) (both_classes : ℕ)
  (total_students_eq : total_students = 100)
  (both_classes_eq : both_classes = 10)
  (math_class_relation : math_class = 4 * (physics_class - both_classes + 10))
  (total_students_eq : total_students = physics_class + math_class - both_classes) :
  math_class = 88 :=
sorry

end number_of_math_students_l101_101656


namespace impossible_event_l101_101604

noncomputable def EventA := ∃ (ω : ℕ), ω = 0 ∨ ω = 1
noncomputable def EventB := ∃ (t : ℤ), t >= 0
noncomputable def Bag := {b : String // b = "White"}
noncomputable def EventC := ∀ (x : Bag), x.val ≠ "Red"
noncomputable def EventD := ∀ (a b : ℤ), (a > 0 ∧ b < 0) → a > b

theorem impossible_event:
  (EventA ∧ EventB ∧ EventD) →
  EventC :=
by
  sorry

end impossible_event_l101_101604


namespace fruit_bowl_l101_101714

variable {A P B : ℕ}

theorem fruit_bowl : (P = A + 2) → (B = P + 3) → (A + P + B = 19) → B = 9 :=
by
  intros h1 h2 h3
  sorry

end fruit_bowl_l101_101714


namespace rolling_a_6_on_10th_is_random_event_l101_101540

-- Definition of what it means for an event to be "random"
def is_random_event (event : ℕ → Prop) : Prop := 
  ∃ n : ℕ, event n

-- Condition: A die roll outcome for getting a 6
def die_roll_getting_6 (roll : ℕ) : Prop := 
  roll = 6

-- The main theorem to state the problem and the conclusion
theorem rolling_a_6_on_10th_is_random_event (event : ℕ → Prop) 
  (h : ∀ n, event n = die_roll_getting_6 n) : 
  is_random_event (event) := 
  sorry

end rolling_a_6_on_10th_is_random_event_l101_101540


namespace geometric_sequence_ratio_l101_101206

variables {a b c q : ℝ}

theorem geometric_sequence_ratio (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_sequence : ∃ q : ℝ, (a + b + c) * q = b + c - a ∧
                         (a + b + c) * q^2 = c + a - b ∧
                         (a + b + c) * q^3 = a + b - c) :
  q^3 + q^2 + q = 1 := 
sorry

end geometric_sequence_ratio_l101_101206


namespace graph_of_equation_l101_101906

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := by
  sorry

end graph_of_equation_l101_101906


namespace sage_reflection_day_l101_101278

theorem sage_reflection_day 
  (day_of_reflection_is_jan_1 : Prop)
  (equal_days_in_last_5_years : Prop)
  (new_year_10_years_ago_was_friday : Prop)
  (reflections_in_21st_century : Prop) : 
  ∃ (day : String), day = "Thursday" :=
by
  sorry

end sage_reflection_day_l101_101278


namespace tables_count_l101_101820

def total_tables (four_legged_tables three_legged_tables : Nat) : Nat :=
  four_legged_tables + three_legged_tables

theorem tables_count
  (four_legged_tables three_legged_tables : Nat)
  (total_legs : Nat)
  (h1 : four_legged_tables = 16)
  (h2 : total_legs = 124)
  (h3 : 4 * four_legged_tables + 3 * three_legged_tables = total_legs) :
  total_tables four_legged_tables three_legged_tables = 36 :=
by
  sorry

end tables_count_l101_101820


namespace coronavirus_transmission_l101_101787

theorem coronavirus_transmission (x : ℝ) 
  (H: (1 + x) ^ 2 = 225) : (1 + x) ^ 2 = 225 :=
  by
    sorry

end coronavirus_transmission_l101_101787


namespace Bo_needs_to_learn_per_day_l101_101759

theorem Bo_needs_to_learn_per_day
  (total_flashcards : ℕ)
  (known_percentage : ℚ)
  (days_to_learn : ℕ)
  (h1 : total_flashcards = 800)
  (h2 : known_percentage = 0.20)
  (h3 : days_to_learn = 40) : 
  total_flashcards * (1 - known_percentage) / days_to_learn = 16 := 
by
  sorry

end Bo_needs_to_learn_per_day_l101_101759


namespace total_chairs_l101_101753

theorem total_chairs (living_room_chairs kitchen_chairs : ℕ) (h1 : living_room_chairs = 3) (h2 : kitchen_chairs = 6) :
  living_room_chairs + kitchen_chairs = 9 := by
  sorry

end total_chairs_l101_101753


namespace arcade_playtime_l101_101606

noncomputable def cost_per_six_minutes : ℝ := 0.50
noncomputable def total_spent : ℝ := 15
noncomputable def minutes_per_interval : ℝ := 6
noncomputable def minutes_per_hour : ℝ := 60

theorem arcade_playtime :
  (total_spent / cost_per_six_minutes) * minutes_per_interval / minutes_per_hour = 3 :=
by
  sorry

end arcade_playtime_l101_101606


namespace censusSurveys_l101_101989

-- Definitions corresponding to the problem conditions
inductive Survey where
  | TVLifespan
  | ManuscriptReview
  | PollutionInvestigation
  | StudentSizeSurvey

open Survey

-- The aim is to identify which surveys are more suitable for a census.
def suitableForCensus (s : Survey) : Prop :=
  match s with
  | TVLifespan => False  -- Lifespan destruction implies sample survey.
  | ManuscriptReview => True  -- Significant and needs high accuracy, thus census.
  | PollutionInvestigation => False  -- Broad scope implies sample survey.
  | StudentSizeSurvey => True  -- Manageable scope makes census appropriate.

-- The theorem to be formalized.
theorem censusSurveys : (suitableForCensus ManuscriptReview) ∧ (suitableForCensus StudentSizeSurvey) :=
  by sorry

end censusSurveys_l101_101989


namespace envelopes_left_l101_101351

theorem envelopes_left (initial_envelopes : ℕ) (envelopes_per_friend : ℕ) (number_of_friends : ℕ) (remaining_envelopes : ℕ) :
  initial_envelopes = 37 → envelopes_per_friend = 3 → number_of_friends = 5 → remaining_envelopes = initial_envelopes - (envelopes_per_friend * number_of_friends) → remaining_envelopes = 22 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end envelopes_left_l101_101351


namespace min_f_x_eq_one_implies_a_eq_zero_or_two_l101_101778

theorem min_f_x_eq_one_implies_a_eq_zero_or_two (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x + a| = 1) → (a = 0 ∨ a = 2) := by
  sorry

end min_f_x_eq_one_implies_a_eq_zero_or_two_l101_101778


namespace inradius_of_right_triangle_l101_101839

theorem inradius_of_right_triangle (a b c r : ℝ) (h : a^2 + b^2 = c^2) :
  r = (1/2) * (a + b - c) :=
sorry

end inradius_of_right_triangle_l101_101839


namespace max_value_of_sinx_over_2_minus_cosx_l101_101619

theorem max_value_of_sinx_over_2_minus_cosx (x : ℝ) : 
  ∃ y_max, y_max = (Real.sqrt 3) / 3 ∧ ∀ y, y = (Real.sin x) / (2 - Real.cos x) → y ≤ y_max :=
sorry

end max_value_of_sinx_over_2_minus_cosx_l101_101619


namespace max_x_add_2y_l101_101560

theorem max_x_add_2y (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) :
  x + 2 * y ≤ 4 :=
sorry

end max_x_add_2y_l101_101560


namespace factor_polynomial_l101_101527

theorem factor_polynomial :
  ∀ (x : ℤ), 9 * (x + 3) * (x + 4) * (x + 7) * (x + 8) - 5 * x^2 = (x^2 + 4) * (9 * x^2 + 22 * x + 342) :=
by
  intro x
  sorry

end factor_polynomial_l101_101527


namespace a_4_is_zero_l101_101021

def a_n (n : ℕ) : ℕ := n^2 - 2*n - 8

theorem a_4_is_zero : a_n 4 = 0 := 
by
  sorry

end a_4_is_zero_l101_101021


namespace distinct_triangle_not_isosceles_l101_101502

theorem distinct_triangle_not_isosceles (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : 
  ¬(a = b ∨ b = c ∨ c = a) :=
by {
  sorry
}

end distinct_triangle_not_isosceles_l101_101502


namespace geometric_series_sum_150_terms_l101_101672

theorem geometric_series_sum_150_terms (a : ℕ) (r : ℝ)
  (h₁ : a = 250)
  (h₂ : (a - a * r ^ 50) / (1 - r) = 625)
  (h₃ : (a - a * r ^ 100) / (1 - r) = 1225) :
  (a - a * r ^ 150) / (1 - r) = 1801 := by
  sorry

end geometric_series_sum_150_terms_l101_101672


namespace island_width_l101_101963

theorem island_width (area length width : ℕ) (h₁ : area = 50) (h₂ : length = 10) : width = area / length := by 
  sorry

end island_width_l101_101963


namespace necessary_but_not_sufficient_ellipse_l101_101918

def is_ellipse (m : ℝ) : Prop := 
  1 < m ∧ m < 3 ∧ m ≠ 2

theorem necessary_but_not_sufficient_ellipse (m : ℝ) :
  (1 < m ∧ m < 3) → (m ≠ 2) → is_ellipse m :=
by
  intros h₁ h₂
  have h : 1 < m ∧ m < 3 ∧ m ≠ 2 := ⟨h₁.left, h₁.right, h₂⟩
  exact h

end necessary_but_not_sufficient_ellipse_l101_101918


namespace original_bet_l101_101677

-- Define conditions and question
def payout_formula (B P : ℝ) : Prop :=
  P = (3 / 2) * B

def received_payment := 60

-- Define the Lean theorem statement
theorem original_bet (B : ℝ) (h : payout_formula B received_payment) : B = 40 :=
by
  sorry

end original_bet_l101_101677


namespace simplify_fraction_l101_101221

def expr1 : ℚ := 3
def expr2 : ℚ := 2
def expr3 : ℚ := 3
def expr4 : ℚ := 4
def expected : ℚ := 12 / 5

theorem simplify_fraction : (expr1 / (expr2 - (expr3 / expr4))) = expected := by
  sorry

end simplify_fraction_l101_101221


namespace coprime_divisibility_l101_101018

theorem coprime_divisibility (p q r P Q R : ℕ)
  (hpq : Nat.gcd p q = 1) (hpr : Nat.gcd p r = 1) (hqr : Nat.gcd q r = 1)
  (h : ∃ k : ℤ, (P:ℤ) * (q*r) + (Q:ℤ) * (p*r) + (R:ℤ) * (p*q) = k * (p*q * r)) :
  ∃ a b c : ℤ, (P:ℤ) = a * (p:ℤ) ∧ (Q:ℤ) = b * (q:ℤ) ∧ (R:ℤ) = c * (r:ℤ) :=
by
  sorry

end coprime_divisibility_l101_101018


namespace LeahsCoinsValueIs68_l101_101343

def LeahsCoinsWorthInCents (p n d : Nat) : Nat :=
  p * 1 + n * 5 + d * 10

theorem LeahsCoinsValueIs68 {p n d : Nat} (h1 : p + n + d = 17) (h2 : n + 2 = p) :
  LeahsCoinsWorthInCents p n d = 68 := by
  sorry

end LeahsCoinsValueIs68_l101_101343


namespace trapezoid_combined_area_correct_l101_101928

noncomputable def combined_trapezoid_area_proof : Prop :=
  let EF : ℝ := 60
  let GH : ℝ := 40
  let altitude_EF_GH : ℝ := 18
  let trapezoid_EFGH_area : ℝ := (1 / 2) * (EF + GH) * altitude_EF_GH

  let IJ : ℝ := 30
  let KL : ℝ := 25
  let altitude_IJ_KL : ℝ := 10
  let trapezoid_IJKL_area : ℝ := (1 / 2) * (IJ + KL) * altitude_IJ_KL

  let combined_area : ℝ := trapezoid_EFGH_area + trapezoid_IJKL_area

  combined_area = 1175

theorem trapezoid_combined_area_correct : combined_trapezoid_area_proof := by
  sorry

end trapezoid_combined_area_correct_l101_101928


namespace find_m_through_point_l101_101945

theorem find_m_through_point :
  ∃ m : ℝ, ∀ (x y : ℝ), ((y = (m - 1) * x - 4) ∧ (x = 2) ∧ (y = 4)) → m = 5 :=
by 
  -- Sorry can be used here to skip the proof as instructed
  sorry

end find_m_through_point_l101_101945


namespace find_divisor_l101_101538

theorem find_divisor (x : ℤ) : 83 = 9 * x + 2 → x = 9 :=
by
  sorry

end find_divisor_l101_101538


namespace josh_remaining_marbles_l101_101747

def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7
def remaining_marbles : ℕ := 9

theorem josh_remaining_marbles : initial_marbles - lost_marbles = remaining_marbles := by
  sorry

end josh_remaining_marbles_l101_101747


namespace total_cost_of_products_l101_101805

-- Conditions
def smartphone_price := 300
def personal_computer_price := smartphone_price + 500
def advanced_tablet_price := smartphone_price + personal_computer_price

-- Theorem statement for the total cost of one of each product
theorem total_cost_of_products :
  smartphone_price + personal_computer_price + advanced_tablet_price = 2200 := by
  sorry

end total_cost_of_products_l101_101805


namespace part1_part2_l101_101912

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 1) * Real.exp x
noncomputable def h (x : ℝ) : ℝ := -3 * Real.log x + x^3 + (2 * x^2 - 4 * x) * Real.exp x + 7

theorem part1 (a : ℤ) : 
  (∀ x, (a : ℝ) < x ∧ x < a + 5 → ∀ y, (a : ℝ) < y ∧ y < a + 5 → f x ≤ f y) →
  a = -6 ∨ a = -5 ∨ a = -4 :=
sorry

theorem part2 (x : ℝ) (hx : 0 < x) : 
  f x < h x :=
sorry

end part1_part2_l101_101912


namespace evaluate_g_3_times_l101_101828

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2 * n - 1
  else 2 * n + 3

theorem evaluate_g_3_times : g (g (g 3)) = 65 := by
  sorry

end evaluate_g_3_times_l101_101828


namespace team_B_task_alone_optimal_scheduling_l101_101621

-- Condition definitions
def task_completed_in_18_months (A : Nat → Prop) : Prop := A 18
def work_together_complete_task_in_10_months (A B : Nat → Prop) : Prop := 
  ∃ n m : ℕ, n = 2 ∧ A n ∧ B m ∧ m = 10 ∧ ∀ x y : ℕ, (x / y = 1 / 18 + 1 / (n + 10))

-- Question 1
theorem team_B_task_alone (B : Nat → Prop) : ∃ x : ℕ, x = 27 := sorry

-- Conditions for the second theorem
def team_a_max_time (a : ℕ) : Prop := a ≤ 6
def team_b_max_time (b : ℕ) : Prop := b ≤ 24
def positive_integers (a b : ℕ) : Prop := a > 0 ∧ b > 0 
def total_work_done (a b : ℕ) : Prop := (a / 18) + (b / 27) = 1

-- Question 2
theorem optimal_scheduling (A B : Nat → Prop) : 
  ∃ a b : ℕ, team_a_max_time a ∧ team_b_max_time b ∧ positive_integers a b ∧
             (a / 18 + b / 27 = 1) → min_cost := sorry

end team_B_task_alone_optimal_scheduling_l101_101621


namespace distance_sum_conditions_l101_101316

theorem distance_sum_conditions (a : ℚ) (k : ℚ) :
  abs (20 * a - 20 * k - 190) = 4460 ∧ abs (20 * a^2 - 20 * k - 190) = 2755 →
  a = -37 / 2 ∨ a = 39 / 2 :=
sorry

end distance_sum_conditions_l101_101316


namespace alice_bob_speed_l101_101826

theorem alice_bob_speed (x : ℝ) (h : x = 3 + 2 * Real.sqrt 7) :
  x^2 - 5 * x - 14 = 8 + 2 * Real.sqrt 7 - 5 := by
sorry

end alice_bob_speed_l101_101826


namespace units_digit_of_8_pow_120_l101_101095

theorem units_digit_of_8_pow_120 : (8 ^ 120) % 10 = 6 := 
by
  sorry

end units_digit_of_8_pow_120_l101_101095


namespace range_of_a_l101_101920

theorem range_of_a (a : ℝ) (x : ℝ) : (x^2 + 2*x > 3) → (x > a) → (¬ (x^2 + 2*x > 3) → ¬ (x > a)) → a ≥ 1 :=
by
  intros hp hq hr
  sorry

end range_of_a_l101_101920


namespace expand_product_l101_101420

theorem expand_product (y : ℝ) : 5 * (y - 6) * (y + 9) = 5 * y^2 + 15 * y - 270 := 
by
  sorry

end expand_product_l101_101420


namespace expression_value_l101_101930

theorem expression_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) 
  (h₃ : (a^2 * b^2) / (a^4 - 2 * b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1/3 := 
by
  sorry

end expression_value_l101_101930


namespace original_number_is_115_l101_101067

-- Define the original number N, the least number to be subtracted (given), and the divisor
variable (N : ℤ) (k : ℤ)

-- State the condition based on the problem's requirements
def least_number_condition := ∃ k : ℤ, N - 28 = 87 * k

-- State the proof problem: Given the condition, prove the original number
theorem original_number_is_115 (h : least_number_condition N) : N = 115 := 
by
  sorry

end original_number_is_115_l101_101067


namespace lying_dwarf_number_is_possible_l101_101954

def dwarfs_sum (a1 a2 a3 a4 a5 a6 a7 : ℕ) : Prop :=
  a2 = a1 ∧
  a3 = a1 + a2 ∧
  a4 = a1 + a2 + a3 ∧
  a5 = a1 + a2 + a3 + a4 ∧
  a6 = a1 + a2 + a3 + a4 + a5 ∧
  a7 = a1 + a2 + a3 + a4 + a5 + a6 ∧
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = 58

theorem lying_dwarf_number_is_possible (a1 a2 a3 a4 a5 a6 a7 : ℕ) :
  dwarfs_sum a1 a2 a3 a4 a5 a6 a7 →
  (a1 = 13 ∨ a1 = 26) :=
sorry

end lying_dwarf_number_is_possible_l101_101954


namespace shorter_side_length_l101_101854

theorem shorter_side_length (L W : ℝ) (h1 : L * W = 91) (h2 : 2 * L + 2 * W = 40) :
  min L W = 7 :=
by
  sorry

end shorter_side_length_l101_101854


namespace g_at_three_l101_101205

noncomputable def g : ℝ → ℝ := sorry

axiom g_functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_nonzero_at_zero : g 0 ≠ 0
axiom g_at_one : g 1 = 2

theorem g_at_three : g 3 = 8 := sorry

end g_at_three_l101_101205


namespace time_3339_minutes_after_midnight_l101_101626

def minutes_since_midnight (minutes : ℕ) : ℕ × ℕ :=
  let hours := minutes / 60
  let remaining_minutes := minutes % 60
  (hours, remaining_minutes)

def time_after_midnight (start_time : ℕ × ℕ) (hours : ℕ) (minutes : ℕ) : ℕ × ℕ :=
  let (start_hours, start_minutes) := start_time
  let total_minutes := start_hours * 60 + start_minutes + hours * 60 + minutes
  let end_hours := total_minutes / 60
  let end_minutes := total_minutes % 60
  (end_hours, end_minutes)

theorem time_3339_minutes_after_midnight :
  time_after_midnight (0, 0) 55 39 = (7, 39) :=
by
  sorry

end time_3339_minutes_after_midnight_l101_101626


namespace c_share_correct_l101_101151

def investment_a : ℕ := 5000
def investment_b : ℕ := 15000
def investment_c : ℕ := 30000
def total_profit : ℕ := 5000

def total_investment : ℕ := investment_a + investment_b + investment_c
def c_ratio : ℚ := investment_c / total_investment
def c_share : ℚ := total_profit * c_ratio

theorem c_share_correct : c_share = 3000 := by
  sorry

end c_share_correct_l101_101151


namespace hat_cost_l101_101000

noncomputable def cost_of_hat (H : ℕ) : Prop :=
  let cost_shirts := 3 * 5
  let cost_jeans := 2 * 10
  let cost_hats := 4 * H
  let total_cost := 51
  cost_shirts + cost_jeans + cost_hats = total_cost

theorem hat_cost : ∃ H : ℕ, cost_of_hat H ∧ H = 4 :=
by 
  sorry

end hat_cost_l101_101000


namespace percentage_gain_on_powerlifting_total_l101_101403

def initialTotal : ℝ := 2200
def initialWeight : ℝ := 245
def weightIncrease : ℝ := 8
def finalWeight : ℝ := initialWeight + weightIncrease
def liftingRatio : ℝ := 10
def finalTotal : ℝ := finalWeight * liftingRatio

theorem percentage_gain_on_powerlifting_total :
  ∃ (P : ℝ), initialTotal * (1 + P / 100) = finalTotal :=
by
  sorry

end percentage_gain_on_powerlifting_total_l101_101403


namespace find_expression_roots_l101_101145

-- Define the roots of the given quadratic equation
def is_root (α : ℝ) : Prop := α ^ 2 - 2 * α - 1 = 0

-- Define the main statement to be proven
theorem find_expression_roots (α β : ℝ) (hα : is_root α) (hβ : is_root β) :
  5 * α ^ 4 + 12 * β ^ 3 = 169 := sorry

end find_expression_roots_l101_101145


namespace investment_percentage_l101_101529

theorem investment_percentage (x : ℝ) :
  (4000 * (x / 100) + 3500 * 0.04 + 2500 * 0.064 = 500) ↔ (x = 5) :=
by
  sorry

end investment_percentage_l101_101529


namespace geometric_sum_thm_l101_101129

variable (S : ℕ → ℝ)

theorem geometric_sum_thm (h1 : S n = 48) (h2 : S (2 * n) = 60) : S (3 * n) = 63 :=
sorry

end geometric_sum_thm_l101_101129


namespace probability_two_green_in_four_l101_101299

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def bag_marbles := 12
def green_marbles := 5
def blue_marbles := 3
def yellow_marbles := 4
def total_picked := 4
def green_picked := 2
def remaining_marbles := bag_marbles - green_marbles
def non_green_picked := total_picked - green_picked

theorem probability_two_green_in_four : 
  (choose green_marbles green_picked * choose remaining_marbles non_green_picked : ℚ) / (choose bag_marbles total_picked) = 14 / 33 := by
  sorry

end probability_two_green_in_four_l101_101299


namespace petya_wins_l101_101859

theorem petya_wins (n : ℕ) : n = 111 → (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → ∃ x : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ (n - k - x) % 10 = 0) → wins_optimal_play := sorry

end petya_wins_l101_101859


namespace value_of_expression_l101_101648

open Real

theorem value_of_expression {a : ℝ} (h : a^2 + 4 * a - 5 = 0) : 3 * a^2 + 12 * a = 15 :=
by sorry

end value_of_expression_l101_101648


namespace total_students_proof_l101_101520

variable (studentsA studentsB : ℕ) (ratioAtoB : ℕ := 3/2)
variable (percentA percentB : ℕ := 10/100)
variable (diffPercent : ℕ := 20/100)
variable (extraStudentsInA : ℕ := 190)
variable (totalStudentsB : ℕ := 650)

theorem total_students_proof :
  (studentsB = totalStudentsB) ∧ 
  ((percentA * studentsA - diffPercent * studentsB = extraStudentsInA) ∧
  (studentsA / studentsB = ratioAtoB)) →
  (studentsA + studentsB = 1625) :=
by
  sorry

end total_students_proof_l101_101520


namespace percent_employed_females_l101_101588

theorem percent_employed_females (total_population employed_population employed_males : ℝ)
  (h1 : employed_population = 0.6 * total_population)
  (h2 : employed_males = 0.48 * total_population) :
  ((employed_population - employed_males) / employed_population) * 100 = 20 := 
by
  sorry

end percent_employed_females_l101_101588


namespace number_of_meetings_l101_101023

-- Definitions based on the given conditions
def track_circumference : ℕ := 300
def boy1_speed : ℕ := 7
def boy2_speed : ℕ := 3
def both_start_simultaneously := true

-- The theorem to prove
theorem number_of_meetings (h1 : track_circumference = 300) (h2 : boy1_speed = 7) (h3 : boy2_speed = 3) (h4 : both_start_simultaneously) : 
  ∃ n : ℕ, n = 1 := 
sorry

end number_of_meetings_l101_101023


namespace solve_for_b_l101_101797

noncomputable def g (a b : ℝ) (x : ℝ) := 1 / (2 * a * x + 3 * b)

theorem solve_for_b (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) :
  (g a b (2) = 1 / (4 * a + 3 * b)) → (4 * a + 3 * b = 1 / 2) → b = (1 - 4 * a) / 3 :=
by
  sorry

end solve_for_b_l101_101797
