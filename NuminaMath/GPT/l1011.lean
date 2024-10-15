import Mathlib

namespace NUMINAMATH_GPT_graph_two_intersecting_lines_l1011_101113

theorem graph_two_intersecting_lines (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 3 * x * y ↔ x = 0 ∨ y = 0 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_graph_two_intersecting_lines_l1011_101113


namespace NUMINAMATH_GPT_range_of_a_l1011_101116

def set_A (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < 2 * a + 1}
def set_B : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) : (set_A a ∩ set_B = ∅) ↔ (a ≤ -2 ∨ (a > -2 ∧ a ≤ -1/2) ∨ a ≥ 2) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1011_101116


namespace NUMINAMATH_GPT_rob_has_12_pennies_l1011_101152

def total_value_in_dollars (quarters dimes nickels pennies : ℕ) : ℚ :=
  (quarters * 25 + dimes * 10 + nickels * 5 + pennies) / 100

theorem rob_has_12_pennies
  (quarters : ℕ) (dimes : ℕ) (nickels : ℕ) (pennies : ℕ)
  (h1 : quarters = 7) (h2 : dimes = 3) (h3 : nickels = 5) 
  (h4 : total_value_in_dollars quarters dimes nickels pennies = 2.42) :
  pennies = 12 :=
by
  sorry

end NUMINAMATH_GPT_rob_has_12_pennies_l1011_101152


namespace NUMINAMATH_GPT_solve_equation_l1011_101169

theorem solve_equation (x : ℝ) (hx : (x + 1) ≠ 0) :
  (x = -3 / 4) ∨ (x = -1) ↔ (x^3 + x^2 + x + 1) / (x + 1) = x^2 + 4 * x + 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1011_101169


namespace NUMINAMATH_GPT_solve_inequality_l1011_101178

theorem solve_inequality : {x : ℝ | (2 * x - 7) * (x - 3) / x ≥ 0} = {x | (0 < x ∧ x ≤ 3) ∨ (x ≥ 7 / 2)} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1011_101178


namespace NUMINAMATH_GPT_magician_decks_l1011_101183

theorem magician_decks :
  ∀ (initial_decks price_per_deck earnings decks_sold decks_left_unsold : ℕ),
  initial_decks = 5 →
  price_per_deck = 2 →
  earnings = 4 →
  decks_sold = earnings / price_per_deck →
  decks_left_unsold = initial_decks - decks_sold →
  decks_left_unsold = 3 :=
by
  intros initial_decks price_per_deck earnings decks_sold decks_left_unsold
  intros h_initial h_price h_earnings h_sold h_left
  rw [h_initial, h_price, h_earnings] at *
  sorry

end NUMINAMATH_GPT_magician_decks_l1011_101183


namespace NUMINAMATH_GPT_coat_price_reduction_l1011_101130

variable (original_price reduction : ℝ)

theorem coat_price_reduction
  (h_orig : original_price = 500)
  (h_reduct : reduction = 350)
  : reduction / original_price * 100 = 70 := 
sorry

end NUMINAMATH_GPT_coat_price_reduction_l1011_101130


namespace NUMINAMATH_GPT_zhou_yu_age_at_death_l1011_101128

theorem zhou_yu_age_at_death (x : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 9)
    (h₂ : ∃ age : ℕ, age = 10 * (x - 3) + x)
    (h₃ : x^2 = 10 * (x - 3) + x) :
    x^2 = 10 * (x - 3) + x :=
by
  sorry

end NUMINAMATH_GPT_zhou_yu_age_at_death_l1011_101128


namespace NUMINAMATH_GPT_intersecting_x_value_l1011_101124

theorem intersecting_x_value : 
  (∃ x y : ℝ, y = 3 * x - 17 ∧ 3 * x + y = 103) → 
  (∃ x : ℝ, x = 20) :=
by
  sorry

end NUMINAMATH_GPT_intersecting_x_value_l1011_101124


namespace NUMINAMATH_GPT_number_of_teams_l1011_101187

-- Define the conditions
def math_club_girls : ℕ := 4
def math_club_boys : ℕ := 7
def team_girls : ℕ := 3
def team_boys : ℕ := 3

-- Compute the number of ways to choose 3 girls from 4 girls
def choose_comb_girls : ℕ := Nat.choose math_club_girls team_girls

-- Compute the number of ways to choose 3 boys from 7 boys
def choose_comb_boys : ℕ := Nat.choose math_club_boys team_boys

-- Formulate the goal statement
theorem number_of_teams : choose_comb_girls * choose_comb_boys = 140 := by
  sorry

end NUMINAMATH_GPT_number_of_teams_l1011_101187


namespace NUMINAMATH_GPT_wade_total_spent_l1011_101189

def sandwich_cost : ℕ := 6
def drink_cost : ℕ := 4
def num_sandwiches : ℕ := 3
def num_drinks : ℕ := 2

def total_cost : ℕ :=
  (num_sandwiches * sandwich_cost) + (num_drinks * drink_cost)

theorem wade_total_spent : total_cost = 26 := by
  sorry

end NUMINAMATH_GPT_wade_total_spent_l1011_101189


namespace NUMINAMATH_GPT_imaginary_part_of_f_i_div_i_is_one_l1011_101164

def f (x : ℂ) : ℂ := x^3 - 1

theorem imaginary_part_of_f_i_div_i_is_one 
    (i : ℂ) (h : i^2 = -1) :
    ( (f i) / i ).im = 1 := 
sorry

end NUMINAMATH_GPT_imaginary_part_of_f_i_div_i_is_one_l1011_101164


namespace NUMINAMATH_GPT_quadratic_polynomial_l1011_101134

theorem quadratic_polynomial (x y : ℝ) (hx : x + y = 12) (hy : x * (3 * y) = 108) : 
  (t : ℝ) → t^2 - 12 * t + 36 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_polynomial_l1011_101134


namespace NUMINAMATH_GPT_length_of_room_l1011_101156

def area_of_room : ℝ := 10
def width_of_room : ℝ := 2

theorem length_of_room : width_of_room * 5 = area_of_room :=
by
  sorry

end NUMINAMATH_GPT_length_of_room_l1011_101156


namespace NUMINAMATH_GPT_shopkeeper_loss_percent_l1011_101197

theorem shopkeeper_loss_percent (cost_price goods_lost_percent profit_percent : ℝ)
    (h_cost_price : cost_price = 100)
    (h_goods_lost_percent : goods_lost_percent = 0.4)
    (h_profit_percent : profit_percent = 0.1) :
    let initial_revenue := cost_price * (1 + profit_percent)
    let goods_lost_value := cost_price * goods_lost_percent
    let remaining_goods_value := cost_price - goods_lost_value
    let remaining_revenue := remaining_goods_value * (1 + profit_percent)
    let loss_in_revenue := initial_revenue - remaining_revenue
    let loss_percent := (loss_in_revenue / initial_revenue) * 100
    loss_percent = 40 := sorry

end NUMINAMATH_GPT_shopkeeper_loss_percent_l1011_101197


namespace NUMINAMATH_GPT_max_frac_sum_l1011_101137

theorem max_frac_sum {n : ℕ} (h_n : n > 1) :
  ∀ (a b c d : ℕ), (a + c ≤ n) ∧ (b > 0) ∧ (d > 0) ∧
  (a * d + b * c < b * d) → 
  ↑a / ↑b + ↑c / ↑d ≤ (1 - 1 / ( ⌊(2*n : ℝ)/3 + 1/6⌋₊ + 1) * ( ⌊(2*n : ℝ)/3 + 1/6⌋₊ * (n - ⌊(2*n : ℝ)/3 + 1/6⌋₊) + 1)) :=
by sorry

end NUMINAMATH_GPT_max_frac_sum_l1011_101137


namespace NUMINAMATH_GPT_smallest_multiple_of_7_not_particular_l1011_101132

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (λ d acc => acc + d) 0

def is_particular_integer (n : ℕ) : Prop :=
  n % (sum_of_digits n) ^ 2 = 0

theorem smallest_multiple_of_7_not_particular :
  ∃ n, n > 0 ∧ n % 7 = 0 ∧ ¬ is_particular_integer n ∧ ∀ m, m > 0 ∧ m % 7 = 0 ∧ ¬ is_particular_integer m → n ≤ m :=
  by
    use 7
    sorry

end NUMINAMATH_GPT_smallest_multiple_of_7_not_particular_l1011_101132


namespace NUMINAMATH_GPT_verify_mass_percentage_l1011_101145

-- Define the elements in HBrO3
def hydrogen : String := "H"
def bromine : String := "Br"
def oxygen : String := "O"

-- Define the given molar masses
def molar_masses (e : String) : Float :=
  if e = hydrogen then 1.01
  else if e = bromine then 79.90
  else if e = oxygen then 16.00
  else 0.0

-- Define the molar mass of HBrO3
def molar_mass_HBrO3 : Float := 128.91

-- Function to calculate mass percentage of a given element in HBrO3
def mass_percentage (e : String) : Float :=
  if e = bromine then 79.90 / molar_mass_HBrO3 * 100
  else if e = hydrogen then 1.01 / molar_mass_HBrO3 * 100
  else if e = oxygen then 48.00 / molar_mass_HBrO3 * 100
  else 0.0

-- The proof problem statement
theorem verify_mass_percentage (e : String) (h : e ∈ [hydrogen, bromine, oxygen]) : mass_percentage e = 0.78 :=
sorry

end NUMINAMATH_GPT_verify_mass_percentage_l1011_101145


namespace NUMINAMATH_GPT_third_square_area_difference_l1011_101151

def side_length (p : ℕ) : ℕ :=
  p / 4

def area (s : ℕ) : ℕ :=
  s * s

theorem third_square_area_difference
  (p1 p2 p3 : ℕ)
  (h1 : p1 = 60)
  (h2 : p2 = 48)
  (h3 : p3 = 36)
  : area (side_length p3) = area (side_length p1) - area (side_length p2) :=
by
  sorry

end NUMINAMATH_GPT_third_square_area_difference_l1011_101151


namespace NUMINAMATH_GPT_tyler_age_l1011_101144

theorem tyler_age (T B : ℕ) (h1 : T = B - 3) (h2 : T + B = 11) : T = 4 :=
  sorry

end NUMINAMATH_GPT_tyler_age_l1011_101144


namespace NUMINAMATH_GPT_cos_4theta_l1011_101127

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (4 * θ) = 17 / 32 :=
sorry

end NUMINAMATH_GPT_cos_4theta_l1011_101127


namespace NUMINAMATH_GPT_total_sampled_students_l1011_101114

-- Define the total number of students in each grade
def students_in_grade12 : ℕ := 700
def students_in_grade11 : ℕ := 700
def students_in_grade10 : ℕ := 800

-- Define the number of students sampled from grade 10
def sampled_from_grade10 : ℕ := 80

-- Define the total number of students in the school
def total_students : ℕ := students_in_grade12 + students_in_grade11 + students_in_grade10

-- Prove that the total number of students sampled (x) is equal to 220
theorem total_sampled_students : 
  (sampled_from_grade10 : ℚ) / (students_in_grade10 : ℚ) * (total_students : ℚ) = 220 := 
by
  sorry

end NUMINAMATH_GPT_total_sampled_students_l1011_101114


namespace NUMINAMATH_GPT_find_x_l1011_101150

theorem find_x (x : ℕ) (hv1 : x % 6 = 0) (hv2 : x^2 > 144) (hv3 : x < 30) : x = 18 ∨ x = 24 :=
  sorry

end NUMINAMATH_GPT_find_x_l1011_101150


namespace NUMINAMATH_GPT_determine_constants_and_sum_l1011_101146

theorem determine_constants_and_sum (A B C x : ℝ) (h₁ : A = 3) (h₂ : B = 5) (h₃ : C = 40 / 3)
  (h₄ : (x + B) * (A * x + 40) / ((x + C) * (x + 5)) = 3) :
  ∀ x : ℝ, x ≠ -5 → x ≠ -40 / 3 → (-(5 : ℝ) + -40 / 3 = -55 / 3) :=
sorry

end NUMINAMATH_GPT_determine_constants_and_sum_l1011_101146


namespace NUMINAMATH_GPT_remaining_pages_l1011_101112

theorem remaining_pages (total_pages : ℕ) (science_project_percentage : ℕ) (math_homework_pages : ℕ)
  (h1 : total_pages = 120)
  (h2 : science_project_percentage = 25) 
  (h3 : math_homework_pages = 10) : 
  total_pages - (total_pages * science_project_percentage / 100) - math_homework_pages = 80 := by
  sorry

end NUMINAMATH_GPT_remaining_pages_l1011_101112


namespace NUMINAMATH_GPT_Ryan_has_28_marbles_l1011_101139

theorem Ryan_has_28_marbles :
  ∃ R : ℕ, (12 + R) - (1/4 * (12 + R)) * 2 = 20 ∧ R = 28 :=
by
  sorry

end NUMINAMATH_GPT_Ryan_has_28_marbles_l1011_101139


namespace NUMINAMATH_GPT_solution_set_inequality_l1011_101181

theorem solution_set_inequality (x : ℝ) :
  (3 * x + 2 ≥ 1 ∧ (5 - x) / 2 < 0) ↔ (-1 / 3 ≤ x ∧ x < 5) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1011_101181


namespace NUMINAMATH_GPT_quadratic_function_has_specific_k_l1011_101170

theorem quadratic_function_has_specific_k (k : ℤ) :
  (∀ x : ℝ, ∃ y : ℝ, y = (k-1)*x^(k^2-k+2) + k*x - 1) ↔ k = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_has_specific_k_l1011_101170


namespace NUMINAMATH_GPT_simplify_sqrt_neg2_squared_l1011_101167

theorem simplify_sqrt_neg2_squared : 
  Real.sqrt ((-2 : ℝ)^2) = 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_neg2_squared_l1011_101167


namespace NUMINAMATH_GPT_benny_number_of_days_worked_l1011_101133

-- Define the conditions
def total_hours_worked : ℕ := 18
def hours_per_day : ℕ := 3

-- Define the problem statement in Lean
theorem benny_number_of_days_worked : (total_hours_worked / hours_per_day) = 6 := 
by
  sorry

end NUMINAMATH_GPT_benny_number_of_days_worked_l1011_101133


namespace NUMINAMATH_GPT_cost_price_approx_l1011_101117

noncomputable def cost_price (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  selling_price / (1 + profit_percent / 100)

theorem cost_price_approx :
  ∀ (selling_price profit_percent : ℝ),
  selling_price = 2552.36 →
  profit_percent = 6 →
  abs (cost_price selling_price profit_percent - 2407.70) < 0.01 :=
by
  intros selling_price profit_percent h1 h2
  sorry

end NUMINAMATH_GPT_cost_price_approx_l1011_101117


namespace NUMINAMATH_GPT_money_left_over_l1011_101173

def initial_amount : ℕ := 120
def sandwich_fraction : ℚ := 1 / 5
def museum_ticket_fraction : ℚ := 1 / 6
def book_fraction : ℚ := 1 / 2

theorem money_left_over :
  let sandwich_cost := initial_amount * sandwich_fraction
  let museum_ticket_cost := initial_amount * museum_ticket_fraction
  let book_cost := initial_amount * book_fraction
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  initial_amount - total_spent = 16 :=
by
  sorry

end NUMINAMATH_GPT_money_left_over_l1011_101173


namespace NUMINAMATH_GPT_slope_parallel_to_line_l1011_101186

theorem slope_parallel_to_line (x y : ℝ) (h : 3 * x - 6 * y = 15) :
  (∃ m, (∀ b, y = m * x + b) ∧ (∀ k, k ≠ m → ¬ 3 * x - 6 * (k * x + b) = 15)) →
  ∃ p, p = 1/2 :=
sorry

end NUMINAMATH_GPT_slope_parallel_to_line_l1011_101186


namespace NUMINAMATH_GPT_range_of_a_l1011_101119

noncomputable def f (a x : ℝ) : ℝ := (Real.log (x^2 - a * x + 5)) / (Real.log a)

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (ha0 : 0 < a) (ha1 : a ≠ 1) 
  (hx₁x₂ : x₁ < x₂) (hx₂ : x₂ ≤ a / 2) 
  (hf : (f a x₂ - f a x₁ < 0)) : 
  1 < a ∧ a < 2 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1011_101119


namespace NUMINAMATH_GPT_percentage_calculation_l1011_101158

theorem percentage_calculation (P : ℝ) : 
    (P / 100) * 24 + 0.10 * 40 = 5.92 ↔ P = 8 :=
by 
    sorry

end NUMINAMATH_GPT_percentage_calculation_l1011_101158


namespace NUMINAMATH_GPT_sandy_money_taken_l1011_101195

-- Condition: Let T be the total money Sandy took for shopping, and it is known that 70% * T = $224
variable (T : ℝ)
axiom h : 0.70 * T = 224

-- Theorem to prove: T is 320
theorem sandy_money_taken : T = 320 :=
by 
  sorry

end NUMINAMATH_GPT_sandy_money_taken_l1011_101195


namespace NUMINAMATH_GPT_plains_total_square_miles_l1011_101148

theorem plains_total_square_miles (RegionB : ℝ) (h1 : RegionB = 200) (RegionA : ℝ) (h2 : RegionA = RegionB - 50) : 
  RegionA + RegionB = 350 := 
by 
  sorry

end NUMINAMATH_GPT_plains_total_square_miles_l1011_101148


namespace NUMINAMATH_GPT_slower_speed_l1011_101120

theorem slower_speed (x : ℝ) (h_walk_faster : 12 * (100 / x) - 100 = 20) : x = 10 :=
by sorry

end NUMINAMATH_GPT_slower_speed_l1011_101120


namespace NUMINAMATH_GPT_number_of_boys_l1011_101129

theorem number_of_boys (x : ℕ) (boys girls : ℕ)
  (initialRatio : girls / boys = 5 / 6)
  (afterLeavingRatio : (girls - 20) / boys = 2 / 3) :
  boys = 120 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_number_of_boys_l1011_101129


namespace NUMINAMATH_GPT_arithmetic_sequence_second_term_l1011_101190

theorem arithmetic_sequence_second_term (a1 a5 : ℝ) (h1 : a1 = 2020) (h5 : a5 = 4040) : 
  ∃ d a2 : ℝ, a2 = a1 + d ∧ d = (a5 - a1) / 4 ∧ a2 = 2525 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_second_term_l1011_101190


namespace NUMINAMATH_GPT_triangle_area_l1011_101191

theorem triangle_area :
  ∃ (a b c : ℕ), a + b + c = 12 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ ∃ (A : ℝ), 
  A = Real.sqrt (6 * (6 - a) * (6 - b) * (6 - c)) ∧ A = 6 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l1011_101191


namespace NUMINAMATH_GPT_length_other_diagonal_l1011_101172

variables (d1 d2 : ℝ) (Area : ℝ)

theorem length_other_diagonal 
  (h1 : Area = 432)
  (h2 : d1 = 36) :
  d2 = 24 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_length_other_diagonal_l1011_101172


namespace NUMINAMATH_GPT_dot_product_is_constant_l1011_101121

-- Define the trajectory C as the parabola given by the equation y^2 = 4x
def trajectory (x y : ℝ) : Prop := y^2 = 4 * x

-- Prove the range of k for the line passing through point (-1, 0) and intersecting trajectory C
def valid_slope (k : ℝ) : Prop := (-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1)

-- Prove that ∀ D ≠ A, B on the parabola y^2 = 4x, and lines DA and DB intersect vertical line through (1, 0) on points P, Q, OP ⋅ OQ = 5
theorem dot_product_is_constant (D A B P Q : ℝ × ℝ) 
  (hD : trajectory D.1 D.2)
  (hA : trajectory A.1 A.2)
  (hB : trajectory B.1 B.2)
  (hDiff : D ≠ A ∧ D ≠ B)
  (hP : P = (1, (D.2 * A.2 + 4) / (D.2 + A.2))) 
  (hQ : Q = (1, (D.2 * B.2 + 4) / (D.2 + B.2))) :
  (1 + (D.2 * A.2 + 4) / (D.2 + A.2)) * (1 + (D.2 * B.2 + 4) / (D.2 + B.2)) = 5 :=
sorry

end NUMINAMATH_GPT_dot_product_is_constant_l1011_101121


namespace NUMINAMATH_GPT_boats_equation_correct_l1011_101199

theorem boats_equation_correct (x : ℕ) (h1 : x ≤ 8) (h2 : 4 * x + 6 * (8 - x) = 38) : 
    4 * x + 6 * (8 - x) = 38 :=
by
  sorry

end NUMINAMATH_GPT_boats_equation_correct_l1011_101199


namespace NUMINAMATH_GPT_ratio_M_N_l1011_101140

variables {M Q P N R : ℝ}

-- Conditions
def condition1 : M = 0.40 * Q := sorry
def condition2 : Q = 0.25 * P := sorry
def condition3 : N = 0.75 * R := sorry
def condition4 : R = 0.60 * P := sorry

-- Theorem to prove
theorem ratio_M_N : M / N = 2 / 9 := sorry

end NUMINAMATH_GPT_ratio_M_N_l1011_101140


namespace NUMINAMATH_GPT_probability_of_event_A_l1011_101101

/-- The events A and B are independent, and it is given that:
  1. P(A) > 0
  2. P(A) = 2 * P(B)
  3. P(A or B) = 8 * P(A and B)

We need to prove that P(A) = 1/3. 
-/
theorem probability_of_event_A (P_A P_B : ℝ) (hP_indep : P_A * P_B = P_A) 
  (hP_A_pos : P_A > 0) (hP_A_eq_2P_B : P_A = 2 * P_B) 
  (hP_or_eq_8P_and : P_A + P_B - P_A * P_B = 8 * P_A * P_B) : 
  P_A = 1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_event_A_l1011_101101


namespace NUMINAMATH_GPT_line_equation_l1011_101135

open Real

theorem line_equation (x y : Real) : 
  (3 * x + 2 * y - 1 = 0) ↔ (y = (-(3 / 2)) * x + 2.5) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l1011_101135


namespace NUMINAMATH_GPT_power_sums_fifth_l1011_101106

noncomputable def compute_power_sums (α β γ : ℂ) : ℂ :=
  α^5 + β^5 + γ^5

theorem power_sums_fifth (α β γ : ℂ)
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 10) :
  compute_power_sums α β γ = 47.2 :=
sorry

end NUMINAMATH_GPT_power_sums_fifth_l1011_101106


namespace NUMINAMATH_GPT_cost_per_foot_of_fence_l1011_101174

theorem cost_per_foot_of_fence 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h_area : area = 289) 
  (h_total_cost : total_cost = 4080) 
  : total_cost / (4 * (Real.sqrt area)) = 60 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_foot_of_fence_l1011_101174


namespace NUMINAMATH_GPT_rita_bought_4_pounds_l1011_101194

-- Define the conditions
def card_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def amount_left : ℝ := 35.68

-- Define the theorem to prove the number of pounds of coffee bought is 4
theorem rita_bought_4_pounds :
  (card_amount - amount_left) / cost_per_pound = 4 := by sorry

end NUMINAMATH_GPT_rita_bought_4_pounds_l1011_101194


namespace NUMINAMATH_GPT_village_population_500_l1011_101111

variable (n : ℝ) -- Define the variable for population increase
variable (initial_population : ℝ) -- Define the variable for the initial population

-- Conditions from the problem
def first_year_increase : Prop := initial_population * (3 : ℝ) = n
def initial_population_def : Prop := initial_population = n / 3
def second_year_increase_def := ((n / 3 + n) * (n / 100 )) = 300

-- Define the final population formula
def population_after_two_years : ℝ := (initial_population + n + 300)

theorem village_population_500 (n : ℝ) (initial_population: ℝ) :
  first_year_increase n initial_population →
  initial_population_def n initial_population →
  second_year_increase_def n →
  population_after_two_years n initial_population = 500 :=
by sorry

#check village_population_500

end NUMINAMATH_GPT_village_population_500_l1011_101111


namespace NUMINAMATH_GPT_diamond_calculation_l1011_101108

def diamond (a b : ℚ) : ℚ := (a - b) / (1 + a * b)

theorem diamond_calculation : diamond 1 (diamond 2 (diamond 3 (diamond 4 5))) = 87 / 59 :=
by
  sorry

end NUMINAMATH_GPT_diamond_calculation_l1011_101108


namespace NUMINAMATH_GPT_second_polygon_sides_l1011_101103

theorem second_polygon_sides (s : ℝ) (n : ℝ) (h1 : 50 * 3 * s = n * s) : n = 150 := 
by
  sorry

end NUMINAMATH_GPT_second_polygon_sides_l1011_101103


namespace NUMINAMATH_GPT_smallest_sum_of_five_consecutive_primes_divisible_by_three_l1011_101109

-- Definition of the conditions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (a b c d e : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧
  (b = a + 1 ∨ b = a + 2) ∧ (c = b + 1 ∨ c = b + 2) ∧
  (d = c + 1 ∨ d = c + 2) ∧ (e = d + 1 ∨ e = d + 2)

theorem smallest_sum_of_five_consecutive_primes_divisible_by_three :
  ∃ a b c d e, consecutive_primes a b c d e ∧ a + b + c + d + e = 39 ∧ 39 % 3 = 0 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_five_consecutive_primes_divisible_by_three_l1011_101109


namespace NUMINAMATH_GPT_min_value_x_3y_6z_l1011_101198

theorem min_value_x_3y_6z (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0 ∧ xyz = 27) : x + 3 * y + 6 * z ≥ 27 :=
sorry

end NUMINAMATH_GPT_min_value_x_3y_6z_l1011_101198


namespace NUMINAMATH_GPT_first_platform_length_is_150_l1011_101157

-- Defining the conditions
def train_length : ℝ := 150
def first_platform_time : ℝ := 15
def second_platform_length : ℝ := 250
def second_platform_time : ℝ := 20

-- The distance covered when crossing the first platform is length of train + length of first platform
def distance_first_platform (L : ℝ) : ℝ := train_length + L

-- The distance covered when crossing the second platform is length of train + length of a known 250 m platform
def distance_second_platform : ℝ := train_length + second_platform_length

-- We are to prove that the length of the first platform, given the conditions, is 150 meters.
theorem first_platform_length_is_150 : ∃ L : ℝ, (distance_first_platform L / distance_second_platform) = (first_platform_time / second_platform_time) ∧ L = 150 :=
by
  let L := 150
  have h1 : distance_first_platform L = train_length + L := rfl
  have h2 : distance_second_platform = train_length + second_platform_length := rfl
  have h3 : distance_first_platform L / distance_second_platform = first_platform_time / second_platform_time :=
    by sorry
  use L
  exact ⟨h3, rfl⟩

end NUMINAMATH_GPT_first_platform_length_is_150_l1011_101157


namespace NUMINAMATH_GPT_f_at_one_f_extremes_l1011_101107

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x > 0 → f x = f x
axiom f_multiplicative : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

theorem f_at_one : f 1 = 0 := sorry

theorem f_extremes (hf_sub_one_fifth : f (1 / 5) = -1) :
  ∃ c d : ℝ, (∀ x : ℝ, 1 / 25 ≤ x ∧ x ≤ 125 → c ≤ f x ∧ f x ≤ d) ∧
  c = -2 ∧ d = 3 := sorry

end NUMINAMATH_GPT_f_at_one_f_extremes_l1011_101107


namespace NUMINAMATH_GPT_scientific_notation_flu_virus_diameter_l1011_101171

theorem scientific_notation_flu_virus_diameter :
  0.000000823 = 8.23 * 10^(-7) :=
sorry

end NUMINAMATH_GPT_scientific_notation_flu_virus_diameter_l1011_101171


namespace NUMINAMATH_GPT_percentage_solution_l1011_101123

variable (x y : ℝ)
variable (P : ℝ)

-- Conditions
axiom cond1 : 0.20 * (x - y) = (P / 100) * (x + y)
axiom cond2 : y = (1 / 7) * x

-- Theorem statement
theorem percentage_solution : P = 15 :=
by 
  -- Sorry means skipping the proof
  sorry

end NUMINAMATH_GPT_percentage_solution_l1011_101123


namespace NUMINAMATH_GPT_negation_of_p_l1011_101192

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, x ≥ 2

-- State the proof problem as a Lean theorem
theorem negation_of_p : (∀ x : ℝ, x ≥ 2) → ∃ x₀ : ℝ, x₀ < 2 :=
by
  intro h
  -- Define how the proof would generally proceed
  -- as the negation of a universal statement is an existential statement.
  sorry

end NUMINAMATH_GPT_negation_of_p_l1011_101192


namespace NUMINAMATH_GPT_find_m_minus_n_l1011_101115

-- Define line equations, parallelism, and perpendicularity
def line1 (x y : ℝ) : Prop := 3 * x - 6 * y + 1 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := x - m * y + 2 = 0
def line3 (x y : ℝ) (n : ℝ) : Prop := n * x + y + 3 = 0

def parallel (m1 m2 : ℝ) : Prop := m1 = m2
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_m_minus_n (m n : ℝ) (h_parallel : parallel (1/2) (1/m)) (h_perpendicular: perpendicular (1/2) (-1/n)) : m - n = 0 :=
sorry

end NUMINAMATH_GPT_find_m_minus_n_l1011_101115


namespace NUMINAMATH_GPT_expand_product_l1011_101196

theorem expand_product (x : ℝ) : (x + 2) * (x + 5) = x^2 + 7 * x + 10 := 
by 
  sorry

end NUMINAMATH_GPT_expand_product_l1011_101196


namespace NUMINAMATH_GPT_simplify_expression_l1011_101141

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 1) (h₂ : a ≠ 1 / 2) :
    1 - 1 / (1 - a / (1 - a)) = -a / (1 - 2 * a) := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1011_101141


namespace NUMINAMATH_GPT_evaluate_propositions_l1011_101179

variable (x y : ℝ)

def p : Prop := (x > y) → (-x < -y)
def q : Prop := (x < y) → (x^2 > y^2)

theorem evaluate_propositions : (p x y ∨ q x y) ∧ (p x y ∧ ¬q x y) := by
  -- Correct answer: \( \boxed{\text{C}} \)
  sorry

end NUMINAMATH_GPT_evaluate_propositions_l1011_101179


namespace NUMINAMATH_GPT_problem_statement_l1011_101176

def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := x - 2

theorem problem_statement : f (g 5) - g (f 5) = -8 := by sorry

end NUMINAMATH_GPT_problem_statement_l1011_101176


namespace NUMINAMATH_GPT_washer_cost_difference_l1011_101142

theorem washer_cost_difference (W D : ℝ) 
  (h1 : W + D = 1200) (h2 : D = 490) : W - D = 220 :=
sorry

end NUMINAMATH_GPT_washer_cost_difference_l1011_101142


namespace NUMINAMATH_GPT_sqrt_of_1024_l1011_101143

theorem sqrt_of_1024 (x : ℝ) (h1 : x > 0) (h2 : x ^ 2 = 1024) : x = 32 :=
sorry

end NUMINAMATH_GPT_sqrt_of_1024_l1011_101143


namespace NUMINAMATH_GPT_percentage_of_40_eq_140_l1011_101177

theorem percentage_of_40_eq_140 (p : ℝ) (h : (p / 100) * 40 = 140) : p = 350 :=
sorry

end NUMINAMATH_GPT_percentage_of_40_eq_140_l1011_101177


namespace NUMINAMATH_GPT_poly_eq_zero_or_one_l1011_101185

noncomputable def k : ℝ := 2 -- You can replace 2 with any number greater than 1.

theorem poly_eq_zero_or_one (P : ℝ → ℝ) 
  (h1 : k > 1) 
  (h2 : ∀ x : ℝ, P (x ^ k) = (P x) ^ k) : 
  (∀ x, P x = 0) ∨ (∀ x, P x = 1) :=
sorry

end NUMINAMATH_GPT_poly_eq_zero_or_one_l1011_101185


namespace NUMINAMATH_GPT_option_C_is_correct_l1011_101122

theorem option_C_is_correct :
  (-3 - (-2) ≠ -5) ∧
  (-|(-1:ℝ)/3| + 1 ≠ 4/3) ∧
  (4 - 4 / 2 = 2) ∧
  (3^2 / 6 * (1/6) ≠ 9) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_option_C_is_correct_l1011_101122


namespace NUMINAMATH_GPT_percent_owning_only_cats_l1011_101163

theorem percent_owning_only_cats (total_students dogs cats both : ℕ) (h1 : total_students = 500)
  (h2 : dogs = 150) (h3 : cats = 80) (h4 : both = 25) : (cats - both) / total_students * 100 = 11 :=
by
  sorry

end NUMINAMATH_GPT_percent_owning_only_cats_l1011_101163


namespace NUMINAMATH_GPT_max_value_f_on_interval_l1011_101166

open Real

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_f_on_interval :
  ∃ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), ∀ y ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f y ≤ f x ∧ f x = 23 := by
  sorry

end NUMINAMATH_GPT_max_value_f_on_interval_l1011_101166


namespace NUMINAMATH_GPT_value_of_N_l1011_101125

theorem value_of_N (N : ℕ) (x y z w s : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
    (h_pos_z : 0 < z) (h_pos_w : 0 < w) (h_pos_s : 0 < s) (h_sum : x + y + z + w + s = N)
    (h_comb : Nat.choose N 4 = 3003) : N = 18 := 
by
  sorry

end NUMINAMATH_GPT_value_of_N_l1011_101125


namespace NUMINAMATH_GPT_trigonometric_inequality_l1011_101154

theorem trigonometric_inequality (a b : ℝ) (ha : 0 < a ∧ a < Real.pi / 2) (hb : 0 < b ∧ b < Real.pi / 2) :
  5 / Real.cos a ^ 2 + 5 / (Real.sin a ^ 2 * Real.sin b ^ 2 * Real.cos b ^ 2) ≥ 27 * Real.cos a + 36 * Real.sin a :=
sorry

end NUMINAMATH_GPT_trigonometric_inequality_l1011_101154


namespace NUMINAMATH_GPT_proof_u_g_3_l1011_101131

noncomputable def u (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)

noncomputable def g (x : ℝ) : ℝ := 7 - u x

theorem proof_u_g_3 :
  u (g 3) = Real.sqrt (37 - 5 * Real.sqrt 17) :=
sorry

end NUMINAMATH_GPT_proof_u_g_3_l1011_101131


namespace NUMINAMATH_GPT_diagonal_AC_possibilities_l1011_101182

/-
In a quadrilateral with sides AB, BC, CD, and DA, the length of diagonal AC must 
satisfy the inequalities determined by the triangle inequalities for triangles 
ABC and CDA. Prove the number of different whole numbers that could be the 
length of diagonal AC is 13.
-/

def number_of_whole_numbers_AC (AB BC CD DA : ℕ) : ℕ :=
  if 6 < AB ∧ AB < 20 then 19 - 7 + 1 else sorry

theorem diagonal_AC_possibilities : number_of_whole_numbers_AC 7 13 15 10 = 13 :=
  by
    sorry

end NUMINAMATH_GPT_diagonal_AC_possibilities_l1011_101182


namespace NUMINAMATH_GPT_reaction_produces_nh3_l1011_101110

-- Define the Chemical Equation as a structure
structure Reaction where
  reagent1 : ℕ -- moles of NH4NO3
  reagent2 : ℕ -- moles of NaOH
  product  : ℕ -- moles of NH3

-- Given conditions
def reaction := Reaction.mk 2 2 2

-- Theorem stating that given 2 moles of NH4NO3 and 2 moles of NaOH,
-- the number of moles of NH3 formed is 2 moles.
theorem reaction_produces_nh3 (r : Reaction) (h1 : r.reagent1 = 2)
  (h2 : r.reagent2 = 2) : r.product = 2 := by
  sorry

end NUMINAMATH_GPT_reaction_produces_nh3_l1011_101110


namespace NUMINAMATH_GPT_new_average_income_l1011_101168

theorem new_average_income (old_avg_income : ℝ) (num_members : ℕ) (deceased_income : ℝ) 
  (old_avg_income_eq : old_avg_income = 735) (num_members_eq : num_members = 4) 
  (deceased_income_eq : deceased_income = 990) : 
  ((old_avg_income * num_members) - deceased_income) / (num_members - 1) = 650 := 
by sorry

end NUMINAMATH_GPT_new_average_income_l1011_101168


namespace NUMINAMATH_GPT_find_a1_over_1_minus_q_l1011_101193

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem find_a1_over_1_minus_q 
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h1 : a 1 + a 2 + a 3 + a 4 = 3)
  (h2 : a 5 + a 6 + a 7 + a 8 = 48) :
  (a 1) / (1 - q) = -1 / 5 :=
sorry

end NUMINAMATH_GPT_find_a1_over_1_minus_q_l1011_101193


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1011_101118

theorem problem_part1 (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : x^2 - 2 * x * y + 4 * y^2 = 11) :
  x * y = 1 :=
sorry

theorem problem_part2 (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : x^2 - 2 * x * y + 4 * y^2 = 11) :
  x^2 * y - 2 * x * y^2 = 3 :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1011_101118


namespace NUMINAMATH_GPT_tan_subtraction_formula_l1011_101175

theorem tan_subtraction_formula 
  (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 := 
by
  sorry

end NUMINAMATH_GPT_tan_subtraction_formula_l1011_101175


namespace NUMINAMATH_GPT_price_of_basic_computer_l1011_101153

variable (C P : ℝ)

theorem price_of_basic_computer 
    (h1 : C + P = 2500)
    (h2 : P = (1 / 6) * (C + 500 + P)) : 
  C = 2000 :=
by
  sorry

end NUMINAMATH_GPT_price_of_basic_computer_l1011_101153


namespace NUMINAMATH_GPT_angle_bisector_slope_l1011_101188

theorem angle_bisector_slope :
  ∀ m1 m2 : ℝ, m1 = 2 → m2 = 4 → (∃ k : ℝ, k = (6 - Real.sqrt 21) / (-7) → k = (-6 + Real.sqrt 21) / 7) :=
by
  sorry

end NUMINAMATH_GPT_angle_bisector_slope_l1011_101188


namespace NUMINAMATH_GPT_pythagorean_theorem_l1011_101180

theorem pythagorean_theorem {a b c p q : ℝ} 
  (h₁ : p * c = a ^ 2) 
  (h₂ : q * c = b ^ 2)
  (h₃ : p + q = c) : 
  c ^ 2 = a ^ 2 + b ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_pythagorean_theorem_l1011_101180


namespace NUMINAMATH_GPT_arithmetic_mean_of_fractions_l1011_101184

theorem arithmetic_mean_of_fractions :
  let a := (5 : ℚ) / 8
  let b := (9 : ℚ) / 16
  let c := (11 : ℚ) / 16
  a = (b + c) / 2 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_fractions_l1011_101184


namespace NUMINAMATH_GPT_sum_first_2009_terms_arith_seq_l1011_101162

variable {a : ℕ → ℝ}

-- Given condition a_1004 + a_1005 + a_1006 = 3
axiom H : a 1004 + a 1005 + a 1006 = 3

-- Arithmetic sequence definition
def is_arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_arith_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem sum_first_2009_terms_arith_seq
  (d : ℝ) (h_arith_seq : is_arith_seq a d)
  : sum_arith_seq a 2009 = 2009 := 
by
  sorry

end NUMINAMATH_GPT_sum_first_2009_terms_arith_seq_l1011_101162


namespace NUMINAMATH_GPT_quarters_needed_to_buy_items_l1011_101138

-- Define the costs of each item in cents
def cost_candy_bar : ℕ := 25
def cost_chocolate : ℕ := 75
def cost_juice : ℕ := 50

-- Define the quantities of each item
def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Define the value of a quarter in cents
def value_of_quarter : ℕ := 25

-- Define the total cost of the items
def total_cost : ℕ := (num_candy_bars * cost_candy_bar) + (num_chocolates * cost_chocolate) + (num_juice_packs * cost_juice)

-- Calculate the number of quarters needed
def num_quarters_needed : ℕ := total_cost / value_of_quarter

-- The theorem to prove that the number of quarters needed is 11
theorem quarters_needed_to_buy_items : num_quarters_needed = 11 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_quarters_needed_to_buy_items_l1011_101138


namespace NUMINAMATH_GPT_solve_modular_equation_l1011_101102

theorem solve_modular_equation (x : ℤ) :
  (15 * x + 2) % 18 = 7 % 18 ↔ x % 6 = 1 % 6 := by
  sorry

end NUMINAMATH_GPT_solve_modular_equation_l1011_101102


namespace NUMINAMATH_GPT_tan_add_pi_over_3_l1011_101155

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = Real.sqrt 3) :
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_add_pi_over_3_l1011_101155


namespace NUMINAMATH_GPT_taozi_is_faster_than_xiaoxiao_l1011_101105

theorem taozi_is_faster_than_xiaoxiao : 
  let taozi_speed := 210
  let xiaoxiao_distance := 500
  let xiaoxiao_time := 3
  let xiaoxiao_speed := xiaoxiao_distance / xiaoxiao_time
  taozi_speed > xiaoxiao_speed
:= by
  let taozi_speed := 210
  let xiaoxiao_distance := 500
  let xiaoxiao_time := 3
  let xiaoxiao_speed := xiaoxiao_distance / xiaoxiao_time
  sorry

end NUMINAMATH_GPT_taozi_is_faster_than_xiaoxiao_l1011_101105


namespace NUMINAMATH_GPT_perfect_square_factors_count_450_l1011_101161

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end NUMINAMATH_GPT_perfect_square_factors_count_450_l1011_101161


namespace NUMINAMATH_GPT_total_marks_l1011_101104

variable (marks_in_music marks_in_maths marks_in_arts marks_in_social_studies : ℕ)

def marks_conditions : Prop :=
  marks_in_maths = marks_in_music - (1/10) * marks_in_music ∧
  marks_in_maths = marks_in_arts - 20 ∧
  marks_in_social_studies = marks_in_music + 10 ∧
  marks_in_music = 70

theorem total_marks 
  (h : marks_conditions marks_in_music marks_in_maths marks_in_arts marks_in_social_studies) :
  marks_in_music + marks_in_maths + marks_in_arts + marks_in_social_studies = 296 :=
by
  sorry

end NUMINAMATH_GPT_total_marks_l1011_101104


namespace NUMINAMATH_GPT_find_M_coordinate_l1011_101159

-- Definitions of the given points
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := ⟨1, 0, 2⟩
def B : Point3D := ⟨1, -3, 1⟩
def M (y : ℝ) : Point3D := ⟨0, y, 0⟩

-- Definition for the squared distance between two points
def dist_sq (p1 p2 : Point3D) : ℝ :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2

-- Main theorem statement
theorem find_M_coordinate (y : ℝ) : 
  dist_sq (M y) A = dist_sq (M y) B → y = -1 :=
by
  simp [dist_sq, A, B, M]
  sorry

end NUMINAMATH_GPT_find_M_coordinate_l1011_101159


namespace NUMINAMATH_GPT_remainder_when_divided_l1011_101126

theorem remainder_when_divided (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_l1011_101126


namespace NUMINAMATH_GPT_probability_of_selecting_cooking_l1011_101136

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_cooking_l1011_101136


namespace NUMINAMATH_GPT_reducedRatesFraction_l1011_101165

variable (total_hours_per_week : ℕ := 168)
variable (reduced_rate_hours_weekdays : ℕ := 12 * 5)
variable (reduced_rate_hours_weekends : ℕ := 24 * 2)

theorem reducedRatesFraction
  (h1 : total_hours_per_week = 7 * 24)
  (h2 : reduced_rate_hours_weekdays = 12 * 5)
  (h3 : reduced_rate_hours_weekends = 24 * 2) :
  (reduced_rate_hours_weekdays + reduced_rate_hours_weekends) / total_hours_per_week = 9 / 14 := 
  sorry

end NUMINAMATH_GPT_reducedRatesFraction_l1011_101165


namespace NUMINAMATH_GPT_exp_fn_max_min_diff_l1011_101160

theorem exp_fn_max_min_diff (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (max (a^1) (a^0) - min (a^1) (a^0)) = 1 / 2 → (a = 1 / 2 ∨ a = 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_exp_fn_max_min_diff_l1011_101160


namespace NUMINAMATH_GPT_complement_A_U_l1011_101100

-- Define the universal set U and set A as given in the problem.
def U : Set ℕ := { x | x ≥ 3 }
def A : Set ℕ := { x | x * x ≥ 10 }

-- Prove that the complement of A with respect to U is {3}.
theorem complement_A_U :
  (U \ A) = {3} :=
by
  sorry

end NUMINAMATH_GPT_complement_A_U_l1011_101100


namespace NUMINAMATH_GPT_calculate_expression_l1011_101149

theorem calculate_expression :
  ((650^2 - 350^2) * 3 = 900000) := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1011_101149


namespace NUMINAMATH_GPT_polygon_interior_angle_sum_l1011_101147

theorem polygon_interior_angle_sum (n : ℕ) (h : (n - 2) * 180 = 1800) : n = 12 :=
by sorry

end NUMINAMATH_GPT_polygon_interior_angle_sum_l1011_101147
