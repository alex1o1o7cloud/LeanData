import Mathlib

namespace measure_orthogonal_trihedral_angle_sum_measure_polyhedral_angles_l1438_143879

theorem measure_orthogonal_trihedral_angle (d : ℕ) (a : ℝ) (n : ℕ) 
(h1 : d = 3) (h2 : a = π / 2) (h3 : n = 8) : 
  ∃ measure : ℝ, measure = π / 2 :=
by
  sorry

theorem sum_measure_polyhedral_angles (d : ℕ) (a : ℝ) (n : ℕ) 
(h1 : d = 3) (h2 : a = π / 2) (h3 : n = 8) 
(h4 : n * a = 4 * π) : 
  ∃ sum_measure : ℝ, sum_measure = 4 * π :=
by
  sorry

end measure_orthogonal_trihedral_angle_sum_measure_polyhedral_angles_l1438_143879


namespace find_speed_l1438_143874

noncomputable def distance : ℝ := 600
noncomputable def speed1 : ℝ := 50
noncomputable def meeting_distance : ℝ := distance / 2
noncomputable def departure_time1 : ℝ := 7
noncomputable def departure_time2 : ℝ := 8
noncomputable def meeting_time : ℝ := 13

theorem find_speed (x : ℝ) : 
  (meeting_distance / speed1 = meeting_time - departure_time1) ∧
  (meeting_distance / x = meeting_time - departure_time2) → 
  x = 60 :=
by
  sorry

end find_speed_l1438_143874


namespace length_OC_l1438_143871

theorem length_OC (a b : ℝ) (h_perpendicular : ∀ x, x^2 + a * x + b = 0 → x = 1 ∨ x = b) : 
  1 = 1 :=
by 
  sorry

end length_OC_l1438_143871


namespace spongebob_price_l1438_143815

variable (x : ℝ)

theorem spongebob_price (h : 30 * x + 12 * 1.5 = 78) : x = 2 :=
by
  -- Given condition: 30 * x + 12 * 1.5 = 78
  sorry

end spongebob_price_l1438_143815


namespace find_x_l1438_143890

-- Define the condition as a theorem
theorem find_x (x : ℝ) (h : (1 + 3 + x) / 3 = 3) : x = 5 :=
by
  sorry  -- Placeholder for the proof

end find_x_l1438_143890


namespace number_of_girls_l1438_143862

theorem number_of_girls
  (total_students : ℕ)
  (ratio_girls : ℕ) (ratio_boys : ℕ) (ratio_non_binary : ℕ)
  (h_ratio : ratio_girls = 3 ∧ ratio_boys = 2 ∧ ratio_non_binary = 1)
  (h_total : total_students = 72) :
  ∃ (k : ℕ), 3 * k = (total_students * 3) / 6 ∧ 3 * k = 36 :=
by
  sorry

end number_of_girls_l1438_143862


namespace mean_of_xyz_l1438_143848

theorem mean_of_xyz (x y z : ℚ) (eleven_mean : ℚ)
  (eleven_sum : eleven_mean = 32)
  (fourteen_sum : 14 * 45 = 630)
  (new_mean : 14 * 45 = 630) :
  (x + y + z) / 3 = 278 / 3 :=
by
  sorry

end mean_of_xyz_l1438_143848


namespace house_number_is_fourteen_l1438_143849

theorem house_number_is_fourteen (a b c n : ℕ) (h1 : a * b * c = 40) (h2 : a + b + c = n) (h3 : 
  ∃ (a b c : ℕ), a * b * c = 40 ∧ (a = 1 ∧ b = 5 ∧ c = 8) ∨ (a = 2 ∧ b = 2 ∧ c = 10) ∧ n = 14) :
  n = 14 :=
sorry

end house_number_is_fourteen_l1438_143849


namespace tens_digit_of_M_l1438_143895

def P (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a * b

def S (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  a + b

theorem tens_digit_of_M {M : ℕ} (h : 10 ≤ M ∧ M < 100) (h_eq : M = P M + S M + 6) :
  M / 10 = 1 ∨ M / 10 = 2 :=
sorry

end tens_digit_of_M_l1438_143895


namespace mallory_travel_expenses_l1438_143889

theorem mallory_travel_expenses (fuel_tank_cost : ℕ) (fuel_tank_miles : ℕ) (total_miles : ℕ) (food_ratio : ℚ)
  (h_fuel_tank_cost : fuel_tank_cost = 45)
  (h_fuel_tank_miles : fuel_tank_miles = 500)
  (h_total_miles : total_miles = 2000)
  (h_food_ratio : food_ratio = 3/5) :
  ∃ total_cost : ℕ, total_cost = 288 :=
by
  sorry

end mallory_travel_expenses_l1438_143889


namespace find_x_values_l1438_143802

theorem find_x_values (x y : ℕ) (hx: x > y) (hy: y > 0) (h: x + y + x * y = 101) :
  x = 50 ∨ x = 16 :=
sorry

end find_x_values_l1438_143802


namespace multiplication_in_S_l1438_143823

-- Define the set S as given in the conditions
variable (S : Set ℝ)

-- Condition 1: 1 ∈ S
def condition1 : Prop := 1 ∈ S

-- Condition 2: ∀ a b ∈ S, a - b ∈ S
def condition2 : Prop := ∀ a b : ℝ, a ∈ S → b ∈ S → (a - b) ∈ S

-- Condition 3: ∀ a ∈ S, a ≠ 0 → 1 / a ∈ S
def condition3 : Prop := ∀ a : ℝ, a ∈ S → a ≠ 0 → (1 / a) ∈ S

-- Theorem to prove: ∀ a b ∈ S, ab ∈ S
theorem multiplication_in_S (h1 : condition1 S) (h2 : condition2 S) (h3 : condition3 S) :
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a * b) ∈ S := 
  sorry

end multiplication_in_S_l1438_143823


namespace sum_of_roots_l1438_143887

theorem sum_of_roots (p : ℝ) (h : (4 - p) / 2 = 9) : (p / 2 = 7) :=
by 
  sorry

end sum_of_roots_l1438_143887


namespace staircase_ways_four_steps_l1438_143851

theorem staircase_ways_four_steps : 
  let one_step := 1
  let two_steps := 2
  let three_steps := 3
  let four_steps := 4
  1           -- one step at a time
  + 3         -- combination of one and two steps
  + 2         -- combination of one and three steps
  + 1         -- two steps at a time
  + 1 = 8     -- all four steps in one stride
:= by
  sorry

end staircase_ways_four_steps_l1438_143851


namespace multiple_of_other_number_l1438_143877

theorem multiple_of_other_number 
(m S L : ℕ) 
(hl : L = 33) 
(hrel : L = m * S - 3) 
(hsum : L + S = 51) : 
m = 2 :=
by
  sorry

end multiple_of_other_number_l1438_143877


namespace notebook_problem_l1438_143805

theorem notebook_problem
    (total_notebooks : ℕ)
    (cost_price_A : ℕ)
    (cost_price_B : ℕ)
    (total_cost_price : ℕ)
    (selling_price_A : ℕ)
    (selling_price_B : ℕ)
    (discount_A : ℕ)
    (profit_condition : ℕ)
    (x y m : ℕ) 
    (h1 : total_notebooks = 350)
    (h2 : cost_price_A = 12)
    (h3 : cost_price_B = 15)
    (h4 : total_cost_price = 4800)
    (h5 : selling_price_A = 20)
    (h6 : selling_price_B = 25)
    (h7 : discount_A = 30)
    (h8 : 12 * x + 15 * y = 4800)
    (h9 : x + y = 350)
    (h10 : selling_price_A * m + selling_price_B * m + (x - m) * selling_price_A * 7 / 10 + (y - m) * cost_price_B - total_cost_price ≥ profit_condition):
    x = 150 ∧ m ≥ 128 :=
by
    sorry

end notebook_problem_l1438_143805


namespace sum_of_primes_no_solution_congruence_l1438_143842

theorem sum_of_primes_no_solution_congruence :
  2 + 5 = 7 :=
by
  sorry

end sum_of_primes_no_solution_congruence_l1438_143842


namespace exists_natural_2001_digits_l1438_143854

theorem exists_natural_2001_digits (N : ℕ) (hN: N = 5 * 10^2000 + 1) : 
  ∃ K : ℕ, (K = N) ∧ (N^(2001) % 10^2001 = N % 10^2001) :=
by
  sorry

end exists_natural_2001_digits_l1438_143854


namespace even_function_x_lt_0_l1438_143835

noncomputable def f (x : ℝ) : ℝ :=
if h : x >= 0 then 2^x + 1 else 2^(-x) + 1

theorem even_function_x_lt_0 (x : ℝ) (hx : x < 0) : f x = 2^(-x) + 1 :=
by {
  sorry
}

end even_function_x_lt_0_l1438_143835


namespace max_value_of_x_plus_y_l1438_143838

theorem max_value_of_x_plus_y (x y : ℝ) (h : x^2 / 16 + y^2 / 9 = 1) : x + y ≤ 5 :=
sorry

end max_value_of_x_plus_y_l1438_143838


namespace num_valid_seat_permutations_l1438_143891

/-- 
  The number of ways eight people can switch their seats in a circular 
  arrangement such that no one sits in the same, adjacent, or directly 
  opposite chair they originally occupied is 5.
-/
theorem num_valid_seat_permutations : 
  ∃ (σ : Equiv.Perm (Fin 8)), 
  (∀ i : Fin 8, σ i ≠ i) ∧ 
  (∀ i : Fin 8, σ i ≠ if i.val < 7 then i + 1 else 0) ∧ 
  (∀ i : Fin 8, σ i ≠ if i.val < 8 / 2 then (i + 8 / 2) % 8 else (i - 8 / 2) % 8) :=
  sorry

end num_valid_seat_permutations_l1438_143891


namespace k_inequality_l1438_143873

noncomputable def k_value : ℝ :=
  5

theorem k_inequality (x : ℝ) :
  (x * (2 * x + 3) < k_value) ↔ (x > -5 / 2 ∧ x < 1) :=
sorry

end k_inequality_l1438_143873


namespace sum_from_1_to_60_is_1830_sum_from_51_to_60_is_555_l1438_143855

-- Definition for the sum of the first n natural numbers
def sum_upto (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition for the sum from 1 to 60
def sum_1_to_60 : ℕ := sum_upto 60

-- Definition for the sum from 1 to 50
def sum_1_to_50 : ℕ := sum_upto 50

-- Proof problem 1
theorem sum_from_1_to_60_is_1830 : sum_1_to_60 = 1830 := 
by
  sorry

-- Definition for the sum from 51 to 60
def sum_51_to_60 : ℕ := sum_1_to_60 - sum_1_to_50

-- Proof problem 2
theorem sum_from_51_to_60_is_555 : sum_51_to_60 = 555 := 
by
  sorry

end sum_from_1_to_60_is_1830_sum_from_51_to_60_is_555_l1438_143855


namespace inequality_transformations_l1438_143898

theorem inequality_transformations (a b : ℝ) (h : a > b) :
  (3 * a > 3 * b) ∧ (a + 2 > b + 2) ∧ (-5 * a < -5 * b) :=
by
  sorry

end inequality_transformations_l1438_143898


namespace real_function_as_sum_of_symmetric_graphs_l1438_143840

theorem real_function_as_sum_of_symmetric_graphs (f : ℝ → ℝ) :
  ∃ (g h : ℝ → ℝ), (∀ x, g x + h x = f x) ∧ (∀ x, g x = g (-x)) ∧ (∀ x, h (1 + x) = h (1 - x)) :=
sorry

end real_function_as_sum_of_symmetric_graphs_l1438_143840


namespace cos_sq_plus_two_sin_double_l1438_143858

theorem cos_sq_plus_two_sin_double (α : ℝ) (h : Real.tan α = 3 / 4) : Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 :=
by
  sorry

end cos_sq_plus_two_sin_double_l1438_143858


namespace cube_volume_l1438_143832

theorem cube_volume (length width : ℝ) (h_length : length = 48) (h_width : width = 72) :
  let area := length * width
  let side_length_in_inches := Real.sqrt (area / 6)
  let side_length_in_feet := side_length_in_inches / 12
  let volume := side_length_in_feet ^ 3
  volume = 8 :=
by
  sorry

end cube_volume_l1438_143832


namespace point_probability_in_cone_l1438_143826

noncomputable def volume_of_cone (S : ℝ) (h : ℝ) : ℝ :=
  (1/3) * S * h

theorem point_probability_in_cone (P M : ℝ) (S_ABC : ℝ) (h_P h_M : ℝ)
  (h_volume_condition : volume_of_cone S_ABC h_P ≤ volume_of_cone S_ABC h_M / 3) :
  (1 - (2 / 3) ^ 3) = 19 / 27 :=
by
  sorry

end point_probability_in_cone_l1438_143826


namespace total_population_l1438_143892

variables (b g t : ℕ)

-- Conditions
def cond1 := b = 4 * g
def cond2 := g = 2 * t

-- Theorem statement
theorem total_population (h1 : cond1 b g) (h2 : cond2 g t) : b + g + t = 11 * b / 8 :=
by sorry

end total_population_l1438_143892


namespace find_a_l1438_143856

theorem find_a (x y a : ℝ) (hx_pos_even : x > 0 ∧ ∃ n : ℕ, x = 2 * n) (hx_le_y : x ≤ y) 
  (h_eq_zero : |3 * y - 18| + |a * x - y| = 0) : 
  a = 3 ∨ a = 3 / 2 ∨ a = 1 :=
sorry

end find_a_l1438_143856


namespace area_ratio_is_correct_l1438_143800

noncomputable def areaRatioInscribedCircumscribedOctagon (r : ℝ) : ℝ :=
  let s1 := r * Real.sin (67.5 * Real.pi / 180)
  let s2 := r * Real.sin (67.5 * Real.pi / 180) / Real.cos (67.5 * Real.pi / 180)
  (s2 / s1) ^ 2

theorem area_ratio_is_correct (r : ℝ) : areaRatioInscribedCircumscribedOctagon r = 2 + Real.sqrt 2 :=
by
  sorry

end area_ratio_is_correct_l1438_143800


namespace star_six_three_l1438_143836

-- Definition of the operation
def star (a b : ℕ) : ℕ := 4 * a + 5 * b - 2 * a * b

-- Statement to prove
theorem star_six_three : star 6 3 = 3 := by
  sorry

end star_six_three_l1438_143836


namespace domain_of_f_l1438_143834

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 2*x - 3)

theorem domain_of_f : 
  {x : ℝ | (x^2 - 2*x - 3) ≠ 0} = {x : ℝ | x < -1} ∪ {x : ℝ | -1 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l1438_143834


namespace tan_585_eq_one_l1438_143813

theorem tan_585_eq_one : Real.tan (585 * Real.pi / 180) = 1 :=
by
  sorry

end tan_585_eq_one_l1438_143813


namespace line_connecting_centers_l1438_143868

-- Define the first circle equation
def circle1 (x y : ℝ) := x^2 + y^2 - 4*x + 6*y = 0

-- Define the second circle equation
def circle2 (x y : ℝ) := x^2 + y^2 - 6*x = 0

-- Define the line equation
def line_eq (x y : ℝ) := 3*x - y - 9 = 0

-- Prove that the line connecting the centers of the circles has the given equation
theorem line_connecting_centers :
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y → line_eq x y := 
sorry

end line_connecting_centers_l1438_143868


namespace soccer_team_arrangements_l1438_143811

theorem soccer_team_arrangements : 
  ∃ (n : ℕ), n = 2 * (Nat.factorial 11)^2 := 
sorry

end soccer_team_arrangements_l1438_143811


namespace find_f_five_l1438_143864

-- Define the function f and the conditions as given in the problem.
variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)
variable (h₁ : ∀ x y : ℝ, f (x - y) = f x * g y)
variable (h₂ : ∀ y : ℝ, g y = Real.exp (-y))
variable (h₃ : ∀ x : ℝ, f x ≠ 0)

-- Goal: Prove that f(5) = e^{2.5}.
theorem find_f_five : f 5 = Real.exp 2.5 :=
by
  -- Proof is omitted as per the instructions.
  sorry

end find_f_five_l1438_143864


namespace constant_term_zero_quadratic_l1438_143850

theorem constant_term_zero_quadratic (m : ℝ) :
  (-m^2 + 1 = 0) → m = -1 :=
by
  intro h
  sorry

end constant_term_zero_quadratic_l1438_143850


namespace problem_ab_cd_eq_l1438_143876

theorem problem_ab_cd_eq (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 5)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = 14) :
  ab + cd = 45 := 
by
  sorry

end problem_ab_cd_eq_l1438_143876


namespace Julia_played_with_kids_l1438_143816

theorem Julia_played_with_kids :
  (∃ k : ℕ, k = 4) ∧ (∃ n : ℕ, n = 4 + 12) → (n = 16) :=
by
  sorry

end Julia_played_with_kids_l1438_143816


namespace possible_sums_of_digits_l1438_143859

-- Defining the main theorem
theorem possible_sums_of_digits 
  (A B C : ℕ) (hA : 0 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 0 ≤ C ∧ C ≤ 9)
  (hdiv : (A + 6 + 2 + 8 + B + 7 + C + 3) % 9 = 0) :
  A + B + C = 1 ∨ A + B + C = 10 ∨ A + B + C = 19 :=
by
  sorry

end possible_sums_of_digits_l1438_143859


namespace range_of_b_l1438_143837

variable (a b c : ℝ)

theorem range_of_b (h1 : a * c = b^2) (h2 : a + b + c = 3) : -3 ≤ b ∧ b ≤ 1 :=
sorry

end range_of_b_l1438_143837


namespace daily_profit_1200_impossible_daily_profit_1600_l1438_143807

-- Definitions of given conditions
def avg_shirts_sold_per_day : ℕ := 30
def profit_per_shirt : ℕ := 40

-- Function for the number of shirts sold given a price reduction
def shirts_sold (x : ℕ) : ℕ := avg_shirts_sold_per_day + 2 * x

-- Function for the profit per shirt given a price reduction
def new_profit_per_shirt (x : ℕ) : ℕ := profit_per_shirt - x

-- Function for the daily profit given a price reduction
def daily_profit (x : ℕ) : ℕ := (new_profit_per_shirt x) * (shirts_sold x)

-- Proving the desired conditions in Lean

-- Part 1: Prove that reducing the price by 25 yuan results in a daily profit of 1200 yuan
theorem daily_profit_1200 (x : ℕ) : daily_profit x = 1200 ↔ x = 25 :=
by
  { sorry }

-- Part 2: Prove that a daily profit of 1600 yuan is not achievable
theorem impossible_daily_profit_1600 (x : ℕ) : daily_profit x ≠ 1600 :=
by
  { sorry }

end daily_profit_1200_impossible_daily_profit_1600_l1438_143807


namespace value_of_expression_at_x_4_l1438_143882

theorem value_of_expression_at_x_4 :
  ∀ (x : ℝ), x = 4 → (x^2 - 2 * x - 8) / (x - 4) = 6 :=
by
  intro x hx
  sorry

end value_of_expression_at_x_4_l1438_143882


namespace find_x_l1438_143818

theorem find_x (x y : ℕ) (h1 : x / y = 12 / 3) (h2 : y = 27) : x = 108 := by
  sorry

end find_x_l1438_143818


namespace area_of_third_region_l1438_143885

theorem area_of_third_region (A B C : ℝ) 
    (hA : A = 24) 
    (hB : B = 13) 
    (hTotal : A + B + C = 48) : 
    C = 11 := 
by 
  sorry

end area_of_third_region_l1438_143885


namespace reverse_9_in_5_operations_reverse_52_in_27_operations_not_reverse_52_in_17_operations_not_reverse_52_in_26_operations_l1438_143888

-- Definition: reversing a deck of n cards in k operations
def can_reverse_deck (n k : ℕ) : Prop := sorry -- Placeholder definition

-- Proof Part (a)
theorem reverse_9_in_5_operations :
  can_reverse_deck 9 5 :=
sorry

-- Proof Part (b)
theorem reverse_52_in_27_operations :
  can_reverse_deck 52 27 :=
sorry

-- Proof Part (c)
theorem not_reverse_52_in_17_operations :
  ¬can_reverse_deck 52 17 :=
sorry

-- Proof Part (d)
theorem not_reverse_52_in_26_operations :
  ¬can_reverse_deck 52 26 :=
sorry

end reverse_9_in_5_operations_reverse_52_in_27_operations_not_reverse_52_in_17_operations_not_reverse_52_in_26_operations_l1438_143888


namespace probability_yellow_ball_l1438_143875

-- Definitions of the conditions
def white_balls : ℕ := 2
def yellow_balls : ℕ := 3
def total_balls : ℕ := white_balls + yellow_balls

-- Theorem statement
theorem probability_yellow_ball : (yellow_balls : ℚ) / total_balls = 3 / 5 :=
by
  -- Using tactics to facilitate the proof
  simp [yellow_balls, total_balls]
  sorry

end probability_yellow_ball_l1438_143875


namespace eval_expression_l1438_143869

open Real

noncomputable def e : ℝ := 2.71828

theorem eval_expression : abs (5 * e - 15) = 1.4086 := by
  sorry

end eval_expression_l1438_143869


namespace count_triples_not_div_by_4_l1438_143866

theorem count_triples_not_div_by_4 :
  {n : ℕ // n = 117 ∧ ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 5 → 1 ≤ b ∧ b ≤ 5 → 1 ≤ c ∧ c ≤ 5 → (a + b) * (a + c) * (b + c) % 4 ≠ 0} :=
sorry

end count_triples_not_div_by_4_l1438_143866


namespace total_dots_not_visible_proof_l1438_143884

def total_dots_on_one_die : ℕ := 21

def total_dots_on_five_dice : ℕ := 5 * total_dots_on_one_die

def visible_numbers : List ℕ := [1, 2, 3, 1, 4, 5, 6, 2]

def sum_visible_numbers : ℕ := visible_numbers.sum

def total_dots_not_visible (total : ℕ) (visible_sum : ℕ) : ℕ :=
  total - visible_sum

theorem total_dots_not_visible_proof :
  total_dots_not_visible total_dots_on_five_dice sum_visible_numbers = 81 :=
by
  sorry

end total_dots_not_visible_proof_l1438_143884


namespace balls_in_boxes_l1438_143809

-- Definition of the combinatorial function
def combinations (n k : ℕ) : ℕ :=
  n.choose k

-- Problem statement in Lean
theorem balls_in_boxes :
  combinations 7 2 = 21 :=
by
  -- Since the proof is not required here, we place sorry to skip the proof.
  sorry

end balls_in_boxes_l1438_143809


namespace hundredth_power_remainders_l1438_143820

theorem hundredth_power_remainders (a : ℤ) : 
  (a % 5 = 0 → a^100 % 125 = 0) ∧ (a % 5 ≠ 0 → a^100 % 125 = 1) :=
by
  sorry

end hundredth_power_remainders_l1438_143820


namespace sum_squares_divisible_by_7_implies_both_divisible_l1438_143897

theorem sum_squares_divisible_by_7_implies_both_divisible (a b : ℤ) (h : 7 ∣ (a^2 + b^2)) : 7 ∣ a ∧ 7 ∣ b :=
sorry

end sum_squares_divisible_by_7_implies_both_divisible_l1438_143897


namespace findTwoHeaviestStonesWith35Weighings_l1438_143899

-- Define the problem with conditions
def canFindTwoHeaviestStones (stones : Fin 32 → ℝ) (weighings : ℕ) : Prop :=
  ∀ (balanceScale : (Fin 32 × Fin 32) → Bool), weighings ≤ 35 → 
  ∃ (heaviest : Fin 32) (secondHeaviest : Fin 32), 
  (heaviest ≠ secondHeaviest) ∧ 
  (∀ i : Fin 32, stones heaviest ≥ stones i) ∧ 
  (∀ j : Fin 32, j ≠ heaviest → stones secondHeaviest ≥ stones j)

-- Formally state the theorem
theorem findTwoHeaviestStonesWith35Weighings (stones : Fin 32 → ℝ) :
  canFindTwoHeaviestStones stones 35 :=
sorry -- Proof is omitted

end findTwoHeaviestStonesWith35Weighings_l1438_143899


namespace money_distribution_l1438_143896

theorem money_distribution (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 360) (h3 : C = 60) : A + B + C = 500 := by
  sorry

end money_distribution_l1438_143896


namespace abs_ineq_solution_set_l1438_143880

theorem abs_ineq_solution_set (x : ℝ) :
  |x - 5| + |x + 3| ≥ 10 ↔ x ≤ -4 ∨ x ≥ 6 :=
by
  sorry

end abs_ineq_solution_set_l1438_143880


namespace parabola_expression_l1438_143829

theorem parabola_expression (a c : ℝ) (h1 : a = 1/4 ∨ a = -1/4) (h2 : ∀ x : ℝ, x = 1 → (a * x^2 + c = 0)) :
  (a = 1/4 ∧ c = -1/4) ∨ (a = -1/4 ∧ c = 1/4) :=
by {
  sorry
}

end parabola_expression_l1438_143829


namespace find_constant_l1438_143831

noncomputable def expr (x C : ℝ) : ℝ :=
  (x - 1) * (x - 3) * (x - 4) * (x - 6) + C

theorem find_constant :
  (∀ x : ℝ, expr x (-0.5625) ≥ 1) → expr 3.5 (-0.5625) = 1 :=
by
  sorry

end find_constant_l1438_143831


namespace min_value_a_plus_b_plus_c_l1438_143867

theorem min_value_a_plus_b_plus_c (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : 9 * a + 4 * b = a * b * c) : a + b + c ≥ 10 :=
by
  sorry

end min_value_a_plus_b_plus_c_l1438_143867


namespace frac_3125_over_1024_gt_e_l1438_143822

theorem frac_3125_over_1024_gt_e : (3125 : ℝ) / 1024 > Real.exp 1 := sorry

end frac_3125_over_1024_gt_e_l1438_143822


namespace ratio_of_girls_to_boys_l1438_143812

-- Define conditions
def num_boys : ℕ := 40
def children_per_counselor : ℕ := 8
def num_counselors : ℕ := 20

-- Total number of children
def total_children : ℕ := num_counselors * children_per_counselor

-- Number of girls
def num_girls : ℕ := total_children - num_boys

-- The ratio of girls to boys
def girls_to_boys_ratio : ℚ := num_girls / num_boys

-- The theorem we need to prove
theorem ratio_of_girls_to_boys : girls_to_boys_ratio = 3 := by
  sorry

end ratio_of_girls_to_boys_l1438_143812


namespace coords_with_respect_to_origin_l1438_143821

/-- Given that the coordinates of point A are (-1, 2), prove that the coordinates of point A
with respect to the origin in the plane rectangular coordinate system xOy are (-1, 2). -/
theorem coords_with_respect_to_origin (A : (ℝ × ℝ)) (h : A = (-1, 2)) : A = (-1, 2) :=
sorry

end coords_with_respect_to_origin_l1438_143821


namespace bumper_car_line_total_in_both_lines_l1438_143893

theorem bumper_car_line (x y Z : ℕ) (hZ : Z = 25 - x + y) : Z = 25 - x + y :=
by
  sorry

theorem total_in_both_lines (x y Z : ℕ) (hZ : Z = 25 - x + y) : 40 - x + y = Z + 15 :=
by
  sorry

end bumper_car_line_total_in_both_lines_l1438_143893


namespace weekly_caloric_allowance_l1438_143894

-- Define the given conditions
def average_daily_allowance : ℕ := 2000
def daily_reduction_goal : ℕ := 500
def intense_workout_extra_calories : ℕ := 300
def moderate_exercise_extra_calories : ℕ := 200
def days_intense_workout : ℕ := 2
def days_moderate_exercise : ℕ := 3
def days_rest : ℕ := 2

-- Lean statement to prove the total weekly caloric intake
theorem weekly_caloric_allowance :
  (days_intense_workout * (average_daily_allowance - daily_reduction_goal + intense_workout_extra_calories)) +
  (days_moderate_exercise * (average_daily_allowance - daily_reduction_goal + moderate_exercise_extra_calories)) +
  (days_rest * (average_daily_allowance - daily_reduction_goal)) = 11700 := by
  sorry

end weekly_caloric_allowance_l1438_143894


namespace chocolate_bar_count_l1438_143872

theorem chocolate_bar_count (bar_weight : ℕ) (box_weight : ℕ) (H1 : bar_weight = 125) (H2 : box_weight = 2000) : box_weight / bar_weight = 16 :=
by
  sorry

end chocolate_bar_count_l1438_143872


namespace small_barrel_5_tons_l1438_143808

def total_oil : ℕ := 95
def large_barrel_capacity : ℕ := 6
def small_barrel_capacity : ℕ := 5

theorem small_barrel_5_tons :
  ∃ (num_large_barrels num_small_barrels : ℕ),
  num_small_barrels = 1 ∧
  total_oil = (num_large_barrels * large_barrel_capacity) + (num_small_barrels * small_barrel_capacity) :=
by
  sorry

end small_barrel_5_tons_l1438_143808


namespace tangent_line_equation_l1438_143881

-- Definitions for the conditions
def curve (x : ℝ) : ℝ := 2 * x^2 - x
def point_of_tangency : ℝ × ℝ := (1, 1)

-- Statement of the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), (b = 1 - 3 * 1) ∧ 
  (m = 3) ∧ 
  ∀ (x y : ℝ), y = m * x + b → 3 * x - y - 2 = 0 :=
by
  sorry

end tangent_line_equation_l1438_143881


namespace least_distinct_values_l1438_143843

variable (L : List Nat) (h_len : L.length = 2023) (mode : Nat) 
variable (h_mode_unique : ∀ x ∈ L, L.count x ≤ 15 → x = mode)
variable (h_mode_count : L.count mode = 15)

theorem least_distinct_values : ∃ k, k = 145 ∧ (∀ d ∈ L, List.count d L ≤ 15) :=
by
  sorry

end least_distinct_values_l1438_143843


namespace triangle_perpendicular_bisector_properties_l1438_143824

variables {A B C A1 A2 B1 B2 C1 C2 : Type} (triangle : triangle A B C)
  (A1_perpendicular : dropping_perpendicular_to_bisector A )
  (A2_perpendicular : dropping_perpendicular_to_bisector A )
  (B1_perpendicular : dropping_perpendicular_to_bisector B )
  (B2_perpendicular : dropping_perpendicular_to_bisector B )
  (C1_perpendicular : dropping_perpendicular_to_bisector C )
  (C2_perpendicular : dropping_perpendicular_to_bisector C )
  
-- Defining required structures
structure triangle (A B C : Type) :=
  (AB BC CA : ℝ)

structure dropping_perpendicular_to_bisector (v : Type) :=
  (perpendicular_to_bisector : ℝ)

namespace triangle_properties

theorem triangle_perpendicular_bisector_properties :
  2 * (A1_perpendicular.perpendicular_to_bisector + A2_perpendicular.perpendicular_to_bisector + 
       B1_perpendicular.perpendicular_to_bisector + B2_perpendicular.perpendicular_to_bisector + 
       C1_perpendicular.perpendicular_to_bisector + C2_perpendicular.perpendicular_to_bisector) = 
  (triangle.AB + triangle.BC + triangle.CA) :=
sorry

end triangle_properties

end triangle_perpendicular_bisector_properties_l1438_143824


namespace cost_to_produce_program_l1438_143804

theorem cost_to_produce_program
  (advertisement_revenue : ℝ)
  (number_of_copies : ℝ)
  (price_per_copy : ℝ)
  (desired_profit : ℝ)
  (total_revenue : ℝ)
  (revenue_from_sales : ℝ)
  (cost_to_produce : ℝ) :
  advertisement_revenue = 15000 →
  number_of_copies = 35000 →
  price_per_copy = 0.5 →
  desired_profit = 8000 →
  total_revenue = advertisement_revenue + desired_profit →
  revenue_from_sales = number_of_copies * price_per_copy →
  total_revenue = revenue_from_sales + cost_to_produce →
  cost_to_produce = 5500 :=
by
  sorry

end cost_to_produce_program_l1438_143804


namespace sin_add_pi_over_three_l1438_143845

theorem sin_add_pi_over_three (α : ℝ) (h : Real.sin (α - 2 * Real.pi / 3) = 1 / 4) : 
  Real.sin (α + Real.pi / 3) = -1 / 4 := by
  sorry

end sin_add_pi_over_three_l1438_143845


namespace triplet_A_sums_to_2_triplet_B_sums_to_2_triplet_C_sums_to_2_l1438_143886

theorem triplet_A_sums_to_2 : (1/4 + 1/4 + 3/2 = 2) := by
  sorry

theorem triplet_B_sums_to_2 : (3 + -1 + 0 = 2) := by
  sorry

theorem triplet_C_sums_to_2 : (0.2 + 0.7 + 1.1 = 2) := by
  sorry

end triplet_A_sums_to_2_triplet_B_sums_to_2_triplet_C_sums_to_2_l1438_143886


namespace perfect_square_iff_l1438_143878

theorem perfect_square_iff (x y z : ℕ) (hx : x ≥ y) (hy : y ≥ z) (hz : z > 0) :
  ∃ k : ℕ, 4^x + 4^y + 4^z = k^2 ↔ ∃ b : ℕ, b > 0 ∧ x = 2 * b - 1 + z ∧ y = b + z :=
by
  sorry

end perfect_square_iff_l1438_143878


namespace intersection_point_l1438_143861

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4 * x + 3) / (2 * x - 6)
noncomputable def g (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) / (x - 3)

theorem intersection_point (a b c : ℝ) (h_asymp : ¬(2 = 0) ∧ (a ≠ 0 ∨ b ≠ 0)) (h_perpendicular : True) (h_y_intersect : g a b c 0 = 1) (h_intersects : f (-1) = g a b c (-1)):
  f 1 = 0 :=
by
  dsimp [f, g] at *
  sorry

end intersection_point_l1438_143861


namespace sally_initial_orange_balloons_l1438_143828

def initial_orange_balloons (found_orange : ℝ) (total_orange : ℝ) : ℝ := 
  total_orange - found_orange

theorem sally_initial_orange_balloons : initial_orange_balloons 2.0 11 = 9 := 
by
  sorry

end sally_initial_orange_balloons_l1438_143828


namespace complement_of_A_in_U_l1438_143830

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set A
def A : Set ℕ := {3, 4, 5}

-- Statement to prove the complement of A with respect to U
theorem complement_of_A_in_U : U \ A = {1, 2, 6} := 
  by sorry

end complement_of_A_in_U_l1438_143830


namespace price_difference_l1438_143817

noncomputable def original_price (final_sale_price discount : ℝ) := final_sale_price / (1 - discount)

noncomputable def after_price_increase (price after_increase : ℝ) := price * (1 + after_increase)

theorem price_difference (final_sale_price : ℝ) (discount : ℝ) (price_increase : ℝ) 
    (h1 : final_sale_price = 85) (h2 : discount = 0.15) (h3 : price_increase = 0.25) : 
    after_price_increase final_sale_price price_increase - original_price final_sale_price discount = 6.25 := 
by 
    sorry

end price_difference_l1438_143817


namespace greatest_divisor_of_product_of_any_four_consecutive_integers_l1438_143806

theorem greatest_divisor_of_product_of_any_four_consecutive_integers :
  ∀ (n : ℕ), 0 < n →
  ∃ k : ℕ, k * 24 = (n * (n + 1) * (n + 2) * (n + 3)) := by
  sorry

end greatest_divisor_of_product_of_any_four_consecutive_integers_l1438_143806


namespace logarithmic_inequality_l1438_143847

theorem logarithmic_inequality : 
  (a = Real.log 9 / Real.log 2) →
  (b = Real.log 27 / Real.log 3) →
  (c = Real.log 15 / Real.log 5) →
  a > b ∧ b > c :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  sorry

end logarithmic_inequality_l1438_143847


namespace digit_to_make_multiple_of_5_l1438_143841

theorem digit_to_make_multiple_of_5 (d : ℕ) (h : 0 ≤ d ∧ d ≤ 9) 
  (N := 71360 + d) : (N % 5 = 0) → (d = 0 ∨ d = 5) :=
by
  sorry

end digit_to_make_multiple_of_5_l1438_143841


namespace angle_between_vectors_is_90_degrees_l1438_143852

noncomputable def vec_angle (v₁ v₂ : ℝ × ℝ) : ℝ :=
sorry -- This would be the implementation that calculates the angle between two vectors

theorem angle_between_vectors_is_90_degrees
  (A B C O : ℝ × ℝ)
  (h1 : dist O A = dist O B)
  (h2 : dist O A = dist O C)
  (h3 : dist O B = dist O C)
  (h4 : 2 • (A - O) = (B - O) + (C - O)) :
  vec_angle (B - A) (C - A) = 90 :=
sorry

end angle_between_vectors_is_90_degrees_l1438_143852


namespace determine_friends_l1438_143814

inductive Grade
| first
| second
| third
| fourth

inductive Name
| Petya
| Kolya
| Alyosha
| Misha
| Dima
| Borya
| Vasya

inductive Surname
| Ivanov
| Krylov
| Petrov
| Orlov

structure Friend :=
  (name : Name)
  (surname : Surname)
  (grade : Grade)

def friends : List Friend :=
  [ {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first},
    {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second},
    {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third},
    {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ]

theorem determine_friends : ∃ l : List Friend, 
  {name := Name.Dima, surname := Surname.Ivanov, grade := Grade.first} ∈ l ∧
  {name := Name.Misha, surname := Surname.Krylov, grade := Grade.second} ∈ l ∧
  {name := Name.Borya, surname := Surname.Petrov, grade := Grade.third} ∈ l ∧
  {name := Name.Vasya, surname := Surname.Orlov, grade := Grade.fourth} ∈ l :=
by 
  use friends
  repeat { simp [friends] }


end determine_friends_l1438_143814


namespace pallet_weight_l1438_143810

theorem pallet_weight (box_weight : ℕ) (num_boxes : ℕ) (total_weight : ℕ) 
  (h1 : box_weight = 89) (h2 : num_boxes = 3) : total_weight = 267 := by
  sorry

end pallet_weight_l1438_143810


namespace parameterized_curve_is_line_l1438_143819

theorem parameterized_curve_is_line :
  ∀ (t : ℝ), ∃ (m b : ℝ), y = 5 * ((x - 5) / 3) - 3 → y = (5 * x - 34) / 3 := 
by
  sorry

end parameterized_curve_is_line_l1438_143819


namespace determine_true_proposition_l1438_143857

def proposition_p : Prop :=
  ∃ x : ℝ, Real.tan x > 1

def proposition_q : Prop :=
  let focus_distance := 3/4 -- Distance from the focus to the directrix in y = (1/3)x^2
  focus_distance = 1/6

def true_proposition : Prop :=
  proposition_p ∧ ¬proposition_q

theorem determine_true_proposition :
  (proposition_p ∧ ¬proposition_q) = true_proposition :=
by
  sorry -- Proof will go here

end determine_true_proposition_l1438_143857


namespace gcd_180_270_eq_90_l1438_143883

theorem gcd_180_270_eq_90 : Nat.gcd 180 270 = 90 := sorry

end gcd_180_270_eq_90_l1438_143883


namespace tree_growth_rate_consistency_l1438_143865

theorem tree_growth_rate_consistency (a b : ℝ) :
  (a + b) / 2 = 0.15 ∧ (1 + a) * (1 + b) = 0.90 → ∃ a b : ℝ, (a + b) / 2 = 0.15 ∧ (1 + a) * (1 + b) = 0.90 := by
  sorry

end tree_growth_rate_consistency_l1438_143865


namespace part1_part2_l1438_143870

-- Definition of p: x² + 2x - 8 < 0
def p (x : ℝ) : Prop := x^2 + 2 * x - 8 < 0

-- Definition of q: (x - 1 + m)(x - 1 - m) ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := (x - 1 + m) * (x - 1 - m) ≤ 0

-- Define A as the set of real numbers that satisfy p
def A : Set ℝ := { x | p x }

-- Define B as the set of real numbers that satisfy q when m = 2
def B (m : ℝ) : Set ℝ := { x | q x m }

theorem part1 : A ∩ B 2 = { x | -1 ≤ x ∧ x < 2 } :=
sorry

-- Prove that m ≥ 5 is the range for which p is a sufficient but not necessary condition for q
theorem part2 : ∀ m : ℝ, (∀ x: ℝ, p x → q x m) ∧ (∃ x: ℝ, q x m ∧ ¬p x) ↔ m ≥ 5 :=
sorry

end part1_part2_l1438_143870


namespace find_number_l1438_143844

theorem find_number : ∃ n : ℕ, ∃ q : ℕ, ∃ r : ℕ, q = 6 ∧ r = 4 ∧ n = 9 * q + r ∧ n = 58 :=
by
  sorry

end find_number_l1438_143844


namespace half_angle_second_quadrant_l1438_143803

theorem half_angle_second_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (k * π + π / 4 < α / 2 ∧ α / 2 < k * π + π / 2) :=
sorry

end half_angle_second_quadrant_l1438_143803


namespace digit_1C3_multiple_of_3_l1438_143846

theorem digit_1C3_multiple_of_3 :
  (∃ C : Fin 10, (1 + C.val + 3) % 3 = 0) ∧
  (∀ C : Fin 10, (1 + C.val + 3) % 3 = 0 → (C.val = 2 ∨ C.val = 5 ∨ C.val = 8)) :=
by
  sorry

end digit_1C3_multiple_of_3_l1438_143846


namespace no_valid_placement_for_digits_on_45gon_l1438_143827

theorem no_valid_placement_for_digits_on_45gon (f : Fin 45 → Fin 10) :
  ¬ ∀ (a b : Fin 10), a ≠ b → ∃ (i j : Fin 45), i ≠ j ∧ f i = a ∧ f j = b :=
by {
  sorry
}

end no_valid_placement_for_digits_on_45gon_l1438_143827


namespace exists_unique_t_exists_m_pos_l1438_143833

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem exists_unique_t (m : ℝ) (h : m = 1) : 
  ∃! (t : ℝ), t ∈ Set.Ioc (1 / 2) 1 ∧ deriv (f 1) t = 0 := sorry

theorem exists_m_pos : ∃ (m : ℝ), 0 < m ∧ m < 1 ∧ ∀ (x : ℝ), 0 < x → f m x > 0 := sorry

end exists_unique_t_exists_m_pos_l1438_143833


namespace real_solution_count_l1438_143825

theorem real_solution_count : 
  ∃ (n : ℕ), n = 1 ∧
    ∀ x : ℝ, 
      (3 * x / (x ^ 2 + 2 * x + 4) + 4 * x / (x ^ 2 - 4 * x + 4) = 1) ↔ (x = 2) :=
by
  sorry

end real_solution_count_l1438_143825


namespace paths_A_to_C_l1438_143860

theorem paths_A_to_C :
  let paths_AB := 2
  let paths_BD := 3
  let paths_DC := 3
  let paths_AC_direct := 1
  paths_AB * paths_BD * paths_DC + paths_AC_direct = 19 :=
by
  sorry

end paths_A_to_C_l1438_143860


namespace find_a1_l1438_143853

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSequence (a d : α) (n : ℕ) : α :=
  a + (n - 1) * d

noncomputable def sumOfArithmeticSequence (a d : α) (n : ℕ) : α :=
  n * a + d * (n * (n - 1) / 2)

theorem find_a1 (a1 d : α) :
  arithmeticSequence a1 d 2 + arithmeticSequence a1 d 8 = 34 →
  sumOfArithmeticSequence a1 d 4 = 38 →
  a1 = 5 :=
by
  intros h1 h2
  sorry

end find_a1_l1438_143853


namespace average_marks_two_classes_correct_l1438_143863

axiom average_marks_first_class : ℕ → ℕ → ℕ
axiom average_marks_second_class : ℕ → ℕ → ℕ
axiom combined_average_marks_correct : ℕ → ℕ → Prop

theorem average_marks_two_classes_correct :
  average_marks_first_class 39 45 = 39 * 45 →
  average_marks_second_class 35 70 = 35 * 70 →
  combined_average_marks_correct (average_marks_first_class 39 45) (average_marks_second_class 35 70) :=
by
  intros h1 h2
  sorry

end average_marks_two_classes_correct_l1438_143863


namespace question1_effective_purification_16days_question2_min_mass_optimal_purification_l1438_143801

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then x^2 / 16 + 2
else if x > 4 then (x + 14) / (2 * x - 2)
else 0

-- Effective Purification Conditions
def effective_purification (m : ℝ) (x : ℝ) : Prop := m * f x ≥ 4

-- Optimal Purification Conditions
def optimal_purification (m : ℝ) (x : ℝ) : Prop := 4 ≤ m * f x ∧ m * f x ≤ 10

-- Proof for Question 1
theorem question1_effective_purification_16days (x : ℝ) (hx : 0 < x ∧ x ≤ 16) :
  effective_purification 4 x :=
by sorry

-- Finding Minimum m for Optimal Purification within 7 days
theorem question2_min_mass_optimal_purification :
  ∃ m : ℝ, (16 / 7 ≤ m ∧ m ≤ 10 / 3) ∧ ∀ (x : ℝ), (0 < x ∧ x ≤ 7) → optimal_purification m x :=
by sorry

end question1_effective_purification_16days_question2_min_mass_optimal_purification_l1438_143801


namespace study_group_books_l1438_143839

theorem study_group_books (x n : ℕ) (h1 : n = 5 * x - 2) (h2 : n = 4 * x + 3) : x = 5 ∧ n = 23 := by
  sorry

end study_group_books_l1438_143839
