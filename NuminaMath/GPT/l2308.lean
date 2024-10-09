import Mathlib

namespace find_abc_and_sqrt_l2308_230822

theorem find_abc_and_sqrt (a b c : ℤ) (h1 : 3 * a - 2 * b - 1 = 9) (h2 : a + 2 * b = -8) (h3 : c = Int.floor (2 + Real.sqrt 7)) :
  a = 2 ∧ b = -2 ∧ c = 4 ∧ (Real.sqrt (a - b + c) = 2 * Real.sqrt 2 ∨ Real.sqrt (a - b + c) = -2 * Real.sqrt 2) :=
by
  -- proof details go here
  sorry

end find_abc_and_sqrt_l2308_230822


namespace original_weight_of_potatoes_l2308_230865

theorem original_weight_of_potatoes (W : ℝ) (h : W / (W / 2) = 36) : W = 648 := by
  sorry

end original_weight_of_potatoes_l2308_230865


namespace minimize_fraction_l2308_230826

theorem minimize_fraction (n : ℕ) (h : 0 < n) : 
  (n = 9) ↔ (∀ m : ℕ, 0 < m → (n / 3 + 27 / n) ≤ (m / 3 + 27 / m)) :=
by
  sorry

end minimize_fraction_l2308_230826


namespace functional_relationship_y1_daily_gross_profit_1120_first_10_days_total_gross_profit_W_l2308_230811

-- Conditions for y1
def cost_price : ℕ := 60
def selling_price_first_10_days : ℕ := 80
def y1 : ℕ → ℕ := fun x => x * x - 8 * x + 56
def items_sold_day4 : ℕ := 40
def items_sold_day6 : ℕ := 44

-- Conditions for y2
def selling_price_post_10_days : ℕ := 100
def y2 : ℕ → ℕ := fun x => 2 * x + 8
def gross_profit_condition : ℕ := 1120

-- 1) Prove functional relationship of y1.
theorem functional_relationship_y1 (x : ℕ) (h4 : y1 4 = 40) (h6 : y1 6 = 44) : 
  y1 x = x * x - 8 * x + 56 := 
by
  sorry

-- 2) Prove value of x for daily gross profit $1120 on any day within first 10 days.
theorem daily_gross_profit_1120_first_10_days (x : ℕ) (h4 : y1 4 = 40) (h6 : y1 6 = 44) (gp : (selling_price_first_10_days - cost_price) * y1 x = gross_profit_condition) : 
  x = 8 := 
by
  sorry

-- 3) Prove total gross profit W and range for 26 < x ≤ 31.
theorem total_gross_profit_W (x : ℕ) (h : 26 < x ∧ x ≤ 31) : 
  (100 - (cost_price - 2 * (y2 x - 60))) * (y2 x) = 8 * x * x - 96 * x - 512 := 
by
  sorry

end functional_relationship_y1_daily_gross_profit_1120_first_10_days_total_gross_profit_W_l2308_230811


namespace basic_astrophysics_degrees_l2308_230881

def budget_allocation : Nat := 100
def microphotonics_perc : Nat := 14
def home_electronics_perc : Nat := 19
def food_additives_perc : Nat := 10
def genetically_modified_perc : Nat := 24
def industrial_lubricants_perc : Nat := 8

def arc_of_sector (percentage : Nat) : Nat := percentage * 360 / budget_allocation

theorem basic_astrophysics_degrees :
  arc_of_sector (budget_allocation - (microphotonics_perc + home_electronics_perc + food_additives_perc + genetically_modified_perc + industrial_lubricants_perc)) = 90 :=
  by
  sorry

end basic_astrophysics_degrees_l2308_230881


namespace hyperbola_eccentricity_l2308_230866

theorem hyperbola_eccentricity 
  (a b e : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : e = Real.sqrt (1 + (b^2 / a^2))) 
  (h4 : e ≤ Real.sqrt 5) : 
  e = 2 := 
sorry

end hyperbola_eccentricity_l2308_230866


namespace quadratic_inequality_real_solutions_l2308_230814

theorem quadratic_inequality_real_solutions (c : ℝ) (h1 : 0 < c) (h2 : c < 16) :
  ∃ x : ℝ, x^2 - 8*x + c < 0 :=
sorry

end quadratic_inequality_real_solutions_l2308_230814


namespace solve_for_x_l2308_230801

theorem solve_for_x (x : ℕ) : (1 : ℚ) / 2 = x / 8 → x = 4 := by
  sorry

end solve_for_x_l2308_230801


namespace simplify_fraction_l2308_230809

theorem simplify_fraction : (8 / (5 * 42) = 4 / 105) :=
by
    sorry

end simplify_fraction_l2308_230809


namespace ratio_IM_IN_l2308_230846

noncomputable def compute_ratio (IA IB IC ID : ℕ) (M N : ℕ) : ℚ :=
  (IA * IC : ℚ) / (IB * ID : ℚ)

theorem ratio_IM_IN (IA IB IC ID : ℕ) (hIA : IA = 12) (hIB : IB = 16) (hIC : IC = 14) (hID : ID = 11) :
  compute_ratio IA IB IC ID = 21 / 22 := by
  rw [hIA, hIB, hIC, hID]
  sorry

end ratio_IM_IN_l2308_230846


namespace cost_of_paint_per_kg_l2308_230886

/-- The cost of painting one square foot is Rs. 50. -/
theorem cost_of_paint_per_kg (side_length : ℝ) (cost_total : ℝ) (coverage_per_kg : ℝ) (total_surface_area : ℝ) (total_paint_needed : ℝ) (cost_per_kg : ℝ) 
  (h1 : side_length = 20)
  (h2 : cost_total = 6000)
  (h3 : coverage_per_kg = 20)
  (h4 : total_surface_area = 6 * side_length^2)
  (h5 : total_paint_needed = total_surface_area / coverage_per_kg)
  (h6 : cost_per_kg = cost_total / total_paint_needed) :
  cost_per_kg = 50 :=
sorry

end cost_of_paint_per_kg_l2308_230886


namespace rectangle_area_is_432_l2308_230872

-- Definition of conditions and problem in Lean 4
noncomputable def circle_radius : ℝ := 6
noncomputable def rectangle_ratio_length_width : ℝ := 3 / 1
noncomputable def calculate_rectangle_area (radius : ℝ) (ratio : ℝ) : ℝ :=
  let diameter := 2 * radius
  let width := diameter
  let length := ratio * width
  length * width

-- Lean statement to prove the area
theorem rectangle_area_is_432 : calculate_rectangle_area circle_radius rectangle_ratio_length_width = 432 := by
  sorry

end rectangle_area_is_432_l2308_230872


namespace find_value_of_x_squared_plus_one_over_x_squared_l2308_230873

theorem find_value_of_x_squared_plus_one_over_x_squared (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
by 
  sorry

end find_value_of_x_squared_plus_one_over_x_squared_l2308_230873


namespace solve_for_x_l2308_230803

theorem solve_for_x (x : ℝ) (h : 3 * x + 1 = -(5 - 2 * x)) : x = -6 :=
by
  sorry

end solve_for_x_l2308_230803


namespace coffee_price_l2308_230869

theorem coffee_price (qd : ℝ) (d : ℝ) (rp : ℝ) :
  qd = 4.5 ∧ d = 0.25 → rp = 12 :=
by 
  sorry

end coffee_price_l2308_230869


namespace millions_place_correct_l2308_230884

def number := 345000000
def hundred_millions_place := number / 100000000 % 10  -- 3
def ten_millions_place := number / 10000000 % 10  -- 4
def millions_place := number / 1000000 % 10  -- 5

theorem millions_place_correct : millions_place = 5 := 
by 
  -- Mathematical proof goes here
  sorry

end millions_place_correct_l2308_230884


namespace trees_died_proof_l2308_230849

def treesDied (original : Nat) (remaining : Nat) : Nat := original - remaining

theorem trees_died_proof : treesDied 20 4 = 16 := by
  -- Here we put the steps needed to prove the theorem, which is essentially 20 - 4 = 16.
  sorry

end trees_died_proof_l2308_230849


namespace max_intersection_distance_l2308_230856

theorem max_intersection_distance :
  let C1_x (α : ℝ) := 2 + 2 * Real.cos α
  let C1_y (α : ℝ) := 2 * Real.sin α
  let C2_x (β : ℝ) := 2 * Real.cos β
  let C2_y (β : ℝ) := 2 + 2 * Real.sin β
  let l1 (α : ℝ) := α
  let l2 (α : ℝ) := α - Real.pi / 6
  (0 < Real.pi / 2) →
  let OP (α : ℝ) := 4 * Real.cos α
  let OQ (α : ℝ) := 4 * Real.sin (α - Real.pi / 6)
  let pq_prod (α : ℝ) := OP α * OQ α
  ∀α, 0 < α ∧ α < Real.pi / 2 → pq_prod α ≤ 4 := by
  sorry

end max_intersection_distance_l2308_230856


namespace original_proposition_converse_inverse_contrapositive_l2308_230895

def is_integer (x : ℝ) : Prop := ∃ (n : ℤ), x = n
def is_real (x : ℝ) : Prop := true

theorem original_proposition (x : ℝ) : is_integer x → is_real x := 
by sorry

theorem converse (x : ℝ) : ¬(is_real x → is_integer x) := 
by sorry

theorem inverse (x : ℝ) : ¬((¬ is_integer x) → (¬ is_real x)) := 
by sorry

theorem contrapositive (x : ℝ) : (¬ is_real x) → (¬ is_integer x) := 
by sorry

end original_proposition_converse_inverse_contrapositive_l2308_230895


namespace trigonometric_identity_proof_l2308_230854

variable (α : ℝ)

theorem trigonometric_identity_proof
  (h : Real.tan (α + Real.pi / 4) = -3) :
  Real.cos (2 * α) + 2 * Real.sin (2 * α) = 1 :=
by
  sorry

end trigonometric_identity_proof_l2308_230854


namespace quadratic_solution_l2308_230853

theorem quadratic_solution (x : ℝ) : 
  x^2 - 2 * x - 3 = 0 → (x = 3 ∨ x = -1) :=
by 
  sorry

end quadratic_solution_l2308_230853


namespace log_ordering_l2308_230842

theorem log_ordering {x a b c : ℝ} (h1 : 1 < x) (h2 : x < 10) (ha : a = Real.log x^2) (hb : b = Real.log (Real.log x)) (hc : c = (Real.log x)^2) :
  a > c ∧ c > b :=
by
  sorry

end log_ordering_l2308_230842


namespace fish_in_aquarium_l2308_230871

theorem fish_in_aquarium (initial_fish : ℕ) (added_fish : ℕ) (h1 : initial_fish = 10) (h2 : added_fish = 3) : initial_fish + added_fish = 13 := by
  sorry

end fish_in_aquarium_l2308_230871


namespace least_n_divisibility_l2308_230859

theorem least_n_divisibility :
  ∃ n : ℕ, 0 < n ∧ (∀ k : ℕ, 2 ≤ k ∧ k ≤ n → k ∣ (n - 1)^2) ∧ (∃ k : ℕ, 2 ≤ k ∧ k ≤ n ∧ ¬ k ∣ (n - 1)^2) ∧ n = 3 :=
by
  sorry

end least_n_divisibility_l2308_230859


namespace retailer_profit_percentage_l2308_230897

theorem retailer_profit_percentage (items_sold : ℕ) (profit_per_item : ℝ) (discount_rate : ℝ)
  (discounted_items_needed : ℝ) (total_profit : ℝ) (item_cost : ℝ) :
  items_sold = 100 → 
  profit_per_item = 30 →
  discount_rate = 0.05 →
  discounted_items_needed = 156.86274509803923 →
  total_profit = 3000 →
  (discounted_items_needed * ((item_cost + profit_per_item) * (1 - discount_rate) - item_cost) = total_profit) →
  ((profit_per_item / item_cost) * 100 = 16) :=
by {
  sorry 
}

end retailer_profit_percentage_l2308_230897


namespace shared_candy_equally_l2308_230850

def Hugh_candy : ℕ := 8
def Tommy_candy : ℕ := 6
def Melany_candy : ℕ := 7
def total_people : ℕ := 3

theorem shared_candy_equally : 
  (Hugh_candy + Tommy_candy + Melany_candy) / total_people = 7 := 
by 
  sorry

end shared_candy_equally_l2308_230850


namespace jackson_earnings_l2308_230848

def hourly_rate_usd : ℝ := 5
def hourly_rate_gbp : ℝ := 3
def hourly_rate_jpy : ℝ := 400

def hours_vacuuming : ℝ := 2
def sessions_vacuuming : ℝ := 2

def hours_washing_dishes : ℝ := 0.5
def hours_cleaning_bathroom := hours_washing_dishes * 3

def exchange_rate_gbp_to_usd : ℝ := 1.35
def exchange_rate_jpy_to_usd : ℝ := 0.009

def earnings_in_usd : ℝ := (hours_vacuuming * sessions_vacuuming * hourly_rate_usd)
def earnings_in_gbp : ℝ := (hours_washing_dishes * hourly_rate_gbp)
def earnings_in_jpy : ℝ := (hours_cleaning_bathroom * hourly_rate_jpy)

def converted_gbp_to_usd : ℝ := earnings_in_gbp * exchange_rate_gbp_to_usd
def converted_jpy_to_usd : ℝ := earnings_in_jpy * exchange_rate_jpy_to_usd

def total_earnings_usd : ℝ := earnings_in_usd + converted_gbp_to_usd + converted_jpy_to_usd

theorem jackson_earnings : total_earnings_usd = 27.425 := by
  sorry

end jackson_earnings_l2308_230848


namespace log_equation_l2308_230888

theorem log_equation (x : ℝ) (h0 : x < 1) (h1 : (Real.log x / Real.log 10)^3 - 3 * (Real.log x / Real.log 10) = 243) :
  (Real.log x / Real.log 10)^4 - 4 * (Real.log x / Real.log 10) = 6597 :=
by
  sorry

end log_equation_l2308_230888


namespace number_of_perfect_numbers_l2308_230832

-- Define the concept of a perfect number
def perfect_number (a b : ℕ) : ℕ := (a + b)^2

-- Define the proposition we want to prove
theorem number_of_perfect_numbers : ∃ n : ℕ, n = 15 ∧ 
  ∀ p, ∃ a b : ℕ, p = perfect_number a b ∧ p < 200 :=
sorry

end number_of_perfect_numbers_l2308_230832


namespace sqrt_sum_bounds_l2308_230818

theorem sqrt_sum_bounds (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
    4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2 - b)^2) + 
                   Real.sqrt (b^2 + (2 - c)^2) + 
                   Real.sqrt (c^2 + (2 - d)^2) + 
                   Real.sqrt (d^2 + (2 - a)^2) ∧
    Real.sqrt (a^2 + (2 - b)^2) + 
    Real.sqrt (b^2 + (2 - c)^2) + 
    Real.sqrt (c^2 + (2 - d)^2) + 
    Real.sqrt (d^2 + (2 - a)^2) ≤ 8 :=
sorry

end sqrt_sum_bounds_l2308_230818


namespace problem_solution_l2308_230864

theorem problem_solution (a0 a1 a2 a3 a4 a5 : ℝ) :
  (1 + 2*x)^5 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 →
  a0 + a2 + a4 = 121 := 
sorry

end problem_solution_l2308_230864


namespace find_rate_percent_l2308_230857

theorem find_rate_percent (SI P T : ℝ) (h1 : SI = 160) (h2 : P = 800) (h3 : T = 5) : P * (4:ℝ) * T / 100 = SI :=
by
  sorry

end find_rate_percent_l2308_230857


namespace gain_percent_is_100_l2308_230807

variable {C S : ℝ}

-- Given conditions
axiom h1 : 50 * C = 25 * S
axiom h2 : S = 2 * C

-- Prove the gain percent is 100%
theorem gain_percent_is_100 (h1 : 50 * C = 25 * S) (h2 : S = 2 * C) : (S - C) / C * 100 = 100 :=
by
  sorry

end gain_percent_is_100_l2308_230807


namespace f_value_at_3_l2308_230885

def f (x : ℝ) := 2 * (x + 1) + 1

theorem f_value_at_3 : f 3 = 9 :=
by sorry

end f_value_at_3_l2308_230885


namespace min_value_expression_l2308_230845

theorem min_value_expression (x y : ℝ) : (∃ z : ℝ, (forall x y : ℝ, z ≤ 5*x^2 + 4*y^2 - 8*x*y + 2*x + 4) ∧ z = 3) :=
sorry

end min_value_expression_l2308_230845


namespace min_passengers_to_fill_bench_l2308_230877

theorem min_passengers_to_fill_bench (width_per_passenger : ℚ) (total_seat_width : ℚ) (num_seats : ℕ):
  width_per_passenger = 1/6 → total_seat_width = num_seats → num_seats = 6 → 3 ≥ (total_seat_width / width_per_passenger) :=
by
  intro h1 h2 h3
  sorry

end min_passengers_to_fill_bench_l2308_230877


namespace max_value_of_b_minus_a_l2308_230868

theorem max_value_of_b_minus_a (a b : ℝ) (h₀ : a < 0)
  (h₁ : ∀ x : ℝ, a < x ∧ x < b → (x^2 + 2017 * a) * (x + 2016 * b) ≥ 0) :
  b - a ≤ 2017 :=
sorry

end max_value_of_b_minus_a_l2308_230868


namespace apply_f_2019_times_l2308_230838

noncomputable def f (x : ℝ) : ℝ := (1 - x^3) ^ (-1/3 : ℝ)

theorem apply_f_2019_times (x : ℝ) (n : ℕ) (h : n = 2019) (hx : x = 2018) : 
  (f^[n]) x = 2018 :=
by
  sorry

end apply_f_2019_times_l2308_230838


namespace smallest_prime_with_digit_sum_23_l2308_230889

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem smallest_prime_with_digit_sum_23 :
  ∃ p : ℕ, Prime p ∧ digit_sum p = 23 ∧ ∀ q : ℕ, Prime q ∧ digit_sum q = 23 → p ≤ q :=
by
  sorry

end smallest_prime_with_digit_sum_23_l2308_230889


namespace carlos_goal_l2308_230867

def july_books : ℕ := 28
def august_books : ℕ := 30
def june_books : ℕ := 42

theorem carlos_goal (goal : ℕ) :
  goal = june_books + july_books + august_books := by
  sorry

end carlos_goal_l2308_230867


namespace find_percentage_l2308_230878

theorem find_percentage (P : ℝ) (h : P / 100 * 3200 = 0.20 * 650 + 190) : P = 10 :=
by 
  sorry

end find_percentage_l2308_230878


namespace females_count_l2308_230855

-- Defining variables and constants
variables (P M F : ℕ)
-- The condition given the total population
def town_population := P = 600
-- The condition given the proportion of males
def proportion_of_males := M = P / 3
-- The condition determining the number of females
def number_of_females := F = P - M

-- The theorem stating the number of females is 400
theorem females_count (P M F : ℕ) (h1 : town_population P)
  (h2 : proportion_of_males P M) 
  (h3 : number_of_females P M F) : 
  F = 400 := 
sorry

end females_count_l2308_230855


namespace behavior_on_1_2_l2308_230887

/-- Definition of an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

/-- Definition of being decreasing on an interval -/
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≥ f y

/-- Definition of having a minimum value on an interval -/
def has_minimum_on (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  ∀ x, a ≤ x → x ≤ b → f x ≥ m

theorem behavior_on_1_2 
  {f : ℝ → ℝ} 
  (h_odd : is_odd_function f) 
  (h_dec : is_decreasing_on f (-2) (-1)) 
  (h_min : has_minimum_on f (-2) (-1) 3) :
  is_decreasing_on f 1 2 ∧ ∀ x, 1 ≤ x → x ≤ 2 → f x ≤ -3 := 
by 
  sorry

end behavior_on_1_2_l2308_230887


namespace greatest_integer_n_l2308_230804

theorem greatest_integer_n (n : ℤ) : n^2 - 9 * n + 20 ≤ 0 → n ≤ 5 := sorry

end greatest_integer_n_l2308_230804


namespace value_of_a2_sub_b2_l2308_230816

theorem value_of_a2_sub_b2 (a b : ℝ) (h1 : a + b = 6) (h2 : a - b = 2) : a^2 - b^2 = 12 :=
by
  sorry

end value_of_a2_sub_b2_l2308_230816


namespace larger_number_is_1671_l2308_230839

variable (L S : ℕ)

noncomputable def problem_conditions :=
  L - S = 1395 ∧ L = 6 * S + 15

theorem larger_number_is_1671 (h : problem_conditions L S) : L = 1671 := by
  sorry

end larger_number_is_1671_l2308_230839


namespace rectangle_area_l2308_230861

-- Define the width and length of the rectangle
def w : ℚ := 20 / 3
def l : ℚ := 2 * w

-- Define the perimeter constraint
def perimeter_condition : Prop := 2 * (l + w) = 40

-- Define the area of the rectangle
def area : ℚ := l * w

-- The theorem to prove
theorem rectangle_area : perimeter_condition → area = 800 / 9 :=
by
  intro h
  have hw : w = 20 / 3 := rfl
  have hl : l = 2 * w := rfl
  have hp : 2 * (l + w) = 40 := h
  sorry

end rectangle_area_l2308_230861


namespace probability_three_red_before_two_green_l2308_230870

noncomputable def probability_red_chips_drawn_before_green (red_chips green_chips : ℕ) (total_chips : ℕ) : ℚ := sorry

theorem probability_three_red_before_two_green 
    (red_chips green_chips : ℕ) (total_chips : ℕ)
    (h_red : red_chips = 3) (h_green : green_chips = 2) 
    (h_total: total_chips = red_chips + green_chips) :
  probability_red_chips_drawn_before_green red_chips green_chips total_chips = 3 / 10 :=
  sorry

end probability_three_red_before_two_green_l2308_230870


namespace ellipse_k_range_l2308_230820

theorem ellipse_k_range
  (k : ℝ)
  (h1 : k - 4 > 0)
  (h2 : 10 - k > 0)
  (h3 : k - 4 > 10 - k) :
  7 < k ∧ k < 10 :=
sorry

end ellipse_k_range_l2308_230820


namespace sin_C_value_area_of_triangle_l2308_230835

open Real
open Classical

variable {A B C a b c : ℝ}

-- Given conditions
axiom h1 : b = sqrt 2
axiom h2 : c = 1
axiom h3 : cos B = 3 / 4

-- Proof statements
theorem sin_C_value : sin C = sqrt 14 / 8 := sorry

theorem area_of_triangle : 1 / 2 * b * c * sin (B + C) = sqrt 7 / 4 := sorry

end sin_C_value_area_of_triangle_l2308_230835


namespace intersection_point_for_m_l2308_230891

variable (n : ℕ) (x_0 y_0 : ℕ)
variable (h₁ : n ≥ 2)
variable (h₂ : y_0 ^ 2 = n * x_0 - 1)
variable (h₃ : y_0 = x_0)

theorem intersection_point_for_m (m : ℕ) (hm : 0 < m) : ∃ k : ℕ, k ≥ 2 ∧ (y_0 ^ m = x_0 ^ m) ∧ (y_0 ^ m) ^ 2 = k * (x_0 ^ m) - 1 :=
by
  sorry

end intersection_point_for_m_l2308_230891


namespace gcd_abcd_dcba_l2308_230876

-- Definitions based on the conditions
def abcd (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d
def dcba (a b c d : ℕ) : ℕ := 1000 * d + 100 * c + 10 * b + a
def consecutive_digits (a b c d : ℕ) : Prop := (b = a + 1) ∧ (c = a + 2) ∧ (d = a + 3)

-- Theorem statement
theorem gcd_abcd_dcba (a b c d : ℕ) (h : consecutive_digits a b c d) : 
  Nat.gcd (abcd a b c d + dcba a b c d) 1111 = 1111 :=
sorry

end gcd_abcd_dcba_l2308_230876


namespace cuboid_dimensions_sum_l2308_230810

theorem cuboid_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 45) 
  (h2 : B * C = 80) 
  (h3 : C * A = 180) : 
  A + B + C = 145 / 9 :=
sorry

end cuboid_dimensions_sum_l2308_230810


namespace find_n_l2308_230833

-- Define the original and new parabola conditions
def original_parabola (x : ℝ) : ℝ := x^2 + 4 * x + 3
noncomputable def new_parabola (x n : ℝ) : ℝ := (x - n + 2)^2 - 1

-- Define the conditions for points A and B lying on the new parabola
def point_A (n : ℝ) : Prop := ∃ y₁ : ℝ, new_parabola 2 n = y₁
def point_B (n : ℝ) : Prop := ∃ y₂ : ℝ, new_parabola 4 n = y₂

-- Define the condition that y1 > y2
def points_condition (n : ℝ) : Prop := ∃ y₁ y₂ : ℝ, new_parabola 2 n = y₁ ∧ new_parabola 4 n = y₂ ∧ y₁ > y₂

-- Prove that n = 6 is the necessary value given the conditions
theorem find_n : ∀ n, (0 < n) → point_A n ∧ point_B n ∧ points_condition n → n = 6 :=
  by
    sorry

end find_n_l2308_230833


namespace time_A_reaches_destination_l2308_230875

theorem time_A_reaches_destination (x t : ℝ) (h_ratio : (4 * t) = 3 * (t + 0.5)) : (t + 0.5) = 2 :=
by {
  -- derived by algebraic manipulation
  sorry
}

end time_A_reaches_destination_l2308_230875


namespace sqrt_inequality_l2308_230824

theorem sqrt_inequality : 2 * Real.sqrt 2 - Real.sqrt 7 < Real.sqrt 6 - Real.sqrt 5 := by sorry

end sqrt_inequality_l2308_230824


namespace geometric_sequence_first_term_l2308_230894

theorem geometric_sequence_first_term (a b c : ℕ) (r : ℕ) (h1 : r = 2) (h2 : b = a * r)
  (h3 : c = b * r) (h4 : 32 = c * r) (h5 : 64 = 32 * r) :
  a = 4 :=
by sorry

end geometric_sequence_first_term_l2308_230894


namespace OH_squared_correct_l2308_230844

noncomputable def OH_squared (O H : Point) (a b c R : ℝ) : ℝ :=
  9 * R^2 - (a^2 + b^2 + c^2)

theorem OH_squared_correct :
  ∀ (O H : Point) (a b c : ℝ) (R : ℝ),
    R = 7 →
    a^2 + b^2 + c^2 = 29 →
    OH_squared O H a b c R = 412 := by
  intros O H a b c R hR habc
  simp [OH_squared, hR, habc]
  sorry

end OH_squared_correct_l2308_230844


namespace sum_of_roots_is_k_over_5_l2308_230883

noncomputable def sum_of_roots 
  (x1 x2 k d : ℝ) 
  (hx : x1 ≠ x2) 
  (h1 : 5 * x1^2 - k * x1 = d) 
  (h2 : 5 * x2^2 - k * x2 = d) : ℝ :=
x1 + x2

theorem sum_of_roots_is_k_over_5 
  {x1 x2 k d : ℝ} 
  (hx : x1 ≠ x2) 
  (h1 : 5 * x1^2 - k * x1 = d) 
  (h2 : 5 * x2^2 - k * x2 = d) : 
  sum_of_roots x1 x2 k d hx h1 h2 = k / 5 :=
sorry

end sum_of_roots_is_k_over_5_l2308_230883


namespace min_toys_to_add_l2308_230837

theorem min_toys_to_add (T x : ℕ) (h1 : T % 12 = 3) (h2 : T % 18 = 3) :
  ((T + x) % 7 = 0) → x = 4 :=
by
  sorry

end min_toys_to_add_l2308_230837


namespace width_of_playground_is_250_l2308_230836

noncomputable def total_area_km2 : ℝ := 0.6
def num_playgrounds : ℕ := 8
def length_of_playground_m : ℝ := 300

theorem width_of_playground_is_250 :
  let total_area_m2 := total_area_km2 * 1000000
  let area_of_one_playground := total_area_m2 / num_playgrounds
  let width_of_playground := area_of_one_playground / length_of_playground_m
  width_of_playground = 250 := by
  sorry

end width_of_playground_is_250_l2308_230836


namespace power_function_decreasing_n_value_l2308_230890

theorem power_function_decreasing_n_value (n : ℤ) (f : ℝ → ℝ) :
  (∀ x : ℝ, 0 < x → f x = (n^2 + 2 * n - 2) * x^(n^2 - 3 * n)) →
  (∀ x y : ℝ, 0 < x ∧ 0 < y → x < y → f y < f x) →
  n = 1 := 
by
  sorry

end power_function_decreasing_n_value_l2308_230890


namespace find_piglets_l2308_230821

theorem find_piglets (chickens piglets goats sick_animals : ℕ) 
  (h1 : chickens = 26) 
  (h2 : goats = 34) 
  (h3 : sick_animals = 50) 
  (h4 : (chickens + piglets + goats) / 2 = sick_animals) : piglets = 40 := 
by
  sorry

end find_piglets_l2308_230821


namespace total_cost_of_books_l2308_230843

def book_cost (num_mathbooks num_artbooks num_sciencebooks cost_mathbook cost_artbook cost_sciencebook : ℕ) : ℕ :=
  (num_mathbooks * cost_mathbook) + (num_artbooks * cost_artbook) + (num_sciencebooks * cost_sciencebook)

theorem total_cost_of_books :
  let num_mathbooks := 2
  let num_artbooks := 3
  let num_sciencebooks := 6
  let cost_mathbook := 3
  let cost_artbook := 2
  let cost_sciencebook := 3
  book_cost num_mathbooks num_artbooks num_sciencebooks cost_mathbook cost_artbook cost_sciencebook = 30 :=
by
  sorry

end total_cost_of_books_l2308_230843


namespace johns_total_weekly_gas_consumption_l2308_230880

-- Definitions of conditions
def highway_mpg : ℝ := 30
def city_mpg : ℝ := 25
def work_miles_each_way : ℝ := 20
def work_days_per_week : ℝ := 5
def highway_miles_each_way : ℝ := 15
def city_miles_each_way : ℝ := 5
def leisure_highway_miles_per_week : ℝ := 30
def leisure_city_miles_per_week : ℝ := 10
def idling_gas_consumption_per_week : ℝ := 0.3

-- Proof problem
theorem johns_total_weekly_gas_consumption :
  let work_commute_miles_per_week := work_miles_each_way * 2 * work_days_per_week
  let highway_miles_work := highway_miles_each_way * 2 * work_days_per_week
  let city_miles_work := city_miles_each_way * 2 * work_days_per_week
  let total_highway_miles := highway_miles_work + leisure_highway_miles_per_week
  let total_city_miles := city_miles_work + leisure_city_miles_per_week
  let highway_gas_consumption := total_highway_miles / highway_mpg
  let city_gas_consumption := total_city_miles / city_mpg
  (highway_gas_consumption + city_gas_consumption + idling_gas_consumption_per_week) = 8.7 := by
  sorry

end johns_total_weekly_gas_consumption_l2308_230880


namespace gain_percentage_l2308_230879

theorem gain_percentage (selling_price gain : ℝ) (h_selling : selling_price = 90) (h_gain : gain = 15) : 
  (gain / (selling_price - gain)) * 100 = 20 := 
by
  sorry

end gain_percentage_l2308_230879


namespace complement_of_M_in_U_l2308_230815

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}

theorem complement_of_M_in_U : (U \ M) = {2, 4, 6} :=
by
  sorry

end complement_of_M_in_U_l2308_230815


namespace coefficient_of_x_in_expansion_l2308_230882

noncomputable def binomial_expansion_term (r : ℕ) : ℤ :=
  (-1)^r * (2^(5-r)) * Nat.choose 5 r

theorem coefficient_of_x_in_expansion :
  binomial_expansion_term 3 = -40 := by
  sorry

end coefficient_of_x_in_expansion_l2308_230882


namespace inequality_solution_set_l2308_230805

theorem inequality_solution_set (x : ℝ) : (|x - 1| + 2 * x > 4) ↔ (x > 3) := 
sorry

end inequality_solution_set_l2308_230805


namespace smallest_possible_c_l2308_230817

theorem smallest_possible_c 
  (a b c : ℕ) (hp : a > 0 ∧ b > 0 ∧ c > 0) 
  (hg : b^2 = a * c) 
  (ha : 2 * c = a + b) : 
  c = 2 :=
by
  sorry

end smallest_possible_c_l2308_230817


namespace ages_sum_l2308_230874

theorem ages_sum (Beckett_age Olaf_age Shannen_age Jack_age : ℕ) 
  (h1 : Beckett_age = 12) 
  (h2 : Olaf_age = Beckett_age + 3) 
  (h3 : Shannen_age = Olaf_age - 2) 
  (h4 : Jack_age = 2 * Shannen_age + 5) : 
  Beckett_age + Olaf_age + Shannen_age + Jack_age = 71 := 
by
  sorry

end ages_sum_l2308_230874


namespace combined_rocket_height_l2308_230830

theorem combined_rocket_height :
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 :=
by
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  sorry

end combined_rocket_height_l2308_230830


namespace rubber_boat_lost_time_l2308_230827

theorem rubber_boat_lost_time (a b : ℝ) (x : ℝ) (h : (5 - x) * (a - b) + (6 - x) * b = a + b) : x = 4 :=
  sorry

end rubber_boat_lost_time_l2308_230827


namespace rate_of_interest_l2308_230802

theorem rate_of_interest (SI P T R : ℝ) 
  (hSI : SI = 4016.25) 
  (hP : P = 6693.75) 
  (hT : T = 5) 
  (h : SI = (P * R * T) / 100) : 
  R = 12 :=
by 
  sorry

end rate_of_interest_l2308_230802


namespace problem_l2308_230806

noncomputable def f (x : ℝ) (a b : ℝ) := (b - 2^x) / (2^(x+1) + a)

theorem problem (a b k : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) →
  (f 0 a b = 0) → (f (-1) a b = -f 1 a b) → 
  a = 2 ∧ b = 1 ∧ 
  (∀ x y : ℝ, x < y → f x a b > f y a b) ∧ 
  (∀ x : ℝ, x ≥ 1 → f (k * 3^x) a b + f (3^x - 9^x + 2) a b > 0 → k < 4 / 3) :=
by
  sorry

end problem_l2308_230806


namespace consecutive_even_product_6digit_l2308_230828

theorem consecutive_even_product_6digit :
  ∃ (a b c : ℕ), 
  (a % 2 = 0) ∧ (b = a + 2) ∧ (c = a + 4) ∧ 
  (Nat.digits 10 (a * b * c)).length = 6 ∧ 
  (Nat.digits 10 (a * b * c)).head! = 2 ∧ 
  (Nat.digits 10 (a * b * c)).getLast! = 2 ∧ 
  (a * b * c = 287232) :=
by
  sorry

end consecutive_even_product_6digit_l2308_230828


namespace infinite_primes_p_solutions_eq_p2_l2308_230892

theorem infinite_primes_p_solutions_eq_p2 :
  ∃ᶠ p in Filter.atTop, Prime p ∧ 
  (∃ (S : Finset (ZMod p × ZMod p × ZMod p)),
    S.card = p^2 ∧ ∀ (x y z : ZMod p), (3 * x^3 + 4 * y^4 + 5 * z^3 - y^4 * z = 0) ↔ (x, y, z) ∈ S) :=
sorry

end infinite_primes_p_solutions_eq_p2_l2308_230892


namespace solve_x_l2308_230858

theorem solve_x :
  ∀ (x y z w : ℤ),
    x = y + 7 →
    y = z + 15 →
    z = w + 25 →
    w = 65 →
    x = 112 :=
by
  intros x y z w
  intros h1 h2 h3 h4
  sorry

end solve_x_l2308_230858


namespace correct_ordering_of_powers_l2308_230823

theorem correct_ordering_of_powers :
  (6 ^ 8) < (3 ^ 15) ∧ (3 ^ 15) < (8 ^ 10) :=
by
  -- Define the expressions for each power
  let a := (8 : ℕ) ^ 10
  let b := (3 : ℕ) ^ 15
  let c := (6 : ℕ) ^ 8
  
  -- To utilize the values directly in inequalities
  have h1 : (c < b) := sorry -- Proof that 6^8 < 3^15
  have h2 : (b < a) := sorry -- Proof that 3^15 < 8^10

  exact ⟨h1, h2⟩ -- Conjunction of h1 and h2 to show 6^8 < 3^15 < 8^10

end correct_ordering_of_powers_l2308_230823


namespace total_roses_in_a_week_l2308_230831

theorem total_roses_in_a_week : 
  let day1 := 24 
  let day2 := day1 + 6
  let day3 := day2 + 6
  let day4 := day3 + 6
  let day5 := day4 + 6
  let day6 := day5 + 6
  let day7 := day6 + 6
  (day1 + day2 + day3 + day4 + day5 + day6 + day7) = 294 :=
by
  sorry

end total_roses_in_a_week_l2308_230831


namespace cylinder_volume_l2308_230812

theorem cylinder_volume (h : ℝ) (H1 : π * h ^ 2 = 4 * π) : (π * (h / 2) ^ 2 * h) = 2 * π :=
by
  sorry

end cylinder_volume_l2308_230812


namespace xy_equals_18_l2308_230896

theorem xy_equals_18 (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 :=
by
  sorry

end xy_equals_18_l2308_230896


namespace distance_is_30_l2308_230851

-- Define given conditions
def total_distance : ℕ := 120
def trips : ℕ := 4

-- Define the distance from Mrs. Hilt's desk to the water fountain
def distance_to_water_fountain : ℕ := total_distance / trips

-- Prove the distance is 30 feet
theorem distance_is_30 : distance_to_water_fountain = 30 :=
by
  -- Utilizing the division defined in distance_to_water_fountain
  sorry

end distance_is_30_l2308_230851


namespace algebra_expression_value_l2308_230899

theorem algebra_expression_value (a b : ℝ) (h : a - 2 * b = -1) : 1 - 2 * a + 4 * b = 3 :=
by
  sorry

end algebra_expression_value_l2308_230899


namespace tom_savings_by_having_insurance_l2308_230829

noncomputable def insurance_cost_per_month : ℝ := 20
noncomputable def total_months : ℕ := 24
noncomputable def surgery_cost : ℝ := 5000
noncomputable def insurance_coverage_rate : ℝ := 0.80

theorem tom_savings_by_having_insurance :
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  savings = 3520 :=
by
  let total_insurance_cost := (insurance_cost_per_month * total_months)
  let insurance_coverage := (insurance_coverage_rate * surgery_cost)
  let out_of_pocket_cost := (surgery_cost - insurance_coverage)
  let savings := (surgery_cost - total_insurance_cost - out_of_pocket_cost)
  sorry

end tom_savings_by_having_insurance_l2308_230829


namespace probability_10_products_expected_value_of_products_l2308_230800

open ProbabilityTheory

/-- Probability calculations for worker assessment. -/
noncomputable def worker_assessment_probability (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p^9 * (10 - 9*p)

/-- Expected value of total products produced and debugged by Worker A -/
noncomputable def expected_products (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  20 - 10*p - 10*p^9 + 10*p^10

/-- Theorem 1: Prove that the probability that Worker A ends the assessment by producing only 10 products is p^9(10 - 9p). -/
theorem probability_10_products (p : ℝ) (h : 0 < p ∧ p < 1) :
  worker_assessment_probability p h = p^9 * (10 - 9*p) := by
  sorry

/-- Theorem 2: Prove the expected value E(X) of the total number of products produced and debugged by Worker A is 20 - 10p - 10p^9 + 10p^{10}. -/
theorem expected_value_of_products (p : ℝ) (h : 0 < p ∧ p < 1) :
  expected_products p h = 20 - 10*p - 10*p^9 + 10*p^10 := by
  sorry

end probability_10_products_expected_value_of_products_l2308_230800


namespace number_of_rhombuses_l2308_230825

-- Definition: A grid with 25 small equilateral triangles arranged in a larger triangular pattern
def equilateral_grid (n : ℕ) : Prop :=
  n = 25

-- Theorem: Proving the number of rhombuses that can be formed from the grid
theorem number_of_rhombuses (n : ℕ) (h : equilateral_grid n) : ℕ :=
  30 

-- Main proof statement
example (n : ℕ) (h : equilateral_grid n) : number_of_rhombuses n h = 30 :=
by
  sorry

end number_of_rhombuses_l2308_230825


namespace inequality_holds_for_all_x_l2308_230808

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (x^2 + m * x - 1) / (2 * x^2 - 2 * x + 3) < 1) ↔ -6 < m ∧ m < 2 := 
sorry -- Proof to be provided

end inequality_holds_for_all_x_l2308_230808


namespace distribute_books_l2308_230863

-- Definition of books and people
def num_books : Nat := 2
def num_people : Nat := 10

-- The main theorem statement that we need to prove.
theorem distribute_books : (num_people ^ num_books) = 100 :=
by
  -- Proof body
  sorry

end distribute_books_l2308_230863


namespace min_value_of_expression_l2308_230860

theorem min_value_of_expression 
  (a b : ℝ) 
  (h : a > 0) 
  (h₀ : b > 0) 
  (h₁ : 2*a + b = 2) : 
  ∃ c : ℝ, c = (8*a + b) / (a*b) ∧ c = 9 :=
sorry

end min_value_of_expression_l2308_230860


namespace number_of_squares_centered_at_60_45_l2308_230841

noncomputable def number_of_squares_centered_at (cx : ℕ) (cy : ℕ) : ℕ :=
  let aligned_with_axes := 45
  let not_aligned_with_axes := 2025
  aligned_with_axes + not_aligned_with_axes

theorem number_of_squares_centered_at_60_45 : number_of_squares_centered_at 60 45 = 2070 := 
  sorry

end number_of_squares_centered_at_60_45_l2308_230841


namespace combination_8_5_l2308_230834

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l2308_230834


namespace gcf_lcm_360_210_l2308_230847

theorem gcf_lcm_360_210 :
  let factorization_360 : ℕ × ℕ × ℕ × ℕ := (3, 2, 1, 0) -- Prime exponents for 2, 3, 5, 7
  let factorization_210 : ℕ × ℕ × ℕ × ℕ := (1, 1, 1, 1) -- Prime exponents for 2, 3, 5, 7
  gcd (2^3 * 3^2 * 5 : ℕ) (2 * 3 * 5 * 7 : ℕ) = 30 ∧
  lcm (2^3 * 3^2 * 5 : ℕ) (2 * 3 * 5 * 7 : ℕ) = 2520 :=
by {
  let factorization_360 := (3, 2, 1, 0)
  let factorization_210 := (1, 1, 1, 1)
  sorry
}

end gcf_lcm_360_210_l2308_230847


namespace jessica_current_age_l2308_230819

-- Define the conditions
def jessicaOlderThanClaire (jessica claire : ℕ) : Prop :=
  jessica = claire + 6

def claireAgeInTwoYears (claire : ℕ) : Prop :=
  claire + 2 = 20

-- State the theorem to prove
theorem jessica_current_age : ∃ jessica claire : ℕ, 
  jessicaOlderThanClaire jessica claire ∧ claireAgeInTwoYears claire ∧ jessica = 24 := 
sorry

end jessica_current_age_l2308_230819


namespace total_number_of_guests_l2308_230898

theorem total_number_of_guests (A C S : ℕ) (hA : A = 58) (hC : C = A - 35) (hS : S = 2 * C) : 
  A + C + S = 127 := 
by
  sorry

end total_number_of_guests_l2308_230898


namespace find_q_l2308_230852

theorem find_q (q : ℤ) (h1 : lcm (lcm 12 16) (lcm 18 q) = 144) : q = 1 := sorry

end find_q_l2308_230852


namespace max_min_f_in_rectangle_l2308_230862

def f (x y : ℝ) : ℝ := x^3 + y^3 + 6 * x * y

def in_rectangle (x y : ℝ) : Prop := 
  (-3 ≤ x ∧ x ≤ 1) ∧ (-3 ≤ y ∧ y ≤ 2)

theorem max_min_f_in_rectangle :
  ∃ (x_max y_max x_min y_min : ℝ),
    in_rectangle x_max y_max ∧ in_rectangle x_min y_min ∧
    (∀ x y, in_rectangle x y → f x y ≤ f x_max y_max) ∧
    (∀ x y, in_rectangle x y → f x_min y_min ≤ f x y) ∧
    f x_max y_max = 21 ∧ f x_min y_min = -55 :=
by
  sorry

end max_min_f_in_rectangle_l2308_230862


namespace min_packs_120_cans_l2308_230893

theorem min_packs_120_cans (p8 p16 p32 : ℕ) (total_cans packs_needed : ℕ) :
  total_cans = 120 →
  p8 * 8 + p16 * 16 + p32 * 32 = total_cans →
  packs_needed = p8 + p16 + p32 →
  (∀ (q8 q16 q32 : ℕ), q8 * 8 + q16 * 16 + q32 * 32 = total_cans → q8 + q16 + q32 ≥ packs_needed) →
  packs_needed = 5 :=
by {
  sorry
}

end min_packs_120_cans_l2308_230893


namespace daily_rental_cost_l2308_230813

theorem daily_rental_cost (rental_fee_per_day : ℝ) (mileage_rate : ℝ) (budget : ℝ) (max_miles : ℝ) 
  (h1 : mileage_rate = 0.20) 
  (h2 : budget = 88.0) 
  (h3 : max_miles = 190.0) :
  rental_fee_per_day = 50.0 := 
by
  sorry

end daily_rental_cost_l2308_230813


namespace ball_reaches_height_less_than_2_after_6_bounces_l2308_230840

theorem ball_reaches_height_less_than_2_after_6_bounces :
  ∃ (k : ℕ), 16 * (2/3) ^ k < 2 ∧ ∀ (m : ℕ), m < k → 16 * (2/3) ^ m ≥ 2 :=
by
  sorry

end ball_reaches_height_less_than_2_after_6_bounces_l2308_230840
