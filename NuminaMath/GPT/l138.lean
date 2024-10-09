import Mathlib

namespace greatest_three_digit_multiple_of_17_l138_13828

theorem greatest_three_digit_multiple_of_17 : ∃ (n : ℕ), (n % 17 = 0) ∧ (100 ≤ n ∧ n ≤ 999) ∧ (∀ m, (m % 17 = 0) ∧ (100 ≤ m ∧ m ≤ 999) → m ≤ 986) := 
by sorry

end greatest_three_digit_multiple_of_17_l138_13828


namespace greatest_root_of_g_l138_13877

noncomputable def g (x : ℝ) : ℝ := 10 * x^4 - 16 * x^2 + 6

theorem greatest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → y ≤ x := 
by
  sorry

end greatest_root_of_g_l138_13877


namespace sum_of_squares_2222_l138_13896

theorem sum_of_squares_2222 :
  ∀ (N : ℕ), (∃ (k : ℕ), N = 2 * 10^k - 1) → (∀ (a b : ℤ), N = a^2 + b^2 ↔ N = 2) :=
by sorry

end sum_of_squares_2222_l138_13896


namespace Nils_has_300_geese_l138_13889

variables (A x k n : ℕ)

def condition1 (A x k n : ℕ) : Prop :=
  A = k * x * n

def condition2 (A x k n : ℕ) : Prop :=
  A = (k + 20) * x * (n - 50)

def condition3 (A x k n : ℕ) : Prop :=
  A = (k - 10) * x * (n + 100)

theorem Nils_has_300_geese (A x k n : ℕ) :
  condition1 A x k n →
  condition2 A x k n →
  condition3 A x k n →
  n = 300 :=
by
  intros h1 h2 h3
  sorry

end Nils_has_300_geese_l138_13889


namespace baseball_card_devaluation_l138_13804

variable (x : ℝ) -- Note: x will represent the yearly percent decrease in decimal form (e.g., x = 0.10 for 10%)

theorem baseball_card_devaluation :
  (1 - x) * (1 - x) = 0.81 → x = 0.10 :=
by
  sorry

end baseball_card_devaluation_l138_13804


namespace original_list_length_l138_13870

variable (n m : ℕ)   -- number of integers and the mean respectively
variable (l : List ℤ) -- the original list of integers

def mean (l : List ℤ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Condition 1: Appending 25 increases mean by 3
def condition1 (l : List ℤ) : Prop :=
  mean (25 :: l) = mean l + 3

-- Condition 2: Appending -4 to the enlarged list decreases the mean by 1.5
def condition2 (l : List ℤ) : Prop :=
  mean (-4 :: 25 :: l) = mean (25 :: l) - 1.5

theorem original_list_length (l : List ℤ) (h1 : condition1 l) (h2 : condition2 l) : l.length = 4 := by
  sorry

end original_list_length_l138_13870


namespace value_of_a_l138_13801

theorem value_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 + x + a^2 - 1 = 0 → x = 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end value_of_a_l138_13801


namespace find_a_l138_13816

theorem find_a (x a : ℝ) : 
  (a + 2 = 0) ↔ (a = -2) :=
by
  sorry

end find_a_l138_13816


namespace num_children_proof_l138_13835

-- Definitions and Main Problem
def legs_of_javier : ℕ := 2
def legs_of_wife : ℕ := 2
def legs_per_child : ℕ := 2
def legs_per_dog : ℕ := 4
def legs_of_cat : ℕ := 4
def num_dogs : ℕ := 2
def num_cats : ℕ := 1
def total_legs : ℕ := 22

-- Proof problem: Prove that the number of children (num_children) is equal to 3
theorem num_children_proof : ∃ num_children : ℕ, legs_of_javier + legs_of_wife + (num_children * legs_per_child) + (num_dogs * legs_per_dog) + (num_cats * legs_of_cat) = total_legs ∧ num_children = 3 :=
by
  -- Proof goes here
  sorry

end num_children_proof_l138_13835


namespace final_position_is_east_8km_total_fuel_consumption_is_4_96liters_l138_13808

-- Define the travel distances
def travel_distances : List ℤ := [17, -9, 7, 11, -15, -3]

-- Define the fuel consumption rate
def fuel_consumption_rate : ℝ := 0.08

-- Theorem stating the final position
theorem final_position_is_east_8km :
  List.sum travel_distances = 8 :=
by
  sorry

-- Theorem stating the total fuel consumption
theorem total_fuel_consumption_is_4_96liters :
  (List.sum (travel_distances.map fun x => |x| : List ℝ)) * fuel_consumption_rate = 4.96 :=
by
  sorry

end final_position_is_east_8km_total_fuel_consumption_is_4_96liters_l138_13808


namespace matrix_multiplication_example_l138_13821

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![-4, 5]]
def vector1 : Fin 2 → ℤ := ![4, -2]
def scalar : ℤ := 2
def result : Fin 2 → ℤ := ![32, -52]

theorem matrix_multiplication_example :
  scalar • (matrix1.mulVec vector1) = result := by
  sorry

end matrix_multiplication_example_l138_13821


namespace alex_buys_15_pounds_of_wheat_l138_13855

theorem alex_buys_15_pounds_of_wheat (w o : ℝ) (h1 : w + o = 30) (h2 : 72 * w + 36 * o = 1620) : w = 15 :=
by
  sorry

end alex_buys_15_pounds_of_wheat_l138_13855


namespace find_x_l138_13836

theorem find_x (x : ℝ) (h_pos : x > 0) (h_area : (1 / 2) * x * (3 * x) = 54) : x = 6 :=
by
  sorry

end find_x_l138_13836


namespace copier_cost_l138_13812

noncomputable def total_time : ℝ := 4 + 25 / 60
noncomputable def first_quarter_hour_cost : ℝ := 6
noncomputable def hourly_cost : ℝ := 8
noncomputable def time_after_first_quarter_hour : ℝ := total_time - 0.25
noncomputable def remaining_cost : ℝ := time_after_first_quarter_hour * hourly_cost
noncomputable def total_cost : ℝ := first_quarter_hour_cost + remaining_cost

theorem copier_cost :
  total_cost = 39.33 :=
by
  -- This statement remains to be proved.
  sorry

end copier_cost_l138_13812


namespace max_profit_l138_13871

noncomputable def fixed_cost : ℝ := 2.5
noncomputable def var_cost (x : ℕ) : ℝ :=
  if x < 80 then (x^2 + 10 * x) * 1e4
  else (51 * x - 1450) * 1e4
noncomputable def revenue (x : ℕ) : ℝ := 500 * x * 1e4
noncomputable def profit (x : ℕ) : ℝ := revenue x - var_cost x - fixed_cost * 1e4

theorem max_profit (x : ℕ) :
  (∀ y : ℕ, profit y ≤ 43200 * 1e4) ∧ profit 100 = 43200 * 1e4 := by
  sorry

end max_profit_l138_13871


namespace max_distance_covered_l138_13832

theorem max_distance_covered 
  (D : ℝ)
  (h1 : (D / 2) / 5 + (D / 2) / 4 = 6) : 
  D = 40 / 3 :=
by
  sorry

end max_distance_covered_l138_13832


namespace max_area_basketball_court_l138_13843

theorem max_area_basketball_court : 
  ∃ l w : ℝ, 2 * l + 2 * w = 400 ∧ l ≥ 100 ∧ w ≥ 50 ∧ l * w = 10000 :=
by {
  -- We are skipping the proof for now
  sorry
}

end max_area_basketball_court_l138_13843


namespace solve_rational_equation_solve_quadratic_equation_l138_13810

-- Statement for the first equation
theorem solve_rational_equation (x : ℝ) (h : x ≠ 1) : 
  (x / (x - 1) + 2 / (1 - x) = 2) → (x = 0) :=
by intro h1; sorry

-- Statement for the second equation
theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 + 6 * x - 3 = 0) → (x = 1/2 ∨ x = -3) :=
by intro h1; sorry

end solve_rational_equation_solve_quadratic_equation_l138_13810


namespace find_f1_plus_g1_l138_13876

variables (f g : ℝ → ℝ)

-- Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x
def function_equation (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x - g x = x^3 - 2*x^2 + 1

theorem find_f1_plus_g1 
  (hf : even_function f)
  (hg : odd_function g)
  (hfg : function_equation f g):
  f 1 + g 1 = -2 :=
by {
  sorry
}

end find_f1_plus_g1_l138_13876


namespace walls_divided_equally_l138_13875

-- Define the given conditions
def num_people : ℕ := 5
def num_rooms : ℕ := 9
def rooms_with_4_walls : ℕ := 5
def walls_per_room_4 : ℕ := 4
def rooms_with_5_walls : ℕ := 4
def walls_per_room_5 : ℕ := 5

-- Calculate the total number of walls
def total_walls : ℕ := (rooms_with_4_walls * walls_per_room_4) + (rooms_with_5_walls * walls_per_room_5)

-- Define the expected result
def walls_per_person : ℕ := total_walls / num_people

-- Theorem statement: Each person should paint 8 walls.
theorem walls_divided_equally : walls_per_person = 8 := by
  sorry

end walls_divided_equally_l138_13875


namespace a_and_b_together_finish_in_40_days_l138_13893

theorem a_and_b_together_finish_in_40_days (D : ℕ) 
    (W : ℕ)
    (day_with_b : ℕ)
    (remaining_days_a : ℕ)
    (a_alone_days : ℕ)
    (a_b_together : D = 40)
    (ha : (remaining_days_a = 15) ∧ (a_alone_days = 20) ∧ (day_with_b = 10))
    (work_done_total : 10 * (W / D) + 15 * (W / a_alone_days) = W) :
    D = 40 := 
    sorry

end a_and_b_together_finish_in_40_days_l138_13893


namespace decrement_value_each_observation_l138_13884

theorem decrement_value_each_observation 
  (n : ℕ) 
  (original_mean updated_mean : ℝ) 
  (n_pos : n = 50) 
  (original_mean_value : original_mean = 200)
  (updated_mean_value : updated_mean = 153) :
  (original_mean * n - updated_mean * n) / n = 47 :=
by
  sorry

end decrement_value_each_observation_l138_13884


namespace find_integers_l138_13865

theorem find_integers (n : ℤ) : (n^2 - 13 * n + 36 < 0) ↔ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 :=
by
  sorry

end find_integers_l138_13865


namespace find_number_l138_13888

theorem find_number (x : ℤ) (h : (85 + x) * 1 = 9637) : x = 9552 :=
by
  sorry

end find_number_l138_13888


namespace below_sea_level_is_negative_l138_13840
-- Lean 4 statement


theorem below_sea_level_is_negative 
  (above_sea_pos : ∀ x : ℝ, x > 0 → x = x)
  (below_sea_neg : ∀ x : ℝ, x < 0 → x = x) : 
  (-25 = -25) :=
by 
  -- here we are supposed to provide the proof but we are skipping it with sorry
  sorry

end below_sea_level_is_negative_l138_13840


namespace min_value_of_f_l138_13825

noncomputable def f (x : ℝ) : ℝ :=
  1 / (Real.sqrt (x^2 + 2)) + Real.sqrt (x^2 + 2)

theorem min_value_of_f :
  ∃ x : ℝ, f x = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end min_value_of_f_l138_13825


namespace smallest_number_div_by_225_with_digits_0_1_l138_13894

theorem smallest_number_div_by_225_with_digits_0_1 :
  ∃ n : ℕ, (∀ d ∈ n.digits 10, d = 0 ∨ d = 1) ∧ 225 ∣ n ∧ (∀ m : ℕ, (∀ d ∈ m.digits 10, d = 0 ∨ d = 1) ∧ 225 ∣ m → n ≤ m) ∧ n = 11111111100 :=
sorry

end smallest_number_div_by_225_with_digits_0_1_l138_13894


namespace line_equation_M_l138_13802

theorem line_equation_M (x y : ℝ) : 
  (∃ c1 m1 : ℝ, m1 = 2 / 3 ∧ c1 = 4 ∧ 
  (∃ m2 c2 : ℝ, m2 = 2 * m1 ∧ c2 = (1 / 2) * c1 ∧ y = m2 * x + c2)) → 
  y = (4 / 3) * x + 2 := 
sorry

end line_equation_M_l138_13802


namespace initial_mean_corrected_l138_13869

theorem initial_mean_corrected (M : ℝ) (H : 30 * M + 30 = 30 * 151) : M = 150 :=
sorry

end initial_mean_corrected_l138_13869


namespace positive_difference_l138_13880

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 14) : y - x = 9.714 :=
sorry

end positive_difference_l138_13880


namespace min_employees_to_hire_l138_13868

-- Definitions of the given conditions
def employees_cust_service : ℕ := 95
def employees_tech_support : ℕ := 80
def employees_both : ℕ := 30

-- The theorem stating the minimum number of new employees to hire
theorem min_employees_to_hire (n : ℕ) :
  n = (employees_cust_service - employees_both) 
    + (employees_tech_support - employees_both) 
    + employees_both → 
  n = 145 := sorry

end min_employees_to_hire_l138_13868


namespace polynomial_coeff_sum_l138_13883

theorem polynomial_coeff_sum :
  let f : ℕ → ℕ := λ x => (x^2 + 1) * (x - 1)^8
  let a_0 := f 2
  let a_sum := f 3
  a_sum - a_0 = 2555 :=
by
  let f : ℕ → ℕ := λ x => (x^2 + 1) * (x - 1)^8
  let a_0 := f 2
  let a_sum := f 3
  show a_sum - a_0 = 2555
  sorry

end polynomial_coeff_sum_l138_13883


namespace sequence_not_divisible_by_7_l138_13891

theorem sequence_not_divisible_by_7 (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 1200) : ¬ (7 ∣ (9^n + 1)) :=
by
  sorry

end sequence_not_divisible_by_7_l138_13891


namespace intersection_A_B_subset_A_B_l138_13847

-- Definitions for the sets A and B
def set_A (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x ≤ a + 3}
def set_B : Set ℝ := {x | x < -1 ∨ x > 5}

-- First proof problem: Intersection
theorem intersection_A_B (a : ℝ) (ha : a = -2) :
  set_A a ∩ set_B = {x | -5 ≤ x ∧ x < -1} :=
sorry

-- Second proof problem: Subset
theorem subset_A_B (a : ℝ) :
  set_A a ⊆ set_B ↔ (a ≤ -4 ∨ a ≥ 3) :=
sorry

end intersection_A_B_subset_A_B_l138_13847


namespace employee_y_payment_l138_13848

theorem employee_y_payment (X Y : ℝ) (h1 : X + Y = 616) (h2 : X = 1.2 * Y) : Y = 280 :=
by
  sorry

end employee_y_payment_l138_13848


namespace complement_subset_lemma_l138_13866

-- Definitions for sets P and Q
def P : Set ℝ := {x | 0 < x ∧ x < 1}

def Q : Set ℝ := {x | x^2 + x - 2 ≤ 0}

-- Definition for complement of a set
def C_ℝ (A : Set ℝ) : Set ℝ := {x | ¬(x ∈ A)}

-- Prove the required relationship
theorem complement_subset_lemma : C_ℝ Q ⊆ C_ℝ P :=
by
  -- The proof steps will go here
  sorry

end complement_subset_lemma_l138_13866


namespace six_digit_number_reversed_by_9_l138_13859

-- Hypothetical function to reverse digits of a number
def reverseDigits (n : ℕ) : ℕ := sorry

theorem six_digit_number_reversed_by_9 :
  ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n * 9 = reverseDigits n :=
by
  sorry

end six_digit_number_reversed_by_9_l138_13859


namespace parametric_to_standard_equation_l138_13803

theorem parametric_to_standard_equation (x y t : ℝ) 
(h1 : x = 4 * t + 1) 
(h2 : y = -2 * t - 5) : 
x + 2 * y + 9 = 0 :=
by
  sorry

end parametric_to_standard_equation_l138_13803


namespace negation_of_exists_eq_sin_l138_13845

theorem negation_of_exists_eq_sin : ¬ (∃ x : ℝ, x = Real.sin x) ↔ ∀ x : ℝ, x ≠ Real.sin x :=
by
  sorry

end negation_of_exists_eq_sin_l138_13845


namespace polygon_P_properties_l138_13890

-- Definitions of points A, B, and C
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (1, 0.5, 0)
def C : (ℝ × ℝ × ℝ) := (0, 0.5, 1)

-- Condition of cube intersection and plane containing A, B, and C
def is_corner_of_cube (p : ℝ × ℝ × ℝ) : Prop :=
  p = A

def are_midpoints_of_cube_edges (p₁ p₂ : ℝ × ℝ × ℝ) : Prop :=
  (p₁ = B ∧ p₂ = C)

-- Polygon P resulting from the plane containing A, B, and C intersecting the cube
def num_sides_of_polygon (p : ℝ × ℝ × ℝ) : ℕ := 5 -- Given the polygon is a pentagon

-- Area of triangle ABC
noncomputable def area_triangle_ABC : ℝ :=
  (1/2) * (Real.sqrt 1.5)

-- Area of polygon P
noncomputable def area_polygon_P : ℝ :=
  (11/6) * area_triangle_ABC

-- Theorem stating that polygon P has 5 sides and the ratio of its area to the area of triangle ABC is 11/6
theorem polygon_P_properties (A B C : (ℝ × ℝ × ℝ))
  (hA : is_corner_of_cube A) (hB : are_midpoints_of_cube_edges B C) :
  num_sides_of_polygon A = 5 ∧ area_polygon_P / area_triangle_ABC = (11/6) :=
by sorry

end polygon_P_properties_l138_13890


namespace susan_probability_exactly_three_blue_marbles_l138_13837

open ProbabilityTheory

noncomputable def probability_blue_marbles (n_blue n_red : ℕ) (total_trials drawn_blue : ℕ) : ℚ :=
  let total_marbles := n_blue + n_red
  let prob_blue := (n_blue : ℚ) / total_marbles
  let prob_red := (n_red : ℚ) / total_marbles
  let n_comb := Nat.choose total_trials drawn_blue
  (n_comb : ℚ) * (prob_blue ^ drawn_blue) * (prob_red ^ (total_trials - drawn_blue))

theorem susan_probability_exactly_three_blue_marbles :
  probability_blue_marbles 8 7 7 3 = 35 * (1225621 / 171140625) :=
by
  sorry

end susan_probability_exactly_three_blue_marbles_l138_13837


namespace uma_income_is_20000_l138_13807

/-- Given that the ratio of the incomes of Uma and Bala is 4 : 3, 
the ratio of their expenditures is 3 : 2, and both save $5000 at the end of the year, 
prove that Uma's income is $20000. -/
def uma_bala_income : Prop :=
  ∃ (x y : ℕ), (4 * x - 3 * y = 5000) ∧ (3 * x - 2 * y = 5000) ∧ (4 * x = 20000)
  
theorem uma_income_is_20000 : uma_bala_income :=
  sorry

end uma_income_is_20000_l138_13807


namespace boat_upstream_speed_l138_13841

variable (Vb Vc : ℕ)

def boat_speed_upstream (Vb Vc : ℕ) : ℕ := Vb - Vc

theorem boat_upstream_speed (hVb : Vb = 50) (hVc : Vc = 20) : boat_speed_upstream Vb Vc = 30 :=
by sorry

end boat_upstream_speed_l138_13841


namespace shorter_steiner_network_l138_13873

-- Define the variables and inequality
noncomputable def side_length (a : ℝ) : ℝ := a
noncomputable def diagonal_network_length (a : ℝ) : ℝ := 2 * a * Real.sqrt 2
noncomputable def steiner_network_length (a : ℝ) : ℝ := a * (1 + Real.sqrt 3)

theorem shorter_steiner_network {a : ℝ} (h₀ : 0 < a) :
  diagonal_network_length a > steiner_network_length a :=
by
  -- Proof to be provided (skipping it with sorry)
  sorry

end shorter_steiner_network_l138_13873


namespace GCF_LCM_proof_l138_13851

-- Define GCF (greatest common factor)
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM (least common multiple)
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCF_LCM_proof :
  GCF (LCM 9 21) (LCM 14 15) = 21 :=
by
  sorry

end GCF_LCM_proof_l138_13851


namespace batsman_average_after_17_l138_13811

variable (x : ℝ)
variable (total_runs_16 : ℝ := 16 * x)
variable (runs_17 : ℝ := 90)
variable (new_total_runs : ℝ := total_runs_16 + runs_17)
variable (new_average : ℝ := new_total_runs / 17)

theorem batsman_average_after_17 :
  (total_runs_16 + runs_17 = 17 * (x + 3)) → new_average = x + 3 → new_average = 42 :=
by
  intros h1 h2
  sorry

end batsman_average_after_17_l138_13811


namespace segment_length_eq_ten_l138_13899

theorem segment_length_eq_ten (x : ℝ) (h : |x - 3| = 5) : |8 - (-2)| = 10 :=
by {
  sorry
}

end segment_length_eq_ten_l138_13899


namespace factorize_x2_plus_2x_l138_13853

theorem factorize_x2_plus_2x (x : ℝ) : x^2 + 2*x = x * (x + 2) :=
by sorry

end factorize_x2_plus_2x_l138_13853


namespace determine_flower_responsibility_l138_13857

-- Define the structure of the grid
structure Grid (m n : ℕ) :=
  (vertices : Fin m → Fin n → Bool) -- True if gardener lives at the vertex

-- Define a function to determine if 3 gardeners are nearest to a flower
def is_nearest (i j fi fj : ℕ) : Bool :=
  -- Assume this function gives true if the gardener at (i, j) is one of the 3 nearest to the flower at (fi, fj)
  sorry

-- The main theorem statement
theorem determine_flower_responsibility 
  {m n : ℕ} 
  (G : Grid m n) 
  (i j : Fin m) 
  (k : Fin n) 
  (h : G.vertices i k = true) 
  : ∃ (fi fj : ℕ), is_nearest (i : ℕ) (k : ℕ) fi fj = true := 
sorry

end determine_flower_responsibility_l138_13857


namespace positive_integer_solutions_l138_13813

theorem positive_integer_solutions (a b : ℕ) (h_pos_ab : 0 < a ∧ 0 < b) :
  (∃ k : ℕ, k = a^2 / (2 * a * b^2 - b^3 + 1) ∧ 0 < k) ↔
  ∃ n : ℕ, (a = 2 * n ∧ b = 1) ∨ (a = n ∧ b = 2 * n) ∨ (a = 8 * n^4 - n ∧ b = 2 * n) :=
by
  sorry

end positive_integer_solutions_l138_13813


namespace arnold_and_danny_age_l138_13858

theorem arnold_and_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 9) : x = 4 :=
sorry

end arnold_and_danny_age_l138_13858


namespace roger_final_money_is_correct_l138_13861

noncomputable def initial_money : ℝ := 84
noncomputable def birthday_money : ℝ := 56
noncomputable def found_money : ℝ := 20
noncomputable def spent_on_game : ℝ := 35
noncomputable def spent_percentage : ℝ := 0.15

noncomputable def final_money 
  (initial_money birthday_money found_money spent_on_game spent_percentage : ℝ) : ℝ :=
  let total_before_spending := initial_money + birthday_money + found_money
  let remaining_after_game := total_before_spending - spent_on_game
  let spent_on_gift := spent_percentage * remaining_after_game
  remaining_after_game - spent_on_gift

theorem roger_final_money_is_correct :
  final_money initial_money birthday_money found_money spent_on_game spent_percentage = 106.25 :=
by
  sorry

end roger_final_money_is_correct_l138_13861


namespace solve_equation_l138_13842

theorem solve_equation:
  ∀ x : ℝ, (x + 1) / 3 - 1 = (5 * x - 1) / 6 → x = -1 :=
by
  intro x
  intro h
  sorry

end solve_equation_l138_13842


namespace find_principal_amount_l138_13872

theorem find_principal_amount 
  (P₁ : ℝ) (r₁ t₁ : ℝ) (S₁ : ℝ)
  (P₂ : ℝ) (r₂ t₂ : ℝ) (C₂ : ℝ) :
  S₁ = (P₁ * r₁ * t₁) / 100 →
  C₂ = P₂ * ( (1 + r₂) ^ t₂ - 1) →
  S₁ = C₂ / 2 →
  P₁ = 2800 :=
by
  sorry

end find_principal_amount_l138_13872


namespace ruth_train_track_length_l138_13878

theorem ruth_train_track_length (n : ℕ) (R : ℕ)
  (h_sean : 72 = 8 * n)
  (h_ruth : 72 = R * n) : 
  R = 8 :=
by
  sorry

end ruth_train_track_length_l138_13878


namespace axel_vowels_written_l138_13805

theorem axel_vowels_written (total_alphabets number_of_vowels n : ℕ) (h1 : total_alphabets = 10) (h2 : number_of_vowels = 5) (h3 : total_alphabets = number_of_vowels * n) : n = 2 :=
by
  sorry

end axel_vowels_written_l138_13805


namespace factorize_expression_l138_13864

variable {a b x y : ℝ}

theorem factorize_expression :
  (x^2 - y^2) * a^2 - (x^2 - y^2) * b^2 = (x + y) * (x - y) * (a + b) * (a - b) :=
by
  sorry

end factorize_expression_l138_13864


namespace simplify_expression_l138_13856

theorem simplify_expression (a : ℝ) (h : a > 0) : 
  (a^2 / (a * (a^3) ^ (1 / 2)) ^ (1 / 3)) = a^(7 / 6) :=
sorry

end simplify_expression_l138_13856


namespace mask_donation_equation_l138_13819

theorem mask_donation_equation (x : ℝ) : 
  1 + (1 + x) + (1 + x)^2 = 4.75 :=
sorry

end mask_donation_equation_l138_13819


namespace no_fermat_in_sequence_l138_13815

def sequence_term (n k : ℕ) : ℕ :=
  (k - 2) * n * (n - 1) / 2 + n

def is_fermat_number (a : ℕ) : Prop :=
  ∃ m : ℕ, a = 2^(2^m) + 1

theorem no_fermat_in_sequence (k n : ℕ) (hk : k > 2) (hn : n > 2) :
  ¬ is_fermat_number (sequence_term n k) :=
sorry

end no_fermat_in_sequence_l138_13815


namespace put_letters_in_mailboxes_l138_13831

theorem put_letters_in_mailboxes :
  (3:ℕ)^4 = 81 :=
by
  sorry

end put_letters_in_mailboxes_l138_13831


namespace cookie_portion_l138_13879

theorem cookie_portion :
  ∃ (total_cookies : ℕ) (remaining_cookies : ℕ) (cookies_senior_ate : ℕ) (cookies_senior_took_second_day : ℕ) 
    (cookies_senior_put_back : ℕ) (cookies_junior_took : ℕ),
  total_cookies = 22 ∧
  remaining_cookies = 11 ∧
  cookies_senior_ate = 3 ∧
  cookies_senior_took_second_day = 3 ∧
  cookies_senior_put_back = 2 ∧
  cookies_junior_took = 7 ∧
  4 / 22 = 2 / 11 :=
by
  existsi 22, 11, 3, 3, 2, 7
  sorry

end cookie_portion_l138_13879


namespace spade_to_heart_l138_13867

-- Definition for spade and heart can be abstract geometric shapes
structure Spade := (arcs_top: ℕ) (stem_bottom: ℕ)
structure Heart := (arcs_top: ℕ) (pointed_bottom: ℕ)

-- Condition: the spade symbol must be cut into three parts
def cut_spade (s: Spade) : List (ℕ × ℕ) :=
  [(s.arcs_top, 0), (0, s.stem_bottom), (0, s.stem_bottom)]

-- Define a function to verify if the rearranged parts form a heart
def can_form_heart (pieces: List (ℕ × ℕ)) : Prop :=
  pieces = [(1, 0), (0, 1), (0, 1)]

-- The theorem that the spade parts can form a heart
theorem spade_to_heart (s: Spade) (h: Heart):
  (cut_spade s) = [(s.arcs_top, 0), (0, s.stem_bottom), (0, s.stem_bottom)] →
  can_form_heart [(s.arcs_top, 0), (s.stem_bottom, 0), (s.stem_bottom, 0)] := 
by
  sorry


end spade_to_heart_l138_13867


namespace find_deducted_salary_l138_13874

noncomputable def dailyWage (weeklySalary : ℝ) (workingDays : ℕ) : ℝ := weeklySalary / workingDays

noncomputable def totalDeduction (dailyWage : ℝ) (absentDays : ℕ) : ℝ := dailyWage * absentDays

noncomputable def deductedSalary (weeklySalary : ℝ) (totalDeduction : ℝ) : ℝ := weeklySalary - totalDeduction

theorem find_deducted_salary
  (weeklySalary : ℝ := 791)
  (workingDays : ℕ := 5)
  (absentDays : ℕ := 4)
  (dW := dailyWage weeklySalary workingDays)
  (tD := totalDeduction dW absentDays)
  (dS := deductedSalary weeklySalary tD) :
  dS = 158.20 := 
  by
    sorry

end find_deducted_salary_l138_13874


namespace count_real_numbers_a_with_integer_roots_l138_13882

theorem count_real_numbers_a_with_integer_roots :
  ∃ (S : Finset ℝ), (∀ (a : ℝ), (∃ (x y : ℤ), x^2 + a*x + 9*a = 0 ∧ y^2 + a*y + 9*a = 0) ↔ a ∈ S) ∧ S.card = 8 :=
by
  sorry

end count_real_numbers_a_with_integer_roots_l138_13882


namespace negative_number_reciprocal_eq_self_l138_13860

theorem negative_number_reciprocal_eq_self (x : ℝ) (hx : x < 0) (h : 1 / x = x) : x = -1 :=
by
  sorry

end negative_number_reciprocal_eq_self_l138_13860


namespace curve_line_and_circle_l138_13824

theorem curve_line_and_circle : 
  ∀ x y : ℝ, (x^3 + x * y^2 = 2 * x) ↔ (x = 0 ∨ x^2 + y^2 = 2) :=
by
  sorry

end curve_line_and_circle_l138_13824


namespace slices_served_yesterday_l138_13895

theorem slices_served_yesterday
  (lunch_slices : ℕ)
  (dinner_slices : ℕ)
  (total_slices_today : ℕ)
  (h1 : lunch_slices = 7)
  (h2 : dinner_slices = 5)
  (h3 : total_slices_today = 12) :
  (total_slices_today - (lunch_slices + dinner_slices) = 0) :=
by {
  sorry
}

end slices_served_yesterday_l138_13895


namespace range_of_a_l138_13854

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ (a < -2 ∨ a > 2) :=
by
  sorry

end range_of_a_l138_13854


namespace acceptable_arrangements_correct_l138_13852

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n + 1) * factorial n

-- Define the total number of people
def total_people := 8

-- Calculate the total arrangements of 8 people
def total_arrangements := factorial total_people

-- Calculate the arrangements where Alice and Bob are together
def reduced_people := total_people - 1
def alice_bob_arrangements := factorial reduced_people * factorial 2

-- Calculate the acceptable arrangements where Alice and Bob are not together
def acceptable_arrangements := total_arrangements - alice_bob_arrangements

-- The theorem statement, asserting the correct answer
theorem acceptable_arrangements_correct : acceptable_arrangements = 30240 :=
by
  sorry

end acceptable_arrangements_correct_l138_13852


namespace solve_for_x_l138_13814

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : (9*x)^18 = (27*x)^9 ↔ x = 1/3 :=
by sorry

end solve_for_x_l138_13814


namespace integer_pairs_solution_l138_13820

def is_satisfied_solution (x y : ℤ) : Prop :=
  x^2 + y^2 = x + y + 2

theorem integer_pairs_solution :
  ∀ (x y : ℤ), is_satisfied_solution x y ↔ (x, y) = (-1, 0) ∨ (x, y) = (-1, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (0, 2) ∨ (x, y) = (1, -1) ∨ (x, y) = (1, 2) ∨ (x, y) = (2, 0) ∨ (x, y) = (2, 1) :=
by
  sorry

end integer_pairs_solution_l138_13820


namespace inequality_proof_l138_13829

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (hSum : a + b + c = 1)

theorem inequality_proof :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 1 / 2 :=
by
  sorry

end inequality_proof_l138_13829


namespace ellipse_foci_coordinates_l138_13838

theorem ellipse_foci_coordinates :
  ∃ x y : Real, (3 * x^2 + 4 * y^2 = 12) ∧ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l138_13838


namespace max_integer_a_real_roots_l138_13885

theorem max_integer_a_real_roots :
  ∀ (a : ℤ), (∃ (x : ℝ), (a + 1 : ℝ) * x^2 - 2 * x + 3 = 0) → a ≤ -2 :=
by
  sorry

end max_integer_a_real_roots_l138_13885


namespace train_speed_proof_l138_13806

noncomputable def speed_of_train (train_length : ℝ) (time_seconds : ℝ) (man_speed : ℝ) : ℝ :=
  let train_length_km := train_length / 1000
  let time_hours := time_seconds / 3600
  let relative_speed := train_length_km / time_hours
  relative_speed - man_speed

theorem train_speed_proof :
  speed_of_train 605 32.99736021118311 6 = 60.028 :=
by
  unfold speed_of_train
  -- Direct substitution and expected numerical simplification
  norm_num
  sorry

end train_speed_proof_l138_13806


namespace lines_perpendicular_and_intersect_l138_13850

variable {a b : ℝ}

theorem lines_perpendicular_and_intersect 
  (h_ab_nonzero : a * b ≠ 0)
  (h_orthogonal : a + b = 0) : 
  ∃ p, p ≠ 0 ∧ 
    (∀ x y, x = -y * b^2 → y = 0 → p = (x, y)) ∧ 
    (∀ x y, y = x / a^2 → x = 0 → p = (x, y)) ∧ 
    (∀ x y, x = -y * b^2 ∧ y = x / a^2 → x = 0 ∧ y = 0) := 
sorry

end lines_perpendicular_and_intersect_l138_13850


namespace trajectory_midpoint_chord_l138_13809

theorem trajectory_midpoint_chord (x y : ℝ) 
  (h₀ : y^2 = 4 * x) : (y^2 = 2 * x - 2) :=
sorry

end trajectory_midpoint_chord_l138_13809


namespace second_shirt_price_l138_13830

-- Define the conditions
def price_first_shirt := 82
def price_third_shirt := 90
def min_avg_price_remaining_shirts := 104
def total_shirts := 10
def desired_avg_price := 100

-- Prove the price of the second shirt
theorem second_shirt_price : 
  ∀ (P : ℝ), 
  (price_first_shirt + P + price_third_shirt + 7 * min_avg_price_remaining_shirts = total_shirts * desired_avg_price) → 
  P = 100 :=
by
  sorry

end second_shirt_price_l138_13830


namespace simplify_fraction_l138_13881

theorem simplify_fraction (m : ℝ) (hm : m ≠ 0) : (3 * m^3) / (6 * m^2) = m / 2 :=
by
  sorry

end simplify_fraction_l138_13881


namespace cafeteria_pies_l138_13887

theorem cafeteria_pies (total_apples initial_apples_per_pie held_out_apples : ℕ) (h : total_apples = 150) (g : held_out_apples = 24) (p : initial_apples_per_pie = 15) :
  ((total_apples - held_out_apples) / initial_apples_per_pie) = 8 :=
by
  -- problem-specific proof steps would go here
  sorry

end cafeteria_pies_l138_13887


namespace frog_jump_plan_l138_13800

-- Define the vertices of the hexagon
inductive Vertex
| A | B | C | D | E | F

open Vertex

-- Define adjacency in the regular hexagon
def adjacent (v1 v2 : Vertex) : Prop :=
  match v1, v2 with
  | A, B | A, F | B, C | B, A | C, D | C, B | D, E | D, C | E, F | E, D | F, A | F, E => true
  | _, _ => false

-- Define the problem
def frog_jump_sequences_count : ℕ :=
  26

theorem frog_jump_plan : frog_jump_sequences_count = 26 := 
  sorry

end frog_jump_plan_l138_13800


namespace Mary_is_2_l138_13834

variable (M J : ℕ)

/-- Given the conditions from the problem, Mary's age can be determined to be 2. -/
theorem Mary_is_2 (h1 : J - 5 = M + 2) (h2 : J = 2 * M + 5) : M = 2 := by
  sorry

end Mary_is_2_l138_13834


namespace find_a_tangent_line_eq_l138_13827

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x - 1) * Real.exp x

theorem find_a (a : ℝ) : f 1 (-3) = 0 → a = 1 := by
  sorry

theorem tangent_line_eq (x : ℝ) (e : ℝ) : x = 1 ∧ f 1 x = Real.exp 1 → 
    (4 * Real.exp 1 * x - y - 3 * Real.exp 1 = 0) := by
  sorry

end find_a_tangent_line_eq_l138_13827


namespace largest_equal_cost_l138_13818

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_digit_sum (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

theorem largest_equal_cost :
  ∃ (n : ℕ), n < 500 ∧ digit_sum n = binary_digit_sum n ∧ ∀ m < 500, digit_sum m = binary_digit_sum m → m ≤ 247 :=
by
  sorry

end largest_equal_cost_l138_13818


namespace almond_butter_servings_l138_13817

def servings_of_almond_butter (tbsp_in_container : ℚ) (tbsp_per_serving : ℚ) : ℚ :=
  tbsp_in_container / tbsp_per_serving

def container_holds : ℚ := 37 + 2/3

def serving_size : ℚ := 3

theorem almond_butter_servings :
  servings_of_almond_butter container_holds serving_size = 12 + 5/9 := 
by
  sorry

end almond_butter_servings_l138_13817


namespace product_of_integers_with_cubes_sum_189_l138_13839

theorem product_of_integers_with_cubes_sum_189 :
  ∃ a b : ℤ, a^3 + b^3 = 189 ∧ a * b = 20 :=
by
  -- The proof is omitted for brevity.
  sorry

end product_of_integers_with_cubes_sum_189_l138_13839


namespace sufficient_but_not_necessary_condition_l138_13886

theorem sufficient_but_not_necessary_condition (x : ℝ) (hx : x > 1) : x > 1 → x^2 > 1 ∧ ∀ y, (y^2 > 1 → ¬(y ≥ 1) → y < -1) := 
by
  sorry

end sufficient_but_not_necessary_condition_l138_13886


namespace terry_spent_total_l138_13846

def total_amount_spent (monday_spent tuesday_spent wednesday_spent : ℕ) : ℕ := 
  monday_spent + tuesday_spent + wednesday_spent

theorem terry_spent_total 
  (monday_spent : ℕ)
  (hmonday : monday_spent = 6)
  (tuesday_spent : ℕ)
  (htuesday : tuesday_spent = 2 * monday_spent)
  (wednesday_spent : ℕ)
  (hwednesday : wednesday_spent = 2 * (monday_spent + tuesday_spent)) :
  total_amount_spent monday_spent tuesday_spent wednesday_spent = 54 :=
by
  sorry

end terry_spent_total_l138_13846


namespace double_acute_angle_l138_13863

theorem double_acute_angle (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
sorry

end double_acute_angle_l138_13863


namespace comp_functions_l138_13844

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 5

theorem comp_functions (x : ℝ) : f (g x) = 6 * x - 7 :=
by
  sorry

end comp_functions_l138_13844


namespace abs_inequality_solution_l138_13823

theorem abs_inequality_solution :
  {x : ℝ | |2 * x + 1| > 3} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 1} :=
by
  sorry

end abs_inequality_solution_l138_13823


namespace remainder_of_5032_div_28_l138_13833

theorem remainder_of_5032_div_28 : 5032 % 28 = 20 :=
by
  sorry

end remainder_of_5032_div_28_l138_13833


namespace sin_2phi_l138_13826

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l138_13826


namespace find_age_l138_13862

-- Define the age variables
variables (P Q : ℕ)

-- Define the conditions
def condition1 : Prop := (P - 3) * 3 = (Q - 3) * 4
def condition2 : Prop := (P + 6) * 6 = (Q + 6) * 7

-- Prove that, given the conditions, P equals 15
theorem find_age (h1 : condition1 P Q) (h2 : condition2 P Q) : P = 15 :=
sorry

end find_age_l138_13862


namespace speed_boat_in_still_water_l138_13849

variable (V_b V_s t : ℝ)

def speed_of_boat := V_b

axiom stream_speed : V_s = 26

axiom time_relation : 2 * (t : ℝ) = 2 * t

axiom distance_relation : (V_b + V_s) * t = (V_b - V_s) * (2 * t)

theorem speed_boat_in_still_water : V_b = 78 :=
by {
  sorry
}

end speed_boat_in_still_water_l138_13849


namespace simplify_expression_l138_13897

theorem simplify_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + b^3 = 3 * (a + b)) :
  (a / b) + (b / a) - (3 / (a * b)) = 1 := 
sorry

end simplify_expression_l138_13897


namespace max_value_of_operation_l138_13892

theorem max_value_of_operation : 
  ∃ (n : ℤ), (10 ≤ n ∧ n ≤ 99) ∧ 4 * (300 - n) = 1160 := by
  sorry

end max_value_of_operation_l138_13892


namespace distance_by_land_l138_13822

theorem distance_by_land (distance_by_sea total_distance distance_by_land : ℕ)
  (h1 : total_distance = 601)
  (h2 : distance_by_sea = 150)
  (h3 : total_distance = distance_by_land + distance_by_sea) : distance_by_land = 451 := by
  sorry

end distance_by_land_l138_13822


namespace angle_sum_equal_l138_13898

theorem angle_sum_equal 
  (AB AC DE DF : ℝ)
  (h_AB_AC : AB = AC)
  (h_DE_DF : DE = DF)
  (angle_BAC angle_EDF : ℝ)
  (h_angle_BAC : angle_BAC = 40)
  (h_angle_EDF : angle_EDF = 50)
  (angle_DAC angle_ADE : ℝ)
  (h_angle_DAC : angle_DAC = 70)
  (h_angle_ADE : angle_ADE = 65) :
  angle_DAC + angle_ADE = 135 := 
sorry

end angle_sum_equal_l138_13898
