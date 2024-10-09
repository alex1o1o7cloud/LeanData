import Mathlib

namespace problem_statement_l1312_131208

theorem problem_statement : 
  (∀ (base : ℤ) (exp : ℕ), (-3) = base ∧ 2 = exp → (base ^ exp ≠ -9)) :=
by
  sorry

end problem_statement_l1312_131208


namespace average_problem_l1312_131291

noncomputable def avg2 (a b : ℚ) := (a + b) / 2
noncomputable def avg3 (a b c : ℚ) := (a + b + c) / 3

theorem average_problem :
  avg3 (avg3 2 2 1) (avg2 1 2) 1 = 25 / 18 :=
by
  sorry

end average_problem_l1312_131291


namespace find_x_l1312_131249

theorem find_x 
  (b : ℤ) (h_b : b = 0) 
  (a z y x w : ℤ)
  (h1 : z + a = 1)
  (h2 : y + z + a = 0)
  (h3 : x + y + z = a)
  (h4 : w + x + y = z)
  :
  x = 2 :=
by {
    sorry
}    

end find_x_l1312_131249


namespace seven_b_equals_ten_l1312_131286

theorem seven_b_equals_ten (a b : ℚ) (h1 : 5 * a + 2 * b = 0) (h2 : a = b - 2) : 7 * b = 10 := 
sorry

end seven_b_equals_ten_l1312_131286


namespace find_function_satisfying_property_l1312_131210

noncomputable def example_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + y^2) = f (x^2 - y^2) + f (2 * x * y)

theorem find_function_satisfying_property (f : ℝ → ℝ) (h : ∀ x, 0 ≤ f x) (hf : example_function f) :
  ∃ a : ℝ, 0 ≤ a ∧ ∀ x : ℝ, f x = a * x^2 :=
sorry

end find_function_satisfying_property_l1312_131210


namespace quadratic_roots_x_no_real_solution_y_l1312_131268

theorem quadratic_roots_x (x : ℝ) : 
  x^2 - 4*x + 3 = 0 ↔ (x = 3 ∨ x = 1) := sorry

theorem no_real_solution_y (y : ℝ) : 
  ¬∃ y : ℝ, 4*y^2 - 3*y + 2 = 0 := sorry

end quadratic_roots_x_no_real_solution_y_l1312_131268


namespace painted_surface_area_is_33_l1312_131262

/-- 
Problem conditions:
    1. We have 14 unit cubes each with side length 1 meter.
    2. The cubes are arranged in a rectangular formation with dimensions 3x3x1.
The question:
    Prove that the total painted surface area is 33 square meters.
-/
def total_painted_surface_area (cubes : ℕ) (dim_x dim_y dim_z : ℕ) : ℕ :=
  let top_area := dim_x * dim_y
  let side_area := 2 * (dim_x * dim_z + dim_y * dim_z + (dim_z - 1) * dim_x)
  top_area + side_area

theorem painted_surface_area_is_33 :
  total_painted_surface_area 14 3 3 1 = 33 :=
by
  -- Proof would go here
  sorry

end painted_surface_area_is_33_l1312_131262


namespace ounces_per_container_l1312_131223

def weight_pounds : ℝ := 3.75
def num_containers : ℕ := 4
def pound_to_ounces : ℕ := 16

theorem ounces_per_container :
  (weight_pounds * pound_to_ounces) / num_containers = 15 :=
by
  sorry

end ounces_per_container_l1312_131223


namespace remainder_of_f_100_div_100_l1312_131233

def pascal_triangle_row_sum (n : ℕ) : ℕ :=
  2^n - 2

theorem remainder_of_f_100_div_100 : 
  (pascal_triangle_row_sum 100) % 100 = 74 :=
by
  sorry

end remainder_of_f_100_div_100_l1312_131233


namespace energy_loss_per_bounce_l1312_131271

theorem energy_loss_per_bounce
  (h : ℝ) (t : ℝ) (g : ℝ) (y : ℝ)
  (h_conds : h = 0.2)
  (t_conds : t = 18)
  (g_conds : g = 10)
  (model : t = Real.sqrt (2 * h / g) + 2 * (Real.sqrt (2 * h * y / g)) / (1 - Real.sqrt y)) :
  1 - y = 0.36 :=
by
  sorry

end energy_loss_per_bounce_l1312_131271


namespace smallest_number_of_marbles_l1312_131251

theorem smallest_number_of_marbles (M : ℕ) (h1 : M ≡ 2 [MOD 5]) (h2 : M ≡ 2 [MOD 6]) (h3 : M ≡ 2 [MOD 7]) (h4 : 1 < M) : M = 212 :=
by sorry

end smallest_number_of_marbles_l1312_131251


namespace price_of_each_cupcake_l1312_131264

variable (x : ℝ)

theorem price_of_each_cupcake (h : 50 * x + 40 * 0.5 = 2 * 40 + 20 * 2) : x = 2 := 
by 
  sorry

end price_of_each_cupcake_l1312_131264


namespace max_true_statements_l1312_131257

theorem max_true_statements (x : ℝ) :
  (∀ x, -- given the conditions
    (0 < x^2 ∧ x^2 < 1) →
    (x^2 > 1) →
    (-1 < x ∧ x < 0) →
    (0 < x ∧ x < 1) →
    (0 < x - x^2 ∧ x - x^2 < 1)) →
  -- Prove the maximum number of these statements that can be true is 3
  (∃ (count : ℕ), count = 3) :=
sorry

end max_true_statements_l1312_131257


namespace jasmine_stops_at_S_l1312_131201

-- Definitions of the given conditions
def circumference : ℕ := 60
def total_distance : ℕ := 5400
def quadrants : ℕ := 4
def laps (distance circumference : ℕ) := distance / circumference
def isMultiple (a b : ℕ) := b ∣ a
def onSamePoint (distance circumference : ℕ) := (distance % circumference) = 0

-- The theorem to be proved: Jasmine stops at point S after running the total distance
theorem jasmine_stops_at_S 
  (circumference : ℕ) (total_distance : ℕ) (quadrants : ℕ)
  (h1 : circumference = 60) 
  (h2 : total_distance = 5400)
  (h3 : quadrants = 4)
  (h4 : laps total_distance circumference = 90)
  (h5 : isMultiple total_distance circumference)
  : onSamePoint total_distance circumference := 
  sorry

end jasmine_stops_at_S_l1312_131201


namespace square_of_any_real_number_not_always_greater_than_zero_l1312_131274

theorem square_of_any_real_number_not_always_greater_than_zero (a : ℝ) : 
    (∀ x : ℝ, x^2 ≥ 0) ∧ (exists x : ℝ, x = 0 ∧ x^2 = 0) :=
by {
  sorry
}

end square_of_any_real_number_not_always_greater_than_zero_l1312_131274


namespace percentage_of_students_choose_harvard_l1312_131266

theorem percentage_of_students_choose_harvard
  (total_applicants : ℕ)
  (acceptance_rate : ℝ)
  (students_attend_harvard : ℕ)
  (students_attend_other : ℝ)
  (percentage_attended_harvard : ℝ) :
  total_applicants = 20000 →
  acceptance_rate = 0.05 →
  students_attend_harvard = 900 →
  students_attend_other = 0.10 →
  percentage_attended_harvard = ((students_attend_harvard / (total_applicants * acceptance_rate)) * 100) →
  percentage_attended_harvard = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_of_students_choose_harvard_l1312_131266


namespace point_not_in_region_l1312_131252

theorem point_not_in_region (A B C D : ℝ × ℝ) :
  (A = (0, 0) ∧ 3 * A.1 + 2 * A.2 < 6) ∧
  (B = (1, 1) ∧ 3 * B.1 + 2 * B.2 < 6) ∧
  (C = (0, 2) ∧ 3 * C.1 + 2 * C.2 < 6) ∧
  (D = (2, 0) ∧ ¬ ( 3 * D.1 + 2 * D.2 < 6 )) :=
by {
  sorry
}

end point_not_in_region_l1312_131252


namespace intersection_of_A_and_B_l1312_131235

-- Definitions of sets A and B
def A : Set ℤ := {1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

-- Statement of the theorem
theorem intersection_of_A_and_B : A ∩ B = {1, 3} :=
  sorry

end intersection_of_A_and_B_l1312_131235


namespace final_prices_l1312_131297

noncomputable def hat_initial_price : ℝ := 15
noncomputable def hat_first_discount : ℝ := 0.20
noncomputable def hat_second_discount : ℝ := 0.40

noncomputable def gloves_initial_price : ℝ := 8
noncomputable def gloves_first_discount : ℝ := 0.25
noncomputable def gloves_second_discount : ℝ := 0.30

theorem final_prices :
  let hat_price_after_first_discount := hat_initial_price * (1 - hat_first_discount)
  let hat_final_price := hat_price_after_first_discount * (1 - hat_second_discount)
  let gloves_price_after_first_discount := gloves_initial_price * (1 - gloves_first_discount)
  let gloves_final_price := gloves_price_after_first_discount * (1 - gloves_second_discount)
  hat_final_price = 7.20 ∧ gloves_final_price = 4.20 :=
by
  sorry

end final_prices_l1312_131297


namespace quadratic_real_roots_l1312_131225

theorem quadratic_real_roots (k : ℝ) (h : ∀ x : ℝ, k * x^2 - 4 * x + 1 = 0) : k ≤ 4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_real_roots_l1312_131225


namespace triangle_is_equilateral_l1312_131215

theorem triangle_is_equilateral (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + ac + bc) : a = b ∧ b = c :=
by
  sorry

end triangle_is_equilateral_l1312_131215


namespace tax_free_amount_correct_l1312_131284

-- Definitions based on the problem conditions
def total_value : ℝ := 1720
def tax_paid : ℝ := 78.4
def tax_rate : ℝ := 0.07

-- Definition of the tax-free amount we need to prove
def tax_free_amount : ℝ := 600

-- Main theorem to prove
theorem tax_free_amount_correct : 
  ∃ X : ℝ, 0.07 * (total_value - X) = tax_paid ∧ X = tax_free_amount :=
by 
  use 600
  simp
  sorry

end tax_free_amount_correct_l1312_131284


namespace complement_intersection_U_l1312_131288

-- Definitions of the sets based on the given conditions
def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Definition of the complement of a set with respect to another set
def complement (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Statement asserting the equivalence
theorem complement_intersection_U :
  complement U (M ∩ N) = {1, 4} :=
by
  sorry

end complement_intersection_U_l1312_131288


namespace negation_of_exists_proposition_l1312_131296

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 - 2 * x > 2) ↔ (∀ x : ℝ, x^2 - 2 * x ≤ 2) :=
by
  sorry

end negation_of_exists_proposition_l1312_131296


namespace fraction_zero_when_x_eq_3_l1312_131259

theorem fraction_zero_when_x_eq_3 : ∀ x : ℝ, x = 3 → (x^6 - 54 * x^3 + 729) / (x^3 - 27) = 0 :=
by
  intro x hx
  rw [hx]
  sorry

end fraction_zero_when_x_eq_3_l1312_131259


namespace equation_of_plane_l1312_131299

noncomputable def parametric_form (s t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 2 * s - 3 * t, 4 - s + 2 * t, 1 - 3 * s - t)

theorem equation_of_plane (x y z : ℝ) : 
  (∃ s t : ℝ, parametric_form s t = (x, y, z)) → 5 * x + 11 * y + 7 * z - 61 = 0 :=
by
  sorry

end equation_of_plane_l1312_131299


namespace complex_number_in_fourth_quadrant_l1312_131260

variable {a b : ℝ}

theorem complex_number_in_fourth_quadrant (a b : ℝ): 
  (a^2 + 1 > 0) ∧ (-b^2 - 1 < 0) → 
  ((a^2 + 1, -b^2 - 1).fst > 0 ∧ (a^2 + 1, -b^2 - 1).snd < 0) :=
by
  intro h
  exact h

#check complex_number_in_fourth_quadrant

end complex_number_in_fourth_quadrant_l1312_131260


namespace cuboid_edge_lengths_l1312_131220

theorem cuboid_edge_lengths (
  a b c : ℕ
) (h_volume : a * b * c + a * b + b * c + c * a + a + b + c = 2000) :
  (a = 28 ∧ b = 22 ∧ c = 2) ∨ 
  (a = 28 ∧ b = 2 ∧ c = 22) ∨
  (a = 22 ∧ b = 28 ∧ c = 2) ∨
  (a = 22 ∧ b = 2 ∧ c = 28) ∨
  (a = 2 ∧ b = 28 ∧ c = 22) ∨
  (a = 2 ∧ b = 22 ∧ c = 28) :=
sorry

end cuboid_edge_lengths_l1312_131220


namespace least_number_to_add_l1312_131224

theorem least_number_to_add (a : ℕ) (p q r : ℕ) (h : a = 1076) (hp : p = 41) (hq : q = 59) (hr : r = 67) :
  ∃ k : ℕ, k = 171011 ∧ (a + k) % (lcm p (lcm q r)) = 0 :=
sorry

end least_number_to_add_l1312_131224


namespace find_pqr_abs_l1312_131236

variables {p q r : ℝ}

-- Conditions as hypotheses
def conditions (p q r : ℝ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
  (p^2 + 2/q = q^2 + 2/r) ∧ (q^2 + 2/r = r^2 + 2/p)

-- Statement of the theorem
theorem find_pqr_abs (h : conditions p q r) : |p * q * r| = 2 :=
sorry

end find_pqr_abs_l1312_131236


namespace sequence_bound_l1312_131295

/-- This definition states that given the initial conditions and recurrence relation
for a sequence of positive integers, the 2021st term is greater than 2^2019. -/
theorem sequence_bound (a : ℕ → ℕ) (h_initial : a 2 > a 1)
  (h_recurrence : ∀ n, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 2021 > 2 ^ 2019 :=
sorry

end sequence_bound_l1312_131295


namespace contrapositive_false_of_implication_false_l1312_131269

variable (p q : Prop)

-- The statement we need to prove: If "if p then q" is false, 
-- then "if not q then not p" must be false.
theorem contrapositive_false_of_implication_false (h : ¬ (p → q)) : ¬ (¬ q → ¬ p) :=
by
sorry

end contrapositive_false_of_implication_false_l1312_131269


namespace reverse_digits_difference_l1312_131226

theorem reverse_digits_difference (q r : ℕ) (x y : ℕ) 
  (hq : q = 10 * x + y)
  (hr : r = 10 * y + x)
  (hq_r_pos : q > r)
  (h_diff_lt_20 : q - r < 20)
  (h_max_diff : q - r = 18) :
  x - y = 2 := 
by
  sorry

end reverse_digits_difference_l1312_131226


namespace candy_bar_cost_l1312_131253

-- Definitions of conditions
def soft_drink_cost : ℕ := 4
def num_soft_drinks : ℕ := 2
def num_candy_bars : ℕ := 5
def total_cost : ℕ := 28

-- Proof Statement
theorem candy_bar_cost : (total_cost - num_soft_drinks * soft_drink_cost) / num_candy_bars = 4 := by
  sorry

end candy_bar_cost_l1312_131253


namespace sufficient_and_necessary_cond_l1312_131254

theorem sufficient_and_necessary_cond (x : ℝ) : |x| > 2 ↔ (x > 2) :=
sorry

end sufficient_and_necessary_cond_l1312_131254


namespace sum_of_decimals_as_fraction_l1312_131250

theorem sum_of_decimals_as_fraction :
  let x := (0 : ℝ) + 1 / 3;
  let y := (0 : ℝ) + 2 / 3;
  let z := (0 : ℝ) + 2 / 5;
  x + y + z = 7 / 5 :=
by
  let x := (0 : ℝ) + 1 / 3
  let y := (0 : ℝ) + 2 / 3
  let z := (0 : ℝ) + 2 / 5
  show x + y + z = 7 / 5
  sorry

end sum_of_decimals_as_fraction_l1312_131250


namespace sin_300_eq_neg_sqrt3_div_2_l1312_131213

theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l1312_131213


namespace remainder_base12_div_9_l1312_131289

def base12_to_decimal (n : ℕ) : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12 + 3

theorem remainder_base12_div_9 : (base12_to_decimal 2543) % 9 = 8 := by
  unfold base12_to_decimal
  -- base12_to_decimal 2543 is 4227
  show 4227 % 9 = 8
  sorry

end remainder_base12_div_9_l1312_131289


namespace fraction_problem_l1312_131221

theorem fraction_problem :
  (1 / 4 + 3 / 8) - 1 / 8 = 1 / 2 :=
by
  -- The proof steps are skipped
  sorry

end fraction_problem_l1312_131221


namespace division_result_l1312_131219

theorem division_result (k q : ℕ) (h₁ : k % 81 = 11) (h₂ : 81 > 0) : k / 81 = q + 11 / 81 :=
  sorry

end division_result_l1312_131219


namespace three_solutions_exists_l1312_131255

theorem three_solutions_exists (n : ℕ) (h_pos : 0 < n) (h_sol : ∃ x y : ℤ, x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ x1 y1 x2 y2 x3 y3 : ℤ, (x1^3 - 3 * x1 * y1^2 + y1^3 = n) ∧ (x2^3 - 3 * x2 * y2^2 + y2^3 = n) ∧ (x3^3 - 3 * x3 * y3^2 + y3^3 = n) ∧ (x1, y1) ≠ (x2, y2) ∧ (x2, y2) ≠ (x3, y3) ∧ (x1, y1) ≠ (x3, y3) :=
by
  sorry

end three_solutions_exists_l1312_131255


namespace linda_total_distance_l1312_131209

theorem linda_total_distance :
  ∃ x: ℕ, 
    (x > 0) ∧ (60 % x = 0) ∧
    ((x + 5) > 0) ∧ (60 % (x + 5) = 0) ∧
    ((x + 10) > 0) ∧ (60 % (x + 10) = 0) ∧
    ((x + 15) > 0) ∧ (60 % (x + 15) = 0) ∧
    (60 / x + 60 / (x + 5) + 60 / (x + 10) + 60 / (x + 15) = 25) :=
by
  sorry

end linda_total_distance_l1312_131209


namespace find_abs_of_y_l1312_131287

theorem find_abs_of_y (x y : ℝ) (h1 : x^2 + y^2 = 1) (h2 : 20 * x^3 - 15 * x = 3) : 
  |20 * y^3 - 15 * y| = 4 := 
sorry

end find_abs_of_y_l1312_131287


namespace product_of_consecutive_integers_sqrt_73_l1312_131222

theorem product_of_consecutive_integers_sqrt_73 : 
  ∃ (m n : ℕ), (m < n) ∧ ∃ (j k : ℕ), (j = 8) ∧ (k = 9) ∧ (m = j) ∧ (n = k) ∧ (m * n = 72) := by
  sorry

end product_of_consecutive_integers_sqrt_73_l1312_131222


namespace option_C_sets_same_l1312_131234

-- Define the sets for each option
def option_A_set_M : Set (ℕ × ℕ) := {(3, 2)}
def option_A_set_N : Set (ℕ × ℕ) := {(2, 3)}

def option_B_set_M : Set (ℕ × ℕ) := {p | p.1 + p.2 = 1}
def option_B_set_N : Set ℕ := { y | ∃ x, x + y = 1 }

def option_C_set_M : Set ℕ := {4, 5}
def option_C_set_N : Set ℕ := {5, 4}

def option_D_set_M : Set ℕ := {1, 2}
def option_D_set_N : Set (ℕ × ℕ) := {(1, 2)}

-- Prove that option C sets represent the same set
theorem option_C_sets_same : option_C_set_M = option_C_set_N := by
  sorry

end option_C_sets_same_l1312_131234


namespace down_payment_amount_l1312_131230

-- Define the monthly savings per person
def monthly_savings_per_person : ℤ := 1500

-- Define the number of people
def number_of_people : ℤ := 2

-- Define the total monthly savings
def total_monthly_savings : ℤ := monthly_savings_per_person * number_of_people

-- Define the number of years they will save
def years_saving : ℤ := 3

-- Define the number of months in a year
def months_in_year : ℤ := 12

-- Define the total number of months
def total_months : ℤ := years_saving * months_in_year

-- Define the total savings needed for the down payment
def total_savings_needed : ℤ := total_monthly_savings * total_months

-- Prove that the total amount needed for the down payment is $108,000
theorem down_payment_amount : total_savings_needed = 108000 := by
  -- This part requires a proof, which we skip with sorry
  sorry

end down_payment_amount_l1312_131230


namespace max_xy_max_xy_value_l1312_131202

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 3 * y = 12) : x * y ≤ 3 :=
sorry

theorem max_xy_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + 3 * y = 12) : x * y = 3 → x = 3 / 2 ∧ y = 2 :=
sorry

end max_xy_max_xy_value_l1312_131202


namespace frequency_in_interval_l1312_131292

-- Definitions for the sample size and frequencies in given intervals
def sample_size : ℕ := 20
def freq_10_20 : ℕ := 2
def freq_20_30 : ℕ := 3
def freq_30_40 : ℕ := 4
def freq_40_50 : ℕ := 5

-- The goal: Prove that the frequency of the sample in the interval (10, 50] is 0.7
theorem frequency_in_interval (h₁ : sample_size = 20)
                              (h₂ : freq_10_20 = 2)
                              (h₃ : freq_20_30 = 3)
                              (h₄ : freq_30_40 = 4)
                              (h₅ : freq_40_50 = 5) :
  ((freq_10_20 + freq_20_30 + freq_30_40 + freq_40_50) : ℝ) / sample_size = 0.7 := 
by
  sorry

end frequency_in_interval_l1312_131292


namespace seating_arrangement_l1312_131241

def num_ways_seated (total_passengers : ℕ) (window_seats : ℕ) : ℕ :=
  window_seats * (total_passengers - 1) * (total_passengers - 2) * (total_passengers - 3)

theorem seating_arrangement (passengers_seats taxi_window_seats : ℕ)
  (h1 : passengers_seats = 4) (h2 : taxi_window_seats = 2) :
  num_ways_seated passengers_seats taxi_window_seats = 12 :=
by
  -- proof will go here
  sorry

end seating_arrangement_l1312_131241


namespace ammonium_chloride_reacts_with_potassium_hydroxide_l1312_131290

/-- Prove that 1 mole of ammonium chloride is required to react with 
    1 mole of potassium hydroxide to form 1 mole of ammonia, 
    1 mole of water, and 1 mole of potassium chloride, 
    given the balanced chemical equation:
    NH₄Cl + KOH → NH₃ + H₂O + KCl
-/
theorem ammonium_chloride_reacts_with_potassium_hydroxide :
    ∀ (NH₄Cl KOH NH₃ H₂O KCl : ℕ), 
    (NH₄Cl + KOH = NH₃ + H₂O + KCl) → 
    (NH₄Cl = 1) → 
    (KOH = 1) → 
    (NH₃ = 1) → 
    (H₂O = 1) → 
    (KCl = 1) → 
    NH₄Cl = 1 :=
by
  intros
  sorry

end ammonium_chloride_reacts_with_potassium_hydroxide_l1312_131290


namespace polygon_sides_equation_l1312_131275

theorem polygon_sides_equation (n : ℕ) 
  (h1 : (n-2) * 180 = 4 * 360) : n = 10 := 
by 
  sorry

end polygon_sides_equation_l1312_131275


namespace min_a_plus_b_l1312_131204

theorem min_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + b = 1) : a + b >= 4 :=
sorry

end min_a_plus_b_l1312_131204


namespace quadratic_condition_l1312_131283

noncomputable def quadratic_sufficiency (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + x + m = 0 → m < 1/4

noncomputable def quadratic_necessity (m : ℝ) : Prop :=
  (∃ (x : ℝ), x^2 + x + m = 0) → m ≤ 1/4

theorem quadratic_condition (m : ℝ) : 
  (m < 1/4 → quadratic_sufficiency m) ∧ ¬ quadratic_necessity m := 
sorry

end quadratic_condition_l1312_131283


namespace evaluate_expression_l1312_131293

theorem evaluate_expression : (10^9) / ((2 * 10^6) * 3) = 500 / 3 :=
by sorry

end evaluate_expression_l1312_131293


namespace intersection_in_fourth_quadrant_l1312_131294

theorem intersection_in_fourth_quadrant (k : ℝ) :
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ (-6 < k) ∧ (k < -2) :=
by
  sorry

end intersection_in_fourth_quadrant_l1312_131294


namespace total_fencing_l1312_131238

def playground_side_length : ℕ := 27
def garden_length : ℕ := 12
def garden_width : ℕ := 9

def perimeter_square (side : ℕ) : ℕ := 4 * side
def perimeter_rectangle (length width : ℕ) : ℕ := 2 * length + 2 * width

theorem total_fencing (side playground_side_length : ℕ) (garden_length garden_width : ℕ) :
  perimeter_square playground_side_length + perimeter_rectangle garden_length garden_width = 150 :=
by
  sorry

end total_fencing_l1312_131238


namespace geometric_sum_S_40_l1312_131242

variable (S : ℕ → ℝ)

-- Conditions
axiom sum_S_10 : S 10 = 18
axiom sum_S_20 : S 20 = 24

-- Proof statement
theorem geometric_sum_S_40 : S 40 = 80 / 3 :=
by
  sorry

end geometric_sum_S_40_l1312_131242


namespace difference_between_waiter_and_twenty_less_l1312_131277

-- Definitions for the given conditions
def total_slices : ℕ := 78
def ratio_buzz : ℕ := 5
def ratio_waiter : ℕ := 8
def total_ratio : ℕ := ratio_buzz + ratio_waiter
def slices_per_part : ℕ := total_slices / total_ratio
def buzz_share : ℕ := ratio_buzz * slices_per_part
def waiter_share : ℕ := ratio_waiter * slices_per_part
def twenty_less_waiter : ℕ := waiter_share - 20

-- The proof statement
theorem difference_between_waiter_and_twenty_less : 
  waiter_share - twenty_less_waiter = 20 :=
by sorry

end difference_between_waiter_and_twenty_less_l1312_131277


namespace exists_ab_odd_n_exists_ab_odd_n_gt3_l1312_131212

-- Define the required conditions
def gcd_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define a helper function to identify odd positive integers
def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem exists_ab_odd_n (n : ℕ) (h : is_odd n) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ gcd_coprime (a * b * (a + b)) n :=
sorry

theorem exists_ab_odd_n_gt3 (n : ℕ) (h1 : is_odd n) (h2 : n > 3) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ gcd_coprime (a * b * (a + b)) n ∧ n ∣ (a - b) = false :=
sorry

end exists_ab_odd_n_exists_ab_odd_n_gt3_l1312_131212


namespace x_cubed_inverse_cubed_l1312_131200

theorem x_cubed_inverse_cubed (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 :=
by sorry

end x_cubed_inverse_cubed_l1312_131200


namespace joe_money_left_l1312_131270

theorem joe_money_left
  (initial_money : ℕ) (notebook_cost : ℕ) (notebooks : ℕ)
  (book_cost : ℕ) (books : ℕ) (pen_cost : ℕ) (pens : ℕ)
  (sticker_pack_cost : ℕ) (sticker_packs : ℕ) (charity : ℕ)
  (remaining_money : ℕ) :
  initial_money = 150 →
  notebook_cost = 4 →
  notebooks = 7 →
  book_cost = 12 →
  books = 2 →
  pen_cost = 2 →
  pens = 5 →
  sticker_pack_cost = 6 →
  sticker_packs = 3 →
  charity = 10 →
  remaining_money = 60 →
  remaining_money = 
    initial_money - 
    ((notebooks * notebook_cost) + 
     (books * book_cost) + 
     (pens * pen_cost) + 
     (sticker_packs * sticker_pack_cost) + 
     charity) := 
by
  intros; sorry

end joe_money_left_l1312_131270


namespace nail_pierces_one_cardboard_only_l1312_131240

/--
Seryozha cut out two identical figures from cardboard. He placed them overlapping
at the bottom of a rectangular box. The bottom turned out to be completely covered. 
A nail was driven into the center of the bottom. Prove that it is possible for the 
nail to pierce one cardboard piece without piercing the other.
-/
theorem nail_pierces_one_cardboard_only 
  (identical_cardboards : Prop)
  (overlapping : Prop)
  (fully_covered_bottom : Prop)
  (nail_center : Prop) 
  : ∃ (layout : Prop), layout ∧ nail_center → nail_pierces_one :=
sorry

end nail_pierces_one_cardboard_only_l1312_131240


namespace solve_inequality_l1312_131258

theorem solve_inequality (a x : ℝ) (ha : a ≠ 0) :
  (a > 0 → (x^2 - 5 * a * x + 6 * a^2 > 0 ↔ (x < 2 * a ∨ x > 3 * a))) ∧
  (a < 0 → (x^2 - 5 * a * x + 6 * a^2 > 0 ↔ (x < 3 * a ∨ x > 2 * a))) :=
by
  sorry

end solve_inequality_l1312_131258


namespace add_fractions_l1312_131281

theorem add_fractions: (2 / 5) + (3 / 8) = 31 / 40 := 
by 
  sorry

end add_fractions_l1312_131281


namespace speed_of_second_train_is_16_l1312_131273

def speed_second_train (v : ℝ) : Prop :=
  ∃ t : ℝ, 
    (20 * t = v * t + 70) ∧ -- Condition: the first train traveled 70 km more than the second train
    (20 * t + v * t = 630)  -- Condition: total distance between stations

theorem speed_of_second_train_is_16 : speed_second_train 16 :=
by
  sorry

end speed_of_second_train_is_16_l1312_131273


namespace probability_x_plus_y_lt_3_in_rectangle_l1312_131214

noncomputable def probability_problem : ℚ :=
let rect_area := (4 : ℚ) * 3
let tri_area := (1 / 2 : ℚ) * 3 * 3
tri_area / rect_area

theorem probability_x_plus_y_lt_3_in_rectangle :
  probability_problem = 3 / 8 :=
sorry

end probability_x_plus_y_lt_3_in_rectangle_l1312_131214


namespace Abhay_takes_1_hour_less_than_Sameer_l1312_131279

noncomputable def Sameer_speed := 42 / (6 - 2)
noncomputable def Abhay_time_doubled_speed := 42 / (2 * 7)
noncomputable def Sameer_time := 42 / Sameer_speed

theorem Abhay_takes_1_hour_less_than_Sameer
  (distance : ℝ := 42)
  (Abhay_speed : ℝ := 7)
  (Sameer_speed : ℝ := Sameer_speed)
  (time_Sameer : ℝ := distance / Sameer_speed)
  (time_Abhay_doubled_speed : ℝ := distance / (2 * Abhay_speed)) :
  time_Sameer - time_Abhay_doubled_speed = 1 :=
by
  sorry

end Abhay_takes_1_hour_less_than_Sameer_l1312_131279


namespace adults_in_each_group_l1312_131228

theorem adults_in_each_group (A : ℕ) :
  (∃ n : ℕ, n >= 17 ∧ n * 15 = 255) →
  (∃ m : ℕ, m * A = 255 ∧ m >= 17) →
  A = 15 :=
by
  intros h_child_groups h_adult_groups
  -- Use sorry to skip the proof
  sorry

end adults_in_each_group_l1312_131228


namespace natural_pairs_prime_l1312_131261

theorem natural_pairs_prime (x y : ℕ) (p : ℕ) (hp : Nat.Prime p) (h_eq : p = xy^2 / (x + y))
  : (x, y) = (2, 2) ∨ (x, y) = (6, 2) :=
sorry

end natural_pairs_prime_l1312_131261


namespace value_of_a_l1312_131203

-- Definitions based on conditions
def A (a : ℝ) : Set ℝ := {1, 2, a}
def B : Set ℝ := {1, 7}

-- Theorem statement
theorem value_of_a (a : ℝ) (h : B ⊆ A a) : a = 7 :=
sorry

end value_of_a_l1312_131203


namespace F_3_f_5_eq_24_l1312_131298

def f (a : ℤ) : ℤ := a - 2
def F (a b : ℤ) : ℤ := b^3 - a

theorem F_3_f_5_eq_24 : F 3 (f 5) = 24 := by
  sorry

end F_3_f_5_eq_24_l1312_131298


namespace distance_between_trees_l1312_131217

theorem distance_between_trees
  (yard_length : ℕ)
  (num_trees : ℕ)
  (h_yard_length : yard_length = 441)
  (h_num_trees : num_trees = 22) :
  (yard_length / (num_trees - 1)) = 21 :=
by
  sorry

end distance_between_trees_l1312_131217


namespace positive_difference_solutions_abs_l1312_131272

theorem positive_difference_solutions_abs (x1 x2 : ℝ) 
  (h1 : 2 * x1 - 3 = 18 ∨ 2 * x1 - 3 = -18) 
  (h2 : 2 * x2 - 3 = 18 ∨ 2 * x2 - 3 = -18) : 
  |x1 - x2| = 18 :=
sorry

end positive_difference_solutions_abs_l1312_131272


namespace digit_205_of_14_div_360_l1312_131263

noncomputable def decimal_expansion_of_fraction (n d : ℕ) : ℕ → ℕ := sorry

theorem digit_205_of_14_div_360 : 
  decimal_expansion_of_fraction 14 360 205 = 8 :=
sorry

end digit_205_of_14_div_360_l1312_131263


namespace remaining_payment_l1312_131206
noncomputable def total_cost (deposit : ℝ) (percentage : ℝ) : ℝ :=
  deposit / percentage

noncomputable def remaining_amount (deposit : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost - deposit

theorem remaining_payment (deposit : ℝ) (percentage : ℝ) (total_cost : ℝ) (remaining_amount : ℝ) :
  deposit = 140 → percentage = 0.1 → total_cost = deposit / percentage → remaining_amount = total_cost - deposit → remaining_amount = 1260 :=
by
  intros
  sorry

end remaining_payment_l1312_131206


namespace solution_l1312_131265

noncomputable def problem_statement (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

theorem solution (f : ℝ → ℝ) (h : problem_statement f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b * x^2 := by
  sorry

end solution_l1312_131265


namespace find_f_10_l1312_131267

def f (x : Int) : Int := sorry

axiom condition_1 : f 1 + 1 > 0
axiom condition_2 : ∀ x y : Int, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom condition_3 : ∀ x : Int, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := by
  sorry

end find_f_10_l1312_131267


namespace train_speed_proof_l1312_131211

variables (distance_to_syracuse total_time_hours return_trip_speed average_speed_to_syracuse : ℝ)

def question_statement : Prop :=
  distance_to_syracuse = 120 ∧
  total_time_hours = 5.5 ∧
  return_trip_speed = 38.71 →
  average_speed_to_syracuse = 50

theorem train_speed_proof :
  question_statement distance_to_syracuse total_time_hours return_trip_speed average_speed_to_syracuse :=
by
  -- sorry is used to indicate that the proof is omitted
  sorry

end train_speed_proof_l1312_131211


namespace sequence_an_expression_l1312_131232

theorem sequence_an_expression (a : ℕ → ℕ) : 
  a 1 = 1 ∧ (∀ n : ℕ, n ≥ 1 → (a n / n - a (n - 1) / (n - 1)) = 2) → (∀ n : ℕ, a n = 2 * n * n - n) :=
by
  sorry

end sequence_an_expression_l1312_131232


namespace candy_eaten_l1312_131244

theorem candy_eaten 
  {initial_pieces remaining_pieces eaten_pieces : ℕ} 
  (h₁ : initial_pieces = 12) 
  (h₂ : remaining_pieces = 3) 
  (h₃ : eaten_pieces = initial_pieces - remaining_pieces) 
  : eaten_pieces = 9 := 
by 
  sorry

end candy_eaten_l1312_131244


namespace friends_count_l1312_131243

-- Define the given conditions
def initial_chicken_wings := 2
def additional_chicken_wings := 25
def chicken_wings_per_person := 3

-- Define the total number of chicken wings
def total_chicken_wings := initial_chicken_wings + additional_chicken_wings

-- Define the target number of friends in the group
def number_of_friends := total_chicken_wings / chicken_wings_per_person

-- The theorem stating that the number of friends is 9
theorem friends_count : number_of_friends = 9 := by
  sorry

end friends_count_l1312_131243


namespace smallest_number_of_contestants_solving_all_problems_l1312_131282

theorem smallest_number_of_contestants_solving_all_problems
    (total_contestants : ℕ)
    (solve_first : ℕ)
    (solve_second : ℕ)
    (solve_third : ℕ)
    (solve_fourth : ℕ)
    (H1 : total_contestants = 100)
    (H2 : solve_first = 90)
    (H3 : solve_second = 85)
    (H4 : solve_third = 80)
    (H5 : solve_fourth = 75)
  : ∃ n, n = 30 := by
  sorry

end smallest_number_of_contestants_solving_all_problems_l1312_131282


namespace probability_of_xiao_li_l1312_131239

def total_students : ℕ := 5
def xiao_li : ℕ := 1

noncomputable def probability_xiao_li_chosen : ℚ :=
  (xiao_li : ℚ) / (total_students : ℚ)

theorem probability_of_xiao_li : probability_xiao_li_chosen = 1 / 5 :=
sorry

end probability_of_xiao_li_l1312_131239


namespace minimum_distance_midpoint_l1312_131245

theorem minimum_distance_midpoint 
    (θ : ℝ)
    (P : ℝ × ℝ := (-4, 4))
    (C1_standard : ∀ (x y : ℝ), (x + 4)^2 + (y - 3)^2 = 1)
    (C2_standard : ∀ (x y : ℝ), x^2 / 64 + y^2 / 9 = 1)
    (Q : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ))
    (M : ℝ × ℝ := (-2 + 4 * Real.cos θ, 2 + 3 / 2 * Real.sin θ))
    (C3_standard : ∀ (x y : ℝ), x - 2*y - 7 = 0) :
    ∃ (θ : ℝ), θ = Real.arcsin (-3/5) ∧ (θ = Real.arccos 4/5) ∧
    (∀ (d : ℝ), d = abs (5 * Real.sin (Real.arctan (4 / 3) - θ) - 13) / Real.sqrt 5 ∧ 
    d = 8 * Real.sqrt 5 / 5) :=
sorry

end minimum_distance_midpoint_l1312_131245


namespace determine_value_of_x_l1312_131216

theorem determine_value_of_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 2 = 6 * y) : x = 48 :=
by
  sorry

end determine_value_of_x_l1312_131216


namespace supermarket_selection_expected_value_l1312_131278

noncomputable def small_supermarkets := 72
noncomputable def medium_supermarkets := 24
noncomputable def large_supermarkets := 12
noncomputable def total_supermarkets := small_supermarkets + medium_supermarkets + large_supermarkets
noncomputable def selected_supermarkets := 9

-- Problem (I)
noncomputable def small_selected := (small_supermarkets * selected_supermarkets) / total_supermarkets
noncomputable def medium_selected := (medium_supermarkets * selected_supermarkets) / total_supermarkets
noncomputable def large_selected := (large_supermarkets * selected_supermarkets) / total_supermarkets

theorem supermarket_selection :
  small_selected = 6 ∧ medium_selected = 2 ∧ large_selected = 1 :=
sorry

-- Problem (II)
noncomputable def further_analysis := 3
noncomputable def prob_small := small_selected / selected_supermarkets
noncomputable def E_X := prob_small * further_analysis

theorem expected_value :
  E_X = 2 :=
sorry

end supermarket_selection_expected_value_l1312_131278


namespace each_person_gets_9_wings_l1312_131256

noncomputable def chicken_wings_per_person (initial_wings : ℕ) (additional_wings : ℕ) (friends : ℕ) : ℕ :=
  (initial_wings + additional_wings) / friends

theorem each_person_gets_9_wings :
  chicken_wings_per_person 20 25 5 = 9 :=
by
  sorry

end each_person_gets_9_wings_l1312_131256


namespace general_formula_sequence_l1312_131207

theorem general_formula_sequence (a : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h_rec : ∀ n : ℕ, n > 0 → a (n + 1) = 4 * a n + 3) :
  ∀ n : ℕ, n > 0 → a n = 4^n - 1 :=
by 
  sorry

end general_formula_sequence_l1312_131207


namespace joe_speed_first_part_l1312_131248

theorem joe_speed_first_part (v : ℝ) :
  let d1 := 420 -- distance of the first part in miles
  let d2 := 120 -- distance of the second part in miles
  let v2 := 40  -- speed during the second part in miles per hour
  let d_total := d1 + d2 -- total distance
  let avg_speed := 54 -- average speed in miles per hour
  let t1 := d1 / v -- time for the first part
  let t2 := d2 / v2 -- time for the second part
  let t_total := t1 + t2 -- total time
  (d_total / t_total) = avg_speed -> v = 60 :=
by
  intros
  sorry

end joe_speed_first_part_l1312_131248


namespace parabola_equation_l1312_131237

open Real

theorem parabola_equation (vertex focus : ℝ × ℝ) (h_vertex : vertex = (0, 0)) (h_focus : focus = (0, 3)) :
  ∃ a : ℝ, x^2 = 12 * y := by
  sorry

end parabola_equation_l1312_131237


namespace largest_smallest_difference_l1312_131205

theorem largest_smallest_difference (a b c d : ℚ) (h₁ : a = 2.5) (h₂ : b = 22/13) (h₃ : c = 0.7) (h₄ : d = 32/33) :
  max (max a b) (max c d) - min (min a b) (min c d) = 1.8 := by
  sorry

end largest_smallest_difference_l1312_131205


namespace incorrect_reasoning_form_l1312_131285

-- Define what it means to be a rational number
def is_rational (x : ℚ) : Prop := true

-- Define what it means to be a fraction
def is_fraction (x : ℚ) : Prop := true

-- Define what it means to be an integer
def is_integer (x : ℤ) : Prop := true

-- State the premises as hypotheses
theorem incorrect_reasoning_form (h1 : ∃ x : ℚ, is_rational x ∧ is_fraction x)
                                 (h2 : ∀ z : ℤ, is_rational z) :
  ¬ (∀ z : ℤ, is_fraction z) :=
by
  -- We are stating the conclusion as a hypothesis that needs to be proven incorrect
  sorry

end incorrect_reasoning_form_l1312_131285


namespace total_cost_of_square_park_l1312_131246

-- Define the cost per side and number of sides
def cost_per_side : ℕ := 56
def sides_of_square : ℕ := 4

-- The total cost of fencing the park
def total_cost_of_fencing (cost_per_side : ℕ) (sides_of_square : ℕ) : ℕ := cost_per_side * sides_of_square

-- The statement we need to prove
theorem total_cost_of_square_park : total_cost_of_fencing cost_per_side sides_of_square = 224 :=
by sorry

end total_cost_of_square_park_l1312_131246


namespace distance_travelled_l1312_131218

theorem distance_travelled (speed time distance : ℕ) 
  (h1 : speed = 25)
  (h2 : time = 5)
  (h3 : distance = speed * time) : 
  distance = 125 :=
by
  sorry

end distance_travelled_l1312_131218


namespace cindy_arrival_speed_l1312_131229

def cindy_speed (d t1 t2 t3: ℕ) : Prop :=
  (d = 20 * t1) ∧ 
  (d = 10 * (t2 + 3 / 4)) ∧
  (t3 = t1 + 1 / 2) ∧
  (20 * t1 = 10 * (t2 + 3 / 4)) -> 
  (d / (t3) = 12)

theorem cindy_arrival_speed (t1 t2: ℕ) (h₁: t2 = t1 + 3 / 4) (d: ℕ) (h2: d = 20 * t1) (h3: t3 = t1 + 1 / 2) :
  cindy_speed d t1 t2 t3 := by
  sorry

end cindy_arrival_speed_l1312_131229


namespace sufficient_but_not_necessary_l1312_131276

-- Define what it means for α to be of the form (π/6 + 2kπ) where k ∈ ℤ
def is_pi_six_plus_two_k_pi (α : ℝ) : Prop :=
  ∃ k : ℤ, α = Real.pi / 6 + 2 * k * Real.pi

-- Define the condition sin α = 1 / 2
def sin_is_half (α : ℝ) : Prop :=
  Real.sin α = 1 / 2

-- The theorem stating that the given condition is a sufficient but not necessary condition
theorem sufficient_but_not_necessary (α : ℝ) :
  is_pi_six_plus_two_k_pi α → sin_is_half α ∧ ¬ (sin_is_half α → is_pi_six_plus_two_k_pi α) :=
by
  sorry

end sufficient_but_not_necessary_l1312_131276


namespace train_length_proof_l1312_131247

-- Defining the conditions
def speed_kmph : ℕ := 72
def platform_length : ℕ := 250  -- in meters
def time_seconds : ℕ := 26

-- Conversion factor from kmph to m/s
def kmph_to_mps (v : ℕ) : ℕ := (v * 1000) / 3600

-- The main goal: the length of the train
def train_length (speed_kmph : ℕ) (platform_length : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_mps := kmph_to_mps speed_kmph
  let total_distance := speed_mps * time_seconds
  total_distance - platform_length

theorem train_length_proof : train_length speed_kmph platform_length time_seconds = 270 := 
by 
  unfold train_length kmph_to_mps
  sorry

end train_length_proof_l1312_131247


namespace number_of_books_l1312_131231

-- Define the given conditions as variables
def movies_in_series : Nat := 62
def books_read : Nat := 4
def books_yet_to_read : Nat := 15

-- State the proposition we need to prove
theorem number_of_books : (books_read + books_yet_to_read) = 19 :=
by
  sorry

end number_of_books_l1312_131231


namespace even_numbers_average_l1312_131280

theorem even_numbers_average (n : ℕ) (h1 : 2 * (n * (n + 1)) = 22 * n) : n = 10 :=
by
  sorry

end even_numbers_average_l1312_131280


namespace last_four_digits_of_power_of_5_2017_l1312_131227

theorem last_four_digits_of_power_of_5_2017 :
  (5 ^ 2017 % 10000) = 3125 :=
by
  sorry

end last_four_digits_of_power_of_5_2017_l1312_131227
