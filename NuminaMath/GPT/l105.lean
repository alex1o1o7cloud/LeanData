import Mathlib

namespace find_X_l105_105887

theorem find_X (X : ℝ) (h : (X + 200 / 90) * 90 = 18200) : X = 18000 :=
sorry

end find_X_l105_105887


namespace sequence_x_value_l105_105412

theorem sequence_x_value
  (z y x : ℤ)
  (h1 : z + (-2) = -1)
  (h2 : y + 1 = -2)
  (h3 : x + (-3) = 1) :
  x = 4 := 
sorry

end sequence_x_value_l105_105412


namespace num_true_statements_l105_105910

theorem num_true_statements :
  (if (2 : ℝ) = 2 then (2 : ℝ)^2 - 4 = 0 else false) ∧
  ((∀ (x : ℝ), x^2 - 4 = 0 → x = 2) ∨ (∃ (x : ℝ), x^2 - 4 = 0 ∧ x ≠ 2)) ∧
  ((∀ (x : ℝ), x ≠ 2 → x^2 - 4 ≠ 0) ∨ (∃ (x : ℝ), x ≠ 2 ∧ x^2 - 4 = 0)) ∧
  ((∀ (x : ℝ), x^2 - 4 ≠ 0 → x ≠ 2) ∨ (∃ (x : ℝ), x^2 - 4 ≠ 0 ∧ x = 2)) :=
sorry

end num_true_statements_l105_105910


namespace solution_exists_for_100_100_l105_105860

def exists_positive_integers_sum_of_cubes (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a^3 + b^3 + c^3 + d^3 = x

theorem solution_exists_for_100_100 : exists_positive_integers_sum_of_cubes (100 ^ 100) :=
by
  sorry

end solution_exists_for_100_100_l105_105860


namespace find_p_l105_105243

variable (p q : ℝ) (k : ℕ)

theorem find_p (h_sum : ∀ (α β : ℝ), α + β = 2) (h_prod : ∀ (α β : ℝ), α * β = k) (hk : k > 0) :
  p = -2 := by
  sorry

end find_p_l105_105243


namespace shaded_area_percentage_is_100_l105_105832

-- Definitions and conditions
def square_side := 6
def square_area := square_side * square_side

def rect1_area := 2 * 2
def rect2_area := (5 * 5) - (3 * 3)
def rect3_area := 6 * 6

-- Percentage shaded calculation
def shaded_area := square_area
def percentage_shaded := (shaded_area / square_area) * 100

-- Lean 4 statement for the problem
theorem shaded_area_percentage_is_100 :
  percentage_shaded = 100 :=
by
  sorry

end shaded_area_percentage_is_100_l105_105832


namespace tangent_sum_formula_l105_105264

noncomputable def theta : ℝ := sorry -- Placeholder for specific value of θ

theorem tangent_sum_formula :
  (θ ∈ Ioo (-π / 2) 0) →
  (cos θ = sqrt 17 / 17) →
  tan (θ + π / 4) = -3 / 5 :=
by
  intros h1 h2
  sorry

end tangent_sum_formula_l105_105264


namespace hypotenuse_length_l105_105291

theorem hypotenuse_length (a b c : ℝ) (h_right_angled : c^2 = a^2 + b^2) (h_sum_of_squares : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l105_105291


namespace chair_cost_l105_105034

/--
Nadine went to a garage sale and spent $56. She bought a table for $34 and 2 chairs.
Each chair cost the same amount.
Prove that one chair cost $11.
-/
theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) (total_cost : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  total_cost = 56 - 34 →
  total_cost / num_chairs = 11 :=
by
  sorry

end chair_cost_l105_105034


namespace semicircle_circumference_approx_l105_105221

def rectangle_length : ℝ := 20
def rectangle_breadth : ℝ := 14
def rectangle_perimeter : ℝ := 2 * (rectangle_length + rectangle_breadth)
def side_of_square : ℝ := rectangle_perimeter / 4
def diameter_of_semicircle : ℝ := side_of_square
def circumference_of_semicircle : ℝ := (Real.pi * diameter_of_semicircle) / 2 + diameter_of_semicircle

theorem semicircle_circumference_approx :
  abs(circumference_of_semicircle - 43.70) < 0.01 :=
by
  sorry

end semicircle_circumference_approx_l105_105221


namespace main_theorem_l105_105534

variable (a : ℝ)

def M : Set ℝ := {x | x > 1 / 2 ∧ x < 1} ∪ {x | x > 1}
def N : Set ℝ := {x | x > 0 ∧ x ≤ 1 / 2}

theorem main_theorem : M ∩ N = ∅ :=
by
  sorry

end main_theorem_l105_105534


namespace parallel_condition_l105_105128

def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem parallel_condition (x : ℝ) : 
  let a := (2, 1)
  let b := (3 * x ^ 2 - 1, x)
  (x = 1 → are_parallel a b) ∧ 
  ∃ x', x' ≠ 1 ∧ are_parallel a (3 * x' ^ 2 - 1, x') :=
by
  sorry

end parallel_condition_l105_105128


namespace average_pushups_is_correct_l105_105381

theorem average_pushups_is_correct :
  ∀ (David Zachary Emily : ℕ),
    David = 510 →
    Zachary = David - 210 →
    Emily = David - 132 →
    (David + Zachary + Emily) / 3 = 396 :=
by
  intro David Zachary Emily hDavid hZachary hEmily
  -- All calculations and proofs will go here, but we'll leave them as sorry for now.
  sorry

end average_pushups_is_correct_l105_105381


namespace not_diff_of_squares_count_l105_105727

theorem not_diff_of_squares_count :
  ∃ n : ℕ, n ≤ 1000 ∧
  (∀ k : ℕ, k > 0 ∧ k ≤ 1000 → (∃ a b : ℤ, k = a^2 - b^2 ↔ k % 4 ≠ 2)) ∧
  n = 250 :=
sorry

end not_diff_of_squares_count_l105_105727


namespace expected_turns_formula_l105_105990

noncomputable def expected_turns (n : ℕ) : ℝ :=
  n + 0.5 - (n - 0.5) * (1 / (Real.sqrt (Real.pi * (n - 1))))

theorem expected_turns_formula (n : ℕ) (h : n > 1) :
  expected_turns n = n + 0.5 - (n - 0.5) * (1 / (Real.sqrt (Real.pi * (n - 1)))) :=
by
  unfold expected_turns
  sorry

end expected_turns_formula_l105_105990


namespace factorization_ce_sum_eq_25_l105_105778

theorem factorization_ce_sum_eq_25 {C E : ℤ} (h : (C * x - 13) * (E * x - 7) = 20 * x^2 - 87 * x + 91) : 
  C * E + C = 25 :=
sorry

end factorization_ce_sum_eq_25_l105_105778


namespace count_non_representable_as_diff_of_squares_l105_105725

theorem count_non_representable_as_diff_of_squares :
  let count := (Finset.filter (fun n => ∃ k, n = 4 * k + 2 ∧ 1 ≤ n ∧ n ≤ 1000) (Finset.range 1001)).card in
  count = 250 :=
by
  sorry

end count_non_representable_as_diff_of_squares_l105_105725


namespace opposite_of_neg_2_is_2_l105_105809

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l105_105809


namespace points_deducted_for_incorrect_answer_is_5_l105_105747

-- Define the constants and variables used in the problem
def total_questions : ℕ := 30
def points_per_correct_answer : ℕ := 20
def correct_answers : ℕ := 19
def incorrect_answers : ℕ := total_questions - correct_answers
def final_score : ℕ := 325

-- Define a function that models the total score calculation
def calculate_final_score (points_deducted_per_incorrect : ℕ) : ℕ :=
  (correct_answers * points_per_correct_answer) - (incorrect_answers * points_deducted_per_incorrect)

-- The theorem that states the problem and expected solution
theorem points_deducted_for_incorrect_answer_is_5 :
  ∃ (x : ℕ), calculate_final_score x = final_score ∧ x = 5 :=
by
  sorry

end points_deducted_for_incorrect_answer_is_5_l105_105747


namespace max_discount_rate_l105_105621

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l105_105621


namespace star_3_4_equals_8_l105_105735

def star (a b : ℕ) : ℕ := 4 * a + 5 * b - 2 * a * b

theorem star_3_4_equals_8 : star 3 4 = 8 := by
  sorry

end star_3_4_equals_8_l105_105735


namespace circle_eq_l105_105335

theorem circle_eq (x y : ℝ) (h k r : ℝ) (hc : h = 3) (kc : k = 1) (rc : r = 5) :
  (x - h)^2 + (y - k)^2 = r^2 ↔ (x - 3)^2 + (y - 1)^2 = 25 :=
by
  sorry

end circle_eq_l105_105335


namespace inequality_true_l105_105132

variables {a b : ℝ}
variables (c : ℝ)

theorem inequality_true (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) : a^3 * b^2 < a^2 * b^3 :=
by sorry

end inequality_true_l105_105132


namespace remainder_of_2_pow_23_mod_5_l105_105609

theorem remainder_of_2_pow_23_mod_5 
    (h1 : (2^2) % 5 = 4)
    (h2 : (2^3) % 5 = 3)
    (h3 : (2^4) % 5 = 1) :
    (2^23) % 5 = 3 :=
by
  sorry

end remainder_of_2_pow_23_mod_5_l105_105609


namespace opposite_of_neg_2_is_2_l105_105807

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l105_105807


namespace chair_cost_l105_105033

/--
Nadine went to a garage sale and spent $56. She bought a table for $34 and 2 chairs.
Each chair cost the same amount.
Prove that one chair cost $11.
-/
theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) (total_cost : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  total_cost = 56 - 34 →
  total_cost / num_chairs = 11 :=
by
  sorry

end chair_cost_l105_105033


namespace binomial_8_choose_4_l105_105511

theorem binomial_8_choose_4 : nat.choose 8 4 = 70 :=
by sorry

end binomial_8_choose_4_l105_105511


namespace opposite_of_neg_two_is_two_l105_105813

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l105_105813


namespace opposite_of_neg_two_l105_105796

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l105_105796


namespace nails_per_plank_l105_105710

theorem nails_per_plank {total_nails planks : ℕ} (h1 : total_nails = 4) (h2 : planks = 2) :
  total_nails / planks = 2 := by
  sorry

end nails_per_plank_l105_105710


namespace roadRepairDays_l105_105089

-- Definitions from the conditions
def dailyRepairLength1 : ℕ := 6
def daysToFinish1 : ℕ := 8
def totalLengthOfRoad : ℕ := dailyRepairLength1 * daysToFinish1
def dailyRepairLength2 : ℕ := 8
def daysToFinish2 : ℕ := totalLengthOfRoad / dailyRepairLength2

-- Theorem to be proven
theorem roadRepairDays :
  daysToFinish2 = 6 :=
by
  sorry

end roadRepairDays_l105_105089


namespace max_cables_cut_l105_105056

/-- 
Prove that given 200 computers connected by 345 cables initially forming a single cluster, after 
cutting cables to form 8 clusters, the maximum possible number of cables that could have been 
cut is 153.
--/
theorem max_cables_cut (computers : ℕ) (initial_cables : ℕ) (final_clusters : ℕ) (initial_clusters : ℕ) 
  (minimal_cables : ℕ) (cuts : ℕ) : 
  computers = 200 ∧ initial_cables = 345 ∧ final_clusters = 8 ∧ initial_clusters = 1 ∧ 
  minimal_cables = computers - final_clusters ∧ 
  cuts = initial_cables - minimal_cables →
  cuts = 153 := 
sorry

end max_cables_cut_l105_105056


namespace ratio_melina_alma_age_l105_105597

theorem ratio_melina_alma_age
  (A M : ℕ)
  (alma_score : ℕ)
  (h1 : M = 60)
  (h2 : alma_score = 40)
  (h3 : A + M = 2 * alma_score)
  : M / A = 3 :=
by
  sorry

end ratio_melina_alma_age_l105_105597


namespace opposite_of_neg_two_is_two_l105_105787

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l105_105787


namespace ef_plus_e_l105_105194

-- Define the polynomial expression
def polynomial_expr (y : ℤ) := 15 * y^2 - 82 * y + 48

-- Define the factorized form
def factorized_form (E F : ℤ) (y : ℤ) := (E * y - 16) * (F * y - 3)

-- Define the main statement to prove
theorem ef_plus_e : ∃ E F : ℤ, E * F + E = 20 ∧ ∀ y : ℤ, polynomial_expr y = factorized_form E F y :=
by {
  sorry
}

end ef_plus_e_l105_105194


namespace rectangle_dimensions_l105_105022

theorem rectangle_dimensions (x : ℝ) (h : 4 * x * x = 120) : x = Real.sqrt 30 ∧ 4 * x = 4 * Real.sqrt 30 :=
by
  sorry

end rectangle_dimensions_l105_105022


namespace doctor_lawyer_ratio_l105_105408

variables {d l : ℕ} -- Number of doctors and lawyers

-- Conditions
def avg_age_group (d l : ℕ) : Prop := (40 * d + 55 * l) / (d + l) = 45

-- Theorem: Given the conditions, the ratio of doctors to lawyers is 2:1.
theorem doctor_lawyer_ratio (hdl : avg_age_group d l) : d / l = 2 :=
sorry

end doctor_lawyer_ratio_l105_105408


namespace stored_energy_in_doubled_square_l105_105454

noncomputable def energy (q : ℝ) (d : ℝ) : ℝ := q^2 / d

theorem stored_energy_in_doubled_square (q d : ℝ) (h : energy q d * 4 = 20) :
  energy q (2 * d) * 4 = 10 := by
  -- Add steps: Show that energy proportional to 1/d means energy at 2d is half compared to at d
  sorry

end stored_energy_in_doubled_square_l105_105454


namespace leak_empties_tank_in_4_hours_l105_105484

theorem leak_empties_tank_in_4_hours
  (A_fills_in : ℝ)
  (A_with_leak_fills_in : ℝ) : 
  (∀ (L : ℝ), A_fills_in = 2 ∧ A_with_leak_fills_in = 4 → L = (1 / 4) → 1 / L = 4) :=
by 
  sorry

end leak_empties_tank_in_4_hours_l105_105484


namespace polynomial_identity_l105_105185

theorem polynomial_identity (x : ℝ) (h₁ : x^5 - 3*x + 2 = 0) (h₂ : x ≠ 1) : 
  x^4 + x^3 + x^2 + x + 1 = 3 := 
by 
  sorry

end polynomial_identity_l105_105185


namespace power_function_point_l105_105271

theorem power_function_point (a : ℝ) (h : (2 : ℝ) ^ a = (1 / 2 : ℝ)) : a = -1 :=
by sorry

end power_function_point_l105_105271


namespace find_f_2012_l105_105900

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1 / 4
axiom f_condition2 : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem find_f_2012 : f 2012 = -1 / 4 := 
sorry

end find_f_2012_l105_105900


namespace halfway_fraction_l105_105199

theorem halfway_fraction (a b : ℚ) (h1 : a = 2 / 9) (h2 : b = 1 / 3) :
  (a + b) / 2 = 5 / 18 :=
by
  sorry

end halfway_fraction_l105_105199


namespace solve_pair_l105_105853

theorem solve_pair (x y : ℕ) (h₁ : x = 12785 ∧ y = 12768 ∨ x = 11888 ∧ y = 11893 ∨ x = 12784 ∧ y = 12770 ∨ x = 1947 ∧ y = 1945) :
  1983 = 1982 * 11888 - 1981 * 11893 :=
by {
  sorry
}

end solve_pair_l105_105853


namespace trapezoid_perimeter_l105_105556

open Real EuclideanGeometry

/-- In the trapezoid ABCD, the bases AD and BC are 8 and 18 respectively. It is known that
the circumcircle of triangle ABD is tangent to the lines BC and CD. Prove that the perimeter
of the trapezoid is 56. -/
theorem trapezoid_perimeter (A B C D : Point)
  (h1 : dist A D = 8) (h2 : dist B C = 18)
  (h_circum : ∃ O r, Circle O r ∧ Tangent O r A B D B C ∧ Tangent O r A B D C D) :
  dist A B + dist A D + dist B C + dist C D = 56 := 
sorry

end trapezoid_perimeter_l105_105556


namespace max_discount_rate_l105_105668

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l105_105668


namespace min_value_frac_l105_105395

theorem min_value_frac (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = 10) : 
  ∃ x, (x = (1 / m) + (4 / n)) ∧ (∀ y, y = (1 / m) + (4 / n) → y ≥ 9 / 10) :=
sorry

end min_value_frac_l105_105395


namespace number_of_ways_l105_105743

constant Grid : Type
constant cell : Type
constant cells_2_or_5 : Finset cell
constant cells_3_or_4 : Finset cell
constant cells_1_or_6_one : Finset cell
constant cells_1_or_6_two : Finset cell

-- The problem grid setup
axiom six_by_six_grid (grid : Grid) : 
  ∀ r c, 0 ≤ r < 6 → 0 ≤ c < 6 → ∃! n, 1 ≤ n ∧ n ≤ 6 ∧ ∃ filled, filled n r c

-- The constraints for cells to be filled
axiom constraints (grid : Grid) : 
  Finset.card cells_2_or_5 = 4 →
  Finset.card cells_3_or_4 = 4 →
  Finset.card cells_1_or_6_one = 4 →
  Finset.card cells_1_or_6_two = 4 →

-- The actual cells can only be filled by the specific conditions
  (∀ cell ∈ cells_2_or_5, filled_with_two_or_five grid cell) →
  (∀ cell ∈ cells_3_or_4, filled_with_three_or_four grid cell) →
  (∀ cell ∈ cells_1_or_6_one, filled_with_one_or_six grid cell) →
  (∀ cell ∈ cells_1_or_6_two, filled_with_one_or_six grid cell)

-- The final proof statement that the number of fills equals 16
theorem number_of_ways (grid : Grid) : number_of_ways_to_fill (grid) = 16 := 
sorry

end number_of_ways_l105_105743


namespace steven_has_19_peaches_l105_105300

-- Conditions
def jill_peaches : ℕ := 6
def steven_peaches : ℕ := jill_peaches + 13

-- Statement to prove
theorem steven_has_19_peaches : steven_peaches = 19 :=
by {
    -- Proof steps would go here
    sorry
}

end steven_has_19_peaches_l105_105300


namespace num_solutions_eq_three_l105_105957

theorem num_solutions_eq_three :
  (∃ n : Nat, (x : ℝ) → (x^2 - 4) * (x^2 - 1) = (x^2 + 3 * x + 2) * (x^2 - 8 * x + 7) → n = 3) :=
sorry

end num_solutions_eq_three_l105_105957


namespace space_left_over_l105_105482

theorem space_left_over (D B : ℕ) (wall_length desk_length bookcase_length : ℝ) (h_wall : wall_length = 15)
  (h_desk : desk_length = 2) (h_bookcase : bookcase_length = 1.5) (h_eq : D = B)
  (h_max : 2 * D + 1.5 * B ≤ wall_length) :
  ∃ w : ℝ, w = wall_length - (D * desk_length + B * bookcase_length) ∧ w = 1 :=
by
  sorry

end space_left_over_l105_105482


namespace paper_area_difference_l105_105538

def sheet1_length : ℕ := 14
def sheet1_width : ℕ := 12
def sheet2_length : ℕ := 9
def sheet2_width : ℕ := 14

def area (length : ℕ) (width : ℕ) : ℕ := length * width

def combined_area (length : ℕ) (width : ℕ) : ℕ := 2 * area length width

theorem paper_area_difference :
  combined_area sheet1_length sheet1_width - combined_area sheet2_length sheet2_width = 84 := 
by 
  sorry

end paper_area_difference_l105_105538


namespace max_discount_rate_l105_105663

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l105_105663


namespace linear_system_reduction_transformation_l105_105345

theorem linear_system_reduction_transformation :
  ∀ (use_substitution_or_elimination : Bool), 
    (use_substitution_or_elimination = true) ∨ (use_substitution_or_elimination = false) → 
    "Reduction and transformation" = "Reduction and transformation" :=
by
  intro use_substitution_or_elimination h
  sorry

end linear_system_reduction_transformation_l105_105345


namespace opposite_of_neg_two_l105_105820

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l105_105820


namespace trajectory_of_Q_l105_105531

variables {P Q M : ℝ × ℝ}

-- Define the conditions as Lean predicates
def is_midpoint (M P Q : ℝ × ℝ) : Prop :=
  M = (0, 4) ∧ M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def point_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 - 2 = 0

-- Define the theorem that needs to be proven
theorem trajectory_of_Q :
  (∃ P Q M : ℝ × ℝ, is_midpoint M P Q ∧ point_on_line P) →
  ∃ Q : ℝ × ℝ, (∀ P : ℝ × ℝ, point_on_line P → is_midpoint (0,4) P Q → Q.1 + Q.2 - 6 = 0) :=
by sorry

end trajectory_of_Q_l105_105531


namespace rahul_share_of_payment_l105_105441

def work_rate_rahul : ℚ := 1 / 3
def work_rate_rajesh : ℚ := 1 / 2
def total_payment : ℚ := 150

theorem rahul_share_of_payment : (work_rate_rahul / (work_rate_rahul + work_rate_rajesh)) * total_payment = 60 := by
  sorry

end rahul_share_of_payment_l105_105441


namespace average_weight_14_children_l105_105951

theorem average_weight_14_children 
  (average_weight_boys : ℕ → ℤ → ℤ)
  (average_weight_girls : ℕ → ℤ → ℤ)
  (total_children : ℕ)
  (total_weight : ℤ)
  (total_average_weight : ℤ)
  (boys_count : ℕ)
  (girls_count : ℕ)
  (boys_average : ℤ)
  (girls_average : ℤ) :
  boys_count = 8 →
  girls_count = 6 →
  boys_average = 160 →
  girls_average = 130 →
  total_children = boys_count + girls_count →
  total_weight = average_weight_boys boys_count boys_average + average_weight_girls girls_count girls_average →
  average_weight_boys boys_count boys_average = boys_count * boys_average →
  average_weight_girls girls_count girls_average = girls_count * girls_average →
  total_average_weight = total_weight / total_children →
  total_average_weight = 147 :=
by
  sorry

end average_weight_14_children_l105_105951


namespace area_of_gray_region_l105_105241

open Real

-- Define the circles and the radii.
def circleC_center : Prod Real Real := (5, 5)
def radiusC : Real := 5

def circleD_center : Prod Real Real := (15, 5)
def radiusD : Real := 5

-- The main theorem stating the area of the gray region bound by the circles and the x-axis.
theorem area_of_gray_region : 
  let area_rectangle := (10:Real) * (5:Real)
  let area_sectors := (2:Real) * ((1/4) * (5:Real)^2 * π)
  area_rectangle - area_sectors = 50 - 12.5 * π :=
by
  sorry

end area_of_gray_region_l105_105241


namespace part_a_part_b_l105_105057

-- Define a polygon type
structure Polygon :=
  (sides : ℕ)
  (area : ℝ)
  (grid_size : ℕ)

-- Define a function to verify drawable polygon
def DrawablePolygon (p : Polygon) : Prop :=
  ∃ (n : ℕ), p.grid_size = n ∧ p.area = n ^ 2

-- Part (a): 20-sided polygon with an area of 9
theorem part_a : DrawablePolygon {sides := 20, area := 9, grid_size := 3} :=
by
  sorry

-- Part (b): 100-sided polygon with an area of 49
theorem part_b : DrawablePolygon {sides := 100, area := 49, grid_size := 7} :=
by
  sorry

end part_a_part_b_l105_105057


namespace lemonade_stand_total_profit_l105_105371

theorem lemonade_stand_total_profit :
  let day1_revenue := 21 * 4
  let day1_expenses := 10 + 5 + 3
  let day1_profit := day1_revenue - day1_expenses

  let day2_revenue := 18 * 5
  let day2_expenses := 12 + 6 + 4
  let day2_profit := day2_revenue - day2_expenses

  let day3_revenue := 25 * 4
  let day3_expenses := 8 + 4 + 3 + 2
  let day3_profit := day3_revenue - day3_expenses

  let total_profit := day1_profit + day2_profit + day3_profit

  total_profit = 217 := by
    sorry

end lemonade_stand_total_profit_l105_105371


namespace calculate_taxes_l105_105546

def gross_pay : ℝ := 4500
def tax_rate_1 : ℝ := 0.10
def tax_rate_2 : ℝ := 0.15
def tax_rate_3 : ℝ := 0.20
def income_bracket_1 : ℝ := 1500
def income_bracket_2 : ℝ := 2000
def income_bracket_remaining : ℝ := gross_pay - income_bracket_1 - income_bracket_2
def standard_deduction : ℝ := 100

theorem calculate_taxes :
  let tax_1 := tax_rate_1 * income_bracket_1
  let tax_2 := tax_rate_2 * income_bracket_2
  let tax_3 := tax_rate_3 * income_bracket_remaining
  let total_tax := tax_1 + tax_2 + tax_3
  let tax_after_deduction := total_tax - standard_deduction
  tax_after_deduction = 550 :=
by
  sorry

end calculate_taxes_l105_105546


namespace blackBurgerCost_l105_105186

def ArevaloFamilyBill (smokySalmonCost blackBurgerCost chickenKatsuCost totalBill : ℝ) : Prop :=
  smokySalmonCost = 40 ∧ chickenKatsuCost = 25 ∧ 
  totalBill = smokySalmonCost + blackBurgerCost + chickenKatsuCost + 
    0.15 * (smokySalmonCost + blackBurgerCost + chickenKatsuCost)

theorem blackBurgerCost (smokySalmonCost chickenKatsuCost change : ℝ) (B : ℝ) 
  (h1 : smokySalmonCost = 40) 
  (h2 : chickenKatsuCost = 25)
  (h3 : 100 - change = 92) 
  (h4 : ArevaloFamilyBill smokySalmonCost B chickenKatsuCost 92) : 
  B = 15 :=
sorry

end blackBurgerCost_l105_105186


namespace area_of_PINE_l105_105387

def PI := 6
def IN := 15
def NE := 6
def EP := 25
def sum_angles := 60 

theorem area_of_PINE : 
  (∃ (area : ℝ), area = (100 * Real.sqrt 3) / 3) := 
sorry

end area_of_PINE_l105_105387


namespace solveSystem_l105_105448

variable {r p q x y z : ℝ}

theorem solveSystem :
  
  -- The given system of equations
  (x + r * y - q * z = 1) ∧
  (-r * x + y + p * z = r) ∧ 
  (q * x - p * y + z = -q) →

  -- Solution equivalence using determined
  x = (1 - r ^ 2 + p ^ 2 - q ^ 2) / (1 + r ^ 2 + p ^ 2 + q ^ 2) :=
by sorry

end solveSystem_l105_105448


namespace verify_z_relationship_l105_105965

variable {x y z : ℝ}

theorem verify_z_relationship (h1 : x > y) (h2 : y > 1) :
  z = (x + 3) - 2 * (y - 5) → z = x - 2 * y + 13 :=
by
  intros
  sorry

end verify_z_relationship_l105_105965


namespace half_angle_quadrants_l105_105266

variable (k : ℤ) (α : ℝ)

-- Conditions
def is_second_quadrant (α : ℝ) (k : ℤ) : Prop :=
  2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

-- Question: Determine the quadrant(s) in which α / 2 lies under the given condition.
theorem half_angle_quadrants (α : ℝ) (k : ℤ) 
  (h : is_second_quadrant α k) : 
  ((k * Real.pi + Real.pi / 4 < α / 2) ∧ (α / 2 < k * Real.pi + Real.pi / 2)) ↔ 
  (∃ (m : ℤ), (2 * m * Real.pi < α / 2 ∧ α / 2 < 2 * m * Real.pi + Real.pi)) ∨ ( ∃ (m : ℤ), (2 * m * Real.pi + Real.pi < α / 2 ∧ α / 2 < 2 * m * Real.pi + 2 * Real.pi)) := 
sorry

end half_angle_quadrants_l105_105266


namespace hyperbola_eccentricity_is_2_l105_105130

noncomputable theory

-- Condition: Definitions based on the problem description
variables (a b c : ℝ) (e : ℝ)
def hyperbola (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def focus1 : ℝ × ℝ := (-c, 0)
def focus2 : ℝ × ℝ := (c, 0)
def asymptote (x y : ℝ) : Prop := y = (b / a) * x
def symmetric_point_on_circle (p : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : Prop := 
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

-- Question/Proposition to prove: The eccentricity of the hyperbola is 2
theorem hyperbola_eccentricity_is_2
  (h_hyperbola : ∀ x y, hyperbola a b x y)
  (h_focus1 : focus1 c = (-c, 0))
  (h_focus2 : focus2 c = (c, 0))
  (h_asymptote_distance : (c * b) / (real.sqrt(b^2 + a^2)) = b)
  (h_symmetric : symmetric_point_on_circle (c, 0) (-c, 0) (real.abs c)) :
  e = 2 :=
sorry

end hyperbola_eccentricity_is_2_l105_105130


namespace inradius_of_regular_tetrahedron_l105_105553

theorem inradius_of_regular_tetrahedron (h r : ℝ) (S : ℝ) 
  (h_eq: 4 * (1/3) * S * r = (1/3) * S * h) : r = (1/4) * h :=
sorry

end inradius_of_regular_tetrahedron_l105_105553


namespace vector_subtraction_proof_l105_105123

def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (1, -6)
def scalar1 : ℝ := 2
def scalar2 : ℝ := 3

theorem vector_subtraction_proof :
  v1 - (scalar2 • (scalar1 • v2)) = (-3, 32) := by
  sorry

end vector_subtraction_proof_l105_105123


namespace B_completes_remaining_work_in_2_days_l105_105076

theorem B_completes_remaining_work_in_2_days 
  (A_work_rate : ℝ) (B_work_rate : ℝ) (total_work : ℝ) 
  (A_days_to_complete : A_work_rate = 1 / 2) 
  (B_days_to_complete : B_work_rate = 1 / 6) 
  (combined_work_1_day : A_work_rate + B_work_rate = 2 / 3) : 
  (total_work - (A_work_rate + B_work_rate)) / B_work_rate = 2 := 
by
  sorry

end B_completes_remaining_work_in_2_days_l105_105076


namespace transform_unit_square_l105_105167

-- Define the unit square vertices in the xy-plane
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (0, 1)

-- Transformation functions from the xy-plane to the uv-plane
def transform_u (x y : ℝ) : ℝ := x^2 - y^2
def transform_v (x y : ℝ) : ℝ := x * y

-- Vertex transformation results
def O_image : ℝ × ℝ := (transform_u 0 0, transform_v 0 0)  -- (0,0)
def A_image : ℝ × ℝ := (transform_u 1 0, transform_v 1 0)  -- (1,0)
def B_image : ℝ × ℝ := (transform_u 1 1, transform_v 1 1)  -- (0,1)
def C_image : ℝ × ℝ := (transform_u 0 1, transform_v 0 1)  -- (-1,0)

-- The Lean 4 theorem statement
theorem transform_unit_square :
  O_image = (0, 0) ∧
  A_image = (1, 0) ∧
  B_image = (0, 1) ∧
  C_image = (-1, 0) :=
  by sorry

end transform_unit_square_l105_105167


namespace cubic_has_three_natural_roots_l105_105118

theorem cubic_has_three_natural_roots (p : ℝ) :
  (∃ (x1 x2 x3 : ℕ), 5 * (x1:ℝ)^3 - 5 * (p + 1) * (x1:ℝ)^2 + (71 * p - 1) * (x1:ℝ) + 1 = 66 * p ∧
                     5 * (x2:ℝ)^3 - 5 * (p + 1) * (x2:ℝ)^2 + (71 * p - 1) * (x2:ℝ) + 1 = 66 * p ∧
                     5 * (x3:ℝ)^3 - 5 * (p + 1) * (x3:ℝ)^2 + (71 * p - 1) * (x3:ℝ) + 1 = 66 * p) ↔ p = 76 :=
by sorry

end cubic_has_three_natural_roots_l105_105118


namespace ebay_ordered_cards_correct_l105_105560

noncomputable def initial_cards := 4
noncomputable def father_cards := 13
noncomputable def cards_given_to_dexter := 29
noncomputable def cards_kept := 20
noncomputable def bad_cards := 4

theorem ebay_ordered_cards_correct :
  let total_before_ebay := initial_cards + father_cards
  let total_after_giving_and_keeping := cards_given_to_dexter + cards_kept
  let ordered_before_bad := total_after_giving_and_keeping - total_before_ebay
  let ebay_ordered_cards := ordered_before_bad + bad_cards
  ebay_ordered_cards = 36 :=
by
  sorry

end ebay_ordered_cards_correct_l105_105560


namespace ellipse_area_l105_105233

-- Definitions based on the conditions
def cylinder_height : ℝ := 10
def cylinder_base_radius : ℝ := 1

-- Equivalent Proof Problem Statement
theorem ellipse_area
  (h : ℝ := cylinder_height)
  (r : ℝ := cylinder_base_radius)
  (ball_position_lower : ℝ := -4) -- derived from - (h / 2 - r)
  (ball_position_upper : ℝ := 4) -- derived from  (h / 2 - r)
  : (π * 4 * 2 = 16 * π) :=
by
  sorry

end ellipse_area_l105_105233


namespace lily_patch_cover_entire_lake_l105_105550

noncomputable def days_to_cover_half (initial_days : ℕ) := 33

theorem lily_patch_cover_entire_lake (initial_days : ℕ) (h : days_to_cover_half initial_days = 33) :
  initial_days + 1 = 34 :=
by
  sorry

end lily_patch_cover_entire_lake_l105_105550


namespace min_number_of_each_coin_l105_105430

def total_cost : ℝ := 1.30 + 0.75 + 0.50 + 0.45

def nickel_value : ℝ := 0.05
def dime_value : ℝ := 0.10
def quarter_value : ℝ := 0.25
def half_dollar_value : ℝ := 0.50

def min_coins :=
  ∃ (n q d h : ℕ), 
  (n ≥ 1) ∧ (q ≥ 1) ∧ (d ≥ 1) ∧ (h ≥ 1) ∧ 
  ((n * nickel_value) + (q * quarter_value) + (d * dime_value) + (h * half_dollar_value) = total_cost)

theorem min_number_of_each_coin :
  min_coins ↔ (5 * half_dollar_value + 1 * quarter_value + 2 * dime_value + 1 * nickel_value = total_cost) :=
by sorry

end min_number_of_each_coin_l105_105430


namespace investment_simple_compound_l105_105283

theorem investment_simple_compound (P y : ℝ) 
    (h1 : 600 = P * y * 2 / 100)
    (h2 : 615 = P * (1 + y/100)^2 - P) : 
    P = 285.71 :=
by
    sorry

end investment_simple_compound_l105_105283


namespace first_stopover_distance_l105_105018

theorem first_stopover_distance 
  (total_distance : ℕ) 
  (second_stopover_distance : ℕ) 
  (distance_after_second_stopover : ℕ) :
  total_distance = 436 → 
  second_stopover_distance = 236 → 
  distance_after_second_stopover = 68 →
  second_stopover_distance - (total_distance - second_stopover_distance - distance_after_second_stopover) = 104 :=
by
  intros
  sorry

end first_stopover_distance_l105_105018


namespace find_a1_l105_105717

-- Definitions used in the conditions
variables {a : ℕ → ℝ} -- Sequence a(n)
variable (n : ℕ) -- Number of terms
noncomputable def arithmeticSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))

noncomputable def arithmeticSeq (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a m + (n - m) * (a 2 - a 1)

theorem find_a1 (h_seq : arithmeticSeq a)
  (h_sum_first_100 : arithmeticSum a 100 = 100)
  (h_sum_last_100 : arithmeticSum (λ i => a (i + 900)) 100 = 1000) :
  a 1 = 101 / 200 :=
  sorry

end find_a1_l105_105717


namespace larger_number_is_50_l105_105737

theorem larger_number_is_50 (x y : ℤ) (h1 : 4 * y = 5 * x) (h2 : y - x = 10) : y = 50 :=
sorry

end larger_number_is_50_l105_105737


namespace hypotenuse_length_l105_105294

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 := 
by
  sorry

end hypotenuse_length_l105_105294


namespace eraser_cost_l105_105368

variable (P E : ℝ)
variable (h1 : E = P / 2)
variable (h2 : 20 * P = 80)

theorem eraser_cost : E = 2 := by 
  sorry

end eraser_cost_l105_105368


namespace matrix_power_101_l105_105921

noncomputable def matrix_B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem matrix_power_101 :
  (matrix_B ^ 101) = ![![0, 0, 1], ![1, 0, 0], ![0, 1, 0]] :=
  sorry

end matrix_power_101_l105_105921


namespace man_to_son_age_ratio_l105_105991

-- Definitions based on conditions
variable (son_age : ℕ) (man_age : ℕ)
variable (h1 : man_age = son_age + 18) -- The man is 18 years older than his son
variable (h2 : 2 * (son_age + 2) = man_age + 2) -- In two years, the man's age will be a multiple of the son's age
variable (h3 : son_age = 16) -- The present age of the son is 16

-- Theorem statement to prove the desired ratio
theorem man_to_son_age_ratio (son_age man_age : ℕ) (h1 : man_age = son_age + 18) (h2 : 2 * (son_age + 2) = man_age + 2) (h3 : son_age = 16) :
  (man_age + 2) / (son_age + 2) = 2 :=
by
  sorry

end man_to_son_age_ratio_l105_105991


namespace find_g_neg_one_l105_105456

theorem find_g_neg_one (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 1 / 2 → g x + g ((x + 2) / (2 - 4 * x)) = 3 * x) : 
  g (-1) = - 3 / 2 := 
sorry

end find_g_neg_one_l105_105456


namespace x_range_condition_l105_105457

-- Define the inequality and conditions
def inequality (x : ℝ) : Prop := x^2 + 2 * x < 8

-- The range of x must be (-4, 2)
theorem x_range_condition (x : ℝ) : inequality x → x > -4 ∧ x < 2 :=
by
  intro h
  sorry

end x_range_condition_l105_105457


namespace cuboid_edge_length_l105_105590

theorem cuboid_edge_length (x : ℝ) (h1 : 5 * 6 * x = 120) : x = 4 :=
by
  sorry

end cuboid_edge_length_l105_105590


namespace smaller_circle_radius_l105_105086

theorem smaller_circle_radius (A1 A2 : ℝ) 
  (h1 : A1 + 2 * A2 = 25 * Real.pi) 
  (h2 : ∃ d : ℝ, A1 + d = A2 ∧ A2 + d = A1 + 2 * A2) : 
  ∃ r : ℝ, r^2 = 5 ∧ Real.pi * r^2 = A1 :=
by
  sorry

end smaller_circle_radius_l105_105086


namespace student_A_probability_student_B_expectation_l105_105061

theorem student_A_probability : 
  (P : ℙ(ℕ → Prop))
  (P_A : P(λ n, set.to_finset n = {1,2,3}) = (1 / 2))
  (P_A_shots : ∀ a1 a2 a3 : Prop, 
    P ({#a1, a2, a3} ∈ tfinset.univ) = (1 / 2))
  (independence : ∀ a b : Prop, 
    P (a ∩ b) = P a * P b) :
  P({
    c a1 a2 a3 s1 s2 s3 | (P a1 ∩ P a2 ∩ P a3 = 2 / 3) 
  ∨ (P a1 ∩ P a2 ∩ P a3 = 3 / 3)
  }) = (1 / 2) :=
by
  /* Detailed proof steps, assuming required here, add: */
  sorry

theorem student_B_expectation :
  (P : ℙ(ℕ → Prop))
  (P_B : P(λ n, set.to_finset n = {1,2,3}) = (2 / 3))
  (independence : ∀ a b : Prop, 
    P (a ∩ b) = P a * P b)
  (X2 : P B(2 shots) = (4 / 9))
  (X3 : P B(3 shots) = (1 / 3))
  (X4 : P B(4 shots) = (2 / 9)) :
  ∑ b • P_B(X) = (25 / 9) :=
by
  /* Expected Value computation proof here, add: */
  sorry


end student_A_probability_student_B_expectation_l105_105061


namespace opposite_of_neg2_l105_105802

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l105_105802


namespace find_m_l105_105147

theorem find_m (m : ℤ) : m < 2 * Real.sqrt 3 ∧ 2 * Real.sqrt 3 < m + 1 → m = 3 :=
sorry

end find_m_l105_105147


namespace probability_before_third_ring_l105_105821

-- Definitions of the conditions
def prob_first_ring : ℝ := 0.2
def prob_second_ring : ℝ := 0.3

-- Theorem stating that the probability of being answered before the third ring is 0.5
theorem probability_before_third_ring : prob_first_ring + prob_second_ring = 0.5 :=
by
  sorry

end probability_before_third_ring_l105_105821


namespace find_inverse_mod_36_l105_105265

-- Given condition
def inverse_mod_17 := (17 * 23) % 53 = 1

-- Definition for the problem statement
def inverse_mod_36 : Prop := (36 * 30) % 53 = 1

theorem find_inverse_mod_36 (h : inverse_mod_17) : inverse_mod_36 :=
sorry

end find_inverse_mod_36_l105_105265


namespace equivalent_problem_l105_105306

-- Definitions that correspond to conditions
def valid_n (n : ℕ) : Prop := n < 13 ∧ (4 * n) % 13 = 1

-- The equivalent proof problem
theorem equivalent_problem (n : ℕ) (h : valid_n n) : ((3 ^ n) ^ 4 - 3) % 13 = 6 := by
  sorry

end equivalent_problem_l105_105306


namespace sum_to_common_fraction_l105_105878

theorem sum_to_common_fraction :
  (2 / 10 + 3 / 100 + 4 / 1000 + 5 / 10000 + 6 / 100000 : ℚ) = 733 / 12500 := by
  sorry

end sum_to_common_fraction_l105_105878


namespace equilateral_triangle_side_length_l105_105539

theorem equilateral_triangle_side_length (side_length_of_square : ℕ) (h : side_length_of_square = 21) :
    let total_length_of_string := 4 * side_length_of_square
    let side_length_of_triangle := total_length_of_string / 3
    side_length_of_triangle = 28 :=
by
  sorry

end equilateral_triangle_side_length_l105_105539


namespace values_of_b_for_real_root_l105_105516

noncomputable def polynomial_has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^5 + b * x^4 - x^3 + b * x^2 - x + b = 0

theorem values_of_b_for_real_root :
  {b : ℝ | polynomial_has_real_root b} = {b : ℝ | b ≤ -1 ∨ b ≥ 1} :=
sorry

end values_of_b_for_real_root_l105_105516


namespace opposite_of_neg_two_l105_105817

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l105_105817


namespace ratio_is_two_l105_105949

noncomputable def ratio_of_altitude_to_base (area base : ℕ) : ℕ :=
  have h : ℕ := area / base
  h / base

theorem ratio_is_two (area base : ℕ) (h : ℕ)  (h_area : area = 288) (h_base : base = 12) (h_altitude : h = area / base) : ratio_of_altitude_to_base area base = 2 :=
  by
    sorry 

end ratio_is_two_l105_105949


namespace negation_of_proposition_l105_105198

theorem negation_of_proposition (m : ℤ) : 
  (¬ (∃ x : ℤ, x^2 + 2*x + m ≤ 0)) ↔ ∀ x : ℤ, x^2 + 2*x + m > 0 :=
sorry

end negation_of_proposition_l105_105198


namespace finite_discrete_points_3_to_15_l105_105773

def goldfish_cost (n : ℕ) : ℕ := 18 * n

theorem finite_discrete_points_3_to_15 : 
  ∀ (n : ℕ), 3 ≤ n ∧ n ≤ 15 → 
  ∃ (C : ℕ), C = goldfish_cost n ∧ ∃ (x : ℕ), (n, C) = (x, goldfish_cost x) :=
by
  sorry

end finite_discrete_points_3_to_15_l105_105773


namespace sum_of_decimals_as_common_fraction_l105_105882

theorem sum_of_decimals_as_common_fraction:
  (2 / 10 : ℚ) + (3 / 100 : ℚ) + (4 / 1000 : ℚ) + (5 / 10000 : ℚ) + (6 / 100000 : ℚ) = 733 / 3125 :=
by
  -- Explanation and proof steps are omitted as directed.
  sorry

end sum_of_decimals_as_common_fraction_l105_105882


namespace initial_winning_percentage_calc_l105_105741

variable (W : ℝ)
variable (initial_matches : ℝ := 120)
variable (additional_wins : ℝ := 70)
variable (final_matches : ℝ := 190)
variable (final_average : ℝ := 0.52)
variable (initial_wins : ℝ := 29)

noncomputable def winning_percentage_initial :=
  (initial_wins / initial_matches) * 100

theorem initial_winning_percentage_calc :
  (W = initial_wins) →
  ((W + additional_wins) / final_matches = final_average) →
  winning_percentage_initial = 24.17 :=
by
  intros
  sorry

end initial_winning_percentage_calc_l105_105741


namespace no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant_l105_105161

theorem no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant :
    ∀ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
                     (0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) →
                     (a^2 + b^2 + c^2 + d^2 = 100) → False := by
  sorry

end no_combination_of_four_squares_equals_100_no_repeat_order_irrelevant_l105_105161


namespace largest_n_satisfies_conditions_l105_105122

theorem largest_n_satisfies_conditions :
  ∃ (n m a : ℤ), n = 313 ∧ n^2 = (m + 1)^3 - m^3 ∧ ∃ (k : ℤ), 2 * n + 103 = k^2 :=
by
  sorry

end largest_n_satisfies_conditions_l105_105122


namespace fraction_subtraction_l105_105071

theorem fraction_subtraction : (1 / 6 : ℚ) - (5 / 12) = -1 / 4 := 
by sorry

end fraction_subtraction_l105_105071


namespace greatest_perfect_square_power_of_3_under_200_l105_105174

theorem greatest_perfect_square_power_of_3_under_200 :
  ∃ n : ℕ, n < 200 ∧ (∃ k : ℕ, k % 2 = 0 ∧ n = 3 ^ k) ∧ ∀ m : ℕ, (m < 200 ∧ (∃ k : ℕ, k % 2 = 0 ∧ m = 3 ^ k)) → m ≤ n :=
  sorry

end greatest_perfect_square_power_of_3_under_200_l105_105174


namespace opposite_of_neg_two_is_two_l105_105786

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l105_105786


namespace opposite_of_neg_two_l105_105799

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l105_105799


namespace tank_overflows_after_24_minutes_l105_105984

theorem tank_overflows_after_24_minutes 
  (rateA : ℝ) (rateB : ℝ) (t : ℝ) 
  (hA : rateA = 1) 
  (hB : rateB = 4) :
  t - 1/4 * rateB + t * rateA = 1 → t = 2/5 :=
by 
  intros h
  -- the proof steps go here
  sorry

end tank_overflows_after_24_minutes_l105_105984


namespace other_number_more_than_42_l105_105053

theorem other_number_more_than_42 (a b : ℕ) (h1 : a + b = 96) (h2 : a = 42) : b - a = 12 := by
  sorry

end other_number_more_than_42_l105_105053


namespace money_collected_l105_105928

theorem money_collected
  (households_per_day : ℕ)
  (days : ℕ)
  (half_give_money : ℕ → ℕ)
  (total_money_collected : ℕ)
  (households_give_money : ℕ) :
  households_per_day = 20 →  
  days = 5 →
  total_money_collected = 2000 →
  half_give_money (households_per_day * days) = (households_per_day * days) / 2 →
  households_give_money = (households_per_day * days) / 2 →
  total_money_collected / households_give_money = 40
:= sorry

end money_collected_l105_105928


namespace sum_to_common_fraction_l105_105879

theorem sum_to_common_fraction :
  (2 / 10 + 3 / 100 + 4 / 1000 + 5 / 10000 + 6 / 100000 : ℚ) = 733 / 12500 := by
  sorry

end sum_to_common_fraction_l105_105879


namespace maximum_discount_rate_l105_105646

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l105_105646


namespace hyperbola_n_range_l105_105272

noncomputable def hyperbola_range_n (m n : ℝ) : Set ℝ :=
  {n | ∃ (m : ℝ), (m^2 + n) + (3 * m^2 - n) = 4 ∧ ((m^2 + n) * (3 * m^2 - n) > 0) }

theorem hyperbola_n_range : ∀ n : ℝ, n ∈ hyperbola_range_n m n ↔ -1 < n ∧ n < 3 :=
by
  sorry

end hyperbola_n_range_l105_105272


namespace opposite_of_neg_two_l105_105818

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l105_105818


namespace regular_tetrahedron_fourth_vertex_l105_105366

theorem regular_tetrahedron_fourth_vertex :
  ∃ (x y z : ℤ), 
    ((x, y, z) = (0, 0, 6) ∨ (x, y, z) = (0, 0, -6)) ∧
    ((x - 0) ^ 2 + (y - 0) ^ 2 + (z - 0) ^ 2 = 36) ∧
    ((x - 6) ^ 2 + (y - 0) ^ 2 + (z - 0) ^ 2 = 36) ∧
    ((x - 5) ^ 2 + (y - 0) ^ 2 + (z - 6) ^ 2 = 36) := 
by
  sorry

end regular_tetrahedron_fourth_vertex_l105_105366


namespace months_b_after_a_started_business_l105_105096

theorem months_b_after_a_started_business
  (A_initial : ℝ)
  (B_initial : ℝ)
  (profit_ratio : ℝ)
  (A_investment_time : ℕ)
  (B_investment_time : ℕ)
  (investment_ratio : A_initial * A_investment_time / (B_initial * B_investment_time) = profit_ratio) :
  B_investment_time = 6 :=
by
  -- Given:
  -- A_initial = 3500
  -- B_initial = 10500
  -- profit_ratio = 2 / 3
  -- A_investment_time = 12 months
  -- B_investment_time = 12 - x months
  -- We need to prove that x = 6 months such that investment ratio matches profit ratio.
  sorry

end months_b_after_a_started_business_l105_105096


namespace fraction_multiplication_l105_105377

theorem fraction_multiplication : (1 / 2) * (1 / 3) * (1 / 6) * 108 = 3 := by
  sorry

end fraction_multiplication_l105_105377


namespace cubic_has_three_natural_roots_l105_105119

theorem cubic_has_three_natural_roots (p : ℝ) :
  (∃ (x1 x2 x3 : ℕ), 5 * (x1:ℝ)^3 - 5 * (p + 1) * (x1:ℝ)^2 + (71 * p - 1) * (x1:ℝ) + 1 = 66 * p ∧
                     5 * (x2:ℝ)^3 - 5 * (p + 1) * (x2:ℝ)^2 + (71 * p - 1) * (x2:ℝ) + 1 = 66 * p ∧
                     5 * (x3:ℝ)^3 - 5 * (p + 1) * (x3:ℝ)^2 + (71 * p - 1) * (x3:ℝ) + 1 = 66 * p) ↔ p = 76 :=
by sorry

end cubic_has_three_natural_roots_l105_105119


namespace eliot_account_balance_l105_105940

variable (A E F : ℝ)

theorem eliot_account_balance
  (h1 : A > E)
  (h2 : F > A)
  (h3 : A - E = (1 : ℝ) / 12 * (A + E))
  (h4 : F - A = (1 : ℝ) / 8 * (F + A))
  (h5 : 1.1 * A = 1.2 * E + 21)
  (h6 : 1.05 * F = 1.1 * A + 40) :
  E = 210 := 
sorry

end eliot_account_balance_l105_105940


namespace opposite_of_neg_two_l105_105781

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l105_105781


namespace geom_series_sum_l105_105069

noncomputable def geom_sum (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1) 

theorem geom_series_sum (S : ℕ) (a r n : ℕ) (eq1 : a = 1) (eq2 : r = 3)
  (eq3 : 19683 = a * r^(n-1)) (S_eq : S = geom_sum a r n) : 
  S = 29524 :=
by
  sorry

end geom_series_sum_l105_105069


namespace max_discount_rate_l105_105623

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l105_105623


namespace min_val_of_q_l105_105926

theorem min_val_of_q (p q : ℕ) (h1 : 72 / 487 < p / q) (h2 : p / q < 18 / 121) : 
  ∃ p q : ℕ, (72 / 487 < p / q) ∧ (p / q < 18 / 121) ∧ q = 27 :=
sorry

end min_val_of_q_l105_105926


namespace roots_in_arithmetic_progression_l105_105968

theorem roots_in_arithmetic_progression (a b c : ℝ) :
  (∃ x1 x2 x3 : ℝ, (x2 = (x1 + x3) / 2) ∧ (x1 + x2 + x3 = -a) ∧ (x1 * x3 + x2 * (x1 + x3) = b) ∧ (x1 * x2 * x3 = -c)) ↔ 
  (27 * c = 3 * a * b - 2 * a^3 ∧ 3 * b ≤ a^2) :=
sorry

end roots_in_arithmetic_progression_l105_105968


namespace total_rainbow_nerds_is_36_l105_105359

def purple_candies : ℕ := 10
def yellow_candies : ℕ := purple_candies + 4
def green_candies : ℕ := yellow_candies - 2
def total_candies : ℕ := purple_candies + yellow_candies + green_candies

theorem total_rainbow_nerds_is_36 : total_candies = 36 := by
  sorry

end total_rainbow_nerds_is_36_l105_105359


namespace max_discount_rate_l105_105669

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l105_105669


namespace cape_may_shark_sightings_l105_105237

def total_shark_sightings (D C : ℕ) : Prop :=
  D + C = 40

def cape_may_sightings (D C : ℕ) : Prop :=
  C = 2 * D - 8

theorem cape_may_shark_sightings : 
  ∃ (C D : ℕ), total_shark_sightings D C ∧ cape_may_sightings D C ∧ C = 24 :=
by
  sorry

end cape_may_shark_sightings_l105_105237


namespace positive_difference_l105_105464

theorem positive_difference (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * y - 3 * x = 10) : |y - x| = 12 := by
  sorry

end positive_difference_l105_105464


namespace range_of_a_l105_105268

theorem range_of_a (x y z a : ℝ) 
    (h1 : x > 0) 
    (h2 : y > 0) 
    (h3 : z > 0) 
    (h4 : x + y + z = 1) 
    (h5 : a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) : 
    0 < a ∧ a ≤ 7 / 27 := 
  sorry

end range_of_a_l105_105268


namespace geometric_body_view_circle_l105_105007

theorem geometric_body_view_circle (P : Type) (is_circle : P → Prop) (is_sphere : P → Prop)
  (is_cylinder : P → Prop) (is_cone : P → Prop) (is_rectangular_prism : P → Prop) :
  (∀ x, is_sphere x → is_circle x) →
  (∃ x, is_cylinder x ∧ is_circle x) →
  (∃ x, is_cone x ∧ is_circle x) →
  ¬ (∃ x, is_rectangular_prism x ∧ is_circle x) :=
by
  intros h_sphere h_cylinder h_cone h_rectangular_prism
  sorry

end geometric_body_view_circle_l105_105007


namespace sofia_total_time_for_5_laps_sofia_total_time_in_minutes_and_seconds_l105_105944

noncomputable def calculate_time (distance1 distance2 speed1 speed2 : ℕ) : ℕ := 
  (distance1 / speed1) + (distance2 / speed2)

noncomputable def total_time_per_lap := calculate_time 200 100 4 6

theorem sofia_total_time_for_5_laps : total_time_per_lap * 5 = 335 := 
  by sorry

def converted_time (total_seconds : ℕ) : ℕ × ℕ :=
  (total_seconds / 60, total_seconds % 60)

theorem sofia_total_time_in_minutes_and_seconds :
  converted_time (total_time_per_lap * 5) = (5, 35) :=
  by sorry

end sofia_total_time_for_5_laps_sofia_total_time_in_minutes_and_seconds_l105_105944


namespace preferred_dividend_rate_l105_105763

noncomputable def dividend_rate_on_preferred_shares
  (preferred_shares : ℕ)
  (common_shares : ℕ)
  (par_value : ℕ)
  (semi_annual_dividend_common : ℚ)
  (total_annual_dividend : ℚ)
  (dividend_rate_preferred : ℚ) : Prop :=
  preferred_shares * par_value * (dividend_rate_preferred / 100) +
  2 * (common_shares * par_value * (semi_annual_dividend_common / 100)) =
  total_annual_dividend

theorem preferred_dividend_rate
  (h1 : 1200 = 1200)
  (h2 : 3000 = 3000)
  (h3 : 50 = 50)
  (h4 : 3.5 = 3.5)
  (h5 : 16500 = 16500) :
  dividend_rate_on_preferred_shares 1200 3000 50 3.5 16500 10 :=
by sorry

end preferred_dividend_rate_l105_105763


namespace william_marbles_l105_105613

theorem william_marbles :
  let initial_marbles := 10
  let shared_marbles := 3
  (initial_marbles - shared_marbles) = 7 := 
by
  sorry

end william_marbles_l105_105613


namespace sum_a5_a8_eq_six_l105_105745

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) := ∀ {m n : ℕ}, a (m + 1) / a m = a (n + 1) / a n

theorem sum_a5_a8_eq_six (h_seq : geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_condition : a 6 * a 4 + 2 * a 8 * a 5 + a 9 * a 7 = 36) :
  a 5 + a 8 = 6 := 
sorry

end sum_a5_a8_eq_six_l105_105745


namespace remainder_division_l105_105313

theorem remainder_division : ∃ (r : ℕ), 271 = 30 * 9 + r ∧ r = 1 :=
by
  -- Details of the proof would be filled here
  sorry

end remainder_division_l105_105313


namespace two_digit_number_possible_options_l105_105434

theorem two_digit_number_possible_options
  (N : ℕ)
  (h1 : 10 ≤ N ∧ N < 100)
  (h2 : (N % 3 = 0 ∨ N % 3 ≠ 0) ∧
        (N % 4 = 0 ∨ N % 4 ≠ 0) ∧
        (N % 5 = 0 ∨ N % 5 ≠ 0) ∧
        (N % 9 = 0 ∨ N % 9 ≠ 0) ∧
        (N % 10 = 0 ∨ N % 10 ≠ 0) ∧
        (N % 15 = 0 ∨ N % 15 ≠ 0) ∧
        (N % 18 = 0 ∨ N % 18 ≠ 0) ∧
        (N % 30 = 0 ∨ N % 30 ≠ 0)) :
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end two_digit_number_possible_options_l105_105434


namespace triangle_perimeter_l105_105328

theorem triangle_perimeter (A r p : ℝ) (hA : A = 75) (hr : r = 2.5) :
  A = r * (p / 2) → p = 60 := by
  intros
  sorry

end triangle_perimeter_l105_105328


namespace find_second_sum_l105_105980

theorem find_second_sum (S : ℤ) (x : ℤ) (h_S : S = 2678)
  (h_eq_interest : x * 3 * 8 = (S - x) * 5 * 3) : (S - x) = 1648 :=
by {
  sorry
}

end find_second_sum_l105_105980


namespace shortest_distance_to_left_focus_l105_105823

def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

def left_focus : ℝ × ℝ := (-5, 0)

theorem shortest_distance_to_left_focus : 
  ∃ P : ℝ × ℝ, 
  hyperbola P.1 P.2 ∧ 
  (∀ Q : ℝ × ℝ, hyperbola Q.1 Q.2 → dist Q left_focus ≥ dist P left_focus) ∧ 
  dist P left_focus = 2 :=
sorry

end shortest_distance_to_left_focus_l105_105823


namespace children_marbles_problem_l105_105696

theorem children_marbles_problem (n x N : ℕ) 
  (h1 : N = n * x)
  (h2 : 1 + (N - 1) / 10 = x) :
  n = 9 ∧ x = 9 :=
by
  sorry

end children_marbles_problem_l105_105696


namespace arithmetic_progression_K_l105_105578

theorem arithmetic_progression_K (K : ℕ) : 
  (∃ n : ℕ, K = 30 * n - 1) ↔ (K^K + 1) % 30 = 0 :=
sorry

end arithmetic_progression_K_l105_105578


namespace inequality_proof_l105_105565

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (a - b + c) * (1 / a - 1 / b + 1 / c) ≥ 1 :=
by
  sorry

end inequality_proof_l105_105565


namespace cinnamon_swirl_eaters_l105_105522

theorem cinnamon_swirl_eaters (total_pieces : ℝ) (jane_pieces : ℝ) (equal_pieces : total_pieces / jane_pieces = 3 ) : 
  (total_pieces = 12) ∧ (jane_pieces = 4) → total_pieces / jane_pieces = 3 := 
by 
  sorry

end cinnamon_swirl_eaters_l105_105522


namespace employees_six_years_or_more_percentage_l105_105867

theorem employees_six_years_or_more_percentage 
  (Y : ℕ)
  (Total : ℝ := (3 * Y:ℝ) + (4 * Y:ℝ) + (7 * Y:ℝ) - (2 * Y:ℝ) + (6 * Y:ℝ) + (1 * Y:ℝ))
  (Employees_Six_Years : ℝ := (6 * Y:ℝ) + (1 * Y:ℝ))
  : Employees_Six_Years / Total * 100 = 36.84 :=
by
  sorry

end employees_six_years_or_more_percentage_l105_105867


namespace geometric_series_sum_example_l105_105507

-- Define the finite geometric series
def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- State the theorem
theorem geometric_series_sum_example :
  geometric_series_sum (1/2) (1/2) 8 = 255 / 256 :=
by
  sorry

end geometric_series_sum_example_l105_105507


namespace seq_formula_l105_105901

noncomputable def seq {a : Nat → ℝ} (h1 : a 2 - a 1 = 1) (h2 : ∀ n, a (n + 1) - a n = 2 * (n - 1) + 1) : Nat → ℝ :=
sorry

theorem seq_formula {a : Nat → ℝ} 
  (h1 : a 2 - a 1 = 1)
  (h2 : ∀ n, a (n + 1) - a n = 2 * (n - 1) + 1)
  (n : Nat) : a n = 2 ^ n - 1 :=
sorry

end seq_formula_l105_105901


namespace infinite_either_interval_exists_rational_infinite_elements_l105_105698

variable {ε : ℝ} (x : ℕ → ℝ) (hε : ε > 0) (hεlt : ε < 1/2)

-- Problem 1
theorem infinite_either_interval (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) :
  (∃ N : ℕ, ∀ n ≥ N, x n < 1/2) ∨ (∃ N : ℕ, ∀ n ≥ N, x n ≥ 1/2) :=
sorry

-- Problem 2
theorem exists_rational_infinite_elements (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) (hε : ε > 0) (hεlt : ε < 1/2) :
  ∃ (α : ℚ), 0 ≤ α ∧ α ≤ 1 ∧ ∃ N : ℕ, ∀ n ≥ N, x n ∈ [α - ε, α + ε] :=
sorry

end infinite_either_interval_exists_rational_infinite_elements_l105_105698


namespace cost_of_bananas_and_cantaloupe_l105_105257

variable (a b c d : ℝ)

theorem cost_of_bananas_and_cantaloupe :
  (a + b + c + d = 30) →
  (d = 3 * a) →
  (c = a - b) →
  (b + c = 6) :=
by
  intros h1 h2 h3
  sorry

end cost_of_bananas_and_cantaloupe_l105_105257


namespace work_together_time_l105_105979

theorem work_together_time :
  let man_work_rate := 1/20
  let father_work_rate := 1/20
  let son_work_rate := 1/25
  let combined_work_rate := man_work_rate + father_work_rate + son_work_rate
  let days_to_complete := 1 / combined_work_rate 
  (days_to_complete - 100 / 14).abs < 0.01 :=
by
  -- Definitions of individual work rates
  let man_work_rate : ℚ := 1 / 20
  let father_work_rate : ℚ := 1 / 20
  let son_work_rate : ℚ := 1 / 25
  
  -- Combined work rate calculation
  let combined_work_rate := man_work_rate + father_work_rate + son_work_rate
  let combined_work_rate_simplified : ℚ := (1 / 20) + (1 / 20) + (1 / 25)
  
  -- Days to complete the job calculation
  let days_to_complete := 1 / combined_work_rate
  
  -- Compared with the expected value
  have h : (days_to_complete - 100 / 14).abs < 0.01 := 
    by sorry
  exact h

end work_together_time_l105_105979


namespace total_fish_caught_l105_105255

-- Definitions based on conditions
def sums : List ℕ := [7, 9, 14, 14, 19, 21]

-- Statement of the proof problem
theorem total_fish_caught : 
  (∃ (a b c d : ℕ), [a+b, a+c, a+d, b+c, b+d, c+d] = sums) → 
  ∃ (a b c d : ℕ), a + b + c + d = 28 :=
by 
  sorry

end total_fish_caught_l105_105255


namespace smallest_positive_multiple_l105_105521

theorem smallest_positive_multiple (a : ℕ) (h₁ : a % 6 = 0) (h₂ : a % 15 = 0) : a = 30 :=
sorry

end smallest_positive_multiple_l105_105521


namespace max_discount_rate_l105_105673

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l105_105673


namespace matrix_power_A_2023_l105_105755

open Matrix

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![0, -1, 0],
    ![1, 0, 0],
    ![0, 0, 1]
  ]

theorem matrix_power_A_2023 :
  A ^ 2023 = ![
    ![0, 1, 0],
    ![-1, 0, 0],
    ![0, 0, 1]
  ] :=
sorry

end matrix_power_A_2023_l105_105755


namespace max_discount_rate_l105_105676

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l105_105676


namespace count_three_digit_integers_divisible_by_11_and_5_l105_105000

def count_three_digit_multiples (a b: ℕ) : ℕ :=
  let lcm := Nat.lcm a b
  let first_multiple := (100 + lcm - 1) / lcm
  let last_multiple := 999 / lcm
  last_multiple - first_multiple + 1

theorem count_three_digit_integers_divisible_by_11_and_5 : 
  count_three_digit_multiples 11 5 = 17 := by 
  sorry

end count_three_digit_integers_divisible_by_11_and_5_l105_105000


namespace multiply_123_32_125_l105_105236

theorem multiply_123_32_125 : 123 * 32 * 125 = 492000 := by
  sorry

end multiply_123_32_125_l105_105236


namespace radius_of_arch_bridge_l105_105903

theorem radius_of_arch_bridge :
  ∀ (AB CD AD r : ℝ),
    AB = 12 →
    CD = 4 →
    AD = AB / 2 →
    r^2 = AD^2 + (r - CD)^2 →
    r = 6.5 :=
by
  intros AB CD AD r hAB hCD hAD h_eq
  sorry

end radius_of_arch_bridge_l105_105903


namespace max_ab_is_nine_l105_105401

noncomputable def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- If a > 0, b > 0, and the function f(x) = 4x^3 - ax^2 - 2bx + 2 has an extremum at x = 1, then the maximum value of ab is 9. -/
theorem max_ab_is_nine {a b : ℝ}
  (ha : a > 0) (hb : b > 0)
  (extremum_x1 : deriv (f a b) 1 = 0) :
  a * b ≤ 9 :=
sorry

end max_ab_is_nine_l105_105401


namespace positive_integer_sum_l105_105250

theorem positive_integer_sum (n : ℤ) :
  (n > 0) ∧
  (∀ stamps_cannot_form : ℤ, (∀ a b c : ℤ, 7 * a + n * b + (n + 2) * c ≠ 120) ↔
  ¬ (0 ≤ 7*a ∧ 7*a ≤ 120 ∧ 0 ≤ n*b ∧ n*b ≤ 120 ∧ 0 ≤ (n + 2)*c ∧ (n + 2)*c ≤ 120)) ∧
  (∀ postage_formed : ℤ, (120 < postage_formed ∧ postage_formed ≤ 125 →
  ∃ a b c : ℤ, 7 * a + n * b + (n + 2) * c = postage_formed)) →
  n = 21 :=
by {
  -- proof omittted 
  sorry
}

end positive_integer_sum_l105_105250


namespace tangents_and_fraction_l105_105003

theorem tangents_and_fraction
  (α β : ℝ)
  (tan_diff : Real.tan (α - β) = 2)
  (tan_beta : Real.tan β = 4) :
  (7 * Real.sin α - Real.cos α) / (7 * Real.sin α + Real.cos α) = 7 / 5 :=
sorry

end tangents_and_fraction_l105_105003


namespace welders_started_on_other_project_l105_105226

theorem welders_started_on_other_project
  (r : ℝ) (x : ℝ) (W : ℝ)
  (h1 : 16 * r * 8 = W)
  (h2 : (16 - x) * r * 24 = W - 16 * r) :
  x = 11 :=
by
  sorry

end welders_started_on_other_project_l105_105226


namespace cape_may_shark_sightings_l105_105238

def total_shark_sightings (D C : ℕ) : Prop :=
  D + C = 40

def cape_may_sightings (D C : ℕ) : Prop :=
  C = 2 * D - 8

theorem cape_may_shark_sightings : 
  ∃ (C D : ℕ), total_shark_sightings D C ∧ cape_may_sightings D C ∧ C = 24 :=
by
  sorry

end cape_may_shark_sightings_l105_105238


namespace kris_suspension_days_per_instance_is_three_l105_105302

-- Define the basic parameters given in the conditions
def total_fingers_toes : ℕ := 20
def total_bullying_instances : ℕ := 20
def multiplier : ℕ := 3

-- Define total suspension days according to the conditions
def total_suspension_days : ℕ := multiplier * total_fingers_toes

-- Define the goal: to find the number of suspension days per instance
def suspension_days_per_instance : ℕ := total_suspension_days / total_bullying_instances

-- The theorem to prove that Kris was suspended for 3 days per instance
theorem kris_suspension_days_per_instance_is_three : suspension_days_per_instance = 3 := by
  -- Skip the actual proof, focus only on the statement
  sorry

end kris_suspension_days_per_instance_is_three_l105_105302


namespace range_of_k_l105_105284

noncomputable def h (x : ℝ) (k : ℝ) : ℝ := 2 * x - k / x + k / 3

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → 2 + k / x^2 > 0) ↔ k ≥ -2 :=
by
  sorry

end range_of_k_l105_105284


namespace multiply_or_divide_inequality_by_negative_number_l105_105610

theorem multiply_or_divide_inequality_by_negative_number {a b c : ℝ} (h : a < b) (hc : c < 0) :
  c * a > c * b ∧ a / c > b / c :=
sorry

end multiply_or_divide_inequality_by_negative_number_l105_105610


namespace time_spent_per_piece_l105_105317

-- Conditions
def number_of_chairs : ℕ := 7
def number_of_tables : ℕ := 3
def total_furniture : ℕ := number_of_chairs + number_of_tables
def total_time_spent : ℕ := 40

-- Proof statement
theorem time_spent_per_piece : total_time_spent / total_furniture = 4 :=
by
  -- Proof goes here
  sorry

end time_spent_per_piece_l105_105317


namespace digits_difference_l105_105774

theorem digits_difference (d A B : ℕ) (h1 : d > 6) (h2 : (B + A) * d + 2 * A = d^2 + 7 * d + 2)
  (h3 : B + A = 10) (h4 : 2 * A = 8) : A - B = 3 :=
by 
  sorry

end digits_difference_l105_105774


namespace quadratic_vertex_ordinate_l105_105889

theorem quadratic_vertex_ordinate :
  let a := 2
  let b := -4
  let c := -1
  let vertex_x := -b / (2 * a)
  let vertex_y := a * vertex_x ^ 2 + b * vertex_x + c
  vertex_y = -3 :=
by
  sorry

end quadratic_vertex_ordinate_l105_105889


namespace transform_expression_l105_105540

theorem transform_expression (y Q : ℝ) (h : 5 * (3 * y + 7 * Real.pi) = Q) : 
  10 * (6 * y + 14 * Real.pi + 3) = 4 * Q + 30 := 
by 
  sorry

end transform_expression_l105_105540


namespace angle_between_clock_hands_at_3_05_l105_105213

theorem angle_between_clock_hands_at_3_05 :
  let minute_angle := 5 * 6
  let hour_angle := (5 / 60) * 30
  let initial_angle := 3 * 30
  initial_angle - minute_angle + hour_angle = 62.5 := by
  sorry

end angle_between_clock_hands_at_3_05_l105_105213


namespace find_third_root_l105_105111

-- Define the polynomial
def poly (a b x : ℚ) : ℚ := a * x^3 + 2 * (a + b) * x^2 + (b - 2 * a) * x + (10 - a)

-- Define the roots condition
def is_root (a b x : ℚ) : Prop := poly a b x = 0

-- Given conditions and required proof
theorem find_third_root (a b : ℚ) (ha : a = 350 / 13) (hb : b = -1180 / 13) :
  is_root a b (-1) ∧ is_root a b 4 → 
  ∃ r : ℚ, is_root a b r ∧ r ≠ -1 ∧ r ≠ 4 ∧ r = 61 / 35 :=
by sorry

end find_third_root_l105_105111


namespace remainder_eq_52_l105_105519

noncomputable def polynomial : Polynomial ℤ := Polynomial.C 1 * Polynomial.X ^ 4 + Polynomial.C (-4) * Polynomial.X ^ 2 + Polynomial.C 7

theorem remainder_eq_52 : Polynomial.eval (-3) polynomial = 52 :=
by
    sorry

end remainder_eq_52_l105_105519


namespace opposite_of_neg_two_l105_105794

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l105_105794


namespace part_one_part_two_l105_105137

noncomputable def f (x a : ℝ) : ℝ :=
  Real.log (1 + x) + a * Real.cos x

noncomputable def g (x : ℝ) : ℝ :=
  f x 2 - 1 / (1 + x)

theorem part_one (a : ℝ) : 
  (∀ x, f x a = Real.log (1 + x) + a * Real.cos x) ∧ 
  f 0 a = 2 ∧ 
  (∀ x, x + f (0:ℝ) a = x + 2) → 
  a = 2 := 
sorry

theorem part_two : 
  (∀ x, g x = Real.log (1 + x) + 2 * Real.cos x - 1 / (1 + x)) →
  (∃ y, -1 < y ∧ y < (Real.pi / 2) ∧ g y = 0) ∧ 
  (∀ x, -1 < x ∧ x < (Real.pi / 2) → g x ≠ 0) →
  (∃! y, -1 < y ∧ y < (Real.pi / 2) ∧ g y = 0) :=
sorry

end part_one_part_two_l105_105137


namespace answered_both_l105_105909

variables (A B : Type)
variables {test_takers : Type}

-- Defining the conditions
def pa : ℝ := 0.80  -- 80% answered first question correctly
def pb : ℝ := 0.75  -- 75% answered second question correctly
def pnone : ℝ := 0.05 -- 5% answered neither question correctly

-- Formal problem statement
theorem answered_both (test_takers: Type) : 
  (pa + pb - (1 - pnone)) = 0.60 :=
by
  sorry

end answered_both_l105_105909


namespace product_of_odd_implies_sum_is_odd_l105_105315

theorem product_of_odd_implies_sum_is_odd (a b c : ℤ) (h : a * b * c % 2 = 1) : (a + b + c) % 2 = 1 :=
sorry

end product_of_odd_implies_sum_is_odd_l105_105315


namespace no_positive_divisor_of_2n2_square_l105_105304

theorem no_positive_divisor_of_2n2_square (n : ℕ) (hn : n > 0) : 
  ∀ d : ℕ, d > 0 → d ∣ 2 * n ^ 2 → ¬∃ x : ℕ, x ^ 2 = d ^ 2 * n ^ 2 + d ^ 3 := 
by
  sorry

end no_positive_divisor_of_2n2_square_l105_105304


namespace min_value_of_squares_l105_105542

theorem min_value_of_squares (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 := 
by
  -- Proof omitted
  sorry

end min_value_of_squares_l105_105542


namespace find_fruit_cost_l105_105391

-- Define the conditions
def muffin_cost : ℝ := 2
def francis_muffin_count : ℕ := 2
def francis_fruit_count : ℕ := 2
def kiera_muffin_count : ℕ := 2
def kiera_fruit_count : ℕ := 1
def total_cost : ℝ := 17

-- Define the cost of each fruit cup
variable (F : ℝ)

-- The statement to be proved
theorem find_fruit_cost (h : francis_muffin_count * muffin_cost 
                + francis_fruit_count * F 
                + kiera_muffin_count * muffin_cost 
                + kiera_fruit_count * F = total_cost) : 
                F = 1.80 :=
by {
  sorry
}

end find_fruit_cost_l105_105391


namespace problem_real_numbers_l105_105402

theorem problem_real_numbers (a b c d r : ℝ) 
  (h1 : b + c + d = r * a) 
  (h2 : a + c + d = r * b) 
  (h3 : a + b + d = r * c) 
  (h4 : a + b + c = r * d) : 
  r = 3 ∨ r = -1 :=
sorry

end problem_real_numbers_l105_105402


namespace albert_needs_more_money_l105_105686

def cost_of_paintbrush : ℝ := 1.50
def cost_of_paints : ℝ := 4.35
def cost_of_easel : ℝ := 12.65
def amount_already_has : ℝ := 6.50

theorem albert_needs_more_money : 
  (cost_of_paintbrush + cost_of_paints + cost_of_easel) - amount_already_has = 12.00 := 
by
  sorry

end albert_needs_more_money_l105_105686


namespace correct_letter_is_P_l105_105158

variable (x : ℤ)

-- Conditions
def date_behind_C := x
def date_behind_A := x + 2
def date_behind_B := x + 11
def date_behind_P := x + 13 -- Based on problem setup
def date_behind_Q := x + 14 -- Continuous sequence assumption
def date_behind_R := x + 15 -- Continuous sequence assumption
def date_behind_S := x + 16 -- Continuous sequence assumption

-- Proof statement
theorem correct_letter_is_P :
  ∃ y, (y = date_behind_P ∧ x + y = date_behind_A + date_behind_B) := by
  sorry

end correct_letter_is_P_l105_105158


namespace shift_graph_sin_cos_l105_105338

open Real

theorem shift_graph_sin_cos :
  ∀ x : ℝ, sin (2 * x + π / 3) = cos (2 * (x + π / 12) - π / 3) :=
by
  sorry

end shift_graph_sin_cos_l105_105338


namespace find_shorter_piece_length_l105_105490

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x = 8

theorem find_shorter_piece_length : ∃ x : ℕ, (20 - x) > 0 ∧ 2 * x = (20 - x) + 4 ∧ shorter_piece_length x :=
by
  -- There exists an x that satisfies the conditions
  use 8
  -- Prove the conditions are met
  sorry

end find_shorter_piece_length_l105_105490


namespace additional_savings_is_300_l105_105229

-- Define constants
def price_per_window : ℕ := 120
def discount_threshold : ℕ := 10
def discount_per_window : ℕ := 10
def free_window_threshold : ℕ := 5

-- Define the number of windows Alice needs
def alice_windows : ℕ := 9

-- Define the number of windows Bob needs
def bob_windows : ℕ := 12

-- Define the function to calculate total cost without discount
def cost_without_discount (n : ℕ) : ℕ := n * price_per_window

-- Define the function to calculate cost with discount
def cost_with_discount (n : ℕ) : ℕ :=
  let full_windows := n - n / free_window_threshold
  let discounted_price := if n > discount_threshold then price_per_window - discount_per_window else price_per_window
  full_windows * discounted_price

-- Define the function to calculate savings when windows are bought separately
def savings_separately : ℕ :=
  (cost_without_discount alice_windows + cost_without_discount bob_windows) 
  - (cost_with_discount alice_windows + cost_with_discount bob_windows)

-- Define the function to calculate savings when windows are bought together
def savings_together : ℕ :=
  let combined_windows := alice_windows + bob_windows
  cost_without_discount combined_windows - cost_with_discount combined_windows

-- Prove that the additional savings when buying together is $300
theorem additional_savings_is_300 : savings_together - savings_separately = 300 := by
  -- missing proof
  sorry

end additional_savings_is_300_l105_105229


namespace circle_radius_zero_l105_105517

theorem circle_radius_zero : ∀ (x y : ℝ), x^2 + 10 * x + y^2 - 4 * y + 29 = 0 → 0 = 0 :=
by intro x y h
   sorry

end circle_radius_zero_l105_105517


namespace seashells_total_l105_105039

theorem seashells_total (s m : Nat) (hs : s = 18) (hm : m = 47) : s + m = 65 := 
by
  -- We are just specifying the theorem statement here
  sorry

end seashells_total_l105_105039


namespace ratio_of_distances_l105_105480

theorem ratio_of_distances
  (w x y : ℝ)
  (hw : w > 0)
  (hx : x > 0)
  (hy : y > 0)
  (h_eq_time : y / w = x / w + (x + y) / (5 * w)) :
  x / y = 2 / 3 :=
by
  sorry

end ratio_of_distances_l105_105480


namespace license_plates_count_l105_105145

theorem license_plates_count :
  (20 * 6 * 20 * 10 * 26 = 624000) :=
by
  sorry

end license_plates_count_l105_105145


namespace cleaner_for_dog_stain_l105_105311

theorem cleaner_for_dog_stain (D : ℝ) (H : 6 * D + 3 * 4 + 1 * 1 = 49) : D = 6 :=
by 
  -- Proof steps would go here, but we are skipping the proof.
  sorry

end cleaner_for_dog_stain_l105_105311


namespace expression_equals_384_l105_105355

noncomputable def problem_expression : ℤ :=
  2021^4 - 4 * 2023^4 + 6 * 2025^4 - 4 * 2027^4 + 2029^4

theorem expression_equals_384 : problem_expression = 384 := by
  sorry

end expression_equals_384_l105_105355


namespace total_value_correct_l105_105833

-- Define conditions
def import_tax_rate : ℝ := 0.07
def tax_paid : ℝ := 109.90
def tax_exempt_value : ℝ := 1000

-- Define total value
def total_value (V : ℝ) : Prop :=
  V - tax_exempt_value = tax_paid / import_tax_rate

-- Theorem stating that the total value is $2570
theorem total_value_correct : total_value 2570 := by
  sorry

end total_value_correct_l105_105833


namespace zachary_needs_more_money_l105_105709

def cost_of_football : ℝ := 3.75
def cost_of_shorts : ℝ := 2.40
def cost_of_shoes : ℝ := 11.85
def zachary_money : ℝ := 10.00
def total_cost : ℝ := cost_of_football + cost_of_shorts + cost_of_shoes
def amount_needed : ℝ := total_cost - zachary_money

theorem zachary_needs_more_money : amount_needed = 7.00 := by
  sorry

end zachary_needs_more_money_l105_105709


namespace saleswoman_commission_l105_105388

theorem saleswoman_commission (S : ℝ)
  (h1 : (S > 500) )
  (h2 : (0.20 * 500 + 0.50 * (S - 500)) = 0.3125 * S) : 
  S = 800 :=
sorry

end saleswoman_commission_l105_105388


namespace Lisa_pay_per_hour_is_15_l105_105144

-- Given conditions:
def Greta_hours : ℕ := 40
def Greta_pay_per_hour : ℕ := 12
def Lisa_hours : ℕ := 32

-- Define Greta's earnings based on the given conditions:
def Greta_earnings : ℕ := Greta_hours * Greta_pay_per_hour

-- The main statement to prove:
theorem Lisa_pay_per_hour_is_15 (h1 : Greta_earnings = Greta_hours * Greta_pay_per_hour) 
                                (h2 : Greta_earnings = Lisa_hours * L) :
  L = 15 :=
by sorry

end Lisa_pay_per_hour_is_15_l105_105144


namespace woodworker_tables_l105_105370

theorem woodworker_tables (L C_leg C T_leg : ℕ) (hL : L = 40) (hC_leg : C_leg = 4) (hC : C = 6) (hT_leg : T_leg = 4) :
  T = (L - C * C_leg) / T_leg := by
  sorry

end woodworker_tables_l105_105370


namespace find_m_l105_105898

variable (m : ℝ)

def vector_oa : ℝ × ℝ := (-1, 2)
def vector_ob : ℝ × ℝ := (3, m)

def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem find_m
  (h : orthogonal (vector_oa) (vector_ob m)) :
  m = 3 / 2 := by
  sorry

end find_m_l105_105898


namespace sum_of_integers_l105_105002

theorem sum_of_integers (a b : ℤ) (h : (Int.sqrt (a - 2023) + |b + 2023| = 1)) : a + b = 1 ∨ a + b = -1 :=
by
  sorry

end sum_of_integers_l105_105002


namespace find_remainder_q_neg2_l105_105344

-- Define q(x)
def q (x : ℝ) (D E F : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 6

-- The given conditions in the problem
variable {D E F : ℝ}
variable (h_q_2 : q 2 D E F = 14)

-- The statement we aim to prove
theorem find_remainder_q_neg2 (h_q_2 : q 2 D E F = 14) : q (-2) D E F = 14 :=
sorry

end find_remainder_q_neg2_l105_105344


namespace find_direction_vector_l105_105363

def line_parametrization (v d : ℝ × ℝ) (t x y : ℝ) : ℝ × ℝ :=
  (v.fst + t * d.fst, v.snd + t * d.snd)

theorem find_direction_vector : 
  ∀ d: ℝ × ℝ, ∀ t: ℝ,
    ∀ (v : ℝ × ℝ) (x y : ℝ), 
    v = (-3, -1) → 
    y = (2 * x + 3) / 5 →
    x + 3 ≤ 0 →
    dist (line_parametrization v d t x y) (-3, -1) = t →
    d = (5/2, 1) :=
by
  intros d t v x y hv hy hcond hdist
  sorry

end find_direction_vector_l105_105363


namespace max_discount_rate_l105_105674

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l105_105674


namespace percentage_equivalence_l105_105478

theorem percentage_equivalence (x : ℝ) : 0.3 * 0.6 * 0.7 * x = 0.126 * x :=
by
  sorry

end percentage_equivalence_l105_105478


namespace total_miles_ran_l105_105573

theorem total_miles_ran (miles_monday miles_wednesday miles_friday : ℕ)
  (h1 : miles_monday = 3)
  (h2 : miles_wednesday = 2)
  (h3 : miles_friday = 7) :
  miles_monday + miles_wednesday + miles_friday = 12 := 
by
  sorry

end total_miles_ran_l105_105573


namespace solve_inequality_range_of_a_l105_105724

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Define the set A
def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 2 }

-- First part: Solve the inequality f(x) ≤ 3a^2 + 1 when a ≠ 0
-- Solution would be translated in a theorem
theorem solve_inequality (a : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, f x a ≤ 3 * a^2 + 1 → if a > 0 then -a ≤ x ∧ x ≤ 3 * a else -3 * a ≤ x ∧ x ≤ a :=
sorry

-- Second part: Find the range of a if there exists no x0 ∈ A such that f(x0) ≤ A is false
theorem range_of_a (a : ℝ) :
  (∀ x ∈ A, f x a > 0) ↔ a < 1 :=
sorry

end solve_inequality_range_of_a_l105_105724


namespace max_discount_rate_l105_105658

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l105_105658


namespace find_number_l105_105829

theorem find_number (x : ℝ) (h : 0.62 * x - 50 = 43) : x = 150 :=
sorry

end find_number_l105_105829


namespace symmetric_point_is_correct_l105_105296

/-- A point in 2D Cartesian coordinates -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Defining the point P with given coordinates -/
def P : Point := {x := 2, y := 3}

/-- Defining the symmetry of a point with respect to the origin -/
def symmetric_origin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- States that the symmetric point of P (2, 3) with respect to the origin is (-2, -3) -/
theorem symmetric_point_is_correct :
  symmetric_origin P = {x := -2, y := -3} :=
by
  sorry

end symmetric_point_is_correct_l105_105296


namespace sequence_missing_number_l105_105082

theorem sequence_missing_number : 
  ∃ x, (x - 21 = 7 ∧ 37 - x = 9) ∧ x = 28 := by
  sorry

end sequence_missing_number_l105_105082


namespace zero_function_solution_l105_105702

theorem zero_function_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^3 + y^3) = f (x^3) + 3 * x^2 * f (x) * f (y) + 3 * (f (x) * f (y))^2 + y^6 * f (y)) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end zero_function_solution_l105_105702


namespace max_discount_rate_l105_105664

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l105_105664


namespace relationship_between_a_and_b_l105_105895

theorem relationship_between_a_and_b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b)
    (h₃ : ∀ x : ℝ, |(3 * x + 1) - 4| < a → |x - 1| < b) : a ≥ 3 * b :=
by
  -- Applying the given conditions, we want to demonstrate that a ≥ 3b.
  sorry

end relationship_between_a_and_b_l105_105895


namespace number_of_students_with_at_least_two_pets_l105_105744

-- Definitions for the sets of students
def total_students := 50
def dog_students := 35
def cat_students := 40
def rabbit_students := 10
def dog_and_cat_students := 20
def dog_and_rabbit_students := 5
def cat_and_rabbit_students := 0  -- Assuming minimal overlap

-- Problem Statement
theorem number_of_students_with_at_least_two_pets :
  (dog_and_cat_students + dog_and_rabbit_students + cat_and_rabbit_students) = 25 :=
by
  sorry

end number_of_students_with_at_least_two_pets_l105_105744


namespace smallest_positive_m_l105_105830

theorem smallest_positive_m {m p q : ℤ} (h_eq : 12 * p^2 - m * p - 360 = 0) (h_pq : p * q = -30) :
  (m = 12 * (p + q)) → 0 < m → m = 12 :=
by
  sorry

end smallest_positive_m_l105_105830


namespace geometric_sequence_min_value_l105_105260

theorem geometric_sequence_min_value
  (a : ℕ → ℝ)
  (h1 : ∀ n, 0 < a n) 
  (h2 : a 9 = 9 * a 7)
  (exists_m_n : ∃ m n, a m * a n = 9 * (a 1)^2):
  ∀ m n, (m + n = 4) → (1 / m + 9 / n) ≥ 4 :=
by
  intros m n h
  sorry

end geometric_sequence_min_value_l105_105260


namespace slope_of_asymptotes_l105_105124

noncomputable def hyperbola_asymptote_slope (x y : ℝ) : Prop :=
  (x^2 / 144 - y^2 / 81 = 1)

theorem slope_of_asymptotes (x y : ℝ) (h : hyperbola_asymptote_slope x y) :
  ∃ m : ℝ, m = 3 / 4 ∨ m = -3 / 4 :=
sorry

end slope_of_asymptotes_l105_105124


namespace evaluate_dollar_op_l105_105382

def dollar_op (x y : ℤ) := x * (y + 2) + 2 * x * y

theorem evaluate_dollar_op : dollar_op 4 (-1) = -4 :=
by
  -- Proof steps here
  sorry

end evaluate_dollar_op_l105_105382


namespace Talia_total_distance_l105_105776

variable (Talia : Type)
variable (house park store : Talia)

-- Define the distances given in the conditions
variable (distance : Talia → Talia → ℕ)
variable (h2p : distance house park = 5)
variable (p2s : distance park store = 3)
variable (s2h : distance store house = 8)

-- Define the total distance function
def total_distance (t : Talia) : ℕ :=
  distance house park + distance park store + distance store house

-- Lean 4 theorem statement
theorem Talia_total_distance : total_distance Talia house park store distance = 16 :=
by
  simp [total_distance, h2p, p2s, s2h]
  sorry

end Talia_total_distance_l105_105776


namespace leaves_problem_l105_105919

noncomputable def leaves_dropped_last_day (L : ℕ) (n : ℕ) : ℕ :=
  L - n * (L / 10)

theorem leaves_problem (L : ℕ) (n : ℕ) (h1 : L = 340) (h2 : leaves_dropped_last_day L n = 204) :
  n = 4 :=
by {
  sorry
}

end leaves_problem_l105_105919


namespace arithmetic_sequence_product_l105_105566

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := sorry

theorem arithmetic_sequence_product (a_1 a_6 a_7 a_4 a_9 : ℝ) (d : ℝ) :
  a_1 = 2 →
  a_6 = a_1 + 5 * d →
  a_7 = a_1 + 6 * d →
  a_6 * a_7 = 15 →
  a_4 = a_1 + 3 * d →
  a_9 = a_1 + 8 * d →
  a_4 * a_9 = 234 / 25 :=
sorry

end arithmetic_sequence_product_l105_105566


namespace desired_average_sale_l105_105492

theorem desired_average_sale
  (sale1 sale2 sale3 sale4 sale5 sale6 : ℕ)
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h4 : sale4 = 7230)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 7991) :
  (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = 7000 :=
by
  sorry

end desired_average_sale_l105_105492


namespace sqrt_eighteen_simplifies_l105_105040

open Real

theorem sqrt_eighteen_simplifies :
  sqrt 18 = 3 * sqrt 2 :=
by
  sorry

end sqrt_eighteen_simplifies_l105_105040


namespace average_weight_of_all_children_l105_105324

theorem average_weight_of_all_children 
  (Boys: ℕ) (Girls: ℕ) (Additional: ℕ)
  (avgWeightBoys: ℚ) (avgWeightGirls: ℚ) (avgWeightAdditional: ℚ) :
  Boys = 8 ∧ Girls = 5 ∧ Additional = 3 ∧ 
  avgWeightBoys = 160 ∧ avgWeightGirls = 130 ∧ avgWeightAdditional = 145 →
  ((Boys * avgWeightBoys + Girls * avgWeightGirls + Additional * avgWeightAdditional) / (Boys + Girls + Additional) = 148) :=
by
  intros
  sorry

end average_weight_of_all_children_l105_105324


namespace slope_of_BC_l105_105261

theorem slope_of_BC
  (h₁ : ∀ x y : ℝ, (x^2 / 8) + (y^2 / 2) = 1)
  (h₂ : ∀ A : ℝ × ℝ, A = (2, 1))
  (h₃ : ∀ k₁ k₂ : ℝ, k₁ + k₂ = 0) :
  ∃ k : ℝ, k = 1 / 2 :=
by
  sorry

end slope_of_BC_l105_105261


namespace dogwood_trees_total_is_100_l105_105963

def initial_dogwood_trees : ℕ := 39
def trees_planted_today : ℕ := 41
def trees_planted_tomorrow : ℕ := 20
def total_dogwood_trees : ℕ := initial_dogwood_trees + trees_planted_today + trees_planted_tomorrow

theorem dogwood_trees_total_is_100 : total_dogwood_trees = 100 := by
  sorry  -- Proof goes here

end dogwood_trees_total_is_100_l105_105963


namespace intersection_distance_l105_105325

theorem intersection_distance (p q : ℕ) (h1 : p = 65) (h2 : q = 2) :
  p - q = 63 := 
by
  sorry

end intersection_distance_l105_105325


namespace sum_of_decimals_as_common_fraction_l105_105885

theorem sum_of_decimals_as_common_fraction:
  (2 / 10 : ℚ) + (3 / 100 : ℚ) + (4 / 1000 : ℚ) + (5 / 10000 : ℚ) + (6 / 100000 : ℚ) = 733 / 3125 :=
by
  -- Explanation and proof steps are omitted as directed.
  sorry

end sum_of_decimals_as_common_fraction_l105_105885


namespace opposite_of_neg_2_is_2_l105_105806

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l105_105806


namespace tiffany_math_homework_pages_l105_105469

def math_problems (m : ℕ) : ℕ := 3 * m
def reading_problems : ℕ := 4 * 3
def total_problems (m : ℕ) : ℕ := math_problems m + reading_problems

theorem tiffany_math_homework_pages (m : ℕ) (h : total_problems m = 30) : m = 6 :=
by
  sorry

end tiffany_math_homework_pages_l105_105469


namespace sum_of_squares_l105_105899

theorem sum_of_squares (a b c : ℝ) (h1 : ab + bc + ca = 4) (h2 : a + b + c = 17) : a^2 + b^2 + c^2 = 281 :=
by
  sorry

end sum_of_squares_l105_105899


namespace cost_of_eraser_is_1_l105_105369

variables (P : ℝ)  -- Price of a pencil
variables (num_pencils : ℝ) (num_erasers : ℝ) (total_revenue : ℝ)

-- Conditions
def condition1 := num_erasers = 2 * num_pencils
def condition2 := num_pencils = 20
def condition3 := total_revenue = 80
def condition4 := total_revenue = num_pencils * P + num_erasers * (1/2 * P)

-- Theorem: The cost of an eraser is 1 dollar
theorem cost_of_eraser_is_1 : 
    (P : ℝ) → (condition1) → (condition2) → (condition3) → (condition4) → (1/2 * P) = 1 :=
by
  sorry

end cost_of_eraser_is_1_l105_105369


namespace determinant_transformation_l105_105712

theorem determinant_transformation 
  (p q r s : ℝ)
  (h : Matrix.det ![![p, q], ![r, s]] = 6) :
  Matrix.det ![![p, 9 * p + 4 * q], ![r, 9 * r + 4 * s]] = 24 := 
sorry

end determinant_transformation_l105_105712


namespace actual_order_correct_l105_105386

-- Define the actual order of the students.
def actual_order := ["E", "D", "A", "C", "B"]

-- Define the first person's prediction and conditions.
def first_person_prediction := ["A", "B", "C", "D", "E"]
def first_person_conditions (pos1 pos2 pos3 pos4 pos5 : String) : Prop :=
  (pos1 ≠ "A") ∧ (pos2 ≠ "B") ∧ (pos3 ≠ "C") ∧ (pos4 ≠ "D") ∧ (pos5 ≠ "E") ∧
  (pos1 ≠ "B") ∧ (pos2 ≠ "A") ∧ (pos2 ≠ "C") ∧ (pos3 ≠ "B") ∧ (pos3 ≠ "D") ∧
  (pos4 ≠ "C") ∧ (pos4 ≠ "E") ∧ (pos5 ≠ "D")

-- Define the second person's prediction and conditions.
def second_person_prediction := ["D", "A", "E", "C", "B"]
def second_person_conditions (pos1 pos2 pos3 pos4 pos5 : String) : Prop :=
  ((pos1 = "D") ∨ (pos2 = "D") ∨ (pos3 = "D") ∨ (pos4 = "D") ∨ (pos5 = "D")) ∧
  ((pos1 = "A") ∨ (pos2 = "A") ∨ (pos3 = "A") ∨ (pos4 = "A") ∨ (pos5 = "A")) ∧
  (pos1 ≠ "D" ∨ pos2 ≠ "A") ∧ (pos2 ≠ "A" ∨ pos3 ≠ "E") ∧ (pos3 ≠ "E" ∨ pos4 ≠ "C") ∧ (pos4 ≠ "C" ∨ pos5 ≠ "B")

-- The theorem to prove the actual order.
theorem actual_order_correct :
  ∃ (pos1 pos2 pos3 pos4 pos5 : String),
    first_person_conditions pos1 pos2 pos3 pos4 pos5 ∧
    second_person_conditions pos1 pos2 pos3 pos4 pos5 ∧
    [pos1, pos2, pos3, pos4, pos5] = actual_order :=
by sorry

end actual_order_correct_l105_105386


namespace contrapositive_equivalent_l105_105203

variable {α : Type*} (A B : Set α) (x : α)

theorem contrapositive_equivalent : (x ∈ A → x ∈ B) ↔ (x ∉ B → x ∉ A) :=
by
  sorry

end contrapositive_equivalent_l105_105203


namespace pattern_E_cannot_be_formed_l105_105455

-- Define the basic properties of the tile and the patterns
inductive Tile
| rhombus (diag_coloring : Bool) -- representing black-and-white diagonals

inductive Pattern
| optionA
| optionB
| optionC
| optionD
| optionE

-- The given tile is a rhombus with a certain coloring scheme
def given_tile : Tile := Tile.rhombus true

-- The statement to prove
theorem pattern_E_cannot_be_formed : 
  ¬ (∃ f : Pattern → Tile, f Pattern.optionE = given_tile) :=
sorry

end pattern_E_cannot_be_formed_l105_105455


namespace correct_total_weight_6_moles_Al2_CO3_3_l105_105475

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

def num_atoms_Al : ℕ := 2
def num_atoms_C : ℕ := 3
def num_atoms_O : ℕ := 9

def molecular_weight_Al2_CO3_3 : ℝ :=
  (num_atoms_Al * atomic_weight_Al) +
  (num_atoms_C * atomic_weight_C) +
  (num_atoms_O * atomic_weight_O)

def num_moles : ℝ := 6

def total_weight_6_moles_Al2_CO3_3 : ℝ := num_moles * molecular_weight_Al2_CO3_3

theorem correct_total_weight_6_moles_Al2_CO3_3 :
  total_weight_6_moles_Al2_CO3_3 = 1403.94 :=
by
  unfold total_weight_6_moles_Al2_CO3_3
  unfold num_moles
  unfold molecular_weight_Al2_CO3_3
  unfold num_atoms_Al num_atoms_C num_atoms_O atomic_weight_Al atomic_weight_C atomic_weight_O
  sorry

end correct_total_weight_6_moles_Al2_CO3_3_l105_105475


namespace pair_d_not_meet_goal_l105_105449

-- Defining the marks Sophie has already obtained in her first three tests
def first_three_marks : List ℕ := [73, 82, 85]

-- Function to calculate sum of a list of integers
def sum_list (lst: List ℕ) : ℕ := lst.foldr (· + ·) 0

-- Defining Sophie's goal average and the number of total tests
def goal_average : ℕ := 80
def num_tests : ℕ := 5
def total_goal_sum : ℕ := num_tests * goal_average

-- Calculating the sum of marks already obtained
def sum_first_three := sum_list first_three_marks

-- Calculating the required sum for the remaining two tests
def required_sum : ℕ := total_goal_sum - sum_first_three

-- Defining the pairs of marks for the remaining tests
def pair_a : List ℕ := [79, 82]
def pair_b : List ℕ := [70, 91]
def pair_c : List ℕ := [76, 86]
def pair_d : List ℕ := [73, 83]

-- Creating a theorem to prove that pair_d does not meet the goal
theorem pair_d_not_meet_goal : sum_list pair_d < required_sum := by
  sorry

end pair_d_not_meet_goal_l105_105449


namespace days_y_worked_l105_105486

theorem days_y_worked 
  (W : ℝ) 
  (x_days : ℝ) (h1 : x_days = 36)
  (y_days : ℝ) (h2 : y_days = 24)
  (x_remaining_days : ℝ) (h3 : x_remaining_days = 18)
  (d : ℝ) :
  d * (W / y_days) + x_remaining_days * (W / x_days) = W → d = 12 :=
by
  -- Mathematical proof goes here
  sorry

end days_y_worked_l105_105486


namespace compare_game_A_and_C_l105_105088

-- Probability definitions for coin toss
def p_heads := 2/3
def p_tails := 1/3

-- Probability of winning Game A
def prob_win_A := (p_heads^3) + (p_tails^3)

-- Probability of winning Game C
def prob_win_C := (p_heads^3 + p_tails^3)^2

-- Theorem statement to compare chances of winning Game A to Game C
theorem compare_game_A_and_C : prob_win_A - prob_win_C = 2/9 := by sorry

end compare_game_A_and_C_l105_105088


namespace repeated_number_divisible_by_1001001_l105_105992

theorem repeated_number_divisible_by_1001001 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  (1000000 * (100 * a + 10 * b + c) + 1000 * (100 * a + 10 * b + c) + (100 * a + 10 * b + c)) % 1001001 = 0 := 
by 
  sorry

end repeated_number_divisible_by_1001001_l105_105992


namespace octal_to_decimal_equiv_l105_105500

-- Definitions for the octal number 724
def d0 := 4
def d1 := 2
def d2 := 7

-- Definition for the base
def base := 8

-- Calculation of the decimal equivalent
def calc_decimal : ℕ :=
  d0 * base^0 + d1 * base^1 + d2 * base^2

-- The proof statement
theorem octal_to_decimal_equiv : calc_decimal = 468 := by
  sorry

end octal_to_decimal_equiv_l105_105500


namespace opposite_of_neg_two_l105_105797

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l105_105797


namespace max_discount_rate_l105_105662

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l105_105662


namespace part1_part2_l105_105761

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

end part1_part2_l105_105761


namespace max_discount_rate_l105_105672

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l105_105672


namespace simplify_expression_l105_105925

variable (a b c d x : ℝ)
variable (hab : a ≠ b)
variable (hac : a ≠ c)
variable (had : a ≠ d)
variable (hbc : b ≠ c)
variable (hbd : b ≠ d)
variable (hcd : c ≠ d)

theorem simplify_expression :
  ( ( (x + a)^4 / ((a - b)*(a - c)*(a - d)) )
  + ( (x + b)^4 / ((b - a)*(b - c)*(b - d)) )
  + ( (x + c)^4 / ((c - a)*(c - b)*(c - d)) )
  + ( (x + d)^4 / ((d - a)*(d - b)*(d - c)) ) = a + b + c + d + 4*x ) :=
  sorry

end simplify_expression_l105_105925


namespace question_A_question_B_question_C_question_D_l105_105611

open Real
open Probability

theorem question_A : 
  let data := [1, 2, 4, 5, 6, 8, 9]
  let n := 4.2
  data.nth (5 - 1) = 6 := sorry

theorem question_B (X : ℕ → ℝ) (n : ℕ) (hX : X ∼ binomial n (1/3)) (hE : E (3*X + 1) = 6) : 
  n = 5 := sorry

theorem question_C (x y : ℝ) (b : ℝ) :
  let regression_eq := λ x : ℝ, b * x + 1.8
  average x = 2 →
  average y = 20 →
  b = 9.1 := sorry

theorem question_D (x y : ℝ) (chisq : ℝ) : 
  (chisq < some_threshold) → not (x and y are more related) := sorry

end question_A_question_B_question_C_question_D_l105_105611


namespace tan_alpha_one_value_l105_105734

theorem tan_alpha_one_value {α : ℝ} (h : Real.tan α = 1) :
    (2 * Real.sin α ^ 2 + 1) / Real.sin (2 * α) = 2 :=
by
  sorry

end tan_alpha_one_value_l105_105734


namespace proposition_contrapositive_same_truth_value_l105_105838

variable {P : Prop}

theorem proposition_contrapositive_same_truth_value (P : Prop) :
  (P → P) = (¬P → ¬P) := 
sorry

end proposition_contrapositive_same_truth_value_l105_105838


namespace maximum_discount_rate_l105_105640

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l105_105640


namespace ratio_of_B_to_C_l105_105586

-- Definitions based on conditions
def A := 40
def C := A + 20
def total := 220
def B := total - A - C

-- Theorem statement
theorem ratio_of_B_to_C : B / C = 2 :=
by
  -- Placeholder for proof
  sorry

end ratio_of_B_to_C_l105_105586


namespace total_wrappers_collected_l105_105232

theorem total_wrappers_collected :
  let Andy_wrappers := 34
  let Max_wrappers := 15
  let Zoe_wrappers := 25
  Andy_wrappers + Max_wrappers + Zoe_wrappers = 74 :=
by
  let Andy_wrappers := 34
  let Max_wrappers := 15
  let Zoe_wrappers := 25
  show Andy_wrappers + Max_wrappers + Zoe_wrappers = 74
  sorry

end total_wrappers_collected_l105_105232


namespace least_possible_value_of_m_plus_n_l105_105171

noncomputable def least_possible_sum (m n : ℕ) : ℕ :=
m + n

theorem least_possible_value_of_m_plus_n (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0)
  (h3 : Nat.gcd (m + n) 330 = 1)
  (h4 : m^m % n^n = 0)
  (h5 : m % n ≠ 0) : 
  least_possible_sum m n = 98 := 
sorry

end least_possible_value_of_m_plus_n_l105_105171


namespace find_line_m_l105_105245

noncomputable def reflect_point_across_line 
  (P : ℝ × ℝ) (a b c : ℝ) : ℝ × ℝ :=
  let line_vector := (a, b)
  let scaling_factor := -2 * ((a * P.1 + b * P.2 + c) / (a^2 + b^2))
  ((P.1 + scaling_factor * a), (P.2 + scaling_factor * b))

theorem find_line_m (P P'' : ℝ × ℝ) (a b : ℝ) (c : ℝ := 0)
  (h₁ : P = (2, -3))
  (h₂ : a * 1 + b * 4 = 0)
  (h₃ : P'' = (1, 4))
  (h₄ : reflect_point_across_line (reflect_point_across_line P a b c) a b c = P'') :
  4 * P''.1 - P''.2 = 0 :=
by
  sorry

end find_line_m_l105_105245


namespace maximum_discount_rate_l105_105642

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l105_105642


namespace cubic_roots_natural_numbers_l105_105121

theorem cubic_roots_natural_numbers (p : ℝ) :
  (∃ x1 x2 x3 : ℕ, (5 * (x1 : ℝ)^3 - 5 * (p + 1) * (x1 : ℝ)^2 + (71 * p - 1) * (x1 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x2 : ℝ)^3 - 5 * (p + 1) * (x2 : ℝ)^2 + (71 * p - 1) * (x2 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x3 : ℝ)^3 - 5 * (p + 1) * (x3 : ℝ)^2 + (71 * p - 1) * (x3 : ℝ) + 1 = 66 * p)) →
  p = 76 :=
sorry

end cubic_roots_natural_numbers_l105_105121


namespace friends_total_sales_l105_105836

theorem friends_total_sales :
  (Ryan Jason Zachary : ℕ) →
  (H1 : Ryan = Jason + 50) →
  (H2 : Jason = Zachary + (3 * Zachary / 10)) →
  (H3 : Zachary = 40 * 5) →
  Ryan + Jason + Zachary = 770 :=
by
  sorry

end friends_total_sales_l105_105836


namespace max_discount_rate_l105_105675

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l105_105675


namespace sum_of_decimals_is_fraction_l105_105871

theorem sum_of_decimals_is_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 1466 / 6250 := 
by
  sorry

end sum_of_decimals_is_fraction_l105_105871


namespace time_between_shark_sightings_l105_105575

def earnings_per_photo : ℕ := 15
def fuel_cost_per_hour : ℕ := 50
def hunting_hours : ℕ := 5
def expected_profit : ℕ := 200

theorem time_between_shark_sightings :
  (hunting_hours * 60) / ((expected_profit + (fuel_cost_per_hour * hunting_hours)) / earnings_per_photo) = 10 :=
by 
  sorry

end time_between_shark_sightings_l105_105575


namespace max_discount_rate_l105_105670

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l105_105670


namespace rectangle_length_is_16_l105_105200

noncomputable def rectangle_length (b : ℝ) (c : ℝ) : ℝ :=
  let pi := Real.pi
  let full_circle_circumference := 2 * c
  let radius := full_circle_circumference / (2 * pi)
  let diameter := 2 * radius
  let side_length_of_square := diameter
  let perimeter_of_square := 4 * side_length_of_square
  let perimeter_of_rectangle := perimeter_of_square
  let length_of_rectangle := (perimeter_of_rectangle / 2) - b
  length_of_rectangle

theorem rectangle_length_is_16 :
  rectangle_length 14 23.56 = 16 :=
by
  sorry

end rectangle_length_is_16_l105_105200


namespace time_solution_l105_105834

-- Define the condition as a hypothesis
theorem time_solution (x : ℝ) (h : x / 4 + (24 - x) / 2 = x) : x = 9.6 :=
by
  -- Proof skipped
  sorry

end time_solution_l105_105834


namespace linear_regression_eq_l105_105780

noncomputable def x_vals : List ℝ := [3, 7, 11]
noncomputable def y_vals : List ℝ := [10, 20, 24]

theorem linear_regression_eq :
  ∃ a b : ℝ, (a = 5.75) ∧ (b = 1.75) ∧ (∀ x, ∃ y, y = a + b * x) := sorry

end linear_regression_eq_l105_105780


namespace inequality_solution_set_l105_105332

open Set -- Open the Set namespace to work with sets in Lean

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (2 - x) ≥ 1 ↔ (x ∈ Icc (3 / 4) 2 \ {2}) := 
by
  sorry

end inequality_solution_set_l105_105332


namespace multiply_millions_l105_105385

theorem multiply_millions :
  (5 * 10^6) * (8 * 10^6) = 40 * 10^12 :=
by 
  sorry

end multiply_millions_l105_105385


namespace max_discount_rate_l105_105653

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l105_105653


namespace earphone_cost_l105_105348

/-- 
The cost of the earphone purchased on Friday can be calculated given:
1. The mean expenditure over 7 days is 500.
2. The expenditures for Monday, Tuesday, Wednesday, Thursday, Saturday, and Sunday are 450, 600, 400, 500, 550, and 300, respectively.
3. On Friday, the expenditures include a pen costing 30 and a notebook costing 50.
-/
theorem earphone_cost
  (mean_expenditure : ℕ)
  (mon tue wed thur sat sun : ℕ)
  (pen_cost notebook_cost : ℕ)
  (mean_expenditure_eq : mean_expenditure = 500)
  (mon_eq : mon = 450)
  (tue_eq : tue = 600)
  (wed_eq : wed = 400)
  (thur_eq : thur = 500)
  (sat_eq : sat = 550)
  (sun_eq : sun = 300)
  (pen_cost_eq : pen_cost = 30)
  (notebook_cost_eq : notebook_cost = 50)
  : ∃ (earphone_cost : ℕ), earphone_cost = 620 := 
by
  sorry

end earphone_cost_l105_105348


namespace find_A_for_diamondsuit_l105_105733

-- Define the operation
def diamondsuit (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

-- Define the specific instance of the operation equated to 57
theorem find_A_for_diamondsuit :
  ∃ A : ℝ, diamondsuit A 10 = 57 ↔ A = 20 := by
  sorry

end find_A_for_diamondsuit_l105_105733


namespace painters_work_days_l105_105557

/-
It takes five painters working at the same rate 1.5 work-days to finish a job.
If only four painters are available, prove how many work-days will it take them to finish the job, working at the same rate.
-/

theorem painters_work_days (days5 : ℚ) (h : days5 = 3 / 2) :
  ∃ days4 : ℚ, 5 * days5 = 4 * days4 ∧ days4 = 15 / 8 :=
  by
    use 15 / 8
    split
    · calc
        5 * days5 = 5 * (3 / 2) : by rw h
        ... = 15 / 2 : by norm_num
        ... = 4 * (15 / 8) : by norm_num
    · refl

end painters_work_days_l105_105557


namespace quadratic_has_real_roots_find_m_l105_105139

theorem quadratic_has_real_roots (m : ℝ) :
  let discriminant := (-4) ^ 2 - 4 * 1 * (-2 * m + 5) in
  discriminant ≥ 0 ↔ m ≥ 1 / 2 :=
by
  let discriminant := (-4) ^ 2 - 4 * 1 * (-2 * m + 5)
  split
  { intro h
    sorry -- This proof would show that if the discriminant is non-negative, then m ≥ 1/2
  }
  { intro h
    sorry -- This proof would show that if m ≥ 1/2, then the discriminant is non-negative
  }

theorem find_m (m : ℝ) (x1 x2 : ℝ) :
  (x1 + x2 = 4) →
  (x1 * x2 = -2 * m + 5) →
  (x1 * x2 + x1 + x2 = m ^ 2 + 6) →
  m ≥ 1 / 2 →
  m = 1 :=
by
  intros h1 h2 h3 h4
  sorry -- This proof would show that given the conditions, m must be 1

end quadratic_has_real_roots_find_m_l105_105139


namespace max_discount_rate_l105_105652

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l105_105652


namespace correct_simplification_l105_105073

theorem correct_simplification (m a b x y : ℝ) :
  ¬ (4 * m - m = 3) ∧
  ¬ (a^2 * b - a * b^2 = 0) ∧
  ¬ (2 * a^3 - 3 * a^3 = a^3) ∧
  (x * y - 2 * x * y = - x * y) :=
by {
  sorry
}

end correct_simplification_l105_105073


namespace positive_solution_in_interval_l105_105431

def quadratic (x : ℝ) := x^2 + 3 * x - 5

theorem positive_solution_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ quadratic x = 0 :=
sorry

end positive_solution_in_interval_l105_105431


namespace exponent_division_l105_105969

-- We need to reformulate the given condition into Lean definitions
def twenty_seven_is_three_cubed : Prop := 27 = 3^3

-- Using the condition to state the problem
theorem exponent_division (h : twenty_seven_is_three_cubed) : 
  3^15 / 27^3 = 729 :=
by
  sorry

end exponent_division_l105_105969


namespace opposite_of_neg_two_l105_105819

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l105_105819


namespace divisor_inequality_l105_105043

variable (k p : Nat)

-- Conditions
def is_prime (p : Nat) : Prop := Nat.Prime p
def is_divisor_of (k d : Nat) : Prop := ∃ m : Nat, d = k * m

-- Given conditions and the theorem to be proved
theorem divisor_inequality (h1 : k > 3) (h2 : is_prime p) (h3 : is_divisor_of k (2^p + 1)) : k ≥ 2 * p + 1 :=
  sorry

end divisor_inequality_l105_105043


namespace cost_of_chairs_l105_105031

-- Given conditions
def total_spent : ℕ := 56
def cost_of_table : ℕ := 34
def number_of_chairs : ℕ := 2

-- The target definition
def cost_of_one_chair : ℕ := 11

-- Statement to prove
theorem cost_of_chairs (N : ℕ) (T : ℕ) (C : ℕ) (x : ℕ) (hN : N = total_spent) (hT : T = cost_of_table) (hC : C = number_of_chairs) : x = cost_of_one_chair ↔ N - T = C * x :=
by
  sorry

end cost_of_chairs_l105_105031


namespace hairstylist_weekly_earnings_l105_105850

-- Definition of conditions as given in part a)
def cost_normal_haircut := 5
def cost_special_haircut := 6
def cost_trendy_haircut := 8

def number_normal_haircuts_per_day := 5
def number_special_haircuts_per_day := 3
def number_trendy_haircuts_per_day := 2

def working_days_per_week := 7

-- The goal is to prove that the hairstylist's weekly earnings equal to 413 dollars
theorem hairstylist_weekly_earnings : 
  (number_normal_haircuts_per_day * cost_normal_haircut +
  number_special_haircuts_per_day * cost_special_haircut +
  number_trendy_haircuts_per_day * cost_trendy_haircut) * 
  working_days_per_week = 413 := 
by sorry -- We use "by sorry" to skip the proof

end hairstylist_weekly_earnings_l105_105850


namespace range_of_m_value_of_m_l105_105138

-- Define the quadratic equation and the condition for having real roots
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x - 2*m + 5

-- Condition for the quadratic equation to have real roots
def discriminant_nonnegative (m : ℝ) : Prop := (4^2 - 4*1*(-2*m + 5)) ≥ 0

-- Define Vieta's formulas for the roots of the quadratic equation
def vieta_sum_roots (x1 x2 : ℝ) : Prop := x1 + x2 = 4
def vieta_product_roots (x1 x2 : ℝ) (m : ℝ) : Prop := x1 * x2 = -2*m + 5

-- Given condition with the roots
def condition_on_roots (x1 x2 m : ℝ) : Prop := x1 * x2 + x1 + x2 = m^2 + 6

-- Prove the range of m
theorem range_of_m (m : ℝ) : 
  discriminant_nonnegative m → m ≥ 1/2 := by 
  sorry

-- Prove the value of m based on the given root condition
theorem value_of_m (x1 x2 m : ℝ) : 
  vieta_sum_roots x1 x2 → 
  vieta_product_roots x1 x2 m → 
  condition_on_roots x1 x2 m → 
  m = 1 := by 
  sorry

end range_of_m_value_of_m_l105_105138


namespace min_sugar_l105_105374

theorem min_sugar (f s : ℝ) (h₁ : f ≥ 8 + (3/4) * s) (h₂ : f ≤ 2 * s) : s ≥ 32 / 5 :=
sorry

end min_sugar_l105_105374


namespace opposite_of_neg_two_is_two_l105_105815

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l105_105815


namespace x_axis_intercept_of_line_l105_105956

theorem x_axis_intercept_of_line (x : ℝ) : (∃ x, 2*x + 1 = 0) → x = - 1 / 2 :=
  by
    intro h
    obtain ⟨x, h1⟩ := h
    have : 2 * x + 1 = 0 := h1
    linarith [this]

end x_axis_intercept_of_line_l105_105956


namespace study_tour_arrangement_l105_105499

def number_of_arrangements (classes routes : ℕ) (max_selected_route : ℕ) : ℕ :=
  if classes = 4 ∧ routes = 4 ∧ max_selected_route = 2 then 240 else 0

theorem study_tour_arrangement :
  number_of_arrangements 4 4 2 = 240 :=
by sorry

end study_tour_arrangement_l105_105499


namespace arithmetic_sequence_10th_term_l105_105450

theorem arithmetic_sequence_10th_term (a d : ℤ) :
    (a + 4 * d = 26) →
    (a + 7 * d = 50) →
    (a + 9 * d = 66) := by
  intros h1 h2
  sorry

end arithmetic_sequence_10th_term_l105_105450


namespace max_discount_rate_l105_105638

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l105_105638


namespace sum_of_squares_of_coeffs_l105_105216

theorem sum_of_squares_of_coeffs :
  let p := 3 * (x^5 + 5 * x^3 + 2 * x + 1)
  let coeffs := [3, 15, 6, 3]
  coeffs.map (λ c => c^2) |>.sum = 279 := by
  sorry

end sum_of_squares_of_coeffs_l105_105216


namespace possible_numbers_l105_105436

theorem possible_numbers (N : ℕ) 
    (h1 : 10 ≤ N) (h2 : N ≤ 99)
    (h3 : (N % 3 = 0) ∧ (N % 4 = 0) ∧ (N % 5 = 0) ∧ (N % 9 = 0) ∧ (N % 10 = 0) ∧ 
          (N % 15 = 0) ∧ (N % 18 = 0) ∧ (N % 30 = 0) ∨ 
          (N % 3 ≠ 0) + (N % 4 ≠ 0) + (N % 5 ≠ 0) + (N % 9 ≠ 0) + 
          (N % 10 ≠ 0) + (N % 15 ≠ 0) + (N % 18 ≠ 0) + (N % 30 ≠ 0) = 4) :
   N = 36 ∨ N = 45 ∨ N = 72 :=
by {
  sorry
}

end possible_numbers_l105_105436


namespace volume_of_region_l105_105888

theorem volume_of_region : 
  ∀ (x y z : ℝ),
  abs (x + y + z) + abs (x - y + z) ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 → 
  ∃ V : ℝ, V = 62.5 :=
  sorry

end volume_of_region_l105_105888


namespace find_a12_a14_l105_105393

noncomputable def S (n : ℕ) (a_n : ℕ → ℝ) (b : ℝ) : ℝ := a_n n ^ 2 + b * n

noncomputable def is_arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ (a1 : ℝ) (c : ℝ), ∀ n : ℕ, a_n n = a1 + (n - 1) * c

theorem find_a12_a14
  (a_n : ℕ → ℝ)
  (b : ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, S n = a_n n ^ 2 + b * n)
  (h2 : S 25 = 100)
  (h3 : is_arithmetic_sequence a_n) :
  a_n 12 + a_n 14 = 5 :=
sorry

end find_a12_a14_l105_105393


namespace work_days_l105_105827

/-- A needs 20 days to complete the work alone. B needs 10 days to complete the work alone.
    The total work must be completed in 12 days. We need to find how many days B must work 
    before A continues, such that the total work equals the full task. -/
theorem work_days (x : ℝ) (h0 : 0 ≤ x ∧ x ≤ 12) (h1 : 1 / 10 * x + 1 / 20 * (12 - x) = 1) : x = 8 := by
  sorry

end work_days_l105_105827


namespace tallest_stack_is_b_l105_105062

def number_of_pieces_a : ℕ := 8
def number_of_pieces_b : ℕ := 11
def number_of_pieces_c : ℕ := 6

def height_per_piece_a : ℝ := 2
def height_per_piece_b : ℝ := 1.5
def height_per_piece_c : ℝ := 2.5

def total_height_a : ℝ := number_of_pieces_a * height_per_piece_a
def total_height_b : ℝ := number_of_pieces_b * height_per_piece_b
def total_height_c : ℝ := number_of_pieces_c * height_per_piece_c

theorem tallest_stack_is_b : (total_height_b = 16.5) ∧ (total_height_b > total_height_a) ∧ (total_height_b > total_height_c) := 
by
  sorry

end tallest_stack_is_b_l105_105062


namespace shirt_final_price_is_correct_l105_105095

noncomputable def final_price_percentage (initial_price : ℝ) : ℝ :=
  let first_discount := initial_price * 0.80
  let second_discount := first_discount * 0.90
  let anniversary_addition := second_discount * 1.05
  let final_price := anniversary_addition * 1.15
  final_price / initial_price * 100

theorem shirt_final_price_is_correct (initial_price : ℝ) : final_price_percentage initial_price = 86.94 := by
  sorry

end shirt_final_price_is_correct_l105_105095


namespace opposite_of_neg_two_is_two_l105_105811

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l105_105811


namespace companion_value_4164_smallest_N_satisfies_conditions_l105_105424

-- Define relevant functions
def G (N : ℕ) : ℕ :=
  let digits := [N / 1000 % 10, N / 100 % 10, N / 10 % 10, N % 10]
  digits.sum

def P (N : ℕ) : ℕ :=
  (N / 1000 % 10) * (N / 100 % 10)

def Q (N : ℕ) : ℕ :=
  (N / 10 % 10) * (N % 10)

def companion_value (N : ℕ) : ℚ :=
  |(G N : ℤ) / ((P N : ℤ) - (Q N : ℤ))|

-- Proof problem for part (1)
theorem companion_value_4164 : companion_value 4164 = 3 / 4 := sorry

-- Proof problem for part (2)
theorem smallest_N_satisfies_conditions :
  ∀ (N : ℕ), N > 1000 ∧ N < 10000 ∧ (∀ d, N / 10^d % 10 ≠ 0) ∧ (N / 1000 % 10 + N % 10) % 9 = 0 ∧ G N = 16 ∧ companion_value N = 4 → N = 2527 := sorry

end companion_value_4164_smallest_N_satisfies_conditions_l105_105424


namespace translation_coordinates_l105_105133

theorem translation_coordinates
  (a b : ℝ)
  (h₁ : 4 = a + 2)
  (h₂ : -3 = b - 6) :
  (a, b) = (2, 3) :=
by
  sorry

end translation_coordinates_l105_105133


namespace hotel_charge_comparison_l105_105045

theorem hotel_charge_comparison (R G P : ℝ) 
  (h1 : P = R - 0.70 * R)
  (h2 : P = G - 0.10 * G) :
  ((R - G) / G) * 100 = 170 :=
by
  sorry

end hotel_charge_comparison_l105_105045


namespace rectangle_length_difference_l105_105528

variable (s l w : ℝ)

-- Conditions
def condition1 : Prop := 2 * (l + w) = 4 * s + 4
def condition2 : Prop := w = s - 2

-- Theorem to prove
theorem rectangle_length_difference
  (s l w : ℝ)
  (h1 : condition1 s l w)
  (h2 : condition2 s w) : l = s + 4 :=
by
sorry

end rectangle_length_difference_l105_105528


namespace original_number_l105_105093

theorem original_number (x : ℤ) (h : 5 * x - 9 = 51) : x = 12 :=
sorry

end original_number_l105_105093


namespace part1_part2_part3_l105_105758

-- Problem 1
theorem part1 (a b: ℝ) (f g: ℝ → ℝ) (m: ℝ):
  (∀ x, f x = log x - a * x) →
  (∀ x, g x = b / x) →
  (f 1 = m ∧ g 1 = m) →
  (f' 1 = g' 1) →
  (∀ x, tangent_line_at f 1 = tangent_line_at g 1) →
  (∀ y, x - 2 * y - 2 = 0)  := 
by sorry

-- Problem 2
theorem part2 (a: ℝ) (f: ℝ → ℝ):
  (∀ x, f x = log x - a * x) →
  ((∃ c, ∀ x, f' x = c) ∧ (∀ x, f x ≠ 0)) →
  (a > 1 / real.exp 1) := 
by sorry

-- Problem 3
theorem part3 (a: ℝ) (b: ℝ) (F: ℝ → ℝ):
  (0 < a) →
  (b = 1) →
  (∀ x, F x = log x - a * x - 1 / x) →
  (0 < a ∧ a < log 2 + 1 / 2 → ∀ x ∈ set.Icc 1 2, F x ≥ -a - 1) ∧ 
  (a ≥ log 2 + 1 / 2 → ∀ x ∈ set.Icc 1 2, F x ≥ log 2 - 1 / 2 - 2 * a) := 
by sorry

end part1_part2_part3_l105_105758


namespace negation_proof_l105_105197

theorem negation_proof :
  ¬(∀ x : ℝ, x > 0 → Real.exp x > x + 1) ↔ ∃ x : ℝ, x > 0 ∧ Real.exp x ≤ x + 1 :=
by sorry

end negation_proof_l105_105197


namespace day_of_week_after_45_days_l105_105508

theorem day_of_week_after_45_days (day_of_week : ℕ → String) (birthday_is_tuesday : day_of_week 0 = "Tuesday") : day_of_week 45 = "Friday" :=
by
  sorry

end day_of_week_after_45_days_l105_105508


namespace clock_resale_price_l105_105087

theorem clock_resale_price
    (C : ℝ)  -- original cost of the clock to the store
    (H1 : 0.40 * C = 100)  -- condition: difference between original cost and buy-back price is $100
    (H2 : ∀ (C : ℝ), resell_price = 1.80 * (0.60 * C))  -- store sold the clock again with a 80% profit on buy-back
    : resell_price = 270 := 
by
  sorry

end clock_resale_price_l105_105087


namespace max_smaller_boxes_fit_l105_105541

theorem max_smaller_boxes_fit (length_large width_large height_large : ℝ)
  (length_small width_small height_small : ℝ)
  (h1 : length_large = 6)
  (h2 : width_large = 5)
  (h3 : height_large = 4)
  (hs1 : length_small = 0.60)
  (hs2 : width_small = 0.50)
  (hs3 : height_small = 0.40) :
  length_large * width_large * height_large / (length_small * width_small * height_small) = 1000 := 
  by
  sorry

end max_smaller_boxes_fit_l105_105541


namespace value_of_m_l105_105912

-- Define the condition of the quadratic equation
def quadratic_equation (x m : ℝ) := x^2 - 2*x + m

-- State the equivalence to be proved
theorem value_of_m (m : ℝ) : (∃ x : ℝ, x = 1 ∧ quadratic_equation x m = 0) → m = 1 :=
by
  sorry

end value_of_m_l105_105912


namespace maxwell_meets_brad_l105_105350

theorem maxwell_meets_brad :
  ∃ t : ℝ, t = 2 ∧ 
  (∀ distance max_speed brad_speed start_time, 
   distance = 14 ∧ 
   max_speed = 4 ∧ 
   brad_speed = 6 ∧ 
   start_time = 1 → 
   max_speed * (t + start_time) + brad_speed * t = distance) :=
by
  use 1
  sorry

end maxwell_meets_brad_l105_105350


namespace mariela_cards_received_l105_105845

theorem mariela_cards_received (cards_in_hospital : ℕ) (cards_at_home : ℕ) 
  (h1 : cards_in_hospital = 403) (h2 : cards_at_home = 287) : 
  cards_in_hospital + cards_at_home = 690 := 
by 
  sorry

end mariela_cards_received_l105_105845


namespace correct_quotient_remainder_sum_l105_105400

theorem correct_quotient_remainder_sum :
  ∃ N : ℕ, (N % 23 = 17 ∧ N / 23 = 3) ∧ (∃ q r : ℕ, N = 32 * q + r ∧ r < 32 ∧ q + r = 24) :=
by
  sorry

end correct_quotient_remainder_sum_l105_105400


namespace maximum_discount_rate_l105_105628

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l105_105628


namespace evaluate_product_l105_105515

open BigOperators

noncomputable def product_term (n : ℕ) : ℝ := 1 - 2 / n

theorem evaluate_product :
  ∏ (n : ℕ) in Finset.range 98 \ {0, 1, 2}, product_term (n + 3) = 49 / 150 :=
by
  rw [Finset.range_eq_Ico, Finset.Ico_filter_lt_eq_Ico]; simp
  sorry

end evaluate_product_l105_105515


namespace factory_correct_decision_prob_l105_105090

def prob_correct_decision (p : ℝ) : ℝ :=
  let prob_all_correct := p * p * p
  let prob_two_correct_one_incorrect := 3 * p * p * (1 - p)
  prob_all_correct + prob_two_correct_one_incorrect

theorem factory_correct_decision_prob : prob_correct_decision 0.8 = 0.896 :=
by
  sorry

end factory_correct_decision_prob_l105_105090


namespace base_length_of_isosceles_triangle_triangle_l105_105916

section Geometry

variable {b m x : ℝ}

-- Define the conditions
def isosceles_triangle (b : ℝ) : Prop :=
∀ {A B C : ℝ}, A = b ∧ B = b -- representing an isosceles triangle with two equal sides

def segment_length (m : ℝ) : Prop :=
∀ {D E : ℝ}, D - E = m -- the segment length between points where bisectors intersect sides is m

-- The theorem we want to prove
theorem base_length_of_isosceles_triangle_triangle (h1 : isosceles_triangle b) (h2 : segment_length m) : x = b * m / (b - m) :=
sorry

end Geometry

end base_length_of_isosceles_triangle_triangle_l105_105916


namespace find_angle_A_area_bound_given_a_l105_105526

-- (1) Given the condition, prove that \(A = \frac{\pi}{3}\).
theorem find_angle_A
  {A B C : ℝ} {a b c : ℝ}
  (h1 : a / (Real.cos A) + b / (Real.cos B) + c / (Real.cos C) = (Real.sqrt 3) * c * (Real.sin B) / (Real.cos B * Real.cos C)) :
  A = Real.pi / 3 :=
sorry

-- (2) Given a = 4, prove the area S satisfies \(S \leq 4\sqrt{3}\).
theorem area_bound_given_a
  {A B C : ℝ} {a b c S : ℝ}
  (ha : a = 4)
  (hA : A = Real.pi / 3)
  (h1 : a / (Real.cos A) + b / (Real.cos B) + c / (Real.cos C) = (Real.sqrt 3) * c * (Real.sin B) / (Real.cos B * Real.cos C))
  (hS : S = 1 / 2 * b * c * Real.sin A) :
  S ≤ 4 * Real.sqrt 3 :=
sorry

end find_angle_A_area_bound_given_a_l105_105526


namespace sum_of_possible_values_l105_105958

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 4) = -21) : (∃ x y : ℝ, x * (x - 4) = -21 ∧ y * (y - 4) = -21 ∧ x + y = 4) :=
sorry

end sum_of_possible_values_l105_105958


namespace ramu_profit_percent_l105_105580

def ramu_bought_car : ℝ := 48000
def ramu_repair_cost : ℝ := 14000
def ramu_selling_price : ℝ := 72900

theorem ramu_profit_percent :
  let total_cost := ramu_bought_car + ramu_repair_cost
  let profit := ramu_selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  profit_percent = 17.58 := 
by
  -- Definitions and setting up the proof environment
  let total_cost := ramu_bought_car + ramu_repair_cost
  let profit := ramu_selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  sorry

end ramu_profit_percent_l105_105580


namespace min_value_of_a2_b2_l105_105543

theorem min_value_of_a2_b2 (a b : ℝ) : 
  (∀ x : ℝ, (∃ r : ℕ, r = 3 ∧ binomial 6 r * a^(6-r) * b^r * x^(12 - 3*r) = 20 * x^3)) → a * b = 1 → a^2 + b^2 ≥ 2 := 
by
  sorry

end min_value_of_a2_b2_l105_105543


namespace ratio_ab_l105_105399

variable (x y a b : ℝ)
variable (h1 : 4 * x - 2 * y = a)
variable (h2 : 6 * y - 12 * x = b)
variable (h3 : b ≠ 0)

theorem ratio_ab : 4 * x - 2 * y = a ∧ 6 * y - 12 * x = b ∧ b ≠ 0 → a / b = -1 / 3 := by
  sorry

end ratio_ab_l105_105399


namespace solve_boys_left_l105_105337

--given conditions
variable (boys_initial girls_initial boys_left girls_entered children_end: ℕ)
variable (h_boys_initial : boys_initial = 5)
variable (h_girls_initial : girls_initial = 4)
variable (h_girls_entered : girls_entered = 2)
variable (h_children_end : children_end = 8)

-- Problem definition
def boys_left_proof : Prop :=
  ∃ (B : ℕ), boys_left = B ∧ boys_initial - B + girls_initial + girls_entered = children_end ∧ B = 3

-- The statement to be proven
theorem solve_boys_left : boys_left_proof boys_initial girls_initial boys_left girls_entered children_end := by
  -- Proof will be provided here
  sorry

end solve_boys_left_l105_105337


namespace intersect_once_l105_105244

theorem intersect_once (x : ℝ) : 
  (∀ y, y = 3 * Real.log x ↔ y = Real.log (3 * x)) → (∃! x, 3 * Real.log x = Real.log (3 * x)) :=
by 
  sorry

end intersect_once_l105_105244


namespace lowest_dropped_score_l105_105165

theorem lowest_dropped_score (A B C D : ℕ) 
  (h1 : (A + B + C + D) / 4 = 90)
  (h2 : (A + B + C) / 3 = 85) :
  D = 105 :=
by
  sorry

end lowest_dropped_score_l105_105165


namespace bushels_given_away_l105_105506

-- Definitions from the problem conditions
def initial_bushels : ℕ := 50
def ears_per_bushel : ℕ := 14
def remaining_ears : ℕ := 357

-- Theorem to prove the number of bushels given away
theorem bushels_given_away : 
  initial_bushels * ears_per_bushel - remaining_ears = 24 * ears_per_bushel :=
by
  sorry

end bushels_given_away_l105_105506


namespace opposite_of_neg2_l105_105803

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l105_105803


namespace max_discount_rate_l105_105677

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l105_105677


namespace total_apples_after_transactions_l105_105594

def initial_apples : ℕ := 65
def percentage_used : ℕ := 20
def apples_bought : ℕ := 15

theorem total_apples_after_transactions :
  (initial_apples * (1 - percentage_used / 100)) + apples_bought = 67 := 
by
  sorry

end total_apples_after_transactions_l105_105594


namespace abc_product_le_two_l105_105760

theorem abc_product_le_two (a b c : ℝ) (ha : 0 ≤ a) (ha2 : a ≤ 2) (hb : 0 ≤ b) (hb2 : b ≤ 2) (hc : 0 ≤ c) (hc2 : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 :=
sorry

end abc_product_le_two_l105_105760


namespace probability_interval_l105_105201

theorem probability_interval (P_A P_B : ℚ) (h1 : P_A = 5/6) (h2 : P_B = 3/4) :
  ∃ p : ℚ, (5/12 ≤ p ∧ p ≤ 3/4) :=
sorry

end probability_interval_l105_105201


namespace freds_change_l105_105175

theorem freds_change (ticket_cost : ℝ) (num_tickets : ℕ) (borrowed_movie_cost : ℝ) (total_paid : ℝ) 
  (h_ticket_cost : ticket_cost = 5.92) 
  (h_num_tickets : num_tickets = 2) 
  (h_borrowed_movie_cost : borrowed_movie_cost = 6.79) 
  (h_total_paid : total_paid = 20) : 
  total_paid - (num_tickets * ticket_cost + borrowed_movie_cost) = 1.37 := 
by 
  sorry

end freds_change_l105_105175


namespace sin_b_in_triangle_l105_105406

theorem sin_b_in_triangle (a b : ℝ) (sin_A sin_B : ℝ) (h₁ : a = 2) (h₂ : b = 1) (h₃ : sin_A = 1 / 3) 
  (h₄ : sin_B = (b * sin_A) / a) : sin_B = 1 / 6 :=
by
  have h₅ : sin_B = 1 / 6 := by 
    sorry
  exact h₅

end sin_b_in_triangle_l105_105406


namespace number_of_dogs_l105_105312

theorem number_of_dogs (cost_price selling_price total_amount : ℝ) (profit_percentage : ℝ)
    (h1 : cost_price = 1000)
    (h2 : profit_percentage = 0.30)
    (h3 : total_amount = 2600)
    (h4 : selling_price = cost_price + (profit_percentage * cost_price)) :
    total_amount / selling_price = 2 :=
by
  sorry

end number_of_dogs_l105_105312


namespace sum_of_decimals_as_fraction_l105_105874

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 733 / 3125 := by
  sorry

end sum_of_decimals_as_fraction_l105_105874


namespace triangle_leg_length_l105_105554

theorem triangle_leg_length (perimeter_square : ℝ)
                            (base_triangle : ℝ)
                            (area_equality : ∃ (side_square : ℝ) (height_triangle : ℝ),
                                4 * side_square = perimeter_square ∧
                                side_square * side_square = (1/2) * base_triangle * height_triangle)
                            : ∃ (y : ℝ), y = 22.5 :=
by
  -- Placeholder proof
  sorry

end triangle_leg_length_l105_105554


namespace zhuzhuxia_defeats_monsters_l105_105230

theorem zhuzhuxia_defeats_monsters {a : ℕ} (H1 : zhuzhuxia_total_defeated_monsters = 20) :
  zhuzhuxia_total_defeated_by_monsters = 8 :=
sorry

end zhuzhuxia_defeats_monsters_l105_105230


namespace chocolate_chip_cookies_count_l105_105157

theorem chocolate_chip_cookies_count (h1 : 5 / 2 = 20 / (x : ℕ)) : x = 8 := 
by
  sorry -- Proof to be implemented

end chocolate_chip_cookies_count_l105_105157


namespace smallest_number_is_32_l105_105210

theorem smallest_number_is_32 (a b c : ℕ) (h1 : a + b + c = 90) (h2 : b = 25) (h3 : c = 25 + 8) : a = 32 :=
by {
  sorry
}

end smallest_number_is_32_l105_105210


namespace number_of_partitions_indistinguishable_balls_into_boxes_l105_105732

/-- The number of distinct ways to partition 6 indistinguishable balls into 3 indistinguishable boxes is 7. -/
theorem number_of_partitions_indistinguishable_balls_into_boxes :
  ∃ n : ℕ, n = 7 := sorry

end number_of_partitions_indistinguishable_balls_into_boxes_l105_105732


namespace fraction_subtraction_simplification_l105_105692

/-- Given that 57 equals 19 times 3, we want to prove that (8/19) - (5/57) equals 1/3. -/
theorem fraction_subtraction_simplification :
  8 / 19 - 5 / 57 = 1 / 3 := by
  sorry

end fraction_subtraction_simplification_l105_105692


namespace product_of_three_numbers_l105_105598

theorem product_of_three_numbers:
  ∃ (a b c : ℚ), 
    a + b + c = 30 ∧ 
    a = 2 * (b + c) ∧ 
    b = 5 * c ∧ 
    a * b * c = 2500 / 9 :=
by {
  sorry
}

end product_of_three_numbers_l105_105598


namespace renaming_not_unnoticeable_l105_105843

-- Define the conditions as necessary structures for cities and connections
structure City := (name : String)
structure Connection := (city1 city2 : City)

-- Definition of the king's list of connections
def kingList : List Connection := sorry  -- The complete list of connections

-- The renaming function represented generically
def rename (c1 c2 : City) : City := sorry  -- The renaming function which is unspecified here

-- The main theorem statement
noncomputable def renaming_condition (c1 c2 : City) : Prop :=
  -- This condition represents that renaming preserves the king's perception of connections
  ∀ c : City, sorry  -- The specific condition needs full details of renaming logic

-- The theorem to prove, which states that the renaming is not always unnoticeable
theorem renaming_not_unnoticeable : ∃ c1 c2 : City, ¬ renaming_condition c1 c2 := sorry

end renaming_not_unnoticeable_l105_105843


namespace sweater_markup_l105_105985

-- Conditions
variables (W R : ℝ)
axiom h1 : 0.40 * R = 1.20 * W

-- Theorem statement
theorem sweater_markup (W R : ℝ) (h1 : 0.40 * R = 1.20 * W) : (R - W) / W * 100 = 200 :=
sorry

end sweater_markup_l105_105985


namespace initial_juggling_objects_l105_105561

theorem initial_juggling_objects (x : ℕ) : (∀ i : ℕ, i = 5 → x + 2*i = 13) → x = 3 :=
by 
  intro h
  sorry

end initial_juggling_objects_l105_105561


namespace juliette_and_marco_money_comparison_l105_105323

noncomputable def euro_to_dollar (eur : ℝ) : ℝ := eur * 1.5

theorem juliette_and_marco_money_comparison :
  (600 - euro_to_dollar 350) / 600 * 100 = 12.5 := by
sorry

end juliette_and_marco_money_comparison_l105_105323


namespace probability_calc_l105_105103

noncomputable def probability_no_distinct_positive_real_roots : ℚ :=
  let pairs_count := 169
  let valid_pairs_count := 17
  1 - (valid_pairs_count / pairs_count : ℚ)

theorem probability_calc :
  probability_no_distinct_positive_real_roots = 152 / 169 := by sorry

end probability_calc_l105_105103


namespace base_length_first_tri_sail_l105_105027

-- Define the areas of the sails
def area_rect_sail : ℕ := 5 * 8
def area_second_tri_sail : ℕ := (4 * 6) / 2

-- Total canvas area needed
def total_canvas_area_needed : ℕ := 58

-- Calculate the total area so far (rectangular sail + second triangular sail)
def total_area_so_far : ℕ := area_rect_sail + area_second_tri_sail

-- Define the height of the first triangular sail
def height_first_tri_sail : ℕ := 4

-- Define the area needed for the first triangular sail
def area_first_tri_sail : ℕ := total_canvas_area_needed - total_area_so_far

-- Prove that the base length of the first triangular sail is 3 inches
theorem base_length_first_tri_sail : ∃ base : ℕ, base = 3 ∧ (base * height_first_tri_sail) / 2 = area_first_tri_sail := by
  use 3
  have h1 : (3 * 4) / 2 = 6 := by sorry -- This is a placeholder for actual calculation
  exact ⟨rfl, h1⟩

end base_length_first_tri_sail_l105_105027


namespace lawn_width_is_60_l105_105995

theorem lawn_width_is_60
  (length : ℕ)
  (width : ℕ)
  (road_width : ℕ)
  (cost_per_sq_meter : ℕ)
  (total_cost : ℕ)
  (area_of_lawn : ℕ)
  (total_area_of_roads : ℕ)
  (intersection_area : ℕ)
  (area_cost_relation : total_area_of_roads * cost_per_sq_meter = total_cost)
  (intersection_included : (road_width * length + road_width * width - intersection_area) = total_area_of_roads)
  (length_eq : length = 80)
  (road_width_eq : road_width = 10)
  (cost_eq : cost_per_sq_meter = 2)
  (total_cost_eq : total_cost = 2600)
  (intersection_area_eq : intersection_area = road_width * road_width)
  : width = 60 :=
by
  sorry

end lawn_width_is_60_l105_105995


namespace range_of_m_l105_105049

theorem range_of_m (α : ℝ) (m : ℝ) (h : (α > π ∧ α < 3 * π / 2) ∨ (α > 3 * π / 2 ∧ α < 2 * π)) :
  -1 < (Real.sin α) ∧ (Real.sin α) < 0 ∧ (Real.sin α) = (2 * m - 3) / (4 - m) → 
  m ∈ Set.Ioo (-1 : ℝ) (3 / 2 : ℝ) :=
by
  sorry

end range_of_m_l105_105049


namespace exist_two_quadrilaterals_l105_105512

-- Define the structure of a quadrilateral with four sides and two diagonals
structure Quadrilateral :=
  (s1 : ℝ) -- side 1
  (s2 : ℝ) -- side 2
  (s3 : ℝ) -- side 3
  (s4 : ℝ) -- side 4
  (d1 : ℝ) -- diagonal 1
  (d2 : ℝ) -- diagonal 2

-- The theorem stating the existence of two quadrilaterals satisfying the given conditions
theorem exist_two_quadrilaterals :
  ∃ (quad1 quad2 : Quadrilateral),
  quad1.s1 < quad2.s1 ∧ quad1.s2 < quad2.s2 ∧ quad1.s3 < quad2.s3 ∧ quad1.s4 < quad2.s4 ∧
  quad1.d1 > quad2.d1 ∧ quad1.d2 > quad2.d2 :=
by
  sorry

end exist_two_quadrilaterals_l105_105512


namespace sector_area_correct_l105_105286

-- Definitions based on the conditions
def sector_perimeter := 16 -- cm
def central_angle := 2 -- radians
def radius := 4 -- The radius computed from perimeter condition

-- Lean 4 statement to prove the equivalent math problem
theorem sector_area_correct : ∃ (s : ℝ), 
  (∀ (r : ℝ), (2 * r + r * central_angle = sector_perimeter → r = 4) → 
  (s = (1 / 2) * central_angle * (radius) ^ 2) → 
  s = 16) :=
by 
  sorry

end sector_area_correct_l105_105286


namespace percentage_decrease_correct_l105_105988

theorem percentage_decrease_correct :
  ∀ (p : ℝ), (1 + 0.25) * (1 - p) = 1 → p = 0.20 :=
by
  intro p
  intro h
  sorry

end percentage_decrease_correct_l105_105988


namespace evaluate_expression_l105_105247

theorem evaluate_expression : (733 * 733) - (732 * 734) = 1 :=
by
  sorry

end evaluate_expression_l105_105247


namespace max_discount_rate_l105_105639

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l105_105639


namespace alpha_beta_sum_pi_over_2_l105_105280

theorem alpha_beta_sum_pi_over_2 (α β : ℝ) (hα : 0 < α) (hα_lt : α < π / 2) (hβ : 0 < β) (hβ_lt : β < π / 2) (h : Real.sin (α + β) = Real.sin α ^ 2 + Real.sin β ^ 2) : α + β = π / 2 :=
by
  -- Proof steps would go here
  sorry

end alpha_beta_sum_pi_over_2_l105_105280


namespace farmer_land_l105_105491

theorem farmer_land (A : ℝ) (A_nonneg : A ≥ 0) (cleared_land : ℝ) 
  (soybeans wheat potatoes vegetables corn : ℝ) 
  (h_cleared : cleared_land = 0.95 * A) 
  (h_soybeans : soybeans = 0.35 * cleared_land) 
  (h_wheat : wheat = 0.40 * cleared_land) 
  (h_potatoes : potatoes = 0.15 * cleared_land) 
  (h_vegetables : vegetables = 0.08 * cleared_land) 
  (h_corn : corn = 630) 
  (cleared_sum : soybeans + wheat + potatoes + vegetables + corn = cleared_land) :
  A = 33158 := 
by 
  sorry

end farmer_land_l105_105491


namespace find_b_plus_m_l105_105471

-- Definitions of the constants and functions based on the given conditions.
variables (m b : ℝ)

-- The first line equation passing through (5, 8).
def line1 := 8 = m * 5 + 3

-- The second line equation passing through (5, 8).
def line2 := 8 = 4 * 5 + b

-- The goal statement we need to prove.
theorem find_b_plus_m (h1 : line1 m) (h2 : line2 b) : b + m = -11 :=
sorry

end find_b_plus_m_l105_105471


namespace xiaohua_distance_rounds_l105_105498

def length := 5
def width := 3
def perimeter (a b : ℕ) := (a + b) * 2
def total_distance (perimeter : ℕ) (laps : ℕ) := perimeter * laps

theorem xiaohua_distance_rounds :
  total_distance (perimeter length width) 3 = 30 :=
by sorry

end xiaohua_distance_rounds_l105_105498


namespace circular_pipes_equivalence_l105_105703

/-- Determine how many circular pipes with an inside diameter 
of 2 inches are required to carry the same amount of water as 
one circular pipe with an inside diameter of 8 inches. -/
theorem circular_pipes_equivalence 
  (d_small d_large : ℝ)
  (h1 : d_small = 2)
  (h2 : d_large = 8) :
  (d_large / 2) ^ 2 / (d_small / 2) ^ 2 = 16 :=
by
  sorry

end circular_pipes_equivalence_l105_105703


namespace lisa_eggs_l105_105570

theorem lisa_eggs :
  ∃ x : ℕ, (5 * 52) * (4 * x + 3 + 2) = 3380 ∧ x = 2 :=
by
  sorry

end lisa_eggs_l105_105570


namespace polynomial_at_3mnplus1_l105_105258

noncomputable def polynomial_value (x : ℤ) : ℤ := x^2 + 4 * x + 6

theorem polynomial_at_3mnplus1 (m n : ℤ) (h₁ : 2 * m + n + 2 = m + 2 * n) (h₂ : m - n + 2 ≠ 0) :
  polynomial_value (3 * (m + n + 1)) = 3 := 
by 
  sorry

end polynomial_at_3mnplus1_l105_105258


namespace cricket_initial_matches_l105_105009

theorem cricket_initial_matches (x : ℝ) :
  (0.28 * x + 60 = 0.52 * (x + 60)) → x = 120 :=
by
  sorry

end cricket_initial_matches_l105_105009


namespace money_raised_is_correct_l105_105416

noncomputable def cost_per_dozen : ℚ := 2.40
noncomputable def selling_price_per_donut : ℚ := 1
noncomputable def dozens : ℕ := 10

theorem money_raised_is_correct :
  (dozens * 12 * selling_price_per_donut - dozens * cost_per_dozen) = 96 := by
sorry

end money_raised_is_correct_l105_105416


namespace correct_operation_l105_105218

variable (m n : ℝ)

-- Define the statement to be proved
theorem correct_operation : (-2 * m * n) ^ 2 = 4 * m ^ 2 * n ^ 2 :=
by sorry

end correct_operation_l105_105218


namespace steven_shipment_boxes_l105_105947

theorem steven_shipment_boxes (num_trucks : ℕ) (truck_capacity : ℕ)
  (light_box_weight heavy_box_weight : ℕ) :
  num_trucks = 3 →
  truck_capacity = 2000 →
  light_box_weight = 10 →
  heavy_box_weight = 40 →
  ∃ (total_boxes : ℕ), total_boxes = 240 :=
by 
  intros h_num_trucks h_truck_capacity h_light_box_weight h_heavy_box_weight
  use 240
  sorry

end steven_shipment_boxes_l105_105947


namespace find_value_of_expression_l105_105699

theorem find_value_of_expression
  (k m : ℕ)
  (hk : 3^(k - 1) = 9)
  (hm : 4^(m + 2) = 64) :
  2^(3*k + 2*m) = 2^11 :=
by 
  sorry

end find_value_of_expression_l105_105699


namespace greatest_int_less_than_neg_19_div_3_l105_105067

theorem greatest_int_less_than_neg_19_div_3 : ∃ n : ℤ, n = -7 ∧ n < (-19 / 3 : ℚ) ∧ (-19 / 3 : ℚ) < n + 1 := 
by
  sorry

end greatest_int_less_than_neg_19_div_3_l105_105067


namespace train_speed_conversion_l105_105098

def km_per_hour_to_m_per_s (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

theorem train_speed_conversion (speed_kmph : ℕ) (h : speed_kmph = 108) :
  km_per_hour_to_m_per_s speed_kmph = 30 :=
by
  rw [h]
  sorry

end train_speed_conversion_l105_105098


namespace shark_sightings_in_cape_may_l105_105240

theorem shark_sightings_in_cape_may (x : ℕ) (hx : x + (2 * x - 8) = 40) : 2 * x - 8 = 24 := 
by 
  sorry

end shark_sightings_in_cape_may_l105_105240


namespace range_of_m_for_inequality_l105_105079

theorem range_of_m_for_inequality (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + m * x + m - 6 < 0) ↔ m < 8 := 
sorry

end range_of_m_for_inequality_l105_105079


namespace number_of_ways_difference_of_squares_l105_105303

-- Lean statement
theorem number_of_ways_difference_of_squares (n k : ℕ) (h1 : n > 10^k) (h2 : n % 10^k = 0) (h3 : k ≥ 2) :
  ∃ D, D = k^2 - 1 ∧ ∀ (a b : ℕ), n = a^2 - b^2 → D = k^2 - 1 :=
by
  sorry

end number_of_ways_difference_of_squares_l105_105303


namespace problem_statement_l105_105435

-- Definitions based on conditions
def two_digit_number (N : ℕ) := N >= 10 ∧ N < 100
def divisible_by (a b : ℕ) := a % b = 0
def mistaken_exactly (N : ℕ) := (if divisible_by N 3 then 0 else 1) +
                               (if divisible_by N 4 then 0 else 1) +
                               (if divisible_by N 5 then 0 else 1) +
                               (if divisible_by N 9 then 0 else 1) +
                               (if divisible_by N 10 then 0 else 1) +
                               (if divisible_by N 15 then 0 else 1) +
                               (if divisible_by N 18 then 0 else 1) +
                               (if divisible_by N 30 then 0 else 1)

-- Lean 4 statement
theorem problem_statement (N : ℕ) (h_two_digit : two_digit_number N) (h_mistaken : mistaken_exactly N = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 := sorry

end problem_statement_l105_105435


namespace hairstylist_earnings_per_week_l105_105851

theorem hairstylist_earnings_per_week :
  let cost_normal := 5
  let cost_special := 6
  let cost_trendy := 8
  let haircuts_normal := 5
  let haircuts_special := 3
  let haircuts_trendy := 2
  let days_per_week := 7
  let daily_earnings := cost_normal * haircuts_normal + cost_special * haircuts_special + cost_trendy * haircuts_trendy
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 413 := sorry

end hairstylist_earnings_per_week_l105_105851


namespace sequence_root_formula_l105_105896

theorem sequence_root_formula {a : ℕ → ℝ} 
    (h1 : ∀ n, (a (n + 1))^2 = (a n)^2 + 4)
    (h2 : a 1 = 1)
    (h3 : ∀ n, a n > 0) :
    ∀ n, a n = Real.sqrt (4 * n - 3) := 
sorry

end sequence_root_formula_l105_105896


namespace option_a_option_d_l105_105129

theorem option_a (n m : ℕ) (h1 : 1 ≤ n) (h2 : 1 ≤ m) (h3 : n > m) : 
  Nat.choose n m = Nat.choose n (n - m) := 
sorry

theorem option_d (n m : ℕ) (h1 : 1 ≤ n) (h2 : 1 ≤ m) (h3 : n > m) : 
  Nat.choose n m + Nat.choose n (m - 1) = Nat.choose (n + 1) m := 
sorry

end option_a_option_d_l105_105129


namespace Pooja_speed_3_l105_105767

variable (Roja_speed Pooja_speed : ℝ)
variable (t d : ℝ)

theorem Pooja_speed_3
  (h1 : Roja_speed = 6)
  (h2 : t = 4)
  (h3 : d = 36)
  (h4 : d = t * (Roja_speed + Pooja_speed)) :
  Pooja_speed = 3 :=
by
  sorry

end Pooja_speed_3_l105_105767


namespace petals_per_rose_l105_105414

theorem petals_per_rose
    (roses_per_bush : ℕ)
    (bushes : ℕ)
    (bottles : ℕ)
    (oz_per_bottle : ℕ)
    (petals_per_oz : ℕ)
    (petals : ℕ)
    (ounces : ℕ := bottles * oz_per_bottle)
    (total_petals : ℕ := ounces * petals_per_oz)
    (petals_per_bush : ℕ := total_petals / bushes)
    (petals_per_rose : ℕ := petals_per_bush / roses_per_bush) :
    petals_per_oz = 320 →
    roses_per_bush = 12 →
    bushes = 800 →
    bottles = 20 →
    oz_per_bottle = 12 →
    petals_per_rose = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end petals_per_rose_l105_105414


namespace fraction_of_coins_1780_to_1799_l105_105582

theorem fraction_of_coins_1780_to_1799 :
  let num_states_1780_to_1789 := 12,
      num_states_1790_to_1799 := 5,
      total_states := 30 in
  (num_states_1780_to_1789 + num_states_1790_to_1799) / total_states = 17 / 30 :=
by
  sorry

end fraction_of_coins_1780_to_1799_l105_105582


namespace fg_at_3_l105_105150

-- Define the functions f and g according to the conditions
def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x + 2)^2

theorem fg_at_3 : f (g 3) = 103 :=
by
  sorry

end fg_at_3_l105_105150


namespace area_difference_is_correct_l105_105777

noncomputable def circumference_1 : ℝ := 264
noncomputable def circumference_2 : ℝ := 352

noncomputable def radius_1 : ℝ := circumference_1 / (2 * Real.pi)
noncomputable def radius_2 : ℝ := circumference_2 / (2 * Real.pi)

noncomputable def area_1 : ℝ := Real.pi * radius_1^2
noncomputable def area_2 : ℝ := Real.pi * radius_2^2

noncomputable def area_difference : ℝ := area_2 - area_1

theorem area_difference_is_correct :
  abs (area_difference - 4305.28) < 1e-2 :=
by
  sorry

end area_difference_is_correct_l105_105777


namespace white_area_correct_l105_105599

def total_sign_area : ℕ := 8 * 20
def black_area_C : ℕ := 8 * 1 + 2 * (1 * 3)
def black_area_A : ℕ := 2 * (8 * 1) + 2 * (1 * 2)
def black_area_F : ℕ := 8 * 1 + 2 * (1 * 4)
def black_area_E : ℕ := 3 * (1 * 4)

def total_black_area : ℕ := black_area_C + black_area_A + black_area_F + black_area_E
def white_area : ℕ := total_sign_area - total_black_area

theorem white_area_correct : white_area = 98 :=
  by 
    sorry -- State the theorem without providing the proof.

end white_area_correct_l105_105599


namespace expression_I_evaluation_expression_II_evaluation_l105_105704

theorem expression_I_evaluation :
  ( (3 / 2) ^ (-2: ℤ) - (49 / 81) ^ (0.5: ℝ) + (0.008: ℝ) ^ (-2 / 3: ℝ) * (2 / 25) ) = (5 / 3) := 
by
  sorry

theorem expression_II_evaluation :
  ( (Real.logb 2 2) ^ 2 + (Real.logb 10 20) * (Real.logb 10 5) ) = (17 / 9) := 
by
  sorry

end expression_I_evaluation_expression_II_evaluation_l105_105704


namespace line_x_intercept_l105_105494

theorem line_x_intercept (P Q : ℝ × ℝ) (hP : P = (2, 3)) (hQ : Q = (6, 7)) :
  ∃ x, (x, 0) = (-1, 0) ∧ ∃ (m : ℝ), m = (Q.2 - P.2) / (Q.1 - P.1) ∧ ∀ (x y : ℝ), y = m * (x - P.1) + P.2 := 
  sorry

end line_x_intercept_l105_105494


namespace monotonic_decreasing_interval_l105_105047

noncomputable def f (x : ℝ) : ℝ := x - Real.log x

theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 1) → f x < f (x + 1) := 
by 
  -- sorry is used because the actual proof is not required
  sorry

end monotonic_decreasing_interval_l105_105047


namespace max_discount_rate_l105_105656

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l105_105656


namespace chair_cost_l105_105030

-- Define the conditions
def total_spent : ℕ := 56
def table_cost : ℕ := 34
def num_chairs : ℕ := 2

-- Define the statement we need to prove
theorem chair_cost :
  ∃ (chair_cost : ℕ), 2 * chair_cost + table_cost = total_spent ∧ chair_cost = 11 :=
by
  use 11
  split
  sorry -- proof goes here, skipped as per instructions

end chair_cost_l105_105030


namespace horner_method_v2_l105_105714

def f(x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.reverse.foldl (λ acc c => acc * x + c) 0

theorem horner_method_v2 :
  horner_eval [1, 5, 10, 10, 5, 1] 2 = 24 :=
by
  sorry

end horner_method_v2_l105_105714


namespace hairstylist_earnings_per_week_l105_105852

theorem hairstylist_earnings_per_week :
  let cost_normal := 5
  let cost_special := 6
  let cost_trendy := 8
  let haircuts_normal := 5
  let haircuts_special := 3
  let haircuts_trendy := 2
  let days_per_week := 7
  let daily_earnings := cost_normal * haircuts_normal + cost_special * haircuts_special + cost_trendy * haircuts_trendy
  let weekly_earnings := daily_earnings * days_per_week
  weekly_earnings = 413 := sorry

end hairstylist_earnings_per_week_l105_105852


namespace max_discount_rate_l105_105666

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l105_105666


namespace max_discount_rate_l105_105634

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l105_105634


namespace total_cages_used_l105_105681

def num_puppies : Nat := 45
def num_adult_dogs : Nat := 30
def num_kittens : Nat := 25

def puppies_sold : Nat := 39
def adult_dogs_sold : Nat := 15
def kittens_sold : Nat := 10

def cage_capacity_puppies : Nat := 3
def cage_capacity_adult_dogs : Nat := 2
def cage_capacity_kittens : Nat := 2

def remaining_puppies : Nat := num_puppies - puppies_sold
def remaining_adult_dogs : Nat := num_adult_dogs - adult_dogs_sold
def remaining_kittens : Nat := num_kittens - kittens_sold

def cages_for_puppies : Nat := (remaining_puppies + cage_capacity_puppies - 1) / cage_capacity_puppies
def cages_for_adult_dogs : Nat := (remaining_adult_dogs + cage_capacity_adult_dogs - 1) / cage_capacity_adult_dogs
def cages_for_kittens : Nat := (remaining_kittens + cage_capacity_kittens - 1) / cage_capacity_kittens

def total_cages : Nat := cages_for_puppies + cages_for_adult_dogs + cages_for_kittens

-- Theorem stating the final goal
theorem total_cages_used : total_cages = 18 := by
  sorry

end total_cages_used_l105_105681


namespace number_difference_l105_105083

theorem number_difference:
  ∀ (number : ℝ), 0.30 * number = 63.0000000000001 →
  (3 / 7) * number - 0.40 * number = 6.00000000000006 := by
  sorry

end number_difference_l105_105083


namespace thieves_cloth_equation_l105_105163

theorem thieves_cloth_equation (x y : ℤ) 
  (h1 : y = 6 * x + 5)
  (h2 : y = 7 * x - 8) :
  6 * x + 5 = 7 * x - 8 :=
by
  sorry

end thieves_cloth_equation_l105_105163


namespace four_painters_workdays_l105_105559

theorem four_painters_workdays :
  (∃ (c : ℝ), ∀ (n : ℝ) (d : ℝ), n * d = c) →
  (p5 : ℝ) (d5 : ℝ) (p5 * d5 = 7.5) →
  ∀ D : ℝ, 4 * D = 7.5 →
  D = (1 + 7/8) := 
by {
  sorry
}

end four_painters_workdays_l105_105559


namespace percent_of_liquidX_in_solutionB_l105_105024

theorem percent_of_liquidX_in_solutionB (P : ℝ) (h₁ : 0.8 / 100 = 0.008) 
(h₂ : 1.5 / 100 = 0.015) 
(h₃ : 300 * 0.008 = 2.4) 
(h₄ : 1000 * 0.015 = 15) 
(h₅ : 15 - 2.4 = 12.6) 
(h₆ : 12.6 / 700 = P) : 
P * 100 = 1.8 :=
by sorry

end percent_of_liquidX_in_solutionB_l105_105024


namespace arithmetic_sequence_problem_l105_105168

variable (a : ℕ → ℤ) -- The arithmetic sequence as a function from natural numbers to integers
variable (S : ℕ → ℤ) -- Sum of the first n terms of the sequence

-- Conditions
variable (h1 : S 8 = 4 * a 3) -- Sum of the first 8 terms is 4 times the third term
variable (h2 : a 7 = -2)      -- The seventh term is -2

-- Proven Goal
theorem arithmetic_sequence_problem : a 9 = -6 := 
by sorry -- This is a placeholder for the proof

end arithmetic_sequence_problem_l105_105168


namespace middle_number_is_10_l105_105602

theorem middle_number_is_10 (A B C : ℝ) (h1 : B - C = A - B) (h2 : A * B = 85) (h3 : B * C = 115) : B = 10 :=
by
  sorry

end middle_number_is_10_l105_105602


namespace fred_change_l105_105176

theorem fred_change (ticket_price : ℝ) (tickets_count : ℕ) (borrowed_movie_cost : ℝ) (paid_amount : ℝ) :
  ticket_price = 5.92 →
  tickets_count = 2 →
  borrowed_movie_cost = 6.79 →
  paid_amount = 20 →
  let total_cost := tickets_count * ticket_price + borrowed_movie_cost in
  let change := paid_amount - total_cost in
  change = 1.37 :=
begin
  intros,
  sorry
end

end fred_change_l105_105176


namespace opposite_of_a_is_2022_l105_105740

theorem opposite_of_a_is_2022 (a : Int) (h : -a = -2022) : a = 2022 := by
  sorry

end opposite_of_a_is_2022_l105_105740


namespace calculate_expression_l105_105234

theorem calculate_expression : (0.25)^(-0.5) + (1/27)^(-1/3) - 625^(0.25) = 0 := 
by 
  sorry

end calculate_expression_l105_105234


namespace lines_parallel_coeff_l105_105285

theorem lines_parallel_coeff (a : ℝ) :
  (∀ x y: ℝ, a * x + 2 * y = 0 → 3 * x + (a + 1) * y + 1 = 0) ↔ (a = -3 ∨ a = 2) :=
by
  sorry

end lines_parallel_coeff_l105_105285


namespace solve_consecutive_integers_solve_consecutive_even_integers_l105_105722

-- Conditions: x, y, z, w are positive integers and x + y + z + w = 46.
def consecutive_integers_solution (x y z w : ℕ) : Prop :=
  x < y ∧ y < z ∧ z < w ∧ (x + 1 = y) ∧ (y + 1 = z) ∧ (z + 1 = w) ∧ (x + y + z + w = 46)

def consecutive_even_integers_solution (x y z w : ℕ) : Prop :=
  x < y ∧ y < z ∧ z < w ∧ (x + 2 = y) ∧ (y + 2 = z) ∧ (z + 2 = w) ∧ (x + y + z + w = 46)

-- Proof that consecutive integers can solve the equation II (x + y + z + w = 46)
theorem solve_consecutive_integers : ∃ x y z w : ℕ, consecutive_integers_solution x y z w :=
sorry

-- Proof that consecutive even integers can solve the equation II (x + y + z + w = 46)
theorem solve_consecutive_even_integers : ∃ x y z w : ℕ, consecutive_even_integers_solution x y z w :=
sorry

end solve_consecutive_integers_solve_consecutive_even_integers_l105_105722


namespace abs_a_lt_abs_b_add_abs_c_l105_105738

theorem abs_a_lt_abs_b_add_abs_c (a b c : ℝ) (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end abs_a_lt_abs_b_add_abs_c_l105_105738


namespace angle_bc_l105_105267

variables (a b c : ℝ → ℝ → Prop) (theta : ℝ)

-- Definitions of parallelism and angle conditions
def parallel (x y : ℝ → ℝ → Prop) : Prop := ∀ p q r s : ℝ, x p q → y r s → p - q = r - s

def angle_between (x y : ℝ → ℝ → Prop) (θ : ℝ) : Prop := sorry  -- Assume we have a definition for angle between lines

-- Given conditions
axiom parallel_ab : parallel a b
axiom angle_ac : angle_between a c theta

-- Theorem statement
theorem angle_bc : angle_between b c theta :=
sorry

end angle_bc_l105_105267


namespace find_difference_of_max_and_min_values_l105_105966

noncomputable def v (a b : Int) : Int := a * (-4) + b

theorem find_difference_of_max_and_min_values :
  let v0 := 3
  let v1 := v v0 12
  let v2 := v v1 6
  let v3 := v v2 10
  let v4 := v v3 (-8)
  (max (max (max (max v0 v1) v2) v3) v4) - (min (min (min (min v0 v1) v2) v3) v4) = 62 :=
by
  sorry

end find_difference_of_max_and_min_values_l105_105966


namespace rectangle_area_l105_105994

theorem rectangle_area (y : ℝ) (w : ℝ) (h : w > 0) (h_diag : y^2 = 10 * w^2) : 
  (3 * w)^2 * w = 3 * (y^2 / 10) :=
by sorry

end rectangle_area_l105_105994


namespace dodecahedron_edge_probability_l105_105212

def numVertices := 20
def pairsChosen := Nat.choose 20 2  -- Calculates combination (20 choose 2)
def edgesPerVertex := 3
def numEdges := (numVertices * edgesPerVertex) / 2
def probability : ℚ := numEdges / pairsChosen

theorem dodecahedron_edge_probability :
  probability = 3 / 19 :=
by
  -- The proof is skipped as per the instructions
  sorry

end dodecahedron_edge_probability_l105_105212


namespace units_digit_n_l105_105251

theorem units_digit_n (m n : ℕ) (hm : m % 10 = 9) (h : m * n = 18^5) : n % 10 = 2 :=
sorry

end units_digit_n_l105_105251


namespace fraction_sum_reciprocal_ge_two_l105_105579

theorem fraction_sum_reciprocal_ge_two (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : 
  (a / b) + (b / a) ≥ 2 :=
sorry

end fraction_sum_reciprocal_ge_two_l105_105579


namespace rita_bought_5_dresses_l105_105941

def pants_cost := 3 * 12
def jackets_cost := 4 * 30
def total_cost_pants_jackets := pants_cost + jackets_cost
def amount_spent := 400 - 139
def total_cost_dresses := amount_spent - total_cost_pants_jackets - 5
def number_of_dresses := total_cost_dresses / 20

theorem rita_bought_5_dresses : number_of_dresses = 5 :=
by sorry

end rita_bought_5_dresses_l105_105941


namespace sum_of_decimals_as_fraction_l105_105875

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 733 / 3125 := by
  sorry

end sum_of_decimals_as_fraction_l105_105875


namespace solve_system_l105_105173

theorem solve_system (x y z : ℤ) 
  (h1 : x + 3 * y = 20)
  (h2 : x + y + z = 25)
  (h3 : x - z = 5) : 
  x = 14 ∧ y = 2 ∧ z = 9 := 
  sorry

end solve_system_l105_105173


namespace max_discount_rate_l105_105650

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l105_105650


namespace portrait_is_in_Silver_l105_105208

def Gold_inscription (located_in : String → Prop) : Prop := located_in "Gold"
def Silver_inscription (located_in : String → Prop) : Prop := ¬located_in "Silver"
def Lead_inscription (located_in : String → Prop) : Prop := ¬located_in "Gold"

def is_true (inscription : Prop) : Prop := inscription
def is_false (inscription : Prop) : Prop := ¬inscription

noncomputable def portrait_in_Silver_Given_Statements : Prop :=
  ∃ located_in : String → Prop,
    (is_true (Gold_inscription located_in) ∨ is_true (Silver_inscription located_in) ∨ is_true (Lead_inscription located_in)) ∧
    (is_false (Gold_inscription located_in) ∨ is_false (Silver_inscription located_in) ∨ is_false (Lead_inscription located_in)) ∧
    located_in "Silver"

theorem portrait_is_in_Silver : portrait_in_Silver_Given_Statements :=
by {
    sorry
}

end portrait_is_in_Silver_l105_105208


namespace num_mittens_per_box_eq_six_l105_105378

theorem num_mittens_per_box_eq_six 
    (num_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ)
    (h1 : num_boxes = 4) (h2 : scarves_per_box = 2) (h3 : total_clothing = 32) :
    (total_clothing - num_boxes * scarves_per_box) / num_boxes = 6 :=
by
  sorry

end num_mittens_per_box_eq_six_l105_105378


namespace one_greater_than_one_l105_105179

theorem one_greater_than_one (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) : a > 1 ∨ b > 1 ∨ c > 1 :=
by
  sorry

end one_greater_than_one_l105_105179


namespace probability_last_digit_7_l105_105757

def lastDigit (n: ℕ) : ℕ := n % 10

theorem probability_last_digit_7 (a : ℕ) 
  (h1 : a ∈ finset.range 101 \ finset.singleton 0) :
  let prob := (finset.card (finset.filter (λ x, lastDigit (3^x) = 7) (finset.range 101))) / (finset.card (finset.range 101)) in
  prob = (1 / 4 : ℚ) :=
by
  sorry

end probability_last_digit_7_l105_105757


namespace polynomial_integer_values_l105_105276

theorem polynomial_integer_values (a b c d : ℤ) (h1 : ∃ (n : ℤ), n = (a * (-1)^3 + b * (-1)^2 - c * (-1) - d))
  (h2 : ∃ (n : ℤ), n = (a * 0^3 + b * 0^2 - c * 0 - d))
  (h3 : ∃ (n : ℤ), n = (a * 1^3 + b * 1^2 - c * 1 - d))
  (h4 : ∃ (n : ℤ), n = (a * 2^3 + b * 2^2 - c * 2 - d)) :
  ∀ x : ℤ, ∃ m : ℤ, m = a * x^3 + b * x^2 - c * x - d :=
by {
  -- proof goes here
  sorry
}

end polynomial_integer_values_l105_105276


namespace find_a_given_conditions_l105_105905

theorem find_a_given_conditions (a : ℤ)
  (hA : ∃ (x : ℤ), x = 12 ∨ x = a^2 + 4 * a ∨ x = a - 2)
  (hA_contains_minus3 : ∃ (x : ℤ), (-3 = x) ∧ (x = 12 ∨ x = a^2 + 4 * a ∨ x = a - 2)) : a = -3 := 
by
  sorry

end find_a_given_conditions_l105_105905


namespace maximum_discount_rate_l105_105643

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l105_105643


namespace opposite_of_neg_two_is_two_l105_105788

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l105_105788


namespace highest_a_value_l105_105299

theorem highest_a_value (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 143) : a = 23 :=
sorry

end highest_a_value_l105_105299


namespace juan_distance_l105_105166

def running_time : ℝ := 80.0
def speed : ℝ := 10.0
def distance : ℝ := running_time * speed

theorem juan_distance :
  distance = 800.0 :=
by
  sorry

end juan_distance_l105_105166


namespace find_x_l105_105854

theorem find_x (x : ℝ) (hx : x > 0) (condition : (2 * x / 100) * x = 10) : x = 10 * Real.sqrt 5 :=
by
  sorry

end find_x_l105_105854


namespace solve_riccati_eqn_l105_105322

noncomputable def riccati_solution (C : ℝ) : ℝ → ℝ :=
  λ x, real.exp x - (1 / (x + C))

theorem solve_riccati_eqn (C : ℝ) :
  ∀ y, (∃ x, y = real.exp x + (real.exp x - (1 / (x + C)))) →
       (deriv y - y^2 + 2 * real.exp x * y = real.exp (2 * x) + real.exp x) :=
begin
  sorry
end

end solve_riccati_eqn_l105_105322


namespace number_of_equilateral_triangles_in_lattice_l105_105109

-- Definitions representing the conditions of the problem
def is_unit_distance (a b : ℕ) : Prop :=
  true -- Assume true as we are not focusing on the definition

def expanded_hexagonal_lattice (p : ℕ) : Prop :=
  true -- Assume true as the specific construction details are abstracted

-- The target theorem statement
theorem number_of_equilateral_triangles_in_lattice 
  (lattice : ℕ → Prop) (dist : ℕ → ℕ → Prop) 
  (h₁ : ∀ p, lattice p → dist p p) 
  (h₂ : ∀ p, (expanded_hexagonal_lattice p) ↔ lattice p ∧ dist p p) : 
  ∃ n, n = 32 :=
by 
  existsi 32
  sorry

end number_of_equilateral_triangles_in_lattice_l105_105109


namespace opposite_of_neg2_l105_105804

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l105_105804


namespace sum_of_squares_l105_105937

-- Define the proposition as a universal statement 
theorem sum_of_squares (a b : ℝ) : a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by
  sorry

end sum_of_squares_l105_105937


namespace remainder_of_37_div_8_is_5_l105_105135

theorem remainder_of_37_div_8_is_5 : ∃ A B : ℤ, 37 = 8 * A + B ∧ 0 ≤ B ∧ B < 8 ∧ B = 5 := 
by
  sorry

end remainder_of_37_div_8_is_5_l105_105135


namespace maximum_discount_rate_l105_105647

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l105_105647


namespace min_colored_cells_65x65_l105_105214

def grid_size : ℕ := 65
def total_cells : ℕ := grid_size * grid_size

-- Define a function that calculates the minimum number of colored cells needed
noncomputable def min_colored_cells_needed (N: ℕ) : ℕ := (N * N) / 3

-- The main theorem stating the proof problem
theorem min_colored_cells_65x65 (H: grid_size = 65) : 
  min_colored_cells_needed grid_size = 1408 :=
by {
  sorry
}

end min_colored_cells_65x65_l105_105214


namespace max_crosses_4x10_proof_l105_105841

def max_crosses_4x10 (table : Matrix ℕ ℕ Bool) : ℕ :=
  sorry -- Placeholder for actual function implementation

theorem max_crosses_4x10_proof (table : Matrix ℕ ℕ Bool) (h : ∀ i < 4, ∃ j < 10, table i j = tt) :
  max_crosses_4x10 table = 30 :=
sorry

end max_crosses_4x10_proof_l105_105841


namespace sum_of_decimals_is_fraction_l105_105870

theorem sum_of_decimals_is_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 1466 / 6250 := 
by
  sorry

end sum_of_decimals_is_fraction_l105_105870


namespace max_discount_rate_l105_105665

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l105_105665


namespace who_made_statements_and_fate_l105_105470

namespace IvanTsarevichProblem

-- Define the characters and their behaviors
inductive Animal
| Bear : Animal
| Fox : Animal
| Wolf : Animal

def always_true (s : Prop) : Prop := s
def always_false (s : Prop) : Prop := ¬s
def alternates (s1 s2 : Prop) : Prop := s1 ∧ ¬s2

-- Statements made by the animals
def statement1 (save_die : Bool) : Prop := save_die = true
def statement2 (safe_sound_save : Bool) : Prop := safe_sound_save = true
def statement3 (safe_lose : Bool) : Prop := safe_lose = true

-- Analyze truth based on behaviors
noncomputable def belongs_to (a : Animal) (s : Prop) : Prop :=
  match a with
  | Animal.Bear => always_true s
  | Animal.Fox => always_false s
  | Animal.Wolf =>
    match s with
    | ss => alternates (ss = true) (ss = false)

-- Given conditions
axiom h1 : statement1 false -- Fox lies, so "You will save the horse. But you will die." is false
axiom h2 : statement2 false -- Wolf alternates, so "You will stay safe and sound. And you will save the horse." is a mix
axiom h3 : statement3 true  -- Bear tells the truth, so "You will survive. But you will lose the horse." is true

-- Conclusion: Animal who made each statement
theorem who_made_statements_and_fate : 
  belongs_to Animal.Fox (statement1 false) ∧ 
  belongs_to Animal.Wolf (statement2 false) ∧ 
  belongs_to Animal.Bear (statement3 true) ∧ 
  (¬safe_lose) := sorry

end IvanTsarevichProblem

end who_made_statements_and_fate_l105_105470


namespace maximum_discount_rate_l105_105631

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l105_105631


namespace triangle_inequality_at_vertex_l105_105428

-- Define the edge lengths of the tetrahedron and the common vertex label
variables {a b c d e f S : ℝ}

-- Conditions for the edge lengths and vertex label
axiom edge_lengths :
  a + b + c = S ∧
  a + d + e = S ∧
  b + d + f = S ∧
  c + e + f = S

-- The theorem to be proven
theorem triangle_inequality_at_vertex :
  a + b + c = S →
  a + d + e = S →
  b + d + f = S →
  c + e + f = S →
  (a ≤ b + c) ∧
  (b ≤ c + a) ∧
  (c ≤ a + b) ∧
  (a ≤ d + e) ∧
  (d ≤ e + a) ∧
  (e ≤ a + d) ∧
  (b ≤ d + f) ∧
  (d ≤ f + b) ∧
  (f ≤ b + d) ∧
  (c ≤ e + f) ∧
  (e ≤ f + c) ∧
  (f ≤ c + e) :=
sorry

end triangle_inequality_at_vertex_l105_105428


namespace marbles_jack_gave_l105_105016

-- Definitions based on conditions
def initial_marbles : ℕ := 22
def final_marbles : ℕ := 42

-- Theorem stating that the difference between final and initial marbles Josh collected is the marbles Jack gave
theorem marbles_jack_gave :
  final_marbles - initial_marbles = 20 :=
  sorry

end marbles_jack_gave_l105_105016


namespace shark_sightings_in_cape_may_l105_105239

theorem shark_sightings_in_cape_may (x : ℕ) (hx : x + (2 * x - 8) = 40) : 2 * x - 8 = 24 := 
by 
  sorry

end shark_sightings_in_cape_may_l105_105239


namespace hyperbola_eccentricity_squared_l105_105720

/-- Given that F is the right focus of the hyperbola 
    \( C: \frac{x^2}{a^2} - \frac{y^2}{b^2} = 1 \) with \( a > 0 \) and \( b > 0 \), 
    a line perpendicular to the x-axis is drawn through point F, 
    intersecting one asymptote of the hyperbola at point M. 
    If \( |FM| = 2a \), denote the eccentricity of the hyperbola as \( e \). 
    Prove that \( e^2 = \frac{1 + \sqrt{17}}{2} \).
 -/
theorem hyperbola_eccentricity_squared (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3: c^2 = a^2 + b^2) (h4: b * c = 2 * a^2) : 
  (c / a)^2 = (1 + Real.sqrt 17) / 2 := 
sorry

end hyperbola_eccentricity_squared_l105_105720


namespace sum_of_digits_of_N_l105_105685

theorem sum_of_digits_of_N (N : ℕ) (hN : N * (N + 1) / 2 = 3003) :
  (Nat.digits 10 N).sum = 14 :=
sorry

end sum_of_digits_of_N_l105_105685


namespace question1_question2_question3_l105_105035

-- Define the scores and relevant statistics for seventh and eighth grades
def seventh_grade_scores : List ℕ := [96, 86, 96, 86, 99, 96, 90, 100, 89, 82]
def eighth_grade_C_scores : List ℕ := [94, 90, 92]
def total_eighth_grade_students : ℕ := 800

def a := 40
def b := 93
def c := 96

-- Define given statistics from the table
def seventh_grade_mean := 92
def seventh_grade_variance := 34.6
def eighth_grade_mean := 91
def eighth_grade_median := 93
def eighth_grade_mode := 100
def eighth_grade_variance := 50.4

-- Proof for question 1
theorem question1 : (a = 40) ∧ (b = 93) ∧ (c = 96) :=
by sorry

-- Proof for question 2 (stability comparison)
theorem question2 : seventh_grade_variance < eighth_grade_variance :=
by sorry

-- Proof for question 3 (estimating number of excellent students)
theorem question3 : (7 / 10 : ℝ) * total_eighth_grade_students = 560 :=
by sorry

end question1_question2_question3_l105_105035


namespace valid_m_values_l105_105868

theorem valid_m_values :
  ∃ (m : ℕ), (m ∣ 720) ∧ (m ≠ 1) ∧ (m ≠ 720) ∧ ((720 / m) > 1) ∧ ((30 - 2) = 28) := 
sorry

end valid_m_values_l105_105868


namespace max_discount_rate_l105_105637

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l105_105637


namespace points_among_transformations_within_square_l105_105596

def projection_side1 (A : ℝ × ℝ) : ℝ × ℝ := (A.1, 2 - A.2)
def projection_side2 (A : ℝ × ℝ) : ℝ × ℝ := (-A.1, A.2)
def projection_side3 (A : ℝ × ℝ) : ℝ × ℝ := (A.1, -A.2)
def projection_side4 (A : ℝ × ℝ) : ℝ × ℝ := (2 - A.1, A.2)

def within_square (A : ℝ × ℝ) : Prop := 
  0 ≤ A.1 ∧ A.1 ≤ 1 ∧ 0 ≤ A.2 ∧ A.2 ≤ 1

theorem points_among_transformations_within_square (A : ℝ × ℝ)
  (H1 : within_square A)
  (H2 : within_square (projection_side1 A))
  (H3 : within_square (projection_side2 (projection_side1 A)))
  (H4 : within_square (projection_side3 (projection_side2 (projection_side1 A))))
  (H5 : within_square (projection_side4 (projection_side3 (projection_side2 (projection_side1 A))))) :
  A = (1 / 3, 1 / 3) := sorry

end points_among_transformations_within_square_l105_105596


namespace count_non_representable_as_diff_of_squares_l105_105726

theorem count_non_representable_as_diff_of_squares :
  let count := (Finset.filter (fun n => ∃ k, n = 4 * k + 2 ∧ 1 ≤ n ∧ n ≤ 1000) (Finset.range 1001)).card in
  count = 250 :=
by
  sorry

end count_non_representable_as_diff_of_squares_l105_105726


namespace total_amount_l105_105151

theorem total_amount (a b c total first : ℕ)
  (h1 : a = 1 / 2) (h2 : b = 2 / 3) (h3 : c = 3 / 4)
  (h4 : first = 204)
  (ratio_sum : a * 12 + b * 12 + c * 12 = 23)
  (first_ratio : a * 12 = 6) :
  total = 23 * (first / 6) → total = 782 :=
by 
  sorry

end total_amount_l105_105151


namespace product_divisible_by_14_l105_105922

theorem product_divisible_by_14 (a b c d : ℤ) (h : 7 * a + 8 * b = 14 * c + 28 * d) : 14 ∣ a * b := 
sorry

end product_divisible_by_14_l105_105922


namespace sums_ratio_l105_105857

theorem sums_ratio (total_sums : ℕ) (sums_right : ℕ) (sums_wrong: ℕ) (h1 : total_sums = 24) (h2 : sums_right = 8) (h3 : sums_wrong = total_sums - sums_right) :
  sums_wrong / Nat.gcd sums_wrong sums_right = 2 ∧ sums_right / Nat.gcd sums_wrong sums_right = 1 := by
  sorry

end sums_ratio_l105_105857


namespace even_k_l105_105923

theorem even_k :
  ∀ (a b n k : ℕ),
  1 ≤ a → 1 ≤ b → 0 < n →
  2^n - 1 = a * b →
  (a * b + a - b - 1) % 2^k = 0 →
  (a * b + a - b - 1) % 2^(k+1) ≠ 0 →
  Even k :=
by
  intros a b n k ha hb hn h1 h2 h3
  sorry

end even_k_l105_105923


namespace geometric_sequence_sum_l105_105309

theorem geometric_sequence_sum (a1 r : ℝ) (S : ℕ → ℝ) :
  S 2 = 3 → S 4 = 15 →
  (∀ n, S n = a1 * (1 - r^n) / (1 - r)) → S 6 = 63 :=
by
  intros hS2 hS4 hSn
  sorry

end geometric_sequence_sum_l105_105309


namespace find_cost_prices_l105_105373

-- These represent the given selling prices of the items.
def SP_computer_table : ℝ := 3600
def SP_office_chair : ℝ := 5000
def SP_bookshelf : ℝ := 1700

-- These represent the percentage markups and discounts as multipliers.
def markup_computer_table : ℝ := 1.20
def markup_office_chair : ℝ := 1.25
def discount_bookshelf : ℝ := 0.85

-- The problem requires us to find the cost prices. We will define these as variables.
variable (C O B : ℝ)

theorem find_cost_prices :
  (SP_computer_table = C * markup_computer_table) ∧
  (SP_office_chair = O * markup_office_chair) ∧
  (SP_bookshelf = B * discount_bookshelf) →
  (C = 3000) ∧ (O = 4000) ∧ (B = 2000) :=
by
  sorry

end find_cost_prices_l105_105373


namespace problem_l105_105379

def expr : ℤ := 7^2 - 4 * 5 + 2^2

theorem problem : expr = 33 := by
  sorry

end problem_l105_105379


namespace ratio_of_votes_l105_105858

theorem ratio_of_votes (up_votes down_votes : ℕ) (h_up : up_votes = 18) (h_down : down_votes = 4) : (up_votes / Nat.gcd up_votes down_votes) = 9 ∧ (down_votes / Nat.gcd up_votes down_votes) = 2 :=
by
  sorry

end ratio_of_votes_l105_105858


namespace solve_for_x_l105_105252

-- Definitions of the conditions
def condition (x : ℚ) : Prop :=
  (x^2 - 6 * x + 8) / (x^2 - 9 * x + 14) = (x^2 - 3 * x - 18) / (x^2 - 2 * x - 24)

-- Statement of the theorem
theorem solve_for_x (x : ℚ) (h : condition x) : x = -5 / 4 :=
by 
  sorry

end solve_for_x_l105_105252


namespace geometric_sequence_sum_10_l105_105751

theorem geometric_sequence_sum_10 (a : ℕ) (r : ℕ) (h : r = 2) (sum5 : a + r * a + r^2 * a + r^3 * a + r^4 * a = 1) : 
    a * (1 - r^10) / (1 - r) = 33 := 
by 
    sorry

end geometric_sequence_sum_10_l105_105751


namespace brenda_friends_l105_105376

def total_slices (pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ := pizzas * slices_per_pizza
def total_people (total_slices : ℕ) (slices_per_person : ℕ) : ℕ := total_slices / slices_per_person
def friends (total_people : ℕ) : ℕ := total_people - 1

theorem brenda_friends (pizzas : ℕ) (slices_per_pizza : ℕ) 
  (slices_per_person : ℕ) (pizzas_ordered : pizzas = 5) 
  (slices_per_pizza_value : slices_per_pizza = 4) 
  (slices_per_person_value : slices_per_person = 2) :
  friends (total_people (total_slices pizzas slices_per_pizza) slices_per_person) = 9 :=
by
  rw [pizzas_ordered, slices_per_pizza_value, slices_per_person_value]
  sorry

end brenda_friends_l105_105376


namespace probability_of_both_making_basket_l105_105607

noncomputable def P : Set ℕ → ℚ :=
  sorry

def A : Set ℕ := sorry
def B : Set ℕ := sorry

axiom prob_A : P A = 2 / 5
axiom prob_B : P B = 1 / 2
axiom independent : P (A ∩ B) = P A * P B

theorem probability_of_both_making_basket :
  P (A ∩ B) = 1 / 5 :=
by
  rw [independent, prob_A, prob_B]
  norm_num

end probability_of_both_making_basket_l105_105607


namespace solve_eq_n_fact_plus_n_eq_n_pow_k_l105_105864

theorem solve_eq_n_fact_plus_n_eq_n_pow_k :
  ∀ (n k : ℕ), 0 < n → 0 < k → (n! + n = n^k ↔ (n, k) = (2, 2) ∨ (n, k) = (3, 2) ∨ (n, k) = (5, 3)) :=
by
  sorry

end solve_eq_n_fact_plus_n_eq_n_pow_k_l105_105864


namespace opposite_of_neg_two_l105_105785

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l105_105785


namespace Rachel_total_score_l105_105224

theorem Rachel_total_score
    (points_per_treasure : ℕ)
    (treasures_first_level : ℕ)
    (treasures_second_level : ℕ)
    (h1 : points_per_treasure = 9)
    (h2 : treasures_first_level = 5)
    (h3 : treasures_second_level = 2) : 
    (points_per_treasure * treasures_first_level + points_per_treasure * treasures_second_level = 63) :=
by
    sorry

end Rachel_total_score_l105_105224


namespace maximum_discount_rate_l105_105630

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l105_105630


namespace number_of_zeros_l105_105275

noncomputable def f (a : ℝ) (x : ℝ) := a * x^2 - 2 * a * x + a + 1
noncomputable def g (b : ℝ) (x : ℝ) := b * x^3 - 2 * b * x^2 + b * x - 4 / 27

theorem number_of_zeros (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  ∃! (x : ℝ), g b (f a x) = 0 := sorry

end number_of_zeros_l105_105275


namespace sauna_max_couples_l105_105054

def max_couples (n : ℕ) : ℕ :=
  n - 1

theorem sauna_max_couples (n : ℕ) (rooms unlimited_capacity : Prop) (no_female_male_cohabsimult : Prop)
                          (males_shared_room_constraint females_shared_room_constraint : Prop)
                          (males_known_iff_wives_known : Prop) : max_couples n = n - 1 := 
  sorry

end sauna_max_couples_l105_105054


namespace prism_faces_l105_105466

theorem prism_faces (E V F : ℕ) (n : ℕ) 
  (h1 : E + V = 40) 
  (h2 : E = 3 * F - 6) 
  (h3 : V - E + F = 2)
  (h4 : V = 2 * n)
  : F = 10 := 
by
  sorry

end prism_faces_l105_105466


namespace largest_common_divisor_510_399_l105_105340

theorem largest_common_divisor_510_399 : ∃ d, d ∣ 510 ∧ d ∣ 399 ∧ ∀ e, e ∣ 510 ∧ e ∣ 399 → e ≤ d :=
begin
  use 57,
  split,
  { sorry },  -- placeholder for proof that 57 divides 510
  split,
  { sorry },  -- placeholder for proof that 57 divides 399
  { assume e h,
    sorry }  -- placeholder for proof that any common divisor must be <= 57
end

end largest_common_divisor_510_399_l105_105340


namespace perfect_square_iff_n_eq_5_l105_105248

theorem perfect_square_iff_n_eq_5 (n : ℕ) (h_pos : 0 < n) :
  ∃ m : ℕ, n * 2^(n-1) + 1 = m^2 ↔ n = 5 := by
  sorry

end perfect_square_iff_n_eq_5_l105_105248


namespace symmetric_y_axis_function_l105_105403

theorem symmetric_y_axis_function (f g : ℝ → ℝ) (h : ∀ (x : ℝ), g x = 3^x + 1) :
  (∀ x, f x = f (-x)) → (∀ x, f x = g (-x)) → (∀ x, f x = 3^(-x) + 1) :=
by
  intros h1 h2
  sorry

end symmetric_y_axis_function_l105_105403


namespace binomial_identity_l105_105223

open BigOperators

theorem binomial_identity :
  (∑ k in ({1, 3, 5} : Finset ℕ), if even k then 0 else 2 * binom 10 k) - binom 10 3 + (binom 10 5) / 2 = 2^4 := 
by
  sorry

end binomial_identity_l105_105223


namespace maximum_discount_rate_l105_105645

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l105_105645


namespace max_discount_rate_l105_105679

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l105_105679


namespace EllenBreadMakingTime_l105_105514

-- Definitions based on the given problem
def RisingTimeTypeA : ℕ → ℝ := λ n => n * 4
def BakingTimeTypeA : ℕ → ℝ := λ n => n * 2.5
def RisingTimeTypeB : ℕ → ℝ := λ n => n * 3.5
def BakingTimeTypeB : ℕ → ℝ := λ n => n * 3

def TotalTime (nA nB : ℕ) : ℝ :=
  (RisingTimeTypeA nA + BakingTimeTypeA nA) +
  (RisingTimeTypeB nB + BakingTimeTypeB nB)

theorem EllenBreadMakingTime :
  TotalTime 3 2 = 32.5 := by
  sorry

end EllenBreadMakingTime_l105_105514


namespace no_valid_bases_l105_105771

theorem no_valid_bases
  (x y : ℕ)
  (h1 : 4 * x + 9 = 4 * y + 1)
  (h2 : 4 * x^2 + 7 * x + 7 = 3 * y^2 + 2 * y + 9)
  (hx : x > 1)
  (hy : y > 1)
  : false :=
by
  sorry

end no_valid_bases_l105_105771


namespace toms_age_l105_105483

variable (T J : ℕ)

theorem toms_age :
  (J - 6 = 3 * (T - 6)) ∧ (J + 4 = 2 * (T + 4)) → T = 16 :=
by
  intros h
  sorry

end toms_age_l105_105483


namespace original_number_exists_l105_105736

theorem original_number_exists (x : ℤ) (h1 : x * 16 = 3408) (h2 : 0.016 * 2.13 = 0.03408) : x = 213 := 
by 
  sorry

end original_number_exists_l105_105736


namespace opposite_of_neg_2_is_2_l105_105810

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l105_105810


namespace fraction_power_calc_l105_105107

theorem fraction_power_calc : 
  (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
sorry

end fraction_power_calc_l105_105107


namespace cindy_pens_ratio_is_one_l105_105346

noncomputable def pens_owned_initial : ℕ := 25
noncomputable def pens_given_by_mike : ℕ := 22
noncomputable def pens_given_to_sharon : ℕ := 19
noncomputable def pens_owned_final : ℕ := 75

def pens_before_cindy (initial_pens mike_pens : ℕ) : ℕ := initial_pens + mike_pens
def pens_before_sharon (final_pens sharon_pens : ℕ) : ℕ := final_pens + sharon_pens
def pens_given_by_cindy (pens_before_sharon pens_before_cindy : ℕ) : ℕ := pens_before_sharon - pens_before_cindy
def ratio_pens_given_cindy (cindy_pens pens_before_cindy : ℕ) : ℚ := cindy_pens / pens_before_cindy

theorem cindy_pens_ratio_is_one :
    ratio_pens_given_cindy
        (pens_given_by_cindy (pens_before_sharon pens_owned_final pens_given_to_sharon)
                             (pens_before_cindy pens_owned_initial pens_given_by_mike))
        (pens_before_cindy pens_owned_initial pens_given_by_mike) = 1 := by
    sorry

end cindy_pens_ratio_is_one_l105_105346


namespace x_pow_twelve_l105_105148

theorem x_pow_twelve (x : ℝ) (h : x + 1/x = 3) : x^12 = 322 :=
sorry

end x_pow_twelve_l105_105148


namespace player_one_wins_l105_105600

theorem player_one_wins (initial_coins : ℕ) (h_initial : initial_coins = 2015) : 
  ∃ first_move : ℕ, (1 ≤ first_move ∧ first_move ≤ 99 ∧ first_move % 2 = 1) ∧ 
  (∀ move : ℕ, (2 ≤ move ∧ move ≤ 100 ∧ move % 2 = 0) → 
   ∃ next_move : ℕ, (1 ≤ next_move ∧ next_move ≤ 99 ∧ next_move % 2 = 1) → 
   initial_coins - first_move - move - next_move < 101) → first_move = 95 :=
by 
  sorry

end player_one_wins_l105_105600


namespace max_discount_rate_l105_105659

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l105_105659


namespace card_distribution_count_l105_105890

def card_distribution_ways : Nat := sorry

theorem card_distribution_count :
  card_distribution_ways = 9 := sorry

end card_distribution_count_l105_105890


namespace find_cd_l105_105343

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

theorem find_cd (c d : ℝ) 
  (h1 : g c d 2 = -7) 
  (h2 : g c d (-1) = -25) : 
  (c, d) = (2, 8) := 
by
  sorry

end find_cd_l105_105343


namespace exist_c_l105_105418

theorem exist_c (p : ℕ) (r : ℤ) (a b : ℤ) [Fact (Nat.Prime p)]
  (hp1 : r^7 ≡ 1 [ZMOD p])
  (hp2 : r + 1 - a^2 ≡ 0 [ZMOD p])
  (hp3 : r^2 + 1 - b^2 ≡ 0 [ZMOD p]) :
  ∃ c : ℤ, (r^3 + 1 - c^2) ≡ 0 [ZMOD p] :=
by
  sorry

end exist_c_l105_105418


namespace coefficient_x3_y4_in_expansion_l105_105065

theorem coefficient_x3_y4_in_expansion :
  let n := 7
  let a := 3
  let b := 4
  binom n a = 35 :=
by
  sorry

end coefficient_x3_y4_in_expansion_l105_105065


namespace remainder_div_by_3_not_divisible_by_9_l105_105341

theorem remainder_div_by_3 (x : ℕ) (h : x = 1493826) : x % 3 = 0 :=
by sorry

theorem not_divisible_by_9 (x : ℕ) (h : x = 1493826) : x % 9 ≠ 0 :=
by sorry

end remainder_div_by_3_not_divisible_by_9_l105_105341


namespace opposite_of_neg_two_is_two_l105_105814

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l105_105814


namespace max_value_inequality_l105_105421

theorem max_value_inequality (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  3 * x + 4 * y + 6 * z ≤ Real.sqrt 53 := by
  sorry

end max_value_inequality_l105_105421


namespace consecutive_number_other_17_l105_105962

theorem consecutive_number_other_17 (a b : ℕ) (h1 : b = 17) (h2 : a + b = 35) (h3 : a + b % 5 = 0) : a = 18 :=
sorry

end consecutive_number_other_17_l105_105962


namespace sample_size_is_100_l105_105964

-- Conditions:
def scores_from_students := 100
def sampling_method := "simple random sampling"
def goal := "statistical analysis of senior three students' exam performance"

-- Problem statement:
theorem sample_size_is_100 :
  scores_from_students = 100 →
  sampling_method = "simple random sampling" →
  goal = "statistical analysis of senior three students' exam performance" →
  scores_from_students = 100 := by
sorry

end sample_size_is_100_l105_105964


namespace nina_basketball_cards_l105_105574

theorem nina_basketball_cards (cost_toy cost_shirt cost_card total_spent : ℕ) (n_toys n_shirts n_cards n_packs_result : ℕ)
  (h1 : cost_toy = 10)
  (h2 : cost_shirt = 6)
  (h3 : cost_card = 5)
  (h4 : n_toys = 3)
  (h5 : n_shirts = 5)
  (h6 : total_spent = 70)
  (h7 : n_packs_result =  2)
  : (3 * cost_toy + 5 * cost_shirt + n_cards * cost_card = total_spent) → n_cards = n_packs_result :=
by
  sorry

end nina_basketball_cards_l105_105574


namespace unique_solution_l105_105390

-- Define the system of equations
def system_of_equations (m x y : ℝ) := 
  (m + 1) * x - y - 3 * m = 0 ∧ 4 * x + (m - 1) * y + 7 = 0

-- Define the determinant condition
def determinant_nonzero (m : ℝ) := m^2 + 3 ≠ 0

-- Theorem to prove there is exactly one solution
theorem unique_solution (m x y : ℝ) : 
  determinant_nonzero m → ∃! (x y : ℝ), system_of_equations m x y :=
by
  sorry

end unique_solution_l105_105390


namespace nina_total_money_l105_105983

def original_cost_widget (C : ℝ) : ℝ := C
def num_widgets_nina_can_buy_original (C : ℝ) : ℝ := 6
def num_widgets_nina_can_buy_reduced (C : ℝ) : ℝ := 8
def cost_reduction : ℝ := 1.5

theorem nina_total_money (C : ℝ) (hc : 6 * C = 8 * (C - cost_reduction)) : 
  6 * C = 36 :=
by
  sorry

end nina_total_money_l105_105983


namespace average_age_remains_l105_105950

theorem average_age_remains (total_age : ℕ) (leaving_age : ℕ) (remaining_people : ℕ) (initial_people_avg : ℕ) 
                            (total_age_eq : total_age = initial_people_avg * 8) 
                            (new_total_age : ℕ := total_age - leaving_age)
                            (new_avg : ℝ := new_total_age / remaining_people) :
  (initial_people_avg = 25) ∧ (leaving_age = 20) ∧ (remaining_people = 7) → new_avg = 180 / 7 := 
by
  sorry

end average_age_remains_l105_105950


namespace maximum_discount_rate_l105_105644

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l105_105644


namespace compare_y1_y2_l105_105438

-- Define the function
def f (x : ℝ) : ℝ := -3 * x + 1

-- Define the points
def y1 := f 1
def y2 := f 3

-- The theorem to be proved
theorem compare_y1_y2 : y1 > y2 :=
by
  -- Proof placeholder
  sorry

end compare_y1_y2_l105_105438


namespace angles_on_y_axis_l105_105595

theorem angles_on_y_axis :
  {θ : ℝ | ∃ k : ℤ, (θ = 2 * k * Real.pi + Real.pi / 2) ∨ (θ = 2 * k * Real.pi + 3 * Real.pi / 2)} =
  {θ : ℝ | ∃ n : ℤ, θ = n * Real.pi + Real.pi / 2} :=
by 
  sorry

end angles_on_y_axis_l105_105595


namespace third_test_point_l105_105749

noncomputable def test_points : ℝ × ℝ × ℝ :=
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  (x1, x2, x3)

theorem third_test_point :
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  x1 > x2 → x3 = 3.528 :=
by
  intros
  sorry

end third_test_point_l105_105749


namespace final_expression_l105_105152

theorem final_expression (y : ℝ) : (3 * (1 / 2 * (12 * y + 3))) = 18 * y + 4.5 :=
by
  sorry

end final_expression_l105_105152


namespace All_Yarns_are_Zorps_and_Xings_l105_105915

-- Define the basic properties
variables {α : Type}
variable (Zorp Xing Yarn Wit Vamp : α → Prop)

-- Given conditions
axiom all_Zorps_are_Xings : ∀ z, Zorp z → Xing z
axiom all_Yarns_are_Xings : ∀ y, Yarn y → Xing y
axiom all_Wits_are_Zorps : ∀ w, Wit w → Zorp w
axiom all_Yarns_are_Wits : ∀ y, Yarn y → Wit y
axiom all_Yarns_are_Vamps : ∀ y, Yarn y → Vamp y

-- Proof problem
theorem All_Yarns_are_Zorps_and_Xings : 
  ∀ y, Yarn y → (Zorp y ∧ Xing y) :=
sorry

end All_Yarns_are_Zorps_and_Xings_l105_105915


namespace angle_bisector_slope_l105_105588

theorem angle_bisector_slope (k : ℚ) : 
  (∀ x : ℚ, (y = 2 * x ∧ y = 4 * x) → (y = k * x)) → k = -12 / 7 :=
sorry

end angle_bisector_slope_l105_105588


namespace neznika_number_l105_105437

theorem neznika_number (N : ℕ) :
  10 ≤ N ∧ N ≤ 99 ∧
  (divisible_by N 3 ∧ divisible_by N 4 ∧ divisible_by N 5 ∧
   divisible_by N 9 ∧ divisible_by N 10 ∧ divisible_by N 15 ∧
   divisible_by N 18 ∧ divisible_by N 30) →
  (N = 36 ∨ N = 45 ∨ N = 72) :=
begin
  sorry
end

end neznika_number_l105_105437


namespace max_discount_rate_l105_105633

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l105_105633


namespace max_discount_rate_l105_105632

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l105_105632


namespace ratio_of_ages_is_six_l105_105862

-- Definitions of ages
def Cody_age : ℕ := 14
def Grandmother_age : ℕ := 84

-- The ratio we want to prove
def age_ratio : ℕ := Grandmother_age / Cody_age

-- The theorem stating the ratio is 6
theorem ratio_of_ages_is_six : age_ratio = 6 :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_ages_is_six_l105_105862


namespace sum_of_n_values_l105_105477

theorem sum_of_n_values : 
  ∑ n in {n : ℤ | ∃ (d : ℤ), d * (2 * n - 1) = 24}.toFinset, n = 2 :=
by
  sorry

end sum_of_n_values_l105_105477


namespace max_discount_rate_l105_105649

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l105_105649


namespace stock_and_bond_value_relation_l105_105572

-- Definitions for conditions
def more_valuable_shares : ℕ := 14
def less_valuable_shares : ℕ := 26
def face_value_bond : ℝ := 1000
def coupon_rate_bond : ℝ := 0.06
def discount_rate_bond : ℝ := 0.03
def total_assets_value : ℝ := 2106

-- Lean statement for the proof problem
theorem stock_and_bond_value_relation (x y : ℝ) 
    (h1 : face_value_bond * (1 - discount_rate_bond) = 970)
    (h2 : 27 * x + y = total_assets_value) :
    y = 2106 - 27 * x :=
by
  sorry

end stock_and_bond_value_relation_l105_105572


namespace smallest_n_l105_105840

theorem smallest_n (n : ℕ) (h : 10 - n ≥ 0) : 
  (9 / 10) * (8 / 9) * (7 / 8) * (6 / 7) * (5 / 6) * (4 / 5) < 0.5 → n = 6 :=
by
  sorry

end smallest_n_l105_105840


namespace complement_of_A_in_U_l105_105277

open Set

variable (U : Set ℕ) (A : Set ℕ)

theorem complement_of_A_in_U (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 6}) :
  (U \ A) = {1, 3, 5} := by 
  sorry

end complement_of_A_in_U_l105_105277


namespace students_in_grades_v_vi_l105_105327

theorem students_in_grades_v_vi (n a b c p q : ℕ) (h1 : n = 100*a + 10*b + c)
  (h2 : a * b * c = p) (h3 : (p / 10) * (p % 10) = q) : n = 144 :=
sorry

end students_in_grades_v_vi_l105_105327


namespace jane_paid_five_l105_105415

noncomputable def cost_of_apple : ℝ := 0.75
noncomputable def change_received : ℝ := 4.25
noncomputable def amount_paid : ℝ := cost_of_apple + change_received

theorem jane_paid_five : amount_paid = 5.00 :=
by
  sorry

end jane_paid_five_l105_105415


namespace bread_baked_on_monday_l105_105587

def loaves_wednesday : ℕ := 5
def loaves_thursday : ℕ := 7
def loaves_friday : ℕ := 10
def loaves_saturday : ℕ := 14
def loaves_sunday : ℕ := 19

def increment (n m : ℕ) : ℕ := m - n

theorem bread_baked_on_monday : 
  increment loaves_wednesday loaves_thursday = 2 →
  increment loaves_thursday loaves_friday = 3 →
  increment loaves_friday loaves_saturday = 4 →
  increment loaves_saturday loaves_sunday = 5 →
  loaves_sunday + 6 = 25 :=
by 
  sorry

end bread_baked_on_monday_l105_105587


namespace monotonic_decreasing_interval_l105_105048

variable (x : ℝ)

def f (x : ℝ) : ℝ := x - log x 

noncomputable def f' (x : ℝ) := deriv f x

theorem monotonic_decreasing_interval : ( {x : ℝ | 0 < x ∧ x < 1} = { x | x ∈ Ioo 0 1}) ↔ ( ∀ x, f' x < 0 ) :=
sorry

end monotonic_decreasing_interval_l105_105048


namespace sum_ratio_is_one_l105_105419

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)

-- Conditions
def arithmetic_sequence (a_n S_n : ℕ → ℝ) := ∀ n : ℕ, S_n = (n * (a_n 1 + a_n n)) / 2
def ratio_condition (a_n : ℕ → ℝ) := a_n 5 / a_n 3 = 5 / 9

-- The theorem to prove
theorem sum_ratio_is_one (a_n S_n : ℕ → ℝ) 
  [arithmetic_sequence a_n S_n] 
  [ratio_condition a_n] : 
  S_n 9 / S_n 5 = 1 := 
sorry

end sum_ratio_is_one_l105_105419


namespace increasing_interval_cos_2alpha_l105_105274

noncomputable theory

open Real

-- Definition of the given function f
def f (x : ℝ) : ℝ := cos x * (sin x + cos x) - 1 / 2

-- Statement of the first proof: Interval where f(x) is monotonically increasing
theorem increasing_interval (k : ℤ) : 
  ∃ (a b : ℝ), [a, b] = [k * π - 3 * π / 8, k * π + π / 8] ∧ ∀ x y, 
  x ∈ [a, b] → y ∈ [a, b] → x < y → f x < f y := 
sorry

-- Statement of the second proof: value of cos(2α) given a specific condition
theorem cos_2alpha (α : ℝ) 
  (hα : α ∈ Ioo (π / 8) (3 * π / 8)) 
  (h : f α = sqrt 2 / 6) : 
  cos (2 * α) = (sqrt 2 - 4) / 6 := 
sorry

end increasing_interval_cos_2alpha_l105_105274


namespace simplify_expression_l105_105768

theorem simplify_expression : (625:ℝ)^(1/4) * (256:ℝ)^(1/2) = 80 := 
by 
  sorry

end simplify_expression_l105_105768


namespace percentage_increase_ticket_price_l105_105417

-- Definitions for the conditions
def last_year_income := 100.0
def clubs_share_last_year := 0.10 * last_year_income
def rental_cost := 0.90 * last_year_income
def new_clubs_share := 0.20
def new_income := rental_cost / (1 - new_clubs_share)

-- Lean 4 theorem statement
theorem percentage_increase_ticket_price : 
  new_income = 112.5 → ((new_income - last_year_income) / last_year_income * 100) = 12.5 := 
by
  sorry

end percentage_increase_ticket_price_l105_105417


namespace lefty_jazz_non_basketball_l105_105046

-- Definitions
def total_members : ℕ := 30
def left_handed_members : ℕ := 12
def jazz_loving_members : ℕ := 20
def right_handed_non_jazz_non_basketball : ℕ := 5
def basketball_players : ℕ := 10
def left_handed_jazz_loving_basketball_players : ℕ := 3

-- Problem Statement: Prove the number of lefty jazz lovers who do not play basketball.
theorem lefty_jazz_non_basketball (x : ℕ) :
  (x + left_handed_jazz_loving_basketball_players) + (left_handed_members - x - left_handed_jazz_loving_basketball_players) + 
  (jazz_loving_members - x - left_handed_jazz_loving_basketball_players) + 
  right_handed_non_jazz_non_basketball + left_handed_jazz_loving_basketball_players = 
  total_members → x = 4 :=
by
  sorry

end lefty_jazz_non_basketball_l105_105046


namespace right_triangle_perimeter_l105_105356

theorem right_triangle_perimeter 
  (a b c : ℕ) (h : a = 11) (h1 : a * a + b * b = c * c) (h2 : a < c) : a + b + c = 132 :=
  sorry

end right_triangle_perimeter_l105_105356


namespace polynomial_nonnegative_l105_105440

theorem polynomial_nonnegative (x : ℝ) : x^4 - x^3 + 3x^2 - 2x + 2 ≥ 0 := 
by 
  sorry

end polynomial_nonnegative_l105_105440


namespace pieces_of_gum_per_nickel_l105_105316

-- Definitions based on the given conditions
def initial_nickels : ℕ := 5
def remaining_nickels : ℕ := 2
def total_gum_pieces : ℕ := 6

-- We need to prove that Quentavious gets 2 pieces of gum per nickel.
theorem pieces_of_gum_per_nickel 
  (initial_nickels remaining_nickels total_gum_pieces : ℕ)
  (h1 : initial_nickels = 5)
  (h2 : remaining_nickels = 2)
  (h3 : total_gum_pieces = 6) :
  total_gum_pieces / (initial_nickels - remaining_nickels) = 2 :=
by {
  sorry
}

end pieces_of_gum_per_nickel_l105_105316


namespace derivative_at_x0_l105_105305

open Function Filter

variable {α : Type*} {f : α → ℝ} {x_0 : α}

theorem derivative_at_x0 
  (h_diff : DifferentiableAt ℝ f x_0)
  (h_limit : tendsto (λ Δx : ℝ, (f (x_0 - 3 * Δx) - f x_0) / Δx) (𝓝 0) (𝓝 1)) :
  deriv f x_0 = -1/3 :=
by
  sorry

end derivative_at_x0_l105_105305


namespace sibling_age_difference_l105_105544

theorem sibling_age_difference 
  (x : ℕ) 
  (h : 3 * x + 2 * x + 1 * x = 90) : 
  3 * x - x = 30 := 
by 
  sorry

end sibling_age_difference_l105_105544


namespace total_rainbow_nerds_is_36_l105_105360

def purple_candies : ℕ := 10
def yellow_candies : ℕ := purple_candies + 4
def green_candies : ℕ := yellow_candies - 2
def total_candies : ℕ := purple_candies + yellow_candies + green_candies

theorem total_rainbow_nerds_is_36 : total_candies = 36 := by
  sorry

end total_rainbow_nerds_is_36_l105_105360


namespace smallest_positive_multiple_l105_105520

theorem smallest_positive_multiple (a : ℕ) (h₁ : a % 6 = 0) (h₂ : a % 15 = 0) : a = 30 :=
sorry

end smallest_positive_multiple_l105_105520


namespace domain_of_f_l105_105474

noncomputable def f (x : ℝ) := Real.log x / Real.log 6

noncomputable def g (x : ℝ) := Real.log x / Real.log 5

noncomputable def h (x : ℝ) := Real.log x / Real.log 3

open Set

theorem domain_of_f :
  (∀ x, x > 7776 → ∃ y, y = (h ∘ g ∘ f) x) :=
by
  sorry

end domain_of_f_l105_105474


namespace possible_numbers_l105_105432

theorem possible_numbers (N : ℕ) (h_digit : 10 ≤ N ∧ N ≤ 99)
  (h_claimed_divisors : ∀ d ∈ [3, 4, 5, 9, 10, 15, 18, 30], d ∣ N ∨ ¬ d ∣ N)
  (h_mistakes : Nat.countp (λ d, ¬ d ∣ N) [3, 4, 5, 9, 10, 15, 18, 30] = 4) :
  N = 36 ∨ N = 45 ∨ N = 72 :=
by
  sorry

end possible_numbers_l105_105432


namespace profit_percentage_l105_105347

theorem profit_percentage (cost_price selling_price : ℝ) (h₁ : cost_price = 32) (h₂ : selling_price = 56) : 
  ((selling_price - cost_price) / cost_price) * 100 = 75 :=
by
  sorry

end profit_percentage_l105_105347


namespace smallest_possible_z_l105_105410

theorem smallest_possible_z :
  ∃ (z : ℕ), (z = 6) ∧ 
  ∃ (u w x y : ℕ), u < w ∧ w < x ∧ x < y ∧ y < z ∧ 
  u.succ = w ∧ w.succ = x ∧ x.succ = y ∧ y.succ = z ∧ 
  u^3 + w^3 + x^3 + y^3 = z^3 :=
by
  use 6
  sorry

end smallest_possible_z_l105_105410


namespace julia_tuesday_kids_l105_105017

-- Definitions based on conditions
def kids_on_monday : ℕ := 11
def tuesday_more_than_monday : ℕ := 1

-- The main statement to be proved
theorem julia_tuesday_kids : (kids_on_monday + tuesday_more_than_monday) = 12 := by
  sorry

end julia_tuesday_kids_l105_105017


namespace sum_of_n_values_l105_105476

theorem sum_of_n_values (n_values : List ℤ) 
  (h : ∀ n ∈ n_values, ∃ k : ℤ, 24 = k * (2 * n - 1)) : n_values.sum = 2 :=
by
  -- Proof to be provided.
  sorry

end sum_of_n_values_l105_105476


namespace find_f2_g2_l105_105396

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x
def equation (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x - g x = x^3 + 2^(-x)

theorem find_f2_g2 (f g : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : odd_function g)
  (h3 : equation f g) :
  f 2 + g 2 = -2 :=
sorry

end find_f2_g2_l105_105396


namespace part1_part2_l105_105063

-- Define the operation * on integers
def op (a b : ℤ) : ℤ := a^2 - b + a * b

-- Prove that 2 * 3 = 7 given the defined operation
theorem part1 : op 2 3 = 7 := 
sorry

-- Prove that (-2) * (op 2 (-3)) = 1 given the defined operation
theorem part2 : op (-2) (op 2 (-3)) = 1 := 
sorry

end part1_part2_l105_105063


namespace roots_of_polynomial_inequality_l105_105568

theorem roots_of_polynomial_inequality :
  (∃ (p q r s : ℂ), (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) ∧
  (p * q * r * s = 3) ∧ (p*q + p*r + p*s + q*r + q*s + r*s = 11)) →
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3) :=
by
  sorry

end roots_of_polynomial_inequality_l105_105568


namespace hotel_charge_problem_l105_105352

theorem hotel_charge_problem (R G P : ℝ) 
  (h1 : P = 0.5 * R) 
  (h2 : P = 0.9 * G) : 
  (R - G) / G * 100 = 80 :=
by
  sorry

end hotel_charge_problem_l105_105352


namespace area_of_triangle_l105_105288

-- Definitions
variables {A B C : Type}
variables {i j k : ℕ}
variables (AB AC : ℝ)
variables (s t : ℝ)
variables (sinA : ℝ) (cosA : ℝ)

-- Conditions 
axiom sin_A : sinA = 4 / 5
axiom dot_product : s * t * cosA = 6

-- The problem theorem
theorem area_of_triangle : (1 / 2) * s * t * sinA = 4 :=
by
  sorry

end area_of_triangle_l105_105288


namespace sum_of_relatively_prime_integers_l105_105822

theorem sum_of_relatively_prime_integers (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
  (h3 : x * y + x + y = 154) (h4 : Nat.gcd x y = 1) (h5 : x < 30) (h6 : y < 30) : 
  x + y = 34 :=
sorry -- proof

end sum_of_relatively_prime_integers_l105_105822


namespace n_salary_eq_260_l105_105060

variables (m n : ℕ)
axiom total_salary : m + n = 572
axiom m_salary : m = 120 * n / 100

theorem n_salary_eq_260 : n = 260 :=
by
  sorry

end n_salary_eq_260_l105_105060


namespace range_of_m_l105_105533

noncomputable def f (m x : ℝ) : ℝ :=
  Real.log x + m / x

theorem range_of_m (m : ℝ) :
  (∀ (a b : ℝ), a > 0 → b > 0 → a ≠ b → (f m b - f m a) / (b - a) < 1) →
  m ≥ 1 / 4 :=
by
  sorry

end range_of_m_l105_105533


namespace percentage_neither_l105_105493

theorem percentage_neither (total_teachers high_blood_pressure heart_trouble both_conditions : ℕ)
  (h1 : total_teachers = 150)
  (h2 : high_blood_pressure = 90)
  (h3 : heart_trouble = 60)
  (h4 : both_conditions = 30) :
  100 * (total_teachers - (high_blood_pressure + heart_trouble - both_conditions)) / total_teachers = 20 :=
by
  sorry

end percentage_neither_l105_105493


namespace find_oranges_l105_105753

def A : ℕ := 3
def B : ℕ := 1

theorem find_oranges (O : ℕ) : A + B + O + (A + 4) + 10 * B + 2 * (A + 4) = 39 → O = 4 :=
by 
  intros h
  sorry

end find_oranges_l105_105753


namespace race_time_comparison_l105_105824

noncomputable def townSquare : ℝ := 3 / 4 -- distance of one lap in miles
noncomputable def laps : ℕ := 7 -- number of laps
noncomputable def totalDistance : ℝ := laps * townSquare -- total distance of the race in miles
noncomputable def thisYearTime : ℝ := 42 -- time taken by this year's winner in minutes
noncomputable def lastYearTime : ℝ := 47.25 -- time taken by last year's winner in minutes

noncomputable def thisYearPace : ℝ := thisYearTime / totalDistance -- pace of this year's winner in minutes per mile
noncomputable def lastYearPace : ℝ := lastYearTime / totalDistance -- pace of last year's winner in minutes per mile
noncomputable def timeDifference : ℝ := lastYearPace - thisYearPace -- the difference in pace

theorem race_time_comparison : timeDifference = 1 := by
  sorry

end race_time_comparison_l105_105824


namespace M_intersect_N_l105_105172

-- Definition of the sets M and N
def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≤ x}

-- Proposition to be proved
theorem M_intersect_N : M ∩ N = {0, 1} := 
by 
  sorry

end M_intersect_N_l105_105172


namespace student_correct_sums_l105_105997

-- Defining variables R and W along with the given conditions
variables (R W : ℕ)

-- Given conditions as Lean definitions
def condition1 := W = 5 * R
def condition2 := R + W = 180

-- Statement of the problem to prove R equals 30
theorem student_correct_sums :
  (W = 5 * R) → (R + W = 180) → R = 30 :=
by
  -- Import needed definitions and theorems from Mathlib
  sorry -- skipping the proof

end student_correct_sums_l105_105997


namespace simplify_expression_l105_105970

theorem simplify_expression : (2468 * 2468) / (2468 + 2468) = 1234 :=
by
  sorry

end simplify_expression_l105_105970


namespace triangle_angles_l105_105902

-- Define the problem and the conditions as Lean statements.
theorem triangle_angles (x y z : ℝ) 
  (h1 : y + 150 + 160 = 360)
  (h2 : z + 150 + 160 = 360)
  (h3 : x + y + z = 180) : 
  x = 80 ∧ y = 50 ∧ z = 50 := 
by 
  sorry

end triangle_angles_l105_105902


namespace num_positive_integers_l105_105523

-- Definitions
def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

-- Problem statement
theorem num_positive_integers (n : ℕ) (h : n = 2310) :
  (∃ count, count = 3 ∧ (∀ m : ℕ, m > 0 → is_divisor (m^2 - 2) n → count = 3)) := by
  sorry

end num_positive_integers_l105_105523


namespace consistency_condition_l105_105577

theorem consistency_condition (x y z a b c d : ℝ)
  (h1 : y + z = a)
  (h2 : x + y = b)
  (h3 : x + z = c)
  (h4 : x + y + z = d) : a + b + c = 2 * d :=
by sorry

end consistency_condition_l105_105577


namespace john_pushups_less_l105_105974

theorem john_pushups_less (zachary david john : ℕ) 
  (h1 : zachary = 19)
  (h2 : david = zachary + 39)
  (h3 : david = 58)
  (h4 : john < david) : 
  david - john = 0 :=
sorry

end john_pushups_less_l105_105974


namespace weight_of_dried_grapes_l105_105524

def fresh_grapes_initial_weight : ℝ := 25
def fresh_grapes_water_percentage : ℝ := 0.90
def dried_grapes_water_percentage : ℝ := 0.20

theorem weight_of_dried_grapes :
  (fresh_grapes_initial_weight * (1 - fresh_grapes_water_percentage)) /
  (1 - dried_grapes_water_percentage) = 3.125 := by
  -- Proof omitted
  sorry

end weight_of_dried_grapes_l105_105524


namespace quadratic_roots_negative_l105_105342

theorem quadratic_roots_negative (m : ℝ) :
  ∀ (x₁ x₂ : ℝ), 3 * x₁ ^ 2 + 6 * x₁ + m = 0 ∧ 3 * x₂ ^ 2 + 6 * x₂ + m = 0 ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = -2 ∧ 0 < x₁ * x₂ ↔ m = 1 ∨ m = 2 ∨ m = 3 := 
by
  sorry

end quadratic_roots_negative_l105_105342


namespace largest_amount_received_back_l105_105978

theorem largest_amount_received_back 
  (x y x_lost y_lost : ℕ) 
  (h1 : 20 * x + 100 * y = 3000) 
  (h2 : x_lost + y_lost = 16) 
  (h3 : x_lost = y_lost + 2 ∨ x_lost = y_lost - 2) 
  : (3000 - (20 * x_lost + 100 * y_lost) = 2120) :=
sorry

end largest_amount_received_back_l105_105978


namespace find_setC_l105_105719

def setA := {x : ℝ | x^2 - 3 * x + 2 = 0}
def setB (a : ℝ) := {x : ℝ | a * x - 2 = 0}
def union_condition (a : ℝ) : Prop := (setA ∪ setB a) = setA
def setC := {a : ℝ | union_condition a}

theorem find_setC : setC = {0, 1, 2} :=
by
  sorry

end find_setC_l105_105719


namespace replaced_person_weight_l105_105452

theorem replaced_person_weight :
  ∀ (old_avg_weight new_person_weight incr_weight : ℕ),
    old_avg_weight * 8 + incr_weight = new_person_weight →
    incr_weight = 16 →
    new_person_weight = 81 →
    (old_avg_weight - (new_person_weight - incr_weight) / 8) = 65 :=
by
  intros old_avg_weight new_person_weight incr_weight h1 h2 h3
  -- TODO: Proof goes here
  sorry

end replaced_person_weight_l105_105452


namespace problem_solution_l105_105253

noncomputable def f1 (x : ℝ) : ℝ := -2 * x + 2 * Real.sqrt 2
noncomputable def f2 (x : ℝ) : ℝ := Real.sin x
noncomputable def f3 (x : ℝ) : ℝ := x + (1 / x)
noncomputable def f4 (x : ℝ) : ℝ := Real.exp x
noncomputable def f5 (x : ℝ) : ℝ := -2 * Real.log x

def has_inverse_proportion_point (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∃ x ∈ domain, x * f x = 1

theorem problem_solution :
  (has_inverse_proportion_point f1 univ) ∧
  (has_inverse_proportion_point f2 (Set.Icc 0 (2 * Real.pi))) ∧
  ¬ (has_inverse_proportion_point f3 (Set.Ioi 0)) ∧
  (has_inverse_proportion_point f4 univ) ∧
  ¬ (has_inverse_proportion_point f5 (Set.Ioi 0)) :=
by
  sorry

end problem_solution_l105_105253


namespace monthly_rent_is_1300_l105_105195

def shop_length : ℕ := 10
def shop_width : ℕ := 10
def annual_rent_per_square_foot : ℕ := 156

def area_of_shop : ℕ := shop_length * shop_width
def annual_rent_for_shop : ℕ := annual_rent_per_square_foot * area_of_shop

def monthly_rent : ℕ := annual_rent_for_shop / 12

theorem monthly_rent_is_1300 : monthly_rent = 1300 := by
  sorry

end monthly_rent_is_1300_l105_105195


namespace determine_A_l105_105001

theorem determine_A (x y A : ℝ) 
  (h : (x + y) ^ 3 - x * y * (x + y) = (x + y) * A) : 
  A = x^2 + x * y + y^2 := 
by
  sorry

end determine_A_l105_105001


namespace log_sum_eq_five_l105_105723

variable {a : ℕ → ℝ}

-- Conditions from the problem
def geometric_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) = 3 * a n 

def sum_condition (a : ℕ → ℝ) : Prop :=
a 2 + a 4 + a 9 = 9

-- The mathematical statement to prove
theorem log_sum_eq_five (h1 : geometric_seq a) (h2 : sum_condition a) :
  Real.logb 3 (a 5 + a 7 + a 9) = 5 := 
sorry

end log_sum_eq_five_l105_105723


namespace analysis_method_sufficient_conditions_l105_105231

theorem analysis_method_sufficient_conditions (P : Prop) (analysis_method : ∀ (Q : Prop), (Q → P) → Q) :
  ∀ Q, (Q → P) → Q :=
by
  -- Proof is skipped
  sorry

end analysis_method_sufficient_conditions_l105_105231


namespace quadrangular_prism_volume_l105_105125

theorem quadrangular_prism_volume
  (perimeter : ℝ)
  (side_length : ℝ)
  (height : ℝ)
  (volume : ℝ)
  (H1 : perimeter = 32)
  (H2 : side_length = perimeter / 4)
  (H3 : height = side_length)
  (H4 : volume = side_length * side_length * height) :
  volume = 512 := by
    sorry

end quadrangular_prism_volume_l105_105125


namespace opposite_of_neg_two_l105_105816

theorem opposite_of_neg_two :
  (∃ y : ℤ, -2 + y = 0) → y = 2 :=
begin
  intro h,
  cases h with y hy,
  linarith,
end

end opposite_of_neg_two_l105_105816


namespace cos_even_function_l105_105114

theorem cos_even_function : ∀ x : ℝ, Real.cos (-x) = Real.cos x := 
by 
  sorry

end cos_even_function_l105_105114


namespace albert_needs_more_money_l105_105687

def cost_of_paintbrush : ℝ := 1.50
def cost_of_paints : ℝ := 4.35
def cost_of_easel : ℝ := 12.65
def amount_already_has : ℝ := 6.50

theorem albert_needs_more_money : 
  (cost_of_paintbrush + cost_of_paints + cost_of_easel) - amount_already_has = 12.00 := 
by
  sorry

end albert_needs_more_money_l105_105687


namespace sum_of_all_four_is_zero_l105_105290

variables {a b c d : ℤ}

theorem sum_of_all_four_is_zero (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum_rows : a + b = c + d) 
  (h_product_columns : a * c = b * d) :
  a + b + c + d = 0 := 
sorry

end sum_of_all_four_is_zero_l105_105290


namespace complex_quadrant_l105_105397

theorem complex_quadrant (a b : ℝ) (h : (a + Complex.I) / (b - Complex.I) = 2 - Complex.I) :
  (a < 0 ∧ b < 0) :=
by
  sorry

end complex_quadrant_l105_105397


namespace find_a_l105_105281

theorem find_a (a : ℝ) (h : a^2 + a^2 / 4 = 5) : a = 2 ∨ a = -2 := 
sorry

end find_a_l105_105281


namespace algebraic_expression_value_l105_105155

-- Define given condition
def condition (x : ℝ) : Prop := 3 * x^2 - 2 * x - 1 = 2

-- Define the target expression
def target_expression (x : ℝ) : ℝ := -9 * x^2 + 6 * x - 1

-- The theorem statement
theorem algebraic_expression_value (x : ℝ) (h : condition x) : target_expression x = -10 := by
  sorry

end algebraic_expression_value_l105_105155


namespace find_other_number_l105_105187

theorem find_other_number
  (a b : ℕ)
  (HCF : ℕ)
  (LCM : ℕ)
  (h1 : HCF = 12)
  (h2 : LCM = 396)
  (h3 : a = 36)
  (h4 : HCF * LCM = a * b) :
  b = 132 :=
by
  sorry

end find_other_number_l105_105187


namespace crossword_solution_correct_l105_105945

noncomputable def vertical_2 := "счет"
noncomputable def vertical_3 := "евро"
noncomputable def vertical_4 := "доллар"
noncomputable def vertical_5 := "вклад"
noncomputable def vertical_6 := "золото"
noncomputable def vertical_7 := "ломбард"

noncomputable def horizontal_1 := "обмен"
noncomputable def horizontal_2 := "система"
noncomputable def horizontal_3 := "ломбард"

theorem crossword_solution_correct :
  (vertical_2 = "счет") ∧
  (vertical_3 = "евро") ∧
  (vertical_4 = "доллар") ∧
  (vertical_5 = "вклад") ∧
  (vertical_6 = "золото") ∧
  (vertical_7 = "ломбард") ∧
  (horizontal_1 = "обмен") ∧
  (horizontal_2 = "система") ∧
  (horizontal_3 = "ломбард") :=
by
  sorry

end crossword_solution_correct_l105_105945


namespace max_discount_rate_l105_105678

theorem max_discount_rate (cost_price selling_price min_profit_margin x : ℝ) 
  (h_cost : cost_price = 4)
  (h_selling : selling_price = 5)
  (h_margin : min_profit_margin = 0.4) :
  0 ≤ x ∧ x ≤ 12 ↔ (selling_price * (1 - x / 100) - cost_price) ≥ min_profit_margin := by
  sorry

end max_discount_rate_l105_105678


namespace steve_and_laura_meet_time_l105_105184

structure PathsOnParallelLines where
  steve_speed : ℝ
  laura_speed : ℝ
  path_separation : ℝ
  art_diameter : ℝ
  initial_distance_hidden : ℝ

def meet_time (p : PathsOnParallelLines) : ℝ :=
  sorry -- To be proven

-- Define the specific case for Steve and Laura
def steve_and_laura_paths : PathsOnParallelLines :=
  { steve_speed := 3,
    laura_speed := 1,
    path_separation := 240,
    art_diameter := 80,
    initial_distance_hidden := 230 }

theorem steve_and_laura_meet_time :
  meet_time steve_and_laura_paths = 45 :=
  sorry

end steve_and_laura_meet_time_l105_105184


namespace correct_statements_l105_105839

-- Define the problem's elements and conditions
noncomputable def Plane (α : Type*) := α → α → Prop
variable {α : Type*}

-- Define properties of planes
def parallel_to_line (P : Plane α) (l : α → Prop) : Prop := ∀ {x y}, l x → l y → P x y
def parallel_to_plane (P1 P2 : Plane α) := ∀ {x y}, P1 x y → P2 x y
def perpendicular_to_line (P : Plane α) (l : α → Prop) : Prop := ∀ {x}, l x → ∀ {y}, not (P x y)
def equal_angles_with_line (P : Plane α) (l : α → Prop) (angle : α → ℝ) : Prop := ∀ {x}, l x → angle x = angle y

-- Statements to prove as per the problem
theorem correct_statements :
  (∀ (P1 P2 : Plane α) (l : α → Prop), parallel_to_line P1 l → parallel_to_line P2 l → parallel_to_plane P1 P2) ∧
  (∀ (P1 P2 : Plane α), parallel_to_plane P1 P2 → parallel_to_plane P2 P1) ∧
  ¬ (∀ (P1 P2 : Plane α) (l : α → Prop), perpendicular_to_line P1 l → perpendicular_to_line P2 l → parallel_to_plane P1 P2) ∧
  ¬ (∀ (P1 P2 : Plane α) (l : α → Prop) (angle : α → ℝ), equal_angles_with_line P1 l angle → equal_angles_with_line P2 l angle → parallel_to_plane P1 P2) :=
by sorry

end correct_statements_l105_105839


namespace boxes_calculation_proof_l105_105336

variable (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_box : ℕ)
variable (total_eggs : ℕ := baskets * eggs_per_basket)
variable (boxes_needed : ℕ := total_eggs / eggs_per_box)

theorem boxes_calculation_proof :
  baskets = 21 →
  eggs_per_basket = 48 →
  eggs_per_box = 28 →
  boxes_needed = 36 :=
by
  intros
  sorry

end boxes_calculation_proof_l105_105336


namespace min_delivery_time_l105_105825

def elf_delivery_time (distances : Fin 63 → ℕ) (times : Fin 63 → ℕ) : ℕ :=
  (Finset.univ : Finset (Fin 63)).sup (λ i, i.val * distances i)

theorem min_delivery_time (distances : Fin 63 → ℕ) (times : Fin 63 → ℕ)
  (h_distinct : Function.Injective distances)
  (h_range : ∀ i, distances i ∈ Finset.range 64 \ Finset.range 1) :
  elf_delivery_time distances times = 1024 :=
begin
  sorry
end

end min_delivery_time_l105_105825


namespace maximum_discount_rate_l105_105629

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l105_105629


namespace find_c_l105_105364

-- Define conditions as Lean statements
theorem find_c :
  ∀ (c n : ℝ), 
  (n ^ 2 + 1 / 16 = 1 / 4) → 
  2 * n = c → 
  c < 0 → 
  c = - (Real.sqrt 3) / 2 :=
by
  intros c n h1 h2 h3
  sorry

end find_c_l105_105364


namespace marble_prob_red_or_white_l105_105084

def marble_bag_prob (total_marbles : ℕ) (blue_marbles : ℕ) (red_marbles : ℕ) (white_marbles : ℕ) : ℚ :=
  (red_marbles + white_marbles : ℚ) / total_marbles

theorem marble_prob_red_or_white :
  let total_marbles := 20
  let blue_marbles := 5
  let red_marbles := 7
  let white_marbles := total_marbles - (blue_marbles + red_marbles)
  marble_bag_prob total_marbles blue_marbles red_marbles white_marbles = 3 / 4 :=
by
  sorry

end marble_prob_red_or_white_l105_105084


namespace divided_number_l105_105092

theorem divided_number (x y : ℕ) (h1 : 7 * x + 5 * y = 146) (h2 : y = 11) : x + y = 24 :=
sorry

end divided_number_l105_105092


namespace hairstylist_weekly_earnings_l105_105849

-- Definition of conditions as given in part a)
def cost_normal_haircut := 5
def cost_special_haircut := 6
def cost_trendy_haircut := 8

def number_normal_haircuts_per_day := 5
def number_special_haircuts_per_day := 3
def number_trendy_haircuts_per_day := 2

def working_days_per_week := 7

-- The goal is to prove that the hairstylist's weekly earnings equal to 413 dollars
theorem hairstylist_weekly_earnings : 
  (number_normal_haircuts_per_day * cost_normal_haircut +
  number_special_haircuts_per_day * cost_special_haircut +
  number_trendy_haircuts_per_day * cost_trendy_haircut) * 
  working_days_per_week = 413 := 
by sorry -- We use "by sorry" to skip the proof

end hairstylist_weekly_earnings_l105_105849


namespace square_garden_tiles_l105_105856

theorem square_garden_tiles (n : ℕ) (h : 2 * n - 1 = 25) : n^2 = 169 :=
by
  sorry

end square_garden_tiles_l105_105856


namespace common_difference_is_neg4_max_sum_of_6_max_n_for_positive_S_l105_105052

noncomputable def first_term := 23
noncomputable def common_difference : Int := -4

theorem common_difference_is_neg4 (d : Int) 
  (h_pos_6 : (first_term + 5 * d) > 0) 
  (h_neg_7 : (first_term + 6 * d) < 0) : 
  d = -4 := 
sorry

theorem max_sum_of_6 : S_n = 78 :=
sorry

theorem max_n_for_positive_S (n : ℕ)
  (h_sn : S_n > 0) : 
  n <= 12 :=
sorry

end common_difference_is_neg4_max_sum_of_6_max_n_for_positive_S_l105_105052


namespace two_positive_numbers_inequality_three_positive_numbers_inequality_l105_105943

theorem two_positive_numbers_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) * (1 / a + 1 / b) ≥ 4 :=
by sorry

theorem three_positive_numbers_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 9 :=
by sorry

end two_positive_numbers_inequality_three_positive_numbers_inequality_l105_105943


namespace minimum_moves_to_find_coin_l105_105055

/--
Consider a circle of 100 thimbles with a coin hidden under one of them. 
You can check four thimbles per move. After each move, the coin moves to a neighboring thimble.
Prove that the minimum number of moves needed to guarantee finding the coin is 33.
-/
theorem minimum_moves_to_find_coin 
  (N : ℕ) (hN : N = 100) (M : ℕ) (hM : M = 4) :
  ∃! k : ℕ, k = 33 :=
by sorry

end minimum_moves_to_find_coin_l105_105055


namespace tangent_with_min_slope_has_given_equation_l105_105503

-- Define the given function f(x)
def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

-- Define the derivative of the function f(x)
def f_prime (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 6

-- Define the coordinates of the tangent point
def tangent_point : ℝ × ℝ := (-1, f (-1))

-- Define the equation of the tangent line at the point with the minimum slope
def tangent_line_equation (x y : ℝ) : Prop := 3 * x - y - 11 = 0

-- Main theorem statement that needs to be proved
theorem tangent_with_min_slope_has_given_equation :
  tangent_line_equation (-1) (f (-1)) :=
sorry

end tangent_with_min_slope_has_given_equation_l105_105503


namespace relationship_y1_y2_l105_105270

theorem relationship_y1_y2 (k b y1 y2 : ℝ) (h₀ : k < 0) (h₁ : y1 = k * (-1) + b) (h₂ : y2 = k * 1 + b) : y1 > y2 := 
by
  sorry

end relationship_y1_y2_l105_105270


namespace value_of_a_plus_b_l105_105404

theorem value_of_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, (x > -4 ∧ x < 1) ↔ (ax^2 + bx - 2 > 0)) → 
  a = 1/2 → 
  b = 3/2 → 
  a + b = 2 := 
by 
  intro h cond_a cond_b 
  rw [cond_a, cond_b]
  norm_num

end value_of_a_plus_b_l105_105404


namespace measure_smaller_angle_east_northwest_l105_105091

/-- A mathematical structure for a circle with 12 rays forming congruent central angles. -/
structure CircleWithRays where
  rays : Finset (Fin 12)  -- There are 12 rays
  congruent_angles : ∀ i, i ∈ rays

/-- The measure of the central angle formed by each ray is 30 degrees (since 360/12 = 30). -/
def central_angle_measure : ℝ := 30

/-- The measure of the smaller angle formed between the ray pointing East and the ray pointing Northwest is 150 degrees. -/
theorem measure_smaller_angle_east_northwest (c : CircleWithRays) : 
  ∃ angle : ℝ, angle = 150 := by
  sorry

end measure_smaller_angle_east_northwest_l105_105091


namespace roots_inequality_l105_105445

theorem roots_inequality (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) :
  -1 ≤ z ∧ z ≤ 13 / 3 :=
sorry

end roots_inequality_l105_105445


namespace recreation_spending_l105_105982

theorem recreation_spending : 
  ∀ (W : ℝ), 
  (last_week_spent : ℝ) -> last_week_spent = 0.20 * W →
  (this_week_wages : ℝ) -> this_week_wages = 0.80 * W →
  (this_week_spent : ℝ) -> this_week_spent = 0.40 * this_week_wages →
  this_week_spent / last_week_spent * 100 = 160 :=
by
  sorry

end recreation_spending_l105_105982


namespace intersection_complement_l105_105142

variable {M : Set ℝ := {-1, 0, 1, 3}}

def N : Set ℝ := {x | x^2 - x - 2 ≥ 0}

def complement_N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem intersection_complement :
  M ∩ complement_N = {0, 1} :=
  sorry

end intersection_complement_l105_105142


namespace original_number_of_boys_l105_105453

theorem original_number_of_boys (n : ℕ) (W : ℕ) 
  (h1 : W = n * 35) 
  (h2 : W + 40 = (n + 1) * 36) 
  : n = 4 :=
sorry

end original_number_of_boys_l105_105453


namespace maximum_discount_rate_l105_105641

theorem maximum_discount_rate (cost_price selling_price : ℝ) (min_profit_margin : ℝ) :
  cost_price = 4 ∧ 
  selling_price = 5 ∧ 
  min_profit_margin = 0.4 → 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 ∧ x = 12 :=
by
  sorry

end maximum_discount_rate_l105_105641


namespace sum_of_decimals_as_common_fraction_l105_105883

theorem sum_of_decimals_as_common_fraction:
  (2 / 10 : ℚ) + (3 / 100 : ℚ) + (4 / 1000 : ℚ) + (5 / 10000 : ℚ) + (6 / 100000 : ℚ) = 733 / 3125 :=
by
  -- Explanation and proof steps are omitted as directed.
  sorry

end sum_of_decimals_as_common_fraction_l105_105883


namespace sum_of_corners_is_164_l105_105764

section CheckerboardSum

-- Define the total number of elements in the 9x9 grid
def num_elements := 81

-- Define the positions of the corners
def top_left : ℕ := 1
def top_right : ℕ := 9
def bottom_left : ℕ := 73
def bottom_right : ℕ := 81

-- Define the sum of the corners
def corner_sum : ℕ := top_left + top_right + bottom_left + bottom_right

-- State the theorem
theorem sum_of_corners_is_164 : corner_sum = 164 :=
by
  exact sorry

end CheckerboardSum

end sum_of_corners_is_164_l105_105764


namespace sum_lent_is_1500_l105_105481

/--
A person lent a certain sum of money at 4% per annum at simple interest.
In 4 years, the interest amounted to Rs. 1260 less than the sum lent.
Prove that the sum lent was Rs. 1500.
-/
theorem sum_lent_is_1500
  (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ)
  (h1 : r = 4) (h2 : t = 4)
  (h3 : I = P - 1260)
  (h4 : I = P * r * t / 100):
  P = 1500 :=
by
  sorry

end sum_lent_is_1500_l105_105481


namespace lia_quadrilateral_rod_count_l105_105425

theorem lia_quadrilateral_rod_count :
  let rods := {n : ℕ | 1 ≤ n ∧ n ≤ 40}
  let selected_rods := {5, 10, 20}
  let remaining_rods := rods \ selected_rods
  rod_count = 26 ∧ (∃ d ∈ remaining_rods, 
    (5 + 10 + 20) > d ∧ (10 + 20 + d) > 5 ∧ (5 + 20 + d) > 10 ∧ (5 + 10 + d) > 20)
:=
sorry

end lia_quadrilateral_rod_count_l105_105425


namespace ambulance_ride_cost_correct_l105_105015

noncomputable def total_bill : ℝ := 12000
noncomputable def medication_percentage : ℝ := 0.40
noncomputable def imaging_tests_percentage : ℝ := 0.15
noncomputable def surgical_procedure_percentage : ℝ := 0.20
noncomputable def overnight_stays_percentage : ℝ := 0.25
noncomputable def food_cost : ℝ := 300
noncomputable def consultation_fee : ℝ := 80

noncomputable def ambulance_ride_cost := total_bill - (food_cost + consultation_fee)

theorem ambulance_ride_cost_correct :
  ambulance_ride_cost = 11620 :=
by
  sorry

end ambulance_ride_cost_correct_l105_105015


namespace circle_area_l105_105913

theorem circle_area (r : ℝ) (h : 2 * (1 / (2 * π * r)) = r / 2) : π * r^2 = 2 := 
by 
  sorry

end circle_area_l105_105913


namespace find_digit_x_l105_105189

def base7_number (x : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + x * 7 + 4

def is_divisible_by_19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

theorem find_digit_x : is_divisible_by_19 (base7_number 4) :=
sorry

end find_digit_x_l105_105189


namespace max_integer_value_l105_105004

theorem max_integer_value (x : ℝ) : ∃ (m : ℤ), m = 53 ∧ ∀ y : ℝ, (1 + 13 / (3 * y^2 + 9 * y + 7) ≤ m) := 
sorry

end max_integer_value_l105_105004


namespace free_throw_percentage_l105_105358

theorem free_throw_percentage (p : ℚ) :
  (1 - p)^2 + 2 * p * (1 - p) = 16 / 25 → p = 3 / 5 :=
by
  sorry

end free_throw_percentage_l105_105358


namespace ratio_of_areas_l105_105011

theorem ratio_of_areas 
  (A B C D E F : Type)
  (AB AC AD : ℝ)
  (h1 : AB = 130)
  (h2 : AC = 130)
  (h3 : AD = 26)
  (CF : ℝ)
  (h4 : CF = 91)
  (BD : ℝ)
  (h5 : BD = 104)
  (AF : ℝ)
  (h6 : AF = 221)
  (EF DE BE CE : ℝ)
  (h7 : EF / DE = 91 / 104)
  (h8 : CE / BE = 3.5) :
  EF * CE = 318.5 * DE * BE :=
sorry

end ratio_of_areas_l105_105011


namespace bird_stork_difference_l105_105488

theorem bird_stork_difference :
  let initial_birds := 3
  let initial_storks := 4
  let additional_birds := 2
  let total_birds := initial_birds + additional_birds
  total_birds - initial_storks = 1 := 
by
  let initial_birds := 3
  let initial_storks := 4
  let additional_birds := 2
  let total_birds := initial_birds + additional_birds
  show total_birds - initial_storks = 1
  sorry

end bird_stork_difference_l105_105488


namespace total_amount_spent_l105_105105

def cost_per_dozen_apples : ℕ := 40
def cost_per_dozen_pears : ℕ := 50
def dozens_apples : ℕ := 14
def dozens_pears : ℕ := 14

theorem total_amount_spent : (dozens_apples * cost_per_dozen_apples + dozens_pears * cost_per_dozen_pears) = 1260 := 
  by
  sorry

end total_amount_spent_l105_105105


namespace smallest_perimeter_of_acute_triangle_with_consecutive_sides_l105_105068

theorem smallest_perimeter_of_acute_triangle_with_consecutive_sides :
  ∃ (a : ℕ), (a > 1) ∧ (∃ (b c : ℕ), b = a + 1 ∧ c = a + 2 ∧ (∃ (C : ℝ), a^2 + b^2 - c^2 < 0 ∧ c = 4)) ∧ (a + (a + 1) + (a + 2) = 9) :=
by {
  sorry
}

end smallest_perimeter_of_acute_triangle_with_consecutive_sides_l105_105068


namespace cost_per_mile_l105_105339

def miles_per_week : ℕ := 3 * 50 + 4 * 100
def weeks_per_year : ℕ := 52
def miles_per_year : ℕ := miles_per_week * weeks_per_year
def weekly_fee : ℕ := 100
def yearly_total_fee : ℕ := 7800
def yearly_weekly_fees : ℕ := 52 * weekly_fee
def yearly_mile_fees := yearly_total_fee - yearly_weekly_fees
def pay_per_mile := yearly_mile_fees / miles_per_year

theorem cost_per_mile : pay_per_mile = 909 / 10000 := by
  -- proof will be added here
  sorry

end cost_per_mile_l105_105339


namespace hypotenuse_length_l105_105292

theorem hypotenuse_length (a b c : ℝ) (h_right_angled : c^2 = a^2 + b^2) (h_sum_of_squares : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l105_105292


namespace probability_of_sum_odd_is_correct_l105_105986

noncomputable def probability_sum_odd : ℚ :=
  let total_balls := 13
  let drawn_balls := 7
  let total_ways := Nat.choose total_balls drawn_balls
  let favorable_ways := 
    Nat.choose 7 5 * Nat.choose 6 2 + 
    Nat.choose 7 3 * Nat.choose 6 4 + 
    Nat.choose 7 1 * Nat.choose 6 6
  favorable_ways / total_ways

theorem probability_of_sum_odd_is_correct :
  probability_sum_odd = 847 / 1716 :=
by
  -- Proof goes here
  sorry

end probability_of_sum_odd_is_correct_l105_105986


namespace vector_addition_l105_105510

def v1 : ℝ × ℝ := (3, -6)
def v2 : ℝ × ℝ := (2, -9)
def v3 : ℝ × ℝ := (-1, 3)
def c1 : ℝ := 4
def c2 : ℝ := 5
def result : ℝ × ℝ := (23, -72)

theorem vector_addition :
  c1 • v1 + c2 • v2 - v3 = result :=
by
  sorry

end vector_addition_l105_105510


namespace fourth_term_eq_156_l105_105242

-- Definition of the sequence term
def seq_term (n : ℕ) : ℕ :=
  (List.range n).map (λ k => 5^k) |>.sum

-- Theorem to prove the fourth term equals 156
theorem fourth_term_eq_156 : seq_term 4 = 156 :=
sorry

end fourth_term_eq_156_l105_105242


namespace games_left_is_correct_l105_105537

-- Define the initial number of DS games
def initial_games : ℕ := 98

-- Define the number of games given away
def games_given_away : ℕ := 7

-- Define the number of games left
def games_left : ℕ := initial_games - games_given_away

-- Theorem statement to prove that the number of games left is 91
theorem games_left_is_correct : games_left = 91 :=
by
  -- Currently, we use sorry to skip the actual proof part.
  sorry

end games_left_is_correct_l105_105537


namespace max_discount_rate_l105_105616

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l105_105616


namespace compute_F_2_f_3_l105_105567

def f (a : ℝ) : ℝ := a^2 - 3 * a + 2
def F (a b : ℝ) : ℝ := b + a^3

theorem compute_F_2_f_3 : F 2 (f 3) = 10 :=
by
  sorry

end compute_F_2_f_3_l105_105567


namespace symmetric_line_eq_l105_105193

theorem symmetric_line_eq (a b : ℝ) (ha : a ≠ 0) : 
  (∃ k m : ℝ, (∀ x: ℝ, ax + b = (k * ( -x)) + m ∧ (k = 1/a ∧ m = b/a )))  := 
sorry

end symmetric_line_eq_l105_105193


namespace cafeteria_extra_fruit_l105_105461

theorem cafeteria_extra_fruit 
    (red_apples : ℕ)
    (green_apples : ℕ)
    (students : ℕ)
    (total_apples := red_apples + green_apples)
    (apples_taken := students)
    (extra_apples := total_apples - apples_taken)
    (h1 : red_apples = 42)
    (h2 : green_apples = 7)
    (h3 : students = 9) :
    extra_apples = 40 := 
by 
  sorry

end cafeteria_extra_fruit_l105_105461


namespace find_C_l105_105975

variable (A B C : ℕ)

theorem find_C (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 360) : C = 60 := by
  sorry

end find_C_l105_105975


namespace sum_of_reciprocals_of_transformed_roots_l105_105759

theorem sum_of_reciprocals_of_transformed_roots :
  ∀ (a b c : ℂ), (a^3 - a + 1 = 0) → (b^3 - b + 1 = 0) → (c^3 - c + 1 = 0) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  (1/(a+1) + 1/(b+1) + 1/(c+1) = -2) :=
by
  intros a b c ha hb hc habc
  sorry

end sum_of_reciprocals_of_transformed_roots_l105_105759


namespace opposite_of_neg2_l105_105801

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l105_105801


namespace total_number_of_chickens_is_76_l105_105013

theorem total_number_of_chickens_is_76 :
  ∀ (hens roosters chicks : ℕ),
    hens = 12 →
    (∀ n, roosters = hens / 3) →
    (∀ n, chicks = hens * 5) →
    hens + roosters + chicks = 76 :=
by
  intros hens roosters chicks h1 h2 h3
  rw [h1, h2 12, h3 12]
  ring
  sorry

end total_number_of_chickens_is_76_l105_105013


namespace seashells_total_l105_105038

theorem seashells_total (sam_seashells : ℕ) (mary_seashells : ℕ) (h1 : sam_seashells = 18) (h2 : mary_seashells = 47) : sam_seashells + mary_seashells = 65 :=
by
  rw [h1, h2]
  exact rfl

end seashells_total_l105_105038


namespace sum_of_squares_l105_105936

-- Define the proposition as a universal statement 
theorem sum_of_squares (a b : ℝ) : a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by
  sorry

end sum_of_squares_l105_105936


namespace x_intercept_of_perpendicular_line_l105_105458

theorem x_intercept_of_perpendicular_line 
  (a : ℝ)
  (l1 : ℝ → ℝ → Prop)
  (l1_eq : ∀ x y, l1 x y ↔ (a+3)*x + y - 4 = 0)
  (l2 : ℝ → ℝ → Prop)
  (l2_eq : ∀ x y, l2 x y ↔ x + (a-1)*y + 4 = 0)
  (perpendicular : ∀ x y, l1 x y → l2 x y → (a+3)*(a-1) = -1) :
  (∃ x : ℝ, l1 x 0 ∧ x = 2) :=
sorry

end x_intercept_of_perpendicular_line_l105_105458


namespace mutually_exclusive_not_complementary_l105_105211

-- Definitions of events
def EventA (n : ℕ) : Prop := n % 2 = 1
def EventB (n : ℕ) : Prop := n % 2 = 0
def EventC (n : ℕ) : Prop := n % 2 = 0
def EventD (n : ℕ) : Prop := n = 2 ∨ n = 4

-- Mutual exclusivity and complementarity
def mutually_exclusive {α : Type} (A B : α → Prop) : Prop :=
∀ x, ¬ (A x ∧ B x)

def complementary {α : Type} (A B : α → Prop) : Prop :=
∀ x, A x ∨ B x

-- The statement to be proved
theorem mutually_exclusive_not_complementary :
  mutually_exclusive EventA EventD ∧ ¬ complementary EventA EventD :=
by sorry

end mutually_exclusive_not_complementary_l105_105211


namespace parallelogram_sides_are_parallel_l105_105497

theorem parallelogram_sides_are_parallel 
  {a b c : ℤ} (h_area : c * (a^2 + b^2) = 2011 * b) : 
  (∃ k : ℤ, a = 2011 * k ∧ (b = 2011 ∨ b = -2011)) :=
by
  sorry

end parallelogram_sides_are_parallel_l105_105497


namespace train_speed_l105_105075

theorem train_speed (distance time : ℝ) (h₁ : distance = 240) (h₂ : time = 4) : 
  ((distance / time) * 3.6) = 216 := 
by 
  rw [h₁, h₂] 
  sorry

end train_speed_l105_105075


namespace students_transferred_l105_105160

theorem students_transferred (initial_students : ℝ) (students_left : ℝ) (end_students : ℝ) :
  initial_students = 42.0 →
  students_left = 4.0 →
  end_students = 28.0 →
  initial_students - students_left - end_students = 10.0 :=
by
  intros
  sorry

end students_transferred_l105_105160


namespace compute_r_l105_105308

variables {j p t m n x y r : ℝ}

theorem compute_r
    (h1 : j = 0.75 * p)
    (h2 : j = 0.80 * t)
    (h3 : t = p - r * p / 100)
    (h4 : m = 1.10 * p)
    (h5 : n = 0.70 * m)
    (h6 : j + p + t = m * n)
    (h7 : x = 1.15 * j)
    (h8 : y = 0.80 * n)
    (h9 : x * y = (j + p + t) ^ 2) : r = 6.25 := by
  sorry

end compute_r_l105_105308


namespace average_speed_l105_105220

-- Define the conditions as constants and theorems
def distance1 : ℝ := 240
def distance2 : ℝ := 420
def time_diff : ℝ := 3

theorem average_speed : ∃ v t : ℝ, distance1 = v * t ∧ distance2 = v * (t + time_diff) → v = 60 := 
by
  sorry

end average_speed_l105_105220


namespace find_PF_2_l105_105739

-- Define the hyperbola and points
def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 3) = 1
def PF_1 := 3
def a := 2
def two_a := 2 * a

-- State the theorem
theorem find_PF_2 (PF_2 : ℝ) (cond1 : PF_1 = 3) (cond2 : abs (PF_1 - PF_2) = two_a) : PF_2 = 7 :=
sorry

end find_PF_2_l105_105739


namespace california_vs_texas_license_plates_l105_105110

theorem california_vs_texas_license_plates :
  (26^4 * 10^4) - (26^3 * 10^3) = 4553200000 :=
by
  sorry

end california_vs_texas_license_plates_l105_105110


namespace prob_X_greater_than_4_l105_105136

noncomputable def normalDist (μ σ : ℝ) : Measure ℝ := sorry

variable (μ σ : ℝ) [isNormalDist : isProbabilityDistribution (normalDist μ σ)]
variable (X : ℝ → ℝ)
variable (h1 : μ = 2)
variable (h2 : ∃ σ, X ~ normalDist μ σ)
variable (h3 : P (fun x => 0 < X x ∧ X x < 4) = 0.8)

-- We need to express the aim of the problem
theorem prob_X_greater_than_4 : P (λ x => X x > 4) = 0.1 :=
by
  sorry

end prob_X_greater_than_4_l105_105136


namespace max_discount_rate_l105_105651

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l105_105651


namespace quadractic_b_value_l105_105127

def quadratic_coefficients (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem quadractic_b_value :
  ∀ (a b c : ℝ), quadratic_coefficients 1 (-2) (-3) (x : ℝ) → 
  b = -2 := by
  sorry

end quadractic_b_value_l105_105127


namespace pour_tea_into_containers_l105_105987

-- Define the total number of containers
def total_containers : ℕ := 80

-- Define the amount of tea that Geraldo drank in terms of containers
def geraldo_drank_containers : ℚ := 3.5

-- Define the amount of tea that Geraldo consumed in terms of pints
def geraldo_drank_pints : ℕ := 7

-- Define the conversion factor from pints to gallons
def pints_per_gallon : ℕ := 8

-- Question: How many gallons of tea were poured into the containers?
theorem pour_tea_into_containers 
  (total_containers : ℕ)
  (geraldo_drank_containers : ℚ)
  (geraldo_drank_pints : ℕ)
  (pints_per_gallon : ℕ) :
  (total_containers * (geraldo_drank_pints / geraldo_drank_containers) / pints_per_gallon) = 20 :=
by
  sorry

end pour_tea_into_containers_l105_105987


namespace range_of_a_l105_105008

open Real

theorem range_of_a
  (a : ℝ)
  (curve : ∀ θ : ℝ, ∃ p : ℝ × ℝ, p = (a + 2 * cos θ, a + 2 * sin θ))
  (distance_two_points : ∀ θ : ℝ, dist (0,0) (a + 2 * cos θ, a + 2 * sin θ) = 2) :
  (-2 * sqrt 2 < a ∧ a < 0) ∨ (0 < a ∧ a < 2 * sqrt 2) :=
sorry

end range_of_a_l105_105008


namespace area_to_be_painted_l105_105917

def wall_height : ℕ := 8
def wall_length : ℕ := 15
def glass_painting_height : ℕ := 3
def glass_painting_length : ℕ := 5

theorem area_to_be_painted :
  (wall_height * wall_length) - (glass_painting_height * glass_painting_length) = 105 := by
  sorry

end area_to_be_painted_l105_105917


namespace find_s_over_r_l105_105444

-- Define the function
def f (k : ℝ) : ℝ := 9 * k ^ 2 - 6 * k + 15

-- Define constants
variables (d r s : ℝ)

-- Define the main theorem to be proved
theorem find_s_over_r : 
  (∀ k : ℝ, f k = d * (k + r) ^ 2 + s) → s / r = -42 :=
by
  sorry

end find_s_over_r_l105_105444


namespace cubic_roots_natural_numbers_l105_105120

theorem cubic_roots_natural_numbers (p : ℝ) :
  (∃ x1 x2 x3 : ℕ, (5 * (x1 : ℝ)^3 - 5 * (p + 1) * (x1 : ℝ)^2 + (71 * p - 1) * (x1 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x2 : ℝ)^3 - 5 * (p + 1) * (x2 : ℝ)^2 + (71 * p - 1) * (x2 : ℝ) + 1 = 66 * p) ∧
                   (5 * (x3 : ℝ)^3 - 5 * (p + 1) * (x3 : ℝ)^2 + (71 * p - 1) * (x3 : ℝ) + 1 = 66 * p)) →
  p = 76 :=
sorry

end cubic_roots_natural_numbers_l105_105120


namespace class_size_l105_105748

theorem class_size (g : ℕ) (h1 : g + (g + 3) = 44) (h2 : g^2 + (g + 3)^2 = 540) : g + (g + 3) = 44 :=
by
  sorry

end class_size_l105_105748


namespace value_of_q_l105_105307

-- Define the problem in Lean 4

variable (a d q : ℝ) (h0 : a ≠ 0)
variables (M P : Set ℝ)
variable (hM : M = {a, a + d, a + 2 * d})
variable (hP : P = {a, a * q, a * q * q})
variable (hMP : M = P)

theorem value_of_q : q = -1 :=
by
  sorry

end value_of_q_l105_105307


namespace seating_arrangement_fixed_pairs_l105_105918

theorem seating_arrangement_fixed_pairs 
  (total_chairs : ℕ) 
  (total_people : ℕ) 
  (specific_pair_adjacent : Prop)
  (comb : ℕ) 
  (four_factorial : ℕ) 
  (two_factorial : ℕ) 
  : total_chairs = 6 → total_people = 5 → specific_pair_adjacent → comb = Nat.choose 6 4 → 
    four_factorial = Nat.factorial 4 → two_factorial = Nat.factorial 2 → 
    Nat.choose 6 4 * Nat.factorial 4 * Nat.factorial 2 = 720 
  := by
  intros
  sorry

end seating_arrangement_fixed_pairs_l105_105918


namespace lemonade_quarts_water_l105_105826

-- Definitions derived from the conditions
def total_parts := 6 + 2 + 1 -- Sum of all ratio parts
def parts_per_gallon : ℚ := 1.5 / total_parts -- Volume per part in gallons
def parts_per_quart : ℚ := parts_per_gallon * 4 -- Volume per part in quarts
def water_needed : ℚ := 6 * parts_per_quart -- Quarts of water needed

-- Statement to prove
theorem lemonade_quarts_water : water_needed = 4 := 
by sorry

end lemonade_quarts_water_l105_105826


namespace average_score_is_67_l105_105766

def scores : List ℕ := [55, 67, 76, 82, 55]
def num_of_subjects : ℕ := List.length scores
def total_score : ℕ := List.sum scores
def average_score : ℕ := total_score / num_of_subjects

theorem average_score_is_67 : average_score = 67 := by
  sorry

end average_score_is_67_l105_105766


namespace max_alligators_in_days_l105_105847

noncomputable def days := 616
noncomputable def weeks := 88  -- derived from 616 / 7
noncomputable def alligators_per_week := 1

theorem max_alligators_in_days
  (h1 : weeks = days / 7)
  (h2 : ∀ (w : ℕ), alligators_per_week = 1) :
  weeks * alligators_per_week = 88 := by
  sorry

end max_alligators_in_days_l105_105847


namespace root_calculation_l105_105106

theorem root_calculation :
  (Real.sqrt (Real.sqrt (0.00032 ^ (1 / 5)) ^ (1 / 4))) = 0.6687 :=
by
  sorry

end root_calculation_l105_105106


namespace opposite_of_neg_two_is_two_l105_105789

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l105_105789


namespace cookie_cost_l105_105509

theorem cookie_cost 
    (initial_amount : ℝ := 100)
    (latte_cost : ℝ := 3.75)
    (croissant_cost : ℝ := 3.50)
    (days : ℕ := 7)
    (num_cookies : ℕ := 5)
    (remaining_amount : ℝ := 43) :
    (initial_amount - remaining_amount - (days * (latte_cost + croissant_cost))) / num_cookies = 1.25 := 
by
  sorry

end cookie_cost_l105_105509


namespace roots_of_quadratic_l105_105131

variable {γ δ : ℝ}

theorem roots_of_quadratic (hγ : γ^2 - 5*γ + 6 = 0) (hδ : δ^2 - 5*δ + 6 = 0) : 
  8*γ^5 + 15*δ^4 = 8425 := 
by
  sorry

end roots_of_quadratic_l105_105131


namespace inequality_solution_system_l105_105333

theorem inequality_solution_system {x : ℝ} :
  (2 * x - 1 > 5) ∧ (-x < -6) ↔ (x > 6) :=
by
  sorry

end inequality_solution_system_l105_105333


namespace total_questions_asked_l105_105010

theorem total_questions_asked (drew_correct : ℕ := 20) (drew_wrong : ℕ := 6) (carla_correct : ℕ := 14):
  let carla_wrong := 2 * drew_wrong in
  let drew_total := drew_correct + drew_wrong in
  let carla_total := carla_correct + carla_wrong in
  (drew_total + carla_total) = 52 := by
  sorry

end total_questions_asked_l105_105010


namespace steven_ships_boxes_l105_105946

-- Translate the conditions into Lean definitions and state the theorem
def truck_weight_limit : ℕ := 2000
def truck_count : ℕ := 3
def pair_weight : ℕ := 10 + 40
def boxes_per_pair : ℕ := 2

theorem steven_ships_boxes :
  ((truck_weight_limit / pair_weight) * boxes_per_pair * truck_count) = 240 := by
  sorry

end steven_ships_boxes_l105_105946


namespace smallest_b_l105_105207

theorem smallest_b (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : a * b * c = 360) : b = 3 :=
sorry

end smallest_b_l105_105207


namespace find_a_l105_105701

def star (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a (a : ℝ) (h : star a 4 = 17) : a = 49 / 3 :=
by sorry

end find_a_l105_105701


namespace prob_no_1_or_6_l105_105217

theorem prob_no_1_or_6 :
  ∀ (a b c : ℕ), (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) →
  (8 / 27 : ℝ) = (4 / 6) * (4 / 6) * (4 / 6) :=
by
  intros a b c h
  sorry

end prob_no_1_or_6_l105_105217


namespace find_w_squared_l105_105907

theorem find_w_squared (w : ℝ) :
  (w + 15)^2 = (4 * w + 9) * (3 * w + 6) →
  w^2 = ((-21 + Real.sqrt 7965) / 22)^2 ∨ 
        w^2 = ((-21 - Real.sqrt 7965) / 22)^2 :=
by sorry

end find_w_squared_l105_105907


namespace solve_quadratic_l105_105334

theorem solve_quadratic : ∀ x : ℝ, 2 * x^2 + 5 * x = 0 ↔ x = 0 ∨ x = -5/2 :=
by
  intro x
  sorry

end solve_quadratic_l105_105334


namespace ada_original_seat_l105_105706

theorem ada_original_seat {a b c d e : ℕ} : 
  (a ∈ {1, 5}) → 
  b = a + 2 → 
  c = a - 1 → 
  d ≠ e → 
  (a, 5) ∈ (1, 5) → 
  a = 2 :=
by
  -- Placeholder proof.
  sorry

end ada_original_seat_l105_105706


namespace middle_digit_base8_l105_105993

theorem middle_digit_base8 (M : ℕ) (e : ℕ) (d f : Fin 8) 
  (M_base8 : M = 64 * d + 8 * e + f)
  (M_base10 : M = 100 * f + 10 * e + d) :
  e = 6 :=
by sorry

end middle_digit_base8_l105_105993


namespace prisoners_can_be_freed_l105_105196

-- Condition: We have 100 prisoners and 100 drawers.
def prisoners : Nat := 100
def drawers : Nat := 100

-- Predicate to represent the strategy
def successful_strategy (strategy: (Fin prisoners) → (Fin drawers) → Bool) : Bool :=
  -- We use a hypothetical strategy function to model this
  (true) -- Placeholder for the actual strategy computation

-- Statement: Prove that there exists a strategy where all prisoners finding their names has a probability greater than 30%.
theorem prisoners_can_be_freed :
  ∃ strategy: (Fin prisoners) → (Fin drawers) → Bool, 
    (successful_strategy strategy) ∧ (0.3118 > 0.3) :=
sorry

end prisoners_can_be_freed_l105_105196


namespace tiffany_lives_after_bonus_stage_l105_105604

theorem tiffany_lives_after_bonus_stage :
  let initial_lives := 250
  let lives_lost := 58
  let remaining_lives := initial_lives - lives_lost
  let additional_lives := 3 * remaining_lives
  let final_lives := remaining_lives + additional_lives
  final_lives = 768 :=
by
  let initial_lives := 250
  let lives_lost := 58
  let remaining_lives := initial_lives - lives_lost
  let additional_lives := 3 * remaining_lives
  let final_lives := remaining_lives + additional_lives
  exact sorry

end tiffany_lives_after_bonus_stage_l105_105604


namespace max_two_alphas_l105_105170

theorem max_two_alphas (k : ℕ) (α : ℕ → ℝ) (hα : ∀ n, ∃! i p : ℕ, n = ⌊p * α i⌋ + 1) : k ≤ 2 := 
sorry

end max_two_alphas_l105_105170


namespace problem1_problem2_problem3_l105_105225

-- Proof for part 1
theorem problem1 (α : ℝ) (h : Real.tan α = 3) :
  (3 * Real.sin α + Real.cos α) / (Real.sin α - 2 * Real.cos α) = 10 :=
sorry

-- Proof for part 2
theorem problem2 (α : ℝ) :
  (-Real.sin (Real.pi + α) + Real.sin (-α) - Real.tan (2 * Real.pi + α)) / 
  (Real.tan (α + Real.pi) + Real.cos (-α) + Real.cos (Real.pi - α)) = -1 :=
sorry

-- Proof for part 3
theorem problem3 (α : ℝ) (h : Real.sin α + Real.cos α = 1 / 2) (hα : 0 < α ∧ α < Real.pi) :
  Real.sin α * Real.cos α = -3 / 8 :=
sorry

end problem1_problem2_problem3_l105_105225


namespace sqrt_value_l105_105392

theorem sqrt_value {A B C : ℝ} (x y : ℝ) 
  (h1 : A = 5 * Real.sqrt (2 * x + 1)) 
  (h2 : B = 3 * Real.sqrt (x + 3)) 
  (h3 : C = Real.sqrt (10 * x + 3 * y)) 
  (h4 : A + B = C) 
  (h5 : 2 * x + 1 = x + 3) : 
  Real.sqrt (2 * y - x^2) = 14 :=
by
  sorry

end sqrt_value_l105_105392


namespace total_rainbow_nerds_l105_105361

-- Definitions based on the conditions
def num_purple_candies : ℕ := 10
def num_yellow_candies : ℕ := num_purple_candies + 4
def num_green_candies : ℕ := num_yellow_candies - 2

-- The statement to be proved
theorem total_rainbow_nerds : num_purple_candies + num_yellow_candies + num_green_candies = 36 := by
  -- Using the provided definitions to automatically infer
  sorry

end total_rainbow_nerds_l105_105361


namespace problem_l105_105545

theorem problem (n : ℝ) (h : (n - 2009)^2 + (2008 - n)^2 = 1) : (n - 2009) * (2008 - n) = 0 := 
by
  sorry

end problem_l105_105545


namespace solve_equation_l105_105080

theorem solve_equation :
  ∀ y : ℤ, 4 * (y - 1) = 1 - 3 * (y - 3) → y = 2 :=
by
  intros y h
  sorry

end solve_equation_l105_105080


namespace moses_income_l105_105429

theorem moses_income (investment : ℝ) (percentage : ℝ) (dividend_rate : ℝ) (income : ℝ)
  (h1 : investment = 3000) (h2 : percentage = 0.72) (h3 : dividend_rate = 0.0504) :
  income = 210 :=
sorry

end moses_income_l105_105429


namespace total_volume_of_cubes_l105_105181

theorem total_volume_of_cubes :
  let sarah_side_length := 3
  let sarah_num_cubes := 8
  let tom_side_length := 4
  let tom_num_cubes := 4
  let sarah_volume := sarah_num_cubes * sarah_side_length^3
  let tom_volume := tom_num_cubes * tom_side_length^3
  sarah_volume + tom_volume = 472 := by
  -- Definitions coming from conditions
  let sarah_side_length := 3
  let sarah_num_cubes := 8
  let tom_side_length := 4
  let tom_num_cubes := 4
  let sarah_volume := sarah_num_cubes * sarah_side_length^3
  let tom_volume := tom_num_cubes * tom_side_length^3
  -- Total volume of all cubes
  have h : sarah_volume + tom_volume = 472 := by sorry

  exact h

end total_volume_of_cubes_l105_105181


namespace hexagon_diagonals_l105_105525

-- Define a hexagon as having 6 vertices
def hexagon_vertices : ℕ := 6

-- From one vertex of a hexagon, there are (6 - 1) vertices it can potentially connect to
def potential_connections (vertices : ℕ) : ℕ := vertices - 1

-- Remove the two adjacent vertices to count diagonals
def diagonals_from_vertex (connections : ℕ) : ℕ := connections - 2

theorem hexagon_diagonals : diagonals_from_vertex (potential_connections hexagon_vertices) = 3 := by
  -- The proof is intentionally left as a sorry placeholder.
  sorry

end hexagon_diagonals_l105_105525


namespace bridge_must_hold_weight_l105_105014

def weight_of_full_can (soda_weight empty_can_weight : ℕ) : ℕ :=
  soda_weight + empty_can_weight

def total_weight_of_full_cans (num_full_cans weight_per_full_can : ℕ) : ℕ :=
  num_full_cans * weight_per_full_can

def total_weight_of_empty_cans (num_empty_cans empty_can_weight : ℕ) : ℕ :=
  num_empty_cans * empty_can_weight

theorem bridge_must_hold_weight :
  let num_full_cans := 6
  let soda_weight := 12
  let empty_can_weight := 2
  let num_empty_cans := 2
  let weight_per_full_can := weight_of_full_can soda_weight empty_can_weight
  let total_full_cans_weight := total_weight_of_full_cans num_full_cans weight_per_full_can
  let total_empty_cans_weight := total_weight_of_empty_cans num_empty_cans empty_can_weight
  total_full_cans_weight + total_empty_cans_weight = 88 := by
  sorry

end bridge_must_hold_weight_l105_105014


namespace expression_remainder_l105_105835

theorem expression_remainder (n : ℤ) (h : n % 5 = 3) : (n + 1) % 5 = 4 :=
by
  sorry

end expression_remainder_l105_105835


namespace opposite_of_neg_two_l105_105795

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l105_105795


namespace mod_division_l105_105314

theorem mod_division (N : ℕ) (h₁ : N = 5 * 2 + 0) : N % 4 = 2 :=
by sorry

end mod_division_l105_105314


namespace base13_addition_l105_105998

/--
Given two numbers in base 13: 528₁₃ and 274₁₃, prove that their sum is 7AC₁₃.
-/
theorem base13_addition :
  let u1 := 8
  let t1 := 2
  let h1 := 5
  let u2 := 4
  let t2 := 7
  let h2 := 2
  -- Add the units digits: 8 + 4 = 12; 12 is C in base 13
  let s1 := 12 -- 'C' in base 13
  let carry1 := 1
  -- Add the tens digits along with the carry: 2 + 7 + 1 = 10; 10 is A in base 13
  let s2 := 10 -- 'A' in base 13
  -- Add the hundreds digits: 5 + 2 = 7
  let s3 := 7 -- 7 in base 13
  s1 = 12 ∧ s2 = 10 ∧ s3 = 7 :=
by
  sorry

end base13_addition_l105_105998


namespace bedroom_curtain_width_l105_105372

theorem bedroom_curtain_width
  (initial_fabric_area : ℕ)
  (living_room_curtain_area : ℕ)
  (fabric_left : ℕ)
  (bedroom_curtain_height : ℕ)
  (bedroom_curtain_area : ℕ)
  (bedroom_curtain_width : ℕ) :
  initial_fabric_area = 16 * 12 →
  living_room_curtain_area = 4 * 6 →
  fabric_left = 160 →
  bedroom_curtain_height = 4 →
  bedroom_curtain_area = 168 - 160 →
  bedroom_curtain_area = bedroom_curtain_width * bedroom_curtain_height →
  bedroom_curtain_width = 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Skipping the proof
  sorry

end bedroom_curtain_width_l105_105372


namespace opposite_of_neg_two_l105_105793

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l105_105793


namespace simplify_expression_l105_105041

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = 2 * x⁻¹ * y⁻¹ * z⁻¹ :=
by
  sorry

end simplify_expression_l105_105041


namespace geometric_sequence_a2_value_l105_105297

theorem geometric_sequence_a2_value
  (a : ℕ → ℝ)
  (a1 a2 a3 : ℝ)
  (h1 : a 1 = a1)
  (h2 : a 2 = a2)
  (h3 : a 3 = a3)
  (h_pos : ∀ n, 0 < a n)
  (h_geo : ∀ n, a (n + 1) = a 1 * (a 2) ^ n)
  (h_sum : a1 + a2 + a3 = 18)
  (h_inverse_sum : 1/a1 + 1/a2 + 1/a3 = 2)
  : a2 = 3 :=
sorry

end geometric_sequence_a2_value_l105_105297


namespace original_square_perimeter_l105_105367

theorem original_square_perimeter (p : ℕ) (x : ℕ) 
  (h1: p = 56) 
  (h2: 28 * x = p) : 4 * (2 * (x + 4 * x)) = 40 :=
by
  sorry

end original_square_perimeter_l105_105367


namespace range_of_a_l105_105259

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + 2 * x - Real.exp x

theorem range_of_a (h : ∀ m n : ℝ, 0 < m → 0 < n → m > n → (f a m - f a n) / (m - n) < 2) :
  a ≤ Real.exp 1 / (2 * 1) := 
sorry

end range_of_a_l105_105259


namespace length_AB_l105_105422

theorem length_AB :
  ∀ (A B : ℝ × ℝ) (k : ℝ),
    (A.2 = k * A.1 - 2) ∧ (B.2 = k * B.1 - 2) ∧ (A.2^2 = 8 * A.1) ∧ (B.2^2 = 8 * B.1) ∧
    ((A.1 + B.1) / 2 = 2) →
  dist A B = 2 * Real.sqrt 15 :=
by
  sorry

end length_AB_l105_105422


namespace reggie_father_money_l105_105443

theorem reggie_father_money :
  let books := 5
  let cost_per_book := 2
  let amount_left := 38
  books * cost_per_book + amount_left = 48 :=
by
  sorry

end reggie_father_money_l105_105443


namespace count_diff_squares_not_representable_1_to_1000_l105_105729

def num_not_diff_squares (n : ℕ) : ℕ :=
  (n + 1) / 4 * (if (n + 1) % 4 >= 2 then 1 else 0)

theorem count_diff_squares_not_representable_1_to_1000 :
  num_not_diff_squares 999 = 250 := 
sorry

end count_diff_squares_not_representable_1_to_1000_l105_105729


namespace circle_area_increase_l105_105287

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  percentage_increase = 125 := 
by {
  -- The proof will be written here.
  sorry
}

end circle_area_increase_l105_105287


namespace weight_of_each_bag_l105_105690

theorem weight_of_each_bag 
  (total_potatoes_weight : ℕ) (damaged_potatoes_weight : ℕ) 
  (bag_price : ℕ) (total_revenue : ℕ) (sellable_potatoes_weight : ℕ) (number_of_bags : ℕ) 
  (weight_of_each_bag : ℕ) :
  total_potatoes_weight = 6500 →
  damaged_potatoes_weight = 150 →
  sellable_potatoes_weight = total_potatoes_weight - damaged_potatoes_weight →
  bag_price = 72 →
  total_revenue = 9144 →
  number_of_bags = total_revenue / bag_price →
  weight_of_each_bag * number_of_bags = sellable_potatoes_weight →
  weight_of_each_bag = 50 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end weight_of_each_bag_l105_105690


namespace length_of_new_section_l105_105996

-- Definitions from the conditions
def area : ℕ := 35
def width : ℕ := 7

-- The problem statement
theorem length_of_new_section (h : area = 35 ∧ width = 7) : 35 / 7 = 5 :=
by
  -- We'll provide the proof later
  sorry

end length_of_new_section_l105_105996


namespace max_discount_rate_l105_105619

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l105_105619


namespace max_discount_rate_l105_105667

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l105_105667


namespace hexagon_colorings_correct_l105_105591

noncomputable def hexagon_colorings : Nat :=
  let colors := ["blue", "orange", "purple"]
  2 -- As determined by the solution.

theorem hexagon_colorings_correct :
  hexagon_colorings = 2 :=
by
  sorry

end hexagon_colorings_correct_l105_105591


namespace opposite_of_neg_two_l105_105800

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l105_105800


namespace unit_prices_max_helmets_A_l105_105072

open Nat Real

-- Given conditions
variables (x y : ℝ)
variables (m : ℕ)

def wholesale_price_A := 30
def wholesale_price_B := 20
def price_difference := 15
def revenue_A := 450
def revenue_B := 600
def total_helmets := 100
def budget := 2350

-- Part 1: Prove the unit prices of helmets A and B
theorem unit_prices :
  ∃ (price_A price_B : ℝ), 
    (price_A = price_B + price_difference) ∧ 
    (revenue_B / price_B = 2 * revenue_A / price_A) ∧
    (price_B = 30) ∧
    (price_A = 45) :=
by
  sorry

-- Part 2: Prove the maximum number of helmets of type A that can be purchased
theorem max_helmets_A :
  ∃ (m : ℕ), 
    (30 * m + 20 * (total_helmets - m) ≤ budget) ∧
    (m ≤ 35) :=
by
  sorry

end unit_prices_max_helmets_A_l105_105072


namespace find_set_T_l105_105584

namespace MathProof 

theorem find_set_T (S : Finset ℕ) (hS : ∀ x ∈ S, x > 0) :
  ∃ T : Finset ℕ, S ⊆ T ∧ ∀ x ∈ T, x ∣ (T.sum id) :=
by
  sorry

end MathProof 

end find_set_T_l105_105584


namespace find_x_l105_105981

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the proof goal
theorem find_x (x : ℝ) : 2 * f x - 19 = f (x - 4) → x = 4 :=
by
  sorry

end find_x_l105_105981


namespace food_consumption_reduction_l105_105548

noncomputable def reduction_factor (n p : ℝ) : ℝ :=
  (n * p) / ((n - 0.05 * n) * (p + 0.2 * p))

theorem food_consumption_reduction (n p : ℝ) (h : n > 0 ∧ p > 0) :
  (1 - reduction_factor n p) * 100 = 12.28 := by
  sorry

end food_consumption_reduction_l105_105548


namespace product_of_abc_l105_105205

noncomputable def abc_product (a b c : ℝ) : ℝ :=
  a * b * c

theorem product_of_abc (a b c m : ℝ) 
    (h1 : a + b + c = 300)
    (h2 : m = 5 * a)
    (h3 : m = b + 14)
    (h4 : m = c - 14) : 
    abc_product a b c = 664500 :=
by sorry

end product_of_abc_l105_105205


namespace count_diff_squares_not_representable_1_to_1000_l105_105730

def num_not_diff_squares (n : ℕ) : ℕ :=
  (n + 1) / 4 * (if (n + 1) % 4 >= 2 then 1 else 0)

theorem count_diff_squares_not_representable_1_to_1000 :
  num_not_diff_squares 999 = 250 := 
sorry

end count_diff_squares_not_representable_1_to_1000_l105_105730


namespace heroes_on_the_back_l105_105612

theorem heroes_on_the_back (total_heroes front_heroes : ℕ) (h1 : total_heroes = 9) (h2 : front_heroes = 2) :
  total_heroes - front_heroes = 7 := by
  sorry

end heroes_on_the_back_l105_105612


namespace albert_needs_more_money_l105_105688

-- Definitions derived from the problem conditions
def cost_paintbrush : ℝ := 1.50
def cost_paints : ℝ := 4.35
def cost_easel : ℝ := 12.65
def money_albert_has : ℝ := 6.50

-- Statement asserting the amount of money Albert needs
theorem albert_needs_more_money : (cost_paintbrush + cost_paints + cost_easel) - money_albert_has = 12 :=
by
  sorry

end albert_needs_more_money_l105_105688


namespace problem1_problem2_l105_105694

-- Problem 1: Prove the expression evaluates to 8
theorem problem1 : (1:ℝ) * (- (1 / 2)⁻¹) + (3 - Real.pi)^0 + (-3)^2 = 8 := 
by
  sorry

-- Problem 2: Prove the expression simplifies to 9a^6 - 2a^2
theorem problem2 (a : ℝ) : a^2 * a^4 - (-2 * a^2)^3 - 3 * a^2 + a^2 = 9 * a^6 - 2 * a^2 := 
by
  sorry

end problem1_problem2_l105_105694


namespace hypotenuse_length_l105_105293

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 := 
by
  sorry

end hypotenuse_length_l105_105293


namespace triangle_area_l105_105101

theorem triangle_area (a b c : ℕ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25)
  (right_triangle : a^2 + b^2 = c^2) : 
  (1/2 : ℚ) * (a * b) = 84 := 
by
  -- Sorry is used as we are only providing the statement, not the full proof.
  sorry

end triangle_area_l105_105101


namespace fractions_with_same_denominators_fractions_with_same_numerators_fractions_with_different_numerators_and_denominators_l105_105279

theorem fractions_with_same_denominators {a b c : ℤ} (h_c : c ≠ 0) :
  (a > b → a / (c:ℚ) > b / (c:ℚ)) ∧ (a < b → a / (c:ℚ) < b / (c:ℚ)) :=
by sorry

theorem fractions_with_same_numerators {a c d : ℤ} (h_c : c ≠ 0) (h_d : d ≠ 0) :
  (c < d → a / (c:ℚ) > a / (d:ℚ)) ∧ (c > d → a / (c:ℚ) < a / (d:ℚ)) :=
by sorry

theorem fractions_with_different_numerators_and_denominators {a b c d : ℤ} (h_c : c ≠ 0) (h_d : d ≠ 0) :
  a > b ∧ c < d → a / (c:ℚ) > b / (d:ℚ) :=
by sorry

end fractions_with_same_denominators_fractions_with_same_numerators_fractions_with_different_numerators_and_denominators_l105_105279


namespace math_scores_120_or_higher_l105_105680

-- Define the normal distribution with mean 110 and variance 100
def examScoresDistribution : ProbabilityMassFunction ℝ :=
  ProbabilityMassFunction.normal (110 : ℝ) (10 : ℝ)

-- Given condition
axiom condition1 : ∀ (x : ℝ), 100 ≤ x ∧ x ≤ 110 → examScoresDistribution.prob_mass x = 0.34

noncomputable def numberOfStudents (total_students: ℕ) : ℕ :=
  let P_x_ge_120 := (1 - 0.34) / 2
  P_x_ge_120 * total_students

-- We need to show that if the conditions hold, then the number of students scoring 120 or higher is 8
theorem math_scores_120_or_higher :
  let total_students := 50 in
  numberOfStudents total_students = 8 :=
by sorry

end math_scores_120_or_higher_l105_105680


namespace cube_face_expression_l105_105691

theorem cube_face_expression (a b c : ℤ) (h1 : 3 * a + 2 = 17) (h2 : 7 * b - 4 = 10) (h3 : a + 3 * b - 2 * c = 11) : 
  a - b * c = 5 :=
by sorry

end cube_face_expression_l105_105691


namespace old_hen_weight_unit_l105_105999

theorem old_hen_weight_unit (w : ℕ) (units : String) (opt1 opt2 opt3 opt4 : String)
  (h_opt1 : opt1 = "grams") (h_opt2 : opt2 = "kilograms") (h_opt3 : opt3 = "tons") (h_opt4 : opt4 = "meters") (h_w : w = 2) : 
  (units = opt2) :=
sorry

end old_hen_weight_unit_l105_105999


namespace circle_intersection_zero_l105_105906

theorem circle_intersection_zero :
  (∀ θ : ℝ, ∀ r1 : ℝ, r1 = 3 * Real.cos θ → ∀ r2 : ℝ, r2 = 6 * Real.sin (2 * θ) → False) :=
by 
  sorry

end circle_intersection_zero_l105_105906


namespace probability_A_and_B_same_last_hour_l105_105828
open Classical

-- Define the problem conditions
def attraction_count : ℕ := 6
def total_scenarios : ℕ := attraction_count * attraction_count
def favorable_scenarios : ℕ := attraction_count

-- Define the probability calculation
def probability_same_attraction : ℚ := favorable_scenarios / total_scenarios

-- The proof problem statement
theorem probability_A_and_B_same_last_hour : 
  probability_same_attraction = 1 / 6 :=
sorry

end probability_A_and_B_same_last_hour_l105_105828


namespace find_fraction_l105_105606

noncomputable def condition_eq : ℝ := 5
noncomputable def condition_gq : ℝ := 7

theorem find_fraction {FQ HQ : ℝ} (h : condition_eq * FQ = condition_gq * HQ) :
  FQ / HQ = 7 / 5 :=
by
  have eq_mul : condition_eq = 5 := by rfl
  have gq_mul : condition_gq = 7 := by rfl
  rw [eq_mul, gq_mul] at h
  have h': 5 * FQ = 7 * HQ := h
  field_simp [←h']
  sorry

end find_fraction_l105_105606


namespace find_train_speed_l105_105099

def train_speed (v t_pole t_stationary d_stationary : ℕ) : ℕ := v

theorem find_train_speed (v : ℕ) (t_pole : ℕ) (t_stationary : ℕ) (d_stationary : ℕ) :
  t_pole = 5 →
  t_stationary = 25 →
  d_stationary = 360 →
  25 * v = 5 * v + d_stationary →
  v = 18 :=
by intros h1 h2 h3 h4; sorry

end find_train_speed_l105_105099


namespace intersection_A_B_l105_105141

namespace MathProof

open Set

def A := {y : ℝ | ∃ x : ℝ, y = x^2 - 2 * x}
def B := {y : ℝ | ∃ x : ℝ, y = -x^2 + 2 * x + 6}

theorem intersection_A_B : A ∩ B = Icc (-1 : ℝ) 7 :=
by
  sorry

end MathProof

end intersection_A_B_l105_105141


namespace decompose_x_l105_105074

def x : ℝ × ℝ × ℝ := (6, 12, -1)
def p : ℝ × ℝ × ℝ := (1, 3, 0)
def q : ℝ × ℝ × ℝ := (2, -1, 1)
def r : ℝ × ℝ × ℝ := (0, -1, 2)

theorem decompose_x :
  x = (4 : ℝ) • p + q - r :=
sorry

end decompose_x_l105_105074


namespace intersection_A_B_l105_105927

def A : Set ℝ := {x | 1 < x}
def B : Set ℝ := {y | y ≤ 2}
def expected_intersection : Set ℝ := {z | 1 < z ∧ z ≤ 2}

theorem intersection_A_B : (A ∩ B) = expected_intersection :=
by
  -- Proof to be completed
  sorry

end intersection_A_B_l105_105927


namespace minimum_value_x_2y_l105_105394

theorem minimum_value_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = x * y) : x + 2 * y = 8 :=
sorry

end minimum_value_x_2y_l105_105394


namespace max_discount_rate_l105_105657

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l105_105657


namespace sum_to_common_fraction_l105_105880

theorem sum_to_common_fraction :
  (2 / 10 + 3 / 100 + 4 / 1000 + 5 / 10000 + 6 / 100000 : ℚ) = 733 / 12500 := by
  sorry

end sum_to_common_fraction_l105_105880


namespace terms_before_five_l105_105731

theorem terms_before_five (a₁ : ℤ) (d : ℤ) (n : ℤ) :
  a₁ = 75 → d = -5 → (a₁ + (n - 1) * d = 5) → n - 1 = 14 :=
by
  intros h1 h2 h3
  sorry

end terms_before_five_l105_105731


namespace max_discount_rate_l105_105622

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l105_105622


namespace candy_ratio_l105_105102

theorem candy_ratio (chocolate_bars M_and_Ms marshmallows total_candies : ℕ)
  (h1 : chocolate_bars = 5)
  (h2 : M_and_Ms = 7 * chocolate_bars)
  (h3 : total_candies = 25 * 10)
  (h4 : marshmallows = total_candies - chocolate_bars - M_and_Ms) :
  marshmallows / M_and_Ms = 6 :=
by
  sorry

end candy_ratio_l105_105102


namespace option_b_is_correct_l105_105219

def is_linear (equation : String) : Bool :=
  -- Pretend implementation that checks if the given equation is linear
  -- This function would parse the string and check the linearity condition
  true -- This should be replaced by actual linearity check

def has_two_unknowns (system : List String) : Bool :=
  -- Pretend implementation that checks if the system contains exactly two unknowns
  -- This function would analyze the variables in the system
  true -- This should be replaced by actual unknowns count check

def is_system_of_two_linear_equations (system : List String) : Bool :=
  -- Checking both conditions: Each equation is linear and contains exactly two unknowns
  (system.all is_linear) && (has_two_unknowns system)

def option_b := ["x + y = 1", "x - y = 2"]

theorem option_b_is_correct :
  is_system_of_two_linear_equations option_b := 
  by
    unfold is_system_of_two_linear_equations
    -- Assuming the placeholder implementations of is_linear and has_two_unknowns
    -- actually verify the required properties, this should be true
    sorry

end option_b_is_correct_l105_105219


namespace max_discount_rate_l105_105617

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l105_105617


namespace simplify_sub_polynomials_l105_105446

def f (r : ℝ) : ℝ := 2 * r^3 + r^2 + 5 * r - 4
def g (r : ℝ) : ℝ := r^3 + 3 * r^2 + 7 * r - 2

theorem simplify_sub_polynomials (r : ℝ) : f r - g r = r^3 - 2 * r^2 - 2 * r - 2 := by
  sorry

end simplify_sub_polynomials_l105_105446


namespace circumscribed_circle_area_l105_105989

theorem circumscribed_circle_area (side_length : ℝ) (h : side_length = 12) :
  ∃ (A : ℝ), A = 48 * π :=
by
  sorry

end circumscribed_circle_area_l105_105989


namespace elena_alex_total_dollars_l105_105513

theorem elena_alex_total_dollars :
  (5 / 6 : ℚ) + (7 / 15 : ℚ) = (13 / 10 : ℚ) :=
by
    sorry

end elena_alex_total_dollars_l105_105513


namespace interview_passing_probability_l105_105746

def probability_of_passing_interview (p : ℝ) : ℝ :=
  p + (1 - p) * p + (1 - p) * (1 - p) * p

theorem interview_passing_probability : probability_of_passing_interview 0.7 = 0.973 :=
by
  -- proof steps to be filled
  sorry

end interview_passing_probability_l105_105746


namespace ratio_of_sum_of_terms_l105_105420

theorem ratio_of_sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 5 / 9) : S 9 / S 5 = 1 := 
  sorry

end ratio_of_sum_of_terms_l105_105420


namespace opposite_of_neg_2_is_2_l105_105808

theorem opposite_of_neg_2_is_2 : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_2_is_2_l105_105808


namespace max_discount_rate_l105_105661

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l105_105661


namespace find_20_paise_coins_l105_105614

theorem find_20_paise_coins (x y : ℕ) (h1 : x + y = 324) (h2 : 20 * x + 25 * y = 7100) : x = 200 :=
by
  -- Given the conditions, we need to prove x = 200.
  -- Steps and proofs are omitted here.
  sorry

end find_20_paise_coins_l105_105614


namespace workers_in_workshop_l105_105465

theorem workers_in_workshop (W : ℕ) (h1 : W ≤ 100) (h2 : W % 3 = 0) (h3 : W % 25 = 0)
  : W = 75 ∧ W / 3 = 25 ∧ W * 8 / 100 = 6 :=
by
  sorry

end workers_in_workshop_l105_105465


namespace ratio_of_numbers_l105_105960

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : 0 < a) (h3 : b < a) (h4 : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
sorry

end ratio_of_numbers_l105_105960


namespace cat_finishes_food_on_next_wednesday_l105_105583

def cat_food_consumption_per_day : ℚ :=
  (1 / 4) + (1 / 6)

def total_food_on_day (n : ℕ) : ℚ :=
  n * cat_food_consumption_per_day

def total_cans : ℚ := 8

theorem cat_finishes_food_on_next_wednesday :
  total_food_on_day 10 = total_cans := sorry

end cat_finishes_food_on_next_wednesday_l105_105583


namespace suraj_average_increase_l105_105077

namespace SurajAverage

theorem suraj_average_increase (A : ℕ) (h : (16 * A + 112) / 17 = A + 6) : (A + 6) = 16 :=
  by
  sorry

end SurajAverage

end suraj_average_increase_l105_105077


namespace curve_to_polar_l105_105959

noncomputable def polar_eq_of_curve (x y : ℝ) (ρ θ : ℝ) : Prop :=
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ (x ^ 2 + y ^ 2 - 2 * x = 0) → (ρ = 2 * Real.cos θ)

theorem curve_to_polar (x y ρ θ : ℝ) :
  polar_eq_of_curve x y ρ θ :=
sorry

end curve_to_polar_l105_105959


namespace total_percent_decrease_baseball_card_l105_105357

theorem total_percent_decrease_baseball_card
  (original_value : ℝ)
  (first_year_decrease : ℝ := 0.20)
  (second_year_decrease : ℝ := 0.30)
  (value_after_first_year : ℝ := original_value * (1 - first_year_decrease))
  (final_value : ℝ := value_after_first_year * (1 - second_year_decrease))
  (total_percent_decrease : ℝ := ((original_value - final_value) / original_value) * 100) :
  total_percent_decrease = 44 :=
by 
  sorry

end total_percent_decrease_baseball_card_l105_105357


namespace arithmetic_contains_geometric_progression_l105_105439

theorem arithmetic_contains_geometric_progression (a d : ℕ) (h_pos : d > 0) :
  ∃ (a' : ℕ) (r : ℕ), a' = a ∧ r = 1 + d ∧ (∀ k : ℕ, ∃ n : ℕ, a' * r^k = a + (n-1)*d) :=
by
  sorry

end arithmetic_contains_geometric_progression_l105_105439


namespace max_discount_rate_l105_105648

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l105_105648


namespace solve_equation_1_solve_equation_2_solve_equation_3_l105_105042

theorem solve_equation_1 (x : ℝ) : (x^2 - 3 * x = 0) ↔ (x = 0 ∨ x = 3) := sorry

theorem solve_equation_2 (x : ℝ) : (4 * x^2 - x - 5 = 0) ↔ (x = 5/4 ∨ x = -1) := sorry

theorem solve_equation_3 (x : ℝ) : (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2/3) := sorry

end solve_equation_1_solve_equation_2_solve_equation_3_l105_105042


namespace pythagorean_triple_divisibility_l105_105935

theorem pythagorean_triple_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (∃ k₃, k₃ ∣ a ∨ k₃ ∣ b) ∧
  (∃ k₄, k₄ ∣ a ∨ k₄ ∣ b ∧ 2 ∣ k₄) ∧
  (∃ k₅, k₅ ∣ a ∨ k₅ ∣ b ∨ k₅ ∣ c) :=
by
  sorry

end pythagorean_triple_divisibility_l105_105935


namespace talia_total_distance_l105_105775

-- Definitions from the conditions
def distance_house_to_park : ℝ := 5
def distance_park_to_store : ℝ := 3
def distance_store_to_house : ℝ := 8

-- The theorem to be proven
theorem talia_total_distance : distance_house_to_park + distance_park_to_store + distance_store_to_house = 16 := by
  sorry

end talia_total_distance_l105_105775


namespace min_value_t2_t1_l105_105254

open Real

noncomputable def h (t : ℝ) : ℝ := 
  if t > 1 / 2 then exp (t - 1) else 0

noncomputable def g (t : ℝ) : ℝ := 
  if t > 1 / 2 then log (2 * t - 1) + 2 else 0

theorem min_value_t2_t1 : ∃ t1 t2 : ℝ, t1 > 1/2 ∧ t2 > 1/2 ∧ h t1 = g t2 ∧ (t2 - t1 = -log 2) :=
by
  sorry

end min_value_t2_t1_l105_105254


namespace min_value_y_l105_105459

noncomputable def y (x : ℝ) : ℝ := x + 4 / (x - 1)

theorem min_value_y : ∀ x > 1, y x ≥ 5 ∧ (y x = 5 ↔ x = 3) :=
by
  intros x hx
  sorry

end min_value_y_l105_105459


namespace correct_operation_l105_105971

theorem correct_operation (a b : ℝ) :
  2 * a^2 * b - 3 * a^2 * b = - (a^2 * b) :=
by
  sorry

end correct_operation_l105_105971


namespace parallelogram_theorem_l105_105162

noncomputable def parallelogram (A B C D O : Type) (θ : ℝ) :=
  let DBA := θ
  let DBC := 3 * θ
  let CAB := 9 * θ
  let ACB := 180 - (9 * θ + 3 * θ)
  let AOB := 180 - 12 * θ
  let s := ACB / AOB
  s = 4 / 5

theorem parallelogram_theorem (A B C D O : Type) (θ : ℝ) 
  (h1: θ > 0): parallelogram A B C D O θ := by
  sorry

end parallelogram_theorem_l105_105162


namespace max_discount_rate_l105_105671

theorem max_discount_rate (c p m : ℝ) (h₀ : c = 4) (h₁ : p = 5) (h₂ : m = 0.4) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (p * (1 - x / 100) - c ≥ m) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l105_105671


namespace opposite_of_neg_two_l105_105798

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l105_105798


namespace chengdu_chongqing_scientific_notation_l105_105742

theorem chengdu_chongqing_scientific_notation:
  (185000 : ℝ) = 1.85 * 10^5 :=
sorry

end chengdu_chongqing_scientific_notation_l105_105742


namespace opposite_of_neg_two_l105_105782

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l105_105782


namespace wendy_third_day_miles_l105_105932

theorem wendy_third_day_miles (miles_day1 miles_day2 total_miles : ℕ)
  (h1 : miles_day1 = 125)
  (h2 : miles_day2 = 223)
  (h3 : total_miles = 493) :
  total_miles - (miles_day1 + miles_day2) = 145 :=
by sorry

end wendy_third_day_miles_l105_105932


namespace find_n_l105_105705

theorem find_n (n : ℤ) (h : n * 1296 / 432 = 36) : n = 12 :=
sorry

end find_n_l105_105705


namespace cost_of_chairs_l105_105032

-- Given conditions
def total_spent : ℕ := 56
def cost_of_table : ℕ := 34
def number_of_chairs : ℕ := 2

-- The target definition
def cost_of_one_chair : ℕ := 11

-- Statement to prove
theorem cost_of_chairs (N : ℕ) (T : ℕ) (C : ℕ) (x : ℕ) (hN : N = total_spent) (hT : T = cost_of_table) (hC : C = number_of_chairs) : x = cost_of_one_chair ↔ N - T = C * x :=
by
  sorry

end cost_of_chairs_l105_105032


namespace sum_of_decimals_as_fraction_l105_105876

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 733 / 3125 := by
  sorry

end sum_of_decimals_as_fraction_l105_105876


namespace find_certain_number_l105_105081

theorem find_certain_number : 
  ∃ (certain_number : ℕ), 1038 * certain_number = 173 * 240 ∧ certain_number = 40 :=
by
  sorry

end find_certain_number_l105_105081


namespace leo_total_points_l105_105914

theorem leo_total_points (x y : ℕ) (h1 : x + y = 50) :
  0.4 * (x : ℝ) * 3 + 0.5 * (y : ℝ) * 2 = 0.2 * (x : ℝ) + 50 :=
by sorry

end leo_total_points_l105_105914


namespace determinant_evaluation_l105_105869

theorem determinant_evaluation (x z : ℝ) :
  (Matrix.det ![
    ![1, x, z],
    ![1, x + z, z],
    ![1, x, x + z]
  ]) = x * z - z * z := 
sorry

end determinant_evaluation_l105_105869


namespace fraction_of_second_year_students_not_declared_major_l105_105929

theorem fraction_of_second_year_students_not_declared_major (T : ℕ) :
  (1 / 2 : ℝ) * (1 - (1 / 3 * (1 / 5))) = 7 / 15 :=
by
  sorry

end fraction_of_second_year_students_not_declared_major_l105_105929


namespace role_of_scatter_plot_correct_l105_105330

-- Definitions for problem context
def role_of_scatter_plot (role : String) : Prop :=
  role = "Roughly judging whether variables are linearly related"

-- Problem and conditions
theorem role_of_scatter_plot_correct :
  role_of_scatter_plot "Roughly judging whether variables are linearly related" :=
by 
  sorry

end role_of_scatter_plot_correct_l105_105330


namespace problem1_l105_105117

theorem problem1 : 3 * 403 + 5 * 403 + 2 * 403 + 401 = 4431 := by 
  sorry

end problem1_l105_105117


namespace sum_of_squares_eq_expansion_l105_105938

theorem sum_of_squares_eq_expansion (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 :=
sorry

end sum_of_squares_eq_expansion_l105_105938


namespace greatest_possible_remainder_l105_105143

theorem greatest_possible_remainder (x : ℕ) (h: x % 7 ≠ 0) : (∃ r < 7, r = x % 7) ∧ x % 7 ≤ 6 := by
  sorry

end greatest_possible_remainder_l105_105143


namespace max_discount_rate_l105_105635

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l105_105635


namespace prob_at_least_one_wrong_l105_105097

-- Defining the conditions in mathlib
def prob_wrong : ℝ := 0.1
def num_questions : ℕ := 3

-- Proving the main statement
theorem prob_at_least_one_wrong : 1 - (1 - prob_wrong) ^ num_questions = 0.271 := by
  sorry

end prob_at_least_one_wrong_l105_105097


namespace statement_B_statement_D_l105_105532

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.cos x + Real.sqrt 3 * Real.sin x) - Real.sqrt 3 + 1

theorem statement_B (x₁ x₂ : ℝ) (h1 : -π / 12 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 5 * π / 12) :
  f x₁ < f x₂ := sorry

theorem statement_D (x₁ x₂ x₃ : ℝ) (h1 : π / 3 ≤ x₁) (h2 : x₁ ≤ π / 2) (h3 : π / 3 ≤ x₂) (h4 : x₂ ≤ π / 2) (h5 : π / 3 ≤ x₃) (h6 : x₃ ≤ π / 2) :
  f x₁ + f x₂ - f x₃ > 2 := sorry

end statement_B_statement_D_l105_105532


namespace power_mod_l105_105518

theorem power_mod (n : ℕ) : (3 ^ 2017) % 17 = 3 := 
by
  sorry

end power_mod_l105_105518


namespace min_value_fraction_l105_105460

variables {X : ℝ → ℝ} {σ m n : ℝ}

/-- X is normally distributed with mean 10 and variance σ², and 
      P(X > 12) = m, P(8 ≤ X ≤ 10) = n. 
    We need to show: 
      min (2 / m + 1 / n) = 6 + 4 * Real.sqrt 2. -/
theorem min_value_fraction (hX : ∀ x, X x ∼ Normal 10 (σ^2))
    (hPm : P(λ x, X x > 12) = m)
    (hPn : P(λ x, 8 ≤ X x ∧ X x ≤ 10) = n) : 
    6 + 4 * Real.sqrt 2 ≤ 2 / m + 1 / n :=
sorry

end min_value_fraction_l105_105460


namespace water_usage_eq_13_l105_105228

theorem water_usage_eq_13 (m x : ℝ) (h : 16 * m = 10 * m + (x - 10) * 2 * m) : x = 13 :=
by sorry

end water_usage_eq_13_l105_105228


namespace cannot_form_triangle_l105_105972

theorem cannot_form_triangle {a b c : ℝ} (h1 : a = 2) (h2 : b = 3) (h3 : c = 6) : 
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by
  sorry

end cannot_form_triangle_l105_105972


namespace opposite_of_neg_two_l105_105784

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l105_105784


namespace sum_to_common_fraction_l105_105881

theorem sum_to_common_fraction :
  (2 / 10 + 3 / 100 + 4 / 1000 + 5 / 10000 + 6 / 100000 : ℚ) = 733 / 12500 := by
  sorry

end sum_to_common_fraction_l105_105881


namespace negation_of_proposition_l105_105593

theorem negation_of_proposition (x y : ℝ): (x + y > 0 → x > 0 ∧ y > 0) ↔ ¬ ((x + y ≤ 0) → (x ≤ 0 ∨ y ≤ 0)) :=
by sorry

end negation_of_proposition_l105_105593


namespace determine_fake_coin_weight_l105_105064

theorem determine_fake_coin_weight
  (coins : Fin 25 → ℤ) 
  (fake_coin : Fin 25) 
  (all_same_weight : ∀ (i j : Fin 25), i ≠ fake_coin → j ≠ fake_coin → coins i = coins j)
  (fake_diff_weight : ∃ (x : Fin 25), (coins x ≠ coins fake_coin)) :
  ∃ (is_heavy : Bool), 
    (is_heavy = true ↔ coins fake_coin > coins (Fin.ofNat 0)) ∨ 
    (is_heavy = false ↔ coins fake_coin < coins (Fin.ofNat 0)) :=
  sorry

end determine_fake_coin_weight_l105_105064


namespace number_of_roots_l105_105019

def S : Set ℚ := { x : ℚ | 0 < x ∧ x < (5 : ℚ)/8 }

def f (x : ℚ) : ℚ := 
  match x.num, x.den with
  | num, den => num / den + 1

theorem number_of_roots (h : ∀ q p, (p, q) = 1 → (q : ℚ) / p ∈ S → ((q + 1 : ℚ) / p = (2 : ℚ) / 3)) :
  ∃ n : ℕ, n = 7 :=
sorry

end number_of_roots_l105_105019


namespace kelly_games_left_l105_105564

-- Definitions based on conditions
def original_games := 80
def additional_games := 31
def games_to_give_away := 105

-- Total games after finding more games
def total_games := original_games + additional_games

-- Number of games left after giving away
def games_left := total_games - games_to_give_away

-- Theorem statement
theorem kelly_games_left : games_left = 6 :=
by
  -- The proof will be here
  sorry

end kelly_games_left_l105_105564


namespace profit_percentage_l105_105859

theorem profit_percentage (SP CP : ℕ) (h₁ : SP = 800) (h₂ : CP = 640) : (SP - CP) / CP * 100 = 25 :=
by 
  sorry

end profit_percentage_l105_105859


namespace prob_all_four_even_dice_l105_105215

noncomputable def probability_even (n : ℕ) : ℚ := (3 / 6)^n

theorem prob_all_four_even_dice : probability_even 4 = 1 / 16 := 
by
  sorry

end prob_all_four_even_dice_l105_105215


namespace circle_radius_5_l105_105711

theorem circle_radius_5 (k : ℝ) : 
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ↔ k = -40 :=
by
  sorry

end circle_radius_5_l105_105711


namespace min_value_frac_l105_105535

theorem min_value_frac (a c : ℝ) (h1 : 0 < a) (h2 : 0 < c) (h3 : a * c = 4) : 
  ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, y = (1 / c + 9 / a) → y ≥ x :=
by sorry

end min_value_frac_l105_105535


namespace largest_of_seven_consecutive_numbers_l105_105485

theorem largest_of_seven_consecutive_numbers (avg : ℕ) (h : avg = 20) :
  ∃ n : ℕ, n + 6 = 23 := 
by
  sorry

end largest_of_seven_consecutive_numbers_l105_105485


namespace player_one_win_l105_105601

theorem player_one_win (total_coins : ℕ) (coins_taken_by_player1 : ℕ) 
  (player1_moves : ∀ coins_remaining : ℕ, (1 ≤ coins_remaining ∧ coins_remaining ≤ 99) → coins_remaining % 2 = 1)
  (player2_moves : ∀ coins_remaining : ℕ, (2 ≤ coins_remaining ∧ coins_remaining ≤ 100) → coins_remaining % 2 = 0) :
  total_coins = 2015 → coins_taken_by_player1 = 95 → 
  ∃ strategy, ∀ turn : ℕ, (turn % 2 = 1 → strategy turn coins_taken_by_player1 = true) ∧ (turn % 2 = 0 → strategy turn coins_taken_by_player1 = false) :=
begin
  sorry
end

end player_one_win_l105_105601


namespace computer_price_increase_l105_105153

theorem computer_price_increase (d : ℝ) (h : 2 * d = 585) : d * 1.2 = 351 := by
  sorry

end computer_price_increase_l105_105153


namespace total_rainbow_nerds_l105_105362

-- Definitions based on the conditions
def num_purple_candies : ℕ := 10
def num_yellow_candies : ℕ := num_purple_candies + 4
def num_green_candies : ℕ := num_yellow_candies - 2

-- The statement to be proved
theorem total_rainbow_nerds : num_purple_candies + num_yellow_candies + num_green_candies = 36 := by
  -- Using the provided definitions to automatically infer
  sorry

end total_rainbow_nerds_l105_105362


namespace ada_original_seat_l105_105707

-- Define the problem conditions
def initial_seats : List ℕ := [1, 2, 3, 4, 5]  -- seat numbers

def bea_move (seat : ℕ) : ℕ := seat + 2  -- Bea moves 2 seats to the right
def ceci_move (seat : ℕ) : ℕ := seat - 1  -- Ceci moves 1 seat to the left
def switch (seats : (ℕ × ℕ)) : (ℕ × ℕ) := (seats.2, seats.1)  -- Dee and Edie switch seats

-- The final seating positions (end seats are 1 or 5 for Ada)
axiom ada_end_seat : ∃ final_seat : ℕ, final_seat ∈ [1, 5]  -- Ada returns to an end seat

-- Prove Ada was originally sitting in seat 2
theorem ada_original_seat (final_seat : ℕ) (h₁ : ∃ (s₁ s₂ : ℕ), s₁ ≠ s₂ ∧ bea_move s₁ ≠ final_seat ∧ ceci_move s₂ ≠ final_seat ∧ switch (s₁, s₂).2 ≠ final_seat) : 2 ∈ initial_seats :=
by
  sorry

end ada_original_seat_l105_105707


namespace opposite_of_neg_two_is_two_l105_105790

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l105_105790


namespace milk_left_is_correct_l105_105023

noncomputable def lily_initial_milk : ℚ := 5
noncomputable def milk_given_to_james : ℚ := 11 / 4

theorem milk_left_is_correct : lily_initial_milk - milk_given_to_james = 9 / 4 :=
by
  sorry

end milk_left_is_correct_l105_105023


namespace perimeter_of_regular_polygon_l105_105365

theorem perimeter_of_regular_polygon (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ)
  (h1 : n = 3) (h2 : side_length = 5) (h3 : exterior_angle = 120) : 
  n * side_length = 15 :=
by
  sorry

end perimeter_of_regular_polygon_l105_105365


namespace sum_of_digits_of_greatest_prime_divisor_of_4095_l105_105831

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def greatest_prime_divisor_of_4095 : ℕ := 13

theorem sum_of_digits_of_greatest_prime_divisor_of_4095 :
  sum_of_digits greatest_prime_divisor_of_4095 = 4 := by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_4095_l105_105831


namespace difference_in_ages_27_l105_105953

def conditions (a b : ℕ) : Prop :=
  10 * b + a = (1 / 2) * (10 * a + b) + 6 ∧
  10 * a + b + 2 = 5 * (10 * b + a - 4)

theorem difference_in_ages_27 {a b : ℕ} (h : conditions a b) :
  (10 * a + b) - (10 * b + a) = 27 :=
sorry

end difference_in_ages_27_l105_105953


namespace maximum_discount_rate_l105_105626

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l105_105626


namespace solution_set_of_inequality_l105_105204

theorem solution_set_of_inequality (x : ℝ) : x^2 - 5 * |x| + 6 < 0 ↔ (-3 < x ∧ x < -2) ∨ (2 < x ∧ x < 3) :=
  sorry

end solution_set_of_inequality_l105_105204


namespace notebooks_left_l105_105178

theorem notebooks_left (bundles : ℕ) (notebooks_per_bundle : ℕ) (groups : ℕ) (students_per_group : ℕ) : 
  bundles = 5 ∧ notebooks_per_bundle = 25 ∧ groups = 8 ∧ students_per_group = 13 →
  bundles * notebooks_per_bundle - groups * students_per_group = 21 := 
by sorry

end notebooks_left_l105_105178


namespace inscribed_square_side_length_l105_105159

theorem inscribed_square_side_length (a h : ℝ) (ha_pos : 0 < a) (hh_pos : 0 < h) :
    ∃ x : ℝ, x = (h * a) / (a + h) :=
by
  -- Code here demonstrates the existence of x such that x = (h * a) / (a + h)
  sorry

end inscribed_square_side_length_l105_105159


namespace mary_needs_more_sugar_l105_105762

def recipe_sugar := 14
def sugar_already_added := 2
def sugar_needed := recipe_sugar - sugar_already_added

theorem mary_needs_more_sugar : sugar_needed = 12 := by
  sorry

end mary_needs_more_sugar_l105_105762


namespace find_number_l105_105282

theorem find_number (a b x : ℝ) (H1 : 2 * a = x * b) (H2 : a * b ≠ 0) (H3 : (a / 3) / (b / 2) = 1) : x = 3 :=
by
  sorry

end find_number_l105_105282


namespace find_function_α_l105_105169

theorem find_function_α (α : ℝ) (hα : 0 < α) 
  (f : ℕ+ → ℝ) (h : ∀ k m : ℕ+, α * m ≤ k ∧ k < (α + 1) * m → f (k + m) = f k + f m) :
  ∃ b : ℝ, ∀ n : ℕ+, f n = b * n :=
sorry

end find_function_α_l105_105169


namespace opposite_of_neg_two_l105_105783

theorem opposite_of_neg_two : ∃ x : ℤ, (-2 + x = 0) ∧ x = 2 := 
begin
  sorry
end

end opposite_of_neg_two_l105_105783


namespace cost_per_ball_is_two_l105_105028

def cost_per_tennis_ball (packs : ℕ) (balls_per_pack : ℕ) (total_cost : ℕ) : ℝ :=
  total_cost / (packs * balls_per_pack)

theorem cost_per_ball_is_two :
  (cost_per_tennis_ball 4 3 24) = 2 :=
by
  sorry

end cost_per_ball_is_two_l105_105028


namespace sector_arc_length_l105_105716

theorem sector_arc_length (r : ℝ) (θ : ℝ) (L : ℝ) (h₁ : r = 1) (h₂ : θ = 60 * π / 180) : L = π / 3 :=
by
  sorry

end sector_arc_length_l105_105716


namespace gcd_7854_13843_l105_105693

theorem gcd_7854_13843 : Nat.gcd 7854 13843 = 1 := 
  sorry

end gcd_7854_13843_l105_105693


namespace ratio_area_triangle_circle_l105_105855

open Real

theorem ratio_area_triangle_circle
  (l r : ℝ)
  (h : ℝ := sqrt 2 * l)
  (h_eq_perimeter : 2 * l + h = 2 * π * r) :
  (1 / 2 * l^2) / (π * r^2) = (π * (3 - 2 * sqrt 2)) / 2 :=
by
  sorry

end ratio_area_triangle_circle_l105_105855


namespace opposite_of_neg_two_is_two_l105_105812

theorem opposite_of_neg_two_is_two (a : ℤ) (h : a = -2) : a + 2 = 0 := by
  rw [h]
  norm_num

end opposite_of_neg_two_is_two_l105_105812


namespace supermarket_profit_and_discount_l105_105227

theorem supermarket_profit_and_discount :
  ∃ (x : ℕ) (nB1 nB2 : ℕ) (discount_rate : ℝ),
    22*x + 30*(nB1) = 6000 ∧
    nB1 = (1 / 2 : ℝ) * x + 15 ∧
    150 * (29 - 22) + 90 * (40 - 30) = 1950 ∧
    nB2 = 3 * nB1 ∧
    150 * (29 - 22) + 270 * (40 * (1 - discount_rate / 100) - 30) = 2130 ∧
    discount_rate = 8.5 := sorry

end supermarket_profit_and_discount_l105_105227


namespace max_min_trig_expression_correct_l105_105756

noncomputable def max_min_trig_expression (a b : ℝ) : ℝ × ℝ :=
(let max_value := Real.sqrt (a^2 + b^2) in
 let min_value := - Real.sqrt (a^2 + b^2) in
 (max_value, min_value))

theorem max_min_trig_expression_correct (a b : ℝ) :
  max_min_trig_expression a b = (Real.sqrt (a^2 + b^2), -Real.sqrt (a^2 + b^2)) :=
by
  sorry

end max_min_trig_expression_correct_l105_105756


namespace intersection_of_A_and_B_range_of_a_l105_105112

open Set

namespace ProofProblem

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | x ≥ 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 2 ≤ x ∧ x < 3} := 
sorry

theorem range_of_a (a : ℝ) :
  (B ∪ C a) = C a → a ≤ 3 :=
sorry

end ProofProblem

end intersection_of_A_and_B_range_of_a_l105_105112


namespace sum_of_decimals_is_fraction_l105_105873

theorem sum_of_decimals_is_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 1466 / 6250 := 
by
  sorry

end sum_of_decimals_is_fraction_l105_105873


namespace problem_b_amount_l105_105977

theorem problem_b_amount (a b : ℝ) (h1 : a + b = 1210) (h2 : (4/5) * a = (2/3) * b) : b = 453.75 :=
sorry

end problem_b_amount_l105_105977


namespace min_value_2xy_minus_2x_minus_y_l105_105718

theorem min_value_2xy_minus_2x_minus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 2/y = 1) :
  2 * x * y - 2 * x - y ≥ 8 :=
sorry

end min_value_2xy_minus_2x_minus_y_l105_105718


namespace find_m_l105_105146

theorem find_m (m : ℝ) (a a1 a2 a3 a4 a5 a6 : ℝ) 
  (h1 : (1 + m)^6 = a + a1 + a2 + a3 + a4 + a5 + a6) 
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 = 63)
  (h3 : a = 1) : m = 1 ∨ m = -3 := 
by
  sorry

end find_m_l105_105146


namespace max_discount_rate_l105_105660

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l105_105660


namespace range_of_a_l105_105273

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then -x^2 + 4 * x else Real.logb 2 x - a

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) → 1 < a :=
sorry

end range_of_a_l105_105273


namespace pasha_mistake_l105_105844

theorem pasha_mistake :
  ¬ (∃ (K R O S C T P : ℕ), K < 10 ∧ R < 10 ∧ O < 10 ∧ S < 10 ∧ C < 10 ∧ T < 10 ∧ P < 10 ∧
    K ≠ R ∧ K ≠ O ∧ K ≠ S ∧ K ≠ C ∧ K ≠ T ∧ K ≠ P ∧
    R ≠ O ∧ R ≠ S ∧ R ≠ C ∧ R ≠ T ∧ R ≠ P ∧
    O ≠ S ∧ O ≠ C ∧ O ≠ T ∧ O ≠ P ∧
    S ≠ C ∧ S ≠ T ∧ S ≠ P ∧
    C ≠ T ∧ C ≠ P ∧ T ≠ P ∧
    10000 * K + 1000 * R + 100 * O + 10 * S + S + 2011 = 10000 * C + 1000 * T + 100 * A + 10 * P + T) :=
sorry

end pasha_mistake_l105_105844


namespace solve_quadratic_eq_1_solve_quadratic_eq_2_l105_105769

-- Proof for Equation 1
theorem solve_quadratic_eq_1 : ∀ x : ℝ, x^2 - 4 * x + 1 = 0 ↔ (x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) :=
by sorry

-- Proof for Equation 2
theorem solve_quadratic_eq_2 : ∀ x : ℝ, 5 * x - 2 = (2 - 5 * x) * (3 * x + 4) ↔ (x = 2 / 5 ∨ x = -5 / 3) :=
by sorry

end solve_quadratic_eq_1_solve_quadratic_eq_2_l105_105769


namespace rationalize_denominator_eqn_l105_105318

theorem rationalize_denominator_eqn : 
  let expr := (3 + Real.sqrt 2) / (2 - Real.sqrt 5)
  let rationalized := -6 - 3 * Real.sqrt 5 - 2 * Real.sqrt 2 - Real.sqrt 10
  let A := -6
  let B := -2
  let C := 2
  expr = rationalized ∧ A * B * C = -24 :=
by
  sorry

end rationalize_denominator_eqn_l105_105318


namespace bounded_regions_l105_105920

noncomputable def regions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => regions n + n + 1

theorem bounded_regions (n : ℕ) :
  (regions n = n * (n + 1) / 2 + 1) := by
  sorry

end bounded_regions_l105_105920


namespace inequality_holds_l105_105865

theorem inequality_holds (x : ℝ) : (∀ y : ℝ, y > 0 → (4 * (x^2 * y^2 + 4 * x * y^2 + 4 * x^2 * y + 16 * y^2 + 12 * x^2 * y) / (x + y) > 3 * x^2 * y)) ↔ x > 0 := 
sorry

end inequality_holds_l105_105865


namespace solve_table_assignment_l105_105206

noncomputable def table_assignment (T_1 T_2 T_3 T_4 : Set (Fin 4 × Fin 4)) : Prop :=
  let Albert := T_4
  let Bogdan := T_2
  let Vadim := T_1
  let Denis := T_3
  (∀ x, x ∈ Vadim ↔ x ∉ (Albert ∪ Bogdan)) ∧
  (∀ x, x ∈ Denis ↔ x ∉ (Bogdan ∪ Vadim)) ∧
  Albert = T_4 ∧
  Bogdan = T_2 ∧
  Vadim = T_1 ∧
  Denis = T_3

theorem solve_table_assignment (T_1 T_2 T_3 T_4 : Set (Fin 4 × Fin 4)) :
  table_assignment T_1 T_2 T_3 T_4 :=
sorry

end solve_table_assignment_l105_105206


namespace average_of_four_numbers_l105_105529

theorem average_of_four_numbers (a b c d : ℝ) 
  (h1 : b + c + d = 24) (h2 : a + c + d = 36)
  (h3 : a + b + d = 28) (h4 : a + b + c = 32) :
  (a + b + c + d) / 4 = 10 := 
sorry

end average_of_four_numbers_l105_105529


namespace proportional_function_l105_105721

theorem proportional_function (k m : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = k * x) →
  f 2 = -4 →
  (∀ x, f x + m = -2 * x + m) →
  f 2 = -4 ∧ (f 1 + m = 1) →
  k = -2 ∧ m = 3 := 
by
  intros h1 h2 h3 h4
  sorry

end proportional_function_l105_105721


namespace length_QR_l105_105331

-- Let's define the given conditions and the theorem to prove

-- Define the lengths of the sides of the triangle
def PQ : ℝ := 4
def PR : ℝ := 7
def PM : ℝ := 3.5

-- Define the median formula
def median_formula (PQ PR QR PM : ℝ) := PM = 0.5 * Real.sqrt (2 * PQ^2 + 2 * PR^2 - QR^2)

-- The theorem to prove: QR = 9
theorem length_QR 
  (hPQ : PQ = 4) 
  (hPR : PR = 7) 
  (hPM : PM = 3.5) 
  (hMedian : median_formula PQ PR QR PM) : 
  QR = 9 :=
sorry  -- proof will be here

end length_QR_l105_105331


namespace problem1_solution_problem2_solution_l105_105183

-- Problem 1
theorem problem1_solution (x y : ℝ) : (2 * x - y = 3) ∧ (x + y = 3) ↔ (x = 2 ∧ y = 1) := by
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) : (x / 4 + y / 3 = 3) ∧ (3 * x - 2 * (y - 1) = 11) ↔ (x = 6 ∧ y = 9 / 2) := by
  sorry

end problem1_solution_problem2_solution_l105_105183


namespace students_on_seventh_day_day_of_week_day_when_3280_students_know_secret_l105_105931

noncomputable def numStudentsKnowingSecret (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem students_on_seventh_day :
  (numStudentsKnowingSecret 7) = 3280 :=
by
  sorry

theorem day_of_week (n : ℕ) : String :=
  if n % 7 = 0 then "Monday" else
  if n % 7 = 1 then "Tuesday" else
  if n % 7 = 2 then "Wednesday" else
  if n % 7 = 3 then "Thursday" else
  if n % 7 = 4 then "Friday" else
  if n % 7 = 5 then "Saturday" else
  "Sunday"

theorem day_when_3280_students_know_secret :
  day_of_week 7 = "Sunday" :=
by
  sorry

end students_on_seventh_day_day_of_week_day_when_3280_students_know_secret_l105_105931


namespace circle_diameter_from_area_l105_105473

theorem circle_diameter_from_area (A : ℝ) (h : A = 225 * Real.pi) : ∃ d : ℝ, d = 30 :=
  by
  have r := Real.sqrt (225)
  have d := 2 * r
  exact ⟨d, sorry⟩

end circle_diameter_from_area_l105_105473


namespace intersection_unique_point_l105_105188

theorem intersection_unique_point
    (h1 : ∀ (x y : ℝ), 2 * x + 3 * y = 6)
    (h2 : ∀ (x y : ℝ), 4 * x - 3 * y = 6)
    (h3 : ∀ y : ℝ, 2 = 2)
    (h4 : ∀ x : ℝ, y = 2 / 3)
    : ∃! (x y : ℝ), (2 * x + 3 * y = 6) ∧ (4 * x - 3 * y = 6) ∧ (x = 2) ∧ (y = 2 / 3) := 
by
    sorry

end intersection_unique_point_l105_105188


namespace max_students_seated_l105_105504

-- Define the number of seats in the i-th row
def seats_in_row (i : ℕ) : ℕ := 10 + 2 * i

-- Define the maximum number of students that can be seated in the i-th row
def max_students_in_row (i : ℕ) : ℕ := (seats_in_row i + 1) / 2

-- Sum the maximum number of students for all 25 rows
def total_max_students : ℕ := (Finset.range 25).sum max_students_in_row

-- The theorem statement
theorem max_students_seated : total_max_students = 450 := by
  sorry

end max_students_seated_l105_105504


namespace smaller_number_of_ratio_4_5_lcm_180_l105_105354

theorem smaller_number_of_ratio_4_5_lcm_180 {a b : ℕ} (h_ratio : 4 * b = 5 * a) (h_lcm : Nat.lcm a b = 180) : a = 144 :=
by
  sorry

end smaller_number_of_ratio_4_5_lcm_180_l105_105354


namespace infinite_seq_partition_infinite_seq_within_interval_l105_105697

open Mathlib

variable (x : ℕ → ℝ) (ε : ℝ)

-- Condition: Infinite sequence of real numbers in [0, 1)
constant seq_bounded : ∀ n, 0 ≤ x n ∧ x n < 1

-- Condition: ε is strictly between 0 and 1/2
constant eps_cond : ε > 0 ∧ ε < 1/2

-- Problem 1: Prove that either [0, 1/2) or [1/2, 1) contains infinitely many elements
theorem infinite_seq_partition :
  (∃∞ n, 0 ≤ x n ∧ x n < 1/2) ∨ (∃∞ n, 1/2 ≤ x n ∧ x n < 1) := sorry

-- Problem 2: Prove that there exists a rational number α ∈ [0, 1] such that infinitely many elements are within [α - ε, α + ε]
theorem infinite_seq_within_interval :
  ∃ (α : ℚ), 0 ≤ α ∧ α ≤ 1 ∧ ∃∞ n, (α.toReal - ε ≤ x n ∧ x n ≤ α.toReal + ε) := sorry

end infinite_seq_partition_infinite_seq_within_interval_l105_105697


namespace eval_expression_l105_105235

theorem eval_expression : (2 ^ (-1 : ℤ)) + (Real.sin (Real.pi / 6)) - (Real.pi - 3.14) ^ (0 : ℤ) + abs (-3) - Real.sqrt 9 = 0 := by
  sorry

end eval_expression_l105_105235


namespace sum_of_decimals_as_fraction_l105_105877

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 733 / 3125 := by
  sorry

end sum_of_decimals_as_fraction_l105_105877


namespace triangle_area_l105_105100

theorem triangle_area (a b c : ℕ) (h : a * a + b * b = c * c) : 1 / 2 * a * b = 84 := by
  have ha : a = 7 := rfl
  have hb : b = 24 := rfl
  have hc : c = 25 := rfl
  rw [ha, hb, hc] at h
  norm_num at h
  norm_num
  sorry

end triangle_area_l105_105100


namespace smallest_value_m_plus_n_l105_105389

theorem smallest_value_m_plus_n (m n : ℕ) (h : 3 * n^3 = 5 * m^2) : m + n = 60 :=
sorry

end smallest_value_m_plus_n_l105_105389


namespace P_zero_eq_zero_l105_105094

open Polynomial

noncomputable def P (x : ℝ) : ℝ := sorry

axiom distinct_roots : ∃ y : Fin 17 → ℝ, Function.Injective y ∧ ∀ i, P (y i ^ 2) = 0

theorem P_zero_eq_zero : P 0 = 0 :=
by
  sorry

end P_zero_eq_zero_l105_105094


namespace find_n_l105_105020

theorem find_n (n : ℕ) (composite_n : n > 1 ∧ ¬Prime n) : 
  ((∃ m : ℕ, ∀ d : ℕ, d ∣ n ∧ 1 < d ∧ d < n → d + 1 ∣ m ∧ 1 < d + 1 ∧ d + 1 < m) ↔ 
    (n = 4 ∨ n = 8)) :=
by sorry

end find_n_l105_105020


namespace odd_and_increasing_l105_105479

-- Define the function f(x) = e^x - e^{-x}
noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

-- We want to prove that this function is both odd and increasing.
theorem odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
sorry

end odd_and_increasing_l105_105479


namespace runway_show_duration_l105_105496

theorem runway_show_duration
  (evening_wear_time : ℝ) (bathing_suits_time : ℝ) (formal_wear_time : ℝ) (casual_wear_time : ℝ)
  (evening_wear_sets : ℕ) (bathing_suits_sets : ℕ) (formal_wear_sets : ℕ) (casual_wear_sets : ℕ)
  (num_models : ℕ) :
  evening_wear_time = 4 → bathing_suits_time = 2 → formal_wear_time = 3 → casual_wear_time = 2.5 →
  evening_wear_sets = 4 → bathing_suits_sets = 2 → formal_wear_sets = 3 → casual_wear_sets = 5 →
  num_models = 10 →
  (evening_wear_time * evening_wear_sets + bathing_suits_time * bathing_suits_sets
   + formal_wear_time * formal_wear_sets + casual_wear_time * casual_wear_sets) * num_models = 415 :=
by
  intros
  sorry

end runway_show_duration_l105_105496


namespace maximum_perimeter_triangle_area_l105_105750

-- Part 1: Maximum Perimeter
theorem maximum_perimeter (a b c : ℝ) (A B C : ℝ) 
  (h_c : c = 2) 
  (h_C : C = Real.pi / 3) :
  (a + b + c) ≤ 6 :=
sorry

-- Part 2: Area under given trigonometric condition
theorem triangle_area (A B C a b c : ℝ) 
  (h_c : 2 * Real.sin (2 * A) + Real.sin (2 * B + C) = Real.sin C) :
  (1/2 * a * b * Real.sin C) = (2 * Real.sqrt 6) / 3 :=
sorry

end maximum_perimeter_triangle_area_l105_105750


namespace odd_solutions_eq_iff_a_le_neg3_or_a_ge3_l105_105886

theorem odd_solutions_eq_iff_a_le_neg3_or_a_ge3 (a : ℝ) :
  (∃! x : ℝ, -1 ≤ x ∧ x ≤ 5 ∧ (a - 3 * x^2 + Real.cos (9 * Real.pi * x / 2)) * Real.sqrt (3 - a * x) = 0) ↔ (a ≤ -3 ∨ a ≥ 3) := 
by
  sorry

end odd_solutions_eq_iff_a_le_neg3_or_a_ge3_l105_105886


namespace binom_floor_divisible_l105_105569

theorem binom_floor_divisible {p n : ℕ}
  (hp : Prime p) :
  (Nat.choose n p - n / p) % p = 0 := 
by
  sorry

end binom_floor_divisible_l105_105569


namespace cannot_determine_a_l105_105411

theorem cannot_determine_a 
  (n : ℝ) 
  (p : ℝ) 
  (a : ℝ) 
  (line_eq : ∀ (x y : ℝ), x = 5 * y + 5) 
  (pt1 : a = 5 * n + 5) 
  (pt2 : a + 2 = 5 * (n + p) + 5) : p = 0.4 → ¬∀ a' : ℝ, a = a' :=
by
  sorry

end cannot_determine_a_l105_105411


namespace total_bus_capacity_l105_105156

def left_seats : ℕ := 15
def right_seats : ℕ := left_seats - 3
def people_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 8

theorem total_bus_capacity :
  (left_seats + right_seats) * people_per_seat + back_seat_capacity = 89 := by
  sorry

end total_bus_capacity_l105_105156


namespace range_of_a_l105_105154

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a ≥ 0) → (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l105_105154


namespace area_of_park_l105_105050

-- Definitions of conditions
def ratio_length_breadth (L B : ℝ) : Prop := L / B = 1 / 3
def cycling_time_distance (speed time perimeter : ℝ) : Prop := perimeter = speed * time

theorem area_of_park :
  ∃ (L B : ℝ),
    ratio_length_breadth L B ∧
    cycling_time_distance 12 (8 / 60) (2 * (L + B)) ∧
    L * B = 120000 := by
  sorry

end area_of_park_l105_105050


namespace degree_of_p_x2_q_x4_l105_105772

-- Definitions to capture the given problem conditions
def is_degree_3 (p : Polynomial ℝ) : Prop := p.degree = 3
def is_degree_6 (q : Polynomial ℝ) : Prop := q.degree = 6

-- Statement of the proof problem
theorem degree_of_p_x2_q_x4 (p q : Polynomial ℝ) (hp : is_degree_3 p) (hq : is_degree_6 q) :
  (p.comp (Polynomial.X ^ 2) * q.comp (Polynomial.X ^ 4)).degree = 30 :=
sorry

end degree_of_p_x2_q_x4_l105_105772


namespace megan_math_problems_l105_105126

theorem megan_math_problems (num_spelling_problems num_problems_per_hour num_hours total_problems num_math_problems : ℕ) 
  (h1 : num_spelling_problems = 28)
  (h2 : num_problems_per_hour = 8)
  (h3 : num_hours = 8)
  (h4 : total_problems = num_problems_per_hour * num_hours)
  (h5 : total_problems = num_spelling_problems + num_math_problems) :
  num_math_problems = 36 := 
by
  sorry

end megan_math_problems_l105_105126


namespace find_fourth_number_l105_105351

variables (A B C D E F : ℝ)

theorem find_fourth_number
  (h1 : A + B + C + D + E + F = 180)
  (h2 : A + B + C + D = 100)
  (h3 : D + E + F = 105) :
  D = 25 :=
by
  sorry

end find_fourth_number_l105_105351


namespace one_plus_i_squared_eq_two_i_l105_105713

theorem one_plus_i_squared_eq_two_i (i : ℂ) (h : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end one_plus_i_squared_eq_two_i_l105_105713


namespace problem_solution_l105_105070

theorem problem_solution :
  (204^2 - 196^2) / 16 = 200 :=
by
  sorry

end problem_solution_l105_105070


namespace chair_cost_l105_105029

-- Define the conditions
def total_spent : ℕ := 56
def table_cost : ℕ := 34
def num_chairs : ℕ := 2

-- Define the statement we need to prove
theorem chair_cost :
  ∃ (chair_cost : ℕ), 2 * chair_cost + table_cost = total_spent ∧ chair_cost = 11 :=
by
  use 11
  split
  sorry -- proof goes here, skipped as per instructions

end chair_cost_l105_105029


namespace complement_B_in_A_l105_105423

noncomputable def A : Set ℝ := {x | x < 2}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x < 2}

theorem complement_B_in_A : {x | x ∈ A ∧ x ∉ B} = {x | x ≤ 1} :=
by
  sorry

end complement_B_in_A_l105_105423


namespace corrected_mean_l105_105326

theorem corrected_mean (n : ℕ) (incorrect_mean : ℝ) (incorrect_observation correct_observation : ℝ)
  (h_n : n = 50)
  (h_incorrect_mean : incorrect_mean = 30)
  (h_incorrect_observation : incorrect_observation = 23)
  (h_correct_observation : correct_observation = 48) :
  (incorrect_mean * n - incorrect_observation + correct_observation) / n = 30.5 :=
by
  sorry

end corrected_mean_l105_105326


namespace evaluate_expression_l105_105246

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) : (a^b)^b + (b^a)^a = 593 := by
  sorry

end evaluate_expression_l105_105246


namespace max_discount_rate_l105_105655

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l105_105655


namespace find_equation_l105_105502

theorem find_equation (x : ℝ) : 
  (3 + x < 1 → false) ∧
  ((x - 67 + 63 = x - 4) → false) ∧
  ((4.8 + x = x + 4.8) → false) ∧
  (x + 0.7 = 12 → true) := 
sorry

end find_equation_l105_105502


namespace amy_total_equals_bob_total_l105_105407

def original_price : ℝ := 120.00
def sales_tax_rate : ℝ := 0.08
def discount_rate : ℝ := 0.25
def additional_discount : ℝ := 0.10
def num_sweaters : ℕ := 4

def calculate_amy_total (original_price : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (additional_discount : ℝ) (num_sweaters : ℕ) : ℝ :=
  let price_with_tax := original_price * (1.0 + sales_tax_rate)
  let discounted_price := price_with_tax * (1.0 - discount_rate)
  let final_price := discounted_price * (1.0 - additional_discount)
  final_price * (num_sweaters : ℝ)
  
def calculate_bob_total (original_price : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (additional_discount : ℝ) (num_sweaters : ℕ) : ℝ :=
  let discounted_price := original_price * (1.0 - discount_rate)
  let further_discounted_price := discounted_price * (1.0 - additional_discount)
  let price_with_tax := further_discounted_price * (1.0 + sales_tax_rate)
  price_with_tax * (num_sweaters : ℝ)

theorem amy_total_equals_bob_total :
  calculate_amy_total original_price sales_tax_rate discount_rate additional_discount num_sweaters =
  calculate_bob_total original_price sales_tax_rate discount_rate additional_discount num_sweaters :=
by
  sorry

end amy_total_equals_bob_total_l105_105407


namespace max_discount_rate_l105_105620

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l105_105620


namespace value_of_expression_l105_105262

theorem value_of_expression (a b c : ℚ) (h1 : a * b * c < 0) (h2 : a + b + c = 0) :
    (a - b - c) / |a| + (b - c - a) / |b| + (c - a - b) / |c| = 2 :=
by
  sorry

end value_of_expression_l105_105262


namespace ratio_of_profits_is_2_to_3_l105_105576

-- Conditions
def Praveen_initial_investment := 3220
def Praveen_investment_duration := 12
def Hari_initial_investment := 8280
def Hari_investment_duration := 7

-- Effective capital contributions
def Praveen_effective_capital : ℕ := Praveen_initial_investment * Praveen_investment_duration
def Hari_effective_capital : ℕ := Hari_initial_investment * Hari_investment_duration

-- Theorem statement to be proven
theorem ratio_of_profits_is_2_to_3 : (Praveen_effective_capital : ℚ) / Hari_effective_capital = 2 / 3 :=
by sorry

end ratio_of_profits_is_2_to_3_l105_105576


namespace total_number_of_animals_l105_105603

theorem total_number_of_animals 
  (rabbits ducks chickens : ℕ)
  (h1 : chickens = 5 * ducks)
  (h2 : ducks = rabbits + 12)
  (h3 : rabbits = 4) : 
  chickens + ducks + rabbits = 100 :=
by
  sorry

end total_number_of_animals_l105_105603


namespace prove_proposition_false_l105_105894

def proposition (a : ℝ) := ∃ x : ℝ, x^2 - 4*a*x + 3 < 0

theorem prove_proposition_false : proposition 0 = False :=
by
sorry

end prove_proposition_false_l105_105894


namespace billy_sisters_count_l105_105375

theorem billy_sisters_count 
  (S B : ℕ) -- S is the number of sisters, B is the number of brothers
  (h1 : B = 2 * S) -- Billy has twice as many brothers as sisters
  (h2 : 2 * (B + S) = 12) -- Billy gives 2 sodas to each sibling to give out the 12 pack
  : S = 2 := 
  by sorry

end billy_sisters_count_l105_105375


namespace tangent_addition_tangent_subtraction_l105_105472

theorem tangent_addition (a b : ℝ) : 
  Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b) :=
sorry

theorem tangent_subtraction (a b : ℝ) : 
  Real.tan (a - b) = (Real.tan a - Real.tan b) / (1 + Real.tan a * Real.tan b) :=
sorry

end tangent_addition_tangent_subtraction_l105_105472


namespace totalPeaches_l105_105489

-- Define the number of red, yellow, and green peaches
def redPeaches := 7
def yellowPeaches := 15
def greenPeaches := 8

-- Define the total number of peaches and the proof statement
theorem totalPeaches : redPeaches + yellowPeaches + greenPeaches = 30 := by
  sorry

end totalPeaches_l105_105489


namespace sum_of_edges_equals_74_l105_105209

def V (pyramid : ℕ) : ℕ := pyramid

def E (pyramid : ℕ) : ℕ := 2 * (V pyramid - 1)

def sum_of_edges (pyramid1 pyramid2 pyramid3 : ℕ) : ℕ :=
  E pyramid1 + E pyramid2 + E pyramid3

theorem sum_of_edges_equals_74 (V₁ V₂ V₃ : ℕ) (h : V₁ + V₂ + V₃ = 40) :
  sum_of_edges V₁ V₂ V₃ = 74 :=
sorry

end sum_of_edges_equals_74_l105_105209


namespace triangle_height_l105_105589

theorem triangle_height (base area height : ℝ)
    (h_base : base = 4)
    (h_area : area = 16)
    (h_area_formula : area = (base * height) / 2) :
    height = 8 :=
by
  sorry

end triangle_height_l105_105589


namespace ordered_pair_unique_l105_105113

theorem ordered_pair_unique (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x^y + 1 = y^x) (h2 : 2 * x^y = y^x + 13) : (x, y) = (1, 14) :=
by
  sorry

end ordered_pair_unique_l105_105113


namespace time_A_to_complete_race_l105_105549

noncomputable def km_race_time (V_B : ℕ) : ℚ :=
  940 / V_B

theorem time_A_to_complete_race : km_race_time 6 = 156.67 := by
  sorry

end time_A_to_complete_race_l105_105549


namespace simplify_expression_l105_105585

theorem simplify_expression (x : ℝ) :
  4*x^3 + 5*x + 6*x^2 + 10 - (3 - 6*x^2 - 4*x^3 + 2*x) = 8*x^3 + 12*x^2 + 3*x + 7 :=
by
  sorry

end simplify_expression_l105_105585


namespace rectangle_area_y_l105_105451

theorem rectangle_area_y (y : ℝ) (h_y_pos : y > 0)
  (h_area : (3 * y = 21)) : y = 7 :=
by
  sorry

end rectangle_area_y_l105_105451


namespace domain_of_log_x_squared_sub_2x_l105_105192

theorem domain_of_log_x_squared_sub_2x (x : ℝ) : x^2 - 2 * x > 0 ↔ x < 0 ∨ x > 2 :=
by
  sorry

end domain_of_log_x_squared_sub_2x_l105_105192


namespace carson_clouds_l105_105695

theorem carson_clouds (C D : ℕ) (h1 : D = 3 * C) (h2 : C + D = 24) : C = 6 :=
by
  sorry

end carson_clouds_l105_105695


namespace sam_initial_money_l105_105180

theorem sam_initial_money (num_books cost_per_book money_left initial_money : ℤ) 
  (h1 : num_books = 9) 
  (h2 : cost_per_book = 7) 
  (h3 : money_left = 16)
  (h4 : initial_money = num_books * cost_per_book + money_left) :
  initial_money = 79 := 
by
  -- Proof is not required, hence we use sorry to complete the statement.
  sorry

end sam_initial_money_l105_105180


namespace painters_workdays_l105_105558

theorem painters_workdays (d₁ d₂ : ℚ) (p₁ p₂ : ℕ)
  (h1 : p₁ = 5) (h2 : p₂ = 4) (rate: 5 * d₁ = 7.5) :
  (p₂:ℚ) * d₂ = 7.5 → d₂ = 1 + 7 / 8 :=
by
  sorry

end painters_workdays_l105_105558


namespace marbles_per_pack_l105_105754

theorem marbles_per_pack (total_marbles : ℕ) (leo_packs manny_packs neil_packs total_packs : ℕ) 
(h1 : total_marbles = 400) 
(h2 : leo_packs = 25) 
(h3 : manny_packs = total_packs / 4) 
(h4 : neil_packs = total_packs / 8) 
(h5 : leo_packs + manny_packs + neil_packs = total_packs) : 
total_marbles / total_packs = 10 := 
by sorry

end marbles_per_pack_l105_105754


namespace total_population_expression_l105_105295

variables (b g t: ℕ)

-- Assuming the given conditions
def condition1 := b = 4 * g
def condition2 := g = 8 * t

-- The theorem to prove
theorem total_population_expression (h1 : condition1 b g) (h2 : condition2 g t) :
    b + g + t = 41 * b / 32 := sorry

end total_population_expression_l105_105295


namespace lucas_mod_prime_zero_l105_105462

-- Define the Lucas sequence
def lucas : ℕ → ℕ
| 0 => 1       -- Note that in the mathematical problem L_1 is given as 1. Therefore we adjust for 0-based index in programming.
| 1 => 3
| (n + 2) => lucas n + lucas (n + 1)

-- Main theorem statement
theorem lucas_mod_prime_zero (p : ℕ) (hp : Nat.Prime p) : (lucas p - 1) % p = 0 := by
  sorry

end lucas_mod_prime_zero_l105_105462


namespace no_such_functions_exist_l105_105115

theorem no_such_functions_exist (f g : ℝ → ℝ) :
  ¬ (∀ x y : ℝ, x ≠ y → |f x - f y| + |g x - g y| > 1) :=
sorry

end no_such_functions_exist_l105_105115


namespace probability_neither_nearsighted_l105_105058

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

end probability_neither_nearsighted_l105_105058


namespace fixed_point_of_transformed_exponential_l105_105955

variable (a : ℝ)
variable (h_pos : 0 < a)
variable (h_ne_one : a ≠ 1)

theorem fixed_point_of_transformed_exponential :
    (∃ x y : ℝ, (y = a^(x-2) + 2) ∧ (y = x) ∧ (x = 2) ∧ (y = 3)) :=
by {
    sorry -- Proof goes here
}

end fixed_point_of_transformed_exponential_l105_105955


namespace geometric_sequence_a_equals_minus_four_l105_105908

theorem geometric_sequence_a_equals_minus_four (a : ℝ) 
(h : (2 * a + 2) ^ 2 = a * (3 * a + 3)) : a = -4 :=
sorry

end geometric_sequence_a_equals_minus_four_l105_105908


namespace units_digit_squares_eq_l105_105942

theorem units_digit_squares_eq (x y : ℕ) (hx : x % 10 + y % 10 = 10) :
  (x * x) % 10 = (y * y) % 10 :=
by
  sorry

end units_digit_squares_eq_l105_105942


namespace widget_production_difference_l105_105930

variable (w t : ℕ)
variable (h_wt : w = 2 * t)

theorem widget_production_difference (w t : ℕ)
    (h_wt : w = 2 * t) :
  (w * t) - ((w + 5) * (t - 3)) = t + 15 :=
by 
  sorry

end widget_production_difference_l105_105930


namespace sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3_l105_105191

variable (x : ℝ)

theorem sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3 :
        (0 < x ∧ x < 5) → (|x - 2| < 3) :=
by sorry

theorem not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3 :
        (|x - 2| < 3) → (0 < x ∧ x < 5) :=
by sorry

theorem sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3 :
        (0 < x ∧ x < 5) ↔ (|x - 2| < 3) → false :=
by sorry

end sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3_l105_105191


namespace quadratic_condition_l105_105190

noncomputable def quadratic_sufficiency (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + x + m = 0 → m < 1/4

noncomputable def quadratic_necessity (m : ℝ) : Prop :=
  (∃ (x : ℝ), x^2 + x + m = 0) → m ≤ 1/4

theorem quadratic_condition (m : ℝ) : 
  (m < 1/4 → quadratic_sufficiency m) ∧ ¬ quadratic_necessity m := 
sorry

end quadratic_condition_l105_105190


namespace lcm_150_294_l105_105608

theorem lcm_150_294 : Nat.lcm 150 294 = 7350 := by
  sorry

end lcm_150_294_l105_105608


namespace maximum_discount_rate_l105_105625

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l105_105625


namespace vector_scalar_operations_l105_105863

-- Define the vectors
def v1 : ℤ × ℤ := (2, -9)
def v2 : ℤ × ℤ := (-1, -6)

-- Define the scalars
def c1 : ℤ := 4
def c2 : ℤ := 3

-- Define the scalar multiplication of vectors
def scale (c : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (c * v.1, c * v.2)

-- Define the vector subtraction
def sub (v w : ℤ × ℤ) : ℤ × ℤ := (v.1 - w.1, v.2 - w.2)

-- State the theorem
theorem vector_scalar_operations :
  sub (scale c1 v1) (scale c2 v2) = (11, -18) :=
by
  sorry

end vector_scalar_operations_l105_105863


namespace largest_volume_sold_in_august_is_21_l105_105779

def volumes : List ℕ := [13, 15, 16, 17, 19, 21]

theorem largest_volume_sold_in_august_is_21
  (sold_volumes_august : List ℕ)
  (sold_volumes_september : List ℕ) :
  sold_volumes_august.length = 3 ∧
  sold_volumes_september.length = 2 ∧
  2 * (sold_volumes_september.sum) = sold_volumes_august.sum ∧
  (sold_volumes_august ++ sold_volumes_september).sum = volumes.sum →
  21 ∈ sold_volumes_august :=
sorry

end largest_volume_sold_in_august_is_21_l105_105779


namespace three_friends_expenses_l105_105467

theorem three_friends_expenses :
  let ticket_cost := 7
  let number_of_tickets := 3
  let popcorn_cost := 1.5
  let number_of_popcorn := 2
  let milk_tea_cost := 3
  let number_of_milk_tea := 3
  let total_expenses := (ticket_cost * number_of_tickets) + (popcorn_cost * number_of_popcorn) + (milk_tea_cost * number_of_milk_tea)
  let amount_per_friend := total_expenses / 3
  amount_per_friend = 11 := 
by
  sorry

end three_friends_expenses_l105_105467


namespace minimum_value_of_tan_sum_l105_105552

open Real

theorem minimum_value_of_tan_sum :
  ∀ {A B C : ℝ}, 
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π ∧ 
  2 * sin A ^ 2 + sin B ^ 2 = 2 * sin C ^ 2 ->
  ( ∃ t : ℝ, ( t = 1 / tan A + 1 / tan B + 1 / tan C ) ∧ t = sqrt 13 / 2 ) := 
sorry

end minimum_value_of_tan_sum_l105_105552


namespace point_in_third_quadrant_l105_105269

open Complex

-- Define that i is the imaginary unit
def imaginary_unit : ℂ := Complex.I

-- Define the condition i * z = 1 - 2i
def condition (z : ℂ) : Prop := imaginary_unit * z = (1 : ℂ) - 2 * imaginary_unit

-- Prove that the point corresponding to the complex number z is located in the third quadrant
theorem point_in_third_quadrant (z : ℂ) (h : condition z) : z.re < 0 ∧ z.im < 0 := sorry

end point_in_third_quadrant_l105_105269


namespace sum_of_squares_eq_expansion_l105_105939

theorem sum_of_squares_eq_expansion (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 :=
sorry

end sum_of_squares_eq_expansion_l105_105939


namespace oranges_in_second_group_l105_105447

namespace oranges_problem

-- Definitions coming from conditions
def cost_of_apple : ℝ := 0.21
def total_cost_1 : ℝ := 1.77
def total_cost_2 : ℝ := 1.27
def num_apples_group1 : ℕ := 6
def num_oranges_group1 : ℕ := 3
def num_apples_group2 : ℕ := 2
def cost_of_orange : ℝ := 0.17
def num_oranges_group2 : ℕ := 5 -- derived from the solution involving $0.85/$0.17.

-- Price calculation functions and conditions
def price_group1 (cost_of_orange : ℝ) : ℝ :=
  num_apples_group1 * cost_of_apple + num_oranges_group1 * cost_of_orange

def price_group2 (num_oranges_group2 cost_of_orange : ℝ) : ℝ :=
  num_apples_group2 * cost_of_apple + num_oranges_group2 * cost_of_orange

theorem oranges_in_second_group :
  (price_group1 cost_of_orange = total_cost_1) →
  (price_group2 num_oranges_group2 cost_of_orange = total_cost_2) →
  num_oranges_group2 = 5 :=
by
  intros h1 h2
  sorry

end oranges_problem

end oranges_in_second_group_l105_105447


namespace gage_skating_time_l105_105256

theorem gage_skating_time :
  let minutes_skated_first_5_days := 75 * 5
  let minutes_skated_next_3_days := 90 * 3
  let total_minutes_skated_8_days := minutes_skated_first_5_days + minutes_skated_next_3_days
  let total_minutes_required := 85 * 9
  let minutes_needed_ninth_day := total_minutes_required - total_minutes_skated_8_days
  minutes_needed_ninth_day = 120 :=
by
  let minutes_skated_first_5_days := 75 * 5
  let minutes_skated_next_3_days := 90 * 3
  let total_minutes_skated_8_days := minutes_skated_first_5_days + minutes_skated_next_3_days
  let total_minutes_required := 85 * 9
  let minutes_needed_ninth_day := total_minutes_required - total_minutes_skated_8_days
  sorry

end gage_skating_time_l105_105256


namespace transform_quadratic_to_squared_form_l105_105329

theorem transform_quadratic_to_squared_form :
  ∀ x : ℝ, 2 * x^2 - 3 * x + 1 = 0 → (x - 3 / 4)^2 = 1 / 16 :=
by
  intro x h
  sorry

end transform_quadratic_to_squared_form_l105_105329


namespace regression_line_equation_chi_square_relation_l105_105501

-- Assuming the necessary dataset and functions are defined

-- Data1 for regression line calculation
def x : List ℝ := [1, 2, 3, 4, 5]
def y : List ℝ := [120, 100, 90, 75, 65]
def sum_xy : ℝ := 1215
def n : ℝ := 5
def mean_x : ℝ := (x.sum) / n
def mean_y : ℝ := (y.sum) / n

-- Data2 for chi-square test
def a : ℝ := 15
def b : ℝ := 10
def c : ℝ := 25
def d : ℝ := 50
def total_accidents : ℝ := 100
def chi_square_critical_value_95 : ℝ := 3.841

-- Part 1: Statement for Regression Line Equation
theorem regression_line_equation :
  let β := (sum_xy - n * mean_x * mean_y) / (x.foldr (λ xi acc, xi^2 + acc) 0 - n * mean_x^2),
      α := mean_y - β * mean_x in
  β = -13.5 ∧ α = 130.5 :=
by  sorry

-- Part 2: Statement for Chi-Square Test
theorem chi_square_relation :
  let chi_square := (total_accidents * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) in
  chi_square > chi_square_critical_value_95 :=
by sorry

end regression_line_equation_chi_square_relation_l105_105501


namespace mabel_age_l105_105025

theorem mabel_age (n : ℕ) (h : n * (n + 1) / 2 = 28) : n = 7 :=
sorry

end mabel_age_l105_105025


namespace power_function_increasing_m_eq_2_l105_105006

theorem power_function_increasing_m_eq_2 (m : ℝ) :
  (∀ x > 0, (m^2 - m - 1) * x^m > 0) → m = 2 :=
by
  sorry

end power_function_increasing_m_eq_2_l105_105006


namespace arithmetic_geometric_sequence_l105_105468

theorem arithmetic_geometric_sequence (a d : ℤ) (h1 : ∃ a d, (a - d) * a * (a + d) = 1000)
  (h2 : ∃ a d, a^2 = 2 * (a - d) * ((a + d) + 7)) :
  d = 8 ∨ d = -15 :=
by sorry

end arithmetic_geometric_sequence_l105_105468


namespace lamp_pricing_problem_l105_105085

theorem lamp_pricing_problem
  (purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_sales_volume : ℝ)
  (sales_decrease_rate : ℝ)
  (desired_profit : ℝ) :
  purchase_price = 30 →
  initial_selling_price = 40 →
  initial_sales_volume = 600 →
  sales_decrease_rate = 10 →
  desired_profit = 10000 →
  (∃ (selling_price : ℝ) (sales_volume : ℝ), selling_price = 50 ∧ sales_volume = 500) :=
by
  intros h_purchase h_initial_selling h_initial_sales h_sales_decrease h_desired_profit
  sorry

end lamp_pricing_problem_l105_105085


namespace violet_balloons_count_l105_105752

-- Define the initial number of violet balloons
def initial_violet_balloons := 7

-- Define the number of violet balloons Jason lost
def lost_violet_balloons := 3

-- Define the remaining violet balloons after losing some
def remaining_violet_balloons := initial_violet_balloons - lost_violet_balloons

-- Prove that the remaining violet balloons is equal to 4
theorem violet_balloons_count : remaining_violet_balloons = 4 :=
by
  sorry

end violet_balloons_count_l105_105752


namespace pages_per_sheet_is_one_l105_105563

-- Definition of conditions
def stories_per_week : Nat := 3
def pages_per_story : Nat := 50
def num_weeks : Nat := 12
def reams_bought : Nat := 3
def sheets_per_ream : Nat := 500

-- Calculate total pages written over num_weeks (short stories only)
def total_pages : Nat := stories_per_week * pages_per_story * num_weeks

-- Calculate total sheets available
def total_sheets : Nat := reams_bought * sheets_per_ream

-- Calculate pages per sheet, rounding to nearest whole number
def pages_per_sheet : Nat := (total_pages / total_sheets)

-- The main statement to prove
theorem pages_per_sheet_is_one : pages_per_sheet = 1 :=
by
  sorry

end pages_per_sheet_is_one_l105_105563


namespace motorcycle_wheels_l105_105551

/--
In a parking lot, there are cars and motorcycles. Each car has 5 wheels (including one spare) 
and each motorcycle has a certain number of wheels. There are 19 cars in the parking lot.
Altogether all vehicles have 117 wheels. There are 11 motorcycles at the parking lot.
--/
theorem motorcycle_wheels (num_cars num_motorcycles total_wheels wheels_per_car wheels_per_motorcycle : ℕ)
  (h1 : wheels_per_car = 5) 
  (h2 : num_cars = 19) 
  (h3 : total_wheels = 117) 
  (h4 : num_motorcycles = 11) 
  : wheels_per_motorcycle = 2 :=
by
  sorry

end motorcycle_wheels_l105_105551


namespace common_tangent_x_eq_neg1_l105_105897
open Real

-- Definitions of circles C₁ and C₂
def circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
def circle2 := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = 16}

-- Statement of the problem
theorem common_tangent_x_eq_neg1 :
  ∀ (x : ℝ) (y : ℝ),
    (x, y) ∈ circle1 ∧ (x, y) ∈ circle2 → x = -1 :=
sorry

end common_tangent_x_eq_neg1_l105_105897


namespace total_seven_flights_time_l105_105108

def time_for_nth_flight (n : ℕ) : ℕ :=
  25 + (n - 1) * 8

def total_time_for_flights (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => time_for_nth_flight (k + 1))

theorem total_seven_flights_time :
  total_time_for_flights 7 = 343 :=
  by
    sorry

end total_seven_flights_time_l105_105108


namespace lcm_of_numbers_is_91_l105_105051

def ratio (a b : ℕ) (p q : ℕ) : Prop := p * b = q * a

theorem lcm_of_numbers_is_91 (a b : ℕ) (h_ratio : ratio a b 7 13) (h_gcd : Nat.gcd a b = 15) :
  Nat.lcm a b = 91 := 
by sorry

end lcm_of_numbers_is_91_l105_105051


namespace center_of_circle_l105_105952

theorem center_of_circle : ∃ c : ℝ × ℝ, 
  (∃ r : ℝ, ∀ x y : ℝ, (x - c.1) * (x - c.1) + (y - c.2) * (y - c.2) = r ↔ x^2 + y^2 - 6*x - 2*y - 15 = 0) → c = (3, 1) :=
by 
  sorry

end center_of_circle_l105_105952


namespace albert_needs_more_money_l105_105689

-- Definitions derived from the problem conditions
def cost_paintbrush : ℝ := 1.50
def cost_paints : ℝ := 4.35
def cost_easel : ℝ := 12.65
def money_albert_has : ℝ := 6.50

-- Statement asserting the amount of money Albert needs
theorem albert_needs_more_money : (cost_paintbrush + cost_paints + cost_easel) - money_albert_has = 12 :=
by
  sorry

end albert_needs_more_money_l105_105689


namespace solve_congruence_l105_105182

theorem solve_congruence :
  ∃ a m : ℕ, m ≥ 2 ∧ a < m ∧ a + m = 27 ∧ (10 * x + 3 ≡ 7 [MOD 15]) → x ≡ 12 [MOD 15] := 
by
  sorry

end solve_congruence_l105_105182


namespace find_y_l105_105149

theorem find_y (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = -5) : y = 28 := by
  sorry

end find_y_l105_105149


namespace staff_members_attended_meeting_l105_105383

theorem staff_members_attended_meeting
  (n_doughnuts_served : ℕ)
  (e_each_staff_member : ℕ)
  (n_doughnuts_left : ℕ)
  (h1 : n_doughnuts_served = 50)
  (h2 : e_each_staff_member = 2)
  (h3 : n_doughnuts_left = 12) :
  (n_doughnuts_served - n_doughnuts_left) / e_each_staff_member = 19 := 
by
  sorry

end staff_members_attended_meeting_l105_105383


namespace range_of_m_l105_105911

theorem range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) (hineq : 4 / (x + 1) + 1 / y < m^2 + (3 / 2) * m) :
  m < -3 ∨ m > 3 / 2 :=
by sorry

end range_of_m_l105_105911


namespace simplify_and_evaluate_expression_l105_105321

theorem simplify_and_evaluate_expression (m : ℕ) (h : m = 2) :
  ( (↑m + 1) / (↑m - 1) + 1 ) / ( (↑m + m^2) / (m^2 - 2*m + 1) ) - ( 2 - 2*↑m ) / ( m^2 - 1 ) = 4 / 3 :=
by sorry

end simplify_and_evaluate_expression_l105_105321


namespace sum_of_decimals_is_fraction_l105_105872

theorem sum_of_decimals_is_fraction :
  (0.2 : ℚ) + (0.03 : ℚ) + (0.004 : ℚ) + (0.0005 : ℚ) + (0.00006 : ℚ) = 1466 / 6250 := 
by
  sorry

end sum_of_decimals_is_fraction_l105_105872


namespace mary_total_cards_l105_105571

def mary_initial_cards := 33
def torn_cards := 6
def cards_given_by_sam := 23

theorem mary_total_cards : mary_initial_cards - torn_cards + cards_given_by_sam = 50 :=
  by
    sorry

end mary_total_cards_l105_105571


namespace apples_per_box_l105_105426

variable (A : ℕ) -- Number of apples packed in a box

-- Conditions
def normal_boxes_per_day := 50
def days_per_week := 7
def boxes_first_week := normal_boxes_per_day * days_per_week * A
def boxes_second_week := (normal_boxes_per_day * A - 500) * days_per_week
def total_apples := 24500

-- Theorem
theorem apples_per_box : boxes_first_week + boxes_second_week = total_apples → A = 40 :=
by
  sorry

end apples_per_box_l105_105426


namespace problem1_problem2_problem3_problem4_l105_105078

-- SI. 1
theorem problem1 (b m n : ℕ) (A : ℝ) (h1 : b = 4) (h2 : m = 1) (h3 : n = 1) (h4 : A = (b^m)^n + b^(m+n)) :
  A = 20 := 
by 
  simp [h1, h2, h3, h4]; 
  sorry

-- SI. 2
theorem problem2 (A B : ℝ) (h1 : A = 20) (h2 : 2^A = B^10) (h3 : 0 < B) :
  B = 4 := 
by 
  simp [h1, h2, h3]; 
  sorry

-- SI. 3
theorem problem3 (B C : ℝ) (h1 : B = 4) (h2 : (sqrt((20 * B + 45) / C)) = C) :
  C = 5 := 
by 
  simp [h1, h2];
  sorry

-- SI. 4
theorem problem4 (C D : ℝ) (h1 : C = 5) (h2 : D = C * (Real.sin (30 : ℝ) * (Real.pi / 180))) :
  D = 2.5 := 
by 
  simp [h1, h2];
  sorry

end problem1_problem2_problem3_problem4_l105_105078


namespace length_of_bridge_correct_l105_105353

noncomputable def L_train : ℝ := 180
noncomputable def v_km_per_hr : ℝ := 60  -- speed in km/hr
noncomputable def t : ℝ := 25

-- Convert speed from km/hr to m/s
noncomputable def km_per_hr_to_m_per_s (v: ℝ) : ℝ := v * (1000 / 3600)
noncomputable def v : ℝ := km_per_hr_to_m_per_s v_km_per_hr

-- Distance covered by the train while crossing the bridge
noncomputable def d : ℝ := v * t

-- Length of the bridge
noncomputable def L_bridge : ℝ := d - L_train

theorem length_of_bridge_correct :
  L_bridge = 236.75 :=
  by
    sorry

end length_of_bridge_correct_l105_105353


namespace train_speed_clicks_l105_105177

theorem train_speed_clicks (x : ℝ) (v : ℝ) (t : ℝ) 
  (h1 : v = x * 5280 / 60) 
  (h2 : t = 25) 
  (h3 : 70 * t = v * 25) : v = 70 := sorry

end train_speed_clicks_l105_105177


namespace parallel_condition_l105_105530

theorem parallel_condition (a : ℝ) : (a = -1) ↔ (¬ (a = -1 ∧ a ≠ 1)) ∧ (¬ (a ≠ -1 ∧ a = 1)) :=
by
  sorry

end parallel_condition_l105_105530


namespace tiling_possible_values_of_n_l105_105846

-- Define the sizes of the grid and the tiles
def grid_size : ℕ × ℕ := (9, 7)
def l_tile_size : ℕ := 3  -- L-shaped tile composed of three unit squares
def square_tile_size : ℕ := 4  -- square tile composed of four unit squares

-- Formalize the properties of the grid and the constraints for the tiling
def total_squares : ℕ := grid_size.1 * grid_size.2
def white_squares (n : ℕ) : ℕ := 3 * n
def black_squares (n : ℕ) : ℕ := n
def total_black_squares : ℕ := 20
def total_white_squares : ℕ := total_squares - total_black_squares

-- The main theorem statement
theorem tiling_possible_values_of_n (n : ℕ) : 
  (n = 2 ∨ n = 5 ∨ n = 8 ∨ n = 11 ∨ n = 14 ∨ n = 17 ∨ n = 20) ↔
  (3 * (total_white_squares - 2 * (20 - n)) / 3 + n = 23 ∧ n + (total_black_squares - n) = 20) :=
sorry

end tiling_possible_values_of_n_l105_105846


namespace smallest_three_digit_common_multiple_of_3_and_5_l105_105973

theorem smallest_three_digit_common_multiple_of_3_and_5 : 
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m) :=
by 
  sorry

end smallest_three_digit_common_multiple_of_3_and_5_l105_105973


namespace solution_is_correct_l105_105976

-- Initial conditions
def initial_volume : ℝ := 6
def initial_concentration : ℝ := 0.40
def target_concentration : ℝ := 0.50

-- Given that we start with 2.4 liters of pure alcohol in a 6-liter solution
def initial_pure_alcohol : ℝ := initial_volume * initial_concentration

-- Expected result after adding x liters of pure alcohol
def final_solution_volume (x : ℝ) : ℝ := initial_volume + x
def final_pure_alcohol (x : ℝ) : ℝ := initial_pure_alcohol + x

-- Equation to prove
theorem solution_is_correct (x : ℝ) :
  (final_pure_alcohol x) / (final_solution_volume x) = target_concentration ↔ 
  x = 1.2 := 
sorry

end solution_is_correct_l105_105976


namespace represent_sum_and_product_eq_231_l105_105319

theorem represent_sum_and_product_eq_231 :
  ∃ (x y z w : ℕ), x = 3 ∧ y = 7 ∧ z = 11 ∧ w = 210 ∧ (231 = x + y + z + w) ∧ (231 = x * y * z) :=
by
  -- The proof is omitted here.
  sorry

end represent_sum_and_product_eq_231_l105_105319


namespace order_of_x_given_conditions_l105_105487

variables (x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅ : ℝ)

def system_equations :=
  x₁ + x₂ + x₃ = a₁ ∧
  x₂ + x₃ + x₄ = a₂ ∧
  x₃ + x₄ + x₅ = a₃ ∧
  x₄ + x₅ + x₁ = a₄ ∧
  x₅ + x₁ + x₂ = a₅

def a_descending_order :=
  a₁ > a₂ ∧
  a₂ > a₃ ∧
  a₃ > a₄ ∧
  a₄ > a₅

theorem order_of_x_given_conditions (h₁ : system_equations x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅) :
  a_descending_order a₁ a₂ a₃ a₄ a₅ →
  x₃ > x₁ ∧ x₁ > x₄ ∧ x₄ > x₂ ∧ x₂ > x₅ := sorry

end order_of_x_given_conditions_l105_105487


namespace randy_trip_length_l105_105037

-- Define the conditions
noncomputable def fraction_gravel := (1/4 : ℚ)
noncomputable def miles_pavement := (30 : ℚ)
noncomputable def fraction_dirt := (1/6 : ℚ)

-- The proof statement
theorem randy_trip_length :
  ∃ x : ℚ, (fraction_gravel + fraction_dirt + (miles_pavement / x) = 1) ∧ x = 360 / 7 := 
by
  sorry

end randy_trip_length_l105_105037


namespace opposite_of_neg_two_l105_105792

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l105_105792


namespace solve_eq_l105_105442

theorem solve_eq {x : ℝ} (h : x + 2 * Real.sqrt x - 8 = 0) : x = 4 :=
by
  sorry

end solve_eq_l105_105442


namespace part1_part2_l105_105413
-- Importing the entire Mathlib library for required definitions

-- Define the sequence a_n with the conditions given in the problem
def a : ℕ → ℚ
| 0       => 1
| (n + 1) => a n / (2 * a n + 1)

-- Prove the given claims
theorem part1 (n : ℕ) : a n = (1 : ℚ) / (2 * n + 1) :=
sorry

def b (n : ℕ) : ℚ := a n * a (n + 1)

-- The sum of the first n terms of the sequence b_n is denoted as T_n
def T : ℕ → ℚ
| 0       => 0
| (n + 1) => T n + b n

-- Prove the given sum
theorem part2 (n : ℕ) : T n = (n : ℚ) / (2 * n + 1) :=
sorry

end part1_part2_l105_105413


namespace yellow_marbles_count_l105_105301

-- Definitions based on given conditions
def blue_marbles : ℕ := 10
def green_marbles : ℕ := 5
def black_marbles : ℕ := 1
def probability_black : ℚ := 1 / 28
def total_marbles : ℕ := 28

-- Problem statement to prove
theorem yellow_marbles_count :
  (total_marbles = blue_marbles + green_marbles + black_marbles + n) →
  (probability_black = black_marbles / total_marbles) →
  n = 12 :=
by
  intros; sorry

end yellow_marbles_count_l105_105301


namespace smallest_divisor_l105_105249

theorem smallest_divisor (k n : ℕ) (x y : ℤ) :
  (∃ n : ℕ, k ∣ 2^n + 15) ∧ (∃ x y : ℤ, k = 3 * x^2 - 4 * x * y + 3 * y^2) → k = 23 := by
  sorry

end smallest_divisor_l105_105249


namespace blocks_used_l105_105581

theorem blocks_used (initial_blocks used_blocks : ℕ) (h_initial : initial_blocks = 78) (h_left : initial_blocks - used_blocks = 59) : used_blocks = 19 := by
  sorry

end blocks_used_l105_105581


namespace opposite_of_neg2_l105_105805

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l105_105805


namespace match_Tile_C_to_Rectangle_III_l105_105059

-- Define the structure for a Tile
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the given tiles
def Tile_A : Tile := { top := 5, right := 3, bottom := 7, left := 2 }
def Tile_B : Tile := { top := 3, right := 6, bottom := 2, left := 8 }
def Tile_C : Tile := { top := 7, right := 9, bottom := 1, left := 3 }
def Tile_D : Tile := { top := 1, right := 8, bottom := 5, left := 9 }

-- The proof problem: Prove that Tile C should be matched to Rectangle III
theorem match_Tile_C_to_Rectangle_III : (Tile_C = { top := 7, right := 9, bottom := 1, left := 3 }) → true := 
by
  intros
  sorry

end match_Tile_C_to_Rectangle_III_l105_105059


namespace xyz_value_l105_105044

theorem xyz_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
    (hx : x * (y + z) = 162)
    (hy : y * (z + x) = 180)
    (hz : z * (x + y) = 198)
    (h_sum : x + y + z = 26) :
    x * y * z = 2294.67 :=
by
  sorry

end xyz_value_l105_105044


namespace domain_of_v_l105_105066

noncomputable def v (x : ℝ) : ℝ := 1 / (x - 1)^(1 / 3)

theorem domain_of_v :
  {x : ℝ | ∃ y : ℝ, y ≠ 0 ∧ y = (v x)} = {x | x ≠ 1} := by
  sorry

end domain_of_v_l105_105066


namespace maria_nickels_l105_105427

theorem maria_nickels (dimes quarters_initial quarters_additional : ℕ) (total_amount : ℚ) 
  (Hd : dimes = 4) (Hqi : quarters_initial = 4) (Hqa : quarters_additional = 5) (Htotal : total_amount = 3) : 
  (dimes * 0.10 + quarters_initial * 0.25 + quarters_additional * 0.25 + n/20) = total_amount → n = 7 :=
  sorry

end maria_nickels_l105_105427


namespace jerry_total_hours_at_field_l105_105562
-- Import the entire necessary library

-- Lean statement of the problem
theorem jerry_total_hours_at_field 
  (games_per_daughter : ℕ)
  (practice_hours_per_game : ℕ)
  (game_duration : ℕ)
  (daughters : ℕ)
  (h1: games_per_daughter = 8)
  (h2: practice_hours_per_game = 4)
  (h3: game_duration = 2)
  (h4: daughters = 2)
 : (game_duration * games_per_daughter * daughters + practice_hours_per_game * games_per_daughter * daughters) = 96 :=
by
  -- Proof not required, so we skip it with sorry
  sorry

end jerry_total_hours_at_field_l105_105562


namespace equivalent_terminal_angle_l105_105948

theorem equivalent_terminal_angle :
  ∃ n : ℤ, 660 = n * 360 - 420 := 
by
  sorry

end equivalent_terminal_angle_l105_105948


namespace Craig_initial_apples_l105_105700

variable (j : ℕ) (shared : ℕ) (left : ℕ)

theorem Craig_initial_apples (HJ : j = 11) (HS : shared = 7) (HL : left = 13) :
  shared + left = 20 := by
  sorry

end Craig_initial_apples_l105_105700


namespace min_value_of_a_plus_2b_l105_105005

theorem min_value_of_a_plus_2b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 2 / a + 1 / b = 1) : a + 2 * b = 4 :=
sorry

end min_value_of_a_plus_2b_l105_105005


namespace trapezoid_perimeter_l105_105555

noncomputable def length_AD : ℝ := 8
noncomputable def length_BC : ℝ := 18
noncomputable def length_AB : ℝ := 12 -- Derived from tangency and symmetry considerations
noncomputable def length_CD : ℝ := 18

theorem trapezoid_perimeter (ABCD : Π (a b c d : Type), a → b → c → d → Prop)
  (AD BC AB CD : ℝ)
  (h1 : AD = 8) (h2 : BC = 18) (h3 : AB = 12) (h4 : CD = 18)
  : AD + BC + AB + CD = 56 :=
by
  rw [h1, h2, h3, h4]
  norm_num

end trapezoid_perimeter_l105_105555


namespace total_earnings_l105_105837

theorem total_earnings (zachary_games : ℕ) (price_per_game : ℝ) (jason_percentage_increase : ℝ) (ryan_extra : ℝ)
  (h1 : zachary_games = 40) (h2 : price_per_game = 5) (h3 : jason_percentage_increase = 0.30) (h4 : ryan_extra = 50) :
  let z_earnings := zachary_games * price_per_game,
      j_earnings := z_earnings * (1 + jason_percentage_increase),
      r_earnings := j_earnings + ryan_extra
  in z_earnings + j_earnings + r_earnings = 770 := by
  sorry

end total_earnings_l105_105837


namespace gasoline_added_correct_l105_105848

def tank_capacity := 48
def initial_fraction := 3 / 4
def final_fraction := 9 / 10

def gasoline_at_initial_fraction (capacity: ℝ) (fraction: ℝ) : ℝ := capacity * fraction
def gasoline_at_final_fraction (capacity: ℝ) (fraction: ℝ) : ℝ := capacity * fraction
def gasoline_added (initial: ℝ) (final: ℝ) : ℝ := final - initial

theorem gasoline_added_correct (capacity: ℝ) (initial_fraction: ℝ) (final_fraction: ℝ)
  (h_capacity : capacity = 48) (h_initial : initial_fraction = 3 / 4) (h_final : final_fraction = 9 / 10) :
  gasoline_added (gasoline_at_initial_fraction capacity initial_fraction) (gasoline_at_final_fraction capacity final_fraction) = 7.2 :=
by
  sorry

end gasoline_added_correct_l105_105848


namespace gold_stickers_for_second_student_l105_105961

theorem gold_stickers_for_second_student :
  (exists f : ℕ → ℕ,
      f 1 = 29 ∧
      f 3 = 41 ∧
      f 4 = 47 ∧
      f 5 = 53 ∧
      f 6 = 59 ∧
      (∀ n, f (n + 1) - f n = 6 ∨ f (n + 2) - f n = 12)) →
  (∃ f : ℕ → ℕ, f 2 = 35) :=
by
  sorry

end gold_stickers_for_second_student_l105_105961


namespace quadratic_rational_root_contradiction_l105_105036

def int_coefficients (a b c : ℤ) : Prop := true  -- Placeholder for the condition that coefficients are integers

def is_rational_root (a b c p q : ℤ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ p.gcd q = 1 ∧ a * p^2 + b * p * q + c * q^2 = 0  -- p/q is a rational root in simplest form

def ear_even (b c : ℤ) : Prop :=
  b % 2 = 0 ∨ c % 2 = 0

def assume_odd (a b c : ℤ) : Prop :=
  a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0

theorem quadratic_rational_root_contradiction (a b c p q : ℤ)
  (h1 : int_coefficients a b c)
  (h2 : a ≠ 0)
  (h3 : is_rational_root a b c p q)
  (h4 : ear_even b c) :
  assume_odd a b c :=
sorry

end quadratic_rational_root_contradiction_l105_105036


namespace maximum_discount_rate_l105_105627

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l105_105627


namespace vector_CD_l105_105289

-- Define the vector space and the vectors a and b
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V)
variables (a b : V)

-- Define the conditions
def is_on_line (D A B : V) := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (D = t • A + (1 - t) • B)
def da_eq_2bd (D A B : V) := (A - D) = 2 • (D - B)

-- Define the triangle ABC and the specific vectors CA and CB
variables (CA := C - A) (CB := C - B)
variable (H1 : is_on_line D A B)
variable (H2 : da_eq_2bd D A B)
variable (H3 : CA = a)
variable (H4 : CB = b)

-- Prove the conclusion
theorem vector_CD (H1 : is_on_line D A B) (H2 : da_eq_2bd D A B)
  (H3 : CA = a) (H4 : CB = b) : 
  (C - D) = (1/3 : ℝ) • a + (2/3 : ℝ) • b :=
sorry

end vector_CD_l105_105289


namespace missing_fraction_is_two_l105_105463

theorem missing_fraction_is_two :
  (1/2) + (-5/6) + (1/5) + (1/4) + (-9/20) + (-5/6) + 2 = 0.8333333333333334 := by
  sorry

end missing_fraction_is_two_l105_105463


namespace find_m_value_l105_105298

def symmetric_inverse (g : ℝ → ℝ) (h : ℝ → ℝ) :=
  ∀ x, g (h x) = x ∧ h (g x) = x

def symmetric_y_axis (f : ℝ → ℝ) (g : ℝ → ℝ) :=
  ∀ x, f x = g (-x)

theorem find_m_value :
  (∀ g, symmetric_inverse g (Real.exp) → (∀ f, symmetric_y_axis f g → (∀ m, f m = -1 → m = - (1 / Real.exp 1)))) := by
  sorry

end find_m_value_l105_105298


namespace ratio_blue_yellow_l105_105026

theorem ratio_blue_yellow (total_butterflies blue_butterflies black_butterflies : ℕ)
  (h_total : total_butterflies = 19)
  (h_blue : blue_butterflies = 6)
  (h_black : black_butterflies = 10) :
  (blue_butterflies : ℚ) / (total_butterflies - blue_butterflies - black_butterflies : ℚ) = 2 / 1 := 
by {
  sorry
}

end ratio_blue_yellow_l105_105026


namespace combined_area_correct_l105_105891

def popsicle_stick_length_gino : ℚ := 9 / 2
def popsicle_stick_width_gino : ℚ := 2 / 5
def popsicle_stick_length_me : ℚ := 6
def popsicle_stick_width_me : ℚ := 3 / 5

def number_of_sticks_gino : ℕ := 63
def number_of_sticks_me : ℕ := 50

def side_length_square : ℚ := number_of_sticks_gino / 4 * popsicle_stick_length_gino
def area_square : ℚ := side_length_square ^ 2

def length_rectangle : ℚ := (number_of_sticks_me / 2) * popsicle_stick_length_me
def width_rectangle : ℚ := (number_of_sticks_me / 2) * popsicle_stick_width_me
def area_rectangle : ℚ := length_rectangle * width_rectangle

def combined_area : ℚ := area_square + area_rectangle

theorem combined_area_correct : combined_area = 6806.25 := by
  sorry

end combined_area_correct_l105_105891


namespace intervals_of_monotonicity_range_of_m_l105_105893

open Real

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := 2 * (x - a) / (x ^ 2 + b * x + 1)

theorem intervals_of_monotonicity (a b : ℝ) (h_odd : ∀ x, f a b x = - f a b (-x)) (h_a_b_zero : a = 0 ∧ b = 0) :
  (∀ x, f 0 0 x = 2 * x / (x ^ 2 + 1)) → 
  (∀ x, (f 0 0 x > f 0 0 (x - 1) ∧ -1 < x ∧ x < 1) ∨ (f 0 0 ↑x < f 0 0 (x - 1) ∧ (x > 1 ∨ x < -1))) :=
by sorry

theorem range_of_m (a b : ℝ) (h_odd : ∀ x, f a b x = - f a b (-x)) (h_a_b_zero : a = 0 ∧ b = 0) :
  (∀ m, (∃ x, 2 * m - 1 > f 0 0 x) ↔ m > 0) :=
by sorry

end intervals_of_monotonicity_range_of_m_l105_105893


namespace team_winning_percentage_l105_105866

theorem team_winning_percentage :
  let first_games := 100
  let remaining_games := 125 - first_games
  let won_first_games := 75
  let percentage_won := 50
  let won_remaining_games := Nat.ceil ((percentage_won : ℝ) / 100 * remaining_games)
  let total_won_games := won_first_games + won_remaining_games
  let total_games := 125
  let winning_percentage := (total_won_games : ℝ) / total_games * 100
  winning_percentage = 70.4 :=
by sorry

end team_winning_percentage_l105_105866


namespace maximum_discount_rate_l105_105624

theorem maximum_discount_rate (c s p_m : ℝ) (h1 : c = 4) (h2 : s = 5) (h3 : p_m = 0.4) :
  ∃ x : ℝ, s * (1 - x / 100) - c ≥ p_m ∧ ∀ y, y > x → s * (1 - y / 100) - c < p_m :=
by
  -- Establish the context and conditions
  sorry

end maximum_discount_rate_l105_105624


namespace parabola_directrix_l105_105954

theorem parabola_directrix (x y : ℝ) (h : y = 2 * x^2) : y = - (1 / 8) :=
sorry

end parabola_directrix_l105_105954


namespace product_increased_five_times_l105_105202

variables (A B : ℝ)

theorem product_increased_five_times (h : A * B = 1.6) : (5 * A) * (5 * B) = 40 :=
by
  sorry

end product_increased_five_times_l105_105202


namespace bus_stops_duration_per_hour_l105_105842

def speed_without_stoppages : ℝ := 90
def speed_with_stoppages : ℝ := 84
def distance_covered_lost := speed_without_stoppages - speed_with_stoppages

theorem bus_stops_duration_per_hour :
  distance_covered_lost / speed_without_stoppages * 60 = 4 :=
by
  sorry

end bus_stops_duration_per_hour_l105_105842


namespace find_A_l105_105708

theorem find_A (A : ℝ) (h : 4 * A + 5 = 33) : A = 7 :=
  sorry

end find_A_l105_105708


namespace not_diff_of_squares_count_l105_105728

theorem not_diff_of_squares_count :
  ∃ n : ℕ, n ≤ 1000 ∧
  (∀ k : ℕ, k > 0 ∧ k ≤ 1000 → (∃ a b : ℤ, k = a^2 - b^2 ↔ k % 4 ≠ 2)) ∧
  n = 250 :=
sorry

end not_diff_of_squares_count_l105_105728


namespace sum_of_decimals_as_common_fraction_l105_105884

theorem sum_of_decimals_as_common_fraction:
  (2 / 10 : ℚ) + (3 / 100 : ℚ) + (4 / 1000 : ℚ) + (5 / 10000 : ℚ) + (6 / 100000 : ℚ) = 733 / 3125 :=
by
  -- Explanation and proof steps are omitted as directed.
  sorry

end sum_of_decimals_as_common_fraction_l105_105884


namespace petya_can_write_divisible_by_2019_l105_105765

open Nat

theorem petya_can_write_divisible_by_2019 (M : ℕ) (h : ∃ k : ℕ, M = (10^k - 1) / 9) : ∃ N : ℕ, (N = (10^M - 1) / 9) ∧ 2019 ∣ N :=
by
  sorry

end petya_can_write_divisible_by_2019_l105_105765


namespace kids_in_group_l105_105505

theorem kids_in_group :
  ∃ (K : ℕ), (∃ (A : ℕ), A + K = 9 ∧ 2 * A = 14) ∧ K = 2 :=
by
  sorry

end kids_in_group_l105_105505


namespace problem_l105_105380

def expr : ℤ := 7^2 - 4 * 5 + 2^2

theorem problem : expr = 33 := by
  sorry

end problem_l105_105380


namespace subset_A_l105_105140

open Set

theorem subset_A (A : Set ℝ) (h : A = { x | x > -1 }) : {0} ⊆ A :=
by
  sorry

end subset_A_l105_105140


namespace functions_identified_l105_105527

variable (n : ℕ) (hn : n > 1)
variable {f : ℕ → ℝ → ℝ}

-- Define the conditions f1, f2, ..., fn
axiom cond_1 (x y : ℝ) : f 1 x + f 1 y = f 2 x * f 2 y
axiom cond_2 (x y : ℝ) : f 2 (x^2) + f 2 (y^2) = f 3 x * f 3 y
axiom cond_3 (x y : ℝ) : f 3 (x^3) + f 3 (y^3) = f 4 x * f 4 y
-- ... Similarly define conditions up to cond_n
axiom cond_n (x y : ℝ) : f n (x^n) + f n (y^n) = f 1 x * f 1 y

theorem functions_identified (i : ℕ) (hi₁ : 1 ≤ i) (hi₂ : i ≤ n) (x : ℝ) :
  f i x = 0 ∨ f i x = 2 :=
sorry

end functions_identified_l105_105527


namespace opposite_of_neg_two_l105_105791

theorem opposite_of_neg_two : ∃ x : ℤ, -2 + x = 0 ∧ x = 2 :=
by {
  existsi 2,
  split,
  { refl },
  { refl }
}

end opposite_of_neg_two_l105_105791


namespace probability_x_gt_3y_l105_105934

noncomputable def rect_region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3020 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3010}

theorem probability_x_gt_3y : 
  (∫ p in rect_region, if p.1 > 3 * p.2 then 1 else (0:ℝ)) / 
  (∫ p in rect_region, (1:ℝ)) = 1007 / 6020 := sorry

end probability_x_gt_3y_l105_105934


namespace cannot_transform_with_swap_rows_and_columns_l105_105222

def initialTable : Matrix (Fin 3) (Fin 3) ℕ :=
![![1, 2, 3], ![4, 5, 6], ![7, 8, 9]]

def goalTable : Matrix (Fin 3) (Fin 3) ℕ :=
![![1, 4, 7], ![2, 5, 8], ![3, 6, 9]]

theorem cannot_transform_with_swap_rows_and_columns :
  ¬ ∃ (is_transformed_by_swapping : Matrix (Fin 3) (Fin 3) ℕ → Matrix (Fin 3) (Fin 3) ℕ → Prop),
    is_transformed_by_swapping initialTable goalTable :=
by sorry

end cannot_transform_with_swap_rows_and_columns_l105_105222


namespace max_discount_rate_l105_105654

theorem max_discount_rate (cp sp : ℝ) (h1 : cp = 4) (h2 : sp = 5) : 
  ∃ x : ℝ, 5 * (1 - x / 100) - 4 ≥ 0.4 → x ≤ 12 := by
  use 12
  intro h
  sorry

end max_discount_rate_l105_105654


namespace midpoint_of_symmetric_chord_on_ellipse_l105_105398

theorem midpoint_of_symmetric_chord_on_ellipse
  (A B : ℝ × ℝ) -- coordinates of points A and B
  (hA : (A.1^2 / 16) + (A.2^2 / 4) = 1) -- A lies on the ellipse
  (hB : (B.1^2 / 16) + (B.2^2 / 4) = 1) -- B lies on the ellipse
  (symm : 2 * (A.1 + B.1) / 2 - 2 * (A.2 + B.2) / 2 - 3 = 0) -- A and B are symmetric about the line
  : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (2, 1 / 2) :=
  sorry

end midpoint_of_symmetric_chord_on_ellipse_l105_105398


namespace smallest_c_value_l105_105592

theorem smallest_c_value :
  ∃ a b c : ℕ, a * b * c = 3990 ∧ a + b + c = 56 ∧ a > 0 ∧ b > 0 ∧ c > 0 :=
by {
  -- Skipping proof as instructed
  sorry
}

end smallest_c_value_l105_105592


namespace number_of_samples_from_retired_l105_105116

def ratio_of_forms (retired current students : ℕ) : Prop :=
retired = 3 ∧ current = 7 ∧ students = 40

def total_sampled_forms := 300

theorem number_of_samples_from_retired :
  ∃ (xr : ℕ), ratio_of_forms 3 7 40 → xr = (300 / (3 + 7 + 40)) * 3 :=
sorry

end number_of_samples_from_retired_l105_105116


namespace sqrt_9_minus_1_eq_2_l105_105861

theorem sqrt_9_minus_1_eq_2 : Real.sqrt 9 - 1 = 2 := by
  sorry

end sqrt_9_minus_1_eq_2_l105_105861


namespace actual_plot_area_in_acres_l105_105682

-- Define the conditions
def base1_cm := 18
def base2_cm := 12
def height_cm := 8
def scale_cm_to_miles := 5
def sq_mile_to_acres := 640

-- Prove the question which is to find the actual plot area in acres
theorem actual_plot_area_in_acres : 
  (1/2 * (base1_cm + base2_cm) * height_cm * (scale_cm_to_miles ^ 2) * sq_mile_to_acres) = 1920000 :=
by
  sorry

end actual_plot_area_in_acres_l105_105682


namespace total_chickens_l105_105012

   def number_of_hens := 12
   def hens_to_roosters_ratio := 3
   def chicks_per_hen := 5

   theorem total_chickens (h : number_of_hens = 12)
                          (r : hens_to_roosters_ratio = 3)
                          (c : chicks_per_hen = 5) :
     number_of_hens + (number_of_hens / hens_to_roosters_ratio) + (number_of_hens * chicks_per_hen) = 76 :=
   by
     sorry
   
end total_chickens_l105_105012


namespace train_speed_l105_105684

variable (length : ℕ) (time : ℕ)
variable (h_length : length = 120)
variable (h_time : time = 6)

theorem train_speed (length time : ℕ) (h_length : length = 120) (h_time : time = 6) :
  length / time = 20 := by
  sorry

end train_speed_l105_105684


namespace max_discount_rate_l105_105618

theorem max_discount_rate (cp sp : ℝ) (min_profit_margin discount_rate : ℝ) 
  (h_cost : cp = 4) 
  (h_sell : sp = 5) 
  (h_profit : min_profit_margin = 0.1) :
  discount_rate ≤ 12 :=
by 
  sorry

end max_discount_rate_l105_105618


namespace height_of_cylinder_l105_105683

theorem height_of_cylinder (r_hemisphere : ℝ) (r_cylinder : ℝ) (h_cylinder : ℝ) :
  r_hemisphere = 7 → r_cylinder = 3 → h_cylinder = 2 * Real.sqrt 10 :=
by
  intro r_hemisphere_eq r_cylinder_eq
  sorry

end height_of_cylinder_l105_105683


namespace range_of_x_satisfying_inequality_l105_105164

def otimes (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem range_of_x_satisfying_inequality :
  { x : ℝ | otimes x (x - 2) < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end range_of_x_satisfying_inequality_l105_105164


namespace neznaika_mistake_correct_numbers_l105_105433

theorem neznaika_mistake_correct_numbers (N : ℕ) :
  (10 ≤ N) ∧ (N ≤ 99) ∧
  ¬ (N % 30 = 0) ∧
  (N % 3 = 0) ∧
  ¬ (N % 10 = 0) ∧
  ((N % 9 = 0) ∨ (N % 15 = 0) ∨ (N % 18 = 0)) ∧
  (N % 5 ≠ 0) ∧
  (N % 4 ≠ 0) → N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end neznaika_mistake_correct_numbers_l105_105433


namespace negation_of_proposition_l105_105904

theorem negation_of_proposition {c : ℝ} (h : ∃ (c : ℝ), c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) :
  ∀ (c : ℝ), c > 0 → ¬ ∃ x : ℝ, x^2 - x + c = 0 :=
by
  sorry

end negation_of_proposition_l105_105904


namespace P_plus_Q_l105_105924

theorem P_plus_Q (P Q : ℝ) (h : ∀ x, x ≠ 3 → (P / (x - 3) + Q * (x - 2) = (-5 * x^2 + 20 * x + 36) / (x - 3))) : P + Q = 46 :=
sorry

end P_plus_Q_l105_105924


namespace sally_total_spent_l105_105320

-- Define the prices paid by Sally for peaches after the coupon and for cherries
def P_peaches : ℝ := 12.32
def C_cherries : ℝ := 11.54

-- State the problem to prove that the total amount Sally spent is 23.86
theorem sally_total_spent : P_peaches + C_cherries = 23.86 := by
  sorry

end sally_total_spent_l105_105320


namespace cubic_geometric_progression_l105_105134

theorem cubic_geometric_progression (a b c : ℝ) (α β γ : ℝ) 
    (h_eq1 : α + β + γ = -a) 
    (h_eq2 : α * β + α * γ + β * γ = b) 
    (h_eq3 : α * β * γ = -c) 
    (h_gp : ∃ k q : ℝ, α = k / q ∧ β = k ∧ γ = k * q) : 
    a^3 * c - b^3 = 0 :=
by
  sorry

end cubic_geometric_progression_l105_105134


namespace sum_positive_implies_at_least_one_positive_l105_105405

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l105_105405


namespace parabola_ratio_l105_105715

theorem parabola_ratio (P : ℝ) (A B F : ℝ × ℝ)
  (hA : A = (2, 4))
  (hB : B = (8, -8))
  (h_parabola : ∀ {x y : ℝ}, y^2 = 2 * P * x → true)
  (hF : F = (P, 0))
  (hA_on_parabola : (A.snd)^2 = 2 * P * A.fst)
  (hP_val : P = 4) :
  |dist A F : dist B F| = 2 / 5 :=
by
  sorry

end parabola_ratio_l105_105715


namespace horner_method_v3_correct_l105_105967

-- Define the polynomial function according to Horner's method
def horner (x : ℝ) : ℝ :=
  (((((3 * x - 2) * x + 2) * x - 4) * x) * x - 7)

-- Given the value of x
def x_val : ℝ := 2

-- Define v_3 based on the polynomial evaluated at x = 2 using Horner's method
def v3 : ℝ := horner x_val

-- Theorem stating what we need to prove
theorem horner_method_v3_correct : v3 = 16 :=
  by
    sorry

end horner_method_v3_correct_l105_105967


namespace jill_water_filled_jars_l105_105349

variable (gallons : ℕ) (quart_halfGallon_gallon : ℕ)
variable (h_eq : gallons = 14)
variable (h_eq_n : quart_halfGallon_gallon = 3 * 8)
variable (h_total : quart_halfGallon_gallon = 24)

theorem jill_water_filled_jars :
  3 * (gallons * 4 / 7) = 24 :=
sorry

end jill_water_filled_jars_l105_105349


namespace number_of_ways_to_fill_grid_l105_105104

noncomputable def totalWaysToFillGrid (S : Finset ℕ) : ℕ :=
  S.card.choose 5

theorem number_of_ways_to_fill_grid : totalWaysToFillGrid ({1, 2, 3, 4, 5, 6} : Finset ℕ) = 6 :=
by
  sorry

end number_of_ways_to_fill_grid_l105_105104


namespace Part1_Part2_l105_105409

-- For part 1
theorem Part1 : @fin.prob 0.5 (i → bool) 6 = \sum_{k=0}^{2} {binomial 6 k * (0.5)^k * (0.5)^(6-k)}
begin
  sorry
end

-- For part 2
theorem Part2 (n : nat) (E : ℝ := 0.5 * n) (D : ℝ := 0.25 * n) : 
  (1 - D / ((0.1 * n) ^ 2) ≥ 0.98) → (n ≥ 1250) :=
begin
  rw sub_ge at h,
  rw ge_eq_le at h,
  rw le_sub_iff_add_le at h,
  rw div_le_iff at h,
  norm_num at h,
  rw [pow_two, ge_eq_le] at h,
  rw mul_inv at h,
  exact h
  sorry
end


end Part1_Part2_l105_105409


namespace math_problem_l105_105021

theorem math_problem (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p + q + r = 0) :
    (p^2 * q^2 / ((p^2 - q * r) * (q^2 - p * r)) +
    p^2 * r^2 / ((p^2 - q * r) * (r^2 - p * q)) +
    q^2 * r^2 / ((q^2 - p * r) * (r^2 - p * q))) = 1 :=
by
  sorry

end math_problem_l105_105021


namespace max_discount_rate_l105_105636

theorem max_discount_rate
  (cost_price : ℝ := 4)
  (selling_price : ℝ := 5)
  (min_profit_margin : ℝ := 0.1)
  (x : ℝ) :
  (selling_price * (1 - x / 100) - cost_price ≥ cost_price * min_profit_margin) ↔ (x ≤ 12) :=
by
  intro h
  rw [sub_le_iff_le_add] at h
  rw [mul_sub, mul_one] at h
  rw [le_sub_iff_add_le]
  linarith

end max_discount_rate_l105_105636


namespace min_value_of_reciprocal_sums_l105_105278

variable {a b : ℝ}

theorem min_value_of_reciprocal_sums (ha : a ≠ 0) (hb : b ≠ 0) (h : 4 * a ^ 2 + b ^ 2 = 1) :
  (1 / a ^ 2) + (1 / b ^ 2) = 9 := by
  sorry

end min_value_of_reciprocal_sums_l105_105278


namespace wendy_third_day_miles_l105_105933

theorem wendy_third_day_miles (miles_day1 miles_day2 total_miles : ℕ)
  (h1 : miles_day1 = 125)
  (h2 : miles_day2 = 223)
  (h3 : total_miles = 493) :
  total_miles - (miles_day1 + miles_day2) = 145 :=
by sorry

end wendy_third_day_miles_l105_105933


namespace monkey_tree_height_l105_105495

theorem monkey_tree_height (hours: ℕ) (hop ft_per_hour : ℕ) (slip ft_per_hour : ℕ) (net_progress : ℕ) (final_hour : ℕ) (total_height : ℕ) :
  (hours = 18) ∧
  (hop = 3) ∧
  (slip = 2) ∧
  (net_progress = hop - slip) ∧
  (net_progress = 1) ∧
  (final_hour = 1) ∧
  (total_height = (hours - 1) * net_progress + hop) ∧
  (total_height = 20) :=
by
  sorry

end monkey_tree_height_l105_105495


namespace transylvanian_is_sane_human_l105_105615

def Transylvanian : Type := sorry -- Placeholder type for Transylvanian
def Human : Transylvanian → Prop := sorry
def Sane : Transylvanian → Prop := sorry
def InsaneVampire : Transylvanian → Prop := sorry

/-- The Transylvanian stated: "Either I am a human, or I am sane." -/
axiom statement (T : Transylvanian) : Human T ∨ Sane T

/-- Insane vampires only make true statements. -/
axiom insane_vampire_truth (T : Transylvanian) : InsaneVampire T → (Human T ∨ Sane T)

/-- Insane vampires cannot be sane or human. -/
axiom insane_vampire_condition (T : Transylvanian) : InsaneVampire T → ¬ Human T ∧ ¬ Sane T

theorem transylvanian_is_sane_human (T : Transylvanian) :
  ¬ (InsaneVampire T) → (Human T ∧ Sane T) := sorry

end transylvanian_is_sane_human_l105_105615


namespace part1_part2_l105_105263

open Set Real

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem part1 (m : ℝ) (h : Disjoint A (B m)) : m ∈ Iio 2 ∪ Ioi 4 := 
sorry

theorem part2 (m : ℝ) (h : A ∪ (univ \ (B m)) = univ) : m ∈ Iic 3 := 
sorry

end part1_part2_l105_105263


namespace solve_equation_1_solve_equation_2_l105_105770

open Real

theorem solve_equation_1 (x : ℝ) (h_ne1 : x + 1 ≠ 0) (h_ne2 : x - 3 ≠ 0) : 
  (5 / (x + 1) = 1 / (x - 3)) → x = 4 :=
by
    intro h
    sorry

theorem solve_equation_2 (x : ℝ) (h_ne1 : x - 4 ≠ 0) (h_ne2 : 4 - x ≠ 0) :
    (3 - x) / (x - 4) = 1 / (4 - x) - 2 → False :=
by
    intro h
    sorry

end solve_equation_1_solve_equation_2_l105_105770


namespace floor_eq_48_iff_l105_105384

-- Define the real number set I to be [8, 49/6)
def I : Set ℝ := { x | 8 ≤ x ∧ x < 49/6 }

-- The main statement to be proven
theorem floor_eq_48_iff (x : ℝ) : (Int.floor (x * Int.floor x) = 48) ↔ x ∈ I := 
by
  sorry

end floor_eq_48_iff_l105_105384


namespace common_card_cost_l105_105605

def totalDeckCost (rareCost uncommonCost commonCost numRares numUncommons numCommons : ℝ) : ℝ :=
  (numRares * rareCost) + (numUncommons * uncommonCost) + (numCommons * commonCost)

theorem common_card_cost (numRares numUncommons numCommons : ℝ) (rareCost uncommonCost totalCost : ℝ) : 
  numRares = 19 → numUncommons = 11 → numCommons = 30 → 
  rareCost = 1 → uncommonCost = 0.5 → totalCost = 32 → 
  commonCost = 0.25 :=
by 
  intros 
  sorry

end common_card_cost_l105_105605


namespace maximum_area_of_rectangular_playground_l105_105536

theorem maximum_area_of_rectangular_playground (P : ℕ) (A : ℕ) (h : P = 150) :
  ∃ (x y : ℕ), x + y = 75 ∧ A ≤ x * y ∧ A = 1406 :=
sorry

end maximum_area_of_rectangular_playground_l105_105536


namespace proof_problem_l105_105892

-- Given definitions
def A := { y : ℝ | ∃ x : ℝ, y = x^2 + 1 }
def B := { p : ℝ × ℝ | ∃ x : ℝ, p.snd = x^2 + 1 }

-- Theorem to prove 1 ∉ B and 2 ∈ A
theorem proof_problem : 1 ∉ B ∧ 2 ∈ A :=
by
  sorry

end proof_problem_l105_105892


namespace stratified_sampling_category_A_l105_105547

def total_students_A : ℕ := 2000
def total_students_B : ℕ := 3000
def total_students_C : ℕ := 4000
def total_students : ℕ := total_students_A + total_students_B + total_students_C
def total_selected : ℕ := 900

theorem stratified_sampling_category_A :
  (total_students_A * total_selected) / total_students = 200 :=
by
  sorry

end stratified_sampling_category_A_l105_105547


namespace total_bouncy_balls_l105_105310

-- Definitions of the given quantities
def r : ℕ := 4 -- number of red packs
def y : ℕ := 8 -- number of yellow packs
def g : ℕ := 4 -- number of green packs
def n : ℕ := 10 -- number of balls per pack

-- Proof statement to show the correct total number of balls
theorem total_bouncy_balls : r * n + y * n + g * n = 160 := by
  sorry

end total_bouncy_balls_l105_105310
