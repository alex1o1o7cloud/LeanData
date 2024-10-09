import Mathlib

namespace problem_equivalent_statement_l1485_148505

-- Conditions as Lean definitions
def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def periodic_property (f : ℝ → ℝ) := ∀ x, x ≥ 0 → f (x + 2) = -f x
def specific_interval (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 8

-- The main theorem
theorem problem_equivalent_statement (f : ℝ → ℝ) 
  (hf_even : is_even_function f) 
  (hf_periodic : periodic_property f) 
  (hf_specific : specific_interval f) :
  f (-2013) + f 2014 = 1 / 3 := 
sorry

end problem_equivalent_statement_l1485_148505


namespace find_x_value_l1485_148550

theorem find_x_value : (8 = 2^3) ∧ (8 * 8^32 = 8^33) ∧ (8^33 = 2^99) → ∃ x, 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 = 2^x ∧ x = 99 :=
by
  intros h
  sorry

end find_x_value_l1485_148550


namespace carpet_needed_in_sq_yards_l1485_148548

theorem carpet_needed_in_sq_yards :
  let length := 15
  let width := 10
  let area_sq_feet := length * width
  let conversion_factor := 9
  let area_sq_yards := area_sq_feet / conversion_factor
  area_sq_yards = 16.67 := by
  sorry

end carpet_needed_in_sq_yards_l1485_148548


namespace area_ratio_of_squares_l1485_148588

theorem area_ratio_of_squares (a b : ℕ) (h : 4 * a = 4 * (4 * b)) : (a * a) = 16 * (b * b) :=
by
  sorry

end area_ratio_of_squares_l1485_148588


namespace solve_trig_eq_l1485_148516

theorem solve_trig_eq :
  ∀ x k : ℤ, 
    (x = 2 * π / 3 + 2 * k * π ∨
     x = 7 * π / 6 + 2 * k * π ∨
     x = -π / 6 + 2 * k * π)
    → (|Real.cos x| + Real.cos (3 * x)) / (Real.sin x * Real.cos (2 * x)) = -2 * Real.sqrt 3 := 
by
  intros x k h
  sorry

end solve_trig_eq_l1485_148516


namespace solve_for_percentage_l1485_148514

-- Define the constants and variables
variables (P : ℝ)

-- Define the given conditions
def condition : Prop := (P / 100 * 1600 = P / 100 * 650 + 190)

-- Formalize the conjecture: if the conditions hold, then P = 20
theorem solve_for_percentage (h : condition P) : P = 20 :=
sorry

end solve_for_percentage_l1485_148514


namespace beau_age_today_l1485_148586

-- Definitions based on conditions
def sons_are_triplets : Prop := ∀ (i j : Nat), i ≠ j → i = 0 ∨ i = 1 ∨ i = 2 → j = 0 ∨ j = 1 ∨ j = 2
def sons_age_today : Nat := 16
def sum_of_ages_equals_beau_age_3_years_ago (beau_age_3_years_ago : Nat) : Prop :=
  beau_age_3_years_ago = 3 * (sons_age_today - 3)

-- Proposition to prove
theorem beau_age_today (beau_age_3_years_ago : Nat) (h_triplets : sons_are_triplets) 
  (h_ages_sum : sum_of_ages_equals_beau_age_3_years_ago beau_age_3_years_ago) : 
  beau_age_3_years_ago + 3 = 42 := 
by
  sorry

end beau_age_today_l1485_148586


namespace abby_damon_weight_l1485_148530

theorem abby_damon_weight (a' b' c' d' : ℕ) (h1 : a' + b' = 265) (h2 : b' + c' = 250) (h3 : c' + d' = 280) :
  a' + d' = 295 :=
  sorry -- Proof goes here

end abby_damon_weight_l1485_148530


namespace solve_for_x_l1485_148579

theorem solve_for_x (x : ℝ) (hx_pos : x > 0) (h_eq : 3 * x^2 + 13 * x - 10 = 0) : x = 2 / 3 :=
sorry

end solve_for_x_l1485_148579


namespace sasha_remainder_l1485_148567

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end sasha_remainder_l1485_148567


namespace ronalds_egg_sharing_l1485_148546

theorem ronalds_egg_sharing (total_eggs : ℕ) (eggs_per_friend : ℕ) (num_friends : ℕ) 
  (h1 : total_eggs = 16) (h2 : eggs_per_friend = 2) 
  (h3 : num_friends = total_eggs / eggs_per_friend) : 
  num_friends = 8 := 
by 
  sorry

end ronalds_egg_sharing_l1485_148546


namespace ratio_of_longer_side_to_square_l1485_148515

theorem ratio_of_longer_side_to_square (s a b : ℝ) (h1 : a * b = 2 * s^2) (h2 : a = 2 * b) : a / s = 2 :=
by
  sorry

end ratio_of_longer_side_to_square_l1485_148515


namespace calculate_sum_l1485_148599

theorem calculate_sum : (-2) + 1 = -1 :=
by 
  sorry

end calculate_sum_l1485_148599


namespace problem_1_problem_2_l1485_148578

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem problem_1 (x : ℝ) : f x ≥ 2 ↔ (x ≤ -7 ∨ x ≥ 5 / 3) :=
sorry

theorem problem_2 : ∃ x : ℝ, f x = -9 / 2 :=
sorry

end problem_1_problem_2_l1485_148578


namespace probability_either_A1_or_B1_not_both_is_half_l1485_148553

-- Definitions of the students
inductive Student
| A : ℕ → Student
| B : ℕ → Student
| C : ℕ → Student

-- Excellent grades students
def math_students := [Student.A 1, Student.A 2, Student.A 3]
def physics_students := [Student.B 1, Student.B 2]
def chemistry_students := [Student.C 1, Student.C 2]

-- Total number of ways to select one student from each category
def total_ways : ℕ := 3 * 2 * 2

-- Number of ways either A_1 or B_1 is selected but not both
def special_ways : ℕ := 1 * 1 * 2 + 2 * 1 * 2

-- Probability calculation
def probability := (special_ways : ℚ) / total_ways

-- Theorem to be proven
theorem probability_either_A1_or_B1_not_both_is_half :
  probability = 1 / 2 := by
  sorry

end probability_either_A1_or_B1_not_both_is_half_l1485_148553


namespace find_polynomial_coefficients_l1485_148589

-- Define the quadratic polynomial q(x) = ax^2 + bx + c
def polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions for polynomial
axiom condition1 (a b c : ℝ) : polynomial a b c (-2) = 9
axiom condition2 (a b c : ℝ) : polynomial a b c 1 = 2
axiom condition3 (a b c : ℝ) : polynomial a b c 3 = 10

-- Conjecture for the polynomial q(x)
theorem find_polynomial_coefficients : 
  ∃ (a b c : ℝ), 
    polynomial a b c (-2) = 9 ∧
    polynomial a b c 1 = 2 ∧
    polynomial a b c 3 = 10 ∧
    a = 19 / 15 ∧
    b = -2 / 15 ∧
    c = 13 / 15 :=
by {
  -- Placeholder proof
  sorry
}

end find_polynomial_coefficients_l1485_148589


namespace solution_set_of_inequality_l1485_148539

variable (a x : ℝ)

theorem solution_set_of_inequality (h : 0 < a ∧ a < 1) :
  (a - x) * (x - (1/a)) > 0 ↔ a < x ∧ x < 1/a :=
sorry

end solution_set_of_inequality_l1485_148539


namespace avg_divisible_by_4_between_15_and_55_eq_34_l1485_148591

theorem avg_divisible_by_4_between_15_and_55_eq_34 :
  let numbers := (List.filter (λ x => x % 4 = 0) (List.range' 16 37))
  (List.sum numbers) / (numbers.length) = 34 := by
  sorry

end avg_divisible_by_4_between_15_and_55_eq_34_l1485_148591


namespace other_root_of_quadratic_eq_l1485_148528

theorem other_root_of_quadratic_eq (m : ℝ) (q : ℝ) :
  (∃ x : ℝ, x ≠ q ∧ 3 * x^2 + m * x - 7 = 0) →
  (3 * q^2 + m * q - 7 = 0) →
  q = -7 / 3 :=
by
  intro h
  sorry

end other_root_of_quadratic_eq_l1485_148528


namespace clea_ride_escalator_time_l1485_148545

theorem clea_ride_escalator_time (x y k : ℝ) (h1 : 80 * x = y) (h2 : 30 * (x + k) = y) : (y / k) + 5 = 53 :=
by {
  sorry
}

end clea_ride_escalator_time_l1485_148545


namespace analogical_reasoning_correct_l1485_148552

variable (a b c : Real)

theorem analogical_reasoning_correct (h : c ≠ 0) (h_eq : (a + b) * c = a * c + b * c) : 
  (a + b) / c = a / c + b / c :=
  sorry

end analogical_reasoning_correct_l1485_148552


namespace gcd_lcm_product_l1485_148598

theorem gcd_lcm_product (a b : ℕ) (ha : a = 18) (hb : b = 42) :
  Nat.gcd a b * Nat.lcm a b = 756 :=
by
  rw [ha, hb]
  sorry

end gcd_lcm_product_l1485_148598


namespace largest_quantity_l1485_148559

noncomputable def D := (2007 / 2006) + (2007 / 2008)
noncomputable def E := (2007 / 2008) + (2009 / 2008)
noncomputable def F := (2008 / 2007) + (2008 / 2009)

theorem largest_quantity : D > E ∧ D > F :=
by { sorry }

end largest_quantity_l1485_148559


namespace solve_system_of_equations_l1485_148537

theorem solve_system_of_equations (m b : ℤ) 
  (h1 : 3 * m + b = 11)
  (h2 : -4 * m - b = 11) : 
  m = -22 ∧ b = 77 :=
  sorry

end solve_system_of_equations_l1485_148537


namespace same_color_probability_l1485_148547

theorem same_color_probability 
  (B R : ℕ)
  (hB : B = 5)
  (hR : R = 5)
  : (B + R = 10) → (1/2 * 4/9 + 1/2 * 4/9 = 4/9) := by
  intros
  sorry

end same_color_probability_l1485_148547


namespace union_M_N_equals_set_x_ge_1_l1485_148523

-- Definitions of M and N based on the conditions from step a)
def M : Set ℝ := { x | x - 2 > 0 }

def N : Set ℝ := { y | ∃ x : ℝ, y = Real.sqrt (x^2 + 1) }

-- Statement of the theorem
theorem union_M_N_equals_set_x_ge_1 : (M ∪ N) = { x : ℝ | x ≥ 1 } := 
sorry

end union_M_N_equals_set_x_ge_1_l1485_148523


namespace inequality_true_l1485_148512

theorem inequality_true (a b : ℝ) (hab : a < b) (hb : b < 0) (ha : a < 0) : (b / a) < 1 :=
by
  sorry

end inequality_true_l1485_148512


namespace domain_of_g_l1485_148504

-- Define the function f and specify the domain of f(x+1)
def f : ℝ → ℝ := sorry
def domain_f_x_plus_1 : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3} -- Domain of f(x+1) is [-1, 3]

-- Define the definition of the function g where g(x) = f(x^2)
def g (x : ℝ) : ℝ := f (x^2)

-- Prove that the domain of g(x) is [-2, 2]
theorem domain_of_g : {x | -2 ≤ x ∧ x ≤ 2} = {x | ∃ (y : ℝ), (0 ≤ y ∧ y ≤ 4) ∧ (x = y ∨ x = -y)} :=
by 
  sorry

end domain_of_g_l1485_148504


namespace complement_A_intersect_B_eq_l1485_148570

def setA : Set ℝ := { x : ℝ | |x - 2| ≤ 2 }

def setB : Set ℝ := { y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2 }

def A_intersect_B := setA ∩ setB

def complement (A : Set ℝ) : Set ℝ := { x : ℝ | x ∉ A }

theorem complement_A_intersect_B_eq {A : Set ℝ} {B : Set ℝ} 
  (hA : A = { x : ℝ | |x - 2| ≤ 2 })
  (hB : B = { y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2 }) :
  complement (A ∩ B) = { x : ℝ | x ≠ 0 } :=
by
  sorry

end complement_A_intersect_B_eq_l1485_148570


namespace grade_on_second_test_l1485_148526

variable (first_test_grade second_test_average : ℕ)
#check first_test_grade
#check second_test_average

theorem grade_on_second_test :
  first_test_grade = 78 →
  second_test_average = 81 →
  (first_test_grade + (second_test_average * 2 - first_test_grade)) / 2 = second_test_average →
  second_test_grade = 84 :=
by
  intros h1 h2 h3
  sorry

end grade_on_second_test_l1485_148526


namespace john_labor_cost_l1485_148542

def plank_per_tree : ℕ := 25
def table_cost : ℕ := 300
def profit : ℕ := 12000
def trees_chopped : ℕ := 30
def planks_per_table : ℕ := 15
def total_table_revenue := (trees_chopped * plank_per_tree / planks_per_table) * table_cost
def labor_cost := total_table_revenue - profit

theorem john_labor_cost :
  labor_cost = 3000 :=
by
  sorry

end john_labor_cost_l1485_148542


namespace true_inverse_of_opposites_true_contrapositive_of_real_roots_l1485_148581

theorem true_inverse_of_opposites (X Y : Int) :
  (X = -Y) → (X + Y = 0) :=
by 
  sorry

theorem true_contrapositive_of_real_roots (q : Real) :
  (¬ ∃ x : Real, x^2 + 2*x + q = 0) → (q > 1) :=
by
  sorry

end true_inverse_of_opposites_true_contrapositive_of_real_roots_l1485_148581


namespace third_cyclist_speed_l1485_148597

theorem third_cyclist_speed (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : 
  ∃ V : ℝ, V = (a + 3 * b + Real.sqrt (a^2 - 10 * a * b + 9 * b^2)) / 4 :=
by
  sorry

end third_cyclist_speed_l1485_148597


namespace necessarily_positive_l1485_148506

theorem necessarily_positive (x y z : ℝ) (hx : -1 < x ∧ x < 1) 
                      (hy : -1 < y ∧ y < 0) 
                      (hz : 1 < z ∧ z < 2) : 
    y + z > 0 := 
by
  sorry

end necessarily_positive_l1485_148506


namespace factorize_expression_l1485_148590

theorem factorize_expression (a b x y : ℝ) :
  9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) :=
by
  sorry

end factorize_expression_l1485_148590


namespace solutions_count_l1485_148587

noncomputable def number_of_solutions (a : ℝ) : ℕ :=
if a < 0 then 1
else if 0 ≤ a ∧ a < Real.exp 1 then 0
else if a = Real.exp 1 then 1
else if a > Real.exp 1 then 2
else 0

theorem solutions_count (a : ℝ) :
  (a < 0 ∧ number_of_solutions a = 1) ∨
  (0 ≤ a ∧ a < Real.exp 1 ∧ number_of_solutions a = 0) ∨
  (a = Real.exp 1 ∧ number_of_solutions a = 1) ∨
  (a > Real.exp 1 ∧ number_of_solutions a = 2) :=
by {
  sorry
}

end solutions_count_l1485_148587


namespace complex_fraction_eval_l1485_148585

theorem complex_fraction_eval (i : ℂ) (hi : i^2 = -1) : (3 + i) / (1 + i) = 2 - i := 
by 
  sorry

end complex_fraction_eval_l1485_148585


namespace a_plus_b_in_D_l1485_148566

def setA : Set ℤ := {x | ∃ k : ℤ, x = 4 * k}
def setB : Set ℤ := {x | ∃ m : ℤ, x = 4 * m + 1}
def setC : Set ℤ := {x | ∃ n : ℤ, x = 4 * n + 2}
def setD : Set ℤ := {x | ∃ t : ℤ, x = 4 * t + 3}

theorem a_plus_b_in_D (a b : ℤ) (ha : a ∈ setB) (hb : b ∈ setC) : a + b ∈ setD := by
  sorry

end a_plus_b_in_D_l1485_148566


namespace sum_of_three_numbers_l1485_148582

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a <= 10) (h2 : 10 <= c)
  (h3 : (a + 10 + c) / 3 = a + 8)
  (h4 : (a + 10 + c) / 3 = c - 20) :
  a + 10 + c = 66 :=
by
  sorry

end sum_of_three_numbers_l1485_148582


namespace floor_add_self_eq_14_5_iff_r_eq_7_5_l1485_148509

theorem floor_add_self_eq_14_5_iff_r_eq_7_5 (r : ℝ) : 
  (⌊r⌋ + r = 14.5) ↔ r = 7.5 :=
by
  sorry

end floor_add_self_eq_14_5_iff_r_eq_7_5_l1485_148509


namespace John_Anna_total_eBooks_l1485_148592

variables (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) 

def John_bought (Anna_bought : ℕ) : ℕ := Anna_bought - 15
def John_left (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) : ℕ := John_bought Anna_bought - eBooks_lost_by_John

theorem John_Anna_total_eBooks (Anna_bought_eq_50 : Anna_bought = 50)
    (John_bought_eq_35 : John_bought Anna_bought = 35) (eBooks_lost_eq_3 : eBooks_lost_by_John = 3) :
    (Anna_bought + John_left Anna_bought eBooks_lost_by_John = 82) :=
by sorry

end John_Anna_total_eBooks_l1485_148592


namespace proportion_first_number_l1485_148561

theorem proportion_first_number (x : ℝ) (h : x / 5 = 0.96 / 8) : x = 0.6 :=
by
  sorry

end proportion_first_number_l1485_148561


namespace nancy_history_books_l1485_148533

/-- Nancy started with 46 books in total on the cart.
    She shelved 8 romance books and 4 poetry books from the top section.
    She shelved 5 Western novels and 6 biographies from the bottom section.
    Half the books on the bottom section were mystery books.
    Prove that Nancy shelved 12 history books.
-/
theorem nancy_history_books 
  (total_books : ℕ)
  (romance_books : ℕ)
  (poetry_books : ℕ)
  (western_novels : ℕ)
  (biographies : ℕ)
  (bottom_books_half_mystery : ℕ)
  (history_books : ℕ) :
  (total_books = 46) →
  (romance_books = 8) →
  (poetry_books = 4) →
  (western_novels = 5) →
  (biographies = 6) →
  (bottom_books_half_mystery = 11) →
  (history_books = total_books - ((romance_books + poetry_books) + (2 * (western_novels + biographies)))) →
  history_books = 12 :=
by
  intros
  sorry

end nancy_history_books_l1485_148533


namespace updated_mean_of_observations_l1485_148500

theorem updated_mean_of_observations
    (number_of_observations : ℕ)
    (initial_mean : ℝ)
    (decrement_per_observation : ℝ)
    (h1 : number_of_observations = 50)
    (h2 : initial_mean = 200)
    (h3 : decrement_per_observation = 15) :
    (initial_mean * number_of_observations - decrement_per_observation * number_of_observations) / number_of_observations = 185 :=
by {
    sorry
}

end updated_mean_of_observations_l1485_148500


namespace book_distribution_methods_l1485_148502

theorem book_distribution_methods :
  let novels := 2
  let picture_books := 2
  let students := 3
  (number_ways : ℕ) = 12 :=
by
  sorry

end book_distribution_methods_l1485_148502


namespace min_score_needed_l1485_148519

/-- 
Given the list of scores and the targeted increase in the average score,
ascertain that the minimum score required on the next test to achieve the
new average is 110.
 -/
theorem min_score_needed 
  (scores : List ℝ) 
  (target_increase : ℝ) 
  (new_score : ℝ) 
  (total_scores : ℝ)
  (current_average : ℝ) 
  (target_average : ℝ) 
  (needed_score : ℝ) :
  (total_scores = 86 + 92 + 75 + 68 + 88 + 84) ∧
  (current_average = total_scores / 6) ∧
  (target_average = current_average + target_increase) ∧
  (new_score = total_scores + needed_score) ∧
  (target_average = new_score / 7) ->
  needed_score = 110 :=
by
  sorry

end min_score_needed_l1485_148519


namespace unique_solution_condition_l1485_148557

theorem unique_solution_condition (a b : ℝ) : (4 * x - 6 + a = (b + 1) * x + 2) → b ≠ 3 :=
by
  intro h
  -- Given the condition equation
  have eq1 : 4 * x - 6 + a = (b + 1) * x + 2 := h
  -- Simplify to the form (3 - b) * x = 8 - a
  sorry

end unique_solution_condition_l1485_148557


namespace cube_side_length_of_paint_cost_l1485_148529

theorem cube_side_length_of_paint_cost (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (total_cost : ℝ) (side_length : ℝ) :
  cost_per_kg = 20 ∧ coverage_per_kg = 15 ∧ total_cost = 200 →
  6 * side_length ^ 2 = (total_cost / cost_per_kg) * coverage_per_kg →
  side_length = 5 :=
by
  intros h1 h2
  sorry

end cube_side_length_of_paint_cost_l1485_148529


namespace tangent_slope_at_one_l1485_148551

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x + Real.sqrt x

theorem tangent_slope_at_one :
  (deriv f 1) = 3 / 2 :=
by
  sorry

end tangent_slope_at_one_l1485_148551


namespace kelly_spends_correct_amount_l1485_148595

noncomputable def total_cost_with_discount : ℝ :=
  let mango_cost_per_pound := (0.60 : ℝ) * 2
  let orange_cost_per_pound := (0.40 : ℝ) * 4
  let mango_total_cost := 5 * mango_cost_per_pound
  let orange_total_cost := 5 * orange_cost_per_pound
  let total_cost_without_discount := mango_total_cost + orange_total_cost
  let discount := 0.10 * total_cost_without_discount
  let total_cost_with_discount := total_cost_without_discount - discount
  total_cost_with_discount

theorem kelly_spends_correct_amount :
  total_cost_with_discount = 12.60 := by
  sorry

end kelly_spends_correct_amount_l1485_148595


namespace tips_fraction_l1485_148554

theorem tips_fraction (S T : ℝ) (h : T / (S + T) = 0.6363636363636364) : T / S = 1.75 :=
sorry

end tips_fraction_l1485_148554


namespace problem1_problem2_l1485_148568

variable (a : ℝ)

def quadratic_roots (a x : ℝ) : Prop := a*x^2 + 2*x + 1 = 0

-- Problem 1: If 1/2 is a root, find the set A
theorem problem1 (h : quadratic_roots a (1/2)) : 
  {x : ℝ | quadratic_roots (a) x } = { -1/4, 1/2 } :=
sorry

-- Problem 2: If A contains exactly one element, find the set B consisting of such a
theorem problem2 (h : ∃! (x : ℝ), quadratic_roots a x ) : 
  {a : ℝ | ∃! (x : ℝ), quadratic_roots a x} = { 0, 1 } :=
sorry

end problem1_problem2_l1485_148568


namespace solve_fractional_equation_l1485_148540

theorem solve_fractional_equation (x : ℚ) (h: x ≠ 1) : 
  (x / (x - 1) = 3 / (2 * x - 2) - 2) ↔ (x = 7 / 6) := 
by
  sorry

end solve_fractional_equation_l1485_148540


namespace sqrt_xyz_sum_l1485_148583

theorem sqrt_xyz_sum {x y z : ℝ} (h₁ : y + z = 24) (h₂ : z + x = 26) (h₃ : x + y = 28) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 83655 := by
  sorry

end sqrt_xyz_sum_l1485_148583


namespace common_fraction_difference_l1485_148544

def repeating_decimal := 23 / 99
def non_repeating_decimal := 23 / 100
def fraction_difference := 23 / 9900

theorem common_fraction_difference : repeating_decimal - non_repeating_decimal = fraction_difference := 
by
  sorry

end common_fraction_difference_l1485_148544


namespace eight_digit_increasing_numbers_mod_1000_l1485_148536

theorem eight_digit_increasing_numbers_mod_1000 : 
  ((Nat.choose 17 8) % 1000) = 310 := 
by 
  sorry -- Proof not required as per instructions

end eight_digit_increasing_numbers_mod_1000_l1485_148536


namespace hearts_per_card_l1485_148596

-- Definitions of the given conditions
def num_suits := 4
def num_cards_total := 52
def num_cards_per_suit := num_cards_total / num_suits
def cost_per_cow := 200
def total_cost := 83200
def num_cows := total_cost / cost_per_cow

-- The mathematical proof problem translated to Lean 4:
theorem hearts_per_card :
    (2 * (num_cards_total / num_suits) = num_cows) → (num_cows = 416) → (num_cards_total / num_suits = 208) :=
by
  intros h1 h2
  sorry

end hearts_per_card_l1485_148596


namespace tank_fill_time_l1485_148558

-- Define the conditions
def capacity := 800
def rate_A := 40
def rate_B := 30
def rate_C := -20

def net_rate_per_cycle := rate_A + rate_B + rate_C
def cycle_duration := 3
def total_cycles := capacity / net_rate_per_cycle
def total_time := total_cycles * cycle_duration

-- The proof that tank will be full after 48 minutes
theorem tank_fill_time : total_time = 48 := by
  sorry

end tank_fill_time_l1485_148558


namespace samantha_routes_l1485_148518

-- Define the positions relative to the grid
structure Position where
  x : Int
  y : Int

-- Define the initial conditions and path constraints
def house : Position := ⟨-3, -2⟩
def sw_corner_of_park : Position := ⟨0, 0⟩
def ne_corner_of_park : Position := ⟨8, 5⟩
def school : Position := ⟨11, 8⟩

-- Define the combinatorial function for calculating number of ways
def binom (n k : Nat) : Nat := Nat.choose n k

-- Route segments based on the constraints
def ways_house_to_sw_corner : Nat := binom 5 2
def ways_through_park : Nat := 1
def ways_ne_corner_to_school : Nat := binom 6 3

-- Total number of routes
def total_routes : Nat := ways_house_to_sw_corner * ways_through_park * ways_ne_corner_to_school

-- The statement to be proven
theorem samantha_routes : total_routes = 200 := by
  sorry

end samantha_routes_l1485_148518


namespace apples_left_l1485_148534

theorem apples_left (initial_apples : ℕ) (ricki_removes : ℕ) (samson_removes : ℕ) 
  (h1 : initial_apples = 74) 
  (h2 : ricki_removes = 14) 
  (h3 : samson_removes = 2 * ricki_removes) : 
  initial_apples - (ricki_removes + samson_removes) = 32 := 
by
  sorry

end apples_left_l1485_148534


namespace john_money_left_l1485_148543

def cost_of_drink (q : ℝ) : ℝ := q
def cost_of_small_pizza (q : ℝ) : ℝ := cost_of_drink q
def cost_of_large_pizza (q : ℝ) : ℝ := 4 * cost_of_drink q
def total_cost (q : ℝ) : ℝ := 2 * cost_of_drink q + 2 * cost_of_small_pizza q + cost_of_large_pizza q
def initial_money : ℝ := 50
def remaining_money (q : ℝ) : ℝ := initial_money - total_cost q

theorem john_money_left (q : ℝ) : remaining_money q = 50 - 8 * q :=
by
  sorry

end john_money_left_l1485_148543


namespace find_sum_abc_l1485_148564

-- Define the real numbers a, b, c
variables {a b c : ℝ}

-- Define the conditions that a, b, c are positive reals.
axiom ha_pos : 0 < a
axiom hb_pos : 0 < b
axiom hc_pos : 0 < c

-- Define the condition that a^2 + b^2 + c^2 = 989
axiom habc_sq : a^2 + b^2 + c^2 = 989

-- Define the condition that (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013
axiom habc_sq_sum : (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013

-- The proposition to be proven
theorem find_sum_abc : a + b + c = 32 :=
by
  -- ...(proof goes here)
  sorry

end find_sum_abc_l1485_148564


namespace distance_on_dirt_section_distance_on_muddy_section_l1485_148521

section RaceProblem

variables {v_h v_d v_m : ℕ} (initial_gap : ℕ)

-- Problem conditions
def highway_speed := 150 -- km/h
def dirt_road_speed := 60 -- km/h
def muddy_section_speed := 18 -- km/h
def initial_gap_start := 300 -- meters

-- Convert km/h to m/s
def to_m_per_s (speed : ℕ) : ℕ := (speed * 1000) / 3600

-- Speeds in m/s
def highway_speed_mps := to_m_per_s highway_speed
def dirt_road_speed_mps := to_m_per_s dirt_road_speed
def muddy_section_speed_mps := to_m_per_s muddy_section_speed

-- Questions
theorem distance_on_dirt_section :
  ∃ (d : ℕ), (d = 120) :=
sorry

theorem distance_on_muddy_section :
  ∃ (d : ℕ), (d = 36) :=
sorry

end RaceProblem

end distance_on_dirt_section_distance_on_muddy_section_l1485_148521


namespace find_principal_l1485_148525

theorem find_principal
  (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ)
  (hA : A = 896)
  (hr : r = 0.05)
  (ht : t = 12 / 5) :
  P = 800 ↔ A = P * (1 + r * t) :=
by {
  sorry
}

end find_principal_l1485_148525


namespace max_days_for_C_l1485_148569

-- Define the durations of the processes and the total project duration
def A := 2
def B := 5
def D := 4
def T := 9

-- Define the condition to prove the maximum days required for process C
theorem max_days_for_C (x : ℕ) (h : 2 + x + 4 = 9) : x = 3 := by
  sorry

end max_days_for_C_l1485_148569


namespace lars_bakes_for_six_hours_l1485_148577

variable (h : ℕ)

-- Conditions
def bakes_loaves : ℕ := 10 * h
def bakes_baguettes : ℕ := 15 * h
def total_breads : ℕ := bakes_loaves h + bakes_baguettes h

-- Proof goal
theorem lars_bakes_for_six_hours (h : ℕ) (H : total_breads h = 150) : h = 6 :=
sorry

end lars_bakes_for_six_hours_l1485_148577


namespace circle_radius_l1485_148527

theorem circle_radius (D : ℝ) (h : D = 14) : (D / 2) = 7 :=
by
  sorry

end circle_radius_l1485_148527


namespace ratio_of_investments_l1485_148560

variable (A B C : ℝ) (k : ℝ)

-- Conditions
def investments_ratio := (6 * k + 5 * k + 4 * k = 7250) ∧ (5 * k - 6 * k = 250)

-- Theorem we need to prove
theorem ratio_of_investments (h : investments_ratio k) : (A / B = 6 / 5) ∧ (B / C = 5 / 4) := 
  sorry

end ratio_of_investments_l1485_148560


namespace right_triangle_lengths_l1485_148531

theorem right_triangle_lengths (a b c : ℝ) (h1 : c + b = 2 * a) (h2 : c^2 = a^2 + b^2) : 
  b = 3 / 4 * a ∧ c = 5 / 4 * a := 
by
  sorry

end right_triangle_lengths_l1485_148531


namespace unique_k_for_equal_power_l1485_148511

theorem unique_k_for_equal_power (k : ℕ) (hk : 0 < k) (h : ∃ m n : ℕ, n > 1 ∧ (3 ^ k + 5 ^ k = m ^ n)) : k = 1 :=
by
  sorry

end unique_k_for_equal_power_l1485_148511


namespace solution_exists_unique_l1485_148556

variable (a b c : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)

theorem solution_exists_unique (x y z : ℝ)
  (hx : x = (b + c) / 2)
  (hy : y = (c + a) / 2)
  (hz : z = (a + b) / 2)
  (h1 : x + y + z = a + b + c)
  (h2 : 4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c) :
  x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2 :=
by
  sorry

end solution_exists_unique_l1485_148556


namespace logs_left_after_3_hours_l1485_148541

theorem logs_left_after_3_hours : 
  ∀ (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (time : ℕ),
  initial_logs = 6 →
  burn_rate = 3 →
  add_rate = 2 →
  time = 3 →
  initial_logs + (add_rate * time) - (burn_rate * time) = 3 := 
by
  intros initial_logs burn_rate add_rate time h1 h2 h3 h4
  sorry

end logs_left_after_3_hours_l1485_148541


namespace minimum_meals_needed_l1485_148524

theorem minimum_meals_needed (total_jam : ℝ) (max_per_meal : ℝ) (jars : ℕ) (max_jar_weight : ℝ):
  (total_jam = 50) → (max_per_meal = 5) → (jars ≥ 50) → (max_jar_weight ≤ 1) →
  (jars * max_jar_weight = total_jam) →
  jars ≥ 12 := sorry

end minimum_meals_needed_l1485_148524


namespace find_original_wage_l1485_148563

theorem find_original_wage (W : ℝ) (h : 1.50 * W = 51) : W = 34 :=
sorry

end find_original_wage_l1485_148563


namespace radius_inner_circle_l1485_148576

theorem radius_inner_circle (s : ℝ) (n : ℕ) (d : ℝ) (r : ℝ) :
  s = 4 ∧ n = 16 ∧ d = s / 4 ∧ ∀ k, k = d / 2 → r = (Real.sqrt (s^2 / 4 + k^2) - k) / 2 
  → r = Real.sqrt 4.25 / 2 :=
by
  sorry

end radius_inner_circle_l1485_148576


namespace find_num_officers_l1485_148517

noncomputable def num_officers (O : ℕ) : Prop :=
  let avg_salary_all := 120
  let avg_salary_officers := 440
  let avg_salary_non_officers := 110
  let num_non_officers := 480
  let total_salary :=
    avg_salary_all * (O + num_non_officers)
  let salary_officers :=
    avg_salary_officers * O
  let salary_non_officers :=
    avg_salary_non_officers * num_non_officers
  total_salary = salary_officers + salary_non_officers

theorem find_num_officers : num_officers 15 :=
sorry

end find_num_officers_l1485_148517


namespace annual_interest_payment_l1485_148501

noncomputable def principal : ℝ := 9000
noncomputable def rate : ℝ := 9 / 100
noncomputable def time : ℝ := 1
noncomputable def interest : ℝ := principal * rate * time

theorem annual_interest_payment : interest = 810 := by
  sorry

end annual_interest_payment_l1485_148501


namespace juliet_older_than_maggie_l1485_148580

-- Definitions from the given conditions
def Juliet_age : ℕ := 10
def Ralph_age (J : ℕ) : ℕ := J + 2
def Maggie_age (R : ℕ) : ℕ := 19 - R

-- Theorem statement
theorem juliet_older_than_maggie :
  Juliet_age - Maggie_age (Ralph_age Juliet_age) = 3 :=
by
  sorry

end juliet_older_than_maggie_l1485_148580


namespace alley_width_l1485_148510

noncomputable def calculate_width (l k h : ℝ) : ℝ :=
  l / 2

theorem alley_width (k h l w : ℝ) (h1 : k = (l * (Real.sin (Real.pi / 3)))) (h2 : h = (l * (Real.sin (Real.pi / 6)))) :
  w = calculate_width l k h :=
by
  sorry

end alley_width_l1485_148510


namespace probability_different_colors_l1485_148565

def total_chips := 7 + 5 + 4

def probability_blue_draw : ℚ := 7 / total_chips
def probability_red_draw : ℚ := 5 / total_chips
def probability_yellow_draw : ℚ := 4 / total_chips
def probability_different_color (color1_prob color2_prob : ℚ) : ℚ := color1_prob * (1 - color2_prob)

theorem probability_different_colors :
  (probability_blue_draw * probability_different_color 7 (7 / total_chips)) +
  (probability_red_draw * probability_different_color 5 (5 / total_chips)) +
  (probability_yellow_draw * probability_different_color 4 (4 / total_chips)) 
  = 83 / 128 := 
by 
  sorry

end probability_different_colors_l1485_148565


namespace exp_mono_increasing_of_gt_l1485_148573

variable {a b : ℝ}

theorem exp_mono_increasing_of_gt (h : a > b) : (2 : ℝ) ^ a > (2 : ℝ) ^ b :=
by sorry

end exp_mono_increasing_of_gt_l1485_148573


namespace find_n_from_ratio_l1485_148507

theorem find_n_from_ratio (a b n : ℕ) (h : (a + 3 * b) ^ n = 4 ^ n)
  (h_ratio : 4 ^ n / 2 ^ n = 64) : 
  n = 6 := 
by
  sorry

end find_n_from_ratio_l1485_148507


namespace total_tissues_brought_l1485_148538

def number_students_group1 : Nat := 9
def number_students_group2 : Nat := 10
def number_students_group3 : Nat := 11
def tissues_per_box : Nat := 40

theorem total_tissues_brought : 
  (number_students_group1 + number_students_group2 + number_students_group3) * tissues_per_box = 1200 := 
by 
  sorry

end total_tissues_brought_l1485_148538


namespace max_value_of_f_l1485_148522

noncomputable def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x - 1/2

theorem max_value_of_f : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ (∀ y, (0 ≤ y ∧ y ≤ 2) → f y ≤ f x) ∧ f x = -3 :=
by
  sorry

end max_value_of_f_l1485_148522


namespace value_of_expression_l1485_148555

variable (x y : ℝ)

theorem value_of_expression 
  (h1 : x + Real.sqrt (x * y) + y = 9)
  (h2 : x^2 + x * y + y^2 = 27) :
  x - Real.sqrt (x * y) + y = 3 :=
sorry

end value_of_expression_l1485_148555


namespace time_after_12345_seconds_is_13_45_45_l1485_148508

def seconds_in_a_minute := 60
def minutes_in_an_hour := 60
def initial_hour := 10
def initial_minute := 45
def initial_second := 0
def total_seconds := 12345

def time_after_seconds (hour minute second : Nat) (elapsed_seconds : Nat) : (Nat × Nat × Nat) :=
  let total_initial_seconds := hour * 3600 + minute * 60 + second
  let total_final_seconds := total_initial_seconds + elapsed_seconds
  let final_hour := total_final_seconds / 3600
  let remaining_seconds_after_hour := total_final_seconds % 3600
  let final_minute := remaining_seconds_after_hour / 60
  let final_second := remaining_seconds_after_hour % 60
  (final_hour, final_minute, final_second)

theorem time_after_12345_seconds_is_13_45_45 :
  time_after_seconds initial_hour initial_minute initial_second total_seconds = (13, 45, 45) :=
by
  sorry

end time_after_12345_seconds_is_13_45_45_l1485_148508


namespace radius_of_cone_l1485_148520

theorem radius_of_cone (A : ℝ) (g : ℝ) (R : ℝ) (hA : A = 15 * Real.pi) (hg : g = 5) : R = 3 :=
sorry

end radius_of_cone_l1485_148520


namespace at_least_one_zero_l1485_148571

theorem at_least_one_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) : 
  a = 0 ∨ b = 0 ∨ c = 0 := 
sorry

end at_least_one_zero_l1485_148571


namespace range_of_a_l1485_148535

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 + 2*x + a ≥ 0) ↔ a ≥ -8 := 
sorry

end range_of_a_l1485_148535


namespace toy_store_shelves_l1485_148575

theorem toy_store_shelves (initial_bears : ℕ) (shipment_bears : ℕ) (bears_per_shelf : ℕ)
                          (h_initial : initial_bears = 5) (h_shipment : shipment_bears = 7) 
                          (h_per_shelf : bears_per_shelf = 6) : 
                          (initial_bears + shipment_bears) / bears_per_shelf = 2 :=
by
  sorry

end toy_store_shelves_l1485_148575


namespace find_k_l1485_148532

open Real

variables (a b : ℝ × ℝ) (k : ℝ)

def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

theorem find_k (ha : a = (1, 2)) (hb : b = (-2, 4)) (perpendicular : dot_product (k • a + b) b = 0) :
  k = - (10 / 3) :=
by
  sorry

end find_k_l1485_148532


namespace squareInPentagon_l1485_148513

-- Definitions pertinent to the problem
structure Pentagon (α : Type) [AddCommGroup α] :=
(A B C D E : α) 

def isRegularPentagon {α : Type} [AddCommGroup α] [LinearOrder α] (P : Pentagon α) : Prop :=
  -- Conditions for a regular pentagon (typically involving equal side lengths and equal angles)
  sorry

def inscribedSquareExists {α : Type} [AddCommGroup α] (P : Pentagon α) : Prop :=
  -- There exists a square inscribed in the pentagon P with vertices on four different sides
  sorry

-- The main theorem to state the proof problem
theorem squareInPentagon {α : Type} [AddCommGroup α] [LinearOrder α] (P : Pentagon α)
  (hP : isRegularPentagon P) : inscribedSquareExists P :=
sorry

end squareInPentagon_l1485_148513


namespace smallest_lcm_of_4digit_integers_with_gcd_5_l1485_148572

theorem smallest_lcm_of_4digit_integers_with_gcd_5 :
  ∃ (a b : ℕ), 1000 ≤ a ∧ a < 10000 ∧ 1000 ≤ b ∧ b < 10000 ∧ gcd a b = 5 ∧ lcm a b = 201000 :=
by
  sorry

end smallest_lcm_of_4digit_integers_with_gcd_5_l1485_148572


namespace expansion_terms_count_l1485_148549

-- Define the number of terms in the first polynomial
def first_polynomial_terms : ℕ := 3

-- Define the number of terms in the second polynomial
def second_polynomial_terms : ℕ := 4

-- Prove that the number of terms in the expansion is 12
theorem expansion_terms_count : first_polynomial_terms * second_polynomial_terms = 12 :=
by
  sorry

end expansion_terms_count_l1485_148549


namespace stephanie_bills_l1485_148503

theorem stephanie_bills :
  let electricity_bill := 120
  let electricity_paid := 0.80 * electricity_bill
  let gas_bill := 80
  let gas_paid := (3 / 4) * gas_bill
  let additional_gas_payment := 10
  let water_bill := 60
  let water_paid := 0.65 * water_bill
  let internet_bill := 50
  let internet_paid := 6 * 5
  let internet_remaining_before_discount := internet_bill - internet_paid
  let internet_discount := 0.10 * internet_remaining_before_discount
  let phone_bill := 45
  let phone_paid := 0.20 * phone_bill
  let remaining_electricity := electricity_bill - electricity_paid
  let remaining_gas := gas_bill - (gas_paid + additional_gas_payment)
  let remaining_water := water_bill - water_paid
  let remaining_internet := internet_remaining_before_discount - internet_discount
  let remaining_phone := phone_bill - phone_paid
  (remaining_electricity + remaining_gas + remaining_water + remaining_internet + remaining_phone) = 109 :=
by
  sorry

end stephanie_bills_l1485_148503


namespace population_Lake_Bright_l1485_148594

-- Definition of total population
def T := 80000

-- Definition of population of Gordonia
def G := (1 / 2) * T

-- Definition of population of Toadon
def Td := (60 / 100) * G

-- Proof that the population of Lake Bright is 16000
theorem population_Lake_Bright : T - (G + Td) = 16000 :=
by {
    -- Leaving the proof as sorry
    sorry
}

end population_Lake_Bright_l1485_148594


namespace max_consecutive_integers_sum_lt_1000_l1485_148593

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end max_consecutive_integers_sum_lt_1000_l1485_148593


namespace hyogeun_weight_l1485_148574

noncomputable def weights_are_correct : Prop :=
  ∃ H S G : ℝ, 
    H + S + G = 106.6 ∧
    G = S - 7.7 ∧
    S = H - 4.8 ∧
    H = 41.3

theorem hyogeun_weight : weights_are_correct :=
by
  sorry

end hyogeun_weight_l1485_148574


namespace solve_for_x_opposites_l1485_148584

theorem solve_for_x_opposites (x : ℝ) (h : -2 * x = -(3 * x - 1)) : x = 1 :=
by {
  sorry
}

end solve_for_x_opposites_l1485_148584


namespace find_R_l1485_148562

theorem find_R (R : ℝ) (h_diff : ∃ a b : ℝ, a ≠ b ∧ (a - b = 12 ∨ b - a = 12) ∧ a + b = 2 ∧ a * b = -R) : R = 35 :=
by
  obtain ⟨a, b, h_neq, h_diff_12, h_sum, h_prod⟩ := h_diff
  sorry

end find_R_l1485_148562
