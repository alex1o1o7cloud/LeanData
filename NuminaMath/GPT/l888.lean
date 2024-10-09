import Mathlib

namespace remainder_when_summed_divided_by_15_l888_88845

theorem remainder_when_summed_divided_by_15 (k j : ℤ) (x y : ℤ)
  (hx : x = 60 * k + 47)
  (hy : y = 45 * j + 26) :
  (x + y) % 15 = 13 := 
sorry

end remainder_when_summed_divided_by_15_l888_88845


namespace find_third_number_l888_88821

-- Define the conditions
def equation1_valid : Prop := (5 * 3 = 15) ∧ (5 * 2 = 10) ∧ (2 * 1000 + 3 * 100 + 5 = 1022)
def equation2_valid : Prop := (9 * 2 = 18) ∧ (9 * 4 = 36) ∧ (4 * 1000 + 2 * 100 + 9 = 3652)

-- The theorem to prove
theorem find_third_number (h1 : equation1_valid) (h2 : equation2_valid) : (7 * 2 = 14) ∧ (7 * 5 = 35) ∧ (5 * 1000 + 2 * 100 + 7 = 547) :=
by 
  sorry

end find_third_number_l888_88821


namespace joey_return_speed_l888_88887

theorem joey_return_speed
    (h1: 1 = (2 : ℝ) / u)
    (h2: (4 : ℝ) / (1 + t) = 3)
    (h3: u = 2)
    (h4: t = 1 / 3) :
    (2 : ℝ) / t = 6 :=
by
  sorry

end joey_return_speed_l888_88887


namespace find_number_l888_88857

theorem find_number (x : ℝ) (h₁ : 0.40 * x = 130 + 190) : x = 800 :=
sorry

end find_number_l888_88857


namespace program_output_is_1023_l888_88874

-- Definition placeholder for program output.
def program_output : ℕ := 1023

-- Theorem stating the program's output.
theorem program_output_is_1023 : program_output = 1023 := 
by 
  -- Proof details are omitted.
  sorry

end program_output_is_1023_l888_88874


namespace line_intersects_circle_l888_88882

theorem line_intersects_circle (a : ℝ) :
  ∃ (x y : ℝ), (y = a * x + 1) ∧ ((x - 1) ^ 2 + y ^ 2 = 4) :=
by
  sorry

end line_intersects_circle_l888_88882


namespace work_done_by_6_men_and_11_women_l888_88807

-- Definitions based on conditions
def work_completed_by_men (men : ℕ) (days : ℕ) : ℚ := men / (8 * days)
def work_completed_by_women (women : ℕ) (days : ℕ) : ℚ := women / (12 * days)
def combined_work_rate (men : ℕ) (women : ℕ) (days : ℕ) : ℚ := 
  work_completed_by_men men days + work_completed_by_women women days

-- Problem statement
theorem work_done_by_6_men_and_11_women :
  combined_work_rate 6 11 12 = 1 := by
  sorry

end work_done_by_6_men_and_11_women_l888_88807


namespace polynomial_abc_value_l888_88883

theorem polynomial_abc_value (a b c : ℝ) (h : a * (x^2) + b * x + c = (x - 1) * (x - 2)) : a * b * c = -6 :=
by
  sorry

end polynomial_abc_value_l888_88883


namespace technicians_count_l888_88802

theorem technicians_count (avg_all : ℕ) (avg_tech : ℕ) (avg_other : ℕ) (total_workers : ℕ)
  (h1 : avg_all = 750) (h2 : avg_tech = 900) (h3 : avg_other = 700) (h4 : total_workers = 20) :
  ∃ T O : ℕ, (T + O = total_workers) ∧ ((T * avg_tech + O * avg_other) = total_workers * avg_all) ∧ (T = 5) :=
by
  sorry

end technicians_count_l888_88802


namespace tiles_difference_between_tenth_and_eleventh_square_l888_88872

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ :=
  3 + 2 * (n - 1)

-- Define the area of the nth square
def area (n : ℕ) : ℕ :=
  (side_length n) ^ 2

-- The math proof statement
theorem tiles_difference_between_tenth_and_eleventh_square : area 11 - area 10 = 88 :=
by 
  -- Proof goes here, but we use sorry to skip it for now
  sorry

end tiles_difference_between_tenth_and_eleventh_square_l888_88872


namespace smallest_number_first_digit_is_9_l888_88869

def sum_of_digits (n : Nat) : Nat :=
  (n.digits 10).sum

def first_digit (n : Nat) : Nat :=
  n.digits 10 |>.headD 0

theorem smallest_number_first_digit_is_9 :
  ∃ N : Nat, sum_of_digits N = 2020 ∧ ∀ M : Nat, (sum_of_digits M = 2020 → N ≤ M) ∧ first_digit N = 9 :=
by
  sorry

end smallest_number_first_digit_is_9_l888_88869


namespace dollars_sum_l888_88822

theorem dollars_sum : 
  (5 / 8 : ℝ) + (2 / 5) = 1.025 :=
by
  sorry

end dollars_sum_l888_88822


namespace triangle_inequalities_l888_88811

theorem triangle_inequalities (a b c : ℝ) (h : a < b + c) : b < a + c ∧ c < a + b := 
  sorry

end triangle_inequalities_l888_88811


namespace find_x_given_total_area_l888_88848

theorem find_x_given_total_area :
  ∃ x : ℝ, (16 * x^2 + 36 * x^2 + 6 * x^2 + 3 * x^2 = 1100) ∧ (x = Real.sqrt (1100 / 61)) :=
sorry

end find_x_given_total_area_l888_88848


namespace alice_preferred_numbers_l888_88861

def is_multiple_of_7 (n : ℕ) : Prop :=
  n % 7 = 0

def is_not_multiple_of_3 (n : ℕ) : Prop :=
  ¬ (n % 3 = 0)

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def alice_pref_num (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧ is_multiple_of_7 n ∧ is_not_multiple_of_3 n ∧ is_prime (digit_sum n)

theorem alice_preferred_numbers :
  ∀ n, alice_pref_num n ↔ n = 119 ∨ n = 133 ∨ n = 140 := 
sorry

end alice_preferred_numbers_l888_88861


namespace value_of_f_1985_l888_88879

def f : ℝ → ℝ := sorry -- Assuming the existence of f, let ℝ be the type of real numbers

-- Given condition as a hypothesis
axiom functional_eq (x y : ℝ) : f (x + y) = f (x^2) + f (2 * y)

-- The main theorem we want to prove
theorem value_of_f_1985 : f 1985 = 0 :=
by
  sorry

end value_of_f_1985_l888_88879


namespace vectors_parallel_x_value_l888_88846

theorem vectors_parallel_x_value :
  ∀ (x : ℝ), (∀ a b : ℝ × ℝ, a = (2, 1) → b = (4, x+1) → (a.1 / b.1 = a.2 / b.2)) → x = 1 :=
by
  intros x h
  sorry

end vectors_parallel_x_value_l888_88846


namespace dodecahedron_has_150_interior_diagonals_l888_88852

def dodecahedron_diagonals (vertices : ℕ) (adjacent : ℕ) : ℕ :=
  let total := vertices * (vertices - adjacent - 1) / 2
  total

theorem dodecahedron_has_150_interior_diagonals :
  dodecahedron_diagonals 20 4 = 150 :=
by
  sorry

end dodecahedron_has_150_interior_diagonals_l888_88852


namespace total_transaction_loss_l888_88876

-- Define the cost and selling prices given the conditions
def cost_price_house (h : ℝ) := (7 / 10) * h = 15000
def cost_price_store (s : ℝ) := (5 / 4) * s = 15000

-- Define the loss calculation for the transaction
def transaction_loss : Prop :=
  ∃ (h s : ℝ),
    (7 / 10) * h = 15000 ∧
    (5 / 4) * s = 15000 ∧
    h + s - 2 * 15000 = 3428.57

-- The theorem stating the transaction resulted in a loss of $3428.57
theorem total_transaction_loss : transaction_loss :=
by
  sorry

end total_transaction_loss_l888_88876


namespace smallest_n_not_divisible_by_10_l888_88817

theorem smallest_n_not_divisible_by_10 :
  ∃ n > 2016, (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := 
sorry

end smallest_n_not_divisible_by_10_l888_88817


namespace sequence_problem_l888_88888

theorem sequence_problem (a : ℕ → ℝ) (pos_terms : ∀ n, a n > 0)
  (h1 : a 1 = 2)
  (recurrence : ∀ n, (a n + 1) * a (n + 2) = 1)
  (h2 : a 2 = a 6) :
  a 11 + a 12 = (11 / 18) + ((Real.sqrt 5 - 1) / 2) := by
  sorry

end sequence_problem_l888_88888


namespace bridget_gave_erasers_l888_88820

variable (p_start : ℕ) (p_end : ℕ) (e_b : ℕ)

theorem bridget_gave_erasers (h1 : p_start = 8) (h2 : p_end = 11) (h3 : p_end = p_start + e_b) :
  e_b = 3 := by
  sorry

end bridget_gave_erasers_l888_88820


namespace ellipse_foci_condition_l888_88810

theorem ellipse_foci_condition {m : ℝ} :
  (1 < m ∧ m < 2) ↔ (∃ (x y : ℝ), (x^2 / (m - 1) + y^2 / (3 - m) = 1) ∧ (3 - m > m - 1) ∧ (m - 1 > 0) ∧ (3 - m > 0)) :=
by
  sorry

end ellipse_foci_condition_l888_88810


namespace factorize_xy2_minus_x_l888_88842

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l888_88842


namespace total_pears_l888_88836

def jason_pears : Nat := 46
def keith_pears : Nat := 47
def mike_pears : Nat := 12

theorem total_pears : jason_pears + keith_pears + mike_pears = 105 := by
  sorry

end total_pears_l888_88836


namespace price_per_gallon_in_NC_l888_88890

variable (P : ℝ)
variable (price_nc := P) -- price per gallon in North Carolina
variable (price_va := P + 1) -- price per gallon in Virginia
variable (gallons_nc := 10) -- gallons bought in North Carolina
variable (gallons_va := 10) -- gallons bought in Virginia
variable (total_cost := 50) -- total amount spent on gas

theorem price_per_gallon_in_NC :
  (gallons_nc * price_nc) + (gallons_va * price_va) = total_cost → price_nc = 2 :=
by
  sorry

end price_per_gallon_in_NC_l888_88890


namespace inequality_among_three_vars_l888_88837

theorem inequality_among_three_vars 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h : x + y + z ≥ 3) : 
  (
    1 / (x + y + z ^ 2) + 
    1 / (y + z + x ^ 2) + 
    1 / (z + x + y ^ 2) 
  ) ≤ 1 := 
  sorry

end inequality_among_three_vars_l888_88837


namespace roots_polynomial_expression_l888_88818

theorem roots_polynomial_expression (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a * b + a * c + b * c = -1)
  (h3 : a * b * c = -2) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = 0 :=
by
  sorry

end roots_polynomial_expression_l888_88818


namespace positive_difference_largest_prime_factors_l888_88896

theorem positive_difference_largest_prime_factors :
  let p1 := 139
  let p2 := 29
  p1 - p2 = 110 := sorry

end positive_difference_largest_prime_factors_l888_88896


namespace average_weight_of_remaining_boys_l888_88854

theorem average_weight_of_remaining_boys :
  ∀ (total_boys remaining_boys_num : ℕ)
    (avg_weight_22 remaining_boys_avg_weight total_class_avg_weight : ℚ),
    total_boys = 30 →
    remaining_boys_num = total_boys - 22 →
    avg_weight_22 = 50.25 →
    total_class_avg_weight = 48.89 →
    (remaining_boys_num : ℚ) * remaining_boys_avg_weight =
    total_boys * total_class_avg_weight - 22 * avg_weight_22 →
    remaining_boys_avg_weight = 45.15 :=
by
  intros total_boys remaining_boys_num avg_weight_22 remaining_boys_avg_weight total_class_avg_weight
         h_total_boys h_remaining_boys_num h_avg_weight_22 h_total_class_avg_weight h_equation
  sorry

end average_weight_of_remaining_boys_l888_88854


namespace max_profit_l888_88815

noncomputable def profit (x : ℝ) : ℝ :=
  10 * (x - 40) * (100 - x)

theorem max_profit (x : ℝ) (hx : x > 40) :
  (profit 70 = 9000) ∧ ∀ y > 40, profit y ≤ 9000 := by
  sorry

end max_profit_l888_88815


namespace no_solutions_abs_eq_3x_plus_6_l888_88853

theorem no_solutions_abs_eq_3x_plus_6 : ¬ ∃ x : ℝ, |x| = 3 * (|x| + 2) :=
by {
  sorry
}

end no_solutions_abs_eq_3x_plus_6_l888_88853


namespace interval_of_x_l888_88801

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l888_88801


namespace second_part_of_sum_l888_88856

-- Defining the problem conditions
variables (x : ℚ)
def sum_parts := (2 * x) + (1/2 * x) + (1/4 * x)

theorem second_part_of_sum :
  sum_parts x = 104 →
  (1/2 * x) = 208 / 11 :=
by
  intro h
  sorry

end second_part_of_sum_l888_88856


namespace number_of_chinese_l888_88831

theorem number_of_chinese (total americans australians chinese : ℕ) 
    (h_total : total = 49)
    (h_americans : americans = 16)
    (h_australians : australians = 11)
    (h_chinese : chinese = total - americans - australians) :
    chinese = 22 :=
by
    rw [h_total, h_americans, h_australians] at h_chinese
    exact h_chinese

end number_of_chinese_l888_88831


namespace mixed_number_multiplication_l888_88824

def mixed_to_improper (a : Int) (b : Int) (c : Int) : Rat :=
  a + (b / c)

theorem mixed_number_multiplication : 
  let a := 5
  let b := mixed_to_improper 7 2 5
  a * b = (37 : Rat) :=
by
  intros
  sorry

end mixed_number_multiplication_l888_88824


namespace distance_to_left_focus_l888_88804

theorem distance_to_left_focus (P : ℝ × ℝ) 
  (h1 : P.1^2 / 100 + P.2^2 / 36 = 1) 
  (h2 : dist P (50 - 100 / 9, P.2) = 17 / 2) :
  dist P (-50 - 100 / 9, P.2) = 66 / 5 :=
sorry

end distance_to_left_focus_l888_88804


namespace difference_of_two_numbers_l888_88834

theorem difference_of_two_numbers
  (L : ℕ) (S : ℕ) 
  (hL : L = 1596) 
  (hS : 6 * S + 15 = 1596) : 
  L - S = 1333 := 
by
  sorry

end difference_of_two_numbers_l888_88834


namespace squirrel_pine_cones_l888_88865

theorem squirrel_pine_cones (x y : ℕ) (hx : 26 - 10 + 9 + (x + 14)/2 = x/2) (hy : y + 5 - 18 + 9 + (x + 14)/2 = x/2) :
  x = 86 := sorry

end squirrel_pine_cones_l888_88865


namespace value_of_a_minus_b_l888_88859

theorem value_of_a_minus_b (a b : ℤ) (h1 : 2020 * a + 2024 * b = 2040) (h2 : 2022 * a + 2026 * b = 2044) :
  a - b = 1002 :=
sorry

end value_of_a_minus_b_l888_88859


namespace example_problem_l888_88877

def diamond (a b : ℕ) : ℕ := a^3 + 3 * a^2 * b + 3 * a * b^2 + b^3

theorem example_problem : diamond 3 2 = 125 := by
  sorry

end example_problem_l888_88877


namespace determine_range_of_k_l888_88832

noncomputable def inequality_holds_for_all_x (k : ℝ) : Prop :=
  ∀ (x : ℝ), x^4 + (k - 1) * x^2 + 1 ≥ 0

theorem determine_range_of_k (k : ℝ) : inequality_holds_for_all_x k ↔ k ≥ 1 := sorry

end determine_range_of_k_l888_88832


namespace joe_used_225_gallons_l888_88862

def initial_paint : ℕ := 360

def paint_first_week (initial : ℕ) : ℕ := initial / 4

def remaining_paint_after_first_week (initial : ℕ) : ℕ :=
  initial - paint_first_week initial

def paint_second_week (remaining : ℕ) : ℕ := remaining / 2

def total_paint_used (initial : ℕ) : ℕ :=
  paint_first_week initial + paint_second_week (remaining_paint_after_first_week initial)

theorem joe_used_225_gallons :
  total_paint_used initial_paint = 225 :=
by
  sorry

end joe_used_225_gallons_l888_88862


namespace proof_expr_28_times_35_1003_l888_88850

theorem proof_expr_28_times_35_1003 :
  (5^1003 + 7^1004)^2 - (5^1003 - 7^1004)^2 = 28 * 35^1003 :=
by
  sorry

end proof_expr_28_times_35_1003_l888_88850


namespace minimum_seats_occupied_l888_88803

theorem minimum_seats_occupied (total_seats : ℕ) (h : total_seats = 180) : 
  ∃ occupied_seats : ℕ, occupied_seats = 45 ∧ 
  ∀ additional_person,
    (∀ i : ℕ, i < total_seats → 
     (occupied_seats ≤ i → i < occupied_seats + 1 ∨ i > occupied_seats + 1)) →
    additional_person = occupied_seats + 1  :=
by
  sorry

end minimum_seats_occupied_l888_88803


namespace ducks_to_total_ratio_l888_88825

-- Definitions based on the given conditions
def totalBirds : ℕ := 15
def costPerChicken : ℕ := 2
def totalCostForChickens : ℕ := 20

-- Proving the desired ratio of ducks to total number of birds
theorem ducks_to_total_ratio : (totalCostForChickens / costPerChicken) + d = totalBirds → d = 15 - (totalCostForChickens / costPerChicken) → 
  (totalCostForChickens / costPerChicken) + d = totalBirds → d = totalBirds - (totalCostForChickens / costPerChicken) →
  d = 5 → (totalBirds - (totalCostForChickens / costPerChicken)) / totalBirds = 1 / 3 :=
by
  sorry

end ducks_to_total_ratio_l888_88825


namespace train_speed_l888_88858

noncomputable def train_length : ℝ := 2500
noncomputable def time_to_cross_pole : ℝ := 35

noncomputable def speed_in_kmph (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed :
  speed_in_kmph train_length time_to_cross_pole = 257.14 := by
  sorry

end train_speed_l888_88858


namespace eval_fraction_power_l888_88819

theorem eval_fraction_power : (0.5 ^ 4 / 0.05 ^ 3) = 500 := by
  sorry

end eval_fraction_power_l888_88819


namespace f_2017_plus_f_2016_l888_88849

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom f_even_shift : ∀ x : ℝ, f (x + 2) = f (-x + 2)
axiom f_at_neg1 : f (-1) = -1

theorem f_2017_plus_f_2016 : f 2017 + f 2016 = 1 :=
by
  sorry

end f_2017_plus_f_2016_l888_88849


namespace num_squares_in_6x6_grid_l888_88830

/-- Define the number of kxk squares in an nxn grid -/
def num_squares (n k : ℕ) : ℕ := (n + 1 - k) * (n + 1 - k)

/-- Prove the total number of different squares in a 6x6 grid is 86 -/
theorem num_squares_in_6x6_grid : 
  (num_squares 6 1) + (num_squares 6 2) + (num_squares 6 3) + (num_squares 6 4) = 86 :=
by sorry

end num_squares_in_6x6_grid_l888_88830


namespace projection_of_b_onto_a_l888_88855

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_squared := a.1 * a.1 + a.2 * a.2
  let scalar := dot_product / magnitude_squared
  (scalar * a.1, scalar * a.2)

theorem projection_of_b_onto_a :
  vector_projection (2, -1) (6, 2) = (4, -2) :=
by
  simp [vector_projection]
  sorry

end projection_of_b_onto_a_l888_88855


namespace find_value_of_x_y_l888_88840

theorem find_value_of_x_y (x y : ℝ) (h1 : |x| + x + y = 10) (h2 : |y| + x - y = 12) : x + y = 18 / 5 :=
by
  sorry

end find_value_of_x_y_l888_88840


namespace cookie_cost_proof_l888_88841

def cost_per_cookie (total_spent : ℕ) (days : ℕ) (cookies_per_day : ℕ) : ℕ :=
  total_spent / (days * cookies_per_day)

theorem cookie_cost_proof : cost_per_cookie 1395 31 3 = 15 := by
  sorry

end cookie_cost_proof_l888_88841


namespace pastries_sold_is_correct_l888_88843

-- Definitions of the conditions
def initial_pastries : ℕ := 56
def remaining_pastries : ℕ := 27

-- Statement of the theorem
theorem pastries_sold_is_correct : initial_pastries - remaining_pastries = 29 :=
by
  sorry

end pastries_sold_is_correct_l888_88843


namespace exists_subset_no_three_ap_l888_88871

-- Define the set S_n
def S (n : ℕ) : Finset ℕ := (Finset.range ((3^n + 1) / 2 + 1)).image (λ i => i + 1)

-- Define the property of no three elements forming an arithmetic progression
def no_three_form_ap (M : Finset ℕ) : Prop :=
  ∀ a b c, a ∈ M → b ∈ M → c ∈ M → a < b → b < c → 2 * b ≠ a + c

-- Define the theorem statement
theorem exists_subset_no_three_ap (n : ℕ) :
  ∃ M : Finset ℕ, M ⊆ S n ∧ M.card = 2^n ∧ no_three_form_ap M :=
sorry

end exists_subset_no_three_ap_l888_88871


namespace quadratic_solution_exists_for_any_a_b_l888_88860

theorem quadratic_solution_exists_for_any_a_b (a b : ℝ) : 
  ∃ x : ℝ, (a^6 - b^6)*x^2 + 2*(a^5 - b^5)*x + (a^4 - b^4) = 0 := 
by
  -- The proof would go here
  sorry

end quadratic_solution_exists_for_any_a_b_l888_88860


namespace not_necessarily_divisible_by_28_l888_88898

theorem not_necessarily_divisible_by_28 (k : ℤ) (h : 7 ∣ (k * (k + 1) * (k + 2))) : ¬ (28 ∣ (k * (k + 1) * (k + 2))) :=
sorry

end not_necessarily_divisible_by_28_l888_88898


namespace find_number_l888_88873

noncomputable def percentage_of (p : ℝ) (n : ℝ) := p / 100 * n

noncomputable def fraction_of (f : ℝ) (n : ℝ) := f * n

theorem find_number :
  ∃ x : ℝ, percentage_of 40 60 = fraction_of (4/5) x + 4 ∧ x = 25 :=
by
  sorry

end find_number_l888_88873


namespace number_of_divisible_factorials_l888_88829

theorem number_of_divisible_factorials:
  ∃ (count : ℕ), count = 36 ∧ ∀ n, 1 ≤ n ∧ n ≤ 50 → (∃ k : ℕ, n! = k * (n * (n + 1)) / 2) ↔ n ≤ n - 14 :=
sorry

end number_of_divisible_factorials_l888_88829


namespace pencils_per_friend_l888_88884

theorem pencils_per_friend (total_pencils num_friends : ℕ) (h_total : total_pencils = 24) (h_friends : num_friends = 3) : total_pencils / num_friends = 8 :=
by
  -- Proof would go here
  sorry

end pencils_per_friend_l888_88884


namespace log_ratio_l888_88847

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : log_base 4 a = log_base 6 b)
  (h4 : log_base 6 b = log_base 9 (a + b)) :
  b / a = (1 + Real.sqrt 5) / 2 := sorry

end log_ratio_l888_88847


namespace square_adjacent_to_multiple_of_5_l888_88813

theorem square_adjacent_to_multiple_of_5 (n : ℤ) (h : n % 5 ≠ 0) : (∃ k : ℤ, n^2 = 5 * k + 1) ∨ (∃ k : ℤ, n^2 = 5 * k - 1) := 
by
  sorry

end square_adjacent_to_multiple_of_5_l888_88813


namespace sum_of_four_consecutive_even_integers_l888_88885

theorem sum_of_four_consecutive_even_integers (x : ℕ) (hx : x > 4) :
  (x - 4) * (x - 2) * x * (x + 2) = 48 * (4 * x) → (x - 4) + (x - 2) + x + (x + 2) = 28 := by
{
  sorry
}

end sum_of_four_consecutive_even_integers_l888_88885


namespace required_large_loans_l888_88800

-- We start by introducing the concepts of the number of small, medium, and large loans
def small_loans : Type := ℕ
def medium_loans : Type := ℕ
def large_loans : Type := ℕ

-- Definition of the conditions as two scenarios
def Scenario1 (m s b : ℕ) : Prop := (m = 9 ∧ s = 6 ∧ b = 1)
def Scenario2 (m s b : ℕ) : Prop := (m = 3 ∧ s = 2 ∧ b = 3)

-- Definition of the problem
theorem required_large_loans (m s b : ℕ) (H1 : Scenario1 m s b) (H2 : Scenario2 m s b) :
  b = 4 :=
sorry

end required_large_loans_l888_88800


namespace percentage_forgot_homework_l888_88863

def total_students_group_A : ℕ := 30
def total_students_group_B : ℕ := 50
def forget_percentage_A : ℝ := 0.20
def forget_percentage_B : ℝ := 0.12

theorem percentage_forgot_homework :
  let num_students_forgot_A := forget_percentage_A * total_students_group_A
  let num_students_forgot_B := forget_percentage_B * total_students_group_B
  let total_students_forgot := num_students_forgot_A + num_students_forgot_B
  let total_students := total_students_group_A + total_students_group_B
  let percentage_forgot := (total_students_forgot / total_students) * 100
  percentage_forgot = 15 := sorry

end percentage_forgot_homework_l888_88863


namespace find_number_of_students_l888_88878

theorem find_number_of_students (N T : ℕ) 
  (avg_mark_all : T = 80 * N) 
  (avg_mark_exclude : (T - 150) / (N - 5) = 90) : 
  N = 30 := by
  sorry

end find_number_of_students_l888_88878


namespace max_value_of_f_l888_88886

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^4 - 3*x^2 - 6*x + 13) - Real.sqrt (x^4 - x^2 + 1)

theorem max_value_of_f : ∃ x : ℝ, f x = Real.sqrt 10 :=
sorry

end max_value_of_f_l888_88886


namespace range_of_a1_l888_88897

theorem range_of_a1 (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 1 / (2 - a n)) (h2 : ∀ n, a (n + 1) > a n) :
  a 1 < 1 :=
sorry

end range_of_a1_l888_88897


namespace bowls_initially_bought_l888_88899

theorem bowls_initially_bought 
  (x : ℕ) 
  (cost_per_bowl : ℕ := 13) 
  (revenue_per_bowl : ℕ := 17)
  (sold_bowls : ℕ := 108)
  (profit_percentage : ℝ := 23.88663967611336) 
  (approx_x : ℝ := 139) :
  (23.88663967611336 / 100) * (cost_per_bowl : ℝ) * (x : ℝ) = 
    (sold_bowls * revenue_per_bowl) - (sold_bowls * cost_per_bowl) → 
  abs ((x : ℝ) - approx_x) < 0.5 :=
by
  sorry

end bowls_initially_bought_l888_88899


namespace apprentice_time_l888_88844

theorem apprentice_time
  (x y : ℝ)
  (h1 : 7 * x + 4 * y = 5 / 9)
  (h2 : 11 * x + 8 * y = 17 / 18)
  (hy : y > 0) :
  1 / y = 24 :=
by
  sorry

end apprentice_time_l888_88844


namespace ratio_of_x_intercepts_l888_88893

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v: ℝ)
  (h1: 0 = 8 * u + b) (h2: 0 = 4 * v + b) : u / v = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l888_88893


namespace red_pairs_count_l888_88881

theorem red_pairs_count (blue_shirts red_shirts total_pairs blue_blue_pairs : ℕ)
  (h1 : blue_shirts = 63) 
  (h2 : red_shirts = 81) 
  (h3 : total_pairs = 72) 
  (h4 : blue_blue_pairs = 21)
  : (red_shirts - (blue_shirts - blue_blue_pairs * 2)) / 2 = 30 :=
by
  sorry

end red_pairs_count_l888_88881


namespace range_of_a_l888_88806

theorem range_of_a (a : ℝ) (hx : ∀ x : ℝ, x ≥ 1 → x^2 + a * x + 9 ≥ 0) : a ≥ -6 := 
sorry

end range_of_a_l888_88806


namespace find_largest_x_l888_88870

theorem find_largest_x : 
  ∃ x : ℝ, (4 * x ^ 3 - 17 * x ^ 2 + x + 10 = 0) ∧ 
           (∀ y : ℝ, 4 * y ^ 3 - 17 * y ^ 2 + y + 10 = 0 → y ≤ x) ∧ 
           x = (25 + Real.sqrt 545) / 8 :=
sorry

end find_largest_x_l888_88870


namespace varphi_le_one_varphi_l888_88864

noncomputable def f (a x : ℝ) := -a * Real.log x

-- Definition of the minimum value function φ for a > 0
noncomputable def varphi (a : ℝ) := -a * Real.log a

theorem varphi_le_one (a : ℝ) (h : 0 < a) : varphi a ≤ 1 := 
by sorry

theorem varphi'_le (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
    (1 - Real.log a) ≤ (1 - Real.log b) := 
by sorry

end varphi_le_one_varphi_l888_88864


namespace triangle_right_if_condition_l888_88826

variables (a b c : ℝ) (A B C : ℝ)
-- Condition: Given 1 + cos A = (b + c) / c
axiom h1 : 1 + Real.cos A = (b + c) / c 

-- To prove: a^2 + b^2 = c^2
theorem triangle_right_if_condition (h1 : 1 + Real.cos A = (b + c) / c) : a^2 + b^2 = c^2 :=
  sorry

end triangle_right_if_condition_l888_88826


namespace dodecahedron_edge_coloring_l888_88867

-- Define the properties of the dodecahedron
structure Dodecahedron :=
  (faces : Fin 12)          -- 12 pentagonal faces
  (edges : Fin 30)         -- 30 edges
  (vertices : Fin 20)      -- 20 vertices
  (edge_faces : Fin 30 → Fin 2) -- Each edge contributes to two faces

-- Prove the number of valid edge colorations such that each face has an even number of red edges
theorem dodecahedron_edge_coloring : 
    (∃ num_colorings : ℕ, num_colorings = 2^11) :=
sorry

end dodecahedron_edge_coloring_l888_88867


namespace prob_insurance_A_or_B_prob_exactly_one_no_insurance_out_of_three_l888_88823

noncomputable def P_A : ℝ := 0.5
noncomputable def P_B_not_A : ℝ := 0.3
noncomputable def P_B : ℝ := 0.6  -- given from solution step
noncomputable def P_C : ℝ := 1 - (1 - P_A) * (1 - P_B)
noncomputable def P_D : ℝ := (1 - P_A) * (1 - P_B)
noncomputable def P_E : ℝ := 3 * P_D * (P_C ^ 2)

theorem prob_insurance_A_or_B :
  P_C = 0.8 :=
by
  sorry

theorem prob_exactly_one_no_insurance_out_of_three :
  P_E = 0.384 :=
by
  sorry

end prob_insurance_A_or_B_prob_exactly_one_no_insurance_out_of_three_l888_88823


namespace complex_coordinates_l888_88851

theorem complex_coordinates : (⟨(-1:ℝ), (-1:ℝ)⟩ : ℂ) = (⟨0,1⟩ : ℂ) * (⟨-2,0⟩ : ℂ) / (⟨1,1⟩ : ℂ) :=
by
  sorry

end complex_coordinates_l888_88851


namespace johns_watermelon_weight_l888_88833

-- Michael's largest watermelon weighs 8 pounds
def michael_weight : ℕ := 8

-- Clay's watermelon weighs three times the size of Michael's watermelon
def clay_weight : ℕ := 3 * michael_weight

-- John's watermelon weighs half the size of Clay's watermelon
def john_weight : ℕ := clay_weight / 2

-- Prove that John's watermelon weighs 12 pounds
theorem johns_watermelon_weight : john_weight = 12 := by
  sorry

end johns_watermelon_weight_l888_88833


namespace coefficient_x2_is_negative_40_l888_88816

noncomputable def x2_coefficient_in_expansion (a : ℕ) : ℤ :=
  (-1)^3 * a^2 * Nat.choose 5 2

theorem coefficient_x2_is_negative_40 :
  x2_coefficient_in_expansion 2 = -40 :=
by
  sorry

end coefficient_x2_is_negative_40_l888_88816


namespace john_spending_l888_88889

open Nat Real

noncomputable def cost_of_silver (silver_ounce: Real) (silver_price: Real) : Real :=
  silver_ounce * silver_price

noncomputable def quantity_of_gold (silver_ounce: Real): Real :=
  2 * silver_ounce

noncomputable def cost_per_ounce_gold (silver_price: Real) (multiplier: Real): Real :=
  silver_price * multiplier

noncomputable def cost_of_gold (gold_ounce: Real) (gold_price: Real) : Real :=
  gold_ounce * gold_price

noncomputable def total_cost (cost_silver: Real) (cost_gold: Real): Real :=
  cost_silver + cost_gold

theorem john_spending :
  let silver_ounce := 1.5
  let silver_price := 20
  let gold_multiplier := 50
  let cost_silver := cost_of_silver silver_ounce silver_price
  let gold_ounce := quantity_of_gold silver_ounce
  let gold_price := cost_per_ounce_gold silver_price gold_multiplier
  let cost_gold := cost_of_gold gold_ounce gold_price
  let total := total_cost cost_silver cost_gold
  total = 3030 :=
by
  sorry

end john_spending_l888_88889


namespace negation_proposition_l888_88875

theorem negation_proposition {x : ℝ} (h : ∀ x > 0, Real.sin x > 0) : ∃ x > 0, Real.sin x ≤ 0 :=
sorry

end negation_proposition_l888_88875


namespace find_valid_pairs_l888_88809

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def valid_pair (p q : ℕ) : Prop :=
  p < 2005 ∧ q < 2005 ∧ is_prime p ∧ is_prime q ∧ q ∣ p^2 + 8 ∧ p ∣ q^2 + 8

theorem find_valid_pairs :
  ∀ p q, valid_pair p q → (p, q) = (2, 2) ∨ (p, q) = (881, 89) ∨ (p, q) = (89, 881) :=
sorry

end find_valid_pairs_l888_88809


namespace no_rectangular_prism_equal_measures_l888_88839

theorem no_rectangular_prism_equal_measures (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0): 
  ¬ (4 * (a + b + c) = 2 * (a * b + b * c + c * a) ∧ 2 * (a * b + b * c + c * a) = a * b * c) :=
by
  sorry

end no_rectangular_prism_equal_measures_l888_88839


namespace find_k_l888_88868

def green_balls : ℕ := 7

noncomputable def probability_green (k : ℕ) : ℚ := green_balls / (green_balls + k)
noncomputable def probability_purple (k : ℕ) : ℚ := k / (green_balls + k)

noncomputable def winning_for_green : ℤ := 3
noncomputable def losing_for_purple : ℤ := -1

noncomputable def expected_value (k : ℕ) : ℚ :=
  (probability_green k) * (winning_for_green : ℚ) + (probability_purple k) * (losing_for_purple : ℚ)

theorem find_k (k : ℕ) (h : expected_value k = 1) : k = 7 :=
  sorry

end find_k_l888_88868


namespace find_smallest_value_l888_88808

noncomputable def smallest_value (a b c d : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2

theorem find_smallest_value (a b c d : ℝ) (h1: a + b = 18)
  (h2: ab + c + d = 85) (h3: ad + bc = 180) (h4: cd = 104) :
  smallest_value a b c d = 484 :=
sorry

end find_smallest_value_l888_88808


namespace g_value_at_8_l888_88838

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

theorem g_value_at_8 (g : ℝ → ℝ) (h1 : ∀ x : ℝ, g x = (1/216) * (x - (a^3)) * (x - (b^3)) * (x - (c^3))) 
  (h2 : g 0 = 1) 
  (h3 : ∀ a b c : ℝ, f (a) = 0 ∧ f (b) = 0 ∧ f (c) = 0) : 
  g 8 = 0 :=
sorry

end g_value_at_8_l888_88838


namespace sunday_saturday_ratio_is_two_to_one_l888_88827

-- Define the conditions as given in the problem
def total_pages : ℕ := 360
def saturday_morning_read : ℕ := 40
def saturday_night_read : ℕ := 10
def remaining_pages : ℕ := 210

-- Define Ethan's total pages read so far
def total_read : ℕ := total_pages - remaining_pages

-- Define pages read on Saturday
def saturday_total_read : ℕ := saturday_morning_read + saturday_night_read

-- Define pages read on Sunday
def sunday_total_read : ℕ := total_read - saturday_total_read

-- Define the ratio of pages read on Sunday to pages read on Saturday
def sunday_to_saturday_ratio : ℕ := sunday_total_read / saturday_total_read

-- Theorem statement: ratio of pages read on Sunday to pages read on Saturday is 2:1
theorem sunday_saturday_ratio_is_two_to_one : sunday_to_saturday_ratio = 2 :=
by
  -- This part should contain the detailed proof
  sorry

end sunday_saturday_ratio_is_two_to_one_l888_88827


namespace min_value_expression_ge_512_l888_88828

noncomputable def min_value_expression (a b c : ℝ) : ℝ :=
  (a^3 + 4*a^2 + a + 1) * (b^3 + 4*b^2 + b + 1) * (c^3 + 4*c^2 + c + 1) / (a * b * c)

theorem min_value_expression_ge_512 {a b c : ℝ} 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  min_value_expression a b c ≥ 512 :=
by
  sorry

end min_value_expression_ge_512_l888_88828


namespace sum_of_products_equal_l888_88814

theorem sum_of_products_equal 
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ)
  (h1 : a1 + a2 + a3 = b1 + b2 + b3)
  (h2 : b1 + b2 + b3 = c1 + c2 + c3)
  (h3 : c1 + c2 + c3 = a1 + b1 + c1)
  (h4 : a1 + b1 + c1 = a2 + b2 + c2)
  (h5 : a2 + b2 + c2 = a3 + b3 + c3) :
  a1 * b1 * c1 + a2 * b2 * c2 + a3 * b3 * c3 = a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3 :=
by 
  sorry

end sum_of_products_equal_l888_88814


namespace find_line_eq_l888_88812

noncomputable def line_eq (x y : ℝ) : Prop :=
  (∃ a : ℝ, a ≠ 0 ∧ (a * x - y = 0 ∨ x + y - a = 0)) 

theorem find_line_eq : line_eq 2 3 :=
by
  sorry

end find_line_eq_l888_88812


namespace factor_is_given_sum_l888_88835

theorem factor_is_given_sum (P Q : ℤ)
  (h1 : ∀ x : ℝ, (x^2 + 3 * x + 7) * (x^2 + (-3) * x + 7) = x^4 + P * x^2 + Q) :
  P + Q = 54 := 
sorry

end factor_is_given_sum_l888_88835


namespace arithmetic_sum_nine_l888_88866

noncomputable def arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a 1 + a n)

theorem arithmetic_sum_nine (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h2 : a 4 = 9)
  (h3 : a 6 = 11) : arithmetic_sequence_sum a 9 = 90 :=
by
  sorry

end arithmetic_sum_nine_l888_88866


namespace sin_tan_identity_of_cos_eq_tan_identity_l888_88892

open Real

variable (α : ℝ)
variable (hα : α ∈ Ioo 0 π)   -- α is in the interval (0, π)
variable (hcos : cos (2 * α) = 2 * cos (α + π / 4))

theorem sin_tan_identity_of_cos_eq_tan_identity : 
  sin (2 * α) = 1 ∧ tan α = 1 :=
by
  sorry

end sin_tan_identity_of_cos_eq_tan_identity_l888_88892


namespace FriedChickenDinner_orders_count_l888_88891

-- Defining the number of pieces of chicken used by each type of order
def piecesChickenPasta := 2
def piecesBarbecueChicken := 3
def piecesFriedChickenDinner := 8

-- Defining the number of orders for Chicken Pasta and Barbecue Chicken
def numChickenPastaOrders := 6
def numBarbecueChickenOrders := 3

-- Defining the total pieces of chicken needed for all orders
def totalPiecesOfChickenNeeded := 37

-- Defining the number of pieces of chicken needed for Chicken Pasta and Barbecue orders
def piecesNeededChickenPasta : Nat := piecesChickenPasta * numChickenPastaOrders
def piecesNeededBarbecueChicken : Nat := piecesBarbecueChicken * numBarbecueChickenOrders

-- Defining the total pieces of chicken needed for Chicken Pasta and Barbecue orders
def piecesNeededChickenPastaAndBarbecue : Nat := piecesNeededChickenPasta + piecesNeededBarbecueChicken

-- Calculating the pieces of chicken needed for Fried Chicken Dinner orders
def piecesNeededFriedChickenDinner : Nat := totalPiecesOfChickenNeeded - piecesNeededChickenPastaAndBarbecue

-- Defining the number of Fried Chicken Dinner orders
def numFriedChickenDinnerOrders : Nat := piecesNeededFriedChickenDinner / piecesFriedChickenDinner

-- Proving Victor has 2 Fried Chicken Dinner orders
theorem FriedChickenDinner_orders_count : numFriedChickenDinnerOrders = 2 := by
  unfold numFriedChickenDinnerOrders
  unfold piecesNeededFriedChickenDinner
  unfold piecesNeededChickenPastaAndBarbecue
  unfold piecesNeededBarbecueChicken
  unfold piecesNeededChickenPasta
  unfold totalPiecesOfChickenNeeded
  unfold numBarbecueChickenOrders
  unfold piecesBarbecueChicken
  unfold numChickenPastaOrders
  unfold piecesChickenPasta
  sorry

end FriedChickenDinner_orders_count_l888_88891


namespace job_planned_completion_days_l888_88805

noncomputable def initial_days_planned (W D : ℝ) := 6 * (W / D) = (W - 3 * (W / D)) / 3

theorem job_planned_completion_days (W : ℝ ) : 
  ∃ D : ℝ, initial_days_planned W D ∧ D = 6 := 
sorry

end job_planned_completion_days_l888_88805


namespace roots_are_simplified_sqrt_form_l888_88880

theorem roots_are_simplified_sqrt_form : 
  ∃ m p n : ℕ, gcd m p = 1 ∧ gcd p n = 1 ∧ gcd m n = 1 ∧
    (∀ x : ℝ, (3 * x^2 - 8 * x + 1 = 0) ↔ 
    (x = (m : ℝ) + (Real.sqrt n)/(p : ℝ) ∨ x = (m : ℝ) - (Real.sqrt n)/(p : ℝ))) ∧
    n = 13 :=
by
  sorry

end roots_are_simplified_sqrt_form_l888_88880


namespace center_of_circle_eq_minus_two_four_l888_88894

theorem center_of_circle_eq_minus_two_four : 
  ∀ (x y : ℝ), x^2 + 4 * x + y^2 - 8 * y + 16 = 0 → (x, y) = (-2, 4) :=
by {
  sorry
}

end center_of_circle_eq_minus_two_four_l888_88894


namespace range_of_m_l888_88895

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + 2 ≥ m

def proposition_q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → -(7 - 3*m)^x > -(7 - 3*m)^y

theorem range_of_m (m : ℝ) :
  (proposition_p m ∧ ¬ proposition_q m) ∨ (¬ proposition_p m ∧ proposition_q m) ↔ (1 < m ∧ m < 2) :=
sorry

end range_of_m_l888_88895
