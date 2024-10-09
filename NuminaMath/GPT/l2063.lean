import Mathlib

namespace proof_problem_l2063_206309

open Set Real

noncomputable def f (x : ℝ) : ℝ := sin x
noncomputable def g (x : ℝ) : ℝ := cos x
def U : Set ℝ := univ
def M : Set ℝ := {x | f x ≠ 0}
def N : Set ℝ := {x | g x ≠ 0}
def C_U (s : Set ℝ) : Set ℝ := U \ s

theorem proof_problem :
  {x : ℝ | f x * g x = 0} = (C_U M) ∪ (C_U N) :=
by
  sorry

end proof_problem_l2063_206309


namespace inequality_proof_l2063_206380

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z) ^ 2 * (y * z + z * x + x * y) ^ 2 ≤ 
  3 * (y^2 + y * z + z^2) * (z^2 + z * x + x^2) * (x^2 + x * y + y^2) := 
sorry

end inequality_proof_l2063_206380


namespace positive_integer_solutions_eq_8_2_l2063_206340

-- Define the variables and conditions in the problem
def positive_integer_solution_count_eq (n m : ℕ) : Prop :=
  ∀ (x₁ x₂ x₃ x₄ : ℕ),
    x₂ = m →
    (x₁ + x₂ + x₃ + x₄ = n) →
    (x₁ > 0 ∧ x₃ > 0 ∧ x₄ > 0) →
    -- Number of positive integer solutions should be 10
    (x₁ + x₃ + x₄ = 6)

-- Statement of the theorem
theorem positive_integer_solutions_eq_8_2 : positive_integer_solution_count_eq 8 2 := sorry

end positive_integer_solutions_eq_8_2_l2063_206340


namespace compute_fraction_l2063_206353

theorem compute_fraction : ((5 * 7) - 3) / 9 = 32 / 9 := by
  sorry

end compute_fraction_l2063_206353


namespace fraction_addition_l2063_206337

-- Definitions from conditions
def frac1 : ℚ := 18 / 42
def frac2 : ℚ := 2 / 9
def simplified_frac1 : ℚ := 3 / 7
def simplified_frac2 : ℚ := frac2
def common_denom_frac1 : ℚ := 27 / 63
def common_denom_frac2 : ℚ := 14 / 63

-- The problem statement to prove
theorem fraction_addition :
  frac1 + frac2 = 41 / 63 := by
  sorry

end fraction_addition_l2063_206337


namespace sum_of_squares_largest_multiple_of_7_l2063_206368

theorem sum_of_squares_largest_multiple_of_7
  (N : ℕ) (a : ℕ) (h1 : N = a^2 + (a + 1)^2 + (a + 2)^2)
  (h2 : N < 10000)
  (h3 : 7 ∣ N) :
  N = 8750 := sorry

end sum_of_squares_largest_multiple_of_7_l2063_206368


namespace bird_families_left_near_mountain_l2063_206322

def total_bird_families : ℕ := 85
def bird_families_flew_to_africa : ℕ := 23
def bird_families_flew_to_asia : ℕ := 37

theorem bird_families_left_near_mountain : total_bird_families - (bird_families_flew_to_africa + bird_families_flew_to_asia) = 25 := by
  sorry

end bird_families_left_near_mountain_l2063_206322


namespace extremum_condition_l2063_206347

noncomputable def y (a x : ℝ) : ℝ := Real.exp (a * x) + 3 * x

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, (y a x = 0) ∧ ∀ x' > x, y a x' < y a x) → a < -3 :=
by
  sorry

end extremum_condition_l2063_206347


namespace gcd_128_144_256_l2063_206358

theorem gcd_128_144_256 : Nat.gcd (Nat.gcd 128 144) 256 = 128 :=
  sorry

end gcd_128_144_256_l2063_206358


namespace minimize_quadratic_expression_l2063_206334

noncomputable def quadratic_expression (b : ℝ) : ℝ :=
  (1 / 3) * b^2 + 7 * b - 6

theorem minimize_quadratic_expression : ∃ b : ℝ, quadratic_expression b = -10.5 :=
  sorry

end minimize_quadratic_expression_l2063_206334


namespace cleaning_project_l2063_206348

theorem cleaning_project (x : ℕ) : 12 + x = 2 * (15 - x) := sorry

end cleaning_project_l2063_206348


namespace num_children_attended_show_l2063_206379

def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := 13
def num_adults : ℕ := 183
def total_revenue : ℕ := 5122

theorem num_children_attended_show : ∃ C : ℕ, (num_adults * ticket_price_adult + C * ticket_price_child = total_revenue) ∧ C = 28 :=
by
  sorry

end num_children_attended_show_l2063_206379


namespace william_total_tickets_l2063_206393

def initial_tickets : ℕ := 15
def additional_tickets : ℕ := 3
def total_tickets : ℕ := initial_tickets + additional_tickets

theorem william_total_tickets :
  total_tickets = 18 := by
  -- proof goes here
  sorry

end william_total_tickets_l2063_206393


namespace ratio_elephants_to_others_l2063_206328

theorem ratio_elephants_to_others (L P E : ℕ) (h1 : L = 2 * P) (h2 : L = 200) (h3 : L + P + E = 450) :
  E / (L + P) = 1 / 2 :=
by
  sorry

end ratio_elephants_to_others_l2063_206328


namespace temp_neg_represents_below_zero_l2063_206332

-- Definitions based on the conditions in a)
def above_zero (x: ℤ) : Prop := x > 0
def below_zero (x: ℤ) : Prop := x < 0

-- Proof problem derived from c)
theorem temp_neg_represents_below_zero (t1 t2: ℤ) 
  (h1: above_zero t1) (h2: t1 = 10) 
  (h3: below_zero t2) (h4: t2 = -3) : 
  -t2 = 3 :=
by
  sorry

end temp_neg_represents_below_zero_l2063_206332


namespace find_range_a_l2063_206318

noncomputable def sincos_inequality (x a θ : ℝ) : Prop :=
  (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1 / 8

theorem find_range_a :
  (∀ (x : ℝ) (θ : ℝ), θ ∈ Set.Icc 0 (Real.pi / 2) → sincos_inequality x a θ)
  ↔ a ≥ 7 / 2 ∨ a ≤ Real.sqrt 6 :=
sorry

end find_range_a_l2063_206318


namespace total_amount_paid_l2063_206329

/-- Conditions -/
def days_in_may : Nat := 31
def rate_per_day : ℚ := 0.5
def days_book1_borrowed : Nat := 20
def days_book2_borrowed : Nat := 31
def days_book3_borrowed : Nat := 31

/-- Question and Proof -/
theorem total_amount_paid : rate_per_day * (days_book1_borrowed + days_book2_borrowed + days_book3_borrowed) = 41 := by
  sorry

end total_amount_paid_l2063_206329


namespace maximum_marks_l2063_206354

theorem maximum_marks (M : ℝ) (h1 : 212 + 25 = 237) (h2 : 0.30 * M = 237) : M = 790 := 
by
  sorry

end maximum_marks_l2063_206354


namespace solution_set_of_inequality_l2063_206305

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x - 1) / (2 - x) ≥ 0} = {x : ℝ | 1 / 3 ≤ x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l2063_206305


namespace volume_of_tetrahedron_ABCD_l2063_206314

noncomputable def tetrahedron_volume_proof (S: ℝ) (AB AD BD: ℝ) 
    (angle_ABD_DBC_CBA angle_ADB_BDC_CDA angle_ACB_ACD_BCD: ℝ) : ℝ :=
if h1 : S = 1 ∧ AB = AD ∧ BD = (Real.sqrt 2) / 2
    ∧ angle_ABD_DBC_CBA = 180 ∧ angle_ADB_BDC_CDA = 180 
    ∧ angle_ACB_ACD_BCD = 90 then
  (1 / 24)
else
  0

-- Statement to prove
theorem volume_of_tetrahedron_ABCD : tetrahedron_volume_proof 1 AB AD ((Real.sqrt 2) / 2) 180 180 90 = (1 / 24) :=
by sorry

end volume_of_tetrahedron_ABCD_l2063_206314


namespace geometric_sequence_product_l2063_206319

variable {a : ℕ → ℝ}
variable {r : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_product 
  (h : is_geometric_sequence a r)
  (h_cond : a 4 * a 6 = 10) :
  a 2 * a 8 = 10 := 
sorry

end geometric_sequence_product_l2063_206319


namespace fence_cost_l2063_206308

noncomputable def price_per_foot (total_cost : ℝ) (perimeter : ℝ) : ℝ :=
  total_cost / perimeter

theorem fence_cost (area : ℝ) (total_cost : ℝ) (price : ℝ) :
  area = 289 → total_cost = 4012 → price = price_per_foot 4012 (4 * (Real.sqrt 289)) → price = 59 :=
by
  intros h_area h_cost h_price
  sorry

end fence_cost_l2063_206308


namespace sequence_periodic_a2014_l2063_206303

theorem sequence_periodic_a2014 (a : ℕ → ℚ) 
  (h1 : a 1 = -1/4) 
  (h2 : ∀ n > 1, a n = 1 - (1 / (a (n - 1)))) : 
  a 2014 = -1/4 :=
sorry

end sequence_periodic_a2014_l2063_206303


namespace find_number_l2063_206373

theorem find_number (N : ℕ) (h1 : N / 3 = 8) (h2 : N / 8 = 3) : N = 24 :=
by
  sorry

end find_number_l2063_206373


namespace polynomial_equivalence_l2063_206343

theorem polynomial_equivalence (x y : ℝ) (h : y = x + 1/x) :
  (x^2 * (y^2 + 2*y - 5) = 0) ↔ (x^4 + 2*x^3 - 3*x^2 + 2*x + 1 = 0) :=
by
  sorry

end polynomial_equivalence_l2063_206343


namespace remove_terms_to_make_sum_l2063_206389

theorem remove_terms_to_make_sum (a b c d e f : ℚ) (h₁ : a = 1/3) (h₂ : b = 1/5) (h₃ : c = 1/7) (h₄ : d = 1/9) (h₅ : e = 1/11) (h₆ : f = 1/13) :
  a + b + c + d + e + f - e - f = 3/2 :=
by
  sorry

end remove_terms_to_make_sum_l2063_206389


namespace find_k_l2063_206387

-- Define the vector operations and properties

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def vector_smul (k : ℝ) (a : ℝ × ℝ) : ℝ × ℝ := (k * a.1, k * a.2)
def vectors_parallel (a b : ℝ × ℝ) : Prop := 
  (a.1 * b.2 = a.2 * b.1)

-- Given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Statement of the problem
theorem find_k (k : ℝ) : 
  vectors_parallel (vector_add (vector_smul k a) b) (vector_add a (vector_smul (-3) b)) 
  → k = -1 / 3 :=
by
  sorry

end find_k_l2063_206387


namespace sum_of_integers_between_neg20_5_and_10_5_l2063_206360

theorem sum_of_integers_between_neg20_5_and_10_5 :
  let a := -20
  let l := 10
  let n := (l - a) / 1 + 1
  let S := n / 2 * (a + l)
  S = -155 := by
{
  sorry
}

end sum_of_integers_between_neg20_5_and_10_5_l2063_206360


namespace min_people_for_no_empty_triplet_60_l2063_206394

noncomputable def min_people_for_no_empty_triplet (total_chairs : ℕ) : ℕ :=
  if h : total_chairs % 3 = 0 then total_chairs / 3 else sorry

theorem min_people_for_no_empty_triplet_60 :
  min_people_for_no_empty_triplet 60 = 20 :=
by
  sorry

end min_people_for_no_empty_triplet_60_l2063_206394


namespace solve_system_of_equations_general_solve_system_of_equations_zero_case_1_solve_system_of_equations_zero_case_2_solve_system_of_equations_zero_case_3_solve_system_of_equations_special_cases_l2063_206359

-- Define the conditions
variables (a b c x y z: ℝ) 

-- Define the system of equations
def system_of_equations (a b c x y z : ℝ) : Prop :=
  (a * y + b * x = c) ∧
  (c * x + a * z = b) ∧
  (b * z + c * y = a)

-- Define the general solution
def solution (a b c x y z : ℝ) : Prop :=
  x = (b^2 + c^2 - a^2) / (2 * b * c) ∧
  y = (a^2 + c^2 - b^2) / (2 * a * c) ∧
  z = (a^2 + b^2 - c^2) / (2 * a * b)

-- Define the proof problem statement
theorem solve_system_of_equations_general (a b c x y z : ℝ) (h : system_of_equations a b c x y z) 
      (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : solution a b c x y z :=
  sorry

-- Special cases
theorem solve_system_of_equations_zero_case_1 (b c x y z : ℝ) (h : system_of_equations 0 b c x y z) : c = 0 :=
  sorry

theorem solve_system_of_equations_zero_case_2 (a b c x y z : ℝ) (h1 : a = 0) (h2 : b = 0) (h3: c ≠ 0) : c = 0 :=
  sorry

theorem solve_system_of_equations_zero_case_3 (b c x y z : ℝ) (h : system_of_equations 0 b c x y z) : x = c / b ∧ 
      (c * x = b) :=
  sorry

-- Following special cases more concisely
theorem solve_system_of_equations_special_cases (a b c x y z : ℝ) 
      (h : system_of_equations a b c x y z) (h1: a = 0 ∨ b = 0 ∨ c = 0): 
      (∃ k : ℝ, x = k ∧ y = -k ∧ z = k)  
    ∨ (∃ k : ℝ, x = k ∧ y = k ∧ z = -k)
    ∨ (∃ k : ℝ, x = -k ∧ y = k ∧ z = k) :=
  sorry

end solve_system_of_equations_general_solve_system_of_equations_zero_case_1_solve_system_of_equations_zero_case_2_solve_system_of_equations_zero_case_3_solve_system_of_equations_special_cases_l2063_206359


namespace min_xy_value_l2063_206323

theorem min_xy_value (x y : ℝ) (hx : 1 < x) (hy : 1 < y) (hlog : Real.log x / Real.log 2 * Real.log y / Real.log 2 = 1) : x * y = 4 :=
by sorry

end min_xy_value_l2063_206323


namespace hall_reunion_attendees_l2063_206336

noncomputable def Oates : ℕ := 40
noncomputable def both : ℕ := 10
noncomputable def total : ℕ := 100
noncomputable def onlyOates := Oates - both
noncomputable def onlyHall := total - onlyOates - both
noncomputable def Hall := onlyHall + both

theorem hall_reunion_attendees : Hall = 70 := by {
  sorry
}

end hall_reunion_attendees_l2063_206336


namespace total_price_of_books_l2063_206369

theorem total_price_of_books
  (total_books : ℕ)
  (math_books_cost : ℕ)
  (history_books_cost : ℕ)
  (math_books_bought : ℕ)
  (total_books_eq : total_books = 80)
  (math_books_cost_eq : math_books_cost = 4)
  (history_books_cost_eq : history_books_cost = 5)
  (math_books_bought_eq : math_books_bought = 10) :
  (math_books_bought * math_books_cost + (total_books - math_books_bought) * history_books_cost = 390) := 
by
  sorry

end total_price_of_books_l2063_206369


namespace newspapers_sold_correct_l2063_206304

def total_sales : ℝ := 425.0
def magazines_sold : ℝ := 150
def newspapers_sold : ℝ := total_sales - magazines_sold

theorem newspapers_sold_correct : newspapers_sold = 275.0 := by
  sorry

end newspapers_sold_correct_l2063_206304


namespace value_of_m_l2063_206361

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2 * x + 1 else 2 * (-x) + 1

theorem value_of_m (m : ℝ) (heven : ∀ x : ℝ, f (-x) = f x)
  (hpos : ∀ x : ℝ, x ≥ 0 → f x = 2 * x + 1)
  (hfm : f m = 5) : m = 2 ∨ m = -2 :=
sorry

end value_of_m_l2063_206361


namespace Igor_colored_all_cells_l2063_206385

theorem Igor_colored_all_cells (m n : ℕ) (h1 : 9 * m = 12 * n) (h2 : 0 < m ∧ m ≤ 4) (h3 : 0 < n ∧ n ≤ 3) :
  m = 4 ∧ n = 3 :=
by {
  sorry
}

end Igor_colored_all_cells_l2063_206385


namespace kevin_food_spending_l2063_206346

theorem kevin_food_spending :
  let total_budget := 20
  let samuel_ticket := 14
  let samuel_food_and_drinks := 6
  let kevin_drinks := 2
  let kevin_food_and_drinks := total_budget - (samuel_ticket + samuel_food_and_drinks) - kevin_drinks
  kevin_food_and_drinks = 4 :=
by
  sorry

end kevin_food_spending_l2063_206346


namespace common_integer_solutions_l2063_206301

theorem common_integer_solutions
    (y : ℤ)
    (h1 : -4 * y ≥ 2 * y + 10)
    (h2 : -3 * y ≤ 15)
    (h3 : -5 * y ≥ 3 * y + 24)
    (h4 : y ≤ -1) :
  y = -3 ∨ y = -4 ∨ y = -5 :=
by 
  sorry

end common_integer_solutions_l2063_206301


namespace volume_of_cube_in_pyramid_l2063_206396

open Real

noncomputable def side_length_of_base := 2
noncomputable def height_of_equilateral_triangle := sqrt 6
noncomputable def cube_side_length := sqrt 6 / 3
noncomputable def volume_of_cube := cube_side_length ^ 3

theorem volume_of_cube_in_pyramid 
  (side_length_of_base : ℝ) (height_of_equilateral_triangle : ℝ) (cube_side_length : ℝ) :
  volume_of_cube = 2 * sqrt 6 / 9 := 
by
  sorry

end volume_of_cube_in_pyramid_l2063_206396


namespace parabola_focus_l2063_206345

theorem parabola_focus (a : ℝ) (h1 : ∀ x y, x^2 = a * y ↔ y = x^2 / a)
(h2 : focus_coordinates = (0, 5)) : a = 20 := 
sorry

end parabola_focus_l2063_206345


namespace div_recurring_decimal_l2063_206306

def recurringDecimalToFraction (q : ℚ) (h : q = 36/99) : ℚ := by
  sorry

theorem div_recurring_decimal : 12 / recurringDecimalToFraction 0.36 sorry = 33 :=
by
  sorry

end div_recurring_decimal_l2063_206306


namespace combined_tax_rate_l2063_206378

-- Definitions of the problem conditions
def tax_rate_Mork : ℝ := 0.40
def tax_rate_Mindy : ℝ := 0.25

-- Asserts the condition that Mindy earned 4 times as much as Mork
def income_ratio (income_Mindy income_Mork : ℝ) := income_Mindy = 4 * income_Mork

-- The theorem to be proved: The combined tax rate is 28%.
theorem combined_tax_rate (income_Mork income_Mindy total_income total_tax : ℝ)
  (h_income_ratio : income_ratio income_Mindy income_Mork)
  (total_income_eq : total_income = income_Mork + income_Mindy)
  (total_tax_eq : total_tax = tax_rate_Mork * income_Mork + tax_rate_Mindy * income_Mindy) :
  total_tax / total_income = 0.28 := sorry

end combined_tax_rate_l2063_206378


namespace g_composed_g_has_exactly_two_distinct_real_roots_l2063_206324

theorem g_composed_g_has_exactly_two_distinct_real_roots (d : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 + 4 * x + d) = 0 ∧ (y^2 + 4 * y + d) = 0) ↔ d = 8 :=
sorry

end g_composed_g_has_exactly_two_distinct_real_roots_l2063_206324


namespace length_of_faster_train_proof_l2063_206341

-- Definitions based on the given conditions
def faster_train_speed_kmh := 72 -- in km/h
def slower_train_speed_kmh := 36 -- in km/h
def time_to_cross_seconds := 18 -- in seconds

-- Conversion factor from km/h to m/s
def kmh_to_ms := 5 / 18

-- Define the relative speed in m/s
def relative_speed_ms := (faster_train_speed_kmh - slower_train_speed_kmh) * kmh_to_ms

-- Length of the faster train in meters
def length_of_faster_train := relative_speed_ms * time_to_cross_seconds

-- The theorem statement for the Lean prover
theorem length_of_faster_train_proof : length_of_faster_train = 180 := by
  sorry

end length_of_faster_train_proof_l2063_206341


namespace rectangle_difference_l2063_206399

theorem rectangle_difference (L B : ℝ) (h1 : 2 * (L + B) = 266) (h2 : L * B = 4290) :
  L - B = 23 :=
sorry

end rectangle_difference_l2063_206399


namespace draw_two_green_marbles_probability_l2063_206333

theorem draw_two_green_marbles_probability :
  let red := 5
  let green := 3
  let white := 7
  let total := red + green + white
  (green / total) * ((green - 1) / (total - 1)) = 1 / 35 :=
by
  sorry

end draw_two_green_marbles_probability_l2063_206333


namespace weight_of_A_l2063_206371

theorem weight_of_A
  (W_A W_B W_C W_D W_E : ℕ)
  (H_A H_B H_C H_D : ℕ)
  (Age_A Age_B Age_C Age_D : ℕ)
  (hw1 : (W_A + W_B + W_C) / 3 = 84)
  (hh1 : (H_A + H_B + H_C) / 3 = 170)
  (ha1 : (Age_A + Age_B + Age_C) / 3 = 30)
  (hw2 : (W_A + W_B + W_C + W_D) / 4 = 80)
  (hh2 : (H_A + H_B + H_C + H_D) / 4 = 172)
  (ha2 : (Age_A + Age_B + Age_C + Age_D) / 4 = 28)
  (hw3 : (W_B + W_C + W_D + W_E) / 4 = 79)
  (hh3 : (H_B + H_C + H_D + H_E) / 4 = 173)
  (ha3 : (Age_B + Age_C + Age_D + (Age_A - 3)) / 4 = 27)
  (hw4 : W_E = W_D + 7)
  : W_A = 79 := 
sorry

end weight_of_A_l2063_206371


namespace find_y_intercept_l2063_206374

theorem find_y_intercept (a b : ℝ) (h1 : (3 : ℝ) ≠ (7 : ℝ))
  (h2 : -2 = a * 3 + b) (h3 : 14 = a * 7 + b) :
  b = -14 :=
sorry

end find_y_intercept_l2063_206374


namespace find_b_l2063_206300

theorem find_b (a b c : ℕ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : 1 < c):
  (∀ N : ℝ, N ≠ 1 → (N^(3/a) * N^(2/(ab)) * N^(1/(abc)) = N^(39/48))) → b = 4 :=
  by
  sorry

end find_b_l2063_206300


namespace combined_population_after_two_years_l2063_206395

def population_after_years (initial_population : ℕ) (yearly_changes : List (ℕ → ℕ)) : ℕ :=
  yearly_changes.foldl (fun pop change => change pop) initial_population

def townA_change_year1 (pop : ℕ) : ℕ :=
  pop + (pop * 8 / 100) + 200 - 100

def townA_change_year2 (pop : ℕ) : ℕ :=
  pop + (pop * 10 / 100) + 200 - 100

def townB_change_year1 (pop : ℕ) : ℕ :=
  pop - (pop * 2 / 100) + 50 - 200

def townB_change_year2 (pop : ℕ) : ℕ :=
  pop - (pop * 1 / 100) + 50 - 200

theorem combined_population_after_two_years :
  population_after_years 15000 [townA_change_year1, townA_change_year2] +
  population_after_years 10000 [townB_change_year1, townB_change_year2] = 27433 := 
  sorry

end combined_population_after_two_years_l2063_206395


namespace expected_teachers_with_masters_degree_l2063_206317

theorem expected_teachers_with_masters_degree
  (prob: ℚ) (teachers: ℕ) (h_prob: prob = 1/4) (h_teachers: teachers = 320) :
  prob * teachers = 80 :=
by
  sorry

end expected_teachers_with_masters_degree_l2063_206317


namespace father_l2063_206349

theorem father's_age (M F : ℕ) (h1 : M = 2 * F / 5) (h2 : M + 6 = (F + 6) / 2) : F = 30 :=
by
  sorry

end father_l2063_206349


namespace min_m_plus_n_l2063_206372

theorem min_m_plus_n (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m * n - 2 * m - 3 * n = 20) : 
  m + n = 20 :=
sorry

end min_m_plus_n_l2063_206372


namespace summation_indices_equal_l2063_206356

theorem summation_indices_equal
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bound : ∀ i, a i ≤ 100)
  (h_length : ∀ i, i < 16) :
  ∃ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ a i + a j = a k + a l := 
by {
  sorry
}

end summation_indices_equal_l2063_206356


namespace trigonometric_identity_l2063_206388

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 / 2 :=
by
  sorry

end trigonometric_identity_l2063_206388


namespace gathering_people_total_l2063_206391

theorem gathering_people_total (W S B : ℕ) (hW : W = 26) (hS : S = 22) (hB : B = 17) :
  W + S - B = 31 :=
by
  sorry

end gathering_people_total_l2063_206391


namespace triplet_sums_to_two_l2063_206307

theorem triplet_sums_to_two :
  (3 / 4 + 1 / 4 + 1 = 2) ∧
  (1.2 + 0.8 + 0 = 2) ∧
  (0.5 + 1.0 + 0.5 = 2) ∧
  (3 / 5 + 4 / 5 + 3 / 5 = 2) ∧
  (2 - 3 + 3 = 2) :=
by
  sorry

end triplet_sums_to_two_l2063_206307


namespace lcm_1_to_10_l2063_206344

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l2063_206344


namespace robot_possible_path_lengths_l2063_206351

theorem robot_possible_path_lengths (n : ℕ) (valid_path: ∀ (i : ℕ), i < n → (i % 4 = 0 ∨ i % 4 = 1 ∨ i % 4 = 2 ∨ i % 4 = 3)) :
  (n % 4 = 0) :=
by
  sorry

end robot_possible_path_lengths_l2063_206351


namespace arithmetic_sequence_sum_mul_three_eq_3480_l2063_206367

theorem arithmetic_sequence_sum_mul_three_eq_3480 :
  let a := 50
  let d := 3
  let l := 95
  let n := ((l - a) / d + 1 : ℕ)
  let sum := n * (a + l) / 2
  3 * sum = 3480 := by
  sorry

end arithmetic_sequence_sum_mul_three_eq_3480_l2063_206367


namespace measure_of_angle_D_in_scalene_triangle_l2063_206316

-- Define the conditions
def is_scalene (D E F : ℝ) : Prop :=
  D ≠ E ∧ E ≠ F ∧ D ≠ F

-- Define the measure of angles based on the given conditions
def measure_of_angle_D (D E F : ℝ) : Prop :=
  E = 2 * D ∧ F = 40

-- Define the sum of angles in a triangle
def triangle_angle_sum (D E F : ℝ) : Prop :=
  D + E + F = 180

theorem measure_of_angle_D_in_scalene_triangle (D E F : ℝ) (h_scalene : is_scalene D E F) 
  (h_measures : measure_of_angle_D D E F) (h_sum : triangle_angle_sum D E F) : D = 140 / 3 :=
by 
  sorry

end measure_of_angle_D_in_scalene_triangle_l2063_206316


namespace rate_of_stream_l2063_206350

theorem rate_of_stream (x : ℝ) (h1 : ∀ (distance : ℝ), (24 : ℝ) > 0) (h2 : ∀ (distance : ℝ), (distance / (24 - x)) = 3 * (distance / (24 + x))) : x = 12 :=
by
  sorry

end rate_of_stream_l2063_206350


namespace problem_statement_l2063_206315

theorem problem_statement :
  ∀ (x : ℝ),
    (5 * x - 10 = 15 * x + 5) →
    (5 * (x + 3) = 15 / 2) :=
by
  intros x h
  sorry

end problem_statement_l2063_206315


namespace geometric_sequence_product_l2063_206370

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geom : ∀ n : ℕ, a (n + 1) = r * a n)
variable (h_condition : a 5 * a 14 = 5)

theorem geometric_sequence_product :
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  sorry

end geometric_sequence_product_l2063_206370


namespace sum_of_cubes_l2063_206381

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) : x^3 + y^3 = -10 :=
by
  sorry

end sum_of_cubes_l2063_206381


namespace timothy_tea_cups_l2063_206390

theorem timothy_tea_cups (t : ℕ) (h : 6 * t + 60 = 120) : t + 12 = 22 :=
by
  sorry

end timothy_tea_cups_l2063_206390


namespace mrs_randall_total_teaching_years_l2063_206312

def years_teaching_third_grade : ℕ := 18
def years_teaching_second_grade : ℕ := 8

theorem mrs_randall_total_teaching_years : years_teaching_third_grade + years_teaching_second_grade = 26 :=
by
  sorry

end mrs_randall_total_teaching_years_l2063_206312


namespace cookies_from_dough_l2063_206357

theorem cookies_from_dough :
  ∀ (length width : ℕ), length = 24 → width = 18 →
  ∃ (side : ℕ), side = Nat.gcd length width ∧ (length / side) * (width / side) = 12 :=
by
  intros length width h_length h_width
  simp only [h_length, h_width]
  use Nat.gcd length width
  simp only [Nat.gcd_rec]
  sorry

end cookies_from_dough_l2063_206357


namespace least_number_to_multiply_l2063_206383

theorem least_number_to_multiply (x : ℕ) :
  (72 * x) % 112 = 0 → x = 14 :=
by 
  sorry

end least_number_to_multiply_l2063_206383


namespace measure_of_angle_l2063_206342

theorem measure_of_angle (x : ℝ) (h : 90 - x = 3 * x - 10) : x = 25 :=
by
  sorry

end measure_of_angle_l2063_206342


namespace distance_proof_l2063_206377

-- Definitions from the conditions
def avg_speed_to_retreat := 50
def avg_speed_back_home := 75
def total_round_trip_time := 10
def distance_between_home_and_retreat := 300

-- Theorem stating the problem
theorem distance_proof 
  (D : ℝ)
  (h1 : D / avg_speed_to_retreat + D / avg_speed_back_home = total_round_trip_time) :
  D = distance_between_home_and_retreat :=
sorry

end distance_proof_l2063_206377


namespace arithmetic_geometric_sequence_l2063_206311

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
    (h1 : a 1 = 3)
    (h2 : a 1 + a 3 + a 5 = 21)
    (h3 : ∀ n, a (n + 1) = a n * q) :
  a 2 * a 4 = 36 := 
sorry

end arithmetic_geometric_sequence_l2063_206311


namespace no_sum_of_two_squares_l2063_206375

theorem no_sum_of_two_squares (n : ℤ) (h : n % 4 = 3) : ¬∃ a b : ℤ, n = a^2 + b^2 := 
by
  sorry

end no_sum_of_two_squares_l2063_206375


namespace quadratic_root_a_value_l2063_206355

theorem quadratic_root_a_value (a : ℝ) :
  (∃ x : ℝ, x = -2 ∧ x^2 + (3 / 2) * a * x - a^2 = 0) → (a = 1 ∨ a = -4) := 
by
  intro h
  sorry

end quadratic_root_a_value_l2063_206355


namespace value_of_f_ln3_l2063_206397

def f : ℝ → ℝ := sorry

theorem value_of_f_ln3 (f_symm : ∀ x : ℝ, f (x + 1) = f (-x + 1))
  (f_exp : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = Real.exp (-x)) :
  f (Real.log 3) = 3 * Real.exp (-2) :=
by
  sorry

end value_of_f_ln3_l2063_206397


namespace inequality_holds_l2063_206382

variable (a b c d : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (hd : d > 0)
variable (h : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2)

theorem inequality_holds (ha : a > 0)
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (h : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2) :
  (1 - a) / (a^2 - a + 1) + (1 - b) / (b^2 - b + 1) + (1 - c) / (c^2 - c + 1) + (1 - d) / (d^2 - d + 1) ≥ 0 := by
  sorry

end inequality_holds_l2063_206382


namespace math_problem_solution_l2063_206327

theorem math_problem_solution (x y : ℝ) : 
  abs x + x + 5 * y = 2 ∧ abs y - y + x = 7 → x + y + 2009 = 2012 :=
by {
  sorry
}

end math_problem_solution_l2063_206327


namespace option_C_correct_l2063_206352

variable (a b : ℝ)

theorem option_C_correct (h : a > b) : -15 * a < -15 * b := 
  sorry

end option_C_correct_l2063_206352


namespace fish_remaining_l2063_206364

theorem fish_remaining
  (initial_guppies : ℕ)
  (initial_angelfish : ℕ)
  (initial_tiger_sharks : ℕ)
  (initial_oscar_fish : ℕ)
  (sold_guppies : ℕ)
  (sold_angelfish : ℕ)
  (sold_tiger_sharks : ℕ)
  (sold_oscar_fish : ℕ)
  (initial_total : ℕ := initial_guppies + initial_angelfish + initial_tiger_sharks + initial_oscar_fish)
  (sold_total : ℕ := sold_guppies + sold_angelfish + sold_tiger_sharks + sold_oscar_fish)
  (remaining : ℕ := initial_total - sold_total) :
  initial_guppies = 94 →
  initial_angelfish = 76 →
  initial_tiger_sharks = 89 →
  initial_oscar_fish = 58 →
  sold_guppies = 30 →
  sold_angelfish = 48 →
  sold_tiger_sharks = 17 →
  sold_oscar_fish = 24 →
  remaining = 198 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  norm_num
  sorry

end fish_remaining_l2063_206364


namespace product_square_preceding_div_by_12_l2063_206366

theorem product_square_preceding_div_by_12 (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) :=
by
  sorry

end product_square_preceding_div_by_12_l2063_206366


namespace calc_expression_l2063_206339

-- Define the fractions and whole number in the problem
def frac1 : ℚ := 5/6
def frac2 : ℚ := 1 + 1/6
def whole : ℚ := 2

-- Define the expression to be proved
def expression : ℚ := (frac1) - (-whole) + (frac2)

-- The theorem to be proved
theorem calc_expression : expression = 4 :=
by { sorry }

end calc_expression_l2063_206339


namespace find_f_2023_l2063_206392

def is_strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ {a b : ℕ}, a < b → f a < f b

theorem find_f_2023 (f : ℕ → ℕ)
  (h_inc : is_strictly_increasing f)
  (h_relation : ∀ m n : ℕ, f (n + f m) = f n + m + 1) :
  f 2023 = 2024 :=
sorry

end find_f_2023_l2063_206392


namespace annabelle_savings_l2063_206338

noncomputable def weeklyAllowance : ℕ := 30
noncomputable def junkFoodFraction : ℚ := 1 / 3
noncomputable def sweetsCost : ℕ := 8

theorem annabelle_savings :
  let junkFoodCost := weeklyAllowance * junkFoodFraction
  let totalSpent := junkFoodCost + sweetsCost
  let savings := weeklyAllowance - totalSpent
  savings = 12 := 
by
  sorry

end annabelle_savings_l2063_206338


namespace hyperbola_condition_l2063_206321

theorem hyperbola_condition (k : ℝ) : 
  (∀ x y : ℝ, (x^2 / (1 + k)) - (y^2 / (1 - k)) = 1 → (-1 < k ∧ k < 1)) ∧ 
  ((-1 < k ∧ k < 1) → ∀ x y : ℝ, (x^2 / (1 + k)) - (y^2 / (1 - k)) = 1) :=
sorry

end hyperbola_condition_l2063_206321


namespace geometric_series_problem_l2063_206330

noncomputable def geometric_series_sum (a r : ℝ) : ℝ := a / (1 - r)

theorem geometric_series_problem
  (c d : ℝ)
  (h : geometric_series_sum (c/d) (1/d) = 6) :
  geometric_series_sum (c/(c + 2 * d)) (1/(c + 2 * d)) = 3 / 4 := by
  sorry

end geometric_series_problem_l2063_206330


namespace triangle_inequality_l2063_206331

variable {a b c : ℝ}

theorem triangle_inequality (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  (a^2 + 2 * b * c) / (b^2 + c^2) + (b^2 + 2 * a * c) / (c^2 + a^2) + (c^2 + 2 * a * b) / (a^2 + b^2) > 3 :=
by {
  sorry
}

end triangle_inequality_l2063_206331


namespace odd_function_symmetry_l2063_206376

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then x^2 else sorry

theorem odd_function_symmetry (x : ℝ) (k : ℕ) (h1 : ∀ y, f (-y) = -f y)
  (h2 : ∀ y, f y = f (2 - y)) (h3 : ∀ y, 0 < y ∧ y ≤ 1 → f y = y^2) :
  k = 45 / 4 → f k = -9 / 16 :=
by
  intros _
  sorry

end odd_function_symmetry_l2063_206376


namespace sum_volumes_of_spheres_sum_volumes_of_tetrahedrons_l2063_206320

noncomputable def volume_of_spheres (V : ℝ) : ℝ :=
  V * (27 / 26)

noncomputable def volume_of_tetrahedrons (V : ℝ) : ℝ :=
  (3 * V * Real.sqrt 3) / (13 * Real.pi)

theorem sum_volumes_of_spheres (V : ℝ) : 
  (∑' n : ℕ, (V * (1/27)^n)) = volume_of_spheres V :=
sorry

theorem sum_volumes_of_tetrahedrons (V : ℝ) (r : ℝ) : 
  (∑' n : ℕ, (8/9 / Real.sqrt 3 * (r^3) * (1/27)^n * (1/26))) = volume_of_tetrahedrons V :=
sorry

end sum_volumes_of_spheres_sum_volumes_of_tetrahedrons_l2063_206320


namespace athlete_distance_l2063_206310

theorem athlete_distance (t : ℝ) (v_kmh : ℝ) (v_ms : ℝ) (d : ℝ)
  (h1 : t = 24)
  (h2 : v_kmh = 30.000000000000004)
  (h3 : v_ms = v_kmh * 1000 / 3600)
  (h4 : d = v_ms * t) :
  d = 200 := 
sorry

end athlete_distance_l2063_206310


namespace center_of_circle_is_correct_l2063_206384

-- Define the conditions as Lean functions and statements
def is_tangent (x y : ℝ) : Prop :=
  (3 * x + 4 * y = 48) ∨ (3 * x + 4 * y = -12)

def is_on_line (x y : ℝ) : Prop := x = y

-- Define the proof statement
theorem center_of_circle_is_correct (x y : ℝ) (h1 : is_tangent x y) (h2 : is_on_line x y) :
  (x, y) = (18 / 7, 18 / 7) :=
sorry

end center_of_circle_is_correct_l2063_206384


namespace least_value_expression_l2063_206386

theorem least_value_expression : ∃ x : ℝ, ∀ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 ≥ 2094
∧ ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 = 2094 := by
  sorry

end least_value_expression_l2063_206386


namespace geometric_sequence_sum_l2063_206335

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (ha1 : q ≠ 0)
  (h1 : a 1 + a 2 = 3) 
  (h2 : a 3 + a 4 = (a 1 + a 2) * q^2)
  : a 5 + a 6 = 48 :=
by
  sorry

end geometric_sequence_sum_l2063_206335


namespace Kims_final_score_l2063_206325

def easy_points : ℕ := 2
def average_points : ℕ := 3
def hard_points : ℕ := 5
def expert_points : ℕ := 7

def easy_correct : ℕ := 6
def average_correct : ℕ := 2
def hard_correct : ℕ := 4
def expert_correct : ℕ := 3

def complex_problems_bonus : ℕ := 1
def complex_problems_solved : ℕ := 2

def penalty_per_incorrect : ℕ := 1
def easy_incorrect : ℕ := 1
def average_incorrect : ℕ := 2
def hard_incorrect : ℕ := 2
def expert_incorrect : ℕ := 3

theorem Kims_final_score : 
  (easy_correct * easy_points + 
   average_correct * average_points + 
   hard_correct * hard_points + 
   expert_correct * expert_points + 
   complex_problems_solved * complex_problems_bonus) - 
   (easy_incorrect * penalty_per_incorrect + 
    average_incorrect * penalty_per_incorrect + 
    hard_incorrect * penalty_per_incorrect + 
    expert_incorrect * penalty_per_incorrect) = 53 :=
by 
  sorry

end Kims_final_score_l2063_206325


namespace find_f_six_l2063_206326

theorem find_f_six (f : ℕ → ℤ) (h : ∀ (x : ℕ), f (x + 1) = x^2 - 4) : f 6 = 21 :=
by
sorry

end find_f_six_l2063_206326


namespace tan_ratio_l2063_206313

theorem tan_ratio (a b : ℝ) (ha : 0 < a ∧ a < π/2) (hb : 0 < b ∧ b < π/2)
  (h1 : Real.sin (a + b) = 5/8) (h2 : Real.sin (a - b) = 3/8) :
  (Real.tan a) / (Real.tan b) = 4 :=
by
  sorry

end tan_ratio_l2063_206313


namespace dave_books_about_outer_space_l2063_206362

theorem dave_books_about_outer_space (x : ℕ) 
  (H1 : 8 + 3 = 11) 
  (H2 : 11 * 6 = 66) 
  (H3 : 102 - 66 = 36) 
  (H4 : 36 / 6 = x) : 
  x = 6 := 
by
  sorry

end dave_books_about_outer_space_l2063_206362


namespace sum_of_arithmetic_sequence_l2063_206302

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ) 
  (h₁ : S 4 = 2) 
  (h₂ : S 8 = 6) 
  : S 12 = 12 := 
by
  sorry

end sum_of_arithmetic_sequence_l2063_206302


namespace range_of_a_l2063_206363

theorem range_of_a (a : ℝ)
  (h : ∀ x y : ℝ, (4 * x - 3 * y - 2 = 0) → (x^2 + y^2 - 2 * a * x + 4 * y + a^2 - 12 = 0) → x ≠ y) :
  -6 < a ∧ a < 4 :=
by
  sorry

end range_of_a_l2063_206363


namespace convert_to_base10_sum_l2063_206365

def base8_to_dec (d2 d1 d0 : Nat) : Nat :=
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

def base13_to_dec (d2 d1 d0 : Nat) : Nat :=
  d2 * 13^2 + d1 * 13^1 + d0 * 13^0

def convert_537_8 : Nat :=
  base8_to_dec 5 3 7

def convert_4C5_13 : Nat :=
  base13_to_dec 4 12 5

theorem convert_to_base10_sum : 
  convert_537_8 + convert_4C5_13 = 1188 := 
by 
  sorry

end convert_to_base10_sum_l2063_206365


namespace solve_cubic_fraction_l2063_206398

noncomputable def problem_statement (x : ℝ) :=
  (x = (-(3:ℝ) + Real.sqrt 13) / 4) ∨ (x = (-(3:ℝ) - Real.sqrt 13) / 4)

theorem solve_cubic_fraction (x : ℝ) (h : (x^3 + 2*x^2 + 3*x + 5) / (x + 2) = x + 4) : 
  problem_statement x :=
by
  sorry

end solve_cubic_fraction_l2063_206398
