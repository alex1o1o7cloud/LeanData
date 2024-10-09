import Mathlib

namespace find_x_l2296_229645

-- Define the variables and conditions
def x := 27
axiom h : (5 / 3) * x = 45

-- Main statement to be proved
theorem find_x : x = 27 :=
by
  have : (5 / 3) * x = 45 := h
  sorry

end find_x_l2296_229645


namespace find_distance_l2296_229611

theorem find_distance (T D : ℝ) 
  (h1 : D = 5 * (T + 0.2)) 
  (h2 : D = 6 * (T - 0.25)) : 
  D = 13.5 :=
by
  sorry

end find_distance_l2296_229611


namespace quadratic_inequality_solution_l2296_229651

theorem quadratic_inequality_solution (m: ℝ) (h: m > 1) :
  { x : ℝ | x^2 + (m - 1) * x - m ≥ 0 } = { x | x ≤ -m ∨ x ≥ 1 } :=
sorry

end quadratic_inequality_solution_l2296_229651


namespace bob_spending_over_limit_l2296_229626

theorem bob_spending_over_limit : 
  ∀ (necklace_price book_price limit total_cost amount_over_limit : ℕ),
  necklace_price = 34 →
  book_price = necklace_price + 5 →
  limit = 70 →
  total_cost = necklace_price + book_price →
  amount_over_limit = total_cost - limit →
  amount_over_limit = 3 :=
by
  intros
  sorry

end bob_spending_over_limit_l2296_229626


namespace find_a_b_sum_l2296_229632

theorem find_a_b_sum (a b : ℕ) (h1 : 830 - (400 + 10 * a + 7) = 300 + 10 * b + 4)
    (h2 : ∃ k : ℕ, 300 + 10 * b + 4 = 7 * k) : a + b = 2 :=
by
  sorry

end find_a_b_sum_l2296_229632


namespace initial_house_cats_l2296_229699

theorem initial_house_cats (H : ℕ) (H_condition : 13 + H - 10 = 8) : H = 5 :=
by
-- sorry provides a placeholder to skip the actual proof
sorry

end initial_house_cats_l2296_229699


namespace each_vaccine_costs_45_l2296_229675

theorem each_vaccine_costs_45
    (num_vaccines : ℕ)
    (doctor_visit_cost : ℝ)
    (insurance_coverage : ℝ)
    (trip_cost : ℝ)
    (total_payment : ℝ) :
    num_vaccines = 10 ->
    doctor_visit_cost = 250 ->
    insurance_coverage = 0.80 ->
    trip_cost = 1200 ->
    total_payment = 1340 ->
    (∃ (vaccine_cost : ℝ), vaccine_cost = 45) :=
by {
    sorry
}

end each_vaccine_costs_45_l2296_229675


namespace unique_solution_for_exponential_eq_l2296_229659

theorem unique_solution_for_exponential_eq (a y : ℕ) (h_a : a ≥ 1) (h_y : y ≥ 1) :
  3^(2*a-1) + 3^a + 1 = 7^y ↔ (a = 1 ∧ y = 1) := by
  sorry

end unique_solution_for_exponential_eq_l2296_229659


namespace average_cd_l2296_229687

theorem average_cd (c d : ℝ) (h : (4 + 6 + 8 + c + d) / 5 = 18) : (c + d) / 2 = 36 := 
by
  -- The proof goes here
  sorry

end average_cd_l2296_229687


namespace old_geometry_book_pages_l2296_229671

def old_pages := 340
def new_pages := 450
def deluxe_pages := 915

theorem old_geometry_book_pages : 
  (new_pages = 2 * old_pages - 230) ∧ 
  (deluxe_pages = new_pages + old_pages + 125) ∧ 
  (deluxe_pages ≥ old_pages + old_pages / 10) 
  → old_pages = 340 := by
  sorry

end old_geometry_book_pages_l2296_229671


namespace remainder_of_repeated_23_l2296_229615

theorem remainder_of_repeated_23 {n : ℤ} (n : ℤ) (hn : n = 23 * 10^(2*23)) : 
  (n % 32) = 19 :=
sorry

end remainder_of_repeated_23_l2296_229615


namespace solution_set_of_inequality_l2296_229628

theorem solution_set_of_inequality :
  {x : ℝ | (x^2 - 2*x - 3) * (x^2 + 1) < 0} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end solution_set_of_inequality_l2296_229628


namespace negation_of_proposition_l2296_229629

theorem negation_of_proposition (p : ∀ (x : ℝ), x^2 + 1 > 0) :
  ∃ (x : ℝ), x^2 + 1 ≤ 0 ↔ ¬ (∀ (x : ℝ), x^2 + 1 > 0) :=
by
  sorry

end negation_of_proposition_l2296_229629


namespace consecutive_numbers_perfect_square_l2296_229658

theorem consecutive_numbers_perfect_square (a : ℕ) (h : a ≥ 1) : 
  (a * (a + 1) * (a + 2) * (a + 3) + 1) = (a^2 + 3 * a + 1)^2 :=
by sorry

end consecutive_numbers_perfect_square_l2296_229658


namespace dot_product_in_triangle_l2296_229625

noncomputable def ab := 3
noncomputable def ac := 2
noncomputable def bc := Real.sqrt 10

theorem dot_product_in_triangle : 
  let AB := ab
  let AC := ac
  let BC := bc
  (AB = 3) → (AC = 2) → (BC = Real.sqrt 10) → 
  ∃ cosA, (cosA = (AB^2 + AC^2 - BC^2) / (2 * AB * AC)) →
  ∃ dot_product, (dot_product = AB * AC * cosA) ∧ dot_product = 3 / 2 :=
by
  sorry

end dot_product_in_triangle_l2296_229625


namespace solve_inequality_l2296_229697

theorem solve_inequality (x : ℝ) : -7/3 < x ∧ x < 7 → |x+2| + |x-2| < x + 7 :=
by
  intro h
  sorry

end solve_inequality_l2296_229697


namespace sum_of_15_consecutive_integers_perfect_square_l2296_229643

open Nat

-- statement that defines the conditions and the objective of the problem
theorem sum_of_15_consecutive_integers_perfect_square :
  ∃ n k : ℕ, 15 * (n + 7) = k^2 ∧ 15 * (n + 7) ≥ 225 := 
sorry

end sum_of_15_consecutive_integers_perfect_square_l2296_229643


namespace solve_quadratic_equation_l2296_229672

theorem solve_quadratic_equation (x : ℝ) : x^2 - 4*x + 3 = 0 ↔ (x = 1 ∨ x = 3) := 
by 
  sorry

end solve_quadratic_equation_l2296_229672


namespace monotonic_increasing_interval_f_l2296_229607

noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2 * x + 8)

theorem monotonic_increasing_interval_f :
  ∃ I : Set ℝ, (I = Set.Icc (-2) 1) ∧ (∀x1 ∈ I, ∀x2 ∈ I, x1 ≤ x2 → f x1 ≤ f x2) :=
sorry

end monotonic_increasing_interval_f_l2296_229607


namespace bus_seating_options_l2296_229665

theorem bus_seating_options :
  ∃! (x y : ℕ), 21*x + 10*y = 241 :=
sorry

end bus_seating_options_l2296_229665


namespace necessarily_negative_b_plus_3b_squared_l2296_229602

theorem necessarily_negative_b_plus_3b_squared
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 1) :
  b + 3 * b^2 < 0 :=
sorry

end necessarily_negative_b_plus_3b_squared_l2296_229602


namespace tree_growth_per_year_l2296_229649

-- Defining the initial height and age.
def initial_height : ℕ := 5
def initial_age : ℕ := 1

-- Defining the height and age after a certain number of years.
def height_at_7_years : ℕ := 23
def age_at_7_years : ℕ := 7

-- Calculating the total growth and number of years.
def total_height_growth : ℕ := height_at_7_years - initial_height
def years_of_growth : ℕ := age_at_7_years - initial_age

-- Stating the theorem to be proven.
theorem tree_growth_per_year : total_height_growth / years_of_growth = 3 :=
by
  sorry

end tree_growth_per_year_l2296_229649


namespace sum_of_number_and_reverse_is_perfect_square_iff_l2296_229679

def is_two_digit (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def reverse_of (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem sum_of_number_and_reverse_is_perfect_square_iff :
  ∀ n : ℕ, is_two_digit n →
    is_perfect_square (n + reverse_of n) ↔
      n = 29 ∨ n = 38 ∨ n = 47 ∨ n = 56 ∨ n = 65 ∨ n = 74 ∨ n = 83 ∨ n = 92 :=
by
  sorry

end sum_of_number_and_reverse_is_perfect_square_iff_l2296_229679


namespace conference_attendees_l2296_229640

theorem conference_attendees (w m : ℕ) (h1 : w + m = 47) (h2 : 16 + (w - 1) = m) : w = 16 ∧ m = 31 :=
by
  sorry

end conference_attendees_l2296_229640


namespace total_legs_correct_l2296_229646

def num_ants : ℕ := 12
def num_spiders : ℕ := 8
def legs_per_ant : ℕ := 6
def legs_per_spider : ℕ := 8
def total_legs := num_ants * legs_per_ant + num_spiders * legs_per_spider

theorem total_legs_correct : total_legs = 136 :=
by
  sorry

end total_legs_correct_l2296_229646


namespace smallest_possible_value_of_n_l2296_229655

theorem smallest_possible_value_of_n :
  ∃ n : ℕ, (60 * n = (x + 6) * x * (x + 6) ∧ (x > 0) ∧ gcd 60 n = x + 6) ∧ n = 93 :=
by
  sorry

end smallest_possible_value_of_n_l2296_229655


namespace simplify_expression_l2296_229617

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ( (x^6 - 1) / (3 * x^3) )^2) = Real.sqrt (x^12 + 7 * x^6 + 1) / (3 * x^3) :=
by sorry

end simplify_expression_l2296_229617


namespace quadratic_function_is_parabola_l2296_229619

theorem quadratic_function_is_parabola (a : ℝ) (b : ℝ) (c : ℝ) :
  ∃ k h, ∀ x, (y = a * (x - h)^2 + k) ∧ a ≠ 0 → (y = 3 * (x - 2)^2 + 6) → (a = 3 ∧ h = 2 ∧ k = 6) → ∀ x, (y = 3 * (x - 2)^2 + 6) := 
by
  sorry

end quadratic_function_is_parabola_l2296_229619


namespace neg_09_not_in_integers_l2296_229606

def negative_numbers : Set ℝ := {x | x < 0}
def fractions : Set ℝ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}
def integers : Set ℝ := {x | ∃ (n : ℤ), x = n}
def rational_numbers : Set ℝ := {x | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b}

theorem neg_09_not_in_integers : -0.9 ∉ integers :=
by {
  sorry
}

end neg_09_not_in_integers_l2296_229606


namespace area_of_rhombus_l2296_229608

/-- Given the radii of the circles circumscribed around triangles EFG and EGH
    are 10 and 20, respectively, then the area of rhombus EFGH is 30.72√3. -/
theorem area_of_rhombus (R1 R2 : ℝ) (A : ℝ) :
  R1 = 10 → R2 = 20 → A = 30.72 * Real.sqrt 3 :=
by sorry

end area_of_rhombus_l2296_229608


namespace negation_abs_lt_zero_l2296_229666

theorem negation_abs_lt_zero : ¬ (∀ x : ℝ, |x| < 0) ↔ ∃ x : ℝ, |x| ≥ 0 := 
by 
  sorry

end negation_abs_lt_zero_l2296_229666


namespace num_quadricycles_l2296_229614

theorem num_quadricycles (b t q : ℕ) (h1 : b + t + q = 10) (h2 : 2 * b + 3 * t + 4 * q = 30) : q = 2 :=
by sorry

end num_quadricycles_l2296_229614


namespace find_g2_l2296_229698

theorem find_g2
  (g : ℝ → ℝ)
  (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x ^ 2) :
  g 2 = 19 / 16 := 
sorry

end find_g2_l2296_229698


namespace local_language_letters_l2296_229637

theorem local_language_letters (n : ℕ) (h : 1 + 2 * n = 139) : n = 69 :=
by
  -- Proof skipped
  sorry

end local_language_letters_l2296_229637


namespace find_x2_div_c2_add_y2_div_a2_add_z2_div_b2_l2296_229684

variable (a b c x y z : ℝ)

theorem find_x2_div_c2_add_y2_div_a2_add_z2_div_b2 
  (h1 : a * (x / c) + b * (y / a) + c * (z / b) = 5) 
  (h2 : c / x + a / y + b / z = 0) : 
  x^2 / c^2 + y^2 / a^2 + z^2 / b^2 = 25 := 
sorry

end find_x2_div_c2_add_y2_div_a2_add_z2_div_b2_l2296_229684


namespace range_of_x_l2296_229696

-- Define the ceiling function for ease of use.
noncomputable def ceil (x : ℝ) : ℤ := ⌈x⌉

theorem range_of_x (x : ℝ) (h1 : ceil (2 * x + 1) = 5) (h2 : ceil (2 - 3 * x) = -3) :
  (5 / 3 : ℝ) ≤ x ∧ x < 2 :=
by
  sorry

end range_of_x_l2296_229696


namespace present_age_of_son_l2296_229622

theorem present_age_of_son :
  (∃ (S F : ℕ), F = S + 22 ∧ (F + 2) = 2 * (S + 2)) → ∃ (S : ℕ), S = 20 :=
by
  sorry

end present_age_of_son_l2296_229622


namespace f_comp_g_eq_g_comp_f_iff_l2296_229634

variable {R : Type} [CommRing R]

def f (m n : R) (x : R) : R := m * x ^ 2 + n
def g (p q : R) (x : R) : R := p * x + q

theorem f_comp_g_eq_g_comp_f_iff (m n p q : R) :
  (∀ x : R, f m n (g p q x) = g p q (f m n x)) ↔ n * (1 - p ^ 2) - q * (1 - m) = 0 :=
by
  sorry

end f_comp_g_eq_g_comp_f_iff_l2296_229634


namespace value_of_a6_l2296_229652

theorem value_of_a6 (a : ℕ → ℝ) (h_positive : ∀ n, 0 < a n)
  (h_a1 : a 1 = 1) (h_a2 : a 2 = 2)
  (h_recurrence : ∀ n, 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2) :
  a 6 = 4 := 
sorry

end value_of_a6_l2296_229652


namespace kellan_wax_remaining_l2296_229685

def remaining_wax (initial_A : ℕ) (initial_B : ℕ)
                  (spill_A : ℕ) (spill_B : ℕ)
                  (use_car_A : ℕ) (use_suv_B : ℕ) : ℕ :=
  let remaining_A := initial_A - spill_A - use_car_A
  let remaining_B := initial_B - spill_B - use_suv_B
  remaining_A + remaining_B

theorem kellan_wax_remaining
  (initial_A : ℕ := 10) 
  (initial_B : ℕ := 15)
  (spill_A : ℕ := 3) 
  (spill_B : ℕ := 4)
  (use_car_A : ℕ := 4) 
  (use_suv_B : ℕ := 5) :
  remaining_wax initial_A initial_B spill_A spill_B use_car_A use_suv_B = 9 :=
by sorry

end kellan_wax_remaining_l2296_229685


namespace speed_of_sound_l2296_229669

theorem speed_of_sound (time_heard : ℕ) (time_occured : ℕ) (distance : ℝ) : 
  time_heard = 30 * 60 + 20 → 
  time_occured = 30 * 60 → 
  distance = 6600 → 
  (distance / ((time_heard - time_occured) / 3600)) / 3600 = 330 :=
by 
  intros h1 h2 h3
  sorry

end speed_of_sound_l2296_229669


namespace find_sachin_age_l2296_229689

variables (S R : ℕ)

def sachin_young_than_rahul_by_4_years (S R : ℕ) : Prop := R = S + 4
def ratio_of_ages (S R : ℕ) : Prop := 7 * R = 9 * S

theorem find_sachin_age (S R : ℕ) (h1 : sachin_young_than_rahul_by_4_years S R) (h2 : ratio_of_ages S R) : S = 14 := 
by sorry

end find_sachin_age_l2296_229689


namespace cost_of_first_20_kgs_l2296_229682

theorem cost_of_first_20_kgs (l q : ℕ)
  (h1 : 30 * l + 3 * q = 168)
  (h2 : 30 * l + 6 * q = 186) :
  20 * l = 100 :=
by
  sorry

end cost_of_first_20_kgs_l2296_229682


namespace shorter_leg_of_right_triangle_l2296_229654

theorem shorter_leg_of_right_triangle {a b : ℕ} (hypotenuse : ℕ) (h : hypotenuse = 41) (h_right_triangle : a^2 + b^2 = hypotenuse^2) (h_ineq : a < b) : a = 9 :=
by {
  -- proof to be filled in 
  sorry
}

end shorter_leg_of_right_triangle_l2296_229654


namespace log_identity_proof_l2296_229694

theorem log_identity_proof (lg : ℝ → ℝ) (h1 : lg 50 = lg 2 + lg 25) (h2 : lg 25 = 2 * lg 5) :
  (lg 2)^2 + lg 2 * lg 50 + lg 25 = 2 :=
by sorry

end log_identity_proof_l2296_229694


namespace Laura_running_speed_l2296_229691

noncomputable def running_speed (x : ℝ) :=
  let biking_time := 30 / (3 * x + 2)
  let running_time := 10 / x
  let total_time := biking_time + running_time
  total_time = 3

theorem Laura_running_speed : ∃ x : ℝ, running_speed x ∧ abs (x - 6.35) < 0.01 :=
sorry

end Laura_running_speed_l2296_229691


namespace Irene_age_is_46_l2296_229605

-- Definitions based on the given conditions
def Eddie_age : ℕ := 92
def Becky_age : ℕ := Eddie_age / 4
def Irene_age : ℕ := 2 * Becky_age

-- Theorem we aim to prove that Irene's age is 46
theorem Irene_age_is_46 : Irene_age = 46 := by
  sorry

end Irene_age_is_46_l2296_229605


namespace find_c_l2296_229624

-- Define the quadratic polynomial with given conditions
def quadratic (b c x y : ℝ) : Prop :=
  y = x^2 + b * x + c

-- Define the condition that the polynomial passes through two particular points
def passes_through_points (b c : ℝ) : Prop :=
  (quadratic b c 1 4) ∧ (quadratic b c 5 4)

-- The theorem stating c is 9 given the conditions
theorem find_c (b c : ℝ) (h : passes_through_points b c) : c = 9 :=
by {
  sorry
}

end find_c_l2296_229624


namespace exists_ints_for_inequalities_l2296_229674

theorem exists_ints_for_inequalities (a b : ℝ) (ε : ℝ) (hε : ε > 0) :
  ∃ (n : ℕ) (k m : ℤ), |(n * a) - k| < ε ∧ |(n * b) - m| < ε :=
by
  sorry

end exists_ints_for_inequalities_l2296_229674


namespace Connie_correct_number_l2296_229636

theorem Connie_correct_number (x : ℤ) (h : x + 2 = 80) : x - 2 = 76 := by
  sorry

end Connie_correct_number_l2296_229636


namespace simplify_expression_l2296_229612

-- Define the given expression
def given_expr (x y : ℝ) := 3 * x + 4 * y + 5 * x^2 + 2 - (8 - 5 * x - 3 * y - 2 * x^2)

-- Define the expected simplified expression
def simplified_expr (x y : ℝ) := 7 * x^2 + 8 * x + 7 * y - 6

-- Theorem statement to prove the equivalence of the expressions
theorem simplify_expression (x y : ℝ) : 
  given_expr x y = simplified_expr x y := sorry

end simplify_expression_l2296_229612


namespace find_base_a_l2296_229638

theorem find_base_a 
  (a : ℕ)
  (C_a : ℕ := 12) :
  (3 * a^2 + 4 * a + 7) + (5 * a^2 + 7 * a + 9) = 9 * a^2 + 2 * a + C_a →
  a = 14 :=
by
  intros h
  sorry

end find_base_a_l2296_229638


namespace jane_current_age_l2296_229644

theorem jane_current_age (J : ℕ) (h1 : ∀ t : ℕ, t = 13 → 25 + t = 2 * (J + t)) : J = 6 :=
by {
  sorry
}

end jane_current_age_l2296_229644


namespace a_squared_plus_b_squared_less_than_c_squared_l2296_229621

theorem a_squared_plus_b_squared_less_than_c_squared 
  (a b c : Real) 
  (h : a^2 + b^2 + a * b + b * c + c * a < 0) : 
  a^2 + b^2 < c^2 := 
  by 
  sorry

end a_squared_plus_b_squared_less_than_c_squared_l2296_229621


namespace mean_daily_profit_l2296_229623

theorem mean_daily_profit 
  (mean_first_15_days : ℝ) 
  (mean_last_15_days : ℝ) 
  (n : ℝ) 
  (m1_days : ℝ) 
  (m2_days : ℝ) : 
  (mean_first_15_days = 245) → 
  (mean_last_15_days = 455) → 
  (m1_days = 15) → 
  (m2_days = 15) → 
  (n = 30) →
  (∀ P, P = (245 * 15 + 455 * 15) / 30) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end mean_daily_profit_l2296_229623


namespace total_blocks_l2296_229630

def initial_blocks := 2
def multiplier := 3
def father_blocks := multiplier * initial_blocks

theorem total_blocks :
  initial_blocks + father_blocks = 8 :=
by 
  -- skipping the proof with sorry
  sorry

end total_blocks_l2296_229630


namespace original_average_age_l2296_229639

variable (A : ℕ)
variable (N : ℕ := 2)
variable (new_avg_age : ℕ := 32)
variable (age_decrease : ℕ := 4)

theorem original_average_age :
  (A * N + new_avg_age * 2) / (N + 2) = A - age_decrease → A = 40 := 
by
  sorry

end original_average_age_l2296_229639


namespace total_chickens_after_purchase_l2296_229609

def initial_chickens : ℕ := 400
def percentage_died : ℕ := 40
def times_to_buy : ℕ := 10

noncomputable def chickens_died : ℕ := (percentage_died * initial_chickens) / 100
noncomputable def chickens_remaining : ℕ := initial_chickens - chickens_died
noncomputable def chickens_bought : ℕ := times_to_buy * chickens_died
noncomputable def total_chickens : ℕ := chickens_remaining + chickens_bought

theorem total_chickens_after_purchase : total_chickens = 1840 :=
by
  sorry

end total_chickens_after_purchase_l2296_229609


namespace original_cube_volume_l2296_229601

theorem original_cube_volume (V₂ : ℝ) (s : ℝ) (h₀ : V₂ = 216) (h₁ : (2 * s) ^ 3 = V₂) : s ^ 3 = 27 := by
  sorry

end original_cube_volume_l2296_229601


namespace find_common_difference_l2296_229663

theorem find_common_difference 
  (a : ℕ → ℝ)
  (a1 : a 1 = 5)
  (a25 : a 25 = 173)
  (h : ∀ n : ℕ, a (n+1) = a 1 + n * (a 2 - a 1)) : 
  a 2 - a 1 = 7 :=
by 
  sorry

end find_common_difference_l2296_229663


namespace triangle_area_is_zero_l2296_229670

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_sub (p1 p2 : Point3D) : Point3D := {
  x := p1.x - p2.x,
  y := p1.y - p2.y,
  z := p1.z - p2.z
}

def scalar_vector_mult (k : ℝ) (v : Point3D) : Point3D := {
  x := k * v.x,
  y := k * v.y,
  z := k * v.z
}

theorem triangle_area_is_zero : 
  let u := Point3D.mk 2 1 (-1)
  let v := Point3D.mk 5 4 1
  let w := Point3D.mk 11 10 5
  vector_sub w u = scalar_vector_mult 3 (vector_sub v u) →
-- If the points u, v, w are collinear, the area of the triangle formed by these points is zero:
  ∃ area : ℝ, area = 0 :=
by {
  sorry
}

end triangle_area_is_zero_l2296_229670


namespace compute_b_l2296_229603

theorem compute_b (x y b : ℝ) (h1 : 4 * x + 2 * y = b) (h2 : 3 * x + 7 * y = 3 * b) (hx : x = 3) : b = 66 :=
sorry

end compute_b_l2296_229603


namespace radical_product_l2296_229633

def fourth_root (x : ℝ) : ℝ := x ^ (1/4)
def third_root (x : ℝ) : ℝ := x ^ (1/3)
def square_root (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_product :
  fourth_root 81 * third_root 27 * square_root 9 = 27 := 
by
  sorry

end radical_product_l2296_229633


namespace green_pill_cost_l2296_229657

theorem green_pill_cost (p g : ℕ) (h1 : g = p + 1) (h2 : 14 * (p + g) = 546) : g = 20 :=
by
  sorry

end green_pill_cost_l2296_229657


namespace isosceles_triangle_perimeter_l2296_229656

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : (a = 3 ∨ a = 7)) (h2 : (b = 3 ∨ b = 7)) (h3 : a ≠ b) : 
  ∃ (c : ℕ), (a = 7 ∧ b = 3 ∧ c = 17) ∨ (a = 3 ∧ b = 7 ∧ c = 17) := 
by
  sorry

end isosceles_triangle_perimeter_l2296_229656


namespace product_of_triangle_areas_not_end_in_1988_l2296_229688

theorem product_of_triangle_areas_not_end_in_1988
  (a b c d : ℕ)
  (h1 : a * c = b * d)
  (hp : (a * b * c * d) = (a * c)^2)
  : ¬(∃ k : ℕ, (a * b * c * d) = 10000 * k + 1988) :=
sorry

end product_of_triangle_areas_not_end_in_1988_l2296_229688


namespace find_p_q_sum_l2296_229635

noncomputable def roots (r1 r2 r3 : ℝ) := (r1 + r2 + r3 = 11 ∧ r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3) ∧ 
                                         (∀ x : ℝ, x^3 - 11*x^2 + (r1 * r2 + r2 * r3 + r3 * r1) * x - r1 * r2 * r3 = 0)

theorem find_p_q_sum : ∃ (p q : ℝ), roots 2 4 5 → p + q = 78 :=
by
  sorry

end find_p_q_sum_l2296_229635


namespace central_angle_of_sector_l2296_229620

theorem central_angle_of_sector (R r n : ℝ) (h_lateral_area : 2 * π * r^2 = π * r * R) 
  (h_arc_length : (n * π * R) / 180 = 2 * π * r) : n = 180 :=
by 
  sorry

end central_angle_of_sector_l2296_229620


namespace find_other_root_l2296_229600

theorem find_other_root 
  (m : ℚ) 
  (h : 3 * 3^2 + m * 3 - 5 = 0) :
  (1 - 3) * (x : ℚ) = 0 :=
sorry

end find_other_root_l2296_229600


namespace find_b_l2296_229668

def h (x : ℝ) : ℝ := 5 * x + 7

theorem find_b (b : ℝ) : h b = 0 ↔ b = -7 / 5 := by
  sorry

end find_b_l2296_229668


namespace max_value_of_f_product_of_zeros_l2296_229662

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := Real.log x - a * x + b
 
theorem max_value_of_f (a b x1 x2 : ℝ) (h : 0 < a) (hz1 : Real.log x1 - a * x1 + b = 0) (hz2 : Real.log x2 - a * x2 + b = 0) : f (1 / a) a b = -Real.log a - 1 + b :=
by
  sorry

theorem product_of_zeros (a b x1 x2 : ℝ) (h : 0 < a) (hz1 : Real.log x1 - a * x1 + b = 0) (hz2 : Real.log x2 - a * x2 + b = 0) (hx_ne : x1 ≠ x2) : x1 * x2 < 1 / (a * a) :=
by
  sorry

end max_value_of_f_product_of_zeros_l2296_229662


namespace parabola_axis_of_symmetry_l2296_229613

theorem parabola_axis_of_symmetry (p : ℝ) :
  (∀ x : ℝ, x = 3 → -x^2 - p*x + 2 = -x^2 - (-6)*x + 2) → p = -6 :=
by sorry

end parabola_axis_of_symmetry_l2296_229613


namespace find_other_person_money_l2296_229610

noncomputable def other_person_money (mias_money : ℕ) : ℕ :=
  let x := (mias_money - 20) / 2
  x

theorem find_other_person_money (mias_money : ℕ) (h_mias_money : mias_money = 110) : 
  other_person_money mias_money = 45 := by
  sorry

end find_other_person_money_l2296_229610


namespace minimum_cuts_for_11_sided_polygons_l2296_229695

theorem minimum_cuts_for_11_sided_polygons (k : ℕ) :
  (∀ k, (11 * 252 + 3 * (k + 1 - 252) ≤ 4 * k + 4)) ∧ (252 ≤ (k + 1)) ∧ (4 * k + 4 ≥ 11 * 252 + 3 * (k + 1 - 252))
  ∧ (11 * 252 + 3 * (k + 1 - 252) ≤ 4 * k + 4) → (k ≥ 2012) ∧ (k = 2015) := 
sorry

end minimum_cuts_for_11_sided_polygons_l2296_229695


namespace common_tangents_count_l2296_229618

-- Define the first circle Q1
def Q1 (x y : ℝ) := x^2 + y^2 = 9

-- Define the second circle Q2
def Q2 (x y : ℝ) := (x - 3)^2 + (y - 4)^2 = 1

-- Prove the number of common tangents between Q1 and Q2
theorem common_tangents_count :
  ∃ n : ℕ, n = 4 ∧ ∀ x y : ℝ, Q1 x y ∧ Q2 x y -> n = 4 := sorry

end common_tangents_count_l2296_229618


namespace integer_roots_l2296_229678

-- Define the polynomial
def polynomial (x : ℤ) : ℤ := x^3 - 4 * x^2 - 7 * x + 10

-- Define the proof problem statement
theorem integer_roots :
  {x : ℤ | polynomial x = 0} = {1, -2, 5} :=
by
  sorry

end integer_roots_l2296_229678


namespace largest_possible_value_of_m_l2296_229616

theorem largest_possible_value_of_m :
  ∃ (X Y Z : ℕ), 0 ≤ X ∧ X ≤ 7 ∧ 0 ≤ Y ∧ Y ≤ 7 ∧ 0 ≤ Z ∧ Z ≤ 7 ∧
                 (64 * X + 8 * Y + Z = 475) ∧ 
                 (144 * Z + 12 * Y + X = 475) := 
sorry

end largest_possible_value_of_m_l2296_229616


namespace arithmetic_sequence_sum_l2296_229683

variable {S : ℕ → ℕ}

theorem arithmetic_sequence_sum (h1 : S 3 = 15) (h2 : S 9 = 153) : S 6 = 66 :=
sorry

end arithmetic_sequence_sum_l2296_229683


namespace set1_eq_set2_eq_set3_eq_set4_eq_set5_eq_l2296_229627

open Set

-- (1) The set of integers whose absolute value is not greater than 2
theorem set1_eq : { x : ℤ | |x| ≤ 2 } = {-2, -1, 0, 1, 2} := sorry

-- (2) The set of positive numbers less than 10 that are divisible by 3
theorem set2_eq : { x : ℕ | x < 10 ∧ x > 0 ∧ x % 3 = 0 } = {3, 6, 9} := sorry

-- (3) The set {x | x = |x|, x < 5, x ∈ 𝕫}
theorem set3_eq : { x : ℕ | x < 5 } = {0, 1, 2, 3, 4} := sorry

-- (4) The set {(x, y) | x + y = 6, x ∈ ℕ⁺, y ∈ ℕ⁺}
theorem set4_eq : { p : ℕ × ℕ | p.1 + p.2 = 6 ∧ p.1 > 0 ∧ p.2 > 0 } = {(1, 5), (2, 4), (3, 3), (4, 2), (5, 1) } := sorry

-- (5) The set {-3, -1, 1, 3, 5}
theorem set5_eq : {-3, -1, 1, 3, 5} = { x : ℤ | ∃ k : ℤ, x = 2 * k - 1 ∧ -1 ≤ k ∧ k ≤ 3 } := sorry

end set1_eq_set2_eq_set3_eq_set4_eq_set5_eq_l2296_229627


namespace not_directly_nor_inversely_proportional_A_not_directly_nor_inversely_proportional_D_l2296_229660

def equationA (x y : ℝ) : Prop := 2 * x + 3 * y = 5
def equationD (x y : ℝ) : Prop := 4 * x + 2 * y = 8

def directlyProportional (x y : ℝ) : Prop := ∃ k : ℝ, y = k * x
def inverselyProportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

theorem not_directly_nor_inversely_proportional_A (x y : ℝ) :
  equationA x y → ¬ (directlyProportional x y ∨ inverselyProportional x y) := 
sorry

theorem not_directly_nor_inversely_proportional_D (x y : ℝ) :
  equationD x y → ¬ (directlyProportional x y ∨ inverselyProportional x y) := 
sorry

end not_directly_nor_inversely_proportional_A_not_directly_nor_inversely_proportional_D_l2296_229660


namespace evaluate_expression_l2296_229677

theorem evaluate_expression : (3 / (2 - (4 / (-5)))) = (15 / 14) :=
by
  sorry

end evaluate_expression_l2296_229677


namespace largest_constant_inequality_l2296_229661

theorem largest_constant_inequality :
  ∃ C, C = 3 ∧
  (∀ (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ),
  (x₁ + x₂ + x₃ + x₄ + x₅ + x₆)^2 ≥ 
  C * (x₁ * (x₂ + x₃) + x₂ * (x₃ + x₄) + x₃ * (x₄ + x₅) + x₄ * (x₅ + x₆) + x₅ * (x₆ + x₁) + x₆ * (x₁ + x₂))) :=

sorry

end largest_constant_inequality_l2296_229661


namespace find_y_in_set_l2296_229680

noncomputable def arithmetic_mean (s : List ℝ) : ℝ :=
  s.sum / s.length

theorem find_y_in_set :
  ∀ (y : ℝ), arithmetic_mean [8, 15, 20, 5, y] = 12 ↔ y = 12 :=
by
  intro y
  unfold arithmetic_mean
  simp [List.sum_cons, List.length_cons]
  sorry

end find_y_in_set_l2296_229680


namespace motorcycle_licenses_count_l2296_229681

theorem motorcycle_licenses_count : (3 * (10 ^ 6) = 3000000) :=
by
  sorry -- Proof would go here.

end motorcycle_licenses_count_l2296_229681


namespace eggs_per_week_is_84_l2296_229664

-- Define the number of pens
def number_of_pens : Nat := 4

-- Define the number of emus per pen
def emus_per_pen : Nat := 6

-- Define the number of days in a week
def days_in_week : Nat := 7

-- Define the number of eggs per female emu per day
def eggs_per_female_emu_per_day : Nat := 1

-- Calculate the total number of emus
def total_emus : Nat := number_of_pens * emus_per_pen

-- Calculate the number of female emus
def female_emus : Nat := total_emus / 2

-- Calculate the number of eggs per day
def eggs_per_day : Nat := female_emus * eggs_per_female_emu_per_day

-- Calculate the number of eggs per week
def eggs_per_week : Nat := eggs_per_day * days_in_week

-- The theorem to prove
theorem eggs_per_week_is_84 : eggs_per_week = 84 := by
  sorry

end eggs_per_week_is_84_l2296_229664


namespace correct_option_is_C_l2296_229686

variable (a b : ℝ)

def option_A : Prop := (a - b) ^ 2 = a ^ 2 - b ^ 2
def option_B : Prop := a ^ 2 + a ^ 2 = a ^ 4
def option_C : Prop := (a ^ 2) ^ 3 = a ^ 6
def option_D : Prop := a ^ 2 * a ^ 2 = a ^ 6

theorem correct_option_is_C : option_C a :=
by
  sorry

end correct_option_is_C_l2296_229686


namespace find_number_l2296_229604

theorem find_number (N Q : ℕ) (h1 : N = 11 * Q) (h2 : Q + N + 11 = 71) : N = 55 :=
by {
  sorry
}

end find_number_l2296_229604


namespace find_a_l2296_229690

noncomputable def binomialExpansion (a : ℚ) (x : ℚ) := (x - a / x) ^ 6

theorem find_a (a : ℚ) (A : ℚ) (B : ℚ) (hA : A = 15 * a ^ 2) (hB : B = -20 * a ^ 3) (hB_value : B = 44) :
  a = -22 / 5 :=
by
  sorry -- skipping the proof

end find_a_l2296_229690


namespace measure_85_liters_l2296_229653

theorem measure_85_liters (C1 C2 C3 : ℕ) (capacity : ℕ) : 
  (C1 = 0 ∧ C2 = 0 ∧ C3 = 1 ∧ capacity = 85) → 
  (∃ weighings : ℕ, weighings ≤ 8 ∧ C1 = 85 ∨ C2 = 85 ∨ C3 = 85) :=
by 
  sorry

end measure_85_liters_l2296_229653


namespace multiple_of_michael_trophies_l2296_229692

-- Conditions
def michael_current_trophies : ℕ := 30
def michael_trophies_increse : ℕ := 100
def total_trophies_in_three_years : ℕ := 430

-- Proof statement
theorem multiple_of_michael_trophies (x : ℕ) :
  (michael_current_trophies + michael_trophies_increse) + (michael_current_trophies * x) = total_trophies_in_three_years → x = 10 := 
by
  sorry

end multiple_of_michael_trophies_l2296_229692


namespace plane_through_points_and_perpendicular_l2296_229673

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def plane_eq (A B C D : ℝ) (P : Point3D) : Prop :=
  A * P.x + B * P.y + C * P.z + D = 0

def vector_sub (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def cross_product (u v : Point3D) : Point3D :=
  ⟨u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x⟩

def is_perpendicular (normal1 normal2 : Point3D) : Prop :=
  normal1.x * normal2.x + normal1.y * normal2.y + normal1.z * normal2.z = 0

theorem plane_through_points_and_perpendicular
  (P1 P2 : Point3D)
  (A B C D : ℝ)
  (n_perp : Point3D)
  (normal1_eq : n_perp = ⟨2, -1, 4⟩)
  (eqn_given : plane_eq 2 (-1) 4 7 P1)
  (vec := vector_sub P1 P2)
  (n := cross_product vec n_perp)
  (eqn : plane_eq 11 (-10) (-9) (-33) P1) :
  (plane_eq 11 (-10) (-9) (-33) P2 ∧ is_perpendicular n n_perp) :=
sorry

end plane_through_points_and_perpendicular_l2296_229673


namespace clock_chime_time_l2296_229648

theorem clock_chime_time (t : ℕ) (h : t = 12) (k : 4 * (t / (4 - 1)) = 12) :
  12 * (t / (4 - 1)) - (12 - 1) * (t / (4 - 1)) = 44 :=
by {
  sorry
}

end clock_chime_time_l2296_229648


namespace select_7_jury_l2296_229693

theorem select_7_jury (students : Finset ℕ) (jury : Finset ℕ)
  (likes : ℕ → Finset ℕ) (h_students : students.card = 100)
  (h_jury : jury.card = 25) (h_likes : ∀ s ∈ students, (likes s).card = 10) :
  ∃ (selected_jury : Finset ℕ), selected_jury.card = 7 ∧ ∀ s ∈ students, ∃ j ∈ selected_jury, j ∈ (likes s) :=
sorry

end select_7_jury_l2296_229693


namespace walking_speed_l2296_229631

theorem walking_speed (total_time : ℕ) (distance : ℕ) (rest_interval : ℕ) (rest_time : ℕ) (rest_periods: ℕ) 
  (total_rest_time: ℕ) (total_walking_time: ℕ) (hours: ℕ) 
  (H1 : total_time = 332) 
  (H2 : distance = 50) 
  (H3 : rest_interval = 10) 
  (H4 : rest_time = 8)
  (H5 : rest_periods = distance / rest_interval - 1) 
  (H6 : total_rest_time = rest_periods * rest_time)
  (H7 : total_walking_time = total_time - total_rest_time) 
  (H8 : hours = total_walking_time / 60) : 
  (distance / hours) = 10 :=
by {
  -- proof omitted
  sorry
}

end walking_speed_l2296_229631


namespace positive_diff_solutions_l2296_229647

theorem positive_diff_solutions : 
  (∃ x₁ x₂ : ℝ, ( (9 - x₁^2 / 4)^(1/3) = -3) ∧ ((9 - x₂^2 / 4)^(1/3) = -3) ∧ ∃ (d : ℝ), d = |x₁ - x₂| ∧ d = 24) :=
by
  sorry

end positive_diff_solutions_l2296_229647


namespace first_bag_weight_l2296_229650

def weight_of_first_bag (initial_weight : ℕ) (second_bag : ℕ) (total_weight : ℕ) : ℕ :=
  total_weight - second_bag - initial_weight

theorem first_bag_weight : weight_of_first_bag 15 10 40 = 15 :=
by
  unfold weight_of_first_bag
  sorry

end first_bag_weight_l2296_229650


namespace box_volume_l2296_229642

theorem box_volume (l w h V : ℝ) 
  (h1 : l * w = 30) 
  (h2 : w * h = 18) 
  (h3 : l * h = 10) 
  : V = l * w * h → V = 90 :=
by 
  intro volume_eq
  sorry

end box_volume_l2296_229642


namespace pi_over_2_irrational_l2296_229667

def is_rational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop :=
  ¬ is_rational x

theorem pi_over_2_irrational : is_irrational (Real.pi / 2) :=
by sorry

end pi_over_2_irrational_l2296_229667


namespace find_u_value_l2296_229641

theorem find_u_value (u : ℤ) : ∀ (y : ℤ → ℤ), 
  (y 2 = 8) → (y 4 = 14) → (y 6 = 20) → 
  (∀ x, (x % 2 = 0) → (y (x + 2) = y x + 6)) → 
  y 18 = u → u = 56 :=
by
  intros y h2 h4 h6 pattern h18
  sorry

end find_u_value_l2296_229641


namespace inappropriate_character_choice_l2296_229676

-- Definitions and conditions
def is_main_character (c : String) : Prop := 
  c = "Gryphon" ∨ c = "Mock Turtle"

def characters : List String := ["Lobster", "Gryphon", "Mock Turtle"]

-- Theorem statement
theorem inappropriate_character_choice : 
  ¬ is_main_character "Lobster" :=
by 
  sorry

end inappropriate_character_choice_l2296_229676
