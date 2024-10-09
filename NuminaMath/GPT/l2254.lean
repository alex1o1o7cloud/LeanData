import Mathlib

namespace cost_of_first_variety_l2254_225412

theorem cost_of_first_variety (x : ℝ) (cost2 : ℝ) (cost_mix : ℝ) (ratio : ℝ) :
    cost2 = 8.75 →
    cost_mix = 7.50 →
    ratio = 0.625 →
    (x - cost_mix) / (cost2 - cost_mix) = ratio →
    x = 8.28125 := 
by
  intros h1 h2 h3 h4
  sorry

end cost_of_first_variety_l2254_225412


namespace cookies_per_child_l2254_225449

theorem cookies_per_child (total_cookies : ℕ) (adults : ℕ) (children : ℕ) (fraction_eaten_by_adults : ℚ) 
  (h1 : total_cookies = 120) (h2 : adults = 2) (h3 : children = 4) (h4 : fraction_eaten_by_adults = 1/3) :
  total_cookies * (1 - fraction_eaten_by_adults) / children = 20 := 
by
  sorry

end cookies_per_child_l2254_225449


namespace arithmetic_sequence_common_difference_l2254_225456

theorem arithmetic_sequence_common_difference 
  (a l S : ℕ) (h1 : a = 5) (h2 : l = 50) (h3 : S = 495) :
  (∃ d n : ℕ, l = a + (n-1) * d ∧ S = n * (a + l) / 2 ∧ d = 45 / 17) :=
by
  sorry

end arithmetic_sequence_common_difference_l2254_225456


namespace final_solution_percentage_l2254_225480

variable (initial_volume replaced_fraction : ℝ)
variable (initial_concentration replaced_concentration : ℝ)

noncomputable
def final_acid_percentage (initial_volume replaced_fraction initial_concentration replaced_concentration : ℝ) : ℝ :=
  let remaining_volume := initial_volume * (1 - replaced_fraction)
  let replaced_volume := initial_volume * replaced_fraction
  let remaining_acid := remaining_volume * initial_concentration
  let replaced_acid := replaced_volume * replaced_concentration
  let total_acid := remaining_acid + replaced_acid
  let final_volume := initial_volume
  (total_acid / final_volume) * 100

theorem final_solution_percentage :
  final_acid_percentage 100 0.5 0.5 0.3 = 40 :=
by
  sorry

end final_solution_percentage_l2254_225480


namespace arithmetic_sequence_middle_term_l2254_225409

theorem arithmetic_sequence_middle_term :
  ∀ (a b : ℕ) (z : ℕ), a = 9 → b = 81 → z = (a + b) / 2 → z = 45 :=
by
  intros a b z h_a h_b h_z
  rw [h_a, h_b] at h_z
  exact h_z

end arithmetic_sequence_middle_term_l2254_225409


namespace number_of_pupils_l2254_225469

theorem number_of_pupils (n : ℕ) : (83 - 63) / n = 1 / 2 → n = 40 :=
by
  intro h
  -- This is where the proof would go.
  sorry

end number_of_pupils_l2254_225469


namespace arccos_zero_eq_pi_div_two_l2254_225440

theorem arccos_zero_eq_pi_div_two : Real.arccos 0 = Real.pi / 2 :=
by
  sorry

end arccos_zero_eq_pi_div_two_l2254_225440


namespace Shara_borrowed_6_months_ago_l2254_225487

theorem Shara_borrowed_6_months_ago (X : ℝ) (h1 : ∃ n : ℕ, (X / 2 - 4 * 10 = 20) ∧ (X / 2 = n * 10)) :
  ∃ m : ℕ, m * 10 = X / 2 → m = 6 := 
sorry

end Shara_borrowed_6_months_ago_l2254_225487


namespace intersection_complement_A_B_l2254_225462

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | x < 1}

theorem intersection_complement_A_B : A ∩ (U \ B) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_complement_A_B_l2254_225462


namespace op_4_6_l2254_225422

-- Define the operation @ in Lean
def op (a b : ℕ) : ℤ := 2 * (a : ℤ)^2 - 2 * (b : ℤ)^2

-- State the theorem to prove
theorem op_4_6 : op 4 6 = -40 :=
by sorry

end op_4_6_l2254_225422


namespace apple_equals_pear_l2254_225448

-- Define the masses of the apple and pear.
variable (A G : ℝ)

-- The equilibrium condition on the balance scale.
axiom equilibrium_condition : A + 2 * G = 2 * A + G

-- Prove the mass of an apple equals the mass of a pear.
theorem apple_equals_pear (A G : ℝ) (h : A + 2 * G = 2 * A + G) : A = G :=
by
  -- Proof goes here, but we use sorry to indicate the proof's need.
  sorry

end apple_equals_pear_l2254_225448


namespace sequence_contains_composite_l2254_225415

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem sequence_contains_composite (a : ℕ → ℕ) (h : ∀ n, a (n+1) = 2 * a n + 1 ∨ a (n+1) = 2 * a n - 1) :
  ∃ n, is_composite (a n) :=
sorry

end sequence_contains_composite_l2254_225415


namespace cube_sphere_surface_area_l2254_225427

open Real

noncomputable def cube_edge_length := 1
noncomputable def cube_space_diagonal := sqrt 3
noncomputable def sphere_radius := cube_space_diagonal / 2
noncomputable def sphere_surface_area := 4 * π * (sphere_radius ^ 2)

theorem cube_sphere_surface_area :
  sphere_surface_area = 3 * π :=
by
  sorry

end cube_sphere_surface_area_l2254_225427


namespace max_value_m_l2254_225473

theorem max_value_m (a b : ℝ) (ha : a > 0) (hb : b > 0) (m : ℝ)
  (h : (2 / a) + (1 / b) ≥ m / (2 * a + b)) : m ≤ 9 :=
sorry

end max_value_m_l2254_225473


namespace ratio_of_white_socks_l2254_225457

theorem ratio_of_white_socks 
  (total_socks : ℕ) (blue_socks : ℕ)
  (h_total_socks : total_socks = 180)
  (h_blue_socks : blue_socks = 60) :
  (total_socks - blue_socks) * 3 = total_socks * 2 :=
by
  sorry

end ratio_of_white_socks_l2254_225457


namespace arman_is_6_times_older_than_sister_l2254_225437

def sisterWasTwoYearsOldFourYearsAgo := 2
def yearsAgo := 4
def armansAgeInFourYears := 40

def currentAgeOfSister := sisterWasTwoYearsOldFourYearsAgo + yearsAgo
def currentAgeOfArman := armansAgeInFourYears - yearsAgo

theorem arman_is_6_times_older_than_sister :
  currentAgeOfArman = 6 * currentAgeOfSister :=
by
  sorry

end arman_is_6_times_older_than_sister_l2254_225437


namespace find_p_l2254_225426

/-- Given conditions about the coordinates of points on a line, we want to prove p = 3. -/
theorem find_p (m n p : ℝ) 
  (h1 : m = n / 3 - 2 / 5)
  (h2 : m + p = (n + 9) / 3 - 2 / 5) 
  : p = 3 := by 
  sorry

end find_p_l2254_225426


namespace trapezoid_area_l2254_225447

-- Definitions based on the given conditions
variable (BD AC h : ℝ)
variable (BD_perpendicular_AC : BD * AC = 0)
variable (BD_val : BD = 13)
variable (h_val : h = 12)

-- Statement of the theorem to prove the area of the trapezoid
theorem trapezoid_area (BD AC h : ℝ)
  (BD_perpendicular_AC : BD * AC = 0)
  (BD_val : BD = 13)
  (h_val : h = 12) :
  0.5 * 13 * 12 = 1014 / 5 := sorry

end trapezoid_area_l2254_225447


namespace decreasing_interval_range_of_a_l2254_225451

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem decreasing_interval :
  (∀ x > 0, deriv f x = 1 + log x) →
  { x : ℝ | 0 < x ∧ x < 1/e } = { x | 0 < x ∧ deriv f x < 0 } :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≥ -x^2 + a * x - 6) →
  a ≤ 5 + log 2 :=
sorry

end decreasing_interval_range_of_a_l2254_225451


namespace box_distribution_l2254_225496

theorem box_distribution (A P S : ℕ) (h : A + P + S = 22) : A ≥ 8 ∨ P ≥ 8 ∨ S ≥ 8 := 
by 
-- The next step is to use proof by contradiction, assuming the opposite.
sorry

end box_distribution_l2254_225496


namespace angle_CBD_is_48_degrees_l2254_225491

theorem angle_CBD_is_48_degrees :
  ∀ (A B D C : Type) (α β γ δ : ℝ), 
    α = 28 ∧ β = 46 ∧ C ∈ [B, D] ∧ γ = 30 → 
    δ = 48 := 
by 
  sorry

end angle_CBD_is_48_degrees_l2254_225491


namespace range_of_x_l2254_225465

theorem range_of_x (x : ℝ) (h1 : (x + 2) * (x - 3) ≤ 0) (h2 : |x + 1| ≥ 2) : 
  1 ≤ x ∧ x ≤ 3 :=
sorry

end range_of_x_l2254_225465


namespace relationship_above_l2254_225410

noncomputable def a : ℝ := Real.log 5 / Real.log 2
noncomputable def b : ℝ := Real.log 15 / (2 * Real.log 2)
noncomputable def c : ℝ := Real.sqrt 2

theorem relationship_above (ha : a = Real.log 5 / Real.log 2) 
                           (hb : b = Real.log 15 / (2 * Real.log 2))
                           (hc : c = Real.sqrt 2) : a > b ∧ b > c :=
by
  sorry

end relationship_above_l2254_225410


namespace sales_price_calculation_l2254_225476

variables (C S : ℝ)
def gross_profit := 1.25 * C
def gross_profit_value := 30

theorem sales_price_calculation 
  (h1: gross_profit C = 30) :
  S = 54 :=
sorry

end sales_price_calculation_l2254_225476


namespace smallest_integer_divisible_20_perfect_cube_square_l2254_225401

theorem smallest_integer_divisible_20_perfect_cube_square :
  ∃ (n : ℕ), n > 0 ∧ n % 20 = 0 ∧ (∃ (m : ℕ), n^2 = m^3) ∧ (∃ (k : ℕ), n^3 = k^2) ∧ n = 1000000 :=
by {
  sorry -- Replace this placeholder with an appropriate proof.
}

end smallest_integer_divisible_20_perfect_cube_square_l2254_225401


namespace shift_right_symmetric_l2254_225418

open Real

/-- Given the function y = sin(2x + π/3), after shifting the graph of the function right
    by φ (0 < φ < π/2) units, the resulting graph is symmetric about the y-axis.
    Prove that the value of φ is 5π/12.
-/
theorem shift_right_symmetric (φ : ℝ) (hφ₁ : 0 < φ) (hφ₂ : φ < π / 2)
  (h_sym : ∃ k : ℤ, -2 * φ + π / 3 = k * π + π / 2) : φ = 5 * π / 12 :=
sorry

end shift_right_symmetric_l2254_225418


namespace master_efficiency_comparison_l2254_225464

theorem master_efficiency_comparison (z_parts : ℕ) (z_hours : ℕ) (l_parts : ℕ) (l_hours : ℕ)
    (hz : z_parts = 5) (hz_time : z_hours = 8)
    (hl : l_parts = 3) (hl_time : l_hours = 4) :
    (z_parts / z_hours : ℚ) < (l_parts / l_hours : ℚ) → false :=
by
  -- This is a placeholder for the proof, which is not needed as per the instructions.
  sorry

end master_efficiency_comparison_l2254_225464


namespace sufficient_condition_for_p_l2254_225460

theorem sufficient_condition_for_p (m : ℝ) (h : 1 < m) : ∀ x : ℝ, x^2 - 2 * x + m > 0 :=
sorry

end sufficient_condition_for_p_l2254_225460


namespace original_fraction_l2254_225423

def fraction (a b c : ℕ) := 10 * a + b / 10 * c + a

theorem original_fraction (a b c : ℕ) (ha: a < 10) (hb : b < 10) (hc : c < 10) (h : b ≠ c):
  (fraction a b c = b / c) →
  (fraction 6 4 1 = 64 / 16) ∨ (fraction 9 8 4 = 98 / 49) ∨
  (fraction 9 5 1 = 95 / 19) ∨ (fraction 6 5 2 = 65 / 26) :=
sorry

end original_fraction_l2254_225423


namespace combined_income_is_16800_l2254_225414

-- Given conditions
def ErnieOldIncome : ℕ := 6000
def ErnieCurrentIncome : ℕ := (4 * ErnieOldIncome) / 5
def JackCurrentIncome : ℕ := 2 * ErnieOldIncome

-- Proof that their combined income is $16800
theorem combined_income_is_16800 : ErnieCurrentIncome + JackCurrentIncome = 16800 := by
  sorry

end combined_income_is_16800_l2254_225414


namespace conic_section_is_hyperbola_l2254_225453

theorem conic_section_is_hyperbola : 
  ∀ (x y : ℝ), x^2 + 2 * x - 8 * y^2 = 0 → (∃ a b h k : ℝ, (x + 1)^2 / a^2 - (y - 0)^2 / b^2 = 1) := 
by 
  intros x y h_eq;
  sorry

end conic_section_is_hyperbola_l2254_225453


namespace julia_drove_miles_l2254_225494

theorem julia_drove_miles :
  ∀ (daily_rental_cost cost_per_mile total_paid : ℝ),
    daily_rental_cost = 29 →
    cost_per_mile = 0.08 →
    total_paid = 46.12 →
    total_paid - daily_rental_cost = cost_per_mile * 214 :=
by
  intros _ _ _ d_cost_eq cpm_eq tp_eq
  -- calculation and proof steps will be filled here
  sorry

end julia_drove_miles_l2254_225494


namespace expand_product_eq_l2254_225483

theorem expand_product_eq :
  (∀ (x : ℤ), (x^3 - 3 * x^2 + 3 * x - 1) * (x^2 + 3 * x + 3) = x^5 - 3 * x^3 - x^2 + 3 * x) :=
by
  intro x
  sorry

end expand_product_eq_l2254_225483


namespace students_with_exactly_two_skills_l2254_225486

-- Definitions based on the conditions:
def total_students : ℕ := 150
def students_can_write : ℕ := total_students - 60 -- 150 - 60 = 90
def students_can_direct : ℕ := total_students - 90 -- 150 - 90 = 60
def students_can_produce : ℕ := total_students - 40 -- 150 - 40 = 110

-- The theorem statement
theorem students_with_exactly_two_skills :
  students_can_write + students_can_direct + students_can_produce - total_students = 110 := 
sorry

end students_with_exactly_two_skills_l2254_225486


namespace triangle_evaluation_l2254_225425

def triangle (a b : ℤ) : ℤ := a^2 - 2 * b

theorem triangle_evaluation : triangle (-2) (triangle 3 2) = -6 := by
  sorry

end triangle_evaluation_l2254_225425


namespace gcd_two_5_digit_integers_l2254_225444

theorem gcd_two_5_digit_integers (a b : ℕ) 
  (h1 : 10^4 ≤ a ∧ a < 10^5)
  (h2 : 10^4 ≤ b ∧ b < 10^5)
  (h3 : 10^8 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^9) :
  Nat.gcd a b < 10^2 :=
by
  sorry  -- Skip the proof

end gcd_two_5_digit_integers_l2254_225444


namespace sum_a_b_is_nine_l2254_225490

theorem sum_a_b_is_nine (a b : ℤ) (h1 : a > b) (h2 : b > 0) 
    (h3 : (b + 2 - a)^2 + (a - b)^2 + (b + 2 + a)^2 + (a + b)^2 = 324) 
    (h4 : ∃ a' b', a' = a ∧ b' = b ∧ (b + 2 - a) * 1 = -(b + 2 - a)) : 
  a + b = 9 :=
sorry

end sum_a_b_is_nine_l2254_225490


namespace monic_polynomial_roots_l2254_225436

theorem monic_polynomial_roots (r1 r2 r3 : ℝ) (h : ∀ x : ℝ, x^3 - 4*x^2 + 5 = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) :
  ∀ x : ℝ, x^3 - 12*x^2 + 135 = 0 ↔ x = 3*r1 ∨ x = 3*r2 ∨ x = 3*r3 :=
by
  sorry

end monic_polynomial_roots_l2254_225436


namespace ratio_between_two_numbers_l2254_225485

noncomputable def first_number : ℕ := 48
noncomputable def lcm_value : ℕ := 432
noncomputable def second_number : ℕ := 9 * 24  -- Derived from the given conditions in the problem

def ratio (a b : ℕ) : ℚ := (a : ℚ) / (b : ℚ)

theorem ratio_between_two_numbers 
  (A B : ℕ) 
  (hA : A = first_number) 
  (hLCM : Nat.lcm A B = lcm_value) 
  (hB : B = 9 * 24) : 
  ratio A B = 1 / 4.5 :=
by
  -- Proof would go here
  sorry

end ratio_between_two_numbers_l2254_225485


namespace decreasing_range_of_a_l2254_225497

noncomputable def f (a x : ℝ) : ℝ := (Real.sqrt (2 - a * x)) / (a - 1)

theorem decreasing_range_of_a (a : ℝ) :
    (∀ x y : ℝ, 0 ≤ x → x ≤ 1/2 → 0 ≤ y → y ≤ 1/2 → x < y → f a y < f a x) ↔ (a < 0 ∨ (1 < a ∧ a ≤ 4)) :=
by
  sorry

end decreasing_range_of_a_l2254_225497


namespace problem_statement_l2254_225435

def U : Set ℤ := {x | True}
def A : Set ℤ := {-1, 1, 3, 5, 7, 9}
def B : Set ℤ := {-1, 5, 7}
def complement (B : Set ℤ) : Set ℤ := {x | x ∉ B}

theorem problem_statement : (A ∩ (complement B)) = {1, 3, 9} :=
by {
  sorry
}

end problem_statement_l2254_225435


namespace shobha_current_age_l2254_225402

variable (S B : ℕ)
variable (h_ratio : 4 * B = 3 * S)
variable (h_future_age : S + 6 = 26)

theorem shobha_current_age : B = 15 :=
by
  sorry

end shobha_current_age_l2254_225402


namespace greatest_integer_solution_l2254_225446

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 12 * n + 28 ≤ 0) : 6 ≤ n :=
sorry

end greatest_integer_solution_l2254_225446


namespace amy_money_left_l2254_225443

def amount_left (initial_amount doll_price board_game_price comic_book_price doll_qty board_game_qty comic_book_qty board_game_discount sales_tax_rate : ℝ) :
    ℝ :=
  let cost_dolls := doll_qty * doll_price
  let cost_board_games := board_game_qty * board_game_price
  let cost_comic_books := comic_book_qty * comic_book_price
  let discounted_cost_board_games := cost_board_games * (1 - board_game_discount)
  let total_cost_before_tax := cost_dolls + discounted_cost_board_games + cost_comic_books
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  initial_amount - total_cost_after_tax

theorem amy_money_left :
  amount_left 100 1.25 12.75 3.50 3 2 4 0.10 0.08 = 56.04 :=
by
  sorry

end amy_money_left_l2254_225443


namespace noah_large_paintings_last_month_l2254_225424

-- problem definitions
def large_painting_price : ℕ := 60
def small_painting_price : ℕ := 30
def small_paintings_sold_last_month : ℕ := 4
def sales_this_month : ℕ := 1200

-- to be proven
theorem noah_large_paintings_last_month (L : ℕ) (last_month_sales_eq : large_painting_price * L + small_painting_price * small_paintings_sold_last_month = S) 
   (this_month_sales_eq : 2 * S = sales_this_month) :
  L = 8 :=
sorry

end noah_large_paintings_last_month_l2254_225424


namespace min_value_one_div_a_plus_one_div_b_l2254_225463

theorem min_value_one_div_a_plus_one_div_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (1 / a + 1 / b) ≥ 4 :=
by
  sorry

end min_value_one_div_a_plus_one_div_b_l2254_225463


namespace prove_non_negative_axbycz_l2254_225477

variable {a b c x y z : ℝ}

theorem prove_non_negative_axbycz
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) :
  a * x + b * y + c * z ≥ 0 := 
sorry

end prove_non_negative_axbycz_l2254_225477


namespace distinct_real_roots_find_k_values_l2254_225459

-- Question 1: Prove the equation has two distinct real roots
theorem distinct_real_roots (k : ℝ) : 
  (2 * k + 1) ^ 2 - 4 * (k ^ 2 + k) > 0 :=
  by sorry

-- Question 2: Find the values of k when triangle ABC is a right triangle
theorem find_k_values (k : ℝ) : 
  (k = 3 ∨ k = 12) ↔ 
  (∃ (AB AC : ℝ), 
    AB ≠ AC ∧ AB = k ∧ AC = k + 1 ∧ (AB^2 + AC^2 = 5^2 ∨ AC^2 + 5^2 = AB^2)) :=
  by sorry

end distinct_real_roots_find_k_values_l2254_225459


namespace min_value_expr_l2254_225445

theorem min_value_expr : ∃ x : ℝ, (15 - x) * (9 - x) * (15 + x) * (9 + x) = -5184 :=
by
  sorry

end min_value_expr_l2254_225445


namespace invalid_votes_percentage_is_correct_l2254_225406

-- Definitions based on conditions
def total_votes : ℕ := 5500
def other_candidate_votes : ℕ := 1980
def valid_votes_percentage_other : ℚ := 0.45

-- Derived values
def valid_votes : ℚ := other_candidate_votes / valid_votes_percentage_other
def invalid_votes : ℚ := total_votes - valid_votes
def invalid_votes_percentage : ℚ := (invalid_votes / total_votes) * 100

-- Proof statement
theorem invalid_votes_percentage_is_correct :
  invalid_votes_percentage = 20 := sorry

end invalid_votes_percentage_is_correct_l2254_225406


namespace determine_phi_l2254_225416

theorem determine_phi (f : ℝ → ℝ) (φ : ℝ): 
  (∀ x : ℝ, f x = 2 * Real.sin (2 * x + 3 * φ)) ∧ 
  (∀ x : ℝ, f (-x) = -f x) → 
  (∃ k : ℤ, φ = k * Real.pi / 3) :=
by 
  sorry

end determine_phi_l2254_225416


namespace radius_of_sphere_l2254_225433

theorem radius_of_sphere {r x : ℝ} (h1 : 15^2 + x^2 = r^2) (h2 : r = x + 12) :
    r = 123 / 8 :=
  by
  sorry

end radius_of_sphere_l2254_225433


namespace f_value_2009_l2254_225484

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_2009
    (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
    (h2 : f 0 ≠ 0) :
    f 2009 = 1 :=
sorry

end f_value_2009_l2254_225484


namespace students_play_both_l2254_225470

variable (students total_students football cricket neither : ℕ)
variable (H1 : total_students = 420)
variable (H2 : football = 325)
variable (H3 : cricket = 175)
variable (H4 : neither = 50)
  
theorem students_play_both (H1 : total_students = 420) (H2 : football = 325) 
    (H3 : cricket = 175) (H4 : neither = 50) : 
    students = 325 + 175 - (420 - 50) :=
by sorry

end students_play_both_l2254_225470


namespace triangle_no_real_solution_l2254_225454

theorem triangle_no_real_solution (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (habc : a + b > c ∧ b + c > a ∧ c + a > b) :
  ¬ (∀ x, x^2 - 2 * b * x + 2 * a * c = 0 ∧
       x^2 - 2 * c * x + 2 * a * b = 0 ∧
       x^2 - 2 * a * x + 2 * b * c = 0) :=
by
  intro H
  sorry

end triangle_no_real_solution_l2254_225454


namespace tile_difference_l2254_225431

theorem tile_difference :
  let initial_blue_tiles := 20
  let initial_green_tiles := 15
  let first_border_tiles := 18
  let second_border_tiles := 18
  let total_green_tiles := initial_green_tiles + first_border_tiles + second_border_tiles
  let total_blue_tiles := initial_blue_tiles
  total_green_tiles - total_blue_tiles = 31 := 
by
  sorry

end tile_difference_l2254_225431


namespace grain_spilled_correct_l2254_225411

variable (original_grain : ℕ) (remaining_grain : ℕ) (grain_spilled : ℕ)

theorem grain_spilled_correct : 
  original_grain = 50870 → remaining_grain = 918 → grain_spilled = original_grain - remaining_grain → grain_spilled = 49952 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end grain_spilled_correct_l2254_225411


namespace max_x_squared_plus_y_squared_l2254_225420

theorem max_x_squared_plus_y_squared (x y : ℝ) 
  (h : 3 * x^2 + 2 * y^2 = 2 * x) : x^2 + y^2 ≤ 4 / 9 :=
sorry

end max_x_squared_plus_y_squared_l2254_225420


namespace range_of_m_l2254_225472

noncomputable def f (a x : ℝ) := a * (x^2 + 1) + Real.log x

theorem range_of_m (a m : ℝ) (h₁ : a ∈ Set.Ioo (-4 : ℝ) (-2))
  (h₂ : ∀ x ∈ Set.Icc (1 : ℝ) (3), ma - f a x > a^2) : m ≤ -2 := 
sorry

end range_of_m_l2254_225472


namespace smallest_value_of_y_l2254_225489

theorem smallest_value_of_y : 
  (∃ y : ℝ, 6 * y^2 - 41 * y + 55 = 0 ∧ ∀ z : ℝ, 6 * z^2 - 41 * z + 55 = 0 → y ≤ z) →
  ∃ y : ℝ, y = 2.5 :=
by sorry

end smallest_value_of_y_l2254_225489


namespace sum_of_operations_l2254_225450

def operation (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem sum_of_operations : operation 12 5 + operation 8 3 = 174 := by
  sorry

end sum_of_operations_l2254_225450


namespace quadratic_real_roots_and_value_l2254_225407

theorem quadratic_real_roots_and_value (m x1 x2: ℝ) 
  (h1: ∀ (a: ℝ), ∃ (b c: ℝ), a = x^2 - 4 * x - 2 * m + 5) 
  (h2: x1 * x2 + x1 + x2 = m^2 + 6):
  m ≥ 1/2 ∧ m = 1 := 
by
  sorry

end quadratic_real_roots_and_value_l2254_225407


namespace phi_cannot_be_chosen_l2254_225481

theorem phi_cannot_be_chosen (θ φ : ℝ) (hθ : -π/2 < θ ∧ θ < π/2) (hφ : 0 < φ ∧ φ < π)
  (h1 : 3 * Real.sin θ = 3 * Real.sqrt 2 / 2) 
  (h2 : 3 * Real.sin (-2*φ + θ) = 3 * Real.sqrt 2 / 2) : φ ≠ 5*π/4 :=
by
  sorry

end phi_cannot_be_chosen_l2254_225481


namespace right_triangle_perimeter_l2254_225482

theorem right_triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) (perimeter : ℝ)
  (h1 : area = 180) 
  (h2 : leg1 = 30) 
  (h3 : (1 / 2) * leg1 * leg2 = area)
  (h4 : hypotenuse^2 = leg1^2 + leg2^2)
  (h5 : leg2 = 12) 
  (h6 : hypotenuse = 2 * Real.sqrt 261) :
  perimeter = 42 + 2 * Real.sqrt 261 :=
by
  sorry

end right_triangle_perimeter_l2254_225482


namespace hyperbola_equation_l2254_225408

theorem hyperbola_equation
  (a b m n e e' c' : ℝ)
  (h1 : 2 * a^2 + b^2 = 2)
  (h2 : e * e' = 1)
  (h_c : c' = e * m)
  (h_b : b^2 = m^2 - n^2)
  (h_e : e = n / m) : 
  y^2 - x^2 = 2 := 
sorry

end hyperbola_equation_l2254_225408


namespace x_squared_plus_y_squared_l2254_225439

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : (1/x) + (1/y) = 5) 
  (h2 : x * y + x + y = 11) : 
  x^2 + y^2 = 2893 / 36 := 
by 
  sorry

end x_squared_plus_y_squared_l2254_225439


namespace regular_train_passes_by_in_4_seconds_l2254_225417

theorem regular_train_passes_by_in_4_seconds
    (l_high_speed : ℕ)
    (l_regular : ℕ)
    (t_observed : ℕ)
    (v_relative : ℕ)
    (h_length_high_speed : l_high_speed = 80)
    (h_length_regular : l_regular = 100)
    (h_time_observed : t_observed = 5)
    (h_velocity : v_relative = l_regular / t_observed) :
    v_relative * 4 = l_high_speed :=
by
  sorry

end regular_train_passes_by_in_4_seconds_l2254_225417


namespace unique_point_value_l2254_225430

noncomputable def unique_point_condition : Prop :=
  ∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + 12 = 0

theorem unique_point_value (d : ℝ) : unique_point_condition ↔ d = 12 := 
sorry

end unique_point_value_l2254_225430


namespace train_passes_tree_in_28_seconds_l2254_225468

def km_per_hour_to_meter_per_second (km_per_hour : ℕ) : ℕ :=
  km_per_hour * 1000 / 3600

def pass_tree_time (length : ℕ) (speed_kmh : ℕ) : ℕ :=
  length / (km_per_hour_to_meter_per_second speed_kmh)

theorem train_passes_tree_in_28_seconds :
  pass_tree_time 490 63 = 28 :=
by
  sorry

end train_passes_tree_in_28_seconds_l2254_225468


namespace range_of_d_l2254_225421

theorem range_of_d (d : ℝ) : (∃ x : ℝ, |2017 - x| + |2018 - x| ≤ d) ↔ d ≥ 1 :=
sorry

end range_of_d_l2254_225421


namespace find_six_y_minus_four_squared_l2254_225478

theorem find_six_y_minus_four_squared (y : ℝ) (h : 3 * y^2 + 6 = 5 * y + 15) :
  (6 * y - 4)^2 = 134 :=
by
  sorry

end find_six_y_minus_four_squared_l2254_225478


namespace problem_l2254_225499

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

def p : Prop := ∀ x : ℝ, x ≠ 0 → f x ≥ 4 ∧ (∃ x : ℝ, x > 0 ∧ f x = 4)

def q : Prop := ∀ (A B C : ℝ) (a b c : ℝ),
  A > B ↔ a > b

theorem problem : (¬p) ∧ q :=
sorry

end problem_l2254_225499


namespace simplify_expression_l2254_225405

theorem simplify_expression (x y : ℝ) : 
  (5 * x ^ 2 - 3 * x + 2) * (107 - 107) + (7 * y ^ 2 + 4 * y - 1) * (93 - 93) = 0 := 
by 
  sorry

end simplify_expression_l2254_225405


namespace divides_expression_l2254_225403

theorem divides_expression (y : ℕ) (hy : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end divides_expression_l2254_225403


namespace karen_box_crayons_l2254_225467

theorem karen_box_crayons (judah_crayons : ℕ) (gilbert_crayons : ℕ) (beatrice_crayons : ℕ) (karen_crayons : ℕ)
  (h1 : judah_crayons = 8)
  (h2 : gilbert_crayons = 4 * judah_crayons)
  (h3 : beatrice_crayons = 2 * gilbert_crayons)
  (h4 : karen_crayons = 2 * beatrice_crayons) :
  karen_crayons = 128 :=
by
  sorry

end karen_box_crayons_l2254_225467


namespace depth_of_well_l2254_225442

noncomputable def volume_of_cylinder (radius : ℝ) (depth : ℝ) : ℝ :=
  Real.pi * radius^2 * depth

theorem depth_of_well (volume depth : ℝ) (r : ℝ) : 
  r = 1 ∧ volume = 25.132741228718345 ∧ 2 * r = 2 → depth = 8 :=
by
  intros h
  sorry

end depth_of_well_l2254_225442


namespace find_omega_find_period_and_intervals_find_solution_set_l2254_225441

noncomputable def omega_condition (ω : ℝ) :=
  0 < ω ∧ ω < 2

noncomputable def function_fx (ω : ℝ) (x : ℝ) := 
  3 * Real.sin (2 * ω * x + Real.pi / 3)

noncomputable def center_of_symmetry_condition (ω : ℝ) := 
  function_fx ω (-Real.pi / 6) = 0

noncomputable def period_condition (ω : ℝ) :=
  Real.pi / abs ω

noncomputable def intervals_of_increase (ω : ℝ) (x : ℝ) : Prop :=
  ∃ k : ℤ, ((Real.pi / 12 + k * Real.pi) ≤ x) ∧ (x < (5 * Real.pi / 12 + k * Real.pi))

noncomputable def solution_set_fx_ge_half (x : ℝ) : Prop :=
  ∃ k : ℤ, (Real.pi / 12 + k * Real.pi) ≤ x ∧ (x ≤ 5 * Real.pi / 12 + k * Real.pi)

theorem find_omega : ∀ ω : ℝ, omega_condition ω ∧ center_of_symmetry_condition ω → omega = 1 := sorry

theorem find_period_and_intervals : 
  ∀ ω : ℝ, omega_condition ω ∧ (ω = 1) → period_condition ω = Real.pi :=
sorry

theorem find_solution_set :
  ∀ ω : ℝ, omega_condition ω ∧ (ω = 1) → (∀ x, solution_set_fx_ge_half x) :=
sorry

end find_omega_find_period_and_intervals_find_solution_set_l2254_225441


namespace sqrt_sum_inequality_l2254_225458

open Real

theorem sqrt_sum_inequality (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 2) : 
  sqrt (2 * x + 1) + sqrt (2 * y + 1) + sqrt (2 * z + 1) ≤ sqrt 21 :=
sorry

end sqrt_sum_inequality_l2254_225458


namespace total_shaded_area_is_71_l2254_225498

-- Define the dimensions of the first rectangle
def rect1_length : ℝ := 4
def rect1_width : ℝ := 12

-- Define the dimensions of the second rectangle
def rect2_length : ℝ := 5
def rect2_width : ℝ := 7

-- Define the dimensions of the overlap area
def overlap_length : ℝ := 3
def overlap_width : ℝ := 4

-- Define the area calculation
def area (length width : ℝ) : ℝ := length * width

-- Calculate the areas of the rectangles and the overlap
def rect1_area : ℝ := area rect1_length rect1_width
def rect2_area : ℝ := area rect2_length rect2_width
def overlap_area : ℝ := area overlap_length overlap_width

-- Total shaded area calculation
def total_shaded_area : ℝ := rect1_area + rect2_area - overlap_area

-- Proof statement to show that the total shaded area is 71 square units
theorem total_shaded_area_is_71 : total_shaded_area = 71 := by
  sorry

end total_shaded_area_is_71_l2254_225498


namespace six_digit_palindromes_count_l2254_225471

open Nat

theorem six_digit_palindromes_count :
  let digits := {d | 0 ≤ d ∧ d ≤ 9}
  let a_digits := {a | 1 ≤ a ∧ a ≤ 9}
  let b_digits := digits
  let c_digits := digits
  ∃ (total : ℕ), (∀ a ∈ a_digits, ∀ b ∈ b_digits, ∀ c ∈ c_digits, True) → total = 900 :=
by
  sorry

end six_digit_palindromes_count_l2254_225471


namespace certain_percentage_l2254_225461

variable {x p : ℝ}

theorem certain_percentage (h1 : 0.40 * x = 160) : p * x = 200 ↔ p = 0.5 := 
by
  sorry

end certain_percentage_l2254_225461


namespace lara_sees_leo_for_six_minutes_l2254_225474

-- Define constants for speeds and initial distances
def lara_speed : ℕ := 60
def leo_speed : ℕ := 40
def initial_distance : ℕ := 1
def time_to_minutes (t : ℚ) : ℚ := t * 60
-- Define the condition that proves Lara can see Leo for 6 minutes
theorem lara_sees_leo_for_six_minutes :
  lara_speed > leo_speed ∧
  initial_distance > 0 ∧
  (initial_distance : ℚ) / (lara_speed - leo_speed) * 2 = (6 : ℚ) / 60 :=
by
  sorry

end lara_sees_leo_for_six_minutes_l2254_225474


namespace find_particular_number_l2254_225400

theorem find_particular_number (x : ℤ) (h : x - 29 + 64 = 76) : x = 41 :=
by
  sorry

end find_particular_number_l2254_225400


namespace simplify_and_sum_of_exponents_l2254_225404

-- Define the given expression
def radicand (x y z : ℝ) : ℝ := 40 * x ^ 5 * y ^ 7 * z ^ 9

-- Define what cube root stands for
noncomputable def cbrt (a : ℝ) := a ^ (1 / 3 : ℝ)

-- Define the simplified expression outside the cube root
noncomputable def simplified_outside_exponents (x y z : ℝ) : ℝ := x * y * z ^ 3

-- Define the sum of the exponents outside the radical
def sum_of_exponents_outside (x y z : ℝ) : ℝ := (1 + 1 + 3 : ℝ)

-- Statement of the problem in Lean
theorem simplify_and_sum_of_exponents (x y z : ℝ) :
  sum_of_exponents_outside x y z = 5 :=
by 
  sorry

end simplify_and_sum_of_exponents_l2254_225404


namespace prove_a2_b2_c2_zero_l2254_225479

theorem prove_a2_b2_c2_zero (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c = 0) (h5 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) : a^2 + b^2 + c^2 = 0 := 
by 
  sorry

end prove_a2_b2_c2_zero_l2254_225479


namespace sin_expression_value_l2254_225419

theorem sin_expression_value (α : ℝ) (h : Real.cos (α + π / 5) = 4 / 5) :
  Real.sin (2 * α + 9 * π / 10) = 7 / 25 :=
sorry

end sin_expression_value_l2254_225419


namespace polar_to_cartesian_l2254_225475

theorem polar_to_cartesian (θ ρ : ℝ) (h : ρ = 2 * Real.cos θ) :
  ∃ x y : ℝ, (x=ρ*Real.cos θ ∧ y=ρ*Real.sin θ) ∧ (x-1)^2 + y^2 = 1 :=
by
  sorry

end polar_to_cartesian_l2254_225475


namespace sin_minus_cos_eq_one_l2254_225492

theorem sin_minus_cos_eq_one (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = 1) : x = Real.pi / 2 :=
by sorry

end sin_minus_cos_eq_one_l2254_225492


namespace min_possible_range_l2254_225455

theorem min_possible_range (A B C : ℤ) : 
  (A + 15 ≤ C ∧ B + 25 ≤ C ∧ C ≤ A + 45) → C - A ≤ 45 :=
by
  intros h
  have h1 : A + 15 ≤ C := h.1
  have h2 : B + 25 ≤ C := h.2.1
  have h3 : C ≤ A + 45 := h.2.2
  sorry

end min_possible_range_l2254_225455


namespace consecutive_lucky_years_l2254_225438

def is_lucky (Y : ℕ) : Prop := 
  let first_two_digits := Y / 100
  let last_two_digits := Y % 100
  Y % (first_two_digits + last_two_digits) = 0

theorem consecutive_lucky_years : ∃ Y : ℕ, is_lucky Y ∧ is_lucky (Y + 1) :=
by
  sorry

end consecutive_lucky_years_l2254_225438


namespace div_equivalence_l2254_225488

theorem div_equivalence (a b c : ℝ) (h1: a / b = 3) (h2: b / c = 2 / 5) : c / a = 5 / 6 :=
by sorry

end div_equivalence_l2254_225488


namespace find_y_l2254_225434

theorem find_y (x y : ℕ) (h1 : x % y = 9) (h2 : (x : ℝ) / y = 86.12) : y = 75 :=
sorry

end find_y_l2254_225434


namespace box_calories_l2254_225429

theorem box_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  (cookies_per_bag * bags_per_box) * calories_per_cookie = 1600 :=
by
  sorry

end box_calories_l2254_225429


namespace xiaoming_correct_answers_l2254_225432

theorem xiaoming_correct_answers (x : ℕ) (h1 : x ≤ 10) (h2 : 5 * x - (10 - x) > 30) : x ≥ 7 := 
by
  sorry

end xiaoming_correct_answers_l2254_225432


namespace correct_calculation_l2254_225493

theorem correct_calculation (x : ℤ) (h : x - 32 = 33) : x + 32 = 97 := 
by 
  sorry

end correct_calculation_l2254_225493


namespace lindsey_integer_l2254_225495

theorem lindsey_integer (n : ℕ) (a b c : ℤ) (h1 : n < 50)
                        (h2 : n = 6 * a - 1)
                        (h3 : n = 8 * b - 5)
                        (h4 : n = 3 * c + 2) :
  n = 41 := 
  by sorry

end lindsey_integer_l2254_225495


namespace ratio_s_t_l2254_225466

variable {b s t : ℝ}
variable (hb : b ≠ 0)
variable (h1 : s = -b / 8)
variable (h2 : t = -b / 4)

theorem ratio_s_t : s / t = 1 / 2 :=
by
  sorry

end ratio_s_t_l2254_225466


namespace min_eq_one_implies_x_eq_one_l2254_225452

open Real

theorem min_eq_one_implies_x_eq_one (x : ℝ) (h : min (1/2 + x) (x^2) = 1) : x = 1 := 
sorry

end min_eq_one_implies_x_eq_one_l2254_225452


namespace find_k_values_l2254_225413

theorem find_k_values (k : ℚ) 
  (h1 : ∀ k, ∃ m, m = (3 * k + 9) / (7 - k))
  (h2 : ∀ k, m = 2 * k) : 
  (k = 9 / 2 ∨ k = 1) :=
by
  sorry

end find_k_values_l2254_225413


namespace translate_line_upwards_l2254_225428

-- Define the original line equation
def original_line_eq (x : ℝ) : ℝ := 3 * x - 3

-- Define the translation operation
def translate_upwards (y_translation : ℝ) (line_eq : ℝ → ℝ) (x : ℝ) : ℝ :=
  line_eq x + y_translation

-- Define the proof problem
theorem translate_line_upwards :
  ∀ (x : ℝ), translate_upwards 5 original_line_eq x = 3 * x + 2 :=
by
  intros x
  simp [translate_upwards, original_line_eq]
  sorry

end translate_line_upwards_l2254_225428
