import Mathlib

namespace NUMINAMATH_GPT_angles_same_terminal_side_l1294_129405

def angle_equiv (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem angles_same_terminal_side : angle_equiv (-390 : ℝ) (330 : ℝ) :=
sorry

end NUMINAMATH_GPT_angles_same_terminal_side_l1294_129405


namespace NUMINAMATH_GPT_integer_solutions_for_even_ratio_l1294_129466

theorem integer_solutions_for_even_ratio (a : ℤ) (h : ∃ k : ℤ, (a = 2 * k * (1011 - k))): 
  a = 1010 ∨ a = 1012 ∨ a = 1008 ∨ a = 1014 ∨ a = 674 ∨ a = 1348 ∨ a = 0 ∨ a = 2022 :=
sorry

end NUMINAMATH_GPT_integer_solutions_for_even_ratio_l1294_129466


namespace NUMINAMATH_GPT_bookmarks_per_day_l1294_129427

theorem bookmarks_per_day (pages_now : ℕ) (pages_end_march : ℕ) (days_in_march : ℕ) (pages_added : ℕ) (pages_per_day : ℕ)
  (h1 : pages_now = 400)
  (h2 : pages_end_march = 1330)
  (h3 : days_in_march = 31)
  (h4 : pages_added = pages_end_march - pages_now)
  (h5 : pages_per_day = pages_added / days_in_march) :
  pages_per_day = 30 := sorry

end NUMINAMATH_GPT_bookmarks_per_day_l1294_129427


namespace NUMINAMATH_GPT_dot_product_a_b_l1294_129479

-- Define the given vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 2)

-- Define the dot product function
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- State the theorem with the correct answer
theorem dot_product_a_b : dot_product a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_a_b_l1294_129479


namespace NUMINAMATH_GPT_intersection_A_B_l1294_129448

def set_A (x : ℝ) : Prop := 2 * x^2 + 5 * x - 3 ≤ 0

def set_B (x : ℝ) : Prop := -2 < x

theorem intersection_A_B :
  {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | -2 < x ∧ x ≤ 1/2} := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_A_B_l1294_129448


namespace NUMINAMATH_GPT_bugs_diagonally_at_least_9_unoccupied_l1294_129471

theorem bugs_diagonally_at_least_9_unoccupied (bugs : ℕ × ℕ → Prop) :
  let board_size := 9
  let cells := (board_size * board_size)
  let black_cells := 45
  let white_cells := 36
  ∃ unoccupied_cells ≥ 9, true := 
sorry

end NUMINAMATH_GPT_bugs_diagonally_at_least_9_unoccupied_l1294_129471


namespace NUMINAMATH_GPT_f_x_minus_one_l1294_129459

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 5

theorem f_x_minus_one (x : ℝ) : f (x - 1) = x^2 - 4 * x + 8 :=
by
  sorry

end NUMINAMATH_GPT_f_x_minus_one_l1294_129459


namespace NUMINAMATH_GPT_Clea_Rides_Escalator_Alone_l1294_129469

-- Defining the conditions
variables (x y k : ℝ)
def Clea_Walking_Speed := x
def Total_Distance := y = 75 * x
def Time_with_Moving_Escalator := 30 * (x + k) = y
def Escalator_Speed := k = 1.5 * x

-- Stating the proof problem
theorem Clea_Rides_Escalator_Alone : 
  Total_Distance x y → 
  Time_with_Moving_Escalator x y k → 
  Escalator_Speed x k → 
  y / k = 50 :=
by
  intros
  sorry

end NUMINAMATH_GPT_Clea_Rides_Escalator_Alone_l1294_129469


namespace NUMINAMATH_GPT_area_of_right_triangle_l1294_129480

theorem area_of_right_triangle (h : ℝ) 
  (a b : ℝ) 
  (h_a_triple : b = 3 * a)
  (h_hypotenuse : h ^ 2 = a ^ 2 + b ^ 2) : 
  (1 / 2) * a * b = (3 * h ^ 2) / 20 :=
by
  sorry

end NUMINAMATH_GPT_area_of_right_triangle_l1294_129480


namespace NUMINAMATH_GPT_correct_operation_l1294_129490

theorem correct_operation (a b x y m : Real) :
  (¬((a^2 * b)^2 = a^2 * b^2)) ∧
  (¬(a^6 / a^2 = a^3)) ∧
  (¬((x + y)^2 = x^2 + y^2)) ∧
  ((-m)^7 / (-m)^2 = -m^5) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1294_129490


namespace NUMINAMATH_GPT_rotated_line_equation_l1294_129496

-- Define the original equation of the line
def original_line (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define the rotated line equation we want to prove
def rotated_line (x y : ℝ) : Prop := -x + 2 * y + 4 = 0

-- Proof problem statement in Lean 4
theorem rotated_line_equation :
  ∀ (x y : ℝ), original_line x y → rotated_line x y :=
by
  sorry

end NUMINAMATH_GPT_rotated_line_equation_l1294_129496


namespace NUMINAMATH_GPT_geometric_sequence_expression_l1294_129403

theorem geometric_sequence_expression (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 2 = 1)
(h2 : a 3 * a 5 = 2 * a 7) : a n = 1 / 2 ^ (n - 2) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_expression_l1294_129403


namespace NUMINAMATH_GPT_compute_difference_of_squares_l1294_129486

theorem compute_difference_of_squares : (303^2 - 297^2) = 3600 := by
  sorry

end NUMINAMATH_GPT_compute_difference_of_squares_l1294_129486


namespace NUMINAMATH_GPT_find_f8_l1294_129424

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : f (x + y) = f x * f y
axiom initial_condition : f 2 = 4

theorem find_f8 : f 8 = 256 := by
  sorry

end NUMINAMATH_GPT_find_f8_l1294_129424


namespace NUMINAMATH_GPT_sangeun_initial_money_l1294_129408

theorem sangeun_initial_money :
  ∃ (X : ℝ), 
  ((X / 2 - 2000) / 2 - 2000 = 0) ∧ 
  X = 12000 :=
by sorry

end NUMINAMATH_GPT_sangeun_initial_money_l1294_129408


namespace NUMINAMATH_GPT_find_g1_l1294_129431

noncomputable def g : ℝ → ℝ := sorry

axiom g_property (x : ℝ) (hx : x ≠ 1 / 2) : g x + g ((x + 2) / (2 - 4 * x)) = 2 * x + 1

theorem find_g1 : g 1 = 39 / 11 :=
by
  sorry

end NUMINAMATH_GPT_find_g1_l1294_129431


namespace NUMINAMATH_GPT_complementary_angles_difference_l1294_129429

-- Given that the measures of two complementary angles are in the ratio 4:1,
-- we want to prove that the positive difference between the measures of the two angles is 54 degrees.

theorem complementary_angles_difference (x : ℝ) (h_complementary : 4 * x + x = 90) : 
  abs (4 * x - x) = 54 :=
by
  sorry

end NUMINAMATH_GPT_complementary_angles_difference_l1294_129429


namespace NUMINAMATH_GPT_max_smaller_boxes_fit_l1294_129438

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

end NUMINAMATH_GPT_max_smaller_boxes_fit_l1294_129438


namespace NUMINAMATH_GPT_fraction_of_girls_l1294_129476

variable {T G B : ℕ}
variable (ratio : ℚ)

theorem fraction_of_girls (X : ℚ) (h1 : ∀ (G : ℕ) (T : ℕ), X * G = (1/4) * T)
  (h2 : ratio = 5 / 3) (h3 : ∀ (G : ℕ) (B : ℕ), B / G = ratio) :
  X = 2 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_fraction_of_girls_l1294_129476


namespace NUMINAMATH_GPT_inequality_proof_l1294_129428

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1294_129428


namespace NUMINAMATH_GPT_dragons_total_games_l1294_129421

noncomputable def numberOfGames (y x : ℕ) (h1 : x = 6 * y / 10) (h2 : x + 9 = (62 * (y + 12)) / 100) : ℕ :=
y + 12

theorem dragons_total_games (y x : ℕ) (h1 : x = 6 * y / 10) (h2 : x + 9 = (62 * (y + 12)) / 100) :
  numberOfGames y x h1 h2 = 90 := 
sorry

end NUMINAMATH_GPT_dragons_total_games_l1294_129421


namespace NUMINAMATH_GPT_cubes_even_sum_even_l1294_129485

theorem cubes_even_sum_even (p q : ℕ) (h : Even (p^3 - q^3)) : Even (p + q) := sorry

end NUMINAMATH_GPT_cubes_even_sum_even_l1294_129485


namespace NUMINAMATH_GPT_min_value_f_l1294_129461

noncomputable def f (x : ℝ) := 2 * (Real.sin x)^3 + (Real.cos x)^2

theorem min_value_f : ∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 26 / 27 :=
by
  sorry

end NUMINAMATH_GPT_min_value_f_l1294_129461


namespace NUMINAMATH_GPT_length_of_each_part_l1294_129432

-- Conditions
def total_length_in_inches : ℕ := 6 * 12 + 8
def parts_count : ℕ := 4

-- Question
theorem length_of_each_part : total_length_in_inches / parts_count = 20 :=
by
  -- leave the proof as a sorry
  sorry

end NUMINAMATH_GPT_length_of_each_part_l1294_129432


namespace NUMINAMATH_GPT_quadratic_one_real_root_positive_n_l1294_129402

theorem quadratic_one_real_root_positive_n (n : ℝ) (h : (n ≠ 0)) :
  (∃ x : ℝ, (x^2 - 6*n*x - 9*n) = 0) ∧
  (∀ x y : ℝ, (x^2 - 6*n*x - 9*n) = 0 → (y^2 - 6*n*y - 9*n) = 0 → x = y) ↔
  n = 0 := by
  sorry

end NUMINAMATH_GPT_quadratic_one_real_root_positive_n_l1294_129402


namespace NUMINAMATH_GPT_relationship_among_numbers_l1294_129435

theorem relationship_among_numbers :
  let a := 0.7 ^ 2.1
  let b := 0.7 ^ 2.5
  let c := 2.1 ^ 0.7
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_relationship_among_numbers_l1294_129435


namespace NUMINAMATH_GPT_contains_all_integers_l1294_129413

def is_closed_under_divisors (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, b ∣ a → a ∈ A → b ∈ A

def contains_product_plus_one (A : Set ℕ) : Prop :=
  ∀ {a b : ℕ}, 1 < a → a < b → a ∈ A → b ∈ A → (1 + a * b) ∈ A

theorem contains_all_integers
  (A : Set ℕ)
  (h1 : is_closed_under_divisors A)
  (h2 : contains_product_plus_one A)
  (h3 : ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ 1 < a ∧ 1 < b ∧ 1 < c) :
  ∀ n : ℕ, n > 0 → n ∈ A := 
  by 
    sorry

end NUMINAMATH_GPT_contains_all_integers_l1294_129413


namespace NUMINAMATH_GPT_exponent_of_5_in_30_fact_l1294_129407

def count_powers_of_5 (n : ℕ) : ℕ :=
  if n < 5 then 0
  else n / 5 + count_powers_of_5 (n / 5)

theorem exponent_of_5_in_30_fact : count_powers_of_5 30 = 7 := 
  by
    sorry

end NUMINAMATH_GPT_exponent_of_5_in_30_fact_l1294_129407


namespace NUMINAMATH_GPT_composite_evaluation_at_two_l1294_129404

-- Define that P(x) is a polynomial with coefficients in {0, 1}
def is_binary_coefficient_polynomial (P : Polynomial ℤ) : Prop :=
  ∀ (n : ℕ), P.coeff n = 0 ∨ P.coeff n = 1

-- Define that P(x) can be factored into two nonconstant polynomials with integer coefficients
def is_reducible_to_nonconstant_polynomials (P : Polynomial ℤ) : Prop :=
  ∃ (f g : Polynomial ℤ), f.degree > 0 ∧ g.degree > 0 ∧ P = f * g

theorem composite_evaluation_at_two {P : Polynomial ℤ}
  (h1 : is_binary_coefficient_polynomial P)
  (h2 : is_reducible_to_nonconstant_polynomials P) :
  ∃ (m n : ℤ), m > 1 ∧ n > 1 ∧ P.eval 2 = m * n := sorry

end NUMINAMATH_GPT_composite_evaluation_at_two_l1294_129404


namespace NUMINAMATH_GPT_cannot_form_1x1x2_blocks_l1294_129481

theorem cannot_form_1x1x2_blocks :
  let edge_length := 7
  let total_cubes := edge_length * edge_length * edge_length
  let central_cube := (3, 3, 3)
  let remaining_cubes := total_cubes - 1
  let checkerboard_color (x y z : Nat) : Bool := (x + y + z) % 2 = 0
  let num_white (k : Nat) := if k % 2 = 0 then 25 else 24
  let num_black (k : Nat) := if k % 2 = 0 then 24 else 25
  let total_white := 170
  let total_black := 171
  total_black > total_white →
  ¬(remaining_cubes % 2 = 0 ∧ total_white % 2 = 0 ∧ total_black % 2 = 0) → 
  ∀ (block: Nat × Nat × Nat → Bool) (x y z : Nat), block (x, y, z) = ((x*y*z) % 2 = 0) := sorry

end NUMINAMATH_GPT_cannot_form_1x1x2_blocks_l1294_129481


namespace NUMINAMATH_GPT_gcd_calculation_l1294_129423

theorem gcd_calculation : 
  Nat.gcd (111^2 + 222^2 + 333^2) (110^2 + 221^2 + 334^2) = 3 := 
by
  sorry

end NUMINAMATH_GPT_gcd_calculation_l1294_129423


namespace NUMINAMATH_GPT_fraction_product_l1294_129411

theorem fraction_product :
  (7 / 4) * (8 / 14) * (16 / 24) * (32 / 48) * (28 / 7) * (15 / 9) *
  (50 / 25) * (21 / 35) = 32 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_l1294_129411


namespace NUMINAMATH_GPT_M_inter_N_M_union_not_N_l1294_129492

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | x > 0}

theorem M_inter_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 3} := 
sorry

theorem M_union_not_N :
  M ∪ {x | x ≤ 0} = {x | x ≤ 3} := 
sorry

end NUMINAMATH_GPT_M_inter_N_M_union_not_N_l1294_129492


namespace NUMINAMATH_GPT_inequality_system_no_solution_l1294_129454

theorem inequality_system_no_solution (a : ℝ) : ¬ (∃ x : ℝ, x ≤ 5 ∧ x > a) ↔ a ≥ 5 :=
sorry

end NUMINAMATH_GPT_inequality_system_no_solution_l1294_129454


namespace NUMINAMATH_GPT_find_common_ratio_l1294_129436

variable (a_n : ℕ → ℝ)
variable (q : ℝ)
variable (n : ℕ)

theorem find_common_ratio (h1 : a_n 1 = 2) (h2 : a_n 4 = 16) (h_geom : ∀ n, a_n n = a_n (n - 1) * q)
  : q = 2 := by
  sorry

end NUMINAMATH_GPT_find_common_ratio_l1294_129436


namespace NUMINAMATH_GPT_roots_product_of_quadratic_equation_l1294_129415

variables (a b : ℝ)

-- Given that a and b are roots of the quadratic equation x^2 - ax + b = 0
-- and given conditions that a + b = 5 and ab = 6,
-- prove that a * b = 6.
theorem roots_product_of_quadratic_equation 
  (h₁ : a + b = 5) 
  (h₂ : a * b = 6) : 
  a * b = 6 := 
by 
 sorry

end NUMINAMATH_GPT_roots_product_of_quadratic_equation_l1294_129415


namespace NUMINAMATH_GPT_geometric_progression_product_l1294_129414

theorem geometric_progression_product (n : ℕ) (S R : ℝ) (hS : S > 0) (hR : R > 0)
  (h_sum : ∃ (a q : ℝ), a > 0 ∧ q > 0 ∧ S = a * (q^n - 1) / (q - 1))
  (h_reciprocal_sum : ∃ (a q : ℝ), a > 0 ∧ q > 0 ∧ R = (1 - q^n) / (a * q^(n-1) * (q - 1))) :
  ∃ P : ℝ, P = (S / R)^(n / 2) := sorry

end NUMINAMATH_GPT_geometric_progression_product_l1294_129414


namespace NUMINAMATH_GPT_emma_deposit_withdraw_ratio_l1294_129451

theorem emma_deposit_withdraw_ratio (initial_balance withdrawn new_balance : ℤ) 
  (h1 : initial_balance = 230) 
  (h2 : withdrawn = 60) 
  (h3 : new_balance = 290) 
  (deposited : ℤ) 
  (h_deposit : new_balance = initial_balance - withdrawn + deposited) :
  (deposited / withdrawn = 2) := 
sorry

end NUMINAMATH_GPT_emma_deposit_withdraw_ratio_l1294_129451


namespace NUMINAMATH_GPT_sequence_a_n_perfect_square_l1294_129440

theorem sequence_a_n_perfect_square :
  (∃ a : ℕ → ℤ, ∃ b : ℕ → ℤ,
    a 0 = 1 ∧ b 0 = 0 ∧
    (∀ n : ℕ, a (n + 1) = 7 * a n + 6 * b n - 3) ∧
    (∀ n : ℕ, b (n + 1) = 8 * a n + 7 * b n - 4) ∧
    (∀ n : ℕ, ∃ k : ℤ, a n = k^2)) :=
sorry

end NUMINAMATH_GPT_sequence_a_n_perfect_square_l1294_129440


namespace NUMINAMATH_GPT_next_unique_digits_date_l1294_129489

-- Define the conditions
def is_after (d1 d2 : String) : Prop := sorry -- Placeholder, needs a date comparison function
def has_8_unique_digits (date : String) : Prop := sorry -- Placeholder, needs a function to check unique digits

-- Specify the problem and assertion
theorem next_unique_digits_date :
  ∀ date : String, is_after date "11.08.1999" → has_8_unique_digits date → date = "17.06.2345" :=
by
  sorry

end NUMINAMATH_GPT_next_unique_digits_date_l1294_129489


namespace NUMINAMATH_GPT_ingrid_income_l1294_129472

theorem ingrid_income (I : ℝ) (h1 : 0.30 * 56000 = 16800) 
  (h2 : ∀ (I : ℝ), 0.40 * I = 0.4 * I) 
  (h3 : 0.35625 * (56000 + I) = 16800 + 0.4 * I) : 
  I = 49142.86 := 
by 
  sorry

end NUMINAMATH_GPT_ingrid_income_l1294_129472


namespace NUMINAMATH_GPT_log_a1_plus_log_a9_l1294_129426

variable {a : ℕ → ℝ}
variable {log : ℝ → ℝ}

-- Assume the provided conditions
axiom is_geometric_sequence : ∀ n, a (n + 1) / a n = a 1 / a 0
axiom a3a5a7_eq_one : a 3 * a 5 * a 7 = 1
axiom log_mul : ∀ x y, log (x * y) = log x + log y
axiom log_one_eq_zero : log 1 = 0

theorem log_a1_plus_log_a9 : log (a 1) + log (a 9) = 0 := 
by {
    sorry
}

end NUMINAMATH_GPT_log_a1_plus_log_a9_l1294_129426


namespace NUMINAMATH_GPT_dietitian_lunch_fraction_l1294_129498

theorem dietitian_lunch_fraction
  (total_calories : ℕ)
  (recommended_calories : ℕ)
  (extra_calories : ℕ)
  (h1 : total_calories = 40)
  (h2 : recommended_calories = 25)
  (h3 : extra_calories = 5)
  : (recommended_calories + extra_calories) / total_calories = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_dietitian_lunch_fraction_l1294_129498


namespace NUMINAMATH_GPT_modular_home_total_cost_l1294_129477

theorem modular_home_total_cost :
  let kitchen_sqft := 400
  let bathroom_sqft := 200
  let bedroom_sqft := 300
  let living_area_cost_per_sqft := 110
  let kitchen_cost := 28000
  let bathroom_cost := 12000
  let bedroom_cost := 18000
  let total_sqft := 3000
  let required_kitchens := 1
  let required_bathrooms := 2
  let required_bedrooms := 3
  let total_cost := required_kitchens * kitchen_cost +
                    required_bathrooms * bathroom_cost +
                    required_bedrooms * bedroom_cost +
                    (total_sqft - (required_kitchens * kitchen_sqft + required_bathrooms * bathroom_sqft + required_bedrooms * bedroom_sqft)) * living_area_cost_per_sqft
  total_cost = 249000 := 
by
  let kitchen_sqft := 400
  let bathroom_sqft := 200
  let bedroom_sqft := 300
  let living_area_cost_per_sqft := 110
  let kitchen_cost := 28000
  let bathroom_cost := 12000
  let bedroom_cost := 18000
  let total_sqft := 3000
  let required_kitchens := 1
  let required_bathrooms := 2
  let required_bedrooms := 3
  let total_cost := required_kitchens * kitchen_cost +
                    required_bathrooms * bathroom_cost +
                    required_bedrooms * bedroom_cost +
                    (total_sqft - (required_kitchens * kitchen_sqft + required_bathrooms * bathroom_sqft + required_bedrooms * bedroom_sqft)) * living_area_cost_per_sqft
  have h : total_cost = 249000 := sorry
  exact h

end NUMINAMATH_GPT_modular_home_total_cost_l1294_129477


namespace NUMINAMATH_GPT_remaining_apples_l1294_129450

def initial_apples : ℕ := 20
def shared_apples : ℕ := 7

theorem remaining_apples : initial_apples - shared_apples = 13 :=
by
  sorry

end NUMINAMATH_GPT_remaining_apples_l1294_129450


namespace NUMINAMATH_GPT_count_nine_in_1_to_1000_l1294_129475

def count_nine_units : ℕ := 100
def count_nine_tens : ℕ := 100
def count_nine_hundreds : ℕ := 100

theorem count_nine_in_1_to_1000 :
  count_nine_units + count_nine_tens + count_nine_hundreds = 300 :=
by
  sorry

end NUMINAMATH_GPT_count_nine_in_1_to_1000_l1294_129475


namespace NUMINAMATH_GPT_product_n_equals_7200_l1294_129439

theorem product_n_equals_7200 :
  (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) * (3 ^ 2 + 1) = 7200 := by
  sorry

end NUMINAMATH_GPT_product_n_equals_7200_l1294_129439


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_l1294_129467

theorem arithmetic_sequence_general_formula (a : ℤ) :
  ∀ n : ℕ, n ≥ 1 → (∃ a_1 a_2 a_3 : ℤ, a_1 = a - 1 ∧ a_2 = a + 1 ∧ a_3 = a + 3) →
  (a + 2 * n - 3 = a - 1 + (n - 1) * 2) :=
by
  intros n hn h_exists
  rcases h_exists with ⟨a_1, a_2, a_3, h1, h2, h3⟩
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_l1294_129467


namespace NUMINAMATH_GPT_find_x_l1294_129442

theorem find_x (x : ℕ) : 3 * 2^x + 5 * 2^x = 2048 → x = 8 := by
  sorry

end NUMINAMATH_GPT_find_x_l1294_129442


namespace NUMINAMATH_GPT_find_minimal_sum_l1294_129497

theorem find_minimal_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  (x * (x + 1)) ∣ (y * (y + 1)) →
  ¬(x ∣ y ∨ x ∣ (y + 1)) →
  ¬((x + 1) ∣ y ∨ (x + 1) ∣ (y + 1)) →
  x = 14 ∧ y = 35 ∧ x^2 + y^2 = 1421 :=
sorry

end NUMINAMATH_GPT_find_minimal_sum_l1294_129497


namespace NUMINAMATH_GPT_geometric_mean_45_80_l1294_129456

theorem geometric_mean_45_80 : ∃ x : ℝ, x^2 = 45 * 80 ∧ (x = 60 ∨ x = -60) := 
by 
  sorry

end NUMINAMATH_GPT_geometric_mean_45_80_l1294_129456


namespace NUMINAMATH_GPT_number_of_classes_l1294_129449

theorem number_of_classes (sheets_per_class_per_day : ℕ) 
                          (total_sheets_per_week : ℕ) 
                          (school_days_per_week : ℕ) 
                          (h1 : sheets_per_class_per_day = 200) 
                          (h2 : total_sheets_per_week = 9000) 
                          (h3 : school_days_per_week = 5) : 
                          total_sheets_per_week / school_days_per_week / sheets_per_class_per_day = 9 :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_number_of_classes_l1294_129449


namespace NUMINAMATH_GPT_mean_equals_l1294_129447

theorem mean_equals (z : ℝ) :
    (7 + 10 + 15 + 21) / 4 = (18 + z) / 2 → z = 8.5 := 
by
    intro h
    sorry

end NUMINAMATH_GPT_mean_equals_l1294_129447


namespace NUMINAMATH_GPT_probability_of_C_l1294_129473

-- Definitions of probabilities for regions A, B, and D
def P_A : ℚ := 3 / 8
def P_B : ℚ := 1 / 4
def P_D : ℚ := 1 / 8

-- Sum of probabilities must be 1
def total_probability : ℚ := 1

-- The main proof statement
theorem probability_of_C : 
  P_A + P_B + P_D + (P_C : ℚ) = total_probability → P_C = 1 / 4 := sorry

end NUMINAMATH_GPT_probability_of_C_l1294_129473


namespace NUMINAMATH_GPT_sqrt_200_eq_10_sqrt_2_l1294_129412

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_200_eq_10_sqrt_2_l1294_129412


namespace NUMINAMATH_GPT_ice_cream_ordering_ways_l1294_129445

-- Define the possible choices for each category.
def cone_choices : Nat := 2
def scoop_choices : Nat := 1 + 10 + 20  -- Total choices for 1, 2, and 3 scoops.
def topping_choices : Nat := 1 + 4 + 6  -- Total choices for no topping, 1 topping, and 2 toppings.

-- Theorem to state the number of ways ice cream can be ordered.
theorem ice_cream_ordering_ways : cone_choices * scoop_choices * topping_choices = 748 := by
  let calc_cone := cone_choices  -- Number of cone choices.
  let calc_scoop := scoop_choices  -- Number of scoop combinations.
  let calc_topping := topping_choices  -- Number of topping combinations.
  have h1 : calc_cone * calc_scoop * calc_topping = 748 := sorry  -- Calculation hint.
  exact h1

end NUMINAMATH_GPT_ice_cream_ordering_ways_l1294_129445


namespace NUMINAMATH_GPT_deductive_reasoning_example_is_A_l1294_129487

def isDeductive (statement : String) : Prop := sorry

-- Define conditions
def optionA : String := "Since y = 2^x is an exponential function, the function y = 2^x passes through the fixed point (0,1)"
def optionB : String := "Guessing the general formula for the sequence 1/(1×2), 1/(2×3), 1/(3×4), ... as a_n = 1/(n(n+1)) (n ∈ ℕ⁺)"
def optionC : String := "Drawing an analogy from 'In a plane, two lines perpendicular to the same line are parallel' to infer 'In space, two planes perpendicular to the same plane are parallel'"
def optionD : String := "From the circle's equation in the Cartesian coordinate plane (x-a)² + (y-b)² = r², predict that the equation of a sphere in three-dimensional Cartesian coordinates is (x-a)² + (y-b)² + (z-c)² = r²"

theorem deductive_reasoning_example_is_A : isDeductive optionA :=
by
  sorry

end NUMINAMATH_GPT_deductive_reasoning_example_is_A_l1294_129487


namespace NUMINAMATH_GPT_eiffel_tower_scale_l1294_129493

theorem eiffel_tower_scale (height_model : ℝ) (height_actual : ℝ) (h_model : height_model = 30) (h_actual : height_actual = 984) : 
  height_actual / height_model = 32.8 := by
  sorry

end NUMINAMATH_GPT_eiffel_tower_scale_l1294_129493


namespace NUMINAMATH_GPT_solve_hours_l1294_129457

variable (x y : ℝ)

-- Conditions
def Condition1 : x > 0 := sorry
def Condition2 : y > 0 := sorry
def Condition3 : (2:ℝ) / 3 * y / x + (3 * x * y - 2 * y^2) / (3 * x) = x * y / (x + y) + 2 := sorry
def Condition4 : 2 * y / (x + y) = (3 * x - 2 * y) / (3 * x) := sorry

-- Question: How many hours would it take for A and B to complete the task alone?
theorem solve_hours : x = 6 ∧ y = 3 := 
by
  -- Use assumed conditions and variables to define the context
  have h1 := Condition1
  have h2 := Condition2
  have h3 := Condition3
  have h4 := Condition4
  -- Combine analytical relationship and solve for x and y 
  sorry

end NUMINAMATH_GPT_solve_hours_l1294_129457


namespace NUMINAMATH_GPT_necessary_condition_not_sufficient_condition_l1294_129455

def P (x : ℝ) := x > 0
def Q (x : ℝ) := x > -2

theorem necessary_condition : ∀ x: ℝ, P x → Q x := 
by sorry

theorem not_sufficient_condition : ∃ x: ℝ, Q x ∧ ¬ P x := 
by sorry

end NUMINAMATH_GPT_necessary_condition_not_sufficient_condition_l1294_129455


namespace NUMINAMATH_GPT_fraction_to_decimal_l1294_129406

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = 0.3125 := 
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l1294_129406


namespace NUMINAMATH_GPT_abs_inequality_example_l1294_129443

theorem abs_inequality_example (x : ℝ) : abs (5 - x) < 6 ↔ -1 < x ∧ x < 11 :=
by 
  sorry

end NUMINAMATH_GPT_abs_inequality_example_l1294_129443


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1294_129465

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : S 3 = 6)
  (h2 : S 9 = 27) :
  S 6 = 15 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1294_129465


namespace NUMINAMATH_GPT_proof_inequality_l1294_129499

theorem proof_inequality (x : ℝ) : (3 ≤ |x + 2| ∧ |x + 2| ≤ 7) ↔ (1 ≤ x ∧ x ≤ 5 ∨ -9 ≤ x ∧ x ≤ -5) :=
by
  sorry

end NUMINAMATH_GPT_proof_inequality_l1294_129499


namespace NUMINAMATH_GPT_notebook_distribution_l1294_129444

theorem notebook_distribution (x : ℕ) : 
  (∃ k₁ : ℕ, x = 3 * k₁ + 1) ∧ (∃ k₂ : ℕ, x = 4 * k₂ - 2) → (x - 1) / 3 = (x + 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_notebook_distribution_l1294_129444


namespace NUMINAMATH_GPT_pyramid_layers_total_l1294_129446

-- Since we are dealing with natural number calculations, noncomputable is generally not needed.

-- Definition of the pyramid layers and the number of balls in each layer
def number_of_balls (n : ℕ) : ℕ := n ^ 2

-- Given conditions for the layers
def third_layer_balls : ℕ := number_of_balls 3
def fifth_layer_balls : ℕ := number_of_balls 5

-- Statement of the problem proving that their sum is 34
theorem pyramid_layers_total : third_layer_balls + fifth_layer_balls = 34 := by
  sorry -- proof to be provided

end NUMINAMATH_GPT_pyramid_layers_total_l1294_129446


namespace NUMINAMATH_GPT_greatest_root_f_l1294_129422

noncomputable def f (x : ℝ) : ℝ := 21 * x ^ 4 - 20 * x ^ 2 + 3

theorem greatest_root_f :
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x :=
sorry

end NUMINAMATH_GPT_greatest_root_f_l1294_129422


namespace NUMINAMATH_GPT_number_of_tests_initially_l1294_129425

theorem number_of_tests_initially (n : ℕ) (h1 : (90 * n) / n = 90)
  (h2 : ((90 * n) - 75) / (n - 1) = 95) : n = 4 :=
sorry

end NUMINAMATH_GPT_number_of_tests_initially_l1294_129425


namespace NUMINAMATH_GPT_game_ends_in_draw_for_all_n_l1294_129401

noncomputable def andrey_representation_count (n : ℕ) : ℕ := 
  -- The function to count Andrey's representation should be defined here
  sorry

noncomputable def petya_representation_count (n : ℕ) : ℕ := 
  -- The function to count Petya's representation should be defined here
  sorry

theorem game_ends_in_draw_for_all_n (n : ℕ) (h : 0 < n) : 
  andrey_representation_count n = petya_representation_count n :=
  sorry

end NUMINAMATH_GPT_game_ends_in_draw_for_all_n_l1294_129401


namespace NUMINAMATH_GPT_intersection_points_form_rectangle_l1294_129434

theorem intersection_points_form_rectangle
  (x y : ℝ)
  (h1 : x * y = 8)
  (h2 : x^2 + y^2 = 34) :
  ∃ (a b u v : ℝ), (a * b = 8) ∧ (a^2 + b^2 = 34) ∧ 
  (u * v = 8) ∧ (u^2 + v^2 = 34) ∧
  ((a = x ∧ b = y) ∨ (a = y ∧ b = x)) ∧ 
  ((u = -x ∧ v = -y) ∨ (u = -y ∧ v = -x)) ∧
  ((a = u ∧ b = v) ∨ (a = v ∧ b = u)) ∧ 
  ((x = -u ∧ y = -v) ∨ (x = -v ∧ y = -u)) ∧
  (
    (a, b) ≠ (u, v) ∧ (a, b) ≠ (-u, -v) ∧ 
    (a, b) ≠ (v, u) ∧ (a, b) ≠ (-v, -u) ∧
    (u, v) ≠ (-a, -b) ∧ (u, v) ≠ (b, a) ∧ 
    (u, v) ≠ (-b, -a)
  ) :=
by sorry

end NUMINAMATH_GPT_intersection_points_form_rectangle_l1294_129434


namespace NUMINAMATH_GPT_units_digit_of_fraction_l1294_129416

theorem units_digit_of_fraction :
  ((30 * 31 * 32 * 33 * 34) / 400) % 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_fraction_l1294_129416


namespace NUMINAMATH_GPT_yard_length_l1294_129474

-- Definition of the problem conditions
def num_trees : Nat := 11
def distance_between_trees : Nat := 15

-- Length of the yard is given by the product of (num_trees - 1) and distance_between_trees
theorem yard_length :
  (num_trees - 1) * distance_between_trees = 150 :=
by
  sorry

end NUMINAMATH_GPT_yard_length_l1294_129474


namespace NUMINAMATH_GPT_find_line_eq_l1294_129433

-- Definitions for the conditions
def passes_through_M (l : ℝ × ℝ) : Prop :=
  l = (1, 2)

def segment_intercepted_length (l : ℝ × ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ,
    ∀ p : ℝ × ℝ, l p → ((4 * p.1 + 3 * p.2 + 1 = 0 ∨ 4 * p.1 + 3 * p.2 + 6 = 0) ∧ (A = p ∨ B = p)) ∧
    dist A B = Real.sqrt 2

-- Predicates for the lines to be proven
def line_eq1 (p : ℝ × ℝ) : Prop :=
  p.1 + 7 * p.2 = 15

def line_eq2 (p : ℝ × ℝ) : Prop :=
  7 * p.1 - p.2 = 5

-- The proof problem statement
theorem find_line_eq (l : ℝ × ℝ → Prop) :
  passes_through_M (1, 2) →
  segment_intercepted_length l →
  (∀ p, l p → line_eq1 p) ∨ (∀ p, l p → line_eq2 p) :=
by
  sorry

end NUMINAMATH_GPT_find_line_eq_l1294_129433


namespace NUMINAMATH_GPT_mapping_image_l1294_129462

theorem mapping_image (f : ℕ → ℕ) (h : ∀ x, f x = x + 1) : f 3 = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_mapping_image_l1294_129462


namespace NUMINAMATH_GPT_division_result_l1294_129400

theorem division_result (x : ℕ) (h : x + 8 = 88) : x / 10 = 8 := by
  sorry

end NUMINAMATH_GPT_division_result_l1294_129400


namespace NUMINAMATH_GPT_airline_flights_increase_l1294_129495

theorem airline_flights_increase (n k : ℕ) 
  (h : (n + k) * (n + k - 1) / 2 - n * (n - 1) / 2 = 76) :
  (n = 6 ∧ n + k = 14) ∨ (n = 76 ∧ n + k = 77) :=
by
  sorry

end NUMINAMATH_GPT_airline_flights_increase_l1294_129495


namespace NUMINAMATH_GPT_zoo_ticket_problem_l1294_129464

theorem zoo_ticket_problem :
  ∀ (total_amount adult_ticket_cost children_ticket_cost : ℕ)
    (num_adult_tickets : ℕ),
  total_amount = 119 →
  adult_ticket_cost = 21 →
  children_ticket_cost = 14 →
  num_adult_tickets = 4 →
  6 = (num_adult_tickets + (total_amount - num_adult_tickets * adult_ticket_cost) / children_ticket_cost) :=
by 
  intros total_amount adult_ticket_cost children_ticket_cost num_adult_tickets 
         total_amt_eq adult_ticket_cost_eq children_ticket_cost_eq num_adult_tickets_eq
  sorry

end NUMINAMATH_GPT_zoo_ticket_problem_l1294_129464


namespace NUMINAMATH_GPT_integral_identity_proof_l1294_129418

noncomputable def integral_identity : Prop :=
  ∫ x in (0 : Real)..(Real.pi / 2), (Real.cos (Real.cos x))^2 + (Real.sin (Real.sin x))^2 = Real.pi / 2

theorem integral_identity_proof : integral_identity :=
sorry

end NUMINAMATH_GPT_integral_identity_proof_l1294_129418


namespace NUMINAMATH_GPT_smallest_piece_to_cut_l1294_129488

theorem smallest_piece_to_cut (x : ℕ) 
  (h1 : 9 - x > 0) 
  (h2 : 16 - x > 0) 
  (h3 : 18 - x > 0) :
  7 ≤ x ∧ 9 - x + 16 - x ≤ 18 - x :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_piece_to_cut_l1294_129488


namespace NUMINAMATH_GPT_fraction_ratio_x_div_y_l1294_129437

theorem fraction_ratio_x_div_y (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
(h4 : y / (x + z) = (x - y) / z) 
(h5 : y / (x + z) = x / (y + 2 * z)) :
  x / y = 2 / 3 := 
  sorry

end NUMINAMATH_GPT_fraction_ratio_x_div_y_l1294_129437


namespace NUMINAMATH_GPT_original_proposition_true_converse_false_l1294_129417

-- Lean 4 statement for the equivalent proof problem
theorem original_proposition_true_converse_false (a b : ℝ) : 
  (a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ ¬((a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
sorry

end NUMINAMATH_GPT_original_proposition_true_converse_false_l1294_129417


namespace NUMINAMATH_GPT_eighty_first_number_in_set_l1294_129482

theorem eighty_first_number_in_set : ∃ n : ℕ, n = 81 ∧ ∀ k : ℕ, (k = 8 * (n - 1) + 5) → k = 645 := by
  sorry

end NUMINAMATH_GPT_eighty_first_number_in_set_l1294_129482


namespace NUMINAMATH_GPT_range_of_k_l1294_129483

theorem range_of_k (k : ℝ) :
  ∀ x : ℝ, ∃ a b c : ℝ, (a = k-1) → (b = -2) → (c = 1) → (a ≠ 0) → ((b^2 - 4 * a * c) ≥ 0) → k ≤ 2 ∧ k ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l1294_129483


namespace NUMINAMATH_GPT_nth_equation_l1294_129441

theorem nth_equation (n : ℕ) : (n + 1)^2 - 1 = n * (n + 2) := 
by 
  sorry

end NUMINAMATH_GPT_nth_equation_l1294_129441


namespace NUMINAMATH_GPT_karsyn_total_payment_l1294_129491

def initial_price : ℝ := 600
def discount_rate : ℝ := 0.20
def phone_case_cost : ℝ := 25
def screen_protector_cost : ℝ := 15
def store_discount_rate : ℝ := 0.05
def sales_tax_rate : ℝ := 0.035

noncomputable def total_payment : ℝ :=
  let discounted_price := discount_rate * initial_price
  let total_cost := discounted_price + phone_case_cost + screen_protector_cost
  let store_discount := store_discount_rate * total_cost
  let discounted_total := total_cost - store_discount
  let tax := sales_tax_rate * discounted_total
  discounted_total + tax

theorem karsyn_total_payment : total_payment = 157.32 := by
  sorry

end NUMINAMATH_GPT_karsyn_total_payment_l1294_129491


namespace NUMINAMATH_GPT_find_angle_A_l1294_129463

noncomputable def angle_A (a b : ℝ) (B : ℝ) : ℝ :=
  Real.arcsin ((a * Real.sin B) / b)

theorem find_angle_A :
  ∀ (a b : ℝ) (angle_B : ℝ), 0 < a → 0 < b → 0 < angle_B → angle_B < 180 →
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  angle_B = 60 →
  angle_A a b angle_B = 45 :=
by
  intros a b angle_B h1 h2 h3 h4 ha hb hB
  have ha' : a = Real.sqrt 2 := ha
  have hb' : b = Real.sqrt 3 := hb
  have hB' : angle_B = 60 := hB
  -- Proof omitted for demonstration
  sorry

end NUMINAMATH_GPT_find_angle_A_l1294_129463


namespace NUMINAMATH_GPT_time_taken_by_C_l1294_129470

theorem time_taken_by_C (days_A B C : ℕ) (work_done_A work_done_B work_done_C : ℚ) 
  (h1 : days_A = 40) (h2 : work_done_A = 10 * (1/40)) 
  (h3 : days_B = 40) (h4 : work_done_B = 10 * (1/40)) 
  (h5 : work_done_C = 1/2)
  (h6 : 10 * work_done_C = 1/2) :
  (10 * 2) = 20 := 
sorry

end NUMINAMATH_GPT_time_taken_by_C_l1294_129470


namespace NUMINAMATH_GPT_min_dist_on_circle_l1294_129460

theorem min_dist_on_circle :
  let P (θ : ℝ) := (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ)
  let M := (0, 2)
  ∃ θ_min : ℝ, 
    (∀ θ : ℝ, 
      let dist (P : ℝ × ℝ) (M : ℝ × ℝ) := Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2)
      dist (P θ) M ≥ dist (P θ_min) M) ∧ 
    dist (P θ_min) M = Real.sqrt 5 - 1 := sorry

end NUMINAMATH_GPT_min_dist_on_circle_l1294_129460


namespace NUMINAMATH_GPT_problem_solution_l1294_129494

noncomputable def expression_value : ℝ :=
  ((12.983 * 26) / 200) ^ 3 * Real.log 5 / Real.log 10

theorem problem_solution : expression_value = 3.361 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1294_129494


namespace NUMINAMATH_GPT_maximum_automobiles_on_ferry_l1294_129458

-- Define the conditions
def ferry_capacity_tons : ℕ := 50
def automobile_min_weight : ℕ := 1600
def automobile_max_weight : ℕ := 3200

-- Define the conversion factor from tons to pounds
def ton_to_pound : ℕ := 2000

-- Define the converted ferry capacity in pounds
def ferry_capacity_pounds := ferry_capacity_tons * ton_to_pound

-- Proof statement
theorem maximum_automobiles_on_ferry : 
  ferry_capacity_pounds / automobile_min_weight = 62 :=
by
  -- Given: ferry capacity is 50 tons and 1 ton = 2000 pounds
  -- Therefore, ferry capacity in pounds is 50 * 2000 = 100000 pounds
  -- The weight of the lightest automobile is 1600 pounds
  -- Maximum number of automobiles = 100000 / 1600 = 62.5
  -- Rounding down to the nearest whole number gives 62
  sorry  -- Proof steps would be filled here

end NUMINAMATH_GPT_maximum_automobiles_on_ferry_l1294_129458


namespace NUMINAMATH_GPT_toll_constant_l1294_129420

theorem toll_constant (t : ℝ) (x : ℝ) (constant : ℝ) : 
  (t = 1.50 + 0.50 * (x - constant)) → 
  (x = 18 / 2) → 
  (t = 5) → 
  constant = 2 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_toll_constant_l1294_129420


namespace NUMINAMATH_GPT_landscape_breadth_l1294_129478

theorem landscape_breadth (L B : ℕ) (h1 : B = 8 * L)
  (h2 : 3200 = 1 / 9 * (L * B))
  (h3 : B * B = 28800) :
  B = 480 := by
  sorry

end NUMINAMATH_GPT_landscape_breadth_l1294_129478


namespace NUMINAMATH_GPT_speed_of_A_is_7_l1294_129484

theorem speed_of_A_is_7
  (x : ℝ)
  (h1 : ∀ t : ℝ, t = 1)
  (h2 : ∀ y : ℝ, y = 3)
  (h3 : ∀ n : ℕ, n = 10)
  (h4 : x + 3 = 10) :
  x = 7 := by
  sorry

end NUMINAMATH_GPT_speed_of_A_is_7_l1294_129484


namespace NUMINAMATH_GPT_D_72_is_22_l1294_129410

def D (n : ℕ) : ℕ :=
   -- function definition for D that satisfies the problem's conditions
   sorry

theorem D_72_is_22 : D 72 = 22 :=
by sorry

end NUMINAMATH_GPT_D_72_is_22_l1294_129410


namespace NUMINAMATH_GPT_John_spent_15_dollars_on_soap_l1294_129468

theorem John_spent_15_dollars_on_soap (number_of_bars : ℕ) (weight_per_bar : ℝ) (cost_per_pound : ℝ)
  (h1 : number_of_bars = 20) (h2 : weight_per_bar = 1.5) (h3 : cost_per_pound = 0.5) :
  (number_of_bars * weight_per_bar * cost_per_pound) = 15 :=
by
  sorry

end NUMINAMATH_GPT_John_spent_15_dollars_on_soap_l1294_129468


namespace NUMINAMATH_GPT_max_height_reached_l1294_129419

def h (t : ℝ) : ℝ := -20 * t^2 + 50 * t + 10

theorem max_height_reached : ∃ (t : ℝ), h t = 41.25 :=
by
  use 1.25
  sorry

end NUMINAMATH_GPT_max_height_reached_l1294_129419


namespace NUMINAMATH_GPT_find_value_of_a_l1294_129430

theorem find_value_of_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53 ^ 2017 + a) % 13 = 0) : a = 12 := 
by 
  sorry

end NUMINAMATH_GPT_find_value_of_a_l1294_129430


namespace NUMINAMATH_GPT_quadratic_roots_relation_l1294_129452

variable (a b c X1 X2 : ℝ)

theorem quadratic_roots_relation (h : a ≠ 0) : 
  (X1 + X2 = -b / a) ∧ (X1 * X2 = c / a) :=
sorry

end NUMINAMATH_GPT_quadratic_roots_relation_l1294_129452


namespace NUMINAMATH_GPT_min_students_l1294_129453

theorem min_students (b g : ℕ) 
  (h1 : 3 * b = 2 * g) 
  (h2 : ∃ k : ℕ, b + g = 10 * k) : b + g = 38 :=
sorry

end NUMINAMATH_GPT_min_students_l1294_129453


namespace NUMINAMATH_GPT_smallest_positive_integer_ends_in_7_and_divisible_by_5_l1294_129409

theorem smallest_positive_integer_ends_in_7_and_divisible_by_5 : 
  ∃ n : ℤ, n > 0 ∧ n % 10 = 7 ∧ n % 5 = 0 ∧ n = 37 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_ends_in_7_and_divisible_by_5_l1294_129409
