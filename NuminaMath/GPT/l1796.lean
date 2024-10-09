import Mathlib

namespace price_of_A_is_40_l1796_179604

theorem price_of_A_is_40
  (p_a p_b : ℕ)
  (h1 : p_a = 2 * p_b)
  (h2 : 400 / p_a = 400 / p_b - 10) : p_a = 40 := 
by
  sorry

end price_of_A_is_40_l1796_179604


namespace cricketer_average_after_22nd_inning_l1796_179689

theorem cricketer_average_after_22nd_inning (A : ℚ) 
  (h1 : 21 * A + 134 = (A + 3.5) * 22)
  (h2 : 57 = A) :
  A + 3.5 = 60.5 :=
by
  exact sorry

end cricketer_average_after_22nd_inning_l1796_179689


namespace sitio_proof_l1796_179650

theorem sitio_proof :
  (∃ t : ℝ, t = 4 + 7 + 12 ∧ 
    (∃ f : ℝ, 
      (∃ s : ℝ, s = 6 + 5 + 10 ∧ t = 23 ∧ f = 23 - s) ∧
      f = 2) ∧
    (∃ cost_per_hectare : ℝ, cost_per_hectare = 2420 / (4 + 12) ∧ 
      (∃ saci_spent : ℝ, saci_spent = 6 * cost_per_hectare ∧ saci_spent = 1320))) :=
by sorry

end sitio_proof_l1796_179650


namespace min_value_fraction_l1796_179655

theorem min_value_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b > 0) (h₃ : 2 * a + b = 1) : 
  ∃ x, x = 8 ∧ ∀ y, (y = (1 / a) + (2 / b)) → y ≥ x :=
sorry

end min_value_fraction_l1796_179655


namespace calculate_expression_l1796_179646

theorem calculate_expression :
  5 * Real.sqrt 3 + (Real.sqrt 4 + 2 * Real.sqrt 3) = 7 * Real.sqrt 3 + 2 :=
by sorry

end calculate_expression_l1796_179646


namespace students_taking_both_chorus_and_band_l1796_179685

theorem students_taking_both_chorus_and_band (total_students : ℕ) 
                                             (chorus_students : ℕ)
                                             (band_students : ℕ)
                                             (not_enrolled_students : ℕ) : 
                                             total_students = 50 ∧
                                             chorus_students = 18 ∧
                                             band_students = 26 ∧
                                             not_enrolled_students = 8 →
                                             ∃ (both_chorus_and_band : ℕ), both_chorus_and_band = 2 :=
by
  intros h
  sorry

end students_taking_both_chorus_and_band_l1796_179685


namespace sum_f_values_l1796_179629

noncomputable def f (x : ℤ) : ℤ := (x - 1)^3 + 1

theorem sum_f_values :
  (f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7) = 13 :=
by
  sorry

end sum_f_values_l1796_179629


namespace blue_candies_count_l1796_179688

theorem blue_candies_count (total_pieces red_pieces : Nat) (h1 : total_pieces = 3409) (h2 : red_pieces = 145) : total_pieces - red_pieces = 3264 := 
by
  -- Proof will be provided here
  sorry

end blue_candies_count_l1796_179688


namespace one_third_pow_3_eq_3_pow_nineteen_l1796_179691

theorem one_third_pow_3_eq_3_pow_nineteen (y : ℤ) (h : (1 / 3 : ℝ) * (3 ^ 20) = 3 ^ y) : y = 19 :=
by
  sorry

end one_third_pow_3_eq_3_pow_nineteen_l1796_179691


namespace hannahs_vegetarian_restaurant_l1796_179625

theorem hannahs_vegetarian_restaurant :
  let total_weight_of_peppers := 0.6666666666666666
  let weight_of_green_peppers := 0.3333333333333333
  total_weight_of_peppers - weight_of_green_peppers = 0.3333333333333333 :=
by
  sorry

end hannahs_vegetarian_restaurant_l1796_179625


namespace carmen_more_miles_l1796_179608

-- Definitions for the conditions
def carmen_distance : ℕ := 90
def daniel_distance : ℕ := 75

-- The theorem statement
theorem carmen_more_miles : carmen_distance - daniel_distance = 15 :=
by
  sorry

end carmen_more_miles_l1796_179608


namespace parabola_points_l1796_179622

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_points :
  ∃ (a c m n : ℝ),
  a = 2 ∧ c = -2 ∧
  parabola a 1 c 2 = m ∧
  parabola a 1 c n = -2 ∧
  m = 8 ∧
  n = -1 / 2 :=
by
  use 2, -2, 8, -1/2
  simp [parabola]
  sorry

end parabola_points_l1796_179622


namespace employees_original_number_l1796_179640

noncomputable def original_employees_approx (employees_remaining : ℝ) (reduction_percent : ℝ) : ℝ :=
  employees_remaining / (1 - reduction_percent)

theorem employees_original_number (employees_remaining : ℝ) (reduction_percent : ℝ) (original : ℝ) :
  employees_remaining = 462 → reduction_percent = 0.276 →
  abs (original_employees_approx employees_remaining reduction_percent - original) < 1 →
  original = 638 :=
by
  intros h_remaining h_reduction h_approx
  sorry

end employees_original_number_l1796_179640


namespace shifted_parabola_eq_l1796_179623

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := -(x^2)

-- Define the transformation for shifting left 2 units
def shift_left (x : ℝ) : ℝ := x + 2

-- Define the transformation for shifting down 3 units
def shift_down (y : ℝ) : ℝ := y - 3

-- Define the new parabola equation after shifting
def new_parabola (x : ℝ) : ℝ := shift_down (original_parabola (shift_left x))

-- The theorem to be proven
theorem shifted_parabola_eq : new_parabola x = -(x + 2)^2 - 3 := by
  sorry

end shifted_parabola_eq_l1796_179623


namespace cost_to_buy_450_candies_l1796_179610

-- Define a structure representing the problem conditions
structure CandyStore where
  candies_per_box : Nat
  regular_price : Nat
  discounted_price : Nat
  discount_threshold : Nat

-- Define parameters for this specific problem
def store : CandyStore :=
  { candies_per_box := 15,
    regular_price := 5,
    discounted_price := 4,
    discount_threshold := 10 }

-- Define the cost function with the given conditions
def cost (store : CandyStore) (candies : Nat) : Nat :=
  let boxes := candies / store.candies_per_box
  if boxes >= store.discount_threshold then
    boxes * store.discounted_price
  else
    boxes * store.regular_price

-- State the theorem we want to prove
theorem cost_to_buy_450_candies (store : CandyStore) (candies := 450) :
  store.candies_per_box = 15 →
  store.discounted_price = 4 →
  store.discount_threshold = 10 →
  cost store candies = 120 := by
  sorry

end cost_to_buy_450_candies_l1796_179610


namespace symmedian_length_l1796_179609

theorem symmedian_length (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ AS : ℝ, AS = (b * c^2 / (b^2 + c^2)) * (Real.sqrt (2 * b^2 + 2 * c^2 - a^2)) :=
sorry

end symmedian_length_l1796_179609


namespace free_cytosine_molecules_req_l1796_179637

-- Definition of conditions
def DNA_base_pairs := 500
def AT_percentage := 34 / 100
def CG_percentage := 1 - AT_percentage

-- The total number of bases
def total_bases := 2 * DNA_base_pairs

-- The number of C or G bases
def CG_bases := total_bases * CG_percentage

-- Finally, the total number of free cytosine deoxyribonucleotide molecules 
def free_cytosine_molecules := 2 * CG_bases

-- Problem statement: Prove that the number of free cytosine deoxyribonucleotide molecules required is 1320
theorem free_cytosine_molecules_req : free_cytosine_molecules = 1320 :=
by
  -- conditions are defined, the proof is omitted
  sorry

end free_cytosine_molecules_req_l1796_179637


namespace compute_expression_l1796_179681

theorem compute_expression :
  (75 * 1313 - 25 * 1313 + 50 * 1313 = 131300) :=
by
  sorry

end compute_expression_l1796_179681


namespace fleas_after_treatment_l1796_179668

theorem fleas_after_treatment
  (F : ℕ)  -- F is the number of fleas the dog has left after the treatments
  (half_fleas : ℕ → ℕ)  -- Function representing halving fleas
  (initial_fleas := F + 210)  -- Initial number of fleas before treatment
  (half_fleas_def : ∀ n, half_fleas n = n / 2)  -- Definition of half_fleas function
  (condition : F = (half_fleas (half_fleas (half_fleas (half_fleas initial_fleas)))))  -- Condition given in the problem
  :
  F = 14 := 
  sorry

end fleas_after_treatment_l1796_179668


namespace minimum_value_of_a2b_l1796_179694

noncomputable def minimum_value (a b : ℝ) := a + 2 * b

theorem minimum_value_of_a2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / (2 * a + b) + 1 / (b + 1) = 1) :
  minimum_value a b = (2 * Real.sqrt 3 + 1) / 2 :=
sorry

end minimum_value_of_a2b_l1796_179694


namespace find_f_neg1_l1796_179695

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x - 1 else -2^(-x) + 2*x + 1

theorem find_f_neg1 : f (-1) = -3 :=
by
  -- The proof is omitted.
  sorry

end find_f_neg1_l1796_179695


namespace quadratic_coefficients_l1796_179678

theorem quadratic_coefficients :
  ∀ (a b c : ℤ), (2 * a * a - b * a - 5 = 0) → (a = 2 ∧ b = -1) :=
by
  intros a b c H
  sorry

end quadratic_coefficients_l1796_179678


namespace find_result_l1796_179605

def f (x : ℝ) : ℝ := 2 * x + 1
def g (x : ℝ) : ℝ := 4 * x - 3

theorem find_result : f (g 3) - g (f 3) = -6 := by
  sorry

end find_result_l1796_179605


namespace number_of_technicians_l1796_179634

-- Define the problem statements
variables (T R : ℕ)

-- Conditions based on the problem description
def condition1 : Prop := T + R = 42
def condition2 : Prop := 3 * T + R = 56

-- The main goal to prove
theorem number_of_technicians (h1 : condition1 T R) (h2 : condition2 T R) : T = 7 :=
by
  sorry -- Proof is omitted as per instructions

end number_of_technicians_l1796_179634


namespace ratio_of_playground_area_to_total_landscape_area_l1796_179628

theorem ratio_of_playground_area_to_total_landscape_area {B L : ℝ} 
    (h1 : L = 8 * B)
    (h2 : L = 240)
    (h3 : 1200 = (240 * B * L) / (240 * B)) :
    1200 / (240 * B) = 1 / 6 :=
sorry

end ratio_of_playground_area_to_total_landscape_area_l1796_179628


namespace value_of_nested_f_l1796_179673

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_nested_f : f (f (f (f (f (f (-1)))))) = 3432163846882600 := by
  sorry

end value_of_nested_f_l1796_179673


namespace seq_100_eq_11_div_12_l1796_179684

def seq (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 1 / 3
  else if n ≥ 3 then (2 - seq (n - 1)) / (3 * seq (n - 2) + 1)
  else 0 -- This line handles the case n < 1, but shouldn't ever be used in practice.

theorem seq_100_eq_11_div_12 : seq 100 = 11 / 12 :=
  sorry

end seq_100_eq_11_div_12_l1796_179684


namespace coeff_sum_eq_twenty_l1796_179603

theorem coeff_sum_eq_twenty 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ)
  (h : ((2 * x - 3) ^ 5) = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) :
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ + 5 * a₅ = 20 :=
by
  sorry

end coeff_sum_eq_twenty_l1796_179603


namespace find_positive_integer_n_l1796_179615

theorem find_positive_integer_n (n : ℕ) (h₁ : 200 % n = 5) (h₂ : 395 % n = 5) : n = 13 :=
sorry

end find_positive_integer_n_l1796_179615


namespace cubic_equation_roots_l1796_179611

theorem cubic_equation_roots :
  (∀ x : ℝ, (x^3 - 7*x^2 + 36 = 0) → (x = -2 ∨ x = 3 ∨ x = 6)) ∧
  ∃ (x1 x2 x3 : ℝ), (x1 * x2 = 18) ∧ (x1 * x2 * x3 = -36) :=
by
  sorry

end cubic_equation_roots_l1796_179611


namespace largest_of_four_consecutive_even_numbers_l1796_179665

-- Conditions
def sum_of_four_consecutive_even_numbers (x : ℤ) : Prop :=
  x + (x + 2) + (x + 4) + (x + 6) = 92

-- Proof statement
theorem largest_of_four_consecutive_even_numbers (x : ℤ) 
  (h : sum_of_four_consecutive_even_numbers x) : x + 6 = 26 :=
by
  sorry

end largest_of_four_consecutive_even_numbers_l1796_179665


namespace integer_root_of_quadratic_eq_l1796_179636

theorem integer_root_of_quadratic_eq (m : ℤ) (hm : ∃ x : ℤ, m * x^2 + 2 * (m - 5) * x + (m - 4) = 0) : m = -4 ∨ m = 4 ∨ m = -16 :=
sorry

end integer_root_of_quadratic_eq_l1796_179636


namespace black_stones_count_l1796_179692

theorem black_stones_count (T W B : ℕ) (hT : T = 48) (hW1 : 4 * W = 37 * 2 + 26) (hB : B = T - W) : B = 23 :=
by
  sorry

end black_stones_count_l1796_179692


namespace smallest_p_l1796_179666

theorem smallest_p (p q : ℕ) (h1 : p + q = 2005) (h2 : (5:ℚ)/8 < p / q) (h3 : p / q < (7:ℚ)/8) : p = 772 :=
sorry

end smallest_p_l1796_179666


namespace total_bags_l1796_179600

theorem total_bags (people : ℕ) (bags_per_person : ℕ) (h_people : people = 4) (h_bags_per_person : bags_per_person = 8) : people * bags_per_person = 32 := by
  sorry

end total_bags_l1796_179600


namespace breadth_of_boat_l1796_179606

theorem breadth_of_boat
  (L : ℝ) (h : ℝ) (m : ℝ) (g : ℝ) (ρ : ℝ) (B : ℝ)
  (hL : L = 3)
  (hh : h = 0.01)
  (hm : m = 60)
  (hg : g = 9.81)
  (hρ : ρ = 1000) :
  B = 2 := by
  sorry

end breadth_of_boat_l1796_179606


namespace compare_powers_l1796_179627

theorem compare_powers :
  100^100 > 50^50 * 150^50 := sorry

end compare_powers_l1796_179627


namespace frenchwoman_present_l1796_179677

theorem frenchwoman_present
    (M_F M_R W_R : ℝ)
    (condition_1 : M_F > M_R + W_R)
    (condition_2 : W_R > M_F + M_R) 
    : false :=
by
  -- We would assume the opposite of what we know to lead to a contradiction here.
  -- This is a placeholder to indicate the proof should lead to a contradiction.
  sorry

end frenchwoman_present_l1796_179677


namespace lcm_problem_l1796_179670

theorem lcm_problem :
  ∃ k_values : Finset ℕ, (∀ k ∈ k_values, (60^10 : ℕ) = Nat.lcm (Nat.lcm (10^10) (12^12)) k) ∧ k_values.card = 121 :=
by
  sorry

end lcm_problem_l1796_179670


namespace find_b_and_c_find_b_with_c_range_of_b_l1796_179621

-- Part (Ⅰ)
theorem find_b_and_c (b c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 + 2 * b * x + c)
  (h_zeros : f (-1) = 0 ∧ f 1 = 0) : b = 0 ∧ c = -1 := sorry

-- Part (Ⅱ)
theorem find_b_with_c (b : ℝ) (f : ℝ → ℝ)
  (x1 x2 : ℝ) 
  (h_def : ∀ x, f x = x^2 + 2 * b * x + (b^2 + 2 * b + 3))
  (h_eq : (x1 + 1) * (x2 + 1) = 8) 
  (h_roots : f x1 = 0 ∧ f x2 = 0) : b = -2 := sorry

-- Part (Ⅲ)
theorem range_of_b (b : ℝ) (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_f_def : ∀ x, f x = x^2 + 2 * b * x + (-1 - 2 * b))
  (h_f_1 : f 1 = 0)
  (h_g_def : ∀ x, g x = f x + x + b)
  (h_intervals : ∀ x, 
    ((-3 < x) ∧ (x < -2) → g x > 0) ∧
    ((-2 < x) ∧ (x < 0) → g x < 0) ∧
    ((0 < x) ∧ (x < 1) → g x < 0) ∧
    ((1 < x) → g x > 0)) : (1/5) < b ∧ b < (5/7) := sorry

end find_b_and_c_find_b_with_c_range_of_b_l1796_179621


namespace annual_interest_rate_l1796_179687

theorem annual_interest_rate (initial_amount final_amount : ℝ) 
  (h_initial : initial_amount = 90) 
  (h_final : final_amount = 99) : 
  ((final_amount - initial_amount) / initial_amount) * 100 = 10 :=
by {
  sorry
}

end annual_interest_rate_l1796_179687


namespace f_monotonically_decreasing_range_of_a_tangent_intersection_l1796_179633

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 + 2
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 2 * a * x

-- Part (I)
theorem f_monotonically_decreasing (a : ℝ) (x : ℝ) :
  (a > 0 → 0 < x ∧ x < (2 / 3) * a → f' x a < 0) ∧
  (a = 0 → ¬∃ x, f' x a < 0) ∧
  (a < 0 → (2 / 3) * a < x ∧ x < 0 → f' x a < 0) :=
sorry

-- Part (II)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f' x a ≥ abs x - 3 / 4) → (-1 ≤ a ∧ a ≤ 1) :=
sorry

-- Part (III)
theorem tangent_intersection (a : ℝ) :
  (a = 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ ∃ t : ℝ, (t - x1^3 - 2 = 3 * x1^2 * (2 - x1)) ∧
  (t - x2^3 - 2 = 3 * x2^2 * (2 - x2)) ∧ 2 ≤ t ∧ t ≤ 10 ∧
  ∀ t', (t' - x1^3 - 2 = 3 * x1^2 * (2 - x1)) ∧
  (t' - x2^3 - 2 = 3 * x2^2 * (2 - x2)) → t' ≤ 10) :=
sorry

end f_monotonically_decreasing_range_of_a_tangent_intersection_l1796_179633


namespace point_not_in_third_quadrant_l1796_179675

theorem point_not_in_third_quadrant (x y : ℝ) (h : y = -x + 1) : ¬(x < 0 ∧ y < 0) :=
by
  sorry

end point_not_in_third_quadrant_l1796_179675


namespace inversely_proportional_x_y_l1796_179656

noncomputable def k := 320

theorem inversely_proportional_x_y (x y : ℕ) (h1 : x * y = k) :
  (∀ x, y = 10 → x = 32) ↔ (x = 32) :=
by
  sorry

end inversely_proportional_x_y_l1796_179656


namespace minimum_value_A_l1796_179676

theorem minimum_value_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + c^2 * a = 3) : 
  (a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3) ≥ 6 :=
by
  sorry

end minimum_value_A_l1796_179676


namespace rowing_distance_correct_l1796_179626

variable (D : ℝ) -- distance to the place
variable (speed_in_still_water : ℝ := 10) -- rowing speed in still water
variable (current_speed : ℝ := 2) -- speed of the current
variable (total_time : ℝ := 30) -- total time for round trip
variable (effective_speed_with_current : ℝ := speed_in_still_water + current_speed) -- effective speed with current
variable (effective_speed_against_current : ℝ := speed_in_still_water - current_speed) -- effective speed against current

theorem rowing_distance_correct : 
  D / effective_speed_with_current + D / effective_speed_against_current = total_time → 
  D = 144 := 
by
  intros h
  sorry

end rowing_distance_correct_l1796_179626


namespace solve_for_m_l1796_179659

theorem solve_for_m (x m : ℝ) (h1 : 2 * 1 - m = -3) : m = 5 :=
by
  sorry

end solve_for_m_l1796_179659


namespace point_lies_on_graph_l1796_179619

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 2|

theorem point_lies_on_graph (a : ℝ) : f (-a) = f (a) :=
by
  sorry

end point_lies_on_graph_l1796_179619


namespace cakes_difference_l1796_179699

theorem cakes_difference (cakes_made : ℕ) (cakes_sold : ℕ) (cakes_bought : ℕ) 
  (h1 : cakes_made = 648) (h2 : cakes_sold = 467) (h3 : cakes_bought = 193) :
  (cakes_sold - cakes_bought = 274) :=
by
  sorry

end cakes_difference_l1796_179699


namespace intersection_point_of_circle_and_line_l1796_179664

noncomputable def circle_parametric (α : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos α, 2 * Real.sin α)
noncomputable def line_polar (rho θ : ℝ) : Prop := rho * Real.sin θ = 2

theorem intersection_point_of_circle_and_line :
  ∃ (α : ℝ) (rho θ : ℝ), circle_parametric α = (1, 2) ∧ line_polar rho θ := sorry

end intersection_point_of_circle_and_line_l1796_179664


namespace balloons_remaining_intact_l1796_179680

def initial_balloons : ℕ := 200
def blown_up_after_half_hour (n : ℕ) : ℕ := n / 5
def remaining_balloons_after_half_hour (n : ℕ) : ℕ := n - blown_up_after_half_hour n

def percentage_of_remaining_balloons_blow_up (remaining : ℕ) : ℕ := remaining * 30 / 100
def remaining_balloons_after_one_hour (remaining : ℕ) : ℕ := remaining - percentage_of_remaining_balloons_blow_up remaining

def durable_balloons (remaining : ℕ) : ℕ := remaining * 10 / 100
def non_durable_balloons (remaining : ℕ) (durable : ℕ) : ℕ := remaining - durable

def twice_non_durable (non_durable : ℕ) : ℕ := non_durable * 2

theorem balloons_remaining_intact : 
  (remaining_balloons_after_half_hour initial_balloons) - 
  (percentage_of_remaining_balloons_blow_up 
    (remaining_balloons_after_half_hour initial_balloons)) - 
  (twice_non_durable 
    (non_durable_balloons 
      (remaining_balloons_after_one_hour 
        (remaining_balloons_after_half_hour initial_balloons)) 
      (durable_balloons 
        (remaining_balloons_after_one_hour 
          (remaining_balloons_after_half_hour initial_balloons))))) = 
  0 := 
by
  sorry

end balloons_remaining_intact_l1796_179680


namespace domain_f_l1796_179658

open Real

def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2) - 3

theorem domain_f :
  {x : ℝ | g x > 0} = {x : ℝ | x < 0 ∨ x > 3} :=
by 
  sorry

end domain_f_l1796_179658


namespace exist_positive_m_l1796_179671

theorem exist_positive_m {n p q : ℕ} (hn_pos : 0 < n) (hp_prime : Prime p) (hq_prime : Prime q) 
  (h1 : pq ∣ n ^ p + 2) (h2 : n + 2 ∣ n ^ p + q ^ p) : ∃ m : ℕ, q ∣ 4 ^ m * n + 2 := 
sorry

end exist_positive_m_l1796_179671


namespace chuck_bicycle_trip_l1796_179693

theorem chuck_bicycle_trip (D : ℝ) (h1 : D / 16 + D / 24 = 3) : D = 28.80 :=
by
  sorry

end chuck_bicycle_trip_l1796_179693


namespace max_discount_l1796_179616

variable (x : ℝ)

theorem max_discount (h1 : (1 + 0.8) * x = 360) : 360 - 1.2 * x = 120 := 
by
  sorry

end max_discount_l1796_179616


namespace part1_part2_part3_l1796_179647

theorem part1 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0 ↔ x < -3 ∨ x > -2) : k = -2/5 :=
sorry

theorem part2 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) : k < -Real.sqrt 6 / 6 :=
sorry

theorem part3 (k : ℝ) (h₀ : k ≠ 0) (h : ∀ x : ℝ, ¬ (k * x^2 - 2 * x + 6 * k < 0)) : k ≥ Real.sqrt 6 / 6 :=
sorry

end part1_part2_part3_l1796_179647


namespace minimum_value_l1796_179644

noncomputable def problem_statement : Prop :=
  ∃ (a b : ℝ), (∃ (x : ℝ), x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) ∧ (a^2 + b^2 = 4 / 5)

-- This line states that the minimum possible value of a^2 + b^2, given the condition, is 4/5.
theorem minimum_value (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 :=
  sorry

end minimum_value_l1796_179644


namespace incorrect_average_initially_l1796_179635

theorem incorrect_average_initially (S : ℕ) :
  (S + 25) / 10 = 46 ↔ (S + 65) / 10 = 50 := by
  sorry

end incorrect_average_initially_l1796_179635


namespace letters_in_small_envelopes_l1796_179612

theorem letters_in_small_envelopes (total_letters : ℕ) (large_envelopes : ℕ) (letters_per_large_envelope : ℕ) (letters_in_small_envelopes : ℕ) :
  total_letters = 80 →
  large_envelopes = 30 →
  letters_per_large_envelope = 2 →
  letters_in_small_envelopes = total_letters - (large_envelopes * letters_per_large_envelope) →
  letters_in_small_envelopes = 20 :=
by
  intros ht hl he hs
  rw [ht, hl, he] at hs
  exact hs

#check letters_in_small_envelopes

end letters_in_small_envelopes_l1796_179612


namespace pascal_30th_31st_numbers_l1796_179617

-- Definitions based on conditions
def pascal_triangle_row_34 (k : ℕ) : ℕ := Nat.choose 34 k

-- Problem statement in Lean 4: proving the equations
theorem pascal_30th_31st_numbers :
  pascal_triangle_row_34 29 = 278256 ∧
  pascal_triangle_row_34 30 = 46376 :=
by
  sorry

end pascal_30th_31st_numbers_l1796_179617


namespace sequence_sum_100_eq_200_l1796_179645

theorem sequence_sum_100_eq_200
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = 2)
  (h4 : ∀ n : ℕ, a n * a (n + 1) * a (n + 2) ≠ 1)
  (h5 : ∀ n : ℕ, a n * a (n + 1) * a (n + 2) * a (n + 3) = a n + a (n + 1) + a (n + 2) + a (n + 3)) :
  (Finset.range 100).sum (a ∘ Nat.succ) = 200 := by
  sorry

end sequence_sum_100_eq_200_l1796_179645


namespace factorization_correct_l1796_179662

noncomputable def factor_polynomial (x : ℝ) : ℝ := 4 * x^3 - 4 * x^2 + x

theorem factorization_correct (x : ℝ) : 
  factor_polynomial x = x * (2 * x - 1)^2 :=
by
  sorry

end factorization_correct_l1796_179662


namespace consecutive_integers_sum_l1796_179660

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : a < Real.sqrt 3) (h3 : Real.sqrt 3 < b) : a + b = 3 :=
sorry

end consecutive_integers_sum_l1796_179660


namespace time_to_cover_escalator_l1796_179639

def escalator_speed : ℝ := 12
def escalator_length : ℝ := 160
def person_speed : ℝ := 8

theorem time_to_cover_escalator :
  (escalator_length / (escalator_speed + person_speed)) = 8 := by
  sorry

end time_to_cover_escalator_l1796_179639


namespace only_positive_integer_a_squared_plus_2a_is_perfect_square_l1796_179698

/-- Prove that the only positive integer \( a \) for which \( a^2 + 2a \) is a perfect square is \( a = 0 \). -/
theorem only_positive_integer_a_squared_plus_2a_is_perfect_square :
  ∀ (a : ℕ), (∃ (k : ℕ), a^2 + 2*a = k^2) → a = 0 :=
by
  intro a h
  sorry

end only_positive_integer_a_squared_plus_2a_is_perfect_square_l1796_179698


namespace distance_from_two_eq_three_l1796_179652

theorem distance_from_two_eq_three (x : ℝ) (h : |x - 2| = 3) : x = -1 ∨ x = 5 :=
sorry

end distance_from_two_eq_three_l1796_179652


namespace extremum_of_function_l1796_179661

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x

theorem extremum_of_function :
  (∀ x, f x ≥ -Real.exp 1) ∧ (f 1 = -Real.exp 1) ∧ (∀ M, ∃ x, f x > M) :=
by
  sorry

end extremum_of_function_l1796_179661


namespace polynomial_evaluation_l1796_179613

def f (x : ℝ) : ℝ := sorry

theorem polynomial_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 + 1) = x^4 + 6 * x^2 + 2) :
  f (x^2 - 3) = x^4 - 2 * x^2 - 7 :=
sorry

end polynomial_evaluation_l1796_179613


namespace extreme_value_of_f_range_of_values_for_a_l1796_179696

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem extreme_value_of_f :
  ∃ x_min : ℝ, f x_min = 1 :=
sorry

theorem range_of_values_for_a :
  ∀ a : ℝ, (∀ x : ℝ, f x ≥ (x^3) / 6 + a) → a ≤ 1 :=
sorry

end extreme_value_of_f_range_of_values_for_a_l1796_179696


namespace fractional_equation_m_value_l1796_179638

theorem fractional_equation_m_value {x m : ℝ} (hx : 0 < x) (h : 3 / (x - 4) = 1 - (x + m) / (4 - x))
: m = -1 := sorry

end fractional_equation_m_value_l1796_179638


namespace inequality_for_positive_real_numbers_l1796_179653

theorem inequality_for_positive_real_numbers (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    x^2 + 2*y^2 + 3*z^2 > x*y + 3*y*z + z*x := 
by 
  sorry

end inequality_for_positive_real_numbers_l1796_179653


namespace union_complement_l1796_179642

def universalSet : Set ℤ := { x | x^2 < 9 }

def A : Set ℤ := {1, 2}

def B : Set ℤ := {-2, -1, 2}

def complement_I_B : Set ℤ := universalSet \ B

theorem union_complement :
  A ∪ complement_I_B = {0, 1, 2} :=
by
  sorry

end union_complement_l1796_179642


namespace gravel_weight_40_pounds_l1796_179620

def weight_of_gravel_in_mixture (total_weight : ℝ) (sand_fraction : ℝ) (water_fraction : ℝ) : ℝ :=
total_weight - (sand_fraction * total_weight + water_fraction * total_weight)

theorem gravel_weight_40_pounds
  (total_weight : ℝ) (sand_fraction : ℝ) (water_fraction : ℝ) 
  (h1 : total_weight = 40) (h2 : sand_fraction = 1 / 4) (h3 : water_fraction = 2 / 5) :
  weight_of_gravel_in_mixture total_weight sand_fraction water_fraction = 14 :=
by
  -- Proof omitted
  sorry

end gravel_weight_40_pounds_l1796_179620


namespace last_day_of_third_quarter_l1796_179654

def is_common_year (year: Nat) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0) 

def days_in_month (year: Nat) (month: Nat) : Nat :=
  if month = 2 then 28
  else if month = 4 ∨ month = 6 ∨ month = 9 ∨ month = 11 then 30
  else 31

def last_day_of_month (year: Nat) (month: Nat) : Nat :=
  days_in_month year month

theorem last_day_of_third_quarter (year: Nat) (h : is_common_year year) : last_day_of_month year 9 = 30 :=
by
  sorry

end last_day_of_third_quarter_l1796_179654


namespace tobee_points_l1796_179686

theorem tobee_points (T J S : ℕ) (h1 : J = T + 6) (h2 : S = 2 * (T + 3) - 2) (h3 : T + J + S = 26) : T = 4 := 
by
  sorry

end tobee_points_l1796_179686


namespace production_in_three_minutes_l1796_179641

noncomputable def production_rate_per_machine (total_bottles : ℕ) (num_machines : ℕ) : ℕ :=
  total_bottles / num_machines

noncomputable def production_per_minute (machines : ℕ) (rate_per_machine : ℕ) : ℕ :=
  machines * rate_per_machine

noncomputable def total_production (production_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  production_per_minute * minutes

theorem production_in_three_minutes :
  ∀ (total_bottles : ℕ) (num_machines : ℕ) (machines : ℕ) (minutes : ℕ),
  total_bottles = 16 → num_machines = 4 → machines = 8 → minutes = 3 →
  total_production (production_per_minute machines (production_rate_per_machine total_bottles num_machines)) minutes = 96 :=
by
  intros total_bottles num_machines machines minutes h_total_bottles h_num_machines h_machines h_minutes
  sorry

end production_in_three_minutes_l1796_179641


namespace finishing_order_l1796_179672

-- Definitions of conditions
def athletes := ["Grisha", "Sasha", "Lena"]

def overtakes : (String → ℕ) := 
  fun athlete =>
    if athlete = "Grisha" then 10
    else if athlete = "Sasha" then 4
    else if athlete = "Lena" then 6
    else 0

-- All three were never at the same point at the same time
def never_same_point_at_same_time : Prop := True -- Simplified for translation purpose

-- The main theorem stating the finishing order given the provided conditions
theorem finishing_order :
  never_same_point_at_same_time →
  (overtakes "Grisha" = 10) →
  (overtakes "Sasha" = 4) →
  (overtakes "Lena" = 6) →
  athletes = ["Grisha", "Sasha", "Lena"] :=
  by
    intro h1 h2 h3 h4
    exact sorry -- The proof is not required, just ensuring the statement is complete.


end finishing_order_l1796_179672


namespace eval_expression_l1796_179631

theorem eval_expression : 5 * 7 + 9 * 4 - 36 / 3 = 59 :=
by sorry

end eval_expression_l1796_179631


namespace sum_of_two_consecutive_negative_integers_l1796_179614

theorem sum_of_two_consecutive_negative_integers (n : ℤ) (h : n * (n + 1) = 812) (h_neg : n < 0 ∧ (n + 1) < 0) : 
  n + (n + 1) = -57 :=
sorry

end sum_of_two_consecutive_negative_integers_l1796_179614


namespace star_3_5_l1796_179624

def star (a b : ℕ) : ℕ := a^2 + 3 * a * b + b^2

theorem star_3_5 : star 3 5 = 79 := 
by
  sorry

end star_3_5_l1796_179624


namespace units_digit_x4_invx4_l1796_179648

theorem units_digit_x4_invx4 (x : ℝ) (h : x^2 - 12 * x + 1 = 0) : 
  (x^4 + (1 / x)^4) % 10 = 2 := 
by
  sorry

end units_digit_x4_invx4_l1796_179648


namespace five_diff_numbers_difference_l1796_179682

theorem five_diff_numbers_difference (S : Finset ℕ) (hS_size : S.card = 5) 
    (hS_range : ∀ x ∈ S, x ≤ 10) : 
    ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ c ≠ d ∧ a - b = c - d ∧ a - b ≠ 0 :=
by
  sorry

end five_diff_numbers_difference_l1796_179682


namespace function_increasing_on_R_l1796_179618

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

theorem function_increasing_on_R (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a ≤ f x₂ a) ↔ (2 ≤ a ∧ a < 3) :=
by
  sorry

end function_increasing_on_R_l1796_179618


namespace range_of_a_l1796_179601

theorem range_of_a (a : ℝ) (h : ∃ α β : ℝ, (α + β = -(a^2 - 1)) ∧ (α * β = a - 2) ∧ (1 < α ∧ β < 1) ∨ (α < 1 ∧ 1 < β)) :
  -2 < a ∧ a < 1 :=
sorry

end range_of_a_l1796_179601


namespace prime_between_30_40_with_remainder_l1796_179602

theorem prime_between_30_40_with_remainder :
  ∃ n : ℕ, Prime n ∧ 30 < n ∧ n < 40 ∧ n % 9 = 4 ∧ n = 31 :=
by
  sorry

end prime_between_30_40_with_remainder_l1796_179602


namespace probability_sum_less_than_16_l1796_179649

-- The number of possible outcomes when three six-sided dice are rolled
def total_outcomes : ℕ := 6 * 6 * 6

-- The number of favorable outcomes where the sum of the dice is less than 16
def favorable_outcomes : ℕ := (6 * 6 * 6) - (3 + 3 + 3 + 1)

-- The probability that the sum of the dice is less than 16
def probability_less_than_16 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_less_than_16 : probability_less_than_16 = 103 / 108 := 
by sorry

end probability_sum_less_than_16_l1796_179649


namespace opposite_numbers_pow_sum_zero_l1796_179697

theorem opposite_numbers_pow_sum_zero (a b : ℝ) (h : a + b = 0) : a^5 + b^5 = 0 :=
by sorry

end opposite_numbers_pow_sum_zero_l1796_179697


namespace product_of_bc_l1796_179632

theorem product_of_bc (b c : ℤ) 
  (h : ∀ r, r^2 - r - 2 = 0 → r^5 - b * r - c = 0) : b * c = 110 :=
sorry

end product_of_bc_l1796_179632


namespace problem1_problem2_l1796_179651

-- Problem 1
theorem problem1 : (1 / 2) ^ (-2 : ℤ) - (Real.pi - Real.sqrt 5) ^ 0 - Real.sqrt 20 = 3 - 2 * Real.sqrt 5 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ -1) : 
  ((x ^ 2 - 2 * x + 1) / (x ^ 2 - 1)) / ((x - 1) / (x ^ 2 + x)) = x :=
by sorry

end problem1_problem2_l1796_179651


namespace range_of_a_l1796_179669

-- Define sets A and B
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def set_B (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Mathematical statement to be proven
theorem range_of_a (a : ℝ) : (∃ x, x ∈ set_A ∧ x ∈ set_B a) → a ≥ -1 :=
by
  sorry

end range_of_a_l1796_179669


namespace original_plan_trees_per_day_l1796_179674

theorem original_plan_trees_per_day (x : ℕ) :
  (∃ x, (960 / x - 960 / (2 * x) = 4)) → x = 120 := 
sorry

end original_plan_trees_per_day_l1796_179674


namespace remainder_53_pow_10_div_8_l1796_179630

theorem remainder_53_pow_10_div_8 : (53^10) % 8 = 1 := 
by sorry

end remainder_53_pow_10_div_8_l1796_179630


namespace system_solution_l1796_179667

theorem system_solution (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 3) : x - y = 3 :=
by
  -- proof goes here
  sorry

end system_solution_l1796_179667


namespace S6_equals_63_l1796_179683

variable {S : ℕ → ℕ}

-- Define conditions
axiom S_n_geometric_sequence (a : ℕ → ℕ) (n : ℕ) : n ≥ 1 → S n = (a 0) * ((a 1)^(n) -1) / (a 1 - 1)
axiom S_2_eq_3 : S 2 = 3
axiom S_4_eq_15 : S 4 = 15

-- State theorem
theorem S6_equals_63 : S 6 = 63 := by
  sorry

end S6_equals_63_l1796_179683


namespace find_y_in_terms_of_abc_l1796_179663

theorem find_y_in_terms_of_abc 
  (x y z a b c : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0)
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0)
  (h1 : xy / (x - y) = a)
  (h2 : xz / (x - z) = b)
  (h3 : yz / (y - z) = c) :
  y = bcx / ((b + c) * x - bc) := 
sorry

end find_y_in_terms_of_abc_l1796_179663


namespace tablecloth_width_l1796_179643

theorem tablecloth_width (length_tablecloth : ℕ) (napkins_count : ℕ) (napkin_length : ℕ) (napkin_width : ℕ) (total_material : ℕ) (width_tablecloth : ℕ) :
  length_tablecloth = 102 →
  napkins_count = 8 →
  napkin_length = 6 →
  napkin_width = 7 →
  total_material = 5844 →
  total_material = length_tablecloth * width_tablecloth + napkins_count * (napkin_length * napkin_width) →
  width_tablecloth = 54 :=
by
  intros h1 h2 h3 h4 h5 h_eq
  sorry

end tablecloth_width_l1796_179643


namespace alternating_sequence_probability_l1796_179607

theorem alternating_sequence_probability : 
  let total_balls := 10 -- Total number of balls
  let white_balls := 5 -- Number of white balls
  let black_balls := 5 -- Number of black balls
  let successful_sequences := 2 -- Number of successful alternating sequences (BWBWBWBWBW and WBWBWBWBWB)
  let total_arrangements := Nat.choose total_balls white_balls -- Binomial coefficient for total arrangements
  (successful_sequences : ℚ) / total_arrangements = 1 / 126 :=
by
  sorry

end alternating_sequence_probability_l1796_179607


namespace park_area_l1796_179679

theorem park_area (l w : ℝ) (h1 : l + w = 40) (h2 : l = 3 * w) : l * w = 300 :=
by
  sorry

end park_area_l1796_179679


namespace find_large_number_l1796_179690

theorem find_large_number (L S : ℤ)
  (h1 : L - S = 2415)
  (h2 : L = 21 * S + 15) : 
  L = 2535 := 
sorry

end find_large_number_l1796_179690


namespace eight_p_plus_one_is_composite_l1796_179657

theorem eight_p_plus_one_is_composite (p : ℕ) (hp : Nat.Prime p) (h8p1 : Nat.Prime (8 * p - 1)) : ¬ Nat.Prime (8 * p + 1) :=
by
  sorry

end eight_p_plus_one_is_composite_l1796_179657
