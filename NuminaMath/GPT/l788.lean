import Mathlib

namespace cos_angle_sum_eq_negative_sqrt_10_div_10_l788_78826

theorem cos_angle_sum_eq_negative_sqrt_10_div_10 
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) :
  Real.cos (α + π / 4) = - Real.sqrt 10 / 10 := by
  sorry

end cos_angle_sum_eq_negative_sqrt_10_div_10_l788_78826


namespace total_mile_times_l788_78873

-- Define the conditions
def Tina_time : ℕ := 6  -- Tina runs a mile in 6 minutes

def Tony_time : ℕ := Tina_time / 2  -- Tony runs twice as fast as Tina

def Tom_time : ℕ := Tina_time / 3  -- Tom runs three times as fast as Tina

-- Define the proof statement
theorem total_mile_times : Tony_time + Tina_time + Tom_time = 11 := by
  sorry

end total_mile_times_l788_78873


namespace total_weight_of_containers_l788_78874

theorem total_weight_of_containers (x y z : ℕ) :
  x + y = 162 →
  y + z = 168 →
  z + x = 174 →
  x + y + z = 252 :=
by
  intros hxy hyz hzx
  -- proof skipped
  sorry

end total_weight_of_containers_l788_78874


namespace generalized_inequality_l788_78848

theorem generalized_inequality (n k : ℕ) (h1 : 3 ≤ n) (h2 : 1 ≤ k ∧ k ≤ n) : 
  2^n + 5^n > 2^(n - k) * 5^k + 2^k * 5^(n - k) := 
by 
  sorry

end generalized_inequality_l788_78848


namespace largest_divisor_for_odd_n_l788_78812

noncomputable def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem largest_divisor_for_odd_n (n : ℤ) (h : is_odd n ∧ n > 0) : 
  15 ∣ (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) := 
by 
  sorry

end largest_divisor_for_odd_n_l788_78812


namespace tan_of_angle_in_fourth_quadrant_l788_78888

theorem tan_of_angle_in_fourth_quadrant (α : ℝ) (h1 : Real.sin α = -5 / 13) (h2 : α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) :
  Real.tan α = -5 / 12 :=
sorry

end tan_of_angle_in_fourth_quadrant_l788_78888


namespace selling_price_correct_l788_78860

-- Define the conditions
def cost_per_cupcake : ℝ := 0.75
def total_cupcakes_burnt : ℕ := 24
def total_eaten_first : ℕ := 5
def total_eaten_later : ℕ := 4
def net_profit : ℝ := 24
def total_cupcakes_made : ℕ := 72
def total_cost : ℝ := total_cupcakes_made * cost_per_cupcake
def total_eaten : ℕ := total_eaten_first + total_eaten_later
def total_sold : ℕ := total_cupcakes_made - total_eaten
def revenue (P : ℝ) : ℝ := total_sold * P

-- Prove the correctness of the selling price P
theorem selling_price_correct : 
  ∃ P : ℝ, revenue P - total_cost = net_profit ∧ (P = 1.24) :=
by
  sorry

end selling_price_correct_l788_78860


namespace changed_answers_percentage_l788_78810

variables (n : ℕ) (a b c d : ℕ)

theorem changed_answers_percentage (h1 : a + b + c + d = 100)
  (h2 : a + d + c = 50)
  (h3 : a + c = 60)
  (h4 : b + d = 40) :
  10 ≤ c + d ∧ c + d ≤ 90 :=
  by sorry

end changed_answers_percentage_l788_78810


namespace binomial_coefficient_divisible_by_p_l788_78844

theorem binomial_coefficient_divisible_by_p (p k : ℕ) (hp : Nat.Prime p) (hk1 : 0 < k) (hk2 : k < p) :
  p ∣ (Nat.factorial p / (Nat.factorial k * Nat.factorial (p - k))) :=
by
  sorry

end binomial_coefficient_divisible_by_p_l788_78844


namespace probability_of_double_tile_is_one_fourth_l788_78857

noncomputable def probability_double_tile : ℚ :=
  let total_pairs := (7 * 7) / 2
  let double_pairs := 7
  double_pairs / total_pairs

theorem probability_of_double_tile_is_one_fourth :
  probability_double_tile = 1 / 4 :=
by
  sorry

end probability_of_double_tile_is_one_fourth_l788_78857


namespace number_of_outfits_l788_78899

-- Define the counts of each item
def redShirts : Nat := 6
def greenShirts : Nat := 4
def pants : Nat := 7
def greenHats : Nat := 10
def redHats : Nat := 9

-- Total number of outfits satisfying the conditions
theorem number_of_outfits :
  (redShirts * greenHats * pants) + (greenShirts * redHats * pants) = 672 :=
by
  sorry

end number_of_outfits_l788_78899


namespace unique_solution_values_l788_78840

theorem unique_solution_values (a : ℝ) :
  (∃! x : ℝ, a * x^2 - x + 1 = 0) ↔ (a = 0 ∨ a = 1 / 4) :=
by
  sorry

end unique_solution_values_l788_78840


namespace greatest_prime_factor_of_154_l788_78801

theorem greatest_prime_factor_of_154 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 154 ∧ (∀ q : ℕ, Nat.Prime q → q ∣ 154 → q ≤ p) :=
  sorry

end greatest_prime_factor_of_154_l788_78801


namespace tiling_possible_l788_78850

theorem tiling_possible (n x : ℕ) (hx : 7 * x = n^2) : ∃ k : ℕ, n = 7 * k :=
by sorry

end tiling_possible_l788_78850


namespace simplify_sum_of_square_roots_l788_78852

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l788_78852


namespace cos_value_l788_78889

theorem cos_value (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (4 * π / 3 - α) = -3 / 5 := 
by 
  sorry

end cos_value_l788_78889


namespace money_last_weeks_l788_78895

-- Define the conditions
def dollars_mowing : ℕ := 68
def dollars_weed_eating : ℕ := 13
def dollars_per_week : ℕ := 9

-- Define the total money made
def total_dollars := dollars_mowing + dollars_weed_eating

-- State the theorem to prove the question
theorem money_last_weeks : (total_dollars / dollars_per_week) = 9 :=
by
  sorry

end money_last_weeks_l788_78895


namespace largest_whole_number_l788_78885

theorem largest_whole_number (n : ℤ) (h : (1 : ℝ) / 4 + n / 8 < 2) : n ≤ 13 :=
by {
  sorry
}

end largest_whole_number_l788_78885


namespace area_difference_l788_78864

theorem area_difference (r1 d2 : ℝ) (h1 : r1 = 30) (h2 : d2 = 15) : 
  π * r1^2 - π * (d2 / 2)^2 = 843.75 * π :=
by
  sorry

end area_difference_l788_78864


namespace binom_six_two_l788_78855

-- Define the binomial coefficient function
def binom (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- State the theorem
theorem binom_six_two : binom 6 2 = 15 := by
  sorry

end binom_six_two_l788_78855


namespace charge_two_hours_l788_78833

def charge_first_hour (F A : ℝ) : Prop := F = A + 25
def total_charge_five_hours (F A : ℝ) : Prop := F + 4 * A = 250
def total_charge_two_hours (F A : ℝ) : Prop := F + A = 115

theorem charge_two_hours (F A : ℝ) 
  (h1 : charge_first_hour F A)
  (h2 : total_charge_five_hours F A) : 
  total_charge_two_hours F A :=
by
  sorry

end charge_two_hours_l788_78833


namespace no_parallelepiped_exists_l788_78823

theorem no_parallelepiped_exists 
  (xyz_half_volume: ℝ)
  (xy_plus_yz_plus_zx_half_surface_area: ℝ) 
  (sum_of_squares_eq_4: ℝ) : 
  ¬(∃ x y z : ℝ, (x * y * z = xyz_half_volume) ∧ 
                 (x * y + y * z + z * x = xy_plus_yz_plus_zx_half_surface_area) ∧ 
                 (x^2 + y^2 + z^2 = sum_of_squares_eq_4)) := 
by
  let xyz_half_volume := 2 * Real.pi / 3
  let xy_plus_yz_plus_zx_half_surface_area := Real.pi
  let sum_of_squares_eq_4 := 4
  sorry

end no_parallelepiped_exists_l788_78823


namespace translate_line_down_l788_78879

theorem translate_line_down (k : ℝ) (b : ℝ) : 
  (∀ x : ℝ, b = 0 → (y = k * x - 3) = (y = k * x - 3)) :=
by
  sorry

end translate_line_down_l788_78879


namespace complete_the_square_eqn_l788_78856

theorem complete_the_square_eqn (x b c : ℤ) (h_eqn : x^2 - 10 * x + 15 = 0) (h_form : (x + b)^2 = c) : b + c = 5 := by
  sorry

end complete_the_square_eqn_l788_78856


namespace correct_calculation_l788_78845

-- Definitions of the conditions
def condition_A (a : ℝ) : Prop := a^2 + a^2 = a^4
def condition_B (a : ℝ) : Prop := 3 * a^2 + 2 * a^2 = 5 * a^2
def condition_C (a : ℝ) : Prop := a^4 - a^2 = a^2
def condition_D (a : ℝ) : Prop := 3 * a^2 - 2 * a^2 = 1

-- The theorem statement
theorem correct_calculation (a : ℝ) : condition_B a := by 
sorry

end correct_calculation_l788_78845


namespace positive_int_sum_square_l788_78800

theorem positive_int_sum_square (M : ℕ) (h_pos : 0 < M) (h_eq : M^2 + M = 12) : M = 3 :=
by
  sorry

end positive_int_sum_square_l788_78800


namespace largest_number_is_correct_l788_78859

theorem largest_number_is_correct (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 3) : c = 33.25 :=
by
  sorry

end largest_number_is_correct_l788_78859


namespace lino_shells_l788_78813

theorem lino_shells (picked_up : ℝ) (put_back : ℝ) (remaining_shells : ℝ) :
  picked_up = 324.0 → 
  put_back = 292.0 → 
  remaining_shells = picked_up - put_back → 
  remaining_shells = 32.0 :=
by
  intros h1 h2 h3
  sorry

end lino_shells_l788_78813


namespace remainders_mod_m_l788_78802

theorem remainders_mod_m {m n b : ℤ} (h_coprime : Int.gcd m n = 1) :
    (∀ r : ℤ, 0 ≤ r ∧ r < m → ∃ k : ℤ, 0 ≤ k ∧ k < n ∧ ((b + k * n) % m = r)) :=
by
  sorry

end remainders_mod_m_l788_78802


namespace solution_set_of_inequality_l788_78893

theorem solution_set_of_inequality (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = f x) (h2 : ∀ x, 0 ≤ x → f x = x - 1) :
  { x : ℝ | f (x - 1) > 1 } = { x | x < -1 ∨ x > 3 } :=
by
  sorry

end solution_set_of_inequality_l788_78893


namespace inequality_solution_set_l788_78896

theorem inequality_solution_set (a b c : ℝ)
  (h1 : ∀ x, (ax^2 + bx + c > 0 ↔ -3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, ¬ (bx + c > 0 ↔ x > 6)) ∧ (∀ x, (cx^2 + bx + a < 0 ↔ -1/3 < x ∧ x < 1/2)) :=
by
  sorry

end inequality_solution_set_l788_78896


namespace genetic_recombination_does_not_occur_during_dna_replication_l788_78818

-- Definitions based on conditions
def dna_replication_spermatogonial_cells : Prop := 
  ∃ dna_interphase: Prop, ∃ dna_unwinding: Prop, 
    ∃ gene_mutation: Prop, ∃ protein_synthesis: Prop,
      dna_interphase ∧ dna_unwinding ∧ gene_mutation ∧ protein_synthesis

def genetic_recombination_not_occur : Prop :=
  ¬ ∃ genetic_recombination: Prop, genetic_recombination

-- Proof problem statement
theorem genetic_recombination_does_not_occur_during_dna_replication : 
  dna_replication_spermatogonial_cells → genetic_recombination_not_occur :=
by sorry

end genetic_recombination_does_not_occur_during_dna_replication_l788_78818


namespace solution_set_l788_78831

noncomputable def f : ℝ → ℝ := sorry

-- The function f is defined to be odd.
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- The function f is increasing on (-∞, 0).
axiom increasing_f : ∀ x y : ℝ, x < y ∧ y < 0 → f x < f y

-- Given f(2) = 0
axiom f_at_2 : f 2 = 0

-- Prove the solution set for x f(x + 1) < 0
theorem solution_set : { x : ℝ | x * f (x + 1) < 0 } = {x : ℝ | (-3 < x ∧ x < -1) ∨ (0 < x ∧ x < 1)} :=
by
  sorry

end solution_set_l788_78831


namespace rectangle_perimeter_l788_78875

theorem rectangle_perimeter (t s : ℝ) (h : t ≥ s) : 2 * (t - s) + 2 * s = 2 * t := 
by 
  sorry

end rectangle_perimeter_l788_78875


namespace M_inter_N_eq_M_l788_78842

-- Definitions of the sets M and N
def M : Set ℝ := {x | abs (x - 1) < 1}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- The desired equality
theorem M_inter_N_eq_M : M ∩ N = M := 
by
  sorry

end M_inter_N_eq_M_l788_78842


namespace uncle_money_given_l788_78881

-- Definitions
def lizzy_mother_money : Int := 80
def lizzy_father_money : Int := 40
def candy_expense : Int := 50
def total_money_now : Int := 140

-- Theorem to prove
theorem uncle_money_given : (total_money_now - ((lizzy_mother_money + lizzy_father_money) - candy_expense)) = 70 := 
  by
    sorry

end uncle_money_given_l788_78881


namespace savings_percentage_correct_l788_78817

variables (price_jacket : ℕ) (price_shirt : ℕ) (price_hat : ℕ)
          (discount_jacket : ℕ) (discount_shirt : ℕ) (discount_hat : ℕ)

def original_total_cost (price_jacket price_shirt price_hat : ℕ) : ℕ :=
  price_jacket + price_shirt + price_hat

def savings (price : ℕ) (discount : ℕ) : ℕ :=
  price * discount / 100

def total_savings (price_jacket price_shirt price_hat : ℕ)
  (discount_jacket discount_shirt discount_hat : ℕ) : ℕ :=
  (savings price_jacket discount_jacket) + (savings price_shirt discount_shirt) + (savings price_hat discount_hat)

def total_savings_percentage (price_jacket price_shirt price_hat : ℕ)
  (discount_jacket discount_shirt discount_hat : ℕ) : ℕ :=
  total_savings price_jacket price_shirt price_hat discount_jacket discount_shirt discount_hat * 100 /
  original_total_cost price_jacket price_shirt price_hat

theorem savings_percentage_correct :
  total_savings_percentage 100 50 30 30 60 50 = 4167 / 100 :=
sorry

end savings_percentage_correct_l788_78817


namespace point_outside_circle_l788_78839

theorem point_outside_circle (a b : ℝ) (h : ∃ (x y : ℝ), (a*x + b*y = 1 ∧ x^2 + y^2 = 1)) : a^2 + b^2 ≥ 1 :=
sorry

end point_outside_circle_l788_78839


namespace number_of_ordered_pairs_l788_78882

-- Formal statement of the problem in Lean 4
theorem number_of_ordered_pairs : 
  ∃ (n : ℕ), n = 128 ∧ 
  ∀ (a b : ℝ), (∃ (x y : ℤ), (a * x + b * y = 1) ∧ (x^2 + y^2 = 65)) ↔ n = 128 :=
sorry

end number_of_ordered_pairs_l788_78882


namespace inverse_proportion_function_increasing_l788_78836

theorem inverse_proportion_function_increasing (m : ℝ) :
  (∀ x1 x2 : ℝ, (0 < x1) → (x1 < x2) → (y = (m - 5) / x1) < (y = (m - 5) / x2)) ↔ m < 5 :=
by
  sorry

end inverse_proportion_function_increasing_l788_78836


namespace smallest_value_A_B_C_D_l788_78846

theorem smallest_value_A_B_C_D :
  ∃ (A B C D : ℕ), 
  (A < B) ∧ (B < C) ∧ (C < D) ∧ -- A, B, C are in arithmetic sequence and B, C, D in geometric sequence
  (C = B + (B - A)) ∧  -- A, B, C form an arithmetic sequence with common difference d = B - A
  (C = (4 * B) / 3) ∧  -- Given condition
  (D = (4 * C) / 3) ∧ -- B, C, D form geometric sequence with common ratio 4/3
  ((∃ k, D = k * 9) ∧ -- D must be an integer, ensuring B must be divisible by 9
   A + B + C + D = 43) := 
sorry

end smallest_value_A_B_C_D_l788_78846


namespace problem1_problem2_l788_78816

open Set Real

-- Given A and B
def A (a : ℝ) : Set ℝ := {x | x > a}
def B : Set ℝ := {y | y > -1}

-- Problem 1: If A = B, then a = -1
theorem problem1 (a : ℝ) (h : A a = B) : a = -1 := by
  sorry

-- Problem 2: If (complement of A) ∩ B ≠ ∅, find the range of a
theorem problem2 (a : ℝ) (h : (compl (A a)) ∩ B ≠ ∅) : a ∈ Ioi (-1) := by
  sorry

end problem1_problem2_l788_78816


namespace number_of_mixed_groups_l788_78824

theorem number_of_mixed_groups (n_children n_groups n_games boy_vs_boy girl_vs_girl mixed_games : ℕ) (h_children : n_children = 90) (h_groups : n_groups = 30) (h_games_per_group : n_games = 3) (h_boy_vs_boy : boy_vs_boy = 30) (h_girl_vs_girl : girl_vs_girl = 14) (h_total_games : mixed_games = 46) :
  (∀ g : ℕ, g * 2 = mixed_games → g = 23) :=
by
  intros g hg
  sorry

end number_of_mixed_groups_l788_78824


namespace retail_price_before_discount_l788_78821

variable (R : ℝ) -- Let R be the retail price of each machine before the discount

theorem retail_price_before_discount :
    let wholesale_price := 126
    let machines := 10
    let bulk_discount_rate := 0.05
    let profit_margin := 0.20
    let sales_tax_rate := 0.07
    let discount_rate := 0.10

    -- Calculate wholesale total price
    let wholesale_total := machines * wholesale_price

    -- Calculate bulk purchase discount
    let bulk_discount := bulk_discount_rate * wholesale_total

    -- Calculate total amount paid
    let amount_paid := wholesale_total - bulk_discount

    -- Calculate profit per machine
    let profit_per_machine := profit_margin * wholesale_price
    
    -- Calculate total profit
    let total_profit := machines * profit_per_machine

    -- Calculate sales tax on profit
    let tax_on_profit := sales_tax_rate * total_profit

    -- Calculate total amount after paying tax
    let total_amount_after_tax := (amount_paid + total_profit) - tax_on_profit

    -- Express total selling price after discount
    let total_selling_after_discount := machines * (0.90 * R)

    -- Total selling price after discount is equal to total amount after tax
    (9 * R = total_amount_after_tax) →
    R = 159.04 :=
by
  sorry

end retail_price_before_discount_l788_78821


namespace speed_of_A_is_3_l788_78878

theorem speed_of_A_is_3:
  (∃ x : ℝ, 3 * x + 3 * (x + 2) = 24) → x = 3 :=
by
  sorry

end speed_of_A_is_3_l788_78878


namespace expression_evaluation_l788_78898

theorem expression_evaluation (m : ℝ) (h : m = Real.sqrt 2023 + 2) : m^2 - 4 * m + 5 = 2024 :=
by sorry

end expression_evaluation_l788_78898


namespace floor_function_solution_l788_78819

def floor_eq_x_solutions : Prop :=
  ∀ x : ℤ, (⌊(x : ℝ) / 2⌋ + ⌊(x : ℝ) / 4⌋ = x) ↔ x = 0 ∨ x = -3 ∨ x = -2 ∨ x = -5

theorem floor_function_solution: floor_eq_x_solutions :=
by
  intro x
  sorry

end floor_function_solution_l788_78819


namespace unit_digit_15_pow_100_l788_78870

theorem unit_digit_15_pow_100 : ((15^100) % 10) = 5 := 
by sorry

end unit_digit_15_pow_100_l788_78870


namespace measure_of_angle_A_l788_78822

variables (A B C a b c : ℝ)
variables (triangle_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
variables (sides_relation : (a^2 + b^2 - c^2) * tan A = a * b)

theorem measure_of_angle_A :
  A = π / 6 :=
by 
  sorry

end measure_of_angle_A_l788_78822


namespace time_to_travel_A_to_C_is_6_l788_78853

-- Assume the existence of a real number t representing the time taken
-- Assume constant speed r for the river current and p for the power boat relative to the river.
variables (t r p : ℝ)

-- Conditions
axiom condition1 : p > 0
axiom condition2 : r > 0
axiom condition3 : t * (1.5 * (p + r)) + (p - r) * (12 - t) = 12 * r

-- Define the time taken for the power boat to travel from A to C
def time_from_A_to_C : ℝ := t

-- The proof problem: Prove time_from_A_to_C = 6 under the given conditions
theorem time_to_travel_A_to_C_is_6 : time_from_A_to_C = 6 := by
  sorry

end time_to_travel_A_to_C_is_6_l788_78853


namespace binary_111_to_decimal_l788_78868

-- Define a function to convert binary list to decimal
def binaryToDecimal (bin : List ℕ) : ℕ :=
  bin.reverse.enumFrom 0 |>.foldl (λ acc ⟨i, b⟩ => acc + b * (2 ^ i)) 0

-- Assert the equivalence between the binary number [1, 1, 1] and its decimal representation 7
theorem binary_111_to_decimal : binaryToDecimal [1, 1, 1] = 7 :=
  by
  sorry

end binary_111_to_decimal_l788_78868


namespace luncheon_cost_l788_78830

variable (s c p : ℝ)
variable (eq1 : 5 * s + 9 * c + 2 * p = 6.50)
variable (eq2 : 7 * s + 14 * c + 3 * p = 9.45)
variable (eq3 : 4 * s + 8 * c + p = 5.20)

theorem luncheon_cost :
  s + c + p = 1.30 :=
by
  sorry

end luncheon_cost_l788_78830


namespace problem_statement_l788_78804

-- Definition of operation nabla
def nabla (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2

-- Main theorem statement
theorem problem_statement : nabla 2 (nabla 0 (nabla 1 7)) = 71859 :=
by
  -- Computational proof
  sorry

end problem_statement_l788_78804


namespace product_of_two_numbers_l788_78863

theorem product_of_two_numbers (x y : ℚ) 
  (h1 : x + y = 8 * (x - y)) 
  (h2 : x * y = 15 * (x - y)) : 
  x * y = 100 / 7 := 
by 
  sorry

end product_of_two_numbers_l788_78863


namespace f_f_1_equals_4_l788_78807

noncomputable def f (x : ℝ) : ℝ :=
  if x > 2 then x + 1 / (x - 2) else x^2 + 2

theorem f_f_1_equals_4 : f (f 1) = 4 := by sorry

end f_f_1_equals_4_l788_78807


namespace amounts_are_correct_l788_78811

theorem amounts_are_correct (P Q R S : ℕ) 
    (h1 : P + Q + R + S = 10000)
    (h2 : R = 2 * P)
    (h3 : R = 3 * Q)
    (h4 : S = P + Q) :
    P = 1875 ∧ Q = 1250 ∧ R = 3750 ∧ S = 3125 := by
  sorry

end amounts_are_correct_l788_78811


namespace walnut_trees_in_park_l788_78887

def num_current_walnut_trees (num_plant : ℕ) (num_total : ℕ) : ℕ :=
  num_total - num_plant

theorem walnut_trees_in_park :
  num_current_walnut_trees 6 10 = 4 :=
by
  -- By the definition of num_current_walnut_trees
  -- We have 10 (total) - 6 (to be planted) = 4 (current)
  sorry

end walnut_trees_in_park_l788_78887


namespace magpies_gather_7_trees_magpies_not_gather_6_trees_l788_78892

-- Define the problem conditions.
def trees (n : ℕ) := (∀ (i : ℕ), i < n → ∃ (m : ℕ), m = i * 10)

-- Define the movement condition for magpies.
def magpie_move (n : ℕ) (d : ℕ) :=
  (∀ (i j : ℕ), i < n ∧ j < n ∧ i ≠ j → ∃ (k : ℕ), k = d ∧ ((i + d < n ∧ j - d < n) ∨ (i - d < n ∧ j + d < n)))

-- Prove that all magpies can gather on one tree for 7 trees.
theorem magpies_gather_7_trees : 
  ∃ (i : ℕ), i < 7 ∧ trees 7 ∧ magpie_move 7 (i * 10) → True :=
by
  -- proof steps here, which are not necessary for the task
  sorry

-- Prove that all magpies cannot gather on one tree for 6 trees.
theorem magpies_not_gather_6_trees : 
  ∀ (i : ℕ), i < 6 ∧ trees 6 ∧ magpie_move 6 (i * 10) → False :=
by
  -- proof steps here, which are not necessary for the task
  sorry

end magpies_gather_7_trees_magpies_not_gather_6_trees_l788_78892


namespace carnations_percentage_l788_78825

-- Definition of the total number of flowers
def total_flowers (F : ℕ) : Prop := 
  F > 0

-- Definition of the pink roses condition
def pink_roses_condition (F : ℕ) : Prop := 
  (1 / 2) * (3 / 5) * F = (3 / 10) * F

-- Definition of the red carnations condition
def red_carnations_condition (F : ℕ) : Prop := 
  (1 / 3) * (2 / 5) * F = (2 / 15) * F

-- Definition of the total pink flowers
def pink_flowers_condition (F : ℕ) : Prop :=
  (3 / 5) * F > 0

-- Proof that the percentage of the flowers that are carnations is 50%
theorem carnations_percentage (F : ℕ) (h_total : total_flowers F) (h_pink_roses : pink_roses_condition F) (h_red_carnations : red_carnations_condition F) (h_pink_flowers : pink_flowers_condition F) :
  (1 / 2) * 100 = 50 :=
by
  sorry

end carnations_percentage_l788_78825


namespace solve_for_x_l788_78815

theorem solve_for_x (x : ℤ) (h : 20 * 14 + x = 20 + 14 * x) : x = 20 := 
by 
  sorry

end solve_for_x_l788_78815


namespace average_weight_women_l788_78827

variable (average_weight_men : ℕ) (number_of_men : ℕ)
variable (average_weight : ℕ) (number_of_women : ℕ)
variable (average_weight_all : ℕ) (total_people : ℕ)

theorem average_weight_women (h1 : average_weight_men = 190) 
                            (h2 : number_of_men = 8)
                            (h3 : average_weight_all = 160)
                            (h4 : total_people = 14) 
                            (h5 : number_of_women = 6):
  average_weight = 120 := 
by
  sorry

end average_weight_women_l788_78827


namespace value_of_a_l788_78890

theorem value_of_a (a : ℝ) (H1 : A = a) (H2 : B = 1) (H3 : C = a - 3) (H4 : C + B = 0) : a = 2 := by
  sorry

end value_of_a_l788_78890


namespace max_value_in_interval_l788_78866

theorem max_value_in_interval :
  ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → x^4 - 2 * x^2 + 5 ≤ 13 :=
by
  sorry

end max_value_in_interval_l788_78866


namespace find_positive_real_solutions_l788_78851

variable {x_1 x_2 x_3 x_4 x_5 : ℝ}

theorem find_positive_real_solutions
  (h1 : (x_1^2 - x_3 * x_5) * (x_2^2 - x_3 * x_5) ≤ 0)
  (h2 : (x_2^2 - x_4 * x_1) * (x_3^2 - x_4 * x_1) ≤ 0)
  (h3 : (x_3^2 - x_5 * x_2) * (x_4^2 - x_5 * x_2) ≤ 0)
  (h4 : (x_4^2 - x_1 * x_3) * (x_5^2 - x_1 * x_3) ≤ 0)
  (h5 : (x_5^2 - x_2 * x_4) * (x_1^2 - x_2 * x_4) ≤ 0)
  (hx1 : 0 < x_1)
  (hx2 : 0 < x_2)
  (hx3 : 0 < x_3)
  (hx4 : 0 < x_4)
  (hx5 : 0 < x_5) :
  x_1 = x_2 ∧ x_2 = x_3 ∧ x_3 = x_4 ∧ x_4 = x_5 :=
by
  sorry

end find_positive_real_solutions_l788_78851


namespace max_band_members_l788_78806

variable (r x m : ℕ)

noncomputable def band_formation (r x m: ℕ) :=
  m = r * x + 4 ∧
  m = (r - 3) * (x + 2) ∧
  m < 100

theorem max_band_members (r x m : ℕ) (h : band_formation r x m) : m = 88 :=
by
  sorry

end max_band_members_l788_78806


namespace value_v3_at_1_horners_method_l788_78843

def f (x : ℝ) : ℝ := 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

theorem value_v3_at_1_horners_method :
  let v0 : ℝ := 5
  let v1 : ℝ := v0 * 1 + 2
  let v2 : ℝ := v1 * 1 + 3.5
  let v3 : ℝ := v2 * 1 - 2.6
  let v4 : ℝ := v3 * 1 + 1.7
  let result : ℝ := v4 * 1 - 0.8
  v3 = 7.9 :=
by
  let v0 : ℝ := 5
  let v1 : ℝ := v0 * 1 + 2
  let v2 : ℝ := v1 * 1 + 3.5
  let v3 : ℝ := v2 * 1 - 2.6
  let v4 : ℝ := v3 * 1 + 1.7
  let result : ℝ := v4 * 1 - 0.8
  exact sorry

end value_v3_at_1_horners_method_l788_78843


namespace samantha_interest_l788_78861

-- Definitions based on problem conditions
def P : ℝ := 2000
def r : ℝ := 0.08
def n : ℕ := 5

-- Compound interest calculation
noncomputable def A : ℝ := P * (1 + r) ^ n
noncomputable def Interest : ℝ := A - P

-- Theorem statement with Lean 4
theorem samantha_interest : Interest = 938.656 := 
by 
  sorry

end samantha_interest_l788_78861


namespace swap_equality_l788_78871

theorem swap_equality {a1 b1 a2 b2 : ℝ} 
  (h1 : a1^2 + b1^2 = 1)
  (h2 : a2^2 + b2^2 = 1)
  (h3 : a1 * a2 + b1 * b2 = 0) :
  b1 = a2 ∨ b1 = -a2 :=
by sorry

end swap_equality_l788_78871


namespace terminal_side_third_quadrant_l788_78834

theorem terminal_side_third_quadrant (α : ℝ) (k : ℤ) 
  (hα : (π / 2) + 2 * k * π < α ∧ α < π + 2 * k * π) : 
  ¬(π + 2 * k * π < α / 3 ∧ α / 3 < (3 / 2) * π + 2 * k * π) :=
by
  sorry

end terminal_side_third_quadrant_l788_78834


namespace calculate_teena_speed_l788_78803

noncomputable def Teena_speed (t c t_ahead_in_1_5_hours : ℝ) : ℝ :=
  let distance_initial_gap := 7.5
  let coe_speed := 40
  let time_in_hours := 1.5
  let distance_coe_travels := coe_speed * time_in_hours
  let total_distance_teena_needs := distance_coe_travels + distance_initial_gap + t_ahead_in_1_5_hours
  total_distance_teena_needs / time_in_hours

theorem calculate_teena_speed :
  (Teena_speed 7.5 40 15) = 55 :=
  by
  -- skipped proof
  sorry

end calculate_teena_speed_l788_78803


namespace distinct_bead_arrangements_l788_78872

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n-1)

theorem distinct_bead_arrangements : factorial 8 / (8 * 2) = 2520 := 
  by sorry

end distinct_bead_arrangements_l788_78872


namespace probability_of_sum_23_l788_78858

def is_valid_time (h m : ℕ) : Prop :=
  0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def sum_of_time_digits (h m : ℕ) : ℕ :=
  sum_of_digits h + sum_of_digits m

theorem probability_of_sum_23 :
  (∃ h m, is_valid_time h m ∧ sum_of_time_digits h m = 23) →
  (4 / 1440 : ℚ) = (1 / 360 : ℚ) :=
by
  sorry

end probability_of_sum_23_l788_78858


namespace average_power_heater_l788_78886

structure Conditions where
  (M : ℝ)    -- mass of the piston
  (tau : ℝ)  -- time period τ
  (a : ℝ)    -- constant acceleration
  (c : ℝ)    -- specific heat at constant volume
  (R : ℝ)    -- universal gas constant

theorem average_power_heater (cond : Conditions) : 
  let P := cond.M * cond.a^2 * cond.tau / 2 * (1 + cond.c / cond.R)
  P = (cond.M * cond.a^2 * cond.tau / 2) * (1 + cond.c / cond.R) :=
by
  sorry

end average_power_heater_l788_78886


namespace customer_pays_correct_amount_l788_78854

def wholesale_price : ℝ := 4
def markup : ℝ := 0.25
def discount : ℝ := 0.05

def retail_price : ℝ := wholesale_price * (1 + markup)
def discount_amount : ℝ := retail_price * discount
def customer_price : ℝ := retail_price - discount_amount

theorem customer_pays_correct_amount : customer_price = 4.75 := by
  -- proof steps would go here, but we are skipping them as instructed
  sorry

end customer_pays_correct_amount_l788_78854


namespace pears_left_l788_78835

theorem pears_left (jason_pears : ℕ) (keith_pears : ℕ) (mike_ate : ℕ) (total_pears : ℕ) (pears_left : ℕ) 
  (h1 : jason_pears = 46) 
  (h2 : keith_pears = 47) 
  (h3 : mike_ate = 12) 
  (h4 : total_pears = jason_pears + keith_pears) 
  (h5 : pears_left = total_pears - mike_ate) 
  : pears_left = 81 :=
by
  sorry

end pears_left_l788_78835


namespace acute_angle_probability_l788_78828

noncomputable def prob_acute_angle : ℝ :=
  let m_values := [1, 2, 3, 4, 5, 6]
  let outcomes_count := (36 : ℝ)
  let good_outcomes_count := (15 : ℝ)
  good_outcomes_count / outcomes_count

theorem acute_angle_probability :
  prob_acute_angle = 5 / 12 :=
by
  sorry

end acute_angle_probability_l788_78828


namespace polynomial_divisibility_l788_78883

theorem polynomial_divisibility (a b x y : ℤ) : 
  ∃ k : ℤ, (a * x + b * y)^3 + (b * x + a * y)^3 = k * (a + b) * (x + y) := by
  sorry

end polynomial_divisibility_l788_78883


namespace maximize_revenue_l788_78897

theorem maximize_revenue (p : ℝ) (h₁ : p ≤ 30) (h₂ : p = 18.75) : 
  ∃(R : ℝ), R = p * (150 - 4 * p) :=
by
  sorry

end maximize_revenue_l788_78897


namespace domain_log2_x_minus_1_l788_78805

theorem domain_log2_x_minus_1 (x : ℝ) : (1 < x) ↔ (∃ y : ℝ, y = Real.logb 2 (x - 1)) := by
  sorry

end domain_log2_x_minus_1_l788_78805


namespace line_perpendicular_slope_l788_78814

theorem line_perpendicular_slope (m : ℝ) :
  let slope1 := (1 / 2) 
  let slope2 := (-2 / m)
  slope1 * slope2 = -1 → m = 1 := 
by
  -- The proof will go here
  sorry

end line_perpendicular_slope_l788_78814


namespace intersection_A_B_l788_78869

def A : Set ℝ := { y | ∃ x : ℝ, y = |x| }
def B : Set ℝ := { y | ∃ x : ℝ, y = 1 - 2*x - x^2 }

theorem intersection_A_B :
  A ∩ B = { y | 0 ≤ y ∧ y ≤ 2 } :=
sorry

end intersection_A_B_l788_78869


namespace subset_eq_possible_sets_of_B_l788_78891

theorem subset_eq_possible_sets_of_B (B : Set ℕ) 
  (h1 : {1, 2} ⊆ B)
  (h2 : B ⊆ {1, 2, 3, 4}) :
  B = {1, 2} ∨ B = {1, 2, 3} ∨ B = {1, 2, 4} :=
sorry

end subset_eq_possible_sets_of_B_l788_78891


namespace pure_imaginary_solution_l788_78877

theorem pure_imaginary_solution (a : ℝ) (ha : a + 5 * Complex.I / (1 - 2 * Complex.I) = a + (1 : ℂ) * Complex.I) :
  a = 2 :=
by
  sorry

end pure_imaginary_solution_l788_78877


namespace sum_of_coordinates_of_A_l788_78838

open Real

theorem sum_of_coordinates_of_A (A B C : ℝ × ℝ) (h1 : B = (2, 8)) (h2 : C = (5, 2))
  (h3 : ∃ (k : ℝ), A = ((2 * (B.1:ℝ) + C.1) / 3, (2 * (B.2:ℝ) + C.2) / 3) ∧ k = 1/3) :
  A.1 + A.2 = 9 :=
sorry

end sum_of_coordinates_of_A_l788_78838


namespace contrapositive_negation_l788_78849

-- Define the main condition of the problem
def statement_p (x y : ℝ) : Prop :=
  (x - 1) * (y + 2) = 0 → (x = 1 ∨ y = -2)

-- Prove the contrapositive of statement_p
theorem contrapositive (x y : ℝ) : 
  (x ≠ 1 ∧ y ≠ -2) → ¬ ((x - 1) * (y + 2) = 0) :=
by 
  sorry

-- Prove the negation of statement_p
theorem negation (x y : ℝ) : 
  ((x - 1) * (y + 2) = 0) → ¬ (x = 1 ∨ y = -2) :=
by 
  sorry

end contrapositive_negation_l788_78849


namespace reckha_code_count_l788_78880

theorem reckha_code_count :
  let total_codes := 1000
  let codes_with_one_digit_different := 27
  let permutations_of_045 := 2
  let original_code := 1
  total_codes - codes_with_one_digit_different - permutations_of_045 - original_code = 970 :=
by
  let total_codes := 1000
  let codes_with_one_digit_different := 27
  let permutations_of_045 := 2
  let original_code := 1
  show total_codes - codes_with_one_digit_different - permutations_of_045 - original_code = 970
  sorry

end reckha_code_count_l788_78880


namespace solve_inequality_l788_78867

theorem solve_inequality (x : ℝ) : 3 * x^2 + 7 * x + 2 < 0 ↔ -1 < x ∧ x < -2/3 := by
  sorry

end solve_inequality_l788_78867


namespace charity_race_finished_racers_l788_78847

theorem charity_race_finished_racers :
  let initial_racers := 50
  let joined_after_20_minutes := 30
  let doubled_after_30_minutes := 2
  let dropped_racers := 30
  let total_racers_after_20_minutes := initial_racers + joined_after_20_minutes
  let total_racers_after_50_minutes := total_racers_after_20_minutes * doubled_after_30_minutes
  let finished_racers := total_racers_after_50_minutes - dropped_racers
  finished_racers = 130 := by
    sorry

end charity_race_finished_racers_l788_78847


namespace solve_CD_l788_78841

noncomputable def find_CD : Prop :=
  ∃ C D : ℝ, (C = 11 ∧ D = 0) ∧ (∀ x : ℝ, x ≠ -4 ∧ x ≠ 12 → 
    (7 * x - 3) / ((x + 4) * (x - 12)) = C / (x + 4) + D / (x - 12))

theorem solve_CD : find_CD :=
sorry

end solve_CD_l788_78841


namespace negate_proposition_l788_78837

theorem negate_proposition :
    (¬ ∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by
  sorry

end negate_proposition_l788_78837


namespace triangle_PQR_min_perimeter_l788_78832

theorem triangle_PQR_min_perimeter (PQ PR QR : ℕ) (QJ : ℕ) 
  (hPQ_PR : PQ = PR) (hQJ_10 : QJ = 10) (h_pos_QR : 0 < QR) :
  QR * 2 + PQ * 2 = 96 :=
  sorry

end triangle_PQR_min_perimeter_l788_78832


namespace percentage_error_in_area_l788_78884

theorem percentage_error_in_area (s : ℝ) (x : ℝ) (h₁ : s' = 1.08 * s) 
  (h₂ : s^2 = (2 * A)) (h₃ : x^2 = (2 * A)) : 
  (abs ((1.1664 * s^2 - s^2) / s^2 * 100) - 17) ≤ 0.5 := 
sorry

end percentage_error_in_area_l788_78884


namespace no_such_integers_exist_l788_78829

theorem no_such_integers_exist :
  ¬ ∃ (a b : ℕ), a ≥ 1 ∧ b ≥ 1 ∧ ∃ k₁ k₂ : ℕ, (a^5 * b + 3 = k₁^3) ∧ (a * b^5 + 3 = k₂^3) :=
by
  sorry

end no_such_integers_exist_l788_78829


namespace number_of_solutions_l788_78820

-- Given conditions
def positiveIntSolution (x y : ℤ) : Prop := x > 0 ∧ y > 0 ∧ 4 * x + 7 * y = 2001

-- Theorem statement
theorem number_of_solutions : ∃ (count : ℕ), 
  count = 71 ∧ ∃ f : Fin count → ℤ × ℤ,
    (∀ i, positiveIntSolution (f i).1 (f i).2) :=
by
  sorry

end number_of_solutions_l788_78820


namespace tangent_line_coordinates_l788_78876

theorem tangent_line_coordinates :
  ∃ x₀ : ℝ, ∃ y₀ : ℝ, (x₀ = 1 ∧ y₀ = Real.exp 1) ∧
  (∀ x : ℝ, ∀ y : ℝ, y = Real.exp x → ∃ m : ℝ, 
    (m = Real.exp 1 ∧ (y - y₀ = m * (x - x₀))) ∧
    (0 - y₀ = m * (0 - x₀))) := sorry

end tangent_line_coordinates_l788_78876


namespace points_calculation_correct_l788_78894

-- Definitions
def points_per_enemy : ℕ := 9
def total_enemies : ℕ := 11
def enemies_undestroyed : ℕ := 3
def enemies_destroyed : ℕ := total_enemies - enemies_undestroyed

def points_earned : ℕ := enemies_destroyed * points_per_enemy

-- Theorem statement
theorem points_calculation_correct : points_earned = 72 := by
  sorry

end points_calculation_correct_l788_78894


namespace find_m_b_l788_78808

theorem find_m_b (m b : ℚ) :
  (3 * m - 14 = 2) ∧ (m ^ 2 - 6 * m + 15 = b) →
  m = 16 / 3 ∧ b = 103 / 9 := by
  intro h
  rcases h with ⟨h1, h2⟩
  -- proof steps here
  sorry

end find_m_b_l788_78808


namespace card_distribution_count_l788_78809

theorem card_distribution_count : 
  ∃ (methods : ℕ), methods = 18 ∧ 
  ∃ (cards : Finset ℕ),
  ∃ (envelopes : Finset (Finset ℕ)), 
  cards = {1, 2, 3, 4, 5, 6} ∧ 
  envelopes.card = 3 ∧ 
  (∀ e ∈ envelopes, (e.card = 2) ∧ ({1, 2} ⊆ e → ∃ e1 e2, {e1, e2} ∈ envelopes ∧ {e1, e2} ⊆ cards \ {1, 2})) ∧ 
  (∀ c1 ∈ cards, ∃ e ∈ envelopes, c1 ∈ e) :=
by
  sorry

end card_distribution_count_l788_78809


namespace toll_for_18_wheel_truck_l788_78862

-- Define the number of axles given the conditions
def num_axles (total_wheels rear_axle_wheels front_axle_wheels : ℕ) : ℕ :=
  let rear_axles := (total_wheels - front_axle_wheels) / rear_axle_wheels
  rear_axles + 1

-- Define the toll calculation given the number of axles
def toll (axles : ℕ) : ℝ :=
  1.50 + 0.50 * (axles - 2)

-- Constants specific to the problem
def total_wheels : ℕ := 18
def rear_axle_wheels : ℕ := 4
def front_axle_wheels : ℕ := 2

-- Calculate the number of axles for the given truck
def truck_axles : ℕ := num_axles total_wheels rear_axle_wheels front_axle_wheels

-- The actual statement to prove
theorem toll_for_18_wheel_truck : toll truck_axles = 3.00 :=
  by
    -- proof will go here
    sorry

end toll_for_18_wheel_truck_l788_78862


namespace missing_coin_value_l788_78865

-- Definitions based on the conditions
def value_of_dime := 10 -- Value of 1 dime in cents
def value_of_nickel := 5 -- Value of 1 nickel in cents
def num_dimes := 1
def num_nickels := 2
def total_value_found := 45 -- Total value found in cents

-- Statement to prove the missing coin's value
theorem missing_coin_value : 
  (total_value_found - (num_dimes * value_of_dime + num_nickels * value_of_nickel)) = 25 := 
by
  sorry

end missing_coin_value_l788_78865
