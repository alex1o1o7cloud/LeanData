import Mathlib

namespace factor_diff_of_squares_l915_91572

-- Define the expression t^2 - 49 and show it is factored as (t - 7)(t + 7)
theorem factor_diff_of_squares (t : ℝ) : t^2 - 49 = (t - 7) * (t + 7) := by
  sorry

end factor_diff_of_squares_l915_91572


namespace find_f3_l915_91569

theorem find_f3 (a b : ℝ) (f : ℝ → ℝ)
  (h1 : f 1 = 3)
  (h2 : f 2 = 6)
  (h3 : ∀ x, f x = a * x^2 + b * x + 1) :
  f 3 = 10 :=
sorry

end find_f3_l915_91569


namespace tooth_extraction_cost_l915_91518

noncomputable def cleaning_cost : ℕ := 70
noncomputable def filling_cost : ℕ := 120
noncomputable def root_canal_cost : ℕ := 400
noncomputable def crown_cost : ℕ := 600
noncomputable def bridge_cost : ℕ := 800

noncomputable def crown_discount : ℕ := (crown_cost * 20) / 100
noncomputable def bridge_discount : ℕ := (bridge_cost * 10) / 100

noncomputable def total_cost_without_extraction : ℕ := 
  cleaning_cost + 
  3 * filling_cost + 
  root_canal_cost + 
  (crown_cost - crown_discount) + 
  (bridge_cost - bridge_discount)

noncomputable def root_canal_and_one_filling : ℕ := 
  root_canal_cost + filling_cost

noncomputable def dentist_bill : ℕ := 
  11 * root_canal_and_one_filling

theorem tooth_extraction_cost : 
  dentist_bill - total_cost_without_extraction = 3690 :=
by
  -- The proof would go here
  sorry

end tooth_extraction_cost_l915_91518


namespace pi_is_irrational_l915_91559

theorem pi_is_irrational :
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ π = p / q) :=
by
  sorry

end pi_is_irrational_l915_91559


namespace exists_geometric_weak_arithmetic_l915_91553

theorem exists_geometric_weak_arithmetic (m : ℕ) (hm : 3 ≤ m) :
  ∃ (k : ℕ) (a : ℕ → ℕ), 
    (∀ i, 1 ≤ i → i ≤ m → a i = k^(m - i)*(k + 1)^(i - 1)) ∧
    ((∀ i, 1 ≤ i → i < m → a i < a (i + 1)) ∧ 
    ∃ (x : ℕ → ℕ) (d : ℕ), 
      (x 0 ≤ a 1 ∧ 
      ∀ i, 1 ≤ i → i < m → (x i ≤ a (i + 1) ∧ a (i + 1) < x (i + 1)) ∧ 
      ∀ i, 0 ≤ i → i < m - 1 → x (i + 1) - x i = d)) :=
by
  sorry

end exists_geometric_weak_arithmetic_l915_91553


namespace find_A_l915_91502

def diamond (A B : ℝ) : ℝ := 5 * A + 3 * B + 7

theorem find_A (A : ℝ) (h : diamond A 5 = 82) : A = 12 :=
by
  unfold diamond at h
  sorry

end find_A_l915_91502


namespace pop_spending_original_l915_91511

-- Given conditions
def total_spent := 150
def crackle_spending (P : ℝ) := 3 * P
def snap_spending (P : ℝ) := 2 * crackle_spending P

-- Main statement to prove
theorem pop_spending_original : ∃ P : ℝ, snap_spending P + crackle_spending P + P = total_spent ∧ P = 15 :=
by
  sorry

end pop_spending_original_l915_91511


namespace discarded_second_number_l915_91554

-- Define the conditions
def avg_original_50 : ℝ := 38
def total_sum_50_numbers : ℝ := 50 * avg_original_50
def discarded_first : ℝ := 45
def avg_remaining_48 : ℝ := 37.5
def total_sum_remaining_48 : ℝ := 48 * avg_remaining_48
def sum_discarded := total_sum_50_numbers - total_sum_remaining_48

-- Define the proof statement
theorem discarded_second_number (x : ℝ) (h : discarded_first + x = sum_discarded) : x = 55 :=
by
  sorry

end discarded_second_number_l915_91554


namespace find_fraction_l915_91533

theorem find_fraction (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 20 * b) / (b + 20 * a) = 3) : a / b = 0.33 :=
sorry

end find_fraction_l915_91533


namespace profit_percentage_l915_91537

theorem profit_percentage (C S : ℝ) (h : 17 * C = 16 * S) : (S - C) / C * 100 = 6.25 := by
  sorry

end profit_percentage_l915_91537


namespace perfect_square_sum_l915_91523

-- Define the numbers based on the given conditions
def A (n : ℕ) : ℕ := 4 * (10^(2 * n) - 1) / 9
def B (n : ℕ) : ℕ := 2 * (10^(n + 1) - 1) / 9
def C (n : ℕ) : ℕ := 8 * (10^n - 1) / 9

-- Define the main theorem to be proved
theorem perfect_square_sum (n : ℕ) : 
  ∃ k, A n + B n + C n + 7 = k * k :=
sorry

end perfect_square_sum_l915_91523


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l915_91570

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l915_91570


namespace no_real_solution_l915_91575

theorem no_real_solution (x : ℝ) : x + 64 / (x + 3) ≠ -13 :=
by {
  -- Proof is not required, so we mark it as sorry.
  sorry
}

end no_real_solution_l915_91575


namespace fixed_point_of_line_l915_91515

theorem fixed_point_of_line (a : ℝ) (x y : ℝ)
  (h : ∀ a : ℝ, a * x + y + 1 = 0) :
  x = 0 ∧ y = -1 := 
by
  sorry

end fixed_point_of_line_l915_91515


namespace compare_neg_thirds_and_halves_l915_91593

theorem compare_neg_thirds_and_halves : (-1 : ℚ) / 3 > (-1 : ℚ) / 2 :=
by
  sorry

end compare_neg_thirds_and_halves_l915_91593


namespace julie_money_left_l915_91509

def cost_of_bike : ℕ := 2345
def initial_savings : ℕ := 1500

def mowing_rate : ℕ := 20
def mowing_jobs : ℕ := 20

def paper_rate : ℚ := 0.40
def paper_jobs : ℕ := 600

def dog_rate : ℕ := 15
def dog_jobs : ℕ := 24

def earnings_from_mowing : ℕ := mowing_rate * mowing_jobs
def earnings_from_papers : ℚ := paper_rate * paper_jobs
def earnings_from_dogs : ℕ := dog_rate * dog_jobs

def total_earnings : ℚ := earnings_from_mowing + earnings_from_papers + earnings_from_dogs
def total_money_available : ℚ := initial_savings + total_earnings

def money_left_after_purchase : ℚ := total_money_available - cost_of_bike

theorem julie_money_left : money_left_after_purchase = 155 := sorry

end julie_money_left_l915_91509


namespace friends_count_l915_91564

def bananas_total : ℝ := 63
def bananas_per_friend : ℝ := 21.0

theorem friends_count : bananas_total / bananas_per_friend = 3 := sorry

end friends_count_l915_91564


namespace shaded_area_fraction_l915_91529

/-- The fraction of the larger square's area that is inside the shaded rectangle 
    formed by the points (2,2), (3,2), (3,5), and (2,5) on a 6 by 6 grid 
    is 1/12. -/
theorem shaded_area_fraction : 
  let grid_size := 6
  let rectangle_points := [(2, 2), (3, 2), (3, 5), (2, 5)]
  let rectangle_length := 1
  let rectangle_height := 3
  let rectangle_area := rectangle_length * rectangle_height
  let square_area := grid_size^2
  rectangle_area / square_area = 1 / 12 := 
by 
  sorry

end shaded_area_fraction_l915_91529


namespace correct_choice_l915_91522

theorem correct_choice : 2 ∈ ({0, 1, 2} : Set ℕ) :=
sorry

end correct_choice_l915_91522


namespace minimum_perimeter_l915_91555

/-
Given:
1. (a: ℤ), (b: ℤ), (c: ℤ)
2. (a ≠ b)
3. 2 * a + 10 * c = 2 * b + 8 * c
4. 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2
5. 10 * c / 8 * c = 5 / 4

Prove:
The minimum perimeter is 1180 
-/

theorem minimum_perimeter (a b c : ℤ) 
(h1 : a ≠ b)
(h2 : 2 * a + 10 * c = 2 * b + 8 * c)
(h3 : 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2)
(h4 : 10 * c / 8 * c = 5 / 4) :
2 * a + 10 * c = 1180 ∨ 2 * b + 8 * c = 1180 :=
sorry

end minimum_perimeter_l915_91555


namespace hypotenuse_length_l915_91560

-- Definition of the right triangle with the given leg lengths
structure RightTriangle :=
  (BC AC AB : ℕ)
  (right : BC^2 + AC^2 = AB^2)

-- The theorem we need to prove
theorem hypotenuse_length (T : RightTriangle) (h1 : T.BC = 5) (h2 : T.AC = 12) :
  T.AB = 13 :=
by
  sorry

end hypotenuse_length_l915_91560


namespace compare_fractions_l915_91536

theorem compare_fractions {x : ℝ} (h : 3 < x ∧ x < 4) : 
  (2 / 3) > ((5 - x) / 3) :=
by sorry

end compare_fractions_l915_91536


namespace regular_polygon_sides_l915_91592

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l915_91592


namespace scientific_notation_of_819000_l915_91565

theorem scientific_notation_of_819000 : (819000 : ℝ) = 8.19 * 10^5 :=
by
  sorry

end scientific_notation_of_819000_l915_91565


namespace range_of_a_l915_91585

variable (a : ℝ)
variable (f : ℝ → ℝ)

-- Conditions
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def fWhenNegative (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x = 9 * x + a^2 / x + 7

def fNonNegativeCondition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x ≥ a + 1

-- Theorem to prove
theorem range_of_a (odd_f : isOddFunction f) (f_neg : fWhenNegative f a) 
  (nonneg_cond : fNonNegativeCondition f a) : 
  a ≤ -8 / 7 :=
by
  sorry

end range_of_a_l915_91585


namespace g_675_eq_42_l915_91587

-- Define the function g on positive integers
def g : ℕ → ℕ := sorry

-- State the conditions
axiom g_multiplicative : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g_15 : g 15 = 18
axiom g_45 : g 45 = 24

-- The theorem we want to prove
theorem g_675_eq_42 : g 675 = 42 := 
by 
  sorry

end g_675_eq_42_l915_91587


namespace circumscribed_sphere_eqn_l915_91543

-- Define vertices of the tetrahedron
variables {A_1 A_2 A_3 A_4 : Point}

-- Define barycentric coordinates
variables {x_1 x_2 x_3 x_4 : ℝ}

-- Define edge lengths
variables {a_12 a_13 a_14 a_23 a_24 a_34: ℝ}

-- Define the equation of the circumscribed sphere in barycentric coordinates
theorem circumscribed_sphere_eqn (h1 : A_1 ≠ A_2) (h2 : A_1 ≠ A_3) (h3 : A_1 ≠ A_4)
                                 (h4 : A_2 ≠ A_3) (h5 : A_2 ≠ A_4) (h6 : A_3 ≠ A_4) :
    (x_1 * x_2 * a_12^2 + x_1 * x_3 * a_13^2 + x_1 * x_4 * a_14^2 +
     x_2 * x_3 * a_23^2 + x_2 * x_4 * a_24^2 + x_3 * x_4 * a_34^2) = 0 :=
 sorry

end circumscribed_sphere_eqn_l915_91543


namespace fib_seventh_term_l915_91517

-- Defining the Fibonacci sequence
def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fib n + fib (n + 1)

-- Proving the value of the 7th term given 
-- fib(5) = 5 and fib(6) = 8
theorem fib_seventh_term : fib 7 = 13 :=
by {
    -- Conditions have been used in the definition of Fibonacci sequence
    sorry
}

end fib_seventh_term_l915_91517


namespace intersection_M_N_eq_M_inter_N_l915_91541

def M : Set ℝ := { x | x^2 - 4 > 0 }
def N : Set ℝ := { x | x < 0 }
def M_inter_N : Set ℝ := { x | x < -2 }

theorem intersection_M_N_eq_M_inter_N : M ∩ N = M_inter_N := 
by
  sorry

end intersection_M_N_eq_M_inter_N_l915_91541


namespace union_of_sets_l915_91544

def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {3, 4, 5}

theorem union_of_sets : M ∪ N = {2, 3, 4, 5} := by
  sorry

end union_of_sets_l915_91544


namespace total_gold_is_100_l915_91521

-- Definitions based on conditions
def GregsGold : ℕ := 20
def KatiesGold : ℕ := GregsGold * 4
def TotalGold : ℕ := GregsGold + KatiesGold

-- Theorem to prove
theorem total_gold_is_100 : TotalGold = 100 := by
  sorry

end total_gold_is_100_l915_91521


namespace fill_tanker_time_l915_91520

/-- Given that pipe A can fill the tanker in 60 minutes and pipe B can fill the tanker in 40 minutes,
    prove that the time T to fill the tanker if pipe B is used for half the time and both pipes 
    A and B are used together for the other half is equal to 30 minutes. -/
theorem fill_tanker_time (T : ℝ) (hA : ∀ (a : ℝ), a = 1/60) (hB : ∀ (b : ℝ), b = 1/40) :
  (T / 2) * (1 / 40) + (T / 2) * (1 / 24) = 1 → T = 30 :=
by
  sorry

end fill_tanker_time_l915_91520


namespace quadratic_minimum_value_l915_91563

theorem quadratic_minimum_value (p q : ℝ) (h_min_value : ∀ x : ℝ, 2 * x^2 + p * x + q ≥ 10) :
  q = 10 + p^2 / 8 :=
by
  sorry

end quadratic_minimum_value_l915_91563


namespace minimum_value_of_f_l915_91550

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3 * x + 9

-- State the theorem about the minimum value of the function
theorem minimum_value_of_f : ∃ x : ℝ, f x = 7 ∧ ∀ y : ℝ, f y ≥ 7 := sorry

end minimum_value_of_f_l915_91550


namespace fewer_hours_worked_l915_91549

noncomputable def total_earnings_summer := 6000
noncomputable def total_weeks_summer := 10
noncomputable def hours_per_week_summer := 50
noncomputable def total_earnings_school_year := 8000
noncomputable def total_weeks_school_year := 40

noncomputable def hourly_wage := total_earnings_summer / (hours_per_week_summer * total_weeks_summer)
noncomputable def total_hours_school_year := total_earnings_school_year / hourly_wage
noncomputable def hours_per_week_school_year := total_hours_school_year / total_weeks_school_year
noncomputable def fewer_hours_per_week := hours_per_week_summer - hours_per_week_school_year

theorem fewer_hours_worked :
  fewer_hours_per_week = hours_per_week_summer - (total_earnings_school_year / hourly_wage / total_weeks_school_year) := by
  sorry

end fewer_hours_worked_l915_91549


namespace mass_percentage_ba_in_bao_l915_91547

-- Define the constants needed in the problem
def molarMassBa : ℝ := 137.33
def molarMassO : ℝ := 16.00

-- Calculate the molar mass of BaO
def molarMassBaO : ℝ := molarMassBa + molarMassO

-- Express the problem as a Lean theorem for proof
theorem mass_percentage_ba_in_bao : 
  (molarMassBa / molarMassBaO) * 100 = 89.55 := by
  sorry

end mass_percentage_ba_in_bao_l915_91547


namespace gcd_b2_add_11b_add_28_b_add_6_eq_2_l915_91558

theorem gcd_b2_add_11b_add_28_b_add_6_eq_2 {b : ℤ} (h : ∃ k : ℤ, b = 1836 * k) : 
  Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
by
  sorry

end gcd_b2_add_11b_add_28_b_add_6_eq_2_l915_91558


namespace most_likely_number_of_red_balls_l915_91567

-- Define the conditions
def total_balls : ℕ := 20
def red_ball_frequency : ℝ := 0.8

-- Define the statement we want to prove
theorem most_likely_number_of_red_balls : red_ball_frequency * total_balls = 16 :=
by sorry

end most_likely_number_of_red_balls_l915_91567


namespace average_death_rate_l915_91506

-- Definitions and given conditions
def birth_rate_per_two_seconds := 6
def net_increase_per_day := 172800

-- Calculate number of seconds in a day as a constant
def seconds_per_day : ℕ := 24 * 60 * 60

-- Define the net increase per second
def net_increase_per_second : ℕ := net_increase_per_day / seconds_per_day

-- Define the birth rate per second
def birth_rate_per_second : ℕ := birth_rate_per_two_seconds / 2

-- The final proof statement
theorem average_death_rate : 
  ∃ (death_rate_per_two_seconds : ℕ), 
    death_rate_per_two_seconds = birth_rate_per_two_seconds - 2 * net_increase_per_second := 
by 
  -- We are required to prove this statement
  use (birth_rate_per_second - net_increase_per_second) * 2
  sorry

end average_death_rate_l915_91506


namespace min_value_of_sum_l915_91588

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (1 / (2 * a)) + (1 / b) = 1) :
  a + 2 * b = 9 / 2 :=
sorry

end min_value_of_sum_l915_91588


namespace negation_proof_converse_proof_l915_91571

-- Define the proposition
def prop_last_digit_zero_or_five (n : ℤ) : Prop := (n % 10 = 0) ∨ (n % 10 = 5)
def divisible_by_five (n : ℤ) : Prop := ∃ k : ℤ, n = 5 * k

-- Negation of the proposition
def negation_prop : Prop :=
  ∃ n : ℤ, prop_last_digit_zero_or_five n ∧ ¬ divisible_by_five n

-- Converse of the proposition
def converse_prop : Prop :=
  ∀ n : ℤ, ¬ prop_last_digit_zero_or_five n → ¬ divisible_by_five n

theorem negation_proof : negation_prop :=
  sorry  -- to be proved

theorem converse_proof : converse_prop :=
  sorry  -- to be proved

end negation_proof_converse_proof_l915_91571


namespace find_50th_term_arithmetic_sequence_l915_91557

theorem find_50th_term_arithmetic_sequence :
  let a₁ := 3
  let d := 7
  let a₅₀ := a₁ + (50 - 1) * d
  a₅₀ = 346 :=
by
  let a₁ := 3
  let d := 7
  let a₅₀ := a₁ + (50 - 1) * d
  show a₅₀ = 346
  sorry

end find_50th_term_arithmetic_sequence_l915_91557


namespace system_of_linear_eq_with_two_variables_l915_91538

-- Definitions of individual equations
def eqA (x : ℝ) : Prop := 3 * x - 2 = 5
def eqB (x : ℝ) : Prop := 6 * x^2 - 2 = 0
def eqC (x y : ℝ) : Prop := 1 / x + y = 3
def eqD (x y : ℝ) : Prop := 5 * x + y = 2

-- The main theorem to prove that D is a system of linear equations with two variables
theorem system_of_linear_eq_with_two_variables :
    (∃ x y : ℝ, eqD x y) ∧ (¬∃ x : ℝ, eqA x) ∧ (¬∃ x : ℝ, eqB x) ∧ (¬∃ x y : ℝ, eqC x y) :=
by
  sorry

end system_of_linear_eq_with_two_variables_l915_91538


namespace sector_angle_l915_91562

theorem sector_angle (r l : ℝ) (h₁ : 2 * r + l = 4) (h₂ : 1/2 * l * r = 1) : l / r = 2 :=
by
  sorry

end sector_angle_l915_91562


namespace simplify_and_evaluate_l915_91505

theorem simplify_and_evaluate (a : ℤ) (h : a = -2) :
  ( ((a + 7) / (a - 1) - 2 / (a + 1)) / ((a^2 + 3 * a) / (a^2 - 1)) = -1/2 ) :=
by
  sorry

end simplify_and_evaluate_l915_91505


namespace range_of_a_l915_91535

-- Define the inequality condition
def inequality (a x : ℝ) : Prop := (a-2)*x^2 + 2*(a-2)*x < 4

-- The main theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, inequality a x) ↔ (-2 : ℝ) < a ∧ a ≤ 2 := by
  sorry

end range_of_a_l915_91535


namespace johns_total_due_l915_91590

noncomputable def total_amount_due (initial_amount : ℝ) (first_charge_rate : ℝ) 
  (second_charge_rate : ℝ) (third_charge_rate : ℝ) : ℝ := 
  let after_first_charge := initial_amount * first_charge_rate
  let after_second_charge := after_first_charge * second_charge_rate
  let after_third_charge := after_second_charge * third_charge_rate
  after_third_charge

theorem johns_total_due : total_amount_due 500 1.02 1.03 1.025 = 538.43 := 
  by
    -- The proof would go here.
    sorry

end johns_total_due_l915_91590


namespace find_k_value_l915_91507

theorem find_k_value : ∀ (x y k : ℝ), x = 2 → y = -1 → y - k * x = 7 → k = -4 := 
by
  intros x y k hx hy h
  sorry

end find_k_value_l915_91507


namespace tan_double_angle_cos_beta_l915_91524

theorem tan_double_angle (α β : ℝ) (h1 : Real.sin α = 4 * Real.sqrt 3 / 7) 
  (h2 : Real.cos (β - α) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < Real.pi / 2) : 
  Real.tan (2 * α) = -(8 * Real.sqrt 3) / 47 :=
  sorry

theorem cos_beta (α β : ℝ) (h1 : Real.sin α = 4 * Real.sqrt 3 / 7) 
  (h2 : Real.cos (β - α) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < Real.pi / 2) : 
  Real.cos β = 1 / 2 :=
  sorry

end tan_double_angle_cos_beta_l915_91524


namespace new_person_weight_l915_91589

variable {W : ℝ} -- Total weight of the original group of 15 people
variable {N : ℝ} -- Weight of the new person

theorem new_person_weight
  (avg_increase : (W - 90 + N) / 15 = (W - 90) / 14 + 3.7)
  : N = 55.5 :=
sorry

end new_person_weight_l915_91589


namespace total_people_participated_l915_91591

theorem total_people_participated 
  (N f p : ℕ)
  (h1 : N = f * p)
  (h2 : N = (f - 10) * (p + 1))
  (h3 : N = (f - 25) * (p + 3)) : 
  N = 900 :=
by 
  sorry

end total_people_participated_l915_91591


namespace fraction_of_red_knights_magical_l915_91526

variable {knights : ℕ}
variable {red_knights : ℕ}
variable {blue_knights : ℕ}
variable {magical_knights : ℕ}
variable {magical_red_knights : ℕ}
variable {magical_blue_knights : ℕ}

axiom total_knights : knights > 0
axiom red_knights_fraction : red_knights = (3 * knights) / 8
axiom blue_knights_fraction : blue_knights = (5 * knights) / 8
axiom magical_knights_fraction : magical_knights = knights / 4
axiom magical_fraction_relation : 3 * magical_blue_knights = magical_red_knights

theorem fraction_of_red_knights_magical :
  (magical_red_knights : ℚ) / red_knights = 3 / 7 :=
by
  sorry

end fraction_of_red_knights_magical_l915_91526


namespace complement_of_M_in_U_is_correct_l915_91514

def U : Set ℤ := {1, -2, 3, -4, 5, -6}
def M : Set ℤ := {1, -2, 3, -4}
def complement_M_in_U : Set ℤ := {5, -6}

theorem complement_of_M_in_U_is_correct : (U \ M) = complement_M_in_U := by
  sorry

end complement_of_M_in_U_is_correct_l915_91514


namespace find_distance_between_foci_l915_91597

noncomputable def distance_between_foci (pts : List (ℝ × ℝ)) : ℝ :=
  let c := (1, -1)  -- center of the ellipse
  let x1 := (1, 3)
  let x2 := (1, -5)
  let y := (7, -5)
  let b := 4       -- semi-minor axis length
  let a := 2 * Real.sqrt 13  -- semi-major axis length
  let foci_distance := 2 * Real.sqrt (a^2 - b^2)
  foci_distance

theorem find_distance_between_foci :
  distance_between_foci [(1, 3), (7, -5), (1, -5)] = 12 :=
by
  sorry

end find_distance_between_foci_l915_91597


namespace sum_of_b_for_one_solution_l915_91530

theorem sum_of_b_for_one_solution :
  let A := 3
  let C := 12
  ∀ b : ℝ, ((b + 5)^2 - 4 * A * C = 0) → (b = 7 ∨ b = -17) → (7 + (-17)) = -10 :=
by
  intro A C b
  sorry

end sum_of_b_for_one_solution_l915_91530


namespace model_to_statue_ratio_l915_91566

theorem model_to_statue_ratio (h_statue : ℝ) (h_model : ℝ) (h_statue_eq : h_statue = 60) (h_model_eq : h_model = 4) :
  (h_statue / h_model) = 15 := by
  sorry

end model_to_statue_ratio_l915_91566


namespace rectangle_perimeter_l915_91504

theorem rectangle_perimeter {b : ℕ → ℕ} {W H : ℕ}
  (h1 : ∀ i, b i ≠ b (i+1))
  (h2 : b 9 = W / 2)
  (h3 : gcd W H = 1)

  (h4 : b 1 + b 2 = b 3)
  (h5 : b 1 + b 3 = b 4)
  (h6 : b 3 + b 4 = b 5)
  (h7 : b 4 + b 5 = b 6)
  (h8 : b 2 + b 3 + b 5 = b 7)
  (h9 : b 2 + b 7 = b 8)
  (h10 : b 1 + b 4 + b 6 = b 9)
  (h11 : b 6 + b 9 = b 7 + b 8) : 
  2 * (W + H) = 266 :=
  sorry

end rectangle_perimeter_l915_91504


namespace allison_craft_items_l915_91500

def glue_sticks (A B : Nat) : Prop := A = B + 8
def construction_paper (A B : Nat) : Prop := B = 6 * A

theorem allison_craft_items (Marie_glue_sticks Marie_paper_packs : Nat)
    (h1 : Marie_glue_sticks = 15)
    (h2 : Marie_paper_packs = 30) :
    ∃ (Allison_glue_sticks Allison_paper_packs total_items : Nat),
        glue_sticks Allison_glue_sticks Marie_glue_sticks ∧
        construction_paper Allison_paper_packs Marie_paper_packs ∧
        total_items = Allison_glue_sticks + Allison_paper_packs ∧
        total_items = 28 :=
by
    sorry

end allison_craft_items_l915_91500


namespace alcohol_percentage_in_original_solution_l915_91540

theorem alcohol_percentage_in_original_solution
  (P : ℚ)
  (alcohol_in_new_mixture : ℚ)
  (original_solution_volume : ℚ)
  (added_water_volume : ℚ)
  (new_mixture_volume : ℚ)
  (percentage_in_new_mixture : ℚ) :
  original_solution_volume = 11 →
  added_water_volume = 3 →
  new_mixture_volume = original_solution_volume + added_water_volume →
  percentage_in_new_mixture = 33 →
  alcohol_in_new_mixture = (percentage_in_new_mixture / 100) * new_mixture_volume →
  (P / 100) * original_solution_volume = alcohol_in_new_mixture →
  P = 42 :=
by
  sorry

end alcohol_percentage_in_original_solution_l915_91540


namespace largest_sphere_radius_on_torus_l915_91545

theorem largest_sphere_radius_on_torus :
  ∀ r : ℝ, 16 + (r - 1)^2 = (r + 2)^2 → r = 13 / 6 :=
by
  intro r
  intro h
  sorry

end largest_sphere_radius_on_torus_l915_91545


namespace circle_radius_l915_91574

theorem circle_radius (x y : ℝ) : x^2 - 10 * x + y^2 + 4 * y + 13 = 0 → (x - 5)^2 + (y + 2)^2 = 4^2 :=
by
  sorry

end circle_radius_l915_91574


namespace find_k_l915_91584

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^2 + k * x - 7

-- Define the given condition f(5) - g(5) = 20
def condition (k : ℝ) : Prop := f 5 - g 5 k = 20

-- The theorem to prove that k = 16.4
theorem find_k : ∃ k : ℝ, condition k ∧ k = 16.4 :=
by
  sorry

end find_k_l915_91584


namespace tg_half_angle_inequality_l915_91519

variable (α β γ : ℝ)

theorem tg_half_angle_inequality 
  (h : α + β + γ = 180) : 
  (Real.tan (α / 2)) * (Real.tan (β / 2)) * (Real.tan (γ / 2)) ≤ (Real.sqrt 3) / 9 := 
sorry

end tg_half_angle_inequality_l915_91519


namespace number_of_bookshelves_l915_91582

-- Definitions based on the conditions
def books_per_shelf : ℕ := 2
def total_books : ℕ := 38

-- Statement to prove
theorem number_of_bookshelves (books_per_shelf total_books : ℕ) : total_books / books_per_shelf = 19 :=
by sorry

end number_of_bookshelves_l915_91582


namespace multiple_of_1897_l915_91542

theorem multiple_of_1897 (n : ℕ) : ∃ k : ℤ, 2903^n - 803^n - 464^n + 261^n = k * 1897 := by
  sorry

end multiple_of_1897_l915_91542


namespace arithmetic_geometric_sequence_solution_l915_91594

theorem arithmetic_geometric_sequence_solution 
  (a1 a2 b1 b2 b3 : ℝ) 
  (h1 : -2 * 2 + a2 = a1)
  (h2 : a1 * 2 - 8 = a2)
  (h3 : b2 ^ 2 = -2 * -8)
  (h4 : b2 = -4) :
  (a2 - a1) / b2 = 1 / 2 :=
by 
  sorry

end arithmetic_geometric_sequence_solution_l915_91594


namespace projection_of_vec_c_onto_vec_b_l915_91532

def vec (x y : ℝ) : Prod ℝ ℝ := (x, y)

noncomputable def projection_of_c_onto_b := 
  let a := vec 2 3
  let b := vec (-4) 7
  let c := vec (-2) (-3)
  let dot_product_c_b := (-2) * (-4) + (-3) * 7
  let magnitude_b := Real.sqrt ((-4)^2 + 7^2)
  dot_product_c_b / magnitude_b
  
theorem projection_of_vec_c_onto_vec_b : 
  let a := vec 2 3
  let b := vec (-4) 7
  let c := vec (-2) (-3)
  let projection := projection_of_c_onto_b
  a + c = vec 0 0 ->
  projection = - Real.sqrt 65 / 5 := by
    sorry

end projection_of_vec_c_onto_vec_b_l915_91532


namespace ryan_days_learning_l915_91510

-- Definitions based on conditions
def hours_per_day_chinese : ℕ := 4
def total_hours_chinese : ℕ := 24

-- Theorem stating the number of days Ryan learns
theorem ryan_days_learning : total_hours_chinese / hours_per_day_chinese = 6 := 
by 
  -- Divide the total hours spent on Chinese learning by hours per day
  sorry

end ryan_days_learning_l915_91510


namespace expression_change_l915_91539

theorem expression_change (a b c : ℝ) : 
  a - (2 * b - 3 * c) = a + (-2 * b + 3 * c) := 
by sorry

end expression_change_l915_91539


namespace perpendicular_line_through_point_l915_91599

theorem perpendicular_line_through_point (x y : ℝ) : (x, y) = (0, -3) ∧ (∀ x y : ℝ, 2 * x + 3 * y - 6 = 0) → 3 * x - 2 * y - 6 = 0 :=
by
  sorry

end perpendicular_line_through_point_l915_91599


namespace first_offset_length_l915_91598

theorem first_offset_length (diagonal : ℝ) (offset2 : ℝ) (area : ℝ) (h_diagonal : diagonal = 50) (h_offset2 : offset2 = 8) (h_area : area = 450) :
  ∃ offset1 : ℝ, offset1 = 10 :=
by
  sorry

end first_offset_length_l915_91598


namespace total_days_spent_on_island_l915_91548

noncomputable def first_expedition_weeks := 3
noncomputable def second_expedition_weeks := first_expedition_weeks + 2
noncomputable def last_expedition_weeks := 2 * second_expedition_weeks
noncomputable def total_weeks := first_expedition_weeks + second_expedition_weeks + last_expedition_weeks
noncomputable def total_days := 7 * total_weeks

theorem total_days_spent_on_island : total_days = 126 := by
  sorry

end total_days_spent_on_island_l915_91548


namespace harriet_ran_48_miles_l915_91581

def total_distance : ℕ := 195
def katarina_distance : ℕ := 51
def equal_distance (n : ℕ) : Prop := (total_distance - katarina_distance) = 3 * n
def harriet_distance : ℕ := 48

theorem harriet_ran_48_miles
  (total_eq : total_distance = 195)
  (kat_eq : katarina_distance = 51)
  (equal_dist_eq : equal_distance harriet_distance) :
  harriet_distance = 48 :=
by
  sorry

end harriet_ran_48_miles_l915_91581


namespace find_M_l915_91531

theorem find_M (M : ℤ) (h1 : 22 < M) (h2 : M < 24) : M = 23 := by
  sorry

end find_M_l915_91531


namespace adjacent_abby_bridget_probability_l915_91516
open Nat

-- Define the conditions
def total_kids := 6
def grid_rows := 3
def grid_cols := 2
def middle_row := 2
def abby_and_bridget := 2

-- Define the probability calculation
theorem adjacent_abby_bridget_probability :
  let total_arrangements := 6!
  let num_ways_adjacent :=
    (2 * abby_and_bridget) * (total_kids - abby_and_bridget)!
  let total_outcomes := total_arrangements
  (num_ways_adjacent / total_outcomes : ℚ) = 4 / 15
:= sorry

end adjacent_abby_bridget_probability_l915_91516


namespace parabola_vertex_coordinates_l915_91556

theorem parabola_vertex_coordinates :
  ∃ (x y : ℝ), y = (x - 2)^2 ∧ (x, y) = (2, 0) :=
sorry

end parabola_vertex_coordinates_l915_91556


namespace stopped_babysitting_16_years_ago_l915_91586

-- Definitions of given conditions
def started_babysitting_age (Jane_age_start : ℕ) := Jane_age_start = 16
def age_half_constraint (Jane_age child_age : ℕ) := child_age ≤ Jane_age / 2
def current_age (Jane_age_now : ℕ) := Jane_age_now = 32
def oldest_babysat_age_now (child_age_now : ℕ) := child_age_now = 24

-- The proposition to be proved
theorem stopped_babysitting_16_years_ago 
  (Jane_age_start Jane_age_now child_age_now : ℕ)
  (h1 : started_babysitting_age Jane_age_start)
  (h2 : ∀ (Jane_age child_age : ℕ), age_half_constraint Jane_age child_age → Jane_age > Jane_age_start → child_age_now = 24 → Jane_age = 24)
  (h3 : current_age Jane_age_now)
  (h4 : oldest_babysat_age_now child_age_now) :
  Jane_age_now - Jane_age_start = 16 :=
by sorry

end stopped_babysitting_16_years_ago_l915_91586


namespace bad_carrots_l915_91573

-- Conditions
def carrots_picked_by_vanessa := 17
def carrots_picked_by_mom := 14
def good_carrots := 24
def total_carrots := carrots_picked_by_vanessa + carrots_picked_by_mom

-- Question and Proof
theorem bad_carrots :
  total_carrots - good_carrots = 7 :=
by
  -- Placeholder for proof
  sorry

end bad_carrots_l915_91573


namespace max_m_value_l915_91528

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 1 / b = 1 / 4) : ∃ m : ℝ, (∀ a b : ℝ,  a > 0 ∧ b > 0 ∧ (2 / a + 1 / b = 1 / 4) → 2 * a + b ≥ 4 * m) ∧ m = 7 / 4 :=
sorry

end max_m_value_l915_91528


namespace distance_between_parallel_lines_l915_91534

-- Definition of the first line l1
def line1 (x y : ℝ) (c1 : ℝ) : Prop := 3 * x + 4 * y + c1 = 0

-- Definition of the second line l2
def line2 (x y : ℝ) (c2 : ℝ) : Prop := 6 * x + 8 * y + c2 = 0

-- The problem statement in Lean:
theorem distance_between_parallel_lines (c1 c2 : ℝ) :
  ∃ d : ℝ, d = |2 * c1 - c2| / 10 :=
sorry

end distance_between_parallel_lines_l915_91534


namespace quadratic_function_range_l915_91503

noncomputable def quadratic_range : Set ℝ := {y | -2 ≤ y ∧ y < 2}

theorem quadratic_function_range :
  ∀ y : ℝ, 
    (∃ x : ℝ, -2 < x ∧ x < 1 ∧ y = x^2 + 2 * x - 1) ↔ (y ∈ quadratic_range) :=
by
  sorry

end quadratic_function_range_l915_91503


namespace ninth_grade_students_eq_l915_91552

-- Let's define the conditions
def total_students : ℕ := 50
def seventh_grade_students (x : ℕ) : ℕ := 2 * x - 1
def eighth_grade_students (x : ℕ) : ℕ := x

-- Define the expression for ninth grade students based on the conditions
def ninth_grade_students (x : ℕ) : ℕ :=
  total_students - (seventh_grade_students x + eighth_grade_students x)

-- The theorem statement to prove
theorem ninth_grade_students_eq (x : ℕ) : ninth_grade_students x = 51 - 3 * x :=
by
  sorry

end ninth_grade_students_eq_l915_91552


namespace find_k_l915_91501

-- Defining the vectors and the condition for parallelism
def vector_a := (2, 1)
def vector_b (k : ℝ) := (k, 3)

def vector_parallel_condition (k : ℝ) : Prop :=
  let a2b := (2 + 2 * k, 7)
  let a2nb := (4 - k, -1)
  (2 + 2 * k) * (-1) = 7 * (4 - k)

theorem find_k (k : ℝ) (h : vector_parallel_condition k) : k = 6 :=
by
  sorry

end find_k_l915_91501


namespace largest_in_eight_consecutive_integers_l915_91577

theorem largest_in_eight_consecutive_integers (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) = 4304) :
  n + 7 = 544 :=
by
  sorry

end largest_in_eight_consecutive_integers_l915_91577


namespace goods_train_speed_l915_91525

theorem goods_train_speed (Vm : ℝ) (T : ℝ) (L : ℝ) (Vg : ℝ) :
  Vm = 50 → T = 9 → L = 280 →
  Vg = ((L / T) - (Vm * 1000 / 3600)) * 3600 / 1000 →
  Vg = 62 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end goods_train_speed_l915_91525


namespace interval_of_a_l915_91512

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

-- Define the sequence a_n
noncomputable def a_n (a : ℝ) (n : ℕ) : ℝ :=
  f a n.succ  -- since ℕ in Lean includes 0, use n.succ to start from 1

-- The main theorem to prove
theorem interval_of_a (a : ℝ) : (∀ n : ℕ, n ≠ 0 → a_n a n < a_n a (n + 1)) → 2 < a ∧ a < 3 :=
by
  sorry

end interval_of_a_l915_91512


namespace columbia_distinct_arrangements_l915_91561

theorem columbia_distinct_arrangements : 
  let total_letters := 8
  let repeat_I := 2
  let repeat_U := 2
  Nat.factorial total_letters / (Nat.factorial repeat_I * Nat.factorial repeat_U) = 90720 := by
  sorry

end columbia_distinct_arrangements_l915_91561


namespace log_400_cannot_be_computed_l915_91546

theorem log_400_cannot_be_computed :
  let log_8 : ℝ := 0.9031
  let log_9 : ℝ := 0.9542
  let log_7 : ℝ := 0.8451
  (∀ (log_2 log_3 log_5 : ℝ), log_2 = 1 / 3 * log_8 → log_3 = 1 / 2 * log_9 → log_5 = 1 → 
    (∀ (log_val : ℝ), 
      (log_val = log_21 → log_21 = log_3 + log_7 → log_val = (1 / 2) * log_9 + log_7)
      ∧ (log_val = log_9_over_8 → log_9_over_8 = log_9 - log_8)
      ∧ (log_val = log_126 → log_126 = log_2 + log_7 + log_9 → log_val = (1 / 3) * log_8 + log_7 + log_9)
      ∧ (log_val = log_0_875 → log_0_875 = log_7 - log_8)
      ∧ (log_val = log_400 → log_400 = log_8 + 1 + log_5) 
      → False))
:= 
sorry

end log_400_cannot_be_computed_l915_91546


namespace values_of_x_l915_91568

theorem values_of_x (x : ℕ) (h : Nat.choose 18 x = Nat.choose 18 (3 * x - 6)) : x = 3 ∨ x = 6 :=
by
  sorry

end values_of_x_l915_91568


namespace code_length_is_4_l915_91513

-- Definitions based on conditions provided
def code_length : ℕ := 4 -- Each code consists of 4 digits
def total_codes_with_leading_zeros : ℕ := 10^code_length -- Total possible codes allowing leading zeros
def total_codes_without_leading_zeros : ℕ := 9 * 10^(code_length - 1) -- Total possible codes disallowing leading zeros
def codes_lost_if_no_leading_zeros : ℕ := total_codes_with_leading_zeros - total_codes_without_leading_zeros -- Codes lost if leading zeros are disallowed
def manager_measured_codes_lost : ℕ := 10000 -- Manager's incorrect measurement

-- Theorem to be proved based on the problem
theorem code_length_is_4 : code_length = 4 :=
by
  sorry

end code_length_is_4_l915_91513


namespace sin_C_of_arithmetic_sequence_l915_91508

theorem sin_C_of_arithmetic_sequence 
  (A B C : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = Real.pi) 
  (h3 : Real.cos A = 2 / 3) 
  : Real.sin C = (Real.sqrt 5 + 2 * Real.sqrt 3) / 6 :=
sorry

end sin_C_of_arithmetic_sequence_l915_91508


namespace range_of_x_l915_91583

theorem range_of_x (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -2) :
  x > 1/3 ∨ x < -1/2 :=
sorry

end range_of_x_l915_91583


namespace arithmetic_sequence_l915_91578

theorem arithmetic_sequence {a b : ℤ} :
  (-1 < a ∧ a < b ∧ b < 8) ∧
  (8 - (-1) = 9) ∧
  (a + b = 7) →
  (a = 2 ∧ b = 5) :=
by
  sorry

end arithmetic_sequence_l915_91578


namespace explicit_formula_l915_91580

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem explicit_formula (x1 x2 : ℝ) (h1 : x1 ∈ Set.Icc (-1 : ℝ) 1) (h2 : x2 ∈ Set.Icc (-1 : ℝ) 1) :
  f x = x^3 - 3 * x ∧ |f x1 - f x2| ≤ 4 :=
by
  sorry

end explicit_formula_l915_91580


namespace inequality_proof_l915_91576

theorem inequality_proof (x y z : ℝ) : 
    x^4 + y^4 + z^2 + 1 ≥ 2 * x * (x * y^2 - x + z + 1) :=
by
  sorry

end inequality_proof_l915_91576


namespace interval_of_x_l915_91551

theorem interval_of_x (x : ℝ) :
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) := by
  sorry

end interval_of_x_l915_91551


namespace mode_of_dataset_l915_91595

def dataset : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def frequency (n : ℕ) (l : List ℕ) : ℕ :=
  l.count n

theorem mode_of_dataset :
  (∀ n ≠ 3, frequency n dataset ≤ 3) ∧ frequency 3 dataset = 3 :=
by
  sorry

end mode_of_dataset_l915_91595


namespace range_of_m_l915_91596

theorem range_of_m (m : ℝ) : (∀ x : ℝ, ∀ y : ℝ, (2 ≤ x ∧ x ≤ 3) → (3 ≤ y ∧ y ≤ 6) → m * x^2 - x * y + y^2 ≥ 0) ↔ (m ≥ 0) :=
by
  sorry

end range_of_m_l915_91596


namespace vertex_angle_is_130_8_l915_91579

-- Define the given conditions
variables {a b h : ℝ}

def is_isosceles_triangle (a b h : ℝ) : Prop :=
  a^2 = b * 3 * h ∧ b = 2 * h

-- Define the obtuse condition on the vertex angle
def vertex_angle_obtuse (a b h : ℝ) : Prop :=
  ∃ θ : ℝ, 120 < θ ∧ θ < 180 ∧ θ = (130.8 : ℝ)

-- The formal proof statement using Lean 4
theorem vertex_angle_is_130_8 (a b h : ℝ) 
  (h1 : is_isosceles_triangle a b h)
  (h2 : vertex_angle_obtuse a b h) : 
  ∃ (φ : ℝ), φ = 130.8 :=
sorry

end vertex_angle_is_130_8_l915_91579


namespace average_minutes_per_day_l915_91527

theorem average_minutes_per_day
  (f : ℕ) -- Number of fifth graders
  (third_grade_minutes : ℕ := 10)
  (fourth_grade_minutes : ℕ := 18)
  (fifth_grade_minutes : ℕ := 12)
  (third_grade_students : ℕ := 3 * f)
  (fourth_grade_students : ℕ := (3 / 2) * f) -- Assumed to work with integer or rational numbers
  (fifth_grade_students : ℕ := f)
  (total_minutes_third_grade : ℕ := third_grade_minutes * third_grade_students)
  (total_minutes_fourth_grade : ℕ := fourth_grade_minutes * fourth_grade_students)
  (total_minutes_fifth_grade : ℕ := fifth_grade_minutes * fifth_grade_students)
  (total_minutes : ℕ := total_minutes_third_grade + total_minutes_fourth_grade + total_minutes_fifth_grade)
  (total_students : ℕ := third_grade_students + fourth_grade_students + fifth_grade_students) :
  (total_minutes / total_students : ℝ) = 12.55 :=
by
  sorry

end average_minutes_per_day_l915_91527
