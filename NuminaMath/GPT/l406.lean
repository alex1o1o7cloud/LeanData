import Mathlib

namespace total_wheels_in_garage_l406_40630

theorem total_wheels_in_garage 
    (num_bicycles : ℕ)
    (num_cars : ℕ)
    (wheels_per_bicycle : ℕ)
    (wheels_per_car : ℕ) 
    (num_bicycles_eq : num_bicycles = 9)
    (num_cars_eq : num_cars = 16)
    (wheels_per_bicycle_eq : wheels_per_bicycle = 2)
    (wheels_per_car_eq : wheels_per_car = 4) :
    num_bicycles * wheels_per_bicycle + num_cars * wheels_per_car = 82 := 
by
    sorry

end total_wheels_in_garage_l406_40630


namespace largest_n_l406_40682

def a_n (n : ℕ) (d_a : ℤ) : ℤ := 1 + (n-1) * d_a
def b_n (n : ℕ) (d_b : ℤ) : ℤ := 3 + (n-1) * d_b

theorem largest_n (d_a d_b : ℤ) (n : ℕ) :
  (a_n n d_a * b_n n d_b = 2304 ∧ a_n 1 d_a = 1 ∧ b_n 1 d_b = 3) 
  → n ≤ 20 := 
sorry

end largest_n_l406_40682


namespace find_x_plus_y_l406_40618

theorem find_x_plus_y (x y : ℝ) 
  (h1 : |x| + x + y = 14) 
  (h2 : x + |y| - y = 16) : 
  x + y = -2 := 
by
  sorry

end find_x_plus_y_l406_40618


namespace heartsuit_example_l406_40647

def heartsuit (x y: ℤ) : ℤ := 4 * x + 6 * y

theorem heartsuit_example : heartsuit 3 8 = 60 :=
by
  sorry

end heartsuit_example_l406_40647


namespace analytical_expression_when_x_in_5_7_l406_40651

noncomputable def f : ℝ → ℝ := sorry

lemma odd_function (x : ℝ) : f (-x) = -f x := sorry
lemma symmetric_about_one (x : ℝ) : f (1 - x) = f (1 + x) := sorry
lemma values_between_zero_and_one (x : ℝ) (h : 0 < x ∧ x ≤ 1) : f x = x := sorry

theorem analytical_expression_when_x_in_5_7 (x : ℝ) (h : 5 < x ∧ x ≤ 7) :
  f x = 6 - x :=
sorry

end analytical_expression_when_x_in_5_7_l406_40651


namespace fg_3_eq_7_l406_40646

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x - 2) ^ 2

theorem fg_3_eq_7 : f (g 3) = 7 :=
by
  sorry

end fg_3_eq_7_l406_40646


namespace solve_diophantine_equation_l406_40602

def is_solution (m n : ℕ) : Prop := 2^m - 3^n = 1

theorem solve_diophantine_equation : 
  { (m, n) : ℕ × ℕ | is_solution m n } = { (1, 0), (2, 1) } :=
by
  sorry

end solve_diophantine_equation_l406_40602


namespace exist_column_remove_keeps_rows_distinct_l406_40687

theorem exist_column_remove_keeps_rows_distinct 
    (n : ℕ) 
    (table : Fin n → Fin n → Char) 
    (h_diff_rows : ∀ i j : Fin n, i ≠ j → ∃ k : Fin n, table i k ≠ table j k) 
    : ∃ col_to_remove : Fin n, ∀ i j : Fin n, i ≠ j → (table i ≠ table j) :=
sorry

end exist_column_remove_keeps_rows_distinct_l406_40687


namespace ted_and_mike_seeds_l406_40698

noncomputable def ted_morning_seeds (T : ℕ) (mike_morning_seeds : ℕ) (mike_afternoon_seeds : ℕ) (total_seeds : ℕ) : Prop :=
  mike_morning_seeds = 50 ∧
  mike_afternoon_seeds = 60 ∧
  total_seeds = 250 ∧
  T + (mike_afternoon_seeds - 20) + (mike_morning_seeds + mike_afternoon_seeds) = total_seeds ∧
  2 * mike_morning_seeds = T

theorem ted_and_mike_seeds :
  ∃ T : ℕ, ted_morning_seeds T 50 60 250 :=
by {
  sorry
}

end ted_and_mike_seeds_l406_40698


namespace solution_correct_l406_40690

noncomputable def solution_set : Set ℝ :=
  {x | (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x)}

theorem solution_correct (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2))) < 1 / 4 ↔ (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x) :=
by sorry

end solution_correct_l406_40690


namespace find_constants_l406_40621

theorem find_constants :
  ∃ P Q : ℚ, (∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    (4 * x + 7) / (x^2 - 3 * x - 18) = P / (x - 6) + Q / (x + 3)) ∧
    P = 31 / 9 ∧ Q = 5 / 9 :=
by
  sorry

end find_constants_l406_40621


namespace max_value_of_sum_l406_40674

theorem max_value_of_sum (x y z : ℝ) (h : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) 
  (eq : x^2 + y^2 + z^2 + x + 2*y + 3*z = (13 : ℝ) / 4) : x + y + z ≤ 3 / 2 :=
sorry

end max_value_of_sum_l406_40674


namespace problem_l406_40654

theorem problem (x y : ℝ)
  (h1 : 1 / x + 1 / y = 4)
  (h2 : 1 / x - 1 / y = -6) :
  x + y = -4 / 5 :=
sorry

end problem_l406_40654


namespace find_a_l406_40663

theorem find_a (a b c : ℤ) (h_vertex : ∀ x, (x - 2)*(x - 2) * a + 3 = a*x*x + b*x + c) 
  (h_point : (a*(3 - 2)*(3 -2) + 3 = 6)) : a = 3 :=
by
  sorry

end find_a_l406_40663


namespace disloyal_bound_l406_40656

variable {p n : ℕ}

/-- A number is disloyal if its GCD with n is not 1 -/
def isDisloyal (x : ℕ) (n : ℕ) := Nat.gcd x n ≠ 1

theorem disloyal_bound (p : ℕ) (n : ℕ) (hp : p.Prime) (hn : n % p^2 = 0) :
  (∃ D : Finset ℕ, (∀ x ∈ D, isDisloyal x n) ∧ D.card ≤ (n - 1) / p) :=
sorry

end disloyal_bound_l406_40656


namespace elisa_target_amount_l406_40696

def elisa_current_amount : ℕ := 37
def elisa_additional_amount : ℕ := 16

theorem elisa_target_amount : elisa_current_amount + elisa_additional_amount = 53 :=
by
  sorry

end elisa_target_amount_l406_40696


namespace inhabitable_fraction_l406_40688

theorem inhabitable_fraction 
  (total_land_fraction : ℚ)
  (inhabitable_land_fraction : ℚ)
  (h1 : total_land_fraction = 1 / 3)
  (h2 : inhabitable_land_fraction = 3 / 4):
  total_land_fraction * inhabitable_land_fraction = 1 / 4 := 
by
  sorry

end inhabitable_fraction_l406_40688


namespace geometric_sequence_sum_a_l406_40684

theorem geometric_sequence_sum_a (a : ℤ) (S : ℕ → ℤ) (a_n : ℕ → ℤ) 
  (h1 : ∀ n : ℕ, S n = 2^n + a)
  (h2 : ∀ n : ℕ, a_n n = if n = 1 then S 1 else S n - S (n - 1)) :
  a = -1 :=
by
  sorry

end geometric_sequence_sum_a_l406_40684


namespace winner_percentage_l406_40608

theorem winner_percentage (V_winner V_margin V_total : ℕ) (h_winner: V_winner = 806) (h_margin: V_margin = 312) (h_total: V_total = V_winner + (V_winner - V_margin)) :
  ((V_winner: ℚ) / V_total) * 100 = 62 := by
  sorry

end winner_percentage_l406_40608


namespace coordinates_of_P_l406_40607

def P : Prod Int Int := (-1, 2)

theorem coordinates_of_P :
  P = (-1, 2) := 
  by
    -- The proof is omitted as per instructions
    sorry

end coordinates_of_P_l406_40607


namespace smallest_k_for_Δk_un_zero_l406_40628

def u (n : ℕ) : ℤ := n^3 - n

def Δ (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0     => u
  | (k+1) => λ n => Δ k u (n+1) - Δ k u n

theorem smallest_k_for_Δk_un_zero (u : ℕ → ℤ) (h : ∀ n, u n = n^3 - n) :
  ∀ n, Δ 4 u n = 0 ∧ (∀ k < 4, ∃ n, Δ k u n ≠ 0) :=
by
  sorry

end smallest_k_for_Δk_un_zero_l406_40628


namespace smallest_number_plus_3_divisible_by_18_70_100_21_l406_40666

/-- 
The smallest number such that when increased by 3 is divisible by 18, 70, 100, and 21.
-/
theorem smallest_number_plus_3_divisible_by_18_70_100_21 : 
  ∃ n : ℕ, (∃ k : ℕ, n + 3 = k * 18) ∧ (∃ l : ℕ, n + 3 = l * 70) ∧ (∃ m : ℕ, n + 3 = m * 100) ∧ (∃ o : ℕ, n + 3 = o * 21) ∧ n = 6297 :=
sorry

end smallest_number_plus_3_divisible_by_18_70_100_21_l406_40666


namespace angle_A_value_cos_A_minus_2x_value_l406_40680

open Real

-- Let A, B, and C be the internal angles of triangle ABC.
variable {A B C x : ℝ}

-- Given conditions
axiom triangle_angles : A + B + C = π
axiom sinC_eq_2sinAminusB : sin C = 2 * sin (A - B)
axiom B_is_pi_over_6 : B = π / 6
axiom cosAplusx_is_neg_third : cos (A + x) = -1 / 3

-- Proof goals
theorem angle_A_value : A = π / 3 := by sorry

theorem cos_A_minus_2x_value : cos (A - 2 * x) = 7 / 9 := by sorry

end angle_A_value_cos_A_minus_2x_value_l406_40680


namespace goods_train_length_l406_40600

noncomputable def speed_kmh : ℕ := 72  -- Speed of the goods train in km/hr
noncomputable def platform_length : ℕ := 280  -- Length of the platform in meters
noncomputable def time_seconds : ℕ := 26  -- Time taken to cross the platform in seconds
noncomputable def speed_mps : ℤ := speed_kmh * 1000 / 3600 -- Speed of the goods train in meters/second

theorem goods_train_length : 20 * time_seconds = 280 + 240 :=
by
  sorry

end goods_train_length_l406_40600


namespace inequality_proof_l406_40678

theorem inequality_proof (b c : ℝ) (hb : 0 < b) (hc : 0 < c) :
  (b - c) ^ 2011 * (b + c) ^ 2011 * (c - b) ^ 2011 ≥ 
  (b ^ 2011 - c ^ 2011) * (b ^ 2011 + c ^ 2011) * (c ^ 2011 - b ^ 2011) := 
by
  sorry

end inequality_proof_l406_40678


namespace water_tank_height_l406_40649

theorem water_tank_height (r h : ℝ) (V : ℝ) (V_water : ℝ) (a b : ℕ) 
  (h_tank : h = 120) (r_tank : r = 20) (V_tank : V = (1/3) * π * r^2 * h) 
  (V_water_capacity : V_water = 0.4 * V) :
  a = 48 ∧ b = 2 ∧ V = 16000 * π ∧ V_water = 6400 * π ∧ 
  h_water = 48 * (2^(1/3) / 1) ∧ (a + b = 50) :=
by
  sorry

end water_tank_height_l406_40649


namespace find_smallest_d_l406_40660

theorem find_smallest_d (d : ℕ) : (5 + 6 + 2 + 4 + 8 + d) % 9 = 0 → d = 2 :=
by
  sorry

end find_smallest_d_l406_40660


namespace length_of_rectangular_garden_l406_40641

theorem length_of_rectangular_garden (P B : ℝ) (h₁ : P = 1200) (h₂ : B = 240) :
  ∃ L : ℝ, P = 2 * (L + B) ∧ L = 360 :=
by
  sorry

end length_of_rectangular_garden_l406_40641


namespace correct_equation_l406_40694

namespace MathProblem

def is_two_digit_positive_integer (P : ℤ) : Prop :=
  10 ≤ P ∧ P < 100

def equation_A : Prop :=
  ∀ x : ℤ, x^2 + (-98)*x + 2001 = (x - 29) * (x - 69)

def equation_B : Prop :=
  ∀ x : ℤ, x^2 + (-110)*x + 2001 = (x - 23) * (x - 87)

def equation_C : Prop :=
  ∀ x : ℤ, x^2 + 110*x + 2001 = (x + 23) * (x + 87)

def equation_D : Prop :=
  ∀ x : ℤ, x^2 + 98*x + 2001 = (x + 29) * (x + 69)

theorem correct_equation :
  is_two_digit_positive_integer 98 ∧ equation_D :=
  sorry

end MathProblem

end correct_equation_l406_40694


namespace find_original_number_l406_40611

theorem find_original_number (x : ℚ) (h : 1 + 1 / x = 8 / 3) : x = 3 / 5 := by
  sorry

end find_original_number_l406_40611


namespace volume_pyramid_ABC_l406_40699

structure Point where
  x : ℝ
  y : ℝ

def triangle_volume (A B C : Point) : ℝ :=
  -- The implementation would calculate the volume of the pyramid formed
  -- by folding along the midpoint sides.
  sorry

theorem volume_pyramid_ABC :
  let A := Point.mk 0 0
  let B := Point.mk 30 0
  let C := Point.mk 20 15
  triangle_volume A B C = 900 :=
by
  -- To be filled with the proof
  sorry

end volume_pyramid_ABC_l406_40699


namespace number_of_divisors_of_8_fact_l406_40681

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l406_40681


namespace angle_sum_around_point_l406_40650

theorem angle_sum_around_point (p q r s t : ℝ) (h : p + q + r + s + t = 360) : p = 360 - q - r - s - t :=
by
  sorry

end angle_sum_around_point_l406_40650


namespace find_c1_minus_c2_l406_40675

-- Define the conditions of the problem
variables (c1 c2 : ℝ)
variables (x y : ℝ)
variables (h1 : (2 : ℝ) * x + 3 * y = c1)
variables (h2 : (3 : ℝ) * x + 2 * y = c2)
variables (sol_x : x = 2)
variables (sol_y : y = 1)

-- Define the theorem to be proven
theorem find_c1_minus_c2 : c1 - c2 = -1 := 
by
  sorry

end find_c1_minus_c2_l406_40675


namespace range_of_expression_l406_40614

theorem range_of_expression (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 ≤ β ∧ β ≤ π / 2) :
    -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π :=
by
  sorry

end range_of_expression_l406_40614


namespace find_x_l406_40679

theorem find_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 5 = 5 * y + 2) :
  x = (685 + 25 * Real.sqrt 745) / 6 :=
by
  sorry

end find_x_l406_40679


namespace domain_of_function_l406_40668

theorem domain_of_function :
  ∀ x : ℝ, (0 ≤ x ∧ x ≠ 1) ↔ (∃ y : ℝ, y = 1 / (Real.sqrt x - 1)) := by
  sorry

end domain_of_function_l406_40668


namespace least_possible_value_a2008_l406_40669

theorem least_possible_value_a2008 
  (a : ℕ → ℤ) 
  (h1 : ∀ n, a n < a (n + 1)) 
  (h2 : ∀ i j k l, 1 ≤ i → i < j → j ≤ k → k < l → i + l = j + k → a i + a l > a j + a k)
  : a 2008 ≥ 2015029 :=
sorry

end least_possible_value_a2008_l406_40669


namespace initial_money_l406_40643

-- Let M represent the initial amount of money Mrs. Hilt had.
variable (M : ℕ)

-- Condition 1: Mrs. Hilt bought a pencil for 11 cents.
def pencil_cost : ℕ := 11

-- Condition 2: She had 4 cents left after buying the pencil.
def amount_left : ℕ := 4

-- Proof problem statement: Prove that M = 15 given the above conditions.
theorem initial_money (h : M = pencil_cost + amount_left) : M = 15 :=
by
  sorry

end initial_money_l406_40643


namespace impossible_to_achieve_desired_piles_l406_40609

def initial_piles : List ℕ := [51, 49, 5]

def desired_piles : List ℕ := [52, 48, 5]

def combine_piles (x y : ℕ) : ℕ := x + y

def divide_pile (x : ℕ) (h : x % 2 = 0) : List ℕ := [x / 2, x / 2]

theorem impossible_to_achieve_desired_piles :
  ∀ (piles : List ℕ), 
    (piles = initial_piles) →
    (∀ (p : List ℕ), 
      (p = desired_piles) → 
      False) :=
sorry

end impossible_to_achieve_desired_piles_l406_40609


namespace expression_subtracted_from_3_pow_k_l406_40644

theorem expression_subtracted_from_3_pow_k (k : ℕ) (h : 15^k ∣ 759325) : 3^k - 0 = 1 :=
sorry

end expression_subtracted_from_3_pow_k_l406_40644


namespace work_completion_time_l406_40677

noncomputable def rate_b : ℝ := 1 / 24
noncomputable def rate_a : ℝ := 2 * rate_b
noncomputable def combined_rate : ℝ := rate_a + rate_b
noncomputable def completion_time : ℝ := 1 / combined_rate

theorem work_completion_time :
  completion_time = 8 :=
by
  sorry

end work_completion_time_l406_40677


namespace element_with_36_36_percentage_is_O_l406_40619

-- Define the chemical formula N2O and atomic masses
def chemical_formula : String := "N2O"
def atomic_mass_N : Float := 14.01
def atomic_mass_O : Float := 16.00

-- Define the molar mass of N2O
def molar_mass_N2O : Float := (2 * atomic_mass_N) + (1 * atomic_mass_O)

-- Mass of nitrogen in N2O
def mass_N_in_N2O : Float := 2 * atomic_mass_N

-- Mass of oxygen in N2O
def mass_O_in_N2O : Float := 1 * atomic_mass_O

-- Mass percentages
def mass_percentage_N : Float := (mass_N_in_N2O / molar_mass_N2O) * 100
def mass_percentage_O : Float := (mass_O_in_N2O / molar_mass_N2O) * 100

-- Prove that the element with a mass percentage of 36.36% is oxygen
theorem element_with_36_36_percentage_is_O : mass_percentage_O = 36.36 := sorry

end element_with_36_36_percentage_is_O_l406_40619


namespace ratio_A_B_l406_40692

variable (A B C : ℕ)

theorem ratio_A_B 
  (h1: A + B + C = 98) 
  (h2: B = 30) 
  (h3: (B : ℚ) / C = 5 / 8) 
  : (A : ℚ) / B = 2 / 3 :=
sorry

end ratio_A_B_l406_40692


namespace value_of_B_minus_3_plus_A_l406_40613

theorem value_of_B_minus_3_plus_A (A B : ℝ) (h : A + B = 5) : B - 3 + A = 2 :=
by 
  sorry

end value_of_B_minus_3_plus_A_l406_40613


namespace integer_solutions_m3_eq_n3_plus_n_l406_40683

theorem integer_solutions_m3_eq_n3_plus_n (m n : ℤ) (h : m^3 = n^3 + n) : m = 0 ∧ n = 0 :=
sorry

end integer_solutions_m3_eq_n3_plus_n_l406_40683


namespace evaluate_expression_l406_40610

theorem evaluate_expression : -20 + 8 * (10 / 2) - 4 = 16 :=
by
  sorry -- Proof to be completed

end evaluate_expression_l406_40610


namespace pages_of_shorter_book_is_10_l406_40672

theorem pages_of_shorter_book_is_10
  (x : ℕ) 
  (h_diff : ∀ (y : ℕ), x = y - 10)
  (h_divide : (x + 10) / 2 = x) 
  : x = 10 :=
by
  sorry

end pages_of_shorter_book_is_10_l406_40672


namespace a_plus_b_eq_six_l406_40624

theorem a_plus_b_eq_six (a b : ℤ) (k : ℝ) (h1 : k = a + Real.sqrt b)
  (h2 : ∀ k > 0, |Real.log k / Real.log 2 - Real.log (k + 6) / Real.log 2| = 1) :
  a + b = 6 :=
by
  sorry

end a_plus_b_eq_six_l406_40624


namespace hyperbola_asymptote_slope_l406_40604

theorem hyperbola_asymptote_slope :
  (∃ m : ℚ, m > 0 ∧ ∀ x : ℚ, ∀ y : ℚ, ((x*x/16 - y*y/25 = 1) → (y = m * x ∨ y = -m * x))) → m = 5/4 :=
sorry

end hyperbola_asymptote_slope_l406_40604


namespace arithmetic_sequence_general_formula_l406_40667

def f (x : ℝ) : ℝ := x^2 - 4 * x + 2

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

theorem arithmetic_sequence_general_formula (a : ℕ → ℝ) (x : ℝ)
  (h_arith : arithmetic_seq a)
  (h1 : a 1 = f (x + 1))
  (h2 : a 2 = 0)
  (h3 : a 3 = f (x - 1)) :
  ∀ n, a n = 2 * n - 4 ∨ a n = 4 - 2 * n :=
by
  sorry

end arithmetic_sequence_general_formula_l406_40667


namespace smallest_palindrome_in_bases_2_and_4_l406_40655

def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  let repr := n.digits base
  repr = repr.reverse

theorem smallest_palindrome_in_bases_2_and_4 (x : ℕ) :
  (x > 15) ∧ is_palindrome x 2 ∧ is_palindrome x 4 → x = 17 :=
by
  sorry

end smallest_palindrome_in_bases_2_and_4_l406_40655


namespace condition_of_inequality_l406_40676

theorem condition_of_inequality (x y : ℝ) (h : x^2 + y^2 ≤ 2 * (x + y - 1)) : x = 1 ∧ y = 1 :=
by
  sorry

end condition_of_inequality_l406_40676


namespace pizza_cost_l406_40633

theorem pizza_cost (soda_cost jeans_cost start_money quarters_left : ℝ) (quarters_value : ℝ) (total_left : ℝ) (pizza_cost : ℝ) :
  soda_cost = 1.50 → 
  jeans_cost = 11.50 → 
  start_money = 40 → 
  quarters_left = 97 → 
  quarters_value = 0.25 → 
  total_left = quarters_left * quarters_value → 
  pizza_cost = start_money - total_left - (soda_cost + jeans_cost) → 
  pizza_cost = 2.75 :=
by
  sorry

end pizza_cost_l406_40633


namespace angie_total_taxes_l406_40673

theorem angie_total_taxes:
  ∀ (salary : ℕ) (N_1 N_2 N_3 N_4 T_1 T_2 T_3 T_4 U_1 U_2 U_3 U_4 left_over : ℕ),
  salary = 80 →
  N_1 = 12 → T_1 = 8 → U_1 = 5 →
  N_2 = 15 → T_2 = 6 → U_2 = 7 →
  N_3 = 10 → T_3 = 9 → U_3 = 6 →
  N_4 = 14 → T_4 = 7 → U_4 = 4 →
  left_over = 18 →
  T_1 + T_2 + T_3 + T_4 = 30 :=
by
  intros salary N_1 N_2 N_3 N_4 T_1 T_2 T_3 T_4 U_1 U_2 U_3 U_4 left_over
  sorry

end angie_total_taxes_l406_40673


namespace john_coffees_per_day_l406_40659

theorem john_coffees_per_day (x : ℕ)
  (h1 : ∀ p : ℕ, p = 2)
  (h2 : ∀ p : ℕ, p = p + p / 2)
  (h3 : ∀ n : ℕ, n = x / 2)
  (h4 : ∀ d : ℕ, 2 * x - 3 * (x / 2) = 2) :
  x = 4 :=
by
  sorry

end john_coffees_per_day_l406_40659


namespace Carol_rectangle_length_l406_40631

theorem Carol_rectangle_length :
  (∃ (L : ℕ), (L * 15 = 4 * 30) → L = 8) :=
by
  sorry

end Carol_rectangle_length_l406_40631


namespace payment_for_30_kilograms_l406_40617

-- Define the price calculation based on quantity x
def payment_amount (x : ℕ) : ℕ :=
  if x ≤ 10 then 20 * x
  else 16 * x + 40

-- Prove that for x = 30, the payment amount y equals 520
theorem payment_for_30_kilograms : payment_amount 30 = 520 := by
  sorry

end payment_for_30_kilograms_l406_40617


namespace bound_on_k_l406_40601

variables {n k : ℕ}
variables (a : ℕ → ℕ) (h1 : 1 ≤ k) (h2 : ∀ i j, 1 ≤ i → j ≤ k → i < j → a i < a j)
variables (h3 : ∀ i, a i ≤ n) (h4 : (∀ i j : ℕ, i ≤ j → i ≤ k → j ≤ k → a i ≠ a j))
variables (h5 : (∀ i j : ℕ, i ≤ j → i ≤ k → j ≤ k → ∀ m p, m ≤ p → m ≤ k → p ≤ k → a i + a j ≠ a m + a p))

theorem bound_on_k : k ≤ Nat.floor (Real.sqrt (2 * n) + 1) :=
sorry

end bound_on_k_l406_40601


namespace modulo_calculation_l406_40697

theorem modulo_calculation : (68 * 97 * 113) % 25 = 23 := by
  sorry

end modulo_calculation_l406_40697


namespace integer_roots_if_q_positive_no_integer_roots_if_q_negative_l406_40661

theorem integer_roots_if_q_positive (p q : ℤ) (hq : q > 0) :
  (∃ x1 x2 : ℤ, x1 * x2 = q ∧ x1 + x2 = p) ∧
  (∃ y1 y2 : ℤ, y1 * y2 = q ∧ y1 + y2 = p + 1) :=
sorry

theorem no_integer_roots_if_q_negative (p q : ℤ) (hq : q < 0) :
  ¬ ((∃ x1 x2 : ℤ, x1 * x2 = q ∧ x1 + x2 = p) ∧
  (∃ y1 y2 : ℤ, y1 * y2 = q ∧ y1 + y2 = p + 1)) :=
sorry

end integer_roots_if_q_positive_no_integer_roots_if_q_negative_l406_40661


namespace birdseed_needed_weekly_birdseed_needed_l406_40693

def parakeet_daily_consumption := 2
def parrot_daily_consumption := 14
def finch_daily_consumption := parakeet_daily_consumption / 2
def num_parakeets := 3
def num_parrots := 2
def num_finches := 4
def days_in_week := 7

theorem birdseed_needed :
  num_parakeets * parakeet_daily_consumption +
  num_parrots * parrot_daily_consumption +
  num_finches * finch_daily_consumption = 38 :=
by
  sorry

theorem weekly_birdseed_needed :
  38 * days_in_week = 266 :=
by
  sorry

end birdseed_needed_weekly_birdseed_needed_l406_40693


namespace exist_non_quadratic_residues_sum_l406_40635

noncomputable section

def is_quadratic_residue_mod (p a : ℤ) : Prop :=
  ∃ x : ℤ, x^2 ≡ a [ZMOD p]

theorem exist_non_quadratic_residues_sum {p : ℤ} (hp : p > 5) (hp_modeq : p ≡ 1 [ZMOD 4]) (a : ℤ) : 
  ∃ b c : ℤ, a = b + c ∧ ¬is_quadratic_residue_mod p b ∧ ¬is_quadratic_residue_mod p c :=
sorry

end exist_non_quadratic_residues_sum_l406_40635


namespace restaurant_cost_l406_40620

theorem restaurant_cost (total_people kids adult_cost : ℕ)
  (h1 : total_people = 12)
  (h2 : kids = 7)
  (h3 : adult_cost = 3) :
  total_people - kids * adult_cost = 15 := by
  sorry

end restaurant_cost_l406_40620


namespace students_answered_both_correctly_l406_40626

theorem students_answered_both_correctly :
  ∀ (total_students set_problem function_problem both_incorrect x : ℕ),
    total_students = 50 → 
    set_problem = 40 →
    function_problem = 31 →
    both_incorrect = 4 →
    x = total_students - both_incorrect - (set_problem + function_problem - total_students) →
    x = 25 :=
by
  intros total_students set_problem function_problem both_incorrect x
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  exact h5

end students_answered_both_correctly_l406_40626


namespace females_watch_eq_seventy_five_l406_40632

-- Definition of conditions
def males_watch : ℕ := 85
def females_dont_watch : ℕ := 120
def total_watch : ℕ := 160
def total_dont_watch : ℕ := 180

-- Definition of the proof problem
theorem females_watch_eq_seventy_five :
  total_watch - males_watch = 75 :=
by
  sorry

end females_watch_eq_seventy_five_l406_40632


namespace find_seventh_term_l406_40625

theorem find_seventh_term :
  ∃ r : ℚ, ∃ (a₁ a₇ a₁₀ : ℚ), 
    a₁ = 12 ∧ 
    a₁₀ = 78732 ∧ 
    a₇ = a₁ * r^6 ∧ 
    a₁₀ = a₁ * r^9 ∧ 
    a₇ = 8748 :=
by
  sorry

end find_seventh_term_l406_40625


namespace jafaris_candy_l406_40664

-- Define the conditions
variable (candy_total : Nat)
variable (taquon_candy : Nat)
variable (mack_candy : Nat)

-- Assume the conditions from the problem
axiom candy_total_def : candy_total = 418
axiom taquon_candy_def : taquon_candy = 171
axiom mack_candy_def : mack_candy = 171

-- Define the statement to be proved
theorem jafaris_candy : (candy_total - (taquon_candy + mack_candy)) = 76 :=
by
  -- Proof goes here
  sorry

end jafaris_candy_l406_40664


namespace divisibility_of_powers_l406_40627

theorem divisibility_of_powers (a b c d m : ℤ) (h_odd : m % 2 = 1)
  (h_sum_div : m ∣ (a + b + c + d))
  (h_sum_squares_div : m ∣ (a^2 + b^2 + c^2 + d^2)) : 
  m ∣ (a^4 + b^4 + c^4 + d^4 + 4 * a * b * c * d) :=
sorry

end divisibility_of_powers_l406_40627


namespace parallel_slope_l406_40638

theorem parallel_slope (x y : ℝ) (h : 3 * x + 6 * y = -21) : 
    ∃ m : ℝ, m = -1 / 2 :=
by
  sorry

end parallel_slope_l406_40638


namespace marathon_time_l406_40637

noncomputable def marathon_distance : ℕ := 26
noncomputable def first_segment_distance : ℕ := 10
noncomputable def first_segment_time : ℕ := 1
noncomputable def remaining_distance : ℕ := marathon_distance - first_segment_distance
noncomputable def pace_percentage : ℕ := 80
noncomputable def initial_pace : ℕ := first_segment_distance / first_segment_time
noncomputable def remaining_pace : ℕ := (initial_pace * pace_percentage) / 100
noncomputable def remaining_time : ℕ := remaining_distance / remaining_pace
noncomputable def total_time : ℕ := first_segment_time + remaining_time

theorem marathon_time : total_time = 3 := by
  -- Proof omitted: hence using sorry
  sorry

end marathon_time_l406_40637


namespace distinct_integers_sum_l406_40615

theorem distinct_integers_sum (n : ℕ) (h : n > 3) (a : Fin n → ℤ)
  (h1 : ∀ i, 1 ≤ a i) (h2 : ∀ i j, i < j → a i < a j) (h3 : ∀ i, a i ≤ 2 * n - 3) :
  ∃ (i j k l m : Fin n), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ 
  k ≠ l ∧ k ≠ m ∧ l ≠ m ∧ a i + a j = a k + a l ∧ a k + a l = a m :=
by
  sorry

end distinct_integers_sum_l406_40615


namespace binomial_probability_l406_40648

theorem binomial_probability (n : ℕ) (p : ℝ) (h1 : (n * p = 300)) (h2 : (n * p * (1 - p) = 200)) :
    p = 1 / 3 :=
by
  sorry

end binomial_probability_l406_40648


namespace sum_of_translated_parabolas_l406_40691

noncomputable def parabola_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := - (a * x^2 + b * x + c)

noncomputable def translated_right (a b c : ℝ) (x : ℝ) : ℝ := parabola_equation a b c (x - 3)

noncomputable def translated_left (a b c : ℝ) (x : ℝ) : ℝ := reflected_parabola a b c (x + 3)

theorem sum_of_translated_parabolas (a b c x : ℝ) : 
  (translated_right a b c x) + (translated_left a b c x) = -12 * a * x - 6 * b :=
sorry

end sum_of_translated_parabolas_l406_40691


namespace change_percentage_difference_l406_40653

theorem change_percentage_difference 
  (initial_yes : ℚ) (initial_no : ℚ) (initial_undecided : ℚ)
  (final_yes : ℚ) (final_no : ℚ) (final_undecided : ℚ)
  (h_initial : initial_yes = 0.4 ∧ initial_no = 0.3 ∧ initial_undecided = 0.3)
  (h_final : final_yes = 0.6 ∧ final_no = 0.1 ∧ final_undecided = 0.3) :
  (final_yes - initial_yes + initial_no - final_no) = 0.2 := by
sorry

end change_percentage_difference_l406_40653


namespace func4_same_domain_range_as_func1_l406_40634

noncomputable def func1_domain : Set ℝ := {x | 0 < x}
noncomputable def func1_range : Set ℝ := {y | 0 < y}

noncomputable def func4_domain : Set ℝ := {x | 0 < x}
noncomputable def func4_range : Set ℝ := {y | 0 < y}

theorem func4_same_domain_range_as_func1 :
  (func4_domain = func1_domain) ∧ (func4_range = func1_range) :=
sorry

end func4_same_domain_range_as_func1_l406_40634


namespace sin_double_angle_l406_40662

open Real

theorem sin_double_angle (α : ℝ) (h : tan α = -3/5) : sin (2 * α) = -15/17 :=
by
  -- We are skipping the proof here
  sorry

end sin_double_angle_l406_40662


namespace pine_trees_multiple_of_27_l406_40629

noncomputable def numberOfPineTrees (n : ℕ) : ℕ := 27 * n

theorem pine_trees_multiple_of_27 (oak_trees : ℕ) (max_trees_per_row : ℕ) (rows_of_oak : ℕ) :
  oak_trees = 54 → max_trees_per_row = 27 → rows_of_oak = oak_trees / max_trees_per_row →
  ∃ n : ℕ, numberOfPineTrees n = 27 * n :=
by
  intros
  use (oak_trees - rows_of_oak * max_trees_per_row) / 27
  sorry

end pine_trees_multiple_of_27_l406_40629


namespace solve_m_problem_l406_40685

theorem solve_m_problem :
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0) →
  m ∈ Set.Ico (-1/4 : ℝ) 2 :=
sorry

end solve_m_problem_l406_40685


namespace frogs_count_l406_40636

variables (Alex Brian Chris LeRoy Mike : Type) 

-- Definitions for the species
def toad (x : Type) : Prop := ∃ p : Prop, p -- Dummy definition for toads
def frog (x : Type) : Prop := ∃ p : Prop, ¬p -- Dummy definition for frogs

-- Conditions
axiom Alex_statement : (toad Alex) → (∃ x : ℕ, x = 3) ∧ (frog Alex) → (¬(∃ x : ℕ, x = 3))
axiom Brian_statement : (toad Brian) → (toad Mike) ∧ (frog Brian) → (frog Mike)
axiom Chris_statement : (toad Chris) → (toad LeRoy) ∧ (frog Chris) → (frog LeRoy)
axiom LeRoy_statement : (toad LeRoy) → (toad Chris) ∧ (frog LeRoy) → (frog Chris)
axiom Mike_statement : (toad Mike) → (∃ x : ℕ, x < 3) ∧ (frog Mike) → (¬(∃ x : ℕ, x < 3))

theorem frogs_count (total : ℕ) : total = 5 → 
  (∃ frog_count : ℕ, frog_count = 2) :=
by
  -- Leaving the proof as a sorry placeholder
  sorry

end frogs_count_l406_40636


namespace equal_distribution_arithmetic_sequence_l406_40640

theorem equal_distribution_arithmetic_sequence :
  ∃ a d : ℚ, (a - 2 * d) + (a - d) = (a + (a + d) + (a + 2 * d)) ∧
  5 * a = 5 ∧
  a + 2 * d = 2 / 3 :=
by
  sorry

end equal_distribution_arithmetic_sequence_l406_40640


namespace value_of_a_plus_b_is_zero_l406_40612

noncomputable def sum_geometric_sequence (a b : ℝ) (n : ℕ) : ℝ :=
  a * 2^n + b

theorem value_of_a_plus_b_is_zero (a b : ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = sum_geometric_sequence a b n) :
  a + b = 0 := 
sorry

end value_of_a_plus_b_is_zero_l406_40612


namespace candy_mixture_price_l406_40689

theorem candy_mixture_price
  (price_first_per_kg : ℝ) (price_second_per_kg : ℝ) (weight_ratio : ℝ) (weight_second : ℝ) 
  (h1 : price_first_per_kg = 10) 
  (h2 : price_second_per_kg = 15) 
  (h3 : weight_ratio = 3) 
  : (price_first_per_kg * weight_ratio * weight_second + price_second_per_kg * weight_second) / 
    (weight_ratio * weight_second + weight_second) = 11.25 :=
by
  sorry

end candy_mixture_price_l406_40689


namespace range_of_a_l406_40606

open Real

noncomputable def doesNotPassThroughSecondQuadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 ≠ 0

theorem range_of_a : {a : ℝ | doesNotPassThroughSecondQuadrant a} = {a : ℝ | 2 ≤ a } :=
by
  ext
  sorry

end range_of_a_l406_40606


namespace number_of_correct_answers_l406_40695

-- We define variables C (number of correct answers) and W (number of wrong answers).
variables (C W : ℕ)

-- Define the conditions given in the problem.
def conditions :=
  C + W = 75 ∧ 4 * C - W = 125

-- Define the theorem which states that the number of correct answers is 40.
theorem number_of_correct_answers
  (h : conditions C W) :
  C = 40 :=
sorry

end number_of_correct_answers_l406_40695


namespace necessary_but_not_sufficient_l406_40605

section geometric_progression

variables {a b c : ℝ}

def geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, a = b / r ∧ c = b * r

def necessary_condition (a b c : ℝ) : Prop :=
  a * c = b^2

theorem necessary_but_not_sufficient :
  (geometric_progression a b c → necessary_condition a b c) ∧
  (¬ (necessary_condition a b c → geometric_progression a b c)) :=
by sorry

end geometric_progression

end necessary_but_not_sufficient_l406_40605


namespace intersection_A_B_l406_40657

def A : Set ℤ := {x | abs x < 3}
def B : Set ℤ := {x | abs x > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by sorry

end intersection_A_B_l406_40657


namespace boys_camp_total_l406_40623

theorem boys_camp_total (T : ℕ) 
  (h1 : 0.20 * T = (0.20 : ℝ) * T) 
  (h2 : (0.30 : ℝ) * (0.20 * T) = (0.30 : ℝ) * (0.20 * T)) 
  (h3 : (0.70 : ℝ) * (0.20 * T) = 63) :
  T = 450 :=
by
  sorry

end boys_camp_total_l406_40623


namespace part1_part2_l406_40645

noncomputable def f (a x : ℝ) : ℝ :=
  a * Real.sin x - 1/2 * Real.cos (2 * x) + a - 3/a + 1/2

theorem part1 (a : ℝ) (h₀ : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 0 1 := sorry

theorem part2 (a : ℝ) (h₀ : a ≠ 0) (h₁ : a ≥ 2) :
  (∃ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 2 3 := sorry

end part1_part2_l406_40645


namespace sum_opposite_signs_eq_zero_l406_40686

theorem sum_opposite_signs_eq_zero (x y : ℝ) (h : x * y < 0) : x + y = 0 :=
sorry

end sum_opposite_signs_eq_zero_l406_40686


namespace quiz_competition_top_three_orders_l406_40622

theorem quiz_competition_top_three_orders :
  let participants := 4
  let top_positions := 3
  let permutations := (Nat.factorial participants) / (Nat.factorial (participants - top_positions))
  permutations = 24 := 
by
  sorry

end quiz_competition_top_three_orders_l406_40622


namespace unique_solution_a_eq_4_l406_40665

theorem unique_solution_a_eq_4 (a : ℝ) (h : ∀ x1 x2 : ℝ, (a * x1^2 + a * x1 + 1 = 0 ∧ a * x2^2 + a * x2 + 1 = 0) → x1 = x2) : a = 4 :=
sorry

end unique_solution_a_eq_4_l406_40665


namespace sufficient_not_necessary_condition_l406_40616

variable (x y : ℝ)

theorem sufficient_not_necessary_condition :
  (x > 1 ∧ y > 1) → (x + y > 2 ∧ x * y > 1) ∧
  ¬((x + y > 2 ∧ x * y > 1) → (x > 1 ∧ y > 1)) :=
by
  sorry

end sufficient_not_necessary_condition_l406_40616


namespace boat_speed_in_still_water_l406_40671

variable (B S : ℝ)

def downstream_speed := 10
def upstream_speed := 4

theorem boat_speed_in_still_water :
  B + S = downstream_speed → 
  B - S = upstream_speed → 
  B = 7 :=
by
  intros h₁ h₂
  -- We would insert the proof steps here
  sorry

end boat_speed_in_still_water_l406_40671


namespace groupB_is_conditional_control_l406_40642

-- Definitions based on conditions
def groupA_medium (nitrogen_sources : Set String) : Prop := nitrogen_sources = {"urea"}
def groupB_medium (nitrogen_sources : Set String) : Prop := nitrogen_sources = {"urea", "nitrate"}

-- The property that defines a conditional control in this context.
def conditional_control (control_sources : Set String) (experimental_sources : Set String) : Prop :=
  control_sources ≠ experimental_sources ∧ "urea" ∈ control_sources ∧ "nitrate" ∈ experimental_sources

-- Prove that Group B's experiment forms a conditional control
theorem groupB_is_conditional_control :
  ∃ nitrogen_sourcesA nitrogen_sourcesB, groupA_medium nitrogen_sourcesA ∧ groupB_medium nitrogen_sourcesB ∧
  conditional_control nitrogen_sourcesA nitrogen_sourcesB :=
by
  sorry

end groupB_is_conditional_control_l406_40642


namespace cos_seven_pi_six_eq_neg_sqrt_three_div_two_l406_40639

noncomputable def cos_seven_pi_six : Real :=
  Real.cos (7 * Real.pi / 6)

theorem cos_seven_pi_six_eq_neg_sqrt_three_div_two :
  cos_seven_pi_six = -Real.sqrt 3 / 2 :=
sorry

end cos_seven_pi_six_eq_neg_sqrt_three_div_two_l406_40639


namespace max_yellow_apples_max_total_apples_l406_40603

-- Definitions for the conditions
def num_green_apples : Nat := 10
def num_yellow_apples : Nat := 13
def num_red_apples : Nat := 18

-- Predicate for the stopping condition
def stop_condition (green yellow red : Nat) : Prop :=
  green < yellow ∧ yellow < red

-- Proof problem for maximum number of yellow apples
theorem max_yellow_apples (green yellow red : Nat) :
  num_green_apples = 10 →
  num_yellow_apples = 13 →
  num_red_apples = 18 →
  (∀ g y r, stop_condition g y r → y ≤ 13) →
  yellow ≤ 13 :=
sorry

-- Proof problem for maximum total number of apples
theorem max_total_apples (green yellow red : Nat) :
  num_green_apples = 10 →
  num_yellow_apples = 13 →
  num_red_apples = 18 →
  (∀ g y r, stop_condition g y r → g + y + r ≤ 39) →
  green + yellow + red ≤ 39 :=
sorry

end max_yellow_apples_max_total_apples_l406_40603


namespace ada_original_seat_l406_40658

theorem ada_original_seat {positions : Fin 6 → Fin 6} 
  (Bea Ceci Dee Edie Fred Ada: Fin 6)
  (h1: Ada = 0)
  (h2: positions (Bea + 1) = Bea)
  (h3: positions (Ceci - 2) = Ceci)
  (h4: positions Dee = Edie ∧ positions Edie = Dee)
  (h5: positions Fred = Fred) :
  Ada = 1 → Bea = 1 → Ceci = 3 → Dee = 4 → Edie = 5 → Fred = 6 → Ada = 1 :=
by
  intros
  sorry

end ada_original_seat_l406_40658


namespace positive_real_solution_l406_40670

theorem positive_real_solution (x : ℝ) (h : 0 < x)
  (h_eq : (1/3) * (2 * x^2 + 3) = (x^2 - 40 * x - 8) * (x^2 + 20 * x + 4)) :
  x = 20 + Real.sqrt 409 :=
sorry

end positive_real_solution_l406_40670


namespace min_value_of_ratio_l406_40652

noncomputable def min_ratio (a b c d : ℕ) : ℝ :=
  let num := 1000 * a + 100 * b + 10 * c + d
  let denom := a + b + c + d
  (num : ℝ) / (denom : ℝ)

theorem min_value_of_ratio : 
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ 
  min_ratio a b c d = 60.5 :=
by
  sorry

end min_value_of_ratio_l406_40652
