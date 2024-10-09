import Mathlib

namespace minimum_even_N_for_A_2015_turns_l1869_186915

noncomputable def a (n : ℕ) : ℕ :=
  6 * 2^n - 4

def A_minimum_even_moves_needed (k : ℕ) : ℕ :=
  2015 - 1

theorem minimum_even_N_for_A_2015_turns :
  ∃ N : ℕ, 2 ∣ N ∧ A_minimum_even_moves_needed 2015 ≤ N ∧ a 1007 = 6 * 2^1007 - 4 := by
  sorry

end minimum_even_N_for_A_2015_turns_l1869_186915


namespace no_such_polynomials_exists_l1869_186905

theorem no_such_polynomials_exists :
  ¬ ∃ (f g : Polynomial ℚ), (∀ x y : ℚ, f.eval x * g.eval y = x^200 * y^200 + 1) := 
by 
  sorry

end no_such_polynomials_exists_l1869_186905


namespace monotonic_intervals_slope_tangent_line_inequality_condition_l1869_186924

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * (a + 2) * x^2 + 2 * a * x
noncomputable def g (a x : ℝ) : ℝ := (1/2) * (a - 5) * x^2

theorem monotonic_intervals (a : ℝ) (h : a ≥ 4) :
  (∀ x, deriv (f a) x = x^2 - (a + 2) * x + 2 * a) ∧
  ((∀ x, x < 2 → deriv (f a) x > 0) ∧ (∀ x, x > a → deriv (f a) x > 0)) ∧
  (∀ x, 2 < x ∧ x < a → deriv (f a) x < 0) :=
sorry

theorem slope_tangent_line (a : ℝ) (h : a ≥ 4) :
  (∀ x, deriv (f a) x = x^2 - (a + 2) * x + 2 * a) ∧
  (∀ x_0 y_0 k, y_0 = f a x_0 ∧ k = deriv (f a) x_0 ∧ k ≥ -(25/4) →
    4 ≤ a ∧ a ≤ 7) :=
sorry

theorem inequality_condition (a : ℝ) (h : a ≥ 4) :
  (∀ x_1 x_2, 3 ≤ x_1 ∧ x_1 < x_2 ∧ x_2 ≤ 4 →
    abs (f a x_1 - f a x_2) > abs (g a x_1 - g a x_2)) →
  (14/3 ≤ a ∧ a ≤ 6) :=
sorry

end monotonic_intervals_slope_tangent_line_inequality_condition_l1869_186924


namespace janeth_balloons_l1869_186931

/-- Janeth's total remaining balloons after accounting for burst ones. -/
def total_remaining_balloons (round_bags : Nat) (round_per_bag : Nat) (burst_round : Nat)
    (long_bags : Nat) (long_per_bag : Nat) (burst_long : Nat)
    (heart_bags : Nat) (heart_per_bag : Nat) (burst_heart : Nat) : Nat :=
  let total_round := round_bags * round_per_bag - burst_round
  let total_long := long_bags * long_per_bag - burst_long
  let total_heart := heart_bags * heart_per_bag - burst_heart
  total_round + total_long + total_heart

theorem janeth_balloons :
  total_remaining_balloons 5 25 5 4 35 7 3 40 3 = 370 :=
by
  let round_bags := 5
  let round_per_bag := 25
  let burst_round := 5
  let long_bags := 4
  let long_per_bag := 35
  let burst_long := 7
  let heart_bags := 3
  let heart_per_bag := 40
  let burst_heart := 3
  show total_remaining_balloons round_bags round_per_bag burst_round long_bags long_per_bag burst_long heart_bags heart_per_bag burst_heart = 370
  sorry

end janeth_balloons_l1869_186931


namespace Kolya_can_form_triangles_l1869_186994

theorem Kolya_can_form_triangles :
  ∃ (K1a K1b K1c K3a K3b K3c V1 V2 V3 : ℝ), 
  (K1a + K1b + K1c = 1) ∧
  (K3a + K3b + K3c = 1) ∧
  (V1 + V2 + V3 = 1) ∧
  (K1a = 0.5) ∧ (K1b = 0.25) ∧ (K1c = 0.25) ∧
  (K3a = 0.5) ∧ (K3b = 0.25) ∧ (K3c = 0.25) ∧
  (∀ (V1 V2 V3 : ℝ), V1 + V2 + V3 = 1 → 
  (
    (K1a + V1 > K3b ∧ K1a + K3b > V1 ∧ V1 + K3b > K1a) ∧ 
    (K1b + V2 > K3a ∧ K1b + K3a > V2 ∧ V2 + K3a > K1b) ∧ 
    (K1c + V3 > K3c ∧ K1c + K3c > V3 ∧ V3 + K3c > K1c)
  )) :=
sorry

end Kolya_can_form_triangles_l1869_186994


namespace smallest_digit_divisible_by_9_l1869_186942

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (5 + 2 + 8 + 4 + 6 + d) % 9 = 0 ∧ ∀ e : ℕ, (5 + 2 + 8 + 4 + 6 + e) % 9 = 0 → d ≤ e := 
by {
  sorry
}

end smallest_digit_divisible_by_9_l1869_186942


namespace minimum_value_g_l1869_186999

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - x - 2

def g (x : ℝ) : ℝ := (x + a)^2 - (x + a) - 2 + x

theorem minimum_value_g (a : ℝ) :
  (if 1 ≤ a then g a (-1) = a^2 - 3 * a - 1 else
   if -3 < a ∧ a < 1 then g a (-a) = -a - 2 else
   if a ≤ -3 then g a 3 = a^2 + 5 * a + 7 else false) :=
by
  sorry

end minimum_value_g_l1869_186999


namespace proof_problem_l1869_186938

theorem proof_problem (a b : ℝ) (H1 : ∀ x : ℝ, (ax^2 - 3*x + 6 > 4) ↔ (x < 1 ∨ x > b)) :
  a = 1 ∧ b = 2 ∧
  (∀ c : ℝ, (ax^2 - (a*c + b)*x + b*c < 0) ↔ 
   (if c > 2 then 2 < x ∧ x < c
    else if c < 2 then c < x ∧ x < 2
    else false)) :=
by
  sorry

end proof_problem_l1869_186938


namespace lorelai_jellybeans_l1869_186944

variable (Gigi Rory Luke Lane Lorelai : ℕ)
variable (h1 : Gigi = 15)
variable (h2 : Rory = Gigi + 30)
variable (h3 : Luke = 2 * Rory)
variable (h4 : Lane = Gigi + 10)
variable (h5 : Lorelai = 3 * (Gigi + Luke + Lane))

theorem lorelai_jellybeans : Lorelai = 390 := by
  sorry

end lorelai_jellybeans_l1869_186944


namespace problem1_problem2_l1869_186902

-- Problem 1: Prove the solution set of the given inequality
theorem problem1 (x : ℝ) : (|x - 2| + 2 * |x - 1| > 5) ↔ (x < -1/3 ∨ x > 3) := 
sorry

-- Problem 2: Prove the range of values for 'a' such that the inequality holds
theorem problem2 (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ |a - 2|) ↔ (a ≤ 3/2) :=
sorry

end problem1_problem2_l1869_186902


namespace sample_size_correct_l1869_186971

def population_size : Nat := 8000
def sampled_students : List Nat := List.replicate 400 1 -- We use 1 as a placeholder for the heights

theorem sample_size_correct : sampled_students.length = 400 := by
  sorry

end sample_size_correct_l1869_186971


namespace best_fitting_model_l1869_186922

theorem best_fitting_model :
  ∀ (R1 R2 R3 R4 : ℝ), R1 = 0.976 → R2 = 0.776 → R3 = 0.076 → R4 = 0.351 →
  R1 = max R1 (max R2 (max R3 R4)) :=
by
  intros R1 R2 R3 R4 hR1 hR2 hR3 hR4
  rw [hR1, hR2, hR3, hR4]
  sorry

end best_fitting_model_l1869_186922


namespace average_branches_per_foot_l1869_186947

theorem average_branches_per_foot :
  let b1 := 200
  let h1 := 50
  let b2 := 180
  let h2 := 40
  let b3 := 180
  let h3 := 60
  let b4 := 153
  let h4 := 34
  (b1 / h1 + b2 / h2 + b3 / h3 + b4 / h4) / 4 = 4 := by
  sorry

end average_branches_per_foot_l1869_186947


namespace largest_square_perimeter_l1869_186957

-- Define the conditions
def rectangle_length : ℕ := 80
def rectangle_width : ℕ := 60

-- Define the theorem to prove
theorem largest_square_perimeter : 4 * rectangle_width = 240 := by
  -- The proof steps are omitted
  sorry

end largest_square_perimeter_l1869_186957


namespace min_value_l1869_186927

noncomputable def min_value_expr (a b c d : ℝ) : ℝ :=
  (a + b) / c + (b + c) / a + (c + d) / b

theorem min_value 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  min_value_expr a b c d ≥ 6 
  := sorry

end min_value_l1869_186927


namespace decrease_in_demand_l1869_186997

theorem decrease_in_demand (init_price new_price demand : ℝ) (init_demand : ℕ) (price_increase : ℝ) (original_revenue new_demand : ℝ) :
  init_price = 20 ∧ init_demand = 500 ∧ price_increase = 5 ∧ demand = init_price + price_increase ∧ 
  original_revenue = init_price * init_demand ∧ new_demand ≤ init_demand ∧ 
  new_demand * demand ≥ original_revenue → 
  init_demand - new_demand = 100 :=
by 
  sorry

end decrease_in_demand_l1869_186997


namespace cos_2theta_plus_sin_2theta_l1869_186972

theorem cos_2theta_plus_sin_2theta (θ : ℝ) (h : 3 * Real.sin θ = Real.cos θ) : 
  Real.cos (2 * θ) + Real.sin (2 * θ) = 7 / 5 :=
by
  sorry

end cos_2theta_plus_sin_2theta_l1869_186972


namespace smallest_N_exists_l1869_186991

theorem smallest_N_exists : ∃ N : ℕ, 
  (N % 9 = 8) ∧
  (N % 8 = 7) ∧
  (N % 7 = 6) ∧
  (N % 6 = 5) ∧
  (N % 5 = 4) ∧
  (N % 4 = 3) ∧
  (N % 3 = 2) ∧
  (N % 2 = 1) ∧
  N = 503 :=
by {
  sorry
}

end smallest_N_exists_l1869_186991


namespace number_of_valid_subsets_l1869_186988

def setA : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def oddSet : Finset ℕ := {1, 3, 5, 7}
def evenSet : Finset ℕ := {2, 4, 6}

theorem number_of_valid_subsets : 
  (oddSet.powerset.card * (evenSet.powerset.card - 1) - oddSet.powerset.card) = 96 :=
by sorry

end number_of_valid_subsets_l1869_186988


namespace distinct_book_arrangements_l1869_186940

def num_books := 7
def num_identical_books := 3
def num_unique_books := num_books - num_identical_books

theorem distinct_book_arrangements :
  (Nat.factorial num_books) / (Nat.factorial num_identical_books) = 840 := 
  by 
  sorry

end distinct_book_arrangements_l1869_186940


namespace tom_age_ratio_l1869_186900

-- Definitions of the variables
variables (T : ℕ) (N : ℕ)

-- Conditions given in the problem
def condition1 : Prop := T = 2 * (T / 2)
def condition2 : Prop := (T - 3) = 3 * (T / 2 - 12)

-- The ratio theorem to prove
theorem tom_age_ratio (h1 : condition1 T) (h2 : condition2 T) : T / N = 22 :=
by
  sorry

end tom_age_ratio_l1869_186900


namespace decreasing_sufficient_condition_l1869_186941

theorem decreasing_sufficient_condition {a : ℝ} (h_pos : 0 < a) (h_neq_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x > a^y) →
  (∀ x y : ℝ, x < y → (a-2)*x^3 > (a-2)*y^3) :=
by
  sorry

end decreasing_sufficient_condition_l1869_186941


namespace trajectory_of_center_of_moving_circle_l1869_186963

theorem trajectory_of_center_of_moving_circle
  (x y : ℝ)
  (C1 : (x + 4)^2 + y^2 = 2)
  (C2 : (x - 4)^2 + y^2 = 2) :
  ((x = 0) ∨ (x^2 / 2 - y^2 / 14 = 1)) :=
sorry

end trajectory_of_center_of_moving_circle_l1869_186963


namespace max_n_leq_V_l1869_186914

theorem max_n_leq_V (n : ℤ) (V : ℤ) (h1 : 102 * n^2 <= V) (h2 : ∀ k : ℤ, (102 * k^2 <= V) → k <= 8) : V >= 6528 :=
sorry

end max_n_leq_V_l1869_186914


namespace lila_stickers_correct_l1869_186973

-- Defining the constants for number of stickers each has
def Kristoff_stickers : ℕ := 85
def Riku_stickers : ℕ := 25 * Kristoff_stickers
def Lila_stickers : ℕ := 2 * (Kristoff_stickers + Riku_stickers)

-- The theorem to prove
theorem lila_stickers_correct : Lila_stickers = 4420 := 
by {
  sorry
}

end lila_stickers_correct_l1869_186973


namespace sakshi_work_days_l1869_186954

theorem sakshi_work_days (x : ℝ) (efficiency_tanya : ℝ) (days_tanya : ℝ) 
  (h_efficiency : efficiency_tanya = 1.25) 
  (h_days : days_tanya = 4)
  (h_relationship : x / efficiency_tanya = days_tanya) : 
  x = 5 :=
by 
  -- Lean proof would go here
  sorry

end sakshi_work_days_l1869_186954


namespace probability_divisible_by_five_l1869_186953

def is_three_digit_number (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000

def ends_with_five (n: ℕ) : Prop := n % 10 = 5

def divisible_by_five (n: ℕ) : Prop := n % 5 = 0

theorem probability_divisible_by_five {N : ℕ} (h1: is_three_digit_number N) (h2: ends_with_five N) : 
  ∃ p : ℚ, p = 1 ∧ ∀ n, (is_three_digit_number n ∧ ends_with_five n) → (divisible_by_five n) :=
by
  sorry

end probability_divisible_by_five_l1869_186953


namespace divide_oranges_into_pieces_l1869_186986

-- Definitions for conditions
def oranges : Nat := 80
def friends : Nat := 200
def pieces_per_friend : Nat := 4

-- Theorem stating the problem and the answer
theorem divide_oranges_into_pieces :
    (oranges > 0) → (friends > 0) → (pieces_per_friend > 0) →
    ((friends * pieces_per_friend) / oranges = 10) :=
by
  intros
  sorry

end divide_oranges_into_pieces_l1869_186986


namespace total_days_2000_to_2003_correct_l1869_186916

-- Define the days in each type of year
def days_in_leap_year : ℕ := 366
def days_in_common_year : ℕ := 365

-- Define each year and its corresponding number of days
def year_2000 := days_in_leap_year
def year_2001 := days_in_common_year
def year_2002 := days_in_common_year
def year_2003 := days_in_common_year

-- Calculate the total number of days from 2000 to 2003
def total_days_2000_to_2003 : ℕ := year_2000 + year_2001 + year_2002 + year_2003

theorem total_days_2000_to_2003_correct : total_days_2000_to_2003 = 1461 := 
by
  unfold total_days_2000_to_2003 year_2000 year_2001 year_2002 year_2003 
        days_in_leap_year days_in_common_year 
  exact rfl

end total_days_2000_to_2003_correct_l1869_186916


namespace simplify_expr1_simplify_expr2_l1869_186912

variable (a b x y : ℝ)

theorem simplify_expr1 : 6 * a + 7 * b^2 - 9 + 4 * a - b^2 + 6 = 10 * a + 6 * b^2 - 3 :=
by
  sorry

theorem simplify_expr2 : 5 * x - 2 * (4 * x + 5 * y) + 3 * (3 * x - 4 * y) = 6 * x - 22 * y :=
by
  sorry

end simplify_expr1_simplify_expr2_l1869_186912


namespace total_milk_consumed_l1869_186992

theorem total_milk_consumed (regular_milk : ℝ) (soy_milk : ℝ) (H1 : regular_milk = 0.5) (H2: soy_milk = 0.1) :
    regular_milk + soy_milk = 0.6 :=
  by
  sorry

end total_milk_consumed_l1869_186992


namespace sum_of_digits_l1869_186936

theorem sum_of_digits :
  ∃ (a b : ℕ), (4 * 100 + a * 10 + 5) + 457 = (9 * 100 + b * 10 + 2) ∧
                (((9 + 2) - b) % 11 = 0) ∧
                (a + b = 4) :=
sorry

end sum_of_digits_l1869_186936


namespace fixed_monthly_charge_l1869_186970

variables (F C_J : ℝ)

-- Conditions
def january_bill := F + C_J = 46
def february_bill := F + 2 * C_J = 76

-- The proof goal
theorem fixed_monthly_charge
  (h_jan : january_bill F C_J)
  (h_feb : february_bill F C_J)
  (h_calls : C_J = 30) : F = 16 :=
by sorry

end fixed_monthly_charge_l1869_186970


namespace juniper_initial_bones_l1869_186965

theorem juniper_initial_bones (B : ℕ) (h : 2 * B - 2 = 6) : B = 4 := 
by
  sorry

end juniper_initial_bones_l1869_186965


namespace find_l_in_triangle_l1869_186981

/-- In triangle XYZ, if XY = 5, YZ = 12, XZ = 13, and YM is the angle bisector from vertex Y with YM = l * sqrt 2, then l equals 60/17. -/
theorem find_l_in_triangle (XY YZ XZ : ℝ) (YM l : ℝ) (hXY : XY = 5) (hYZ : YZ = 12) (hXZ : XZ = 13) (hYM : YM = l * Real.sqrt 2) : 
    l = 60 / 17 :=
sorry

end find_l_in_triangle_l1869_186981


namespace tetrahedron_volume_formula_l1869_186952

variables (r₀ S₀ S₁ S₂ S₃ V : ℝ)

theorem tetrahedron_volume_formula
  (h : V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀) :
  V = (1/3) * (S₁ + S₂ + S₃ - S₀) * r₀ :=
by { sorry }

end tetrahedron_volume_formula_l1869_186952


namespace Luke_spent_money_l1869_186975

theorem Luke_spent_money : ∀ (initial_money additional_money current_money x : ℕ),
  initial_money = 48 →
  additional_money = 21 →
  current_money = 58 →
  (initial_money + additional_money - current_money) = x →
  x = 11 :=
by
  intros initial_money additional_money current_money x h1 h2 h3 h4
  sorry

end Luke_spent_money_l1869_186975


namespace number_of_chocolates_l1869_186948

-- Define the dimensions of the box
def W_box := 30
def L_box := 20
def H_box := 5

-- Define the dimensions of one chocolate
def W_chocolate := 6
def L_chocolate := 4
def H_chocolate := 1

-- Calculate the volume of the box
def V_box := W_box * L_box * H_box

-- Calculate the volume of one chocolate
def V_chocolate := W_chocolate * L_chocolate * H_chocolate

-- Lean theorem statement for the proof problem
theorem number_of_chocolates : V_box / V_chocolate = 125 := 
by
  sorry

end number_of_chocolates_l1869_186948


namespace fraction_of_age_l1869_186984

theorem fraction_of_age (jane_age_current : ℕ) (years_since_babysit : ℕ) (age_oldest_babysat_current : ℕ) :
  jane_age_current = 32 →
  years_since_babysit = 12 →
  age_oldest_babysat_current = 23 →
  ∃ (f : ℚ), f = 11 / 20 :=
by
  intros
  sorry

end fraction_of_age_l1869_186984


namespace find_sum_l1869_186966

variable {a : ℕ → ℝ} {r : ℝ}

-- Conditions: a_n > 0 for all n
axiom pos : ∀ n : ℕ, a n > 0

-- Given equation: a_1 * a_5 + 2 * a_3 * a_5 + a_3 * a_7 = 25
axiom given_eq : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25

theorem find_sum : a 3 + a 5 = 5 :=
by
  sorry

end find_sum_l1869_186966


namespace min_value_fract_ineq_l1869_186939

theorem min_value_fract_ineq (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (1 / a + 9 / b) ≥ 16 := 
sorry

end min_value_fract_ineq_l1869_186939


namespace range_of_a_l1869_186950

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (1 / 2) * Real.log x

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := (2 * a * x^2 + 1) / (2 * x)

def p (a : ℝ) : Prop := ∀ x, 1 ≤ x → f_prime (a) (x) ≤ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1

theorem range_of_a (a : ℝ) : (p a ∧ q a) → -1 < a ∧ a ≤ -1 / 2 :=
by
  sorry

end range_of_a_l1869_186950


namespace min_gumballs_to_ensure_four_same_color_l1869_186901

/-- A structure to represent the number of gumballs of each color. -/
structure Gumballs :=
(red : ℕ)
(white : ℕ)
(blue : ℕ)
(green : ℕ)

def gumball_machine : Gumballs := { red := 10, white := 9, blue := 8, green := 6 }

/-- Theorem to state the minimum number of gumballs required to ensure at least four of any color. -/
theorem min_gumballs_to_ensure_four_same_color 
  (g : Gumballs) 
  (h1 : g.red = 10)
  (h2 : g.white = 9)
  (h3 : g.blue = 8)
  (h4 : g.green = 6) : 
  ∃ n, n = 13 := 
sorry

end min_gumballs_to_ensure_four_same_color_l1869_186901


namespace combined_percentage_basketball_l1869_186919

theorem combined_percentage_basketball (N_students : ℕ) (S_students : ℕ) 
  (N_percent_basketball : ℚ) (S_percent_basketball : ℚ) :
  N_students = 1800 → S_students = 3000 →
  N_percent_basketball = 0.25 → S_percent_basketball = 0.35 →
  ((N_students * N_percent_basketball) + (S_students * S_percent_basketball)) / (N_students + S_students) * 100 = 31 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  norm_num
  sorry

end combined_percentage_basketball_l1869_186919


namespace train_speed_initial_l1869_186969

variable (x : ℝ)
variable (v : ℝ)
variable (average_speed : ℝ := 40 / 3)
variable (initial_distance : ℝ := x)
variable (initial_speed : ℝ := v)
variable (next_distance : ℝ := 4 * x)
variable (next_speed : ℝ := 20)

theorem train_speed_initial : 
  (5 * x) / ((x / v) + (x / 5)) = 40 / 3 → v = 40 / 7 :=
by
  -- Definition of average speed in the context of the problem
  let t1 := x / v
  let t2 := (4 * x) / 20
  let total_distance := 5 * x
  let total_time := t1 + t2
  have avg_speed_eq : total_distance / total_time = 40 / 3 := by sorry
  sorry

end train_speed_initial_l1869_186969


namespace angle_C_exceeds_120_degrees_l1869_186926

theorem angle_C_exceeds_120_degrees 
  (a b : ℝ) (h_a : a = Real.sqrt 3) (h_b : b = Real.sqrt 3) (c : ℝ) (h_c : c > 3) :
  ∀ (C : ℝ), C = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) 
             → C > 120 :=
by
  sorry

end angle_C_exceeds_120_degrees_l1869_186926


namespace Tammy_average_speed_second_day_l1869_186946

theorem Tammy_average_speed_second_day : 
  ∀ (t v : ℝ), 
    (t + (t - 2) + (t + 1) = 20) → 
    (7 * v + 5 * (v + 0.5) + 8 * (v + 1.5) = 85) → 
    (v + 0.5 = 4.025) := 
by 
  intros t v ht hv 
  sorry

end Tammy_average_speed_second_day_l1869_186946


namespace a2b_etc_ge_9a2b2c2_l1869_186933

theorem a2b_etc_ge_9a2b2c2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 :=
by
  sorry

end a2b_etc_ge_9a2b2c2_l1869_186933


namespace line_tangent_to_parabola_l1869_186923

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4 * x + 7 * y + k = 0 ↔ y^2 = 16 * x) → k = 49 :=
by
  sorry

end line_tangent_to_parabola_l1869_186923


namespace problems_per_page_l1869_186909

def total_problems : ℕ := 60
def finished_problems : ℕ := 20
def remaining_pages : ℕ := 5

theorem problems_per_page :
  (total_problems - finished_problems) / remaining_pages = 8 :=
by
  sorry

end problems_per_page_l1869_186909


namespace students_with_one_problem_l1869_186920

theorem students_with_one_problem :
  ∃ (n_1 n_2 n_3 n_4 n_5 n_6 n_7 : ℕ) (k_1 k_2 k_3 k_4 k_5 k_6 k_7 : ℕ),
    (n_1 + n_2 + n_3 + n_4 + n_5 + n_6 + n_7 = 39) ∧
    (n_1 * k_1 + n_2 * k_2 + n_3 * k_3 + n_4 * k_4 + n_5 * k_5 + n_6 * k_6 + n_7 * k_7 = 60) ∧
    (k_1 ≠ 0) ∧ (k_2 ≠ 0) ∧ (k_3 ≠ 0) ∧ (k_4 ≠ 0) ∧ (k_5 ≠ 0) ∧ (k_6 ≠ 0) ∧ (k_7 ≠ 0) ∧
    (k_1 ≠ k_2) ∧ (k_1 ≠ k_3) ∧ (k_1 ≠ k_4) ∧ (k_1 ≠ k_5) ∧ (k_1 ≠ k_6) ∧ (k_1 ≠ k_7) ∧
    (k_2 ≠ k_3) ∧ (k_2 ≠ k_4) ∧ (k_2 ≠ k_5) ∧ (k_2 ≠ k_6) ∧ (k_2 ≠ k_7) ∧
    (k_3 ≠ k_4) ∧ (k_3 ≠ k_5) ∧ (k_3 ≠ k_6) ∧ (k_3 ≠ k_7) ∧
    (k_4 ≠ k_5) ∧ (k_4 ≠ k_6) ∧ (k_4 ≠ k_7) ∧
    (k_5 ≠ k_6) ∧ (k_5 ≠ k_7) ∧
    (k_6 ≠ k_7) ∧
    (n_1 = 33) :=
sorry

end students_with_one_problem_l1869_186920


namespace opposite_exprs_have_value_l1869_186985

theorem opposite_exprs_have_value (x : ℝ) : (4 * x - 8 = -(3 * x - 6)) → x = 2 :=
by
  intro h
  sorry

end opposite_exprs_have_value_l1869_186985


namespace polynomial_sum_of_coefficients_l1869_186983

theorem polynomial_sum_of_coefficients {v : ℕ → ℝ} (h1 : v 1 = 7)
  (h2 : ∀ n : ℕ, v (n + 1) - v n = 5 * n - 2) :
  ∃ (a b c : ℝ), (∀ n : ℕ, v n = a * n^2 + b * n + c) ∧ (a + b + c = 7) :=
by
  sorry

end polynomial_sum_of_coefficients_l1869_186983


namespace value_of_expression_l1869_186937

noncomputable def line_does_not_pass_through_third_quadrant (k b : ℝ) : Prop :=
k < 0 ∧ b ≥ 0

theorem value_of_expression 
  (k b a e m n c d : ℝ) 
  (h_line : line_does_not_pass_through_third_quadrant k b)
  (h_a_gt_e : a > e)
  (hA : a * k + b = m)
  (hB : e * k + b = n)
  (hC : -m * k + b = c)
  (hD : -n * k + b = d) :
  (m - n) * (c - d) ^ 3 > 0 :=
sorry

end value_of_expression_l1869_186937


namespace pauline_spent_in_all_l1869_186990

theorem pauline_spent_in_all
  (cost_taco_shells : ℝ := 5)
  (cost_bell_pepper : ℝ := 1.5)
  (num_bell_peppers : ℕ := 4)
  (cost_meat_per_pound : ℝ := 3)
  (num_pounds_meat : ℝ := 2) :
  (cost_taco_shells + num_bell_peppers * cost_bell_pepper + num_pounds_meat * cost_meat_per_pound = 17) :=
by
  sorry

end pauline_spent_in_all_l1869_186990


namespace solution_when_a_is_1_solution_for_arbitrary_a_l1869_186977

-- Let's define the inequality and the solution sets
def inequality (a x : ℝ) : Prop :=
  ((a + 1) * x - 3) / (x - 1) < 1

def solutionSet_a_eq_1 (x : ℝ) : Prop :=
  1 < x ∧ x < 2

def solutionSet_a_eq_0 (x : ℝ) : Prop :=
  1 < x
  
def solutionSet_a_lt_0 (a x : ℝ) : Prop :=
  x < (2 / a) ∨ 1 < x

def solutionSet_0_lt_a_lt_2 (a x : ℝ) : Prop :=
  1 < x ∧ x < (2 / a)

def solutionSet_a_eq_2 : Prop :=
  false

def solutionSet_a_gt_2 (a x : ℝ) : Prop :=
  (2 / a) < x ∧ x < 1

-- Prove the solution for a = 1
theorem solution_when_a_is_1 : ∀ (x : ℝ), inequality 1 x ↔ solutionSet_a_eq_1 x :=
by sorry

-- Prove the solution for arbitrary real number a
theorem solution_for_arbitrary_a : ∀ (a x : ℝ),
  (a < 0 → inequality a x ↔ solutionSet_a_lt_0 a x) ∧
  (a = 0 → inequality a x ↔ solutionSet_a_eq_0 x) ∧
  (0 < a ∧ a < 2 → inequality a x ↔ solutionSet_0_lt_a_lt_2 a x) ∧
  (a = 2 → inequality a x → solutionSet_a_eq_2) ∧
  (a > 2 → inequality a x ↔ solutionSet_a_gt_2 a x) :=
by sorry

end solution_when_a_is_1_solution_for_arbitrary_a_l1869_186977


namespace find_A_l1869_186993

theorem find_A (A B : ℝ) (h1 : B = 10 * A) (h2 : 211.5 = B - A) : A = 23.5 :=
by {
  sorry
}

end find_A_l1869_186993


namespace compute_expression_l1869_186911

theorem compute_expression : 65 * 1313 - 25 * 1313 = 52520 := by
  sorry

end compute_expression_l1869_186911


namespace find_m_range_l1869_186961

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | -1 ≤ x ∧ x ≤ m + 1 }

theorem find_m_range (m : ℝ) : (B m ⊆ A) ↔ (-2 ≤ m ∧ m ≤ 3) := by
  sorry

end find_m_range_l1869_186961


namespace number_of_complete_decks_l1869_186987

theorem number_of_complete_decks (total_cards : ℕ) (additional_cards : ℕ) (cards_per_deck : ℕ) 
(h1 : total_cards = 319) (h2 : additional_cards = 7) (h3 : cards_per_deck = 52) : 
total_cards - additional_cards = (cards_per_deck * 6) :=
by
  sorry

end number_of_complete_decks_l1869_186987


namespace proof_problem_l1869_186976

-- Define the system of equations
def system_of_equations (x y a : ℝ) : Prop :=
  (3 * x + y = 2 + 3 * a) ∧ (x + 3 * y = 2 + a)

-- Define the condition x + y < 0
def condition (x y : ℝ) : Prop := x + y < 0

-- Prove that if the system of equations has a solution with x + y < 0, then a < -1 and |1 - a| + |a + 1 / 2| = 1 / 2 - 2 * a
theorem proof_problem (x y a : ℝ) (h1 : system_of_equations x y a) (h2 : condition x y) :
  a < -1 ∧ |1 - a| + |a + 1 / 2| = (1 / 2) - 2 * a := 
sorry

end proof_problem_l1869_186976


namespace ramola_rank_last_is_14_l1869_186934

-- Define the total number of students
def total_students : ℕ := 26

-- Define Ramola's rank from the start
def ramola_rank_start : ℕ := 14

-- Define a function to calculate the rank from the last given the above conditions
def ramola_rank_from_last (total_students ramola_rank_start : ℕ) : ℕ :=
  total_students - ramola_rank_start + 1

-- Theorem stating that Ramola's rank from the last is 14th
theorem ramola_rank_last_is_14 :
  ramola_rank_from_last total_students ramola_rank_start = 14 :=
by
  -- Proof goes here
  sorry

end ramola_rank_last_is_14_l1869_186934


namespace three_friends_expenses_l1869_186978

theorem three_friends_expenses :
  let ticket_cost := 7
  let number_of_tickets := 3
  let popcorn_cost := 1.5
  let number_of_popcorn := 2
  let milk_tea_cost := 3
  let number_of_milk_tea := 3
  let total_expenses := (ticket_cost * number_of_tickets) + (popcorn_cost * number_of_popcorn) + (milk_tea_cost * number_of_milk_tea)
  let amount_per_friend := total_expenses / 3
  amount_per_friend = 11 := 
by
  sorry

end three_friends_expenses_l1869_186978


namespace xy_sum_l1869_186928

theorem xy_sum (x y : ℝ) (h1 : x^3 - 6 * x^2 + 15 * x = 12) (h2 : y^3 - 6 * y^2 + 15 * y = 16) : x + y = 4 := 
sorry

end xy_sum_l1869_186928


namespace amanda_average_speed_l1869_186951

def amanda_distance1 : ℝ := 450
def amanda_time1 : ℝ := 7.5
def amanda_distance2 : ℝ := 420
def amanda_time2 : ℝ := 7

def total_distance : ℝ := amanda_distance1 + amanda_distance2
def total_time : ℝ := amanda_time1 + amanda_time2
def expected_average_speed : ℝ := 60

theorem amanda_average_speed :
  (total_distance / total_time) = expected_average_speed := by
  sorry

end amanda_average_speed_l1869_186951


namespace sum_of_positive_factors_36_l1869_186964

def sum_of_divisors (n : Nat) : Nat :=
  (List.range (n + 1)).filter (fun m => n % m = 0) |>.sum

theorem sum_of_positive_factors_36 : sum_of_divisors 36 = 91 := by
  sorry

end sum_of_positive_factors_36_l1869_186964


namespace max_min_S_l1869_186925

theorem max_min_S (x y : ℝ) (h : (x - 1)^2 + (y + 2)^2 = 4) : 
  (∃ S_max S_min : ℝ, S_max = 4 + 2 * Real.sqrt 5 ∧ S_min = 4 - 2 * Real.sqrt 5 ∧ 
  (∀ S : ℝ, (∃ (x y : ℝ), (x - 1)^2 + (y + 2)^2 = 4 ∧ S = 2 * x + y) → S ≤ S_max ∧ S ≥ S_min)) :=
sorry

end max_min_S_l1869_186925


namespace binom_eight_three_l1869_186932

theorem binom_eight_three : Nat.choose 8 3 = 56 := by
  sorry

end binom_eight_three_l1869_186932


namespace gumballs_result_l1869_186943

def gumballs_after_sharing_equally (initial_joanna : ℕ) (initial_jacques : ℕ) (multiplier : ℕ) : ℕ :=
  let joanna_total := initial_joanna + initial_joanna * multiplier
  let jacques_total := initial_jacques + initial_jacques * multiplier
  (joanna_total + jacques_total) / 2

theorem gumballs_result :
  gumballs_after_sharing_equally 40 60 4 = 250 :=
by
  sorry

end gumballs_result_l1869_186943


namespace stream_speed_l1869_186906

-- Definitions based on conditions
def speed_in_still_water : ℝ := 5
def distance_downstream : ℝ := 100
def time_downstream : ℝ := 10

-- The required speed of the stream
def speed_of_stream (v : ℝ) : Prop :=
  distance_downstream = (speed_in_still_water + v) * time_downstream

-- Proof statement: the speed of the stream is 5 km/hr
theorem stream_speed : ∃ v, speed_of_stream v ∧ v = 5 := 
by
  use 5
  unfold speed_of_stream
  sorry

end stream_speed_l1869_186906


namespace cups_of_rice_in_afternoon_l1869_186930

-- Definitions for conditions
def morning_cups : ℕ := 3
def evening_cups : ℕ := 5
def fat_per_cup : ℕ := 10
def weekly_total_fat : ℕ := 700

-- Theorem statement
theorem cups_of_rice_in_afternoon (morning_cups evening_cups fat_per_cup weekly_total_fat : ℕ) :
  (weekly_total_fat - (morning_cups + evening_cups) * fat_per_cup * 7) / fat_per_cup = 14 :=
by
  sorry

end cups_of_rice_in_afternoon_l1869_186930


namespace candle_lighting_time_l1869_186949

theorem candle_lighting_time 
  (l : ℕ) -- initial length of the candles
  (t_diff : ℤ := 206) -- the time difference in minutes, correlating to 1:34 PM.
  : t_diff = 206 :=
by sorry

end candle_lighting_time_l1869_186949


namespace maximum_value_of_reciprocals_l1869_186989

theorem maximum_value_of_reciprocals (c b : ℝ) (h0 : 0 < b ∧ b < c)
  (e1 : ℝ) (e2 : ℝ)
  (h1 : e1 = c / (Real.sqrt (c^2 + (2 * b)^2)))
  (h2 : e2 = c / (Real.sqrt (c^2 - b^2)))
  (h3 : 1 / e1^2 + 4 / e2^2 = 5) :
  ∃ max_val, max_val = 5 / 2 :=
by
  sorry

end maximum_value_of_reciprocals_l1869_186989


namespace route_speeds_l1869_186945

theorem route_speeds (x : ℝ) (hx : x > 0) :
  (25 / x) - (21 / (1.4 * x)) = (20 / 60) := by
  sorry

end route_speeds_l1869_186945


namespace fill_in_blank_with_warning_l1869_186967

-- Definitions corresponding to conditions
def is_noun (word : String) : Prop :=
  -- definition of being a noun
  sorry

def corresponds_to_chinese_hint (word : String) (hint : String) : Prop :=
  -- definition of corresponding to a Chinese hint
  sorry

-- The theorem we want to prove
theorem fill_in_blank_with_warning : ∀ word : String, 
  (is_noun word ∧ corresponds_to_chinese_hint word "警告") → word = "warning" :=
by {
  sorry
}

end fill_in_blank_with_warning_l1869_186967


namespace polynomial_A_l1869_186913

variables {a b : ℝ} (A : ℝ)
variables (h1 : 2 ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0)

theorem polynomial_A (h : A / (2 * a * b) = 1 - 4 * a ^ 2) : 
  A = 2 * a * b - 8 * a ^ 3 * b :=
by
  sorry

end polynomial_A_l1869_186913


namespace clock_ticks_six_times_l1869_186962

-- Define the conditions
def time_between_ticks (ticks : Nat) : Nat :=
  ticks - 1

def interval_duration (total_time : Nat) (ticks : Nat) : Nat :=
  total_time / time_between_ticks ticks

def number_of_ticks (total_time : Nat) (interval_time : Nat) : Nat :=
  total_time / interval_time + 1

-- Given conditions
def specific_time_intervals : Nat := 30
def eight_oclock_intervals : Nat := 42

-- Proven result
theorem clock_ticks_six_times : number_of_ticks specific_time_intervals (interval_duration eight_oclock_intervals 8) = 6 := 
sorry

end clock_ticks_six_times_l1869_186962


namespace must_be_odd_l1869_186998

theorem must_be_odd (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) :=
sorry

end must_be_odd_l1869_186998


namespace kanul_total_amount_l1869_186958

theorem kanul_total_amount (T : ℝ) (h1 : 35000 + 40000 + 0.2 * T = T) : T = 93750 := 
by
  sorry

end kanul_total_amount_l1869_186958


namespace max_x_y3_z4_l1869_186974

noncomputable def max_value_expression (x y z : ℝ) : ℝ :=
  x + y^3 + z^4

theorem max_x_y3_z4 (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 1) :
  max_value_expression x y z ≤ 1 :=
sorry

end max_x_y3_z4_l1869_186974


namespace product_of_x_and_y_l1869_186955

theorem product_of_x_and_y :
  ∀ (x y : ℝ), (∀ p : ℝ × ℝ, (p = (x, 6) ∨ p = (10, y)) → p.2 = (1 / 2) * p.1) → x * y = 60 :=
by
  intros x y h
  have hx : 6 = (1 / 2) * x := by exact h (x, 6) (Or.inl rfl)
  have hy : y = (1 / 2) * 10 := by exact h (10, y) (Or.inr rfl)
  sorry

end product_of_x_and_y_l1869_186955


namespace find_k_perpendicular_l1869_186929

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, 1)
def vec_b : ℝ × ℝ := (2, -3)

-- Define a function for the vector k * a - 2 * b
def vec_expression (k : ℝ) : ℝ × ℝ :=
  (k * vec_a.1 - 2 * vec_b.1, k * vec_a.2 - 2 * vec_b.2)

-- Prove that if the dot product of vec_expression k and vec_a is zero, then k = -1
theorem find_k_perpendicular (k : ℝ) :
  ((vec_expression k).1 * vec_a.1 + (vec_expression k).2 * vec_a.2 = 0) → k = -1 :=
by
  sorry

end find_k_perpendicular_l1869_186929


namespace bicycle_distance_l1869_186935

theorem bicycle_distance (P_b P_f : ℝ) (h1 : P_b = 9) (h2 : P_f = 7) (h3 : ∀ D : ℝ, D / P_f = D / P_b + 10) :
  315 = 315 :=
by
  sorry

end bicycle_distance_l1869_186935


namespace coefficient_of_x_is_nine_l1869_186968

theorem coefficient_of_x_is_nine (x : ℝ) (c : ℝ) (h : x = 0.5) (eq : 2 * x^2 + c * x - 5 = 0) : c = 9 :=
by
  sorry

end coefficient_of_x_is_nine_l1869_186968


namespace common_diff_necessary_sufficient_l1869_186921

section ArithmeticSequence

variable {α : Type*} [OrderedAddCommGroup α] {a : ℕ → α} {d : α}

-- Define an arithmetic sequence with common difference d
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Prove that d > 0 is the necessary and sufficient condition for a_2 > a_1
theorem common_diff_necessary_sufficient (a : ℕ → α) (d : α) :
    (is_arithmetic_sequence a d) → (d > 0 ↔ a 2 > a 1) :=
by
  sorry

end ArithmeticSequence

end common_diff_necessary_sufficient_l1869_186921


namespace new_stamps_ratio_l1869_186917

theorem new_stamps_ratio (x : ℕ) (h1 : 7 * x = P) (h2 : 4 * x = Q)
  (h3 : P - 8 = 8 + (Q + 8)) : (P - 8) / gcd (P - 8) (Q + 8) = 6 ∧ (Q + 8) / gcd (P - 8) (Q + 8) = 5 :=
by
  sorry

end new_stamps_ratio_l1869_186917


namespace perimeter_of_regular_nonagon_l1869_186959

def regular_nonagon_side_length := 3
def number_of_sides := 9

theorem perimeter_of_regular_nonagon (h1 : number_of_sides = 9) (h2 : regular_nonagon_side_length = 3) :
  9 * 3 = 27 :=
by
  sorry

end perimeter_of_regular_nonagon_l1869_186959


namespace prize_selection_count_l1869_186995

theorem prize_selection_count :
  (Nat.choose 20 1) * (Nat.choose 19 2) * (Nat.choose 17 4) = 8145600 := 
by 
  sorry

end prize_selection_count_l1869_186995


namespace least_distinct_values_l1869_186980

theorem least_distinct_values (lst : List ℕ) (h_len : lst.length = 2023) (h_mode : ∃ m, (∀ n ≠ m, lst.count n < lst.count m) ∧ lst.count m = 13) : ∃ x, x = 169 :=
by
  sorry

end least_distinct_values_l1869_186980


namespace solve_problem_l1869_186996

noncomputable def problem_statement : Prop :=
  ∀ (T0 Ta T t1 T1 h t2 T2 : ℝ),
    T0 = 88 ∧ Ta = 24 ∧ T1 = 40 ∧ t1 = 20 ∧
    T1 - Ta = (T0 - Ta) * ((1/2)^(t1/h)) ∧
    T2 = 32 ∧ T2 - Ta = (T1 - Ta) * ((1/2)^(t2/h)) →
    t2 = 10

theorem solve_problem : problem_statement := sorry

end solve_problem_l1869_186996


namespace complex_multiplication_l1869_186960

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : (2 + i) * (1 - 3 * i) = 5 - 5 * i := 
by
  sorry

end complex_multiplication_l1869_186960


namespace correct_operation_l1869_186904

theorem correct_operation : -5 * 3 = -15 :=
by sorry

end correct_operation_l1869_186904


namespace shortest_chord_length_l1869_186907

/-- The shortest chord passing through point D given the conditions provided. -/
theorem shortest_chord_length
  (O : Point) (D : Point) (r : ℝ) (OD : ℝ)
  (h_or : r = 5) (h_od : OD = 3) :
  ∃ (AB : ℝ), AB = 8 := 
  sorry

end shortest_chord_length_l1869_186907


namespace subscription_total_l1869_186903

theorem subscription_total (a b c : ℝ) (h1 : a = b + 4000) (h2 : b = c + 5000) (h3 : 15120 / 36000 = a / (a + b + c)) : 
  a + b + c = 50000 :=
by 
  sorry

end subscription_total_l1869_186903


namespace range_a_ge_one_l1869_186910

theorem range_a_ge_one (a : ℝ) (x : ℝ) 
  (p : Prop := |x + 1| > 2) 
  (q : Prop := x > a) 
  (suff_not_necess_cond : ¬p → ¬q) : a ≥ 1 :=
sorry

end range_a_ge_one_l1869_186910


namespace marcy_total_spears_l1869_186956

-- Define the conditions
def can_make_spears_from_sapling (spears_per_sapling : ℕ) (saplings : ℕ) : ℕ :=
  spears_per_sapling * saplings

def can_make_spears_from_log (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  spears_per_log * logs

-- Number of spears Marcy can make from 6 saplings and 1 log
def total_spears (spears_per_sapling : ℕ) (saplings : ℕ) (spears_per_log : ℕ) (logs : ℕ) : ℕ :=
  can_make_spears_from_sapling spears_per_sapling saplings + can_make_spears_from_log spears_per_log logs

-- Given conditions
theorem marcy_total_spears (saplings : ℕ) (logs : ℕ) : 
  total_spears 3 6 9 1 = 27 :=
by
  sorry

end marcy_total_spears_l1869_186956


namespace variance_stability_l1869_186979

theorem variance_stability (S2_A S2_B : ℝ) (hA : S2_A = 1.1) (hB : S2_B = 2.5) : ¬(S2_B < S2_A) :=
by {
  sorry
}

end variance_stability_l1869_186979


namespace range_for_a_l1869_186982

def f (a : ℝ) (x : ℝ) := 2 * x^3 - a * x^2 + 1

def two_zeros_in_interval (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (1/2 ≤ x1 ∧ x1 ≤ 2) ∧ (1/2 ≤ x2 ∧ x2 ≤ 2) ∧ x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0

theorem range_for_a {a : ℝ} : (3/2 : ℝ) < a ∧ a ≤ (17/4 : ℝ) ↔ two_zeros_in_interval a :=
by sorry

end range_for_a_l1869_186982


namespace f_one_eq_minus_one_third_f_of_a_f_is_odd_l1869_186908

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (2^x + 1)

theorem f_one_eq_minus_one_third : f 1 = -1/3 := 
by sorry

theorem f_of_a (a : ℝ) : f a = (1 - 2^a) / (2^a + 1) := 
by sorry

theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

end f_one_eq_minus_one_third_f_of_a_f_is_odd_l1869_186908


namespace exists_circular_chain_of_four_l1869_186918

-- Let A and B be the two teams, each with a set of players.
variable {A B : Type}
-- Assume there exists a relation "beats" that determines match outcomes.
variable (beats : A → B → Prop)

-- Each player in both teams has at least one win and one loss against the opposite team.
axiom each_has_win_and_loss (a : A) : ∃ b1 b2 : B, beats a b1 ∧ ¬beats a b2 ∧ b1 ≠ b2
axiom each_has_win_and_loss' (b : B) : ∃ a1 a2 : A, beats a1 b ∧ ¬beats a2 b ∧ a1 ≠ a2

-- Main theorem: Exist four players forming a circular chain of victories.
theorem exists_circular_chain_of_four :
  ∃ (a1 a2 : A) (b1 b2 : B), beats a1 b1 ∧ ¬beats a1 b2 ∧ beats a2 b2 ∧ ¬beats a2 b1 ∧ b1 ≠ b2 ∧ a1 ≠ a2 :=
sorry

end exists_circular_chain_of_four_l1869_186918
