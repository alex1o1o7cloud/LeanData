import Mathlib

namespace negation_prop_l129_12970

theorem negation_prop (x : ℝ) : (¬ (∀ x : ℝ, Real.exp x > x^2)) ↔ (∃ x : ℝ, Real.exp x ≤ x^2) :=
by
  sorry

end negation_prop_l129_12970


namespace omitted_decimal_sum_is_integer_l129_12981

def numbers : List ℝ := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

theorem omitted_decimal_sum_is_integer :
  1.05 + 1.15 + 1.25 + 1.4 + (15 : ℝ) + 1.6 + 1.75 + 1.85 + 1.95 = 27 :=
by sorry

end omitted_decimal_sum_is_integer_l129_12981


namespace find_equation_l129_12987

theorem find_equation (x : ℝ) : 
  (3 + x < 1 → false) ∧
  ((x - 67 + 63 = x - 4) → false) ∧
  ((4.8 + x = x + 4.8) → false) ∧
  (x + 0.7 = 12 → true) := 
sorry

end find_equation_l129_12987


namespace binary_addition_to_decimal_l129_12917

theorem binary_addition_to_decimal : (0b111111111 + 0b1000001 = 576) :=
by {
  sorry
}

end binary_addition_to_decimal_l129_12917


namespace kim_trip_time_l129_12927

-- Definitions
def distance_freeway : ℝ := 120
def distance_mountain : ℝ := 25
def speed_ratio : ℝ := 4
def time_mountain : ℝ := 75

-- The problem statement
theorem kim_trip_time : ∃ t_freeway t_total : ℝ,
  t_freeway = distance_freeway / (speed_ratio * (distance_mountain / time_mountain)) ∧
  t_total = time_mountain + t_freeway ∧
  t_total = 165 := by
  sorry

end kim_trip_time_l129_12927


namespace cone_height_l129_12902

theorem cone_height (V : ℝ) (π : ℝ) (r h : ℝ) (sqrt2 : ℝ) :
  V = 9720 * π →
  sqrt2 = Real.sqrt 2 →
  h = r * sqrt2 →
  V = (1/3) * π * r^2 * h →
  h = 38.7 :=
by
  intros
  sorry

end cone_height_l129_12902


namespace no_natural_numbers_for_squares_l129_12976

theorem no_natural_numbers_for_squares :
  ∀ x y : ℕ, ¬(∃ k m : ℕ, k^2 = x^2 + y ∧ m^2 = y^2 + x) :=
by sorry

end no_natural_numbers_for_squares_l129_12976


namespace tall_wins_min_voters_l129_12971

structure VotingSetup where
  total_voters : ℕ
  districts : ℕ
  sections_per_district : ℕ
  voters_per_section : ℕ
  voters_majority_in_section : ℕ
  districts_to_win : ℕ
  sections_to_win_district : ℕ

def contest_victory (setup : VotingSetup) (min_voters : ℕ) : Prop :=
  setup.total_voters = 105 ∧
  setup.districts = 5 ∧
  setup.sections_per_district = 7 ∧
  setup.voters_per_section = 3 ∧
  setup.voters_majority_in_section = 2 ∧
  setup.districts_to_win = 3 ∧
  setup.sections_to_win_district = 4 ∧
  min_voters = 24

theorem tall_wins_min_voters : ∃ min_voters, contest_victory ⟨105, 5, 7, 3, 2, 3, 4⟩ min_voters :=
by { use 24, sorry }

end tall_wins_min_voters_l129_12971


namespace buckets_required_l129_12906

theorem buckets_required (C : ℚ) (N : ℕ) (h : 250 * (4/5 : ℚ) * C = N * C) : N = 200 :=
by
  sorry

end buckets_required_l129_12906


namespace rebecca_has_more_eggs_than_marbles_l129_12969

-- Given conditions
def eggs : Int := 20
def marbles : Int := 6

-- Mathematically equivalent statement to prove
theorem rebecca_has_more_eggs_than_marbles :
    eggs - marbles = 14 :=
by
    sorry

end rebecca_has_more_eggs_than_marbles_l129_12969


namespace verify_solution_l129_12958

variable (x y : ℝ)

-- Conditions
def condition1 : Prop := x - y = 9
def condition2 : Prop := 4 * x + 3 * y = 1

-- Proof problem statement
theorem verify_solution
  (h1 : condition1 x y)
  (h2 : condition2 x y) :
  x = 4 ∧ y = -5 :=
sorry

end verify_solution_l129_12958


namespace teacher_arrangements_l129_12912

theorem teacher_arrangements (T : Fin 30 → ℕ) (h1 : T 1 < T 2 ∧ T 2 < T 3 ∧ T 3 < T 4 ∧ T 4 < T 5)
  (h2 : ∀ i : Fin 4, T (i + 1) ≥ T i + 3)
  (h3 : 1 ≤ T 1)
  (h4 : T 5 ≤ 26) :
  ∃ n : ℕ, n = 26334 := by
  sorry

end teacher_arrangements_l129_12912


namespace arithmetic_sequence_proof_l129_12956

variable (n : ℕ)
variable (a_n S_n : ℕ → ℤ)

noncomputable def a : ℕ → ℤ := 48 - 8 * n
noncomputable def S : ℕ → ℤ := -4 * (n ^ 2) + 44 * n

axiom a_3 : a 3 = 24
axiom S_11 : S 11 = 0

theorem arithmetic_sequence_proof :
  a n = 48 - 8 * n ∧
  S n = -4 * n ^ 2 + 44 * n ∧
  ∃ n, S n = 120 ∧ (n = 5 ∨ n = 6) :=
by
  unfold a S
  sorry

end arithmetic_sequence_proof_l129_12956


namespace find_x_l129_12922

theorem find_x (x : ℝ) (h : x + 5 * 12 / (180 / 3) = 41) : x = 40 :=
sorry

end find_x_l129_12922


namespace cannot_be_n_plus_2_l129_12955

theorem cannot_be_n_plus_2 (n : ℕ) : 
  ¬(∃ Y, (Y = n + 2) ∧ 
         ((Y = n - 3) ∨ (Y = n - 1) ∨ (Y = n + 5))) := 
by {
  sorry
}

end cannot_be_n_plus_2_l129_12955


namespace inequality_represents_area_l129_12973

theorem inequality_represents_area (a : ℝ) :
  (if a > 1 then ∀ (x y : ℝ), x + (a - 1) * y + 3 > 0 ↔ y < - (x + 3) / (a - 1)
  else ∀ (x y : ℝ), x + (a - 1) * y + 3 > 0 ↔ y > - (x + 3) / (a - 1)) :=
by sorry

end inequality_represents_area_l129_12973


namespace c_share_l129_12934

theorem c_share (A B C : ℕ) 
    (h1 : A = 1/2 * B) 
    (h2 : B = 1/2 * C) 
    (h3 : A + B + C = 406) : 
    C = 232 := by 
    sorry

end c_share_l129_12934


namespace final_selling_price_l129_12967

-- Define the conditions in Lean
def cost_price_A : ℝ := 150
def profit_A_rate : ℝ := 0.20
def profit_B_rate : ℝ := 0.25

-- Define the function to calculate selling price based on cost price and profit rate
def selling_price (cost_price : ℝ) (profit_rate : ℝ) : ℝ :=
  cost_price + (profit_rate * cost_price)

-- The theorem to be proved
theorem final_selling_price :
  selling_price (selling_price cost_price_A profit_A_rate) profit_B_rate = 225 :=
by
  -- The proof is omitted
  sorry

end final_selling_price_l129_12967


namespace solve_coffee_problem_l129_12979

variables (initial_stock new_purchase : ℕ)
           (initial_decaf_percentage new_decaf_percentage : ℚ)
           (total_stock total_decaf weight_percentage_decaf : ℚ)

def coffee_problem :=
  initial_stock = 400 ∧
  initial_decaf_percentage = 0.20 ∧
  new_purchase = 100 ∧
  new_decaf_percentage = 0.50 ∧
  total_stock = initial_stock + new_purchase ∧
  total_decaf = initial_stock * initial_decaf_percentage + new_purchase * new_decaf_percentage ∧
  weight_percentage_decaf = (total_decaf / total_stock) * 100

theorem solve_coffee_problem : coffee_problem 400 100 0.20 0.50 500 130 26 :=
by {
  sorry
}

end solve_coffee_problem_l129_12979


namespace smallest_n_l129_12931

-- Define the conditions as predicates
def condition1 (n : ℕ) : Prop := (n + 2018) % 2020 = 0
def condition2 (n : ℕ) : Prop := (n + 2020) % 2018 = 0

-- The main theorem statement using these conditions
theorem smallest_n (n : ℕ) : 
  (∃ n, condition1 n ∧ condition2 n ∧ (∀ m, condition1 m ∧ condition2 m → n ≤ m)) ↔ n = 2030102 := 
by 
    sorry

end smallest_n_l129_12931


namespace total_animals_seen_l129_12911

theorem total_animals_seen (lions_sat : ℕ) (elephants_sat : ℕ) 
                           (buffaloes_sun : ℕ) (leopards_sun : ℕ)
                           (rhinos_mon : ℕ) (warthogs_mon : ℕ) 
                           (h_sat : lions_sat = 3 ∧ elephants_sat = 2)
                           (h_sun : buffaloes_sun = 2 ∧ leopards_sun = 5)
                           (h_mon : rhinos_mon = 5 ∧ warthogs_mon = 3) :
  lions_sat + elephants_sat + buffaloes_sun + leopards_sun + rhinos_mon + warthogs_mon = 20 := by
  sorry

end total_animals_seen_l129_12911


namespace magic_triangle_largest_S_l129_12907

theorem magic_triangle_largest_S :
  ∃ (S : ℕ) (a b c d e f g : ℕ),
    (10 ≤ a) ∧ (a ≤ 16) ∧
    (10 ≤ b) ∧ (b ≤ 16) ∧
    (10 ≤ c) ∧ (c ≤ 16) ∧
    (10 ≤ d) ∧ (d ≤ 16) ∧
    (10 ≤ e) ∧ (e ≤ 16) ∧
    (10 ≤ f) ∧ (f ≤ 16) ∧
    (10 ≤ g) ∧ (g ≤ 16) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧
    (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧
    (e ≠ f) ∧ (e ≠ g) ∧
    (f ≠ g) ∧
    (S = a + b + c) ∧
    (S = c + d + e) ∧
    (S = e + f + a) ∧
    (S = g + b + c) ∧
    (S = g + d + e) ∧
    (S = g + f + a) ∧
    ((a + b + c) + (c + d + e) + (e + f + a) = 91 - g) ∧
    (S = 26) := sorry

end magic_triangle_largest_S_l129_12907


namespace correct_inequality_l129_12974

theorem correct_inequality (a b c d : ℝ)
    (hab : a > b) (hb0 : b > 0)
    (hcd : c > d) (hd0 : d > 0) :
    Real.sqrt (a / d) > Real.sqrt (b / c) :=
by
    sorry

end correct_inequality_l129_12974


namespace rhombus_diagonal_length_l129_12962

theorem rhombus_diagonal_length (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 80) (h2 : area = 2480) (h3 : area = (d1 * d2) / 2) : d1 = 62 :=
by sorry

end rhombus_diagonal_length_l129_12962


namespace work_duration_l129_12913

theorem work_duration (p q r : ℕ) (Wp Wq Wr : ℕ) (t1 t2 : ℕ) (T : ℝ) :
  (Wp = 20) → (Wq = 12) → (Wr = 30) →
  (t1 = 4) → (t2 = 4) →
  (T = (t1 + t2 + (4/15 * Wr) / (1/(Wr) + 1/(Wq) + 1/(Wp)))) →
  T = 9.6 :=
by
  intros;
  sorry

end work_duration_l129_12913


namespace two_rel_prime_exists_l129_12932

theorem two_rel_prime_exists (A : Finset ℕ) (h1 : A.card = 2011) (h2 : ∀ x ∈ A, 1 ≤ x ∧ x ≤ 4020) : 
  ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ Nat.gcd a b = 1 :=
by
  sorry

end two_rel_prime_exists_l129_12932


namespace steve_final_height_l129_12961

-- Define the initial height and growth in inches
def initial_height_feet := 5
def initial_height_inches := 6
def growth_inches := 6

-- Define the conversion factors and total height after growth
def feet_to_inches (feet: Nat) := feet * 12

theorem steve_final_height : feet_to_inches initial_height_feet + initial_height_inches + growth_inches = 72 := by
  sorry

end steve_final_height_l129_12961


namespace petya_vasya_same_sum_l129_12953

theorem petya_vasya_same_sum :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 2^99 * (2^100 - 1) :=
by
  sorry

end petya_vasya_same_sum_l129_12953


namespace prove_m_eq_n_l129_12963

variable (m n : ℕ)

noncomputable def p := m + n + 1

theorem prove_m_eq_n 
  (is_prime : Prime p) 
  (divides : p ∣ 2 * (m^2 + n^2) - 1) : 
  m = n :=
by
  sorry

end prove_m_eq_n_l129_12963


namespace smaller_number_l129_12957

theorem smaller_number (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4851) : min a b = 53 :=
sorry

end smaller_number_l129_12957


namespace sin_double_angle_l129_12998

open Real

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioc (π / 2) π) (h2 : sin α = 4 / 5) :
  sin (2 * α) = -24 / 25 :=
by
  sorry

end sin_double_angle_l129_12998


namespace monthly_earnings_l129_12923

variable (e : ℕ) (s : ℕ) (p : ℕ) (t : ℕ)

-- conditions
def half_monthly_savings := s = e / 2
def car_price := p = 16000
def saving_months := t = 8
def total_saving := s * t = p

theorem monthly_earnings : ∀ (e s p t : ℕ), 
  half_monthly_savings e s → 
  car_price p → 
  saving_months t → 
  total_saving s t p → 
  e = 4000 :=
by
  intros e s p t h1 h2 h3 h4
  sorry

end monthly_earnings_l129_12923


namespace average_after_discard_l129_12918

theorem average_after_discard (sum_50 : ℝ) (avg_50 : sum_50 = 2200) (a b : ℝ) (h1 : a = 45) (h2 : b = 55) :
  (sum_50 - (a + b)) / 48 = 43.75 :=
by
  -- Given conditions: sum_50 = 2200, a = 45, b = 55
  -- We need to prove (sum_50 - (a + b)) / 48 = 43.75
  sorry

end average_after_discard_l129_12918


namespace smallest_solution_proof_l129_12929

noncomputable def smallest_solution : ℝ :=
  let n := 11
  let a := 0.533
  n + a

theorem smallest_solution_proof :
  ∃ (x : ℝ), ⌊x^2⌋ - ⌊x⌋^2 = 21 ∧ x = smallest_solution :=
by
  use smallest_solution
  sorry

end smallest_solution_proof_l129_12929


namespace sum_of_numbers_l129_12928

theorem sum_of_numbers (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 10) (h3 : x^2 - y^2 = 50) : x + y = 5 :=
by
  sorry

end sum_of_numbers_l129_12928


namespace larger_number_is_2997_l129_12980

theorem larger_number_is_2997 (L S : ℕ) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 := 
by
  sorry

end larger_number_is_2997_l129_12980


namespace mean_greater_than_median_l129_12919

theorem mean_greater_than_median (x : ℕ) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5 
  let median := x + 4 
  mean - median = 4 :=
by 
  sorry

end mean_greater_than_median_l129_12919


namespace joanne_main_job_hours_l129_12910

theorem joanne_main_job_hours (h : ℕ) (earn_main_job : ℝ) (earn_part_time : ℝ) (hours_part_time : ℕ) (days_week : ℕ) (total_weekly_earn : ℝ) :
  earn_main_job = 16.00 →
  earn_part_time = 13.50 →
  hours_part_time = 2 →
  days_week = 5 →
  total_weekly_earn = 775 →
  days_week * earn_main_job * h + days_week * earn_part_time * hours_part_time = total_weekly_earn →
  h = 8 :=
by
  sorry

end joanne_main_job_hours_l129_12910


namespace find_b_l129_12939

theorem find_b (b : ℤ) :
  ∃ (r₁ r₂ : ℤ), (r₁ = -9) ∧ (r₁ * r₂ = 36) ∧ (r₁ + r₂ = -b) → b = 13 :=
by {
  sorry
}

end find_b_l129_12939


namespace boat_speed_of_stream_l129_12988

theorem boat_speed_of_stream :
  ∀ (x : ℝ), 
    (∀ s_b : ℝ, s_b = 18) → 
    (∀ d1 d2 : ℝ, d1 = 48 → d2 = 32 → d1 / (18 + x) = d2 / (18 - x)) → 
    x = 3.6 :=
by 
  intros x h_speed h_distance
  sorry

end boat_speed_of_stream_l129_12988


namespace range_of_a_l129_12916

-- Given definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2 * a * x + 1 

-- Monotonicity condition on the interval [1, 2]
def is_monotonic (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x ≤ 2 → 1 ≤ y → y ≤ 2 → (x ≤ y → f x a ≤ f y a) ∨ (x ≤ y → f x a ≥ f y a)

-- The proof objective
theorem range_of_a (a : ℝ) : is_monotonic a → (a ≤ -2 ∨ a ≥ -1) := 
sorry

end range_of_a_l129_12916


namespace evaluate_fg_l129_12901

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x - 5

theorem evaluate_fg : f (g 4) = 9 := by
  sorry

end evaluate_fg_l129_12901


namespace solve_for_x_l129_12942

theorem solve_for_x (x : ℝ) (h : 1 - 1 / (1 - x) ^ 3 = 1 / (1 - x)) : x = 1 :=
sorry

end solve_for_x_l129_12942


namespace proposition_D_l129_12997

theorem proposition_D (a b c d : ℝ) (h1 : a < b) (h2 : c < d) : a + c < b + d :=
sorry

end proposition_D_l129_12997


namespace frequency_even_numbers_facing_up_l129_12909

theorem frequency_even_numbers_facing_up (rolls : ℕ) (event_occurrences : ℕ) (h_rolls : rolls = 100) (h_event : event_occurrences = 47) : (event_occurrences / (rolls : ℝ)) = 0.47 :=
by
  sorry

end frequency_even_numbers_facing_up_l129_12909


namespace sum_of_largest_three_l129_12925

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l129_12925


namespace steel_scrap_problem_l129_12960

theorem steel_scrap_problem 
  (x y : ℝ)
  (h1 : x + y = 140)
  (h2 : 0.05 * x + 0.40 * y = 42) :
  x = 40 ∧ y = 100 :=
by
  -- Solution steps are not required here
  sorry

end steel_scrap_problem_l129_12960


namespace problem_proof_l129_12908

-- Definitions based on conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

-- Theorem to prove
theorem problem_proof (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end problem_proof_l129_12908


namespace incorrect_weight_estimation_l129_12992

variables (x y : ℝ)

/-- Conditions -/
def regression_equation (x : ℝ) : ℝ := 0.85 * x - 85.71

/-- Incorrect conclusion -/
theorem incorrect_weight_estimation : regression_equation 160 ≠ 50.29 :=
by 
  sorry

end incorrect_weight_estimation_l129_12992


namespace find_f_at_6_5_l129_12977

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom functional_equation (x : ℝ) : f (x + 2) = - (1 / f x)
axiom initial_condition (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) : f x = x - 2

theorem find_f_at_6_5 : f 6.5 = -0.5 := by
  sorry

end find_f_at_6_5_l129_12977


namespace no_base_satisfies_l129_12930

def e : ℕ := 35

theorem no_base_satisfies :
  ∀ (base : ℝ), (1 / 5)^e * (1 / 4)^18 ≠ 1 / 2 * (base)^35 :=
by
  sorry

end no_base_satisfies_l129_12930


namespace ratio_perimeter_pentagon_to_square_l129_12995

theorem ratio_perimeter_pentagon_to_square
  (a : ℝ) -- Let a be the length of each side of the square
  (T_perimeter S_perimeter : ℝ) 
  (h1 : T_perimeter = S_perimeter) -- Given the perimeter of the triangle equals the perimeter of the square
  (h2 : S_perimeter = 4 * a) -- Given the perimeter of the square is 4 times the length of its side
  (P_perimeter : ℝ)
  (h3 : P_perimeter = (T_perimeter + S_perimeter) - 2 * a) -- Perimeter of the pentagon considering shared edge
  :
  P_perimeter / S_perimeter = 3 / 2 := 
sorry

end ratio_perimeter_pentagon_to_square_l129_12995


namespace find_f1_l129_12900

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f1
  (h1 : ∀ x : ℝ, |f x - x^2| ≤ 1/4)
  (h2 : ∀ x : ℝ, |f x + 1 - x^2| ≤ 3/4) :
  f 1 = 3/4 := 
sorry

end find_f1_l129_12900


namespace largest_tile_size_l129_12996

def length_cm : ℕ := 378
def width_cm : ℕ := 525

theorem largest_tile_size :
  Nat.gcd length_cm width_cm = 21 := by
  sorry

end largest_tile_size_l129_12996


namespace base5_division_l129_12903

theorem base5_division :
  ∀ (a b : ℕ), a = 1121 ∧ b = 12 → 
   ∃ (q r : ℕ), (a = b * q + r) ∧ (r < b) ∧ (q = 43) :=
by sorry

end base5_division_l129_12903


namespace find_2xy2_l129_12985

theorem find_2xy2 (x y : ℤ) (h : y^2 + 2 * x^2 * y^2 = 20 * x^2 + 412) : 2 * x * y^2 = 288 :=
sorry

end find_2xy2_l129_12985


namespace positive_difference_is_329_l129_12965

-- Definitions of the fractions involved
def fraction1 : ℚ := (7^2 + 7^2) / 7
def fraction2 : ℚ := (7^2 * 7^2) / 7

-- Statement of the positive difference proof
theorem positive_difference_is_329 : abs (fraction2 - fraction1) = 329 := by
  -- Skipping the proof here
  sorry

end positive_difference_is_329_l129_12965


namespace total_triangles_in_grid_l129_12948

-- Conditions
def bottom_row_triangles : Nat := 3
def next_row_triangles : Nat := 2
def top_row_triangles : Nat := 1
def additional_triangle : Nat := 1

def small_triangles := bottom_row_triangles + next_row_triangles + top_row_triangles + additional_triangle

-- Combining the triangles into larger triangles
def larger_triangles := 1 -- Formed by combining 4 small triangles
def largest_triangle := 1 -- Formed by combining all 7 small triangles

-- Math proof problem
theorem total_triangles_in_grid : small_triangles + larger_triangles + largest_triangle = 9 :=
by
  sorry

end total_triangles_in_grid_l129_12948


namespace inequality_proof_l129_12945

variable {a b c : ℝ}

theorem inequality_proof (h : a > b) : (a / (c^2 + 1)) > (b / (c^2 + 1)) := by
  sorry

end inequality_proof_l129_12945


namespace algebraic_expression_evaluation_l129_12975

-- Given condition and goal statement
theorem algebraic_expression_evaluation (a b : ℝ) (h : a - 2 * b + 3 = 0) : 5 + 2 * b - a = 8 :=
by sorry

end algebraic_expression_evaluation_l129_12975


namespace min_seats_occupied_l129_12943

theorem min_seats_occupied (n : ℕ) (h : n = 150) : ∃ k : ℕ, k = 37 ∧ ∀ m : ℕ, m > k → ∃ i : ℕ, i < k ∧ m - k ≥ 2 := sorry

end min_seats_occupied_l129_12943


namespace votes_cast_l129_12989

theorem votes_cast (V : ℝ) (hv1 : 0.35 * V + (0.35 * V + 1800) = V) : V = 6000 :=
sorry

end votes_cast_l129_12989


namespace union_M_N_l129_12972

def M : Set ℕ := {1, 2}
def N : Set ℕ := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by
  sorry

end union_M_N_l129_12972


namespace cannot_be_sum_of_six_consecutive_odd_integers_l129_12920

theorem cannot_be_sum_of_six_consecutive_odd_integers (S : ℕ) :
  (S = 90 ∨ S = 150) ->
  ∀ n : ℤ, ¬(S = n + (n+2) + (n+4) + (n+6) + (n+8) + (n+10)) :=
by
  intro h
  intro n
  cases h
  case inl => 
    sorry
  case inr => 
    sorry

end cannot_be_sum_of_six_consecutive_odd_integers_l129_12920


namespace average_speed_round_trip_l129_12914

def time_to_walk_uphill := 30 -- in minutes
def time_to_walk_downhill := 10 -- in minutes
def distance_one_way := 1 -- in km

theorem average_speed_round_trip :
  (2 * distance_one_way) / ((time_to_walk_uphill + time_to_walk_downhill) / 60) = 3 := by
  sorry

end average_speed_round_trip_l129_12914


namespace units_digit_of_n_l129_12982

def units_digit (x : ℕ) : ℕ := x % 10

theorem units_digit_of_n 
  (m n : ℕ) 
  (h1 : m * n = 21 ^ 6) 
  (h2 : units_digit m = 7) : 
  units_digit n = 3 := 
sorry

end units_digit_of_n_l129_12982


namespace food_company_total_food_l129_12936

theorem food_company_total_food (boxes : ℕ) (kg_per_box : ℕ) (full_boxes : boxes = 388) (weight_per_box : kg_per_box = 2) :
  boxes * kg_per_box = 776 :=
by
  -- the proof would go here
  sorry

end food_company_total_food_l129_12936


namespace S8_eq_90_l129_12905

-- Definitions and given conditions
def arithmetic_seq (a : ℕ → ℤ) : Prop := ∃ d, ∀ n, a (n + 1) - a n = d
def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop := ∀ n, S n = (n * (a 1 + a n)) / 2
def condition_a4 (a : ℕ → ℤ) : Prop := a 4 = 18 - a 5

-- Prove that S₈ = 90
theorem S8_eq_90 (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : arithmetic_seq a)
  (h_sum : sum_of_first_n_terms a S)
  (h_cond : condition_a4 a) : S 8 = 90 :=
by
  sorry

end S8_eq_90_l129_12905


namespace probability_four_red_four_blue_l129_12966

noncomputable def urn_probability : ℚ :=
  let initial_red := 2
  let initial_blue := 1
  let operations := 5
  let final_red := 4
  let final_blue := 4
  -- calculate the probability using given conditions, this result is directly derived as 2/7
  2 / 7

theorem probability_four_red_four_blue :
  urn_probability = 2 / 7 :=
by
  sorry

end probability_four_red_four_blue_l129_12966


namespace school_children_count_l129_12964

theorem school_children_count (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 370)) : C = 740 :=
by sorry

end school_children_count_l129_12964


namespace polar_to_cartesian_eq_polar_circle_area_l129_12993

theorem polar_to_cartesian_eq (p θ x y : ℝ) (h : p = 2 * Real.cos θ)
  (hx : x = p * Real.cos θ) (hy : y = p * Real.sin θ) :
  x^2 - 2 * x + y^2 = 0 := sorry

theorem polar_circle_area (p θ : ℝ) (h : p = 2 * Real.cos θ) :
  Real.pi = Real.pi := (by ring)


end polar_to_cartesian_eq_polar_circle_area_l129_12993


namespace initial_average_runs_l129_12947

theorem initial_average_runs (A : ℝ) (h : 10 * A + 65 = 11 * (A + 3)) : A = 32 :=
  by sorry

end initial_average_runs_l129_12947


namespace line_passes_through_fixed_point_range_of_k_no_second_quadrant_min_area_triangle_l129_12952

-- Problem 1: The line passes through a fixed point
theorem line_passes_through_fixed_point (k : ℝ) : ∃ P : ℝ × ℝ, P = (1, -2) ∧ (∀ x y, k * x - y - 2 - k = 0 → P = (x, y)) :=
by
  sorry

-- Problem 2: Range of values for k if the line does not pass through the second quadrant
theorem range_of_k_no_second_quadrant (k : ℝ) : ¬ (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ k * x - y - 2 - k = 0) → k ∈ Set.Ici (0) :=
by
  sorry

-- Problem 3: Minimum area of triangle AOB
theorem min_area_triangle (k : ℝ) :
  let A := (2 + k) / k
  let B := -2 - k
  (∀ x y, k * x - y - 2 - k = 0 ↔ (x = A ∧ y = 0) ∨ (x = 0 ∧ y = B)) →
  ∃ S : ℝ, S = 4 ∧ (∀ x y : ℝ, (k = 2 ∧ k * x - y - 4 = 0) → S = 4) :=
by
  sorry

end line_passes_through_fixed_point_range_of_k_no_second_quadrant_min_area_triangle_l129_12952


namespace shaded_region_area_l129_12915

theorem shaded_region_area (d : ℝ) (L : ℝ) (n : ℕ) (r : ℝ) (A : ℝ) (T : ℝ):
  d = 3 → L = 24 → L = n * d → n * 2 = 16 → r = d / 2 → 
  A = (1 / 2) * π * r ^ 2 → T = 16 * A → T = 18 * π :=
  by
  intros d_eq L_eq Ln_eq semicircle_count r_eq A_eq T_eq_total
  sorry

end shaded_region_area_l129_12915


namespace part1_part2_l129_12933

noncomputable def f (x : ℝ) : ℝ := (Real.exp (-x) - Real.exp x) / 2

theorem part1 (h_odd : ∀ x, f (-x) = -f x) (g : ℝ → ℝ) (h_even : ∀ x, g (-x) = g x)
  (h_g_def : ∀ x, g x = f x + Real.exp x) :
  ∀ x, f x = (Real.exp (-x) - Real.exp x) / 2 := sorry

theorem part2 : {x : ℝ | f x ≥ 3 / 4} = {x | x ≤ -Real.log 2} := sorry

end part1_part2_l129_12933


namespace gcd_lcm_product_24_60_l129_12904

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 :=
by
  sorry

end gcd_lcm_product_24_60_l129_12904


namespace problem_statement_l129_12946

noncomputable def a : ℝ := Real.tan (1 / 2)
noncomputable def b : ℝ := Real.tan (2 / Real.pi)
noncomputable def c : ℝ := Real.sqrt 3 / Real.pi

theorem problem_statement : a < c ∧ c < b := by
  sorry

end problem_statement_l129_12946


namespace perfect_square_for_n_l129_12983

theorem perfect_square_for_n 
  (a b : ℕ)
  (h1 : ∃ x : ℕ, ab = x^2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y^2) 
  : ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z^2 :=
by
  let n := ab
  have h3 : n > 1 := sorry
  have h4 : ∃ z : ℕ, (a + n) * (b + n) = z^2 := sorry
  exact ⟨n, h3, h4⟩

end perfect_square_for_n_l129_12983


namespace profit_percentage_correct_l129_12959

noncomputable def CP : ℝ := 460
noncomputable def SP : ℝ := 542.8
noncomputable def profit : ℝ := SP - CP
noncomputable def profit_percentage : ℝ := (profit / CP) * 100

theorem profit_percentage_correct :
  profit_percentage = 18 := by
  sorry

end profit_percentage_correct_l129_12959


namespace calculate_expression_l129_12991

theorem calculate_expression :
    (2^(1/2) * 4^(1/2)) + (18 / 3 * 3) - 8^(3/2) = 18 - 14 * Real.sqrt 2 := 
by 
  sorry

end calculate_expression_l129_12991


namespace linear_function_k_range_l129_12926

theorem linear_function_k_range (k b : ℝ) (h1 : k ≠ 0) (h2 : ∃ x : ℝ, (x = 2) ∧ (-3 = k * x + b)) (h3 : 0 < b ∧ b < 1) : -2 < k ∧ k < -3 / 2 :=
by
  sorry

end linear_function_k_range_l129_12926


namespace expression_evaluation_l129_12949

theorem expression_evaluation (p q : ℝ) (h : p / q = 4 / 5) : (25 / 7 + (2 * q - p) / (2 * q + p)) = 4 :=
by {
  sorry
}

end expression_evaluation_l129_12949


namespace total_cost_is_9220_l129_12954

-- Define the conditions
def hourly_rate := 60
def hours_per_day := 8
def total_days := 14
def cost_of_parts := 2500

-- Define the total cost the car's owner had to pay based on conditions
def total_hours := hours_per_day * total_days
def labor_cost := total_hours * hourly_rate
def total_cost := labor_cost + cost_of_parts

-- Theorem stating that the total cost is $9220
theorem total_cost_is_9220 : total_cost = 9220 := by
  sorry

end total_cost_is_9220_l129_12954


namespace right_triangle_x_value_l129_12924

variable (BM MA BC CA x h d : ℝ)

theorem right_triangle_x_value (BM MA BC CA x h d : ℝ)
  (h4 : BM + MA = BC + CA)
  (h5 : BM = x)
  (h6 : BC = h)
  (h7 : CA = d) :
  x = h * d / (2 * h + d) := 
sorry

end right_triangle_x_value_l129_12924


namespace sum_of_four_smallest_divisors_l129_12921

-- Define a natural number n and divisors d1, d2, d3, d4
def is_divisor (d n : ℕ) : Prop := ∃ k : ℕ, n = k * d

-- Primary problem condition (sum of four divisors equals 2n)
def sum_of_divisors_eq (n d1 d2 d3 d4 : ℕ) : Prop := d1 + d2 + d3 + d4 = 2 * n

-- Assume the four divisors of n are distinct
def distinct (d1 d2 d3 d4 : ℕ) : Prop := d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4

-- State the Lean proof problem
theorem sum_of_four_smallest_divisors (n d1 d2 d3 d4 : ℕ) (h1 : d1 < d2) (h2 : d2 < d3) (h3 : d3 < d4) 
    (h_div1 : is_divisor d1 n) (h_div2 : is_divisor d2 n) (h_div3 : is_divisor d3 n) (h_div4 : is_divisor d4 n)
    (h_sum : sum_of_divisors_eq n d1 d2 d3 d4) (h_distinct : distinct d1 d2 d3 d4) : 
    (d1 + d2 + d3 + d4 = 10 ∨ d1 + d2 + d3 + d4 = 11 ∨ d1 + d2 + d3 + d4 = 12) := 
sorry

end sum_of_four_smallest_divisors_l129_12921


namespace triangle_inequality_l129_12950

variables (a b c : ℝ)

theorem triangle_inequality (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : a + b > c) (h₄ : b + c > a) (h₅ : c + a > b) :
  (|a^2 - b^2| / c) + (|b^2 - c^2| / a) ≥ (|c^2 - a^2| / b) :=
by
  sorry

end triangle_inequality_l129_12950


namespace average_of_a_b_l129_12940

theorem average_of_a_b (a b : ℚ) (h1 : b = 2 * a) (h2 : (4 + 6 + 8 + a + b) / 5 = 17) : (a + b) / 2 = 33.5 := 
by
  sorry

end average_of_a_b_l129_12940


namespace product_of_sequence_is_243_l129_12951

theorem product_of_sequence_is_243 : 
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049) = 243 := 
by
  sorry

end product_of_sequence_is_243_l129_12951


namespace ashton_remaining_items_l129_12990

variables (pencil_boxes : ℕ) (pens_boxes : ℕ) (pencils_per_box : ℕ) (pens_per_box : ℕ)
          (given_pencils_brother : ℕ) (distributed_pencils_friends : ℕ)
          (distributed_pens_friends : ℕ)

def total_initial_pencils := 3 * 14
def total_initial_pens := 2 * 10

def remaining_pencils := total_initial_pencils - 6 - 12
def remaining_pens := total_initial_pens - 8
def remaining_items := remaining_pencils + remaining_pens

theorem ashton_remaining_items : remaining_items = 36 :=
sorry

end ashton_remaining_items_l129_12990


namespace eccentricity_of_ellipse_l129_12944

theorem eccentricity_of_ellipse (k : ℝ) (h_k : k > 0)
  (focus : ∃ (x : ℝ), (x, 0) = ⟨3, 0⟩) :
  ∃ e : ℝ, e = (Real.sqrt 3 / 2) := 
sorry

end eccentricity_of_ellipse_l129_12944


namespace min_value_expr_min_value_achieved_l129_12937

theorem min_value_expr (x : ℝ) (hx : x > 0) : 4*x + 1/x^4 ≥ 5 :=
by
  sorry

theorem min_value_achieved (x : ℝ) : x = 1 → 4*x + 1/x^4 = 5 :=
by
  sorry

end min_value_expr_min_value_achieved_l129_12937


namespace perpendicular_planes_implies_perpendicular_line_l129_12986

-- Definitions of lines and planes and their properties in space
variable {Space : Type}
variable (m n l : Line Space) -- Lines in space
variable (α β γ : Plane Space) -- Planes in space

-- Conditions: m, n, and l are non-intersecting lines, α, β, and γ are non-intersecting planes
axiom non_intersecting_lines : ¬ (m = n) ∧ ¬ (m = l) ∧ ¬ (n = l)
axiom non_intersecting_planes : ¬ (α = β) ∧ ¬ (α = γ) ∧ ¬ (β = γ)

-- To prove: if α ⊥ γ, β ⊥ γ, and α ∩ β = l, then l ⊥ γ
theorem perpendicular_planes_implies_perpendicular_line
  (h1 : α ⊥ γ) 
  (h2 : β ⊥ γ)
  (h3 : α ∩ β = l) : l ⊥ γ := 
  sorry

end perpendicular_planes_implies_perpendicular_line_l129_12986


namespace complex_div_eq_l129_12984

open Complex

def z := 4 - 2 * I

theorem complex_div_eq :
  (z + I = 4 - I) →
  (z / (4 + 2 * I) = (3 - 4 * I) / 5) :=
by
  sorry

end complex_div_eq_l129_12984


namespace find_fifth_term_l129_12968

noncomputable def geometric_sequence_fifth_term (a r : ℝ) (h₁ : a * r^2 = 16) (h₂ : a * r^6 = 2) : ℝ :=
  a * r^4

theorem find_fifth_term (a r : ℝ) (h₁ : a * r^2 = 16) (h₂ : a * r^6 = 2) : geometric_sequence_fifth_term a r h₁ h₂ = 2 := sorry

end find_fifth_term_l129_12968


namespace width_of_carpet_is_1000_cm_l129_12999

noncomputable def width_of_carpet_in_cm (total_cost : ℝ) (cost_per_meter : ℝ) (length_of_room : ℝ) : ℝ :=
  let total_length_of_carpet := total_cost / cost_per_meter
  let width_of_carpet_in_meters := total_length_of_carpet / length_of_room
  width_of_carpet_in_meters * 100

theorem width_of_carpet_is_1000_cm :
  width_of_carpet_in_cm 810 4.50 18 = 1000 :=
by sorry

end width_of_carpet_is_1000_cm_l129_12999


namespace find_percentage_l129_12935

theorem find_percentage (P : ℝ) : 
  (P / 100) * 100 - 40 = 30 → P = 70 :=
by
  intros h
  sorry

end find_percentage_l129_12935


namespace placement_ways_l129_12938

theorem placement_ways (rows cols crosses : ℕ) (h1 : rows = 3) (h2 : cols = 4) (h3 : crosses = 4)
  (condition : ∀ r : Fin rows, ∃ c : Fin cols, r < rows ∧ c < cols) : 
  (∃ n, n = (3 * 6 * 2) → n = 36) :=
by 
  -- Proof placeholder
  sorry

end placement_ways_l129_12938


namespace cos_of_angle_complement_l129_12941

theorem cos_of_angle_complement (α : ℝ) (h : 90 - α = 30) : Real.cos α = 1 / 2 :=
by
  sorry

end cos_of_angle_complement_l129_12941


namespace min_value_of_square_sum_l129_12978

theorem min_value_of_square_sum (x y : ℝ) 
  (h1 : (x + 5) ^ 2 + (y - 12) ^ 2 = 14 ^ 2) : 
  x^2 + y^2 = 1 := 
sorry

end min_value_of_square_sum_l129_12978


namespace quadratic_roots_one_is_twice_l129_12994

theorem quadratic_roots_one_is_twice (a b c : ℝ) (m : ℝ) :
  (∃ x1 x2 : ℝ, 2 * x1^2 - (2 * m + 1) * x1 + m^2 - 9 * m + 39 = 0 ∧ x2 = 2 * x1) ↔ m = 10 ∨ m = 7 :=
by 
  sorry

end quadratic_roots_one_is_twice_l129_12994
