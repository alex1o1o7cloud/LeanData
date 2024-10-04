import Mathlib

namespace students_in_other_religious_communities_l233_233897

theorem students_in_other_religious_communities 
  (total_students : ℕ)
  (percentage_muslims : ℝ)
  (percentage_hindus : ℝ)
  (percentage_sikhs : ℝ)
  (percentage_christians : ℝ)
  (percentage_buddhists : ℝ)
  (percentage_jews : ℝ)
  (percentage_jains : ℝ)
  (total_students = 1500)
  (percentage_muslims = 30.5 / 100)
  (percentage_hindus = 23.25 / 100)
  (percentage_sikhs = 14.75 / 100)
  (percentage_christians = 9.35 / 100)
  (percentage_buddhists = 4.65 / 100)
  (percentage_jews = 2.25 / 100)
  (percentage_jains = 1.4 / 100) :
  ∃ (students_other_communities : ℕ), students_other_communities = 208 := by
  sorry

end students_in_other_religious_communities_l233_233897


namespace circle_equation_through_intersections_l233_233154

theorem circle_equation_through_intersections 
  (h₁ : ∀ x y : ℝ, x^2 + y^2 + 6 * x - 4 = 0 ↔ x^2 + y^2 + 6 * y - 28 = 0)
  (h₂ : ∀ x y : ℝ, x - y - 4 = 0) : 
  ∃ x y : ℝ, (x - 1/2) ^ 2 + (y + 7 / 2) ^ 2 = 89 / 2 :=
by sorry

end circle_equation_through_intersections_l233_233154


namespace nonnegative_solutions_count_l233_233260

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.finite.to_finset.card = 1 :=
by
  sorry

end nonnegative_solutions_count_l233_233260


namespace valid_votes_B_and_C_l233_233345

-- Definitions based on conditions
variable (V : ℕ) (valid_votes A_votes B_votes C_votes : ℕ)
def total_votes := 5720
def percent_valid := 0.75
def percent_A_exceeds_B := 0.15
def percent_C := 0.10

-- Conditions
axiom total_votes_def : V = total_votes
axiom valid_votes_def : valid_votes = (percent_valid * total_votes).to_nat
axiom A_votes_def : A_votes = B_votes + (percent_A_exceeds_B * total_votes).to_nat
axiom C_votes_def : C_votes = (percent_C * total_votes).to_nat
axiom valid_votes_sum_def : A_votes + B_votes + C_votes = valid_votes

-- Question: Prove that B_votes + C_votes = 2002
theorem valid_votes_B_and_C : B_votes + C_votes = 2002 :=
by 
  sorry

end valid_votes_B_and_C_l233_233345


namespace total_number_of_students_l233_233034

-- Statement translating the problem conditions and conclusion
theorem total_number_of_students (rank_from_right rank_from_left total : ℕ) 
  (h_right : rank_from_right = 13) 
  (h_left : rank_from_left = 8) 
  (total_eq : total = rank_from_right + rank_from_left - 1) : 
  total = 20 := 
by 
  -- Proof is skipped
  sorry

end total_number_of_students_l233_233034


namespace find_min_max_u_l233_233787

noncomputable def conditions (x y : ℝ) : Prop :=
  let z1 := x + sqrt 11 + (y : ℂ)*complex.i
  let z2 := x - sqrt 11 + (y : ℂ)*complex.i
  complex.abs z1 + complex.abs z2 = 12

theorem find_min_max_u (x y : ℝ) (h : conditions x y) : 
  ∃ (umin umax : ℝ), umin = 0 ∧ umax = 30 :=
by
  let u := abs (5 * x - 6 * y - 30)
  use [0, 30]
  sorry

end find_min_max_u_l233_233787


namespace nonnegative_solutions_count_l233_233264

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.finite.to_finset.card = 1 :=
by
  sorry

end nonnegative_solutions_count_l233_233264


namespace arithmetic_progression_general_formula_geometric_progression_condition_l233_233182

-- Arithmetic progression problem
theorem arithmetic_progression_general_formula :
  ∀ (a : ℕ → ℚ) (S : ℕ → ℚ),
  (∀ n, a n > 0) →
  S 10 = 70 →
  a 1 = 1 →
  (∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2) →
  (∀ n, a n = 1 + (n - 1) * (4/3)) :=
by
  sorry

-- Geometric progression problem
theorem geometric_progression_condition :
  ∀ (a : ℕ → ℚ) (S : ℕ → ℚ),
  (∀ n, a n > 0) →
  a 1 = 1 →
  a 4 = 1/8 →
  (∀ n, S n = a 1 * (1 - (1/2)^n) / (1 - 1/2)) →
  (∃ n, S n > 100 * a n ∧ ∀ m < n, S m ≤ 100 * a m) :=
by
  use 7
  sorry

end arithmetic_progression_general_formula_geometric_progression_condition_l233_233182


namespace cake_remaining_after_4_trips_l233_233672

theorem cake_remaining_after_4_trips :
  ∀ (cake_portion_left_after_trip : ℕ → ℚ), 
    cake_portion_left_after_trip 0 = 1 ∧
    (∀ n, cake_portion_left_after_trip (n + 1) = cake_portion_left_after_trip n / 2) →
    cake_portion_left_after_trip 4 = 1 / 16 :=
by
  intros cake_portion_left_after_trip h
  have h0 : cake_portion_left_after_trip 0 = 1 := h.1
  have h1 : ∀ n, cake_portion_left_after_trip (n + 1) = cake_portion_left_after_trip n / 2 := h.2
  sorry

end cake_remaining_after_4_trips_l233_233672


namespace friend_gain_is_20_percent_l233_233061

noncomputable def original_cost : ℝ := 52325.58
noncomputable def loss_percentage : ℝ := 0.14
noncomputable def friend_selling_price : ℝ := 54000
noncomputable def friend_percentage_gain : ℝ :=
  ((friend_selling_price - (original_cost * (1 - loss_percentage))) / (original_cost * (1 - loss_percentage))) * 100

theorem friend_gain_is_20_percent :
  friend_percentage_gain = 20 := by
  sorry

end friend_gain_is_20_percent_l233_233061


namespace problem_statement_l233_233946

theorem problem_statement 
  (w x y z : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z = 945) :
  2 * w + 3 * x + 5 * y + 7 * z = 21 :=
by
  sorry

end problem_statement_l233_233946


namespace greatest_prime_factor_of_341_l233_233612

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l233_233612


namespace base_five_equivalent_of_156_is_1111_l233_233512

theorem base_five_equivalent_of_156_is_1111 : nat_to_base 5 156 = [1, 1, 1, 1] := 
sorry

end base_five_equivalent_of_156_is_1111_l233_233512


namespace log_implies_exp_l233_233935

variables {a b : ℝ}

-- Assume a and b are positive numbers that are not equal to 1
def pos_non_one (x : ℝ) : Prop := 0 < x ∧ x ≠ 1

-- The logarithmic inequality
def log_inequality (a b : ℝ) : Prop := (Real.log 2 / Real.log a) < (Real.log 2 / Real.log b)

-- The exponential inequality
def exp_inequality (a b : ℝ) : Prop := 3^a > 3^b ∧ 3^b > 3

-- The necessary but not sufficient condition
theorem log_implies_exp (h1 : pos_non_one a) (h2 : pos_non_one b) (h3 : a ≠ b) :
  log_inequality a b → exp_inequality a b := 
sorry

end log_implies_exp_l233_233935


namespace inclination_angle_range_l233_233211

theorem inclination_angle_range
  (k : ℝ) (alpha : ℝ) 
  (h1 : -1 ≤ k) 
  (h2 : k < real.sqrt 3) 
  (h3 : k = real.tan alpha) 
  (h4 : 0 ≤ alpha) 
  (h5 : alpha < real.pi) : 
  (0 ≤ alpha ∧ alpha < real.pi / 3) ∨ (3 * real.pi / 4 ≤ alpha ∧ alpha < real.pi) :=
by
  sorry

end inclination_angle_range_l233_233211


namespace value_of_coins_l233_233965

theorem value_of_coins (n d : ℕ) (hn : n + d = 30)
    (hv : 10 * n + 5 * d = 5 * n + 10 * d + 90) :
    300 - 5 * n = 180 := by
  sorry

end value_of_coins_l233_233965


namespace rationalize_denominator_ABC_value_l233_233410

def A := 11 / 4
def B := 5 / 4
def C := 5

theorem rationalize_denominator : 
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

theorem ABC_value :
  A * B * C = 275 :=
sorry

end rationalize_denominator_ABC_value_l233_233410


namespace maximum_n_l233_233165

def g (x : ℕ) : ℕ := sorry -- definition of g here

def S_n (n : ℕ) : ℕ := 
  ∑ k in (finset.range (3^(n-1))).filter (λ k, k > 0), g (3 * k)

theorem maximum_n (n : ℕ) (h : n < 1000) : 
  (∃ n : ℕ, n < 1000 ∧ is_square (S_n n) ∧ ∀ m < 1000, is_square (S_n m) → m ≤ n) ↔ n = 629 :=
sorry

end maximum_n_l233_233165


namespace statement_I_true_statement_II_false_statement_III_false_correct_answer_l233_233770

def floor (x : ℝ) : ℤ := ⌊x⌋

theorem statement_I_true (x : ℝ) : floor (x - 1) = floor x - 1 :=
by sorry

theorem statement_II_false (x y : ℝ) : ¬ (floor (x - y) = floor x - floor y) :=
by sorry

theorem statement_III_false (x y : ℝ) (hy : y ≠ 0) : ¬ (floor (x / y) = floor x / floor y) :=
by sorry

theorem correct_answer : ∀ (x y : ℝ), y ≠ 0 → 
  (statement_I_true x ∧ statement_II_false x y ∧ statement_III_false x y) :=
by
  intro x y hy
  constructor
  · apply statement_I_true
  · apply statement_II_false
  · apply statement_III_false hy
  sorry

end statement_I_true_statement_II_false_statement_III_false_correct_answer_l233_233770


namespace num_perfect_squares_2_to_150_l233_233314

theorem num_perfect_squares_2_to_150 : 
  (finset.filter (λ n : ℕ, 2 ≤ n^2 ∧ n^2 ≤ 150) (finset.range 151)).card = 11 := 
sorry

end num_perfect_squares_2_to_150_l233_233314


namespace value_of_f_6_l233_233327

def f : ℤ → ℤ
| n := if n = 4 then 20 else f (n - 1) - n

theorem value_of_f_6 (n : ℤ) (H1 : f 4 = 20) (H2 : ∀ n, f n = f (n-1) - n) : f 6 = 9 :=
by sorry

end value_of_f_6_l233_233327


namespace greatest_prime_factor_of_341_is_17_l233_233560

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l233_233560


namespace greatest_prime_factor_341_l233_233516

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l233_233516


namespace projection_of_v1_onto_v2_l233_233158

-- Definitions for the given vectors.
def v₁ : ℝ × ℝ := (5, -3)
def v₂ : ℝ × ℝ := (-1, 4)

-- Proof statement.
theorem projection_of_v1_onto_v2 :
  (let dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 in
   let norm_squared (v : ℝ × ℝ) : ℝ := v.1 * v.1 + v.2 * v.2 in
   let proj (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (dot_product v1 v2 / norm_squared v2) * v2 in
   proj v₁ v₂) = (1, -4) :=
by
  sorry -- The proof steps are omitted as per instructions.

end projection_of_v1_onto_v2_l233_233158


namespace nonnegative_solutions_eq_1_l233_233302

variable (x : ℝ)

theorem nonnegative_solutions_eq_1 : (x^2 = -6 * x) → (∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀) → (x ∈ {0}) :=
by
  assume (h1: x^2 = -6 * x)
  assume (h2: ∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀)
  sorry

end nonnegative_solutions_eq_1_l233_233302


namespace sum_x_coordinates_eq_l233_233126

-- Define the piecewise linear function f(x)
def f (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < -1 then 3 * x
  else if -1 ≤ x ∧ x < 1 then -2 * x + 1
  else if 1 ≤ x ∧ x ≤ 3 then 3 * x - 2
  else 0  -- outside the defined intervals

-- Define the function g(x)
def g (x : ℝ) : ℝ := 5 - x

-- The main theorem to prove
theorem sum_x_coordinates_eq : 
  ∑ (x : ℝ) in {x | f x = g x}.toFinset, x = 7 / 4 :=
by
  -- Proof is not required, so we use sorry
  sorry

end sum_x_coordinates_eq_l233_233126


namespace sum_of_six_distinct_real_roots_l233_233442

noncomputable def g (x : Real) : Real := sorry

theorem sum_of_six_distinct_real_roots (h_sym : ∀ x : Real, g (3 + x) = g (3 - x))
  (h_roots : ∃ (s : Fin 6 → Real), (∀ i, g (s i) = 0) ∧ (Function.Injective s)) :
  (∑ i : Fin 6, s i) = 18 :=
by
  sorry

end sum_of_six_distinct_real_roots_l233_233442


namespace train_speed_correct_l233_233082

-- Define the problem conditions
def length_of_train : ℝ := 300  -- length in meters
def time_to_cross_pole : ℝ := 18  -- time in seconds

-- Conversion factors
def meters_to_kilometers : ℝ := 0.001
def seconds_to_hours : ℝ := 1 / 3600

-- Define the conversions
def distance_in_kilometers := length_of_train * meters_to_kilometers
def time_in_hours := time_to_cross_pole * seconds_to_hours

-- Define the speed calculation
def speed_of_train := distance_in_kilometers / time_in_hours

-- The theorem to prove
theorem train_speed_correct : speed_of_train = 60 := 
by
  sorry

end train_speed_correct_l233_233082


namespace count_congruent_numbers_less_than_500_l233_233853

-- Definitions of the conditions
def is_congruent_to_modulo (n a m : ℕ) : Prop := (n % m) = a

-- Main problem statement: Proving that the count of numbers under 500 that satisfy the conditions is 71.
theorem count_congruent_numbers_less_than_500 : 
  { n : ℕ | n < 500 ∧ is_congruent_to_modulo n 3 7 }.card = 71 :=
by
  sorry

end count_congruent_numbers_less_than_500_l233_233853


namespace find_a_if_pure_imaginary_l233_233318

variables (a : ℝ) (i : ℂ) (h1 : i^2 = -1)
noncomputable def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_a_if_pure_imaginary :
  is_pure_imaginary (∃ (a : ℝ), (⟦(2 * a + i) / (1 - 2 * i)⟧ = 0)) → a = 1 :=
sorry

end find_a_if_pure_imaginary_l233_233318


namespace original_integer_is_21_l233_233168

theorem original_integer_is_21 (a b c d : ℕ) 
  (h1 : (a + b + c) / 3 + d = 29) 
  (h2 : (a + b + d) / 3 + c = 23) 
  (h3 : (a + c + d) / 3 + b = 21) 
  (h4 : (b + c + d) / 3 + a = 17) : 
  d = 21 :=
sorry

end original_integer_is_21_l233_233168


namespace fish_initially_tagged_l233_233892

theorem fish_initially_tagged :
  ∀ (T N : ℕ),
  (∀ (total_catch second_tagged : ℕ),
    total_catch = 50 → second_tagged = 2 →
    (T / (N : ℚ)) = (second_tagged / total_catch)) →
  N = 1000 →
  T = 40 :=
by
  intros T N H N_eq
  have H1 := H 50 2 rfl rfl
  have H2 : (T : ℚ) = 40 := by
    rw [N_eq] at H1
    norm_cast
    linarith
  norm_cast at H2
  exact H2

example :
  ∃ (T N : ℕ),
  N = 1000 ∧
  (∀ (total_catch second_tagged : ℕ),
    total_catch = 50 → second_tagged = 2 →
    (T / (N : ℚ)) = (second_tagged / total_catch)) ∧
  T = 40 :=
⟨40, 1000, rfl, by
  intros total_catch second_tagged hc hst
  cases hc
  cases hst
  norm_cast
  linarith, rfl⟩

end fish_initially_tagged_l233_233892


namespace num_possible_arrangements_l233_233775

open Finset

def is_adjacent (r1 r2 : ℕ) : Prop := 
  (r1 = r2 + 1) ∨ (r1 + 1 = r2)

theorem num_possible_arrangements :
  let rooms := {1, 2, 3, 4} : finset ℕ in
  let arrangements := {s : finset (ℕ × char) | 
    ∃ (a b c d : ℕ), 
      a ∈ rooms ∧ b ∈ rooms ∧ c ∈ rooms ∧ d ∈ rooms ∧
      b ≠ 2 ∧ 
      ((b = 1 ∧ c = 2) ∨ 
       (b = 3 ∧ (c = 2 ∨ c = 4)) ∨ 
       (b = 4 ∧ c = 3)) ∧ 
      s = {(a, 'A'), (b, 'B'), (c, 'C'), (d, 'D')} ∧
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
  } in
  arrangements.card = 8 :=
by
  sorry

end num_possible_arrangements_l233_233775


namespace ivan_petrovich_lessons_daily_l233_233037

def donations_per_month (L k : ℕ) : ℕ := 21 * (k / 3) * 1000

theorem ivan_petrovich_lessons_daily (L k : ℕ) (h1 : 24 = 8 + L + 2 * L + k) (h2 : k = 16 - 3 * L)
    (income_from_lessons : 21 * (3 * L) * 1000)
    (rent_income : 14000)
    (monthly_expenses : 70000)
    (charity_donations : donations_per_month L k) :
  L = 2 ∧ charity_donations = 70000 := 
begin
  sorry
end

end ivan_petrovich_lessons_daily_l233_233037


namespace interval_length_and_sum_rel_prime_inequality_l233_233656

theorem interval_length_and_sum_rel_prime_inequality :
  ∃ (a b : ℚ) (m n : ℕ), gcd m n = 1 ∧ a ≤ b ∧ ([∀ x : ℚ, (abs (5 * x^2 - 2 / 5) ≤ abs (x - 8)) → (x ≥ a ∧ x ≤ b)] ∧ b - a = m / n ∧ m + n = 18 :=
by
  sorry

end interval_length_and_sum_rel_prime_inequality_l233_233656


namespace nonnegative_solutions_count_l233_233263

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.finite.to_finset.card = 1 :=
by
  sorry

end nonnegative_solutions_count_l233_233263


namespace arithmetic_progression_sum_geometric_progression_sum_l233_233181

-- Arithmetic Progression Problem
theorem arithmetic_progression_sum (d : ℚ) :
  let S_n (n : ℕ) := n * (1 + (n - 1) / 2 * d) in
  S_n 10 = 70 → 
  ∀ n, S_n n = n * (1 + (n - 1) / 2 * (4 / 3)) := 
by
  sorry

-- Geometric Progression Problem
theorem geometric_progression_sum (q : ℚ) :
  let a_n (n : ℕ) := 1 * q ^ (n - 1) in
  let S_n (n : ℕ) := (1 * (1 - q ^ n)) / (1 - q) in
  a_n 4 = 1 / 8 → 
  S_n n > 100 * a_n n → 
  n ≥ 7 :=
by
  sorry

end arithmetic_progression_sum_geometric_progression_sum_l233_233181


namespace greatest_prime_factor_341_l233_233591

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l233_233591


namespace weight_of_bowling_ball_l233_233421

-- Definition of weights of the canoes and bowling balls
variables (canoe_weight : ℕ) (bowling_ball_weight : ℕ)

// Statement of the conditions
def condition_1 := 7 * bowling_ball_weight = 3 * canoe_weight
def condition_2 := canoe_weight = 28

-- The final theorem to prove the weight of a bowling ball
theorem weight_of_bowling_ball : bowling_ball_weight = 12 :=
by
  unfold condition_1 condition_2
  have h1 : 7 * bowling_ball_weight = 3 * 28 := by sorry -- This represents the instantiation of the first condition
  have h2 : 3 * 28 = 84 := by norm_num
  rw h2 at h1
  have h3 : bowling_ball_weight = 12 := by norm_num; exact h1
  exact h3

end weight_of_bowling_ball_l233_233421


namespace log_sum_geometric_sequence_l233_233059

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem log_sum_geometric_sequence 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : geometric_sequence a)
  (h3 : a 2 * a 8 = 4) :
  ∑ n in finset.range 9, Real.log 2 (a n) = 9 := 
sorry

end log_sum_geometric_sequence_l233_233059


namespace uki_cupcakes_per_day_l233_233498

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def daily_cookies : ℝ := 10
def daily_biscuits : ℝ := 20
def total_earnings : ℝ := 350
def days : ℝ := 5

-- Define the number of cupcakes baked per day
def cupcakes_per_day (x : ℝ) : Prop :=
  let earnings_cupcakes := price_cupcake * x * days
  let earnings_cookies := price_cookie * daily_cookies * days
  let earnings_biscuits := price_biscuit * daily_biscuits * days
  earnings_cupcakes + earnings_cookies + earnings_biscuits = total_earnings

-- The statement to be proven
theorem uki_cupcakes_per_day : cupcakes_per_day 20 :=
by 
  sorry

end uki_cupcakes_per_day_l233_233498


namespace jason_cutting_grass_time_over_weekend_l233_233919

theorem jason_cutting_grass_time_over_weekend 
    (time_small_lawn : ℕ := 25)
    (time_medium_lawn : ℕ := 30)
    (time_large_lawn : ℕ := 40)
    (break_time : ℕ := 5)
    (extra_time_weather : ℕ := 10)
    -- Saturday
    (n_small_lawns_sat : ℕ := 2)
    (n_medium_lawns_sat : ℕ := 4)
    (n_large_lawns_sat : ℕ := 2)
    -- Sunday
    (n_medium_lawns_sun : ℕ := 6)
    (n_large_lawns_sun : ℕ := 2)
    : 
    let
        -- Time on Saturday
        time_sat := (n_small_lawns_sat * time_small_lawn) + (n_medium_lawns_sat * time_medium_lawn) + (n_large_lawns_sat * time_large_lawn) + 
                    ((n_small_lawns_sat + n_medium_lawns_sat + n_large_lawns_sat - 1) * break_time),
        -- Time on Sunday
        time_sun := 
            (n_medium_lawns_sun * (time_medium_lawn + extra_time_weather)) + 
            (n_large_lawns_sun * (time_large_lawn + extra_time_weather)) +
            ((n_medium_lawns_sun + n_large_lawns_sun - 1) * break_time),
        -- Total time in minutes
        total_time := time_sat + time_sun,
        -- Total time in hours
        total_time_hours := total_time / 60
    in
        total_time_hours = 11 := 
sorry

end jason_cutting_grass_time_over_weekend_l233_233919


namespace greatest_prime_factor_341_l233_233579

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l233_233579


namespace greatest_prime_factor_of_341_is_17_l233_233559

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l233_233559


namespace count_congruent_numbers_less_than_500_l233_233850

-- Definitions of the conditions
def is_congruent_to_modulo (n a m : ℕ) : Prop := (n % m) = a

-- Main problem statement: Proving that the count of numbers under 500 that satisfy the conditions is 71.
theorem count_congruent_numbers_less_than_500 : 
  { n : ℕ | n < 500 ∧ is_congruent_to_modulo n 3 7 }.card = 71 :=
by
  sorry

end count_congruent_numbers_less_than_500_l233_233850


namespace calculate_second_discount_l233_233054

theorem calculate_second_discount
  (P : ℝ) (S : ℝ) (D : ℝ)
  (hP : P = 9502.923976608186)
  (hS : S = 6500) :
  S = P * 0.8 * (1 - D / 100) →
  D = 14.47368421052632 :=
begin
  sorry  -- Proof to be filled in
end

end calculate_second_discount_l233_233054


namespace second_place_votes_l233_233893

-- Define the candidates and their votes
def votes_c1 := 12
def votes_c5 := 4
def total_votes := 36
def votes_c2_possible (x : ℕ) := x ∈ {8, 9}

theorem second_place_votes : Σ x, votes_c2_possible x :=
by
  sorry

end second_place_votes_l233_233893


namespace number_of_dogs_in_kennel_l233_233891

-- Define the conditions
def long_furred_brown_dogs : ℕ := 11
def long_furred_dogs : ℕ := 26
def brown_dogs : ℕ := 22
def neither_long_furred_nor_brown_dogs : ℕ := 8

-- Define the total number of dogs
def total_dogs (long_furred_brown_dogs long_furred_dogs brown_dogs neither_long_furred_nor_brown_dogs : ℕ) : ℕ :=
  let long_furred_or_brown_dogs := long_furred_dogs + brown_dogs - long_furred_brown_dogs
  long_furred_or_brown_dogs + neither_long_furred_nor_brown_dogs

-- Define the theorem to prove
theorem number_of_dogs_in_kennel :
  total_dogs long_furred_brown_dogs long_furred_dogs brown_dogs neither_long_furred_nor_brown_dogs = 45 :=
by {
  -- Use the definitions to simplify the goal
  unfold total_dogs,
  -- Direct computation showing conclusion holds
  have : 26 + 22 - 11 + 8 = 45, from rfl,
  exact this,
  sorry
}

end number_of_dogs_in_kennel_l233_233891


namespace greatest_prime_factor_341_l233_233524

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l233_233524


namespace area_of_hexagon_l233_233947

def isRegularHexagon (A B C D E F : Type) : Prop := sorry
def isInsideQuadrilateral (P : Type) (A B C D : Type) : Prop := sorry
def areaTriangle (P X Y : Type) : Real := sorry

theorem area_of_hexagon (A B C D E F P : Type)
    (h1 : isRegularHexagon A B C D E F)
    (h2 : isInsideQuadrilateral P A B C D)
    (h3 : areaTriangle P B C = 20)
    (h4 : areaTriangle P A D = 23) :
    ∃ area : Real, area = 189 :=
sorry

end area_of_hexagon_l233_233947


namespace zach_brother_cookies_4_l233_233649

variable (cookies_monday cookies_tuesday cookies_wednesday cookies_end cookies_brother_eaten : ℕ)

def baked_cookies_conditions :=
  cookies_monday = 32 ∧
  cookies_tuesday = cookies_monday / 2 ∧
  cookies_wednesday = cookies_tuesday * 3 ∧
  cookies_end = cookies_monday + cookies_tuesday + cookies_wednesday - cookies_brother_eaten ∧
  cookies_end = 92

theorem zach_brother_cookies_4 : 
  baked_cookies_conditions cookies_monday cookies_tuesday cookies_wednesday cookies_end cookies_brother_eaten →
  cookies_brother_eaten = 4 :=
by
  intro h,
  let ⟨h1, h2, h3, h4, h5⟩ := h,
  sorry

end zach_brother_cookies_4_l233_233649


namespace prob_at_least_3_is_correct_expected_value_X_is_correct_l233_233990

-- Define the events
def needs_device_A : Prop := true -- Placeholder, actual definition would depend on probability space definition
def needs_device_B : Prop := true -- Placeholder
def needs_device_C : Prop := true -- Placeholder
def needs_device_D : Prop := true -- Placeholder

-- Probabilities of each person needing the device
axiom prob_A : ℝ := 0.6
axiom prob_B : ℝ := 0.5
axiom prob_C : ℝ := 0.5
axiom prob_D : ℝ := 0.4

-- Define independence of events
axiom independence : ∀ (P Q : Prop), P ∧ Q = (P * Q) -- Placeholder for actual independence definition

-- Probability calculation for at least 3 people
def at_least_3 : ℝ :=
  prob_A * prob_B * prob_C * prob_D + 
  (1 - prob_A) * prob_B * prob_C * prob_D + 
  prob_A * (1 - prob_B) * prob_C * prob_D + 
  prob_A * prob_B * (1 - prob_C) * prob_D + 
  prob_A * prob_B * prob_C * (1 - prob_D)

-- Expected value calculation for X
def P_X_0 : ℝ := (1 - prob_A) * (1 - prob_B) * (1 - prob_C) * (1 - prob_D)
def P_X_1 : ℝ := prob_A * (1 - prob_B) * (1 - prob_C) * (1 - prob_D) +
                (1 - prob_A) * prob_B * (1 - prob_C) * (1 - prob_D) +
                (1 - prob_A) * (1 - prob_B) * prob_C * (1 - prob_D) +
                (1 - prob_A) * (1 - prob_B) * (1 - prob_C) * prob_D
def P_X_2 : ℝ := -- Placeholder for calculation
def P_X_3 : ℝ := -- Placeholder for calculation
def P_X_4 : ℝ := prob_A * prob_B * prob_C * prob_D

def expected_value_X : ℝ := 0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 + 3 * P_X_3 + 4 * P_X_4

-- Statements to prove
theorem prob_at_least_3_is_correct : at_least_3 = 0.31 := by sorry
theorem expected_value_X_is_correct : expected_value_X = 2 := by sorry

end prob_at_least_3_is_correct_expected_value_X_is_correct_l233_233990


namespace solution_set_min_value_fraction_l233_233826

def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part (1)
theorem solution_set (x : ℝ) : f x + |x + 1| < 2 ↔ 0 < x ∧ x < 2 / 3 :=
by
  sorry

def g (x : ℝ) : ℝ := f x + f (x - 1)
def a : ℝ := 2
def m : ℝ
def n : ℝ

-- Assume m + n = a (= 2) and m, n > 0
axiom h1 : m + n = 2
axiom h2 : m > 0
axiom h3 : n > 0

-- Part (2)
theorem min_value_fraction (h: m + n = a) (h4 : g x = a) (x : ℝ) : (4 / m + 1 / n) = 9 / 2 :=
by
  sorry

end solution_set_min_value_fraction_l233_233826


namespace greatest_prime_factor_of_341_is_17_l233_233558

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l233_233558


namespace graph_parts_of_unit_circle_l233_233443

noncomputable def problem_statement (x y : ℝ) (n : ℤ) :=
  arcsin x + arccos y = n * π ∧ 
  (-π/2 <= arcsin x ∧ arcsin x <= π/2) ∧
  (0 <= arccos y ∧ arccos y <= π)

theorem graph_parts_of_unit_circle (x y : ℝ) (n : ℤ) :
  problem_statement x y n → (x^2 + y^2 = 1 ∧ (x <= 0 ∧ y >= 0) ∨ (x >= 0 ∧ y <= 0)) := 
sorry

end graph_parts_of_unit_circle_l233_233443


namespace train_speed_l233_233078

theorem train_speed (length_m : ℕ) (time_s : ℕ) (length_km : ℝ) (time_hr : ℝ) 
(length_conversion : length_km = (length_m : ℝ) / 1000)
(time_conversion : time_hr = (time_s : ℝ) / 3600)
(speed : ℝ) (speed_formula : speed = length_km / time_hr) :
  length_m = 300 → time_s = 18 → speed = 60 :=
by
  intros h1 h2
  rw [h1, h2] at *
  simp [length_conversion, time_conversion, speed_formula]
  norm_num
  sorry

end train_speed_l233_233078


namespace greatest_prime_factor_341_l233_233513

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l233_233513


namespace concyclic_points_l233_233041

noncomputable def cyclic_quadrilateral (A B C D : Point) : Prop :=
  ∃ (circle : Circle), ∀ (P : Point), P ∈ {A, B, C, D} → P ∈ circle

noncomputable def intersection_point (l1 l2 : Line) : Point := sorry
noncomputable def midpoint (P Q : Point) : Point := sorry
noncomputable def circumcircle (T : Triangle) : Circle := sorry
noncomputable def second_intersection (c1 c2 : Circle) : Point := sorry
noncomputable def are_concyclic (points : List Point) : Prop := sorry

theorem concyclic_points {A B C D E F M N K : Point} :
  cyclic_quadrilateral A B C D →
  E = intersection_point (line_through A C) (line_through B D) →
  F = intersection_point (line_through A B) (line_through C D) →
  M = midpoint A B →
  N = midpoint C D →
  K = second_intersection (circumcircle ⟨A, B, E⟩) (circumcircle ⟨A, C, N⟩) →
  are_concyclic [F, K, M, N] :=
by
  intros hcyclic hE hF hM hN hK
  sorry

end concyclic_points_l233_233041


namespace expression_evaluation_l233_233706

theorem expression_evaluation : 
  (3.14 - Real.pi)^0 + abs (Real.sqrt 2 - 1) + (1 / 2)^(-1:ℤ) - Real.sqrt 8 = 2 - Real.sqrt 2 :=
by sorry

end expression_evaluation_l233_233706


namespace factor_expression_l233_233143

theorem factor_expression (b : ℚ) : 
  294 * b^3 + 63 * b^2 - 21 * b = 21 * b * (14 * b^2 + 3 * b - 1) :=
by 
  sorry

end factor_expression_l233_233143


namespace greatest_prime_factor_of_341_l233_233541

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l233_233541


namespace greatest_prime_factor_341_l233_233527

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l233_233527


namespace equipment_total_cost_l233_233464

-- Definition of costs for each item of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80

-- Number of players
def num_players : ℕ := 16

-- Total cost for one player
def total_cost_one_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Total cost for all players
def total_cost_all_players : ℝ := total_cost_one_player * num_players

-- Theorem to prove
theorem equipment_total_cost : total_cost_all_players = 752 := by
  sorry

end equipment_total_cost_l233_233464


namespace problem_l233_233802

theorem problem:
  ∀ k : Real, (2 - Real.sqrt 2 / 2 ≤ k ∧ k ≤ 2 + Real.sqrt 2 / 2) →
  (11 - 6 * Real.sqrt 2) / 4 ≤ (3 / 2 * (k - 1)^2 + 1 / 2) ∧ 
  (3 / 2 * (k - 1)^2 + 1 / 2 ≤ (11 + 6 * Real.sqrt 2) / 4) :=
by
  intros k hk
  sorry

end problem_l233_233802


namespace circumscribed_quadrilateral_arc_sum_l233_233684

theorem circumscribed_quadrilateral_arc_sum 
  (a b c d : ℝ) 
  (h : a + b + c + d = 360) : 
  (1/2 * (b + c + d)) + (1/2 * (a + c + d)) + (1/2 * (a + b + d)) + (1/2 * (a + b + c)) = 540 :=
by
  sorry

end circumscribed_quadrilateral_arc_sum_l233_233684


namespace Andy_and_Carlos_tie_for_first_l233_233105

def AndyLawnArea (A : ℕ) := 3 * A
def CarlosLawnArea (A : ℕ) := A / 4
def BethMowingRate := 90
def CarlosMowingRate := BethMowingRate / 3
def AndyMowingRate := BethMowingRate * 4

theorem Andy_and_Carlos_tie_for_first (A : ℕ) (hA_nonzero : 0 < A) :
  (AndyLawnArea A / AndyMowingRate) = (CarlosLawnArea A / CarlosMowingRate) ∧
  (AndyLawnArea A / AndyMowingRate) < (A / BethMowingRate) :=
by
  unfold AndyLawnArea CarlosLawnArea BethMowingRate CarlosMowingRate AndyMowingRate
  sorry

end Andy_and_Carlos_tie_for_first_l233_233105


namespace alice_has_winning_strategy_l233_233975

def alice_has_winning_strategy_condition (nums : List ℤ) : Prop :=
  nums.length = 17 ∧ ∀ x ∈ nums, ¬ (x % 17 = 0)

theorem alice_has_winning_strategy (nums : List ℤ) (H : alice_has_winning_strategy_condition nums) : ∃ (f : List ℤ → List ℤ), ∀ k, (f^[k] nums).sum % 17 = 0 :=
sorry

end alice_has_winning_strategy_l233_233975


namespace new_area_eq_1_12_original_area_l233_233032

variable (L W : ℝ)
def increased_length (L : ℝ) : ℝ := 1.40 * L
def decreased_width (W : ℝ) : ℝ := 0.80 * W
def original_area (L W : ℝ) : ℝ := L * W
def new_area (L W : ℝ) : ℝ := (increased_length L) * (decreased_width W)

theorem new_area_eq_1_12_original_area (L W : ℝ) :
  new_area L W = 1.12 * (original_area L W) :=
by
  sorry

end new_area_eq_1_12_original_area_l233_233032


namespace sin_smallest_angle_l233_233887

noncomputable def smallest_angle_sine (a b c : ℝ) (BC CA AB : ℝ → ℝ → ℝ) (h : 20 * a * BC + 15 * b * CA + 12 * c * AB = 0) : Prop :=
  sin_a = 3 / 5

theorem sin_smallest_angle (a b c : ℝ) (BC CA AB : ℝ → ℝ → ℝ) (h : 20 * a * BC + 15 * b * CA + 12 * c * AB = 0) :
  smallest_angle_sine a b c BC CA AB h :=
sorry

end sin_smallest_angle_l233_233887


namespace transform_polynomial_roots_l233_233945

noncomputable def p : Polynomial ℚ := Polynomial.Coeff 27 *X^3 - Polynomial.Coeff 108 * X^2 + Polynomial.Coeff 81 * X + Polynomial.Coeff 162

theorem transform_polynomial_roots {r1 r2 r3 : ℚ} 
    (h_roots : Polynomial.root (Polynomial.Coeff 1*X^3 - Polynomial.Coeff 4 * X^2 + Polynomial.Coeff 1 * X + Polynomial.Coeff 6) r1 
            ∧ Polynomial.root (Polynomial.Coeff 1*X^3 - Polynomial.Coeff 4 * X^2 + Polynomial.Coeff 1 * X + Polynomial.Coeff 6) r2 
            ∧ Polynomial.root (Polynomial.Coeff 1*X^3 - Polynomial.Coeff 4 * X^2 + Polynomial.Coeff 1 * X + Polynomial.Coeff 6) r3) :
    Polynomial.root (Polynomial.Coeff 1 * X^3 - Polynomial.Coeff 12 * X^2 + Polynomial.Coeff 9 * X + Polynomial.Coeff 162) (3 * r1)
    ∧ Polynomial.root (Polynomial.Coeff 1 * X^3 - Polynomial.Coeff 12 * X^2 + Polynomial.Coeff 9 * X + Polynomial.Coeff 162) (3 * r2)
    ∧ Polynomial.root (Polynomial.Coeff 1 * X^3 - Polynomial.Coeff 12 * X^2 + Polynomial.Coeff 9 * X + Polynomial.Coeff 162) (3 * r3) :=
sorry

end transform_polynomial_roots_l233_233945


namespace max_area_WXYZ_l233_233367

noncomputable theory

-- Definitions according to the given problem
variables {A B C W X Y Z : Type} [fintype A] [fintype B] [fintype C]
variables (triangle_ABC : triangle A B C)
variables (square_WXYZ : square W X Y Z)
variables (angle_A : real) (BC_len : real)

def conditions : Prop :=
angle_A = (135 / 2) ∧ BC_len = 15 ∧
(∃ (W_on_AB : W ∈ segment A B) (X_on_AC : X ∈ segment A C) (Z_on_BC : Z ∈ segment B C),
triangle_similar (triangle Z B W) (triangle A B C))

-- Specification of the maximum area of the square
def max_area_of_square (s : real) : Prop :=
s ≤ (15 / (2 * real.sqrt (real.sqrt 2))) ∧
((15 / (2 * real.sqrt (real.sqrt 2))) ^ 2 = (225 * real.sqrt 2) / 8)

-- The proof problem statement
theorem max_area_WXYZ (s : real) :
conditions angle_A BC_len → max_area_of_square s :=
begin
  intro h,
  sorry
end

end max_area_WXYZ_l233_233367


namespace percentage_divisible_by_6_l233_233640

-- Defining the sets S and T using Lean
def S := {n : ℕ | 1 ≤ n ∧ n ≤ 120}
def T := {n : ℕ | n ∈ S ∧ 6 ∣ n}

-- Proving the percentage of elements in T with respect to S is 16.67%
theorem percentage_divisible_by_6 : 
  (↑(T.card) : ℚ) / (S.card) * 100 = 16.67 := sorry

end percentage_divisible_by_6_l233_233640


namespace find_b_l233_233888

theorem find_b (a : ℝ) (A : ℝ) (B : ℝ) (b : ℝ)
  (ha : a = 5) 
  (hA : A = Real.pi / 6) 
  (htanB : Real.tan B = 3 / 4)
  (hsinB : Real.sin B = 3 / 5):
  b = 6 := 
by 
  sorry

end find_b_l233_233888


namespace nonneg_solutions_count_l233_233253

theorem nonneg_solutions_count :
  {x : ℝ | x^2 + 6 * x = 0 ∧ 0 ≤ x}.card = 1 :=
by
  sorry

end nonneg_solutions_count_l233_233253


namespace a_1966_eq_1024_l233_233171

-- Define the sequence a_n
def a : ℕ → ℕ
| 1     := 1996
| (k+1) := Nat.floor (Real.sqrt (Finset.sum (Finset.range k) (λ i, a (i+1))))

-- The theorem we need to prove
theorem a_1966_eq_1024 : a 1966 = 1024 := by
  -- Proof is omitted here
  sorry

end a_1966_eq_1024_l233_233171


namespace O2_tangent_to_O1_O2_intersects_O1_at_AB_l233_233438

-- Define the equations of the circles and their properties
def circle_eq (h k r : ℝ) : set (ℝ × ℝ) := {p | (p.1 - h)^2 + (p.2 - k)^2 = r}

-- Definitions based on conditions in step a)
def O1_eq := circle_eq 0 (-1) 4
def O2_center := (2, 1)

-- Proof problem 1: Show equivalent equations if O2 is tangent to O1
theorem O2_tangent_to_O1 :
  (∀ p : ℝ × ℝ, p ∈ O1_eq → p ∈ O1_eq) →
  (circle_eq 2 1 (2 * sqrt 2 - 2) = {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 12 - 8 * sqrt 2} ∧
   ∀ p : ℝ, p ∈ ({(p.1 + p.2 + 1 - 2 * sqrt 2 = 0)})) :=
sorry

-- Proof problem 2: Show possible equations of O2 such that |AB| = 2√2
theorem O2_intersects_O1_at_AB :
  (∀ p : ℝ × ℝ, p ∈ O1_eq ∧ p ∈ O1_eq ∃ AB_len,
    AB_len = 2 * sqrt 2 →
    ( circle_eq 2 1 2 = {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 4} ∨
      circle_eq 2 1 (sqrt 20) = {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 20})) :=
sorry

end O2_tangent_to_O1_O2_intersects_O1_at_AB_l233_233438


namespace percentage_increase_is_correct_l233_233046

-- Define the given conditions
def original_number : ℝ := 70
def final_number : ℝ := 105

-- Define the percentage increase formula
def percentage_increase (original final : ℝ) : ℝ := ((final - original) / original) * 100

-- State the theorem to prove
theorem percentage_increase_is_correct : percentage_increase original_number final_number = 50 := by
  sorry

end percentage_increase_is_correct_l233_233046


namespace percentage_divisible_by_6_l233_233637

theorem percentage_divisible_by_6 : 
  let numbers_less_than_or_equal_to_120 := (list.range 120).map (λ x, x + 1) in
  let divisible_by_6 := numbers_less_than_or_equal_to_120.filter (λ x, x % 6 = 0) in
  let percent := (divisible_by_6.length : ℚ) / 120 * 100 in
  percent = 16.67 :=
by 
  sorry

end percentage_divisible_by_6_l233_233637


namespace find_y_rotation_l233_233117

def rotate_counterclockwise (A : Point) (B : Point) (θ : ℝ) : Point := sorry
def rotate_clockwise (A : Point) (B : Point) (θ : ℝ) : Point := sorry

variable {A B C : Point}
variable {y : ℝ}

theorem find_y_rotation
  (h1 : rotate_counterclockwise A B 450 = C)
  (h2 : rotate_clockwise A B y = C)
  (h3 : y < 360) :
  y = 270 :=
sorry

end find_y_rotation_l233_233117


namespace area_of_octagon_l233_233194

noncomputable def octagon_area (BDEF : Type) (AB BC : ℝ) (is_square : Prop) (h : AB = 2) (h' : BC = 2) : ℝ :=
  (∑ i in (range 4), (1/2 * AB * BC)) - 8

theorem area_of_octagon
  (BDEF : Type) (AB BC : ℝ)
  (is_square : Prop) (kab : AB = 2) (kbc : BC = 2) :
  octagon_area BDEF AB BC is_square kab kbc = 16 + 8 * Real.sqrt 2 := by
  sorry

end area_of_octagon_l233_233194


namespace nonnegative_solutions_count_l233_233308

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.card = 1 := 
sorry

end nonnegative_solutions_count_l233_233308


namespace train_crossing_time_l233_233688

def train_length : ℝ := 140
def bridge_length : ℝ := 235.03
def speed_kmh : ℝ := 45

noncomputable def speed_mps : ℝ := speed_kmh * (1000 / 3600)
noncomputable def total_distance : ℝ := train_length + bridge_length

theorem train_crossing_time :
  (total_distance / speed_mps) = 30.0024 :=
by
  sorry

end train_crossing_time_l233_233688


namespace sofa_price_is_correct_l233_233924

def price_sofa (invoice_total armchair_price table_price : ℕ) (armchair_count : ℕ) : ℕ :=
  invoice_total - (armchair_price * armchair_count + table_price)

theorem sofa_price_is_correct
  (invoice_total : ℕ)
  (armchair_price : ℕ)
  (table_price : ℕ)
  (armchair_count : ℕ)
  (sofa_price : ℕ)
  (h_invoice : invoice_total = 2430)
  (h_armchair_price : armchair_price = 425)
  (h_table_price : table_price = 330)
  (h_armchair_count : armchair_count = 2)
  (h_sofa_price : sofa_price = 1250) :
  price_sofa invoice_total armchair_price table_price armchair_count = sofa_price :=
by
  sorry

end sofa_price_is_correct_l233_233924


namespace _l233_233107

variable (a b c : ℝ)
variable (ε : a > b)
variable (δ : b > 0)
variable (γ : c > 0)

def ellipse : Prop := ∀ (x y : ℝ), (x^2)/(a^2) + (y^2)/(b^2) = 1
def right_focus : Prop := ∃ F : ℝ × ℝ, F = (c, 0)
def line_intersects_ellipse (k : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, ∀ (x y : ℝ), y = k * (x - c) → (x^2)/(a^2) + ((k*(x - c))^2)/(b^2) = 1

def theorem : Prop :=
  ∀ (λ₁ λ₂ : ℝ) (k : ℝ)
  (PA_AF : ∀ {A : ℝ × ℝ} {P : ℝ × ℝ}, (P.1 - A.1) = λ₁ * (A.1 - c))
  (PB_BF : ∀ {B : ℝ × ℝ} {P : ℝ × ℝ}, (P.1 - B.1) = λ₂ * (B.1 - c)),
  λ₁ + λ₂ = - (2 * a^2) / b^2

lemma math_problem : theorem a b c := 
  by sorry

end _l233_233107


namespace train_speed_l233_233086

def distance := 300 -- meters
def time := 18 -- seconds

noncomputable def speed_kmh := 
  let speed_ms := distance / time -- speed in meters per second
  speed_ms * 3.6 -- convert to kilometers per hour

theorem train_speed : speed_kmh = 60 := 
  by
    -- The proof steps are omitted
    sorry

end train_speed_l233_233086


namespace concyclic_points_l233_233894

theorem concyclic_points
  (ABCD : Quadrilateral)
  (E F : Point)
  (BE_perpendicular_AB : is_perpendicular (line_through E B) (line_through A B))
  (CF_perpendicular_AE : is_perpendicular (line_through C F) (line_through A E))
  (circle_AC : Circle AC)
  (circle_AE : Circle AE)
  (P Q M N : Point)
  (P_on_circle_AC : P ∈ circle_AC)
  (Q_on_circle_AC : Q ∈ circle_AC)
  (M_on_circle_AE : M ∈ circle_AE)
  (N_on_circle_AE : N ∈ circle_AE)
  : are_concyclic P N Q M :=
sorry

end concyclic_points_l233_233894


namespace sphere_radius_eq_six_l233_233001

-- Defining the sphere's radius and its surface area formula
def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- Defining the cylinder's radius, height and its curved surface area formula
def cylinder_surface_area (r h : ℝ) : ℝ := 2 * Real.pi * r * h

-- Problem Conditions:
axiom sphere_eq_cylinder : 
  ∀ (r : ℝ), sphere_surface_area r = cylinder_surface_area 6 12

-- Stating the theorem to prove the radius of the sphere is 6 cm
theorem sphere_radius_eq_six :
  ∃ (r : ℝ), sphere_eq_cylinder r ∧ r = 6 := sorry

end sphere_radius_eq_six_l233_233001


namespace function_machine_output_l233_233911

-- Define the initial input
def input : ℕ := 12

-- Define the function machine steps
def functionMachine (x : ℕ) : ℕ :=
  if x * 3 <= 20 then (x * 3) / 2
  else (x * 3) - 2

-- State the property we want to prove
theorem function_machine_output : functionMachine 12 = 34 :=
by
  -- Skip the proof
  sorry

end function_machine_output_l233_233911


namespace tom_sleep_hours_l233_233484

-- Define initial sleep hours and increase fraction
def initial_sleep_hours : ℕ := 6
def increase_fraction : ℚ := 1 / 3

-- Define the function to calculate increased sleep
def increased_sleep_hours (initial : ℕ) (fraction : ℚ) : ℚ :=
  initial * fraction

-- Define the function to calculate total sleep hours
def total_sleep_hours (initial : ℕ) (increased : ℚ) : ℚ :=
  initial + increased

-- Theorem stating Tom's total sleep hours per night after the increase
theorem tom_sleep_hours (initial : ℕ) (fraction : ℚ) (increased : ℚ) (total : ℚ) :
  initial = initial_sleep_hours →
  fraction = increase_fraction →
  increased = increased_sleep_hours initial fraction →
  total = total_sleep_hours initial increased →
  total = 8 :=
by
  intros h_init h_frac h_incr h_total
  rw [h_init, h_frac] at h_incr
  rw [h_init, h_incr] at h_total
  sorry

end tom_sleep_hours_l233_233484


namespace solve_inequality_l233_233425

theorem solve_inequality (x : ℝ) :
  (1 / 2) ^ (x^2 - 2 * x + 3) < (1 / 2) ^ (2 * x^2 + 3 * x - 3) -> -6 < x ∧ x < 1 :=
by
  sorry

end solve_inequality_l233_233425


namespace greatest_prime_factor_341_l233_233586

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l233_233586


namespace original_number_eq_9999876_l233_233972

theorem original_number_eq_9999876 (x : ℕ) (h : x + 9876 = 10 * x + 9 + 876) : x = 999 :=
by {
  -- Simplify the equation and solve for x
  sorry
}

end original_number_eq_9999876_l233_233972


namespace nonnegative_solutions_eq1_l233_233244

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l233_233244


namespace capacities_correct_l233_233978

noncomputable def capacity_of_first_container : ℕ := 63
noncomputable def capacity_of_second_container : ℕ := 84
noncomputable def initial_water_first : ℕ := 49
noncomputable def initial_water_second : ℕ := 56

noncomputable def condition_1 (C1 C2 : ℕ) : Prop :=
  initial_water_second - (C1 - initial_water_first) = C2 / 2

noncomputable def condition_2 (C1 C2 : ℕ) : Prop :=
  initial_water_first - (C2 - initial_water_second) = C1 / 3

theorem capacities_correct :
  ∃ C1 C2 : ℕ, condition_1 C1 C2 ∧ condition_2 C1 C2 ∧ C1 = capacity_of_first_container ∧ C2 = capacity_of_second_container :=
by {
  -- We state the conditions given in the problem
  use 63, 84,
  split,
  -- Prove the first condition
  {
    unfold condition_1,
    norm_num,
  },
  split,
  -- Prove the second condition
  {
    unfold condition_2,
    norm_num,
  },
  split,
  -- Prove C1 = 63
  {
    refl,
  },
  -- Prove C2 = 84
  {
    refl,
  },
  sorry
}

end capacities_correct_l233_233978


namespace additional_passengers_per_carriage_l233_233005

noncomputable def carriages := 4
noncomputable def seats_per_carriage := 25
noncomputable def total_trains := 3
noncomputable def total_passengers := 420

theorem additional_passengers_per_carriage : 
    ∃ x : ℕ, total_trains * (carriages * seats_per_carriage + carriages * x) = total_passengers ∧ x = 10 :=
by
  let x := (total_passengers - total_trains * carriages * seats_per_carriage) / (total_trains * carriages)
  use x
  split
  sorry
  sorry

end additional_passengers_per_carriage_l233_233005


namespace expression_indeterminate_l233_233748

-- Given variables a, b, c, d which are real numbers
variables {a b c d : ℝ}

-- Statement asserting that the expression is indeterminate under given conditions
theorem expression_indeterminate
  (h : true) :
  ¬∃ k, (a^2 + b^2 - c^2 - 2 * b * d)/(a^2 + c^2 - b^2 - 2 * c * d) = k :=
sorry

end expression_indeterminate_l233_233748


namespace complex_set_representation_l233_233454

variable {Z : ℂ}
variable {a : ℝ} (h : ∃ a : ℝ, ∀ Z : ℂ, arg Z = a)

theorem complex_set_representation (Z : ℂ) (ha : arg Z = a) : ∃ w : ℂ, w = ∃ Z : ℂ, Z ∈ { w : ℂ | arg w = -2 * a } :=
by
  sorry

end complex_set_representation_l233_233454


namespace negation_of_universal_l233_233189

theorem negation_of_universal (p : ℝ → Prop) :
  (∀ x : ℝ, x > real.sin x) → ¬ (∀ x : ℝ, x > real.sin x) ↔ (∃ x : ℝ, x ≤ real.sin x) :=
by {
  intro h,
  apply classical.not_forall,
  exact exists_congr (λ x, (not_iff_comm.mp (lt_or_ge x (real.sin x)))).mp h,
}

end negation_of_universal_l233_233189


namespace gcd_yz_min_value_l233_233872

theorem gcd_yz_min_value (x y z : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) 
  (hxy_gcd : Nat.gcd x y = 224) (hxz_gcd : Nat.gcd x z = 546) : 
  Nat.gcd y z = 14 := 
sorry

end gcd_yz_min_value_l233_233872


namespace IvanPetrovich_daily_lessons_and_charity_l233_233040

def IvanPetrovichConditions (L k : ℕ) : Prop :=
  24 = 8 + 3*L + k ∧
  3000 * L * 21 + 14000 = 70000 + (7000 * k / 3)

theorem IvanPetrovich_daily_lessons_and_charity
  (L k : ℕ) (h : IvanPetrovichConditions L k) :
  L = 2 ∧ 7000 * k / 3 = 70000 := 
by
  sorry

end IvanPetrovich_daily_lessons_and_charity_l233_233040


namespace sum_of_sequence_l233_233661

theorem sum_of_sequence (n : ℕ) :
  let a (k : ℕ) := ∑ i in finset.range k, 2^i
  let s (n : ℕ) := ∑ k in finset.range n, (a (k + 1))
  s n = 2^(n + 1) - n - 2 :=
sorry

end sum_of_sequence_l233_233661


namespace percentage_of_integers_divisible_by_6_up_to_120_l233_233622

theorem percentage_of_integers_divisible_by_6_up_to_120 : 
  let total := 120
      divisible_by_6 := λ n, n % 6 = 0
      count := (list.range (total + 1)).countp divisible_by_6
      percentage := (count.toFloat / total.toFloat) * 100
  in percentage = 16.67 :=
by
  sorry

end percentage_of_integers_divisible_by_6_up_to_120_l233_233622


namespace max_vertices_seen_l233_233991

-- Define the properties of the cubes and the conditions
structure Cube (A1 A2 A3 A4 A5 A6 A7 A8 : ℝ × ℝ × ℝ) :=
(mk :: 
  distinct_vertices : ∀ (i j : Fin 8), i ≠ j → (A1, A2, A3, A4, A5, A6, A7, A8) i ≠ (A1, A2, A3, A4, A5, A6, A7, A8) j
  inside : ∀ (P : ℝ × ℝ × ℝ) (A : Fin 8), ¬ (P = A) → segment P. (A1, A2, A3, A4, A5, A6, A7, A8) A \∈ interior)

-- Define the point being able to see a vertex
def sees_vertex (P : ℝ × ℝ × ℝ) (cube : Cube A1 A2 A3 A4 A5 A6 A7 A8) (A : Fin 8) : Prop :=
∃ Q ∈ (segment P. (cube (A)) A), Q ∈ interior ∧ ∀ R ∈ interior, R ≠ cube A

-- The theorem statement
theorem max_vertices_seen (P : ℝ × ℝ × ℝ) (cube : Cube) :
  ∃ A : Fin 8, sees_vertex P cube A → ∀ B ≠ A, ¬ sees_vertex P cube B :=
begin
  sorry
end

end max_vertices_seen_l233_233991


namespace greatest_prime_factor_341_l233_233517

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l233_233517


namespace y_intercept_of_line_l233_233458

theorem y_intercept_of_line (m x1 y1 : ℝ) (x_intercept : x1 = 4) (y_intercept_at_x1_zero : y1 = 0) (m_value : m = -3) :
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ∧ x = 0 → y = b) ∧ b = 12 :=
by
  sorry

end y_intercept_of_line_l233_233458


namespace modulus_of_power_l233_233955

open Complex

theorem modulus_of_power (x y : ℚ) (h : (x ^ 2 + y ^ 2 = 1)) :
  ∀ n : ℤ, ∃ r : ℚ, |(x + y * Complex.I) ^ (2 * n) - 1| = r :=
by
  sorry

end modulus_of_power_l233_233955


namespace total_points_correct_l233_233725

-- Define the scores
def Marius (Darius : ℕ) : ℕ := Darius + 3
def Matt (Darius : ℕ) : ℕ := Darius + 5

-- Define the total points function
def total_points (Darius : ℕ) : ℕ :=
  Darius + Marius Darius + Matt Darius

-- Specific value for Darius's score
def Darius_score : ℕ := 10

-- The theorem that proves the total score is 38 given Darius's score
theorem total_points_correct :
  total_points Darius_score = 38 :=
by
  sorry

end total_points_correct_l233_233725


namespace sara_total_quarters_l233_233420

theorem sara_total_quarters (pennies quarters dimes euros : ℕ) (exchange_rate quarter_value : ℝ) :
    pennies = 100 → 
    quarters = 783 → 
    dimes = 450 → 
    euros = 250 → 
    exchange_rate = 1.18 → 
    quarter_value = 0.25 → 
    let usd_from_euros := euros * exchange_rate,
        new_quarters_from_usd := usd_from_euros / quarter_value,
        total_quarters := quarters + new_quarters_from_usd.to_nat in
    total_quarters = 1963 := 
by
  intros
  sorry

end sara_total_quarters_l233_233420


namespace probability_green_jelly_bean_l233_233665

theorem probability_green_jelly_bean :
  let red := 10
  let green := 9
  let yellow := 5
  let blue := 7
  let total := red + green + yellow + blue
  (green : ℚ) / (total : ℚ) = 9 / 31 := by
  sorry

end probability_green_jelly_bean_l233_233665


namespace greatest_prime_factor_of_341_l233_233568

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l233_233568


namespace find_constants_for_intersection_point_l233_233918

variables {A B C F G Q : Type}
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
variables [AddCommGroup F] [AddCommGroup G] [AddCommGroup Q]
variables [Module ℚ A] [Module ℚ B] [Module ℚ C] [Module ℚ F] [Module ℚ G] [Module ℚ Q]
variables [AffineSpace ℚ A] [AffineSpace ℚ B] [AffineSpace ℚ C]

-- Conditions: 
-- F lies on BC such that BF:FC = 2:1
def condition1 (F B C : A) : F = (2 / 3) • C + (1 / 3) • B := sorry

-- G lies on AC such that AG:GC = 3:2
def condition2 (G A C : A) : G = (2 / 5) • A + (3 / 5) • C := sorry

-- Q is the intersection of lines BG and AF
def intersection (Q B G A F : A) : Q = (2 / 3) • F + (1 / 3) • B ∧ Q = (3 / 5) • G + (2 / 5) • A := sorry

-- Prove that the constants x, y, z for Q are (2/5, 1/3, 4/9)
theorem find_constants_for_intersection_point (A B C F G Q : A)
  (cond1 : condition1 F B C)
  (cond2 : condition2 G A C)
  (intersect : intersection Q B G A F) : 
  ∃ x y z : ℚ, Q = x • A + y • B + z • C ∧ x = 2/5 ∧ y = 1/3 ∧ z = 4/9 :=
sorry

end find_constants_for_intersection_point_l233_233918


namespace greatest_prime_factor_of_341_l233_233551

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l233_233551


namespace coefficient_x2_in_expansion_l233_233350

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Statement to prove the coefficient of the x^2 term in (x + 1)^42 is 861
theorem coefficient_x2_in_expansion :
  (binomial 42 2) = 861 := by
  sorry

end coefficient_x2_in_expansion_l233_233350


namespace sum_fixed_points_pow_four_l233_233174

def fixed_points (σ : Perm (Fin 2013)) : Nat := 
  (Finset.univ.filter (λ k => σ k = k)).card

def S : Finset (Perm (Fin 2013)) := Finset.univ

theorem sum_fixed_points_pow_four : 
  ∑ σ in S, (fixed_points σ)^4 = 15 * Nat.factorial 2013 := 
by
  sorry

end sum_fixed_points_pow_four_l233_233174


namespace nonnegative_solutions_count_l233_233262

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.finite.to_finset.card = 1 :=
by
  sorry

end nonnegative_solutions_count_l233_233262


namespace true_statements_about_f_l233_233791

noncomputable def f : ℝ → ℝ :=
λ x, if x ∈ set.Ico (-1 : ℝ) 1 then x else f (x - 2 * ⌊(x + 1) / 2⌋ )

def g (x : ℝ) : ℝ := 1 / x

theorem true_statements_about_f :
  (∀ x : ℝ, f (x + 2) = f x) ∧
  (∀ k : ℤ, f (2 * k) = 0) ∧
  (∃! x : ℝ, x ∈ set.Ico (-3 : ℝ) 3 ∧ f x = g x) :=
by
  -- Proof goes here
  sorry

end true_statements_about_f_l233_233791


namespace nonnegative_solutions_count_l233_233265

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.finite.to_finset.card = 1 :=
by
  sorry

end nonnegative_solutions_count_l233_233265


namespace driving_speed_ratio_l233_233142

theorem driving_speed_ratio
  (x : ℝ) (y : ℝ)
  (h1 : y = 2 * x) :
  y / x = 2 := by
  sorry

end driving_speed_ratio_l233_233142


namespace greatest_prime_factor_of_341_l233_233572

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l233_233572


namespace trailing_zeros_product_s_l233_233954

-- Define s(n) as the sum of the digits of n
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the product P
def product_s : ℕ :=
  (Finset.range 100).prod (λ n, sum_of_digits (n + 1))

-- Define the number of trailing zeros in a number
def trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else n.factorization 10

theorem trailing_zeros_product_s : trailing_zeros product_s = 19 := 
  sorry

end trailing_zeros_product_s_l233_233954


namespace triangle_ratio_sum_eq_19_24_l233_233357

open_locale classical

variables {A B C D E F : Type} [IncidencePlane A Type]
variables {BD DC AE EB : ℝ}
variables {D_in_BC : BC_segment D}
variables {E_in_AB : AB_segment E}
variables {intersection_AD_CE : AD_intersect_CE F}

noncomputable def ratio_sum : ℝ :=
  let BD_DC_ratio := (2 : ℝ) in
  let AE_EB_ratio := (3 : ℝ) in
  have h₁ : BD / DC = BD_DC_ratio, from sorry,
  have h₂ : AE / EB = AE_EB_ratio, from sorry,
  let EF := 1 / (BD_DC_ratio + 1) in
  let AF := (BD_DC_ratio + 1) / (AE_EB_ratio + 1) in
  EF / (1 - EF) + AF / (BD_DC_ratio + 1 - AF)

theorem triangle_ratio_sum_eq_19_24 :
  ratio_sum = 19 / 24 :=
sorry

end triangle_ratio_sum_eq_19_24_l233_233357


namespace triangle_angle_α_l233_233917

theorem triangle_angle_α
  (A B C D E : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
  (h1 : ∀ (x y : A), x = y → True)
  (h2 : ∀ (x y : B), x = y → True)
  (h3 : ∀ (x y : C), x = y → True)
  (AB AC : ℝ)
  (hAB_AC : AB = AC)
  (α β γ : ℝ)
  (angle_ACD : γ = 20)
  (angle_ABE : β = 30)
  : α = 30 :=
begin
  sorry
end

end triangle_angle_α_l233_233917


namespace no_n_for_equal_sums_l233_233379

theorem no_n_for_equal_sums (n : ℕ) (h : n ≠ 0) :
  let s1 := (3 * n^2 + 7 * n) / 2
  let s2 := (3 * n^2 + 37 * n) / 2
  s1 ≠ s2 :=
by
  let s1 := (3 * n^2 + 7 * n) / 2
  let s2 := (3 * n^2 + 37 * n) / 2
  sorry

end no_n_for_equal_sums_l233_233379


namespace greatest_prime_factor_341_l233_233531

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l233_233531


namespace general_formula_a_n_formula_T_n_l233_233933

-- Given condition of the sequence sum
def S_n (n : ℕ) : ℕ := n^2 + n

-- General formula for the sequence a_n
def a_n (n : ℕ) : ℕ := 2 * n

-- Prove the general formula for a_n
theorem general_formula_a_n (n : ℕ)  :
  S_n n - S_n (n - 1) = 2 * n := by
  sorry

-- Definition of T_n based on a_n
def T_n (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, a_n (i + 1) * 3^(a_n (i + 1)))

-- Prove the given formula for T_n
theorem formula_T_n (n : ℕ) :
  T_n n = (1 / 32 : ℝ) * ((8 * n - 1) * 9^(n + 1) + 9) := by
  have H : T_n n = 2 * (Finset.range n).sum (λ i, (i + 1) * 9^(i + 1)) := by
    unfold T_n
    rw [a_n]
    refl
  sorry

end general_formula_a_n_formula_T_n_l233_233933


namespace ellipse_bisector_x_bound_l233_233213

theorem ellipse_bisector_x_bound {a b x₀ : ℝ} (h : a > b ∧ b > 0) 
  (ellipse : ∀ {x y : ℝ}, ∀ p ∈ set_of (λ p, (p.1 : ℝ)^2 / a^2 + (p.2 : ℝ)^2 / b^2 = 1), 
      (∃ x₁ y₁ x₂ y₂ : ℝ, (x₁, y₁) ∈ p ∧ (x₂, y₂) ∈ p ∧ let m₁ := x₁ + x₂,
                                                 m₂ := y₁ + y₂,
                                                 k := (- (x₂ - x₁) / (y₂ - y_1)),
                                                 bisector_eq := (λ x, -k * (x - m₁/2) + m₂/2),
                                                 in (x₀, 0) ∈ bisector_eq)) :
  -((a^2 - b^2) / a) < x₀ ∧ x₀ < ((a^2 - b^2) / a) :=
sorry

end ellipse_bisector_x_bound_l233_233213


namespace max_elements_in_T_l233_233369

universe u

def is_mod_5_pair_sum (a b : ℕ) : Prop :=
  (a + b) % 5 = 0

def is_valid_set (s : Finset ℕ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x ≠ y → ¬ is_mod_5_pair_sum x y

def candidate_set : Finset ℕ :=
  (Finset.range 26).filter (λ n, n > 0)

theorem max_elements_in_T : ∃ (T : Finset ℕ), is_valid_set T ∧ T.card = 5 :=
by 
  use {1, 2, 3, 4, 5}
  split
  {
    intros x y hx hy hneq
    repeat { cases hx <|> cases hy }
    ;
    norm_num,
  }
  norm_num

end max_elements_in_T_l233_233369


namespace math_exam_110_l233_233898

open Real

noncomputable def normal_distribution (μ : ℝ) (σ : ℝ) (x : ℝ) : ℝ := 
1 / (σ * sqrt (2 * π)) * exp (- (x - μ) ^ 2 / (2 * σ ^ 2))

theorem math_exam_110 {n : ℕ} (h1: n = 1000)
  (mean : ℝ) (variance : ℝ)
  (h2: mean = 90)
  (h3: ∀ x: ℝ, normal_distribution mean (sqrt variance) x ∈ set.Icc 70 110 = 3 / 5) : 
  ∃ s : ℝ, s ≥ 110 ∧ s ≈ 200 :=
begin
  sorry
end

end math_exam_110_l233_233898


namespace find_a4_l233_233798

variable (a : ℕ → ℤ)

def S (n : ℕ) : ℤ := (n * (a 1 + a n)) / 2

theorem find_a4 (h₁ : S 5 = 25) (h₂ : a 2 = 3) : a 4 = 7 := by
  sorry

end find_a4_l233_233798


namespace similar_triangles_length_eq_l233_233657

theorem similar_triangles_length_eq
  (PQ QR : ℝ) (ST : ℝ) (PQR_sim_STU : similar (triangle P Q R) (triangle S T U))
  (hPQ : PQ = 10) (hQR : QR = 15) (hST : ST = 8) :
  ∃ TU : ℝ, TU = 12 :=
by
  sorry

end similar_triangles_length_eq_l233_233657


namespace count_divisible_by_75_l233_233315

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem count_divisible_by_75 :
  let f (a b : ℕ) := a * 1000 + b * 100 + 25
  (nat| a b : ℕ, is_digit a ∧ is_digit b ∧ (a + b + 7) % 3 = 0) = 33 := sorry

end count_divisible_by_75_l233_233315


namespace total_points_l233_233723

theorem total_points (darius_score marius_score matt_score total_points : ℕ) 
    (h1 : darius_score = 10) 
    (h2 : marius_score = darius_score + 3) 
    (h3 : matt_score = darius_score + 5) 
    (h4 : total_points = darius_score + marius_score + matt_score) : 
    total_points = 38 :=
by sorry

end total_points_l233_233723


namespace f_compose_one_ninth_l233_233821

noncomputable def f : ℝ → ℝ := 
  λ x, if x ≤ 0 then 2^x else Real.log 3 x

theorem f_compose_one_ninth : f (f (1 / 9)) = 1 / 4 :=
by
  sorry

end f_compose_one_ninth_l233_233821


namespace average_salary_l233_233654

theorem average_salary (a b c d e : ℕ) (h1 : a = 8000) (h2 : b = 5000) (h3 : c = 16000) (h4 : d = 7000) (h5 : e = 9000) :
  (a + b + c + d + e) / 5 = 9000 :=
by
  sorry

end average_salary_l233_233654


namespace problem1_problem2_problem3_l233_233144

-- Statement for the first problem
theorem problem1 (x : ℚ) : 
  16 * (6 * x - 1) * (2 * x - 1) * (3 * x + 1) * (x - 1) + 25 = (24 * x^2 - 16 * x - 3)^2 :=
by sorry

-- Statement for the second problem
theorem problem2 (x : ℚ) : 
  (6 * x - 1) * (2 * x - 1) * (3 * x - 1) * (x - 1) + x^2 = (6 * x^2 - 6 * x + 1)^2 :=
by sorry

-- Statement for the third problem
theorem problem3 (x : ℚ) : 
  (6 * x - 1) * (4 * x - 1) * (3 * x - 1) * (x - 1) + 9 * x^4 = (9 * x^2 - 7 * x + 1)^2 :=
by sorry

end problem1_problem2_problem3_l233_233144


namespace teachers_can_sit_in_middle_l233_233777

-- Definitions for the conditions
def num_students : ℕ := 4
def num_teachers : ℕ := 3
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def permutations (n r : ℕ) : ℕ := factorial n / factorial (n - r)

-- Definition statements
def num_ways_teachers : ℕ := permutations num_teachers num_teachers
def num_ways_students : ℕ := permutations num_students num_students

-- Main theorem statement
theorem teachers_can_sit_in_middle : num_ways_teachers * num_ways_students = 144 := by
  -- Calculation goes here but is omitted for brevity
  sorry

end teachers_can_sit_in_middle_l233_233777


namespace coordinates_of_point_P_l233_233328

theorem coordinates_of_point_P
  (θ : ℝ) (r : ℝ) (P : ℝ × ℝ)
  (hθ : θ = -π / 4)
  (hr : r = 2)
  (hP : ∀ (x y : ℝ), P = (x, y) → (sin θ = y / r) ∧ (cos θ = x / r)) :
  P = (real.sqrt 2, -real.sqrt 2) :=
by
  sorry

end coordinates_of_point_P_l233_233328


namespace part1_part2_l233_233795

noncomputable def a (n : ℕ+) : ℤ := 6 * 2 ^ (n - 1) - 3
def S (n : ℕ+) : ℤ := 2 * a n - 3 * n

theorem part1 (h1 : S 1 = a 1) (h2 : S 2 = a 1 + a 2) (h3 : S 3 = a 1 + a 2 + a 3) :
  a 1 = 3 ∧ a 2 = 9 ∧ a 3 = 21 :=
by
  sorry

theorem part2 (exists_lambda : ∃ λ, (a 2 + λ) ^ 2 = ((a 1 + λ) * (a 3 + λ))) :
  ∃ λ, λ = 3 ∧ a n = 6 * 2 ^ (n - 1) - 3 :=
by
  sorry

end part1_part2_l233_233795


namespace lines_through_integer_points_l233_233218

-- Define the conditions.
def line (a b x y : ℤ) := a * x + b * y = 2017
def circle (x y : ℤ) := x * x + y * y = 100

-- State the theorem.
theorem lines_through_integer_points : 
    (∃ (a b x y : ℤ), line a b x y ∧ circle x y ∧ 
     ∃ (x y : ℤ), (x * x + y * y = 100) ∧ (a * x + b * y = 2017) ) → 
    ∃ (n : ℕ), n = 24 := 
by
    sorry

end lines_through_integer_points_l233_233218


namespace percentage_of_integers_divisible_by_6_up_to_120_l233_233619

theorem percentage_of_integers_divisible_by_6_up_to_120 : 
  let total := 120
      divisible_by_6 := λ n, n % 6 = 0
      count := (list.range (total + 1)).countp divisible_by_6
      percentage := (count.toFloat / total.toFloat) * 100
  in percentage = 16.67 :=
by
  sorry

end percentage_of_integers_divisible_by_6_up_to_120_l233_233619


namespace evaluate_expression_at_two_l233_233424

theorem evaluate_expression_at_two : 
  let expr := (2 * 2 - 1) ^ 2 + (2 + 3) * (2 - 3) - 4 * (2 - 1) in
  expr = 0 := 
by
  let x := 2
  let expr := (2 * x - 1) ^ 2 + (x + 3) * (x - 3) - 4 * (x - 1)
  sorry

end evaluate_expression_at_two_l233_233424


namespace measure_angle_BFE_l233_233356

theorem measure_angle_BFE (A B C E F : Type)
  [triangle : triangle A B C]
  (hE : lies_on E B C)
  (h_symmetry : symmetric E B C)
  (h_angle_bisector : angle_bisector B F A C)
  (h90 : angle_measure B A C = 90) :
  angle_measure B F E = 67.5 :=
sorry

end measure_angle_BFE_l233_233356


namespace compost_loading_time_l233_233146

theorem compost_loading_time :
  let steven_rate := 75 -- pounds per minute
  let darrel_rate := 10 -- pounds per minute
  let combined_rate := steven_rate + darrel_rate -- pounds per minute
  let total_compost := 2550 -- pounds
  (total_compost / combined_rate) = 30 :=
by {
  -- Definitions
  let steven_rate := 75 -- pounds per minute
  let darrel_rate := 10 -- pounds per minute
  let combined_rate := steven_rate + darrel_rate -- pounds per minute
  let total_compost := 2550 -- pounds
  -- Proof
  show (total_compost / combined_rate) = 30,
  sorry
}

end compost_loading_time_l233_233146


namespace angle_measure_of_E_l233_233341

theorem angle_measure_of_E (E F G H : ℝ) 
  (h1 : E = 3 * F) 
  (h2 : E = 4 * G) 
  (h3 : E = 6 * H) 
  (h_sum : E + F + G + H = 360) : 
  E = 206 := 
by 
  sorry

end angle_measure_of_E_l233_233341


namespace distribute_ways_l233_233317

/-- There are 5 distinguishable balls and 4 distinguishable boxes.
The total number of ways to distribute these balls into the boxes is 1024. -/
theorem distribute_ways : (4 : ℕ) ^ (5 : ℕ) = 1024 := by
  sorry

end distribute_ways_l233_233317


namespace greatest_prime_factor_341_l233_233587

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l233_233587


namespace train_speed_l233_233080

theorem train_speed (length_m : ℕ) (time_s : ℕ) (length_km : ℝ) (time_hr : ℝ) 
(length_conversion : length_km = (length_m : ℝ) / 1000)
(time_conversion : time_hr = (time_s : ℝ) / 3600)
(speed : ℝ) (speed_formula : speed = length_km / time_hr) :
  length_m = 300 → time_s = 18 → speed = 60 :=
by
  intros h1 h2
  rw [h1, h2] at *
  simp [length_conversion, time_conversion, speed_formula]
  norm_num
  sorry

end train_speed_l233_233080


namespace cos_675_eq_sqrt2_div_2_l233_233711

theorem cos_675_eq_sqrt2_div_2 : Real.cos (675 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by 
  sorry

end cos_675_eq_sqrt2_div_2_l233_233711


namespace rationalize_denominator_l233_233418

theorem rationalize_denominator :
  let A := 5
  let B := 2
  let C := 1
  let D := 4
  A + B + C + D = 12 :=
by
  sorry

end rationalize_denominator_l233_233418


namespace real_roots_exists_of_a_ne_zero_real_roots_exists_of_a_ne_one_l233_233772

theorem real_roots_exists_of_a_ne_zero (a : ℝ) : 
  (a ≠ 0) → (a < -5 - 2*Real.sqrt 6 ∨ 2*Real.sqrt 6 - 5 < a ∧ a < 0 ∨ a > 0) ↔ 
  (∃ x : ℝ, a*x^2 + (a+1)*x - 2 = 0) := sorry

theorem real_roots_exists_of_a_ne_one (a : ℝ) : 
  (a ≠ 1) → (a < 1 ∨ (1 < a ∧ a < 3) ∨ a > 3) ↔ 
  (∃ x : ℝ, (1 - a)*x^2 + (a + 1)*x - 2 = 0) := sorry

end real_roots_exists_of_a_ne_zero_real_roots_exists_of_a_ne_one_l233_233772


namespace train_speed_is_60_kmph_l233_233076

-- Define the conditions
def length_of_train : ℕ := 300 -- in meters
def time_to_cross_pole : ℕ := 18 -- in seconds

-- Define the conversions
def meters_to_kilometers (m : ℕ) : ℝ := m / 1000.0
def seconds_to_hours (s : ℕ) : ℝ := s / 3600.0

-- Define the speed calculation
def speed_km_per_hr (distance_km : ℝ) (time_hr : ℝ) : ℝ := distance_km / time_hr

-- Prove that the speed of the train is 60 km/hr
theorem train_speed_is_60_kmph :
  speed_km_per_hr (meters_to_kilometers length_of_train) (seconds_to_hours time_to_cross_pole) = 60 := 
  by
    sorry

end train_speed_is_60_kmph_l233_233076


namespace problem_statement_l233_233786

variable (n k : ℕ)

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

axiom nat_star (n : ℕ) : n ∈ finset.range (n + 1)

theorem problem_statement (h₁ : nat_star n ∧ nat_star k) (h₂ : k ≤ n) (h₃ : binomial_coefficient n k = binomial_coefficient (n-1) (k-1)) :
  (finset.sum (finset.range (n + 1)) (λ k, (k * binomial_coefficient n k))) = n * 2^(n-1) ∧ 
  (finset.sum (finset.range (n + 1)) (λ k, (k^2 * binomial_coefficient n k))) = n * (n + 1) * 2^(n-2) :=
sorry

end problem_statement_l233_233786


namespace probability_composite_is_correct_l233_233873

noncomputable def probability_composite : ℚ :=
  1 - (25 / (8^6))

theorem probability_composite_is_correct :
  probability_composite = 262119 / 262144 :=
by
  sorry

end probability_composite_is_correct_l233_233873


namespace percentage_of_girls_l233_233461

theorem percentage_of_girls (B G : ℕ) (h1 : B + G = 400) (h2 : B = 80) :
  (G * 100) / (B + G) = 80 :=
by sorry

end percentage_of_girls_l233_233461


namespace not_all_odd_l233_233108

theorem not_all_odd (a₁ a₂ a₃ a₄ a₅ b : ℤ)
  (h : a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 = b^2) :
  ¬ (odd a₁ ∧ odd a₂ ∧ odd a₃ ∧ odd a₄ ∧ odd a₅ ∧ odd b) :=
by sorry

end not_all_odd_l233_233108


namespace train_speed_l233_233079

theorem train_speed (length_m : ℕ) (time_s : ℕ) (length_km : ℝ) (time_hr : ℝ) 
(length_conversion : length_km = (length_m : ℝ) / 1000)
(time_conversion : time_hr = (time_s : ℝ) / 3600)
(speed : ℝ) (speed_formula : speed = length_km / time_hr) :
  length_m = 300 → time_s = 18 → speed = 60 :=
by
  intros h1 h2
  rw [h1, h2] at *
  simp [length_conversion, time_conversion, speed_formula]
  norm_num
  sorry

end train_speed_l233_233079


namespace quadratic_rewrite_ab_value_l233_233968

theorem quadratic_rewrite_ab_value:
  ∃ a b c : ℤ, (∀ x: ℝ, 16*x^2 + 40*x + 18 = (a*x + b)^2 + c) ∧ a * b = 20 :=
by
  -- We'll add the definitions derived from conditions here
  sorry

end quadratic_rewrite_ab_value_l233_233968


namespace planes_count_l233_233403

-- Define the conditions as given in the problem.
def total_wings : ℕ := 90
def wings_per_plane : ℕ := 2

-- Define the number of planes calculation based on conditions.
def number_of_planes : ℕ := total_wings / wings_per_plane

-- Prove that the number of planes is 45.
theorem planes_count : number_of_planes = 45 :=
by 
  -- The proof steps are omitted as specified.
  sorry

end planes_count_l233_233403


namespace nonnegative_solutions_count_l233_233261

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.finite.to_finset.card = 1 :=
by
  sorry

end nonnegative_solutions_count_l233_233261


namespace max_isosceles_triangles_l233_233398

theorem max_isosceles_triangles (points_on_line : Finset ℝ) (O : ℝ) (h : points_on_line.card = 100) :
  ∃ T : Finset (Finset ℝ), (∀ t ∈ T, t.card = 3 ∧ ∃ A B C ∈ t, (A = B ∨ A = C ∨ B = C)) ∧ T.card = 150 :=
sorry

end max_isosceles_triangles_l233_233398


namespace range_of_a_l233_233961

theorem range_of_a (a : ℝ) :
  (let f := λ x, if x < 0 then 2^x - 3 else real.sqrt (x + 1) in f a > 1) →
  a ∈ set.Ioi 0 :=
begin
  sorry
end

end range_of_a_l233_233961


namespace nonnegative_solution_count_l233_233272

theorem nonnegative_solution_count : 
  ∃! x : ℝ, x^2 = -6 * x ∧ 0 ≤ x := 
begin
  sorry
end

end nonnegative_solution_count_l233_233272


namespace number_of_nonnegative_solutions_l233_233240

-- Define the equation x^2 = -6x
def equation (x : ℝ) : Prop := x^2 = -6 * x

-- Define the condition for a nonnegative solution
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Define the main theorem to prove the number of nonnegative solutions
theorem number_of_nonnegative_solutions : 
  {x : ℝ // equation x ∧ nonnegative x}.to_finset.card = 1 :=
sorry

end number_of_nonnegative_solutions_l233_233240


namespace simplify_trig_identity_l233_233423

theorem simplify_trig_identity (α β : ℝ) : 
  (Real.cos (α + β) * Real.cos β + Real.sin (α + β) * Real.sin β) = Real.cos α :=
by
  sorry

end simplify_trig_identity_l233_233423


namespace nonnegative_solutions_eq1_l233_233247

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l233_233247


namespace power_function_range_a_l233_233208

theorem power_function_range_a (f : ℝ → ℝ) (a : ℝ) 
  (h₁ : f 4 = 1 / 2)
  (h₂ : ∀ x, f x = x ^ (-1 / 4))
  (h₃ : f (a + 1) < f (10 - 2 * a))
  : 3 < a ∧ a < 5 :=
by
  sorry

end power_function_range_a_l233_233208


namespace smaller_angle_clock_3_40_l233_233848

theorem smaller_angle_clock_3_40 : 
  let minute_pos := 240
  let hour_pos := 90 + (40 / 60) * 30
  let smaller_angle := abs (minute_pos - hour_pos)
  minute_pos = 240 ∧ hour_pos = 110 -> smaller_angle = 130 :=
by
  sorry

end smaller_angle_clock_3_40_l233_233848


namespace intersection_eq_l233_233841

variable M : Set Int := Set.ofList [-2, 0, 2, 4]
variable N : Set Int := { x | x^2 < 9 }

theorem intersection_eq : M ∩ N = Set.ofList [-2, 0, 2] := by
  sorry

end intersection_eq_l233_233841


namespace greatest_prime_factor_341_l233_233522

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l233_233522


namespace nonnegative_solutions_count_l233_233311

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.card = 1 := 
sorry

end nonnegative_solutions_count_l233_233311


namespace annulus_area_of_tangent_segments_l233_233789

theorem annulus_area_of_tangent_segments (r : ℝ) (l : ℝ) (region_area : ℝ) 
  (h_rad : r = 3) (h_len : l = 6) : region_area = 9 * Real.pi :=
sorry

end annulus_area_of_tangent_segments_l233_233789


namespace expression_to_diophantine_l233_233407

theorem expression_to_diophantine (x : ℝ) (y : ℝ) (n : ℕ) :
  (∃ (A B : ℤ), (x - y) ^ (2 * n + 1) = (A * x - B * y) ∧ (1969 : ℤ) * A^2 - (1968 : ℤ) * B^2 = 1) :=
sorry

end expression_to_diophantine_l233_233407


namespace base_five_equivalent_of_156_is_1111_l233_233511

theorem base_five_equivalent_of_156_is_1111 : nat_to_base 5 156 = [1, 1, 1, 1] := 
sorry

end base_five_equivalent_of_156_is_1111_l233_233511


namespace evaluate_fraction_sum_l233_233196

theorem evaluate_fraction_sum (a b c : ℝ) (h : a ≠ 40) (h_a : b ≠ 75) (h_b : c ≠ 85)
  (h_cond : (a / (40 - a)) + (b / (75 - b)) + (c / (85 - c)) = 8) :
  (8 / (40 - a)) + (15 / (75 - b)) + (17 / (85 - c)) = 40 := 
sorry

end evaluate_fraction_sum_l233_233196


namespace mildred_blocks_l233_233422

theorem mildred_blocks (initial_blocks found_blocks : ℕ) (h₁ : initial_blocks = 2) (h₂ : found_blocks = 84) : initial_blocks + found_blocks = 86 :=
by
  rw [h₁, h₂]
  -- complete the rest of the proof
  sorry

end mildred_blocks_l233_233422


namespace triangle_inequality_l233_233842

variables (a b c : ℝ) (n : ℕ)
def s : ℝ := (a + b + c) / 2

theorem triangle_inequality (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) (h4 : 1 ≤ n) :
    (a^n / (b + c)) + (b^n / (c + a)) + (c^n / (a + b)) ≥ (2 / 3)^(n - 2) * s a b c ^ (n - 1) :=
sorry

end triangle_inequality_l233_233842


namespace points_form_convex_quadrilateral_l233_233187

theorem points_form_convex_quadrilateral (points : Set Point) (h₁ : points.card = 5) 
(h₂ : ∀ p q r : Point, p ∈ points → q ∈ points → r ∈ points → p ≠ q → q ≠ r → p ≠ r → 
¬ collinear p q r) :
∃ (a b c d : Point), a ∈ points ∧ b ∈ points ∧ c ∈ points ∧ d ∈ points ∧ 
¬ collinear a b c ∧ ¬ collinear a b d ∧ ¬ collinear a c d ∧ ¬ collinear b c d ∧ convex_hull({a, b, c, d}) = {a, b, c, d} :=
sorry

end points_form_convex_quadrilateral_l233_233187


namespace percentage_divisible_by_6_l233_233641

-- Defining the sets S and T using Lean
def S := {n : ℕ | 1 ≤ n ∧ n ≤ 120}
def T := {n : ℕ | n ∈ S ∧ 6 ∣ n}

-- Proving the percentage of elements in T with respect to S is 16.67%
theorem percentage_divisible_by_6 : 
  (↑(T.card) : ℚ) / (S.card) * 100 = 16.67 := sorry

end percentage_divisible_by_6_l233_233641


namespace correct_propositions_l233_233097

variables {P A B C M : Type} [Nonempty P] [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty M]
noncomputable def equilateral_triangle (a b c : A) : Prop := a = b ∧ b = c

noncomputable def dihedral_equal (p a b c : Type) : Prop := sorry -- Define the property that the dihedral angles are equal

noncomputable def isosceles_lateral_faces (p a b c : Type) : Prop := sorry -- Define the property that all lateral faces are isosceles triangles

noncomputable def orthocenter_projection (a p b c : Type) : Prop := sorry -- Define the property for the orthocenter projection

noncomputable def equal_edges (p a b c : Type) : Prop := sorry -- Define the property that all edges are equal

noncomputable def equal_distances (a b m p c : Type) : Prop := sorry -- Define the property of equal distances for point M

theorem correct_propositions
  (eq_triangle : equilateral_triangle A B C)
  (prop1 : dihedral_equal P A B C)
  (prop2 : isosceles_lateral_faces P A B C)
  (prop3 : orthocenter_projection A P B C)
  (prop4 : equal_edges P A B C)
  (prop5 : equal_distances A B M P C) :
  (prop1 ∧ prop3 ∧ prop4 ∧ prop5) ∧ (¬prop2) :=
begin
  sorry
end

end correct_propositions_l233_233097


namespace parabola_equation_l233_233439

theorem parabola_equation (h_axis : ∃ p > 0, x = p / 2) :
  ∃ p > 0, y^2 = -2 * p * x :=
by 
  -- proof steps will be added here
  sorry

end parabola_equation_l233_233439


namespace evaluate_expression_l233_233320

theorem evaluate_expression (x : ℕ) (h : x = 5) : 2 * x ^ 2 + 3 = 53 :=
by {
  subst h,
  -- Proof will be here
  sorry,
}

end evaluate_expression_l233_233320


namespace incorrect_neg_p_l233_233027

theorem incorrect_neg_p (p : ∀ x : ℝ, x ≥ 1) : ¬ (∀ x : ℝ, x < 1) :=
sorry

end incorrect_neg_p_l233_233027


namespace nonnegative_solutions_eq_1_l233_233298

variable (x : ℝ)

theorem nonnegative_solutions_eq_1 : (x^2 = -6 * x) → (∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀) → (x ∈ {0}) :=
by
  assume (h1: x^2 = -6 * x)
  assume (h2: ∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀)
  sorry

end nonnegative_solutions_eq_1_l233_233298


namespace twenty_is_80_percent_of_what_number_l233_233492

theorem twenty_is_80_percent_of_what_number : ∃ y : ℕ, (20 : ℚ) / y = 4 / 5 ∧ y = 25 := by
  sorry

end twenty_is_80_percent_of_what_number_l233_233492


namespace queenie_earnings_782_l233_233866

def queenie_daily_earnings : ℕ := 150

def queenie_overtime_earnings (hours : ℕ) : ℕ :=
 match hours with
 | 0 => 0
 | 1 => 5
 | n + 1 => queenie_overtime_earnings n + 2
 
def total_overtime_earnings (hours : ℕ) : ℕ :=
list.sum (list.map queenie_overtime_earnings (list.range hours + 1))

-- Calculate Queenie's total earnings for 5 days with 4 hours of overtime
def queenie_total_earnings (days : ℕ) (overtime_hours : ℕ) : ℕ :=
(days * queenie_daily_earnings) + (total_overtime_earnings overtime_hours)

theorem queenie_earnings_782 : queenie_total_earnings 5 4 = 782 := by
  sorry

end queenie_earnings_782_l233_233866


namespace greatest_prime_factor_of_341_l233_233604

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l233_233604


namespace circles_intersect_if_and_only_if_l233_233844

noncomputable def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 10 * y + 1 = 0

noncomputable def circle2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 2 * y - m = 0

theorem circles_intersect_if_and_only_if (m : ℝ) :
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) ↔ -1 < m ∧ m < 79 :=
by {
  sorry
}

end circles_intersect_if_and_only_if_l233_233844


namespace area_of_shaded_L_shaped_region_l233_233149

theorem area_of_shaded_L_shaped_region (
  side_ABCD : ℕ := 6
  side_inner1 : ℕ := 1
  side_inner2 : ℕ := 2
  side_inner3 : ℕ := 3
) : 
  let area_total := side_ABCD * side_ABCD
  let area_inner1 := side_inner1 * side_inner1
  let area_inner2 := side_inner2 * side_inner2
  let area_inner3 := side_inner3 * side_inner3
  area_total - (area_inner1 + area_inner2 + area_inner3) = 22 :=
by
  sorry

end area_of_shaded_L_shaped_region_l233_233149


namespace decreasing_function_condition_l233_233215

theorem decreasing_function_condition :
  (∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 ≠ x2 → (x1 - x2) * ((1 / x1 - x1) - (1 / x2 - x2)) < 0) :=
by
  -- Proof outline goes here
  sorry

end decreasing_function_condition_l233_233215


namespace eva_total_marks_correct_l233_233737

-- Definitions based on conditions
def math_marks_second_sem : ℕ := 80
def arts_marks_second_sem : ℕ := 90
def science_marks_second_sem : ℕ := 90

def math_marks_first_sem : ℕ := math_marks_second_sem + 10
def arts_marks_first_sem : ℕ := arts_marks_second_sem - 15
def science_marks_first_sem : ℕ := science_marks_second_sem - (science_marks_second_sem / 3)

def total_marks_first_sem : ℕ := math_marks_first_sem + arts_marks_first_sem + science_marks_first_sem
def total_marks_second_sem : ℕ := math_marks_second_sem + arts_marks_second_sem + science_marks_second_sem

def total_marks_both_sems : ℕ := total_marks_first_sem + total_marks_second_sem

-- Theorem to be proved
theorem eva_total_marks_correct : total_marks_both_sems = 485 := by
  -- Here, we state that we need to prove the total marks sum up to 485
  sorry

end eva_total_marks_correct_l233_233737


namespace obtuse_triangle_circle_l233_233475

theorem obtuse_triangle_circle
    (points : Fin 100 → ℝ × ℝ)
    (h_distance : ∀ i j, dist (points i) (points j) ≤ 1)
    (h_obtuse : ∀ i j k, ∃ p q r, (p = points i ∨ p = points j ∨ p = points k) ∧ 
                                (q = points i ∨ q = points j ∨ q = points k) ∧
                                (r = points i ∨ r = points j ∨ r = points k) ∧
                                ∠pqr > π / 2 ∨ ∠pqr = π / 2 ∨ ∠prq > π / 2 ∨ ∠prq = π / 2 ∨ ∠qpr > π / 2 ∨ ∠qpr = π / 2) :
    ∃ center radius, radius = 0.5 ∧ ∀ i, dist (points i) center ≤ radius :=
    sorry

end obtuse_triangle_circle_l233_233475


namespace nonneg_solutions_count_l233_233256

theorem nonneg_solutions_count :
  {x : ℝ | x^2 + 6 * x = 0 ∧ 0 ≤ x}.card = 1 :=
by
  sorry

end nonneg_solutions_count_l233_233256


namespace greatest_prime_factor_of_341_l233_233535

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l233_233535


namespace quad_root_sum_product_l233_233804

theorem quad_root_sum_product (α β : ℝ) (h₁ : α ≠ β) (h₂ : α * α - 5 * α - 2 = 0) (h₃ : β * β - 5 * β - 2 = 0) : 
  α + β + α * β = 3 := 
by
  sorry

end quad_root_sum_product_l233_233804


namespace compost_loading_time_l233_233145

theorem compost_loading_time :
  let steven_rate := 75 -- pounds per minute
  let darrel_rate := 10 -- pounds per minute
  let combined_rate := steven_rate + darrel_rate -- pounds per minute
  let total_compost := 2550 -- pounds
  (total_compost / combined_rate) = 30 :=
by {
  -- Definitions
  let steven_rate := 75 -- pounds per minute
  let darrel_rate := 10 -- pounds per minute
  let combined_rate := steven_rate + darrel_rate -- pounds per minute
  let total_compost := 2550 -- pounds
  -- Proof
  show (total_compost / combined_rate) = 30,
  sorry
}

end compost_loading_time_l233_233145


namespace point_on_sphere_with_limited_visible_asteroids_l233_233686

theorem point_on_sphere_with_limited_visible_asteroids :
  ∀ (planet : Type) [sphere planet] (asteroids : set (point planet)) (n : ℕ),
  finite asteroids → card asteroids = 37 →
  ∃ (p : point planet), visible_asteroids_from p asteroids ≤ 17 :=
by
  -- Proof to be filled in later
  sorry

end point_on_sphere_with_limited_visible_asteroids_l233_233686


namespace nonnegative_solution_count_nonnegative_solution_count_unique_l233_233286

theorem nonnegative_solution_count (x : ℝ) :
  (x^2 = -6 * x) → (x ≥ 0) → x = 0 :=
by
  sorry

-- Prove that the number of nonnegative solutions is 1
theorem nonnegative_solution_count_unique :
  {x : ℝ // x ≥ 0 ∧ x^2 = -6 * x}.to_finset.card = 1 :=
by
  sorry

end nonnegative_solution_count_nonnegative_solution_count_unique_l233_233286


namespace part_a_part_b_l233_233387

open Real

noncomputable def f : ℝ → ℝ := sorry -- Define f and prove it's a distinct continuous function
noncomputable def g : ℝ → ℝ := sorry -- Define g and prove it's a distinct continuous function

-- Define the conditions
axiom f_cont : ContinuousOn f (Set.Icc 0 1)
axiom g_cont : ContinuousOn g (Set.Icc 0 1)
axiom f_pos : ∀ ⦃x : ℝ⦄, x ∈ (Set.Icc 0 1) → 0 < f x
axiom g_pos : ∀ ⦃x : ℝ⦄, x ∈ (Set.Icc 0 1) → 0 < g x
axiom f_neq_g : ∃ x ∈ (Set.Icc 0 1), f x ≠ g x
axiom int_fg_eq : ∫ x in 0..1, f x = ∫ x in 0..1, g x

-- Define the sequence x_n
noncomputable def x_n (n : ℕ) : ℝ :=
  ∫ x in 0..1, (f x)^(n+1) / (g x)^n

-- Part (a): Prove that \lim_{n \to \infty} x_n = \infty
theorem part_a : Tendsto (λ (n: ℕ), x_n n) atTop atTop :=
sorry

-- Part (b): Prove that the sequence (x_n)_{n ≥ 0} is monotone
theorem part_b : Monotone (x_n) :=
sorry

end part_a_part_b_l233_233387


namespace volume_of_ABCD_tetrahedron_l233_233349

noncomputable def tetrahedron_volume (a b d : ℝ) : ℝ :=
  1/6 * a * b * d

theorem volume_of_ABCD_tetrahedron :
  let AB := 1
  let CD := 3
  let distance := 2
  let angle := 60 * (π / 180)
  let volume := tetrahedron_volume AB CD (distance * sin angle)
  volume = sqrt 3 :=
by
  -- The proof would go here, we skip it using sorry
  sorry

end volume_of_ABCD_tetrahedron_l233_233349


namespace necessary_not_sufficient_condition_l233_233044

variable (x : ℝ)

theorem necessary_not_sufficient_condition :
  (−1 < x ∧ x < 0) → (2 * x < 1) ∧ ¬((2 * x < 1) → (−1 < x ∧ x < 0)) :=
by
  intro hx
  have h_necessary : (−1 < x) → (2 * x < 1) := sorry
  have h_not_sufficient : ¬((2 * x < 1) → (−1 < x)) := sorry
  exact ⟨h_necessary hx, h_not_sufficient⟩

end necessary_not_sufficient_condition_l233_233044


namespace right_triangle_x_value_l233_233343

variable (BM MA BC CA x h d : ℝ)

theorem right_triangle_x_value (BM MA BC CA x h d : ℝ)
  (h4 : BM + MA = BC + CA)
  (h5 : BM = x)
  (h6 : BC = h)
  (h7 : CA = d) :
  x = h * d / (2 * h + d) := 
sorry

end right_triangle_x_value_l233_233343


namespace remaining_surface_area_unchanged_l233_233733

noncomputable def original_cube_surface_area : Nat := 6 * 4 * 4

def corner_cube_surface_area : Nat := 3 * 2 * 2

def remaining_surface_area (original_cube_surface_area : Nat) (corner_cube_surface_area : Nat) : Nat :=
  original_cube_surface_area

theorem remaining_surface_area_unchanged :
  remaining_surface_area original_cube_surface_area corner_cube_surface_area = 96 := 
by
  sorry

end remaining_surface_area_unchanged_l233_233733


namespace integral_of_f_l233_233940

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 1 then sqrt (1 - x^2) else x^2 - 1

theorem integral_of_f :
  ∫ x in -1..2, f x = (Real.pi / 2) + (4 / 3) :=
by
  sorry

end integral_of_f_l233_233940


namespace greatest_prime_factor_of_341_l233_233547

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l233_233547


namespace parabola_focus_l233_233156

theorem parabola_focus (a b c : ℝ) (h_eq : ∀ x : ℝ, 2 * x^2 + 8 * x - 1 = a * (x + b)^2 + c) :
  ∃ focus : ℝ × ℝ, focus = (-2, -71 / 8) :=
sorry

end parabola_focus_l233_233156


namespace min_value_of_function_l233_233447

theorem min_value_of_function : ∃ x : ℝ, ∀ x : ℝ, x * (x + 1) * (x + 2) * (x + 3) ≥ -1 :=
by
  sorry

end min_value_of_function_l233_233447


namespace unique_solution_abs_eq_l233_233729

theorem unique_solution_abs_eq : ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| :=
by
  sorry

end unique_solution_abs_eq_l233_233729


namespace nonneg_solutions_count_l233_233255

theorem nonneg_solutions_count :
  {x : ℝ | x^2 + 6 * x = 0 ∧ 0 ≤ x}.card = 1 :=
by
  sorry

end nonneg_solutions_count_l233_233255


namespace greatest_prime_factor_341_l233_233576

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l233_233576


namespace sum_of_proper_divisors_540_l233_233138

-- Define the number and its prime factorization
def n := 540
def prime_factorization := (2^2, 3^3, 5)

-- Definition for the sum of the divisors of a number
def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := n.factorization.to_divisors in
  factors.sum

-- Define the proper divisors of the number
def sum_of_proper_divisors (n : ℕ) : ℕ :=
  sum_of_divisors(n) - n

-- The theorem that we're proving
theorem sum_of_proper_divisors_540 :
  sum_of_proper_divisors n = 1140 :=
by sorry

end sum_of_proper_divisors_540_l233_233138


namespace tom_sleep_increase_l233_233486

theorem tom_sleep_increase :
  ∀ (initial_sleep : ℕ) (increase_by : ℚ), 
  initial_sleep = 6 → 
  increase_by = 1/3 → 
  initial_sleep + increase_by * initial_sleep = 8 :=
by 
  intro initial_sleep increase_by h1 h2
  simp [*, add_mul, mul_comm]
  sorry

end tom_sleep_increase_l233_233486


namespace sum_of_x_coords_of_Q3_l233_233048

/-- Given a 50-gon Q1 with vertices' x-coordinates summing to 150, after sequential transformations
to form polygons Q2 and Q3, prove the sum of x-coordinates of vertices of Q3 equals 0. -/
theorem sum_of_x_coords_of_Q3 (x : Fin 50 → ℝ) :
  (∑ i, x i) = 150 →
  let Q2_x : Fin 50 → ℝ := λ i, (x i + x (i + 1) % 50) / 2 in
  let Q2_y : Fin 50 → ℝ := λ i, 0 in
  let Q3_x : Fin 50 → ℝ := λ i, - Q2_y i in
  (∑ i, Q3_x i) = 0 :=
by
  intros sum_x_coords_150 Q2_x Q2_y Q3_x
  sorry

end sum_of_x_coords_of_Q3_l233_233048


namespace last_digit_base_4_of_77_l233_233020

theorem last_digit_base_4_of_77 : (77 % 4) = 1 :=
by
  sorry

end last_digit_base_4_of_77_l233_233020


namespace neg_q_imp_neg_p_neg_q_not_imp_p_neg_p_is_necessary_not_sufficient_for_neg_q_l233_233378

def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem neg_q_imp_neg_p : (¬q x) → (¬p x) := sorry
theorem neg_q_not_imp_p : ¬(¬q x → p x) := sorry

theorem neg_p_is_necessary_not_sufficient_for_neg_q (x : ℝ) : 
  ((¬q x) → (¬p x)) ∧ ¬((¬q x) → p x) :=
by
  exact and.intro neg_q_imp_neg_p neg_q_not_imp_p

end neg_q_imp_neg_p_neg_q_not_imp_p_neg_p_is_necessary_not_sufficient_for_neg_q_l233_233378


namespace percentage_divisible_by_6_l233_233635

theorem percentage_divisible_by_6 : 
  let numbers_less_than_or_equal_to_120 := (list.range 120).map (λ x, x + 1) in
  let divisible_by_6 := numbers_less_than_or_equal_to_120.filter (λ x, x % 6 = 0) in
  let percent := (divisible_by_6.length : ℚ) / 120 * 100 in
  percent = 16.67 :=
by 
  sorry

end percentage_divisible_by_6_l233_233635


namespace greatest_prime_factor_341_l233_233515

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l233_233515


namespace triangle_area_is_150_l233_233902

variables (A B C H : Type)
variables [RightTriangle ABC]
variables (AB AC BC BH HC : ℝ)
variables (S : ℝ)

-- Conditions
def AB_length : AB = 15 := by sorry
def HC_length : HC = 16 := by sorry
def right_angle : RightTriangleABC ABC := by sorry
def BH_perp_AC : Perpendicular BH AC := by sorry

-- Proof statement
theorem triangle_area_is_150 : S = 150 :=
  begin 
    sorry 
  end

end triangle_area_is_150_l233_233902


namespace find_BF_l233_233650

variable {ℝ : Type} [LinearOrderedField ℝ]

structure Point :=
(x : ℝ) (y : ℝ)

def quadrilateral_ABCD (A B C D E F : Point) : Prop :=
  A.x = 0 ∧ A.y = 0 ∧ 
  B.x = 9 ∧ B.y = 5.4 ∧
  C.x = 12 ∧ C.y = 0 ∧
  D.x = 4 ∧ D.y = -6 ∧
  E.x = 4 ∧ E.y = 0 ∧
  F.x = 9 ∧ F.y = 0 ∧
  ∠A = 90 ∧ ∠C = 90 ∧
  (D.y - E.y) / (D.x - E.x) = ∞ ∧ -- DE is perpendicular to AC.
  (B.y - F.y) / (B.x - F.x) = ∞ -- BF is perpendicular to AC.

theorem find_BF (A B C D E F : Point) (h: quadrilateral_ABCD A B C D E F) :
  distance B F = 10.8 :=
sorry

end find_BF_l233_233650


namespace greatest_prime_factor_of_341_l233_233571

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l233_233571


namespace overtime_rate_is_correct_l233_233690

/-
Define the parameters:
ordinary_rate: Rate per hour for ordinary time in dollars
total_hours: Total hours worked in a week
overtime_hours: Overtime hours worked in a week
total_earnings: Total earnings for the week in dollars
-/

def ordinary_rate : ℝ := 0.60
def total_hours : ℝ := 50
def overtime_hours : ℝ := 8
def total_earnings : ℝ := 32.40

noncomputable def overtime_rate : ℝ :=
(total_earnings - ordinary_rate * (total_hours - overtime_hours)) / overtime_hours

theorem overtime_rate_is_correct :
  overtime_rate = 0.90 :=
by
  sorry

end overtime_rate_is_correct_l233_233690


namespace clark_discount_l233_233100

theorem clark_discount (price_per_part : ℕ) (number_of_parts : ℕ) (amount_paid : ℕ)
  (h1 : price_per_part = 80)
  (h2 : number_of_parts = 7)
  (h3 : amount_paid = 439) : 
  (number_of_parts * price_per_part) - amount_paid = 121 := by
  sorry

end clark_discount_l233_233100


namespace workdays_ride_l233_233481

-- Define the conditions
def work_distance : ℕ := 20
def weekend_ride : ℕ := 200
def speed : ℕ := 25
def hours_per_week : ℕ := 16

-- Define the question
def total_distance : ℕ := speed * hours_per_week
def distance_during_workdays : ℕ := total_distance - weekend_ride
def round_trip_distance : ℕ := 2 * work_distance

theorem workdays_ride : 
  (distance_during_workdays / round_trip_distance) = 5 :=
by
  sorry

end workdays_ride_l233_233481


namespace greatest_overlap_l233_233664

-- Defining the conditions based on the problem statement
def percentage_internet (n : ℕ) : Prop := n = 35
def percentage_snacks (m : ℕ) : Prop := m = 70

-- The theorem to prove the greatest possible overlap
theorem greatest_overlap (n m k : ℕ) (hn : percentage_internet n) (hm : percentage_snacks m) : 
  k ≤ 35 :=
by sorry

end greatest_overlap_l233_233664


namespace radian_to_degree_conversion_l233_233717

-- Define the condition as a constant
def radian_to_degree (x : ℝ) : ℝ := x * (180 / Real.pi)

-- Radian value to be converted
def rad_value := - (23 / 12) * Real.pi

-- Expected degree value after conversion
def expected_deg := -345

-- The Lean statement to prove
theorem radian_to_degree_conversion : radian_to_degree rad_value = expected_deg :=
by 
sorutorial

end radian_to_degree_conversion_l233_233717


namespace paint_fraction_left_l233_233057

theorem paint_fraction_left : 
  let initial_paint := 1 in
  let first_day_usage := 1/4 * initial_paint in
  let second_day_usage := 1/2 * (initial_paint - first_day_usage) in
  let third_day_usage := 1/3 * (initial_paint - first_day_usage - second_day_usage) in
  (initial_paint - first_day_usage - second_day_usage - third_day_usage) = 1/4 :=
by
  sorry

end paint_fraction_left_l233_233057


namespace combination_sum_l233_233120

-- We define the sum of the combination binomials
def sum_combinations (start : ℕ) (stop : ℕ) : ℕ :=
  (finset.range (stop - start + 1)).sum (λ i, nat.choose (start + i) i)

-- The main theorem we want to prove
theorem combination_sum : sum_combinations 2 19 = 1140 :=
by sorry

end combination_sum_l233_233120


namespace count_complex_numbers_l233_233871

theorem count_complex_numbers (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h : a + b ≤ 5) : 
  ∃ n, n = 10 := 
by
  sorry

end count_complex_numbers_l233_233871


namespace percentage_divisible_by_6_l233_233639

-- Defining the sets S and T using Lean
def S := {n : ℕ | 1 ≤ n ∧ n ≤ 120}
def T := {n : ℕ | n ∈ S ∧ 6 ∣ n}

-- Proving the percentage of elements in T with respect to S is 16.67%
theorem percentage_divisible_by_6 : 
  (↑(T.card) : ℚ) / (S.card) * 100 = 16.67 := sorry

end percentage_divisible_by_6_l233_233639


namespace greatest_prime_factor_of_341_l233_233570

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l233_233570


namespace number_of_nonnegative_solutions_l233_233233

-- Define the equation x^2 = -6x
def equation (x : ℝ) : Prop := x^2 = -6 * x

-- Define the condition for a nonnegative solution
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Define the main theorem to prove the number of nonnegative solutions
theorem number_of_nonnegative_solutions : 
  {x : ℝ // equation x ∧ nonnegative x}.to_finset.card = 1 :=
sorry

end number_of_nonnegative_solutions_l233_233233


namespace sprite_volume_l233_233056

theorem sprite_volume (maaza pepsi total_cans gcd_maaza_pepsi : ℕ) (H1 : maaza = 20) (H2 : pepsi = 144) (H3 : total_cans = 133) (H4 : nat.gcd maaza pepsi = 4) :
  ∃ sprite, sprite = 368 :=
begin
  use 92 * 4, -- number of cans for Sprite * volume per can
  norm_num,
end

end sprite_volume_l233_233056


namespace sampling_estimate_l233_233030

-- Define the condition: a small taste (sample) is taken from the soup.
def small_taste (sample: Type) : Prop := 
  sample = "small taste"

-- Define the whole population represented by the soup.
def whole_soup_population (population: Type) : Prop := 
  population = "entire soup"

-- The action represents sample estimation in statistics.
theorem sampling_estimate (sample population: Type) 
  (h1: small_taste sample) 
  (h2: whole_soup_population population) : 
  sample = "sample estimation to estimate the population" :=
sorry

end sampling_estimate_l233_233030


namespace area_of_isosceles_trapezoid_l233_233446

theorem area_of_isosceles_trapezoid (a b : ℝ) :
  let midpoint_projection_condition := true in -- Assuming the given condition holds
  area a b = (a + b) * Real.sqrt((a - b) * (a + 3 * b)) / 4 :=
by sorry

end area_of_isosceles_trapezoid_l233_233446


namespace range_of_a_l233_233882

theorem range_of_a 
  (a : ℝ)
  (h : ∀ x ∈ set.Icc (0 : ℝ) 2, x^3 + x^2 + a < 0) : 
  a < -12 := 
  sorry

end range_of_a_l233_233882


namespace equipment_total_cost_l233_233462

-- Definition of costs for each item of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80

-- Number of players
def num_players : ℕ := 16

-- Total cost for one player
def total_cost_one_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Total cost for all players
def total_cost_all_players : ℝ := total_cost_one_player * num_players

-- Theorem to prove
theorem equipment_total_cost : total_cost_all_players = 752 := by
  sorry

end equipment_total_cost_l233_233462


namespace sum_of_prime_factors_210630_l233_233616

theorem sum_of_prime_factors_210630 : (2 + 3 + 5 + 7 + 17 + 59) = 93 := by
  -- Proof to be provided
  sorry

end sum_of_prime_factors_210630_l233_233616


namespace trapezoid_y_square_l233_233368

noncomputable def isosceles_trapezoid (EF GH y : ℝ) :=
  (EFGH : Quadrilateral) (EF GH EFGH) ∧
  (EG = FH = y) ∧
  (circle_tangent_to_segments E G H O)

theorem trapezoid_y_square :
  ∀ (EF GH y : ℝ),
  isosceles_trapezoid EF GH y →
  EF = 100 →
  GH = 25 →
  circle_with_center_on_EF_tangent_to_EG_and_FH y →
  y^2 = 781.25 :=
begin
  sorry
end

end trapezoid_y_square_l233_233368


namespace nonneg_solutions_count_l233_233259

theorem nonneg_solutions_count :
  {x : ℝ | x^2 + 6 * x = 0 ∧ 0 ≤ x}.card = 1 :=
by
  sorry

end nonneg_solutions_count_l233_233259


namespace students_taking_either_geometry_or_history_but_not_both_l233_233476

theorem students_taking_either_geometry_or_history_but_not_both
    (students_in_both : ℕ)
    (students_in_geometry : ℕ)
    (students_only_in_history : ℕ)
    (students_in_both_cond : students_in_both = 15)
    (students_in_geometry_cond : students_in_geometry = 35)
    (students_only_in_history_cond : students_only_in_history = 18) :
    (students_in_geometry - students_in_both + students_only_in_history = 38) :=
by
  sorry

end students_taking_either_geometry_or_history_but_not_both_l233_233476


namespace solve_for_r_l233_233365

theorem solve_for_r : ∃ r : ℚ, 16 = 2^(5 * r + 1) ∧ r = 3 / 5 :=
by
  use 3 / 5
  split
  sorry

end solve_for_r_l233_233365


namespace odd_indexed_terms_geometric_sequence_l233_233354

open Nat

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (2 * n + 3) = r * a (2 * n + 1)

theorem odd_indexed_terms_geometric_sequence (b : ℕ → ℝ) (h : ∀ n, b n * b (n + 1) = 3 ^ n) :
  is_geometric_sequence b 3 :=
by
  sorry

end odd_indexed_terms_geometric_sequence_l233_233354


namespace maximum_good_subset_size_l233_233384

noncomputable theory

def Sn (n : ℕ) : set (vector bool (2^n)) := 
  {a | ∀ i, 1 ≤ i ∧ i ≤ 2^n → a.nth (i - 1) = tt ∨ a.nth (i - 1) = ff}

def d {n : ℕ} (a b : vector bool (2^n)) : ℕ :=
  finset.card (finset.filter (λ i, a.nth i ≠ b.nth i) (finset.range (2^n)))

def good_subset {n : ℕ} (A : set (vector bool (2^n))) : Prop :=
  ∀ ⦃a b⦄, a ∈ A → b ∈ A → a ≠ b → d a b ≥ 2^(n-1)

theorem maximum_good_subset_size (n : ℕ) : 
  ∃ (A : set (vector bool (2^n))), good_subset A ∧ finset.card A = 2^(n+1) :=
sorry

end maximum_good_subset_size_l233_233384


namespace equivalent_angle_l233_233731

theorem equivalent_angle (k : ℤ) :
  ∃ angle : ℝ, (angle = (k * 360 - 315) * (Float.pi / 180)) ∧
               (angle ≡ 9 * Float.pi / 4 [MOD 2 * Float.pi]) := 
sorry

end equivalent_angle_l233_233731


namespace boys_count_l233_233344

-- Define the number of girls
def girls : ℕ := 635

-- Define the number of boys as being 510 more than the number of girls
def boys : ℕ := girls + 510

-- Prove that the number of boys in the school is 1145
theorem boys_count : boys = 1145 := by
  sorry

end boys_count_l233_233344


namespace unique_function_sqrt_one_minus_x_squared_l233_233927

-- We begin by defining the conditions
variable (f : ℝ → ℝ)
variable (cont_f : ContinuousOn f (Set.Icc (-1:ℝ) 1))
variable (h1 : ∀ x ∈ Set.Icc (-1:ℝ) 1, f x = (2 - x^2) / 2 * f ((x^2) / (2 - x^2)))
variable (h2 : f 0 = 1)
variable (h3 : ∃ L : ℝ, Filter.Tendsto (λ x, f x / Real.sqrt (1 - x)) (𝓝[<] (1:ℝ)) (𝓝 L))

-- Now we formulate the problem statement to prove that f(x) = sqrt(1 - x^2) for all x ∈ [-1, 1]
theorem unique_function_sqrt_one_minus_x_squared :
  (∀ x ∈ (Set.Icc (-1:ℝ) 1), f x = Real.sqrt (1 - x^2)) :=
by sorry

end unique_function_sqrt_one_minus_x_squared_l233_233927


namespace sum_of_sequence_l233_233944

noncomputable def sequence (n : ℕ) : ℕ → ℕ
| 0       := 0
| 1       := 1
| (k + 2) := (n - 1) * sequence (k + 1) - (n - k) * sequence k / (k + 1)

theorem sum_of_sequence (n : ℕ) (h₀ : 0 < n) :
  ∑ i in Finset.range n, sequence n i = 2 ^ (n - 1) :=
sorry

end sum_of_sequence_l233_233944


namespace log_eighteen_fifteen_l233_233370

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_eighteen_fifteen (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  log_base 18 15 = (b - a + 1) / (a + 2 * b) :=
by sorry

end log_eighteen_fifteen_l233_233370


namespace ramon_twice_loui_age_in_future_l233_233348

theorem ramon_twice_loui_age_in_future : 
  ∀ (x : ℕ), 
  (∀ t : ℕ, t = 23 → 
            t * 2 = 46 → 
            ∀ r : ℕ, r = 26 → 
                      26 + x = 46 → 
                      x = 20) := 
by sorry

end ramon_twice_loui_age_in_future_l233_233348


namespace best_estimate_l233_233029

-- Define the problem conditions in Lean
variables (x : ℝ)
def is_negative : Prop := x < 0
def between_neg2_neg1 : Prop := -2 < x ∧ x < -1
def options : list ℝ := [1.3, -1.3, -2.7, 0.7, -0.7]

-- The proof statement
theorem best_estimate (h1 : is_negative x) (h2 : between_neg2_neg1 x) : x = -1.3 :=
sorry

end best_estimate_l233_233029


namespace train_speed_l233_233088

def distance := 300 -- meters
def time := 18 -- seconds

noncomputable def speed_kmh := 
  let speed_ms := distance / time -- speed in meters per second
  speed_ms * 3.6 -- convert to kilometers per hour

theorem train_speed : speed_kmh = 60 := 
  by
    -- The proof steps are omitted
    sorry

end train_speed_l233_233088


namespace miles_to_drive_l233_233920

def total_miles : ℕ := 1200
def miles_driven : ℕ := 768
def miles_remaining : ℕ := total_miles - miles_driven

theorem miles_to_drive : miles_remaining = 432 := by
  -- Proof goes here, omitted as per instructions
  sorry

end miles_to_drive_l233_233920


namespace greatest_prime_factor_of_341_l233_233608

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l233_233608


namespace nonnegative_solutions_eq_1_l233_233303

variable (x : ℝ)

theorem nonnegative_solutions_eq_1 : (x^2 = -6 * x) → (∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀) → (x ∈ {0}) :=
by
  assume (h1: x^2 = -6 * x)
  assume (h2: ∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀)
  sorry

end nonnegative_solutions_eq_1_l233_233303


namespace greatest_prime_factor_341_l233_233574

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l233_233574


namespace number_of_paths_A_to_B_l233_233053

-- Define the conditions of the lattice and paths
variables (A B : Type) -- Points A and B
  (Paths : Type) -- Type of paths between points
  (red blue green yellow : Paths → Paths → Prop) -- Types of directed arrows (edges in lattice)
  (unique_red : ∀ p, red A p → ∃! q, q = p) -- Only one way to reach each red arrow from A
  (ways_blue_from_red : ∀ r, red A r → (∃ b1 b2, blue r b1 ∧ blue r b2) ∧ (ways_to_blue : 6))
  (ways_green_from_blue : ∀ b, blue A b → (∃ g1 g2, green b g1 ∧ green b g2) ∧ (ways_to_green : 180))
  (ways_yellow_from_green : ∀ g, green A g → (∃ y1 y2, yellow g y1 ∧ yellow g y2) ∧ (ways_to_yellow : 3240))

-- Define the main theorem stating the result
theorem number_of_paths_A_to_B : ∃ num_paths, num_paths = 6480 :=
by
  have num_paths := ways_to_yellow * 2
  exact ⟨num_paths, by simp [num_paths, ways_to_yellow]⟩

sorry

end number_of_paths_A_to_B_l233_233053


namespace total_points_l233_233724

theorem total_points (darius_score marius_score matt_score total_points : ℕ) 
    (h1 : darius_score = 10) 
    (h2 : marius_score = darius_score + 3) 
    (h3 : matt_score = darius_score + 5) 
    (h4 : total_points = darius_score + marius_score + matt_score) : 
    total_points = 38 :=
by sorry

end total_points_l233_233724


namespace max_projection_area_l233_233655

theorem max_projection_area (side_length : ℝ) (angle : ℝ) (S : ℝ) :
  side_length = 1 → 
  angle = (π / 3) → 
  S = (sqrt 3 / 4) * side_length ^ 2 → 
  max_area_projection side_length angle = S :=
begin
  -- lean will automatically show true based on inputs and conclusions.
  sorry
end

end max_projection_area_l233_233655


namespace number_of_nonnegative_solutions_l233_233234

-- Define the equation x^2 = -6x
def equation (x : ℝ) : Prop := x^2 = -6 * x

-- Define the condition for a nonnegative solution
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Define the main theorem to prove the number of nonnegative solutions
theorem number_of_nonnegative_solutions : 
  {x : ℝ // equation x ∧ nonnegative x}.to_finset.card = 1 :=
sorry

end number_of_nonnegative_solutions_l233_233234


namespace arnel_number_of_boxes_l233_233701

def arnel_kept_pencils : ℕ := 10
def number_of_friends : ℕ := 5
def pencils_per_friend : ℕ := 8
def pencils_per_box : ℕ := 5

theorem arnel_number_of_boxes : ∃ (num_boxes : ℕ), 
  (number_of_friends * pencils_per_friend) + arnel_kept_pencils = num_boxes * pencils_per_box ∧ 
  num_boxes = 10 := sorry

end arnel_number_of_boxes_l233_233701


namespace percentage_divisible_by_6_l233_233633

theorem percentage_divisible_by_6 : 
  let numbers_less_than_or_equal_to_120 := (list.range 120).map (λ x, x + 1) in
  let divisible_by_6 := numbers_less_than_or_equal_to_120.filter (λ x, x % 6 = 0) in
  let percent := (divisible_by_6.length : ℚ) / 120 * 100 in
  percent = 16.67 :=
by 
  sorry

end percentage_divisible_by_6_l233_233633


namespace letters_with_dot_no_straight_line_theorem_l233_233337

variables {D S : ℕ}
variables (letters_with_dot_and_straight_line : ℕ)
variables (letters_with_straight_line_no_dot : ℕ)
variables (total_letters : ℕ)

-- Conditions from the problem
def condition1 := letters_with_dot_and_straight_line
def condition2 := letters_with_straight_line_no_dot
def condition3 := total_letters

-- Define the quantity of interest
def letters_with_dot_no_straight_line : ℕ := D - letters_with_dot_and_straight_line

-- Theorem statement that needs to be proved
theorem letters_with_dot_no_straight_line_theorem
  (h1 : letters_with_dot_and_straight_line = 16)
  (h2 : letters_with_straight_line_no_dot = 30)
  (h3 : total_letters = 50) :
  letters_with_dot_no_straight_line = 4 :=
by { sorry }

end letters_with_dot_no_straight_line_theorem_l233_233337


namespace number_of_terms_before_4_appears_l233_233316

-- Define the parameters of the arithmetic sequence
def first_term : ℤ := 100
def common_difference : ℤ := -4
def nth_term (n : ℕ) : ℤ := first_term + common_difference * (n - 1)

-- Problem: Prove that the number of terms before the number 4 appears in this sequence is 24.
theorem number_of_terms_before_4_appears :
  ∃ n : ℕ, nth_term n = 4 ∧ n - 1 = 24 := 
by
  sorry

end number_of_terms_before_4_appears_l233_233316


namespace income_left_at_end_of_year_l233_233062

variable (I : ℝ) -- Monthly income at the beginning of the year
variable (food_expense : ℝ := 0.35 * I) 
variable (education_expense : ℝ := 0.25 * I)
variable (transportation_expense : ℝ := 0.15 * I)
variable (medical_expense : ℝ := 0.10 * I)
variable (initial_expenses : ℝ := food_expense + education_expense + transportation_expense + medical_expense)
variable (remaining_income : ℝ := I - initial_expenses)
variable (house_rent : ℝ := 0.80 * remaining_income)

variable (annual_income : ℝ := 12 * I)
variable (annual_expenses : ℝ := 12 * (initial_expenses + house_rent))

variable (increased_food_expense : ℝ := food_expense * 1.05)
variable (increased_education_expense : ℝ := education_expense * 1.05)
variable (increased_transportation_expense : ℝ := transportation_expense * 1.05)
variable (increased_medical_expense : ℝ := medical_expense * 1.05)
variable (total_increased_expenses : ℝ := increased_food_expense + increased_education_expense + increased_transportation_expense + increased_medical_expense)

variable (new_income : ℝ := 1.10 * I)
variable (new_remaining_income : ℝ := new_income - total_increased_expenses)

variable (new_house_rent : ℝ := 0.80 * new_remaining_income)

variable (final_remaining_income : ℝ := new_income - (total_increased_expenses + new_house_rent))

theorem income_left_at_end_of_year : 
  final_remaining_income / new_income * 100 = 2.15 := 
  sorry

end income_left_at_end_of_year_l233_233062


namespace log_six_two_l233_233780

noncomputable def log_six (x : ℝ) : ℝ := Real.log x / Real.log 6

theorem log_six_two (a : ℝ) (h : log_six 3 = a) : log_six 2 = 1 - a :=
by
  sorry

end log_six_two_l233_233780


namespace fractional_sum_l233_233934

noncomputable def greatest_integer (t : ℝ) : ℝ := ⌊t⌋
noncomputable def fractional_part (t : ℝ) : ℝ := t - greatest_integer t

theorem fractional_sum (x : ℝ) (h : x^3 + (1/x)^3 = 18) : 
  fractional_part x + fractional_part (1/x) = 1 :=
sorry

end fractional_sum_l233_233934


namespace average_area_l233_233981

-- Define points and segments.
def A := (0, 0)
def B := (0, 1)
def C := (1, 1)
def S := set.to_finset ({(0,0), (0,1)} ∪ {(0,1), (1,1)})

-- Define areas for rotations.
noncomputable def area_swept (θ : ℝ) : ℝ :=
  let r := real.sqrt 2
  (θ / (2 * real.pi)) * (real.pi * (r ^ 2))

-- Define X1, X2 based on 45-degree (π/4 radians) rotations.
noncomputable def X1 := area_swept (real.pi / 4)
noncomputable def X2 := area_swept (real.pi / 4)

-- Define the theorem to prove the average area.
theorem average_area : (X1 + X2) / 2 = real.pi / 4 :=
  by sorry

end average_area_l233_233981


namespace total_games_l233_233095

/-- Definition of the number of games Alyssa went to this year -/
def games_this_year : Nat := 11

/-- Definition of the number of games Alyssa went to last year -/
def games_last_year : Nat := 13

/-- Definition of the number of games Alyssa plans to go to next year -/
def games_next_year : Nat := 15

/-- Statement to prove the total number of games Alyssa will go to in all -/
theorem total_games : games_this_year + games_last_year + games_next_year = 39 := by
  -- A sorry placeholder to skip the proof
  sorry

end total_games_l233_233095


namespace percentage_divisible_by_6_l233_233636

theorem percentage_divisible_by_6 : 
  let numbers_less_than_or_equal_to_120 := (list.range 120).map (λ x, x + 1) in
  let divisible_by_6 := numbers_less_than_or_equal_to_120.filter (λ x, x % 6 = 0) in
  let percent := (divisible_by_6.length : ℚ) / 120 * 100 in
  percent = 16.67 :=
by 
  sorry

end percentage_divisible_by_6_l233_233636


namespace find_k_plus_b_l233_233810

noncomputable def common_tangent_condition (k b : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), (x₁ + 1 = x₂) ∧ 
                (k = 1 / (1 + x₁)) ∧ 
                (k = 1 / x₂) ∧ 
                (kx + b = (1 / (1 + x₁)) * x - (x₁ / (1 + x₁)) + ln (1 + x₁)) ∧ 
                (kx + b = (1 / x₂) * x + ln x₂ + 1)

theorem find_k_plus_b (k b : ℝ) (h : common_tangent_condition k b) : k + b = 3 - ln 2 :=
sorry

end find_k_plus_b_l233_233810


namespace Margie_can_drive_200_miles_l233_233390

/--
  Margie's car can go 40 miles per gallon of gas, and the price of gas is $5 per gallon.
  Prove that Margie can drive 200 miles with $25 worth of gas.
-/
theorem Margie_can_drive_200_miles (miles_per_gallon price_per_gallon money_available : ℕ) 
  (h1 : miles_per_gallon = 40) (h2 : price_per_gallon = 5) (h3 : money_available = 25) : 
  (money_available / price_per_gallon) * miles_per_gallon = 200 :=
by 
  /- The proof goes here -/
  sorry

end Margie_can_drive_200_miles_l233_233390


namespace train_speed_l233_233081

theorem train_speed (length_m : ℕ) (time_s : ℕ) (length_km : ℝ) (time_hr : ℝ) 
(length_conversion : length_km = (length_m : ℝ) / 1000)
(time_conversion : time_hr = (time_s : ℝ) / 3600)
(speed : ℝ) (speed_formula : speed = length_km / time_hr) :
  length_m = 300 → time_s = 18 → speed = 60 :=
by
  intros h1 h2
  rw [h1, h2] at *
  simp [length_conversion, time_conversion, speed_formula]
  norm_num
  sorry

end train_speed_l233_233081


namespace train_speed_is_60_kmph_l233_233075

-- Define the conditions
def length_of_train : ℕ := 300 -- in meters
def time_to_cross_pole : ℕ := 18 -- in seconds

-- Define the conversions
def meters_to_kilometers (m : ℕ) : ℝ := m / 1000.0
def seconds_to_hours (s : ℕ) : ℝ := s / 3600.0

-- Define the speed calculation
def speed_km_per_hr (distance_km : ℝ) (time_hr : ℝ) : ℝ := distance_km / time_hr

-- Prove that the speed of the train is 60 km/hr
theorem train_speed_is_60_kmph :
  speed_km_per_hr (meters_to_kilometers length_of_train) (seconds_to_hours time_to_cross_pole) = 60 := 
  by
    sorry

end train_speed_is_60_kmph_l233_233075


namespace Vasilisa_transformed_carpet_l233_233907

-- Defining the conditions
def is_rectangular (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0

def original_carpet_length : ℕ := 9
def original_carpet_width : ℕ := 12
def cut_out_length : ℕ := 1
def cut_out_width : ℕ := 8
def final_square_side : ℕ := 10

-- Defining the proof statement
theorem Vasilisa_transformed_carpet :
  is_rectangular original_carpet_length original_carpet_width →
  original_carpet_length * original_carpet_width - cut_out_length * cut_out_width = final_square_side * final_square_side →
  (∃ (parts : list (ℕ × ℕ)), -- defines existence of parts of the carpet
    (∀ p ∈ parts, is_rectangular p.1 p.2) ∧ -- each part is rectangular
    (sum (λ p, p.1 * p.2) parts = final_square_side * final_square_side) ∧ -- parts sum to the final area
    (true)) -- additional properties can be added here if needed

end Vasilisa_transformed_carpet_l233_233907


namespace cars_without_paying_l233_233697

theorem cars_without_paying (total_cars : ℕ) (percent_with_tickets : ℚ) (fraction_with_passes : ℚ)
  (h1 : total_cars = 300)
  (h2 : percent_with_tickets = 0.75)
  (h3 : fraction_with_passes = 1/5) :
  let cars_with_tickets := percent_with_tickets * total_cars
  let cars_with_passes := fraction_with_passes * cars_with_tickets
  total_cars - (cars_with_tickets + cars_with_passes) = 30 :=
by
  -- Placeholder proof
  sorry

end cars_without_paying_l233_233697


namespace tangent_lines_range_l233_233832

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * Real.sqrt x

theorem tangent_lines_range (a : ℝ) :
  (∃ t1 t2 : ℝ, t1 ≠ t2 ∧ tangent_to f t1 g a ∧ tangent_to f t2 g a) ↔ 0 < a ∧ a < 2 :=
sorry

end tangent_lines_range_l233_233832


namespace find_m_for_one_solution_l233_233429

theorem find_m_for_one_solution (m : ℚ) :
  (∀ x : ℝ, 3*x^2 - 7*x + m = 0 → (∃! y : ℝ, 3*y^2 - 7*y + m = 0)) → m = 49/12 := by
  sorry

end find_m_for_one_solution_l233_233429


namespace greatest_prime_factor_341_l233_233593

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l233_233593


namespace greatest_prime_factor_341_l233_233592

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l233_233592


namespace find_angle_of_sector_l233_233436

noncomputable def radius : ℝ := 12
noncomputable def area_of_sector : ℝ := 51.54285714285714
noncomputable def pi_value : ℝ := Real.pi
noncomputable def sector_formula (θ : ℝ) (r : ℝ) : ℝ := (θ / 360) * pi_value * r ^ 2

theorem find_angle_of_sector : ∃ θ : ℝ, θ ≈ 41.01 ∧ sector_formula θ radius = area_of_sector :=
by {
  sorry
}

end find_angle_of_sector_l233_233436


namespace min_value_at_x_eq_2_l233_233831

open Real

noncomputable def f (x t : ℝ) : ℝ := x^3 - 2 * t * x^2 + t^2 * x

theorem min_value_at_x_eq_2 (t : ℝ) : (∃ x : ℝ, (f x 2) = 0) ∧ 
  (∀ x : ℝ, deriv (f x t) x = deriv (f x 2) 2) → t = 2 := by
  sorry

end min_value_at_x_eq_2_l233_233831


namespace n_power_four_plus_sixtyfour_power_n_composite_l233_233750

theorem n_power_four_plus_sixtyfour_power_n_composite (n : ℕ) : ∃ m k, m * k = n^4 + 64^n ∧ m > 1 ∧ k > 1 :=
by
  sorry

end n_power_four_plus_sixtyfour_power_n_composite_l233_233750


namespace part1_part2_l233_233838

noncomputable
def a : ℕ → ℝ
| 1       := 1/3
| (n + 1) := -2 * (a n)^2 + 2 * (a n)

def b (n : ℕ) : ℝ := 1/2 - a n

theorem part1 (n : ℕ) (h : 0 < n) : a n ∈ set.Ioo (0 : ℝ) (1/2) :=
by sorry

theorem part2 (n : ℕ) : (∑ i in finset.range n, 1 / b (i + 1)) ≥ 3^(n+1) - 3 :=
by sorry

end part1_part2_l233_233838


namespace meaningful_expression_range_l233_233879

theorem meaningful_expression_range (x : ℝ) :
  (2 - x ≥ 0) ∧ (x - 2 ≠ 0) → x < 2 :=
by
  sorry

end meaningful_expression_range_l233_233879


namespace percent_divisible_by_6_l233_233623

theorem percent_divisible_by_6 (N : ℕ) (hN : N = 120) :
  (∃ M, M = (finset.univ.filter (λ n : ℕ, n ≤ N ∧ n % 6 = 0)).card ∧ M * 6 = N) →
  (M.to_real / N.to_real) * 100 = 16.66666667 :=
by
  intros h
  sorry

end percent_divisible_by_6_l233_233623


namespace intersection_A_complement_B_l233_233964

-- Definition of the universal set U
def U : Set ℝ := Set.univ

-- Definition of the set A
def A : Set ℝ := {x | x^2 - 2 * x < 0}

-- Definition of the set B
def B : Set ℝ := {x | x > 1}

-- Definition of the complement of B in U
def complement_B : Set ℝ := {x | x ≤ 1}

-- The intersection A ∩ complement_B
def intersection : Set ℝ := {x | 0 < x ∧ x ≤ 1}

-- The theorem to prove
theorem intersection_A_complement_B : A ∩ complement_B = intersection :=
by
  -- Proof goes here
  sorry

end intersection_A_complement_B_l233_233964


namespace small_rectangle_area_l233_233406

theorem small_rectangle_area
  (length_large : ℝ) (width_large : ℝ)
  (h1 : length_large = 10) (h2 : width_large = 8) :
  let length_small := length_large / 2
      width_small := width_large / 2
      area_small := length_small * width_small
  in area_small = 20 :=
by
  rw [h1, h2]
  let length_small := 10 / 2
  let width_small := 8 / 2
  let area_small := length_small * width_small
  show area_small = 20
  sorry

end small_rectangle_area_l233_233406


namespace centroid_triangle_property_l233_233008

theorem centroid_triangle_property 
  (A B C D D' E E' F F' O : Type)
  [is_centroid O A B C]
  [lines_through O A B D]
  [lines_through O A C D']
  [lines_through O C A E]
  [lines_through O C B E']
  [lines_through O B C F]
  [lines_through O B A F'] :
  AB * (1 / AD + 1 / BF') + AC * (1 / AD' + 1 / CE) + BC * (1 / BF + 1 / CE') = 9 := 
sorry

end centroid_triangle_property_l233_233008


namespace total_equipment_cost_l233_233466

-- Define the cost of each piece of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8

-- Define the number of players
def players : ℕ := 16

-- Define the total cost of equipment for one player
def equipment_cost_per_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Define the total cost for all players
def total_cost : ℝ := players * equipment_cost_per_player

-- The proof problem to be stated:
theorem total_equipment_cost (jc sc k p : ℝ) (n : ℕ) :
  jc = 25 ∧ sc = 15.2 ∧ k = 6.8 ∧ p = 16 →
  total_cost = 752 :=
by
  intro h
  rcases h with ⟨hc1, hc2, hc3, hc4⟩
  simp [total_cost, equipment_cost_per_player, hc1, hc2, hc3, hc4]
  exact sorry

end total_equipment_cost_l233_233466


namespace fraction_of_girls_is_one_third_l233_233478

-- Define the number of children and number of boys
def total_children : Nat := 45
def boys : Nat := 30

-- Calculate the number of girls
def girls : Nat := total_children - boys

-- Calculate the fraction of girls
def fraction_of_girls : Rat := (girls : Rat) / (total_children : Rat)

theorem fraction_of_girls_is_one_third : fraction_of_girls = 1 / 3 :=
by
  sorry -- Proof is not required

end fraction_of_girls_is_one_third_l233_233478


namespace entree_cost_difference_l233_233230

theorem entree_cost_difference 
  (total_cost : ℕ)
  (entree_cost : ℕ)
  (dessert_cost : ℕ)
  (h1 : total_cost = 23)
  (h2 : entree_cost = 14)
  (h3 : total_cost = entree_cost + dessert_cost) :
  entree_cost - dessert_cost = 5 :=
by
  sorry

end entree_cost_difference_l233_233230


namespace time_to_cross_tree_eq_120_l233_233047

/-
  Problem statement:
  Given that a train is 1200 meters long and it takes 160 seconds to pass a 400-meter platform,
  prove that the time it takes for the train to cross a tree is 120 seconds.
-/

-- Definition of the conditions
def l_train : ℝ := 1200
def l_platform : ℝ := 400
def t_pass_platform : ℝ := 160

-- Calculation of the speed of the train
def speed_train : ℝ := (l_train + l_platform) / t_pass_platform

-- Time it takes to cross the tree
def time_to_cross_tree := l_train / speed_train

-- The theorem stating the time to cross a tree
theorem time_to_cross_tree_eq_120 : time_to_cross_tree = 120 := by
  sorry

end time_to_cross_tree_eq_120_l233_233047


namespace calculate_expression_l233_233709

theorem calculate_expression : (3.14 - Real.pi)^0 + |Real.sqrt 2 - 1| + (1/2 : ℝ)^(-1) - Real.sqrt 8 = 2 - Real.sqrt 2 :=
by
  sorry

end calculate_expression_l233_233709


namespace cone_properties_l233_233139

-- Defining parameters for the cone
def diameter : ℝ := 12
def height : ℝ := 9
def radius : ℝ := diameter / 2
def slant_height : ℝ := Real.sqrt (radius ^ 2 + height ^ 2)

-- Volume of the cone
def cone_volume : ℝ := (1 / 3) * Real.pi * radius ^ 2 * height

-- Surface area of the cone
def cone_surface_area : ℝ := (Real.pi * radius ^ 2) + (Real.pi * radius * slant_height)

-- Theorem statement
theorem cone_properties 
    (diameter_eq : diameter = 12)
    (height_eq : height = 9)
    (radius_eq : radius = diameter / 2)
    (slant_height_eq : slant_height = Real.sqrt (radius ^ 2 + height ^ 2))
    : cone_volume = 108 * Real.pi ∧ cone_surface_area = Real.pi * (36 + 6 * Real.sqrt 117) :=
by
  have radius_calc : radius = 6 := by rw [radius_eq, diameter_eq]; norm_num,
  have slant_height_calc : slant_height = Real.sqrt 117 := by rw [slant_height_eq, radius_calc, height_eq]; norm_num; congr,
  have volume_calc : cone_volume = 108 * Real.pi := by rw [cone_volume, radius_calc, height_eq]; norm_num,
  have surface_area_calc : cone_surface_area = Real.pi * (36 + 6 * Real.sqrt 117) := 
    by rw [cone_surface_area, radius_calc, slant_height_calc]; ring,
  exact ⟨volume_calc, surface_area_calc⟩

end cone_properties_l233_233139


namespace b_amount_after_investing_l233_233031

-- Define the conditions
variables (x y : ℝ)
def condition1 := x + y = 1210
def condition2 := (4 / 15) * x
def condition3 := (2 / 5) * y
def condition4 := (3 / 5) * (x - (4 / 15) * x) = (2 / 3) * (y - (2 / 5) * y)
def condition5 := (1 / 3) * x

-- Define the theorem stating the problem
theorem b_amount_after_investing (h1: condition1) (h2: condition2) (h3: condition3) (h4: condition4) (h5: condition5) :
  (3 / 5) * y = 380.29 :=
sorry

end b_amount_after_investing_l233_233031


namespace calculate_log_difference_l233_233114

def log := real.log10
axiom log_diff (a b : ℝ) : log a - log b = log (a / b)
axiom log_base10 : log 10 = 1
axiom log_power (x : ℝ) : log (10^x) = x * log 10

theorem calculate_log_difference : log 4 - log 4000 = -3 := by
  have h1 : log 4 - log 4000 = log (4 / 4000),
  from log_diff 4 4000,
  have h2 : log (4 / 4000) = log (1 / 1000),
  from congr_arg log (by norm_num),
  have h3 : log (1 / 1000) = log (10 ^ (-3)),
  from congr_arg log (by norm_num),
  have h4 : log (10 ^ (-3)) = (-3) * log 10,
  from log_power (-3),
  have h5 : (-3) * log 10 = -3,
  from congr_arg (has_mul.mul (-3)) log_base10,
  rw [h1, h2, h3, h4, h5]

end calculate_log_difference_l233_233114


namespace min_value_of_y_l233_233758

noncomputable def y (x : ℝ) : ℝ :=
  2 * Real.sin (Real.pi / 3 - x) - Real.cos (Real.pi / 6 + x)

theorem min_value_of_y : ∃ x : ℝ, y x = -1 := by
  sorry

end min_value_of_y_l233_233758


namespace remainder_250_div_k_l233_233167

theorem remainder_250_div_k {k : ℕ} (h1 : 0 < k) (h2 : 180 % (k * k) = 12) : 250 % k = 10 := by
  sorry

end remainder_250_div_k_l233_233167


namespace min_lambda_l233_233815

noncomputable def a_n (n : ℕ) : ℝ :=
  2 * n - 1

noncomputable def b_n (n : ℕ) : ℝ :=
  (2 : ℝ)^(n-1) / (2 * n - 1)

noncomputable def sum_b_inv_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1), 1 / b_n i

theorem min_lambda (n : ℕ) (h_pos : ∀ i ∈ Finset.range (n + 1), a_n i > 0) :
  (∃ λ, λ > sum_b_inv_n n) ↔ λ ≥ 6 :=
  sorry

end min_lambda_l233_233815


namespace imaginary_part_of_z_l233_233818

theorem imaginary_part_of_z (z : ℂ) (h : (1 + complex.i) * z = 2) : z.im = -1 :=
sorry

end imaginary_part_of_z_l233_233818


namespace total_faces_is_198_l233_233012

-- Definitions for the number of dice and geometrical shapes brought by each person:
def TomDice : ℕ := 4
def TimDice : ℕ := 5
def TaraDice : ℕ := 3
def TinaDice : ℕ := 2
def TonyCubes : ℕ := 1
def TonyTetrahedrons : ℕ := 3
def TonyIcosahedrons : ℕ := 2

-- Definitions for the number of faces for each type of dice or shape:
def SixSidedFaces : ℕ := 6
def EightSidedFaces : ℕ := 8
def TwelveSidedFaces : ℕ := 12
def TwentySidedFaces : ℕ := 20
def CubeFaces : ℕ := 6
def TetrahedronFaces : ℕ := 4
def IcosahedronFaces : ℕ := 20

-- We want to prove that the total number of faces is 198:
theorem total_faces_is_198 : 
  (TomDice * SixSidedFaces) + 
  (TimDice * EightSidedFaces) + 
  (TaraDice * TwelveSidedFaces) + 
  (TinaDice * TwentySidedFaces) + 
  (TonyCubes * CubeFaces) + 
  (TonyTetrahedrons * TetrahedronFaces) + 
  (TonyIcosahedrons * IcosahedronFaces) 
  = 198 := 
by {
  sorry
}

end total_faces_is_198_l233_233012


namespace range_of_a_l233_233800

variable (a : ℝ) (x : ℝ)

def p := (x^2 - 5*x - 6 ≤ 0)

def q := (x^2 - 2*x + 1 - 4*a^2 ≤ 0)

theorem range_of_a (hp : ∀ x, ¬p x → (x < -1 ∨ x > 6))
  (hq : ∀ x, ¬q x → (x < 1 - 2*a ∨ x > 1 + 2*a))
  (h : ∀ x, (¬p x → ¬q x) ∧ ¬(¬p x ↔ ¬q x)) :
  a ≥ 5 / 2 :=
by
  sorry

end range_of_a_l233_233800


namespace eva_total_marks_l233_233738

theorem eva_total_marks
    (math_score_s2 : ℕ) (arts_score_s2 : ℕ) (science_score_s2 : ℕ)
    (math_diff : ℕ) (arts_diff : ℕ) (science_frac_diff : ℚ)
    (math_score_s2_eq : math_score_s2 = 80)
    (arts_score_s2_eq : arts_score_s2 = 90)
    (science_score_s2_eq : science_score_s2 = 90)
    (math_diff_eq : math_diff = 10)
    (arts_diff_eq : arts_diff = 15)
    (science_frac_diff_eq : science_frac_diff = 1/3) : 
  (math_score_s2 + 10 + (math_score_s2 + math_diff) + 
   (arts_score_s2 + 90 - 15) + (arts_score_s2 + arts_diff) + 
   (science_score_s2 + 90 - (1/3) * 90) + (science_score_s2 + science_score_s2 * 1/3)) = 485 := 
by
  sorry

end eva_total_marks_l233_233738


namespace nonnegative_solutions_eq_1_l233_233299

variable (x : ℝ)

theorem nonnegative_solutions_eq_1 : (x^2 = -6 * x) → (∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀) → (x ∈ {0}) :=
by
  assume (h1: x^2 = -6 * x)
  assume (h2: ∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀)
  sorry

end nonnegative_solutions_eq_1_l233_233299


namespace solution_of_inequality_system_l233_233460

theorem solution_of_inequality_system (x : ℝ) : 
  (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x) ∧ (x < 1) := 
by sorry

end solution_of_inequality_system_l233_233460


namespace count_congruent_to_3_mod_7_lt_500_l233_233858

theorem count_congruent_to_3_mod_7_lt_500 : 
  ∃ n, n = 71 ∧ ∀ x, 0 < x ∧ x < 500 ∧ x % 7 = 3 ↔ ∃ k, 0 ≤ k ∧ k ≤ 70 ∧ x = 3 + 7 * k :=
sorry

end count_congruent_to_3_mod_7_lt_500_l233_233858


namespace greatest_prime_factor_341_l233_233519

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l233_233519


namespace calculate_f_20_l233_233385

-- Define the piecewise function
def f (x : ℕ) : ℕ :=
  if x ≤ 8 then x^2 + 1 else x - 15

-- The proof problem
theorem calculate_f_20 : f (f (f 20)) = -4 := 
sorry

end calculate_f_20_l233_233385


namespace tan_pi_minus_alpha_l233_233197

noncomputable def angle_in_second_quadrant (α : ℝ) : Prop := α ∈ Set.Ioo (π / 2) π
def sin_of_alpha (α : ℝ) := 1 / 3

theorem tan_pi_minus_alpha (α : ℝ) (hα : angle_in_second_quadrant α) (h_sin : sin α = 1 / 3) :
  tan (π - α) = √2 / 4 :=
  sorry

end tan_pi_minus_alpha_l233_233197


namespace find_lambda_l233_233906

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Defining points and vectors in lean
variables {A B C D E F : V}
variable (λ : ℝ)

-- Parallelogram definition
def is_parallelogram (A B C D : V) : Prop :=
  ∃ X Y Z W : V, Z - X = D - A ∧ W - Y = C - B ∧ X = A ∧ Y = B ∧ Z = D ∧ W = C

-- Conditions definitions
def condition_1 (A E D : V) : Prop := E - A = 2 • (D - E)
def condition_2 (B F C : V) : Prop := B - F = F - C
def condition_3 (A C E F : V) (λ : ℝ) : Prop := C - A = λ • (E - A) + (F - A)

-- Theorem statement
theorem find_lambda (h₁ : is_parallelogram A B C D)
                    (h₂ : condition_1 A E D)
                    (h₃ : condition_2 B F C)
                    (h₄ : condition_3 A C E F λ) :
  λ = 3 / 4 :=
sorry

end find_lambda_l233_233906


namespace greatest_prime_factor_341_l233_233530

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l233_233530


namespace greatest_prime_factor_341_l233_233590

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l233_233590


namespace nonnegative_solutions_eq1_l233_233245

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l233_233245


namespace sqrt_xy_gt_cbrt_xyz_l233_233788

theorem sqrt_xy_gt_cbrt_xyz (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  real.sqrt (x^2 + y^2) > real.cbrt (x^3 + y^3) :=
sorry

end sqrt_xy_gt_cbrt_xyz_l233_233788


namespace rationalize_result_l233_233412

noncomputable def rationalize_denominator (x y : ℚ) (sqrt_c : ℚ) : ℚ :=
  let numerator := x + sqrt_c
  let denominator := y - sqrt_c
  (numerator * (y + sqrt_c)) / (denominator * (y + sqrt_c))

theorem rationalize_result :
  let sqrt_5 := Real.sqrt 5
  let expr := rationalize_denominator 2 3 sqrt_5
  let A := 11 / 4
  let B := 5 / 4
  let C := 5
  expr = A + B * sqrt_5 ∧ A * B * C = 275 / 16 := 
sorry

end rationalize_result_l233_233412


namespace percent_divisible_by_six_up_to_120_l233_233632

theorem percent_divisible_by_six_up_to_120 : 
  let total_numbers := 120
  let divisible_by_six := total_numbers / 6
  let percentage := (divisible_by_six * 100) / total_numbers
  percentage = 50 / 3 := sorry

end percent_divisible_by_six_up_to_120_l233_233632


namespace count_random_events_l233_233096

-- Definitions based on conditions in the problem
def total_products : ℕ := 100
def genuine_products : ℕ := 95
def defective_products : ℕ := 5
def drawn_products : ℕ := 6

-- Events definitions
def event_1 := drawn_products > defective_products  -- at least 1 genuine product
def event_2 := drawn_products ≥ 3  -- at least 3 defective products
def event_3 := drawn_products = defective_products  -- all 6 are defective
def event_4 := drawn_products - 2 = 4  -- 2 defective and 4 genuine products

-- Dummy definition for random event counter state in the problem context
def random_events : ℕ := 2

-- Main theorem statement
theorem count_random_events :
  (event_1 → true) ∧ 
  (event_2 ∧ ¬ event_3 ∧ event_4) →
  random_events = 2 :=
by
  sorry

end count_random_events_l233_233096


namespace bulk_discount_percentage_l233_233922

/-- 
John has to buy 300 ball bearings. Normally each ball bearing costs $1, but there is a sale at a price 
of $0.75 per ball bearing. He saved $120 by buying all ball bearings during the sale rather than 
one at a time. What is the percentage of the bulk discount he gets?
-/
theorem bulk_discount_percentage 
  (n_machines : ℕ := 10) 
  (n_bearings_per_machine : ℕ := 30) 
  (normal_cost_per_bearing : ℝ := 1) 
  (sale_cost_per_bearing : ℝ := 0.75) 
  (total_savings : ℝ := 120) 
  (total_bearings : ℕ) 
  (total_cost_normal : ℝ) 
  (total_cost_sale : ℝ) 
  (cost_with_bulk_discount : ℝ) 
  (bulk_discount : ℝ) 
  (bulk_discount_percentage : ℝ) :
  total_bearings = n_machines * n_bearings_per_machine ∧
  total_cost_normal = total_bearings * normal_cost_per_bearing ∧
  total_cost_sale = total_bearings * sale_cost_per_bearing ∧
  cost_with_bulk_discount = total_cost_normal - total_savings ∧
  bulk_discount = total_cost_sale - cost_with_bulk_discount ∧
  bulk_discount_percentage = (bulk_discount / total_cost_sale) * 100 :=
begin
  sorry
end

end bulk_discount_percentage_l233_233922


namespace maximum_good_subset_size_l233_233383

noncomputable theory

def Sn (n : ℕ) : set (vector bool (2^n)) := 
  {a | ∀ i, 1 ≤ i ∧ i ≤ 2^n → a.nth (i - 1) = tt ∨ a.nth (i - 1) = ff}

def d {n : ℕ} (a b : vector bool (2^n)) : ℕ :=
  finset.card (finset.filter (λ i, a.nth i ≠ b.nth i) (finset.range (2^n)))

def good_subset {n : ℕ} (A : set (vector bool (2^n))) : Prop :=
  ∀ ⦃a b⦄, a ∈ A → b ∈ A → a ≠ b → d a b ≥ 2^(n-1)

theorem maximum_good_subset_size (n : ℕ) : 
  ∃ (A : set (vector bool (2^n))), good_subset A ∧ finset.card A = 2^(n+1) :=
sorry

end maximum_good_subset_size_l233_233383


namespace greatest_prime_factor_of_341_l233_233544

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l233_233544


namespace largest_of_five_consecutive_integers_l233_233766

   theorem largest_of_five_consecutive_integers (n1 n2 n3 n4 n5 : ℕ) 
     (h1: 0 < n1) (h2: n1 + 1 = n2) (h3: n2 + 1 = n3) (h4: n3 + 1 = n4)
     (h5: n4 + 1 = n5) (h6: n1 * n2 * n3 * n4 * n5 = 15120) : n5 = 10 :=
   sorry
   
end largest_of_five_consecutive_integers_l233_233766


namespace sine_of_smallest_angle_triangle_l233_233188

theorem sine_of_smallest_angle_triangle (A B C : Type) [Triangle A B C]
  (hC : angle A B C = 90)
  (hNotIso : ¬(isosceles_right_trianlge A B C))
  (hSinB : sin (angle B) = n)
  (hSmallest : angle B < angle A) : 
  0 < n ∧ n < sqrt (1/2) :=
sorry

end sine_of_smallest_angle_triangle_l233_233188


namespace greatest_prime_factor_341_l233_233518

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l233_233518


namespace greatest_prime_factor_of_341_l233_233606

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l233_233606


namespace find_k_from_line_and_circle_intersection_l233_233835

theorem find_k_from_line_and_circle_intersection (k : ℝ) :
  (∃ k : ℝ, ∀ α : ℝ, (let x := 2 * Real.cos α, y := 3 + 2 * Real.sin α in
  (x, y) ∈ {p : ℝ × ℝ | ∃ x y, y = k * x + 1 ∧ x^2 + (y - 3)^2 = 4}) ∧ dist (2 * Real.cos α, 3 + 2 * Real.sin α) (2 * Real.cos (α + π), 3 + 2 * Real.sin (α + π)) = 2 * Real.sqrt 3) ↔ k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
sorry

end find_k_from_line_and_circle_intersection_l233_233835


namespace polynomial_with_given_root_and_x_coeff_l233_233159

noncomputable def quadratic_polynomial : ℚ[X] :=
-5 * X^2 + 10 * X - 25

theorem polynomial_with_given_root_and_x_coeff
  (p : ℚ[X])
  (h_root : p.eval (1 + 2*I) = 0)
  (h_real : ∀ x : ℚ, p.eval x ∈ ℚ)
  (h_x_coeff : p.coeff 1 = 10) :
  p = quadratic_polynomial := 
sorry

end polynomial_with_given_root_and_x_coeff_l233_233159


namespace value_of_a_l233_233915

theorem value_of_a (m n a : ℚ) 
  (h₁ : m = 5 * n + 5) 
  (h₂ : m + 2 = 5 * (n + a) + 5) : 
  a = 2 / 5 :=
by
  sorry

end value_of_a_l233_233915


namespace nonnegative_solutions_eq1_l233_233242

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l233_233242


namespace rationalize_denominator_l233_233415

theorem rationalize_denominator :
  let expr := (2 + Real.sqrt 5) / (3 - Real.sqrt 5),
      A := 11 / 4,
      B := 5 / 4,
      C := 5
  in expr = (A + B * Real.sqrt C) → (A * B * C) = 275 / 16 :=
by
  intros
  sorry

end rationalize_denominator_l233_233415


namespace charrua_number_exists_l233_233502

def is_charrua (n : ℕ) : Prop :=
  (∀ d ∈ digits n, d > 1) ∧ (∀ d_list, d_list.length = 4 → (list.prod d_list) ∣ n)

theorem charrua_number_exists (k : ℕ) : ∃ n : ℕ, is_charrua n ∧ (nat.log10 n).to_nat > k :=
by
  sorry

end charrua_number_exists_l233_233502


namespace sin_cos_identity_l233_233212

variable (θ φ : ℝ)

def angle_measurements (θ φ : ℝ) : Prop :=
θ = 45 ∧ φ = 135

theorem sin_cos_identity (h : angle_measurements θ φ) :
  sin (θ * (π / 180)) = -(cos (φ * (π / 180))) :=
by
  rcases h with ⟨h1, h2⟩
  sorry

end sin_cos_identity_l233_233212


namespace eva_total_marks_l233_233740

theorem eva_total_marks
    (math_score_s2 : ℕ) (arts_score_s2 : ℕ) (science_score_s2 : ℕ)
    (math_diff : ℕ) (arts_diff : ℕ) (science_frac_diff : ℚ)
    (math_score_s2_eq : math_score_s2 = 80)
    (arts_score_s2_eq : arts_score_s2 = 90)
    (science_score_s2_eq : science_score_s2 = 90)
    (math_diff_eq : math_diff = 10)
    (arts_diff_eq : arts_diff = 15)
    (science_frac_diff_eq : science_frac_diff = 1/3) : 
  (math_score_s2 + 10 + (math_score_s2 + math_diff) + 
   (arts_score_s2 + 90 - 15) + (arts_score_s2 + arts_diff) + 
   (science_score_s2 + 90 - (1/3) * 90) + (science_score_s2 + science_score_s2 * 1/3)) = 485 := 
by
  sorry

end eva_total_marks_l233_233740


namespace puppies_per_cage_l233_233066

theorem puppies_per_cage (initial_puppies sold_puppies cages remaining_puppies puppies_per_cage : ℕ)
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : cages = 3)
  (h4 : remaining_puppies = initial_puppies - sold_puppies)
  (h5 : puppies_per_cage = remaining_puppies / cages) :
  puppies_per_cage = 5 := by
  sorry

end puppies_per_cage_l233_233066


namespace mindy_total_l233_233392

def roundToNearestDollar (x : Float) : Int := 
  Int.floor (x + 0.5)

def totalRoundedAmount (amounts : List Float) : Int :=
  (amounts.map roundToNearestDollar).sum

theorem mindy_total (a1 a2 a3 a4 : Float) (h1 : a1 = 2.47) (h2 : a2 = 7.51) (h3 : a3 = 11.56) (h4 : a4 = 4.98) :
  totalRoundedAmount [a1, a2, a3, a4] = 27 :=
by
  sorry

end mindy_total_l233_233392


namespace focus_coordinates_eq_l233_233756

noncomputable def focus_of_parabola (p: ℝ) (x: ℝ) : ℝ × ℝ :=
  ((p / 2), 0)

theorem focus_coordinates_eq {x: ℝ} (h: y^2 = (1/2)*x) : 
  (focus_of_parabola (1/4)) = (1/8, 0) :=
sorry

end focus_coordinates_eq_l233_233756


namespace comparison_abc_l233_233938

noncomputable def a : ℝ := (Real.exp 1 + 2) / Real.log (Real.exp 1 + 2)
noncomputable def b : ℝ := 2 / Real.log 2
noncomputable def c : ℝ := (Real.exp 1)^2 / (4 - Real.log 4)

theorem comparison_abc : c < b ∧ b < a :=
by {
  sorry
}

end comparison_abc_l233_233938


namespace tom_sleep_hours_l233_233483

-- Define initial sleep hours and increase fraction
def initial_sleep_hours : ℕ := 6
def increase_fraction : ℚ := 1 / 3

-- Define the function to calculate increased sleep
def increased_sleep_hours (initial : ℕ) (fraction : ℚ) : ℚ :=
  initial * fraction

-- Define the function to calculate total sleep hours
def total_sleep_hours (initial : ℕ) (increased : ℚ) : ℚ :=
  initial + increased

-- Theorem stating Tom's total sleep hours per night after the increase
theorem tom_sleep_hours (initial : ℕ) (fraction : ℚ) (increased : ℚ) (total : ℚ) :
  initial = initial_sleep_hours →
  fraction = increase_fraction →
  increased = increased_sleep_hours initial fraction →
  total = total_sleep_hours initial increased →
  total = 8 :=
by
  intros h_init h_frac h_incr h_total
  rw [h_init, h_frac] at h_incr
  rw [h_init, h_incr] at h_total
  sorry

end tom_sleep_hours_l233_233483


namespace greatest_prime_factor_341_l233_233514

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l233_233514


namespace AB_adjacent_permutations_l233_233474

theorem AB_adjacent_permutations : 
  let n := 4 in
  let people := ["A", "B", "C", "D"] in
  let valid_permutations := { perm | 
    perm ∈ permutations people ∧ 
    (let idxA := perm.indexOf "A" in 
     let idxB := perm.indexOf "B" in 
     abs (idxA - idxB) = 1) 
  } in
  card valid_permutations = 12 :=
begin
  sorry
end

end AB_adjacent_permutations_l233_233474


namespace sum_difference_l233_233987

def arithmetic_series_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def set_A_sum : ℕ :=
  arithmetic_series_sum 42 2 25

def set_B_sum : ℕ :=
  arithmetic_series_sum 62 2 25

theorem sum_difference :
  set_B_sum - set_A_sum = 500 :=
by
  sorry

end sum_difference_l233_233987


namespace greatest_prime_factor_of_341_l233_233543

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l233_233543


namespace games_within_division_l233_233666

theorem games_within_division (N M : ℕ) (h1 : N > 3 * M) (h2 : M > 5) (h3 : 2 * N + 6 * M = 76) : 2 * N = 40 :=
by {
  sorry
}

end games_within_division_l233_233666


namespace paint_needed_for_small_spheres_l233_233980

-- Given constants
constant R : ℝ -- Radius of the large sphere
constant r : ℝ -- Radius of each small sphere
constant paint_large : ℝ := 2.4 -- Paint required for the large sphere in kg

-- Volume equality between large sphere and 64 small spheres
axiom volume_relation : (4 / 3) * Real.pi * R^3 = 64 * (4 / 3) * Real.pi * r^3

-- Surface area relation and amount of paint needed
theorem paint_needed_for_small_spheres : 
(64 * 4 * Real.pi * r^2 = 4 * (4 * Real.pi * R^2)) → 
4 * paint_large = 9.6 := by
  intros h1
  sorry

end paint_needed_for_small_spheres_l233_233980


namespace element_with_48_percent_in_CaCO3_l233_233336

structure Element :=
  (name : String)
  (molar_mass : Float)

def CaCO3 := [Element.mk "Ca" 40.08,
              Element.mk "C" 12.01,
              Element.mk "O" 16.00]

def molar_mass_total (compound : List Element) (ratios : List Float) : Float :=
  List.foldl (λ acc (p : Element × Float) => acc + p.1.molar_mass * p.2) 0 
    (List.zip compound ratios)

def mass_percentage (element : Element) (compound : List Element) (ratios : List Float) : Float :=
  (element.molar_mass * (List.get (List.index_of element compound) ratios)) / 
  molar_mass_total compound ratios * 100

noncomputable def element_with_mass_percentage_approx (compound : List Element) (ratios : List Float) (percent : Float) : Element :=
  List.foldl (λ acc elem => 
    if Float.abs (mass_percentage elem compound ratios - percent) < Float.abs (mass_percentage acc compound ratios - percent) 
    then elem
    else acc) (List.head! compound) compound

theorem element_with_48_percent_in_CaCO3 : element_with_mass_percentage_approx CaCO3 [1, 1, 3] 48 = Element.mk "O" 16.00 :=
  by sorry

end element_with_48_percent_in_CaCO3_l233_233336


namespace left_side_eq_one_plus_a_l233_233914

variable (a : ℝ) (n : ℕ) -- Define variables a and n, a in the reals and n in the natural numbers.

theorem left_side_eq_one_plus_a (h₁ : a ≠ 1) (h₂ : n = 1) : (1 + a + a^2 + ... + a^n) = 1 + a := by
  sorry -- Proof omitted

end left_side_eq_one_plus_a_l233_233914


namespace rational_sum_l233_233877

theorem rational_sum (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : x + y = 7 ∨ x + y = 3 := 
sorry

end rational_sum_l233_233877


namespace number_of_ones_and_twos_l233_233060

theorem number_of_ones_and_twos (N : ℕ) (hN : Nat.digits 10 N = 100) 
  (h1 : ∀ d ∈ Nat.digits 10 N, d = 1 ∨ d = 2) 
  (h2 : Nat.digits 10 N % 3 = 0)
  (h3 : ∀ idx1 idx2, idx1 < idx2 → Nat.digits 10 N idx1 = 2 → Nat.digits 10 N idx2 = 2 → 
          ∃ k, idx2 = idx1 + 2 * k) :
  (∃ a b, a + b = 100 ∧ a + 2 * b = 100 ∧ b = 2) :=
by
  sorry -- Proof skipped

end number_of_ones_and_twos_l233_233060


namespace Mille_suckers_l233_233971

theorem Mille_suckers:
  let pretzels := 64
  let goldfish := 4 * pretzels
  let baggies := 16
  let items_per_baggie := 22
  let total_items_needed := baggies * items_per_baggie
  let total_pretzels_and_goldfish := pretzels + goldfish
  let suckers := total_items_needed - total_pretzels_and_goldfish
  suckers = 32 := 
by sorry

end Mille_suckers_l233_233971


namespace lucas_mod_8_100th_term_l233_233434

def lucas_sequence : ℕ → ℤ
| 1 := 1
| 2 := 3
| n := lucas_sequence (n - 1) + lucas_sequence (n - 2)

theorem lucas_mod_8_100th_term : (lucas_sequence 100) % 8 = 7 := 
by sorry

end lucas_mod_8_100th_term_l233_233434


namespace smallest_circle_area_of_triangle_l233_233360

theorem smallest_circle_area_of_triangle
  (a b c : ℝ)
  (ha : a = 4)
  (hb : b = 2)
  (hc : c = 2 * Real.sqrt 3)
  (hSc : abs (Real.sin C - Real.sqrt 3 * Real.sin B) = 0) :
  ∃ R, 2 * R = (c / (abs (Real.sin C))) ∧
        (π * R^2) = 4 * π := 
by
  let radius := 2
  use radius
  split
  . --> Proof that 2 * radius = c / sin C (abbreviated proof)
        sorry
  . --> Proof that π * radius^2 = 4π (abbreviated proof)
        sorry

end smallest_circle_area_of_triangle_l233_233360


namespace general_term_sum_abs_l233_233814

def seq_sum (n : ℕ) : ℤ := 2 * n^2 - 3 * n - 10

def a_n (n : ℕ) : ℤ :=
  if n = 1 then -11
  else 4 * n - 5

def T_n (n : ℕ) : ℤ :=
  if n = 1 then 11
  else 2 * n^2 - 3 * n + 12

theorem general_term (n : ℕ) :
  a_n n = (seq_sum n - seq_sum (n - 1)) :=
begin
  sorry -- proof to be filled in
end

theorem sum_abs (n : ℕ) :
  T_n n = ∑ i in (finset.range n).image (+1), |a_n i| :=
begin
  sorry -- proof to be filled in
end

end general_term_sum_abs_l233_233814


namespace degenerate_ellipse_single_point_c_l233_233431

theorem degenerate_ellipse_single_point_c (c : ℝ) :
  (∀ x y : ℝ, 2 * x^2 + y^2 + 8 * x - 10 * y + c = 0 → x = -2 ∧ y = 5) →
  c = 33 :=
by
  intros h
  sorry

end degenerate_ellipse_single_point_c_l233_233431


namespace equivalent_proof_problem_l233_233226

variable {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Define given conditions
def condition1 : Prop := ‖a‖ = 2
def condition2 : Prop := ‖b‖ = 1
def condition3 : Prop := ‖a - b‖ = 2

-- Define and prove the first value
noncomputable def part1 : Prop :=
  (2 • a + b) ⬝ (a - 2 • b) = 9 / 2

-- Define and prove the second value
noncomputable def part2 : Prop :=
  ‖a + b‖ = real.sqrt 6

-- Proof problem combining all conditions
theorem equivalent_proof_problem :
  condition1 → condition2 → condition3 → part1 ∧ part2 :=
by
  sorry

end equivalent_proof_problem_l233_233226


namespace number_of_elements_cong_set_l233_233854

/-- Define a set of integers less than 500 and congruent to 3 modulo 7 -/
def cong_set : Set ℕ := {n | n < 500 ∧ n % 7 = 3}

/-- The theorem stating the number of elements in cong_set is 72 -/
theorem number_of_elements_cong_set : Set.card cong_set = 72 :=
sorry

end number_of_elements_cong_set_l233_233854


namespace min_value_is_10_over_3_l233_233958

noncomputable def min_value (x y z : ℝ) : ℝ :=
  if h : (x + y + z = 3 ∧ y = 2 * x ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  then (1/x + 1/y + 1/z) 
  else 0

theorem min_value_is_10_over_3 : ∀ (x y z : ℝ), 
  (x + y + z = 3) ∧ (y = 2 * x) ∧ (x > 0) ∧ (y > 0) ∧ (z > 0) -> 
  min_value x y z = 10 / 3 :=
by
  intros x y z h
  rw min_value
  by_cases h : (x + y + z = 3 ∧ y = 2 * x ∧ x > 0 ∧ y > 0 ∧ z > 0)
  . rw dif_pos h
    sorry
  . contradiction

end min_value_is_10_over_3_l233_233958


namespace nonnegative_solutions_count_l233_233312

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.card = 1 := 
sorry

end nonnegative_solutions_count_l233_233312


namespace ordered_triple_unique_solution_l233_233472

theorem ordered_triple_unique_solution :
  let a : ℝ^3 := ![\[2, 2, 2\]]
  let b : ℝ^3 := ![\[3, -2, 1\]]
  let c : ℝ^3 := ![\[3, 3, -4\]]
  let v : ℝ^3 := ![\[3, -1, 8\]]
  let p := 5 / 3
  let q := 0
  let r := -10 / 17
  v = p • a + q • b + r • c := by
  simp [a, b, c, v, p, q, r]
  sorry

end ordered_triple_unique_solution_l233_233472


namespace quadratic_solution_set_l233_233333

theorem quadratic_solution_set (a b : ℝ)
  (h1 : ∀ x : ℝ, ax + b > 0 ↔ x ∈ Iio (-3)) :
  {x | bx^2 - (a + 2 * b) * x - 2 * b < 0} = Iio (-2 / 3) ∪ Ioi 3 := 
sorry

end quadratic_solution_set_l233_233333


namespace ratio_of_area_l233_233068

theorem ratio_of_area (P T a b c : ℝ) (hP_in_triangle : P ∈ T) 
(hParallel_a : parallel (line_through P) (side1 T))
(hParallel_b : parallel (line_through P) (side2 T))
(hParallel_c : parallel (line_through P) (side3 T))
(hSum_sides : side1 T = a + b + c ∧ side2 T = a + b + c ∧ side3 T = a + b + c)
(hArea_proportional : ∀ x, area (triangle_through P parallel_to x) = x^2)
(hTotal_area_T : area T = (a + b + c)^2) :
  let f := (a^2 + b^2 + c^2) / (a + b + c)^2 in
  f ≥ 1 / 3 ∧ (f = 1 / 3 ↔ is_centroid P T) :=
sorry

end ratio_of_area_l233_233068


namespace greatest_prime_factor_of_341_l233_233536

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l233_233536


namespace abs_sum_values_l233_233876

theorem abs_sum_values (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : 
  x + y = 7 ∨ x + y = 3 := 
by
  sorry

end abs_sum_values_l233_233876


namespace shortest_distance_between_ant_and_spider_l233_233111

-- Definitions based on the problem conditions
def ant_position (a : ℝ) : ℝ × ℝ := (a, real.sqrt (1 - a^2))
def spider_position (s : ℝ) : ℝ × ℝ := (s, 0)

-- Condition on the relationship between the ant's and spider's positions
def spider_position_relation (a : ℝ) : ℝ := 1 - 2 * a

-- Distance formula between ant and spider
def distance (a s : ℝ) : ℝ :=
  real.sqrt ((a - s)^2 + (real.sqrt (1 - a^2))^2)

-- Main theorem to prove the shortest distance
theorem shortest_distance_between_ant_and_spider : 
  ∃ a : ℝ, distance a (spider_position_relation a) = real.sqrt 14 / 4 :=
sorry

end shortest_distance_between_ant_and_spider_l233_233111


namespace prime_if_p_eq_1_l233_233152

theorem prime_if_p_eq_1 (p : ℕ) : (2^(2*...*2) + 9 is prime) ↔ p = 1 :=
sorry

end prime_if_p_eq_1_l233_233152


namespace probability_of_event_correct_l233_233408

def within_interval (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ Real.pi

def tan_in_range (x : ℝ) : Prop :=
  -1 ≤ Real.tan x ∧ Real.tan x ≤ Real.sqrt 3

def valid_subintervals (x : ℝ) : Prop :=
  within_interval x ∧ tan_in_range x

def interval_length (a b : ℝ) : ℝ :=
  b - a

noncomputable def probability_of_event : ℝ :=
  (interval_length 0 (Real.pi / 3) + interval_length (3 * Real.pi / 4) Real.pi) / Real.pi

theorem probability_of_event_correct :
  probability_of_event = 7 / 12 := sorry

end probability_of_event_correct_l233_233408


namespace find_a_minus_b_l233_233805

theorem find_a_minus_b (a b c d : ℕ) (h1 : a^5 = b^4) (h2 : c^3 = d^2) (h3 : c - a = 9) : a - b = -16 :=
by sorry

end find_a_minus_b_l233_233805


namespace percent_divisible_by_6_l233_233626

theorem percent_divisible_by_6 (N : ℕ) (hN : N = 120) :
  (∃ M, M = (finset.univ.filter (λ n : ℕ, n ≤ N ∧ n % 6 = 0)).card ∧ M * 6 = N) →
  (M.to_real / N.to_real) * 100 = 16.66666667 :=
by
  intros h
  sorry

end percent_divisible_by_6_l233_233626


namespace total_points_scored_l233_233720

theorem total_points_scored 
  (darius_score : ℕ) 
  (marius_score : ℕ) 
  (matt_score : ℕ) 
  (h1 : marius_score = darius_score + 3) 
  (h2 : darius_score = matt_score - 5)
  (h3 : darius_score = 10) : darius_score + marius_score + matt_score = 38 :=
by
  sorry

end total_points_scored_l233_233720


namespace Sahil_transportation_charges_l233_233985

theorem Sahil_transportation_charges
  (cost_machine : ℝ)
  (cost_repair : ℝ)
  (actual_selling_price : ℝ)
  (profit_percentage : ℝ)
  (transportation_charges : ℝ)
  (h1 : cost_machine = 12000)
  (h2 : cost_repair = 5000)
  (h3 : profit_percentage = 0.50)
  (h4 : actual_selling_price = 27000)
  (h5 : transportation_charges + (cost_machine + cost_repair) * (1 + profit_percentage) = actual_selling_price) :
  transportation_charges = 1500 :=
by
  sorry

end Sahil_transportation_charges_l233_233985


namespace nonnegative_solutions_eq_one_l233_233291

theorem nonnegative_solutions_eq_one : 
  (finset.univ.filter (λ x : ℝ, x^2 = -6 * x ∧ 0 <= x)).card = 1 := 
by 
  sorry

end nonnegative_solutions_eq_one_l233_233291


namespace find_c_values_l233_233913

noncomputable def line_intercept_product (c : ℝ) : Prop :=
  let x_intercept := -c / 8
  let y_intercept := -c / 5
  x_intercept * y_intercept = 24

theorem find_c_values :
  ∃ c : ℝ, (line_intercept_product c) ∧ (c = 8 * Real.sqrt 15 ∨ c = -8 * Real.sqrt 15) :=
by
  sorry

end find_c_values_l233_233913


namespace last_student_position_l233_233004

theorem last_student_position :
  ∃ n, n = 1458 ∧ 
    (∀ k, k > 0 → 
      let num_students := 2012 in
      let positions := list.range' 1 num_students in
      (k = 1 → positions.countp (λ p, p % 2 = 0) = 1006) →
      ∀ m, m >= 1 → ∃ r, r < 3 ∧ ∀ j, (j >= 1 → n = j % (3^m))
    ) :=
begin
  sorry
end

end last_student_position_l233_233004


namespace base_five_equivalent_of_156_is_1111_l233_233510

theorem base_five_equivalent_of_156_is_1111 : nat_to_base 5 156 = [1, 1, 1, 1] := 
sorry

end base_five_equivalent_of_156_is_1111_l233_233510


namespace greatest_prime_factor_341_l233_233583

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l233_233583


namespace profit_per_meter_is_30_l233_233073

-- Define the conditions given in the problem
def total_meters : ℕ := 40
def selling_price : ℝ := 8200
def total_profit : ℝ := 1200

-- Define the cost price based on the conditions
def cost_price (sp : ℝ) (profit : ℝ) : ℝ := sp - profit

-- Calculate profit per meter
def profit_per_meter (profit : ℝ) (meters : ℕ) : ℝ := profit / meters

-- The statement to prove
theorem profit_per_meter_is_30 : profit_per_meter total_profit total_meters = 30 :=
by 
  unfold profit_per_meter
  unfold total_profit total_meters
  rw [nat.cast_bit0, nat.cast_bit0, nat.cast_one]
  norm_num
  done

end profit_per_meter_is_30_l233_233073


namespace nonnegative_solutions_eq_one_l233_233287

theorem nonnegative_solutions_eq_one : 
  (finset.univ.filter (λ x : ℝ, x^2 = -6 * x ∧ 0 <= x)).card = 1 := 
by 
  sorry

end nonnegative_solutions_eq_one_l233_233287


namespace y_intercept_of_lineB_l233_233966

-- Define the linear equation y = 3x - 6
def line1 (x : ℝ) : ℝ := 3 * x - 6

-- Define the parallel line b passing through point (3, 4)
def lineB (x : ℝ) : ℝ := 3 * x + c

-- The point (3, 4) lying on line b
def point_on_lineB : Prop := lineB 3 = 4

-- Formalize the proof goal: the y-intercept of line b
theorem y_intercept_of_lineB : point_on_lineB → (c = -5) := 
by
  sorry

end y_intercept_of_lineB_l233_233966


namespace juice_fraction_left_l233_233361

theorem juice_fraction_left (initial_juice : ℝ) (given_juice : ℝ) (remaining_juice : ℝ) : 
  initial_juice = 5 → given_juice = 18/4 → remaining_juice = initial_juice - given_juice → remaining_juice = 1/2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end juice_fraction_left_l233_233361


namespace length_of_train_is_approx_300_l233_233689

-- Define the given conditions
def train_speed_kmph : ℝ := 68
def man_speed_kmph : ℝ := 8
def passing_time_s : ℝ := 17.998560115190784

-- Conversion factor
def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

-- Relative speed
def relative_speed_mps : ℝ :=
  kmph_to_mps (train_speed_kmph - man_speed_kmph)

-- Length of the train
def length_of_train : ℝ :=
  relative_speed_mps * passing_time_s

-- The theorem to prove
theorem length_of_train_is_approx_300 :
  abs (length_of_train - 300) < 0.01 := 
by
  sorry

end length_of_train_is_approx_300_l233_233689


namespace nonnegative_solutions_eq1_l233_233248

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l233_233248


namespace reciprocal_of_2023_l233_233451

theorem reciprocal_of_2023 :
  1 / 2023 = 1 / (2023 : ℝ) :=
by
  sorry

end reciprocal_of_2023_l233_233451


namespace maximum_sum_S6_l233_233908

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + (n - 1) * d

def sum_arithmetic_sequence (a d : α) (n : ℕ) : α :=
  (n : α) / 2 * (2 * a + (n - 1) * d)

theorem maximum_sum_S6 (a d : α)
  (h1 : arithmetic_sequence a d 3 + arithmetic_sequence a d 10 < 0)
  (h2 : sum_arithmetic_sequence a d 11 > 0) :
  ∀ n : ℕ, sum_arithmetic_sequence a d n ≤ sum_arithmetic_sequence a d 6 :=
by sorry

end maximum_sum_S6_l233_233908


namespace joey_pays_monica_l233_233393

theorem joey_pays_monica (M J C : ℝ) (h1: M < J) (h2: J < C) :
  let total_expense := M + J + C
  let equal_share := total_expense / 3
  let joey_additional_payment := (M + C - 2 * J) / 3
  (joey_additional_payment = (C - J - M) / 3) :=
by
  indulge in
  let total_expense := M + J + C
  let equal_share := total_expense / 3
  let joey_additional_payment := (M + C - 2 * J) / 3
  sorry 

end joey_pays_monica_l233_233393


namespace true_propositions_l233_233820

def proposition1 (a b : ℝ) : Prop :=
  ∀ (a + b : ℚ), (a : ℚ) ∧ (b : ℚ)

def proposition2 (a b : ℝ) : Prop :=
  (a + b ≥ 2) → (a ≥ 1 ∨ b ≥ 1)

def proposition3 (a b x : ℝ) : Prop :=
  (a * x + b > 0) → (x > -b / a)

def proposition4 (a b c : ℝ) : Prop :=
  (∃ x, a * x^2 + b * x + c = 0 ∧ x = 1) ↔ a + b + c = 0

theorem true_propositions : 
  (¬ proposition1 ∧ proposition2 ∧ ¬ proposition3 ∧ proposition4) := 
  by sorry

end true_propositions_l233_233820


namespace tod_north_distance_l233_233482

-- Given conditions as variables
def speed : ℕ := 25  -- speed in miles per hour
def time : ℕ := 6    -- time in hours
def west_distance : ℕ := 95  -- distance to the west in miles

-- Prove the distance to the north given conditions
theorem tod_north_distance : time * speed - west_distance = 55 := by
  sorry

end tod_north_distance_l233_233482


namespace greatest_prime_factor_of_341_l233_233610

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l233_233610


namespace nonnegative_solution_count_l233_233270

theorem nonnegative_solution_count : 
  ∃! x : ℝ, x^2 = -6 * x ∧ 0 ≤ x := 
begin
  sorry
end

end nonnegative_solution_count_l233_233270


namespace square_floor_tile_count_l233_233071

theorem square_floor_tile_count (n : ℕ) (h : 2 * n - 1 = 49) : n^2 = 625 := by
  sorry

end square_floor_tile_count_l233_233071


namespace total_pears_picked_l233_233364

theorem total_pears_picked (Keith_picked : ℕ) (Jason_picked : ℕ) (Joan_picked : ℕ) (h1 : Keith_picked = 3) (h2 : Jason_picked = 2) (h3 : Joan_picked = 5) : Keith_picked + Jason_picked = 5 :=
by
  rw [h1, h2]
  exact rfl

end total_pears_picked_l233_233364


namespace volume_and_surface_area_of_pyramid_l233_233995

theorem volume_and_surface_area_of_pyramid
  (l : ℝ)
  (h_pos : 0 < l) :
  let h := l / real.sqrt 2
  let base_area := (l / real.sqrt 2) ^ 2
  let volume := (1 / 3) * base_area * h
  let surface_area := base_area + 2 * ((1 / 2) * (l / real.sqrt 2) ^ 2) + 2 * ((1 / 2) * (l / real.sqrt 2) * l)
  in volume = l^3 * real.sqrt 2 / 12 ∧ surface_area = (l^2 * (2 + real.sqrt 2)) / 2 := 
by
  sorry

end volume_and_surface_area_of_pyramid_l233_233995


namespace intersection_eq_l233_233192

/-
Define the sets A and B
-/
def setA : Set ℝ := {-1, 0, 1, 2}
def setB : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

/-
Lean statement to prove the intersection A ∩ B equals {1, 2}
-/
theorem intersection_eq :
  setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_eq_l233_233192


namespace arithmetic_sequence_geometric_sequence_l233_233353

-- Arithmetic sequence proof problem
theorem arithmetic_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n - a (n - 1) = 2) :
  ∀ n, a n = 2 * n - 1 :=
by 
  sorry

-- Geometric sequence proof problem
theorem geometric_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, n ≥ 2 → a n / a (n - 1) = 2) :
  ∀ n, a n = 2 ^ (n - 1) :=
by 
  sorry

end arithmetic_sequence_geometric_sequence_l233_233353


namespace greatest_prime_factor_of_341_l233_233539

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l233_233539


namespace percent_divisible_by_six_up_to_120_l233_233630

theorem percent_divisible_by_six_up_to_120 : 
  let total_numbers := 120
  let divisible_by_six := total_numbers / 6
  let percentage := (divisible_by_six * 100) / total_numbers
  percentage = 50 / 3 := sorry

end percent_divisible_by_six_up_to_120_l233_233630


namespace find_xy_l233_233751

theorem find_xy (x y : ℝ) (h1 : x^5 + y^5 = 33) (h2 : x + y = 3) :
  (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by
  sorry

end find_xy_l233_233751


namespace magnitude_of_z_l233_233785

variable (z : ℂ)
variable (i : ℂ) [Complex.is_imaginary_unit i]

-- Given the condition
axiom h : i * z + 2 = z - 2 * i

-- Prove the magnitude of z is 2
theorem magnitude_of_z :  |z| = 2 :=
by
  sorry

end magnitude_of_z_l233_233785


namespace number_of_elements_cong_set_l233_233857

/-- Define a set of integers less than 500 and congruent to 3 modulo 7 -/
def cong_set : Set ℕ := {n | n < 500 ∧ n % 7 = 3}

/-- The theorem stating the number of elements in cong_set is 72 -/
theorem number_of_elements_cong_set : Set.card cong_set = 72 :=
sorry

end number_of_elements_cong_set_l233_233857


namespace expression_for_f_when_x_lt_0_l233_233790

noncomputable section

variable (f : ℝ → ℝ)

theorem expression_for_f_when_x_lt_0
  (hf_neg : ∀ x : ℝ, f (-x) = -f x)
  (hf_pos : ∀ x : ℝ, x > 0 → f x = x * abs (x - 2)) :
  ∀ x : ℝ, x < 0 → f x = x * abs (x + 2) :=
by
  sorry

end expression_for_f_when_x_lt_0_l233_233790


namespace coefficient_of_friction_correct_l233_233670

noncomputable def coefficient_of_friction (R Fg: ℝ) (α: ℝ) : ℝ :=
  (1 - R * real.cos α) / (R * real.sin α)

theorem coefficient_of_friction_correct
  (Fg: ℝ)
  (α: ℝ)
  (R: ℝ := 11 * Fg)
  (hα: α = real.pi * 80 / 180):
  coefficient_of_friction R Fg α = 0.17 :=
by
  sorry

end coefficient_of_friction_correct_l233_233670


namespace rational_abs_eq_l233_233869

theorem rational_abs_eq (a : ℚ) (h : |-3 - a| = 3 + |a|) : 0 ≤ a := 
by
  sorry

end rational_abs_eq_l233_233869


namespace line_equation_l233_233681

-- Define the conditions as given in the problem
def passes_through (P : ℝ × ℝ) (line : ℝ × ℝ) : Prop :=
  line.fst * P.fst + line.snd * P.snd + 1 = 0

def equal_intercepts (line : ℝ × ℝ) : Prop :=
  line.fst = line.snd

theorem line_equation (P : ℝ × ℝ) (hP : P = (-2, -1)) :
  (∃ (k : ℝ), passes_through P (1, -2 * k)) ∨ (∃ (m : ℝ), passes_through P (1, m) ∧ m = - 1) :=
sorry

end line_equation_l233_233681


namespace nonnegative_solutions_eq_one_l233_233292

theorem nonnegative_solutions_eq_one : 
  (finset.univ.filter (λ x : ℝ, x^2 = -6 * x ∧ 0 <= x)).card = 1 := 
by 
  sorry

end nonnegative_solutions_eq_one_l233_233292


namespace num_non_empty_proper_subsets_of_A_range_of_m_such_that_A_superset_B_l233_233124

-- Define the set A
def A : set ℤ := {x | x^2 - 3 * x - 10 ≤ 0}

-- Number of non-empty proper subsets of A
theorem num_non_empty_proper_subsets_of_A : 
  ∃ n, (∀ A_integers = ({-2, -1, 0, 1, 2, 3, 4, 5}: set ℤ)), n = 2^8 - 2 := sorry

-- Define the set B
def B (m : ℤ) : set ℤ := {x | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

-- Range of m such that A ⊇ B
theorem range_of_m_such_that_A_superset_B : 
  ∃ m : ℤ, A ⊇ B m ↔ (m < -2 ∨ (-1 ≤ m ∧ m ≤ 2)) := sorry

end num_non_empty_proper_subsets_of_A_range_of_m_such_that_A_superset_B_l233_233124


namespace finite_nat_numbers_with_sum_of_digits_2019th_powers_eq_n_l233_233700

theorem finite_nat_numbers_with_sum_of_digits_2019th_powers_eq_n :
  ∃ N : ℕ, ∀ n : ℕ, n > N → 
  (∑ d in (n.digits 10), d ^ 2019) ≠ n :=
sorry

end finite_nat_numbers_with_sum_of_digits_2019th_powers_eq_n_l233_233700


namespace greatest_prime_factor_of_341_l233_233540

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l233_233540


namespace rectangle_to_square_l233_233979

theorem rectangle_to_square (a : ℝ) : 
  ∃(parts : list (set (ℝ × ℝ))), 
    (∀p ∈ parts, is_rectangle p ∧ area p = area ⟨a, a⟩) ∧
    (∃s : set (ℝ × ℝ), is_square s ∧ area s = 5 * area ⟨a, a⟩ ∧ covers s parts) :=
begin
  sorry
end

def is_rectangle (p : set (ℝ × ℝ)) : Prop := sorry
def is_square (s : set (ℝ × ℝ)) : Prop := sorry
def area (p : set (ℝ × ℝ)) : ℝ := sorry
def covers (s : set (ℝ × ℝ)) (parts : list (set (ℝ × ℝ))) : Prop := sorry

end rectangle_to_square_l233_233979


namespace cars_in_front_parking_lot_initially_l233_233366

-- Given conditions
variables (F : ℕ) 
variable h1 : 2 * F = 2 * F -- Number of cars in the back parking lot is two times the number in the front
variable h2 : F + 2 * F = 3 * F -- Total number of cars initially
variable h3 : 400 + 300 = 700 -- Total number of cars at the end of the play
variable h4 : 700 - 300 = 400 -- Number of cars before the additional 300 arrived

theorem cars_in_front_parking_lot_initially (h : 3 * F = 400) : F = 133 :=
by {
  -- The proof will involve showing that F = 133 given h: 3 * F = 400
  trace "Since 3*F = 400, we need to show F = 133.";
  sorry
}

end cars_in_front_parking_lot_initially_l233_233366


namespace mary_saturday_earning_l233_233391

theorem mary_saturday_earning 
  (h1 : ∀ d ∈ ["Monday", "Wednesday", "Friday"], MaryWorksHours d = 9)
  (h2 : ∀ d ∈ ["Tuesday", "Thursday"], MaryWorksHours d = 5)
  (h3 : MaryWorksHours "Saturday" = 4 ∨ MaryWorksHours "Saturday" = 0)
  (h4 : MaryWorksHours "Sunday" = 0)
  (h5 : MaryWeeklyEarnings false = 407)
  (h6 : MaryWeeklyEarnings true = 483) :
  MarySaturdayHourlyWage = 19 := sorry

end mary_saturday_earning_l233_233391


namespace polynomial_P_l233_233373

noncomputable def P (x : ℝ) : ℝ := x^3 + 3 * x^2 + 5 * x + 4
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

lemma a_b_c_are_roots_of_cubic : a^3 + 4 * a^2 + 6 * a + 8 = 0 ∧ b^3 + 4 * b^2 + 6 * b + 8 = 0 ∧ c^3 + 4 * c^2 + 6 * c + 8 = 0 := sorry

lemma P_satisfies_conditions (P : ℝ → ℝ) : 
  P a = b + c ∧ 
  P b = a + c ∧ 
  P c = a + b ∧ 
  P (a + b + c) = -20 := 
begin
  split,
  { -- P(a) = b + c
    sorry },
  split,
  { -- P(b) = a + c
    sorry },
  split,
  { -- P(c) = a + b
    sorry },
  { -- P(a + b + c) = -20
    sorry },
end

theorem polynomial_P (P : ℝ → ℝ) : P = λ x, x^3 + 3 * x^2 + 5 * x + 4 :=
begin
  apply funext,
  intro x,
  have h : P = λ x, x^3 + 3 * x^2 + 5 * x + 4,
  { sorry },
  exact h x
end

end polynomial_P_l233_233373


namespace number_of_nonnegative_solutions_l233_233241

-- Define the equation x^2 = -6x
def equation (x : ℝ) : Prop := x^2 = -6 * x

-- Define the condition for a nonnegative solution
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Define the main theorem to prove the number of nonnegative solutions
theorem number_of_nonnegative_solutions : 
  {x : ℝ // equation x ∧ nonnegative x}.to_finset.card = 1 :=
sorry

end number_of_nonnegative_solutions_l233_233241


namespace smallest_positive_integer_with_remainders_l233_233615

theorem smallest_positive_integer_with_remainders : ∃ (x : ℕ), ∀ y : ℕ, y > 0 → 
  ((x % 2 = 1) ∧ (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 10 = 9)) ∧
  ((y % 2 = 1) ∧ (y % 3 = 2) ∧ (y % 4 = 3) ∧ (y % 10 = 9)) → x ≤ y :=
by
  use 59
  split
  {
    intros
    split
    { exact x % 2 = 1 }
    split
    { exact x % 3 = 2 }
    split
    { exact x % 4 = 3 }
    { exact x % 10 = 9 }
  }
  sorry

end smallest_positive_integer_with_remainders_l233_233615


namespace fraction_evaluation_l233_233747

theorem fraction_evaluation : (1 / (2 + 1 / (3 + 1 / 4))) = (13 / 30) := by
  sorry

end fraction_evaluation_l233_233747


namespace inequality_proof_l233_233936

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c)) ≥ (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) :=
by
  sorry

end inequality_proof_l233_233936


namespace nonnegative_solutions_eq_one_l233_233295

theorem nonnegative_solutions_eq_one : 
  (finset.univ.filter (λ x : ℝ, x^2 = -6 * x ∧ 0 <= x)).card = 1 := 
by 
  sorry

end nonnegative_solutions_eq_one_l233_233295


namespace number_of_divisors_of_8n3_l233_233377

theorem number_of_divisors_of_8n3 (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ p : ℕ, nat.prime p ∧ n = p ^ 16) : 
  nat.totient (8 * n ^ 3) = 196 := by
  sorry

end number_of_divisors_of_8n3_l233_233377


namespace ca1_l233_233325

theorem ca1 {
  a b : ℝ
} (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := 
by
  sorry

end ca1_l233_233325


namespace reflected_beam_deviation_l233_233680

theorem reflected_beam_deviation (α β : ℝ) : 
  let deviation_angle := 2 * Real.arcsin (Real.sin α * Real.sin β) in
  deviation_angle = 2 * Real.arcsin (Real.sin α * Real.sin β) :=
by sorry

end reflected_beam_deviation_l233_233680


namespace find_number_l233_233326

-- Define the condition that k is a non-negative integer
def is_nonnegative_int (k : ℕ) : Prop := k ≥ 0

-- Define the condition that 18^k is a divisor of the number n
def is_divisor (n k : ℕ) : Prop := 18^k ∣ n

-- The main theorem statement
theorem find_number (n k : ℕ) (h_nonneg : is_nonnegative_int k) (h_eq : 6^k - k^6 = 1) (h_div : is_divisor n k) : n = 1 :=
  sorry

end find_number_l233_233326


namespace probability_A_not_losing_l233_233404

variable (P_A_wins : ℝ)
variable (P_draw : ℝ)
variable (P_A_not_losing : ℝ)

theorem probability_A_not_losing 
  (h1 : P_A_wins = 0.3) 
  (h2 : P_draw = 0.5) 
  (h3 : P_A_not_losing = P_A_wins + P_draw) :
  P_A_not_losing = 0.8 :=
sorry

end probability_A_not_losing_l233_233404


namespace greatest_prime_factor_341_l233_233575

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l233_233575


namespace real_solutions_f_eq_x_l233_233157

noncomputable def f (x: ℝ) := Σ' (n : ℕ) (hn : 2 ≤ n + 2 ∧ n + 2 ≤ 101), (n + 1) / (x - (n + 2))

theorem real_solutions_f_eq_x : 
  (∃ x : ℝ, f x = x) ↔ (#ℝ = 101) := 
by
  sorry

end real_solutions_f_eq_x_l233_233157


namespace garden_perimeter_equals_104_l233_233022

theorem garden_perimeter_equals_104 :
  let playground_length := 16
  let playground_width := 12
  let playground_area := playground_length * playground_width
  let garden_width := 4
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * garden_length + 2 * garden_width
  playground_area = 192 ∧ garden_perimeter = 104 :=
by {
  -- Declarations
  let playground_length := 16
  let playground_width := 12
  let playground_area := playground_length * playground_width
  let garden_width := 4
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * garden_length + 2 * garden_width

  -- Assertions
  have area_playground : playground_area = 192 := by sorry
  have perimeter_garden : garden_perimeter = 104 := by sorry

  -- Conclusion
  exact ⟨area_playground, perimeter_garden⟩
}

end garden_perimeter_equals_104_l233_233022


namespace probability_X_eq_zero_minimum_k_for_nonnegative_expected_profit_l233_233058

noncomputable def profit_probability_eq_zero : ℚ :=
  let num_balls := 6 in
  let draws := 3 in
  let specified_color_prob := 1 / num_balls in
  let other_color_prob := 1 - specified_color_prob in
  draws * specified_color_prob * (other_color_prob ^ 2)

theorem probability_X_eq_zero :
  profit_probability_eq_zero = 25 / 72 :=
by
  sorry

noncomputable def expected_profit (k : ℕ) : ℚ :=
  let num_balls := 6 in
  let draws := 3 in
  let specified_color_prob := 1 / num_balls in
  let other_color_prob := 1 - specified_color_prob in
  k * (specified_color_prob ^ 3) + 
  (-1) * (other_color_prob ^ 3) + 
  1 * (draws * (specified_color_prob ^ 2) * other_color_prob) + 
  0 * (draws * specified_color_prob * (other_color_prob ^ 2))

theorem minimum_k_for_nonnegative_expected_profit :
  ∃ k, expected_profit k ≥ 0 ∧ k = 110 :=
by
  sorry

end probability_X_eq_zero_minimum_k_for_nonnegative_expected_profit_l233_233058


namespace problem_equiv_l233_233195

theorem problem_equiv : 
  ∀ (a b : ℝ) (a_n S : ℕ → ℝ),
    (∀ n : ℕ, a_n (2 + Real.sin (n * Real.pi / 2)) = n * (2 + Real.cos (n * Real.pi))) →
    (∀ n : ℕ, S (4 * n) = a * n^2 + b * n) →
    a - b = 5 := 
by
    intros a b a_n S h1 h2
    sorry

end problem_equiv_l233_233195


namespace IvanPetrovich_daily_lessons_and_charity_l233_233039

def IvanPetrovichConditions (L k : ℕ) : Prop :=
  24 = 8 + 3*L + k ∧
  3000 * L * 21 + 14000 = 70000 + (7000 * k / 3)

theorem IvanPetrovich_daily_lessons_and_charity
  (L k : ℕ) (h : IvanPetrovichConditions L k) :
  L = 2 ∧ 7000 * k / 3 = 70000 := 
by
  sorry

end IvanPetrovich_daily_lessons_and_charity_l233_233039


namespace clark_discount_l233_233099

noncomputable def price_per_part : ℕ := 80
noncomputable def num_parts : ℕ := 7
noncomputable def total_paid : ℕ := 439

theorem clark_discount : (price_per_part * num_parts - total_paid) = 121 :=
by
  -- proof goes here
  sorry

end clark_discount_l233_233099


namespace ivan_petrovich_lessons_daily_l233_233038

def donations_per_month (L k : ℕ) : ℕ := 21 * (k / 3) * 1000

theorem ivan_petrovich_lessons_daily (L k : ℕ) (h1 : 24 = 8 + L + 2 * L + k) (h2 : k = 16 - 3 * L)
    (income_from_lessons : 21 * (3 * L) * 1000)
    (rent_income : 14000)
    (monthly_expenses : 70000)
    (charity_donations : donations_per_month L k) :
  L = 2 ∧ charity_donations = 70000 := 
begin
  sorry
end

end ivan_petrovich_lessons_daily_l233_233038


namespace percent_divisible_by_six_up_to_120_l233_233628

theorem percent_divisible_by_six_up_to_120 : 
  let total_numbers := 120
  let divisible_by_six := total_numbers / 6
  let percentage := (divisible_by_six * 100) / total_numbers
  percentage = 50 / 3 := sorry

end percent_divisible_by_six_up_to_120_l233_233628


namespace BC_distance_l233_233480

noncomputable def distance_ABC (n : ℝ) : ℝ := 10 * Real.sqrt 2 * n
noncomputable def angle_A := Real.pi / 3  -- 60 degrees in radians
noncomputable def angle_B := 5 * Real.pi / 12  -- 75 degrees in radians
noncomputable def angle_C := Real.pi / 4  -- 45 degrees in radians

theorem BC_distance (n : ℝ) : 
    let AB := 10 * Real.sqrt 2 * n in
    let sinA := Real.sin (Real.pi / 3) in
    let sinC := Real.sin (Real.pi / 4) in
    let BC := AB * (sinA / sinC) 
    in BC = 10 * Real.sqrt 3 * n :=
by
  sorry

end BC_distance_l233_233480


namespace union_of_complements_l233_233843

def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {x | x^2 + 4 = 5 * x}
def complement_U (S : Set ℕ) : Set ℕ := {x ∈ U | x ∉ S}

theorem union_of_complements :
  complement_U A ∪ complement_U B = {0, 2, 3, 4, 5} := by
sorry

end union_of_complements_l233_233843


namespace arithmetic_progression_pairs_l233_233728

noncomputable theory

open Real

theorem arithmetic_progression_pairs :
  {p : ℝ × ℝ | 10, p.1, p.2, p.1 * p.2 is arithmetic_progression} = {(4, -2), (2.5, -5)} :=
sorry

end arithmetic_progression_pairs_l233_233728


namespace rationalize_denominator_ABC_value_l233_233409

def A := 11 / 4
def B := 5 / 4
def C := 5

theorem rationalize_denominator : 
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

theorem ABC_value :
  A * B * C = 275 :=
sorry

end rationalize_denominator_ABC_value_l233_233409


namespace number_of_sequences_l233_233929

theorem number_of_sequences (n : ℕ) (a : Fin (n + 1) → Fin 4) :
  (n = ∑ i in Finset.range (n + 1), (2^i) * a i) →
  (∃ b_n : ℕ, b_n = (n / 2).floor + 1) :=
by
  sorry

end number_of_sequences_l233_233929


namespace greatest_prime_factor_341_l233_233599

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l233_233599


namespace coin_toss_tails_count_l233_233025

theorem coin_toss_tails_count (flips : ℕ) (frequency_heads : ℝ) (h_flips : flips = 20) (h_frequency_heads : frequency_heads = 0.45) : 
  (20 : ℝ) * (1 - 0.45) = 11 := 
by
  sorry

end coin_toss_tails_count_l233_233025


namespace swimming_pool_volume_l233_233113

noncomputable def radius := 10
noncomputable def height := 5
noncomputable def diameter := 2 * radius

theorem swimming_pool_volume :
  diameter = 20 → height = 5 →
  (π * radius^2 * height + (2 / 3) * π * radius^3) = (3500 / 3) * π :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end swimming_pool_volume_l233_233113


namespace value_of_m_l233_233176

variable (a m : ℝ)
variable (h1 : a > 0)
variable (h2 : -a*m^2 + 2*a*m + 3 = 3)
variable (h3 : m ≠ 0)

theorem value_of_m : m = 2 :=
by
  sorry

end value_of_m_l233_233176


namespace tangent_parallel_lines_l233_233996

variable (S₁ S₂ : Type) 
variables (O₁ O₂ A B K L : S₁) 

-- Defining the circles and their centers
variable [IsCircle S₁ S₂]

-- Centers of circles
variable [Center O₁ S₁]
variable [Center O₂ S₂]

-- Points of intersection
variable [Intersects S₁ S₂ A]
variable [Intersects S₁ S₂ B]

-- Tangents intersecting line segments at points K and L
variable (tangent_A_to_B_O₂ : Tangent S₁ A K)
variable (tangent_S₁_to_A_B_O₁ : Tangent S₂ A L)

-- Prove parallelism
theorem tangent_parallel_lines : KL ∥ O₁ O₂ := 
sorry

end tangent_parallel_lines_l233_233996


namespace permutation_log_product_eq_one_l233_233956

theorem permutation_log_product_eq_one (π : {σ : Finset.Perm (Finset.Icc 2 2012) // true}) :
  (Finset.prod (Finset.filter (λ i, 2 ≤ i ∧ i ≤ 2012) (Finset.Icc 2 2012)) 
    (λ i, Real.log (π.val i) / Real.log i)) = 1 := 
by 
  sorry

end permutation_log_product_eq_one_l233_233956


namespace initial_amount_of_liquid_A_l233_233679

theorem initial_amount_of_liquid_A (A B : ℕ) (x : ℕ) (h1 : 4 * x = A) (h2 : x = B) (h3 : 4 * x + x = 5 * x)
    (h4 : 4 * x - 8 = 3 * (x + 8) / 2) : A = 16 :=
  by
  sorry

end initial_amount_of_liquid_A_l233_233679


namespace find_m_and_n_eq_max_loquat_le_60_l233_233419

/-- Define all the conditions and the resulting proof problems based on them --/
noncomputable def find_m_and_n (m n : ℕ) : Prop :=
  (n = m + 28) ∧ (80 * m + 120 * n = 4960)

noncomputable def max_loquat (x : ℕ) : Prop :=
  ∃ y : ℕ, 160 * 12 + 8 * x - 12 * x + 36 * y = 5280 ∧
  (10 * x + 16 * (160 - x) + 42 * y - (8 * x + 12 * (160 - x) + 36 * y) ≥ 1120)

-- Statement to find m and n
theorem find_m_and_n_eq : find_m_and_n 8 36 :=
  by
    unfold find_m_and_n
    exact ⟨rfl, rfl⟩

-- Statement to prove that the maximum amount of loquat the store can wholesale is 60 kg
theorem max_loquat_le_60 : max_loquat 60 :=
  by
    unfold max_loquat
    exact ⟨62, rfl, sorry⟩  -- Not including the actual calculations

end find_m_and_n_eq_max_loquat_le_60_l233_233419


namespace total_equipment_cost_l233_233465

-- Define the cost of each piece of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8

-- Define the number of players
def players : ℕ := 16

-- Define the total cost of equipment for one player
def equipment_cost_per_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Define the total cost for all players
def total_cost : ℝ := players * equipment_cost_per_player

-- The proof problem to be stated:
theorem total_equipment_cost (jc sc k p : ℝ) (n : ℕ) :
  jc = 25 ∧ sc = 15.2 ∧ k = 6.8 ∧ p = 16 →
  total_cost = 752 :=
by
  intro h
  rcases h with ⟨hc1, hc2, hc3, hc4⟩
  simp [total_cost, equipment_cost_per_player, hc1, hc2, hc3, hc4]
  exact sorry

end total_equipment_cost_l233_233465


namespace nonnegative_solution_count_l233_233276

theorem nonnegative_solution_count : 
  ∃! x : ℝ, x^2 = -6 * x ∧ 0 ≤ x := 
begin
  sorry
end

end nonnegative_solution_count_l233_233276


namespace skipping_times_eq_l233_233453

theorem skipping_times_eq (x : ℝ) (h : x > 0) :
  180 / x = 240 / (x + 5) :=
sorry

end skipping_times_eq_l233_233453


namespace greatest_prime_factor_of_341_l233_233611

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l233_233611


namespace children_group_size_l233_233110

theorem children_group_size (x : ℕ) (h1 : 255 % 17 = 0) (h2: ∃ n : ℕ, n * 17 = 255) 
                            (h3 : ∀ a c, a = c → a = 255 → c = 255 → x = 17) : 
                            (255 / x = 15) → x = 17 :=
by
  sorry

end children_group_size_l233_233110


namespace right_angled_triangles_in_set_A_l233_233840

-- Define the problem statement in Lean
theorem right_angled_triangles_in_set_A (n : ℕ) (h : n ≥ 2) :
  let A := {z : ℂ | ∃ (k : ℕ), k < 2 * n ∧ z = complex.exp(2 * real.pi * complex.I * (k : ℝ) / n)} in
  (n % 2 = 1 → (set.count (λ (x : complex × complex × complex), 
    (x.fst.fst - x.fst.snd).re * (x.fst.fst - x.snd).re +
    (x.fst.snd - x.snd).re * (x.fst.snd - x.fst.fst).re = 0) (set.unordered { (x, y, z) | x ∈ A ∧ y ∈ A ∧ z ∈ A})) = 2 * n * (n - 1)) ∧
  (n % 2 = 0 → (set.count (λ (x : complex × complex × complex), 
    (x.fst.fst - x.fst.snd).re * (x.fst.fst - x.snd).re +
    (x.fst.snd - x.snd).re * (x.fst.snd - x.fst.fst).re = 0) (set.unordered { (x, y, z) | x ∈ A ∧ y ∈ A ∧ z ∈ A})) = 2 * n^2) :=
by sorry

end right_angled_triangles_in_set_A_l233_233840


namespace count_valid_pairs_l233_233801

open Finset

def A : Finset ℕ := {0, 1, 2, 3, 4, 5, 9}

theorem count_valid_pairs : (∑ a in A, ∑ b in A, if a ≠ b ∧ a > 0 ∧ b < 4 then 1 else 0) = 21 := by
  sorry

end count_valid_pairs_l233_233801


namespace div_f_n_l233_233769

-- The mathematical problem statement
theorem div_f_n {f : ℕ → ℕ} (h1 : ∀ n ≥ 1, ∃ lst : List (List ℕ),
  lst.All (λ l, l.head = 1 ∧ (∀ i < l.length - 1, l.get i ∣ l.get (i + 1))) ∧
  lst.All (λ l, l.last = n) ∧
  f n = lst.length)
  (N : ℕ) (hN : N ≥ 1) :
  ∃ n : ℕ, n ≥ 1 ∧ N ∣ f n :=
by
  sorry

end div_f_n_l233_233769


namespace percentage_divisible_by_6_l233_233638

-- Defining the sets S and T using Lean
def S := {n : ℕ | 1 ≤ n ∧ n ≤ 120}
def T := {n : ℕ | n ∈ S ∧ 6 ∣ n}

-- Proving the percentage of elements in T with respect to S is 16.67%
theorem percentage_divisible_by_6 : 
  (↑(T.card) : ℚ) / (S.card) * 100 = 16.67 := sorry

end percentage_divisible_by_6_l233_233638


namespace evaluate_expression_at_x_eq_3_l233_233745

theorem evaluate_expression_at_x_eq_3 : (∃ x : ℕ, x = 3 ∧ (x^x)^x = 19683) :=
by 
  use 3
  split
  { refl }
  { sorry }

end evaluate_expression_at_x_eq_3_l233_233745


namespace remainder_of_division_l233_233763

noncomputable def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 1
noncomputable def g (x : ℝ) : ℝ := (x - 3) ^ 2
noncomputable def remainder (x : ℝ) : ℝ := 324 * x - 488

theorem remainder_of_division :
  ∀ (x : ℝ), (f x) % (g x) = remainder x :=
sorry

end remainder_of_division_l233_233763


namespace greatest_prime_factor_341_l233_233520

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l233_233520


namespace complete_pairs_of_socks_l233_233118

def initial_pairs_blue : ℕ := 20
def initial_pairs_green : ℕ := 15
def initial_pairs_red : ℕ := 15

def lost_socks_blue : ℕ := 3
def lost_socks_green : ℕ := 2
def lost_socks_red : ℕ := 2

def donated_socks_blue : ℕ := 10
def donated_socks_green : ℕ := 15
def donated_socks_red : ℕ := 10

def purchased_pairs_blue : ℕ := 5
def purchased_pairs_green : ℕ := 3
def purchased_pairs_red : ℕ := 2

def gifted_pairs_blue : ℕ := 2
def gifted_pairs_green : ℕ := 1

theorem complete_pairs_of_socks : 
  (initial_pairs_blue - 1 - (donated_socks_blue / 2) + purchased_pairs_blue + gifted_pairs_blue) +
  (initial_pairs_green - 1 - (donated_socks_green / 2) + purchased_pairs_green + gifted_pairs_green) +
  (initial_pairs_red - 1 - (donated_socks_red / 2) + purchased_pairs_red) = 43 := by
  sorry

end complete_pairs_of_socks_l233_233118


namespace cylinder_radius_ratio_l233_233493

theorem cylinder_radius_ratio (h : ℝ) (r1 r2 V1 V2 : ℝ) (V1_eq : V1 = 40) (V2_eq : V2 = 360) (height_eq : ∀ h1 h2, h1 = h2) (V1_formula : V1 = π * r1^2 * h) (V2_formula : V2 = π * r2^2 * h) : r1 / r2 = 1 / 3 := 
by
  -- Using the given formulas and conditions to prove the ratio of radii
  sorry

end cylinder_radius_ratio_l233_233493


namespace train_speed_is_60_kmph_l233_233074

-- Define the conditions
def length_of_train : ℕ := 300 -- in meters
def time_to_cross_pole : ℕ := 18 -- in seconds

-- Define the conversions
def meters_to_kilometers (m : ℕ) : ℝ := m / 1000.0
def seconds_to_hours (s : ℕ) : ℝ := s / 3600.0

-- Define the speed calculation
def speed_km_per_hr (distance_km : ℝ) (time_hr : ℝ) : ℝ := distance_km / time_hr

-- Prove that the speed of the train is 60 km/hr
theorem train_speed_is_60_kmph :
  speed_km_per_hr (meters_to_kilometers length_of_train) (seconds_to_hours time_to_cross_pole) = 60 := 
  by
    sorry

end train_speed_is_60_kmph_l233_233074


namespace second_pipe_time_l233_233496

theorem second_pipe_time (T : ℝ) (h_approx: T ≈ 22.86) :
  (1 / 20 + 1 / T) * (2 / 3) * 16 = 1 :=
by
-- Here is a place where assumptions should be made clear
have h: (1 / 20 + 1 / T) * (2 / 3) * 16 = 1,
  sorry -- Calculation details omitted
exact h

end second_pipe_time_l233_233496


namespace greatest_prime_factor_341_l233_233595

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l233_233595


namespace part_I_part_II_l233_233962

-- Define f and its derivative
def f (x b : ℝ) : ℝ := (x + b) * Real.log x
noncomputable def f' (x b : ℝ) : ℝ := Real.log x + b / x + 1

-- Statement (I) - Prove b = 1 given the slope condition
theorem part_I (h : f' 1 b = 2) : b = 1 := sorry

-- Define g and its derivative
def g (x a b : ℝ) : ℝ := Real.exp x * ((f x b) / (x + 1) - a)
noncomputable def g' (x a b : ℝ) : ℝ := (1 / x - a + Real.log x) * Real.exp x

-- Define h and its derivative
def h (x : ℝ) : ℝ := 1 / x + Real.log x
noncomputable def h' (x : ℝ) : ℝ := (x - 1) / (x^2)

-- Statement (II) - Prove range of a is (-∞, 1] given the monotonicity condition
theorem part_II (h_monitonic : ∀ x > 0, g' x a 1 ≥ 0) : a ≤ 1 := sorry

end part_I_part_II_l233_233962


namespace nonnegative_solutions_count_l233_233313

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.card = 1 := 
sorry

end nonnegative_solutions_count_l233_233313


namespace union_is_faction_l233_233970

variable {D : Type} (is_faction : Set D → Prop)
variable (A B : Set D)

-- Define the complement
def complement (S : Set D) : Set D := {x | x ∉ S}

-- State the given condition
axiom faction_complement_union (A B : Set D) : 
  is_faction A → is_faction B → is_faction (complement (A ∪ B))

-- The theorem to prove
theorem union_is_faction (A B : Set D) :
  is_faction A → is_faction B → is_faction (A ∪ B) := 
by
  -- Proof goes here
  sorry

end union_is_faction_l233_233970


namespace parachute_drop_speed_l233_233440

-- Given F = kSv^2
variables {F S v : ℝ}

theorem parachute_drop_speed (F S v : ℝ) (h1 : F = 2) (h2 : S = 0.5) (h3 : v = 10) (h4 : ∀ F S v, F = 0.04 * S * (v ^ 2)) :
  ∃ v, F = 32 ∧ S = 2 ∧ v = 20 := 
by
  have k := 0.04 -- from condition that F = 2, S = 0.5, v = 10, thus k = 2 / (0.5 * 100) = 0.04
  use 20
  split
  { -- prove F = 32 for S = 2 and v = 20
    rw [h4 32 2 20],
    norm_num },
  split
  { -- condition for S = 2
    norm_num },
  { -- condition for v = 20
    norm_num }

end parachute_drop_speed_l233_233440


namespace solve_inequality_l233_233135

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_nonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def f_at_one_third (f : ℝ → ℝ) : Prop :=
  f (1/3) = 0

theorem solve_inequality (f : ℝ → ℝ) (x : ℝ) :
  even_function f →
  increasing_on_nonnegatives f →
  f_at_one_third f →
  (0 < x ∧ x < 1/2) ∨ (x > 2) ↔ f (Real.logb (1/8) x) > 0 :=
by
  -- the proof will be filled in here
  sorry

end solve_inequality_l233_233135


namespace value_of_a_l233_233658

theorem value_of_a (a : ℝ) (h : a = -a) : a = 0 :=
by
  sorry

end value_of_a_l233_233658


namespace johns_sister_age_l233_233923

variable (j d s : ℝ)

theorem johns_sister_age 
  (h1 : j = d - 15)
  (h2 : j + d = 100)
  (h3 : s = j - 5) :
  s = 37.5 := 
sorry

end johns_sister_age_l233_233923


namespace number_of_nonnegative_solutions_l233_233239

-- Define the equation x^2 = -6x
def equation (x : ℝ) : Prop := x^2 = -6 * x

-- Define the condition for a nonnegative solution
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Define the main theorem to prove the number of nonnegative solutions
theorem number_of_nonnegative_solutions : 
  {x : ℝ // equation x ∧ nonnegative x}.to_finset.card = 1 :=
sorry

end number_of_nonnegative_solutions_l233_233239


namespace nonneg_solutions_count_l233_233257

theorem nonneg_solutions_count :
  {x : ℝ | x^2 + 6 * x = 0 ∧ 0 ≤ x}.card = 1 :=
by
  sorry

end nonneg_solutions_count_l233_233257


namespace roots_of_unity_in_polynomial_l233_233137

noncomputable def polynomial_has_three_root_of_unity (a b c : ℤ) : Prop :=
  ∃ z₁ z₂ z₃ : ℂ, (z₁ ^ 3 = 1 ∧ (z₁ ^ 2 = z₁.conj ∨ z₁ = 1) ∧
                  z₂ ^ 3 = 1 ∧ (z₂ ^ 2 = z₂.conj ∨ z₂ = 1) ∧
                  z₃ ^ 3 = 1 ∧ (z₃ ^ 2 = z₃.conj ∨ z₃ = 1)) ∧
                 (z₁ ≠ z₂) ∧ (z₂ ≠ z₃) ∧ (z₁ ≠ z₃) ∧
                 (z₁ + z₂ + z₃ = -a) ∧
                 (z₁ * z₂ + z₂ * z₃ + z₃ * z₁ = b) ∧
                 (z₁ * z₂ * z₃ = -c)

theorem roots_of_unity_in_polynomial (a b c : ℤ) (h : polynomial_has_three_root_of_unity a b c) :
    ∃ (root_set : set ℂ), root_set ⊆ {z : ℂ | z^3 = 1} ∧ root_set.card = 3 :=
sorry

end roots_of_unity_in_polynomial_l233_233137


namespace marked_up_percentage_is_15_l233_233389

def CP : ℝ := 540
def SP : ℝ := 456
def discount_percentage : ℝ := 26.570048309178745

def MP : ℝ := SP / (1 - (discount_percentage / 100))

def marked_up_percentage : ℝ := ((MP - CP) / CP) * 100

theorem marked_up_percentage_is_15 : marked_up_percentage = 15 := by
  sorry

end marked_up_percentage_is_15_l233_233389


namespace geometric_sequence_property_l233_233340

variable {G : Type} [Group G]
variable {a : ℕ → G}

theorem geometric_sequence_property 
(h₁ : a 4 = 4) 
(h₂ : ∀ n : ℕ, a n * a (n + 2) = (a (n + 1)) ^ 2) : 
a 3 * a 5 = 16 :=
by 
  -- Placeholder for proof
  sorry

end geometric_sequence_property_l233_233340


namespace sphere_radius_eq_six_l233_233002

-- Defining the sphere's radius and its surface area formula
def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- Defining the cylinder's radius, height and its curved surface area formula
def cylinder_surface_area (r h : ℝ) : ℝ := 2 * Real.pi * r * h

-- Problem Conditions:
axiom sphere_eq_cylinder : 
  ∀ (r : ℝ), sphere_surface_area r = cylinder_surface_area 6 12

-- Stating the theorem to prove the radius of the sphere is 6 cm
theorem sphere_radius_eq_six :
  ∃ (r : ℝ), sphere_eq_cylinder r ∧ r = 6 := sorry

end sphere_radius_eq_six_l233_233002


namespace fraction_unclaimed_l233_233744

def exists_fraction_unclaimed (x : ℕ) : Prop :=
  let claimed_by_Eva := (1 / 2 : ℚ) * x
  let remaining_after_Eva := x - claimed_by_Eva
  let claimed_by_Liam := (3 / 8 : ℚ) * x
  let remaining_after_Liam := remaining_after_Eva - claimed_by_Liam
  let claimed_by_Noah := (1 / 8 : ℚ) * remaining_after_Eva
  let remaining_after_Noah := remaining_after_Liam - claimed_by_Noah
  remaining_after_Noah / x = (75 / 128 : ℚ)

theorem fraction_unclaimed {x : ℕ} : exists_fraction_unclaimed x :=
by
  sorry

end fraction_unclaimed_l233_233744


namespace find_sequence_values_l233_233916

-- Definitions of the sequence conditions
def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 17 = 66 ∧ ∃ k b, (∀ n, a n = k * n + b)

-- The goal: proving the general formula and specific value
theorem find_sequence_values : sequence a → (∀ n, a n = 4 * n - 2) ∧ (a 2015 = 8058) :=
by
  sorry

end find_sequence_values_l233_233916


namespace nonneg_solutions_count_l233_233254

theorem nonneg_solutions_count :
  {x : ℝ | x^2 + 6 * x = 0 ∧ 0 ≤ x}.card = 1 :=
by
  sorry

end nonneg_solutions_count_l233_233254


namespace percentage_greater_than_88_l233_233335

theorem percentage_greater_than_88 (x : ℝ) (percentage : ℝ) (h1 : x = 110) (h2 : x = 88 + (percentage * 88)) : percentage = 0.25 :=
by
  sorry

end percentage_greater_than_88_l233_233335


namespace magnitude_of_z_l233_233376

theorem magnitude_of_z (z : ℂ) (i : ℂ) (h_i : i = complex.I) (h : 1 + z * i = 2 * i) : abs z = real.sqrt 5 :=
by
  sorry

end magnitude_of_z_l233_233376


namespace range_of_h_l233_233904

noncomputable def g (x : ℝ) : ℝ := (4 * x) / (x - 1)
noncomputable def g_inv (y : ℝ) : ℝ := y / (y - 4)
noncomputable def h (x : ℝ) : ℝ := x + g_inv(x)

theorem range_of_h : (∀ x > 1, (h(x) : ℝ)) → (range h = Icc 9 (⊤ : ℝ)) :=
by
  sorry

end range_of_h_l233_233904


namespace part1_l233_233825

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.exp x + Real.log (x + 1) - a * Real.sin x

theorem part1 (a : ℝ) (h : a > 2) : ∃ (c : ℝ), 0 < c ∧ c < Real.pi / 2 ∧ deriv (λ x, f x a) c = 0 :=
by
  sorry

end part1_l233_233825


namespace determine_m_for_unique_solution_l233_233428

-- Define the quadratic equation and the condition for a unique solution
def quadratic_eq_has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

-- Define the specific quadratic equation and its discriminant
def specific_quadratic_eq (m : ℝ) : Prop :=
  quadratic_eq_has_one_solution 3 (-7) m

-- State the main theorem to prove the value of m
theorem determine_m_for_unique_solution :
  specific_quadratic_eq (49 / 12) :=
by
  unfold specific_quadratic_eq quadratic_eq_has_one_solution
  sorry

end determine_m_for_unique_solution_l233_233428


namespace probability_no_practice_l233_233449

def prob_has_practice : ℚ := 5 / 8

theorem probability_no_practice : 
  1 - prob_has_practice = 3 / 8 := 
by
  sorry

end probability_no_practice_l233_233449


namespace milk_for_new_batch_l233_233967

theorem milk_for_new_batch (flour_new : ℕ) (flour_old : ℕ) (milk_old : ℕ) (h1 : flour_new = 1200) (h2 : flour_old = 400) (h3 : milk_old = 80) :
  let ratio := milk_old / flour_old in
  let portions := flour_new / flour_old in
  let total_milk := portions * milk_old in
  total_milk = 240 :=
by
  rw [h1, h2, h3]
  let ratio := milk_old / flour_old
  let portions := flour_new / flour_old
  let total_milk := portions * milk_old
  simp [ratio, portions, total_milk]
  sorry

end milk_for_new_batch_l233_233967


namespace B_cubed_v_l233_233371

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![5, 0], [0, 5]]

def v : Fin 2 → ℝ := ![10, -4]

theorem B_cubed_v : (B^3).mulVec v = ![1250, -500] := by
  sorry

end B_cubed_v_l233_233371


namespace cone_vertex_angle_spheres_touching_l233_233007

theorem cone_vertex_angle_spheres_touching (r1 r2 r3 : ℝ) (radius : ℝ → ℝ) 
    (vertex : ℝ × ℝ × ℝ) (O1 O2 O3 : ℝ × ℝ × ℝ) (C : ℝ × ℝ × ℝ) :
    (radius O1 = 2) ∧ (radius O2 = 2) ∧ (radius O3 = 1) ∧ 
    (dist C O1 = dist C O2) ∧ 
    (C.3 = 0) ∧  -- C is on the table, assuming Z coordinate represents height
    (dist O1 O2 = 4) ∧ (dist O1 O3 = 3) ∧ (dist O2 O3 = 3) ∧
    (radius (vertex) = 2) ∧
    (∃ k, dist C (vertex) = k ∧ dist (vertex) O1 = k ∧ dist (vertex) O2 = k) → 
    let α := (2:ℝ) * arctan ((sqrt 5 - 2) / 3) in 
    cos(α) = (cos α - ((2:ℝ) / sqrt 5) * sin(α))
    sorry -- no proof is required

end cone_vertex_angle_spheres_touching_l233_233007


namespace problem_statement_l233_233941

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * sin (2 * x) + b * cos (2 * x)

theorem problem_statement (a b : ℝ) (h : a * b ≠ 0) (x : ℝ) 
    (hf : ∀ x, f a b x ≤ abs (f a b (π / 6))) :
  f a b (-π / 6 - x) + f a b x = 0 :=
sorry

end problem_statement_l233_233941


namespace car_X_probability_l233_233895

theorem car_X_probability :
  ∃ P_X : ℚ, P_X = 1 / 2 ∧ 
  let P_Y := 1 / 4
  let P_Z := 1 / 3
  in P_X + P_Y + P_Z = 13 / 12 :=
begin
  sorry
end

end car_X_probability_l233_233895


namespace simplify_expression_l233_233651

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ( ((a ^ (4 / 3 / 5)) ^ (3 / 2)) / ((a ^ (4 / 1 / 5)) ^ 3) ) /
  ( ((a * (a ^ (2 / 3) * b ^ (1 / 3))) ^ (1 / 2)) ^ 4) * 
  (a ^ (1 / 4) * b ^ (1 / 8)) ^ 6 = 1 / ((a ^ (2 / 12)) * (b ^ (1 / 12))) :=
by
  sorry

end simplify_expression_l233_233651


namespace nonnegative_solutions_eq_1_l233_233304

variable (x : ℝ)

theorem nonnegative_solutions_eq_1 : (x^2 = -6 * x) → (∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀) → (x ∈ {0}) :=
by
  assume (h1: x^2 = -6 * x)
  assume (h2: ∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀)
  sorry

end nonnegative_solutions_eq_1_l233_233304


namespace function_unique_l233_233150

theorem function_unique {f : ℚ → ℚ} (h1 : ∀ (x : ℚ), 0 < x → ∀ (a : ℚ), 0 < a → a < 1 → f (a * x) ≤ (f x) ^ a)
    (h2 : ∀ (x y : ℚ), 0 < x → 0 < y → f (x + y) ≤ f x * f y) : 
    ∀ (x : ℚ), 0 < x → f x = 1 := 
begin 
    sorry
end

end function_unique_l233_233150


namespace acceptable_mass_l233_233673

theorem acceptable_mass : 
  ∀ (m : ℝ), (m = 34.80) → (34.75 ≤ m ∧ m ≤ 35.25) :=
by
  intros m hm
  rw hm
  exact ⟨le_of_lt (by norm_num), le_of_lt (by norm_num)⟩

end acceptable_mass_l233_233673


namespace greatest_prime_factor_341_l233_233581

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l233_233581


namespace magnitude_of_3a_minus_b_find_lambda_l233_233227

-- Condition Definitions
def vector_a := (1, 2)
def vector_b := (-3, 4)

-- Proof statement for part (1)
theorem magnitude_of_3a_minus_b :
  let a := vector_a in
  let b := vector_b in
  ∥(3 * a.1 - b.1, 3 * a.2 - b.2)∥ = 2 * Real.sqrt 10 :=
by
  sorry

-- Definition of perpendicularity for part (2)
def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

-- Proof statement for part (2)
theorem find_lambda :
  let a := vector_a in
  let b := vector_b in
  let lambda : ℝ := -1 in
  perpendicular a (a.1 - 3 * lambda + 4 * lambda, a.2 + 2 * lambda + 4 * lambda) :=
by
  sorry

end magnitude_of_3a_minus_b_find_lambda_l233_233227


namespace clark_discount_l233_233098

noncomputable def price_per_part : ℕ := 80
noncomputable def num_parts : ℕ := 7
noncomputable def total_paid : ℕ := 439

theorem clark_discount : (price_per_part * num_parts - total_paid) = 121 :=
by
  -- proof goes here
  sorry

end clark_discount_l233_233098


namespace nonnegative_solutions_eq_one_l233_233294

theorem nonnegative_solutions_eq_one : 
  (finset.univ.filter (λ x : ℝ, x^2 = -6 * x ∧ 0 <= x)).card = 1 := 
by 
  sorry

end nonnegative_solutions_eq_one_l233_233294


namespace problem_1_exists_problem_2_exists_l233_233205

-- Given definition: Center of circle lies on line x - 2y = 0
def lies_on_line (x y : ℝ) : Prop := x - 2 * y = 0

-- Given definition: Chord intercepted on x-axis has a length of 2sqrt(3)
def chord_length (r : ℝ) : Prop := 2 * sqrt 3 = 2 * sqrt (r^2 - (sqrt 3)^2)

-- Given definition for circle equation
def circle_eq (x y cx cy r : ℝ) : Prop := (x - cx)^2 + (y - cy)^2 = r^2

-- Problem 1: Prove standard equation of circle
theorem problem_1_exists (a : ℝ) : 
  (a > 0 ∧ lies_on_line (2 * a) a ∧ chord_length (2 * a)) → 
  ∃ (x y : ℝ), circle_eq x y 2 1 2 := sorry

-- Given definition for line equation
def line_eq (x y b : ℝ) : Prop := y = -2 * x + b

-- Given definition: The discriminant condition for intersection
def disc_condition (b : ℝ) : Prop := b^2 - 10 * b + 5 < 0

-- Problem 2: Prove value of b
theorem problem_2_exists (b : ℝ) : 
  (∃ a, a > 0 ∧ lies_on_line (2 * a) a ∧ chord_length (2 * a) ∧ ∀ x y, line_eq x y b → circle_eq x y 2 1 2) →
  disc_condition b → 
  b = (5 + sqrt 15) / 2 ∨ b = (5 - sqrt 15) / 2 := sorry

end problem_1_exists_problem_2_exists_l233_233205


namespace greatest_prime_factor_of_341_l233_233538

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l233_233538


namespace greatest_prime_factor_of_341_is_17_l233_233561

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l233_233561


namespace rationalize_result_l233_233414

noncomputable def rationalize_denominator (x y : ℚ) (sqrt_c : ℚ) : ℚ :=
  let numerator := x + sqrt_c
  let denominator := y - sqrt_c
  (numerator * (y + sqrt_c)) / (denominator * (y + sqrt_c))

theorem rationalize_result :
  let sqrt_5 := Real.sqrt 5
  let expr := rationalize_denominator 2 3 sqrt_5
  let A := 11 / 4
  let B := 5 / 4
  let C := 5
  expr = A + B * sqrt_5 ∧ A * B * C = 275 / 16 := 
sorry

end rationalize_result_l233_233414


namespace percentage_of_integers_divisible_by_6_up_to_120_l233_233620

theorem percentage_of_integers_divisible_by_6_up_to_120 : 
  let total := 120
      divisible_by_6 := λ n, n % 6 = 0
      count := (list.range (total + 1)).countp divisible_by_6
      percentage := (count.toFloat / total.toFloat) * 100
  in percentage = 16.67 :=
by
  sorry

end percentage_of_integers_divisible_by_6_up_to_120_l233_233620


namespace clark_discount_l233_233101

theorem clark_discount (price_per_part : ℕ) (number_of_parts : ℕ) (amount_paid : ℕ)
  (h1 : price_per_part = 80)
  (h2 : number_of_parts = 7)
  (h3 : amount_paid = 439) : 
  (number_of_parts * price_per_part) - amount_paid = 121 := by
  sorry

end clark_discount_l233_233101


namespace parameter_for_three_distinct_solutions_l233_233773

open Polynomial

theorem parameter_for_three_distinct_solutions (a : ℝ) :
  (∀ x : ℝ, x^4 - 40 * x^2 + 144 = a * (x^2 + 4 * x - 12)) →
  (∀ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 → 
  (x1^4 - 40 * x1^2 + 144 = a * (x1^2 + 4 * x1 - 12) ∧ 
   x2^4 - 40 * x2^2 + 144 = a * (x2^2 + 4 * x2 - 12) ∧ 
   x3^4 - 40 * x3^2 + 144 = a * (x3^2 + 4 * x3 - 12) ∧
   x4^4 - 40 * x4^2 + 144 = a * (x4^2 + 4 * x4 - 12))) → a = 48 :=
by
  sorry

end parameter_for_three_distinct_solutions_l233_233773


namespace isosceles_trapezoid_area_l233_233102

variable (h a b c d : ℕ)
noncomputable def trapezoid_area (h a b c d : ℕ) : ℕ :=
  (1/2) * (a + b) * h

theorem isosceles_trapezoid_area :
  ∀ (leg diagonal longer_base shorter_base : ℕ),
  is_isosceles_trapezoid leg diagonal longer_base shorter_base →
  area_is_correct leg diagonal longer_base shorter_base :=
by {
  intros leg diagonal longer_base shorter_base iso_trap
  let height := 24
  let a := 50
  let b := 14
  let area := trapezoid_area height a b
  exact area = 768
  sorry
}

end isosceles_trapezoid_area_l233_233102


namespace five_prime_sum_of_divisors_in_1_to_50_l233_233374

def sum_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ i => n % i = 0).sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (List.range (n - 2)).all (λ k => n % (k + 2) ≠ 0)

def count_prime_f_sum_of_divisors_in_range (n : ℕ) : ℕ :=
  (List.range' 1 (n + 1)).countp (λ i => is_prime (sum_of_divisors i))

theorem five_prime_sum_of_divisors_in_1_to_50 :
  count_prime_f_sum_of_divisors_in_range 50 = 5 :=
sorry

end five_prime_sum_of_divisors_in_1_to_50_l233_233374


namespace train_speed_l233_233089

def distance := 300 -- meters
def time := 18 -- seconds

noncomputable def speed_kmh := 
  let speed_ms := distance / time -- speed in meters per second
  speed_ms * 3.6 -- convert to kilometers per hour

theorem train_speed : speed_kmh = 60 := 
  by
    -- The proof steps are omitted
    sorry

end train_speed_l233_233089


namespace number_of_integers_congruent_to_3_mod_7_less_than_500_l233_233864

theorem number_of_integers_congruent_to_3_mod_7_less_than_500 : 
  {n : ℕ | n < 500 ∧ n % 7 = 3}.card = 71 :=
sorry

end number_of_integers_congruent_to_3_mod_7_less_than_500_l233_233864


namespace yearly_exports_calculation_l233_233663

variable (Y : Type) 
variable (fruit_exports_total yearly_exports : ℝ)
variable (orange_exports : ℝ := 4.25 * 10^6)
variable (fruit_exports_percent : ℝ := 0.20)
variable (orange_exports_fraction : ℝ := 1/6)

-- The main statement to prove
theorem yearly_exports_calculation
  (h1 : yearly_exports * fruit_exports_percent = fruit_exports_total)
  (h2 : fruit_exports_total * orange_exports_fraction = orange_exports) :
  yearly_exports = 127.5 * 10^6 :=
by
  -- Proof (omitted)
  sorry

end yearly_exports_calculation_l233_233663


namespace problem_statement_l233_233036

noncomputable def inequality (n : ℕ) (hn : n ≥ 1) (x : ℕ → ℝ)
  (hx_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < x i)
  (hx_order : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → x i ≤ x j) 
  (hx_ratio : ∀ i, 1 ≤ i ∧ i ≤ n → x 1 ≥ (x i) / i) : Prop :=
  (∑ i in Finset.range n, x (i + 1)) / (n * (∏ i in Finset.range n, x (i + 1) ^ (1 / (n : ℝ)))) ≤
  (n + 1) / (2 * (nat.factorial n) ^ (1 / (n : ℝ)))

theorem problem_statement (n : ℕ) (hn : n ≥ 1) (x : ℕ → ℝ) 
  (hx_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < x i)
  (hx_order : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → x i ≤ x j) 
  (hx_ratio : ∀ i, 1 ≤ i ∧ i ≤ n → x 1 ≥ (x i) / i) :
  inequality n hn x hx_pos hx_order hx_ratio := 
  sorry

end problem_statement_l233_233036


namespace angle_DEF_is_72_l233_233042

-- Definitions
variables (A B C D E F : Type*)
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space D] [metric_space E] [metric_space F]
variables (ABCD : rectangle A B C D)
variables (CEF : equilateral_triangle C E F)
variable (angle_ABD : ℝ)
hypothesis (h_ABD : angle_ABD = 6)

-- To Prove
theorem angle_DEF_is_72 (h : angle_ABD = 6) : ∠ DEF = 72 :=
  sorry

end angle_DEF_is_72_l233_233042


namespace find_n_plus_c_l233_233494

variables (n c : ℝ)

-- Conditions from the problem
def line1 := ∀ (x y : ℝ), (x = 4) → (y = 11) → (y = n * x + 3)
def line2 := ∀ (x y : ℝ), (x = 4) → (y = 11) → (y = 5 * x + c)

theorem find_n_plus_c (h1 : line1 n)
                      (h2 : line2 c) :
  n + c = -7 := by
  sorry

end find_n_plus_c_l233_233494


namespace greatest_prime_factor_of_341_l233_233566

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l233_233566


namespace nonnegative_solution_count_l233_233271

theorem nonnegative_solution_count : 
  ∃! x : ℝ, x^2 = -6 * x ∧ 0 ≤ x := 
begin
  sorry
end

end nonnegative_solution_count_l233_233271


namespace nonnegative_solutions_eq1_l233_233246

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l233_233246


namespace price_of_33_kgs_l233_233699

variable (l q : ℕ)

def apples_cost (kgs: ℕ) (l q: ℕ): ℕ :=
  if kgs <= 30 then kgs * l
  else 30 * l + (kgs - 30) * q

theorem price_of_33_kgs (h1 : apples_cost 25 l q = 250)
                        (h2 : apples_cost 36 l q = 420) : 
  apples_cost 33 l q = 360 :=
begin
  sorry
end

end price_of_33_kgs_l233_233699


namespace find_a_value_l233_233845

noncomputable def lines_parallel_but_not_coincident (a : ℝ) : Prop :=
  let l1 := λ x y : ℝ, a * x - y + a = 0
  let l2 := λ x y : ℝ, (2 * a - 3) * x + a * y - a = 0
  let are_parallel := a / (-1) = (2 * a - 3) / a
  let not_coincident := ¬(∀ x y : ℝ, l1 x y ↔ l2 x y)
  are_parallel ∧ not_coincident

theorem find_a_value : ∃ a : ℝ, a = -3 ∧ lines_parallel_but_not_coincident a :=
by
  use -3
  sorry

end find_a_value_l233_233845


namespace eva_total_marks_correct_l233_233736

-- Definitions based on conditions
def math_marks_second_sem : ℕ := 80
def arts_marks_second_sem : ℕ := 90
def science_marks_second_sem : ℕ := 90

def math_marks_first_sem : ℕ := math_marks_second_sem + 10
def arts_marks_first_sem : ℕ := arts_marks_second_sem - 15
def science_marks_first_sem : ℕ := science_marks_second_sem - (science_marks_second_sem / 3)

def total_marks_first_sem : ℕ := math_marks_first_sem + arts_marks_first_sem + science_marks_first_sem
def total_marks_second_sem : ℕ := math_marks_second_sem + arts_marks_second_sem + science_marks_second_sem

def total_marks_both_sems : ℕ := total_marks_first_sem + total_marks_second_sem

-- Theorem to be proved
theorem eva_total_marks_correct : total_marks_both_sems = 485 := by
  -- Here, we state that we need to prove the total marks sum up to 485
  sorry

end eva_total_marks_correct_l233_233736


namespace nonneg_solutions_count_l233_233252

theorem nonneg_solutions_count :
  {x : ℝ | x^2 + 6 * x = 0 ∧ 0 ≤ x}.card = 1 :=
by
  sorry

end nonneg_solutions_count_l233_233252


namespace greatest_prime_factor_of_341_is_17_l233_233557

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l233_233557


namespace nonnegative_solutions_count_l233_233309

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.card = 1 := 
sorry

end nonnegative_solutions_count_l233_233309


namespace function_analysis_l233_233677

theorem function_analysis
  (ω > 0)
  (|φ| < π / 2)
  (f : ℝ → ℝ)
  (h1 : f = λ x, sin (ω * x + φ))
  (h2 : f (π / 4) = 1)
  (h3 : f (7 * π / 12) = -1) :
  (f = λ x, sin (3 * x - π / 4)) ∧
  (∑ x in {x | 0 ≤ x ∧ x ≤ 2 * π ∧ sin(3 * x - π / 4) = a}, x = 11 * π / 2) :=
sorry

end function_analysis_l233_233677


namespace nonnegative_solutions_eq1_l233_233243

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l233_233243


namespace harvey_initial_steaks_proof_l233_233231

def harveys_initial_steaks : Nat :=
  let steaks_left := 12
  let steaks_sold_last := 4
  let total_steaks_sold := 17
  let steaks_before_last_sale := steaks_left + steaks_sold_last
  steaks_before_last_sale + total_steaks_sold

theorem harvey_initial_steaks_proof : harveys_initial_steaks = 33 :=
by
  let steaks_left := 12
  let steaks_sold_last := 4
  let total_steaks_sold := 17
  let steaks_before_last_sale := steaks_left + steaks_sold_last
  have h1 : steaks_before_last_sale = 16 := by 
    simp [steaks_before_last_sale, steaks_left, steaks_sold_last]
  have h2 : harveys_initial_steaks = steaks_before_last_sale + total_steaks_sold := by 
    simp [harveys_initial_steaks, steaks_before_last_sale, total_steaks_sold]
  have h3 : h2 = 16 + 17 := by 
    rw [h1]
  have h4 : 16 + 17 = 33 := by 
    simp
  rw [h3, h4]
  exact sorry

end harvey_initial_steaks_proof_l233_233231


namespace total_shoes_count_l233_233678

-- Define the concepts and variables related to the conditions
def num_people := 10
def num_people_regular_shoes := 4
def num_people_sandals := 3
def num_people_slippers := 3
def num_shoes_regular := 2
def num_shoes_sandals := 1
def num_shoes_slippers := 1

-- Goal: Prove that the total number of shoes kept outside is 20
theorem total_shoes_count :
  (num_people_regular_shoes * num_shoes_regular) +
  (num_people_sandals * num_shoes_sandals * 2) +
  (num_people_slippers * num_shoes_slippers * 2) = 20 :=
by
  sorry

end total_shoes_count_l233_233678


namespace area_of_triangle_l233_233803

-- Define the hyperbola with given parameters
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define conditions for foci
def is_foci (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = (-5, 0) ∧ F2 = (5, 0)

-- Define the point P on the hyperbola and the angle condition
def on_hyperbola_and_right_angle (P F1 F2 : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ F1 = (-5, 0) ∧ F2 = (5, 0) ∧ ∀ A B : ℝ × ℝ, (A = F1 ∧ B = F2) → ∡ A P B = 90

-- State the theorem
theorem area_of_triangle (P F1 F2 : ℝ × ℝ) (h1 : on_hyperbola_and_right_angle P F1 F2) : 
  ∃ S : ℝ, S = 16 :=
begin
  sorry
end

end area_of_triangle_l233_233803


namespace total_savings_calculation_l233_233444

theorem total_savings_calculation
  (income : ℕ)
  (ratio_income_to_expenditure : ℕ)
  (ratio_expenditure_to_income : ℕ)
  (tax_rate : ℚ)
  (investment_rate : ℚ)
  (expenditure : ℕ)
  (taxes : ℚ)
  (investments : ℚ)
  (total_savings : ℚ)
  (h_income : income = 17000)
  (h_ratio : ratio_income_to_expenditure / ratio_expenditure_to_income = 5 / 4)
  (h_tax_rate : tax_rate = 0.15)
  (h_investment_rate : investment_rate = 0.1)
  (h_expenditure : expenditure = (income / 5) * 4)
  (h_taxes : taxes = 0.15 * income)
  (h_investments : investments = 0.1 * income)
  (h_total_savings : total_savings = income - (expenditure + taxes + investments)) :
  total_savings = 900 :=
by
  sorry

end total_savings_calculation_l233_233444


namespace greatest_price_book_l233_233112

theorem greatest_price_book (p : ℕ) (B : ℕ) (D : ℕ) (F : ℕ) (T : ℚ) 
  (h1 : B = 20) 
  (h2 : D = 200) 
  (h3 : F = 5)
  (h4 : T = 0.07) 
  (h5 : ∀ p, 20 * p * (1 + T) ≤ (D - F)) : 
  p ≤ 9 :=
by
  sorry

end greatest_price_book_l233_233112


namespace a_ge_b_l233_233943

noncomputable def a (x : ℝ) (m : ℕ) : ℝ := (Real.log x)^m + (Real.log x)^(-m)
noncomputable def b (x : ℝ) (n : ℕ) : ℝ := (Real.log x)^n + (Real.log x)^(-n)

theorem a_ge_b (x : ℝ) (m n : ℕ) (h1 : m > n) (h2 : 1 < x) (h3 : 0 < n) : a x m ≥ b x n := sorry

end a_ge_b_l233_233943


namespace radius_of_sphere_l233_233000

-- Definitions for surface areas
def surfaceAreaSphere (r : ℝ) : ℝ := 4 * Real.pi * r^2
def curvedSurfaceAreaCylinder (r_cylinder : ℝ) (h : ℝ) : ℝ := 2 * Real.pi * r_cylinder * h

-- Given conditions
def h : ℝ := 12
def d : ℝ := 12
def r_cylinder : ℝ := d / 2

-- Problem statement
theorem radius_of_sphere :
  (surfaceAreaSphere r = curvedSurfaceAreaCylinder r_cylinder h) → r = 6 :=
by
  sorry

end radius_of_sphere_l233_233000


namespace person_dining_minutes_l233_233109

theorem person_dining_minutes
  (initial_angle : ℕ)
  (final_angle : ℕ)
  (time_spent : ℕ)
  (minute_angle_per_minute : ℕ)
  (hour_angle_per_minute : ℕ)
  (h1 : initial_angle = 110)
  (h2 : final_angle = 110)
  (h3 : minute_angle_per_minute = 6)
  (h4 : hour_angle_per_minute = minute_angle_per_minute / 12)
  (h5 : time_spent = (final_angle - initial_angle) / (minute_angle_per_minute / (minute_angle_per_minute / 12) - hour_angle_per_minute)) :
  time_spent = 40 := sorry

end person_dining_minutes_l233_233109


namespace amount_spent_on_milk_is_1500_l233_233692

def total_salary (saved : ℕ) (saving_percent : ℕ) : ℕ := 
  saved / (saving_percent / 100)

def total_spent_excluding_milk (rent groceries education petrol misc : ℕ) : ℕ := 
  rent + groceries + education + petrol + misc

def amount_spent_on_milk (total_salary total_spent savings : ℕ) : ℕ := 
  total_salary - total_spent - savings

theorem amount_spent_on_milk_is_1500 :
  let rent := 5000
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let misc := 2500
  let savings := 2000
  let saving_percent := 10
  let salary := total_salary savings saving_percent
  let spent_excluding_milk := total_spent_excluding_milk rent groceries education petrol misc
  amount_spent_on_milk salary spent_excluding_milk savings = 1500 :=
by {
  sorry
}

end amount_spent_on_milk_is_1500_l233_233692


namespace isosceles_cosine_vertex_angle_right_triangle_area_l233_233808

theorem isosceles_cosine_vertex_angle
  (a b c : ℝ)
  (A B C : ℝ)
  (h1: a = 2 * c)
  (h2: b = 2 * c)
  (h3 : sin(B)^2 = 2 * sin(A) * sin(C)) :
  cos(C) = (7 / 8) :=
by
  sorry

theorem right_triangle_area
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : B = (π / 2))
  (h2 : b = sqrt(2))
  (h3 : sin(B)^2 = 2 * sin(A) * sin(C))
  (h4 : a = c) :
  (sqrt(2) * sqrt(2) / 2) = 1 :=
by
  sorry

end isosceles_cosine_vertex_angle_right_triangle_area_l233_233808


namespace sum_binomials_eq_l233_233765

theorem sum_binomials_eq : 
  (Nat.choose 6 1) + (Nat.choose 6 2) + (Nat.choose 6 3) + (Nat.choose 6 4) + (Nat.choose 6 5) = 62 :=
by
  sorry

end sum_binomials_eq_l233_233765


namespace nonnegative_solutions_count_l233_233306

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.card = 1 := 
sorry

end nonnegative_solutions_count_l233_233306


namespace count_prime_solutions_l233_233203

def is_prime (n : ℕ) : Prop := n = 2 ∨ (2 < n ∧ ¬ ∃ m, 2 ≤ m ∧ m < n ∧ m ∣ n)

theorem count_prime_solutions:
  {x y z : ℕ} 
  (hx : is_prime x)
  (hy : is_prime y)
  (hz : is_prime z)
  (hxy : x ≤ y)
  (hyz : y ≤ z)
  (hxyz_eq : x^2 + y^2 + z^2 = 2019) :
  {x y z | is_prime x ∧ is_prime y ∧ is_prime z ∧ x ≤ y ∧ y ≤ z ∧ x^2 + y^2 + z^2 = 2019}.card = 5 := 
sorry

end count_prime_solutions_l233_233203


namespace centroid_circumcircle_property_l233_233932

theorem centroid_circumcircle_property
  (A B C : ℝ^3)
  (G : ℝ^3 := (A + B + C) / 3)
  (R : ℝ)
  (P : ℝ^3)
  (hP : ∥P∥ = R)
  (hCircumcircle : ∥P - A∥ = R ∧ ∥P - B∥ = R ∧ ∥P - C∥ = R) :
  ∥P - A∥^2 + ∥P - B∥^2 + ∥P - C∥^2 - ∥P - G∥^2 = (14 / 3) * R^2 := 
by
  sorry

end centroid_circumcircle_property_l233_233932


namespace product_of_terms_form_l233_233017

theorem product_of_terms_form 
  (a b c d : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) :
  ∃ p q : ℝ, 
    (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5) = p + q * Real.sqrt 5 
    ∧ 0 ≤ p 
    ∧ 0 ≤ q := 
by
  let p := a * c + 5 * b * d
  let q := a * d + b * c
  use p, q
  sorry

end product_of_terms_form_l233_233017


namespace percent_divisible_by_6_l233_233625

theorem percent_divisible_by_6 (N : ℕ) (hN : N = 120) :
  (∃ M, M = (finset.univ.filter (λ n : ℕ, n ≤ N ∧ n % 6 = 0)).card ∧ M * 6 = N) →
  (M.to_real / N.to_real) * 100 = 16.66666667 :=
by
  intros h
  sorry

end percent_divisible_by_6_l233_233625


namespace a_50_eq_4_pow_49_l233_233134

noncomputable def a : ℕ → ℝ
| 0 := 1
| (n + 1) := (64 * (a n)^3)^(1 / 3)

theorem a_50_eq_4_pow_49 : a 49 = 4^49 := sorry

end a_50_eq_4_pow_49_l233_233134


namespace tangent_circles_pass_through_homothety_center_l233_233713

-- Define the necessary structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def is_tangent_to_line (ω : Circle) (L : ℝ → ℝ) : Prop :=
  sorry -- Definition of tangency to a line

def is_tangent_to_circle (ω : Circle) (C : Circle) : Prop :=
  sorry -- Definition of tangency to another circle

theorem tangent_circles_pass_through_homothety_center
  (L : ℝ → ℝ) (C : Circle) (ω : Circle)
  (H_ext H_int : ℝ × ℝ)
  (H_tangency_line : is_tangent_to_line ω L)
  (H_tangency_circle : is_tangent_to_circle ω C) :
  ∃ P Q : ℝ × ℝ, 
    (is_tangent_to_line ω L ∧ is_tangent_to_circle ω C) →
    (P = Q ∧ (P = H_ext ∨ P = H_int)) :=
by
  sorry

end tangent_circles_pass_through_homothety_center_l233_233713


namespace smallest_positive_period_and_increasing_intervals_of_f_l233_233827

noncomputable def f (x : ℝ) : ℝ := 2 * cos x * (sin x + cos x) - 1

-- The smallest positive period of f(x)
def smallest_positive_period_f : ℝ := π

-- The intervals of monotonic increase for f(x) on [0, π]
def increasing_intervals_f : set (set ℝ) := {[set.Icc 0 (π / 8)], [set.Icc (5 * π / 8) π]}

theorem smallest_positive_period_and_increasing_intervals_of_f :
  ∃ T (intervals : set (set ℝ)), 
    (T = smallest_positive_period_f) ∧ 
    (intervals = increasing_intervals_f) :=
by
    use smallest_positive_period_f,
    use increasing_intervals_f,
    sorry

end smallest_positive_period_and_increasing_intervals_of_f_l233_233827


namespace greatest_prime_factor_of_341_l233_233605

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l233_233605


namespace h_even_function_l233_233217

def f (x : ℝ) : ℝ := Real.log (1 + 2 * x)
def g (x : ℝ) : ℝ := Real.log (1 - 2 * x)
def h (x : ℝ) : ℝ := f x + g x

theorem h_even_function (x : ℝ) (h₁ : -1 / 2 < x) (h₂ : x < 1 / 2) : h x = h (-x) := by
  sorry

end h_even_function_l233_233217


namespace sum_first_7_l233_233201

variable {α : Type*} [LinearOrderedField α]

-- Definitions for the arithmetic sequence
noncomputable def arithmetic_sequence (a d : α) (n : ℕ) : α :=
  a + d * (n - 1)

noncomputable def sum_of_first_n_terms (a d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

-- Conditions
variable {a d : α} -- Initial term and common difference of the arithmetic sequence
variable (h : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 + arithmetic_sequence a d 6 = 12)

-- Proof statement
theorem sum_first_7 (a d : α) (h : arithmetic_sequence a d 2 + arithmetic_sequence a d 4 + arithmetic_sequence a d 6 = 12) : 
  sum_of_first_n_terms a d 7 = 28 := 
by 
  sorry

end sum_first_7_l233_233201


namespace greatest_prime_factor_of_341_l233_233563

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l233_233563


namespace cube_root_of_64_l233_233997

theorem cube_root_of_64 : ∃ a : ℝ, a^3 = 64 ∧ a = 4 :=
by
  use 4
  split
  case left =>
    show 4^3 = 64
    calc
      4^3 = 64 : by norm_num
  case right =>
    show 4 = 4
    rfl

end cube_root_of_64_l233_233997


namespace greatest_prime_factor_341_l233_233523

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l233_233523


namespace evaTotalMarksCorrect_l233_233741

-- Definition of marks scored by Eva in each subject across semesters
def evaMathsMarksSecondSemester : Nat := 80
def evaArtsMarksSecondSemester : Nat := 90
def evaScienceMarksSecondSemester : Nat := 90

def evaMathsMarksFirstSemester : Nat := evaMathsMarksSecondSemester + 10
def evaArtsMarksFirstSemester : Nat := evaArtsMarksSecondSemester - 15
def evaScienceMarksFirstSemester : Nat := evaScienceMarksSecondSemester - (evaScienceMarksSecondSemester / 3)

-- Total marks in each semester
def totalMarksFirstSemester : Nat := evaMathsMarksFirstSemester + evaArtsMarksFirstSemester + evaScienceMarksFirstSemester
def totalMarksSecondSemester : Nat := evaMathsMarksSecondSemester + evaArtsMarksSecondSemester + evaScienceMarksSecondSemester

-- Combined total
def evaTotalMarks : Nat := totalMarksFirstSemester + totalMarksSecondSemester

-- Statement to prove
theorem evaTotalMarksCorrect : evaTotalMarks = 485 := 
by
  -- This needs to be proved as per the conditions and calculations above
  sorry

end evaTotalMarksCorrect_l233_233741


namespace problem_solution_l233_233868

theorem problem_solution (a b : ℝ) (h : (a - 2 * complex.I) * complex.I = b - complex.I) : a + b = 1 :=
sorry

end problem_solution_l233_233868


namespace ζ_sum_8_l233_233957

open Complex

def ζ1 : ℂ := sorry
def ζ2 : ℂ := sorry
def ζ3 : ℂ := sorry

def e1 := ζ1 + ζ2 + ζ3
def e2 := ζ1 * ζ2 + ζ2 * ζ3 + ζ3 * ζ1
def e3 := ζ1 * ζ2 * ζ3

axiom h1 : e1 = 2
axiom h2 : e1^2 - 2 * e2 = 8
axiom h3 : (e1^2 - 2 * e2)^2 - 2 * (e2^2 - 2 * e1 * e3) = 26

theorem ζ_sum_8 : ζ1^8 + ζ2^8 + ζ3^8 = 219 :=
by {
  -- The proof goes here, omitting solution steps as instructed.
  sorry
}

end ζ_sum_8_l233_233957


namespace arithmetic_progression_subsets_count_l233_233839

noncomputable def count_arithmetic_progression_subsets : Nat :=
  -- let S be the set {1, 2, 3, ..., 40}
  let S := Finset.range 41 \ Finset.singleton 0
  
  -- counting subsets A ⊆ S containing exactly 3 elements that form an arithmetic progression.
  -- We know the answer is 380 based on the problem statement.
  Nat.choose 20 2 + Nat.choose 20 2

theorem arithmetic_progression_subsets_count :
  count_arithmetic_progression_subsets = 380 :=
by
  sorry

end arithmetic_progression_subsets_count_l233_233839


namespace lateral_surface_area_of_cylinder_l233_233070

theorem lateral_surface_area_of_cylinder (Q : ℝ) (hQ : Q ≥ 0) :
    let r := Real.sqrt Q in
    let h := Real.sqrt Q in
    2 * π * r * h = 2 * π * Q := 
by
  sorry

end lateral_surface_area_of_cylinder_l233_233070


namespace arithmetic_progression_sum_geometric_progression_sum_l233_233180

-- Arithmetic Progression Problem
theorem arithmetic_progression_sum (d : ℚ) :
  let S_n (n : ℕ) := n * (1 + (n - 1) / 2 * d) in
  S_n 10 = 70 → 
  ∀ n, S_n n = n * (1 + (n - 1) / 2 * (4 / 3)) := 
by
  sorry

-- Geometric Progression Problem
theorem geometric_progression_sum (q : ℚ) :
  let a_n (n : ℕ) := 1 * q ^ (n - 1) in
  let S_n (n : ℕ) := (1 * (1 - q ^ n)) / (1 - q) in
  a_n 4 = 1 / 8 → 
  S_n n > 100 * a_n n → 
  n ≥ 7 :=
by
  sorry

end arithmetic_progression_sum_geometric_progression_sum_l233_233180


namespace pool_length_l233_233992

def volume_of_pool (width length depth : ℕ) : ℕ :=
  width * length * depth

def volume_of_water (volume : ℕ) (capacity : ℝ) : ℝ :=
  volume * capacity

theorem pool_length (L : ℕ) (width depth : ℕ) (capacity : ℝ) (drain_rate drain_time : ℕ) (h_capacity : capacity = 0.80)
  (h_width : width = 50) (h_depth : depth = 10)
  (h_drain_rate : drain_rate = 60) (h_drain_time : drain_time = 1000)
  (h_drain_volume : volume_of_water (volume_of_pool width L depth) capacity = drain_rate * drain_time) :
  L = 150 :=
by
  sorry

end pool_length_l233_233992


namespace count_congruent_to_3_mod_7_lt_500_l233_233861

theorem count_congruent_to_3_mod_7_lt_500 : 
  ∃ n, n = 71 ∧ ∀ x, 0 < x ∧ x < 500 ∧ x % 7 = 3 ↔ ∃ k, 0 ≤ k ∧ k ≤ 70 ∧ x = 3 + 7 * k :=
sorry

end count_congruent_to_3_mod_7_lt_500_l233_233861


namespace prime_factor_4k1_l233_233202

noncomputable def seq : ℕ → ℕ
| 0     := 0
| 1     := 1
| 2     := 2
| (n+3) := 2 * seq (n+2) + seq (n+1)

theorem prime_factor_4k1 (n : ℕ) (hn : n ≥ 5) : 
  ∃ p : ℕ, prime p ∧ p ∣ seq n ∧ p % 4 = 1 :=
sorry

end prime_factor_4k1_l233_233202


namespace greatest_prime_factor_of_341_l233_233607

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l233_233607


namespace find_value_of_m_l233_233644

variables (x y m : ℝ)

theorem find_value_of_m (h1 : y ≥ x) (h2 : x + 3 * y ≤ 4) (h3 : x ≥ m) (hz_max : ∀ z, (z = x - 3 * y) → z ≤ 8) :
  m = -4 :=
sorry

end find_value_of_m_l233_233644


namespace number_of_elements_cong_set_l233_233855

/-- Define a set of integers less than 500 and congruent to 3 modulo 7 -/
def cong_set : Set ℕ := {n | n < 500 ∧ n % 7 = 3}

/-- The theorem stating the number of elements in cong_set is 72 -/
theorem number_of_elements_cong_set : Set.card cong_set = 72 :=
sorry

end number_of_elements_cong_set_l233_233855


namespace value_of_sum_plus_five_l233_233401

theorem value_of_sum_plus_five (a b : ℕ) (h : 4 * a^2 + 4 * b^2 + 8 * a * b = 100) :
  (a + b) + 5 = 10 :=
sorry

end value_of_sum_plus_five_l233_233401


namespace greatest_prime_factor_of_341_is_17_l233_233556

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l233_233556


namespace tan_cot_inequality_l233_233372

open Real

theorem tan_cot_inequality (n : ℕ) (θ : Fin n → ℝ)
  (hθ : ∀ i, 0 < θ i ∧ θ i < π / 2) :
  (∑ i, tan (θ i)) * (∑ i, cot (θ i))  ≥ (∑ i, sin (θ i))^2 + (∑ i, cos (θ i))^2 :=
by sorry

end tan_cot_inequality_l233_233372


namespace ratio_of_sums_is_28_l233_233884

theorem ratio_of_sums_is_28 (a : ℝ) (r : ℝ) (h : r = 3) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 28 :=
by
  rw h
  sorry

end ratio_of_sums_is_28_l233_233884


namespace find_number_l233_233487

theorem find_number (x : ℝ) (h : 20 / x = 0.8) : x = 25 := 
by
  sorry

end find_number_l233_233487


namespace neg_p_is_true_l233_233173

-- Given condition
def p_statement : Prop := ∀ x : ℝ, real.sqrt (x^2) = x

-- Proof problem to prove the negation of p_statement is true
theorem neg_p_is_true : ¬ p_statement :=
begin
  sorry -- Proof goes here
end

end neg_p_is_true_l233_233173


namespace PA_sq_QB_eq_QA_sq_PB_l233_233910

variable {Point : Type}

-- Define the equilateral triangle and involved points
variables {A B C M N K L P Q : Point}

-- Define midpoints and line intersections
variable [EquilateralTriangle A B C]
variable [Midpoint M A B]
variable [Midpoint N A C]
variable [LineIntersectsCircumcircle MN K L A B C]
variable [IntersectsLine CK AB P]
variable [IntersectsLine CL AB Q]

-- Goal: Prove the given geometrical relationship
theorem PA_sq_QB_eq_QA_sq_PB : 
  PA^2 * QB = QA^2 * PB := 
  sorry

end PA_sq_QB_eq_QA_sq_PB_l233_233910


namespace sequence_2020_tanh_identity_l233_233928

theorem sequence_2020_tanh_identity:
  let f : ℕ → ℝ := λ n : ℕ, Nat.recOn n 2020 (λ n' fn, (fn + 2020) / (2020 * fn + 1))
  in f 2020 = Real.tanh (2021 * Real.atanh 2020) :=
by
  sorry

end sequence_2020_tanh_identity_l233_233928


namespace tower_surface_area_is_1430_l233_233394

noncomputable def total_surface_area_of_tower (volumes : List ℕ) (extra_area : ℕ) : ℕ :=
  let side_length v := Int.natAbs (Int.ofNat v).cbrt
  let surface_area s := 6 * s * s
  let adjusted_surface_area (s_prev s_cur : ℕ) := surface_area s_cur - s_prev * s_prev
  let total_area := (volumes.map side_length).foldl (fun (acc, s_prev) s_cur =>
    let area := adjusted_surface_area s_prev s_cur in
    (acc + area, s_cur)) (0, 0: ℕ)
  total_area.1 + extra_area

theorem tower_surface_area_is_1430 :
  total_surface_area_of_tower [1, 8, 27, 64, 125, 216, 343, 512, 729] 4 = 1430 := sorry

end tower_surface_area_is_1430_l233_233394


namespace intersect_A_B_when_a_1_subset_A_B_range_a_l233_233388

def poly_eqn (x : ℝ) : Prop := -x ^ 2 - 2 * x + 8 = 0

def sol_set_A : Set ℝ := {x | poly_eqn x}

def inequality (a x : ℝ) : Prop := a * x - 1 ≤ 0

def sol_set_B (a : ℝ) : Set ℝ := {x | inequality a x}

theorem intersect_A_B_when_a_1 :
  sol_set_A ∩ sol_set_B 1 = { -4 } :=
sorry

theorem subset_A_B_range_a (a : ℝ) :
  sol_set_A ⊆ sol_set_B a ↔ (-1 / 4 : ℝ) ≤ a ∧ a ≤ 1 / 2 :=
sorry
 
end intersect_A_B_when_a_1_subset_A_B_range_a_l233_233388


namespace collinear_points_x_value_l233_233329

variables {A B C : ℝ × ℝ}

-- Define points A, B, and C with the given coordinates
def pointA : ℝ × ℝ := (3, -2)
def pointB : ℝ × ℝ := (-9, 4)
def pointC : ℝ → ℝ × ℝ := fun x => (x, 0)

-- Define the function to compute the slope between two points
noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

-- Collinearity condition represented by equal slopes
theorem collinear_points_x_value (x : ℝ) : 
  slope pointA pointB = slope pointA (pointC x) → 
  x = -1 := by
  sorry

end collinear_points_x_value_l233_233329


namespace non_congruent_triangles_with_perimeter_20_l233_233232
-- Import the complete Mathlib library

-- The statement of our theorem
theorem non_congruent_triangles_with_perimeter_20 : 
  ∃ (triangles : Finset (ℤ × ℤ × ℤ)), 
  (∀ t ∈ triangles, let ⟨a, b, c⟩ := t in a ≤ b ∧ b ≤ c ∧ a + b + c = 20 ∧ a + b > c) ∧ 
  triangles.card = 40 :=
sorry

end non_congruent_triangles_with_perimeter_20_l233_233232


namespace nonnegative_solutions_eq_one_l233_233290

theorem nonnegative_solutions_eq_one : 
  (finset.univ.filter (λ x : ℝ, x^2 = -6 * x ∧ 0 <= x)).card = 1 := 
by 
  sorry

end nonnegative_solutions_eq_one_l233_233290


namespace nonnegative_solutions_eq1_l233_233249

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l233_233249


namespace f_2015_2016_l233_233198

theorem f_2015_2016 (f : ℤ → ℤ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_periodic : ∀ x, f (x + 2) = -f x)
  (h_f1 : f 1 = 2) :
  f 2015 + f 2016 = -2 :=
sorry

end f_2015_2016_l233_233198


namespace nonnegative_solution_count_nonnegative_solution_count_unique_l233_233283

theorem nonnegative_solution_count (x : ℝ) :
  (x^2 = -6 * x) → (x ≥ 0) → x = 0 :=
by
  sorry

-- Prove that the number of nonnegative solutions is 1
theorem nonnegative_solution_count_unique :
  {x : ℝ // x ≥ 0 ∧ x^2 = -6 * x}.to_finset.card = 1 :=
by
  sorry

end nonnegative_solution_count_nonnegative_solution_count_unique_l233_233283


namespace gold_coins_l233_233989

theorem gold_coins (c n : ℕ) 
  (h₁ : n = 8 * (c - 1))
  (h₂ : n = 5 * c + 4) :
  n = 24 :=
by
  sorry

end gold_coins_l233_233989


namespace train_speed_correct_l233_233085

-- Define the problem conditions
def length_of_train : ℝ := 300  -- length in meters
def time_to_cross_pole : ℝ := 18  -- time in seconds

-- Conversion factors
def meters_to_kilometers : ℝ := 0.001
def seconds_to_hours : ℝ := 1 / 3600

-- Define the conversions
def distance_in_kilometers := length_of_train * meters_to_kilometers
def time_in_hours := time_to_cross_pole * seconds_to_hours

-- Define the speed calculation
def speed_of_train := distance_in_kilometers / time_in_hours

-- The theorem to prove
theorem train_speed_correct : speed_of_train = 60 := 
by
  sorry

end train_speed_correct_l233_233085


namespace merchant_discount_l233_233063

variables (CP : ℝ) (MP SP : ℝ) (discount percentage_profit : ℝ)

def marked_price (CP : ℝ) : ℝ := CP + 0.60 * CP
def selling_price (CP : ℝ) : ℝ := CP + 0.20 * CP
def discount_value (MP SP : ℝ) : ℝ := MP - SP
def discount_percentage (discount MP : ℝ) : ℝ := (discount / MP) * 100

theorem merchant_discount :
  marked_price CP = 160 → 
  selling_price CP = 120 → 
  discount_value 160 120 = 40 →
  discount_percentage 40 160 = 25 :=
by
  sorry

end merchant_discount_l233_233063


namespace infinite_solutions_k_eq_5_l233_233732

theorem infinite_solutions_k_eq_5 (k : ℝ) : (∃∞ x : ℝ, 3 * (5 + k * x) = 15 * x + 15) → k = 5 := 
by
  intro h
  sorry

end infinite_solutions_k_eq_5_l233_233732


namespace count_congruent_to_3_mod_7_lt_500_l233_233859

theorem count_congruent_to_3_mod_7_lt_500 : 
  ∃ n, n = 71 ∧ ∀ x, 0 < x ∧ x < 500 ∧ x % 7 = 3 ↔ ∃ k, 0 ≤ k ∧ k ≤ 70 ∧ x = 3 + 7 * k :=
sorry

end count_congruent_to_3_mod_7_lt_500_l233_233859


namespace greatest_prime_factor_341_l233_233594

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l233_233594


namespace integer_points_in_intersection_l233_233716

noncomputable def numIntegerPointsInIntersection : ℕ :=
  let pairs := Finset.filter (λ (xyz : ℤ × ℤ × ℤ),
    let (x, y, z) := xyz in
    x^2 + y^2 + (z - 10)^2 ≤ 64 ∧ x^2 + y^2 + (z - 2)^2 ≤ 36) (Finset.univ : Finset (ℤ × ℤ × ℤ))
  pairs.card

theorem integer_points_in_intersection :
  numIntegerPointsInIntersection = <correct_number_of_points> :=
sorry

end integer_points_in_intersection_l233_233716


namespace gold_coins_percent_l233_233104

variable (total_objects beads papers coins silver_gold total_gold : ℝ)
variable (h1 : total_objects = 100)
variable (h2 : beads = 15)
variable (h3 : papers = 10)
variable (h4 : silver_gold = 30)
variable (h5 : total_gold = 52.5)

theorem gold_coins_percent : (total_objects - beads - papers) * (100 - silver_gold) / 100 = total_gold :=
by 
  -- Insert proof here
  sorry

end gold_coins_percent_l233_233104


namespace truck_travel_distance_l233_233091

variable (d1 d2 g1 g2 : ℝ)
variable (rate : ℝ)

-- Define the conditions
axiom condition1 : d1 = 300
axiom condition2 : g1 = 10
axiom condition3 : rate = d1 / g1
axiom condition4 : g2 = 15

-- Define the goal
theorem truck_travel_distance : d2 = rate * g2 := by
  -- axiom assumption placeholder
  exact sorry

end truck_travel_distance_l233_233091


namespace prove_M_l233_233779

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {2, 3}
def M : Set ℕ := {x | x ∈ P ∧ x ∉ Q}

theorem prove_M :
  M = {1} :=
by
  sorry

end prove_M_l233_233779


namespace max_sum_of_ten_consecutive_terms_l233_233346

theorem max_sum_of_ten_consecutive_terms (a : ℕ → ℕ)
  (H1 : ∀ i, ∏ j in Icc i (i + 14), a j = 10^6)
  (H2 : ∀ i, ∑ j in Icc i (i + 9), a j = S) :
  S = 208 :=
sorry

end max_sum_of_ten_consecutive_terms_l233_233346


namespace greatest_prime_factor_341_l233_233582

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l233_233582


namespace time_to_process_600_parts_l233_233811

theorem time_to_process_600_parts:
  (let x := 600 in y = 0.01 * x + 0.5) → y = 6.5 :=
sorry

end time_to_process_600_parts_l233_233811


namespace tangent_slope_correct_l233_233792

noncomputable def slope_of_directrix (focus: ℝ × ℝ) (p1: ℝ × ℝ) (p2: ℝ × ℝ) : ℝ :=
  let c1 := p1
  let c2 := p2
  let radius1 := Real.sqrt ((c1.1 + 1)^2 + (c1.2 + 1)^2)
  let radius2 := Real.sqrt ((c2.1 - 2)^2 + (c2.2 - 2)^2)
  let dist := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let slope := (focus.2 - p1.2) / (focus.1 - p1.1)
  let tangent_slope := (9 : ℝ) / (7 : ℝ) + (4 * Real.sqrt 2) / 7
  tangent_slope

theorem tangent_slope_correct :
  (slope_of_directrix (0, 0) (-1, -1) (2, 2) = (9 + 4 * Real.sqrt 2) / 7) ∨
  (slope_of_directrix (0, 0) (-1, -1) (2, 2) = (9 - 4 * Real.sqrt 2) / 7) :=
by
  -- Proof omitted here
  sorry

end tangent_slope_correct_l233_233792


namespace zoo_feeding_ways_l233_233093

-- Noncomputable is used for definitions that are not algorithmically computable
noncomputable def numFeedingWays : Nat :=
  4 * 3 * 3 * 2 * 2

theorem zoo_feeding_ways :
  ∀ (pairs : Fin 4 → (String × String)), -- Representing pairs of animals
  numFeedingWays = 144 :=
by
  sorry

end zoo_feeding_ways_l233_233093


namespace complement_of_P_l233_233223

open Set

theorem complement_of_P :
  let U := ℝ
  let P := {y : ℝ | ∃ (x : ℝ), y = 1 / x ∧ 0 < x ∧ x < 1}
  complement U P = {x : ℝ | x ≤ 1} :=
by {
  sorry
}

end complement_of_P_l233_233223


namespace subsets_with_intersection_property_l233_233983

open Set

theorem subsets_with_intersection_property :
  ∃ (A : Fin 16 → Set ℕ), (∀ z ∈ {x : ℕ | x <= 10000}, ∃ (B : Finset (Fin 16)), B.card = 8 ∧ (z ∈ ⋂ i ∈ B, A i)) :=
begin
  sorry
end

end subsets_with_intersection_property_l233_233983


namespace even_function_m_value_l233_233207

theorem even_function_m_value (m : ℝ) (f : ℝ → ℝ) (h : f = λ x, x^2 + (m-2)*x + 1) : 
  (∀ x, f x = f (-x)) → m = 2 :=
by
  intro h_even
  sorry

end even_function_m_value_l233_233207


namespace parabola_axis_of_symmetry_l233_233880

theorem parabola_axis_of_symmetry (a b c : ℝ) (h₁ : (-6 : ℝ) = (-b + sqrt (b^2 - 4*a*c)) / (2*a))
  (h₂ : (4 : ℝ) = (-b - sqrt (b^2 - 4*a*c)) / (2*a)) : (-1 : ℝ) = (-6 + 4) / 2 :=
begin
  -- Proof is omitted in this exercise
  sorry
end

end parabola_axis_of_symmetry_l233_233880


namespace sin_cos_eq_values_l233_233136

theorem sin_cos_eq_values (θ : ℝ) (hθ : 0 < θ ∧ θ ≤ 2 * Real.pi) :
  (∃ t : ℝ, 
    0 < t ∧ 
    t ≤ 2 * Real.pi ∧ 
    (2 + 4 * Real.sin t - 3 * Real.cos (2 * t) = 0)) ↔ (∃ n : ℕ, n = 4) :=
by 
  sorry

end sin_cos_eq_values_l233_233136


namespace proof_problem_l233_233209

noncomputable def hyperbola_E_eq (x y : ℝ) : Prop :=
  y^2 / 9 - x^2 / 4 = 1

noncomputable def passes_through (x y : ℝ) (a b : ℝ) : Prop :=
  y = b ∧ x = a

noncomputable def asymptotes_E_and_F (x y : ℝ) : Prop :=
  y^2 / 9 - x^2 / 4 = 1 ∨ x^2 / 4 - y^2 / 9 = 1

noncomputable def midpoint_line (A B : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

noncomputable def line_eq (A B : ℝ × ℝ) : ℝ × Prop :=
  let slope := (B.2 - A.2) / (B.1 - A.1) in
  (slope, ∀ x y : ℝ, y - A.2 = slope * (x - A.1))

noncomputable def tangent_line_eq (x₁ y₁ x y : ℝ) : Prop :=
  9 * x₁ * x - 4 * y₁ * y + 36 = 0

noncomputable def point_moves_on_line (P : ℝ × ℝ) : Prop :=
  3 * P.1 - 4 * P.2 + 6 = 0

theorem proof_problem :
  (∃ (x y : ℝ), passes_through x y (-2) (3 * real.sqrt 2) ∧ asymptotes_E_and_F x y) →
  ¬(∃ (x y : ℝ), y^2 / 18 - x^2 / 8 = 1) ∧ 
  (∃ (A B : ℝ × ℝ), midpoint_line A B (1, 4) ∧ line_eq A B = (9/16, ∀ x y, y = 9/16 * x + 55)) ∧
  (∃ (x₁ y₁ : ℝ), tangent_line_eq x₁ y₁ x₁ y₁) ∧
  (∀ P : ℝ × ℝ, point_moves_on_line P → (∃ (l : ℝ × Prop), passes_through l.1 (3, 6))) :=
sorry

end proof_problem_l233_233209


namespace units_digit_of_N_l233_233921

theorem units_digit_of_N :
  ∃ (N : ℕ), (∀ (n : ℕ), n < 7 -> 
    ∃ (a b : ℕ), N = a^2 + b /\
      (a^2 + b - (nat.sqrt(a^2 + b))^2 = nat.pnat_coe b)) /
      (N % 10 = 3)  :=
begin
  sorry
end

end units_digit_of_N_l233_233921


namespace cos_value_l233_233781

theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by sorry

end cos_value_l233_233781


namespace polynomial_division_l233_233375

open Polynomial

-- Define the theorem statement
theorem polynomial_division (f g : ℤ[X])
  (h : ∀ n : ℤ, f.eval n ∣ g.eval n) :
  ∃ (h : ℤ[X]), g = f * h :=
sorry

end polynomial_division_l233_233375


namespace parallel_vectors_sum_coords_l233_233228

theorem parallel_vectors_sum_coords
  (x y : ℝ)
  (a b : ℝ × ℝ × ℝ)
  (h_a : a = (2, x, 3))
  (h_b : b = (-4, 2, y))
  (h_parallel : ∃ k : ℝ, a = k • b) :
  x + y = -7 :=
sorry

end parallel_vectors_sum_coords_l233_233228


namespace train_speed_l233_233087

def distance := 300 -- meters
def time := 18 -- seconds

noncomputable def speed_kmh := 
  let speed_ms := distance / time -- speed in meters per second
  speed_ms * 3.6 -- convert to kilometers per hour

theorem train_speed : speed_kmh = 60 := 
  by
    -- The proof steps are omitted
    sorry

end train_speed_l233_233087


namespace greatest_prime_factor_341_l233_233588

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l233_233588


namespace train_speed_correct_l233_233084

-- Define the problem conditions
def length_of_train : ℝ := 300  -- length in meters
def time_to_cross_pole : ℝ := 18  -- time in seconds

-- Conversion factors
def meters_to_kilometers : ℝ := 0.001
def seconds_to_hours : ℝ := 1 / 3600

-- Define the conversions
def distance_in_kilometers := length_of_train * meters_to_kilometers
def time_in_hours := time_to_cross_pole * seconds_to_hours

-- Define the speed calculation
def speed_of_train := distance_in_kilometers / time_in_hours

-- The theorem to prove
theorem train_speed_correct : speed_of_train = 60 := 
by
  sorry

end train_speed_correct_l233_233084


namespace nonnegative_solution_count_l233_233274

theorem nonnegative_solution_count : 
  ∃! x : ℝ, x^2 = -6 * x ∧ 0 ≤ x := 
begin
  sorry
end

end nonnegative_solution_count_l233_233274


namespace probability_neither_prime_nor_composite_l233_233883

/-- Definition of prime number: A number is prime if it has exactly two distinct positive divisors -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- Definition of composite number: A number is composite if it has more than two positive divisors -/
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m : ℕ, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

/-- Given the number in the range 1 to 98 -/
def neither_prime_nor_composite (n : ℕ) : Prop := n = 1

/-- Probability function for uniform probability in a discrete sample space -/
def probability (event_occurrences total_possibilities : ℕ) : ℚ := event_occurrences / total_possibilities

theorem probability_neither_prime_nor_composite :
    probability 1 98 = 1 / 98 := by
  sorry

end probability_neither_prime_nor_composite_l233_233883


namespace original_price_second_store_l233_233986

-- Definitions of the conditions
def price_first_store : ℝ := 950
def discount_first_store : ℝ := 0.06
def discount_second_store : ℝ := 0.05
def price_difference : ℝ := 19

-- Define the discounted price function
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

-- State the main theorem
theorem original_price_second_store :
  ∃ P : ℝ, 
    (discounted_price price_first_store discount_first_store - discounted_price P discount_second_store = price_difference) ∧ 
    P = 960 :=
by
  sorry

end original_price_second_store_l233_233986


namespace count_congruent_numbers_less_than_500_l233_233852

-- Definitions of the conditions
def is_congruent_to_modulo (n a m : ℕ) : Prop := (n % m) = a

-- Main problem statement: Proving that the count of numbers under 500 that satisfy the conditions is 71.
theorem count_congruent_numbers_less_than_500 : 
  { n : ℕ | n < 500 ∧ is_congruent_to_modulo n 3 7 }.card = 71 :=
by
  sorry

end count_congruent_numbers_less_than_500_l233_233852


namespace number_of_nonnegative_solutions_l233_233237

-- Define the equation x^2 = -6x
def equation (x : ℝ) : Prop := x^2 = -6 * x

-- Define the condition for a nonnegative solution
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Define the main theorem to prove the number of nonnegative solutions
theorem number_of_nonnegative_solutions : 
  {x : ℝ // equation x ∧ nonnegative x}.to_finset.card = 1 :=
sorry

end number_of_nonnegative_solutions_l233_233237


namespace percentage_of_integers_divisible_by_6_up_to_120_l233_233618

theorem percentage_of_integers_divisible_by_6_up_to_120 : 
  let total := 120
      divisible_by_6 := λ n, n % 6 = 0
      count := (list.range (total + 1)).countp divisible_by_6
      percentage := (count.toFloat / total.toFloat) * 100
  in percentage = 16.67 :=
by
  sorry

end percentage_of_integers_divisible_by_6_up_to_120_l233_233618


namespace last_integer_cannot_be_1_l233_233976

   theorem last_integer_cannot_be_1 (n : ℕ) (H : n = 2012) :
     ∀ lst : List ℕ, lst = List.range' 1 H.succ →
     (∃! k : ℕ, (∀ a b ∈ lst, lst.erase a.erase b.append (abs (a - b)) = lst) → k ≠ 1) :=
   by
     sorry
   
end last_integer_cannot_be_1_l233_233976


namespace fair_coin_three_flips_probability_l233_233026

theorem fair_coin_three_flips_probability :
  ∀ (prob : ℕ → ℚ) (independent : ∀ n, prob n = 1 / 2),
    prob 0 * prob 1 * prob 2 = 1 / 8 := 
by
  intros prob independent
  sorry

end fair_coin_three_flips_probability_l233_233026


namespace nonnegative_solution_count_l233_233277

theorem nonnegative_solution_count : 
  ∃! x : ℝ, x^2 = -6 * x ∧ 0 ≤ x := 
begin
  sorry
end

end nonnegative_solution_count_l233_233277


namespace triangle_area_collinear_l233_233355

theorem triangle_area_collinear (A B C a b c : ℝ)
  (hC : C = π / 3)
  (hc : c = 2)
  (hBA : B - A = 2 * π / 3 - 2 * A)
  (h_collinear_m_n : ∃ m n : ℝ × ℝ, m = (sin C + sin (B - A), 4) ∧ n = (sin 2 * A, 1) ∧ 
    (m.snd * n.fst = n.snd * m.fst)) :
  let area1 := 2 * sqrt 3 / 3 in
  let area2 := 4 * sqrt 13 / 13 in
  ∃ area : ℝ, (area = area1 ∨ area = area2) := 
sorry

end triangle_area_collinear_l233_233355


namespace largest_six_consecutive_composites_less_than_40_l233_233755

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) := ¬ is_prime n ∧ n > 1

theorem largest_six_consecutive_composites_less_than_40 :
  ∃ (seq : ℕ → ℕ) (i : ℕ),
    (∀ j : ℕ, j < 6 → is_composite (seq (i + j))) ∧ 
    (seq i < 40) ∧ 
    (seq (i+1) < 40) ∧ 
    (seq (i+2) < 40) ∧ 
    (seq (i+3) < 40) ∧ 
    (seq (i+4) < 40) ∧ 
    (seq (i+5) < 40) ∧ 
    seq (i+5) = 30 
:= sorry

end largest_six_consecutive_composites_less_than_40_l233_233755


namespace undefined_hydrogen_production_l233_233849

-- Define the chemical species involved as follows:
structure ChemQty where
  Ethane : ℕ
  Oxygen : ℕ
  CarbonDioxide : ℕ
  Water : ℕ

-- Balanced reaction equation
def balanced_reaction : ChemQty :=
  { Ethane := 2, Oxygen := 7, CarbonDioxide := 4, Water := 6 }

-- Given conditions as per problem scenario
def initial_state : ChemQty :=
  { Ethane := 1, Oxygen := 2, CarbonDioxide := 0, Water := 0 }

-- The statement reflecting the unclear result of the reaction under the given conditions.
theorem undefined_hydrogen_production :
  initial_state.Oxygen < balanced_reaction.Oxygen / balanced_reaction.Ethane * initial_state.Ethane →
  ∃ water_products : ℕ, water_products ≤ 6 * initial_state.Ethane / 2 := 
by
  -- Due to incomplete reaction
  sorry

end undefined_hydrogen_production_l233_233849


namespace evaluate_expression_min_value_l233_233746

theorem evaluate_expression_min_value (x : ℝ) (h : 0 < x ∧ x < π/2) : 
  (tan x + cot x)^2 + (sin x + sec x)^2 = 3 := 
sorry

end evaluate_expression_min_value_l233_233746


namespace miss_tree_class_children_count_l233_233006

noncomputable def number_of_children (n: ℕ) : ℕ := 7 * n + 2

theorem miss_tree_class_children_count (n : ℕ) :
  (20 < number_of_children n) ∧ (number_of_children n < 30) ∧ 7 * n + 2 = 23 :=
by {
  sorry
}

end miss_tree_class_children_count_l233_233006


namespace length_PQ_in_triangle_l233_233889

theorem length_PQ_in_triangle
  (D E F P Q : Type)
  (DE EF DF : ℝ)
  (hDE : DE = 17)
  (hEF : EF = 18)
  (hDF : DF = 19)
  (midpoint_P : P = midpoint D E)
  (perpendicular_Q : orthogonal_projection D EF Q)
  : dist P Q = 8.5 :=
by
  sorry

end length_PQ_in_triangle_l233_233889


namespace number_of_integers_congruent_to_3_mod_7_less_than_500_l233_233865

theorem number_of_integers_congruent_to_3_mod_7_less_than_500 : 
  {n : ℕ | n < 500 ∧ n % 7 = 3}.card = 71 :=
sorry

end number_of_integers_congruent_to_3_mod_7_less_than_500_l233_233865


namespace tennis_tournament_l233_233899

theorem tennis_tournament (n x : ℕ) 
    (p : ℕ := 4 * n) 
    (m : ℕ := (p * (p - 1)) / 2) 
    (r_women : ℕ := 3 * x) 
    (r_men : ℕ := 2 * x) 
    (total_wins : ℕ := r_women + r_men) 
    (h_matches : m = total_wins) 
    (h_ratio : r_women = 3 * x ∧ r_men = 2 * x ∧ 4 * n * (4 * n - 1) = 10 * x): 
    n = 4 :=
by
  sorry

end tennis_tournament_l233_233899


namespace maximize_revenue_price_l233_233051

def number_of_books_sold (p : ℝ) : ℝ :=
  if p ≤ 15 then 150 - 6 * p else 120 - 4 * p

def revenue (p : ℝ) : ℝ :=
  p * (number_of_books_sold p)

theorem maximize_revenue_price (p : ℝ) (h₁ : p ≥ 0) (h₂ : p ≤ 30) :
  (∀ p', (h₁ : p' ≥ 0) ∧ (h₂ : p' ≤ 30) → revenue p' ≤ revenue 15) :=
by
  sorry

end maximize_revenue_price_l233_233051


namespace area_ratio_problem_l233_233903

theorem area_ratio_problem
  (A B C : ℝ) -- Areas of the corresponding regions
  (m n : ℕ)  -- Given ratios
  (PQR_is_right_triangle : true)  -- PQR is a right-angled triangle (placeholder condition)
  (RSTU_is_rectangle : true)  -- RSTU is a rectangle (placeholder condition)
  (ratio_A_B : A / B = m / 2)  -- Ratio condition 1
  (ratio_A_C : A / C = n / 1)  -- Ratio condition 2
  (PTS_sim_TQU_sim_PQR : true)  -- Similar triangles (placeholder condition)
  : n = 9 := 
sorry

end area_ratio_problem_l233_233903


namespace symmetric_point_midpoint_l233_233799

structure Point3D (α : Type _) :=
  (x : α)
  (y : α)
  (z : α)

def symmetricMidpointCoordinate (A B : Point3D ℝ) : Point3D ℝ :=
  let Mx := (A.x + B.x) / 2
  let My := (A.y + B.y) / 2
  let Mz := (A.z + B.z) / 2
  Point3D.mk (-Mx) My Mz

theorem symmetric_point_midpoint (A B : Point3D ℝ) :
  A = Point3D.mk (-3) 1 (-4) →
  B = Point3D.mk 7 1 0 →
  symmetricMidpointCoordinate A B = Point3D.mk (-2) 1 (-2) := by
  intros hA hB
  -- proof omitted
  sorry

end symmetric_point_midpoint_l233_233799


namespace import_tax_applied_amount_l233_233682

theorem import_tax_applied_amount 
    (total_value : ℝ) 
    (import_tax_paid : ℝ)
    (tax_rate : ℝ) 
    (excess_amount : ℝ) 
    (condition1 : total_value = 2580) 
    (condition2 : import_tax_paid = 110.60) 
    (condition3 : tax_rate = 0.07) 
    (condition4 : import_tax_paid = tax_rate * (total_value - excess_amount)) : 
    excess_amount = 1000 :=
by
  sorry

end import_tax_applied_amount_l233_233682


namespace a_and_b_together_complete_work_in_12_days_l233_233033

-- Define the rate of work for b
def R_b : ℚ := 1 / 60

-- Define the rate of work for a based on the given condition that a is four times as fast as b
def R_a : ℚ := 4 * R_b

-- Define the combined rate of work for a and b working together
def R_a_plus_b : ℚ := R_a + R_b

-- Define the target time
def target_time : ℚ := 12

-- Proof statement
theorem a_and_b_together_complete_work_in_12_days :
  (R_a_plus_b * target_time) = 1 :=
by
  -- Proof omitted
  sorry

end a_and_b_together_complete_work_in_12_days_l233_233033


namespace rationalize_denominator_l233_233417

theorem rationalize_denominator :
  let expr := (2 + Real.sqrt 5) / (3 - Real.sqrt 5),
      A := 11 / 4,
      B := 5 / 4,
      C := 5
  in expr = (A + B * Real.sqrt C) → (A * B * C) = 275 / 16 :=
by
  intros
  sorry

end rationalize_denominator_l233_233417


namespace eva_total_marks_correct_l233_233735

-- Definitions based on conditions
def math_marks_second_sem : ℕ := 80
def arts_marks_second_sem : ℕ := 90
def science_marks_second_sem : ℕ := 90

def math_marks_first_sem : ℕ := math_marks_second_sem + 10
def arts_marks_first_sem : ℕ := arts_marks_second_sem - 15
def science_marks_first_sem : ℕ := science_marks_second_sem - (science_marks_second_sem / 3)

def total_marks_first_sem : ℕ := math_marks_first_sem + arts_marks_first_sem + science_marks_first_sem
def total_marks_second_sem : ℕ := math_marks_second_sem + arts_marks_second_sem + science_marks_second_sem

def total_marks_both_sems : ℕ := total_marks_first_sem + total_marks_second_sem

-- Theorem to be proved
theorem eva_total_marks_correct : total_marks_both_sems = 485 := by
  -- Here, we state that we need to prove the total marks sum up to 485
  sorry

end eva_total_marks_correct_l233_233735


namespace total_fourth_division_is_260_l233_233900

open Real

def total_students : ℝ := 500

def division_criteria (subject : String) (percents : List ℝ) : List ℝ := 
  percents.map (λ p => p * total_students / 100)

def math_divisions := division_criteria "Math" [10, 40, 35, 15]
def english_divisions := division_criteria "English" [15, 45, 30, 10]
def science_divisions := division_criteria "Science" [8, 50, 35, 7]
def social_studies_divisions := division_criteria "Social Studies" [20, 35, 35, 10]
def language_divisions := division_criteria "Language" [12, 48, 30, 10]

def fourth_division_students (subject_divisions : List ℝ) := subject_divisions.getLast!

def total_fourth_division_students : ℝ :=
  fourth_division_students math_divisions +
  fourth_division_students english_divisions +
  fourth_division_students science_divisions +
  fourth_division_students social_studies_divisions +
  fourth_division_students language_divisions

theorem total_fourth_division_is_260 : total_fourth_division_students = 260 :=
  sorry

end total_fourth_division_is_260_l233_233900


namespace function_is_identity_l233_233939

theorem function_is_identity (f : ℕ+ → ℤ)
  (h1 : ∀ n : ℕ+, true)  -- (1) f(n) is defined for every positive integer n
  (h2 : ∀ n : ℕ+, f(n) ∈ ℤ)  -- (2) f(n) is an integer
  (h3 : f 2 = 2)  -- (3) f(2) = 2
  (h4 : ∀ m n : ℕ+, f(m * n) = f(m) * f(n))  -- (4) f(m * n) = f(m) * f(n)
  (h5 : ∀ m n : ℕ+, m > n → f(m) > f(n)) :  -- (5) f(m) > f(n) when m > n
  ∀ n : ℕ+, f(n) = n := sorry

end function_is_identity_l233_233939


namespace sum_proof_l233_233321

theorem sum_proof (X Y : ℝ) (hX : 0.45 * X = 270) (hY : 0.35 * Y = 210) : 
  (0.75 * X) + (0.55 * Y) = 780 := by
  sorry

end sum_proof_l233_233321


namespace max_blocks_fit_l233_233021

-- Define the dimensions of the block
def block_length : ℕ := 3
def block_width : ℕ := 1
def block_height : ℕ := 1

-- Define the dimensions of the box
def box_length : ℕ := 5
def box_width : ℕ := 3
def box_height : ℕ := 2

-- Theorem stating the maximum number of blocks that can fit in the box
theorem max_blocks_fit :
  (box_length * box_width * box_height) / (block_length * block_width * block_height) = 15 := sorry

end max_blocks_fit_l233_233021


namespace greatest_prime_factor_341_l233_233598

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l233_233598


namespace nonnegative_solutions_eq_one_l233_233288

theorem nonnegative_solutions_eq_one : 
  (finset.univ.filter (λ x : ℝ, x^2 = -6 * x ∧ 0 <= x)).card = 1 := 
by 
  sorry

end nonnegative_solutions_eq_one_l233_233288


namespace angle_bisector_problem_l233_233018

variable (O F A B A' B' : Point)
variable (a b a' b' : ℝ)
variable (h_intersections : between O A F ∧ between O B F ∧ between O A' F ∧ between O B' F)
variable (h_lengths : dist O A = a ∧ dist O B = b ∧ dist O A' = a' ∧ dist O B' = b')

theorem angle_bisector_problem
  (O F A B A' B' : Point) (a b a' b' : ℝ)
  (h_intersections : between O A F ∧ between O B F ∧ between O A' F ∧ between O B' F)
  (h_lengths : dist O A = a ∧ dist O B = b ∧ dist O A' = a' ∧ dist O B' = b') :
  1/a + 1/b = 1/a' + 1/b' :=
sorry

end angle_bisector_problem_l233_233018


namespace base_10_to_base_5_l233_233509

noncomputable def base_five_equivalent (n : ℕ) : ℕ :=
  let (d1, r1) := div_mod n (5 * 5 * 5) in
  let (d2, r2) := div_mod r1 (5 * 5) in
  let (d3, r3) := div_mod r2 5 in
  let (d4, r4) := div_mod r3 1 in
  d1 * 1000 + d2 * 100 + d3 * 10 + d4

theorem base_10_to_base_5 : base_five_equivalent 156 = 1111 :=
by
  -- Include the proof here
  sorry

end base_10_to_base_5_l233_233509


namespace cyclic_points_on_quadrilateral_l233_233948

theorem cyclic_points_on_quadrilateral (A B C D P Q R S : Point)
  (h1 : ConvexQuadrilateral A B C D) 
  (h2 : IsAngleBisector B D (∠ A B C))
  (h3 : IsCircumcircleIntersectsSide (triangle A B C) (AD) P)
  (h4 : IsCircumcircleIntersectsSide (triangle A B C) (CD) Q)
  (h5 : ParallelTo (LineThrough D) (AC))
  (h6 : LineIntersectsAt (LineThrough D ParallelTo AC) (BC) R)
  (h7 : LineIntersectsAt (LineThrough D ParallelTo AC) (BA) S) :
  Cyclic P Q R S :=
by
  sorry

end cyclic_points_on_quadrilateral_l233_233948


namespace greatest_prime_factor_of_341_l233_233609

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l233_233609


namespace train_speed_is_60_kmph_l233_233077

-- Define the conditions
def length_of_train : ℕ := 300 -- in meters
def time_to_cross_pole : ℕ := 18 -- in seconds

-- Define the conversions
def meters_to_kilometers (m : ℕ) : ℝ := m / 1000.0
def seconds_to_hours (s : ℕ) : ℝ := s / 3600.0

-- Define the speed calculation
def speed_km_per_hr (distance_km : ℝ) (time_hr : ℝ) : ℝ := distance_km / time_hr

-- Prove that the speed of the train is 60 km/hr
theorem train_speed_is_60_kmph :
  speed_km_per_hr (meters_to_kilometers length_of_train) (seconds_to_hours time_to_cross_pole) = 60 := 
  by
    sorry

end train_speed_is_60_kmph_l233_233077


namespace cube_root_of_64_l233_233998

theorem cube_root_of_64 : ∃ a : ℝ, a^3 = 64 ∧ a = 4 :=
by
  use 4
  split
  case left =>
    show 4^3 = 64
    calc
      4^3 = 64 : by norm_num
  case right =>
    show 4 = 4
    rfl

end cube_root_of_64_l233_233998


namespace greatest_prime_factor_341_l233_233602

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l233_233602


namespace susie_sold_24_slices_l233_233432

theorem susie_sold_24_slices (slices_price whole_pizza_price total_revenue slices_revenue : ℕ) 
  (h1 : slices_price = 3)
  (h2 : whole_pizza_price = 15)
  (h3 : total_revenue = 117)
  (h4 : slices_revenue = total_revenue - 3 * whole_pizza_price) 
  (h5 : slices_revenue = 72) 
    : (slices_revenue / slices_price) = 24 :=
by
  rw [h1, h5]
  norm_num
  sorry

end susie_sold_24_slices_l233_233432


namespace percent_divisible_by_6_l233_233627

theorem percent_divisible_by_6 (N : ℕ) (hN : N = 120) :
  (∃ M, M = (finset.univ.filter (λ n : ℕ, n ≤ N ∧ n % 6 = 0)).card ∧ M * 6 = N) →
  (M.to_real / N.to_real) * 100 = 16.66666667 :=
by
  intros h
  sorry

end percent_divisible_by_6_l233_233627


namespace probability_product_multiple_of_36_l233_233334

theorem probability_product_multiple_of_36 :
  ∃ s : finset ℕ, s = {6, 8, 9, 12} ∧
    (finset.card (finset.pairs s) = 6) ∧ 
    let valid_pairs := (finset.filter (λ p, 36 ∣ p.1 * p.2) (finset.pairs s)) in 
    (finset.card valid_pairs / finset.card (finset.pairs s) = 1 / 3) := sorry

end probability_product_multiple_of_36_l233_233334


namespace find_y_intercept_l233_233455

theorem find_y_intercept (m : ℝ) (x_intercept : ℝ × ℝ) (hx : x_intercept = (4, 0)) (hm : m = -3) : ∃ y_intercept : ℝ × ℝ, y_intercept = (0, 12) := 
by
  sorry

end find_y_intercept_l233_233455


namespace calculate_expression_l233_233708

theorem calculate_expression : (3.14 - Real.pi)^0 + |Real.sqrt 2 - 1| + (1/2 : ℝ)^(-1) - Real.sqrt 8 = 2 - Real.sqrt 2 :=
by
  sorry

end calculate_expression_l233_233708


namespace triangle_acute_l233_233450

theorem triangle_acute (t : ℝ) (ht : t > 0) : 
  let a := 6 * t in 
  let b := 8 * t in 
  let c := 9 * t in 
  a^2 + b^2 > c^2 :=
by
  let a := 6 * t
  let b := 8 * t
  let c := 9 * t
  sorry

end triangle_acute_l233_233450


namespace central_angle_of_sector_l233_233179

theorem central_angle_of_sector (r l : ℝ) (h1 : l + 2 * r = 4) (h2 : (1 / 2) * l * r = 1) : l / r = 2 :=
by
  -- The proof should be provided here
  sorry

end central_angle_of_sector_l233_233179


namespace ca1_l233_233324

theorem ca1 {
  a b : ℝ
} (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := 
by
  sorry

end ca1_l233_233324


namespace greatest_prime_factor_341_l233_233589

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l233_233589


namespace greatest_prime_factor_341_l233_233580

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l233_233580


namespace equipment_total_cost_l233_233463

-- Definition of costs for each item of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.20
def socks_cost : ℝ := 6.80

-- Number of players
def num_players : ℕ := 16

-- Total cost for one player
def total_cost_one_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Total cost for all players
def total_cost_all_players : ℝ := total_cost_one_player * num_players

-- Theorem to prove
theorem equipment_total_cost : total_cost_all_players = 752 := by
  sorry

end equipment_total_cost_l233_233463


namespace knife_value_l233_233013

def sheep_sold (n : ℕ) : ℕ := n * n

def valid_units_digits (m : ℕ) : Bool :=
  (m ^ 2 = 16) ∨ (m ^ 2 = 36)

theorem knife_value (n : ℕ) (k : ℕ) (m : ℕ) (H1 : sheep_sold n = n * n) (H2 : n = 10 * k + m) (H3 : valid_units_digits m = true) :
  2 = 2 :=
by
  sorry

end knife_value_l233_233013


namespace calculate_mod_sum_l233_233704

theorem calculate_mod_sum :
  (2⁻¹ + 2⁻² + 2⁻³ + 2⁻⁴ + 2⁻⁵ + 2⁻⁶ + 2⁻⁷ : ℤ) % 11 = 8 := by
{
  have h1 := (by rw [Int.pow_nat_cast, Int.mod_eq_of_lt, Nat.mod_self] : (2 ^ 7 : ℤ) % 11 = 7),
  have h_inv7 := (by rw [←Int.mul_left_inj, Int.mul_comm, Int.mul_one, Int.ofNat_inj, Int.coe_nat_eq_coe_nat_iff, Nat.mul_mod_right] : (7 : ℤ)⁻¹ % 11 = 8),
  have h_inv6 := (by rw [Int.mul_right_inj, Int.mul_comm, Int.mul_one, Int.coe_nat_eq_coe_nat_iff, Nat.mul_mod_right] : (6 : ℤ)⁻¹ % 11 = 6),
  have h_inv3 := (by rw [Int.mul_right_inj, Int.mul_comm, Int.mul_one, Int.coe_nat_eq_coe_nat_iff, Nat.mul_mod_right] : (3 : ℤ)⁻¹ % 11 = 3),
  have h_inv2 := (by rw [Int.mul_right_inj, Int.mul_comm, Int.mul_one, Int.coe_nat_eq_coe_nat_iff, Nat.mul_mod_right] : (2 : ℤ)⁻¹ % 11 = 6),
  
  sorry
}

end calculate_mod_sum_l233_233704


namespace unique_point_at_1_l233_233331

theorem unique_point_at_1 (m : ℝ) :
  (∃! x ∈ set.Icc (0 : ℝ) m, real.sin (x + real.pi / 3) = 1) ↔ m = real.pi / 6 :=
by sorry

end unique_point_at_1_l233_233331


namespace nonnegative_solution_count_nonnegative_solution_count_unique_l233_233285

theorem nonnegative_solution_count (x : ℝ) :
  (x^2 = -6 * x) → (x ≥ 0) → x = 0 :=
by
  sorry

-- Prove that the number of nonnegative solutions is 1
theorem nonnegative_solution_count_unique :
  {x : ℝ // x ≥ 0 ∧ x^2 = -6 * x}.to_finset.card = 1 :=
by
  sorry

end nonnegative_solution_count_nonnegative_solution_count_unique_l233_233285


namespace range_of_a_l233_233807

theorem range_of_a 
  (f : ℝ → ℝ) 
  (h_even : ∀ x, f x = f (-x))
  (h_monotone : ∀ x y, (0 ≤ x ∧ x < y) → f y < f x) 
  (a : ℝ) 
  (h_ineq : f (Real.log 2 (1 / a)) < f (-1 / 2)) : 
  (0 < a ∧ a < Real.sqrt 2 / 2) ∨ (a > Real.sqrt 2) :=
  sorry

end range_of_a_l233_233807


namespace mutual_acquaintance_or_none_l233_233501

theorem mutual_acquaintance_or_none (n : ℕ) (h : n > 3) (G : Finset (Fin n)) 
  (d : Fin n → ℕ) 
  (condition : ∀ (i : Fin n), ∀ (R : Fin n → Prop), (∀ j, R j ↔ j ∈ G ∧ j ≠ i) → 
                (d i = (∑ j in G, if i = j then 0 else d j))) :
  (∀ i, d i = 0) ∨ (∀ i, d i = n-1) :=
sorry

end mutual_acquaintance_or_none_l233_233501


namespace nonnegative_solution_count_l233_233269

theorem nonnegative_solution_count : 
  ∃! x : ℝ, x^2 = -6 * x ∧ 0 ≤ x := 
begin
  sorry
end

end nonnegative_solution_count_l233_233269


namespace problem_solution_l233_233214

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) := (Real.exp x) / (x^2 - m*x + 1)

-- Define the intervals for m
def monotonic_intervals (m : ℝ) : Prop :=
  if m = 0 then 
    ∀ x y : ℝ, x < y → f x 0 < f y 0
  else if 0 < m ∧ m < 2 then 
    (∀ x : ℝ, x < 1 → f x m < f (x + 1) m) ∧
    (∀ x : ℝ, 1 < x ∧ x < m + 1 → f x m > f (x + 1) m) ∧
    (∀ x : ℝ, x > m + 1 → f x m < f (x + 1) m)
  else if -2 < m ∧ m < 0 then 
    (∀ x : ℝ, x < m + 1 → f x m < f (x + 1) m) ∧
    (∀ x : ℝ, m + 1 < x ∧ x < 1 → f x m > f (x + 1) m)
  else 
    false

-- Define the relationship for m in (0, 1/2]
def relationship_m (m : ℝ) : Prop :=
  0 < m ∧ m ≤ 1/2 → let M := f (m + 1) m in
                    let N := m + 1 in
                    M > N

-- Theorem combining both parts
theorem problem_solution (m : ℝ) : 
  monotonic_intervals m ∧ relationship_m m :=
    sorry

end problem_solution_l233_233214


namespace segment_DM_twice_median_BP_l233_233977

open EuclideanGeometry

-- Definitions based on provided conditions
def triangle (A B C : Type) := isTriangle A B C
def square (A B D E : Type) := isSquare A B D E
def median (B P : Type) := isMedian B P

-- Lean 4 Statement
theorem segment_DM_twice_median_BP
  (A B C D E K M P : Type)
  (triangle_ABC : triangle A B C)
  (square_ABDE : square A B D E)
  (square_BCKM : square B C K M)
  (midpoint_P : isMidpoint A C P)
  (D_outputside: isOutside D A B)
  (M_outputside: isOutside M B C)
  (P_on_AC: isOnLine P A C)
  (D_on_AB: isOnSegment D A B)
  (M_on_BC: isOnSegment M B C) : 
  segment_len D M = 2 * segment_len B P := sorry

end segment_DM_twice_median_BP_l233_233977


namespace nonnegative_solution_count_l233_233273

theorem nonnegative_solution_count : 
  ∃! x : ℝ, x^2 = -6 * x ∧ 0 ≤ x := 
begin
  sorry
end

end nonnegative_solution_count_l233_233273


namespace find_number_l233_233674

theorem find_number (x : ℕ) (h1 : x > 7) (h2 : x ≠ 8) : x = 9 := by
  sorry

end find_number_l233_233674


namespace lim_transformation_l233_233867

variable {α β : Type}
variables {f : α → β} {x₀ : α} [HasLimit α β]

-- Hypothesis
def hyp : Prop := limit (λ Δx, (f (x₀ + Δx) - f x₀) / Δx) 0 = k

-- Theorem
theorem lim_transformation (h : hyp) :
  limit (λ Δx, (f (x₀ + 2 * Δx) - f x₀) / Δx) 0 = 2 * k :=
sorry

end lim_transformation_l233_233867


namespace margin_eq_l233_233694

variable (P S : ℝ)
variable (t n : ℕ)

-- Definition of the margin made on the article
def margin (P : ℝ) (n : ℕ) : ℝ := P / n

-- Assumption that selling price includes tax
def selling_price_includes_tax (P : ℝ) (t : ℕ) : ℝ := P * (1 + t / 100)

-- Provide the theorem to prove the relationship between margin and selling price
theorem margin_eq (P S : ℝ) (t n : ℕ) (h : S = selling_price_includes_tax P t) :
  margin P n = S / (n * (1 + t / 100)) := by
  sorry

end margin_eq_l233_233694


namespace order_of_magnitude_l233_233783

noncomputable def a : ℝ := 2 ^ 1.2
noncomputable def b : ℝ := 2 ^ 0.2
noncomputable def c : ℝ := log 5 (4 : ℝ)

theorem order_of_magnitude : a > b ∧ b > c :=
by
  sorry

end order_of_magnitude_l233_233783


namespace polygon_triangle_even_l233_233675

theorem polygon_triangle_even (n m : ℕ) (h : (3 * m - n) % 2 = 0) : (m + n) % 2 = 0 :=
sorry

noncomputable def number_of_distinct_interior_sides (n m : ℕ) : ℕ :=
(3 * m - n) / 2

noncomputable def number_of_distinct_interior_vertices (n m : ℕ) : ℕ :=
(m - n + 2) / 2

end polygon_triangle_even_l233_233675


namespace unique_a_b_l233_233768

-- Define the properties of the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * a * x + b else 7 - 2 * x

-- The function satisfies f(f(x)) = x for all x in its domain
theorem unique_a_b (a b : ℝ) (h : ∀ x : ℝ, f a b (f a b x) = x) : a + b = 13 / 4 :=
sorry

end unique_a_b_l233_233768


namespace dot_product_calc_l233_233847

def vec2 := (ℝ × ℝ)

def a (x : ℝ) : vec2 := (1, x + 1)
def b (x : ℝ) : vec2 := (1 - x, 2)

def dot_product (u v : vec2) : ℝ := u.1 * v.1 + u.2 * v.2

def perpendicular (u v : vec2) : Prop := dot_product u v = 0

theorem dot_product_calc :
  ∀ x : ℝ, 
    perpendicular (a x) (b x) →
    dot_product ((a x).1 + (b x).1, (a x).2 + (b x).2) ((a x).1 - (b x).1, (a x).2 - (b x).2) = -15 :=
by
  intro x h_perp
  have hx : x = -3 := sorry -- This step is derived from the perpendicular condition
  rw [hx]
  -- Calculation of vectors and their dot product
  have a_val : a (-3) = (1, -2) := rfl
  have b_val : b (-3) = (4, 2) := rfl
  have sum : (1 + 4, -2 + 2) = (5, 0) := rfl
  have diff : (1 - 4, -2 - 2) = (-3, -4) := rfl
  rw [a_val, b_val, sum, diff]
  exact rfl -- The dot product calculation 5 * (-3) + 0 * (-4) = -15

end dot_product_calc_l233_233847


namespace greatest_prime_factor_of_341_l233_233603

theorem greatest_prime_factor_of_341 : ∃ p, Prime p ∧ p ∣ 341 ∧ ∀ q, Prime q ∧ q ∣ 341 → q ≤ p :=
by
  let factors := [3, 7, 17]
  have h : 341 = 17 * 3 * 7 := by sorry
  exists 17
  split
  · exact PrimeNat.prime_17
  split
  · exact dvd.intro (3 * 7) rfl
  intro q hpq hq
  have H : q ∈ factors := by sorry
  exact List.mem_le_of_mod_le (PrimeNat.factors_unique H)

end greatest_prime_factor_of_341_l233_233603


namespace f_2015_l233_233206

noncomputable def f : ℝ → ℝ := sorry
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_periodic : ∀ x : ℝ, f (x - 2) = -f x
axiom f_initial_segment : ∀ x : ℝ, (-1 ≤ x ∧ x ≤ 0) → f x = 2^x

theorem f_2015 : f 2015 = 1 / 2 :=
by
  -- Proof goes here
  sorry

end f_2015_l233_233206


namespace greatest_prime_factor_341_l233_233529

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l233_233529


namespace intersection_M_N_l233_233222

-- Definitions of the sets M and N based on the conditions
def M (x : ℝ) : Prop := ∃ (y : ℝ), y = Real.log (x^2 - 3*x - 4)
def N (y : ℝ) : Prop := ∃ (x : ℝ), y = 2^(x - 1)

-- The proof statement
theorem intersection_M_N : { x : ℝ | M x } ∩ { x : ℝ | ∃ y : ℝ, N y ∧ y = Real.log (x^2 - 3*x - 4) } = { x : ℝ | x > 4 } :=
by
  sorry

end intersection_M_N_l233_233222


namespace find_f_4_l233_233332

-- Assume we have a function f and its inverse such that f⁻¹(x) = x^2
variable (f : ℝ → ℝ)
variable (h_inv : ∀ x : ℝ, 0 < x → (Function.LeftInverse (λ x : ℝ, x * x) f))

-- Define the property we need to prove
theorem find_f_4 (h_inv : ∀ x : ℝ, 0 < x → f (x * x) = x) : f 4 = 2 :=
by
  -- Here, we would provide the actual proof
  sorry

end find_f_4_l233_233332


namespace find_number_l233_233489

theorem find_number (x : ℝ) (h : 20 / x = 0.8) : x = 25 := 
by
  sorry

end find_number_l233_233489


namespace greatest_prime_factor_341_l233_233596

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l233_233596


namespace arithmetic_progression_general_formula_geometric_progression_condition_l233_233183

-- Arithmetic progression problem
theorem arithmetic_progression_general_formula :
  ∀ (a : ℕ → ℚ) (S : ℕ → ℚ),
  (∀ n, a n > 0) →
  S 10 = 70 →
  a 1 = 1 →
  (∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2) →
  (∀ n, a n = 1 + (n - 1) * (4/3)) :=
by
  sorry

-- Geometric progression problem
theorem geometric_progression_condition :
  ∀ (a : ℕ → ℚ) (S : ℕ → ℚ),
  (∀ n, a n > 0) →
  a 1 = 1 →
  a 4 = 1/8 →
  (∀ n, S n = a 1 * (1 - (1/2)^n) / (1 - 1/2)) →
  (∃ n, S n > 100 * a n ∧ ∀ m < n, S m ≤ 100 * a m) :=
by
  use 7
  sorry

end arithmetic_progression_general_formula_geometric_progression_condition_l233_233183


namespace password_guess_probability_l233_233648

-- Define the possible characters and numbers
def letters := {'A', 'a', 'B', 'b'}
def digits := {4, 5, 6}

-- Define the total number of combinations
def total_combinations := letters.card * digits.card

-- Define the probability of guessing correctly
def probability := 1 / total_combinations

-- The statement to prove
theorem password_guess_probability :
  probability = 1 / 12 := 
by
  -- proof goes here
  sorry

end password_guess_probability_l233_233648


namespace groups_with_common_student_l233_233477

theorem groups_with_common_student (students : Finset ℕ) (groups : Finset (Finset ℕ))
  (Hstudents : students.card = 1997)
  (Hgroups : ∀ g ∈ groups, g.card = 3)
  (Hgroup_students : ∀ g ∈ groups, g ⊆ students) :
  ∀ selected_groups ⊆ groups, selected_groups.card = 1998 →
  ∃ g1 g2 ∈ selected_groups, (g1 ≠ g2 ∧ (g1 ∩ g2).card = 1) :=
sorry

end groups_with_common_student_l233_233477


namespace eq_solutions_l233_233752

theorem eq_solutions (x : ℂ) : x^6 + 729 = 0 ↔ 
  x = 3*complex.i ∨ x = -3*complex.i ∨ 
  x = (-3*complex.i / 2 + 3*real.sqrt 3 / 2) ∨ 
  x = (-3*complex.i / 2 - 3*real.sqrt 3 / 2) ∨ 
  x = (3*complex.i / 2 - 3*real.sqrt 3 / 2) ∨ 
  x = (3*complex.i / 2 + 3*real.sqrt 3 / 2) := sorry

end eq_solutions_l233_233752


namespace average_weight_of_all_children_l233_233896

theorem average_weight_of_all_children 
    (boys_weight_avg : ℕ)
    (number_of_boys : ℕ)
    (girls_weight_avg : ℕ)
    (number_of_girls : ℕ)
    (tall_boy_weight : ℕ)
    (ht1 : boys_weight_avg = 155)
    (ht2 : number_of_boys = 8)
    (ht3 : girls_weight_avg = 130)
    (ht4 : number_of_girls = 6)
    (ht5 : tall_boy_weight = 175)
    : (boys_weight_avg * (number_of_boys - 1) + tall_boy_weight + girls_weight_avg * number_of_girls) / (number_of_boys + number_of_girls) = 146 :=
by
  sorry

end average_weight_of_all_children_l233_233896


namespace chessboard_queen_placements_l233_233397

theorem chessboard_queen_placements :
  ∃ (n : ℕ), n = 864 ∧
  (∀ (qpos : Finset (Fin 8 × Fin 8)), 
    qpos.card = 3 ∧
    (∀ (q1 q2 q3 : Fin 8 × Fin 8), 
      q1 ∈ qpos ∧ q2 ∈ qpos ∧ q3 ∈ qpos ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 → 
      (q1.1 = q2.1 ∨ q1.2 = q2.2 ∨ abs (q1.1 - q2.1) = abs (q1.2 - q2.2)) ∧ 
      (q1.1 = q3.1 ∨ q1.2 = q3.2 ∨ abs (q1.1 - q3.1) = abs (q1.2 - q3.2)) ∧ 
      (q2.1 = q3.1 ∨ q2.2 = q3.2 ∨ abs (q2.1 - q3.1) = abs (q2.2 - q3.2)))) ↔ n = 864
:=
by
  sorry

end chessboard_queen_placements_l233_233397


namespace number_of_nonnegative_solutions_l233_233236

-- Define the equation x^2 = -6x
def equation (x : ℝ) : Prop := x^2 = -6 * x

-- Define the condition for a nonnegative solution
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Define the main theorem to prove the number of nonnegative solutions
theorem number_of_nonnegative_solutions : 
  {x : ℝ // equation x ∧ nonnegative x}.to_finset.card = 1 :=
sorry

end number_of_nonnegative_solutions_l233_233236


namespace number_of_rectangles_in_partition_l233_233687

-- Definitions for conditions
variables {Square : Type} [Plane Square] 
variables (k l : ℕ) -- given k and l as natural numbers for conditions

-- Statement of the problem
theorem number_of_rectangles_in_partition 
  (h1 : ∀ (vline : Line), vline.parallel_to_vertical → 
        vline.does_not_contain_segment → vline.intersects_exactly k rectangles)
  (h2 : ∀ (hline : Line), hline.parallel_to_horizontal → 
        hline.does_not_contain_segment → hline.intersects_exactly l rectangles)
  : total_rectangles_in_partition Square = k * l :=
sorry

end number_of_rectangles_in_partition_l233_233687


namespace rect_to_polar_example_l233_233127

def rect_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := 
    if x > 0 && y >= 0 then Real.arctan (y / x)
    else if x < 0 then Real.arctan (y / x) + Real.pi
    else if x > 0 && y < 0 then Real.arctan (y / x) + 2 * Real.pi
    else if x = 0 && y > 0 then Real.pi / 2
    else if x = 0 && y < 0 then 3 * Real.pi / 2
    else 0 -- We can assume the origin case doesn't happen since r must be > 0
  (r, θ)

theorem rect_to_polar_example :
  rect_to_polar (2 * Real.sqrt 2) (-2) = (2 * Real.sqrt 3, 7 * Real.pi / 4) :=
by
  sorry

end rect_to_polar_example_l233_233127


namespace coordinates_of_Q_l233_233359

variables {X Y Z G H F Q : Type} [AffineSpace X] [AffineSpace Y] [AffineSpace Z]
variables [InnerProductSpace ℝ X] [InnerProductSpace ℝ Y] [InnerProductSpace ℝ Z]

def lying_on_segment (a b : X) (r s : ℝ) (P : X) : Prop :=
  ∃ t, t ≥ 0 ∧ t ≤ 1 ∧ P = t • a + (1 - t) • b

def triangle_XYZ_conditions (X Y Z G H F : X) : Prop :=
  ∃ (rG rH rF : ℝ), rG = 3/4 ∧ rH = 1/4 ∧ rF = 2/3 ∧
    lying_on_segment X Z rG (1 - rG) G ∧
    lying_on_segment X Y rH (1 - rH) H ∧
    lying_on_segment Y Z rF (1 - rF) F

theorem coordinates_of_Q (X Y Z G H F Q : X)
  (h : triangle_XYZ_conditions X Y Z G H F) :
  ∃ x y z : ℝ, Q = x • X + y • Y + z • Z ∧ x = 5/18 ∧ y = 1/6 ∧ z = 4/9 := 
sorry

end coordinates_of_Q_l233_233359


namespace find_f_60_l233_233441

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition.

axiom functional_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom f_48 : f 48 = 36

theorem find_f_60 : f 60 = 28.8 := by 
  sorry

end find_f_60_l233_233441


namespace Cyclic_Quadrilateral_l233_233797

open EuclideanGeometry

-- Definitions for the 2D plane (ℝ²), points, lines, and concepts in Euclidean geometry
def Triangle (A B C : Point) :=
  ¬Collinear A B C

def Incenter (A B C I : Point) :=
  ∀ P, AngleBisector A B C P ↔ AngleBisector B C A P ↔ AngleBisector C A B P ∧ Concur A B C I

def PerpendicularBisector (X Y Z : Point) :=
  ∀ P, ∃ M, Midpoint M Y Z ∧ Perpendicular M X P

def AngleBisector (A B C D : Point) :=
  ∃ P, (∠ P B C = ∠ A B P)

def Cyclic (A B C D : Point) :=
  InscribedAngle A B C D ∧ InscribedAngle C D A B

theorem Cyclic_Quadrilateral (A B C I S T : Point) (hT : Triangle A B C)
  (hIinc : Incenter A B C I) 
  (hS : AngleBisector A (A + B) (A + C) S ∧ PerpendicularBisector B C S)
  (hT : AngleBisector B (B + C) (B + A) T ∧ PerpendicularBisector B C T) :
  Cyclic C I S T :=
by sorry

end Cyclic_Quadrilateral_l233_233797


namespace quadrilateral_area_l233_233140

-- Define the basic geometrical setup
variables {A B C D : Type*}
variables [DistinctPoints : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D]
variable (s : ℝ) (E F : Type*)

-- Assumptions about the equilateral triangles ABC and ABD
def equilateral_triangle (A B C : Type*) (s : ℝ) : Prop :=
  EuclideanGeometry.EquilateralTriangle A B C s

-- Distinct points A, B, C, and D
variable [DistinctABCD : DistinctPoints]

-- Setup of the problem
axiom ABC_eq_tri : equilateral_triangle A B C 10
axiom ABD_eq_tri : equilateral_triangle A B D 10

-- Assumptions on the locations of E and F
axiom EA_len : distance A E = 8
axiom EB_len : distance B E = 3
axiom FD_len : distance D F = 8
axiom FB_len : distance B F = 3

noncomputable def area_AEDF := area AEDF

-- The final theorem statement
theorem quadrilateral_area : area_AEDF = (91 * real.sqrt 3) / 4 := 
by {
  sorry
}

end quadrilateral_area_l233_233140


namespace nonnegative_solution_count_nonnegative_solution_count_unique_l233_233281

theorem nonnegative_solution_count (x : ℝ) :
  (x^2 = -6 * x) → (x ≥ 0) → x = 0 :=
by
  sorry

-- Prove that the number of nonnegative solutions is 1
theorem nonnegative_solution_count_unique :
  {x : ℝ // x ≥ 0 ∧ x^2 = -6 * x}.to_finset.card = 1 :=
by
  sorry

end nonnegative_solution_count_nonnegative_solution_count_unique_l233_233281


namespace position_of_UKIMC_l233_233003

open Finset

def dictionary_order_position {α : Type*} [DecidableEq α] [LinearOrder α] (s : Finset α) (l : List α) : Option ℕ :=
  let permutations := s.perms.sort
  permutations.find_index (λ p, p = l)

theorem position_of_UKIMC : dictionary_order_position (finset.cons 'U' (finset.cons 'K' (finset.cons 'M' (finset.cons 'I' (finset.cons 'C' finset.empty) sorry) sorry) sorry) sorry) ['U', 'K', 'I', 'M', 'C'] = some 110 :=
sorry

end position_of_UKIMC_l233_233003


namespace rationalize_result_l233_233413

noncomputable def rationalize_denominator (x y : ℚ) (sqrt_c : ℚ) : ℚ :=
  let numerator := x + sqrt_c
  let denominator := y - sqrt_c
  (numerator * (y + sqrt_c)) / (denominator * (y + sqrt_c))

theorem rationalize_result :
  let sqrt_5 := Real.sqrt 5
  let expr := rationalize_denominator 2 3 sqrt_5
  let A := 11 / 4
  let B := 5 / 4
  let C := 5
  expr = A + B * sqrt_5 ∧ A * B * C = 275 / 16 := 
sorry

end rationalize_result_l233_233413


namespace set_intersection_l233_233193

noncomputable def A := { x : ℝ | x > 3 }
noncomputable def B := { x : ℤ | 2 < x ∧ x < 6 }

theorem set_intersection :
  A ∩ B = {4, 5} :=
sorry

end set_intersection_l233_233193


namespace number_of_nonnegative_solutions_l233_233238

-- Define the equation x^2 = -6x
def equation (x : ℝ) : Prop := x^2 = -6 * x

-- Define the condition for a nonnegative solution
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Define the main theorem to prove the number of nonnegative solutions
theorem number_of_nonnegative_solutions : 
  {x : ℝ // equation x ∧ nonnegative x}.to_finset.card = 1 :=
sorry

end number_of_nonnegative_solutions_l233_233238


namespace find_radius_of_large_ball_l233_233452

-- Define the volumes of small and large balls given the radius
def volume_sphere (r : ℝ) : ℝ := (4/3) * Real.pi * (r^3)

-- The problem statement as Lean 4 type proposition
theorem find_radius_of_large_ball (r_small : ℝ) (n : ℕ) (R_large : ℝ) 
  (h_small_ball_radius : r_small = 3) 
  (h_num_small_balls : n = 9)
  (h_volume_equality : n * volume_sphere r_small = volume_sphere R_large) :
  R_large = 9 :=
by
  sorry -- skipping the proof, as per the instructions.

end find_radius_of_large_ball_l233_233452


namespace vertical_asymptote_lemma_l233_233771

def has_v_asymptote (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ¬ (∃ L : ℝ, tendsto f (𝓝 a) (𝓝 L))

def vertical_asymptote_at (x : ℝ) : Prop :=
  x = 3 / 2

theorem vertical_asymptote_lemma : ∀ x : ℝ,
  vertical_asymptote_at x ↔ has_v_asymptote (λ x, (x + 3) / (4*x - 6)) x := 
by {
  sorry
}

end vertical_asymptote_lemma_l233_233771


namespace twenty_is_80_percent_of_what_number_l233_233491

theorem twenty_is_80_percent_of_what_number : ∃ y : ℕ, (20 : ℚ) / y = 4 / 5 ∧ y = 25 := by
  sorry

end twenty_is_80_percent_of_what_number_l233_233491


namespace value_of_m_l233_233177

variable (a m : ℝ)
variable (h1 : a > 0)
variable (h2 : -a*m^2 + 2*a*m + 3 = 3)
variable (h3 : m ≠ 0)

theorem value_of_m : m = 2 :=
by
  sorry

end value_of_m_l233_233177


namespace twenty_is_80_percent_of_what_number_l233_233490

theorem twenty_is_80_percent_of_what_number : ∃ y : ℕ, (20 : ℚ) / y = 4 / 5 ∧ y = 25 := by
  sorry

end twenty_is_80_percent_of_what_number_l233_233490


namespace symmetric_trapezoids_in_rhombus_l233_233645

theorem symmetric_trapezoids_in_rhombus:
  ∀ (K L M N : ℝ) (h1 : K ≠ L) (h2 : L ≠ M) (h3 : M ≠ N) (h4 : N ≠ K)
  (α: ℝ) (h_angle_range : 60 < α ∧ α < 90)
  (h_symmetric : symmetric_trapezoid (K, L, M, N)),
  number_of_max_area_trapezoids (K, L, M, N, α) = 6 :=
begin
  sorry
end

end symmetric_trapezoids_in_rhombus_l233_233645


namespace geometric_sequence_frac_general_term_formula_sum_first_n_terms_l233_233184

noncomputable def seq_a : ℕ → ℚ
| 1     := 1 / 2
| (n+2) := seq_a (n+1) / (2 - seq_a (n+1))

def seq_c (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  1 / a n - 1

-- Define the sequence b_n
def seq_b (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (2 * n - 1) / (a n)

-- Define the sum of the first n terms S_n
def sum_b (b : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ k in finset.range n, b (k+1)

theorem geometric_sequence_frac {a : ℕ → ℚ} (n : ℕ) (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n ≥ 2, a n = a (n - 1) / (2 - a (n - 1))) :
  ∃ r : ℚ, ∃ T : ℕ → ℚ, T 1 = 1 ∧ T n = r ^ (n - 1) :=
sorry

theorem general_term_formula {a : ℕ → ℚ} (n : ℕ) (h₁ : a 1 = 1 / 2)
  (h₂ : ∀ n ≥ 2, a n = a (n - 1) / (2 - a (n - 1))) :
  a n = 1 / (2^(n-1) + 1)
:= sorry

theorem sum_first_n_terms (n : ℕ) (h₁ : seq_a 1 = 1 / 2)
  (h₂ : ∀ n ≥ 2, seq_a n = seq_a (n - 1) / (2 - seq_a (n - 1))) :
  sum_b (seq_b seq_a) n = (2 * n - 3) * 2^n + 3 + n^2
:= sorry

end geometric_sequence_frac_general_term_formula_sum_first_n_terms_l233_233184


namespace domain_all_real_numbers_l233_233753

theorem domain_all_real_numbers (k : ℝ) :
  (∀ x : ℝ, -7 * x^2 - 4 * x + k ≠ 0) ↔ k < -4 / 7 := by
  sorry

end domain_all_real_numbers_l233_233753


namespace y_n_sq_eq_3_x_n_sq_add_1_l233_233204

def x : ℕ → ℤ
| 0       => 0
| 1       => 1
| (n + 1) => 4 * x n - x (n - 1)

def y : ℕ → ℤ
| 0       => 1
| 1       => 2
| (n + 1) => 4 * y n - y (n - 1)

theorem y_n_sq_eq_3_x_n_sq_add_1 (n : ℕ) : y n ^ 2 = 3 * (x n) ^ 2 + 1 :=
sorry

end y_n_sq_eq_3_x_n_sq_add_1_l233_233204


namespace right_triangle_legs_sum_squares_area_l233_233153

theorem right_triangle_legs_sum_squares_area:
  ∀ (a b c : ℝ), 
  (0 < a) → (0 < b) → (0 < c) → 
  (a^2 + b^2 = c^2) → 
  (1 / 2 * a * b = 24) → 
  (a^2 + b^2 = 48) → 
  (a = 2 * Real.sqrt 6 ∧ b = 2 * Real.sqrt 6 ∧ c = 4 * Real.sqrt 3) := 
by
  sorry

end right_triangle_legs_sum_squares_area_l233_233153


namespace parabola_distance_ratio_l233_233836

open Real

theorem parabola_distance_ratio (p : ℝ) (M N : ℝ × ℝ)
  (h1 : p = 4)
  (h2 : M.snd ^ 2 = 2 * p * M.fst)
  (h3 : N.snd ^ 2 = 2 * p * N.fst)
  (h4 : (M.snd - 2 * N.snd) * (M.snd + 2 * N.snd) = 48) :
  |M.fst + 2| = 4 * |N.fst + 2| := sorry

end parabola_distance_ratio_l233_233836


namespace candles_shared_equally_l233_233698

theorem candles_shared_equally :
  ∀ (Aniyah Ambika Bree Caleb : ℕ),
  Aniyah = 6 * Ambika → Ambika = 4 → Bree = 0 → Caleb = 0 →
  (Aniyah + Ambika + Bree + Caleb) / 4 = 7 :=
by
  intros Aniyah Ambika Bree Caleb h1 h2 h3 h4
  sorry

end candles_shared_equally_l233_233698


namespace a²_minus_b²_l233_233322

theorem a²_minus_b² (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := 
by
  sorry

end a²_minus_b²_l233_233322


namespace cannot_transform_to_two_red_points_l233_233500

-- Define the initial setup and operations
inductive Color
| blue
| red

def initial_points : list Color := [Color.blue, Color.blue]

def operation_a (points : list Color) : list Color → list Color := 
-- (Inserts a new red point and changes the colors of its two neighbors to the opposite colors)
sorry

def operation_b (points : list Color) : list Color → list Color := 
-- (Removes a red point if there are at least three points and changes the colors of its two neighbors to the opposite colors)
sorry

-- Define the goal statement:
def goal_reached (points : list Color) : Prop :=
points = [Color.red, Color.red]

-- We need to prove:
theorem cannot_transform_to_two_red_points : ¬ (∃ points, 
(start = initial_points → ∃ (operation_sequence: list (list Color → list Color)), 
(list.foldl (λ pts op, op pts) initial_points operation_sequence = points) ∧ 
goal_reached points)) := 
sorry

end cannot_transform_to_two_red_points_l233_233500


namespace variance_of_data_set_l233_233796

theorem variance_of_data_set (m : ℝ) (h_mean : (6 + 7 + 8 + 9 + m) / 5 = 8) :
    (1/5) * ((6-8)^2 + (7-8)^2 + (8-8)^2 + (9-8)^2 + (m-8)^2) = 2 := 
sorry

end variance_of_data_set_l233_233796


namespace five_digit_number_use_all_digits_max_sum_l233_233499

theorem five_digit_number_use_all_digits_max_sum :
  ∃ A B : ℕ,
  (digits A ∪ digits B = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (A % 100 = 0) ∧
  (B % 100 = 0) ∧ 
  (A + B = 183900) ∧
  ((A = 96400) ∨ (A = 87500)) :=
sorry

end five_digit_number_use_all_digits_max_sum_l233_233499


namespace millet_exceeds_half_seeds_l233_233396

noncomputable def total_seeds (n : ℕ) : ℝ := n + 1
noncomputable def millet_after_birds (millet : ℝ) : ℝ := 0.5 * millet
noncomputable def millet_each_day (day : ℕ) : ℕ → ℝ
| 0 => 0.4
| (n + 1) => millet_after_birds (millet_each_day n) + 0.4

theorem millet_exceeds_half_seeds : ∃ day : ℕ, 2 * millet_each_day day > total_seeds day := by
  have h3 : millet_each_day 3 = 0.9 := by
    rw [millet_each_day, millet_each_day, millet_each_day, millet_each_day]
    norm_num
  use 3
  calc
    2 * 0.9 > 2 := by norm_num


end millet_exceeds_half_seeds_l233_233396


namespace soda_cost_132_cents_l233_233016

theorem soda_cost_132_cents
  (b s : ℕ)
  (h1 : 3 * b + 2 * s + 30 = 510)
  (h2 : 2 * b + 3 * s = 540) 
  : s = 132 :=
by
  sorry

end soda_cost_132_cents_l233_233016


namespace percentage_difference_height_l233_233874

-- Define the heights of persons B, A, and C
variables (H_B H_A H_C : ℝ)

-- Condition: Person A's height is 30% less than person B's height
def person_A_height : Prop := H_A = 0.70 * H_B

-- Condition: Person C's height is 20% more than person A's height
def person_C_height : Prop := H_C = 1.20 * H_A

-- The proof problem: Prove that the percentage difference between H_B and H_C is 16%
theorem percentage_difference_height (h1 : person_A_height H_B H_A) (h2 : person_C_height H_A H_C) :
  ((H_B - H_C) / H_B) * 100 = 16 :=
by
  sorry

end percentage_difference_height_l233_233874


namespace no_real_solution_l233_233988

theorem no_real_solution :
  ¬ ∃ x : ℝ, (3 * x ^ 2 / (x - 2) - (5 * x + 4) / 4 + (10 - 9 * x) / (x - 2) + 2 = 0) :=
sorry

end no_real_solution_l233_233988


namespace find_f_pi_over_12_l233_233881

-- Define the function transformations and conditions
def f (x : Real) : Real := Real.sin (2 * x - Real.pi / 2) - 1

theorem find_f_pi_over_12 : 
  f (Real.pi / 12) = - (Real.sqrt 3 + 2) / 2 :=
by
  sorry

end find_f_pi_over_12_l233_233881


namespace greatest_prime_factor_341_l233_233521

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l233_233521


namespace cups_of_sugar_l233_233969

theorem cups_of_sugar (flour_total flour_added sugar : ℕ) (h₁ : flour_total = 10) (h₂ : flour_added = 7) (h₃ : flour_total - flour_added = sugar + 1) :
  sugar = 2 :=
by
  sorry

end cups_of_sugar_l233_233969


namespace nonnegative_solution_count_l233_233275

theorem nonnegative_solution_count : 
  ∃! x : ℝ, x^2 = -6 * x ∧ 0 ≤ x := 
begin
  sorry
end

end nonnegative_solution_count_l233_233275


namespace find_natural_numbers_l233_233767

def f (m : ℕ) : ℕ :=
  ∑ d in finset.range m, if nat.coprime d m then d else 0

axiom euler_totient_eq (m : ℕ) : 2 * f m = m * nat.totient m

theorem find_natural_numbers :
  {n : ℕ // ∃ k ℓ : ℕ, f (n^k) = n^ℓ} =
  {2, 3, 4, 6} := sorry

end find_natural_numbers_l233_233767


namespace transform_quadratic_roots_l233_233963

-- Define the conditions of the quadratic polynomial and Vieta's formulas
variables {p q r u v : ℝ}
def quadratic (p q r : ℝ) := p ≠ 0 ∧ (u + v = -q / p) ∧ (u * v = r / p)

-- Prove that the equation with roots 2pu + q and 2pv + q is y^2 - q/(4p)y + r = 0
theorem transform_quadratic_roots {p q r u v : ℝ} (h : quadratic p q r) :
  ∃ y : polynomial ℝ, (polynomial.X^2 - (q / (4 * p)) * polynomial.X + polynomial.C r) = y :=
sorry

end transform_quadratic_roots_l233_233963


namespace greatest_prime_factor_of_341_l233_233569

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l233_233569


namespace volunteering_ways_l233_233693

theorem volunteering_ways (s1 s2 s3 s4 : Bool) :
  (∀ (s : Bool), s = true ∨ s = false) →
  ∃ (n : ℕ), n = 14 ∧
  let ways := if (s1 = s2 ∧ s2 = s3 ∧ s3 ≠ s4) ∨ (s1 = s2 ∧ s2 ≠ s3 ∧ s3 = s4) ∨ (s1 ≠ s2 ∧ s2 ≠ s3 ∧ s3 ≠ s4)
              then 14
              else 0
  in ways = n :=
by
  sorry

end volunteering_ways_l233_233693


namespace greatest_prime_factor_341_l233_233525

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l233_233525


namespace scale_of_enlargement_l233_233734

theorem scale_of_enlargement (G : Type) [Graph G] (f : G → G) 
  (h : ∀ (e : edge G), size (f e) = 4 * size e) : 
  enlargement_scale f = 4 :=
sorry

end scale_of_enlargement_l233_233734


namespace minimum_distance_sum_tetrahedron_l233_233448

-- Define a regular tetrahedron with a given edge length
structure Tetrahedron :=
  (A B C D : ℝ × ℝ × ℝ)
  (edge_length : ℝ)
  (regular : dist A B = edge_length ∧ dist A C = edge_length ∧ dist A D = edge_length ∧
             dist B C = edge_length ∧ dist B D = edge_length ∧ dist C D = edge_length)

-- Define the centroid of the tetrahedron
def centroid (T : Tetrahedron) : ℝ × ℝ × ℝ :=
  let (x1, y1, z1) := T.A in
  let (x2, y2, z2) := T.B in
  let (x3, y3, z3) := T.C in
  let (x4, y4, z4) := T.D in
  ((x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4, (z1 + z2 + z3 + z4) / 4)

-- Define the sum of distances from a point to all vertices of the tetrahedron
def sum_of_distances (P : ℝ × ℝ × ℝ) (T : Tetrahedron) : ℝ :=
  dist P T.A + dist P T.B + dist P T.C + dist P T.D

-- The main theorem stating the minimum value of sum of the distances from a point to the vertices is √6
theorem minimum_distance_sum_tetrahedron (T : Tetrahedron) (h : T.edge_length = 1) :
  ∃ P : ℝ × ℝ × ℝ, sum_of_distances P T = sqrt 6 :=
begin
  let O := centroid T,
  use O,
  sorry
end

end minimum_distance_sum_tetrahedron_l233_233448


namespace remaining_fish_count_l233_233132

def initial_fish_counts : Type := (ℕ, ℕ, ℕ, ℕ)
def sold_fish_counts : Type := (ℕ, ℕ, ℕ, ℕ)

-- Define the initial number of fish
def initial_counts : initial_fish_counts := (94, 76, 89, 58)

-- Define the number of fish sold
def sold_counts : sold_fish_counts := (30, 48, 17, 24)

theorem remaining_fish_count : 
  let (guppies, angelfish, tiger_sharks, oscar_fish) := initial_counts in
  let (sold_guppies, sold_angelfish, sold_tiger_sharks, sold_oscar_fish) := sold_counts in
  guppies - sold_guppies + (angelfish - sold_angelfish) + (tiger_sharks - sold_tiger_sharks) + (oscar_fish - sold_oscar_fish) = 198 :=
by
  sorry

end remaining_fish_count_l233_233132


namespace shaded_area_l233_233503

theorem shaded_area (r1 r2 : ℝ) (h1 : r2 = 3 * r1) (h2 : r1 = 2) : 
  π * (r2 ^ 2) - π * (r1 ^ 2) = 32 * π :=
by
  sorry

end shaded_area_l233_233503


namespace analytical_expression_negative_analytical_expression_solve_fx_eq_2x_l233_233186

variable {f : ℝ → ℝ}

-- Conditions
def odd_function (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f(-x) = -f(x)

def domain_of_f (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, x ∈ ℝ

def f_for_positive (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → f(x) = x^2 - 3

-- Proof goals
theorem analytical_expression_negative (h1 : odd_function f) (h2 : domain_of_f f) (h3 : f_for_positive f) :
  ∀ x : ℝ, x < 0 → f(x) = -x^2 + 3 :=
sorry

theorem analytical_expression (h1 : odd_function f) (h2 : domain_of_f f) (h3 : f_for_positive f) :
  ∀ x : ℝ, f(x) = 
    if x < 0 then -x^2 + 3 
    else if x = 0 then 0
    else x^2 - 3 :=
sorry

theorem solve_fx_eq_2x (h1 : odd_function f) (h2 : domain_of_f f) (h3 : f_for_positive f) :
  { x : ℝ | f(x) = 2 * x } = {-3, 0, 3} :=
sorry

end analytical_expression_negative_analytical_expression_solve_fx_eq_2x_l233_233186


namespace cube_root_of_64_l233_233999

theorem cube_root_of_64 : ∃ a : ℝ, a^3 = 64 ∧ a = 4 :=
by
  use 4
  split
  case left =>
    show 4^3 = 64
    calc
      4^3 = 64 : by norm_num
  case right =>
    show 4 = 4
    rfl

end cube_root_of_64_l233_233999


namespace daragh_initial_bears_l233_233718

variables (initial_bears eden_initial_bears eden_final_bears favorite_bears shared_bears_per_sister : ℕ)
variables (sisters : ℕ)

-- Given conditions
axiom h1 : eden_initial_bears = 10
axiom h2 : eden_final_bears = 14
axiom h3 : favorite_bears = 8
axiom h4 : sisters = 3

-- Derived condition
axiom h5 : shared_bears_per_sister = eden_final_bears - eden_initial_bears
axiom h6 : initial_bears = favorite_bears + (shared_bears_per_sister * sisters)

-- The theorem to prove
theorem daragh_initial_bears : initial_bears = 20 :=
by
  -- Insert proof here
  sorry

end daragh_initial_bears_l233_233718


namespace integer_values_of_a_l233_233151

theorem integer_values_of_a (x : ℤ) (a : ℤ)
  (h : x^3 + 3*x^2 + a*x + 11 = 0) :
  a = -155 ∨ a = -15 ∨ a = 13 ∨ a = 87 :=
sorry

end integer_values_of_a_l233_233151


namespace total_points_l233_233722

theorem total_points (darius_score marius_score matt_score total_points : ℕ) 
    (h1 : darius_score = 10) 
    (h2 : marius_score = darius_score + 3) 
    (h3 : matt_score = darius_score + 5) 
    (h4 : total_points = darius_score + marius_score + matt_score) : 
    total_points = 38 :=
by sorry

end total_points_l233_233722


namespace project_budget_status_l233_233715

theorem project_budget_status:
  (let project_A_budget := 42000
       project_A_duration := 12
       project_A_actual_spent := 23700
       project_A_months := 8
       project_B_budget := 66000
       project_B_duration := 12
       project_B_actual_spent := 39000
       project_B_months := 7
       project_C_budget := 30000
       project_C_duration := 12
       project_C_actual_spent := 14000
       project_C_months := 6
       total_budgets := project_A_budget + project_B_budget + project_C_budget
       total_actual_spent := project_A_actual_spent + project_B_actual_spent + project_C_actual_spent) in
   (let expected_spent (budget duration months) := months * (budget / duration) in
      let under_over_budget (expected_spent actual_spent) := expected_spent - actual_spent in
      let project_A_status := under_over_budget (expected_spent project_A_budget project_A_duration project_A_months) project_A_actual_spent in
      let project_B_status := under_over_budget (expected_spent project_B_budget project_B_duration project_B_months) project_B_actual_spent in
      let project_C_status := under_over_budget (expected_spent project_C_budget project_C_duration project_C_months) project_C_actual_spent in
   project_A_status = 4300 ∧
   project_B_status = -500 ∧
   project_C_status = 1000 ∧
   (total_actual_spent / total_budgets) * 100 = 55.58) :=
by
  sorry

end project_budget_status_l233_233715


namespace parallel_lines_l233_233925

open set

noncomputable def midpoint {α : Type*} [field α] [add_comm_group α] [vector_space α] 
  (p₁ p₂ : α) : α := (p₁ + p₂) / 2

theorem parallel_lines
  {α : Type*} [field α] [add_comm_group α] [vector_space α] 
  (A B C E F K L N M : α)
  (hM : M = midpoint B C)
  (hE : ∃ (λ p, p ∈ line A B), E)
  (hF : ∃ (λ p, p ∈ line A C), F)
  (hK : ∃ (p : α), p ∈ line B F ∧ p ∈ line C E ∧ p = K)
  (hL1 : L ∈ parallel (line C E) (line A B))
  (hL2 : L ∈ parallel (line B C) (line C E))
  (hN : N ∈ line A M ∧ N ∈ line C L) 
  : same_direction (line K N) (line F L) :=
sorry

end parallel_lines_l233_233925


namespace quadrilateral_area_l233_233683

noncomputable def det_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * real.abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem quadrilateral_area :
  let A := (1, 2)
  let B := (1, -1)
  let C := (4, -1)
  let D := (7, 8)
  det_area 1 2 1 (-1) 4 (-1) + det_area 1 2 4 (-1) 7 8 = 22.5 := sorry

end quadrilateral_area_l233_233683


namespace percent_divisible_by_six_up_to_120_l233_233629

theorem percent_divisible_by_six_up_to_120 : 
  let total_numbers := 120
  let divisible_by_six := total_numbers / 6
  let percentage := (divisible_by_six * 100) / total_numbers
  percentage = 50 / 3 := sorry

end percent_divisible_by_six_up_to_120_l233_233629


namespace nonnegative_solutions_eq1_l233_233250

theorem nonnegative_solutions_eq1 : (∃ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x^2 = -6 * x → x = 0) := by
  sorry

end nonnegative_solutions_eq1_l233_233250


namespace nonneg_solutions_count_l233_233258

theorem nonneg_solutions_count :
  {x : ℝ | x^2 + 6 * x = 0 ∧ 0 ≤ x}.card = 1 :=
by
  sorry

end nonneg_solutions_count_l233_233258


namespace intersection_M_N_l233_233221

def M (x : ℝ) : Prop := (x - 3) / (x + 1) > 0
def N (x : ℝ) : Prop := 3 * x + 2 > 0

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 3 < x} :=
by
  sorry

end intersection_M_N_l233_233221


namespace largest_integer_before_log2_3010_l233_233613

theorem largest_integer_before_log2_3010 :
  (⌊ ∑ x in Finset.range 3010, log 2 (x + 1) - log 2 x ⌋) = 11 := 
begin
  sorry
end

end largest_integer_before_log2_3010_l233_233613


namespace imaginary_part_of_complex_fraction_l233_233942

noncomputable def imaginary_unit : ℂ := complex.I

theorem imaginary_part_of_complex_fraction : 
  (complex.im ((5 * imaginary_unit) / (2 - imaginary_unit)) = 2) := 
by
  -- This is the statement.
  -- The proof is omitted here.
  sorry

end imaginary_part_of_complex_fraction_l233_233942


namespace todd_ingredients_l233_233010

variables (B R N : ℕ) (P A : ℝ) (I : ℝ)

def todd_problem (B R N : ℕ) (P A I : ℝ) : Prop := 
  B = 100 ∧ 
  R = 110 ∧ 
  N = 200 ∧ 
  P = 0.75 ∧ 
  A = 65 ∧ 
  I = 25

theorem todd_ingredients :
  todd_problem 100 110 200 0.75 65 25 :=
by sorry

end todd_ingredients_l233_233010


namespace greatest_prime_factor_341_l233_233600

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l233_233600


namespace real_number_solution_l233_233024

theorem real_number_solution : ∃ x : ℝ, x = 3 + 6 / (1 + 6 / x) ∧ x = 3 * Real.sqrt 2 :=
by
  sorry

end real_number_solution_l233_233024


namespace problem_f_f_3_eq_13_over_9_l233_233823

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then x^2 + 1 else 2 / x

theorem problem_f_f_3_eq_13_over_9 : f (f 3) = 13 / 9 :=
by
  sorry

end problem_f_f_3_eq_13_over_9_l233_233823


namespace tangent_line_equation_l233_233155
-- Import the necessary Lean library

-- Define the function and the point
def curve (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

theorem tangent_line_equation : 
  let x0 := 0
  let y0 := 2
  let tangent_slope := -1
  let tangent_line := λ x y : ℝ, x + y - 2
  ∀ x y, 
  HasDerivAt curve tangent_slope x0 ∧
  curve x0 = y0 →
  tangent_line (x0 + 1) (y0 - 1) = 0 := 
by
  sorry

end tangent_line_equation_l233_233155


namespace sample_size_is_40_l233_233905

theorem sample_size_is_40 (total_students : ℕ) (sample_students : ℕ) (h1 : total_students = 240) (h2 : sample_students = 40) : sample_students = 40 :=
by
  sorry

end sample_size_is_40_l233_233905


namespace pages_with_same_units_digit_l233_233050

theorem pages_with_same_units_digit :
  let original_pages := Finset.range 74
      new_page (x : ℕ) := 74 - x
      share_units_digit (x : ℕ) := x % 10 = new_page x % 10
  in Finset.filter share_units_digit original_pages = 15 :=
begin
  sorry
end

end pages_with_same_units_digit_l233_233050


namespace partial_fraction_sum_l233_233703

theorem partial_fraction_sum :
  (∃ A B C D E : ℝ, 
    (∀ x : ℝ, x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -5 → 
    (1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5))) ∧
    (A + B + C + D + E = 1 / 30)) :=
sorry

end partial_fraction_sum_l233_233703


namespace greatest_prime_factor_341_l233_233528

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l233_233528


namespace usamo_2009_p1_l233_233351

open Classical

noncomputable def problem_statement 
  {O P A B C D E F : Point}
  (h1 : Circle O ∩ Circle P = {A, B})
  (h2 : Collinear {O, C, D})
  (h3 : Collinear {P, E, F})
  (h4 : Concyclic {C, E, D, F}) : 
  Prop :=
  ∃ K : Point, 
    OnLine K (Line AB) ∧ 
    OnLine K (Line CD) ∧ 
    OnLine K (Line EF) ∧ 
    OnCircle (Circumcenter {C, E, D, F}) (Line AB)

theorem usamo_2009_p1 
  {O P A B C D E F : Point}
  (h1 : Circle O ∩ Circle P = {A, B})
  (h2 : Collinear {O, C, D})
  (h3 : Collinear {P, E, F})
  (h4 : Concyclic {C, E, D, F}) :
  problem_statement h1 h2 h3 h4 :=
sorry

end usamo_2009_p1_l233_233351


namespace greatest_prime_factor_341_l233_233585

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l233_233585


namespace greatest_prime_factor_of_341_is_17_l233_233554

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l233_233554


namespace greatest_prime_factor_of_341_l233_233542

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l233_233542


namespace greatest_prime_factor_341_l233_233601

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l233_233601


namespace sequence_a_n_eq_5050_l233_233885

theorem sequence_a_n_eq_5050 (a : ℕ → ℕ) (h1 : ∀ n > 1, (n - 1) * a n = (n + 1) * a (n - 1)) (h2 : a 1 = 1) : 
  a 100 = 5050 := 
by
  sorry

end sequence_a_n_eq_5050_l233_233885


namespace greatest_prime_factor_of_341_l233_233550

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l233_233550


namespace total_points_correct_l233_233727

-- Define the scores
def Marius (Darius : ℕ) : ℕ := Darius + 3
def Matt (Darius : ℕ) : ℕ := Darius + 5

-- Define the total points function
def total_points (Darius : ℕ) : ℕ :=
  Darius + Marius Darius + Matt Darius

-- Specific value for Darius's score
def Darius_score : ℕ := 10

-- The theorem that proves the total score is 38 given Darius's score
theorem total_points_correct :
  total_points Darius_score = 38 :=
by
  sorry

end total_points_correct_l233_233727


namespace greatest_prime_factor_of_341_l233_233549

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l233_233549


namespace sum_series_l233_233710

theorem sum_series : (∑ i in finset.range 200, (-1)^i * (i + 1)) = 100 := 
by 
  sorry

end sum_series_l233_233710


namespace nonnegative_solution_count_nonnegative_solution_count_unique_l233_233279

theorem nonnegative_solution_count (x : ℝ) :
  (x^2 = -6 * x) → (x ≥ 0) → x = 0 :=
by
  sorry

-- Prove that the number of nonnegative solutions is 1
theorem nonnegative_solution_count_unique :
  {x : ℝ // x ≥ 0 ∧ x^2 = -6 * x}.to_finset.card = 1 :=
by
  sorry

end nonnegative_solution_count_nonnegative_solution_count_unique_l233_233279


namespace joan_remaining_oranges_l233_233363

def total_oranges_joan_picked : ℕ := 37
def oranges_sara_sold : ℕ := 10

theorem joan_remaining_oranges : total_oranges_joan_picked - oranges_sara_sold = 27 := by
  sorry

end joan_remaining_oranges_l233_233363


namespace only_n_for_integer_a_minus_b_l233_233164

theorem only_n_for_integer_a_minus_b (n : ℤ) (A B : ℝ) 
  (hA : A = real.sqrt (n^2 + 24)) (hB : B = real.sqrt (n^2 - 9))
  (hAB_int : ∃ m : ℤ, A - B = m) : n = 5 :=
sorry

end only_n_for_integer_a_minus_b_l233_233164


namespace evaTotalMarksCorrect_l233_233743

-- Definition of marks scored by Eva in each subject across semesters
def evaMathsMarksSecondSemester : Nat := 80
def evaArtsMarksSecondSemester : Nat := 90
def evaScienceMarksSecondSemester : Nat := 90

def evaMathsMarksFirstSemester : Nat := evaMathsMarksSecondSemester + 10
def evaArtsMarksFirstSemester : Nat := evaArtsMarksSecondSemester - 15
def evaScienceMarksFirstSemester : Nat := evaScienceMarksSecondSemester - (evaScienceMarksSecondSemester / 3)

-- Total marks in each semester
def totalMarksFirstSemester : Nat := evaMathsMarksFirstSemester + evaArtsMarksFirstSemester + evaScienceMarksFirstSemester
def totalMarksSecondSemester : Nat := evaMathsMarksSecondSemester + evaArtsMarksSecondSemester + evaScienceMarksSecondSemester

-- Combined total
def evaTotalMarks : Nat := totalMarksFirstSemester + totalMarksSecondSemester

-- Statement to prove
theorem evaTotalMarksCorrect : evaTotalMarks = 485 := 
by
  -- This needs to be proved as per the conditions and calculations above
  sorry

end evaTotalMarksCorrect_l233_233743


namespace find_m_for_one_solution_l233_233430

theorem find_m_for_one_solution (m : ℚ) :
  (∀ x : ℝ, 3*x^2 - 7*x + m = 0 → (∃! y : ℝ, 3*y^2 - 7*y + m = 0)) → m = 49/12 := by
  sorry

end find_m_for_one_solution_l233_233430


namespace midpoint_x_coordinate_l233_233219

-- Definition for the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Definition for the focus of the parabola
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Definition for the line intersecting the parabola at points A and B
def line (m y x : ℝ) : Prop := m * y = x - 1

-- Definition for |AB| = 8
def segment_length (A B : ℝ × ℝ) : Prop := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8

-- The main statement that needs to be proved
theorem midpoint_x_coordinate (A B : ℝ × ℝ) (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h_parabolaA : parabola A.1 A.2) (h_parabolaB : parabola B.1 B.2)
  (h_focus : focus (1, 0)) (h_lineA : line m A.2 A.1) (h_lineB : line m B.2 B.1)
  (h_segment_length : segment_length A B) :
  (A.1 + B.1) / 2 = 3 :=
sorry

end midpoint_x_coordinate_l233_233219


namespace range_of_a_product_greater_than_one_l233_233829

namespace ProofProblem

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + x^2 - a * x + 2

variables {x1 x2 a : ℝ}

-- Conditions
axiom f_has_two_distinct_zeros : f x1 a = 0 ∧ f x2 a = 0 ∧ x1 ≠ x2

-- Goal 1: Prove the range of a
theorem range_of_a : a ∈ Set.Ioi 3 := sorry  -- Formal expression for (3, +∞) in Lean

-- Goal 2: Prove x1 * x2 > 1 given that a is in the correct range
theorem product_greater_than_one (ha : a ∈ Set.Ioi 3) : x1 * x2 > 1 := sorry

end ProofProblem

end range_of_a_product_greater_than_one_l233_233829


namespace additional_fertilizer_on_final_day_l233_233676

noncomputable def normal_usage_per_day : ℕ := 2
noncomputable def total_days : ℕ := 9
noncomputable def total_fertilizer_used : ℕ := 22

theorem additional_fertilizer_on_final_day :
  total_fertilizer_used - (normal_usage_per_day * total_days) = 4 := by
  sorry

end additional_fertilizer_on_final_day_l233_233676


namespace percentage_divisible_by_6_l233_233634

theorem percentage_divisible_by_6 : 
  let numbers_less_than_or_equal_to_120 := (list.range 120).map (λ x, x + 1) in
  let divisible_by_6 := numbers_less_than_or_equal_to_120.filter (λ x, x % 6 = 0) in
  let percent := (divisible_by_6.length : ℚ) / 120 * 100 in
  percent = 16.67 :=
by 
  sorry

end percentage_divisible_by_6_l233_233634


namespace probability_of_selecting_blue_tile_l233_233669

theorem probability_of_selecting_blue_tile :
  let total_tiles := 70
  let blue_tiles := finset.filter (λ x, x % 7 = 3) (finset.range (total_tiles + 1))
  ((finset.card blue_tiles : ℚ) / total_tiles) = (1 / 7) :=
by
  sorry

end probability_of_selecting_blue_tile_l233_233669


namespace total_games_played_l233_233072

theorem total_games_played (G : ℕ) 
  (h1 : 0.40 * 30 = 12) 
  (h2 : ∀ R, G = 30 + R) 
  (h3 : ∀ R, 0.80 * R) 
  (h4 : 0.60 * G) 
  (h5 : (12 + 0.80 * (G - 30)) = (0.60 * G)) :
  G = 60 :=
by 
  sorry

end total_games_played_l233_233072


namespace nonnegative_solutions_count_l233_233266

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.finite.to_finset.card = 1 :=
by
  sorry

end nonnegative_solutions_count_l233_233266


namespace greatest_prime_factor_of_341_l233_233537

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l233_233537


namespace magnitude_angle_between_vectors_l233_233224

def a : ℝ × ℝ := (1, real.sqrt 3)
def b : ℝ × ℝ := (real.sqrt 3, 1)

theorem magnitude_angle_between_vectors :
  let θ := real.arccos ((a.1 * b.1 + a.2 * b.2) / (real.sqrt (a.1^2 + a.2^2) * real.sqrt (b.1^2 + b.2^2))) in
  θ = real.pi / 6 :=
by 
  sorry

end magnitude_angle_between_vectors_l233_233224


namespace fewest_trips_l233_233229

theorem fewest_trips (max_carry total_objects : ℕ) (h1 : max_carry = 3) (h2 : total_objects = 17) : 
  (total_objects + max_carry - 1) / max_carry = 6 :=
by
  rw [h1, h2]
  sorry

end fewest_trips_l233_233229


namespace percentage_divisible_by_6_l233_233642

-- Defining the sets S and T using Lean
def S := {n : ℕ | 1 ≤ n ∧ n ≤ 120}
def T := {n : ℕ | n ∈ S ∧ 6 ∣ n}

-- Proving the percentage of elements in T with respect to S is 16.67%
theorem percentage_divisible_by_6 : 
  (↑(T.card) : ℚ) / (S.card) * 100 = 16.67 := sorry

end percentage_divisible_by_6_l233_233642


namespace find_y_intercept_l233_233456

theorem find_y_intercept (m : ℝ) (x_intercept : ℝ × ℝ) (hx : x_intercept = (4, 0)) (hm : m = -3) : ∃ y_intercept : ℝ × ℝ, y_intercept = (0, 12) := 
by
  sorry

end find_y_intercept_l233_233456


namespace nonnegative_solution_count_nonnegative_solution_count_unique_l233_233278

theorem nonnegative_solution_count (x : ℝ) :
  (x^2 = -6 * x) → (x ≥ 0) → x = 0 :=
by
  sorry

-- Prove that the number of nonnegative solutions is 1
theorem nonnegative_solution_count_unique :
  {x : ℝ // x ≥ 0 ∧ x^2 = -6 * x}.to_finset.card = 1 :=
by
  sorry

end nonnegative_solution_count_nonnegative_solution_count_unique_l233_233278


namespace total_team_cost_l233_233469

-- Define the costs of individual items and the number of players
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8
def num_players : ℕ := 16

-- Define the total cost for equipment for one player
def player_cost : ℝ :=
  jersey_cost + shorts_cost + socks_cost

-- The main theorem stating the total cost for all players
theorem total_team_cost : num_players * player_cost = 752 := by
  sorry

end total_team_cost_l233_233469


namespace proof_of_range_of_b_over_a_point_not_on_line_l233_233175

variable (a b c : ℝ)
variable (x₁ x₂ : ℝ)
variable (f g : ℝ → ℝ)

axiom f_def : f = λ x, a * x^2 + b * x + c
axiom f_zeros : f x₁ = 0 ∧ f x₂ = 0
axiom f_at_one : f 1 = 2 * a
axiom a_gt_c : a > c
axiom g_def : g = λ x, f (x - x₁) + f (x - x₂)
axiom g_max : ∀ x ∈ Icc (0 : ℝ) (1 : ℝ), g x ≤ 2 / a

theorem proof_of_range_of_b_over_a : 
  (a * a > 0) → ((b / a > 2 * real.sqrt 2 - 2) ∨ (b / a < -2 * real.sqrt 2 - 2)) := 
sorry

theorem point_not_on_line : (a > c) ∧ ((∀ x ∈ Icc (0 : ℝ) (1 : ℝ), g x ≤ 2 / a) → ¬ (a + b = 1)) := 
sorry

end proof_of_range_of_b_over_a_point_not_on_line_l233_233175


namespace expression_evaluation_l233_233707

theorem expression_evaluation : 
  (3.14 - Real.pi)^0 + abs (Real.sqrt 2 - 1) + (1 / 2)^(-1:ℤ) - Real.sqrt 8 = 2 - Real.sqrt 2 :=
by sorry

end expression_evaluation_l233_233707


namespace two_angles_less_than_60_degrees_l233_233793

theorem two_angles_less_than_60_degrees
  (a d : ℝ) (h : 0 < d) :
  ∃ (α β : ℝ), α < 60 ∧ β < 60 ∧ 
  (a - d < a ∧ a < a + d ∧ 
   α = real.acos ((a + d)^2 + a^2 - (a - d)^2 / (2 * (a + d) * a)) ∧
   β = real.acos ((a - d)^2 + a^2 - (a + d)^2 / (2 * (a - d) * a))) :=
sorry

end two_angles_less_than_60_degrees_l233_233793


namespace number_of_elements_cong_set_l233_233856

/-- Define a set of integers less than 500 and congruent to 3 modulo 7 -/
def cong_set : Set ℕ := {n | n < 500 ∧ n % 7 = 3}

/-- The theorem stating the number of elements in cong_set is 72 -/
theorem number_of_elements_cong_set : Set.card cong_set = 72 :=
sorry

end number_of_elements_cong_set_l233_233856


namespace probability_red_or_white_l233_233049

noncomputable def total_marbles : ℕ := 50
noncomputable def blue_marbles : ℕ := 5
noncomputable def red_marbles : ℕ := 9
noncomputable def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white : 
  (red_marbles + white_marbles) / total_marbles = 9 / 10 :=
by sorry

end probability_red_or_white_l233_233049


namespace passes_through_point_P_l233_233824

-- Definition of the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1) + 3

-- Conditions
variables (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)

-- Proof statement
theorem passes_through_point_P : f a 0 = 3 :=
by
  -- This is where the proof would go
  sorry

end passes_through_point_P_l233_233824


namespace cos_675_eq_sqrt2_div_2_l233_233712

theorem cos_675_eq_sqrt2_div_2 : Real.cos (675 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by 
  sorry

end cos_675_eq_sqrt2_div_2_l233_233712


namespace range_of_x_sq_add_y_sq_l233_233200

theorem range_of_x_sq_add_y_sq (x y : ℝ) (h : x^2 + y^2 = 4 * x) : 
  ∃ (a b : ℝ), a ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ b ∧ a = 0 ∧ b = 16 :=
by
  sorry

end range_of_x_sq_add_y_sq_l233_233200


namespace total_team_cost_l233_233470

-- Define the costs of individual items and the number of players
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8
def num_players : ℕ := 16

-- Define the total cost for equipment for one player
def player_cost : ℝ :=
  jersey_cost + shorts_cost + socks_cost

-- The main theorem stating the total cost for all players
theorem total_team_cost : num_players * player_cost = 752 := by
  sorry

end total_team_cost_l233_233470


namespace tangent_position_is_six_oclock_l233_233125

-- Define the radii of the larger and smaller disks
def radius_large : ℝ := 30
def radius_small : ℝ := 15

-- Assume the initial conditions
def initial_tangent_position : ℝ := 0 -- 12 o'clock position

-- Define a function to calculate the position on the larger disk after one complete roll of the smaller disk
def final_tangent_position (r_large r_small : ℝ) : ℝ :=
  (1 / (r_large / r_small) * 180) % 360 -- Move by 180° each full roll

-- The theorem to prove that the final position where the smaller disk is tangent is 6 o'clock
theorem tangent_position_is_six_oclock :
  final_tangent_position radius_large radius_small = 180 :=
by
  sorry

end tangent_position_is_six_oclock_l233_233125


namespace solve_for_x_l233_233617

theorem solve_for_x :
  ∃ x : ℝ, x ≠ 0 ∧ (9 * x) ^ 18 = (27 * x) ^ 9 + 81 * x ∧ x = 1 / 3 :=
by
  sorry

end solve_for_x_l233_233617


namespace arithmetic_sequence_solution_l233_233185

theorem arithmetic_sequence_solution (a : ℕ → ℝ) (q : ℝ) (S_n : ℕ → ℝ)
    (h1 : q > 0)
    (h2 : 2 * a 3 = a 5 - 3 * a 4) 
    (h3 : a 2 * a 4 * a 6 = 64) 
    (h4 : ∀ n, S_n n = (1 - q^n) / (1 - q) * a 1) :
    q = 2 ∧ (∀ n, S_n n = (2^n - 1) / 2) := 
  by
  sorry

end arithmetic_sequence_solution_l233_233185


namespace wheel_speed_l233_233953

theorem wheel_speed (s : ℝ) (t : ℝ) :
  (12 / 5280) * 3600 = s * t →
  (12 / 5280) * 3600 = (s + 4) * (t - (1 / 18000)) →
  s = 8 :=
by
  intro h1 h2
  sorry

end wheel_speed_l233_233953


namespace max_profit_thousand_rubles_l233_233092

theorem max_profit_thousand_rubles :
  ∃ x y : ℕ, 
    (80 * x + 100 * y = 2180) ∧ 
    (10 * x + 70 * y ≤ 700) ∧ 
    (23 * x + 40 * y ≤ 642) := 
by
  -- proof goes here
  sorry

end max_profit_thousand_rubles_l233_233092


namespace f_11_5_equals_neg_1_l233_233659

-- Define the function f with the given properties
axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom periodic_function (f : ℝ → ℝ) : ∀ x, f (x + 2) = f x
axiom f_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x

-- State the theorem to be proved
theorem f_11_5_equals_neg_1 (f : ℝ → ℝ) 
  (odd_f : ∀ x, f (-x) = -f x)
  (periodic_f : ∀ x, f (x + 2) = f x)
  (f_int : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f (11.5) = -1 :=
sorry

end f_11_5_equals_neg_1_l233_233659


namespace range_of_a_l233_233160

noncomputable def f (a x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

def no_solution_in_interval (a : ℝ) : Prop :=
  ∀ (x : ℝ), 0 < x ∧ x < 1/2 → f a x ≠ 0

theorem range_of_a : {a : ℝ | no_solution_in_interval a} = {a | a ∈ Ici (2 - 4 * Real.log 2)} :=
  sorry

end range_of_a_l233_233160


namespace inequality_solution_l233_233459

def solution_set_inequality : Set ℝ := {x | x < -1/3 ∨ x > 1/2}

theorem inequality_solution (x : ℝ) : 
  (2 * x - 1) / (3 * x + 1) > 0 ↔ x ∈ solution_set_inequality :=
by 
  sorry

end inequality_solution_l233_233459


namespace find_number_l233_233488

theorem find_number (x : ℝ) (h : 20 / x = 0.8) : x = 25 := 
by
  sorry

end find_number_l233_233488


namespace count_congruent_numbers_less_than_500_l233_233851

-- Definitions of the conditions
def is_congruent_to_modulo (n a m : ℕ) : Prop := (n % m) = a

-- Main problem statement: Proving that the count of numbers under 500 that satisfy the conditions is 71.
theorem count_congruent_numbers_less_than_500 : 
  { n : ℕ | n < 500 ∧ is_congruent_to_modulo n 3 7 }.card = 71 :=
by
  sorry

end count_congruent_numbers_less_than_500_l233_233851


namespace lines_in_dual_have_edges_l233_233122

-- Define the basic elements: regular polyhedron, its dual, and their relationships
variables (P : Type) [RegularPolyhedron P] (S : Sphere) (L : Set (Line))
variables (D : Type) [RegularPolyhedron D]

-- Assume relationships: the sphere is tangent to all edges of the polyhedron
axiom sphere_tangent_to_edges :
  ∀ (e : Edge P), tangent S e

-- Define the construction of lines through midpoints of edges, perpendicular to edges and radius
axiom lines_through_midpoints :
  ∀ (e : Edge P), 
  midpoint_line e S ∈ L ∧ truly_perpendicular (midpoint_line e S) e (radius_line S e.midpoint)

-- Define the relationship between dual polyhedron and the given one
axiom dual_polyhedron_properties :
  ∀ (v : Vertex D), dual_to P v ∧ sphere_tangent_to_edges_of_dual S v

-- Prove the main statement
theorem lines_in_dual_have_edges (e : Edge P) : 
  ∀ l ∈ L, contains_edge (dual_edge l) D :=
sorry

end lines_in_dual_have_edges_l233_233122


namespace calculate_area_of_SGRTH_l233_233119

noncomputable def area_of_shaded_region (R P Q S T G H : Point) (radius : ℝ) := 
  let PR := dist P R 
  let PQ := dist P Q
  let rect_area := PQ * (2 * radius)
  let triangle_area := 2 * (radius * radius / 2)
  let sector_area := 2 * (π * radius^2 / 8)
  rect_area - triangle_area - sector_area

theorem calculate_area_of_SGRTH : 
  ∀ (P Q R S T G H : Point) (dist_PR : ℝ) (radius : ℝ),
    mid_point R P Q → 
    dist P R = dist_PR → 
    dist_PR = 3 * sqrt 2 → 
    radius = 3 →
    tangent RS P → tangent RT Q → common_tangent GH P Q → 
    area_of_shaded_region R P Q S T G H radius = (18 * sqrt 2) - 9 - (9 * π / 4) :=
by intros; sorry

end calculate_area_of_SGRTH_l233_233119


namespace cos_shift_l233_233009

theorem cos_shift :
  ∀ (x : ℝ), cos (2 * (x - π / 2)) = cos (2 * x - π) := 
by
  intro x
  sorry

end cos_shift_l233_233009


namespace sequence_difference_l233_233685

theorem sequence_difference (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n + n) : a 2017 - a 2016 = 2016 :=
sorry

end sequence_difference_l233_233685


namespace eva_total_marks_l233_233739

theorem eva_total_marks
    (math_score_s2 : ℕ) (arts_score_s2 : ℕ) (science_score_s2 : ℕ)
    (math_diff : ℕ) (arts_diff : ℕ) (science_frac_diff : ℚ)
    (math_score_s2_eq : math_score_s2 = 80)
    (arts_score_s2_eq : arts_score_s2 = 90)
    (science_score_s2_eq : science_score_s2 = 90)
    (math_diff_eq : math_diff = 10)
    (arts_diff_eq : arts_diff = 15)
    (science_frac_diff_eq : science_frac_diff = 1/3) : 
  (math_score_s2 + 10 + (math_score_s2 + math_diff) + 
   (arts_score_s2 + 90 - 15) + (arts_score_s2 + arts_diff) + 
   (science_score_s2 + 90 - (1/3) * 90) + (science_score_s2 + science_score_s2 * 1/3)) = 485 := 
by
  sorry

end eva_total_marks_l233_233739


namespace greatest_prime_factor_341_l233_233584

theorem greatest_prime_factor_341 : ∃ (p : ℕ), prime p ∧ p ∣ 341 ∧ ∀ q, prime q ∧ q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_341_l233_233584


namespace probability_of_other_girl_l233_233339

theorem probability_of_other_girl (A B : Prop) (P : Prop → ℝ) 
    (hA : P A = 3 / 4) 
    (hAB : P (A ∧ B) = 1 / 4) : 
    P (B ∧ A) / P A = 1 / 3 := by 
  -- The proof is skipped using the sorry keyword.
  sorry

end probability_of_other_girl_l233_233339


namespace nonnegative_solutions_eq_1_l233_233296

variable (x : ℝ)

theorem nonnegative_solutions_eq_1 : (x^2 = -6 * x) → (∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀) → (x ∈ {0}) :=
by
  assume (h1: x^2 = -6 * x)
  assume (h2: ∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀)
  sorry

end nonnegative_solutions_eq_1_l233_233296


namespace remaining_fish_count_l233_233131

def initial_fish_counts : Type := (ℕ, ℕ, ℕ, ℕ)
def sold_fish_counts : Type := (ℕ, ℕ, ℕ, ℕ)

-- Define the initial number of fish
def initial_counts : initial_fish_counts := (94, 76, 89, 58)

-- Define the number of fish sold
def sold_counts : sold_fish_counts := (30, 48, 17, 24)

theorem remaining_fish_count : 
  let (guppies, angelfish, tiger_sharks, oscar_fish) := initial_counts in
  let (sold_guppies, sold_angelfish, sold_tiger_sharks, sold_oscar_fish) := sold_counts in
  guppies - sold_guppies + (angelfish - sold_angelfish) + (tiger_sharks - sold_tiger_sharks) + (oscar_fish - sold_oscar_fish) = 198 :=
by
  sorry

end remaining_fish_count_l233_233131


namespace van_capacity_l233_233133

theorem van_capacity (s a v : ℕ) (h1 : s = 2) (h2 : a = 6) (h3 : v = 2) : (s + a) / v = 4 := by
  sorry

end van_capacity_l233_233133


namespace base_10_to_base_5_l233_233508

noncomputable def base_five_equivalent (n : ℕ) : ℕ :=
  let (d1, r1) := div_mod n (5 * 5 * 5) in
  let (d2, r2) := div_mod r1 (5 * 5) in
  let (d3, r3) := div_mod r2 5 in
  let (d4, r4) := div_mod r3 1 in
  d1 * 1000 + d2 * 100 + d3 * 10 + d4

theorem base_10_to_base_5 : base_five_equivalent 156 = 1111 :=
by
  -- Include the proof here
  sorry

end base_10_to_base_5_l233_233508


namespace greatest_diff_l233_233652

theorem greatest_diff (x y : ℤ) (hx1 : 6 < x) (hx2 : x < 10) (hy1 : 10 < y) (hy2 : y < 17) : y - x = 7 :=
sorry

end greatest_diff_l233_233652


namespace incorrect_statement_d_l233_233647

-- Define conditions
def propA (x : ℝ) : Prop := x^2 + x + 1 < 0
def propB (x : ℝ) : Prop := x^2 + x + 1 ≥ 0
def eq1 (x : ℝ) : Prop := x^2 - 4x + 3 = 0
def converse_eq1 (x : ℝ) : Prop := (x ≠ 1) → x^2 - 4x + 3 ≠ 0

-- Define the propositions p and q
def p : Prop := ∃ x : ℝ, propA x
def q : Prop := false  --((For the sake of this proof, we know it's false from the solution))

-- The theorem stating D is incorrect
theorem incorrect_statement_d (p q : Prop) (h : ¬ (p ∧ q)) : (¬ p ∨ ¬ q) :=
by
  sorry

end incorrect_statement_d_l233_233647


namespace altitude_segment_product_eq_half_side_diff_square_l233_233386

noncomputable def altitude_product (a b c t m m_1: ℝ) :=
  m * m_1 = (b^2 + c^2 - a^2) / 2

theorem altitude_segment_product_eq_half_side_diff_square {a b c t m m_1: ℝ}
  (hm : m = 2 * t / a)
  (hm_1 : m_1 = a * (b^2 + c^2 - a^2) / (4 * t)) :
  altitude_product a b c t m m_1 :=
by sorry

end altitude_segment_product_eq_half_side_diff_square_l233_233386


namespace find_base_length_of_isosceles_triangle_l233_233696

noncomputable def is_isosceles_triangle_with_base_len (a b : ℝ) : Prop :=
  a = 2 ∧ ((a + a + b = 5) ∨ (a + b + b = 5))

theorem find_base_length_of_isosceles_triangle :
  ∃ (b : ℝ), is_isosceles_triangle_with_base_len 2 b ∧ (b = 1.5 ∨ b = 2) :=
by
  sorry

end find_base_length_of_isosceles_triangle_l233_233696


namespace min_colors_needed_l233_233121

theorem min_colors_needed (n : ℕ) (clr : ℕ) (C : Fin n → Fin n → Fin clr)
  (h_distinct : ∀ (i j k l : Fin n), 1 ≤ i + 1 < j + 1 ∧ 1 ≤ k + 1 < l + 1 → C i k ≠ C j k ∧ C j k ≠ C j l ∧ C i k ≠ C j l) :
  clr ≥ 2 * n - 1 := by
  sorry

end min_colors_needed_l233_233121


namespace max_acute_triangles_l233_233400

def circle_with_16_points := set (fin 16)

noncomputable def num_triangles : ℕ := (16.choose 3)

noncomputable def num_non_acute_angles : ℕ :=
  let m_chords := (λ m : ℕ, if m ≤ 7 then 16 * m else if m = 7 then 8 * m else 0) in
  finset.sum (finset.range 8) m_chords

noncomputable def num_acute_triangles (num_triangles num_non_acute_angles : ℕ) : ℕ :=
  num_triangles - num_non_acute_angles

theorem max_acute_triangles : num_acute_triangles num_triangles num_non_acute_angles = 168 :=
  by
    sorry

end max_acute_triangles_l233_233400


namespace number_of_integers_congruent_to_3_mod_7_less_than_500_l233_233862

theorem number_of_integers_congruent_to_3_mod_7_less_than_500 : 
  {n : ℕ | n < 500 ∧ n % 7 = 3}.card = 71 :=
sorry

end number_of_integers_congruent_to_3_mod_7_less_than_500_l233_233862


namespace parallel_OP_BD_l233_233952

theorem parallel_OP_BD (k : Type*) [circle k] (O A B C D P : k) 
  (h1 : center O k) 
  (h2 : points_on_circle [A, B, C, D] k) 
  (h3 : diameter AB k) 
  (h4 : circumcircle_intersects COD AC P 2) : 
  parallel (line_through O P) (line_through B D) :=
sorry

end parallel_OP_BD_l233_233952


namespace total_cost_of_office_supplies_l233_233064

-- Define the conditions
def cost_of_pencil : ℝ := 0.5
def cost_of_folder : ℝ := 0.9
def count_of_pencils : ℕ := 24
def count_of_folders : ℕ := 20

-- Define the theorem to prove
theorem total_cost_of_office_supplies
  (cop : ℝ := cost_of_pencil)
  (cof : ℝ := cost_of_folder)
  (ncp : ℕ := count_of_pencils)
  (ncg : ℕ := count_of_folders) :
  cop * ncp + cof * ncg = 30 :=
sorry

end total_cost_of_office_supplies_l233_233064


namespace percentage_of_integers_divisible_by_6_up_to_120_l233_233621

theorem percentage_of_integers_divisible_by_6_up_to_120 : 
  let total := 120
      divisible_by_6 := λ n, n % 6 = 0
      count := (list.range (total + 1)).countp divisible_by_6
      percentage := (count.toFloat / total.toFloat) * 100
  in percentage = 16.67 :=
by
  sorry

end percentage_of_integers_divisible_by_6_up_to_120_l233_233621


namespace math_problem_l233_233115

theorem math_problem : 
  0.0081^(1/4) + (4^(-3/4))^2 + (sqrt 8)^(-4/3) - 16^(-0.75) + 2^(Real.logBase 2 5) = 5.55 :=
by
  sorry

end math_problem_l233_233115


namespace count_congruent_to_3_mod_7_lt_500_l233_233860

theorem count_congruent_to_3_mod_7_lt_500 : 
  ∃ n, n = 71 ∧ ∀ x, 0 < x ∧ x < 500 ∧ x % 7 = 3 ↔ ∃ k, 0 ≤ k ∧ k ≤ 70 ∧ x = 3 + 7 * k :=
sorry

end count_congruent_to_3_mod_7_lt_500_l233_233860


namespace simplify_expression_l233_233806

variable (a b c : ℝ)

-- Conditions
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : b + c > a
axiom triangle_inequality3 : c + a > b

-- Proof problem
theorem simplify_expression : |a - b - c| + |b - c + a| + |c - a - b| = a + 3b - c :=
by
  sorry

end simplify_expression_l233_233806


namespace greatest_prime_factor_of_341_is_17_l233_233562

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l233_233562


namespace nonneg_solutions_count_l233_233251

theorem nonneg_solutions_count :
  {x : ℝ | x^2 + 6 * x = 0 ∧ 0 ≤ x}.card = 1 :=
by
  sorry

end nonneg_solutions_count_l233_233251


namespace hyperbola_asymptotes_slope_l233_233714

open Real

theorem hyperbola_asymptotes_slope (m : ℝ) : 
  (∀ x y : ℝ, (y ^ 2 / 16) - (x ^ 2 / 9) = 1 → (y = m * x ∨ y = -m * x)) → 
  m = 4 / 3 := 
by 
  sorry

end hyperbola_asymptotes_slope_l233_233714


namespace nonnegative_solutions_count_l233_233310

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.card = 1 := 
sorry

end nonnegative_solutions_count_l233_233310


namespace problem1_problem2_l233_233043

-- Problem 1
theorem problem1 (m : ℝ) : (∀ x : ℝ, (m - 2) * x ^ 2 + 2 * (m - 2) * x - 4 < 0) → m ∈ Icc (-2 : ℝ) 2 := sorry

-- Problem 2
theorem problem2 (x : ℝ) : (∀ m ∈ Icc (-1 : ℝ) 1, 2 * x ^ 2 + m * x - 3 < 0) → x ∈ Ioo (-1 : ℝ) 1 := sorry

end problem1_problem2_l233_233043


namespace acute_triangle_condition_l233_233643

theorem acute_triangle_condition (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 0) (h3 : B > 0) (h4 : C > 0)
    (h5 : A + B > 90) (h6 : B + C > 90) (h7 : C + A > 90) : A < 90 ∧ B < 90 ∧ C < 90 :=
sorry

end acute_triangle_condition_l233_233643


namespace fish_remaining_l233_233130

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

end fish_remaining_l233_233130


namespace product_4_6_7_14_l233_233023

theorem product_4_6_7_14 : 4 * 6 * 7 * 14 = 2352 := by
  sorry

end product_4_6_7_14_l233_233023


namespace nonnegative_solution_count_nonnegative_solution_count_unique_l233_233280

theorem nonnegative_solution_count (x : ℝ) :
  (x^2 = -6 * x) → (x ≥ 0) → x = 0 :=
by
  sorry

-- Prove that the number of nonnegative solutions is 1
theorem nonnegative_solution_count_unique :
  {x : ℝ // x ≥ 0 ∧ x^2 = -6 * x}.to_finset.card = 1 :=
by
  sorry

end nonnegative_solution_count_nonnegative_solution_count_unique_l233_233280


namespace tan_pi_minus_alpha_l233_233782

theorem tan_pi_minus_alpha {α : ℝ} (h1 : real.sin α = -5 / 13) (h2 : α > 3 * real.pi / 2 ∧ α < 2 * real.pi) :
  real.tan (real.pi - α) = 5 / 12 :=
sorry

end tan_pi_minus_alpha_l233_233782


namespace greatest_prime_factor_of_341_l233_233552

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l233_233552


namespace all_functions_increasing_on_non_negative_l233_233028

noncomputable def f1 (x : ℝ) : ℝ := 2 * x
noncomputable def f2 (x : ℝ) : ℝ := x^2 + 2 * x - 1
noncomputable def f3 (x : ℝ) : ℝ := |x + 2|
noncomputable def f4 (x : ℝ) : ℝ := |x| + 2

theorem all_functions_increasing_on_non_negative :
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f1(x) ≤ f1(y)) ∧
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f2(x) ≤ f2(y)) ∧
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f3(x) ≤ f3(y)) ∧
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f4(x) ≤ f4(y)) :=
by
  sorry

end all_functions_increasing_on_non_negative_l233_233028


namespace eggs_needed_l233_233993

theorem eggs_needed (s o a se e : ℕ) (hs : s = 53) (ho : o = 35) (ha : a = 75) (hse : se = 37) (he : e = 25) :
  2 * (s * 0.5 + o * 1 + a * 2 + se * 1.5 + e) = 584 :=
by
  have hs_omelets := 53 * 0.5
  have ho_omelets := 35 * 1
  have ha_omelets := 75 * 2
  have hse_omelets := 37 * 1.5
  have total_omelets := hs_omelets + ho_omelets + ha_omelets + hse_omelets
  have total_omelets_with_extra := total_omelets + 25
  have eggs := total_omelets_with_extra * 2
  show 2 * (53 * 0.5 + 35 * 1 + 75 * 2 + 37 * 1.5 + 25) = 584,
  sorry

end eggs_needed_l233_233993


namespace radical_axis_exists_l233_233190

structure Point where
  x : ℝ
  y : ℝ

structure LineSegment where
  start : Point
  end : Point

def length (l : LineSegment) : ℝ :=
  real.sqrt ((l.end.x - l.start.x)^2 + (l.end.y - l.start.y)^2)

def Midpoint (A B : Point) : Point :=
  { x := (A.x + B.x) / 2, y := (A.y + B.y) / 2 }

noncomputable def perpendicularAtMidpoint (A B : Point) : LineSegment :=
  let M := Midpoint A B
  let dx := B.x - A.x
  let dy := B.y - A.y
  LineSegment.mk M { x := M.x - dy, y := M.y + dx }

theorem radical_axis_exists 
  (O1 A1 O2 A2 : Point)
  (l_O1A1 : LineSegment)
  (l_O2A2 : LineSegment)
  (h1 : length l_O1A1 = length (LineSegment.mk O2 A2))
  (h2 : length l_O2A2 = length (LineSegment.mk O1 A1)) :
  ∃ N : Point, isRadicalAxis O1 O2 A1 A2 N :=
by
  sorry

end radical_axis_exists_l233_233190


namespace nonnegative_solutions_eq_one_l233_233289

theorem nonnegative_solutions_eq_one : 
  (finset.univ.filter (λ x : ℝ, x^2 = -6 * x ∧ 0 <= x)).card = 1 := 
by 
  sorry

end nonnegative_solutions_eq_one_l233_233289


namespace exists_k_for_prime_gt_11_l233_233930

theorem exists_k_for_prime_gt_11 (p : ℕ) (hp_prime : p.prime) (hp_gt_11 : p > 11) :
  ∃ k : ℕ, p * k = (10^(p-1) - 1) / 9 :=
sorry

end exists_k_for_prime_gt_11_l233_233930


namespace nonnegative_solutions_count_l233_233305

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.card = 1 := 
sorry

end nonnegative_solutions_count_l233_233305


namespace number_of_integers_congruent_to_3_mod_7_less_than_500_l233_233863

theorem number_of_integers_congruent_to_3_mod_7_less_than_500 : 
  {n : ℕ | n < 500 ∧ n % 7 = 3}.card = 71 :=
sorry

end number_of_integers_congruent_to_3_mod_7_less_than_500_l233_233863


namespace find_abc_value_l233_233870

noncomputable def abc_value (a b c : ℝ) : ℝ :=
  a * b * c

theorem find_abc_value (a b c : ℝ) (h1 : a * b = 30 * Real.cbrt 3) (h2 : a * c = 42 * Real.cbrt 3) (h3 : b * c = 21 * Real.cbrt 3) : 
  abc_value a b c = 210 :=
sorry

end find_abc_value_l233_233870


namespace tom_sleep_increase_l233_233485

theorem tom_sleep_increase :
  ∀ (initial_sleep : ℕ) (increase_by : ℚ), 
  initial_sleep = 6 → 
  increase_by = 1/3 → 
  initial_sleep + increase_by * initial_sleep = 8 :=
by 
  intro initial_sleep increase_by h1 h2
  simp [*, add_mul, mul_comm]
  sorry

end tom_sleep_increase_l233_233485


namespace compost_loading_time_l233_233147

theorem compost_loading_time (rate_steven : ℕ) (rate_darrel : ℕ) (total_composite : ℕ) :
  (rate_steven = 75) →
  (rate_darrel = 10) →
  (total_composite = 2550) →
  (total_composite / (rate_steven + rate_darrel) = 30) :=
by
  intros h_steven h_darrel h_composite
  rw [h_steven, h_darrel, h_composite]
  norm_num
  sorry

end compost_loading_time_l233_233147


namespace powers_of_i_cyclic_l233_233141

theorem powers_of_i_cyclic {i : ℂ} (h_i_squared : i^2 = -1) :
  i^(66) + i^(103) = -1 - i :=
by {
  -- Providing the proof steps as sorry.
  -- This is a placeholder for the actual proof.
  sorry
}

end powers_of_i_cyclic_l233_233141


namespace percent_decrease_l233_233653

variable (OriginalPrice : ℝ) (SalePrice : ℝ)

theorem percent_decrease : 
  OriginalPrice = 100 → 
  SalePrice = 30 → 
  ((OriginalPrice - SalePrice) / OriginalPrice) * 100 = 70 :=
by
  intros h1 h2
  sorry

end percent_decrease_l233_233653


namespace intersection_sum_x_coordinates_l233_233330

theorem intersection_sum_x_coordinates (a : ℝ) (h : ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ π/2 ∧ 0 ≤ x₂ ∧ x₂ ≤ π/2 ∧ x₁ ≠ x₂ ∧ sin (2 * x₁ - π/4) = a ∧ sin (2 * x₂ - π/4) = a ) :
  (∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ ≤ π/2 ∧ 0 ≤ x₂ ∧ x₂ ≤ π/2 ∧ x₁ ≠ x₂ ∧ sin (2 * x₁ - π/4) = a ∧ sin (2 * x₂ - π/4) = a ∧ x₁ + x₂ = 3 * π / 4) :=
sorry

end intersection_sum_x_coordinates_l233_233330


namespace sphere_contains_both_circles_l233_233014

section SphereContainsBothCircles

variables {K1 K2 : Type} [Circle K1] [Circle K2]
variables (A B : Point)
variables (O1 O2 : Point)

-- Conditions
axiom intersects_at_two_points : A ≠ B ∧ A ∈ K1 ∧ A ∈ K2 ∧ B ∈ K1 ∧ B ∈ K2
axiom circles_not_plane : ¬Coplanar K1 K2

-- Equivalent Lean 4 statement
theorem sphere_contains_both_circles :
  ∃ (S : Sphere), ∀ P, (P ∈ K1 ∨ P ∈ K2) → P ∈ S :=
sorry

end SphereContainsBothCircles

end sphere_contains_both_circles_l233_233014


namespace collinear_A_B_D_perpendicular_vectors_l233_233846

variables {α : Type*} [inner_product_space ℝ α]
variables (e₁ e₂ : α)

-- Conditions
def unit_vectors : Prop :=
  ∥e₁∥ = 1 ∧ ∥e₂∥ = 1 ∧ real.angle e₁ e₂ = real.pi / 3

def vectors_AB_BC_CD : Prop :=
  let AB := e₁ + e₂ in
  let BC := 2 • e₁ + 8 • e₂ in
  let CD := 3 • (e₁ - e₂) in 
  True

-- Proof Problem Part 1
theorem collinear_A_B_D (h1 : unit_vectors e₁ e₂) (h2 : vectors_AB_BC_CD e₁ e₂) :
  let A := (0 : α) in
  let B := e₁ + e₂ in
  let D := B + (2 • e₁ + 8 • e₂) + 3 • (e₁ - e₂) in
  collinear ℝ ({A, B, D} : set α) :=
sorry

-- Proof Problem Part 2
theorem perpendicular_vectors (h1 : unit_vectors e₁ e₂) :
  let u := 2 • e₁ + e₂ in
  let v := e₁ - (5 / 4 : ℝ) • e₂ in
  ⟪u, v⟫ = 0 :=
sorry

end collinear_A_B_D_perpendicular_vectors_l233_233846


namespace greatest_prime_factor_of_341_is_17_l233_233553

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l233_233553


namespace folded_paper_points_sum_l233_233067

theorem folded_paper_points_sum (p q : ℝ) :
  let mid := (λ x y : ℝ, (x + y) / 2) in
  let slope := (λ x1 y1 x2 y2 : ℝ, (y2 - y1) / (x2 - x1)) in
  
  (mid 2 6, mid 3 1) = (4, 2) ∧
  slope 2 3 6 1 = -1/2 ∧
  slope 8 4 p q = -1/2 ∧
  (p - 8) * 1 = (q - 4) * (-2) ∧
  mid 8 p, mid 4 q = (2 * mid 8 p - 6) ∧
  p = 16 - 2 * q ∧
  q = p → p + q = 10.7 :=
by
  intros
  sorry

end folded_paper_points_sum_l233_233067


namespace largest_angle_135_l233_233949

theorem largest_angle_135 {A B C P Q R : ℝ}
  (h1 : cos A = sin P)
  (h2 : cos B = sin Q)
  (h3 : cos C = sin R) :
  ∃θ, θ = 135 ∧ (θ = A ∨ θ = B ∨ θ = C ∨ θ = P ∨ θ = Q ∨ θ = R) := 
sorry

end largest_angle_135_l233_233949


namespace correct_play_order_l233_233776

-- Definitions for each player's total points after two rounds
def player_A_points := 23
def player_B_points := 20
def player_C_points := 18
def player_D_points := 16

-- Definition for the tile played by player A in the third round
def tile_A_third_round := (6, 2)

-- Domino tiles used in the problem
def dominoes := [(6, 5), (6, 6), (6, 4), (5, 5), (6, 3), (5, 4), (4, 4), (5, 3), (6, 2)]

-- Placeholder to state the sequence to be proven
def correct_order := [
  (1, (6, 5)),  -- A: 6-5
  (2, (5, 5)),  -- B: 5-5
  (3, (5, 4)),  -- C: 5-4
  (4, (4, 4)),  -- D: 4-4
  (5, (6, 6)),  -- A: 6-6
  (6, (6, 4)),  -- B: 6-4
  (7, (6, 3)),  -- C: 6-3
  (8, (5, 3))   -- D: 5-3
]

-- Theorem statement to prove the order based on conditions
theorem correct_play_order : 
  ∃ (order : List (ℕ × (ℕ × ℕ))), 
    order = correct_order ∧ 
    player_A_points = 23 ∧ 
    player_B_points = 20 ∧ 
    player_C_points = 18 ∧ 
    player_D_points = 16 ∧ 
    nth order 4 = option.some (1, (6, 2)) :=
sorry

end correct_play_order_l233_233776


namespace murtha_pebbles_collection_l233_233973

theorem murtha_pebbles_collection :
  let initial_pebbles := 3 in
  let pebbles_each_day (n : ℕ) := n + 1 in
  let total_days := 15 in
  ∑ i in finset.range total_days, pebbles_each_day i + initial_pebbles = 138 := 
by sorry
 
end murtha_pebbles_collection_l233_233973


namespace smallest_k_exists_permutation_4_real_roots_l233_233161

theorem smallest_k_exists_permutation_4_real_roots :
  ∃ (k : ℝ), k > 0 ∧
  ∀ (a b c d : ℝ), a ≥ k → b ≥ k → c ≥ k → d ≥ k → a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (∃ (p q r s : ℝ), {p, q, r, s} = {a, b, c, d} ∧ 
  ∀ (x : ℝ),
  (x^2 + p * x + q) * (x^2 + r * x + s) = 0 → 
  (is_roots_real_distinct (x^2 + p * x + q) (x^2 + r * x + s))) ∧ k = 4 :=
by sorry

end smallest_k_exists_permutation_4_real_roots_l233_233161


namespace greatest_prime_factor_of_341_l233_233546

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l233_233546


namespace rational_sum_l233_233878

theorem rational_sum (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : x + y = 7 ∨ x + y = 3 := 
sorry

end rational_sum_l233_233878


namespace find_k_l233_233813

variables {a : ℕ → ℕ} {S : ℕ → ℕ} 

def arithmetic_seq (d a1 : ℕ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

def sum_first_n_terms (S a : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def geometric_seq (b1 b2 b3 : ℕ) : Prop :=
  b2 * b2 = b1 * b3

theorem find_k (a1 d k : ℕ)
  (ha : arithmetic_seq d a1)
  (h_sum : sum_first_n_terms S a)
  (h2 : a 2 + a 4 = 6)
  (h6 : a 6 = S 3)
  (hk_seq : geometric_seq (a k) (a (3 * k)) (S (2 * k))) :
  k = 4 := sorry

end find_k_l233_233813


namespace greatest_prime_factor_341_l233_233597

theorem greatest_prime_factor_341 : ∃ p, prime p ∧ p ∣ 341 ∧ p = 17 := 
by {
  use 17,
  split,
  { exact prime_of_factor 17, -- 17 is a prime number
  },
  {
    split,
    {
      norm_num,
      -- proof that 17 divides 341
      sorry,
    },
    -- p == 17
    refl,
  },
}

end greatest_prime_factor_341_l233_233597


namespace zero_sum_of_cosines_in_interval_l233_233784

noncomputable def f (θ : ℝ) : ℝ := sin (4 * θ) + sin (3 * θ)

theorem zero_sum_of_cosines_in_interval
  (θ1 θ2 θ3 : ℝ)
  (h_distinct : θ1 ≠ θ2 ∧ θ2 ≠ θ3 ∧ θ1 ≠ θ3)
  (h_zeros : f θ1 = 0 ∧ f θ2 = 0 ∧ f θ3 = 0)
  (h_interval : θ1 ∈ Ioo 0 π ∧ θ2 ∈ Ioo 0 π ∧ θ3 ∈ Ioo 0 π) :
  θ1 + θ2 + θ3 = (12 / 7) * π ∧
  cos θ1 * cos θ2 * cos θ3 = 1 / 8 ∧
  cos θ1 + cos θ2 + cos θ3 = -1 / 2 :=
sorry

end zero_sum_of_cosines_in_interval_l233_233784


namespace total_points_scored_l233_233721

theorem total_points_scored 
  (darius_score : ℕ) 
  (marius_score : ℕ) 
  (matt_score : ℕ) 
  (h1 : marius_score = darius_score + 3) 
  (h2 : darius_score = matt_score - 5)
  (h3 : darius_score = 10) : darius_score + marius_score + matt_score = 38 :=
by
  sorry

end total_points_scored_l233_233721


namespace distance_to_place_l233_233065

theorem distance_to_place (rowing_speed still_water : ℝ) (downstream_speed : ℝ)
                         (upstream_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  rowing_speed = 10 → downstream_speed = 2 → upstream_speed = 3 →
  total_time = 10 → distance = 44.21 → 
  (distance / (rowing_speed + downstream_speed) + distance / (rowing_speed - upstream_speed)) = 10 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3]
  field_simp
  sorry

end distance_to_place_l233_233065


namespace part_1_part_2_l233_233191

open Set

section Problem

variable (x m : ℝ)

def A : Set ℝ := {x | (x + 1) * (x - 5) ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0}

theorem part_1 (h : A ⊆ B m) : m ≥ 4 :=
sorry

theorem part_2 (h : (A ∩ B m).card = 3) : 1 ≤ m ∧ m < 2 :=
sorry

end Problem

end part_1_part_2_l233_233191


namespace circle_polar_eq_l233_233352

def polar_point := {ρ : ℝ, θ : ℝ}

noncomputable def P : polar_point := ⟨√2, π / 4⟩

noncomputable def line_intersect_polar_axis : (ρ : ℝ) (θ : ℝ) : polar_point → polar_point := 
  λ ρ θ, if ρ * sin (θ - π / 3) = -√(3)/2 then ⟨1, 0⟩ else ⟨0, 0⟩  -- Simplification just for the example

theorem circle_polar_eq (C P : polar_point) (center_eq : C = ⟨1, 0⟩) (contains_P : P.ρ = √2 ∧ P.θ = π / 4) :
  (∀ θ, C.ρ = 2 * cos θ) :=
by sorry

end circle_polar_eq_l233_233352


namespace greatest_prime_factor_of_341_is_17_l233_233555

theorem greatest_prime_factor_of_341_is_17 : ∃ p : ℕ, prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, q ∣ 341 → prime q → q ≤ p :=
sorry

end greatest_prime_factor_of_341_is_17_l233_233555


namespace greatest_prime_factor_of_341_l233_233534

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l233_233534


namespace percentage_of_copper_first_alloy_l233_233106

theorem percentage_of_copper_first_alloy :
  ∃ x : ℝ, 
  (66 * x / 100) + (55 * 21 / 100) = 121 * 15 / 100 ∧
  x = 10 := 
sorry

end percentage_of_copper_first_alloy_l233_233106


namespace base_five_of_156_is_1111_l233_233506

def base_five_equivalent (n : ℕ) : ℕ := sorry

theorem base_five_of_156_is_1111 :
  base_five_equivalent 156 = 1111 :=
sorry

end base_five_of_156_is_1111_l233_233506


namespace fish_remaining_l233_233129

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

end fish_remaining_l233_233129


namespace arithmetic_problem_l233_233662

theorem arithmetic_problem :
  12.1212 + 17.0005 - 9.1103 = 20.0114 :=
sorry

end arithmetic_problem_l233_233662


namespace find_price_per_package_l233_233695

-- Definitions and conditions as given in the problem

-- Price per package
variable P : ℝ

-- Number of packages sold
variable n : ℝ := 50

-- Payment received
variable payment : ℝ := 1150

-- Equation representing the payment received
def total_revenue (P : ℝ) : ℝ := 10 * P + 40 * (4 / 5) * P

-- Lean statement to prove the price per package of gaskets
theorem find_price_per_package :
  total_revenue P = payment → P = 27.38 :=
begin
  sorry
end

end find_price_per_package_l233_233695


namespace proof_problem_l233_233210

noncomputable def hyperbola_E_eq (x y : ℝ) : Prop :=
  y^2 / 9 - x^2 / 4 = 1

noncomputable def passes_through (x y : ℝ) (a b : ℝ) : Prop :=
  y = b ∧ x = a

noncomputable def asymptotes_E_and_F (x y : ℝ) : Prop :=
  y^2 / 9 - x^2 / 4 = 1 ∨ x^2 / 4 - y^2 / 9 = 1

noncomputable def midpoint_line (A B : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

noncomputable def line_eq (A B : ℝ × ℝ) : ℝ × Prop :=
  let slope := (B.2 - A.2) / (B.1 - A.1) in
  (slope, ∀ x y : ℝ, y - A.2 = slope * (x - A.1))

noncomputable def tangent_line_eq (x₁ y₁ x y : ℝ) : Prop :=
  9 * x₁ * x - 4 * y₁ * y + 36 = 0

noncomputable def point_moves_on_line (P : ℝ × ℝ) : Prop :=
  3 * P.1 - 4 * P.2 + 6 = 0

theorem proof_problem :
  (∃ (x y : ℝ), passes_through x y (-2) (3 * real.sqrt 2) ∧ asymptotes_E_and_F x y) →
  ¬(∃ (x y : ℝ), y^2 / 18 - x^2 / 8 = 1) ∧ 
  (∃ (A B : ℝ × ℝ), midpoint_line A B (1, 4) ∧ line_eq A B = (9/16, ∀ x y, y = 9/16 * x + 55)) ∧
  (∃ (x₁ y₁ : ℝ), tangent_line_eq x₁ y₁ x₁ y₁) ∧
  (∀ P : ℝ × ℝ, point_moves_on_line P → (∃ (l : ℝ × Prop), passes_through l.1 (3, 6))) :=
sorry

end proof_problem_l233_233210


namespace total_equipment_cost_l233_233467

-- Define the cost of each piece of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8

-- Define the number of players
def players : ℕ := 16

-- Define the total cost of equipment for one player
def equipment_cost_per_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Define the total cost for all players
def total_cost : ℝ := players * equipment_cost_per_player

-- The proof problem to be stated:
theorem total_equipment_cost (jc sc k p : ℝ) (n : ℕ) :
  jc = 25 ∧ sc = 15.2 ∧ k = 6.8 ∧ p = 16 →
  total_cost = 752 :=
by
  intro h
  rcases h with ⟨hc1, hc2, hc3, hc4⟩
  simp [total_cost, equipment_cost_per_player, hc1, hc2, hc3, hc4]
  exact sorry

end total_equipment_cost_l233_233467


namespace find_eccentricity_l233_233169

variable (a b c e : ℝ)
variable (Γ : Type) -- ellipse
variable (C : Type) -- Monge circle
variable (h1 : a^2 + b^2 = 3 * b^2) -- Condition derived from Monge circle equation

-- lemmas and additional assumptions
lemma a_squared_eq_2b_squared : a^2 = 2 * b^2 := 
by linarith

lemma eccentricity_of_ellipse (h2 : a^2 = b^2 + c^2) : e = real.sqrt (1 - b^2 / a^2) := 
by simp [h2, ← a_squared_eq_2b_squared]

theorem find_eccentricity :
  e = real.sqrt (1 - (b^2 / (2 * b^2))) := 
by sorry

#eval find_eccentricity

end find_eccentricity_l233_233169


namespace finish_11th_l233_233342

noncomputable def place_in_race (place: Fin 15) := ℕ

variables (Dana Ethan Alice Bob Chris Flora : Fin 15)

def conditions := 
  Dana.val + 3 = Ethan.val ∧
  Alice.val = Bob.val - 2 ∧
  Chris.val = Flora.val - 5 ∧
  Flora.val = Dana.val + 2 ∧
  Ethan.val = Alice.val - 3 ∧
  Bob.val = 6

theorem finish_11th (h : conditions Dana Ethan Alice Bob Chris Flora) : Flora.val = 10 :=
  by sorry

end finish_11th_l233_233342


namespace equal_sum_of_radii_l233_233982

theorem equal_sum_of_radii (A B C D : ℝ) (R : ℝ) :
  let r1 := 4 * R * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2),
      r2 := 4 * R * Real.sin (B / 2) * Real.sin (C / 2) * Real.sin (D / 2),
      r3 := 4 * R * Real.sin (A / 2) * Real.sin (D / 2) * Real.sin (B / 2),
      r4 := 4 * R * Real.sin (C / 2) * Real.sin (D / 2) * Real.sin (A / 2) in 
  r1 + r2 = r3 + r4 :=
by
  -- Proof steps to be filled in
  sorry

end equal_sum_of_radii_l233_233982


namespace intersecting_curve_and_line_l233_233761

theorem intersecting_curve_and_line (k : ℝ) :
  let curve := λ x, real.sqrt (4 - x^2) in
  let line := λ x, k * x - 2 * k + 4 in
  (∀ x, curve x = line x → false) ↔ k ∈ set.Ioc (3/4) 1 :=
by 
  sorry

end intersecting_curve_and_line_l233_233761


namespace greatest_prime_factor_341_l233_233573

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l233_233573


namespace quadratic_inequality_count_l233_233834

-- Definitions of the inequalities
def ineq1 (x : ℝ) : Prop := 3 * x + 4 < 0
def ineq2 (x : ℝ) (m : ℝ) : Prop := x^2 + m * x - 1 > 0
def ineq3 (x : ℝ) (a : ℝ) : Prop := a * x^2 + 4 * x - 7 > 0
def ineq4 (x : ℝ) : Prop := x^2 < 0

-- Definition of what it means to be a quadratic inequality
def is_quadratic_inequality (ineq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, ineq x = (a * x^2 + b * x + c > 0)

-- Statement that there are exactly 2 quadratic inequalities
theorem quadratic_inequality_count (m a : ℝ) :
  (is_quadratic_inequality (ineq1) = false) ∧
  (is_quadratic_inequality (λ x, ineq2 x m) = true) ∧
  (is_quadratic_inequality (λ x, ineq3 x a) = true) ∧
  (is_quadratic_inequality (ineq4) = false) →
  2 :=
by 
  sorry

end quadratic_inequality_count_l233_233834


namespace base_five_of_156_is_1111_l233_233505

def base_five_equivalent (n : ℕ) : ℕ := sorry

theorem base_five_of_156_is_1111 :
  base_five_equivalent 156 = 1111 :=
sorry

end base_five_of_156_is_1111_l233_233505


namespace scooter_initial_value_l233_233471

noncomputable def initialValue (endValue : ℝ) : ℝ :=
  let depreciationFactor := (3 / 4 : ℝ)^5
  endValue / depreciationFactor

theorem scooter_initial_value :
  ∀ (V₀ : ℝ), initialValue 9492.1875 = 40000 :=
by
  intro V₀
  have h := initialValue 9492.1875
  rw [h]
  sorry

end scooter_initial_value_l233_233471


namespace monthly_food_cost_correct_l233_233362

/-- The cost of the adoption fee. -/
def adoption_fee : ℝ := 50

/-- The cost of the vet visits for the first year. -/
def vet_visits_cost : ℝ := 500

/-- The cost of the toys Jenny bought for the cat. -/
def toys_cost : ℝ := 200

/-- The total amount Jenny spent on the cat in the first year. -/
def total_spent : ℝ := 625

/-- Jenny and her girlfriend agreed to split the adoption and vet costs evenly. -/
def shared_adoption_cost : ℝ := adoption_fee / 2

def shared_vet_cost : ℝ := vet_visits_cost / 2

/-- The total shared costs Jenny paid -/
def total_shared_cost : ℝ := shared_adoption_cost + shared_vet_cost

/-- The total amount Jenny spent on food. -/
def total_food_cost : ℝ := total_spent - total_shared_cost - toys_cost

/-- The monthly cost of food given the total food cost for the year. -/
def monthly_food_cost : ℝ := total_food_cost / 12

/-- Proof that the monthly food cost is $12.50 given the conditions --/
theorem monthly_food_cost_correct : monthly_food_cost = 12.50 := 
by 
  -- This is a placeholder for the actual proof
  sorry

end monthly_food_cost_correct_l233_233362


namespace square_inequality_l233_233319

theorem square_inequality (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end square_inequality_l233_233319


namespace sum_expression_l233_233837

variable {ℕ : Type*}

def sequence (n : ℕ) : ℕ -- define the sequence a_n without specifying its exact form
  := sorry -- the definition is not required here

def sum_of_first_n_terms (n : ℕ) : ℕ -- define the sum of the first n terms S_n
  := sorry -- the definition is not required here

axiom sum_definition (n : ℕ) : sum_of_first_n_terms n = 2 * sequence n - 2^n -- given condition

theorem sum_expression (n : ℕ) : sum_of_first_n_terms n = n * 2^n := 
  sorry

end sum_expression_l233_233837


namespace p_n_ge_one_p_mn_eq_p_m_of_p_n_l233_233163

noncomputable def p_n (n : ℤ) (x : ℝ) : ℝ :=
  (1 / 2) * ((x + real.sqrt (x^2 - 1))^n + (x - real.sqrt (x^2 - 1))^n)

-- Hypotheses
variables {x : ℝ} (hx : x ≥ 1) (n m : ℤ)

theorem p_n_ge_one :
  p_n n x ≥ 1 := 
  sorry

theorem p_mn_eq_p_m_of_p_n :
  p_n (m * n) x = p_n m (p_n n x) := 
  sorry

end p_n_ge_one_p_mn_eq_p_m_of_p_n_l233_233163


namespace median_free_throws_is_24_l233_233667

def free_throws : List ℕ := [22, 25, 21, 31, 17, 27, 23, 27, 30, 19]

theorem median_free_throws_is_24 : (median free_throws) = 24 := by
  /-
  1. Arrange the list in ascending order.
  2. Find the 5th and 6th terms in the sorted list.
  3. Calculate the average of these terms to determine the median.
  -/
  sorry

end median_free_throws_is_24_l233_233667


namespace find_omega_and_extrema_l233_233216

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := (√3 / 2) - √3 * (sin (ω * x))^2 - (sin (ω * x)) * (cos (ω * x))

theorem find_omega_and_extrema :
  ∃ ω : ℝ, ω > 0 ∧ (∀ x, f ω x = -sin (2 * ω * x - π / 3)) ∧
  (ω = 1) ∧
  ∀ x, π ≤ x ∧ x ≤ 3 * π / 2 → 
    ∃ maximum, ∃ minimum,
    (maximum = f 1 π) ∧ (minimum = f 1 (17 * π / 12)) ∧
    (maximum = √3 / 2) ∧ (minimum = -1) :=
sorry

end find_omega_and_extrema_l233_233216


namespace part_I_part_II_l233_233828

-- Part (I) 
theorem part_I (f : ℝ → ℝ) (m : ℝ) (hf : ∀ x, f x = log x - (m / 2) * x^2 + x) (hm : m > 0)
  (h_cond : ∀ x, f x ≤ m * x - 1 / 2) :
  m ≥ 1 :=
sorry

-- Part (II)
theorem part_II (f : ℝ → ℝ) (m : ℝ) (hf : ∀ x, f x = log x - (m / 2) * x^2 + x)
  (h_m_neg : m = -1) (h_cond : ∀ x1 x2, f x1 + f x2 = 0) :
  ∀ x1 x2, x1 > 0 → x2 > 0 → x1 + x2 ≥ Real.sqrt 3 - 1 :=
sorry

end part_I_part_II_l233_233828


namespace distance_from_P_to_x_axis_l233_233819

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1}

def foci (a b : ℝ) : ℝ :=
  real.sqrt (a ^ 2 - b ^ 2)

def is_right_triangle {α : Type*} [inner_product_space ℝ α] (A B C : α) : Prop :=
  ⟪B - A, C - A⟫ = 0 ∨ ⟪C - A, B - A⟫ = 0 ∨ ⟪B - C, A - C⟫ = 0

theorem distance_from_P_to_x_axis
  (a b : ℝ) (P F1 F2 : ℝ × ℝ)
  (hE : P ∈ ellipse a b)
  (hf1f2 : F1 = (-foci a b, 0) ∨ F1 = (foci a b, 0))
  (hF1F2 : F2 = (-F1.1, 0))
  (hRT : is_right_triangle P F1 F2)
  (a_eq : a = 4)
  (b_eq : b = 3) :
  ∃ (d : ℝ), d = abs P.2 ∧ d = 9 / 4 :=
sorry

end distance_from_P_to_x_axis_l233_233819


namespace coefficient_of_friction_correct_l233_233671

noncomputable def coefficient_of_friction (R Fg: ℝ) (α: ℝ) : ℝ :=
  (1 - R * real.cos α) / (R * real.sin α)

theorem coefficient_of_friction_correct
  (Fg: ℝ)
  (α: ℝ)
  (R: ℝ := 11 * Fg)
  (hα: α = real.pi * 80 / 180):
  coefficient_of_friction R Fg α = 0.17 :=
by
  sorry

end coefficient_of_friction_correct_l233_233671


namespace drainage_time_l233_233435

noncomputable def hourly_drainage_rate : ℝ := 1 / (15 * 24)

noncomputable def pump_work_rate (days: ℝ) (pumps: ℝ) : ℝ :=
  days * 24 * pumps * hourly_drainage_rate

theorem drainage_time (x: ℝ) :
  let total_drainage := 3 * 24 * hourly_drainage_rate + (hourly_drainage_rate + hourly_drainage_rate) * x in
  total_drainage = 1 →
  x = 144 :=
by
  sorry

end drainage_time_l233_233435


namespace greatest_prime_factor_of_341_l233_233564

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l233_233564


namespace length_BD_l233_233909

-- Define the structures
structure Point := (x : ℝ) (y : ℝ)
structure Triangle := (A : Point) (B : Point) (C : Point)

-- Given conditions as definitions
def AB (t : Triangle) : ℝ := 45
def AC (t : Triangle) : ℝ := 60

def right_angle_at (t : Triangle) : Prop :=
  t.A.x = 0 ∧ t.A.y = 0 ∧ t.B.y = 0 ∧ t.C.x = 0

def perpendicular (D : Point) (B C: Point) (A: Point) : Prop :=
  A.x = D.x ∧ D.y * B.y + D.y * C.y = 0

-- The theorem to be proved
theorem length_BD (t : Triangle) (D : Point) 
  (h_right : right_angle_at t) 
  (h_AB : AB t = dist t.A t.B) 
  (h_AC : AC t = dist t.A t.C) 
  (h_AD_perp : perpendicular D t.B t.C t.A) : 
  dist t.B D = 27 :=
sorry

end length_BD_l233_233909


namespace arithmetic_series_sum_l233_233764

theorem arithmetic_series_sum (k : ℕ) :
  let a1 := k^2 + 2
  let d := 2
  let n := k + 1
  let an := a1 + (n-1) * d
  let Sn := n / 2 * (a1 + an)
  Sn = k^3 + 2k^2 + 3k + 2 :=
sorry

end arithmetic_series_sum_l233_233764


namespace percent_divisible_by_6_l233_233624

theorem percent_divisible_by_6 (N : ℕ) (hN : N = 120) :
  (∃ M, M = (finset.univ.filter (λ n : ℕ, n ≤ N ∧ n % 6 = 0)).card ∧ M * 6 = N) →
  (M.to_real / N.to_real) * 100 = 16.66666667 :=
by
  intros h
  sorry

end percent_divisible_by_6_l233_233624


namespace day_of_week_first_day_of_month_l233_233433

theorem day_of_week_first_day_of_month (D: ℕ) (h: D = 26) : ∃ day : string, day = "Friday" :=
by
  -- Given that the 26th day of the month is a Tuesday,
  -- We need to prove that the 1st day of the same month is a Friday

  -- Let's use sorry to skip the actual proof
  sorry

end day_of_week_first_day_of_month_l233_233433


namespace nonnegative_solution_count_nonnegative_solution_count_unique_l233_233284

theorem nonnegative_solution_count (x : ℝ) :
  (x^2 = -6 * x) → (x ≥ 0) → x = 0 :=
by
  sorry

-- Prove that the number of nonnegative solutions is 1
theorem nonnegative_solution_count_unique :
  {x : ℝ // x ≥ 0 ∧ x^2 = -6 * x}.to_finset.card = 1 :=
by
  sorry

end nonnegative_solution_count_nonnegative_solution_count_unique_l233_233284


namespace nonnegative_solutions_eq_1_l233_233301

variable (x : ℝ)

theorem nonnegative_solutions_eq_1 : (x^2 = -6 * x) → (∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀) → (x ∈ {0}) :=
by
  assume (h1: x^2 = -6 * x)
  assume (h2: ∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀)
  sorry

end nonnegative_solutions_eq_1_l233_233301


namespace convert_2_years_to_months_convert_48_hours_to_days_convert_6_weeks_to_days_total_days_march_to_august_february_days_in_leap_and_common_months_with_31_and_30_days_l233_233045

-- Definitions for the conditions provided
def years_to_months (years : ℕ) : ℕ := years * 12
def hours_to_days (hours : ℕ) : ℕ := hours / 24
def weeks_to_days (weeks : ℕ) : ℕ := weeks * 7
def days_in_months (months : List (ℕ × ℕ)) : ℕ := months.map (λ x => x.2).sum
def february_days (is_leap_year : Bool) : ℕ := if is_leap_year then 29 else 28

-- Conditions translating the given problem
theorem convert_2_years_to_months : years_to_months 2 = 24 := by
  simp [years_to_months]
  norm_num

theorem convert_48_hours_to_days : hours_to_days 48 = 2 := by
  simp [hours_to_days]
  norm_num

theorem convert_6_weeks_to_days : weeks_to_days 6 = 42 := by
  simp [weeks_to_days]
  norm_num

theorem total_days_march_to_august : days_in_months [(3, 31), (4, 30), (5, 31), (6, 30), (7, 31), (8, 31)] = 184 := by
  simp [days_in_months]
  norm_num

theorem february_days_in_leap_and_common : february_days true = 29 ∧ february_days false = 28 := by
  simp [february_days]
  norm_num

theorem months_with_31_and_30_days : ∃ (m31 m30 : ℕ), m31 = 7 ∧ m30 = 4 := by
  use 7, 4
  split
  norm_num
  norm_num

end convert_2_years_to_months_convert_48_hours_to_days_convert_6_weeks_to_days_total_days_march_to_august_february_days_in_leap_and_common_months_with_31_and_30_days_l233_233045


namespace general_formula_l233_233794

noncomputable def a : ℕ → ℚ
| 0       := 1
| 1       := 1
| (n + 2) := n * a n / (n + 2)

theorem general_formula (n : ℕ) :
  a n = 2 / (n * (n + 1)) :=
sorry

end general_formula_l233_233794


namespace division_result_l233_233749

theorem division_result : 210 / (15 + 12 * 3 - 6) = 210 / 45 :=
by
  sorry

end division_result_l233_233749


namespace geometric_progression_common_ratio_l233_233094

theorem geometric_progression_common_ratio (b : ℕ) (q : ℕ) 
  (h₁ : ∀ n : ℕ, n ≥ 1 → b * q^(n-1) ∈ ℕ) 
  (h₂ : b * q^2 + b * q^4 + b * q^6 = 819 * 6^2016) :
  q = 1 ∨ q = 2 ∨ q = 3 ∨ q = 4 :=
by sorry

end geometric_progression_common_ratio_l233_233094


namespace nonnegative_solutions_eq_1_l233_233300

variable (x : ℝ)

theorem nonnegative_solutions_eq_1 : (x^2 = -6 * x) → (∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀) → (x ∈ {0}) :=
by
  assume (h1: x^2 = -6 * x)
  assume (h2: ∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀)
  sorry

end nonnegative_solutions_eq_1_l233_233300


namespace number_of_nonnegative_solutions_l233_233235

-- Define the equation x^2 = -6x
def equation (x : ℝ) : Prop := x^2 = -6 * x

-- Define the condition for a nonnegative solution
def nonnegative (x : ℝ) : Prop := x ≥ 0

-- Define the main theorem to prove the number of nonnegative solutions
theorem number_of_nonnegative_solutions : 
  {x : ℝ // equation x ∧ nonnegative x}.to_finset.card = 1 :=
sorry

end number_of_nonnegative_solutions_l233_233235


namespace only_quadratic_equation_l233_233646

def is_quadratic (eq : String) : Prop := 
  eq = "3x^{2} - 5x = 0"

theorem only_quadratic_equation : 
  is_quadratic "3x^{2} - 5x = 0" ∧ ¬ is_quadratic "x^{2} - \frac{1}{x} = 2023" ∧ 
  ¬ is_quadratic "y - 3x = 0" ∧ ¬ is_quadratic "x^{3} + x^{2} + 1 = 0" :=
by
  sorry

end only_quadratic_equation_l233_233646


namespace union_of_sets_l233_233347

theorem union_of_sets (A B : Set α) : A ∪ B = { x | x ∈ A ∨ x ∈ B } :=
by
  sorry

end union_of_sets_l233_233347


namespace base_five_of_156_is_1111_l233_233504

def base_five_equivalent (n : ℕ) : ℕ := sorry

theorem base_five_of_156_is_1111 :
  base_five_equivalent 156 = 1111 :=
sorry

end base_five_of_156_is_1111_l233_233504


namespace crayon_count_l233_233479

theorem crayon_count (C : ℕ)
  (h1 : 1/3 * C = (nat.cast (C / 3)))
  (h2 : 56 + 20/100 * C + 1/3 * C = C) :
  C = 120 :=
begin
  sorry
end

end crayon_count_l233_233479


namespace y_intercept_of_line_l233_233457

theorem y_intercept_of_line (m x1 y1 : ℝ) (x_intercept : x1 = 4) (y_intercept_at_x1_zero : y1 = 0) (m_value : m = -3) :
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ∧ x = 0 → y = b) ∧ b = 12 :=
by
  sorry

end y_intercept_of_line_l233_233457


namespace greatest_prime_factor_of_341_l233_233533

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, is_prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, is_prime q ∧ q ∣ 341 → q ≤ p :=
begin
  -- Our goal is to show that p = 31 meets these conditions.
  sorry
end

end greatest_prime_factor_of_341_l233_233533


namespace train_y_completion_l233_233497

variables {dist : ℝ} {time_x : ℝ} {time_y : ℝ} (vy : ℝ)

-- Conditions
def total_distance := 180
def train_x_speed := total_distance / time_x
def distance_train_x_met := 80
def time_train_x_met := distance_train_x_met / train_x_speed

-- Conditions for Lean proof
axiom train_x_time_completion (h1 : total_distance = 180) (hx : time_x = 5) : train_x_speed = 36
axiom met_condition (h2 : distance_train_x_met = 80) : time_train_x_met = 2.22
axiom total_meeting_distance (h3 : total_distance = 180) (dx : distance_train_x_met = 80)
  : dx + (total_distance - dx) = total_distance

-- Proof problem
theorem train_y_completion (h1 : total_distance = 180) (hx : time_x = 5) (h2 : distance_train_x_met = 80)
  (hy : vy = (total_distance - distance_train_x_met) / time_train_x_met) : 
  time_y = total_distance / vy :=
begin
  sorry
end

end train_y_completion_l233_233497


namespace find_coefficients_sum_l233_233778

theorem find_coefficients_sum :
  let a := 1
  let f (x : ℝ) := (1 - 2 * x)^10
  let a0 : ℝ := a
  let a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ
  (f x = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 + a9 * x^9 + a10 * x^10) : Prop →
  (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 = 0) : Prop →
  (a1 + 2 * a2 + 3 * a3 + 4 * a4 + 5 * a5 + 6 * a6 + 7 * a7 + 8 * a8 + 9 * a9 + 10 * a10 = 20) : Prop →
  (10 * a1 + 9 * a2 + 8 * a3 + 7 * a4 + 6 * a5 + 5 * a6 + 4 * a7 + 3 * a8 + 2 * a9 + a10 = -20) :=
by
  intros a f a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 h₁ h₂ h₃
  sorry

end find_coefficients_sum_l233_233778


namespace find_m_range_l233_233816

theorem find_m_range (m : ℝ) :
  (∃ (P Q R : ℝ × ℝ), P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧
   (P.1 ^ 2 + P.2 ^ 2 = 4) ∧ (Q.1 ^ 2 + Q.2 ^ 2 = 4) ∧ (R.1 ^ 2 + R.2 ^ 2 = 4) ∧
   (abs (P.2 + P.1 - m) / real.sqrt 2 = 1) ∧ 
   (abs (Q.2 + Q.1 - m) / real.sqrt 2 = 1) ∧ 
   (abs (R.2 + R.1 - m) / real.sqrt 2 = 1)) ↔ (-real.sqrt 2 ≤ m ∧ m ≤ real.sqrt 2) :=
sorry

end find_m_range_l233_233816


namespace fiona_reaches_goal_without_predators_l233_233162

def lily_pad (n : ℕ) : Prop := n ≤ 15

def predators_on_pads (4 9 : ℕ) : Prop := true

def fiona_start (n : ℕ) : Prop := n = 0

def fiona_goal (n : ℕ) : Prop := n = 14

def hop_to_next (p : ℕ) : Prop := p = p + 1

def jump_two_pads (p : ℕ) : Prop := p = p + 2

noncomputable def probability_reach_goal :
  ℕ → ℕ → Prop
| 0, 14 := (56 / 729 : ℚ)
| _, _  := 0

theorem fiona_reaches_goal_without_predators :
  ∀ n : ℕ,
    lily_pad n →
    fiona_start 0 →
    fiona_goal 14 →
    predators_on_pads 4 9 →
    probability_reach_goal 0 14 = (56 / 729 : ℚ) :=
sorry

end fiona_reaches_goal_without_predators_l233_233162


namespace greatest_prime_factor_of_341_l233_233565

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l233_233565


namespace combined_points_kjm_l233_233402

theorem combined_points_kjm {P B K J M H C E: ℕ} 
  (total_points : P + B + K + J + M = 81)
  (paige_points : P = 21)
  (brian_points : B = 20)
  (karen_jennifer_michael_sum : K + J + M = 40)
  (karen_scores : ∀ p, K = 2 * p + 5 * (H - p))
  (jennifer_scores : ∀ p, J = 2 * p + 5 * (C - p))
  (michael_scores : ∀ p, M = 2 * p + 5 * (E - p)) :
  K + J + M = 40 :=
by sorry

end combined_points_kjm_l233_233402


namespace length_PI_incenter_l233_233358

theorem length_PI_incenter (P Q R I : Type) (PQ PR QR : ℝ) (hPQ : PQ = 40) (hPR : PR = 42) (hQR : QR = 58) 
  (hI : I = incenter P Q R) : dist P I = 30 := by
  sorry

end length_PI_incenter_l233_233358


namespace map_distance_l233_233399

theorem map_distance (scale_cm : ℝ) (scale_km : ℝ) (actual_distance_km : ℝ) 
  (h1 : scale_cm = 0.4) (h2 : scale_km = 5.3) (h3 : actual_distance_km = 848) :
  actual_distance_km / (scale_km / scale_cm) = 64 :=
by
  rw [h1, h2, h3]
  -- Further steps would follow here, but to ensure code compiles
  -- and there is no assumption directly from solution steps, we use sorry.
  sorry

end map_distance_l233_233399


namespace find_a_l233_233754

noncomputable def problem_statement (x y a b : ℝ) : Prop :=
  (arccos ((4 + y) / 4) = arccos (x - a)) ∧
  (x^2 + y^2 - 4 * x + 8 * y = b)

theorem find_a (a : ℝ) :
  (∀ b : ℝ, ∀ x y : ℝ, problem_statement x y a b → (∃! (u v : ℝ), problem_statement u v a b)) ↔ 
  (a ≤ -15 ∨ a ≥ 19) :=
by 
  sorry

end find_a_l233_233754


namespace goblins_return_l233_233890

theorem goblins_return (n : ℕ) (f : Fin n → Fin n) (h1 : ∀ a, ∃! b, f a = b) (h2 : ∀ b, ∃! a, f a = b) : 
  ∃ k : ℕ, ∀ x : Fin n, (f^[k]) x = x := 
sorry

end goblins_return_l233_233890


namespace total_points_scored_l233_233719

theorem total_points_scored 
  (darius_score : ℕ) 
  (marius_score : ℕ) 
  (matt_score : ℕ) 
  (h1 : marius_score = darius_score + 3) 
  (h2 : darius_score = matt_score - 5)
  (h3 : darius_score = 10) : darius_score + marius_score + matt_score = 38 :=
by
  sorry

end total_points_scored_l233_233719


namespace nonnegative_solutions_count_l233_233268

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.finite.to_finset.card = 1 :=
by
  sorry

end nonnegative_solutions_count_l233_233268


namespace part1_part2_l233_233959

noncomputable section

variables (a x : ℝ)

def P : Prop := x^2 - 4*a*x + 3*a^2 < 0
def Q : Prop := abs (x - 3) ≤ 1

-- Part 1: If a=1 and P ∨ Q, prove the range of x is 1 < x ≤ 4
theorem part1 (h1 : a = 1) (h2 : P a x ∨ Q x) : 1 < x ∧ x ≤ 4 :=
sorry

-- Part 2: If ¬P is necessary but not sufficient for ¬Q, prove the range of a is 4/3 ≤ a ≤ 2
theorem part2 (h : (¬P a x → ¬Q x) ∧ (¬Q x → ¬P a x → False)) : 4/3 ≤ a ∧ a ≤ 2 :=
sorry

end part1_part2_l233_233959


namespace find_a11_l233_233812

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a11 (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 1 * a 4 = 20)
  (h3 : a 0 + a 5 = 9) :
  a 10 = 25 / 4 :=
sorry

end find_a11_l233_233812


namespace rationalize_denominator_l233_233416

theorem rationalize_denominator :
  let expr := (2 + Real.sqrt 5) / (3 - Real.sqrt 5),
      A := 11 / 4,
      B := 5 / 4,
      C := 5
  in expr = (A + B * Real.sqrt C) → (A * B * C) = 275 / 16 :=
by
  intros
  sorry

end rationalize_denominator_l233_233416


namespace money_left_in_wallet_l233_233691

def olivia_initial_money : ℕ := 54
def olivia_spent_money : ℕ := 25

theorem money_left_in_wallet : olivia_initial_money - olivia_spent_money = 29 :=
by
  sorry

end money_left_in_wallet_l233_233691


namespace find_cd_l233_233702

noncomputable def period := (3 / 4) * Real.pi
noncomputable def x_value := (1 / 8) * Real.pi
noncomputable def y_value := 3
noncomputable def tangent_value := Real.tan (Real.pi / 6) -- which is 1 / sqrt(3)
noncomputable def c_value := 3 * Real.sqrt 3

theorem find_cd (c d : ℝ) 
  (h_period : d = 4 / 3) 
  (h_point : y_value = c * Real.tan (d * x_value)) :
  c * d = 4 * Real.sqrt 3 := 
sorry

end find_cd_l233_233702


namespace question_1_question_2_question_3_l233_233984

variable (a b : ℝ)

-- (a * b)^n = a^n * b^n for natural numbers n
theorem question_1 (n : ℕ) : (a * b)^n = a^n * b^n := sorry

-- Calculate 2^5 * (-1/2)^5
theorem question_2 : 2^5 * (-1/2)^5 = -1 := sorry

-- Calculate (-0.125)^2022 * 2^2021 * 4^2020
theorem question_3 : (-0.125)^2022 * 2^2021 * 4^2020 = 1 / 32 := sorry

end question_1_question_2_question_3_l233_233984


namespace nonnegative_solution_count_nonnegative_solution_count_unique_l233_233282

theorem nonnegative_solution_count (x : ℝ) :
  (x^2 = -6 * x) → (x ≥ 0) → x = 0 :=
by
  sorry

-- Prove that the number of nonnegative solutions is 1
theorem nonnegative_solution_count_unique :
  {x : ℝ // x ≥ 0 ∧ x^2 = -6 * x}.to_finset.card = 1 :=
by
  sorry

end nonnegative_solution_count_nonnegative_solution_count_unique_l233_233282


namespace base_10_to_base_5_l233_233507

noncomputable def base_five_equivalent (n : ℕ) : ℕ :=
  let (d1, r1) := div_mod n (5 * 5 * 5) in
  let (d2, r2) := div_mod r1 (5 * 5) in
  let (d3, r3) := div_mod r2 5 in
  let (d4, r4) := div_mod r3 1 in
  d1 * 1000 + d2 * 100 + d3 * 10 + d4

theorem base_10_to_base_5 : base_five_equivalent 156 = 1111 :=
by
  -- Include the proof here
  sorry

end base_10_to_base_5_l233_233507


namespace greatest_prime_factor_of_341_l233_233548

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l233_233548


namespace greatest_prime_factor_341_l233_233578

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l233_233578


namespace determine_m_for_unique_solution_l233_233427

-- Define the quadratic equation and the condition for a unique solution
def quadratic_eq_has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

-- Define the specific quadratic equation and its discriminant
def specific_quadratic_eq (m : ℝ) : Prop :=
  quadratic_eq_has_one_solution 3 (-7) m

-- State the main theorem to prove the value of m
theorem determine_m_for_unique_solution :
  specific_quadratic_eq (49 / 12) :=
by
  unfold specific_quadratic_eq quadratic_eq_has_one_solution
  sorry

end determine_m_for_unique_solution_l233_233427


namespace minimal_communication_l233_233445

-- Define the conditions under which the problem operates
def card_numbers : Set ℕ := { n | ∃ k, n = 2^k ∧ k < 100 } 

-- The sum of 2^0, 2^1, ..., 2^99 represented in Lean
def sum_of_powers : ℕ := (List.range 100).foldl (λ s k => s + 2^k) 0

-- Prove that communicating {1, 2, 4, ..., 2^99, 2^100 - 1} ensures survival
theorem minimal_communication :
  (∀ n ∈ card_numbers, ∃ k, n = 2^k) ∧
  (sum_of_powers = 2^100 - 1) →
  ∀ (communicated_numbers : Set ℕ), communicated_numbers = card_numbers ∪ {sum_of_powers } →
  communicated_numbers.card = 101 →
  (∀ subset_sum ∈ communicated_numbers, ∃ subset ∈ (T : Set ℕ) → T ⊆ card_numbers ∪ singleton sum_of_powers ∧ subset.sum = subset_sum) →
  true := 
by 
  sorry

end minimal_communication_l233_233445


namespace find_f2_l233_233172

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f2 (h : f a b (-2) = 2) : f a b 2 = 0 := by
  sorry

end find_f2_l233_233172


namespace area_isosceles_trapezoid_l233_233103

-- Define the problem setup
variables {b1 : ℝ} {θ : ℝ} {sinθ : ℝ} {a h x : ℝ}

-- Define the given conditions
def is_isosceles_trapezoid_circumscribed_circle (trapezoid : Type) : Prop :=
  ∀ (a b c d : ℝ), -- sides of the trapezoid
    b1 = 16 ∧
    θ = arcsin(0.8) ∧
    (sinθ = 0.8) ∧
    (b + d = a + c) -- property of being circumscribed

-- Define and prove the area of the trapezoid
theorem area_isosceles_trapezoid (h : ℝ) (a : ℝ) (x : ℝ)
  (H1 : is_isosceles_trapezoid_circumscribed_circle trapezoid)
  (H2 : b1 = 16)
  (H3 : θ = arcsin(0.8))
  (H4 : h = 0.8 * a) :
  1 / 2 * (b1 + x) * h = 80 :=
by
  sorry

end area_isosceles_trapezoid_l233_233103


namespace train_speed_correct_l233_233083

-- Define the problem conditions
def length_of_train : ℝ := 300  -- length in meters
def time_to_cross_pole : ℝ := 18  -- time in seconds

-- Conversion factors
def meters_to_kilometers : ℝ := 0.001
def seconds_to_hours : ℝ := 1 / 3600

-- Define the conversions
def distance_in_kilometers := length_of_train * meters_to_kilometers
def time_in_hours := time_to_cross_pole * seconds_to_hours

-- Define the speed calculation
def speed_of_train := distance_in_kilometers / time_in_hours

-- The theorem to prove
theorem train_speed_correct : speed_of_train = 60 := 
by
  sorry

end train_speed_correct_l233_233083


namespace a_pow_four_mod_7_l233_233380

theorem a_pow_four_mod_7 (a : ℤ) (h : a ≡ a⁻¹ [ZMOD 7]) : a^4 ≡ 1 [ZMOD 7] := by
  sorry

end a_pow_four_mod_7_l233_233380


namespace greatest_prime_factor_of_341_l233_233567

theorem greatest_prime_factor_of_341 : ∃ p : ℕ, nat.prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, nat.prime q ∧ q ∣ 341 → q ≤ p :=
begin
  use 19,
  split,
  { exact nat.prime_19 },
  split,
  { exact dvd.intro 17 rfl },
  { intros q hq,
    rcases hq with ⟨hq_prime, hq_dvd⟩,
    rw nat.dvd_prime at hq_dvd,
    { cases hq_dvd,
      { exact le_rfl },
      { linarith [hq_prime.ne_zero, nat.prime.ne_one nat.prime_19] } },
    { exact nat.prime_19 },
    { intro h,
      linarith [hq_prime.ne_zero, ne.symm h] } }
end

end greatest_prime_factor_of_341_l233_233567


namespace total_team_cost_l233_233468

-- Define the costs of individual items and the number of players
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8
def num_players : ℕ := 16

-- Define the total cost for equipment for one player
def player_cost : ℝ :=
  jersey_cost + shorts_cost + socks_cost

-- The main theorem stating the total cost for all players
theorem total_team_cost : num_players * player_cost = 752 := by
  sorry

end total_team_cost_l233_233468


namespace part1_solution_set_part2_range_m_l233_233960

def f (x : ℝ) : ℝ := |x + 2| - |x - 1|

theorem part1_solution_set :
  ∀ x, f(x) > 1 ↔ x > 0 :=
sorry

theorem part2_range_m (m : ℝ) :
  (∃ x, f(x) + 4 ≥ |1 - 2 * m|) → -3 ≤ m ∧ m ≤ 4 :=
sorry

end part1_solution_set_part2_range_m_l233_233960


namespace third_derivative_of_f_l233_233757

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) * (Real.log x) ^ 2

theorem third_derivative_of_f : 
  deriv (deriv (deriv f)) = λ x, (4 * Real.log x * (3 - x) - 18) / (x ^ 3) :=
by
  sorry

end third_derivative_of_f_l233_233757


namespace boy_speed_in_km_per_hr_l233_233052

-- Define the given conditions as Lean definitions
def side_length : ℝ := 20
def time_seconds : ℝ := 24

-- Define the perimeter of the square field
def perimeter : ℝ := 4 * side_length

-- Define the distance covered by the boy
def distance_covered : ℝ := perimeter

-- Define the speed in meters per second
def speed_m_s : ℝ := distance_covered / time_seconds

-- Conversion factors
def meter_to_km (m : ℝ) : ℝ := m * 0.001
def second_to_hour (s : ℝ) : ℝ := s / 3600

-- Convert speed from m/s to km/hr
def speed_km_hr : ℝ := meter_to_km(speed_m_s) / second_to_hour(1)

-- State the theorem to be proved
theorem boy_speed_in_km_per_hr : speed_km_hr = 12 := by
  sorry

end boy_speed_in_km_per_hr_l233_233052


namespace rationalize_denominator_ABC_value_l233_233411

def A := 11 / 4
def B := 5 / 4
def C := 5

theorem rationalize_denominator : 
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = A + B * Real.sqrt C :=
sorry

theorem ABC_value :
  A * B * C = 275 :=
sorry

end rationalize_denominator_ABC_value_l233_233411


namespace max_good_subset_size_l233_233381

open Finset

-- Define the set Sn
def S_n (n : ℕ) : Finset (Fin n →₀ ℤ) := 
  univ.filter (λ f, ∀ i, f i = 0 ∨ f i = 1)

-- Define the distance function d
def d {n : ℕ} (a b : Fin n →₀ ℤ) : ℤ := 
  ∑ i, abs (a i - b i)

-- Define good subset predicate
def good_subset {n : ℕ} (A : Finset (Fin n →₀ ℤ)) : Prop := 
  ∀ a b ∈ A, a ≠ b → d a b ≥ 2 ^ (n - 1)

-- Prove the maximum size of a good subset is 2^(n+1)
theorem max_good_subset_size (n : ℕ) (A : Finset (Fin n →₀ ℤ)) (h1: A ⊆ S_n n) (h2: good_subset A) : 
  A.card ≤ 2 ^ (n + 1) :=
sorry

end max_good_subset_size_l233_233381


namespace larger_circle_radius_l233_233015

/-- Two concentric circles, radii ratio 1:4, with specific properties of a diameter and chord.
    Prove the radius of the larger circle -/
theorem larger_circle_radius :
  ∀ (r : ℝ), (r > 0) →
  ∃ R : ℝ, (16 = R) ∧ 
  (let smaller_circle_radius := r,
       larger_circle_radius := 4 * r,
       AC := 2 * larger_circle_radius,
       BC_tangent := 1,
       B_distance := 8 in
       larger_circle_radius = 4 * smaller_circle_radius) :=
begin
  sorry
end

end larger_circle_radius_l233_233015


namespace calculation_l233_233116

theorem calculation : abs (-3) + (sqrt 2 - 1) ^ 0 - (sqrt 3) ^ 2 = 1 :=
by
  sorry

end calculation_l233_233116


namespace hyperbola_foci_rotation_l233_233123

theorem hyperbola_foci_rotation (x y : ℝ) :
  (∃ x y : ℝ, 4 * x * x - y * y + 8 * x - 4 * y - 4 = 0) →
  (foci: Set (ℝ × ℝ)) :=
  let a := 1 in
  let b := 2 in
  let c := Real.sqrt (a * a + b * b) in
  ((1 + c) / Real.sqrt 2, -3 / Real.sqrt 2) ∈ foci ∧
  ((1 - c) / Real.sqrt 2, -3 / Real.sqrt 2) ∈ foci :=
by 
  sorry

end hyperbola_foci_rotation_l233_233123


namespace a²_minus_b²_l233_233323

theorem a²_minus_b² (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := 
by
  sorry

end a²_minus_b²_l233_233323


namespace osmanthus_trees_variance_l233_233774

theorem osmanthus_trees_variance :
  let n := 4
  let p := 4 / 5
  let ξ := binomial n p
  ξ.variance = 16 / 25 :=
by
  sorry

end osmanthus_trees_variance_l233_233774


namespace p_iff_q_l233_233199

-- Defining conditions as propositions
def p (a : ℝ) :=
  ∀ x y : ℝ, x < y → x < -1 → y < -1 → |x + a| ≤ |y + a|

def q (a : ℝ) :=
  0 < a ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → x > -1 → y > -1 → log a (x + 1) < log a (y + 1)

-- Statement of the proof problem
theorem p_iff_q (a : ℝ) : p a ↔ q a :=
sorry

end p_iff_q_l233_233199


namespace symmetry_axis_one_of_cos_2x_minus_sin_2x_l233_233730

noncomputable def symmetry_axis (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 2) - Real.pi / 8

theorem symmetry_axis_one_of_cos_2x_minus_sin_2x :
  symmetry_axis (-Real.pi / 8) :=
by
  use 0
  simp
  sorry

end symmetry_axis_one_of_cos_2x_minus_sin_2x_l233_233730


namespace graph_B_correct_l233_233833

def f (x : ℝ) : ℝ :=
if x >= -3 ∧ x <= 0 then 
  -2 - x
else if x >= 0 ∧ x <= 2 then 
  Real.sqrt (4 - (x - 2)^2) - 2
else if x >= 2 ∧ x <= 3 then 
  2 * (x - 2)
else 
  0

def g (x : ℝ) : ℝ := -f x

theorem graph_B_correct : 
  (∀ x, g x = 
  if x >= -3 ∧ x <= 0 then 
    -(-2 - x)
  else if x >= 0 ∧ x <= 2 then 
    - (Real.sqrt (4 - (x - 2)^2) - 2)
  else if x >= 2 ∧ x <= 3 then 
    - (2 * (x - 2))
  else 
    0) :=
begin
  sorry
end

end graph_B_correct_l233_233833


namespace domain_of_f_l233_233437

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (1 - x))

theorem domain_of_f : ∀ x : ℝ, f x ∈ ℝ ↔ x < 1 :=
by
  -- Proof details will go here
  sorry

end domain_of_f_l233_233437


namespace total_points_correct_l233_233726

-- Define the scores
def Marius (Darius : ℕ) : ℕ := Darius + 3
def Matt (Darius : ℕ) : ℕ := Darius + 5

-- Define the total points function
def total_points (Darius : ℕ) : ℕ :=
  Darius + Marius Darius + Matt Darius

-- Specific value for Darius's score
def Darius_score : ℕ := 10

-- The theorem that proves the total score is 38 given Darius's score
theorem total_points_correct :
  total_points Darius_score = 38 :=
by
  sorry

end total_points_correct_l233_233726


namespace minimum_distance_MN_l233_233809

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (1/2 : ℝ) * x^2
def g (x : ℝ) : ℝ := x - 1
def h (x : ℝ) : ℝ := Real.exp x - (1/2 : ℝ) * x^2 + 1 - x

theorem minimum_distance_MN : ∀ (x1 x2 : ℝ),
  (0 ≤ x1) → (0 < x2) → (f x1 = g x2) → (x2 - x1 ≥ 2) :=
by
  intros x1 x2 h_x1_nonneg h_x2_pos h_eq
  sorry

end minimum_distance_MN_l233_233809


namespace closest_point_on_line_y_eq_3x_plus_2_l233_233759

theorem closest_point_on_line_y_eq_3x_plus_2 (x y : ℝ) :
  ∃ (p : ℝ × ℝ), p = (-1 / 2, 1 / 2) ∧ y = 3 * x + 2 ∧ p = (x, y) :=
by
-- We skip the proof steps and provide the statement only
sorry

end closest_point_on_line_y_eq_3x_plus_2_l233_233759


namespace F_on_bisector_l233_233926

/-- Let  M  be the midpoint of the side  BC  of an acute triangle  ABC . 
    Incircle of the triangle  ABM  is tangent to the side  AB  at the point  D . 
    Incircle of the triangle  ACM  is tangent to the side  AC  at the point  E . 
    Let  F  be the point such that the quadrilateral  DMEF  is a parallelogram. 
    Prove that  F  lies on the bisector of  ∠ BAC . -/
theorem F_on_bisector (A B C M D E F : Point) 
  (hM_midpoint : M = midpoint B C) 
  (hD_tangent_ABM : tangent_at_incircle A B M D)
  (hE_tangent_ACM : tangent_at_incircle A C M E)
  (hParallelogram : parallelogram D M E F) : 
  lies_on_bisector F A B C :=
sorry

end F_on_bisector_l233_233926


namespace sum_fraction_bound_l233_233931

theorem sum_fraction_bound (x : Fin 2002 → ℝ) 
  (hpos : ∀ i, 1 ≤ i → i ≤ 2001 → 0 < x i)
  (hineq : ∀ i, 2 ≤ i → i ≤ 2001 → x i ^ 2 ≥ x 1 ^ 2 + (Finset.range (i - 1)).sum (λ j, x (j + 1) ^ 2 / (j + 1) ^ 3))
  : (Finset.range 2000).sum (λ i, x (i + 2) / (Finset.range (i + 1)).sum (λ j, x (j + 1))) > 1.999 := 
sorry

end sum_fraction_bound_l233_233931


namespace meaningful_expression_range_l233_233762

theorem meaningful_expression_range (x : ℝ) : 
  (7 - x ≥ 0) → (2x - 6 ≠ 0) ↔ (x ≤ 7) ∧ (x ≠ 3) := 
by 
  sorry

end meaningful_expression_range_l233_233762


namespace higher_room_amount_higher_60_l233_233974

variable (higher_amount : ℕ)

theorem higher_room_amount_higher_60 
  (total_rent : ℕ) (amount_credited_50 : ℕ)
  (total_reduction : ℕ)
  (condition1 : total_rent = 400)
  (condition2 : amount_credited_50 = 50)
  (condition3 : total_reduction = total_rent / 4)
  (condition4 : 10 * higher_amount - 10 * amount_credited_50 = total_reduction) :
  higher_amount = 60 := 
sorry

end higher_room_amount_higher_60_l233_233974


namespace lim_m_exists_lim_m_zero_l233_233166

noncomputable def m (r : ℝ) : ℝ := 
  Inf ((λ (p : ℤ × ℤ), abs (r - real.sqrt (p.1 ^ 2 + 2 * p.2 ^ 2))) '' set.univ)

open real

theorem lim_m_exists (r : ℝ) (h : r > 0) :
  ∃ L : ℝ, filter.tendsto (λ r, m r) filter.at_top (nhds L) :=
sorry

theorem lim_m_zero (r : ℝ) (h : r > 0) :
  filter.tendsto (λ r, m r) filter.at_top (nhds 0) :=
sorry

#check lim_m_exists
#check lim_m_zero

end lim_m_exists_lim_m_zero_l233_233166


namespace product_of_y_coordinates_l233_233405

noncomputable def point := (ℝ × ℝ)

def line_x_equals (c : ℝ) (p : point) : Prop := p.1 = c

def distance (p q : point) : ℝ := real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

def is_on_line_x_4 (p : point) : Prop := line_x_equals 4 p

def is_distance_8 (p : point) : Prop := distance p (1, 5) = 8

theorem product_of_y_coordinates :
  let y1 := 5 + real.sqrt 55;
  let y2 := 5 - real.sqrt 55 in
  y1 * y2 = -30 :=
by
  sorry

end product_of_y_coordinates_l233_233405


namespace range_of_a_l233_233822

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≥ 1 then x * Real.log x - a * x^2 else a^x

theorem range_of_a (a : ℝ) : (∀ x y, x < y → f a y < f a x) ↔ (1/2 ≤ a ∧ a < 1) :=
by
  sorry

end range_of_a_l233_233822


namespace jimin_initial_candies_l233_233395

theorem jimin_initial_candies : 
  let candies_given_to_yuna := 25
  let candies_given_to_sister := 13
  candies_given_to_yuna + candies_given_to_sister = 38 := 
  by 
    sorry

end jimin_initial_candies_l233_233395


namespace abs_sum_values_l233_233875

theorem abs_sum_values (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : 
  x + y = 7 ∨ x + y = 3 := 
by
  sorry

end abs_sum_values_l233_233875


namespace greatest_prime_factor_of_341_l233_233545

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end greatest_prime_factor_of_341_l233_233545


namespace nonnegative_solutions_eq_1_l233_233297

variable (x : ℝ)

theorem nonnegative_solutions_eq_1 : (x^2 = -6 * x) → (∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀) → (x ∈ {0}) :=
by
  assume (h1: x^2 = -6 * x)
  assume (h2: ∃ x₀ : ℝ, x₀ ≥ 0 ∧ x = x₀)
  sorry

end nonnegative_solutions_eq_1_l233_233297


namespace find_pennies_l233_233011

theorem find_pennies (quarters dimes nickels total_dollars : ℕ) 
    (value_quarter value_dime value_nickel value_penny : ℝ)
    (q_condition : quarters = 10)
    (d_condition : dimes = 3)
    (n_condition : nickels = 4)
    (td_condition : total_dollars = 5)
    (vq_condition : value_quarter = 0.25)
    (vd_condition : value_dime = 0.10)
    (vn_condition : value_nickel = 0.05) 
    (vp_condition : value_penny = 0.01) :
  let total_value := total_dollars - 
    (quarters * value_quarter + dimes * value_dime + nickels * value_nickel) in
  total_value / value_penny = 200 :=
by
  sorry

end find_pennies_l233_233011


namespace vertex_on_x_axis_l233_233473

theorem vertex_on_x_axis (c : ℝ) : (∃ (h : ℝ), (h, 0) = ((-(-8) / (2 * 1)), c - (-8)^2 / (4 * 1))) → c = 16 :=
by
  sorry

end vertex_on_x_axis_l233_233473


namespace greatest_prime_factor_341_l233_233526

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l233_233526


namespace max_abs_ge_one_plus_a_l233_233950

open Real

def f (a x : ℝ) : ℝ := a * x^3 + (1 - 4 * a) * x^2 + (5 * a - 1) * x + (3 - 5 * a)
def g (a x : ℝ) : ℝ := (1 - a) * x^3 - x^2 + (2 - a) * x - (3 * a + 1)
def h (a x : ℝ) : ℝ := max (|f a x|) (|g a x|)

theorem max_abs_ge_one_plus_a (a : ℝ) (h_a : 0 < a ∧ a < 1) (x : ℝ) : h a x ≥ 1 + a := by
  sorry

end max_abs_ge_one_plus_a_l233_233950


namespace percent_divisible_by_six_up_to_120_l233_233631

theorem percent_divisible_by_six_up_to_120 : 
  let total_numbers := 120
  let divisible_by_six := total_numbers / 6
  let percentage := (divisible_by_six * 100) / total_numbers
  percentage = 50 / 3 := sorry

end percent_divisible_by_six_up_to_120_l233_233631


namespace nonnegative_solutions_eq_one_l233_233293

theorem nonnegative_solutions_eq_one : 
  (finset.univ.filter (λ x : ℝ, x^2 = -6 * x ∧ 0 <= x)).card = 1 := 
by 
  sorry

end nonnegative_solutions_eq_one_l233_233293


namespace collinear_points_l233_233069

open Classical

variables {P E O Q A B C D : Type} 
variables (circumcenter : (P × P × P) → P) 
variables (triangle : (P × P × P) → Prop) 
variables (intersects : (P → P → P → Prop) → P × P → P → Prop)
variables (are_collinear : (P × P × P) → Prop)

axiom distinct_points (P E : P) : P ≠ E
axiom definitions (A B C D : P) (P : P)
(def1 : circumcenter (P, A, B) = _) (def2 : circumcenter (P, B, C) = _)
(def3 : circumcenter (P, C, D) = _) (def4 : circumcenter (P, D, A) = _)
(def5 : intersects (fun p q => p = q) (circumcenter (P, A, B), circumcenter (P, C, D)) Q)

theorem collinear_points : are_collinear (O, E, Q) :=
sorry

end collinear_points_l233_233069


namespace truck_travel_distance_l233_233090

variable (d1 d2 g1 g2 : ℝ)
variable (rate : ℝ)

-- Define the conditions
axiom condition1 : d1 = 300
axiom condition2 : g1 = 10
axiom condition3 : rate = d1 / g1
axiom condition4 : g2 = 15

-- Define the goal
theorem truck_travel_distance : d2 = rate * g2 := by
  -- axiom assumption placeholder
  exact sorry

end truck_travel_distance_l233_233090


namespace greatest_number_of_identical_bouquets_l233_233035

theorem greatest_number_of_identical_bouquets (white_tulips : ℕ) (red_tulips : ℕ) (gcd_white_red : ℕ)
  (h_white : white_tulips = 21)
  (h_red : red_tulips = 91)
  (h_gcd : Nat.gcd white_tulips red_tulips = gcd_white_red) :
  gcd_white_red = 7 ∧ (white_tulips / gcd_white_red) = 3 :=
by 
  rw [h_white, h_red] at h_gcd
  have : Nat.gcd 21 91 = 7, from rfl
  split
  . exact rfl
  . rw [Nat.div_eq_of_eq_mul_left (by norm_num : 7 ≠ 0) (by norm_num : 21 = 7 * 3)]
    exact rfl

end greatest_number_of_identical_bouquets_l233_233035


namespace product_of_two_numbers_l233_233760

theorem product_of_two_numbers : 
  ∀ (x y : ℝ), (x + y = 60) ∧ (x - y = 10) → x * y = 875 :=
by
  intros x y h
  sorry

end product_of_two_numbers_l233_233760


namespace min_shots_guarantee_l233_233912

-- Define the dimensions of the grid
def grid : Type := Fin 7 × Fin 7

-- Define the shapes of the enemy ship
inductive Shape
| bar : Fin 4 → Fin 1 → Shape
| L_shaped : Fin 3 → Fin 3 → Shape

-- Definition of a 4-cell ship (our Shape)
def is_ship_shape (s : Shape) : Prop :=
  match s with
  | Shape.bar _ _ => true
  | Shape.L_shaped _ _ => true

-- Define a hit as one cell being hit
def hit (p : grid) : Prop := sorry

-- Define the minimum number of shots required to guarantee a hit
def min_shots : ℕ := 20

-- The theorem we need to prove
theorem min_shots_guarantee (ship : Shape) (ship_placed : (Fin 7 × Fin 7) → Prop) (h1 : is_ship_shape ship) :
  ∃ shots : (Fin 7 × Fin 7) → Prop, (@fintype.card (Fin 7 × Fin 7) _ {p // shots p = true}) = min_shots ∧
  ∀ (pos : grid), (ship_placed pos) → (∃ (p : grid), shots p ∧ hit p) :=
sorry

end min_shots_guarantee_l233_233912


namespace compost_loading_time_l233_233148

theorem compost_loading_time (rate_steven : ℕ) (rate_darrel : ℕ) (total_composite : ℕ) :
  (rate_steven = 75) →
  (rate_darrel = 10) →
  (total_composite = 2550) →
  (total_composite / (rate_steven + rate_darrel) = 30) :=
by
  intros h_steven h_darrel h_composite
  rw [h_steven, h_darrel, h_composite]
  norm_num
  sorry

end compost_loading_time_l233_233148


namespace investment_ratio_l233_233901

noncomputable def ratio_A_B (profit : ℝ) (profit_C : ℝ) (ratio_A_C : ℝ) (ratio_C_A : ℝ) := 
  3 / 1

theorem investment_ratio (total_profit : ℝ) (C_profit : ℝ) (A_C_ratio : ℝ) (C_A_ratio : ℝ) :
  total_profit = 60000 → C_profit = 20000 → A_C_ratio = 3 / 2 → ratio_A_B total_profit C_profit A_C_ratio C_A_ratio = 3 / 1 :=
by 
  intros h1 h2 h3
  sorry

end investment_ratio_l233_233901


namespace range_of_a_extreme_points_l233_233830

-- Given function
def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2

-- Proving the range of real number a
theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → f x a ≤ x) : a ≥ 2 / Real.exp 2 :=
sorry

-- Function g and its extreme points
def g (x : ℝ) (a : ℝ) : ℝ := f x a - x

theorem extreme_points (x1 x2 a : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 ≠ x2)
  (h4 : ∀ x : ℝ, g x a = 0 → x = x1 ∨ x = x2) : 
  1 / (Real.log x1) + 1 / (Real.log x2) > 2 * a * Real.exp 1 := 
sorry

end range_of_a_extreme_points_l233_233830


namespace simplify_expression_l233_233937

noncomputable def q (x a b c d : ℝ) :=
  (x + a)^4 / ((a - b) * (a - c) * (a - d))
  + (x + b)^4 / ((b - a) * (b - c) * (b - d))
  + (x + c)^4 / ((c - a) * (c - b) * (c - d))
  + (x + d)^4 / ((d - a) * (d - b) * (d - c))

theorem simplify_expression (a b c d x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) :
  q x a b c d = a + b + c + d + 4 * x :=
by
  sorry

end simplify_expression_l233_233937


namespace triangle_has_area_40_l233_233817

def side_length (area: ℕ) : ℕ := Nat.sqrt area

noncomputable def triangle_area (b h: ℕ) : ℝ := (1 / 2) * b * h

theorem triangle_has_area_40 (sq1 sq2 sq3: ℕ) :
  sq1 = 64 → sq2 = 49 → sq3 = 100 →
  let s1 := side_length sq1 in
  let s2 := side_length sq2 in
  let s3 := side_length sq3 in
  let diag_s2 := s2 * Real.sqrt 2 in
  triangle_area s3 s1 = 40 :=
by 
  intro h1 h2 h3
  let s1 := side_length 64
  let s2 := side_length 49
  let s3 := side_length 100
  let diag_s2 := s2 * Real.sqrt 2
  have step1 : s1 = 8 := by sorry
  have step2 : s2 = 7 := by sorry
  have step3 : s3 = 10 := by sorry
  have step4 : triangle_area s3 s1 = 40 := by sorry
  exact step4

end triangle_has_area_40_l233_233817


namespace greatest_prime_factor_341_l233_233577

theorem greatest_prime_factor_341 : ∃ p : ℕ, prime p ∧ p.factor 341 ∧ (∀ q : ℕ, prime q ∧ q.factor 341 → q ≤ p) ∧ p = 17 :=
by sorry

end greatest_prime_factor_341_l233_233577


namespace f_is_periodic_l233_233951

noncomputable def f : ℝ → ℝ := sorry

def a : ℝ := sorry

axiom exists_a_gt_zero : a > 0

axiom functional_eq (x : ℝ) : f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)

theorem f_is_periodic : ∀ x : ℝ, f (x + 2 * a) = f x := sorry

end f_is_periodic_l233_233951


namespace std_dev_same_for_A_and_B_l233_233338

def data_sample_A : List ℝ := [82, 84, 84, 86, 86, 86, 88, 88, 88, 88]
def data_sample_B : List ℝ := data_sample_A.map (λ x, x + 2)

def std_dev (data : List ℝ) : ℝ :=
  let mean := data.sum / (data.length : ℝ)
  let variance := data.map (λ x, (x - mean) ^ 2).sum / (data.length : ℝ)
  real.sqrt variance

theorem std_dev_same_for_A_and_B :
  std_dev data_sample_A = std_dev data_sample_B :=
by
  -- proof will go here
  sorry

end std_dev_same_for_A_and_B_l233_233338


namespace arithmetic_progression_a_eq_1_l233_233886

theorem arithmetic_progression_a_eq_1 
  (a : ℝ) 
  (h1 : 6 + 2 * a - 1 = 10 + 5 * a - (6 + 2 * a)) : 
  a = 1 :=
by
  sorry

end arithmetic_progression_a_eq_1_l233_233886


namespace least_multiple_of_36_with_digit_product_multiple_of_9_l233_233614

def is_multiple_of_36 (n : ℕ) : Prop :=
  n % 36 = 0

def product_of_digits_multiple_of_9 (n : ℕ) : Prop :=
  ∃ d : List ℕ, (n = List.foldl (λ x y => x * 10 + y) 0 d) ∧ (List.foldl (λ x y => x * y) 1 d) % 9 = 0

theorem least_multiple_of_36_with_digit_product_multiple_of_9 : ∃ n : ℕ, is_multiple_of_36 n ∧ product_of_digits_multiple_of_9 n ∧ n = 36 :=
by
  sorry

end least_multiple_of_36_with_digit_product_multiple_of_9_l233_233614


namespace nonnegative_solutions_count_l233_233307

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.card = 1 := 
sorry

end nonnegative_solutions_count_l233_233307


namespace solve_inequality_l233_233426

theorem solve_inequality (a : ℝ) (h : 0 < a) : 
  (a > 1 → {x : ℝ | 0 < x ∧ x < 1/a ∨ x > a}) ∧
  (a = 1 → {x : ℝ | 0 < x ∧ x ≠ 1}) ∧
  (0 < a ∧ a < 1 → {x : ℝ | 0 < x ∧ x < a ∨ x > 1/a}) :=
begin
  sorry
end

end solve_inequality_l233_233426


namespace greatest_prime_factor_341_l233_233532

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l233_233532


namespace point_C_divides_AE_in_1_2_ratio_l233_233178

/-- Given a right-angled triangle ABC with ∠BCA = 90°, and a point D on the extension
of the hypotenuse BC such that the line AD is tangent to the circumscribed circle ω
of triangle ABC. The line AC intersects the circumscribed circle of triangle ABD at point E.
It turns out that the angle bisector of ∠ADE is tangent to the circle ω.
We need to prove that point C divides the segment AE in the ratio 1:2. -/
theorem point_C_divides_AE_in_1_2_ratio (A B C D E: Type)
  [IsRightTriangle A B C] 
  [Tangency : Tangent (Line.mk A D) (Circumcircle A B C)] 
  [CircumcircleIntersect : Intersect (Line.mk A C) (Circumcircle A B D) E]
  [AngleBisectorTangency : Tangent (AngleBisector (∠A D E)) (Circumcircle A B C)] :
  divides_segment_in_ratio C A E 1 2 :=
sorry

end point_C_divides_AE_in_1_2_ratio_l233_233178


namespace evaTotalMarksCorrect_l233_233742

-- Definition of marks scored by Eva in each subject across semesters
def evaMathsMarksSecondSemester : Nat := 80
def evaArtsMarksSecondSemester : Nat := 90
def evaScienceMarksSecondSemester : Nat := 90

def evaMathsMarksFirstSemester : Nat := evaMathsMarksSecondSemester + 10
def evaArtsMarksFirstSemester : Nat := evaArtsMarksSecondSemester - 15
def evaScienceMarksFirstSemester : Nat := evaScienceMarksSecondSemester - (evaScienceMarksSecondSemester / 3)

-- Total marks in each semester
def totalMarksFirstSemester : Nat := evaMathsMarksFirstSemester + evaArtsMarksFirstSemester + evaScienceMarksFirstSemester
def totalMarksSecondSemester : Nat := evaMathsMarksSecondSemester + evaArtsMarksSecondSemester + evaScienceMarksSecondSemester

-- Combined total
def evaTotalMarks : Nat := totalMarksFirstSemester + totalMarksSecondSemester

-- Statement to prove
theorem evaTotalMarksCorrect : evaTotalMarks = 485 := 
by
  -- This needs to be proved as per the conditions and calculations above
  sorry

end evaTotalMarksCorrect_l233_233742


namespace max_good_subset_size_l233_233382

open Finset

-- Define the set Sn
def S_n (n : ℕ) : Finset (Fin n →₀ ℤ) := 
  univ.filter (λ f, ∀ i, f i = 0 ∨ f i = 1)

-- Define the distance function d
def d {n : ℕ} (a b : Fin n →₀ ℤ) : ℤ := 
  ∑ i, abs (a i - b i)

-- Define good subset predicate
def good_subset {n : ℕ} (A : Finset (Fin n →₀ ℤ)) : Prop := 
  ∀ a b ∈ A, a ≠ b → d a b ≥ 2 ^ (n - 1)

-- Prove the maximum size of a good subset is 2^(n+1)
theorem max_good_subset_size (n : ℕ) (A : Finset (Fin n →₀ ℤ)) (h1: A ⊆ S_n n) (h2: good_subset A) : 
  A.card ≤ 2 ^ (n + 1) :=
sorry

end max_good_subset_size_l233_233382


namespace nonnegative_solutions_count_l233_233267

theorem nonnegative_solutions_count :
  {x : ℝ | x^2 = -6 * x ∧ 0 ≤ x}.finite.to_finset.card = 1 :=
by
  sorry

end nonnegative_solutions_count_l233_233267


namespace original_average_l233_233994

theorem original_average (A : ℝ) (h : (2 * (12 * A)) / 12 = 100) : A = 50 :=
by
  sorry

end original_average_l233_233994


namespace total_water_in_containers_l233_233019

/-
We have four containers. The first three contain water, while the fourth is empty. 
The second container holds twice as much water as the first, and the third holds twice as much water as the second. 
We transfer half of the water from the first container, one-third of the water from the second container, 
and one-quarter of the water from the third container into the fourth container. 
Now, there are 26 liters of water in the fourth container. Prove that initially, 
there were 84 liters of water in total in the first three containers.
-/

theorem total_water_in_containers (x : ℕ) (h1 : x / 2 + 2 * x / 3 + x = 26) : x + 2 * x + 4 * x = 84 := 
sorry

end total_water_in_containers_l233_233019


namespace smallest_blocks_needed_for_wall_l233_233668

noncomputable def smallest_number_of_blocks (wall_length : ℕ) (wall_height : ℕ) (block_length1 : ℕ) (block_length2 : ℕ) (block_length3 : ℝ) : ℕ :=
  let blocks_per_odd_row := wall_length / block_length1
  let blocks_per_even_row := wall_length / block_length1 - 1 + 2
  let odd_rows := wall_height / 2 + 1
  let even_rows := wall_height / 2
  odd_rows * blocks_per_odd_row + even_rows * blocks_per_even_row

theorem smallest_blocks_needed_for_wall :
  smallest_number_of_blocks 120 7 2 1 1.5 = 423 :=
by
  sorry

end smallest_blocks_needed_for_wall_l233_233668


namespace cos_angle_eq_l233_233170

noncomputable def angle_cos (P Q : ℝ × ℝ) : ℝ :=
let OP := (P.1, P.2)
let OQ := (Q.1, Q.2)
let dot_product := OP.1 * OQ.1 + OP.2 * OQ.2
let magnitude_OP := real.sqrt (OP.1 ^ 2 + OP.2 ^ 2)
let magnitude_OQ := real.sqrt (OQ.1 ^ 2 + OQ.2 ^ 2)
in dot_product / (magnitude_OP * magnitude_OQ)

theorem cos_angle_eq (P Q : ℝ × ℝ) (hP : P = (1, 1)) (hQ : Q = (-2, 1))
: angle_cos P Q = - (real.sqrt 10) / 10 :=
sorry

end cos_angle_eq_l233_233170


namespace correctness_of_statements_l233_233660

theorem correctness_of_statements (p q : Prop) (x y : ℝ) : 
  (¬ (p ∧ q) → (p ∨ q)) ∧
  ((xy = 0) → ¬(x^2 + y^2 = 0)) ∧
  ¬(∀ (L P : ℝ → ℝ), (∃ x, L x = P x) ↔ (∃ x, L x = P x ∧ ∀ x₁ x₂, x₁ ≠ x₂ → L x₁ ≠ P x₂)) →
  (0 + 1 + 0 = 1) :=
by
  sorry

end correctness_of_statements_l233_233660


namespace area_OBEC_l233_233495

noncomputable def E := (4, 4)
noncomputable def A := (16 / 3, 0)
noncomputable def B := (0, 16)
noncomputable def C := (6, 0)
noncomputable def O := (0, 0)

def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))

def area_quadrilateral (P Q R S : ℝ × ℝ) : ℝ :=
  area_triangle P Q R + area_triangle P R S

theorem area_OBEC :
  area_quadrilateral O B E C = 188 / 3 :=
by
  sorry

end area_OBEC_l233_233495


namespace intersection_A_B_l233_233220

open Set Real -- Opens necessary namespaces for sets and real numbers

-- Definitions for the sets A and B
def A : Set ℝ := {x | 1 / x < 1}
def B : Set ℝ := {x | x > -1}

-- The proof statement for the intersection of sets A and B
theorem intersection_A_B : A ∩ B = (Ioo (-1 : ℝ) 0) ∪ (Ioi 1) :=
by
  sorry -- Proof not included

end intersection_A_B_l233_233220


namespace max_radius_of_cylinder_in_crate_l233_233055

noncomputable def crate_side_length : ℝ := 20
def cylinder_height := crate_side_length
def max_diameter := crate_side_length

theorem max_radius_of_cylinder_in_crate :
  let r := max_diameter / 2 in r = 10 :=
by
  sorry

end max_radius_of_cylinder_in_crate_l233_233055


namespace calculate_arc_length_l233_233705

noncomputable def arc_length : ℝ :=
  ∫ x in 0..(π/6), (1 / Real.cos x)

theorem calculate_arc_length :
  arc_length = (1 / 2) * Real.log 3 :=
by
  sorry

end calculate_arc_length_l233_233705


namespace number_of_dogs_l233_233128

def legs_in_pool : ℕ := 24
def human_legs : ℕ := 4
def legs_per_dog : ℕ := 4

theorem number_of_dogs : (legs_in_pool - human_legs) / legs_per_dog = 5 :=
by
  sorry

end number_of_dogs_l233_233128


namespace minimum_dot_product_l233_233225

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 9) = 1

def K : (ℝ × ℝ) := (2, 0)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

theorem minimum_dot_product (M N : ℝ × ℝ) (hM : ellipse M.1 M.2) (hN : ellipse N.1 N.2) (h : dot_product (vector_sub M K) (vector_sub N K) = 0) :
  ∃ α β : ℝ, 0 ≤ α ∧ α < 2 * Real.pi ∧ 0 ≤ β ∧ β < 2 * Real.pi ∧ M = (6 * Real.cos α, 3 * Real.sin α) ∧ N = (6 * Real.cos β, 3 * Real.sin β) ∧
  (∃ C : ℝ, C = 23 / 3 ∧ ∀ M N, ellipse M.1 M.2 → ellipse N.1 N.2 → dot_product (vector_sub M K) (vector_sub N K) = 0 → dot_product (vector_sub M K) (vector_sub (vector_sub M N) K) >= C) :=
sorry

end minimum_dot_product_l233_233225
