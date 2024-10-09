import Mathlib

namespace promotional_event_probabilities_l1715_171509

def P_A := 1 / 1000
def P_B := 1 / 100
def P_C := 1 / 20
def P_A_B_C := P_A + P_B + P_C
def P_A_B := P_A + P_B
def P_complement_A_B := 1 - P_A_B

theorem promotional_event_probabilities :
  P_A = 1 / 1000 ∧
  P_B = 1 / 100 ∧
  P_C = 1 / 20 ∧
  P_A_B_C = 61 / 1000 ∧
  P_complement_A_B = 989 / 1000 :=
by
  sorry

end promotional_event_probabilities_l1715_171509


namespace slices_per_person_l1715_171580

theorem slices_per_person
  (small_pizza_slices : ℕ)
  (large_pizza_slices : ℕ)
  (small_pizzas_purchased : ℕ)
  (large_pizzas_purchased : ℕ)
  (george_slices : ℕ)
  (bob_extra : ℕ)
  (susie_divisor : ℕ)
  (bill_slices : ℕ)
  (fred_slices : ℕ)
  (mark_slices : ℕ)
  (ann_slices : ℕ)
  (kelly_multiplier : ℕ) :
  small_pizza_slices = 4 →
  large_pizza_slices = 8 →
  small_pizzas_purchased = 4 →
  large_pizzas_purchased = 3 →
  george_slices = 3 →
  bob_extra = 1 →
  susie_divisor = 2 →
  bill_slices = 3 →
  fred_slices = 3 →
  mark_slices = 3 →
  ann_slices = 2 →
  kelly_multiplier = 2 →
  (2 * (small_pizzas_purchased * small_pizza_slices + large_pizzas_purchased * large_pizza_slices -
    (george_slices + (george_slices + bob_extra) + (george_slices + bob_extra) / susie_divisor +
     bill_slices + fred_slices + mark_slices + ann_slices + ann_slices * kelly_multiplier))) =
    (small_pizzas_purchased * small_pizza_slices + large_pizzas_purchased * large_pizza_slices -
    (george_slices + (george_slices + bob_extra) + (george_slices + bob_extra) / susie_divisor +
     bill_slices + fred_slices + mark_slices + ann_slices + ann_slices * kelly_multiplier)) :=
by
  sorry

end slices_per_person_l1715_171580


namespace point_on_curve_l1715_171586

theorem point_on_curve :
  let x := -3 / 4
  let y := 1 / 2
  x^2 = (y^2 - 1) ^ 2 :=
by
  sorry

end point_on_curve_l1715_171586


namespace solve_for_x_l1715_171582

theorem solve_for_x : ∃ x : ℚ, -3 * x - 8 = 4 * x + 3 ∧ x = -11 / 7 :=
by
  sorry

end solve_for_x_l1715_171582


namespace weight_of_replaced_sailor_l1715_171567

theorem weight_of_replaced_sailor (avg_increase : ℝ) (total_sailors : ℝ) (new_sailor_weight : ℝ) : 
  avg_increase = 1 ∧ total_sailors = 8 ∧ new_sailor_weight = 64 → 
  ∃ W, W = 56 :=
by
  intro h
  sorry

end weight_of_replaced_sailor_l1715_171567


namespace ratio_of_average_speeds_l1715_171549

-- Definitions based on the conditions
def distance_AB := 600 -- km
def distance_AC := 300 -- km
def time_Eddy := 3 -- hours
def time_Freddy := 3 -- hours

def speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

def speed_Eddy := speed distance_AB time_Eddy
def speed_Freddy := speed distance_AC time_Freddy

theorem ratio_of_average_speeds : (speed_Eddy / speed_Freddy) = 2 :=
by 
  -- Proof is skipped, so we use sorry
  sorry

end ratio_of_average_speeds_l1715_171549


namespace AM_GM_inequality_min_value_l1715_171539

theorem AM_GM_inequality_min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b)^2 + (b / c)^2 + (c / a)^2 ≥ 3 :=
by
  sorry

end AM_GM_inequality_min_value_l1715_171539


namespace max_sum_of_factors_l1715_171554

theorem max_sum_of_factors (A B C : ℕ) (h1 : A * B * C = 2310) (h2 : A ≠ B) (h3 : B ≠ C) (h4 : A ≠ C) (h5 : 0 < A) (h6 : 0 < B) (h7 : 0 < C) : 
  A + B + C ≤ 42 := 
sorry

end max_sum_of_factors_l1715_171554


namespace ratio_of_lemons_l1715_171543

theorem ratio_of_lemons :
  ∃ (L J E I : ℕ), 
  L = 5 ∧ 
  J = L + 6 ∧ 
  J = E / 3 ∧ 
  E = I / 2 ∧ 
  L + J + E + I = 115 ∧ 
  J / E = 1 / 3 :=
by
  sorry

end ratio_of_lemons_l1715_171543


namespace step_count_initial_l1715_171584

theorem step_count_initial :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (11 * y - x = 64) ∧ (10 * x + y = 26) :=
by
  sorry

end step_count_initial_l1715_171584


namespace number_of_zeros_of_f_l1715_171544

noncomputable def f (x : ℝ) : ℝ := 2^x - 3*x

theorem number_of_zeros_of_f : ∃ a b : ℝ, (f a = 0 ∧ f b = 0 ∧ a ≠ b) ∧ ∀ x : ℝ, f x = 0 → x = a ∨ x = b :=
sorry

end number_of_zeros_of_f_l1715_171544


namespace value_of_a_plus_d_l1715_171574

theorem value_of_a_plus_d (a b c d : ℕ) (h1 : a + b = 16) (h2 : b + c = 9) (h3 : c + d = 3) : a + d = 10 := 
by 
  sorry

end value_of_a_plus_d_l1715_171574


namespace first_divisor_exists_l1715_171583

theorem first_divisor_exists (m d : ℕ) :
  (m % d = 47) ∧ (m % 24 = 23) ∧ (d > 47) → d = 72 :=
by
  sorry

end first_divisor_exists_l1715_171583


namespace problem1_l1715_171520

theorem problem1 {a b c : ℝ} (h : a + b + c = 2) : a^2 + b^2 + c^2 + 2 * a * b * c < 2 :=
sorry

end problem1_l1715_171520


namespace fabric_amount_for_each_dress_l1715_171525

def number_of_dresses (total_hours : ℕ) (hours_per_dress : ℕ) : ℕ :=
  total_hours / hours_per_dress 

def fabric_per_dress (total_fabric : ℕ) (number_of_dresses : ℕ) : ℕ :=
  total_fabric / number_of_dresses

theorem fabric_amount_for_each_dress (total_fabric : ℕ) (hours_per_dress : ℕ) (total_hours : ℕ) :
  total_fabric = 56 ∧ hours_per_dress = 3 ∧ total_hours = 42 →
  fabric_per_dress total_fabric (number_of_dresses total_hours hours_per_dress) = 4 :=
by
  sorry

end fabric_amount_for_each_dress_l1715_171525


namespace a_pow_10_add_b_pow_10_eq_123_l1715_171577

variable (a b : ℕ) -- better as non-negative integers for sequence progression

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem a_pow_10_add_b_pow_10_eq_123 : a^10 + b^10 = 123 := by
  sorry

end a_pow_10_add_b_pow_10_eq_123_l1715_171577


namespace seventh_observation_l1715_171538

-- Definitions from the conditions
def avg_original (x : ℕ) := 13
def num_observations_original := 6
def total_original := num_observations_original * (avg_original 0) -- 6 * 13 = 78

def avg_new := 12
def num_observations_new := num_observations_original + 1 -- 7
def total_new := num_observations_new * avg_new -- 7 * 12 = 84

-- The proof goal statement
theorem seventh_observation : (total_new - total_original) = 6 := 
  by
    -- Placeholder for the proof
    sorry

end seventh_observation_l1715_171538


namespace sum_of_ages_l1715_171513

theorem sum_of_ages 
  (a1 a2 a3 : ℕ) 
  (h1 : a1 ≠ a2) 
  (h2 : a1 ≠ a3) 
  (h3 : a2 ≠ a3) 
  (h4 : 1 ≤ a1 ∧ a1 ≤ 9) 
  (h5 : 1 ≤ a2 ∧ a2 ≤ 9) 
  (h6 : 1 ≤ a3 ∧ a3 ≤ 9) 
  (h7 : a1 * a2 = 18) 
  (h8 : a3 * min a1 a2 = 28) : 
  a1 + a2 + a3 = 18 := 
sorry

end sum_of_ages_l1715_171513


namespace sum_largest_and_second_smallest_l1715_171506

-- Define the list of numbers
def numbers : List ℕ := [10, 11, 12, 13, 14]

-- Define a predicate to get the largest number
def is_largest (n : ℕ) : Prop := ∀ x ∈ numbers, x ≤ n

-- Define a predicate to get the second smallest number
def is_second_smallest (n : ℕ) : Prop :=
  ∃ a b, (a ∈ numbers ∧ b ∈ numbers ∧ a < b ∧ b < n ∧ ∀ x ∈ numbers, (x < a ∨ x > b))

-- The main goal: To prove that the sum of the largest number and the second smallest number is 25
theorem sum_largest_and_second_smallest : 
  ∃ l s, is_largest l ∧ is_second_smallest s ∧ l + s = 25 := 
sorry

end sum_largest_and_second_smallest_l1715_171506


namespace problem_l1715_171570

theorem problem (m n : ℚ) (h : m - n = -2/3) : 7 - 3 * m + 3 * n = 9 := 
by {
  -- Place a sorry here as we do not provide the proof 
  sorry
}

end problem_l1715_171570


namespace soldier_score_9_points_l1715_171587

-- Define the conditions and expected result in Lean 4
theorem soldier_score_9_points (shots : List ℕ) :
  shots.length = 10 ∧
  (∀ shot ∈ shots, shot = 7 ∨ shot = 8 ∨ shot = 9 ∨ shot = 10) ∧
  shots.count 10 = 4 ∧
  shots.sum = 90 →
  shots.count 9 = 3 :=
by 
  sorry

end soldier_score_9_points_l1715_171587


namespace pets_percentage_of_cats_l1715_171558

theorem pets_percentage_of_cats :
  ∀ (total_pets dogs as_percentage bunnies cats_percentage : ℕ),
    total_pets = 36 →
    dogs = total_pets * as_percentage / 100 →
    as_percentage = 25 →
    bunnies = 9 →
    cats_percentage = (total_pets - (dogs + bunnies)) * 100 / total_pets →
    cats_percentage = 50 :=
by
  intros total_pets dogs as_percentage bunnies cats_percentage
  sorry

end pets_percentage_of_cats_l1715_171558


namespace odd_function_increasing_function_l1715_171563

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (1 + Real.exp x) - 0.5

theorem odd_function (x : ℝ) : f (-x) = -f (x) :=
  by sorry

theorem increasing_function : ∀ x y : ℝ, x < y → f x < f y :=
  by sorry

end odd_function_increasing_function_l1715_171563


namespace average_of_remaining_two_numbers_l1715_171596

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95) 
  (h_avg_ab : (a + b) / 2 = 3.8) 
  (h_avg_cd : (c + d) / 2 = 3.85) :
  ((e + f) / 2) = 4.2 := 
by 
  sorry

end average_of_remaining_two_numbers_l1715_171596


namespace complex_number_equality_l1715_171536

theorem complex_number_equality (a b : ℂ) : a - b = 0 ↔ a = b := sorry

end complex_number_equality_l1715_171536


namespace range_of_a_l1715_171575

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem range_of_a (a : ℝ) (h : ∀ x, ∃ y, y = (3 : ℝ) * x^2 + 2 * a * x + (a + 6) ∧ (y = 0)) :
  (a < -3 ∨ a > 6) :=
by { sorry }

end range_of_a_l1715_171575


namespace subset_0_in_X_l1715_171564

def X : Set ℝ := {x | x > -1}

theorem subset_0_in_X : {0} ⊆ X :=
by
  sorry

end subset_0_in_X_l1715_171564


namespace smallest_k_l1715_171505

theorem smallest_k (m n k : ℤ) (h : 221 * m + 247 * n + 323 * k = 2001) (hk : k > 100) : 
∃ k', k' = 111 ∧ k' > 100 :=
by
  sorry

end smallest_k_l1715_171505


namespace smallest_prime_sum_of_five_distinct_primes_l1715_171561

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def distinct (a b c d e : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

theorem smallest_prime_sum_of_five_distinct_primes :
  ∃ a b c d e : ℕ, is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ distinct a b c d e ∧ (a + b + c + d + e = 43) ∧ is_prime 43 :=
sorry

end smallest_prime_sum_of_five_distinct_primes_l1715_171561


namespace raven_current_age_l1715_171518

variable (R P : ℕ) -- Raven's current age, Phoebe's current age
variable (h₁ : P = 10) -- Phoebe is currently 10 years old
variable (h₂ : R + 5 = 4 * (P + 5)) -- In 5 years, Raven will be 4 times as old as Phoebe

theorem raven_current_age : R = 55 := 
by
  -- h2: R + 5 = 4 * (P + 5)
  -- h1: P = 10
  sorry

end raven_current_age_l1715_171518


namespace series_value_is_correct_l1715_171515

noncomputable def check_series_value : ℚ :=
  let p : ℚ := 1859 / 84
  let q : ℚ := -1024 / 63
  let r : ℚ := 512 / 63
  let m : ℕ := 3907
  let n : ℕ := 84
  100 * m + n

theorem series_value_is_correct : check_series_value = 390784 := 
by 
  sorry

end series_value_is_correct_l1715_171515


namespace inv_3i_minus_2inv_i_eq_neg_inv_5i_l1715_171533

-- Define the imaginary unit i such that i^2 = -1
def i : ℂ := Complex.I
axiom i_square : i^2 = -1

-- Proof statement
theorem inv_3i_minus_2inv_i_eq_neg_inv_5i : (3 * i - 2 * (1 / i))⁻¹ = -i / 5 :=
by
  -- Replace these steps with the corresponding actual proofs
  sorry

end inv_3i_minus_2inv_i_eq_neg_inv_5i_l1715_171533


namespace flower_count_l1715_171599

theorem flower_count (R L T : ℕ) (h1 : R = L + 22) (h2 : R = T - 20) (h3 : L + R + T = 100) : R = 34 :=
by
  sorry

end flower_count_l1715_171599


namespace sufficient_but_not_necessary_condition_l1715_171552

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (|a - b^2| + |b - a^2| ≤ 1) → ((a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧ 
  ∃ (a b : ℝ), ((a - 1/2)^2 + (b - 1/2)^2 ≤ 3/2) ∧ ¬ (|a - b^2| + |b - a^2| ≤ 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1715_171552


namespace sum_of_factors_l1715_171590

theorem sum_of_factors (x y : ℕ) :
  let exp := (27 * x ^ 6 - 512 * y ^ 6)
  let factor1 := (3 * x ^ 2 - 8 * y ^ 2)
  let factor2 := (3 * x ^ 2 + 8 * y ^ 2)
  let factor3 := (9 * x ^ 4 - 24 * x ^ 2 * y ^ 2 + 64 * y ^ 4)
  let sum := 3 + (-8) + 3 + 8 + 9 + (-24) + 64
  (factor1 * factor2 * factor3 = exp) ∧ (sum = 55) := 
by
  sorry

end sum_of_factors_l1715_171590


namespace range_of_a_l1715_171510

theorem range_of_a {a : ℝ} (h : ∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) :
  a ≤ -1 ∧ a ≠ -2 := 
sorry

end range_of_a_l1715_171510


namespace positive_difference_between_loans_l1715_171594

noncomputable def loan_amount : ℝ := 12000

noncomputable def option1_interest_rate : ℝ := 0.08
noncomputable def option1_years_1 : ℕ := 3
noncomputable def option1_years_2 : ℕ := 9

noncomputable def option2_interest_rate : ℝ := 0.09
noncomputable def option2_years : ℕ := 12

noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate)^years

noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal + principal * rate * years

noncomputable def payment_at_year_3 : ℝ :=
  compound_interest loan_amount option1_interest_rate option1_years_1 / 3

noncomputable def remaining_balance_after_3_years : ℝ :=
  compound_interest loan_amount option1_interest_rate option1_years_1 - payment_at_year_3

noncomputable def total_payment_option1 : ℝ :=
  payment_at_year_3 + compound_interest remaining_balance_after_3_years option1_interest_rate option1_years_2

noncomputable def total_payment_option2 : ℝ :=
  simple_interest loan_amount option2_interest_rate option2_years

noncomputable def positive_difference : ℝ :=
  abs (total_payment_option1 - total_payment_option2)

theorem positive_difference_between_loans : positive_difference = 1731 := by
  sorry

end positive_difference_between_loans_l1715_171594


namespace common_chord_through_vertex_l1715_171589

-- Define the structure for the problem
def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x

def passes_through (x y x_f y_f : ℝ) : Prop := (x - x_f) * (x - x_f) + y * y = 0

noncomputable def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- The main statement to prove
theorem common_chord_through_vertex (p : ℝ)
  (A B C D : ℝ × ℝ)
  (hA : parabola A.snd A.fst p)
  (hB : parabola B.snd B.fst p)
  (hC : parabola C.snd C.fst p)
  (hD : parabola D.snd D.fst p)
  (hAB_f : passes_through A.fst A.snd (focus p).fst (focus p).snd)
  (hCD_f : passes_through C.fst C.snd (focus p).fst (focus p).snd) :
  ∃ k : ℝ, ∀ x y : ℝ, (x + k = 0) → (y + k = 0) :=
by sorry

end common_chord_through_vertex_l1715_171589


namespace total_hours_driven_l1715_171588

/-- Jade and Krista went on a road trip for 3 days. Jade drives 8 hours each day, and Krista drives 6 hours each day. Prove the total number of hours they drove altogether is 42. -/
theorem total_hours_driven (days : ℕ) (hours_jade_per_day : ℕ) (hours_krista_per_day : ℕ)
  (h1 : days = 3) (h2 : hours_jade_per_day = 8) (h3 : hours_krista_per_day = 6) :
  3 * 8 + 3 * 6 = 42 := 
by
  sorry

end total_hours_driven_l1715_171588


namespace sum_of_solutions_of_quadratic_eq_l1715_171569

theorem sum_of_solutions_of_quadratic_eq :
  (∀ x : ℝ, 5 * x^2 - 3 * x - 2 = 0) → (∀ a b : ℝ, a = 5 ∧ b = -3 → -b / a = 3 / 5) :=
by
  sorry

end sum_of_solutions_of_quadratic_eq_l1715_171569


namespace particular_solution_satisfies_l1715_171592

noncomputable def particular_solution (x : ℝ) : ℝ :=
  (1/3) * Real.exp (-4 * x) - (1/3) * Real.exp (2 * x) + (x ^ 2 + 3 * x) * Real.exp (2 * x)

def initial_conditions (f df : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧ df 0 = 1

def differential_equation (f df ddf : ℝ → ℝ) : Prop :=
  ∀ x, ddf x + 2 * df x - 8 * f x = (12 * x + 20) * Real.exp (2 * x)

theorem particular_solution_satisfies :
  ∃ C1 C2 : ℝ, initial_conditions (λ x => C1 * Real.exp (-4 * x) + C2 * Real.exp (2 * x) + particular_solution x) 
              (λ x => -4 * C1 * Real.exp (-4 * x) + 2 * C2 * Real.exp (2 * x) + (2 * x^2 + 8 * x + 3) * Real.exp (2 * x)) ∧ 
              differential_equation (λ x => C1 * Real.exp (-4 * x) + C2 * Real.exp (2 * x) + particular_solution x) 
                                  (λ x => -4 * C1 * Real.exp (-4 * x) + 2 * C2 * Real.exp (2 * x) + (2 * x^2 + 8 * x + 3) * Real.exp (2 * x)) 
                                  (λ x => 16 * C1 * Real.exp (-4 * x) + 4 * C2 * Real.exp (2 * x) + (4 * x^2 + 12 * x + 1) * Real.exp (2 * x)) :=
sorry

end particular_solution_satisfies_l1715_171592


namespace unique_rectangle_Q_l1715_171545

noncomputable def rectangle_Q_count (a : ℝ) :=
  let x := (3 * a) / 2
  let y := a / 2
  if x < 2 * a then 1 else 0

-- The main theorem
theorem unique_rectangle_Q (a : ℝ) (h : a > 0) :
  rectangle_Q_count a = 1 :=
sorry

end unique_rectangle_Q_l1715_171545


namespace sum_of_solutions_sum_of_possible_values_l1715_171503

theorem sum_of_solutions (y : ℝ) (h : y^2 = 81) : y = 9 ∨ y = -9 :=
sorry

theorem sum_of_possible_values (y : ℝ) (h : y^2 = 81) : (∀ x, x = 9 ∨ x = -9 → x = 9 ∨ x = -9 → x = 9 + (-9)) :=
by
  have y_sol : y = 9 ∨ y = -9 := sum_of_solutions y h
  sorry

end sum_of_solutions_sum_of_possible_values_l1715_171503


namespace cube_difference_positive_l1715_171566

theorem cube_difference_positive (a b : ℝ) (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cube_difference_positive_l1715_171566


namespace arithmetic_base_conversion_l1715_171553

-- We start with proving base conversions

def convert_base3_to_base10 (n : ℕ) : ℕ := 1 * (3^0) + 2 * (3^1) + 1 * (3^2)

def convert_base7_to_base10 (n : ℕ) : ℕ := 6 * (7^0) + 5 * (7^1) + 4 * (7^2) + 3 * (7^3)

def convert_base9_to_base10 (n : ℕ) : ℕ := 6 * (9^0) + 7 * (9^1) + 8 * (9^2) + 9 * (9^3)

-- Prove the main equality

theorem arithmetic_base_conversion:
  (2468 : ℝ) / convert_base3_to_base10 121 + convert_base7_to_base10 3456 - convert_base9_to_base10 9876 = -5857.75 :=
by
  have h₁ : convert_base3_to_base10 121 = 16 := by native_decide
  have h₂ : convert_base7_to_base10 3456 = 1266 := by native_decide
  have h₃ : convert_base9_to_base10 9876 = 7278 := by native_decide
  rw [h₁, h₂, h₃]
  sorry

end arithmetic_base_conversion_l1715_171553


namespace two_digit_numbers_division_condition_l1715_171550

theorem two_digit_numbers_division_condition {n x y q : ℕ} (h1 : 10 * x + y = n)
  (h2 : n % 6 = x)
  (h3 : n / 10 = 3) (h4 : n % 10 = y) :
  n = 33 ∨ n = 39 := 
sorry

end two_digit_numbers_division_condition_l1715_171550


namespace sufficient_not_necessary_condition_l1715_171573

theorem sufficient_not_necessary_condition (x : ℝ) : (x > 0 → |x| = x) ∧ (|x| = x → x ≥ 0) :=
by
  sorry

end sufficient_not_necessary_condition_l1715_171573


namespace factorize_x9_minus_512_l1715_171507

theorem factorize_x9_minus_512 : 
  ∀ (x : ℝ), x^9 - 512 = (x - 2) * (x^2 + 2 * x + 4) * (x^6 + 8 * x^3 + 64) := by
  intro x
  sorry

end factorize_x9_minus_512_l1715_171507


namespace ratio_of_gold_and_copper_l1715_171542

theorem ratio_of_gold_and_copper
  (G C : ℝ)
  (hG : G = 11)
  (hC : C = 5)
  (hA : (11 * G + 5 * C) / (G + C) = 8) : G = C :=
by
  sorry

end ratio_of_gold_and_copper_l1715_171542


namespace adults_collectively_ate_l1715_171571

theorem adults_collectively_ate (A : ℕ) (C : ℕ) (total_cookies : ℕ) (share : ℝ) (each_child_gets : ℕ)
  (hC : C = 4) (hTotal : total_cookies = 120) (hShare : share = 1/3) (hEachChild : each_child_gets = 20)
  (children_gets : ℕ) (hChildrenGets : children_gets = C * each_child_gets) :
  children_gets = (2/3 : ℝ) * total_cookies → (share : ℝ) * total_cookies = 40 :=
by
  -- Placeholder for simplified proof
  sorry

end adults_collectively_ate_l1715_171571


namespace abc_divides_sum_pow_31_l1715_171565

theorem abc_divides_sum_pow_31 (a b c : ℕ) 
  (h1 : a ∣ b^5)
  (h2 : b ∣ c^5)
  (h3 : c ∣ a^5) : 
  abc ∣ (a + b + c) ^ 31 := 
sorry

end abc_divides_sum_pow_31_l1715_171565


namespace tickets_sold_l1715_171537

theorem tickets_sold (S G : ℕ) (hG : G = 388) (h_total : 4 * S + 6 * G = 2876) :
  S + G = 525 := by
  sorry

end tickets_sold_l1715_171537


namespace scale_total_length_l1715_171579

/-- Defining the problem parameters. -/
def number_of_parts : ℕ := 5
def length_of_each_part : ℕ := 18

/-- Theorem stating the total length of the scale. -/
theorem scale_total_length : number_of_parts * length_of_each_part = 90 :=
by
  sorry

end scale_total_length_l1715_171579


namespace lengths_C_can_form_triangle_l1715_171504

-- Definition of sets of lengths
def lengths_A := (3, 6, 9)
def lengths_B := (3, 5, 9)
def lengths_C := (4, 6, 9)
def lengths_D := (2, 6, 4)

-- Triangle condition for a given set of lengths
def can_form_triangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Proof problem statement 
theorem lengths_C_can_form_triangle : can_form_triangle 4 6 9 :=
by
  sorry

end lengths_C_can_form_triangle_l1715_171504


namespace inradius_of_triangle_area_three_times_perimeter_l1715_171548

theorem inradius_of_triangle_area_three_times_perimeter (A p s r : ℝ) (h1 : A = 3 * p) (h2 : p = 2 * s) (h3 : A = r * s) (h4 : s ≠ 0) :
  r = 6 :=
sorry

end inradius_of_triangle_area_three_times_perimeter_l1715_171548


namespace slope_of_line_l1715_171576

theorem slope_of_line : ∀ (x y : ℝ), 4 * x - 7 * y = 28 → y = (4/7) * x - 4 :=
by
  sorry

end slope_of_line_l1715_171576


namespace no_students_unable_to_partner_l1715_171514

def students_males_females :=
  let males_6th_class1 : Nat := 17
  let females_6th_class1 : Nat := 13
  let males_6th_class2 : Nat := 14
  let females_6th_class2 : Nat := 18
  let males_6th_class3 : Nat := 15
  let females_6th_class3 : Nat := 17
  let males_7th_class : Nat := 22
  let females_7th_class : Nat := 20

  let total_males := males_6th_class1 + males_6th_class2 + males_6th_class3 + males_7th_class
  let total_females := females_6th_class1 + females_6th_class2 + females_6th_class3 + females_7th_class

  total_males == total_females

theorem no_students_unable_to_partner : students_males_females = true := by
  -- Skipping the proof
  sorry

end no_students_unable_to_partner_l1715_171514


namespace new_tax_rate_is_30_percent_l1715_171560

theorem new_tax_rate_is_30_percent
  (original_rate : ℝ)
  (annual_income : ℝ)
  (tax_saving : ℝ)
  (h1 : original_rate = 0.45)
  (h2 : annual_income = 48000)
  (h3 : tax_saving = 7200) :
  (100 * (original_rate * annual_income - tax_saving) / annual_income) = 30 := 
sorry

end new_tax_rate_is_30_percent_l1715_171560


namespace kilos_of_bananas_l1715_171508

-- Define the conditions
def initial_money := 500
def remaining_money := 426
def cost_per_kilo_potato := 2
def cost_per_kilo_tomato := 3
def cost_per_kilo_cucumber := 4
def cost_per_kilo_banana := 5
def kilos_potato := 6
def kilos_tomato := 9
def kilos_cucumber := 5

-- Total cost of potatoes, tomatoes, and cucumbers
def total_cost_vegetables : ℕ := 
  (kilos_potato * cost_per_kilo_potato) +
  (kilos_tomato * cost_per_kilo_tomato) +
  (kilos_cucumber * cost_per_kilo_cucumber)

-- Money spent on bananas
def money_spent_on_bananas : ℕ := initial_money - remaining_money - total_cost_vegetables

-- The proof problem statement
theorem kilos_of_bananas : money_spent_on_bananas / cost_per_kilo_banana = 14 :=
by
  -- The sorry is a placeholder for the proof
  sorry

end kilos_of_bananas_l1715_171508


namespace infinite_geometric_series_common_ratio_l1715_171529

theorem infinite_geometric_series_common_ratio
  (a S : ℝ)
  (h₁ : a = 500)
  (h₂ : S = 4000)
  (h₃ : S = a / (1 - (r : ℝ))) :
  r = 7 / 8 :=
by
  sorry

end infinite_geometric_series_common_ratio_l1715_171529


namespace gcd_lcm_252_l1715_171527

theorem gcd_lcm_252 {a b : ℕ} (h : Nat.gcd a b * Nat.lcm a b = 252) :
  ∃ S : Finset ℕ, S.card = 8 ∧ ∀ d ∈ S, d = Nat.gcd a b :=
by sorry

end gcd_lcm_252_l1715_171527


namespace glycerin_percentage_proof_l1715_171511

-- Conditions given in problem
def original_percentage : ℝ := 0.90
def original_volume : ℝ := 4
def added_volume : ℝ := 0.8

-- Total glycerin in original solution
def glycerin_amount : ℝ := original_percentage * original_volume

-- Total volume after adding water
def new_volume : ℝ := original_volume + added_volume

-- Desired percentage proof statement
theorem glycerin_percentage_proof : 
  (glycerin_amount / new_volume) * 100 = 75 := 
by
  sorry

end glycerin_percentage_proof_l1715_171511


namespace a8_eq_128_l1715_171595

-- Definitions of conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ} {q : ℝ}

-- Conditions
axiom a2_eq_2 : a 2 = 2
axiom a3_mul_a4_eq_32 : a 3 * a 4 = 32
axiom is_geometric : is_geometric_sequence a q

-- Statement to prove
theorem a8_eq_128 : a 8 = 128 :=
sorry

end a8_eq_128_l1715_171595


namespace john_change_received_is_7_l1715_171531

def cost_per_orange : ℝ := 0.75
def num_oranges : ℝ := 4
def amount_paid : ℝ := 10.0
def total_cost : ℝ := num_oranges * cost_per_orange
def change_received : ℝ := amount_paid - total_cost

theorem john_change_received_is_7 : change_received = 7 :=
by
  sorry

end john_change_received_is_7_l1715_171531


namespace carly_practice_backstroke_days_per_week_l1715_171557

theorem carly_practice_backstroke_days_per_week 
  (butterfly_hours_per_day : ℕ) 
  (butterfly_days_per_week : ℕ) 
  (backstroke_hours_per_day : ℕ) 
  (total_hours_per_month : ℕ)
  (weeks_per_month : ℕ)
  (d : ℕ)
  (h1 : butterfly_hours_per_day = 3)
  (h2 : butterfly_days_per_week = 4)
  (h3 : backstroke_hours_per_day = 2)
  (h4 : total_hours_per_month = 96)
  (h5 : weeks_per_month = 4)
  (h6 : total_hours_per_month - (butterfly_hours_per_day * butterfly_days_per_week * weeks_per_month) = backstroke_hours_per_day * d * weeks_per_month) :
  d = 6 := by
  sorry

end carly_practice_backstroke_days_per_week_l1715_171557


namespace probability_top_card_king_l1715_171530

theorem probability_top_card_king :
  let total_cards := 52
  let total_kings := 4
  let probability := total_kings / total_cards
  probability = 1 / 13 :=
by
  -- sorry to skip the proof
  sorry

end probability_top_card_king_l1715_171530


namespace yunkyung_work_per_day_l1715_171528

theorem yunkyung_work_per_day (T : ℝ) (h : T > 0) (H : T / 3 = 1) : T / 3 = 1/3 := 
by sorry

end yunkyung_work_per_day_l1715_171528


namespace total_time_to_fill_tank_l1715_171519

-- Definitions as per conditions
def tank_fill_time_for_one_tap (total_time : ℕ) : Prop :=
  total_time = 16

def number_of_taps_for_second_half (num_taps : ℕ) : Prop :=
  num_taps = 4

-- Theorem statement to prove the total time taken to fill the tank
theorem total_time_to_fill_tank : ∀ (time_one_tap time_total : ℕ),
  tank_fill_time_for_one_tap time_one_tap →
  number_of_taps_for_second_half 4 →
  time_total = 10 :=
by
  intros time_one_tap time_total h1 h2
  -- Proof needed here
  sorry

end total_time_to_fill_tank_l1715_171519


namespace ed_money_left_after_hotel_stay_l1715_171556

theorem ed_money_left_after_hotel_stay 
  (night_rate : ℝ) (morning_rate : ℝ) 
  (initial_money : ℝ) (hours_night : ℕ) (hours_morning : ℕ) 
  (remaining_money : ℝ) : 
  night_rate = 1.50 → morning_rate = 2.00 → initial_money = 80 → 
  hours_night = 6 → hours_morning = 4 → 
  remaining_money = 63 :=
by
  intros h1 h2 h3 h4 h5
  let cost_night := night_rate * hours_night
  let cost_morning := morning_rate * hours_morning
  let total_cost := cost_night + cost_morning
  let money_left := initial_money - total_cost
  sorry

end ed_money_left_after_hotel_stay_l1715_171556


namespace root_expression_value_l1715_171591

theorem root_expression_value 
  (m : ℝ) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2021 = 2024 := 
by 
  sorry

end root_expression_value_l1715_171591


namespace w_z_ratio_l1715_171501

theorem w_z_ratio (w z : ℝ) (h : (1/w + 1/z) / (1/w - 1/z) = 2023) : (w + z) / (w - z) = -2023 :=
by sorry

end w_z_ratio_l1715_171501


namespace clips_and_earnings_l1715_171512

variable (x y z : ℝ)
variable (h_y : y = x / 2)
variable (totalClips : ℝ := 48 * x + y)
variable (avgEarning : ℝ := z / totalClips)

theorem clips_and_earnings :
  totalClips = 97 * x / 2 ∧ avgEarning = 2 * z / (97 * x) :=
by
  sorry

end clips_and_earnings_l1715_171512


namespace problem_statement_l1715_171526

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def condition1 := a 1 = 1
def condition2 := (a 3 + a 4) / (a 1 + a 2) = 4
def increasing := q > 0

-- Definition of S_n
def sum_geom (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)

theorem problem_statement (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h_geom : increasing_geometric_sequence a q) 
  (h_condition1 : condition1 a) 
  (h_condition2 : condition2 a) 
  (h_increasing : increasing q)
  (h_sum_geom : sum_geom a q S) : 
  S 5 = 31 :=
sorry

end problem_statement_l1715_171526


namespace triangle_is_right_angled_l1715_171523

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

def is_right_angle_triangle (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  dot_product AB BC = 0

theorem triangle_is_right_angled :
  let A := { x := 2, y := 5 }
  let B := { x := 5, y := 2 }
  let C := { x := 10, y := 7 }
  is_right_angle_triangle A B C :=
by
  sorry

end triangle_is_right_angled_l1715_171523


namespace A_oplus_B_eq_l1715_171535

def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def symm_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M
def A : Set ℝ := {y | ∃ x:ℝ, y = 3^x}
def B : Set ℝ := {y | ∃ x:ℝ, y = -(x-1)^2 + 2}

theorem A_oplus_B_eq : symm_diff A B = {y | y ≤ 0} ∪ {y | y > 2} := by {
  sorry
}

end A_oplus_B_eq_l1715_171535


namespace div_inside_parentheses_l1715_171502

theorem div_inside_parentheses :
  100 / (6 / 2) = 100 / 3 :=
by
  sorry

end div_inside_parentheses_l1715_171502


namespace crossword_solution_correct_l1715_171540

noncomputable def vertical_2 := "счет"
noncomputable def vertical_3 := "евро"
noncomputable def vertical_4 := "доллар"
noncomputable def vertical_5 := "вклад"
noncomputable def vertical_6 := "золото"
noncomputable def vertical_7 := "ломбард"

noncomputable def horizontal_1 := "обмен"
noncomputable def horizontal_2 := "система"
noncomputable def horizontal_3 := "ломбард"

theorem crossword_solution_correct :
  (vertical_2 = "счет") ∧
  (vertical_3 = "евро") ∧
  (vertical_4 = "доллар") ∧
  (vertical_5 = "вклад") ∧
  (vertical_6 = "золото") ∧
  (vertical_7 = "ломбард") ∧
  (horizontal_1 = "обмен") ∧
  (horizontal_2 = "система") ∧
  (horizontal_3 = "ломбард") :=
by
  sorry

end crossword_solution_correct_l1715_171540


namespace set_intersection_complement_l1715_171524

open Set

noncomputable def A : Set ℝ := { x | abs (x - 1) > 2 }
noncomputable def B : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }
noncomputable def notA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
noncomputable def targetSet : Set ℝ := { x | 2 < x ∧ x ≤ 3 }

theorem set_intersection_complement :
  (notA ∩ B) = targetSet :=
  by
  sorry

end set_intersection_complement_l1715_171524


namespace boys_in_first_group_l1715_171546

theorem boys_in_first_group (x : ℕ) (h₁ : 5040 = 360 * x) : x = 14 :=
by {
  sorry
}

end boys_in_first_group_l1715_171546


namespace sum_123_consecutive_even_numbers_l1715_171568

theorem sum_123_consecutive_even_numbers :
  let n := 123
  let a := 2
  let d := 2
  let sum_arithmetic_series (n a l : ℕ) := n * (a + l) / 2
  let last_term := a + (n - 1) * d
  sum_arithmetic_series n a last_term = 15252 :=
by
  sorry

end sum_123_consecutive_even_numbers_l1715_171568


namespace radius_of_inscribed_circle_XYZ_l1715_171555

noncomputable def radius_of_inscribed_circle (XY XZ YZ : ℝ) : ℝ :=
  let s := (XY + XZ + YZ) / 2
  let area := Real.sqrt (s * (s - XY) * (s - XZ) * (s - YZ))
  let r := area / s
  r

theorem radius_of_inscribed_circle_XYZ :
  radius_of_inscribed_circle 26 15 17 = 2 * Real.sqrt 42 / 29 :=
by
  sorry

end radius_of_inscribed_circle_XYZ_l1715_171555


namespace total_pencils_correct_l1715_171517

def Mitchell_pencils := 30
def Antonio_pencils := Mitchell_pencils - 6
def total_pencils := Antonio_pencils + Mitchell_pencils

theorem total_pencils_correct : total_pencils = 54 := by
  sorry

end total_pencils_correct_l1715_171517


namespace Giovanni_burgers_l1715_171521

theorem Giovanni_burgers : 
  let toppings := 10
  let patty_choices := 4
  let topping_combinations := 2 ^ toppings
  let total_combinations := patty_choices * topping_combinations
  total_combinations = 4096 :=
by
  sorry

end Giovanni_burgers_l1715_171521


namespace taller_tree_height_l1715_171559

theorem taller_tree_height :
  ∀ (h : ℕ), 
    ∃ (h_s : ℕ), (h_s = h - 24) ∧ (5 * h = 7 * h_s) → h = 84 :=
by
  sorry

end taller_tree_height_l1715_171559


namespace cover_points_with_circles_l1715_171541

theorem cover_points_with_circles (n : ℕ) (points : Fin n → ℝ × ℝ)
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → min (dist (points i) (points j)) (min (dist (points j) (points k)) (dist (points i) (points k))) ≤ 1) :
  ∃ (a b : Fin n), ∀ (p : Fin n), dist (points p) (points a) ≤ 1 ∨ dist (points p) (points b) ≤ 1 := 
sorry

end cover_points_with_circles_l1715_171541


namespace allen_mother_age_l1715_171516

variable (A M : ℕ)

theorem allen_mother_age (h1 : A = M - 25) (h2 : (A + 3) + (M + 3) = 41) : M = 30 :=
by
  sorry

end allen_mother_age_l1715_171516


namespace diff_of_squares_l1715_171572

theorem diff_of_squares (x y : ℝ) (h1 : x + y = 5) (h2 : x - y = 10) : x^2 - y^2 = 50 := by
  sorry

end diff_of_squares_l1715_171572


namespace largest_angle_is_120_l1715_171547

variable (d e f : ℝ)
variable (h1 : d + 3 * e + 3 * f = d^2)
variable (h2 : d + 3 * e - 3 * f = -4)

theorem largest_angle_is_120 (h1 : d + 3 * e + 3 * f = d^2) (h2 : d + 3 * e - 3 * f = -4) : 
  ∃ (F : ℝ), F = 120 :=
by
  sorry

end largest_angle_is_120_l1715_171547


namespace find_original_number_l1715_171532

theorem find_original_number (x : ℝ)
  (h1 : 3 * (2 * x + 9) = 51) : x = 4 :=
sorry

end find_original_number_l1715_171532


namespace lower_limit_brother_opinion_l1715_171500

variables (w B : ℝ)

-- Conditions
-- Arun's weight is between 61 and 72 kg
def arun_cond := 61 < w ∧ w < 72
-- Arun's brother's opinion: greater than B, less than 70
def brother_cond := B < w ∧ w < 70
-- Arun's mother's view: not greater than 64
def mother_cond :=  w ≤ 64

-- Given the average
def avg_weight := 63

theorem lower_limit_brother_opinion (h_arun : arun_cond w) (h_brother: brother_cond w B) (h_mother: mother_cond w) (h_avg: avg_weight = (B + 64)/2) : 
  B = 62 :=
sorry

end lower_limit_brother_opinion_l1715_171500


namespace number_of_solutions_l1715_171597

theorem number_of_solutions :
  ∃ (x y z : ℝ), 
    (x = 4036 - 4037 * Real.sign (y - z)) ∧ 
    (y = 4036 - 4037 * Real.sign (z - x)) ∧ 
    (z = 4036 - 4037 * Real.sign (x - y)) :=
sorry

end number_of_solutions_l1715_171597


namespace min_sum_one_over_xy_l1715_171585

theorem min_sum_one_over_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 6) : 
  ∃ c, (∀ x y, (x > 0) → (y > 0) → (x + y = 6) → (c ≤ (1/x + 1/y))) ∧ (c = 2 / 3) :=
by 
  sorry

end min_sum_one_over_xy_l1715_171585


namespace who_is_who_l1715_171598

-- Define the types for inhabitants
inductive Inhabitant
| A : Inhabitant
| B : Inhabitant

-- Define the property of being a liar
def is_liar (x : Inhabitant) : Prop := 
  match x with
  | Inhabitant.A  => false -- Initial assumption, to be refined
  | Inhabitant.B  => false -- Initial assumption, to be refined

-- Define the statement made by A
def statement_by_A : Prop :=
  (is_liar Inhabitant.A ∧ ¬ is_liar Inhabitant.B)

-- The main theorem to prove
theorem who_is_who (h : ¬statement_by_A) :
  is_liar Inhabitant.A ∧ is_liar Inhabitant.B :=
by
  -- Proof goes here
  sorry

end who_is_who_l1715_171598


namespace max_consecutive_integers_lt_1000_l1715_171534

theorem max_consecutive_integers_lt_1000 : 
  ∃ n : ℕ, (n * (n + 1)) / 2 < 1000 ∧ ∀ m : ℕ, m > n → (m * (m + 1)) / 2 ≥ 1000 :=
sorry

end max_consecutive_integers_lt_1000_l1715_171534


namespace sector_max_area_l1715_171593

theorem sector_max_area (P : ℝ) (R l S : ℝ) :
  (P > 0) → (2 * R + l = P) → (S = 1/2 * R * l) →
  (R = P / 4) ∧ (S = P^2 / 16) :=
by
  sorry

end sector_max_area_l1715_171593


namespace initial_concentration_is_27_l1715_171578

-- Define given conditions
variables (m m_c : ℝ) -- initial mass of solution and salt
variables (x : ℝ) -- initial percentage concentration of salt
variables (h1 : m_c = (x / 100) * m) -- initial concentration definition
variables (h2 : m > 0) (h3 : x > 0) -- non-zero positive mass and concentration

theorem initial_concentration_is_27 (h_evaporated : (m / 5) * 2 * (x / 100) = m_c) 
  (h_new_concentration : (x + 3) = (m_c * 100) / (9 * m / 10)) 
  : x = 27 :=
by
  sorry

end initial_concentration_is_27_l1715_171578


namespace ratio_of_segments_l1715_171551

theorem ratio_of_segments (E F G H : ℝ) (h_collinear : E < F ∧ F < G ∧ G < H)
  (hEF : F - E = 3) (hFG : G - F = 6) (hEH : H - E = 20) : (G - E) / (H - F) = 9 / 17 := by
  sorry

end ratio_of_segments_l1715_171551


namespace minimum_value_of_z_l1715_171581

/-- Given the constraints: 
1. x - y + 5 ≥ 0,
2. x + y ≥ 0,
3. x ≤ 3,

Prove that the minimum value of z = (x + y + 2) / (x + 3) is 1/3.
-/
theorem minimum_value_of_z : 
  ∀ (x y : ℝ), 
    (x - y + 5 ≥ 0) ∧ 
    (x + y ≥ 0) ∧ 
    (x ≤ 3) → 
    ∃ (z : ℝ), 
      z = (x + y + 2) / (x + 3) ∧
      z = 1 / 3 :=
by
  intros x y h
  sorry

end minimum_value_of_z_l1715_171581


namespace theta_value_l1715_171522

theorem theta_value (theta : ℝ) (h1 : 0 ≤ theta ∧ theta ≤ 90)
    (h2 : Real.cos 60 = Real.cos 45 * Real.cos theta) : theta = 45 :=
  sorry

end theta_value_l1715_171522


namespace kaleb_initial_games_l1715_171562

-- Let n be the number of games Kaleb started out with
def initial_games (n : ℕ) : Prop :=
  let sold_games := 46
  let boxes := 6
  let games_per_box := 5
  n = sold_games + boxes * games_per_box

-- Now we state the theorem
theorem kaleb_initial_games : ∃ n, initial_games n ∧ n = 76 :=
  by sorry

end kaleb_initial_games_l1715_171562
