import Mathlib

namespace largest_common_in_range_l1653_165396

-- Definitions for the problem's conditions
def first_seq (n : ℕ) : ℕ := 3 + 8 * n
def second_seq (m : ℕ) : ℕ := 5 + 9 * m

-- Statement of the theorem we are proving
theorem largest_common_in_range : 
  ∃ n m : ℕ, first_seq n = second_seq m ∧ 1 ≤ first_seq n ∧ first_seq n ≤ 200 ∧ first_seq n = 131 := by
  sorry

end largest_common_in_range_l1653_165396


namespace probability_not_green_l1653_165369

theorem probability_not_green :
  let red_balls := 6
  let yellow_balls := 3
  let black_balls := 4
  let green_balls := 5
  let total_balls := red_balls + yellow_balls + black_balls + green_balls
  let not_green_balls := red_balls + yellow_balls + black_balls
  total_balls = 18 ∧ not_green_balls = 13 → (not_green_balls : ℚ) / total_balls = 13 / 18 := 
by
  intros
  sorry

end probability_not_green_l1653_165369


namespace frustum_volume_and_lateral_surface_area_l1653_165315

theorem frustum_volume_and_lateral_surface_area (h : ℝ) 
    (A1 A2 : ℝ) (r R : ℝ) (V S_lateral : ℝ) : 
    A1 = 4 * Real.pi → 
    A2 = 25 * Real.pi → 
    h = 4 → 
    r = 2 → 
    R = 5 → 
    V = (1 / 3) * (A1 + A2 + Real.sqrt (A1 * A2)) * h → 
    S_lateral = Real.pi * r * Real.sqrt (h ^ 2 + (R - r) ^ 2) + Real.pi * R * Real.sqrt (h ^ 2 + (R - r) ^ 2) → 
    V = 42 * Real.pi ∧ S_lateral = 35 * Real.pi := by
  sorry

end frustum_volume_and_lateral_surface_area_l1653_165315


namespace liquidX_percentage_l1653_165335

variable (wA wB : ℝ) (pA pB : ℝ) (mA mB : ℝ)

-- Conditions
def weightA : ℝ := 200
def weightB : ℝ := 700
def percentA : ℝ := 0.8
def percentB : ℝ := 1.8

-- The question and answer.
theorem liquidX_percentage :
  (percentA / 100 * weightA + percentB / 100 * weightB) / (weightA + weightB) * 100 = 1.58 := by
  sorry

end liquidX_percentage_l1653_165335


namespace find_line_equation_l1653_165368

theorem find_line_equation (a b : ℝ) :
  (2 * a + 3 * b = 0 ∧ a * b < 0) ↔ (3 * a - 2 * b = 0 ∨ a - b + 1 = 0) :=
by
  sorry

end find_line_equation_l1653_165368


namespace intersection_A_B_l1653_165316

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_A_B :
  A ∩ B = { 1, 3 } :=
sorry

end intersection_A_B_l1653_165316


namespace consecutive_green_balls_l1653_165319

theorem consecutive_green_balls : ∃ (fill_ways : ℕ), fill_ways = 21 ∧ 
  (∃ (boxes : Fin 6 → Bool), 
    (∀ i, boxes i = true → 
      (∀ j, boxes j = true → (i ≤ j ∨ j ≤ i)) ∧ 
      ∃ k, boxes k = true)) :=
by
  sorry

end consecutive_green_balls_l1653_165319


namespace bill_due_in_months_l1653_165380

theorem bill_due_in_months
  (TD : ℝ) (FV : ℝ) (R_annual : ℝ) (m : ℝ) 
  (h₀ : TD = 270)
  (h₁ : FV = 2520)
  (h₂ : R_annual = 16) :
  m = 9 :=
by
  sorry

end bill_due_in_months_l1653_165380


namespace avg_of_second_largest_and_second_smallest_is_eight_l1653_165397

theorem avg_of_second_largest_and_second_smallest_is_eight :
  ∀ (a b c d e : ℕ), 
  a + b + c + d + e = 40 → 
  a < b ∧ b < c ∧ c < d ∧ d < e →
  ((d + b) / 2 : ℕ) = 8 := 
by
  intro a b c d e hsum horder
  /- the proof goes here, but we use sorry to skip it -/
  sorry

end avg_of_second_largest_and_second_smallest_is_eight_l1653_165397


namespace value_of_x_l1653_165361

theorem value_of_x (x : ℝ) (h : x = 88 * 1.2) : x = 105.6 :=
by
  sorry

end value_of_x_l1653_165361


namespace sufficient_but_not_necessary_for_reciprocal_l1653_165342

theorem sufficient_but_not_necessary_for_reciprocal (x : ℝ) : (x > 1 → 1/x < 1) ∧ (¬ (1/x < 1 → x > 1)) :=
by
  sorry

end sufficient_but_not_necessary_for_reciprocal_l1653_165342


namespace find_savings_l1653_165389

noncomputable def savings (income expenditure : ℕ) : ℕ :=
  income - expenditure

theorem find_savings (I E : ℕ) (h_ratio : I = 9 * E) (h_income : I = 18000) : savings I E = 2000 :=
by
  sorry

end find_savings_l1653_165389


namespace evaluate_expression_l1653_165390

-- Defining the conditions and constants as per the problem statement
def factor_power_of_2 (n : ℕ) : ℕ :=
  if n % 8 = 0 then 3 else 0 -- Greatest power of 2 in 360
  
def factor_power_of_5 (n : ℕ) : ℕ :=
  if n % 5 = 0 then 1 else 0 -- Greatest power of 5 in 360

def expression (b a : ℕ) : ℚ := (2 / 3)^(b - a)

noncomputable def target_value : ℚ := 9 / 4

theorem evaluate_expression : expression (factor_power_of_5 360) (factor_power_of_2 360) = target_value := 
  by
    sorry

end evaluate_expression_l1653_165390


namespace find_a_for_chord_length_l1653_165391

theorem find_a_for_chord_length :
  ∀ a : ℝ, ((∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 1 ∧ (2 * x - y + a = 0)) 
  → ((2 * 1 - 1 + a = 0) → a = -1)) :=
by
  sorry

end find_a_for_chord_length_l1653_165391


namespace sequence_formula_correct_l1653_165399

-- Define the sequence S_n
def S (n : ℕ) : ℤ := n^2 - 2

-- Define the general term of the sequence a_n
def a (n : ℕ) : ℤ :=
  if n = 1 then -1 else 2 * n - 1

-- Theorem to prove that for the given S_n, the defined a_n is correct
theorem sequence_formula_correct (n : ℕ) (h : n > 0) : 
  a n = if n = 1 then -1 else S n - S (n - 1) :=
by sorry

end sequence_formula_correct_l1653_165399


namespace equiv_proof_problem_l1653_165388

theorem equiv_proof_problem (b c : ℝ) (h1 : b ≠ 1 ∨ c ≠ 1) (h2 : ∃ n : ℝ, b = 1 + n ∧ c = 1 + 2 * n) (h3 : b * 1 = c * c) : 
  100 * (b - c) = 75 := 
by sorry

end equiv_proof_problem_l1653_165388


namespace sum_reciprocals_eq_three_l1653_165385

-- Define nonzero real numbers x and y with their given condition
variables (x y : ℝ) (hx : x ≠ 0) (hy: y ≠ 0) (h : x + y = 3 * x * y)

-- State the theorem to prove the sum of reciprocals of x and y is 3
theorem sum_reciprocals_eq_three (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) : (1 / x) + (1 / y) = 3 :=
sorry

end sum_reciprocals_eq_three_l1653_165385


namespace evaluate_polynomial_at_4_l1653_165332

noncomputable def polynomial_horner (x : ℤ) : ℤ :=
  (((((3 * x + 6) * x - 20) * x - 8) * x + 15) * x + 9)

theorem evaluate_polynomial_at_4 :
  polynomial_horner 4 = 3269 :=
by
  sorry

end evaluate_polynomial_at_4_l1653_165332


namespace root_interval_l1653_165378

noncomputable def f (x : ℝ) : ℝ := 3^x + 3 * x - 8

theorem root_interval (h₁ : f 1 < 0) (h₂ : f 1.5 > 0) (h₃ : f 1.25 < 0) (h₄ : f 2 > 0) :
  ∃ x, 1.25 < x ∧ x < 1.5 ∧ f x = 0 :=
sorry

end root_interval_l1653_165378


namespace simplify_fraction_l1653_165353

theorem simplify_fraction :
  (18 / 462) + (35 / 77) = 38 / 77 := 
by sorry

end simplify_fraction_l1653_165353


namespace max_cubes_fit_l1653_165381

theorem max_cubes_fit (L S : ℕ) (hL : L = 10) (hS : S = 2) : (L * L * L) / (S * S * S) = 125 := by
  sorry

end max_cubes_fit_l1653_165381


namespace compare_P_Q_l1653_165333

noncomputable def P : ℝ := Real.sqrt 7 - 1
noncomputable def Q : ℝ := Real.sqrt 11 - Real.sqrt 5

theorem compare_P_Q : P > Q :=
sorry

end compare_P_Q_l1653_165333


namespace team_leaders_lcm_l1653_165322

/-- Amanda, Brian, Carla, and Derek are team leaders rotating every
    5, 8, 10, and 12 weeks respectively. Given that this week they all are leading
    projects together, prove that they will all lead projects together again in 120 weeks. -/
theorem team_leaders_lcm :
  Nat.lcm (Nat.lcm 5 8) (Nat.lcm 10 12) = 120 := 
  by
  sorry

end team_leaders_lcm_l1653_165322


namespace arithmetic_sequence_x_value_l1653_165363

theorem arithmetic_sequence_x_value (x : ℝ) (a2 a1 d : ℝ)
  (h1 : a1 = 1 / 3)
  (h2 : a2 = x - 2)
  (h3 : d = 4 * x + 1 - a2)
  (h2_eq_d_a1 : a2 - a1 = d) : x = - (8 / 3) :=
by
  -- Proof yet to be completed
  sorry

end arithmetic_sequence_x_value_l1653_165363


namespace perimeter_of_square_l1653_165312

variable (s : ℝ) (side_length : ℝ)
def is_square_side_length_5 (s : ℝ) : Prop := s = 5
theorem perimeter_of_square (h: is_square_side_length_5 s) : 4 * s = 20 := sorry

end perimeter_of_square_l1653_165312


namespace find_some_number_l1653_165337

theorem find_some_number : 
  ∃ x : ℝ, 
  (6 + 9 * 8 / x - 25 = 5) ↔ (x = 3) :=
by 
  sorry

end find_some_number_l1653_165337


namespace nancy_potatoes_l1653_165379

theorem nancy_potatoes (sandy_potatoes total_potatoes : ℕ) (h1 : sandy_potatoes = 7) (h2 : total_potatoes = 13) :
    total_potatoes - sandy_potatoes = 6 :=
by
  sorry

end nancy_potatoes_l1653_165379


namespace count_sets_B_l1653_165329

open Set

def A : Set ℕ := {1, 2}

theorem count_sets_B (B : Set ℕ) (h1 : A ∪ B = {1, 2, 3}) : 
  (∃ Bs : Finset (Set ℕ), ∀ b ∈ Bs, A ∪ b = {1, 2, 3} ∧ Bs.card = 4) := sorry

end count_sets_B_l1653_165329


namespace no_prime_numbers_divisible_by_91_l1653_165334

-- Define the concept of a prime number.
def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define the factors of 91.
def factors_of_91 (n : ℕ) : Prop :=
  n = 7 ∨ n = 13

-- State the problem formally: there are no prime numbers divisible by 91.
theorem no_prime_numbers_divisible_by_91 :
  ∀ p : ℕ, is_prime p → ¬ (91 ∣ p) :=
by
  intros p prime_p div91
  sorry

end no_prime_numbers_divisible_by_91_l1653_165334


namespace sum_sq_roots_cubic_l1653_165320

noncomputable def sum_sq_roots (r s t : ℝ) : ℝ :=
  r^2 + s^2 + t^2

theorem sum_sq_roots_cubic :
  ∀ r s t, (2 * r^3 + 3 * r^2 - 5 * r + 1 = 0) →
           (2 * s^3 + 3 * s^2 - 5 * s + 1 = 0) →
           (2 * t^3 + 3 * t^2 - 5 * t + 1 = 0) →
           (r + s + t = -3 / 2) →
           (r * s + r * t + s * t = 5 / 2) →
           sum_sq_roots r s t = -11 / 4 :=
by 
  intros r s t h₁ h₂ h₃ sum_roots prod_roots
  sorry

end sum_sq_roots_cubic_l1653_165320


namespace total_parents_surveyed_l1653_165370

-- Define the given conditions
def percent_agree : ℝ := 0.20
def percent_disagree : ℝ := 0.80
def disagreeing_parents : ℕ := 640

-- Define the statement to prove
theorem total_parents_surveyed :
  ∃ (total_parents : ℕ), disagreeing_parents = (percent_disagree * total_parents) ∧ total_parents = 800 :=
by
  sorry

end total_parents_surveyed_l1653_165370


namespace Dave_won_tickets_l1653_165350

theorem Dave_won_tickets :
  ∀ (tickets_toys tickets_clothes total_tickets : ℕ),
  (tickets_toys = 8) →
  (tickets_clothes = 18) →
  (tickets_clothes = tickets_toys + 10) →
  (total_tickets = tickets_toys + tickets_clothes) →
  total_tickets = 26 :=
by
  intros tickets_toys tickets_clothes total_tickets h1 h2 h3 h4
  have h5 : tickets_clothes = 8 + 10 := by sorry
  have h6 : tickets_clothes = 18 := by sorry
  have h7 : tickets_clothes = 18 := by sorry
  exact sorry

end Dave_won_tickets_l1653_165350


namespace four_digit_number_divisible_by_9_l1653_165346

theorem four_digit_number_divisible_by_9
    (a b c d e f g h i j : ℕ)
    (h₀ : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
               f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
               g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
               h ≠ i ∧ h ≠ j ∧
               i ≠ j )
    (h₁ : a + b + c + d + e + f + g + h + i + j = 45)
    (h₂ : 100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 * g + 100 * h + 10 * i + j) :
  ((1000 * g + 100 * h + 10 * i + j) % 9 = 0) := sorry

end four_digit_number_divisible_by_9_l1653_165346


namespace adoption_days_l1653_165331

theorem adoption_days (P0 P_in P_adopt_rate : Nat) (P_total : Nat) (hP0 : P0 = 3) (hP_in : P_in = 3) (hP_adopt_rate : P_adopt_rate = 3) (hP_total : P_total = P0 + P_in) :
  P_total / P_adopt_rate = 2 := 
by
  sorry

end adoption_days_l1653_165331


namespace simplify_expression_l1653_165351

variable (x y : ℝ)

theorem simplify_expression:
  3*x^2 - 3*(2*x^2 + 4*y) + 2*(x^2 - y) = -x^2 - 14*y := 
by 
  sorry

end simplify_expression_l1653_165351


namespace new_average_age_l1653_165373

theorem new_average_age (avg_age : ℕ) (num_people : ℕ) (leaving_age : ℕ) (remaining_people : ℕ) :
  avg_age = 40 →
  num_people = 8 →
  leaving_age = 25 →
  remaining_people = 7 →
  (avg_age * num_people - leaving_age) / remaining_people = 42 :=
by
  sorry

end new_average_age_l1653_165373


namespace balance_scale_measurements_l1653_165356

theorem balance_scale_measurements {a b c : ℕ}
    (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 11) :
    ∀ w : ℕ, 1 ≤ w ∧ w ≤ 11 → ∃ (x y z : ℤ), w = abs (x * a + y * b + z * c) :=
sorry

end balance_scale_measurements_l1653_165356


namespace factorization_correct_l1653_165372

theorem factorization_correct (m : ℤ) : m^2 - 1 = (m - 1) * (m + 1) :=
by {
  -- sorry, this is a place-holder for the proof.
  sorry
}

end factorization_correct_l1653_165372


namespace cubic_sum_l1653_165362

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 2) :
  a^3 + b^3 + c^3 = 26 :=
by
  sorry

end cubic_sum_l1653_165362


namespace Pete_latest_time_to_LA_l1653_165360

def minutesInHour := 60
def minutesOfWalk := 10
def minutesOfTrain := 80
def departureTime := 7 * minutesInHour + 30

def latestArrivalTime : Prop :=
  9 * minutesInHour = departureTime + minutesOfWalk + minutesOfTrain 

theorem Pete_latest_time_to_LA : latestArrivalTime :=
by
  sorry

end Pete_latest_time_to_LA_l1653_165360


namespace infinite_primes_of_the_year_2022_l1653_165341

theorem infinite_primes_of_the_year_2022 :
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p % 2 = 1 ∧ p ^ 2022 ∣ n ^ 2022 + 2022 :=
sorry

end infinite_primes_of_the_year_2022_l1653_165341


namespace common_number_is_eleven_l1653_165375

theorem common_number_is_eleven 
  (a b c d e f g h i : ℝ)
  (H1 : (a + b + c + d + e) / 5 = 7)
  (H2 : (e + f + g + h + i) / 5 = 10)
  (H3 : (a + b + c + d + e + f + g + h + i) / 9 = 74 / 9) :
  e = 11 := 
sorry

end common_number_is_eleven_l1653_165375


namespace f_of_5_eq_1_l1653_165301

noncomputable def f : ℝ → ℝ := sorry

theorem f_of_5_eq_1
    (h1 : ∀ x : ℝ, f (-x) = -f x)
    (h2 : ∀ x : ℝ, f (-x) + f (x + 3) = 0)
    (h3 : f (-1) = 1) :
    f 5 = 1 :=
sorry

end f_of_5_eq_1_l1653_165301


namespace days_per_week_equals_two_l1653_165309

-- Definitions based on conditions
def hourly_rate : ℕ := 10
def hours_per_delivery : ℕ := 3
def total_weeks : ℕ := 6
def total_earnings : ℕ := 360

-- Proof statement: determine the number of days per week Jamie delivers flyers is 2
theorem days_per_week_equals_two (d : ℕ) :
  10 * (total_weeks * d * hours_per_delivery) = total_earnings → d = 2 := by
  sorry

end days_per_week_equals_two_l1653_165309


namespace tv_power_consumption_l1653_165321

-- Let's define the problem conditions
def hours_per_day : ℕ := 4
def days_per_week : ℕ := 7
def weekly_cost : ℝ := 49              -- in cents
def cost_per_kwh : ℝ := 14             -- in cents

-- Define the theorem to prove the TV power consumption is 125 watts per hour
theorem tv_power_consumption : (weekly_cost / cost_per_kwh) / (hours_per_day * days_per_week) * 1000 = 125 :=
by
  sorry

end tv_power_consumption_l1653_165321


namespace range_of_k_l1653_165383

theorem range_of_k (k : ℝ) : (x^2 + k * y^2 = 2) ∧ (k > 0) ∧ (k < 1) ↔ (0 < k ∧ k < 1) :=
by
  sorry

end range_of_k_l1653_165383


namespace product_of_numbers_l1653_165366

theorem product_of_numbers (a b c m : ℚ) (h_sum : a + b + c = 240)
    (h_m_a : 6 * a = m) (h_m_b : m = b - 12) (h_m_c : m = c + 12) :
    a * b * c = 490108320 / 2197 :=
by 
  sorry

end product_of_numbers_l1653_165366


namespace tourists_count_l1653_165357

theorem tourists_count :
  ∃ (n : ℕ), (1 / 2 * n + 1 / 3 * n + 1 / 4 * n = 39) :=
by
  use 36
  sorry

end tourists_count_l1653_165357


namespace min_value_abs_b_minus_c_l1653_165354

-- Define the problem conditions
def condition1 (a b c : ℝ) : Prop :=
  (a - 2 * b - 1)^2 + (a - c - Real.log c)^2 = 0

-- Define the theorem to be proved
theorem min_value_abs_b_minus_c {a b c : ℝ} (h : condition1 a b c) : |b - c| = 1 :=
sorry

end min_value_abs_b_minus_c_l1653_165354


namespace func_translation_right_symm_yaxis_l1653_165327

def f (x : ℝ) : ℝ := sorry

theorem func_translation_right_symm_yaxis (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x - 1) = e ^ (-x)) :
  ∀ x, f x = e ^ (-x - 1) := sorry

end func_translation_right_symm_yaxis_l1653_165327


namespace books_sold_online_l1653_165325

theorem books_sold_online (X : ℤ) 
  (h1: 743 = 502 + (37 + X) + (74 + X + 34) - 160) : 
  X = 128 := 
by sorry

end books_sold_online_l1653_165325


namespace division_by_negative_divisor_l1653_165387

theorem division_by_negative_divisor : 15 / (-3) = -5 :=
by sorry

end division_by_negative_divisor_l1653_165387


namespace distance_between_stripes_l1653_165308

theorem distance_between_stripes
  (h1 : ∀ (curbs_are_parallel : Prop), curbs_are_parallel → true)
  (h2 : ∀ (distance_between_curbs : ℝ), distance_between_curbs = 60 → true)
  (h3 : ∀ (length_of_curb : ℝ), length_of_curb = 20 → true)
  (h4 : ∀ (stripe_length : ℝ), stripe_length = 75 → true) :
  ∃ (d : ℝ), d = 16 :=
by
  sorry

end distance_between_stripes_l1653_165308


namespace cranberry_juice_cost_l1653_165386

theorem cranberry_juice_cost 
  (cost_per_ounce : ℕ) (number_of_ounces : ℕ) 
  (h1 : cost_per_ounce = 7) 
  (h2 : number_of_ounces = 12) : 
  cost_per_ounce * number_of_ounces = 84 := 
by 
  sorry

end cranberry_juice_cost_l1653_165386


namespace find_B_inter_complement_U_A_l1653_165323

-- Define Universal set U
def U : Set ℤ := {-1, 0, 1, 2, 3, 4}

-- Define Set A
def A : Set ℤ := {2, 3}

-- Define complement of A relative to U
def complement_U_A : Set ℤ := U \ A

-- Define set B
def B : Set ℤ := {1, 4}

-- The goal to prove
theorem find_B_inter_complement_U_A : B ∩ complement_U_A = {1, 4} :=
by 
  have h1 : A = {2, 3} := rfl
  have h2 : U = {-1, 0, 1, 2, 3, 4} := rfl
  have h3 : B = {1, 4} := rfl
  sorry

end find_B_inter_complement_U_A_l1653_165323


namespace waiter_earnings_l1653_165305

theorem waiter_earnings (total_customers tipping_customers no_tip_customers tips_each : ℕ) (h1 : total_customers = 7) (h2 : no_tip_customers = 4) (h3 : tips_each = 9) (h4 : tipping_customers = total_customers - no_tip_customers) :
  tipping_customers * tips_each = 27 :=
by sorry

end waiter_earnings_l1653_165305


namespace janice_trash_fraction_l1653_165313

noncomputable def janice_fraction : ℚ :=
  let homework := 30
  let cleaning := homework / 2
  let walking_dog := homework + 5
  let total_tasks := homework + cleaning + walking_dog
  let total_time := 120
  let time_left := 35
  let time_spent := total_time - time_left
  let trash_time := time_spent - total_tasks
  trash_time / homework

theorem janice_trash_fraction : janice_fraction = 1 / 6 :=
by
  sorry

end janice_trash_fraction_l1653_165313


namespace base_number_min_sum_l1653_165302

theorem base_number_min_sum (a b : ℕ) (h₁ : 5 * a + 2 = 2 * b + 5) : a + b = 9 :=
by {
  -- this proof is skipped with sorry
  sorry
}

end base_number_min_sum_l1653_165302


namespace min_value_of_square_sum_l1653_165303

theorem min_value_of_square_sum (x y : ℝ) (h : (x-1)^2 + y^2 = 16) : ∃ (a : ℝ), a = x^2 + y^2 ∧ a = 9 :=
by 
  sorry

end min_value_of_square_sum_l1653_165303


namespace music_marks_l1653_165330

variable (M : ℕ) -- Variable to represent marks in music

/-- Conditions -/
def science_marks : ℕ := 70
def social_studies_marks : ℕ := 85
def total_marks : ℕ := 275
def physics_marks : ℕ := M / 2

theorem music_marks :
  science_marks + M + social_studies_marks + physics_marks M = total_marks → M = 80 :=
by
  sorry

end music_marks_l1653_165330


namespace range_of_a_l1653_165324

noncomputable def f (a b x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - b * x

theorem range_of_a (a b x : ℝ) (h1 : ∀ x > 0, (1/x) - a * x - b ≠ 0) (h2 : ∀ x > 0, x = 1 → (1/x) - a * x - b = 0) : 
  (1 - a) = b ∧ a > -1 :=
by
  sorry

end range_of_a_l1653_165324


namespace roots_subtraction_l1653_165358

theorem roots_subtraction (a b : ℝ) (h_roots : a * b = 20 ∧ a + b = 12) (h_order : a > b) : a - b = 8 :=
sorry

end roots_subtraction_l1653_165358


namespace percentage_increase_l1653_165394

theorem percentage_increase (d : ℝ) (v_current v_reduce v_increase t_reduce t_increase : ℝ) (h1 : d = 96)
  (h2 : v_current = 8) (h3 : v_reduce = v_current - 4) (h4 : t_reduce = d / v_reduce) 
  (h5 : t_increase = d / v_increase) (h6 : t_reduce = t_current + 16) (h7 : t_increase = t_current - 16) :
  (v_increase - v_current) / v_current * 100 = 50 := 
sorry

end percentage_increase_l1653_165394


namespace hexagon_same_length_probability_l1653_165338

noncomputable def hexagon_probability_same_length : ℚ :=
  let sides := 6
  let diagonals := 9
  let total_segments := sides + diagonals
  let probability_side_first := (sides : ℚ) / total_segments
  let probability_diagonal_first := (diagonals : ℚ) / total_segments
  let probability_second_side := (sides - 1 : ℚ) / (total_segments - 1)
  let probability_second_diagonal_same_length := 2 / (total_segments - 1)
  probability_side_first * probability_second_side + 
  probability_diagonal_first * probability_second_diagonal_same_length

theorem hexagon_same_length_probability : hexagon_probability_same_length = 11 / 35 := 
  sorry

end hexagon_same_length_probability_l1653_165338


namespace winner_more_than_third_l1653_165340

theorem winner_more_than_third (W S T F : ℕ) (h1 : F = 199) 
(h2 : W = F + 105) (h3 : W = S + 53) (h4 : W + S + T + F = 979) : 
W - T = 79 :=
by
  -- Here, the proof steps would go, but they are not required as per instructions.
  sorry

end winner_more_than_third_l1653_165340


namespace inequality_to_prove_l1653_165314

variable (x y z : ℝ)

axiom h1 : 0 ≤ x
axiom h2 : 0 ≤ y
axiom h3 : 0 ≤ z
axiom h4 : y * z + z * x + x * y = 1

theorem inequality_to_prove : x * (1 - y)^2 * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) ≤ (4 / 9) * Real.sqrt 3 :=
by 
  -- The proof is omitted.
  sorry

end inequality_to_prove_l1653_165314


namespace wheat_grains_approximation_l1653_165352

theorem wheat_grains_approximation :
  let total_grains : ℕ := 1536
  let wheat_per_sample : ℕ := 28
  let sample_size : ℕ := 224
  let wheat_estimate : ℕ := total_grains * wheat_per_sample / sample_size
  wheat_estimate = 169 := by
  sorry

end wheat_grains_approximation_l1653_165352


namespace selling_prices_max_profit_strategy_l1653_165364

theorem selling_prices (x y : ℕ) (hx : y - x = 30) (hy : 2 * x + 3 * y = 740) : x = 130 ∧ y = 160 :=
by
  sorry

theorem max_profit_strategy (m : ℕ) (hm : 20 ≤ m ∧ m ≤ 80) 
(hcost : 90 * m + 110 * (80 - m) ≤ 8400) : m = 20 ∧ (80 - m) = 60 :=
by
  sorry

end selling_prices_max_profit_strategy_l1653_165364


namespace johns_previous_salary_l1653_165395

-- Conditions
def johns_new_salary : ℝ := 70
def percent_increase : ℝ := 0.16666666666666664

-- Statement
theorem johns_previous_salary :
  ∃ x : ℝ, x + percent_increase * x = johns_new_salary ∧ x = 60 :=
by
  sorry

end johns_previous_salary_l1653_165395


namespace rectangular_prism_surface_area_l1653_165398

/-- The surface area of a rectangular prism with edge lengths 2, 3, and 4 is 52. -/
theorem rectangular_prism_surface_area :
  let a := 2
  let b := 3
  let c := 4
  2 * (a * b + a * c + b * c) = 52 :=
by
  let a := 2
  let b := 3
  let c := 4
  show 2 * (a * b + a * c + b * c) = 52
  sorry

end rectangular_prism_surface_area_l1653_165398


namespace least_positive_integer_to_add_l1653_165347

theorem least_positive_integer_to_add (n : ℕ) (h_start : n = 525) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 ∧ k = 4 :=
by {
  sorry
}

end least_positive_integer_to_add_l1653_165347


namespace number_of_blocks_l1653_165359

theorem number_of_blocks (total_amount : ℕ) (gift_worth : ℕ) (workers_per_block : ℕ) (h1 : total_amount = 4000) (h2 : gift_worth = 4) (h3 : workers_per_block = 100) :
  (total_amount / gift_worth) / workers_per_block = 10 :=
by
-- This part will be proven later, hence using sorry for now
sorry

end number_of_blocks_l1653_165359


namespace smallest_number_of_marbles_l1653_165371

theorem smallest_number_of_marbles 
  (r w b bl n : ℕ) 
  (h : r + w + b + bl = n)
  (h1 : r * (r - 1) * (r - 2) * (r - 3) = 24 * w * b * (r * (r - 1) / 2))
  (h2 : r * (r - 1) * (r - 2) * (r - 3) = 24 * bl * b * (r * (r - 1) / 2))
  (h_no_neg : 4 ≤ r):
  n = 18 :=
sorry

end smallest_number_of_marbles_l1653_165371


namespace compare_numbers_l1653_165304

theorem compare_numbers : 222^2 < 22^22 ∧ 22^22 < 2^222 :=
by {
  sorry
}

end compare_numbers_l1653_165304


namespace larry_wins_prob_l1653_165348

def probability_larry_wins (pLarry pJulius : ℚ) : ℚ :=
  let r := (1 - pLarry) * (1 - pJulius)
  pLarry * (1 / (1 - r))

theorem larry_wins_prob : probability_larry_wins (2 / 3) (1 / 3) = 6 / 7 :=
by
  -- Definitions for probabilities
  let pLarry := 2 / 3
  let pJulius := 1 / 3
  have r := (1 - pLarry) * (1 - pJulius)
  have S := pLarry * (1 / (1 - r))
  -- Expected result
  have expected := 6 / 7
  -- Prove the result equals the expected
  sorry

end larry_wins_prob_l1653_165348


namespace simplify_fraction_l1653_165307

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 2 + 1) + 2 / (Real.sqrt 3 - 1))) = Real.sqrt 3 - Real.sqrt 2 :=
by
  sorry

end simplify_fraction_l1653_165307


namespace xy_square_value_l1653_165365

theorem xy_square_value (x y : ℝ) (h1 : x * (x + y) = 24) (h2 : y * (x + y) = 72) : (x + y)^2 = 96 :=
by
  sorry

end xy_square_value_l1653_165365


namespace determine_counterfeit_coin_l1653_165345

theorem determine_counterfeit_coin (wt_1 wt_2 wt_3 wt_5 : ℕ) (coin : ℕ) :
  (wt_1 = 1) ∧ (wt_2 = 2) ∧ (wt_3 = 3) ∧ (wt_5 = 5) ∧
  (coin = wt_1 ∨ coin = wt_2 ∨ coin = wt_3 ∨ coin = wt_5) ∧
  (coin ≠ 1 ∨ coin ≠ 2 ∨ coin ≠ 3 ∨ coin ≠ 5) → 
  ∃ (counterfeit : ℕ), (counterfeit = 1 ∨ counterfeit = 2 ∨ counterfeit = 3 ∨ counterfeit = 5) ∧ 
  (counterfeit ≠ 1 ∧ counterfeit ≠ 2 ∧ counterfeit ≠ 3 ∧ counterfeit ≠ 5) :=
by
  sorry

end determine_counterfeit_coin_l1653_165345


namespace ratio_y_x_l1653_165382

variable {c x y : ℝ}

-- Conditions stated as assumptions
theorem ratio_y_x (h1 : x = 0.80 * c) (h2 : y = 1.25 * c) : y / x = 25 / 16 :=
by
  sorry

end ratio_y_x_l1653_165382


namespace packs_of_red_bouncy_balls_l1653_165344

/-- Given the following conditions:
1. Kate bought 6 packs of yellow bouncy balls.
2. Each pack contained 18 bouncy balls.
3. Kate bought 18 more red bouncy balls than yellow bouncy balls.
Prove that the number of packs of red bouncy balls Kate bought is 7. -/
theorem packs_of_red_bouncy_balls (packs_yellow : ℕ) (balls_per_pack : ℕ) (extra_red_balls : ℕ)
  (h1 : packs_yellow = 6)
  (h2 : balls_per_pack = 18)
  (h3 : extra_red_balls = 18)
  : (packs_yellow * balls_per_pack + extra_red_balls) / balls_per_pack = 7 :=
by
  sorry

end packs_of_red_bouncy_balls_l1653_165344


namespace find_y_plus_inv_y_l1653_165336

theorem find_y_plus_inv_y (y : ℝ) (h : y^3 + 1 / y^3 = 110) : y + 1 / y = 5 :=
sorry

end find_y_plus_inv_y_l1653_165336


namespace train_length_proof_l1653_165300

noncomputable def length_of_first_train (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (5 / 18) -- convert to m/s
  let total_distance := relative_speed * time
  total_distance - length2

theorem train_length_proof (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) :
  speed1 = 120 →
  speed2 = 80 →
  time = 9 →
  length2 = 270.04 →
  length_of_first_train speed1 speed2 time length2 = 230 :=
by
  intros h1 h2 h3 h4
  -- Use the defined function and simplify
  rw [h1, h2, h3, h4]
  simp [length_of_first_train]
  sorry

end train_length_proof_l1653_165300


namespace arithmetic_expression_result_l1653_165355

theorem arithmetic_expression_result :
  (24 / (8 + 2 - 5)) * 7 = 33.6 :=
by
  sorry

end arithmetic_expression_result_l1653_165355


namespace largest_triangle_angle_l1653_165306

theorem largest_triangle_angle (y : ℝ) (h1 : 45 + 60 + y = 180) : y = 75 :=
by { sorry }

end largest_triangle_angle_l1653_165306


namespace vanya_first_place_l1653_165339

theorem vanya_first_place {n : ℕ} {E A : Finset ℕ} (e_v : ℕ) (a_v : ℕ)
  (he_v : e_v = n)
  (h_distinct_places : E.card = (E ∪ A).card)
  (h_all_worse : ∀ e_i ∈ E, e_i ≠ e_v → ∃ a_i ∈ A, a_i > e_i)
  : a_v = 1 := 
sorry

end vanya_first_place_l1653_165339


namespace cookie_cost_1_l1653_165392

theorem cookie_cost_1 (C : ℝ) 
  (h1 : ∀ c, c > 0 → 1.2 * c = c + 0.2 * c)
  (h2 : 50 * (1.2 * C) = 60) :
  C = 1 :=
by
  sorry

end cookie_cost_1_l1653_165392


namespace walking_rate_on_escalator_l1653_165318

theorem walking_rate_on_escalator 
  (escalator_speed person_time : ℝ) 
  (escalator_length : ℝ) 
  (h1 : escalator_speed = 12) 
  (h2 : person_time = 15) 
  (h3 : escalator_length = 210) 
  : (∃ v : ℝ, escalator_length = (v + escalator_speed) * person_time ∧ v = 2) :=
by
  use 2
  rw [h1, h2, h3]
  sorry

end walking_rate_on_escalator_l1653_165318


namespace opposite_of_B_is_I_l1653_165328

inductive Face
| A | B | C | D | E | F | G | H | I

open Face

def opposite_face (f : Face) : Face :=
  match f with
  | A => G
  | B => I
  | C => H
  | D => F
  | E => E
  | F => F
  | G => A
  | H => C
  | I => B

theorem opposite_of_B_is_I : opposite_face B = I :=
  by
    sorry

end opposite_of_B_is_I_l1653_165328


namespace prime_eq_sum_of_two_squares_l1653_165377

theorem prime_eq_sum_of_two_squares (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 4 = 1) : 
  ∃ a b : ℤ, p = a^2 + b^2 := 
sorry

end prime_eq_sum_of_two_squares_l1653_165377


namespace distinct_arrangements_BOOKKEEPER_l1653_165311

theorem distinct_arrangements_BOOKKEEPER :
  let n := 9
  let nO := 2
  let nK := 2
  let nE := 3
  ∃ arrangements : ℕ,
  arrangements = Nat.factorial n / (Nat.factorial nO * Nat.factorial nK * Nat.factorial nE) ∧
  arrangements = 15120 :=
by { sorry }

end distinct_arrangements_BOOKKEEPER_l1653_165311


namespace fifty_third_number_is_2_pow_53_l1653_165374

theorem fifty_third_number_is_2_pow_53 :
  ∀ n : ℕ, (n = 53) → ∃ seq : ℕ → ℕ, (seq 1 = 2) ∧ (∀ k : ℕ, seq (k+1) = 2 * seq k) ∧ (seq n = 2 ^ 53) :=
  sorry

end fifty_third_number_is_2_pow_53_l1653_165374


namespace range_of_a_l1653_165326

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (1/2) * x^2

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → 0 < x₁ → 0 < x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) ≥ 2) ↔ (1 ≤ a) :=
by
  sorry

end range_of_a_l1653_165326


namespace f_neg1_plus_f_2_l1653_165393

def f (x : ℤ) : ℤ :=
  if x ≤ 0 then 4 * x else 2 * x

theorem f_neg1_plus_f_2 : f (-1) + f 2 = 0 := 
by
  -- Definition of f is provided above and conditions are met in that.
  sorry

end f_neg1_plus_f_2_l1653_165393


namespace snack_cost_is_five_l1653_165367

-- Define the cost of one ticket
def ticket_cost : ℕ := 18

-- Define the total number of people
def total_people : ℕ := 4

-- Define the total cost for tickets and snacks
def total_cost : ℕ := 92

-- Define the unknown cost of one set of snacks
def snack_cost := 92 - 4 * 18

-- Statement asserting that the cost of one set of snacks is $5
theorem snack_cost_is_five : snack_cost = 5 := by
  sorry

end snack_cost_is_five_l1653_165367


namespace tank_filling_time_l1653_165317

theorem tank_filling_time (p q r s : ℝ) (leakage : ℝ) :
  (p = 1 / 6) →
  (q = 1 / 12) →
  (r = 1 / 24) →
  (s = 1 / 18) →
  (leakage = -1 / 48) →
  (1 / (p + q + r + s + leakage) = 48 / 15.67) :=
by
  intros hp hq hr hs hleak
  rw [hp, hq, hr, hs, hleak]
  norm_num
  sorry

end tank_filling_time_l1653_165317


namespace max_b_minus_a_l1653_165343

theorem max_b_minus_a (a b : ℝ) (h_a: a < 0) (h_ineq: ∀ x : ℝ, (3 * x^2 + a) * (2 * x + b) ≥ 0) : 
b - a = 1 / 3 := 
sorry

end max_b_minus_a_l1653_165343


namespace total_students_l1653_165349

theorem total_students (ratio_boys : ℕ) (ratio_girls : ℕ) (num_girls : ℕ) 
  (h_ratio : ratio_boys = 8) (h_ratio_girls : ratio_girls = 5) (h_num_girls : num_girls = 175) : 
  ratio_boys * (num_girls / ratio_girls) + num_girls = 455 :=
by
  sorry

end total_students_l1653_165349


namespace GP_GQ_GR_proof_l1653_165384

open Real

noncomputable def GP_GQ_GR_sum (XY XZ YZ : ℝ) (G : (ℝ × ℝ × ℝ)) (P Q R : (ℝ × ℝ × ℝ)) : ℝ :=
  let GP := dist G P
  let GQ := dist G Q
  let GR := dist G R
  GP + GQ + GR

theorem GP_GQ_GR_proof (XY XZ YZ : ℝ) (hXY : XY = 4) (hXZ : XZ = 3) (hYZ : YZ = 5)
  (G P Q R : (ℝ × ℝ × ℝ))
  (GP := dist G P) (GQ := dist G Q) (GR := dist G R)
  (hG : GP_GQ_GR_sum XY XZ YZ G P Q R = GP + GQ + GR) :
  GP + GQ + GR = 47 / 15 :=
sorry

end GP_GQ_GR_proof_l1653_165384


namespace problem_statement_l1653_165376

theorem problem_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : x - Real.sqrt x ≤ y - 1 / 4 ∧ y - 1 / 4 ≤ x + Real.sqrt x) :
  y - Real.sqrt y ≤ x - 1 / 4 ∧ x - 1 / 4 ≤ y + Real.sqrt y :=
sorry

end problem_statement_l1653_165376


namespace eggs_sold_l1653_165310

/-- Define the notion of trays and eggs in this context -/
def trays_of_eggs : ℤ := 30

/-- Define the initial collection of trays by Haman -/
def initial_trays : ℤ := 10

/-- Define the number of trays dropped by Haman -/
def dropped_trays : ℤ := 2

/-- Define the additional trays that Haman's father told him to collect -/
def additional_trays : ℤ := 7

/-- Define the total eggs sold -/
def total_eggs_sold : ℤ :=
  (initial_trays - dropped_trays) * trays_of_eggs + additional_trays * trays_of_eggs

-- Theorem to prove the total eggs sold
theorem eggs_sold : total_eggs_sold = 450 :=
by 
  -- Insert proof here
  sorry

end eggs_sold_l1653_165310
