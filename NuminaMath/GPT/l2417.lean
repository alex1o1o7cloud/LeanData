import Mathlib

namespace combination_sum_l2417_241701

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Given conditions
axiom combinatorial_identity (n r : ℕ) : combination n r + combination n (r + 1) = combination (n + 1) (r + 1)

-- The theorem we aim to prove
theorem combination_sum : combination 8 2 + combination 8 3 + combination 9 2 = 120 := 
by
  sorry

end combination_sum_l2417_241701


namespace hens_on_farm_l2417_241709

theorem hens_on_farm (H R : ℕ) (h1 : H = 9 * R - 5) (h2 : H + R = 75) : H = 67 :=
by
  sorry

end hens_on_farm_l2417_241709


namespace min_value_l2417_241715

variables (a b c : ℝ)
variable (hpos : a > 0 ∧ b > 0 ∧ c > 0)
variable (hsum : a + b + c = 1)

theorem min_value (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hsum : a + b + c = 1) :
  9 * a^2 + 4 * b^2 + (1/4) * c^2 = 36 / 157 := 
sorry

end min_value_l2417_241715


namespace mean_yoga_practice_days_l2417_241721

noncomputable def mean_number_of_days (counts : List ℕ) (days : List ℕ) : ℚ :=
  let total_days := List.zipWith (λ c d => c * d) counts days |>.sum
  let total_students := counts.sum
  total_days / total_students

def counts : List ℕ := [2, 4, 5, 3, 2, 1, 3]
def days : List ℕ := [1, 2, 3, 4, 5, 6, 7]

theorem mean_yoga_practice_days : mean_number_of_days counts days = 37 / 10 := 
by 
  sorry

end mean_yoga_practice_days_l2417_241721


namespace right_triangle_hypotenuse_l2417_241789

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h₁ : a + b + c = 40) 
  (h₂ : a * b = 60) 
  (h₃ : a^2 + b^2 = c^2) : c = 18.5 := 
by 
  sorry

end right_triangle_hypotenuse_l2417_241789


namespace quadrilateral_trapezium_l2417_241725

theorem quadrilateral_trapezium (a b c d : ℝ) 
  (h1 : a / 6 = b / 7) 
  (h2 : b / 7 = c / 8) 
  (h3 : c / 8 = d / 9) 
  (h4 : a + b + c + d = 360) : 
  ((a + c = 180) ∨ (b + d = 180)) :=
by
  sorry

end quadrilateral_trapezium_l2417_241725


namespace find_initial_cookies_l2417_241773

-- Definitions based on problem conditions
def initial_cookies (x : ℕ) : Prop :=
  let after_eating := x - 2
  let after_buying := after_eating + 37
  after_buying = 75

-- Main statement to be proved
theorem find_initial_cookies : ∃ x, initial_cookies x ∧ x = 40 :=
by
  sorry

end find_initial_cookies_l2417_241773


namespace ratio_fenced_region_l2417_241794

theorem ratio_fenced_region (L W : ℝ) (k : ℝ) 
  (area_eq : L * W = 200)
  (fence_eq : 2 * W + L = 40)
  (mult_eq : L = k * W) :
  k = 2 :=
by
  sorry

end ratio_fenced_region_l2417_241794


namespace find_a_interval_l2417_241754

theorem find_a_interval :
  ∀ {a : ℝ}, (∃ b x y : ℝ, x = abs (y + a) + 4 / a ∧ x^2 + y^2 + 24 + b * (2 * y + b) = 10 * x) ↔ (a < 0 ∨ a ≥ 2 / 3) :=
by {
  sorry
}

end find_a_interval_l2417_241754


namespace annual_subscription_cost_l2417_241702

-- Definitions based on the conditions

def monthly_cost : ℝ := 10
def months_per_year : ℕ := 12
def discount_rate : ℝ := 0.20

-- The statement based on the correct answer
theorem annual_subscription_cost : 
  (monthly_cost * months_per_year) * (1 - discount_rate) = 96 := 
by
  sorry

end annual_subscription_cost_l2417_241702


namespace value_of_r_when_n_is_2_l2417_241719

-- Define the given conditions
def s : ℕ := 2 ^ 2 + 1
def r : ℤ := 3 ^ s - s

-- Prove that r equals 238 when n = 2
theorem value_of_r_when_n_is_2 : r = 238 := by
  sorry

end value_of_r_when_n_is_2_l2417_241719


namespace total_clips_correct_l2417_241768

def clips_in_april : ℕ := 48
def clips_in_may : ℕ := clips_in_april / 2
def total_clips : ℕ := clips_in_april + clips_in_may

theorem total_clips_correct : total_clips = 72 := by
  sorry

end total_clips_correct_l2417_241768


namespace possible_scenario_l2417_241713

variable {a b c d : ℝ}

-- Conditions
def abcd_positive : a * b * c * d > 0 := sorry
def a_less_than_c : a < c := sorry
def bcd_negative : b * c * d < 0 := sorry

-- Statement
theorem possible_scenario :
  (a < 0) ∧ (b > 0) ∧ (c < 0) ∧ (d > 0) :=
sorry

end possible_scenario_l2417_241713


namespace geom_arith_seq_first_term_is_two_l2417_241746

theorem geom_arith_seq_first_term_is_two (b q a d : ℝ) 
  (hq : q ≠ 1) 
  (h_geom_first : b = a + d) 
  (h_geom_second : b * q = a + 3 * d) 
  (h_geom_third : b * q^2 = a + 6 * d) 
  (h_prod : b * b * q * b * q^2 = 64) :
  b = 2 :=
by
  sorry

end geom_arith_seq_first_term_is_two_l2417_241746


namespace variance_of_numbers_l2417_241734

noncomputable def variance (s : List ℕ) : ℚ :=
  let mean := (s.sum : ℚ) / s.length
  let sqDiffs := s.map (λ n => (n - mean) ^ 2)
  sqDiffs.sum / s.length

def avg_is_34 (s : List ℕ) : Prop := (s.sum : ℚ) / s.length = 34

theorem variance_of_numbers (x : ℕ) 
  (h : avg_is_34 [31, 38, 34, 35, x]) : variance [31, 38, 34, 35, x] = 6 := 
by
  sorry

end variance_of_numbers_l2417_241734


namespace solve_for_x_l2417_241717

theorem solve_for_x (x : ℝ) : 
  2.5 * ((3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5)) = 2000.0000000000002 → 
  x = 2.5 :=
by 
  sorry

end solve_for_x_l2417_241717


namespace solution1_solution2_l2417_241762

noncomputable def problem1 (x : ℝ) : Prop :=
  4 * x^2 - 25 = 0

theorem solution1 (x : ℝ) : problem1 x ↔ x = 5 / 2 ∨ x = -5 / 2 :=
by sorry

noncomputable def problem2 (x : ℝ) : Prop :=
  (x + 1)^3 = -27

theorem solution2 (x : ℝ) : problem2 x ↔ x = -4 :=
by sorry

end solution1_solution2_l2417_241762


namespace constant_term_binomial_expansion_l2417_241744

theorem constant_term_binomial_expansion : ∃ T, (∀ x : ℝ, T = (2 * x - 1 / (2 * x)) ^ 6) ∧ T = -20 := 
by
  sorry

end constant_term_binomial_expansion_l2417_241744


namespace part1_part2_l2417_241738

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 1 / (x + a)

theorem part1 (a x : ℝ):
  a ≥ 1 → x > 0 → f x a ≥ 0 := 
sorry

theorem part2 (a : ℝ):
  0 < a ∧ a ≤ 2 / 3 → ∃! x, x > -a ∧ f x a = 0 :=
sorry

end part1_part2_l2417_241738


namespace correct_statement_l2417_241795

-- Define the conditions as assumptions

/-- Condition 1: To understand the service life of a batch of new energy batteries, a sampling survey can be used. -/
def condition1 : Prop := True

/-- Condition 2: If the probability of winning a lottery is 2%, then buying 50 of these lottery tickets at once will definitely win. -/
def condition2 : Prop := False

/-- Condition 3: If the average of two sets of data, A and B, is the same, SA^2=2.3, SB^2=4.24, then set B is more stable. -/
def condition3 : Prop := False

/-- Condition 4: Rolling a die with uniform density and getting a score of 0 is a certain event. -/
def condition4 : Prop := False

-- The main theorem to prove the correct statement is A
theorem correct_statement : condition1 = True ∧ condition2 = False ∧ condition3 = False ∧ condition4 = False :=
by
  constructor; repeat { try { exact True.intro }; try { exact False.elim (by sorry) } }

end correct_statement_l2417_241795


namespace factorize_m_l2417_241731

theorem factorize_m (m : ℝ) : m^2 - 4 * m - 5 = (m + 1) * (m - 5) := 
sorry

end factorize_m_l2417_241731


namespace area_of_remaining_figure_l2417_241765
noncomputable def π := Real.pi

theorem area_of_remaining_figure (R : ℝ) (chord_length : ℝ) (C : ℝ) 
  (h : chord_length = 8) (hC : C = R) : (π * R^2 - 2 * π * (R / 2)^2) = 12.57 := by
  sorry

end area_of_remaining_figure_l2417_241765


namespace min_abs_phi_l2417_241757

theorem min_abs_phi {f : ℝ → ℝ} (h : ∀ x, f x = 3 * Real.sin (2 * x + φ) ∧ ∀ x, f (x) = f (2 * π / 3 - x)) :
  |φ| = π / 6 :=
by
  sorry

end min_abs_phi_l2417_241757


namespace max_b_c_value_l2417_241714

theorem max_b_c_value (a b c : ℕ) (h1 : a > b) (h2 : a + b = 18) (h3 : c - b = 2) : b + c = 18 :=
sorry

end max_b_c_value_l2417_241714


namespace store_paid_price_l2417_241730

theorem store_paid_price (selling_price : ℕ) (less_amount : ℕ) 
(h1 : selling_price = 34) (h2 : less_amount = 8) : ∃ p : ℕ, p = selling_price - less_amount ∧ p = 26 := 
by
  sorry

end store_paid_price_l2417_241730


namespace bond_interest_percentage_l2417_241750

noncomputable def interest_percentage_of_selling_price (face_value interest_rate : ℝ) (selling_price : ℝ) : ℝ :=
  (face_value * interest_rate) / selling_price * 100

theorem bond_interest_percentage :
  let face_value : ℝ := 5000
  let interest_rate : ℝ := 0.07
  let selling_price : ℝ := 5384.615384615386
  interest_percentage_of_selling_price face_value interest_rate selling_price = 6.5 :=
by
  sorry

end bond_interest_percentage_l2417_241750


namespace paul_erasers_l2417_241726

theorem paul_erasers (E : ℕ) (E_crayons : E + 353 = 391) : E = 38 := 
by
  sorry

end paul_erasers_l2417_241726


namespace total_tea_consumption_l2417_241782

variables (S O P : ℝ)

theorem total_tea_consumption : 
  S + O = 11 →
  P + O = 15 →
  P + S = 13 →
  S + O + P = 19.5 :=
by
  intros h1 h2 h3
  sorry

end total_tea_consumption_l2417_241782


namespace smallest_class_size_l2417_241748

theorem smallest_class_size (n : ℕ) (h : 5 * n + 1 > 40) : ∃ k : ℕ, k >= 41 :=
by sorry

end smallest_class_size_l2417_241748


namespace temperature_difference_l2417_241778

theorem temperature_difference 
    (freezer_temp : ℤ) (room_temp : ℤ) (temperature_difference : ℤ) 
    (h1 : freezer_temp = -4) 
    (h2 : room_temp = 18) : 
    temperature_difference = room_temp - freezer_temp := 
by 
  sorry

end temperature_difference_l2417_241778


namespace compare_slopes_l2417_241792

noncomputable def f (p q r x : ℝ) := x^3 + p * x^2 + q * x + r

noncomputable def s (p q x : ℝ) := 3 * x^2 + 2 * p * x + q

theorem compare_slopes (p q r a b c : ℝ) (hb : b ≠ 0) (ha : a ≠ c) 
  (hfa : f p q r a = 0) (hfc : f p q r c = 0) : a > c → s p q a > s p q c := 
by
  sorry

end compare_slopes_l2417_241792


namespace parabola_vertex_l2417_241781

theorem parabola_vertex :
  (∃ h k, ∀ x, (x^2 - 2 = ((x - h) ^ 2) + k) ∧ (h = 0) ∧ (k = -2)) :=
by
  sorry

end parabola_vertex_l2417_241781


namespace find_m_l2417_241711

-- Definitions for the system of equations and the condition
def system_of_equations (x y m : ℝ) :=
  2 * x + 6 * y = 25 ∧ 6 * x + 2 * y = -11 ∧ x - y = m - 1

-- Statement to prove
theorem find_m (x y m : ℝ) (h : system_of_equations x y m) : m = -8 :=
  sorry

end find_m_l2417_241711


namespace age_of_second_replaced_man_l2417_241799

theorem age_of_second_replaced_man (avg_age_increase : ℕ) (avg_new_men_age : ℕ) (first_replaced_age : ℕ) (total_men : ℕ) (new_age_sum : ℕ) :
  avg_age_increase = 1 →
  avg_new_men_age = 34 →
  first_replaced_age = 21 →
  total_men = 12 →
  new_age_sum = 2 * avg_new_men_age →
  47 - (new_age_sum - (first_replaced_age + x)) = 12 →
  x = 35 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end age_of_second_replaced_man_l2417_241799


namespace smallest_four_digit_divisible_by_34_l2417_241783

/-- Define a four-digit number. -/
def is_four_digit (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000

/-- Define a number to be divisible by another number. -/
def divisible_by (n k : ℕ) : Prop :=
  k ∣ n

/-- Prove that the smallest four-digit number divisible by 34 is 1020. -/
theorem smallest_four_digit_divisible_by_34 : ∃ n : ℕ, is_four_digit n ∧ divisible_by n 34 ∧ 
    (∀ m : ℕ, is_four_digit m → divisible_by m 34 → n ≤ m) :=
  sorry

end smallest_four_digit_divisible_by_34_l2417_241783


namespace interval_sum_l2417_241712

theorem interval_sum (m n : ℚ) (h : ∀ x : ℚ, m < x ∧ x < n ↔ (mx - 1) / (x + 3) > 0) :
  m + n = -10 / 3 :=
sorry

end interval_sum_l2417_241712


namespace minimum_value_problem_l2417_241779

theorem minimum_value_problem (x y : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : x * y = 1 / 2) : 
  ∃ m : ℝ, m = 10 ∧ ∀ z, z = (2 / (1 - x) + 1 / (1 - y)) → z ≥ m :=
by
  sorry

end minimum_value_problem_l2417_241779


namespace at_least_one_not_less_than_two_l2417_241790

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  (a + 1 / b) ≥ 2 ∨ (b + 1 / c) ≥ 2 ∨ (c + 1 / a) ≥ 2 :=
sorry

end at_least_one_not_less_than_two_l2417_241790


namespace eq1_solution_eq2_solution_eq3_solution_eq4_solution_l2417_241753

-- Equation 1: 3x^2 - 2x - 1 = 0
theorem eq1_solution (x : ℝ) : 3 * x ^ 2 - 2 * x - 1 = 0 ↔ (x = -1/3 ∨ x = 1) :=
by sorry

-- Equation 2: (y + 1)^2 - 4 = 0
theorem eq2_solution (y : ℝ) : (y + 1) ^ 2 - 4 = 0 ↔ (y = 1 ∨ y = -3) :=
by sorry

-- Equation 3: t^2 - 6t - 7 = 0
theorem eq3_solution (t : ℝ) : t ^ 2 - 6 * t - 7 = 0 ↔ (t = 7 ∨ t = -1) :=
by sorry

-- Equation 4: m(m + 3) - 2m = 0
theorem eq4_solution (m : ℝ) : m * (m + 3) - 2 * m = 0 ↔ (m = 0 ∨ m = -1) :=
by sorry

end eq1_solution_eq2_solution_eq3_solution_eq4_solution_l2417_241753


namespace medicine_supply_duration_l2417_241756

noncomputable def pillDuration (numPills : ℕ) (pillFractionPerThreeDays : ℚ) : ℚ :=
  let pillPerDay := pillFractionPerThreeDays / 3
  let daysPerPill := 1 / pillPerDay
  numPills * daysPerPill

theorem medicine_supply_duration (numPills : ℕ) (pillFractionPerThreeDays : ℚ) (daysPerMonth : ℚ) :
  numPills = 90 →
  pillFractionPerThreeDays = 1 / 3 →
  daysPerMonth = 30 →
  pillDuration numPills pillFractionPerThreeDays / daysPerMonth = 27 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp [pillDuration]
  sorry

end medicine_supply_duration_l2417_241756


namespace sin_365_1_eq_m_l2417_241791

noncomputable def sin_value (θ : ℝ) : ℝ := Real.sin (Real.pi * θ / 180)
variables (m : ℝ) (h : sin_value 5.1 = m)

theorem sin_365_1_eq_m : sin_value 365.1 = m :=
by sorry

end sin_365_1_eq_m_l2417_241791


namespace negation_correct_l2417_241740

def original_statement (x : ℝ) : Prop := x > 0 → x^2 + 3 * x - 2 > 0

def negated_statement (x : ℝ) : Prop := x > 0 ∧ x^2 + 3 * x - 2 ≤ 0

theorem negation_correct : (¬ ∀ x, original_statement x) ↔ ∃ x, negated_statement x := by
  sorry

end negation_correct_l2417_241740


namespace only_set_B_is_right_angle_triangle_l2417_241728

def is_right_angle_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem only_set_B_is_right_angle_triangle :
  is_right_angle_triangle 3 4 5 ∧ ¬is_right_angle_triangle 1 2 2 ∧ ¬is_right_angle_triangle 3 4 9 ∧ ¬is_right_angle_triangle 4 5 7 :=
by
  -- proof steps omitted
  sorry

end only_set_B_is_right_angle_triangle_l2417_241728


namespace statement_A_statement_B_statement_C_statement_D_l2417_241751

variable (a b : ℝ)

-- Given conditions
axiom positive_a : 0 < a
axiom positive_b : 0 < b
axiom condition : a + 2 * b = 2 * a * b

-- Prove the statements
theorem statement_A : a + 2 * b ≥ 4 := sorry
theorem statement_B : ¬ (a + b ≥ 4) := sorry
theorem statement_C : ¬ (a * b ≤ 2) := sorry
theorem statement_D : a^2 + 4 * b^2 ≥ 8 := sorry

end statement_A_statement_B_statement_C_statement_D_l2417_241751


namespace cookout_2006_kids_l2417_241784

def kids_2004 : ℕ := 60
def kids_2005 : ℕ := kids_2004 / 2
def kids_2006 : ℕ := (2 * kids_2005) / 3

theorem cookout_2006_kids : kids_2006 = 20 := by
  sorry

end cookout_2006_kids_l2417_241784


namespace greatest_value_divisible_by_3_l2417_241736

theorem greatest_value_divisible_by_3 :
  ∃ (a : ℕ), (168026 + 1000 * a) % 3 = 0 ∧ a ≤ 9 ∧ ∀ b : ℕ, (168026 + 1000 * b) % 3 = 0 → b ≤ 9 → a ≥ b :=
sorry

end greatest_value_divisible_by_3_l2417_241736


namespace lastTwoNonZeroDigits_of_80_fact_is_8_l2417_241763

-- Define the factorial function
def fac : ℕ → ℕ
  | 0     => 1
  | (n+1) => (n+1) * fac n

-- Define the function to find the last two nonzero digits of a factorial
def lastTwoNonZeroDigits (n : ℕ) : ℕ := sorry -- Placeholder logic for now

-- State the problem as a theorem
theorem lastTwoNonZeroDigits_of_80_fact_is_8 :
  lastTwoNonZeroDigits 80 = 8 :=
sorry

end lastTwoNonZeroDigits_of_80_fact_is_8_l2417_241763


namespace percentage_decrease_in_selling_price_l2417_241797

theorem percentage_decrease_in_selling_price (S M : ℝ) 
  (purchase_price : S = 240 + M)
  (markup_percentage : M = 0.25 * S)
  (gross_profit : S - 16 = 304) : 
  (320 - 304) / 320 * 100 = 5 := 
by
  sorry

end percentage_decrease_in_selling_price_l2417_241797


namespace percent_of_motorists_receive_speeding_tickets_l2417_241764

theorem percent_of_motorists_receive_speeding_tickets
    (p_exceed : ℝ)
    (p_no_ticket : ℝ)
    (h1 : p_exceed = 0.125)
    (h2 : p_no_ticket = 0.20) : 
    (0.8 * p_exceed) * 100 = 10 :=
by
  sorry

end percent_of_motorists_receive_speeding_tickets_l2417_241764


namespace three_digit_number_equality_l2417_241747

theorem three_digit_number_equality :
  ∃ (x y z : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧
  (100 * x + 10 * y + z = x^2 + y + z^3) ∧
  (100 * x + 10 * y + z = 357) :=
by
  sorry

end three_digit_number_equality_l2417_241747


namespace inequality_solution_interval_l2417_241729

noncomputable def solve_inequality (x : ℝ) : Prop :=
  -2 < (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5) ∧
  (x^2 - 4 * x + 5) ≠ 0 ∧
  (3 * x^2 - 24 * x + 25) / (x^2 - 4 * x + 5) > 0 ∧
  (- x^2 - 8 * x + 5) / (x^2 - 4 * x + 5) < 0

theorem inequality_solution_interval (x : ℝ) :
  solve_inequality x :=
sorry

end inequality_solution_interval_l2417_241729


namespace number_is_165_l2417_241723

def is_between (n a b : ℕ) : Prop := a ≤ n ∧ n ≤ b
def is_odd (n : ℕ) : Prop := n % 2 = 1
def contains_digit_5 (n : ℕ) : Prop := ∃ k : ℕ, 10^k * 5 ≤ n ∧ n < 10^(k+1) * 5
def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem number_is_165 : 
  (is_between 165 144 169) ∧ 
  (is_odd 165) ∧ 
  (contains_digit_5 165) ∧ 
  (is_divisible_by_3 165) :=
by 
  sorry 

end number_is_165_l2417_241723


namespace constant_term_of_expansion_l2417_241745

theorem constant_term_of_expansion (x : ℝ) : 
  (∃ c : ℝ, c = 15 ∧ ∀ r : ℕ, r = 1 → (Nat.choose 5 r * 3^r * x^((5-5*r)/2) = c)) :=
by
  sorry

end constant_term_of_expansion_l2417_241745


namespace convex_2k_vertices_l2417_241798

theorem convex_2k_vertices (k : ℕ) (h1 : 2 ≤ k) (h2 : k ≤ 50)
    (P : Finset (EuclideanSpace ℝ (Fin 2)))
    (hP : P.card = 100) (M : Finset (EuclideanSpace ℝ (Fin 2)))
    (hM : M.card = k) : 
  ∃ V : Finset (EuclideanSpace ℝ (Fin 2)), V.card = 2 * k ∧ ∀ m ∈ M, m ∈ convexHull ℝ V :=
by
  sorry

end convex_2k_vertices_l2417_241798


namespace lcm_36_98_is_1764_l2417_241718

theorem lcm_36_98_is_1764 : Nat.lcm 36 98 = 1764 := by
  sorry

end lcm_36_98_is_1764_l2417_241718


namespace not_make_all_numbers_equal_l2417_241720

theorem not_make_all_numbers_equal (n : ℕ) (h : n ≥ 3)
  (a : Fin n → ℕ) (h1 : ∃ (i : Fin n), a i = 1 ∧ (∀ (j : Fin n), j ≠ i → a j = 0)) :
  ¬ ∃ x, ∀ i : Fin n, a i = x :=
by
  sorry

end not_make_all_numbers_equal_l2417_241720


namespace smallest_prime_factor_of_1917_l2417_241704

theorem smallest_prime_factor_of_1917 : ∃ p : ℕ, Prime p ∧ (p ∣ 1917) ∧ (∀ q : ℕ, Prime q ∧ (q ∣ 1917) → q ≥ p) :=
by
  sorry

end smallest_prime_factor_of_1917_l2417_241704


namespace point_D_not_in_region_l2417_241767

-- Define the condition that checks if a point is not in the region defined by 3x + 2y < 6
def point_not_in_region (x y : ℝ) : Prop :=
  ¬ (3 * x + 2 * y < 6)

-- Define the points
def A := (0, 0)
def B := (1, 1)
def C := (0, 2)
def D := (2, 0)

-- The proof problem as a Lean statement
theorem point_D_not_in_region : point_not_in_region (2:ℝ) (0:ℝ) :=
by
  show point_not_in_region 2 0
  sorry

end point_D_not_in_region_l2417_241767


namespace bricks_lay_calculation_l2417_241700

theorem bricks_lay_calculation (b c d : ℕ) (h1 : 0 < c) (h2 : 0 < d) : 
  ∃ y : ℕ, y = (b * (b + d) * (c + d))/(c * d) :=
sorry

end bricks_lay_calculation_l2417_241700


namespace f_derivative_at_1_intervals_of_monotonicity_l2417_241793

def f (x : ℝ) := x^3 - 3 * x^2 + 10
def f' (x : ℝ) := 3 * x^2 - 6 * x

theorem f_derivative_at_1 : f' 1 = -3 := by
  sorry

theorem intervals_of_monotonicity :
  (∀ x : ℝ, x < 0 → f' x > 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 2 → f' x < 0) ∧
  (∀ x : ℝ, x > 2 → f' x > 0) := by
  sorry

end f_derivative_at_1_intervals_of_monotonicity_l2417_241793


namespace team_matches_per_season_l2417_241775

theorem team_matches_per_season (teams_count total_games : ℕ) (h1 : teams_count = 50) (h2 : total_games = 4900) : 
  ∃ n : ℕ, n * (teams_count - 1) * teams_count / 2 = total_games ∧ n = 2 :=
by
  sorry

end team_matches_per_season_l2417_241775


namespace highest_weekly_sales_is_60_l2417_241769

/-- 
Given that a convenience store sold 300 bags of chips in a month,
and the following weekly sales pattern:
1. In the first week, 20 bags were sold.
2. In the second week, there was a 2-for-1 promotion, tripling the sales to 60 bags.
3. In the third week, a 10% discount doubled the sales to 40 bags.
4. In the fourth week, sales returned to the first week's number, 20 bags.
Prove that the number of bags of chips sold during the week with the highest demand is 60.
-/
theorem highest_weekly_sales_is_60 
  (total_sales : ℕ)
  (week1_sales : ℕ)
  (week2_sales : ℕ)
  (week3_sales : ℕ)
  (week4_sales : ℕ)
  (h_total : total_sales = 300)
  (h_week1 : week1_sales = 20)
  (h_week2 : week2_sales = 3 * week1_sales)
  (h_week3 : week3_sales = 2 * week1_sales)
  (h_week4 : week4_sales = week1_sales) :
  max (max week1_sales week2_sales) (max week3_sales week4_sales) = 60 := 
sorry

end highest_weekly_sales_is_60_l2417_241769


namespace slope_of_parallel_line_l2417_241710

theorem slope_of_parallel_line :
  ∀ (x y : ℝ), 3 * x - 6 * y = 12 → ∃ m : ℝ, m = 1 / 2 := 
by
  intros x y h
  sorry

end slope_of_parallel_line_l2417_241710


namespace percent_non_filler_l2417_241749

def burger_weight : ℕ := 120
def filler_weight : ℕ := 30

theorem percent_non_filler : 
  let total_weight := burger_weight
  let filler := filler_weight
  let non_filler := total_weight - filler
  (non_filler / total_weight : ℚ) * 100 = 75 := by
  sorry

end percent_non_filler_l2417_241749


namespace cos_product_l2417_241786

theorem cos_product : 
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) * (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12)) = 1 / 8 := 
by
  sorry

end cos_product_l2417_241786


namespace smallest_m_for_integral_solutions_l2417_241724

theorem smallest_m_for_integral_solutions : ∃ (m : ℕ), (∀ (p q : ℤ), (15 * (p * p) - m * p + 630 = 0 ∧ 15 * (q * q) - m * q + 630 = 0) → (m = 195)) :=
sorry

end smallest_m_for_integral_solutions_l2417_241724


namespace equal_real_roots_of_quadratic_l2417_241787

theorem equal_real_roots_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x + 3 = 0 ∧ 
               (∀ y : ℝ, 3 * y^2 - m * y + 3 = 0 → y = x)) → 
  m = 6 ∨ m = -6 :=
by
  sorry  -- proof to be filled in.

end equal_real_roots_of_quadratic_l2417_241787


namespace solve_inequality_l2417_241705

theorem solve_inequality (x : ℝ) :
  (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 9 * x - 6) ↔ (2 < x) ∧ (x < 3) := by
  sorry

end solve_inequality_l2417_241705


namespace circus_accommodation_l2417_241755

theorem circus_accommodation : 246 * 4 = 984 := by
  sorry

end circus_accommodation_l2417_241755


namespace line_slope_and_intersection_l2417_241727

theorem line_slope_and_intersection:
  (∀ x y : ℝ, x^2 + x / 4 + y / 5 = 1 → ∀ m : ℝ, m = -5 / 4) ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 → ¬ (x^2 + x / 4 + y / 5 = 1)) :=
by
  sorry

end line_slope_and_intersection_l2417_241727


namespace prove_temperature_on_Thursday_l2417_241703

def temperature_on_Thursday 
  (temps : List ℝ)   -- List of temperatures for 6 days.
  (avg : ℝ)          -- Average temperature for the week.
  (sum_six_days : ℝ) -- Sum of temperature readings for 6 days.
  (days : ℕ := 7)    -- Number of days in the week.
  (missing_day : ℕ := 1)  -- One missing day (Thursday).
  (thurs_temp : ℝ := 99.8) -- Temperature on Thursday to be proved.
: Prop := (avg * days) - sum_six_days = thurs_temp

theorem prove_temperature_on_Thursday 
  : temperature_on_Thursday [99.1, 98.2, 98.7, 99.3, 99, 98.9] 99 593.2 :=
by
  sorry

end prove_temperature_on_Thursday_l2417_241703


namespace ninety_seven_squared_l2417_241776

theorem ninety_seven_squared :
  97 * 97 = 9409 :=
by sorry

end ninety_seven_squared_l2417_241776


namespace proof_problem_l2417_241760

theorem proof_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a + b - 1 / (2 * a) - 2 / b = 3 / 2) :
  (a < 1 → b > 2) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y - 1 / (2 * x) - 2 / y = 3 / 2 → x + y ≥ 3) :=
by
  sorry

end proof_problem_l2417_241760


namespace find_x_l2417_241761

def vector := ℝ × ℝ

def a : vector := (1, 1)
def b (x : ℝ) : vector := (2, x)

def vector_add (u v : vector) : vector :=
(u.1 + v.1, u.2 + v.2)

def scalar_mul (k : ℝ) (v : vector) : vector :=
(k * v.1, k * v.2)

def vector_sub (u v : vector) : vector :=
(u.1 - v.1, u.2 - v.2)

def are_parallel (u v : vector) : Prop :=
∃ k : ℝ, u = scalar_mul k v

theorem find_x (x : ℝ) : are_parallel (vector_add a (b x)) (vector_sub (scalar_mul 4 (b x)) (scalar_mul 2 a)) → x = 2 :=
by
  sorry

end find_x_l2417_241761


namespace exist_distinct_xy_divisibility_divisibility_implies_equality_l2417_241758

-- Part (a)
theorem exist_distinct_xy_divisibility (n : ℕ) (h_n : n > 0) :
  ∃ (x y : ℕ), x ≠ y ∧ (∀ j : ℕ, 1 ≤ j ∧ j ≤ n → (x + j) ∣ (y + j)) :=
sorry

-- Part (b)
theorem divisibility_implies_equality (x y : ℕ) (h : ∀ j : ℕ, (x + j) ∣ (y + j)) : 
  x = y :=
sorry

end exist_distinct_xy_divisibility_divisibility_implies_equality_l2417_241758


namespace remainder_17_pow_2047_mod_23_l2417_241733

theorem remainder_17_pow_2047_mod_23 : (17 ^ 2047) % 23 = 11 := 
by
  sorry

end remainder_17_pow_2047_mod_23_l2417_241733


namespace find_a_l2417_241737

noncomputable def f' (x : ℝ) (a : ℝ) := 2 * x^3 + a * x^2 + x

theorem find_a (a : ℝ) (h : f' 1 a = 9) : a = 6 :=
by
  sorry

end find_a_l2417_241737


namespace cola_cost_l2417_241742

theorem cola_cost (h c : ℝ) (h1 : 3 * h + 2 * c = 360) (h2 : 2 * h + 3 * c = 390) : c = 90 :=
by
  sorry

end cola_cost_l2417_241742


namespace solve_fun_problem_l2417_241743

variable (f : ℝ → ℝ)

-- Definitions of the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_monotonic_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- The main theorem
theorem solve_fun_problem (h_even : is_even f) (h_monotonic : is_monotonic_on_pos f) :
  {x : ℝ | f (x + 1) = f (2 * x)} = {1, -1 / 3} := 
sorry

end solve_fun_problem_l2417_241743


namespace algebraic_expression_values_l2417_241780

-- Defining the given condition
def condition (x y : ℝ) : Prop :=
  x^4 + 6 * x^2 * y + 9 * y^2 + 2 * x^2 + 6 * y + 4 = 7

-- Defining the target expression
def target_expression (x y : ℝ) : ℝ :=
  x^4 + 6 * x^2 * y + 9 * y^2 - 2 * x^2 - 6 * y - 1

-- Stating the theorem to be proved
theorem algebraic_expression_values (x y : ℝ) (h : condition x y) :
  target_expression x y = -2 ∨ target_expression x y = 14 :=
by
  sorry

end algebraic_expression_values_l2417_241780


namespace smallest_distance_l2417_241785

open Real

/-- Let A be a point on the circle (x-3)^2 + (y-4)^2 = 16,
and let B be a point on the parabola x^2 = 8y.
The smallest possible distance AB is √34 - 4. -/
theorem smallest_distance 
  (A B : ℝ × ℝ)
  (hA : (A.1 - 3)^2 + (A.2 - 4)^2 = 16)
  (hB : (B.1)^2 = 8 * B.2) :
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ sqrt 34 - 4 := 
sorry

end smallest_distance_l2417_241785


namespace find_m_l2417_241708

def vector (α : Type) := α × α

noncomputable def dot_product {α} [Add α] [Mul α] (a b : vector α) : α :=
a.1 * b.1 + a.2 * b.2

theorem find_m (m : ℝ) (a : vector ℝ) (b : vector ℝ) (h₁ : a = (1, 2)) (h₂ : b = (m, 1)) (h₃ : dot_product a b = 0) : 
m = -2 :=
by
  sorry

end find_m_l2417_241708


namespace yellow_red_chair_ratio_l2417_241759

variable (Y B : ℕ)
variable (red_chairs : ℕ := 5)
variable (total_chairs : ℕ := 43)

-- Condition: There are 2 fewer blue chairs than yellow chairs
def blue_chairs_condition : Prop := B = Y - 2

-- Condition: Total number of chairs
def total_chairs_condition : Prop := red_chairs + Y + B = total_chairs

-- Prove the ratio of yellow chairs to red chairs is 4:1
theorem yellow_red_chair_ratio (h1 : blue_chairs_condition Y B) (h2 : total_chairs_condition Y B) :
  (Y / red_chairs) = 4 := 
sorry

end yellow_red_chair_ratio_l2417_241759


namespace pink_highlighters_count_l2417_241732

-- Define the necessary constants and types
def total_highlighters : ℕ := 12
def yellow_highlighters : ℕ := 2
def blue_highlighters : ℕ := 4

-- We aim to prove that the number of pink highlighters is 6
theorem pink_highlighters_count : ∃ (pink_highlighters : ℕ), 
  pink_highlighters = total_highlighters - (yellow_highlighters + blue_highlighters) ∧
  pink_highlighters = 6 :=
by
  sorry

end pink_highlighters_count_l2417_241732


namespace probability_A_not_lose_l2417_241707

theorem probability_A_not_lose (p_win p_draw : ℝ) (h_win : p_win = 0.3) (h_draw : p_draw = 0.5) :
  (p_win + p_draw = 0.8) :=
by
  rw [h_win, h_draw]
  norm_num

end probability_A_not_lose_l2417_241707


namespace gcd_840_1764_evaluate_polynomial_at_2_l2417_241722

-- Define the Euclidean algorithm steps and prove the gcd result
theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

-- Define the polynomial and evaluate it using Horner's method
def polynomial := λ x : ℕ => 2 * (x ^ 4) + 3 * (x ^ 3) + 5 * x - 4

theorem evaluate_polynomial_at_2 : polynomial 2 = 62 := by
  sorry

end gcd_840_1764_evaluate_polynomial_at_2_l2417_241722


namespace number_of_people_l2417_241777

-- Definitions based on conditions
def per_person_cost (x : ℕ) : ℕ :=
  if x ≤ 30 then 100 else max 72 (100 - 2 * (x - 30))

def total_cost (x : ℕ) : ℕ :=
  x * per_person_cost x

-- Main theorem statement
theorem number_of_people (x : ℕ) (h1 : total_cost x = 3150) (h2 : x > 30) : x = 35 :=
by {
  sorry
}

end number_of_people_l2417_241777


namespace train_speed_is_correct_l2417_241770

-- Define the conditions.
def length_of_train : ℕ := 1800 -- Length of the train in meters.
def time_to_cross_platform : ℕ := 60 -- Time to cross the platform in seconds (1 minute).

-- Define the statement that needs to be proved.
def speed_of_train : ℕ := (2 * length_of_train) / time_to_cross_platform

-- State the theorem.
theorem train_speed_is_correct :
  speed_of_train = 60 := by
  sorry -- Proof is not required.

end train_speed_is_correct_l2417_241770


namespace basil_has_winning_strategy_l2417_241739

-- Definitions based on conditions
def piles : Nat := 11
def stones_per_pile : Nat := 10
def peter_moves (n : Nat) := n = 1 ∨ n = 2 ∨ n = 3
def basil_moves (n : Nat) := n = 1 ∨ n = 2 ∨ n = 3

-- The main theorem to prove Basil has a winning strategy
theorem basil_has_winning_strategy 
  (total_stones : Nat := piles * stones_per_pile) 
  (peter_first : Bool := true) :
  exists winning_strategy_for_basil, 
    ∀ (piles_remaining : Nat) (sum_stones_remaining : Nat),
    sum_stones_remaining = piles_remaining * stones_per_pile ∨
    (1 ≤ piles_remaining ∧ piles_remaining ≤ piles) ∧
    (0 ≤ sum_stones_remaining ∧ sum_stones_remaining ≤ total_stones)
    → winning_strategy_for_basil = true := 
sorry -- The proof is omitted

end basil_has_winning_strategy_l2417_241739


namespace problem_l2417_241771

theorem problem (a : ℝ) : (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → (a > 3 ∨ a < -1) :=
by
  sorry

end problem_l2417_241771


namespace profit_difference_l2417_241766

-- Definitions of the conditions
def car_cost : ℕ := 100
def cars_per_month : ℕ := 4
def car_revenue : ℕ := 50

def motorcycle_cost : ℕ := 250
def motorcycles_per_month : ℕ := 8
def motorcycle_revenue : ℕ := 50

-- Calculation of profits
def car_profit : ℕ := (cars_per_month * car_revenue) - car_cost
def motorcycle_profit : ℕ := (motorcycles_per_month * motorcycle_revenue) - motorcycle_cost

-- Prove that the profit difference is 50 dollars
theorem profit_difference : (motorcycle_profit - car_profit) = 50 :=
by
  -- Statements to assert conditions and their proofs go here
  sorry

end profit_difference_l2417_241766


namespace original_selling_price_l2417_241788

variable (P : ℝ)
variable (S : ℝ) 

-- Conditions
axiom profit_10_percent : S = 1.10 * P
axiom profit_diff : 1.17 * P - S = 42

-- Goal
theorem original_selling_price : S = 660 := by
  sorry

end original_selling_price_l2417_241788


namespace smallest_percent_coffee_tea_l2417_241772

theorem smallest_percent_coffee_tea (C T : ℝ) (hC : C = 50) (hT : T = 60) : 
  ∃ x, x = C + T - 100 ∧ x = 10 :=
by
  sorry

end smallest_percent_coffee_tea_l2417_241772


namespace karlson_expenditure_exceeds_2000_l2417_241706

theorem karlson_expenditure_exceeds_2000 :
  ∃ n m : ℕ, 25 * n + 340 * m > 2000 :=
by {
  -- proof must go here
  sorry
}

end karlson_expenditure_exceeds_2000_l2417_241706


namespace problem_solution_l2417_241796

theorem problem_solution (a b : ℝ) (ha : |a| = 5) (hb : b = -3) :
  a + b = 2 ∨ a + b = -8 :=
by sorry

end problem_solution_l2417_241796


namespace expand_polynomials_l2417_241735

variable (x : ℝ)

theorem expand_polynomials : 
  (3 * x^2 - 4 * x + 3) * (-4 * x^2 + 2 * x - 6) = -12 * x^4 + 22 * x^3 - 38 * x^2 + 30 * x - 18 :=
  by
  sorry

end expand_polynomials_l2417_241735


namespace polygon_sides_l2417_241741

theorem polygon_sides (n : ℕ) 
  (h1 : ∀ (i : ℕ), i < n → 180 - 360 / n = 150) : n = 12 := by
  sorry

end polygon_sides_l2417_241741


namespace total_tiles_in_square_hall_l2417_241752

theorem total_tiles_in_square_hall
  (s : ℕ) -- integer side length of the square hall
  (black_tiles : ℕ)
  (total_tiles : ℕ)
  (all_tiles_white_or_black : ∀ (x : ℕ), x ≤ total_tiles → x = black_tiles ∨ x = total_tiles - black_tiles)
  (black_tiles_count : black_tiles = 153 + 3) : total_tiles = 6084 :=
by
  sorry

end total_tiles_in_square_hall_l2417_241752


namespace taxi_charge_l2417_241716

theorem taxi_charge :
  ∀ (initial_fee additional_charge_per_segment total_distance total_charge : ℝ),
  initial_fee = 2.05 →
  total_distance = 3.6 →
  total_charge = 5.20 →
  (total_charge - initial_fee) / (5/2 * total_distance) = 0.35 :=
by
  intros initial_fee additional_charge_per_segment total_distance total_charge
  intros h_initial_fee h_total_distance h_total_charge
  -- Proof here
  sorry

end taxi_charge_l2417_241716


namespace part1_part2_l2417_241774

open Set Real

noncomputable def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def B (m : ℝ) : Set ℝ := {x | m - 3 ≤ x ∧ x ≤ m + 3}
noncomputable def C : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem part1 (m : ℝ) (h : A ∩ B m = C) : m = 5 :=
  sorry

theorem part2 (m : ℝ) (h : A ⊆ (B m)ᶜ) : m < -4 ∨ 6 < m :=
  sorry

end part1_part2_l2417_241774
