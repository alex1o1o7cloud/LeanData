import Mathlib

namespace NUMINAMATH_GPT_maximum_f_l246_24692

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def f (p : ℝ) : ℝ :=
  binomial_coefficient 20 2 * p^2 * (1 - p)^18

theorem maximum_f :
  ∃ p_0 : ℝ, 0 < p_0 ∧ p_0 < 1 ∧ f p = f (0.1) := sorry

end NUMINAMATH_GPT_maximum_f_l246_24692


namespace NUMINAMATH_GPT_distance_between_points_l246_24629

open Real

theorem distance_between_points :
  let P := (1, 3)
  let Q := (-5, 7)
  dist P Q = 2 * sqrt 13 :=
by
  let P := (1, 3)
  let Q := (-5, 7)
  sorry

end NUMINAMATH_GPT_distance_between_points_l246_24629


namespace NUMINAMATH_GPT_arith_seq_ratio_l246_24671

variable {S T : ℕ → ℚ}

-- Conditions
def is_arith_seq_sum (S : ℕ → ℚ) (a : ℕ → ℚ) :=
  ∀ n, S n = n * (2 * a 1 + (n - 1) * a n) / 2

def ratio_condition (S T : ℕ → ℚ) :=
  ∀ n, S n / T n = (2 * n - 1) / (3 * n + 2)

-- Main theorem
theorem arith_seq_ratio
  (a b : ℕ → ℚ)
  (h1 : is_arith_seq_sum S a)
  (h2 : is_arith_seq_sum T b)
  (h3 : ratio_condition S T)
  : a 7 / b 7 = 25 / 41 :=
sorry

end NUMINAMATH_GPT_arith_seq_ratio_l246_24671


namespace NUMINAMATH_GPT_arithmetic_seq_a12_l246_24609

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + (n - 1) * d

theorem arithmetic_seq_a12 (a : ℕ → ℝ) (a1 d : ℝ) 
  (h_arith : arithmetic_seq a a1 d)
  (h7_and_9 : a 7 + a 9 = 16)
  (h4 : a 4 = 1) :
  a 12 = 15 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a12_l246_24609


namespace NUMINAMATH_GPT_complex_number_is_purely_imaginary_l246_24646

theorem complex_number_is_purely_imaginary (a : ℂ) : 
  (a^2 - a - 2 = 0) ∧ (a^2 - 3*a + 2 ≠ 0) ↔ a = -1 :=
by 
  sorry

end NUMINAMATH_GPT_complex_number_is_purely_imaginary_l246_24646


namespace NUMINAMATH_GPT_boat_trip_duration_l246_24626

noncomputable def boat_trip_time (B P : ℝ) : Prop :=
  (P = 4 * B) ∧ (B + P = 10)

theorem boat_trip_duration (B P : ℝ) (h : boat_trip_time B P) : B = 2 :=
by
  cases h with
  | intro hP hTotal =>
    sorry

end NUMINAMATH_GPT_boat_trip_duration_l246_24626


namespace NUMINAMATH_GPT_left_square_side_length_l246_24633

theorem left_square_side_length (x : ℕ) (h1 : x + (x + 17) + (x + 11) = 52) : x = 8 :=
sorry

end NUMINAMATH_GPT_left_square_side_length_l246_24633


namespace NUMINAMATH_GPT_find_angle_BAC_l246_24657

-- Definitions and Hypotheses
variables (A B C P : Type) (AP PC AB AC : Real) (angle_BPC : Real)

-- Hypotheses
-- AP = PC
-- AB = AC
-- angle BPC = 120 
axiom AP_eq_PC : AP = PC
axiom AB_eq_AC : AB = AC
axiom angle_BPC_eq_120 : angle_BPC = 120

-- Theorem
theorem find_angle_BAC (AP_eq_PC : AP = PC) (AB_eq_AC : AB = AC) (angle_BPC_eq_120 : angle_BPC = 120) : angle_BAC = 60 :=
sorry

end NUMINAMATH_GPT_find_angle_BAC_l246_24657


namespace NUMINAMATH_GPT_max_profit_at_80_l246_24604

-- Definitions based on conditions
def cost_price : ℝ := 40
def functional_relationship (x : ℝ) : ℝ := -x + 140
def profit (x : ℝ) : ℝ := (x - cost_price) * functional_relationship x

-- Statement to prove that maximum profit is achieved at x = 80
theorem max_profit_at_80 : (40 ≤ 80) → (80 ≤ 80) → profit 80 = 2400 := by
  sorry

end NUMINAMATH_GPT_max_profit_at_80_l246_24604


namespace NUMINAMATH_GPT_base_length_of_parallelogram_l246_24666

theorem base_length_of_parallelogram (area : ℝ) (base altitude : ℝ)
  (h1 : area = 98)
  (h2 : altitude = 2 * base) :
  base = 7 :=
by
  sorry

end NUMINAMATH_GPT_base_length_of_parallelogram_l246_24666


namespace NUMINAMATH_GPT_solution_for_m_exactly_one_solution_l246_24689

theorem solution_for_m_exactly_one_solution (m : ℚ) : 
  (∀ x : ℚ, (x - 3) / (m * x + 4) = 2 * x → 
            (2 * m * x^2 + 7 * x + 3 = 0)) →
  (49 - 24 * m = 0) → 
  m = 49 / 24 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_solution_for_m_exactly_one_solution_l246_24689


namespace NUMINAMATH_GPT_remainder_2519_div_6_l246_24660

theorem remainder_2519_div_6 : ∃ q r, 2519 = 6 * q + r ∧ 0 ≤ r ∧ r < 6 ∧ r = 5 := 
by
  sorry

end NUMINAMATH_GPT_remainder_2519_div_6_l246_24660


namespace NUMINAMATH_GPT_fraction_shaded_area_l246_24651

theorem fraction_shaded_area (l w : ℕ) (h_l : l = 15) (h_w : w = 20)
  (h_qtr : (1 / 4: ℝ) * (l * w) = 75) (h_shaded : (1 / 5: ℝ) * 75 = 15) :
  (15 / (l * w): ℝ) = 1 / 20 :=
by
  sorry

end NUMINAMATH_GPT_fraction_shaded_area_l246_24651


namespace NUMINAMATH_GPT_son_age_l246_24693

theorem son_age (S M : ℕ) (h1 : M = S + 30) (h2 : M + 2 = 2 * (S + 2)) : S = 28 := 
by
  -- The proof can be filled in here.
  sorry

end NUMINAMATH_GPT_son_age_l246_24693


namespace NUMINAMATH_GPT_total_amount_shared_l246_24669

theorem total_amount_shared (a b c : ℝ)
  (h1 : a = 1/3 * (b + c))
  (h2 : b = 2/7 * (a + c))
  (h3 : a = b + 20) : 
  a + b + c = 720 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_shared_l246_24669


namespace NUMINAMATH_GPT_trigonometric_identity_l246_24690

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α + Real.pi / 6) = 1 / 3) :
  Real.cos (2 * α - 2 * Real.pi / 3) = -7 / 9 :=
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l246_24690


namespace NUMINAMATH_GPT_find_father_age_l246_24661

variable (M F : ℕ)

noncomputable def age_relation_1 : Prop := M = (2 / 5) * F
noncomputable def age_relation_2 : Prop := M + 5 = (1 / 2) * (F + 5)

theorem find_father_age (h1 : age_relation_1 M F) (h2 : age_relation_2 M F) : F = 25 := by
  sorry

end NUMINAMATH_GPT_find_father_age_l246_24661


namespace NUMINAMATH_GPT_probability_Cecilia_rolls_4_given_win_l246_24652

noncomputable def P_roll_Cecilia_4_given_win : ℚ :=
  let P_C1_4 := 1/6
  let P_W_C := 1/5
  let P_W_C_given_C1_4 := (4/6)^4
  let P_C1_4_and_W_C := P_C1_4 * P_W_C_given_C1_4
  let P_C1_4_given_W_C := P_C1_4_and_W_C / P_W_C
  P_C1_4_given_W_C

theorem probability_Cecilia_rolls_4_given_win :
  P_roll_Cecilia_4_given_win = 256 / 1555 :=
by 
  -- Here the proof would go, but we include sorry for now.
  sorry

end NUMINAMATH_GPT_probability_Cecilia_rolls_4_given_win_l246_24652


namespace NUMINAMATH_GPT_walter_percent_of_dollar_l246_24610

theorem walter_percent_of_dollar
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)
  (penny_value : Nat := 1)
  (nickel_value : Nat := 5)
  (dime_value : Nat := 10)
  (dollar_value : Nat := 100)
  (total_value := pennies * penny_value + nickels * nickel_value + dimes * dime_value) :
  pennies = 2 ∧ nickels = 3 ∧ dimes = 2 →
  (total_value * 100) / dollar_value = 37 :=
by
  sorry

end NUMINAMATH_GPT_walter_percent_of_dollar_l246_24610


namespace NUMINAMATH_GPT_correct_judgment_l246_24678

def P := Real.pi < 2
def Q := Real.pi > 3

theorem correct_judgment : (P ∨ Q) ∧ ¬P := by
  sorry

end NUMINAMATH_GPT_correct_judgment_l246_24678


namespace NUMINAMATH_GPT_number_of_cherry_pie_days_l246_24638

theorem number_of_cherry_pie_days (A C : ℕ) (h1 : A + C = 7) (h2 : 12 * A = 12 * C + 12) : C = 3 :=
sorry

end NUMINAMATH_GPT_number_of_cherry_pie_days_l246_24638


namespace NUMINAMATH_GPT_divides_polynomial_difference_l246_24650

def P (a b c d x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem divides_polynomial_difference (a b c d x y : ℤ) (hxneqy : x ≠ y) :
  (x - y) ∣ (P a b c d x - P a b c d y) :=
by
  sorry

end NUMINAMATH_GPT_divides_polynomial_difference_l246_24650


namespace NUMINAMATH_GPT_range_of_x_l246_24664

theorem range_of_x (x : ℝ) (h1 : 2 ≤ |x - 5|) (h2 : |x - 5| ≤ 10) (h3 : 0 < x) : 
  (0 < x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 15) := 
sorry

end NUMINAMATH_GPT_range_of_x_l246_24664


namespace NUMINAMATH_GPT_problem_a_problem_c_l246_24627

variable {a b : ℝ}

theorem problem_a (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 1) : ab ≤ 1 / 8 :=
by
  sorry

theorem problem_c (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a + 2 * b = 1) : 1 / a + 2 / b ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_problem_a_problem_c_l246_24627


namespace NUMINAMATH_GPT_find_q_value_l246_24655

theorem find_q_value (q : ℚ) (x y : ℚ) (hx : x = 5 - q) (hy : y = 3*q - 1) : x = 3*y → q = 4/5 :=
by
  sorry

end NUMINAMATH_GPT_find_q_value_l246_24655


namespace NUMINAMATH_GPT_man_speed_against_current_proof_l246_24640

def man_speed_with_current : ℝ := 15
def speed_of_current : ℝ := 2.5
def man_speed_against_current : ℝ := 10

theorem man_speed_against_current_proof 
  (V_m : ℝ) 
  (h_with_current : V_m + speed_of_current = man_speed_with_current) :
  V_m - speed_of_current = man_speed_against_current := 
by 
  sorry

end NUMINAMATH_GPT_man_speed_against_current_proof_l246_24640


namespace NUMINAMATH_GPT_smallest_odd_prime_factor_l246_24643

theorem smallest_odd_prime_factor (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  (2023 ^ 8 + 1) % p = 0 ↔ p = 17 := 
by
  sorry

end NUMINAMATH_GPT_smallest_odd_prime_factor_l246_24643


namespace NUMINAMATH_GPT_max_value_x_div_y_l246_24684

variables {x y a b : ℝ}

theorem max_value_x_div_y (h1 : x ≥ y) (h2 : y > 0) (h3 : 0 ≤ a) (h4 : a ≤ x) (h5 : 0 ≤ b) (h6 : b ≤ y) 
  (h7 : (x - a)^2 + (y - b)^2 = x^2 + b^2) (h8 : x^2 + b^2 = y^2 + a^2) :
  x / y ≤ (2 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_max_value_x_div_y_l246_24684


namespace NUMINAMATH_GPT_number_of_marks_for_passing_l246_24631

theorem number_of_marks_for_passing (T P : ℝ) 
  (h1 : 0.40 * T = P - 40) 
  (h2 : 0.60 * T = P + 20) 
  (h3 : 0.45 * T = P - 10) :
  P = 160 :=
by
  sorry

end NUMINAMATH_GPT_number_of_marks_for_passing_l246_24631


namespace NUMINAMATH_GPT_find_digit_l246_24615

theorem find_digit (p q r : ℕ) (hq : p ≠ q) (hr : p ≠ r) (hq' : q ≠ r) 
    (hp_pos : 0 < p ∧ p < 10)
    (hq_pos : 0 < q ∧ q < 10)
    (hr_pos : 0 < r ∧ r < 10)
    (h1 : 10 * p + q = 17)
    (h2 : 10 * p + r = 13)
    (h3 : p + q + r = 11) : 
    q = 7 :=
sorry

end NUMINAMATH_GPT_find_digit_l246_24615


namespace NUMINAMATH_GPT_candle_height_half_after_9_hours_l246_24634

-- Define the initial heights and burn rates
def initial_height_first : ℝ := 12
def burn_rate_first : ℝ := 2
def initial_height_second : ℝ := 15
def burn_rate_second : ℝ := 3

-- Define the height functions after t hours
def height_first (t : ℝ) : ℝ := initial_height_first - burn_rate_first * t
def height_second (t : ℝ) : ℝ := initial_height_second - burn_rate_second * t

-- Prove that at t = 9, the height of the first candle is half the height of the second candle
theorem candle_height_half_after_9_hours : height_first 9 = 0.5 * height_second 9 := by
  sorry

end NUMINAMATH_GPT_candle_height_half_after_9_hours_l246_24634


namespace NUMINAMATH_GPT_coefficient_of_x_l246_24696

theorem coefficient_of_x : 
  let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2)
  ∃ (a b c : ℝ), expr = a * x^2 + b * x + c ∧ b = 5 := by
    let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2)
    exact sorry

end NUMINAMATH_GPT_coefficient_of_x_l246_24696


namespace NUMINAMATH_GPT_bacteria_growth_final_count_l246_24659

theorem bacteria_growth_final_count (initial_count : ℕ) (t : ℕ) 
(h1 : initial_count = 10) 
(h2 : t = 7) 
(h3 : ∀ n : ℕ, (n * 60) = t * 60 → 2 ^ n = 128) : 
(initial_count * 2 ^ t) = 1280 := 
by
  sorry

end NUMINAMATH_GPT_bacteria_growth_final_count_l246_24659


namespace NUMINAMATH_GPT_squirrel_acorns_initial_stash_l246_24621

theorem squirrel_acorns_initial_stash (A : ℕ) 
  (h1 : 3 * (A / 3 - 60) = 30) : A = 210 := 
sorry

end NUMINAMATH_GPT_squirrel_acorns_initial_stash_l246_24621


namespace NUMINAMATH_GPT_smallest_sum_of_inverses_l246_24607

theorem smallest_sum_of_inverses 
  (x y : ℕ) (hx : x ≠ y) (h1 : 0 < x) (h2 : 0 < y) (h_condition : (1 / x : ℚ) + 1 / y = 1 / 15) :
  x + y = 64 := 
sorry

end NUMINAMATH_GPT_smallest_sum_of_inverses_l246_24607


namespace NUMINAMATH_GPT_spinner_win_sector_area_l246_24613

open Real

theorem spinner_win_sector_area (r : ℝ) (P : ℝ)
  (h_r : r = 8) (h_P : P = 3 / 7) : 
  ∃ A : ℝ, A = 192 * π / 7 :=
by
  sorry

end NUMINAMATH_GPT_spinner_win_sector_area_l246_24613


namespace NUMINAMATH_GPT_moores_law_transistors_l246_24687

-- Define the initial conditions
def initial_transistors : ℕ := 500000
def doubling_period : ℕ := 2 -- in years
def transistors_doubling (n : ℕ) : ℕ := initial_transistors * 2^n

-- Calculate the number of doubling events from 1995 to 2010
def years_spanned : ℕ := 15
def number_of_doublings : ℕ := years_spanned / doubling_period

-- Expected number of transistors in 2010
def expected_transistors_in_2010 : ℕ := 64000000

theorem moores_law_transistors :
  transistors_doubling number_of_doublings = expected_transistors_in_2010 :=
sorry

end NUMINAMATH_GPT_moores_law_transistors_l246_24687


namespace NUMINAMATH_GPT_percentage_increase_correct_l246_24619

def bookstore_earnings : ℕ := 60
def tutoring_earnings : ℕ := 40
def new_bookstore_earnings : ℕ := 100
def additional_tutoring_fee : ℕ := 15
def old_total_earnings : ℕ := bookstore_earnings + tutoring_earnings
def new_total_earnings : ℕ := new_bookstore_earnings + (tutoring_earnings + additional_tutoring_fee)
def overall_percentage_increase : ℚ := (((new_total_earnings - old_total_earnings : ℚ) / old_total_earnings) * 100)

theorem percentage_increase_correct :
  overall_percentage_increase = 55 := sorry

end NUMINAMATH_GPT_percentage_increase_correct_l246_24619


namespace NUMINAMATH_GPT_largest_k_inequality_l246_24603

theorem largest_k_inequality :
  ∃ k : ℝ, (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → a + b + c = 3 → a^3 + b^3 + c^3 - 3 ≥ k * (3 - a * b - b * c - c * a)) ∧ k = 5 :=
sorry

end NUMINAMATH_GPT_largest_k_inequality_l246_24603


namespace NUMINAMATH_GPT_centered_hexagonal_seq_l246_24647

def is_centered_hexagonal (a : ℕ) : Prop :=
  ∃ n : ℕ, a = 3 * n^2 - 3 * n + 1

def are_sequences (a b c d : ℕ) : Prop :=
  (b = 2 * a - 1) ∧ (d = c^2) ∧ (a + b = c + d)

theorem centered_hexagonal_seq (a : ℕ) :
  (∃ b c d, are_sequences a b c d) ↔ is_centered_hexagonal a :=
sorry

end NUMINAMATH_GPT_centered_hexagonal_seq_l246_24647


namespace NUMINAMATH_GPT_find_a_l246_24694

variable (U : Set ℝ) (A : Set ℝ) (a : ℝ)

theorem find_a (hU_def : U = {2, 3, a^2 - a - 1})
               (hA_def : A = {2, 3})
               (h_compl : U \ A = {1}) :
  a = -1 ∨ a = 2 := 
sorry

end NUMINAMATH_GPT_find_a_l246_24694


namespace NUMINAMATH_GPT_rahul_meena_work_together_l246_24644

theorem rahul_meena_work_together (days_rahul : ℚ) (days_meena : ℚ) (combined_days : ℚ) :
  days_rahul = 5 ∧ days_meena = 10 → combined_days = 10 / 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_rahul_meena_work_together_l246_24644


namespace NUMINAMATH_GPT_number_of_pupils_in_class_l246_24649

-- Defining the conditions
def wrongMark : ℕ := 79
def correctMark : ℕ := 45
def averageIncreasedByHalf : ℕ := 2  -- Condition representing average increased by half

-- The goal is to prove the number of pupils is 68
theorem number_of_pupils_in_class (n S : ℕ) (h1 : wrongMark = 79) (h2 : correctMark = 45)
(h3 : averageIncreasedByHalf = 2) 
(h4 : S + (wrongMark - correctMark) = (3 / 2) * S) :
  n = 68 :=
  sorry

end NUMINAMATH_GPT_number_of_pupils_in_class_l246_24649


namespace NUMINAMATH_GPT_tangent_parallel_l246_24679

noncomputable def f (x : ℝ) := x^4 - x

theorem tangent_parallel (P : ℝ × ℝ) (hP : P = (1, 0)) :
  (∃ x y : ℝ, P = (x, y) ∧ (fderiv ℝ f x) 1 = 3 / 1) ↔ P = (1, 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_parallel_l246_24679


namespace NUMINAMATH_GPT_first_train_cross_time_is_10_seconds_l246_24663

-- Definitions based on conditions
def length_of_train := 120 -- meters
def time_second_train_cross_telegraph_post := 15 -- seconds
def distance_cross_each_other := 240 -- meters
def time_cross_each_other := 12 -- seconds

-- The speed of the second train
def speed_second_train := length_of_train / time_second_train_cross_telegraph_post -- m/s

-- The relative speed of both trains when crossing each other
def relative_speed := distance_cross_each_other / time_cross_each_other -- m/s

-- The speed of the first train
def speed_first_train := relative_speed - speed_second_train -- m/s

-- The time taken by the first train to cross the telegraph post
def time_first_train_cross_telegraph_post := length_of_train / speed_first_train -- seconds

-- Proof statement
theorem first_train_cross_time_is_10_seconds :
  time_first_train_cross_telegraph_post = 10 := by
  sorry

end NUMINAMATH_GPT_first_train_cross_time_is_10_seconds_l246_24663


namespace NUMINAMATH_GPT_robinson_family_children_count_l246_24645

theorem robinson_family_children_count 
  (m : ℕ) -- mother's age
  (f : ℕ) (f_age : f = 50) -- father's age is 50
  (x : ℕ) -- number of children
  (y : ℕ) -- average age of children
  (h1 : (m + 50 + x * y) / (2 + x) = 22)
  (h2 : (m + x * y) / (1 + x) = 18) :
  x = 6 := 
sorry

end NUMINAMATH_GPT_robinson_family_children_count_l246_24645


namespace NUMINAMATH_GPT_shuttlecock_weight_probability_l246_24617

variable (p_lt_4_8 : ℝ) -- Probability that its weight is less than 4.8 g
variable (p_le_4_85 : ℝ) -- Probability that its weight is not greater than 4.85 g

theorem shuttlecock_weight_probability (h1 : p_lt_4_8 = 0.3) (h2 : p_le_4_85 = 0.32) :
  p_le_4_85 - p_lt_4_8 = 0.02 :=
by
  sorry

end NUMINAMATH_GPT_shuttlecock_weight_probability_l246_24617


namespace NUMINAMATH_GPT_temperature_on_friday_l246_24654

theorem temperature_on_friday 
  (M T W Th F : ℤ) 
  (h1 : (M + T + W + Th) / 4 = 48) 
  (h2 : (T + W + Th + F) / 4 = 46) 
  (h3 : M = 43) : 
  F = 35 := 
by
  sorry

end NUMINAMATH_GPT_temperature_on_friday_l246_24654


namespace NUMINAMATH_GPT_units_digit_of_33_pow_33_mul_22_pow_22_l246_24653

theorem units_digit_of_33_pow_33_mul_22_pow_22 :
  (33 ^ (33 * (22 ^ 22))) % 10 = 1 :=
sorry

end NUMINAMATH_GPT_units_digit_of_33_pow_33_mul_22_pow_22_l246_24653


namespace NUMINAMATH_GPT_range_of_a_l246_24630

noncomputable def f (a x : ℝ) : ℝ :=
if h : a ≤ x ∧ x < 0 then -((1/2)^x)
else if h' : 0 ≤ x ∧ x ≤ 4 then -(x^2) + 2*x
else 0

theorem range_of_a (a : ℝ) (h : ∀ x, f a x ∈ Set.Icc (-8 : ℝ) (1 : ℝ)) : 
  a ∈ Set.Ico (-3 : ℝ) 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l246_24630


namespace NUMINAMATH_GPT_problem1_problem2_l246_24601

theorem problem1 (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
    (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := sorry

theorem problem2 (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
    (A + C) / (2 * B + A) = 9 / 5 := sorry

end NUMINAMATH_GPT_problem1_problem2_l246_24601


namespace NUMINAMATH_GPT_students_walk_fraction_l246_24667

theorem students_walk_fraction
  (school_bus_fraction : ℚ := 1/3)
  (car_fraction : ℚ := 1/5)
  (bicycle_fraction : ℚ := 1/8) :
  (1 - (school_bus_fraction + car_fraction + bicycle_fraction) = 41/120) :=
by
  sorry

end NUMINAMATH_GPT_students_walk_fraction_l246_24667


namespace NUMINAMATH_GPT_solve_diophantine_l246_24612

theorem solve_diophantine : ∀ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ (x^3 - y^3 = x * y + 61) → (x, y) = (6, 5) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_solve_diophantine_l246_24612


namespace NUMINAMATH_GPT_hyperbola_proof_l246_24605

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 4 = 1

def hyperbola_conditions (origin : ℝ × ℝ) (eccentricity : ℝ) (radius : ℝ) (focus : ℝ × ℝ) : Prop :=
  origin = (0, 0) ∧
  focus.1 = 0 ∧
  eccentricity = Real.sqrt 5 / 2 ∧
  radius = 2

theorem hyperbola_proof :
  ∃ (C : ℝ → ℝ → Prop),
    (∀ (x y : ℝ), hyperbola_conditions (0, 0) (Real.sqrt 5 / 2) 2 (0, c) → 
    C x y ↔ hyperbola_equation x y) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_proof_l246_24605


namespace NUMINAMATH_GPT_paint_cost_of_cube_l246_24681

theorem paint_cost_of_cube (cost_per_kg : ℕ) (coverage_per_kg : ℕ) (side_length : ℕ) (total_cost : ℕ) 
  (h1 : cost_per_kg = 20)
  (h2 : coverage_per_kg = 15)
  (h3 : side_length = 5)
  (h4 : total_cost = 200) : 
  (6 * side_length^2 / coverage_per_kg) * cost_per_kg = total_cost :=
by
  sorry

end NUMINAMATH_GPT_paint_cost_of_cube_l246_24681


namespace NUMINAMATH_GPT_find_x_l246_24691

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x^n

theorem find_x (x : ℝ) (H : series_sum x = 16) : 
  x = (33 - Real.sqrt 129) / 32 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l246_24691


namespace NUMINAMATH_GPT_parabolas_intersect_at_points_l246_24695

theorem parabolas_intersect_at_points :
  ∃ (x y : ℝ), (y = 3 * x^2 - 5 * x + 1 ∧ y = 4 * x^2 + 3 * x + 1) ↔ ((x = 0 ∧ y = 1) ∨ (x = -8 ∧ y = 233)) := 
sorry

end NUMINAMATH_GPT_parabolas_intersect_at_points_l246_24695


namespace NUMINAMATH_GPT_odd_n_divisibility_l246_24628

theorem odd_n_divisibility (n : ℤ) : (∃ a : ℤ, n ∣ 4 * a^2 - 1) ↔ (n % 2 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_odd_n_divisibility_l246_24628


namespace NUMINAMATH_GPT_exists_n_consecutive_non_prime_or_prime_power_l246_24665

theorem exists_n_consecutive_non_prime_or_prime_power (n : ℕ) (h : n > 0) :
  ∃ (seq : Fin n → ℕ), (∀ i, ¬ (Nat.Prime (seq i)) ∧ ¬ (∃ p k : ℕ, p.Prime ∧ k > 1 ∧ seq i = p ^ k)) :=
by
  sorry

end NUMINAMATH_GPT_exists_n_consecutive_non_prime_or_prime_power_l246_24665


namespace NUMINAMATH_GPT_range_of_m_l246_24618

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * m * x + 4 = 0) → x > 1) ↔ (2 ≤ m ∧ m < 5/2) := sorry

end NUMINAMATH_GPT_range_of_m_l246_24618


namespace NUMINAMATH_GPT_triangle_is_isosceles_l246_24674

theorem triangle_is_isosceles
  (A B C : ℝ)
  (h_triangle : A + B + C = π)
  (h_condition : 2 * Real.cos B * Real.sin C = Real.sin A) :
  B = C :=
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l246_24674


namespace NUMINAMATH_GPT_quadratic_equation_equivalence_l246_24699

theorem quadratic_equation_equivalence
  (a_0 a_1 a_2 : ℝ)
  (r s : ℝ)
  (h_roots : a_0 + a_1 * r + a_2 * r^2 = 0 ∧ a_0 + a_1 * s + a_2 * s^2 = 0)
  (h_a2_nonzero : a_2 ≠ 0) :
  (∀ x, a_0 ≠ 0 ↔ a_0 + a_1 * x + a_2 * x^2 = a_0 * (1 - x / r) * (1 - x / s)) :=
sorry

end NUMINAMATH_GPT_quadratic_equation_equivalence_l246_24699


namespace NUMINAMATH_GPT_distance_travelled_l246_24639

theorem distance_travelled
  (d : ℝ)                   -- distance in kilometers
  (train_speed : ℝ)         -- train speed in km/h
  (ship_speed : ℝ)          -- ship speed in km/h
  (time_difference : ℝ)     -- time difference in hours
  (h1 : train_speed = 48)
  (h2 : ship_speed = 60)
  (h3 : time_difference = 2) :
  d = 480 := 
by
  sorry

end NUMINAMATH_GPT_distance_travelled_l246_24639


namespace NUMINAMATH_GPT_triangle_geometric_sequence_sine_rule_l246_24682

noncomputable def sin60 : Real := Real.sqrt 3 / 2

theorem triangle_geometric_sequence_sine_rule 
  {a b c : Real} 
  {A B C : Real} 
  (h1 : a / b = b / c) 
  (h2 : A = 60 * Real.pi / 180) :
  b * Real.sin B / c = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_geometric_sequence_sine_rule_l246_24682


namespace NUMINAMATH_GPT_hyperbola_equation_l246_24685

-- Definitions of the conditions
def is_asymptote_1 (y x : ℝ) : Prop :=
  y = 2 * x

def is_asymptote_2 (y x : ℝ) : Prop :=
  y = -2 * x

def passes_through_focus (x y : ℝ) : Prop :=
  x = 1 ∧ y = 0

-- The statement to be proved
theorem hyperbola_equation :
  (∀ x y : ℝ, passes_through_focus x y → x^2 - (y^2 / 4) = 1) :=
sorry

end NUMINAMATH_GPT_hyperbola_equation_l246_24685


namespace NUMINAMATH_GPT_lcm_9_16_21_eq_1008_l246_24680

theorem lcm_9_16_21_eq_1008 : Nat.lcm (Nat.lcm 9 16) 21 = 1008 := by
  sorry

end NUMINAMATH_GPT_lcm_9_16_21_eq_1008_l246_24680


namespace NUMINAMATH_GPT_find_ab_l246_24624

theorem find_ab (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 48) : a * b = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l246_24624


namespace NUMINAMATH_GPT_insurance_covers_80_percent_l246_24642

-- Definitions from the problem conditions
def cost_per_aid : ℕ := 2500
def num_aids : ℕ := 2
def johns_payment : ℕ := 1000

-- Total cost of hearing aids
def total_cost : ℕ := cost_per_aid * num_aids

-- Insurance payment
def insurance_payment : ℕ := total_cost - johns_payment

-- The theorem to prove
theorem insurance_covers_80_percent :
  (insurance_payment * 100 / total_cost) = 80 :=
by
  sorry

end NUMINAMATH_GPT_insurance_covers_80_percent_l246_24642


namespace NUMINAMATH_GPT_base_subtraction_l246_24620

-- Define the base 8 number 765432_8 and its conversion to base 10
def base8Number : ℕ := 7 * (8^5) + 6 * (8^4) + 5 * (8^3) + 4 * (8^2) + 3 * (8^1) + 2 * (8^0)

-- Define the base 9 number 543210_9 and its conversion to base 10
def base9Number : ℕ := 5 * (9^5) + 4 * (9^4) + 3 * (9^3) + 2 * (9^2) + 1 * (9^1) + 0 * (9^0)

-- Lean 4 statement for the proof problem
theorem base_subtraction : (base8Number : ℤ) - (base9Number : ℤ) = -67053 := by
    sorry

end NUMINAMATH_GPT_base_subtraction_l246_24620


namespace NUMINAMATH_GPT_regular_tetrahedron_fourth_vertex_l246_24616

theorem regular_tetrahedron_fourth_vertex :
  ∃ (x y z : ℤ), 
    ((x, y, z) = (0, 0, 6) ∨ (x, y, z) = (0, 0, -6)) ∧
    ((x - 0) ^ 2 + (y - 0) ^ 2 + (z - 0) ^ 2 = 36) ∧
    ((x - 6) ^ 2 + (y - 0) ^ 2 + (z - 0) ^ 2 = 36) ∧
    ((x - 5) ^ 2 + (y - 0) ^ 2 + (z - 6) ^ 2 = 36) := 
by
  sorry

end NUMINAMATH_GPT_regular_tetrahedron_fourth_vertex_l246_24616


namespace NUMINAMATH_GPT_remainder_of_789987_div_8_l246_24648

theorem remainder_of_789987_div_8 : (789987 % 8) = 3 := by
  sorry

end NUMINAMATH_GPT_remainder_of_789987_div_8_l246_24648


namespace NUMINAMATH_GPT_line_passes_through_point_l246_24635

theorem line_passes_through_point :
  ∀ (m : ℝ), (∃ y : ℝ, y - 2 = m * (-1) + m) :=
by
  intros m
  use 2
  sorry

end NUMINAMATH_GPT_line_passes_through_point_l246_24635


namespace NUMINAMATH_GPT_no_3_digit_even_sum_27_l246_24698

/-- Predicate for a 3-digit number -/
def is_3_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Predicate for an even number -/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Function to compute the digit sum of a number -/
def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- Theorem: There are no 3-digit numbers with a digit sum of 27 that are even -/
theorem no_3_digit_even_sum_27 : 
  ∀ n : ℕ, is_3_digit n → digit_sum n = 27 → is_even n → false :=
by
  sorry

end NUMINAMATH_GPT_no_3_digit_even_sum_27_l246_24698


namespace NUMINAMATH_GPT_number_of_real_solutions_l246_24662

theorem number_of_real_solutions (x : ℝ) (n : ℤ) : 
  (3 : ℝ) * x^2 - 27 * (n : ℝ) + 29 = 0 → n = ⌊x⌋ →  ∃! x, (3 : ℝ) * x^2 - 27 * (⌊x⌋ : ℝ) + 29 = 0 := 
sorry

end NUMINAMATH_GPT_number_of_real_solutions_l246_24662


namespace NUMINAMATH_GPT_trajectory_of_M_l246_24608

theorem trajectory_of_M (M : ℝ × ℝ) (h : (M.2 < 0 → M.1 = 0) ∧ (M.2 ≥ 0 → M.1 ^ 2 = 8 * M.2)) :
  (M.2 < 0 → M.1 = 0) ∧ (M.2 ≥ 0 → M.1 ^ 2 = 8 * M.2) :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_M_l246_24608


namespace NUMINAMATH_GPT_w_janous_conjecture_l246_24697

theorem w_janous_conjecture (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (z^2 - x^2) / (x + y) + (x^2 - y^2) / (y + z) + (y^2 - z^2) / (z + x) ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_w_janous_conjecture_l246_24697


namespace NUMINAMATH_GPT_seats_needed_l246_24670

-- Definitions based on the problem's condition
def children : ℕ := 58
def children_per_seat : ℕ := 2

-- Theorem statement to prove
theorem seats_needed : children / children_per_seat = 29 :=
by
  sorry

end NUMINAMATH_GPT_seats_needed_l246_24670


namespace NUMINAMATH_GPT_mass_percentage_of_Ba_l246_24625

theorem mass_percentage_of_Ba {BaX : Type} {molar_mass_Ba : ℝ} {compound_mass : ℝ} {mass_Ba : ℝ}:
  molar_mass_Ba = 137.33 ∧ 
  compound_mass = 100 ∧
  mass_Ba = 66.18 →
  (mass_Ba / compound_mass * 100) = 66.18 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_of_Ba_l246_24625


namespace NUMINAMATH_GPT_extreme_value_and_inequality_l246_24632

theorem extreme_value_and_inequality
  (f : ℝ → ℝ)
  (a c : ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_extreme : f 1 = -2)
  (h_f_def : ∀ x : ℝ, f x = a * x^3 + c * x)
  (h_a_c : a = 1 ∧ c = -3) :
  (∀ x : ℝ, x < -1 → deriv f x > 0) ∧
  (∀ x : ℝ, -1 < x ∧ x < 1 → deriv f x < 0) ∧
  (∀ x : ℝ, 1 < x → deriv f x > 0) ∧
  f (-1) = 2 ∧
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 → |f x₁ - f x₂| < 4) :=
by sorry

end NUMINAMATH_GPT_extreme_value_and_inequality_l246_24632


namespace NUMINAMATH_GPT_range_of_m_real_roots_l246_24688

theorem range_of_m_real_roots (m : ℝ) : 
  (∀ x : ℝ, ∃ k l : ℝ, k = 2*x ∧ l = m - x^2 ∧ k^2 - 4*l ≥ 0) ↔ m ≤ 1 := 
sorry

end NUMINAMATH_GPT_range_of_m_real_roots_l246_24688


namespace NUMINAMATH_GPT_reservoir_capacity_l246_24641

-- Definitions based on the conditions
def storm_deposit : ℚ := 120 * 10^9
def final_full_percentage : ℚ := 0.85
def initial_full_percentage : ℚ := 0.55
variable (C : ℚ) -- total capacity of the reservoir in gallons

-- The statement we want to prove
theorem reservoir_capacity :
  final_full_percentage * C - initial_full_percentage * C = storm_deposit →
  C = 400 * 10^9
:= by
  sorry

end NUMINAMATH_GPT_reservoir_capacity_l246_24641


namespace NUMINAMATH_GPT_lcm_of_numbers_l246_24672

-- Define the conditions given in the problem
def ratio (a b : ℕ) : Prop := 7 * b = 13 * a
def hcf_23 (a b : ℕ) : Prop := Nat.gcd a b = 23

-- Main statement to prove
theorem lcm_of_numbers (a b : ℕ) (h_ratio : ratio a b) (h_hcf : hcf_23 a b) : Nat.lcm a b = 2093 := by
  sorry

end NUMINAMATH_GPT_lcm_of_numbers_l246_24672


namespace NUMINAMATH_GPT_arithmetic_seq_a7_value_l246_24636

theorem arithmetic_seq_a7_value {a : ℕ → ℝ} (h_positive : ∀ n, 0 < a n)
    (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h_eq : 3 * a 6 - (a 7) ^ 2 + 3 * a 8 = 0) : a 7 = 6 :=
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a7_value_l246_24636


namespace NUMINAMATH_GPT_required_more_visits_l246_24600

-- Define the conditions
def n := 395
def m := 2
def v1 := 135
def v2 := 112
def v3 := 97

-- Define the target statement
theorem required_more_visits : (n * m) - (v1 + v2 + v3) = 446 := by
  sorry

end NUMINAMATH_GPT_required_more_visits_l246_24600


namespace NUMINAMATH_GPT_problem_statement_l246_24675

theorem problem_statement (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) 
  (h_eq : x + y + z = 1/x + 1/y + 1/z) : 
  x + y + z ≥ Real.sqrt ((x * y + 1) / 2) + Real.sqrt ((y * z + 1) / 2) + Real.sqrt ((z * x + 1) / 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l246_24675


namespace NUMINAMATH_GPT_total_weight_of_lifts_l246_24606

theorem total_weight_of_lifts 
  (F S : ℕ)
  (h1 : F = 400)
  (h2 : 2 * F = S + 300) :
  F + S = 900 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_lifts_l246_24606


namespace NUMINAMATH_GPT_find_divisor_l246_24614

-- Definitions from the condition
def original_number : ℕ := 724946
def least_number_subtracted : ℕ := 6
def remaining_number : ℕ := original_number - least_number_subtracted

theorem find_divisor (h1 : remaining_number % least_number_subtracted = 0) :
  Nat.gcd original_number least_number_subtracted = 2 :=
sorry

end NUMINAMATH_GPT_find_divisor_l246_24614


namespace NUMINAMATH_GPT_sum_distances_l246_24611

noncomputable def lengthAB : ℝ := 2
noncomputable def lengthA'B' : ℝ := 5
noncomputable def midpointAB : ℝ := lengthAB / 2
noncomputable def midpointA'B' : ℝ := lengthA'B' / 2
noncomputable def distancePtoD : ℝ := 0.5
noncomputable def proportionality_constant : ℝ := lengthA'B' / lengthAB

theorem sum_distances : distancePtoD + (proportionality_constant * distancePtoD) = 1.75 := by
  sorry

end NUMINAMATH_GPT_sum_distances_l246_24611


namespace NUMINAMATH_GPT_first_term_geometric_l246_24623

-- Definition: geometric sequence properties
variables (a r : ℚ) -- sequence terms are rational numbers
variables (n : ℕ)

-- Conditions: fifth and sixth terms of a geometric sequence
def fifth_term_geometric (a r : ℚ) : ℚ := a * r^4
def sixth_term_geometric (a r : ℚ) : ℚ := a * r^5

-- Proof: given conditions
theorem first_term_geometric (a r : ℚ) (h1 : fifth_term_geometric a r = 48) 
  (h2 : sixth_term_geometric a r = 72) : a = 768 / 81 :=
by {
  sorry
}

end NUMINAMATH_GPT_first_term_geometric_l246_24623


namespace NUMINAMATH_GPT_sum_digits_increment_l246_24658

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_increment (n : ℕ) (h : sum_digits n = 1365) : 
  sum_digits (n + 1) = 1360 :=
by
  sorry

end NUMINAMATH_GPT_sum_digits_increment_l246_24658


namespace NUMINAMATH_GPT_triangle_inequality_l246_24677

noncomputable def area_triangle (a b c : ℝ) : ℝ := sorry -- Definition of area, but implementation is not required.

theorem triangle_inequality (a b c : ℝ) (S_triangle : ℝ):
  1 - (8 * ((a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2) / (a + b + c) ^ 2)
  ≤ 432 * S_triangle ^ 2 / (a + b + c) ^ 4
  ∧ 432 * S_triangle ^ 2 / (a + b + c) ^ 4
  ≤ 1 - (2 * ((a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2) / (a + b + c) ^ 2) :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_triangle_inequality_l246_24677


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l246_24637

theorem arithmetic_sequence_value (a : ℕ → ℝ) (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_cond : a 3 + a 9 = 15 - a 6) : a 6 = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l246_24637


namespace NUMINAMATH_GPT_probability_one_red_ball_distribution_of_X_l246_24676

-- Definitions of probabilities
def C (n k : ℕ) : ℕ := Nat.choose n k

def P_one_red_ball : ℚ := (C 2 1 * C 3 2 : ℚ) / C 5 3

#check (1 : ℚ)
#check (3 : ℚ)
#check (5 : ℚ)
def X_distribution (i : ℕ) : ℚ :=
  if i = 0 then (C 3 3 : ℚ) / C 5 3
  else if i = 1 then (C 2 1 * C 3 2 : ℚ) / C 5 3
  else if i = 2 then (C 2 2 * C 3 1 : ℚ) / C 5 3
  else 0

-- Statement to prove
theorem probability_one_red_ball : 
  P_one_red_ball = 3 / 5 := 
sorry

theorem distribution_of_X :
  Π i, (i = 0 → X_distribution i = 1 / 10) ∧
       (i = 1 → X_distribution i = 3 / 5) ∧
       (i = 2 → X_distribution i = 3 / 10) :=
sorry

end NUMINAMATH_GPT_probability_one_red_ball_distribution_of_X_l246_24676


namespace NUMINAMATH_GPT_new_ratio_l246_24683

theorem new_ratio (J: ℝ) (F: ℝ) (F_new: ℝ): 
  J = 59.99999999999997 → 
  F / J = 3 / 2 → 
  F_new = F + 10 → 
  F_new / J = 5 / 3 :=
by
  intros hJ hF hF_new
  sorry

end NUMINAMATH_GPT_new_ratio_l246_24683


namespace NUMINAMATH_GPT_euro_operation_example_l246_24686

def euro_operation (x y : ℕ) : ℕ := 3 * x * y

theorem euro_operation_example : euro_operation 3 (euro_operation 4 5) = 540 :=
by sorry

end NUMINAMATH_GPT_euro_operation_example_l246_24686


namespace NUMINAMATH_GPT_sum_of_roots_of_equation_l246_24668

theorem sum_of_roots_of_equation :
  (∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x - a) * (x - b) = 0 ∧ a + b = 1) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_equation_l246_24668


namespace NUMINAMATH_GPT_num_girls_in_school_l246_24602

noncomputable def total_students : ℕ := 1600
noncomputable def sample_students : ℕ := 200
noncomputable def girls_less_than_boys_in_sample : ℕ := 10

-- Equations from conditions
def boys_in_sample (B G : ℕ) : Prop := G = B - girls_less_than_boys_in_sample
def sample_size (B G : ℕ) : Prop := B + G = sample_students

-- Proportion condition
def proportional_condition (G G_total : ℕ) : Prop := G * total_students = G_total * sample_students

-- Total number of girls in the school
def total_girls_in_school (G_total : ℕ) : Prop := G_total = 760

theorem num_girls_in_school :
  ∃ B G G_total : ℕ, boys_in_sample B G ∧ sample_size B G ∧ proportional_condition G G_total ∧ total_girls_in_school G_total :=
sorry

end NUMINAMATH_GPT_num_girls_in_school_l246_24602


namespace NUMINAMATH_GPT_rank_matA_l246_24656

def matA : Matrix (Fin 4) (Fin 5) ℤ :=
  ![![5, 7, 12, 48, -14],
    ![9, 16, 24, 98, -31],
    ![14, 24, 25, 146, -45],
    ![11, 12, 24, 94, -25]]

theorem rank_matA : Matrix.rank matA = 3 :=
by
  sorry

end NUMINAMATH_GPT_rank_matA_l246_24656


namespace NUMINAMATH_GPT_y_decreases_as_x_less_than_4_l246_24622

theorem y_decreases_as_x_less_than_4 (x : ℝ) : (x < 4) → ((x - 4)^2 + 3 < (4 - 4)^2 + 3) :=
by
  sorry

end NUMINAMATH_GPT_y_decreases_as_x_less_than_4_l246_24622


namespace NUMINAMATH_GPT_arithmetic_square_root_16_l246_24673

theorem arithmetic_square_root_16 : ∀ x : ℝ, x ≥ 0 → x^2 = 16 → x = 4 :=
by
  intro x hx h
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_16_l246_24673
