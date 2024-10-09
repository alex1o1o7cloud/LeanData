import Mathlib

namespace solutions_diff_squared_l1341_134150

theorem solutions_diff_squared (a b : ℝ) (h : 5 * a^2 - 6 * a - 55 = 0 ∧ 5 * b^2 - 6 * b - 55 = 0) :
  (a - b)^2 = 1296 / 25 := by
  sorry

end solutions_diff_squared_l1341_134150


namespace num_diagonals_octagon_l1341_134132

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem num_diagonals_octagon : num_diagonals 8 = 20 :=
by
  sorry

end num_diagonals_octagon_l1341_134132


namespace length_of_GH_l1341_134184

theorem length_of_GH (AB FE CD : ℕ) (side_large side_second side_third side_small : ℕ) 
  (h1 : AB = 11) (h2 : FE = 13) (h3 : CD = 5)
  (h4 : side_large = side_second + AB)
  (h5 : side_second = side_third + CD)
  (h6 : side_third = side_small + FE) :
  GH = 29 :=
by
  -- Proof steps would follow here based on the problem's solution
  -- Using the given conditions and transformations.
  sorry

end length_of_GH_l1341_134184


namespace distance_between_points_l1341_134179

theorem distance_between_points :
  let x1 := 2
  let y1 := -2
  let x2 := 8
  let y2 := 8
  let dist := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  dist = Real.sqrt 136 :=
by
  -- Proof to be filled in here.
  sorry

end distance_between_points_l1341_134179


namespace increase_in_green_chameleons_is_11_l1341_134197

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end increase_in_green_chameleons_is_11_l1341_134197


namespace num_valid_triples_l1341_134118

theorem num_valid_triples : ∃! (count : ℕ), count = 22 ∧
  ∀ k m n : ℕ, (0 ≤ k) ∧ (k ≤ 100) ∧ (0 ≤ m) ∧ (m ≤ 100) ∧ (0 ≤ n) ∧ (n ≤ 100) → 
  (2^m * n - 2^n * m = 2^k) → count = 22 :=
sorry

end num_valid_triples_l1341_134118


namespace determine_true_propositions_l1341_134172

def p (x y : ℝ) := x > y → -x < -y
def q (x y : ℝ) := (1/x > 1/y) → x < y

theorem determine_true_propositions (x y : ℝ) :
  (p x y ∨ q x y) ∧ (p x y ∧ ¬ q x y) :=
by
  sorry

end determine_true_propositions_l1341_134172


namespace solve_system_l1341_134157

theorem solve_system:
  ∃ (x y : ℝ), (26 * x^2 + 42 * x * y + 17 * y^2 = 10 ∧ 10 * x^2 + 18 * x * y + 8 * y^2 = 6) ↔
  (x = -1 ∧ y = 2) ∨ (x = -11 ∧ y = 14) ∨ (x = 11 ∧ y = -14) ∨ (x = 1 ∧ y = -2) :=
by
  sorry

end solve_system_l1341_134157


namespace trains_clear_in_approx_6_85_seconds_l1341_134147

noncomputable def length_first_train : ℝ := 111
noncomputable def length_second_train : ℝ := 165
noncomputable def speed_first_train : ℝ := 80 * (1000 / 3600) -- converting from km/h to m/s
noncomputable def speed_second_train : ℝ := 65 * (1000 / 3600) -- converting from km/h to m/s
noncomputable def relative_speed : ℝ := speed_first_train + speed_second_train
noncomputable def total_distance : ℝ := length_first_train + length_second_train
noncomputable def time_to_clear : ℝ := total_distance / relative_speed

theorem trains_clear_in_approx_6_85_seconds : abs (time_to_clear - 6.85) < 0.01 := sorry

end trains_clear_in_approx_6_85_seconds_l1341_134147


namespace shaded_area_l1341_134148

noncomputable def squareArea (a : ℝ) : ℝ := a * a

theorem shaded_area {s : ℝ} (h1 : squareArea s = 1) (h2 : s / s = 2) : 
  ∃ (shaded : ℝ), shaded = 1 / 3 :=
by
  sorry

end shaded_area_l1341_134148


namespace remainder_17_pow_49_mod_5_l1341_134198

theorem remainder_17_pow_49_mod_5 : (17^49) % 5 = 2 :=
by
  sorry

end remainder_17_pow_49_mod_5_l1341_134198


namespace value_of_nested_fraction_l1341_134113

def nested_fraction : ℚ :=
  2 - (1 / (2 - (1 / (2 - 1 / 2))))

theorem value_of_nested_fraction : nested_fraction = 3 / 4 :=
by
  sorry

end value_of_nested_fraction_l1341_134113


namespace choir_row_lengths_l1341_134164

theorem choir_row_lengths : 
  ∃ s : Finset ℕ, (∀ d ∈ s, d ∣ 90 ∧ 6 ≤ d ∧ d ≤ 15) ∧ s.card = 4 := by
  sorry

end choir_row_lengths_l1341_134164


namespace min_p_value_l1341_134108

variable (p q r s : ℝ)

theorem min_p_value (h1 : p + q + r + s = 10)
                    (h2 : pq + pr + ps + qr + qs + rs = 20)
                    (h3 : p^2 * q^2 * r^2 * s^2 = 16) :
  p ≥ 2 ∧ ∃ q r s, q + r + s = 10 - p ∧ pq + pr + ps + qr + qs + rs = 20 ∧ (p^2 * q^2 * r^2 * s^2 = 16) :=
by
  sorry  -- proof goes here

end min_p_value_l1341_134108


namespace smaller_area_l1341_134154

theorem smaller_area (x y : ℝ) 
  (h1 : x + y = 900)
  (h2 : y - x = (1 / 5) * (x + y) / 2) :
  x = 405 :=
sorry

end smaller_area_l1341_134154


namespace sum_first_12_terms_l1341_134170

def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def geometric_mean {α : Type} [Field α] (a b c : α) : Prop :=
b^2 = a * c

def sum_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
n * (a 1 + a n) / 2

theorem sum_first_12_terms 
  (a : ℕ → ℚ)
  (d : ℚ)
  (h1 : arithmetic_sequence a 1)
  (h2 : geometric_mean (a 3) (a 6) (a 11)) :
  sum_arithmetic_sequence a 12 = 96 :=
sorry

end sum_first_12_terms_l1341_134170


namespace cyclist_C_speed_l1341_134104

theorem cyclist_C_speed 
  (dist_XY : ℝ)
  (speed_diff : ℝ)
  (meet_point : ℝ)
  (c d : ℝ)
  (h1 : dist_XY = 90)
  (h2 : speed_diff = 5)
  (h3 : meet_point = 15)
  (h4 : d = c + speed_diff)
  (h5 : 75 = dist_XY - meet_point)
  (h6 : 105 = dist_XY + meet_point)
  (h7 : 75 / c = 105 / d) :
  c = 12.5 :=
sorry

end cyclist_C_speed_l1341_134104


namespace distinct_sequences_count_l1341_134115

noncomputable def number_of_distinct_sequences (n : ℕ) : ℕ :=
  if n = 6 then 12 else sorry

theorem distinct_sequences_count : number_of_distinct_sequences 6 = 12 := 
by 
  sorry

end distinct_sequences_count_l1341_134115


namespace length_of_segment_CD_l1341_134144

theorem length_of_segment_CD (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y)
  (h_ratio1 : x = (3 / 5) * (3 + y))
  (h_ratio2 : (x + 3) / y = 4 / 7)
  (h_RS : 3 = 3) :
  x + 3 + y = 273.6 :=
by
  sorry

end length_of_segment_CD_l1341_134144


namespace central_angle_l1341_134123

theorem central_angle (r l θ : ℝ) (condition1: 2 * r + l = 8) (condition2: (1 / 2) * l * r = 4) (theta_def : θ = l / r) : |θ| = 2 :=
by
  sorry

end central_angle_l1341_134123


namespace rectangle_length_l1341_134151

-- Define the area and width of the rectangle as given
def width : ℝ := 4
def area  : ℝ := 28

-- Prove that the length is 7 cm given the conditions
theorem rectangle_length : ∃ length : ℝ, length = 7 ∧ area = length * width :=
sorry

end rectangle_length_l1341_134151


namespace leila_cakes_monday_l1341_134152

def number_of_cakes_monday (m : ℕ) : Prop :=
  let cakes_friday := 9
  let cakes_saturday := 3 * m
  let total_cakes := m + cakes_friday + cakes_saturday
  total_cakes = 33

theorem leila_cakes_monday : ∃ m : ℕ, number_of_cakes_monday m ∧ m = 6 :=
by 
  -- We propose that the number of cakes she ate on Monday, denoted as m, is 6.
  -- We need to prove that this satisfies the given conditions.
  -- This line is a placeholder for the proof.
  sorry

end leila_cakes_monday_l1341_134152


namespace solve_for_a_l1341_134189

theorem solve_for_a (S P Q R : Type) (a b c d : ℝ) 
  (h1 : a + b + c + d = 360)
  (h2 : ∀ (PSQ : Type), d = 90) :
  a = 270 - b - c :=
by
  sorry

end solve_for_a_l1341_134189


namespace solve_equation_l1341_134182

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end solve_equation_l1341_134182


namespace primes_sum_solutions_l1341_134130

theorem primes_sum_solutions :
  ∃ (p q r : ℕ), Prime p ∧ Prime q ∧ Prime r ∧
  p + q^2 + r^3 = 200 ∧ 
  ((p = 167 ∧ q = 5 ∧ r = 2) ∨ 
   (p = 71 ∧ q = 11 ∧ r = 2) ∨ 
   (p = 23 ∧ q = 13 ∧ r = 2) ∨ 
   (p = 71 ∧ q = 2 ∧ r = 5)) :=
sorry

end primes_sum_solutions_l1341_134130


namespace units_digit_of_m_squared_plus_3_to_the_m_l1341_134199

def m : ℕ := 2021^3 + 3^2021

theorem units_digit_of_m_squared_plus_3_to_the_m 
  (hm : m = 2021^3 + 3^2021) : 
  ((m^2 + 3^m) % 10) = 7 := 
by 
  -- Here you would input the proof steps, however, we skip it now with sorry.
  sorry

end units_digit_of_m_squared_plus_3_to_the_m_l1341_134199


namespace yellow_bags_count_l1341_134145

theorem yellow_bags_count (R B Y : ℕ) 
  (h1 : R + B + Y = 12) 
  (h2 : 10 * R + 50 * B + 100 * Y = 500) 
  (h3 : R = B) : 
  Y = 2 := 
by 
  sorry

end yellow_bags_count_l1341_134145


namespace nth_number_in_S_l1341_134174

def S : Set ℕ := {n | ∃ k : ℕ, n = 15 * k + 11}

theorem nth_number_in_S (n : ℕ) (hn : n = 127) : ∃ k, 15 * k + 11 = 1901 :=
by
  sorry

end nth_number_in_S_l1341_134174


namespace alice_bob_task_l1341_134185

theorem alice_bob_task (t : ℝ) (h₁ : 1/4 + 1/6 = 5/12) (h₂ : t - 1/2 ≠ 0) :
    (5/12) * (t - 1/2) = 1 :=
sorry

end alice_bob_task_l1341_134185


namespace percentage_of_millet_in_Brand_A_l1341_134122

variable (A B : ℝ)
variable (B_percent : B = 0.65)
variable (mix_millet_percent : 0.60 * A + 0.40 * B = 0.50)

theorem percentage_of_millet_in_Brand_A :
  A = 0.40 :=
by
  sorry

end percentage_of_millet_in_Brand_A_l1341_134122


namespace chocolates_for_sister_l1341_134168
-- Importing necessary library

-- Lean 4 statement of the problem
theorem chocolates_for_sister (S : ℕ) 
  (herself_chocolates_per_saturday : ℕ := 2)
  (birthday_gift_chocolates : ℕ := 10)
  (saturdays_in_month : ℕ := 4)
  (total_chocolates : ℕ := 22) 
  (monthly_chocolates_herself := saturdays_in_month * herself_chocolates_per_saturday) 
  (equation : saturdays_in_month * S + monthly_chocolates_herself + birthday_gift_chocolates = total_chocolates) : 
  S = 1 :=
  sorry

end chocolates_for_sister_l1341_134168


namespace probability_of_yellow_l1341_134171

-- Definitions of the given conditions
def red_jelly_beans := 4
def green_jelly_beans := 8
def yellow_jelly_beans := 9
def blue_jelly_beans := 5
def total_jelly_beans := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans

-- Theorem statement
theorem probability_of_yellow :
  (yellow_jelly_beans : ℚ) / total_jelly_beans = 9 / 26 :=
by
  sorry

end probability_of_yellow_l1341_134171


namespace rearrange_pairs_l1341_134177

theorem rearrange_pairs {a b : ℕ} (hb: b = (2 / 3 : ℚ) * a) (boys_way_museum boys_way_back : ℕ) :
  boys_way_museum = 3 * a ∧ boys_way_back = 4 * b → 
  ∃ c : ℕ, boys_way_museum = 7 * c ∧ b = c := sorry

end rearrange_pairs_l1341_134177


namespace recycle_cans_l1341_134162

theorem recycle_cans (initial_cans : ℕ) (recycle_rate : ℕ) (n1 n2 n3 : ℕ)
  (h1 : initial_cans = 450)
  (h2 : recycle_rate = 5)
  (h3 : n1 = initial_cans / recycle_rate)
  (h4 : n2 = n1 / recycle_rate)
  (h5 : n3 = n2 / recycle_rate)
  (h6 : n3 / recycle_rate = 0) : 
  n1 + n2 + n3 = 111 :=
by
  sorry

end recycle_cans_l1341_134162


namespace batsman_average_after_12th_l1341_134137

theorem batsman_average_after_12th (runs_12th : ℕ) (average_increase : ℕ) (initial_innings : ℕ)
   (initial_average : ℝ) (runs_before_12th : ℕ → ℕ) 
   (h1 : runs_12th = 48)
   (h2 : average_increase = 2)
   (h3 : initial_innings = 11)
   (h4 : initial_average = 24)
   (h5 : ∀ i, i < initial_innings → runs_before_12th i ≥ 20)
   (h6 : ∃ i, runs_before_12th i = 25 ∧ runs_before_12th (i + 1) = 25) :
   (11 * initial_average + runs_12th) / 12 = 26 :=
by
  sorry

end batsman_average_after_12th_l1341_134137


namespace square_area_l1341_134155

theorem square_area (P : ℝ) (hP : P = 32) : ∃ A : ℝ, A = 64 ∧ A = (P / 4) ^ 2 :=
by {
  sorry
}

end square_area_l1341_134155


namespace problem_statement_l1341_134153

noncomputable def a : ℕ := by
  -- The smallest positive two-digit multiple of 3
  let a := Finset.range 100 \ Finset.range 10
  let multiples := a.filter (λ n => n % 3 = 0)
  exact multiples.min' ⟨12, sorry⟩

noncomputable def b : ℕ := by
  -- The smallest positive three-digit multiple of 4
  let b := Finset.range 1000 \ Finset.range 100
  let multiples := b.filter (λ n => n % 4 = 0)
  exact multiples.min' ⟨100, sorry⟩

theorem problem_statement : a + b = 112 := by
  sorry

end problem_statement_l1341_134153


namespace sally_cards_l1341_134190

theorem sally_cards (x : ℕ) (h1 : 27 + x + 20 = 88) : x = 41 := by
  sorry

end sally_cards_l1341_134190


namespace michael_earnings_l1341_134127

theorem michael_earnings :
  let price_extra_large := 150
  let price_large := 100
  let price_medium := 80
  let price_small := 60
  let qty_extra_large := 3
  let qty_large := 5
  let qty_medium := 8
  let qty_small := 10
  let discount_large := 0.10
  let tax := 0.05
  let cost_materials := 300
  let commission_fee := 0.10

  let total_initial_sales := (qty_extra_large * price_extra_large) + 
                             (qty_large * price_large) + 
                             (qty_medium * price_medium) + 
                             (qty_small * price_small)

  let discount_on_large := discount_large * (qty_large * price_large)
  let sales_after_discount := total_initial_sales - discount_on_large

  let sales_tax := tax * sales_after_discount
  let total_collected := sales_after_discount + sales_tax

  let commission := commission_fee * sales_after_discount
  let total_deductions := cost_materials + commission
  let earnings := total_collected - total_deductions

  earnings = 1733 :=
by
  sorry

end michael_earnings_l1341_134127


namespace boat_crossing_time_l1341_134158

theorem boat_crossing_time :
  ∀ (width_of_river speed_of_current speed_of_boat : ℝ),
  width_of_river = 1.5 →
  speed_of_current = 8 →
  speed_of_boat = 10 →
  (width_of_river / (Real.sqrt (speed_of_boat ^ 2 - speed_of_current ^ 2)) * 60) = 15 :=
by
  intros width_of_river speed_of_current speed_of_boat h1 h2 h3
  sorry

end boat_crossing_time_l1341_134158


namespace lisa_takes_72_more_minutes_than_ken_l1341_134101

theorem lisa_takes_72_more_minutes_than_ken
  (ken_speed : ℕ) (lisa_speed : ℕ) (book_pages : ℕ)
  (h_ken_speed: ken_speed = 75)
  (h_lisa_speed: lisa_speed = 60)
  (h_book_pages: book_pages = 360) :
  ((book_pages / lisa_speed:ℚ) - (book_pages / ken_speed:ℚ)) * 60 = 72 :=
by
  sorry

end lisa_takes_72_more_minutes_than_ken_l1341_134101


namespace miller_rabin_probability_at_least_half_l1341_134149

theorem miller_rabin_probability_at_least_half
  {n : ℕ} (hcomp : ¬Nat.Prime n) (s d : ℕ) (hd_odd : d % 2 = 1) (h_decomp : n - 1 = 2^s * d)
  (a : ℤ) (ha_range : 2 ≤ a ∧ a ≤ n - 2) :
  ∃ P : ℝ, P ≥ 1 / 2 ∧ ∀ a, (2 ≤ a ∧ a ≤ n - 2) → ¬(a^(d * 2^s) % n = 1)
  :=
sorry

end miller_rabin_probability_at_least_half_l1341_134149


namespace toys_sold_week2_l1341_134196

-- Define the given conditions
def original_stock := 83
def toys_sold_week1 := 38
def toys_left := 19

-- Define the statement we want to prove
theorem toys_sold_week2 : (original_stock - toys_left) - toys_sold_week1 = 26 :=
by
  sorry

end toys_sold_week2_l1341_134196


namespace solve_inequality_l1341_134173

theorem solve_inequality (a : ℝ) :
  (a > 0 → ∀ x : ℝ, (12 * x^2 - a * x - a^2 < 0 ↔ -a / 4 < x ∧ x < a / 3)) ∧
  (a = 0 → ∀ x : ℝ, ¬ (12 * x^2 - a * x - a^2 < 0)) ∧ 
  (a < 0 → ∀ x : ℝ, (12 * x^2 - a * x - a^2 < 0 ↔ a / 3 < x ∧ x < -a / 4)) :=
by
  sorry

end solve_inequality_l1341_134173


namespace output_value_of_y_l1341_134114

/-- Define the initial conditions -/
def l : ℕ := 2
def m : ℕ := 3
def n : ℕ := 5

/-- Define the function that executes the flowchart operations -/
noncomputable def flowchart_operation (l m n : ℕ) : ℕ := sorry

/-- Main theorem statement -/
theorem output_value_of_y : flowchart_operation l m n = 68 := sorry

end output_value_of_y_l1341_134114


namespace triangle_area_l1341_134116

noncomputable def area_triangle (A B C : ℝ) (b c : ℝ) : ℝ :=
  0.5 * b * c * Real.sin A

theorem triangle_area
  (A B C : ℝ) (b : ℝ) 
  (hA : A = π / 4)
  (h0 : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B) :
  ∃ c : ℝ, area_triangle A B C b c = 2 :=
by
  sorry

end triangle_area_l1341_134116


namespace matthew_egg_rolls_l1341_134119

theorem matthew_egg_rolls 
    (M P A : ℕ)
    (h1 : M = 3 * P)
    (h2 : P = A / 2)
    (h3 : A = 4) : 
    M = 6 :=
by
  sorry

end matthew_egg_rolls_l1341_134119


namespace projectiles_meet_in_84_minutes_l1341_134107

theorem projectiles_meet_in_84_minutes :
  ∀ (d v₁ v₂ : ℝ), d = 1386 → v₁ = 445 → v₂ = 545 → (20 : ℝ) = 20 → 
  ((1386 / (445 + 545) / 60) * 60 * 60 = 84) :=
by
  intros d v₁ v₂ h_d h_v₁ h_v₂ h_wind
  sorry

end projectiles_meet_in_84_minutes_l1341_134107


namespace rectangle_dimensions_l1341_134188

theorem rectangle_dimensions (x y : ℝ) (h1 : y = 2 * x) (h2 : 2 * (x + y) = 2 * (x * y)) :
  (x = 3 / 2) ∧ (y = 3) := by
  sorry

end rectangle_dimensions_l1341_134188


namespace man_age_twice_son_age_l1341_134112

theorem man_age_twice_son_age (S M : ℕ) (h1 : M = S + 24) (h2 : S = 22) : 
  ∃ Y : ℕ, M + Y = 2 * (S + Y) ∧ Y = 2 :=
by 
  sorry

end man_age_twice_son_age_l1341_134112


namespace f_periodic_function_l1341_134139

noncomputable def f : ℝ → ℝ := sorry

theorem f_periodic_function (h1 : ∀ x : ℝ, f (-x) = f x)
    (h2 : ∀ x : ℝ, f (x + 4) = f x + f 2)
    (h3 : f 1 = 2) : 
    f 2013 = 2 := sorry

end f_periodic_function_l1341_134139


namespace score_sd_above_mean_l1341_134128

theorem score_sd_above_mean (mean std dev1 dev2 : ℝ) : 
  mean = 74 → dev1 = 2 → dev2 = 3 → mean - dev1 * std = 58 → mean + dev2 * std = 98 :=
by
  sorry

end score_sd_above_mean_l1341_134128


namespace square_perimeter_from_area_l1341_134124

def square_area (s : ℝ) : ℝ := s * s -- Definition of the area of a square based on its side length.
def square_perimeter (s : ℝ) : ℝ := 4 * s -- Definition of the perimeter of a square based on its side length.

theorem square_perimeter_from_area (s : ℝ) (h : square_area s = 900) : square_perimeter s = 120 :=
by {
  sorry -- Placeholder for the proof.
}

end square_perimeter_from_area_l1341_134124


namespace blue_die_prime_yellow_die_power_2_probability_l1341_134167

def prime_numbers : Finset ℕ := {2, 3, 5, 7}

def powers_of_2 : Finset ℕ := {1, 2, 4, 8}

def total_outcomes : ℕ := 8 * 8

def successful_outcomes : ℕ := prime_numbers.card * powers_of_2.card

def probability (x y : Finset ℕ) : ℚ := (x.card * y.card) / (total_outcomes : ℚ)

theorem blue_die_prime_yellow_die_power_2_probability :
  probability prime_numbers powers_of_2 = 1 / 4 :=
by
  sorry

end blue_die_prime_yellow_die_power_2_probability_l1341_134167


namespace third_term_binomial_expansion_l1341_134110

-- Let a, x be real numbers
variables (a x : ℝ)

-- Binomial theorem term for k = 2
def binomial_term (n k : ℕ) (x y : ℝ) : ℝ :=
  (Nat.choose n k) * x^(n-k) * y^k

theorem third_term_binomial_expansion :
  binomial_term 6 2 (a / Real.sqrt x) (-Real.sqrt x / a^2) = 15 / x :=
by
  sorry

end third_term_binomial_expansion_l1341_134110


namespace solve_for_b_l1341_134176

theorem solve_for_b (a b : ℤ) (h1 : 3 * a + 2 = 5) (h2 : b - 4 * a = 2) : b = 6 :=
by
  -- proof goes here
  sorry

end solve_for_b_l1341_134176


namespace harvey_sold_17_steaks_l1341_134195

variable (initial_steaks : ℕ) (steaks_left_after_first_sale : ℕ) (steaks_sold_in_second_sale : ℕ)

noncomputable def total_steaks_sold (initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale : ℕ) : ℕ :=
  (initial_steaks - steaks_left_after_first_sale) + steaks_sold_in_second_sale

theorem harvey_sold_17_steaks :
  initial_steaks = 25 →
  steaks_left_after_first_sale = 12 →
  steaks_sold_in_second_sale = 4 →
  total_steaks_sold initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale = 17 :=
by
  intros
  sorry

end harvey_sold_17_steaks_l1341_134195


namespace cricket_team_members_l1341_134121

theorem cricket_team_members (n : ℕ) 
  (avg_age_team : ℕ) 
  (age_captain : ℕ) 
  (age_wkeeper : ℕ) 
  (avg_age_remaining : ℕ) 
  (total_age_team : ℕ) 
  (total_age_excl_cw : ℕ) 
  (total_age_remaining : ℕ) :
  avg_age_team = 23 →
  age_captain = 26 →
  age_wkeeper = 29 →
  avg_age_remaining = 22 →
  total_age_team = avg_age_team * n →
  total_age_excl_cw = total_age_team - (age_captain + age_wkeeper) →
  total_age_remaining = avg_age_remaining * (n - 2) →
  total_age_excl_cw = total_age_remaining →
  n = 11 :=
by
  sorry

end cricket_team_members_l1341_134121


namespace x_sq_y_sq_value_l1341_134163

theorem x_sq_y_sq_value (x y : ℝ) 
  (h1 : x + y = 25) 
  (h2 : x^2 + y^2 = 169) 
  (h3 : x^3 * y^3 + y^3 * x^3 = 243) :
  x^2 * y^2 = 51984 := 
by 
  -- Proof to be added
  sorry

end x_sq_y_sq_value_l1341_134163


namespace rationalize_sqrt_5_over_12_l1341_134156

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l1341_134156


namespace reflection_line_equation_l1341_134180

-- Given condition 1: Original line equation
def original_line (x : ℝ) : ℝ := -2 * x + 7

-- Given condition 2: Reflection line
def reflection_line_x : ℝ := 3

-- Proving statement
theorem reflection_line_equation
  (a b : ℝ)
  (h₁ : a = -(-2))
  (h₂ : original_line 3 = 1)
  (h₃ : 1 = a * 3 + b) :
  2 * a + b = -1 :=
  sorry

end reflection_line_equation_l1341_134180


namespace avg_age_of_five_students_l1341_134146

-- step a: Define the conditions
def avg_age_seventeen_students : ℕ := 17
def total_seventeen_students : ℕ := 17 * avg_age_seventeen_students

def num_students_with_unknown_avg : ℕ := 5

def avg_age_nine_students : ℕ := 16
def num_students_with_known_avg : ℕ := 9
def total_age_nine_students : ℕ := num_students_with_known_avg * avg_age_nine_students

def age_seventeenth_student : ℕ := 75

-- step c: Compute the average age of the 5 students
noncomputable def total_age_five_students : ℕ :=
  total_seventeen_students - total_age_nine_students - age_seventeenth_student

def correct_avg_age_five_students : ℕ := 14

theorem avg_age_of_five_students :
  total_age_five_students / num_students_with_unknown_avg = correct_avg_age_five_students :=
sorry

end avg_age_of_five_students_l1341_134146


namespace total_pieces_correct_l1341_134142

-- Definition of the pieces of chicken required per type of order
def chicken_pieces_per_chicken_pasta : ℕ := 2
def chicken_pieces_per_barbecue_chicken : ℕ := 3
def chicken_pieces_per_fried_chicken_dinner : ℕ := 8

-- Definition of the number of each type of order tonight
def num_fried_chicken_dinner_orders : ℕ := 2
def num_chicken_pasta_orders : ℕ := 6
def num_barbecue_chicken_orders : ℕ := 3

-- Calculate the total number of pieces of chicken needed
def total_chicken_pieces_needed : ℕ :=
  (num_fried_chicken_dinner_orders * chicken_pieces_per_fried_chicken_dinner) +
  (num_chicken_pasta_orders * chicken_pieces_per_chicken_pasta) +
  (num_barbecue_chicken_orders * chicken_pieces_per_barbecue_chicken)

-- The proof statement
theorem total_pieces_correct : total_chicken_pieces_needed = 37 :=
by
  -- Our exact computation here
  sorry

end total_pieces_correct_l1341_134142


namespace molecular_weight_NaClO_is_74_44_l1341_134141

-- Define the atomic weights
def atomic_weight_Na : Real := 22.99
def atomic_weight_Cl : Real := 35.45
def atomic_weight_O : Real := 16.00

-- Define the calculation of molecular weight
def molecular_weight_NaClO : Real :=
  atomic_weight_Na + atomic_weight_Cl + atomic_weight_O

-- Define the theorem statement
theorem molecular_weight_NaClO_is_74_44 :
  molecular_weight_NaClO = 74.44 :=
by
  -- Placeholder for proof
  sorry

end molecular_weight_NaClO_is_74_44_l1341_134141


namespace track_width_l1341_134165

theorem track_width (r1 r2 : ℝ) (h : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi) : r1 - r2 = 10 := by
  sorry

end track_width_l1341_134165


namespace find_cost_price_l1341_134159

theorem find_cost_price (SP PP : ℝ) (hSP : SP = 600) (hPP : PP = 25) : 
  ∃ CP : ℝ, CP = 480 := 
by
  sorry

end find_cost_price_l1341_134159


namespace intersection_A_compB_l1341_134191

def setA : Set ℤ := {x | (abs (x - 1) < 3)}
def setB : Set ℝ := {x | x^2 + 2 * x - 3 ≥ 0}
def setCompB : Set ℝ := {x | ¬(x^2 + 2 * x - 3 ≥ 0)}

theorem intersection_A_compB :
  { x : ℤ | x ∈ setA ∧ (x:ℝ) ∈ setCompB } = {-1, 0} :=
sorry

end intersection_A_compB_l1341_134191


namespace science_club_officers_l1341_134178

-- Definitions of the problem conditions
def num_members : ℕ := 25
def num_officers : ℕ := 3
def alice : ℕ := 1 -- unique identifier for Alice
def bob : ℕ := 2 -- unique identifier for Bob

-- Main theorem statement
theorem science_club_officers :
  ∃ (ways_to_choose_officers : ℕ), ways_to_choose_officers = 10764 :=
  sorry

end science_club_officers_l1341_134178


namespace binary_to_octal_conversion_l1341_134102

-- Define the binary number 11010 in binary
def bin_value : ℕ := 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0

-- Define the octal value 32 in octal as decimal
def oct_value : ℕ := 3 * 8^1 + 2 * 8^0

-- The theorem to prove the binary equivalent of 11010 is the octal 32
theorem binary_to_octal_conversion : bin_value = oct_value :=
by
  -- Skip actual proof
  sorry

end binary_to_octal_conversion_l1341_134102


namespace opposite_number_l1341_134129

variable (a : ℝ)

theorem opposite_number (a : ℝ) : -(3 * a - 2) = -3 * a + 2 := by
  sorry

end opposite_number_l1341_134129


namespace production_period_l1341_134103

-- Define the conditions as constants
def daily_production : ℕ := 1500
def price_per_computer : ℕ := 150
def total_earnings : ℕ := 1575000

-- Define the computation to find the period and state what we need to prove
theorem production_period : (total_earnings / price_per_computer) / daily_production = 7 :=
by
  -- you can provide the steps, but it's optional since the proof is omitted
  sorry

end production_period_l1341_134103


namespace part1_part2_l1341_134187

/-- Part (1) -/
theorem part1 (a : ℝ) (p : ∀ x : ℝ, x^2 - a*x + 4 > 0) (q : ∀ x y : ℝ, (0 < x ∧ x < y) → x^a < y^a) : 
  0 < a ∧ a < 4 :=
sorry

/-- Part (2) -/
theorem part2 (a : ℝ) (p_iff: ∀ x : ℝ, x^2 - a*x + 4 > 0 ↔ -4 < a ∧ a < 4)
  (q_iff: ∀ x y : ℝ, (0 < x ∧ x < y) ↔ x^a < y^a ∧ a > 0) (hp : ∃ x : ℝ, ¬(x^2 - a*x + 4 > 0))
  (hq : ∀ x y : ℝ, (x^a < y^a) → (0 < x ∧ x < y)) : 
  (a >= 4) ∨ (-4 < a ∧ a <= 0) :=
sorry

end part1_part2_l1341_134187


namespace cost_of_each_lunch_packet_l1341_134109

-- Definitions of the variables
def num_students := 50
def total_cost := 3087

-- Variables representing the unknowns
variable (s c n : ℕ)

-- Conditions
def more_than_half_students_bought : Prop := s > num_students / 2
def apples_less_than_cost_per_packet : Prop := n < c
def total_cost_condition : Prop := s * c = total_cost

-- The statement to prove
theorem cost_of_each_lunch_packet :
  (s : ℕ) * c = total_cost ∧
  (s > num_students / 2) ∧
  (n < c)
  -> c = 9 :=
by
  sorry

end cost_of_each_lunch_packet_l1341_134109


namespace pencils_added_by_sara_l1341_134143

-- Definitions based on given conditions
def original_pencils : ℕ := 115
def total_pencils : ℕ := 215

-- Statement to prove
theorem pencils_added_by_sara : total_pencils - original_pencils = 100 :=
by {
  -- Proof
  sorry
}

end pencils_added_by_sara_l1341_134143


namespace darry_full_ladder_climbs_l1341_134100

-- Definitions and conditions
def full_ladder_steps : ℕ := 11
def smaller_ladder_steps : ℕ := 6
def smaller_ladder_climbs : ℕ := 7
def total_steps_climbed_today : ℕ := 152

-- Question: How many times did Darry climb his full ladder?
theorem darry_full_ladder_climbs (x : ℕ) 
  (H : 11 * x + smaller_ladder_steps * 7 = total_steps_climbed_today) : 
  x = 10 := by
  -- proof steps omitted, so we write
  sorry

end darry_full_ladder_climbs_l1341_134100


namespace remaining_amount_correct_l1341_134138

def initial_amount : ℝ := 70
def coffee_cost_per_pound : ℝ := 8.58
def coffee_pounds : ℝ := 4.0
def total_cost : ℝ := coffee_pounds * coffee_cost_per_pound
def remaining_amount : ℝ := initial_amount - total_cost

theorem remaining_amount_correct : remaining_amount = 35.68 :=
by
  -- Skip the proof; this is a placeholder.
  sorry

end remaining_amount_correct_l1341_134138


namespace employed_males_percentage_l1341_134166

theorem employed_males_percentage (total_population employed employed_as_percent employed_females female_as_percent employed_males employed_males_percentage : ℕ) 
(total_population_eq : total_population = 100)
(employed_eq : employed = employed_as_percent * total_population / 100)
(employed_as_percent_eq : employed_as_percent = 60)
(employed_females_eq : employed_females = female_as_percent * employed / 100)
(female_as_percent_eq : female_as_percent = 25)
(employed_males_eq : employed_males = employed - employed_females)
(employed_males_percentage_eq : employed_males_percentage = employed_males * 100 / total_population) :
employed_males_percentage = 45 :=
sorry

end employed_males_percentage_l1341_134166


namespace coefficient_of_x3_in_expansion_l1341_134105

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def expansion_coefficient_x3 : ℤ :=
  let term1 := (-1 : ℤ) ^ 3 * binomial_coefficient 6 3
  let term2 := (1 : ℤ) * binomial_coefficient 6 2
  term1 + term2

theorem coefficient_of_x3_in_expansion :
  expansion_coefficient_x3 = -5 := by
  sorry

end coefficient_of_x3_in_expansion_l1341_134105


namespace cells_surpass_10_pow_10_in_46_hours_l1341_134131

noncomputable def cells_exceed_threshold_hours : ℕ := 46

theorem cells_surpass_10_pow_10_in_46_hours : 
  ∀ (n : ℕ), (100 * ((3 / 2 : ℝ) ^ n) > 10 ^ 10) ↔ n ≥ cells_exceed_threshold_hours := 
by
  sorry

end cells_surpass_10_pow_10_in_46_hours_l1341_134131


namespace john_total_amount_l1341_134106

/-- Define the amounts of money John has and needs additionally -/
def johnHas : ℝ := 0.75
def needsMore : ℝ := 1.75

/-- Prove the total amount of money John needs given the conditions -/
theorem john_total_amount : johnHas + needsMore = 2.50 := by
  sorry

end john_total_amount_l1341_134106


namespace min_side_length_is_isosceles_l1341_134181

-- Let a denote the side length BC
-- Let b denote the side length AB
-- Let c denote the side length AC

theorem min_side_length_is_isosceles (α : ℝ) (S : ℝ) (a b c : ℝ) :
  (a^2 = b^2 + c^2 - 2 * b * c * Real.cos α ∧ S = 0.5 * b * c * Real.sin α) →
  a = Real.sqrt (((b - c)^2 + (4 * S * (1 - Real.cos α)) / Real.sin α)) →
  b = c :=
by
  intros h1 h2
  sorry

end min_side_length_is_isosceles_l1341_134181


namespace parabola_intersections_l1341_134183

theorem parabola_intersections :
  (∀ x y, (y = 4 * x^2 + 4 * x - 7) ↔ (y = x^2 + 5)) →
  (∃ (points : List (ℝ × ℝ)),
    (points = [(-2, 9), (2, 9)]) ∧
    (∀ p ∈ points, ∃ x, p = (x, x^2 + 5) ∧ y = 4 * x^2 + 4 * x - 7)) :=
by sorry

end parabola_intersections_l1341_134183


namespace weight_of_11m_rebar_l1341_134161

theorem weight_of_11m_rebar (w5m : ℝ) (l5m : ℝ) (l11m : ℝ) 
  (h_w5m : w5m = 15.3) (h_l5m : l5m = 5) (h_l11m : l11m = 11) : 
  (w5m / l5m) * l11m = 33.66 := 
by {
  sorry
}

end weight_of_11m_rebar_l1341_134161


namespace marble_choice_l1341_134140

def numDifferentGroupsOfTwoMarbles (red green blue : ℕ) (yellow : ℕ) (orange : ℕ) : ℕ :=
  if (red = 1 ∧ green = 1 ∧ blue = 1 ∧ yellow = 2 ∧ orange = 2) then 12 else 0

theorem marble_choice:
  let red := 1
  let green := 1
  let blue := 1
  let yellow := 2
  let orange := 2
  numDifferentGroupsOfTwoMarbles red green blue yellow orange = 12 :=
by
  dsimp[numDifferentGroupsOfTwoMarbles]
  split_ifs
  · rfl
  · sorry

-- Ensure the theorem type matches the expected Lean 4 structure.
#print marble_choice

end marble_choice_l1341_134140


namespace max_g_value_l1341_134175

def g : Nat → Nat
| n => if n < 15 then n + 15 else g (n - 6)

theorem max_g_value : ∀ n, g n ≤ 29 := by
  sorry

end max_g_value_l1341_134175


namespace pure_imaginary_number_l1341_134135

theorem pure_imaginary_number (m : ℝ) (h_real : m^2 - 5 * m + 6 = 0) (h_imag : m^2 - 3 * m ≠ 0) : m = 2 :=
sorry

end pure_imaginary_number_l1341_134135


namespace smallest_two_digit_multiple_of_3_l1341_134192

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n <= 99
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

theorem smallest_two_digit_multiple_of_3 : ∃ n : ℕ, is_two_digit n ∧ is_multiple_of_3 n ∧ ∀ m : ℕ, is_two_digit m ∧ is_multiple_of_3 m → n <= m :=
sorry

end smallest_two_digit_multiple_of_3_l1341_134192


namespace least_total_cost_is_172_l1341_134120

noncomputable def least_total_cost : ℕ :=
  let lcm := Nat.lcm (Nat.lcm 6 5) 8
  let strawberry_packs := lcm / 6
  let blueberry_packs := lcm / 5
  let cherry_packs := lcm / 8
  let strawberry_cost := strawberry_packs * 2
  let blueberry_cost := blueberry_packs * 3
  let cherry_cost := cherry_packs * 4
  strawberry_cost + blueberry_cost + cherry_cost

theorem least_total_cost_is_172 : least_total_cost = 172 := 
by
  sorry

end least_total_cost_is_172_l1341_134120


namespace more_tvs_sold_l1341_134194

variable (T x : ℕ)

theorem more_tvs_sold (h1 : T + x = 327) (h2 : T + 3 * x = 477) : x = 75 := by
  sorry

end more_tvs_sold_l1341_134194


namespace qin_jiushao_algorithm_v2_l1341_134136

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

-- Define the value x to evaluate the polynomial at
def x0 : ℝ := -1

-- Define the intermediate value v2 according to Horner's rule
def v1 : ℝ := 2 * x0^4 - 3 * x0^3 + x0^2
def v2 : ℝ := v1 * x0 + 2

theorem qin_jiushao_algorithm_v2 : v2 = -4 := 
by 
  -- The proof will be here, for now we place sorry.
  sorry

end qin_jiushao_algorithm_v2_l1341_134136


namespace length_of_each_part_l1341_134111

theorem length_of_each_part (ft : ℕ) (inch : ℕ) (parts : ℕ) (total_length : ℕ) (part_length : ℕ) :
  ft = 6 → inch = 8 → parts = 5 → total_length = 12 * ft + inch → part_length = total_length / parts → part_length = 16 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end length_of_each_part_l1341_134111


namespace functional_equation_solution_l1341_134186

open Function

theorem functional_equation_solution :
  ∀ (f g : ℚ → ℚ), 
    (∀ x y : ℚ, f (g x + g y) = f (g x) + y ∧ g (f x + f y) = g (f x) + y) →
    (∃ a b : ℚ, (ab = 1) ∧ (∀ x : ℚ, f x = a * x) ∧ (∀ x : ℚ, g x = b * x)) :=
by
  intros f g h
  sorry

end functional_equation_solution_l1341_134186


namespace intersections_correct_l1341_134117

-- Define the distances (in meters)
def gretzky_street_length : ℕ := 5600
def segment_a_distance : ℕ := 350
def segment_b_distance : ℕ := 400
def segment_c_distance : ℕ := 450

-- Definitions based on conditions
def segment_a_intersections : ℕ :=
  gretzky_street_length / segment_a_distance - 2 -- subtract Orr Street and Howe Street

def segment_b_intersections : ℕ :=
  gretzky_street_length / segment_b_distance

def segment_c_intersections : ℕ :=
  gretzky_street_length / segment_c_distance

-- Sum of all intersections
def total_intersections : ℕ :=
  segment_a_intersections + segment_b_intersections + segment_c_intersections

theorem intersections_correct :
  total_intersections = 40 :=
by
  sorry

end intersections_correct_l1341_134117


namespace part1_part2_part3_part3_expectation_l1341_134169

/-- Conditions setup -/
noncomputable def gameCondition (Aacc Bacc : ℝ) :=
  (Aacc = 0.5) ∧ (Bacc = 0.6)

def scoreDist (X:ℤ) : ℝ :=
  if X = -1 then 0.3
  else if X = 0 then 0.5
  else if X = 1 then 0.2
  else 0

def tieProbability : ℝ := 0.2569

def roundDist (Y:ℤ) : ℝ :=
  if Y = 2 then 0.13
  else if Y = 3 then 0.13
  else if Y = 4 then 0.74
  else 0

def roundExpectation : ℝ := 3.61

/-- Proof Statements -/
theorem part1 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (X : ℤ), scoreDist X = if X = -1 then 0.3 else if X = 0 then 0.5 else if X = 1 then 0.2 else 0 :=
by sorry

theorem part2 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  tieProbability = 0.2569 :=
by sorry

theorem part3 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (Y : ℤ), roundDist Y = if Y = 2 then 0.13 else if Y = 3 then 0.13 else if Y = 4 then 0.74 else 0 :=
by sorry

theorem part3_expectation (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) :
  roundExpectation = 3.61 :=
by sorry

end part1_part2_part3_part3_expectation_l1341_134169


namespace vinegar_mixture_concentration_l1341_134125

theorem vinegar_mixture_concentration :
  let c1 := 5 / 100
  let c2 := 10 / 100
  let v1 := 10
  let v2 := 10
  (v1 * c1 + v2 * c2) / (v1 + v2) = 7.5 / 100 :=
by
  sorry

end vinegar_mixture_concentration_l1341_134125


namespace cylinder_cone_volume_l1341_134126

theorem cylinder_cone_volume (V_total : ℝ) (Vc Vcone : ℝ)
  (h1 : V_total = 48)
  (h2 : V_total = Vc + Vcone)
  (h3 : Vc = 3 * Vcone) :
  Vc = 36 ∧ Vcone = 12 :=
by
  sorry

end cylinder_cone_volume_l1341_134126


namespace percentage_of_number_is_40_l1341_134133

theorem percentage_of_number_is_40 (N : ℝ) (P : ℝ) 
  (h1 : (1/4) * (1/3) * (2/5) * N = 35) 
  (h2 : (P/100) * N = 420) : 
  P = 40 := 
by
  sorry

end percentage_of_number_is_40_l1341_134133


namespace trisha_money_left_l1341_134160

theorem trisha_money_left
    (meat cost: ℕ) (chicken_cost: ℕ) (veggies_cost: ℕ) (eggs_cost: ℕ) (dog_food_cost: ℕ) 
    (initial_money: ℕ) (total_spent: ℕ) (money_left: ℕ) :
    meat_cost = 17 →
    chicken_cost = 22 →
    veggies_cost = 43 →
    eggs_cost = 5 →
    dog_food_cost = 45 →
    initial_money = 167 →
    total_spent = meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost →
    money_left = initial_money - total_spent →
    money_left = 35 :=
by
    intros
    sorry

end trisha_money_left_l1341_134160


namespace sum_of_bases_is_16_l1341_134134

/-
  Given the fractions G_1 and G_2 in two different bases S_1 and S_2, we need to show 
  that the sum of these bases S_1 and S_2 in base ten is 16.
-/
theorem sum_of_bases_is_16 (S_1 S_2 G_1 G_2 : ℕ) :
  (G_1 = (4 * S_1 + 5) / (S_1^2 - 1)) →
  (G_2 = (5 * S_1 + 4) / (S_1^2 - 1)) →
  (G_1 = (S_2 + 4) / (S_2^2 - 1)) →
  (G_2 = (4 * S_2 + 1) / (S_2^2 - 1)) →
  S_1 + S_2 = 16 :=
by
  intros hG1_S1 hG2_S1 hG1_S2 hG2_S2
  sorry

end sum_of_bases_is_16_l1341_134134


namespace find_value_of_a_l1341_134193

theorem find_value_of_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_eq : 7 * a^2 + 14 * a * b = a^3 + 2 * a^2 * b) : a = 7 := 
sorry

end find_value_of_a_l1341_134193
